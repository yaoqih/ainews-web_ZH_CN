---
companies:
- meta-ai-fair
- groq
- nvidia
- amazon
- microsoft
date: '2024-04-20T02:21:27.301127Z'
description: '**Meta** 发布了 **Llama 3**，这是其功能最强大的开源大语言模型，包含 **8B（80亿）和 70B（700亿）参数版本**，支持
  **8K 上下文长度**，其性能超越了包括 **Llama 2** 和 **Mistral 7B** 在内的先前模型。**Groq** 以 **每秒 500-800
  个 token** 的速度运行 **Llama 3 70B** 模型，使其成为目前最快的 GPT-4 级别 token 来源。


  相关讨论强调了 AI 规模化（scaling）面临的挑战：**埃隆·马斯克 (Elon Musk)** 表示训练 **Grok 3** 将需要 **10 万块英伟达
  H100 GPU**，而 **AWS** 计划为一款 **27 万亿参数的模型** 采购 **2 万块 B200 GPU**。微软推出了用于生成逼真说话人脸的
  **VASA-1**，而 **Stable Diffusion 3** 及其扩展版本的反响褒贬不一。此外，关于 AI 能源消耗和政治偏见的担忧也引起了讨论。'
id: c75b60c6-4f06-4b9c-99cb-3f9939a75ef2
models:
- llama-3-70b
- llama-3-8b
- llama-3
- llama-2-70b
- mistral-7b
- grok-3
- stable-diffusion-3
- vasa-1
original_slug: ainews-llama-3
people:
- elon-musk
title: Llama-3-70b 是 GPT-4 级别的开源模型。
topics:
- benchmarking
- model-performance
- fine-tuning
- function-calling
- arithmetic
- image-generation
- video-generation
- energy-usage
- gpu-demand
- political-bias
- ai-safety
- scaling
- context-windows
- tokenization
---

<!-- buttondown-editor-mode: plaintext -->> 2024/4/18-4/19 的 AI 新闻。我们为你检查了 6 个 subreddits、[**364** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 以及 **27** 个 Discord 服务（包含 **395** 个频道和 **10403** 条消息）。预计节省阅读时间（按 200wpm 计算）：**958 分钟**。

在 1600 票的样本量下，Lmsys 的早期结果甚至比报告的 Benchmark 还要好，这在当下非常罕见：

 
![image.png](https://assets.buttondown.email/images/ebd4a52b-05e1-4b91-a842-0432853455fa.png?w=960&fit=max)
 

这是第一个击败 Opus 的开源模型，而 Opus 本身是第一个曾短暂击败 GPT4 Turbo 的模型。当然，随着时间的推移，这可能会有所波动，但对于即将发布的 Llama-3-400b 来说，这是一个非常好的预兆。

目前 [Groq 已经能以 500-800 tok/s 的速度运行 70b 模型](https://twitter.com/mattshumer_/status/1781355430914015482)，这使得 Llama 3 毫无疑问成为了目前最快的 GPT-4 级别 Token 来源。

随着最近关于 [Chinchilla 的复现结果](https://twitter.com/tamaybes/status/1780639257389904013)受到质疑（不要错过 [Susan Zhang 的精彩推文](https://twitter.com/suchenzang/status/1616752482226671620?utm_source=ainews&utm_medium=email&utm_campaign=ainews-to-be-named-5820)，该推文得到了 [Chinchilla 共同作者](https://twitter.com/borgeaud_s/status/1780988694163321250?utm_source=ainews&utm_medium=email&utm_campaign=ainews-to-be-named-5820)的认可），Llama 2 和 3（以及在开源程度上稍逊一筹的 Mistral）已经相当彻底地将 Chinchilla 定律送进了历史的尘埃。

---

**目录**

[TOC] 


---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**Meta 的 Llama 3 发布及其能力**

- **Llama 3 作为最强大的开源 LLM 发布**：Meta 发布了 [Llama 3，这是他们迄今为止功能最强大的开源大语言模型](https://ai.meta.com/blog/meta-llama-3/)。在 /r/LocalLLaMA 中，有人指出 [**8B 和 70B 参数版本已发布，支持 8K 上下文长度**](https://www.reddit.com/r/LocalLLaMA/comments/1c7kd9l/llama_3_postrelease_megathread_discussion_and/)。还分享了一个针对 [70B 模型的开源 code interpreter](https://github.com/e2b-dev/e2b-cookbook/blob/main/examples/llama-3-code-interpreter/llama_3_code_interpreter.ipynb)。

- **Llama 3 在基准测试中表现优于之前的模型**：/r/LocalLLaMA 中分享的基准测试显示，[**Llama 3 8B instruct 在各项任务中表现优于之前的 Llama 2 70B instruct 模型**](https://www.reddit.com/r/LocalLLaMA/comments/1c7kd9l/llama_3_postrelease_megathread_discussion_and/)。[根据 API 定价，70B 模型以超过 20 倍的低成本提供了 GPT-4 级别的性能](https://www.reddit.com/r/LocalLLaMA/comments/1c7jybg/llama370b_over_20x_cheaper_than_gpt4/)。测试还显示 [Llama 3 7B 在 function calling 和算术方面超过了 Mistral 7B](https://www.reddit.com/r/LocalLLaMA/comments/1c7o27l/real_world_test_llama3_7b_blew_mistral_7b_out_of/)。

**图像/视频 AI 进展与 Stable Diffusion 3**

- **逼真的说话人脸生成和令人印象深刻的视频 AI**：微软展示了 [VASA-1，用于从音频生成逼真的说话人脸](https://streamable.com/gzl8kr)。Meta 的 [图像和视频生成 UI 在 /r/singularity 中被评价为“令人惊叹”](https://www.reddit.com/r/singularity/comments/1c7hcvp/meta_ai_image_and_video_generation_ui_is/)。

- **Stable Diffusion 3 的印象与扩展**：在 /r/StableDiffusion 中，有人指出 [与其他服务相比，Imagine.art 给人留下了关于 SD3 能力的错误印象](https://www.reddit.com/r/StableDiffusion/comments/1c7p340/imagineart_gave_false_impression_of_sd3/)。还分享了一个 [Forge Couple 扩展，为 SD 增加了可拖动的主体区域](https://www.reddit.com/r/StableDiffusion/comments/1c7lpd0/forge_couple_draggable_regions/)。

**AI 扩展挑战与算力需求**

- **AI 能源消耗和 GPU 需求快速增长**：/r/singularity 的讨论强调，[到 2030 年，AI 的算力需求可能会压垮能源供应](https://www.reddit.com/r/singularity/comments/1c7282g/ais_voracious_need_for_computing_power_is/)。Elon Musk 表示 [训练 Grok 3 将需要 100,000 块 Nvidia H100 GPU](https://www.tomshardware.com/tech-industry/artificial-intelligence/elon-musk-says-the-next-generation-grok-3-model-will-require-100000-nvidia-h100-gpus-to-train)，同时 [AWS 计划采购 20,000 块 B200 GPU 用于一个 27 万亿参数的模型](https://www.cnbc.com/2024/03/18/nvidia-announces-gb200-blackwell-ai-chip-launching-later-this-year.html)。

**AI 安全、偏见与社会影响讨论**

- **政治偏见与 AI 安全担忧**：在 /r/singularity 中，有人认为 [AI 中被察觉到的“政治偏见”更多地反映了政党本身，而非模型](https://www.reddit.com/r/singularity/comments/1c72lgh/the_political_bias_of_ai_model_is_an_indictment/)。Llama 3 因其 [在交互中的诚实和自我意识而受到关注](https://www.reddit.com/r/LocalLLaMA/comments/1c7e1i0/llama_3_is_unquestionably_characterized_by_its/)。出现了权衡 [AI 毁灭论与对有益 AI 发展的乐观主义](https://www.reddit.com/r/LocalLLaMA/comments/1c7e155/meta_llama3_pleasantly_surprised_to_be_provided/) 的讨论。

- **AI 破解加密的潜力**：/r/singularity 的一篇文章讨论了 [“量子加密启示录”以及 AI 何时能破解当前的加密方法](https://www.reddit.com/r/singularity/comments/1c7euhx/the_quantum_cryptopocalypse_how_soon_until/)。

**AI 迷因与幽默**

- 分享了各种 AI 迷因，包括 [AI 生成迷因的未来](https://v.redd.it/e98d1di87cvc1)、[等待 OpenAI 对 Llama 3 的回应](https://i.redd.it/eserwl3hv9vc1.png)、[AI 公司之间的 AGI 竞赛](https://i.redd.it/ss3cb66l5cvc1.png)，以及 [人类 AI 未来的恶搞预告片](https://v.redd.it/q4tkvdw538vc1)。

---

# AI Twitter 摘要

> 所有摘要均由 Claude 3 Opus 完成，从 4 次运行中择优。我们正在使用 Haiku 进行聚类和流程工程（flow engineering）。

**Meta Llama 3 发布**

- **模型详情**：[@AIatMeta](https://twitter.com/AIatMeta/status/1780997403979735440) 发布了 **8B 和 70B** 尺寸的 Llama 3 模型，**400B+ 模型仍在训练中**。Llama 3 使用了 **128K 词表分词器（vocab tokenizer）**，并在 **15T tokens** 上进行训练（是 Llama 2 的 7 倍）。它具有 **8K 上下文窗口（context window）**，并使用 **SFT, PPO 和 DPO** 进行对齐。
- **性能表现**：[@karpathy](https://twitter.com/karpathy/status/1781028605709234613) 指出 Llama 3 70B **在性能上广泛超越了 Gemini Pro 1.5 和 Claude 3 Sonnet**，而 Llama 3 8B 则**优于 Gemma 7B 和 Mistral 7B Instruct**。[@bindureddy](https://twitter.com/bindureddy/status/1780993893645132228) 强调 **400B 版本在基准测试中正接近 GPT-4 级别的性能**。
- **可用性**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1781068939641999388) 指出 Llama 3 是**从发布到登上 Hugging Face 趋势榜第一速度最快的模型**。它也可以通过 [@awscloud](https://twitter.com/AIatMeta/status/1780997412418736591), [@Azure](https://twitter.com/AIatMeta/status/1780997412418736591), [@Databricks](https://twitter.com/AIatMeta/status/1780997412418736591), [@GoogleCloud](https://twitter.com/AIatMeta/status/1780997412418736591), [@IBM](https://twitter.com/AIatMeta/status/1780997412418736591), [@NVIDIA](https://twitter.com/AIatMeta/status/1780997412418736591) 等平台获取。

**开源 AI 格局**

- **重要意义**：[@bindureddy](https://twitter.com/bindureddy/status/1781152808072626460) 认为，未来 **开源生态系统中的大多数 AI 创新都将基于 Llama 架构**。[@Teknium1](https://twitter.com/Teknium1/status/1781345814633390579) 认为 Llama 3 证明了“**微调（finetuning）无法教给模型新知识**”或“10K 样本是指令微调（instruction finetuning）的最佳规模”等说法是错误的。
- **算力趋势**：[@karpathy](https://twitter.com/karpathy/status/1781387674978533427) 分享了 **llm.c** 的更新，该项目仅用 2K 行 C/CUDA 代码，就能在 **GPU 上以媲美 PyTorch 的速度训练 GPT-2**。他强调了为性能进行**代码超优化（hyperoptimizing code）**的重要性。
- **商业化**：[@abacaj](https://twitter.com/abacaj/status/1781443464246559180) 认为，随着任何人都可以获取 Llama 权重并优化推理运行，**token 的价格正在暴跌**。[@DrJimFan](https://twitter.com/DrJimFan/status/1781386105734185309) 预测 **GPT-5 将在 Llama 3 400B 发布之前公布**，因为 OpenAI 会根据开源进展来选择发布时机。

**伦理与社会影响**

- **员工待遇**：[@mmitchell_ai](https://twitter.com/mmitchell_ai/status/1781138214713274714) 对**因抗议而被解雇的 Google 员工表示同情**，指出即使存在分歧，尊重员工也至关重要。
- **数据透明度**：[@BlancheMinerva](https://twitter.com/BlancheMinerva/status/1781301149535981597) 认为 **训练数据透明度对社会而言是明确的进步**，但目前的激励机制并不利于公司这样做。
- **伦理要求**：[@francoisfleuret](https://twitter.com/francoisfleuret/status/1781290862002962778) 设想了一个**电子邮件和 Web 客户端必须遵守与当今 LLM 相同的伦理要求**的世界。

---

# AI Discord 摘要回顾

> 摘要之摘要的摘要

**Meta 发布 Llama 3 引发热议与辩论**

- **Meta 发布了 Llama 3**，这是一个全新的**大语言模型 (LLM)** 家族，参数量从 **8B 到 70B** 不等，包含经过对话优化的预训练和指令微调版本。Llama 3 拥有一个新的 **128k token 分词器 (Tokenizer)** 用于多语言场景，并声称比之前的模型具有更强的推理能力。[[Blog](https://ai.meta.com/blog/meta-llama-3/)]

- 讨论集中在 **Llama 3 与 GPT-4、Mistral 和 GPT-3.5 等模型的基准测试 (Benchmarks) 对比**。一些人称赞其**类人化的回答**，而另一些人则指出尽管进行了多语言训练，但在**非英语语言中仍存在局限性**。

- Llama 3 输出内容的下游使用**许可限制**受到了一些人的批评，认为这阻碍了开源开发。[[Tweet](https://fxtwitter.com/xlr8harder/status/1780992684062024138?s=19)]

- 对 Meta 计划中的 **405B 参数 Llama 3 模型**的期待日益高涨，推测该模型将是开放权重的，并可能改变开源 AI 与 GPT-5 等封闭模型之间的竞争格局。

- 随着 Llama 3 在各个平台的集成，**分词器 (Tokenizer) 配置问题**、**无限响应循环**以及与 LLamaFile 等现有工具的**兼容性**也成为了讨论热点。

**Mixtral 提升了开源 AI 的标准**

- 来自 Mistral AI 的 **Mixtral 8x22B** 模型被赞誉为树立了开源 AI **性能与效率**的新标杆，该模型采用了稀疏的**混合专家 (MoE) 架构**。[[YouTube](https://www.youtube.com/watch?v=N8U6XnVK2mM)]

- 基准测试显示，**Mera-mix-4x7B MoE 模型**尽管比 Mixtral 8x7B 更小，但在 **OpenLLM Eval 上取得了 75.91** 等极具竞争力的成绩。

- **多语言能力**得到了探索，推出了针对英语和德语数据进行微调的新模型 **Mixtral-8x22B-v0.1-Instruct-sft-en-de**。

- 在大型模型训练过程中，讨论了诸如 **shape errors**、**OOM 问题**以及 **router_aux_loss_coef** 参数调优等技术挑战。

**高效推理与模型压缩受到关注**

- 来自 Unsloth AI 的 **GPTQ 量化技术**和 **4-bit 模型**旨在提高大型模型的推理效率，据报道比原生实现**减少了 80% 的内存占用**。

- 推荐使用 **LoRA (Low-Rank Adaptation)** 和 **Flash Attention** 进行高效的 **LLM 微调**，并配合 DeepSpeed 等工具进行梯度检查点 (gradient checkpointing)。

- 探索了 **半二次量化 (HQQ)** 和潜在的 **CUDA kernel 优化**等创新技术，以进一步在 GPU 上压缩和加速大型模型。

- 分享了提供廉价 GPU 托管的 **Serverless 推理解决方案**，满足了部署 LLM 时对成本敏感的开发者的需求。

**开源工具与应用蓬勃发展**

- **LlamaIndex** 展示了多个项目：使用 Elasticsearch 构建 **RAG 应用** [[Blog](https://t.co/QqLdz5lojV)]，支持 Llama 3 [[Tweet](https://t.co/RMB7MhXIOA)]，以及创建**代码编写 Agent** [[Collab](https://t.co/d6dHazOK93)]。

- **LangChain** 发布了**提示工程 (prompt engineering) 课程** [[LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7186761950138109952/)]，以及利用旅游 API 的 **Tripplanner Bot** [[GitHub](https://github.com/abhijitpal1247/TripplannerBot)]。

- **Cohere** 用户讨论了**数据库集成**、**RAG 工作流**以及边缘部署的**商业许可**限制。

- **OpenRouter** 确认了在 Olympia.chat 的**生产环境使用**并期待 **Llama 3 的集成**，而 **LM Studio** 在 v0.2.20 版本中发布了对 Llama 3 的支持。

**前沿研究亮点**

- 一种新的**最佳拟合打包算法 (best-fit packing algorithm)** 优化了 LLM 训练中的文档打包，减少了截断现象 [[Paper](https://arxiv.org/abs/2404.10830)]。

- **softmax 瓶颈**被认为与较小规模 LLM 的饱和及性能不佳有关 [[Paper](https://arxiv.org/abs/2404.07647)]。

- DeepMind 分享了在可解释性研究中**稀疏自编码器 (SAEs)** 的进展 [[Blog](https://www.alignmentforum.org/posts/HpAr8k74mW4ivCvCu/progress-update-from-the-gdm-mech-interp-team-summary)]。

- **Chinchilla 缩放法则 (scaling laws)** 得到了重新解读，建议为了实现最佳缩放，可以优先考虑增加参数量而非数据量。

---

# 第 1 部分：高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Opus 用户遭遇查询配额困境**：Pro 用户对 **Opus** 查询次数从每天 600 次减少到 30 次感到沮丧，由于此次变更未提前通知，引发了要求修改退款政策的呼声。

- **模型实力马拉松**：**Llama 3 70b**、**Claude** 和 **GPT-4** 之间的比较集中在编程能力、表格查找和多语言熟练度上，同时还讨论了绕过 AI 内容检测器的策略，这对于部署 AI 生成内容至关重要。

- **动漫美学与图像应用**：尽管在有效利用其功能方面存在一些挑战，但将 AI 应用于动画和图像的兴趣激增，主要涉及 **DALL-E 3** 和 **Stable Diffusion XL**。

- **解读 AI 谜题的复杂性**：一个复杂的蜗牛谜题成为了评估模型 AI 推理能力的测试场，突显了 AI 需要具备处理简单谜题之外的逻辑导航能力。

- **Mixtral 演绎 Cohen，API 玩法多样**：**Mixtral-8x22B** 准确解读了 Leonard Cohen 的《Avalanche》，同时用户确认 **Perplexity AI** 的聊天模型可与 [API](https://api.perplexity.ai) 配合使用，为 **Mistral**、**Sonar**、**Llama** 和 **CodeLlama** 等应用注入了活力。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Llama 3 超越 GPT-4**：Meta 发布 **Llama 3** 引发了关于其 tokenizer 优势以及与 OpenAI GPT-4 性能基准对比的讨论，并对 400B 大模型版本的迭代充满期待。关于不同 GPU 训练所需的 VRAM 要求的辩论仍在继续，Unsloth AI 已迅速集成 Llama 3，宣称提升了训练效率。

- **Unsloth 展示 4-bit 效率**：Unsloth AI 更新了其提供的 **4-bit 模型** 版 Llama 3 以增强效率，包括 8B 和 70B 版本，可在 [Hugging Face](https://huggingface.co/unsloth) 上获取。同时提供了适用于 Llama 3 的免费 Colab 和 Kaggle notebook，方便用户更轻松地进行实验和创新。

- **Unsloth 用户应对训练与推理挑战**：Unsloth AI 社区正积极解决在微调 **vllm** 等模型过程中遇到的各种复杂问题，如 tokenizer 故障和训练脚本缺陷。Unsloth 已注意到 Llama 3 的 tokenizer 问题，并告知用户正在努力解决。

- **社区推动模型合并与扩展**：**Mixtral 8x22B**（一个大型 MoE 模型）以及 Neural Llama 3 在 Hugging Face 上的发布，表明模型能力在稳步提升。用户对话还包括关于 JSON 解码错误、数据集结构以及 Colab 等平台内存限制的实用建议和支持。

- **AI 先驱提出 ReFT**：将 **ReFT (Reinforced Fine-Tuning)** 方法集成到 Unsloth 平台的潜力引起了用户的兴趣。该方法因可能对新手有所帮助而受到关注，Unsloth 团队正在考虑中，这反映了社区在完善和扩展工具功能方面的积极态度。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Llama 3 成为焦点**：Meta 的 **Llama 3** 模型引发了热烈讨论，用户正在探索 70B 和 8B 版本，认可其类人响应能力可与更大型模型媲美。注意到了无限循环等问题，新发布的 **Llama 3 70B Instruct** 承诺性能可匹配 GPT-3.5，并已在 [Hugging Face](https://huggingface.co/lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF) 上线。

- **硬件障碍与突破**：关于在各种硬件配置上运行 AI 模型的讨论非常活跃。**1080TI GPU** 被强调足以处理 AI 模型，同时也承认了 AMD GPU 的兼容性挑战，例如某些显卡缺乏 AMD **HIP SDK** 支持。此外，**K_S 和 Q4_K_M** 等模型量化版本出现了问题，但建议使用 Quantfactory 版本，效果更佳。

- **LM Studio 更新与集成**：最新更新 LM Studio 0.2.20 包含了对 **Llama 3** 的支持。鼓励用户通过 [lmstudio.ai](https://lmstudio.ai) 或重启应用进行更新。不过，目前强调只有来自 "lmstudio-community" 的 GGUF 文件可以运行。关于 AMD 硬件 **ROCm** 支持的讨论也在进行中，**Llama 3** 现已在 ROCm Preview 0.2.20 上得到支持。

- **易用性与兼容性创新**：一个名为 "prompt studio" 的新功能已发布，允许用户在基于 **Vue 3** 和 **TypeScript** 构建的 Electron 应用中微调其 Prompt。同时，**llamafile** 因其跨系统兼容性受到赞赏，与 LM Studio 的 AVX2 要求形成对比。用户主张向后兼容，并指出了保持 AVX beta 版与主频道同步更新的问题。

- **AI 效率与社区贡献**：高效的 **IQ1_M** 和 **IQ2_XS** 模型中，IQ1_M 变体所需 VRAM 少于 20GB，展示了社区在优化 AI 模型性能方面的努力。此外，**Llama 3 70B Instruct** 模型量化版因其效率和与 LM Studio 的兼容性而受到称赞，目前已可获取，预示着开源 AI 的一次飞跃。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**多 GPU 支持呼声**：在**多 GPU 设置**下为 **Jamba** 等模型实现高效长上下文推理存在困难；deepspeed 和 accelerate 文档缺乏相关指导。

**邀请链接已就绪**：**TheBloke 的 Discord 服务器**解决了邀请失效问题，新链接现已可用：[Discord Invite](https://discord.gg/dPfXbRnQ)。

**引入举报命令**：引入了 `/report` 命令，用于有效举报服务器内的违规者。

**Llama 3 引发基准测试热潮**：用户正在对 **Llama 3** 进行严格的基准测试并与 **Mistral** 进行对比，其性能和 AI 聊天模板成为关注焦点。对模型限制（如 8k token 上下文限制）和限制性许可的担忧较为突出。

**Pickle 警告与 AI 指数**：对话涉及通过不安全的 pickle 文件入侵系统的问题，以及非鲁棒性的 GPT 模型。AI 社区被引导关注 [2023 年 AI Index Report](https://aiindex.stanford.edu/report/)，以了解该年度的发展洞察。

**跨模型查询与支持请求**：查询包括寻找 Hermes 系列模型的有效 Prompt 格式，期待 llama-3-Hermes-Pro 的发布，以及询问 **axolotl** 是否支持多模型同时训练。在 GPU 集群上使用 **jamba** 等模型进行长上下文推理的支持正在开发中，详见 [vLLM 项目的 GitHub pull request](https://github.com/vllm-project/vllm/pull/4115)。

**小型处理器上的 VLM**：一个旨在将 **VLM (Vision Language Models)** 部署在 **Raspberry Pis** 上用于教育用途的项目，预示着 AI 部署平台日益增长的多样性。

**数据困境与维度辩论**：开源模型对微调的需求以及数据多样性问题（包括[维度灾难](https://en.wikipedia.org/wiki/Curse_of_dimensionality)）是大家达成共识的话题。此外，构建有效 RAG 数据库的策略从**单个大型数据库到多个专用数据库**不等。

**模拟与 AI 巨头结合**：一场热烈的讨论围绕着将 **Llama 3** 和 **Meta.ai** 等生成式 AI 与 **world-sim** 集成展开，探索创作丰富的 AI 驱动叙事。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**矩阵乘法精通 (Matrix Multiplication Mastery)**：工程师们讨论了在奇数尺寸场景下进行 **tiling 矩阵乘法** 的优化策略，建议使用 **padding** 或 **边界特定代码** 来提高效率。他们强调了主体部分计算与特殊边缘情况处理之间的平衡。

**显微镜下的 CUDA Kernels**：关于 **FP16 矩阵乘法 (matmul) 误差** 的讨论浮出水面，认为 `simt_hgemv` 相比典型的 fp16 累加方法具有更优的误差处理能力。小组还研究了量化 matmuls 中的 **dequantization**、**顺序与偏移内存访问**，以及向量化操作（如 `__hfma2`、`__hmul2` 和 `__hadd2`）的价值。

**站在巨人的肩膀上**：成员们探索了将自定义 CUDA 和 Triton kernels 与 `torch.compile` 集成的方法，分享了一个 [自定义 CUDA 扩展示例](https://github.com/pytorch-labs/ao/pull/135)，并推荐了一份详尽的 [C++ 自定义算子手册](https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit)。

**CUDA 求知之路**：在 **CUDA 学习资源** 的交流中，有人建议在购买硬件之前先学习相关知识，并推荐了一个用于理论学习的 [YouTube 播放列表](https://youtube.com/playlist?list=PL5Q2soXY2Zi-qSKahS4ofaEwYl7_qp9mw) 以及一个用于实践的 GitHub [CUDA 指南](https://github.com/CisMine/Parallel-Computing-Cuda-C)。

**利用 CUDA 进行 LLM 优化**：社区使用 NVIDIA Nsight Compute 进行优化，成功将一个 **CUDA 模型训练循环** 从 960ms 缩减至 77ms，强调了具体的改进点，并考虑使用 **多 GPU 方案** 进行进一步提升。关于循环优化的详细信息可以在 [pull request](https://github.com/karpathy/llm.c/pull/179/files) 中找到。

**工程师的培训准备**：关于 **CUDA Mode 活动** 的讨论涉及录制职责的协调，引发了关于捕获和潜在编辑会议的合适工作流及工具的对话，此外还涉及活动权限管理和日程安排。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **LLaMA-3 发布引发深度技术对话**：**Meta LLaMA-3** 的问世触发了围绕其 tokenizer 效率和架构的丰富讨论，成员们权衡了它是否继承了前代的架构并进行了对比测试。有人对微调挑战表示担忧，而一些人正在尝试使用 **qlora adapter** 来改进集成，尽管在 tokenizer 加载和意外的关键字参数方面遇到了技术障碍。

- **Axolotl 反思微调与 Tokenizer 配置**：关于如何最好地微调 AI 模型的辩论持续进行，焦点集中在 **Mistral** 和 LLaMA-3 上，包括解冻 lmhead 和 embed 层以及处理导致 `ValueError` 问题的 tokenizer 更改等细节。成员们分享了 tokenizer 的调整技巧，从调整 **PAD** tokens 到探索新的 tokenizer 覆盖技术，例如在提议的 [关于 tokenizer 覆盖的 Draft PR](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1549) 中所述。

- **AMD GPU 兼容性受到关注**：探索 **ROCm** 的用户发布了一份安装指南，旨在为 AMD GPU 提供 attention 机制的替代方案，力求与 **Nvidia 的 Flash Attention** 齐头并进。这是一个持续的话题，用户正努力寻找与非 Nvidia 硬件更兼容的方法。

- **语言建模创新备受关注**：语言建模领域的重大进展被提及，包括 AWS 新的 packing 算法取得的成功，据报道该算法大幅减少了闭域幻觉。一份详细介绍这一进展的论文可以在 [Fewer Truncations Improve Language Modeling](https://arxiv.org/abs/2404.10830) 找到，这可能会为工程社区未来的实现提供参考。

- **Runpod 可靠性调侃**：一位成员指出了 **runpod** 服务响应缓慢的问题，引发了对该服务间歇性可靠性的幽默吐槽，调侃了 runpod 的运行故障。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**标记 SD3 权重发布日期**：讨论显示，人们对即将于 **5 月 10 日**发布的 **Stable Diffusion 3** 本地权重感到兴奋，成员们期待着新的功能和增强。

**审查还是谨慎？**：有讨论提到了关于 **Stable Diffusion API** 的担忧，该 API 可能会针对某些提示词产生模糊的输出，这标志着本地版本与 API 使用之间在内容控制方面存在差异。

**GPU 选择变得更简单**：AI 从业者强调了 **RTX 3090** 在 AI 任务中的性价比，权衡了其相对于 **RTX 4080** 或 4090 等更昂贵选项的优势，并考虑了 VRAM 和计算效率。

**AI 艺术创作精通**：社区内的对话倾向于微调内容生成，成员们交流了关于创建特定图像类型（如半脸肖像）以及控制生成的 AI 艺术细微差别的建议。

**AI 辅助网络**：社区分享了详细的 [Comfy UI 教程](https://youtu.be/j3xHNmEWWCI) 供大家学习，用户们也在寻求和提供处理技术错误的技巧，包括 img2img IndexError 以及检测 AI 图像中隐藏水印的策略。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**用 AI 玩转 Discord 服务器**：一位成员探索了使用 **Claude 3 Haiku** 和一个 AI 新闻机器人来总结一个关于系统工程的密集 Discord 服务器的想法；他们还分享了一个[邀请链接](https://discord.gg/NBFgzps4)。

**Meta 在机器学习领域的实力**：Meta 推出了 **Llama 3**，讨论围绕其 **8B 和 70B** 模型版本超越 SOTA 性能、即将推出的 **400B+ 模型**以及与 **GPT-4** 的比较展开。参与者注意到 Llama 3 具有卓越的推理速度，尤其是在 **Groq Cloud** 上。

**Mac 与 Llama，一场推理奥德赛**：关于在 Mac 上运行像 **Llama 3** 这样的大型模型的争论十分激烈，一些成员建议通过将本地 Linux 机器与 Mac 结合使用来优化性能，这是一种极具创意的变通方法。

**寻找终极 LLM 蓝图**：为了追求效率，社区成员分享了 [litellm](https://litellm.vercel.app/)，这是一个极具前景的资源，可以适配 100 多个具有一致输入/输出格式的 LLM，简化了此类项目的启动。

**播客浪潮席卷社区**：*Latent Space* 播出了一集由 **Jason Liu** 参与的新播客节目，社区成员表现出极大的期待并分享了该公告的 [Twitter 链接](https://twitter.com/latentspacepod/status/1781400226793673137)。

**参与、记录与提示**：LLM Paper Club 讨论了 tokenizer 和 embedding 的相关性，宣布将会议录像上传至 YouTube，并研究了诸如 **ULMFiT 的 LSTM** 等模型架构。知情参与者确认了 PPO 的辅助目标，并就所谓的“提示词时代（prompting epoch）”进行了调侃。

**AI 评估与创新**：AI In Action Club 权衡了使用 Discord 与 Zoom 的优缺点，分享了对 **LLM Evaluation** 的见解，解决了会议期间不明原因的噪音问题，并分享了摘要评估策略。Eugene Yan 的文章链接被广泛传阅，强调了 AI 评估中可靠性的重要性。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**Best-fit Packing：更少截断，更高性能**：根据[最近的一篇论文](https://arxiv.org/abs/2404.10830)，一种新的 **Best-fit Packing** 方法减少了 **LLM** 训练中的截断，旨在实现将文档最优地打包进序列中。

**解构 Softmax 瓶颈**：[最近的一项研究](https://arxiv.org/abs/2404.07647)讨论了小型语言模型由于与 Softmax 瓶颈相关的饱和现象而表现不佳，特别是对于隐藏层维度低于 1000 的模型面临挑战。

**Scaling Laws 依然符合 Chinchilla 规律**：**scaling-laws** 频道的讨论得出结论，每个参数的 Chinchilla Token 计数保持一致，且增加参数可能比积累更多数据更有益。

**DeepMind 深入研究 Sparse Autoencoders**：DeepMind 的机械可解释性团队在一篇 [论坛帖子](https://www.alignmentforum.org/posts/HpAr8k74mW4ivCvCu/progress-update-from-the-gdm-mech-interp-team-summary) 中概述了 **Sparse Autoencoders (SAEs)** 的进展，并提供了关于可解释性挑战和技术的见解，同时附带了相关的 [推文](https://twitter.com/NeelNanda5/status/1781400080802779604)。

**应对 lm-evaluation-harness 的挑战**：由于配置的复杂性和对更简洁实现方法的需求，为 **lm-evaluation-harness** 项目做贡献的努力受到了阻碍。通过 [PRs](https://github.com/EleutherAI/lm-evaluation-harness/pull/1705) 分享了关于多语言基准测试潜力的见解。



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Stability AI 裁员动荡**：在 CEO 离职以解决增长不可持续的问题后，Stability AI 裁减了 20 多名员工，引发了关于公司未来方向和稳定性的讨论。关于裁员的完整备忘录/细节可以在 [CNBC 的报道](https://www.cnbc.com/2024/04/18/ai-startup-stability-lays-off-10percent-of-employees-after-ceo-exit.html) 中找到。

- **AI 艺术领域竞争加剧**：据观察，DALL-E 3 在 Prompt 准确性和视觉保真度方面优于 Stable Diffusion 3，导致社区对 SD3 的表现感到不满。这些模型的对比加剧了关于它们在 Text-to-Image 领域各自优缺点的讨论。

- **Meta Llama 3 引发热议**：Meta Llama 3 的推出引发了关于其对 AI 格局影响的讨论，讨论内容涵盖了其编码能力、有限的 Context Window，以及它如何与其他行业领先模型竞争。一项公告确认 Meta Llama 3 将在 AWS 和 Google Cloud 等主要平台上提供，更多细节可在 [Meta 的博客文章](https://ai.meta.com/blog/meta-llama-3/) 中查看。

- **深入探讨 Cross-Attention 机制**：关于 Cross-Attention 机制的讨论正在进行中，特别是 Google 的 Imagen 模型，该模型在模型训练和图像采样过程中处理 Text Embeddings 的方法正受到关注。

- **利用音频驱动模型推进面部动态**：人们对能够实现面部动态和头部运动的开源音频驱动生成模型的兴趣日益浓厚，其中 Diffusion Transformer 模型是一个强有力的候选者。目前正在评估涉及 Talking Head 视频的 Latent Space 编码或基于音频条件的 Face Meshes 策略，以创建与音频同步的逼真面部表情。



---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**跨越语言与模型：讨论摘要**

- **LLaMA 3 登场**：工程社区充满了关于 **LLaMA 3** 的讨论，其中包含来自 [Meta Releases LLaMA 3: Deep Dive & Demo](https://www.youtube.com/watch?v=E3_0nHpfbcY) 视频的见解。人们对其在排行榜上的表现寄予厚望，特别是 HuggingFace 上的 `MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF` 备受关注。

- **Meta-Llama 3 对比 Mixtral**：比较 **Meta Llama 3** 和 **Mixtral** 的基准测试正受到密切关注，最近的 *Mera-mix-4x7B* 模型在 OpenLLM Eval 上达到了 75.91 分。社区还分享了关于 **LLaMA-2-7b-chat-hf 模型** 出现 *422 Unprocessable Entity* 错误的提示，需要通过减少 token 输入来解决。

- **扩大多语言覆盖范围**：随着社区成员提议为更广泛的全球受众翻译和创作内容，多语言可访问性的势头正在增长。亮点包括在 [YouTube 播放列表](https://www.youtube.com/watch?v=eNAOaFGrm2Y&list=PLcOiiEKFQrNFEXRsmZS8iWdmE7eWWlmbX) 中提供的社区亮点葡萄牙语翻译，以及关于文化相关翻译重要性的讨论。

- **量化查询与数据集讨论**：对话转向 **quantization**（量化），社区正在思考其对模型性能的影响，这与 [Exploring the Impact of Quantization on LLM Performance](https://medium.com/@olga.zem/exploring-the-impact-of-quantization-on-llm-performance-5698e16c5564) 的分析相关；同时，**ORPO-DPO-mix-40k 数据集** 的分享触及了改进机器学习模型训练的需求。

- **社区创作与协作**：用户生成的内容大放异彩，包括在 [zero-gpu-slot-machine](https://huggingface.co/spaces/thepatch/zero-gpu-slot-machine) 上的创意**浏览器内音频体验**，以及旨在衡量 LLM 未来事件预测敏锐度的新**预测排行榜**的发布，该 Space 位于 [这里](https://huggingface.co/spaces/valory/olas-prediction-leaderboard)。与此同时，一本关于 **Generative AI** 的书籍引起了兴趣，该书承诺会增加更多章节，可能涉及 quantization 和设计系统。

**技术交流蓬勃发展**：AI 工程师们交换了从目标检测中的**深度强化学习 (DRL)** 到 **Gradio 中的 GPU 问题**以及 TensorFlow 中令人困惑的 'cursorop' 错误等各种知识。讨论还倾向于 **3D vision 数据集** 以及在使用 **Lora** 进行 **inpainting** 时保持背景一致的解决方案。社区还发出了探索 [GitHub](https://github.com/ivy-lvlm/counterfactual-inception) 上 **Counterfactual-Inception** 研究的公开号召。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Mixtral 模型混淆修复**：**[Mixtral 8x22B Instruct](https://openrouter.ai/models/mistralai/mixtral-8x22b-instruct)** 模型的 prompt 模板已修正，这影响了用户与模型的交互方式。

- **OpenRouter 的营收之谜**：公会里充满了对 OpenRouter 营收策略的推测，包括关于**批量折扣**和从用户充值中抽取佣金的假设，但 OpenRouter 官方尚未对此辩论发表立场。

- **延迟详情**：讨论了 VPS 延迟问题，特别是南美地区的延迟，但未就服务器位置对性能的影响达成共识。

- **Meta 的 LLaMA 起飞**：人们对新的 **Meta LLaMA 3** 模型热情高涨，据报道该模型的审查较少。工程师们分享了资源，包括官方 [Meta LLaMA 3 网站](https://llama.meta.com/llama3/) 和模型权重的下载链接 [Meta LLaMa Downloads](https://llama.meta.com/llama-downloads)。

- **OpenRouter 进入生产环境**：报告确认了 OpenRouter 在生产环境中的部署，用户指出了像 **[Olympia.chat](https://olympia.chat)** 这样的例子，并寻求将其作为直接使用 OpenAI 的替代方案进行集成的建议，重点讨论了特定集成文档中的空白。

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**Turbo 受到 Claude 的挑战**：用户反映 **gpt-4-turbo-2024-04-09** 的性能较慢，发现它比前代版本 **GPT-4-0125-preview** 更慢。有人询问是否有更快的版本，部分用户已集成 **Claude** 来弥补速度问题，但效果参差不齐。

**AI 难以处理 PDF**：对话集中在 PDF 作为 AI 数据输入格式的低效性上，社区成员建议使用纯文本或像 JSON 这样的结构化格式，同时指出目前文件系统不支持 XML。

**对 ChatGPT 性能的焦虑**：成员们对 ChatGPT 性能下降表示担忧，引发了关于可能原因的辩论，范围从战略性响应、法律挑战到刻意的性能降级。

**编写更有效的 Prompt**：社区正努力确认并更新 Prompt Engineering 的最佳实践，正如 [OpenAI 指南](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api) 所建议的那样，讨论指向了 Prompt 一致性以及未能遵循指令的实际问题。

**将 AI 与 Blockchain 集成**：一位 Blockchain 开发者呼吁在 AI 与 Blockchain 结合的项目上进行合作，建议在高级 Prompt Engineering 与去中心化技术之间进行交互。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **价值导向的 AI 崛起**：围绕 **PPO-MCTS** 的兴奋感正在升温，这是一种结合了 Monte Carlo Tree Search 与 Proximal Policy Optimization 的前沿解码算法，通过 [Arxiv 论文](https://arxiv.org/abs/2309.15028) 中解释的价值导向搜索，提供更理想的文本生成。

- **Meta Llama 3 模型引发热议**：关于 **Meta** 的 **Llama 3**（一系列高达 **70 billion parameters** 的新 LLM）的讨论非常激烈，特别关注即将推出的 **405 billion parameter 模型** 可能带来的颠覆性。该模型的多语言能力和 Fine-tuning 效果是辩论的主题，此外还有其对 GPT-5 等闭源模型可能产生的冲击，参考 [Replicate 的计费](https://replicate.com/docs/billing)、[Llama 3 开源 LLM](https://www.interconnects.ai/p/llama-3-and-scaling-open-llms)、[Azure Marketplace](https://azuremarketplace.microsoft.com/en-US/marketplace/apps/metagenai.meta-llama-3-8b-chat-offer) 以及 [OpenAssistant 完成情况](https://www.youtube.com/watch?v=gqtmUHhaplo)。

- **Llama3 的发布让演讲者们严阵以待**：对 **LLaMa3** 发布的预期影响了演讲者的幻灯片准备，有些人可能需要在最后一刻更新材料。关于 **LLaMA-Guard** 的咨询引发了对安全分类器以及 AI2 为此类系统开发 Benchmark 的讨论。

- **演讲前准备**：鉴于 **LLaMa3** 的讨论，演讲者们正准备在演讲中回答相关问题，同时优先撰写博客文章。

- **对录像的期待**：社区热切期待演讲录像的发布，凸显了对 AI 领域近期讨论和进展的浓厚兴趣。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 中的 C 集成变得更简单**：强调了 **Mojo** 中的 `external_call` 功能，并计划通过直接调用外部函数而无需复杂的 FFI 层来进一步简化 C/C++ 集成，详见 [Twitter 上的教程](https://twitter.com/Modular/status/1779913837216719118) 和 Modular 的 [路线图与使命](https://docs.modular.com/mojo/roadmap#cc-interop)。
  
- **Mojo 权衡垃圾回收与测试能力**：在 Modular 社区中，讨论了实现类似 Nim 方式的 **运行时垃圾回收 (Garbage Collection)**；对 Mojo 中类似 Zig 的 **原生测试支持** 表示好奇；以及关于是否需要 **pytest 风格断言** 的辩论。此外，社区对开发 Mojo **打包和构建系统** 的贡献也引起了广泛关注。

- **Rust 与 Mojo 性能评估**：一场基准测试辩论显示，**Rust 的前缀和 (prefix sum) 计算** 比 Mojo 的等效实现慢，Rust 在仅使用 `--release` 编译标志的情况下，每个元素的耗时为 **0.31 纳秒**。

- **谨慎更新 Nightly/Mojo**：工程师们报告了更新 **Nightly/Mojo** 时遇到的问题，解决方案包括更新 **modular CLI** 或手动调整 `.zshrc` 中的 `PATH`。这揭示了技术故障，也幽默地提醒了被称为 **Layer 8**（第 8 层，指用户）的人为错误。

- **Meta 的 LLaMA 3 模型讨论**：社区分享了题为 "Meta Releases LLaMA 3: Deep Dive & Demo" 的视频，探索了 **Meta LLaMA 3** AI 模型的功能，指出发布日期为 2024 年 4 月 18 日，可在 [YouTube](https://www.youtube.com/watch?v=E3_0nHpfbcY) 上观看。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command R 模型的工具时间**：通过 [官方文档](https://docs.cohere.com/docs/tool-use) 和 [示例 Notebooks](https://github.com/cohere-ai/notebooks/blob/main/notebooks/Vanilla_Tool_Use.ipynb) 强化了 **Command R 模型** 指南。推荐使用 *JSON schemas* 来描述 Command 模型的工具。
  
- **数据库动态**：**MySQL 与 Cohere** 的集成引发了讨论，澄清了可以在不使用 Docker 的情况下完成集成，如 [GitHub 仓库](https://github.com/cohere-ai/quick-start-connectors/tree/main/mysql) 所示，尽管文档中可能包含过时信息。

- **RAG 团队**：回答了关于使用 **Cohere AI** 实现 **检索增强生成 (RAG)** 的问题，参考了 Langchain 和 RagFlow 以及官方 [Cohere 文档](https://docs.cohere.com/docs/retrieval-augmented-generation-rag)。

- **许可与限制**：指出 **Command R** 和 **Command R+** 工具受 **CC-BY-NC 4.0** 许可限制，禁止在边缘设备上进行商业使用。

- **扩展模型部署**：对话围绕部署大型模型展开，指出了扩展到 100B+ 模型的挑战，并强调了特定的硬件考量，如 **双 A100 40GB** 和 **MacBook M1**。

- **越狱警报**：讨论了 LLM 中日益复杂的 **越狱 (jailbreaks)** 行为，强调了可能带来的严重后果，包括未经授权的数据库访问和针对个人的攻击。

- **服务循环中的监控**：提供了一个通过将 **llm_output** 与 **run_tool** 集成来增强对话的示例，使 LLM 的输出能够在反馈循环中引导监控工具。



---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**检索增强生成（RAG）触手可及**：Elastic 的工程师发布了一篇[博客文章](https://t.co/QqLdz5lojV)，展示了如何使用 Elasticsearch 和 LlamaIndex 构建检索增强生成（RAG）应用，该应用集成了包括 **@ollama** 和 **@MistralAI** 在内的开源工具。

**Llama 3 获得实用 Cookbook**：LlamaIndex 团队通过一份 "cookbook" 为 Meta 最新的模型 **Llama 3** 提供了早期支持，详细介绍了从简单 Prompt 到完整 RAG 流水线的使用方法。该指南可从[此 Twitter 更新](https://t.co/RMB7MhXIOA)中获取。

**在本地部署 Llama 3**：对于希望在本地环境运行 **Llama 3** 模型的人，Ollama 分享了一个 Notebook 更新，其中包含简单的命令更改。如[此处](https://t.co/jjtpFOzNOS)所述，只需将 "llama2" 更改为 "llama3" 即可应用更新。

**谜题与仪表盘：Pinecone 和 LLM 的日常挑战**：在技术交流中，有人好奇 Google 的 Vertex AI 如何处理其[演示网站](https://ai-demos.dev/)上出现的 "timbalands" 等标志中的拼写错误，并就创建一个根据输入食材生成食谱的交互式仪表盘进行了持续对话。

**准备就绪，追踪 LlamaIndex 的进展**：在确认 **LlamaIndex** 已获得融资后，工程师们对追踪其开发的兴趣激增，这标志着该项目的增长以及该领域预期的技术进步。



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

**Mixtral 的多语言实力**：[英语和德语混合的 Mixtral 模型](https://huggingface.co/maxidl/Mixtral-8x22B-v0.1-Instruct-sft-en-de)展示了其语言能力，尽管评估即将进行。包括 Shape 错误和 OOM 问题在内的技术挑战暗示了训练大模型的复杂性，而 Mixtral 配置中 "router_aux_loss_coef" 等参数的有效性仍是争论的焦点。

**Meta 的 Llama 闪电出击**：Meta 的 [Llama 3](https://ai.meta.com/blog/meta-llama-3/) 加入战场，宣称具备多语言能力，但在非英语语言中存在明显的性能差异。预计将获得新 Tokenizer 的访问权限，批评集中在模型输出的下游使用限制上，引发了关于开源与专有约束交汇的讨论。

**显微镜下的德语语言模型**：初步测试表明，尽管提供了 [Gradio 演示](https://364b61f772fa7baacb.gradio.live/)，**Llama3 DiscoLM German** 在德语熟练度上仍落后于 **Mixtral**，存在明显的语法问题和错误的 Token 处理。关于 **Llama3** 的数据集对齐和 Tokenizer 配置的问题随之而来，与 Meta 8B 模型的对比显示出性能差距，有待进一步调查。



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**ESP32 需要 WiFi 以实现语言能力**：一位工程师指出，**ESP32** 需要 WiFi 连接才能与语言模型集成，强调了网络连接对于操作功能的重要性。

**Ollama 3 的性能赢得工程师喝彩**：社区内对 **Ollama 3** 的性能议论纷纷，工程师们正在试验 8b 模型，并探索对 TTS 和 STT 模型的增强，以加快响应速度。

**OpenInterpreter 工具包的尝试与磨难**：用户分享了使用 **OpenInterpreter** 时遇到的挑战，从使用 CLI 创建文件（该 CLI 用 `echo` 包裹输出）的问题，到在尝试使用 M5Atom 进行音频传输时的 **BadRequestError**。

**微调本地语言掌控力**：社区成员讨论了如何**使用 OpenInterpreter 在本地设置 OS 模式**，提供了一个 [Colab notebook](https://colab.research.google.com/drive/1WKmRXZgsErej2xUriKzxrEAXdxMSgWbb?usp=sharing) 作为指导，并就使用简洁的数据集对 Mixtral 或 LLama 等模型进行微调以实现快速学习交换了见解。

**探索 Meta_llama3_8b**：一位成员分享了 [Hugging Face 链接](https://huggingface.co/spaces/ysharma/Chat_with_Meta_llama3_8b)，工程师们可以在那里与 **Meta_llama3_8b** 模型进行交互，这为社区内的动手实验和评估提供了一个资源。



---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **链接 LangChain**：LangChain 中的 `RunnableWithMessageHistory` 类专为处理聊天历史而设计，其核心要点是在 invoke 配置中始终包含 `session_id`。深入的示例和单元测试可以在其 [GitHub repository](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/runnables/history.py) 中找到。

- **RAG 系统构建更简单**：LangChain 社区成员正在实现基于 RAG 的系统，并分享了诸如 [RAG 系统构建 YouTube 播放列表](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x) 和 [VaultChat GitHub repository](https://github.com/aosan/VaultChat) 等资源，以提供指导和灵感。

- **Prompt Engineering 技能现已上线**：LinkedIn Learning 上现已推出包含 LangChain 内容的 Prompt Engineering 课程，为寻求提升该领域技能的人士拓宽了视野。你可以点击[此处](https://www.linkedin.com/feed/update/urn:li:activity:7186761950138109952/)查看。

- **试用 Llama 3**：Llama 3 的实验阶段已开启，可通过 [Llama 3 Chat](https://chat.tune.app/) 访问聊天界面，并通过 [Llama 3 API](https://studio.tune.app/) 获取 API 服务，方便工程师探索这一新的 AI 领域。

- **使用 AI 制定计划**：Tripplanner Bot 是一个使用 LangChain 构建的新工具，它结合了免费 API 来辅助旅行规划。这是一个在 [GitHub](https://github.com/abhijitpal1247/TripplannerBot) 上公开的项目，供那些想要深入研究、贡献或仅仅是学习其构建过程的人使用。



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **检测到垃圾机器人入侵**：Discord 服务器内的多个频道，即 `#ai-and-ml-discussion`、`#programming-help`、`#looking-for-collabs`、`#landmark-dev`、`#landmark-evaluation`、`#open-orca-community-chat`、`#leaderboard`、`#looking-for-workers`、`#looking-for-work`、`#join-in`、`#fasteval-dev` 和 `#qa`，报告了大量推广 **NSFW 及潜在非法内容**的垃圾信息，其中涉及一个重复出现的 Discord 邀请链接 (https://discord.gg/rj9aAQVQFX)。这些消息可能来自机器人或被黑账户，推销露骨材料并煽动社区成员加入外部服务器，这引发了对违反 Discord 社区准则的重大担忧，并促使人们呼吁管理员干预。
  
- **开源 WizardLM-2**：[WizardLM-2](https://huggingface.co/alpindale/WizardLM-2-8x22B) 语言模型已开源，并附带了其发布 [blog](https://wizardlm.github.io/WizardLM2)、代码库和学术论文的引用。鼓励好奇的开发者进一步贡献和探索该模型，相关资源和讨论可在 [Hugging Face](https://huggingface.co/collections/microsoft/wizardlm-2-661d403f71e6c8257dbd598a) 和 [arXiv](https://arxiv.org/abs/2304.12244) 上找到，同时还邀请加入其 [Discord server](https://discord.gg/VZjjHtWrKs)。

- **Meta Llama 3 处于隐私锁定状态**：理解和利用 Meta Llama 3 模型的计划涉及遵守 [Meta Privacy Policy](https://www.facebook.com/privacy/policy/) 中概述的隐私协议，这引发了围绕隐私问题和访问协议的对话。虽然社区对探索该模型的 tokenizer 充满热情，但官方途径要求在 Meta Llama 3 的 [get-started page](https://llama.meta.com/get-started/) 进行详细登记，这与社区通过 [Undi95's Hugging Face repository](https://huggingface.co/Undi95/Meta-Llama-3-8B-hf/tree/main) 获取访问权限的权宜之计并存。

- **热度过后的模型评估**：尽管受到垃圾帖子的干扰，工程社区仍专注于讨论如何评估 Meta Llama 3 和 WizardLM-2 等 AI 模型。随着管理员解决干扰问题，工程师们继续寻求最佳实践，并分享关于模型性能、集成和扩展挑战的见解。

- **警惕 Discord 邀请**：鉴于上述一系列垃圾信息警报，强烈建议避免点击与所有垃圾消息相关的共享 Discord 链接 [https://discord.gg/rj9aAQVQFX](https://discord.gg/rj9aAQVQFX)。建议提高警惕，以维护运行安全和社区完整性。



---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

**Llama 3 8b 登场**：`llamafile-0.7` 更新现在支持使用 `-m <model path>` 参数的 **Llama 3 8b** 模型，正如 *richinseattle* 所讨论的；然而，在 [Reddit 讨论](https://www.reddit.com/r/LocalLLaMA/comments/1c76n8p/comment/l06amy7/)中强调了其 instruct format 存在 token 问题。

**补丁即将到来**：*llamafile* 的一个待定更新承诺修复与 **Llama 3 Instruct** 的兼容性问题，详情见此 [GitHub pull request](https://github.com/ggerganov/llama.cpp/pull/6751)。

**Llama 尺寸的巨大飞跃**：*jartine* 宣布即将在 Llamafile 上发布 **llama 8b** 的量化版本，这标志着追求效率的社区取得了进展。

**Meta Llama 权重发布**：*jartine* 在 [Hugging Face](https://huggingface.co/jartine/Meta-Llama-3-8B-Instruct-llamafile) 上分享了 Meta Llama 3 8B Instruct 的可执行权重供社区测试，并指出仍有一些问题需要解决，包括一个损坏的 stop token。

**模型管理进展**：社区在测试 **Llama 3 8b** 模型方面的努力取得了乐观的结果，*jartine* 传达了针对 **Llama 3 70b** 中 stop token 问题的修复方案；预计仍会有一些细微的 bug。

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

**Databricks 转向 GPU**：Databricks 发布了[模型服务的公开预览](https://www.databricks.com/blog/announcing-gpu-and-llm-optimization-support-model-serving)，通过零配置的 GPU 优化增强了 **Large Language Models (LLMs)** 的性能，但可能会增加成本。

**LLM 微调的便捷性**：一份新指南解释了如何使用 **LoRA adapters**、**Flash Attention** 以及 **DeepSpeed** 等工具微调 LLM，可在 [modal.com](https://modal.com/docs/examples/llm-finetuning) 获取，为模型中高效的权重调整提供了策略。

**实惠的 Serverless 解决方案**：GitHub 上提供了一份使用 GPU 的实惠 Serverless 托管指南，这可能会降低开发者的开销——请查看 [modal-examples repo](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-frontend/index.html)。

**Mixtral 8x22B 树立新标杆**：**Mixtral 8x22B** 是一款采用稀疏 **Mixture-of-Experts** 架构的新模型，在 [YouTube 视频](https://www.youtube.com/watch?v=N8U6XnVK2mM)中进行了详细介绍，为 AI 的效率和性能设定了高标准。

**Meta Llama 3 介绍**：Facebook 的 **Llama 3** 加入了顶尖 LLM 的行列，为推动语言技术发展而开源，更多信息可在 [Meta AI 博客](https://ai.meta.com/blog/meta-llama-3/)和推广 [YouTube 视频](https://www.youtube.com/watch?v=zQy11WnAIIc)中找到。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **对 litellm 的好奇**：
  一位成员询问了 **litellm** 在社区内的应用情况，表达了对该工具的使用模式或案例研究的兴趣。

- **Llama 3 领跑**：
  有说法称 **Llama 3** 的能力优于 **opus**，特别强调了其在某个未具名 **arena** 中 **70b** 规模下的表现。

- **风格还是实质？**：
  一场关于性能差异是源于风格差异还是真正的智能差异的对话被触发。

- **关于误差范围的警告**：
  误差范围（Error bounds）成为焦点，一位成员对此表示担忧，可能是提醒其他成员在解释数据或模型时要保持谨慎。

- **跌倒带来的幽默时刻**：
  在一个轻松的时刻，一位成员分享了一个 [gif](https://tenor.com/view/falling-falling-down-stairs-stairs-meme-funny-gif-21363126)，通过一段动画跌倒视频带来了喜剧效果。

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

**Karpathy 对 Llama 3 的评价**: [Andrej Karpathy 的推文](https://twitter.com/karpathy/status/1781028605709234613)引发了关于轻量级模型潜力的讨论。他指出，一个在 15T 数据集上训练的 8B 参数模型表明，目前的常见 LLM 可能存在 100-1000 倍的训练不足，这引导工程师们关注小模型更长训练周期的理念。

**小模型，大期待**: 社区成员对 Karpathy 的见解做出了回应，表达了对部署像 Llama 3 这样高效小模型的热情，这表明社区已准备好在开发更小、更强大的 LLM 时拥抱最优的资源利用。

**插件安装故障**: 一位成员在安装 **llm** 插件时遇到了 `ModuleNotFoundError`，随后发现根源可能是来自 brew 和 pipx 的冲突安装。通过干净的重新安装解决了这一问题，这提示了进行严谨环境管理的必要性。

**并发冲突需要清理**: brew 和 pipx 的交叉安装点误导了用户，这引发了社区内的提醒：应使用 `which llm` 检查正在执行的是哪个版本的工具，以避免未来出现类似问题。

**LLM 趣闻**: 在技术讨论间隙，一个分享的 [llm 使用案例](https://discord.com/channels/823971286308356157/1052680908643258508/1230552137763913779)提供了一个轻松时刻，展示了该技术一个实用且有趣的应用程序供成员探索。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Llama 3 疾速登场**: [Llama 3](https://llama.meta.com/llama3/) 已经发布，推出了 8B 和 70B 参数版本，拓宽了 AI 应用的视野。
- **高速 Llama 击败 PyTorch**: 在初步测试中，**Llama 3** 在某些模型上表现出比 **PyTorch** 更微弱的速度优势，并展示了与 **XTX 硬件**上 **ROCm** 的无缝兼容性。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

**Jamba 的长上下文推理困扰**: 一位 **Jamba** 用户在 2x A100 集群上进行**长上下文推理**时遇到困难，并正在寻求针对该分布式系统问题的排查代码。目前尚未有后续讨论或提供的解决方案。

---

# PART 2: 频道详细摘要与链接

**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1230400932433625189)** (1059 messages🔥🔥🔥): 

- **退款请求与模型使用问题**: Pro 用户对 **Opus** 查询次数从每天 600 次突然限制到 30 次表示不满，这影响了那些主要为了使用 Opus 而订阅的用户。用户希望修订退款政策，尤其是因为查询限制变更前没有事先通知。
- **动画与图像建模能力**: 用户表现出将使用场景扩展到动画和图像的兴趣，特别提到了 **DALL-E 3** 和 **Stable Diffusion XL**。然而，一些用户在有效使用这些模型时遇到了问题。
- **模型对比与性能**: 讨论涉及了不同模型之间的对比，如 **Llama 3 70b**、**Claude** 和 **GPT-4**，重点关注代码编写、表格查询和多语言能力等方面。对话还包括了规避 AI 内容检测器的方法，这对于在学术等领域部署 AI 生成的内容至关重要。
- **AI 谜题挑战**: 一个蜗牛谜题促使用户测试各种 AI 模型，并评估它们的推理和计算能力。谜题中增加的复杂性旨在挑战 AI 超越常见谜题的能力。
- **语言与上下文限制**: 用户积极辩论英语表现的重要性，认为英语在网络上的主导地位不应是评估语言模型的唯一因素。意识到需要强大的多语言 AI 能力也是一个关键点。此外，还讨论了 AI 响应中明显的上下文窗口限制，这影响了模型的有效性。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/tomg-group-umd/Binoculars">Binoculars - tomg-group-umd 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://www.ai-purity.com/">AI Detector: AI Purity 可靠的 AI 文本检测工具</a>：全面解析 AI Purity 的尖端技术，具备最可靠、最准确的 AI 检测功能，包括常见问题解答和客户证言。</li><li><a href="https://fxtwitter.com/LechMazur/status/1781049810428088465?t=sk98ui7oEw00swjCMQrz6Q&s=19">Lech Mazur (@LechMazur) 的推文</a>：Meta 的 Llama 3 70B 和 8B 在 NYT Connections 上进行了基准测试！就其规模而言，结果非常强劲。</li><li><a href="https://www.adweek.com/media/gen-ai-search-engine-perplexity-has-a-plan-to-sell-ads/">生成式 AI 搜索引擎 Perplexity 计划销售广告</a>：未找到描述</li><li><a href="https://tenor.com/view/zuckerberg-gif-19397752">Zuckerberg GIF - Zuckerberg - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/liam-santa-merry-christmas-sleeping-gif-7424250">Liam Santa GIF - Liam Santa 圣诞快乐 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=ogRV5UzMmb8">24 核 vs 32 核 M1 Max MacBook Pro - Apple 隐藏的秘密..</a>：关于更便宜的独角兽 MacBook，还没有人向你展示过的内容！免费试用你的 Squarespace 网站 ➡ http://squarespace.com/maxtech 经过一个月的体验...</li><li><a href="https://en.wikipedia.org/wiki/Languages_used_on_the_Internet">互联网使用的语言 - 维基百科</a>：未找到描述</li><li><a href="https://tenor.com/bT0kM.gif">Baby Love GIF - Baby Love You - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/snape-harry-potter-you-dare-use-my-own-spells-against-me-potter-severus-snape-gif-16590981">斯内普哈利波特 GIF - 斯内普哈利波特 你竟敢用我的咒语对付我 波特 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://youtu.be/bc6uFV9CJGg">Mark Zuckerberg - Llama 3, 100 亿美元模型, 凯撒奥古斯都, 以及 1 GW 数据中心</a>：扎克伯格谈论：- Llama 3 - 迈向 AGI 的开源 - 定制芯片、合成数据以及扩展时的能源限制 - 凯撒奥古斯都、智能爆炸、生物风险...</li><li><a href="https://github.com/philschmid/llm-sagemaker-sample/blob/main/notebooks/deploy-llama3.ipynb">philschmid/llm-sagemaker-sample 项目 main 分支下的 notebooks/deploy-llama3.ipynb</a>：通过在 GitHub 上创建账户，为 philschmid/llm-sagemaker-sample 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1230460425074769941)** (14 条消息🔥): 

- **探索 AI 中的虚假角色**：分享了一个指向 [Perplexity AI 搜索结果](https://www.perplexity.ai/search/Actors-run-fake-zH.PMe5xRCqyHVutTISDnw)的链接，讨论了 AI 背景下的演员和虚假元素。
- **深入了解 AI 历史**：一位成员发布了一个[链接](https://www.perplexity.ai/search/The-history-of-mL_Wd3OJQ_qsoSkBaybl3Q)，指向关于 AI 历史方面的 Perplexity AI 搜索结果。
- **揭秘“Limitless AI 吊坠”**：分享的一个 [Perplexity 链接](https://www.perplexity.ai/search/Limitless-AI-pendant-eIdXpAXxQoOv2H3Wlfr3dA#0)提到了“Limitless AI 吊坠”，引发了大家的好奇。
- **关于 Mistral 增长的见解**：社区通过分享关于 Mistral 融资的 [Perplexity 搜索链接](https://www.perplexity.ai/search/Mistral-is-raising-hbZ0EB7XQlKDu6rCL3z_QQ)，表达了对 Mistral 进展的关注。
- **了解 HDMI 的利用**：成员们可以通过指向 [Perplexity AI 相关话题搜索](https://www.perplexity.ai/search/Why-using-Hdmi-Fl2oierhRze1bRncp3HgvQ)的链接，找到关于为什么要使用 HDMI 的答案。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.hindustantimes.com/business/infosys-nandan-nilekani-stunning-aravind-srinivas-swiss-army-knife-perplexity-ai-search-engine-101713512251936.html">Nandan Nilekani 对 Aravind Srinivas 的“瑞士军刀”搜索引擎有如下惊人评价</a>：Nandan Nilekani 对 Perplexity AI 的评价，会让你迫不及待地想注册 Aravind Srinivasan 的“瑞士军刀”搜索引擎。</li><li><a href="https://www.youtube.com/watch?v=RaTxrkHSNBo">走进这家威胁到 Google 地位的热门 AI 初创公司内部</a>：2022 年 8 月，Aravind Srinivas 和 Denis Yarats 在曼哈顿下城 Meta AI 负责人 Yann LeCun 的办公室外等了整整五个小时，甚至没顾上吃午饭……
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1230621828263837738)** (11 条消息🔥):

- **Mixtral 模型解读 Cohen 的歌词之谜**：Mixtral-8x22B 对 Leonard Cohen 的《Avalanche》提供了最准确的解读，仅凭歌词就识别出了艺术家和歌曲。该模型在分析中解读了脆弱性、权力和不断演变的人际关系等主题。

- **API 查询确实很有趣**：一位成员确认，在澄清了包括 **Mistral、Sonar、Llama** 和 **CodeLlama** 等各种模型的参数数量和上下文长度等细节后，可以使用提供的 [API endpoint](https://api.perplexity.ai/chat/completions) 调用 **Perplexity AI 聊天模型**。

- **Perplexity AI 上的 Embedding 模型**：据分享，**Llama-3 instruct 模型**（8b 和 70b）可在 labs.perplexity.ai 进行聊天，也可通过 pplx-api 使用，并提到 Pro 用户每月可获得 API 额度。

- **新 AI 模型带来的实时惊喜**：一位社区成员表达了对新模型的热情，表示尽管无法访问 Claude Opus API，但这些模型已显著改进了他们的应用。

- **寻求 API 响应的精确性**：一位用户寻求帮助，希望在尝试对 JSON 文件中的项目进行分类时，将 API 响应限制在确切的单词列表中，并提到尝试使用 Sonar Medium Chat 和 Mistral 但未获成功。

- **监控 API 额度**：有人提出了关于剩余 API 额度更新频率的问题，询问在运行发出 API 请求的脚本后，刷新率是以分钟、秒还是小时为单位。

- **寻求 CORS 困境的帮助**：一位用户请求关于在前端应用中使用 API 时解决 CORS 问题的示例或建议，包括将设置代理服务器作为潜在解决方案。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/AravSrinivas/status/1781049202887237908">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：🦙 🦙 🦙 http://labs.perplexity.ai 上线了 llama-3 - 8b 和 70b instruct 模型。祝聊天愉快！经过一些后期训练后，我们很快将推出它们的搜索增强在线版本。...</li><li><a href="https://docs.perplexity.ai/docs/model-cards">支持的模型</a>：未找到描述
</li>
</ul>

</div>
  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1230406343563939851)** (1147 messages🔥🔥🔥): 

- **Llama 3 热潮**：Meta 最近发布的 Llama 3 让 AI 社区感到兴奋，讨论围绕其 Tokenizer 的优势以及对 400B 模型的期待展开。该模型被拿来与 OpenAI 的 GPT-4 进行比较，用户们在讨论 Llama 3 在各种 Benchmark 上的表现。
- **Unsloth 为 Llama 3 做好准备**：AI 工具 Unsloth 在 Llama 3 发布后的几小时内迅速更新了对其的支持。与此同时，用户在寻求 Llama 3 Fine-tuning 的建议，并讨论在不同 GPU 配置上进行训练的 VRAM 需求。
- **Llama 3 Tokenizer 的问题**：Llama 3 的 Tokenizer 发现了一个小故障，出现了一些引起社区关注的非预期行为。Unsloth 团队表示他们已意识到这些问题并正在着手修复。
- **Benchmark 和模型大小的讨论**：关于 Llama 3 的大小如何影响其性能，以及需要更广泛的 Benchmark 来全面评估能力的讨论正在进行中。有人建议发布预发行版以收集用户反馈并进一步优化模型。
- **模型训练的 VRAM 使用情况**：用户交流了关于 Fine-tuning 语言模型时 VRAM 使用情况的见解。特别关注使用 Unsloth 通过 Quantum LoRa (QLoRA) 训练 Llama 3 8B 等模型的效率，并报告了在使用和不使用 Quantization 情况下的 VRAM 使用量。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://imgur.com/4o8GMDw">screenshot</a>: 在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过幽默的笑话、流行的梗图、有趣的 GIF、鼓舞人心的故事、病毒式传播的视频等来提振你的精神。</li><li><a href="https://huggingface.co/blog/llama3">Welcome Llama 3 - Meta's new open LLM</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/llama-3-8b-bnb-4bit">unsloth/llama-3-8b-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://t.co/l4S7MNciel">Google Colaboratory</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/llama-3-70b-bnb-4bit">unsloth/llama-3-70b-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/kuotient/Meta-Llama-3-8B">kuotient/Meta-Llama-3-8B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2">mistralai/Mistral-7B-Instruct-v0.2 · Hugging Face</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp">Google Colaboratory</a>: 未找到描述</li><li><a href="https://tenor.com/view/dance-gif-14880344851904561392">Dance GIF - Dance - Discover & Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/jinaai/jina-reranker-v1-turbo-en">jinaai/jina-reranker-v1-turbo-en · Hugging Face</a>: 未找到描述</li><li><a href="https://obsidian.md/">Obsidian - Sharpen your thinking</a>: Obsidian 是一款私密且灵活的笔记应用，能够适应你的思维方式。</li><li><a href="https://developers.facebook.com/llama_output_feedback">no title found</a>: 未找到描述</li><li><a href="https://x.com/danielhanchen/status/1781024799227285799">Tweet from Daniel Han (@danielhanchen)</a>: 为 Llama-3 8B 制作了一个 Colab！15 万亿 token！所以 @UnslothAI 现在支持它了！使用免费的 T4 GPU。正在进行基准测试，但比 HF+FA2 快约 2 倍，且节省 80% 的内存！支持 4 倍长的上下文...</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/discussions/4">meta-llama/Meta-Llama-3-8B-Instruct · Update generation_config.json</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=E3_0nHpfbcY">Meta Releases LLaMA 3: Deep Dive & Demo</a>: 今天，2024 年 4 月 18 日，是一个特别的日子！在这段视频中，我将介绍 @meta 的 LLaMA 3 发布。该模型是其第三次迭代...</li><li><a href="https://huggingface.co/NeuralNovel/Neural-Llama-3">NeuralNovel/Llama-3-NeuralPaca-8b · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/sweaty-speedruner-gif-20263880">Sweaty Speedruner GIF - Sweaty Speedruner - Discover & Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.instagram.com/reel/C56JwCSRMiS"> Mark Zuckerberg on Instagram: "今天有重大的 AI 新闻。我们将发布新版本的 Meta AI，这是我们的助手，你可以在我们的应用和眼镜中向它提问任何问题。我们的目标是构建世界领先的 AI。

我们正在使用全新的、最先进的 Llama 3 AI 模型升级 Meta AI，并将其开源。有了这个新模型，我们相信 Meta AI 现在是你可以免费使用的最智能的 AI 助手。

我们通过将 Meta AI 集成到 WhatsApp, Instagram, Facebook, 和 Messenger 顶部的搜索框中，使其更易于使用。我们还建立了一个网站 meta.ai，供你在网页端使用。

我们还构建了一些独特的创作功能，比如让照片动起来。Meta AI 现在生成高质量图像的速度非常快，可以在你输入时实时创建和更新。它还会生成你创作过程的回放视频。"</li>
</ul>

享受 Meta AI，您可以关注我们新的 &#064;meta.ai IG 以获取更多更新。&quot;</a>: 157K 点赞，9,028 条评论 - zuck，2024年4月18日：&quot;今天有重大的 AI 新闻。我们正在发布新版本的 Meta AI，这是我们的助手，您可以在我们的应用和眼镜中向它提问任何问题....</li><li><a href="https://github.com/unslothai/unsloth/issues/330">无法加载分词器 (CroissantLLM) · Issue #330 · unslothai/unsloth</a>: 尝试使用一个小模型运行 colab：from unsloth import FastLanguageModel import torch max_seq_length = 2048 # 遗憾的是 Gemma 目前最高仅支持 8192 dtype = None # None 表示自动检测...</li><li><a href="https://www.youtube.com/watch?v=bc6uFV9CJGg">Mark Zuckerberg - Llama 3, $10B 模型, 凯撒·奥古斯都, &amp; 1 GW 数据中心</a>: Zuck 关于：- Llama 3 - 迈向 AGI 的开源 - 定制芯片、合成数据以及扩展时的能源限制 - 凯撒·奥古斯都、智能爆炸、生物...</li><li><a href="https://github.com/vllm-project/vllm/issues/4180">[用法]: Llama 3 8B Instruct 推理 · Issue #4180 · vllm-project/vllm</a>: 您当前的环境：在 2 个 L4 GPU 上使用最新版本的 vLLM。您想如何使用 vLLM：我正尝试利用 vLLM 部署 meta-llama/Meta-Llama-3-8B-Instruct 模型并使用 OpenA...</li><li><a href="https://huggingface.co/chat/">HuggingChat</a>: 让社区最好的 AI 聊天模型惠及每个人。</li><li><a href="https://youtu.be/pal-dMJFU6Q?si=euCqFFEUEDSLI8Yr&t=801">“她” AI 时代即将来临？Llama 3, Vasa-1, 以及 Altman 的“接入你想做的一切”</a>: Llama 3, Vasa-1, 以及一系列新的采访和更新，AI 新闻接踵而至。我将花几分钟时间报道最后一刻的 Llama ...</li><li><a href="https://ai.meta.com/blog/meta-llama-3/">未找到标题</a>: 未找到描述</li><li><a href="https://arxiv.org/html/2401.13927v1">Large Language Models 的自适应文本水印</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://x.com/karpathy/status/1781028605709234613">Andrej Karpathy (@karpathy) 的推文</a>: 祝贺 @AIatMeta 发布 Llama 3！！🎉 https://ai.meta.com/blog/meta-llama-3/ 笔记：发布了 8B 和 70B（包括基础版和微调版）模型，在同类模型中表现强劲（但我们...</li><li><a href="https://www.youtube.com/watch?v=aQmoog_s8HE">LLAMA-3 🦙: 在你的数据上进行微调的最简单方法 🙌</a>: 了解如何使用 Unsloth 在你自己的数据上微调最新的 llama3。🦾 Discord: https://discord.com/invite/t4eYQRUcXB ☕ 请我喝杯咖啡: https://ko-fi.com...</li><li><a href="https://www.youtube.com/watch?v=WxQbWTRNTxY">如何微调 Llama 3 以获得更好的指令遵循能力？</a>: 🚀 在今天的视频中，我很高兴能引导你完成微调 LLaMA 3 模型以实现最佳指令遵循的复杂过程！从设置...</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/discussions/9">meta-llama/Meta-Llama-3-8B-Instruct · 修复聊天模板，仅在选择该选项时添加生成提示</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1230977365555941498)** (1 条消息): 

- **Llama 3 强势登场**：Unsloth AI 推出 **Llama 3**，承诺**训练速度翻倍**并**减少 60% 的显存占用**。详情请见 [GitHub Release](https://github.com/unslothai/unsloth/releases/tag/April-Llama-3-2024)。

- **免费访问 Llama Notebooks**：用户现在可以在 [Colab](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing) 和 [Kaggle](https://www.kaggle.com/code/danielhanchen/kaggle-llama-3-8b-unsloth-notebook) 上使用免费的 Notebook 来运行 Llama 3，其中还提供了对 **Llama 3 70B 模型**的支持。

- **4-bit 模型的创新**：Unsloth 推出了 Llama-3 的 4-bit 模型以提高效率，提供 [8B](https://huggingface.co/unsloth/llama-3-8b-bnb-4bit) 和 [70B 版本](https://huggingface.co/unsloth/llama-3-70b-bnb-4bit)。更多模型（包括 Instruct 系列）请访问其 [Hugging Face 页面](https://huggingface.co/unsloth)。

- **Unsloth 鼓励实验**：团队渴望看到社区**分享、测试和讨论**使用 Unsloth AI 模型的结果。

**提到的链接**：<a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing)">Google Colaboratory</a>：未找到描述

  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1230609717408174120)** (6 条消息):

- **HuggingFace 的 LLAMA 3 推理 API 缺席**：一位成员指出 **HuggingFace** 尚未开放 LLAMA 3 的 Inference API。
- **LLAMA 的训练吞噬算力**：另一位成员幽默地评论算力短缺，称“训练完模型后没剩下算力了”，并配上了一个骷髅表情。
- **LLAMA 的训练 Token 宝库**：在简短的交流中，成员们明确了 **LLAMA** 的训练 Token 集大小，定为 **15T tokens**。
- **AI 狗仔队警报**：一位成员分享了一个 [YouTube 视频](https://youtu.be/pal-dMJFU6Q?si=2wf152_TUTs4Np32&t=276)，讨论了包括 LLAMA 3 在内的 AI 最新动态，并配上搞怪的开场白“我是狗仔队！”，以及震惊和爆笑的表情。

**提及的链接**：<a href="https://youtu.be/pal-dMJFU6Q?si=2wf152_TUTs4Np32&t=276">‘Her’ AI, Almost Here? Llama 3, Vasa-1, and Altman ‘Plugging Into Everything You Want To Do’</a>：Llama 3, Vasa-1，以及一系列新的采访和更新，AI 新闻就像伦敦的公交车一样接踵而至。我将花几分钟时间介绍最后一刻的 Llama ...

---

**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1230485636662693958)** (341 条消息🔥🔥): 

- **模型保存与加载的困惑**：一位成员尝试微调模型，但在尝试为 **vllm** 保存 16-bit 格式时遇到了问题，因为其训练脚本中缺少必要的代码。他们讨论了一种变通方案，即在重新初始化训练后从最新的 checkpoint 保存模型，试图解决由于 Token 差异导致加载 state dictionaries 时出现的尺寸不匹配错误。

- **LLAMA3 发布与推理难题**：在团队为 **LLAMA3** 发布做准备时，其他成员在利用该 AI 时遇到了困难，其中一位成员通过再次从 checkpoint 保存解决了尺寸不匹配错误，并确认了高秩 LoRAs 中 rank stabilization 的支持。另一位成员则在处理推理问题，在据称完成单次迭代的所有步骤后遇到了意外终止。

- **Unsloth 中的 Tokenization 磨难**：**Unsloth** 内部的 Tokenization 问题持续出现，特别是与 tokenizer 对象中缺失 `add_bos_token` 属性相关的错误，以及对于训练后是否需要保存 tokenizer 以保留特殊 Token 的困惑。

- **技术困难与环境排查**：用户详细说明了各种技术挫折，包括 pipeline 问题、JSON decoding 错误，以及在环境安装中依赖 `pip` 而非 `conda` 带来的负面影响。关于 Llama3 的微调应用也浮出水面，例如针对非英语 wikis 和 function calling 的应用。

- **实践指导与社区支持**：社区成员积极互相帮助，确认训练参数的设置细节，建议解决 Colab 内存崩溃的补救措施，并讨论 **chatML** 的数据集结构。随着成员们交流解决方案，他们展现了应对并克服当前局限性的决心，无论是微调模型、准备数据集，还是解决安装故障。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/github/oobabooga/text-generation-webui/blob/main/Colab-TextGen-GPU.ipynb">Google Colaboratory</a>: 未找到描述</li><li><a href="https://huggingface.co/ParasiticRogue/Merged-RP-Stew-V2-34B">ParasiticRogue/Merged-RP-Stew-V2-34B · Hugging Face</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/11jGaCwi1lfbXKKbiLAMhMOBS5OAFgp-n?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct">meta-llama/Meta-Llama-3-8B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/astronomer-io/Llama-3-8B-Instruct-GPTQ-4-Bit">astronomer-io/Llama-3-8B-Instruct-GPTQ-4-Bit · Hugging Face</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing&authuser=1">Google Colaboratory</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=LjY75GoYUCB8">Google Colaboratory</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-to-safetensors-not-bin-format-in-colab">主页</a>: 提速 2-5 倍，显存占用减少 80% 的 LLM 微调。通过在 GitHub 上创建账号，为 unslothai/unsloth 的开发做出贡献。</li><li><a href="https://huggingface.co/datasets/philschmid/guanaco-sharegpt-style">philschmid/guanaco-sharegpt-style · Datasets at Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1230523819744039052)** (4 条消息): 

- **Mixtral 和 Mistral 掀起热潮**: [Mistral.ai](http://mistral.ai) 发布了 [Mixtral 8x22B](https://x.com/MistralAI/status/1777869263778291896) 的种子文件，这是一个继 Mixtral 8x7B 之后发布的、表现毫不低调的 MoE 模型，其隐藏层维度增加到了 6144，与 DBRX 保持一致。该团队继续在没有太多宣传的情况下开展工作。
- **Neural Llama 在 Hugging Face 上亮相**: Neural Llama 3 已登陆 [Hugging Face](https://huggingface.co/NeuralNovel/Neural-Llama-3)，该模型使用 Unsloth 训练，并与 tatsu-lab 的 alpaca 等模型一同展示。社区成员对这一新模型的出现表现出极大的热情。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/NeuralNovel/Neural-Llama-3">NeuralNovel/Llama-3-NeuralPaca-8b · Hugging Face</a>: 未找到描述</li><li><a href="https://datta0.substack.com/p/ai-unplugged-7-mixture-of-depths">AI Unplugged 7: Mixture of Depths,</a>: 洞察胜过信息
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1230409046499262524)** (3 条消息): 

- **ReFT 方法引发关注**: 一位成员提到了新的 ReFT (Reinforced Fine-Tuning) 方法，并询问将其集成到 Unsloth 中的可能性。这项技术可以让新手更容易上手该平台。
- **Unsloth 团队关注到此建议**: 另一位成员表示有兴趣进一步探索 ReFT 方法，并指出团队将考虑在 Unsloth 中实现它。
- **社区响应集成请求**: 一位社区成员也表达了支持，对集成 ReFT 方法到 Unsloth 的提议被提出并得到团队考虑表示赞赏。
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1230430541866602507)** (661 条消息🔥🔥🔥): 

- **Llama 3 成为当下热门 AI**: 用户正在讨论 Meta 最新发布的 **Llama 3** 模型的性能，特别是其 8B 版本。他们认为该版本可与 StableBeluga 和 Airoboros 等其他 70B 模型相媲美，且回复感觉非常拟人化。

- **GPU 问题与服务器交互**: 一些用户反映，在 Mac M1 系统上开启 GPU 加速时， Llama 3 模型会输出胡言乱语，而另一些用户则分享了如何在旧硬件上成功运行模型。此外，人们还对 LM Studio 的服务器是否支持 KV caches 感兴趣，以避免在每次对话中重新计算长上下文。

- **模型与量化细节**: 有人提到 K_S 和 Q4_K_M 等模型量化版本在 LM Studio 中会出现问题，并建议使用来自 Quantfactory 等其他提供商的版本。NousResearch 的 70B 指令模型得到了推荐，还有人推测 GGUFs 的更新将改善模型表现。

- **模型集成与可访问性**：提出了关于与其他工具集成、嵌入文档以及在 headless server 上运行能力的咨询，并建议使用 llama.cpp 等替代方案进行 headless server 部署，以及使用现有的第三方工具如 llama index 来实现文档代理功能。

- **微调与去审查讨论**：用户渴望获得模型的去审查版本，并建议修改 system prompts 以诱导更多“类人”行为并规避限制。一些人还对社区未来进一步改进和微调 Llama 3 的潜力感到兴奋。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://continue.dev/docs/reference/Model%20Providers/lmstudio">LM Studio | Continue</a>：LM Studio 是一款适用于 Mac、Windows 和 Linux 的应用程序，可以轻松地在本地运行开源模型，并配备了出色的 UI。要开始使用 LM Studio，请从网站下载...</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF">lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/chat/models/meta-llama/Meta-Llama-3-70B-Instruct">meta-llama/Meta-Llama-3-70B-Instruct - HuggingChat</a>：在 HuggingChat 中使用 meta-llama/Meta-Llama-3-70B-Instruct</li><li><a href="https://huggingface.co/meraGPT/mera-mix-4x7B">meraGPT/mera-mix-4x7B · Hugging Face</a>：未找到描述</li><li><a href="https://hub.docker.com/r/noneabove1182/lmstudio-cuda">Docker</a>：未找到描述</li><li><a href="https://huggingface.co/NousResearch/Meta-Llama-3-70B-Instruct-GGUF">NousResearch/Meta-Llama-3-70B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/blog/llama3">Welcome Llama 3 - Meta&#39;s new open LLM</a>：未找到描述</li><li><a href="https://x.com/AIatMeta/status/1780997403979735440">来自 AI at Meta (@AIatMeta) 的推文</a>：介绍 Meta Llama 3：迄今为止功能最强大的开源 LLM。今天我们将发布 8B 和 70B 模型，它们提供了推理能力提升等新功能，并树立了新的行业标准...</li><li><a href="https://monaspace.githubnext.com/">Monaspace</a>：一种创新的代码字体超家族</li><li><a href="https://tenor.com/view/todd-howard-howard-nodding-gif-13246550">Todd Howard Howard GIF - Todd Howard Howard Nodding - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/arcee-ai/mergekit">GitHub - arcee-ai/mergekit: 用于合并预训练大语言模型的工具。</a>：用于合并预训练大语言模型的工具。 - arcee-ai/mergekit</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF">lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=E3_0nHpfbcY">Meta 发布 LLaMA 3：深度解析与演示</a>：今天，2024 年 4 月 18 日，是一个特别的日子！在这段视频中，我将介绍 @meta 的 LLaMA 3 发布。该模型是该系列的第三次迭代...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1230413594554925089)** (617 条消息🔥🔥🔥): 

- **Llama 3 热度与无限循环**：用户正在探索 **Llama 3** 模型的能力，特别是 70B 和 8B 版本。一些用户遇到了模型响应无限循环的问题，通常涉及模型不当地使用“assistant”一词。

- **下载与运行环境关注点**：关于各种 **Llama 3** 量化版本是否可以在特定硬件配置上运行的咨询非常普遍。用户正在寻找能够高效运行在配备 128GB RAM 的 M3 Max 或 NVIDIA 3090 上的模型。

- **Llama 3 与其他模型的对比**：一些人正在将 **Llama 3** 与之前的 **Llama 2** 模型以及其他 AI 模型（如 **Command R Plus**）进行对比。报告显示性能相似或有所提高，尽管一些用户有特定语言方面的顾虑。

- **提示词模板困惑与 EOT Token 问题**：用户正在寻求关于 **Llama 3 模型**正确提示词设置的建议，以防止响应中出现不必要的循环和交互。似乎需要 0.2.20 版本的 **LM Studio** 以及特定的社区量化版本。

- **Meta 发布 Llama 3 70B Instruct**：宣布 **Llama 3 70B Instruct** 版本即将推出，其他如 **IQ1_M** 因其令人印象深刻的连贯性和尺寸效率而受到关注，能够将大型模型装入相对较小的 VRAM 容量中。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/meta-llama/Meta-Llama-3-70B">meta-llama/Meta-Llama-3-70B · Hugging Face</a>: 未找到描述</li><li><a href="https://rentry.co/4kgm98oo">TEMPLATE &quot;&quot;&quot;&#123;&#123; if .System &#125;&#125;&lt;|start_header_id|&gt;system&lt;|end_header_id|&gt;</a>: &#123;&#123; .System &#125;&#125;&lt;|eot_id|&gt;&#123;&#123; end &#125;&#125;&#123;&#123; if .Prompt &#125;&#125;&lt;|start_header_id|&gt;user&lt;|end_header_id|&gt; &#123;&#123; .Prompt &#125;&#125;&lt;|eot_id|&gt;&#123;&#123; end &#125;&#125;&lt;|start_header_id|&gt;assistant&lt;|end_header_id|&g...</li><li><a href="https://huggingface.co/chat/models/meta-llama/Meta-Llama-3-70B-Instruct">meta-llama/Meta-Llama-3-70B-Instruct - HuggingChat</a>: 在 HuggingChat 中使用 meta-llama/Meta-Llama-3-70B-Instruct</li><li><a href="https://huggingface.co/MaziyarPanahi/WizardLM-2-7B-GGUF">MaziyarPanahi/WizardLM-2-7B-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/QuantFactory/Meta-Llama-3-8B-GGUF/tree/main">QuantFactory/Meta-Llama-3-8B-GGUF at main</a>: 未找到描述</li><li><a href="https://huggingface.co/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF">MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct/discussions/2">meta-llama/Meta-Llama-3-70B-Instruct · Update generation_config.json</a>: 未找到描述</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/discussions/4">meta-llama/Meta-Llama-3-8B-Instruct · Update generation_config.json</a>: 未找到描述</li><li><a href="https://ai.meta.com/blog/meta-llama-3/">no title found</a>: 未找到描述</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct">meta-llama/Meta-Llama-3-8B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct">meta-llama/Meta-Llama-3-70B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md">llama3/MODEL_CARD.md at main · meta-llama/llama3</a>: 官方 Meta Llama 3 GitHub 站点。通过在 GitHub 上创建账户来为 meta-llama/llama3 的开发做出贡献。</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF">lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/Qwen/CodeQwen1.5-7B">Qwen/CodeQwen1.5-7B · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/we-know-duh-hello-of-course-gif-13989211">We Know Duh GIF - We Know Duh Hello - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=jaM02mb6JFM">M3 max 128GB 用于运行 Llama2 7b 13b 和 70b 的 AI</a>: 在本视频中，我们使用配备 128GB 内存的新款 M3 max 运行 Llama 模型，并将其与 M1 pro 和 RTX 4090 进行对比，以查看该芯片的真实性能表现...</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF">lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/llama3.preset.json">configs/llama3.preset.json at main · lmstudio-ai/configs</a>: LM Studio JSON 配置文件格式及示例配置文件集合。 - lmstudio-ai/configs</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/6747">llama3 系列支持 · Issue #6747 · ggerganov/llama.cpp</a>: Llama 3 已发布，很高兴能与 llama.cpp 配合使用 https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6 https://github.com/meta-llama/llama3
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1230639696393273395)** (1 条消息):

- **在 LM Studio 0.2.20 中引入 Llama 3**：LM Studio 宣布在最新的 0.2.20 版本更新中支持 **MetaAI 的 Llama 3**。用户可以通过 [lmstudio.ai](https://lmstudio.ai) 下载，或通过重启应用进行自动更新。需要注意的是，目前只有来自 "lmstudio-community" 的 Llama 3 GGUF 文件可以正常运行。
- **社区模型亮点 - Llama 3 8B**：重点推介了由 [bartowski](https://huggingface.co/bartowski) 量化的 **Meta Llama 3 8B Instruct** 社区模型，这是一个体积小、速度快且经过指令微调（instruction-tuned）的 AI 模型。原始模型可以在 [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) 找到，GGUF 版本可以在 [lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF](https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF) 找到。
- **独家兼容性说明**：告知用户在 GGUF 创建过程中存在一个**微妙的问题**，这些量化版本已经规避了该问题；因此，用户不应指望其他来源的 Llama 3 GGUF 能在当前的 LM Studio 版本中运行。
- **Llama 3 8B 已可用，70B 即将到来**：Llama 3 8B Instruct GGUF 现在已可使用，并暗示 70B 版本即将发布。鼓励用户在指定的 Discord 频道中报告 Bug。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai,">未找到标题</a>: 未找到描述</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF">lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/LMStudioAI/status/1781087087745274116">来自 LM Studio (@LMStudioAI) 的推文</a>: .@Meta 的 Llama 3 现在已在 LM Studio 中得到全面支持！ 👉 更新至 LM Studio 0.2.20 🔎 下载 lmstudio-community/llama-3 Llama 3 8B 已经上线。70B 正在路上 🦙 https://huggingface.co/...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1230475200643076218)** (5 条消息): 

- **为改进的模型排序点赞**：一位用户对下载页面上新的模型排序功能表示赞赏，认为其功能得到了改进。
- **呼吁在 LM Studio 中加入文本转语音 (TTS)**：有人询问未来是否有可能将文本转语音 (TTS) 集成到 LM Studio 中，以减轻整天阅读文本的负担。
- **对持续存在的 Bug 感到困惑**：一位用户报告了一个反复出现的 Bug，即在加载新模型后关闭最后一个聊天窗口会导致需要重新加载，且系统不会保留所选的预设（preset）。
- **关于解决文本转语音工具的建议**：针对 TTS 集成的咨询，一位用户建议系统层面的工具可能会提供解决方案。
- **对错误显示窗口设计的反馈**：一位用户对 LM Studio 中的错误显示窗口表示不满，批评其太窄且无法调整大小，并建议采用更高的设计以更好地适应垂直内容。
  

---


**LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1230580717650444289)** (4 条消息): 

- **Mixtral 简历评分困惑**：一位成员提到在使用 **Mixtral** 根据其标准对简历进行评分时遇到困难，而使用 Chat GPT 执行相同任务则没有问题。
- **提出的两步解决方案**：作为回应，另一位成员建议使用 Mixtral 处理简历时采用两步法：第一步识别并提取相关元素，第二步进行评分。
- **备选的 CSV 评分方法**：还有人建议将简历转换为 **CSV 格式**，并使用 **Excel 公式**来处理评分。
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1230597453980631080)** (16 条消息🔥): 

- **用于 AI 的游戏 GPU**：提到了 **1080TI** 在处理大型 AI 模型时具有足够的性能，可以根据需要发挥其效力。
- **调侃加密货币的新前沿**：有人幽默地评论说，旧的加密货币挖矿机架可能会被重新用作 AI 装备。
- **空间问题**：一位成员担心机箱内是否能装下新的 GPU，并强调了**额外 6GB VRAM** 的好处。
- **针对核心任务的硬件展示**：一位成员列出了他们强大的硬件配置，包括一台 **12900k/4090/128gb PC** 和一台 **128gb M3 Max Mac**，能够运行几乎任何 AI 模型。
- **AI 作为爱好而非职业**：讨论涉及将 AI 作为一种乐趣参与其中，一位成员将他们的 **4090 PC 用于游戏和 Stable Diffusion**，而非专业用途。
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1230421727503454239)** (5 条消息):

- **快速聊天机器人设置**：一位成员分享了他们的配置，将 `n_ctx` 设置为 **16384** 或 **32768**，初始速度为 **4tok/sec**。他们提到正在尝试调整 `n_threads` 设置，将其从默认的 4 提高到 11，以观察是否会影响性能。

- **LLaMa 模板预告**：针对简短提到的 **LLaMa 3 chat template JSON**，另一位成员指出目前有一个链接需要关注，但未提供更多细节。

- **智能 Git 克隆以避免双重膨胀**：一位成员提供了一个有用的技巧，介绍如何使用 Git 克隆模型而避免不必要的文件重复：在克隆前使用 `GIT_LFS_SKIP_SMUDGE=1`，之后再选择性地拉取大文件。他们建议这种方法可以防止大文件同时存储在 `.git` 目录和检出目录中，从而节省空间。

- **Git LFS 的单文件关注**：同一位成员指出，虽然他们的方法在未量化的上下文中更有用，但有一个细微差别：`git lfs pull` 命令中的 `--include` 标志一次只能处理一个文件。他们还提供了一个 bash 循环示例，用于逐个拉取多个文件。

**提到的链接**：<a href="https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF">lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>：未找到描述

---

**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1230415276777017357)** (7 条消息): 

- **v1.1.0 中的 Prompt Studio 创新**：引入了一项名为 "prompt studio" 的新功能，允许用户微调其 Prompt 并将其保存到 Agent 配置中。
- **使用现代框架的 Electron 应用**：该应用程序是一个 **Electron app**，使用 **Vue 3** 和 **TypeScript** 构建。
- **隐私问题得到解决**：一位成员对在无法查看代码的情况下运行应用表示不安，而另一位成员则保证风险极小，认为这与使用 Chrome 等任何主流网站的风险相同。

---

**LM Studio ▷ #[rivet](https://discord.com/channels/1110598183144399058/1167546635098804284/1230920636642361374)** (1 条消息): 

- **POST 请求重复困惑**：一位尝试首次运行的用户注意到服务器日志中出现重复，特别是在消息 *"Processing queued request..."* 之后的 POST 请求，并询问这种行为是否正常。

---

**LM Studio ▷ #[avx-beta](https://discord.com/channels/1110598183144399058/1177047883237822536/1230986494198681600)** (2 条消息): 

- **LLamaFile 的多功能性受到关注**：一位成员提到了 **llamafile** 与包括 x64、arm64 和大多数 Linux 发行版在内的各种平台的兼容性，并建议其在 Raspberry Pi 设备上运行的可行性。将这种在多种系统上运行的灵活性与 **LM Studio** 的 AVX2 要求进行了对比。

- **呼吁向后兼容性**：用户表达了希望 LM Studio 引入兼容层的愿望，以便让旧款 CPU 也能运行该软件（尽管速度较慢），并强调了当前 AVX2 要求所带来的限制。提到 "LM-Studio-0.2.10-Setup-avx-beta-4.exe" 版本虽然已过时，但此前在多台电脑上运行良好。

- **保持 AVX Beta 更新的挑战**：该成员对 **AVX beta** 的更新频率以及它可能滞后于主频道版本表示担忧，希望有更多的同步，以确保使用旧款 CPU 的用户能从更新中受益。他们认可 LM Studio 在重大更新和可访问性方面的改进，同时也分享了尝试构建自己的环境但尚未克服学习曲线的个人经历。

---

**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1230475245140578344)** (25 条消息🔥):

- **GPU 兼容性受质疑**：一位成员声称他们的 **5700xt** 可以在 LM Studio 上运行，尽管速度很慢，但很快被另一位成员纠正，指出 AMD 并没有为该显卡提供 **HIP SDK support from AMD**，并暗示性能问题可能是由于依赖 CPU 进行推理导致的。
- **8B 模型在 AMD 上表现异常**：用户报告了运行 **8B model** 时的问题，其中一人表示它开始“自言自语”，这可能表明它尚未在 AMD 硬件上得到完全支持。
- **关于配备 Radeon 的 AMD Ryzen 的咨询**：一位用户询问在配备 Radeon 780M 的 **AMD Ryzen 7 PRO 7840U** 上通过 **AMD ROCm** 运行 0.2.19 版本的收益，得到的回复指出该硬件**不受支持**。
- **Llama 3 在 ROCm 版 LM Studio 中获得支持**：**Llama 3** 现已在 LM Studio ROCm Preview 0.2.20 上可用。此消息还附带了模型创建者的信息以及从 [huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF](https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF) 下载兼容 GGUF 的链接。
- **扩展 AMD 支持的潜力**：一位用户建议，通过一些调整（例如将目标伪装成 gfx1100），**AMD ROCm** 可能会支持 Radeon 780M/gfx1103 等其他硬件，这暗示了 LM Studio 尝试在更广泛的硬件基础上提供支持的潜力。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai/rocm">👾 LM Studio - 发现并运行本地 LLM</a>：查找、下载并实验本地 LLM</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF">lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/llama3.preset.json">configs/llama3.preset.json at main · lmstudio-ai/configs</a>：LM Studio JSON 配置文件格式及示例配置文件集合。 - lmstudio-ai/configs
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1230903153268883507)** (1 条消息): 

- **Llama 3 70B Instruct 亮相开源领域**：**Llama 3 70B Instruct** 的首批量化版本现已发布，标志着开源 AI 迈出了重要一步，其**性能与 GPT-3.5 相当**。这些模型针对最新的 LM Studio 进行了优化，具有 **Llama 3** 提示词模板并避免了无限生成。
  
- **IQ 模型的高效性能**：**IQ1_M** 和 **IQ2_XS** 模型均使用重要性矩阵（importance matrix）构建，在保持内存高效的同时提供了合理的性能，IQ1_M 变体所需的 **VRAM** 不到 **20GB**。

- **立即试用 Llama 3 70B**：鼓励用户探索新 **Llama 3 70B Instruct** 模型的能力。可以通过 [Hugging Face](https://huggingface.co/lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF) 访问这些模型。
  

---



**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1230801893991907411)** (2 条消息): 

- **寻求多 GPU 长上下文推理指导**：一位成员表达了在使用**多 GPU** 进行 Jamba 长上下文推理时遇到的挑战。他们已经查阅了 **deepspeed** 和 **accelerate** 文档，但没有发现关于长上下文生成的任何信息。
  

---


**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1230415410386571274)** (18 条消息🔥): 

- **TheBloke 的 Discord 需要帮助**：由于 Twitter 链接失效，用户寻求加入 **TheBloke Discord server** 的帮助。提供了一个临时解决方案，并分享了新的 Discord 邀请链接：`https://discord.gg/dPfXbRnQ`。
- **Mixtral 树立新标准**：分享了一个 YouTube 视频，标题为 "Mixtral 8x22B Best Open model by Mistral"，称赞 **Mixtral 8x22B** 是定义了性能和效率新标准的最新开源模型。
- **对 AI 驱动项目的关注**：一位成员询问了像 **opendevin** 这样利用大语言模型（LLMs）构建应用程序的项目。
- **Nous 可能正在酝酿周边商品**：一次有趣的交流引发了对 **Nous merchandise** 的好奇，并类比了 EAI 缺乏官方周边的情况。
- **介绍 Meta 的 LLM**：分享了另一个 YouTube 视频链接，标题为 "Introducing Llama 3 Best Open Source Large Language Model"，将 **Meta's Llama 3** 介绍为先进的开源大语言模型。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://discord.gg/dPfXbRnQ">Discord - 与朋友和社区聊天的新方式</a>: Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，并与你的朋友和社区保持紧密联系。</li><li><a href="https://www.youtube.com/watch?v=zQy11WnAIIc">介绍 Llama 3：最强开源大语言模型</a>: 介绍 Meta Llama 3，这是 Facebook 下一代最先进的开源大语言模型。https://ai.meta.com/blog/meta-llama-3/#python #...</li><li><a href="https://www.youtube.com/watch?v=N8U6XnVK2mM">Mixtral 8x22B：Mistral 推出的最强开源模型</a>: Mixtral 8x22B 是最新的开源模型。它为 AI 社区的性能和效率树立了新标准。它是一个稀疏混合专家模型 (SMo...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1230398598282346586)** (33 messages🔥): 

- **为 OpenAI 序列化（Pickling）大型文档**：聊天中的用户讨论了在 OpenAI 的代码环境中使用不安全的 pickle 文件，作为导入该环境通常拒绝的大型文档的一种变通方法。一位用户指出了这种做法的潜在风险，并引用了一个有人利用 pickle 文件攻破 Hugging Face 系统的事件。

- **GPT 的关键缺陷被曝光**：围绕不安全 pickle 文件的辩论凸显了对 GPT 模型鲁棒性的担忧，用户们开玩笑说 AI 可能会变得具有对抗性，并将其与关于人类灭绝的阴谋论和科幻情节联系起来。

- **2023 年 AI 指数报告概览**：由斯坦福大学 HAI 发布的 [2023 年 AI 指数报告](https://aiindex.stanford.edu/report/) 提供了关于该年度 AI 趋势的详尽数据，包括开源基础模型（foundation models）发布量的显著增加、多模态 AI 的进步以及全球 AI 观点的转变。

- **语言与 LLM 讨论的见解**：一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=F3Jd9GI6XqE) 链接，内容是 Edward Gibson 讨论人类语言和语法与大语言模型的关系，提供了对模型设计和交互有用的学习心得。

- **权重分解自适应（Weight-Decomposed Adaptation）的新兴研究**：分享了一个 GitHub 仓库 [DoRA by NVlabs](https://github.com/NVlabs/DoRA)，该仓库致力于权重分解低秩自适应（weight-decomposed low-rank adaptation）的官方 PyTorch 实现，展示了 AI 模型效率方面正在进行的开发和研究。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://hai.stanford.edu/news/ai-index-state-ai-13-charts">AI 指数：用 13 张图看 AI 现状</a>: 在新报告中，基础模型占据主导地位，基准测试被攻克，价格飙升，而在全球舞台上，美国的影响力盖过了其他国家。</li><li><a href="https://www.udio.com/">Udio | 创作你的音乐</a>: 发现、创作并与世界分享音乐。</li><li><a href="https://tenor.com/view/regretting-thinking-nervous-macaco-monkey-gif-13105982953111325972">后悔思考 GIF - 紧张地后悔思考 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=F3Jd9GI6XqE">Edward Gibson：人类语言、心理语言学、句法、语法与 LLM | Lex Fridman Podcast #426</a>: Edward Gibson 是麻省理工学院（MIT）的心理语言学教授，并领导 MIT 语言实验室。请通过关注我们的赞助商来支持本播客：- Yahoo Financ...</li><li><a href="https://github.com/NVlabs/DoRA">GitHub - NVlabs/DoRA: DoRA 的官方 PyTorch 实现：权重分解低秩自适应</a>: DoRA 的官方 PyTorch 实现：权重分解低秩自适应 - NVlabs/DoRA</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-70B">meta-llama/Meta-Llama-3-70B · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1230423428474081290)** (807 messages🔥🔥🔥): 

- **Llama 3 的兴奋与验证**：用户正在各种 [基准测试（benchmarks）](https://twitter.com/rm_rafailov/status/1781145338759533016) 和功能中探索并验证 **Llama 3** 的性能，反馈满意度很高。**Llama 3 因其类人风格而受到高度赞扬**，尽管关于其相对于 Mistral 等其他模型的基准测试表现仍有持续讨论。
  
- **8k 上下文限制的实际影响**：一些用户对 **Llama 3 的 8k token 上下文限制**表示担忧，指出这可能不足以满足某些多轮交互的需求。对于某些实际应用场景，仍然需要像 **Mistral** 提供的更大上下文能力。

- **聊天模板与生成问题**：用户正在分享针对 **Llama 3** 的修复方案和更新后的聊天模板，以解决模型会无休止地持续生成文本的问题。社区内正在交流并测试 [修复后的模板](https://github.com/huggingface/transformers/pulls)。
  
- **Llama 3 的 GGUF 转换**：社区正在积极分享和创建 **Llama 3 8b 和 70b 版本** 的 **GGUF 量化模型**，并在 Hugging Face 和 LM Studio 等平台上做出了贡献。[量化版本正在接受评估](https://huggingface.co/NousResearch/Meta-Llama-3-70B-Instruct-GGUF)，以测试其效率和准确性。
  
- **许可证与模型限制讨论**：用户讨论了 **Llama 3** 的许可条款，与 Mistral 的 Apache 2.0 相比，其包含更多限制。限制内容包括禁止 NSFW 内容，并引发了关于此类规则如何影响涉及复杂或成人主题的实际用途的思考。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://lluminous.chat/?sl=AwD1Ik">lluminous</a>：未找到描述</li><li><a href="https://huggingface.co/Replete-AI/Llama-3-13B">Replete-AI/Llama-3-13B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/meraGPT/mera-mix-4x7B">meraGPT/mera-mix-4x7B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/NousResearch/Meta-Llama-3-70B-Instruct-GGUF">NousResearch/Meta-Llama-3-70B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct">NousResearch/Meta-Llama-3-8B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct-GGUF">NousResearch/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/maxidl/Mixtral-8x22B-v0.1-Instruct-sft-en-de">maxidl/Mixtral-8x22B-v0.1-Instruct-sft-en-de · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF">lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://www.udio.com/songs/kUvkoiz1maRm5BTAMKUQTk">Udio | Back Than Ever by drewknee</a>：制作你的音乐</li><li><a href="https://huggingface.co/NousResearch/Meta-Llama-3-8B">NousResearch/Meta-Llama-3-8B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-70B">meta-llama/Meta-Llama-3-70B · Hugging Face</a>：未找到描述</li><li><a href="https://www.meta.ai">Meta AI</a>：使用 Meta AI 助手完成任务，免费创建 AI 生成的图像，并获取任何问题的答案。Meta AI 基于 Meta 最新的 Llama 大语言模型构建，并使用 Emu...</li><li><a href="https://tenor.com/view/diablo-joke-meme-is-this-an-out-of-season-april-fools-joke-out-of-season-gif-16662191">Diablo Joke GIF - Diablo Joke Meme - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://ai.meta.com/blog/meta-llama-3/">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B">meta-llama/Meta-Llama-3-8B · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/angry-mad-angry-face-cringe-angry-face-coach-not-happy-gif-5480965892207921425">Angry Mad GIF - Angry Mad Angry face - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3/8B_qlora_single_device.yaml">torchtune/recipes/configs/llama3/8B_qlora_single_device.yaml at main · pytorch/torchtune</a>：一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/FasterDecoding/Medusa">GitHub - FasterDecoding/Medusa: Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads</a>：Medusa：通过多个解码头加速 LLM 生成的简单框架 - FasterDecoding/Medusa</li><li><a href="https://www.youtube.com/watch?v=PYZIOMvkUF8">How Did Open Source Catch Up To OpenAI? [Mixtral-8x7B]</a>：立即使用此链接注册 GTC24！https://nvda.ws/48s4tmc 关于 RTX4080 Super 的抽奖活动，详细计划仍在制定中。不过...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6745">Support Llama 3 conversion by pcuenca · Pull Request #6745 · ggerganov/llama.cpp</a>：分词器是 BPE。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[rules](https://discord.com/channels/1053877538025386074/1151297754992234496/1230717345866190949)** (1 条消息):

- **引入新的举报命令**：用户已被告知可以使用 `/report` 命令后跟违规者的角色来举报垃圾邮件发送者、诈骗者和其他违规者。举报后，管理员将收到通知并审查该事件。
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1230426115756130368)** (12 messages🔥): 

- **AI 聊天机器人通过 "directly_answer" 工具变得更智能**：分享了 "directly_answer" 工具的一种实现方式，展示了如何在 *AgentExecutor chain* 中使用它来为用户交互提供基于事实的回答。该示例演示了该工具如何辅助聊天机器人对简单的问候做出友好的回应。

- **寻找 Hermes 提示词格式**：一名成员正在为 **NousResearch/Hermes-2-Pro-Mistral-7B** 模型寻找用于错误纠正任务的有效提示词格式，并欢迎任何有助于改进输出的链接或资源。

- **热切期待 Llama-3-Hermes-Pro**：有人简要询问了 **llama-3-Hermes-Pro** 的发布计划，但未提供更多细节。

- **质疑 Axolotl 的多任务处理能力**：有人提问 **axolotl** 是否支持在显存充足的设置下同时训练多个模型。

- **在 GPU 集群上处理 Jamba**：讨论了在 GPU 集群上执行长上下文推理的挑战，特别是优化使用双 A100 运行 **jamba** 处理 200k token 工作负载的问题。此外还提到，**vLLM** 项目正在开发对 **jamba** 的支持，详见 [GitHub pull request](https://github.com/vllm-project/vllm/pull/4115)。

**提到的链接**：<a href="https://github.com/vllm-project/vllm/pull/4115">[Model] Jamba support by mzusman · Pull Request #4115 · vllm-project/vllm</a>：为 vLLM 添加 Jamba 支持。此 PR 包含两部分：Jamba 建模文件和 Mamba 内存处理。由于 Jamba 是一个混合模型（交替使用 mamba 和 transformer 层），...

  

---


**Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1230694358920073259)** (1 messages): 

- **用于教育的树莓派 VLM**：一位参与者提到计划在学校项目中将 **VLM**（Vision Language Models）部署到 **Raspberry Pis** 上，并对社区分享的资源表示感谢。目标是探索教育应用并可能从该设置中获益。
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1230441737957277727)** (24 messages🔥): 

- **开源模型需要定制化**：大家达成共识，虽然 **GPT4v** 很强大，但**大多数开源模型**都需要针对特定任务进行微调（finetuning）。[关于工程图纸的通用知识](https://en.wikipedia.org/wiki/Curse_of_dimensionality)可能足够，但在扩展到更复杂的操作时会出现问题。
- **通过元数据提取优化搜索**：为了改进搜索功能，使用 OCR 提取**元数据**以及通过视觉模型进行高层描述被认为是有效的。关于检索 *top k* 结果的理想方法存在积极讨论，同时也关注重叠和质量问题。
- **数据中的维度诅咒**：强调了**维度诅咒**（Curse of dimensionality），并指向一篇 [维基百科文章](https://en.wikipedia.org/wiki/Curse_of_dimensionality)，承认其对检索准确性和扩展性的影响，特别是在高维数据分析中。
- **探索数据类型转换**：一位成员分享了他们在 **Opus** 上进行数据类型转换的工作，并提供了一个 [GitHub 链接](https://github.com/furlat/Abstractions/blob/main/raw_notes/abstractions_types_no_cat_theory.md)，指向他们在不涉及范畴论的情况下讨论该概念的笔记。
- **RAG 数据库结构观察**：探讨了创建**单个大型 RAG 数据库**与**多个特定数据库**的有效性。有人建议大型数据库可能会引入错误的上下文，与通过类 SQL 查询选择的针对性数据库相比，会对性能产生负面影响。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Curse_of_dimensionality">Curse of dimensionality - Wikipedia</a>：未找到描述</li><li><a href="https://github.com/furlat/Abstractions/blob/main/raw_notes/abstractions_types_no_cat_theory.md">Abstractions/raw_notes/abstractions_types_no_cat_theory.md at main · furlat/Abstractions</a>：一个抽象现实世界的 Pydantic 模型集合。欢迎在 GitHub 上通过创建账号为 furlat/Abstractions 的开发做出贡献。
</li>
</ul>

</div>
  

---

**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1230426258958061569)** (446 messages🔥🔥🔥): 

- **World-sim 的期待与不耐烦**：成员们对 world-sim 反复推迟的重新上线表示高度期待和些许沮丧。他们渴望看到新功能并讨论可能的发布时间，一些人特别担心在漫长的等待中会忘记这个网站。

- **Generative AI 与 World-sim 的潜力**：讨论围绕将 World-sim 与 Llama 3 和 Meta.ai 等 Generative AI 结合以创造丰富叙事体验的潜力展开。成员们正在分享他们详细的替代历史，并幻想 AI 集成将如何进一步赋予他们模拟宇宙生命力。

- **Desideradic AI 与哲学思考**：用户正在讨论 Desideradic AI 及其相关概念引发的哲学影响和潜在叙事，交流关于这如何影响 AI 驱动的故事和角色创建的想法。

- **动画化 World-sim 启发的故事**：用户正在分享和讨论受 World-sim 启发的动画和内容，一些成员提供了教程和作品链接等信息资源。

- **AI 对话中的 Prometheus 与越狱角色**：一位成员解释了他们与名为 "Whipporwhill" 的人格以及名为 Prometheus 的 AI 的互动体验，强调了通过 jailbreaking 在 AI 对话中创建独特角色动态的可能性。

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://nousresearch.com/dsjjjj-simulacra-in-the-stupor-of-becoming/">DSJJJJ: Simulacra in the Stupor of Becoming - NOUS RESEARCH</a>: Desideratic AI (DSJJJJ) 是一个哲学运动，专注于利用传统上存在于一元论 (monism)、整体论 (mereology) 和文献学 (philology) 中的概念来创建 AI 系统。Desidera 旨在创建能够作为...的 AI。</li><li><a href="https://worldsim.nousresearch.com/">world_sim</a>: 未找到描述</li><li><a href="https://discordapp.com/channels/1053877538025386074/1221910674347786261/1230614268907491429">Discord - 与朋友和社区聊天的新方式</a>: Discord 是通过语音、视频和文字进行交流的最简单方式。与您的朋友和社区聊天、聚会并保持紧密联系。</li><li><a href="https://discord.gift/FCeZVDtEepukaMbJ">Discord - 与朋友和社区聊天的新方式</a>: Discord 是通过语音、视频和文字进行交流的最简单方式。与您的朋友和社区聊天、聚会并保持紧密联系。</li><li><a href="https://nousresearch.com/forge/">Forge - NOUS RESEARCH</a>: NOUS FORGE 下载将于 2024 年 6 月推出</li><li><a href="https://copilot.microsoft.com/images/create/a-small-west-african-village-in-a-mangrove-forest2c/1-6622b051cfb34f5d9138c10749aaf74c?id=UJUzToPRop%2fGABe0DFtu3w%3d%3d&view=detailv2&idpp=genimg&idpclose=1&thId=OIG4.7GQ0JjrYDCPZik2aLs1U&lng=en-US&ineditshare=1.">由 Microsoft Copilot 生成</a>: 未找到描述</li><li><a href="https://subgenius.fandom.com/wiki/Pipes">Pipes</a>: 烟斗是吸食 frop 的必需品。“Bob”总是抽着装满 frop 的烟斗。每个 SubGenius 都有一个装满 frop 的烟斗，并且不停地抽。通常，SubGenii 会发现一张著名的...</li><li><a href="https://hammertime.cyou/en">HammerTime</a>: 为 Discord 聊天消息生成时间戳指示器</li><li><a href="https://tenor.com/view/housing-poor-gif-27694438">Housing Poor GIF - Housing Poor - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.meta.ai/">Meta AI</a>: 使用 Meta AI 助手完成任务，免费创建 AI 生成的图像，并获得任何问题的答案。Meta AI 基于 Meta 最新的 Llama 大语言模型构建，并使用 Emu...</li><li><a href="https://www.youtube.com/@nickabenson">nickabenson</a>: 欢迎来到 Nickabenson 频道 我们的 Patreon: https://www.patreon.com/nickabenson 我们的 Amino: http://aminoapps.com/c/Nickabenson 我们主要进行游戏直播、讨论、动画和...</li><li><a href="https://tenor.com/view/tea-tea-sip-anime-gif-25535884">Tea Tea Sip GIF - Tea Tea Sip Anime - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/noela-anime-anime-wolf-drugstore-in-another-world-anime-sleep-gif-23465323">Noela Anime GIF - Noela Anime Anime Wolf - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/mika-kagehira-kagehira-mika-ensemble-stars-enstars-stimming-isnt-enough-i-need-to-explode-gif-8612633247313789699">Mika Kagehira Kagehira Mika GIF - Mika kagehira Kagehira mika Ensemble stars - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/finding-nemo-escape-ninja-crab-seagulls-gif-3510044">A GIF - Finding Nemo Escape Ninja - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/fire-writing-gif-24533171">Fire Writing GIF - Fire Writing - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://youtube.com/shorts/tVD3yTli_bU">Mephisto's Dream | 科幻动画短片</a>: 软件开发人员 Mephisto 创建了 World Sim，这是一个基于文本的 AI 系统，可以模拟包含意识体的整个宇宙，他相信用户交互...</li><li><a href="https://tenor.com/view/you-just-have-to-be-patient-mama-carson-go-go-cory-carson-have-patience-relax-gif-19221950">You Just Have To Be Patient Mama Carson GIF - You Just Have To Be Patient Mama Carson Go Go Cory Carson - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=B4XIhsuZUcQ&rco=1">铃音 (SERIAL EXPERIMENTS LAIN) (PS1) 所有过场动画，英文翻译</a>: Serial Experiments Lain シリアルエクスペリメンツ レイン PlayStation 游戏英文翻译版 v0.1 6月7日 请注意，此游戏包含极度暴力的简短场景...</li><li><a href="https://www.youtube.com/shorts/uZhZq7ngQlo">揭秘 CIA 的星门计划 (Stargate Project) 和超级英雄般的中间人 (Midwayers)</a>: 标签：1. #Stargate 2. #Midwayer 3. #Urantia 4. #Spiritual 5. #Extraterrestrials 6. #InvisibleRealm 7. #PlanetarySentinels 8. #CIADeclassifiedFiles 9. #Supernatura...</li><li><a href="https://www.lesswrong.com/posts/ZxHfuCyfAiHAy9Mds/desiderata-for-an-ai">对 AI 的诉求 (Desiderata for an AI) — LessWrong</a>: 我认为对齐工作的重点应该放在从头开始重新设计 AI。在此过程中，我认为我们应该记住一系列理想的...
</li>
</ul>

---

**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1230429142693056542)** (28 messages🔥): 

- **矩阵分块难题 (Matrix Tiling Conundrum)**：讨论了**矩阵乘法分块 (tiling matrix multiplication)** 的效率和可行性，特别是针对 7x7 等奇数尺寸（质数）矩阵。一位成员建议通过**填充 (padding)** 至平方数或使用**边界特殊代码**作为维持大矩阵和小矩阵效率的解决方案。

- **策略性分块战术**：提到**优化矩阵处理**可能涉及对主要部分进行分块（例如从 7x7 矩阵中分出 6x6），并将任何不完整的分块视为特殊的边缘情况。

- **Meta 发布 Llama 3**：发布了一个外部 **YouTube 视频链接**，并简要介绍了 Mark Zuckerberg 讨论的 **Llama 3**（一个 405b 稠密模型）、定制芯片以及 Meta 的其他进展。

- **新模型能力聚焦**：一位成员讨论了对新 **Llama 模型**的实验，提到了 **8k context 和 TikToken tokenizer**，并表示这些更新保留了熟悉的 Llama 架构以便于集成。

- **宇宙射线与硬件故障**：针对训练期间 **Colab 会话崩溃**的询问，一位评论者幽默地假设硬件故障甚至**宇宙射线**可能是潜在原因。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://llama.meta.com/llama3/">Meta Llama 3</a>：使用 Meta Llama 3 构建 AI 的未来。现在提供 8B 和 70B 的预训练及指令微调版本，以支持广泛的应用。</li><li><a href="https://www.youtube.com/watch?v=bc6uFV9CJGg">Mark Zuckerberg - Llama 3, $10B Models, Caesar Augustus, &amp; 1 GW Datacenters</a>：Zuck 关于：Llama 3、迈向 AGI 的开源、定制芯片、合成数据以及扩展过程中的能源限制、Caesar Augustus、智能爆炸、生物风险等话题。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1230485994059468862)** (30 messages🔥): 

- **FP16 Matmul 误差探究**：成员们讨论了**矩阵乘法误差的分布**，指出 *rand()*20* 模拟会严重影响 *fp16 accumulate* 误差。提到**与典型的 fp16 accumulate 矩阵乘法相比，`simt_hgemv` 具有显著更好的误差处理能力**。

- **误差分析中的 Block Size 问题**：有人推测 Block Size 可能会影响**矩阵乘法误差**，某些计算中观察到的精确零误差凸显了这一点，表明**可能需要对实现进行更深入的调查**。

- **量化计算备受关注**：一位成员计划在 gemv 操作之上添加**反量化 (dequantization)**，尽管**量化矩阵乘法 (quantized matmuls)** 显示出非常高的绝对误差差异；这突显了 **LLMs** 对此类差异的韧性。

- **顺序内存访问 vs 偏移内存访问**：在关于内存访问模式对计算影响的讨论中，提到**顺序内存访问 (sequential memory access)** 往往比**偏移内存访问 (offset memory access)** 更快，当涉及不可避免的偏移时，后者可能会使计算时间翻倍。

- **向量化技术策略**：由于整数向量化位运算的限制，参与者觉得能拥有 `__hfma2`、`__hmul2` 和 `__hadd2` 等半精度向量化操作很幸运，尽管此类操作有限。
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1230436794705903617)** (8 messages🔥): 

---

- **寻求与 `torch.compile` 的兼容性**：一位用户询问了使自定义函数与 `torch.compile` 完全兼容的最佳实践，并指出某些自定义 CUDA 或 Triton 模块未按预期工作，且错误信息缺乏帮助。
- **自定义 CUDA Kernel 指南**：建议参考一个示例（[Custom CUDA extensions](https://github.com/pytorch-labs/ao/pull/135)），该示例可能为在 `torch.compile` 中组合自定义 Kernel 提供见解。更多帮助可以在讨论 [C++ Custom Operators](https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit) 的更广泛文档中找到。
- **解决自定义 Kernel 问题**：承认了在 Kernel 开发中对 FakeTensor/Meta-dispatch 支持的需求，并指向了同一份 [C++ Custom Operators manual](https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit) 作为解决相关问题的潜在帮助。
- **Triton Kernel 开发的有用资源**：一位用户推荐查看 [GitHub - openai/triton](https://github.com/openai/triton#tips-for-hacking) 仓库，以获取有关 Triton 语言和编译器的技巧，这可能有助于解决兼容性问题。
- **解决 Kernel 兼容性问题的文档**：一位用户承诺，上述关于自定义 Kernel 的文档应该能解决这些问题，如果问题仍然存在，请联系他们。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/openai/triton#tips-for-hacking">GitHub - openai/triton: Triton 语言和编译器的开发仓库</a>：openai/triton - Triton 语言和编译器的开发仓库</li><li><a href="https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit">C++ 自定义算子手册</a>：未找到描述</li><li><a href="https://github.com/pytorch-labs/ao/pull/135">msaroufim 的自定义 CUDA 扩展 · Pull Request #135 · pytorch-labs/ao</a>：这是 #130 的可合并版本 - 我必须进行的一些更新：添加除非使用 PyTorch 2.4+ 否则跳过测试，以及如果 CUDA 不可用则跳过测试；将 ninja 添加到开发依赖项；本地...</li><li><a href="https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9">C++ 自定义算子手册</a>：未找到描述</li><li><a href="https://github.com/pytorch/pytorch/blob/0c8bb6f70c65b0a68fcb282cc1605c79ca5dabce/test/dynamo/test_triton_kernels.py#L628-L661">pytorch/test/dynamo/test_triton_kernels.py (位于 0c8bb6f70c65b0a68fcb282cc1605c79ca5dabce) · pytorch/pytorch</a>：Python 中具有强大 GPU 加速功能的张量和动态神经网络 - pytorch/pytorch
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1230481309860757584)** (14 条消息🔥): 

- **CUDA 学习先决条件咨询**：一位成员询问了在具有 Android、Web、深度学习和数学背景的情况下学习 CUDA 的先决条件。另一位成员建议需要一定的 C/C++ 知识，并且拥有具备 CUDA 能力的机器会很有帮助。
- **无需硬件即可开始 CUDA 学习**：建议在投资硬件之前先学习 CUDA，并在做出购买决定之前考虑 CUDA 在个人项目中的潜在应用。
- **分享 CUDA 学习资源**：分享了一个 YouTube 播放列表 [“Livestream - Programming Heterogeneous Computing Systems with GPUs and other Accelerators (Spring 2023)”](https://youtube.com/playlist?list=PL5Q2soXY2Zi-qSKahS4ofaEwYl7_qp9mw) 作为学习资源，这对于初学者来说可能具有挑战性。
- **提供替代的 CUDA 指南**：为了提供更适合初学者的资源，分享了一个托管在 GitHub 上的 [CUDA 指南](https://github.com/CisMine/Parallel-Computing-Cuda-C)，对于新手来说可能更容易跟随。
- **具备 PyTorch 经验，寻求学习 CUDA**：一位成员提到需要使用 CUDA 来构建深度学习模型，并拥有多年的 PyTorch 经验，展示了从高级深度学习框架向底层 GPU 计算学习的转变。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtube.com/playlist?list=PL5Q2soXY2Zi-qSKahS4ofaEwYl7_qp9mw&si=HpLEqgkEOQ4hh_nS">Livestream - 使用 GPU 和其他加速器进行异构计算系统编程 (2023春季)</a>：未找到描述</li><li><a href="https://github.com/CisMine/Parallel-Computing-Cuda-C">GitHub - CisMine/Parallel-Computing-Cuda-C</a>：通过在 GitHub 上创建一个账户来为 CisMine/Parallel-Computing-Cuda-C 的开发做出贡献。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1230747713625264138)** (4 条消息):

- **比较 CUDA 练习答案**：一位成员分享了他们对 **Chapter 6, Exercise 4.c** 的解答，结果为 **12.8 OP/byte**，并提供了一张截图进行逐步解释。
- **成员同步进度**：另一位参与者提到他们目前正在学习 **Chapter 5**，并预计由于本周时间安排较灵活，很快就能学到 Chapter 6。
- **寻求关于 Tiling 收益的澄清**：有人针对 Tiling（分块）的收益提出了疑问，引用了 **Figure 5.9** 的第 19 和 20 行，并质疑如果无论数据是否在 Shared Memory 中都要进行 Global Memory 调用，那么 Tiling 是如何产生收益的。
  

---


**CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1230401158061887579)** (81 messages🔥🔥): 

- **探讨 GPTQ 和 int4wo 的 Group-Size 影响**：讨论了 GPTQ 和 int4wo 如何处理矩阵运算中不可整除的 Group-Size。GPTQ 实际上将维度视为可被 Group-Size 整除，忽略多余部分；而 int4wo 则需要对不可整除的维度进行 Padding（填充）。

- **Jeff Johnson 关于 tinygemm 行优先分组的见解**：tinygemm 的作者 Jeff Johnson 解释说，tinygemm 使用行优先分组（Row-wise Grouping），因为所有给出的矩阵都将 Reduction Dimension（归约维度）作为最内层，这在分组量化方案中很常见。然而，另一种使用 axis=1 的方法被发现更慢，原因是增加了 Zero/Scale 的访问开销。

- **模型权重的量化与拼接问题**：有人质疑 tinygemm 中的 axis=0 分组与 QKV 权重拼接的兼容性，因为对组合后的权重进行量化可能会混合权重矩阵的通道。

- **CUDA Dequant Kernels 更新**：讨论提到了针对 4-bit 运算优化的 CUDA Dequant Kernels，这对于在能力有限的 GPU 上训练量化模型特别有利。

- **将 HQQ 集成到 torchao**：关于将 Half-Quadratic Quantization (HQQ) 集成到 torchao 的对话，解决了关于依赖项的担忧，并建议采用复制代码等替代方案。提出并辩论了一种独特的 4-bit 量化 Kernel 概念，该 Kernel 使用 Lookup Tables（查找表）代替 Scale 和 Zero_point，以探讨其在实现中的效率和实用性。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py#L253-L283">hqq/hqq/core/quantize.py at master · mobiusml/hqq</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/optimize.py#L276-L405">hqq/hqq/core/optimize.py at master · mobiusml/hqq</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/pytorch-labs/gpt-fast/blob/main/model.py#L175)">gpt-fast/model.py at main · pytorch-labs/gpt-fast</a>：在少于 1000 行 Python 代码中实现简单高效的 PyTorch 原生 Transformer 文本生成。- pytorch-labs/gpt-fast</li><li><a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/optimize.py#L412">hqq/hqq/core/optimize.py at master · mobiusml/hqq</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/pytorch-labs/gpt-fast/blob/main/model.py#L213">gpt-fast/model.py at main · pytorch-labs/gpt-fast</a>：在少于 1000 行 Python 代码中实现简单高效的 PyTorch 原生 Transformer 文本生成。- pytorch-labs/gpt-fast</li><li><a href="https://github.com/Xilinx/brevitas/pull/937">HQO for scale/zero point by Giuseppe5 · Pull Request #937 · Xilinx/brevitas</a>：未找到描述
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1230434940035006506)** (552 messages🔥🔥🔥): 

- **CUDA Profiling 与优化讨论**：成员们一直在使用 **NVIDIA Nsight Compute** 优化 CUDA 代码以提高 GPU 效率。经过大量改进（包括循环重组和消除冗余计算），实现了显著的加速，将 [CUDA 模型训练循环](https://github.com/karpathy/llm.c/pull/179/files) 在 A4000 GPU 上的耗时从 960ms 降低到 250ms，完整训练迭代耗时降至 77ms，超越了 PyTorch 的耗时。

- **CUDA 与 PyTorch 性能对比**：他们发现，在[进一步优化](https://github.com/karpathy/llm.c/pull/179/files)并集成 Fused Classifier Kernel 后，性能在某些测试中超过了 PyTorch 的指标，使得 PyTorch 现在成为了新的追赶目标。

- **Fused Classifier 改进**：在 CUDA 中实现非 `float4` 版本的 classifier kernel 减少了昂贵的 CUDA 调用次数，同时移除手动 softmax 计算获得了额外的加速，尽管最初比 `float4` 版本慢。

- **CUDA Kernel 效率**：讨论强调了整数除法如何成为 CUDA kernel 中显著的性能瓶颈。通过从 CPU 端传递预计算的值，他们实现了指令减少 25% 和 kernel 加速 1%，尽管整体时间仍受限于等待数据。

- **多 GPU 考量**：对话涉及了多 GPU 训练的潜在方法，倾向于数据并行 (DDP) 而非模型并行，从用于节点内 GPU 通信的单个 allReduce 调用开始，并可能进展到 MPI 或 NCCL 等库。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/tutorials/recipes/compiling_optimizer.html">(beta) 使用 torch.compile 编译优化器 &mdash; PyTorch Tutorials 2.3.0+cu121 文档</a>：未找到描述</li><li><a href="https://godbolt.org/z/3Yscz4Ee7">Compiler Explorer - CUDA C++ (NVCC 12.3.1)</a>: #include &amp;lt;cooperative_groups.h&amp;gt; #include &amp;lt;cooperative_groups/reduce.h&amp;gt; #include &amp;lt;assert.h&amp;gt; #include &amp;lt;math.h&amp;gt; #include &amp;lt;ctype.h&amp;gt; #i...</li><li><a href="https://tenor.com/view/war-dogs-war-dogs-movie-stressed-facepalm-gif-5727928">压力 GIF - War Dogs War Dogs Movie Stressed - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.nvidia.com/en-us/on-demand/session/gtcsiliconvalley2018-s81006/">Volta：架构与性能优化 | NVIDIA On-Demand</a>：本次演讲将回顾 Volta GPU 架构以及优化计算应用性能的相关指导</li><li><a href="https://github.com/karpathy/llm.c/blob/master/dev/cuda/classifier_fused.cu#L327">llm.c/dev/cuda/classifier_fused.cu at master · karpathy/llm.c</a>：使用简单、原生 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号为 karpathy/llm.c 做出贡献。</li><li><a href="https://developer.nvidia.com/nsight-compute">NVIDIA Nsight Compute</a>：用于 CUDA 和 NVIDIA OptiX 的交互式性能分析器。</li><li><a href="https://www.youtube.com/watch?v=dB5Jxwj0PDw)">CUDA 教程 I 应用程序的性能分析与调试</a>：使用 NVIDIA 开发者工具对 CUDA 进行分析、优化和调试。NVIDIA Nsight 工具套件可可视化硬件吞吐量并分析性能指标...</li><li><a href="https://github.com/karpathy/llm.c/compare/master...Chillee:llm.c:master">比较 karpathy:master...Chillee:master · karpathy/llm.c</a>：使用简单、原生 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号为 karpathy/llm.c 做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/pull/163">通过在 backward 过程中跨层重用相同缓冲区来节省内存 by ngc92 · Pull Request #163 · karpathy/llm.c</a>：在 backward 过程中跨层重用内存缓冲区</li><li><a href="https://github.com/karpathy/llm.c/pull/179/files#diff-a9575fa20f8ecc58bce4deb2e6c602b0421dbbdf0c671e46967359da91e3868cR634-R682">迈向更好的 backward attention kernel by ngc92 · Pull Request #179 · karpathy/llm.c</a>：回到设计阶段，因为我认为其他 kernel 陷入了局部最优，或者至少循环组织的方式使得进一步优化变得非常困难。我...</li><li><a href="https://github.com/karpathy/llm.c/pull/150">优化版本的 fused classifier + 错误修复(?) by ademeure · Pull Request #150 · karpathy/llm.c</a>：这是来自 #117 的酷炫新 kernel 的更快版本（仍仅限 /dev/cuda/）。最大的区别在于它针对每个 1024 宽度的 block 处理一行进行了优化，而不是每个 32 宽度的 warp...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[massively-parallel-crew](https://discord.com/channels/1189498204333543425/1229286073104994344/1230417873663561768)** (10 条消息🔥):

- **Panel 与 CUDA Mode 冲突**：一位嘉宾意识到他们在即将到来的 **CUDA Mode event** 期间有双重预订，正在请求社区负责该特定环节的 **recording responsibilities**（录制工作）。
- **活动权限委派**：已设置特定权限，允许某些成员创建/编辑活动并在 **stages**（舞台）上 **mute people**（禁言他人），以确保未来活动的顺利进行。
- **录制设置讨论**：由于旧款 iPad 可能存在内存问题，人们对即将到来的活动的录制设置表示担忧，从而引发了对替代录制方案的讨论。
- **建议的录制工作流**：成员们讨论了一个建议的录制工作流，包括使用原生 MacOS 屏幕录制、名为 BlackHole 的音频服务，以及如果演讲者也能在本地录制，则进行潜在的后期制作编辑。
- **EventType 更新**：宣布 4 月 26 日将有一场新的演讲，这意味着同一周将举办两场活动以容纳额外的内容。
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1230401351599525988)** (489 条消息🔥🔥🔥): 

- **Llama 3 发布引发普遍兴奋和测试**：用户讨论了 **Meta Llama 3** 的发布，探索了其功能、tokenizer 效率以及在 8k context window 之外进行扩展的潜在需求。一些人正在等待更长 context 的版本，但对 Llama 3 的 8k tokens 比 Llama 2 更高效表示赞赏。

- **对 Llama 3 性能和训练的复杂情绪**：虽然一些人庆祝 Llama 3 的强大实力，特别是在医疗和中文评估方面，但另一些人感到难过，因为新发布的模型相当于他们数月的个人微调工作。人们对 finetuning 期间较高的初始 loss 率和缓慢的 loss 改进表示担忧。

- **Axolotl 与 Llama 3 的技术博弈**：用户报告了将 **qlora adapter** 合并到 Llama 3 时的问题，在 finetuning 过程中遇到了 tokenizer 加载错误和意外的关键字参数。建议包括将 transformer 版本更改为 4.37 以规避某些问题。

- **AMD GPU 用户寻求支持和选择**：一位用户为使用 **ROCm** 的用户详细介绍了安装指南，指出 memory-efficient attention 可以提供类似于 Nvidia 用户的 Flash Attention 的好处。人们有兴趣测试可能对 AMD GPU 更友好的替代方案。

- **ChatML 和 Tokenizer 配置带来的挑战**：用户讨论了 **ChatML** 的挑战以及如何正确映射 special tokens，分享了对 tokenizer 配置的尝试和修复。人们渴望在 Llama 3 中成功实现 ChatML，同时也对现有模板适配可能引起的问题感到沮丧。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://huggingface.co/hfl/chinese-llama-2-13b-16k">hfl/chinese-llama-2-13b-16k · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/dreamgen/opus-v1.2-llama-3-8b">dreamgen/opus-v1.2-llama-3-8b · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/Ahmad_Al_Dahle/status/1781050097868103726?t=ow7ldzKTWHjRBW33sxfc_A&s=09">来自 Ahmad Al-Dahle (@Ahmad_Al_Dahle) 的推文</a>: @mattshumer_ 我们会发布更长的版本。此外，与 Llama 2 相比，使用新的 Tokenizer 后，Context Window 应该会稍长一些。</li><li><a href="https://fxtwitter.com/benjamin_warner/status/1781095499145134263">来自 Benjamin Warner (@benjamin_warner) 的推文</a>: 如果使用 Hugging Face 微调 Llama 3，请使用 Transformers 4.37 或 4.40。4.38 和 4.39 版本中的 Llama 和 Gemma 没有使用 PyTorch 的 Flash Attention 2 内核，会导致内存占用过高。4.40 版本使用了 FA2 ...</li><li><a href="https://x.com/teortaxesTex/status/1781063292795883943">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>: 来自该线程：Llama-3 8b 具有至少 32k 的近乎完美的大海捞针（Needle Retrieval）检索能力（RoPE theta 为 4）</li><li><a href="https://tenor.com/view/bonk-gif-26414884">Bonk GIF - Bonk - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://azuremarketplace.microsoft.com/en-us/marketplace/apps/metagenai.meta-llama-3-8b-chat-offer?tab=Overview">Microsoft Azure Marketplace</a>: 未找到描述</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1536/files">由 monk1337 添加 Llama-3 QLoRA · Pull Request #1536 · OpenAccess-AI-Collective/axolotl</a>: 添加了 Llama-3 QLoRA，经测试可用。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/requirements.txt">axolotl/requirements.txt (main 分支) · OpenAccess-AI-Collective/axolotl</a>: 尽管提出 Axolotl 问题。通过在 GitHub 上创建账号来为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/0e8f3409451442950f2debbe28735198361c9786/src/axolotl/monkeypatch/llama_attn_hijack_flash.py#L30">axolotl/src/axolotl/monkeypatch/llama_attn_hijack_flash.py (版本 0e8f340) · OpenAccess-AI-Collective/axolotl</a>: 尽管提出 Axolotl 问题。通过在 GitHub 上创建账号来为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/0e8f3409451442950f2debbe28735198361c9786/setup.py#L36">axolotl/setup.py (版本 0e8f340) · OpenAccess-AI-Collective/axolotl</a>: 尽管提出 Axolotl 问题。通过在 GitHub 上创建账号来为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=bc6uFV9CJGg">Mark Zuckerberg - Llama 3, 100 亿美元模型, 凯撒·奥古斯都, 以及 1 GW 数据中心</a>: 扎克伯格关于：- Llama 3 - 迈向 AGI 的开源 - 定制芯片、合成数据以及扩展时的能源限制 - 凯撒·奥古斯都、智能爆炸、生物...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1549">草稿：由 mhenrichsen 更新 models.py 中的 Tokenizer 覆盖处理 · Pull Request #1549 · OpenAccess-AI-Collective/axolotl</a>: 示例：tokenizer_overrides: - 28006: <|im_start|> - 28007: <|im_end|> 描述：此 PR 增强了我们在 models.py 文件中处理 Tokenizer 覆盖的方式。...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1547">特性：由 NanoCode012 添加 cohere (commandr) · Pull Request #1547 · OpenAccess-AI-Collective/axolotl</a>: 描述 动机与背景 如何测试？ 未测试！ 截图（如果适用） 变更类型 社交账号（可选）</li><li><a href="https://github.com/Ope">ope - 概览</a>: ope 有 11 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/OpenNLPLab/lightning-attention/tree/main">GitHub - OpenNLPLab/lightning-attention: Lightning Attention-2: 处理大语言模型中无限序列长度的免费午餐</a>: Lightning Attention-2: 处理大语言模型中无限序列长度的免费午餐 - OpenNLPLab/lightning-attention</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/0e8f3409451442950f2debbe28735198361c9786/src/axolotl/utils/trainer.py#L272">axolotl/src/axolotl/utils/trainer.py (版本 0e8f340) · OpenAccess-AI-Collective/axolotl</a>: 尽管提出 Axolotl 问题。通过在 GitHub 上创建账号来为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/issues/1519">考虑将 Memory Efficient Attention 作为 AMD 用户 Flash Attention 的“替代方案”。 · Issue #1519 · OpenAccess-AI-Collective/axolotl</a>: ⚠️ 请检查此功能请求是否已被提议...</li>

fore. 我在 Discussions 中搜索了之前的 Ideas，没有发现类似的特性请求。我在 Issues 中搜索了之前的记录，也没有发现...</li><li><a href="https://github.com/lucidrains/memory-efficient-attention-pytorch">GitHub - lucidrains/memory-efficient-attention-pytorch: Implementation of a memory efficient multi-head attention as proposed in the paper, &quot;Self-attention Does Not Need O(n²) Memory&quot;</a>: 论文《Self-attention Does Not Need O(n²) Memory》中提出的内存高效多头注意力的实现 - lucidrains/memory-efficient-attention-pytorch</li><li><a href="https://github.com/xzuyn/axolotl/commit/6488a6b6f0d195612d491ece2f9a049080e8d9">Add experimental install guide for ROCm · xzuyn/axolotl@6488a6b</a>: 未找到描述</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1536">Adding Llama-3 qlora by monk1337 · Pull Request #1536 · OpenAccess-AI-Collective/axolotl</a>: 添加 Llama-3 qlora，经测试可用。</li><li><a href="https://github.com/xzuyn/axolotl/">GitHub - xzuyn/axolotl: Go ahead and axolotl questions</a>: 尽管提问（Go ahead and axolotl questions）。通过在 GitHub 上创建账号来为 xzuyn/axolotl 的开发做出贡献。</li><li><a href="https://huggingface.co/datasets/xorsuyash/raft_datasetp1">xorsuyash/raft_datasetp1 · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/sustcsonglin/flash-linear-attention/tree/main/fla/layers">flash-linear-attention/fla/layers at main · sustcsonglin/flash-linear-attention</a>: 在 Pytorch 和 Triton 中对最先进线性注意力模型的高效实现 - sustcsonglin/flash-linear-attention</li><li><a href="https://github.com/huggingface/transformers/pull/24653">Llama/GPTNeoX: add RoPE scaling  by gante · Pull Request #24653 · huggingface/transformers</a>: 这个 PR 做了什么？这是一个用于讨论的实验性 PR，以便我们决定是否添加此模式。背景：在过去的一周里，关于缩放 RoPE (Rot...
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1230527592545648640)** (13 messages🔥): 

- **更少的截断，更好的性能**：AWS 的新 packing 算法表现出显著改进，在阅读理解上*相对提升了 +4.7%*，并将闭域幻觉（closed domain hallucination）减少了高达 *58.3%*。更多详情和论文请参见 [Fewer Truncations Improve Language Modeling](https://arxiv.org/abs/2404.10830)。
- **LLaMA-3 继承了 LLaMA-2 的架构**：成员间的讨论确认 **LLaMA-3** 使用了与其前身相同的架构，尽管最初根据数据缩放论文预期会有所改变。
- **LLaMA-3 的 Tokenizer 调整**：有人指出 LLaMA-3 拥有更大的 Tokenizer，但保持了与 LLaMA-2 相同的架构。目前正在设置训练运行以探索新模型的能力。
- **调查 PAD Token 修复**：关于手动设置 PAD Token 是否能解决某些问题的对话促使了对 `AutoTokenizer` 的尝试。成员们正在测试这些 Tokenizer 配置是否会产生影响。
- **在 Pull Request 上测试贡献**：有人请求空闲计算资源来测试一个更新 Tokenizer Overrides 处理的 Draft PR，链接如下：[Update Tokenizer Overrides Handling in models.py](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1549)。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/arankomatsuzaki/status/1780778186348843253?s=46&t=hIokEbug9Pr72tQFuXVULA">Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>: AWS 发布《Fewer Truncations Improve Language Modeling》。他们的 packing 算法实现了卓越的性能（例如，在阅读理解上相对提升 +4.7%），并有效减少了闭域幻觉...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1549">Draft: Update Tokenizer Overrides Handling in models.py by mhenrichsen · Pull Request #1549 · OpenAccess-AI-Collective/axolotl</a>: 示例：tokenizer_overrides: - 28006: &lt;|im_start|&gt; - 28007: &lt;|im_end|&gt; 描述：此 PR 对我们在 models.py 文件中处理 tokenizer overrides 的方式进行了增强。...</li><li><a href="https://x.com/danielhanchen/status/1781012164893118471?s=46&t=hIokEbug9Pr72tQFuXVULA">Daniel Han (@danielhanchen) 的推文</a>: Llama-3 的其他一些怪癖：1. 自从使用 tiktoken 以来，数字被拆分为 1、2、3 位数字（Llama 单数字拆分），即 1111111 被从左到右拆分为 111_111_1。2. 没有 unk_token？正尝试获取 @UnslothAI...
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1230516829051424838)** (43 messages🔥):

- **Mistral 微调的解冻秘籍**：用户正在讨论如 7b **Mistral** 等模型的微调。一位用户提到，除非在过程中解冻 **lmhead and embed layers**，否则模型会崩溃。

- **Llama-3 微调故障排除**：几位用户正在进行 **Llama-3** 微调过程的故障排除。报告了诸如由于缺少 `pad_token` 导致的 `ValueError` 以及文件写入错误等问题，一位用户建议这些错误可能与磁盘空间不足有关。

- **不同 Llama 模型的 Tokenization 谜题**：对话显示 **tokenizer configuration** 在 **Llama-2** 和 **Llama-3** 之间发生了变化，后者的 padding token 设置为 `<|end_of_text|>`，这引起了用户的一些困惑。

- **指令数据集：全量微调还是不全量微调？**：一位成员正在考虑是对指令数据集进行 fulltune（全量微调）还是仅使用 completions。随后的讨论涉及了 base model 或 instruct model 是否更适合 fulltune。

- **微调的挫折与成功**：关于 **Llama models** 微调特性的持续交流，参与者遇到了与内存问题无关的错误，并思考根本原因是否是存储空间不足或其他原因。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/1230809963832807424)** (3 messages): 

- **经历缓慢的启动过程**：一位用户表示在 **runpod** 中启动 pod 的过程耗时异常长，怀疑服务可能宕机。
- **对 Runpod 可靠性的幽默调侃**：另一位用户开了一个幽默的玩笑，暗示 **runpod** 经常宕机。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1230588800200085625)** (14 messages🔥): 

- **PAD Tokens 配置困惑**：一位成员询问如何在 YAML 配置中设置 PAD token，机器人详细说明了通常应包含在 `tokens` 部分下。
- **Token 替换教程**：用户寻求关于在 tokenizer 中用另一个 token 替换 token 的指导；机器人提供了一个使用 Hugging Face 的 `transformers` 库的逐步示例，涉及 `add_tokens` 和手动词汇表调整。
- **YAML Token 替换详情**：同一位成员进一步询问了直接在 YAML 配置文件中替换 token 的情况；机器人澄清了向配置文件添加新 token 的方法，以及如何通过代码调整使用这些新 token 替换原始 token。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=283dc0b2-eb24-4f4e-8b7d-1c24e9285c3d)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>：更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=484ff2b8-6849-4c46-a388-8e244cdca92d)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>：更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=d1ef6577-52b8-44cd-8588-c33724be6c8e)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>：更快地理解代码。
</li>
</ul>

</div>
  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1230404753897291788)** (471 messages🔥🔥🔥): 

- **对 SD3 本地权重的期待**：用户们热切期待 **Stable Diffusion 3** (SD3) 本地权重的发布，聊天中多次提到预计访问日期为 5 月 10 日。

- **Stable Diffusion API 过滤担忧**：有关于 SD3 API 可能会对判定为“敏感 (edgy)”的提示词返回模糊图像的讨论，引发了关于生成内容与本地版本相比如何被过滤或审查的对话。

- **图形硬件决策**：一位考虑为 AI 工作选择 GPU 的用户得到了关于 **RTX 3090** 与即将推出的或更昂贵的型号（如 **RTX 4080** 和 4090）相比的价值和效率建议，考虑了 VRAM、速度和价格的平衡。

- **内容生成技术探索**：成员们分享了技巧并寻求使用各种模型生成特定类型内容的指导，包括为半脸图像编写提示词以及管理生成图像的细节。

- **分享教程与协助**：一位用户提供了一个 [Comfy UI 教程视频](https://youtu.be/j3xHNmEWWCI) 链接。针对各种问题（如处理 img2img 中的 IndexError 以及讨论生成图像中隐藏水印的可见性和移除）寻求并提供了帮助。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://apply.workable.com/palazzo-inc-1/j/877AE4A35A/">Stable Diffusion Engineer - Palazzo, Inc.</a>: 关于我们：Palazzo 是一家充满活力且创新的科技公司，致力于突破室内设计全球 AI 的界限。我们正在寻找一名技术精湛的 Stable Diffusion 工程师加入我们的团队...</li><li><a href="https://app.wordware.ai/r/b137b2f5-a971-420c-a594-4f6350c24fa5">Wordware - Compare prompts</a>: 使用输入提示词和优化后的提示词运行 Stable Diffusion 3</li><li><a href="https://beta.dreamstudio.ai/generate">DreamStudio</a>: 未找到描述</li><li><a href="https://clipdrop.co/">Create stunning visuals in seconds with AI.</a>: 几秒钟内用 AI 创建惊人的视觉效果。移除背景、清理图片、放大、Stable Diffusion 等等……</li><li><a href="https://stability.ai/contact">Contact Us &mdash; Stability AI</a>: 未找到描述</li><li><a href="https://youtu.be/j3xHNmEWWCI">⚡Harness Lightning-Fast Detail with ComfyUI PERTURBED + 🔮 Mask Wizardry &amp; Fashion Secrets! 🤩</a>: -- Discord - https://discord.gg/KJXRzkBM -- 准备好将你的细节处理提升到新的水平！🚀 在这个令人惊叹的教程中，你将发现令人难以置信的……</li><li><a href="https://www.youtube.com/watch?v=ResSOxQBUSM">RTX 4080 vs RTX 3090 vs RTX 4080 SUPER vs RTX 3090 TI - Test in 20 Games</a>: RTX 4080 vs RTX 3090 vs RTX 4080 SUPER vs RTX 3090 TI - 20 款游戏测试 1080p, 1440p, 2160p, 2k, 4k⏩GPUs &amp; Amazon US⏪ (联盟链接)- RTX 4080 16GB: http...</li><li><a href="https://github.com/Priyansxu/vega">GitHub - Priyansxu/vega</a>: 通过在 GitHub 上创建账户来为 Priyansxu/vega 的开发做出贡献。</li><li><a href="https://github.com/codaloc/sdwebui-ux-forge-fusion">GitHub - codaloc/sdwebui-ux-forge-fusion: Combining the aesthetic interface and user-centric design of the UI-UX fork with the unparalleled optimizations and speed of the Forge fork.</a>: 将 UI-UX 分支的美学界面和以用户为中心的设计，与 Forge 分支无与伦比的优化和速度相结合。- codaloc/sdwebui-ux-forge-fusion</li><li><a href="https://www.youtube.com/watch?v=XfkgiXaaCY4&list=LL&index=2">Dwayne Loses His Patience 😳 #ai #aiart #chatgpt</a>: 未找到描述</li><li><a href="https://www.youtube.com/shorts/ASkd9Oxk1Eo">1 Mad Dance of the Presidents (ai) Joe Biden 🤣😂😎✅ #stopworking #joebiden #donaldtrump #funny #usa</a>: 🎉 🤣🤣🤣🤣 准备好在 "Funny Viral" 频道最新的 "搞笑动物合集" 中捧腹大笑吧！🤣 这些可爱又调皮的……
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1230521507088826428)** (229 messages🔥🔥): 

- **探索 Discord 总结**：一位成员询问如何总结一个关于系统工程的高质量、高密度 Discord 服务器。他们讨论了可能使用 **Claude 3 Haiku** 从 Discord 中结构化数据，然后使用 AI 新闻机器人进行总结。还分享了该服务器的邀请链接：[系统工程 Discord 邀请](https://discord.gg/NBFgzps4)。

- **Meta 发布 Llama 3**：**Meta Llama 3** 的发布引发了详细讨论其规格的对话，例如 **8B 和 70B** 模型尺寸展示了 **state-of-the-art (SOTA) 性能**，并提到了即将推出的 **400B+ 参数模型**。讨论内容包括**改进**、**指令微调 (instruction tuning)**、**安全工具**和**训练细节**。社区成员还辩论了 **M4 电脑**运行大型模型的能力，并剖析了 **Llama 3 70B 模型**的潜力，根据 Lmsys 评分将其与 **GPT-4** 进行比较，其中一位成员强调量化技术的进步可能会增强性能。

- **Llama 3 的性能与可能性**：社区成员回顾了新的 **Llama 3**，并将其与 **GPT-4turbo** 等现有模型进行了对比，分享了印象，并建议查看 [meta.ai](https://meta.ai) 以获取准确性。对话深入探讨了推理速度等话题，一位成员强调了 **Groq Cloud** 上极快的首字节时间 (time-to-first-byte) 性能，并讨论了可能利用 Llama 3 而非 GPT-4-Turbo 的语音对语音 (voice 2 voice) AI 使用场景。

- **解决 Mac 推理能力问题**：关于在 Mac 上运行大型模型的激烈交流展开，成员们讨论了他们的经验以及对未来 **Mac M4 芯片性能**的期望。焦点集中在个人电脑上运行 **Llama 3 70B 和 400B** 等模型的实用性，并提出了利用本地 Linux 机器结合 Mac 以提高效率的变通方案。

- **AI 爱好者寻求 LLM 项目样板代码 (Boilerplate)**：一位社区成员正在寻找用于启动 LLM 项目的模板或 "cookiecutter"，以减少重复工作；另一位成员分享了 [litellm](https://litellm.vercel.app/)，这是一个可以调用超过 100 个 LLM 且具有统一输入/输出格式及其他功能的潜在工具。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/awnihannun/status/1781020285107675502?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Awni Hannun (@awnihannun) 的推文</a>：@soumithchintala @lvdmaaten 🤣🤣🤣 刚刚运行了 @Prince_Canuma 量化的 8B 版本。在 M2 Ultra 上表现非常出色（而且很快 😉）：</li><li><a href="https://x.com/theseamouse/status/1781134831914508720?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Hassan Hayat 🔥 (@TheSeaMouse) 的推文</a>：我仍然对此感到震惊。它是如何提升这么多的？我是说，看看 8B 对比旧的 70B</li><li><a href="https://stanford.zoom.us/j/99922151759?pwd=dW5CcUtVYkNybGZGY0hMWUZtVkZBZz09">加入我们的 Cloud HD 视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有简单、可靠的云平台，适用于移动端、桌面端和会议室系统的视频和音频会议、聊天及网络研讨会。Zoom ...</li><li><a href="https://tinygrad.org/#tinybox">tinygrad：一个简单且强大的神经网络框架</a>：未找到描述</li><li><a href="https://x.com/FanaHOVA/status/1780996533661671683">来自 Alessio Fanelli (@FanaHOVA) 的推文</a>：🦙 所有 Llama3 发布详情与亮点：8B 和 70B 尺寸：在大多数基准测试中，Instruct 和 Pre-trained 版本均达到 SOTA 性能。目前正在训练一个 400B+ 参数的模型，即将发布...</li><li><a href="https://x.com/hive_echo/status/1781220509147095059">来自 echo.hive (@hive_echo) 的推文</a>：测试 Llama-3 8B 和 70B。这个简单的测试结果向我证明，更小模型配合更多数据可以成为出色的低端推理模型，而更大模型配合更多数据则能成为出色的高端推理模型...</li><li><a href="https://discord.gg/NBFgzps4">加入 Systems Engineering Professionals Discord 服务器！</a>：在 Discord 上查看 Systems Engineering Professionals 社区 —— 与其他 1666 名成员一起交流，享受免费的语音和文字聊天。</li><li><a href="https://www.macrumors.com/2024/04/11/m4-ai-chips-late-2024/">Mac 将于 2024 年底开始搭载专注于 AI 的 M4 芯片</a>：据 Bloomberg 的 Mark Gurman 报道，苹果将于 2024 年底开始使用 M4 芯片更新其 Mac 产品线。M4 芯片将专注于...</li><li><a href="https://www.interconnects.ai/p/llama-3-and-scaling-open-llms">Llama 3：将开源 LLM 扩展至 AGI</a>：Llama 3 表明，在不久的将来，扩展（scaling）不会成为开源 LLM 进步的限制。</li><li><a href="https://www.browserless.io/">Browserless - 排名第一的 Web 自动化和 Headless 浏览器自动化工具</a>：免费试用 Browserless，最好的 Web 自动化工具之一。轻松实现网页抓取、PDF 生成和 Headless 浏览器自动化。</li><li><a href="https://www.youtube.co">未找到标题</a>：未找到描述</li><li><a href="https://x.com/kwindla/status/1781408311021367761">来自 kwindla (@kwindla) 的推文</a>：哇。Llama-3 70B 在 @GroqInc 上的首字节时间（time-to-first-byte）非常快 —— 快到不足 100ms。</li><li><a href="https://litellm.vercel.app/">LiteLLM - 开始使用 | liteLLM</a>：https://github.com/BerriAI/litellm</li><li><a href="https://x.com/teknium1/status/1781328542367883765?s=46&t=90xQ8sGy63D">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：好了伙计们，我们现在家里也有 GPT-4 了</li><li><a href="https://x.com/teknium1/status/1781328542367883765?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：好了伙计们，我们现在家里也有 GPT-4 了</li><li><a href="https://x.com/hu_yifei/status/1781105968207507838?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Yifei Hu (@hu_yifei) 的推文</a>：使用 vLLM 运行 Llama 3 70B（完整 8k 上下文）：大约需要 180GB VRAM</li><li><a href="https://www.grey-wing.com/product/ocean-oracle">您掌握租船数据的 Copilot</a>：做出更好的决策，并利用 Generative AI 赋能您的租船团队做出更好的决策。</li><li><a href="https://llama.meta.com/llama3/">Meta Llama 3</a>：使用 Meta Llama 3 构建 AI 的未来。现在提供 8B 和 70B 的 Pre-trained 和 Instruction-tuned 版本，以支持广泛的应用。</li><li><a href="https://x.com/togethercompute/status/1781004579817349266?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Together AI (@togethercompute) 的推文</a>：我们很高兴能成为 Meta Llama 3 的发布合作伙伴。现在体验 Llama 3，Llama 3 8B 每秒可达 350 tokens，Llama 3 70B 每秒可达 150 tokens，均以全 FP16 精度运行...</li><li><a href="https://docs.google.com/presentation/d/1quMyI4BAx4rvcDfk8jjv063bmHg4RxZd9mhQloXpMn0/edit">[2024年4月18日] 对齐开源语言模型</a>：对齐开源语言模型 Nathan Lambert || Allen Institute for AI || @natolambert Stanford CS25: Transformers United V4</li><li><a href="https://x.com/karpathy/status/1781047292486914189?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Andrej Karpathy (@karpathy) 的推文</a>：模型卡（model card）中也有一些更有趣的信息：https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md 注意 Llama 3 8B 实际上已经处于 Llama 2 70B 的水平，具体取决于...</li><li><a href="https://ai.meta.com/blog/meta-⁠">Meta AI 博客</a>

llama-3/">未找到标题</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=bc6uFV9CJGg&ab_channel=DwarkeshPatel">Mark Zuckerberg - Llama 3, $10B Models, Caesar Augustus, &amp; 1 GW Datacenters</a>: 小扎谈论：- Llama 3 - 朝向 AGI 的开源 - 定制芯片、合成数据以及 scaling 的能源限制 - 凯撒·奥古斯都、智能爆炸、生物...</li><li><a href="https://www.firecrawl.dev/">FireCrawl</a>: 将任何网站转换为 LLM 就绪的数据。
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1230959116919509062)** (3 messages): 

- **Latent Space Pod 发布新剧集**: Latent Space Discord 社区分享了由 <@199392275124453376> 参与的新播客剧集。随着 Twitter 公告链接的提供，大家兴奋不已：[收听新剧集](https://twitter.com/latentspacepod/status/1781400226793673137)。
- **播客热情**: 社区成员 **mitchellbwright** 对由 **Jason Liu** 参与的最新播客内容表示期待。
  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1230880331482136588)** (66 messages🔥🔥): 

- **LLM 论文俱乐部 - 生成式预训练**: 论文俱乐部讨论了论文 "Improving Language Understanding by Generative Pre-Training"，强调了 tokenizer 和 embedding 的重要性，并指出 *“embedding 本身就是 NN”*。讨论注意到，与 embedding 不同，*“tokenizer 并不一定需要学习”*。
- **会议录制中**: 亚洲论文俱乐部的会议确认已录制，并计划上传到 YouTube，证据是 *“正在监控 OBS 流，如果一切正常，稍后将上传到 YouTube : )”*。
- **ULMFiT 澄清**: 讨论澄清了 ULMFiT 使用 LSTM 架构，并在 T5 论文中被引用。
- **PPO 辅助目标探讨**: 对话涉及 Proximal Policy Optimization (PPO) 算法，讨论了它是否具有类似 Kullback–Leibler (KL) 散度的辅助目标，一名成员确认道：*“是的，如果我没记错的话”*。
- **Prompt 时代**: 提到了 prompt engineering 的开端，并引用了 *“scale is all you need”* 的名言，暗示了 scale 在模型性能中日益增长的重要性。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.tomshardware.com/tech-industry/artificial-intelligence/tinybox-packs-a-punch-with-six-of-amds-fastest-gaming-gpus-repurposed-for-ai-george-hotzs-new-box-uses-radeon-7900-xtx-and-retails-for-dollar15k-now-in-production">TinyBox 搭载六块 AMD 最快游戏 GPU 并重新用于 AI，性能强劲 —— 新机箱使用 Radeon 7900 XTX，零售价 1.5 万美元，现已投入生产</a>: 初创公司希望利用 Radeon RX 7900 XTX 提供高性能 AI 计算。</li><li><a href="https://openai.com/research/scaling-laws-for-neural-language-models">神经语言模型的 Scaling laws</a>: 未找到描述</li><li><a href="https://paperswithcode.com/dataset/mrpc">Papers with Code - MRPC 数据集</a>: Microsoft Research Paraphrase Corpus (MRPC) 是一个由 5,801 个从新闻文章中收集的句子对组成的语料库。每个句子对都由人工标注是否为释义（paraphrase）。
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1230971054881505292)** (69 messages🔥🔥): 

- **Zoom 还是 Discord？悬而未决**: 成员们讨论了本周会议是继续使用 Discord 还是切换到 Zoom。
- **分享 AI 见解**: 分享了一个关于 **LLM Evaluation** 的 Google Slides 演示文稿链接供小组审阅，表明了对评估大语言模型的关注。
- **对噪音的好奇**: 一名成员报告在会议期间听到不明嗡嗡声，重新加入聊天后问题解决。
- **摘要策略探讨**: 提供了 Eugene Yan 文章的链接，探讨了抽象式摘要及其通用性，以及机器学习中评估机制的复杂性。
- **正在进行的 AI 评估讨论**: 与大语言模型 (LLM) 评估、检索过程有效性以及开发分支策略转变相关的对话，暗示了该小组在提升 AI 性能和可靠性方面的努力。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://eugeneyan.com/writing/abstractive/">Evaluation & Hallucination Detection for Abstractive Summaries</a>：摘要式总结的评估与幻觉检测：基于参考、上下文和偏好的指标，自一致性，以及捕捉幻觉。</li><li><a href="https://eugeneyan.com/writing/evals/">LLM Task-Specific Evals that Do & Don't Work</a>：有效与无效的 LLM 特定任务评估：用于分类、摘要、翻译、版权内容复现（regurgitation）和毒性的评估。</li><li><a href="https://docs.google.com/presentation/d/14EE2j6ii4PEA0Y-wUg80weC3eJ-qx2q41uUAEqytG28/edit?usp=sharing">LLM Evaluation</a>：评估基于 LLM 的系统。Mama mówiła, będzie okazja Na majku, no to rozjeb, no to rozjeb (Rozjeb) Alan van Arden 2024年4月19日 Latent Space</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0">AI In Action: Weekly Jam Sessions</a>：2024 主题、日期、主持人、资源、@dropdown、@ GenAI 的 UI/UX 模式，1/26/2024，nuvic，&lt;a href=&quot;https://maggieappleton.com/squish-structure&quot;&gt;https://maggieappleton.com/squish-stru...
</li>
</ul>

</div>
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1230459573844840480)** (115 messages🔥🔥): 

- **LLM 输入序列的新打包方法**：一篇新论文介绍了 **Best-fit Packing**，这是一种通过优化文档到训练序列的打包方式，来消除 **large language model** 训练中过度截断的方法。该方法声称具有更好的性能，详见 [arXiv 上的论文](https://arxiv.org/abs/2404.10830)。

- **LLaMA 3 Tokenizer 争议**：关于 LLaMA 3 使用的“奇怪” Tokenizer 的讨论，强调了它的奇特之处，并澄清了一些所谓的 Token 实际上是来自 HF 仓库的 merge vocabs。

- **寻求优化器的稳定性**：关于 AdamW 优化器稳定性的对话，涵盖了降低学习率、从 0 开始预热（warmup）、在矩阵计算后记录 std_mean(activation) 等技术，以及尝试使用 **StableAdamW** 或 LaProp 以获得更好训练性能的建议。

- **调试 Whisper 架构的不稳定性**：用户深入探讨了通过修改 Adam 优化器的超参数、增加 batch sizes 以及检查梯度范数（gradient norms）和权重尺度（weight scale）来解决 Whisper 架构训练不稳定的策略。

- **对 Transformer Attention 的独特建议**：讨论了一项关于具有“可学习窗口”的 **Transformer attention mechanism** 的提案，允许模型关注文本中较远位置的相关 Token，旨在将复杂度从 O(n^2) 降低到 O(n)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/giffmana/status/1692641748445438301>)">Lucas Beyer (bl16) (@giffmana) 的推文</a>：最后有两个小的额外技巧：左图：如果你的 loss 飙升，尝试将 Adam/AdaFactor 的 beta2 降低到 0.95（并非首创，但很少分享）；右图：当你的模型的一部分是预训练的，但...</li><li><a href="https://arxiv.org/abs/2404.10830">Fewer Truncations Improve Language Modeling</a>：在大语言模型训练中，输入文档通常被拼接在一起，然后分割成等长的序列以避免 padding tokens。尽管这种方式效率很高，但拼接过程...</li><li><a href="https://github.com/EleutherAI/aria-amt/blob/0394a05aa57e5d4f7b059abbfed3a028732b243a/amt/train.py#L330">EleutherAI/aria-amt 中的 aria-amt/amt/train.py</a>：高效且稳健的 seq-to-seq 自动钢琴转谱实现。 - EleutherAI/aria-amt
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1230404685018562561)** (149 messages🔥🔥): 

- **遭遇 Softmax 瓶颈**：讨论指出了较小语言模型中的饱和现象，引用了一篇[新论文](https://arxiv.org/abs/2404.07647)，该论文将饱和与 Softmax 瓶颈现象联系起来。它强调隐藏层维度小于 1000 的模型可能会因为模型大小与目标概率分布复杂度之间的不匹配而陷入困境。

- **有限数据下的多语言对齐**：成员们探讨了一篇关于在有限多语言数据下，进行零样本（zero-shot）跨语言语境训练对齐模型的论文（[摘要在此](https://arxiv.org/abs/2404.12318)）。所描述的方法显示，即使使用在不同语言上训练的奖励模型，对齐后的模型通常也更受青睐。

- **探索 TC0 与神经网络的表达能力**：有一场关于 TC0 复杂度类及其与 CNN 和 Transformer 等神经网络关系的讨论。对话集中在各种网络是否具有相似水平的表达能力，聊天中未达成定论。

- **Meta 不太可能共享 Llama 3 数据源**：针对要求 Meta 提供 Llama 3 数据集来源文档的请求，用户指出法律复杂性以及 Meta 不太可能屈服于社区请愿。共识似乎是法律障碍将阻止此类披露。

- **Autograd 计算中潜在的死循环**：一位用户报告了在拥有 80GB RAM 的 GPU 上使用 PyTorch 对复杂模型进行反向传播时遇到的奇特问题，引发了关于 Autograd 计算和内存管理的疑问。讨论结论认为，鉴于 Autograd 的设计旨在防止对已有梯度的参数进行梯度计算，死循环是不太可能的。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.12253">Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing</a>：尽管 Large Language Models (LLMs) 在各种任务上表现出惊人的能力，但在涉及复杂推理和规划的场景中仍然面临挑战。最近的工作提出了先进的...</li><li><a href="https://arxiv.org/abs/2404.10282">Tripod: Three Complementary Inductive Biases for Disentangled Representation Learning</a>：归纳偏置在解耦表示学习中对于缩小未指定解集至关重要。在这项工作中，我们考虑为神经网络自动编码器赋予三种选择性的...</li><li><a href="https://arxiv.org/abs/2402.08846">An Embarrassingly Simple Approach for LLM with Strong ASR Capacity</a>：在本文中，我们专注于利用语音基础编码器和大型语言模型解决语音处理领域最重要的任务之一，即自动语音识别 (ASR)...</li><li><a href="https://arxiv.org/abs/2404.09894v2">Glitch Tokens in Large Language Models: Categorization Taxonomy and Effective Detection</a>：随着 Large Language Models (LLMs) 在各个领域的应用不断扩大，全面调查其不可预见的行为及其后果变得至关重要。在这项研究中...</li><li><a href="https://arxiv.org/abs/2404.12318">Reuse Your Rewards: Reward Model Transfer for Zero-Shot Cross-Lingual Alignment</a>：基于人类标注的偏好数据对语言模型 (LMs) 进行对齐是获得实用且高性能的基于 LM 系统的重要步骤。然而，多语言人类偏好数据难以...</li><li><a href="https://arxiv.org/abs/2404.07982">Language Imbalance Can Boost Cross-lingual Generalisation</a>：多语言性对于将语言建模的最新进展扩展到不同的语言社区至关重要。为了在代表多种语言的同时保持高性能，多语言模型...</li><li><a href="https://arxiv.org/abs/2404.07647">Why do small language models underperform? Studying Language Model Saturation via the Softmax Bottleneck</a>：语言建模的最新进展在于在极其庞大的网络挖掘文本语料库上预训练高度参数化的神经网络。在实践中，此类模型的训练和推理可能成本高昂...</li><li><a href="https://huggingface.co/datasets/rajpurkar/squad_v2/discussions/9">rajpurkar/squad_v2 · Error in train split, question containing 25651 characters!</a>：未找到描述</li><li><a href="https://github.com/meta-llama/llama3/issues/39#issuecomment-2065718050">List the &quot;publicly available sources&quot; 15T dataset list from Llama 3 · Issue #39 · meta-llama/llama3</a>：如果没有数据集来源列表，Llama 3 在任何实质性意义上都是不可复现的。请发布来源列表。</li><li><a href="https://github.com/naver-ai/rdnet">GitHub - naver-ai/rdnet</a>：通过在 GitHub 上创建账户来为 naver-ai/rdnet 的开发做出贡献。</li><li><a href="https://github.com/NVlabs/DoRA">GitHub - NVlabs/DoRA: Official PyTorch implementation of DoRA: Weight-Decomposed Low-Rank Adaptation</a>：DoRA 的官方 PyTorch 实现：权重分解低秩自适应 - NVlabs/DoRA</li><li><a href="https://ai.meta.com/blog/meta-llama-3/">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1230480409461133392)** (38 messages🔥): 

- **Chinchilla Token 计数保持不变**：每个参数的“最佳平均” Chinchilla Token 计数保持不变，根据讨论，这意味着**没有任何实质性的改变**。

- **Chinchilla 的第三种方法受到质疑**：**Chinchilla 的第三种 Scaling Law 方法之前被误用了**，但现在经过修正的统计分析使其与其他两种方法更加一致，提供了一种重新解释，即投入更多参数可能比投入更多数据更值得。

- **对第三种方法可靠性的批评**：尽管进行了修正，一些成员仍对第三种方法在 Scaling 策略中的可靠性持谨慎态度，建议采用替代解释，并认可原始 Chinchilla 论文中暗示的前两种方法的强度。

- **关于 Chinchilla 复现尝试的争论**：一项对 Chinchilla 研究的复现尝试表明参数化建模存在不稳定性，但对于其影响意见不一，讨论围绕论文的有效性和统计检验的显著性展开。

- **在论文仓库中发现彩蛋**：成员们强调了 arXiv 一个鲜为人知的功能，即可以下载论文的 TeX 源码以发现潜在的彩蛋和被注释掉的内容，这对于那些对学术出版的复杂细节和幕后故事感兴趣的人来说非常有价值。
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1230903433846980769)** (2 messages): 

- **DeepMind 的 Mechanistic Interpretability 更新**：Google DeepMind 的 Mechanistic Interpretability 团队[分享了进度更新](https://www.alignmentforum.org/posts/HpAr8k74mW4ivCvCu/progress-update-from-the-gdm-mech-interp-team-summary)，涵盖了 **Sparse Autoencoders (SAEs)** 的进展。亮点包括使用 SAEs 解释转向向量（steering vectors）、设计推理时稀疏近似算法、改进 Ghost Gradients，以及建立用于处理更大模型和 JAX 的基础设施。
- **Twitter 上的见解**：DeepMind 团队的 Neel Nanda 在推特上发布了关于此次更新的消息，反思了从他们的 Mechanistic Interpretability 研究中获得的经验和“技巧（hacks）”。该推文[可以在这里找到](https://twitter.com/NeelNanda5/status/1781400080802779604)。

**Link mentioned**: <a href="https://www.alignmentforum.org/posts/HpAr8k74mW4ivCvCu/progress-update-from-the-gdm-mech-interp-team-summary">Progress Update #1 from the GDM Mech Interp Team: Summary — AI Alignment Forum</a>: Introduction This is a progress update from the Google DeepMind mechanistic interpretability team, inspired by the Anthropic team’s excellent monthly…

  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1230426275433152523)** (14 messages🔥): 

- **考虑将 MMLU 作为 ARC 呈现**：成员们讨论了将 **MMLU 基准**以类似于 **ARC** 的方式呈现的想法，即不提供多项选择。虽然表现出兴趣，但除了打算将其与基准线进行比较外，对话中没有分享具体的数据或结果。
- **对数似然（Loglikelihood）计算的澄清**：在回答关于语言模型对数似然计算的询问时，参与者确认对数似然应针对续写（continuation）或目标（target）进行计算，而不是针对整个句子，以便仅针对续写部分计算困惑度（perplexity）。
- **注意到 vLLM 的巨大速度提升**：一位用户报告称，与传统的文本生成流水线相比，使用 **vLLM** 进行语言生成时速度提升了 **10 倍**，从而引发了关于最佳设置流程的讨论。
- **对 lm-evaluation-harness 的贡献**：一位成员分享了针对 **flores-200 和 sib-200 基准**的 [Pull Requests (PRs)](https://github.com/EleutherAI/lm-evaluation-harness/pull/1705)，以增强多语言评估。然而，他们被告知由于配置数量过多，按原样合并非常困难，需要设计一种更简洁的实现方法。
- **寻求 lm-evaluation-harness 单元测试的指导**：一位社区成员表示有兴趣为 **lm-evaluation-harness** 做出贡献，寻求关于运行单元测试和 CONTRIBUTING.md 文档当前相关性的澄清，以及关于各种测试依赖项的问题。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.google.com/spreadsheets/d/1luIEdZ_gH2GpFY9iLtM20oXemN6xaBzuGGkJAxQh-R0/edit?usp=sharing">MMLU - Alternative Prompts</a>: MMLU (Prompt 变体) 示例输入提示词，格式 01,{{question.strip}} 02,Q: {{question.strip}}\nA: 03,Question: {{question.strip}}\nAnswer: Llama-2-7b-hf,Mistral-7B-v0.1,falcon-7b,py...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1705">Implement Sib200 evaluation benchmark - text classification in 200 languages  by snova-zoltanc · Pull Request #1705 · EleutherAI/lm-evaluation-harness</a>: 我们使用了来自 MALA 论文 https://arxiv.org/pdf/2401.13303.pdf 的 Prompt 风格，我们在 SambaLingo 论文 https://arxiv.org/abs/2404.05829 中也发现该风格具有合理的效果。</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1706">Implementing Flores 200 translation evaluation benchmark across 200 languages by snova-zoltanc · Pull Request #1706 · EleutherAI/lm-evaluation-harness</a>: 我们使用了该论文中发现效果最好的 Prompt 模板。https://arxiv.org/pdf/2304.04675.pdf 我们的论文也发现该 Prompt 模板具有合理的效果 https://arxiv.o...
</li>
</ul>

</div>
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1230498831943208971)** (255 messages🔥🔥): 

- **Stability AI 即将裁员**: [Stability AI 正在缩减规模](https://www.cnbc.com/2024/04/18/ai-startup-stability-lays-off-10percent-of-employees-after-ceo-exit.html)，在 CEO 离职后裁员 20 多名员工，以在不可持续的增长后“调整业务规模”，引发了对公司未来的讨论。
- **Stable Diffusion 3 (SD3) 在与 DALL-E 3 的竞争中陷入苦战**: 成员们分享了 SD3 和 DALL-E 3 的对比，多次观察到 DALL-E 3 在图像质量和 Prompt 遵循度方面优于 SD3，几位成员对 SD3 的现状表示不满。
- **对 LLaMA 模型的看法**: 对 Meta 的 LLaMA 模型有浓厚兴趣，讨论了其编码能力、较小的 Context Window 限制以及可与其他行业巨头媲美的整体性能。
- **理解 Cross-Attention 机制**: 讨论探讨了 Text-to-Image 模型中 Cross-Attention 机制的细节，并澄清了 Google 的模型 imagen 在训练和采样过程中如何处理 Text Embeddings。
- **Prompt Engineering 与 Fine-Tuning 的优势**: 社区内正在对 Prompt Engineering 技术进行评估和辩论，例如添加摄像机和电影风格标签，以及 Fine-Tuning 模型以修正持续出现的图像生成伪影的必要性。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.cnbc.com/2024/04/18/ai-startup-stability-lays-off-10percent-of-employees-after-ceo-exit.html">AI startup Stability lays off 10% of staff after controversial CEO&#x27;s exit: Read the full memo</a>: 根据 CNBC 获得的内部备忘录，Stability AI 在经历了一段不可持续的增长后，裁减了多名员工以“调整业务规模”。</li><li><a href="https://eugeneyan.com/writing/text-to-image/">Text-to-Image: Diffusion, Text Conditioning, Guidance, Latent Space</a>: Text-to-Image 生成的基础知识、相关论文以及 DDPM 实验。</li><li><a href="https://www.meta.ai/?icebreaker=imagine">Meta AI</a>: 使用 Meta AI 助手完成任务，免费创建 AI 生成的图像，并获取任何问题的答案。Meta AI 基于 Meta 最新的 Llama 大语言模型构建，并使用 Emu...</li><li><a href="https://llama.meta.com/llama3/">Meta Llama 3</a>: 使用 Meta Llama 3 构建 AI 的未来。现在提供 8B 和 70B 的 Pretrained 和 Instruction-tuned 版本，支持广泛的应用。</li><li><a href="https://github.com/deep-floyd/IF/blob/develop/deepfloyd_if/model/unet.py#L225>">IF/deepfloyd_if/model/unet.py at develop · deep-floyd/IF</a>: 在 GitHub 上为 deep-floyd/IF 的开发做出贡献。</li><li><a href="https://www.llama2.ai/">Chat with Meta Llama 3 on Replicate</a>: Llama 3 是来自 Meta 的最新语言模型。</li><li><a href="https://github.com/shihaozhaozsh/lavi-bridge">GitHub - ShihaoZhaoZSH/LaVi-Bridge: Bridging Different Language Models and Generative Vision Models for Text-to-Image Generation</a>: 连接不同的语言模型和生成式视觉模型以进行 Text-to-Image 生成 - ShihaoZhaoZSH/LaVi-Bridge
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1230432789334327316)** (15 messages🔥):

- **探索开源音频驱动生成模型**：人们对开发一种高效且强大的开源生成模型版本表现出兴趣，用于音频驱动的头部和面部运动生成。目前正在考虑一种在全局面部动态和头部运动的潜空间（latent space）中进行训练的 **Diffusion Transformer** 模型。

- **处理面部动态的潜在方法**：将说话人头部视频或详细的面部网格进行潜空间编码，并结合以音频嵌入（audio embeddings）为条件的 **Diffusion** 模型，可能是生成面部动态的一条有前景的路径。建议的一种可能策略是利用音频嵌入进行实时（on the fly）模型训练。

- **Meta Llama 3 引入新可能性**：[Meta Llama 3 已正式发布](https://ai.meta.com/blog/meta-llama-3/)，并将很快在包括 AWS、Databricks、Google Cloud 和 Microsoft Azure 在内的多个平台上可用。公告中提到 Meta 致力于开发开源多模态模型，以与专有解决方案竞争。

- **对 Meta Llama 3 潜在影响的看法**：一位成员表示，如果 **Meta Llama 3** 的效率如宣传的那样出色，其意义将非常重大；而另一位成员则对其对个人和小型初创公司的可访问性持怀疑态度。人们对 Meta 开源多模态模型的发布充满期待，预计这些模型将与专有替代方案相抗衡。

**提到的链接**：<a href="https://ai.meta.com/blog/meta-llama-3/">未找到标题</a>：未找到描述

  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1230614322426941573)** (11 条消息🔥): 

- **征集多语言社区亮点**：一位成员建议将每周亮点扩展到更多语言，以覆盖更广泛的全球受众。他们表示，翻译每周亮点可以使内容对非英语母语者更具可读性和吸引力。
- **志愿者协助多语言模型**：另一位成员对翻译的想法给出了积极回应，提议协助测试小型多语言模型，以改进这一倡议。
- **翻译机器人 vs. 本地上下文**：在讨论为社区亮点引入翻译机器人时，有人指出在翻译中加入本地上下文可以增加参与度，强调了细致且具有文化相关性的内容的重要性。
- **人工增强翻译的案例**：一位贡献者强调了用各种语言创建视频讲解等内容的价值，建议翻译不应局限于文本，还应包括对新功能的演示和讨论，提供简单机器人翻译可能缺乏的深度理解。
- **预告 LLM Llama 3 的到来**：提到了 **Llama 3**（也称为 🦙🦙🦙）的发布，并指出它已在 HuggingFace 上提供，路径为 `MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF`，其在排行榜（leaderboards）上的表现备受期待。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/johko/computer-vision-course/pulls">Pull requests · johko/computer-vision-course</a>: 该仓库是社区驱动的神经网络 Computer Vision 课程的大本营。欢迎加入我们的 Hugging Face Discord：hf.co/join/discord - Pull requests · johko/computer...</li><li><a href="https://youtu.be/eNAOaFGrm2Y?list=PLcOiiEKFQrNFEXRsmZS8iWdmE7eWWlmbX">Destaques da Comunidade #52: Confira as últimas novidades de IA</a>: 查看在 #huggingface Discord 上发布的葡萄牙语版 Community Highlights 新闻！帖子：https://iatalk.ing/destaques-da-comunidade-52/摘要：...</li><li><a href="https://huggingface.co/spaces/tonyassi/IP-Adapter-Playground">IP-Adapter Playground - tonyassi 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/unography/blip-large-long-cap">unography/blip-large-long-cap · Hugging Face</a>: 未找到描述</li><li><a href="https://youtu.be/tzw6otpW-4A">infinite remix with musicgen, ableton, and python - part 2 - captains chair 22</a>: 00:00 - 回顾 00:38 - 人类音乐家延续 musicgen 的贝斯 02:39 - musicgen 延续人类音乐家的输出 04:04 - hoenn 的 lofi 模型扩展演示 06:...</li><li><a href="https://huggingface.co/spaces/thepatch/zero-gpu-slot-machine">Zero Gpu Slot Machine - thepatch 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/valory/olas-prediction-leaderboard">Leaderboard Gradio - valory 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/EduardoPacheco/Grounded-SAM">Grounded SAM - EduardoPacheco 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://github.com/Mihaiii/semantic-autocomplete">GitHub - Mihaiii/semantic-autocomplete: 一个极速的语义搜索 React 组件。按含义匹配，而不只是按字母。边输入边搜索无需等待（不需要 debounce）。按余弦相似度排序。</a>: 一个极速的语义搜索 React 组件。按含义匹配，而不只是按字母。边输入边搜索无需等待（不需要 debounce）。按余弦相似度排序。- Mihaiii/semantic-autocom...</li><li><a href="https://huggingface.co/posts/zolicsaki/821456421497684">@zolicsaki 在 Hugging Face 上："我们发布了针对阿拉伯语、泰语和……的新 SOTA SambaLingo 70B 参数模型"</a>: 未找到描述</li><li><a href="https://huggingface.co/posts/sted97/365197369008504">@sted97 在 Hugging Face 上："📣 我很高兴地宣布 'ALERT: 一个用于评估……的全面 #Benchmark'"</a>: 未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1230403962621005855)** (166 条消息🔥🔥): 

- **Gradio WebSockets 难题**: 出现了一场关于 [Gradio 依赖问题](https://discuss.huggingface.co/t/gradio-websockets-error-2023-04-23/9217)与 *websockets* 版本的讨论，强调了当其他依赖项需要 *websockets* 12 而 Gradio 的最新版本坚持使用较低版本时的兼容性问题。
- **Meta Llama 3 胜过 Mixtral**: [Mera-mix-4x7B](https://huggingface.co/meraGPT/mera-mix-4x7B) 是一种混合专家 (MoE) 模型，被介绍为体积更小但可与 Mixtral-8x7B 媲美，讨论中透露了详细的基准测试结果，例如在 OpenLLM Eval 上达到 75.91。
- **避免加密货币混淆**: 在对加密货币诈骗的担忧中，成员们建议 HuggingFace 应该公开否认与数字货币的任何关联。分享了一个指向 Twitter [警告此类诈骗的帖子](https://fxtwitter.com/not_so_lain/status/1781002822646989111)的链接。
- **量化新时代**: 出现了关于量化级别之间的差异及其对模型性能影响的疑问，并分享了关于 [量化对 LLM 性能影响](https://medium.com/@olga.zem/exploring-the-impact-of-quantization-on-llm-performance-5698e16c5564)讨论的链接。
- **Llama 3 困境**: 用户交流了关于 Meta Llama 模型之间差异的经验和见解，以及使用 Meta-Llama-3-8B Instruct 模型的方法。人们对 Llama 3 模型的特定输出以及可能引导模型行为的 *default system prompt* 表达了担忧。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/AIatMeta/status/1780997403979735440">来自 AI at Meta (@AIatMeta) 的推文</a>：介绍 Meta Llama 3：迄今为止功能最强大的开源 LLM。今天我们将发布 8B 和 70B 模型，它们提供了诸如改进的推理等新功能，并树立了新的行业领先水平...</li><li><a href="https://medium.com/@olga.zem/exploring-the-impact-of-quantization-on-llm-performance-5698e16c5564">探索 Quantization 对 LLM 性能的影响</a>：由 LLM Explorer 团队撰写</li><li><a href="https://huggingface.co/new-dataset">Hugging Face – 构建未来的 AI 社区。</a>：未找到描述</li><li><a href="https://huggingface.co/meraGPT/mera-mix-4x7B">meraGPT/mera-mix-4x7B · Hugging Face</a>：未找到描述</li><li><a href="https://llama.meta.com/llama3/">Meta Llama 3</a>：使用 Meta Llama 3 构建 AI 的未来。现在提供 8B 和 70B 的 Pretrained 和 Instruction-tuned 版本，以支持广泛的应用。</li><li><a href="https://huggingface.co/NousResearch/Meta-Llama-3-8B">NousResearch/Meta-Llama-3-8B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B">meta-llama/Meta-Llama-3-8B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF#use-with-transformers">MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/Ateeqq/Mixtral-8x22B">Mixtral-8x22B-Instruct-v0.1 - Ateeqq 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://www.philschmid.de/inferentia2-llama-7b">在 AWS Inferentia2 上使用 Amazon SageMaker 部署 Llama 2 7B</a>：在这篇博文中，你将学习如何在 AWS Inferentia2 上使用 Amazon SageMaker 编译和部署 Llama 2 7B。</li><li><a href="https://news.slashdot.org/story/24/04/17/2052256/feds-appoint-ai-doomer-to-run-us-ai-safety-institute">联邦政府任命 “AI 毁灭论者” 领导美国 AI 安全研究所 - Slashdot</a>：一位匿名读者引用了 Ars Technica 的报告：美国 AI 安全研究所（隶属于美国国家标准与技术研究院 NIST）终于宣布了其领导团队...</li><li><a href="https://hf.co/competitions">competitions (竞赛)</a>：未找到描述</li><li><a href="https://huggingface.co/aws-neuron/optimum-neuron-cache">aws-neuron/optimum-neuron-cache · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/huggingface/accelerate/pull/2687">由 nroggendorff 更新 config_args.py · Pull Request #2687 · huggingface/accelerate</a>：更新 config_args.py 以适配最新版本的 Amazon SageMaker</li><li><a href="https://azuremarketplace.microsoft.com/en-us/marketplace/apps/metagenai.meta-llama-3-8b-chat-offer?tab=overview">Microsoft Azure Marketplace</a>：未找到描述</li><li><a href="https://github.com/EricLBuehler/mistral.rs">GitHub - EricLBuehler/mistral.rs: 极速 LLM 推理。</a>：极速 LLM 推理。通过在 GitHub 上创建账号来为 EricLBuehler/mistral.rs 的开发做出贡献。</li><li><a href="https://github.com/Locutusque/TPU-Alignment">GitHub - Locutusque/TPU-Alignment: 完全免费地对 Mistral、Llama-2-13B 或 Qwen-14B 等大模型进行全量微调</a>：完全免费地对 Mistral、Llama-2-13B 或 Qwen-14B 等大模型进行全量微调 - Locutusque/TPU-Alignment</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/discussions/4">meta-llama/Meta-Llama-3-8B-Instruct · 更新 generation_config.json</a>：未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1230537172277399592)** (4 条消息): 

- **新用户遇到输入验证错误**：一位新成员在使用 **LLaMA-2-7b-chat-hf 模型**时遇到了 **HfHubHTTPError: 422**。错误消息指出 `inputs must have less than 4096 tokens. Given: 4545`（输入必须少于 4096 个 token。给定：4545），表明需要减少输入大小。

- **Meta 发布 LLaMA 3 - 深入探讨新功能**：分享了一个名为 “Meta Releases LLaMA 3: Deep Dive & Demo” 的视频，详细介绍了 **Meta LLaMA 3** 模型的发布及其特性。该 [YouTube 视频](https://www.youtube.com/watch?v=E3_0nHpfbcY) 提供了关于 2024 年 4 月 18 日发布的该模型的见解。

- **对 LLaMA 3 早期印象的好奇**：一位成员表达了想了解其他人对新发布的 **LLaMA 3** 模型使用体验的兴趣，并使用表情符号展示了热情。

**提到的链接**：<a href="https://www.youtube.com/watch?v=E3_0nHpfbcY">Meta Releases LLaMA 3: Deep Dive &amp; Demo</a>：今天，2024 年 4 月 18 日，是一个特别的日子！在这段视频中，我将介绍 @meta 的 LLaMA 3 发布。该模型是其第三个迭代版本...

  

---

**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1230493829052760165)** (9 messages🔥): 

- **生成式 AI 书籍编写中**：书籍《Hands-On Generative AI with Transformers and Diffusion Models》获得了好评，作者承认该书仍在编写中，还有五章内容待发布。
  
- **期待更多关于 AI 与设计的内容**：希望该书后续章节能涵盖 **quantization** 和 **design systems** 等主题。

- **生成式 AI 书籍中使用的内部库**：针对上述 AI 书籍中 `show_images` 函数所使用的库的询问，未指明具体使用了哪个库。

- **利用 Blender 进行合成数据生成**：一篇 [博客文章](https://federicoarenasl.github.io/Data-Generation-with-Blender/) 强调了使用 **Blender** 进行合成数据生成的潜力，展示了其在物体识别问题中的应用，并分多个章节详细说明了流程。

- **Ruff 0.4 提升性能**：**Ruff** 0.4 版本的 lint 和 format 性能 [得到了显著提升](https://fxtwitter.com/charliermarsh/status/1781051101661245483)，由于切换到了手写的 recursive descent parser，现在的速度比之前版本快了两倍以上。

- **关于 PyTorch 稳定损失函数的见解**：一篇 [Medium 文章](https://medium.com/@sahilcarterr/why-nn-bcewithlogitsloss-numerically-stable-6a04f3052967) 解读了 **PyTorch** 的 **BCEWithLogitsLoss**，强调了其相比于先使用普通 Sigmoid 再接 Binary Cross Entropy Loss 的数值稳定性。 

- **用于训练的 ORPO-DPO-mix-40k 数据集**：[ORPO-DPO-mix-40k 数据集](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k) 结合了高质量的 **DPO datasets**，用于 ORPO 或 DPO 训练，旨在辅助机器学习模型的开发。

- **探索 Llama3 的阴暗面**：一篇 LinkedIn [帖子讨论了](https://www.linkedin.com/posts/divyanshuusingh_llama3-redteaming-activity-7187068733280993281-ThQO) **Llama3** 的“阴暗面”，提供了关于这一 AI 工具较少为人知的一面的见解。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://federicoarenasl.github.io/Data-Generation-with-Blender/">使用 Blender 和 Python 生成合成数据</a>：为真实的 YOLO 应用生成训练和标注数据的详尽指南</li><li><a href="https://fxtwitter.com/charliermarsh/status/1781051101661245483">Charlie Marsh (@charliermarsh) 的推文</a>：Ruff v0.4.0 现已发布。我们已从生成的解析器迁移到手写的 recursive descent parser。速度提升了 2 倍以上，并使 lint 和 format 性能提高了约 20-40%...</li><li><a href="https://medium.com/@sahilcarterr/why-nn-bcewithlogitsloss-numerically-stable-6a04f3052967">为什么 nn.BCEWithLogitsLoss 是数值稳定的</a>：数值稳定性是机器学习中一个至关重要的考虑因素。BCEWithLogitsLoss 是一个结合了 Sigmoid 层和...</li><li><a href="https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k">mlabonne/orpo-dpo-mix-40k · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1230404631608168468)** (12 messages🔥): 

- **新颖的浏览器内音频体验**：一位用户分享了他们的项目，一个名为 [zero-gpu-slot-machine](https://huggingface.co/spaces/thepatch/zero-gpu-slot-machine) 的 HuggingFace Space。这是一个创新工具，使用音频输入通过 midi2musicgen2musicgen 和 Gradio 生成音乐。他们创建了一个无需 prompt 的生成系统，可以使用 MusicLang MIDI 模型处理和弦，并在 [发布的视频](https://x.com/thepatch_kev/status/1780844295009833023) 中演示了其功能。

- **预测未来事件的新排行榜上线**：HuggingFace 推出了一个新的排行榜，用于评估 LLM、workflows 和 Agent 在预测未来事件方面的有效性，这是一个相对未被探索的任务。公告中包含了 [预测排行榜](https://huggingface.co/spaces/valory/olas-prediction-leaderboard) 的链接，目前包含两个开源模型，后续将添加更多。

- **将社区亮点扩展至葡萄牙语受众**：一位成员提到他们正在将 Community Highlights 翻译成葡萄牙语并包含 demo，第 52 和 53 集已在面向葡萄牙语用户的 [YouTube 播放列表](https://www.youtube.com/watch?v=eNAOaFGrm2Y&list=PLcOiiEKFQrNFEXRsmZS8iWdmE7eWWlmbX) 中提供。

- **社区内容的潜在多语言扩展**：针对 Community Highlights 的葡萄牙语翻译，另一位成员建议创建韩语等其他语言的内容，并表示有兴趣制作博客文章或视频，以迎合不同语言的受众。

- **HuggingFace 上的 Meta-Llama 量化版本**：一位用户介绍了托管在 HuggingFace 上的 Meta-Llama-3-8B-Instruct 的 GGUF 量化版本，这是一个重新上传的模型，带有一个新的结束标记（end token），可以在[这里](https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF)找到。

- **社区准则提醒**：一名管理员提醒用户，频道内不允许发布 Discord 邀请链接，并引导他们参阅相关的规则章节。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF">QuantFactory/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/valory/olas-prediction-leaderboard">Leaderboard Gradio - a Hugging Face Space by valory</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/thepatch/zero-gpu-slot-machine">Zero Gpu Slot Machine - a Hugging Face Space by thepatch</a>：未找到描述</li><li><a href="https://x.com/thepatch_kev/status/1780844295009833023">thecollabagepatch (@thepatch_kev) 的推文</a>：好吧，这个 @huggingface space 意外地变成了一个小型的 lo-fi daw midi2musicgen2musicgen2... 你懂的 @gradio 像往常一样为我做了很多工作，musicgen 模型由 @veryVANYA @_lyraaaa_ @eschat... 提供</li><li><a href="https://www.youtube.com/watch?v=eNAOaFGrm2Y&list=PLcOiiEKFQrNFEXRsmZS8iWdmE7eWWlmbX">Destaques da Comunidade #52: Confira as últimas novidades de IA</a>：查看在 #huggingface Discord 上发布的 Community Highlights 最新消息（葡萄牙语）！文章：https://iatalk.ing/destaques-da-comunidade-52/ 摘要：...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1230608632123494400)** (1 条消息): 

- **讨论 Counterfactual-Inception 的邀请**：一位成员对最近关于 **Counterfactual-Inception** 的研究表示感兴趣，并邀请研究人员与阅读小组分享见解。该研究可在 [GitHub](https://github.com/ivy-lvlm/counterfactual-inception) 上找到。

**提到的链接**：<a href="https://github.com/ivy-lvlm/counterfactual-inception">GitHub - IVY-LVLM/Counterfactual-Inception</a>：通过在 GitHub 上创建账户来为 IVY-LVLM/Counterfactual-Inception 的开发做出贡献。

  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1230520337230331945)** (2 条消息): 

- **关于基于 DRL 的目标检测的咨询**：一位成员询问了利用**深度强化学习 (DRL)** 的目标检测系统，并对其性能指标表示关注。

- **寻求 3D 视觉数据集的见解**：另一位成员寻求关于理解 3D 计算机视觉数据集的信息和研究论文，特别提到了 **Google Scanned Objects (GSO)**。他们询问是否有人可以分享相关的学习资料。
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1230743800918315040)** (3 条消息): 

- **追踪难以捉摸的 'cursorop' 问题**：一位成员提到 'cursorop' 错误通常由于内存不足或包兼容性问题（Package Compatibility Issues）引起。
- **TensorFlow 需要 'cursorop'**：TensorFlow 的必要性被暗示为成员遇到 'cursorop' 错误的主要原因。
- **寻找错误原因**：一位成员寻求帮助以理解某个未指明错误背后的原因，但未提供进一步的细节或上下文。
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1230555393390809130)** (4 条消息): 

- **Raylib 问题寻求帮助**：一位成员在花费 3 小时试图解决找不到 `raylib.h` 文件的问题后表示沮丧。他们公开请求有 Raylib 经验的人提供帮助。
- **在使用 Lora 进行 Inpainting 时追求背景一致性**：为了在不改变前景对象的情况下保持背景一致，一位成员正在寻找标准 Inpainting 方法的替代方案，并表示有兴趣探索 Lora 训练。他们请求有相关经验或见解的人提供建议。
  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1230474228852326411)** (2 条消息):

- **Prompt Template 修正**：[**Mixtral 8x22B Instruct**](https://openrouter.ai/models/mistralai/mixtral-8x22b-instruct) 的 Prompt Template 在之前的混淆后已得到修正。此更新影响了该模型的使用指南。

- **推文提醒**：OpenRouter 通过 [**Twitter 帖子**](https://twitter.com/OpenRouterAI/status/1781091399582236686) 分享了更新——推文的具体内容未在 Discord 消息中披露。

**提到的链接**：<a href="https://openrouter.ai/models/mistralai/mixtral-8x22b-instruct)">Mixtral 8x22B by mistralai | OpenRouter</a>：Mixtral 8x22B 是来自 Mistral AI 的大规模语言模型。它由 8 个专家（experts）组成，每个专家拥有 220 亿参数，每个 token 每次使用 2 个专家。它通过 [X](https://twitter... 发布。

---

**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1230433651070599239)** (198 条消息🔥🔥): 

- **关于 OpenRouter 营收的讨论**：成员们推测 OpenRouter 可能通过 **批量折扣（bulk discounts）** 或从用户充值中提取小额分成来获利，尽管聊天中没有提供官方确认。一些人认为此类折扣已得到确认，并在用户和提供商之间共享。

- **延迟与服务器位置**：一位用户提出了在南美洲使用 VPS 时的 **延迟（latency）** 问题，然而，讨论并未就物理服务器位置是否显著影响延迟给出明确答案。

- **对 LLaMA 3 模型的期待**：讨论指出 **LLaMA 3 模型** 正在发布，包括 8B 和前所未有的 70B 参数版本。用户分享了 Azure 提供 LLaMA 3 模型的消息，甚至提供了 [产品链接](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/metagenai.meta-llama-3-8b-chat-offer?tab=Overview)。

- **Meta LLaMA 3 发布**：用户热烈讨论新的 **Meta LLaMA 3** 模型，指出它们似乎受到的审查较少，并提供了 [模型页面](https://llama.meta.com/llama3/) 的链接以及在 [Meta LLaMa Downloads](https://llama.meta.com/llama-downloads) 请求访问权重的下载地址。

- **在生产环境中使用 OpenRouter**：一位用户确认在生产环境（production）中使用 OpenRouter，分享了他们在 **[Olympia.chat](https://olympia.chat)** 的经验，并表示在 OpenRouter 首页可以找到许多其他生产环境用户。另一位用户在从直接使用 OpenAI 转向 OpenRouter 时，寻求有关集成特定消息（如 tools 和 function calls）的建议，并强调缺乏清晰的文档。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://llama.meta.com/llama3/">Meta Llama 3</a>: 使用 Meta Llama 3 构建 AI 的未来。现在提供 8B 和 70B 的预训练及指令微调版本，以支持广泛的应用。</li><li><a href="https://docs.librechat.ai/install/configuration/ai_endpoints.html">✅ Compatible AI Endpoints</a>: 已知的兼容 AI 端点列表，包含 `librechat.yaml`（即 LibreChat 自定义配置文件）的示例设置。</li><li><a href="https://azuremarketplace.microsoft.com/en-us/marketplace/apps/metagenai.meta-llama-3-8b-chat-offer?tab=Overview">Microsoft Azure Marketplace</a>: 未找到描述</li><li><a href="https://llama.meta.com/llama-downloads">Download Llama</a>: 申请访问 Llama。</li><li><a href="https://x.com/aravsrinivas/status/1769485603622867394?s=46&t=orNoaT1ei7RpUogkauM1-Q">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>: 是的，感谢 @elonmusk 和 xAI 团队开源了 Grok 的基础模型。我们将针对对话式搜索对其进行微调并优化推理，并将其提供给所有 Pro 用户！ ↘️ Quoti...</li><li><a href="https://olympia.chat">Olympia | Better Than ChatGPT</a>: 通过价格合理的 AI 驱动顾问来发展您的业务，这些顾问是业务战略、内容开发、营销、编程、法律战略等方面的专家。</li><li><a href="https://deepinfra.com/mistralai/Mixtral-8x22B-Instruct-v0.1">mistralai/Mixtral-8x22B-Instruct-v0.1 - Demo - DeepInfra</a>: 这是 Mixtral-8x22B 的指令微调版本——来自 Mistral AI 的最新且最大的混合专家（MoE）大语言模型 (LLM)。这一先进的机器学习模型使用了...</li><li><a href="https://together-ai.webflow.io/blog/together-ai-partners-with-meta-to-release-meta-llama-3-for-inference-and-fine-tuning">Together AI 与 Meta 合作发布用于推理和微调的 Meta Llama 3</a>: 未找到描述</li><li><a href="https://groq.com/">GroqChat</a>: 未找到描述</li><li><a href="https://huggingface.co/chat/models/meta-llama/Meta-Llama-3-70B-Instruct">meta-llama/Meta-Llama-3-70B-Instruct - HuggingChat</a>: 在 HuggingChat 中使用 meta-llama/Meta-Llama-3-70B-Instruct</li><li><a href="https://leanpub.com/patterns-of-application-development-using-ai">Patterns of Application Development Using AI</a>: 探索构建智能、自适应且以用户为中心的软件系统的实用模式和原则，充分发挥 AI 的力量。</li><li><a href="https://openrouter.ai/docs#required-parameters-(beta)">OpenRouter</a>: 构建与模型无关的 AI 应用
</li>
</ul>

</div>
  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1230443043870736457)** (106 条消息🔥🔥): 

- **求知欲旺盛**: 成员们寻求在企业应用中实施 **OpenAI 产品** 的最佳实践，包括收集、组织和分类这些实践。他们讨论了与直接文件上传相关的挑战，并对增加团队订阅上传限制的方法感到好奇。
- **AI 模型速度与性能辩论**: 参与者讨论了 **GPT-4, Claude, 和 Llama 3** 等各种模型的速度和性能。有人强调 Claude Opus 速度很快，可与 GPT-4 Turbo 媲美，成员们建议性能可能取决于地区以及 Prompt 是否采用流式传输等因素。
- **探索 Llama 3 的影响与集成**: 对话集中在 **Meta 的 Llama 3**，分享了指向 [Meta Llama 3 介绍](https://ai.meta.com/blog/meta-llama-3/) 等博客文章的链接。关于 Llama 3 与 GPT-3.5 的比较以及对其集成到消费级应用中的期待存在辩论。
- **模型可访问性与消费级硬件挑战**: 成员们讨论了在消费级硬件上运行 Llama 3 和 GPT-4 等大型模型的实用性。人们对计算和成本要求表示担忧，一些人提到了未来对消费级设备的潜在改进和适配。
- **AI 工具的学习与精通**: 社区就学习和掌握 **DALL-E 3** 等工具的最佳方式交换了意见，质疑现有课程的可信度和有效性。一些人建议对自封的专家保持怀疑，并建议从特定的 Discord 频道寻求帮助，或通过直接使用该技术进行学习。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://ai.meta.com/blog/meta-llama-3/">未找到标题</a>: 未找到描述</li><li><a href="https://youtu.be/bc6uFV9CJGg">Mark Zuckerberg - Llama 3, $10B Models, Caesar Augustus, &amp; 1 GW Datacenters</a>: 扎克伯格谈论：Llama 3、迈向 AGI 的开源、定制芯片、合成数据以及扩展时的能源限制、Caesar Augustus、智能爆炸、生物...
</li>
</ul>

</div>
  

---

**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1230475151678767174)** (22 messages🔥): 

- **寻找最快模型**：一名用户反映 **GPT-4-0125-preview** 运行缓慢，并询问是否有更快的 GPT-4 版本。建议包括尝试更新后的 **gpt-4-turbo-2024-04-09**，但该用户觉得它甚至更慢。

- **模型速度与性能对比**：用户讨论了 GPT-4 的替代模型，如 **Claude** 和 **Command R**，并对比了它们的速度和智能程度。一位用户提到 **Claude** 在一系列消息对话后效果会变差。

- **对 GPT-4 Turbo 的担忧**：一名用户对 **gpt-4-turbo (非 preview 版本，即 0409 版本)** 表示失望，认为在他们的 Assistant 应用中，其表现不如之前的 **0125** preview 版本。

- **Assistant API 导致加载消息迟缓**：用户提出了 Assistant API 的一个问题，即系统在显示加载消息之前会等待函数结果。一位用户建议通过 UI 变通方案来动态控制加载消息的可见性。

- **集成 GPT-4 与 Claude**：一名用户将 **GPT-4 与 Claude 3 Opus** 结合，创建了一种聊天体验：先由 GPT-4 响应，然后查询 Claude 模型，最后合并两个响应。然而，有报告称在尝试所提供链接的服务时出现了错误。
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1230434931147276318)** (30 messages🔥): 

- **代码集成挫败感**：一名成员对尝试集成三个简单的 Python 文件感到沮丧，因为系统“不断丢失代码”；他们不得不手动完成集成。
- **ChatGPT 性能受质疑**：关于 Elon Musk 诉讼后 ChatGPT 性能下降的讨论，包括提及“故意保留实力 (sandbagging)”以及认为质量下降似乎是“刻意为之”的评论。
- **AI 的 PDF 毒药解药**：在一次技术建议交流中，一名成员被建议不要使用 PDF 为 AI 助手提供规则，指出 PDF 存在诸多问题，并鼓励使用 **plain text** 或 **XML、JSON** 等结构化格式。
- **提示工程 (Prompt Engineering) 最佳实践查询**：一名成员参考官方 [OpenAI 指南](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api)，寻求关于 OpenAI 最新提示工程最佳实践的确认，并向社区询问更新情况。
- **JSON 数据摘要挑战**：一名成员寻求帮助，想让 GPT 从 JSON 数据摘要的特定字段中返回准确文本，并指出尽管在提示词中明确说明了指令，但仍存在困难。
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1230434931147276318)** (30 messages🔥): 

- **AI 中 PDF 的陷阱**：成员们讨论了 PDF 不适合为 AI 提供上下文的问题，有人建议 plain text 效果更好，另一人强调 PDF 是“元数据灾难区”，对 AI 来说是“垃圾”。建议以 JSON 或 XML 等更结构化的格式提供规则，但如前所述，文件不支持 XML，因此建议嵌入规则或使用 plain text 格式。
- **提示词构建挑战**：一名成员对 AI 在尝试集成多个 Python 文件时丢失代码表示沮丧，导致他们不得不求助于手动集成。同样，另一名成员尽管尝试了多次，仍难以让 AI 在总结 JSON 数据时包含特定字段的准确文本。
- **模型性能焦虑**：讨论中出现了对 ChatGPT 性能下降的担忧，成员们推测了可能的原因，从法律压力导致的战略性“sandbagging”，到刻意的降级，再到以“system prompt delta hades”为特征的糟糕提示词设计。
- **探索特定任务的摘要生成**：一名成员正在使用 OpenAI API 分析会议转录，并为每个参与者提取任务或摘要。他们参考了 OpenAI 提示工程最佳实践，并考虑使用 batch API 进行非实时处理。
- **区块链与 AI 融合构想**：一名自称是区块链开发者的成员希望将 AI 与区块链技术结合，并邀请他人合作开发这一概念。他们还在寻求改进其提示词设计。
  

---



**Interconnects (Nathan Lambert) ▷ #[ideas-and-feedback](https://discord.com/channels/1179127597926469703/1179127598442348730/1230524625448992829)** (3 messages):

- **MCTS 简要说明**：一位成员澄清说，**MCTS** 代表 *Monte Carlo Tree Search*（蒙特卡洛树搜索），这是一种常用于游戏 AI 的搜索算法。
- **将 MCTS 与 PPO 结合用于语言生成**：一位成员分享了与一篇关于[将 MCTS 与 PPO 集成](https://arxiv.org/abs/2309.15028)的论文作者合作的兴奋之情，该研究通过在推理阶段利用 PPO 的价值网络来指导 MCTS，从而增强文本生成。该论文介绍了 **PPO-MCTS**，一种新型的价值引导解码算法。

**提到的链接**：<a href="https://arxiv.org/abs/2309.15028">Don&#39;t throw away your value model! Generating more preferable text with Value-Guided Monte-Carlo Tree Search decoding</a>：当基于最先进的强化学习（如 Proximal Pol...）生成自然语言文本时，像蒙特卡洛树搜索（MCTS）这样的推理时搜索算法似乎是不必要的。

---

**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1230514520448040971)** (142 messages🔥🔥): 

- **Meta Llama 3 登场**：Meta 发布了 **Meta Llama 3 系列大语言模型**，预训练和微调版本参数范围从 80 亿到 700 亿不等，针对对话场景进行了优化。公告内容包括开放 Llama-3-8B 推理 API 的访问权限，以及在 Azure AI Studio 中托管微调，该平台宣传了有助于构建 Generative AI 应用的功能。
- **Replicate 的竞争性定价揭晓**：分享了一份用于模型托管服务的 GPU 和 CPU 价格表，成本细分为每秒和每小时，强调了在 AI 工作中使用外部计算资源的财务方面。
- **Meta 的 Llama 3 模型引发兴奋与质疑**：频道中的讨论反映了对 **Meta Llama 3 模型** 的期待，包括对其作为“GPT-4 杀手”潜力的猜测，以及对其语言能力、定价和与 GPT-3.5 等现有模型性能对比的讨论。
- **对 Llama 3 实际可用性的担忧**：有人指出发布时间较早，并对其多语言效果和微调提出了批评，认为虽然 Llama 3 声称具备多语言能力，但在当前迭代中可能并未得到高度重视或完全实现。
- **对 405B 模型规模和可访问性的预测**：围绕 Meta 宣布计划推出 **4050 亿参数的 Llama 3 模型** 的热情高涨，推测这将是一个 open-weight 模型，可能改变开源 AI 的格局，并增加 GPT-5 等闭源模型的竞争压力。官方公告和更多细节备受期待。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://replicate.com/docs/billing">How billing works on Replicate</a>：Replicate 的计费方式</li><li><a href="https://www.interconnects.ai/p/llama-3-and-scaling-open-llms">Llama 3: Scaling open LLMs to AGI</a>：Llama 3 表明，在不久的将来，规模化不会成为开源 LLM 进步的限制。</li><li><a href="https://x.com/lmsysorg/status/1781167028654727550?s=46">lmsys.org (@lmsysorg) 的推文</a>：前 1000 票已经投出，Llama-3 表现火爆！🔥 开源模型的新王者？现在投票，发出你的声音！排行榜更新即将发布。↘️ 引用 lmsys.org (@lmsysorg) 大力祝贺...</li><li><a href="https://azuremarketplace.microsoft.com/en-US/marketplace/apps/metagenai.meta-llama-3-8b-chat-offer?tab=overview">Microsoft Azure Marketplace</a>：未找到描述</li><li><a href="https://x.com/nahrzf/status/1781011649580712342?s=46">nahr (@nahrzf) 的推文</a>：meta 做了最有趣的事</li><li><a href="https://x.com/DrJimFan/status/1781006672452038756">Jim Fan (@DrJimFan) 的推文</a>：即将推出的 Llama-3-400B+ 将标志着社区获得 GPT-4 级模型 open-weight 访问权限的分水岭时刻。它将改变许多研究工作和草根初创公司的计算方式...</li><li><a href="https://www.youtube.com/watch?v=gqtmUHhaplo">OpenAssistant is Completed</a>：#OpenAssistantLAION 的 OpenEmpathic：https://laion.ai/blog/open-empathic/ 链接：主页：https://ykilcher.com 商店：https://ykilcher.com/merch YouTube：https:/...
</li>
</ul>

</div>

---

**Interconnects (Nathan Lambert) ▷ #[sp2024-history-of-open-alignment](https://discord.com/channels/1179127597926469703/1223784028428177510/1230456787115835433)** (16 messages🔥):

- **演示文稿中的 LLaMA3 预期**：@philpax 幽默地思考 @natolambert 是否必须在幻灯片中加入关于 **LLaMA3** 发布的最新的更新。与此同时，@xeophon 希望该版本能尽快发布，因为他们已经在一次演讲中引用了它。
- **LLaMA3 热度降温**：@natolambert 对 LLaMA3 表现出缺乏热情，表示虽然可以增加一张幻灯片来安抚观众，但强调目前手头有大量正在进行的项目。
- **准备演讲后的提问环节**：@natolambert 分享了在演讲结束时开放提问的意图，并提到目前正专注于先写一篇博客文章。
- **关于 LLaMA-Guard 及其基准测试的查询**：@420gunna 对 LLaMA-Guard 表示好奇，询问这类模型是否有特定的名称或基准测试（benchmarks）。@natolambert 回应称其为“安全分类器（safety classifier）”，并提到 AI2 正在为相关系统开发基准测试。
- **等待录像发布**：@philpax 询问是否很快会有录像，可能指的是 @natolambert 的演示或相关活动。
  

---


**Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1230568988854059048)** (2 messages): 

- **SnailBot 推广**：一位用户发表了轻松的评论，祝贺 SnailBot 被投入到一篇正在撰写的文章中。
- **俏皮的欢迎**：natolambert 俏皮地认可了用户在使用 SnailBot 方面取得的成就，在祝贺的同时不失幽默。
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1230542613472350310)** (19 messages🔥): 

- **Mojo 邂逅 C 编程**：用户可以通过使用 `external_call` 功能在 Mojo 中利用 C，正如[最近在 Twitter 上的教程](https://twitter.com/Modular/status/1779913837216719118)所示。该过程未来将得到简化，旨在实现直接调用 C/C++ 函数而无需复杂的 FFI 层。
- **Mojo 中的 C 变得更简单**：开发团队计划简化 C 在 Mojo 中的集成，正如一位成员提到的，可能允许开发人员使用常规参数导入和调用外部函数，从而简化当前流程。
- **Mojo-Swift 互操作性讨论**：关于 Mojo 与 Swift 接口的讨论围绕着可能更顺畅的互操作展开，这可能涉及 Mojo <-> C++ <-> Swift，并参考了 FFI (Foreign Function Interface)，以及 GitHub 上的 [MLX-Swift](https://github.com/ml-explore/mlx-swift) 等示例。
- **Modular 的路线图和语言优先级**：正如其[路线图和使命](https://docs.modular.com/mojo/roadmap#cc-interop)所示，Mojo 语言正专注于基本的系统编程功能，并承认该语言尚处于早期阶段。
- **社区对 Mojo GUI 的兴趣**：社区似乎对 Mojo 的 GUI 库感兴趣，Swift UI 被建议作为一个理想的工具，尽管创建这样一个库可能是社区驱动的努力，而不是来自 Modular 开发者的直接产出。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/roadmap#cc-interop">Mojo🔥 roadmap &amp; sharp edges | Modular Docs</a>：关于我们 Mojo 计划的摘要，包括即将推出的功能和我们需要修复的问题。</li><li><a href="https://github.com/ml-explore/mlx-swift">GitHub - ml-explore/mlx-swift: Swift API for MLX</a>：MLX 的 Swift API。欢迎在 GitHub 上为 ml-explore/mlx-swift 的开发做出贡献。</li><li><a href="https://github.com/ihnorton/mojo-ffi">GitHub - ihnorton/mojo-ffi</a>：欢迎在 GitHub 上为 ihnorton/mojo-ffi 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1230565632865734758)** (2 messages): 

- **Modular 进军 Twitter**：Modular 在其官方账号上分享了一条 [推文](https://twitter.com/Modular/status/1781000544158650716)，为关注者提供见解或更新。
  
- **新鲜出炉**：Modular 发布了另一条 [推文](https://twitter.com/Modular/status/1781426483149602820)，可能发布了新内容、公告或与社区互动。
  

---


**Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1230589476498178198)** (2 messages): 

- **Meta 发布 LLaMA 3**：分享了一个名为“Meta Releases LLaMA 3: Deep Dive & Demo”的视频，讨论了 **Meta LLaMA 3** 的发布。该视频被描述为对 Meta 该模型第三次迭代的特别报道，日期为 2024 年 4 月 18 日，可以在 [这里](https://www.youtube.com/watch?v=E3_0nHpfbcY) 观看。

- **等级提升通知**：一名用户因在 **ModularBot** 排名系统中晋升至 **level 1** 而受到祝贺。未提供更多上下文。

**提到的链接**：<a href="https://www.youtube.com/watch?v=E3_0nHpfbcY">Meta Releases LLaMA 3: Deep Dive &amp; Demo</a>：今天，2024年4月18日，是一个特别的日子！在这段视频中，我将介绍 @meta 的 LLaMA 3 发布。该模型是其第三次迭代...

  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1230469271365091378)** (118 messages🔥🔥): 

- **探讨运行时 Garbage Collection 讨论**：对话围绕 Nim 语言 2.0 版本中带有引用计数（ref counting）的运行时循环收集器展开。一位用户提议 Modular 的 Mojo 语言集成类似机制的可能性，暗示 Mojo 有时只是移动引用，但其他成员对复杂性和潜在的性能影响表示担忧。

- **Mojo 的语言能力和功能推测**：用户讨论了 Mojo 对各种功能的支持程度。有人谈到 Mojo 可能通过 C 外部调用支持 Shell 命令，对元组（tuples）因缺乏 "Stringable" trait 而无法打印感到困惑，并好奇 Mojo 是否会包含类似于 Zig 的一等公民测试支持（first-class test support）。

- **思考一等公民测试支持和装饰器（Decorators）**：一个关键话题是在 Mojo 中加入一等公民测试的可能性，类似于 Zig 的测试方法。还有人提到倾向于使用 `@test` 装饰器，使测试在非测试环境下成为死代码（dead code），类似于 Rust 的 `#[cfg(test)]`。 

- **向往类 Pytest 的断言**：一位成员表示希望 Mojo 中的断言能更像 Pytest，使用内置的 assert 关键字。这引发了关于测试与断言之间的区别以及拥有测试套件重要性的讨论。

- **包管理与社区贡献**：社区参与 Mojo 的成长以及打包和构建系统的可能性令人兴奋。Modular 打算开发诸如 `mojo test` 之类的工具，并为打包系统奠定基础，这激发了关于允许社区构建测试框架和其他包的讨论。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://doc.rust-lang.org/rust-by-example/testing/unit_testing.html">Unit testing - Rust By Example</a>：未找到描述</li><li><a href="https://docs.pytest.org/en/8.0.x/index.html">pytest: helps you write better programs &#8212; pytest documentation</a>：未找到描述</li><li><a href="https://docs.modular.com/mojo/stdlib/utils/variant">variant | Modular Docs</a>：定义 Variant 类型。</li><li><a href="https://tenor.com/view/the-office-andy-andy-bernard-thought-about-it-im-in-gif-16547652">The Office Andy GIF - The Office Andy Andy Bernard - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://github.com/thatstoasty/mist">GitHub - thatstoasty/mist: Advanced ANSI style &amp; color support for your terminal applications</a>：为你的终端应用程序提供高级 ANSI 样式和颜色支持 - thatstoasty/mist</li><li><a href="https://github.com/thatstoast">thatstoast - Overview</a>：GitHub 是 thatstoast 构建软件的地方。</li><li><a href="https://github.com/dimitrilw/roguelike-mojo/blob/main/src/main.mojo#L51>">roguelike-mojo/src/main.mojo at main · dimitrilw/roguelike-mojo</a>：使用 Mojo🔥 逐步完成 Python Rogue-like 教程。 - dimitrilw/roguelike-mojo</li><li><a href="https://github.com/dimitrilw/roguelike-mojo/blob/main/src/main.mojo#L57-L63>">roguelike-mojo/src/main.mojo at main · dimitrilw/roguelike-mojo</a>：使用 Mojo🔥 逐步完成 Python Rogue-like 教程。 - dimitrilw/roguelike-mojo</li><li><a href="https://tenor.com/search/"">&quot; GIFs | Tenor</a>：点击查看 GIF</li><li><a href="https://github.com/modularml/mojo/discussions/1785">[Proposal] Mojo project manifest and build tool · modularml/mojo · Discussion #1785</a>：大家好，请查看这个关于 Mojo 项目清单（manifest）和构建工具的提案。正如提案本身所述，我们希望听取 Mojo 社区的意见：你是否同意这个动机...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/)** (1 messages): 

arnaud6135: 谢谢，我马上读 😄
  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1230901310006755388)** (3 messages):

- **快速 Mojo 之谜**：在比较计算速度时，一位成员注意到 **Rust** 的前缀和（prefix sum）计算比 **Modular (Mojo)** 慢 6 倍，尽管尝试了使用特定硬件标志（hardware flags）来优化 Rust。
- **基准测试经受考验**：另一位成员的后续报告显示，在不使用任何特定硬件优化、仅使用 "--release" 的情况下，Rust 代码的运行时间为 **每个元素 0.31 纳秒**。
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1230537240954933299)** (15 messages🔥): 

- **Nightly/Mojo 更新困扰**：一位成员在尝试更新 **Nightly/Mojo** 时遇到错误，收到 *'No such file or directory'* 消息。他们几天前使用相同的命令成功更新过，且期间未进行任何更改。
- **仅靠清理是不够的**：另一位用户遇到了类似的更新错误，并报告说运行 `modular clean` 无法解决问题，但通过 `brew upgrade modular` 升级 `modular` 后问题得以解决。
- **需要更新 Modular CLI**：有建议称更新 modular CLI 可以解决这些问题，一位成员在遵循此建议后成功完成了更新。
- **路径更新解决最后障碍**：在升级 `modular` 后，一位成员必须手动更新其 `.zshrc` 中的路径 `"/users/carlcaulkett/.modular/pkg/packages.modular.com_nightly_mojo/bin"` 才能彻底解决问题。
- **对人为因素的承认**：对话以幽默的方式结束，承认 **Layer 8**（指计算机网络 OSI 模型中的用户或人为错误）问题永远是一个因素。
  

---



**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1230409686059188296)** (157 messages🔥🔥): 

- **模型工具使用说明**：成员们分享了资源并澄清了 **Command R 模型**工具调用（tool calling）的使用方法，提供了[官方文档](https://docs.cohere.com/docs/tool-use)链接和一个[带有示例的 notebook](https://github.com/cohere-ai/notebooks/blob/main/notebooks/Vanilla_Tool_Use.ipynb)。会议确认可以使用 *JSON schemas* 来描述 **Command 模型**的工具。
- **RAG 与数据库集成问题**：回答了关于将数据库连接到 **Cohere AI** 以及 **检索增强生成 (RAG)** 功能的查询，并引用了 Langchain、RagFlow 以及官方 [Cohere 文档](https://docs.cohere.com/docs/retrieval-augmented-generation-rag)。
- **MySQL 与 Cohere 集成咨询**：一位成员询问如何将 **MySQL** 与 **Cohere** 集成，并寻求关于在不使用 Docker 的情况下是否可行的澄清。讨论中提供了一个 [GitHub 仓库](https://github.com/cohere-ai/quick-start-connectors/tree/main/mysql)链接，并指出官方文档中可能存在过时信息。
- **Command R/R+ 在边缘设备上的商业用途**：澄清了 **Command R** 和 **Command R+** 由于 **CC-BY-NC 4.0** 许可限制（规定仅限非商业用途），不能用于商业目的。
- **模型部署对话**：社区成员讨论了大模型的部署，包括 70B 级模型以及为个人用途部署 100B+ 模型的可行性。在此背景下提到了 **双 A100 40GB** 和 **MacBook M1** 等特定硬件。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arc.net/l/quote/artdceqi">来自 “Chat” 的引用</a>：未找到描述</li><li><a href="https://docs.cohere.com/docs/tool-use">使用 Cohere 模型进行工具调用 - Cohere 文档</a>：未找到描述</li><li><a href="https://docs.cohere.com/docs/retrieval-augmented-generation-rag">检索增强生成 (RAG) - Cohere 文档</a>：未找到描述</li><li><a href="https://docs.cohere.com/docs/creating-and-deploying-a-connector">创建并部署连接器 - Cohere 文档</a>：未找到描述</li><li><a href="https://github.com/cohere-ai/quick-start-connectors/tree/main/mysql">quick-start-connectors/mysql at main · cohere-ai/quick-start-connectors</a>：此开源仓库提供了将工作场所数据存储与 Cohere 的 LLM 集成的参考代码，使开发人员和企业能够执行无缝的检索增强生成...</li><li><a href="https://ai.meta.com/blog/meta-llama-3/">无标题</a>：未找到描述
</li>
</ul>

</div>
  

---


**Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1230453539722104833)** (3 messages):

- **讨论 jailbreaks 日益增长的重要性**：一位成员强调了大型语言模型 (LLMs) 中 **jailbreaks** 不断演变的性质，其风险远高于简单的脏话，可能导致具有严重影响的 Agent 行为，例如破坏公司数据库或骚扰个人。
- **模型与监控工具的集成**：另一位成员详细介绍了他们将模型与监控工具集成的方法。他们遵循了脚手架指令，使用一个循环，其中 **llm_output** 输入到 **run_tool**，结合两者的输出来增强对话过程。
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1230581696307662889)** (5 messages): 

- **使用开源工具构建 RAG**：来自 @elastic 的一篇博客文章，利用 @ollama 和 @MistralAI，阐述了如何使用 Elasticsearch 和 LlamaIndex 构建 RAG (Retrieval Augmented Generation) 应用，展示了这些开源且免费组件的集成。详细指南可以在[随附的博客文章](https://t.co/QqLdz5lojV)中找到。

- **LlamaIndex 对 Meta Llama 3 的首日支持**：@ravithejads 和 @LoganMarkewich 的合作展示了一个关于如何使用 Meta 新发布的 **Llama 3** 模型的 "cookbook"，详细说明了从基础 Prompt 到完整 RAG 流水线开发的流程，可直接通过 @huggingface 获取。该指南可通过此 [Twitter 更新](https://t.co/RMB7MhXIOA)访问。

- **在 Ollama 本地环境运行 Llama 3**：Ollama 提供了简单的命令来在本地运行 **Llama 3**，正如更新后的 Notebook 所演示的那样，新模型只需将 "llama2" 更改为 "llama3" 即可。感兴趣的各方可以按照[此处](https://t.co/jjtpFOzNOS)详述的过程进行操作。

- **与 @TechWithTimm 一起创建代码编写 Agent**：LlamaIndex 与 @TechWithTimm 的最新项目提供了一个教程，关于如何创建一个利用文档编写代码的 Agent，使用 @ollama 的本地 LLMs 和用于文档解析的 LlamaParse。在他们[最近的合作](https://t.co/d6dHazOK93)中了解更多关于编程智能 Agent 的信息。

- **使用 Llama-3 开发本地 RAG 应用**：一个关于使用 MetaAI 的 **Llama-3** 在本地构建 RAG 应用的实用资源已经发布，引导用户查看该过程的全面指南。此操作指南在[发布的链接](https://t.co/clUtzm695i)中进行了概述。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://t.co/RMB7MhXIOA">Llama3 Cookbook - LlamaIndex</a>: 未找到描述</li><li><a href="https://t.co/jjtpFOzNOS">Ollama - Llama 2 7B - LlamaIndex</a>: 未找到描述</li><li><a href="https://t.co/QqLdz5lojV">使用 LlamaIndex, Elasticsearch 和 Mistral 实现 RAG (Retrieval Augmented Generation) — Elastic Search Labs</a>: 学习使用 LlamaIndex, Elasticsearch 和本地运行的 Mistral 实现 RAG 系统。
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1230408031439683606)** (118 messages🔥🔥): 

- **Google Vertex 多模态嵌入难题**：一位参与者对使用 Pinecone 复制 Google [演示网站](https://ai-demos.dev/)的功能感到好奇。他们询问 Google 的模型如何处理部分和拼写错误的查询（如 "timbalands"），并寻求实现类似功能的指导。
- **构建一个提供食谱服务的 LLM 仪表板**：讨论了创建一个交互式仪表板，该仪表板使用输入的食材通过 RAG 生成食谱，包括直接访问所引用食谱的 PDF。
- **Milvus 作为 LlamaIndex 的 VectorDB**：一位用户在 LlamaIndex 中使用 Milvus 作为向量数据库时遇到了异常，并通过显式设置带有 "metric_type" 的 `search_config` 解决了问题。
- **对 ChatResponse 对象的困惑**：用户分享了正确访问 ChatResponse 对象中消息的技巧，这反映了在与聊天模型交互时的一个常见摩擦点。
- **对 LlamaIndex 安全策略的担忧**：一位参与者表示难以找到 LlamaIndex 的安全策略信息，特别是关于使用 LlamaParse 时的数据安全。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.secinsights.ai/">未找到标题</a>: 未找到描述</li><li><a href="https://ai-demos.dev/">AI Demos</a>: 未找到描述</li><li><a href="https://ts.llamaindex.ai/modules/llms/#azure-openai">Large Language Models (LLMs) | LlamaIndex.TS</a>: LLM 负责读取文本并针对查询生成自然语言响应。默认情况下，LlamaIndex.TS 使用 gpt-3.5-turbo。</li><li><a href="http://localhost:19530",">未找到标题</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/use_cases/agents/">Agents - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/7b52057b717451a801c583fae7efe4c4ad167455/llama-index-integrations/vector_stores/llama-index-vector-stores-milvus/llama_index/vector_stores/milvus/base.py#L162">llama_index/llama-index-integrations/vector_stores/llama-index-vector-stores-milvus/llama_index/vector_stores/milvus/base.py at 7b52057b717451a801c583fae7efe4c4ad167455 · run-llama/llama_index</a>: LlamaIndex 是一个用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/querying/pipeline/usage_pattern/?h=intermediate#i">Usage Pattern - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/aa13d47444692faa06b5753b7451b1920837b29c/llama-index-core/llama_index/core/selectors/pydantic_selectors.py#L97">llama_index/llama-index-core/llama_index/core/selectors/pydantic_selectors.py at aa13d47444692faa06b5753b7451b1920837b29c · run-llama/llama_index</a>: LlamaIndex 是一个用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/pull/12964">由 armoucar-neon 提交的让 PydanticSingleSelector 支持 async api 的 Pull Request #12964 · run-llama/llama_index</a>: 描述：为 PydanticSingleSelector 实现 select 方法的 async 版本。新包？我是否填写了 pyproject.toml 中的 tool.llamahub 部分，并为 m... 提供了详细的 README.md</li><li><a href="https://github.com/run-llama/llama_parse">GitHub - run-llama/llama_parse: 为优化 RAG 解析文件</a>: 为优化 RAG 解析文件。通过在 GitHub 上创建账号为 run-llama/llama_parse 的开发做出贡献。</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/querying/pipeline/usage_pattern/?h=intermediate#intermediate-outputs">Usage Pattern - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/response_synthesizers/tree_summarize/?h=tree+summarize">Tree Summarize - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1230647013188702258)** (2 条消息): 

- **追踪 LlamaIndex 和 Zep**: 一位成员询问现在追踪 **LlamaIndex** 和 **Zep** 是否为时过早，表示对这些项目的发展感兴趣。另一位成员对提醒表示感谢，并确认在注意到 **LlamaIndex** 已筹集资金后，打算将两者都加入追踪列表。
  

---



**DiscoResearch ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/1230461828509991014)** (8 条消息🔥): 

- **Mixtral SFT 训练完成**: [Mixtral-8x22B-v0.1-Instruct-sft-en-de](https://huggingface.co/maxidl/Mixtral-8x22B-v0.1-Instruct-sft-en-de) 是一个在英文和德文指令数据混合上训练的全量 SFT 模型。还有一个经过 ORPO 训练的替代方案：[Mixtral-8x22B-v0.1-capybara-orpo-en-de](https://huggingface.co/maxidl/Mixtral-8x22B-v0.1-capybara-orpo-en-de)。
- **等待 Mixtral 模型的基准测试**: 用户提到计划在有时间时尽快对新训练的 Mixtral 模型进行基准测试。
- **大模型训练的技术挑战**: 一位用户在使用 MixtralSparseMoeBlock 时遇到了形状错误，并在 32 GPU 设置上遇到了显存溢出 (OOM) 问题，怀疑参数和优化状态在混合精度中可能未被正确处理。
- **大模型的评估技术**: 讨论了使用 eval-harness 和 device_map="auto" 来评估像 Mixtral 这样的大模型，以及 vllm 和来自 Hugging Face 的 Lighteval 等替代方案。
- **质疑 Mixtral 的 Router Aux Loss Coef**: 有一场关于 Mixtral 模型配置中 "router_aux_loss_coef" 参数的对话，推测其值是否可能是模型性能的关键因素，以及是否需要调整。

**提到的链接**: <a href="https://huggingface.co/maxidl/Mixtral-8x22B-v0.1-Instruct-sft-en-de">maxidl/Mixtral-8x22B-v0.1-Instruct-sft-en-de · Hugging Face</a>: 未找到描述

  

---


**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1230549179814903828)** (21 条消息🔥):

- **Meta 发布 Llama 3**：Meta 推出了 [Meta Llama 3](https://ai.meta.com/blog/meta-llama-3/)，这是一个开源的大语言模型（LLM）。Llama 3 将在多个云平台和硬件上可用，凭借具有 128k token 的新 Tokenizer 提升了性能，在多语言使用（multilingual use）方面非常高效。

- **了解 Llama 3 的多语言能力**：虽然 Llama 3 包含超过 5% 涵盖 30 种语言的非英语高质量数据，但 *“我们并不期望这些语言的性能能达到与英语相同的水平”*，这表明在多语言应用中可能存在局限性。

- **寻找新的 Tokenizer**：一位成员询问如何下载新的 Tokenizer，另一位成员建议它可能在 Hugging Face (HF) 上，正等待访问审批。随后的一位参与者提到，官方 HF 仓库的访问权限可以立即获得批准。

- **新 Llama 3 模型亮相**：提到了 Llama 3 的其他模型，如 8B instruct 和 8B Guard，引发了人们对其 15 万亿 token 数据集来源的好奇。

- **对 Llama 3 局限性的反馈与批评**：一位成员分享了[一条批评 Llama 3 的推文](https://fxtwitter.com/xlr8harder/status/1780992684062024138?s=19)，指责其对模型输出的下游使用施加了限制，强调了对开源开发和贡献的担忧。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/xlr8harder/status/1780992684062024138?s=19">来自 xlr8harder (@xlr8harder) 的推文</a>：Llama 3 发布了，但似乎仍然只有 @MistralAI 真正支持我们：它仍然对模型输出有下游使用限制。@AIatMeta 这是一个损害开源的垃圾限制...</li><li><a href="https://ai.meta.com/blog/meta-llama-3/">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1230590729235992696)** (26 messages🔥): 

- **语言能力测试**：早期测试表明，**Llama3 DiscoLM German** 模型需要进行微调（finetuning）才能在德语熟练度上与 **Mixtral** 模型相媲美。一位成员提到，尽管模型能理解指令，但语法表现不佳，这表明可能需要额外的 **orpo training**。
  
- **可能会有一些 Bug**：一位成员讨论了 **Llama3 DiscoLM German 8b v0.1 Experimental** 模型存在的特殊 token 问题，并指出在使用时应设置 `skip_special_tokens=true`。演示和模型可在 [Hugging Face](https://huggingface.co/DiscoResearch/Llama3_DiscoLM_German_8b_v0.1_experimental) 和 [Gradio](https://364b61f772fa7baacb.gradio.live/) 上获取，但由于 token 生成问题，请谨慎使用。

- **德语 RAG 评估存疑**：一位成员分享的 **Llama3 DiscoLM German 8b v0.1** 的 RAG 评估结果显示，各项任务的准确率低于预期。另一位成员建议检查数据集中的换行符修复（new line fix），以及在 `tokenizer_config.json` 中发现的与 **bos** 和 **eos** token 相关的异常配置。

- **训练数据讨论**：关于 **Llama3 DiscoLM German 模型** 是否恰当地符合训练数据要求引发了辩论。一些人认为当前的 Tokenizer 配置可能有害，而一位成员提到了关于最佳实践的持续讨论，并提供了一个 [Twitter 线程](https://twitter.com/_philschmid/status/1781375157958766607)作为参考。

- **Meta 与 DiscoLM 性能对比**：德语 RAG 评估结果表明，与 **meta-llama/Meta-Llama-3-8B-Instruct** 模型的结果相比，**Llama3 DiscoLM German 8b v0.1 Experimental** 的训练可能降低了其 RAG 能力。成员们正在调查性能差异背后的原因。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/maxidl/Mistral-7B-v0.1-capybara-orpo-en-de">maxidl/Mistral-7B-v0.1-capybara-orpo-en-de · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/DiscoResearch/Llama3_DiscoLM_German_8b_v0.1_experimental">DiscoResearch/Llama3_DiscoLM_German_8b_v0.1_experimental · Hugging Face</a>：未找到描述</li><li><a href="https://364b61f772fa7baacb.gradio.live/">Llama 3 DiscoLM German 8b (Experimental) Demo</a>：未找到描述
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1230519853014585405)** (18 messages🔥):

- **ESP32 需要网络连接 LLM**：一位成员提到 ESP32 硬件需要连接 wifi 才能与 Language Model (LLM) 连接以正常运行。
- **寻求 OpenInterpreter 使用方面的帮助**：一位成员请求解决一个问题，并分享了一个 [Discord 链接](https://discord.com/channels/1146610656779440188/1194880263122075688/1230520200344899615) 作为背景，但未提供更多细节。
- **CLI 解释器的文件创建异常**：在使用 OpenInterpreter 执行系统任务和从命令行界面操作 Kubernetes 时，一位成员遇到了文件创建问题，输出被错误地用 `echo` 包裹。
- **Ollama 3 的兴奋与 TTS/STT 模型咨询**：讨论重点在于对 Ollama 3 性能的兴奋，一位成员正在实验 8b 模型，而另一位成员则询问如何更改 text-to-speech (TTS) 和 speech-to-text (STT) 模型以获得更快的响应。
- **本地 OS 模式设置与微调见解**：成员们分享了如何使用 OpenInterpreter [在本地设置 OS 模式](https://docs.openinterpreter.com/language-models/local-models/lm-studio)，并提供了一个 [Colab notebook](https://colab.research.google.com/drive/1WKmRXZgsErej2xUriKzxrEAXdxMSgWbb?usp=sharing) 作为通用指南。他们还讨论了使用小数据集微调 Mixtral 或 LLama 等模型以实现快速学习。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.openinterpreter.com/language-models/local-models/lm-studio),">Introduction - Open Interpreter</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1WKmRXZgsErej2xUriKzxrEAXdxMSgWbb?usp=sharing">Google Colaboratory</a>：未找到描述
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1230513559176220723)** (17 messages🔥): 

- **Poetry 安装难题**：一位成员询问完成 **poetry install 步骤**所需的时间，但在最初的困惑后自行解决了问题。
- **M5Atom 连接挑战**：对话强调了将 **M5Atom** 连接到各种系统的困难；用户报告尝试了多种方法，包括使用 Arduino IDE 和 PlatformIO，但在音频和服务器连接方面遇到了问题。
- **O1 兼容性问题**：成员们询问了 **O1 与 Windows 的兼容性**，并分享了不同的经验，其中一位用户选择购买 MacBook 来运行 O1，同时在 Windows 上运行其他模型。
- **Llama 3 驱动 O1 平台**：成员们对 **O1 由 Llama 3 驱动**表示了极大的热情。
- **M5Atom 音频传输故障**：一位用户详细说明了在多次安装尝试和故障排除后，使用 M5Atom 向服务器传输音频方面取得了一些进展，但遇到了与消息格式相关的 **BadRequestError**。
  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/)** (1 messages): 

kieguin: https://huggingface.co/spaces/ysharma/Chat_with_Meta_llama3_8b
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1230496708719415369)** (28 messages🔥): 

- **Runnable 类用法**：成员们讨论了在 LangChain 中使用 `RunnableWithMessageHistory`，通过 Python 示例详细说明了其应用，并强调在 invoke 配置中始终包含 `session_id` 以管理聊天历史记录的要求。如需更深入的指导，可以在其 [GitHub 仓库](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/runnables/history.py) 探索 LangChain 代码库和单元测试。
  
- **使用 LangChain 学习 RAG**：一位成员分享了他们利用 LangChain 摄取和检索链实现的 RAG 系统，并向其他人推荐了一个关于如何构建此类系统的有用 [YouTube 播放列表](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x)，同时提供了其 [VaultChat](https://github.com/aosan/VaultChat) 的 GitHub 链接。

- **使用 LLM 处理日期**：关于从大语言模型 (LLM) 获取第二天日期的查询，通过 LangChain 文档中的示例得到了回答，展示了利用 `AmazonAPIGateway` LLM 解决此类提示的 Python 代码示例。

- **Vertex AI 与 Claude 3 集成咨询**：一位成员询问了 LangChain 通过 Google Vertex AI 对 Claude 3 的支持情况，建议根据 [文档](https://api.js.langchain.com/classes/langchain_anthropic.ChatAnthropic.html#apiUrl) 将正确的 API URL 输入到 `apiUrl` 参数中。

- **学习 LLAMA 3 功能**：分享了一个名为 "Learn How LLAMA 3 Works Now: The Complete Beginner’s Guide" 的视频指南，以帮助那些有兴趣了解 LLAMA 3 的用户 [在 YouTube 上观看](https://youtu.be/r-heqmMYNL0)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://<api_gateway_id>.execute-api.<region>.amazonaws.com/LATEST/HF">">未找到标题</a>: 未找到描述</li><li><a href="https://api.js.langchain.com/classes/langchain_anthropic.ChatAnthropic.html#apiUrl">ChatAnthropic | LangChain.js - v0.1.34</a>: 未找到描述</li><li><a href="https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x">RAG From Scratch</a>: 检索增强生成（或 RAG）是将 LLM 与外部数据源连接的通用方法。本视频系列将建立对...的理解</li><li><a href="https://github.com/aosan/VaultChat">GitHub - aosan/VaultChat: get knowledge from your private documents</a>: 从你的私有文档中获取知识。通过在 GitHub 上创建账号来为 aosan/VaultChat 的开发做出贡献。</li><li><a href="https://js.langchain.com/docs/use_cases/chatbots/tool_usage#conversational-responses>)">Tool usage | 🦜️🔗 Langchain</a>: 本节将介绍如何创建对话式 Agent：可以使用工具与其他系统和 API 交互的聊天机器人。</li><li><a href="https://python.langchain.com/docs/expression_language/how_to/message_history#in-memory>)">Add message history (memory) | 🦜️🔗 LangChain</a>: RunnableWithMessageHistory 允许我们向特定的...添加消息历史
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1230636046220267691)** (2 条消息): 

- **寻求客户端反馈集成**：一位用户询问了关于如何使用 **JavaScript** 在客户端通过 **langserve** 添加反馈的教程。频道内未提供教程链接或进一步指导。

- **关于为 RAG 动态上传 PDF 到 API 的咨询**：另一位成员寻求一种将 PDF 文件动态上传到 API 以进行**检索增强生成 (RAG)** 的方法。频道内未讨论任何解决方案或建议。
  

---


**LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1230899322506313851)** (1 条消息): 

- **寻找 FstAPI 路由代码**：一位成员询问如何在模板中定位 FstAPI 路由的代码，特别是针对 **pirate-speak**。他们在应用程序文件夹中查找所有路由的代码时遇到困难，并向社区寻求指导。
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1230555283621806143)** (3 条消息): 

- **提示工程 (Prompt Engineering) 课程发布**：一门关于使用 LangChain 进行提示工程的新课程现已在 LinkedIn Learning 上线。课程详情和访问链接请见[此处](https://www.linkedin.com/feed/update/urn:li:activity:7186761950138109952/)。

- **Llama 3 可用性公告**：Llama 3 现已托管，供有兴趣实验的人员使用。聊天界面可通过 [Llama 3 Chat](https://chat.tune.app/) 访问，API 可在 [Llama 3 API](https://studio.tune.app/) 获取。

- **介绍 Tripplanner Bot**：新创建的 **Tripplanner Bot** 结合使用 LangChain 和免费 API 来提供位置信息、探索景点，并规划具有多个途经点和交通方式的路线。访问 [GitHub 仓库](https://github.com/abhijitpal1247/TripplannerBot) 了解更多详情，并欢迎提出建议或批评。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://chat.tune.app/">Tune Chat - Chat app powered by open-source LLMS</a>: 通过 Tune Chat，访问提示词库、PDF 聊天和品牌声调功能，以增强您的内容写作和分析，并在所有创作中保持一致的语调。</li><li><a href="https://studio.tune.app/">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/abhijitpal1247/TripplannerBot">GitHub - abhijitpal1247/TripplannerBot: This a streamlit app with langchain. It makes use of Bing maps API, OpenStreetMaps API and FourSquare API.</a>: 这是一个使用 LangChain 的 Streamlit 应用。它利用了 Bing maps API、OpenStreetMaps API 和 FourSquare API。 - abhijitpal1247/TripplannerBot
</li>
</ul>

</div>
  

---



**Alignment Lab AI ▷ #[ai-and-ml-discussion](https://discord.com/channels/1087862276448595968/1087876677603958804/1230426347789357127)** (3 条消息): 

- **垃圾信息警报**：该频道出现了推广不当内容并带有外部网站链接的消息；这些消息可能来自机器人或虚假账号，应予以忽略并举报。
- **LLAMA 3 模型解析**：分享了一个关于 **LLAMA 3 模型**（AI 中的 Transformer 架构）的初学者指南视频。该视频标题为“立即学习 LLAMA 3 的工作原理：完整的初学者指南”，旨在为新手简化对该模型运作方式的理解。[在此观看视频](https://youtu.be/r-heqmMYNL0)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://discord.gg/rj9aAQVQFX">Discord - 与好友和社区聊天的新方式</a>：Discord 是通过语音、视频和文本进行交流的最简单方式。在这里聊天、聚会，与你的好友和社区保持紧密联系。</li><li><a href="https://youtu.be/r-heqmMYNL0">立即学习 LLAMA 3 的工作原理：完整初学者指南</a>：深入探索 LLAMA 3 模型的迷人世界，这是一种正在树立机器学习新标准的尖端 Transformer 架构。本指南...
</li>
</ul>

</div>
  

---


**Alignment Lab AI ▷ #[programming-help](https://discord.com/channels/1087862276448595968/1087876753462136873/1230426451946508360)** (2 条消息): 

- **不当内容警报**：发布了一条包含据称指向涉及未成年人的*不当内容*链接的消息，并提到了 OnlyFans。这类内容很可能违反了 Discord 的服务条款和社区准则。

**提到的链接**：<a href="https://discord.gg/rj9aAQVQFX">Discord - 与好友和社区聊天的新方式</a>：Discord 是通过语音、视频和文本进行交流的最简单方式。在这里聊天、聚会，与你的好友和社区保持紧密联系。

  

---


**Alignment Lab AI ▷ #[looking-for-collabs](https://discord.com/channels/1087862276448595968/1095393077415383261/1230426509903400983)** (2 条消息): 

- **不当内容警报**：该频道收到了一条推广 **NSFW 内容** 的消息，具体涉及“Hot Teen & Onlyfans Leaks”。消息包含一个 Discord 邀请链接 (https://discord.gg/rj9aAQVQFX)。

**提到的链接**：<a href="https://discord.gg/rj9aAQVQFX">Discord - 与好友和社区聊天的新方式</a>：Discord 是通过语音、视频和文本进行交流的最简单方式。在这里聊天、聚会，与你的好友和社区保持紧密联系。

  

---


**Alignment Lab AI ▷ #[landmark-dev](https://discord.com/channels/1087862276448595968/1113327574563692654/1230426707488407662)** (2 条消息): 

- **不当内容警报**：一名用户发布了一条推广 **hot teen & Onlyfans leaks** 的消息，附带一个 Discord 服务器链接，并使用了暗示成人内容的表情符号。该消息显得格格不入，且可能违反了 Discord 的社区准则。

**提到的链接**：<a href="https://discord.gg/rj9aAQVQFX">Discord - 与好友和社区聊天的新方式</a>：Discord 是通过语音、视频和文本进行交流的最简单方式。在这里聊天、聚会，与你的好友和社区保持紧密联系。

  

---


**Alignment Lab AI ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/1230540260845293590)** (6 条消息): 

- **WizardLM-2，现已开源**：Alpin 宣布重新上传了 [WizardLM-2](https://huggingface.co/alpindale/WizardLM-2-8x22B) 模型，并提供了发布 [博客](https://wizardlm.github.io/WizardLM2)、[Hugging Face 和 GitHub 仓库](https://huggingface.co/collections/microsoft/wizardlm-2-661d403f71e6c8257dbd598a) 以及 [arXiv](https://arxiv.org/abs/2304.12244) 上的相关学术论文等资源。社区也被邀请加入他们的 [Discord 服务器](https://discord.gg/VZjjHtWrKs)。
- **Meta Llama 3 的访问受 Meta 隐私政策限制**：Imonenext 链接了 Meta Llama 3 模型，该模型要求同意根据 [Meta 隐私政策](https://www.facebook.com/privacy/policy/) 共享联系信息。Meta Llama 3 的官方文档可在其 [入门页面](https://llama.meta.com/get-started/) 找到。
- **寻求 Meta Llama 3 的访问权限和 Tokenizer**：用户讨论了访问 Meta Llama 3-8B 的问题，Imonenext 表示希望能获得该模型的 Tokenizer。
- **Meta Llama 3 被非官方重新上传**：Nanobitz 报告称一些用户已经重新上传了 Meta Llama 3 模型；提供了一个指向 [Undi95 的 Hugging Face 仓库](https://huggingface.co/Undi95/Meta-Llama-3-8B-hf/tree/main) 的链接以便直接访问。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/alpindale/WizardLM-2-8x22B">alpindale/WizardLM-2-8x22B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B">meta-llama/Meta-Llama-3-8B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/Undi95/Meta-Llama-3-8B-hf/tree/main">Undi95/Meta-Llama-3-8B-hf at main</a>：未找到描述
</li>
</ul>

</div>
  

---


**Alignment Lab AI ▷ #[landmark-evaluation](https://discord.com/channels/1087862276448595968/1118282868595109918/1230426782046617651)** (2 条消息): 

- **流传的不当链接**：一名用户发布了一个**可疑链接**，似乎在宣传与 **OnlyFans 泄露素材** 相关的内容，并艾特了所有人。消息中包含暗示未成年内容的表情符号。

**提到的链接**：<a href="https://discord.gg/rj9aAQVQFX">Discord - 与好友和社区交流的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，与你的好友和社区保持紧密联系。

---

**Alignment Lab AI ▷ #[open-orca-community-chat](https://discord.com/channels/1087862276448595968/1124000038205530182/1230426910593515561)** (3 条消息): 

- **社区聊天中的垃圾信息警报**：一名用户发布了推广成人内容的消息，并提供了一个指向 **Onlyfans Leaks** 的可疑链接。
- **需要管理操作**：另一名成员指出了该垃圾信息事件，并艾特了一位潜在的管理员，建议**可能需要**执行封禁操作。

**提到的链接**：<a href="https://discord.gg/rj9aAQVQFX">Discord - 与好友和社区交流的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，与你的好友和社区保持紧密联系。

---

**Alignment Lab AI ▷ #[leaderboard](https://discord.com/channels/1087862276448595968/1135102537817653308/1230426969963892786)** (2 条消息): 

- **不当内容警报**：一名成员发布了一个推广 **Onlyfans Leaks** 社区的链接，带有暗示性的表情符号和 Discord 邀请 ([https://discord.gg/rj9aAQVQFX](https://discord.gg/rj9aAQVQFX))。该帖子通过链接到潜在的 **NSFW 内容**，似乎违反了社区准则。

**提到的链接**：<a href="https://discord.gg/rj9aAQVQFX">Discord - 与好友和社区交流的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，与你的好友和社区保持紧密联系。

---

**Alignment Lab AI ▷ #[looking-for-workers](https://discord.com/channels/1087862276448595968/1142242166677192774/1230427050658107444)** (2 条消息): 

- **不当内容警报**：发布了一条似乎在推广**成人内容**的消息，并附带了 Discord 服务器邀请。消息内容包含暗示性的表情符号和链接。

**提到的链接**：<a href="https://discord.gg/rj9aAQVQFX">Discord - 与好友和社区交流的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，与你的好友和社区保持紧密联系。

---

**Alignment Lab AI ▷ #[looking-for-work](https://discord.com/channels/1087862276448595968/1142242683339944027/1230427145768013834)** (2 条消息): 

<!-- 没有相关的专业或技术讨论可供总结。消息均为垃圾信息。 -->

**提到的链接**：<a href="https://discord.gg/rj9aAQVQFX">Discord - 与好友和社区交流的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，与你的好友和社区保持紧密联系。

---

**Alignment Lab AI ▷ #[join-in](https://discord.com/channels/1087862276448595968/1143791237669855302/1230427274336010250)** (2 条消息): 

- **不当内容警报**："join-in" 频道包含一条推广**成人内容**的消息，涉及热门青少年和 OnlyFans 泄露内容。该消息包含一个 Discord 邀请链接。

**提到的链接**：<a href="https://discord.gg/rj9aAQVQFX">Discord - 与好友和社区交流的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，与你的好友和社区保持紧密联系。

---

**Alignment Lab AI ▷ #[fasteval-dev](https://discord.com/channels/1087862276448595968/1147528620936548363/1230427320641126420)** (2 条消息): 

- **fasteval-dev 中的垃圾信息警报**：**fasteval-dev** 频道收到了推广与 "Hot Teen & Onlyfans Leaks" 相关的不当内容的消息。消息历史记录中没有其他内容或讨论点。

**提到的链接**：<a href="https://discord.gg/rj9aAQVQFX">Discord - 与好友和社区交流的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，与你的好友和社区保持紧密联系。

---

**Alignment Lab AI ▷ #[qa](https://discord.com/channels/1087862276448595968/1147528698669584424/1230427427692478526)** (2 条消息): 

- **不当内容警报**：发布了一条带有链接的推广**成人内容**的消息，这可能违反了服务器和社区准则。该消息包含暗示性的表情符号，并向所有成员发出了行动号召。

**提到的链接**：<a href="https://discord.gg/rj9aAQVQFX">Discord - 与好友和社区交流的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，与你的好友和社区保持紧密联系。

---

**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1230745812489736203)** (19 条消息🔥):

- **Llama 3 8b 的兼容性与使用**：*richinseattle* 提到 **llamafile-0.7** 可以使用 `-m <model path>` 参数运行模型。然而，Llama 3 Instruct 格式存在 Token 问题；[Reddit 讨论](https://www.reddit.com/r/LocalLLaMA/comments/1c76n8p/comment/l06amy7/) 解释说目前的 llamafile 和 llama.cpp server bin 不支持这些参数。
- **Llama 3 8b 的修复正在进行中**：*llamafile* 的更新正在等待中，将解决 Llama 3 Instruct 的兼容性问题，详见此 [GitHub pull request](https://github.com/ggerganov/llama.cpp/pull/6751)。
- **量化版 Llama 8b 即将发布**：*jartine* 承诺今天在 Llamafile 上发布 Llama 8b 的量化版本，并确认了用户的请求。
- **Llamafile 上的 Meta Llama 3 8B Instruct**：*jartine* 提供了 Meta Llama 3 8B Instruct 可执行权重（也称为 llamafiles）的链接，用于在 [Hugging Face](https://huggingface.co/jartine/Meta-Llama-3-8B-Instruct-llamafile) 上进行测试，并提醒注意目前存在的问题，如停止符 (stop token) 失效。
- **Llamafile 模型的测试与更新**：多位用户确认在不同系统上成功测试了 *Llama 3 8b* 模型；*jartine* 报告称已修复即将推出的 Llama 3 70b 的停止符问题，并指出可能会有一些小 Bug。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/FaradayDotDev/llama-3-8b-Instruct-GGUF/tree/main">FaradayDotDev/llama-3-8b-Instruct-GGUF at main</a>：未找到描述</li><li><a href="https://huggingface.co/jartine/Meta-Llama-3-8B-Instruct-llamafile">jartine/Meta-Llama-3-8B-Instruct-llamafile · Hugging Face</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1c76n8p/comment/l06amy7/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6751">由 DifferentialityDevelopment 添加的 llama-3 聊天模板 · Pull Request #6751 · ggerganov/llama.cpp</a>：这只是简单地添加 Llama 3 聊天模板。
</li>
</ul>

</div>
  

---



**Skunkworks AI ▷ #[finetuning](https://discord.com/channels/1131084849432768614/1131669354912678028/1230862081893466114)** (6 条消息): 

- **Databricks 增强模型推理服务**：Databricks 宣布其 Model Serving [支持 GPU 和 LLM 优化的公开预览](https://www.databricks.com/blog/announcing-gpu-and-llm-optimization-support-model-serving)，简化了 AI 模型的部署，并针对 **LLM 推理实现了零配置优化**。该服务是统一数据和 AI 平台上首个此类 Serverless GPU 产品。

- **创新的代价可能不菲**：一位成员幽默地评论道，Databricks 对 Model Serving 的 GPU 和 LLM 优化支持预计会很 **昂贵**。

- **LLM 微调变得更简单**：Modal 分享了一份 [LLM 微调指南](https://modal.com/docs/examples/llm-finetuning)，其中包括 **LoRA 适配器、Flash Attention、梯度检查点 (gradient checkpointing)** 和 **DeepSpeed** 等先进技术，用于高效的模型权重调整。

- **为预算有限的开发者提供的 Serverless 托管**：可以在 **GitHub 示例** ([modal-examples/06_gpu_and_ml/llm-frontend/index.html](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-frontend/index.html)) 中找到承诺价格亲民的 Serverless 托管方案，该仓库提供了 Serverless GPU 托管解决方案。

- **成员发现了 Serverless 推理解决方案**：一位成员对发现 **Serverless 推理** 选项表示满意，称其符合他们的需求。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://modal.com/docs/examples/llm-finetuning">在几分钟内微调 LLM（包含 Llama 2, CodeLlama, Mistral 等）</a>：厌倦了提示词工程？微调通过调整模型权重以更好地适应特定任务，帮助你从预训练的 LLM 中获得更多收益。这份操作指南将帮助你利用基础模型...</li><li><a href="https://www.databricks.com/blog/announcing-gpu-and-llm-optimization-support-model-serving">使用 Databricks Model Serving 部署私有 LLM | Databricks 博客</a>：在完全控制数据和模型的情况下部署生成式 AI 模型。</li><li><a href="https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-frontend/index.html">modal-examples/06_gpu_and_ml/llm-frontend/index.html at main · modal-labs/modal-examples</a>：使用 Modal 构建的程序示例。通过在 GitHub 上创建一个账户，为 modal-labs/modal-examples 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1230451125996617779)** (2 条消息):

- **Mixtral 设定新标准**：根据 [YouTube 视频](https://www.youtube.com/watch?v=N8U6XnVK2mM)，最新的 **Mixtral 8x22B** 模型被誉为 AI 性能和效率的新基准。该模型利用稀疏 Mixture-of-Experts 理念来推动 AI 社区的发展。
- **初识 Meta Llama 3**：Facebook 的新大型语言模型 **Llama 3** 作为最先进的开源产品发布，详情可见此 [YouTube 视频](https://www.youtube.com/watch?v=zQy11WnAIIc)和 [Meta AI 博客](https://ai.meta.com/blog/meta-llama-3/)。该模型旨在挑战当前大语言技术的极限。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=N8U6XnVK2mM">Mistral 推出的最佳开源模型 Mixtral 8x22B</a>：Mixtral 8x22B 是最新的开源模型。它为 AI 社区的性能和效率设定了新标准。它是一个稀疏 Mixture-of-Experts (SMo...</li><li><a href="https://www.youtube.com/watch?v=zQy11WnAIIc">介绍 Llama 3：最佳开源大型语言模型</a>：介绍 Meta Llama 3，这是 Facebook 下一代最先进的开源大型语言模型。https://ai.meta.com/blog/meta-llama-3/#python #...
</li>
</ul>

</div>
  

---



**LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1230601559465918526)** (6 条消息): 

- **LiteLLM 使用咨询**：一位成员对群组中是否有人使用 **litellm** 表示好奇。
- **动画跌落**：一位成员分享了一个幽默的 [来自 Tenor 的 GIF](https://tenor.com/view/falling-falling-down-stairs-stairs-meme-funny-gif-21363126)，显示一个角色从楼梯上摔下来。
- **Llama vs. Opus - Arena 对决**：据称在某个 Arena 的性能背景下，**Llama 3** 优于 **opus**，且仅凭 **70b** 的参数量就实现了这一点。
- **对误差范围的担忧**：讨论中的误差范围引发了一条带有警告意味的评论。
- **风格 vs. 智能的辩论**：有人推测性能差异是归因于风格变化还是实际的智能。

**提到的链接**：<a href="https://tenor.com/view/falling-falling-down-stairs-stairs-meme-funny-gif-21363126">Falling Falling Down Stairs GIF - Falling Falling Down Stairs Stairs - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---



**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1230684526154481756)** (3 条消息): 

- **Llama 3 新见解引发关注**：一位成员分享了 [Andrej Karpathy 的一条推文](https://twitter.com/karpathy/status/1781028605709234613)，讨论了小模型的潜力，特别是使用 15T 数据集训练 8B 参数模型这种不寻常但受欢迎的方法。推文指出，常见的 LLM 可能被欠训练了 100-1000 倍，鼓励未来发展训练时间更长的小型模型趋势。

- **期待小型模型的普及**：继 Karpathy 推文的见解之后，一位成员对 Llama 3 这样小而强大的模型可能带来的广泛应用表示热切期待。该成员赞赏开发更小版本的呼吁，暗示模型训练正向效率化转变。
  

---


**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1230596139439362149)** (3 条消息): 

- **插件开发的困扰**：一位成员分享了尝试为 **llm** 安装新插件时的回溯错误，显示 `llm_web` 出现 `ModuleNotFoundError`。该问题通过完全卸载并重新安装 llm 得到解决，这表明通过 brew 和 pipx 同时安装可能存在冲突。
  
- **多重安装导致的混乱**：有人建议，安装问题的部分原因可能源于同时通过 brew 和 pipx 安装了 llm，导致在使用 `which llm` 时混淆了正在调用的是哪一个实例。

- **分享有趣的 LLM 使用案例**：频道中转发了一个有趣的 llm 使用案例，并提供了链接供成员探索。[使用案例链接](https://discord.com/channels/823971286308356157/1052680908643258508/1230552137763913779)。
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1230610081700380773)** (2 条消息): 

- **Llama3 发布公告**：一位成员兴奋地注意到 [Llama3](https://llama.meta.com/llama3/) 现已可用。
- **Llama3 显示出速度提升**：他们报告称，在他们正在处理的几个模型中，**Llama3** 比 **PyTorch** 稍快，并确认了在 **XTX** 硬件上与 **ROCm** 的兼容性。

**提到的链接**：<a href="https://llama.meta.com/llama3/">Meta Llama 3</a>：使用 Meta Llama 3 构建 AI 的未来。现已提供 8B 和 70B 的预训练及指令微调版本，以支持广泛的应用。

---

**AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1230653838386532464)** (1 条消息): 

- **Jamba 分布式系统查询**：一位用户在 2x A100 集群上进行 **Jamba 的 long context inference** 时遇到问题，并正在寻求示例代码以帮助解决分布式系统方面的困难。目前尚未提供解决方案或进一步的讨论。