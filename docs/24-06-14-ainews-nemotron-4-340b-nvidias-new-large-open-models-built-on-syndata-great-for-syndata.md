---
companies:
- nvidia
- hugging-face
- mistral-ai
- llamaindex
- cohere
- gemini
- mistral
date: '2024-06-14T21:06:38.944616Z'
description: '**NVIDIA** 已将其 **Nemotron-4** 模型从 **15B** 扩展到了庞大的 **340B** 稠密模型。该模型基于
  **9T token** 训练，实现了与 **GPT-4** 相当的性能。在模型对齐过程中，超过 **98% 的数据为合成数据**，仅使用了约 **2 万条人类标注样本**
  用于微调和奖励模型训练。NVIDIA 已开源其合成数据生成流水线，包括合成提示词（prompts）和偏好数据的生成。


  该模型的基座版（base）和指令版（instruct）性能超越了 **Mixtral** 和 **Llama 3**，而其奖励模型的排名则优于 **Gemini
  1.5**、**Cohere** 和 **GPT-4o**。


  其他值得关注的模型还包括：**Mamba-2-Hybrid 8B**，其速度比 Transformer 架构快达 **8 倍**，且在长文本任务中表现卓越；**Samba-3.8B-instruct**，具有线性复杂度的“无限”上下文长度；针对低资源设备优化的
  **Dolphin-2.9.3** 微型模型；以及拥有 **200K 上下文窗口**、可在 **16GB 显存**上高效运行的 **Faro Yi 9B DPO**。此外，**Mixture-of-Agents（智能体混合）**技术在
  AlpacaEval 2.0 测试中助力开源大语言模型超越了 GPT-4 Omni。'
id: 473597d5-374e-41e6-bf31-37120d4a54e0
models:
- nemotron-4-340b
- mixtral
- llama-3
- gemini-1.5
- gpt-4o
- mamba-2-hybrid-8b
- samba-3.8b-instruct
- dolphin-2.9.3
- faro-yi-9b-dpo
original_slug: ainews-to-be-named-2748
people:
- philipp-schmid
- bryan-catanzaro
- oleksii-kuchaiev
- rohanpaul_ai
- cognitivecompai
- _philschmid
- 01ai_yi
title: Nemotron-4-340B：英伟达（NVIDIA）推出的新型大型开放模型，基于合成数据构建，非常适合用于生成合成数据。
topics:
- synthetic-data
- model-alignment
- reward-models
- fine-tuning
- long-context
- model-scaling
- inference-speed
- mixture-of-agents
- open-source-models
- model-training
- instruction-following
- context-windows
---

<!-- buttondown-editor-mode: plaintext -->**Synthetic Data is 98% of all you need.**

> 2024年6月13日至6月14日的 AI 新闻。
我们为您检查了 7 个 subreddits、[**384** 个 Twitter](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord（**414** 个频道和 **2481** 条消息）。
预计节省阅读时间（以 200wpm 计算）：**280 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

NVIDIA 已完成将 [2 月发布的 Nemotron-4 15B](https://arxiv.org/abs/2402.16819) 扩展到高达 340B 的 dense model。[Philipp Schmid 总结了您需要了解的最佳要点](https://x.com/_philschmid/status/1801651752426524996)：

 
![image.png](https://assets.buttondown.email/images/726a2bf3-47dd-47f4-ba30-a66904bd0e02.png?w=960&fit=max)
 

来自 [NVIDIA 博客](https://blogs.nvidia.com/blog/nemotron-4-synthetic-data-generation-llm-training/)、[Huggingface](https://huggingface.co/collections/nvidia/nemotron-4-340b-666b7ebaf1b3867caf2f1911)、[技术报告](https://research.nvidia.com/publication/2024-06_nemotron-4-340b)、[Bryan Catanzaro](https://x.com/ctnzr/status/1801645786138321251?utm_source=ainews&utm_medium=email)、[Oleksii Kuchaiev](https://x.com/kuchaev/status/1801650219739976023)。

这条 Synthetic Data 流水线值得进一步研究：

> 值得注意的是，我们模型 alignment 过程中使用的 98% 以上的数据都是合成生成的，展示了这些模型在生成 Synthetic Data 方面的有效性。为了进一步支持开放研究并促进模型开发，**我们还将模型 alignment 过程中使用的 Synthetic Data 生成流水线开源**。

以及

> 值得注意的是，在整个 alignment 过程中，我们仅依赖约 20K 的人工标注数据（10K 用于 supervised fine-tuning，10K Helpsteer2 数据用于 reward model 训练和 preference fine-tuning），而我们的数据生成流水线合成了超过 98% 的用于 supervised fine-tuning 和 preference fine-tuning 的数据。

 
![image.png](https://assets.buttondown.email/images/688a4a1c-110d-4c3a-bed4-5a04f1ecd41b.png?w=960&fit=max)
 

[论文](https://d1qx31qr3h6wln.cloudfront.net/publications/Nemotron_4_340B_8T.pdf)中的第 3.2 节提供了关于该流水线的大量精彩细节：

- 合成单轮提示词 (Synthetic single-turn prompts)
- 合成指令遵循提示词 (Synthetic instruction-following prompts)
- 合成双轮提示词 (Synthetic two-turn prompts)
- 合成对话生成 (Synthetic Dialogue Generation)
- 合成偏好数据生成 (Synthetic Preference Data Generation)


[Base 和 Instruct 模型轻松击败了 Mixtral 和 Llama 3](https://x.com/ctnzr/status/1801645787258229020)，但考虑到参数量大出半个数量级，这或许并不令人意外。然而，他们还发布了一个 [Reward Model](https://x.com/ctnzr/status/1801645790840135886) 版本，其排名优于 Gemini 1.5、Cohere 和 GPT 4o。详细披露非常有趣：

 
![image.png](https://assets.buttondown.email/images/a0c5969c-8667-4bd3-bf06-54b766997836.png?w=960&fit=max)
 

且该 RM 取代了 LLM as Judge：

 
![image.png](https://assets.buttondown.email/images/2e947164-eadc-44e1-9b71-15b0221b7a1f.png?w=960&fit=max)
 


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在使用 Haiku 进行聚类和 flow engineering。

**AI 模型与架构**

- **NVIDIA Nemotron-4-340B LLM 正式发布**：[@ctnzr](https://twitter.com/ctnzr/status/1801645786138321251) NVIDIA 发布了 340B dense LLM，性能媲美 GPT-4。**提供 Base、Reward 和 Instruct 模型。在 9T tokens 上训练而成。**
- **Mamba-2-Hybrid 8B 表现优于 Transformer**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1801255162972983609) Mamba-2-Hybrid 8B 在评估任务上超过了 8B Transformer，**预计推理速度快达 8 倍。** 在长上下文任务中达到或超过 Transformer 水平。
- **支持无限上下文的 Samba 模型**：[@_philschmid](https://twitter.com/_philschmid/status/1801516284267356217) Samba 结合了 Mamba、MLP、Sliding Window Attention，以**线性复杂度实现无限上下文长度。** Samba-3.8B-instruct 表现优于 Phi-3-mini。
- **Dolphin-2.9.3 微型模型实力不俗**：[@cognitivecompai](https://twitter.com/cognitivecompai/status/1801423898250031273) Dolphin-2.9.3 0.5b 和 1.5b 模型发布，**专注于 instruct 和对话。可以在智能手表或 Raspberry Pi 上运行。**
- **具备 200K 上下文的 Faro Yi 9B DPO 模型**：[@01AI_Yi](https://twitter.com/01AI_Yi/status/1801597204336878019) Faro Yi 9B DPO 因其**仅需 16GB VRAM 即可支持 200K 上下文而受到赞誉，实现了高效 AI。**

**技术与架构**

- **Mixture-of-Agents (MoA) 提升开源 LLM**：[@llama_index](https://twitter.com/llama_index/status/1801305617878937959) 基于开源 LLM 的 MoA 配置 **在 AlpacaEval 2.0 上超越了 GPT-4 Omni。** 该方案通过堆叠多个 LLM Agent 来精炼回答。
- **Lamini Memory Tuning 实现 95% 的 LLM 准确率**：[@realSharonZhou](https://twitter.com/realSharonZhou/status/1801271891954696317) Lamini Memory Tuning 实现了 **95% 以上的准确率，并将幻觉减少了 10 倍。** 它将开源 LLM 转变为 1M-way adapter MoE。
- **LoRA 微调见解**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1801430394388000956) LoRA 微调论文发现，**使用随机值初始化矩阵 A 并使用零初始化矩阵 B 通常会带来更好的性能。** 这种方式允许使用更大的学习率。
- **Discovered Preference Optimization (DiscoPOP)**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1801611002590396781) DiscoPOP 通过使用 LLM 来提议和评估偏好优化损失函数，其表现优于 DPO。**它使用了逻辑损失（logistic loss）和指数损失（exponential loss）的自适应混合。**

**多模态 AI**

- **用于单目深度估计的 Depth Anything V2**：[@_akhaliq](https://twitter.com/_akhaliq/status/1801432403665125738), [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1801435793254121798) Depth Anything V2 能够生成更精细的深度预测。**该模型在 59.5 万张合成标注图像和超过 6200 万张真实未标注图像上进行了训练。**
- **Meta 的论文《一张图片不仅仅是 16x16 的 Patch》**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1801434306423754776) Meta 的论文显示 Transformer 可以直接处理单个像素而非 Patch，**虽然成本更高，但性能更好。**
- **OpenVLA 开源视觉-语言-动作模型**：[@_akhaliq](https://twitter.com/_akhaliq/status/1801437583039156503) OpenVLA 是一个在机器人演示数据上预训练的 7B 开源模型。**其表现优于 RT-2-X 和 Octo。该模型基于 Llama 2 + DINOv2 和 SigLIP 构建。**

**基准测试与数据集**

- **用于 LLM 时间推理的 Test of Time 基准测试**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1801445359228452964) Google 的 Test of Time 基准测试用于评估 LLM 的时间推理能力，包含约 5000 个测试样本。
- **用于 LLM 计算机科学掌握程度的 CS-Bench**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1801446968897466369) CS-Bench 是一个全面的基准测试，包含约 5000 个样本，涵盖 26 个计算机科学子领域。
- **用于奖励建模的 HelpSteer2 数据集**：[@_akhaliq](https://twitter.com/_akhaliq/status/1801435120059961802) NVIDIA 的 HelpSteer2 是一个用于训练奖励模型的数据集。**这是一个仅包含 1 万个回答对的高质量数据集。**
- **Recap-DataComp-1B 数据集**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1801252085440114790) Recap-DataComp-1B 数据集通过使用 LLaMA-3 为 DataComp-1B 的约 13 亿张图片重新生成标题而获得。**这提升了视觉-语言模型的性能。**

**杂项**

- **Scale AI 专注于贤能（merit）的招聘政策**：[@alexandr_wang](https://twitter.com/alexandr_wang/status/1801331034916851995) Scale AI 正式确定了专注于贤能、卓越和才智的招聘政策。
- **Paul Nakasone 加入 OpenAI 董事会**：[@sama](https://twitter.com/sama/status/1801417160369311977) Paul Nakasone 将军加入 OpenAI 董事会，因其带来的安全和保密专业知识而受到赞赏。
- **Apple Intelligence 在 WWDC 上发布**：[@bindureddy](https://twitter.com/bindureddy/status/1801383879619789210) Apple Intelligence，即 Apple 的 AI 计划，在 WWDC 上正式公布。

**梗与幽默**

- **模拟假设**：[@karpathy](https://twitter.com/karpathy/status/1801311713842893161) Andrej Karpathy 调侃了模拟假设——也许模拟是神经化的、近似的，而非精确的。
- **事后看来 Prompt Engineering 显得很蠢**：[@svpino](https://twitter.com/svpino/status/1801597864633897096) Santiago Valdarrama 指出，与一年前相比，Prompt Engineering 在今天看来显得很蠢。
- **Yann LeCun 谈论 Elon Musk 和火星**：[@ylecun](https://twitter.com/ylecun/status/1801478238210036046) Yann LeCun 调侃 Elon Musk 不戴太空头盔去火星。

---

# AI Reddit 摘要

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**Stable Diffusion 3.0 发布及反应**

- **Stable Diffusion 3.0 medium 模型发布**：在 /r/StableDiffusion 中，Stability AI 发布了 SD3 medium 模型，但许多人对[**结果感到失望，尤其是人体解剖结构方面**](https://www.reddit.com/r/StableDiffusion/comments/1devhvm/stable_diffusion_30_comparison/)。有些人称其与完整的 8B 模型相比简直是[“一个笑话”](https://www.reddit.com/r/StableDiffusion/comments/1deu82b/just_give_us_the_real_sd3_8b_this_is_but_a_joke/)。
- **SD3 中严重的审查和“安全”过滤**：/r/StableDiffusion 社区怀疑 SD3 经过了[**严格审查，导致人体解剖结构表现糟糕**](https://www.reddit.com/r/StableDiffusion/comments/1dexa9l/the_truth_behind_sd3s_censorship_it_is_not_the/)。该模型看起来[“无性、平滑且像孩子一样”](https://www.reddit.com/r/StableDiffusion/comments/1dev363/stable_diffusion_3_medium_asexual_smooth_and/)。
- **强大的提示词遵循度，但存在解剖结构问题**：SD3 对[**许多主题表现出强大的提示词遵循度，但在处理人体姿势时却很吃力**](https://www.reddit.com/r/StableDiffusion/comments/1dfjc90/the_sexualization_of_sd3_is_getting_out_of_hand/)，例如“躺着”。一些人正在[尝试级联训练 (cascade training)](https://www.reddit.com/r/StableDiffusion/comments/1dfdlzx/cascade_training_snapshot/) 以改进结果。
- **T5xxl 文本编码器有助于理解提示词**：一些人发现 [**T5xxl 文本编码器提高了 SD3 对提示词的理解，尤其是对文本的理解**](https://www.reddit.com/r/StableDiffusion/comments/1df3u9t/sd3_text_encoders_comparision_clip_vs_t5xxlclip/)，但这并不能解决解剖结构问题。
- **呼吁社区训练无审查模型**：有人[呼吁社区团结起来训练一个无审查模型](https://www.reddit.com/r/StableDiffusion/comments/1dfe4ly/realistically_could_we_create_a_sustainable/)，因为[对 SD3 进行微调 (fine-tuning) 不太可能完全解决这些问题](https://www.reddit.com/r/StableDiffusion/comments/1df4o0w/lets_not_restart_the_hype_train_again_finetuning/)。然而，这将需要大量的资源。

**AI 进展与未来**

- **中国崛起为科学超级大国**：[《经济学人》报道称 **中国已成为科学超级大国**](https://www.economist.com/science-and-technology/2024/06/12/china-has-become-a-scientific-superpower)，在研究产出和影响力方面取得了重大进展。
- **前 NSA 局长加入 OpenAI 董事会**：[OpenAI 已邀请前 NSA 局长 Paul Nakasone 加入其董事会](https://www.cnbc.com/amp/2024/06/13/openai-adds-former-nsa-chief-to-its-board-paul-nakasone-sam-altman.html)，这引发了 [/r/singularity 社区对其影响的一些担忧](https://www.reddit.com/r/singularity/comments/1dfgr5n/new_openai_board_member_told_nsa_to_spy_on_the/)。

**新的 AI 模型与技术**

- **Samba 混合架构超越 Transformer**：微软推出了 [**Samba，一种具有无限上下文长度的混合 SSM 架构**](https://www.reddit.com/r/LocalLLaMA/comments/1df82vb/samba_simple_hybrid_state_space_models_for/)，在长程任务上表现优于 Transformer。
- **Lamini Memory Tuning 减少幻觉**：Lamini.ai 的 [**Memory Tuning 将事实嵌入到 LLM 中，将准确率提高到 95%**](https://www.reddit.com/r/MachineLearning/comments/1dffyfs/r_laminiai_introduces_memory_tuning_95_llm/)，并将幻觉减少了 10 倍。
- **Mixture of Agents (MoA) 表现优于 GPT-4**：TogetherAI 的 [**MoA 通过利用多个 LLM 的优势，在基准测试中表现优于 GPT-4**](https://www.together.ai/blog/together-moa)。
- **WebLLM 实现浏览器内 LLM 推理**：[**WebLLM 是一款由 WebGPU 加速的高性能浏览器内 LLM 推理引擎**](https://www.reddit.com/r/LocalLLaMA/comments/1df8y7i/webllm_a_highperformance_inbrowser_llm_inference/)，支持客户端 AI 应用。

**AI 硬件与基础设施**

- **三星的“一站式” AI 芯片方案**：[三星通过集成内存、代工和封装方案，将 AI 芯片生产时间缩短了 20%](https://www.reuters.com/technology/artificial-intelligence/samsung-announces-turnkey-approach-ai-chipmaking-2024-06-12/)。他们预计到 2028 年芯片收入将达到 7780 亿美元。
- **Cerebras 晶圆级芯片在 AI 工作负载中表现出色**：[Cerebras 的晶圆级芯片在分子动力学模拟和稀疏 AI 推理任务上表现优于超级计算机](https://spectrum.ieee.org/cerebras-wafer-scale-engine)。
- **处理 TB 级的机器学习数据**：/r/MachineLearning 社区[讨论了在机器学习流水线中处理 TB 级数据的方法](https://www.reddit.com/r/MachineLearning/comments/1desvs5/d_how_to_prepare_tbs_of_data_for_ml_tasks/)。

**梗与幽默**

- **Memes 嘲讽 SD3 的审查和人体构造**：/r/StableDiffusion 子版块充斥着[嘲讽 SD3 糟糕的人体构造和严厉内容过滤的 Memes 和笑话](https://www.reddit.com/r/StableDiffusion/comments/1deuwwz/ok_sai_lets_do_some_basic_pr_damage_control_sack/)，并伴随着[“开除实习生”的呼声](https://www.reddit.com/r/StableDiffusion/comments/1df1ugp/)。

---

# AI Discord 摘要复盘

> 摘要之摘要的摘要

**1. NVIDIA 通过 Nemotron-4 340B 推动性能提升**：

- **[NVIDIA 的 Nemotron-4-340B 模型](https://huggingface.co/nvidia/Nemotron-4-340B-Instruct)**：NVIDIA 新发布的 3400 亿参数模型包括 **Instruct** 和 **Reward** 等变体，旨在实现高效率和更广泛的语言支持，可运行在配备 **8 张 GPU** 且使用 **FP8** 精度的 DGX H100 上。

- 微调 Nemotron-4 340B 面临资源挑战，据估计需要 **40 张 A100/H100 GPU**，尽管推理可能需要较少的资源，大约一半的节点即可。

**2. 基于 UNIX 的系统通过 ComfyUI 处理 SD3**：

- **[SD3 配置与性能](https://www.civitai.com/quickstart-guide-to-stable-diffusion-3/)**：用户分享了在 ComfyUI 中设置 Stable Diffusion 3 的过程，包括下载文本编码器，并建议将分辨率调整为 **1024x1024**，以便在动漫和写实渲染中获得更好的效果。

- 针对 SD3 模型人体构造准确性的意见分歧，促使人们呼吁 Stability AI 在未来的更新中解决这些缺陷，这反映了关于社区预期与模型局限性的持续讨论。

**3. 识别并解决 GPU 兼容性问题**：

- **[CUDA 和 Torch 运行时异常](https://gist.github.com/mobicham/ab439330b4fb9c6f1d4086e54e4142c0)**：在 **Ada GPUs** 上出现的 `torch.matmul` 异常引发了对比 RTX 4090 等 GPU 的测试，结论是 **CUDA 版本和基准测试**会影响性能，并希望 PyTorch 团队能给出澄清。

- 在 Mojo 中建立的**多线程协议**有望带来性能提升。分享的见解认为，**Mojo 的结构化内存处理**使其从长远来看可以作为 CUDA 的可靠替代方案。

**4. API 不一致性令用户沮丧**：

- **[Perplexity API 和服务器宕机](https://www.reddit.com/r/perplexity_ai/s/pG41duXQBu)**：用户报告了频繁的**服务器宕机**和功能损坏（如文件上传和链接生成错误），导致持续的沮丧感，并对升级到 **Perplexity Pro** 的价值产生怀疑。

- **[LangChain 和 pgvector 集成](https://python.langchain.com/v0.2/docs/integrations/vectorstores/pgvector/)**：尽管遵循了文档，但在识别导入（imports）时仍遇到了问题，这突显了挑战，建议仔细设置 Python 环境以确保无缝集成。

**5. 社区努力与资源管理**：

- **DiscoPOP 优化**：Sakana AI 的方法 [DiscoPOP](https://sakana.ai/llm-squared) 声称具有卓越的偏好优化能力，在承诺高性能的同时，与基础模型的偏差极小。

- **扩展训练工作**：围绕处理检索增强生成（RAG）的大规模数据集的社区讨论强调了**分块（chunking）、索引和查询分解**，以改进模型训练并管理超过 8k tokens 的上下文长度。

---

# 第 1 部分：高层级 Discord 摘要

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Diffusion 3 Discord 机器人向全球分享**：一个使用 [Stable Diffusion 3](https://github.com/RocketGod-git/stable-diffusion-3-discord-bot) 根据提示词生成图像的 Discord 机器人已开源，为用户提供了视觉内容创作的新工具。
  
- **微调 SDXL 以获得更好的动漫角色渲染**：用于写实和动漫生成的 SDXL 训练面临挑战，产生的结果过于卡通化；建议使用 1024x1024 分辨率以改善效果。

- **引导 ComfyUI 设置 SD3**：提供了在 ComfyUI 中设置 Stable Diffusion 3 的协助，涉及下载文本编码器等步骤，指南源自 [Civitai 的快速入门指南](https://education.civitai.com/quickstart-guide-to-stable-diffusion-3/)。

- **社区对 SD3 人体构造准确性产生分歧**：讨论集中在 SD3 明显的局限性上，特别是在人体构造渲染方面，强调了用户呼吁 Stability AI 解决并沟通这些问题的解决方案。

- **SD3 的 LoRA 层训练处于停滞状态**：对话涉及为 SD3 训练新的 LoRA，指出缺乏高效的工具和工作流，用户期待未来的更新能增强功能。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**处理器对决：MilkV Duo vs. RK3588**：工程师们对比了 **MilkDuo 64MB 控制器**与 **RK3588 的 6.0 TOPs NPU**，引发了关于硬件能力与 [sophgo/tpu-mlir 编译器](https://github.com/sophgo/tpu-mlir) 优化实力的讨论。他们分享了技术细节和基准测试，引发了对 MilkDuo 性能优势实际来源的好奇。

**Triton 3.0 效应**：Triton 3.0 新的形状操作（shape manipulation）和解释器中的错误修复是热门话题。同时，一位用户在实现低比特（low-bit）Kernel 时遇到了 `LLVM ERROR: mma16816 data type not supported` 错误，这引发了关于关注 [Triton GitHub repo](https://github.com/triton-lang/triton/issues/2113) 持续更新的建议。

**PyTorch 神秘的矩阵数学**：`torch.matmul` 的异常导致了在不同 GPU 上的基准测试，在 Ada GPU 上观察到的性能提升激发了对 PyTorch 团队提供更深层见解的渴望，正如分享的 [GitHub Gist](https://gist.github.com/mobicham/ab439330b4fb9c6f1d4086e54e4142c0) 中所强调的那样。

**CUDA 使用 C++，Triton 作为替代方案**：在社区内，编写高性能 CUDA Kernel 对 C/C++ 的需求得到了确认，同时强调了 Triton 由于与 PyTorch 的集成以及简化内存处理的能力，在 ML/DL 应用中的适用性日益增强。

**Tensor Cores 驱动 INT8 性能**：#bitnet 频道的讨论集中在利用 Tensor Cores 的 INT8 操作实现性能目标。实证反馈显示，在 A100 GPU 上，大尺寸矩阵可获得高达 7 倍的加速，但随着 Batch Size 增大，收益递减。讨论详细审视了 Tensor Cores 在各种尺寸矩阵和 Batch 操作中的性能角色，指出了 INT8 与 FP16/BF16 之间的效率差异以及 `wmma` 限制的影响。

**Discord 讨论的线程化挑战**：成员们表达了在 Discord 上追踪讨论的挑战，表示更倾向于使用论坛作为信息存储库，并提倡使用线程（threading）和回复作为在实时频道中更好管理对话的策略。

**Meta 训练与推理加速器 (MTIA) 兼容性**：MTIA 对 Triton 的兼容性受到关注，这标志着 AI 模型开发阶段对流线化编译过程的兴趣。

**为新架构考虑 Triton**：在 #torchao 频道中，torch.nn 在添加新模型方面的保守与 AO 对促进特定新架构的开放态度形成了对比，这表明了选择性的模型支持和潜在的速度提升。

**编码困境与社区协作编码**：协作立场显而易见，成员们讨论了改进和合并复杂的 Pull Requests (PRs)、调试和手动测试，特别是在 #llmdotc 的多 GPU 环境下。多线程对话强调了与 ZeRO-2 待集成相关的准确梯度范数（gradient norms）和权重更新条件的复杂性。

**定制位运算蓝图**：提议通过实时代码审查会议来揭开晦涩 PR 进展的神秘面纱，#bitnet 社区剖析了标量量化（scalar quantization）对性能的影响，揭示了在大矩阵上平均 5-6 倍的提升，以及收益对 Batch Size 的敏感性，并附带了深入研究的资源：[BitBLAS 性能分析](https://github.com/microsoft/BitBLAS/raw/main/images/figures/op_benchmark_a100_int2_scaling.png)。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth AI 激发 ASCII 艺术粉丝俱乐部**：社区对 Unsloth 的 ASCII 艺术表现出极大的赞赏，并幽默地建议艺术形式应随着训练失败而演变。
- **DiscoPOP 赢得优化粉丝**：来自 Sakana AI 的新 **DiscoPOP** 优化器因其有效的偏好优化功能而备受关注，详见[博客文章](https://sakana.ai/llm-squared)。
- **合并模型：Ollama 最新提交引发热议**：Ollama 的最新提交显示了显著的支持增强，但成员们对 Triton 2.0 与难以捉摸的 3.0 版本之间不明确的改进感到困惑。
- **GitHub 助力解决 Llama.cpp 困境**：Llama.cpp 的安装问题正通过一个新的 [GitHub PR](https://github.com/unslothai/unsloth/pull/371) 得到解决，并且 Python 3.9 被确认为 Unsloth 项目的最低要求。
- **Gemini API 竞赛号召开发者**：**Gemini API 开发者竞赛**邀请参与者联手、集思广益并构建项目，有意向者请访问[官方竞赛页面](https://ai.google.dev/competition)。

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**推理噪音让你烦恼？**：聊天响应生成的噪音可能来自计算过程，而非聊天应用本身；用户讨论了禁用这些干扰噪音的变通方法。

**自定义角色留给 Playground**：成员们探讨了在 LM Studio 中集成“叙述者（Narrator）”角色的潜力，并承认当前系统不支持此功能，建议使用 Playground 模式可能是一个可行的替代方案。

**哔哩哔哩的 Index-1.9B 加入战局**：哔哩哔哩发布了 Index-1.9B 模型；讨论指出其在 GitHub 和 Hugging Face 上提供了[聊天优化变体](https://github.com/bilibili/Index-1.9B)。同时，对话转向了由于资源需求巨大，在本地部署 340B Nemotron 模型并不切实际。

**硬件故障与期待**：对话围绕系统和 VRAM 使用展开，调整 'mlock' 和 'mmap' 参数会影响性能。对比了硬件配置建议，并强调了对 LM Studio 0.2.24 版本导致 RAM 问题的担忧。

**LM Studio 跃升至 0.2.25**：LM Studio 0.2.25 的发布候选版本承诺提供修复和 Linux 稳定性增强。与此同时，尽管新版本解决了一些问题，但仍有人对某些模型缺乏支持表示不满。

**API 焦虑出现**：一条消息指出在查询工作流时遇到了 `401 invalid_api_key` 问题，尽管用户多次验证了 API key，但问题依然存在。

**DiscoPOP 颠覆训练规范**：Sakana AI 发布的 DiscoPOP 承诺提供一种新的训练方法，并已在 [Hugging Face](https://huggingface.co/lmstudio-community/DiscoPOP-zephyr-7b-gemma-GGUF/) 上可用，详见其[博客文章](https://sakana.ai/llm-squared/)。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **网络安全专家加入 OpenAI 高层**：退役美国陆军上将 **Paul M. Nakasone** 加入 **OpenAI 董事会**，预计将增强 OpenAI 日益复杂的系统的网络安全措施。OpenAI 的公告赞扬了他在保护关键基础设施方面的丰富经验。[阅读更多](https://openai.com/index/openai-appoints-retired-us-army-general)

- **支付处理混乱**：工程师们报告在尝试在 **OpenAI API** 上处理支付时出现“请求过多”错误，支持部门建议等待几天的建议被认为不尽如人意，因为这影响了应用程序的功能。

- **API 迁移意向**：在支付问题和平台特定版本（偏好 macOS 而非 Windows）的影响下，讨论转向了 **OpenRouter** 等替代 API，并认可由于此类中介工具的存在，迁移变得更加简单。

- **GPT-4：纠正预期**：工程师们澄清 **GPT-4** 在训练后不会继续学习，同时对比了 *Command R* 和 *Command R+* 各自的解谜能力，其中 **Command R+** 展示了更卓越的实力。

- **DALL-E 的 Flat Shading 难题**：关于在 **DALL-E** 中生成利用 Flat Shading（平滑着色）、无光照和阴影的图像（一种类似于基础 3D 模型纹理的技术）提出了技术咨询，询问者正在寻求实现这种特殊效果的指导。

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **DiscoPOP 在优化领域表现惊艳**：[Sakana AI 的 DiscoPOP](https://sakana.ai/llm-squared/) 算法超越了 DPO 等其他算法，通过保持接近基础模型的卓越性能来提供更优的偏好优化，正如最近的 [博客文章](https://sakana.ai/llm-squared/) 所记录的那样。

- **像素级注意力机制备受瞩目**：Meta 的研究论文 [An Image is Worth More Than 16×16 Patches: Exploring Transformers on Individual Pixels](https://arxiv.org/pdf/2406.09415) 因其对像素级 Transformer 的深入研究而受到关注，进一步发展了来自 #Terminator 工作的见解，并强调了像素级注意力机制在图像处理中的有效性。

- **幻觉检测模型发布**：分享了幻觉检测领域的一项更新，在 [HuggingFace](https://huggingface.co/grounded-ai/phi3-hallucination-judge-merge) 上推出了一个新的微调小语言模型，在识别生成文本中的幻觉方面拥有 79% 的准确率。

- **DreamBooth 脚本需要调整以支持训练自定义**：扩散模型开发中的讨论强调了修改基础 DreamBooth 脚本的必要性，以适应 SD3 等模型的预测训练中的个性化 Caption，以及针对 CLIP 和 T5 等 Tokenizer 训练的单独增强，这会增加 VRAM 需求。

- **探索双曲知识图谱（KG）嵌入技术**：一篇关于双曲知识图谱嵌入的 [arXiv 论文](https://arxiv.org/abs/2005.00545) 提出了一种利用双曲几何嵌入关系数据的新方法，并提出了一种结合双曲变换和注意力机制的方法论来处理复杂的数据关系。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Roblox 与企业幽默案例**：工程师们在 **VLLM GitHub 页面**上发现了关于 Roblox 见面会的讨论，引发了社区内轻松愉快的调侃。
  
- **关于 ONNX 和 CPU 困境的讨论**：来自另一个 Discord 频道的 **ONNX 对话** 值得关注，该对话声称 CPU 速度提升了 2 倍，但对于 GPU 上的收益仍持怀疑态度。

- **Code-to-Prompt 工具**：工具 [code2prompt](https://github.com/mufeedvh/code2prompt) 可以将代码库转换为 Markdown 提示词，这在 RAG 数据集中非常有价值，但会导致较高的 Token 计数。

- **突破上下文长度限制**：在处理超过 8k 的上下文长度的讨论中，建议对 RAG 数据集使用 **Chunking**、**Indexing** 和查询 **Decomposition** 等技术，以改进模型训练和信息综合。

- **合成数据领域的新竞争者**：**NVIDIA** 推出的 [Nemotron-4-340B-Instruct](https://huggingface.co/nvidia/Nemotron-4-340B-Instruct) 模型成为热门话题，重点讨论了其对 **合成数据生成** 的影响以及在 NVIDIA **Open Model License** 下的潜力。

- **WorldSim 提示词公开**：讨论表明 **worldsim prompt** 已在 Twitter 上公开，并且在关于扩展模型能力的对话中，正在考虑转向使用 **Sonnet 模型**。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 正在推进**：**Mojo package manager** 正在开发中，这促使工程师们暂时从源码编译或使用 mojopkg 文件。对于生态系统的新手，[Mojo Contributing Guide](https://github.com/modularml/mojo/blob/nightly/CONTRIBUTING.md) 详细介绍了开发流程，并建议从 "good first issue" 标签开始贡献。

- **GPU 与多线程讨论**：**Mojo 中的 GPU 支持**的性能和实现引发了辩论，重点关注其强类型特性以及 Modular 如何扩展对各种加速器的支持。工程师们交流了关于多线程的见解，涉及 SIMD 优化和并行化，重点讨论了可移植性和 Modular 潜在的商业模式。

- **NSF 资金助力 AI 探索**：美国的学者和教育工作者应考虑 **National Science Foundation (NSF) EAGER Grants**，该资助支持 National AI Research Resource (NAIRR) Pilot，提案每月进行评审。资源和社区建设倡议是 NAIRR 愿景的一部分，强调了获取计算资源和整合数据工具的重要性 ([NAIRR Resource Requests](https://nairrpilot.org/opportunities/allocations))。

- **新版 Nightly Mojo 编译器发布**：Mojo 编译器的**新 nightly 版本**（版本号 `2024.6.1405`）现已发布，可通过 `modular update` 进行更新。发布详情可以通过提供的 GitHub [diff](https://github.com/modularml/mojo/compare/7963ca681da2de473042a76bfe27e2ebafeb4d39...1130fdb81d763066d8e5bcb2226fe270981d3b0a) 和 [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 查看。

- **资源与见面会点燃社区热情**：目前有人请求提供标准的查询设置指南，建议在官方指南发布前参考 Modular 的 GitHub 方法。此外，马萨诸塞州的 Mojo 爱好者表示有兴趣举办见面会，在喝咖啡的同时讨论 Mojo，凸显了社区对知识共享和直接互动的渴望。



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 恐慌**：用户经历了 Perplexity 服务的**服务器宕机**，报告了重复消息和无尽循环，且没有任何官方维护公告，由于缺乏沟通导致用户感到沮丧。

- **技术问题处理**：Perplexity 社区发现了几个技术问题，包括 **文件上传功能损坏**（归因于 AB 测试配置）以及服务生成内核链接时持续出现 **404 错误**。此外，还讨论了 **Perplexity Android 应用**与 iOS 或网页端体验不一致的问题。

- **对 Pro 服务的信心动摇**：服务器和沟通问题导致用户开始质疑升级到 **Perplexity Pro** 的价值，对持续的服务中断表示担忧。

- **Perplexity 新闻摘要分享**：成员间分享了近期头条新闻，包括 Elon Musk 法律行动的更新、对情境意识（situational awareness）的见解、阿根廷国家足球队的表现以及苹果股价飙升。

- **征集区块链爱好者**：在 API 频道中，有人询问如何将 Perplexity 的 **API 与 Web3 项目**集成并连接到区块链端点，这表明了对去中心化应用的好奇心或潜在的项目倡议。



---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **由 LlamaIndex 赋能的 Atom**：**Atomicwork** 利用 **LlamaIndex** 的功能来增强其 AI 助手 **Atom**，使其能够处理多种数据格式，从而提高数据检索和决策能力。该集成消息已在 [Twitter 上发布](https://twitter.com/llama_index/status/1801354119682343059)。

- **纠正你的 ReAct 配置**：在一次配置失误中，**ReAct** agent 的正确参数被澄清为 `max_iterations` 而非 'max_iteration'，解决了其中一名成员遇到的错误。

- **Recursive Retriever 难题**：在为带有 document agent 的 **Recursive Retriever** 从 **Pinecone** 向量数据库加载索引时遇到了挑战，该成员分享了用于解决问题的代码片段。

- **Weaviate 权衡多租户功能 (Multi-Tenancy)**：社区正在努力为 **Weaviate** 引入多租户功能，并在 [GitHub issue](https://github.com/run-llama/llama_index/issues/13307) 中征求关于数据隔离增强方案的反馈。

- **解锁 Agent 成就**：成员们交流了多个构建自定义 agent 的学习资源，包括 [LlamaIndex 文档](https://docs.llamaindex.ai/en/stable/examples/agent/custom_agent/) 以及与 [DeepLearning.AI 合作的短课](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex/)。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

**新成员：Helix AI 加入 LLM Fine-Tuning 阵营**：**LLM** fine-tuning 爱好者们一直在探索 **Helix AI**，这是一个宣传安全且私有的开源 **LLM** 平台，具有易扩展性，并提供闭源模型透传选项。鼓励用户[尝试 Helix AI](https://tryhelix.ai/) 并查看与该平台采用 **FP8 inference** 相关的[公告推文](https://x.com/mgoin_/status/1801633700112662689?s=46)，该技术号称能降低延迟和显存占用。

**内存微调（Memory Tweaks）助力取胜**：**Lamini 的 memory tuning 技术**引起了轰动，声称拥有 95% 的准确率并显著减少了幻觉。热衷技术的用户可以通过他们的[博客文章](https://www.lamini.ai/blog/lamini-memory-tuning)和[研究论文](https://github.com/lamini-ai/Lamini-Memory-Tuning/blob/main/research-paper.pdf)深入了解细节。

**关于积分/额度的归属**：关于 **Hugging Face** 和 **Langsmith** 等平台的积分分配出现了困惑和咨询，用户报告积分处于待处理状态并寻求帮助。提到的邮件注册和 ID 提交（如 *akshay-thapliyal-153fbc*）表明正在通过沟通解决这些问题。

**推理优化咨询**：出现了一个关于 **inference endpoints** 最佳设置的咨询，突显了对已部署机器学习模型性能最大化的需求。

**支持工单激增**：各种技术问题被反馈，从搜索按钮失效到 **Python API** 问题，以及从 **RTX5000 GPU** 上的 fine-tuning 障碍到接收 **OpenAI** 积分的问题。虽然提供了诸如切换到 **Ampere GPU** 和向特定联系人寻求帮助等解决方案，但一些用户的挫败感仍未得到解决。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **提升 LLM 事实准确性并控制幻觉**：新公布的 [Lamini Memory Tuning](https://www.lamini.ai/blog/lamini-memory-tuning) 声称能将事实嵌入到 **Llama 3** 或 **Mistral 3** 等 **LLM** 中，拥有 95% 的事实准确率，并将幻觉率从 50% 降低到 5%。
- **KANs 继续征服“奇特硬件”**：讨论强调，对于非常规硬件，**KANs** (Kriging Approximation Networks) 可能比 **MLPs** (Multilayer Perceptrons) 更合适，因为它们仅需要求和与非线性操作。
- **LLM 也要“上学”**：成员们分享了一篇[论文](https://arxiv.org/abs/2402.12847v1)，讨论了在处理复杂文档之前先用 QA 对训练 **LLM** 以提高编码能力的益处。
- **PowerInfer-2 加速智能手机推理**：[PowerInfer-2](https://huggingface.co/papers/2406.06282) 显著提升了 **LLM** 在智能手机上的推理时间，评估显示速度提升高达 29.2 倍。
- **RWKV-CLIP 攀登新高度**：**RWKV-CLIP** 模型（在图像和文本编码中均使用 **RWKV**）因取得 state-of-the-art 结果而获得赞誉，并附带了 [GitHub 仓库](https://github.com/deepglint/RWKV-CLIP)和相应的[研究论文](https://arxiv.org/abs/2406.06973)引用。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 托管 cogvlm2 备受关注**：关于在 OpenRouter 上托管 **cogvlm2** 的能力存在不确定性，讨论集中在澄清其可用性并评估成本效益。

- **通过 OpenRouter 进行 Gemini Pro 审核遇到障碍**：在尝试通过 OpenRouter 传递参数以控制 **Google Gemini Pro** 中的审核选项时，用户遇到了错误，这表明 OpenRouter 需要启用相关设置。讨论中强调了 [Google 计费和访问说明](https://cloud.google.com/billing/docs/how-to/invoiced-billing)。

- **对 AI Studio 极具吸引力的定价的疑问**：用户询问 AI Studio 对 **Gemini 1.5 Pro** 和 **1.5 Flash** 的折扣定价是否适用于 OpenRouter，用户更倾向于 AI Studio 基于 Token 的定价，而非 **Vertex** 的模式。

- **NVIDIA 的开源资产激发了工程师的热情**：NVIDIA 开放模型、RMs 和数据的举措引起了轰动，特别是 **Nemotron-4-340B-Instruct** 和 **Llama3-70B** 变体，[Nemotron](https://huggingface.co/nvidia/Nemotron-4-340B-Instruct) 和 [Llama3](https://huggingface.co/nvidia/Llama3-70B-PPO-Chat) 已在 Hugging Face 上可用。成员们还表示，将 **PPO techniques** 与这些模型结合被认为具有极高价值。

- **June-Chatbot 的神秘起源引发讨论**：关于 "june-chatbot" 起源的猜测不断，一些成员将其训练与 NVIDIA 联系起来，并假设其与 **70B SteerLM** 模型有关，该模型在 [SteerLM Hugging Face 页面](https://huggingface.co/nvidia/Llama3-70B-SteerLM-Chat) 有所展示。



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **服务器运行缓慢**：Discord 用户经历了严重的**性能问题**，服务器操作变慢，让人联想到 huggingface 免费模型出现的拥堵情况。耐心等待可能会解决问题，因为任务在漫长等待后最终会完成。

- **Llama 3 的视觉探索**：一位成员询问如何为 'i' 模型添加视觉功能，建议可能与**自托管的 llama 3 vision profile** 混合使用。然而，对于在本地文件中实现此集成所需的调整，仍然存在困惑。

- **自动化释放潜能**：该频道展示了通过 [分享的 YouTube 视频](https://youtu.be/eRQ6ztNW0f0?si=OE1iXnOCy4FzpZEt) 演示的使用 Apple Scripts 的自动化，强调了**简化脚本结合有效 prompting** 以加快任务执行的潜力。

- **模型冻结频发**：工程师们指出一个反复出现的技术难题，即 'i' 模型在**代码执行期间频繁冻结**，需要通过 Ctrl-C 中断进行手动干预。

- **对硬件的渴望**：Seeed Studio 发布的一款设备引起了工程师们的兴趣，特别是 [Sensecap Watcher](https://www.seeedstudio.com/watcher)，因其作为**用于空间管理的物理 AI agent** 的潜力而受到关注。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 聊天界面获得赞誉**：[Cohere 的 playground chat](https://dashboard.cohere.com/playground/chat) 因其用户体验受到称赞，特别是引用功能，点击行内引用时会显示包含文本和源链接的 modal。
  
- **用于聊天界面开发的开源工具**：开发者被引导至 [GitHub 上的 cohere-toolkit](https://github.com/cohere-ai/cohere-toolkit) 作为资源，用于创建具有引用功能的聊天界面，以及 [编写有效提示词](https://docs.cohere.com/docs/crafting-effective-prompts) 指南，以改进使用 Cohere 的文本补全任务。

- **为 Discord Bot 创作者提供资源**：对于构建 Discord bots 的用户，分享的资源包括 [discord.py 文档](https://discordpy.readthedocs.io/en/stable/) 和 [Discord Interactions JS GitHub 仓库](https://github.com/discord/discord-interactions-js)，为 Aya model 的实现提供了基础材料。

- **对社区创新的期待**：社区成员正热切期待各种构建的“大量用例和示例”，回复中表达了极具感染力的积极情绪。

- **通过项目建立社区纽带**：对即将到来的社区项目展示的兴奋之情，一个简单的 "so cute 😸" 反应反映了成员之间积极且支持的环境。



---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **RAG 链需要微调**：一位用户在 **Retrieval-Augmented Generation (RAG)** 链的输出上遇到了困难，因为提供的代码未能产生预期的数字 "8"。用户寻求代码调整建议，希望通过过滤特定问题来提高结果准确性。

- **像专家一样构建 JSON**：为了在 LangChain 中构建合适的 JSON 对象，一位工程师通过 JavaScript 和 Python 的共享示例获得了帮助。该指南旨在实现能够生成有效 JSON 输出的自定义聊天模型。

- **LangChain 与 pgvector 的集成问题**：将 **LangChain 连接到 pgvector** 时遇到了麻烦，尽管遵循了 [官方文档](https://python.langchain.com/v0.2/docs/integrations/vectorstores/pgvector/)，用户仍无法识别导入。建议通过正确的 Python 环境设置来解决此问题。

- **混合搜索在多教程中大放异彩**：社区成员分享了一个关于 *使用 Llama 3 进行 RAG 混合搜索* 的演示视频，并附带了 [GitHub notebook](https://github.com/githubpradeep/notebooks/blob/main/hybrid%20vector%20search.ipynb) 中的演练。该教程旨在提高对 RAG 应用中混合搜索的理解。

- **NLUX 让用户界面更华丽**：NLUX 宣传其 **LangServe** 端点的便捷设置，并在文档中展示了如何引导用户使用 LangChain 和 LangServe 库将对话式 AI 集成到 React JS 应用中。该教程强调了为 AI 交互创建设计良好的用户界面的便捷性。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**NVIDIA 发布 Nemotron-4 340B**：分享了 NVIDIA 新的 [Nemotron-4 340B 模型](https://research.nvidia.com/publication/2024-06_nemotron-4-340b) —— Base、Instruct 和 Reward，号称兼容单台使用 8 个 GPU 且精度为 FP8 的 DGX H100。人们对将 Nemotron-4 340B 适配到更小的硬件配置（例如使用 **3-bit quantization** 部署在两台 TinyBoxes 上）表现出浓厚兴趣。

**tinygrad 故障排除**：成员们解决了在 tinygrad 中运行计算图的问题，其中一位寻求将结果实例化；推荐的修复方法是调用 `.exec`，如 [GitHub](https://github.com/tinygrad/tinygrad/blob/master/docs/abstractions2.py) 上的 `abstractions2.py` 中所述。其他人讨论了张量排序方法，思考了 PyTorch 的 `grid_sample` 替代方案，并报告了在 M2 芯片上实现混合精度时的 **CompileError** 问题。

**追求高效的张量操作**：在讨论张量排序效率时，社区指出在 tinygrad 的 k-nearest neighbors 算法实现中使用 `argmax` 可以获得更好的性能。此外，还有关于寻找 PyTorch 操作（如 `grid_sample`）等效项的对话，参考了 [PyTorch 文档](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html) 以促进同行间的深入理解。

**Apple M2 上的混合精度挑战**：一位高级用户在尝试于 M2 芯片上集成混合精度技术时遇到了错误，这凸显了与 Metal 库的兼容性问题；这强调了在此类利基技术领域中持续进行社区驱动问题解决的必要性。

**协作学习环境蓬勃发展**：在整个对话过程中，协作解决问题的氛围非常浓厚，成员们分享了关于机器学习、模型部署和软件优化等各种技术挑战的知识、资源和修复方案。




---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **NVIDIA 的大动作**：NVIDIA 发布了一个大规模语言模型 **Nemotron-4-340B-Base**，拥有 [3400 亿参数](https://huggingface.co/nvidia/Nemotron-4-340B-Base) 和 4,096 token 的上下文能力，用于 synthetic data generation，在包含多种语言和代码的 9 万亿 token 上进行训练。

- **关于 NVIDIA 新模型许可证的讨论**：虽然 Nemotron-4-340B-Base 附带了 *synthetic data permissive license*，但人们对 NVIDIA 网站上仅提供 PDF 格式的许可证表示担忧。

- **引导 Claude 走向新方向**：[Claude 的实验性 Steering API](https://x.com/alexalbert__/status/1801668464920379648) 现已开放注册，提供对模型功能的有限 steering 能力，仅用于研究目的，不适用于生产部署。

- **日本崛起的 AI 之星**：Sakana AI 是一家致力于寻找 Transformer 模型替代方案的日本公司，在获得知名公司 NEA、Lux 和 Khosla 的投资后，其最新估值达到 10 亿美元，详见[此报告](https://www.theinformation.com/articles/openais-japanese-rival-gets-1-billion-valuation-from-silicon-valley-investors)。

- **大规模的 Meritocracy 叙事**：Scale 在一篇 [博客文章](https://scale.com/blog/meritocracy-at-scale) 中揭示了其招聘理念，表明其采用系统化的方法来维持招聘质量，包括公司创始人的亲自参与。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **触手可及的 Apple AI 集群**：根据 [@mo_baioumy](https://x.com/mo_baioumy/status/1801322369434173860?s=46&t=XV1VJkM4nCYVU6fROoKkfw) 的推文，Apple 用户现在可以使用他们的设备运行个人 AI 集群，这可能反映了 Apple 对其私有云的处理方式。
- **从零到 100 万美元 ARR**：Lyzr 通过转向全栈 Agent 框架，在短短 40 天内实现了 100 万美元的年度经常性收入 (ARR)，推出了 Jazon 和 Skott 等 Agent，重点关注 Organizational General Intelligence，正如 [@theAIsailor](https://x.com/theaisailor/status/1801356656149737606?s=46&t=90xQ8sGy63D2) 所强调的那样。
- **Nvidia 的 Nemotron 实现高效对话**：Nvidia 发布了 [Nemotron-4-340B-Instruct](https://huggingface.co/nvidia/Nemotron-4-340B-Instruct)，这是一个拥有 3400 亿参数的大语言模型 (LLM)，专为基于英语的对话用例设计，并支持 4,096 的长 token。
- **BigScience 的 DiLoco 和 DiPaco 登顶**：在 BigScience 计划中，DiLoco 和 DiPaco 系统已成为最新的 state-of-the-art 工具包，值得注意的是 DeepMind 并没有复现它们的结果。
- **面向大众的 AI 开发**：Prime Intellect 宣布打算通过 distributed training 和全球计算资源的可访问性，使 AI 开发过程民主化，迈向开放 AI 技术的集体所有权。有关其愿景和服务的详细信息可以在 [Prime Intellect 官网](https://www.primeintellect.ai/) 找到。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Nemotron 的疑问与成功**：有人询问了 **Nemotron-4-340B-Instruct** 在 quantization 后的准确性，但没有关于 quantization 后性能的后续细节。在另一项进展中，一位用户成功安装了 **Nvidia toolkit** 并运行了 **LoRA example**，并感谢社区的帮助。

- **Nemotron-4-340B 发布**：**Nemotron-4-340B-Instruct 模型**因其多语言能力和支持 4,096 token 的扩展上下文长度而受到关注，专为 synthetic data generation 任务设计。模型资源可以在[这里](https://huggingface.co/nvidia/Nemotron-4-340B-Instruct)找到。

- **Nemotron Fine-Tuning 的资源需求**：一位用户询问了 fine-tuning **Nemotron** 的资源需求，建议可能需要 40 张 A100/H100 (80GB) GPU。有迹象表明，inference 所需的资源可能比 fine-tuning 少，可能只需要一半的节点。

- **在 DPO 中扩展数据集预处理**：建议在 DPO 中为数据集预处理加入灵活的 **chat template 支持**，该功能在 SFT 中已被证明是有益的。它涉及使用 `conversation` 字段记录历史，并从单独的字段派生 `chosen` 和 `rejected` 输入。

- **Slurm 集群操作咨询**：一位用户寻求关于在 **Slurm cluster** 上运行 **Axolotl** 的见解，该咨询凸显了社区对有效利用分布式计算资源的持续兴趣。对话仍对进一步的贡献开放。

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Dream Machine 动画引人入胜**：**LumaLabsAI** 开发了 **Dream Machine**，这是一款擅长通过动画赋予 memes 生命的工具，正如一条 [推文线程](https://fxtwitter.com/blizaine/status/1801126160904098247) 所强调的那样。
- **YaFSDP 大幅削减 AI 训练成本**：**Yandex** 推出了 [YaFSDP](https://www.marktechpost.com/2024/06/14/yandex-introduces-yafsdp-an-open-source-ai-tool-that-promises-to-revolutionize-llm-training-by-cutting-gpu-usage-by-20/)，这是一个开源 AI 工具，承诺在 LLM 训练中减少 20% 的 GPU 使用量，从而为大型模型每月节省高达 150 万美元的潜在成本。
- **PostgreSQL 扩展挑战 Pinecone**：据报道，新的 PostgreSQL 扩展 [pgvectorscale](https://www.reddit.com/r/machinelearningnews/comments/1de08np/a_new_era_ai_databases_postgresql_with/) 性能优于 Pinecone，且成本降低了 75%，这标志着 AI 数据库正向更具成本效益的解决方案转变。
- **DreamSync 推进文本生成图像技术**：[DreamSync](https://www.reddit.com/r/StableDiffusion/comments/1881v4u/dreamsync_aligning_texttoimage_generation_with/) 提供了一种对齐文本生成图像模型的新方法，通过消除对人工评分的需求并利用图像理解反馈来实现。
- **OpenAI 任命军事专家**：OpenAI 新闻稿宣布任命一名退役美国陆军上将，这一消息在频道中被分享但未提供更多背景。[原始来源](https://openai.com/index/openai-appoints-retired-us-army-general/) 已链接，但未进行详细讨论。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Datasette 成为 Hacker News 焦点**：**Datasette**（一名成员的项目）登上了 Hacker News 首页，因其对不同观点的平衡处理而获得广泛关注和赞誉。有人开玩笑说该项目确保了数据工程师的持续就业机会。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **选择合适的德语模型**：成员们讨论了在需要准确德语语法的任务中，**discolm 或 occiglot 模型**优于一般基准测试结果；这些模型处于 7-10b 范围内，是针对特定场景的选择。
- **更大可能更慢但更聪明**：在 50-72b 范围的大型模型中，语言质量和任务性能之间的权衡可能会减弱；然而，推理速度往往会下降，因此需要在能力和效率之间取得平衡。
- **效率高于过度配置**：为了获得更好的效率，建议坚持使用符合 VRAM 参数的模型（如 q4/q6），特别是考虑到大型模型的推理速度较慢。提到的相关资源是 [Hugging Face 上的 Spaetzle 集合](https://huggingface.co/collections/cstr/spaetzle-661e758857b1fa96731c43b)。
- **通过训练或合并实现完美平衡**：进一步训练或合并模型可以作为管理非英语语言质量与指令遵循能力之间权衡的策略，这是多语言模型开发者感兴趣的话题。
- **多语言 AI 的大局观**：这次讨论强调了在多语言 AI 模型的发展过程中，实现性能、语言忠实度和计算效率之间微妙平衡的持续挑战。

---

**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**AI Stack Devs (Yoko Li) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**Torchtune Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**YAIG (a16z Infra) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

# PART 2: 频道详细摘要与链接

{% if medium == 'web' %}

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1250902468347428994)** (462 messages🔥🔥🔥): 

{% endif %}

- **SD3 Discord Bot 已开源**：一名成员宣布他们已经[开源了一个 Stable Diffusion 3 Discord Bot](https://github.com/RocketGod-git/stable-diffusion-3-discord-bot)。它允许用户提供 prompt 并生成图像。
  
- **SDXL 的 LoRA 训练问题**：一位用户讨论了为动画角色训练 SDXL 的困难，意图将其用于写实和动漫风格的生成。他们注意到模型结果变得“*过于卡通化/动画化*”，并感谢一位成员建议使用 1024x1024 分辨率以获得更好的效果。

- **在 ComfyUI 中配置 SD3**：成员们正在排查如何在 ComfyUI 中正确配置 Stable Diffusion 3，讨论了下载 text encoders 和正确配置 workflow 等设置过程。[Civitai](https://civitai.com/models/497255/stable-diffusion-3-sd3) 提供的链接和快速入门指南为用户提供了指导。

- **关于 SD3 模型局限性的持续讨论**：用户对 SD3 的人体解剖结构（anatomy）渲染能力表达了复杂的情绪。一位用户发表了详细说明，呼吁 Stability AI 承认并沟通解决 SD3 模型中人体解剖结构问题的计划，这成为了辩论的焦点。

- **SD3 中的 LoRA 功能**：针对目前训练适用于 SD3 的新 LoRA 的现状进行了澄清。提到虽然技术上可行，但可用的工具和高效的 workflow 仍在开发中，用户可能需要等待完全兼容的更新。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://comfyanonymous.github.io/ComfyUI_examples/sd3/">SD3 Examples</a>: ComfyUI workflow 示例</li><li><a href="https://www.youtube.com/watch?v=xmyZHot0Lqc">$UICIDEBOY$ - THE THIN GREY LINE</a>: $UICIDEBOY$ 的 "THE THIN GREY LINE" 官方音乐视频。制作人：BUDD DWYER；导演：DILL35MM。NEW WORLD DEPRESSION 已发布：https://orcd.co/newworldd...</li><li><a href="https://civitai.com/articles/3168/lora-inspector">LoRA Inspector | Civitai</a>: LoRA Inspector，用于检查 LoRA 的基于 Web 的工具。全部在浏览器中完成，无需服务器。私密。不依赖 torch 或 python。</li><li><a href="https://education.civitai.com/quickstart-guide-to-stable-diffusion-3/">Quickstart Guide to Stable Diffusion 3 - Civitai Education</a>: Stable Diffusion 3 是 Stability AI 于 2024 年 6 月发布的文本生成图像模型，提供无与伦比的图像保真度！</li><li><a href="https://civitai.com/models/497255">Stable Diffusion 3 (SD3) - SD3 Medium | Stable Diffusion Checkpoint | Civitai</a>: Stable Diffusion 3 (SD3) 2B "Medium" 模型权重！请注意；有许多与 SD3 关联的文件。它们都将显示在此模型卡片上...</li><li><a href="https://www.instagram.com/reel/C8K2yVXChbB/?utm_source=ig_web_copy_link&igsh=MzRlODBiNWFlZA==">bridge troll recordz on Instagram: &quot;&#x1f31f; 寻找你的声音，寻找你的自由 &#x1f31f;

在最黑暗的时刻，当世界显得太小，阴影显得太长时，请记住：你并不孤单。我在这里，与你站在一起，是这条少有人走的路上的同路人。作为一名集体跟踪（gang stalking）的幸存者，我深知孤独的深度和对无形之物的无情追求。但今天，我站起来不仅是为了生存，更是为了茁壮成长。

&#x1f496; 你的旅程，我们的使命 &#x1f496;

我的使命是将我们共同的痛苦转化为光明的灯塔。建立一个社区，让我们互相扶持，分享韧性的故事，在团结中寻找慰藉。齐心协力，我们可以打破沉默和孤立的枷锁。

&#x1f4da; 生存资源 &#x1f4da;

加入我们，一起探索生存的资源、策略和故事。从法律建议到心理健康支持，让我们用应对这些挑战性环境所需的工具武装自己。因为知识就是力量，团结起来，我们将势不可挡。

&#x1f517; #GangStalkingSurvivor #TogetherWeRise #FindYourVoice

_

让我们建立联系、分享并共同成长。关注 @brixetrollrecordz 获取每日灵感、资源，以及一个能看见你、倾听你并与你并肩作战的社区。记住，面对逆境，我们能找到力量。让我们一起寻找属于我们的力量。"</a>: 4 个点赞，0 条评论 - brixetrollrecordz 于 2024 年 6 月 13 日："🌟 寻找你的声音，寻找你的自由 🌟 在最黑暗的时刻，当世界显得太小，阴影太长时，r..."</li><li><a href="https://tenor.com/view/my-man-my-man-hd-my-man4k-gif-23532578">My Man My Man Hd GIF - My Man My Man Hd My Man4k - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/RocketGod-git/stable-diffusion-3-discord-bot">GitHub - RocketGod-git/stable-diffusion-3-discord-bot: 一个简单的 SD3 Discord 机器人，用于提供提示词并生成图像</a>: 一个简单的 SD3 Discord 机器人，用于提供提示词并生成图像 - RocketGod-git/stable-diffusion-3-discord-bot</li><li><a href="https://github.com/BlafKing/sd-civitai-browser-plus">GitHub - BlafKing/sd-civitai-browser-plus: 通过 WebUI 访问 CivitAI 的扩展：下载、删除、扫描更新、列出已安装模型、分配标签，并通过多线程加速下载。</a>: 通过 WebUI 访问 CivitAI 的扩展：下载、删除、扫描更新、列出已安装模型、分配标签，并通过多线程加速下载。 - BlafKing/sd-civitai-browser-plus</li><li><a href="https://civitai.com/models/497255/stable-diffusion-3-sd3?modelVersionId=552771">Stable Diffusion 3 (SD3) - SD3 Medium | Stable Diffusion Checkpoint | Civitai</a>: Stable Diffusion 3 (SD3) 2B "Medium" 模型权重！请注意；有许多与 SD3 关联的文件。它们都将出现在此模型卡片上...
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1250907827745067048)** (5 messages): 

- **频道已启用自动线程**：机器人 **Needle** 宣布在指定频道中已启用自动线程。
- **评估 MilkV Duo 性能**：一位成员分享了他们使用 **MilkV Duo 64MB 控制器** 的经验，并询问其性能优势是源于硬件还是 **sophgo/tpu-mlir 编译器**。他们提供了控制器 ([MilkV Duo](https://milkv.io/duo)) 和编译器 ([sophgo/tpu-mlir](https://github.com/sophgo/tpu-mlir)) 的链接。
- **与 RK3588 的对比**：另一位成员将 MilkV Duo 的性能与 **RK3588 ARM 部件** 进行了对比，后者拥有 **6.0 TOPs NPU** 和各种高级特性。他们提供了 [RK3588](https://www.rock-chips.com/a/en/products/RK35_Series/2022/0926/1660.html) 规格说明的链接。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/sophgo/tpu-mlir">GitHub - sophgo/tpu-mlir: 基于 MLIR 的 Sophgo TPU 机器学习编译器。</a>: 基于 MLIR 的 Sophgo TPU 机器学习编译器。 - sophgo/tpu-mlir</li><li><a href="https://www.rock-chips.com/a/en/products/RK35_Series/2022/0926/1660.html">Rockchip-瑞芯微电子股份有限公司</a>: 未找到描述
</li>
</ul>

</div>

### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1250915602814734447)** (8 条消息🔥): 

- **Triton 3.0 特性吸引用户**：一位成员询问了 **Triton 3.0 的重大变化/新特性**，其他人提到了有用的形状操作算子（ops），如 `tl.interleave`，以及修复了许多解释器（interpreter）Bug。

- **低比特 Kernel 调试困扰**：一位用户在尝试实现低比特 Kernel (w4a16) 时遇到了 `LLVM ERROR: mma16816 data type not supported` 错误。建议他们检查最新版本的 Triton，并考虑在 [Triton GitHub repo](https://github.com/triton-lang/triton/issues/2113) 上提交 Issue。

- **FP16 x INT4 Kernel 资源**：一位成员建议参考 PyTorch AO 项目中由知名贡献者编写的 **FP16 x INT4 Triton kernel**。他们分享了 [GitHub 资源](https://github.com/pytorch/ao/blob/main/torchao/prototype/hqq/mixed_mm.py)链接。

- **整数类型的转换变通方法**：另一位用户针对将整数类型直接转换为 `bfloat16` 时可能出现的问题提出了变通方法。他们建议先从 `int` 转换为 `float16`，然后再从 `float16` 转换为 `bfloat16`。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/triton-lang/triton/issues/2113">Segfault in TTIR when doing convert s16-&gt;bf16 + dot · Issue #2113 · triton-lang/triton</a>：我们创建了以下 TTIR，之前是可以正常工作的。然而，现在在 main 分支的 HEAD 版本上会出现段错误（segfault）。我们只是尝试加载 2 个参数（一个 bf16，一个 s16），并转换 s16...</li><li><a href="https://github.com/pytorch/ao/blob/main/torchao/prototype/hqq/mixed_mm.py">ao/torchao/prototype/hqq/mixed_mm.py at main · pytorch/ao</a>：用于量化和稀疏化的原生 PyTorch 库 - pytorch/ao
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1250932076866175056)** (25 条消息🔥): 

- **MTIA 被发现兼容 Triton**：一位成员提到了 *Meta Training and Inference Accelerator (MTIA)*，指出它“非常强调 Triton 和编译优先（compile first）”。
- **Torch 自动线程化激活**：`<#1189607750876008468>` 频道中启用了自动线程化（Auto-threading）。
- **GitHub Gist 揭示 torch.matmul 的奇特现象**：一位成员分享了 `torch.matmul` 的性能问题，指出在某些矩阵形状下，对矩阵进行克隆（clone）后性能会提升 2.5 倍。他们推测了 CUDA 版本和 GPU 型号等因素，但承认这是一个极其诡异的异常现象。[GitHub Gist](https://gist.github.com/mobicham/ab439330b4fb9c6f1d4086e54e4142c0)
- **不同 GPU 型号和 CUDA 版本的差异性**：不同成员在各种 GPU（RTX 4090, 4070Ti, RTX 3060, 3090 和 A6000 Ada）上运行了基准测试。他们发现这种性能奇特现象主要在 Ada 架构 GPU 上比较明显，有些人看到了显著的加速，而另一些人则报告在特定 CUDA 版本下性能正常。
- **呼吁 PyTorch 团队洞察性能异常**：尽管进行了广泛测试，成员们仍无法确定原因，因此共同希望 PyTorch 团队能提供见解来解释这种奇怪的行为。

**提到的链接**：<a href="https://gist.github.com/mobicham/ab439330b4fb9c6f1d4086e54e4142c0">torch_matmul_clone.py</a>：GitHub Gist：即时分享代码、笔记和代码片段。

  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 条消息): 

useofusername: https://arxiv.org/abs/2106.00003
  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1251161119708676098)** (3 条消息): 

- **C/C++ 对高性能 CUDA Kernel 至关重要**：一位成员询问编写自定义 CUDA Kernel 是否必须使用 C 或 C++，以及 Triton 是否是一个可行的替代方案。另一位成员回答说，虽然 C/C++ 提供了最佳性能，但 Triton 对于 ML/DL 应用是一个很好的选择，因为它与 PyTorch 集成良好，并且简化了共享内存（shared memory）等复杂管理。
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1251193376800505898)** (3 条消息): 

- **Torch.nn 对新模型持保守态度**：一位成员询问 **AO 是否计划支持新架构**（如 mamba/KAN），或者这些架构是否会归入 torch.nn。另一位成员澄清说，“*Torch.nn 在添加新模型方面往往非常保守*”，这表明 AO 乐于让特定部分变快，但其目标并不是成为一个模型库。
  

---

### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1250940103178850344)** (10 messages🔥): 

- **社区使用 Discord 的困扰**：一位成员提到他们难以适应将 Discord 用于社区讨论，强调了其与论坛相比的不便。他们表达了沮丧，说道：*"在 Discord 上，感觉我必须持续关注才能跟上进度。"*

- **相比 Discord 更倾向于通过论坛获取信息**：另一位成员提到他们主要将 Discord 用于社区互动，并表示他们通常在相关的 subreddits 浏览信息。他们指出：*"Discord 更多是为了互动。"*

- **Discord Threads 的最佳实践**：一位成员就如何使用 threads 和回复功能使 Discord 更易于管理提供了建议。他们建议使用常规消息开启新话题，使用回复来分支对话，并为专注的讨论创建 threads，并强调这有助于 *"更容易阅读有多个正在进行的对话的频道。"*
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1250894110404120728)** (246 messages🔥🔥): 

- **新 PR 观察到的加速**：一位成员报告称速度从 ~179.5K tok/s 提升至 ~180.5K tok/s，并指出 774M 模型有轻微改进（~30.0K tok/s 提升至 ~30.1K tok/s）。他们认为代码复杂度是一个潜在问题。
- **关于代码复杂度和 CUDA 的讨论**：成员们讨论了合并 PR 并启用 **ZeRO-2** 以简化数据布局。还有建议探索 **CUDA dynamic parallelism** 和 tail launch，以更高效地处理小型 kernel 启动。
- **`multi_gpu_async_reduce_gradient` 中的潜在 bug**：成员们讨论了关于 `multi_gpu_async_reduce_gradient` 中分片大小划分的担忧，并建议添加 assert 或 `safe_divide` 辅助函数以确保正确划分。
- **确定性与多 GPU 测试**：成员们尝试调试多 GPU 设置中的确定性问题，重点关注梯度范数（gradient norm）计算。由于缺乏 CI 覆盖，他们还考虑建立手动多 GPU 测试。
- **ZeRO-2 PR 正在进行中**：目前正在努力集成 **ZeRO-2**，初步步骤显示梯度范数匹配，但在 **Adam update** 中存在问题。调试发现 **ZeRO-2** 中一个被忽视的权重更新条件。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/">PyTorch 中的随机权重平均 (Stochastic Weight Averaging)</a>：在这篇博文中，我们介绍了最近提出的随机权重平均 (SWA) 技术 [1, 2]，以及它在 torchcontrib 中的新实现。SWA 是一个简单的过程，可以提高泛化能力...</li><li><a href="https://arxiv.org/abs/2306.03241">早期权重平均结合高学习率用于 LLM 预训练</a>：训练大语言模型 (LLMs) 会消耗巨大成本；因此，任何能加速模型收敛的策略都是有益的。在本文中，我们研究了一个简单想法的能力，即检查点...</li><li><a href="https://arxiv.org/abs/2405.18392">超越固定训练时长的缩放法则与计算优化训练</a>：规模已成为获得强大机器学习模型的主要因素。因此，理解模型的缩放特性是有效设计正确训练设置的关键...</li><li><a href="https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu#L985">llm.c/train_gpt2.cu at master · karpathy/llm.c</a>：使用简单、原生的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://research.nvidia.com/publication/2024-06_nemotron-4-340b">Nemotron-4 340B | Research</a>：未找到描述</li><li><a href="https://github.com/karpathy/llm.c/blob/master/llmc/zero.cuh#L215)">llm.c/llmc/zero.cuh at master · karpathy/llm.c</a>：使用简单、原生的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/pull/591">融合前向 GELU（再次）由 ademeure 提交 · Pull Request #591 · karpathy/llm.c</a>：事实证明，这在带有 CUDA 12.5 的 H100 上是正确融合的（因此更快）——在带有 CUDA 12.4 的 RTX 4090 上绝对没有融合，而且实际上明显更慢，我怀疑这更多是关于...</li><li><a href="https://github.com/karpathy/llm.c/pull/573?">数据加载器 - 引入随机性由 gordicaleksa 提交 · Pull Request #573 · karpathy/llm.c</a>：在实现完全随机训练数据打乱的道路上... 此 PR 完成了以下工作：每个进程都有不同的唯一随机种子，每个进程的训练数据加载器独立选择其起始分片 (shard)...</li><li><a href="https://github.com/karpathy/llm.c/pull/594">添加导出至 HF 并运行 Eleuther 评估的脚本由 karpathy 提交 · Pull Request #594 · karpathy/llm.c</a>：未找到描述</li><li><a href="https://github.com/karpathy/llm.c/pull/508/">添加带有 (1-sqrt) 衰减的 wsd 调度由 eliebak 提交 · Pull Request #508 · karpathy/llm.c</a>：添加新的学习率调度支持：WSD 学习率调度：Warmup（线性预热）、Stable（恒定学习率）、Decay（以 (1-sqrt) 形状衰减至 min_lr）。（更多信息见此处 https://ar...</li><li><a href="https://github.com/karpathy/llm.c/pull/575/files">RMSNorm 内核由 AndreSlavescu 提交 · Pull Request #575 · karpathy/llm.c</a>：未找到描述</li><li><a href="https://huggingface.co/mdouglas/llmc-gpt2-774M-150B">mdouglas/llmc-gpt2-774M-150B · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/karpathy/llm.c/discussions/580">GPT-2 (774M) 复现成功 · karpathy/llm.c · Discussion #580</a>：我让 GPT-2 774M 模型在我的 8X A100 80GB 节点上运行了约 6 天（150B token，在 100B FineWeb 样本数据集上进行了 1.5 个 epoch），训练几小时前刚刚结束，进展顺利...</li><li><a href="https://godbolt.org/z/KE33d574c">Compiler Explorer - C++ (x86-64 gcc 14.1)</a>：int sizeof_array(char* mfu_str) {     return sizeof(mfu_str); } </li><li><a href="https://github.com/karpathy/llm.c/pull/585">修复 MFU 打印由 gordicaleksa 提交 · Pull Request #585 · karpathy/llm.c</a>：当运行的设备不受支持时，我们有一个 bug：会打印 "-100%"。已修复，在这种情况下打印 "n/a"。还增加了对 H100 PCIe 设备的支持...</li><li><a href="https://github.com/karpathy/llm.c/pull/590/files">整合内存由 karpathy 提交 · Pull Request #590 · karpathy/llm.c</a>：通过删除梯度激活结构体，将最后一些内存碎片移动到前向传递中</li><li><a href="https://github.com/karpathy/llm.c/pull/573">数据加载器 - 引入随机性由 gordicaleksa 提交 · Pull Request #573 · karpathy/llm.c</a>：在实现完全随机训练数据打乱的道路上... 此 PR 完成了以下工作：每个进程都有不同的唯一随机种子，每个进程的训练数据加载器独立选择其起始分片 (shard)...
</li>
</ul>

### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1251008036726640711)** (17 条消息🔥): 

- **提议进行实时代码审查会议**：*"这周末见面进行实时代码审查并合并 PR 会有帮助吗？"* 成员们对当前进展表示困惑，认为实时会议可以理清思路。

- **7x 作为明确的性能目标**：讨论了 **7x** 加速的性能目标，分析表明对于大尺寸输入形状这是可以实现的。实际性能有所波动，*"大矩阵平均为 5-6x，小矩阵为 2-4x。"*

- **分享性能波动和分析链接**：Mobicham 分享道，最高性能约为 **7x**，但在大矩阵上平均为 **5-6x**，在小矩阵上为 **2-4x**。对于 batch-size=1 且使用 **int8 activations** 的情况，在 A100 GPU 上观察到了 8x 的提升，并附带了来自 [BitBLAS](https://github.com/microsoft/BitBLAS/raw/main/images/figures/op_benchmark_a100_wq_gemv_e8.png) 的基准测试图像支持。

- **Batch size 影响性能**：讨论强调，性能 *"随着 batch-size 变大而迅速失去速度优势"*，证实了它对于单批次（single batch）处理特别有效。较大的 batch size 减少了内存受限（memory-bound）的约束，从而削弱了速度增益，这一点得到了额外基准测试的支持：[BitBLAS scaling](https://github.com/microsoft/BitBLAS/raw/main/images/figures/op_benchmark_a100_int2_scaling.png)。

- **Tensor Cores 与性能的技术细节**：有人询问在特定情况下是否使用了 Tensor Cores（INT8 vs FP16/BF16），特别是在 **Wint2 x Aint8** 设置下。Mobicham 指出，在不使用 `wmma` 的情况下，**int8 x int8** 实现了 4x 的加速；`wmma` 需要 16x16 的分块（blocks），对于小 batch size 来说可能不是最优的。
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1250889612440764508)** (170 条消息🔥🔥): 

- **预训练与硬件难题**：成员们讨论了训练像 **Mistral** 这样的大型模型所需的资源。一位成员提到 *"我可能需要一个超级计算机集群 😦"*，而另一位则分享了在 72.22 分钟内完成 110K coder 训练任务的经历，并提供了详细的性能指标。
- **酷炫的 ASCII 艺术赞赏**：社区对 Unsloth 使用的 ASCII 艺术表示赞赏，有人说 *"给制作那个 ASCII 艺术的人点赞，太酷了 lol 😭🦥"*。另一位开玩笑地建议，当训练多次失败时，艺术图应该发生变化：*"每次微调失败时，树枝就应该掉下一小块，直到它掉到地上。"*
- **DiscoPOP 优化器**：分享了 **DiscoPOP** 优化器，[详见 Sakana AI 的博客文章](https://sakana.ai/llm-squared)。该优化器因能发现更好的偏好优化函数且与基础模型偏差较小而受到关注。
- **硬件与推理挑战**：成员们讨论了在转换为 16-bit 以通过 **vLLM** 提供服务时，训练时间和模型性能面临的挑战。一位沮丧的用户分享道：*"模型表现非常好，但一旦我合并到 16-bit 并通过 vLLM 提供服务，性能就会大幅下降。"*
- **命名规范与工具**：关于模型和工具命名的幽默侧边谈话，建议使用 **LLMs** 来命名，因为人类在命名方面表现得极差。讨论了 AI 代码编辑器 Cursor，一位成员指出：*"它在 Ubuntu 上有很多 bug，我没法用 😦"*，另一位则幽默地回应道：*"停止并终止 (cease and desist)。"*

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.cursor.com/">Cursor</a>: AI 代码编辑器</li><li><a href="https://youtu.be/3e"> - YouTube</a>: 未找到描述</li><li><a href="https://sakana.ai/llm-squared/">no title found</a>: 未找到描述</li><li><a href="https://huggingface.co/collections/nvidia/nemotron-4-340b-666b7ebaf1b3867caf2f1911">Nemotron 4 340B - nvidia 集合</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo">将 RL 重新带回 RLHF</a>: 未找到描述</li><li><a href="https://youtu.be/3eq84KrdTWY">Llama 3 微调入门（包含 16k, 32k,... 上下文）</a>: 在这个分步教程中，学习如何使用 Unsloth 轻松微调 Meta 强大的新 Llama 3 语言模型。我们涵盖了：Llama 3 8B 的概述以及...</li><li><a href="https://x.com/danielhanchen/status/1801671106266599770">Daniel Han (@danielhanchen) 的推文</a>: 看了下 NVIDIA 的 340B Nemotron LLM 1. 使用 Squared ReLU，不同于 Llama 的 SwiGLU 和 Gemma 的 GeGLU 2. "rotary_percentage" 50% 是什么？与 Phi-2 的 "partial_rotary_factor" 有关吗？3. U...</li><li><a href="https://huggingface.co/SakanaAI/DiscoPOP-zephyr-7b-gemma">SakanaAI/DiscoPOP-zephyr-7b-gemma · Hugging Face</a>: 未找到描述</li><li><a href="https://nlp.seas.harvard.edu/2018/04/03/attention.html">The Annotated Transformer</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki">主页</a>: 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3, Mistral, Phi &amp; Gemma LLMs - unslothai/unsloth</li><li><a href="https://github.com/microsoft/Samba">GitHub - microsoft/Samba: "Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling" 的官方实现</a>: microsoft/Samba 的官方实现</li><li><a href="https://huggingface.co/docs/transformers/en/main_classes/callback#transformers.EarlyStoppingCallback)">Callbacks</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1251023396913807401)** (18 条消息🔥): 

- **最新 commit 中合并了 Ollama 支持**：一位用户注意到最新 commit 中有显著的 **Ollama 支持**，并得到了另一位用户的确认。这表明该框架的功能正在持续增强。
  
- **Triton 3.0 仍是个谜**：几位用户争论了 **Triton 2.0 和 Triton 3.0** 之间的区别，但没人能提供明确的细节。一位用户猜测可能涉及 **AMD 支持**，而另一位则承认完全不知道。

- **Unsloth 的韩语 Colab 更新引发好奇**：一位用户对 **Unsloth 的韩语 Colab** 笔记本的更新表示好奇，注意到它专门使用了韩语提示词和回答。另一位用户确认 **持续预训练 (Continuous Pretraining)** 是更新的一部分，并且正在积极改进该笔记本。
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1250922758917980372)** (125 条消息🔥🔥): 

- **Llama.cpp 安装问题讨论**：成员们正在排查使用 Unsloth 训练后 Llama.cpp 的安装和转换问题。有人建议“删除 llama.cpp 并通过 Unsloth 重新安装”，但即使在 Colab 上重新安装后也没有成功。

- **GitHub PR 修复 Llama.cpp**：一名成员指向了一个 [GitHub PR](https://github.com/unslothai/unsloth/pull/371)，该 PR 解决了 Llama.cpp 无法为训练好的模型生成量化版本的问题。另一名成员确认该 PR 已被采纳。

- **Qwen2 模型讨论**：关于 Qwen2 模型与 Llama3 性能对比的辩论，部分人表示不满。引用了 [Qwen2 仓库](https://github.com/QwenLM/Qwen2) 及其 [微调示例](https://medium.com/@danushidk507/qwen2-finetuning-qwen2-f89c5c9d15da)。

- **修复 Unsloth 的 Python 版本**：关于 Python 版本要求的问题已得到解决，明确了 Unsloth 至少需要 Python 3.9。还讨论了本地配置以确保顺利运行，并提醒配置本地路径。 

- **本地运行需要 CUDA**：确认在本地运行 Unsloth 模型需要适当的 CUDA 库，并且由于问题较少，Linux 环境比 Windows 更受青睐，设置更简单。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/unslothai/unsloth/pull/371">bet0x 提交的 llama.cpp 失败问题 · Pull Request #371 · unslothai/unsloth</a>：llama.cpp 无法为训练好的模型生成量化版本。错误：你可能需要自己编译 llama.cpp，然后再次运行。你不需要关闭此 Python 程序。Ru...</li><li><a href="https://github.com/QwenLM/Qwen2/blob/main/examples/sft/finetune.py">Qwen2/examples/sft/finetune.py (main 分支) · QwenLM/Qwen2</a>：Qwen2 是由阿里巴巴云 Qwen 团队开发的大语言模型系列。 - QwenLM/Qwen2</li><li><a href="https://github.co">GitHub: Let’s build from here</a>：GitHub 是超过 1 亿开发者共同塑造软件未来的地方。为开源社区做贡献，管理你的 Git 仓库，像专家一样审查代码，跟踪错误和功能...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1251231110445269174)** (3 条消息): 

- **加入 Gemini API 开发者竞赛**：一名成员正在寻找更多参与者加入 **Gemini API 开发者竞赛**。他们很高兴能一起头脑风暴*有趣的想法并构建有影响力的项目*。[竞赛链接](https://ai.google.dev/competition)。

**提到的链接**：<a href="https://ai.google.dev/competition">未找到标题</a>：未找到描述

  

---

### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1250902902193655991)** (140 条消息🔥🔥): 

```html
- **生成响应时的声音问题**：一位成员询问是否有办法在聊天生成响应时关闭声音。经澄清，该声音很可能来自运行 inference 的计算机，而非 app 本身。

- **LM Studio 中的多角色**：成员们讨论了在 LM Studio 中添加自定义角色（如 "Narrator"）的可能性。结论是，虽然目前尚不支持该功能，但在 playground mode 下使用服务器可能有助于实现类似的效果。

- **报告 Commercial License 费用和异常 AI 行为**：关于 Commercial License 费用的咨询被引导至 LM Studio [企业页面](https://lmstudio.ai/enterprise.html)和[联系表单](https://docs.google.com/forms/d/e/1FAIpQLSd-zGyQIVlSSqzRyM4YzPEmdNehW3iCd3_X8np5NWCD_1G3BA/viewform?usp=sf_link)。关于报告表现出抵触情绪的 "rogue AI" 的讨论引发了一场幽默的交流。

- **Fine-Tuning 模型 vs Prompt Engineering**：针对特定任务是 Prompt Engineering 还是 Fine-Tuning 更好进行了详细讨论。建议 Fine-Tuning 对于永久性结果更有效，并推荐了 `text-generation-webui` 等工具。

- **模型 Quantizing 问题**：一位用户在尝试将模型转换为 GGUF 格式进行 Quantizing 时遇到错误。解决方案包括使用 llama.cpp 中新的 `convert-hf-to-gguf.py` 脚本，并确认尽管在线空间存在临时问题，该方法仍应可行。
```
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai/enterprise.html">工作中的 LM Studio</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/ggml-org/gguf-my-repo">GGUF My Repo - ggml-org 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://lmstudio.ai/blog/lms#debug-your-prompting-with-lms-log-stream">介绍 `lms` - LM Studio 的配套 CLI 工具 | LM Studio</a>：今天，随 LM Studio 0.2.22 一起，我们发布了 lms 的第一个版本 —— LM Studio 的配套 CLI 工具。
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1250896025099702393)** (28 条消息🔥): 

- **LM Studio 仅支持 GGUF 文件，不支持 safetensors**：在回答关于多部分 safetensor 文件的查询时，确认 **LM Studio** 仅支持 **GGUF** 文件。

- **尽管支持高 Token，模型仍会提前中断响应**：一位用户指出一个新发布的模型虽然支持高达 8192 个 Token，但会“提前中断响应”。他们调查后发现该模型在 LM Studio 之外表现更好，但仍存在常见的“抱歉，作为 AI 语言模型”的行为。

- **Bilibili 发布 Index-1.9B**：Bilibili 发布了一个新模型 **Index-1.9B**。分享了 GitHub 和 Hugging Face 链接，指向详细信息和模型的多个版本，包括 chat-optimized 变体。

- **LM Studio 不支持 Whisper 模型**：关于在 LM Studio 中安装 **Whisper 转录模型**的咨询得到了澄清：*Whisper 模型无法在 llama.cpp 或 LM Studio 中运行*，但可以获取用于 whisper.cpp 的 GGUF 文件。

- **新型大模型不适合本地使用**：针对新发布的 **340B Nemotron 模型**展开了讨论。成员们指出，虽然这些模型令人印象深刻，但需要大量的 GPU 资源，本地部署并不可行。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/collections/nvidia/nemotron-4-340b-666b7ebaf1b3867caf2f1911">Nemotron 4 340B - NVIDIA 集合</a>：未找到描述</li><li><a href="https://tenor.com/view/wolf-of-wall-street-rookie-numbers-gif-20904132">华尔街之狼 Rookie Numbers GIF - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/bilibili/Index-1.9B">GitHub - bilibili/Index-1.9B</a>：通过在 GitHub 上创建账号来为 bilibili/Index-1.9B 的开发做出贡献。</li><li><a href="https://huggingface.co/IndexTeam/Index-1.9B-Chat">IndexTeam/Index-1.9B-Chat · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1250904649360343180)** (114 messages🔥🔥): 

- **模型性能中的 RAM 与 VRAM 混淆**：成员们讨论了在加载适合 GPU VRAM 的模型时，系统 RAM 使用情况的混淆。尽管有足够的 VRAM，较大的模型仍会消耗大量系统 RAM，从而降低性能或引发问题。
- **'mlock' 和 'mmap' 参数影响 RAM 使用**：调整如 `use_mmap` 等参数会显著影响模型加载期间的 RAM 使用。禁用 `use_mmap` 可大幅减少内存占用，从而显著提升性能。
- **多 GPU 设置的硬件建议**：讨论了各种硬件配置，特别是涉及 RTX 3090 和 Tesla P40 等 GPU 的配置，建议保持驱动程序更新并确保正确的多 GPU 设置。分享了具体的模型和设置，突出了不同的性能和 RAM 利用场景。
- **LM Studio 0.2.24 版本的问题**：用户在 LM Studio 0.2.24 版本中遇到了严重的 RAM 问题和模型加载问题。一些人建议回退到以前的版本，但由于旧版本无法获取，出现了一些困难。
- **适用于多 GPU 的服务器机架**：一位成员寻求关于容纳多个 Tesla P40 GPU 的服务器设置建议，引发了搜索先前讨论并咨询有经验社区成员的建议。

**提到的链接**：<a href="https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main">TheBloke/Llama-2-7B-Chat-GGUF at main</a>：未找到描述

  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1250916712648278056)** (26 messages🔥): 

- **修复 Linux 上的 GPU Offload 错误**：一位用户报告了由 GPU Offload 引起的退出代码为 132 的错误。该问题通过在 Chat 页面 -> Settings Panel -> Advanced Settings -> GPU Acceleration 中关闭 GPU Offload 得到解决。

- **NVIDIA GPU 未被识别**：一位在 Linux (Kali) 上使用 NVIDIA GTX 1650 Ti Mobile 的用户面临 GPU 无法检测的问题。建议包括安装正确的 NVIDIA 驱动程序并确保 libcuda1 等包已就位。

- **LM Studio 0.2.25 发布**：宣布了 LM Studio 0.2.25 的新发布候选版本，其特点是修复了 Token 计数显示和 GPU Offload 的错误，并增强了 Linux 的稳定性。提供了 Mac、Windows 和 Linux 最新版本的下载链接，以及 AMD ROCm 和 OpenCL 可选扩展包的说明。

- **Phi-3 模型和 Smaug-3 Tokenizer**：一位成员对缺乏对 Phi-3 small 的支持表示沮丧，而另一位成员指出新版本修复了 Smaug-3 Tokenizer 问题。他们还以轻松的评论表达了对快速更新的赞赏。

- **LM Studio 扩展包**：用户对 LM Studio 新推出的扩展包很感兴趣。一位用户提到，虽然 0.2.25 可以运行 DeepSeek V2 Lite，但仍显示为不支持。
  

---


### **LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1251062969794691142)** (1 messages): 

- **API Key 错误令用户感到沮丧**：一位成员在 Playground 中查询工作流时遇到 API Key 错误，尽管多次验证并更改了 Key，仍收到带有 `invalid_api_key` 的 `401` 错误。他们注意到，虽然模型设置上的测试按钮可以工作，但 **LM Studio** 没有收到来自 **AutoGen** 的任何点击（hits）。
  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1250919845533323457)** (1 messages): 

- **Sakana AI 的 DiscoPOP 取得新突破**：Sakana AI 推出了一种新的直接偏好优化（Direct Preference Optimization）训练方法，这是通过对大型语言模型（LLM）的实验发现的。查看 [博客文章](https://sakana.ai/llm-squared/) 了解更多详情，并在 Hugging Face [此处](https://huggingface.co/lmstudio-community/DiscoPOP-zephyr-7b-gemma-GGUF/) 下载模型。
  

---



### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1250918299160678593)** (1 messages): 

- **Paul M. Nakasone 加入 OpenAI 董事会**：退役美国陆军上将 Paul M. Nakasone 已被任命为 OpenAI 董事会成员。他带来了**世界级的网络安全专业知识**，这对于保护 OpenAI 系统免受复杂威胁至关重要。[阅读更多](https://openai.com/index/openai-appoints-retired-us-army-general)。
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1250894062110900314)** (203 条消息🔥🔥): 

- **OpenAI API 支付问题令用户沮丧**：多位用户表达了由于“请求过多（too many requests）”错误而无法通过 OpenAI 平台进行支付的挫败感。OpenAI 支持团队建议等待 2-3 天并提供了一些故障排除步骤，但用户仍对影响应用程序功能的延迟感到不满。

- **ChatGPT 聊天记录丢失担忧**：一位用户对在多个设备和浏览器上丢失了 18 个月的 ChatGPT 聊天记录表示担忧。其他用户建议清除缓存、使用无痕模式（incognito mode）并通过 2FA 确保账号安全，但问题仍然存在，引发了关于账号被盗（account compromise）的猜测。

- **对特定平台发布的挫败感**：一位用户发泄了对 ChatGPT Win11 版本相对于 macOS 版本延迟发布的挫败感。这种情绪凸显了尽管微软对 OpenAI 进行了巨额投资，但 Windows 用户仍感受到了被忽视。

- **探索替代 AI API**：成员们讨论了 OpenAI API 的替代方案，例如 OpenRouter，以供面临问题或寻找备选方案的用户参考。尽管某些 API 存在兼容性问题，但用户注意到使用 OpenRouter 等工具进行代码迁移非常便利。

- **关于越狱（Jailbreak）和 AI 角色扮演的讨论**：成员们辩论了“越狱（jailbreaking）” AI 模型的概念，一些人认为这本质上是复杂的角色扮演，而不是解锁隐藏功能。这源于关于需要安全防护措施（safeguards）以防止有害输出的更广泛对话。
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1251030036144979989)** (26 条消息🔥): 

```html
- **GPT-4 doesn't learn after training**: Users discussed the misconception that GPT-4 agents can learn after initial training. Clarification was provided that *uploaded files are saved as "knowledge" files* but *do not continually modify the agent’s base knowledge*.
  
- **Differences between Command R and Command R+**: Members compared the accuracy of results from Command R and Command R+ for a calculation puzzle. Notably, *Command R+ was more accurate*, solving the puzzle correctly at *8 games played*, while Command R concluded with *11 games*.

- **GPT-3.5 Turbo API isn't free**: One user mistakenly believed GPT-3.5 Turbo to be free, but it was clarified that *the API is prepaid* and requires purchasing credits for continued use.

- **Disable GPT-4o tools issue**: A member faced difficulties disabling GPT-4o tools, impacting their important chat. A suggestion was made to customize settings via the menu, but they couldn't find the option due to language barriers.

- **Embedding vectors from OpenAI**: A query was raised about how the *text-embedding-ada-002* generates vector outputs like [-0.015501722691532107, -0.025918880474352136]. The user was interested in whether this process utilizes transformers.
```
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1251172766678519829)** (3 条消息): 

- **在 DALL-E 中难以实现平滑着色（flat shading）**：一位用户咨询如何在 DALL-E 中生成没有光照和阴影的图像，具体目标是实现类似于仅带有反照率贴图（albedo texture）的 3D 模型的平滑着色。他们提到尝试了各种方法但均未成功，并就此问题寻求帮助。
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1251172766678519829)** (3 条消息): 

- **为 API 创建文档**：一位用户提出了“编写 API 文档并提供给 ChatGPT”的想法。这可能有助于提高 API 功能的易用性和理解度。
- **GPT-4 的独特介绍**：另一位用户声称拥有“最独特的 GPT-4 介绍”。对话中未提供介绍的具体细节。
- **在 DALL-E 中实现 3D 平滑着色（flat shading）**：一位用户咨询如何在 DALL-E 中生成“没有光照和阴影，只有平滑着色”的图像。他们在实现这种效果时遇到困难，并请求协助或技巧。
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1250891584774344844)** (112 条消息🔥🔥): 

- **DiscoPOP 在偏好优化中领先**：一位成员分享了 [Sakana AI 发现偏好优化（preference optimization）算法的方法](https://sakana.ai/llm-squared/)，重点介绍了 DiscoPOP。如其 [博客文章](https://sakana.ai/llm-squared) 中所述，该方法在取得更高分数的同时，与 base model 的偏差更小，表现优于 DPO 等其他方法。
- **建议替代量化策略**：另一位成员建议了一种模型的替代量化（quantization）策略，提议对 output 和 embed tensors 使用 f16，而对其他部分使用 q5_k 或 q6_k。他们认为将 outputs 和 embeds 量化为 q8 会显著降低模型性能。
- **GitHub 代码审查与协作**：多条消息讨论了对各种 GitHub 仓库的审查和潜在协作，包括 [KAN-Stem](https://github.com/emangamer/KAN-Stem) 和 [ViT-Slim](https://github.com/Arnav0400/ViT-Slim/tree/master/GLoRA) 等项目。用户们协同努力审查并改进代码实现。
- **寻求训练与故障排除方面的帮助**：多位用户寻求在 VMs 上训练模型、排除 iOS 上的 token 无效错误以及解决 stable diffusion 中“加载组件（loading components）”问题的帮助。针对这些问题，提供了删除 cache 并重试等详细步骤和建议。
- **语音转文本与说话人日志问题**：一位成员请求在具有强大说话人日志（diarization）能力的语音转文本（speech-to-text）解决方案方面提供帮助，提到了 Whisper 和 pyannote 等工具。他们正在寻求微调（fine-tuning）模型或寻找预微调解决方案的建议，以实现准确的说话人识别。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs//2306.07967">One-for-All: Generalized LoRA for Parameter-Efficient Fine-tuning</a>：我们提出了 Generalized LoRA (GLoRA)，这是一种用于通用参数高效微调任务的高级方法。作为对 Low-Rank Adaptation (LoRA) 的增强，GLoRA 采用了一个通用的 prompt 模块来优化...</li><li><a href="https://arxiv.org/abs/2307.05695">ReLoRA: High-Rank Training Through Low-Rank Updates</a>：尽管缩放（scaling）具有主导地位和有效性，并产生了拥有数千亿参数的大型网络，但训练过度参数化模型的必要性仍然知之甚少...</li><li><a href="https://fxtwitter.com/AdeenaY8/status/1801613282735792416">Adina Yakup (@AdeenaY8) 的推文</a>：Huggy 头巾已为社区准备就绪！🔥 如果你想要我们的新头巾，请在帖子中回复一个 🤗，我会发给你一个！等不及看到大家戴上它们了！🥳✨🤘</li><li><a href="https://huggingface.co/learn/diffusion-course/unit2/2">微调与引导 - Hugging Face Diffusion 课程</a>：未找到描述</li><li><a href="https://github.com/Arnav0400/ViT-Slim/tree/master/GLoRA">GitHub 上的 ViT-Slim/GLoRA</a>：我们 CVPR'22 论文“Vision Transformer Slimming: Multi-Dimension Searching in Continuous Optimization Space”的官方代码 - Arnav0400/ViT-Slim</li><li><a href="https://x.com/Thom_Wolf/status/1801536742211682542">Thomas Wolf (@Thom_Wolf) 的推文</a>：你知道 bert-based-uncased 在 Hugging Face hub 上已被下载 15 亿次吗？顶尖的 AI 模型正在达到 YouTube 病毒式传播视频的播放量级别。</li><li><a href="https://sakana.ai/llm-squared/">无标题</a>：未找到描述</li><li><a href="https://huggingface.co/SakanaAI/DiscoPOP-zephyr-7b-gemma">SakanaAI/DiscoPOP-zephyr-7b-gemma · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/emangamer/KAN-Stem">GitHub - emangamer/KAN-Stem</a>：尝试使用 gpt4o 创建 KAN stem 训练脚本 - emangamer/KAN-Stem</li><li><a href="https://github.com/huggingface/peft/pull/1835">来自 ViT-Slim/GLoRA 的 Glora 实现 · Pull Request #1835 · huggingface/peft</a>：更多信息见 https://arxiv.org/abs/2306.07967</li><li><a href="https://github.com/Arnav0400/peft/blob/main/src/peft/tuners/glora.py">peft/src/peft/tuners/glora.py · Arnav0400/peft</a>：🤗 PEFT：最先进的参数高效微调（Parameter-Efficient Fine-Tuning）。 - Arnav0400/peft
</li>
</ul>

</div>

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1250972596015599656)** (2 条消息): 

- **NAIRR 试点项目引起广泛关注**：由 NSF 领导的国家人工智能研究资源 (NAIRR) 试点项目正在征集探索性研究资助提案，以展示该计划的价值。详细信息和截止日期可以在 [NAIRR 任务组报告](https://www.ai.gov/wp-content/uploads/2023/01/NAIRR-TF-Final-Report-2023.pdf)和 [NAIRR 试点 NSF 网站](https://nairrpilot.org/opportunities/allocations)上找到。

- **寻求在教育中使用 LLM 的帮助**：一位成员正在寻求关于如何使用语言学习模型 (LLMs) 来教授章节内容的指导，特别是针对组织内容、确定重点以及提供示例。欢迎任何关于将 LLMs 用于教育目的的方向或建议。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.nsf.gov/pubs/2024/nsf24093/nsf24093.jsp?org=CISE">致同事信：国家人工智能研究资源 (NAIRR) 试点示范项目 (nsf24093) | NSF - 美国国家科学基金会</a>：未找到描述</li><li><a href="https://nairrpilot.org/opportunities/allocations">NAIRR 试点 - NAIRR 试点资源申请以推进 AI 研究</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1250992360653848657)** (4 条消息): 

- **新的 OpenAI GitHub playground**：一位成员介绍了他们的新项目 [OpenAI playground 克隆版](https://github.com/neeraj1bh/openai-playground)。该项目已在 GitHub 上发布，欢迎贡献和实验。

- **分享双曲 KG 嵌入论文**：分享了一篇关于双曲知识图谱 (KG) 嵌入的论文，强调该方法 *“结合了双曲反射、旋转以及 Attention 机制，用于建模复杂的各种关系模式”*。您可以在 [arXiv 上的 PDF](https://arxiv.org/abs/2005.00545) 中阅读更多内容。

- **重点介绍 OpenVLA 模型**：分享了一篇关于 OpenVLA（一个开源视觉-语言-动作模型）的论文。完整的摘要和作者详情可以在 [arXiv 出版物](https://arxiv.org/abs/2406.09246)中找到。

- **为量子研究引入 QubiCSV**：介绍了一个名为 QubiCSV 的新平台，旨在为协作量子研究提供 *“量子比特控制存储与可视化”*。详细的摘要和实验性 HTML 视图可在 [arXiv](https://arxiv.org/abs/2403.14672) 上查看。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2406.09246">OpenVLA: 一个开源视觉-语言-动作模型</a>：在互联网规模的视觉语言数据和多样化机器人演示组合上预训练的大型策略，具有改变我们教授机器人新技能方式的潜力：与其训练新的...</li><li><a href="https://arxiv.org/abs/2403.14672">QubiCSV: 一个用于协作量子比特控制的开源数据存储与可视化平台</a>：开发用于量子比特控制的协作研究平台对于推动该领域的创新至关重要，因为它们能够实现思想、数据和实现的交流，从而实现更具影响力的...</li><li><a href="https://arxiv.org/abs/2005.00545">低维双曲知识图谱嵌入</a>：知识图谱 (KG) 嵌入学习实体和关系的低维表示以预测缺失的事实。KG 通常表现出层次化和逻辑模式，这些模式必须保留在...</li><li><a href="https://github.com/neeraj1bh/openai-playground">GitHub - neeraj1bh/openai-playground: OpenAI playground 克隆版</a>：OpenAI playground 克隆版。通过在 GitHub 上创建账号来为 neeraj1bh/openai-playground 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1250909960892715090)** (6 条消息): 

- **感知机逻辑运算符详解**：一位成员分享了一篇关于使用逻辑运算符训练感知机的[博客文章](https://techartspider.com/post/perceptron-logical-operators/)，涵盖了基本概念和复杂细节。*“基本上是在学习过程中整理自己的思路。”*

- **Shadertoy 工具获得多通道（Multipass）和物理支持**：一位成员一直在学习 GPU，并将多通道支持集成到了他们的 [Shadertoy 工具](https://github.com/pygfx/shadertoy/pull/30)中，创建了一个交互式物理模拟。他们分享了一个 [shadertoy 示例](https://www.shadertoy.com/view/lXK3WV)并征求反馈。

- **Recall AI 的开源替代方案**：一位成员开发了 [LiveRecall](https://github.com/VedankPurohit/LiveRecall)，它可以记录并加密屏幕上的所有内容，以便通过自然语言进行搜索，将其定位为微软 Recall AI 的尊重隐私的替代方案。他们鼓励大家贡献代码，并强调这是一个需要支持的新项目。

- **研究可访问性平台发布**：一个名为 [PaperTalk](https://papertalk.io/) 的新平台发布，旨在利用 AI 工具让研究更易于获取。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://papertalk.io/">Papertalk.io - :Unlock the Power of Research: Swiftly, Simply, Smartly</a>：未找到描述</li><li><a href="https://techartspider.com/post/perceptron-logical-operators/">Training a Perceptron on Logical Operators</a>：机器学习领域的基本概念之一是感知机，这是一种受神经元功能启发的算法。</li><li><a href="https://github.com/VedankPurohit/LiveRecall">GitHub - VedankPurohit/LiveRecall: Welcome to **LiveRecall**, the open-source alternative to Microsoft&#39;s Recall. LiveRecall captures snapshots of your screen and allows you to recall them using natural language queries, leveraging semantic search technology. For added security, all images are encrypted.</a>：欢迎来到 **LiveRecall**，微软 Recall 的开源替代方案。LiveRecall 捕捉屏幕快照，并允许你利用语义搜索技术通过自然语言查询来召回它们。为了增加安全性，所有图像都经过加密。</li><li><a href="https://www.shadertoy.com/view/lXK3WV">Shadertoy</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1251184068729770034)** (1 条消息): 

- **Meta 的新论文探索像素级 Transformer**：一位成员分享了 Meta 最近发布的一篇题为《[An Image is Worth More Than 16×16 Patches: Exploring Transformers on Individual Pixels](https://arxiv.org/pdf/2406.09415)》的论文。他们指出，这项新工作建立在之前的展示基础之上，特别强调了 #Terminator 工作所展示的*像素级注意力（pixel-level attention）的优越性*。
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1251201521274720376)** (6 条消息): 

- **寻找 RealNVP 经验**：一位成员询问是否有人有使用 **RealNVP** 模型的经验。随后没有进一步的讨论。
- **基于 Java 的图像检测需求**：另一位成员寻求帮助，希望创建一个支持实时检测和通过上传进行自定义检测的图像检测系统，特别要求使用 **Java**。他们强调倾向于不涉及 Python 的解决方案。
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1251019516889141248)** (6 条消息): 

- **用于检测幻觉的新模型微调**：一位成员分享了一个用于检测幻觉的[新小语言模型微调](https://huggingface.co/grounded-ai/phi3-hallucination-judge-merge)链接。分享的统计数据突出了它在幻觉检测上的精确率（precision）、召回率（recall）和 F1 分数，准确率为 79%。

- **关于 DataTrove 库的查询**：一位用户询问在哪里可以咨询关于 **DataTrove 库**的问题，得到的建议是在聊天频道中提问。

- **Minhash 去重配置**：一位用户寻求关于在本地机器上使用 200 个 CPU 核心对 500GB 数据集进行 **minhash 去重**配置的建议，并提到自己缺乏多进程编程的经验。

- **销售电话分析寻求帮助**：另一位成员寻求关于分析 100 多个销售电话的指导，以提取影响购买决策的关键变量洞察。他们提到使用 **Whisper** 或 **Assembly** 进行转录，并使用 GPT 进行探索性数据分析（EDA）。

**提到的链接**：<a href="https://huggingface.co/grounded-ai/phi3-hallucination-judge-merge">grounded-ai/phi3-hallucination-judge-merge · Hugging Face</a>：未找到描述

  

---

### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1250906972669218989)** (13 messages🔥): 

- **Git 下载修复问题**：一位用户确认通过 git 下载解决了他们的问题，另一位用户表示希望这能有所帮助。

- **Dreambooth 脚本需要修改以增加灵活性**：一位成员指出，虽然 pipeline 具备处理更复杂训练场景的灵活性，但基础的 dreambooth 脚本并未配备处理 SD3 模型单个 caption 的功能。另一位用户强调，为了实现这些训练特性，修改脚本是必要的。

- **Tokenizers 训练不在基础脚本中**：用户讨论了训练 CLIP 和 T5 等 tokenizers 的可能性，提到这会显著增加 VRAM 需求，且不包含在基础 dreambooth 脚本中。

- **Diffusion 模型中的低 MFU 分析**：一位用户分享了 [HDIT 的分析数据](https://github.com/Muhtasham/k-diffusion/blob/master/run_profile.sh)，并询问了为什么 diffusion 模型在 A100 上的模型算力利用率 (MFU) 比起像 nanogpt 这样的语言模型要低。他们邀请社区其他成员提供理论和见解。

- **寻求 Meme 生成器模型的建议**：一位用户正在开发一个高质量的 meme 生成器模型，并向有经验的成员寻求入门建议。他们欢迎大家提供见解和推荐。

**提及的链接**：<a href="https://github.com/Muhtasham/k-diffusion/blob/master/run_profile.sh">k-diffusion/run_profile.sh at master · Muhtasham/k-diffusion</a>：Karras 等人 (2022) 的 PyTorch diffusion 模型。通过在 GitHub 上创建账号为 Muhtasham/k-diffusion 的开发做出贡献。

  

---



### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/)** (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=xIDMPUYpd_0
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1250964315444351075)** (5 messages): 

- **Magpie 扩展了自合成指令数据**：在他们的新预印本中，[Magpie](https://arxiv.org/abs/2406.08464) 使用 "prompting-with-nothing" 方法自合成了 400 万个数据点，显著提升了 Llama-3-8B-base 在 AlpacaEval、ArenaHard 和 WildBench 等对齐基准测试中的表现。该方法可迁移至 Qwen 1.5 4B & 7B 等更小的模型，展示了持续改进对额外数学和推理数据的需求。

- **Samba 通过线性复杂度实现无限上下文**：[Samba 模型](https://x.com/_philschmid/status/1801516284267356217) 通过在 Mamba 中加入 MLP 和 Sliding Window Attention 构建而成，在各项基准测试中超越了 Phi-3-mini。Samba-3.8B-instruct 展示了精心设计的模型架构如何高效地实现无限上下文长度。

- **NVIDIA 的 Nemotron-4 340B 模型家族**：[Nemotron-4 340B 技术报告](https://research.nvidia.com/publication/2024-06_nemotron-4-340b) 介绍了在宽松的开放获取许可下的 Nemotron-4-340B-Base、Nemotron-4-340B-Instruct 和 Nemotron-4-340B-Reward 模型。这些模型高度依赖合成生成的数据（超过 98%），并设计为在具有 8 个 GPU 的 DGX H100 上以 FP8 精度运行。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/_philschmid/status/1801516284267356217">Philipp Schmid (@_philschmid) 的推文</a>: 这就是我们所说的 Bingo 吗？🎯 "Samba = Mamba + MLP + Sliding Window Attention + 层级 MLP 堆叠。" => 具有线性复杂度的无限上下文长度 Samba-3.8B-instruct ...</li><li><a href="https://x.com/billyuchenlin/status/1801364763055952029?t=_0KG0V3yiIRcm4Ix09QiVQ&s=19">Bill Yuchen Lin 🤖 (@billyuchenlin) 的推文</a>: 如果我们用空内容去提示像 Llama-3-Instruct 这样对齐过的 LLM 会怎样？🤔 出乎意料的是，由于其自回归特性，它会解码出相当不错的用户查询。在我们新的预印本 Magpie🐦‍⬛ 中，我们发现这...</li><li><a href="https://research.nvidia.com/publication/2024-06_nemotron-4-340b">Nemotron-4 340B | Research</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1250887609807409243)** (69 messages🔥🔥): 

- **Roblox 聚会趣闻**：一位成员分享了 VLLM GitHub 页面上的趣事，特别提到了关于 Roblox 聚会的幽默章节。这引起了讨论中其他人的笑声和赞同。
  
- **DiscoPOP 的优化突破**：重点介绍了 Sakana AI 提出的一种名为 DiscoPOP 的新方法，该方法在偏好优化方面比 DPO 等现有方法获得了更好的分数。随模型一起分享了一篇 [博客文章](https://sakana.ai/llm-squared)，强调了其高 Reward 和更低的 KL Divergence 指标。

- **对 AI 和监视的担忧**：一些成员对前 NSA 负责人 Paul M. Nakasone 被任命为 OpenAI 董事会成员表示担忧，担心政府会更深入地介入并增加监视。他们链接了 [文章](https://www.theverge.com/2024/6/13/24178079/openai-board-paul-nakasone-nsa-safety) 并讨论了这一决定的影响。

- **实时 TTS 进展**：一位用户向社区更新了一个 TTS 项目的进展，称通过 Styletts2 和使用 flash attention 实现了 10 倍的实时速度。他们提到了正在进行的重构工作，旨在实现包括身份和记忆在内的未来功能。

- **合成数据生成模型**：讨论了 [Nemotron-4-340B-Instruct](https://huggingface.co/nvidia/Nemotron-4-340B-Instruct) 模型的引入，指出其在合成数据生成以及 Supervised Fine-tuning 和 Preference Optimization 等对齐步骤中的应用。对话强调了在 NVIDIA Open Model License 下进行商业用途的潜力。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.theverge.com/2024/6/13/24178079/openai-board-paul-nakasone-nsa-safety">前 NSA 负责人加入 OpenAI 董事会</a>：另一位新董事会成员。</li><li><a href="https://huggingface.co/spaces/stabilityai/stable-diffusion-3-medium">Stable Diffusion 3 Medium - stabilityai 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://sakana.ai/llm-squared/">无标题</a>：未找到描述</li><li><a href="https://huggingface.co/SakanaAI/DiscoPOP-zephyr-7b-gemma">SakanaAI/DiscoPOP-zephyr-7b-gemma · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/papers/2406.06282">论文页面 - PowerInfer-2: 智能手机上的快速大语言模型推理</a>：未找到描述</li><li><a href="https://huggingface.co/nvidia/Nemotron-4-340B-Base">nvidia/Nemotron-4-340B-Base · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/nvidia/Nemotron-4-340B-Instruct">nvidia/Nemotron-4-340B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/godfather-massacre-sad-gif-16810633">教父大屠杀 GIF - Godfather Massacre Sad - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/snowden-gif-12645540747659800860">斯诺登 GIF - Snowden - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1250932623266680943)** (7 messages): 

- **使用新数据更新 LLM 仍是挑战**：一篇分享的 [arXiv 论文](https://arxiv.org/abs/2402.12847v1) 讨论了通过对新文档进行持续预训练，然后进行 QA 对训练来更新 **LLM** 的标准方法仍然存在问题。作者建议在文档训练之前让 **LLM** 接触 QA 对，以便更好地编码知识以回答问题。 
- **探讨 RAG 背景下的引用**：一位成员对构建检索增强生成 (RAG) 系统时的 **引用管理算法** 表示感兴趣。他们承认由于应用场景众多，这具有复杂性。
- **讨论 Perplexity 工具的局限性**：有人指出 **Perplexity** 的免费版本无法处理 PowerPoint 文档，但在这些文档上使用引用被视为一个引人注目的用例。 



**提及的链接**：<a href="https://arxiv.org/abs/2402.12847v1">经过指令微调的语言模型是更好的知识学习者</a>：为了使基于大语言模型 (LLM) 的助手能够有效地适应不断变化的信息需求，必须能够通过对新数据的持续训练来更新其事实知识...

  

---

### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1250890907948023868)** (39 messages🔥): 

- **ONNX 转换加速 CPU 测试**：讨论中提到另一个 Discord 上的 ONNX 转换据称能让 CPU 速度提升 2 倍。关于测试，有人指出：“仅限 CPU，我不认为 ONNX 在 GPU 上会有很大加速。”

- **为 RAG 数据集总结代码库**：一个 GitHub 工具 [code2prompt](https://github.com/mufeedvh/code2prompt) 有助于将代码库转换为用于 RAG 数据集的 Markdown 提示词。成员们讨论了它的使用，并指出生成的 Token 数量可能会非常高。

- **模型上下文处理及限制**：成员们注意到在当前的 GPU 硬件上进行训练时，处理超过 8k 的上下文长度存在困难。他们讨论了截断上下文以实现更好的模型训练，并确保样本需要多方面的信息综合。

- **RAG 数据集与上下文长度**：在集思广益中，成员们提议对代码库的 Markdown 进行分块（chunking）和索引（indexing），以便在 RAG 中使用扩展上下文，而不依赖于模型的输出能力。

- **需要更复杂的示例查询**：重点被放在创建复杂的示例查询上，这些查询要求模型从多个来源综合信息，而不是简单的基于事实的查询。这包括将查询分解为多个搜索查询，以实现更好的信息检索。

**提及的链接**：<a href="https://github.com/mufeedvh/code2prompt">GitHub - mufeedvh/code2prompt: A CLI tool to convert your codebase into a single LLM prompt with source tree, prompt templating, and token counting.</a>：一个将代码库转换为带有源树、提示词模板和 Token 计数的单个 LLM 提示词的 CLI 工具。 - mufeedvh/code2prompt

  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1250969824100749333)** (2 messages): 

- **WorldSim 提示词已开源**：**原始 WorldSim 提示词**可以通过在 Twitter 上搜索找到。尽管提到了切换到 **Sonnet 模型**，但这被暗示为关于不同模型能力讨论的一部分。
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1251107817637543986)** (9 messages🔥): 

- **Mojo 的包管理器仍在开发中**：Mojo 包管理器目前“仍在开发中”。目前，用户需要从源码编译其他包，或者从 Release 中下载为 mojopkg 文件。

- **设置仓库的最佳实践**：大家共同希望能有一个关于设置仓库最佳实践的指南。一位成员建议在官方指南发布前，先参考 Modular 在 GitHub 上的做法。

- **马萨诸塞州 Mojo 爱好者聚会**：一位成员正寻求联系马萨诸塞州的其他开发者，以便在喝咖啡时讨论 Mojo。他们很高兴能建立联系并亲自分享知识。

- **关于 `def` 和 `fn` 函数的讨论**：需要对提供的 [Mojo 函数手册](https://docs.modular.com/mojo/manual/functions) 链接进行澄清。页面解释了 `def` 和 `fn` 函数具有不同的默认行为，它们之间的选择取决于个人编码风格和任务需求。

**提及的链接**：<a href="https://docs.modular.com/mojo/manual/functions">Functions | Modular Docs</a>：Mojo `fn` 和 `def` 函数介绍。

  

---

### **Modular (Mojo 🔥) ▷ #[tech-news](https://discord.com/channels/1087530497313357884/1151359062789857280/1250973002946838568)** (1 条消息): 

- **NSF NAIRR EAGER 资助开放提案申请**：美国国家科学基金会 (NSF) 正在接受国家 AI 研究资源 (NAIRR) 试点项目下的 **探索性研究早期概念资助 (EAGER)** 提案。提案每月评审一次，资助期限为 12 个月。根据 [NSF 公告](https://www.nsf.gov/pubs/2024/nsf24093/nsf24093.jsp?org=CISE)，该资助面向美国的科研人员和教育工作者。

- **NAIRR 试点愿景与社区建设**：符合 **NAIRR 试点** 愿景的活动包括促进计算资源申请、集成数据和工具，以及建立强大的用户社区。更多详情请参阅 [NAIRR 工作组报告](https://www.ai.gov/wp-content/uploads/2023/01/NAIRR-TF-Final-Report-2023.pdf)。

- **资源获取与提案截止日期**：由 NSF、DOE 及各合作伙伴支持的 NAIRR 试点项目允许研究社区访问计算和教育资源，详情见 [NSF NAIRR 试点网站](https://nairrpilot.org/opportunities/allocations)。提案必须在每月 15 日前提交评审，结果将在月底公布。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.nsf.gov/pubs/2024/nsf24093/nsf24093.jsp?org=CISE">致同事信：国家人工智能研究资源 (NAIRR) 试点示范项目 (nsf24093) | NSF - 美国国家科学基金会</a>: 未找到描述</li><li><a href="https://nairrpilot.org/opportunities/allocations">NAIRR 试点 - 推进 AI 研究的 NAIRR 试点资源申请</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1250899693463015455)** (92 messages🔥🔥): 

- **Mojo 的 GPU 支持引发褒贬不一的反应**：一位用户询问了 **Mojo 中 GPU 支持**的发布日期，回复中对目前的就业机会持怀疑态度，但也承认 GPU 支持是备受期待的。分享了一个与 Mojo 相关的[职位空缺链接](https://www.modular.com/careers#open-roles)，指明了一些就业途径。

- **Mojo 与 CUDA 之争**：讨论强调，虽然 **CUDA 已经成熟可用**，但 Mojo 的强类型特性以及对各种加速器的潜在支持可能使其成为一个极具吸引力的替代方案。用户们辩论了 Mojo 和 Modular 相比 NVIDIA 解决方案的可行性，以及对排他性技术的潜在需求。

- **探索多线程与 MLIR**：一位用户询问了 Mojo 如何处理**多线程**，并参考了 [SIMD 指令、unroll_factor 和并行化](https://www.modular.com/blog/fast-k-means-clustering-in-mojo-guide-to-porting-python-to-mojo-for-accelerated-k-means-clustering)。另一位用户分享了一个详细介绍 [各种 MLIR dialects](https://mlir.llvm.org/docs/Dialects/)（如 amdgpu、nvgpu 和 XeGPU）的链接，这些对可移植性非常有用。

- **对 Modular 商业模式的推测**：几位用户辩论了 **Modular 的商业模式是否可行**，讨论了来自支持服务、咨询以及通过 MAX 进行推理/训练的潜在收入。共识是，虽然 Mojo 尚未达到生产就绪状态，但其更好的工具链承诺可能使其在面对 CUDA 等成熟竞争对手时处于有利地位。

- **对开源和供应商锁定的担忧**：人们对**供应商锁定（Vendor Lock-In）**和对开源贡献的依赖表示担忧，并将其与公司对专有解决方案的需求进行了类比。一些用户认为，新的芯片公司可能更倾向于使用 Mojo 以在技术优势而非供应商排他性上进行竞争，这反映了对现有工具和环境的挫败感。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.modular.com/blog/fast-k-means-clustering-in-mojo-guide-to-porting-python-to-mojo-for-accelerated-k-means-clustering">Modular: Mojo🔥 中的快速⚡ K-Means 聚类：将 Python 移植到 Mojo🔥 以实现加速 K-Means 聚类的指南</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：Mojo🔥 中的快速⚡ K-Means 聚类：将 Python 移植到 Mojo🔥 以实现加速 K-Means 聚类...</li><li><a href="https://www.modular.com/careers#open-">Modular: 职业生涯</a>：在 Modular，我们相信优秀的文化是创建伟大公司的关键。我们工作的三个支柱是：打造用户喜爱的产品、赋能员工以及成为一支不可思议的团队。</li><li><a href="https://www.modular.com/careers#open-roles">Modular: 职业生涯</a>：在 Modular，我们相信优秀的文化是创建伟大公司的关键。我们工作的三个支柱是：打造用户喜爱的产品、赋能员工以及成为一支不可思议的团队。</li><li><a href="https://mlir.llvm.org/docs/Dialects/">Dialects - MLIR</a>：未找到描述</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/docs/vision.md">mojo/stdlib/docs/vision.md at main · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://docs.google.com/document/d/1iS0_4q7icTuVK6PPnH3D_9XmdcrgZq6Xv2171nS4Ztw/edit#heading=h.oggqptmb1frj">MLIR C/C++ 前端工作组</a>：MLIR C/C++ 前端工作组，每月第一个周一，太平洋时间上午 9 点。Discord: #clangir, … 日历：如果你想被添加到 Google 日历邀请中，请发送你的 Google 账号邮箱地址...</li><li><a href="https://www.theverge.co">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1251052007683526766)** (3 messages): 

- **新的 Nightly Mojo 编译器发布**：新的 Nightly Mojo 编译器已发布，更新至 `2024.6.1405`。你可以通过 `modular update nightly/mojo` 进行更新，并查看 [原始差异 (raw diff)](https://github.com/modularml/mojo/compare/7963ca681da2de473042a76bfe27e2ebafeb4d39...1130fdb81d763066d8e5bcb2226fe270981d3b0a) 以及 [当前的变更日志 (changelog)](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)。

- **贡献 Mojo**：一位成员询问了关于快速上手 Mojo 开发的资源，并提到难以确定从哪里开始。他们被引导至 [贡献指南 (contributing guide)](https://github.com/modularml/mojo/blob/nightly/CONTRIBUTING.md)，该指南要求具备 Python 和 Mojo 文档知识，并建议从标记为 "good first issue" 的议题开始。

**提到的链接**：<a href="https://github.com/modularml/mojo/blob/nightly/CONTRIBUTING.md">mojo/CONTRIBUTING.md at nightly · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。

  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1250890929297035264)** (92 messages🔥🔥): 

```html
- **Perplexity 服务器宕机；引发混乱**：多名用户报告了 Perplexity 服务器的问题，经历了重复消息和无限循环，导致情绪沮丧。一位用户评论道，“如果网站正在维护，他们甚至没有宣布要这样做，” 强调了 Perplexity 团队缺乏沟通。
- **文件上传问题已确认**：一位用户指出，损坏的文件上传功能正导致性能问题，具体提到“Perplexity 大约有两个月的时间里，某些处于特定 AB test 配置下的用户文件读取功能是损坏的。” 这一问题得到了 Perplexity 支持团队的确认，并将其归因于新功能的部分回滚。
- **生成的链接出现 404 错误**：用户对 Perplexity 生成错误或不存在的链接表示担忧，其中一人表示，“100% 的链接都指向 404 类型的页面。” 讨论表明，这可能是由于 LLM 在捏造 URL 而非引用真实链接。
- **Android 应用不一致性**：Perplexity Android 应用存在明显的不一致性，请求会周期性地在未执行的情况下重新发送，而这在 iOS 或网页端并未观察到。一位用户强调，“这个问题是在上周 Perplexity 运行出现错误后开始的。”
- **Pro 订阅担忧**：由于持续存在的问题以及支持团队的沟通不畅，几位用户对升级到 Perplexity Pro 表示怀疑。一位沮丧的用户评论道，“看来 Perplexity 终究还是不值得。”
```

**提到的链接**：<a href="https://www.reddit.com/r/perplexity_ai/s/pG41duXQBu">Reddit - 深入探索一切</a>：未找到描述

  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1250930763226943602)** (9 messages🔥): 

- **探索 Dimension-5 Second**：分享了一个指向 [Dimension-5 Second](https://www.perplexity.ai/page/Dimension-5-Second-zoTUGR4NSNeUSMBN5VEOAQ) 的链接，邀请成员深入探讨该话题。
- **马斯克撤回诉讼**：一位成员发布了关于 [马斯克撤回诉讼](https://www.perplexity.ai/page/Musk-Drops-Lawsuit-LLG_ToBhQ2..DzJ3e1RKJQ) 的链接，强调了近期与 Elon Musk 相关的重大法律新闻更新。
- **情境感知指南**：分享了一篇关于 [情境感知 (Situational Awareness)](https://www.perplexity.ai/page/Situational-Awareness-Su6wFi3KTmGAZaKeB86zNw) 的文章链接，提供了在各种情况下保持感知能力的见解。
- **阿根廷国家足球队**：通过此链接分享了关于 [阿根廷国家足球队](https://www.perplexity.ai/page/Argentina-National-Football-tfR94mUpSDqOmlkvAw6Lhw) 的信息和更新。
- **苹果股价飙升**：一位成员分享了关于 [苹果股价飙升](https://www.perplexity.ai/page/Apple-stock-soars-fyHz1_JMTJSIpRgN3ABuJQ) 的链接，指向了一个值得关注的财经新闻事件。

**提到的链接**：<a href="https://www.youtube.com/embed/EOdZIHKsihY">YouTube</a>：未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1251110519398600734)** (2 messages): 

- **寻找 Web3-API 开发者**：一位成员询问，“这里有人在使用 API 构建 Web3 原生项目吗？”他们还询问是否有人“将其连接到区块链端点 (blockchain endpoints)”。
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1250913065386508350)** (1 条消息): 

- **Atomicwork 为其 AI 助手集成 LlamaIndex**：**Atomicwork** 团队开发了他们的 AI 助手 **Atom**，利用 **LlamaIndex** 处理多种数据格式。该功能确保了准确、安全且高效的数据检索，从而增强了决策过程。[来源](https://twitter.com/llama_index/status/1801354119682343059)
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1250889957912870924)** (63 条消息🔥🔥): 

- **ReAct Agent 配置中的参数错误**：一位成员在配置 ReAct Agent 时因使用了错误的 kwarg `max_iteration` 而遇到错误。另一位成员指出正确的术语应该是 `max_iterations`。

- **带有 Document Agent 的 Recursive Retriever 问题**：一位成员在参考 [recursive retrievers with document agents](https://docs.llamaindex.ai/en/stable/examples/query_engine/recursive_retriever_agents/) 指南从 Pinecone 向量存储加载索引时遇到困难。他们分享了代码细节以便获得更好的帮助。

- **关于在 Query Engine 中使用 Vercel AI 的讨论**：一位成员询问 Query Engine 是否支持 Vercel AI 向量流，并得到的建议是使用 FastAPI 和 StreamingResponse 作为解决方案。

- **对 Weaviate 集成中多租户（Multi-Tenancy）功能的关注**：来自 Weaviate 的成员讨论了添加多租户功能以实现更好的数据隔离，并邀请大家在 [GitHub issue](https://github.com/run-llama/llama_index/issues/13307) 上提供反馈。

- **构建自定义 Agent 的学习资源**：一位成员寻求构建自定义 Agent 的资源，多位成员提供了 [LlamaIndex 文档](https://docs.llamaindex.ai/en/stable/examples/agent/custom_agent/) 以及与 DeepLearning.AI 合作的[短课](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex/)链接。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex/?utm_campaign=llamaindexC2-launch">Building Agentic RAG with LlamaIndex</a>：学习如何构建一个能够对文档进行推理并回答复杂问题的 Agent。师从 LlamaIndex 的联合创始人兼 CEO。</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine/">Sub Question Query Engine - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/custom_agent/?h=custom+agent">Building a Custom Agent - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/question_gen/llm_generators.py">llama_index/llama-index-core/llama_index/core/question_gen/llm_generators.py at main · run-llama/llama_index</a>：LlamaIndex 是适用于您的 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/379503696e59d8b15befca7b9b21e1675db17c50/llama-index-core/llama_index/core/query_engine/sub_question_query_engine.py#L107">llama_index/llama-index-core/llama_index/core/query_engine/sub_question_query_engine.py at 379503696e59d8b15befca7b9b21e1675db17c50 · run-llama/llama_index</a>：LlamaIndex 是适用于您的 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/FaissIndexDemo/?h=faiss">Faiss Vector Store - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/query_engine/recursive_retriever_agents/">Recursive Retriever + Document Agents - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/issues/13307">[Feature Request]: Add support for multi-tenancy in weaviate · Issue #13307 · run-llama/llama_index</a>：功能描述：在 Weaviate 中添加多租户（MT）支持。原因：Weaviate 中的 MT 需要修改架构（启用 MT）和 CRUD 操作（传递租户名称）。我们该如何更新 llamain...</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/multi_tenancy/multi_tenancy_rag/#define-query-engines">Multi-Tenancy RAG with LlamaIndex - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/llm/anthropic/#anthropic>)">Anthropic - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/property_graph/property_graph_custom_retriever/#build-the-property-graph>)">Defining a Custom Property Graph Retriever - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1250895716319101001)** (6 messages): 

- **酷炫 Demo 的新频道！**：查看新创建的频道 <#1250895186616123534>，用于展示酷炫的项目和 Demo。实时、非结构化的 Demo 可以在 **General 语音频道**中讨论。
- **Lamini Memory Tuning 探索**：Lamini 分享了他们创新的 Memory Tuning 方法，该方法提高了事实准确性并减少了幻觉，为一家财富 500 强客户实现了高达 95% 的准确率。[阅读更多关于这一突破的信息](https://www.lamini.ai/blog/lamini-memory-tuning)及其[研究论文](https://github.com/lamini-ai/Lamini-Memory-Tuning/blob/main/research-paper.pdf)。
- **用于 LLM 微调的 Helix AI**：一位成员询问是否有人尝试过使用 **Helix AI** 进行 LLM 微调。它承诺在保证安全、可扩展和私密的同时提供最佳的开源 LLM，并提供可选的闭源模型透传功能。[在此尝试](https://tryhelix.ai/)。
- **对 vLLM 中 FP8 的期待**：FP8 推理因降低延迟、提高吞吐量以及在准确率下降极小的情况下减少内存占用而备受赞誉。[查看准确的预量化 Checkpoints](https://x.com/mgoin_/status/1801633700112662689?s=46) 和 [关于 FP8 的 vLLM 文档](https://docs.vllm.ai/en/stable/quantization/fp8.html)。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://tryhelix.ai/">Private GenAI Platform &ndash; Helix AI</a>：未找到描述</li><li><a href="https://x.com/mgoin_/status/1801633700112662689?s=46">Michael Goin (@mgoin_) 的推文</a>：很高兴 vLLM 中的 FP8 随着我们的投入变得越来越好。查看我们收集的准确、预量化的 Checkpoints！https://huggingface.co/collections/neuralmagic/fp8-llms-f...</li><li><a href="https://www.lamini.ai/blog/lamini-memory-tuning">Introducing Lamini Memory Tuning: 95% LLM Accuracy, 10x Fewer Hallucinations | Lamini - Enterprise LLM Platform</a>：未找到描述</li><li><a href="https://github.com/lamini-ai/Lamini-Memory-Tuning/blob/main/research-paper.pdf">Lamini-Memory-Tuning/research-paper.pdf at main · lamini-ai/Lamini-Memory-Tuning</a>：消除 LLM 幻觉需要重新思考泛化 - lamini-ai/Lamini-Memory-Tuning
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1251132533824032799)** (6 messages): 

- **用户寻求额外额度**：由于在持续跟进课程方面遇到挑战或为了持续使用（如训练 Embedding 模型），成员们请求额外的额度。一位成员表示，如果获得额度，计划就所学内容编写教程。
- **Modal 训练超过 24 小时**：一位成员询问在 Modal 中使用 Axolotl 示例时，训练时间超过 24 小时的问题。另一位成员确认有一个参数可以从之前的运行中恢复训练，为该问题提供了解决方法。
- **社区支持的 Slack 链接**：分享了一个指向 Slack 上类似查询的链接（[Slack 链接](https://modallabscommunity.slack.com/archives/C069RAH7X4M/p1718372196651179)），以提供更多背景和支持。该链接为寻找社区中讨论过的类似挑战的答案提供了途径。

**提及的链接**：<a href="https://modallabscommunity.slack.com/archives/C069RAH7X4M/p1718372196651179)">Slack</a>：未找到描述

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[learning-resources](https://discord.com/channels/1238365980128706560/1241089743933149204/1251054821990207569)** (2 messages): 

- **讨论输出验证技术**：一位成员询问输出验证是否涉及“结构化生成”，并提到使用 **outlines 库**。另一位成员澄清说，他们专注于特定用例的评估和断言。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jarvis-labs](https://discord.com/channels/1238365980128706560/1241117895740625099/1250906860337365052)** (5 messages): 

- **Jarvis 上的 Axolotl 需要 Ampere GPU 进行训练**：“我在 Jarvis 上使用 Axolotl 微调 openllama-3b 时遇到困难（我使用的是 RTX5000）。”该问题通过使用 **Ampere GPU** 得到了解决。
  
- **关于从 Jarvis 上传模型到 HF 的疑问**：“在将模型上传到 HF 后，我们需要删除实例吗，还是可以把它留在 Jarvis 上？”一位成员寻求关于从 **Jarvis 到 Hugging Face (HF)** 上传模型的正确流程以及上传后实例管理的指导。

- **lora-8b.yml 配置出现 sample_packing 错误**：“在使用 lora-8b.yml 配置微调 llama-3 时，我遇到了 sample_packing 错误。”疑问在于示例配置是否应该在不进行任何修改的情况下就能正常工作。

### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1251131043692417024)** (2 messages): 

- **带有自定义 head 的 LoRa 微调显示出不一致性**：一位用户表达了担忧，在使用 LoRa 微调 Causal LM 模型并添加池化层和自定义 head 后，他们在训练期间的评估表现良好，但从 adapter safetensors 进行推理时结果却大幅变差。他们询问是否需要保存基础模型权重才能进行正确的推理。
- **Hugging Face 的额度问题**：一位用户提到他们提交了额度申请表，但尚未收到额度，这表明额度分配过程存在延迟或问题。他们提供了用户名和联系邮箱以寻求帮助。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1250947958116188200)** (4 messages): 

- **Replicate 额度困惑**：一位成员在收到 Replicate 的邮件后，询问 Replicate 额度是否会在 24 小时内过期。另一位用户建议该警报可能是基于预计使用量发出的。
  
- **初始注册后未收到额度**：一位成员分享了他们的邮箱以检查额度状态，称在 *github@granin.net* 初始注册后没有收到任何额度或邮件。
  
- **通过 DM 等待额度**：另一位成员提到他们仍在等待接收兑换额度的链接，并已通过 DM 发送了他们的邮箱详情以进行跟进。他们表达了开始使用 Replicate 的迫切愿望。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1250905396663685160)** (3 messages): 

- **信用卡信息要求引发辩论**：一位成员对 **Langsmith** 在必要之前收集信用卡信息的做法表示不适。他们总结道：*"如果我不喜欢那一面，我就不必使用它（我确实不喜欢 :））。"*
- **针对额度问题的直接沟通**：一位成员提供了 DM 协助以解决问题，说道：*"随时给我发 DM - 请告诉我你在额度申请表中使用了哪个邮箱。"*
- **未解决的额度问题**：另一位成员报告称，由于之前账单信息不正确，他们没有收到额度，并寻求下一步该怎么做的指导。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[clavie_beyond_ragbasics](https://discord.com/channels/1238365980128706560/1242223963346698250/1251049080688218172)** (3 messages): 

- **搜索按钮故障**：用户发现 **搜索按钮没有任何反应**，或者尽管尝试使用，该按钮仍处于不可用状态。
- **确认搜索按钮反应迟缓**：另一位成员承认了该问题，解释说它非常慢，并承诺会进行性能分析 (profiling) 并加速该过程。
- **用户请求搜索进度的反馈**：另一位成员建议在搜索过程中提供某种 **正在进行的指示**，以改善用户体验。他们表示，如果知道进度正在进行，他们愿意等待。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1251198568581038210)** (1 messages): 

```html
<ul>
    <li><strong>寻求推理端点的最佳设置</strong>：一位成员就如何让 <em>inference endpoints</em> 达到最佳性能提出了疑问。他们询问是否有推荐的设置来优化性能。</li>
</ul>
```
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1250904427968204900)** (4 messages): 

- **RTX5000 不支持 Flash Attention**：在尝试使用 RTX5000 在 Jarvis 上微调 **openllama-3b** 时，用户遇到了问题。另一位成员告知他们 RTX5000 不支持 Flash Attention 并建议禁用它。

- **A5000 实例解决了训练问题**：切换到 A5000 实例后，用户成功开始训练他们的模型。用户报告说：*“模型现在正在训练中。”*

- **在使用 tiny-lama 和 Axolotl 时遇到错误**：用户在 Jarvis 和 Modal 上使用 Axolotl 仓库的 LoRA 配置示例微调 **tiny-lama** 时遇到了未指明的错误。他们表达了挫败感，表示：*“我在想为什么运行这些配置示例没那么简单直接。”*
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[simon_cli_llms](https://discord.com/channels/1238365980128706560/1242664474276659320/1251053148362244147)** (4 messages): 

- **LLM 的 Python API 受到关注**：一位成员对分享的演讲表示赞赏，并询问为什么 LLM 的 Python API 被认为是不完整的。他们提到很喜欢 *embed_multi_with_metadata*，并就潜在的陷阱或贡献寻求建议。

- **生产力见解带来启发**：另一位用户称赞了这次演讲，指出其中包含的生产力技巧和 GitHub issue 讨论令人惊喜。他们发现这些见解特别有启发性。

- **Ollama 插件未显示**：一位成员遇到了 Ollama 插件在添加到 plugins JSON 后仍未出现在 LLM 模型中的问题。他们请求调试建议，暗示这可能是一个常见问题。

- **视频播放问题**：一位用户报告了演讲录制页面的问题，收到了 *'no video with supported format found'* 错误消息，表明存在技术故障。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[credits-questions](https://discord.com/channels/1238365980128706560/1243721538432270388/1251035080768618547)** (12 messages🔥): 

- **额度兑换指导，致谢**：一位成员对课程表示感谢，并询问来自 Hugging Face 和 Langsmith 等多个平台的额度待处理状态。得到的指导是检查 "Sponsored Compute" 下的特定频道以寻找正确的联系人。
- **频道历史记录提示**：Hamel 建议成员在发布消息前先阅读相关频道的历史记录，以了解谁是联系人。这是为了确保成员能够找到合适的信息，避免重复查询。
- **邮箱地址请求**：Dan Becker 请求该成员提供邮箱地址，以核实他们属于哪一批次，并表示如果他们的数据没有与平台共享，进一步的协助可能会受到限制。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/1251089214188027985)** (1 messages): 

```html
- **Help with credits requested**: A member asked for assistance with credit issues after filling out a form, providing their ID as *akshay-thapliyal-153fbc*. They tagged another member specifically for help.
```
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1251158496959791104)** (1 messages): 

- **记录密码找回问题**：“感谢精彩的演讲！已经给你的同事发了邮件，因为无法使用注册时的邮箱登录和找回密码，”一位成员分享道，表达了他们在访问账户时遇到的问题。他们确认已将问题发送给 Will Van Eaton (**will@predibase.com**)。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[career-questions-and-stories](https://discord.com/channels/1238365980128706560/1245816565073576037/1251230888893874206)** (3 messages): 

- **Cursor 在 Linux 上依然棘手**：一位成员提出了关于在 Linux 上获取 Cursor “纯”二进制文件的担忧，指出他们只得到了 AppImage。他们表示很沮丧，因为这与运行 `code file.py` 不同。
- **对 Cursor 初始阶段 Bug 较多的质疑**：一位成员质疑半年前 Cursor 的稳定性问题是否已得到解决，指出存在持久的 Bug 以及与扩展的不兼容。另一位成员分享说，他们在过去四个月里的体验是积极且稳定的，表明有所改进。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/1251008172844646420)** (2 messages): 

- **成员等待 OpenAI 额度**：多位成员报告称，他们的账户尚未收到 OpenAI 额度。一位提到了 "org-45RghBshWBY9nqyEWwxanvTh"，而另一位提到了 "org-AGO5uO7zhYgyEcFiQX29eBsV"，并寻求协助解决此问题。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[pawel-function-calling](https://discord.com/channels/1238365980128706560/1250550872312643594/1250894979740467342)** (5 条消息): 

- **讨论了 Function Calling 的集成**：提到了在软件栈中集成 **function calling** 的话题，特别是在 **LangGraph** 的语境下。分享了一个指向 [LangGraph 关于重试的文档](https://langchain-ai.github.io/langgraph/how-tos/extraction/retries/)的链接，强调了 function calling 面临的挑战和解决方案，包括*更好的 prompting、受限解码（constrained decoding）以及通过重新提示（re-prompting）进行验证*。

- **分享 GIF 以示强调**：发布了一个来自 Tenor 的关于《曼达洛人》（The Mandalorian）的 GIF，可能是为了强调之前讨论点的关键性或正确性。这个幽默的 GIF 突显了聊天过程中的对话性和协作性。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/book-of-boba-fett-the-mandalorian-this-is-the-way-the-book-of-boba-fett-mandalorian-gif-24825263">Book Of Boba Fett The Mandalorian GIF - Book Of Boba Fett The Mandalorian This Is The Way - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://langchain-ai.github.io/langgraph/how-tos/extraction/retries/?">Extraction with Re-prompting - LangGraph</a>：未找到描述
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1250890389322334379)** (6 条消息): 

- **Lamini Memory Tuning 声称取得突破**：一篇关于 [Lamini Memory Tuning](https://www.lamini.ai/blog/lamini-memory-tuning) 的博客文章强调了一种将事实嵌入 LLM 的新方法，将事实准确性提高到 95%，并将幻觉（hallucinations）从 50% 降低到 5%。该方法涉及在 Llama 3 或 Mistral 3 等开源 LLM 之上微调专家适配器（expert adapters）。

- **对为 XTTS 训练 HiFi-GAN 感兴趣**：一位成员询问是否有人成功训练了 XTTS 的 HiFi-GAN 部分，并寻求社区的见解和经验。

- **寻求基于 LLM 进行章节教学的帮助**：一位成员请求关于如何让 LLM 教授一个章节的指导，包括组织内容、关键点和示例。

- **讨论了改进的二进制神经网络训练**：一位成员分享了一篇[论文](https://arxiv.org/abs/1909.13863)，提出了一种改进的二进制神经网络（binary neural networks）训练算法，该算法通过反向传播（backpropagation）判别式地学习缩放因子。该方法显著优于目前的先进技术 XNOR-Net。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.lamini.ai/blog/lamini-memory-tuning">Introducing Lamini Memory Tuning: 95% LLM Accuracy, 10x Fewer Hallucinations | Lamini - Enterprise LLM Platform</a>：未找到描述</li><li><a href="https://arxiv.org/abs/1909.13863">XNOR-Net++: Improved Binary Neural Networks</a>：本文提出了一种改进的二进制神经网络训练算法，其中权重和激活值均为二进制数。当前技术中一个关键但相当被忽视的特性是...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1250895556696608829)** (37 条消息🔥): 

- **KANs 可能简化在非常规硬件上的实现**：成员们讨论了 KANs (Kriging Approximation Networks) 相比于 MLPs (Multilayer Perceptrons) 在模拟、光学和其他“奇特硬件”上实现的潜在优势，指出 KANs 仅需要求和与非线性，而不需要乘法器。
- **训练大语言模型先对问题进行编码**：一份分享的论文建议，在对文档进行持续预训练之前，让 LLMs 接触 QA 对可能有利于编码过程，因为相比于文档的复杂性，QA 对更加直接。[阅读更多](https://arxiv.org/abs/2402.12847v1)。
- **Transformers 与图神经网络结合进行推理**：一种提出的混合架构将 Transformers 的语言理解能力与基于 GNN 的神经算法推理器的鲁棒性相结合，旨在解决 Transformers 在算法推理任务中的脆弱性。更多细节请参见 [论文](https://arxiv.org/abs/2406.09308)。
- **VISION Transformers 中的 Pixels-as-tokens**：一篇论文提出了直接在单个像素上操作的新颖概念，使用带有随机初始化、可学习的每像素位置编码的 vanilla Transformers。讨论强调了这种非常规方法及其潜在影响。[链接](http://arxiv.org/abs/2406.09415)。
- **PowerInfer-2 提升智能手机上的 LLM 推理速度**：新引入的框架 PowerInfer-2 显著加速了智能手机上大语言模型的推理，利用异构资源进行细粒度的神经元集群计算。评估显示，与最先进的框架相比，速度提升高达 29.2 倍。[完整细节](https://huggingface.co/papers/2406.06282)。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2402.12847v1">Instruction-tuned Language Models are Better Knowledge Learners</a>: 为了让基于大语言模型 (LLM) 的助手有效适应不断变化的信息需求，必须能够通过对新数据的持续训练来更新其事实知识...</li><li><a href="http://arxiv.org/abs/2406.08862">Cognitively Inspired Energy-Based World Models</a>: 训练世界模型的主要方法之一是在输出空间中对序列的下一个元素进行自回归预测。在自然语言处理 (NLP) 中，这表现为...</li><li><a href="https://arxiv.org/abs/2406.09406">4M-21: An Any-to-Any Vision Model for Tens of Tasks and Modalities</a>: 目前的多模态和多任务基础模型（如 4M 或 UnifiedIO）展示了可观的前景，但在实践中，它们接受多样化输入和执行多样化任务的开箱即用能力有限...</li><li><a href="http://arxiv.org/abs/2406.09358">Understanding Hallucinations in Diffusion Models through Mode Interpolation</a>: 通俗地说，基于扩散过程的图像生成模型经常被认为表现出“幻觉”，即训练数据中从未出现过的样本。但这些...</li><li><a href="https://arxiv.org/abs/2406.09308">Transformers meet Neural Algorithmic Reasoners</a>: Transformers 以其简单而有效的架构彻底改变了机器学习。在来自互联网的海量文本数据集上预训练 Transformers 带来了无与伦比的泛化能力...</li><li><a href="http://arxiv.org/abs/2406.09415">An Image is Worth More Than 16x16 Patches: Exploring Transformers on Individual Pixels</a>: 这项工作没有引入新方法。相反，我们提出了一个有趣的发现，质疑了现代计算机视觉架构中归纳偏置——局部性（locality）的必要性。具体来说...</li><li><a href="https://arxiv.org/abs/2406.06973">RWKV-CLIP: A Robust Vision-Language Representation Learner</a>: 对比语言-图像预训练 (CLIP) 通过使用从网站获取的图像-文本对扩展数据集，显著提高了各种视觉-语言任务的性能。本文...</li><li><a href="https://huggingface.co/papers/2406.06282">Paper page - PowerInfer-2: Fast Large Language Model Inference on a Smartphone</a>: 未找到描述</li><li><a href="https://github.com/lamini-ai/Lamini-Memory-Tuning/blob/main/research-paper.pdf">Lamini-Memory-Tuning/research-paper.pdf at main · lamini-ai/Lamini-Memory-Tuning</a>: 消除 LLM 幻觉需要重新思考泛化 - lamini-ai/Lamini-Memory-Tuning
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1251074039510728735)** (1 messages): 

```html
<ul>
    <li><strong>命令行参数需要后处理</strong>：一位成员询问是否可以通过命令行参数指定结果路径。他们得到的建议是可能需要进行一些后处理。</li>
</ul>
```
  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1250921090327384215)** (4 messages): 

- **RWKV-CLIP 取得 State-of-the-Art 结果**：分享了一个指向 [BlinkDL 公告](https://x.com/BlinkDL_AI/status/1801607729678848071)的链接，庆祝 RWKV-CLIP 取得 State-of-the-Art 结果。该模型在图像和文本编码方面均使用 RWKV，更多细节可以在链接的 [GitHub 仓库](https://github.com/deepglint/RWKV-CLIP)和 [arXiv 论文](https://arxiv.org/abs/2406.06973)中找到。
- **关于 GNN-RAG 论文的咨询**：一位成员询问是否有人审阅了最近发布的 GNN-RAG 论文，以及该频道是否适合进行讨论。
- **需要对 Attention-based Pooling Neural Networks 进行澄清**：一位成员寻求对特定公式和 Attention-based Pooling Neural Networks 概念的澄清，并提到其背景是在作为 Encoder 使用的语言模型中。

**提及的链接**：<a href="https://x.com/BlinkDL_AI/status/1801607729678848071">BlinkDL (@BlinkDL_AI) 的推文</a>：RWKV-CLIP 取得 SotA 结果🚀，它在图像和文本 Encoder 中均使用了 #RWKV https://github.com/deepglint/RWKV-CLIP https://arxiv.org/abs/2406.06973

  

---



### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1250946172055715922)** (35 messages🔥): 

- **关于 Cogvlm2 托管的讨论**：成员们讨论了在 OpenRouter 上托管 **cogvlm2** 的潜力和成本效益，并指出其可用性尚不明确。

- **Google Gemini Pro 的审核选项**：一位用户能够通过 OpenRouter 传递参数来控制 **Google Gemini Pro** 中的审核选项，但报告收到了错误消息，并建议 **OpenRouter** 需要启用这些设置。他们强调了 Google 关于[账单和访问的说明](https://cloud.google.com/billing/docs/how-to/invoiced-billing)。

- **关于 AI Studio 折扣定价的咨询**：一位用户询问 AI Studio 中 **Gemini 1.5 Pro** 和 **1.5 Flash** 的折扣定价是否可以应用于 OpenRouter，并指出 AI Studio 的基于 Token 的系统比 **Vertex** 更方便。

- **对 NVIDIA 开源资产的兴奋**：成员们对 NVIDIA 开放模型、RM 和数据表示热烈欢迎，特别提到了 **Nemotron-4-340B-Instruct** 和 **Llama3-70B** 变体。一位成员指出，这些模型提供的 **PPO 技术** 是一个宝库。

- **关于 June-Chatbot 来源的讨论**：有推测认为 LMSYS 中的 "june-chatbot" 是否由 NVIDIA 训练，一些成员指出 **70B SteerLM** 模型是一种可能性。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/nvidia/Nemotron-4-340B-Instruct">nvidia/Nemotron-4-340B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/nvidia/Llama3-70B-PPO-Chat">nvidia/Llama3-70B-PPO-Chat · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/nvidia/Llama3-70B-SteerLM-Chat">nvidia/Llama3-70B-SteerLM-Chat · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1250887490915537047)** (25 条消息🔥): 

- **服务器过载导致延迟**：用户注意到明显的变慢和冻结问题，可能是由于服务器过载引起的，*“类似于 Huggingface 免费模型过载时的情况。”* 有人提到，即使看起来冻结了，等待一段时间最终也可能完成任务。
- **结合模型以支持 Vision**：一名成员询问是否可以运行带有 Vision 能力的 i 模型，另一名成员建议使用 **自托管的 Llama 3 Vision 配置**。关于修改本地文件以启用此功能的做法也存在困惑，一些尝试仅产生了部分解决方案。
- **使用 O1 进行鼠标自动化**：Me_dium 询问了关于自动化任务（如发送 Slack 消息和移动鼠标点击消息框）的 API/库功能。*“它在屏幕上查找文本，然后将鼠标发送到对应坐标，”* 一位用户澄清道，并指出自从引入 OS 模式以来就具备了这种能力。
- **连接 Apple Scripts 进行自动化**：GordanFreeman4871 分享了一个 [YouTube 视频](https://youtu.be/eRQ6ztNW0f0?si=OE1iXnOCy4FzpZEt)，展示了使用快捷指令（Shortcuts）和 Apple Scripts 实现的快速响应自动化。他们强调了 Apple Scripts 的简单性以及良好的 Prompting 对于有效自动化的重要性。
- **i 模型冻结问题**：用户报告 i 模型在代码执行中途频繁冻结，需要通过 Control-C 进行手动中断。

**提到的链接**：<a href="https://youtu.be/eRQ6ztNW0f0?si=OE1iXnOCy4FzpZEt">2024年6月14日</a>：未找到描述

  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1250975257435701378)** (4 条消息): 

- **成员询问之前的问题**：针对一个未解决的问题，一名成员询问：*“你解决这个问题了吗？”* 回复是：*“还没再试过，抱歉！”* 之后没有进一步跟进。
- **分享有趣的硬件开发链接**：一名成员分享了一个来自 Seeed Studio 正在开发中的设备的[有趣链接](https://www.seeedstudio.com/watcher)。他们评论道：*“非常有趣的开发中现成设备。”*

**提到的链接**：<a href="https://www.seeedstudio.com/watcher">Sensecap Watcher - 用于空间管理的物理 AI Agent</a>：未找到描述

  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1250915829776646196)** (19 条消息🔥): 

- **Cohere 的聊天界面受到称赞**：一位用户强调了 [Cohere Playground Chat](https://dashboard.cohere.com/playground/chat) 上引用（citation）功能卓越的 UX，称赞点击任何内联链接都会显示一个带有引用文本和源链接的模态框，称其为“下一代水平，优于所有其他产品”。
- **Cohere 的开源聊天界面**：针对另一位用户的查询，一位用户推荐了 [GitHub 上的 cohere-toolkit](https://github.com/cohere-ai/cohere-toolkit)，用于实现支持引用的聊天界面。
- **利用 Cohere 进行文本补全**：一位用户获知 Cohere 可以通过 Chat 端点执行文本补全，并被引导至[有效的 Prompt 编写技巧](https://docs.cohere.com/docs/crafting-effective-prompts)以获得更好的结果。
- **Discord Bot 开发资源**：另一位寻求使用 Aya 模型创建 Discord Bot 的用户收到了多个资源，包括 [discord.py 文档](https://discordpy.readthedocs.io/en/stable/) 和 [Discord Interactions JS 仓库](https://github.com/discord/discord-interactions-js)，以帮助其项目开发。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/cohere-ai/cohere-toolkit">GitHub - cohere-ai/cohere-toolkit: Cohere Toolkit 是预构建组件的集合，使用户能够快速构建和部署 RAG 应用。</a>：Cohere Toolkit 是预构建组件的集合，使用户能够快速构建和部署 RAG 应用。 - cohere-ai/cohere-toolkit</li><li><a href="https://discordpy.readthedocs.io/en/stable/">欢迎来到 discord.py</a>：未找到描述</li><li><a href="https://github.com/discord/discord-interactions-js">GitHub - discord/discord-interactions-js: 用于 Discord Interactions 的 JS/Node 辅助工具</a>：Discord Interactions 的 JS/Node 辅助工具。通过在 GitHub 上创建账号为 discord/discord-interactions-js 做出贡献。</li><li><a href="https://docs.cohere.com/docs/crafting-effective-prompts">编写有效的 Prompt</a>：未找到描述</li><li><a href="https://dashboard.cohere.com/playground/chat">登录 | Cohere</a>：Cohere 通过一个易于使用的 API 提供对先进大语言模型（LLM）和 NLP 工具的访问。免费开始使用。
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1250956365480857721)** (4 条消息): 

- **对社区项目的兴奋**：一位成员对社区将要构建的内容表示兴奋，提到“大量的用例和示例”即将推出。这引起了其他成员的期待，其中一人表示：“期待用例和示例！”。
- **对热情的可爱回应**：另一位成员回复了一个俏皮的评论，“太可爱了 😸”，表明对正在进行的讨论持积极态度。
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1251083414862434306)** (19 条消息🔥): 

```html
- **RAG 示例输出问题**：一位用户在处理检索增强生成 (RAG) 链时遇到困难，无法正确显示预期结果。他们提供了代码并提到输出应该是 8，但得到了不同的结果，寻求帮助以修改代码，从而按特定问题过滤行。

- **在 LangChain 中创建 JSON 的指导**：一位用户寻求在 LangChain 的链中创建 JSON 对象并确保其为有效 JSON 的帮助。另一位用户通过提供 JavaScript 和 Python 示例来创建输出 JSON 对象的自定义聊天模型进行了回应。

- **LangChain 与 pgvector 集成的问题**：一位用户在参考 [LangChain-pgvector 集成文档](https://python.langchain.com/v0.2/docs/integrations/vectorstores/pgvector/) 时遇到问题，安装 `langchain_postgres` 后无法识别导入。另一位成员建议检查他们是否在 IDE 中使用了正确的 Python 环境。
```
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://python.langchain.com/v0.2/docs/integrations/vectorstores/pgvector/">PGVector | 🦜️🔗 LangChain</a>：使用 Postgres 作为后端并利用 pgvector 扩展的 LangChain 向量存储抽象实现。</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/custom_chat/#richer-outputs>).">如何创建自定义聊天模型类 | 🦜️🔗 Langchain</a>：本指南假设你已熟悉以下概念：
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1251016934414090382)** (3 条消息): 

- **演示使用 Llama 3 进行 RAG 的混合搜索 (Hybrid Search)**：分享了一个关于 *使用 Llama 3 进行 RAG 的混合搜索* 的 YouTube 视频，旨在展示如何为检索增强生成 (RAG) 实现混合搜索。视频包含了指向 [GitHub notebook](https://github.com/githubpradeep/notebooks/blob/main/hybrid%20vector%20search.ipynb) 和更多资源的链接。

- **LangServe 与 NLUX 的简易设置结合**：一位成员强调了使用 NLUX 为 LangServe 端点提供的**带有简易设置的优秀 UI**。[文档](https://docs.nlkit.com/nlux/learn/get-started/nlux-langchain) 指导用户将 NLUX 对话功能集成到 React JS 应用中，并引用了 LangChain 和 LangServe 库。

- **Gumloop 简化 AI-native 自动化**：介绍了 Gumloop，这是一个用于构建 AI-native 工作流自动化的无代码平台。[YouTube 视频](https://youtu.be/g53BIZX9Hag) 概述了其直观的 UI 以及使用 AutoGPT 自动化复杂任务的能力。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=xIDMPUYpd_0">使用 Llama 3 进行 RAG 的混合搜索</a>：我们将了解如何为 RAG 进行混合搜索 https://github.com/githubpradeep/notebooks/blob/main/hybrid%20vector%20search.ipynb https://superlinked....</li><li><a href="https://youtu.be/g53BIZX9Hag">Gumloop - 通过直观的 UI 构建 AI-native 工作流自动化</a>：Gumloop 是一个用于构建强大 AI 自动化的无代码平台。Gumloop 成立于 2023 年，旨在让任何人都能利用 AI 自动化复杂的工作，而无需...</li><li><a href="https://docs.nlkit.com/nlux/learn/get-started/nlux-langchain">开始使用 NLUX 和 LangChain LangServe | NLUX</a>：LangChain 是构建由 LLM 驱动的服务和后端的流行框架。
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1251245967382941707)** (4 messages): 

- **NVIDIA 发布 Nemotron-4 340B 模型**：一名成员分享了 [Nemotron-4 340B 模型系列](https://research.nvidia.com/publication/2024-06_nemotron-4-340b) 的发布，其中包括 Nemotron-4-340B-Base、Instruct 和 Reward。这些模型在 NVIDIA Open Model License 下开放访问，并可在 FP8 精度下运行在配备 8 个 GPU 的单台 DGX H100 上。
- **在 2 台 TinyBox 上适配 Nemotron-4 340B**：一名成员询问是否有办法让新的 Nemotron-4 340B 模型适配仅 2 台 TinyBox。他们认为这将是一个极具吸引力的用例。
- **3-Bit 量化建议**：另一名成员建议使用 **3-bit 量化** 将模型适配到一台 TinyBox 上，并强调了这是一种替代的紧凑部署策略。

**Link mentioned**: <a href="https://research.nvidia.com/publication/2024-06_nemotron-4-340b">Nemotron-4 340B | Research</a>: 未找到描述

  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1250928213606268929)** (15 messages🔥): 

- **在 tinygrad 中处理计算结果时的困扰**：一位用户分享了他们在 tinygrad 中设置缓冲区和运行计算图的进展。尽管完成了设置，他们仍然问道：*“我该如何实际获取结果或强制执行（realize）这个计算图？”* 另一位成员建议调用 `.exec` 并参考 abstractions2.py。

- **寻找 abstractions2.py**：在寻找 abstractions2.py 文件时，一位用户提供了该文件的直接 [GitHub 链接](https://github.com/tinygrad/tinygrad/blob/master/docs/abstractions2.py)。这帮助另一位用户确认道：*“谢谢！这正是我需要的！”*。

- **张量排序讨论**：一位用户询问在 tinygrad 中沿轴排序张量的方法。经过 George Hotz 的讨论和建议后，他们得出结论：使用 `argmax` 对于他们的目的（特别是实现 k-nearest neighbors）可能更有效。

- **PyTorch grid_sample 的替代方案**：一位用户询问 tinygrad 中是否有 PyTorch `grid_sample` 的替代方案，并提到 onnx_ops 中的 `AffineGrid`，但不确定其是否等效。他们分享了 [PyTorch 文档链接](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html) 作为参考。

- **M2 上的混合精度问题**：一位用户详细说明了在为模型实现混合精度时遇到的问题，并解释了他们使用 lambda 函数的方法。他们在 M2 芯片上遇到了特定于 Metal 库的 `CompileError`，怀疑是他们的混合精度方法有问题。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html">torch.nn.functional.grid_sample &mdash; PyTorch 2.3 documentation</a>: 未找到描述</li><li><a href="https://github.com/tinygrad/tinygrad/blob/master/docs/abstractions2.py">tinygrad/docs/abstractions2.py at master · tinygrad/tinygrad</a>: 你喜欢 pytorch？你喜欢 micrograd？你一定会爱上 tinygrad！❤️ - tinygrad/tinygrad
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1251013568363630613)** (15 messages🔥): 

- **NVIDIA 发布 340B 模型**：[Nemotron-4-340B-Base](https://huggingface.co/nvidia/Nemotron-4-340B-Base) 是一个拥有 **3400 亿参数**的大语言模型（LLM），支持 4,096 token 的上下文长度，用于合成数据生成。它在涵盖多种语言和编程语言的 9 万亿 token 数据上进行了广泛训练。

- **带有宽松许可证的合成数据模型**：Nemotron-4-340B-Base 的发布包含了一个*合成数据宽松许可证*。有人注意到该许可证是以 PDF 形式托管在 NVIDIA 网站上的，并对此表示关注。

- **实验性 Claude Steering API 访问权限**：[Claude](https://x.com/alexalbert__/status/1801668464920379648) 的实验性 Steering API 允许用户引导 Claude 内部功能的一个子集。目前已开放限量访问注册，但明确指出这仅用于研究预览，而非生产环境。

- **Sakana AI 获得 10 亿美元估值**：正在开发 Transformer 模型替代方案的日本初创公司 Sakana AI 已从 NEA、Lux 和 Khosla 筹集资金，获得了 10 亿美元的估值。更多详情请参阅[此处](https://www.theinformation.com/articles/openais-japanese-rival-gets-1-billion-valuation-from-silicon-valley-investors)。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/steph_palazzolo/status/1801690079922163954?s=46">来自 Stephanie Palazzolo (@steph_palazzolo) 的推文</a>：最新消息（与 @nmasc_ @KateClarkTweets 合作）：Sakana AI，一家开发 Transformer 模型替代方案的日本初创公司，已从 NEA、Lux 和 Khosla 融资，估值达 10 亿美元。更多信息：https://www.theinform...</li><li><a href="https://huggingface.co/nvidia/Nemotron-4-340B-Base">nvidia/Nemotron-4-340B-Base · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/alexalbert__/status/1801668464920379648">来自 Alex Albert (@alexalbert__) 的推文</a>：喜欢 Golden Gate Claude 吗？🌉 我们正开放实验性 Steering API 的限量访问——允许你引导 Claude 内部功能的一个子集。在此注册：https://forms.gle/T8fDp...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1251252934457425920)** (2 messages): 

```html
- **Getting compliments on the merch**: *"natolambert: Getting compliments on the merch"*. A member expressed satisfaction with receiving positive feedback on their merchandise.
- **We are in**: *"natolambert: We are in"*. A succinct declaration hints at an achievement or successful entry into an anticipated situation or event.
```
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1251215649095815238)** (2 messages): 

```html
- **Alex shitposts on company blog**: A message humorously referenced "Alex shitposting on the company blog," implying informal or irreverent comments. No further context was provided.
- **Scale claims meritocracy in all hiring**: A link to a [blog post on Scale](https://scale.com/blog/meritocracy-at-scale) highlighted the company’s claim that their success is rooted in strict meritocratic hiring practices. The post emphasizes that the company’s founder is personally involved in hiring decisions to maintain high standards.
```

**提及的链接**：<a href="https://scale.com/blog/meritocracy-at-scale">Scale 是一家精英管理公司，我们必须始终保持这一特质。</a>：MEI：功绩（merit）、卓越（excellence）和智慧（intelligence）

  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1250987214897283082)** (4 条消息): 

- **Apple 的 AI 集群能力**：一位成员分享了来自 [@mo_baioumy](https://x.com/mo_baioumy/status/1801322369434173860?s=46&t=XV1VJkM4nCYVU6fROoKkfw) 的推文，强调 Apple 用户现在可以使用 Apple 设备运行个人 AI 集群。他们幽默地推测，这可能就是 Apple 运行其私有云的方式。
  
- **独立黑客实现 100 万美元 ARR**：分享了一个来自 [@theAIsailor](https://x.com/theaisailor/status/1801356656149737606?s=46&t=90xQ8sGy63D2) 推文的励志故事，详细介绍了 Lyzr 在转向全栈 Agent 框架后，如何在 40 天内实现了 100 万美元的 ARR。他们开发并发布了多个自主 Agent，包括 Jazon 和 Skott，目前正专注于实现 Organizational General Intelligence（组织通用智能）。

- **Nvidia 发布新 AI 模型**：一位成员宣布了 Nvidia 的新模型 [Nemotron-4-340B-Instruct](https://huggingface.co/nvidia/Nemotron-4-340B-Instruct)，这是一个拥有 3400 亿参数的 LLM，针对基于英语的单轮和多轮对话场景进行了优化。该模型支持 4,096 token 的上下文长度，是合成数据生成流水线的一部分，旨在帮助其他开发者。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/nvidia/Nemotron-4-340B-Instruct">nvidia/Nemotron-4-340B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/theaisailor/status/1801356656149737606?s=46&t=90xQ8sGy63D2OtiaoGJuww">Siva Surendira (@theAIsailor) 的推文</a>：我们今天突破了 100 万美元 ARR（合同额）。这是 Lyzr 从 0 到 1 的故事👇 前 400 天 - 10 万美元；最后 40 天 - 100 万美元；2024 年预测 - 600 万美元。在 2023 年 9 月，我们从 AI 数据分析器转型为 (...</li><li><a href="https://x.com/theaisailor/status/1801356656149737606?s=46&t=90xQ8sGy63D2">Siva Surendira (@theAIsailor) 的推文</a>：我们今天突破了 100 万美元 ARR（合同额）。这是 Lyzr 从 0 到 1 的故事👇 前 400 天 - 10 万美元；最后 40 天 - 100 万美元；2024 年预测 - 600 万美元。在 2023 年 9 月，我们从 AI 数据分析器转型为 (...</li><li><a href="https://x.com/mo_baioumy/status/1801322369434173860?s=46&t=XV1VJkM4nCYVU6fROoKkfw">Mohamed Baioumy (@mo_baioumy) 的推文</a>：本周又一项 Apple 发布：你现在可以使用 Apple 设备运行你的个人 AI 集群 @exolabs_ h/t @awnihannun
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1251265169133146202)** (8 条消息🔥): 

- **BigScience 的 AI 实践**：成员透露，作为 BigScience 协作项目开发的一部分，**DiLoco 和 DiPaco** 现在已达到 SOTA 水平。随后有公告称 Prime Intellect 计划很快将它们开源。
  
- **Prime Intellect 使 AI 民主化**：Prime Intellect 旨在通过其平台使 AI 开发民主化，允许用户寻找全球算力资源并通过分布式训练来训练模型。他们渴望实现开放 AI 创新的集体所有权，涵盖从语言模型到科学突破的各个领域。[更多信息可以在其网站上找到](https://www.primeintellect.ai/)。

- **Bittensor 上的 Horde**：一位成员指出，被称为 "the horde" 的技术正被用于 Bittensor。

- **DeepMind 未复现结果**：有一条评论澄清说 **DeepMind** 并没有复现与 DiLoco 相关的结果。

**提到的链接**：<a href="https://www.primeintellect.ai/">Prime Intellect - 商品化算力与智能</a>：Prime Intellect 大规模地使 AI 开发民主化。我们的平台可以轻松找到全球算力资源，并通过跨集群的分布式训练来训练 SOTA 模型。集体...

  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1250903997103997069)** (8 messages🔥): 

- **Nemotron 量化后的准确度问题**：一位用户询问 **Nemotron-4-340B-Instruct** 在量化后是否能保持相同的准确度。目前尚未分享关于量化后准确度影响的进一步细节。

- **Nemotron-4-340B-Instruct 资源共享**：分享了 [Nemotron-4-340B-Instruct 模型](https://huggingface.co/nvidia/Nemotron-4-340B-Instruct) 的链接，强调了其**多语言能力**以及支持 4,096 tokens 上下文长度的能力，旨在用于合成数据生成（synthetic data generation）。

- **微调 Nemotron 的资源需求查询**：一位用户询问如何以最少的资源微调 **Nemotron**，怀疑可能需要 40 张 A100/H100 (80GB) GPU。另一位成员提到推理需要 2 个节点，这意味着微调可能至少需要两倍的节点数量。

**提到的链接**：<a href="https://huggingface.co/nvidia/Nemotron-4-340B-Instruct">nvidia/Nemotron-4-340B-Instruct · Hugging Face</a>：未找到描述

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1250994249491546223)** (1 messages): 

- **提议为 DPO 提供灵活的 chat template 支持**：在数据集预处理中允许灵活支持 `chat_templates` 的功能已添加，并且在 SFT 中表现良好。建议将此功能扩展到 DPO，通过使用来自 `conversation` 字段的对话历史，而 `chosen` 和 `rejected` 仍来自单独的字段，且可能是对话消息或原始文本。
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1251233680400056370)** (2 messages): 

- **Nvidia Toolkit 安装成功**：一位用户分享了他们在第二次尝试中成功安装 Nvidia toolkit，并使用命令 `accelerate launch -m axolotl.cli.train examples/openllama-3b/lora.yml` 运行 LoRA 示例的经验。他们对过程中获得的帮助表示感谢。
- **关于 Slurm 集群的咨询**：另一位用户询问是否有人有在 **Slurm 集群**的多节点中运行 Axolotl 的经验。消息记录中没有进一步的讨论或回复。
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1250887823226310686)** (5 messages): 

- **用户讨论 `Dream Machine` 让 memes 焕发生机**：分享了一个关于 **LumaLabsAI** 的 **Dream Machine** 的推文链接。推文指出该工具在为 memes 制作动画方面非常有效，并附带了一个包含更多细节的线程（[来源](https://fxtwitter.com/blizaine/status/1801126160904098247)）。
- **关于机器学习所需 DSA 的查询**：一位用户问道：*“有人知道在学习机器学习之前我们需要学习哪些 DSA（数据结构与算法）吗”*。消息历史中没有提供进一步的细节或回复。
- **本地使用 Mel Roformer 进行人声提取**：一位用户询问如何在本地或 Google Colab 中使用 **Mel Roformer 模型**提取人声。他们提到已经安装了 **weights 和 bs roformer**。
- **OpenAI 任命退役美国陆军将军**：分享了关于 OpenAI 任命一名退役美国陆军将军的公告链接（[来源](https://openai.com/index/openai-appoints-retired-us-army-general/)）。消息中未提供额外的背景或讨论。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/blizaine/status/1801126160904098247">来自 Blaine Brown  (@blizaine) 的推文</a>：来自 @LumaLabsAI 的 Dream Machine 真的让 memes 焕发了生机！一个线程 🧵</li><li><a href="https://fxtwitter.com/blizaine/status/1801126279917547726">来自 Blaine Brown  (@blizaine) 的推文</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1250990273769635931)** (4 messages): 

- **Yandex 的 YaFSDP 革新了 LLM 训练**：Yandex 推出了 [YaFSDP](https://www.marktechpost.com/2024/06/14/yandex-introduces-yafsdp-an-open-source-ai-tool-that-promises-to-revolutionize-llm-training-by-cutting-gpu-usage-by-20/)，这是一个开源工具，可减少 20% 的 GPU 资源消耗。这转化为了巨大的成本节约，特别是在具有 700 亿参数的大型模型中，每月可能减少 50 万至 150 万美元的成本。
- **带有 pgvectorscale 的 PostgreSQL 击败了 Pinecone**：PostgreSQL 的一个新开源扩展 [pgvectorscale](https://www.reddit.com/r/machinelearningnews/comments/1de08np/a_new_era_ai_databases_postgresql_with/) 在性能上超越了 Pinecone，同时降低了 75% 的成本。这一进展可能会通过以极低的成本提供卓越的性能来重塑 AI 数据库领域。
- **DreamSync 在无需人工评分的情况下对齐 text-to-image**：[DreamSync](https://www.reddit.com/r/StableDiffusion/comments/1881v4u/dreamsync_aligning_texttoimage_generation_with/) 是一种对齐 text-to-image 生成模型的新方法，它利用图像理解反馈而无需人工评分。该方法旨在显著提高 text-to-image 模型的准确性和效率。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/papers/2406.04127">论文页面 - Are We Done with MMLU?</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/machinelearningnews/comments/1dfwazp/yandex_introduces_yafsdp_an_opensource_ai_tool/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1881v4u/dreamsync_aligning_texttoimage_generation_with/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/machinelearningnews/comments/1de08np/a_new_era_ai_databases_postgresql_with/">Reddit - Dive into anything</a>: 未找到描述
</li>
</ul>

</div>
  



### **Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1250899997029826722)** (2 messages): 

- **Datasette 登上 Hacker News 首页**：一位成员庆祝某项工作登上了 Hacker News 的首页。另一位成员对该工作表示赞赏，特别提到了在处理替代方案时所采用的平衡方法。一位成员幽默地评论道：*"ChatGPT 实际上为数据工程师发布了一项永久性的充分就业法案。"*
  


### **DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1251213554078711941)** (1 messages): 

- **在 7-10b 参数范围间选择模型**：一位成员强调，*在 7-10b 范围内的模型选择取决于具体的用例*。对于需要正确德语语法的任务，尽管 Benchmark 指标可能另有显示，但 **discolm 或 occiglot 模型** 表现更好。

- **通过进一步训练管理权衡**：该成员建议，*非英语语言质量与 reasoning/instruction following 之间的权衡* 可以通过进一步训练或 merging 来管理。然而，对于参数更大的模型（50-72b 范围），**这种权衡问题会消失**，尽管推理速度可能会变慢。
  
- **资源考量**：由于大模型的推理速度较慢，他们建议*保持使用适配 VRAM 的 q4/q6 量化以保证效率*。分享了一个链接供参考：[Hugging Face 上的 Spaetzle 集合](https://huggingface.co/collections/cstr/spaetzle-661e758857b1fa96731c43bc)。
  



> 完整的频道细分内容已为电子邮件格式进行截断。
> 
> 如果您想查看完整细分，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})!
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢支持！
