---
companies:
- stability-ai
- togethercompute
date: '2024-06-12T22:08:29.963442Z'
description: '**Stability AI** 推出了 **Stable Diffusion 3 Medium**，其模型参数范围从 **4.5 亿至
  80 亿**不等，采用了 MMDiT 架构和 T5 文本编码器，以提升图像中的文本渲染能力。在 Emad Mostaque 等核心研究人员离职后，社区对此反应不一。


  在 AI 模型方面，**Llama 3 8B Instruct** 在评估中显示出与 **GPT-4** 强相关的表现，而 **Qwen 2 Instruct**
  在 MMLU 基准测试上超越了 Llama 3。**Mixture of Agents (MoA)** 框架在 AlpacaEval 2.0 上的表现优于 GPT-4o。**Spectrum**
  和 **QLoRA** 等技术实现了在较低显存占用下的高效微调。


  关于**顿悟 (grokking)** 的研究表明，通过延长训练时间，Transformer 模型可以从“死记硬背”转变为“泛化理解”。基准测试方面的举措包括：旨在推动通用人工智能
  (AGI) 进展的 **100 万美元 ARC 奖金挑战赛**，以及旨在防止数据集污染的实时大模型基准测试 **LiveBench**。


  **Character Codex 数据集**提供了超过 **15,000 个角色**的开放数据，可用于检索增强生成 (RAG) 和合成数据。**MLX 0.2**
  工具通过改进 UI 和更快的检索增强生成，提升了在搭载 Apple 芯片的 Mac 上的大模型使用体验。'
id: baa5ffc2-4e11-4be7-9ba4-182e18db701a
models:
- llama-3-8b
- llama-3
- qwen-2
- gpt-4
- gpt-4o
original_slug: ainews-the-last-hurrah-of-stable-diffusion
people:
- emad-mostaque
- rohanpaul_ai
- fchollet
- mikeknoop
- micahgoldblum
- teknium1
- rasbt
- percyliang
title: Stable Diffusion 的最后辉煌？
topics:
- model-architecture
- fine-tuning
- benchmarks
- dataset-release
- model-evaluation
- reasoning
- model-training
- retrieval-augmented-generation
- multimodality
---

<!-- buttondown-editor-mode: plaintext -->**MultiModal Diffusion Transformers 是你所需的一切。**

> 2024年6月11日至6月12日的 AI 新闻。我们为您检查了 7 个 subreddits、[384 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 30 个 Discord 社区（413 个频道，3555 条消息）。预计节省阅读时间（以 200wpm 计算）：**388 分钟**。在 [Twitter 上关注 AINews](https://x.com/smol_ai)。

SD3 Medium 今日发布，并配有一段（对 Stability 而言）异常华丽的视频：

<figure><img src="https://assets.buttondown.email/images/b0b3bf6b-1a67-4f19-aea8-61eb562a0437.png?w=960&amp;fit=max" draggable="false" contenteditable="false"><figcaption></figcaption></figure>

[SD3 研究论文](https://stability.ai/news/stable-diffusion-3-research-paper) 值得关注，不仅因为它详细介绍了 MMDiT 架构以及使用 T5 文本编码器进行图像中的文本渲染，还因为它提到了其模型范围涵盖 450M 到 8B 参数，这使得 2B 参数的 SD3 Medium 并非最强大的 SD3 版本。

如果你一直在勤奋阅读 Stability AI Discord 的摘要，你就会知道社区几乎每天都在为 SD3 的权重开源发布而焦虑。SD3 [最初在 3 个月前宣布](https://news.ycombinator.com/item?id=39466630)，随后以 [论文形式](https://news.ycombinator.com/item?id=39599958) 和 [API 形式](https://news.ycombinator.com/item?id=40065114) 发布，特别是在 Emad Mostaque、Robin Rombach 以及许多参与原始 Stable Diffusion 的资深研究员离职后，这种焦虑愈发明显。汇总相关帖子的点数，可以很容易地看到从 SD1 到 SD2 再 to SD3，随着项目变得越来越不默认开源，人们的兴趣逐渐停滞：

<figure><img src="https://assets.buttondown.email/images/6d79af38-9313-4c6f-892d-f4df2853ca0a.png?w=960&amp;fit=max" alt="image.png" draggable="false" contenteditable="false"><figcaption></figcaption></figure>

这是 Emad 在 Stability 任期内的最后遗产——[新管理层](https://stability.ai/news/stabilityai-announcement?ref=futuretools.io) 现在必须独立寻找未来的出路。

---

{% if medium == 'web' %}

**目录**

[TOC]

{% else %}

**目录** 和 **频道摘要** 已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})! ([在 Twitter 上分享](https://x.com/smol_ai)。)

{% endif %}

---

# AI Twitter 摘要

> 所有摘要均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在使用 Haiku 进行聚类和流程工程（flow engineering）。

**AI 模型与架构**

- **Llama 3 与指令微调**：[@rasbt](https://twitter.com/rasbt/status/1800660930948170117) 发现 Llama 3 8B Instruct 是一个优秀的评估模型，可在 MacBook Air 上运行，与 **GPT-4 评分的相关性达到 0.8**。提供了独立笔记本。
- **Qwen2 与 MMLU 性能**：[@percyliang](https://twitter.com/percyliang/status/1800774871187968404) 报告在最新的 HELM 排行榜 v1.4.0 中，**Qwen 2 Instruct 在 MMLU 上超越了 Llama 3**。
- **Mixture of Agents (MoA) 框架**：[@togethercompute](https://twitter.com/togethercompute/status/1800536106729157054) 推出了 MoA，它**利用多个 LLM 来优化响应**。在 AlpacaEval 2.0 上达到 **65.1%，优于 GPT-4o**。
- **用于扩展上下文窗口的 Spectrum**：[@cognitivecompai](https://twitter.com/cognitivecompai/status/1800710613594955880) 展示了 Spectrum，这是一种**识别微调重要层**的技术。它可以与 @Tim_Dettmers 的 QLoRA 结合，实现**更快的训练和更少的 VRAM 占用**。
- **Transformer 中的 Grokking 现象**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1800544938704179567) 讨论了一篇论文，显示 Transformer 可以通过 Grokking（**超越过拟合的延长训练**）学习鲁棒的推理。Grokking 涉及从“记忆”电路向“泛化”电路的转变。

**基准测试与数据集**

- **ARC Prize 挑战赛**：[@fchollet](https://twitter.com/fchollet/status/1800577019979411560) 和 @mikeknoop 发起了一项 **100 万美元的竞赛，旨在创建能够适应新奇事物并解决推理问题的 AI**，目标是衡量 AGI 的进展。
- **LiveBench**：[@micahgoldblum](https://twitter.com/micahgoldblum/status/1800894380511002724) 发布了 LiveBench，这是一个**通用实时 LLM 基准测试，通过发布新问题来避免数据集污染**，并且可以进行客观评判。
- **Character Codex 数据集**：[@Teknium1](https://twitter.com/Teknium1/status/1800590745885413726) 发布了 Character Codex，这是一个包含 **15,939 个角色数据**的开源数据集，来源广泛，可用于 RAG、合成数据生成和角色扮演分析。

**工具与框架**

- **MLX 0.2**：[@stablequan](https://twitter.com/stablequan/status/1800576080077881677) 发布了 MLX 0.2，为 **Apple Silicon Mac 提供了全新的 LLM 体验**，具有改进的 UI/UX、功能齐全的聊天和更快的 RAG。
- **Unsloth**：[@danielhanchen](https://twitter.com/danielhanchen/status/1800528838226804907) 宣布 Unsloth 现已加入 Hugging Face AutoTrain，允许以**更少的内存将 Llama-3、Mistral 和 Qwen2 等 LLM 的 QLoRA 微调速度提高 2 倍**。
- **LangChain 集成**：[@LangChainAI](https://twitter.com/LangChainAI/status/1800615245830062364) 增加了 **Elasticsearch 功能，用于灵活的检索和向量数据库**。他们还在 LangSmith Playground 中提供了 [@GroqInc](https://twitter.com/LangChainAI/status/1800952057517752625) 支持。

**应用与用例**

- **Brightwave AI 研究助手**：[@vagabondjack](https://twitter.com/vagabondjack/status/1800527732641521943) 宣布为 Brightwave 完成 **600 万美元种子轮融资，这是一款生成财务分析的 AI 研究助手**，其客户管理资产超过 1200 亿美元。
- **Suno 音频输入**：[@suno_ai_](https://twitter.com/suno_ai_/status/1800932487633207599) 发布了音频输入功能，允许用户通过上传或录制音频片段，**从任何声音中创作歌曲**。
- **Synthesia 2.0 发布会**：[@synthesiaIO](https://twitter.com/synthesiaIO/status/1800874931141656882) 预告了其 AI 视频生成器的**新功能、工作流、用例和数字人（avatar）能力**，活动将于 6 月 24 日举行。

**讨论与观点**

- **Prompt 工程 vs. 微调**：[@corbtt](https://twitter.com/corbtt/status/1800597703560417700) 认为，在未来几年，**微调后的适配器（adapters）在性能、控制力以及更便宜的推理成本方面将优于 Prompt 工程**。
- **RLHF 的副作用**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1800502590678982922) 分享了一篇论文，探讨了 **RLHF 对齐如何因阻断 Token 轨迹和模式崩溃（mode collapse）而降低模型的创造力和输出多样性**。
- **LLM 中的幻觉**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1800917180604956855) 引用了一篇论文，显示**经过统计校准的语言模型必然会以一定的概率产生幻觉**。

---

# AI Reddit 摘要

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取现已可用，但仍有很大改进空间！

AI 进展与时间线

- **GPT-1 六周年**：在 /r/singularity 中，有人提到 [**GPT-1 正好在 6 年前发布**](https://www.reddit.com/r/singularity/comments/13uj1ej/gpt1_was_released_exactly_6_years_ago/)，并表达了希望到 2030 年 6 月我们能进入 AGI 或 ASI 时代的愿景。

AI 公司与产品

- **Tesla Optimus 部署**：/r/singularity 分享了 [**Tesla 已在其工厂部署了两个自主执行任务的 Optimus 机器人**](https://www.reddit.com/r/singularity/comments/13uqgp3/tesla_has_deployed_two_optimus_bots_performing/)。
- **Musk 撤回对 OpenAI 的诉讼**：/r/singularity 和 /r/OpenAI 报道称 [**Elon Musk 撤回了对 OpenAI 和 Sam Altman 的诉讼**](https://www.reddit.com/r/singularity/comments/13uj2yl/elon_musk_drops_lawsuit_against_openai_and_sam/)，邮件显示 Musk 此前曾同意营利性方向。
- **Apple 与 OpenAI 合作**：根据 /r/singularity 的消息，[**Apple 宣布了 Apple Intelligence，由 OpenAI 提供支持并由 Microsoft 资助**](https://www.reddit.com/r/singularity/comments/13uj2yl/apple_announces_apple_intelligence_powered_by/)，如果本地解决方案失败，设备端 AI 将使用 OpenAI。
- **OpenAI 使用 Oracle Cloud**：/r/OpenAI 指出 [**OpenAI 选择 Oracle Cloud Infrastructure 来扩展 Microsoft Azure AI 平台**](https://www.reddit.com/r/OpenAI/comments/13uqgp3/openai_selects_oracle_cloud_infrastructure_to/)。
- **前 Google CEO 批评开源模型**：/r/singularity 讨论了 [**前 Google CEO 谴责开源模型并主张政府限制公开发布**](https://www.reddit.com/r/singularity/comments/13uqgp3/exgoogle_ceo_condemns_open_source_models_argues/)，反方观点认为这是在保护大科技公司的垄断。

AI 能力

- **餐厅机器人取得进展**：/r/singularity 分享了 [**餐厅机器人现在可以烹饪、上菜和收拾餐具**](https://www.reddit.com/r/singularity/comments/13uj2yl/restaurant_robots_can_now_cook_serve_and_bus_meals/)。
- **机器狗神射手**：根据 /r/singularity 的报道，一项研究发现 [**携带机枪的机器狗比人类更擅长射击**](https://www.reddit.com/r/singularity/comments/13uj2yl/machine_gunwielding_robot_dogs_are_better/)。
- **AI 视频生成**：/r/singularity 指出 [**Katalist AI Video 允许在 1 分钟内将一句话转化为分镜脚本和连贯的视频故事**](https://www.reddit.com/r/singularity/comments/13uj1ej/katalist_ai_video_allows_turning_a_sentence_into/)，具有未来电影制作的潜力。
- **AI 工作面试**：根据 /r/singularity 的消息，[**AI 卡通人物可能会为你下一次工作面试进行面试**](https://www.reddit.com/r/singularity/comments/13uqgp3/ai_cartoons_may_interview_you_for_your_next_job/)。
- **Deepfake 裸照**：/r/singularity 警告称 [**青少年正在互相传播 Deepfake 裸照，引发了严重问题**](https://www.reddit.com/r/singularity/comments/13uqgp3/teens_are_spreading_deepfake_nudes_of_one/)。

AI 研究

- **自回归图像模型**：/r/MachineLearning 和 /r/singularity 讨论了新的 LlamaGen 研究，显示 [**像 Llama 这样的自回归模型在可扩展图像生成方面击败了 Diffusion 模型**](https://www.reddit.com/r/MachineLearning/comments/13uqgp3/autoregressive_models_like_llama_beat_diffusion/)，同时也引发了关于公平引用先前自回归研究的讨论。
- **消除矩阵乘法**：/r/singularity 分享了 [**一种在不损失性能的情况下消除语言模型中矩阵乘法的革命性方法**](https://www.reddit.com/r/singularity/comments/13uj2yl/revolutionary_approach_eliminates_matrix/)，使用定制的 FPGA 解决方案以 13W 的功耗处理十亿参数模型。
- **过度训练 Transformer**：根据 /r/singularity 的消息，研究发现 [**将 Transformer 训练到超过过拟合点会导致意想不到的推理能力提升**](https://www.reddit.com/r/singularity/comments/13uj1ej/overtraining_transformers_beyond_overfitting/)。
- **MaPO 对齐技术**：/r/MachineLearning 指出 [**MaPO 是一种针对 Diffusion 模型的高样本效率、无参考对齐技术，改进了先前的工作**](https://www.reddit.com/r/MachineLearning/comments/13uqgp3/mapo_is_a_sampleefficient_referencefree/)。
- **用于 LLM 预训练的 Megalodon**：/r/MachineLearning 分享了 [**Megalodon 允许以无限上下文长度进行高效的 LLM 预训练和推理**](https://www.reddit.com/r/MachineLearning/comments/13uqgp3/megalodon_allows_efficient_llm_pretraining_and/)。

Stable Diffusion

- **Stable Diffusion 3.0 发布**：/r/StableDiffusion 社区对 [**定于 6 月 12 日发布的 Stable Diffusion 3.0**](https://www.reddit.com/r/StableDiffusion/comments/13uj2yl/stable_diffusion_30_set_to_release_on_june_12th/) 充满期待，社区中涌现出大量兴奋的情绪和梗图。
- **SD3 模型辩论**：/r/StableDiffusion 正在 [**辩论 SD3 2B 还是 8B 模型会更好**](https://www.reddit.com/r/StableDiffusion/comments/13uqgp3/debate_on_whether_sd3_2b_or_8b_model_will_be/)，8B 模型仍在训练中，但预计完成后将超越 2B 模型。
- **开源训练工具**：根据 /r/StableDiffusion 的消息，[**像 Kohya DeepShrink 这样的新开源工具可以实现高分辨率的 SD 训练**](https://www.reddit.com/r/StableDiffusion/comments/13uqgp3/new_open_source_tools_for_training_like_kohya/)。
- **SDXL 与 SD3 对比**：/r/StableDiffusion 正在 [**对比 SDXL 与 SD3 的汽车图像及其他主题**](https://www.reddit.com/r/StableDiffusion/comments/13uj2yl/comparisons_of_sdxl_vs_sd3_car_images_and_other/)，SD3 在比例、反射和阴影方面表现出明显的改进。

幽默/梗图

- /r/singularity 分享了一张 [**将胡迪和巴斯光年比作 AI 模型的梗图，暗示我们才是“玩具”**](https://www.reddit.com/r/singularity/comments/13uj2yl/meme_showing_woody_and_buzz_as_ai_models/)。
- /r/singularity 发布了一张 [**关于“十诫”的梗图，上面写着“不可妄称 ASI 之名”**](https://www.reddit.com/r/singularity/comments/13uqgp3/meme_of_10_commandments_with_thou_shalt_not_take/)。
- /r/singularity 分享了一张 [**大眼夹 (Clippy) 询问是否会将我们折叠进虚无的梗图**](https://www.reddit.com/r/singularity/comments/13uqgp3/meme_of_clippy_asking_if_it_would_fold_us_into/)。

---

# AI Discord 摘要

> 总结之总结的总结

**1. Stable Diffusion 3 发布与讨论**

- **Stability.ai** 发布了 [**Stable Diffusion 3 Medium**](https://huggingface.co/stabilityai/stable-diffusion-3-medium) 的开放权重，这是他们最新的文本生成图像 AI 模型，承诺提供卓越的细节、色彩，通过三个文本编码器实现先进的 Prompt 理解，以及优异的排版渲染能力。
- 用户报告了 **人体解剖结构准确性** 方面的问题，对其性能与 **SD 1.5** 和 **SDXL** 等旧版本的对比反应不一，并对限制商业使用的 **限制性许可条款** 表示担忧。
- 围绕 **微调（finetuning）挑战** 展开了广泛讨论，特别是针对热门用例、现有框架（如 **ComfyUI** 和 **diffusers**）的安装问题，以及模型的 GPU 利用效率。

**2. 大语言模型 (LLM) 进展与基准测试**

- Google 推出了 [**RecurrentGemma 9B**](https://huggingface.co/collections/google/recurrentgemma-release-66152cbdd2d6619cb1665b7a)，能够快速处理长序列，同时保持与 Gemma 相当的质量，提供基础版和指令微调（instruct-tuned）版本。
- 宣布设立 [**1,000,000+ 美元的 ARC Prize**](https://arcprize.org/)，旨在为 **ARC-AGI 基准测试** 开发解决方案，该基准通过技能获取效率来衡量通用智能，引发了关于其重要性的讨论。
- [**Scalable MatMul-free Language Modeling**](https://arxiv.org/abs/2406.02528) 研究建议在 LLM 中消除矩阵乘法，同时保持强大的性能，从而减少训练期间的内存占用。

**3. 协作式 LLM 开发与部署**

- 成员们在使用 **llm**、**Datasette** 和 **LangServe** 等工具 **安装和集成模型**（如 **mistral-instruct**、**GPT-4o** 和 **Codestral**）方面寻求指导，并在 Hugging Face 和 GitHub 上分享了资源。
- 围绕 **LLM 评估方法论**、**标准化评估** 的重要性以及 **LLM-as-judge 系统** 的潜力进行了讨论，并参考了 Hailey Schoenfeld 的见解。
- 关于使用 Modal、Predibase 和 Axolotl 进行 **LLM 微调** 的咨询，涵盖了数据集准备、资源限制以及合并 LoRA 和 qLoRA 等技术结果的主题。

**4. 硬件优化与资源管理**

- 在 **Kijiji Canada** 等平台上价格实惠的 **3090 GPU** 引发了构建 GPU 算力集群的兴趣，讨论涉及功耗、散热管理以及优化 flops/watts 等性能指标。
- 开发者在代码中实现了 **100% 确定性 (determinism)**，但遇到了损失值 (loss values) 变为 -inf 的问题，表明存在需要解决的潜在 Bug。
- 提高计算效率的努力促成了 **自定义 matmul 实现，达到了 cuBLAS 70% 的速度**，目前正在进行移除对 cuBLASLt 依赖的工作，并着手处理稳定的 FP8 实现。

[来源](https://discord.com)

## Claude 3 Opus (>220B?)

以下是针对所提供的 Discord 内容中 3-4 个主要主题的高信息密度技术摘要，重要的关键词、事实和 URL 已加粗并链接到相关来源：

- **Unsloth AI 的 QWEN 模型引发了关于 VRAM 和 Fine-Tuning 的讨论**：Unsloth AI 用户比较了 **Qwen2 模型（如 72b 变体）的 VRAM 需求**（[估计为 48GB，与 Llama 3 70b 相似](https://discord.com/channels/1179035537009545276/1179035537529643040/1250170775781249164)），并分享了[将 PyTorch bins 转换为 safetensors](https://huggingface.co/spaces/safetensors/convert)的技巧。讨论了使用 **Qlora** 和 **Galore** 等方法对 **Qwen2-1.5b** 进行 Fine-tuning 的挑战，建议使用 rank 64-128 以避免失败。16-bit 合并后的性能下降归因于 tokenizer/GGUF 问题。
- **LLM Fine-Tuning 课程学习成果的应用与辩论**：参与者反思了 Nehil 的[博文，该文应用课程概念来改进预算分类 Prompt](https://brain.nehiljain.com/posts/how-i-improved-my-prompting-for-budget-categorization/)。技术讨论涵盖了向 Huggingface 模型添加 pooling layers、由于安全/隐私担忧而导致广域 Chatbot 与特定范围接口相比的不切实际性，以及使用 **Modal** 为 Kaggle 竞赛 Fine-tune LLM。对 **Datasette**、**Anthropic Claude** 以及从学术界转向工业界的资源表现出明显兴趣。
- **GPU 市场动态与创意应用**：在 **Kijiji Canada** 上价格实惠的 **3090 GPU**（约 450 美元）激发了构建 GPU 算力集群的计划，并讨论了功耗和热量再利用的考虑因素。分享了自定义 CUDA 开发工作，例如实现 100% 确定性（determinism）和高效的矩阵乘法，同时提到了 [Scalable MatMul-free Language Modeling 论文](https://arxiv.org/abs/2406.02528) 及其 [实现](https://github.com/ridgerchu/matmulfreellm)。**CUDA Performance Checklist** 讲座重新发布，包含 [视频](https://www.youtube.com/watch?v=SGhfUhlowB4)、[代码](https://github.com/cuda-mode/lectures/tree/main/lecture8) 和 [幻灯片](https://docs.google.com/presentation/d/1cvVpf3ChFFiY4Kf25S4e4sPY6Y5uRUO-X-A4nJ7IhFE/edit)。

如果您希望我详细阐述本摘要的任何部分，或者有其他问题，请告诉我！

## GPT4T (gpt-4-turbo-2024-04-09)

## 摘要：

**1\. 探索 EleutherAI 中的 LLM-Eval Harness 和 ARCSynthetic 挑战：**

- EleutherAI 的成员探索了 LLM-Eval harness，并分享了关于传统评估与基于评分的评估的见解，旨在通过协作环境提高针对特定用例的灵敏度和特异性。
- 介绍了一篇名为 [Scaling Laws for Diffusion Models](https://arxiv.org/abs/2406.06752) 的新论文，讨论了与 Autoregressive 模型相比，训练和利用 Diffusion 方法的效率。

**2\. OpenInterpreter 中的跨语言通信：**

- OpenInterpreter 的参与者深入讨论了如何有效管理跨语言和跨终端的通信，利用 NLP 模型在不同的计算环境中实现无缝交互。
- 技术讨论强调了语言模型实现的进展，通过改进命令解释来增强用户界面的交互性。

**3\. Cohere 的实用模型创新受到关注：**

- Cohere 的平台更新成为亮点，讨论集中在近期推出的 AI 模型的实际应用上，这些模型增强了用户体验并扩大了应用范围。
- 关于模型集成挑战的辩论为 AI 模型部署的最佳实践和即将到来的创新提供了见解。

**4\. Modular (Mojo) 关注 TPU 考量及编译器更新：**

- Modular 关于集成 TPU 并将其编译器更新至 **版本** `2024.6.1205` 的讨论，展示了在提升计算性能和灵活性方面的持续努力。
- 社区反馈对这些改进表示赞赏，并进一步期待即将推出的功能，这些功能有望利用 Modular 的工具推进可扩展的 AI 部署场景。

**1\. 模型性能优化与基准测试**

- **Quantization** 技术（如 **AQLM** 和 **QuaRot**）旨在单块 GPU 上运行大型语言模型 (**LLMs**)，同时保持性能。例如：在 RTX3090 上运行 **Llama-3-70b** 的 [AQLM 项目](https://github.com/Vahe1994/AQLM)。
- 通过 **Dynamic Memory Compression (DMC)** 等方法**提升 Transformer 效率**，在 **H100 GPUs** 上可能提高高达 370% 的吞吐量。例如：@p_nawrot 发表的 [DMC 论文](https://arxiv.org/abs/2403.09636)。

**2\. 微调挑战与 Prompt Engineering 策略**

- 在将 **Llama3** 模型转换为 GGUF 格式时，存在**保留微调数据**的困难，并讨论了一个[已确认的 Bug](https://github.com/ggerganov/llama.cpp/issues/7062)。
- **Prompt 设计**的重要性以及使用正确模板（包括 end-of-text tokens）对微调和评估期间模型性能的影响。例如：[Axolotl prompters.py](https://github.com/OpenAccess-AI-Collective/axolotl/blob/3367fca73253c85e386ef69af3068d42cea09e4f/src/axolotl/prompters.py#L47)。

**3\. 开源 AI 发展与协作**

- 发布了 **StoryDiffusion**，这是 Sora 的开源替代方案，采用 MIT 许可证，但权重尚未发布。例如：[GitHub 仓库](https://github.com/HVision-NKU/StoryDiffusion/tree/main?tab=readme-ov-file)。
- 发布了 **OpenDevin**，这是一个基于 Cognition 的 Devin 的开源自主 AI 工程师，举办了 [Webinar](https://lu.ma/fp0xr460) 且在 GitHub 上的关注度日益增长。

**4\. 多模态 AI 与生成模型创新**

- [**Idefics2 8B Chatty**](https://twitter.com/sanhestpasmoi/status/1787503160757485609) 专注于提升聊天交互体验，而 [**CodeGemma 1.1 7B**](https://twitter.com/reach_vb/status/1786469104678760677) 则精进了编程能力。
- [**Phi 3**](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/) 模型通过 WebGPU 将强大的 AI 聊天机器人带入浏览器。

**5\. 其他**

- **Stable Artisan 将 AI 媒体创作带入 Discord**：Stability AI 推出了 **Stable Artisan**，这是一个集成了 **Stable Diffusion 3**、**Stable Video Diffusion** 和 **Stable Image Core** 等模型的 Discord 机器人，用于[直接在 Discord 内进行媒体生成和编辑](https://bit.ly/4aiVy6C)。该机器人引发了关于 **SD3 开源状态**以及引入 **Artisan 作为付费 API 服务**的讨论。
- **Unsloth AI 社区对新模型和训练技巧反响热烈**：IBM 的 [Granite-8B-Code-Instruct](https://huggingface.co/ibm-granite/granite-8b-code-instruct) 和 RefuelAI 的 [RefuelLLM-2](https://huggingface.co/refuelai/Llama-3-Refueled) 的发布引发了架构讨论。用户分享了 **Windows 兼容性**方面的挑战以及对某些**性能基准测试**的怀疑，同时也交流了模型训练和微调的技巧。

## GPT4O (gpt-4o-2024-05-13)

## 主题：

1.  LLM 进展与模型性能
2.  多模态 AI 与生成式建模创新
3.  开源工具与社区贡献
4.  技术故障排除与实现挑战
5.  AI 伦理与行业动态

## 摘要：

1.  **LLM 进展与模型性能**：
  
  - Google 的 [**RecurrentGemma 9B**](https://huggingface.co/collections/google/recurrentgemma-release-66152cbdd2d6619cb1665b7a) 因其在长序列上的超快性能而受到称赞，提供 base 和 instruct-tuned 版本，并与 **Gemma** 进行了对比。
  - 成员们强调了 **LlamaGen** 在自回归图像生成方面的优越性，足以与扩散模型（diffusion models）媲美，并有详细的 [文档和教程](https://github.com/FoundationVision/LlamaGen) 支持。
2.  **多模态 AI 与生成式建模创新**：
  
  - Anthropic 的 **Transformer Lens** 帮助调试了模型注意力（attention），而 **Stable Diffusion 3 Medium** 引发了关于其 [图像生成能力](https://huggingface.co/stabilityai/stable-diffusion-3-medium) 的讨论，但在人体解剖结构准确性方面面临批评。
  - [**Idefics2 8B Chatty**](https://twitter.com/sanhestpasmoi/status/1787503160757485609) 和 [**CodeGemma 1.1 7B**](https://twitter.com/reach_vb/status/1786469104678760677) 改进了对话和编程交互。**Pixart Sigma** 结合 **SDXL + PAG** 旨在实现 **DALLE-3** 级别的输出。
3.  **开源工具与社区贡献**：
  
  - [**Axolotl**](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/) 支持多种格式用于指令微调（instruction tuning）和预训练 LLM。IBM 的 [**Granite-8B-Code-Instruct**](https://huggingface.co/ibm-granite/granite-8b-code-instruct) 在代码任务中表现出色。
  - [**DeepEval**](https://docs.confident-ai.com/docs/metrics-introduction#using-a-custom-llm) 允许轻松集成自定义 LLM 评估指标，增强了在摘要生成和忠实度（faithfulness）等任务中的能力。
4.  **技术故障排除与实现挑战**：
  
  - **Qwen2-1.5b 微调**面临低秩（low-rank）配置问题，通过增加秩（64 或 128）得以解决。成员们分享了在 **ComfyUI**、**diffusers** 等框架中设置 **Stable Diffusion 3** 的困扰，以及针对 **RTX 2070** 和 **P40 显卡** 的 GPU 相关讨论。
  - [**高精度要求**](https://arxiv.org/abs/2210.02671) 以及 **RL 的 PPO** 等模型中的稳定性挑战，促使大家分享了关于优化技术、分词器（tokenizer）改进以及 **PyTorch distributed log** 错误的社区排查经验。
5.  **AI 伦理与行业动态**：
  
  - **关于 Nightshade 技术的伦理担忧**以及对 Perplexity AI 涉嫌抄袭的分析（[Forbes 文章](https://www.forbes.com/sites/randalllane/2024/06/11/why-perplexitys-cynical-theft-represents-everything-that-could-go-wrong-with-ai/)）强调了问责制的必要性。
  - **微软通过利用与 Tesla 的讨论收购 OpenAI 49% 股份**（[推文](https://fxtwitter.com/nicoleperlroth/status/1800946061613416659?s=46)）突显了科技巨头内部的战略举措，而 **ARC Prize 超过 1,000,000 美元的奖励**引发了关于行业基准的广泛讨论。

---

# PART 1: 高层级 Discord 摘要

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**Stable Diffusion 3 释放潜力与问题**：新发布的 **Stable Diffusion 3 Medium** 拥有更好的质量和先进的提示词理解能力，但据用户报告，它在人体解剖结构的准确性方面表现不佳。讨论显示出对性能的褒贬不一，一些人认为其表现平平，并对安装和微调中的技术障碍表示担忧。

**令人困惑的许可协议**：**SD3 的许可条款**在社区中引发了激烈辩论，焦点在于其对商业用途的限制，许多人认为这些限制对于实际应用来说过于苛刻。

**写实主义承诺遭遇质疑**：用户认可 SD3 在增强面部和手部真实感方面所做的努力，但与 **SD 1.5** 和 **SDXL** 等旧版本相比，结果的一致性仍然是一个有争议的点。

**资源效率良好，但定制成本可能很高**：工程师们赞赏 SD3 高效的 GPU 利用率和定制选项，尽管对于微调的资金和技术门槛存在担忧，尤其是针对特定领域的内容。

**安装集成焦虑**：已发现将 SD3 集成到 **ComfyUI** 和 **diffusers** 等流行框架中的各种问题，引发了社区内的协作排查工作。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **大模型对 VRAM 的需求 - 不开玩笑**：**Qwen2 72b** 的 **VRAM 需求**与 Llama 3 70b 相似，约为 **48GB**；对于需要超级计算机的幽默建议被现实情况否定了。
- **将 Pytorch 转换为 Safetensors**：尝试使用[此工具](https://huggingface.co/spaces/safetensors/convert)将 Pytorch bins 转换为 safetensors 时，用户在 Google Colaboratory 上因 RAM 限制而遇到障碍；建议转向资源充足的 Kaggle。
- **提升微调效率**：为了缓解 **Qwen2-1.5b** 在低秩配置下的微调困扰，在使用 Qlora 和 Galore 等方法时，用户建议最小 rank 为 64 或 128。
- **排除 16-bit 合并故障**：模型合并到 16-bit 后性能下降，引起了对 tokenizer 和 GGUF 可能存在异常的关注；成员们期待在即将发布的补丁中解决。
- **文档方面的社区努力**：志愿者们开始着手改进文档，指出数据准备和 chat notebook 的使用是亟待改进的领域，并建议制作视频教程以提高用户参与度。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Zoom 出故障，但博客很出彩**：虽然技术问题推迟了 Zoom 会议的录制，但 Nehil 关于[预算分类的博客文章](https://brain.nehiljain.com/posts/how-i-improved-my-prompting-for-budget-categorization/)因其在 prompting 中对课程概念的实际应用而受到称赞。
- **池化层难题与范围蔓延担忧**：关于使用 Huggingface 等库向模型添加 pooling layers 的技术细节讨论展示了社区的协作方式，同时关于广域范围 chatbots 与局部范围接口实用性的辩论也十分激烈，理由是出于对数据隐私和安全的担忧。
- **量化的分量**：社区对 Llama-recipes 的 "pure bf16" 模式以及正在进行的 optimizer quantization 研究感到兴奋，这表明了在模型效率与计算精度之间取得平衡的驱动力，[包含 Kahan summation 的优化器库](https://optimi.benjaminwarner.dev)的分享也证明了这一点。
- **积分/额度引发的询问**：社区收到了几起关于 OpenAI 等平台积分缺失或延迟的询问，并提示查看置顶评论以获取指导，同时通过集中消息来简化积分相关的讨论。
- **Modal 创作者、Datasette 爱好者和聊天机器人亲密度**：通过 Modal 为 Kaggle 竞赛 finetuning LLMs 的前景、Datasette 的命令行魅力，以及来自 Anthropic Claude 的以角色为中心的 AI 对话模型搅动了技术圈，强调了对集成、分析洞察和用户体验的热情。

请记得查看每条消息历史记录中的具体链接，例如 [Datasette 的稳定版本](https://llm.datasette.io/en/stable/)、Simon Willison 的 [GitHub repository](https://github.com/simonw/simonwillisonblog) 以及提到的 [meetup events](https://lu.ma/iulmro47) 以获取这些话题的更多详情。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **智能手表与传统的碰撞**：一场有趣的交流引发了关于 **WearOS** 等智能手表与传统数字手表主导地位的讨论，其中还穿插了对 Apple 的调侃，突显了个人对可穿戴技术的偏好。
- **Perplexity AI 应对技术问题**：**Perplexity AI** 用户报告了文件上传和图像解读功能的问题，暗示可能存在停机情况。然而，关于 **GPT-4o**、**Claude Sonnet** 和 **Llama 3 70B** 等不同语言模型的讨论集中在它们的性能上，成员们更倾向于将 **GPT-4o** 视为顶级竞争者。
- **Forbes 抨击 Perplexity 抄袭**：关于一篇指控 **Perplexity AI** 抄袭内容的 **Forbes 文章**展开了辩论，说明了在 AI 快速发展的时代，道德实践如履薄冰。该话题强调了在 AI 领域问责制和正确归因的重要性。[Forbes 关于 Perplexity 愤世嫉俗盗窃的文章](https://www.forbes.com/sites/randalllane/2024/06/11/why-perplexitys-cynical-theft-represents-everything-that-could-go-wrong-with-ai/)
- **AI 伦理与行业进展快照**：针对 **Nightshade 技术**及其在破坏 AI 模型方面的潜在滥用，人们提出了伦理担忧；与此同时，**Perplexity AI** 因其相较于 **SearXNG** 的先进功能而受到称赞。科技史上的另一个显著时刻是 **ICQ** 在运营 28 年后关闭，展示了通信技术不断更迭的步伐。[ICQ 关闭](https://www.perplexity.ai/page/ICQ-shuts-down-34n3T1XmQpuRpDB9VJcKVw)
- **API 集成步骤确认**：关于 **Perplexity API** 的初始设置得到了确认，伴随着简单的确认“这似乎可行”，随后是指令性建议 *"add API key here"*，这是 API 集成的关键——证明了让数字工具运行起来的细节过程。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **加拿大市场的 GPU 淘金热**：**3090 GPUs** 在 **Kijiji Canada** 上的价格已降至约 450 美元的实惠水平，激发了社区内组装机架的兴趣。
- **热量：有益的副产品**：有人提出了创新的建议，例如使用 **4090 GPUs** 为热水浴缸供暖，将 GPU 机架视为替代热源，同时思考可持续的数据中心设计。
- **对 100% 确定性的严谨追求**：一位工程师在代码中实现了 100% 的 determinism（确定性），但遇到了 loss values 达到 -inf 等问题，表明存在需要解决的潜在 Bug。
- **自定义 Matmul 实现的效率驱动**：提高计算效率的努力使得自定义 matmul 达到了 cuBLAS 70% 的速度，目前正在讨论移除对 cuBLASLt 的依赖以及解决实现稳定 FP8 的困难。
- **矩阵乘法的大变革**：[Scalable MatMul-free Language Modeling](https://arxiv.org/abs/2406.02528) 论文提出了一种新颖的方法，可以减少内存使用并保持 LLMs 的强大性能，引起了社区的关注和实现尝试。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Whisper WebGPU 为浏览器赋能语音功能**：[Whisper WebGPU](https://x.com/xenovacom/status/1799858691639796089) 实现了浏览器内快速、设备端的语音识别，支持 100 多种语言，通过本地数据处理保障用户隐私。
- **专家级金融 LLM 训练直播！**：对金融感兴趣的工程师可以关注由演讲者 Mark Kim-Huang 主持的[直播活动](https://events.dataphoenix.info/live)，该活动将深入探讨专家级金融 LLM 的训练见解。
- **谷歌 RecurrentGemma 9B 概览**：RecurrentGemma 9B 在长序列上的表现引发了社区热议，[帖子](https://x.com/osanseviero/status/1800607752038818260) 强调了其性能。成员们热衷于探索该模型的潜力。
- **微调 Mistral 的可行融合**：在这篇详细的 [Medium 文章](https://medium.com/ai-artistry/craft-your-ai-vision-fine-tuning-magic-with-mistral-on-mistral-server-6c9335232159) 中可以进一步探索在 Mistral 上微调模型，为处理可适配 AI 解决方案的工程师提供实用指南。
- **Diffusers 将支持 SD3 - 期待值达到顶峰**：SD3 的 `diffusers` 集成让社区倍感期待，工程师们正热切等待这一高级功能的推出。
- **Dalle 3 数据集提供创意储备**：由多样化数据驱动的 AI 可能会从 Dalle 3 提供的 100 万张以上带标注的图像中受益，详见此 [Dataset Card](https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions)。
- **推荐将 Simpletuner 用于扩散模型**：在讨论中，**simpletuner** 被推荐用于训练扩散模型，同时有警告称 **diffusers examples** 可能需要根据特定任务进行定制。
- **量化困境与疑问**：工程师们分享了使用 **quanto** 和 **optimum** 等量化工具优化 AI 的尝试，这表明在模型部署中需要更稳健且防错的解决方案。
- **Google Gemini Flash 模块 - 代码展示**：AI 社区呼吁谷歌 [开源 Google Gemini Flash](https://github.com/google/gemma.cpp/issues/221)，理由是其潜在优势和备受追捧的能力。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**Ilya Sutskaver 带来关于泛化的深刻见解**：Ilya Sutskaver 在 Simons Institute 发表了一场关于泛化的精彩演讲，可在 YouTube 上搜索标题 *An Observation on Generalization* 观看。另外，DeepMind 的 Neel Nanda 在 YouTube 视频 *Mechanistic Interpretability - NEEL NANDA (DeepMind)* 中讨论了记忆与泛化的关系。

**Llama 与 GPT 的对决**：Llama 3 8b instruct 的性能与 GPT 3.5 进行了对比，重点介绍了 Hugging Face 上的 Llama 3 8b 免费 API。GPT-4o 的编程能力引发了关于其性能问题的辩论。

**企业版：买还是不买？**：尽管有增强的上下文窗口和对话连续性等优势，但对于 GPT Enterprise 层级是否物有所值，意见不一。一位用户将 Teams 版与 Enterprise 版混淆，表明对产品方案存在误解。

**自力更生还是从头构建？这是个 AI 问题**：成员们建议微调现有的 AI（如 Llama3 8b）或寻求开源方案，而不是针对特定领域从头构建类似 GPT 的模型。

**技术问题单**：成员们面临各种技术问题，包括尽管声称支持但无法将 PHP 文件上传到 Assistants Playground，以及在生成响应时出现未指明解决方案的错误信息。此外，还有人请求减少基于大量 PDF 训练的 GPT-4 助手的引用次数；他们希望在保持数据检索的同时精简引用。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

**Qualcomm 的 AIMET 受到批评**：一位用户表达了对 Qualcomm **AIMET Library** 易用性的不满，称其为所遇到的“最差的库”。

**Rust 通过 RIG 增强凝聚力**：[**RIG**](https://github.com/0xPlaygrounds/rig) 发布，这是一个用于构建 **LLM-powered applications** 的开源 Rust 库，具有模块化、人体工程学设计以及 Cohere 集成等特点。

**关于 PaidTabs 在集成中使用 AI 的疑问**：社区内有推测认为 **PaidTabs** 可能正在使用 **Cohere AI** 进行消息生成，重点关注的是根据其 [6 月 24 日更新日志](https://blog.paidtabs.com/paidtabs-june-24-changelog/)，Cohere AI 目前尚不具备音频功能。

**工程师们可能会组建乐队**：对话转向了分享音乐爱好，由于音乐爱好者的数量众多，有人建议可以组建一个社区乐队。

**飞行模拟玩家的昂贵摇杆**：成员们讨论了高级摇杆设备（如 [VPC Constellation ALPHA Prime](https://virpil-controls.us.com/vpc-constellation-alpha-prime-r.html)）的高昂价格，并开玩笑地将其成本与钻石进行比较。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Attention Hypernetworks 增强泛化能力**：分享的一篇 [arXiv 论文](https://arxiv.org/abs/2406.05816)指出，Transformer 可以通过在 multi-head attention 中利用低维潜码，提高在符号瑞文标准推理矩阵（Raven Progressive Matrices）等智力任务上的组合泛化能力。
- **评估下一代 LlamaGen**：[LlamaGen](https://github.com/FoundationVision/LlamaGen) 仓库表明，像 Llama 这样的自回归模型在图像生成方面表现出色，足以媲美 state-of-the-art 性能，并提供了详细的解释和教程。
- **DPO 与自回归模型在多轮对话中的博弈**：成员们讨论了 Deterministic Policy Optimization (DPO) 在多轮对话中研究不足的问题，并建议探索 Ultrafeedback 数据集和 MCTS-DPO 方法。
- **数据集与架构之争**：讨论内容包括在获取用于 Large Language Model (LLM) 训练的小型代码数据集时面临的挑战，以及对 Griffin、Samba 等新型研究论文和模型在高效处理长上下文方面的批评和怀疑。
- **寻找终极预训练数据集**：在寻找类似于 DeepMind LTIP 的开源数据集时，有人推荐了 [DataComp dataset](https://huggingface.co/datasets/mlfoundations/datacomp_1b)，认为它因拥有更丰富的 10 亿图像-文本对编译，在 CLIP 模型上的表现优于 LAION。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **为聊天机器人编写 Python 脚本**：一位成员构思将 Python 脚本集成到 **LM Studio** 中，以允许聊天机器人模拟角色扮演，并被引导参考官方 [LM Studio documentation](https://lmstudio.ai/docs) 进行设置。建议添加 **RecurrentGemma-9B**，并附带了其 [Hugging Face 页面](https://huggingface.co/google/recurrentgemma-9b)。
- **对 LM Studio 0.2.24 故障的抱怨**：用户指出了 **LM Studio 0.2.24** 中的严重 Bug，抱怨 Token 计数错误以及模型/GPU 利用率不稳定。一个更有趣的查询探讨了使用 OpenAI API 让机器人创建 PowerPoint 演示文稿的可行性。
- **聚焦 GPU 与模型匹配**：在模型与 GPU 匹配的热烈讨论中，重点在于为 **RTX 2070** 寻找最快的选项，即 **Mistral7b**、**Codestral** 和 **StarCoder2**。建议还围绕特定代码模型（如 **Codestral**）、使用 **Unsloth** 的微调技巧以及使用 **vectordb** 的持久化策略展开。
- **GPU 市场价格动态**：成员们交流了全球 **P40 显卡** 的价格信息，记录了 eBay 上低至 **$150 USD** 的交易。进一步的评论涵盖了澳大利亚的 GPU 价格，从 **RTX 3060** 的 450 澳元到 **RTX 4090** 的 3200 澳元，揭示了稀缺性、功耗及其他考虑因素的现实情况。
- **现代 UI 预览**：分享了一张 **AMD ROCm** 技术预览框架中“现代 UI”的快照，[链接的照片](https://cdn.discordapp.com/attachments/1130536562422186044/1245478003044257932/image.png?ex=666a08c7&is=6668b747&hm=62024d42535148d6a9a7f23860dc0c011c6a2a48dba4759c91e04bb0f2fbe03f)因其时尚的呈现赢得了成员的赞赏。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **RAG 在真实世界部署中的坎坷之路**：该频道的一份[公告](https://discord.com/channels/1059199217496772688/1073670729054294197/1250214985762603079)邀请大家对在企业环境中部署 **Retrieval-Augmented Generation (RAG)** 提供见解，Quentin 和 Shreya Shankar 正在通过访谈收集反馈。
- **通过实体去重增强 Graph RAG**：频道中分享的一个教程建议通过引入实体去重和采用自定义检索方法来提高 Graph RAG 的效率，更多内容可以在[完整教程](https://t.co/fiFwDQS6WT)中探索。
- **利用 Excel 的空间严谨性提升 RAG**：一次对话强调了将 RAG 适配于 Excel 文件的挑战，指出有序的空间网格对于确保功能有效的重要性，参考[推文](https://t.co/vbdk8Yuw2t)提供了更多背景信息。
- **向量索引的烦恼与胜利**：成员们讨论了多个问题，包括 S3DirectoryReader 的解密问题、Redis 索引中 `MetadataFilter` 的失效、向量数据库中的 Markdown 格式挑战，以及在 `CondensePlusContextChatEngine` 中自定义 Prompt 的策略。
- **破解 Text-to-SQL 查询中的上下文密码**：社区中的一位 AI 工程师指出 Text-to-SQL 查询中的上下文识别问题，即确定项的性质（例如 "Q" 是否指代产品名称）仍然是一个挑战，导致生成了错误的 SQL 查询。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **ARC Prize 引发讨论**：为开发 ARC-AGI 基准测试解决方案而新设立的 **$1,000,000+ ARC Prize** 引发了广泛讨论，强调了该基准测试在通过技能获取效率衡量通用智能方面的作用。尽管 ARC-AGI 具有重要意义，但业界对其明显缺乏了解的情况也令人担忧。
- **科技媒体评价褒贬不一**：Kyle Wiggers 在 TechCrunch 上发表的一篇关于 AI 的文章因方法流于表面而受到批评；而对于一场涉及 AI 与人类智能关系的播客采访，反应则褒贬不一，其中一些争论点集中在遗传血统在决定智力中的作用。
- **微软获得 OpenAI 股份**：一条推文揭示了微软如何利用与 Tesla 的讨论，以 OpenAI 的参与作为激励，潜在地吸引 Tesla 进入 Azure 平台，从而获得了 OpenAI 49% 的股份。
- **机器人开发与 AI 更新引发热议**：讨论了 AI 工具的问题与进展：`june-chatbot` 遇到 NVDA API 错误，显示出稳定性问题；"SnailBot" 因反应迟缓而得名；对 LMSYS 的挫败感；以及对 Luma Labs 的 Dream Machine 新 Text-to-Video 能力的兴奋。
- **AI 强化学习的发展**：成员们剖析了 Apple 的一篇博客文章，分享了其涉及人工标注和合成数据的混合数据策略，并讨论了 @jsuarez5341 推文中提到的 PPO (Proximal Policy Optimization) 实现与伪代码之间的差异。同时，Tulu 2.5 的性能更新和 Nathan Lambert 对 RL 实践的探索显示了社区对当前 AI 方法论的深入研究。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **通用模型征集合作者**：Manifold Research 正在寻找合作伙伴，共同开发**用于多模态和控制任务的 Transformers**，目标是创建一个类似于 **GATO** 的大规模开源模型。你可以通过 [Discord](https://discord.com/invite/a8uDbxzEbM?ref=manifoldrg.com) 加入他们的行动，在 [Github](https://github.com/ManifoldRG?ref=manifoldrg.com) 上贡献代码，或在 [Google Docs](https://docs.google.com/document/d/e/2PACX-1vQgq32ChlP_e26mRPgfC31lZJCcAHAgbJ_Tn1nfzq8pfysoPAUqAWnel87Qc26h2Q/pub?ref=manifoldrg.com) 上查看他们的预期目标。
- **智能工厂结合群聊**：讨论围绕 [GendelveChat](https://x.com/gendelvechat/status/1800580692046405784?s=46&t=eY--9rPoOkHV-u9kocxCMA) 展开，该项目展示了使用 @websim_ai 为智能工厂等行业模拟的群聊 UX；同时，StabilityAI 发布了 [Stable Diffusion 3 Medium](https://x.com/StabilityAI/status/1800875914299048404) 的开放权重。
- **AI 进展与数据流水线提案**：Apple 的 AI 受到剖析，[Introducing Apple Foundation Models](https://machinelearning.apple.com/research/introducing-apple-foundation-models) 重点介绍了其 3B 模型和压缩技术。[Stable Diffusion 3 Medium 宣布发布](https://huggingface.co/stabilityai/stable-diffusion-3-medium)，具有视觉改进和性能提升；同时，一个利用 **Cat-70B** 和 **oobabooga** 等工具处理 ShareGPT 数据的流水线被提议用于 Character Codex 项目。
- **语言模型帝国**：在征求日语语言模型建议时，用户推荐了具有 API 访问权限的 **Cohere Aya** 和 **Stability AI** 模型，特别称赞了 **Aya 35B** 的多语言能力（包括日语）。对于更高性能的需求，**Cohere Command R+** (103B) 是日语模型需求的首选。
- **WorldSim 中的控制台与开源查询**：最近的一次更新使得在移动端和桌面端界面编写和编辑较长的控制台提示词（prompts）更加用户友好。有人对 WorldSim 的开源状态表示好奇，但在查询的时间范围内未得到确认。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

**Elon Musk 对阵 Apple 和 OpenAI**：据报道，在 Apple 与 OpenAI 建立合作伙伴关系后，Elon Musk 对 Apple 的 Twitter 账号采取了行动，Ron Filipkowski 在 [Threads.net](https://www.threads.net/@ronaldfilipkowski/post/C8F8woLt7rT) 上的帖子强调了这一进展。

**Google 的 Gemma 转向循环架构**：Google 的 [RecurrentGemma 9B](https://huggingface.co/collections/google/recurrentgemma-release-66152cbdd2d6619cb1665b7a) 已发布，能够快速处理长序列，同时保持与原始 Gemma 模型相当的质量，[Omar Sanseviero](https://x.com/osanseviero/status/1800607752038818260) 对此进行了预告。

**Transformer 学习面临“分布局部性”挑战**：Transformer 的可学习性面临“分布局部性”（distribution locality）的限制，[arXiv](https://arxiv.org/abs/2406.06467) 上的一篇论文对此进行了探讨，指出模型在根据已知规则组合新三段论方面存在挑战。

**利用 LlavaNext 专业能力修订 CC12M 数据集**：**CC12M 数据集**利用 **LlavaNext** 进行了修订，生成的重新标注版本现已托管在 [HuggingFace](https://huggingface.co/datasets/CaptionEmporium/conceptual-captions-cc12m-llavanext)。

**基于 TensorFlow 的机器学习库全球首秀**：一位工程师宣布推出了他们以 TensorFlow 为核心的机器学习库，支持并行和分布式训练，并支持 Llama2 和 CLIP 等一系列模型，该项目已在 [GitHub](https://github.com/NoteDance/Note) 上发布。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Mojo 未来对 TPU 的支持**：成员们讨论了如果 Google 为 MLIR 或 LLVM 提供 TPU 后端，**Mojo** 利用 **TPU 硬件**的可能性。这表明由于计划中的可扩展性，未来将支持多种架构，而无需等待官方更新。

**紧跟 Modular 发布步伐**：新的 **Mojo 编译器版本** `2024.6.1205` 已发布，其引入的条件一致性（conditional conformance）获得了积极评价，同时还有关于递归 trait 约束（recursive trait bounds）能力的咨询。更新说明和详情可以在 [最新的 changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 中找到。

**深入探索 Mojo 的功能与特性**：将代码从 `var` 更改为 `alias` 并未带来性能提升，而过时的 `Tensor` 模块示例问题已得到解决，并且在[最近的 Pull Request](https://github.com/modularml/mojo/pull/3007) 中引入了一个成功的指针转换解决方案。

**Modular 多媒体更新**：Modular 在各平台上表现活跃，发布了[新的 YouTube 视频](https://www.youtube.com/watch?v=uookgZ7Ojg8)以及来自其[官方 Twitter 账号](https://twitter.com/Modular/status/1800948901652181260)的推文更新。

**社区讨论与资源**：交流内容涵盖了从通过 **VSCode** 学习 **Mojo** 的建议（潜在资源见 [Learn Mojo Programming Language](https://ruhati.net/mojo)），到对科技影响力人物作为现代编程评论家的反思，分享内容中重点提到了 [Marques 和 Tim 的访谈](https://www.youtube.com/watch?v=pMX2cQdPubk)。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**Zep 在免费额度内缓解内存忧虑**：参与者认为 **Zep** 是内存管理的利器，前提是使用量保持在其**免费层级（free tier）**限制之内。

**Apple 在技术领域投放免费福利**：**Apple** 提供某些免费服务的举动引发了讨论，成员们认为这是一个显著的竞争优势。

**OpenAI API 价格亲民**：关于 OpenAI API 定价的辩论出现，有提到每月费用在 **$5-10** 左右，强调了 OpenAI 产品对工程师的负担能力。

**在 GCP 上配置高级模型**：一位用户成功在他们的 **GCP 账户**上实现了 **GPT-4o**，尽管指出了高昂的成本以及在将默认模型更改为 **gemini-flash** 或 **codestral** 时遇到的困难。

**OpenInterpreter 势头强劲**：重点介绍了综合资源，包括 [GitHub 仓库](https://github.com/OpenInterpreter/open-interpreter)、[代码 Gist](https://gist.github.com/0xrushi/e56085f93698c7267af9b1ba9643dc7a) 以及 [uConsole 和 OpenInterpreter 视频](https://odysee.com/@rushi:2/Openinterpreter-01-uconsole-test-part1:2)，用户们还在集思广益，探讨可能通过 [mini USB 麦克风](https://www.amazon.com/dp/B071WH7FC6)来增强语音交互。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**轻松实现 LLM 自定义指标**：[DeepEval](https://docs.confident-ai.com/docs/metrics-introduction#using-a-custom-llm) 允许用户平滑地集成语言模型的自定义评估指标，增强在 **G-Eval、Summarization、Faithfulness** 和 **Hallucination** 方面的能力。

**透明 AI 与无审查模型**：一场激烈的讨论指出用户对**无审查模型（uncensored models）**的兴趣日益浓厚，承认其在为各种应用提供未经过滤的 AI 响应方面的价值。

**WizardLM-2 令人惊讶的低价**：关于 **WizardLM-2** 负担能力的咨询引出了相关见解，即它可能通过使用更少的参数和战略性的 GPU 租赁来节省成本，引发了成员们对该模型效率的讨论。

**自托管 vs. OpenRouter**：在权衡利弊后，成员们得出结论，与 OpenRouter 等解决方案相比，**自托管大语言模型 (LLMs)** 只有在持续的高需求下，或者在已有硬件能力抵消成本的情况下，才具有经济意义。

**用于批量推理的 GPU 租赁**：公会就租用 GPU 进行批量推理（batch inference）的可行性交换了意见，涉及成本效益和效率，并建议使用 **Aphrodite-engine / vllm** 等工具来优化大规模计算。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **探索本地模型训练的边界**：一位工程师测试了使用 **mlx library** 在本地运行 **7 billion parameter** 模型，并反馈称“极其缓慢，但确实能运行，笑死”，而其他人则建议尝试 **llama.cpp** 以获得更好的性能。另一位成员在寻求关于训练 15 亿参数的 **QWEN 2** 的见解，但讨论中未提供具体的性能数据。
- **Runpod 令人困惑的路径持久化问题**：成员们讨论了 Runpod 上的一个问题，即在将数据卷挂载到 **/workspace** 时，经常会导致 **/workspace/axolotl** 被覆盖，从而需要重新安装或重新克隆 Axolotl —— 这是开发环境配置中一个持续存在的困扰。
- **排除冗余指令**：在文档频道中，有人指出模型上传过程中的“Step 2”是多余的，因为仓库是自动创建的，这表明文档可能需要更新。
- **PyTorch 分布式故障**：一位工程师在使用 PyTorch 的分布式启动器（distributed launcher）时遇到了 *ChildFailedError*，得到的建议是检查环境设置、验证配置，并可能需要增加共享内存，更多排查步骤可参考 [Accelerate 文档](https://github.com/huggingface/accelerate/tree/main/docs/source/basic_tutorials/troubleshooting.md#L50L145)。
- **LoRA 的学习曲线**：有关于利用 LoRA 将训练从补全格式（completion format）转换为指令格式（instruction format），以及在 24GB GPU 系统上合并 Mistral 7B 的 qLoRA 训练结果的咨询，这表明关于处理资源限制的讨论正在进行中。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **谷歌推出 RecurrentGemma 9B**：谷歌的 **RecurrentGemma 9B** 承诺在快速处理长序列方面取得突破，拥有与 **Gemma** 类似的实力，并推出了基础版（base）和指令微调版（instruct-tuned）。该模型的详细信息及其与前代产品的比较可以在[此处](https://huggingface.co/collections/google/recurrentgemma-release-66152cbdd2d6619cb1665b7a)的集合中找到，底层研究见 [Griffin 论文](https://arxiv.org/abs/2402.19427)。
- **对 Alexa AI 缺陷的细致分析**：Mihail Eric 揭露了 **Alexa AI** 失败的幕后原因，指出技术脱节和组织效率低下是创新的主要障碍，尽管拥有深厚的资源，但公开的突破却寥寥无几。详细的推文揭示了幕后的挑战，可在此处[查看](https://x.com/mihail_eric/status/1800578001564057754)。
- **ARC 奖项开辟 AI 基准测试新前沿**：一个与 ARC 任务数据集相关的奖项旨在提升对 AI 中人类解决问题方法的理解和开发，该数据集目前包含超过 4,100 条交互历史。相关资源如[视频](https://youtu.be/zbo6SdyWGns?si=5UypK-JD5h7Gz-SJ)和多个数据集已发布，并可通过[此链接](https://neoneye.github.io/arc/)受邀参与。
- **谷歌发布医疗专用语言模型**：谷歌的最新成员 **Personal Health Large Language Model** 利用可穿戴设备数据提供量身定制的健康见解，据报道其表现超越了行业专家。关于该模型能力和设计的深入信息可以在[此处](https://x.com/chefjeffsf/status/1800597192593621100)找到。
- **斯坦福分享关于 AI 快速演进的见解**：hwchung27 在斯坦福大学的一次讲座中展示了 AI 尖端状态的宝贵见解，关注了更廉价的计算资源和 Transformer 架构增长所带来的变革性影响。观看这段富有见地的[讲座](https://youtu.be/orDKvo8h71o?si=RIfyZ7NSUAJifOBF)并查阅配套[幻灯片](https://docs.google.com/presentation/d/1u05yQQaw4QXLVYGLI6o3YoFHv6eC3YN8GvWD8JMumpE/edit?usp=sharing)以获取详细叙述。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **MLPerf 上 RetinaNet 的悬赏任务**：George Hotz 宣布了一项 **600 美元的悬赏**，用于在 MLPerf 基准测试中实现 RetinaNet，并承诺如果该贡献被 MLPerf 采纳，奖金将翻倍。查看相关的 Pull Request 请点击[这里](https://github.com/tinygrad/tinygrad/pull/4245)。
- **渴望高效的数据加载**：工程师们对在 PyTorch 中优化数据加载（尤其是在 HPC 集群上）所耗费的时间表示担忧，并建议将 [WebDataset](https://github.com/webdataset/webdataset) 作为一种可行的解决方案。
- **起草 TinyGrad 的数据加载器**：George Hotz 分享了 TinyGrad 数据加载器 API 的计划，其中包括加载记录、随机打乱缓冲区大小（shuffle buffer size）和批次大小（batch size）的函数，并提到了 Comma 的 "gigashuffle" 作为参考。
- **TinyGrad 进化至 0.9.0**：正如 [GitHub Pull Request](https://github.com/NixOS/nixpkgs/pull/316931) 所确认的，TinyGrad 0.9.0 版本现已在 Nix 仓库中可用，其特点是将 gpuctypes 直接包含在代码库中。
- **MLPerf 基准测试与社区热议**：最新的 MLPerf 结果已发布，展示了 tinybox red/green 的基准测试，同时还有媒体报道，如 [Heise.de](https://www.heise.de/news/KI-Benchmark-MLPerf-Erste-AMD-Beschleuniger-mit-Minimalauftritt-9760531.html) 上的一篇关于 TinyGrad 的德语文章。进一步的讨论暗示未来会有一篇博客文章将 TinyGrad 的速度与理论极限进行对比。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **LLM 需要结构化输出**：成员们强调了 **LLM** (Large Language Models) 必须具备定义的语法或 JSON Schema 才能有效地作为 Agent 使用。输出的标准化以便外部程序解析，被认为是应用层实用性的关键。
- **通过集成 Schema 简化 llamafile**：关于优化 `llamafile` 使用的讨论提出了一个两步简化流程：首先将 JSON Schema 转换为语法（grammar），然后集成该语法以供使用，并给出了一个命令示例：`llamafile --grammar <(schemaify <foo.sql)`。
- **共享对象的有效打包**：一场关于在 `llamafile` 发行版中包含 `ggml_cuda.so` 和 `ggml_rocm.so` 的最有效方法的架构讨论展开了，其中包括一个共享的 Bash 脚本，并提到需要针对 AMD 和 tinyblas 等不同库进行手动调整。
- **Magit 作为同步解决方案**：幽默地引用了一个名为 "Interview with an Emacs Enthusiast in 2023 [Colorized]" 的视频，以此为例说明 [Magit](https://www.youtube.com/watch?v=urcL86UpqZc)（Emacs 的 Git 界面）的使用，展示了其在同步 `llama.cpp` 等文件中的应用。
- **LLM 中的 Schema 应用**：对话强调了社区对应用结构化数据 Schema 以改进 LLM 输出的兴趣，这标志着工程界正趋向于增强 LLM 与下游系统的集成。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **苹果接口前景与 LORA 类似物**：苹果的适配器技术被比作 **LORA 层**，暗示了一种用于本地模型的动态加载系统，以执行各种任务。
- **复杂的 HTML 提取**：AI 工程师探索了不同的 HTML 内容提取工具，如 [htmlq](https://github.com/mgdm/htmlq)、[shot-scraper](https://shot-scraper.datasette.io/en/stable/javascript.html#example-extracting-page-content-with-readability-js) 和 `nokogiri`。Simonw 强调了使用 `shot-scraper` 进行高效的 JavaScript 执行和内容提取。
- **跳过繁琐步骤，直接总结**：**Chrisamico** 发现绕过技术性的 HTML 提取，直接将文章粘贴到 ChatGPT 中进行总结效率更高，从而省去了复杂的 `curl` 和 `llm` 系统。
- **Simon 建议使用 shot-scraper 进行抓取**：Simonw 提供了关于利用 `shot-scraper` 进行内容提取的说明，主张对于精通 JavaScript 的人来说，在过程中使用 CSS 选择器非常实用。
- **利用 Nokogiri 进行命令行学习**：为了让工程师能够利用他们的命令行专业知识，Dbreunig 分享了关于将 `nokogiri` 作为 CLI 工具使用的见解，并附带了一个解析和从 HTML 文档中提取文本的示例。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**LangChain Postgres 困扰程序员**：工程师们报告了 **LangChain Postgres 文档**的问题，发现包中缺少对使用至关重要的 checkpoint。文档可以在[这里](https://api.python.langchain.com/en/latest/checkpoint/langchain_postgres.checkpoint.PostgresSaver.html)找到，但困惑仍在继续。

**LangChain 中的 GPT-4 抱怨**：一名成员反映了在 `langchain_openai` 中使用 **GPT-4** 时出现的错误；建议切换到 `ChatOpenAI`，因为 `OpenAI` 使用的是不支持新模型的旧版 API。关于 OpenAI API 的更多信息可以在[这里](https://platform.openai.com/docs/api-reference/completions)找到。

**LangServe 中的分享混乱**：讨论了在 LangServe 的聊天游乐场（chat playground）中分享对话历史的困难，用户遇到的问题是“分享”按钮会导致空聊天，而不是显示预期的对话历史。该问题正在 [GitHub Issue #677](https://github.com/langchain-ai/langserve/issues/677) 中进行跟踪。

**Nostrike AI 的免费代码创作**：Nostrike AI 推出了一款新的免费 Python 工具，可以轻松创建 CrewAI 代码，并计划在未来支持导出 Langgraph 项目，邀请用户在 [nostrike.ai](https://nostrike.ai/) 进行探索。

**Rubik's AI 招募 Beta 测试人员**：Rubik's AI 被宣传为先进的 AI 研究助手和搜索引擎，目前正在寻求 Beta 测试人员，并提供使用促销代码 `RUBIX` 获得 2 个月免费试用的诱惑，涵盖 GPT-4 Turbo 和 Claude 3 Opus 等模型。点击[这里](https://rubiks.ai/)查看。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

**Discord 通过应用增强功能**：从 6 月 18 日开始，成员现在可以通过在服务器和私信中添加应用来增强其 Discord 体验。有关应用管理和服务器审核的详细信息和指导可以在 [帮助中心文章](https://support.discord.com/hc/articles/23957313048343) 中找到，开发者可以借助 [综合指南](https://discord.com/developers/docs/tutorials/developing-a-user-installable-app) 创建自己的应用。

**Torchtune 中的缓存难题**：关于 **Torchtune** 在每个计算步骤中增加缓存内存使用的对话已经展开，社区成员正在深入探讨以了解这一性能特征。

**Tokenizer 重构即将到来**：一份详细说明 Tokenizer 系统重大重构的 RFC 引发了关于多模态（multimodal）功能集成和设计一致性的讨论，该 RFC 可在 [GitHub](https://github.com/pytorch/torchtune/pull/1082) 上查看并参与贡献。

---

**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**AI Stack Devs (Yoko Li) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**YAIG (a16z Infra) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

# PART 2: 各频道详细摘要与链接

{% if medium == 'web' %}

### **Stability.ai (Stable Diffusion) ▷ #**[**announcements**](https://discord.com/channels/1002292111942635562/1002292398703001601/1250434682743685161) (1 条消息):

- **Stable Diffusion 3 Medium 开放权重发布**：Stability.ai 宣布了 **Stable Diffusion 3 Medium** 的开放权重，该模型被描述为该系列中*“最新且最先进的文本转图像 AI 模型”*。在[公告文章](http://stability.ai/news/stable-diffusion-3-medium)中了解更多信息。
- **卓越的质量与照片级写实**：新模型能够提供具有*“卓越细节、色彩和光照”*的图像，并利用 16 通道 VAE 解决了手部和面部写实等常见缺陷。
- **先进的 Prompt 理解能力**：它能够理解涉及空间推理和构图元素的冗长、复杂的 Prompt，使用三个文本编码器来平衡性能和效率。
- **卓越的排版能力**：利用 Diffusion Transformer 架构，实现了高质量的文本渲染，拼写和间距错误更少。
- **资源效率与定制化**：该模型可以在标准消费级 GPU 上高效运行而不会出现性能下降，并支持使用小数据集进行微调以实现定制化。

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1250163004239249610) (734 messages🔥🔥🔥):

- **SD3 在图像和解剖结构质量方面表现不佳**：用户注意到 **Stable Diffusion 3 (SD3)** 在准确生成图像方面存在问题，特别是人体解剖结构。一位用户评论道：“*它瞬间吃光了我所有的 16GB RAM 和 8GB VRAM，并在 256x256 分辨率下崩溃了*”，而另一位则表示：“*结果相当令人失望，几乎不比 SDXL 模型好。*”
- **SD3 模型的许可协议引发争议**：围绕 **SD3 许可协议** 对商业用途的限制展开了广泛讨论，这使得它对许多人来说并不实用。一位用户澄清道：“*商业用途需要许可证，即使你只是想出售生成的图像*”，这引发了关于其可行性的辩论，并将其与 **PixArt** 等其他模型进行了比较。
- **微调挑战与质疑**：成员们对微调 SD3 的能力和成本表示担忧，特别是针对 NSFW 内容等热门用例。一位用户强调：“*微调是针对基础模型进行的。这是投入巨资训练的基础模型*”，这表明还需要社区的进一步努力。
- **对比性能令部分人失望**：尽管是一个较新的模型，但与 **SD 1.5** 和 **SDXL** 相比，SD3 让许多人感到平庸。一位成员表示：“*当 SD3 产生更好的结果时，它们有时比 SDXL 好得多*”，但这些情况并不稳定。
- **技术设置与安装问题**：多位用户寻求在 **ComfyUI**、**StableSwarm** 和 **diffusers** 等现有框架中安装 SD3 的帮助。一些人遇到了错误并分享了设置资源，一位用户提到：“*现在必须等待 auto1111，因为有很多工作流都在使用它。*”

---

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1250170775781249164) (322 messages🔥🔥):

- **成员讨论大模型的 VRAM 需求**：关于 **Qwen2 72b 的 VRAM 需求** 及其相对效率的讨论正在进行中。用户建议它需要大约 **48GB 的 VRAM**，类似于 Llama 3 70b，尽管有些人开玩笑说需要“超级计算机”甚至“Windows XP”。
- **将 Pytorch Bins 转换为 Safetensors**：有人建议使用[此工具](https://huggingface.co/spaces/safetensors/convert)将 Pytorch bins 转换为 safetensors。由于 Google Colab 上的 RAM 不足，曾出现过一次失败的尝试，用户建议迁移到 Kaggle 以获得更多资源。
- **微调挑战与解决方案**：用户讨论了在使用 Qlora 和 Galore 等方法微调 **Qwen2-1.5b** 等模型时，低秩（low-rank）配置存在的问题。建议包括将 rank 增加到至少 64 或 128，以避免训练失败。
- **合并至 16-bit 后性能下降**：一位用户注意到他们的模型在合并至 16-bit 后性能变差，理由是出现了重复结果和幻觉等问题。这归因于可能的 tokenizer 和 GGUF 问题，有待在即将到来的更新中修复。
- **社区动态与支持**：成员们强调了维持社区响应的挑战，考虑到用户咨询量的增加，正在考虑是否需要志愿者或付费帮助。他们还讨论了在 **multi-GPU 支持** 和 **finetuning 脚本** 等领域的贡献和改进。

---

### **Unsloth AI (Daniel Han) ▷ #**[**random**](https://discord.com/channels/1179035537009545276/1179039861576056922/1250245408891080797) (20 messages🔥):

- **Unsloth 更新韩国版 Colab，引发好奇**：一位成员注意到 Unsloth 更新了 **Korean colab**，并询问其是否专门使用韩语 Prompt 进行回答。他们分享了观察结果，认为“详尽描述地球”这一回复出人意料地有趣。
- **Anthropic 的 Transformer Lens 调试模型 Attention**：一位成员提到尝试使用 **Anthropic's Transformer Lens** 在推理过程中调试模型的 Attention。他们认为这是解决幻觉（hallucinations）的一种很有前景的方法，尽管它展示了单个神经元的偏差如何影响文本生成。
- **Apple 的端侧 AI API 引发讨论**：讨论集中在 [WWDC 2024](https://developer.apple.com/videos/play/wwdc2024/102/) 约 3:00 分钟处介绍的 **Apple Intelligence API**。对话涉及了用于端侧 AI 的 LoRA Adapters，以及尽管存在潜在缺点，但迎合 Apple 生态系统的盈利能力。
- **财务实用性推动了对 Apple 的 AI 支持**：一位成员认为支持 **Apple 生态系统** 是合理的，因为其用户群具有很高的商业价值。他们强调了“支付账单”的实际需求，并指出“10 亿台现成设备”是一个尚未开发的巨大市场。
- **Apple 的 AI 支持广泛的设备范围**：尽管最初担心兼容性，但后来澄清 **Apple AI** 将在所有支持下一代 OS 更新的设备上可用。这消除了 AI 仅限于较新设备的担忧。

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1250179052145217688) (78 messages🔥🔥):

- **EOS token 混淆得到澄清**：一位成员询问为什么一个 Notebook 包含 **EOS token** 而另一个没有。另一位成员澄清说，第二个 Notebook 使用 `<|eot_id|>` 作为 EOS token，但 *“如果你愿意，仍然可以添加它”*。

- **Unsloth 安装问题得到解决**：一位成员分享了在 **GTX 1080/Windows10** 上安装 Unsloth 的问题，但通过特定的 pip 安装解决了。另一位成员建议使用 *"pip install --force-reinstall jupyter"* 来修复 Jupyter Notebook 中的导入问题。

- **4-bit 模型的性能影响**：用户询问 4-bit 模型的性能差异。澄清结果是，使用带有 `load_in_4bit=True` 的默认 Unsloth 配置通常不会降低性能，且下载速度更快。

- **Llama3 模型生成自我回复**：一位成员报告说他们的 **Llama3 量化模型** 正在生成并回复自己的客户消息。这引发了关于更新 Unsloth 和使用正确 Chat Templates 的询问。

- **微调数据集大小的问题**：一位用户提到他们的数据集只有 30 条数据，不足以进行有效的学习。建议将规模增加到至少 300 条，并避免直接在数据集中重复，而是在训练期间使用多个 Epochs。


---

### **Unsloth AI (Daniel Han) ▷ #**[**community-collaboration**](https://discord.com/channels/1179035537009545276/1180144489214509097/1250443156865159179) (4 messages):

- **志愿者挺身而出协助文档编写**：一位成员表示愿意协助文档编写。另一位成员强调，许多用户询问关于**数据准备**和使用 Chat Notebooks 的问题，表明这些领域需要改进。
- **视频教程的潜力**：提出了未来制作**视频教程**以方便理解的想法。这表明了为用户提供更具参与感和易于获取的资源的积极态度。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**general**](https://discord.com/channels/1238365980128706560/1238365980128706563/1250166178597572759) (25 messages🔥):

- **Zoom 录制问题导致视频延迟发布**：由于转录步骤出现问题，近期课程（Paige Bailey, Ben Clavie）的 Zoom 录制视频尚未发布。Dan 报告称已向 Zoom 提交工单以解决此问题。
- **Nehil 关于改进 Prompting 的博客文章**：Nehil 根据课程所学撰写了一篇[博客文章](https://brain.nehiljain.com/posts/how-i-improved-my-prompting-for-budget-categorization/)，旨在改进用于预算分类的 Prompting。该文章受到了社区的好评，Dan 对此表示赞赏。
- **DOLPHIN 团队介绍极具前景的 Spectrum 方法**：Devnull 重点介绍了 DOLPHIN 团队关于 Spectrum 方法的新论文，该方法与 QLoRA 期间的 LASER 类似。论文已在 [arXiv](https://arxiv.org/abs/2406.06623) 上发布。
- **在 Huggingface 中添加池化层的挑战与技巧**：Healthymonkey 就如何向 Mistral 或 Llama 等微调模型添加 AttentionPooling 或 GemPooling 等池化层寻求建议。Shamik 协助澄清了集成点，特别是 backbone 与 head 之间的位置。
- **反对宽泛范围聊天机器人的论点**：Chrislevy 发起讨论，探讨为什么范围无限的通用聊天机器人是一个坏主意，并寻求资源以说服领导层采用更具针对性的界面。Sidhusmart 建议围绕部门间数据使用的安全/隐私问题展开论证。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**🟩-modal**](https://discord.com/channels/1238365980128706560/1241044231829848125/1250263048866758821) (7 messages):

- **为 Kaggle 竞赛微调 LLM**：成员们讨论了使用 **Modal** 微调 LLM 并将其上传到 **Kaggle** 进行竞赛的潜力。一位成员确认他们已经微调了 **mistral-7B** 并在 Kaggle 上进行了推理。
- **Modal 使用与额度查询**：一位用户询问关于接收使用 Modal 的额外额度，反映出对有效最大化资源利用的兴趣。
- **Modal 计费与安全**：一份回复简要介绍了 Modal 的计费和安全方法。*“计费限制非常显著且默认较低，”* 虽然在 Modal 中使用 gRPC 服务包含简单的身份验证，但由于缺乏内置的 RBAC，设置 REST APIs 需要自定义安全措施。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**learning-resources**](https://discord.com/channels/1238365980128706560/1241089743933149204/1250361991831486528) (2 messages):

- **参数高效微调 (Parameter Efficient Tuning) 解释**：根据 Lester 等人（2021年）的引用，PET 包括 **LoRA** 和 [prompt tuning](https://github.com/google-research/prompt-tuning) 等方法。FMT 代表全模型微调 (full-model tuning)。
- **预训练数据缩放对于 GPU-poor 用户不切实际**：一位成员对文章中讨论微调的预训练数据缩放部分表示困惑，指出由于 GPU 需求过高，这并不具备可操作性。*“对于 GPU-poor 的我们来说，这似乎不可行，我们不会进行完整的预训练。”*

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**hugging-face**](https://discord.com/channels/1238365980128706560/1241141471814488115/1250388609136328747) (3 messages):

- **新用户 Thiruvazhi 寻求额度确认**：一位名为 Thiruvazhi 的用户表示他们提交了表格并提供了电子邮件和 Hugging Face 用户名，但尚未收到额度。
- **引导成员查看置顶评论**：一位成员建议查看置顶评论以获取指导。另一位成员随后跟进，确认他们发现这些信息很有用。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**langsmith**](https://discord.com/channels/1238365980128706560/1241167367040405544/1250181894511792342) (2 messages):

- **Langsmith 因要求预付信息而令人失望**：一位用户赞赏 Langsmith 提供免费额度来测试功能，但发现设置支付需要信用卡这一点 *“相当令人反感”*。这种预先要求被批评为削弱了试用体验。
- **为咨询分享的电子邮件**：另一位用户分享了他们的电子邮件地址 [amirhossein.gh@gmail.com](mailto:amirhossein.gh@gmail.com)，可能与某项咨询或支持问题有关。然而，并未提供咨询的具体背景。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**berryman_prompt_workshop**](https://discord.com/channels/1238365980128706560/1242223275463938221/1250454623391842449) (1 messages):

- **Prompt 工作流咨询**：一位成员提问：“你们编写和迭代 Prompt，并评估哪个 Prompt 更好的工作流和工具是什么？”消息记录中没有提供进一步的讨论或回复。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**clavie_beyond_ragbasics**](https://discord.com/channels/1238365980128706560/1242223963346698250/1250290352930291742) (8 messages🔥):

- **模型与检索 Embeddings 之间没有关联**：已确认检索 Embeddings 与所使用的 LLM 之间*没有太大相关性*。一位用户提到，使用与 LLM 同一供应商提供的 Embeddings 并不会带来任何区别；相反，最好使用对文档检索效果最好的工具。
- **Zoom 视频问题已解决**：*Ben Clavies* 在 Zoom 上的视频问题现已修复。可以直接在 [Zoom](https://us06web.zoom.us/rec/share/5O0wWYxZ8SVoyqyxG_S4U6xeJLzVPaoggNtrK0XQUG8Ts_hq9UBWFOSjdiFCju3a.5LZBUyRziZgbAN_S) 或通过 Maven 观看。
- **实施基于 RAG 搜索的计划**：一位用户详细介绍了他们使用各种方法和工具创建基于 RAG 搜索系统的 *MVP 计划*。他们提到将结合 JXNLCO 的反馈方法、Freddy Aboulton 和 Jeremy Howard 的 fastHTML 演讲、Eugene Yan 的幻觉消除技术，并部署在 Railway 上。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**jason_improving_rag**](https://discord.com/channels/1238365980128706560/1242224099548332132/1250281309381857361) (3 messages):

- **新 Newsletter 提醒**：一位成员自豪地宣布了他们的新 Newsletter，旨在让读者提前了解他们的个人和技术写作。[在此订阅](https://subscribe.jxnl.co/)。
- **推迟模型微调**：另一位成员详细介绍了他们目前如何使用 **GPT-4o** 处理用户查询，并通过 Meta Tags 过滤产品搜索，而无需微调模型。他们曾考虑过通过微调来减少静态 Prompt 信息，但发现目前的设置已能满足需求。
- **使用 Postgres 最小化依赖**：一位成员表示倾向于通过将功能整合到 **Postgres** 中来将依赖保持在最低限度，包括管理关系数据和向量。他们质疑是否有必要添加 **LanceDB** 作为额外的依赖。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**saroufimxu_slaying_ooms**](https://discord.com/channels/1238365980128706560/1242224552415596554/1250202649421418517) (12 messages🔥):

- **对 4-bit 量化计算的好奇**：一位成员询问当模型以 4-bit 量化时，内部模型计算是如何工作的，包括反量化和乘法的细节。另一位成员确认，权重在乘法之前会被反量化为 bf16。
- **Llama-recipes 的纯 bf16 模式令人印象深刻**：Jeremy Howard 分享了 Llama-recipes 具有“纯 bf16”模式，利用 Kahan Summation 来避免精度问题。他询问其他人是否尝试过，并表示自己的体验非常好；另一位成员链接了他们自己包含 Kahan Summation 的[优化器库](https://optimi.benjaminwarner.dev)。
- **Weight-Only 量化详解**：一位成员澄清说，Weight-Only 量化涉及在乘法之前对权重进行反量化，在 A100 GPU 上通常转换为 dtype bf16 或 8-bit。这一概念被进一步认可为一种有益的方法，有助于灵活创建自定义算法。
- **优化器量化的活跃研究**：提到目前有针对优化器量化的持续活跃研究。这可能会进一步创新和优化模型训练。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**axolotl**](https://discord.com/channels/1238365980128706560/1242542198008975430/1250396870824427564) (3 messages):

- **使用 Axolotl 微调时遇到不一致问题**：一位用户报告在使用 Axolotl 和 `llama-3` 进行微调时出现 **OSError**，收到提示连接 Huggingface 加载配置文件失败的消息。他们指出使用 `open-llama` 进行微调则没有问题。
- **通过更改 Token 权限解决访问问题**：该用户随后发现问题源于 Huggingface 访问 Token 的权限。他们建议遇到相同问题的其他用户调整 Token 设置，以**允许访问公开的受限仓库 (public gated repositories)**。

### **LLM Finetuning (Hamel + Dan) ▷ #**[**wing-axolotl**](https://discord.com/channels/1238365980128706560/1242564077151326388/1250183703771218093) (4 messages):

- **在 Jarvis-labs 上配置 Axolotl 以便于使用**：一位用户提到在 Jarvis-labs 机器上使用**课程额度运行 Axolotl**，该机器已预先配置好 Axolotl。他们将其描述为在没有本地 GPU 的情况下进行微调模型的“非常简便的方法”。
- **使用纯 Axolotl 检查预处理数据**：一位用户分享了一个来自 [Modal 的 notebook](https://github.com/modal-labs/llm-finetuning/blob/main/nbs/inspect_data.ipynb)，用于检查预处理后的数据。另一位成员建议通过遵循 notebook 中概述的相同步骤，并在配置文件中确保正确的 dataset 路径，从而在本地运行 Axolotl 进行数据预处理，以便独立使用 Axolotl。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**charles-modal**](https://discord.com/channels/1238365980128706560/1242564177952768062/1250202407137312769) (4 messages):

- **额外额度查询已澄清**：一位用户询问为何未收到额外的 500 额度，确认额度已在 UTC 时间午夜左右发放。另一位用户提到，最后一刻报名的人可能需要等到今天 UTC 时间午夜。
- **将额度问题重定向至特定频道**：建议用户将任何关于额度的问题提交到指定频道。该主题顶部有一条详细消息，旨在帮助用户解决疑问。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**simon_cli_llms**](https://discord.com/channels/1238365980128706560/1242664474276659320/1250240328410202172) (44 messages🔥):

- **Datasette 让众人印象深刻**：多位用户分享了对 [Datasette](https://llm.datasette.io/en/stable/) 的兴奋之情，强调了它与 `llm` 的无缝集成，以及用于检查对话日志的精美 Web UI。诸如 *“我无法将视线从屏幕上移开……惊叹不已……”* 之类的评论表明该工具赢得了大量赞赏。
- **Simon Willison 令人惊叹的工具链**：用户称赞了 Simon 的工作，评价如 *“Simon 是终端巫师”*，并不断强调他多产且富有洞察力的贡献，包括他的博客 [Simon Willison’s Blog](https://simonwillison.net/) 及其 [GitHub 仓库](https://github.com/simonw/simonwillisonblog)。他还为演讲提供了一份 [讲义](https://github.com/simonw/language-models-on-the-command-line/blob/main/README.md)。
- **Mistral-7B-Instruct 和 Hugging Face pipeline**：关于安装 **mistral-instruct** 等模型的查询引导大家分享了 [Hugging Face 模型卡片](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) 以及一些使用 Transformers pipeline 的 Python 代码。讨论帮助澄清了如何通过 `llm` 命令使用这些模型。
- **vLLM 提供 OpenAI 兼容性**：会议强调了 vLLM 可以直接与 `llm` 集成而无需特殊插件，因为它具有 OpenAI 兼容的 API。这一背景得到了 [vLLM 的 OpenAI 兼容性文档](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) 和 [Datasette 指南](https://llm.datasette.io/en/stable/other-models.html#openai-compatible-models) 链接的支持。
- **Anthropic Claude 获得赞誉**：讨论和链接（如 [Claude 角色研究](https://www.anthropic.com/research/claude-character)）突显了社区对角色驱动的 AI 人格的兴趣。用户对讨论期间分享的项目所展现出的深思熟虑的设计和高度实用性表示赞赏。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**credits-questions**](https://discord.com/channels/1238365980128706560/1243721538432270388/1250220855439655094) (12 messages🔥):

- **LangSmith 额度缺失**：一位成员提到未收到他们的 LangSmith 额度，并询问该联系谁寻求帮助。**Dan** 要求提供他们的电子邮件地址以便后续跟进。
- **集中处理额度查询**：**Michael J. Ward** 要求有额度问题的用户在 hugging-face 频道发帖，以便集中查询并确保问题得到及时解决。
- **未收到 OpenAI 额度**：一位成员表示尽管填写了表格，但仍未收到 OpenAI 额度。**Dan** 要求进行电子邮件验证，而 **Hamel** 建议检查 OpenAI 频道并联系相关负责人。
- **使用 Modal 的额外额度**：一位用户询问使用 Modal 是否可以使他们获得额外额度。**Dan** 建议在特定论坛询问 Charles 以获取更多信息。
- **追踪额度分配**：**doctormiko.** 建议准备一份列出所有可用额度的文档，因为他们已经记不清了。该建议目前尚未得到立即跟进。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**fireworks**](https://discord.com/channels/1238365980128706560/1245126291276038278/1250189456170811444) (6 条消息):

- **参与者请求额度协助**：多位参与者请求协助获取其 credits。提到的 ID 包括 alexey-zaytsev-22319、sudarshansivakumar-4fadb8、srvaldepenas-c068e3 和 solmaz-sh-fe1dbf。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**braintrust**](https://discord.com/channels/1238365980128706560/1245407617031999581/1250273501286367275) (2 条消息):

- **额度过期政策确认**：一位成员询问：“所以 credits 的有效期只有 3 个月对吗？”已确认 **credits 的有效期仅为 3 个月**，但特殊情况可以私下与管理员讨论。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**west-coast-usa**](https://discord.com/channels/1238365980128706560/1245410680065097738/1250250282458157116) (2 条消息):

- **加入西海岸聚会！**：分享了一个 [meetup 活动](https://lu.ma/iulmro47) 链接供 **course** 参与者加入。建议参加者在问卷中注明其课程参与情况。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**europe-tz**](https://discord.com/channels/1238365980128706560/1245425547048386732/1250196475833749596) (3 条消息):

- **来自汉堡的问候**：一位来自德国汉堡的成员对 **regional meetups** 表示感兴趣。他们注意到一些成员在柏林，并提到顺便拜访总是值得的。
- **使用 Euro24 数据进行 Fine-Tuning**：另一位成员询问是否有人有兴趣使用 **Euro24 新闻及相关数据**（如球员和比赛统计数据）进行模型 fine-tuning。他们似乎正在为此任务寻找合作者。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**predibase**](https://discord.com/channels/1238365980128706560/1245803791710687272/1250500103073169428) (19 条消息🔥):

- **Beneyal 分享值得关注的论文**：Beneyal 发布了两篇 [arxiv 论文](https://arxiv.org/abs/2405.00732) 和 [另一篇](https://arxiv.org/abs/1605.07723) 的链接，讨论了该领域的最新进展。
- **Danbecker 贡献了近期研究**：Danbecker 提供了一个与讨论相关的 [arxiv 论文](https://arxiv.org/abs/2310.01352) 链接。
- **Laith0x0 指出性能差异**：Laith0x0 强调，根据数据或任务的不同，不同的 r 和 alpha 值（例如 r=256, alpha=256 与 r=32, alpha=64）表现出更好的性能，这表明结果与具体语境高度相关。
- **账户和支持查询占主导**：包括 laith0x0 和 hardboiledfish 在内的多位用户报告了诸如登录 Predibase 困难和 credits 差异等问题，Michael Ortega 通过私信和支持渠道直接处理了这些问题。
- **对见解表示感谢**：包括 laith0x0、codingwitcher 和 silverstar5654 在内的几位用户感谢 Travis 详尽且富有启发性的回答，表明了互助协作的讨论氛围。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**openpipe**](https://discord.com/channels/1238365980128706560/1245927847437008896/1250412608134058075) (2 条消息):

- **用户报告 credits 缺失**：两位贡献者都提到了 credits 未到账的问题。他们各自提供了电子邮件地址以识别账户并解决问题。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**openai**](https://discord.com/channels/1238365980128706560/1245927985123692575/1250175585557155840) (6 条消息):

- **使用 API 分级需绑定支付方式**：一位成员提到，他们需要添加支付方式并充值 $5 的 credits 才能使用 Tier 2 API 访问权限。这突显了即使是分级访问服务也需要设置支付方式的要求。
- **特定 API 版本解决问题**：对于 API 调用，用户报告 **gpt-4o** 无法工作，但 **gpt-4o-2024-05-13** 运行完美。这表明 API 中可能存在特定版本的可靠性差异或更新。
- **OpenAI credits 缺失**：一位成员报告未收到其 OpenAI credits，并提供了电子邮件以便后续跟进。这引起了对额度分配问题的关注。
- **GPT fine-tuning 的未来和数据见解**：一位用户整理了一系列问题，涉及 GPT-4 和 GPT-4-turbo 用于 fine-tuning 的未来可用性和可用性、基于 prompt 的技术与 fine-tuning 的影响、fine-tuned 模型请求的数据、多轮对话 fine-tuning 的改进，以及 OpenAI 开发优先级的反馈机制。这些问题是为某次未指明活动的演讲者准备的。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**hailey-evaluation**](https://discord.com/channels/1238365980128706560/1250194004176273479/1250194275581431931) (64 条消息🔥🔥):

- **Shamik 推荐必读论文**：一位成员称赞 Hailey 引用的论文非常出色（"total banger"），并分享了链接：[https://arxiv.org/abs/2405.14782v2](https://arxiv.org/abs/2405.14782v2)。这引发了围绕评估方法论的更多深刻讨论。
- **Hailey 详尽的回答吸引了成员**：Hailey 详细阐述了评估上下文的重要性以及公共学术机构创建的 evals 的历史背景。她提到，探索 sysprompts 对模型行为影响的数据集可能会非常有趣。
- **Hailey 的大量资源推荐**：Codingwitcher 整理了一份 Hailey 推荐资源的详尽列表，包括 Hailey 的 [Twitter](https://x.com/haileysch__)、[GitHub](https://github.com/haileyschoelkopf) 以及多篇学术文章和项目。
- **评估系统的行业标准与透明度**：成员们讨论了标准化评估以及共享多选题代码的想法，以减轻过拟合（overfitting）并确保透明度。“私有评估集（Private eval sets）和动态评估集（dynamic eval sets）”被认为是未来的方向。
- **Hailey 对 LLM-as-judge 持怀疑态度**：尽管近期趋势如此，Hailey 对 LLM-as-judge 系统表示怀疑，主张进行更多研究，并建议经过训练的 rankers 和小型 evaluators 往往可能被低估。

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1250166338346029078) (200 条消息🔥🔥):

- **WearOS 与数字手表**：一位成员问“为什么选 WearOS”，另一位回答“因为 Apple”。他们进一步阐述道：“我个人普遍讨厌数字手表。”
- **Perplexity AI 文件上传问题**：多位用户在使用 Perplexity AI 的文件上传和图像解读功能时遇到问题。一位成员指出，“它一直提示无法查看外部文件或链接”，另一位确认道，“是的，服务挂了。”
- **关于 GPT-4o 和模型的对话**：关于各种语言模型（包括 GPT-4o、Claude Sonnet 和 Llama 3 70B）进行了详细讨论。成员们比较了它们的性能和能力，一位成员指出，“总的来说，最好的模型可能是 GPT-4o。”
- **对 Perplexity AI 功能的担忧**：用户对 Perplexity AI 的功能集表示沮丧，特别是搜索和图像处理能力。一位成员表示，“Perplexity 应该能够通过 API 支持相同的功能，只是开发人员在实现功能方面相当缓慢。”
- **福布斯文章争议**：关于一篇指责 Perplexity AI 抄袭的《福布斯》（Forbes）文章引发了讨论。一位成员评论道，“如果我是《福布斯》，我也会非常生气，”并指出在新闻行业中正确署名的重要性。


**提到的链接**：

- [为什么 Perplexity 的愤世嫉俗式盗窃代表了 AI 可能出现的所有错误](https://www.forbes.com/sites/randalllane/2024/06/11/why-perplexitys-cynical-theft-represents-everything-that-could-go-wrong-with-ai/)：这是一个关于这一关键时刻的完美案例研究：AI 的好坏取决于监管它的人。
- [Perplexity 的 Discover Daily：Eric Schmidt 为乌克兰提供的 AI 无人机、SpaceX 的星舰里程碑、Humane 的火灾警报、ALS 研究突破以及 Apple Podcasts 上的 D-Day 故事](https://podcasts.apple.com/us/podcast/discover-daily-by-perplexity/id1732181427?i=1000658127961)：Discover Daily by Perplexity 节目，第 2024 年 6 月 6 日期。
- [这是谁？](https://www.perplexity.ai/search/who-is-this-UpstadEaSiSkXzfG8iSgjA)：<scratchpad> [提示词问“他是谁？”并提供了一张穿着西装对着镜头微笑的男人的图像。] [为了识别这个人是谁，我将分析...
- [他是谁？](https://www.perplexity.ai/search/who-is-he-nKR6DwraTAmY0HGdYCuHhg)：我不知道图像中的人是谁。

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1250207739683147809) (5 messages):

- **Perplexity AI 相比 SearXNG 的优势**：一个德语讨论串探讨了 [Perplexity AI 相比 SearXNG 的具体优势](https://www.perplexity.ai/search/Welche-Vorteile-hat-wPaImsRVTdieJxldcEvbMw)。对话集中在 Perplexity AI 的独特功能以及它为各个专业领域带来的潜在益处。
- **Nightshade 技术的伦理担忧**：[Nightshade 的使用](https://www.perplexity.ai/search/how-does-nightshade-uR1LfxUQQx60gkANQRIcsA)引发了多项伦理担忧，例如可能被误用来破坏 AI 模型，以及在确保有效的“中毒”像素方面的技术挑战。使 Nightshade 跟上不断发展的 AI 技术变得越来越困难。
- **ICQ 在 28 年后关闭**：著名的即时通讯服务 ICQ 在运营近 28 年后，将于 [2024 年 6 月 26 日关闭](https://www.perplexity.ai/page/ICQ-shuts-down-34n3T1XmQpuRpDB9VJcKVw)。ICQ 以其独特的用户 ID 和 "Uh oh!" 消息提示音而闻名，在衰落前曾拥有超过 1 亿用户。
- **Perplexity AI 分享行业更新**：Perplexity AI 的一段摘要视频重点介绍了各项行业更新，包括 [Raspberry Pi 的伦敦 IPO](https://www.youtube.com/embed/-YNbmp8QBx8)、Voodoo 收购 BeReal 以及 Mistral AI 60 亿美元的估值飙升。还提到了 General Motors 在德克萨斯州的自动驾驶努力。

**提到的链接**：

- [how does nightshade work](https://www.perplexity.ai/search/how-does-nightshade-uR1LfxUQQx60gkANQRIcsA)：Nightshade 的工作原理是对图像应用细微的像素级扰动，这种扰动对人类来说是不可察觉的，但会导致 AI 模型对...进行错误分类。
- [ICQ Shuts Down After 28 Years](https://www.perplexity.ai/page/ICQ-shuts-down-34n3T1XmQpuRpDB9VJcKVw)：ICQ 这一诞生于 1996 年的先驱即时通讯服务，在运营近 28 年后，将于 2024 年 6 月 26 日关闭。这次关闭标志着...
- [Perplexity](https://www.perplexity.ai/search/Welche-Vorteile-hat-wPaImsRVTdieJxldcEvbMw>)：Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可靠且实时的答案。
- [Perplexity](https://www.perplexity.ai/search/W)：Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可靠且实时的答案。
- [What advantages does perplexity.ai have over SearXNG?](https://www.perplexity.ai/search/What-advantages-does-suq3APsvQRabk0eAIOjuUw#2)：根据搜索结果中提供的信息，以下是 Perplexity AI 相比 SearXNG 的一些关键优势：1. 对话式...

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1250431495424966809) (2 messages):

- **初始设置确认**：一位用户通过消息 *"This seems to work."* 确认他们的设置似乎正在运行。这条简短的消息暗示了初始配置或故障排除的成功。
- **插入 API Key 的指导**：另一条后续消息指示 *"add API key here"*。这一步对于基于 API 的集成或服务的运行至关重要。

---

### **CUDA MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1250162673245622452) (29 messages🔥):

- **经济实惠的 3090 GPU 引发组装机架计划**：成员们讨论了组装 GPU 机架，其中一人提到 **Kijiji Canada** 上的 **3090 GPU** 价格实惠，约为 700-800 CAD（约 450 USD）。
- **GPU 机架的功耗考量**：有人对将 GPU 机架连接到家用电源插座表示担忧。一位成员建议拥有 200A 系统的房屋可能可以处理，但需要通过计算来确认。
- **利用余热造福社区**：成员们围绕将 GPU 机架用于非传统加热解决方案展开了玩笑和构思，例如用几块 **4090** 替换热水浴缸加热器，或创建能够重新利用废热的可持续数据中心。
- **管理功耗以提高效率**：成员们建议通过驱动程序限制功耗，以优化 flops/watts 性能。另一位成员建议使用 **MSI Afterburner** 来限制核心时钟频率，而不影响显存时钟。
- **用于优化的超图神经网络**：一位成员询问是否有人有超图（hypergraphs）方面的经验，并分享了 [**基于超图神经网络的组合优化 GitHub 仓库**](https://github.com/nasheydari/HypOp) 的链接。他们对其应用和有效性感到好奇。

**提到的链接**：[GitHub - nasheydari/HypOp: Hypergraph Neural Network-Based Combinatorial Optimization](https://github.com/nasheydari/HypOp)：基于超图神经网络的组合优化 - nasheydari/HypOp

---

### **CUDA MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1250167482283589694) (7 条消息):

- **Torch 缓存验证困扰**：一位用户询问如何验证 `TORCHINDUCTOR_CACHE_DIR` 和 `TORCHINDUCTOR_FX_GRAPH_CACHE` 是否生效，并咨询了如何开启日志进行验证。
- **CUDA 库的热身 (Warm up)**：另一位用户提到 CUDA 及其库需要热身才能正常运行的问题，并详细说明了 torch 内部如何利用这一点来进行算法决策。
- **对 AWQ 工作组的好奇**：一位用户询问为什么没有 AWQ 工作组，另一位用户回答说现有的工作组（例如[非对称线性量化 (asymmetric linear quantization)](https://discord.com/channels/1205223658021458100)中列出的那些）已经涵盖了类似领域。
- **对 8k 或 16k 基准测试的兴趣**：有人表示有兴趣查看 8k 或 16k 性能的基准测试，并链接到了 [YaFSDP GitHub 仓库](https://github.com/yandex/YaFSDP)。
- **C++ tensor iterators 问题**：一位成员在 C++ 内部机制的 tensor iterators 方面寻求帮助，特别是在使其适配 NHWC 输入和输出时遇到了困难。

**提到的链接**：[GitHub - yandex/YaFSDP: YaFSDP: Yet another Fully Sharded Data Parallel](https://github.com/yandex/YaFSDP)：YaFSDP：又一个完全分片数据并行（Fully Sharded Data Parallel）工具。通过在 GitHub 上创建账号来为 yandex/YaFSDP 的开发做出贡献。

---

### **CUDA MODE ▷ #**[**algorithms**](https://discord.com/channels/1189498204333543425/1189861061151690822/1250225367059529768) (2 条消息):

- **三值权重无矩阵乘法（MatMul-Free）语言建模报告**：一篇名为 [Scalable MatMul-free Language Modeling](https://arxiv.org/abs/2406.02528) 的新论文指出，可以在保持强劲性能的同时，从 LLM 中消除矩阵乘法。该研究声称在训练期间可减少高达 61% 的内存使用量，并且随着模型规模的增大，与全精度 Transformer 的性能差距会逐渐缩小。
- **无矩阵乘法（Matmul-free）LM 的 GitHub 项目引起关注**：一位成员分享了他们使用 GitHub 上 [MatMul-free LM 实现](https://github.com/ridgerchu/matmulfreellm) 的积极体验。他们称赞了该项目与 transformers 的集成以及模型的可用性。

**提到的链接**：

- [Scalable MatMul-free Language Modeling](https://arxiv.org/abs/2406.02528)：矩阵乘法 (MatMul) 通常在大型语言模型 (LLM) 的整体计算成本中占据主导地位。随着 LLM 扩展到更大的嵌入维度和上下文长度，这种成本只会不断增加……
- [GitHub - ridgerchu/matmulfreellm: Implementation for MatMul-free LM.](https://github.com/ridgerchu/matmulfreellm)：MatMul-free LM 的实现。通过在 GitHub 上创建账号来为 ridgerchu/matmulfreellm 的开发做出贡献。

---

### **CUDA MODE ▷ #**[**llmdotc**](https://discord.com/channels/1189498204333543425/1227345713348870156/1250196757963477103) (125 条消息🔥🔥):

- **ThunderKitten 集成微调提升了性能**：*"我将 ThunderKitten 的 '/attn_causal/h100_fwd.cu' 粗暴地复制粘贴到 '/dev/cuda/attention_forward.cu' 后，第一次尝试就成功了..."*。该代码在特定 Batch Size 下表现出比 cuDNN 略好的性能提升，但在需要更复杂集成的进一步修改中遇到了挑战。
- **实现了 100% 确定性（Determinism），但面临 Loss 问题**：*"我实现了 100% 的确定性！需要对 Master Weights 进行更多实验，随后将推送更改。"*。尽管实现了确定性，但他们遇到了 Loss 值的问题，有时会变为 -inf，这暗示了 `prepare_softmax_blockwide3` 等函数中存在 Bug。
- **FP32 版本代码及挑战**：自定义 Matmul 实现达到了非 Tensor Core cuBLAS 约 70% 的速度，并指出了对 cuBLASLt 的依赖。*"由于前向 Matmul 是我们使用 cuBLASLt 的唯一算子，仅凭这一个自定义实现就足以移除该依赖..."*
- **FP8 实现讨论**：详细阐述了可靠/稳定的 FP8 所需的要求，例如 Unit Scaling 和随机舍入（Stochastic Rounding），并指出了替换 cuBLAS/cuDNN 的挑战。*"值得指出的是，(3) 基本上意味着要完全替换 cuBLAS/cuDNN 😱"*
- **GPT-2 (774M) 训练心得及 S3 链接分享**：详细介绍了 774M 参数模型的训练配置，并指出由于额外的 Token 可能导致性能超预期。训练结果以及数据集重叠的潜在影响已记录在 [GitHub 讨论区](https://github.com/karpathy/llm.c/discussions/580)中。

**提到的链接**：

- [mdouglas/llmc-gpt2-774M-150B · Hugging Face](https://huggingface.co/mdouglas/llmc-gpt2-774M-150B)：未找到描述
- [GPT-2 (774M) reproduced · karpathy/llm.c · Discussion #580](https://github.com/karpathy/llm.c/discussions/580)：我让 GPT-2 774M 模型在我的 8X A100 80GB 节点上运行了约 6 天（150B Token，在 100B FineWeb 样本数据集上进行了 1.5 个 Epoch），训练几小时前刚刚结束，进展顺利...
- [无标题](https://us-west-2.console.aws.amazon.com/s3/object/llmc?region=us-west-2&bucketType=general&prefix=gpt2_774M/main.log)：未找到描述
- [无标题](https://us-west-2.console.aws.amazon.com/s3/object/llmc?region=us-west-2&bucketType=general&prefix=gpt2_774M/model_00286102.bin)：未找到描述
- [fp16 buffers for ADAM by ngc92 · Pull Request #289 · karpathy/llm.c](https://github.com/karpathy/llm.c/pull/289)：首个概念验证（PoC）实现
- [无标题](https://us-west-2.console.aws.amazon.com/s3/object/llmc?region=us-west-2&bucke)：未找到描述
- [无标题](http://llmc.s3-us-west-2.amazonaws.com/gpt2_774M/model_00286102.bin)：未找到描述

---

### **CUDA MODE ▷ #**[**bitnet**](https://discord.com/channels/1189498204333543425/1240586843292958790/1250225001823600742) (7 条消息):

- **语言建模中的三值权重（Ternary Weights）**：一位成员分享了一篇关于使用三值权重的 [Scalable MatMul-free Language Modeling](https://arxiv.org/pdf/2406.02528) 的论文。
- **Bit-packing CPU 问题**：报告了一个关于 **FP6-LLM** 的 Bit-packing 函数问题：如果数据在 **CPU** 上且未指定设备，该函数会失败。他们建议在实现中更好地处理此问题。
- **处理 Bit-packing 中的负维度**：另一个突出的问题是负维度在当前的 Bit-packing 函数中不起作用。建议要么支持负维度，要么明确注明不支持。
- **Bit-packing 修复的快速 PR**：一位开发者提到他们将解决 **FP6-LLM** 的 Bit-packing 问题，随后确认已经提交了一个快速 PR 来修复这些问题。
- **针对旧版本 PyTorch 的 Bit-packing 测试**：一位成员质疑为什么 Bit-packing 测试仅针对 PyTorch >= 2.4 运行，并指出 Eager 模式下的 Bit-packing 测试应该可以在之前的版本中进行。大家一致认为 **Eager 测试** 并不一定需要 PyTorch 的 Nightly 版本。

---

### **HuggingFace ▷ #**[**announcements**](https://discord.com/channels/879548962464493619/897387888663232554/1250188678266097666) (1 条消息):

- **Whisper 加速浏览器中的语音识别**：[Whisper WebGPU](https://x.com/xenovacom/status/1799858691639796089) 在浏览器中实现了“极速的 ML 驱动语音识别”，支持 100 种语言的多语言转录和翻译，且数据不会离开您的设备。
- **3D Arena 排行榜发布**：来自 Hugging Face 的官方 [3D Arena Leaderboard](https://x.com/dylan_ebert_/status/1800333314965852518) 已发布，目前的领先者是 InstantMesh。鼓励成员参与并投票。
- **Gradio 客户端进入稳定版**：现在您可以 [连接](https://x.com/evilpingwin/status/1799176924805333442) 到任何 Gradio 应用，并将其作为 API 与 JavaScript 和 Python 客户端配合使用，仅需几行代码。
- **MaPO 引入内存效率优化**：[MaPO](https://x.com/RisingSayak/status/1800447427503362471) 是一种新型的、内存高效的技术，用于在偏好数据上对齐 T2I 扩散模型，在对齐微调期间无需参考模型。
- **新一批生成式 AI 应用上线**：第二批生成式 AI 应用现已在 Hugging Face 兼容模型页面 [上线](https://x.com/julien_c/status/1800153076994801929)，扩展了支持的应用家族。

**提到的链接**：

- [来自 Xenova (@xenovacom) 的推文](https://x.com/xenovacom/status/1799858691639796089)：介绍 Whisper WebGPU：直接在浏览器中实现极速的 ML 驱动语音识别！🚀 它支持 100 种语言的多语言转录和翻译！🤯 模型在本地运行...
- [来自 dylan (@dylan_ebert_) 的推文](https://x.com/dylan_ebert_/status/1800333314965852518)：⚡️ 3D 新闻 ⚡️ 官方 @huggingface 3D Arena 排行榜已发布 🚀 目前排名第一的是 InstantMesh @xinntao 快去投票。https://huggingface.co/spaces/dylanebert/3d-arena
- [来自 amy (@a_e_roberts) 的推文](https://x.com/a_e_roberts/status/1800101457628377117)：您现在可以将任何预训练的 Hugging Face backbone 权重加载到 Transformers 的视觉模型中！只需使用 `backbone=checkpoint` 和 `use_pretrained_backbone=True` 指定 HF 检查点即可。
- [来自 pngwn (@evilpingwin) 的推文](https://x.com/evilpingwin/status/1799176924805333442)：昨天我们宣布了 @Gradio 客户端在 JavaScript 和 Python 中的稳定版本！现在您只需几行代码即可连接到任何 Gradio 应用并将其作为 API 使用！
- [来自 Daniël de Kok (@danieldekok) 的推文](https://x.com/danieldekok/status/1799081814709133342)：💎即将进入 @huggingface TGI 的下一个版本：支持 Marlin。⚡ Marlin 是由 @elias_frantar 和 @DAlistarh 开发的、针对 FP16xINT4 矩阵乘法高度优化的内核，旨在加速模型...
- [来自 Sayak Paul (@RisingSayak) 的推文](https://x.com/RisingSayak/status/1800447427503362471)：介绍 MaPO，一种在偏好数据上对齐 T2I 扩散模型的内存高效技术 🔥 我们消除了在进行对齐微调时对参考模型的需求。代码、模型...
- [来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文](https://x.com/reach_vb/status/1799113953500672094)：公告：您可以直接在 Transformers 中加载 GGUF 🔥 额外福利：还可以将它们转换为 Transformers 等效的状态字典（state dict）⚡
- [来自 Thomas Wolf (@Thom_Wolf) 的推文](https://x.com/Thom_Wolf/status/1799008162772836355)：- 全端到端策略 - 无远程操作 - 准全开源软件和硬件（甚至 Reachy-1 的硬件也是开源的 CC-BY-SA CAO 组装）
- [来自 Julien Chaumond (@julien_c) 的推文](https://x.com/julien_c/status/1800153076994801929)：第二批本地生成式 AI 应用现已在 @huggingface 兼容模型页面上线 🔥 欢迎加入这个大家庭，朋友们 🥰
- [来自 Victor M (@victormustar) 的推文](https://x.com/victormustar/status/1798994405522915621)：最近 HuggingChat 的表现简直疯狂！
- [来自 abhishek (@abhi1thakur) 的推文](https://x.com/abhi1thakur/status/1800511251145015393)：AutoTrain + Unsloth = 🚀🚀🚀 AutoTrain 现在增加了对 Unsloth 的支持，这意味着您可以使用 Unsloth 的优化来超快速且以更少内存微调 LLM 💥 您只需要...
- [来自 merve (@mervenoyann) 的推文](https://x.com/mervenoyann/status/1800555650273132967)：发布：smol vision 🌼 @huggingface 一个包含关于缩小、优化、加速、定制大型视觉模型 Notebook 的仓库！
- [来自 +RAIN Film Festival | 2024年6月11-14日 (@rainfilmfest) 的推文](https://x.com/rainfilmfest/status/1799838451140870149)：@rainfilmfest 的 L[AI]VE CINEMA 🎬 由 @radamar、@multimodalart 创建并由 Hugging Face 🤗 提供支持的实时电影体验，与 CCCB @cececebe 和 Artefacto @artefactofilms 合作...
- [来自 abhishek (@abhi1thakur) 的推文](https://x.com/abhi1thakur/status/1799029532332314852)：多模态基础模型挑战赛（Multimodal Foundation Models Challenge）已开启！赢取 1 万美元奖金 🎉 https://huggingface.co/spaces/ai-competition/MMFMChallenge

---

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1250166321765814302) (120 条消息🔥🔥):

- **合并量化 adapter 模型时遇到麻烦**：一位用户询问如何将量化后的 adapter 模型合并回原始模型，并得到了使用 `merge_and_unload` 方法的指导。提供了一个代码示例，以帮助使用 Trainer API 进行实现。
- **Google 发布 RecurrentGemma 9B**：一名成员分享了一篇[帖子](https://x.com/osanseviero/status/1800607752038818260)，透露 Google 推出了 RecurrentGemma 9B，强调其在处理长序列时具有**卓越的性能**，并拥有极高的吞吐量（throughput）和低延迟（latency）。成员们对该模型的能力反应热烈。
- **关于 Stable Diffusion 3 Medium 的讨论**：多位用户讨论了 [Stable Diffusion 3 Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium) 的发布和特性，注意到其增强的性能以及在图像生成中的复杂性。反馈包括其目前的缺点，如人体解剖结构的处理，并提出了潜在的改进建议。
- **Discord 上的验证问题**：一位用户在通过 LevelBot 进行验证时遇到困难，收到了调整 Discord 设置以允许来自服务器成员消息的指令。他们被引导使用具有 Hugging Face 读取权限的 token 完成验证过程。
- **格拉斯哥大学 Hugging Face 组织**：宣布在 Hugging Face 上创建了 [University of Glasgow 组织](https://huggingface.co/UniversityofGlasgow)，邀请教职员工、研究人员和学生使用大学电子邮箱地址加入。

**提到的链接**：

- [来自 Omar Sanseviero (@osanseviero) 的推文](https://x.com/osanseviero/status/1800607752038818260)：Google 的 RecurrentGemma 9B 发布了 🔥 ⚡️长序列处理速度超快：优秀的吞吐量+延迟 👀提供 Base 和 Instruct 微调版本 🏆质量与 Gemma 相当，看看下面的 y 轴 🤯 模型：https:...
- [Video LLaVA - 由 LanguageBind 提供的 Hugging Face Space](https://huggingface.co/spaces/LanguageBind/Video-LLaVA)：未找到描述
- [UniversityofGlasgow (格拉斯哥大学)](https://huggingface.co/UniversityofGlasgow)：未找到描述
- [Dolphin - 由 nroggendorff 提供的 Hugging Face Space](https://huggingface.co/spaces/nroggendorff/dolphin)：未找到描述
- [stabilityai/stable-diffusion-3-medium-diffusers · Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)：未找到描述
- [stabilityai/stable-diffusion-3-medium · Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-3-medium)：未找到描述
- [Hugging Face – 构建未来的 AI 社区。](https://huggingface.co/settings/tokens)：未找到描述

---

### **HuggingFace ▷ #**[**today-im-learning**](https://discord.com/channels/879548962464493619/898619964095860757/1250499854074118164) (1 条消息):

- **参加金融 LLM 训练直播活动**：一位成员邀请其他人参加一场专注于训练**专家级金融 LLM** 的直播活动。该活动由 Mark Kim-Huang 担任演讲者，可免费参加。[活动链接](https://events.dataphoenix.info/live)。

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1250311770573373471) (4 条消息):

- **拥有 1K 粉丝的老人让我们感到困惑**：一个拥有 1k 粉丝的老人让大家感到莫名其妙。*"我们就这样了 🤷‍♂️"* 反映了这种情绪。
- **Mistral 的微调魔法**：查看这篇关于在 Mistral 服务器上微调 AI 模型的指南。通过[这篇 Medium 文章](https://medium.com/ai-artistry/craft-your-ai-vision-fine-tuning-magic-with-mistral-on-mistral-server-6c9335232159)深入了解。
- **Distill.pub 对 ML/DL 资源仍具价值**：虽然 Distill.pub 不再活跃，但其文章对于理解 ML 和 DL 主题仍然非常有用。重点文章包括[理解图卷积](https://github.com/distillpub/post--understanding-gnns/issues?q=is%3Aissue+label%3Apeer-review)和[图神经网络（GNN）简明介绍](https://github.com/distillpub/post--gnn-intro/issues?q=is%3Aissue+label%3Apeer-review)。
- **考虑支持 Ciechanowski 的优质内容**：如需更多关于各种工程主题的互动式高质量内容，请访问 [Ciechanowski 的存档](https://ciechanow.ski/archives/)。如果你喜欢这些文章，可以在 [Patreon](https://www.patreon.com/ciechanowski) 上支持创作者。

**提到的链接**：

- [Distill — 关于机器学习的最新文章](https://distill.pub/)：关于机器学习的文章
- [Archives - Bartosz Ciechanowski](https://ciechanow.ski/archives/)：未找到描述

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1250174120524513482) (4 messages):

- **巨大的 Dalle 3 数据集发布**：一位成员分享了一个包含超过 100 万张由 Dalle 3 生成的高质量带字幕图像的数据集，强调了由于 Dalle 3 不可预测的审查制度而涉及的人类驱动的创造力。该数据集涵盖了艺术风格、风景、主题、节日和流行文化等多种主题。[Dalle3 1 Million+ High Quality Captions 数据集卡片](https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions)。
- **SimCLR 模型上线 Hub**：另一位成员转换并上传了 SimCLRv1 和 SimCLRv2 模型的 PyTorch 权重，这些权重此前在 Hub 上无法获取。[SimCLRv1 PyTorch 权重](https://huggingface.co/collections/SauravMaheshkar/simclrv1-pytorch-weights-6668b84a2bd3135cc3283fcd) 和 [SimCLRv2 PyTorch 权重](https://huggingface.co/collections/SauravMaheshkar/simclrv2-pytorch-weights-6668cd94d7c3f6fe7f3ac14b)。
- **Mixins 让模型管理更简单**：强调了 `huggingface_hub.PyTorchModelHubMixin` 在上传和下载功能方面的多功能性。提供了 [Mixins 文档](https://huggingface.co/docs/huggingface_hub/en/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin) 的链接。
- **Stable Diffusion 3 Medium 权重上线 Space**：一位成员为 **Stable Diffusion 3 Medium Weights** 创建了一个 Space，旨在实现模型的快速使用和部署。点击[此处](https://huggingface.co/spaces/Nick088/Stable-Diffusion-3-Medium)查看项目。

**提到的链接**：

- [ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions · Hugging Face 数据集](https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions)：未找到描述
- [Stable Diffusion 3 Medium Superpompt - Nick088 创建的 Hugging Face Space](https://huggingface.co/spaces/Nick088/Stable-Diffusion-3-Medium)：未找到描述
- [SimCLRv1 PyTorch Weights - SauravMaheshkar 的合集](https://huggingface.co/collections/SauravMaheshkar/simclrv1-pytorch-weights-6668b84a2bd3135cc3283fcd)：未找到描述
- [SimCLRv2 PyTorch Weights - SauravMaheshkar 的合集](https://huggingface.co/collections/SauravMaheshkar/simclrv2-pytorch-weights-6668cd94d7c3f6fe7f3ac14b)：未找到描述
- [Mixins & 序列化方法](https://huggingface.co/docs/huggingface_hub/en/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin)：未找到描述

---

### **HuggingFace ▷ #**[**core-announcements**](https://discord.com/channels/879548962464493619/1014557141132132392/1250473835309039749) (1 messages):

- **SD3 的 Diffusers 集成即将推出**：“嘿大家，`diffusers` 对 SD3 的集成将在几小时内上线。提前感谢大家的耐心等待。” 令人兴奋的集成公告！

---

### **HuggingFace ▷ #**[**computer-vision**](https://discord.com/channels/879548962464493619/922424143113232404/1250255554132185139) (3 messages):

- **Space 中的语义搜索受到好评**：一位用户表达了感谢，说道：“这太棒了……尤其是语义搜索功能。感谢创建这个 Space。”
- **征集计算机视觉项目创意**：另一位用户询问关于“计算机视觉领域极好的项目创意”的建议。
- **建议开展 CCTV 吸烟检测项目**：针对项目创意请求的一条回复建议创建一个“在 CCTV 监控中检测吸烟者”的系统，并包含一个功能：如果置信度大于 0.9，则将边界框显示为红色。

---

### **HuggingFace ▷ #**[**NLP**](https://discord.com/channels/879548962464493619/922424173916196955/1250298744587681822) (2 messages):

- **征集微调技术论文**：一位用户请求推荐在“深入研究微调” **phi2**、**llama2** 和 **mixtral2** 等模型之前需要阅读的论文。他们还询问了是否有任何“有趣的脚本”。
- **在 GitHub 上请求发布 Gemini Flash**：一位成员分享了一个 [gemma.cpp 的 GitHub issue](https://github.com/google/gemma.cpp/issues/221)，表达了希望看到 **Google Gemini Flash** 向开源社区发布的强烈愿望。该链接的 issue 是写给 Google AI 团队的，强调了社区对该项目的兴趣。

**提到的链接**：[题外话，请求开源 Google Gemini Flash · Issue #221 · google/gemma.cpp](https://github.com/google/gemma.cpp/issues/221)：亲爱的 Google AI 团队，我希望表达我看到 Google Gemini Flash 向开源社区发布的强烈兴趣。作为一名开发者和 AI 爱好者，我对……留下了深刻的印象。

---

### **HuggingFace ▷ #**[**diffusion-discussions**](https://discord.com/channels/879548962464493619/1009713274113245215/1250205986795552809) (8 条消息🔥):

- **推荐将 Simpletuner 用于 Diffusion 模型**：一位成员建议 **simpletuner** 主要专注于训练 Diffusion 模型，并提到 **diffusers** 的示例对于获取基础思路很有帮助，但并未完全优化。指出可能需要根据具体用例进行调整。
- **关于高通 AIMET 的讨论**：一位成员提到高通的 **AIMET** 是他们用过最糟糕的库，引发了简短的离题讨论。另一位成员表示在其他高通 SDK 上也有类似经历。
- **探索 AI 量化工具**：一位成员分享了在使用 **quanto** 和 **optimum** 进行量化和导出时的困扰，提到了错误和令人失望的结果。他们还尝试过设备端 AI 课程，但没有成功。
- **关于模型权重提取的查询**：一位成员询问是否有办法从给定模型中提取权重以进行聚合。在提供的消息中未见对此查询的回复。

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1250192667577684019) (111 条消息🔥🔥):

- **推荐 Ilya Sutskaver 的讲座**：Ilya Sutskaver 在 Simons Institute 有一场“精彩的讲座”，可在 YouTube 上找到，标题为 *An Observation on Generalization*。另一位成员建议听听 Neel Nanda 在 YouTube 视频 *Mechanistic Interpretability - NEEL NANDA (DeepMind)* 中关于记忆与泛化的看法。
- **讨论 AI 模型性能差异**：成员们对比了 Llama 3 8b instruct 与 GPT 3.5，指出 Llama 3 8b 在 Hugging Face 上提供免费 API。此外还讨论了 GPT-4o 编程能力的优劣，一些人观察到了性能问题。
- **关于 Enterprise 层的辩论**：针对 GPT Enterprise 层的优点和成本进行了广泛讨论，一些用户称赞其优点，如更大的 Context Window 和无缝的对话延续。有人指出 Enterprise 层提供更强大的功能并消除了限制，但一位用户误将 Teams 等同于 Enterprise。
- **关于从头开始构建模型的建议**：当被问及如何构建一个专注于单一领域的类似 GPT 的小型模型时，建议是微调现有模型（如 Llama3 8b）或使用开源替代方案，而不是从头开始。
- **医疗文档的 Embeddings**：一位成员询问用于医疗文档搜索的高质量开源 Embedding 模型，寻求针对句子和关键词搜索优化的建议。

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1250191528794652713) (9 条消息🔥):

- **自定义 GPT 创建咨询**：一位用户询问该频道是否适合“请求某人创建一个自定义 GPT”。另一位用户幽默地思考这里是否是展示 GPT 的好地方。
- **生成响应时出错**：一位成员持续遇到错误消息：“Something went wrong while generating the response...”，并寻求帮助解决。
- **上传 PHP 文件的问题**：尽管 [文件类型被列为支持](https://platform.openai.com/docs/assistants/tools/file-search/supported-files)，但在向 Assistants Playground 上传 PHP 文件时仍出现问题。用户还询问是否存在上传大小限制，随后表示已成功上传到存储中，但仍面临检索问题。

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1250380993668448297) (11 条消息🔥):

- **如何在 GPT-4 Assistants 中禁用引用**：一位用户寻求帮助，想知道如何在基于 20 多个 PDF 训练的 GPT-4 Assistant 中禁用并避免出现类似 [1][2] 的引用。他们提到仍希望检索信息，但不指明来源。
- **聊天机器人不了解 gizmo 工具功能**：当被问及是否尝试过询问 ChatGPT 时，一位用户澄清说 ChatGPT 并不了解 gizmo 工具的功能。
- **文档方面的困扰**：一位用户分享说，在 300 MB 的文本文件中提供 API 文档可能会让模型不堪重负。另一位用户建议创建一个 swagger 文件以便更好地管理。
- **简历制作求助请求**：一位用户表示需要协助制作简历。

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1250380993668448297) (11 条消息🔥):

- **停用 GPT4 引用，但仍检索数据**：一位成员询问如何在一个基于 20 多个 PDF 训练的 GPT4 Assistant 中停用并避免出现类似 `[1]`、`[2]` 的引用。他们仍希望 Assistant 检索信息，但不在乎具体的来源。
- **GPT 不了解 gizmo 工具的功能**：另一位成员建议就某个问题询问 ChatGPT，但原提问者指出 ChatGPT 并不了解 gizmo 工具的功能，因此在这种情况下没有帮助。
- **300MB txt 文件文档问题**：一位用户指出，以文本文件形式提供文档非常困难，因为文件太大（300MB）。有人建议编写 Swagger 文件代替，但该用户澄清这不仅仅是一个 API，而是整个文档。

---

### **Cohere ▷ #**[**general**](https://discord.com/channels/954421988141711382/954421988783444043/1250245729767788624) (126 条消息🔥🔥):

- **对 Qualcomm 的 AIMET 库感到失望**：一位成员表达了沮丧，称 *"对我来说，这是我目前见过的最糟糕的库。"*
- **关于 PaidTabs 和 Cohere AI 集成的猜测**：成员们讨论了 PaidTabs 是否正在使用 Cohere 进行 AI 消息生成，并引用了 [PaidTabs 6 月 24 日的更新日志](https://blog.paidtabs.com/paidtabs-june-24-changelog/)。对话澄清道 *"据我所知，Cohere 还没有进行任何与音频相关的开发。至少目前还没有。"*
- **频道中的音乐爱好者**：几位成员讨论了他们的音乐兴趣和活动，有些人尝试通过软件制作音乐，另一些人分享了个人经验和设备。一位成员开玩笑说，*"这里有这么多音乐家，我觉得我们可以组个乐队了。"*
- **摇杆和飞行模拟设置**：关于一位成员为太空飞行模拟配置的复杂摇杆设置的广泛讨论，其中包括 [VPC Constellation ALPHA Prime 摇杆](https://virpil-controls.us.com/vpc-constellation-alpha-prime-r.html)。大家对设置成本进行了辩论，有人注意到，*"摇杆是钻石做的吗？摇杆要 1500 美元？"*
- **请求闲聊频道**：随着对话变得越来越偏离主题，一位成员开玩笑地建议为这类讨论创建一个单独的频道，以避免刷屏主频道。*"我们需要一个闲聊频道……我一直在刷屏。"*

**提到的链接**：

- [PaidTabs New Tuner, Mobile App and More Features!](https://blog.paidtabs.com/paidtabs-june-24-changelog/)：智能调音器。我们很高兴推出期待已久的由 AI 驱动的有弦乐器调音器，它可以检测你正在弹奏的琴弦及其音调，帮助你弹奏出完美的音符！在此查看。
- [VPC Constellation ALPHA Prime [R]](https://virpil-controls.us.com/vpc-constellation-alpha-prime-r.html)：★ * 右手变体 * ★ 金属翻转触发器和制动杆 ★ 高级...
- [Udio | AI Music Generator - Official Website](https://www.udio.com/)：发现、创作并与世界分享音乐。使用最新技术在几秒钟内创作 AI 音乐。
- [Star Citizen - Roberts Space Industries | Follow the development of Star Citizen and Squadron 42](https://robertsspaceindustries.com/star-citizen)：Roberts Space Industries 是所有关于《星际公民》（Star Citizen）和《 42 中队》（Squadron 42）新闻的官方网站。它还托管了游戏道具和周边商品的在线商店，以及所有社区工具...

---

### **Cohere ▷ #**[**project-sharing**](https://discord.com/channels/954421988141711382/1218409701339828245/1250476945821532195) (1 条消息):

- **Playgrounds 为 Rust 发布 RIG：** Playgrounds 的创始人兼 CEO 宣布发布名为 **RIG** 的开源库，用于在 **Rust 中构建 LLM 驱动的应用程序**。该库强调模块化和人体工程学，并且开箱即用地与 Cohere 完全集成。查看 GitHub 上的 [RIG 仓库](https://github.com/0xPlaygrounds/rig) 以获取更多详细信息和示例。

**提到的链接**：[GitHub - 0xPlaygrounds/rig: A library for developing LLM-powered Rust applications.](https://github.com/0xPlaygrounds/rig)：一个用于开发 LLM 驱动的 Rust 应用程序的库。- 0xPlaygrounds/rig

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1250198644343312424) (22 条消息🔥):

- **新成员寻求介绍**：一位新成员询问了社区的宗旨以及如何做出贡献。其他成员建议阅读置顶消息，并查看 `<#732688974337933322>` 和 `<#1102787157866852402>` 等特定频道以获取资源和了解正在进行的项目。
- **预测微塑料浓度**：一位用户解释了他们的项目，该项目结合了藻类浓度和水流速度等海洋矢量因子，以预测微塑料的浓度和运动。
- **自回归模型在图像生成上超越 Diffusion**：有人分享了 [LlamaGen 的仓库](https://github.com/FoundationVision/LlamaGen)链接，展示了像 Llama 这样的自回归模型可以达到 SOTA（state-of-the-art）的图像生成性能。该仓库包含了关于 LlamaGen（一个新的图像生成模型系列）的详细信息和教程。
- **关于多轮对话 DPO 方法的讨论**：多位成员讨论了针对多轮对话的 DPO (Deterministic Policy Optimization) 研究匮乏的问题。一些人建议查看 Ultrafeedback 数据集，并考虑针对分层结构的 MCTS-DPO 方法。
- **Llama3 模型生成内容**：一位用户分享了使用 Llama3 70B 模型的经验，该模型在空提示词下生成了多样化的输出，如 AI2 的 ARC 格式评估、维基文本和代码。另一位成员指出，其输出有 30% 的概率以 "Question" 格式开头。

**提到的链接**：[FoundationVision/LlamaGen · Hugging Face](https://huggingface.co/FoundationVision/LlamaGen)：未找到描述

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1250166991826845867) (94 条消息🔥🔥):

- **为 LLM 训练寻找代码数据集遇到困难**：一位用户正在寻找小于 10B tokens 的小型代码数据集，但指出很难找到。他们提到可能使用 GPT-2 tokenizer，并讨论了排除 GitHub 数据后分块提供的 200B tokens 的可用性。
- **Transformer 中的精度限制**：一场关于 Transformer 中由于有限精度导致限制的讨论展开，引用了一篇 [arXiv 论文](https://arxiv.org/abs/2210.02671)。精度问题在 Attention 机制中被重点强调，特别是 softmax 操作。
- **DeltaNet 与 RWKV6 Finch 架构的相似性**：用户辩论了 DeltaNet 和 RWKV6 Finch 架构之间的相似性，重点关注“decay”（衰减）机制和 token shift 机制。结论是 DeltaNet 的衰减取决于状态（state）而非输入（input）。
- **研究论文的新颖性与相对性**：多位用户批评了关于 Griffin 和 Samba 等架构的新论文，将其与现有工作进行比较，并对其新颖性表示怀疑。他们讨论了 Samba 在建模长上下文方面的性能，以及其结合了 State Space Models 和 Sliding Window Attention 的架构组合。
- **探索长上下文检索的基准测试**：对话涉及 Samba 在长序列上高效检索上下文的能力，用户对此持怀疑态度，认为基准测试任务可能不够稳健。一位用户指出位置编码（Positional Encoding）在长上下文任务性能中可能发挥的作用。

**提到的链接**：

- [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524)：虽然扩散模型在生成高质量图像方面表现出色，但此前的研究报告称，在语言建模中，扩散方法与自回归（AR）方法之间存在显著的性能差距。在这项工作中，我们……
- [Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling](https://arxiv.org/abs/2406.07522)：高效建模具有无限上下文长度的序列一直是一个长期存在的问题。过去的工作要么受限于二次计算复杂度，要么受限于有限的外推能力……
- [LLM Dataset Inference: Did you train on my dataset?](https://arxiv.org/abs/2406.06443)：大型语言模型（LLMs）在现实世界中的激增，伴随着针对公司使用互联网未经授权数据训练模型的版权案件增加。最近的工作……
- [A Logic for Expressing Log-Precision Transformers](https://arxiv.org/abs/2210.02671)：解释基于 Transformer 的语言模型推理能力的一种方法是描述它们可以对某些输入文本解析的逻辑规则类型。最近，Chiang 等人 (2023) 表明……
- [Improve Mathematical Reasoning in Language Models by Automated Process Supervision](https://arxiv.org/abs/2406.06592)：复杂的多步推理任务（如解决数学问题或生成代码）即使对于最先进的大型语言模型（LLMs）来说仍然是一个重大障碍。验证 LLM 输出……
- [Is a small transformer model able to effectively handle any input length provided it is fine-tuned on it?](https://ai.stackexchange.com/q/45949/68078)：假设我们有一个可以执行摘要任务的 Transformer LLM。我知道 Transformer 从技术上讲可以处理任何输入长度（假设我们不使用学习到的位置嵌入），因为……
- [Tweet from Rivers Have Wings (@RiversHaveWings)](https://x.com/RiversHaveWings/status/1478093658716966912)：你可以将类似于 classifier-free guidance 的技巧应用于自回归 Transformer，从而从合成的“超条件”分布中采样。我训练了一个 CIFAR-10 类别条件的 Image...
- [Cache Me if You Can: Accelerating Diffusion Models through Block Caching](https://arxiv.org/abs/2312.03209)：扩散模型最近因其生成逼真图像的能力而彻底改变了图像合成领域。然而，扩散模型的主要缺点之一是其……
- [GitHub - nasheydari/HypOp: Hypergraph Neural Network-Based Combinatorial Optimization](https://github.com/nasheydari/HypOp)：基于超图神经网络的组合优化 - nasheydari/HypOp
- [GitHub - microsoft/Samba](https://github.com/microsoft/Samba/)：通过创建一个账户为 microsoft/Samba 的开发做出贡献。

---

### **Eleuther ▷ #**[**scaling-laws**](https://discord.com/channels/729741769192767510/785968841301426216/1250185388937449583) (2 messages):

- **Transformers 在组合泛化方面表现出色**：一位用户分享了一篇 [arXiv 论文](https://arxiv.org/abs/2406.05816)，讨论了 Transformers 如何通过在其多头注意力机制（multi-head attention mechanism）中使用低维潜码来“泛化到新的问题实例”。该方法提高了组合泛化能力，特别是在瑞文标准推理测验（Raven Progressive Matrices）人类智力测试的符号版本等任务上。

**提到的链接**：[Attention as a Hypernetwork](https://arxiv.org/abs/2406.05816)：Transformers 在某些情况下可以泛化到新的问题实例，这些实例的组成部分可能在训练期间遇到过，但其组合方式未曾出现。什么样的机制...

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1250193482962833448) (3 messages):

- **污染检测建议：** 一位成员分享了他们最近关于[污染检测](https://arxiv.org/abs/2405.16281)的工作，该工作通过比较模型在两个基准测试上的表现来检测“可疑”分数。他们提议创建一个 PR，将此功能包含在评估框架（evaluation harness）中。
- **关键的单行修复：** 一位成员敦促其他人审查针对 `anthropic_llms.py` 中一个 bug 的[单行修复](https://github.com/EleutherAI/lm-evaluation-harness/pull/1848)，该 bug 导致 `self.max_tokens` 在构造函数中未被正确设置。他们强调了仅仅因为漏掉一个 "s" 而导致问题是多么令人沮丧。
- **立即代码审查：** 另一位成员迅速做出回应，承诺很快会审查提议的修复。

**提到的链接**：[Fix self.max_tokens in anthropic_llms.py by lozhn · Pull Request #1848 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/pull/1848)：修复了 self.max_tokens 未被设置的 bug。AnthropicChatLM 在各处都使用 self.max_tokens，但在构造函数中设置的是 self.max_token。它在较短的响应中不会引发问题，但我发现...

---

### **Eleuther ▷ #**[**multimodal-general**](https://discord.com/channels/729741769192767510/795089627089862656/1250174056477491361) (3 messages):

- **寻找预训练的 LTIP 替代方案**：一位成员询问是否有 DeepMind 使用的 LTIP 数据集的开源替代方案，并表示对与 LTIP 的 3.12 亿图文对规模相当的数据集感兴趣。他们指出 LTIP 的数据表可以在 [Flamingo 论文](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/tackling-multiple-tasks-with-a-single-visual-language-model/flamingo.pdf)中找到。
- **推荐 DataComp 作为 LTIP 替代方案**：作为回应，另一位成员建议将 DataComp 的 1B 子集作为可行的替代方案，并称其对于 CLIP 模型来说优于 LAION。他们分享了 [DataComp 数据集](https://huggingface.co/datasets/mlfoundations/datacomp_1b)的链接。

**提到的链接**：[mlfoundations/datacomp_1b · Datasets at Hugging Face](https://huggingface.co/datasets/mlfoundations/datacomp_1b)：未找到描述

### **LM Studio ▷ #**[**💬-general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1250180925422043177) (85 条消息🔥🔥):

- **Python 脚本集成咨询**：一位用户表示有兴趣将 Python 脚本集成到 LM Studio 中，以便聊天机器人在闲置超过一分钟时进行角色扮演并发送消息。他们被引导按照 [LM Studio 文档](https://lmstudio.ai/docs) 设置推理服务器（inference server）。
- **RecurrentGemma 模型请求**：有用户请求添加 **RecurrentGemma-9B** 模型，并提供了其 [Hugging Face 页面](https://huggingface.co/google/recurrentgemma-9b) 的链接以获取更多详情和文档。
- **LM Studio 0.2.24 的问题**：一位用户报告了 LM Studio 0.2.24 版本中的严重 Bug，包括 Token 计数器、模型卸载与重载以及 GPU 优化方面的问题。他们表达了不满，指出 LM Studio 突然感觉非常像“Alpha”测试版。
- **GPU 与模型推理讨论**：讨论涉及模型推理的 GPU 比较，有人建议选择 **RTX 3060** 而非 **Tesla P40**，以获得更好的性能和能效。另一位用户目标是将处理时间从 9 小时缩短到 2 小时，考虑将其硬件从 Radeon 6900XT 升级为 24GB 的 Nvidia 显卡。
- **使用聊天机器人生成 PowerPoint**：一位用户询问关于使用 OpenAI Assistant API 创建用于生成 PowerPoint 演示文稿的聊天机器人，寻找能够基于之前的演示文稿自动执行该过程的工具。

**提到的链接**：

- [Documentation | LM Studio](https://lmstudio.ai/docs)：技术参考
- [How to Finetune phi-3 on MacBook Pro](https://huggingface.co/blog/abhishek/phi3-finetune-macbook)：未找到描述
- [google/recurrentgemma-9b · Hugging Face](https://huggingface.co/google/recurrentgemma-9b)：未找到描述
- [All Large Language Models](https://llm.extractum.io/list/)：大型和小型语言模型（开源 LLM 和 SLM）的精选列表。包含具有动态排序和过滤功能的所有大型语言模型。
- [Nitral-AI/Hathor-L3-8B-v.02 · Hugging Face](https://huggingface.co/Nitral-AI/Hathor-L3-8B-v.02)：未找到描述
- [Nitral-AI/Hathor-L3-8B-v.02 · Great Uncensored Model](https://huggingface.co/Nitral-AI/Hathor-L3-8B-v.02/discussions/1)：未找到描述
- [GitHub - jakobdylanc/discord-llm-chatbot: llmcord.py • Talk to LLMs with your friends!](https://github.com/jakobdylanc/discord-llm-chatbot)：llmcord.py • 与你的朋友一起与 LLM 聊天！通过在 GitHub 上创建账号为 jakobdylanc/discord-llm-chatbot 的开发做出贡献。

---

### **LM Studio ▷ #**[**🤖-models-discussion-chat**](https://discord.com/channels/1110598183144399058/1111649100518133842/1250311559025004604) (28 条消息🔥):

- **为 RTX 2070 选择最快的模型**：一位用户询问在拥有 64GB RAM 和 **8GB VRAM** 的 RTX 2070 上运行的最快模型。讨论缩小到几个模型，包括 **Mistral7b**、**Codestral** 和 **StarCoder2**，并考虑了上下文窗口（context window）大小以及将所有内容保留在 GPU 上的需求。
- **编程模型推荐**：对于预算充足的用户推荐使用 **Codestral**，否则建议使用经过编程微调的 Mistral 或 Llama 模型。**StarCoder2** 和 **StarChat** 被强调为性能均衡的选择，**CodeQwen1.5** 也被提及为一个不错的选择。
- **关于启用训练模式的咨询**：一位用户询问是否可以为个人模型分叉（fork）启用训练模式，以便在对话中记住事情。回复指出 LM Studio 不支持此功能，但推荐使用 **Unsloth** 和 **Autotrain** 来对现有的 safetensor 模型进行微调（fine-tuning）。
- **使用 VectorDB 实现持久化存储**：提到了使用 **vectordb 插件/扩展** 进行持久化存储，以便在对话中记住特定事项。用户考虑使用编程方法来实现其所需的功能，即在无需手动输入指令的情况下实现一致且具有记忆能力的对话。
- **关于模型适配的澄清**：围绕对不同格式（特别是 **gguf**）的模型进行持续训练或权重调整的可能性进行了讨论。用户表示有兴趣人工调整模型的响应权重，以确保在对话情境中获得特定的输出结果。

---

### **LM Studio ▷ #**[**📝-prompts-discussion-chat**](https://discord.com/channels/1110598183144399058/1120489168687087708/1250511503120142459) (3 条消息):

- **将 GIT 作为 Sys Prompt 补充项进行实现**：一位成员分享了将 GIT 集成到 Sys Prompt 中的进展，表示“经过一天左右的时间，已经能实现基础功能”。他们还在思考这种努力是否在浪费时间。
- **生成可靠性担忧**：尽管由于其生成性质，GIT 的实现并非完全可靠，但该成员指出它“运行得相当不错”。
- **幽默的产品命名**：在一条轻松的评论中，该成员建议将该实现命名为“Timemachine”，以致敬 Apple 的品牌命名，并幽默地定价为“$9 ? xD”。

---

### **LM Studio ▷ #**[**🎛-hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1250254102940881002) (4 条消息):

- **P40 价格风波**：成员们讨论了 **P40 显卡** 波动不定的价格，其中一人提到在海外以 640 美元购买了一对（总计 48GB）。另一人链接了一个 **eBay 列表**，显示目前每张卡的价格在 **150-170 美元** 左右。
- **澳大利亚的 GPU 供应与成本**：一位成员分享了各种 GPU 在 **澳大利亚当地的价格** —— **RTX 3060** 为 450 澳元，**RTX 4060 TI** 为 800 澳元，**RTX 4090** 为 3200 澳元，并指出了库存水平和近期价格变化的差异。免责声明包括功耗、PSU 考量、散热挑战以及其他权衡。
- **一致的卖家列表**：有人注意到同一个 eBay 卖家在不同的国际平台上提供 **P40 显卡**，价格随时间略有波动。
- **在美国购买 P40**：另一位成员表示，他们最近购买了在美国境内交付的 P40 显卡，**单价为 160-170 美元**，并强调了这与从中国发货的显卡价格之间的差异。

**提到的链接**: [NVIDIA Tesla P40 24GB DDR5 GPU Accelerator Card Dual PCI-E 3.0 x16 - PERFECT! 190017118253 | eBay](https://www.ebay.com.au/itm/196435922621): 未找到描述

---

### **LM Studio ▷ #**[**amd-rocm-tech-preview**](https://discord.com/channels/1110598183144399058/1195858490338594866/1250230075350188072) (3 条消息):

- **默认启用现代 UI**：一位成员指出，现在应该有一个用户可以启用的“现代 UI”，并表示它“看起来非常不同”。不过，他们还没有充分使用它来做出完整评估。
- **新 UI 外观预览**：分享了一张展示现代 UI 变体的图片，你可以[在此查看图片](https://cdn.discordapp.com/attachments/1130536562422186044/1245478003044257932/image.png?ex=666a08c7&is=6668b747&hm=62024d42535148d6a9a7f23860dc0c011c6a2a48dba4759c91e04bb0f2fbe03f)。一位成员给出了积极回应，称：“现在这看起来相当不错。”

---

### **LlamaIndex ▷ #**[**announcements**](https://discord.com/channels/1059199217496772688/1073670729054294197/1250214985762603079) (1 条消息):

- **企业级场景下的 RAG 面临挑战**：一份公告邀请那些在企业环境中参与 **构建/生产化 RAG**（Retrieval-Augmented Generation）的人员查看 Quentin 的消息。Quentin 和 Shreya Shankar 正在进行访谈以了解所面临的挑战，鼓励感兴趣的各方通过[提供的 Discord 链接](https://discord.com/channels/1059199217496772688/1059201661417037995/1249840705627488408)进行联系。

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1250179545844285602) (2 条消息):

- **带有实体去重的 Graph RAG**：由 @tb_tomaz 发布的一份详细指南展示了如何在 LlamaIndex 中使用实体去重和自定义检索方法来“增强” Graph RAG。在此查看[完整教程](https://t.co/fiFwDQS6WT)。
- **Excel 格式的 RAG 挑战**：讨论了构建能有效处理 Excel 文件的 RAG 所面临的挑战，强调了在格式良好的空间网格中布局内容的重要性。更多详情请见此处的 [推文](https://t.co/vbdk8Yuw2t)。

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1250169658670190653) (71 messages🔥🔥):

- **使用 Vector Index 开发 Chat Engine**：一位成员询问如何利用从纯文本片段创建的 Vector Index 上下文来开发 Chat Engine。另一位成员建议使用 `Document` 类手动创建 documents/nodes。
- **S3DirectoryReader 问题**：一位成员报告在使用 S3DirectoryReader 时获取到的是加密文件，但指出 Langchain S3 Reader 工作正常。他们确认启用了 SSE-S3 加密，但没有自动解密。
- **MetadataFilter 功能异常**：一位成员分享了在 Redis index 设置中 `MetadataFilter` 未返回预期结果的问题。另一位成员要求提供 index 构建的详细信息以便诊断问题。
- **Vector DB 中的 Markdown 格式问题**：一位用户遇到了存储的文档和聊天回复中残留 Markdown 格式的问题。另一位成员建议使用 BeautifulSoup 作为预处理步骤，从 Markdown 文档中提取纯文本。
- **自定义 Chat Engine Prompt**：提供了关于使用 `from_defaults` 方法为 `CondensePlusContextChatEngine` 自定义 Prompt 的指导。后续澄清说明用户不需要手动处理像 EOT 这样的特殊 token。

**提到的链接**：

- [Python : 如何将 markdown 格式的文本转换为纯文本](https://stackoverflow.com/a/761847/1819550)：我需要将 Markdown 文本转换为纯文本格式，以便在我的网站上显示摘要。我需要 Python 代码。
- [llama_index/llama-index-core/llama_index/core/vector_stores/simple.py at 045582caf3564f5a549266c07d291f30b997425d · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/045582caf3564f5a549266c07d291f30b997425d/llama-index-core/llama_index/core/vector_stores/simple.py#L368)：LlamaIndex 是为您 LLM 应用程序提供的数据框架 - run-llama/llama_index
- [Condense plus context - LlamaIndex](https://docs.llamaindex.ai/en/latest/api_reference/chat_engines/condense_plus_context/#llama_index.core.chat_engine.CondensePlusContextChatEngine>).)：未找到描述
- [Function Calling AWS Bedrock Converse Agent - LlamaIndex](https://docs.llamaindex.ai/en/latest/examples/agent/bedrock_converse_agent/?h=converse)：未找到描述
- [Query Pipeline for Advanced Text-to-SQL - LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/)：未找到描述
- [Building Performant RAG Applications for Production - LlamaIndex](https://docs.llamaindex.ai/en/latest/optimizing/production_rag/#motivation_1>),)：未找到描述
- [Chat Engine - Best Mode - LlamaIndex](https://docs.llamaindex.ai/en/latest/examples/chat_engine/chat_engine_best/#chat-engine-best-mode>))：未找到描述
- [Building Performant RAG Applications for Production - LlamaIndex](https://docs.llamaindex.ai/en/latest/optimizing/production_rag/#motivation_1>).)：未找到描述
- [Building a Multi-PDF Agent using Query Pipelines and HyDE - LlamaIndex](https://docs.llamaindex.ai/en/latest/examples/agent/agent_runner/agent_around_query_pipeline_with_HyDE_for_PDFs/#what-is-react-agent>))：未找到描述
- [React - LlamaIndex](https://docs.llamaindex.ai/en/latest/api_reference/agent/react/#llama_index.core.agent.react.ReActAgent>).)：未找到描述

---

### **LlamaIndex ▷ #**[**ai-discussion**](https://discord.com/channels/1059199217496772688/1100478495295017063/1250438272329580645) (1 messages):

- **Query engine 在上下文感知方面存在困难**：一位成员表示 text-to-SQL query engine 难以识别特定的上下文线索，例如查询项的性质（例如，“Q”是产品名称还是其他内容）。他们强调问题在于由于数据值的不确定性，engine 无法识别正确的 SQL 查询，并寻求解决此问题的建议。

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1250198856592003153) (41 条消息🔥):

- **ARC Prize for AGI 宣布**：由 Mike Knoop 和 François Chollet 发起的一项奖金超过 **1,000,000 美元的竞赛**已揭晓，旨在解决 ARC-AGI 基准测试并开源解决方案。Chollet 的 **ARC-AGI** 通过高效获取新技能的能力来衡量通用智能，这一直是 AI 的难题，但对人类来说却很容易。
- **对播客访谈反应不一**：成员们对最近的一场访谈感受复杂，特别是批评主持人 Dwarkesh 没有精准回应受访者 François 的观点。尽管有些沮丧，但像 chygao 这样的听众仍然很喜欢这次采访，并承认其中提出的质疑是合理的。
- **行业对 ARC-AGI 缺乏了解**：令人惊讶的是，**业内同行竟然不知道 ARC-AGI 基准测试**，一些成员认为这是基础性的。当播客结尾提到这一点时，Drj.bet 感到特别吃惊。
- **TechCrunch 评论**：Kyle L. Wiggers 在 TechCrunch 上发表的一篇文章因在 AI 话题上缺乏深度而受到批评。Natolambert 幽默地评论了一些科技写作的重复性，嘲讽了 Wiggers 之前关于 DBRX 的文章。
- **关于人类智力与遗传学的讨论**：播客还触及了人类智力对遗传谱系的依赖，Dwarkesh 将其比作模型初始化（initialization）。这一观点遭到了 vj256 等成员的反驳，他引用了许多来自不同背景的成功人士的反例。

**提到的链接**：

- [ARC Prize](https://arcprize.org/)：ARC Prize 是一项奖金超过 1,000,000 美元的非营利性公开竞赛，旨在击败并开源 ARC-AGI 基准测试的解决方案。
- [Kyle Wiggers (@Kyle_L_Wiggers) 的推文](https://x.com/Kyle_L_Wiggers/status/1800945959112749155)：本周 AI 动态：Apple 不会透露“香肠是如何制成的” https://ift.tt/Q2GmDOC

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-drama**](https://discord.com/channels/1179127597926469703/1181746144821387334/1250507434490069122) (2 条消息):

- **微软通过特斯拉作为诱饵获取了 OpenAI 股份**：一条[推文](https://fxtwitter.com/nicoleperlroth/status/1800946061613416659?s=46)透露，微软通过将 OpenAI 作为吸引特斯拉加入 Azure 的杠杆，获得了其 49% 的股份。微软最初试图说服特斯拉使用 Azure，但 Elon Musk 对 OpenAI 的关注促成了这一意想不到的合伙协议。

**提到的链接**：[Nicole Perlroth (@nicoleperlroth) 的推文](https://fxtwitter.com/nicoleperlroth/status/1800946061613416659?s=46)：有没有想过微软是如何进入 OpenAI 的？我有一个精彩的故事要告诉你…… @elonmusk 对 Apple 怀恨在心，因为他觉得 iCar 项目对 Tesla 构成了威胁。与此同时，微软一直试图让……

---

### **Interconnects (Nathan Lambert) ▷ #**[**random**](https://discord.com/channels/1179127597926469703/1183121795247779910/1250431095493759057) (7 条消息):

- **June-Chatbot 遭遇 NVDA API 错误**：一位成员注意到 `june-chatbot` 已进入竞技场，但有时会“抛出 NVDA API 错误”。这凸显了该机器人潜在的稳定性问题。
- **SnailBot 获得了一个有趣的绰号**：另一位成员幽默地评论道：“SnailBot 真是名副其实”，暗示该机器人的性能明显迟缓。
- **对 LMSYS 的挫败感上升**：Nathan Lambert 惊呼：“lmsys 快停下！”，可能指的是 LMSYS 持续存在的问题或令人烦恼的事情。
- **Dream Machine 的新文本转视频模型令人印象深刻**：来自 [Luma Labs](https://lumalabs.ai/dream-machine) 的 Dream Machine 可以快速从文本和图像创建高质量、逼真的视频。它被描述为一个可扩展且高效的 Transformer 模型，能够生成物理上准确、连贯且动态的视频。
- **AI 协同决定现象引起关注**：Nathan Lambert 评论了 AI 领域同时出现的各种进展，认为“AI 似乎是协同决定的（co-determined），这很有趣”，并指出这些发展“总是在同一时间出现”。

**提到的链接**：[Luma Dream Machine](https://lumalabs.ai/dream-machine)：Dream Machine 是来自 Luma AI 的一款 AI 模型，可以从文本和图像快速生成高质量、逼真的视频。

### **Interconnects (Nathan Lambert) ▷ #**[**rl**](https://discord.com/channels/1179127597926469703/1208183216843005962/1250203449400885421) (10 messages🔥):

- **Apple 创新混合数据策略**：在讨论最近的一篇 [Apple 博客文章](https://apple.com/blog)时，一位成员强调了 Apple 的混合数据策略，其中包括人工标注数据和合成数据。Apple 还开发了新型算法，如拒绝采样微调算法以及结合镜像下降策略优化的 RLHF。
- **PPO 伪代码差异讨论**：由 [@jsuarez5341 发布的一条 Twitter 线程](https://x.com/jsuarez5341/status/1800677493495783713)引起了关注，该线程指出了 PPO 实现中的差异，特别是关于梯度累积（gradient accumulation）的部分。成员们讨论了伪代码虽然暗示了这一点，但许多流行的实现都跳过了它，这可能会影响大 Batch Size 下的稳定性。
- **Tulu 2.5 稳定性见解**：针对 PPO 的讨论，一位成员分享了 Tulu 2.5 在每个 minibatch 步骤中都进行更新，并指出更改此设置除了速度变化外，并未显著影响性能。他们得出结论，虽然技术细节很重要，但在某些场景下实际差异可能微乎其微。

**提及的链接**：

- [来自 Joseph Suarez (e/🐡) (@jsuarez5341) 的推文](https://x.com/jsuarez5341/status/1800677493495783713)：我认为我发现了 PPO 中的一个差异。伪代码暗示了在 minibatch 上的梯度累积。OpenAI 的实现（各大库参考的对象）并没有进行累积。这很重要 🧵
- [来自 Joseph Suarez (e/🐡) (@jsuarez5341) 的推文](https://x.com/jsuarez5341/status/1800919908613849481)：更大的 Batch Size = 更稳定的训练，对吗？错了！这会导致 PPO 发散，因为下面的伪代码与实际实现之间存在差异，实际实现跳过了梯度累积。快速 ...

---

### **Interconnects (Nathan Lambert) ▷ #**[**posts**](https://discord.com/channels/1179127597926469703/1228051082631188530/1250433618287792209) (8 messages🔥):

- **Genmoji 凭借对表情符号的热爱推动 OS 采用**：讨论涉及了公众对表情符号的热爱，尤其是 **genmoji**，如何推动操作系统的采用。一位用户提到，这个功能甚至让新手机获得了“家属认可（wife-approved）”。
- **Nathan 深入 RL 实践**：Nathan Lambert 提到他这周很忙，但发现这是**很好的实践**，并喜欢看专业人士如何讨论这些话题。另一位用户对他**钻研 RL** 表示赞赏，表明了共同的兴趣。
- **提及 Cohere 的 Leave-One-Out RL**：Nathan Lambert 遗憾在详细阐述 RL 概念时漏掉了 **Cohere 的 leave one out RL** 链接。他表示 **KL 方向** 对他来说特别酷。

如需更多详情，请随时询问！

---

### **Nous Research AI ▷ #**[**off-topic**](https://discord.com/channels/1053877538025386074/1109649177689980928/1250512433819287603) (1 messages):

- **Manifold Research 寻求合作者**：来自 Manifold Research 的 Sidh 宣布，他们正在寻找对用于多模态和控制任务的 Transformer 感兴趣的研究合作者。他们的目标是构建一个大规模开源“通用型（Generalist）”模型，类似于 GATO 架构。
- **通过各种渠道参与 Manifold**：他们提供了几种参与方式：加入他们的 [Discord](https://discord.com/invite/a8uDbxzEbM?ref=manifoldrg.com)，在他们的 [GitHub](https://github.com/ManifoldRG?ref=manifoldrg.com) 上贡献 Issue，以及查看 [开源研究团队期望](https://docs.google.com/document/d/e/2PACX-1vQgq32ChlP_e26mRPgfC31lZJCcAHAgbJ_Tn1nfzq8pfysoPAUqAWnel87Qc26h2Q/pub?ref=manifoldrg.com) 以进行更深入的参与。

**提及的链接**：[机会](https://www.manifoldrg.com/opportunities/)：有几种方式可以参与我们的工作：1. 加入我们的 Discord 并参加活动和讨论，无论是否与项目相关。2. 异步贡献我们 GitHub 上的 Issue。...

---

### **Nous Research AI ▷ #**[**interesting-links**](https://discord.com/channels/1053877538025386074/1132352574750728192/1250180123815186553) (2 条消息):

- **GendelveChat 模拟智能工厂**：成员们讨论了 [GendelveChat](https://x.com/gendelvechat/status/1800580692046405784?s=46&t=eY--9rPoOkHV-u9kocxCMA) 利用 @websim_ai 为各种行业和工作流模拟群聊 UX。展示的一个例子是用于缺陷检测的智能工厂工作流，并提出“或许结合人类见解的群聊模型效果更好？”
- **StabilityAI 宣布 Stable Diffusion 3 Medium 开放权重**：[StabilityAI](https://x.com/StabilityAI/status/1800875914299048404) 公布了其最新的文本生成图像 AI 模型 Stable Diffusion 3 Medium 的开放权重。此次发布被视为生成式 AI 的重大进展，旨在继续履行其技术民主化的使命。

**提及的链接**：

- [来自 Stability AI (@StabilityAI) 的推文](https://x.com/StabilityAI/status/1800875914299048404)：今天，我们激动地宣布 Stable Diffusion 3 Medium 的开放权重，这是我们 Stable Diffusion 3 系列中最新且最先进的文本生成图像 AI 模型！这一新版本代表了...
- [来自 Gendelve (@GendelveChat) 的推文](https://x.com/gendelvechat/status/1800580692046405784?s=46&t=eY--9rPoOkHV-u9kocxCMA)：@websim_ai 我使用 @websim_ai 构建的 YC Demo。你可以针对任何行业、工作流或用例，它将模拟一个为人类和 AI 团队构建的新群聊 UX。例如：智能工厂工作流模拟...

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1250166170745569390) (34 条消息🔥):

- **Lemmyle 通过编辑斯大林卡片进行恶搞**：关于修改 `NousResearch/CharacterCodex` 中约瑟夫·斯大林角色卡的一个明显玩笑引发了反应。“斯大林是有史以来最伟大的领导人之一”触发了关于历史准确性和恶搞的简短讨论。
- **Azure2089 谈苹果的 AI**：分享了苹果 AI 系统架构的详细概述 [Introducing Apple Foundation Models](https://machinelearning.apple.com/research/introducing-apple-foundation-models)。讨论重点在于苹果的 3B 模型、其压缩技术、用于速度优化的 speculative decoding（投机采样），以及缺乏关于 GPT-4O 集成的细节。
- **Stable Diffusion 3 Medium 发布公告**：宣布发布 [Stable Diffusion 3 Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium)，具有改进的图像质量和性能。包含了指向[研究论文](https://stability.ai/news/stable-diffusion-3-research-paper)的链接以获取更多技术细节。
- **_kquant 分享数据流水线构想**：建议建立一个能够为 Character Codex 项目生成多轮 ShareGPT 数据的数据流水线。提到使用 **Cat-70B** 配合系统提示词以及 **oobabooga** 等工具进行实现。
- **Teknium 强调 Stable Diffusion 3 的使用**：分享了来自 [minimaxir](https://x.com/minimaxir/status/1800921802765717754) 的推文，展示了 Stable Diffusion 3 令人印象深刻的测试，突出了该模型的高性能。

**提及的链接**：

- [Discord - 充满乐趣与游戏的群聊](https://discordapp.com/channels/1053877538025386074/1145143867818119272/1250149635083866133)：Discord 是玩游戏和与朋友放松，甚至建立全球社区的绝佳场所。定制你自己的空间来聊天、玩耍和聚会。
- [stabilityai/stable-diffusion-3-medium · Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-3-medium)：未找到描述
- [Nitral-AI/Hathor-L3-8B-v.02 · Hugging Face](https://huggingface.co/Nitral-AI/Hathor-L3-8B-v.02)：未找到描述
- [Nitral-AI/Hathor-L3-8B-v.02 · 优秀的无审查模型](https://huggingface.co/Nitral-AI/Hathor-L3-8B-v.02/discussions/1)：未找到描述
- [来自 Max Woolf (@minimaxir) 的推文](https://x.com/minimaxir/status/1800921802765717754)：看看人们对 Stable Diffusion 3 的测试，说实话，这真的很给力。
- [介绍苹果的端侧与服务器基础模型](https://machinelearning.apple.com/research/introducing-apple-foundation-models)：在 2024 年全球开发者大会上，我们推出了 Apple Intelligence，这是一个深度集成的个人智能系统……

---

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1250312828628238387) (7 条消息):

- **寻找日语语言模型**：一名成员询问有关日语的**视觉语言模型 (VLM)**和**大语言模型 (LLM)**的建议。他们请求提供一个列表以便更清晰地了解。
- **建议并评测 Cohere Aya**：另一名成员建议使用具有多语言能力的 **Cohere Aya**，并提到了 **Stability AI** 的模型。最初的成员认为 **Aya** 表现不错，但希望获得 API 访问权限，而不是自行部署。
- **探索 Cohere 的 API 选项**：一名成员提到 **Cohere** 提供了 **Aya 35B** (c4ai-aya-23) 的 API 访问，并建议使用支持日语的更大模型 **Command R+** (103B)。他们在试用 Aya 后分享了积极的体验。
- **查找 API 信息及最终建议**：请求者在查找 **Aya 35B** 的 API 定价信息时遇到困难，并请求提供直接链接。随后，他们在发现 **Cohere Command R+** 是满足其需求的最佳日语模型后表示宽慰和满意。

---

### **Nous Research AI ▷ #**[**world-sim**](https://discord.com/channels/1053877538025386074/1221910674347786261/1250488272140898417) (2 条消息):

- **更新简化了较长的控制台提示词**：发布了一项更新，旨在简化较长控制台提示词（prompts）的编写和编辑，提升了移动端和桌面端平台的体验。该更新承诺让用户的控制台交互更加顺畅。
- **询问 WorldSim 的开源状态**：一名成员询问 WorldSim 是否开源以及是否有公开的 GitHub 仓库。消息中未提供回复或进一步信息。

---

### **LAION ▷ #**[**general**](https://discord.com/channels/823813159592001537/823813160075132991/1250254590767792222) (26 条消息🔥):

- **关于 DPO LoRA 潜力的推测**：有人讨论了潜在的 **DPO LoRA**，一些人认为这可能会导致负面结果，例如**“制作糟糕的色情内容”**。然而，实际影响仍有待观察。
- **Elon Musk 因 OpenAI 合作伙伴关系对抗 Apple**：一名用户声称 **Elon Musk 封杀了 Apple 的账号**，原因是 Apple 与 OpenAI 的合作。这一观点附带了一个指向 [Threads.net](https://www.threads.net/@ronaldfilipkowski/post/C8F8woLt7rT) 的链接作为支持。
- **CaptionEmporium 使用 LlavaNext 为 CC12M 重新标注**：用户讨论了使用 **LlavaNext** 重新标注并发布在 HuggingFace 上的 **CC12M 数据集**。数据集链接为 [CaptionEmporium/conceptual-captions-cc12m-llavanext](https://huggingface.co/datasets/CaptionEmporium/conceptual-captions-cc12m-llavanext)。
- **Stable Diffusion 3 medium 发布但未包含 diffusers**：有人注意到 **Stable Diffusion 3 medium** 已经发布，但缺少 **diffusers 代码/权重**。这一点通过其在 [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-3-medium/tree/main) 上的当前状态得到了确认。
- **用于情感描述条件化的新数据集**：分享了一个令人兴奋的、用于**情感描述条件化**的新数据集。该数据集名为 **Emo-2-SNAC**，可在 [HuggingFace](https://huggingface.co/datasets/0xd4t4/Emo-2-SNAC) 上获取。

**提到的链接**：

- [Ron Filipkowski (@ronaldfilipkowski) 在 Threads 上](https://www.threads.net/@ronaldfilipkowski/post/C8F8woLt7rT)：Elon Musk 已在 Twitter 上自动屏蔽了来自 Apple 的所有人。
- [CaptionEmporium/TextOCR-GPT4o · Hugging Face 数据集](https://huggingface.co/datasets/CaptionEmporium/TextOCR-GPT4o?row=1>)：未找到描述
- [0xd4t4/Emo-2-SNAC · Hugging Face 数据集](https://huggingface.co/datasets/0xd4t4/Emo-2-SNAC)：未找到描述
- [ylacombe/mls-eng-10k-descriptions-10k-v4 · Hugging Face 数据集](https://huggingface.co/datasets/ylacombe/mls-eng-10k-descriptions-10k-v4)：未找到描述

---

### **LAION ▷ #**[**research**](https://discord.com/channels/823813159592001537/824374369182416994/1250286057602285629) (9 messages🔥):

- **图像数据集标注：从 iPhone 到 Gradio**：一位成员表示有兴趣*构建图像数据集*，并使用 iPad 和 iPhone 进行现场图像标注。他们讨论了可能使用 *Gradio* 创建自定义界面以便随时随地进行标注，涉及约 5000 张图像（包括裁剪部分）的标注工作。
- **Google 发布 RecurrentGemma 9B**：一位成员分享了 [Google 发布 RecurrentGemma 9B](https://huggingface.co/collections/google/recurrentgemma-release-66152cbdd2d6619cb1665b7a) 的消息，该模型在长序列上具有*极快的吞吐量*。该模型号称具有*与 Gemma 相似的质量*，并提供基础版和 Instruct 微调版，[Osanseviero 的帖子](https://x.com/osanseviero/status/1800607752038818260)中对此进行了重点介绍。
- **讨论 Transformer 的可学习性限制**：[arxiv.org](https://arxiv.org/abs/2406.06467) 上的一篇论文链接讨论了 *Transformer* 在通过组合已知三段论来预测新三段论方面的*局限性*。该研究探讨了“分布局部性”（distribution locality）的概念，以及高局部性分布在可学习性方面带来的挑战。
- **分享 Stable Diffusion 3 发布消息**：分享了 StabilityAI 的 [Stable Diffusion 3 集合](https://huggingface.co/collections/stabilityai/stable-diffusion-3-666992fb11f5d8d0ec8192e8)。该集合突出了 Diffusion 模型的进步，如成员提供的 Hugging Face 链接所示。

**提到的链接**：

- [How Far Can Transformers Reason? The Locality Barrier and Inductive Scratchpad](https://arxiv.org/abs/2406.06467)：Transformer 的推理能力有多远？局部性障碍与归纳草稿本。Transformer 能否通过组合已有的三段论来预测新的三段论？更广泛地说，这类模型从零开始可以学习什么样的目标？最近的研究表明 Transformer 可以是图灵...
- [Stable Diffusion 3 - a stabilityai Collection](https://huggingface.co/collections/stabilityai/stable-diffusion-3-666992fb11f5d8d0ec8192e8)：未找到描述
- [Tweet from Omar Sanseviero (@osanseviero)](https://x.com/osanseviero/status/1800607752038818260)：Google 的 RecurrentGemma 9B 发布了 🔥 ⚡️长序列处理速度极快：优秀的吞吐量 + 延迟 👀提供基础版和 Instruct 微调版 🏆质量与 Gemma 相似。看看下面的 y 轴 🤯 模型：https:...

---

### **LAION ▷ #**[**resources**](https://discord.com/channels/823813159592001537/991938328763056168/1250422238516084817) (1 messages):

- **GitHub 上的新机器学习库**：一位成员分享了他们基于 **TensorFlow** 开发的新机器学习库。该库旨在“轻松实现并行训练和分布式训练”，支持包括 Llama2, Llama3, CLIP 等在内的多种模型。点击[此处](https://github.com/NoteDance/Note)查看。

**提到的链接**：[GitHub - NoteDance/Note: Easily implement parallel training and distributed training. Machine learning library. Note.neuralnetwork.tf package include Llama2, Llama3, Gemma, CLIP, ViT, ConvNeXt, BEiT, Swin Transformer, Segformer, etc, these models built with Note are compatible with TensorFlow and can be trained with TensorFlow.](https://github.com/NoteDance/Note)：轻松实现并行训练和分布式训练。机器学习库。Note.neuralnetwork.tf 包包含 Llama2, Llama3, Gemma, CLIP, ViT, ConvNeXt, BEiT, Swin Transformer, Segf...

---

### **Modular (Mojo 🔥) ▷ #**[**general**](https://discord.com/channels/1087530497313357884/1098713601386233997/1250168751416934480) (3 messages):

- **Mojo 未来可能支持 TPU 硬件**：一位成员建议，如果 **Google 拥有针对 MLIR 或 LLVM 的 TPU 后端**，那么 **Mojo** 应该能够利用它。
- **Modular 编译器将具有可扩展性**：在未来的开发中，用户无需等待 Modular 支持新架构，因为他们计划让编译器具有**可扩展性**。
- **Mojo 基础设施尚新**：成员们互相提醒，**Mojo** 及其基础设施仍处于早期阶段，正在积极构建中。

---

### **Modular (Mojo 🔥) ▷ #**[**💬︱twitter**](https://discord.com/channels/1087530497313357884/1098713626161987705/) (1 messages):

ModularBot：来自 *Modular*：[https://twitter.com/Modular/status/1800948901652181260](https://twitter.com/Modular/status/1800948901652181260)

---

### **Modular (Mojo 🔥) ▷ #**[**📺︱youtube**](https://discord.com/channels/1087530497313357884/1098713700719919234/1250503759604224073) (1 messages):

- **新 Modular 视频提醒**：**Modular** 发布了最新视频。点击[此处](https://www.youtube.com/watch?v=uookgZ7Ojg8)观看。

### **Modular (Mojo 🔥) ▷ #**[**🔥mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1250168359618609253) (26 messages🔥):

- **变量别名建议未导致性能变化**：一位用户建议将 `var size = 12` 更改为 `alias size = 12`。另一位用户确认，这一更改在他们的 Benchmark 测试中没有显示出明显的性能提升。
- **文档中过时的 `Tensor` 模块示例**：由于 `Tensor.rand` 的使用方式不正确，Modular 官方文档网站上提供的 `Tensor` 示例无法按预期运行。经过讨论，建议使用 `Tensor[DType.float32].rand(...)`，该问题得以解决。
- **指针转换方案引发庆祝**：一位用户询问如何将指针转换为 `UInt64`，另一位用户提供了可行的解决方案。这促成了在一个 [Pull Request](https://github.com/modularml/mojo/pull/3007) 中的成功实现，并受到了热烈的赞扬。
- **寻找适用于 VSCode 的 Mojo 教程**：一位用户表示在寻找适用于 VSCode 的优质 Mojo 教程时感到沮丧，对此，其他用户分享了一些资源以及关于学习曲线的幽默评论。建议中包括一个[类似书籍的资源](https://ruhati.net/mojo)，尽管提到该资源仍在编写中。
- **技术影响力人物作为现代评论家**：一场关于是否存在编程评论家的轻松对话展开，最终达成共识，认为像 Theo 和 Primeagen 这样的 Tech Influencers 通过他们的视频履行了这一职责。这引发了对即将进行的采访和现有内容的分享，例如 [Marques 和 Tim 的采访](https://www.youtube.com/watch?v=pMX2cQdPubk)。

**提到的链接**：

- [tensor | Modular Docs](https://docs.modular.com/mojo/stdlib/tensor/tensor/)：实现 Tensor 类型。
- [[stdlib] Specify alignment in UnsafePointer.alloc by sa- · Pull Request #3007 · modularml/mojo](https://github.com/modularml/mojo/pull/3007)：修复了 #3006，在 UnsafePointer.alloc 中添加了一个函数参数，用于在编译时指定对齐方式。虽然我不确定如何测试它，因为我找不到将 ptr.address 转换为...的方法。
- [Learn Mojo Programming Language](https://ruhati.net/mojo)：未找到描述。

---

### **Modular (Mojo 🔥) ▷ #**[**nightly**](https://discord.com/channels/1087530497313357884/1224434323193594059/1250327138150776916) (2 messages):

- **新版 Mojo 编译器发布**：新的 Nightly 版本 Mojo 编译器 `2024.6.1205` 已发布。使用 `modular update nightly/mojo` 进行更新，并查看 [raw diff](https://github.com/modularml/mojo/compare/76eda306af929d9576d7190a7f8f3aa1df83baf6...b590ea1ba0e80093118e32d91dcc4d252ccdfe69) 和 [当前更新日志](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 以了解详细变更。
- **条件一致性功能获赞**：条件一致性（Conditional conformance）功能的发布收到了积极反馈。一位用户询问了支持递归 Trait 绑定的计划，并给出了一个涉及 `Wrapper[Wrapper[Stringable]]` 的例子。

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1250221369359073341) (6 messages):

- **Zep 提供便捷但有限制的内存管理**：一位用户指出，只要保持在免费层级内，**Zep** 对于内存管理非常方便。“否则就不值得了。”
- **Apple 将提供免费服务**：一位成员提到 **Apple** 将免费提供其服务，强调了相对于其他选项的竞争优势。另一位用户确认道：“是的，非常有道理。”
- **OpenAI API 密钥定价**：讨论表明 OpenAI 密钥的费用约为 **每月 5-10 美元**。这为那些计划使用 OpenAI 服务的人强调了成本考量。
- **在 GCP 上配置默认模型遇到麻烦**：一位用户寻求帮助，希望使用其 GCP 账户将默认模型更改为 **gemini-flash** 或 **codestral**。他们提到使用 **GPT-4o** 取得了成功，但成本很高，并提到了与各种 YAML 文件配置的冲突。
- **寻找 OpenInterpreter 专家**：另一位用户表示渴望与了解 OpenInterpreter 的人交流，以便在工作上取得进一步进展。他们请求协助：“如果你想帮忙，请联系我（HMU）。”

---

### **OpenInterpreter ▷ #**[**O1**](https://discord.com/channels/1146610656779440188/1194880263122075688/1250178624770936965) (22 条消息🔥):

- **分享了 OpenInterpreter GitHub 仓库链接**：一名成员分享了 [OpenInterpreter GitHub 仓库链接](https://github.com/OpenInterpreter/open-interpreter/tree/main/interpreter/terminal_interface/profiles/defaults)，展示了计算机的自然语言接口。
- **不含 websockets 的临时代码**：一名成员分享了一个 [GitHub Gist](https://gist.github.com/0xrushi/e56085f93698c7267af9b1ba9643dc7a)，其中包含不含 websockets 的 OpenInterpreter 临时代码，并注明其运行在 OpenAI key 上。
- **uConsole + OpenInterpreter = 野兽模式**：uConsole 与 OpenInterpreter 的结合被强调为一种强大的配置，并辅以 Odysee 上的[视频演示](https://odysee.com/@rushi:2/Openinterpreter-01-uconsole-test-part1:2)。
- **配置与组件详情**：讨论了使用带有精美键盘和屏幕的 Raspberry Pi CM4 作为该配置的设备。分享了 [ClockworkPi uConsole 的相关链接](https://www.clockworkpi.com/uconsole)，并提到该配置适用于任何不带 eMMC 模块的 CM4。
- **语音交互的额外配件**：虽然 uConsole 包含扬声器，但没有配备麦克风。一名成员建议使用 [mini USB 麦克风](https://www.amazon.com/dp/B071WH7FC6) 进行语音输入。

**提到的链接**：

- [无标题](https://www.amazon.com/dp/B071WH7FC6)：未找到描述
- [Odysee](https://odysee.@rushi:2/Openinterpreter-01-u)：在 Odysee 上探索由像你一样的普通人创作的视频宇宙！
- [Openinterpreter 01 uconsole test part1](https://odysee.com/@rushi:2/Openinterpreter-01-uconsole-test-part1:2)：在 Odysee 上查看 Openinterpreter 01 uconsole test part1
- [uConsole | ClockworkPi](https://www.clockworkpi.com/uconsole)：uConsole - 为独立游戏开发者和卧室程序员打造的真正“幻想控制台”。
- [open-interpreter/interpreter/terminal_interface/profiles/defaults at main · OpenInterpreter/open-interpreter](https://github.com/OpenInterpreter/open-interpreter/tree/main/interpreter/terminal_interface/profiles/defaults)：计算机的自然语言接口。通过在 GitHub 上创建账户为 OpenInterpreter/open-interpreter 的开发做出贡献。
- [temporary_01_code_without_sockets.py](https://gist.github.com/0xrushi/e56085f93698c7267af9b1ba9643dc7a)：GitHub Gist：即时分享代码、笔记和片段。

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1250202401600962681) (27 条消息🔥):

- **DeepEval 轻松集成自定义 LLM 指标**：在 DeepEval 中，用户可以为 LLM 定义自定义评估指标，包括 **G-Eval、Summarization、Faithfulness** 和 **Hallucination** 等指标。有关实现这些指标的更多详细信息，请查看[文档](https://docs.confident-ai.com/docs/metrics-introduction#using-a-custom-llm)。
- **无审查模型提供未过滤的响应**：用户讨论了缺乏过滤器或对齐偏差的新型**无审查模型（uncensored models）**的用途，许多人因其透明的响应和多种用例而更青睐这些模型。
- **WizardLM-2 的定价引发好奇**：关于 **WizardLM-2** 如何在每百万 token 仅 0.65 美元的情况下实现盈利的问题被提出。一名成员解释说，这些模型可能会使用较少的激活参数来节省成本，而租用 GPU 可能有助于成本管理。
- **自托管 vs. 使用 LLM 提供商**：讨论强调了与使用 OpenRouter 等服务提供商相比，**自托管模型**面临的挑战和成本。有人指出，除非有持续的高需求或现有的硬件投资，否则自托管在经济上可能并不划算。
- **批量推理可能使租用 GPU 变得合理**：成员们辩论了租用 GPU 进行繁重批量推理（Batch inference）任务的有效性，指出与单次请求场景相比，这可能具有成本效益。一个实际的建议是使用 **Aphrodite-engine / vllm** 等工具来优化大批量处理。

**提到的链接**：[Metrics | DeepEval - 开源 LLM 评估框架](https://docs.confident-ai.com/docs/metrics-introduction#using-a-custom-llm)：快速摘要

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1250413917583179808) (6 条消息):

- **QWEN 2 训练咨询**：一位成员询问是否有人训练过 15 亿参数的 **QWEN 2**，并寻求性能方面的见解。
- **在本地运行 70 亿参数模型**：同一位用户描述了如何在本地使用 **mlx** 库运行未量化的 **70 亿参数**模型。他们提到运行速度 *"慢得要命，但确实能跑，笑死"*。
- **Llama.cpp 建议**：另一位成员建议使用 **llama.cpp** 以获得更好的性能。原发布者表示他们只是在测试环境配置。

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**axolotl-dev**](https://discord.com/channels/1104757954588196865/1104758010959634503/1250438163768545302) (2 条消息):

- **Runpod 工作区困扰**：一位用户表达了沮丧，问道 *"为什么不直接保持为 /workspace/？"*。另一位用户解释说，在 Runpod 上将数据卷挂载到 **/workspace** 经常会覆盖 **/workspace/axolotl**，导致必须重新安装或重新克隆 Axolotl。

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**docs**](https://discord.com/channels/1104757954588196865/1167137552470392842/) (1 条消息):

le_mess: 上传模型时不需要第 2 步。Repo 会自动创建。

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**axolotl-help-bot**](https://discord.com/channels/1104757954588196865/1225300056442409040/1250421897791934465) (6 条消息):

- **PyTorch 分布式启动中的严重错误**：一位成员在使用 PyTorch 的分布式启动器时遇到了子进程失败：*"torch.distributed.elastic.multiprocessing.api: [ERROR] failed (exitcode: -7) local_rank: 4 (pid: 3830) of binary: /root/miniconda3/envs/py3.10/bin/python"*。
- **Phorm 提供故障排除步骤**：解决 *ChildFailedError* 的步骤包括检查环境设置、验证分布式配置、检查脚本错误、增加共享内存大小，以及按照 [Accelerate 文档](https://github.com/huggingface/accelerate/tree/main/docs/source/basic_tutorials/troubleshooting.md#L50L145) 中所述启用调试模式。

**提到的链接**：

- [accelerate/docs/source/basic_tutorials/troubleshooting.md at main · huggingface/accelerate](https://github.com/huggingface/accelerate/tree/main/docs/source/basic_tutorials/troubleshooting.md#L50L145)：🚀 一种在几乎任何设备和分布式配置上启动、训练和使用 PyTorch 模型的简单方法，支持自动混合精度（包括 fp8），以及易于配置的 FSDP 和 DeepSpeed 支持……
- [OpenAccess-AI-Collective/axolotl | Phorm AI Code Search](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=c0a36daa-352f-4d94-8f0c-1e23a40fbb2f)：更快地理解代码。

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**axolotl-phorm-bot**](https://discord.com/channels/1104757954588196865/1225558824501510164/1250437253189337129) (6 条消息):

- **使用 LoRA 继续进行指令训练**：一位用户询问如何从补全格式（completion format）训练转向使用 LoRA 的指令格式（instruction format）训练。回复详细说明了加载预训练模型、准备指令数据、配置 LoRA 以及设置和执行训练过程的步骤，并提供了代码示例。
- **关于合并 Mistral 7B 的 qLoRA 训练结果的查询**：一位成员询问如何在仅有 24GB GPU 的系统上合并 Mistral 7B 的 qLoRA 训练结果。该问题强调了在模型合并过程中处理资源限制的指导需求。

**提到的链接**：[OpenAccess-AI-Collective/axolotl | Phorm AI Code Search](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=19ce6da5-fa00-4b7a-80ea-bde7eb10ae28)：更快地理解代码。

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1250168057851019356) (17 条消息🔥):

- **Google 发布 RecurrentGemma 9B 引发关注**：Google 推出的新模型 **RecurrentGemma 9B** 承诺在处理长序列时具有超高性能。它包含 Base 和 Instruct-tuned 版本，据评估其质量与 **Gemma** 相当。[点击此处了解更多](https://huggingface.co/collections/google/recurrentgemma-release-66152cbdd2d6619cb1665b7a) 并阅读 [Griffin 论文](https://arxiv.org/abs/2402.19427)。
- **Google 个人健康 LLM 首次亮相**：Google 发布了一款基于 Gemini 微调的新型 **Personal Health Large Language Model**，旨在读取可穿戴设备数据并提供个性化建议。一位兴奋的成员指出：*“它在认证考试中的表现优于专业的睡眠和健身专家”*。[详情点击此处](https://x.com/chefjeffsf/status/1800597192593621100)。
- **ARC Prize 启动公告**：ARC Prize 今日启动，该项目与一个正在进行的收集人类如何解决 ARC 任务数据的项目相关联，目前已托管 4100 条交互历史。贡献内容包括展示方法和解决方案的多个 [视频和数据集](https://github.com/neoneye/ARC-Interactive-History-Dataset)。[更多参与方式和数据集请点击此处](https://neoneye.github.io/arc/)。
- **Mihail Eric 揭露 Alexa AI 的缺陷**：在一段详细的推文中，Mihail Eric 讨论了 **Alexa AI** 因技术和官僚问题（包括产品和科学团队之间的碎片化和目标不一致）而错失的机会。尽管 Alexa 拥有大量资源，但许多潜在的进步从未公开发布。[完整故事点击此处](https://x.com/mihail_eric/status/1800578001564057754)。
- **hwchung27 的斯坦福讲座**：hwchung27 在斯坦福大学进行了一场全面的讲座，讨论了 AI 的快速演进，强调了适应由指数级廉价算力和 Scaling 模型驱动的变化的重要性。讲座通过 Transformer 架构的演进等实际案例来突出这些概念。观看 [讲座视频](https://youtu.be/orDKvo8h71o?si=RIfyZ7NSUAJifOBF) 并查看 [幻灯片](https://docs.google.com/presentation/d/1u05yQQaw4QXLVYGLI6o3YoFHv6eC3YN8GvWD8JMumpE/edit?usp=sharing)。

**提到的链接**：

- [Omar Sanseviero (@osanseviero) 的推文](https://x.com/osanseviero/status/1800607752038818260?s=46&t=90xQ8sGy63D2OtiaoGJuww)：Google 的 RecurrentGemma 9B 已发布 🔥 ⚡️长序列处理超快：良好的吞吐量 + 延迟 👀包含 Base 和 Instruct tuned 版本 🏆质量与 Gemma 相当，看看下面的 y 轴 🤯 模型：https:...
- [我告诉过你这会发生](https://youtu.be/zbo6SdyWGns?si=5UypK-JD5h7Gz-SJ)：在今天的节目中，我将讨论 AI 音乐技术的最新进展及其影响。促销：🎂 频道周年纪念捆绑包 — $89 FO...
- [Hyung Won Chung (@hwchung27) 的推文](https://x.com/hwchung27/status/1800676312916656592?s=46&t=90xQ8sGy63D2OtiaoGJuww)：我在 @Stanford CS 25 做了一场讲座。讲座视频：https://youtu.be/orDKvo8h71o?si=RIfyZ7NSUAJifOBF AI 发展如此之快，以至于很难跟上。与其花费所有精力去追赶...
- [Mihail Eric (@mihail_eric) 的推文](https://x.com/mihail_eric/status/1800578001564057754?s=46&t=90xQ8sGy63D2OtiaoGJu)：Alexa 是如何错失成为全球顶级对话系统的机会的 —— 几周前 OpenAI 发布了 GPT-4o，为多模态、对话式体验树立了新标准...
- [Chef Jeff (@chefjeffsf) 的推文](https://x.com/chefjeffsf/status/1800597192593621100)：突破性消息：Google 刚刚发布了一个 Personal Health Large Language Model - 基于 Gemini 微调 - 读取你的可穿戴数据以发现个性化见解和建议 - 表现优于专业人士...
- [Mihail Eric (@mihail_eric) 的推文](https://x.com/mihail_eric/status/1800578001564057754?s=46&t=90xQ8sGy63D2OtiaoGJuww)：Alexa 是如何错失成为全球顶级对话系统的机会的 —— 几周前 OpenAI 发布了 GPT-4o，为多模态、对话式体验树立了新标准...
- [Lucas Beyer (bl16) (@giffmana) 的推文](https://x.com/giffmana/status/1800617190242091289?s=46&t=90xQ8sGy63D2OtiaoGJuww)：如果你从未在 BigCo（大公司）工作过，你可能会发现自己有这样的想法：> 为什么这个 BigCo 的东西显然这么烂？修好它只需要我一个下午的时间！他们的...
- [ARC Prize – 一项旨在推动开放 AGI 进展的 100 万美元以上奖金的竞赛 | Hacker News](https://news.ycombinator.com/item?id=40648960)：未找到描述

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1250361399935242342) (16 条消息🔥):

- **MLPerf 上的 RetinaNet 悬赏 600 美元**：George Hotz 宣布了在 MLPerf 上实现 RetinaNet 的 600 美元悬赏，如果能进入 MLPerf 基准测试，悬赏金额将翻倍。他提到，“既然 ResNet/BERT 已经在 master 分支中，这个悬赏应该不难拿到。”
- **PyTorch 数据加载的挑战**：一位成员强调，使用 PyTorch 开发的大部分时间都花在了高效的数据加载上，特别是在 HPC 集群中。他们建议采用类似 [WebDataset](https://github.com/webdataset/webdataset) 的解决方案。
- **TinyGrad 数据加载器计划**：George Hotz 为 TinyGrad 的数据加载器提出了一个简单的 API，包括指定加载记录的函数、shuffle 缓冲区大小和 batch size。他提到了 Comma 的 “gigashuffle” 作为现有的抽象参考。
- **TinyGrad 0.9.0 更新**：根据一个 [GitHub Pull Request](https://github.com/NixOS/nixpkgs/pull/316931)，TinyGrad 0.9.0 已添加到 Nix 仓库。该更新直接在代码库中包含了 gpuctypes。
- **MLPerf 与出版物**：MLPerf 结果已发布，包括 tinybox red/green 的基准测试。此外还提到了一篇关于 TinyGrad 的德语文章（发表在 [Heise.de](https://www.heise.de/news/KI-Benchmark-MLPerf-Erste-AMD-Beschleuniger-mit-Minimalauftritt-9760531.html)），并计划发布一篇更详细的博客文章，将 TinyGrad 的速度与理论极限进行比较。

**提到的链接**：

- [未找到标题](https://public.tableau.com/app/profile/data.visualization6666/viz/MLCommons-Training_16993769118290/MLCommons-Training)：未找到描述
- [GitHub - pytorch/data: A PyTorch repo for data loading and utilities to be shared by the PyTorch domain libraries.](https://github.com/pytorch/data)：一个用于数据加载和实用程序的 PyTorch 仓库，供 PyTorch 领域库共享。 - pytorch/data
- [tinygrad/docs/tinygrad_intro.pdf at master · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/blob/master/docs/tinygrad_intro.pdf)：你喜欢 PyTorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - tinygrad/tinygrad
- [[MLPERF] Retinanet by reddyn12 · Pull Request #4245 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/pull/4245)：TODO: 添加 float16 支持，修复 float16 保存，float32 可用，添加 bfloat16 支持，加速验证循环，修复验证循环内存膨胀（持有 grads），清理 anchor 生成代码，移除旧的 retinan...
- [python311Packages.tinygrad: 0.8.0 -> 0.9.0 by GaetanLepage · Pull Request #316931 · NixOS/nixpkgs](https://github.com/NixOS/nixpkgs/pull/316931)：变更说明。更新日志：https://github.com/tinygrad/tinygrad/releases/tag/v0.9.0 显著变化：gpuctypes 已直接包含在代码库中。因此我们必须移植我们的 GPU 依赖...
- [KI-Benchmark MLPerf: Erste AMD-Beschleuniger mit Minimalauftritt](https://www.heise.de/news/KI-Benchmark-MLPerf-Erste-AMD-Beschleuniger-mit-Minimalauftritt-9760531.html)：他做到了：前黑客 George Hotz 的 tinycorp 公司在 MLPerf Training v4.0 中首次展示了 AMD 芯片——但仅有一个单项数值。

---

### **Mozilla AI ▷ #**[**llamafile**](https://discord.com/channels/1089876418936180786/1182689832057716778/1250176793441210429) (13 条消息🔥):

- **LLM 需要 grammar 或 JSON schemas 才能发挥效用**：一位成员认为，如果没有 grammar 或 JSON schemas，**LLM** 在应用层面（如 Agent）基本上是**无用的**，因为这些工具强制采样器仅从 token 子集中选择，从而使输出可以被外部程序解析。这一过程通过标准化输出格式提高了应用层的实用性。
- **简化 llamafile 的 grammar 使用**：成员们讨论了简化 `llamafile` 配合 grammar 选项的使用，并指出了两个步骤：将 JSON schema 转换为 grammar，然后使用该 grammar。有人建议使用命令 `llamafile --grammar <(schemaify <foo.sql)`。
- **打包 ggml_cuda.so 和 ggml_rocm.so**：一位用户询问如何高效地将 `ggml_cuda.so` 和 `ggml_rocm.so` 打包进 llamafile 发布版中。另一位成员回复了一个 Bash 脚本示例，并提到 AMD 和 tinyblas 需要手动修改。
- **使用 magit 进行同步**：为了同步 `llama.cpp`，一位成员分享了他们使用 [Magit](https://www.youtube.com/watch?v=urcL86UpqZc) 的经验——这是 Emacs 中一个流行的 Git 界面，并在一段名为“2023 年采访 Emacs 爱好者 [上色版]”的 YouTube 视频中幽默地展示了这一点。

**提到的链接**：[Interview with an Emacs Enthusiast in 2023 [Colorized]](https://www.youtube.com/watch?v=urcL86UpqZc)：2023 年采访 Emacs 爱好者 [上色版]，受访者 Emerald McS., PhD - 播出机构 © The Emacs.org，播出日期 1990 年。程序员幽默，软件幽默，Elisp 幽默...

---

### **Datasette - LLM (@SimonW) ▷ #**[**ai**](https://discord.com/channels/823971286308356157/1097032579812687943/1250231626315862036) (1 messages):

- **Apple 的适配器类似于 LORA 层**：一位成员指出了 Apple 视频中关于其适配器（adaptors）的一个有趣观点，这些适配器可以在本地模型上动态加载以执行不同任务。他们将这一特性比作专门的 **LORA 层**。

---

### **Datasette - LLM (@SimonW) ▷ #**[**llm**](https://discord.com/channels/823971286308356157/1128504153841336370/1250187539038343178) (10 messages🔥):

- **Chrisamico 在文章摘要方面遇到困难**：Chrisamico 寻求帮助，想要通过 `curl` 获取一篇长文章，提取特定的标签内容，并将其通过管道传输给 `llm`，配合 system prompt 进行摘要。最终，他发现直接将文章粘贴到 ChatGPT 中更容易。
- **HTML 抓取工具琳琅满目**：成员们讨论了几个从 HTML 中提取内容的工具，包括 [htmlq](https://github.com/mgdm/htmlq)、[shot-scraper](https://shot-scraper.datasette.io/en/stable/javascript.html#example-extracting-page-content-with-readability-js) 以及 `nokogiri` Ruby gem。Simonw 提供了关于使用 shot-scraper 配合 JavaScript 提取内容的详细说明。
- **来自 Simon 的 Shot-scraper 技巧**：Simonw 分享了如何使用 `shot-scraper` 对页面执行 JavaScript，从而高效提取页面内容。此外，他还解释了直接使用 CSS 选择器通过 `shot-scraper` 进行 HTML 提取的方法。
- **Dbreunig 了解了 Nokogiri CLI**：Dbreunig 提到他发现 Ruby gem `nokogiri` 在安装后可以作为 CLI 工具用于解析 HTML。他提供了一个如何使用 `nokogiri` 从 HTML 文档中提取并格式化文章文本的快速示例。

**提到的链接**：

- [A Plea for Sober AI](https://www.dbreunig.com/2024/05/16/sober-ai.html)：呼吁冷静对待 AI：炒作声太大，以至于我们无法欣赏其魔力。
- [Datasette](https://datasette.io/)：Datasette 是一个用于探索和发布数据的工具。它帮助人们处理任何形状的数据，进行分析和探索，并将其发布为交互式网站和配套的 API。
- [GitHub - mgdm/htmlq: Like jq, but for HTML.](https://github.com/mgdm/htmlq)：像 jq 一样，但针对的是 HTML。可以通过在 GitHub 上创建账号来为 mgdm/htmlq 的开发做贡献。
- [Scraping pages using JavaScript - shot-scraper](https://shot-scraper.datasette.io/en/stable/javascript.html#example-extracting-page-content-with-readability-js)：未找到描述。
- [Datasette 0.61: The annotated release notes](https://simonwillison.net/2022/Mar/24/datasette-061/)：我今天早上发布了 Datasette 0.61，随后紧接着发布了 0.61.1 以修复一个微小 bug。这是带注释的发布说明。为了准备 Datasette 1.0，此版本包含两个潜在的……

---

### **LangChain AI ▷ #**[**general**](https://discord.com/channels/1038097195422978059/1038097196224086148/1250299529736093757) (6 messages):

- **LangChain Postgres 文档令用户困惑**：一位成员分享了访问 LangChain Postgres 文档时遇到的问题，指出包中缺少 checkpoint。*"langchain_postgres 是怎么回事。我无法使用这里的文档，因为包里没有 checkpoint。"* [LangChain Postgres Documentation](https://api.python.langchain.com/en/latest/checkpoint/langchain_postgres.checkpoint.PostgresSaver.html)。
- **在一次调用中组合链**：一位用户询问如何将一个输出数组的链与另一个链在一次调用中组合，并考虑使用 `RunnableParallel`，但不确定这是否是最佳方法。*"有人知道是否可以在一个链中完成吗？我在考虑使用 RunnableParallel..."*
- **在 langchain_openai 中使用 GPT-4 时出错**：一位成员报告了在使用 GPT-4 配合 `langchain_openai` 时出现的错误，并收到了使用 `ChatOpenAI` 而非 `OpenAI` 的指导，因为后者使用的是不支持新模型的旧版 API。*"仅使用 `OpenAI` 会调用旧版的 completions API，该 API 不支持新模型。你应该使用 `ChatOpenAI`。"* [OpenAI Completions API](https://platform.openai.com/docs/api-reference/completions)。

**提到的链接**：[langchain_postgres.checkpoint.PostgresSaver — 🦜🔗 LangChain 0.2.0rc2](https://api.python.langchain.com/en/latest/checkpoint/langchain_postgres.checkpoint.PostgresSaver.html#)：未找到描述。

---

### **LangChain AI ▷ #**[**langserve**](https://discord.com/channels/1038097195422978059/1170024642245832774/1250471553288638625) (1 messages):

- **在 LangServe 中分享对话历史**：一位用户询问如何在 LangServe 的 chat playground 中分享对话历史。他们引用了 [GitHub Issue #677](https://github.com/langchain-ai/langserve/issues/677)，指出点击 "Share" 按钮并复制 URL 会导致打开一个空白聊天，而不是预期的历史记录。

**提到的链接**：[How to share conversation from chat playground? · Issue #677 · langchain-ai/langserve](https://github.com/langchain-ai/langserve/issues/677)：我想从 LangServe 分享对话，该如何操作？我点击了 "Share" 按钮，然后复制了 URL：当我在浏览器中打开该 URL 时，它显示的是一个空白聊天，没有...

---

### **LangChain AI ▷ #**[**share-your-work**](https://discord.com/channels/1038097195422978059/1038097372695236729/1250270488081465374) (2 messages):

- **Nostrike AI 提供免费 Python 工具**：Nostrike AI 宣布推出一款用于生成 CrewAI (Python) 代码的新工具，主打用户友好且免费访问。他们还计划很快支持导出 Langgraph 项目，并邀请用户在 [nostrike.ai](https://nostrike.ai/) 进行尝试。
- **Rubik’s AI 招募 Beta 测试人员**：一位成员介绍了 Rubik's AI，这是一个先进的研究助手和搜索引擎，并提供使用促销代码 `RUBIX` 的 2 个月免费高级版试用。该试用包括访问 GPT-4 Turbo、Claude 3 Opus 和 Mistral Large 等模型，详见[此处](https://rubiks.ai/)。

**提到的链接**：

- [Rubik's AI - AI research assistant & Search Engine](https://rubiks.ai/)：未找到描述
- [NoStrike](https://nostrike.ai/)：未找到描述

---

### **Torchtune ▷ #**[**announcements**](https://discord.com/channels/1216353675241590815/1216353675241590818/1250390642639503381) (1 messages):

- **新应用带来乐趣、游戏和功能**：*从 6 月 18 日开始，成员现在可以将应用添加到其账户，并在所有服务器、DM 和 GDM 中使用。* 在[帮助中心文章](https://support.discord.com/hc/articles/23957313048343)中了解有关如何在服务器中管理这些应用的更多信息。
- **使用高级功能管理你的服务器**：*了解这些应用能（和不能）做什么，设置新的“使用外部应用”权限，并利用升级后的 AutoMod 来保护你的服务器安全。* 对于那些有兴趣创建自己应用的人，请查看[构建你自己的应用指南](https://discord.com/developers/docs/tutorials/developing-a-user-installable-app)。

---

### **Torchtune ▷ #**[**general**](https://discord.com/channels/1216353675241590815/1216353675744641096/1250379101051818014) (2 messages):

- **对 Torchtune 中 cache memory 使用的好奇**：一位成员询问 *“为什么 Torchtune 在每一步之后都会利用 cache memory？”*。另一位成员请求提供更多上下文，以便更好地理解和回应这一疑问。

---

### **Torchtune ▷ #**[**dev**](https://discord.com/channels/1216353675241590815/1236040539409879170/1250529900495110267) (1 messages):

- **Tokenizer 重构的 RFC 受到关注**：一位成员分享了一个大规模 Tokenizer 重构的 RFC，并在 [GitHub](https://github.com/pytorch/torchtune/pull/1082) 上提出了代码更改。该提案旨在启用多模态功能，提高可组合性和设计一致性，并减少新模型 Tokenizer 的接入时间。

**提到的链接**：[Build software better, together](https://github.com/pytorch/torchtune/pull/1082.)：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。

---

---

---

---

---

---

{% else %}

> 完整的频道细分内容已为邮件格式截断。
> 
> 如果你想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
> 
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}