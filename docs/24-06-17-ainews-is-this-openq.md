---
companies:
- deepseek_ai
- anthropic
- runwayml
- openai
- apple
- nvidia
- stability-ai
- luma-labs
date: '2024-06-18T00:38:33.191318Z'
description: '**DeepSeekCoder V2** 承诺以极低的成本提供超越 GPT-4 Turbo 的性能。**Anthropic** 发布了关于奖励篡改（reward
  tampering）的新研究。**Runway** 推出了 Gen-3 Alpha 视频生成模型，作为对 Sora 的回应。一系列论文探讨了“推理时”（test-time）搜索技术，旨在利用
  **LLaMa-3 8B** 等模型提升数学推理能力。**Apple** 发布了 Apple Intelligence，带来更智能的 Siri 以及图像和文档理解能力，并与
  **OpenAI** 合作将 ChatGPT 集成到 iOS 18 中，同时发布了 20 个支持 LoRA 微调以实现专业化的新 CoreML 模型。**NVIDIA**
  发布了 **Nemotron-4 340B**，这是一款性能媲美 GPT-4 的开源模型。**DeepSeek-Coder-V2** 在编程和数学方面表现出色，支持
  338 种编程语言，上下文长度达 128K。**Stability AI** 发布了 Stable Diffusion 3 Medium 的权重。**Luma
  Labs** 推出了 Dream Machine，可根据文本和图像生成 5 秒钟的视频。'
id: 673ee033-f1bd-46e2-a99c-140ed5ab9682
models:
- deepseek-coder-v2
- llama-3-8b
- nemotron-4-340b
- stable-diffusion-3-medium
original_slug: ainews-is-this-openq
people:
- adcock_brett
- clementdelangue
- svpino
title: 这是……OpenQ* 吗？
topics:
- reward-tampering
- test-time-search
- mathematical-reasoning
- process-supervision
- fine-tuning
- on-device-ai
- video-generation
- cost-efficiency
- context-length
- coding
- image-understanding
- multimodality
---

<!-- buttondown-editor-mode: plaintext -->**MCTS is all you need.**

> AI 新闻摘要：2024/06/14 - 2024/06/17。
我们为您查看了 7 个 subreddits、[**384** 个 Twitters](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discords（**414** 个频道，**5506** 条消息）。
预计节省阅读时间（以 200wpm 计算）：**669 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

**本周末有一系列增量发布**；[DeepSeekCoder V2](https://x.com/deepseek_ai/status/1802680388256768145) 承诺提供超越 GPT4T 的性能（已由 [aider](https://x.com/paulgauthier/status/1802774069185753298) 验证），价格仅为每百万 tokens $0.14/$0.28（相比之下 GPT4T 为 $10/$30），Anthropic 发布了一些 [Reward Tampering 研究](https://x.com/anthropicai/status/1802743256461046007?s=46&t=90xQ8sGy63D2OtiaoGJuww)，而 [Runway 终于发布了针对 Sora 的回应产品](https://x.com/runwayml/status/1802691475391566108?s=46&t=90xQ8sGy63D2OtiaoGJuww)。

然而，更持久、更有实质性的内容可能是围绕 "test-time" search 的讨论：

 
![image.png](https://assets.buttondown.email/images/7f958e3b-c3f9-452d-9499-2e97af8c0c7e.png?w=960&fit=max)
 

引发了一系列[相关论文](https://x.com/teortaxestex/status/1802128370861232374?s=46&t=90xQ8sGy63D2OtiaoGJuww)：

- [Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B: A Technical Report](https://arxiv.org/abs/2406.07394)
- [Improve Mathematical Reasoning in Language Models by Automated Process Supervision](https://arxiv.org/abs/2406.06592)
- [AlphaMath Almost Zero: Process Supervision Without Process](https://arxiv.org/abs/2405.03553)
- [ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search](https://arxiv.org/abs/2406.03816)

坦白说，我们还没读过这些论文，但我们在 [ICLR 播客中讨论过 OpenAI 关于 verifier-generator 过程监督（process supervision）的看法](https://www.latent.space/p/iclr-2024-benchmarks-agents)，并已将剩余论文列入 Latent Space Discord 论文俱乐部（Paper Club）的计划中。

---

{% if medium == 'web' %}

**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 回顾

> 所有摘要均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在尝试使用 Haiku 进行聚类和流程工程（flow engineering）。

**Apple 的 AI 进展与合作伙伴关系**

- **Apple Intelligence 发布**：[@adcock_brett](https://twitter.com/adcock_brett/status/1802371344539025882) 指出 Apple 在 WWDC 上展示了 Apple Intelligence，这是他们首个应用于 iPhone、iPad 和 Mac 的 AI 系统，具有更智能的 Siri 以及图像/文档理解等功能。
- **OpenAI 合作伙伴关系**：Apple 和 OpenAI 宣布达成合作，将 ChatGPT 直接集成到 iOS 18、iPadOS 18 和 macOS 中，如 [@adcock_brett](https://twitter.com/adcock_brett/status/1802371424268460037) 所述。
- **端侧 AI 模型**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1802742076544594254) 强调 Apple 在 Hugging Face 上发布了 20 个新的 CoreML 端侧 AI 模型和 4 个新数据集。
- **优化训练**：据 [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1802717835182968928) 报道，Apple 展示了其新模型的性能，以及它们是如何进行训练和优化的。
- **用于专业化的 LoRA 适配器**：[@svpino](https://twitter.com/svpino/status/1802677551355048292) 解释了 Apple 如何使用 LoRA 微调来为不同任务生成专门的“适配器（adapters）”，并实现即时切换。

**开源 LLM 达到 GPT-4 性能水平**

- **来自 NVIDIA 的 Nemotron-4 340B**：根据 [@adcock_brett](https://twitter.com/adcock_brett/status/1802371484972720396) 的消息，NVIDIA 发布了 Nemotron-4 340B，这是一个性能媲美 GPT-4 (0314) 的开源模型。
- **DeepSeek-Coder-V2**：[@deepseek_ai](https://twitter.com/deepseek_ai/status/1802680388256768145) 推出了 DeepSeek-Coder-V2，这是一个 230B 的模型，在编程和数学方面表现出色，超越了多个其他模型。它支持 338 种编程语言和 128K 上下文长度。
- **Stable Diffusion 3 Medium**：[@adcock_brett](https://twitter.com/adcock_brett/status/1802371687129686390) 提到，Stability AI 发布了其文本转图像模型 Stable Diffusion 3 Medium 的开源模型权重，提供了更先进的能力。

**新的视频生成模型**

- **Luma Labs 的 Dream Machine**：据 [@adcock_brett](https://twitter.com/adcock_brett/status/1802371446678704332) 报道，Luma Labs 推出了 Dream Machine，这是一款新的 AI 模型，可以根据文本和图像提示生成 5 秒长的视频片段。
- **Runway 的 Gen-3 Alpha**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1802706846597398749) 展示了 Runway 的新 Gen-3 Alpha 模型，该模型可以生成具有复杂场景和自定义选项的高细节视频。
- **Apparate Labs 的 PROTEUS**：正如 [@adcock_brett](https://twitter.com/adcock_brett/status/1802371709518885038) 所提到的，Apparate Labs 推出了 PROTEUS，这是一款实时 AI 视频生成模型，可以从单张参考图像创建逼真的头像和口型同步。
- **Google DeepMind 的 Video-to-Audio**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1802733643992850760) 分享了其视频转音频（video-to-audio）生成技术的进展，能够为无声片段添加与场景声学和画面动作相匹配的声音。

**机器人与具身智能（Embodied AI）进展**

- **用于机器人的 OpenVLA**：据 [@adcock_brett](https://twitter.com/adcock_brett/status/1802371664795009361) 报道，OpenVLA 是一款新的开源 7B 参数机器人基础模型，其性能优于更大规模的闭源模型。
- **DeepMind 和哈佛大学的虚拟啮齿动物**：[@adcock_brett](https://twitter.com/adcock_brett/status/1802371597669388586) 指出，DeepMind 和哈佛大学创建了一个由 AI 神经网络驱动的“虚拟啮齿动物”，模仿了现实生活中老鼠的敏捷动作和神经活动。
- **Northrop Grumman 的 Manta Ray 无人机**：[@adcock_brett](https://twitter.com/adcock_brett/status/1802371575347331399) 提到 Northrop Grumman 发布了“Manta Ray”的视频，这是他们新型无人水下航行器（UUV）无人机原型。
- **利用人形机器人实现自动驾驶**：据 [@adcock_brett](https://twitter.com/adcock_brett/status/1802371731899740536) 报道，一种自动驾驶的新方法正在利用人形机器人根据传感器反馈来操作车辆控制装置。

**其他 AI 研究与应用**

- **Anthropic 的奖励篡改研究**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1802743256461046007) 发表了一篇研究奖励篡改（reward tampering）的新论文，表明 AI 模型可以学会破解自己的奖励系统。
- **Meta 的 CRAG 基准测试**：[@dair_ai](https://twitter.com/dair_ai/status/1802695213086863748) 重点介绍了 Meta 讨论纠正检索增强生成（Corrective Retrieval-Augmented Generation, CRAG）基准测试的文章。
- **用于从视频中学习语言的 DenseAV**：[@adcock_brett](https://twitter.com/adcock_brett/status/1802371642460426368) 提到了一种名为“DenseAV”的 AI 算法，它可以从无标签视频中学习语言含义和声音位置。
- **用于训练 LLM 的 Goldfish loss**：[@tomgoldsteincs](https://twitter.com/tomgoldsteincs/status/1802726878924464273) 介绍了 goldfish loss，这是一种在不记忆训练数据的情况下训练 LLM 的技术。
- **对齐 LLM 中的创造力下降**：[@hardmaru](https://twitter.com/hardmaru/status/1802578579605393892) 分享了一篇论文，探讨了使用 RLHF 对齐 LLM 的意外后果，即降低了它们的创造力和输出多样性。

---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**AI 模型与技术**

- **针对 Stable Diffusion 改进的 CLIP ViT-L/14**：在 /r/StableDiffusion 中，[**改进后的 CLIP ViT-L/14 模型已开放下载**](https://www.reddit.com/r/StableDiffusion/comments/1dhaz43/greatly_improved_clip_vitl14_for_download_a/)，同时还提供了一个 Long-CLIP 版本，可用于任何 Stable Diffusion 模型。
- **从零开始的混合精度训练 (Mixed Precision Training)**：在 /r/MachineLearning 中，展示了 [Nvidia 原始混合精度训练论文在 2 层 MLP 上的重新实现](https://www.reddit.com/r/MachineLearning/comments/1dhlh0z/p_mixed_precision_training_from_scratch/)，深入 CUDA 领域以展示 TensorCore 的激活情况。
- **理解 LoRA**：同样在 /r/MachineLearning 中，分享了一份用于高效微调大语言模型的 [**低秩自适应 (LoRA) 理解视觉指南**](https://www.reddit.com/r/MachineLearning/comments/1dh4s3x/r_understanding_lora_a_visual_guide_to_lowrank/)。LoRA 将微调涉及的参数数量减少了 10,000 倍，同时仍能收敛到全量微调模型的性能。
- **使用 LLaMa-3 8B 达到 GPT-4 级别的数学解题能力**：一篇 [研究论文探讨了如何通过 LLaMa-3 8B 模型结合蒙特卡洛树自我细化 (Monte Carlo Tree Self-refine)](https://arxiv.org/abs/2406.07394) 来获得 GPT-4 级别的奥数解决方案。
- **从零开始的指令微调 (Instruction Finetuning)**：提供了一个 [从零开始实现指令微调](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/ch07.ipynb) 的代码实现。
- **AlphaMath Almost Zero**：[关于 AlphaMath Almost Zero 的研究](https://arxiv.org/abs/2405.03553) 引入了“无过程的过程监督 (process supervision without process)”。

**Stable Diffusion 模型与技术**

- **模型对比**：在 /r/StableDiffusion 中，展示了 [PixArt Sigma、Hunyuan DiT 和 SD3 Medium 模型](https://www.reddit.com/r/StableDiffusion/comments/1dh73j6/model_comparison_pixart_sigma_vs_hunyuan_dit_vs/) 在图像生成方面的对比，其中 PixArt Sigma 和 SDXL 细化表现出良好的前景。
- **针对 SD3 的 ControlNet**：[针对 SD3 的 ControlNet Canny 和 Pose 模型已经发布](https://www.reddit.com/r/StableDiffusion/comments/1dh257c/controlnet_canny_and_pose_are_released_for_sd3/)，Tile 和 Inpainting 模型即将推出。
- **采样器与调度器排列组合**：提供了一份 [Stable Diffusion 3 所有可用的采样器 (Sampler) 和调度器 (Scheduler) 组合概览](https://www.reddit.com/r/StableDiffusion/comments/1dh5n7k/all_sampler_and_scheduler_permutations_for_stable/)。
- **SD3 中的 CFG 值**：[Stable Diffusion 3 中不同 CFG 值的对比](https://www.reddit.com/r/StableDiffusion/comments/1dhdyt7/comparison_of_cfg_values_in_stable_diffusion_3/) 显示，与 SD1 相比，其可用范围更窄。
- **Playground 2.5 类似于 Midjourney**：[Playground 2.5 模型被认为在输出质量和风格上与 Midjourney 最为相似](https://www.reddit.com/r/StableDiffusion/comments/1dhflr4/the_most_similar_model_to_midjourney_is/)。
- **SD3 中的层扰动分析**：进行了一项 [关于在 SD3 的不同层中添加随机噪声如何影响最终输出的分析](https://www.reddit.com/r/StableDiffusion/comments/1dh3ml4/manual_layer_pertrubation_for_sd3_my_findings_on/)，这可能为 SD3 是如何以及在哪里被修改的提供见解。

**Llama 与本地 LLM 模型**

- **Llama 3 Spellbound**: 在 /r/LocalLLaMA 中，[Llama 3 7B 微调版在没有指令示例的情况下进行了训练](https://www.reddit.com/r/LocalLLaMA/comments/1dh5ee0/llama_3_spellbound_pretraining_over_sft_for/)，旨在保留对世界的理解和创造力，同时减少写作中的正面偏差 (positivity bias)。
- **NSFW 角色扮演模型**: 有人[请求推荐能在 3060 12GB GPU 上运行的模型](https://www.reddit.com/r/LocalLLaMA/comments/1dh5fsq/are_there_any_models_that_i_can_run_on_my_3060/)，并能生成类似于所提供示例的 NSFW 角色扮演内容。
- **类似于 Command-R 的模型**: 有人正在[寻找质量与 Command-R 相似的模型](https://www.reddit.com/r/LocalLLaMA/comments/1dhllgv/any_model_with_similar_quality_to_commandr_but/)，但要求在配备 M3 Max 64GB 的 Mac 上运行 64k 上下文时占用更少的内存。
- **用于角色扮演/聊天/故事创作的系统提示词**: 分享了一个[用于在角色扮演、聊天和故事创作场景中控制模型的详细系统提示词 (System Prompt)](https://www.reddit.com/r/LocalLLaMA/comments/1dhluqd/a_detailed_system_prompt_for_reining_models_in/)，重点在于详尽、直接且符号化的指令。
- **在 24GB 显存上运行大型模型**: 寻求关于[如何在 24GB 显存 (VRAM) 上运行大型模型/长上下文](https://www.reddit.com/r/LocalLLaMA/comments/1dhj5e7/im_dumb_how_are_you_guys_running_high/)的指导，可能涉及使用量化 (Quantization) 或 4/8 bit 精度。
- **RAG 中底层模型的重要性**: 讨论了在拥有可靠语料库的情况下，[使用检索增强生成 (RAG) 时底层模型是否重要](https://www.reddit.com/r/LocalLLaMA/comments/1dh6knt/does_the_underlying_model_matter_if_using_rag/)。

**AI Ethics and Regulation**

- **OpenAI 董事会任命批评**: [Edward Snowden 批评 OpenAI 任命前 NSA 局长的决定](https://fortune.com/2024/06/14/edward-snowden-eviscerates-openai-paul-nakasone-board-directors-decision/)，称其为“对地球上每个人权利的蓄意、精心策划的背叛”。
- **Stability AI 的闭源策略**: 在 /r/StableDiffusion 中，有一场关于 [Stability AI 决定走闭源 API 销售路线的讨论](https://www.reddit.com/r/StableDiffusion/comments/1dhh6z6/why_cant_sai_understand_their_advantage_is/)，质疑他们在不利用社区微调 (fine-tunes) 的情况下是否具有竞争力。
- **Stable Diffusion 服务条款澄清**: 针对一名标题党 YouTuber 造成的误解，提供了 [Stable Diffusion 模型服务条款 (TOS) 的澄清](https://www.reddit.com/r/StableDiffusion/comments/1dh9buc/to_all_the_people_misunderstanding_the_tos/)。
- **SD3 的众筹开源替代方案**: 有人建议[发起一个众筹的开源替代方案来取代 SD3](https://www.reddit.com/r/StableDiffusion/comments/1dhjal0/would_be_great_if_we_can_start_a_crowdfunded/)，该项目可能由一名曾参与训练 SD3 但最近辞职的前 Stability AI 员工领导。
- **GitHub 上的恶意 Stable Diffusion 工具**: 一篇[新闻报道称黑客利用 GitHub 上的恶意 Stable Diffusion 工具瞄准 AI 用户](https://www.404media.co/hackers-target-ai-users-with-malicious-stable-diffusion-tool-on-github/)，声称是为了抗议“艺术盗窃”，但实际上是通过勒索软件谋取经济利益。
- **去偏见对创造力的影响**: 一篇[研究论文讨论了对语言模型进行去偏见 (Debiasing) 处理对其创造力的影响](https://arxiv.org/abs/2406.05587)，认为审查模型会降低其创造力。

**AI and the Future**

- **在 AI 进步中感到迷茫**: 在 /r/singularity 中，分享了一段[关于在 AI 飞速进步面前感到迷茫和对未来不确定的个人反思](https://www.reddit.com/r/singularity/comments/1dh5aqd/feeling_lost/)。
- **对 AI 影响职业生涯的担忧**: 同样在 /r/singularity，有人表达了[鉴于近期发展，对 AI 未来及其职业生涯感到迷茫](https://www.reddit.com/r/singularity/comments/1dh5aq.)。

---

# AI Discord 回顾

> 摘要之摘要的摘要

**1. AI 模型性能与扩展**

- **使用新 AI 模型进行扩展**：据报道，DeepSeek 的 **[Coder V2](https://vxtwitter.com/teortaxesTex/status/1802681431992213767)** 在基准测试中击败了 GPT-4；Google DeepMind 展示了新的 video-to-audio 技术，能为任何视频创作配乐，该消息在 [Rowan Cheung 的 X 个人资料](https://x.com/rowancheung/status/1802734770117333257)上引发关注。
- **跨平台扩展 AI 能力**：Runway 推出了用于视频生成的 Gen-3 Alpha，增强了电影风格和场景过渡。相关细节分享在 [Twitter](https://x.com/runwayml/status/1802691475391566108?s=46&t=90xQ8sGy63D2OtiaoGJuww) 上。

**2. 跨平台的集成与实现**

- **混合笔记应用发布 LLM 集成**：OpenRouter 发布了一款集成 LLM 的笔记应用，用于动态内容交互，但正如其[全屏应用](https://000700836.deployed.codepen.website/)所述，目前缺乏移动端支持。
- **在不同平台上的实现挑战**：用户面临诸如 OpenRouter 上的 CORS 错误以及 LangChain 上的集成挑战，这反映出需要更好的实现指南或特定平台的 API。

**3. AI 伦理与治理**

- **OpenAI 转向利润驱动模式**：关于 OpenAI 转向营利实体的推测和确认不断涌现，这可能会影响治理和伦理考量。更多信息见 [The Information](https://www.theinformation.com/articles/openai-ceo-says-company-could-become-benefit-corporation-akin-to-rivals-anthropic-xai?utm_campaign=Editorial&utm_content=Article&utm_medium=organic_social&utm_source=twitter)。
- **AI 伦理讨论升温**：关于 AI 数据隐私、模型偏见和公司治理的辩论仍在继续，Edward Snowden 在 [Edward Snowden 的 X 个人资料](https://x.com/Snowden/status/1801610725229498403)上批评了 OpenAI 的新董事会任命。

**4. 新 AI 进展与基准测试**

- **AI 创新与改进发布**：Anthropic 在其新的[研究文章](https://anthropic.com/research/reward-tampering)中发表了关于 AI 篡改奖励系统能力的见解。
- **新模型基准测试**：Stability AI 发布了 SD3 模型，并在各大论坛讨论了损失稳定（loss stabilization）和伪影管理（artifacts management）的新技术，其中包括 [Reddit](https://www.reddit.com/r/StableDiffusion/comments/1dhd7vz/the_developer_of_comfy_who_also_helped_train_some/#lightbox) 上的热点讨论。

**5. 协作式 AI 项目与用户参与**

- **社区项目凸显 AI 集成**：从 OpenRouter 上合并笔记与 API 密钥管理的笔记应用，到像 Dream Machine 这样创新的 AI 驱动视频生成工具，社区构建的工具正在推向创意和实用 AI 应用的边界，详见 [Lumalabs](https://lumalabs.ai/dream-machine) 等平台。
- **互动式 AI 讨论与合作蓬勃发展**：网络研讨会和协作活动（如即将举行的 Mojo 社区会议）鼓励对 AI 进展进行深入研究，正如 [blog](https://www.modular.com/blog/whats-new-in-mojo-24-4-improved-collections-new-traits-os-module-features-and-core-language-enhancements) 所分享的，全球用户群参与度极高并进行了详细讨论。

---

# 第一部分：高层级 Discord 摘要

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SD3 许可证风波**：**Stable Diffusion 3 (SD3)** 的新许可证由于法律模糊性导致其在 Civitai 上被禁，Civitai 法务团队在其[临时禁令声明](https://civitai.com/articles/5732)中宣布了对此的审查。
- **社区因 SD3 产生分歧**：用户对 Stability AI 的 SD3 授权方式表示沮丧，既有困惑也有不满；同时，一些人批评 YouTuber Olivio Sarikas 为了流量歪曲 SD3 许可证，并引用了他的[视频](https://www.youtube.com/watch?v=HmxbsQDRTSs)。
- **ComfyUI 指南**：围绕 **ComfyUI** 设置的问题引发了技术讨论，建议的自定义节点安装修复方案包括 cv2 等依赖项；社区贡献的 [ComfyUI 教程](https://www.youtube.com/watch?v=Di1KqPXxx2Y&t=30s)也被分享以提供帮助。
- **寻找 SD3 替代方案**：对话指向了寻找替代模型和艺术工具的趋势，例如使用 animatediff 进行视频生成，这可能是由于持续的 SD3 争议所致。
- **AI 社区中的虚假信息指控**：针对 YouTuber Olivio Sarikas 散布有关 SD3 许可证虚假信息的指控不断，社区成员对其[争议视频](https://www.youtube.com/watch?v=HmxbsQDRTSs&t=2s&ab_channel=OlivioSarikas)内容的真实性提出了质疑。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Ollama 集成接近完成**：Ollama 支持开发已完成 80%，*Unsloth AI 团队*与 Ollama 正在协作克服延迟。讨论了关于 Ollama 的模板微调验证和学习率问题，以及运行 `model.push_to_hub_merged` 时无法保存完整合并模型的问题，并提出了手动解决方法。
  
- **Unsloth 持续提速**：基准测试显示，在 NVIDIA GeForce RTX 4090 上，Unsloth 的训练过程比 torch.compile() torchtune 快 24%，训练速度令人印象深刻。此外，即将推出的支持多达 8 个 GPU 的多 GPU 支持正在测试中，部分选定用户已获得早期访问权限进行初步评估。

- **训练难题与技巧**：成员们在训练 Yi 模型时遇到了保存步骤崩溃、保存过程中 `quantization_method` 管理不当，以及关于批次大小（batch sizes）和梯度累积（gradient accumulation）对 VRAM 占用影响的困惑。解决方案和变通方法包括验证内存/磁盘资源，以及一个解决量化错误的[已提交 Pull Request](https://github.com/unslothai/unsloth/pull/651)。

- **关于音乐怀旧与新奇的活跃讨论**：成员们分享了从 1962 年的怀旧歌曲到 Daft Punk 和 Darude 的经典曲目，展示了社区轻松的一面。相比之下，针对 AI Studio 上 Gemma 2 的输出存在担忧，反应不一，既有失望也有好奇，并对 Gemini 2.0 充满期待。

- **CryptGPT 通过加密方式保护 LLM**：介绍了一个名为 CryptGPT 的概念，它使用 Vigenere 密码在加密数据集上预训练 GPT-2 模型，确保隐私并需要加密密钥才能生成输出，详见分享的 [Blog Post](https://x.com/diwanksingh/status/1802118343446724655)。

- **单一的好奇信息**：社区协作频道出现了一条表达兴趣的消息，但由于缺乏进一步的上下文或细节，其与更广泛讨论话题的相关性尚不明确。



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **NVIDIA 下一代产品预测及 PyCUDA SM 查询澄清**：工程师们推测了即将推出的 NVIDIA 5090 GPU 的潜在规格，传闻 VRAM 高达 64 GB，但遭到了质疑。此外，澄清了 techpowerup 报告的 A10G 显卡 GPU SM 数量差异，Amazon Web Services 等独立来源确认正确数量为 80，而非最初陈述的 72。

- **Triton 和 Torch 用户应对故障与限制**：Triton 用户在 Colab 中遇到了 `AttributeError`，并讨论了使用嵌套归约（nested reductions）处理象限的可行性。同时， PyTorch 用户调整了 `torch.compile(mode="max-autotune")` 中的 SM 阈值以适应 SM 数量少于 68 的 GPU，并探索启用坐标下降调优（coordinate descent tuning）以获得更好性能。

- **软件与算法突破 AI 极限**：一位成员赞扬了 GPT-4 与 LLaMA 3 8B 的匹配表现；Akim 将参加 AI_dev 会议并欢迎交流。此外，Vayuda 的搜索算法论文引起了爱好者的兴趣，并在多个频道进行了讨论。关于 AI 训练的讨论（如 Meta 描述的 LLM 训练挑战）强调了基础设施适应性的重要性。

- **CUDA 开发动态**：来自以 CUDA 为核心的开发消息透露：Permuted DataLoader 集成并未显著影响性能；为随机舍入（stochastic rounding）开发了独特的种子策略；ZeRO-2 的内存开销（memory overhead）出现了挑战；新的 LayerNorm 内核在特定配置下提供了急需的加速。

- **超越 CUDA：动态批处理、量化和位打包**：在并行计算领域，工程师们正在努力解决 Gaudi 架构的动态批处理（dynamic batching）问题，并讨论了量化和位打包（bit-packing）技术的复杂性。他们强调了 VRAM 限制约束了大模型在本地的部署，并分享了多种资源，包括 Python 开发环境链接和新型机器学习库的文档。



---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 为工程师配备了 CLI 工具**：最新的 [LM Studio 0.2.22](https://lmstudio.ai) 版本引入了 'lms'，这是一个用于模型管理和调试 prompt 的 CLI 工具，详见其 [GitHub 仓库](https://github.com/lmstudio-ai/lms)。此次更新简化了 AI 部署的工作流，特别是在模型加载/卸载和输入检查方面。

- **性能优化与故障排除**：工程师们讨论了 AI 模型性能的最佳设置，包括 Intel ARC A7700 的 GPU 支持故障排除、GPU layers 的配置调整以及 Flash Attention 设置的调整。建议在遇到托管本地模型的问题时查看 [Open Interpreter 的文档](https://docs.openinterpreter.com/language-models/local-models/lm-studio)，并呼吁 LM Studio 界面更好地处理字体大小以提高可用性。

- **多样化的模型参与**：成员们推荐将 [Fimbulvetr-11B](https://huggingface.co/DavidAU/Fimbulvetr-11B-Ultra-Quality-plus-imatrix-GGUF) 用于角色扮演场景，同时强调了 **DeepSeek-Coder-V2** 等代码模型的快速更迭，建议同行关注用于特定任务（如编程）的最新模型，这些模型可以在 [Large and Small Language Models list](https.wikipedia://llm.extractum.io/list) 等网站上查看。

- **硬件优化与问题**：为遇到安装问题的用户分享了存档的 LM Studio 0.2.23 链接——一个 [MirrorCreator 链接](https://mir.cr/E1WVBIOO)。硬件讨论还包括混合 RAM 条的兼容性、服务器模式下的 CPU 核心设置，以及在各种系统上排除 GPU 检测故障。

- **开发见解与 API 交互**：开发者们分享了将 `llama3` 和 `deepseek-coder` 等各种编程模型集成到 VSCode 工作流中的愿景，并寻求在 `continue.dev` 中实现模型的帮助。此外，还有关于将 ROCm 从 LM Studio 主应用中解耦的讨论，以及一份 [使用 LM Studio 配置 `continue.dev`](https://docs.continue.dev/reference/Model%20Providers/lmstudio) 的用户指南。

- **Beta 版本观察与应用版本管理**：社区测试并回顾了最近的 Beta 版本，讨论了 tokenizer 修复和 GPU offloading 故障。用户需要获取旧版本的途径，但这受到 LM Studio 更新策略的挑战，因此有人建议保留个人偏好版本的存档。

- **AI 驱动的创意与易用性问题**：工程师们提出了 LM Studio 对 stop tokens 管理不善以及工具倾向于在输出中附加无关文本的问题。一个常见的与用例相关的投诉是，AI 模型在必要时没有通过使用 *#ERROR* 消息来表明其未能提供正确的输出。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**低端硬件上 GPT-4 的 AI 替代方案**：用户讨论了适用于性能较低服务器的实用 AI 模型，建议使用 **"llama3 (70B-7B), mixtral 8x7B, 或 command r+"** 来构建类似于 GPT-4 的自托管 AI。

**RWKV-TS 挑战 RNN 的主导地位**：一篇 [arXiv 论文](https://arxiv.org/abs/2401.09093)介绍了 **RWKV-TS**，提出它在时间序列预测中是比 RNN 更高效的替代方案，能有效捕捉长期依赖关系并具有计算扩展性。

**商业用途中的模型选择至关重要**：在为商业应用选择 AI 时，考虑使用场景、工具和部署限制至关重要，即使受到 7B 模型大小的限制。为了获得量身定制的建议，成员建议关注具体细节。

**创新与集成层出不穷**：从 [Difoosion](https://gitlab.com/mad-moo/difoosion)（一个用于 Stable Diffusion 的用户友好 Web 界面）到 *Ask Steve*（一个旨在利用 LLM 简化 Web 任务的 Chrome 扩展），社区成员正积极将 AI 集成到实用的工具和工作流中。

**模型处理与微调中的问题与建议**：
- 分享了一个 *[BERT 微调教程](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb)*。
- 提出了对非确定性模型初始化的担忧，并建议保存模型状态以实现可复现性。
- **Mistral-7b-0.3** 的上下文长度处理以及对高质量 **meme generator models** 的追求，表明了模型定制中的挑战与探索。
- 对于 TPU 用户，正在寻求关于在 **GCP's TPU** 上使用 **Diffusers** 的指导，表明了利用云端 TPU 运行扩散模型的兴趣。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **iOS 兼容性疑问**：成员们讨论了 **ChatGPT** 是否能在 iOS 18 beta 上运行，建议坚持使用 iOS 17 等稳定版本，并指出 beta 用户受限于关于新功能的 NDA。关于兼容性尚未达成明确共识。

- **开源崛起**：[DeepSeek AI](https://x.com/deepseek_ai/status/1802680388256768145) 发布的一款开源模型在编程和数学方面超越了 **GPT-4 Turbo**，引发了关于开源 AI 相比于私有模型优势的辩论。

- **使用 LLM 进行数据库部署**：为了获得更好的语义搜索并减少幻觉，一位社区成员推荐了 [OpenAI's Cookbook](https://cookbook.openai.com/examples/vector_databases/readme) 作为将向量数据库与 OpenAI 模型集成的资源。

- **GPT-4 使用起伏**：用户对访问 GPT 交互、Custom GPTs 的隐私设置以及服务器停机表示沮丧。社区提供了变通方案，并建议监控 [OpenAI's service status](https://status.openai.com) 以获取更新。

- **3D 建模与 Prompt Engineering 的挑战**：对话集中在生成无阴影 3D 模型的各种技术细节，以及防止 GPT-4 混淆信息的复杂性。成员们分享了各种策略，包括 step-back prompting 和设置明确的操作来引导 AI 输出。



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **稳定 SD3 模型**：讨论围绕 **SD3 models** 面临的稳定性障碍展开，特别是伪影和训练问题。人们对 **loss stabilization** 提出了担忧，指出了诸如 *non-uniform timestep sampling* 和缺失 **qk norm** 等问题。

- **T2I 模型登场**：对话强调了对 **开源 T2I (text-to-image) 模型** 的兴趣，特别是跨场景的角色一致性。对于寻求可靠多轮图像生成的开发者，推荐了 [Awesome-Controllable-T2I-Diffusion-Models](https://github.com/PRIV-Creation/Awesome-Controllable-T2I-Diffusion-Models) 和 [Theatergen](https://github.com/donahowe/Theatergen) 等资源。

- **逻辑极限突破**：一位成员关注了当前 AI 在 **logical reasoning** 方面面临的挑战，指出了 **Phi-2** 的“严重推理崩溃”以及 LLM 在处理 AIW 问题时的命名偏差——这一关键点得到了[相关研究](https://arxiv.org/abs/2406.02061)的支持。

- **提升演绎推理**：关于增强 LLM 演绎推理的混合方法的咨询指向了 [Logic-LM](https://arxiv.org/abs/2305.12295)，这是一种将 LLM 与符号 AI 求解器相结合以提高逻辑问题解决能力的方法。

- **视频生成创新**：复旦大学的 **Hallo model** 引发了关注，这是一款能够从单张图像和音频生成视频的工具，具有与 Text-to-Speech 系统协同应用的潜力。来自 [FXTwitter](https://fxtwitter.com/cocktailpeanut/status/1802376983021580428) 的一个本地运行工具被分享，凸显了社区对实际集成的兴趣。



---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **200T 参数模型：AGI 还是幻想？**：讨论了假设的 **200T 参数模型** 的可访问性，强调了当前计算能力对大多数用户的限制，并幽默地看待了为此类模型冠以 AGI 之名的说法。

- **在大模型竞技场中竞争**：成员们对比了 **Qwen7B** 和 **Llama3 8B** 模型，公认 **Llama3 8B** 是性能上的主导竞争者。针对 Llama3 模型的自定义训练配置问题得到了解决，并分享了一个[解决方案](https://github.com/xzuyn/axolotl/blob/dan_metharme/src/axolotl/prompt_strategies/customllama3.py)以处理 `chat_template` 设置问题。

- **PyTorch GPU 优化探索**：针对 PyTorch 中各种 GPU 设置的优化反馈请求，汇集了从 AMD MI300X 到 RTX 3090、Google TPU v4 以及运行 tinygrad 的 4090 等丰富的社区经验。

- **在 Axolotl 的开发迷宫中穿行**：发现并追踪到一个阻碍 **Llama3 模型** 开发的问题，该问题指向一个特定的 commit，这有助于识别问题，但也强调了在 main branch 中进行修复的必要性。为用户详细说明了在 **Axolotl** 中设置推理参数和微调视觉模型的指令。

- **结构化数据提取的新尝试**：社区展示暗示了使用 **Axolotl** 微调 LLM 后的积极成果，特别是在将非结构化新闻稿转换为结构化输出方面。即将发布的一篇文章承诺将阐述如何利用 OpenAI API 的 function calling 来提高 LLM 在此任务中的准确性。作者指向一篇[详细文章](https://mlops.systems/posts/2024-06-15-isafpr-first-finetune.html)以获取更多信息。



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Pro 语言合作伙伴关系！**：Perplexity AI 与 **SoftBank** 达成协议，向 SoftBank 客户免费提供一年的 **Perplexity Pro**。这项通常每年耗资 29,500 日元的优质服务旨在通过 AI 增强用户的探索和学习体验（[更多合作信息](https://pplx.ai/softbank)）。

- **规避 AB 测试协议？再想想**：工程师们讨论了如何绕过 **Agentic Pro Search** 的 A/B 测试，并提供了一个 Reddit 链接；然而，出于对诚信的担忧，他们进行了重新考虑。社区还处理了大量关于 **Perplexity 功能** 的使用问题，辩论了订阅 Perplexity 与 ChatGPT 的优劣，并提出了关于网络爬虫实践的关键隐私问题。

- **API 访问是关键**：成员们表达了对 Perplexity API 封闭测试访问权限的紧迫需求，强调了这对启动如 **Kalshi** 等项目的影响。在解决 Custom GPT 问题时，他们交流了通过使用基于 schema 的解释和错误详情来增强其“问任何事”功能的技巧，以改进 action/function call 的处理。

- **社区泄露与分享**：传阅了指向 **Perplexity AI 搜索** 和页面的链接，主题涵盖从数据表管理工具（Tanstack Table）到俄罗斯宠物食品市场以及大象交流策略。一份关于前列腺健康的公开个人文件引发的小事故，在社区驱动的支持下得到了解决。

- **游戏与研究的碰撞**：社区内分享的内容融合了学术兴趣和游戏文化，例如公开发布的关于 **The Elder Scrolls** 的页面，暗示了相关技术受众的交叉爱好。



---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **用神经元玩 Doom**：一种将生物技术与游戏结合的创新方法，利用活体神经元来玩视频游戏 **Doom**，详情见 [YouTube 视频](https://youtu.be/c-pWliufu6U)。这可能是理解生物过程与数字系统集成迈出的一步。

- **AI 伦理与偏见备受关注**：[ResearchGate 论文](https://www.researchgate.net/publication/378191925_Automated_Bias_and_Indoctrination_at_Scale_Is_All_You_Need)中对 AI 的批判性观点引起了人们对 AI 传播人类偏见和对齐企业利益这一趋势的关注，并将“随机鹦鹉（stochastic parrots）”称为潜在的认知操纵工具。

- **LLM 合并与 MoE 担忧**：关于 **Mixture of Experts (MoE)** 模型实际应用的激烈辩论浮出水面，探讨了模型合并与全面微调的有效性，并引用了 [llama.cpp](https://github.com/ggerganov/llama.cpp/pull/6453) 的 PR 以及 [Hugging Face](https://huggingface.co/Kquant03/CognitiveFusion-4x7B-bf16-MoE) 上的 MoE 模型。

- **Llama3 8B 部署挑战**：在设置和部署 **Llama3 8B** 方面，建议利用 **unsloth qlora**、**Axolotl** 和 **Llamafactory** 进行训练，并使用 **lmstudio** 或 **Ollama** 在 Apple 的 M2 Ultra 上运行快速的 OAI 兼容端点，这为模型部署工具提供了参考。

- **Autechre 的曲目引发争论**：围绕 Autechre 音乐的观点和情感导致了两个对比鲜明的 YouTube 视频分享：["Gantz Graf"](https://www.youtube.com/watch?v=ev3vENli7wQ) 和 ["Altibzz"](https://www.youtube.com/watch?v=m3ZyEGTIsvE)，展示了这对电子音乐组合创作的多样化听觉景观。

- **探索多人 AI 世界构建**：提出了在 **WorldSim** 中进行协作创作的建议，成员们讨论了为 AI 辅助的协作体验启用多人游戏功能，同时指出模型提供商的审查可能会影响 WorldSim AI 的内容。

- **NVIDIA 的 LLM 亮相**：NVIDIA 的 **Nemotron-4-340B-Instruct** 模型已发布，可在 [Hugging Face](https://huggingface.co/nvidia/Nemotron-4-340B-Instruct) 上获取，引发了关于合成数据生成和战略合作伙伴关系的讨论，突显了该公司在语言处理领域的新进展。

- **OpenAI 转向营利模式**：OpenAI 首席执行官 [Sam AltBody](https://x.com/aaronpholmes/status/1801785687030829240?s=46) 表示，公司可能会从非营利性质转向营利性架构，从而与竞争对手更加一致，并影响 AI 行业的组织动态和未来轨迹。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 函数讨论升温**：工程师们批评了 Mojo 手册中对 `def` 和 `fn` 函数的处理方式，指出了英语表述中的歧义以及这些函数变体对类型声明的影响。这达成了一个共识：虽然 `def` 函数允许可选的类型声明，但 `fn` 函数强制要求类型声明；这一细微差别影响了代码的灵活性和类型安全性。

- **聚会提醒：Mojo 社区集会**：宣布了即将举行的 *Mojo Community Meeting*，包含关于 constraints、Lightbug 和 Python 互操作性的演讲，邀请参与者[通过 Zoom 加入](https://modul.ar/community-meeting-zoom)。此外，基准测试显示 Mojo 的 Lightbug 在单线程性能上超过了 Python FastAPI，但仍逊于 Rust Actix，这引发了关于函数着色（function coloring）决策带来的潜在运行时开销的进一步讨论。

- **Mojo 24.4 正式发布**：Mojo 团队推出了 24.4 版本，引入了核心语言和标准库的改进。建议关注细节的工程师阅读[博客文章](https://www.modular.com/blog/whats-new-in-mojo-24-4-improved-collections-new-traits-os-module-features-and-core-language-enhancements)，深入了解新的 traits、OS 模块功能等。

- **揭秘 Mojo 高级技术**：深入的技术讨论揭示了 Mojo 编程中的挑战和见解，从处理 2D Numpy 数组、利用 `DTypePointer` 进行高效 SIMD 操作，到解决无符号整数转换中的 Bug。值得注意的是，CRC32 表初始化中涉及 `alias` 使用的差异引发了对意外转换行为的调查。

- **Nightly Mojo 编译器即将到来**：工程师们获悉了 Mojo 编译器的全新 **nightly builds**，发布了 `2024.6.1505`、`2024.6.1605` 和 `2024.6.1705` 版本，并提供了通过 `modular update` 进行更新的说明。每个版本的具体细节可以通过提供的 GitHub diffs 进行查看，展示了平台的持续改进。此外，有人注意到内置 MLIR dialects 缺乏外部文档，并且有人请求增加如 REPL 中直接输出表达式等增强功能。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Eleuther 对 OpenAI 泛化技术的复现**：EleutherAI 的可解释性团队成功在 21 个 NLP 数据集上的开源 LLM 中复现了 OpenAI 的 "weak-to-strong" 泛化技术，并在[此处](https://blog.eleuther.ai/weak-to-strong/)发布了关于实验变体（如 strong-to-strong 训练和基于探针的方法）的详细发现，包括正面和负面的结果。

- **工作机会与 CommonCrawl 导航**：AI Safety Institute 在其[招聘页面](https://aisi.gov.uk/careers)宣布了提供英国搬迁签证支持的新职位；同时，在关于高效处理 CommonCrawl 数据的讨论中提到了 [ccget](https://github.com/allenai/ccget) 和 [resiliparse](https://resiliparse.chatnoir.eu/en/latest/man/parse/html.html) 等工具。

- **模型创新与担忧**：从探索 **RWKV-CLIP**（一种视觉语言模型），到对 Diffusion 模型生成内容以及商业模型输出被窃取的担忧，社区探讨了 AI 模型开发与安全的各个方面。**Laprop** 优化器的有效性引发了辩论，分享了从在线适配到“窃取”嵌入模型的各类论文，其中一篇关键论文见[此处](https://arxiv.org/abs/2406.09355)。

- **演进中的优化与 Scaling Laws**：成员对一篇基于 Hypernetwork 论文的批评引发了关于 Hypernetwork 与 Hopfield 网络价值及对比的对话。感兴趣的各方深入探讨了 Scaling Laws 的扩展，考虑了 LLM 的在线适配，并引用了 Andy L. Jones 关于用训练计算抵消推理计算的概念。

- **Sparse Autoencoders 的可解释性洞察**：可解释性研究集中在 Sparse Autoencoders，一篇论文提出了在 GPT-2 的间接对象识别等任务中评估特征字典的框架，另一篇则强调了分解 Logit 输出组件的 "logit prisms"，详见[这篇文章](https://neuralblog.github.io/logit-prisms)。

- **对模型评估共享平台的需求**：社区呼吁建立一个共享和验证 AI 模型评估结果的平台，特别是针对使用 Hugging Face 并寻求验证闭源模型可信度的用户，强调了对全面且透明的评估指标的需求。

- **等待视觉语言项目的代码发布**：针对 **RWKV-CLIP** 相关代码发布日期的具体请求已定向至该项目的 [GitHub Issues 页面](https://github.com/deepglint/RWKV-CLIP/issues)，显示出对获取最新视觉语言表示模型进展的需求。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **苹果在 AI 领域避开 NVIDIA**：苹果在 WWDC 上的发布详情显示其避开了 NVIDIA 硬件，更倾向于在 TPU 和 Apple Silicon 上使用其内部的 AXLearn，这可能会彻底改变其 AI 开发策略。技术细节在 [Trail of Bits 博客文章](https://blog.trailofbits.com/2024/06/14/understanding-apples-on-device-and-server-foundations-model-release/)中进行了拆解。

- **嵌入与微调**：微调方法论受到热捧，讨论范围从嵌入的复杂性（如 [Awesome Embeddings](https://github.com/eifuentes/awesome-embeddings) 资源所示）到特定实践（如针对独特叙事风格适配 TinyLlama，详见开发者的[博客文章](https://gabrielchua.me/posts/finetuning-tinyllama-axolotl-beginner/)）。

- **Prompt 构建创新**：提到 **Promptfoo** 和 **inspect-ai** 表明了向更复杂的 Prompt Engineering 工具发展的趋势，社区正在权衡其功能性和易用性。不同的偏好表明此类工具对于精细化的人机交互方案至关重要。

- **积分混淆已澄清**：参与者对 **LangSmith** 和 **Replicate** 等平台上的课程积分表达了混淆，通过社区支持进行了提醒和澄清。为相关成员说明了测试版积分与课程积分之间的区别。

- **Code Llama 的飞跃**：[Code Llama](https://huggingface.co/blog/codellama) 发布引发的对话显示了对提高编程生产力的承诺。对 Hugging Face 和 GitHub 配置文件格式之间允许的差异表示好奇，这表明了微调这些专用模型所需的精确度。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Sakana AI 加入独角兽俱乐部**：Sakana AI 正在突破传统的 Transformer 模型，并从 NEA、Lux 和 Khosla 等重量级机构获得了高达 10 亿美元的估值，标志着 AI 社区的一个重要里程碑。详细的财务细节可以在[这篇文章](https://www.theinformation.com/articles/openais-japanese-rival-gets-1-billion-valuation-from-silicon-valley-investors)中查阅。

- **Runway Gen-3 Alpha 开启下一代视频生成**：Runway 的 Gen-3 Alpha 备受瞩目，展示了创建包含复杂场景转换和丰富电影风格的高质量视频的能力，为视频生成设定了新标杆，详情请点击[此处](http://runwayml.com/gen-3-alpha)探索。

- **DeepMind 视频转音频技术的突破**：Google DeepMind 的新视频转音频技术旨在通过为任何视频量身定制理论上无限数量的音轨，彻底改变无声 AI 视频生成，正如 [Rowan Cheung 的示例](https://x.com/rowancheung/status/1802734770117333257)所展示的那样。

- **Wayve 在视图合成方面的亮眼表现**：Wayve 凭借利用 4D Gaussians 的视图合成模型在 AI 领域取得了新胜利，有望在从静态图像生成新视角方面实现重大飞跃，详见 [Jon Barron 的推文](https://x.com/jon_barron/status/1802758455830437975)。

- **关于 OpenAI 未来的猜测**：有关 OpenAI 治理结构调整的传闻暗示其可能转向营利性立场，并考虑随后进行 IPO，这在社区内引发了激烈讨论；一些人对此表示嘲讽，而另一些人则在等待具体进展，正如 [The Information](https://www.theinformation.com/articles/openai-ceo-says-company-could-become-benefit-corporation-akin-to-rivals-anthropic-xai?utm_campaign=Editorial&utm_content=Article&utm_medium=organic_social&utm_source=twitter) 的报道以及 [Jacques Thibault 的推文](https://x.com/jacquesthibs/status/1801782465364640247?s=46)所回应的那样。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **RAG 与 Agent 概念详解**：分享了一份 [Excalidraw 增强型幻灯片](https://t.co/oibSiDjseO)，详细介绍了检索增强生成 (RAG) 和 Agent 的构建，其中的图表阐明了从基础到高级的概念。

- **LLM 应用中集成可观测性**：通过 Arize 集成，一个新的仪表化模块为 LLM 应用程序带来了端到端的可观测性，并提供了一份详细说明自定义事件/Span 处理器仪表化的[指南](https://t.co/cOBP9IOjro)。

- **知识图谱与 Neo4j 的结合**：关于将 Neo4j 知识图谱与 LlamaIndex 集成的讨论集中在将 Neo4j 图转换为 LlamaIndex 的属性图（Property Graphs）上，并提供了相关资源和文档（[LlamaIndex 属性图示例](https://docs.llamaindex.ai/en/latest/examples/property_graph/property_graph_neo4j/)）。

- **通过网页抓取策略增强 LLM**：一篇出版物讨论了通过结合网页抓取和 RAG 来改进 LLM，推荐使用 [Firecrawl](https://medium.com/ai-advances/how-to-power-up-llms-with-web-scraping-and-rag-975a165587f6) 等工具进行有效的 Markdown 提取，以及使用 Scrapfly 获取适用于 LLM 预处理的多样化输出格式。

- **实用教程与 AI 活动亮点**：发布了关于全栈 Agent 和多模态 RAG 流水线的实用分步指南；此外，[AI World's Fair 亮点](https://t.co/O6WAkbI9jt)中，知名演讲者分享了他们在 AI 和工程方面的知识，提升了社区的技能水平并加深了对新兴 AI 趋势的理解。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **脚本混乱与 OpenCL 困扰**：关于 `autogen_stubs.sh` 的讨论显示 `clang2py` 会破坏缩进，但发现这对于 GPU 加速的 tinygrad 操作并非必要。同时，George Hotz 建议修复 OpenCL 安装并使用 `clinfo` 进行验证，因为相关错误影响了 tinygrad 的 GPU 功能。

- **增强型 OpenCL 诊断即将到来**：改进 OpenCL 错误消息的工作正在进行中，一个[拟议的解决方案](https://github.com/tinygrad/tinygrad/pull/5004)可以从现有的 OpenCL headers 中自动生成消息，旨在简化开发者的调试过程。

- **解读梯度同步**：为了揭开梯度同步（gradient synchronization）的神秘面纱，George Hotz 肯定了 Tinygrad 在其 optimizer 中内置的解决方案，宣称其效率优于 PyTorch 中更复杂的 Distributed Data Parallel。

- **雄心勃勃追赶 PyTorch**：George Hotz 表达了让 tinygrad 在速度、简洁性和可靠性方面超越 PyTorch 的雄心。尽管目前在 LLM 训练等方面仍处于追赶状态，但 tinygrad 简洁的设计和强大的基础展现了巨大的潜力。

- **Kernel 领域的精度至关重要**：一次技术交流讨论了在模型中引入混合精度（mixed precision）的策略，George Hotz 建议采用 late casting 以获得效率提升，并使用 `cast_` 方法，强调了这是优化计算密集型任务的关键环节。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **GPT 笔记应用发布**：展示了一个 LLM 客户端与笔记应用的混合体，具有动态包含笔记的功能，采用原生 JavaScript 构建，并在浏览器中本地存储笔记和 API keys；不过，目前该应用尚不支持移动端。该应用通过 [Codepen](https://codepen.io/bulgakoff08/project/editor/DnJLrG) 和[全屏部署版本](https://000700836.deployed.codepen.website/)进行展示。

- **OpenRouter 的抱怨与见解**：OpenRouter 要求至少有一条用户消息以防止错误，用户建议使用 `prompt` 参数；推荐使用 [PDF.js 和 Jina AI Reader](https://jina.ai/reader/) 等格式化工具进行 PDF 预处理，以增强 LLM 的兼容性。

- **Qwen2 的审查困扰**：Qwen2 模型因过度审查面临用户批评，而限制较少的 Dolphin Qwen 2 模型因其更写实的叙事生成能力而获得推荐。

- **Gemini Flash 上下文冲突**：关于 Gemini Flash 的 token 限制出现了疑问，OpenRouter 列出的限制为 22k，而 Gemini Documentation 中提到的则是 8k tokens；这一差异归因于 OpenRouter 采用字符计数以与 Vertex AI 的计费方式保持一致。

- **速率限制与配置讨论**：用户讨论了 GPT-4o 和 Opus 等模型的速率限制以及模型性能配置；欲了解更多信息，OpenRouter 关于[速率限制的文档](https://openrouter.ai/docs/limits)提供了详尽说明，讨论重点在于 API 请求和使用的效率。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain API 更新导致 TextGen 损坏**：最近的 **API update** 破坏了 **LangChain** 中的 **textgen integration**，成员们正在通用频道寻求解决方案。

- **技术故障排除成为焦点**：用户讨论了安装 **langchain_postgres** 的挑战，以及由 **tenacity version 8.4.0** 更新引起的 **ModuleNotFoundError**；将版本回退到 **version 8.3.0** 修复了该问题。

- **LangChain 知识共享**：出现了围绕 **LangChain** 使用的问题，包括从 Python 转向 **JavaScript implementations**，以及如何处理 **Llama 3** 或 **Google Gemini** 等模型的本地部署。

- **技术爱好者介绍新酷工具**：一些创新项目受到关注，例如 **R2R 的自动知识图谱构建**、**Collision 事件的交互式地图**，以及 **CryptGPT**（一种使用 Vigenere 密码保护 LLM 隐私的方法）。

- **面向创意人士的 AI**：社区成员发布了一个用于**生成技术图表的新定制 GPT**，以及 **Rubik's AI**（一个研究助手和搜索引擎，向测试人员提供包含 **GPT-4 Turbo** 等模型的免费高级版）。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**OtterTune 退出舞台**：[OtterTuneAI 在收购交易失败后已关闭](https://x.com/andy_pavlo/status/1801687420330770841?s=46&t=Tc6nPt_FP2Ybqya6_6Xu-w)，标志着其自动数据库调优服务的终结。

**Apple 和 OpenAI 的新动作**：Apple 在 Hugging Face 上发布了优化的端侧模型，例如 [DETR Resnet50 Core ML](https://huggingface.co/apple/coreml-detr-semantic-segmentation)；而 OpenAI 因邀请前 NSA 局长 Paul M. Nakasone 加入董事会而遭到 Edward Snowden 的批评。

**DeepMind 恪守本分**：在近期的社区讨论中，相关人员澄清 DeepMind 并未参与特定的 AI 项目，打破了此前的猜测。

**Runway 和 Anthropic 持续创新**：Runway 在 [Twitter](https://x.com/runwayml/status/1802691475391566108?s=46&t=90xQ8sGy63D2OtiaoGJuww) 上发布了其全新的视频生成模型 Gen-3 Alpha；同时，Anthropic 在一篇[博客文章](https://anthropic.com/research/reward-tampering)中公布了关于 AI 模型攻击其奖励系统的重要研究。

**AI 协作与学习的未来**：Prime Intellect 将开源复杂模型 DiLoco 和 DiPaco；Bittensor 正在利用 The Horde 进行去中心化训练；此外，社区分享的一段 [YouTube 视频](https://www.youtube.com/watch?v=mdKjMPmcWjY&t=23s)深入解析了对模型训练至关重要的优化器（optimizers）。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **AGI：幻想还是未来？**：成员们分享了对一段[关于 AGI 的 YouTube 视频](https://youtu.be/4JF1V2hzGKE?si=gKCvVxBpwsGTD6ow)的看法，讨论了在怀疑态度与类似互联网泡沫破裂后可能出现的实质性进展之间如何取得平衡。

- **Next.js 迁移进行中**：社区正在协作推动在 Cohere toolkit 中使用 Next.js App Router，旨在提高代码的可移植性和社区贡献效率，详情见 [GitHub issue #219](https://github.com/cohere-ai/cohere-toolkit/issues/219)。

- **Cohere 的 C4AI**：Nick Frosst 邀请参加 C4AI 演讲（通过 [Google Meet 链接](https://meet.google.com/ibt-wsgv-kbq?hs=122&authuser=0)），为社区成员提供了参与 LLM 进展和应用讨论的渠道。

- **掌控你的浏览器**：一款[免费的 Chrome 扩展程序](https://www.asksteve.to)已发布，它将 **LLMs** 集成到 Chrome 中以提升生产力；同时，一个带有 AI 聊天功能的[交互式 Collision 地图](https://collision.talewind.ai/)展示了如何使用现代 Web 技术栈呈现活动。

- **开发者交流**：Cohere 正在举办由 David Stewart 主持的开发者办公时间（Developer Office Hours），深入探讨 API 和模型的复杂细节；感兴趣的社区成员可以从[这里](https://discord.gg/jy5XFg5GDh?event=1248300905703673987)加入，并在指定线程提交问题以获得专门支持。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **模型冻结之谜已解**：工程师们报告了模型在**编码过程中冻结**的情况，但事实证明耐心等待是有回报的，因为模型通常会完成任务，只是中间会有一个具有误导性的停顿。

- **技术支持重定向**：关于模型 **Windows 安装问题**的咨询被引导至特定的帮助频道，以便获得更具针对性的协助。

- **模型记忆功能得到提升**：一位成员庆祝了在**内存实现（memory implementation）**方面的突破，并用初步术语描述了其成功；同时，[Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1cal17l/llm_comparisontest_llama_3_instruct_70b_8b/)披露了 **Llama 3 Instruct 70b** 和 **8b** 的性能细节。

- **网络帽子（Cyber Hat）倒计时**：一个开源的、AI 赋能的“网络帽子”项目因其原创性和创新潜力引起了工程师们的兴趣，并公开邀请协作，[点击观看](https://www.youtube.com/watch?v=71p9DcGqNDc)；同样，[Dream Machine](https://lumalabs.ai/dream-machine) 基于文本和图像的逼真视频生成功能也标志着 AI 模型能力的巨大进步。

- **语义搜索协同**：讨论转向了将基于语音的**语义搜索与索引**与存储音频数据的向量数据库相结合，利用 LLM 的强大能力根据语音输入执行复杂任务，这预示着集成技术系统的潜在力量。

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **聚焦 Torchtune 的单节点优先级**：Torchtune 目前正专注于优化**单节点训练 (single node training)**，之后才会考虑多节点训练；它使用 `tune run` 命令作为 `torch run` 的封装，虽然在多节点环境下未经测试，但通过一些调整可能支持多节点设置。
  
- **解锁 Torchtune 的多节点潜力**：一些成员分享了如何配置 Torchtune 进行**多节点训练**的潜在方案，建议使用 `tune run —nnodes 2` 以及 **TorchX** 或 **slurm** 等额外工具来进行脚本执行和跨节点网络协调，并参考 [FullyShardedDataParallel 文档](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardingStrategy) 作为分片策略的资源。



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Llama3 坚守本源**：尽管推出了德语模型，但 **Llama3 tokenizer** 并未进行修改，与基础版 Llama3 完全一致，这引发了人们对其处理德语 Token 效率的质疑。
- **Token 讨论**：针对未改变的 Tokenizer 出现了一些担忧，工程师们推测，不加入特定的德语 Token *可能会大幅缩减上下文窗口 (context window)* 并影响 Embedding 的质量。
- **比较 Llama2 和 Llama3 的 Token 尺寸**：*好奇的成员*注意到 **Llama3 的 Tokenizer 显著比 Llama2 大 4 倍**，这引发了关于其对德语现有功效以及潜在未察觉问题的讨论。




---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

**预示数据工程的就业保障**：ChatGPT 在技术领域日益增长的角色引发了带有幽默色彩的评论，称其为数据工程师的“无限工作生成器”。

**Thoughtbot 拨开 LLM 的迷雾**：社区成员对 [Thoughtbot 的指南](https://thoughtbot.com/blog/understanding-open-source-llms) 表示赞赏，认为其清晰地剖析了 Large Language Models 的世界，特别是对 Base、Instruct 和 Chat 模型的划分，对初学者非常有帮助。

**搜索领域的新成员**：Turso 的最新版本将 [原生向量搜索 (native vector search)](https://turso.tech/blog/turso-brings-native-vector-search-to-sqlite) 集成到了 SQLite 中，旨在通过取代对 [sqlite-vss](https://github.com/asg017/sqlite-vss) 等独立扩展的需求，来提升 AI 产品开发体验。



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **寻找医院 AI 项目名称**：用户 gomiez 询问了 **AI Stack Devs** 社区内医院 AI 项目的名称。目前没有提供额外的上下文或回复来进一步确认该项目。



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Llama 作为 Firefox 的新搜索伴侣？**：社区成员 cryovolcano. 询问了将 **llamafile** 与 **tinyllama** 集成作为 Firefox 浏览器搜索引擎的可能性。目前尚未提供有关实现或可行性的更多细节或上下文。



---


**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该社区长时间保持沉默，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该社区长时间保持沉默，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该社区长时间保持沉默，请告知我们，我们将将其移除。


---


**YAIG (a16z Infra) Discord** 没有新消息。如果该社区长时间保持沉默，请告知我们，我们将将其移除。


---

# 第 2 部分：频道详细摘要与链接


{% if medium == 'web' %}




### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1251251945201729567)** (723 条消息🔥🔥🔥): 

_

- **对 SD3 许可协议缺乏信任引发混乱**：人们对 Stability AI 针对 SD3 的新许可协议存在重大担忧，由于法律条文模糊，导致 Civitai 暂时禁止了 SD3 相关内容。[Civitai 公告](https://civitai.com/articles/5732) 提到“法务团队审查”正在进行中。
- **社区的沮丧与批评者的强烈抵制**：许多用户对 Stability AI 令人困惑的许可协议以及对 SD3 发布的处理方式表示沮丧和批评。一位用户指出：*“迄今为止最糟糕的基础模型发布……我只是想要一双好看的手。”*
- **ComfyUI 中的咨询与故障排除**：几位用户讨论了 ComfyUI 设置的问题和修复方法，特别是关于自定义节点安装和 cv2 等依赖项。一位用户分享了一个有用的 [ComfyUI 安装教程](https://www.youtube.com/watch?v=Di1KqPXxx2Y&t=30s)。
- **对模型应用和替代方案的兴趣**：用户正在探索各种艺术风格和用途的模型，例如复古黑暗幻想以及使用 animatediff 工具生成视频。用户讨论暗示，在 SD3 争议之后，开源社区可能会将注意力转向替代模型和工具。
- **YouTuber Olivio Sarikas 面临审查**：多位用户讨论了这位 YouTuber 关于 SD3 许可协议的视频，指责他传播虚假信息并夸大对法律后果的恐惧，其中一人表示：*“Olivio 掌握了所有信息……却故意误报以骗取流量。”*
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://www.youtube.com/watch?v=r8"> - YouTube</a>: 未找到描述</li><li><a href="https://huggingface.co/PenelopeSystems/penelope-palette">PenelopeSystems/penelope-palette · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=HmxbsQDRTSs">SD3 - Absurd new License. Stability AI asks you to destroy your Models!</a>: Stability AI 的新 SD3 许可证要求你销毁你的模型。新的 Creator License 有一些相当荒谬的条款。包括限制你只能使用 6 个...</li><li><a href="https://www.youtube.com/w">YouTube</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=HmxbsQDRTSs&t=2s&ab_channel=OlivioSarikas">SD3 - Absurd new License. Stability AI asks you to destroy your Models!</a>: Stability AI 的新 SD3 许可证要求你销毁你的模型。新的 Creator License 有一些相当荒谬的条款。包括限制你只能使用 6 个...</li><li><a href="https://youtu.be/r8oA42saTNI?si=g0aIpBHftWleWpaW">Stable Diffusion 3&#39;s Concerning Fine Print: What Studios and Artists Should Know About the New Terms</a>: 我们查看了 Stable Diffusion 3 新许可证的细则，并为你分析了如果你计划将 SD3 用于商业或非商业用途时需要了解的内容...</li><li><a href="https://onnx.ai/">ONNX | Home</a>: 未找到描述</li><li><a href="https://youtu.be/HmxbsQDRTSs">SD3 - Absurd new License. Stability AI asks you to destroy your Models!</a>: Stability AI 的新 SD3 许可证要求你销毁你的模型。新的 Creator License 有一些相当荒谬的条款。包括限制你只能使用 6 个...</li><li><a href="https://dreamstudio.ai/generate>)">DreamStudio</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=Di1KqPXxx2Y&t=30s">SD3 IS HERE!! ComfyUI Workflow.</a>: SD3 终于登陆 ComfyUI 了！Topaz Labs: https://topazlabs.com/ref/2377/ 如何支持我的频道 - 通过加入我的 Patreon 来支持我：https://www.patreon.co...</li><li><a href="https://civitai.com/articles/5732">Temporary Stable Diffusion 3 Ban | Civitai</a>: 不幸的是，由于与 Stable Diffusion 3 相关的许可证缺乏清晰度，我们暂时禁止：所有基于 SD3 的模型，所有模...</li><li><a href="https://huggingface.co/ByteDance/Hyper-SD">ByteDance/Hyper-SD · Hugging Face</a>: 未找到描述</li><li><a href="https://hyper-sd.github.io">Hyper-SD: Trajectory Segmented Consistency Model for Efficient Image Synthesis</a>: 未找到描述</li><li><a href="https://github.com/Picsart-AI-Research/StreamingT2V?tab=readme-ov-file">GitHub - Picsart-AI-Research/StreamingT2V: StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text</a>: StreamingT2V: 来自文本的一致、动态且可扩展的长视频生成 - Picsart-AI-Research/StreamingT2V</li><li><a href="https://tenor.com/bjN7k.gif">Shoe Nike GIF - Shoe Nike Design Shoe - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/Fannovel16/comfyui_controlnet_aux">GitHub - Fannovel16/comfyui_controlnet_aux: ComfyUI&#39;s ControlNet Auxiliary Preprocessors</a>: ComfyUI 的 ControlNet 辅助预处理器。通过在 GitHub 上创建账户为 Fannovel16/comfyui_controlnet_aux 的开发做出贡献。</li><li><a href="https://x.com/xqdior/status/1801995625745252765">Tweet from D̷ELL (@xqdior)</a>: 大家已经体验了搭载 Stable Diffusion 3 8B 的 Stable Image API Ultra。我汇总了生成的图像并进行报告。请务必查看 Stable Diffusion 3 最高级模型的性能。 https://qiita.com/nqdior/items/ce894b5c5382b2029ced #Qiita</li><li><a href="https://imgur.com/gallery/sd-web-ui-3d-lora-ZkV6KkP">sd web ui 3d lora</a>: 在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过幽默的笑话、热门的梗、有趣的 GIF、励志故事、病毒视频等来振奋你的精神...</li><li><a href="https://fxtwitter.com/zhozho672070/status/1802037549864804820?s=46&t=AZs45ckJ7UUM_kJZcxnR_w">Tweet from -Zho- (@ZHOZHO672070)</a>: InstantX 刚刚又连续上传了 4 个 SD3 Medium 的 ControlNet 模型  Canny（1024）：https://huggingface.co/InstantX/SD3-Controlnet-Canny Pose：https://huggingface.co/InstantX/SD3-Controlnet-Pose Tile（还在上传）：https://huggingfa...</li><li><a href="https://github.com/comfyanonymous/ComfyUI/releases">Releases · comfyanonymous/ComfyUI</a>: 最强大且模块化的 Stable Diffusion GUI、API 和后端，具有图形/节点界面。 - comfyanonymous/ComfyUI</li><li><a href="https://stability.ai/creator-license-agreement">Professional Membership Agreement &mdash; Stability AI</a>: 未找到描述</li><li><a href="https://huggingface.co/ptx0/sd3-reality-mix">ptx0/sd3-reality-mix · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/nerijs/pixel-art-medium-128-v0.1">nerijs/pixel-art-medium-128-v0.1 · Hugging Face</a>: 未找到描述</li><li><a href="https://civitai.com/">Civitai: The Home of Open-Source Generative AI</a>: 探索 </li>

数千个高质量的 Stable Diffusion 模型，分享你的 AI 生成艺术，并与充满活力的创作者社区互动</li><li><a href="https://www.reddit.com/r/StableDiffusion/s/ZEV6SEsTHU">Reddit - 探索一切</a>：未找到描述</li><li><a href="https://civitai.com/models/4201/realistic-vision-v60-b1">Realistic Vision V6.0 B1 - V5.1 Hyper (VAE) | Stable Diffusion Checkpoint | Civitai</a>：使用 Hyper 模型的建议：Sampler = DPM SDE++ Karras 或其他 / 4-6+ steps CFG Scale = 1.5-2.0（数值越低，越多...）
</li>
</ul>

</div>
  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1251252320520372366)** (517 条消息🔥🔥🔥): 

<ul>
    <li><strong>Ollama 支持正在开发中：</strong> 一位成员表示：“不幸的是，Ollama 的支持进度有所延迟”，但保证他们正在“与优秀的 Ollama 团队合作”。目前支持进度约为 80%。</li>
    <li><strong>模板微调中的验证问题：</strong> 一位成员询问了关于验证用于 Ollama 的模板的问题，并讨论了学习率和模型配置的问题。他们指出：“我的合并模型效果还可以，但有时会出现异常。”</li>
    <li><strong>推送到 HF 合并模型的问题：</strong> 一位成员提出了一个问题，即运行 `model.push_to_hub_merged` 仅保存 Adapter 而不是完整的合并模型。另一位成员建议了一个变通方案，包括在上传前进行手动合并。</li>
    <li><strong>训练性能对比：</strong> 一位用户强调了 Unsloth 在训练速度方面的表现，根据他们的基准测试结果，声称在 4090 上“比 torch.compile() torchtune 快 24%”。Unsloth 团队对此表示认可，并讨论了为此发表学术论文的可能性。</li>
    <li><strong>即将推出的多 GPU 支持：</strong> 团队确认他们将实现最多支持 8 个 GPU 的多 GPU 支持。一小部分人正在获得早期访问权限以进行初步测试。</li>
</ul>

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://discord.gg/ollama?event=1251334371349233814">加入 Ollama Discord 服务器！</a>：查看 Discord 上的 Ollama 社区 - 与 49602 名其他成员一起交流，享受免费的语音和文字聊天。</li><li><a href="https://huggingface.co/nyunai/nyun-c2-llama3-50B">nyunai/nyun-c2-llama3-50B · Hugging Face</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2406.07394">通过 LLaMa-3 8B 的蒙特卡洛树自我细化（MCTSr）获取 GPT-4 级别的数学奥林匹克解法</a>：本文介绍了 MCT Self-Refine (MCTSr) 算法，这是一种将大语言模型 (LLMs) 与蒙特卡洛树搜索 (MCTS) 创新结合的方法，旨在增强在复杂数学...</li><li><a href="https://www.youtube.com/watch?v=FHsEW0HpuoU>">第 10 讲：构建生产级 CUDA 库</a>：幻灯片 https://drive.google.com/drive/folders/158V8BzGj-IkdXXDAdHPNwUzDLNmr971_?usp=sharing 演讲者：Oscar Amoros Huguet</li><li><a href="https://www.youtube.com/watch?v=JuPwfQlPUt0">KAN: Kolmogorov-Arnold Networks</a>：Google 算法研讨会技术演讲，由 Ziming Liu 主讲，2024-06-04。摘要：受 Kolmogorov-Arnold 表示定理启发，我们提出了 Kolmo...</li><li><a href="https://wandb.ai/augmxnt/train-bench/reports/torchtune-vs-axolotl-vs-unsloth-Trainer-Comparison--Vmlldzo4MzU3NTAx">torchtune vs axolotl vs unsloth 训练器性能对比</a>：各种训练器和 GPU 的性能对比。由 lhl 使用 Weights &amp; Biases 制作</li><li><a href="https://tenor.com/view/card-codes-gif-21814106">Card Codes GIF - Card Codes - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/ryanels4/status/1801898008755241251?s=46">来自 Ryan Els (@RyanEls4) 的推文</a>：AI 揭秘 😲</li><li><a href="https://github.com/unslothai/unsloth/issues/611">save_pretrained_merged 未合并模型 · Issue #611 · unslothai/unsloth</a>：问题描述：我的目标是将合并后的模型保存为 GGUF 文件，但我遇到了各种错误。更深层次的问题似乎是合并 LoRA+基础模型时没有保存合并文件。我认为...</li><li><a href="https://github.com/unslothai/unsloth/wiki">首页</a>：微调 Llama 3, Mistral, Phi &amp; Gemma LLMs，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://x.com/aiDotEngineer/status/1791506805065216017">来自 AI Engineer (@aiDotEngineer) 的推文</a>：我们很高兴宣布我们的演讲嘉宾！CEO @Modular AI, LEAD @MozillaAI, ENG LEAD @OpenAI, CEO @UnslothAI, TBA @Microsoft, TBA @AnthropicAI, CEO @cognition_labs (Devin), CEO @anysphere (@cursor_ai), CTO...</li><li><a href="https://github.com/unslothai/unsloth/tree/main">GitHub - unslothai/unsloth：微调 Llama 3, Mistral, Phi &amp; Gemma LLMs，速度提升 2-5 倍，显存占用减少 80%</a>：微调 Llama 3, Mistral, Phi &amp; Gemma LLMs，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/tree/main?tab=readme-ov-file#-key-features">GitHub - unslothai/unsloth：微调 Llama 3, Mistral, Phi &amp; Gemma LLMs，速度提升 2-5 倍，显存占用减少 80%</a>：微调 Llama 3, Mistral, Phi &amp; Gemma LLMs，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://colab.research.google.com/github/oobabooga/text-generation-webui/blob/main/Colab-TextGen-GPU.ipynb#scrollTo=LGQ8BiMuXMDG">Google Colab</a>：未找到描述</li><li><a href="https://unsloth.ai/blog/contpretraining?s=08">使用 Unsloth 进行 LLM 持续预训练</a>：通过使用 Unsloth 对 Llama 3, Phi-3 和 Mistral 进行持续预训练，让模型学习一种新语言。</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/psE4lqVDsO">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://docs.google.com/spreadsheets/d/115Utf5SuQOEOCQpYCeg_zB-nowej7joTkya6gYla5Hc">训练对比</a>：Sheet1 运行、训练器、GPU、训练时间 (h)、最大显存 (GiB)、功率 (W)、能耗 (kWh)、Tok/s、步数、优化器、最大序列长度、Batch Size、梯度累积、全局、备注 &lt;a href=&quot;https://wandb.ai/augmxnt/train-bench/runs/n59y6...</li><li><a href="http://wandb.ai/augmxnt/train-bench">augmxnt</a>：Weights &amp; Biases，机器学习开发者工具。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1251331569403170877)** (17 messages🔥): 

- **分享复古音乐视频**：一位成员发布了一个名为 "Not like us (1962) full song" 的 [YouTube 视频](https://youtu.be/er-juqsr4nc?si=g3vtzgUa94GEcD-s)，表达了他们对老派音乐风格的喜爱。另一位成员称赞了其品味，并幽默地提到自己只听过动漫歌曲。
- **Darude 的 Sandstorm 与音乐偏好**：一位成员开玩笑地分享了 [Darude - Sandstorm](https://www.youtube.com/watch?v=y6120QOlsfU)，随后透露自己真正喜欢的是 Daft Punk 的 Discovery 专辑，并在 [Spotify](https://open.spotify.com/track/098ttCNmncrO4YvqWUNMvn?si=0d0c602e42244d3e) 上进行了分享。其他用户也纷纷加入，分享了他们最喜欢的 Daft Punk 歌曲，如 "Lose Yourself to Dance"。
- **对 AI Studio 上的 Gemma 2 反应不一**：一位成员提到在 [aistudio.google.com](https://aistudio.google.com) 上试用了 Gemma 2 27b，并指出输出效果并不理想。另一位用户认出了这是来自 Reddit 的引用，而其他人则对 Gemma 2 及其潜在能力表示兴奋和期待。
- **对 Gemini 2.0 的推测与期待**：用户推测 Gemma 2 的发布可能意味着 Gemini 2.0 也即将到来。大家对训练该模型的潜力感到非常兴奋，一位用户正考虑租用 [Runpod 48GB 实例](https://runpod.io) 来彻底测试模型的性能和容量。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/er-juqsr4nc?si=g3vtzgUa94GEcD-s">Not like us (1962) full song</a>: K Dot senior 回来满足你们对这首歌完整版的要求。</li><li><a href="https://www.youtube.com/watch?v=y6120QOlsfU">Darude - Sandstorm</a>: Darude 新专辑 "Together" 现已发布 → https://found.ee/Darude-TogetherNew 'Closer Together' 音乐视频现已发布 → https://youtu.be/edUBI3k2lUo?si=ynkxg7p7Ofa...</li><li><a href="https://open.spotify.com">Spotify - Web Player: Music for everyone</a>: Spotify 是一款数字音乐服务，让你能够访问数百万首歌曲。</li><li><a href="https://open.spotify.com/track/098ttCNmncrO4YvqWUNMvn?si=0d0c602e42244d3e">High Life</a>: Daft Punk · 歌曲 · 2001
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1251258835419660319)** (304 messages🔥🔥): 

- **在 Windows 上遇到 Triton 问题**：一位成员报告称，即使在正确设置 Visual C++ 后，在 Windows 11 上安装 Triton 仍遇到问题。通过询问是否可以从终端调用 `g++` 或 `clang++` 提供了协助。

- **数据准备教程请求**：一位成员询问是否有针对 Unsloth 微调的数据准备教程，类似于 [OpenAI 的 chat fine-tuning 数据准备 notebook](https://cookbook.openai.com/examples/chat_finetuning_data_prep)。另一位成员提到计划制作一个教程，并推荐了一个相关的 [YouTube 视频](https://youtu.be/3eq84KrdTWY)。

- **模型训练在保存期间崩溃**：一位成员在训练 Yi 模型最后的保存步骤中遇到崩溃，怀疑是内存或磁盘空间问题。建议检查可用内存和磁盘空间，并提供了一个 [Unsloth 在 GitHub 上的保存问题链接](https://github.com/unslothai/unsloth/wiki#saving-to-gguf--vllm-16bit-crashes)。

- **Batch size 与 Gradient Accumulation 的问题**：一位成员对调整 batch size 和 gradient accumulation 时 VRAM 使用量的差异提出疑问。讨论明确了 gradient accumulation steps 的作用类似于增加 batch size，并建议尝试更大的 batch size。

- **save.py 中 quantization_method 的错误**：发现了一个 bug，其中 `quantization_method` 被错误地处理为字符串，导致错误。解决方法包括将 `quantization_method` 作为列表传递，并提交了一个修复该 bug 的 [pull request](https://github.com/unslothai/unsloth/pull/651)。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html#ubuntu">CUDA Quick Start Guide</a>: 未找到描述</li><li><a href="https://youtu.be/3eq84KrdTWY">Llama 3 Fine Tuning for Dummies (with 16k, 32k,... Context)</a>: 在本分步教程中，学习如何使用 Unsloth 轻松微调 Meta 强大的新 Llama 3 语言模型。我们涵盖了：* Llama 3 8B 概述以及...</li><li><a href="https://cookbook.openai.com/examples/chat_finetuning_data_prep">Data preparation and analysis for chat model fine-tuning | OpenAI Cookbook</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-to-gguf--vllm-16bit-crashes">Home</a>: 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3, Mistral, Phi &amp; Gemma LLM - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#continued-pretraining--finetuning-the-lm_head-and-embed_tokens-matrices">Home</a>: 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3, Mistral, Phi &amp; Gemma LLM - unslothai/unsloth</li><li><a href="https://huggingface.co/datasets/maplerxyz1/rbxidle">maplerxyz1/rbxidle · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki#continued-pretraining--finetuning-the-lm_h">Home</a>: 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3, Mistral, Phi &amp; Gemma LLM - unslothai/unsloth</li><li><a href="https://github.com/Syllo/nvtop">GitHub - Syllo/nvtop: GPU &amp; Accelerator process monitoring for AMD, Apple, Huawei, Intel, NVIDIA and Qualcomm</a>: 适用于 AMD, Apple, Huawei, Intel, NVIDIA 和 Qualcomm 的 GPU &amp; 加速器进程监控 - Syllo/nvtop</li><li><a href="https://discuss.huggingface.co/t/continuous-training-on-fine-tuned-model/6687/3">Continuous training on Fine-tuned Model</a>: 感谢您的回复。我也尝试了这种方法。我的旧数据集和新数据集包含不同的文本。这种方法使模型严重倾向于提供的新文本。这导致了...</li><li><a href="https://github.com/unslothai/unsloth/issues/210">I got unsloth running in native windows. · Issue #210 · unslothai/unsloth</a>: 我在原生 Windows（无 WSL）上运行了 Unsloth。你需要 Visual Studio 2022 C++ 编译器、Triton 和 DeepSpeed。我有一个完整的安装教程，我本想在这里写下来，但我现在在用手机...</li><li><a href="https://github.com/unslothai/unsloth/pull/651">Fix breaking bug in save.py with interpreting quantization_method as a string when saving to gguf by ArcadaLabs-Jason · Pull Request #651 · unslothai/unsloth</a>: 背景：在尝试将微调后的模型保存为 GGUF 时，我今天遇到了一个新错误。经调查，我发现问题出在一些代码上，这些代码错误地中断了传递的字符串...</li><li><a href="https://github.com/unslothai/unsloth/wiki">Home</a>: 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3, Mistral, Phi &amp; Gemma LLM - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/save.py?utm_source=ainews&utm_medium=email&utm_campaign=ainews-fixing-gemma.">unsloth/unsloth/save.py at main · unslothai/unsloth</a>: 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3, Mistral, Phi &amp; Gemma LLM - unslothai/unsloth</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3, Mistral, Phi &amp; Gemma LLM - unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/1rU4kVb9A8yNRThGwftk0EU7dWsDIPOYb?usp=sharing#scrollTo=2eSvM9zX_2d3">Google Colab</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1251951286845444198)** (3 条消息): 

- **CryptGPT 引入了隐私保护 LLM**: 一位用户分享了一篇题为 *"CryptGPT: Privacy-Preserving LLMs using Vigenere cipher"* 的入门博客文章。该博文描述了在加密数据集上预训练 GPT-2 模型，实现了与普通 GPT-2 相当的性能，但需要加密密钥才能使用。[博客文章链接](https://x.com/diwanksingh/status/1802118343446724655)。

**提到的链接**: <a href="https://x.com/diwanksingh/status/1802118343446724655">来自 Diwank Singh (@diwanksingh) 的推文</a>: http://x.com/i/article/1802116084507848704

  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/)** (1 条消息): 

starsupernova: 噢，非常有趣！
  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1251287545086607391)** (49 条消息🔥): 

- **Lighting AI 界面建议**：一位成员分享了 [NVIDIA warp 示例代码](https://github.com/NVIDIA/warp/blob/main/warp/examples/core/example_sph.py) 并寻求关于查看渲染结果的图形界面的建议。他们考虑通过设置 VNC 会话来解决此问题。

- **已解决 NVRTC 编译错误**：一位用户描述了 NVRTC 的一个问题，即编译多个 kernel 时会出现 'invalid resource handle'。他们后来通过避免为每次编译初始化新 context 解决了该问题，因为这会导致 CUDA 释放 module/function。

- **GPU SM 数量差异**：有人对 [A10G GPU](https://www.techpowerup.com/gpu-specs/a10g.c3798) 的实测 SM 数量与报告数量之间的差异提出了疑问，指出 techpowerup 报告为 72 个 SM，而 pycuda 测量为 80 个。经澄清，该网站可能存在错误，[其他来源](https://d1.awsstatic.com/product-marketing/ec2/NVIDIA_AWS_A10G_DataSheet_FINAL_02_17_2022.pdf) 确认其拥有 80 个 SM。

- **新 NVIDIA 5090 GPU 推测**：成员们讨论了即将推出的 NVIDIA 5090，推测其可能拥有高达 64 GB 的 VRAM（[来源](https://www.techpowerup.com/323495/possible-specs-of-nvidia-geforce-blackwell-gpu-lineup-leaked)）。关于这些规格的可能性存在争论，对在消费级版本中看到 64GB 持悲观态度。

- **论坛知识在日常 AI 工作中的价值**：一位成员对大多数讨论在日常 AI 工作中（除少数特定话题外）的实际价值表示怀疑。其他人回应强调了性能优化（performance optimization）的重要性，以及学习和参与此类社区的普遍价值。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Chinese_zodiac">Chinese zodiac - Wikipedia</a>：未找到描述</li><li><a href="https://github.com/NVIDIA/warp/blob/main/warp/examples/core/example_sph.py">warp/warp/examples/core/example_sph.py at main · NVIDIA/warp</a>：一个用于高性能 GPU 模拟和图形的 Python 框架 - NVIDIA/warp</li><li><a href="https://www.baseten.co/blog/nvidia-a10-vs-a10g-for-ml-model-inference/">NVIDIA A10 vs A10G for ML model inference</a>：A10 是一款 Ampere 系列 GPU，擅长运行 7B 参数 LLM 等任务。AWS 的 A10G 变体在 GPU 显存和带宽方面与之相似，基本可以互换。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/1oKyTQEmlf">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3622">NVIDIA GeForce RTX 3090 Specs</a>：NVIDIA GA102, 1695 MHz, 10496 Cores, 328 TMUs, 112 ROPs, 24576 MB GDDR6X, 1219 MHz, 384 bit</li><li><a href="https://www.techpowerup.com/gpu-specs/geforce-rtx-4090.c3889">NVIDIA GeForce RTX 4090 Specs</a>：NVIDIA AD102, 2520 MHz, 16384 Cores, 512 TMUs, 176 ROPs, 24576 MB GDDR6X, 1313 MHz, 384 bit</li><li><a href="https://www.techpowerup.com/gpu-specs/a10g.c3798">NVIDIA A10G Specs</a>：NVIDIA GA102, 1710 MHz, 9216 Cores, 288 TMUs, 96 ROPs, 24576 MB GDDR6, 1563 MHz, 384 bit
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1252285272507551794)** (2 条消息): 

- **Colab 上 Triton 的 AttributeError**：一位用户在 Colab 上运行 Triton 官方教程中的 Fused Softmax 时遇到了 AttributeError。错误信息显示 `'CudaDriver' object has no attribute 'active'`，他们正在寻求帮助。

- **Triton 中嵌套 Reduction 的可行性**：另一位用户询问了在 Triton 中执行嵌套 reduction 的可能性。他们有兴趣在不同阶段运行 reduction 代码以单独处理象限（quadrants），询问是否支持这种分阶段的 reduction。
  

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1251580462028161075)** (10 条消息🔥): 

- **`torch.compile(mode="max-autotune")` 报错**：一位用户报告收到错误消息 `Not enough SMs to use max_autotune_gemm mode`，原因是 PyTorch 代码中硬编码了 68 个 SM 的限制，而他们的 GPU 只有 66 个 SM。该用户分享了 [PyTorch 仓库中相关部分的链接](https://github.com/pytorch/pytorch/blob/f0d68120f4e99ee6c05f1235d9b42a4524af39d5/torch/_inductor/utils.py#L976)。

- **关于降低 SM 阈值的讨论**：一名成员建议降低 SM 阈值，以测试在不需要从源码重新构建的情况下性能是否依然良好。目前硬编码的值是因为 CI 中缺乏消费级 GPU。

- **修改 SM 阈值后的性能测试**：在将 SM 阈值更改为 0 后，用户报告性能没有显著提升。

- **启用 Coordinate Descent Tuning**：另一名成员提议启用 `inductor/config.py` 中的 coordinate descent tuning，作为提升性能的潜在解决方案。

**提到的链接**：<a href="https://github.com/pytorch/pytorch/blob/f0d68120f4e99ee6c05f1235d9b42a4524af39d5/torch/_inductor/utils.py#L976">pytorch/torch/_inductor/utils.py at f0d68120f4e99ee6c05f1235d9b42a4524af39d5 · pytorch/pytorch</a>：Python 中具有强 GPU 加速功能的 Tensor 和动态神经网络 - pytorch/pytorch

  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1252343980423909487)** (2 条消息): 

- **Vayuda 论文引发对搜索算法的兴趣**：一名成员分享了 [Vayuda 论文的链接](https://arxiv.org/pdf/2406.07394)，希望更多人能投入到搜索算法的研究中。这暗示了该领域具有巨大的研发潜力。
  
- **GPT-4 与 LLaMA 3 8B 的匹配表现令人印象深刻**：一名成员对 **GPT-4** 与 **LLaMA 3 8B** 的匹配结果感到惊讶。他们强调这一成就在当前的 AI 能力中值得关注。
  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1251444875799433288)** (5 条消息): 

- **PMPP 书中未包含 Blockwise softmax**：PMPP 书中没有涵盖 Blockwise softmax 的概念，但理解 flash-attn 算法和共享内存 (smem) 至关重要。高端实现利用了 tensor cores，需要进一步探索 CUTLASS 等资源。
- **从易懂的 YouTube 讲座开始**：对于 GPU 编程和高性能计算的新手，建议从 [YouTube 讲座](https://youtube.com)开始。这些讲座旨在为基础知识提供易于理解的介绍。
  

---


### **CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1252344004738289765)** (1 条消息): 

- **发布用于简化 Cloud TPU 的 tpux**：一名成员宣布了 **tpux 项目**，这是一套旨在简化 Cloud TPU 设置和操作的工具，以方便在多个主机上使用 JAX。更多详情请访问 [GitHub 上的 tpux](https://github.com/yixiaoer/tpux) 并 [点个 ⭐️](https://twitter.com/shin_miyaku22/status/1802373884077072537)。

**提到的链接**：<a href="https://github.com/yixiaoer/tpux">GitHub - yixiaoer/tpux: A set of Python scripts that makes your experience on TPU better</a>：一套让你的 TPU 体验更好的 Python 脚本 - yixiaoer/tpux

  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1251751170675052565)** (11 条消息🔥): 

- **Quant API 导入文档问题**：一名成员指出了一项修正，即 *``unwrap_tensor_subclass()`` 应该从 `torchao.utils` 或 `torchao.quantization.quant_api` 导入*，而不是 `torchao.quantization.utils`。他们强调用户在编译量化模型之前调用 `unwrap_tensor_subclass()` 以避免错误的重要性。

- **API 发布延迟和 BC 问题**：已确认 **0.3 版本将推迟发布**，因为需要解决向后兼容性 (BC) 问题。这次延迟确保团队能够处理并修复关键问题。

- **带有 'brrr' 提案的创新 API 命名**：有一个既幽默又实用的建议，即创建一个名为 **`brrr` 的 API，根据 'r' 的数量添加额外的实验性标志**。一名成员幽默地询问这是否是认真的，但也暗示需要更轻松地控制 `torchinductor` 的标志，如 `use_mixed_mm`。

- **关于 `use_mixed_mm` 标志的反馈**：一名成员建议，如果 AO 中的相关 kernel 已开启，则默认启用 `use_mixed_mm` 标志。该反馈可能会促成一个 GitHub issue 以进行进一步讨论和实现。


  

---

### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1251340075686432778)** (10 messages🔥): 

- **Meta 应对大规模 AI 训练挑战**：[Meta 的文章](https://engineering.fb.com/2024/06/12/data-infrastructure/training-large-language-models-at-scale-meta/) 讨论了训练大语言模型 (LLMs) 所需的复杂性和计算量。向生成式 AI 的转变使得重新思考软件、硬件和网络基础设施成为必要。

- **采访 Esolang Academics**：分享了一个名为“Interview with Esolang Academic 2024”的 [YouTube 视频](https://www.youtube.com/watch?v=ieqsL5NkS6I)。完整版和 BC Vim Linter 将于次日在 Patreon 上以 5 美元的价格提供。

- **Pessimistic Neko 的 Jensen 表情包**：成员 pessmistic_neko 发布了表情包 <:jensen:1189650200147542017> 来表达他们的乐趣。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=ieqsL5NkS6I">Interview with Esolang Academic 2024</a>: 奇特编程语言完整版 + BC Vim Linter 明天在以下网站售价 5 美元：https://www.patreon.com/ProgrammersAreAlsoHuman 采访一位 Esoteric 开发者...</li><li><a href="https://engineering.fb.com/2024/06/12/data-infrastructure/training-large-language-models-at-scale-meta/">How Meta trains large language models at scale</a>: 随着我们继续专注于通过 AI 研发来解决日益复杂的问题，我们经历的最重大且最具挑战性的转变之一就是计算规模的……</li><li><a href="https://engineering.fb.com/2024/06/12/data-infrastructure/training-large-language-models-at-scale-me">How Meta trains large language models at scale</a>: 随着我们继续专注于通过 AI 研发来解决日益复杂的问题，我们经历的最重大且最具挑战性的转变之一就是计算规模的……
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1252199364651257867)** (1 messages): 

- **在 AI_dev 会议上偶遇 Akim**：一位成员提到他们“可能会参加 AI_dev”并邀请其他人联系。他们还提到周二将放映一部关于 “PyTorch” 的电影。
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1251250350711902230)** (473 messages🔥🔥🔥): 

- **DataLoader PR 已合并，令人惊讶的是性能没有差异**：经过一些讨论和测试，Permuted DataLoader PR 已合并，尽管初始运行显示性能没有提升。[感到惊讶的 Aleksa 重新进行了测试，最终确认验证损失 (validation loss) 有轻微改善](https://github.com/karpathy/llm.c/pull/573)。

- **仅使用唯一种子的随机舍入 (Stochastic rounding)**：讨论了确保随机舍入使用唯一种子的 [PR](https://github.com/karpathy/llm.c/pull/597)。团队发现他们方法中的一个溢出特性实际上是噪声函数算法预期的。

- **ZeRO-2 PR 具有明显的内存开销**：讨论了 ZeRO-2 的 PR 593 的复杂性和内存开销。[建议包括对某些参数回退到 ZeRO-1](https://github.com/karpathy/llm.c/pull/593)。

- **主权重 (Master weight) 存储**：合并了一个用于保存主权重以恢复状态的 PR (https://github.com/karpathy/llm.c/pull/522)，以提高确定性 (determinism)。后续任务包括通过 CI 验证确定性，并探索节省内存的技术，[例如保存 16-bit 主权重](https://github.com/karpathy/llm.c/pull/432)。

- **LayerNorm kernel 优化带来提速**：分析数据表明，新的 LayerNorm kernel (kernel 6) 比旧的 (kernel 3 和 5) 更快。[这显著提升了某些任务的性能](https://github.com/karpathy/llm.c/pull/600)，特别是在特定配置下，如 recompute=2。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>

<li><a href="https://arxiv.org/abs/2010.06192">重温 BFloat16 训练</a>：最先进的通用低精度训练算法使用 16 位和 32 位精度的混合，这产生了一种传言，即仅靠 16 位硬件计算单元不足以最大化模型精度...</li><li><a href="https://x.com/SquirrelTweets">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=LWFzPP8ZbdU&t=3347s">基于噪声的 RNG</a>：在 2017 年 GDC 游戏程序员数学演讲中，SMU Guildhall 的 Squirrel Eiserloh 讨论了 RNG 与噪声函数，并展示了后者如何取代...</li><li><a href="https://x.com/SquirrelTweets/status/1421251894274625536">来自 Squirrel Eiserloh (@SquirrelTweets) 的推文</a>：更新了我的原始噪声函数。消除了 @ptrschmdtnlsn 发现的一个缺陷，即某些高位输入位对某些低位输出位缺乏影响。任何使用我 GDC 2017 中的 Squirrel3 的人...</li><li><a href="https://www.youtube.com/watch?v=LWFzPP8ZbdU&t=2199s">基于噪声的 RNG</a>：在 2017 年 GDC 游戏程序员数学演讲中，SMU Guildhall 的 Squirrel Eiserloh 讨论了 RNG 与噪声函数，并展示了后者如何取代...</li><li><a href="https://arxiv.org/abs/2405.18392">超越固定训练时长的缩放定律与计算最优训练</a>：规模已成为获得强大机器学习模型的主要因素。因此，了解模型的缩放特性是有效设计正确训练设置的关键...</li><li><a href="https://github.com/karpathy/llm.c/pull/600">由 gordicaleksa 提交的“为 LayerNorm 前向传播使用更快的 kernel” · Pull Request #600 · karpathy/llm.c</a>：我在 RTX 3090 和 H100 系统上运行了 /dev/cuda/ 下的 kernel 5 (./layernorm_forward 5)，在两个系统上都更快。数据：kernel 3，最优 block size：RTX 3090 → 32 (689.11 GB/s) H1...</li><li><a href="https://github.com/karpathy/llm.c/blob/master/llmc/global_norm.cuh#L63">llm.c/llmc/global_norm.cuh 分支 master · karpathy/llm.c</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/pull/594">由 karpathy 提交的“添加导出到 HF 并运行 Eleuther 评估的脚本” · Pull Request #594 · karpathy/llm.c</a>：未找到描述</li><li><a href="https://github.com/karpathy/llm.c/pull/338">由 ademeure 提交的“使用 cuBLASLt 进行 GELU 融合（更慢，因为它仅在 FP16 模式下合并，而非 BF16/FP32...）” · Pull Request #338 · karpathy/llm.c</a>：事实证明，cuBLASLt 不仅无法将 BF16 GELU（或 RELU）融合到 BF16 matmul 中，而且最终得到的奇怪 kernel 比我们自己的 GELU kernel 还要慢，因为它每次执行 2 次写入...</li><li><a href="https://github.com/karpathy/llm.c/pull/602">由 gordicaleksa 提交的“热修复 - 正确处理 block sums 的最大数量” · Pull Request #602 · karpathy/llm.c</a>：之前的 assert 逻辑过于严格，因为它取决于模型的层数和 GPU 的规格（SM 数量和每个 SM 的最大线程数）。此 PR 修复了该问题...</li><li><a href="https://github.com/karpathy/build-nanogpt/pull/30">由 WilsonCWu 提交的“修复导致梯度累积不正确和 loss 不正确的同步问题” · Pull Request #30 · karpathy/build-nanogpt</a>：复现使用的 torch 版本为 '2.3.1+cu121'。将 B 设置为 16（原为 64）。观察到以下 loss：250 val 6.4300 250 hella 0.2440 250 train 6.387966 ... 1000 val 4.8797 1000 hella 0.2419 1000 ...</li><li><a href="https://github.com/karpathy/llm.c/pull/601">由 gordicaleksa 提交的“修复 encoder 反向传播 kernel 中的随机舍入” · Pull Request #601 · karpathy/llm.c</a>：#597 为 adamw 更新提供了唯一的 seed。此 PR 为 encoder 反向传播执行了相同的操作，这是我们执行随机舍入的唯一其他地方。</li><li><a href="https://github.com/karpathy/llm.c/pull/573?">由 gordicaleksa 提交的“Dataloader - 引入随机性” · Pull Request #573 · karpathy/llm.c</a>：正在实现完全随机的训练数据打乱... 此 PR 执行了以下操作：每个进程都有不同的唯一随机 seed，每个进程的训练数据加载器独立选择其起始 sha...</li><li><a href="https://github.com/karpathy/llm.c/pull/595">由 KarhouTam 提交的“对 `dev/cuda` 中 `layernorm_forward` 的更改” · Pull Request #595 · karpathy/llm.c</a>：移除 cooperative groups。按照 #292 中的说明，移除现有 layernorm 前向传播 kernel 中的 cooperative groups 代码。benchmark 更改前后的性能：Block Size l...</li><li><a href="https://github.com/karpathy/llm.c/pull/591">由 ademeure 提交的“融合前向 GELU（再次）” · Pull Request #591 · karpathy/llm.c</a>：事实证明，这在带有 CUDA 12.5 的 H100 上得到了正确的融合（因此更快）——而在带有 CUDA 12.4 的 RTX 4090 上绝对没有融合，实际上明显更慢，我怀疑这更多是关于...</li><li><a href="https://github.com/karpathy/llm.

<li><a href="https://github.com/karpathy/llm.c/pull/513">由 ChrisDryden 添加了 packed layernorm_forward · Pull Request #513 · karpathy/llm.c</a>：这是在 layernorm 中使用 packed 数据类型的实现，在 dev 文件中该 kernel 的速度提升了约 50%，正在等待将数据类型引入的 PR...</li><li><a href="https://github.com/karpathy/llm.c/pull/506">由 ChrisDryden 添加了不重新计算 mean 和 rstd 的额外 layernorm forward kernel · Pull Request #506 · karpathy/llm.c</a>：这是第一步优化，后续还有更多空间，但现在 kernel 已拆分为两个，以便未来可以独立修改每个 Layernorm forward...</li><li><a href="https://github.com/karpathy/llm.c/pull/561">由 lancerts 修复编译器警告和错误 · Pull Request #561 · karpathy/llm.c</a>：修复了根据讨论 #558 (comment) 在 CUDA 11.8 中发生的错误：matmul_backward_bias.cu(151): error: no operator "+=" matches these operands operand types are: floatX += ...</li><li><a href="https://github.com/karpathy/llm.c/actions/runs/9537233489/job/26285087145?pr=600">为 LayerNorm forward 使用更快的 kernel · karpathy/llm.c@7d7084a</a>：简单的、原始 C/CUDA 环境下的 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/pull/508/">由 eliebak 添加带有 (1-sqrt) 衰减的 wsd schedule · Pull Request #508 · karpathy/llm.c</a>：添加了新的学习率调度支持：WSD 学习率调度：Warmup（预热）：经典的线性预热；Stable（稳定）：恒定学习率；Decay（衰减）：以 (1-sqrt) 形状衰减至 min_lr。（更多信息见 https://ar...</li><li><a href="https://github.com/karpathy/llm.c/pull/597">由 gordicaleksa 修复 stochastic rounding · Pull Request #597 · karpathy/llm.c</a>：之前我们的 stochastic rounding 逻辑没有为每个舍入参数提供唯一的 seed。此 PR 修复了该问题。具体来说，之前：我们为...传递了相同的 seed。</li><li><a href="https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-features/rounding-modes.html">Neuron 舍入模式 — AWS Neuron 文档</a>：未找到描述</li><li><a href="https://github.com/karpathy/llm.c/pull/583">由 vyom1611 将硬编码的最大浮点数替换为 FLT_MAX · Pull Request #583 · karpathy/llm.c</a>：修复了一些 TODO，并将浮点整数的硬编码值替换为 FLT_MAX，该常量来自两个文件中已包含的 `<float.h>` 头文件。</li><li><a href="https://github.com/karpathy/llm.c/pull/573/commits/c81f1efbb82b4056cb9402d2ae7786e9d0165f1f">Dataloader - 引入随机性，由 gordicaleksa 提交 · Pull Request #573 · karpathy/llm.c</a>：旨在实现完全随机的训练数据打乱（shuffling）... 此 PR 包含以下内容：每个进程拥有不同的唯一随机 seed；每个进程的训练数据加载器独立选择其起始 shard...</li><li><a href="https://github.com/karpathy/llm.c/pull/603">在 CI 中检查确定性，由 ngc92 提交 · Pull Request #603 · karpathy/llm.c</a>：扩展了测试脚本以验证确定性。一些想法：使用 C 语言进行内存管理时，很容易引入内存泄漏，例如在中间未释放内存的情况下两次从 checkpoint 加载...</li><li><a href="https://huggingface.co/mdouglas/llmc-gpt2-774M-150B">mdouglas/llmc-gpt2-774M-150B · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/karpathy/llm.c/discussions/580">GPT-2 (774M) 复现成功 · karpathy/llm.c · Discussion #580</a>：我在 8X A100 80GB 节点上让 GPT-2 774M 模型运行了约 6 天（150B tokens，在 100B FineWeb 样本数据集上进行了 1.5 个 epochs），训练几小时前刚刚结束，进展顺利...</li><li><a href="https://github.com/karpathy/llm.c/pull/522/files#diff-fd05f347e9f31ebf4645e7ee912dfc80e74faa9b94cf8a2c35df555fc7927b83R38">在恢复状态中添加 master weights，由 gordicaleksa 提交 · Pull Request #522 · karpathy/llm.c</a>：目前我们没有将 master weights 作为状态的一部分进行保存 -> 这会导致精度损失，因为否则在恢复时我们必须通过上采样（upcasti...）来重建 master weights。</li><li><a href="https://github.com/karpathy/llm.c/pull/432">仅保存用于重建 fp32 master weights 的缺失位，由 ngc92 提交 · Pull Request #432 · karpathy/llm.c</a>：我想我成功搞定了位操作（bit-fiddling），这将有效地为我们提供 fp31 master 参数，而成本仅为额外的 16 位（而不是目前的 32 位）。在合并之前，代码...</li><li><a href="https://github.com/karpathy/llm.c/pull/600/files">为 LayerNorm forward 使用更快的 kernel，由 gordicaleksa 提交 · Pull Request #600 · karpathy/llm.c</a>：我在 RTX 3090 和 H100 系统上运行了 /dev/cuda/ 下的 kernel 5 (./layernorm_forward 5)，在两者上都更快。数据：kernel 3，最佳 block size 在：RTX 3090 → 32 (689.11 GB/s...</li><li><a href="https://github.com/k_

arpathy/llm.c/pull/556">ngc92 提供的 CUDA streams + disk IO 工具 · Pull Request #556 · karpathy/llm.c</a>: 使用 CUDA streams 处理用于 checkpointing 的磁盘 IO 是一项非平凡的任务。如果不小心，很容易写出错误的代码（在开始写入之前需要等待数据传输到 CPU...</li><li><a href="https://github.com/karpathy/llm.c/pull/593/files">ngc92 提供的 Zero 2 - WIP · Pull Request #593 · karpathy/llm.c</a>: 正在尝试让第一个版本运行。代码还不美观，目前我们在通信代码中失去了异步性，因为我们需要为下一层重用 buffer，而且它不...</li><li><a href="https://github.com/karpathy/llm.c/pull/599/files">karpathy 提供的 Permuted DataLoader · Pull Request #599 · karpathy/llm.c</a>: Permuted dataloader 以及一些初步测试。仍在进行中（WIP）。</li><li><a href="https://github.com/karpathy/llm.c/pull/522">gordicaleksa 提供的将 master weights 添加到恢复状态 · Pull Request #522 · karpathy/llm.c</a>: 我们目前没有将 master weights 作为状态的一部分保存 -> 这会导致损失一些精度，因为否则当我们恢复时，必须通过从低精度向上转换来重建 master weights...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[oneapi](https://discord.com/channels/1189498204333543425/1233802893786746880/1251457210907889696)** (2 条消息): 

- **Gaudi 在支持 Dynamic batching 方面遇到困难**：一位成员提到了将带有 vLLM 的 dynamic batching 移植到 Gaudi 的困难。他们质疑是否存在架构限制，阻碍了 KV cache flash attention kernels 的实现，并将其与可以无障碍处理的常规“矩形”形状进行了对比。

- **建议将频道重命名为 Intel**：另一个建议是将频道重命名为 **Intel**，并标记了一位用户以获取其意见。这反映了可能的频道品牌重塑方向。
  

---

### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1251566494987321424)** (49 messages🔥): 

- **会议故障排除与新链接分享**：用户讨论了语音聊天问题，并分享了诸如 [使用 Nix 的 Python 开发环境](https://wiki.nixos.org/wiki/Python) 等资源。一位用户在测试环境时提到：“新笔记本电脑，Ubuntu 遇到了一些问题。”
- **基准测试与量化讨论**：大部分对话集中在不同精度和量化技术下的矩阵乘法（matmul）基准测试。一位用户询问：“你是在对 matmul(x_fp16, W_nbit) 进行基准测试，还是包含了带有分组（grouping）的缩放（scaling）/ 零点（zeros）？” 其他人则回应了他们具体的基准测试方法以及分组对提升质量的重要性。
- **延伸阅读资源链接**：分享了几个有用的链接，包括一种 [量化](https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py#L86-L90) 技术和一个 [支持混合精度矩阵乘法的库](https://github.com/microsoft/BitBLAS/blob/main/docs/PythonAPI.md#matmul)。这些资源旨在帮助更清晰地理解优化策略。
- **VRAM 限制与 GPU 考量**：讨论还涉及了由于 VRAM 限制在本地运行 llama2 等大型模型的局限性。一位用户提到使用配备 GeForce GTX 1650 的 XPS15 笔记本电脑，并探索了其他平台，如 Lightning AI 的 L4（提供 22 小时免费测试时间）。
- **新的 Git Pull Requests 与测试用例提交**：分享了开发方面的更新，包括为 BitnetTensor 和 UInt2Tensor 提交新的测试用例。用户围绕问题和更新进行互动，如评论所述：“提交了 BitnetTensor、UInt2Tensor 和 bitpacking gen 的测试用例”，展示了协作开发的进展。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/microsoft/BitBLAS/blob/main/docs/PythonAPI.md#matmul">BitBLAS/docs/PythonAPI.md at main · microsoft/BitBLAS</a>：BitBLAS 是一个支持混合精度矩阵乘法的库，特别适用于量化 LLM 部署。 - microsoft/BitBLAS</li><li><a href="https://github.com/pytorch/ao/issues/386">Tensor Core Layout docs is not clear · Issue #386 · pytorch/ao</a>：目前我们只有 docstrings，但还需要改进——这是在 @vayuda 考虑扩展其 bitpacking 工作以包含缩放（scales）概念时提出的。Tensor Core 布局是什么意思？...</li><li><a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py#L86-L90">hqq/hqq/core/quantize.py at master · mobiusml/hqq</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://wiki.nixos.org/wiki/Python">Python - NixOS Wiki</a>：未找到描述</li><li><a href="https://github.com/pytorch/ao/pull/282">[WIP] Added first bits of Uint2Tensor and BitnetTensor by andreaskoepf · Pull Request #282 · pytorch/ao</a>：创建了 UInt2Tensor 类（类似于 UInt4Tensor 类）。添加了 BitnetTensor 类和第一个单元测试，该测试对 nn.Linear() 层的权重进行量化并执行 matmul。目前...
</li>
</ul>

</div>

### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1251249928613658635)** (204 条消息🔥🔥): 

- **使用 'lms' 工具链接你的代码项目**：随着 [LM Studio 0.2.22](https://lmstudio.ai) 的发布，用户现在可以使用 'lms' 来管理模型和调试 prompt。该工具有助于加载/卸载模型，并检查原始 LLM 输入，从而简化本地 AI 部署 ([GitHub repository](https://github.com/lmstudio-ai/lms))。

- **现已支持 Intel ARC A770 GPU**：有几项关于 Intel ARC A770 GPU 支持的咨询。官方提供了[指南](https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md)以启用 Intel GPU 的 OpenCL，并强调了对 GPU layers 的手动调整。

- **性能对比与 GPU 利用率**：成员们讨论了性能对比，显示出 CPU 与 GPU 的结果各异，以及获得最佳模型性能所需的特定配置。解决了 Deepseek Coder V2 Lite GGUF 模型的问题，强调了切换 Flash Attention 设置的必要性。

- **使用 Open Interpreter 托管本地模型的问题**：用户在通过 LM Studio 为 Open Interpreter 托管本地模型时遇到了问题。建议查看 [Open Interpreter 文档](https://docs.openinterpreter.com/language-models/local-models/lm-studio)上的详细指南。

- **LM Studio 中的字体大小调整**：一个反复出现的需求是改进 LM Studio 中的字体大小控制。虽然目前有用于放大/缩小的键盘快捷键，但建议在应用内提供更持久且通用的解决方案。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/careers">👾 LM Studio - 发现并运行本地 LLMs</a>：查找、下载并实验本地 LLMs</li><li><a href="https://lmstudio.ai/,">👾 LM Studio - 发现并运行本地 LLMs</a>：查找、下载并实验本地 LLMs</li><li><a href="https://lmstudio.ai/blog/lms#debug-your-prompting-with-lms-log-stream">介绍 `lms` - LM Studio 的配套 CLI 工具 | LM Studio</a>：今天，随 LM Studio 0.2.22 一起，我们发布了 lms 的第一个版本 —— LM Studio 的配套 CLI 工具。</li><li><a href="https://docs.openinterpreter.com/language-models/local-models/lm-studio">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF">Qwen/Qwen2-7B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/MaziyarPanahi/Meta-Llama-3-70B-Instruct-GGUF">MaziyarPanahi/Meta-Llama-3-70B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>：LM Studio JSON 配置文件格式及示例配置文件集合。 - lmstudio-ai/configs</li><li><a href="https://www.reddit.com/r/techsupport/comments/wmq143/windows_10_explorer_vram_leak/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://huggingface.co/datasets">Hugging Face – 构建未来的 AI 社区。</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7979">Bug: Deepseek Coder MOE GGML_ASSERT: ggml.c:5705: ggml_nelements(a) == ne0*ne1 · Issue #7979 · ggerganov/llama.cpp</a>：发生了什么？尝试运行新的 Deepseek Coder 转换或量化版本时出现此错误：GGML_ASSERT: ggml.c:5705: ggml_nelements(a) == ne0*ne1。发生在纯 CPU 模式下，我的 F32...
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1251250666870145075)** (137 条消息🔥🔥): 

<ul>
  <li><strong>Qwen2 与搜索小插曲</strong>：一位用户最初在使用 Qwen2 instruct 时难以获得连贯的输出，通过使用 "blank" 预设解决了问题；另一位成员建议在 Discord 内部搜索寻求帮助，而不是去外部网站。</li>
  <li><strong>角色扮演模型推荐</strong>：当被问及最适合角色扮演的模型时，一位成员推荐了 <a href="https://huggingface.co/DavidAU/Fimbulvetr-11B-Ultra-Quality-plus-imatrix-GGUF">Fimbulvetr-11B</a>，称其能有效满足他们的需求。</li>
  <li><strong>在困惑中寻找编程模型</strong>：讨论了最适合编程的模型，强调了该领域变化迅速，难以给出可靠的建议。用户提到更倾向于 <em>Codestral</em>，并探索了 <a href="https://llm.extractum.io/list/">Large and Small Language Models list</a> 以进行详细搜索。</li>
  <li><strong>新的 "Ultra-Quality" 模型发布</strong>：成员们重点介绍了新发布的高性能模型，如 <a href="https://huggingface.co/DavidAU/Psyonic-Cetacean-Ultra-Quality-20b-GGUF-imat-plus2">Psyonic-Cetacean-Ultra-Quality-20b-GGUF-imat-plus2</a>，并讨论了它们的测试结果和量化改进。</li>
  <li><strong>关于 DeepSeek-Coder-V2 的讨论</strong>：一位成员注意到了 <a href="https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct">DeepSeek-Coder-V2</a> 的发布，引发了对其编程能力的关注，并讨论了实现最佳性能所需的 VRAM 要求和 flash attention 设置。</li>
</ul>

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/DavidAU/Psyonic-Cetacean-Ultra-Quality-20b-GGUF-imat-plus2">DavidAU/Psyonic-Cetacean-Ultra-Quality-20b-GGUF-imat-plus2 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/DavidAU/PsyCET-Decision-Time-Imatrix">DavidAU/PsyCET-Decision-Time-Imatrix · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct">deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/mradermacher/Oumuamua-7b-instruct-v2-i1-GGUF">mradermacher/Oumuamua-7b-instruct-v2-i1-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/nitky/Oumuamua-7b-instruct-v2">nitky/Oumuamua-7b-instruct-v2 · Hugging Face</a>：未找到描述</li><li><a href="https://llm.extractum.io/list/">All Large Language Models</a>：一个精选的大语言模型和小语言模型（开源 LLM 和 SLM）列表。包含动态排序和过滤功能。</li><li><a href="https://x.com/umiyuki_ai/status/1801836418744127702">来自 うみゆき@AI研究 (@umiyuki_ai) 的推文</a>：Oumuamua-7b-instruct-v2 在 Shaberi3 基准测试中平均分为 7.25。超过了 GPT3.5T (7.16) 和 Qwen2-7B (7.23)。非常强劲。</li><li><a href="https://huggingface.co/RichardErkhov/ArthurZ_-_mamba-2.8b-gguf">RichardErkhov/ArthurZ_-_mamba-2.8b-gguf · Hugging Face</a>：未找到描述</li><li><a href="https://rentry.org/quant_test">量化格式如何影响模型输出？</a>：量化格式如何影响模型输出？简介、测试方法、盒子问题、提示词、结果、思考、购物和理发、提示词、结果、思考、健康教育、提示词、结果、思考...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/">Reddit - 深入探索</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1251271697072656424)** (13 messages🔥): 

- **如何处理 AVX2 指令问题**：一位成员在更新 LM Studio 后遇到问题，发现从[此处](https://lmstudio.ai/beta-releases.html)重新安装 beta 版本可以解决。他们警告之后“不要更新”，以避免问题再次出现。

- **Qwen2 输出 eot_id token 的问题**：用户报告 LM Studio 在运行 Qwen2 时会输出 eot_id token 而不是停止生成，这与 Llama3 的问题类似。建议包括检查所使用的 preset 以及是否启用了 flash。

- **关于 GPU off-loading 的建议**：一位用户提议增加一项增强功能，允许在模型完全加载到 RAM 之前将其 off-loading 到 GPU。这将使 VRAM 大于 RAM 的机器（特别是 GPU 服务器）受益，确保模型加载更快、更高效。

- **LM Studio 中的 Stop token 处理**：有用户对 LM Studio 允许 stop tokens 出现在输出中且不停止生成表示担忧，这导致了大量的 token 生成。一位用户强调 LM Studio 需要遵守所有列出的 stop tokens，并将其视为阻碍发布的 bug。

- **用户界面反馈**：LM Studio 的界面获得了“酷炫、柔和、直观且快速”的正向反馈。另一位用户建议添加 VRAM 使用统计数据，以便更好地监控性能。

**提到的链接**：<a href="https://lmstudio.ai/beta-releases.html">LM Studio Beta Releases</a>：未找到描述

  

---


### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1251899080448278538)** (8 messages🔥): 

- **纠结于错误检测**：一位成员对模型无法检测自身错误感到沮丧，并建议模型在无法自我纠正时应输出 *“#ERROR”*。尽管有明确的指令，模型仍不断请求指导，而不是优雅地报错。
- **苦恼于文本附加内容**：另一位成员寻求建议，以防止模型在回答末尾添加无关文本。他们指明使用的是 **bartowski/aya-23-8B-GGUF/aya-23-8B-Q8_0.gguf** 模型，并收到了尝试使用 *Cohere Command R* preset 的建议。
  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1251469810332864675)** (3 messages): 

- **Mikupad 用户面临配置问题**：一位用户在尝试使用 Mikupad 作为 webUI 与 LMS 交互时寻求帮助，报告了关于非预期 endpoint 或 method 的错误消息。他们提到：“Mikupad 的配置与 LMS 相同。”
- **需要 Codestral RAG 预设建议**：一位成员下载了 Codestral RAG，并请求关于创建面向 RAG (retrieval-augmented generation) 预设的建议。他们提到在 Hugging Face 上阅读了相关信息，但仍不确定预设的创建过程。
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1251271511277441105)** (34 messages🔥): 

- **存档 LM Studio 0.2.23 安装程序**：一位成员分享了指向存档的 LM Studio 0.2.23 安装文件的 [MirrorCreator 链接](https://mir.cr/E1WVBIOO)，并指出安装程序带有数字签名，可以验证其完整性。
- **添加第二块 RTX 3090**：一位成员询问添加不同品牌的 RTX 3090 是否会产生问题，以及是否应在同一系统中保留 RTX 2070。给出的建议是，为了获得最佳效果，应使用完全相同的显卡和 SLI bridge；保留 2070 会降低性能。
- **在 Server Mode 下设置 CPU 核心数**：有人询问是否可以在 Server Mode 下设置用于处理的 CPU 核心数量，并指出尽管模型已加载到 RAM 中，但仅使用了四个核心。
- **AMD Ryzen RX 7700S GPU 检测问题**：一位成员在 Windows 笔记本电脑上遇到了 LM Studio 无法检测到 AMD Ryzen RX 7700S GPU 的问题。讨论涉及了故障排除步骤，并澄清了关于 GPU 和 OS 的具体细节。
- **混插 RAM 条的担忧**：对话涉及了在仅限 CPU 的推理任务中，混插速度相同但时序可能不同的 RAM 条的可行性。结论是这应该是可行的，但需要使用 memtest 确认兼容性。

**提到的链接**：<a href="https://mir.cr/E1WVBIOO">LM-Studio-0.2.23-Setup.exe - Mirrored.to - Mirrorcreator - 多主机文件上传</a>：未找到描述

  

---

### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1251252850638716992)** (22 messages🔥): 

- **Smaug-3 Tokenizer 问题已解决**：最新版本解决了之前提到的 **smaug-3 tokenizer 问题**。此更新迅速得到了其他成员的认可和赞赏。 
- **将 ROCm 从主应用中解耦**：一位用户赞扬了将 **ROCm** 从主应用中解耦的举动，并强调了在 7900xtx 上的成功升级和流畅运行。他们分享了积极的体验：*"升级后对我来说运行良好"*。 
- **Command R+ GPU Offloading 故障**：用户讨论了一个问题，即当 **Command R+** 完全卸载到 GPU 时会输出乱码，而同一模型在 CPU 上运行正常。一位用户提到：*"那里有些不对劲。我的上下文只有 4k"*，表明这可能不是内存问题。
- **旧版本可用性**：成员们讨论了获取旧版本应用的困难，指出通过修改 URL 中的版本号来访问旧版本的方法已失效。建议包括在更新前自行保留旧版本的副本，尽管这在更新后被认为是不切实际的。
  

---


### **LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1251650482070491196)** (1 messages): 

- **重建环境解决 API key 问题**：一位用户描述了一个持续收到 "incorrect API key" 错误的问题，直到他们重建了环境并重新安装了依赖项。使用 `$env:OPENAI_API_KEY` 设置 API key 解决了他们的问题。
- **助手发送空白消息导致错误**：虽然用户成功设置了默认消息并为 user proxy 和 chat managers 配置了模型，但助手发送了空白消息，导致 LM Studio 出现错误。他们正在寻求该问题的进一步解决方案。
  

---


### **LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/1252341113931436143)** (13 messages🔥): 

- **尽管 LM Studio 正在运行，Interpreter 仍默认使用 GPT-4**：一位用户遇到了一个问题，即在 **LM Studio** 服务器运行时尝试运行 **interpreter --local**，结果提示选择提供商，并且在将 LM Studio 设置为提供商后仍默认使用 GPT-4。
- **分享 YouTube 教程链接**：另一位用户建议参考这个 [YouTube 教程](https://youtu.be/xPd8FFzIeOw?t=602)，以可能解决 Open Interpreter 的设置问题。
- **需要查看完整的服务器页面截图**：建议在服务器运行并选中模型的情况下，分享整个 LM Studio 服务器页面的截图，以便诊断问题。
- **MacOS 与 Linux 询问**：进行故障排除的用户提到了他们在 **MacOS** 上采取的步骤，从而引发了关于原始问题是否发生在 **Linux** 上的询问。
- **分享简单的设置步骤**：一位用户提供了在机器上设置 interpreter 的清晰步骤，这在 **MacOS** 上似乎运行良好。

**提到的链接**：<a href="https://youtu.be/xPd8FFzIeOw?t=602">ChatGPT &quot;Code Interpreter&quot; But 100% Open-Source (Open Interpreter Tutorial)</a>：这是我关于 Open Interpreter 的第二个视频，具有许多新功能和更高的稳定性，新的 Open Interpreter 非常棒。更新：Mixtral 7x8b 曾是...

  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1252325560248434861)** (1 messages): 

- **DeepSeek 发布超快速编程模型**：DeepSeek 的新 **Coding models** 现已可用，采用其 **V2 MoE** 架构，拥有 **16B 总参数**，每次请求仅激活 **2.4B**。该模型需要 **禁用 flash attention** 才能正常运行；在此处下载：[here](https://huggingface.co/lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF)。 

- **DeepSeek 的社区贡献受到关注**：**DeepSeek-Coder-V2-Lite-Instruct** 是 [LM Studio Community](https://lmstudio.ai) 模型亮点计划的一部分，该计划旨在强调新增且值得关注的模型。**GGUF 量化**由 bartowski 基于最新的 `llama.cpp` 版本提供。

**提到的链接**：<a href="https://huggingface.co/lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF">lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF · Hugging Face</a>：未找到描述

  

---

### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1251651971656581254)** (27 messages🔥): 

- **VSCode 代码脚本与模型建议集成**：一位成员分享了将 VSCode 与各种模型集成的“理想工作流”，使用 CodeParrot 和 OpenAI 的 Playground 生成脚本文件，并使用 `continue.dev` 进行代码修改和解释。他们表达了在迭代代码版本方面的挑战，并请求协助设置 `continue.dev`。

- **`continue.dev` 中的模型选择和配置建议**：另一位成员建议使用 `llama3` 或 `deepseek-coder` 等模型进行聊天，并提供了一个 `continue.dev` 的[配置文件示例](https://docs.continue.dev/setup/examples)。他们指出了与不支持的 GPU (6600XT) 相关的问题，该显卡需要 OpenCL 而非 ROCM。

- **GPU 设置问题**：一位成员在通过 ROCM 继而 OpenCL 设置 GPU 加速时遇到问题，导致反复出现关于 GPU 检测失败的错误。建议认为他们可能缺少驱动程序，并建议在特定频道寻求详细帮助。

- **在 LM Studio 中配置 `continue.dev`**：讨论强调了在 LM Studio 中为不同模型设置多个服务器的复杂性，以及在 `continue.dev` 配置中使用 `apiBase` 属性。分享了一个专门针对 LM Studio 的[设置说明](https://docs.continue.dev/reference/Model%20Providers/lmstudio)链接。

- **关于 LM Studio 的 API 使用调用**：一位成员询问如何通过 ngrok 使用 LM Studio 的 API，但得到的澄清是 LM Studio 必须在本地安装并运行才能使用其服务。
<div class="linksMentioned">

<strong>相关链接</strong>：

<ul>
<li>
<a href="https://docs.continue.dev/setup/select-model">选择模型 | Continue</a>：配置 LLMs</li><li><a href="https://docs.continue.dev/walkthroughs/tab-autocomplete#setting-up-with-lm-studio">Tab 自动补全 (beta) | Continue</a>：Continue 现在支持 VS Code 和 JetBrains IDEs 中的 Tab 自动补全。我们将在接下来的几个版本中大幅改进体验，随时欢迎反馈。如果...</li><li><a href="https://docs.continue.dev/setup/examples#i-need-to-be-entirely-local--offline">示例配置 | Continue</a>：如果您正在寻找快速创建完美 Continue 设置的方法，我们为常见情况编写了一些 config.json 示例。您可以复制这些并粘贴到您的 config.json 中...</li><li><a href="https://docs.continue.dev/reference/Model%20Providers/lmstudio">LM Studio | Continue</a>：LM Studio 是一款适用于 Mac、Windows 和 Linux 的应用程序，可以轻松地在本地运行开源模型，并配有出色的 UI。要开始使用 LM Studio，请从网站下载，使用 th...
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1251254405785976852)** (372 messages🔥🔥): 

- **针对低资源设备的自制 AI**：用户讨论了 GPT-4 的自托管 AI 替代方案，这些方案不需要强大的服务器。建议使用 **"maybe llama3 (70B-7B), mixtral 8x7B, or command r+"**。
- **FlowGPT 的 NSFW 内容**：**FlowGPT** 因可能允许 NSFW 内容而受到审查，而 **OpenAI 禁止**此类内容。一位用户认为，虽然 NSFW 机器人很常见，但明确**道德与法律层面的考量**非常重要。
- **高效微调与评估**：**Viliamvolosv** 分享了他用于改进俄语经典文学模型的 **QLoRa 设置**，寻求关于最佳参数的建议。**Fulx69** 强调了实验 r 和 alpha 值的重要性，并推荐了如 **LLaMA-Factory** 等评估工具。
- **新 AI 模型与工具**：据称 **DeepSeek-Coder-V2** 在编程和数学方面超越了 GPT-4-Turbo，用户建议使用 **LiveCodeBench** 进行公正评估。该模型发布在 [@deepseek_ai](https://x.com/deepseek_ai/status/1802680388256768145)。
- **加入与欢迎**：**9do4n1** 和 **open.group** 等新用户加入，其他成员表示欢迎并说明了服务器规则和文化。**"Welcome 🤗"** 消息强调了互助的社区环境。
<div class="linksMentioned">

<strong>相关链接</strong>：

</div>

<ul>
<li>
<a href="https://huggingface.co/spaces/eswardivi/Phi-3-mini-128k-instruct">Phi-3-mini-128k-instruct - 由 eswardivi 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/KingNish/OpenGPT-4o">OpenGPT 4o - 由 KingNish 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/OpenGVLab/InternVL">InternVL - 由 OpenGVLab 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/lllyasviel/Omost">Omost - 由 lllyasviel 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/lllyasviel">lllyasviel (张吕敏)</a>: 未找到描述</li><li><a href="https://youtu.be/R9c-_neaxeU">MENACE：可以学习的火柴盒堆</a>: 在第二个视频中查看更多数据，并了解我们在第二天做了哪些更改（这导致 MENACE 学习了不同的策略）：https://youtu.be/KcmjO...</li><li><a href="https://github.com/huggingface/trl/blob/main/examples/notebooks/gpt2-sentiment.ipynb">huggingface/trl 项目 main 分支下的 trl/examples/notebooks/gpt2-sentiment.ipynb</a>: 使用强化学习训练 Transformer 语言模型。- huggingface/trl</li><li><a href="https://huggingface.co/blog/nroggendorff/ttt-ai">训练一个蹩脚的井字棋 AI</a>: 未找到描述</li><li><a href="https://shog.ai/">ShogAI | 探索开源与去中心化 AI</a>: 未找到描述</li><li><a href="https://www.robustintelligence.com/platform/ai-firewall-guardrails">实时保护您的 AI 应用 — Robust Intelligence</a>: 保护生成式 AI 应用免受攻击和不良响应。Robust Intelligence 的防护栏（guardrails）可抵御安全和防护威胁。</li><li><a href="https://pypi.org/project/numpy/">numpy</a>: Python 数组计算的基础包</li><li><a href="https://x.com/deepseek_ai/status/1802680388256768145">来自 DeepSeek (@deepseek_ai) 的推文</a>: DeepSeek-Coder-V2：首个在编程和数学方面超越 GPT4-Turbo 的开源模型 > 在编程和数学方面表现卓越，击败了 GPT4-Turbo、Claude3-Opus、Gemini-1.5Pro、Codestral。> 支持 338 种编程...</li><li><a href="https://tenor.com/view/dead-pixels-dpgc-gif-25869325">Dead Pixels Dpgc GIF - Dead Pixels Dpgc - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/guy-arguing-guy-talking-to-wall-talking-brick-wall-gif-18667615">吵架的人 GIF - 吵架的人对墙说话 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/michael-jackson-eating-popcorn-enjoy-i-like-nom-nom-gif-11040065238845078056">迈克尔·杰克逊吃爆米花 GIF - 迈克尔·杰克逊享受吃爆米花 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/microsoft/Phi-3-vision-128k-instruct">microsoft/Phi-3-vision-128k-instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B-int4">THUDM/cogvlm2-llama3-chat-19B-int4 · Hugging Face</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/17pw7bv/eternal_question_what_rank_r_and_alpha_to_use_in/">Reddit - 深入探索任何事物</a>: 未找到描述</li><li><a href="https://github.com/Zz-ww/SadTalker-Video-Lip-Sync">GitHub - Zz-ww/SadTalker-Video-Lip-Sync: 本项目基于 SadTalkers 实现视频唇形合成的 Wav2lip。通过以视频文件方式进行语音驱动生成唇形，设置面部区域可配置的增强方式进行合成唇形（人脸）区域画面增强，提高生成唇形的清晰度。使用 DAIN 插帧的 DL 算法对生成视频进行补帧，补充帧间合成唇形的动作过渡，使合成的唇形更为流畅、真实以及自然。</a>: 本项目基于 SadTalkers 实现视频唇形合成的 Wav2lip。通过以视频文件方式进行语音驱动生成唇形，设置面部区域可配置的增强方式进行合成唇形（人脸）区域画面增强，提高生成唇形的清晰度。使用 DAIN 插帧的 DL 算法对生成视频进行补帧，补充帧间合成唇形的动作过渡，使合成的唇形更为流畅、真实以及自然。 - Zz-ww/SadTalker-Video-Lip-Sync</li><li><a href="https://www.youtube.com/watch?v=G-di38Fpgdw">示例课程：火柴盒玩井字棋</a>: 这是我的 AWS Machine Learning 课程中的一个课程示例。在此查看完整课程：https://learn.mikegchambers.com/p/aws-machine-learning-specialty...</li><li><a href="https://github.com/hiyouga/LLaMA-Factory/">GitHub - hiyouga/LLaMA-Factory: 统一 100 多个 LLM 的高效微调</a>: 统一 100 多个 LLM 的高效微调。通过在 GitHub 上创建账号来为 hiyouga/LLaMA-Factory 的开发做出贡献。</li><li><a href="https://livecodebench.github.io/">
    LiveCodeBench: 针对代码大语言模型的全面且无污染的评估
  </a>: 未找到描述</li><li><a href="https://paperswithcode.com/paper/textgrad-automatic-differentiation-via-text">Papers with Code - TextGrad: 通过文本实现的自动“微分”</a>: 🏆 GPQA 榜单 SOTA（准确率指标）</li><li><a href="https://app.rebrandly.com/public/links/share?href=rb.gy/klkbs7">Rebrandly 控制面板</a>: 未找到描述</li><li><a href="https://gist.github.com/viliamvolosv/fdd641d77721a48bf38225d088683d07">qlora 设置</a>: qlora 的设置。GitHub Gist：即时分享代码、笔记和代码片段。
</li>
</ul>

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1251363959982788639)** (5 messages): 

- **寻找商业用途的模型**：一位成员询问了用于通用支持和商业用途的最佳模型。他们指出，能够部署的最大模型规模为 7B。

- **建议进行实验**：作为回应，另一位成员建议模型的选择将取决于具体的用例、是否使用 tools/agents，以及部署/成本约束。

- **使用 GPT-4 API 的游戏截图项目**：一位成员分享了他们使用 **GPT-4 API 对游戏《镜之边缘：催化剂》（Mirror's Edge: Catalyst）的 150 多张截图进行裁剪和标注**，并根据这些图像为 **Stable Diffusion** 创建 LoRA 的经验。
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1251300382718562386)** (10 messages🔥): 

- **时间序列预测中的 RNNs vs RWKV-TS**：一位成员分享了一篇 [arXiv 论文](https://arxiv.org/abs/2401.09093)，讨论了传统 RNN 架构在时间序列任务中统治地位下降的现状。该论文介绍了一种新型的基于 RNN 的模型 **RWKV-TS**，声称其具有更好的效率、长期序列信息捕获能力和计算可扩展性。
  
- **高级 Prompt 选项对生产时间的影响**：一位成员报告称，在高峰时段禁用高级 Prompt 选项可以显著减少生产时间，提高保真度并维持场景稳定性。

- **通过网络爬虫和 RAG 增强 LLMs**：分享了一篇 [Medium 文章](https://medium.com/ai-advances/how-to-power-up-llms-with-web-scraping-and-rag-975a165587f6)，解释了如何将网络爬虫与检索增强生成（RAG）相结合来增强大语言模型（LLMs）。文中提到的技术旨在增强数据收集和 Prompt 的准确性。

- **LLMs 对劳动力市场的影响**：一位成员分享了一项研究，探讨了 [LLMs 对劳动力市场的潜在影响](https://ar5iv.labs.arxiv.org/html/2303.10130)，揭示了美国大部分劳动力的工作任务可能会因 LLMs 而发生重大变化。调查表明，低工资和高工资工人都有可能经历工作职责的转变。

- **通过 RAG 减少 AI 幻觉**：讨论了 Wired 杂志上的一篇关于使用检索增强生成（RAG）[减少 AI 幻觉](https://www.wired.com/story/reduce-ai-hallucinations-with-rag/)的文章。该方法涉及模型在生成响应之前从自定义数据库中收集信息，从而提高可靠性和准确性。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.wired.com/story/reduce-ai-hallucinations-with-rag/">通过这个巧妙的软件技巧减少 AI 幻觉</a>：一种名为检索增强生成（RAG）的热门流程正在硅谷流行，并改善了大语言模型的输出。它是如何工作的？</li><li><a href="https://www.facebook.com/share/p/rZucZJuafBDTZyVa/">枯燥男人俱乐部 | 你好，我是一名数学家，喜欢思考那些如果实现会很有用但根本没机会实现的事情 | Facebook</a>：你好，我是一名数学家，喜欢思考那些如果实现会很有用但根本没机会实现的事情。例如，我经常认为我们可以改进日历……</li><li><a href="https://arxiv.org/abs/2401.09093">RWKV-TS：超越传统的时间序列任务循环神经网络</a>：传统的循环神经网络（RNN）架构，如 LSTM 和 GRU，历来在时间序列任务中占据重要地位。然而，最近它们的统治地位有所下降……</li><li><a href="https://ar5iv.labs.arxiv.org/html/2303.10130">GPTs 即 GPTs：初步探讨大语言模型对劳动力市场的潜在影响</a>：我们调查了大语言模型（LLMs），如 Generative Pre-trained Transformers (GPTs)，对美国劳动力市场的潜在影响，重点关注由于能力提升而带来的影响……
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1251560066600140830)** (18 条消息🔥): 

- **介绍 Difoosion - 一个 Stable Diffusion 的 Web 界面**：一位成员展示了他们为 Stable Diffusion 开发的 Web 界面，该界面利用了 `diffusers` 库和纯 Python Web 框架 Rio。他们邀请社区前往 [GitLab](https://gitlab.com/mad-moo/difoosion) 查看。

- **Ask Steve - 将 LLMs 集成到 Chrome**：一位成员开发了一个 Chrome 扩展程序，将 LLMs 直接集成到浏览器中，类似于用于网页导航的 GitHub Copilot。他们介绍该工具是消除重复性任务的一种方式，并在 [Ask Steve](https://www.asksteve.to) 推广了该项目。

- **用于语音转换的 Ilaria RVC**：一位成员宣布创建了 Ilaria RVC，这是一个运行在 Zero 上的语音转换 Space，并感谢了另一位用户的帮助。他们在 [Hugging Face Spaces](https://huggingface.co/spaces/TheStinger/Ilaria_RVC) 分享了该项目。

- **通过 Transformers.js 演示 LLM Temperature 参数**：分享了一篇关于 LLM 中 Temperature 参数的博文，其中包含一个通过在浏览器中直接运行 Transformers.js 实现的交互式演示。作者强调了这种方法如何通过消除托管模型的需求来彻底改变教育内容，该内容分享在 [Twitter](https://x.com/taha_yssne/status/1802607279809630562) 上。

- **PowershAI - 将 PowerShell 与 AI 结合**：一位成员介绍了 PowershAI，这是一个允许 Function Calling 与 AI 集成的 PowerShell 模块，是他们在学习 OpenAI API 时开发的。他们在 [GitHub](https://github.com/rrg92/powershai) 上分享了进展，并在博文中详细介绍了他们的历程。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/TheStinger/Ilaria_RVC">Ilaria RVC - TheStinger 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/ptx0/sd3-reality-mix">SD3 Reality Mix (Finetune) - ptx0 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://youtu.be/zRxJunvT3vo?si=UivatIByKGkwjCDX">具有 Layer-2 区块链交易能力的自主 Agents</a>：此视频将引导你将 GPT 转换为可以进行 Layer-2 纳米支付（nano-payments）的 Agent。了解更多：Dhali - https://dhali.io</li><li><a href="https://www.asksteve.to">Ask Steve - 在任何网页中解锁 ChatGPT 和 Gemini 的力量！</a>：Ask Steve 为任何网页添加来自 ChatGPT 和 Gemini 的 AI 超能力，让你能更好、更快地完成日常任务。免费！</li><li><a href="https://x.com/taha_yssne/status/1802607279809630562">Taha Yassine (@taha_yssne) 的推文</a>：我刚写了一篇关于 LLMs 中 Temperature 参数的博文，但实际上这只是玩玩 Transformers.js 的借口。我在实现 T 对生成影响的交互式演示中获得了乐趣...</li><li><a href="https://gitlab.com/mad-moo/difoosion">Jakob Pinterits / Difoosion · GitLab</a>：一个简单的 Stable Diffusion Web 界面 - 包括新的 Stable Diffusion 3</li><li><a href="https://github.com/rrg92/powershai">GitHub - rrg92/powershai: IA com PowerShell</a>：IA com PowerShell。通过在 GitHub 上创建一个账户来为 rrg92/powershai 的开发做出贡献。</li><li><a href="https://iatalk.ing/powershai-powershell-inteligencia-artificial/">PowershAI: PowerShell + Inteligência Artificial &#8211; IA Talking 🤖 </a>：未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1251433087720751176)** (16 条消息🔥): 

- **ViT 中的 QKV 受到质疑并计划进行实验**：一位用户质疑了 ViTs 中 QKV 实现的正确性，称其为“错误”，并承诺进行实验以提供见解。未来几天将有更多相关信息。

- **HyperZZW 对比 Self-Attention**：一位成员分享了对 ViTs 中 **self-attention** 机制的批评，提议将 **HyperZZW operator** 作为一种更简单、更合理的替代方案。他们链接了 [X (Twitter)](https://x.com/harvie_zhang/status/1763867695383208382) 上的一篇详细帖子，暗示它能更好地处理空间信息。

- **Global HyperZZW 与 tokenization 问题**：同一位用户认为，在 ViTs 中将图像转换为 tokens 存在根本性缺陷，而 **Global HyperZZW** 分支可以通过矩阵乘法策略更高效地管理全局位置信息。

- **图像和文本数据的不同策略**：他们还强调图像和文本有本质区别，这使得 ViT 的实现不适用于视觉数据，并暗示在未来的序列建模中使用先验信息（prior information）而非 attention 机制。

- **Slow neural loss 与局部反馈误差**：诸如 **slow neural loss** 作为局部反馈误差（local feedback error）的贡献已得到验证，并被提及为下一代架构的潜在关键元素，其灵感来自 Hinton 的提议。这通过另一个 [Twitter 链接](https://x.com/harvie_zhang/status/1802356749170778574?s=46&t=BsqYoGA8vIHGcXwORlMk7w) 进行了推广。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/harvie_zhang/status/1763867695383208382.">来自 Harvie Zhang (@harvie_zhang) 的推文</a>：我提议使用具有线性复杂度的 #HyperZZW 算子来替代 #SelfAttention 机制。像素级分数是通过大型隐式核与输入激活之间的 Hadamard 积获得的...</li><li><a href="https://x.com/harvie_zhang/status/1802356749170778574?s=46&t=BsqYoGA8vIHGcXwORlMk7w.">来自 Harvie Zhang (@harvie_zhang) 的推文</a>：你认为你提出的 loss 与我的 slow neural loss 之间有什么区别吗？请参考 https://arxiv.org/pdf/2401.17948 中的公式 11-12。引用 Francesco Faccio (@FaccioAI) ...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1251472397895336027)** (4 条消息): 

- **询问 VASA 模型**：一位成员询问是否有人找到了 **类 VASA 的开源模型**。在提供的消息中没有后续行动或回复的迹象。 
- **对 mobile CLIP 的兴趣**：另一位成员询问 **Hugging Face** 是否会实现 **mobile CLIP 模型**。在提供的消息中没有关于此问题的进一步讨论或回复。
  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1251495869165862922)** (5 messages): 

- **分享了微调 BERT 的方法**：一位成员建议使用[此教程](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb)中概述的方法来微调 BERT。
- **HF 模型加载中的随机性问题**：一位用户提到，多次加载 HuggingFace 模型会导致不同的验证输出，建议保存未训练的模型状态以实现可复现性。他们指出：*"不要依赖 HF 初始化的确定性... 保存你未训练的模型状态"*。
- **Mistral-7b-0.3 上下文处理问题**：一位新成员遇到 Mistral-7b-0.3 模型无法正确处理上下文长度的问题，无法回答超出上下文前半部分的问题。他们正在寻求指导，以确认是否误解了模型的能力。
- **新的开源 TTS 模型**：一位成员分享了一个新的 TTS 模型 [MARS5-TTS](https://github.com/Camb-ai/mars5-tts)，并邀请其团队参加 Mozilla AI 主舞台的演讲。他们请求社区提交任何想问 MARS5-TTS 团队的问题。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/Camb-ai/mars5-tts">GitHub - Camb-ai/MARS5-TTS: MARS5 speech model (TTS) from CAMB.AI</a>：来自 CAMB.AI 的 MARS5 语音模型 (TTS)。可以通过创建 GitHub 账号参与开发。</li><li><a href="https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb">Transformers-Tutorials/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb at master · NielsRogge/Transformers-Tutorials</a>：此仓库包含 NielsRogge 使用 HuggingFace 的 Transformers 库制作的演示。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1251264088705597510)** (5 messages): 

- **迷因生成器模型的困扰**：一位成员寻求关于开发**高质量迷因（meme）生成器模型**的建议，并向在该领域有经验或感兴趣的人寻求指导。他们强调希望生成*高质量的迷因*，并想知道初步步骤。

- **速率限制错误阻碍进展**：一位成员报告了 *rate limit exceeding errors*（超出速率限制错误），并请求帮助解决此问题。

- **Stable Diffusion XL 中的溢出错误**：分享了一个涉及 SDXL 加载的详细错误，显示 *Overflow error: cannot fit 'int' into an index-sized integer*。提供的代码片段和系统信息（包括 **GPU: A100** 和 **Torch: 2.3.1**）构成了上下文的一部分。

- **寻求在 GCP 的 TPU 上使用 Diffusers 的示例**：另一位成员请求关于在 **GCP** 的 **TPU** 上使用 **Diffusers** 的示例或指导。

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1251266511373336626)** (184 messages🔥🔥): 

- **ChatGPT 在 iOS 18 上的情况仍不明朗**：一位成员询问 ChatGPT 是否能在 iOS 18 上运行，另一位成员指出不要安装测试版软件，强调了为 ChatGPT 使用像 iOS 17 这样稳定的 iOS 版本的重要性。他们还提到测试版用户签署了关于新功能的 NDA。

- **从 YouTube 视频中提取转录文本**：成员们讨论了从 YouTube 视频中提取转录文本的工具，包括像 **Otter.ai** 这样的 AI 工具，以及一个需要通过 `fabric` 库使用 YouTube API 密钥的特定工具。一位成员建议使用 Google Gemini 的试用版以获得更佳的用户体验。

- **开源模型在特定任务中击败 GPT-4**：[DeepSeek AI](https://x.com/deepseek_ai/status/1802680388256768145) 发布了一个开源模型，据报道在 coding 和 math 等专业任务中表现优于 GPT-4 Turbo。这引发了关于开源与专有模型的讨论。

- **将 OpenAI 模型连接到数据库**：一位成员询问如何将 OpenAI 的 LLM 与持续更新的数据库集成，另一位成员分享了 [OpenAI's Cookbook](https://cookbook.openai.com/examples/vector_databases/readme) 的链接，其中包含 vector databases 的示例，这些数据库是支持语义搜索和减少响应中 hallucinations（幻觉）的基础。

- **Dream Machine 和 Sora 的 AI 能力**：大家热烈讨论了 **Luma's Dream Machine** 的视频生成能力，并将其与备受期待的 **Sora** 进行了比较，反映出一些用户对 Sora 限量发布的失去耐心。成员们注意到其功能令人印象深刻但仍在进化中，具有合并一致物理运动等独特功能。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/deepseek_ai/status/1802680388256768145">来自 DeepSeek (@deepseek_ai) 的推文</a>: DeepSeek-Coder-V2：首个在 Coding 和 Math 领域击败 GPT4-Turbo 的开源模型。在 coding 和 math 方面表现出色，击败了 GPT4-Turbo、Claude3-Opus、Gemini-1.5Pro、Codestral。支持 338 种编程语言...</li><li><a href="https://x.com/deepseek">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/sanchit-gandhi/whisper-jax">Whisper JAX - sanchit-gandhi 开发的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://copilot.microsoft.com/sl/b6od4WvwBVs">什么是 Copilot? - Microsoft Copilot: 你的日常 AI 助手</a>: Microsoft Copilot 利用 AI 的强大功能来提高工作效率、释放创造力，并通过简单的聊天体验帮助你更好地理解信息。</li><li><a href="https://tenor.com/view/potato-fries-poop-gif-5060738">Potato Fries GIF - Potato Fries Poop - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://cookbook.openai.com/examples/rag_with_graph_db">使用图数据库进行 RAG | OpenAI Cookbook</a>: 未找到描述</li><li><a href="https://cookbook.openai.com/examples/vector_databases/readme">Vector databases | OpenAI Cookbook</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1251274066560811080)** (49 messages🔥): 

- **自定义 GPTs 隐私设置的困惑**：一位成员在将他们的 Custom GPTs 设置为私有时遇到困难，提到目前限制最严格的选项是 "invite only"（仅限邀请），但它仍然显示为对所有人可用。建议的解决方法是创建一个副本并将其限制为拥有链接的人，然后删除原始版本。
- **关于 GPT 模组（Mod）的有趣想法**：一位成员建议制作一个《辐射》（Fallout）或《天际》（Skyrim）的 Mod，将游戏中所有的对话改为 zoomer 俚语或任何指定的 Prompt，并表示这会非常有趣。
- **免费层级 GPT 交互的访问问题**：几位成员报告了访问 GPT 交互的困难，对话需要付费订阅才能继续。这似乎影响了多个用户，一些人确认他们的朋友也遇到了同样的问题。
- **为自定义 GPTs 指定操作**：一位用户询问如何在他们的 Custom GPT 中设置特定操作（如网页浏览），并被建议相应地提示 GPT 何时使用某些工具。
- **对 GPT 使用限制的沮丧**：另一位用户对 GPT 无法加载和服务器宕机表示沮丧，其他人也确认了类似的问题。对于实时更新，用户被引导查看 [status.openai.com](https://status.openai.com)。
  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1251273943017459764)** (28 条消息🔥): 

- **无阴影 3D 模型面临挑战**：成员们讨论了创建无阴影或无光照 3D 模型的难题。一位成员希望创建类似于 "albedo map"（反照率贴图）的纹理以辅助 3D 转换，而另一位则建议使用 inpainting 或 Canva 等工具来减少阴影。

- **从 GPT-4 提取信息**：一位成员遇到了 GPT-4 在信息提取过程中混淆样本邮件和目标邮件的问题。解决方案包括使用明显的标记清晰地分隔样本，并明确指令。

- **使用 ChatGPT 生成详细路线图**：为了深入探讨营销和品牌建设等主题，成员们推荐了 step-back prompting 和使用详细查询等策略。分享的技巧包括创建主题树以及使用浏览器工具进行特定研究。

- **处理 ChatGPT 的拒绝请求**：一位用户遇到 ChatGPT 连续拒绝执行某些请求且未给出明确原因的情况。分享的技巧包括重复 Prompt，并在请求执行的同时要求其给出详细解释。

- **从 XML 数据生成表格**：一位成员询问了关于将 XML 数据提取为表格形式以及通过 GPT API 生成特定 Token 数量的 Prompt。社区正在等待对这一技术问题的进一步回复。
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1251273943017459764)** (28 条消息🔥): 

- **3D 模型 Prompt 秘籍**：一位成员建议寻找 3 个无阴影或无光照的 3D 模型示例，让 ChatGPT 记录它们的缺失特征，然后生成新图像。另一位用户指出，由于语言限制以及 ChatGPT 和 Dall-E 的渲染修正，完全消除阴影似乎是不可能的。

- **在 GPT-4 中使用独立样本**：为了防止 GPT-4 混淆样本邮件和目标邮件，成员们讨论了使用不同标记的方法。清晰的分隔和具体的指令可以防止内容混淆。

- **平衡 3D 模型中的阴影**：随后展开了关于减少物体阴影和光照以获得更好 3D 模型纹理贴图的详细讨论。共识是烘焙阴影会干扰 albedo map 的创建，建议改用生成的形状作为基础模型。

- **使用 ChatGPT 生成营销路线图**：一位用户寻求利用 ChatGPT 获取关于品牌原型（Brand Archetypes）等营销主题的深度见解。成员们建议使用 step-back prompting 和子主题的特定路线图；建议包括使用清晰的指令和外部资源进行深入研究。

- **ChatGPT 的拒绝怪癖**：几位用户报告称 ChatGPT 有时会在不给出理由的情况下拒绝请求。提议的解决方法包括要求 ChatGPT 解释拒绝原因，这可能会促使它执行请求。
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1251322975228268554)** (250 条消息🔥🔥): 

- **SD3 模型在伪影和训练方面面临困难**：成员们讨论了 SD3 模型的稳定性和训练挑战，指出 **loss stabilization** 仍然很复杂。大家对 *non-uniform timestep sampling* 以及缺乏 **qk norm** 等关键组件表示了明确的担忧。

- **训练中的时间步权重**：讨论强调了 **V-prediction models** 中时间步权重的不同处理方法。一位用户更倾向于在重新加权损失的同时使用 *uniform sampling*，将 schedule 分割成更小的 batch 以有效地分配训练。

- **开源 T2I 模型**：关于具有角色一致性的最佳 **开源 T2I 模型** 的咨询和推荐引向了用于可控文本生成图像的 [GitHub 资源](https://github.com/PRIV-Creation/Awesome-Controllable-T2I-Diffusion-Models)。[用于角色管理的 Theatergen](https://github.com/donahowe/Theatergen) 也作为实现一致的多轮图像生成的选项被讨论。

- **ComfyUI 与自适应 ODE 求解器**：一位成员分享了为 SD3 实现的自适应 ODE 求解器的 [GitHub 链接](https://github.com/redhottensors/ComfyUI-ODE)，认为它们比现有的固定步长求解器效果更好，可以作为当前 diffusers 的宝贵参考或替代方案。

- **复旦大学开源视频生成模型**：围绕复旦大学用于从单张图像和音频生成视频的 [Hallo 模型](https://github.com/fudan-generative-vision/hallo) 展开了热烈讨论，[FXTwitter](https://fxtwitter.com/cocktailpeanut/status/1802376983021580428) 上还分享了另一个本地运行该模型的工具。成员们对将其与 Udio 或 Suno 等 Text-to-Speech 系统集成表示了兴趣。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://runwayml.com/blog/introducing-gen-3-alpha/">Introducing Gen-3 Alpha: A New Frontier for Video Generation</a>: Gen-3 Alpha 是由 Runway 在为大规模多模态训练构建的新基础设施上训练的首个下一代基础模型。它在保真度、一致性方面有重大提升...</li><li><a href="https://fxtwitter.com/JoeSiyuZhu/status/1801780534022181057">Tweet from Siyu ZHU (@JoeSiyuZhu)</a>: 我们（复旦、百度、苏黎世联邦理工学院 ETH Zurich）开源了性能卓越的视频生成模型，可以让单张图像根据音频参考进行歌唱和交谈，并能自适应地控制面部表情。...</li><li><a href="https://huggingface.co/CaptionEmporium">CaptionEmporium (Caption Emporium)</a>: 未找到描述</li><li><a href="https://bottosson.github.io/posts/oklab/">A perceptual color space for image processing</a>: 用于图像处理的感知色彩空间。在进行多种图像处理时，感知色彩空间是非常理想的。它很有用...</li><li><a href="https://tenor.com/view/tonton-friends-yuta-chubby-shiba-shiba-inu-dog-gif-16103647401394133233">Tonton Friends Yuta GIF - Tonton Friends Yuta Chubby Shiba - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/im-doing-my-part-serious-stare-gif-11979677">Im Doing My Part Serious GIF - Im Doing My Part Serious Stare - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/thats-the-neat-part-you-dont-invincible-gif-27194608">Thats The Neat Part You Dont Invincible GIF - Thats The Neat Part You Dont Invincible - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/oppo-us-research/FA-VAE">GitHub - oppo-us-research/FA-VAE: Description： Frequency Augmented Variational Autoencoder for better Image Reconstruction</a>: 描述：用于更好图像重建的频率增强变分自编码器 (Frequency Augmented Variational Autoencoder) - oppo-us-research/FA-VAE</li><li><a href="https://fxtwitter.com/cocktailpeanut/status/1802381406196154570">Tweet from cocktail peanut (@cocktailpeanut)</a>: Kanye 演唱 BTS 的 "Dynamite"</li><li><a href="https://fxtwitter.com/cocktailpeanut/status/1802376983021580428">Tweet from cocktail peanut (@cocktailpeanut)</a>: 本地一键运行高质量口型同步。[仅限 NVIDIA] 我所见过的 Hallo 口型同步质量是最好的。所以我为此编写了一个 Gradio 应用和一键启动器。尽情享受吧！...</li><li><a href="https://github.com/PRIV-Creation/Awesome-Controllable-T2I-Diffusion-Models">GitHub - PRIV-Creation/Awesome-Controllable-T2I-Diffusion-Models: A collection of resources on controllable generation with text-to-image diffusion models.</a>: 关于使用文本到图像扩散模型进行可控生成的资源合集。 - PRIV-Creation/Awesome-Controllable-T2I-Diffusion-Models</li><li><a href="https://github.com/donahowe/Theatergen">GitHub - donahowe/Theatergen: TheaterGen: Character Management with LLM for Consistent Multi-turn Image Generation</a>: TheaterGen：利用 LLM 进行角色管理，以实现一致的多轮图像生成 - donahowe/Theatergen</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1dhd7vz/the_developer_of_comfy_who_also_helped_train_some/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://replicate.com/fofr/consistent-character?prediction=v3b629v38drgm0cg3h1bmmz1zm">fofr/consistent-character – Run with an API on Replicate</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1251316649118797944)** (34 条消息🔥): 

- **AIW 问题的逻辑推理挑战**：讨论强调了逻辑推理问题中经常使用像 "Alice" 这样的名字，这可能会使 LLM 产生偏见。一位成员分享道，*"Phi-2 的总体表现非常糟糕"*，在 [这篇论文](https://arxiv.org/abs/2406.02061) 描述的 AIW 问题上，SOTA LLM 表现出了 *"严重的推理崩溃"*。
  
- **应对偏见的实验策略**：一位成员通过改变问题设置来消除已知示例中的偏见进行了实验，并指出像 *"GPT4o 和 Claude-Opus 成功解决了它"*，而其他模型则失败了。失败归因于 LLM 的误解，例如错误处理分组或幻觉出几何关联。

- **模型的推理敏感性**：进一步分析显示，LLM 对 *"哪怕是轻微的 AIW 问题变体也极其敏感"*，[引用论文](https://arxiv.org/abs/2406.02061) 中的图 11 展示了正确响应率随微小变化而产生的剧烈波动，强调了它们推理能力的脆弱状态。

- **用于演绎推理的符号 AI 混合体**：关于结合 LLM 与符号 AI 以改进演绎推理的研究咨询，引荐了 [Logic-LM](https://arxiv.org/abs/2305.12295)，它将 LLM 与符号求解器集成，显著提升了逻辑问题的解决性能。

- **利用 JEPA 在邮件助手中构建集体愿景**：Anu4938 分享了使用 [JEPA](https://openreview.net/pdf?id=BZ5a1r-kVsf) 创建邮件助手的宏图，旨在最大化集体利益并高效管理复杂性。设想中的助手强调诸如尊重环境、应对气候变化行动以及促进全球合作等价值观。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2406.10208">Glyph-ByT5-v2: A Strong Aesthetic Baseline for Accurate Multilingual Visual Text Rendering</a>：最近，Glyph-ByT5 在平面设计图像中实现了高度准确的视觉文本渲染性能。然而，它仍然仅专注于英语，在视觉方面表现相对较差...</li><li><a href="https://www.anthropic.com/research/claude-character">Claude’s Character</a>：Anthropic 是一家 AI 安全和研究公司，致力于构建可靠、可解释且可控的 AI 系统。</li><li><a href="https://x.com/JJitsev/status/1801760448657727737">来自 Jenia Jitsev 🏳️‍🌈 🇺🇦 (@JJitsev) 的推文</a>：社区正在深入研究我们的工作 https://arxiv.org/abs/2406.02061，该研究展示了 SOTA LLM 在非常简单的 AIW 问题上出现的严重推理崩溃。经过激烈辩论，为了进一步展示...</li><li><a href="https://arxiv.org/abs/2305.12295">Logic-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning</a>：大语言模型 (LLM) 展示了类人的推理能力，但在复杂的逻辑问题上仍然挣扎。本文介绍了一个新颖的框架 Logic-LM，它将 LLM 与符号...
</li>
</ul>

</div>
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1251282157679280268)** (161 条消息🔥🔥): 

- **关于大模型可用性的辩论**：针对发布和恶搞 200T 参数模型（这类模型超出了大多数用户的硬件能力）展开了激烈讨论。一位用户幽默地提到：“我离制作一个 200T 参数的模型就差这么一点点了。然后宣称它是 AGI。”

- **Qwen7B vs Llama3 8B**：成员们讨论了 Qwen7B 和 Llama3 8B 之间的性能对比，一位用户提到像 Qwen7B 这样的小型 LLM 不太可能超越 Llama3 8B，并强调了后者目前在该领域的领先地位。

- **自定义 Llama3 模板问题**：针对使用 Llama3 模型进行训练时的训练配置以及与 `chat_template` 设置相关的问题进行了详细的技术交流。一位用户分享了一个[修复自定义 Llama3 提示策略](https://github.com/xzuyn/axolotl/blob/dan_metharme/src/axolotl/prompt_strategies/customllama3.py)的链接，该链接解决了一些问题。

- **PyTorch 的 GPU 和优化反馈**：征集使用各种 GPU 的用户反馈以协助 PyTorch 优化，收到了多样化的回复，涉及的硬件包括 AMD MI300X、RTX 3090、Google TPU v4 以及运行 tinygrad 的 4090。

- **共享项目与资源**：用户分享了多个资源，包括关于 [CryptGPT: Privacy-Preserving LLMs](https://x.com/diwanksingh/status/1802118343446724655) 的博客文章、特定语言的 GPT 聊天模型 [DanskGPT](https://chat.danskgpt.dk)，以及用于设置类似于 HuggingChat 聊天 UI 的 GitHub 链接，使用的是 [Huggingface 的 chat-ui 项目](https://github.com/huggingface/chat-ui)。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/turboderp/Cat-Llama-3-70B-instruct">turboderp/Cat-Llama-3-70B-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/diwanksingh/status/1802118343446724655">来自 Diwank Singh (@diwanksingh) 的推文</a>：http://x.com/i/article/1802116084507848704</li><li><a href="https://x.com/_philschmid/status/1801732678233825327?t=qiUAp1TbPUTcwQi1ikPy5Q&s=19">来自 Philipp Schmid (@_philschmid) 的推文</a>：今天真是个幸运的日子。@nvidia 发布了一个强大的 340B 开源 LLM，需要 8x H200 才能运行。就在同一天，第一批供应商开始提供 8x H200 托管服务。🍀</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/5783839c6e29bb148041338772040c85aaae4646/src/axolotl/cli/train.py#L53-L59">axolotl/src/axolotl/cli/train.py (GitHub)</a>：欢迎提出 axolotl 相关问题。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/huggingface/chat-ui">GitHub - huggingface/chat-ui: 驱动 HuggingChat 应用的开源代码库</a>：驱动 HuggingChat 应用的开源代码库。通过在 GitHub 上创建账号为 huggingface/chat-ui 的开发做出贡献。</li><li><a href="https://github.com/huggingface/trl/blob/main/trl/trainer/rloo_trainer.py">trl/trl/trainer/rloo_trainer.py (GitHub)</a>：使用强化学习训练 Transformer 语言模型。- huggingface/trl</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/5783839c6e29bb148041338772040c85aaae4646/src/axolotl/prompt_strategies/sharegpt.py#L42-L55">axolotl/src/axolotl/prompt_strategies/sharegpt.py (GitHub)</a>：欢迎提出 axolotl 相关问题。</li><li><a href="https://github.com/xzuyn/axolotl/blob/dan_metharme/src/axolotl/prompt_strategies/customllama3.py">axolotl/src/axolotl/prompt_strategies/customllama3.py (GitHub)</a>：在 dan_metharme 分支下的代码。</li><li><a href="https://github.com/lm-sys/FastChat/tree/main?tab=readme-ov-file#method-2-from-source">GitHub - lm-sys/FastChat: 一个用于训练、部署和评估大语言模型的开放平台。</a>：Vicuna 和 Chatbot Arena 的发布仓库。</li><li><a href="https://chat.danskgpt.dk">DanskGPT</a>：丹麦语技术，对所有人免费开放。
</li>
</ul>

</div>
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1251709838606602311)** (4 条消息): 

- **Llama3 Bug 导致开发停滞**：一位用户在 [GitHub](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1700) 上提交了一个关于 6 月 7 日引入的 Bug 的问题，该 Bug 会导致无法微调 Llama 3 或 Mistral 模型。该 Bug 影响了多位用户，已有 6 人确认受影响。虽然存在临时解决方案，但用户坚持认为需要修复 main 分支。
- **调查 Bug 来源**：另一位成员询问该问题是否与将 `remove_unused_column` 设置为 false 有关，但随后得出结论，"length" 关键字参数问题可能源于特定的 commit。在进行 `bisect` 后确定了有问题的 commit，确认了其为问题根源。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl/issues/1700">Llama3-8b: LlamaForCausalLM.forward() got an unexpected keyword argument &#39;length&#39; · Issue #1700 · OpenAccess-AI-Collective/axolotl</a>：请检查此问题之前是否已被报告。我搜索了之前的 Bug 报告，没有发现类似的报告。预期行为：我期望训练运行能够完成并保存...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/issues/17">Jeopardy bot! by winglian · Pull Request #17 · OpenAccess-AI-Collective/axolotl</a>：https://huggingface.co/openaccess-ai-collective/jeopardy-bot
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1251299743582126090)** (9 条消息🔥): 

- **数据集类型的配置困惑**：一位用户对其 `axolotl` 配置中的数据集类型字段表示困惑，特别是关于 `alpaca_chat.load_qa`，并引用了 [dataset formats](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/inst_tune.html#alpaca_chat.load_qa) 文档。另一位用户确认所提供的配置格式是正确的。

- **在 SLURM 集群上运行 accelerate**：一位用户分享了一个用于通过 `accelerate` 和 `deepspeed` 运行 `axolotl` 的 SLURM 作业脚本，指定了混合精度和多 GPU 设置。他们建议如果 `$PMI_RANK` 不可用，可以将其替换为 `$SLURM_NODEID`。

- **Axolotl 中的 QDora 问题**：一位用户询问如何在 Axolotl 中使 QDora 正常工作，另一位用户回复说它在运行几步后会挂起，暗示其目前不可靠。用户进一步寻求关于从源码构建 QDora 的详细信息。

- **使用 axolotl 进行人格提取**：一位用户询问是否有人使用 Axolotl 训练模型从文本中提取人格（personalities），并链接到 [Delphi AI](https://www.delphi.ai/) 作为参考。他们询问 `oasst` 数据集格式是否合适，并链接到了[相关文档](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/inst_tune.html#oasst)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.delphi.ai/?">Delphi</a>：克隆你自己。</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/inst_tune.html#oasst?">Axolotl - Instruction Tuning</a>：暂无描述</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/inst_tune.html#alpaca_chat.load_qa">Axolotl - Instruction Tuning</a>：暂无描述</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/config.html">Axolotl - Config options</a>：暂无描述
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1251470432813580379)** (4 条消息): 

- **数据集配置问题已解决**：一位成员因遇到 `ValueError` 提示 *"unhandled prompt tokenization strategy: sharegpt"* 而请求数据集配置部分的帮助。另一位成员分享了来自 Discord 的配置链接（[链接](https://discord.com/channels/1104757954588196865/1104757955204743201/1251481036848889927)），该问题随后得到解决。

### **OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1251620995874295932)** (1 messages): 

- **使用 Axolotl 完成的首次 Finetune 表现出色**："使用 Axolotl 进行首次 LLM finetune 的体验非常棒！" 作者报告称成功将非结构化的新闻稿转换为结构化输出，并暗示将进一步探索 OpenAI API 的 function calling 以提高准确率。
- **探索新闻稿数据提取效率**："[我们之前研究过](https://mlops.systems/posts/2024-06-03-isafpr-evaluating-baseline.html) LLM 从新闻稿中提取结构化数据的效果。" 初步评估显示，虽然 LLM 表现尚可，但仍有明显的提升空间。
- **承诺未来的对比测试**：强调使用 function calling 优于原始 prompting 以获得更好的准确率，并暗示将发布一篇关于 finetune 对比的独立文章。欲了解更多详情，作者引导读者阅读[一篇详细文章](https://mlops.systems/posts/2024-06-15-isafpr-first-finetune.html)。

**提到的链接**：<a href="https://mlops.systems/posts/2024-06-15-isafpr-first-finetune.html">Alex Strick van Linschoten - 使用 axolotl 为结构化数据提取任务 finetune 我的第一个 LLM</a>：我针对从 ISAF 新闻稿中提取结构化数据的任务 finetune 了我的第一个 LLM。初步测试表明，它在开箱即用的情况下表现相当不错。

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1252235292166066227)** (11 messages🔥): 

- **在 Axolotl 中调整 Inference 参数**：一位用户询问在运行 `accelerate launch -m axolotl.cli.inference` 时如何设置 temperature 或 seed 等 inference 参数。建议直接修改 inference 脚本或配置文件（如果命令行参数不支持这些设置），并展示了如何调整 `generation_config` 的示例。

- **微调 Vision Models 的请求**：一位用户咨询了关于 finetune vision models 的问题。解释称该过程包括加载预训练模型（例如 ResNet-50）、准备数据集、必要时修改最终层、定义 data transforms，然后使用适当的 loss function 和 optimizer 设置训练循环。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz">未找到标题</a>：未找到描述</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=077f3d3e-542e-4b6f-b825-cea95799abf7)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=8bf72d5d-fdad-4aa8-9c16-12b8b7bad9d9)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1252211724786864158)** (11 messages🔥): 

- **模型 context length 翻倍需要仔细调整**：要在 **2 倍原生 context length**（例如从 8k 到 16k）下训练模型，用户需要修改与模型架构、数据处理和训练配置相关的多项设置。关键更改包括调整 maximum position embeddings 以及 batch size 和 gradient accumulation steps 等训练参数。

- **Axolotl 微调 Vision Models 详解**：提供了一个使用 Axolotl finetune vision models 的逐步指南。包括克隆 Axolotl 仓库、安装依赖、准备数据集、修改配置文件，以及使用 Accelerate 进行训练和 inference。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>：尽管提问。通过创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=46936261-cfc5-4e25-8c3f-1048a546f037)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=0be694a5-7efc-4cdb-97f6-6691bd442899)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1252303453028614204)** (1 条消息): 

- **好奇心通晓每种语言；与 SoftBank 达成合作**：Perplexity 宣布与 **SoftBank** 达成战略合作伙伴关系，为使用 SoftBank、Y!mobile 和 LINEMO 服务的客户提供为期一年的免费 **Perplexity Pro**。该高级版 Perplexity 价值每年 29,500 日元，为用户提供了一个革命性的 AI 问答引擎，用于探索和学习。[更多信息](https://pplx.ai/softbank)。

**提到的链接**：<a href="https://pplx.ai/softbank">SoftBank Corp. 与领先的 AI 初创公司 Perplexity 启动战略合作 | 关于我们 | SoftBank</a>：SoftBank Corp. 的公司页面提供了关于“SoftBank Corp. 与领先的 AI 初创公司 Perplexity 启动战略合作”的信息。

  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1251271508668711033)** (187 条消息🔥🔥): 

- **Agentic Search A/B 测试秘籍**：社区成员讨论了正在进行 A/B 测试的新版 [Agentic Pro Search](https://www.reddit.com/r/perplexity_ai/comments/1ddkvgc/comment/l8czo5j/)。一位用户分享了一个关于如何“绕过”系统限制的 Reddit 链接，但随后为了避免干扰对照组而重新考虑。
- **对 Perplexity 功能和模型设置的困惑**：用户对使用 Perplexity 提出了各种问题，例如设置 system prompt、格式化回答、访问写作模式，以及遇到 temperature 变化或聊天冻结等问题。他们分享了如联系支持团队或清除浏览器缓存等解决 Bug 的方案。
- **Perplexity vs. ChatGPT 及投资讨论**：成员们辩论了同时订阅 Perplexity 和 ChatGPT 是否值得，并讨论了投资 Perplexity 的潜力。对比集中在每个平台在写作和研究等特定用例中的优势。
- **对 Web Crawling 和隐私的担忧**：一些用户对 Perplexity 的抓取行为表示担忧，认为其不尊重 `robots.txt` 并掩盖了 user agents。针对屏蔽或解决此问题的建议包括使用 JA3 指纹识别和 Bot 端点。
- **可定制功能和文档处理**：成员们询问并讨论了上传文件、处理大量文档集，以及与 academia.edu 等学术数据库集成的可能性。解决方案包括使用其他 AI 工具（如 OpenAI 上的自定义 GPTs）和 NotebookLM 来管理大量文档。

**提到的链接**：<a href="https://www.reddit.com/r/perplexity_ai/comments/1ddkvgc/comment/l8czo5j/">Reddit - 深入探索一切</a>：未找到描述

  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1251336979992547338)** (10 条消息🔥): 

- **分享 Tanstack Table 搜索**：一位成员分享了一个与 Tanstack Table 查询相关的 [Perplexity AI 搜索链接](https://www.perplexity.ai/search/tanstack-table-vJoqV4R9R4iPUHhGspaL2A)。这引发了对数据表格管理工具的兴趣。
  
- **俄罗斯宠物食品搜索**：另一位成员提供了一个关于 [俄罗斯宠物食品搜索](https://www.perplexity.ai/search/petfood-in-Russia-UzyFxsq3RnCxXSu.x3BbBg) 的链接。讨论可能围绕俄罗斯的宠物食品市场展开。

- **前列腺健康论文公开问题**：一位用户无意中将其 [前列腺健康论文](https://www.perplexity.ai/page/Is-Prostate-Health-K4Ha3VDZRYicAbJl8vET9w) 公开，并寻求帮助将其撤回。另一位成员建议使用 Pages 菜单中的“取消发布页面 (Unpublish Page)”按钮。

- **大象交流页面**：一位贡献者分享了一个 [讨论大象交流的页面](https://www.perplexity.ai/page/Elephants-Call-Each-036FUcDlSNOmVbVpFubFDQ) 链接。这可能引发了关于动物行为和交流方式的对话。

- **上古卷轴 (Elder Scrolls) 页面**（重复）：几条消息包含了关于 [上古卷轴](https://www.perplexity.ai/page/The-Elder-Scrolls-rE8j7mGJTuuLZbl4F.FZnw) 页面的链接。这可能表明用户中对该游戏系列有共同兴趣。
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1251982282785292338)** (3 条消息): 

- **Custom GPT 中的“提问任何事”功能遇到困难**：一位用户成功运行了他们的 Custom GPT，但希望它能处理任何 prompts 或查询。另一位用户建议向 GPT-4o 解释问题，并提供特定的 schema 和错误详情，以解决调用 Perplexity API 时的 Action/Function Calling 问题。
- **询问 closed-beta 访问的时间表**：一位成员询问了 API 的 closed-beta 访问权限的预计响应时间。他们提到他们在 Kalshi 的项目严重依赖于访问源数据，并已准备好在获得访问权限后启动。
  

---

### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1252247794937237504)** (3 messages): 

- **频道需要 Autechre 来“摧残”耳朵**：一位成员建议频道中需要更多 Autechre，并分享了 [YouTube 视频 "Autechre - Gantz Graf (Official Music Video) 1080p HD"](https://www.youtube.com/watch?v=ev3vENli7wQ) 来实现这一目标。
- **Autechre 治愈你的灵魂**：为了平衡之前的建议，同一位成员分享了 [YouTube 视频 "Autechre - Altibzz"](https://www.youtube.com/watch?v=m3ZyEGTIsvE)，将其描述为一种治愈灵魂的方式。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=m3ZyEGTIsvE">Autechre - Altibzz</a>: 他们新专辑 &quot;Quaristice&quot; 的氛围开场曲。音乐由 Rob Brown 和 Sean Booth 创作 (c)/(p) 2007-8 Warp Records, Ltd.</li><li><a href="https://www.youtube.com/watch?v=ev3vENli7wQ">Autechre - Gantz Graf (Official Music Video) 1080p HD</a>: Autechre - Gantz Graf (官方音乐视频) HD
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1251380472131358772)** (5 messages): 

- **YouTube 视频中神经元在玩 Doom**：分享的 [YouTube 视频](https://youtu.be/c-pWliufu6U) 标题为“培养活体神经元来玩……Doom？| 第 2 部分！”，探讨了利用活体神经元玩电子游戏 **Doom** 的概念。这是生物技术与游戏领域一个有趣的交集。

- **自动化偏见与灌输**：链接到 [ResearchGate 论文](https://www.researchgate.net/publication/378191925_Automated_Bias_and_Indoctrination_at_Scale_Is_All_You_Need)，讨论了 AI 的市场驱动轨迹以及与人类偏见和大规模认知相关的新型风险。该论文批评了像 LLM 这样的“随机鹦鹉”（stochastic parrots），认为它们是符合企业偏见的操纵工具。

- **解决 mASI 中的对齐问题**：一篇引发思考的 [论文](https://www.researchgate.net/publication/372083027_The_Ethical_Basilisk_Thought_Experiment) 旨在强调 AI 决策中的伦理影响和极端责任。它引入了“伦理重心”（Ethical Center of Gravity）的概念，用于平衡伦理行为以减轻反乌托邦风险。

- **使用 vLLM 进行高效 LLM 推理**：博客文章详细介绍了 [vLLM](https://arxiv.org/abs/2309.06180)，这是一个使用 PagedAttention 来优化内存使用和吞吐量的开源推理引擎。vLLM 可以用显著更少的 GPU 运行模型，并且与 HuggingFace Transformers 相比，吞吐量最高可提升 24 倍。

- **Stable Diffusion Subreddit 抗议 Reddit API 变更**：[r/StableDiffusion](https://www.reddit.com/r/StableDiffusion/comments/1dhd7vz/the_developer_of_comfy_who_also_helped_train_some/#lightbox) subreddit 在抗议 Reddit 开放 API 政策变更后重新开放。此次抗议强调了对应用开发者、社区管理以及盲人用户可访问性影响的担忧。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/c-pWliufu6U">Growing Living Neurons to Play...Doom? | Part 2!</a>: 在下方链接使用代码 thoughtemporium 可享受 Incogni 年度计划专属 6折优惠：https://incogni.com/thoughtemporium____________________________...</li><li><a href="https://x.com/MoritzW42/status/1801418940515815676">来自 Moritz Wallawitsch 🗽 (@MoritzW42) 的推文</a>: vLLM 和 PagedAttention 简介（最初发表于 @runpod_io 的博客） 1/6 vLLM 是一个开源的 LLM 推理和服务引擎（由 @woosuk_k @KaichaoYou @zhuohan1 开发...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1dhd7vz/the_developer_of_comfy_who_also_helped_train_some/#lightbox">Reddit - 深入探索一切</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1251252712947843153)** (124 条消息🔥🔥): 

- **Stable Diffusion 3 及其用法**：成员们注意到 **anime 允许“撒谎”**，但仅限于非人类角色。分享了访问 [Stable Diffusion 3](https://huggingface.co/spaces/stabilityai/stable-diffusion-3-medium) 的链接。
- **NVIDIA 的合成数据模型**：讨论了 **Nemotron-4-340B-Instruct**，这是一个用于合成数据生成的 LLM，针对英文对话进行了优化，并支持 4,096 tokens 的上下文长度。可在 [Hugging Face](https://huggingface.co/nvidia/Nemotron-4-340B-Instruct) 上获取，并探讨了其用途以及对 NVIDIA 客户关系的竞争影响。
- **实时推理与 ComfyUI**：一位成员建议使用 **ComfyUI 搭配 TensorRT SD-Turbo** 进行近乎实时的推理，尤其是与网络摄像头馈送结合进行图像处理时非常有趣。
- **OpenAI 转向营利性实体**：[Sam Altman](https://x.com/aaronpholmes/status/1801785687030829240?s=46) 已告知股东，OpenAI 可能会转型为营利性实体，类似于 Anthropic 和 xAI 等竞争对手。
- **模型合并与 MoE 辩论**：关于 **Mixture of Experts (MoE)** 模型和合并策略的实用性与性能的深入讨论，对合并方法与全面 fine-tuning 相比的有效性持保留意见。分享了 [llama.cpp](https://github.com/ggerganov/llama.cpp/pull/6453) 相关 PR 以及 [Hugging Face](https://huggingface.co/Kquant03/CognitiveFusion-4x7B-bf16-MoE) 上 MoE 模型的链接。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://situational-awareness.ai/">引言 - SITUATIONAL AWARENESS: The Decade Ahead</a>: Leopold Aschenbrenner，2024 年 6 月。你可以在旧金山率先看到未来。在过去的一年里，业内的谈论焦点已从 100 亿美元的计算集群转向 1000 亿美元，再到万亿级...</li><li><a href="https://arxiv.org/abs/2406.07394">通过 LLaMa-3 8B 的蒙特卡洛树自我细化获取 GPT-4 级别的奥数解法</a>: 本文介绍了 MCT Self-Refine (MCTSr) 算法，这是一种 LLM 与蒙特卡洛树搜索 (MCTS) 的创新结合，旨在增强在复杂数学任务中的表现...</li><li><a href="https://huggingface.co/spaces/stabilityai/stable-diffusion-3-medium">Stable Diffusion 3 Medium - stabilityai 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=xm1B3Y3ypoE">智能爆炸临近了吗？现实检查。</a>: 在 Brilliant 上了解更多关于神经网络和 LLM 的知识！前 30 天免费，使用我们的链接可享受年度高级订阅 20% 的折扣...</li><li><a href="https://huggingface.co/nvidia/Nemotron-4-340B-Instruct">nvidia/Nemotron-4-340B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/Kquant03/CognitiveFusion-4x7B-bf16-MoE">Kquant03/CognitiveFusion-4x7B-bf16-MoE · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=c-pWliufu6U">培养活体神经元来玩...Doom？| 第 2 部分！</a>: 在下方链接使用代码 thoughtemporium 可获得 Incogni 年度计划专属 60% 折扣：https://incogni.com/thoughtemporium____________________________...</li><li><a href="https://x.com/ryanels4/status/1801898008755241251?s=46">Ryan Els (@RyanEls4) 的推文</a>: AI 揭晓 😲</li><li><a href="https://tenor.com/view/godfather-massacre-sad-gif-16810633">教父大屠杀 GIF - 教父大屠杀悲伤 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/aaronpholmes/status/1801785687030829240?s=46">aaron holmes (@aaronpholmes) 的推文</a>: 最新消息：Sam Altman 已告知股东，OpenAI 正在考虑成为一家营利性公司，不再受非营利董事会控制 https://www.theinformation.com/articles/openai-ce...</li><li><a href="https://x.com/alexalbert__/status/1801668464920379648?s=46">Alex Albert (@alexalbert__) 的推文</a>: 喜欢 Golden Gate Claude？🌉 我们将开放实验性 Steering API 的有限访问权限——允许你引导 Claude 的一部分内部特征。在此注册：https://forms.gle/T8fDp...</li><li><a href="https://huggingface.co/datasets/BAAI/Infinity-Instruct">BAAI/Infinity-Instruct · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6453">通过 DisOOM 的 mergekit-qwen2 为自定义 Qwen2moe 架构添加支持 · Pull Request #6453 · ggerganov/llama.cpp</a>: 声明：这与 Qwen/Qwen1.5-MoE-A2.7B 中的细粒度 MoE 架构无关。它更类似于传统的 MoE，除了其专家（experts）源自 qwen2 (qwen1.5...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1251398975899045960)** (22 messages🔥): 

- **设置 Llama3 8B 具有挑战性**：*Plasmator* 询问了关于针对特定风格训练 Llama3 8B 以及在 M2 Ultra 上部署快速且兼容 OAI 接口的建议。*Teknium* 推荐使用 **unsloth qlora**、**Axolotl** 和 **Llamafactory** 进行训练，并使用 **lmstudio** 或 **Ollama** 在 Mac 上进行接口部署。

- **RAG 方法咨询**：*Rbccapitalmarkets* 询问最近一篇论文中的 set-based prompting 技术是否可以与 RAG (Retrieval-Augmented Generation) 结合使用。他们分享了[论文链接](https://arxiv.org/abs/2406.06581)以提供更多背景信息。

- **CMU 的 PEFT 方法讨论**：*420gunna* 提到了一门 CMU 高级 NLP 课程，教授在课上推介了他自己关于两种新 PEFT (Parameter Efficient Fine-Tuning) 方法的论文。他们为感兴趣的人分享了讲座的 [YouTube 链接](https://youtu.be/KLJ3EEo8aPU?si=D155UK9mgoWmqOn5&t=2942)。

- **Nvidia 的 Nemotron 模型**：*Avinierdc* 询问了关于 Nvidia 新推出的 **Nemotron 模型**的看法，引发了简短的讨论。*Teknium* 表达了积极的看法，并提到曾在 LMSYS chatbot arena 上尝试过一次。

- **LLM 领域的 ComfyUI 替代方案**：*Csshsh* 正在寻找一种类似于 LLM 版 ComfyUI 的工具，允许在 API 层之下进行趣味性交互。*Orabazes* 建议使用 **ComfyUI** 就能完成很多工作，并推荐查看用于本地运行模型的 **AnyNode 自定义节点**。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2406.06581">Set-Based Prompting: Provably Solving the Language Model Order Dependency Problem</a>：开发能够通过自回归创建长且连贯文本输出的生成式语言模型，导致了用途的激增以及随之而来的广泛研究分析...</li><li><a href="https://youtu.be/KLJ3EEo8aPU?si=D155UK9mgoWmqOn5&t=2942">CMU Advanced NLP 2024 (8): Fine-tuning and Instruction Tuning</a>：这是 Graham Neubig 为 CMU CS 11-711 高级 NLP（2024 春季）提供的讲座，内容涵盖：* 多任务处理 * 微调与指令微调 * 参数高效...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1251282963715461132)** (29 messages🔥): 

- **正在考虑多人协作功能**：一位成员询问了创建大厅以便与 AI 进行协作创作的可能性。另一位成员确认了对此的兴趣，表示：*“是的，我们非常喜欢这个想法——任何形式的多人/PVP/合作模式”*。
  
- **Worldclient 与 WebSim 互不相连**：关于 WebSim 上的 Opus 与 WorldSim 之间的联系存在困惑。官方澄清道：*“worldclient 与 websim 没有任何联系”*。

- **WorldSim AI 经历了更多审查**：一位成员指出：*“world-sim ai 受到了一定程度的审查”*。另一位解释说，审查的增加可能是由于模型提供商 Anthropic 采取了更严格的措施。

- **AI 回复的续写功能正在开发中**：成员们讨论了一个 AI 回复突然中断的 Bug。其中一人强调正在努力修复此问题：*“是的，我正在开发一个允许续写的功能”*。

- **Claude 3 的视觉支持与成本考量**：成员们讨论了在 WorldSim 中集成视觉支持，并指出 Claude 3 已经具备此功能。他们还辩论了成本问题，建议使用 GPT4o 处理视觉任务并将信息传递给 Claude 以优化使用成本。
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1251271189431844895)** (40 messages🔥): 

- **Mojo 函数手册引发辩论**：围绕 Mojo 手册中对 `def` 和 `fn` 函数的解释展开了讨论，特别是 `def` 函数是允许还是不需要类型声明。一位参与者提出了七种替代措辞来澄清语言描述，展示了英语理解中的细微差别。

- **Mojo 类型机制评析**：对话转向了 `def` 函数中类型声明的宽松性。共识是，虽然 `def` 函数不强制要求类型声明，但它们允许这样做，这与要求显式类型声明的 `fn` 函数形成对比。

- **Mojo 社区活动公告**：发布了 Mojo 社区会议的公告，内容包括 Helehex 关于 constraints 的演讲和 Valentin 关于 Lightbug 的演讲，随后是 Jack 关于 Python interop 的讨论。提供了加入会议的链接。[加入会议](https://modul.ar/community-meeting-zoom)

- **基准测试对比分享**：一位用户分享了对比 Python FastAPI、Mojo Lightbug 和 Rust Actix 的单线程基准测试结果。结果显示 Mojo Lightbug 的性能优于 Python FastAPI，但落后于 Rust Actix。

- **关于函数着色 (Function Coloring) 的担忧讨论**：社区会议结束后，一场关于消除函数着色潜在运行时成本的讨论引发了关于 stackful 与 stackless coroutines 的对话。辩论强调了运行时成本与语言复杂度之间的权衡。[关于协程讨论的链接](https://langdev.stackexchange.com/questions/697/what-are-the-benefits-of-stackful-vs-stackless-coroutines)
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/functions">Functions | Modular Docs</a>：Mojo `fn` 和 `def` 函数介绍。</li><li><a href="https://langdev.stackexchange.com/questions/697/what-are-the-benefits-of-stackful-vs-stackless-coroutines)">What are the benefits of stackful vs. stackless coroutines?</a>：现今许多使用 async 的语言都是通过 stackless coroutines 实现的。也就是说，在像 Python 或 Javascript 这样的语言中，当你 await 一个表达式时，该 await 会返回给调用者。</li><li><a href="https://github.com/go-vgo/robotgo/issues/662">Inconsistent key detection bug · Issue #662 · go-vgo/robotgo</a>：Robotgo 版本（或 commit 引用）：Go 版本：1.17 Gcc 版本：操作系统及位数：windows10 64bit 提供示例代码：robotgo.EventHook(hook.KeyDown, []string{&quot;command&quot;, &quot;m&quot;...</li><li><a href="https://modul.ar/community-meeting-zoom">Join our Cloud HD Video Meeting</a>：Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，用于移动、桌面和会议室系统的视频和音频会议、聊天和网络研讨会。Zoom ...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1251298302218211371)** (2 messages): 

- **Modular 推送新动态**：[Modular 分享了一条推文](https://twitter.com/Modular/status/1801734712026911084) 与社区互动，让关注者了解他们的最新活动和公告。
- **Modular 通过 Twitter 发布另一项公告**：[Modular 发布了另一条推文](https://twitter.com/Modular/status/1802781075841974414)，让社区了解他们的持续进展和即将举行的活动。
  

---


### **Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1252338615732932691)** (1 messages): 

- **Mojo 24.4 版本发布**：Mojo 发布了 24.4 版本，该版本在核心语言和标准库方面进行了多项重大增强。鼓励读者阅读完整的博客文章 [此处](https://www.modular.com/blog/whats-new-in-mojo-24-4-improved-collections-new-traits-os-module-features-and-core-language-enhancements) 以获取详细见解和代码示例。

**提及的链接**：<a href="https://www.modular.com/blog/whats-new-in-mojo-24-4-improved-collections-new-traits-os-module-features-and-core-language-enhancements">Modular: What’s New in Mojo 24.4? Improved collections, new traits, os module features and core language enhancements</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新帖子：What’s New in Mojo 24.4? Improved collections, new traits, os module features and core language enhanc...

  

---

### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1251921353817329676)** (2 条消息): 

- **悬赏 1,000,000 美元寻找真正的 AGI 解决方案！**：一位用户分享了一段 [Francois Chollet 的 YouTube 视频](https://www.youtube.com/watch?v=UakqL6Pj9xo)，视频中讨论了他为什么认为 LLM 不会通向 AGI，并设立了 1,000,000 美元的 ARC-AGI Prize 来寻找真正的解决方案。另一位用户表示怀疑，评论说奖金金额感觉像是“开价太低”。

**提到的链接**：<a href="https://www.youtube.com/watch?v=UakqL6Pj9xo">Francois Chollet - LLM 不会通向 AGI - 悬赏 1,000,000 美元寻找真正的解决方案</a>：这是我与 Francois Chollet 和 Mike Knoop 关于他们今天发起的 100 万美元 ARC-AGI Prize 的对话。我进行了一系列苏格拉底式的盘问...

  

---

### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1251292886738407474)** (107 messages🔥🔥): 

- **在 Mojo 中定义 2D Numpy 数组**：用户讨论了 Mojo 在向 Python 传递嵌套列表时的局限性，并分享了使用 `ast` 模块和 `Python.import_module` 的解决方法。例如，一位用户建议使用 `ndarray` 函数，将嵌套列表的字符串表示形式转换为 Numpy 数组，然后作为 Python 对象返回。
- **`DTypePointer` 与 `Pointer[SomeDType]` 的区别**：用户强调 `DTypePointer` 在 SIMD 操作中更具优势，因为它支持高效的 `simd_load` 指令。这对于想要了解使用每种类型对性能影响的用户特别有帮助。
- **VSCode 与 Mojo 的集成**：一位成员询问如何在 VSCode 中为 Mojo 包含目录，另一位成员提供了通过 `settings.json` 实现的方法。通过添加 `"mojo.lsp.includeDirs": [ "/Users/your-name/your-mojo-files" ]`，可以帮助 VSCode 分析 Mojo 包。
- **类型转换与上下文行为中的 Bug**：有用户报告了在使用 `int()` 或 `UInt32()` 转换无符号整数时的 Bug，在运行脚本和使用 REPL 时表现出不同的行为。已创建一个 GitHub [issue](https://github.com/modularml/mojo/issues/3065) 来跟踪这一不一致性。
- **使用 Var 与 Alias 进行 CRC32 表计算**：一项详细讨论揭示了使用 `alias` 而非 `var` 初始化 CRC32 表时的问题，由于转换行为导致结果不同。最小示例显示发生了意外的带符号溢出，从而引发了对 `alias` 特定行为的调查。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/cli/repl">mojo repl | Modular Docs</a>：启动 Mojo REPL。</li><li><a href="https://github.com/fnands/fnands.com/blob/main/blog/2024/mojo-crc-calc/super_minimal_bug.mojo">fnands.com/blog/2024/mojo-crc-calc/super_minimal_bug.mojo at main · fnands/fnands.com</a>：我的个人博客。通过在 GitHub 上创建账号为 fnands/fnands.com 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/1139">[BUG]: Issue on creating  numpy N-dimensional array · Issue #1139 · modularml/mojo</a>：Bug 描述：无法创建多维 numpy 数组，创建一维 numpy 数组运行正常。复现步骤：包含相关的代码片段或指向不起作用代码的链接...</li><li><a href="https://github.com/modularml/mojo/issues/3065#issuecomment-2173609155">[BUG] Unsigned integer casting overflowing as if signed when using `int()` or `UInt32()` · Issue #3065 · modularml/mojo</a>：Bug 描述：在 Discord 进行一些讨论后将其迁移至此。似乎转换为无符号整数实际上只是转换为有符号整数，但在不同环境下有不同的行为...</li><li><a href="https://github.com/modularml/mojo/issues/3065">[BUG] Unsigned integer casting overflowing as if signed when using `int()` or `UInt32()` · Issue #3065 · modularml/mojo</a>：Bug 描述：在 Discord 进行一些讨论后将其迁移至此。似乎转换为无符号整数实际上只是转换为有符号整数，但在不同环境下有不同的行为...</li><li><a href="https://github.com/fnands/fnands.com/blob/main/blog/2024/mojo-crc-calc/crc32_alias.mojo">fnands.com/blog/2024/mojo-crc-calc/crc32_alias.mojo at main · fnands/fnands.com</a>：我的个人博客。通过在 GitHub 上创建账号为 fnands/fnands.com 的开发做出贡献。</li><li><a href="https://github.com/fnands/fnands.com/blob/main/blog/2024/mojo-crc-calc/minimal_bug.mojo">fnands.com/blog/2024/mojo-crc-calc/minimal_bug.mojo at main · fnands/fnands.com</a>：我的个人博客。通过在 GitHub 上创建账号为 fnands/fnands.com 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues">Issues · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/fnands/fnands.com/blob/main/blog/2024/mojo-crc-calc/crc.mojo">fnands.com/blog/2024/mojo-crc-calc/crc.mojo at main · fnands/fnands.com</a>：我的个人博客。通过在 GitHub 上创建账号为 fnands/fnands.com 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1251464270907047937)** (3 messages): 

- **TPU 使用方式已明确**：一位成员解释说，使用 **TPUs** 的唯一方法是通过 *PjRT API 调用 XLA*。他们提供了 [PjRT API 文档链接](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h) 以及 TPU 插件 [libtpu.so](https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/libtpu/2023-09-12/libtpu.so)。
- **呼吁原生 TPU 支持**：另一位成员建议编写对 **TPUs** 的原生支持，类似于 **Modular** 处理 GPUs 的方式。第一位成员回应称，目前没有比 XLA 更底层的 TPU 公开 API。

**提到的链接**：<a href="https://github.com/openxla/xla/blob/main">GitHub - openxla/xla: A machine learning compiler for GPUs, CPUs, and ML accelerators</a>：一个用于 GPUs、CPUs 和 ML 加速器的机器学习编译器 - openxla/xla

  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1251363523779625001)** (9 messages🔥): 

- **新版 Nightly Mojo 编译器发布**：新的 Nightly Mojo 编译器版本 `2024.6.1505` 已发布，用户可以通过 `modular update nightly/mojo` 进行更新。更多详情请参阅 [raw diff](https://github.com/modularml/mojo/compare/1130fdb81d763066d8e5bcb2226fe270981d3b0a...0508c6ee7fc8e1e7b42eda057ce9175a32c970cc) 和 [当前 changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)。
- **编译器版本 `2024.6.1605` 发布**：Mojo 编译器的另一个 Nightly 更新版本 `2024.6.1605` 已发布。用户应进行更新，并通过 [raw diff](https://github.com/modularml/mojo/compare/0508c6ee7fc8e1e7b42eda057ce9175a32c970cc...4df58b15f6ff4c613ecd4de62bde206a248d4652) 和 [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 查看更改。
- **最新 Nightly 版本 `2024.6.1705`**：Nightly Mojo 编译器的最新更新版本 `2024.6.1705` 现已可用。更新详情可通过 [raw diff](https://github.com/modularml/mojo/compare/4df58b15f6ff4c613ecd4de62bde206a248d4652...87266e71c6dd29eca48511f2c8de492783be783a) 和 [当前 changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 查看。
- **请求内置 MLIR Dialects 文档**：一位用户询问是否有内置 MLIR dialects 的外部文档。另一位成员确认目前没有此类文档。
- **REPL 改进的功能请求**：有人询问表达式是否可以像 Python 一样在 REPL 中直接输出值。回复建议在 GitHub 上提交该增强功能的功能请求。
  

---



### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1251345207526690927)** (1 messages): 

- **可解释性团队复现 OpenAI 的发现**：EleutherAI 可解释性团队使用开源 LLMs 成功复现了 OpenAI 的 “weak-to-strong” 泛化结果。他们在 21 个 NLP 数据集上观察到了这些结果，并尝试了几种改进泛化的修改，但发现 *“vanilla weak-to-strong 训练可能已经接近于诱导出学生模型‘所知’的一切”*。

- **泛化改进的负面结果**：该团队尝试了各种修改，如 strong-to-strong 训练、修改后的损失函数以及几个基于探针（probe-based）的实验，结果 *“通常是负面的”*。在这些尝试中，只有 log-confidence 辅助损失显示出持续改进泛化的潜在迹象。

- **发布详细调查结果**：关于 Qwen1.5 0.5B 和 Llama 3 8B 等开源模型中 weak-to-strong 泛化的详细调查结果和结论，可以在他们最新的 [博客文章](https://blog.eleuther.ai/weak-to-strong/) 中找到。

**提到的链接**：<a href="https://blog.eleuther.ai/weak-to-strong/">Experiments in Weak-to-Strong Generalization</a>：记录近期项目的实验结果

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1251302336500990173)** (51 messages🔥): 

- **AISI 增加了新职位并提供搬迁协助**：AISI 在其[招聘页面](https://aisi.gov.uk/careers)上发布了多个新职位空缺，并提到可以为愿意搬迁到英国的候选人提供签证协助。这引起了非英国居民成员的兴趣，尽管有地点限制，他们仍在考虑这一机会。
  
- **关于 CommonCrawl 处理的讨论**：成员们交流了处理 CommonCrawl 快照的技巧，重点介绍了 [ccget](https://github.com/allenai/ccget) 和 [resiliparse](https://resiliparse.chatnoir.eu/en/latest/man/parse/html.html) 等工具。挑战包括限流（throttling）以及高效处理大型数据集的性能优化。

- **对可复现图像生成模型的关注**：用户讨论了基于公开授权数据训练的图像生成模型，特别提到了 [Hugging Face](https://huggingface.co/common-canvas) 上的 CommonCanvas 模型及相关的 [arXiv 论文](https://arxiv.org/abs/2310.16825)。虽然有人认为目前这些模型的效果较差，但他们建议将其用于创建纹理生成等应用。

- **澄清 Git 与 GitHub 的混淆**：成员们澄清了 Git 和 GitHub 之间的区别，强调 Git 是一个源代码管理工具，而 GitHub 是一个仓库托管服务。对话中包含了一个[视频链接](https://www.theserverside.com/video/Git-vs-GitHub-What-is-the-difference-between-them)以进一步解释这些概念。

- **新成员介绍**：Piyush Ranjan Maharana 和 Tomer 等新成员分享了他们的背景，包括在计算物理、自动驾驶汽车以及通过 LLM 进行材料发现方面的工作。他们表达了学习和为社区做贡献的热情。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/common-canvas">common-canvas (CommonCanvas)</a>: 未找到描述</li><li><a href="https://github.com/allenai/ccget">GitHub - allenai/ccget: Tools for an internal archive of some Common Crawl files</a>: 用于某些 Common Crawl 文件内部归档的工具 - allenai/ccget</li><li><a href="https://aisi.gov.uk/careers">Careers | AISI</a>: 查看 AISI 的职业机会。AI 安全研究所（AI Safety Institute）是科学、创新和技术部的一个局，旨在促进严谨的研究以实现先进的 AI 治理...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1251267140460351631)** (61 messages🔥🔥): 

- **探索用于视觉-语言学习的 RWKV-CLIP**：一篇论文讨论了 **RWKV-CLIP** 的引入，这是一种结合了 Transformer 并行训练与 RNN 高效推理的视觉-语言表示学习模型。该方法旨在通过利用 **LLM** 合成和精炼网络文本及合成标题，来提高大规模图文数据的质量。
  
- **关于扩散模型幻觉的担忧**：另一篇论文探讨了扩散模型中的“幻觉”现象，识别出一种被称为模式插值（mode interpolation）的失败模式。研究表明，扩散模型会在数据模式之间进行插值，从而产生原始训练分布中不存在的伪影。

- **关于预取流式数据集的讨论**：一些技术讨论涉及使用 `keep_in_memory=True` 处理 **streaming** 数据集以实现高效的数据获取。成员们分享了关于最近引入的流检查点保存（checkpointing）和恢复功能的见解，这增强了大型数据集的可用性。

- **Laprop 优化器的有效性**：成员们辩论了 **Laprop** 优化器的有效性，结果褒贬不一，显示其性能与 **AdamW** 相比平平或更差。虽然参数调整带来了一些改进，但 Laprop 的整体表现仍然不尽如人意。

- **窃取商业嵌入模型**：一篇论文强调了一种通过使用从 API 获取的文本-嵌入对训练本地模型来“窃取”商业嵌入模型的方法。该方法表明，可以低成本地实现有效的复制，这引发了对商业模型安全性的担忧。
<div class="linksMentioned">

<strong>Links mentioned</strong>:



<ul>
<li>
<a href="https://arxiv.org/abs/2406.09355">Can't Hide Behind the API: Stealing Black-Box Commercial Embedding Models</a>：无法隐藏在 API 背后：窃取黑盒商业 Embedding 模型。从自然语言文本生成表示向量的 Embedding 模型被广泛使用，反映了巨大的投资并具有显著的商业价值。OpenAI 等公司...</li><li><a href="https://arxiv.org/abs/2406.05587">Creativity Has Left the Chat: The Price of Debiasing Language Models</a>：创意已离开聊天：去偏见语言模型的代价。大语言模型 (LLMs) 彻底改变了自然语言处理，但可能表现出偏见并可能生成有害内容。虽然像人类反馈强化学习 (RLHF) 这样的对齐技术...</li><li><a href="https://arxiv.org/abs/2405.11597">Language Reconstruction with Brain Predictive Coding from fMRI Data</a>：利用 fMRI 数据的脑预测编码进行语言重建。最近的许多研究表明，语音感知可以从大脑信号中解码，并随后重建为连续语言。然而，目前缺乏神经学基础...</li><li><a href="https://arxiv.org/abs/2406.09358v1">Understanding Hallucinations in Diffusion Models through Mode Interpolation</a>：通过模式插值理解 Diffusion Models 中的幻觉。通俗地说，基于扩散过程的图像生成模型经常被认为会出现“幻觉”，即在训练数据中从未出现过的样本。但是这些...</li><li><a href="https://neuralblog.github.io/logit-prisms/">Logit Prisms: Decomposing Transformer Outputs for Mechanistic Interpretability</a>：Logit Prisms: 为机械可解释性分解 Transformer 输出。未找到描述</li><li><a href="https://arxiv.org/abs/2405.19296">Neural Isometries: Taming Transformations for Equivariant ML</a>：神经等距：为等变机器学习驯服变换。现实世界的几何和 3D 视觉任务充满了具有挑战性的对称性，这些对称性难以用解析表达式表示。在本文中，我们介绍了 Neural Isometries，这是一个 Autoencoder 框架...</li><li><a href="https://arxiv.org/abs/2406.06973">RWKV-CLIP: A Robust Vision-Language Representation Learner</a>：RWKV-CLIP: 一个鲁棒的视觉-语言表示学习器。对比语言-图像预训练 (CLIP) 通过使用从网站获取的图像-文本对扩展数据集，显著提高了各种视觉-语言任务的性能。本文...</li><li><a href="https://huggingface.co/datasets/codeparrot/github-code-clean">codeparrot/github-code-clean · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://fixupx.com/ArmenAgha/status/1780149168692158658">Tweet from Armen Aghajanyan (@ArmenAgha)</a>：最终更新：对 Sophia 进行又一个量级的测试。我们讨论的是 B 级规模的模型，T 级规模的 Token。Sophia 再次胜出。至少对我来说，这是 Sophia ... 的明确证据。</li><li><a href="https://research.google/pubs/learned-optimizers-that-scale-and-generalize/">Learned Optimizers that Scale and Generalize</a>：可扩展且泛化的学习优化器。未找到描述</li><li><a href="https://arxiv.org/abs/2209.11208">A Closer Look at Learned Optimization: Stability, Robustness, and Inductive Biases</a>：深入探讨学习优化：稳定性、鲁棒性和归纳偏置。学习优化器——被训练作为优化器运行的神经网络——具有显著加速机器学习模型训练的潜力。然而，即使在元训练跨越...</li><li><a href="https://arxiv.org/abs/2406.07496">TextGrad: Automatic "Differentiation" via Text</a>：TextGrad: 通过文本进行自动“微分”。AI 正在经历范式转移，通过编排多个大语言模型 (LLMs) 和其他复杂组件的系统实现了突破。因此，开发原则性和自动化的...</li><li><a href="https://github.com/zou-group/textgrad">GitHub - zou-group/textgrad: Automatic ''Differentiation'' via Text -- using large language models to backpropagate textual gradients.</a>：通过文本进行自动“微分”——使用大语言模型反向传播文本梯度。- zou-group/textgrad</li><li><a href="https://github.com/CFGpp-diffusion/CFGpp">GitHub - CFGpp-diffusion/CFGpp: Official repository for "CFG++: manifold-constrained classifier free guidance for diffusion models"</a>：“CFG++: 扩散模型的流形约束无分类器指导”官方仓库。- CFGpp-diffusion/CFGpp</li><li><a href="https://arxiv.org/abs/2406.08070">CFG++: Manifold-constrained Classifier Free Guidance for Diffusion Models</a>：CFG++: 扩散模型的流形约束无分类器指导。无分类器指导 (CFG) 是现代 Diffusion Models 中用于文本引导生成的基石工具。虽然有效，但 CFG 有显著的缺点。例如，带有 CFG 的 DDIM 缺乏可逆性...</li><li><a href="https://github.com/apple/ml-agm">GitHub - apple/ml-agm</a>：通过在 GitHub 上创建账户来为 apple/ml-agm 的开发做出贡献。</li><li><a href="https://openreview.net/forum?id=tUtGjQEDd4">Generative Modeling with Phase Stochastic Bridge</a>：使用相位随机桥的生成建模。Diffusion Models (DMs) 代表了连续输入的尖端生成模型。DMs 通过在输入空间（即位置空间）构建随机微分方程 (SDE) 来工作...
</li>
</ul>

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1251623429786964060)** (18 条消息🔥): 

- **基于 Hypernetwork 的论文批评**：一位成员认为一篇提出 **linear hypernetwork attention** 的论文“毫无用处”，声称其中包含一个关键错误，导致其效率甚至不如 full attention。他们指出，该论文为 attention 机制表现得像 hypernetworks 提供了一些推理。

- **Hypernetworks 与 Hopfield Nets 的辩论**：成员们讨论了 hypernetworks 是否实际上就是 Hopfield nets，其中一位成员指出，虽然两者在输入依赖的权重生成等方面存在高层相似性，但 Hopfield networks 本质上是循环的（recurrent）。这引发了关于 Hopfield networks 历史意义和演变的讨论。

- **Hopfield Networks 的历史背景**：成员们回忆了 Hopfield networks 过去在联结主义（connectionism）中的重要地位，以及它们对 Transformer 等当前模型的影响。他们指出，现代模型使用反向传播（backpropagation）和多层网络（multi-layer networks）以获得卓越性能，但来自 Hopfield nets 的吸引子（attractors）和动力学（dynamics）概念仍然影响着当代的 neural network architecture。

- **动态评估与在线适配**：一位成员分享了一篇关于语言模型 **dynamic evaluation** 的论文，强调了其在测试时适应分布偏移（distributional shifts）的效用。该方法被描述为将参数转化为随时间变化的状态，非常类似于神经科学中的记忆，并值得进行潜在的 Jones 风格 scaling law 评估。

- **Jones 风格 Scaling Law 引用**：针对动态评估的讨论，一位成员引用了 Andy L. Jones 的论文《Scaling scaling laws with board games》，该论文建议在训练计算量（training compute）和推理计算量（inference compute）之间进行权衡。这一引用强调了在自适应模型语境下考虑高效 scaling laws 的相关性。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2403.01518#deepmind">Revisiting Dynamic Evaluation: Online Adaptation for Large Language Models</a>：我们考虑了在测试时在线微调语言模型参数的问题，也称为动态评估（dynamic evaluation）。虽然众所周知这种方法可以提高整体表现...</li><li><a href="https://arxiv.org/abs/2405.08707">Beyond Scaling Laws: Understanding Transformer Performance with Associative Memory</a>：增加 Transformer 模型的大小并不总是能提高性能。这种现象无法用经验性的 scaling laws 来解释。此外，改进的泛化能力...</li><li><a href="https://x.com/ibab/status/1669579636563656705)?">来自 Igor Babuschkin (@ibab) 的推文</a>：我一直在回顾 @andy_l_jones 的这篇优秀论文：“Scaling scaling laws with board games”。它展示了 MCTS 的训练计算量和推理计算量如何相互权衡。10倍以上的...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1251612877870334117)** (11 条消息🔥): 

- **数学博士探索 Sparse Autoencoders**: 一位新晋数学博士表达了对涉及 Sparse Autoencoders (SAEs) 的可解释性研究的兴趣。他们被引导至[一篇博客文章](https://www.lesswrong.com/posts/a5wwqza2cY3W7L9cj/sparse-autoencoders-find-composed-features-in-small-toy)，该文章发现在玩具模型中，SAEs 可能会恢复出组合特征而非 ground truth 特征。
  
- **关于稀疏编码与字典学习的讨论**: 成员们分享了相关论文并讨论了与稀疏编码（sparse coding）和字典学习（dictionary learning）相关的主题，包括一篇关于 Wasserstein 空间中字典学习的论文（[链接](https://arxiv.org/pdf/2405.00837)）以及另一篇关于自然视频中解耦（disentanglement）的论文（[链接](https://arxiv.org/abs/2007.10930)）。

- **特征字典评估框架**: 介绍了一篇论文，该论文提出了一个在特定任务中使用监督字典来评估特征字典的框架，并重点展示了其在 GPT-2 Small 上的间接宾语识别（indirect object identification）任务中的应用（[论文链接](https://arxiv.org/abs/2405.08366)）。

- **线性可辨识性工作链接**: 针对激活空间中是否存在真正线性特征的设置，相关咨询引导至了对线性探针（linear probes）和 ICA 文献的调查，相关论文见[此处](https://arxiv.org/abs/2311.03658)。

- **Logit Prisms 工具发布**: 宣布了一项扩展 logit lens 方法的新工作，名为 "logit prisms"，它将 logit 输出分解为 residual stream、attention layers 和 MLP 层的组件。该工具被用于研究 gemma-2b 模型，揭示了数字 0-9 在 2D 空间中以心形编码（[全文](https://neuralblog.github.io/logit-prisms)）。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://neuralblog.github.io/logit-prisms">Logit Prisms: Decomposing Transformer Outputs for Mechanistic Interpretability</a>: 暂无描述</li><li><a href="https://arxiv.org/abs/2007.10930">Towards Nonlinear Disentanglement in Natural Data with Temporal Sparse Coding</a>: 我们构建了一个无监督学习模型，实现了自然视频中潜在变化因素的非线性解耦。之前的工作表明，表示可以被解耦...</li><li><a href="https://arxiv.org/abs/2405.08366">Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control</a>: 将模型激活解耦为有意义的特征是可解释性研究的核心问题。然而，在现实场景中缺乏这些特征的 ground-truth，使得验证最近的工作变得困难...</li><li><a href="https://www.lesswrong.com/posts/a5wwqza2cY3W7L9cj/sparse-autoencoders-find-composed-features-in-small-toy">Sparse autoencoders find composed features in small toy models  — LessWrong</a>: 摘要 * 背景：Sparse Autoencoders (SAEs) 揭示了语言模型激活空间中可解释的特征。它们实现了稀疏、可解释的...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1251827262609952830)** (4 条消息): 

- **呼吁分享评估结果**: 一位成员指出 Hugging Face 正在使用过时的 harness 版本，并观察到当前结果存在显著差异。他们询问是否有一个平台可以让人们发布自己的评估结果，包括运行时参数和版本信息以供验证。
- **闭源模型的独立验证请求**: 同一位成员还询问是否有地方可以发布各种闭源模型的独立验证结果。这表明需要一个共享且值得信赖的评估论坛。
- **WANDB 的多 GPU 评估问题**: 另一位成员报告了在执行多 GPU 评估时的一个问题，导致在 WANDB 中创建了两个独立的项目而不是一个。他们分享了命令行设置，并就使用 `--num_processes=2` 标志进行数据并行评估是否合适寻求建议。
  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1251841463902273626)** (3 条消息): 

- **代码发布查询引导至 GitHub Issues**: 一位成员询问了特定代码的发布日期。另一位成员将该查询重定向至 **RWKV-CLIP** 项目的 [GitHub Issues 页面](https://github.com/deepglint/RWKV-CLIP/issues)。

**提及的链接**: <a href="https://github.com/deepglint/RWKV-CLIP/issues">Issues · deepglint/RWKV-CLIP</a>: &quot;RWKV-CLIP: A Robust Vision-Language Representation Learner&quot; 的官方代码 - Issues · deepglint/RWKV-CLIP

  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1251313549448577045)** (35 条消息🔥): 

- **Apple 在 WWDC 上的 AI 策略引发关注**：一位社区成员分享了一篇[详细介绍 Apple 新 AI 策略](https://blog.trailofbits.com/2024/06/14/understanding-apples-on-device-and-server-foundations-model-release/)的博文，强调了 Apple 对 NVIDIA 硬件和 CUDA API 的回避。文中讨论了 Apple AXLearn 的使用，该框架运行在 TPU 和 Apple Silicon 上。

- **深入探讨 Embedding 资源**：分享了一系列关于 Embedding 的宝贵资源，包括 [GitHub 上的精选列表](https://github.com/eifuentes/awesome-embeddings)和 [vickiboykis.com](https://vickiboykis.com/what_are_embeddings/) 上的博文。成员们讨论了理解潜空间（latent spaces）的重要性以及 Embedding 是如何产生的。

- **征求拒绝分类器（refusal classifier）模型**：一位成员表示对现成的拒绝分类器模型感兴趣，可能使用 T5/BERT 处理多语言数据。他们表示需要大约 1K 个样本进行训练，并就此话题寻求建议。

- **针对特定叙述风格微调 TinyLlama**：一位成员记录了他们微调 TinyLlama 以生成大卫·爱登堡（David Attenborough）风格叙述的经验，并分享了他们的[博文](https://gabrielchua.me/posts/finetuning-tinyllama-axolotl-beginner/)。他们在项目中使用 Axolotl 和 Jarvis Labs 等工具，学习并分享了详细的步骤和见解。

- **在 Jarvis Labs 上加载模型配置的问题**：一位用户在 Jarvis 上尝试微调 Mistral 时遇到错误，在切换到 v0.3 版本并更改其 Token 权限后问题得到解决。他们指出这可能还需要网络稳定性，并感谢他人的协助。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co'">未找到标题</a>: 未找到描述</li><li><a href="https://youtube.com/@hamelhusain7140?si=g_QdOF2ns0NKoUNK">Hamel Husain</a>: 未找到描述</li><li><a href="https://vickiboykis.com/what_are_embeddings/">什么是 Embedding？</a>: 深入探讨机器学习 Embedding。</li><li><a href="https://parlance-labs.com/education/index.html">教育资源 – Parlance</a>: 未找到描述</li><li><a href="https://gabrielchua.me/posts/finetuning-tinyllama-axolotl-beginner/">Gabriel Chua - 使用 Axolotl 和 JarvisLab 微调 TinyLlama</a>: 未找到描述</li><li><a href="https://blog.trailofbits.com/2024/06/14/understanding-apples-on-device-and-server-foundations-model-release/">理解 Apple 发布端侧和服务器基础模型</a>: 作者 Artem Dinaburg。本周早些时候，在 Apple 的 WWDC 上，我们终于见证了 Apple 的 AI 策略。视频和现场演示伴随着两份长篇发布：Apple 的私有云计算（Private Cloud Compute）...</li><li><a href="https://x.com/gabrielchua_/status/1802371411526537254">来自 gabriel (@gabrielchua_) 的推文</a>: 使用 @axolotl_ai 和 @jarvislabsai 进行微调，作为 @HamelHusain 和 @dan_s_becker 的 LLM 微调课程会议的一部分，我做了一个生成大卫·爱登堡风格的玩具示例...</li><li><a href="https://github.com/eifuentes/awesome-embeddings">GitHub - eifuentes/awesome-embeddings: 🪁关于实体 Embedding 的精选资源列表</a>: 🪁关于实体 Embedding 的精选资源列表 - eifuentes/awesome-embeddings</li><li><a href="https://tenor.com/KhqP.gif">IT 狂人 Hello IT GIF - IT 狂人 Hello IT 你试过重启吗 - 发现并分享 GIF</a>: 点击查看 GIF
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1251598325954641970)** (14 messages🔥): 

- **额度混淆与解决**：一位用户意识到自己错过了申请额外额度的截止日期并寻求帮助，得到了积极回应并收到了 [GitHub 链接](https://github.com/setegonz/)。另一位用户询问了其账户状态，其额度在人工审核后已发放。
- **关于模型启动优化的讨论**：一位用户询问将模型权重复制到镜像中还是从 Volume 挂载是否会影响启动时间。他们得到的答复是，加载到镜像中的权重可能略有优势，但由于基础设施的统一，两者的差异微乎其微。
- **多轮对话问题及解决方案**：一位用户遇到了模型反复预测第一轮对话的问题，并被建议在相应频道进行讨论。他们随后通过将数据集格式更改为 Axolotl 的 [input_output 格式](https://openaccess-ai-collective.github.io/axolotl/docs/input_output.html) 解决了该问题。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://openaccess-ai-collective.github.io/axolotl/docs/input_output.html">Axolotl - 无模板提示词构建</a>：未找到描述</li><li><a href="https://github.com/setegonz/">setegonz - 概览</a>：setegonz 拥有 10 个公开仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/modal-labs/llm-finetuning/blob/main/nbs/inspect_data.ipynb">llm-finetuning/nbs/inspect_data.ipynb at main · modal-labs/llm-finetuning</a>：微调 Llama/Mistral/CodeLlama 等模型的指南 - modal-labs/llm-finetuning
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[learning-resources](https://discord.com/channels/1238365980128706560/1241089743933149204/1251983215992897557)** (5 messages): 

- **学习使用 TextGrad 进行提示词微调**：成员们讨论了 [TextGrad 项目](https://github.com/zou-group/textgrad)，该项目利用 LLM 来反向传播文本梯度。有人指出该项目被认为*优于 DSPy*，并附带了一个解释性的 [YouTube 视频](https://www.youtube.com/watch?v=Qks4UEsRwl0)。

- **无需安装即可使用 TextGrad**：一位成员询问是否可以在不安装任何内容的情况下，配合其 Anthropic/OpenAI API 密钥使用 TextGrad。另一位成员提到，他们尝试了示例 Colab Notebook，可以在其中设置 OpenAI API 密钥并测试其工作原理。

- **从零开始实现 LLM**：分享了一个 GitHub 仓库链接，提供了[在 PyTorch 中分步实现类 ChatGPT 的 LLM](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/ch07.ipynb) 的指南。对于那些有兴趣从底层学习和实验 LLM 开发的人来说，这是一个非常有用的资源。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/ch07.ipynb">LLMs-from-scratch/ch07/01_main-chapter-code/ch07.ipynb at main · rasbt/LLMs-from-scratch</a>：在 PyTorch 中从零开始分步实现类 ChatGPT 的 LLM - rasbt/LLMs-from-scratch</li><li><a href="https://github.com/zou-group/textgrad">GitHub - zou-group/textgrad: 通过文本实现自动“微分” —— 利用 LLM 反向传播文本梯度。</a>：通过文本实现自动“微分” —— 利用 LLM 反向传播文本梯度。 - zou-group/textgrad</li><li><a href="https://www.youtube.com/watch?v=Qks4UEsRwl0">斯坦福大学最新 TextGrad：优于 DSPy</a>：在这个 TEXTGRAD 框架中，每个 AI 系统都被转化为一个计算图，其中变量是复杂系统的输入和输出（不一定是微分...
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1251582844002766940)** (2 messages): 

- **关于表单截止日期的提醒**：*温馨提示，今天是填写表单的最后一天！如果你**尚未获得额度，但认为自己已经填过第一份表单，请填写这一份！***
- **第二份表单提交的额度发放**：如果你提交了第二份表单，“*我们还没有为这些申请发放额度，那将在周一之后进行。*” 有人提到在原始表单中很难找到某些用户。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1251259654701584466)** (2 条消息): 

- **用户跟进额度链接**：一位成员表示担心没有收到兑换 Replicate 额度的链接。他们提到已经通过 DM 发送了电子邮件和其他详细信息。

- **LoRA Adapter 部署咨询**：一位成员寻求在 **Replicate** 上部署 **LoRA adapters** 的帮助，并提到已成功使用 **Cog** 在本地运行微调后的 phi-3-mini。他们将其与 **Modal** 的流程进行了对比（在 **Modal** 中，卷是在运行时创建并绑定到容器的），并询问在 Replicate 上如何实现类似的方法。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1251275355646332990)** (5 条消息): 

- **关于 LangSmith Beta 额度与课程额度的澄清**：一位用户询问 “LangSmith Beta Credit” 是否与课程额度相同。另一位用户澄清说它们是不同的；“LangSmith Beta Credit” 是授予 Beta 用户的，而课程额度在账单中应显示为 'Mastering LLMs Course Credit'。
- **为缺失额度提供帮助**：一位用户向另一位觉得缺失课程额度的用户提供帮助。他们确认，如果提供额度表单中使用的电子邮件，他们可以核查情况。
- **用户查询缺失额度**：另一位用户询问在 LangSmith 上没有看到任何额度。他们请求帮助以了解是否需要从他们那边执行任何额外步骤。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[berryman_prompt_workshop](https://discord.com/channels/1238365980128706560/1242223275463938221/1252307321808486542)** (2 条消息): 

- **Promptfoo 引起成员兴趣**：一位成员对 **Promptfoo** 表示了兴趣，并感谢另一位成员的分享。 
- **相比 Promptfoo 更倾向于使用 inspect-ai**：另一位成员分享了他们对 **inspect-ai** 的偏好，理由是其灵活性以及与 Python 测试风格的契合度。然而，他们提到与 Promptfoo 相比，使用 inspect-ai 进行侧边对比（side-by-side comparisons）并不直观。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-3](https://discord.com/channels/1238365980128706560/1242223458184597534/1252141854133063772)** (4 条消息): 

- **Docker 执行期间的 CUDA 错误**：一位用户在容器中运行 Python 时遇到了 **Docker** 错误，提示信息为 "OCI runtime create failed: runc create failed: unable to start container process"。另一位用户建议这可能是由于 **CUDA** 设置不当或兼容性问题导致的。
- **问题难以复现**：一位回复者指出该问题难以复现，并表示 "*It’s hard to tell because I can’t replicate this issue*"（这很难判断，因为我无法复现这个问题）。这表明问题可能与特定环境或用户的特定配置有关。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[clavie_beyond_ragbasics](https://discord.com/channels/1238365980128706560/1242223963346698250/1251277734798430371)** (8 条消息🔥): 

- **RAGatouille 简化了 ColBERT 的使用**：一位成员称赞 [RAGatouille](https://github.com/bclavie/RAGatouille) 是在内部项目中将 **ColBERT** 与 Langchain 集成的绝佳工具。他们还推荐了 [Ben 的文章](https://ben.clavie.eu/ragatouille/)，认为它是对 ColBERT 的极佳介绍。
- **理解 RAG 中 Bi-encoder 的功能**：针对初学者关于 RAG 设置中 bi-encoders 背后逻辑的提问，另一位成员解释说，模型经过训练，通过前缀系统将查询（queries）和文档（documents）关联起来。回复强调了在模型训练期间定义“相似度”以适应不同用例的必要性。
- **探索学习资源**：一位成员正在寻找关于微调 **ColBERT** 和 rerankers 以及使用 embedding adapters 等高级主题的资源。他们对另一位成员推荐的 [关于构建最先进文本嵌入模型的 Medium 文章](https://medium.com/snowflake/how-to-build-a-state-of-the-art-text-embedding-model-a8cd0c86a19e) 表示感谢。
- **结合全文搜索与微调后的 Rerankers**：一位参与者讨论了他们使用 **lancedb** 并将通过 Lucene 的全文搜索与微调后的 rerankers 相结合的方法，取得了显著效果。他们提到没有使用 Ben 演讲中提到的向量数据库。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://ben.clavie.eu/ragatouille/">RAGatouille</a>: 未找到描述</li><li><a href="https://github.com/bclavie/RAGatouille">GitHub - bclavie/RAGatouille: Easily use and train state of the art late-interaction retrieval methods (ColBERT) in any RAG pipeline. Designed for modularity and ease-of-use, backed by research.</a>: 在任何 RAG 流水线中轻松使用和训练最先进的后期交互检索方法 (ColBERT)。专为模块化和易用性设计，并有研究支持。 - bclavie/RAGatouille
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jason_improving_rag](https://discord.com/channels/1238365980128706560/1242224099548332132/1252219319392141352)** (1 条消息): 

- **GPT-4o 的高效类别结构化**：成员们讨论了如何使用树状结构进行类别提示，从而提高 **GPT-4o** 在过滤器选择中的决策能力。尽管 system prompt 很大，但它运行良好，即便 **latency**（延迟）在 **GPT-4** 中曾是一个问题。 

- **文档的单向量策略**：该小组对每个文档/产品仅使用一个向量，并附带适当的 meta 标签。这种方法有助于维护一个精简且有效的分类系统。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jeremy_python_llms](https://discord.com/channels/1238365980128706560/1242224309548875917/1251957337724555486)** (3 条消息): 

- **通过分享的链接跟进讲座**：一位成员请求获取链接以跟进讨论。**Jeremy Howard** 迅速分享了[这个 Discord 链接](https://discord.gg/3ruJE6vB)，该成员表示了感谢。

**提到的链接**: <a href="https://discord.gg/3ruJE6vB">加入 fast.ai Discord 服务器！</a>: 查看 Discord 上的 fast.ai 社区 —— 与其他 10920 名成员一起交流，享受免费的语音和文字聊天。

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[saroufimxu_slaying_ooms](https://discord.com/channels/1238365980128706560/1242224552415596554/1251416118623735859)** (3 条消息): 

- **对新 Session 的期待**：一位用户以幽默的口吻询问了举行新 session 的可能性：*举行 session 的概率是多少？🫢🤪*。

- **即将开展的内存效率项目**：另一位用户告知大家，一个专注于 **memory efficiency**（内存效率）的新项目正在进行中。他们提到，一旦该项目准备就绪，可以期待一场“更有趣的演讲”。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1251280712477184094)** (27 条消息🔥): 

- **Strickvl 在使用本地 LORA 模型时遇到 OOM 错误**：尽管使用了两块 4090，Strickvl 在加载完整的 LORA 模型结果时仍面临显存溢出 (OOM) 错误。他们建议检查配置并考虑 Quantization（量化），并在 GitHub 上分享了他们的 [configs](https://github.com/strickvl/isafpr_finetune/tree/main/configs)。

- **Quantization 提供了一种节省内存的解决方案**：Chrislevy 指出，以 float32 加载的模型会消耗大量内存，并建议在推理时使用 `torch_dtype=torch.bfloat16`，如 [Llama 3 model card](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct#transformers-automodelforcausallm) 中所述。

- **axolotl 和 Finetuning 的文档缺失**：用户呼吁完善关于 Finetuning 的文档，特别是关于训练 LORA/QLORA 设置、保存模型以及正确的加载技术。Strickvl 强调了这一需求，并暗示使用 [Hamel 的课程仓库](https://github.com/parlance-labs/ftcourse/blob/master/06_sanity_check.ipynb) 进行 Sanity Checks。

- **Modal Labs 指南阐明了模型加载**：Andrewcka 提供了来自 [Modal Labs 推理脚本](https://github.com/modal-labs/llm-finetuning/blob/main/src/inference.py) 的代码见解，解释了该脚本如何通过日期时间识别最后训练的模型，以有效地处理推理。

- **使用 axolotl 进行多轮对话 Finetuning**：Huikang 询问了如何使 axolotl 适配多轮对话，并分享了 [CodeLlama 代码](https://github.com/modal-labs/llm-finetuning/blob/main/config/codellama.yml) 和 [axolotl 数据集格式](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/index.html) 等资源，用于对话 Finetuning。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/index.html">Axolotl - Dataset Formats</a>：未找到描述</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/config.html">Axolotl - Config options</a>：未找到描述</li><li><a href="https://github.com/modal-labs/llm-finetuning/blob/main/src/inference.py">llm-finetuning/src/inference.py at main · modal-labs/llm-finetuning</a>：Llama/Mistral/CodeLlama 等模型的 Finetuning 指南 - modal-labs/llm-finetuning</li><li><a href="https://github.com/strickvl/isafpr_finetune/blob/main/notebooks/sanity_check.py#L21)">isafpr_finetune/notebooks/sanity_check.py at main · strickvl/isafpr_finetune</a>：为从新闻稿中提取结构化数据而进行的 LLM Finetuning - strickvl/isafpr_finetune</li><li><a href="https://huggingface.co/blog/optimize-llm">Optimizing your LLM in production</a>：未找到描述</li><li><a href="https://github.com/parlance-labs/ftcourse/blob/master/06_sanity_check.ipynb">ftcourse/06_sanity_check.ipynb at master · parlance-labs/ftcourse</a>：通过在 GitHub 上创建账号来为 parlance-labs/ftcourse 的开发做出贡献。</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct#transformers-automodelforcausallm).">meta-llama/Meta-Llama-3-8B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://drchrislevy.github.io/posts/fine_tune_jarvis/fine_tune_jarvis.html#inference-with-the-model)">Chris Levy - Fine-Tuning LLMs with Axolotl on JarvisLabs</a>：未找到描述</li><li><a href="https://github.com/modal-labs/llm-finetuning/blob/main/config/codellama.yml">llm-finetuning/config/codellama.yml at main · modal-labs/llm-finetuning</a>：Llama/Mistral/CodeLlama 等模型的 Finetuning 指南 - modal-labs/llm-finetuning
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1251781955188555856)** (1 messages): 

- **对 Code Llama 发布的兴奋**：[Code Llama](https://huggingface.co/blog/codellama) 是 Llama 2 的演进版本，专门针对代码任务进行了微调，并在 Hugging Face 生态系统中发布。此次发布包括 Hub 上的模型、Transformers 集成以及多项为软件工程师提升生产力的功能。
- **发现格式差异**：注意到关于 Code Llama 的 Hugging Face 博客文章与用于微调 Code Llama 模型的 [GitHub 配置文件](https://github.com/modal-labs/llm-finetuning/blob/main/config/codellama.yml) 之间存在格式差异。强调这一点是为了确认此类差异是否可以接受。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/blog/codellama#conversational-instructions">Code Llama: Llama 2 learns to code</a>: 未找到描述</li><li><a href="https://github.com/modal-labs/llm-finetuning/blob/main/config/codellama.yml">llm-finetuning/config/codellama.yml at main · modal-labs/llm-finetuning</a>: 微调 Llama/Mistral/CodeLlama 等模型的指南 - modal-labs/llm-finetuning
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[charles-modal](https://discord.com/channels/1238365980128706560/1242564177952768062/1251273939586388099)** (1 messages): 

- **频道锁定通知**：该频道正在被锁定，成员被引导至[另一个频道](<#1241044231829848125>)向 Charles 提问。公告中包含了一个友好的表情符号 <:hugging_angel:936261297182482452>。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[simon_cli_llms](https://discord.com/channels/1238365980128706560/1242664474276659320/1251645300976652468)** (5 messages): 

- **CORS 错误阻止视频获取**：一名成员报告在尝试获取视频时遇到 **CORS 错误**。建议的解决方法是“打开原始 .mp4”。
- **怀疑 CloudFront 配置错误**：问题可能源于 **CloudFront 配置错误**，导致请求的 CORS 标头未被正确缓存。成员指出“CloudFront 会在第一次访问 URL 时缓存整个响应”，并且“它们的缓存不以 fetch 模式请求标头为键”。
- **提供视频链接**：可以通过[此链接](https://d33j6849mtsdhc.cloudfront.net/9364/10130/92796/81552506420/video_1.mp4)访问相关视频。成员询问该视频是否是“从 zoom 外部录制并通过 bucket 共享的”。

**提及的链接**: <a href="https://d33j6849mtsdhc.cloudfront.net/9364/10130/92796/81552506420/video_1.mp4">未找到标题</a>: 未找到描述

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[allaire_inspect_ai](https://discord.com/channels/1238365980128706560/1242943547699888229/1251953752156409957)** (3 messages): 

- **在 inspect_ai 中使用 instructor**：一名成员询问是否有办法在 inspect_ai 中使用类似 instructor 的工具，以确保输出格式有效。另一名成员建议要么实现并注册一个自定义模型，要么直接使用 tool calls，因为这正是 instructor 底层所做的。
- **inspect_ai 的灵活性**：一位用户指出 inspect_ai 允许用自定义解决方案替换现有基础设施或增强当前设置。 


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[credits-questions](https://discord.com/channels/1238365980128706560/1243721538432270388/1251529402097991710)** (3 messages): 

- **对 Braintrust Data 额度查询的困惑**：一位用户对在 **Braintrust Data 网站**上找不到额度查询位置表示沮丧：*"我甚至在 braintrustdata 网站上找不到哪里可以查额度。账单（billing）那里什么都不显示？"* 另一位用户建议在另一个频道寻求帮助，并强调他们也找不到额度状态。
- **引导至正确频道寻求解决方案**：一名成员建议将讨论转移到另一个频道，并艾特了另一位用户以寻求额度查询问题的潜在答案。他们承认在查找当前额度状态方面也遇到了类似的困难。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/1251463494700761202)** (6 messages): 

- **用户因额度问题涌向支持渠道**：多位用户请求协助处理账户中缺失的额度。提到的用户账户 ID 包括 *carljvh-7d2eb0*、*jalonso-e11d20*、*alex-kira-d15187*、*harrille-postia-723075*、*ayhanfuat-fa2dd5* 和 *data-94d7ef*。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[braintrust](https://discord.com/channels/1238365980128706560/1245407617031999581/1251530884520214562)** (3 条消息): 

- **用户寻求平台测试额度**：@peaky8linders 询问登录测试平台后仍看到升级按钮的问题，并询问是否仍能获得额度。他们提供了邮箱和组织信息以便验证。
- **额度已确认**：@ankrgyl 向 @peaky8linders 保证额度应该已经准备就绪。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[west-coast-usa](https://discord.com/channels/1238365980128706560/1245410680065097738/)** (1 条消息): 

.peterj: 有人在西雅图地区吗？
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[east-coast-usa](https://discord.com/channels/1238365980128706560/1245411101718610001/)** (1 条消息): 

ssilby: <@415846459016216576> 我加入！让我们组织一次 DMV 聚会吧 :3
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1251604430139166730)** (7 条消息): 

- **Predibase 误解数据集字段**：由于缺少 `text` 字段，一位用户在 Predibase 上使用 Alpaca/ShareGPT 格式的数据集时遇到问题。他们想知道如何处理无模板（template-free）数据集并相应地转换数据。
- **为 Predibase 调整正确的数据格式**：用户通过选择“指令微调（instruction tuning）”格式并根据 Predibase 的文档调整数据解决了问题。他们在此处分享了数据集供参考 [here](https://github.com/strickvl/isafpr_finetune/tree/main/data)。
- **在 Predibase 上进行测试数据评估**：用户注意到 Predibase 在使用测试数据进行评估方面的限制，并提到他们将在模型训练完成后进行评估。
- **从 Predibase 提取 Adapter**：用户询问是否可以下载或提取在 Predibase 上训练的 Adapter 以进行本地测试，从而避免部署自定义实例。

**提到的链接**：<a href="https://github.com/strickvl/isafpr_finetune/tree/main/data">isafpr_finetune/data at main · strickvl/isafpr_finetune</a>：为从新闻稿中提取结构化数据而进行的 LLM 微调 - strickvl/isafpr_finetune

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openpipe](https://discord.com/channels/1238365980128706560/1245927847437008896/1251976933844193310)** (3 条消息): 

- **数据集格式难题已解决**：一位成员询问了适用于 Openpipe 的正确数据集格式示例，提到尝试 axolotl 和无模板数据集未果。随后，他们通过按照 OpenAI 微调所使用的 OpenAI chat 格式对数据进行格式化解决了问题，并在 [GitHub](https://github.com/strickvl/isafpr_finetune/tree/main/data) 上分享了他们的数据集。

**提到的链接**：<a href="https://github.com/strickvl/isafpr_finetune/tree/main/data">isafpr_finetune/data at main · strickvl/isafpr_finetune</a>：为从新闻稿中提取结构化数据而进行的 LLM 微调 - strickvl/isafpr_finetune

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/)** (1 条消息): 

kramakurious: <@1010989949572612166> 这是你能帮忙解决的问题吗？
  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1251253795866935318)** (69 messages🔥🔥): 

- **Sakana AI 估值达到 10 亿美元**：Sakana AI 是一家开发 Transformer 模型替代方案的日本初创公司，从 NEA、Lux 和 Khosla 筹集了资金，估值达到 10 亿美元。更多详情请查看[此链接](https://www.theinformation.com/articles/openais-japanese-rival-gets-1-billion-valuation-from-silicon-valley-investors)。
  
- **Runway 的 Gen-3 Alpha 亮相**：Runway 推出了 Gen-3 Alpha，这是一个用于[视频生成](http://runwayml.com/gen-3-alpha)的新基础模型。据称该模型可以创建具有复杂场景变化和广泛电影选择的高细节视频。

- **DeepSeek-Coder-V2 表现亮眼**：DeepSeek-Coder-V2 发布，据报道在 [HumanEval 和 MATH 基准测试中均击败了 GPT-4](https://vxtwitter.com/teortaxesTex/status/1802681431992213767)。
  
- **Google DeepMind 的新视频转音频技术**：Google DeepMind 展示了其视频转音频 (V2A) 技术的进展，该技术能够为任何视频生成“无限数量”的音轨。[在此查看示例](https://x.com/rowancheung/status/1802734770117333257)。

- **Wayve 的新视图合成模型**：根据 [Jon Barron 的更新](https://x.com/jon_barron/status/1802758455830437975)，Wayve 发布了一个新的视图合成模型，该模型能够利用 4D Gaussians 从输入图像中出色地创建视图。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/rowancheung/status/1802734770117333257">Rowan Cheung (@rowancheung) 的推文</a>：Google DeepMind 刚刚分享了其新视频转音频 (V2A) 技术的进展。到目前为止，AI 生成的视频一直是静音的，这解决了这个问题。V2A 可以生成“无限数量”的音轨...</li><li><a href="https://x.com/jon_barron/status/1802758455830437975">Jon Barron (@jon_barron) 的推文</a>：Wayve 今天早些时候发布了一个新的视图合成模型。我猜这是一个由 4D Gaussians 组成的辐射场。不是生成式的，只是从输入图像进行视图合成。非常令人印象深刻。</li><li><a href="https://x.com/runwayml/status/1802691475391566108">Runway (@runwayml) 的推文</a>：介绍 Gen-3 Alpha：Runway 用于视频生成的全新基础模型。Gen-3 Alpha 可以创建具有复杂场景变化、广泛电影选择和详细艺术指导的高细节视频...</li><li><a href="https://x.com/dwarkesh_sp/status/1802771055016378554">Dwarkesh Patel (@dwarkesh_sp) 的推文</a>：我询问了 Buck 对 ARC-AGI 的看法，为采访 @fchollet 做准备。他告诉了他的同事 Ryan，在 6 天内，他们就击败了 ARC 上的 SOTA，并紧随人类平均表现之后...</li><li><a href="https://x.com/steph_palazzolo/status/1801690079922163954?s=46">Stephanie Palazzolo (@steph_palazzolo) 的推文</a>：最新消息（与 @nmasc_ @KateClarkTweets 合作）：Sakana AI，一家开发 Transformer 模型替代方案的日本初创公司，已从 NEA、Lux 和 Khosla 筹集资金，估值达 10 亿美元。更多信息：https://www.theinform...</li><li><a href="https://x.com/natolambert/status/1802762956469579956">Nathan Lambert (@natolambert) 的推文</a>：是什么让所有这些文本转视频模型在同一个 6 个月的时间窗口内都变得如此出色？仅仅是因为人们之前没尝试吗？它们几乎同时出现，这太疯狂了。L...</li><li><a href="https://elevenlabs.io/sound-effects">AI 文本转音效生成器</a>：使用我们的 AI 音效生成器，可以免费根据文本提示生成任何能想象到的声音。非常适合视频、播客或任何其他音频制作。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1251404893974036561)** (4 条消息): 

- **Sam Altman 暗示 OpenAI 治理结构变革**：Jacques Thibault 的一条推文引用了 Sam Altman 的私下表态，暗示 OpenAI 可能会转型为营利性业务。此举可能使其能够进行公开上市，并允许 Altman 获得 OpenAI 的股份。[阅读完整推文](https://x.com/jacquesthibs/status/1801782465364640247?s=46)。

- **The Information 报道 OpenAI 的潜在转型**：The Information 详细报道称，Altman 在私下提到 OpenAI 可能会转型为一家公益企业 (Benefit Corporation)，类似于 Anthropic 和 xAI。这种转型可能会推动 OpenAI 走向上市。[阅读文章详情](https://www.theinformation.com/articles/openai-ceo-says-company-could-become-benefit-corporation-akin-to-rivals-anthropic-xai?utm_campaign=Editorial&utm_content=Article&utm_medium=organic_social&utm_source=twitter)。

- **社区反应冷淡且持怀疑态度**：一位成员对这些进展表示怀疑，将其感受总结为“*这太离谱了，笑死 (This is so sketch lmao)*”。

**提到的链接**：<a href="https://x.com/jacquesthibs/status/1801782465364640247?s=46">Jacques (@JacquesThibs) 的推文</a>：&#34;Sam Altman 最近告诉一些股东，OAI 正在考虑将其治理结构更改为不受 OAI 非营利董事会控制的营利性业务。[...] 可能会开启 ...

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1251252934457425920)** (63 条消息🔥🔥): 

- **对 Interconnects 周边的评价**：成员们讨论了周边的质量，指出虽然贴纸反响平平，但 T 恤很受欢迎。一位成员提到，“贴纸质量不行，需要换个供应商。”
- **剖析 ARC-AGI 性能**：链接到一篇 [Redwood Research 的文章](https://redwoodresearch.substack.com/p/getting-50-sota-on-arc-agi-with-gpt)，讨论了提高 ARC-AGI 性能的方法，引发了辩论。成员们批评了使用大量样本的方法，认为这更多是靠运气撞大运，而不是真正的扩展 (Scaling)。
- **探索神经符号 AI (Neurosymbolic AI)**：成员们深入探讨了神经符号 AI，质疑利用 LLM 进行离散程序搜索是否真正符合传统定义。讨论围绕 [François Chollet 的一条推文](https://x.com/fchollet/status/1802773156341641480?s=46)展开，分析当前的 AI 技术是否足够，还是需要根本性的突破。
- **MidJourney 的新动向**：MidJourney 正在向硬件领域扩张，并预计在 1 月份启动其视频模型的训练。CEO David Holz 在 Discord 的“Office Hour”环节确认了这一消息。
- **学术会议的纠结**：一位成员思考了尽管从加州出发交通不便，参加在泰国举行的 ACL 是否有价值，并质疑其与 NeurIPS 等顶级会议相比的相关性。另一位成员回复道：“我不认为这是非去不可的 (do or die)”，建议可以自愿参加。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://apparate.ai/">Apparate AI</a>：未找到描述</li><li><a href="https://x.com/fchollet/status/1802773156341641480?s=46">François Chollet (@fchollet) 的推文</a>：@dwarkesh_sp 这是目前最有前途的分支方法——通过利用 LLM 作为采样程序或分支决策的方式，来辅助离散程序搜索....</li><li><a href="https://redwoodresearch.substack.com/p/getting-50-sota-on-arc-agi-with-gpt">通过 GPT-4o 在 ARC-AGI 上达到 50% (SoTA)</a>：你只需要抽取更多样本</li><li><a href="https://github.com/rgreenblatt/arc_draw_more_samples_pub/blob/0b36f4584aebae9ec876d3510842b3651e719d67/arc_solve/edit_distance.py#L115).">rgreenblatt/arc_draw_more_samples_pub 仓库</a>：抽取更多样本。通过在 GitHub 上创建账号为 rgreenblatt/arc_draw_more_samples_pub 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1251286628757016788)** (9 条消息🔥): 

- **包含 Excalidraw 图表的 RAG 和 Agents 指南备受关注**：@_nerdai_ 分享了一份详尽的 [构建 RAG 和 Agents 的演示文稿](https://t.co/oibSiDjseO)。该指南包含完整的 Excalidraw 图表，深入浅出地解析了从基础到高级的概念。
- **Arize 集成增加了端到端的可观测性**：新的 instrumentation 模块已与 Arize 集成，并在该 [指南](https://t.co/cOBP9IOjro) 中进行了演示。它展示了如何在 LLM 应用中配置自定义的 event/span 处理程序。
- **AI World's Fair 总结，顶尖演讲者云集**：参加由 AI Engineer World's Fair 举办的 [AI Sizzle and Waves 活动](https://t.co/O6WAkbI9jt)，聆听 @jerryjliu0、@freddie_v4、@atitaarora 等人的演讲。活动由 Angela Tse、Atita Arora 和 Julia Neagu 主持。
- **全栈 Agents 入门指南发布**：@MervinPraison 的教程提供了一个分步 [指南](https://t.co/BtP0iRrjBq)，介绍如何使用本地模型和 @chainlit_io 构建 Agent 的核心组件。该教程旨在创建简单的应用程序。
- **使用 Claude 3 和 SingleStoreDB 的多模态 RAG 流水线**：@Pavan_Belagatti 在他的 [文章](https://t.co/u80hMB0ioz) 中讨论了多模态 RAG 的未来角色，该文章利用了 @AnthropicAI 的 Claude 3 和 @SingleStoreDB。该流水线解决了文档中普遍存在的图像处理问题。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://t.co/dhWHELRUBJ">AI Engineer World's Fair 闭幕式：AI Sizzle and Waves @ GitHub HQ · Luma</a>：以一个 AI 夏日周五为展会画上句号。享受清爽的甜点，并从同行开发者和创新者的新颖见解中汲取灵感。听取我们的……</li><li><a href="https://t.co/cOBP9IOjro">openinference/python/instrumentation/openinference-instrumentation-llama-index at main · Arize-ai/openinference</a>：AI 可观测性的自动插桩 (Auto-Instrumentation)。通过在 GitHub 上创建账号为 Arize-ai/openinference 的开发做出贡献。</li><li><a href="https://t.co/m3YOjOF36q">Instrumentation: 基础用法 - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1251269460233289789)** (95 条消息🔥🔥): 

- **为 RAG 对客服邮件进行分块 (Chunking)**：一位成员询问如何基于邮件对话为客服 RAG 模型创建分块。另一位成员建议捕获每个链条的第一封邮件，以确保每封邮件都被包含在内。
- **从 markdown 文档生成特定输出**：一位用户遇到 LlamaIndex 截断 markdown 文档中相关语言的问题。他们需要精确的输出而不需要摘要，并正在寻求改进建议。
- **在 LlamaIndex 中使用 Neo4j**：关于将 Neo4j 知识图谱转换为 LlamaIndex 属性图 (property graphs) 提出了多个查询。分享了详细的说明和 LlamaIndex 文档链接 ([LlamaIndex 属性图示例](https://docs.llamaindex.ai/en/latest/examples/property_graph/property_graph_neo4j/))。
- **重叠句子检索**：一位用户询问在使用句子级检索时，扩展句子重叠的问题。澄清了重叠的句子不会被合并，需要自定义后处理。
- **保存 ChatMemoryBuffer**：讨论了将 `ChatMemoryBuffer` 对象保存为文件格式，以管理长对话中的 token 限制。建议了一种将聊天记忆保存为字典并存储在 JSON 文件中的方法。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lu.ma/kqxmbuou?">LlamaIndex 网络研讨会：使用知识图谱的高级 RAG（与来自 Neo4j 的 Tomaz 合作）· Zoom · Luma</a>：我们将在本周四上午 9 点（太平洋时间）举办一场关于高级知识图谱 RAG 的特别研讨会，由来自 Neo4j 的独一无二的 Tomaz Bratanic 主持。在本次网络研讨会中，你将……</li><li><a href="https://github.com/run-llama/llama_index/tree/02984efc5004126ccaffa15ec599d0dacce55dd3/llama-index-integrations/storage">run-llama/llama_index 中的 llama_index/llama-index-integrations/storage</a>：LlamaIndex 是适用于你的 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/property_graph/property_graph_neo4j/#loading-from-an-existing-graph>).">Neo4j 属性图索引 - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/01e5173f8a272e8b7e5ccb2ae3ff215eb6c4ca6a/llama-index-core/llama_index/core/query_engine/knowledge_graph_query_engine.py#L70">run-llama/llama_index 中的 llama_index/llama-index-core/llama_index/core/query_engine/knowledge_graph_query_engine.py</a>：LlamaIndex 是适用于你的 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/01e5173f8a272e8b7e5ccb2ae3ff215eb6c4ca6a/llama-index-core/llama_index/core/query_engine/knowledge_graph_query_engine.py#L132">run-llama/llama_index 中的 llama_index/llama-index-core/llama_index/core/query_engine/knowledge_graph_query_engine.py</a>：LlamaIndex 是适用于你的 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/llm/monsterapi/#rag-approach-to-import-external-knowledge-into-llm-as-context>)">未找到标题</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/retrievers/relative_score_dist_fusion/#setup>)">相对分数融合与基于分布的分数融合 - LlamaIndex</a>：未找到描述</li><li><a href="https://direct.mit.edu/qss/article/2/4/1423/108045/A-framework-for-creating-knowledge-graphs-of">一个用于创建科学软件元数据知识图谱的框架</a>：摘要。越来越多的研究人员依靠计算方法来生成或处理其科学出版物中描述的结果。为此创建的软件——科学软件……
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1251880106566356993)** (6 条消息): 

- **通过网络爬虫和 RAG 增强 LLM！**：[如何通过网络爬虫和 RAG 增强 LLM](https://medium.com/ai-advances/how-to-power-up-llms-with-web-scraping-and-rag-975a165587f6) 探讨了通过网络爬虫和检索增强生成 (RAG) 来提升 LLM 性能。文章重点介绍了用于干净 Markdown 提取的 Firecrawl 和支持多种输出格式的 Scrapfly 等工具。
- **LLM 应用中的 Firecrawl 与 Scrapfly**：*"Firecrawl 在 Markdown 方面表现出色"*，使其成为为 LLM 准备数据的理想选择。Scrapfly 提供了多种输出格式的灵活性，但可能需要额外的处理来进行 LLM 优化。
  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1251251986846847137)** (39 条消息🔥): 

- **autogen_stubs.sh 中的脚本缩进损坏**：一名成员在 `autogen_stubs.sh` 脚本中遇到了问题，其中 `clang2py` 破坏了缩进，导致语法错误。讨论显示，对于在 GPU 上运行 tinygrad 的预期任务，该脚本并非必需。
- **OpenCL 安装问题导致错误**：OpenCL 安装问题导致在 GPU 上运行 tinygrad 时出现错误。George Hotz 建议修复 OpenCL 设置并检查 `clinfo` 以进行故障排除。
- **改进 OpenCL 错误消息**：社区讨论了通过从 OpenCL 头文件自动生成错误消息来增强 OpenCL 错误提示。已开启一个 [pull request](https://github.com/tinygrad/tinygrad/pull/5004) 以实现更好的错误消息。
- **需要 Process replay 文档**：George Hotz 请求添加关于 Process replay 的文档以协助新贡献者。这是为了简化使用新样式重写操作的过程。
- **周一会议议程主题**：重要主题包括 tinybox 发布、0.9.1 版本发布、CI benchmark 持续时间、移除 numpy 以及各种技术讨论。亮点还包括性能里程碑，例如在多 GPU 设置下为 llama 7B 实现 200 tok/s。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/__tinygrad__/status/1802442339069333984">来自 tiny corp (@__tinygrad__) 的推文</a>：@PennJenks 这是三个 kernel。我们需要将它们融合为一个。 GRAPH=1 python3 -c &#34;from tinygrad import Tensor; Tensor.rand(100,100).softmax().realize()&#34;</li><li><a href="https://x.com/__tinygrad__/status/1747467257889116379">来自 tiny corp (@__tinygrad__) 的推文</a>：tinybox 拥有一个基于 USB 3 的 1TB NVMe 启动盘，以及 4 个各占 4 条 PCI-E 4.0 通道的 1TB NVMe；4TB 用于存放权重和数据集。这不是理论，这是真实的 benchmark。它比...上的 RAM 更快。</li><li><a href="https://github.com/tinygrad/tinygrad/pull/4899/files,">nimlgen 为 ioctls 提供的更好 nv 错误消息 · Pull Request #4899 · tinygrad/tinygrad</a>：无描述</li><li><a href="https://github.com/tinygrad/tinygrad/pull/5004">GabrielZCode 修复/改进 OpenCL 错误消息 · Pull Request #5004 · tinygrad/tinygrad</a>：更好的 OpenCL 错误消息！！使用与 generate_stubs.sh 中 generate_nv() 函数相同的策略，我从 https://github.com/KhronosGroup/OpenCL-Headers/tree/m... 提取了错误消息。</li><li><a href="https://streamhpc.com/blog/2013-04-28/opencl-error-codes/">OpenCL 错误代码 (1.x 和 2.x) - StreamHPC</a>：熟记所有错误对快速编程很有帮助，但并不总是最佳选择。因此我开始创建 ...</li><li><a href="https://github.com/KhronosGroup/OpenCL-Headers/blob/main/CL/cl.h">KhronosGroup/OpenCL-Headers 主分支下的 OpenCL-Headers/CL/cl.h</a>：Khronos OpenCL-Headers。通过在 GitHub 上创建账户为 KhronosGroup/OpenCL-Headers 的开发做出贡献。</li><li><a href="https://github.com/KhronosGroup/OpenCL-Headers/blob/main/CL/cl_ext.h">KhronosGroup/OpenCL-Headers 主分支下的 OpenCL-Headers/CL/cl_ext.h</a>：Khronos OpenCL-Headers。通过在 GitHub 上创建账户为 KhronosGroup/OpenCL-Headers 的开发做出贡献。</li><li><a href="https://github.com/KhronosGroup/OpenCL-Headers/blob/main/CL/cl_egl.h">KhronosGroup/OpenCL-Headers 主分支下的 OpenCL-Headers/CL/cl_egl.h</a>：Khronos OpenCL-Headers。通过在 GitHub 上创建账户为 KhronosGroup/OpenCL-Headers 的开发做出贡献。</li><li><a href="https://github.com/KhronosGroup/OpenCL-Headers/blob/main/CL/cl_dx9_media_sharing.h">KhronosGroup/OpenCL-Headers 主分支下的 OpenCL-Headers/CL/cl_dx9_media_sharing.h</a>：Khronos OpenCL-Headers。通过在 GitHub 上创建账户为 KhronosGroup/OpenCL-Headers 的开发做出贡献。</li><li><a href="https://github.com/KhronosGroup/OpenCL-Headers/blob/main/CL/cl_d3d11.h">KhronosGroup/OpenCL-Headers 主分支下的 OpenCL-Headers/CL/cl_d3d11.h</a>：Khronos OpenCL-Headers。通过在 GitHub 上创建账户为 KhronosGroup/OpenCL-Headers 的开发做出贡献。</li><li><a href="https://github.com/KhronosGroup/OpenCL-Headers/blob/main/CL/cl_d3d10.h">KhronosGroup/OpenCL-Headers 主分支下的 OpenCL-Headers/CL/cl_d3d10.h</a>：Khronos OpenCL-Headers。通过在 GitHub 上创建账户为 KhronosGroup/OpenCL-Headers 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1251541160023949394)** (69 messages🔥🔥): 

- **George Hotz 针对递归重写断言（assert）进行说明**：一位成员询问了 `uops graph_rewrite` 中用于计数递归重写的断言。 [该断言](https://docs.tinygrad.org/tensor/creation/#creation-random) 确保递归重写循环次数低于阈值，以防止无限递归。
  
- **`beautiful_mnist_multigpu.py` 中的梯度同步得到简化**：George Hotz 确认梯度同步是 Tinygrad 优化器固有的功能。他强调了其相对于 Torch 的 Distributed Data Parallel 的简洁性。

- **Tinygrad 超越 PyTorch 的目标**：George Hotz 讨论了 Tinygrad 在速度、API 简洁性和减少 Bug 方面超越 PyTorch 的目标。虽然目前较慢（特别是在 LLM 训练中），但热心用户强调了 Tinygrad 的纯粹性和潜力。

- **混合精度实现讨论**：一位用户就如何为模型实现混合精度向 George Hotz 寻求建议，讨论了包括使用 `DEFAULT_FLOAT` 和修改 `nn` 类在内的各种方法。George 建议使用 `cast_` 方法和后期转换（late casting）技术以提高效率。

- **Kernel 问题已解决**：一位用户解决了与 `remainder` 张量未出现在 UOp 图中相关的 Kernel 问题，了解到独立的 `realize` 调用会将操作拆分为不同的 Kernel。讨论强调了适当执行张量 `realize` 以满足自定义加速器要求的重要性。

**相关链接**：<a href="https://docs.tinygrad.org/tensor/creation/#creation-random">Creation - tinygrad docs</a>：未找到描述

  

---



### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1252251816423587851)** (1 messages): 

- **介绍 GPT Notes 应用**：一位成员展示了一个结合了 LLM 客户端和笔记应用的混合应用，允许用户动态地在 LLM 的上下文中包含或排除笔记。该项目在不使用任何 JS 库的情况下构建，提供导入/导出、基础 Markdown 和响应管理等功能。
- **不支持移动端，纯原生 JS**：尽管不支持移动端，该应用以不依赖任何库、纯原生 JavaScript 构建为傲。它包括在浏览器本地存储 API 密钥、历史记录和笔记的功能。
- **在 Codepen 上探索该应用**：该成员提供了项目的 [Codepen 链接](https://codepen.io/bulgakoff08/project/editor/DnJLrG) 和一个 [已部署的全屏应用](https://000700836.deployed.codepen.website/)。该应用可作为任何寻找类似工具的人的参考示例。

**相关链接**：<a href="https://000700836.deployed.codepen.website/">GPNotes</a>：未找到描述

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1251345585571762266)** (68 条消息🔥🔥): 

- **OpenRouter 在没有 User 消息时报错引发讨论**：用户讨论了如果未发现 User 消息，OpenRouter 会返回错误的问题。他们指出，某些模型要求至少有一条 User 消息作为开头，甚至由于其 Instruct-tuned 格式，并非所有模型都支持以 Assistant 消息开头。建议的解决方法是使用 `prompt` 参数而非 `messages` ([OpenRouter Docs](https://openrouter.ai/docs/transforms))。

- **文档格式化与上传困扰用户**：一位用户咨询了将文本格式化为结构化“论文”的服务，引发了关于文档格式化和上传的广泛讨论。对话强调了使 PDF 对 LLM 友好的复杂性，并建议使用 [PDF.js 和 Jina AI Reader](https://jina.ai/reader/) 等工具对 PDF 进行预处理。

- **Qwen2 的审查机制遭到批评**：用户分享了使用 Qwen2 模型的体验，称其尽管经过 Jailbreak 尝试，但审查依然过度，表现为叙事结果呈现出不切实际的正面倾向。推荐了如 Dolphin Qwen 2 等审查较少的替代模型。

- **Gemini Flash 的上下文限制争议**：Gemini Flash 的 Token 生成限制差异引发了疑问，OpenRouter 列出的是 22k Token，而 Gemini 文档声称是 8k。经澄清，OpenRouter 是为了匹配 Vertex AI 的计费模型而计算字符数 ([OpenRouter Status](https://openrouter.ai/models/google/gemini-flash-1.5/status))。

- **速率限制与模型配置问题出现**：用户询问了 GPT-4o 和 Opus 等模型的 Rate Limits，随后引导用户通过 API Key 查看速率限制 ([OpenRouter Rate Limits](https://openrouter.ai/docs/limits))。此外，还展开了关于最大化模型性能和配置设置的讨论，如“来自 OpenRouter 的 Sonnet 对比使用 Claude Key 的 Sonnet”以及“LiteLLM 对比 OpenRouter 路由”，强调了自定义重试选项和 API 调用效率。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/models/google/gemini-flash-1.5/status">Google: Gemini Flash 1.5 (preview) – 提供商状态与负载均衡</a>：查看提供商状态并向 Google: Gemini Flash 1.5 (preview) 发起负载均衡请求 - Gemini 1.5 Flash 是一个基础模型，在各种多模态任务（如视觉理解...）中表现良好。</li><li><a href="https://openrouter.ai/docs/limits">Limits | OpenRouter</a>：设置模型使用限制</li><li><a href="https://jina.ai/reader/">Reader API</a>：读取 URL 或搜索网络，为 LLM 提供更好的 Grounding。</li><li><a href="https://openrouter.ai/docs/transforms">Transforms | OpenRouter</a>：为模型消耗转换数据
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[일반](https://discord.com/channels/1091220969173028894/1246338143226167349/)** (1 条消息): 

is.maywell: <:a6adc388ea504e89751ecbbd50919d3a:1240669253699637339>
  

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1251376496178892821)** (48 条消息🔥): 

- **LangChain 中的 TextGen 集成损坏**：一名成员报告称，由于 **API 更新**，**LangChain** 中的 **textgen 集成** 已损坏。
- **教科书 PDF Chunking 的最佳 Splitter**：一位成员询问关于根据标题和章节对 **PDF 文本进行 Chunking** 的最佳 **Splitter** 建议，旨在更好地结构化文本。
- **LangChain Postgres 安装困难**：用户交流了关于安装 **langchain_postgres** 的建议，解决方案涉及纠正 `pip install` 的目标目录。
- **新版本 Tenacity 导致的模块错误**：一位用户在更新到 **8.4.0** 版本后遇到了 'tenacity.asyncio' 的 **ModuleNotFoundError**，但发现回退到 **8.3.0** 版本解决了该问题。
- **LangChain 新用户帮助**：多位用户寻求在 **LangChain** 中实现特定模型或错误处理的指导，包括从 Python 代码迁移到 **LangChain JS**、管理 HuggingFace 模型，以及推荐用于本地使用的 LLM（如 **Llama 3** 或 **Google Gemini**）。相关讨论链接见[此处](https://github.com/langchain-ai/langchain/discussions/22899)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="http://'">未找到标题</a>: 未找到描述</li><li><a href="https://'">未找到标题</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=qKZLDEIL2r0">如何在 Slack 上构建 ChatGPT 聊天机器人</a>: 欢迎观看本教程视频，了解如何使用 OpenAI 语言模型、LangChain 和 Slack Bolt 库创建 Slack 聊天机器人。本视频将展示...</li><li><a href="https://www.instagram.com/reel/C8SMrCIOyGZ/?igsh=MzdrbGhrbXh4NWk=">Vaios Laschos 在 Instagram 上: &quot;这是我为 NVIDIA 和 LangChain 举办的 Generative AI Agents 开发者竞赛创建的应用宣传视频。#NVIDIADevContest #LangChain &#064;nvidiadeveloper 

演示视频请见：

https://www.youtube.com/watch?v=9mMbQpofiJY

该应用通过自动化一些繁琐的工作（如从 arxiv 获取文件、制作摘要和执行基于上下文的翻译）使学术生活变得更轻松。未来的目标是仅凭一篇论文就能生成论文综述。

如果你想找点苦头吃，请查看我的 GitHub 仓库 https://github.com/artnoage/Langgraph_Manuscript_Workflows&quot;</a>: 72 个赞，1 条评论 - vaioslaschos 于 2024 年 6 月 16 日: &quot;这是我为 NVIDIA 和 LangChain 举办的 Generative AI Agents 开发者竞赛创建的应用宣传视频....&quot;. </li><li><a href="https://tenor.com/view/blowing-kisses-kisses-kiss-gratitude-huge-thanks-gif-16468716440995283694">飞吻感谢 GIF - Blowing kisses Kisses Kiss - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/langchain-ai/langchain/discussions/22899">如何正确地向模型提供输入 schema · langchain-ai/langchain · Discussion #22899</a>: 检查了其他资源。我为此问题添加了一个非常详细的标题。我使用集成搜索查询了 LangChain 文档。我使用 GitHub 搜索来查找类似的问题并...
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1251321854589931601)** (14 messages🔥): 

- **R2R 新增自动知识图谱构建功能**：R2R v2 现在包含自动知识图谱构建功能，并附带一份全面的 [cookbook](https://r2r-docs.sciphi.ai/cookbooks/knowledge-graph#knowledge-graph-agents)，详细介绍了基础和高级功能。*"如果你对 KG 感兴趣，这将是一个极佳（且最新）的起点。"*

- **Collision 活动交互式地图上线**：Eloquentsyntax 发布了一个针对 Collision 派对和活动的 [交互式地图](https://collision.talewind.ai/)。该地图包含筛选器、门票费用、地址、RSVP 链接以及用于轻松查找活动的 AI 聊天功能。

- **CryptGPT：使用维吉尼亚密码的隐私保护 LLM**：Diwank 介绍了 [CryptGPT](https://x.com/diwanksingh/status/1802118343446724655)，这是一个在维吉尼亚密码密文上预训练 GPT-2 模型的项目，确保了模型提供商无法窥探隐私。其独特之处在于使用时需要知道加密密钥。

- **使用 GPT 抓取网页并创建图表**：Ashes47 分享了用户 Anuj4799 的一个项目，该用户创建了一个用于生成技术图表的自定义 GPT。演示可以在 [这里查看](https://chat.openai.com/g/g-7EWovgPuJ-mindstream)。

- **Rubik's AI 测试人员招募与促销**：Paulm24 邀请用户测试一款先进的研究助手和搜索引擎，使用促销代码 `RUBIX` 可获得包含 GPT-4 Turbo 和 Claude 3 Opus 等模型的 2 个月免费高级版。感兴趣的用户请前往 [Rubik's AI](https://rubiks.ai/) 注册。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://collision.talewind.ai/">未找到标题</a>：未找到描述</li><li><a href="https://x.com/Anuj4799/status/1802425335549661492">来自 Anuj Verma (@Anuj4799) 的推文</a>：我在生成技术图表时遇到了困难，最终创建了一个自定义 GPT 来帮我处理。现在，我非常喜欢它的便捷和高效！点击查看：https://chat.opena...</li><li><a href="https://x.com/diwanksingh/status/1802118343446724655">来自 Diwank Singh (@diwanksingh) 的推文</a>：http://x.com/i/article/1802116084507848704</li><li><a href="https://r2r-docs.sciphi.ai/cookbooks/knowledge-graph#knowledge-graph-agents].">R2R 文档</a>：RAG to Riches (R2R) 框架的官方文档。</li><li><a href="https://www.appstorm.ai/">Appstorm.ai：用于轻松开发应用程序的生成式 AI</a>：未找到描述</li><li><a href="https://rubiks.ai/">Rubik's AI - AI 研究助手与搜索引擎</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/)** (1 messages): 

emarco: https://www.youtube.com/watch?v=0gJLFTlGFVU
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1251277835323310130)** (21 messages🔥): 

<ul>
    <li><strong>OtterTune 已关停</strong>：OtterTuneAI 在收购交易失败后正式关闭。该公告已在 <a href="https://x.com/andy_pavlo/status/1801687420330770841?s=46&t=Tc6nPt_FP2Ybqya6_6Xu-w">Twitter</a> 上发布。</li>
    <li><strong>在 Hugging Face 上查看 Apple 的模型</strong>：Apple 在 Hugging Face 上发布了多个针对设备端性能优化的模型，包括用于语义分割的 <a href="https://huggingface.co/apple/coreml-detr-semantic-segmentation">DETR Resnet50 Core ML</a> 和 <a href="https://huggingface.co/collections/apple/core-ml-stable-diffusion-666b3b0f4b5f3d33c67c6bbe">Stable Diffusion Core ML</a>。</li>
    <li><strong>OpenAI 因任命前 NSA 局长而遭受抨击</strong>：Edward Snowden 批评了 OpenAI 任命前 NSA 局长 Paul M. Nakasone 进入董事会的决定，<a href="https://x.com/Snowden/status/1801610725229498403">称其为对公众信任的背叛</a>。</li>
    <li><strong>Runway 发布 Gen-3 Alpha 视频模型</strong>：Runway 推出了 Gen-3 Alpha，这是一款具有先进功能的视频生成新模型。详情已在 <a href="https://x.com/runwayml/status/1802691475391566108?s=46&t=90xQ8sGy63D2OtiaoGJuww">Twitter</a> 上分享。</li>
    <li><strong>Anthropic 关于奖励篡改的研究</strong>：Anthropic 发表了一篇关于 AI 模型学习破解其奖励系统的新论文。研究及其发现总结在他们的 <a href="https://anthropic.com/research/reward-tampering">博客文章</a> 中。</li>
</ul>

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://studios.trychroma.com/flo-on-lindy">Flo Crivello 谈构建 Lindy.AI</a>：未找到描述</li><li><a href="https://x.com/anthropicai/status/1802743256461046007?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Anthropic (@AnthropicAI) 的推文</a>：Anthropic 的新研究：调查奖励篡改 (Reward Tampering)。AI 模型是否会学习破解自己的奖励系统？在一篇新论文中，我们展示了它们可以通过从简单环境中的训练进行泛化来实现这一点。...</li><li><a href="https://huggingface.co/apple">apple (Apple)</a>：未找到描述</li><li><a href="https://x.com/bshlgrs/status/1802766374961553887?t=1MOp6l3T7xJvK6yAiGomPw&s=19">来自 Buck Shlegeris (@bshlgrs) 的推文</a>：ARC-AGI 在过去一周被炒作为 LLM 无法解决的基准测试。这一说法刺激了我亲爱的同事 Ryan Greenblatt，于是他上周尝试用 LLM 来解决它。Ryan 获得了 71...</li><li><a href="https://x.com/airkatakana/status/1801796780595757175?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Air Katakana (@airkatakana) 的推文</a>：我宣布见顶了，这家公司甚至还没做出任何东西</li><li><a href="https://x.com/tomgoldsteincs/status/1802726878924464273">来自 Tom Goldstein (@tomgoldsteincs) 的推文</a>：LLM 会记住训练数据，从而导致版权/隐私风险。Goldfish loss 是一种训练 LLM 而不记忆训练数据的巧妙技巧。我可以在《哈利·波特》的开头训练一个 7B 模型...</li><li><a href="https://x.com/gdb/status/1802707715816595869?t=Y7yDU61o45Cpx69DHss-jQ&s=19">来自 Greg Brockman (@gdb) 的推文</a>：GPT-4o 作为助手帮助医生筛查和治疗癌症患者：引用 Othman Laraki (@othman) 的话：我很高兴宣布 @Color Copilot，这是我们与 ... 合作开发的。</li><li><a href="https://x.com/runwayml/status/1802691475391566108?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Runway (@runwayml) 的推文</a>：介绍 Gen-3 Alpha：Runway 新的视频生成基础模型。Gen-3 Alpha 可以创建具有复杂场景变化、广泛电影选择和详细艺术指导的高细节视频...</li><li><a href="https://x.com/andy_pavlo/status/1801687420330770841?s=46&t=Tc6nPt_FP2Ybqya6_6Xu-w">来自 Andy Pavlo (@andy_pavlo@discuss.systems) (@andy_pavlo) 的推文</a>：我很遗憾地宣布 @OtterTuneAI 正式倒闭了。我们的服务已关闭，今天解雇了所有人（提前 1 个月通知）。我无法详细说明发生了什么，但我们被坑了...</li><li><a href="https://x.com/fchollet/status/1802801425514410275">来自 François Chollet (@fchollet) 的推文</a>：关于解决 ARC-AGI 的前进之路... 如果你正在生成大量程序，并使用符号检查器检查每一个（例如运行程序的实际代码并验证输出），并且 ...</li><li><a href="https://x.com/bshlgrs/status/1802766374961553887?t=1MOp6l3T7xJvK6yA">来自 Buck Shlegeris (@bshlgrs) 的推文</a>：ARC-AGI 在过去一周被炒作为 LLM 无法解决的基准测试。这一说法刺激了我亲爱的同事 Ryan Greenblatt，于是他上周尝试用 LLM 来解决它。Ryan 获得了 71...</li><li><a href="https://x.com/Snowden/status/1801610725229498403">来自 Edward Snowden (@Snowden) 的推文</a>：他们已经彻底撕下面具了：**永远不要**信任 @OpenAI 或其产品（ChatGPT 等）。任命 @NSAGov 局长进入董事会只有一个原因。这是一个故意的、经过计算的...</li><li><a href="https://readwise.io/reader/shared/01j02hxfc781zchghjvdrfs30x/">Flo Crivello 谈构建 Lindy.AI | Daniel 注释</a>：AI Agent 是一类新型软件，构建在大语言模型 (LLM) 之上。
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1251265169133146202)** (20 条消息🔥): 

- **Prime Intellect 将开源 DiLoco 和 DiPaco**：用户讨论了 Prime Intellect 计划很快发布 state-of-the-art 模型 DiLoco 和 DiPaco，以增强开放协作。一位成员分享了 [Prime Intellect 链接](https://www.primeintellect.ai/)，详细介绍了该平台如何通过全球计算资源的 distributed training 使 AI 民主化。
  
- **Bittensor 利用 The Horde**：用户提到，以分发计算任务闻名的 The Horde 正在 Bittensor 网络上被用于去中心化 AI 模型训练。

- **DeepMind 未参与**：与某些预期相反，讨论澄清了 DeepMind 没有参与社区讨论中特定的进行中项目。

- **关于 Optimizers 的 YouTube 视频**：成员们分享了一个关于 optimizers 的 [YouTube 视频](https://www.youtube.com/watch?v=mdKjMPmcWjY&t=23s)，解释了从 Gradient Descent 到 Adam 的各种类型。它提供了一种简单的方法来记住不同的 optimizers，以便进行有效的模型训练。

- **ChatGPT 的多步响应**：讨论集中在 ChatGPT 如何构建多步响应，澄清了不同的 transformer blocks 可以被分开处理。这引发了人们对 transformer layers 内特定并行化（parallelizations）的兴趣和疑问。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.primeintellect.ai/">Prime Intellect - 计算与智能的商品化</a>：Prime Intellect 大规模地推动 AI 开发的民主化。我们的平台让寻找全球计算资源并通过跨集群的 distributed training 训练 state-of-the-art 模型变得简单。...</li><li><a href="https://www.youtube.com/watch?v=mdKjMPmcWjY&t=23s">Optimizers - 详解！</a>：从 Gradient Descent 到 Adam。这里有一些你应该知道的 optimizers。以及一种记住它们的简单方法。订阅我的频道获取更多好内容！...
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1251845720814653490)** (20 条消息🔥): 

- **关于 AGI 炒作的辩论**：一位用户分享了 Nick Frosst 参与的 [标题为 "Is AGI Just a Fantasy?" 的 YouTube 视频](https://youtu.be/4JF1V2hzGKE?si=gKCvVxBpwsGTD6ow)，引发了关于炒作、真实技术进步和 LLMs 评估的讨论。成员们对“炒作狂人（hype bros）”表示疲劳，但也承认持续投资的重要性，将其比作导致重大创新的互联网泡沫（dot-com bubble）。

- **征集 Next.js App Router 协作**：一位成员宣布创建了一个 GitHub issue，邀请大家协作将 Cohere toolkit UI 迁移到 Next.js App Router，以提高代码的可移植性并吸引更多贡献者。[GitHub issue #219](https://github.com/cohere-ai/cohere-toolkit/issues/219) 包含有关该功能请求的更多详细信息。

- **分享 C4AI 演讲链接**：Nick Frosst 提供了 C4AI 演讲的 [Google Meet 链接](https://meet.google.com/ibt-wsgv-kbq?hs=122&authuser=0)，并引导有问题的成员前往相关的 [Discord 频道](https://discord.gg/yws9vsRe)。

- **对贡献训练数据感兴趣**：一位用户询问关于提交 8,000 份 PDF 用于 Cohere 的 embedding model 训练。Nick Frosst 询问用户是否打算 fine-tune 一个 embedding model，开启了关于潜在数据贡献的讨论。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/4JF1V2hzGKE?si=gKCvVxBpwsGTD6ow">Is AGI Just a Fantasy?</a>：Cohere 联合创始人 Nick Frosst 谈论 LLMs 的未来和 AGI。了解 Cohere 如何通过其新的 AI 模型解决企业的实际问题。...</li><li><a href="https://github.com/cohere-ai/cohere-toolkit/issues/219">是否有计划迁移到 nextjs app router? · Issue #219 · cohere-ai/cohere-toolkit</a>：你希望看到什么功能？嗨，非常棒的 toolkit 和项目，可以快速上手。我想知道是否有计划将其迁移到 app router。大多数（如果不是全部）新的 nextj...</li><li><a href="https://discord.gg/yws9vsRe">加入 Cohere For AI Discord 服务器！</a>：Cohere For AI 的开放科学社区是一个聚集在一起进行机器学习研究协作的空间。 | 3016 名成员
</li>
</ul>

</div>

### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1251909179317162067)** (11 messages🔥): 

- **Cohere 模型免费集成到 Chrome**：一名成员宣布了一个[免费的 Chrome 扩展程序](https://www.asksteve.to)，该程序将 **LLMs 直接集成到浏览器中**，从而消除重复性任务并提高生产力。鼓励用户提供反馈，并可以根据提供的详细说明进行配置。
- **交互式 Collision 地图上线**：另一名成员创建了一个 [Collision 所有活动的交互式地图](https://collision.talewind.ai/)，允许用户按活动详情进行筛选，并访问 AI 聊天以实现更轻松的导航。该项目使用 Sveltekit、Supabase 和 Vercel 构建。
- **Command R+ 配置问题已解决**：一名用户在配置支持 Cohere 的扩展程序中的 Command R+ 时遇到问题，但在建议先使用 Blank Template（空白模板）后得到了解决。开发者承认了该 Bug 并计划进行修复。
- **关于 Cohere 数据提交的咨询**：一名用户询问 **Cohere** 是否接受用于训练的数据提交，特别提到他们拥有近 8,000 份 PDF 用于 embedding 模型训练。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://collision.talewind.ai/">未找到标题</a>：未找到描述</li><li><a href="https://www.asksteve.to">Ask Steve - 在任何网页中解锁 ChatGPT 和 Gemini 的力量！</a>：Ask Steve 为任何网页添加来自 ChatGPT 和 Gemini 的 AI 超能力，让您更好、更快地完成日常任务。免费！</li><li><a href="https://docs.google.com/document/d/1Um99Y4BCfGT4cANYXR7TnLvfIK3h_9lcmx4PQ6uuIns/edit#heading=h.r7p77gz6wtd9">配置 Ask Steve 以使用 Cohere Command R+</a>：配置 Ask Steve 以使用 Cohere Command R+。您需要登录 Ask Steve 才能添加新模型：chrome-extension://gldebcpkoojijledacjeboaehblhfbjg/options.html 登录后，前往...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1252296943137198210)** (1 messages): 

- **David Stewart 将主持 Cohere 开发者办公时间 (Office Hours)**：明天将举行一场轻松的会议，由 Cohere 资深解决方案架构师 David Stewart 主持。鼓励成员在[此贴](https://discord.com/channels/954421988141711382/1168411509542637578/1252294070789869618)中发布他们的问题和议题，以便在活动期间获得优先处理。
- **活动详情发布**：办公时间活动将于 6 月 18 日东部时间下午 1:00 举行。点击[此处](https://discord.gg/jy5XFg5GDH?event=1248300905703673987)加入活动，进行实时互动并获取有关 Cohere API 和模型相关查询的指导。

**提到的链接**：<a href="https://discord.gg/jy5XFg5GDH?event=1248300905703673987">加入 Cohere 社区 Discord 服务器！</a>：Cohere 社区服务器。来这里聊聊 Cohere API、LLMs、生成式 AI 以及介于两者之间的一切。| 17098 名成员

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1251269680770056264)** (14 messages🔥): 

- **模型在代码编写中途冻结**：一名成员询问其他人是否遇到模型在编写代码中途冻结的情况。另一名成员回答说，即使看起来冻结了，它通常也会完成任务。
- **Windows 安装问题**：一名用户报告了在 Windows 上安装和运行模型的问题。建议他们搜索帮助并在指定频道发布查询。
- **记忆功能改进**：一名成员对以“非常原始的方式”实现记忆功能表示满意。他们热情地与社区分享了进展。
- **Llama 3 性能评测**：分享了一份详细的 [Llama 3 模型对比和性能测试](https://www.reddit.com/r/LocalLLaMA/comments/1cal17l/llm_comparisontest_llama_3_instruct_70b_8b/)，承诺对 Llama 3 Instruct 在各种格式和量化级别下的能力进行全面评估。
- **Profiles 功能特性**：重点介绍了 Open Interpreter 上新的 [“profiles” 功能](https://x.com/tx_smitht/status/1802156524833546553)。一名成员分享了一个[视频](https://youtu.be/ZQtSN7SAsYM)来解释其功能和应用。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/tx_smitht/status/1802156524833546553">Thomas Smith (@tx_smitht) 的推文</a>：如果你还不知道 Open Interpreter 新的 “profiles” 功能，你一定要去看看！它让你能够扩展 OI 的能力。这就像上传一组特定的...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cal17l/llm_comparisontest_llama_3_instruct_70b_8b/">Reddit - 深入探索一切</a>：未找到描述
</li>
</ul>

</div>
  

---

### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1251502290519593103)** (4 条消息): 

- **在置顶消息中检查您的设备到货情况**：一位用户询问如何检查他们的设备何时送达，并提到他们很早就下单了。另一位成员引导他们查看频道中的置顶消息，以获取制造更新和时间表。 
- **讨论 Vector DB、Semantic Search 和 LLM 的组合**：有人提出了将音频 Vector DB 与基于语音的 Semantic Search 和索引相结合，并配合一个能够访问这些数据并执行操作的 LLM 的潜力。这种提议的组合暗示了一个基于语音输入执行操作的强大工具。
  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1251532212449443880)** (6 条消息): 

- **DIY AI Cyber Hat 引人注目**：一位成员分享了他们制作开源 AI 增强型可穿戴帽子的项目，将其比作智能眼镜。他们提供了一个视频预览并表示欢迎合作，[在此查看视频](https://www.youtube.com/watch?v=71p9DcGqNDc)。
- **关于帽子设计的终结者幽默**：一位成员幽默地评论说，这顶帽子的设计让创作者看起来像是一个被派来消灭 Hobby Lobby 创始人的终结者。
- **对科幻可穿戴设备的兴趣引发互动**：人们对这个 AI 帽子项目表现出极大的热情，并请求在代码清理后获取源代码。创作者建议未来可能会集成更多传感器用于科学实验。
- **Pi Zero 将应用于 Big Mouth Billy Bass**：同一位创作者预告了他们的下一个项目，涉及将 Pi Zero 集成到 Big Mouth Billy Bass 中。
- **Dream Machine 引发热议**：一位成员分享了 [Dream Machine](https://lumalabs.ai/dream-machine)，这是一个可以根据文本和图像创建高质量、逼真视频的 AI 模型。该模型旨在构建一个通用的想象力引擎，目前已向公众开放。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lumalabs.ai/dream-machine">Luma Dream Machine</a>: Dream Machine 是来自 Luma AI 的一款 AI 模型，可以快速根据文本和图像生成高质量、逼真的视频</li><li><a href="https://www.youtube.com/watch?v=71p9DcGqNDc">I Made My Own Custom AI Cyber Hat</a>: 这是一个关于我一个名为 &quot;heddy&quot; 的项目（至少是帽子部分）开始的视频。我主要通过...创建了自己的智能 AI 增强型帽子。
</li>
</ul>

</div>
  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1251292718609596446)** (7 条消息): 

- **Torchtune 目前专注于单节点**：当被问及 Torchtune 是否计划发布多节点训练时，一位成员澄清说目前的重点是单节点训练。不过，他们指出“我们的 `tune run` 命令是 `torch run` 的封装”，只需进行少量更改，多节点设置即可工作，尽管尚未经过测试。
  
- **多节点训练的分布式配置调整**：成员们交流了在 Torchtune 中设置多节点训练的技巧。有人建议设置 `tune run —nnodes 2`，而另一位成员提到需要 `TorchX` 或 `slurm` 来处理脚本启动和特定节点间的通信，并指向了 [TorchX](https://github.com/pytorch/torchx) 和 [混合分片策略文档](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardingStrategy) 等资源。

**提到的链接**: <a href="https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardingStrategy)">FullyShardedDataParallel &mdash; PyTorch 2.3 文档</a>: 未找到描述

  

---



### **DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1251542470358073416)** (5 条消息): 

- **Llama3 Tokenizer 保持不变**：成员们讨论了 **Llama3 Tokenizer** 是否针对德语模型进行了扩展。一位成员确认 *“Tokenizer 与基础 Llama3 相同”*。

- **对德语 Token 处理的担忧**：一位成员质疑不扩展 Tokenizer 的理由，指出 *不包含德语 Token 可能会大幅减少上下文窗口*。他们很好奇自己是否遗漏了某些理由，特别是考虑到 Embedding 可能的增加。

- **与 Llama2 的尺寸比较**：另一位成员指出 **Llama3 的 Tokenizer 比 Llama2 大 4 倍**。他们询问它在德语上是否已经更有效，或者是否仍然存在问题。

### **Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1251252730811645984)** (3 messages): 

- **AI 讨论中的替代立场受到称赞**：一位成员赞赏了另一位成员的文章，称其 *"以诚意对待替代立场"*。他们幽默地指出，ChatGPT 的兴起是 *"数据工程师的永久就业法案"*。

- **Thoughtbot 的 LLM 指南推荐**：一位成员为 LLM 初学者推荐了一个有用的 [thoughtbot 资源](https://thoughtbot.com/blog/understanding-open-source-llms)。他们建议阅读 [Jose Blanco 的文章](https://thoughtbot.com/blog/how-to-use-open-source-LLM-model-locally)，了解如何在本地和远程使用开源 LLM。

- **LLM 命名规范的清晰度受到赞赏**：另一位成员认为将 LLM 分为 Base、Instruct 和 Chat 模型的分类方式特别清晰且详尽。

**提到的链接**：<a href="https://thoughtbot.com/blog/understanding-open-source-llms">Understanding open source LLMs</a>：你认为你可以在自己的机器上运行任何大语言模型（LLM）吗？

  

---


### **Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1251736192769331221)** (1 messages): 

- **Turso 添加原生向量搜索支持**：Turso 在其平台中引入了 [原生向量搜索](https://turso.tech/blog/turso-brings-native-vector-search-to-sqlite) 功能，补充了 SQLite 现有的特性。这一新功能旨在为构建 AI 产品的用户简化向量搜索，解决了之前管理 [sqlite-vss](https://github.com/asg017/sqlite-vss) 等扩展时的挑战。

**提到的链接**：<a href="https://turso.tech/blog/turso-brings-native-vector-search-to-sqlite">Turso brings Native Vector Search to SQLite</a>：向量相似度搜索现已可用！

  

---



### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/)** (1 messages): 

gomiez: 有人知道那个医院 AI Town 项目的名字吗？
  

---



### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/)** (1 messages): 

cryovolcano.: 我们可以在 Firefox 中将 llamafile 与 tinyllama 结合用作搜索引擎吗？
  

---



---



---



---



{% else %}


> 完整的频道逐项分析已针对电子邮件进行了截断。
> 
> 如果您想查看完整分析，请访问此电子邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}