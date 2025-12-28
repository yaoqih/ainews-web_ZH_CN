---
companies:
- groq
- cerebras
- cursor
- google-deepmind
- anthropic
date: '2024-08-29T00:59:27.113773Z'
description: '**Groq** 在 2024 年初以极快的大语言模型（LLM）推理速度领跑，Mixtral 8x7B 达到了约 450 tokens/秒，Llama
  2 70B 达到了 240 tokens/秒。**Cursor** 推出了一款专门的代码编辑模型，速度高达 1000 tokens/秒。现在，**Cerebras**
  声称凭借其晶圆级芯片实现了最快的推理速度，在全精度下运行 **Llama3.1-8b** 的速度为 1800 tokens/秒，**Llama3.1-70B**
  为 450 tokens/秒，同时拥有极具竞争力的定价和慷慨的免费层级。**谷歌的 Gemini 1.5** 模型在基准测试中展现了显著的提升，尤其是 Gemini-1.5-Flash
  和 Gemini-1.5-Pro。针对消费级硬件优化的新型开源模型如 **CogVideoX-5B** 和 **Mamba-2 (Rene 1.3B)** 也已发布。**Anthropic
  的 Claude** 现在支持提示词缓存（prompt caching），提升了速度和成本效益。*“Cerebras 推理运行 Llama3.1 的速度比 GPU
  解决方案快 20 倍，而价格仅为其五分之一。”*'
id: df31451f-7460-4ef7-b063-1b3d31d35d59
models:
- llama-3.1-8b
- llama-3.1-70b
- gemini-1.5-flash
- gemini-1.5-pro
- cogvideox-5b
- mamba-2
- rene-1.3b
- llama-3.1
- gemini-1.5
- claude
original_slug: ainews-cerebras-inference-faster-better-and
people:
- jeremyphoward
- sam-altman
- nat-friedman
- daniel-gross
- swyx
title: Cerebras 推理：更快、更好、且更便宜
topics:
- inference-speed
- wafer-scale-chips
- prompt-caching
- model-merging
- benchmarking
- open-source-models
- code-editing
- model-optimization
---

<!-- buttondown-editor-mode: plaintext -->**Wafer-scale engines are all you need.**

> 2024/8/27-8/28 的 AI 新闻。我们为您检查了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord (包含 **215** 个频道和 **2366** 条消息)。预计节省阅读时间（按 200wpm 计算）：**239 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

2024 年超快 LLM 推理简史：

- Groq 在 2 月份占据了新闻头条（这里有很多[零散的讨论](https://buttondown.com/ainews/archive/ainews-karpathy-emerges-from-stealth/#latent-space-discord-summary)），它为 Mixtral 8x7B 实现了 [~450 tok/s](https://www.semianalysis.com/p/groq-inference-tokenomics-speed-but)（为 Llama 2 70b 实现了 [240 tok/s](https://groq.com/news_press/groq-lpu-inference-engine-leads-in-first-independent-llm-benchmark/)）。
- 5 月，Cursor [宣传了一款专门的代码编辑模型](https://www.cursor.com/blog/instant-apply)（与 Fireworks 合作开发），其速度达到了 1000 tok/s。

现在终于轮到 Cerebras 大放异彩了。全新的 [Cerebras Inference 服务](https://x.com/CerebrasSystems/status/1828464491677524311) 宣称其 Llama3.1-8b 在 **全精度** 下速度达到 1800 tok/s，价格为 $0.10/mtok；Llama3.1-70B 速度为 450 tokens/s，价格为 $0.60/mtok。毋庸置疑，Cerebras 的全精度定价及其无与伦比的速度，使其突然成为该市场中一个强有力的竞争者。引用他们的营销口号："**Cerebras Inference 运行 Llama3.1 的速度比 GPU 解决方案快 20 倍，而价格仅为 1/5。**" —— 从技术上讲这并不完全准确 —— 大多数推理提供商如 [Together](https://www.together.ai/blog/together-inference-engine-2) 和 [Fireworks](https://fireworks.ai/blog/fireworks-quantization) 倾向于引导用户使用其服务的量化版本，其中 FP8 70B 定价为 $0.88/mtok，INT4 70B 定价为 $0.54。虽然 Cerebras 确实更好，但并没有便宜 5 倍，也没有快 20 倍。

 
![image.png](https://assets.buttondown.email/images/bd520ebb-7aa6-46e6-834d-1af911e2d956.png?w=960&fit=max)
 

> 注：还应注意到他们非常慷慨的免费额度，即 **每天 100 万个免费 token**。

秘密当然在于 Cerebras 的 wafer-scale 芯片（不然你还指望他们说什么呢？）。类似于 Groq 的 LPU 论点，Cerebras 表示将整个模型放入 SRAM 是关键：

 
![image.png](https://assets.buttondown.email/images/2256c2c9-d785-40b4-a851-506531290953.png?w=960&fit=max)
 

该你们出招了，Groq 和 Sambanova。

---

**今日赞助商：Solaris**

Solaris 是位于旧金山的 **早期 AI 初创公司办公室**，现有新的工位和办公室开放！它曾是受 Nat Friedman、Daniel Gross、Sam Altman、YC 等支持的创始人们的总部。**

> Swyx 的评论：我过去 9 个月一直在这里，非常喜欢。如果你正在寻找一个高质量的地方来构建下一个伟大的 AI 初创公司，[请在此与创始人预约时间](https://calendly.com/d/ck2k-955-yqz/solaris-ai-introduction-chat)，并告诉他们是我们推荐的。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录** 和 **频道摘要** 已移至此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。


**AI 模型更新与基准测试**

- **Gemini 1.5 性能**：Google 最新的 Gemini 1.5 模型（Pro/Flash/Flash-9b）在基准测试中表现出显著提升，其中 Gemini-1.5-Flash 从总榜第 23 位跃升至第 6 位。新的 Gemini-1.5-Pro 在编程和数学任务中表现出强劲增长。[@lmsysorg](https://twitter.com/lmsysorg/status/1828506835370065994) 分享了来自 2 万多名社区投票的详细结果。

- **开源模型**：发布了新的开源模型，包括用于文本生成视频的 CogVideoX-5B，可在不到 10GB 的 VRAM 上运行。[@_akhaliq](https://twitter.com/_akhaliq/status/1828429991664594976) 强调了其高质量和高效率。Rene 1.3B，一个 Mamba-2 语言模型，也已发布，在消费级硬件上表现出色。[@awnihannun](https://twitter.com/awnihannun/status/1828513780298588572) 指出其在 M2 Ultra 上的速度接近 200 tokens/sec。

- **Cerebras 推理**：Cerebras 宣布了一个新的推理 API，声称是 Llama 3.1 模型中最快的，8B 模型速度达到 1,800 tokens/sec，70B 模型速度达到 450 tokens/sec。[@AIatMeta](https://twitter.com/AIatMeta/status/1828473483820704233) 验证了这些令人印象深刻的性能数据。

**AI 开发与基础设施**

- **Prompt Caching**：Jeremy Howard 强调了 Prompt Caching 对于提高性能和降低成本的重要性。[@jeremyphoward](https://twitter.com/jeremyphoward/status/1828460632972366089) 指出 Anthropic 的 Claude 现在支持缓存，缓存后的 token 价格便宜 90% 且速度更快。

- **模型合并 (Model Merging)**：分享了一份全面的模型合并技术时间线，追溯了从 90 年代的早期工作到近期在 LLM 对齐和专业化中应用的发展历程。[@cwolferesearch](https://twitter.com/cwolferesearch/status/1828567528710513141) 提供了各个阶段和方法的详细概述。

- **分布式训练**：讨论了分布式社区机器学习（ML）训练的潜力，认为下一个开源 GPT-5 可能会由数百万贡献少量 GPU 算力的人共同构建。[@osanseviero](https://twitter.com/osanseviero/status/1828363215325044870) 概述了该领域的最新突破和未来可能性。

**AI 应用与工具**

- **Claude Artifacts**：Anthropic 向所有 Claude 用户开放了 Artifacts 功能，包括 iOS 和 Android 应用。[@AnthropicAI](https://twitter.com/AnthropicAI/status/1828462522468372600) 分享了该功能的开发过程和广泛采用的情况。

- **AI 驱动的应用**：强调了由 LLM 实时创建移动应用的潜力，并展示了使用 Claude 复制简单游戏的示例。[@alexalbert__](https://twitter.com/alexalbert__/status/1828502920788103363) 演示了这一能力。

- **基于 LLM 的搜索引擎**：提到了一个用于 Web 搜索引擎的多智能体（multi-agent）框架，类似于 Perplexity Pro 和 SearchGPT。[@dl_weekly](https://twitter.com/dl_weekly/status/1828444869473268105) 分享了关于此话题更多信息的链接。

**AI 伦理与监管**

- **AI 监管辩论**：关于 AI 监管的讨论仍在继续，一些人认为支持 AI 监管并不一定意味着支持每一项提议的法案。[@AmandaAskell](https://twitter.com/AmandaAskell/status/1828331638453084583) 强调了良好初始监管的重要性。

- **OpenAI 的动向**：有关 OpenAI 开发名为 "Strawberry" 的强大推理模型以及 "Orion"（GPT-6）计划的报告，引发了关于公司战略及其对竞争潜在影响的讨论。[@bindureddy](https://twitter.com/bindureddy/status/1828450988958851448) 分享了对这些进展的见解。

**其他 AI 见解**

- **AI 中的微交易**：Andrej Karpathy 提出，启用极小额交易（例如 5 美分）可以释放巨大的经济潜力，并改善数字经济中的价值流动。[@karpathy](https://twitter.com/karpathy/status/1828530326613958965) 认为这可能会带来更高效的商业模式和积极的二阶效应。

- **AI 认知研究**：强调了研究 AI 认知而非仅仅研究行为对于理解 AI 系统泛化的重要性。[@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1828538139285299457) 将其类比为心理学中从行为主义的转变。


---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. 开源文本生成视频 AI：CogVideoX 5B 突破**

- **CogVideoX 5B - 开源权重文本生成视频 AI 模型（运行显存低于 10GB VRAM）| 清华 KEG (THUDM)** ([Score: 91, Comments: 13](https://reddit.com//r/LocalLLaMA/comments/1f2gaqt/cogvideox_5b_open_weights_text_to_video_ai_model/))：**CogVideoX 5B** 是由 **清华 KEG (THUDM)** 开发的开源权重文本生成视频 AI 模型，可在低于 **10GB VRAM** 的显存下运行，其中 **2B 模型** 可在 **1080TI** 上运行，**5B 模型** 可在 **3060 GPU** 上运行。该模型系列（包括以 **Apache 2.0 协议** 发布的 **2B 版本**）已在 [Hugging Face](https://huggingface.co/collections/THUDM/cogvideo-66c08e62f1685a3ade464cce) 上线，并附带了 [演示空间 (demo space)](https://huggingface.co/spaces/THUDM/CogVideoX-5B-Space) 和 [研究论文](https://huggingface.co/papers/2408.06072)。

**主题 2. 高效 AI 模型进展：Gemini 1.5 Flash 8B**

- **[Gemini 1.5 Flash 8b,](https://www.unite.ai/google-releases-three-new-experimental-gemini-models/)** ([Score: 95, Comments: 24](https://reddit.com//r/LocalLLaMA/comments/1f2zqwb/gemini_15_flash_8b/))：Google 发布了 **Gemini 1.5 Flash 8B**，这是一款新型小型 AI 模型，尽管其参数量仅为 **80 亿 (8 billion parameters)**，但展现出了令人印象深刻的能力。该模型在多项基准测试中达到了 **SOTA (state-of-the-art)** 性能，包括在某些任务上超越了像 **Llama 2 70B** 这样的大型模型，同时在推理速度和资源需求方面显著更高效。
  - **Gemini 1.5 Flash 8B** 最初在 **6 月份** 的 [第三版 Gemini 1.5 论文](https://arxiv.org/pdf/2403.05530v3) 中被讨论。新版本很可能是原始实验的改进模型，具有更好的基准测试表现。
  - Google 公开 **80 亿参数量** 的做法受到了称赞。目前关于 Google 是否会发布权重存在猜测，但由于 **Gemini 模型** 通常是闭源的（不同于开源的 **Gemma 模型**），这被认为不太可能。
  - 关于 Google 在 Gemini 中使用 **标准 Transformer (standard transformers)** 的讨论也随之而起，这让一些期待自定义架构的用户感到惊讶。该模型的性能引发了与 **GPT-4o-mini** 的对比，暗示了在参数效率方面的潜在进步。

## 全球 AI Reddit 综述

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 模型进展与发布**

- **Google DeepMind 的 GameNGen**：一个由神经网络驱动的游戏引擎，能够以高质量的视觉效果实时交互式地模拟经典游戏 DOOM。这展示了 AI 生成交互式游戏环境的潜力。[来源](https://www.reddit.com/r/singularity/comments/1f3055r/google_deepmind_we_present_gamengen_the_first/)

- **OpenAI 的 "Strawberry" AI**：据报道正准备最早于 2024 年秋季发布。OpenAI 已向国家安全官员展示了该 AI，并正利用它开发另一个名为 "Orion" 的系统。关于其能力的细节目前有限。[来源 1](https://www.reddit.com/r/singularity/comments/1f2hpz1/openai_reportedly_looking_to_launch_strawberry_as/), [来源 2](https://www.reddit.com/r/singularity/comments/1f2iism/openai_shows_strawberry_ai_to_the_feds_and_uses/)

- **Google 的 Gemini 1.5 更新**：Google 推出了 Gemini 1.5 Flash-8B，这是一个改进版的 Gemini 1.5 Pro，具有更强的代码编写和复杂提示词（Prompt）处理能力，以及增强的 Gemini 1.5 Flash 模型。[来源](https://www.reddit.com/r/singularity/comments/1f2mjyg/google_rolls_out_gemini_15_flash8b_stronger/)

**图像生成与处理中的 AI**

- **Flux AI 模型**：一种新型的图像生成 AI 模型，因其创建逼真图像的能力而迅速走红。用户正在尝试在个人照片上训练自定义 LoRA 模型，以生成高度逼真的 AI 自画像。[来源 1](https://www.reddit.com/r/StableDiffusion/comments/1f2yun6/i_am_using_my_generated_photos_from_flux_on/), [来源 2](https://www.reddit.com/r/StableDiffusion/comments/1f2az8r/a_little_observation_on_the_release_of_flux/)

**机器人与具身智能 (Physical AI)**

- **Galbot G1**：由中国初创公司 Galbot 开发的第一代机器人，旨在执行可泛化的长时间任务。具体能力的细节目前有限。[来源](https://www.reddit.com/r/singularity/comments/1f2ilqg/meet_galbot_g1_the_1stgeneration_robot_by_chinese/)

**科学突破**

- **DNA 损伤修复蛋白**：科学家发现了一种名为 DNA 损伤响应蛋白 C (DdrC) 的蛋白质，它可以直接阻止 DNA 损伤。它似乎是“即插即用”的，可能在任何生物体中发挥作用，使其成为癌症预防研究中极具前景的候选者。[来源](https://www.reddit.com/r/singularity/comments/1f2bazr/scientists_have_discovered_a_protein_that_can/)

**AI 伦理与社会影响**

- **媒体中的 AI 生成内容**：围绕社交媒体和娱乐行业中 AI 生成内容日益普及的讨论，引发了关于真实性以及创意产业未来的疑问。[来源](https://www.reddit.com/r/singularity/comments/1f2qt5s/the_comic_book_industry_wants_a_100_ban_on_ai_too/)


---

# AI Discord 摘要回顾

> 由 GPT4O (gpt-4o-2024-05-13) 生成的摘要之摘要的总结


**1. LLM 进展与基准测试**

- **Llama 3.1 API 提供免费访问**：**[Sambanova.ai](https://sambanova.ai/fast-api?api_ref=444868)** 提供了一个免费且有速率限制的 API，用于运行 Llama 3.1 405B、70B 和 8B 模型。该 API 与 OpenAI 兼容，并允许用户使用自己微调的模型。
  - `@user` 分享道，该 API 提供了入门套件和社区支持，以帮助加速开发。
- **Google 发布新款 Gemini 模型**：Google 宣布了三个实验性模型：**[Gemini 1.5 Flash-8B](https://aistudio.google.com)**、**Gemini 1.5 Pro** 以及改进后的 **Gemini 1.5 Flash**。
  - `@OfficialLoganK` 强调，**Gemini 1.5 Pro** 模型在编程和处理复杂提示词方面表现尤为出色。


**2. 模型性能优化与基准测试**

- **OpenRouter 支持 DeepSeek 缓存**：**[OpenRouter](https://platform.deepseek.com/api-docs)** 正在增加对 **DeepSeek 上下文缓存 (context caching)** 的支持，预计可降低高达 90% 的 API 成本。
  - `@user` 分享了关于这一即将推出的功能的信息，旨在优化 API 的成本效率。
- **Hyperbolic 的 BF16 Llama 405B**：Hyperbolic 发布了 **[Llama 3.1 405B 基础模型](https://x.com/hyperbolic_labs/status/1828481468156518691)** 的 **BF16** 变体，补充了 OpenRouter 上现有的 FP8 量化版本。
  - `@hyperbolic_labs` 在推特上发布了关于该新变体的消息，强调了其在实现更高效模型性能方面的潜力。


**3. 开源 AI 发展与协作**

- **IBM 的 Power Scheduler**：IBM 推出了一种名为 **[Power Scheduler](https://x.com/_akhaliq/status/1828267147765702856?s=46)** 的新型学习率调度器，它与 Batch Size 和训练 Token 数量无关。
  - `@_akhaliq` 发推称，该调度器在各种模型大小和架构中始终能取得令人印象深刻的性能。
- **用于实时 AI 的 Daily Bots**：**[Daily Bots](https://x.com/i/status/1825946246886076785)** 推出了支持 RTVI 标准的语音、视觉和视频 AI 超低延迟云服务。
  - `@trydaily` 强调，该平台结合了实时 AI 应用的最佳工具，包括与 LLM 的语音对语音交互。


**4. 多模态 AI 与生成式建模创新**

- **GameNGen：神经游戏引擎**：**[GameNGen](https://gamengen.github.io/)** 是首个完全由神经模型驱动的游戏引擎，能够实现与复杂环境的实时交互。
  - `@user` 分享道，GameNGen 可以在单个 TPU 上以超过每秒 20 帧的速度模拟 **DOOM**，其 PSNR 达到 29.4，与有损 JPEG 压缩相当。
- **Artifacts 登陆 iOS 和 Android**：Anthropic 的项目 **[Artifacts](https://x.com/alexalbert__/status/1828502920788103363?s=46)** 已在 iOS 和 Android 上线，允许用户使用 Claude 实时创建简单的游戏。
  - `@alexalbert__` 强调了这次移动端发布的意义，将 LLM 的能力带到了移动应用中。


**5. 微调挑战与提示工程策略**

- **Unsloth 的持续预训练**：Unsloth 的 [Continued Pretraining](https://unsloth.ai/blog/contpretraining) 功能允许以比 Hugging Face + Flash Attention 2 QLoRA 快 2 倍的速度进行 LLM 预训练，且节省 50% 的 VRAM。
  - `@unsloth` 分享了一个 [Colab notebook](https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing)，用于持续预训练 Mistral v0.3 7b 以学习韩语。
- **使用合成数据进行微调**：在模型微调中使用合成数据的趋势日益增强，**Hermes 3** 等案例凸显了这一点。
  - 一位用户提到，合成数据训练需要复杂的过滤流水线，但这种方式正变得越来越流行。

---

# 第一部分：Discord 高层级摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **游戏化家庭训练？**：一位成员提议了一个“游戏化家庭训练”基准测试工具，声称是为了其求职申请。
- **Triton 配置难题**：一位成员遇到了在使用 llama3 instruct 配合 Triton 以及 tensorrt-llm 或 vllm 后端时，响应生成无法停止的问题。
   - 直接使用 vllm 托管运行完美，这表明其 Triton 配置可能存在问题。
- **Loss 为 0.0 - 日志错误？**：讨论集中在模型训练中“Loss 曲线”的重要性。
   - 一位成员建议 Loss 为 0.0 可能表示日志错误，并质疑由于舍入原因，Loss 达到绝对 0.0 的完美模型的可行性。
- **在 AMD 上微调 Gemma2b - 实验性的挣扎**：一位成员在 AMD 上微调 Gemma2b 模型时遇到困难，将其归因于潜在的日志错误。
   - 其他成员指出 ROCm 的实验性质是导致这些困难的一个因素。
- **模型合并策略：UltraChat 和 Mistral**：一位成员提议将 UltraChat 与基础 Mistral 之间的差异应用到 Mistral-Yarn 上，作为一种模型合并策略。
   - 虽然有些人表示怀疑，但该成员保持乐观，并引用了过去在“被诅咒的模型合并 (cursed model merging)”方面的成功经验。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **VLLM 在 Kaggle 上可以运行了！**：一位用户报告称，使用来自[此数据集](https://www.kaggle.com/datasets/sethmoudry/arc-vllm-wheels)的 **wheels** 成功在 Kaggle 上运行了 **VLLM**。
   - 这是使用 **VLLM 0.5.4** 实现的，该版本被认为相对较新，因为 **0.5.5** 虽然已经发布但尚未广泛使用。
- **Mistral 难以扩展超过 8k**：成员们确认，如果不进行持续预训练，**Mistral** 无法扩展到 8k 以上，且[这是一个已知问题](https://link.to.issue)。
   - 他们还讨论了未来性能增强的潜在途径，包括 **mergekit** 和 **frankenMoE 微调**。
- **同构 AI (Homoiconic AI)：权重即代码？**：一位成员分享了关于 [“Homoiconic AI” 的进度报告](https://x.com/neurallambda/status/1828214178567647584?s=46)，该项目使用 **hypernet** 生成 **autoencoder 权重**，然后通过 In-context learning 改进这些权重。
   - 报告指出，这种“代码即数据 & 数据即代码”的方法可能是推理所必需的，甚至与推理同构。
- **Unsloth 的持续预训练能力**：一位成员分享了 Unsloth 关于[持续预训练的博客文章](https://unsloth.ai/blog/contpretraining)的链接，强调其持续预训练 LLM 的速度比 Hugging Face + Flash Attention 2 QLoRA 快 **2 倍**，且显存占用减少 **50%**。
   - 博客文章还提到使用 [Colab notebook](https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing) 对 Mistral v0.3 7b 进行持续预训练以**学习韩语**。
- **Unsloth vs OpenRLHF：速度与显存效率**：一位用户询问了 Unsloth 和 OpenRLHF 之间的区别，特别是关于它们对微调非量化模型的支持。
   - 一位成员确认 Unsloth 支持非量化模型，并计划很快增加 8bit 支持，强调其与其他微调方法相比，速度显著更快且显存占用更低。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 0.54.0：Gemini 模型与 Shell 命令改进**：最新版本的 **Aider (v0.54.0)** 引入了对 **`gemini/gemini-1.5-pro-exp-0827`** 和 **`gemini/gemini-1.5-flash-exp-0827`** 模型的支持，并增强了 shell 和 `/run` 命令，现在允许在带有 pty 的环境中进行交互式执行。
   - 新开关 **`--[no-]suggest-shell-commands`** 允许自定义配置 shell 命令建议，同时大型项目和 monorepo 项目中改进的自动补全功能提升了 Aider 的性能。
- **Aider 自动化其自身开发**：Aider 在其自身的开发中发挥了重要作用，为本次发布贡献了 **64%** 的代码。
   - 此版本还引入了 **`--upgrade`** 开关，以便从 PyPI 轻松安装最新的 Aider 版本。
- **Gemini 1.5 Pro 基准测试结果喜忧参半**：分享了新 **Gemini 1.5 Pro** 模型的基准测试结果，显示全量编辑格式（whole edit format）的通过率为 **23.3%**，差异编辑格式（diff edit format）的通过率为 **57.9%**。
   - 基准测试是使用 Aider 配合 `gemini/gemini-1.5-pro-exp-0827` 模型以及 `aider --model gemini/gemini-1.5-pro-exp-0827` 命令运行的。
- **GameNGen：首个神经游戏引擎**：论文介绍了 **GameNGen**，这是第一个完全由神经模型驱动的游戏引擎，能够在长轨迹上高质量地实时模拟复杂环境。
   - 该模型可以在单个 TPU 上以每秒超过 20 帧的速度交互式模拟经典游戏 **DOOM**，实现的 PSNR 为 29.4，与有损 JPEG 压缩相当。
- **OpenRouter：Discord 的替代方案？**：一位成员询问 **OpenRouter** 是否与 **Discord** 相同。
   - 另一位成员确认这两个服务对他们来说都运行良好，并引用了 OpenRouter 的状态页面：[https://status.openrouter.ai/](https://status.openrouter.ai/) 作为参考。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 0.3.1 发布**：**LM Studio** 的最新版本为 **v0.3.1**，可在 [lmstudio.aido](https://lmstudio.aido) 获取。
- **LM Studio 在 Linux 上的问题**：一位用户报告称，在没有 `--no-sandbox` 标志的情况下通过 **Steam** 运行 **Linux** 版本的 **LM Studio** 会导致 **SSD 损坏**。
- **不支持 Snapdragon NPU**：一位用户确认 **Snapdragon** 上的 **NPU** 在 **LM Studio** 中无法工作，尽管他们已经在 **Snapdragon** 上安装了 **LM Studio**。
- **LM Studio 的 AMD GPU 支持**：**LM Studio 的 ROCM 版本**目前仅支持最高端的 **AMD GPU**，不支持像 **6700XT** 这样的 **GPU**，从而导致兼容性问题。
- **LM Studio 的安全性测试**：一位用户通过提示 **LLM** 下载程序来测试 **LM Studio 的安全性**，结果得到了幻觉响应，表明实际上并没有发生下载。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 用户陷入上传困境**：用户在上传图片和文件时遇到问题，一些用户尽管在某些浏览器中仍能访问，但丢失了 Pro 订阅状态。
   - 这个问题引起了 Pro 用户的沮丧，特别是由于缺乏修复的信息和预计时间表，导致出现了一些幽默的回应，比如“this is fine”的 GIF 图。
- **Claude 3.5 的每日消息限制：430 条**：Claude 3.5 和其他 Pro 模型受每日 430 条消息的限制，但 Opus 除外，其限制为 50 条。
   - 虽然有些用户还没达到总限制，但许多人发现他们最接近的一次大约是 250 条消息。
- **图片上传问题——归咎于 AWS Rekognition**：无法上传图片归因于达到了 Cloudinary（一种用于图像和视频管理的服务）上的 AWS Rekognition 限制。
   - Perplexity 目前正在努力解决此问题，但尚无修复的预计时间表。
- **Perplexity 搜索：比 ChatGPT 更好？有待商榷**：一些用户声称 Perplexity 的搜索（尤其是 Pro 版）优于其他平台，理由是更好的来源引用和更少的幻觉。
   - 然而，其他人认为 ChatGPT 的自定义选项、RAG 和聊天 UX 更先进，而 Perplexity 的搜索较慢且功能较少，特别是在与 ChatGPT 的文件处理和对话上下文相比时。
- **Perplexity 的特定领域搜索 Chrome 扩展程序**：Perplexity Chrome 扩展程序提供特定领域的搜索功能，允许用户在特定网站内查找信息而无需手动搜索。
   - 这一功能因其在寻找特定域名或网站信息方面的优势而受到一些用户的称赞。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DisTrO vs. SWARM：效率考量**：虽然 DisTrO 在 DDP 下效率极高，但对于在大量弱设备上训练超大型 LLM（超过 100B 参数）的情况，SWARM 可能是更合适的选择。
   - 有成员询问 DisTrO 是否可用于训练这些大型 LLM，并建议了一个使用 10 亿部旧手机、笔记本电脑和台式机的用例，但另一位成员推荐了 SWARM。
- **DPO 训练与 AI 预测响应：探索心智理论 (Theory of Mind)**：一位成员思考了在 DPO 训练中使用 AI 模型预测的用户响应（而非真实响应）的潜在影响。
   - 他们认为这种方法可能会提升模型的心智理论 (Theory of Mind) 能力。
- **模型合并：一种有争议的策略**：一位成员提议将 UltraChat 和 Mistral 基础模型之间的差异合并到 Mistral-Yarn 中作为一种潜在策略，并引用了以往类似成功的案例。
   - 尽管其他人表示怀疑，该成员仍对这种“诅咒式模型合并” (cursed model merging) 技术的有效性保持乐观。
- **Hermes 3 与 Llama 3.1：正面交锋**：一位成员分享了 Hermes 3 和 Llama 3.1 的对比，强调了 Hermes 3 在通用能力方面具有竞争力的表现，甚至更胜一筹。
   - 提供了基准测试链接，展示了 Hermes 3 相对于 Llama 3.1 的优缺点。
- **使用合成数据进行微调：训练的未来？**：成员们讨论了在微调模型中使用合成数据的趋势，并以 Hermes 3 和传闻中的“草莓模型” (strawberry models) 为例。
   - 虽然并非总是推荐，但合成数据训练已势头强劲，尤其是在 Hermes 3 等模型中，但这需要复杂的过滤流水线。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter API 短暂降级**：OpenRouter 经历了 5 分钟的 API 性能下降，但随后发布了补丁，目前已恢复正常。
- **Llama 3.1 405B BF16 端点上线**：[Llama 3.1 405B (base)](https://openrouter.ai/models/meta-llama/llama-3.1-405b) 已更新 BF16 端点。
- **Hyperbolic 部署 BF16 Llama 405B Base**：Hyperbolic 发布了 **Llama 3.1 405B 基础模型** 的 **BF16** 变体。
   - 这是对 **OpenRouter** 上现有 **FP8 量化版本** 的补充。
- **LMSys 排行榜相关性受到质疑**：一位用户质疑 **LMSys 排行榜** 的相关性，认为它可能已经过时。
   - 他们指出像 **Gemini Flash** 这样的新模型表现异常出色。
- **OpenRouter DeepSeek 缓存即将推出**：OpenRouter 正在努力增加对 **DeepSeek 上下文缓存 (context caching)** 的支持。
   - 该功能预计可降低高达 **90%** 的 API 成本。



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **免费 Llama 3.1 405B API**：一位成员分享了 [Sambanova.ai](https://sambanova.ai/fast-api?api_ref=444868) 的链接，该网站提供运行 Llama 3.1 405B、70B 和 8B 的免费、限速 API。
   - 该 API 兼容 OpenAI，允许用户上传自己的微调模型，并提供入门套件和社区支持以加速开发。
- **TRL.X 已弃用**：一位成员指出 TRL.X 已严重过时，且很长时间没有更新。
   - 另一位成员询问它是否仍在维护，或者是否有替代方案。
- **模型训练数据——私有还是公开？**：一位成员询问人们使用什么样的数据来训练大型语言模型。
   - 他们想知道人们是使用私有数据集还是像 Alpaca 这样的公开数据集，然后应用自定义 DPO 或其他无监督技术来提高性能，还是仅仅在非指令微调模型上进行 N-shot 基准测试。
- **逆向蒙特卡洛树搜索**：一位成员建议训练一个模型来逆向执行蒙特卡洛树搜索 (Monte Carlo Tree Search)。
   - 他们提议利用图像识别和生成技术来生成最优树搜索选项，而不是去识别它。
- **计算机视觉研究：论文反馈**：一位成员分享了他们正在进行的计算机视觉扩散模型项目，并正在寻求对其论文草案的反馈。
   - 他们提到大规模测试成本高昂，并请求帮助寻找可以审阅其工作的人员。

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 是否支持 GPT-4o-mini？**: 一位用户询问 `llama_index.llms.openai` 是否支持使用 `gpt-4o-mini` OpenAI 模型。
   - 另一位成员确认目前不支持该模型，并分享了支持的模型列表：`gpt-4`, `gpt-4-32k`, `gpt-4-1106-preview`, `gpt-4-0125-preview`, `gpt-4-turbo-preview`, `gpt-4-vision-preview`, `gpt-4-1106-vision-preview`, `gpt-4-turbo-2024-04-09`, `gpt-4-turbo`, `gpt-4o`, `gpt-4o-2024-05-13`, `gpt-4-0613`, `gpt-4-32k-0613`, `gpt-4-0314`, `gpt-4-32k-0314`, `gpt-3.5-turbo`, `gpt-3.5-turbo-16k`, `gpt-3.5-turbo-0125`, `gpt-3.5-turbo-1106`, `gpt-3.5-turbo-0613`, `gpt-3.5-turbo-16k-0613`, `gpt-3.5-turbo-0301`, `text-davinci-003`, `text-davinci-002`, `gpt-3.5-turbo-instruct`, `text-ada-001`, `text-babbage-001`, `text-curie-001`, `ada`, `babbage`, `curie`, `davinci`, `gpt-35-turbo-16k`, `gpt-35-turbo`, `gpt-35-turbo-0125`, `gpt-35-turbo-1106`, `gpt-35-turbo-0613`, `gpt-35-turbo-16k-0613`。
- **LlamaIndex 的 OpenAI 库需要更新**: 一位成员报告在运行 LlamaIndex 时遇到了与 `gpt-4o-mini` OpenAI 模型相关的错误。
   - 建议他们更新 `llama-index-llms-openai` 库以解决此问题。
- **Pydantic v2 导致 LlamaIndex 出错，但正在修复中**: 一位成员遇到了与 LlamaIndex `v0.11` 和 `pydantic v2` 相关的问题，LLM 会对 `pydantic` 结构产生幻觉。
   - 他们分享了 GitHub 上的 issue 链接，并表示修复程序正在开发中。
- **使用 OpenAILike 解决 GraphRAG 身份验证错误**: 一位成员在使用 GraphRAG 配合自定义网关与 OpenAI API 交互时遇到了身份验证错误。
   - 问题追溯到 GraphRAG 实现内部进行的直接 OpenAI API 调用，建议他们使用 `OpenAILike` 类来解决此问题。
- **构建多 Agent 自然语言转 SQL 聊天机器人**: 一位成员寻求关于使用 LlamaIndex 工具构建多 Agent 系统以驱动 NL-to-SQL-to-NL 聊天机器人的指导。
   - 建议他们考虑使用 workflows 或 reAct agents，但未给出最终定论。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **QLoRA 与 FSDP1 不兼容**: 用户讨论了 **QLoRA** 是否与 **FSDP1** 兼容以进行分布式微调（finetuning），结果确定它们不兼容。
   - 如果未来需要这种兼容性，这是一个需要考虑的开发点。
- **Torch.compile 与 Liger Kernels 在 Torchtune 中的对比**: 一位用户质疑 **Liger kernels** 在 **Torchtune** 中的价值，一位成员回应称他们更倾向于使用 `torch.compile`。
- **全模型编译与逐层编译的性能对比**: 讨论集中在 `torch.compile` 应用于整个模型与应用于单个层时的性能表现。
- **激活检查点（Activation Checkpointing）的影响**: 发现 **Activation Checkpointing** (AC) 会显著影响编译性能。
- **平衡编译粒度下的速度与优化**: 讨论涉及模型编译的粒度，不同层级会影响性能和优化潜力。
   - 目标是在速度和优化之间找到合适的平衡点。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4 会自信地产生幻觉**: 一位成员讨论了 GPT-4 即使在被纠正后仍会自信地提供错误答案的挑战。
   - 他们建议在提示词中加入具体的网络搜索指令，使用像 Perplexity 这样预配置的 LLM，或者设置一个带有网络搜索指令的自定义 GPT。
- **Mini 模型对比 GPT-4**: 一位成员指出，在某些场景下，Mini 模型比 GPT-4 更便宜且表现更好。
   - 他们认为这主要是人类偏好的问题，且所使用的基准测试（benchmark）并不能反映人们真正关心的使用场景。
- **SearchGPT 对比 Perplexity**: 一位成员询问了 Perplexity 相比 SearchGPT 的优势。
   - 另一位成员回答说他们还没试过 Perplexity，但认为 SearchGPT 很准确、偏见极小，且非常适合复杂的搜索。
- **AI 是否有意识？**: 一位成员讨论了 AI 体验类似人类情感的想法，认为将此类体验归因于 AI 可能是一种误解。
   - 他们表示，将理解 AGI 视为需要类人情感可能是一个不切实际的期望。
- **Orion 模型访问权限担忧**: 一位成员对将 Orion-14B-Base 和 Orion-14B-Chat-RAG 等 Orion 模型的访问权限限制在私营部门的潜在后果表示担忧。
   - 他们认为这可能会加剧不平等、扼杀创新并限制更广泛的社会效益，可能导致技术进步仅为精英利益服务的未来。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Google 发布了三个新的 Gemini 模型**：Google 发布了三个实验性的 Gemini 模型：**Gemini 1.5 Flash-8B**、**Gemini 1.5 Pro** 和 **Gemini 1.5 Flash**。 
   - 这些模型可以在 [Aistudio](https://aistudio.google.com) 上进行访问和实验。
- **Gemini 1.5 Pro 专注于 Coding 和复杂 Prompts**：**Gemini 1.5 Pro** 模型被强调在 **coding 和 complex prompts** 方面具有改进的能力。
   - 原始的 Gemini 1.5 Flash 模型也得到了**显著提升**。
- **API 可用性担忧与 Benchmark 怀疑**：存在关于 **API 可用性** 的讨论，一位用户对其功能缺失表示沮丧。
   - 该用户还提到他们尝试在 **RewardBench 上评估 8B 模型**，但认为这是一个虚假的 benchmark。
- **SnailBot 为短链接提供及时通知**：当使用 [livenow.youknow](https://livenow.youknow.com/) 缩短链接时，[SnailBot](https://www.snailbot.com/) 会在用户收到电子邮件之前通过 Discord 发出通知。
   - 然而，该用户也注意到 SnailBot 未能识别 URL 的更改，这表明该工具存在局限性。
- **开源数据可用性与代码授权趋势并行**：一位用户预测，关于数据可用性的开源辩论将遵循与代码授权现状类似的轨迹。
   - 他们认为，虽然 [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) 许可证被认为是开源的，但通常会被避开，而 [MIT](https://opensource.org/licenses/MIT) 许可证被认为更受欢迎。他们相信数据可用性的辩论也会出现类似的趋势，即数据可用性被认为是有益的，但对大多数用户来说最终是可选的。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere API 错误与 Token 计数**：一位用户报告在使用 Langchain 和 Cohere TypeScript 对 Cohere API 进行后续调用时出现 404 错误。
   - 错误消息显示为 "non-json" 响应，这表明 Cohere API 返回了一个 404 页面而不是 JSON 对象。该用户还询问了关于 Cohere API 的 token 计数问题。
- **Aya-23-8b 推理速度**：一位用户询问 Aya-23-8b 模型是否能在 500 毫秒内完成约 50 个 token 的推理。
   - 模型 quantization 被建议作为实现更快推理速度的潜在解决方案。
- **波斯语旅游景点 App**：一款结合了 Cohere AI 和 Google Places API 的 Next.js 应用上线，用于推荐波斯语的旅游景点。
   - 该应用具有详细的信息，包括描述、地址、坐标和照片，均采用高质量的波斯语。
- **应用功能与特性**：该应用利用 Cohere AI 和 Google Places API 的强大功能，提供准确且引人入胜的波斯语旅游建议。
   - 用户可以探索具有详细信息的旅游景点，包括描述、地址、坐标和照片，所有内容均以高质量的波斯语格式呈现。
- **社区反馈与分享**：社区的几位成员表达了试用该应用的兴趣，称赞其功能和创新方法。
   - 该应用已在 GitHub 和 Medium 上公开分享，邀请社区提供反馈和协作。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 编译器消除了循环导入 (Circular Imports)**：Mojo 是一种编译型语言，因此它可以在编译前扫描每个文件并确定 struct 的形状，从而解决循环导入问题，因为它拥有实现 `to_queue` 所需的所有函数。
   - Python 处理循环导入的方式不同，因为它在编译期间按顺序运行所有内容，这会导致 Mojo 编译器可以避免的潜在问题。
- **Mojo 编译器优化 Struct 大小**：Mojo 编译器知道指针的大小，这使得它无需查看 `Queue` 就能计算出 `List` 的形状。
   - Mojo 使用 `Arc` 或其他类型的指针来打破循环导入链。
- **Mojo 对顶级语句 (Top Level Statements) 的潜力**：Mojo 目前没有顶级语句，但预计它们将通过在 `main` 开始之前按导入顺序运行顶级代码来处理循环导入。
   - 这将确保循环导入得到正确且高效的解决。
- **Mojo 异常的性能曲线**：一位用户观察到 Mojo 的性能在约 1125 个字段时急剧上升，推测可能是 **smallvec** 或 **arena** 发生了溢出。
   - 另一位用户建议，原因可能是单个文件包含 **1024 个参数**。
- **Mojo 命名返回槽 (Named Return Slots) 解析**：Mojo 支持命名返回槽，允许使用类似 `fn thing() -> String as result: result = "foo"` 的语法。
   - 该功能似乎旨在用于被调用方帧 (callee frame) 内的 "placement new"，尽管语法可能会发生变化。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Google 发布全新 Gemini 模型**：Google 发布了三个实验性 Gemini 模型：Gemini 1.5 Flash-8B、Gemini 1.5 Pro 以及显著改进的 Gemini 1.5 Flash 模型。
   - 用户可以在 [aistudio.google.com](https://aistudio.google.com/) 探索这些模型。
- **Anthropic 的 Claude 3.5 Sonnet：编程利器**：Anthropic 于 6 月发布的 Claude 3.5 Sonnet 已成为编程任务的强力竞争者，表现优于 ChatGPT。
   - 这一进展可能标志着 LLM 领导地位的转变，Anthropic 有可能占据领先地位。
- **Artifacts 席卷移动端**：Anthropic 的创新项目 Artifacts 已在 iOS 和 Android 上线。
   - 此次移动端发布允许使用 Claude 创建简单的游戏，将 LLM 的力量带入移动应用。
- **Cartesia 的 Sonic：端侧 AI 革命**：专注于普及 AI 的 Cartesia 推出了其首个里程碑：Sonic，全球最快的生成式语音 API。
   - Sonic 旨在将 AI 带到所有设备上，促进与世界之间保护隐私且快速的交互，有望改变机器人、游戏和医疗领域的应用。
- **Cerebras 推理：速度之王**：Cerebras 推出了其推理解决方案，展示了 AI 处理速度的显著提升。
   - 该解决方案由定制硬件和内存技术驱动，速度高达 1800 tokens/s，在速度和设置简易性方面均超越了 Groq。



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **OpenInterpreter 获得新的指令格式**：一位成员建议将 `interpreter.custom_instructions` 设置为使用 `str(" ".join(Messages.system_message))` 的字符串而非列表，以潜在地解决一个问题。
   - 这一更改可能会改进 OpenInterpreter 对自定义指令的处理。
- **Daily Bots 发布，专注于实时 AI**：Daily Bots 是一个用于语音、视觉和视频 AI 的超低延迟云平台，现已发布并专注于实时 AI。
   - Daily Bots 是开源的并支持 RTVI 标准，旨在将实时 AI 的最佳工具、开发者体验和基础设施整合到一个平台中。
- **Bland 的 AI 电话拨打 Agent 结束隐身状态**：Bland 是一款听起来像真人一样的可定制 AI 电话拨打 Agent，已筹集 2200 万美元 A 轮融资，现已结束隐身状态。
   - Bland 可以使用任何语言或声音交谈，针对任何用例进行设计，并能 24/7 全天候同时处理数百万个电话，且不会产生幻觉。
- **Jupyter Book 元数据指南**：一位成员分享了关于使用 Python 代码向 notebook 添加元数据的 Jupyter Book 文档链接。
   - 该文档提供了如何向 Jupyter Book 中的各类内容（如代码、文本和图像）添加元数据的指导。
- **OpenInterpreter 开发持续进行**：一位成员确认 OpenInterpreter 的开发仍在进行中，并分享了 GitHub 上 OpenInterpreter 主仓库的链接。
   - 提交历史记录显示了活跃的开发进展和来自社区的贡献。

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Axolotl 在 Apple Silicon (M3) 上的运行情况**: 一位用户确认 Axolotl 可以在 Apple Silicon 上使用，特别是 M3 芯片。
   - 他们提到在 128GB RAM 的 Macbook 上运行没有出现任何错误，但未提供关于训练速度或是否需要进行自定义设置的细节。
- **IBM 推出 Power Scheduler：一种新的学习率方法**: IBM 推出了一种名为 Power Scheduler 的新型学习率调度器，它与 batch size 和训练 token 数量无关。
   - 该调度器是在对学习率、batch size 和训练 token 之间的相关性进行广泛研究后开发的，揭示了它们之间的幂律关系（power-law relationship）。该调度器在各种模型规模和架构中始终保持令人印象深刻的性能，甚至超越了 state-of-the-art 的小型语言模型。[来自 AK (@_akhaliq) 的推文](https://x.com/_akhaliq/status/1828267147765702856?s=46)
- **Power Scheduler：适用于所有配置的单一学习率**: 这种创新的调度器允许为任何给定的 token 计数、batch size 和模型规模预测最佳学习率。
   - 通过使用公式 `lr = bsz * a * T ^ b!`，可以在不同配置下实现使用单一学习率。[来自 Yikang Shen (@Yikang_Shen) 的推文](https://x.com/yikang_shen/status/1828503458749501714?s=46)
- **QLora FSDP 参数争议**: 关于在 QLora FSDP 示例中正确设置 `fsdp_use_orig_params` 的讨论。
   - 一些成员认为它应该始终设置为 `true`，而另一些成员则不确定，并认为这可能不是一个严格的要求。
- **模型训练中的异常 Token 行为**: 一位成员询问，如果其数据集中的 token 与预训练数据集相比具有异常含义，模型是否应该识别出来。
   - 该成员建议，出现频率高于正态分布的 token 可能是有效训练的一个指标。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 的 "ImageNet 时刻"**: DSPy 的 "ImageNet" 时刻归功于 @BoWang87 实验室在 MEDIQA 挑战赛中的成功，其中基于 DSPy 的解决方案以 12.8% 和 19.6% 的显著优势赢得了两项临床 NLP 竞赛。
   - 这一成功导致 DSPy 的采用率大幅增加，类似于 CNN 在 ImageNet 上表现出色后变得流行。
- **NeurIPS HackerCup：DSPy 的下一个 "ImageNet 时刻"？**: NeurIPS 2024 HackerCup 挑战赛被视为 DSPy 潜在的 "ImageNet 时刻"，类似于卷积神经网络在 ImageNet 上脱颖而出后获得显赫地位。
   - 该挑战赛为 DSPy 提供了一个展示其能力并可能获得更大规模采用的机会。
- **用于代码生成的 DSPy 优化器**: DSPy 正被用于代码生成，最近的一次演讲涵盖了它在 NeurIPS HackerCup 挑战赛中的应用。
   - 这表明 DSPy 不仅对 NLP 有效，对代码生成等其他领域也同样有效。
- **DSPy 入门**: 对于对 DSPy 感兴趣的人，@kristahopsalong 最近在 Weights & Biases 上发表了关于 DSPy 编程入门的演讲，涵盖了其优化器以及使用 2023 HackerCup 数据集的动手演示。
   - 这次演讲为任何有兴趣了解更多关于 DSPy 及其在编程中应用的人提供了一个很好的起点。
- **修改 OpenAI Base URL 和模型**: 一位用户想要将 OpenAI 的 base URL 和模型更改为其他的 LLM（如 OpenRouter API），但找不到实现方法。
   - 他们提供了演示其尝试的代码片段，其中包括在 `dspy.OpenAI` 中设置 `api_base` 和 `model_type` 参数。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 发货至欧洲**: Tinygrad 现在提供发货至欧洲的服务！如需索取报价，请发送电子邮件至 [support@tinygrad.org](mailto:support@tinygrad.org)，并注明您的地址和所需的机箱。
   - Tinygrad 致力于让发货尽可能便捷，并将尽最大努力为您提供所需的机箱！
- **Tinygrad CPU 错误："No module named 'tinygrad.runtime.ops_cpu'"**: 一位用户报告在 CPU 上运行 Tinygrad 时遇到 "ModuleNotFoundError: No module named 'tinygrad.runtime.ops_cpu'" 错误。
   - 回复建议使用设备 "clang"、"llvm" 或 "python" 来在 CPU 上运行，例如：`a = Tensor([1,2,3], device="clang")`。
- **在 Tinygrad 中获取设备数量**: 一位用户询问是否有比使用 `tensor.realize().lazydata.base.realized.allocator.device.count` 更简单的方法来获取 Tinygrad 中的设备数量。
   - 该用户发现 `from tinygrad.device import Device` 和 `Device["device"].allocator.device.count` 提供了一个更直接的解决方案。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **LAION-aesthetic 数据集链接已断开**：一位成员请求获取 LAION-aesthetic 数据集的 Hugging Face 链接，因为 [LAION 官网](https://laion.ai/)上的链接已失效。
   - 另一位成员建议查看 [Hugging Face 上的 CaptionEmporium 数据集](https://huggingface.co/datasets/laion/captionemporium)，将其作为潜在的替代方案或相关数据源。
- **LAION-aesthetic 数据集链接替代方案**：LAION-aesthetic 数据集是一个用于图像描述（captioning）的数据集。
   - 该数据集包含审美评判的各个方面，对于图像描述和图像生成模型可能具有很高的价值。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **在自定义 API 上进行 Llama 3.1 基准测试**：一位用户请求关于使用自定义 API 对 **Llama 3.1** 进行基准测试的建议，特别是针对私有托管的 **Llama 3.1 endpoint** 及其推理流水线（inference pipeline）。
   - 他们正在寻求关于如何有效评估其推理流水线相对于 **Llama 3.1 endpoint** 性能的指导。
- **Gorilla 的 BFCL 排行榜与模型处理器优化**：一位用户提出了一个问题，即他们为函数调用（function-calling）功能实施的某些优化是否会被认为对 **BFCL Leaderboard** 上的其他模型不公平。
   - 他们担心 **BFCL Leaderboard** 对模型处理器（model handler）优化的立场，因为这些优化可能无法推广到所有模型，特别是涉及系统提示词（system prompts）、聊天模板（chat templates）、带约束解码的束搜索（beam search with constrained decoding）以及输出格式化。

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长期处于沉寂状态，请告知我们，我们将将其移除。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该频道长期处于沉寂状态，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长期处于沉寂状态，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该频道长期处于沉寂状态，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该频道长期处于沉寂状态，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期处于沉寂状态，请告知我们，我们将将其移除。

---

# PART 2: 按频道划分的详细摘要和链接

{% if medium == 'web' %}

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1278071921954521171)** (468 条消息🔥🔥🔥): 

> - `Gamification of home training` (家庭训练游戏化)
> - `Triton config` (Triton 配置)
> - `Loss curve` (Loss 曲线)
> - `Model finetuning` (模型微调)
> - `GPU performance` (GPU 性能) 


- **家庭训练游戏化 (Gamification of home training)**：一位成员提出了一个关于训练基准测试工具的新想法，旨在实现“家庭训练游戏化”。
   - 该成员声明他们将把这个想法用于他们的求职申请。 
- **Triton 配置困扰**：一位成员在使用 llama3 instruct 配合 Triton 以及 tensorrt-llm 或 vllm 后端时，遇到了响应生成无法停止的问题。
   - 他们尝试直接使用 vllm 托管，运行非常完美，这表明其 Triton 配置可能存在潜在问题。
- **Loss 曲线分析**：几位成员讨论了模型训练中“Loss 曲线”的重要性，其中一位成员建议 Loss 为 0.0 可能表示日志记录错误。
   - 该成员质疑由于舍入原因，Loss 为 0.0 的完美模型在现实中是否可以实现。
- **Gemma 微调困境**：一位成员报告在 AMD 上微调 Gemma2b 模型时遇到困难，将其归因于可能的日志记录错误。
   - 其他成员指出 ROCm 仍处于实验阶段，这可能是导致困难的原因之一。
- **模型合并策略讨论**：一位成员建议了一种模型合并策略，即将 UltraChat 与基础 Mistral 之间的差异应用到 Mistral-Yarn 上。
   - 该建议遭到了一些人的质疑，但该成员保持乐观，并引用了他们所谓的“诅咒模型合并 (cursed model merging)”的过往成功案例。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://gamengen.github.io/">GameNGen</a>: Diffusion Models Are Real-Time Game Engines (扩散模型是实时游戏引擎)</li><li><a href="https://x.com/Peeplika">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://app.alignedhq.ai/few-shot-examples">Aligned</a>: 未找到描述</li><li><a href="https://huggingface.co/unclemusclez/SmolLM-135M-Instruct-DEVINator">unclemusclez/SmolLM-135M-Instruct-DEVINator · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/mikemin027/Gemma-7b-it-GGUF">Gemma 7b It GGUF - a Hugging Face Space by mikemin027</a>: 未找到描述</li><li><a href="https://huggingface.co/OpenMeditron/Meditron3-8B">OpenMeditron/Meditron3-8B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/unclemusclez/SmolLM-135M-Instruct-DEVINator?show_file_info=model.safetensors>">unclemusclez/SmolLM-135M-Instruct-DEVINator · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/o-hearn-gif-3900414469346077199">O Hearn GIF - O hearn - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/mike-ohearn-gif-5978288995841027384">Mike Ohearn GIF - Mike ohearn - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/mikeohearn-gif-9467924472242763968">Mikeohearn GIF - Mikeohearn - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/pakistan-cricket-fan-pakistan-fan-cricket-angry-fan-angry-fan-angry-man-gif-19825067">Pakistan Cricket Fan Pakistan Fan GIF - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/huggingface/transformers/issues/12062>">Issues · huggingface/transformers</a>: 🤗 Transformers: Pytorch, TensorFlow 和 JAX 的最先进机器学习库。 - Issues · huggingface/transformers</li><li><a href="https://tenor.com/view/huh-cat-gif-26460616">Huh Cat GIF - Huh Cat - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/ohearn-sad-ohearn-mike-ohearn-sad-mike-sad-gif-13532193191719643333">Ohearn Sad Mike Ohearn Sad GIF - 发现并分享 GIF</a>: 点击查看 GIF
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1278097132292935741)** (6 条消息): 

> - `Training AI on CPU` (在 CPU 上训练 AI)
> - `Training AI on Laptop` (在笔记本电脑上训练 AI)
> - `Colab TPU instances` (Colab TPU 实例) 


- **在没有 GPU 的笔记本电脑上训练 AI**：在笔记本电脑的 CPU 上训练 AI 模型将耗费非常长的时间。
   - 建议使用 Kaggle 或 Google Colab 等云端平台以获得更快的训练速度。
- **用于训练的 Colab TPU 实例**：Google Colab 提供对 TPU 实例的访问，这些实例通常是免费使用的。
   - 有一个 [Colab notebook](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/tpu.ipynb) 展示了如何在 Colab 中使用 TPU。



**提到的链接**：<a href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/tpu.ipynb">Google Colab</a>: 未找到描述

  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1278071081822978232)** (4 条消息): 

> - `DisTrO`
> - `GameNGen`
> - `Llama Implementation`
> - `WiM` 


- **Nous Research 的 DisTrO：分布式优化器，通信量减少 1000 倍至 10,000 倍**：Nous Research 发布了 **DisTrO**，这是一种与架构无关的分布式优化器，可将 GPU 间的通信量减少 **1000 倍至 10,000 倍**。
   - 有关 DisTrO 工作原理的初步报告详见[此处](https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf)。
- **Google 推出 GameNGen：基于 Diffusion Models 构建的实时游戏引擎**：**GameNGen** 是首个完全由神经模型驱动的实时游戏引擎，能够在长轨迹上实现高质量的交互式模拟。
   - GameNGen 可以在单个 TPU 上以每秒 **20 帧**的速度模拟 **DOOM**，并达到 **29.4 的 PSNR**——与有损 JPEG 压缩相当。
- **针对长上下文 LLM 的 WiM 推理模式**：**Writing in the Margins (WiM)** 方法是一种新的推理模式，可提升长上下文 LLM 在检索导向型任务中的性能。
   - WiM 利用分段推理（segment-wise inference）来高效处理海量上下文，使推理能力的平均准确率提升了 **7.5%**，聚合任务的 F1-score 提升了 **30.0%**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2408.14906">Writing in the Margins: Better Inference Pattern for Long Context Retrieval</a>: 在本文中，我们介绍了 Writing in the Margins (WiM)，这是一种针对 Large Language Models 的新推理模式，旨在优化检索导向型任务中长输入序列的处理。这 ...</li><li><a href="https://huggingface.co/papers/2408.14837">Paper page - Diffusion Models Are Real-Time Game Engines</a>: 未找到描述</li><li><a href="https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf">DisTrO/A_Preliminary_Report_on_DisTrO.pdf at main · NousResearch/DisTrO</a>: 互联网上的分布式训练。通过在 GitHub 上创建账户来为 NousResearch/DisTrO 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1278077337811816461)** (11 条消息🔥): 

> - `RYFAI`
> - `Tau LLM Series`
> - `Streetwear Flux`
> - `Loadimg`
> - `Bielik-11B` 


- **RYFAI: Open Source AI Assistant**: RYFAI，一个基于 Raspberry Pi 构建的 AI 语音助手，现已开源！您可以在 GitHub 上查看并为该项目做出贡献：[https://github.com/PetertheRedCedar/ryfai](https://github.com/PetertheRedCedar/ryfai)。
- **Tau LLM Series: Vector Database API Improvements**: Tau LLM 系列继续推出关键更新和功能。向量数据库 API 得到了增强，提高了鲁棒性和用户友好度。
   - 项目已上传至 GitHub 以方便协作和分享，并新增了数据加载和目录管理功能。AgentTrainer 和奖励信号的开发仍在继续。
- **Streetwear Flux: AI-Generated Design**: 一个名为 'Streetwear Flux' 的新 Hugging Face 模型已创建，旨在生成受街头服饰启发的设计。
   - 模型卡包含一个用于生成基于文本的街头服饰图形的 Prompt，其特点是包含文本 "TOKYO TECHWEAR STYLE" 及各种视觉元素。
- **Loadimg: Image Loading Library**: 专为加载图像设计的 'loadimg' Python 库在一个月内达到了 100,000 次下载。
   - 您可以在 PyPI 上找到该库：[https://pypi.org/project/loadimg/](https://pypi.org/project/loadimg/)，其源代码位于 GitHub：[https://github.com/not-lain/loadimg](https://github.com/not-lain/loadimg)。
- **Bielik-11B: Polish Language Model**: 一个新的波兰语语言模型 Bielik-11B 已发布，在波兰语和英语基准测试中均取得了顶尖性能。
   - 该模型在 4000 亿个 Token 上进行了训练，拥有 110 亿个参数，利用 PLGrid 环境和 HPC 中心 ACK Cyfronet AGH 进行开发。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/DamarJati/streetwear-flux">DamarJati/streetwear-flux · Hugging Face</a>: 未找到描述</li><li><a href="https://wandb.ai/bghira/preserved-reports/reports/Bghira-s-Search-for-Reliable-Multi-Subject-Training--Vmlldzo5MTY5OTk1">Bghira's Search for Reliable Multi-Subject Training</a>: 问题：角色倾向于融合在一起。建议：在探索 hyper-parameters 后，探索 captioning 和 prompting。由 Bagheera 使用 W&B 制作</li><li><a href="https://github.com/PetertheRedCedar/ryfai">GitHub - PetertheRedCedar/ryfai: This is an AI app designed to bring open source AI models to your fingertips with ease</a>: 这是一款旨在让您轻松触及开源 AI 模型的 AI 应用 - PetertheRedCedar/ryfai</li><li><a href="https://huggingface.co/speakleash/Bielik-11B-v2">speakleash/Bielik-11B-v2 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/speakleash/Bielik-11B-v2.2-Instruct">speakleash/Bielik-11B-v2.2-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://pypistats.org/packages/loadimg">
        PyPI Download Stats
    </a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1278111764113920101)** (3 messages): 

> - `VAEs for Text-Image Generation`
> - `Transformers Library Contribution`
> - `Document Quality Assessment`
> - `Data Augmentation for Document Quality` 


- **VAEs for Text-Image Generation: Why Not More Popular?**: 一位用户询问，为什么通过将图像编码到共享潜空间并利用文本解码器进行解码（或反之亦然）来从图像生成文本的变分自编码器（VAEs），在研究和实际应用中没有被更广泛地采用。
- **Transformers Library Good-First-Issue**: 一位用户询问是否有人对 Transformers 库的 'good-first-issue' 感兴趣，并链接了一个关于 DeformableDetr 权重初始化的特定 issue。
   - 该 issue 旨在确保所有权重初始化都在相应 PretrainedModel 类的 `_init_weights` 方法中执行。
- **Document Quality Assessment: Identifying Blurred, Dark, or Blank Docs**: 一位用户正在寻求一种评估文档质量的方法，特别是在没有现成训练数据的情况下，识别上传的文档是否模糊、过暗或空白。
   - 他们尝试通过抓取公开的打印文档并添加模糊效果进行增强来解决此问题，但效果不佳，因为模拟真实的文档照片极具挑战性。



**Link mentioned**: <a href="https://github.com/huggingface/transformers/issues/29818">Move weight initialization for DeformableDetr · Issue #29818 · huggingface/transformers</a>: 系统信息：不相关。复现：参见 Deformable Detr 建模。预期行为：所有权重初始化应在 xxxPretrainedModel 类的 `_init_weights` 中完成。

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1278071299385725040)** (1 messages): 

> - `Text-Summary trends 2024`
> - `Specialized vs General Models`
> - `Llama Long Context`
> - `System Prompts` 


- **Text-Summary still relevant in 2024**: 文本摘要模型在 2024 年依然重要，但格局正在发生变化。
- **Llama excels with long context**: Llama 通过 System Prompts 处理大量上下文的能力使其成为摘要生成的强大工具。
- **Specialization vs Generalization**: 在专用文本摘要模型和像 Llama 这样的通用模型之间做出选择，取决于具体的任务和预期的结果。


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1278231885503266889)** (1 messages): 

> - `` 


- **This channel is for diffusers specific discussions.**: 本频道专门用于 Diffusers 的相关讨论。
- **Diffusion channel guidance**: 发布了一条欢迎消息，以对频道的预期用途提供指导。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1278066520047747265)** (228 条消息🔥🔥): 

> - `Kaggle 上的 VLLM`
> - `Kaggle 上的 Aphrodite`
> - `Colab 上的 VLLM`
> - `Mistral 的困境`
> - `模型合并 (Model Merging)` 


- **VLLM 在 Kaggle 上可以运行了！**: 有用户报告称，使用来自 [此数据集](https://www.kaggle.com/datasets/sethmoudry/arc-vllm-wheels) 的 **wheels**，成功在 Kaggle 上运行了 **VLLM**。
   - 这是使用 **VLLM 0.5.4** 实现的，该版本仍被认为相对较新，因为 **0.5.5** 虽然已发布但尚未广泛普及。
- **Mistral 在扩展超过 8k 长度时遇到困难**: 成员们确认，如果不进行持续预训练 (continued pretraining)，**Mistral** 无法扩展到 8k 以上，并且[这是一个已知问题](https://link.to.issue)。
   - 他们还讨论了未来性能增强的潜在途径，包括 **mergekit** 和 **frankenMoE finetuning**。
- **Homoiconic AI：权重即代码？**: 一位成员分享了关于 [“Homoiconic AI” 的进展报告](https://x.com/neurallambda/status/1828214178567647584?s=46)，该项目使用 **hypernet** 生成 **autoencoder weights**，然后通过 **in-context learning** 改进这些权重。
   - 报告指出，这种“代码即数据 & 数据即代码”的方法可能是推理所必需的，甚至与推理同构。
- **Unsloth 的持续预训练能力**: 一位成员分享了 Unsloth 关于 [持续预训练的博客文章](https://unsloth.ai/blog/contpretraining)，强调其持续预训练 LLM 的速度比 Hugging Face + Flash Attention 2 QLoRA **快 2 倍**，且 **VRAM 占用减少 50%**。
   - 博客文章还提到使用 [Colab notebook](https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing) 对 Mistral v0.3 7b 进行持续预训练以**学习韩语**。
- **Unsloth vs OpenRLHF：速度与内存效率**: 一位用户询问了 Unsloth 和 OpenRLHF 之间的区别，特别是关于它们对微调未量化模型的支持。
   - 成员确认 Unsloth 支持未量化模型，并计划很快添加 8bit 支持，同时强调其与其他微调方法相比，速度显著更快且内存占用更低。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.kaggle.com/datasets/abdurrafae/vllm-t4-fix">vllm T4 Fix</a>: 未找到描述</li><li><a href="https://huggingface.co/speakleash/Bielik-11B-v2">speakleash/Bielik-11B-v2 · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/neurallambda/status/1828214178567647584?s=46">来自 neurallambda (open agi) (@neurallambda) 的推文</a>: “Homoiconic AI” 进展报告：我们使用 hypernet 生成 autoencoder 的权重，然后进行 in-context learning（掩码重构损失）来改进这些权重...</li><li><a href="https://unsloth.ai/blog/contpretraining">使用 Unsloth 进行 LLM 持续预训练</a>: 通过使用 Unsloth 对 Llama 3, Phi-3 和 Mistral 进行持续预训练，让模型学习一种新语言。</li><li><a href="https://www.kaggle.com/code/cdeotte/infer-34b-with-vllm">使用 vLLM 推理 34B 模型</a>: 使用 Kaggle Notebooks 探索并运行机器学习代码 | 使用来自多个数据源的数据</li><li><a href="https://x.com/neurallambda/status/1828214178567647584?s">来自 neurallambda (open agi) (@neurallambda) 的推文</a>: “Homoiconic AI” 进展报告：我们使用 hypernet 生成 autoencoder 的权重，然后进行 in-context learning（掩码重构损失）来改进这些权重...</li><li><a href="https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf">llava-hf/LLaVA-NeXT-Video-7B-hf · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: 未找到描述</li><li><a href="https://huggingface.co/speakleash/Bielik-11B-v2.2-Instruct">speakleash/Bielik-11B-v2.2-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/Lightning-AI/litgpt">GitHub - Lightning-AI/litgpt: 20+ 高性能 LLMs，包含大规模预训练、微调和部署的方案。</a>: 20+ 高性能 LLMs，包含大规模预训练、微调和部署的方案。 - Lightning-AI/litgpt
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1278070736535158898)** (64 条消息🔥🔥): 

> - `8bit training` (8bit 训练)
> - `Unsloth Cont Pretraining` (Unsloth 持续预训练)
> - `Dataset Size` (数据集大小)
> - `Context Length` (上下文长度)
> - `Model Layer Tuning` (模型层微调)


- **使用 Unsloth 进行 8bit 训练**：一位成员询问是否可以使用 Unsloth 以 8bit 模式训练 Nemo-12b，但获知 Unsloth 目前仅支持 4bit 训练。
   - 随后建议他们使用 Axolotl 或类似的工具进行 8bit 训练。
- **使用 Unsloth 进行持续预训练**：针对用户 LoRA 表现不佳的问题，建议使用 Unsloth 的 [持续预训练 (Continual Pretraining)](https://unsloth.ai/blog/contpretraining) 功能。
   - 与 Hugging Face + Flash Attention 2 QLoRA 相比，该功能允许以 2 倍的速度进行 LLM 的持续预训练，且节省 50% 的 VRAM。
- **数据集大小与学习率**：用户的数据集大小为 30MB，包含 100 万个 token，被认为规模较小。
   - 建议用户增加数据集大小，尝试不同的学习率（learning rate）和训练轮数（epochs），并对输入和输出嵌入（embeddings）进行微调。
- **在 Unsloth 中调整上下文长度**：一位用户询问如何调整 Unsloth/Llama-3-8b-Instruct-bnb-4bit 模型的上下文长度。
   - 他们得到的建议是：更改模型的上下文长度非常复杂，可能会产生不可预见的后果，应尽量避免这样做。
- **理解 Unsloth 中的模型层微调**：一位用户要求澄清 Unsloth notebook 中哪些层用于 LoRA 微调。
   - 他们获知，训练的层数越多，数据对模型的影响就越大，但目前没有万能的配置方案，关于这些层如何运作仍需进一步研究。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://unsloth.ai/blog/contpretraining">使用 Unsloth 进行持续 LLM 预训练</a>：通过使用 Unsloth 对 Llama 3、Phi-3 和 Mistral 进行持续预训练，让模型学习一种新语言。</li><li><a href="https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama">如何微调 Llama-3 并导出到 Ollama | Unsloth 文档</a>：创建自定义个人助手（类似 ChatGPT）并在 Ollama 上本地运行的初学者指南。</li><li><a href="https://github.com/unslothai/unsloth/wiki">首页</a>：微调 Llama 3.1、Mistral、Phi 和 Gemma LLM，速度提升 2-5 倍，内存占用减少 80% - unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/1weTpKOjBZxZJ5PQ-Ql8i6ptAY2x-FWVA?usp=sharing#scrollTo=IqM-T1RTzY6C">Google Colab</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k">Salesforce/xlam-function-calling-60k · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1278077619350405151)** (11 messages🔥): 

> - `Duet Dataset`
> - `Bielik-11B Model`
> - `Herplete-LLM-Llama-3.1-8b`
> - `Unsloth Community` 


- **用于角色扮演的 Duet Dataset**: 发布了一个名为 Duet 的新角色扮演数据集，提供 5k 行经过重写的 COT 问答数据，包含叙述内容并符合故事情节。
   - 该数据集包括来自角色和场景的世界观信息、原始问答以及从角色视角重写的版本。
- **Bielik-11B：波兰语语言模型**: SpeakLeash 发布了一个新的顶级波兰语语言模型 Bielik-11B，包含基础版（base）和指令版（instruct）。
   - 该模型在波兰语文本语料库上进行训练，并利用了 ACK Cyfronet AGH 的 HPC 中心，使用了计算授权编号 PLG/2024/016951。
- **Herplete-LLM-Llama-3.1-8b：持续微调模型**: Herplete-LLM-Llama-3.1-8b 是一个持续微调模型，源自 Replete-AI/Replete-LLM-V2-Llama-3.1-8b，并使用了从 Hermes-3-Llama-3.1-8B 提取的 LoRA。
   - 它获得了其训练所基于的两个模型的技能，包括编程、角色扮演、数学等。
- **Unsloth 社区支持**: 感谢 Unsloth 社区对 Bielik-11B 开发的贡献，特别是使用 Unsloth 进行原型设计和 LoRA 测试。
   - 鼓励成员在专用频道分享他们自己的模型，因为他们是社区中值得信赖的成员。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/Etherll/Herplete-LLM-Llama-3.1-8b">Etherll/Herplete-LLM-Llama-3.1-8b · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/speakleash/Bielik-11B-v2">speakleash/Bielik-11B-v2 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/speakleash/Bielik-11B-v2.2-Instruct">speakleash/Bielik-11B-v2.2-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/G-reen/Duet-v0.5">G-reen/Duet-v0.5 · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/G-reen/Duet_Minitron8b_v0.5">G-reen/Duet_Minitron8b_v0.5 · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/)** (1 messages): 

mrdragonfox: 好吧 - 奇怪的人在做奇怪的事
  

---



### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1278356476624244776)** (1 messages): 

> - `Aider v0.54.0`
> - `New Gemini Models`
> - `Shell Command Improvements`
> - `Aider's Role in Development`
> - `Performance Enhancements` 


- **Aider 添加新的 Gemini 模型**: Aider 的最新版本引入了对 **`gemini/gemini-1.5-pro-exp-0827`** 和 **`gemini/gemini-1.5-flash-exp-0827`** 模型的支持。
- **Shell 命令增强和交互性**: Shell 和 `/run` 命令现在允许在带有 pty 的环境中进行交互式执行。
   - 新开关 **`--[no-]suggest-shell-commands`** 允许自定义配置 Shell 命令建议。
- **Aider 自动化开发**: Aider 在开发此版本中发挥了重要作用，贡献了 **64%** 的代码。
- **Aider 的性能改进**: Aider 的性能得到了提升，特别是在大型和 monorepo 项目中的自动补全功能。
   - 此版本还引入了 **`--upgrade`** 开关，以便从 PyPI 轻松安装最新的 Aider 版本。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1278072306597363826)** (119 条消息🔥🔥): 

> - `Aider v0.54.0`
> - `Gemini 1.5 Pro Benchmark`
> - `OpenRouter vs Discord`
> - `Prompt Caching`
> - `Aider and Sonnet 3.5` 


- **Aider 0.54.0 发布**：发布了 Aider 的新版本 (v0.54.0)，包含多项易用性改进。
   - 发行说明可以在 Aider Discord 频道中找到：[https://discord.com/channels/1131200896827654144/1133060115264712836/1278356476624244776](https://discord.com/channels/1131200896827654144/1133060115264712836/1278356476624244776)
- **Gemini 1.5 Pro Benchmark 结果**：分享了新 Gemini 1.5 Pro 模型的基准测试结果，显示 whole edit 格式的通过率为 23.3%，diff edit 格式的通过率为 57.9%。
   - 基准测试是使用 Aider 运行的，使用了 `gemini/gemini-1.5-pro-exp-0827` 模型和 `aider --model gemini/gemini-1.5-pro-exp-0827` 命令。
- **OpenRouter 和 Discord**：一名成员询问 OpenRouter 是否与 Discord 相同。
   - 另一名成员确认两者对他们来说都运行良好，并引用了 OpenRouter 的状态页面：[https://status.openrouter.ai/](https://status.openrouter.ai/)
- **Prompt Caching 详解**：一名成员讨论了 Prompt Caching 的好处，特别是在使用相同提示词进行大量 API 调用时。
   - 他们提到了一种涉及大量 API 调用任务的理论，其中提示词保持不变，只有输入变量发生变化。
- **Aider 和 Sonnet 3.5**：一名成员报告了使用 `openai/poe-Claude-3.5-Sonnet-200k` 模型成功将 Sonnet 3.5 与 Aider 集成。
   - 他们强调了在使用 Sonnet 3.5 时，Prompt Caching 带来的潜在成本节约，并指出与直接使用 Anthropic API 相比，API 成本存在显著差异。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/usage/tips.html">Tips</a>：使用 aider 进行 AI 配对编程的技巧。</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html">Advanced model settings</a>：为 LLM 配置高级设置。</li><li><a href="https://aider.chat/docs/llms/warnings.html">Model warnings</a>：aider 是你终端里的 AI 配对编程工具。</li><li><a href="https://v0.dev/faq">v0</a>：生成式 UI 游乐场。</li><li><a href="https://gist.github.com/paul-gauthier/5b97e51e1841ede025ab746f960d2b5c">docs.md</a>：GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://ui.shadcn.com/docs">Introduction</a>：设计精美的组件，你可以将其复制并粘贴到你的应用中。易于访问、可定制、开源。</li><li><a href="https://huggingface.co/papers/2408.14354">Paper page - SWE-bench-java: A GitHub Issue Resolving Benchmark for Java</a>：未找到描述。</li><li><a href="https://multi-swe-bench.github.io">Multi-SWE-bench</a>：未找到描述。</li><li><a href="https://aider.chat/docs/leaderboards/#contributing-benchmark-results">Aider LLM Leaderboards</a>：LLM 代码编辑能力的定量基准测试。</li><li><a href="https://github.com/anthropics/anthropic-cookbook/blob/main/misc/prompt_caching.ipynb">anthropic-cookbook/misc/prompt_caching.ipynb at main · anthropics/anthropic-cookbook</a>：展示 Claude 一些有趣且有效用法的手册/配方集合。 - anthropics/anthropic-cookbook</li><li><a href="https://v0.dev/chat/">v0 by Vercel</a>：与 v0 聊天。通过简单的文本提示生成 UI。复制、粘贴、发布。</li><li><a href="https://github.com/paul-gauthier/aider/pull/1200">docs: add benchmark results for new gemini experimental models by cheahjs · Pull Request #1200 · paul-gauthier/aider</a>：添加了昨天宣布的 AI Studio 上 3 个新实验性 Gemini 模型（Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 1.5 Flash-8B）的基准测试结果。</li><li><a href="https://status.openrouter.ai/">OpenRouter Status</a>：OpenRouter 事件历史。</li><li><a href="https://openrouter.ai/docs/requests">Requests | OpenRouter</a>：处理传入和传出请求。</li><li><a href="https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json">litellm/model_prices_and_context_window.json at main · BerriAI/litellm</a>：使用 OpenAI 格式调用 100 多个 LLM API 的 Python SDK 和代理服务器 - [Bedrock, Azure, OpenAI, VertexAI, Cohere, Anthropic, Sagemaker, HuggingFace, Replicate, Groq] - BerriAI/litellm
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1278071990812676136)** (41 messages🔥): 

> - `Aider on Replit`
> - `Commit Message Errors`
> - `Aider Security`
> - `Aider Documentation`
> - `Aider Repo Map` 


- **在 Replit 上运行 Aider**：一位用户询问是否可以在 Replit 上运行 Aider，以便立即托管生成的网站。
   - 另一位用户建议，如果 Replit 上有最近的 Python 版本，这可能是可行的。
- **提交信息错误 (Commit Message Errors)**：多位用户报告遇到持久的错误消息：“Failed to generate commit message! Commit bcf...”。
   - 一位用户建议尝试重试，或者退出并重新进入，而另一位用户提到该问题可能与不同模型的不同 Prompt 有关。
- **Aider 的安全性与数据处理**：一位用户询问 Aider 是否会与配置的模型之外的任何实体通信，特别是关于使用私有 LLM 时的数据安全。
   - 另一位用户确认，据他们所知，不涉及任何遥测 (telemetry) 或代理服务，所有数据都直接发送到 LLM。
- **Aider 文档需要改进**：一位用户指出 Aider 的文档缺乏关于其数据处理实践的明确信息，这对于处理专有代码库的用户来说是一个担忧。
   - 另一位用户表示同意，并建议应在文档中明确说明这些信息。
- **Aider 中的 Repo Map 问题**：一位用户表示在大型代码库中使用 Aider 的 Repo Map 存在困难，发现它包含了一些无关信息。
   - 他们建议增加更好地控制 Repo Map 的能力，例如根据添加的文件类型进行计算。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/llms/openai.html">OpenAI</a>：aider 是你终端里的 AI 配对编程工具</li><li><a href="https://cosine.sh/genie">Genie: SOTA Software engineering model | Cosine - Human Reasoning Lab</a>：未找到描述</li><li><a href="https://pypi.org/project/ConfigArgParse/">ConfigArgParse</a>：argparse 的直接替代品，允许通过配置文件和/或环境变量设置选项。</li><li><a href="https://github.com/theskumar/python-dotenv/blob/main/README.md#multiline-values">python-dotenv/README.md at main · theskumar/python-dotenv</a>：从 .env 文件读取键值对并将其设置为环境变量。它有助于开发遵循 12-factor 原则的应用程序。</li><li><a href="https://openrouter.ai">OpenRouter</a>：LLM 路由与市场</li><li><a href="https://github.com/paul-gauthier/aider">GitHub - paul-gauthier/aider: aider is AI pair programming in your terminal</a>：aider 是你终端里的 AI 配对编程工具。通过在 GitHub 上创建账号为 aider 开发做贡献。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1278066926345654313)** (6 messages): 

> - `GameNGen`
> - `Diffusion Models`
> - `Doom`
> - `Real-time Game Engines` 


- **GameNGen - 第一个神经游戏引擎**：论文介绍了 _GameNGen_，这是第一个完全由神经模型驱动的游戏引擎，能够在复杂环境中实现高质量、长轨迹的实时交互。
   - 该模型可以在单个 TPU 上以每秒超过 20 帧的速度交互式模拟经典游戏 DOOM，其 PSNR 为 29.4，与有损 JPEG 压缩相当。
- **GameNGen 模拟 DOOM**：人们玩 [DOOM](https://en.wikipedia.org/wiki/Doom_(1993_video_game)) 游戏的实时录像完全由 _GameNGen_ 神经模型模拟。
   - 人类评分者在区分游戏短片与模拟短片时，准确率仅略高于随机概率。
- **GameNGen 训练阶段**：_GameNGen_ 的训练分为两个阶段：(1) 一个 RL-agent 学习玩游戏并记录训练过程，(2) 训练一个 Diffusion Model 来生成游戏的视觉输出。
- **使用 Diffusion Models 的实时游戏引擎**：论文探讨了 Diffusion Models 在创建实时游戏引擎方面的潜力。
   - 这项研究突显了 AI 在模拟复杂环境和实时交互方面的重大进展。



**提到的链接**：<a href="https://gamengen.github.io/">GameNGen</a>：Diffusion Models 是实时游戏引擎

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1278078239394500689)** (148 条消息🔥🔥): 

> - `LM Studio 版本`
> - `Linux 上的 LM Studio`
> - `Snapdragon 上的 LM Studio`
> - `AMD GPU 上的 LM Studio`
> - `LLM 与安全` 


- **LM Studio 最新版本**: LM Studio 的最新版本为 v0.3.1，可在 [lmstudio.aido](https://lmstudio.aido) 获取。
- **通过 Steam 在 Linux 上运行 LM Studio**: 一位用户询问如何在不使用 `--no-sandbox` 标志的情况下通过 Steam 运行 Linux 版本的 LM Studio，因为直接运行导致了 SSD 损坏。
- **LM Studio 不支持 Snapdragon NPU**: 一位用户报告称，尽管他们安装了 Snapdragon 版本的 LM Studio，但 Snapdragon 上的 NPU 在 LM Studio 中无法工作。
- **AMD GPU 上的 LM Studio**: LM Studio 的 ROCM 构建版本目前仅支持最高端的 AMD GPU，不支持 6700XT 等 GPU，从而导致兼容性问题。
- **LM Studio 的安全性与越狱**: 一位用户通过提示 LLM 下载程序来测试 LM Studio 的安全性，结果得到了幻觉回复，表明并没有实际进行下载。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="http://localhost:1234`">未找到标题</a>: 未找到描述</li><li><a href="https://lmstudio.ai">👾 LM Studio - 发现并运行本地 LLM</a>: 查找、下载并试用本地 LLM</li><li><a href="https://lmstudio.ai/snapdragon">👾 LM Studio - 发现并运行本地 LLM</a>: 查找、下载并试用本地 LLM</li><li><a href="https://llm.extractum.io/list/">所有大语言模型</a>: 大语言模型和小语言模型（开源 LLM 和 SLM）的精选列表。包含动态排序和过滤功能的所有大语言模型。</li><li><a href="https://github.com/YorkieDev/LMStudioWebUI">GitHub - YorkieDev/LMStudioWebUI: 一个用于 LM Studio 的简单 Web UI 的开发中版本</a>: 一个用于 LM Studio 的简单 Web UI 的开发中版本 - YorkieDev/LMStudioWebUI</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1f2gaqt/cogvideox_5b_open_weights_text_to_video_ai_model/">Reddit - 尽情探索</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1f2gaqt/cogvideox_5b_open_weights_text_to_video_ai_mode">Reddit - 尽情探索</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1278408307484135545)** (8 条消息🔥): 

> - `用于 LLM 的 VRAM 和 RAM`
> - `用于 LLM 的 NPU vs GPU`
> - `用于 GPU 的 PCIE 5.0 x4` 


- **VRAM 和 RAM 决定 LLM 大小**: 你在台式机或笔记本电脑上可以运行的最大模型大小取决于 **VRAM** 和 **RAM** 的组合，其中 **RAM 的速度远慢于** **VRAM**。
- **NPU 不适合 LLM**: **NPU** 并非为快速 LLM 推理而设计，**无法与 GPU 相比**，即使是像 **GTX 1060** 这样过时的 GPU 也不行。
   - 一位成员引用了 **GTX 1060** 的 **Geekbench 6.3.0** 基准测试，结果显示即使是这张旧显卡在通用计算任务中的表现也优于现代 NPU。
- **用于 GPU 的 PCIE 5.0 x4**: 一位成员询问关于在 **PCIE 5.0 x4** 连接上使用 **3090** 的问题。
   - 他们想知道 **x4 模式** 是否提供了足够的带宽，还是需要 **x8**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://browser.geekbench.com/v6/compute/2673681">ASUSTeK COMPUTER INC. Strix GL703GM_GL703GM
 - Geekbench</a>: 未找到描述</li><li><a href="https://browser.geekbench.com/v6/compute/2673219">LENOVO 21N2S01T00
 - Geekbench</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1278067685183328298)** (101 messages🔥🔥): 

> - `Perplexity Pro Issues`
> - `Claude 3.5 Message Limit`
> - `Perplexity Image Upload Issues`
> - `Perplexity Search Quality`
> - `Perplexity vs ChatGPT` 


- **Perplexity Pro 用户对上传问题感到沮丧**：许多用户报告了上传文件和图片的困难，一些人表示尽管在某些浏览器中仍可访问，但他们失去了 Pro 订阅权限。
   - 几位用户对缺乏解决问题的信息和预计时间表表示不满，有些人使用了幽默的回应，例如一张坐在火堆前喝咖啡的狗的漫画，上面写着 "this is fine"（这很好）。
- **Claude 3.5 消息限制：每天 430 条**：Claude 3.5 和其他 Pro 模型的每日消息限制为 430 条，但 Opus 除外，其限制为 50 条。
   - 虽然不针对单一模型，但这种组合限制很少被用户达到，最接近的约为 250 条消息。
- **Perplexity 的图片上传问题与 AWS Rekognition 限制有关**：无法上传图片归因于达到了 Cloudinary（一种用于图像和视频管理的云服务）上的 AWS Rekognition 限制。
   - Perplexity 正在努力解决此问题，但尚未提供修复的预计时间表。
- **Perplexity 搜索质量辩论：Pro vs. ChatGPT**：一些用户报告 Perplexity 的搜索功能（尤其是 Pro 版）优于其他平台，理由是更好的来源引用和更少的幻觉。
   - 然而，其他人认为 ChatGPT 的自定义选项、RAG 和聊天 UX 更先进，而 Perplexity 的搜索较慢且功能较少，特别是与 ChatGPT 处理文件和记忆对话上下文的能力相比。
- **Perplexity 的 Chrome 扩展提供特定域名搜索**：Perplexity Chrome 扩展具有特定域名搜索功能，允许用户在特定网站内查找信息而无需手动搜索。
   - 这一功能被认为是一个显著优势，特别是在寻找特定域名或网站的信息时，并受到了一些用户的称赞。



**Link mentioned**: <a href="https://tenor.com/view/this-is-fine-gif-24177057">This Fine GIF - This Is Fine - Discover &amp; Share GIFs</a>: 点击查看 GIF

  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1278149955214643233)** (9 messages🔥): 

> - `Shareable Threads`
> - `WTF critical thinking`
> - `Claude Prompts`
> - `Australia's Right to Log Off`
> - `China's Renewable Energy` 


- **使 Thread 可共享**：一位用户请求另一位用户确保他们的 Thread 在 Discord 上设置为“可共享”。
- **新的批判性思维方法：WTF?**：一位用户整合了一种新的批判性思维方法，在 "WTF" 概念下结合了 26 种既定方法，涉及 "What, Where, Who, When, Why, and How" 等问题。
   - 该用户声称已在 NLP 中对该方法进行了编码，并乐于分享更多细节和 PDF。
- **Anthropic 的 Claude Prompts**：一位用户分享了 Anthropic 最近发布的 Claude Prompts 链接。
- **澳大利亚的“断联权” (Right to Log Off)**：一位用户提到澳大利亚的新立法赋予了员工在下班后断开联系的权利。
- **中国实现 2030 年可再生能源目标**：一位用户指出，中国已经实现了从各种来源产生可再生能源的 2030 年目标。



**Link mentioned**: <a href="https://www.youtube.com/embed/Rx1Zy9Nm5po">YouTube</a>: 未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1278141370569326663)** (3 messages): 

> - `Perplexity AI Hebrew implementation`
> - `Perplexity API citation feature beta` 


- **Perplexity AI 希伯来语实现 - 缺失链接和图片**：一位用户正尝试将 Perplexity AI 集成到希伯来语事实核查聊天机器人中，并遇到了链接和图片缺失的问题。
   - 该用户收到的回复比 Perplexity 搜索网站短，且包含错误或不存在的链接，以及 404 图片链接。
- **Beta 引用功能 - 等待批准**：另一位用户询问了加入 Perplexity Beta 计划的等待时间，特别是针对引用返回功能。
   - 他们提到在过去几个月中多次申请，但未收到任何回复。

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1278069387521425530)** (75 条消息🔥🔥): 

> - `DisTrO 效率`
> - `训练超大规模 LLM`
> - `DPO 训练`
> - `模型合并 (Model Merging)`
> - `Hermes 3 对比 Llama 3.1` 


- **DisTrO 的效率限制**：一名成员询问 DisTrO 是否可以用于在大量弱设备上训练超大规模 LLM（>100b 参数）。
   - 另一名成员回答说，DisTrO 在 DDP 下效率最高，而对于包含十亿部旧手机、笔记本电脑和台式机的用例，SWARM 可能更合适。
- **使用 AI 预测响应进行 DPO 训练**：一名成员询问在 DPO 训练中，使用 AI 模型预测的用户响应与用户实际响应相比的效果。
   - 他们推测这可能会导致更好的心智理论 (Theory of Mind)。
- **模型合并策略**：一名成员建议将 UltraChat 和基础 Mistral 之间的差异应用到 Mistral-Yarn 上，作为一种潜在的合并策略。
   - 尽管其他人表示怀疑，但该成员仍保持乐观，并引用了过去在他们所谓的“诅咒模型合并 (cursed model merging)”方面的成功尝试。
- **Hermes 3 表现优于 Llama 3.1**：一名成员分享了 Hermes 3 和 Llama 3.1 模型的对比，强调了 Hermes 3 在通用能力方面的竞争力，甚至更胜一筹。
   - 他们提供了基准测试链接，展示了 Hermes 3 与 Llama 3.1 相比的优势和劣势。
- **对 "Llama 3.1 Storm" 模型的担忧**：一名成员担心一个名为 "Storm" 的新 Llama 3.1 微调模型可能是通过窃取 Hermes 的 System Prompt 创建的。
   - 另一名成员确认，该模型的大部分数据实际上源自 Hermes 2。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/neurallambda/status/1828214178567647584?s=46">来自 neurallambda (open agi) (@neurallambda) 的推文</a>：关于“同构 AI (Homoiconic AI)”的进展报告：我们使用 hypernet 生成 autoencoder 的权重，然后进行上下文学习（掩码重构损失）以改进这些权重...</li><li><a href="https://arxiv.org/abs/2408.11029">Scaling Law with Learning Rate Annealing</a>：我们发现神经语言模型的交叉熵损失曲线在训练步数 ($s$) 内经验性地遵循带有学习率 (LR) 退火的缩放法则：$$L(s) = L_0 + A\cdot S_1^{-α} - C...</li><li><a href="https://n3rdware.com/accessories/single-slot-rtx-2000-ada-cooler">Nvidia RTX 2000 Ada 单插槽散热器 | n3rdware</a>：未找到描述</li><li><a href="https://huggingface.co/akjindal53244/Llama-3.1-Storm-8B">akjindal53244/Llama-3.1-Storm-8B · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/reneil1337/status/1828827624900272628">来自 reneil.eth 🕳🐇 (@reneil1337) 的推文</a>：Yo Hermes，很喜欢你对 @naturevrm 的看法 🤘 @NousResearch 必胜 🕳️🐇</li><li><a href="https://x.com/repligate/status/1828604853486014837?s=46">来自 j⧉nus (@repligate) 的推文</a>：Hermes 405b 太搞笑了。它经常表现得像是在疯狂中刚刚醒来，然后大喊“到底发生了什么”之类的话。</li><li><a href="https://arxiv.org/abs/1906.02107">Latent Weights Do Not Exist: Rethinking Binarized Neural Network Optimization</a>：二值化神经网络 (BNN) 的优化目前依赖于实值潜在权重来累积微小的更新步骤。在本文中，我们认为这些潜在权重不能被视为...</li><li><a href="https://www.reddit.com/r/duckduckgo/comments/1f2vrku/ai_assist_should_be_uncensored_and_generate/">Reddit - 深入探讨一切</a>：未找到描述</li><li><a href="https://tenor.com/view/shikanoko-by-murya-gif-9501555167387334429">Shikanoko By Murya GIF - SHIKANOKO BY MURYA - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1278071727665975397)** (23 条消息🔥): 

> - `使用合成数据进行微调`
> - `Hermes 3`
> - `在 CPU 上运行 Llama 3.1 8B`
> - `模型大小与 RAM`
> - `对话主题打标签` 


- **使用合成数据进行微调：一个新前沿**：普遍的误解是认为不建议使用另一个模型生成的数据进行微调，但目前的最前沿技术正高度关注合成数据训练。
   - 例子包括使用了大量合成数据的 Hermes 3，以及传闻中重度依赖合成数据的 "strawberry 模型"，但这需要一个良好的过滤流水线（filtering pipeline）。
- **在 CPU 上运行 Llama 3.1 8B：性能预期**：在没有 GPU 显卡的 VPS/裸机上运行 Llama 3.1 8B 需要 6GB RAM，使用 gguf 时性能较慢，约为每秒 5-10 个 token。
   - 8-bit 量化可能需要 12GB RAM，性能可能在每个 token 2-5 秒左右，但这并不一定是线性的。
- **模型大小与 RAM：实用指南**：8-bit 量化模型的大小约为原始模型参数量的一半，并额外需要 2GB 的开销。
   - 例如，一个 8B 模型在实际模型大小的基础上还需要大约 102GB 的开销。
- **为对话打主题标签：仅 Decoder 还是 Encoder-Decoder？**：对于没有固定类别、使用 2-3 个词为对话打主题标签的任务，可以使用 decoder-only transformer 或 encoder-decoder 架构。
   - 一个建议是微调任何 LLM，并使 LM head 仅输出所需数量的标签，可以参考 LMSys 竞赛获取潜在的代码示例。
- **在 ChatGPT 界面中解释 RAG 系统**：可以在 ChatGPT 界面的语境下解释简单的 RAG 系统：允许用户输入 PDF 链接，然后运行 RAG 机制。
   - 为了区分简单 RAG 系统和更高级的系统，应强调高级系统的额外功能和复杂性，例如整合更复杂的文档处理或检索方法。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 条消息): 

sunhao77: https://arxiv.org/abs/2408.11029
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1278110950871797771)** (4 条消息): 

> - `Flex-attention 可视化工具`
> - `微型 ASIC 矩阵乘法实现` 


- **可视化任何 Flex-Attention**：一个新工具现已可用，用于帮助可视化各种 Flex-attention 图，包括 Bigbird attention、causal attention 和 dilated sliding attention。
   - 该工具可通过以下链接访问：[https://viig99-app-demos-jz7hllm8n2ps6fkkwmotuj.streamlit.app/](https://viig99-app-demos-jz7hllm8n2ps6fkkwmotuj.streamlit.app/)
- **用于 1-bit LLM 的微型 ASIC**：一个 GitHub 仓库展示了一个用于矩阵乘法单元的微型 ASIC 实现，专门为 1-bit LLM 定制。
   - 该实现基于论文《The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits》，可以在 [https://github.com/rejunity/tiny-asic-1_58bit-matrix-mul](https://github.com/rejunity/tiny-asic-1_58bit-matrix-mul) 找到。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/rejunity/tiny-asic-1_58bit-matrix-mul">GitHub - rejunity/tiny-asic-1_58bit-matrix-mul: 针对《1-bit LLM 时代：所有大语言模型都是 1.58 Bits》矩阵乘法单元的微型 ASIC 实现</a>: 针对《1-bit LLM 时代：所有大语言模型都是 1.58 Bits》矩阵乘法单元的微型 ASIC 实现 - rejunity/tiny-asic-1_58bit-matrix-mul</li><li><a href="https://viig99-app-demos-jz7hllm8n2ps6fkkwmotuj.streamlit.app/">无标题</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 条消息): 

sunhao77: https://arxiv.org/abs/2408.11029
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/)** (1 条消息): 

draeician: 如果你不介意的话，我很想看看。
  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1278097775250378867)** (2 条消息): 

> - `OpenRouter API Degradation`
> - `Llama 3.1 405B Update` 


- **OpenRouter API Degradation**: OpenRouter 经历了约 5 分钟的 API 性能下降，但已发布补丁，目前该事件已恢复。
- **Llama 3.1 405B bf16 Endpoint**: [Llama 3.1 405B (base)](https://openrouter.ai/models/meta-llama/llama-3.1-405b) 已更新 bf16 endpoint。



**提到的链接**: <a href="https://openrouter.ai/models/meta-llama/llama-3.1-405b>)">Llama 3.1 405B (base) - API, Providers, Stats</a>: Meta 最新系列的模型 (Llama 3.1) 推出了多种尺寸和版本。通过 API 运行 Llama 3.1 405B (base)

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1278071588549431306)** (89 条消息🔥🔥): 

> - `Hyperbolic 的 BF16 Llama 405B`
> - `LMSys 排行榜`
> - `OpenRouter 的 DeepSeek Caching`
> - `OpenRouter 活动页面的柱状图`
> - `Gemini Flash-8B 性能表现` 


- **Hyperbolic 部署 BF16 Llama 405B Base**: Hyperbolic 发布了 **Llama 3.1 405B Base 模型**的 **BF16** 变体。
   - 这是在 **OpenRouter** 上现有的 **FP8 量化版本**之外新增的选择。
- **LMSys 排行榜：过时了？**: 用户讨论了 **LMSys 排行榜**。
   - 他们认为，由于像 **Gemini Flash** 这样表现异常出色的新模型出现，该排行榜的相关性可能正在下降。
- **OpenRouter 的 DeepSeek Caching：即将推出**: OpenRouter 正在努力添加对 **DeepSeek 的 Context Caching** 的支持。
   - 该功能预计可降低高达 **90%** 的 API 成本。
- **OpenRouter 活动页面柱状图无法加载**: 用户反馈**活动页面的柱状图**无法显示。
   - 此问题似乎影响特定账户，可能是由于前端 Bug 导致的。
- **Gemini Flash-8B：作为 8B 模型表现惊人**: 有用户表示对 **Gemini Flash-8B** 模型的性能印象深刻。
   - 他们指出其表现可与更大版本的 **Flash** 相媲美，尤其是在**多语言能力**方面令人惊叹。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.promptfoo.dev/">安全可靠的 LLM | promptfoo</a>: 通过 30,000 名开发者使用的 AI 红队测试和评估消除风险。发现并修复漏洞，最大化输出质量，捕捉回归问题。</li><li><a href="https://openrouter.ai/activity">活动 | OpenRouter</a>: 查看您在 OpenRouter 上使用模型的情况。</li><li><a href="https://docs.helicone.ai/getting-started/integration-method/openrouter">OpenRouter 集成 - Helicone 开源 LLM 可观测性</a>: 未找到描述</li><li><a href="https://www.goody2.ai">GOODY-2 | 全球最负责任的 AI 模型</a>: 推出具有下一代伦理对齐的新型 AI 模型。立即聊天。</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-405b-inst">Llama 3.1 405B (base) - API、提供商、统计数据</a>: Meta 最新的模型系列 (Llama 3.1) 推出了多种尺寸和版本。通过 API 运行 Llama 3.1 405B (base)</li><li><a href="https://openrouter.ai/docs/responses#querying-cost-and-stats">响应 | OpenRouter</a>: 管理来自模型的响应</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-405b-instruct">Llama 3.1 405B Instruct - API、提供商、统计数据</a>: 备受期待的 400B 级 Llama3 来了！Meta AI 团队拥有 128k 上下文和令人印象深刻的评估分数，继续推动开源 LLM 的前沿。Meta 最新的...</li><li><a href="https://platform.deepseek.com/api-docs/news/news0802/">DeepSeek API 推出磁盘 Context Caching，将价格降低一个数量级 | DeepSeek API 文档</a>: 在大型语言模型 API 使用中，很大一部分用户输入往往是重复的。例如，用户 Prompt 经常包含重复的引用，而在多轮对话中，之前的...</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-405b">Llama 3.1 405B (base) - API、提供商、统计数据</a>: Meta 最新的模型系列 (Llama 3.1) 推出了多种尺寸和版本。通过 API 运行 Llama 3.1 405B (base)</li><li><a href="https://x.com/hyperbolic_labs/status/1828481468156518691">来自 Hyperbolic (@hyperbolic_labs) 的推文</a>: BF16 版本的 Llama 3.1 405B Base：现已在 Hyperbolic 上线 🦙💜 Base 模型比指令微调模型更具创造力和能力，但直到现在它们一直未被充分利用。➡️ 开始使用...</li><li><a href="https://platform.deepseek.com/api-docs">您的首次 API 调用 | DeepSeek API 文档</a>: DeepSeek API 使用与 OpenAI 兼容的 API 格式。通过修改配置，您可以使用 OpenAI SDK 或兼容 OpenAI API 的软件来访问 DeepSeek API。
</li>
</ul>

</div>

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1278109379438051329)** (11 条消息🔥): 

> - `Llama 3.1 405B API`
> - `TRL.X`
> - `Model Training Data`
> - `Monte Carlo Tree Search`
> - `Computer Vision Research` 


- **免费 Llama 3.1 405B API**：一位成员分享了 [Sambanova.ai](https://sambanova.ai/fast-api?api_ref=444868) 的链接，该网站提供免费且有速率限制的 API，用于运行 Llama 3.1 405B、70B 和 8B。
   - 该 API 与 OpenAI 兼容，允许用户使用自己微调的模型，并提供入门套件和社区支持以加速开发。
- **TRL.X 已过时**：一位成员指出 TRL.X 已经严重过时（depreciated），且很长时间没有更新了。
   - 另一位成员询问它是否仍在维护，或者是否有替代方案。
- **模型训练数据选项**：一位成员询问人们使用什么样的数据来训练 LLM。
   - 他们想知道人们是使用私有数据集，还是使用像 Alpaca 这样的公开数据集，然后应用自定义的 DPO 或其他无监督技术来提高性能，或者只是在非指令微调（non-instruction tuned）模型上进行 N-shot 基准测试。
- **反向 Monte Carlo Tree Search**：一位成员建议训练一个模型来反向执行 Monte Carlo Tree Search。
   - 他们提议利用图像识别和生成技术来生成最优的树搜索选项，而不是去识别它。
- **Computer Vision 研究与论文反馈**：一位成员分享说他们正在进行一个 Computer Vision 扩散模型项目，并正在寻求对其论文草稿的反馈。
   - 他们提到大规模测试成本很高，并请求帮助寻找可以审阅其工作的人。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://sambanova.ai/fast-api?api_ref=444868">获取快速且免费的 AI 推理 API | SambaNova Systems</a>：使用 SambaNova 的免费 API，通过极速推理为您的 AI 应用赋能。利用尖端的 RDU 芯片技术体验 AI 的未来。</li><li><a href="https://muhammadnaufil.com).">未找到标题</a>：未找到描述
</li>
</ul>

</div>

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1278067336800374834)** (73 messages🔥🔥): 

> - `LR scaling with batch size`
> - `Adam vs SGD`
> - `MiniCPM paper`
> - `Infinite LRs`
> - `AdamWScale` 


- **Adam LR 缩放：平方根缩放？**：一位成员提出，对于 Adam，学习率 (LR) 应该随 Batch Size 的平方根进行缩放。
- **MiniCPM 的无限 LR 方法**：针对 MiniCPM 论文中关于无限学习率 (infinite learning rates) 的方法展开了讨论。
- **AdamWScale 与初始化**：一位成员询问了初始化论文的重要性，特别是一篇他们认为被忽视的论文，以及是否应该考虑权重衰减 (weight decay) 和 Adam betas 等超参数。
- **临界 Batch Size 与数据分布偏移**：在预训练模型微调 (finetuning) 的背景下，讨论了临界 Batch Size 的概念，这是影响训练性能的关键因素。
- **LLMs Token 预测：等学习定律 (Equi-learning Law)**：分享了一篇论文，该论文引入了一项定律，旨在揭示 LLM 内部如何提高其预测 Next-Token 的能力。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/iscienceluvr/status/1828617875432841490?s=46">来自 Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr) 的推文</a>：Diffusion Models Are Real-Time Game Engines  摘要：https://arxiv.org/abs/2408.14837 项目主页：https://gamengen.github.io/  Google 发布了 GameNGen，这是第一个完全由神经驱动的游戏引擎...</li><li><a href="https://arxiv.org/abs/2408.15237">The Mamba in the Llama: Distilling and Accelerating Hybrid Models</a>：线性 RNN 架构（如 Mamba）在语言建模方面可以与 Transformer 模型竞争，同时具有优越的部署特性。鉴于目前对训练大规模 Tran...</li><li><a href="https://arxiv.org/abs/2408.13442">A Law of Next-Token Prediction in Large Language Models</a>：大语言模型 (LLMs) 已被广泛应用于各个领域，但其黑盒性质对理解这些模型如何处理输入数据构成了重大挑战...</li><li><a href="https://proceedings.neurips.cc/paper/2019/hash/e0eacd983971634327ae1819ea8b6214-Abstract.html">Which Algorithmic Choices Matter at Which Batch Sizes? Insights From a Noisy Quadratic Model</a>：未找到描述</li><li><a href="https://arxiv.org/abs/1812.06162">An Empirical Model of Large-Batch Training</a>：在越来越多的领域中，已经证明深度学习模型可以在不牺牲数据效率的情况下使用相对较大的 Batch Size 进行训练。然而，这种做法的极限...</li><li><a href="https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/">How to Scale Hyperparameters as Batch Size Increases</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/zhangirazerbayev/ocwcourses?row=0">zhangirazerbayev/ocwcourses · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://github.com/pytorch/pytorch/issues/41508">nn.MultiheadAttention causes gradients to become NaN under some use cases · Issue #41508 · pytorch/pytorch</a>：🐛 Bug 在某些使用场景下，将 key_padding_mask 和 attn_mask 与 nn.MultiheadAttention 一起使用会导致梯度变为 NaN。重现步骤：通过 nn.Mu 进行反向传播.....
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1278083379505528935)** (3 messages): 

> - `LlamaIndex`
> - `RAG`
> - `Workflows`
> - `Query Engine`
> - `Developer Competition` 


- **LlamaIndex RAG 应用教程**：由 Wassim Chegham 撰写的综合指南，引导你使用 LlamaIndex 和 Azure OpenAI 构建 Serverless RAG 应用程序。
   - 该指南教授如何利用自有业务数据来增强响应，并提供对 RAG 架构和 LlamaIndex 的深入理解。
- **LlamaIndex Router Query Engine 工作流**：来自 Ravitej Ads 的新 Workflows 教程，展示了利用 Workflows 强大功能重构的 Router Query Engine。
   - 该教程涵盖了将用户查询路由到合适的查询引擎（包括向量、摘要或基于关键词的引擎），并提供了对该过程的宝贵见解。
- **LlamaIndex & Nvidia 开发者竞赛**：由 LlamaIndex 和 Nvidia 联合举办的开发者竞赛，提供超过 9000 美元的现金奖励、硬件、额度等。
   - 为了获胜，开发者必须使用 LlamaIndex 和 Nvidia 技术构建创新的生成式 AI 应用程序，探索 RAG、Agentic 或两者的结合。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1278090834943938680)** (66 条消息🔥🔥): 

> - `LlamaIndex 对 OpenAI 模型的支持`
> - `LlamaIndex 的 openai 库更新`
> - `LlamaIndex 的 pydantic v2 破坏性变更`
> - `GraphRAG 身份验证错误`
> - `用于 NL 到 SQL 的 Multi-Agent 系统` 


- **LlamaIndex 不支持 OpenAI gpt-4o-mini**：一位成员询问 LlamaIndex 的 `llama_index.llms.openai` 库是否支持使用 `gpt-4o-mini` OpenAI 模型。
   - 另一位成员确认目前不支持该模型。他们指出错误消息显示支持的模型包括 `gpt-4`, `gpt-4-32k`, `gpt-4-1106-preview`, `gpt-4-0125-preview`, `gpt-4-turbo-preview`, `gpt-4-vision-preview`, `gpt-4-1106-vision-preview`, `gpt-4-turbo-2024-04-09`, `gpt-4-turbo`, `gpt-4o`, `gpt-4o-2024-05-13`, `gpt-4-0613`, `gpt-4-32k-0613`, `gpt-4-0314`, `gpt-4-32k-0314`, `gpt-3.5-turbo`, `gpt-3.5-turbo-16k`, `gpt-3.5-turbo-0125`, `gpt-3.5-turbo-1106`, `gpt-3.5-turbo-0613`, `gpt-3.5-turbo-16k-0613`, `gpt-3.5-turbo-0301`, `text-davinci-003`, `text-davinci-002`, `gpt-3.5-turbo-instruct`, `text-ada-001`, `text-babbage-001`, `text-curie-001`, `ada`, `babbage`, `curie`, `davinci`, `gpt-35-turbo-16k`, `gpt-35-turbo`, `gpt-35-turbo-0125`, `gpt-35-turbo-1106`, `gpt-35-turbo-0613`, `gpt-35-turbo-16k-0613`。
- **LlamaIndex 的 OpenAI 库更新？**：一位成员报告在使用 LlamaIndex 时遇到与 `gpt-4o-mini` OpenAI 模型相关的错误。
   - 建议他们更新 `llama-index-llms-openai` 库以解决此问题。
- **Pydantic v2 破坏 LlamaIndex**：一位成员遇到了与 LlamaIndex `v0.11` 和 `pydantic v2` 相关的问题，LLM 对 `pydantic` 结构产生了幻觉。
   - 他们分享了 GitHub 上的 Issue 链接，并表示修复程序正在开发中。
- **GraphRAG 身份验证错误**：一位成员在使用 GraphRAG 配合自定义网关与 OpenAI API 交互时遇到了身份验证错误。
   - 问题追溯到 GraphRAG 实现中直接进行的 OpenAI API 调用。建议他们使用 `OpenAILike` 类来解决此问题。
- **用于 NL 到 SQL 到 NL 聊天机器人的 Multi-Agent 系统**：一位成员寻求关于使用 LlamaIndex 工具构建 Multi-Agent 系统以驱动 NL-to-SQL-to-NL 聊天机器人的指导。
   - 建议他们考虑使用 Workflows 或 ReAct Agents，但未给出明确的推荐。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/api_reference/llms/openai_like/#llama_index.llms.openai_like.OpenAILike">Openai like - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/pull/15679">fix tool schemas by logan-markewich · Pull Request #15679 · run-llama/llama_index</a>：pydantic v2 最近的更改导致我们的工具 JSON Schema 遗漏了重命名的 &quot;definitions&quot; 部分</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/embeddings/custom_embeddings/">Custom Embeddings - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1278345551691120780)** (8 条消息🔥): 

> - `QLoRA 分布式训练`
> - `FSDP1/2`
> - `torch.compile`
> - `Liger Kernels`
> - `chunkedCE` 


- **QLoRA 和 FSDP1 不兼容**：一位用户询问关于使用 QLoRA 进行分布式微调的问题，并被告知它与 FSDP1 不兼容。
- **Torch.compile 对比 Liger Kernels**：一位用户询问 Liger Kernels 在 Torchtune 中的实用性，一位成员回答说他们更倾向于使用 `torch.compile`。
- **FSDP2 的要求**：一位用户询问使用 FSDP2 所需的 PyTorch 版本。


  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1278328070511788052)** (49 条消息🔥): 

> - `Torch.compile 性能`
> - `Activation checkpointing (AC)`
> - `模型编译粒度`
> - `模型构建与 KV cache 长度`
> - `Torchtune PRs` 


- **Torch.compile 性能：整网编译 vs. 逐层编译**：讨论围绕 `torch.compile` 应用于整个模型与应用于单个层时的性能差异展开。
- **Activation Checkpointing 及其对编译的影响**：发现 Activation checkpointing (AC) 会显著影响编译性能。
- **模型编译粒度：平衡速度与优化**：讨论了模型编译的粒度，不同层级的粒度会影响性能和优化潜力。
- **KV cache 长度与模型构建**：模型中 key-value (KV) cache 的长度成为讨论点，指出需要为 encoder 和 decoder 组件分别定义。
- **Torchtune PRs 与待处理问题**：提到了与这些讨论相关的几个 Pull Requests (PRs)，包括用于逐层编译的 PR #1419、用于分块交叉熵损失（chunked cross-entropy loss）的 PR #1427，以及将参数解析迁移到配置文件的 PR #1423。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/blob/7b40daa19bfb90c11c8de2a1f74af8147b2fd016/torchtune/modules/transformer.py#L364">torchtune/torchtune/modules/transformer.py at 7b40daa19bfb90c11c8de2a1f74af8147b2fd016 · pytorch/torchtune</a>：一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号来为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/blob/05aeb71bec637f26a5b14e16638d445c1f7a8deb/recipes/full_finetune_single_device.py#L348-L352">torchtune/recipes/full_finetune_single_device.py at 05aeb71bec637f26a5b14e16638d445c1f7a8deb · pytorch/torchtune</a>：一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号来为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pbontrager/torchtune/blob/ade22da4bd66ddc54b0c2aa7a76f28a079eb00b3/torchtune/models/flamingo/_component_builders.py#L233)">torchtune/torchtune/models/flamingo/_component_builders.py at ade22da4bd66ddc54b0c2aa7a76f28a079eb00b3 · pbontrager/torchtune</a>：一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号来为 pbontrager/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/pull/1423">Move argparse to config by RdoubleA · Pull Request #1423 · pytorch/torchtune</a>：上下文：此 PR 的目的是什么？是添加新功能、修复错误、更新测试和/或文档，还是其他（请在此处添加）。TuneArgumentParser 从根本上说是一个配置工具，并且...</li><li><a href="https://github.com/pytorch/torchtune/pull/1419/files">Add per-layer compile support to recipes by yf225 · Pull Request #1419 · pytorch/torchtune</a>：为我们的单设备 LoRA、单设备 full finetune 和 FSDP2 LoRA recipes 启用逐层编译。FSDP2 full finetune 将在后续工作中完成。结果：所有 recipes 均以三次运行...
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1278082055636385975)** (31 条消息🔥): 

> - `LLM Hallucination`
> - `GPT-4 vs. Mini`
> - `SearchGPT vs. Perplexity`
> - `AI Sentience and Emotions`
> - `Orion Model Access` 


- **GPT-4 会自信地给出错误的幻觉答案**：一位成员讨论了 GPT-4 即使在被纠正后，仍会自信地提供错误答案的挑战。
   - 小组辩论了缓解这一问题的策略，建议使用特定的网页搜索指令进行 Prompt 提示，使用像 Perplexity 这样预配置的 LLM，以及可能设置一个带有网页搜索指令的 Custom GPT。
- **Mini 比 GPT-4 更便宜且更好？**：一位成员指出，在某些场景下，Mini 模型比 GPT-4 更便宜，且表现似乎更好。
   - 该成员认为这主要是人类偏好的问题，而且所使用的 Benchmark（基准测试）并不能反映人们真正关心的使用场景。
- **用于网页搜索的 SearchGPT vs. Perplexity**：一位成员询问了 Perplexity 相比 SearchGPT 的优势。
   - 另一位成员回答说，他们还没有尝试过 Perplexity，但认为 SearchGPT 很准确，偏见极小，非常适合复杂的搜索。
- **AI 的感知能力与情感**：一位成员讨论了 AI 体验类似人类情感的想法，认为将此类体验归因于 AI 可能是一种误解。
   - 他们表示，将理解 AGI 视为需要类人情感可能是一个不切实际的期望。
- **Orion 模型访问权限的担忧**：一位成员对限制私营部门访问 Orion 模型（如 Orion-14B-Base 和 Orion-14B-Chat-RAG）的潜在后果表示担忧。
   - 他们认为这可能会加剧不平等、扼杀创新并限制更广泛的社会效益，可能导致技术进步仅为精英利益服务的未来。



**提到的链接**：<a href="https://x.com/TheTechOasis1/status/1827394026808418799">来自 Ignacio de Gregorio (@TheTechOasis1) 的推文</a>：http://x.com/i/article/1827379585861709824

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1278069363584401432)** (20 条消息🔥): 

> - `ChatGPT 4 vs ChatGPT 3.5`
> - `OpenAI and Google's Data Scraping`
> - `Custom GPT's memory feature`
> - `GPTs following multi-step instructions`
> - `Llama 3.1 vs ChatGPT` 


- **ChatGPT 4 的准确性引发辩论**：一位成员认为 ChatGPT 4 并不比 ChatGPT 3.5 好多少，并且 OpenAI 一直在使用工具抓取 YouTube 内容作为训练数据。
- **OpenAI 和 Nvidia 的数据抓取**：据报道，OpenAI 和 Nvidia 都在未经许可的情况下从网站抓取数据。
- **Custom GPT 的记忆功能**：一位成员询问了 Custom GPT 上 Memory 功能的可用性。
- **GPT 在处理多步指令时表现挣扎**：一位成员报告称，在没有用户干预的情况下，很难让 GPT 遵循多步指令。
- **Llama 3.1 vs ChatGPT - 性能对比**：一位成员将 Llama 3.1 8B 与 400B 模型进行了比较，发现较小的模型没有意义，因为较大的模型需要 200GB RAM 的集群。


  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1278077902902137006)** (13 messages🔥): 

> - `Gemini 1.5 Flash-8B`
> - `Gemini 1.5 Pro`
> - `Gemini 1.5 Flash`
> - `Aistudio`
> - `RewardBench` 


- **Google 发布三款全新 Gemini 模型**：Google 宣布发布三款实验性 Gemini 模型：**Gemini 1.5 Flash-8B**、**Gemini 1.5 Pro** 和 **Gemini 1.5 Flash**。
   - 这些模型可以在 [Aistudio](https://aistudio.google.com) 上进行访问和实验。
- **Gemini 1.5 Pro：编程与复杂提示词**：**Gemini 1.5 Pro** 模型被强调在**编程和复杂提示词 (complex prompts)** 方面具有改进的能力。
   - 原始的 Gemini 1.5 Flash 模型也得到了**显著提升**。
- **API 访问与 RewardBench**：讨论围绕 **API 的可用性**展开，一位用户对其功能缺失表示沮丧。
   - 该用户还提到他们尝试在 **RewardBench 上评估 8B 模型**，但认为那是一个虚假的 Benchmark。
- **Q* 假设与 OpenAI 的发布策略**：一位用户询问了 **Q* 假设**的相关性及其与**蒙特卡洛树搜索 (MCTS)** 的联系。
   - 该用户对 OpenAI 最近的发布策略表示怀疑，并暗示今年晚些时候可能会有**更重大的发布**。



**提到的链接**：<a href="https://x.com/officiallogank/status/1828480081574142227?s=46">Logan Kilpatrick (@OfficialLoganK) 的推文</a>：今天，我们推出了三个实验性模型：- 一个新的较小变体 Gemini 1.5 Flash-8B - 一个更强大的 Gemini 1.5 Pro 模型（在编程和复杂提示词上表现更好）- 一个显著改进的 Gem...

  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1278090866812256408)** (1 messages): 

> - `AI Art Accessibility`
> - `AI Art as the Path Forward` 


- **AI 艺术可访问性担忧**：一位用户对未来可能只有富有且人脉广泛的个人才能获得最佳 AI 艺术工具的情况表示担忧。
   - 这种担忧源于一种恐惧，即即使 AI 艺术被接受为前进的方向，它也可能加剧艺术界现有的不平等。
- **AI 艺术是前进之路吗？**：该用户对艺术的未来以及 AI 生成艺术成为主导形式的可能性提出了质疑。
   - 这引发了关于传统艺术家的角色以及在日益由 AI 驱动的世界中人类创造力价值的问题。


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1278076280537092106)** (4 messages): 

> - `Gemini API`
> - `Gemini API Rate Limits`
> - `Gemini API Model Availability` 


- **Gemini API：速率限制是个谜**：一位用户对 Gemini API 的速率限制表示沮丧，将其描述为“极其随机且糟糕”。
   - 他们报告说某些模型可以工作而其他模型不行，这使得有效使用 API 变得困难。
- **Gemini API 用户寻求指导**：一位用户询问了 Gemini API 的使用情况，特别是询问其他人是否在使用它。
   - 他们似乎在速率限制和模型可用性方面遇到了困难。


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1278160349551398943)** (15 messages🔥): 

> - `SnailBot`
> - `Open Source`
> - `Data Availability`
> - `Fair Use` 


- **SnailBot 的通知**：一位用户分享说，当使用 [livenow.youknow](https://livenow.youknow.com/) 缩短链接时，[SnailBot](https://www.snailbot.com/) 会在他们收到电子邮件之前通过 Discord 通知他们。
   - 他们还注意到 SnailBot 没有识别出 URL 的更改，这表明该工具有其局限性。
- **Open Source 辩论与代码许可的相似之处**：一位用户预测，关于数据可用性的 Open Source 辩论将遵循与代码许可现状类似的轨迹。
   - 他们认为，虽然 [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) 许可证被认为是 Open Source，但经常被避开，而 [MIT](https://opensource.org/licenses/MIT) 许可证被认为更受欢迎。他们认为数据可用性辩论将出现类似趋势，即数据可用性被认为是有益的，但对大多数用户来说最终是可选的。
- **数据可用性与 Fair Use**：一位用户评论了之前关于数据可用性的讨论，表达了这样一种观点：数据易于获取是有益的，但对大多数人来说并非必不可少。
   - 这与他们认为 Fair Use 是 Open Source 数据共享的一个关键方面的信念一致，并暗示未来可能会有一篇文章探讨这一观点。


  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1278111284730003518)** (4 messages): 

> - `Cat gifs` 


- **浴缸里的猫 GIF**: 一张 GIF 显示一只猫站在浴缸里，爪子伸开，毛发蓬乱。
   - 这只猫正随着一首不知名的歌曲跳舞，给人一种幽默而古怪的感觉。
- **更多猫咪 GIF 链接**: 聊天中包含了各种以猫为主题的 GIF 链接，包括跳舞的猫、搞笑的猫以及一般的猫咪内容。



**提及的链接**: <a href="https://tenor.com/view/dance-dancing-dancing-cat-cat-cat-dance-gif-4990417705814603993">Dance Dancing GIF - Dance Dancing Dancing cat - Discover &amp; Share GIFs</a>: 点击查看 GIF

  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1278105523404804116)** (25 messages🔥): 

> - `Langchain + Cohere API Errors`
> - `Cohere API Response Errors`
> - `Token Counting for Cohere API`
> - `Aya-23-8b Inference Time`
> - `Model Quantization` 


- **Langchain + Cohere API 错误**: 有用户报告在使用 Langchain 和 Cohere TypeScript 对 Cohere API 进行后续调用时收到 404 错误。
- **Cohere API 响应错误**: 错误消息指示为“非 JSON”响应，这表明 Cohere API 返回了 404 页面而不是 JSON 对象。
- **Cohere API 的 Token 计数**: 用户询问如何计算文本输入中的 Token 数量，这对于理解使用 Cohere API 时的 Token 限制非常重要。
- **Aya-23-8b 推理时间**: 一位用户询问 Aya-23-8b 模型是否能在 500 毫秒内完成约 50 个 Token 的推理。
- **模型量化 (Model Quantization)**: 一位用户建议对模型进行量化可能有助于实现更快的推理时间。



**提及的链接**: <a href="https://docs.cohere.com/reference/tokenize">Tokenize — Cohere</a>: 该端点使用字节对编码 (BPE) 将输入文本拆分为称为 Token 的更小单元。要了解有关 Tokenization 和字节对编码的更多信息，请参阅 Token 页面。

  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1278244252996079701)** (3 messages): 

> - `Persian Tourist Attractions`
> - `Next.js App`
> - `Cohere AI`
> - `Google Places API` 


- **为波斯旅游建议推出的 Next.js 应用**: 推出了一个 Next.js 应用，它结合了 **Cohere AI** 和 **Google Places API**，用**波斯语**推荐旅游景点。
   - 该应用提供详细信息，包括描述、地址、坐标和照片，且语法水平极高。
- **应用功能与特性**: 该应用利用 **Cohere AI** 和 **Google Places API** 的强大功能，提供准确且引人入胜的**波斯语**旅游建议。
   - 用户可以探索旅游景点并获取详细信息，包括描述、地址、坐标和照片，所有内容均以高质量的波斯语格式呈现。
- **社区反馈与托管**: 社区的几位成员对试用该应用表现出浓厚兴趣，称赞其功能和创新方法。
   - 开发者被鼓励托管该应用以提高可访问性，让社区能够亲身体验其优势。
- **分享与协作**: 该应用已在 **GitHub** 和 **Medium** 上公开分享，邀请社区提供反馈和协作。
   - 开发者鼓励社区成员探索代码、提供反馈并为项目的开发做出贡献。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1278128886567407637)** (8 messages🔥): 

> - `Mojo Circular Imports`
> - `Python Circular Imports`
> - `Mojo Compiler Optimization`
> - `Mojo Top Level Statements` 


- **Mojo 处理循环导入**：Mojo 是一种编译型语言，因此它可以在编译前扫描每个文件并确定 `struct` 的形状。
   - 编译器随后可以解析其余操作，因为它拥有实现 `to_queue` 所需的所有函数。
- **Python 在循环导入方面的问题**：在 Python 中，循环导入可能会导致问题，因为它在编译期间按顺序运行所有内容。
   - Mojo 处理循环导入的方法避免了这些问题，因为它不需要按顺序运行所有内容。
- **Mojo 编译器优化 Struct 大小**：Mojo 编译器知道指针的大小，因此它可以直接确定 `List` 的形状，而无需查看 `Queue`。
   - 这允许编译器通过使用 `Arc` 或其他类型的指针来打破循环，从而解析循环导入。
- **Mojo 关于 Top Level Statements 的未来**：Mojo 目前还没有顶级语句，但预计会通过在 `main` 开始前按导入顺序运行顶级代码来处理循环导入。
   - 这将确保循环导入得到正确且高效的解析。



**Link mentioned**: <a href="https://modul.ar/user-feedback">Appointments</a>: no description found

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1278109421788201097)** (21 messages🔥): 

> - `Mojo performance`
> - `Mojo named return slots`
> - `Mojo non-movable types`
> - `Mojo's Ownership Model`
> - `Mojo debugging` 


- **Mojo 异常的性能曲线**：一位用户观察到在约 1125 个字段时性能急剧上升，推测可能是 **smallvec** 或 **arena** 溢出。
   - 另一位用户建议可能是由于 **1024 参数** 的单个文件引起的。
- **Mojo 中的 Named Return Slots**：Mojo 支持命名返回槽，允许使用类似 `fn thing() -> String as result: result = "foo"` 的语法。
   - 语法可能会发生变化，但该功能似乎旨在用于被调用者框架（callee frame）内的 "placement new"。
- **Mojo 中的 Non-movable Types**：Mojo 允许不可移动的类型，例如 `fn foo(a: Int) -> NonMovable:`。
   - 这些类型被编译为等同于 C++ 的 `void foo(int, NonMovable&)`，调用者传递用于初始化的位置。
- **Mojo 所有权模型详解**：Mojo 非常智能且严谨，通常会避免隐式移动并执行 copy->move 优化。
   - 一篇博客文章和视频深入探讨了 Mojo 中的 Ownership 概念，为理解内存管理提供了心理模型。
- **Mojo 的调试策略**：调试是 Mojo 和 MAX 的首要任务，旨在提供比传统 Python、C++ 和 CUDA 技术栈更好的调试体验。
   - 机器学习通常需要在长时间运行的过程后检查程序状态，这需要针对该特定领域量身定制的调试功能。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.modular.com/blog/what-ownership-is-really-about-a-mental-model-approach">Modular: What ownership is really about:  a mental model approach</a>: Ownership 是 Mojo 等现代编程语言中的一个著名概念，旨在提供安全的内存管理编程模型，同时确保高性能。这使得程序...</li><li><a href="https://www.modular.com/blog-all?topic=Developer">Blog-All</a>: 在 Modular，我们相信优秀的文化是创建伟大公司的关键。我们工作的三个支柱是：打造用户喜爱的产品、赋能于人、以及成为一支不可思议的团队。
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1278084511946506292)** (19 条消息🔥): 

> - `Gemini 1.5 Flash Models`
> - `Anthropic's Claude 3.5 Sonnet`
> - `Artifacts on iOS and Android`
> - `Cartesia's Sonic`
> - `Cerebras Inference` 


- **Google 推出全新 Gemini 模型**：Google 宣布发布三个实验性模型：Gemini 1.5 Flash-8B（一个更小的变体）、Gemini 1.5 Pro（在编程和复杂提示词方面更强）以及显著改进的 Gemini 1.5 Flash 模型。
   - 用户可以在 https://aistudio.google.com 试用这些模型。
- **Anthropic 的 Claude 3.5 Sonnet：编程领域的有力竞争者**：Anthropic 于 6 月发布的最新语言模型 Claude 3.5 Sonnet，因其在编程相关任务中优于 ChatGPT 的表现，在软件工程师中势头强劲。
   - 这标志着 LLM 能力的领导地位可能正从 OpenAI 转移，Anthropic 正处于领先地位。
- **Artifacts 登陆移动端**：Artifacts，一个由 Anthropic 开发的项目，已在 iOS 和 Android 上线。
   - 此次发布将 LLM 的力量带到了移动应用中，允许用户使用 Claude 实时创建简单的游戏。
- **Cartesia 的 Sonic：端侧 AI 革命**：专注于开发普及化 AI 的 Cartesia 宣布了其第一个里程碑：Sonic，全球最快的生成式语音 API。
   - 这一进步旨在将 AI 带到每个设备上，实现尊重隐私且快速的全球交互，改变机器人、游戏和医疗等各个领域的应用。
- **Cerebras 推理：极速 AI**：Cerebras 推出了其推理解决方案，展示了 AI 处理速度的显著提升。
   - 该解决方案采用定制硬件并结合了多种内存技术，速度高达 1800 tokens/s，在速度和设置简易性方面均超过了 Groq 的表现。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/officiallogank/status/1828480081574142227?s=46">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>: 今天，我们正在推出三个实验性模型：- 一个新的更小变体，Gemini 1.5 Flash-8B - 一个更强大的 Gemini 1.5 Pro 模型（在编程和复杂提示词方面表现更好）- 一个显著改进的 Gem...</li><li><a href="https://cerebras.vercel.app/">Cerebras Voice</a>: 未找到描述</li><li><a href="https://x.com/alexalbert__/status/1828502920788103363?s=46">来自 Alex Albert (@alexalbert__) 的推文</a>: 我们今天在 iOS 和 Android 上发布了 Artifacts！我整个上午都在用 Claude 复制简单的游戏。我们正接近由 LLM 实时创建移动应用的时代。</li><li><a href="https://cartesia.ai/blog/2024-08-27-on-device">Cartesia</a>: 未找到描述信息</li><li><a href="https://newsletter.pragmaticengineer.com/p/how-anthropic-built-artifacts">Anthropic 如何构建 Artifacts</a>: Artifacts 背后的团队——一种与 Claude 交互的创新方式——分享了他们如何在一个分布式团队中仅用三个月时间构建出这一创新功能。独家细节。</li><li><a href="https://lu.ma/ls">Latent Space (Paper Club &amp; 其他活动) · 活动日历</a>: 在 Luma 上查看并订阅 Latent Space (Paper Club &amp; 其他活动) 的活动。Latent.Space 活动。请点击日历右上方 RSS 图标将其添加到您的日历中。
</li>
</ul>

</div>
  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1278077646298943489)** (11 messages🔥): 

> - `interpreter custom_instructions`
> - `emit images in jupyter`
> - `jupyterbook metadata`
> - `cython code`
> - `openinterpreter development` 


- **interpreter.custom_instructions 问题修复**：一位成员建议将 `interpreter.custom_instructions` 设置为字符串，使用 `str(" ".join(Messages.system_message))` 而不是列表。
   - 他们表示这可能会解决该问题。
- **在 Jupyter 中关闭图像显示**：一位成员请求帮助关闭 Jupyter 默认显示图像的行为。
   - 他们希望找到一种方法来防止图像在 Jupyter 中自动显示。
- **使用 Python 代码向 Jupyter Book 添加元数据**：一位成员分享了关于使用 Python 代码向 notebook 添加元数据的 Jupyter Book 文档链接。
   - 该文档提供了如何向 Jupyter Book 内各种类型的内容添加元数据的指导。
- **Black-Scholes 模型的 Cython 代码示例**：一位成员分享了一个使用 Black-Scholes 模型进行期权定价的 Cython 代码示例。
   - 该代码利用 Cython 的特性进行了速度和效率方面的优化。
- **OpenInterpreter 开发更新**：一位成员确认开发工作仍在进行中，并分享了 GitHub 上 OpenInterpreter 主仓库的链接。
   - 提交历史（commit history）表明了活跃的开发进展和来自社区的贡献。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://jupyterbook.org/en/stable/content/metadata.html#add-tags-using-python-code">Add metadata to your book pages</a>：未找到描述</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/commits/main/">Commits · OpenInterpreter/open-interpreter</a>：计算机的自然语言界面。通过在 GitHub 上创建账号来为 OpenInterpreter/open-interpreter 的开发做出贡献。</li><li><a href="https://nbviewer.org/github/ipython/ipython/blob/1.x/examples/notebooks/Cell%20Magics.ipynb">Jupyter Notebook Viewer</a>：未找到描述
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1278078691624353893)** (4 messages): 

> - `01 Design & Research`
> - `Pre-order Status` 


- **01 品牌文档与设计进展**：最接近品牌文档的是第一次见面会的演示文稿，可以点击 [此处](https://www.canva.com/design/DAF8rbBol3Q/UNivuf8sjxVSveDfMFWpag/edit?utm_content=DAF8rbBol3Q&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) 查看。
   - 更详尽的工业设计进展和文档将在未来几天内发布在 GitHub 的 `hardware/light/manufacturing` 路径下。
- **预订状态更新**：预计很快会发布预订状态的更新。
   - 最近的更新可以在 [此处](https://discord.com/channels/1146610656779440188/1194880263122075688/1266055462063964191) 查看。


  

---

### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1278438171285131306)** (2 messages): 

> - `Daily Bots`
> - `RTVI`
> - `Bland`
> - `Voice AI`
> - `Real-time AI` 


- **Daily Bots 发布，专注于实时 AI**：Daily Bots 是一个用于语音、视觉和视频 AI 的超低延迟云平台，其发布目标是将实时 AI 的最佳工具、开发者体验和基础设施整合到单个平台中。
   - Daily Bots 是开源的，并支持用于构建实时 AI 应用程序的 RTVI 标准，包括与 LLM 的语音对语音交互，延迟低至 500ms。
- **Bland 是一款可定制的 AI 电话呼叫 Agent**：Bland 是一款听起来像真人一样的可定制 AI 电话呼叫 Agent，已获得 2200 万美元的 A 轮融资，目前正脱离隐身状态。
   - Bland 可以使用任何语言或声音进行对话，可针对任何用例进行设计，并能 24/7 全天候同时处理数百万个电话，且不会产生幻觉。
- **合作伙伴与项目**：Daily Bots 在此次发布中与 Anthropic、Cartesia、Deepgram 和 Together Compute 建立了合作伙伴关系。
   - 两个增长最快的开源实时 AI 项目 Pipecat 和 RTVI 源于 Daily Bots 与客户及合作伙伴的合作，开创了生产环境中的实时和语音对语音 AI。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/usebland/status/1828882563588612233?s=46&t=G6jp7iOBtkVuyhaYmaDb0w">来自 Bland.ai (@usebland) 的推文</a>：今天对我们来说是一个重要的里程碑。我们完成了 2200 万美元的 A 轮融资。随着我们脱离隐身状态，我们想正式向您介绍 Bland，您最新的 AI 员工。Bland 是一个...</li><li><a href="https://x.com/i/status/1825946246886076785">来自 Daily (@trydaily) 的推文</a>：今天我们发布了 Daily Bots，这是一个用于语音、视觉和视频 AI 的超低延迟开源云。使用任何 LLM 构建语音对语音交互，对话延迟低至 500ms。通过 Daily ...
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1278418614961307700)** (3 messages): 

> - `Axolotl 对 Apple Silicon (M3) 的支持`
> - `在 Apple Silicon 上进行训练` 


- **Axolotl 在 Apple Silicon (M3) 上运行可行**：一位用户确认 Axolotl 可以在 Apple Silicon 上使用，特别是 M3 芯片。
   - 他们还提到在 128GB RAM 的 Macbook 上使用它，运行正常且没有任何错误，但未提供关于训练速度或是否需要任何自定义的细节。
- **训练速度和方法未指明**：该用户询问了在 M3 Macbook 上的训练速度。
   - 他们询问设置是否需要修改代码，以及用户在训练过程中是采用 Qlora、Lora 还是 full tuning 方法。


  

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1278219805698166836)** (4 messages): 

> - `Power Scheduler`
> - `Learning Rate`
> - `Batch Size`
> - `Training Tokens`
> - `QLora FSDP` 


- **IBM 推出 Power Scheduler：一种全新的 Learning Rate 调度方法**：IBM 引入了一种名为 Power Scheduler 的新型 Learning Rate 调度器，它与 Batch Size 和训练 Token 数量无关。
   - 该调度器是在对 Learning Rate、Batch Size 和训练 Token 之间的相关性进行广泛研究后开发的，揭示了幂律关系。该调度器在各种模型大小和架构中始终表现出令人印象深刻的性能，甚至超越了最先进的小型语言模型。
- **Power Scheduler：适用于所有配置的统一 Learning Rate**：这种创新的调度器允许预测任何给定 Token 数量、Batch Size 和模型大小的最佳 Learning Rate。
   - 通过使用方程：`lr = bsz * a * T ^ b!`，可以在不同配置中采用统一的 Learning Rate。
- **QLora FSDP 示例与 `fsdp_use_orig_params`**：关于在 QLora FSDP 示例中正确设置 `fsdp_use_orig_params` 的讨论。
   - 一些成员认为它应该始终设置为 `true`，而另一些成员则不确定，并建议这可能不是严格要求。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/_akhaliq/status/1828267147765702856?s=46">Tweet from AK (@_akhaliq)</a>: IBM presents Power Scheduler  A Batch Size and Token Number Agnostic Learning Rate Scheduler  discuss: https://huggingface.co/papers/2408.13359  Finding the optimal learning rate for language model pr...</li><li><a href="https://x.com/yikang_shen/status/1828503458749501714?s=46">Tweet from Yikang Shen (@Yikang_Shen)</a>: (3/5) So now we can accurately predict the optimal learning rate for any given number of tokens, batch size, and model size! But more interestingly, we can use one learning rate for all these differen...
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1278099442137829427)** (3 messages): 

> - `token analysis`
> - `model training` 


- **异常 Token 分析**：一位成员询问，如果其数据集中的 Token 与预训练数据集相比具有异常含义，模型是否应该识别这些 Token。
   - 该成员以 Token 'adam' 和 'Fortunately' 为例来演示这一概念。
- **Token 分布差异**：另一位成员建议，出现频率高于正态分布的 Token 可能是有效训练的指标。
   - 该成员的评论侧重于特定 Token 的频率，将其作为模型成功适应特定数据集的潜在迹象。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/)** (1 messages): 

akjindal53244: 目测看起来不错，但我们没有进行任何定量分析。

### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1278175766663790683)** (1 messages): 

> - `DSPy ImageNet moment`
> - `NeurIPS HackerCup 2024`
> - `DSPy for coding`
> - `Weights & Biases DSPy Talk` 


- **DSPy 的 "ImageNet 时刻"**: DSPy 的 "ImageNet" 时刻归功于 @BoWang87 实验室在 MEDIQA 挑战赛中的成功，其中基于 DSPy 的解决方案以 12.8% 和 19.6% 的显著优势赢得了两项临床 NLP 竞赛。
   - 这一成功导致 DSPy 的采用率大幅增加，类似于 CNNs 在 ImageNet 上表现出色后变得流行。
- **NeurIPS HackerCup 挑战赛：DSPy 的下一个 "ImageNet 时刻"？**: NeurIPS 2024 HackerCup 挑战赛被视为 DSPy 潜在的 "ImageNet 时刻"，类似于卷积神经网络在 ImageNet 上取得卓越成就后声名鹊起。
   - 该挑战赛为 DSPy 提供了一个展示其能力的平台，并可能获得更广泛的采用。
- **用于编程的 DSPy：入门指南**: 对于对 DSPy 感兴趣的人，@kristahopsalong 最近在 Weights & Biases 上发表了关于 DSPy 编程入门的演讲，涵盖了其 Optimizers，并使用 2023 HackerCup 数据集进行了动手演示。
   - 该演讲为任何有兴趣深入了解 DSPy 及其在编程中应用的人提供了一个很好的起点。
- **Weights & Biases 上的 DSPy 演讲**: @kristahopsalong 在 Weights & Biases 上的演讲涵盖了关于 DSPy Optimizers 的最新信息，并包含了使用 DSPy 进行代码生成的动手演示。
   - 演讲视频可在 YouTube 观看：[https://www.youtube.com/watch?v=yhYeDGxnuGY](https://www.youtube.com/watch?v=yhYeDGxnuGY)。
- **用于代码生成的 DSPy**: DSPy 正被用于代码生成，最近的一场演讲涵盖了它在 NeurIPS HackerCup 挑战赛中的应用。
   - 这表明 DSPy 不仅在 NLP 领域有效，在代码生成等其他领域也同样出色。



**提到的链接**: <a href="https://x.com/CShorten30/status/1828614227067650495">Connor Shorten (@CShorten30) 的推文</a>：卷积神经网络在极其流行的 ImageNet 数据集上超越了手工设计的图像特征，迎来了它们的 "ImageNet" 时刻。这随后引发了大规模的开发投入...

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1278097025812135936)** (7 messages): 

> - `OpenAI Base URL/Model Change`
> - `IPython Interpreter for DSPy`
> - `MIPRO Interview with Krista Opsahl-Ong` 


- **切换 Base URL 和模型**: 一位用户想要将 OpenAI 的 base URL 和模型更改为其他的 LLM（如 OpenRouter API），但没找到实现方法。
   - 他们提供了展示其尝试的代码片段，其中包括在 `dspy.OpenAI` 中设置 `api_base` 和 `model_type` 参数。
- **用于 DSPy 的 IPython 解释器**: 一位用户询问是否有任何正在开发中（WIP）的功能来实现 DSPy 的 IPython 解释器，并表达了对有状态执行（stateful execution）迭代特性的偏好。
   - 他们提到他们的团队正在构建将受益于此方法的系统，并表示如果设计和理论一致，愿意为 DSPy 贡献一个模块。
- **关于 MIPRO 和 DSPy 访谈 Krista Opsahl-Ong**: 一位用户兴奋地分享了一个播客访谈链接，受访者是 MIPRO（Multi-Prompt Instruction Proposal Optimizer）的主作者 Krista Opsahl-Ong，内容涉及该优化器和 DSPy。
   - 访谈涵盖了自动化 Prompt Engineering、多层语言程序、自我改进 AI 系统、LLMs 与工具使用以及用于代码生成的 DSPy 等主题。



**提到的链接**: <a href="https://x.com/CShorten30/status/1828794722908872902">Connor Shorten (@CShorten30) 的推文</a>：我非常激动地发布我们对来自 @StanfordAILab 的 Krista Opsahl-Ong (@kristahopsalong) 的采访！🔥 Krista 是 MIPRO 的主作者，MIPRO 是 Multi-prompt Instruction Proposal Optimizer 的缩写...

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1278124016191799366)** (1 messages): 

> - `Tinygrad Boxes Europe Shipping` 


- **Tinygrad 发货至欧洲！**: Tinygrad 现在提供向欧洲的发货服务！如需索取报价，请发送电子邮件至 [support@tinygrad.org](mailto:support@tinygrad.org)，并注明您的地址以及您想要的机箱型号。
- **Tinygrad 对发货的承诺**: Tinygrad 致力于让发货尽可能便捷，并将尽最大努力为您提供所需的机箱！

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1278362085515857983)** (5 messages): 

> - `Tinygrad CPU`
> - `Tinygrad Device Count` 


- **Tinygrad CPU 错误："No module named 'tinygrad.runtime.ops_cpu'"**：有用户报告在 CPU 上运行 Tinygrad 时遇到 "ModuleNotFoundError: No module named 'tinygrad.runtime.ops_cpu'" 错误。
   - 回复建议使用设备 "clang"、"llvm" 或 "python" 在 CPU 上运行，例如：`a = Tensor([1,2,3], device="clang")`。
- **在 Tinygrad 中获取 Device Count**：用户询问是否有比使用 `tensor.realize().lazydata.base.realized.allocator.device.count` 更简单的方法来获取 Tinygrad 中的设备数量。
   - 用户发现 `from tinygrad.device import Device` 和 `Device["device"].allocator.device.count` 提供了更直接的解决方案。


  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1278129945368858759)** (3 messages): 

> - `LAION-aesthetic dataset` 


- **LAION-aesthetic 链接失效**：一位成员询问 LAION-aesthetic 数据集的 Hugging Face 链接，因为 LAION 网站上的链接已失效。
- **可能的解决方案：CaptionEmporium**：另一位成员建议探索 Hugging Face 上的 CaptionEmporium 数据集，作为替代方案或相关数据源。


  

---



### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1278142565643714620)** (1 messages): 

> - `Llama 3.1 Benchmarking`
> - `Custom API for Llama 3.1`
> - `Inference Pipeline Benchmarking` 


- **Llama 3.1 基准测试**：一位成员请求关于使用自定义 API 对 **Llama 3.1** 进行基准测试的指导。
   - 他们希望对其公司私有托管的 **Llama 3.1 endpoint** 和推理流水线进行基准测试。
- **Llama 3.1 的自定义 API**：该成员拥有一个私有托管的 **Llama 3.1 endpoint**，并希望对推理流水线进行基准测试。
   - 他们正在寻求如何开始基准测试的指导。
- **推理流水线基准测试**：用户有兴趣为使用 **Llama 3.1** 的**自定义 API** 进行推理流水线基准测试。
   - 他们正在寻找有效评估推理流水线性能的建议和策略。


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1278111204119810091)** (1 messages): 

> - `BFCL Leaderboard`
> - `Model Handler Optimization`
> - `Function Calling Feature` 


- **BFCL 排行榜与不公平优化**：一位用户询问他们为 Function Calling 功能实施的某些优化是否会被认为对 BFCL 排行榜上的其他模型不公平。
   - 用户担心 BFCL 对可能无法推广到所有模型的 Model Handler 优化的立场，特别是关于他们使用的 System Prompts、Chat Templates、带有约束解码的 Beam Search 以及输出格式化。
- **Function Calling 功能的优化**：用户正致力于将这些步骤集成到其 Function Calling 功能的自定义 Model Handler 中。
   - 他们寻求更新 System Prompt、应用 Chat Template、使用带有约束解码的 Beam Search，并根据 Gorilla 指定的格式对模型输出进行格式化。


  

---



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
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}