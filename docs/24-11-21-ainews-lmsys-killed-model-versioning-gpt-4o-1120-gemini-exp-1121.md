---
companies:
- openai
- google-deepmind
- anthropic
- deepseek
- mistral-ai
date: '2024-11-22T00:56:03.268058Z'
description: '**2024年11月21日至11月22日的AI新闻摘要**重点报道了前沿实验室之间激烈的竞争。OpenAI的**gpt-4o-2024-11-20**与Google
  DeepMind的**gemini-exp-1121**在Lmsys排行榜上交替领先。包括**Anthropic**在内的各大领先实验室正呈现出一种新趋势：即使用基于日期的模型标识符，而非传统的版本号。


  **DeepSeek R1**作为一款强有力的开源替代方案备受瞩目，尤其是在中美AI竞争的背景下。**Gemini-Exp-1121**因其在视觉、编程和推理能力的提升而受到赞誉。此外，**MistralAI**在帕罗奥图（Palo
  Alto）开设了新办公室进行扩张，标志着公司的业务增长和人才招聘需求。'
id: 9853bcdc-3ed8-46e6-96b3-4e9c7e438e70
models:
- gpt-4o-2024-11-20
- gemini-exp-1121
- deepseek-r1
original_slug: ainews-lmsys-killed-model-versioning-gpt-4o-1120
people: []
title: LMSys 终结了模型版本化 (gpt 4o 1120, gemini exp 1121)
topics:
- model-release
- model-ranking
- open-source
- vision
- coding
- reasoning
- market-competition
---

<!-- buttondown-editor-mode: plaintext -->**日期就是你所需的一切 (Dates are all you need)。**

> 2024年11月21日至11月22日的 AI 新闻。我们为你检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 服务（**217** 个频道，**2501** 条消息）。为你节省了预计 **237 分钟** 的阅读时间（按每分钟 200 字计算）。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论了！

前沿实验室的竞争动态正变得有些荒谬。我们曾有一个规则，即新的 SOTA 模型总是占据榜首，[上周我们报道了 Gemini Exp 1114](https://buttondown.com/ainews/archive/ainews-gemini-experimental-1114-retakes-1-llm-9071/)，尽管除了其 LMSYS 排名外几乎没有其他有用的细节。但昨天 OpenAI 凭借 [gpt-4o-2024-11-20](https://x.com/lmarena_ai/status/1859307979184689269) 再次超越了它们，幸运的是我们没有报道这个（多亏了 DeepSeek R1），因为它现在[被怀疑是一个更差（但更快）的模型](https://x.com/ArtificialAnlys/status/1859614633654616310)（我们不知道这是否属实，但如果 OpenAI 实际上是将一个 "mini" 模型冠以主线模型的品牌并希望我们不会注意到，那将是一个非常严重的指控）。与此同时，今天 [Gemini Exp 1121](https://x.com/lmarena_ai/status/1859673146837827623) 发布了——再次从 OpenAI 手中夺回了 LMSYS 的榜首位置。

事情变得如此荒谬，以至于[这个调侃](https://x.com/adonis_singh/status/1859682100569571399) [OpenAI 与 Gemini 发布巧合](https://buttondown.com/ainews/archive/ainews-the-ai-search-wars-have-begun-searchgpt/)的笑话看起来都有几分可信：


![image.png](https://assets.buttondown.email/images/1f200ce6-01cb-4ebf-b385-14cb782c8c52.png?w=960&fit=max)


这种完全抛弃模型发布礼节的行为，总是可以用“我们只是想尽快把这些东西交到开发者手中”之类的善意借口来辩解。但我们现在面临的情况是，所有三家前沿实验室（提醒一下，Anthropic 尽管[表现得有些冷嘲热讽](https://x.com/alexalbert__/status/1859676984768688231?s=46)，但也一直在[玩这种“只有日期更新而没有版本号”的游戏](https://buttondown.com/ainews/archive/ainews-claude-35-sonnet-new-gets-computer-use/)）都拥有仅通过日期而非版本号来识别的 SOTA 模型变体，以维持在 LMSYS 上的地位。


![image.png](https://assets.buttondown.email/images/5faaf02f-bfad-471c-b6ae-8eb644203293.png?w=960&fit=max)


我们是不再进行版本命名了吗？希望不是，因为我们仍在讨论 o2、GPT-5、Claude 4 和 Gemini 2，但在 [100k clusters](https://x.com/ServeTheHome/status/1850917031421399543) 扩建过程中的这段过渡期停滞，是一个没人真正满意的局部最小值 (local minima)。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**主题 1. DeepSeek 与全球 AI 进展**

- **[DeepSeek R1 的表现](https://twitter.com/_philschmid/status/1859482879413158062)**：**DeepSeek R1** 与 **OpenAI o1-preview** 进行了对比，其“思考过程（thoughts）”直接流式输出，且在推理过程中未使用 MCTS。[@saranormous](https://twitter.com/saranormous/status/1859455354024927521) 强调了该模型的强大实力，暗示**芯片管制对于来自中国日益增长的竞争是无效的**，[@bindureddy](https://twitter.com/bindureddy/status/1859598807979393527) 也对此表示赞同，并称赞了 R1 的开源特性。
  - **市场影响与预测**：**Deepseek-r1** 作为 OpenAI 等现有领导者的竞争替代方案正受到关注，关于**中美 AI 竞赛**的讨论进一步强调了这一点。

**主题 2. 模型发布与技术发展**

- **[Google 的 Gemini-Exp-1121](https://twitter.com/lmarena_ai/status/1859673146837827623)**：该模型因其在**视觉、编程和创意写作方面的改进**而受到赞誉。[@Lmarena_ai](https://twitter.com/lmarena_ai/status/1859673146837827623) 讨论了它在 **Chatbot Arena 排行榜**上与 GPT-4o 并驾齐驱的上升势头，展示了性能的快速提升。
  - **增强功能**：据 [@_akhaliq](https://twitter.com/_akhaliq/status/1859713144710729853) 称，全新的**编程熟练度**、**更强的推理能力**以及**改进的视觉理解能力**使 Gemini-Exp-1121 成为一股强大的力量。

- **[Mistral 的扩张](https://twitter.com/dchaplot/status/1859398052500721943)**：**MistralAI** 宣布在帕洛阿尔托（Palo Alto）开设新办公室，预示着业务增长并在多个领域提供开放职位。正如 [@sophiamyang](https://twitter.com/sophiamyang/status/1859400690210103557) 所指出的，这一扩张反映了扩大运营规模和人才储备的战略推进。

- **[Claude Pro 与 Google Docs 集成](https://twitter.com/alexalbert__/status/1859664138072621228)**：**Anthropic** 为 Claude AI 增加了 **Google Docs 集成**功能，旨在简化组织层面的**文档管理**。

**主题 3. AI 框架与数据集发布**

- **[SmolTalk 数据集亮相](https://twitter.com/_philschmid/status/1859598525723488478)**：**SmolTalk** 是一个采用 Apache 2.0 协议的 100 万样本数据集，通过新的合成数据集提升了 **SmolLM v2 的性能**。该倡议有望增强各种模型的输出效果，如摘要提取和重写。
  - **数据集集成与性能**：该数据集与 **OpenHermes2.5** 等公共资源相结合，其表现优于在类似模型规模上训练的竞争对手，使其成为语言模型训练中的高影响力资源。

**主题 4. 创新 AI 应用与工具**

- **[LangGraph Agents 与 LangChain 的语音功能](https://twitter.com/LangChainAI/status/1859643185363902719)**：一段视频教程展示了如何使用 **OpenAI 的 Whisper** 进行输入并使用 **ElevenLabs** 进行语音输出，将 LangGraph Agent 转化为**语音启用助手**。
  - **OpenRecovery 对 LangGraph 的使用**：由 [LangChain](https://twitter.com/LangChainAI/status/1859613490081824845) 重点介绍，该应用在成瘾康复中的使用证明了其**实际的适应性和可扩展性**。

**主题 5. 基准测试与行业分析**

- **[AI 性能与行业洞察](https://twitter.com/maximelabonne/status/1859591100475888123)**：Menlo Ventures 发布了一份关于**生成式 AI 演进**的报告，强调了顶级用例和集成策略，并指出 Anthropic 在市场中的份额正在增长。
  - **模型微调与评估**：报告指出，行业正从**微调（fine-tuning）**转向更先进的 **RAG 和 Agentic AI 技术**，强调了 LLM 工程师在优化 AI 应用中的价值。

**主题 6. 迷因/幽默**

- **[与 AI 及 OpenAI 的奇遇](https://twitter.com/aidan_mclau/status/1859445818031210880)**：[@aidan_mclau](https://twitter.com/aidan_mclau/status/1859445818031210880) 幽默地思考了将新语言模型行为归入明确定义的类别的挑战，反映了当前 AI 发展中经常不可预测的本质。
  - **[金融幽默与预测](https://twitter.com/nearcyan/status/1859426783663448349)**：[@nearcyan](https://twitter.com/nearcyan/status/1859426783663448349) 在金融讨论中加入幽默，将 FAANG 公司的“放鸽子（ghosting）”经历与工程领域不断变化的职业格局进行了类比。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. M4 Max 128GB：通过 MLX 以 11 t/s 的速度运行 72B 模型**

- **[M4 Max 128GB 运行 Qwen 72B Q4 MLX，速度达 11 tokens/second。](https://i.redd.it/wbdf0b5e772e1.jpeg)** ([Score: 476, Comments: 181](https://reddit.com/r/LocalLLaMA/comments/1gw9ufb/m4_max_128gb_running_qwen_72b_q4_mlx_at/)): 配备 **128GB** 内存的 **Apple M4 Max** 成功以每秒 **11 tokens** 的速度运行 **Qwen 72B Q4 MLX** 模型。这一性能指标展示了 **Apple silicon** 高效处理大型语言模型（LLM）的能力。
  - 用户讨论了**功耗**和**散热表现**，指出系统在推理过程中功耗高达 **190W**，且运行温度较高。**M4 Max** 在实现这一性能的同时，功耗显著低于配备多块 **NVIDIA 3090** 或 **A6000** 的同类配置。
  - **M4 Max** 的**内存带宽**为 **546 GB/s**，性能为 **11 tokens/second**，相比 **M1 Max**（**409.6 GB/s** 和 **6.17 tokens/second**）有显著提升。用户成功测试了包括 **Qwen 72B**、**Mistral 128B** 以及具有 **32k context** 窗口的小型编程模型在内的多种模型。
  - 讨论对比了组装台式机（仅 GPU 就需约 **4000 美元**）与 **4700 美元** 的 M4 Max 笔记本电脑的成本，许多人强调了 Apple 系统在运行本地 LLM 时的便携性优势和完整解决方案特性，特别是在旅行或电力受限的场所。


- **Mac 用户：适用于 Apple Silicon (MLX) 的新 Mistral Large MLX 量化版本** ([Score: 91, Comments: 23](https://reddit.com/r/LocalLLaMA/comments/1gw6yrg/mac_users_new_mistral_large_mlx_quants_for_apple/)): 一位开发者使用 **MLX-LM** 创建了针对 **Apple Silicon** 优化的 **Mistral Large** **2-bit** 和 **4-bit** 量化版本，其中 **q2** 版本在 **M4 Max** 上运行速度达到 **7.4 tokens/second**，同时占用 **42.3GB RAM**。这些模型可在 [HuggingFace](https://huggingface.co/zachlandes/Mistral-Large-Instruct-2411-Q2-MLX) 上获取，并能在 **LMStudio** 或其他 **MLX** 兼容系统中运行，其在 **M-series** 芯片上的性能有望优于 **GGUF** 模型。
  - 用户询问了性能对比，测试显示 **MLX** 模型在 **Apple Silicon** 上的运行速度比 **GGUF** 版本快约 **20%**，多位用户独立确认了这一点。
  - 问题集中在实际使用上，包括如何通过 **LMStudio** 运行模型，用户可以从 **HuggingFace** 手动下载并放入 **LMStudio cache folder** 以供识别。
  - 用户讨论了硬件兼容性，特别是关于 **M4 Pro 64GB** 运行 **Mistral Large** 变体的能力，并有兴趣将其性能与 **Llama 3.1 70B Q4** 进行对比。


**主题 2. DeepSeek R1-Lite Preview 展示了强大的推理能力**

- **[DeepSeek AI 的 R1-Lite-Preview 展示了它的实力……天哪！！太惊人了！！](https://www.reddit.com/gallery/1gw61g5)** ([Score: 146, Comments: 19](https://reddit.com/r/LocalLLaMA/comments/1gw61g5/here_the_r1litepreview_from_deepseek_ai_showed/)): **DeepSeek 的 R1-Lite-Preview 模型**展示了先进的能力，尽管帖子正文中未提供具体示例或细节。帖子标题表达了对模型性能的热情，但缺乏关于其实际能力或 **benchmarks** 的实质性信息。
  - **Base32** 解码能力在不同模型之间差异显著，**GPT-4** 表现成功，而其他模型则表现挣扎。讨论强调大多数开源模型在处理密码（ciphers）方面表现不佳，尽管由于 **base64** 在训练数据中非常普遍，它们处理 **base64** 的效果很好。
  - 在 **DeepSeek** 的 **R1-Lite-Preview** 中注意到了 **MLX** 知识空白，这表明其参数容量有限，无法涵盖全面的领域知识。这一限制反映出该模型可能比其他当代模型规模更小。
  - 对 **tokenization** 限制的讨论解释了模型在编码/解码任务中的表现，目前的模型使用基于 **token** 而非基于字符的处理方式。用户将这种限制比作人类试图计算看不见的原子——这是一种系统限制，而非智力衡量标准。

- **[DeepSeek R1-Lite 令人印象深刻，甚至让 Qwen 2.5 coder 显得逊色，这就是我这么说的原因，我在最近的 Codeforces 竞赛题目中测试了 R1-Lite（虚拟参赛），它的表现非常……非常出色](https://i.redd.it/8tgij0jc882e1.png)** ([Score: 135, Comments: 44](https://reddit.com/r/LocalLLaMA/comments/1gwcsys/deepseek_r1_lite_is_impressive_so_impressive_it/)): **DeepSeek R1-Lite** 在通过虚拟参赛测试的 **Codeforces** 竞赛编程题目中，展现出优于 **Qwen 2.5** 的性能。发帖者强调了 R1-Lite 的卓越表现，但未提供具体的指标或详细对比。
  - **DeepSeek R1-Lite** 在不同任务中表现参差不齐——在处理**乱序字母 (scrambled letters)**和**数字加密 (number encryption)**方面很成功，但在 **Playfair Cipher** 上始终失败。用户指出它擅长处理如**算法竞赛 (competitive programming)**任务等小范围问题，但在现实世界的编程场景中可能会遇到困难。
  - **R1-Lite** 与 **Qwen 2.5** 的对比测试显示，Qwen 在实际任务中表现更好，有用户报告在 **Unity C# 脚本编写**和实现**射线检测悬挂系统 (raycast suspension system)**方面取得了成功。两个模型都能编写**俄罗斯方块 (Tetris)**，Qwen 仅用一次尝试就完成了，而 R1 用了两次。
  - 用户强调，在**算法竞赛**中的成功并不一定能转化为现实世界的编程能力。建议在 [atcoder.jp](atcoder.jp) 和 **Codeforces** 等平台上使用独特的、最近的题目进行测试，以便更好地评估模型。


**主题 3. Gemini-exp-1121 凭借增强的 Coding 和 Vision 能力登顶 LMSYS**

- **[Google 发布新模型，登顶 LMSYS 排行榜](https://i.redd.it/zzdnaa997b2e1.jpeg)** ([Score: 139, Comments: 53](https://reddit.com/r/LocalLLaMA/comments/1gwoikh/google_releases_new_model_that_tops_lmsys/)): **Google** 发布了 **Gemini-exp-1121**，该模型在 **LMSYS** 排行榜的编程和视觉任务中获得了最高分。该模型代表了对之前 **Gemini** 版本的改进，尽管公告中未提供具体的性能指标。
  - **LMSYS** 排行榜的排名备受争议，用户认为 **Claude** 排名第 7 说明了基准测试的局限性。多位用户报告 **Claude** 在现实应用中优于竞争对手，特别是在编程和技术任务方面。
  - **Gemini** 的新视觉能力可以通过处理完整的图像上下文来实现直接的**漫画翻译 (manga translation)**，相比传统的 **OCR + 翻译** 流程具有优势。这种方法能更好地处理依赖上下文的元素，如角色性别和专业术语。
  - **Google** 和 **OpenAI** 之间出现了一种竞争模式，两家公司不断发布模型以争夺排行榜首位。**Gemini-exp-1121** 的发布似乎是紧随 **OpenAI** 最近发布模型后的战略举措。


**主题 4. Allen AI 的 Tulu 3：基于 Llama 3.1 的开源指令模型**

- **[Tülu 3 —— 一系列具有完全开放数据、评估代码和训练算法的 SOTA 指令模型](https://x.com/allen_ai/status/1859643404847808935)** ([Score: 117, Comments: 23](https://reddit.com/r/LocalLLaMA/comments/1gwl339/tülu_3_a_set_of_stateoftheart_instruct_models/)): **Allen AI** 发布了 **Tülu 3**，这是一个开源指令遵循模型 (instruction-following models) 的集合，提供了对训练数据、评估代码和训练算法的完整访问权限。这些模型旨在提升 **state-of-the-art** 性能，同时在开发过程中保持完全透明。
  - **Tülu 3** 是 **Llama 3.1 微调 (fine-tunes)** 的集合，而非从零开始构建的模型，提供 [8B](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B) 和 [70B](https://huggingface.co/allenai/Llama-3.1-Tulu-3-70B) 版本。社区成员已经创建了 **GGUF 量化版本** 和 **4-bit 变体** 以提高易用性。
  - 性能基准测试显示，**8B 模型** 超过了 **Qwen 2.5 7B Instruct**，而 **70B 模型** 优于 **Qwen 2.5 72B Instruct**、**GPT-4o Mini** 和 **Claude 3.5 Haiku**。发布内容包括全面的训练数据、奖励模型和超参数。
  - **Allen AI** 宣布其完全开源的 **OLMo** 模型系列将在本月获得更新。关于 **Tülu 3** 训练过程的详细讨论可以在[新发布的播客](https://youtu.be/LVXtFnEbNU0)中找到。


**主题 5. NVIDIA KVPress：开源 KV Cache 压缩研究**

- **NVIDIA 发布用于 KV 压缩研究的新仓库** ([Score: 48, Comments: 7](https://reddit.com/r/LocalLLaMA/comments/1gwgc5q/new_nvidia_repo_for_kv_compression_research/)): **NVIDIA** 发布了一个名为 **kvpress** 的开源库，旨在解决大型语言模型中的 **KV cache 压缩** 挑战。例如，**llama 3.1-70B** 模型在 **float16** 精度下处理 **1M tokens** 需要 **330GB** 的内存。该库基于 **🤗 Transformers** 构建，引入了一种新的 **"expected attention"** 方法，并为研究人员开发和基准测试压缩技术提供了工具，代码托管在 [kvpress](https://github.com/NVIDIA/kvpress)。
  - **kvpress** 目前不支持 **KV cache 量化**，但根据 FAQ，它可以与剪枝策略结合使用，在从 **float16** 转换为 **int4** 时，有望实现高达 **4 倍的压缩**。
  - 这是提供的评论中唯一有意义的讨论点——其他评论和回复没有增加实质性的总结信息。


## 其他 AI Subreddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**主题 1. Flux.1 工具套件扩展了 SD 的功能**

- **[FLUX 重大新闻发布。这非常重要。使用 FLUX DEV 的 Inpainting 和 Outpainting 效果优于付费版 Adobe Photoshop。FLUX 团队发布了类似 Canny 和 Depth 的 ControlNet，以及图像变体和概念迁移（如风格迁移或 0-shot 面部迁移）。](https://www.reddit.com/gallery/1gwilop)** ([Score: 739, Comments: 194](https://reddit.com/r/StableDiffusion/comments/1gwilop/huge_flux_news_just_dropped_this_is_just_big/)): **Black Forest Labs** 发布了其 **Flux.1 Tools** 控制套件，其具备的 **inpainting** 和 **outpainting** 能力可与 **Adobe Photoshop** 竞争，同时还包含了针对 **Canny** 和 **Depth** 控制的 **ControlNet** 风格功能。此次发布还包括 **image variation**（图像变体）和 **concept transfer**（概念迁移）工具，支持 **style transfer**（风格迁移）和 **zero-shot face transfer**（zero-shot 面部迁移）功能。
  - **ComfyUI** 在发布首日即提供了对 **Flux Tools** 的支持，并附带详细的实现示例。完整模型需要 **27GB VRAM**，不过 **LoRA 版本** 已在 [Huggingface](https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev-lora/tree/main) 上提供。
  - 社区反馈表明其 **outpainting** 能力非常强大，可与 **Midjourney** 媲美，用户尤其称赞 **Redux IP adapter** 的性能和强度。这些工具已公开用于 **FLUX DEV 模型**，实现细节见 [Black Forest Labs](https://blackforestlabs.ai/flux-1-tools/)。
  - 用户批评了标题党的发布方式，并要求更直接的技术沟通，同时也注意到 [Civitai](https://civitai.com/models/969431/flux-fill-fp8) 上已提供 **FP8 版本**，以满足较低 **VRAM** 的需求。


- **[ComfyUI 首日支持 FLUX 工具](https://blog.comfy.org/day-1-support-for-flux-tools-in-comfyui/)** ([Score: 149, Comments: 26](https://reddit.com/r/StableDiffusion/comments/1gwibxr/day_1_comfyui_support_for_flux_tools/)): **ComfyUI** 在发布当天立即增加了对 **FLUX Tools** 的支持，尽管帖子中未提供集成的具体细节。
  - **ComfyUI** 用户报告已成功集成 **Flux Tools**，**SwarmUI** 也按照 [GitHub](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Model%20Support.md#flux1-tools) 文档为所有模型变体提供原生支持。
  - 用户发现 **Redux** 效果过强且无法与 **FP8_scaled** 模型配合使用的问题，但通过调整 **ConditioningSetTimestepRange** 和 **ConditioningSetAreaStrength** 参数有所改善。建议使用 **ImageCompositeMasked** 或 **Inpaint Crop/Stitch** 节点的正确合成工作流，以防止 **VAE** 降级。
  - 该实现支持 **Redux Adapter**、**Fill Model** 以及 **ControlNet Models & LoRAs**（特别是 **Depth** 和 **Canny**），演示工作流可在 [CivitAI](https://civitai.com/models/862215/proper-flux-control-net-inpainting-with-batch-size-comfyui-alimama) 获取。

- **[Flux Redux 无文本提示词](https://i.redd.it/6hpnyn5fwa2e1.png)** ([评分: 43, 评论: 30](https://reddit.com/r/StableDiffusion/comments/1gwmzc5/flux_redux_with_no_text_prompt/)): **Redux adapter** 测试侧重于无文本提示词的**图像变体 (image variation)** 能力，尽管帖子正文未提供具体细节或结果。
  - **FLUX.1 Redux** 适配器专注于在保持风格和场景的同时生成带有变体的图像，且不会重新生成面部。用户反馈其结果更快、更精确，特别是在更换衣服和背景的 **inpainting**（局部重绘）功能方面。
  - **ComfyUI** 的实现需要将 [sigclip vision model](https://huggingface.co/Comfy-Org/sigclip_vision_384/) 放置在 models/clip_vision 文件夹中。更新和工作流可以在 [ComfyUI 示例页面](https://comfyanonymous.github.io/ComfyUI_examples/flux/)找到。
  - **Flux Tools** 与 ComfyUI 的集成提供了 **ControlNet**、变体以及 **in/outpainting**（局部重绘/扩图）等功能，详见 [Black Forest Labs 文档](https://blackforestlabs.ai/flux-1-tools/)。实现指南可在 [ComfyUI 博客](https://blog.comfy.org/day-1-support-for-flux-tools-in-comfyui/)查阅。


**主题 2. NVIDIA/MIT 发布 SANA：高效的 Sub-1B 参数扩散模型**

- **SANA 的扩散代码刚刚发布** ([评分: 103, 评论: 52](https://reddit.com/r/StableDiffusion/comments/1gwav1d/diffusion_code_for_sana_has_just_released/)): **SANA** 扩散模型的训练和推理代码已由 **NVlabs** 在 **GitHub** 上发布。模型权重预计将在 **HuggingFace** 的 *"Efficient-Large-Model/Sana_1600M_1024px"* 路径下提供，但目前尚无法访问。
  - **SANA** 模型的核心特性是能够直接输出 **4096x4096** 图像，尽管一些用户指出 **UltraPixel** 和 **Cascade** 等模型也能实现这一点。该模型有 **0.6B** 和 **1.6B** 参数两种尺寸，明显小于 **SDXL (2B)** 和 **Flux Dev (12B)**。
  - 该模型由 **NVIDIA**、**MIT** 和**清华大学**的研究人员发布，采用 **CC BY-NC-SA 4.0 许可证**。用户注意到这比 **PixArt-Sigma 的 OpenRail++ 许可证**更具限制性，但对大公司发布模型权重这一罕见举动表示赞赏。
  - 技术讨论集中在该模型的速度优势和**微调 (fine-tuning)** 潜力上，其中 **0.6B 版本**被考虑用于特定专业场景。该模型已在 [HuggingFace](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px/tree/main/checkpoints) 上线，文件大小为 **6.4GB**。


- **[测试 CogVideoX1.5-5B i2v 模型](https://www.reddit.com/gallery/1gwdn8o)** ([评分: 177, 评论: 51](https://reddit.com/r/StableDiffusion/comments/1gwdn8o/testing_the_cogvideox155b_i2v_model/)): 社区对 **CogVideoX1.5-5B** 这一**图生视频 (image-to-video)** 模型进行了测试和评估讨论。帖子正文未提供关于测试程序或结果的足够上下文细节。
  - 该模型的**工作流 (workflow)** 可在 [Civitai](https://civitai.com/models/968568) 获取，建议**分辨率**在 **720p** 以上。根据 **Kijai 文档**，**v1.5** 版本仍处于测试阶段，目前仅支持 **1344x768**。
  - 使用 **4090 GPU**，生成 **1024x640** 分辨率视频约需 **3 分钟**，**1216x832** 约需 **5 分钟**。配备 **24GB VRAM** 的 **3090** 可以在不开启 'enable_sequential_cpu_offload' 功能的情况下运行。
  - 技术限制包括 **GGUF 版本**性能较差且偶尔崩溃、与**动漫风格图像**不兼容，以及在 **Windows** 上尝试以 **1024x640** 分辨率生成 **81 帧**时可能出现 **显存溢出 (OOM)** 问题。


**主题 3. ChatGPT 4o 11月更新：写作能力提升，测试分数下降**

- **[gpt-4o-2024-11-20 在 MMLU、GPQA、MATH 和 SimpleQA 上的得分低于 gpt-4o-2024-08-06](https://i.redd.it/ocb1qkhgk92e1.png)** ([得分: 77, 评论: 17](https://reddit.com/r/OpenAI/comments/1gwhdn4/gpt4o20241120_scores_lower_on_mmlu_gpqa_math_and/)): **GPT-4o** 的 2024 年 11 月更新显示，与 8 月版本相比，在包括 **MMLU**、**GPQA**、**MATH** 和 **SimpleQA** 在内的多个基准测试中性能有所下降。由于缺乏额外的上下文，无法分析具体的得分差异或下降的潜在原因。
  - 根据 [lifearchitect.ai](https://lifearchitect.ai/models-table/) 的数据，最新版 **GPT-4o** 的**性能下降**显著，**GPQA** 下降了 **13.37%**，**MMLU** 下降了 **3.38%**。该模型在某些基准测试中的得分现在低于 **Claude 3.5 Sonnet**、**Llama 3.1 405B**、**Grok 2**，甚至低于 **Grok 2 mini**。
  - 多位用户认为 **OpenAI** 正在针对**创意写作**和用户吸引力进行优化，而非事实准确性，这可能解释了基准测试性能下降的原因。这种权衡导致创意任务有了“令人惊叹”的改进，但牺牲了客观正确性。
  - 用户表达了对**专门模型命名**（如 "*gpt-4o-creative-writing*" 或 "*gpt-4o-coding*"）的渴望，并认为这些变化是由成本优化驱动的。**Anthropic** 的 **Sonnet** 模型也出现了类似的专业化趋势，显示出特定任务的改进和退步。


- **[OpenAI 的新更新使其成为史上最伟大的作词人 (ChatGPT + Suno)](https://v.redd.it/qx79ooj6o72e1)** ([得分: 57, 评论: 27](https://reddit.com/r/OpenAI/comments/1gwb7cy/openais_new_update_turns_it_into_the_greatest/)): **OpenAI** 发布了一项更新，当与 **Suno** 结合使用时，增强了其在歌词方面的创意写作能力。帖子正文中未提供额外的上下文或具体的改进细节。
  - 用户将该 AI 的说唱风格与包括 **Eminem**、**Notorious B.I.G.**、**Talib Kweli** 和 **Blackalicious** 在内的多位艺术家进行比较，一些人认为它超越了 **98%** 的人类说唱。原始来源通过 [Twitter/X](https://x.com/kyleshannon/status/1859355131738734824) 分享。
  - 技术改进似乎集中在 **LLM 的押韵能力**上，同时保持连贯的叙事结构。多位用户注意到 AI 在提供有意义内容的同时保持一致模式的能力。
  - 多条评论对 AI 的快速进步表示担忧，一位用户指出，人类不仅在**数学**和**国际象棋**方面被超越，现在在**说唱**等创意追求方面也被超越。这种情绪暗示了对 AI 能力的重大忧虑。


**主题 4. 需求压力迫使 Claude 免费用户受限于 Haiku**

- **免费账户现在（永久性地？）被路由至 3.5 Haiku** ([得分: 52, 评论: 40](https://reddit.com/r/ClaudeAI/comments/1gwe8fx/free_accounts_are_now_permanently_routed_to_35/)): **Claude** 免费账户现在默认使用 **Haiku 模型**，这是由用户 **u/Xxyz260** 通过对 **2023 年 10 月 7 日**袭击事件的特定提示词测试发现的。这一变化似乎并未公布，用户报告在 **18 小时**内间歇性地访问过 **Sonnet 3.5**，随后又回退到 **Haiku**，这表明 **Anthropic** 可能正在进行负载均衡测试。
  - 用户通过测试确认免费账户接收的是 **Haiku 3.5** 而非 Haiku 3，[测试结果](https://imgur.com/a/SCpsPqp)中的证据表明，模型知识来源于系统提示词而非模型本身。
  - 一个核心担忧是，**Pro 用户**在耗尽 **Sonnet** 额度后无法访问最新的 **Haiku 3.5** 模型，而免费用户却默认获得了更新版本。
  - 关于 **ChatGPT** 相比 **Claude** 变得更具吸引力的讨论（特别是在编程任务方面），用户对 **Anthropic** 处理服务变更的方式以及缺乏透明度表示沮丧。

- **他们打算解决 Claude 服务器过载的问题吗？** ([Score: 27, Comments: 46](https://reddit.com/r/ClaudeAI/comments/1gw7t7c/are_they_gonna_do_something_about_claudes/)): 一位用户报告称，由于**免费账户**的服务器可用性问题，无法访问 **Claude Sonnet 3.5**，同时发现 **Claude Haiku** 在遵循指令方面不可靠。该用户表示，尽管他们将 Claude 广泛用于研究、创意写作和内容创作，但作为一名大学生，每月 **$20** 的 **Claude Pro** 订阅费用过高。
  - **Claude Sonnet** 的**免费使用**受服务器负载影响严重，尤其是在**美国办公时间**，有用户报告长达 **14 小时**无法使用。欧洲用户指出在他们的白天时段访问情况较好。
  - 甚至**付费用户**也遇到了过载问题，这表明 **Anthropic** 的服务器容量限制非常显著。除非公司扩大容量或**训练成本**下降，否则免费用户的情况不太可能改善。
  - 关于 **Haiku 模型版本**（3.0 vs 3.5）存在困惑，用户分享了[对比截图](https://i.postimg.cc/jR8LNVVM/Screenshot-20241121-120819.jpg)并注意到移动应用和网页 UI 显示不一致，这暗示可能存在 **A/B 测试**或 UI Bug。


---

# AI Discord 摘要

> 由 O1-mini 生成的摘要之摘要的总结

**主题 1. 新 AI 模型凭借增强功能实现跨越式发展**

- [**Tülu 3 发布，性能超越 Llama 3.1**](https://x.com/natolambert/status/1859643351441535345): Nathan Lambert 宣布发布 **Tülu 3**，这是一个开放的前沿模型，通过引入一种新型的**带有可验证奖励的强化学习 (Reinforcement Learning with Verifiable Rewards)** 方法，在多项任务中表现优于 **Llama 3.1**。这一进步确保了在实际应用中更高的准确性和可靠性。
- [**Gemini Experimental 1121 登顶 Chatbot Arena 基准测试**](https://x.com/lmarena_ai/status/1859673146837827623): Google DeepMind 的 **Gemini-Exp-1121** 在 Chatbot Arena 中并列第一，超越了 **GPT-4o-1120**。其在**代码编写**和**推理能力**方面的显著提升突显了 AI 模型性能的快速进步。
- [**Qwen 2.5 在代码编辑方面达到 GPT-4o 级别性能**](https://aider.chat/2024/11/21/quantization.html): 开源模型如 **Qwen 2.5 32B** 在 Aider 的代码编辑基准测试中表现出极具竞争力的性能，与 **GPT-4o** 持平。用户强调了模型**量化 (Quantization)** 的关键作用，并指出不同量化级别会导致显著的性能差异。

**主题 2. 先进的微调技术提升模型效率**

- [**Unsloth AI 引入视觉支持，微调速度翻倍**](https://huggingface.co/unsloth/): **Unsloth** 为 **LLaMA**、**Pixtral** 和 **Qwen** 等模型推出了**视觉支持**，通过将微调速度提高 **2 倍**并将显存占用减少 **70%**，增强了开发者的能力。这使得 Unsloth 在基准测试中领先于 **Flash Attention 2 (FA2)** 和 **Hugging Face (HF)**。
- [**上下文位置编码 (CoPE) 增强模型表达能力**](https://arxiv.org/abs/2405.18719): **上下文位置编码 (Contextual Position Encoding, CoPE)** 根据 Token 上下文而非固定计数来调整位置编码，从而产生更具**表达能力的模型**。该方法改进了对 **Flip-Flop** 等选择性任务的处理，而传统的位置编码在这些任务中表现不佳。
- [**AnchorAttention 为长上下文模型减少超过 50% 的训练时间**](https://github.com/haonan3/AnchorContext): 一篇新论文介绍了 **AnchorAttention**，这是一种即插即用的解决方案，在增强长上下文性能的同时，将训练时间缩短了 **50%** 以上。它兼容 [FlashAttention](https://github.com/haonan3/AnchorContext) 和 **FlexAttention**，适用于**视频理解**等应用场景。

**主题 3. 硬件解决方案和性能优化驱动 AI 效率**

- [**基于云端的 GPU 租赁以每月 25-50 美元的价格提升模型速度**](https://github.com/NousResearch/Hermes-3-Llama-3.1-70B)：转向云服务器托管模型每月成本为 **$25-50**，且与本地硬件相比显著提升了模型速度。用户发现云端托管的 GPU 更具性价比且性能更强，避免了本地部署（on-premises）的局限性。
- [**YOLO 在实时视频目标检测中表现卓越**](https://huggingface.co/spaces/prithivMLmods/YOLO-VIDEO)：**YOLO** 仍然是**视频目标检测**的首选，并由 [YOLO-VIDEO](https://huggingface.co/spaces/prithivMLmods/YOLO-VIDEO) 资源提供支持。持续的策略旨在优化 YOLO 在实时处理场景中的性能。
- [**MI300X GPU 在长时间运行时遇到严重的挂起问题**](https://github.com/ROCm/ROCm/issues/4021)：成员报告称，在配合 **axolotl** 进行长达 **12-19 小时** 的长时间运行时，**MI300X** GPU 会出现**间歇性 GPU 挂起**，主要发生在 **6 小时** 之后。这些稳定性问题正在 [GitHub Issue #4021](https://github.com/ROCm/ROCm/issues/4021) 中进行跟踪，包括 **loss** 和 **learning rate** 等详细指标。

**主题 4. API 和集成助力自定义部署与增强**

- [**Hugging Face Endpoints 支持自定义 Handler 文件**](https://huggingface.co/philschmid/distilbert-onnx-banking77/blob/main/handler.py)：**Hugging Face Endpoints** 现在允许使用 [handler.py](https://huggingface.co/philschmid/distilbert-onnx-banking77/blob/main/handler.py) 文件部署自定义 AI 模型，从而实现定制化的预处理和后处理。实现 [EndpointHandler](https://huggingface.co/philschmid/distilbert-onnx-banking77/blob/main/handler.py) 类可确保根据特定需求进行灵活且高效的模型部署。
- [**Model Context Protocol (MCP) 增强本地交互**](https://x.com/btibor91/status/1859385266328531198)：**Anthropic** 的 **Claude Desktop** 现在支持 [Model Context Protocol (MCP)](https://x.com/btibor91/status/1859385266328531198)，能够通过 **Python** 和 **TypeScript SDK** 增强与模型的本地交互。虽然远程连接功能尚待开发，但初始支持已包含多种 SDK，引发了对扩展功能的关注。
- [**OpenRouter API 文档已澄清以实现无缝集成**](https://openrouter.ai/docs/provider-routing#quantization-levels)：用户对 **OpenRouter API** 文档中关于 **context window** 的功能表示困惑。建议进行改进以提高清晰度，协助与 **LangChain** 等工具的无缝集成，并优化高上下文提示词的 **provider selection**。

**主题 5. 全面的模型评估和基准测试对比揭示 AI 进展**

- [**Perplexity Pro 在特定任务的准确性上优于 ChatGPT**](https://aider.chat/2024/11/21/quantization.html)：用户对比了 **Perplexity** 与 **ChatGPT**，指出 **Perplexity** 被认为更准确，并在特定功能上具有优势。一位参与者强调，Perplexity 的某些功能在流行度激增之前就已经开发完成，凸显了其强大的能力。
- [**SageAttention 通过 8-Bit 量化提升注意力机制效率**](https://arxiv.org/abs/2410.02367)：[SageAttention](https://arxiv.org/abs/2410.02367) 方法为 **attention mechanisms** 引入了一种高效的 **8-bit quantization** 方案，在保持 **accuracy** 的同时提升了每秒操作数。这一改进解决了传统上与长序列相关的高计算复杂度问题。
- [**DeepSeek-R1-Lite-Preview 在编程基准测试中展示了卓越的推理能力**](https://api-docs.deepseek.com/news/news1120)：**DeepSeek** 推出了 **DeepSeek-R1-Lite-Preview**，在编程基准测试中展示了令人印象深刻的 **reasoning capabilities**。像 [Zhihong Shao](https://x.com/zhs05232838/status/1859201857593524352) 这样的用户赞扬了它在编程和数学挑战中的表现，突出了其应用价值。

---

# 第 1 部分：Discord 高层级摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 推出视觉支持**：Unsloth 正式推出了 **vision support**，支持对 **LLaMA**、**Pixtral** 和 **Qwen** 等模型进行微调，显著增强了开发者的能力。
   - 该功能将微调速度提升了 **2x**，并减少了 **70%** 的内存占用，使 Unsloth 在基准测试中领先于 **Flash Attention 2 (FA2)** 和 **Hugging Face (HF)**。
- **Qwen 和 LLaMA 微调增强**：用户正在探索微调 **Qwen** 和 **LLaMA** 的 Base 及 Instruct 模型的可行性，讨论集中在创建和合并 **LoRAs**。
   - Unsloth 的视觉支持通过将 **4-bit LoRAs** 转换回 **16-bit** 来简化合并过程，从而优化了微调流程。
- **Llama 3.2 Vision 亮相**：Unsloth 现已支持 **Llama 3.2 Vision** 模型，实现了 **2x 更快** 的训练速度和 **70% 的内存节省**，同时支持 **4-8x 更长的 context lengths**。
   - 此次发布包括用于 **Radiography** 和 **Maths OCR to LaTeX** 等任务的 **Google Colab notebooks**，可通过 [Colab 链接](https://colab.research.google.com/drive/1j0N4XTY1zXXy7mPAhOC1_gMYZ2F2EBlk?usp=sharing) 访问。
- **AnchorAttention 提升长上下文训练**：一篇新论文介绍了 **AnchorAttention**，这是一种增强长上下文性能并将训练时间缩短 **50%** 以上的方法。
   - 该解决方案兼容 [FlashAttention](https://github.com/haonan3/AnchorContext) 和 **FlexAttention**，适用于视频理解等应用。
- **训练 Checkpoint 选择策略**：关于选择合适训练 Checkpoint 的讨论揭示了多种方法，一些成员选择广泛使用 Checkpoint，而另一些成员则主张根据特定指标进行战略性选择。
   - 参与者强调了性能基准测试的重要性，并分享了优化训练工作流中 Checkpoint 选择的经验。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Tülu 3 发布创新**：Nathan Lambert 宣布发布 [Tülu 3](https://x.com/natolambert/status/1859643351441535345)，这是一个在多项任务上超越 **Llama 3.1** 的开源前沿模型，并结合了新型的 **Reinforcement Learning with Verifiable Rewards** 方法。
   - 新模型仅针对准确的生成结果奖励算法，增强了其在实际应用中的性能和可靠性。
- **Nvidia 的 AI Wall 担忧**：来自 [The Economist](https://www.economist.com/business/2024/11/21/nvidias-boss-dismisses-fears-that-ai-has-hit-a-wall) 的一篇文章报道称，尽管社区普遍存在怀疑态度，但 Nvidia 的 CEO 淡化了对 AI 已“撞墙”的担忧。
   - 这一立场加剧了关于 **AI 进步当前轨迹** 以及对持续创新迫切需求的讨论。
- **Gemini 在 Chatbot Arena 的表现**：Google DeepMind 的 [Gemini-Exp-1121](https://x.com/lmarena_ai/status/1859673146837827623) 在 Chatbot Arena 中并列第一，在最近的基准测试中超越了 **GPT-4o-1120**。
   - Gemini-Exp-1121 在 **coding** 和 **reasoning** 能力方面表现出显著提升，突显了 AI 模型性能的快速进步。
- **Reinforcement Learning with Verifiable Rewards**：Tülu 3 采用了一种名为 **Reinforcement Learning with Verifiable Rewards** 的新技术，该技术在受限的数学问题上训练模型并奖励正确输出，详见 Nathan Lambert 的 [推文](https://x.com/natolambert/status/1859643355698786549)。
   - 该方法旨在通过在训练期间严格激励正确回答，确保模型生成结果具有更高的准确性。
- **Anthropic 的 Model Context Protocol**：Anthropic 的 **Claude Desktop** 现在支持 [Model Context Protocol (MCP)](https://x.com/btibor91/status/1859385266328531198)，能够通过 Python 和 TypeScript SDK 增强与模型的本地交互。
   - 虽然初步支持包括各种 SDK，但远程连接功能尚待未来更新，这引发了人们对扩展功能的兴趣。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **SageAttention 增强了 Attention 机制**：[SageAttention](https://arxiv.org/abs/2410.02367) 方法为 **attention 机制**引入了一种高效的量化方法，通过优化计算资源提升了每秒操作数。
   - 该技术在保持**准确性**的同时，解决了与长序列相关的高复杂度问题，使其成为对传统方法的一项极具价值的改进。
- **使用自定义 Handler 文件部署 AI 模型**：**Hugging Face Endpoints** 现在支持使用 [handler.py](https://huggingface.co/philschmid/distilbert-onnx-banking77/blob/main/handler.py) 文件部署自定义 AI 模型，从而实现定制化的预处理和后处理。
   - 实现 [EndpointHandler](https://huggingface.co/philschmid/distilbert-onnx-banking77/blob/main/handler.py) 类可确保模型部署的灵活性和高效性，满足特定的部署需求。
- **开发出自动化 AI 研究助手**：一个新的 Python 程序将本地 **LLMs** 转换为自动化网页研究员，根据用户查询提供详细的摘要和来源。
   - 该助手将查询系统地分解为子主题，提高了从各种在线来源收集和分析信息的效率。
- **YOLO 在视频目标检测中表现出色**：**YOLO** 仍然是视频目标检测的首选，[YOLO-VIDEO](https://huggingface.co/spaces/prithivMLmods/YOLO-VIDEO) 资源为有效实现提供了支持。
   - 讨论强调了优化 YOLO 在视频流中性能的持续策略，解决了与实时处理相关的挑战。
- **MOUSE-I 简化了 Web 服务部署**：**MOUSE-I** 能够利用 [AI 自动化](https://huggingface.co/spaces/VIDraft/mouse1)在 **60 秒**内将简单的 prompt 转换为全球部署的 Web 服务。
   - 该工具非常适合寻求快速部署解决方案而无需进行大量手动配置的初创公司、开发人员和教育工作者。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Perplexity 在准确性上优于 ChatGPT**：用户将 **Perplexity** 与 **ChatGPT** 进行了比较，强调 **Perplexity** 被认为更准确，并在特定功能上具有优势。
   - 一位参与者指出，**Perplexity** 的某些功能在流行度飙升之前就已经在开发中。
- **GPT-4 提升产品分类效率**：一位成员分享了他们使用 **GPT-4** 的 prompt 对产品进行分类的经验，涵盖了从**杂货**到**服装**的各类目，效果显著。
   - 他们指出，虽然分类效果很好，但由于 prompt 结构较长，**token 使用量**很高。
- **GPT-4o 增强了基于图像的分类**：一位成员描述了使用 **GPT-4o** 根据标题和图像对产品进行分类，通过全面的 prompt 结构实现了**极佳的效果**。
   - 然而，他们指出大量的 **token 使用量**对其系统的可扩展性构成了挑战。
- **通过 Prompt 优化简化流程**：讨论集中在如何在保持分类任务中 prompt 有效性的同时，最小化 **token 使用量**。
   - 建议包括探索 **prompt 缓存**等方法，以简化流程并减少冗余。
- **Prompt 缓存降低 Token 消耗**：成员们建议实施 **prompt 缓存技术**，以减少分类工作流中重复的输入 token。
   - 他们建议咨询 API 相关资源，以进一步协助优化 **token 使用量**。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Hermes 3 超出预期**：一位成员青睐 [Hermes 3](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-70B)，因其卓越的写作技巧和对 Prompt 的遵循能力，但指出在高上下文（**16K+**）下会出现短语重复。
   - 这一偏好凸显了 **Hermes 3** 的进步，但也暴露了其在高效处理长上下文方面的局限性。
- **基于云端的 GPU 租赁**：转向云服务器进行模型托管的成本为 **每月 25-50 美元**，且与本地硬件相比提升了模型速度。
   - 用户发现云端托管的 GPU 具有更高的性价比和性能，避免了本地部署的限制。
- **LLM GPU 对比**：成员们对比了 **AMD** 和 **NVIDIA** GPU，最近的驱动更新影响了 AMD 的 ROCM 支持。
   - 共识倾向于选择 **NVIDIA**，因为它在 AI 应用中具有更好的软件兼容性和支持。
- **混合 GPU 配置阻碍性能**：由于共享资源限制，包含 **1x 4090 + 2x A6000** GPU 的配置表现不如其他配置，降低了 Token 生成速率。
   - 用户强调，配置中最慢的 GPU（例如 **4090**）可能会限制整体处理速度。
- **2000 美元本地 LLM 服务器的可行性**：在 **2000 美元** 预算下为 **2-10 个用户** 搭建本地 LLM 服务器，在单 GPU 并发方面面临挑战。
   - 开发者建议使用云解决方案，以缓解与预算和旧硬件相关的性能瓶颈。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Qwen 2.5 性能媲美 GPT-4o**：像 **Qwen 2.5 32B** 这样的开源模型在 Aider 的[代码编辑基准测试](https://aider.chat/2024/11/21/quantization.html)中表现出极具竞争力的性能，与 **GPT-4o** 相当，而效果最差的版本则与 **GPT-3.5 Turbo** 持平。
   - 用户强调了模型 **Quantization**（量化）的重大影响，指出不同的量化级别会导致显著的性能差异。
- **Aider v0.64.0 引入新功能**：最新的 **Aider v0.64.0** 版本包含了用于 Prompt 编写的新 [`/editor`](https://aider.chat/docs/usage/commands.html) 命令，并全面支持 **gpt-4o-2024-11-20**。
   - 此更新增强了 Shell 命令的清晰度，允许用户查看确认信息并无缝选择加入 [Analytics](https://aider.chat/docs/more/analytics.html)。
- **Gemini 模型增强 AI 能力**：截至 **2024 年 11 月 21 日**，[Gemini Experimental Model](https://ai.google.dev/gemini-api/docs/models/experimental-models) 提供了改进的 **Coding**、**Reasoning** 和 **Vision** 能力。
   - 用户正在利用 Gemini 的高级功能来实现更复杂的 AI 交互并提高编码效率。
- **模型量化影响讨论**：正如 Aider 的[量化分析](https://aider.chat/2024/11/21/quantization.html)所强调的，**Model Quantization** 显著影响 AI 性能，尤其是在代码编辑方面。
   - 社区讨论了如何优化量化级别，以有效平衡性能和资源利用率。
- **DeepSeek-R1-Lite-Preview 提升推理能力**：**DeepSeek** 推出了 **DeepSeek-R1-Lite-Preview**，在代码基准测试中展示了令人印象深刻的推理能力，详见其[最新发布](https://api-docs.deepseek.com/news/news1120)。
   - 像 [Zhihong Shao](https://x.com/zhs05232838/status/1859201857593524352) 这样的用户称赞了它在编码和数学挑战中的表现，强调了其实际应用价值。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 发布五个新模型**：OpenRouter 推出了具有改进文本能力的 **GPT-4o**，以及 **Mistral Large** ([link](https://openrouter.ai/mistralai/mistral-large-2411))、**Pixtral Large** ([link](https://openrouter.ai/mistralai/pixtral-large-2411))、**Grok Vision Beta** ([link](https://openrouter.ai/x-ai/grok-vision-beta)) 和 **Gemini Experimental 1114** ([link](https://openrouter.ai/google/gemini-exp-1114))。
   - 这些模型提升了各项 Benchmark 的表现，为 AI 工程师提供了可供探索的高级功能。
- **Mistral Medium 已弃用，建议使用替代方案**：**Mistral Medium** 模型已被弃用，由于 **priority not enabled** 导致访问错误。
   - 建议用户切换到 **Mistral-Large**、**Mistral-Small** 或 **Mistral-Tiny** 以继续使用服务而不受干扰。
- **Gemini Experimental 1121 发布并带来升级**：**Gemini Experimental 1121** 模型已发布，在 coding、reasoning 和 vision 能力方面有所增强。
   - 尽管与 **LearnLM** 模型共享配额限制，社区仍渴望评估其性能和潜在应用。
- **OpenRouter API 文档澄清**：用户对 **OpenRouter API** 文档中关于 **context window** 功能的描述表示困惑。
   - 建议提高文档清晰度，以协助与 LangChain 等工具的无缝集成。
- **请求为 Claude 3.5 提供自定义 Provider Key**：一名成员请求为 **Claude 3.5 Sonnet** 提供 **custom provider key**，因为在主 **Claude app** 上的使用额度已耗尽。
   - 该请求旨在提供一种替代方案来管理使用限制并提升用户体验。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Flux 的 VRAM 烦恼**：成员们讨论了有效使用 **Flux** 的**资源需求**，指出它需要大量的 **VRAM** 且生成图像速度较慢。[Black Forest Labs](https://x.com/bfl_ml/status/1859616264324284619?t=DftoDEhtAigmD4sQvsMl2w&s=19) 发布了 **FLUX.1 Tools**，增强了对其基础模型的控制和可引导性。
   - 一位成员强调，使用 **Loras** 可以增强 **Flux** 在 NSFW 内容方面的输出，尽管 **Flux** 并非专门为此目的训练。
- **优化 SDXL 性能**：对于 **SDXL**，应用 `--xformers` 和 `--no-half-vae` 等最佳实践可以提高在 **12GB VRAM** 系统上的性能。成员们指出，**SDXL** 的衍生模型 **Pony** 需要特殊 token，并且与 **XL Loras** 存在兼容性问题。
   - 这些配置有助于提高 **SDXL** 效率，而 **Pony** 的局限性突显了模型兼容性方面的挑战。
- **使用 SDXL Lightning 增强图像提示词**：一位用户询问如何通过 Python 在 **SDXL Lightning** 中**使用图像提示词 (image prompts)**，特别是将照片插入特定环境中。这展示了社区对结合图像提示词与多样化背景以提升生成能力的兴趣。
   - 讨论表明，利用 Python 集成来增强 **SDXL Lightning** 在图像生成任务中的灵活性已成为一种趋势。
- **缓解生成时间过长的问题**：在使用各种模型时，随机出现的**生成时间过长**令人沮丧，引发了对潜在原因的讨论。成员们推测，内存管理问题（如将资源加载到 **VRAM**）可能会导致这些减速。
   - 解决这些延迟对于改善用户体验至关重要，建议指向优化 **VRAM** 使用以提高生成速度。
- **保障 AI Model 使用安全**：有成员对收到索要钱包地址等个人信息的异常请求表示担忧，导致成员怀疑社区内存在**诈骗者**。鼓励用户报告此类事件以维护**安全**环境。
   - 社区强调安全性，寻求通过主动应对与 **AI model** 滥用相关的潜在威胁来保护其成员。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **上下文位置编码 (CoPE) 提升模型适应性**：一项关于 **[Contextual Position Encoding (CoPE)](https://arxiv.org/abs/2405.18719)** 的提案建议根据 token 上下文而非固定计数来调整位置编码，从而产生更具 **表现力的模型 (expressive models)**。该方法旨在改进对 **Flip-Flop** 等选择性任务的处理，而传统方法在这些任务中表现不佳。
   - 成员们讨论了 CoPE 增强位置编码适应性的潜力，这可能在需要细致理解 token 关系的复杂 NLP 任务中带来更好的性能。
- **Forgetting Transformer 在长上下文任务中超越传统架构**：**Forgetting Transformer** 是一种引入了遗忘门 (forget gate) 的变体，与标准架构相比，在 **长上下文任务 (long-context tasks)** 上表现出更好的性能。值得注意的是，该模型消了对位置嵌入 (position embeddings) 的需求，同时在扩展训练上下文中保持有效性。
   - Forgetting Transformer 的引入为通过更有效地管理长期依赖关系来提升 **LLM 性能** 指明了一个充满希望的方向。
- **Sparse Upcycling 通过推理权衡提升模型质量**：最近的一篇 **[Databricks 论文](https://arxiv.org/abs/2411.08968)** 评估了 **sparse upcycling** 与持续预训练在增强模型方面的权衡，发现 sparse upcycling 能带来更高的 **模型质量**。然而，这种改进伴随着 **40% 的推理时间增加**，凸显了部署方面的挑战。
   - 研究结果强调了在 **模型性能** 与实际部署约束之间取得平衡的难度，强调了模型开发中战略优化方法的必要性。
- **Scaling Laws 以极低的训练成本预测模型性能**：最近的一篇 **[论文](https://arxiv.org/abs/2405.10938)** 介绍了一种观察方法，利用约 100 个公开可用的模型，在不进行直接训练的情况下开发 **scaling laws**，从而能够根据 **规模 (scale)** 预测语言模型的性能。该方法突出了训练 **效率 (efficiency)** 的差异，提出性能取决于一个低维的能力空间。
   - 研究表明，**scaling law 模型** 虽然成本高昂，但仍远低于训练完整目标模型的成本；据报道，Meta 在此类预测上的支出仅为目标模型预算的 **0.1% 到 1%**。
- **lm-eval 增强了剪枝模型的基准测试**：一位用户询问当前版本的 **lm-eval** 是否支持剪枝模型（特别是使用 **WANDA** 的模型）的 zero-shot 基准测试，并遇到了库版本过旧的问题。讨论建议查阅文档以了解现有的限制。
   - 为了解决与 **Groq API** 的集成问题，建议在 `OPENAI_API_KEY` 环境变量中设置 API key，这成功解决了无法识别 API key 参数的问题。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 推出高级功能**：成员们讨论了 **Perplexity Pro** 的各项功能，强调了 Pro 用户可以使用更先进的 **models**，以此将其与 **ChatGPT** 区分开来。
   - 讨论中包含了关于 **搜索 (search)** 和 **工具集成 (tool integration)** 的见解，这些功能提升了用户体验。
- **宝可梦数据助力新型 AI 模型**：一段 [YouTube 视频](https://www.youtube.com/embed/hQhP7ipvgx0) 探讨了如何利用 **Pokémon 数据** 开发新型 **AI 模型**，提供了关于游戏领域技术进步的见解。
   - *这可能会改变数据在 AI 应用中的利用方式*。
- **NVIDIA 的 Omniverse Blueprint 变革 CAD/CAE**：一位成员分享了关于 **NVIDIA Omniverse Blueprint** 的见解，展示了其在设计和模拟领域对 **CAD** 和 **CAE** 的变革潜力。
   - *许多人对其如何将先进技术整合到传统工作流中感到兴奋*。
- **讨论“自带 API Key”模式的采用**：一位成员询问了关于 **自带 API key (bring your own API key)** 构建基于 **Perplexity** 的替代平台的许可问题，并概述了安全的数据管理实践。
   - 这种方法涉及将用户提供的 key 进行 **加密** 并 **存储在 cookies 中**，这引发了关于是否符合 **OpenAI 标准** 的疑问。
- **增强前端应用中的会话管理**：针对简化请求，一位用户通过将 **会话管理 (session management)** 类比为 **存储 session ID 的 cookies**，解释了 Web 应用中的会话管理。
   - 讨论强调了用户身份验证如何依赖于验证会话，而不是直接存储敏感数据。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Truffles 设备引起关注**：被描述为“白色云状半透明物体”的 **Truffles** 设备支持 LLM 的自托管。欲了解更多信息，请访问 [Truffles](https://x.com/itsalltruffles)。
   - 一名成员幽默地称其为“发光的乳房植入物”，突显了其独特的外观。
- **Vercel 收购 Grep 以增强代码搜索**：Vercel 宣布收购 [Grep](https://grep.app/)，以增强开发者在超过 **500,000** 个公共仓库中搜索代码的工具。
   - Grep 的创始人 Dan Fox 将加入 Vercel 的 AI 团队以推进这一能力。
- **Tülu 3 在任务表现上超越 Llama 3**：经过两年开发的 [Tülu 3](https://allenai.org/papers/tulu-3-report.pdf)，通过新的 SFT 数据和优化技术，在特定任务上优于 **Llama 3.1 Instruct**。
   - 项目负责人对他们在 **RLHF** 方面的进展感到兴奋。
- **Black Forest Labs 发布 Flux Tools**：Black Forest Labs 推出了 **Flux Tools**，具有用于图像处理的 inpainting 和 outpainting 功能。用户可以在 [Replicate](https://replicate.com/black-forest-labs) 上运行它。
   - 该套件旨在为其 text-to-image 模型增加可控性。
- **Google 发布 Gemini API 实验性模型**：**Gemini** 发布了新的实验性模型，增强了编程能力。
   - 详情请参阅 [Gemini API 文档](https://ai.google.dev/gemini-api/docs/models/experimental-models)。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DeepSeek R1-Lite 提升 MATH 性能**：传闻称 **DeepSeek R1-Lite** 是一个拥有 **2.4B 激活参数** 的 **16B MOE** 模型，根据一条 [推文](https://x.com/nrehiew_/status/1859265550767067518)，它将 MATH 分数从 **17.1** 显著提升至 **91.6**。
   - 由于成员们对 [微信公告](https://x.com/nrehiew_/status/1859265550767067518) 表示怀疑，认为如此巨大的性能跨越可行性存疑，因此引发了争议。
- **Llama-Mesh 论文引起关注**：一名成员建议阅读 **llama-mesh 论文**，称赞其见解在群组中“相当不错”。
   - 这一建议是在关于推进 AI 架构和协作研究的更广泛对话中提出的。
- **多 Agent 框架面临输出多样性限制**：有人担心，在像“AI 企业家”这样的多 Agent 框架中使用反 Token 化输出，可能会因为丢弃了 KV caches 而导致 **隐藏信息丢失**。
   - 这种潜在的信息丢失可能是导致此类系统中观察到的 **输出多样性受限** 的原因之一。
- **Soft Prompts 落后于 Fine Tuning**：**Soft prompts** 往往被 **fine tuning** 和 **LoRA** 等技术掩盖，后者被认为在开源应用中更有效。
   - 参与者强调，soft prompts 存在泛化能力有限的问题，并且在性能和优化方面需要权衡。
- **CoPilot Arena 发布初始排名**：**CoPilot Arena** 的首届结果在 [LMarena 的博客](https://blog.lmarena.ai/blog/2024/copilot-arena/#initial-leaderboard-and-results) 上揭晓，显示出参与者之间竞争激烈。
   - 然而，该分析仅考虑了旧版本的 **Sonnet**，引发了关于在比赛中使用过时模型所产生影响的讨论。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton Kernel 调试与 GEMM 优化**：用户解决了 **Triton interpreter** 的准确性问题，并讨论了通过 **block size 调整**和 **swizzling 技术**实现的性能提升，参考了 [triton.language.swizzle2d](https://triton-lang.org/main/python-api/generated/triton.language.swizzle2d.html) 等工具。
   - **Triton GEMM** 在 ROCm 上表现出的 **无冲突 (conflict-free)** 性能令人惊讶，引发了关于优化 **GEMM 操作**以提高计算效率的讨论。
- **行优先顺序的 cuBLAS 矩阵乘法**：重点讨论了 `cublasSgemm` 面临的挑战，特别是关于**行优先 (row-major)** 与**列优先 (column-major)** 顺序的操作，详见[相关的 Stack Overflow 帖子](https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication)。
   - 用户辩论了在非方阵矩阵乘法中使用 `CUBLAS_OP_N` 与 `CUBLAS_OP_T` 的影响，并指出了与现有代码库的**兼容性问题**。
- **MI250 GPU 上的 ROCm 编译与 FP16 GEMM**：开发者报告了使用 ROCm 的 `make` 命令时**编译时间**过长的问题，尝试调整 `-j` 标志但改进有限。
   - 关于 MI250 GPU 上 **FP16 GEMM (v3)** 的**输入形状变换**存在困惑，导致用户请求澄清**共享内存 (shared memory)** 和输入形状的相关问题。
- **训练后 (Post-Training) AI 技术的进展**：发布了一项新的综述 [Tulu 3](https://allenai.org/papers/tulu-3-report.pdf)，涵盖了 **RL 中的人类偏好 (Human Preferences in RL)** 和**持续学习 (Continual Learning)** 等**训练后方法**。
   - 讨论了关于 **Constitutional AI** 和**递归摘要 (Recursive Summarization)** 框架的研究，强调了利用**人类反馈**来增强**任务性能**的模型。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 的 GitHub 遍历限制**：用户报告称 **NotebookLM** 难以通过输入仓库主页来遍历 GitHub 仓库，因为它缺乏网站遍历能力。一位成员建议将站点转换为 [Markdown](https://discord.com/channels/1124402182171672732/1124403655819415592/1309003599489007686) 或 PDF 以改进处理效果。
   - 无法直接处理网站使得使用 NotebookLM 进行仓库分析变得复杂，导致用户不得不采用手动内容转换等变通方法。
- **音频提示词 (Audio Prompt) 生成的增强**：一位用户提议通过提供特定的提示词来增强 **NotebookLM**，以生成更具影响力的音频输出，从而改善解释效果和主题理解。
   - 正如社区所讨论的，该策略旨在通过更清晰的音频内容促进对指定主题的深入理解。
- **为特定任务集成多个 LLM**：社区成员分享了针对特定需求定制的使用多个大语言模型 (LLM) 的工作流，并对 **NotebookLM** 的生成能力表示赞赏。
   - 这种方法强调了结合各种 AI 工具来支持基于对话的项目（如用户博客文章中所述）的有效性。
- **ElevenLabs 在文本转语音 (TTS) AI 领域的统治地位**：讨论强调 **ElevenLabs** 是领先的文本转语音 AI，表现优于 RLS 和 Tortoise 等竞争对手。一位用户回忆了在该初创公司融资轮之前的早期体验。
   - **ElevenLabs** 对语音合成和无脸视频创作的影响被强调为行业内的变革性工具。
- **NotebookLM 的稳定性与安全标记 (Safety Flags) 问题**：用户注意到 **NotebookLM** 内部的**安全标记**和不稳定性有所增加，导致功能受限和任务受阻。
   - 社区成员建议通过私信 (DM) 提供示例以便调试，并将这些瞬态问题归因于正在进行的应用程序改进。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **使用 LlamaIndex 和 Redis 构建 AI Agent 架构**：参加即将于 [12 月 12 日](https://twitter.com/llama_index/status/1859354663793066029)举行的网络研讨会，学习如何使用 [LlamaIndex](https://twitter.com/llama_index/status/1859354663793066029) 和 **Redis** 构建 **agentic systems** 以分解复杂任务。
   - 参与者将了解降低**成本**、优化**延迟**的最佳实践，并获得关于**语义缓存 (semantic caching)** 机制的见解。
- **使用 Memgraph 和 LlamaIndex 构建知识图谱**：学习如何设置 **Memgraph** 并将其与 [LlamaIndex](https://twitter.com/llama_index/status/1859658719082041802) 集成，从而从非结构化文本数据中构建**知识图谱 (knowledge graph)**。
   - 本次会议将探讨对构建的图谱进行**自然语言查询**，以及有效**可视化连接**的方法。
- **使用 LlamaParse 进行 PDF 表格提取**：一位成员推荐使用 [LlamaParse](https://github.com/run-llama/llama_parse) 从 PDF 文件中提取表格数据，并强调了其在实现最优 RAG 方面的有效性。
   - 分享了一个提供信息的 [GitHub 链接](https://github.com/run-llama/llama_parse)，详细介绍了其解析功能。
- **Create-Llama 前端配置**：一位用户询问了寻求 Create-Llama 帮助的最佳频道，特别是关于在选择 Express 框架时，新版本中缺少 Next.js 前端选项的问题。
   - 另一位参与者确认可以直接在频道中发布查询以获得团队支持。
- **弃用 Llama-Agents 转而使用 Llama-Deploy**：一位成员注意到在升级到 Llama-index 0.11.20 时存在依赖问题，并指出 **llama-agents** 已被弃用，取而代之的是 [llama_deploy](https://github.com/run-llama/llama_deploy)。
   - 他们提供了 [Llama Deploy GitHub 页面](https://github.com/run-llama/llama_deploy)的链接以获取更多背景信息。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **30 天 Python 挑战**：一位成员分享了他们参加 **30 Days of Python** 挑战的经历，该挑战强调循序渐进的学习，并利用 [GitHub 仓库](https://github.com/Asabeneh/30-Days-Of-Python)获取资源和灵感。
   - 他们正积极参与该仓库的内容，以便在这个结构化的 30 天计划中提升 Python 技能。
- **毕业设计项目 API**：一位成员表示倾向于在他们的毕业设计项目中使用 **Go** 来开发 API，突显了在实际应用中对不同编程语言的探索。
   - 他们的选择反映了社区对利用 Go 的并发特性构建健壮 API 的兴趣。
- **Cohere GitHub 仓库**：一位成员强调 **Cohere GitHub 仓库**（[GitHub 链接](https://github.com/cohere-ai)）是贡献者的绝佳起点，展示了各种项目。
   - 他们鼓励探索仓库中的可用工具，并在不同项目中分享反馈或新想法。
- **用于 RAG 应用的 Cohere Toolkit**：**Cohere Toolkit**（[GitHub 链接](https://github.com/cohere-ai/cohere-toolkit)）被提及为一个专门为 **RAG 应用**设计的高级 UI，旨在促进快速构建和部署。
   - 该工具包包含一系列预构建组件，旨在提高用户生产力。
- **多模态 Embedding 发布**：分享了关于**多模态 Embedding (multimodal embeddings)** 改进的令人振奋的更新，计划于明年年初在 **Bedrock** 和合作伙伴平台上发布。
   - *一名团队成员将标记速率限制 (rate limit) 问题*以供进一步讨论，从而解决可扩展性方面的疑虑。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 的 Async 功能正在开发中**：成员报告称 **Mojo 的 async 功能**目前正在开发中，尚无现成的 async 函数可用。
   - 编译器目前将同步代码转换为异步代码，导致在 async 调用期间执行的是同步操作。
- **Mojo 社区频道发布**：为了方便成员互动，已发布专门的 **Mojo 社区频道**，可通过 [mojo-community](https://prefix.dev/channels/mojo-community) 访问。
   - 该频道作为 Mojo 开发和使用相关持续讨论的中心枢纽。
- **Moonshine ASR 模型在 Mojo 上的性能**：使用 [moonshine.mojo](https://gist.github.com/keveman/ea167957fb6364470cb265c5d9aa9da1) 和 [moonshine.py](https://gist.github.com/keveman/d2aea1a059c9a14972783ede2d6b6862) 对 **Moonshine ASR 模型**进行了基准测试。处理 **10s** 语音的执行时间为 **82ms**，而 ONNX runtime 为 **46ms**。
   - 这表明 Mojo 和 Python 版本相比优化后的 ONNX runtime 慢了 **1.8x**。
- **Mojo 脚本优化挑战**：开发者在 **Mojo 脚本**中传递 `TensorMap` 时遇到了 **Model.execute** 崩溃的问题，由于不支持解包（unpacking），必须手动列出参数。
   - 这些问题突显了 Mojo 代码开发中对脚本优化和改进规范的需求。
- **Mojo 模型中的 CPU 利用率**：用户观察到在 Mojo 中运行模型时 **CPU 利用率**不一致，全量 CPU 能力和超线程被忽略。
   - 这表明需要进一步优化以在模型执行期间最大化资源利用率。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 更新了贡献者指南**：团队发布了 [新指南](https://discord.com/channels/1216353675241590815/1236040539409879170/1309124519134494720) 以帮助 **Torchtune** 维护者和贡献者了解所需功能。
   - 这些指南明确了何时使用 fork 与示例仓库进行演示，从而简化了贡献流程。
- **为 Torchtune 提议扩展包**：一位成员建议引入类似 **torchtune[simpo]** 和 **torchtune[rlhf]** 的扩展包（extender packages），以简化包的包含。
   - 该提案旨在降低复杂性，并在不进行过度检查的情况下有效管理资源问题。
- **针对 max_global_bsz 的二分搜索策略**：建议为 **max_global_bsz** 实现一种“最后成功”的二分搜索方法，默认值为小于数据集的 2 的幂。
   - 该策略将把 **max_iterations** 作为参数以提高效率。
- **关于 UV 易用性的反馈**：一位成员询问了其他人使用 **UV** 的经验，寻求关于其易用性的意见。
   - 另一位成员部分肯定了它的效用，指出它看起来很有吸引力且现代。
- **可选包解决 TorchAO 问题**：讨论了可选包功能是否可以解决用户手动下载 **TorchAO** 的需求。
   - 回复表明，虽然它可能提供一些解决方案，但仍有其他考虑因素需要处理。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Prompt Signature 修改**：一位成员询问如何为了调试目的修改 **prompt signature 格式**，以避免可解析的 **JSON schema 注释**，特别是通过构建 **adapter**。
   - 讨论探索了构建自定义 adapter 等方法，以在 DSPy 中实现 **prompt signature 定制**。
- **DSPy 中的 Adapter 配置**：一位用户建议构建一个 **adapter** 并使用 `dspy.configure(adapter=YourAdapter())` 进行配置以修改 prompt，并指向 `dspy/adapters/` 目录中的现有 adapter 以获取进一步说明。
   - 利用 DSPy 中现有的 adapter 可以帮助实现有效的 **prompt signature 定制**。
- **针对特定情况的短语优化**：关于针对 **bool**、**int** 和 **JSON** 等特定类型调整短语的问题，已澄清这些是基于一套维护的 **model signatures**。
   - *这些短语总体上并不高度依赖于具体的语言模型*，这表明了一种通用的公式化方法。

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Intel AMA 会议定于 11 月 21 日**：参加 **11 月 21 日下午 3 点（PT 时间）**举行的 **Intel Hackathon AMA**，直接与 **Intel 专家**交流。
   - 别忘了[在这里观看直播](https://www.youtube.com/watch?v=_Wm5guUXt54)，并为这次 *Ask Intel Anything* 的机会设置提醒。
- **Quiz 10 发布状态更新**：一位成员询问了 **Quiz 10** 的发布状态，该测验尚未在网站上发布。
   - 另一位成员确认，一旦 **Quiz 10** 上线（可能在 **一两天内**），将发送电子邮件通知。
- **Hackathon 频道混淆事件**：一位成员对 **Quiz 10** 的更新表示感谢，但幽默地承认自己在错误的频道询问了 **hackathon** 的事。
   - 这次交流反映了社区内常见的频道混淆情况，为对话增添了轻松的氛围。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **探讨 int64 Indexing 的必要性**：一位用户质疑在不涉及大型 Tensor 的情况下 **int64 indexing** 的必要性，引发了其他人的思考。
   - 另一位用户链接了 [ops_hip.py](https://github.com/tinygrad/tinygrad/blob/master/extra/backends/ops_hip.py) 文件，为该讨论提供更多背景。
- **剖析 ops_hip.py 文件的差异**：一位成员指出了 tinygrad 仓库中两个 **ops_hip.py** 文件之间的区别，认为前者可能因为错误的 import 而未被维护。
   - 他们注意到后者仅在一个外部 benchmarking 脚本的上下文中被引用，而该脚本也包含错误的 import。
- **ops_hip.py 文件的维护状态**：针对维护疑问，另一位用户确认 **extra** 目录下的 ops_hip.py 未被维护，而 **tinygrad** 版本的在设置 **HIP=1** 时应该可以正常工作。
   - 这表明虽然代码的某些部分可能没有被积极管理，但其他部分仍可以配置为正确运行。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **活动链接出现混淆**：一位成员表示在 **Luma** 上找不到活动链接，寻求对其状态的澄清。
   - **Chiphuyen** 表示抱歉，解释说由于生病，活动未能重新安排。
- **祝愿生病的成员早日康复**：另一位成员感谢 **Chiphuyen** 的更新，并祝愿其早日康复。
   - 这体现了社区在活动管理面临挑战时的支持精神。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **紧急寻求 AI 专家**：用户 **michel.0816** 紧急请求 **AI 专家**，表示迫切需要帮助。
   - 另一位成员建议将问题发布在指定频道，以获得更好的曝光度。
- **Carter Grant 的求职信息**：Carter Grant 是一位在 **React**、**Node.js** 和 **AI/ML** 领域拥有 6 年经验的 **full-stack developer**，他发布了自己的求职信息。
   - 他表示渴望为有意义的项目做出贡献。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **MI300X GPU 在六小时后停滞**：一位成员报告称，在使用 **axolotl** 进行持续 **12-19 小时** 的标准 ablation set **8 x runs** 期间，出现了 **间歇性 GPU 挂起**，主要发生在 **6 小时标记** 之后。
   - 这些稳定性问题已被记录并在 [GitHub Issue #4021](https://github.com/ROCm/ROCm/issues/4021) 中进行跟踪，其中包含 **loss** 和 **learning rate** 等详细指标以提供技术背景。
- **正确进行 Prompting？工程师辩论其必要性**：在 **community-showcase** 频道中，一位成员质疑了正确进行 Prompting 的必要性，并分享了一个 [YouTube 视频](https://youtu.be/m3Izr0wNfQc) 来支持讨论。
   - 这一疑问引发了 AI 工程师之间关于 Prompt Engineering 技术当前相关性和有效性的对话。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **训练 Autoencoder**：一位成员强调了 **训练 Autoencoder** 对于实现模型效率的重要性，重点关注增强性能的技术和实现策略。
   - 对话深入探讨了提高 Autoencoder 性能的方法，包括各种训练技术。
- **Autoencoder 架构的复杂性**：成员们讨论了当前模型中 **Autoencoder 架构的复杂性**，探索了先进结构如何提升模型能力。
   - 不同算法的有效性及其对 Autoencoder 内数据表示的影响是讨论的关键点。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Refact.AI 现场演示**：[Refact.AI](https://github.com/smallcloudai) 团队正在进行现场演示，展示他们的 **autonomous agent** 和创新工具。
   - 点击[此处](https://discord.com/events/1089876418936180786/1300459081181429810)加入直播活动并参与对话。
- **Mozilla 发布 Web Applets**：Mozilla 启动了开源项目 **Web Applets**，旨在为 Web 开发 AI 原生应用。
   - 该项目推广 **open standards** 和可访问性，促进开发者之间的协作，详情见[此处](https://discord.com/channels/1089876418936180786/1231977676458168381)。
- **Mozilla 的 Public AI 倡议**：Mozilla 在过去一年中推进了 **14 个本地 AI 项目**，以倡导 **Public AI** 并构建必要的开发者工具。
   - 该计划旨在通过强调社区参与的协作方式，促进开源 AI 技术的发展。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **关于 Llama 3.2 Prompt 格式的咨询**：一位成员询问了关于 **Llama 3.2** 缺少特定 Prompt 使用的问题，并引用了 [Prompt 格式文档](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/#-function-definitions-in-the-system-prompt-)。
   - 该问题强调了对 System Prompt 中 **function definitions** 明确说明的需求，并强调了其对有效使用的重要性。
- **对 Prompt 适用性的兴趣**：对话显示出人们对理解 **Llama 3.2** 中 **applicability of prompts** 的广泛兴趣。
   - 这反映了关于通过 **effective prompting** 最大化模型性能的最佳实践的持续讨论。

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道详细摘要和链接


{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1308928576355172424)** (497 条消息🔥🔥🔥): 

> - `Unsloth 中的 Vision 支持`
> - `微调 Qwen 和 LLaMA 模型`
> - `多模态模型的数据集准备`
> - `许可与法律注意事项`
> - `模型合并与格式兼容性的挑战` 


- **Unsloth 的 Vision 支持已上线**：Unsloth 正式发布了对视觉模型的支持，允许微调 LLaMA、Pixtral 和 Qwen，极大地增强了开发者的能力。
   - 据报道，这一新功能使微调速度提升了 **2 倍**，并将显存占用降低了 **70%**。
- **Qwen 和 LLaMA 的微调挑战**：用户正在讨论微调 Base 模型和 Instruct 模型的可行性，部分用户对创建和合并 LoRA 表示困惑。
   - Unsloth 的 Vision 支持旨在简化合并过程，将 4-bit LoRA 无缝转回 **16-bit**。
- **视觉模型的数据集准备**：分享了为视觉模型创建数据集的技巧，例如 “unsloth/Radiology_mini” 格式，其中包含图像、ID、字幕和分类。
   - 鼓励社区使用这种结构化格式，使模型训练的数据准备工作更加高效。
- **许可与法律注意事项**：参与者讨论了微调 Mistral 模型时的许可影响，一些人反映在联系团队获取许可时遇到了困难。
   - 也有人担心忽视许可条款会对开源事业的未来产生负面影响。
- **合并与格式兼容性问题**：用户遇到了 4-bit 和 16-bit 模型的兼容性问题，通常需要进行 Upcasting 才能成功合并。
   - 强调了理解这些格式对于有效进行模型训练和实现的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://bsky.app/profile/yasomi.xeiaso.net/post/3lbbfnb7uic2k">Mimi (@yasomi.xeiaso.net)</a>: 蒸汽波粉色系浮世绘风格非常硬核</li><li><a href="https://huggingface.co/datasets/unsloth/LaTeX_OCR">unsloth/LaTeX_OCR · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://unslothai.substack.com/">Unsloth AI | Substack</a>: 欢迎订阅 Unsloth 的新闻通讯，我们将在这里分享关于 AI 的技巧、最新发布等内容！我们最近推出了 unsloth.ai 🦥。点击阅读 Unsloth AI，一个 Substack 公开...</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth 文档</a>: 查看下方列表获取我们所有的 Notebook：</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1gwoqm9/llama_32_vision_">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://datta0.substack.com/p/ai-unplugged-23-ngpt-normalised-transformer">AI Unplugged 23: nGPT 归一化 Transformer, LAUREL, TokenFormer</a>: 洞察胜过信息</li><li><a href="https://github.com/unslothai/unsloth/pull/1082">由 wdlctc 在 Unsloth 中引入 MsT 技术以扩展序列长度 · Pull Request #1082 · unslothai/unsloth</a>: 描述：此 Pull Request 对 LLaMA 模型实现进行了优化，特别针对语言建模头（LM Head）和前向传播。主要变化包括：实现了一个 c...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1gwoqm9/llama_32_vision_finetuning_now_in_unsloth_16gb/">Reddit - 深入探索</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1309224660961071195)** (1 条消息): 

> - `Llama 3.2 Vision`
> - `Vision/Multi-modal Models`
> - `Google Colab Notebooks`
> - `Hugging Face Model Uploads`
> - `Fine-tuning Improvements` 


- **Llama 3.2 Vision 提升性能**：Unsloth 现在支持 **Llama 3.2 Vision 模型**，实现了 **2 倍更快**的训练速度和 **70% 的内存节省**，同时支持 **4-8 倍更长的上下文长度**。
   - 这一增强使 Unsloth 的视觉微调能力领先于 **Flash Attention 2 (FA2)** 和 **Hugging Face (HF)** 的基准测试。
- **适用于 Llama 3.2 的 Google Colab Notebooks**：Unsloth 为用户提供了 **Google Colab notebooks**，用于在 **Radiography**（放射影像）和 **Maths OCR to LaTeX**（数学 OCR 转 LaTeX）等任务上微调 Llama 3.2 Vision。
   - 可以通过提供的 [Colab 链接](https://colab.research.google.com/drive/1j0N4XTY1zXXy7mPAhOC1_gMYZ2F2EBlk?usp=sharing) 访问。
- **Hugging Face 上的精彩更新**：Unsloth 的新模型现已在 **Hugging Face** 上线，包括 **11B** 和 **90B** 版本的 Llama 3.2 Vision。
   - 用户可以通过 [Hugging Face](https://huggingface.co/unsloth/) 探索 **Qwen 2 VL** 和 **Pixtral (12B)** 等模型。
- **微调方法的增强**：Unsloth 训练流程的最新改进使得微调时间比之前的标准快了 **1.5-2 倍**。
   - 这种效率有助于为开发者提供工具，以迅速优化其机器学习工作流。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/vision">使用 Unsloth 进行 Llama 3.2 Vision 微调</a>: 通过 Unsloth 以 2 倍速开源微调 Meta 的 Llama 3.2 Vision、Llava、Qwen 2.5 Vision 模型！初学者友好。</li><li><a href="https://colab.research.google.com/drive/1j0N4XTY1zXXy7mPAhOC1_gMYZ2F2EBlk?usp=sharing)">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1whHb54GNZMrNxIsi2wm2EY_-Pvo2QyKh?usp=sharing)">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1K9ZrdwvZRE96qGkCq_e88FgV3MLnymQq?usp=sharing)">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/18sN803sU23XuJV9Q8On2xgqHSer6-UZF?usp=sharing)">Google Colab</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks)">Unsloth 文档</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/all-our-models)">Unsloth 文档</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1308911100925972501)** (1 条消息): 

> - `Training Checkpoints` 


- **选择合适的训练 Checkpoint**：一位成员提出了关于选择哪个训练 checkpoint 的问题，并询问他人的偏好，因为他自己毫不犹豫地选择了 **200** 个 checkpoints。
   - 表达了对训练 checkpoints 多样化方法的关注，暗示了参与者之间可能进行的讨论。
- **关于 Checkpoint 选择的多样化观点**：另一位参与者分享了他们对训练 checkpoint 选择过程的看法，强调了基于特定指标的更具策略性的方法。
   - 他们鼓励他人在确定 checkpoint 选择时考虑性能基准。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1308913840687419492)** (122 条消息🔥🔥): 

> - `模型训练与预处理`
> - `微调流程`
> - `视觉支持`
> - `使用 Ollama`
> - `Kubernetes vs SLURM 训练对比` 


- **关于微调和预分词（pre-tokenized）数据集的查询**：一位用户询问是否可以在已分词的数据集上使用持续预训练（continued pretraining）脚本，另一位成员建议改为传入未分词的数据集。
   - 这次交流突出了训练设置中的挑战以及数据集格式化的具体细节。
- **微调过程中的评估流程**：用户希望在微调期间每 100 步评估一次模型，并寻求实现建议，同时表达了对每次都要从头开始训练的担忧。
   - 建议包括在训练参数中配置评估数据集，并使用 `resume_from_checkpoint` 功能。
- **视觉支持发布公告**：一位成员宣布视觉支持现已上线，在频道中引发了热烈反响。
   - 对该公告的回应包括一些幽默的调侃，关于如何向之前感兴趣的人分享这一信息。
- **关于在生产环境中使用 Ollama 的讨论**：用户讨论了因其简单性以及与 Docker 镜像的易用性而在生产环境中使用 Ollama，并将其与更复杂的系统进行了对比。
   - 也有人对 Ollama 在生产环境中的性能表示担忧，并将其与 VLLM 等替代方案进行了比较。
- **关于 HPC 中 Kubernetes 与 SLURM 的考量**：用户询问了在训练模型（特别是多 GPU 设置）时使用 Kubernetes 和 SLURM 的区别。
   - 对话表明，对于单 GPU 分配，Kubernetes 可以高效运行，但在请求多个 GPU 时，资源管理方面可能会面临挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.unsloth.ai/basics/saving-and-using-models/saving-to-gguf">保存为 GGUF | Unsloth 文档</a>：将模型保存为 16bit 的 GGUF 格式，以便在 Ollama、Jan AI、Open WebUI 等工具中使用！</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining">持续预训练 | Unsloth 文档</a>：又称持续微调（Continued Finetuning）。Unsloth 允许你进行持续预训练，使模型能够学习一种新语言。</li><li><a href="https://docs.unsloth.ai/basics/finetuning-from-last-checkpoint">从最后一个检查点微调 | Unsloth 文档</a>：检查点（Checkpointing）允许你保存微调进度，以便暂停后继续。
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1309116343173251132)** (3 条消息): 

> - `BFloat16 对 RoPE 的影响`
> - `AnchorAttention 方法`
> - `长上下文训练问题` 


- **BFloat16 在长上下文训练中破坏了 RoPE**：一篇新论文讨论了 **FlashAttention2** 中的 **BFloat16** 转换如何导致 **RoPE** 偏离其预期属性，即使 **RoPE** 是在 **Float32** 中计算的。
   - *BFloat16* 引入了关键的数值误差，导致随着上下文长度增加，相对位置编码显著下降。
- **AnchorAttention 提升长上下文性能**：论文介绍了 **AnchorAttention**，这是一种即插即用的解决方案，可增强长上下文性能，同时缩短超过 **50%** 的训练时间。
   - 该方法具有很强的适应性，支持 [FlashAttention](https://github.com/haonan3/AnchorContext) 和 FlexAttention，适用于包括视频理解在内的多种应用。
- **关于 BFloat16 可能导致的问题的讨论**：社区成员询问了 **BFloat16** 造成的“破坏”所带来的影响，推测这可能与现有的循环（looping）问题有关。
   - 他们引用了之前关于这些循环问题的讨论链接，建议进行更深入的调查可能会大有裨益。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2410.08371">瓶中合并：可微分自适应合并 (DAM) 以及从平均到自动化的路径</a>：通过合并模型，AI 系统可以结合独立语言模型的不同优势，在不需要大量重新训练的情况下实现多种能力的平衡。然而，i...</li><li><a href="https://x.com/Haonan_Wang_/status/1859608786765480516">Haonan Wang (@Haonan_Wang_) 的推文</a>：🚀 新论文📜 当精度遇到位置：BFloat16 在长上下文训练中破坏了 RoPE 🤯 RoPE 坏了是因为... BFloat16！ &gt; 即使 RoPE 是在 Float32 中计算的（如 Llama 3 中...
</li>
</ul>

</div>

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1308901426868781098)** (98 条消息🔥🔥): 

> - `Tülu 3 发布`
> - `Nvidia 的 AI 墙`
> - `Gemini 的性能提升`
> - `Reinforcement Learning 技术`
> - `社区关于模型排名的讨论` 


- **Tülu 3 采用新技术发布**：Nathan Lambert 宣布推出 [Tülu 3](https://x.com/natolambert/status/1859643351441535345)，这是一个在多项任务上超越了 Llama 3.1 的开源前沿模型。
   - 该模型包含一种新颖的 Reinforcement Learning with Verifiable Rewards 方法，仅针对正确的生成内容对算法进行奖励。
- **Nvidia CEO 回应 AI 担忧**：《经济学人》的一篇文章讨论了 Nvidia 掌门人如何淡化 AI 已“撞墙”的担忧，尽管外界普遍存在质疑。
   - 这一表态进一步引发了关于 AI 进展现状和创新紧迫性的讨论。
- **Gemini 在 Chatbot Arena 中的排名飙升**：Google DeepMind 最近发布的 [Gemini-Exp-1121](https://x.com/lmarena_ai/status/1859673146837827623) 在 Chatbot Arena 中并列第一，表现优于之前的基准测试。
   - 它在 coding 和 reasoning 能力方面有显著提升，展示了 AI 的快速进步。
- **RL 技术引发社区关注**：社区就 Tülu 开发过程中遇到的迭代和挑战展开了讨论，特别是关于 Reinforcement Learning 方法。
   - 分享的见解包括意外删除关键模型 checkpoint 等问题，这说明了 RL 的复杂多变。
- **AI 模型间的竞争**：随后展开了关于 Claude 和 Gemini 等模型之间竞争的对话，一些社区成员指出，重点应放在实质性改进上，而非仅仅是输出格式。
   - 成员们幽默地评价了 AI 模型的现状，强调了创新与实用性并重的必要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://excalidraw.com/)">Excalidraw — 让协作白板变得简单</a>：Excalidraw 是一款虚拟协作白板工具，可让您轻松绘制具有手绘感的图表。</li><li><a href="https://x.com/_xjdr/status/1859654054345142727">xjdr (@_xjdr) 的推文</a>：@TheXeophon 哈哈，这也是我找的第一件事。我觉得他们确实做得很出色。</li><li><a href="https://x.com/natolambert/status/1859643351441535345">Nathan Lambert (@natolambert) 的推文</a>：在过去的两年里，我搜寻了所有关于 RLHF（特别是）和 post training（广义上）的可用资源。今天，在一个非常厉害的团队的帮助下，我们为您带来了这些劳动的成果……</li><li><a href="https://x.com/cto_junior/status/1859677125793677572">TDM (e/λ) (@cto_junior) 的推文</a>：@TheXeophon @teortaxesTex 令人惊讶的是，如果你应用样式控制 + coding/hard prompts，这两个模型的更新在排名上几乎保持不变。</li><li><a href="https://www.economist.com/business/2024/11/21/nvidias-boss-dismisses-fears-that-ai-has-hit-a-wall">Nvidia 掌门人否认 AI 已撞墙的担忧</a>：但黄仁勋告诉《经济学人》，达到下一个水平是“紧迫的”</li><li><a href="https://x.com/natolambert/status/1859664963763306762">Nathan Lambert (@natolambert) 的推文</a>：所以 @elonmusk，你也能开源 Grok 的 post training 配方吗？ 引用 Nathan Lambert (@natolambert)：在过去的两年里，我搜寻了所有关于 RLHF 的可用资源……</li><li><a href="https://x.com/lmarena_ai/status/1859673146837827623#">lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>：哇，Chatbot Arena 再次传来重大消息🔥 @GoogleDeepMind 刚刚发布的 Gemini (Exp 1121) 强势回归（+20 分），在 Arena 中与最新的 GPT-4o-1120 并列总榜第一🏅！排名提升自……</li><li><a href="https://x.com/natolambert/status/1859643355698786549">Nathan Lambert (@natolambert) 的推文</a>：直接进入有趣的部分。为了完成我们的模型，我们使用了一种名为 Reinforcement Learning with Verifiable Rewards 的新技术，我们在数学题或带有约束的 prompt 上进行训练，并且只奖励……</li><li><a href="https://x.com/OfficialLoganK/status/1859667244688736419">Logan Kilpatrick (@OfficialLoganK) 的推文</a>：向 gemini-exp-1121 问好！我们最新的实验性 Gemini 模型，具有：- coding 性能显著提升 - 更强的 reasoning 能力 - 改进的视觉理解。可在 Goo...</li><li><a href="https://x.com/alexalbert__/status/1859676984768688231">Alex Albert (@alexalbert__) 的推文</a>：Claude 在真正重要的事情上变得更好，而其他实验室还在竞争 Markdown 输出。</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1309226652080803870)** (18 条消息🔥): 

> - `Post-training 能力`
> - `SFT 与新技能`
> - `关于 LLM 能力的辩论`
> - `缩减的哲学章节`
> - `训练效率` 


- **关于 Post-training 能力的辩论**：一位成员提出了一个问题，即 Post-training 仅仅是挖掘出已有的能力，还是诱导产生了全新的能力，并强调了在当前讨论中需要明确这一点。
   - 另一位成员指出，两者皆有可能，特别是它通常会挖掘出模型内部的潜在能力，这引发了进一步的辩论。
- **SFT 在学习中的作用**：成员们讨论了 Supervised Fine-Tuning (SFT) 如何根据所使用数据的数量和质量，既挖掘出 Base Model 的能力，又引入新的能力。
   - 有人指出，虽然使用极少量的数据就能取得令人印象深刻的结果，但在训练中期进行有针对性的 SFT 可能会显著提升性能。
- **哲学章节的缩减**：论文中哲学章节的删减引起了注意，一位成员幽默地感叹一份“迷你宣言”被压缩到了只有一个段落。
   - 这引发了大家对在长篇论文中保持详细讨论完整性所面临挑战的轻松认可。
- **对有限 Flops 的担忧**：一位成员表示担心当前模型的 Flops 可能太低，无法最大限度地发挥 Post-training 技术的潜在收益。
   - 他们强调，虽然从技术上讲它既是挖掘也是诱导新能力，但从实践角度来看，从 Base Model 中获取直觉更为可行。


  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1309007321560780861)** (120 条消息🔥🔥): 

> - `GPT-4o 性能分析`
> - `Perceptron AI 发布`
> - `Anthropic 的 Model Context Protocol`
> - `AI 中的 Rickrolls`
> - `学术界 AI 资源的问题` 


- **GPT-4o 显示性能指标下降**：OpenAI 在 11 月 20 日发布的 GPT-4o 独立评估显示，其质量得分低于 8 月份的版本，这表明尽管输出速度有所提高，但它可能是一个更小的模型。
   - 建议开发者在由于这些变化而从之前的模型迁移之前，仔细测试工作负载。
- **介绍 Perceptron AI 的 Foundational Models**：Perceptron AI 宣布其专注于开发旨在将智能融入物理世界的 Foundational Models，并声称是同类中的首创。
   - 该声明引发了质疑，因为多家公司都做出过类似承诺，引发了对其独特性的疑问。
- **Claude 开始支持 Model Context Protocol (MCP)**：Anthropic 的 Claude Desktop 正在引入对 Model Context Protocol (MCP) 的支持，允许本地连接功能以增强与模型的交互。
   - 最初的支持附带了各种 SDK，但远程连接仍不可用，引发了对未来更新的好奇。
- **AI 交互中的 Rickrolls 成为一个问题**：AI 模型（如 Lindy AI）意外向用户发送 Rick Astley 音乐视频链接的案例，说明了模型响应中幽默但令人担忧的失误。
   - 这个问题强调了模型训练中潜在的差距，导致了用户意料之外且不想要的结果。
- **对 AI 领域劣质学术资源的批评**：对一名学生依赖一本关于 LLM 的劣质学术书籍的挫败感，突显了 AI 教育中的常见挑战，特别是在可靠资源方面。
   - 该批评展示了 AI 领域的教育材料如何误导用户，一些出版物提供了不准确或过于简单的解释。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/andrewcurran_/status/1859430019099131934?s=61">来自 Andrew Curran (@AndrewCurran_) 的推文</a>: 过去一年企业市场份额变化的视觉化。</li><li><a href="https://x.com/damnsec1/status/1610955934683090944">来自 0xDamian (@damnsec1) 的推文</a>: 被 ChatGPT Rickroll 简直太疯狂了。😂</li><li><a href="https://huggingface.co/datasets/allenai/tulu-3-hardcoded-prompts?row=21">allenai/tulu-3-hardcoded-prompts · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://x.com/ArmenAgha/status/1859646650714821012">来自 Armen Aghajanyan (@ArmenAgha) 的推文</a>: 向我们的新公司 Perceptron AI 问好。Foundational Models 改变了数字领域，现在是时候进入物理世界了。我们正在构建首个专为实时设计的 Foundational Models...</li><li><a href="https://www.together.ai/blog/flux-tools-models-together-apis-canny-depth-image-generation">FLUX Tools 现已通过 Together API 提供：通过 Canny、Depth 和 Redux 模型获得对图像生成的更强控制</a>: 未找到描述</li><li><a href="https://x.com/UserMac29056/status/1859478751995899995">来自 User Mac (@UserMac29056) 的推文</a>: gpt-4o-2024-11-20 简单评估已更新。摘要：基准测试性能变差了</li><li><a href="https://x.com/ArtificialAnlys/status/1859614633654616310">来自 Artificial Analysis (@ArtificialAnlys) 的推文</a>: 等等——新的 GPT-4o 是一个更小、更不智能的模型吗？我们已经完成了对 OpenAI 昨天发布的 GPT-4o 的独立评估，并一致测得明显较低的评估...</li><li><a href="https://x.com/btibor91/status/1859385266328531198">来自 Tibor Blaho (@btibor91) 的推文</a>: Anthropic 的 Model Context Protocol (MCP) 正准备在域名 modelcontextprotocol[.]io 上正式亮相——现在，除了 Python SDK 之外，还有 TypeScript SDK + 完整的...</li><li><a href="https://blackforestlabs.ai/flux-1-tools/">FLUX.1 Tools 介绍</a>: 今天，我们很高兴发布 FLUX.1 Tools，这是一套旨在为我们的基础文本生成图像模型 FLUX.1 增加控制力和可引导性的模型，能够实现修改和重新创建...</li><li><a href="https://techcrunch.com/2024/08/21/this-founder-had-to-train-his-ai-to-not-rickroll-people/">这位创始人不得不训练他的 AI 不要 Rickroll 别人 | TechCrunch</a>: 一个 AI 助手从互联网上学到了太多，最终“Rickroll”了一位客户，而不是分享教程视频。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1309204657805197362)** (34 messages🔥): 

> - `RLHF vs DPO`
> - `Cohere and Alignment`
> - `SnailBot RSS Issues` 


- **关于 AI 模型中 RLHF 的辩论**：许多公司最初怀疑 **RLHF** 的必要性，目前关于其相关性的讨论仍在继续。一位成员询问，不进行 RLHF 是否意味着转向 **DPO**，或者仅仅是指训练后的 **Alignment**。
   - “我正和其中一些人共事，需要点料来吐槽他们！”引发了幽默的互动。
- **Cohere 的历史方法**：一位成员指出，Cohere 在很长一段时间内主要利用 **IFT**，现在正向 **RLHF++** 方法演进。由于 **reinforce-loo 论文源自 Cohere**，这种情况引起了关注。
   - 一位成员对他们的方法表示困惑，称考虑到他们的基础性工作，这显得很奇怪。
- **SnailBot 的技术故障**：讨论透露，SnailBot 正面临重复发布 **voiceovers** 的问题，原因是它位于独立的 RSS 馈送中。一位成员建议构建一个自定义解决方案，以更顺畅地管理 RSS 和 Discord webhook 的交互。
   - 有人提议帮助建立一个新的机器人，并强调需要一个专门涵盖帖子内容的 RSS 馈送。


  

---



### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1308902090005024848)** (182 messages🔥🔥): 

> - `Hugging Face Models`
> - `Voice Conversion Using RVC`
> - `Community Interactions`
> - `MagicQuil Downloads`
> - `Model Training Issues` 


- **Hugging Face 模型使用**：用户讨论了在 Hugging Face 上使用各种模型的问题，包括询问如何在 HuggingChat 中获得更大的文本尺寸以提高可见性。
   - 建议包括将 Spaces 设置为私有，以避免在测试模型时受到公共互动的干扰。
- **使用 RVC 进行语音转换**：一位用户询问如何使用来自 Hugging Face 的 RVC-beta.7z 工具，特别是无论是在模型训练还是音频转换期间是否使用 ov2super。
   - 这场对话引发了关于语音转换工具的设置、功能以及对进度预期的提问。
- **MagicQuil 下载协助**：一位用户寻求关于如何下载用 MagicQuil 制作的作品的帮助，这表明对该过程可能存在困惑。
   - 社区成员表示对 MagicQuil 工具缺乏了解，导致该用户没有得到明确的答案。
- **HuggingChat 的调试与错误**：用户对 HuggingChat 交互过程中频繁出现的错误（如“Error while parsing tool calls”）表示担忧，并询问是否是网站过载。
   - 讨论强调了用户对响应生成问题以及使用平台时潜在过载情况的挫败感。
- **社区动态**：该频道展示了成员之间活跃的互动，包括轻松的玩笑以及对社区内共同经历的认可。
   - 成员们表达了对友谊、指导和支持的看法，增强了讨论中的协作精神。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/josephpollack">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/RVC-beta.7z">RVC-beta.7z · lj1995/VoiceConversionWebUI at main</a>: 未找到描述</li><li><a href="https://tenor.com/view/kitty-sleep-kitten-sleep-cuddling-in-bed-cuddle-kitty-cuddle-gif-707600157996049423">Kitty Sleep Kitten Sleep GIF - Kitty sleep Kitten sleep Cuddling in bed - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.runpod.io">RunPod - 为 AI 构建的云</a>: 在一个云端开发、训练和扩展 AI 模型。通过 GPU Cloud 按需启动 GPU，通过 Serverless 扩展 ML 推理。</li><li><a href="https://github.com/CPJKU/madmom">GitHub - CPJKU/madmom: Python 音频和音乐信号处理库</a>: Python 音频和音乐信号处理库。通过创建账号为 CPJKU/madmom 的开发做出贡献。</li><li><a href="https://blackforestlabs.ai/">Black Forest Labs &#x2d; 前沿 AI 实验室</a>: 来自 Black Forest 的惊人 AI 模型。</li><li><a href="https://tenor.com/view/sunday-cult-of-the-lamb-cult-happy-sunday-god-gif-422811577611096801">Sunday Cult Of The Lamb GIF - Sunday Cult of the lamb Cult - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/simpsons-homer-bart-lisa-join-us-gif-17846376318791889140">Simpsons Homer GIF - Simpsons Homer Bart - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/rabbit-bunny-toilet-yes-come-gif-4686108">Rabbit Bunny GIF - Rabbit Bunny Toilet - Discover &amp; Share GIFs</a>: 点击查看 GIF
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1308899823461601280)** (10 条消息🔥): 

> - `使用 Handler 文件的自定义 AI 模型`
> - `AI 安全研究论文`
> - `自动化 AI 研究助手`
> - `LLM 中的协作学习框架`
> - `高效的 Attention 量化方法` 


- **使用 Handler 文件部署自定义 AI 模型**：Hugging Face Endpoints 允许通过 [handler.py](https://huggingface.co/philschmid/distilbert-onnx-banking77/blob/main/handler.py) 文件部署自定义 AI 模型，方便进行自定义的前处理和后处理。
   - 该 handler 需要实现 [EndpointHandler](https://huggingface.co/philschmid/distilbert-onnx-banking77/blob/main/handler.py) 类，从而确保模型部署的灵活性。
- **Redhat/IBM 的 AI 安全见解**：一篇新的 [研究论文](https://huggingface.co/papers/2411.12275) 讨论了与公开可用的 AI 模型相关的风险，并提出了增强 AI 开发安全性和可靠性的策略。
   - 论文强调了诸如追踪问题和缺乏生命周期流程等挑战，旨在促进 AI 生态系统中更标准化的实践。
- **自动化 AI 研究助手问世**：一个创新的 Python 程序将本地 LLM 转化为自动化网络研究员，根据用户查询提供详细的摘要和来源。
   - 该程序智能地将查询分解为子主题，系统地从网络上收集和分析信息。
- **FreeAL：LLM 的协作学习**：论文 [FreeAL](https://arxiv.org/abs/2311.15614) 提出了一种先进的协作学习框架，旨在降低 LLM 训练中的标签标注成本。
   - 通过利用 LLM 作为主动标注器，它以交互方式蒸馏特定任务的知识，以提高数据集样本的质量。
- **SageAttention：Attention 模型中的量化**：[SageAttention](https://arxiv.org/abs/2410.02367) 方法为 Attention 机制提供了一种高效的量化方法，与当前方法相比，显著提高了每秒操作数。
   - 它在解决 Attention 计算限制的同时提高了准确性，而 Attention 在处理长序列时传统上具有很高的复杂度。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2410.02367">SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration</a>：Transformer 架构在各种模型中占据主导地位。作为 Transformer 的核心，Attention 的计算复杂度为 O(N^2)，而线性变换为 O(N)。当 ...</li><li><a href="https://arxiv.org/abs/2311.15614">FreeAL: Towards Human-Free Active Learning in the Era of Large Language Models</a>：为模型训练收集高质量的标注数据在各种 NLP 任务中是众所周知的耗时且劳动力密集。虽然有大量的解决方案，例如针对小语言模型的主动学习...</li><li><a href="https://whcompsci.github.io/projects/neural-networks/index.html">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/docs/inference-endpoints/main/en/guides/custom_handler#create-custom-inference-handler">创建自定义 Inference Handler</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/fffiloni/clone-git-repo-to-space">Clone Git Repo To Space - fffiloni 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/papers/2411.12275">论文页面 - Building Trust: Foundations of Security, Safety and Transparency in AI</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1gvlzug/i_created_an_ai_research_assistant_that_actually/">Reddit - 深入探索一切</a>：未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1308916077509611601)** (9 messages🔥): 

> - `Neo's Red Pill Journey` (Neo 的红丸之旅)
> - `Prompting Techniques` (Prompting 技巧)
> - `MOUSE-I Web Service` (MOUSE-I Web 服务)
> - `Cinematic Image Generation` (电影级图像生成)
> - `loadimg Downloads Milestone` (loadimg 下载量里程碑)


- **Neo 的红丸将他送回了 60 年代**：围绕一段 [YouTube 视频](https://youtu.be/JugL1okFCqI?si=zn7wpJFaQJnQcJx3)展开的讨论，探索如果 Neo 的红丸将他传送回 20 世纪 60 年代会发生什么。
   - 成员们对这一分享表示兴奋，强调了视频的酷炫之处。
- **Prompting 的重要性**：一段名为“BAD vs GOOD prompting”的 [YouTube 视频](https://youtu.be/m3Izr0wNfQc)提出了一个问题：在当前的应用中，有效的 Prompting 是否仍然必要。
   - 视频描述邀请观众探索 Prompting 技巧在何时以及如何产生差异。
- **MOUSE-I 彻底改变 Web 开发**：介绍了 MOUSE-I，它利用 [AI automation](https://huggingface.co/spaces/VIDraft/mouse1) 在短短 **60 秒**内将一个简单的 Prompt 转换为全球部署的 Web 服务。
   - 它承诺提供即时结果，适用于初创公司、开发者和教育工作者。
- **别具一格的电影级图像**：在 Hugging Face 上分享的一个项目，允许用户以预设的宽屏长宽比创建超过 **200 万像素**的电影级图像。
   - 该工具的链接可以在[这里](https://huggingface.co/spaces/takarajordan/CineDiffusion)找到。
- **loadimg 下载量突破 500,000 次**：一位社区成员庆祝 **loadimg** 已实现 **500,000 次下载**，并且现在兼容 Hugging Face 和 OpenAI SDKs。
   - 这一里程碑展示了该工具日益增长的普及度和兼容性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://whcompsci.github.io/projects/neural-networks/index.html">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/VIDraft/mouse1">Mouse1 - VIDraft 开发的 Hugging Face Space</a>：未找到描述</li><li><a href="https://youtu.be/m3Izr0wNfQc">BAD vs GOOD prompting</a>：让我们通过这段视频看看，现如今我们是否仍需要进行良好的 Prompting，如果存在差异，那么差异点在哪里。欢迎留言...</li><li><a href="https://huggingface.co/spaces/takarajordan/CineDiffusion">CineDiffusion - takarajordan 开发的 Hugging Face Space</a>：未找到描述</li><li><a href="https://github.com/gmontamat/screen-grep">GitHub - gmontamat/screen-grep: Windows Recall 的开源替代方案</a>：Windows Recall 的开源替代方案。通过在 GitHub 上创建账户来为 gmontamat/screen-grep 的开发做出贡献。</li><li><a href="https://appine.tech/app/flux-image-generation">Appine</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 messages): 

crazypistachecat: 抱歉👍
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1308995464884129893)** (6 messages): 

> - `YOLO for Video Object Detection` (用于视频目标检测的 YOLO)
> - `Stable Diffusion`
> - `Autodistill Training Method` (Autodistill 训练方法)


- **YOLO 在视频目标检测中表现出色**：一位成员强调 **YOLO 支持视频目标检测**，使其成为该任务的经典选择。
   - 目前正在讨论如何有效地使用它，并提供了[相关资源链接](https://huggingface.co/spaces/prithivMLmods/YOLO-VIDEO)以供进一步探索。
- **Stable Diffusion 依然是经典**：另一位成员提到 **Stable Diffusion** 是图像处理各种应用中知名且经典的模型。
   - 这引发了关于其与新模型相比的生命力和有效性的讨论。
- **YOLO 标注方面的困扰**：一位用户表示在让 **YOLO** 在线正确标注时遇到困难，正在寻求指导和解决方案。
   - 对话强调了需要明确的指令或支持才能有效利用该模型。
- **Autodistill 简化模型训练**：一位成员分享了 **Autodistill**，它使用较大的 Foundation Models 来训练较小的监督模型，突出了其效率。
   - 他们为那些有兴趣在最少人工干预下更快训练模型的人提供了详细的 [文档链接](https://docs.autodistill.com/)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.autodistill.com/">首页 - Autodistill</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/prithivMLmods/YOLO-VIDEO">YOLO VIDEO - prithivMLmods 开发的 Hugging Face Space</a>：未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1308922972592934953)** (9 messages🔥): 

> - `Pandas 替代方案`
> - `构建本地 Agent`
> - `快速推理框架`
> - `扩展数据处理` 


- **考虑使用 Polars 以实现更快的数据处理**：一位成员建议在处理大型数据集时使用 **Polars** 作为 **Pandas** 的更快替代方案。
   - *Swetha98* 对该建议表示感谢，并计划进一步探索。
- **用于 GPU 加速的 NVIDIA RAPIDS**：有人推荐了 **NVIDIA RAPIDS**，据称它在 GPU 上运行，有助于提升可扩展性。
   - 然而，*Swetha98* 提到缺乏 GPU 访问权限，这使得该建议的实施变得复杂。
- **本地 Agent 框架对比**：*Pizzadrones* 询问了在构建本地 Agent 方面能与 **AnythingLLM** 和 **Ottobot** 竞争的框架。
   - 讨论集中在聊天中提到的这些框架的替代方案。
- **vLLM 被推崇为顶级推理框架**：*Takarajordan_82155* 指出，**vLLM** 目前被公认为 LLM 的最佳推理框架。
   - 这一观点激发了大家探索不同快速推理框架的兴趣。
- **Llama 3.2 领先的推理性能**：在讨论有效的聊天 LLM 时，*Takarajordan_82155* 提议将 **Groq 上的 Llama 3.2 90B** 作为强有力的竞争者。
   - *Abhijithneilabraham* 询问是否有排行榜可以追踪具有快速推理能力的最佳通用聊天 LLM。


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1309067173980799036)** (13 messages🔥): 

> - `像 SSD-1B 这样的开源模型`
> - `在 Google Colab 中使用 SDXL`
> - `Shuttle-3 中的 Token 嵌入` 


- **SSD-1B 替代方案建议**：成员们讨论了 **SSD-1B** 的替代方案，建议使用 **SDXL** 的步进蒸馏（step distilled）版本或 **DreamShaper Lightning**，以获得潜在更好的质量和速度。
   - 分享了 [Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD) 的链接，强调了它在类似用途上的易用性。
- **在 Google Colab 的低显存（VRAM）环境下加载模型**：针对在 GPU 资源较低的情况下加载模型的问题进行了讨论，确认 **SDXL** 的 **6.5GB** 大小在 Google Colab 的 16GB VRAM 中是可控的。
   - 成员们指出，在加载基础模型后，保存模型状态以便将来使用可能会更容易。
- **Shuttle-3 Pipeline 中的 Token 限制**：一位成员询问是否可以将 **shuttle-3** pipeline 中的 Token 嵌入增加到 **77** 以上，目前似乎不支持。
   - 讨论表明，虽然 **CLIP** 编码器可能会截断 Prompt，但 **T5** 编码器可以毫无问题地处理多达 **256** 个 Token。
- **编码对图像生成的影响**：有人提出了关于截断 **CLIP** 编码器数据的影响，以及是否应该优先考虑重要的 Prompt 数据的问题。
   - 成员们发表了意见，指出虽然 **CLIP** 会发生截断，但可能不会显著影响图像生成的结果。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/ByteDance/Hyper-SD">ByteDance/Hyper-SD · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/tianweiy/DMD2">tianweiy/DMD2 · Hugging Face</a>：未找到描述
</li>
</ul>

</div>

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1308925019723796542)** (209 条消息🔥🔥): 

> - `关于 AI 审查的讨论`
> - `关于不可知论的观点`
> - `深网与 AI 访问`
> - `OpenAI 的审查政策`
> - `Perplexity vs ChatGPT` 


- **关于 AI 审查重要性的辩论**：用户讨论了 AI 审查以防止有害用途的必要性，其中一位成员认为 **OpenAI** 旨在通过实施内容审核来避免**法律麻烦**。
   - 讨论中有人担心审核过程中可能存在过度干预和误报，但普遍共识倾向于采取谨慎的做法。
- **不可知论的主观性解释**：关于不可知论是否具有主观性引发了激烈的辩论，成员们断言它代表的是**证据缺乏**而非个人信仰。
   - 一位用户强调不可知论可以被视为一种**健康的思维方式**，而另一位用户则指出声称“主观意见”是多余的。
- **探索 AI 的能力**：讨论集中在 AI 是否可以访问深网（deep web）以及其理论上的运作方式，一些人建议定制化的 AI 可以扫描未经过滤的网络。
   - 一位用户指出，虽然 AI 本身不直接访问网络，但将外部内容输入其中是可能的，这可能会引发伦理问题。
- **OpenAI 对政治问题的处理方式**：参与者一致认为，允许使用 **ChatGPT** 讨论政治，只要不导致有害或误导性的内容。
   - 成员们认识到，虽然政治可以是对话的一部分，但应避免不必要的政治偏见。
- **AI 模型对比**：用户对比了 **Perplexity** 与 **ChatGPT**，强调 Perplexity 被认为更准确，并且在某些功能上具有优势。
   - 一位参与者指出，Perplexity 的某些特定功能在流行之前就已经在开发中了。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/)** (1 条消息): 

grundaypress: 嗨，有人知道为什么在使用 Custom GPTs 时重试按钮消失了吗？
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1308940975636484126)** (3 条消息): 

> - `使用 GPT-4 进行产品分类`
> - `Prompt 优化`
> - `为了效率的 Prompt Caching` 


- **使用 GPT-4 进行产品分类**：一位成员分享了他们使用 GPT-4 的 Prompt 对产品进行分类的经验，涵盖了从**杂货**到**服装**的各类范畴。
   - 他们强调虽然结果非常好，但由于 Prompt 结构冗长，Token 使用量相当高。
- **Token 削减策略**：讨论围绕如何在保持分类 Prompt 有效性的同时最小化 Token 使用量展开。
   - 建议包括探索 **Prompt Caching** 等方法，以简化流程并减少冗余。
- **API 协助参考**：一位用户提供了一个 Discord 频道的链接，以寻求有关提高 Prompt 效率的 API 相关问题的帮助。
   - 该参考旨在帮助社区中面临类似产品分类挑战的其他成员。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1308940975636484126)** (3 条消息): 

> - `使用 GPT-4o 进行产品分类`
> - `Token 优化策略`
> - `API 使用中的 Prompt Caching` 


- **使用 GPT-4o 分类产品**：一位成员描述了如何使用 GPT-4o 根据标题和图像，通过复杂的 Prompt 结构对产品进行分类。
   - *该 Prompt 产生了极佳的结果，但大量的 Token 使用对可扩展性构成了挑战。*
- **高效 Token 使用建议**：另一位成员建议探索 **Prompt Caching** 技术，以减少其产品分类设置中重复的输入 Token。
   - 他们建议参考消息中链接的 API 相关资源，以进一步协助优化 Token 使用。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1308903972869771284)** (63 messages🔥🔥): 

> - `Hermes 3 模型性能`
> - `基于云端的 GPU 租赁`
> - `LLM GPU 对比`
> - `LMS 中的 MLX 模型`
> - `显卡推荐` 


- **Hermes 3 给用户留下深刻印象**：一位成员表达了对 **Hermes 3** 的偏爱，强调了其强大的写作能力和有效遵循 Prompt 的能力。
   - 然而，他们指出在更高上下文 (**16K+**) 下，它倾向于重复短语。
- **用于模型托管的云服务器**：一位成员报告称，他们从本地模型托管转向了云服务器，每月成本约为 **$25-50**，能更有效地运行模型。
   - 由于性能和成本效益，这种转变被认为优于本地硬件。
- **在 AMD 和 NVIDIA 显卡之间做出选择**：成员们讨论了选择 **AMD** 与 **NVIDIA** GPU 的优缺点，最近的驱动程序影响了 AMD 的 ROCM 支持。
   - 共识是，为了获得更好的软件支持和兼容性，建议坚持使用 NVIDIA。
- **对 MLX 模型集成的关注**：鉴于 LMS 最近对其提供了支持，有人请求发布关于有趣的 **MLX 模型** 的公告。
   - MLX 模型的集成被认为是一项有价值的补充，即使这些模型并非 LMS 官方提供。
- **显卡决策**：讨论集中在 **4070 Ti Super** 和 **7900 XTX** 之间的选择，权衡了游戏性能与 LLM 应用等因素。
   - 一位成员指出，虽然 **3090** 在各种任务中表现出色，但 **7900 XTX** 可能是游戏的更便宜替代方案。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-70B">NousResearch/Hermes-3-Llama-3.1-70B · Hugging Face</a>: 未找到描述</li><li><a href="https://www.techpowerup.com/review/gigabyte-geforce-rtx-4070-ti-super-gaming-oc/">Gigabyte GeForce RTX 4070 Ti Super Gaming OC Review</a>: 技嘉 RTX 4070 Ti Super Gaming OC 名副其实，带有工厂超频，额定加速频率为 2655 MHz。它配备了三槽三风扇散热器和双 BIOS，以增加灵活性...
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1308918403293122595)** (135 条消息🔥🔥): 

> - `AI 芯片讨论`
> - `GPU 性能`
> - `构建本地 LLM 服务器`
> - `AMD 设备上的 USB4`
> - `GPU 配置挑战` 


- **混合 GPU 配置的性能问题**：用户报告称，在配置中增加 GPU 往往会降低性能，基准测试显示 1x 4090 + 2x A6000 的表现由于共享资源上限而差于其他组合。
   - 一位用户指出，在 A6000 配置中加入 4090 会降低 Token 生成速率，这强调了*最慢的显卡往往决定了整体速度*。
- **本地 LLM 服务器搭建考量**：一位用户询问 2,000 美元的预算对于支持 2-10 名用户的本地 LLM 服务器是否可行，考虑到单 GPU 并发访问的挑战。
   - 开发者建议探索云解决方案，以避免使用旧硬件和较少 GPU 的预算配置所带来的性能瓶颈。
- **USB4 与 eGPU 设置见解**：用户讨论了使用特定的 GitHub 工具在较新的 AMD 设备上启用 USB4，同时分享了 A4000 和 P40 等外置 GPU 配置的经验。
   - 一位成员报告 A4000 实现了即插即用，但对 P40 的兼容性和潜在的旧版 PCIe 模式问题仍存疑虑。
- **发货延迟与电商挑战**：讨论了来自 AliExpress 等在线供应商的发货延迟，一位用户强调罢工可能影响了交付时间。
   - 总体而言，成员们对不稳定的物流体验及其对硬件组装进度的影响表示沮丧。
- **利用 AMD 固件进行硬件优化**：有建议称 AMD 笔记本电脑用户可以通过 GitHub 工具访问隐藏的固件菜单来修改设置，从而提升硬件性能。
   - 成员们交流了关于启用 MMIO 和 Above 4G Decoding 等设置如何潜在地改善 eGPU 连接和性能的想法。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/fascinating-mr-spock-henoch-star-trek-the-original-series-gif-23404763">Fascinating Mr Spock GIF - Fascinating Mr Spock Henoch - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/sniped-piggy-piggyverse-sniper-piggy-sniper-gif-2353208795072333005">Sniped Piggy GIF - Sniped Piggy Piggyverse - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/senator-palpatine-anakin-skywalker-phantom-menace-gif-10607636">Senator Palpatine Anakin Skywalker GIF - Senator Palpatine Anakin Skywalker Phantom Menace - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://github.com/v1ckxy/LMSPLS">GitHub - v1ckxy/LMSPLS: LM Studio Portable Launch Script (LMS PLS)</a>：LM Studio 便携式启动脚本 (LMS PLS)。通过在 GitHub 上创建账号为 v1ckxy/LMSPLS 的开发做出贡献。</li><li><a href="https://github.com/DavidS95/Smokeless_UMAF">GitHub - DavidS95/Smokeless_UMAF</a>：通过在 GitHub 上创建账号为 DavidS95/Smokeless_UMAF 的开发做出贡献。</li><li><a href="https://x.com/_Holistech_/status/1859395091384893820/photo/1">来自 Holistech (@_Holistech_) 的推文</a>：@rasbt @ivanfioravanti 我希望它们能与我的 M1 Ultra 一起装入我的 6 屏移动折叠 AI 工作站配置中</li><li><a href="https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference">GitHub - XiongjieDai/GPU-Benchmarks-on-LLM-Inference: Multiple NVIDIA GPUs or Apple Silicon for Large Language Model Inference?</a>：多 NVIDIA GPU 还是 Apple Silicon 用于大语言模型推理？- XiongjieDai/GPU-Benchmarks-on-LLM-Inference
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1309243124467372134)** (2 messages): 

> - `Qwen 2.5 模型性能`
> - `Aider v0.64.0 特性`
> - `模型 Quantization 影响`
> - `Aider 中的 Slash Commands`
> - `Context Window 与 Token 成本` 


- **Qwen 2.5 模型媲美 GPT-4o**：像 **Qwen 2.5 32B** 这样的开源模型在 Aider 的代码编辑基准测试中表现出色，足以媲美闭源的前沿模型，但在 **Quantization** 影响方面存在显著差异。
   - 最佳版本可与 **GPT-4o** 竞争，而效果最差的版本则类似于 **GPT-3.5 Turbo**，这提醒用户需密切关注 **Quantization** 效应。
- **Aider v0.64.0 新特性**：最新版本的 Aider 增加了新的 [`/editor`](https://aider.chat/docs/usage/commands.html) 命令用于编写 **Prompt**，并全面支持 **gpt-4o-2024-11-20**。
   - 此更新提高了 **Shell** 命令的清晰度，使用户能够查看确认信息并选择加入 [analytics](https://aider.chat/docs/more/analytics.html)。
- **理解模型 Quantization**：关注模型 **Quantization** 如何影响性能，特别是在代码编辑方面，因为重度量化的模型在云端解决方案中非常普遍。
   - 讨论强调了不同的推理服务方式及其对性能的影响，引导用户选择更好的模型配置。
- **探索 Slash Commands**：Aider 支持多种 **Slash Commands**，如 `/add`、`/architect` 和 `/chat-mode`，简化了用户在聊天环境中的交互。
   - 这些命令增强了编辑和审查能力，促进了高效的沟通和代码管理。
- **注册 Context Window 限制**：用户可以通过在指定目录中创建 `.aider.model.metadata.json` 文件，为参数未知的模型注册 **Context Window** 限制和成本。
   - 该功能有助于更好的资源管理，适应 Aider 中更广泛的模型配置。


<div class="linksMentioned">

<strong>提及链接</strong>:

<ul>
<li>
<a href="https://aider.chat/2024/11/21/quantization.html">Quantization matters</a>：开源 LLM 正变得非常强大，但请注意你（或你的服务商）是如何对模型进行 Quantization 的。这会极大地影响代码编辑能力。</li><li><a href="https://aider.chat/docs/usage/commands.html#entering-multi-line-chat-messages)">In-chat commands</a>：通过 /add、/model 等聊天内命令控制 Aider。</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html#global-extra-params)">Advanced model settings</a>：为 LLM 配置高级设置。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1308905869056151594)** (133 messages🔥🔥): 

> - `Aider 排行榜变更`
> - `OpenRouter 提供商与 Quantization`
> - `Gemini 模型性能`
> - `DeepSeek 模型进展`
> - `AI 模型用户体验` 


- **Aider 排行榜增强**：Aider 在排行榜表格和图表中引入了新的搜索/过滤框，以提高可用性并允许用户快速找到特定模型。
   - 用户注意到这一改进极大地增强了导航和对模型性能数据的访问。
- **对 OpenRouter 模型选择的担忧**：参与者对 **OpenRouter** 使用 **Quantization** 版本的模型表示担忧，这可能会导致误导性的性能预期。
   - 讨论强调了了解 **Quantization** 级别以及 AI 模型所使用的具体提供商的重要性。
- **Gemini 模型性能见解**：最近的基准测试显示新 **Gemini** 模型的表现参差不齐，标准 **diff** 格式和 **diff-fenced** 格式的结果不同。
   - 虽然 **gemini-exp-1121** 在 **diff** 格式上获得了 58% 的分数，但人们对之前迭代和不同格式中观察到的较低分数感到担忧。
- **DeepSeek 模型更新**：**DeepSeek** 宣布了 **DeepSeek-R1-Lite-Preview** 发布带来的新功能，据称在编程基准测试中表现出色。
   - 用户讨论了使用各种模型的实际意义，包括速度和效率优势。
- **AI 助手用户体验**：用户分享了他们与各种 AI 模型互动的幽默轶事，指出了奇特的行为和意想不到的输出。
   - 社区反思了有效利用不同 AI 工具进行编码和其他任务的学习曲线和挑战。


<div class="linksMentioned">

<strong>提及链接</strong>:

</div>

<ul>
<li>
<a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>: LLM 代码编辑能力的定量基准。</li><li><a href="https://www.alibabacloud.com/help/en/model-studio/developer-reference/use-qwen-by-calling-api">
 使用 Qwen API - 阿里云百炼 (Model Studio) - 阿里云文档中心

</a>: 未找到描述</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html#global-extra-params">高级模型设置</a>: 为 LLM 配置高级设置。</li><li><a href="https://openrouter.ai/qwen/qwen-2.5-coder-32b-instruct">Qwen2.5 Coder 32B Instruct - API、供应商、统计数据</a>: Qwen2.5-Coder 是最新的特定代码 Qwen 大语言模型系列（前身为 CodeQwen）。通过 API 运行 Qwen2.5 Coder 32B Instruct</li><li><a href="https://x.com/zhs05232838/status/1859201857593524352">来自 Zhihong Shao (@zhs05232838) 的推文</a>: 我们的 DeepSeek 推理模型在代码和数学方面表现出色。快来试试吧！引用 DeepSeek (@deepseek_ai) 🚀 DeepSeek-R1-Lite-Preview 现已上线：释放超强推理能力！🔍 o1-preview-...</li><li><a href="https://ai.google.dev/gemini-api/docs/models/experimental-models">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/BerriAI/litellm/issues/6857">[Feature]: 支持 OpenRouter 的 "provider" 参数以控制/选择供应商 · Issue #6857 · BerriAI/litellm</a>: 该功能支持 OpenRouter 的多种机制，用于选择你希望请求命中的供应商。这涉及传递一个 provider 参数。目前这会导致错误：import li...</li><li><a href="https://openrouter.ai/docs/provider-routing#quantization-levels">供应商路由 | OpenRouter</a>: 跨多个供应商路由请求</li><li><a href="https://openrouter.ai/docs/provider-routing">供应商路由 | OpenRouter</a>: 跨多个供应商路由请求</li><li><a href="https://openrouter.ai/models">模型 | OpenRouter</a>: 在 OpenRouter 上浏览模型</li><li><a href="https://www.youtube.com/shorts/7smV_9eVM1M">OpenRouter 上的 Qwen #aider #lmsys #qwen #llm #aicoding #huggingface</a>: 未找到描述</li><li><a href="https://openrouter.ai/deepseek/deepseek-chat">DeepSeek V2.5 - API、供应商、统计数据</a>: DeepSeek-V2.5 是结合了 DeepSeek-V2-Chat 和 DeepSeek-Coder-V2-Instruct 的升级版本。通过 API 运行 DeepSeek V2.5</li><li><a href="https://api-docs.deepseek.com/news/news1120">🚀 DeepSeek-R1-Lite-Preview 现已上线：释放超强推理能力！| DeepSeek API 文档</a>: 🔍 在 AIME 和 MATH 基准测试中达到 o1-preview 级别的性能。</li><li><a href="https://openrouter.ai/meta-llama/llama-3.1-70b-instruct/providers">Meta: Llama 3.1 70B Instruct – 供应商状态</a>: 查看供应商状态并向 Meta: Llama 3.1 70B Instruct 发起负载均衡请求 - Meta 最新的模型系列 (Llama 3.1) 推出了多种尺寸和版本。这款 70B 指令微调...</li><li><a href="https://www.alibabacloud.com/en/solutions/generative-ai/qwen">通义千问 (Qwen) - 阿里云</a>: 来自阿里云的高性能基础模型</li><li><a href="https://www.alibabacloud.com/en/product/modelstudio">阿里云百炼 (Model Studio) - 阿里云</a>: 一个一站式生成式 AI 平台，基于 Qwen 和其他流行模型构建理解业务的智能应用</li><li><a href="https://help.aliyun.com/zh/model-studio/getting-started/models">模型列表_大模型服务平台百炼(Model Studio)-阿里云帮助中心</a>: 未找到描述
</li>
</ul>

</div>

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1308900081478668300)** (42 messages🔥): 

> - `使用 Aider 进行循环`
> - `Aider 缓存效率`
> - `在 Aider 中禁用自动补全`
> - `Aider 的 API 审批`
> - `模型组合推荐` 


- **在 Aider 中循环处理清单**：一位用户询问如何通过运行循环来使用 Aider 自动化清单任务以避免 token 限制，建议包括将 Aider 封装在外部脚本中或使用 `-m` 模式。
   - 另一位成员分享了一种命令行脚本方法，利用 `aider --message` 在 shell 脚本中自动化任务。
- **缓存机制的使用**：一位用户对 Aider 相关的高昂成本表示担忧，引发了关于 `--prompt-caching` 功能有效性以及缓存使用频率的讨论。
   - 建议包括探索 `AIDER_CACHE_PROMPTS` 等缓存设置，以分析缓存的成本节约潜力。
- **禁用自动补全功能**：一位用户寻求帮助以禁用 Aider 中的文件自动补全功能，因为该功能在编码时造成了干扰。
   - 建议包括使用 `--no-fancy-input` 选项，尽管用户更希望在保留某些功能的同时不出现自动补全弹出窗口。
- **Aider 的 IT 审批流程**：讨论了让 IT 和法务部门批准使用 Aider 的挑战，重点关注数据保留政策和知识产权问题。
   - 提到有关分析和隐私政策的文档有助于解决疑虑，同时也指出 API 提供商也有各自的政策。
- **项目的模型组合**：一位用户寻求关于目前哪些模型组合对实际项目有效的见解，暗示了 Qwen2.5.1 和 DeepSeek 等值得关注的模型。
   - 这一咨询反映了人们对在 AI 开发的实际应用中优化模型使用的持续兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/usage/caching.html">Prompt caching</a>：Aider 支持 Prompt 缓存，以节省成本并加快编码速度。</li><li><a href="https://aider.chat/docs/scripting.html">Scripting aider</a>：你可以通过命令行或 Python 对 Aider 进行脚本化操作。</li><li><a href="https://aider.chat/docs/more/analytics.html">Analytics</a>：选择性加入，匿名，无个人信息。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1309236740019458100)** (2 messages): 

> - `Gemini API`
> - `uithub` 


- **Gemini API 提升编码和推理能力**：[Gemini Experimental Model](https://ai.google.dev/gemini-api/docs/models/experimental-models) 自 **2024 年 11 月 21 日**起引入了改进的**编码**、**推理**和**视觉能力**。
   - 此次更新旨在通过更先进的理解和功能来增强用户与 AI 的交互。
- **uithub 重新定义 GitHub 交互**：[uithub.com](http://uithub.com) 允许用户在 GitHub 链接中将 'g' 替换为 'u'，以便即时**复制粘贴**仓库交互内容，从而增强 LLM 上下文。
   - 像 **Nick Dobos** 和 **Ian Nuttall** 这样的用户对该工具表示赞赏，Nuttall 指出它能有效地提供**完整的仓库上下文**。
- **社区拥抱 uithub 工具**：多位用户在 Twitter 上分享了他们使用 uithub 的经验，称其为简化 LLM 编码问题的**实用工具**。
   - **Yohei Nakajima** 对发现 uithub 表现出极大的热情，评价其具有很强的实用性和效用。
- **uithub 提供独特功能**：uithub 的功能被拿来与 **--show-repo-map** 进行比较，表明它会生成更多 token，并为特定文件类型提供高级过滤选项。
   - 然而，它缺乏其他工具中可见的某些复杂功能，导致一些用户在处理复杂任务时仍倾向于使用标准的 Aider 工具。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://ai.google.dev/gemini-api/docs/models/experimental-models">未找到标题</a>：未找到描述</li><li><a href="https://uithub.com/">uithub - 轻松向你的 LLM 提问代码问题</a>：未找到描述
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1309006009829687346)** (2 条消息): 

> - `新模型发布`
> - `高上下文 Provider 选择` 


- **本周推出的新模型**：**GPT-4o** 已发布，具备更好的文本写作能力，详情请见 [此处](https://openrouter.ai/openai/gpt-4o-2024-11-20)。其他新模型包括 **Mistral Large** ([链接](https://openrouter.ai/mistralai/mistral-large-2411))、**Pixtral Large** ([链接](https://openrouter.ai/mistralai/pixtral-large-2411))、**Grok Vision Beta** ([链接](https://openrouter.ai/x-ai/grok-vision-beta)) 以及 **Gemini Experimental 1114** ([链接](https://openrouter.ai/google/gemini-exp-1114))。
- **为高上下文 Prompt 选择 Provider**：用户对于如何选择支持高上下文的 Provider 存在困惑；OpenRouter 会自动路由到支持的 Provider。如果你发送了长 Prompt 或设置了较大的 max tokens，上下文较小或最大输出受限的 Provider 将被自动过滤。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/openai/gpt-4o-2024-11-20">GPT-4o (2024-11-20) - API, Providers, Stats</a>: 2024-11-20 版本的 GPT-4o 提供了更高级别的创意写作能力，写作风格更自然、更具吸引力且更具针对性，提升了相关性和可读性。它在处理...方面也表现更好。</li><li><a href="https://openrouter.ai/mistralai/mistral-large-2411">Mistral Large 2411 - API, Providers, Stats</a>: Mistral Large 2 2411 是 [Mistral Large 2](/mistralai/mistral-large) 的更新版本，与 [Pixtral Large 2411](mistralai/pixtral-large-2411) 一同发布。它精通英语、法语、西班牙语、德语...</li><li><a href="https://openrouter.ai/mistralai/pixtral-large-2411">Pixtral Large 2411 - API, Providers, Stats</a>: Pixtral Large 是一个基于 [Mistral Large 2](/mistralai/mistral-large-2411) 构建的 124B 开源权重多模态模型。该模型能够理解文档、图表和自然图像。运行 Pixtra...</li><li><a href="https://openrouter.ai/x-ai/grok-vision-beta">Grok Vision Beta - API, Providers, Stats</a>: Grok Vision Beta 是 xAI 的具有视觉能力的实验性语言模型。通过 API 运行 Grok Vision Beta</li><li><a href="https://openrouter.ai/google/gemini-exp-1114">Gemini Experimental 1114 - API, Providers, Stats</a>: Gemini 11-14 (2024) 实验性模型具有“质量”方面的改进。通过 API 运行 Gemini Experimental 1114
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1308902211547304018)** (162 messages🔥🔥): 

> - `Mistral 模型问题`
> - `OpenRouter API 功能`
> - `Gemini 实验性模型`
> - `文件上传功能`
> - `OpenRouter 社区参与` 


- **Mistral 模型面临弃用**：用户报告 **Mistral Medium** 模型已被弃用，访问时会出现错误，提示该模型未启用 **priority**。
   - 成员建议切换到 **Mistral-Large**、**Mistral-Small** 或 **Mistral-Tiny** 以继续使用服务。
- **OpenRouter API 文档澄清**：用户对 OpenRouter API 文档中的某些功能表示困惑，特别是关于 Context window（上下文窗口）的能力。
   - 建议增强文档的清晰度，以帮助用户将 OpenRouter 与 LangChain 等工具进行集成。
- **新 Gemini 实验性模型更新**：推出了 **Gemini Experimental 1121** 模型，据称在编程、推理和视觉能力方面有所提升。
   - 用户注意到该模型与 **LearnLM** 模型共享现有的配额限制，并对模型的性能表示好奇。
- **模型的文件上传功能**：讨论了文件上传的限制，用户询问是否有模型接受非图像格式。
   - 澄清了支持图像上传，且最近的基础设施升级可能已经取消了之前 **4MB** 的限制。
- **社区建设与创始人见解**：一位用户询问了 OpenRouter 的创立过程及其创建动机。
   - 社区参与被强调为 OpenRouter 发展的关键因素，并提议撰写一篇关于其故事的文章。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/chat">Chatroom | OpenRouter</a>：LLM Chatroom 是一个多模型聊天界面。添加模型并开始聊天！Chatroom 将数据本地存储在您的浏览器中。</li><li><a href="https://openrouter.ai/liquid/lfm-40b:free">LFM 40B MoE (free) - API, Providers, Stats</a>：Liquid 的 40.3B Mixture of Experts (MoE) 模型。通过 API 运行 LFM 40B MoE (free)。</li><li><a href="https://openrouter.ai/meta-llama/llama-3.1-8b-instruct:free">Llama 3.1 8B Instruct (free) - API, Providers, Stats</a>：Meta 最新的模型系列 (Llama 3.1)，推出了多种尺寸和版本。通过 API 运行 Llama 3.1 8B Instruct (free)。</li><li><a href="https://ai.google.dev/gemini-api/docs/models/experimental-models">未找到标题</a>：未找到描述</li><li><a href="https://tenor.com/view/despicable-me-animation-movies-dream-works-minions-gif-13754998145004207015">Despicable Me Animation GIF - Despicable Me Animation Movies - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://status.openrouter.ai/">OpenRouter Status</a>：OpenRouter 故障历史</li><li><a href="https://openrouter.ai/docs/requests#images-_-multimodal-requests">Requests | OpenRouter</a>：处理传入和传出的请求</li><li><a href="https://openrouter.ai/docs/requests#images-_-multimodal-re">Requests | OpenRouter</a>：处理传入和传出的请求
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1309196960049266709)** (1 messages): 

> - `Claude 3.5`
> - `自定义提供商密钥请求` 


- **用户请求 Claude 3.5 Sonnet 的自定义提供商密钥 (custom provider key)**：一位成员请求为 **Claude 3.5 Sonnet** 提供 **custom provider key**，并对 **Claude** 主应用的使用额度耗尽表示沮丧。
   - 他们希望这一请求能为目前的限制提供一个可行的替代方案。
- **对 Claude 应用使用限制的担忧**：讨论强调了关于 **Claude** 主应用 **usage limits**（使用限制）的问题，这导致了用户的挫败感。
   - 成员们正在寻求更有效地管理使用量并改善体验的解决方案。


  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1308901712312144033)** (164 条消息🔥🔥): 

> - `Flux 性能`
> - `SDXL 使用`
> - `图像生成问题`
> - `ControlNet 功能`
> - `AI 模型安全担忧` 


- **Flux 的资源密集型需求**：成员们讨论了有效使用 **Flux** 的**资源需求**，指出它需要大量的 **VRAM**，且生成图像的速度可能较慢。
   - 一位成员强调，使用 **Loras** 可以增强 **Flux** 在 NSFW 内容上的输出，尽管它并非专门为此训练。
- **最大化 SDXL 性能**：对于 **SDXL**，在配置中使用 `--xformers` 和 `--no-half-vae` 等**最佳实践**可以提高在 **12GB VRAM** 系统上的性能。
   - 成员们指出，**Pony**（**SDXL** 的衍生模型）需要特殊的 token，并且在与 **XL Loras** 的兼容性方面存在局限。
- **使用 SDXL Lightning 进行图像提示**：一位用户询问如何通过 Python 在 **SDXL Lightning** 中使用**图像提示 (image prompts)**，特别是将照片插入特定环境中。
   - 对话表明，将图像提示与不同背景结合是增强生成能力的一个热门话题。
- **解决生成延迟问题**：在使用各种模型时，随机出现的**超长生成时间**引发了对潜在根本原因的讨论。
   - 成员们推测，内存管理问题（如将资源加载到 **VRAM** 中）可能是导致这些减速的原因。
- **AI 模型利用与安全**：由于收到索要个人信息（如钱包地址）的可疑请求，成员们认为社区中可能存在**诈骗者**。
   - 鼓励用户举报此类事件，以维护群组内的**安全**环境。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8">Comfy-Org/stable-diffusion-3.5-fp8 · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/bfl_ml/status/1859616264324284619?t=DftoDEhtAigmD4sQvsMl2w&s=19">来自 Black Forest Labs (@bfl_ml) 的推文</a>: 今天，我们很高兴发布 FLUX.1 Tools，这是一套旨在为我们的基础文本生成图像模型 FLUX.1 增加控制力和引导性的模型，能够修改和重建真实的...</li><li><a href="https://huggingface.co/calcuis/sd3.5-medium-gguf/tree/main">calcuis/sd3.5-medium-gguf at main</a>: 未找到描述
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1309026411364745246)** (5 条消息): 

> - `Mamba SSM 层`
> - `数据传输时间`
> - `LLM 自噬过程`
> - `基础模型评估任务`
> - `惠灵顿聚会` 


- **Mamba 的 SSM 层解析**：一位成员询问实值版 **Mamba** 中的 SSM 层是否充当指数移动平均线 (EMA)，这意味着 *`dA(x) = exp(-f(x))`*，其中 `f` 是输入 **x** 的函数。
   - 有建议认为该操作本质上计算了一个逐元素应用的效率 EMA 更新：*`h_t = dA(x) * h_{t - 1} + (1 - dA(x)) * x`*。
- **训练数据传输 vs 处理时间**：讨论了单批次 (batch) 的训练时间短于 **PCI 总线**数据传输时间是否常见。
   - 这引发了对训练工作流中处理大规模数据集效率的担忧。
- **LLM 自噬过程研究**：一位博士生介绍了他们关于 **LLMs** **自噬过程 (autophagy process)** 的研究，并提供了预印本论文链接：[arXiv preprint](https://arxiv.org/abs/2410.12341)。
   - 他们提到在即将发表的论文中，将使用该库来评估模型在崩溃背景下的表现。
- **寻求基础模型的评估任务**：同一位博士生正在寻求除了 **HellaSwag** 等标准任务之外，针对基础模型 (foundational models) 的有趣评估任务建议。
- **新西兰惠灵顿聚会邀请**：该博士生宣布他们即将在新西兰**惠灵顿**开始研究期，并表示有兴趣与当地成员见面。
   - 他们邀请任何居住在惠灵顿且有兴趣聚会的人与其联系。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1308900335749693440)** (48 条消息🔥): 

> - `FlexAttentions`
> - `Position Encoding Techniques`
> - `Forgetting Transformer`
> - `Sparse Upcycling vs Continued Pretraining`
> - `Scale of LLM Training` 


- **FlexAttentions 开发正在进行中**：一名成员表示乐观，认为通过一个月的专注工作，可以相对容易地完成新的 **FlexAttentions** 模型的开发。
   - 这表明人们对改进 Attention 机制以增强模型效率的兴趣日益浓厚。
- **探索新的位置编码方法**：讨论围绕位置编码方法展开，特别是关于 **Contextual Position Encoding (CoPE)** 的提议，该方法根据 Token 上下文而非固定计数进行调整，从而产生更具表现力的模型。
   - 成员们强调了在处理传统方法难以应对的 **Flip-Flop** 等选择性任务方面的潜在改进。
- **Forgetting Transformer 表现优于标准模型**：**Forgetting Transformer** 作为一种引入了 **forget gate** 的变体被介绍，旨在长上下文任务中获得更好的性能，显示出优于标准架构的改进。
   - 该模型不需要位置嵌入（Position Embeddings），并在更长的训练上下文中保持有效的性能。
- **Sparse Upcycling 方法论的权衡**：最近的一篇 **Databricks** 论文评估了 **Sparse Upcycling** 与 **Continued Pretraining** 在模型增强方面的权衡，发现 **Sparse Upcycling** 带来了更好的质量，但代价是增加了推理时间。
   - 这导致推理效率明显**下降了 40%**，强调了在模型性能与实际部署考虑之间取得平衡的挑战。
- **对 LLM 训练基础设施的担忧**：对话中包含了对当前 **network fabric** 和 **bandwidth** 改进滞后于处理能力的批评，特别是在 **TPU** 训练设置的背景下。
   - 讨论考虑了像 **TPU** 的 **topology** 等架构选择如何提供优于传统 **GPU** 设置的优势。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.18719">Contextual Position Encoding: Learning to Count What&#39;s Important</a>：Attention 机制是 Large Language Models (LLMs) 的关键组件，允许序列中的 Token 相互交互，但是顺序无关的。引入 Position Encoding (P...</li><li><a href="https://arxiv.org/abs/2411.13055">Hardware Scaling Trends and Diminishing Returns in Large-Scale Distributed Training</a>：近年来神经网络模型能力的剧增是由 Scaling 模型规模、训练数据和相应的计算资源驱动的。为了开发极其庞大的...</li><li><a href="https://arxiv.org/abs/2411.08968?">Sparse Upcycling: Inference Inefficient Finetuning</a>：小型、经过高度训练的开源 Large Language Models 因其推理效率而被广泛使用，但进一步提高其质量仍是一个挑战。Sparse Upcycling 是一种很有前途的方法...</li><li><a href="https://seaborn.pydata.org/generated/seaborn.heatmap.html">seaborn.heatmap &#8212; seaborn 0.13.2 documentation</a>：未找到描述</li><li><a href="https://x.com/YouJiacheng/status/1859353724713566290">YouJiacheng (@YouJiacheng) 的推文</a>：@hi_tysam 这是一个滑动窗口，如果层数 >1，信息仍然可以传播。</li><li><a href="https://openreview.net/forum?id=q2Lnyegkr8">Forgetting Transformer: Softmax Attention with a Forget Gate</a>：现代循环序列模型的一个基本组件是 forget gate。虽然 Transformer 没有显式的循环形式，但我们展示了 forget gate 可以自然地融入...</li><li><a href="https://cloud.google.com/blog/products/compute/the-worlds-largest-distributed-llm-training-job-on-tpu-v5e">the world’s largest distributed LLM training job on TPU v5e | Google Cloud Blog</a>：我们使用 Multislice Training 在由 50,944 个 Cloud TPU v5e 芯片组成的计算集群上运行了全球最大的 LLM 分布式训练任务。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1308914897366552656)** (7 条消息): 

> - `Scaling Laws`
> - `Evaluation Predictions` (评估预测)
> - `Marius Hobbhahn's Contributions` (Marius Hobbhahn 的贡献)
> - `Meta and OpenAI's Methods` (Meta 和 OpenAI 的方法)
> - `Cost of Scaling Law Training` (Scaling Law 训练成本)


- **Scaling Laws 详解**：最近的一篇 [论文](https://arxiv.org/abs/2405.10938) 讨论了语言模型性能如何随规模变化，并提供了一种观测方法，利用约 100 个公开可用的模型来开发 Scaling Laws，而无需直接训练。
   - 该方法可以突出训练效率的差异，并提出了一种通用的 Scaling Law，其中性能取决于一个低维的能力空间。
- **Marius Hobbhahn 领导评估科学**：Apollo 的 Marius Hobbhahn 因在 Scaling Laws 领域公开倡导评估方法论科学而受到关注。
   - 他的贡献旨在将预测和基准测试建立在可观测的指标上，而不是依赖大规模建模。
- **训练前预测性能**：GPT-4 论文在训练之前就预测了其在流行的 HumanEval 编程基准测试中的得分，误差在 ~1% 以内，展示了有效的评估方法论。
   - 同样，Meta 团队在训练前预测了 Llama-3.1 模型在 Abstract Reasoning Corpus 上的表现，采用了创新的统计方法。
- **成功预测的方法**：HumanEval 的预测方法涉及绘制 Mean Log Pass 率与计算规模的关系图，而 Abstract Reasoning Corpus 则利用了涉及 negative log likelihood 的两步转换。
   - 这些方法证明了基于从现有模型推导出的 Scaling Laws 来准确投影模型能力的能力。
- **Scaling Law 模型的成本**：训练 Scaling Law 模型的成本很高，但显著低于训练完整的最终模型，据报道 Meta 的支出仅为目标模型预算的 0.1% 到 1%。
   - 例如，为了预测一个价值 10 亿美元的模型的能力，如果按照这个预算比例，训练 Scaling Law 模型的成本可能会超过 100 万美元。



**提到的链接**：<a href="https://arxiv.org/abs/2405.10938">Observational Scaling Laws and the Predictability of Language Model Performance</a>：理解语言模型性能如何随规模变化对于基准测试和算法开发至关重要。Scaling Laws 是建立这种理解的一种方法，但其要求...

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1308927065319538769)** (39 messages🔥): 

> - `lm-eval and pruned models`
> - `Using Groq API`
> - `Logits and loglikelihood in QA`
> - `Custom metrics in lm-harness` 


- **lm-eval 对 pruned models 的支持受到质疑**：一位用户询问当前版本的 lm-eval 是否支持 pruned models 的 zero-shot 基准测试，并指出旧版本库存在问题。
   - *他们正在使用 WANDA*，并报告 zero-shot 结果不可靠，引发了关于查阅现有局限性文档的讨论。
- **成功对接 Groq API**：一位用户在尝试连接 Groq API 时遇到了无法识别 API key 参数的问题，并参考其文档进行故障排除。
   - 另一位成员建议将 API key 设置在 `OPENAI_API_KEY` 环境变量中，该问题随后得到解决。
- **在 TruthfulQA 中提取 loglikelihood**：一位用户询问如何获取 TruthfulQA 数据集中答案的 loglikelihood 值，而非标准的准确率指标。
   - 讨论围绕标准 QA 设置以及在调整前后需要更好的 LLM 性能指标展开。
- **lm-harness 中针对 logits 的自定义指标**：一位用户询问是否有方法在 lm-harness 中保存 logits，以分析 LLM 对正确答案的倾向性。
   - 一位成员建议创建一个自定义指标，以便根据评估目的按需操作 logits。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/">GitHub · Build and ship software on a single, collaborative platform</a>: 加入全球应用最广泛、AI 驱动的开发者平台，数百万开发者、企业和最大的开源社区在此构建推动人类进步的软件。</li><li><a href="https://github.com/locuslab/wanda?tab=readme-ov-file#zero-shot-evaluation)">GitHub - locuslab/wanda: A simple and effective LLM pruning approach.</a>: 一种简单且有效的 LLM pruning 方法。通过在 GitHub 上创建账号为 locuslab/wanda 的开发做出贡献。</li><li><a href="https://console.groq.com/docs/overview">GroqCloud</a>: 体验全球最快的推理速度</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/867413f8677f00f6a817262727cbb041bf36192a/lm_eval/models/anthropic_llms.py#L324)">lm-evaluation-harness/lm_eval/models/anthropic_llms.py at 867413f8677f00f6a817262727cbb041bf36192a · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1309167267266691092)** (1 messages): 

> - `Multimodal benchmarks`
> - `LLaVA performance`
> - `Text recognition in images` 


- **寻求多样化的多模态基准测试**：一位成员询问是否有超越“描述照片”等基础任务的**有趣的多模态模型基准测试**。
   - 他们特别希望探索能够评估模型识别并报告**图像中文本**能力的基准测试。
- **LLaVA 表现不如更小的模型**：该成员注意到 **LLaVA** 及其衍生模型在某些任务上的表现甚至不如体积更小的 **smaller models**。
   - 这一观察激发了他们进行进一步测试以了解性能差异的兴趣。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1308908575149658235)** (83 messages🔥🔥): 

> - `Pro Channel Access` (Pro 频道访问权限)
> - `Image Creation on iOS` (iOS 上的图像生成)
> - `Perplexity Pro Features` (Perplexity Pro 功能)
> - `Subscription Issues` (订阅问题)
> - `Discord Support` (Discord 支持)


- **关于 Pro 频道访问权限的担忧**：用户对购买 Pro 版本后访问 Pro 频道表示担忧，其中一位用户表示，尽管点击了提供的链接，最初仍无法访问。
   - 其他人在重新加入 Discord 或获得其他用户帮助后确认已获得访问权限。
- **iOS 上的图像生成**：一位用户询问了 Perplexity iOS 应用内的图像生成功能，另一位用户澄清该功能目前仅在 iPad 上可用。
   - 这引发了关于不同设备间功能限制的讨论。
- **Perplexity Pro 功能亮点**：成员们讨论了 Perplexity Pro 的各种功能，强调了 Pro 用户可使用的更先进模型，以及它与 ChatGPT 的区别。
   - 讨论内容包括关于搜索和工具集成的见解，这些集成增强了用户体验。
- **账户和订阅问题**：几位用户报告了订阅方面的挑战，从兑换码失败到账户关联困难等。
   - 用户被引导至支持邮箱以解决特定问题，随后讨论了如何有效管理多个账户。
- **Discord 支持需求**：关于如何在 Discord 上获得支持的问题不断出现，成员们分享了帮助订阅升级和解决角色访问问题的链接。
   - 社区提供了帮助，一些成员在获得指导后确认其账户问题已成功解决。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/bocchi-the-rock-bocchi-the-rock-awkward-anime-bocchi-the-rock-bocchi-look-around-anime-awkward-gif-27050898">Bocchi The Rock Bocchi The Rock Awkward GIF - Bocchi The Rock Bocchi The Rock Awkward Anime Bocchi The Rock - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://x.com/apostraphi/status/1859627827160629325?s=46">Phi Hoang (@apostraphi) 的推文</a>：如果你问我，这时间花得很值</li><li><a href="https://www.ispreview.co.uk/index.php/2024/11/virgin-media-o2-uk-offer-free-access-to-ai-search-engine-perplexity-pro.html">Virgin Media O2 UK 提供 AI 搜索引擎 Perplexity Pro 的免费访问权限</a>：Virgin Media 和 O2 的各种宽带、移动、电话和电视套餐客户可能会想知道，他们的“Priority”应用正在奖励现有订阅用户。
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1308973694734831636)** (7 messages): 

> - `Pokémon Data AI Model` (宝可梦数据 AI 模型)
> - `Baltic Sea Cable Sabotage` (波罗的海电缆破坏)
> - `Chicken or Egg Paradox Resolution` (鸡生蛋还是蛋生鸡悖论的解决)
> - `NVIDIA's Omniverse Blueprint` (NVIDIA 的 Omniverse Blueprint)
> - `One-Person Startup Era` (一人初创公司时代)


- **宝可梦数据引发新 AI 模型**：一段 YouTube 视频讨论了如何利用 **Pokémon 数据** 来创建 **AI 模型**，提供了关于游戏技术进步的见解。
   - *这可能会改变 AI 应用中利用数据的方式*。
- **调查波罗的海破坏事件**：一个链接引发了对波罗的海**电缆破坏**的关注，突显了潜在的地缘政治紧张局势。
   - *需要进一步讨论其对数字基础设施和安全的影响*。
- **鸡生蛋还是蛋生鸡悖论可能已解决**：最近的一项讨论指向了**鸡生蛋还是蛋生鸡悖论**的可能解决方案，引发了对科学进化的思考。
   - *这可能会引发关于发展和存在的新哲学探究*。
- **NVIDIA 为 CAD/CAE 带来的变革性技术**：一位成员分享了关于 **NVIDIA Omniverse Blueprint** 的见解，展示了其在设计和模拟中对 CAD 和 CAE 的变革潜力。
   - *许多人对其如何将先进技术集成到传统工作流中感到兴奋*。
- **一人初创公司时代兴起**：Sam Altman 挑衅性地表示：**“未来的初创公司将仅由 1 个人和 10,000 个 GPUs 运行”**，暗示了未来的创业趋势。
   - *这一观点反映了科技初创公司和资源利用不断变化的格局*。



**提到的链接**：<a href="https://www.youtube.com/embed/hQhP7ipvgx0">YouTube</a>：未找到描述

  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1309062556157214750)** (7 messages): 

> - `API Rate Limits`
> - `在 Perplexity 中使用自己的 API Key`
> - `前端应用中的 Session Management` 


- **速率限制困惑 (Rate Limit Confusion)**：几位成员正面临 API 的 **rate limit 问题**，询问限制是设定为**每个账户每分钟 50 次请求**还是按 Key 设定。
   - 一位用户已联系 Perplexity 请求提高限制，但尚未收到反馈，并因客户问题表示情况紧急。
- **关于自带 API Key 的咨询**：一位成员询问是否允许**自带 API Key** 来构建使用 Perplexity 的替代平台，并概述了如何安全地管理数据。
   - 该方法涉及将用户提供的 Key 加密并存储在 cookies 中，这引发了关于是否符合 OpenAI 标准的疑问。
- **简化技术概念**：为了响应简化请求，一位用户通过将 **session management** 比作存储 session ID 的 cookies，解释了 Web 应用中的会话管理。
   - 对话强调了用户的身份验证如何依赖于检查有效会话，而无需直接存储敏感数据。



**Link mentioned**: <a href="https://docs.perplexity.ai/guides/rate-limits">no title found</a>: no description found

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1308905759085432873)** (59 messages🔥🔥): 

> - `Truffles 硬件设备`
> - `Vercel 收购 Grep`
> - `Tülu 3 模型发布`
> - `Black Forest Labs 的 Flux Tools`
> - `Gemini API 模型更新` 


- **Truffles 设备引起关注**：成员们回想起了 **Truffles** 设备，它被描述为一个“白色云状半透明物体”，允许用户自托管 LLM [Truffles](https://x.com/itsalltruffles)。一位成员幽默地称其为“发光的乳房植入物”。
- **Vercel 收购 Grep 用于代码搜索**：Vercel 宣布收购 [Grep](https://grep.app/)，以增强开发者在超过 500,000 个公共仓库中搜索代码的工具。创始人 Dan Fox 将加入 Vercel 的 AI 团队继续开发此功能。
- **Tülu 3 表现优于 Llama 3**：据报道，历时两年开发的 [Tülu 3](https://allenai.org/papers/tulu-3-report.pdf) 模型在特定任务上优于 **Llama 3.1 Instruct**，并拥有全新的 SFT 数据和优化技术。项目负责人对其在 RLHF 领域的成就感到兴奋。
- **Flux Tools 的新功能**：Black Forest Labs 发布了 **Flux Tools**，其中包括用于图像处理的 inpainting 和 outpainting 功能。该套件旨在为其 text-to-image 模型增加可控性，鼓励用户尝试在 [Replicate](https://replicate.com/black-forest-labs) 上运行。
- **Google Gemini 模型更新**：**Gemini** 发布了新的实验性模型，重点改进了编程能力。用户可参考 [Gemini API 文档](https://ai.google.dev/gemini-api/docs/models/experimental-models)了解详情。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://vercel.com/blog/vercel-acquires-grep">Vercel acquires Grep to accelerate code search - Vercel</a>：宣布收购 Grep，以进一步推进我们帮助开发者更快工作和交付的使命。</li><li><a href="https://x.com/natolambert/status/1859643351441535345">来自 Nathan Lambert (@natolambert) 的推文</a>：在过去的两年里，我搜寻了所有关于 RLHF（特别是）和广泛的 post training（后期训练）的可用资源。今天，在一个非常优秀的团队的帮助下，我们为您带来了这些劳动的成果……</li><li><a href="https://prannaya.notion.site/Existing-AI-Code-Tools-14105a6b76a480f8bf3af4dae8ee1084?pvs=4">Notion – 笔记、任务、维基和数据库的一站式工作空间。</a>：一款将日常工作应用融合在一起的新工具。它是为您和您的团队打造的一站式工作空间。</li><li><a href="https://x.com/ExaAILabs/status/1859306370010579010">来自 Exa (@ExaAILabs) 的推文</a>：我们正以整整一周的发布活动来庆祝感恩节 🦃 今天 - LinkedIn 上的语义搜索。选择 “LinkedIn profile” 类别，智能搜索数亿个……</li><li><a href="https://x.com/itsalltruffles">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://blog.dottxt.co/say-what-you-mean.html">Say What You Mean: A Response to 'Let Me Speak Freely'</a>：未找到描述</li><li><a href="https://x.com/replicate/status/1859616730915721249?s=46">来自 Replicate (@replicate) 的推文</a>：Black Forest Labs 刚刚发布了 Flux Tools，适用于专业和开源开发模型：- Fill: Inpainting 和 outpainting - Redux: 用于图像变体 - Canny 和 depth controlnets。它们都非常棒……</li><li><a href="https://x.com/bfl_ml/status/1859616264324284619?s=46">来自 Black Forest Labs (@bfl_ml) 的推文</a>：今天，我们很高兴发布 FLUX.1 Tools，这是一套旨在为我们的基础文本生成图像模型 FLUX.1 增加控制力和可引导性的模型套件，能够修改和重新创建真实的以及……</li><li><a href="https://ai.google.dev/gemini-api/docs/models/experimental-models">未找到标题</a>：未找到描述</li><li><a href="https://x.com/markokraemer/status/1859526870867263906">来自 markokraemer (@markokraemer) 的推文</a>：v0 vs bolt vs loveable vs softgen。3 个提示词：1. “制作一个关于蛋白奶昔的高级落地页” 2. “让它更出众” 3. “增加更多板块”。顺便说一下，我当时正在喝蛋白奶昔……</li><li><a href="https://youtu.be/LPZh9BOjkQs?si=Jyqqr-NGyt3dXwlz">大语言模型（LLM）简要解释</a>：在此深入了解：https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi 技术细节讲座：https://youtu.be/KJtZARuO3JY 为……而制作</li><li><a href="https://x.com/karpathy/status/1859305141385691508?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Andrej Karpathy (@karpathy) 的推文</a>：还记得 GPT-2 (124M) 训练运行的 llm.c 复现吗？在 8xH100 上耗时 45 分钟。从那时起，@kellerjordan0（以及现在的许多其他人）在新的 modded-nanogpt 仓库中对其进行了广泛的迭代……</li><li><a href="https://github.com/KellerJordan/modded-nanogpt">GitHub - KellerJordan/modded-nanogpt: 5 分钟内完成 NanoGPT (124M)</a>：5 分钟内完成 NanoGPT (124M)。通过在 GitHub 上创建账户，为 KellerJordan/modded-nanogpt 的开发做出贡献。
</li>
</ul>

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1308963337853472818)** (29 条消息🔥): 

> - `急需 AI 专家`
> - `DeepSeek R1-Lite 规格`
> - `Llama-Mesh 论文推荐`
> - `每日 LLM 发布讨论`
> - `AI 中的用户交互记忆` 


- **急需 AI 专家**: 一名成员紧急寻求 AI 专家，引发了关于所需具体协助类型的各种回应，从部署问题到架构支持。
   - 成员们幽默地建议联系顶级专家，而另一位则指出紧急情况仍然模糊不清，强调了明确需求的必要性。
- **DeepSeek R1-Lite 规格**: 传闻 **DeepSeek R1-Lite** 是一个拥有 **2.4B 激活参数** 的 **16B MOE** 模型，将 MATH 分数从 **17.1 大幅提升至 91.6**。
   - 该传闻引用了微信公告，但遭到了质疑，一名成员对潜在的性能提升表示难以置信。
- **Llama-Mesh 论文推荐**: 一名成员向小组推荐阅读 **llama-mesh 论文**，称其“相当不错”。
   - 在关于 AI 讨论的更广泛对话中，这一呼吁引起了关注。
- **每日 LLM 发布讨论**: 成员们讨论了“每日 LLM 发布”的概念，提到了语言模型不断演进的格局，并带有一丝疲惫感。
   - 一名成员幽默地评论说 **Goodhart arena 已经过时了**，表明了对该领域频繁变化的某种情绪。
- **AI 中的用户交互记忆**: 一名成员询问 AI 系统是否会记住 Twitter 等平台上的先前交互，特别是关于与相同账号互动的情况。
   - 这一询问反映了对 AI 用户记忆能力的广泛好奇，这与各种用户体验和期望相关。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/nrehiew_/status/1859265550767067518">来自 wh (@nrehiew_) 的推文</a>: 传闻 DeepSeek R1-Lite 是一个具有 2.4B 激活参数的 16B MOE，如果属实，他们的 MATH 分数从 17.1 提升到了 91.6。引用 Phil (@phill__1) @nrehiew_ 来自他们的微信公告：</li><li><a href="https://github.com/Lesterpaintstheworld/terminal-velocity">GitHub - Lesterpaintstheworld/terminal-velocity: 由 10 个团队、每个团队 10 个 AI Agent 自主创作的小说</a>: 由 10 个团队、每个团队 10 个 AI Agent 自主创作的小说 - Lesterpaintstheworld/terminal-velocity
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1309052580902731807)** (6 条消息): 

> - `LLM 性能提升`
> - `KV Cache 限制`
> - `Multi-Agent 框架`
> - `Prefix Caching`
> - `Prompt Caching` 


- **Multi-Agent 框架与隐藏信息**: 一名成员担心在“AI 企业家”和“AI 软件工程师”等 Multi-Agent 框架中使用反 Token 化（de-tokenized）输出可能会因为丢弃了 KV Cache 而导致**隐藏信息丢失**。
   - 他们认为这种损失可能解释了在此类框架中观察到的**输出多样性有限**的问题。
- **Prefix Caching vs. KV Cache**: 作为回应，另一名成员询问 Prefix Caching 在推理过程中是否起到与 KV Cache 类似的作用。
   - 讨论显示，Prefix Caching 以前在 API 中不可用，这导致了早期 Agent 框架所经历的局限性。
- **关于缓存的误解**: 对话转向了关于 Prompt Caching 与其他缓存技术等效性的误解。
   - 对这些疏忽的承认表明社区内存在持续的学习曲线。


  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1308969139716620289)** (4 messages): 

> - `Soft Prompts vs Fine Tuning`
> - `CoPilot Arena Results`
> - `LoRA Trade-offs` 


- **Soft Prompts 难以普及**：讨论强调，**Soft Prompts** 经常被 **Fine Tuning** 和 **LoRA** 等技术掩盖，后者通常被认为在开源用例中更有效。尽管具有一些独特优势，但 Soft Prompts 的泛化能力有限，在当前的实践中并未得到广泛利用。
   - 参与者指出，使用 Soft Prompts 可能会涉及权衡，特别是在性能和优化方面。
- **Soft Prompts 的两个潜在用途**：一位成员建议 Soft Prompts 可以服务于两个主要目的：**System Prompt 压缩**和**增强 LoRA/全量 SFT** 应用。他们提到这种策略可以在不严重依赖推理系统的情况下优化模型参数。
   - 这些用途的影响包括潜在的过拟合风险，表明需要谨慎实施。
- **CoPilot Arena 初步结果发布**：**CoPilot Arena** 的首批结果已在 [LMarena 的博客](https://blog.lmarena.ai/blog/2024/copilot-arena/#initial-leaderboard-and-results)上展示，显示出参与者之间的差距出人意料地小。然而，有人指出该分析仅考虑了较旧版本的 **Sonnet**。
   - 这引发了人们对于在竞争环境中使用过时模型的影响，以及这可能如何影响参与者对比的好奇。


  

---



### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1309172096328531980)** (20 messages🔥): 

> - `Debugging Triton Interpreter`
> - `Block Size Discussion`
> - `Triton GEMM and Bank Conflicts`
> - `Boolean Mask Summation Bug`
> - `Swizzling Techniques for Performance` 


- **调试 Triton Interpreter 以确保 Kernel 准确性**：一位用户就调试 **Kernel 准确性**问题寻求建议，该问题在 **Triton Interpreter** 中无法复现，特别是考虑到 Matmuls 的 TF32 已关闭。
   - 建议包括手动将 Tensor 转换为 `tl.float32`，并检查数据与特定 **Block Size** 的兼容性。
- **观察到奇怪的 Block Size 行为**：有报告称 **Triton** 中的准确性问题取决于 **Block Size**，在尺寸 <= 64 时运行良好，但在尺寸 >= 128 时会出现问题。
   - 讨论涉及如何通过潜在的配置剪枝来确保输入 Shape 正确且有效，以避免加载 Shape 冲突。
- **Triton GEMM 的 Bank Conflict 解决方案**：一位用户对 ROCm 上的 **Triton GEMM** 没有冲突感到惊讶，并询问了关于应用 **Swizzling** 以避免 **Bank Conflicts** 的问题。
   - 虽然分享了关于 Block 级 Swizzling 的参考资料，但重点仍在于进一步探索针对 Bank Conflict 解决的特定方法。
- **在 Boolean Mask 求和中发现 Bug**：经过大量的打印和调试，一位用户发现了 **Triton** 中一个与求和转换为 **Int8** Tensor 的 **Bool Masks** 相关的 Bug。
   - 将求和切换为 Max Reduction 解决了该问题，并有人建议为该仓库起草一个最小可复现示例（Minimal Example）。
- **有趣的告别和持续的提问**：随着讨论结束，一位成员表示该睡觉了，而另一位成员则对 AMD 支持表示持续关注。
   - 对话以承诺跟进所提出的技术难题而结束。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://triton-lang.org/main/python-api/generated/triton.language.swizzle2d.html">triton.language.swizzle2d &mdash; Triton  documentation</a>: 未找到描述</li><li><a href="https://github.com/mobiusml/gemlite/blob/master/gemlite/triton_kernels/utils.py#L6-L14">gemlite/gemlite/triton_kernels/utils.py at master · mobiusml/gemlite</a>: CUDA / Triton 中简单快速的低比特 Matmul Kernel - mobiusml/gemlite
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1309179398343102544)** (2 条消息): 

> - `cuBLAS operations`
> - `Matrix Multiplication with cuBLAS`
> - `Row-major vs Column-major order` 


- **cuBLAS 的列主序（Column-Major Order）困境**：一位用户强调了使用以 **列主序** 运行的 `cublasSgemm` 所面临的挑战，并询问为了更好的清晰度，是倾向于使用 `CUBLAS_OP_N` 还是 `CUBLAS_OP_T` 进行调用。
   - 对转置操作的 *显式清晰度* 在处理以 **行主序** 定义的矩阵时可能会导致 *混淆*，因为输出将始终是列主序。
- **Stack Overflow 关于 cuBLAS 的见解**：一位成员分享了一个相关的 [Stack Overflow 帖子](https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication)，该帖子阐明了如何对以 **行主序** 存储的非方阵使用 `cublasSgemm`。
   - 该帖子讨论了在乘法非方阵时使用 **CUBLAS_OP_T** 的限制，并强调了与非转置结果之间潜在的 *冲突*。
- **矩阵声明的复杂性**：有提到由于矩阵是在另一个程序中设置的，因此使用 `IDX2C` 宏将其修改为以 **列主序** 声明是不可行的。
   - 这表明了在调整现有代码库以适应 cuBLAS 库限制时所面临的普遍问题。
- **cuBLAS 的程序限制**：用户现有的代码无法使用转置参数对非方阵进行乘法运算，限制了其在 **矩阵操作** 中的通用性。
   - 这引发了关于 cuBLAS 在处理多种矩阵大小和方向时的灵活性和可用性的疑问。



**提到的链接**：<a href="https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication">cublasSgemm row-major multiplication</a>：我正尝试使用 cublasSgemm 来对两个以行主序存储的非方阵进行乘法运算。我知道这个函数有一个参数可以让你指定是否要进行转置...

  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1308994672726900789)** (1 条消息): 

> - `Llama-2-70B model`
> - `Multi-GPU support` 


- **询问 Llama-2-70B 的多卡支持**：一位成员询问 `llama/generate.py` 是否支持 **Llama-2-70B** 模型以利用多张显卡，特别提到了 **2 张 A100**。
   - 讨论集中在该脚本有效处理 GPU 资源的能力上。
- **探索 GPU 利用率策略**：提出的另一点是关于在多卡上运行像 Llama-2-70B 这样的大型模型时，如何优化 **GPU 利用率**。
   - 成员们讨论了最大化吞吐量并减少瓶颈的潜在策略。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1309078115754184774)** (11 条消息🔥): 

> - `HIP Kernel Rules`
> - `Compilation Time for Examples`
> - `FP16 GEMM on MI250 GPU`
> - `Debugging Kernel`
> - `Triton GEMM on ROCm` 


- **HIP Kernel 规则仍然令人困惑**：一位用户表示在理解和复现 **纯 HIP kernel** 中的规则时存在困难。
   - 他们指出即使经过多次尝试，挑战依然存在。
- **示例的编译时间令人沮丧**：据报告，在用户的机器上使用 `make` 命令编译一个简单的示例需要 **1-2 小时**。
   - 尽管尝试在编译中调整 `-j` 标志，但并未显著提高性能。
- **对 FP16 GEMM 输入形状的困惑**：一位用户分析了 MI250 GPU 上 **FP16 GEMM (v3)** 的变换描述，注意到变换的输入形状不匹配。
   - 他们要求澄清 Shared Memory 和输入形状背后的基本原理。
- **调试减慢了编译速度**：在 Kernel 中插入打印函数会增加编译时间，因为 CK 中有许多静态操作，这些操作会展开多次。
   - 有人建议减小 Tile Size 以提高调试性能。
- **发现 ROCm 上的 Triton GEMM 无冲突**：一位用户惊讶地发现 ROCm 上的 **Triton GEMM** 也表现出 **无冲突（conflict-free）** 特性。
   - 这一见解可能会引发关于 ROCm 优化策略的讨论。


  

---

### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1308917390519832667)** (1 条消息): 

> - `AGX machine code`
> - `Freedesktop tools` 


- **Freedesktop 开发者反汇编 AGX machine code**：Freedesktop 开发者在他们的 [GitHub repository](https://github.com/dougallj/applegpu) 上维护着用于反汇编 **AGX machine code** 的工具。
   - *反汇编 machine code 有助于更好地理解和优化软件性能。*
- **工具优势与用户讨论**：几位成员讨论了使用 **Freedesktop 的反汇编工具** 相比其他分析 machine code 方法的优势。他们强调了这些工具如何简化调试过程并缩短开发时间。


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/)** (1 条消息): 

0x000ff4: 关于 kto 的一点更新，我目前正在进行测试工作
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/)** (1 条消息): 

pradeep1148: https://www.youtube.com/watch?v=XP33Vgn75lM
  

---

### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1309214054442598562)** (1 条消息): 

> - `Post-Training Techniques`
> - `Human Preferences in RL`
> - `Continual Learning`
> - `Constitutional AI`
> - `Recursive Summarization` 


- **Tulu 3 综述发布**：推荐了一篇名为 [Tulu 3](https://allenai.org/papers/tulu-3-report.pdf) 的新综述论文，用于理解 Post-training 方法。
   - 这篇论文为任何对该领域进展感兴趣的人提供了全面的概述。
- **利用人类偏好进行 RL**：论文 [Deep RL from human preferences](https://arxiv.org/abs/1706.03741) 探索了在 RL 任务中通过人类偏好定义复杂目标，并在游戏和机器人运动中展示了有效的结果。
   - 它强调只有 **1%** 的 Agent 交互需要人类反馈，显著降低了监督成本。
- **Continual Learning 见解**：文章 [Comprehensive survey of continual learning](https://arxiv.org/abs/2302.00487) 深入探讨了灾难性遗忘等挑战以及克服这些挑战的方法。
   - 它在 Continual Learning 的基础理论与实际应用之间架起了一座彻底的桥梁。
- **探索 Constitutional AI**：在论文 [Constitutional AI](https://arxiv.org/abs/2212.08073) 中，作者研究了基于稳健的指导原则构建 AI 系统。
   - 该研究包含了该领域知名研究者的贡献，确保了对该主题的多样化视角。
- **利用人类反馈推进摘要技术**：题为 [Recursively summarizing books with human feedback](https://arxiv.org/abs/2109.10862) 的研究解决了使用人类反馈和递归分解来总结整部小说的挑战。
   - 该模型允许人类进行快速监督和评估，从而实现高效且合理的摘要生成。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/1706.03741">Deep reinforcement learning from human preferences</a>: 为了让复杂的强化学习 (RL) 系统与现实环境进行有用的交互，我们需要向这些系统传达复杂的目标。在这项工作中，我们探索了定义在...</li><li><a href="https://arxiv.org/abs/2009.01325">Learning to summarize from human feedback</a>: 随着语言模型变得越来越强大，训练和评估日益受到特定任务所使用的数据和指标的瓶颈限制。例如，摘要模型通常训练于...</li><li><a href="https://arxiv.org/abs/2203.02155">Training language models to follow instructions with human feedback</a>: 扩大语言模型规模并不能本质上提高其遵循用户意图的能力。例如，大型语言模型可能会生成不真实、有毒或根本不...</li><li><a href="https://arxiv.org/abs/2307.15217">Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback</a>: 人类反馈强化学习 (RLHF) 是一种训练 AI 系统以对齐人类目标的技术。RLHF 已成为微调最先进的大型语言...</li><li><a href="https://arxiv.org/abs/2302.00487">A Comprehensive Survey of Continual Learning: Theory, Method and Application</a>: 为了应对现实世界的动态变化，智能系统需要在其整个生命周期中增量地获取、更新、积累和利用知识。这种能力被称为 Continual Learning，它提...</li><li><a href="https://arxiv.org/abs/2212.08073">Constitutional AI: Harmlessness from AI Feedback</a>: 随着 AI 系统变得越来越强大，我们希望寻求它们的帮助来监督其他 AI。我们实验了通过自我改进来训练无害 AI 助手的方法，无需任何人工...</li><li><a href="https://arxiv.org/abs/2305.18290">Direct Preference Optimization: Your Language Model is Secretly a Reward Model</a>: 虽然大规模无监督语言模型 (LMs) 学习到了广泛的世界知识和一些推理技能，但由于完全无监督的...</li><li><a href="https://arxiv.org/abs/2210.10760">Scaling Laws for Reward Model Overoptimization</a>: 在人类反馈强化学习中，通常会针对训练用于预测人类偏好的奖励模型进行优化。由于奖励模型是一个不完美的代理，优化其值...</li><li><a href="https://arxiv.org/abs/2109.10862">Recursively Summarizing Books with Human Feedback</a>: 扩展机器学习的一个主要挑战是训练模型执行人类极难或极其耗时评估的任务。我们展示了在该问题上的进展，针对的任务是...
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1309003599489007686)** (9 条消息🔥): 

> - `NotebookLM and GitHub Repositories`
> - `Audio Prompt Generation`
> - `Using Multiple LLMs`
> - `Table of Contents in Code`
> - `ElevenLabs and Text-to-Speech AI` 


- **NotebookLM 在遍历 GitHub 仓库时遇到困难**：一位用户尝试通过输入仓库主页让 NotebookLM 遍历 GitHub 仓库，但发现效果不佳。另一位成员指出，NotebookLM 缺乏遍历网站的能力，这使得该请求变得复杂。
   - 他们建议使用该网站的 Markdown 版本，或者将页面打印并转换为 PDF 以获得更好的处理效果。
- **生成有影响力的音频提示词**：一位用户建议为 NotebookLM 提供特定的提示词（prompts），以生成对解释说明非常有用的、有影响力的音频输出。这可以帮助他人更好地理解指定的主题。
   - 这种策略旨在通过更清晰的音频内容来增强学习体验。
- **针对特定任务使用多个 LLMs**：一位成员分享了他们的工作流，即根据需求利用多个 LLMs，并赞扬了 NotebookLM 在某些生成任务中的表现。他们此前曾在一篇博客文章中介绍过这种方法。
   - 这一策略突显了利用各种 AI 工具完成基于对话的项目时的通用性和有效性。
- **代码中目录的使用**：在代码中使用目录被提及为一个特别有趣的功能，并指出它描述了每个部分及其行号。这提高了对复杂代码库的导航和理解。
   - 成员们对该功能在编码实践中的实用性表示了热切关注。
- **ElevenLabs 作为领先的 Text-to-Speech AI**：关于 ElevenLabs 的讨论指出其在 Text-to-Speech AI 领域的领先地位，超越了 RLS 和 Tortoise 等竞争对手。该成员回顾了他们在该初创公司融资前早期的体验，认可了其创新潜力。
   - 他们强调了 ElevenLabs 在制作无脸视频（faceless videos）和语音合成方面的重大影响，称其为行业中改变游戏规则的工具。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tokenwisdom.ghost.io/featured/im-an-elevenlabs-pro/">I'm an ElevenLabs Pro!</a>：深入探讨 ElevenLabs，发现语音合成领域的革命：用于动画和表演捕捉的突破性技术。体验其中的魔力 ✨</li><li><a href="https://open.spotify.com/playlist/4hcmaPIiwgHd2rm4SJCjgJ?si=ZNPffZZtSn2fjNiBJMuFJw">'Songs We Sing' Podcast Companion</a>：播放列表 · MrBland · 11 个项目
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1308960321255379015)** (25 条消息🔥): 

> - `Jensen Huang's shoutout`
> - `Podcast generation issues`
> - `Accent preferences`
> - `Functionality requests` 


- **黄仁勋 (Jensen Huang) 赞扬 NotebookLM**：在 NVIDIA 财报电话会议期间，**黄仁勋**提到使用 **NotebookLM** 加载大量文档并收听生成的播客。
   - 这突显了该应用在为用户处理内容方面的多功能性。
- **播客生成出现错误**：多位用户报告 **播客生成** 卡住并导致错误，部分用户需要等待很长时间才能获得输出。
   - 一位用户分享说，在等待两小时后，服务突然恢复正常，这表明可能存在服务器问题。
- **关于更改播客主持人口音的咨询**：一位用户询问是否可以更改播客主持人的口音，表示相比美国口音，更倾向于 **英国口音**。
   - 目前，该功能尚无可用选项。
- **NotebookLM 中期望的功能**：几位用户请求了诸如将音频时长限制在 3 分钟内，以及在 **NotebookLM** 内部翻译音频等功能。
   - 此外，还有关于 API 可能性以及调整主持人脚本表达方式以实现更自然对话的问题。
- **注意到不稳定性及安全标记 (Safety Flags)**：用户注意到 **安全标记** 的增加以及应用程序可能存在的不稳定性，导致任务中的功能受限。
   - 一位用户建议通过私信发送示例以便调试，而另一位用户指出这些瞬时问题可能是由于正在进行的改进所致。



**提到的链接**：<a href="https://www.reddit.com/r/notebooklm/comments/1gw453m/podcast_generation_not_working/">Reddit - Dive into anything</a>：未找到描述内容

  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1308913925383000177)** (2 条消息): 

> - `AI agents 架构`
> - `基于 Redis 的数据后端系统`
> - `知识图谱构建`
> - `自然语言查询`
> - `Memgraph 集成` 


- **使用 LlamaIndex 和 Redis 构建 AI agents**：参加我们 12 月 12 日的网络研讨会，学习如何使用 [LlamaIndex](https://twitter.com/llama_index/status/1859354663793066029) 和 **Redis** 构建 agentic 系统架构，以分解**复杂任务**。
   - 探索降低**成本**和优化**延迟**的最佳实践，以及关于 **semantic caching** 的见解。
- **使用 Memgraph 将数据转换为知识图谱**：学习如何设置 **Memgraph** 并将其与 [LlamaIndex](https://twitter.com/llama_index/status/1859658719082041802) 集成，从非结构化文本数据中构建**知识图谱**。
   - 参与者将探索对其图谱进行**自然语言查询**的方法以及**可视化连接**的技术。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1308912110281166989)** (16 条消息🔥): 

> - `使用 LlamaParse 进行 PDF 表格提取`
> - `Create-Llama 前端选项`
> - `Llama-Agents 弃用`
> - `NDCG 计算查询`
> - `vLLM 错误与用法` 


- **LlamaParse 提供 PDF 数据提取**：一位成员推荐使用 [LlamaParse](https://github.com/run-llama/llama_parse) 从 PDF 文件中提取表格数据，并表示它能有效解析文件以实现最佳 RAG。
   - 他们分享了一个关于其功能的 [GitHub 链接](https://github.com/run-llama/llama_parse)。
- **Create-Llama 前端困惑**：一位用户询问寻求 Create-Llama 帮助的最佳渠道，特别是关于在新版本中选择 Express 框架时缺少 Next.js 前端选项的问题。
   - 另一位参与者确认他们可以直接在频道中发布查询，并会获得团队支持。
- **Llama-Agents 逐步退出，由 Llama-Deploy 取代**：一位成员指出在升级到 Llama-index 0.11.20 时的依赖问题，并建议 **llama-agents** 已被弃用，取而代之的是 [llama_deploy](https://github.com/run-llama/llama_deploy)。
   - 他们提供了 [Llama Deploy GitHub 页面](https://github.com/run-llama/llama_deploy)的链接以获取更多上下文。
- **关于 NDCG 计算的讨论**：一位成员对 llama-index-core 的 [metrics.py 文件](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/evaluation/retrieval/metrics.py#L380)中 NDCG 计算可能存在的错误代码提出了疑问。
   - 他们建议该行应使用 `len(expected_ids)` 来计算最大可达到的 DCG，并征求对其解读的反馈。
- **vLLM 集成帮助**：一位用户报告了与 vLLM 集成相关的 KeyError，特别是在尝试使用 VllmServer 时缺少 'text' 键。
   - 另一位用户建议以 OpenAI API 模式启动 vLLM 并使用 `OpenAILike`，并提供了一个示例代码片段以供参考。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/evaluation/retrieval/metrics.py#L380">llama_index/llama-index-core/llama_index/core/evaluation/retrieval/metrics.py at main · run-llama/llama_index</a>: LlamaIndex 是适用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_deploy">GitHub - run-llama/llama_deploy: Deploy your agentic worfklows to production</a>: 将您的 agentic 工作流部署到生产环境。通过在 GitHub 上创建账号为 run-llama/llama_deploy 做出贡献。</li><li><a href="https://github.com/run-llama/llama_parse">GitHub - run-llama/llama_parse: Parse files for optimal RAG</a>: 为实现最佳 RAG 解析文件。通过在 GitHub 上创建账号为 run-llama/llama_parse 做出贡献。
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1308930713294344213)** (8 条消息🔥): 

> - `30 Days of Python`
> - `Capstone Project API`
> - `Learning Resources` 


- **参与 30 Days of Python 挑战**：一位成员分享了他们参加 **30 Days of Python** 挑战的经历，该挑战强调循序渐进的学习。
   - 他们正在利用 [GitHub 仓库](https://github.com/Asabeneh/30-Days-Of-Python) 获取整个过程中的资源和灵感。
- **关于毕业设计（Capstone Project）技术选型的讨论**：一位成员表示倾向于使用 **Go** 来开发他们的毕业设计，重点是构建一个 API。
   - 这一选择反映了在实际应用中探索不同编程语言的热情。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/hi-hello-greet-cute-puppy-gif-14845557723311629962">Hi Hello GIF - Hi Hello Greet - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/Asabeneh/30-Days-Of-Python">GitHub - Asabeneh/30-Days-Of-Python: 30 days of Python programming challenge is a step-by-step guide to learn the Python programming language in 30 days. This challenge may take more than100 days, follow your own pace.  These videos may help too: https://www.youtube.com/channel/UC7PNRuno1rzYPb1xLa4yktw</a>: 30 days of Python 编程挑战是一个在 30 天内学习 Python 编程语言的循序渐进指南。这个挑战可能会超过 100 天，请按照你自己的节奏进行。这些视频可能也会有帮助...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1309129477443555379)** (3 条消息): 

> - `Cohere Repository`
> - `Cohere Toolkit`
> - `Jupyter Notebooks`
> - `Contribution Guidelines` 


- **探索 Cohere 仓库**：一位成员强调 **Cohere GitHub 仓库**（[GitHub 链接](https://github.com/cohere-ai)）是贡献者的绝佳起点，展示了各种项目。
   - 他们鼓励探索仓库中提供的工具，并在每个项目中分享反馈或新想法。
- **用于 RAG 应用的 Cohere Toolkit**：**Cohere Toolkit**（[GitHub 链接](https://github.com/cohere-ai/cohere-toolkit)）被提及为一个专门为 **RAG 应用** 设计的高级 UI，允许快速构建和部署。
   - 这是一个预构建组件的集合，旨在提高用户生产力。
- **Notebooks 中提供的入门代码**：成员们被引导至 **notebooks** 仓库（[GitHub 链接](https://github.com/cohere-ai/notebooks)），其中包含针对各种用例的入门代码。
   - 这些 Jupyter notebooks 提供了旨在帮助用户熟悉 **Cohere Platform** 的实际示例。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/cohere-ai">cohere.ai</a>: cohere.ai 拥有 46 个可用仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/cohere-ai/cohere-toolkit">GitHub - cohere-ai/cohere-toolkit: Cohere Toolkit is a collection of prebuilt components enabling users to quickly build and deploy RAG applications.</a>: Cohere Toolkit 是预构建组件的集合，使用户能够快速构建和部署 RAG 应用。 - cohere-ai/cohere-toolkit</li><li><a href="https://github.com/cohere-ai/notebooks">GitHub - cohere-ai/notebooks: Code examples and jupyter notebooks for the Cohere Platform</a>: Cohere Platform 的代码示例和 Jupyter notebooks - cohere-ai/notebooks
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1309132867699478528)** (5 messages): 

> - `Multimodal Embeddings Launch`
> - `Research Agent Use Case`
> - `Rate Limit Concerns` 


- **Multimodal Embeddings 计划于明年发布**：令人振奋的消息是，**multimodal embed** 的改进已获得认可，并计划于明年年初在 **Bedrock** 及合作伙伴平台上发布。
   - *一名团队成员将标记速率限制问题*以供进一步讨论。
- **使用 Cohere 技术的创新研究型 Agent**：一名成员创建了一个 [Research Agent](https://researcher.customgpt.ai/)，它可以对特定主题进行 30 分钟的研究，并利用 **Cohere 的多模态 embeddings** 来选择相关图像。
   - 该工具正受到关注，但 **rate limits** 阻碍了其高效生成文章的能力，目前每 3 分钟只能生成 1 篇文章。
- **已提交提高速率限制的支持工单**：该成员提交了支持工单，请求提高 **rate limit**，以提升 Research Agent 的性能。
   - 他们表示，如果产品营销团队感兴趣，这可以作为有效使用多模态 embeddings 的**案例研究 (case study)**。



**提到的链接**：<a href="https://researcher.customgpt.ai/)">CustomGPT.AI Researcher - 基于深度研究创建高质量 AI 内容</a>：使用 CustomGPT.ai Researcher 创建超高质量、品牌安全的文章和研究报告。非常适合内容营销、SEO 和研究报告。

  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/)** (1 messages): 

rachel_47358: https://github.com/harmonydata/harmony
  

---



### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1309039506107666433)** (6 messages): 

> - `Mojo Async Progress`
> - `Mojo Community Channel`
> - `Async Runtime Overhead` 


- **Mojo 的 async 功能仍处于开发中**：成员指出 Mojo 的 async 功能仍在开发中，目前尚无实际可用的 async 函数。
   - 编译器目前将同步代码转换为异步，导致在异步调用期间代码仍以同步方式执行。
- **关于 Mojo 社区频道的讨论**：一个供成员联系和交流的社区频道已经建立，可通过 [mojo-community](https://prefix.dev/channels/mojo-community) 访问。
   - 该频道被确定为 Mojo 相关持续讨论的中心。
- **对 async 运行时开销的担忧**：一名成员提出了关于 Mojo 的 async 运行时是否在没有异步代码运行时也会产生开销的担忧，并对 async 如何编译为状态机表示困惑。
   - 讨论持续围绕在事件循环 (event loop) 中显式运行 async 函数的必要性，以及 Mojo 中 async 运行时的影响展开。


  

---

### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1308916166185455768)** (9 messages🔥): 

> - `Moonshine ASR 模型性能`
> - `Mojo 脚本优化`
> - `CPU 利用率观察` 


- **Moonshine ASR 模型面临速度挑战**：通过 Max 测试了 **Moonshine ASR 模型**的 Mojo 和 Python 版本，处理 **10s** 语音的执行时间为 **82ms**，而直接使用 ONNX 版本仅需 **46ms**。
   - 这表明 Mojo 和 Python 版本相比优化程度更高的 ONNX runtime 有 **1.8 倍的减速**。
- **Mojo 脚本出现崩溃**：在编写 **Mojo 版本**时，由于缺乏解包支持，通过 `TensorMap` 传递参数给 **Model.execute** 会导致崩溃，必须手动列出参数。
   - 这些障碍凸显了脚本的非惯用（unidiomatic）特性，作者希望获得性能提示以提升其 Mojo 技能。
- **Mojo 的优化建议**：一名成员建议通过预分配容量来优化 **tokens 列表**，以避免频繁调用 malloc，从而可能提升性能。
   - 考虑因素包括在最大长度允许的情况下实现 **InlineArray** 进行栈存储，旨在简化执行流程。
- **移除 tokens 后性能保持不变**：优化后，tokens 被从 Mojo 代码中完全移除，仅保留图执行基准测试，但这一更改并未显著影响性能。
   - 作者仍在探索提高其 ASR 模型执行效率的途径。
- **模型的 CPU 利用率问题**：一位用户观察到运行模型时 CPU 使用情况混乱，注意到它没有充分利用 CPU 能力并忽略了超线程（hyperthreads）。
   - 这种运行模型时缺乏并行性的现象表明，可能需要进一步优化以充分利用可用资源。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://gist.github.com/keveman/ea167957fb6364470cb265c5d9aa9da1">moonshine.mojo</a>: moonshine.mojo。GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://gist.github.com/keveman/d2aea1a059c9a14972783ede2d6b6862">moonshine.py</a>: moonshine.py。GitHub Gist：即时分享代码、笔记和片段。
</li>
</ul>

</div>
  

---



### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1309124519134494720)** (9 messages🔥): 

> - `Torchtune 贡献者新指南`
> - `Torchtune 的扩展包`
> - `二分查找方法建议`
> - `UV 的实操经验`
> - `TorchAO 的可选包功能` 


- **Torchtune 即将发布更清晰的指南**：成员们期待很快能有更清晰的指南，以帮助维护者和贡献者理解 **Torchtune** 所需的功能特性。
   - 这些改进可能有助于确定何时使用 fork 还是示例仓库进行演示。
- **关于 Torchtune 扩展包的建议**：一位成员提议使用类似 **torchtune[simpo]** 或 **torchtune[rlhf]** 的扩展包，主张在没有过度检查的情况下简化包的包含。
   - 这种方法旨在降低复杂性并有效管理资源问题。
- **用于 max_global_bsz 的二分查找方法**：一名成员建议对 **max_global_bsz** 使用“最后一次成功”的二分查找，默认值为小于数据集的 2 的幂。
   - 该方法还将结合 **max_iterations** 作为参数以提高效率。
- **关于使用 UV 的讨论**：一位成员询问其他人是否有使用 **UV** 的经验，并对其易用性的评价表示感兴趣。
   - 另一位成员部分肯定了它的效用，指出它看起来很吸引人且具有现代感。
- **可选包可能解决 TorchAO 问题**：有人询问可选包功能是否能解决用户需要手动下载 **TorchAO** 的问题。
   - 回复指出，虽然这可能有帮助，但还有额外的考虑因素需要处理。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1309193528953274408)** (7 条消息): 

> - `Prompt Signature Modification`
> - `Adapter Configuration`
> - `Optimization across Models` 


- **探讨 Prompt Signature 修改**：一位成员询问了为了调试目的覆盖或更改 Prompt Signature 格式的最佳方法，特别是为了避免可解析的 JSON schema 注释。
   - 讨论围绕实现此目标的方法展开，例如构建一个 Adapter。
- **DSPy 中的 Adapter 配置**：一位用户建议构建一个 Adapter，并使用 `dspy.configure(adapter=YourAdapter())` 进行配置以修改 Prompt。
   - 他们还指向了 `dspy/adapters/` 目录中现有的 Adapter 以供进一步参考。
- **针对特定情况的短语优化**：在回答关于针对 bool、int 和 JSON 等特定类型调整短语的问题时，官方澄清这些短语是基于一组维护的模型签名（Model Signatures）的。
   - *这些短语总体上并不高度依赖于单个语言模型*，这表明其制定采用了一种通用的方法。


  

---



### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1308953617528656003)** (1 条消息): 

> - `Intel AMA Session`
> - `Hackathon Insights` 


- **Intel AMA 会话提醒**：欢迎参加明天 **下午 3 点 PT (11/21)** 举行的 **Intel 黑客松 AMA**，届时将有机会与 Intel 专家进行交流。
   - 别忘了[在这里观看直播](https://www.youtube.com/watch?v=_Wm5guUXt54)并设置提醒！
- **Intel 问答机会**：这是一个在 AMA 会话期间直接向 **Intel 专家**提问并获取见解的好机会。
   - 参与者们非常期待听到来自 Intel 团队的最新动态和创新！


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1309056551071514624)** (4 条消息): 

> - `Quiz 10 Release`
> - `Hackathon Discussion` 


- **Quiz 10 尚未发布**：一位成员询问了 **Quiz 10** 的状态，询问是否已在网站上发布。
   - 另一位成员确认尚未发布，并提到一旦发布（可能在 **一两天内**）将发送电子邮件通知。
- **黑客松频道混淆**：一位成员对关于 Quiz 10 的更新表示感谢，但幽默地承认自己在错误的频道询问了 **黑客松** 的事。
   - 这次交流反映了社区内常见的频道混淆情况，为对话增添了轻松的时刻。


  

---



### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1308923419408207952)** (4 条消息): 

> - `int64 indexing`
> - `differences in ops_hip.py`
> - `maintenance of code`
> - `HIP setting in tinygrad` 


- **探讨 int64 索引的必要性**：一位用户质疑在不涉及大型 Tensor 的情况下使用 **int64 索引** 的必要性，促使其他人分享了他们的看法。
   - 另一位用户链接到了 GitHub 上相关的 Issue，为该讨论提供了更多背景。
- **剖析 ops_hip.py 文件的差异**：一位成员指出了 tinygrad 仓库中两个 **ops_hip.py** 文件之间的区别，认为前者可能因为错误的 import 而未被维护。
   - 他们注意到后者仅在一个外部基准测试脚本的上下文中被引用，而该脚本也包含错误的 import。
- **ops_hip.py 文件的维护状态**：针对维护问题的质疑，另一位用户确认 **extra** 目录下的 ops_hip.py 未被维护，而 **tinygrad** 核心版本在设置 **HIP=1** 的情况下应该可以正常工作。
   - 这表明虽然代码的某些部分可能没有被积极管理，但其他部分仍可以配置为正常运行。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/blob/master/extra/backends/ops_hip.py">tinygrad/extra/backends/ops_hip.py at master · tinygrad/tinygrad</a>：你喜欢 pytorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/blob/master/tinygrad/runtime/ops_hip.py">tinygrad/tinygrad/runtime/ops_hip.py at master · tinygrad/tinygrad</a>：你喜欢 pytorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/blob/master/test/external/external_benchmark_hip_compile.py">tinygrad/test/external/external_benchmark_hip_compile.py at master · tinygrad/tinygrad</a>：你喜欢 pytorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - tinygrad/tinygrad
</li>
</ul>

</div>

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1309247649282785300)** (3 条消息): 

> - `活动链接混淆`
> - `重新安排活动` 


- **活动链接混淆**：一名成员表示在 Luma 上找不到活动链接，并询问其状态。
   - *Chiphuyen* 作出回应并道歉，表示由于生病忘记了重新安排活动。
- **祝愿生病的成员早日康复**：另一名成员感谢 *Chiphuyen* 的更新，并祝愿其早日康复。
   - 这展示了社区在面对活动管理挑战时互助支持的精神。


  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1308989709237616670)** (3 条消息): 

> - `AI 专家请求`
> - `Carter Grant 寻求机会` 


- **紧急寻求 AI 专家**：*michel.0816* 紧急寻求 **AI 专家**，表明有迫切的协助需求。
   - 另一名成员建议在指定频道描述问题，以获得更好的曝光。
- **Carter Grant 的求职**：拥有 6 年经验的 **全栈开发人员 (full-stack developer)** Carter Grant 宣布正在寻找工作机会。
   - 他擅长包括 **React**、**Node.js** 和 **AI/ML** 在内的多种技术，并表示渴望为有意义的项目做出贡献。


  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1309044531508809748)** (1 条消息): 

> - `MI300X GPU 问题`
> - `消融集运行`
> - `间歇性 GPU 挂起`
> - `ROCm GitHub Issue` 


- **MI300X 在长时间运行中表现挣扎**：一名成员报告称，在使用 **axolotl** 对标准消融集进行 **12-19 小时** 的 8 路运行时，遇到了 **间歇性 GPU 挂起**。
   - 这些问题似乎大多发生在 **6 小时标记** 之后，引发了在 [GitHub](https://github.com/ROCm/ROCm/issues/4021) 上的讨论和跟踪。
- **短时间运行未发现问题**：该成员指出，经验表明在模型的短时间运行中不会出现 **GPU 挂起**。
   - 这一区别引发了关于使用 axolotl 进行 MI300X 长时间训练任务 **稳定性** 的疑问。
- **在 GitHub 上跟踪 GPU 挂起问题**：MI300X 运行期间持续出现的 **GPU 挂起硬件异常 (GPU hang HW Exceptions)** 已正式记录在 [GitHub Issue #4021](https://github.com/ROCm/ROCm/issues/4021) 中。
   - 描述中包含了 **loss** 和 **learning rate** 等详细指标，强调了该问题的技术背景。



**提到的链接**：<a href="https://github.com/ROCm/ROCm/issues/4021">[Issue]: Intermittent GPU Hang HW Exception by GPU on MI300X when training with axolotl · Issue #4021 · ROCm/ROCm</a>：问题描述：运行 axolotl 时，我遇到了间歇性 GPU 挂起：{&#39;loss&#39;: 0.4589, &#39;grad_norm&#39;: 1.0493940198290594, &#39;learning_rate&#39;: 5.284132841328413e-06, &#39;epoc...

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/)** (1 条消息): 

volko76: 我们仍然需要正确地进行提示 (prompt) 吗？
https://youtu.be/m3Izr0wNfQc
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1309181024604786708)** (2 条消息): 

> - `自动编码器训练` 


- **讨论自动编码器 (Autoencoder) 训练**：一名成员提到了 **训练自动编码器** 的过程，强调了其在实现模型效率方面的重要性。
   - 对话集中在提高自动编码器性能的技术和实现策略上。
- **关于自动编码器的额外见解**：成员们就当前模型中 **自动编码器架构的复杂性** 分享了各种观点。
   - 讨论包括了各种算法的有效性及其对数据表示的影响。


  

---

### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1308958351589113877)** (2 条消息): 

> - `Refact.AI demo`
> - `Web Applets project`
> - `Public AI initiative` 


- **Refact.AI 实时演示发布**：我们很高兴邀请到 [Refact.AI](https://github.com/smallcloudai) 团队成员加入我们的实时演示，讨论他们的 **autonomous agent** 和创新工具。
   - 不要错过参与对话的机会，点击[此处](https://discord.com/events/1089876418936180786/1300459081181429810)加入实时活动。
- **Mozilla 的新项目：Web Applets**：Mozilla 启动了一个名为 **Web Applets** 的早期开源项目，旨在为 Web 开发 AI 原生应用。
   - 该倡议旨在促进 AI 领域的 **open standards** 和可访问性，鼓励开发者之间的协作，详见[此处](https://discord.com/channels/1089876418936180786/1231977676458168381)。
- **倡导 Public AI**：Mozilla 在过去一年中加速了 **14 个本地 AI 项目**，重点是倡导 **Public AI** 并构建必要的开发者工具。
   - 这一努力旨在培养最先进的开源 AI 技术，并以强调社区参与的协作精神为核心。


  

---



### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1309231946924294258)** (1 条消息): 

> - `Llama 3.2 prompt usage` 


- **关于 Llama 3.2 Prompt 格式的查询**：一位成员询问为何没有使用针对 **Llama 3.2** 的特定 Prompt，并引用了 [Prompt 格式文档](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/#-function-definitions-in-the-system-prompt-)。
   - 该问题暗示了对系统 Prompt 中 **function definitions** 的好奇，表明需要对其应用进行澄清。
- **对 Prompt 适用性的兴趣**：对话展示了对理解 Llama 模型（特别是 3.2 版本）中 **applicability of prompts** 的广泛兴趣。
   - 这反映了关于通过 **effective prompting** 最大化模型性能的最佳实践的持续讨论。



**提到的链接**: <a href="https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/#-function-definitions-in-the-system-prompt-">Llama 3.2 | Model Cards and Prompt formats</a>: .

  

---



---



---



{% else %}


> 完整的频道细分内容已为邮件版进行缩减。 
> 
> 如果您想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}