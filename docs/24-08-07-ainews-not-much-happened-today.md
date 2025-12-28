---
companies:
- openai
- mistral-ai
- meta-ai-fair
date: '2024-08-08T01:50:11.687874Z'
description: '**OpenAI** 在其 API 中引入了结构化输出功能，新增了“strict”（严格）模式和“response_format”参数，支持
  **gpt-4-0613**、**gpt-3.5-turbo-0613** 以及新款 **gpt-4o-2024-08-06** 等模型。同时，他们还将 **gpt-4o**
  的价格减半，降至每百万 token 2.50 美元。**Mistral Large 2** 在高难度基准测试和编程任务中的表现优于 **gpt4-turbo**
  和 **claude-3-opus**。**Idefics3-Llama** 提供了多模态能力，并具备 10k token 的上下文窗口。**BigLlama-3.1-1T-Instruct**
  是 **llama-3-120b-instruct** 的扩展版本。新的基准测试“big_model_smell”旨在衡量模型的创造力和可靠性。**Figure
  02** 机器人配备了先进的 AI 硬件，拥有板载视觉语言模型、增强型电池以及语音对语音推理功能。**Yann LeCun** 对加利福尼亚州的 SB1047
  监管法案表示了担忧。'
id: e19ff7ac-ae0f-4018-b347-1bb26b3044ee
models:
- gpt-4-0613
- gpt-3.5-turbo-0613
- gpt-4o-2024-08-06
- mistral-large-2
- gpt4-turbo
- claude-3-opus
- idefics3-llama
- bigllama-3.1-1t-instruct
- llama-3-120b-instruct
original_slug: ainews-not-much-happened-today-4029
people:
- sama
- rohanpaul_ai
- corbtt
- guillaumelample
- mervenoyann
- maximelabonne
- aidan_mclau
- adcock_brett
- ylecun
title: 今天没发生什么事。
topics:
- structured-outputs
- function-calling
- json-schema
- benchmarking
- multimodality
- context-windows
- model-scaling
- ai-hardware
- vision
- speech-processing
- robotics
- ai-regulation
---

<!-- buttondown-editor-mode: plaintext -->**[anonymous](https://x.com/AndrewCurran_/status/1821051919768678701) [strawberries](https://x.com/swyx/status/1821359574068146681) are all you need.**

> 2024年8月6日至8月7日的 AI 新闻。我们为您检查了 7 个 subreddits、[**384** 个 Twitter](https://twitter.com/i/lists/1585430245762441216) 和 **28** 个 Discord（**249** 个频道和 **2423** 条消息）。预计节省阅读时间（以 200wpm 计算）：**247 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

今天没有明显的重大新闻，但有很多有趣的小亮点：

- Mistral Large 的外部评分已经公布，它们在硬核的 Lmsys 提示词以及像 [Aidanbench](https://x.com/aidan_mclau/status/1821334577576644935) 这样的独立基准测试中表现非常出色——达到了 Gemini Pro 级别。
- 从零开始编写一个 [Vision Language Model](https://x.com/hkproj/status/1821081257712705848)！（感谢 Sam Julien 在 LS Discord 中挑选了这一条）
- 新的 PyTorch [FlexAttention](https://x.com/cHHillee/status/1821253769147118004) 包含了所有 Attention 变体，包括 FlashAttention 2（但不包括 FA 3！）的 API，以及[日益流行的局部-全局注意力谱系](https://buttondown.email/ainews/archive/ainews-shazeer-et-al-2024/)，包括 Sliding Window。
- 查看 [Grokfast 优化器](https://x.com/_clashluke/status/1820810798693818761)！

当然，你也可以在 [**Segment Anything 2**](https://www.latent.space/p/sam2) 上多花一个 epoch，它现在已经在 Latent Space Podcast 上线了。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，从 4 次运行中择优。

**OpenAI Structured Outputs 与模型更新**

OpenAI 在其 API 中引入了 Structured Outputs，允许开发者为模型响应强制执行特定的 JSON schemas。该功能目前已在多种模型中得到支持，包括 gpt-4-0613、gpt-3.5-turbo-0613 及其后续版本。[@sama](https://twitter.com/sama/status/1820881534909300769) 宣布了这一备受期待的功能，根据 OpenAI 的评估，该功能在匹配输出 schema 方面实现了 100% 的可靠性。此次更新包括：

- 为 function calling 引入了全新的 "strict" 模式，确保输出与提供的工具定义完全匹配。
- 新增 "response_format" 参数，允许指定 JSON 输出 schemas。
- 推出新模型：gpt-4o-2024-08-06。

[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1820886172476047824) 强调，此次更新在匹配输出 schema 方面达到了 100% 的可靠性，这对于模型不调用工具而是以结构化方式响应用户的下游任务特别有用。

此外，[@corbtt](https://twitter.com/corbtt/status/1820910339388825762) 注意到 OpenAI 在没有正式公告的情况下悄悄将 GPT-4o（原版，非 mini 版）的价格降低了 50%，目前其价格页面显示为 $2.50/1M tokens。

**AI 模型发展与基准测试**

几款新的 AI 模型和基准测试相继发布：

1. Mistral Large 2：[@GuillaumeLample](https://twitter.com/GuillaumeLample/status/1820833645009277388) 宣布发布 Mistral Large 2，它在编程、困难提示词 (hard prompts)、数学和长查询类别中表现异常出色，在某些领域超越了 GPT4-Turbo 和 Claude 3 Opus。它目前在 Arena hard 排行榜上名列前茅，并且是一款 open-weight 模型。

2. Idefics3-Llama：[@mervenoyann](https://twitter.com/mervenoyann/status/1820896952957153762) 介绍了 Idefics3-Llama，这是一款基于 Llama 3.1 的多模态模型，支持任意数量的文本图像交错输入，并拥有 10k tokens 的巨大上下文窗口 (context window)。

3. BigLlama-3.1-1T-Instruct：[@maximelabonne](https://twitter.com/maximelabonne/status/1820746727638323531) 展示了 Meta-Llama-3-120B-Instruct 的扩展版本，通过 Llama 3 70B 的自合并 (self-merge) 创建而成。

4. 新基准测试：[@aidan_mclau](https://twitter.com/rez0__/status/1820853537733021770) 引入了一个名为 "big_model_smell" 的新基准测试，用于衡量创造力、可靠性、注意力和指令遵循 (instruction following) 能力。

**AI 硬件与机器人**

[@adcock_brett](https://twitter.com/adcock_brett/status/1820792697315348640) 介绍了 Figure 02，称其为世界上最先进的 AI 硬件。主要特性包括：

- 6 个摄像头
- 电池容量提升 50% 以上
- 板载 Vision Language Model (VLM)
- 3 倍的 CPU / GPU 算力
- 第 4 代手部
- 集成布线
- 外骨骼结构
- 语音到语音推理 (Speech-to-speech reasoning) 能力

该机器人专为自主运行而设计，包含定制的 2.25 KWh 电池组，目标是每天实现长达 20 小时的有效工作。

**AI 安全与监管**

[@ylecun](https://twitter.com/ylecun/status/1820927645757940178) 对加州的 SB1047 法案（前沿人工智能模型安全创新法案）表示担忧，称其无法解决预期问题，并可能损害学术界、小型科技公司和开源社区的 AI 研发 (AI R&D)。[@fchollet](https://twitter.com/fchollet/status/1820862042934493223) 呼应了这些担忧，认为让开源模型开发者对下游所有微调模型 (fine-tuned models) 负责是毫无道理的，这可能会阻碍开源模型的分享。

**其他 AI 发展**

- [@omarsar0](https://twitter.com/omarsar0/status/1820941367784136718) 讨论了 Structured Outputs 在提高 LLM 应用性能和可靠性方面的重要性。
- [@jeremyphoward](https://twitter.com/jeremyphoward/status/1820955161579417957) 宣布了 FastHTML，这是一个不断增长的实时 FastHTML 代码示例库，用于构建交互式组件和应用程序。
- [@LangChainAI](https://twitter.com/LangChainAI/status/1820966277978235004) 在其针对 Python 和 JavaScript 的最新发布候选版本 (release candidates) 中，引入了对 OpenAI 新 Structured Outputs 功能的支持。

这些进展展示了 AI 模型能力、硬件集成方面的快速进步，以及该领域围绕 AI 安全和监管持续进行的讨论。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. LLM 作为研发中的生产力助推器**

- **[auto-md | 工具 | 一键将文件/压缩包 + GitHub 仓库转换为 Markdown 文档 (.md)](https://i.redd.it/dl555pnlw5hd1.png)** ([评分: 62, 评论: 10](https://reddit.com//r/LocalLLaMA/comments/1em1sqi/automd_tool_one_click_convert_fileszips_github/)): 工具 **auto-md** 已更新 **Windows .exe** 版本，允许用户一键将文件、压缩包和 GitHub 仓库转换为 Markdown 文档。开发者计划很快发布 **Mac app**，并对收到的支持（包括 GitHub stars 和之前帖子的用户反馈）表示感谢。
  - **Dark_Fire_12** 分享了构建类似工具的另一种方法，选择使用**文件扩展名过滤**而不是文件夹深度搜索。他们包含了一张[截图](https://preview.redd.it/xcn7tq8ye6hd1.png?width=1565&format=png&auto=webp&s=9f86806aebdf8d3f37742f3b88aadb7100abf0b8)来展示其实现。
  - **Environmental-Car267** 提到为个人使用创建了两个**类似的工具**：一个用于将代码库复制到剪贴板以便粘贴到 **Sonnet/GPT**，另一个让 AI 自主选择重要文件。这些工具在处理过程中会排除某些文件夹和文件。
- **Google DeepMind 的研究科学家如何使用 LLM** ([评分: 318, 评论: 89](https://reddit.com//r/LocalLLaMA/comments/1elz2ur/how_a_research_scientist_at_google_deepmind_uses/)): **Nicholas Carlini**，**Google DeepMind** 的一名研究科学家，在一篇详细的博客文章中分享了他利用 **Large Language Models (LLMs)** 提升生产力的方法。文章强调了用 AI **增强人类能力**的重大价值，并建议在追求全自动 AI 系统之前，这一中间步骤至关重要。
  - 用户一致认为 **LLMs** 既**被过度炒作也被低估**，许多人要么夸大其能力，要么完全不屑一顾。当在“**知识边界**”操作时，该技术特别有用，有助于填补理解不全的空白。
  - 文章展示了 LLMs 产生**幻觉**的倾向，因为它错误地声称没有 Podman 的 Python 库，尽管存在 [podman-py](https://podman-py.readthedocs.io/en/latest/)。用户强调，评估 LLMs **能做什么**比关注其局限性更重要。
  - 许多用户报告使用 LLMs 带来了显著的**生产力提升**，其中一人估计编码速度**提高了 50%**。LLMs 在学习新技术、自动化日常任务和调试方面特别有帮助，尽管一些人对其在学术写作中的使用表示担忧。


**主题 2. AI 模型压缩与量化进展**

- **将 123B Mistral-Large-Instruct-2407 量化至 35 GB，准确率仅下降 4%。** ([评分: 77, 评论: 54](https://reddit.com//r/LocalLLaMA/comments/1elbn3q/quantize_123b_mistrallargeinstruct2407_to_35_gb/)): 作者使用 **EfficientQAT** 算法将 **123B Mistral-Large-Instruct-2407** 模型从 **228.5 GB** 量化至 **35.5 GB**，在 **5 个零样本推理任务**中平均**准确率仅下降 4%**。该量化模型使用 **INT2 bits** 和 **64 的组大小 (group size)**，采用 **GPTQ v2** 格式打包并上传至 [HuggingFace](https://huggingface.co/ChenMnZ/Mistral-Large-Instruct-2407-EfficientQAT-w2g64-GPTQ)，作者正在寻求将其转换为 GGUF 或 EXL2 格式的帮助。
    - 用户强烈表达了对该量化模型 **GGUF 格式**版本的需求，多条评论请求将当前的 **GPTQ v2 格式**进行转换。
    - 也有人对模型性能表示怀疑，一位用户指出其 **perplexity (困惑度) 增加了 100%**，另一位用户将准确率下降修正为 **5.4%** 而非 4%。
    - 一名用户尝试使用 **Exllamav2 0.1.7** 加载模型但遇到了 **RuntimeError**，这表明该量化版本与当前的加载器存在兼容性问题。


**主题 3. 开源 AI 工具与多模态生成**

- **[开源 Text2Video 生成来了！ChatGLM 的创作者刚刚开源了 CogVideo。](https://github.com/THUDM/CogVideo)** ([Score: 61, Comments: 4](https://reddit.com//r/LocalLLaMA/comments/1elbdvr/open_source_text2video_generation_is_here_the/))：**ChatGLM** 的创作者开源了 **CogVideo**，这是一个**文本生成视频 (text-to-video generation)** 模型。CogVideo 可以根据文本提示生成 **24 FPS**、**256x256 分辨率**的 **5 秒视频**，代表了开源 AI 视频生成能力的重大进步。
    - **CogVideo** 规格：**6 秒**长，**8 FPS**，**720x480 分辨率**，使用 SAT 进行推理需要 **18GB GPU memory**，使用 diffusers 则需要 **36GB**。用户注意到其连贯性良好，但略有卡顿，可以通过 flowframes 修复。
    - 现已提供 CogVideo 的 [ComfyUI wrapper](https://github.com/kijai/ComfyUI-CogVideoXWrapper)，增强了其易用性以及与现有工作流的集成。
    - 该模型的[许可证 (license)](https://github.com/THUDM/CogVideo/blob/main/Model_License) 包含对商业用途的限制，并禁止可能 *“损害中国国家安全和国家统一”* 的用途，这引发了对其开源状态的讨论。

## All AI Reddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 模型开发与发布**

- **Salesforce 的 xLAM-1b 模型**：一个 10 亿参数的模型，[在 Function Calling 方面实现了 70% 的准确率，超越了 GPT 3.5](https://www.reddit.com/r/LocalLLaMA/comments/1dz8g10/salesforce_tiny_giant_xlam1b_model_surpasses_gpt/)。尽管体积相对较小，但被称为“Function Calling 巨人”。

- **带有 Function Calling 的 Phi-3 Mini (6月版)**：Rubra AI 发布了更新后的 Phi-3 Mini 模型，[具备 Function Calling 能力](https://www.reddit.com/r/LocalLLaMA/comments/1dzhe38/phi3_mini_june_with_function_calling/)，可与 Mistral-7b v3 竞争，并优于基础版 Phi-3 Mini。

**AI 研究与应用**

- **Figure 02**：[Figure AI 推出的新型人形机器人](https://www.reddit.com/r/singularity/comments/1elfvmt/introducing_figure_02/)，展示了机器人技术与 AI 集成的进步。

- **AI 图像生成**：关于 [r/StableDiffusion 成为开源图像模型通用中心](https://www.reddit.com/r/StableDiffusion/comments/1ele3ub/this_sub_should_become_the_general_place_for/)的讨论，类似于 r/LocalLLaMA 如何成为 LLM 的中心场所。

**AI 伦理与安全**

- **OpenAI 安全团队辞职**：一个[幽默帖子根据“新 Scaling Laws”预测下一任 OpenAI 安全负责人将于 8 月 30 日辞职](https://www.reddit.com/r/singularity/comments/1em1haa/according_to_new_scaling_laws_the_next_openai/)。这突显了 AI 安全和伦理方面持续存在的挑战。

**AI 对教育和职业的影响**

- **Nick Bostrom 谈长期投资**：Bostrom 建议，[由于 AI 时间线缩短，像大学学位这样的长期投资可能不再值得](https://www.reddit.com/r/singularity/comments/1em1uc1/nick_bostrom_says_it_may_not_be_worth_making/)。这引发了关于 AI 对传统教育和职业道路潜在影响的辩论。

**AI 生成内容**

- **来自平行现实的电影海报**：使用 Flux Pro + SUPIR Upscale 创作的 [AI 生成电影海报](https://www.reddit.com/r/StableDiffusion/comments/1em3etw/movie_posters_from_a_parallel_reality/)，展示了 AI 在视觉艺术方面的创作潜力。

**梗图与幽默**

- 与 AI 和技术相关的各种梗图和幽默帖子，包括 [AI 生成图像的对比](https://www.reddit.com/r/singularity/comments/1elol2v/left_or_right/)以及[对反 AI 情绪的讽刺性看法](https://www.reddit.com/r/singularity/comments/1elhmef/youd_think_that_this_was_made_by_a_17th_century/)。


---

# AI Discord 回顾

> 总结的总结之总结

**1. LLM 进展与基准测试**

- **DeepSeek-V2 在 MT-Bench 上表现优于 GPT-4**：来自 DeepSeek AI 的 [DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2) 迅速攀升至 **ChatbotArena** 和 **MT-Bench** 等排行榜的前列，在超过 50,000 场对决中表现优于 **GPT-4-Turbo** 和 **Claude 3 Opus** 等模型。
   - 用户在 **AlignBench** 和 **MT-Bench** 等基准测试上比较了*模型性能*，[DeepSeek 的发布](https://x.com/deepseek_ai/status/1787478986731429933)引发了热烈讨论。
- **新模型推动 SOTA 发展**：来自 IBM 的 **[Granite-8B-Code-Instruct](https://huggingface.co/ibm-granite/granite-8b-code-instruct)** 等新型开源模型增强了代码任务的指令遵循能力，而 **[DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2)** 则拥有 **236B 参数**。
   - 示例：[DeepSeek-V2 发布公告](https://x.com/deepseek_ai/status/1787478986731429933)。
  
**2. 模型性能优化**

- **AQLM 和 QuaRot 量化 Llama-3-70b**：像 **AQLM** 和 **QuaRot** 这样的**量化 (Quantization)** 技术旨在保持性能的同时，在单个 GPU 上运行像 **Llama-3-70b** 这样的大型语言模型 (**LLMs**)，正如在 RTX3090 上运行的 [AQLM 项目](https://github.com/Vahe1994/AQLM) 所示。
   - *用户讨论了量化方法在优化大模型推理方面的潜在收益和权衡。*
- **DMC 在 H100 GPU 上提升 370% 吞吐量**：根据 `@p_nawrot` 的 [DMC 论文](https://arxiv.org/abs/2403.09636)，通过**动态内存压缩 (Dynamic Memory Compression, DMC)** 提升 **Transformer 效率**的努力，有望在 **H100 GPU** 上实现高达 **370%** 的吞吐量提升。
   - 成员们探索了诸如*将 CUDA 操作与 NVIDIA 的 Thrust 库融合*等技术，以在模型推理期间最大化 GPU 利用率。
- **Thrust 在接近带宽极限时优化 CUDA 操作**：讨论集中在**优化 CUDA 操作**上，例如融合逐元素操作，利用 **NVIDIA 的 Thrust 库**及其 `transform` 功能来实现接近带宽饱和的性能。
   - [Thrust 文档](https://nvidia.github.io/cccl/thrust/api/groups/group__modifying.html#function-for-each) 提供了关于这些优化策略的见解。
  
**3. 微调挑战与 Prompt Engineering 策略**

- **Axolotl 应对 Prompt 设计难题**：在使用 [Axolotl prompters.py](https://github.com/OpenAccess-AI-Collective/axolotl/blob/3367fca73253c85e386ef69af3068d42cea09e4f/src/axolotl/prompters.py#L47) 等工具进行微调和评估期间，**Prompt 设计**和正确模板使用（包括 end-of-text token）对模型性能的影响受到了高度关注。
   - *用户分享了围绕 Prompt Engineering 挑战以实现预期结果的经验和见解。*
- **Logit Bias 调整 Prompt 以获得更多控制**：讨论了 **Prompt Engineering** 的策略，例如将复杂任务拆分为多个 Prompt，并根据 [OpenAI 的 logit bias 指南](https://help.openai.com/en/articles/5247780-using-logit-bias-to-alter-token-probability-with-the-openai-api) 研究 **logit bias** 以实现细粒度控制。
   - 成员们分享了*通过精心设计来增强 Prompt 有效性*的经验和技术。
- ***RET* Token 提升信息检索**：根据一篇 [ArXiv 论文](https://arxiv.org/abs/2404.19705)，研究探索了教导 LLM 在不确定时使用 `<RET>` token 进行**信息检索**，从而提高在低频查询上的性能。
   - 社区讨论了*新颖的 Prompt Engineering 方法*，以扩展语言模型的能力。
  


**4. 多模态 AI 与生成式建模创新**

- ****Idefics2 和 CodeGemma：多模态惊奇****：新的多模态模型如 **[Idefics2 8B Chatty](https://twitter.com/sanhestpasmoi/status/1787503160757485609)** 专注于提升对话交互，而 **[CodeGemma 1.1 7B](https://twitter.com/reach_vb/status/1786469104678760677)** 则精进了编程能力。
   - 这些发布展示了*多模态 AI 能力*在各个领域的快速进展。
- ****Phi3 为 WebGPU 带来强大的 AI 聊天机器人****：**[Phi 3](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/)** 模型使强大的 AI 聊天机器人能够通过 WebGPU 在浏览器中运行，突显了*更易获取且更具隐私性的 AI 交互潜力*。
   - 社区成员讨论了这一进展对用户隐私和控制的影响。
- ****IC-Light 推进开源图像重打光****：开源项目 **[IC-Light](https://github.com/lllyasviel/IC-Light)** 专注于改进**图像重打光 (Image Relighting)** 技术，为不断增长的生成式 AI 开源生态系统做出贡献。
   - 成员们分享了与 *AI 模型驱动的图像处理能力*相关的见解和资源。
  

---

# PART 1: Discord 高层级摘要

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **LoRA 让 Stable Diffusion 变得轻量**：LoRA 模型是 Stable Diffusion 的小型版本，通过修改标准 checkpoint 使体积缩小 **10 到 100 倍**，可以安装在 stable-diffusion-webui/models/Lora 目录下。
   - 要使用这些模型，只需在 prompt 中包含语法 `<lora:filename:1.0>`，从而优化你的工作流。
- **Pony 模型提供锐利的线稿**：**Pony 模型**专为无阴影的干净线稿而设计，与风格 LoRA 结合使用效果最佳。
   - 用户强调，在使用线稿风格 LoRA 时，将 Pony 模型作为基础模型对于实现理想的美学效果至关重要。
- **ControlNet 像魔法一样转换图像**：ControlNet 有助于在保留原始结构的同时将照片转换为线稿，极大地提升了图像处理能力。
   - 社区成员建议使用 depth ControlNet 或 IPAdapter 作为这些转换的有效方法。
- **r/stablediffusion 爆发社区风波**：关于 r/stablediffusion 版块近期管理层变动的讨论揭示了社区驱动项目与公司主导项目之间的紧张关系。
   - 这种反思引发了关于 AI 艺术领域社区动态中所面临的**控制权问题**的热烈对话。
- **AI 硬件争论中怀疑论占据上风**：对于在 ML 任务中使用 **AMD GPU**，社区达成了一致的反对意见，建议倾向于 **NVIDIA**，或者像 **Groq** 这样的替代方案更受青睐。
   - 参与者还谈到了硬件股票的波动性，引发了关于优化 AI 性能的未来选择的讨论。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 微调模型的挫败感**：用户在使用 Unsloth 进行**微调**时面临问题，特别是模型无法正常保存，以及与需要 for_inference() 方法的 **PPO trainer** 集成时的挑战。
   - 许多人指出旧版本的集成更加顺畅，这加剧了社区目前的挫败感。
- **Llama3.1 推理时间不一致**：报告显示，在微调后的 **Llama3.1** 上进行推理时**响应时间不一致**，但在多次调用后有所改善。
   - 建议用户运行测试，以确认初始延迟是否如预期般影响性能。
- **探索 Unsloth 的多 GPU 支持**：Unsloth 的**多 GPU 支持**正处于 Beta 阶段，旨在提高速度并减少 VRAM 占用，目前测试人员签署了 NDA。
   - 参与者预计在进一步完善后将推出付费订阅模式。
- **介绍 BigLlama 3.1-1T-Instruct**：一个新模型 [BigLlama-3.1-1T-Instruct](https://huggingface.co/mlabonne/BigLlama-3.1-1T-Instruct) 正在作为 **Meta-Llama** 的自合并版本进行测试，但用户报告称其合并权重尚无法正常工作。
   - 社区反馈强调，由于训练不完整，该模型目前阶段基本处于**无用**状态。
- **LLaMA3 的高性价比配置**：有人请求在 **RunPod** 上经济高效地运行 **LLaMA3** 的策略，反映了社区对优化部署成本的关注。
   - 成员们讨论了在控制成本的同时管理资源需求的挑战。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Google 通过 Gemma 2 2B 增强 Gemma 系列**：Google 推出了 [Gemma 2 2B](https://huggingface.co/collections/google/gemma-2-2b-release-66a20f3796a2ff2a7c76f98f)，拥有 **2.6B 参数**，专为端侧（on-device）使用设计，并配有 **ShieldGemma** 和 **Gemma Scope** 以实现高级功能。
   - 此次发布使 Gemma 2 在端侧机器学习工具中具有很强的竞争力。
- **Diffusers 与 FLUX 的新集成**：一位成员称赞了新的 [FLUX 的 Diffusers 集成](https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell)，显著提升了文本生成图像的能力。
   - 他们分享了一个 [gist](https://gist.github.com/sayakpaul/b664605caf0aa3bf8585ab109dd5ac9c)，介绍如何在资源有限的情况下高效使用 FLUX。
- **Argilla 2.0 发布，助力更好的数据管理**：[Argilla 2.0](https://huggingface.co/blog/dvilasuero/argilla-2-0) 作为一款专注于数据可用性的强大 AI 工具亮相，承诺为创作者提供增强的管理功能。
   - 社区成员对首个开放合成数据集 **magpie-ultra-v0.1** 表示欢迎，该数据集由 Llama 3.1 生成，旨在改进数据集的创建。
- **OpenAI 推广结构化输出（Structured Outputs）**：OpenAI 发布了一篇 [博客文章](https://openai.com/index/introducing-structured-outputs-in-the-api/)，建议在其 API 中使用结构化输出，但并未过多提及之前的工作。
   - 这一转变凸显了在采用有效实践的同时，对基础性贡献缺乏认可的趋势。
- **命名实体识别（Named Entity Recognition）数据集可用**：一个包含 **5029 份标注简历**（使用 NER 标记了 IT 技能）的数据集已在 [Kaggle](https://www.kaggle.com/datasets/mehyarmlaweh/ner-annotated-cvs) 上发布。
   - 该数据集包含从 PDF 中**手动标注的技能**，并以 **JSON** 格式提供，适用于 **Spacy** 等 NLP 工具。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **使用 AnythingLLM 配置 LM Studio**：在解决文件访问权限和影响性能的硬件限制后，用户成功将 **AnythingLLM** 与 **LM Studio** 配合使用。一位用户确认在加载自定义 **Gemma v2** 模型后运行成功。
   - 几位用户分享了对设置过程中常见问题的见解，重点强调了确保文件路径正确的重要性。
- **优化 LM Studio 的性能设置**：“保持模型在内存中”（Keep Model in Memory）功能引起了褒贬不一的反应，一些用户建议默认禁用该功能以避免不必要的 RAM 占用。专家讨论了其对性能的有限影响，特别是对于较大的模型。
   - 用户分享了经验，指出禁用此功能可以在系统资源和模型性能之间提供更好的平衡。
- **对音频转录功能的关注**：用户表达了自动化音频转录的愿望，但注意到 **LM Studio** 缺乏对音频输入的直接支持。对于优先考虑隐私的用户，讨论了 API 和开源 TTS/STT 解决方案等替代方案。
   - 一些成员报告了使用特定 API 的成功经验，而另一些成员则更倾向于本地解决方案以确保数据机密性。
- **探索多 GPU 配置**：用户寻求关于使用 **ComfyUI** 管理多个 GPU 的建议，并探索有效分配 GPU 资源的脚本。一位用户提议使用启动器来简化 CUDA 设备的设置，而无需修改配置文件。
   - 讨论包括了对 GitHub 上现有脚本的建议，这些脚本可以简化多 GPU 设置。
- **对 Phi-3 模型支持的担忧**：用户对 **llama.cpp** 缺乏 **Phi-3** 模型支持表示担忧，这影响了 **Oobabooga WebUI** 等界面的兼容性。这引发了关于近期更新和社区反应的更广泛讨论。
   - 成员指出，该问题可能需要开发者之间的协调，以确保与最新模型的无缝集成。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Gameboy 模拟器简化 RL**：在 [PufferLib GitHub 仓库](https://github.com/PufferAI/PufferLib/blob/729003f9cb89845cc1a69a65e5a2431b2d0542bd/pufferlib/environments/pokemon_red/environment.py#L15) 中可以找到 **Gameboy 模拟器** 的详细设置，简化了游戏环境中的强化学习（RL）。
   - 这种方法允许用户在不需要进行大量速度优化的情况下探索 RL 概念。
- **PyTorch 2.4 在 CUDA 12.4 上表现不佳**：用户报告了 **PyTorch 2.4** 在 **CUDA 12.4** 上的问题，指出与 **CUDA 12.1** 等早期版本相比性能有所下降。
   - 人们对兼容性以及回退到以前的 CUDA 版本时可能带来的改进表示关注。
- **AMD 声明后 ZLUDA 3 被撤回**：在 AMD 声称发布许可无效后，作者已下架 **ZLUDA 3**，详情见 [GitHub 页面](https://github.com/vosen/ZLUDA)。
   - 这一情况引发了关于 AMD 在开发生态中的角色以及对开源贡献影响的讨论。
- **关于 INT8 量化技术的辩论**：围绕 **INT8 对称量化** 的讨论揭示了在训练期间使用 **127.5** 的缩放比例时，对权重更新偏差的担忧。
   - 成员们辩论了全范围（full range）与受限范围（restricted range）量化的有效性，强调了模型完整性方面的潜在挑战。
- **引入 SARATHI 以提高 LLM 效率**：一个名为 **SARATHI** 的新框架通过采用分块预填充（chunked-prefills）和改进的批处理策略，解决了 LLM 推理中的低效问题。
   - 该方法旨在提高 GPU 利用率，同时减少模型推理过程中流水线并行（pipeline parallelism）的不平衡。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **UltraSteer-V0 数据集发布**：Nvidia 推出了 **UltraSteer-V0 数据集**，包含 **230 万次对话** 和 **280 万个轮次**，并使用 **Llama2-13B-SteerLM-RM** 奖励模型在 **9 个细粒度信号** 上进行了标注。
   - 尽管是 **V0 版本**，但由于经过了 **22 天** 的广泛去重，它具有独特的线程延续性，可在 [Hugging Face](https://huggingface.co/datasets/Avelina/UltraSteer-v0) 上获取。
- **保险模型微调的挑战**：一位用户询问了在 **保险领域** 微调模型的经验，强调了该行业特有的挑战。
   - 这次讨论吸引了关于在保险背景下有效应用 AI 所需的调整和注意事项的建议。
- **Flux AI 能力引发热议**：Flux AI 展示了在 **文本理解**、**提示词理解** 和 **图像生成** 方面的技能，引发了成员们的兴奋。
   - 许多用户称赞其能力，一些人已经在使用其 Pro 版本以获得增强的性能。
- **开放医疗推理任务倡议**：由 **Open Life-Science AI** 协作领导的 **Open Medical Reasoning Tasks** 项目旨在为医疗保健领域的 LLM 编制一份强大的任务列表，邀请各利益相关方贡献力量。
   - 一位成员赞扬了这一协作努力，强调了对推动医疗领域 AI 发展的集体影响；更多详情可在 [GitHub](https://github.com/openlifescience-ai/Open-Medical-Reasoning-Tasks) 查看。
- **MiniCPM-Llama3-V 模型更新**：成员们讨论了 **MiniCPM-Llama3-V** 的最新更新，该模型声称在处理 **多图输入** 和 OCR 任务方面具有改进的能力。
   - 这最初引发了一些怀疑，但随着展示其应用和效果的新示例出现，大家的兴奋感与日俱增。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Web 开发人员转型为 AI Engineering**：讨论强调了由于高需求和 ML 工程师短缺，**Web 开发人员**向 **AI Engineering** 转型的趋势日益增长，参与者分享了关于调整技能组合的见解。
   - 成员们强调，Web 开发人员通常被期望在承担传统开发职责的同时实施 AI 项目。
- **OpenAI 面临领导层变动**：**OpenAI** 的一波领导层变动引发了对公司未来轨迹和稳定性的担忧，在社区中引发了激烈的辩论。
   - 参与者推测了这些离职对 OpenAI 整体方向的潜在影响。
- **Generative AI 变革零售业**：**Generative AI** 应用在零售领域蓬勃发展，特别是在跨平台制作产品描述方面，例如来自 **L'Oreal** 的案例。
   - 讨论提出了关于评估 AI 生成内容有效性的关键点，以及对更好性能指标的需求。
- **Structured Outputs 功能在 GPT-4o 中首次亮相**：**OpenAI** 在 **GPT-4o** 中推出了 Structured Outputs 功能，使模型能够遵循 JSON schemas，且可靠性较之前的模型有所提高。
   - 社区成员认为这一进步是迈向在 AI 中生成更受控且结构化数据输出的重要一步。
- **对 Energy-Based Language Modeling 的怀疑**：关于与 **Extropic AI** 研究员会面的轶事凸显了对其在 **Energy-Based Language Modeling** 知识储备的怀疑，并对其公信力提出了质疑。
   - 这次交流引发了关于新兴初创公司在复杂 AI 领域专业知识的更广泛讨论。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI DevDay 走向全球！**：OpenAI 将在今年秋季举办巡回 **DevDay**，活动地点包括**旧金山**、**伦敦**和**新加坡**，届时将提供动手实践环节和针对开发者的最佳实践。参与者可以直接与 **OpenAI 工程师**交流，观看创新成果的实际应用，详情请见[此处](https://openai.com/devday/)。
   - 该活动承诺为全球开发者提供一个联系平台，分享见解并重新定义 AI 开发实践。
- **DALL-E 3 模型显示出结果的可变性**：成员们讨论了 **DALL-E 3 模型**及其生成结果的可变性，重点对比了 Llama 模型以及安全过滤器的影响。值得注意的是，输出质量的差异归因于 OpenAI 实施的安全措施。
   - 社区正在分析这些差异，同时探索 AI 生成质量和安全问题的细微差别。
- **Search GPT 现已可用！**：**Search GPT** 已正式推出，引发了用户对其功能和应用的兴趣。成员们正在积极讨论如何计划在工作流中利用这一新功能。
   - 此次推出引发了关于 Search GPT 用户体验和实际实施的问题。
- **对 Generative AI 在游戏领域应用的兴奋**：成员们对 Generative AI 在增强游戏体验方面的潜力感到兴奋，特别是在 **BG3** 和 **Pathfinder** 等作品中。他们预见到改进的 AI 能力将带来动态的 NPC 互动。
   - 讨论集中在创建沉浸式环境，使角色设计和玩家选择无缝融合。
- **ChatGPT-4o 的更新引发疑问**：用户注意到 **ChatGPT-4o** 的性能发生了重大变化，推测其近期进行了更新。成员们正在讨论这些变化对输出一致性和用户体验的影响。
   - 对版本 `gpt-4o-2024-08-06` 的观察引发了关于这些更新对开发者和用户未来意味着什么的进一步讨论。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI 出现技术问题**：用户报告了 Perplexity Pro 应用的各种**技术问题**，包括无法切换 LLM 以及库丢失，引发了对其功能的重大担忧。
   - 部分功能意外恢复，表明这可能是间歇性问题而非系统性故障。
- **NVIDIA 的 Blackwell GPU 遭遇延迟**：由于关键的**设计缺陷**和 **CoWoS-L** 封装技术问题，**NVIDIA 的 Blackwell GPU** 已被推迟，需要重新设计处理器晶圆。
   - 这些挫折推迟了生产时间表，影响了对下一代 GPU 的预期。
- **语言模型对比升温**：关于 **GPT-4o** 与 **Turbo** 性能对比的辩论爆发，用户表达了褒贬不一的体验，特别是在响应速度和有效性方面。
   - 一些用户注意到 **GPT-4o** 在处理新指令时表现吃力，引发了重新评估 LLM 能力的呼声。
- **探索内容推荐引擎**：一个旨在开发**内容排序和推荐引擎**的新大学项目引起了关注，强调了在创建相关排序算法时用户输入的需求。
   - 成员们建议利用 **RAG**（检索增强生成）原则来增强项目的有效性。
- **API 功能受到审查**：用户对 **API 差异**表示担忧，用户遇到的数据返回损坏导致了对 API 可靠性的怀疑。
   - 此外，所有 Perplexity API 模型将于 **2024 年 8 月 12 日**弃用，这引起了对未来使用所需调整的关注。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **机械式异常检测（Mechanistic Anomaly Detection）的新方法**：团队研究了使用 [Neel Nanda 的归因补丁技术](https://blog.eleuther.ai/mad_research_update/)对语言模型进行异常检测的“机械式”方法，但基于激活（activations）的传统基准表现更好。
   - 他们发现通过评估整个批次（batches）而非单个点，性能有所提高，但在不同任务中的成功程度各异。
- **关于 SB1047 AI 安全法案的辩论升温**：成员们对 SB1047 进行了激烈的讨论，有人担心它可能会扼杀创新，而另一些人则主张 AI 研究中必要的问责制。
   - 辩论者表示，该法案的责任条款可能会阻碍开放研究工作，表明需要在监管与创新之间取得平衡。
- **Meta 在分布式 AI 训练方面的进展**：在 [ACM SIGCOMM 2024](https://conferences.sigcomm.org/sigcomm/2024/) 上，Meta 展示了关于 [RDMA over Ethernet 用于分布式 AI 训练](https://dl.acm.org/doi/10.1145/3651890.3672233)的论文，重点关注支持训练 **LLAMA 3.1 405B** 等模型的基础设施。
   - 此次演讲强调了大规模 AI 应用引发的日益增长的通信需求。
- **稀疏自编码器（SAE）发展回顾**：成员们引用了一篇 [关于 SAE 的论文](https://transformer-circuits.pub/2023/monosemantic-features/index.html) 以及关于 [扩展 SAE](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) 的后续研究，以了解 SAE 的最新进展。
   - 他们讨论了 SAE 符号的相关性，并分享了包括一个跟踪这些技术格局的 [Google 文档](https://docs.google.com/document/d/1lHvRXJsbi41bNGZ_znGN7DmlLXITXyWyISan7Qx2y6s/edit#heading=h.j9b3g3x1o1z4)在内的资源。
- **lm-eval-harness 见解与用法**：一位用户询问了如何将 **lm-eval-harness** 用于自定义模型，并收到了一个有用的链接，指向一个用于适配 Huggingface 模型类的 [自包含示例](https://github.com/state-spaces/mamba/blob/main/evals/lm_harness_eval.py)。
   - 讨论强调了包含 **BOS** 等特殊 Token，以及从评估结果的 JSON 输出中提取基准测试名称的过程。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **管理 GPU 内存问题**：一位用户报告在使用 **32GB RAM** 的机器运行 **aya** 和 **nomic-embed-text** 等模型时出现 **out-of-memory errors**。有人建议切换到 **CPU**，但这导致性能大幅下降。
   - 这场讨论突显了工程师在处理内存限制时面临的性能权衡，以及优化 GPU 资源的挑战。
- **LangGraph 课程推荐**：用户讨论了各种 **LangGraph 课程**，推荐 **DeepLearning AI 课程** 作为一个可靠的选择，以及 **Udemy** 上的一个进阶课程。普遍感受是目前存在许多适合初学者的资源，但缺乏进阶材料。
   - 这表明 LangGraph 生态系统需要为希望深化技能的从业者提供更多高水平的综合培训。
- **SQL 聊天 Agent 协作**：一位用户寻求开发 **SQL 聊天 Agent 脚本** 的帮助，引发了另一位经验丰富的开发者的协作努力。双方分享了脚本和反馈，展示了社区的支持。
   - 这种互动体现了开发者之间的协作文化，强调通过知识共享来改进 AI 功能。
- **新音乐发现应用发布**：**mood2music** 应用推出，承诺根据用户情绪提供 AI 驱动的音乐推荐。目前该应用正在建立候补名单，并具有独特的音乐策划功能。
   - 该应用在准备发布之际引发了关注，确定了与音乐爱好者互动的潜力。
- **AgentGenesis 助力 AI 开发**：一位成员分享了 **AgentGenesis**，这是一个提供 **copy-paste 代码片段** 的库，用于加速 **Gen AI 应用** 的开发。它旨在提供一个对开发者友好的代码库，从而大幅提高生产力。
   - 该项目邀请社区贡献，旨在简化开发流程，展示了 AI 开发者社区的协作精神。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **John Schulman 离开 OpenAI 加入 Anthropic**：John Schulman 宣布离开 OpenAI，专注于 [Anthropic](https://x.com/johnschulman2/status/1820610863499509855) 的 AI 对齐研究，并表示渴望从事一线技术工作。
   - 他强调这一选择是个人行为，并指出尽管他离开了，但这反映了 OpenAI 对对齐工作的持续支持。
- **GDB 的休假引发猜测**：GDB 决定休假到年底，这引发了对其原因的讨论，包括对过度劳累和健康问题的担忧。
   - 有人推测，在专注于 AGI 开发的紧张几年后，这次休息可能是必不可少的。
- **关于 AI 对齐观点的激烈辩论**：一场关于 AI 对齐不同观点的深入讨论展开，Schulman 倾向于强化学习方法，而其他人则认为这超越了传统方法。
   - 这反映了对于控制超人类 AI 的更广泛担忧，以及对齐从根本上是否是一个深度学习问题的争论。
- **结构化输出革新 API 处理**：最近推出的 [Structured Outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) 允许开发者在不丢失键的情况下实现一致的 schema 匹配。
   - 此外，通过切换到 gpt-4o-2024-08-06 模型，开发者可以节省 **50% 的输入成本** 和 **33% 的输出成本**。
- **DALL·E 面临日益激烈的竞争**：随着新竞争对手的加入，关于 **DALL·E 是否仍保持** 最佳图像生成地位的讨论随之而起，直接对比面临挑战。
   - 成员们指出，在评估竞争能力时，上下文比直觉更重要。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **GPT-4o-2024-08-06 现已上线**：[GPT-4o-2024-08-06](https://openrouter.ai/models/openai/gpt-4o-2024-08-06) 的发布标志着一次显著更新，其输入价格**大幅降低了 50%**，输出价格降低了 **33%**，进一步提升了开发者的可访问性。
   - 值得注意的是，该模型包含一个新的 'refusal' 字段功能，引发了对其功能改进的期待。
- **Gemini Pro 1.5 遭遇资源限制**：用户在使用 **Gemini Pro 1.5** 时遇到了“资源已耗尽 (Resource has been exhausted)”的错误，这与 Google 强制执行的严格速率限制（rate limits）有关。
   - 不幸的是，目前没有补救措施，因为这是直接来自 Google 的限制。
- **导航 OpenRouter API**：关于模型购买的咨询表明，通过 **OpenRouter** 使用模型需要按 token 使用量付费，建议新用户尝试使用 **Lobe Chat** 等界面以获得更简便的交互。
   - 这种方法旨在简化访问并减少用户入门的阻力。
- **Structured Outputs 提升 API 可靠性**：OpenAI 推出了 Structured Outputs，允许开发者直接从 API 请求有效的 JSON 响应，增强了整体可靠性和可用性。
   - 这一举措解决了之前输出格式不一致的问题，旨在实现跨应用程序的更标准化的交互。
- **模型定价波动审查中**：关于 **gpt-4o-2024-08-06** 的 **token limit** 差异的讨论浮出水面，OpenRouter 界面显示的上限低于 OpenAI 的文档说明。
   - 用户正在等待更新，以使系统能力与最新的模型规范准确对齐。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **参加 CodiumAI 关于 RAG 增强编码的网络研讨会**：分享了即将举行的 [CodiumAI 网络研讨会](https://lu.ma/ka5xtyqo) 的提醒，重点是 **RAG 增强的编码助手**。参与者必须通过钱包验证 token 所有权才能参加活动。
   - 研讨会将涵盖 **Retrieval-Augmented Generation (RAG)** 如何提高 AI 生成代码的上下文感知能力，这对于保持软件开发的**高质量**至关重要。
- **使用 RabbitMQ 构建多智能体系统**：一篇博客重点介绍了如何使用 [RabbitMQ](https://www.rabbitmq.com) 创建本地多智能体系统（multi-agent system），并通过 llama-agents 利用 [ollama](https://ollama.com) 和 [qdrant_engine](https://qdrant.tech) 等工具。查看完整指南[请点击这里](https://t.co/IOGpDWkY8A)。
   - 这种设置促进了 Agent 之间的通信，并增强了构建健壮 AI 系统所必需的开发体验。
- **使用 HuggingFace Inference API 获取 Embeddings**：HuggingFace Inference API 允许使用 `TextEmbeddingsInference` 类生成 embedding，详见[此示例](https://docs.llamaindex.ai/en/stable/examples/embeddings/text_embedding_inference/)。它支持模型名称和 embedding 批处理大小等参数以优化性能。
   - 用户强调了它为处理 embedding 带来的效率，这对于训练 AI 模型至关重要。
- **分享 RAG 性能见解**：讨论包括关于 **Retrieval-Augmented Generation** 如何基于**上下文感知**提高生成代码质量的见解。关于使用 **LlamaIndex** 基础设施的高级方法的演示涵盖了实际应用。
   - 与会者可以期待学习*上下文感知生成 (context-aware generation)*，这对于希望改进编码助手的开发者来说至关重要。
- **Llamaparse 的阿拉伯语解析问题**：用户报告称 Llamaparse 在阿拉伯语解析方面存在困难，尽管阿拉伯语是“从右到左”的性质，但生成的却是“从左到右”格式的结果。这引发了关于 Llamaparse 处理语言复杂性能力的重要疑问。
   - 这一反馈标志着在解析应用中适应多种语言方面的一个潜在改进领域。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **LLM 幻觉指数引发关注**：[LLM Hallucination Index](https://www.rungalileo.io/hallucinationindex?utm_medium=paid&utm_source=alpha_signal&utm_campaign=sp) 评估模型对上下文的忠实度，因其被命名为**年度词汇**而备受瞩目。
   - 成员们讨论了该指数对 **Command R Plus** 的准确性，认为其误导了其开源状态。
- **开源定义引发辩论**：关于幻觉指数中开源定义的**分歧**，认为仅发布权重就定义为开源过于宽松。
   - 强调*数据集和训练方法的额外透明度*对于真正的开源状态至关重要。
- **Mistral 的许可证受到审视**：成员们澄清 **Mistral** 模型采用 **Apache 2.0** 许可证，使其符合开源标准，尽管在数据集访问方面存在限制。
   - 讨论显示，许多模型被标记为“开放权重 (open weights)”，但缺乏真正的开源特性。
- **Command R Plus 的商业用途争议**：**Command R Plus** 在知识共享署名-非商业性使用许可 (Creative Commons Attribution Non Commercial license) 下运行，使其在实际上属于闭源。
   - 论文中的开源定义受到审查，成员们主张建立更清晰的标准。
- **Cohere Toolkit 助力学习项目**：**Cohere Toolkit** 被用于一个 AI 研究员项目的学习计划，重点是在**食谱**和**法律案例笔记**等多样化语料库上构建**带有 RAG 的 LLM**。
   - 出现了关于从 **Cohere 模型**迁移到第三方 API（如 **OpenAI Chat GPT** 或 **Gemini 1.5**）的咨询，暗示了更广泛的功能需求。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **InlineList 定义新方向**：`InlineList` 目前缺少 **__moveinit__** 和 **__copyinit__** 功能，但相关进展正在进行中，关键特性即将合并。
   - 成员们将这些开发视为改进核心功能的首要任务。
- **澄清 Mojo 类型：List vs. InlinedFixedVector**：`InlinedFixedVector` 专为 **AnyTrivialRegType** 设计，而 `List` 则服务于 **CollectionElement**，突显了它们在 Mojo 中各自的用途。
   - 讨论涉及正在审查的**小缓冲区优化 (small buffer optimization)**，这可能会提升 `List` 的性能。
- **Mojo 与自定义硬件：一个加速话题**：成员们辩论了在 Mojo 中使用 PCIe 卡等**自定义加速器**的潜力，并质疑在开源发布前是否提供支持。
   - 对性能的担忧强调了有效硬件集成对 **cxl.mem** 的依赖。
- **FPGA 与 CXL IP 模块：硬件开发见解**：讨论涵盖了 **Xilinx VU13P FPGAs** 的使用以及为硬件优化项目集成 **CXL IP 模块**。
   - 一位成员分享了用自定义解决方案替换内核使用的计划，以提高整体效率。
- **对 Mojo 开源未来的期待升温**：人们对 Mojo 作为开源项目的未来感到兴奋，特别是关于对 **RISC-V 向量扩展 (vector extensions)** 的支持。
   - 成员们表达了希望 Mojo 能够显著贡献于他们的项目，尽管目前存在兼容性限制。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **John Schulman 离开 OpenAI 加入 Anthropic**：OpenAI 联合创始人 [John Schulman](https://www.cnbc.com/2024/08/06/openai-co-founder-john-schulman-says-he-will-join-rival-anthropic.html) 在 OpenAI 的 **superalignment team** 解散后，将加入由 **Amazon** 支持的 AI 初创公司 **Anthropic**。
   - 这一转变可能反映了在不断变化的环境中，确保对先进 AI 系统进行控制的持续担忧。
- **开源 AI 面临高昂成本挑战**：**open-source AI** 领域面临重大挑战，特别是 SOTA 模型的高昂训练成本以及获取必要偏好数据的难度。
   - 这些问题导致了竞争性开源模型开发的瓶颈。
- **Meta 的 JASCO 受到质疑**：关于 **Meta JASCO** 的猜测激增，有报道称其“失踪”并可能面临来自 **Udio** 和 **Suno** 的诉讼。
   - 随着社区中不确定性的蔓延，这一传闻可能会阻碍 Meta 的 AI 进展。
- **Doxxing 事件引发隐私担忧**：**Nullbulge** 遭遇了 doxxing（人肉搜索）事件，引发了关于在线隐私和个人声誉风险的讨论。
   - 社区成员指出，操作安全（operational security）中存在的潜在弱点可能有助于减轻未来风险。
- **模型在 270k 参数处遇到准确率瓶颈**：据报道，**270k model** 遇到了准确率平台期，仅达到 **84% 的验证准确率**，这标志着增加参数带来的收益递减。
   - 一位参与者建议，这一趋势表明模型设计需要替代策略。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinygrad 在 Aurora 上的可行性**：成员们讨论了在 **Aurora** 上运行 **tinygrad** 的可行性，因为其依赖 Intel GPU，并强调了对 **A770s** 等张量核心指令的支持。
   - 讨论涉及对 **Aurora 能力** 的预期，其预计将超过 **2 ExaFLOPS**，使其有可能成为史上最快的计算机。
- **张量的预分配技术**：一名成员建议预分配张量并分配切片（slices）可能解决张量操作问题，*George* 确认连续性（contiguity）可以解决该问题。
   - 将 `Buffer` 实例映射回 `DEFINE_GLOBAL` 突显了对清晰度的需求，因为像 *Eigenvector42* 这样的成员对张量流（tensor flow）表示了不确定性。
- **对分布式计算功能的需求**：成员们强调，**tinygrad** 需要成熟的 **distributed computing** 功能才能充分发挥 **Aurora** 的能力。
   - 他们强调，增强这些功能对于更好地利用 Aurora 的计算能力至关重要。
- **FP8 NVIDIA 赏金任务需要双重支持**：有人询问 FP8 NVIDIA 赏金任务是否需要支持 **E4M3** 或 **E5M2**，或者两者都需要，*George* 对两者都给予了积极回应。
   - 这表明了未来开发的一个关键领域，以及对 NVIDIA 要求的支持。
- **OpenMP 线程见解**：关于 **CLANG** 和 **LLVM** 线程的讨论确认了目前主要在单线程上使用，并提到了通过 **OpenMP** 进行增强的可能性。
   - 共享了相关 *tinygrad* GitHub pull requests 的链接，以激励贡献和改进。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Wiseflow 彻底改变了信息挖掘**：[Wiseflow](https://github.com/TeamWiseFlow/wiseflow) 是一款新型敏捷信息挖掘工具，可从多种来源提取并分类简洁信息，增强数据组织。
   - 这一创新工具旨在为信息密集型环境提供最佳检索，满足当前用户需求。
- **HybridAGI 引入神经符号增强**：[HybridAGI](https://github.com/SynaLinks/HybridAGI) 的最新版本整合了以图（graphs）为中心的神经符号系统，改进了 RAG (Retrieval-Augmented Generation) 功能。
   - 主要特性包括旨在简化易用性和增强数据处理流水线的各种 Notebook。
- **LLM 通过 Agent 向 AGI 演进**：关于将 **LLM** 转型为**基于 LLM 的 Agent** 的研究正在进行中，旨在解决这篇 [研究](https://arxiv.org/abs/2408.02479) 中强调的自主性局限。
   - 这强调了建立统一标准来评估作为 Agent 的 LLM 解决方案的必要性。
- **通过推理计算提升性能**：最近的一项研究表明，在推理过程中增加生成的样本数量可以提高性能，正如在 [SWE-bench Lite](https://arxiv.org/abs/2407.21787) 中所见，问题解决率从 **15.9%** 提升到了 **56%**。
   - 样本覆盖率与性能之间的这种关系对于代码编写和形式化证明特别有利。
- **MIPRO 通常优于 BootstrapFewShotWithRandomSearch**：针对相关咨询，有人指出 **MIPRO** 的表现“经常，但不一定”优于 **BootstrapFewShotWithRandomSearch**。
   - 这指出了 MIPRO 的强劲性能，同时也承认了其变数。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **合成数据生成策略**：一名成员询问了关于**合成数据生成**的策略，以增强 **8b 模型**在推理任务上的表现，特别是使用 **Chain of Thought (CoT)** 训练的 **text to SQL**。
   - 他们建议在生成 SQL 查询之前利用合成指令，以潜在地提高模型性能。
- **针对 Gemma 2 27b 的 QLoRA 配置**：讨论集中在针对 **Gemma 2 27b** 的 **QLoRA**，并建议调整 **learning rate** 以兼容 **Flash Attention**。
   - 成员们分享了尝试这些修改的意图，这可能有利于训练。
- **微调上下文长度（Context Length）的见解**：一名成员询问在将 **llama2-13b-hf** 等微调模型的上下文长度设置为 **4k** 后，是否还能进行调整。
   - 另一名成员确认可以增加或减少，并建议对于大幅度调整采用逐步方法以保持性能。
- **用于快速调整的 RoPE Scaling**：关于上下文长度话题，有人建议使用 **RoPE scaling** 进行高效调整。
   - 建议逐渐增加上下文长度以获得最佳效果，特别是对于重大变更。
- **BitsAndBytes GitHub Pull Request 提及**：一名成员强调要关注 **BitsAndBytes GitHub** 上的正确分支，特别提到了 Pull Request **#1220**。
   - 这一细节对于任何参与近期开发或调试的人员来说都至关重要。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **PPO 训练 Recipe 已添加到 Torchtune**：一个端到端的 **PPO 训练 Recipe** 已集成到 Torchtune 中，实现了 **RLHF** 功能。查看详细实现 [点击这里](https://github.com/pytorch/torchtune/pull/1005)。
   - *这一补充简化了强化学习与 Torchtune 工具包之间的集成，* 增强了训练选项。
- **现在支持 Qwen2 模型**：对 **Qwen2 模型**（包括 **7B** 模型）的支持已集成到 Torchtune 的训练 Recipe 中，**1.5B** 和 **0.5B** 模型也即将发布。更多详情见 [这里](https://github.com/pytorch/torchtune/pull/1143)。
   - *这一扩展为社区内的模型实验和微调开辟了更多可能性。*
- **计划为 Llama 3 提供 DPO 支持**：成员们讨论了为 **Llama 3 8B full finetune** 提供 **DPO** 支持的可能性，并对功能增强表示了兴趣。即使没有预构建的配置，任何模型都可以与这些 Recipe 配合使用。
   - *这表明正在持续努力探索更深层次的模型能力。*
- **重构后的 PreferenceDataset 增强了聊天支持**：新重构的 **PreferenceDataset** 现在支持聊天功能，详见 [Pull Request #1276](https://github.com/pytorch/torchtune/pull/1276)。这与之前讨论中建立的统一 **message_transform** 流水线保持一致。
   - *此次更新似乎显著改善了用户与数据集的交互体验。*
- **关于专用模型构建者页面的提案**：一位成员建议为每个模型的构建者（builders）创建一个**专用页面**，以适应不断增加的模型和**多模态 LLM**。*这将使我们能够更好地解释诸如模型下载和配置等重复性细节，* 为用户整合信息。
   - *该提案强调了社区对模型管理中更清晰组织工具的需求。*

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 设置故障排除**：用户报告了设置 **Open Interpreter** 时的问题，特别是在选择本地 Llama 模型时，执行过程中经常遇到 **openai.APIConnectionError**。
   - *一位用户报告称，即使在选择之后，他们的模型仍尝试再次下载。*
- **关于 Open Interpreter 安全措施的咨询**：一位成员对 **Open Interpreter** 如何处理用户数据提出了担忧，特别是数据是否保留在本地机器上。
   - 他们询问了端到端加密标准以及通信过程中是否有任何第三方参与。
- **Open Interpreter 的 Python 兼容性**：一位成员询问 **Open Interpreter** 是否支持 **Python 3.12**，并表示自己是编程初学者。
   - 另一位成员澄清说，目前的兼容性需要 **Python 3.10** 或 **3.11**。
- **Ollama 模型列表命令**：为了探索可用模型，一位成员建议使用命令 `ollama list`，并指出每个模型都有特定的 **VRAM** 需求。
   - 运行模型的指令详见 [Ollama 文档](https://github.com/OpenInterpreter/open-interpreter/blob/main/docs/language-models/local-models/ollama.mdx)，重点强调了资源可用性。
- **远程托管模型的 API Key**：已确认访问付费的远程托管模型必须使用 **API Key**，而本地模型则在指定的 **port** 上运行。
   - 这突显了远程功能中身份验证的重要性。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Llamafile 取得重大里程碑**：团队继续推进 **Llamafile**，在单个文件中提供离线、可访问的 LLM，这让社区成员感到非常兴奋。
   - *社区成员对该项目在可访问性方面的潜在影响表示兴奋。*
- **Mozilla AI 社区征求反馈以换取奖励**：**Mozilla AI 社区**通过一项调查寻求建议，参与者有机会赢取 **$25 礼品卡**。
   - *鼓励成员分享 Mozilla AI 如何通过社区资源更好地支持他们。*
- **庆祝 sqlite-vec 发布派对**：诚邀大家参加 [sqlite-vec 发布派对](https://discord.com/events/1089876418936180786/1265715263999836210)，届时将有核心维护者主持的演示。
   - *参与者将有机会尝试演示并直接与核心团队交流，* 增强他们的动手实践经验。
- **机器学习论文研讨会 (Machine Learning Paper Talks) 中的精彩讨论**：即将举行的 **Machine Learning Paper Talks** 将涵盖 *Communicative Agents* 和 *Extended Mind Transformers*，由一位知名的社区成员主持。
   - *这些环节有望让参与者了解最新的研究并进行充满活力的讨论。*
- **Local AI AMA 的见解**：一场与 Local AI 核心维护者的 [AMA](https://discord.com/events/1089876418936180786/1268967945216721079) 即将举行，讨论自托管替代方案。
   - *这是成员提问并探索 Local AI 实际实现的绝佳机会。*

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **LinkedIn 工程团队重构其 ML 平台**：LinkedIn 正在举办一场直播活动，详细介绍其工程团队对 **ML 平台**的改造及其内部创新。你可以点击[此处](https://www.linkedin.com/events/flytepipelinesinactionwithlinke7218669945767776256/theater/)加入讨论。
   - 该活动强调了对 **machine learning** 最新进展的见解，鼓励参与者在讨论中积极参与并分享想法。
- **直播活动带来实时见解**：活动正在进行中，揭示了 LinkedIn 在 **machine learning** 方面的关键进展，展示了其工程团队使用的策略和技术。
   - 参与者可以积极贡献，为那些对该领域前沿实践感兴趣的人提供了一个协作场所。

---

**Alignment Lab AI Discord** 没有新消息。如果该服务器长期沉寂，请告知我们，我们将予以移除。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该服务器长期沉寂，请告知我们，我们将予以移除。

---

**DiscoResearch Discord** 没有新消息。如果该服务器长期沉寂，请告知我们，我们将予以移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器长期沉寂，请告知我们，我们将予以移除。

---

# 第 2 部分：按频道详细摘要和链接

{% if medium == 'web' %}

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1270099062342947013)** (459 messages🔥🔥🔥): 

> - `Stable Diffusion 中的 LoRA 使用`
> - `用于线稿的 Pony 模型`
> - `ControlNet 与图像转换`
> - `r/stablediffusion 的社区动态`
> - `AI 硬件选择`

- **LoRA 在 Stable Diffusion 中的用法**：LoRA 模型是小型 Stable Diffusion 模型，用于对标准 Checkpoint 模型进行修改，使其体积更小且更易于管理。
   - 用户可以将 LoRA 模型安装在 `stable-diffusion-webui/models/Lora` 目录下，并在提示词（Prompts）中使用 `<lora:filename:1.0>` 语法来包含它们。
- **用于线稿的 Pony 模型**：Pony 模型专为生成无阴影的干净线稿而设计，可以与风格 LoRA 配合使用以增强输出效果。
   - 用户讨论了在应用线稿风格 LoRA 时，需要使用 Pony 模型作为基础模型才能达到理想效果。
- **ControlNet 与图像转换**：ControlNet 可用于将照片转换为线稿等任务，它有助于保持原始图像的结构。
   - 用户建议了多种方法，如使用深度 ControlNet 或 IPAdapter，在 Stable Diffusion 中实现有效的图像转换。
- **r/stablediffusion 的社区动态**：用户反思了 r/stablediffusion 子版块的管理变动和之前的争议，这些争议曾因控制权问题引发社区骚乱。
   - 对话强调了 AI 艺术领域中社区主导项目与公司主导计划之间持续存在的动态关系。
- **AI 硬件选择**：关于硬件偏好的讨论显示出对 AMD GPU 处理机器学习任务的普遍质疑，建议倾向于 NVIDIA 或 Groq 等替代方案。
   - 硬件股票和技术的波动引发了关于未来 AI 性能选择的讨论。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.krea.ai/apps/image/realtime">KREA</a>: 未找到描述</li><li><a href="https://www.xkcd.com/2347/">Dependency</a>: 未找到描述</li><li><a href="https://www.stablediffusiontutorials.com/2024/08/flux-installation.html?m=1">FLUX: Installation with Workflow is Here</a>: 未找到描述</li><li><a href="https://x.com/SomniumSpace/status/1820930960239497445">来自 Somnium Space (@SomniumSpace) 的推文</a>: 我们很高兴发布 Robert Scoble (@Scobleizer) 在 #SomniumConnect2024✨ 上发表的精彩主题演讲。未来 10 年 #AI 将给人类带来什么？这将如何...</li><li><a href="https://huggingface.co/black-forest-labs">black-forest-labs (Black Forest Labs)</a>: 未找到描述</li><li><a href="https://comfyanonymous.github.io/ComfyUI_examples/flux/">Flux Examples</a>: ComfyUI 工作流示例</li><li><a href="https://huggingface.co/THUDM/CogVideoX-2b">THUDM/CogVideoX-2b · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=sMMYSmDHAY8">ComfyUI: Imposing Consistent Light (IC-Light Workflow Tutorial)</a>: 该视频专注于在 ComfyUI 中实现 IC-Light，特别是针对产品摄影。IC-Light 基于 SD1.5，我们使用参考背景和...</li><li><a href="https://x.com/0xkarmatic/status/1820618875517685976">来自 Karma (@0xkarmatic) 的推文</a>: 哇，Greg 也要休假了。</li><li><a href="https://old.reddit.com/r/StableDiffusion/comments/1ekolfd/cfg_how_it_works_in_nonflux_models_vs_flux_code/```">CFG: how it works in non-Flux models vs Flux (code examples)</a>: Flux 的“guidance”值是一个输入到模型中的简单数值。BFL 在蒸馏阶段通过生成一个...</li><li><a href="https://github.com/vosen/ZLUDA">GitHub - vosen/ZLUDA: CUDA on ??? GPUs</a>: 在 ??? GPU 上运行 CUDA。通过在 GitHub 上创建账号为 vosen/ZLUDA 的开发做出贡献。</li><li><a href="https://civitai.com/models/596934/line-art-style-sdxl-pony">Line Art Style [SDXL Pony] - V1 | Stable Diffusion LoRA | Civitai</a>: LINE ART STYLE。这是一个旨在模仿线稿的风格 LoRA，特别是几乎没有阴影/暗部的艺术，以获得干净的黑色线条...</li><li><a href="https://www.youtube.com/watch?v=_kctwd4w7R0">Good Vibrations (Official Music Video)</a>: 高清重制版！由 Marky Mark and The Funky Bunch 演唱的 Good Vibrations 官方音乐视频。#MarkyMark #GoodVibrations #Remastered</li><li><a href="https://civitai.com/models/257749/pony-diffusion-v6-xl">Pony Diffusion V6 XL - V6 (start with this one) | Stable Diffusion Checkpoint | Civitai</a>: Pony Diffusion V6 是一款多功能的 SDXL 微调模型，能够生成各种兽人（anthro）、野性（feral）或类人（humanoids）物种的精彩 SFW 和 NSFW 视觉效果...</li><li><a href="https://stable-diffusion-art.com/lora/#Step_1_Install_the_LoRA_model">What are LoRA models and how to use them in AUTOMATIC1111 - Stable Diffusion Art</a>: LoRA 模型是小型 Stable Diffusion 模型，可对标准 Checkpoint 模型进行微小更改。它们通常比 Checkpoint 小 10 到 100 倍。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1270095045319459039)** (105 条消息🔥🔥): 

> - `Unsloth Fine-tuning Issues` (Unsloth 微调问题)
> - `Model Inference Timing` (模型推理耗时)
> - `Pretraining vs. Continued Pretraining` (Pretraining 与 Continued Pretraining 的区别)
> - `Multi-GPU Support Development` (多 GPU 支持开发进展)
> - `Resources for Learning LLM Inference` (学习 LLM 推理的资源)


- **Unsloth Fine-tuning Issues**: 用户在 Unsloth 中使用微调模型时遇到问题，例如模型无法正常保存，以及在集成到 PPO trainer 时面临挑战（现在需要 `for_inference()` 方法来生成输出）。
   - 社区成员指出，之前的版本与 PPO trainer 配合得更好，对新要求和功能变化感到沮丧。
- **Model Inference Timing**: 有报告称在微调后的 Llama3.1 上运行推理时响应时间不一致，初始加载时间较长，但在多次调用后会有所改善。
   - 建议用户进行测试，以验证这种暂时的缓慢是否确实是导致延迟的原因。
- **Pretraining vs. Continued Pretraining**: 澄清了 Pretraining（预训练）和 Continued Pretraining（持续预训练）之间的区别，社区承认这些术语存在混淆。
   - 这引发了关于在处理语言模型时理解这些概念重要性的讨论。
- **Multi-GPU Support Development**: Unsloth 的多 GPU 支持目前处于 Beta 测试阶段，计划在未来发布，届时将包含在降低 VRAM 占用和提升速度方面的增强功能。
   - 测试人员目前签署了 NDA（保密协议），该功能正在完善中，后续将作为付费订阅版本发布。
- **Resources for Learning LLM Inference**: 社区成员分享了一个生成式 AI 指南链接，其中包含高层级的总结，但指出缺乏关于推理的详细资源。
   - 用户对现有资源表示感谢，同时在寻求更多关于推理技术的具体信息。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=FqfebeAdT073">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://guide.repleteai.com">Nextra: the next docs builder</a>: Nextra：下一代文档构建工具</li><li><a href="https://colab.research.google.com/drive/164cg_O7SV7G8kZr_JXqLd6VC7pd86-1Z#scrollTo=PoPKQjga6obN.">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/kalomaze/Mistral-7b-MoEified-8x">kalomaze/Mistral-7b-MoEified-8x · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/OpenAIDevs/status/1820876430764634115">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>: 在 API 中引入结构化输出（Structured Outputs）——模型输出现在遵循开发者提供的 JSON Schemas。https://openai.com/index/introducing-structured-outputs-in-the-api/</li><li><a href="https://huggingface.co/collections/unsloth/load-4bit-models-4x-faster-659042e3a41c3cbad582e734">Load 4bit models 4x faster - a unsloth Collection</a>: 以 4 倍速加载 4bit 模型 - Unsloth 集合</li><li><a href="https://huggingface.co/collections/unsloth/4bit-instruct-models-6624b1c17fd76cbcf4d435c8">4bit Instruct Models - a unsloth Collection</a>: 4bit 指令模型 - Unsloth 集合</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1270306636795346975)** (10 条消息🔥): 

> - `BigLlama 3.1`
> - `Pokerole Pokémon Prompt`
> - `游戏讨论` 


- **介绍 BigLlama 3.1-1T-Instruct**：一位用户分享了 [BigLlama-3.1-1T-Instruct](https://huggingface.co/mlabonne/BigLlama-3.1-1T-Instruct) 的链接，这是一个使用 [mergekit](https://github.com/cg123/mergekit) 对 Meta-Llama 进行实验性自我合并（self-merge）的模型。该模型被定位为先前版本的继任者，尽管目前仍处于开发中，但其重点在于产出一个合理的模型。
   - 另一位用户强调，目前该模型在某种程度上是**无用的**，因为它尚未针对合并后的权重进行训练。
- **对 Pokerole Pokémon 游戏的兴奋**：一位用户分享了一个 [Pokerole Pokémon 提示词链接](https://www.rpgprompts.com/post/pokerole-pok%C3%A9mon-chatgpt-prompt)，它提供了一种与 AI Game Master 一起玩 Pokémon 的创意方式。该提示词通过捕捉、训练和对战 Pokémon，实现了引人入胜的游戏玩法，捕捉到了该系列的精髓。
   - 用户们表现出极大的热情，其中一人提到：*等等，这个真的很好用*，表明该游戏提示词受到了好评。
- **闲聊游戏对话**：围绕成员们玩游戏的讨论展开，提及了 **Minecraft** 和 **Pokémon**，引发了关于他们游戏体验的进一步对话。用户们互动愉快，在对话中开着关于游戏相关提示词和体验的玩笑。
   - 这种围绕游戏的轻松调侃反映了一个在游戏以及 AI 如何增强此类体验方面有着共同兴趣的社区。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/mlabonne/BigLlama-3.1-1T-Instruct">mlabonne/BigLlama-3.1-1T-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://www.rpgprompts.com/post/pokerole-pok%C3%A9mon-chatgpt-prompt">Pokémon RPG - ChatGPT 提示词 </a>: 此提示词调用了一个 AI 构建的 Game Master，引导你穿越充满活力和令人兴奋的 Pokémon 世界，灵感来自该系列粉丝所熟悉的充满冒险的地区。参与捕捉...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1270118569652260924)** (162 messages🔥🔥): 

> - `Llama Model Training`
> - `Colab Pro Limitations`
> - `GGUF File Usage`
> - `Ollama Integration`
> - `Model Exporting` 


- **Llama Model Training 讨论**：用户讨论了 Llama 模型训练的各个方面，重点关注微调（fine-tuning）过程以及与 Ollama 等平台的集成。
   - 许多人在运行微调后的模型时遇到挑战，并就成功执行所需的配置寻求建议。
- **遇到 Colab Pro 限制**：需要 Colab Pro 才能访问终端（terminal）功能是一个主要问题，因为许多用户打算与没有付费服务访问权限的人分享模型训练知识。
   - 终端被认为是运行 Ollama 相关命令所必需的，这给尝试利用免费资源的用户带来了困扰。
- **GGUF 文件转换过程**：几位用户询问了生成使用 Gpt4All 模型所需的 GGUF 文件的方法，表明他们仍在学习该过程。
   - 共享了关于在训练 Notebook 中何处查找 .gguf 文件的说明，以促进与 Gpt4All 的集成。
- **Ollama 集成与使用**：提供了关于如何使用 Ollama 提供服务并与模型交互的指导，强调在 Python 中使用 subprocess 命令进行执行。
   - 用户讨论了在本地提供模型服务并通过 curl 命令使用 REST API 进行查询的步骤，展示了模型交互的工作流。
- **导出已训练模型**：参与者探索了训练后导出模型的选项，特别是关于在本地环境中使用 Ollama 运行模型。
   - 建议包括在不需要 Colab Pro 的情况下提供模型服务并与其交互，重点是使用本地资源进行模型部署。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=FqfebeAdT073">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1aqlNQi7MMJbynFDyOQteD2t0yVfjb9Zh?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1T-YBVfnphoVc8E2E854qF3jdia2Ll2W2?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://www.runpod.io/serverless-gpu">AI 推理的 Serverless GPU 端点</a>: 使用 RunPod Serverless GPU 端点大规模运行机器学习推理。</li><li><a href="https://huggingface.co/docs/datasets/v2.20.0/loading#:~:text=full%20offline%20mode.-,Slice%20splits,-You%20can%20also>">Load</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1270381106402820237)** (1 messages): 

> - `LLaMA3 configuration`
> - `Cost-effective model running` 


- **寻求高性价比的 LLaMA3 配置**：一位成员请求关于在 RunPod 上以高性价比方式运行 **LLaMA3** 模型所需配置的建议。
   - 这一询问突显了社区对优化模型部署成本的持续关注。
- **LLaMA3 成本管理的挑战**：成员们讨论了在各种平台上运行 **LLaMA3** 等模型时平衡性能和成本的挑战。
   - 几位成员分享了过去的经验，即由于不可预见的资源需求，成本超出了预期。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/)** (1 messages): 

vvelo: https://fxtwitter.com/reach_vb/status/1820493688377643178
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1270493009590616085)** (1 messages): 

> - `Gemma 2 2B`
> - `Diffusers integration with FLUX`
> - `Argilla 2.0`
> - `Whisper Generations`
> - `llm-sagemaker Terraform Module`

- **Google 扩展了 Gemma 系列，推出了 Gemma 2 2B**：Google 推出了 [Gemma 2 2B](https://huggingface.co/collections/google/gemma-2-2b-release-66a20f3796a2ff2a7c76f98f)，新增了一个拥有 **2.6B 参数** 的模型用于端侧（on-device）使用，增强了现有的 Gemma 产品线。
   - 他们还推出了 **ShieldGemma**（一套安全分类器）和 **Gemma Scope**（一套用于增强功能的稀疏自编码器）。
- **令人兴奋的 Diffusers 与 FLUX 集成**：一位成员强调了新的 [Diffusers 对 FLUX 的集成](https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell)，称赞其在文本生成图像（text-to-image generation）方面的先进能力。
   - 他们提供了一个 [gist](https://gist.github.com/sayakpaul/b664605caf0aa3bf8585ab109dd5ac9c)，介绍了如何在资源有限的情况下运行 FLUX，提高了对更广泛受众的可访问性。
- **以数据为中心的工具 Argilla 2.0 发布**：[Argilla 2.0](https://huggingface.co/blog/dvilasuero/argilla-2-0) 作为一款为 AI 创作者打造的强大工具亮相，专注于改进数据管理和易用性。
   - 此外，社区还迎来了首个由 Llama 3.1 驱动的开放合成数据集，名为 **magpie-ultra-v0.1**，旨在提升数据集创建标准。
- **Whisper 生成速度提升 150%！**：通过使用 Medusa heads，Whisper 的生成速度现在提升了 **150%**，且几乎没有准确率下降。
   - 成员们对将 Medusa heads 与 ASR 系统集成的意义表示兴奋，强调了其未来的潜力。
- **llm-sagemaker Terraform 模块简化了部署**：新的 [llm-sagemaker](https://registry.terraform.io/modules/philschmid/llm-sagemaker/aws/latest) Terraform 模块允许将开源 LLM 直接部署到 AWS SageMaker，增强了生产能力。
   - 该模块支持 Llama 3 和 Mistral 等热门模型，并包含可定制的配置，使其对开发者非常友好。


<div class="linksMentioned">

<strong>提到的链接</strong>：

- [Google 发布 Gemma 2 2B, ShieldGemma 和 Gemma Scope](https://huggingface.co/blog/gemma-july-update#use-with-llamacpp)：未找到描述
- [Vaibhav (VB) Srivastav (@reach_vb) 的推文](https://x.com/reach_vb/status/1819023974283518223)：Gemma 2 2B 在浏览器中运行，由 WebLLM 和 WebGPU 驱动！🔥 100% 本地和端侧运行。在不到 24 小时内，我们已经将模型带到了边缘端！⚡ 在下方的 HF Space 中尝试一下：
- [Vaibhav (VB) Srivastav (@reach_vb) 的推文](https://x.com/reach_vb/status/1819469088890261748)：Gemma 2 2B 在免费的 Google Colab 中运行！🤗 由 Transformers 驱动！⚡
- [Georgi Gerganov (@ggerganov) 的推文](https://x.com/ggerganov/status/1818699785152397592)：上手最新 Gemma 2 模型 + llama.cpp 的简单指令：https://huggingface.co/blog/gemma-july-update#use-with-llamacpp
- [Sayak Paul (@RisingSayak) 的推文](https://x.com/RisingSayak/status/1819299449966833972)：你现在应该已经被 @bfl_ml 发布的 FLUX 惊艳到了。真是个了不起的模型，对吧！在与我的伙伴们 @_DhruvNair_、@YiYiMarz 和 @multimoda 冲刺之后，我回到了 Twitter...
- [Gabriel Martín Blázquez (@gabrielmbmb_) 的推文](https://x.com/gabrielmbmb_/status/1819398254867489001)：发布 magpie-ultra-v0.1，这是第一个使用 Llama 3.1 405B 构建的开源合成数据集。使用 distilabel 创建，这是我们迄今为止最先进且计算密集度最高的流水线。https://huggingfac...
- [Vaibhav (VB) Srivastav (@reach_vb) 的推文](https://x.com/reach_vb/status/1820560137892835369)：使用 Medusa heads 让 Whisper 生成速度提升 150%！🔥 基于 Transformers 构建，准确率下降极小。非常令人兴奋的研究领域，Medusa heads 已被证明对 LLM 来说速度极快...
- [merve (@mervenoyann) 的推文](https://x.com/mervenoyann/status/1818613425859145772)：已发布：@huggingface Transformers 文档中新增了 Vision Language Models 任务指南，并更新了 Depth Estimation 任务指南 ⛴️📦 👉🏻 阅读关于 VLM、如何进行流式传输、量化等内容...
- [Philipp Schmid (@_philschmid) 的推文](https://x.com/_philschmid/status/1820360144334496064)：很高兴宣布 “llm-sagemaker”，这是一个新的 Terraform 模块，可以轻松地将 @huggingface 的开源 LLM 部署到 @awscloud SageMaker 实时端点！👀 基础设施即代码 (IaC) 工具对于...
- [merve (@mervenoyann) 的推文](https://x.com/mervenoyann/status/1818675981634109701)：SAMv2 简直好得令人惊叹 😍 了解是什么让这个模型在视频分割方面如此出色，继续阅读 🦆⇓
- [Databricks Mosaic Research (@DbrxMosaicAI) 的推文](https://x.com/DbrxMosaicAI/status/1818407826852921833)：致我们的 StreamingDataset 用户：我们很高兴地宣布支持在 @huggingface 中存储 MDS 数据集。感谢 @orionweller 的贡献！在此查看文档：https://docs.mosaic...

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1270097797437460544)** (239 条消息🔥🔥): 

> - `Hugging Face Datasets 问题`
> - `学校经历`
> - `图像生成工具`
> - `论文与数据集处理`
> - `AI 学习资源` 


- **Hugging Face Datasets 问题解决**：一位用户报告了在加载包含多个 JSON lines 文件的数据集时遇到的问题，引发了关于潜在变通方法和硬编码特征以实现更好 schema 识别的讨论。
   - 其他用户分享了使用 Parquet 格式以实现更便捷访问和 schema 推断可靠性的见解，同时讨论了小文件分块的好处。
- **对学校的复杂心情**：一位成员在度过了挑战性的第一天后表示如释重负，并提到老师在课堂上讲了一些无关的个人故事。
   - 另一位用户对封锁期间的远程课程表示肯定，表达了对这种学习方式的偏好。
- **AI 图像生成探索**：一位用户发现他们的兄弟姐妹正在使用 Meta AI 的应用程序生成猫的图像，这引发了对这些工具影响的担忧。
   - 这引发了关于用户对使用 AI 进行创意任务的感受以及它如何影响社交动态的讨论。
- **论文项目讨论**：一位用户讨论了涉及数据集策划的论文项目，分享了保持数据集易于管理且对未来用户具有可扩展性的策略。
   - 他们提到了使数据集易于使用的重要性，强调了尽管只处理了数据集的一小部分，但他们仍专注于可用性。
- **AI 学习资源**：AI 领域的新手被引导至不同的学习模型资源，强调了熟悉 Hugging Face 工具和术语的重要性。
   - 成员们鼓励探索平台上现有的模型和数据集，以帮助理解和实际应用。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/learn">Hugging Face - Learn</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/fffiloni/audio-to-spectrogram">Audio To Spectrogram - fffiloni 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/docs/hub/repositories-recommendations#sharing-large-datasets-on-the-hub">仓库限制与建议</a>：未找到描述</li><li><a href="https://huggingface.co/THUDM/CogVideoX-2b">THUDM/CogVideoX-2b · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/learn/ml-for-3d-course">欢迎来到 🤗 3D 机器学习课程 - Hugging Face ML for 3D Course</a>：未找到描述</li><li><a href="https://huggingface.co/docs/datasets/v2.20.0/dataset_script#add-dataset-attributes)">创建数据集加载脚本</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/fffiloni/spectrogram-to-music">Riffusion • Spectrogram To Music - fffiloni 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://github.com/buaacyw/MeshAnythingV2">GitHub - buaacyw/MeshAnythingV2: 像人类艺术家一样将任何物体转换为网格。&quot;MeshAnything V2: Artist-Created Mesh Generation With Adjacent Mesh Tokenization&quot; 的官方实现</a>：buaacyw/MeshAnythingV2</li><li><a href="https://github.com/huggingface/datasets/issues/7092">使用多个 jsonlines 文件的 load_dataset 过早解释数据结构 · Issue #7092 · huggingface/datasets</a>：描述了可能与 #6460 相关的错误，使用 datasets.load_dataset(&quot;json&quot;, data_dir= ... ) 加载多个 .jsonl 文件时，如果其中一个文件（可能是第一个文件）包含完整的...</li><li><a href="https://github.com/SonyCSLParis/NeuralDrumMachine/tree/master">GitHub - SonyCSLParis/NeuralDrumMachine</a>：通过在 GitHub 上创建账号来为 SonyCSLParis/NeuralDrumMachine 的开发做出贡献。</li><li><a href="https://huggingface.co/models">Models - Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/docs/hub/en/spaces-overview">Spaces 概览</a>：未找到描述</li><li><a href="https://huggingface.co/spaces">Spaces - Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/launch">Spaces Launch – Hugging Face</a>：未找到描述</li><li><a href="https://github.com/huggingface/transformers/issues">Issues · huggingface/transformers</a>：🤗 Transformers: 为 Pytorch, TensorFlow, 和 JAX 提供的最先进机器学习库。 - Issues · huggingface/transformers
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1270265570016624671)** (3 条消息): 

> - `Linear Algebra for 3D Video Analysis`
> - `Sharing Resources` 


- **探索用于 3D Video Analysis 的 Linear Algebra**：一位成员表示有兴趣学习专门针对 **3D video analysis** 的 **linear algebra**。
   - 这突显了对处理和分析 3D 视觉数据所需的数学基础的浓厚兴趣。
- **征求 Linear Algebra 相关的博客**：另一位成员请求推荐与 linear algebra 相关的**高质量博客或文章**以供学习。
   - *分享有价值的资源*可以极大地提升任何深入研究复杂数学课题的人的学习体验。
- **呼吁分享资源**：一位成员在表达阅读感谢后，鼓励其他人传播所收集的资源。
   - 这反映了**知识共享**的社区精神，营造了一个可以自由交换信息的空间。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1270252132674703493)** (4 条消息): 

> - `Image Synthesis with Transformers`
> - `Integrating Graphs with LLMs` 


- **使用 Transformers 进行高分辨率图像合成**：讨论强调了使用 **transformers** 合成**高分辨率图像**，重点关注 **latent representations** 和**富上下文词汇表 (context-rich vocabulary)** codebook。
   - 强调了*条件图像合成 (Conditioned image synthesis)* 技术对于提升图像质量的价值。
- **将 Graphs 与 LLMs 集成的新方法**：一位成员分享了一个[方法链接](https://arxiv.org/pdf/2405.20684v1)，该方法将 **graphs** 与 **LLMs** 集成，并指出其与 ICML 提出的一种方法相似。
   - 这展示了语言模型*图集成 (graph integration)* 方法论中一个引人注目的进展。
- **探索另一种新颖的图集成方法**：另一位成员发布了[这篇论文](https://arxiv.org/pdf/2402.03973)的链接，讨论了将 **graphs** 与 **LLMs** 集成的其他策略。
   - 重点在于扩展工具包，以提高模型与图结构的兼容性。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1270228072410513409)** (5 条消息): 

> - `Unity ML-Agents Training`
> - `Embodied Agent Platform`
> - `Talking Head Synthesis`
> - `Bilateral Reference for Image Segmentation` 


- **支持多线程的 Unity ML-Agents 训练**：一位成员分享了一个 [YouTube 视频](https://youtube.com/live/XOFMpZsYeXo?feature=share)，展示了他们使用 Unity 6 ML-Agents 进行 SAC Agent 训练的第二部分，重点介绍了为 **CUDA** 训练增加的**多线程支持**。
   - 他们还提到引入了一个**厌倦冷却池 (boredom cooldown pot)**，以鼓励 Agent 在达到**特定厌倦阈值**后选择新的向量。
- **Embodied Agent 平台开发**：分享了一个 **Embodied Agent 平台**的项目页面，其特色是 Agent 可以在 3D 环境中聊天、理解指令并执行任务；请在 [GitHub](https://github.com/thunlp/LEGENT) 上查看。
   - [Hugging Face](https://huggingface.co/spaces/LEGENT/LEGENT) 上也提供了一个**在线 Demo**，以展示其功能。
- **通过 AniTalker 进行 Talking Head 合成**：分享了一个 [Talking Head 合成项目](https://huggingface.co/spaces/Delik/Anitalker)的链接，该项目是 **AniTalker** GitHub 仓库的移植版本，专注于为说话的面部制作动画。
   - 官方 **GitHub 仓库**可以在[这里](https://github.com/X-LANCE/AniTalker)找到，详细介绍了其在多样化面部运动编码中的应用。
- **用于图像分割的 BiRefNet**：一位成员强调了他们参与开源的 [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet)，该网络专为**高分辨率二分图像分割 (dichotomous image segmentation)**而设计，展示了其优于 **RMBG1.4** 的性能。
   - 其他资源包括 [arXiv 论文](https://arxiv.org/pdf/2401.03407)链接和 Hugging Face Spaces 上的演示，强调了其 **SOTA 能力**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/ZhengPeng7/BiRefNet">ZhengPeng7/BiRefNet · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/thunlp/LEGENT">GitHub - thunlp/LEGENT: Open Platform for Embodied Agents</a>：Embodied Agent 开放平台。通过在 GitHub 上创建账号为 thunlp/LEGENT 的开发做出贡献。</li><li><a href="https://huggingface.co/spaces/LEGENT/LEGENT">LEGENT - a Hugging Face Space by LEGENT</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/Delik/Anitalker">Anitalker - a Hugging Face Space by Delik</a>：未找到描述</li><li><a href="https://github.com/X-LANCE/AniTalker">GitHub - X-LANCE/AniTalker: [ACM MM 2024] This is the official code for &quot;AniTalker: Animate Vivid and Diverse Talking Faces through Identity-Decoupled Facial Motion Encoding&quot;</a>：[ACM MM 2024] 这是 &quot;AniTalker: Animate Vivid and Diverse Talking Faces through Identity-Decoupled Facial Motion Encoding&quot; 的官方代码 - X-LANCE/AniTalker</li><li><a href="https://youtube.com/live/XOFMpZsYeXo?feature=share">Unity ML-Agents | Live Agent training from Scratch | Part 2</a>：在 3D 体素世界中的快速 SAC Agent 训练器
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1270468951608131585)** (5 条消息): 

> - `OpenAI's Structured Outputs`
> - `LLMs and Reasoning`
> - `Attention Mechanisms in LLMs` 


- **OpenAI 推广 Structured Outputs**：OpenAI 发布了一篇 [博客文章](https://openai.com/index/introducing-structured-outputs-in-the-api/)，建议在其 API 中使用结构化输出，但并未过多提及之前的相关工作。
   - 这一转变凸显了在采用有效实践的同时，对基础性贡献缺乏认可的趋势。
- **LLM 在真实推理方面表现挣扎**：一位成员讨论了他们的观点，即 LLM 并不像人类那样真正地进行“推理”，认为其推理方式可能会使检索任务变得复杂。
   - 这一比喻将 LLM 的交互比作 Uber 司机在熟悉路线上行驶，而非在点对点之间瞬间移动。
- **Draft Tokens 作为 LLM 的 Scratchpads**：一种理论认为 LLM 需要 Token 作为推理的“草稿纸 (scratchpads)”，并提出引入 Draft Tokens 可以有效提高其推理能力。
   - 这一见解与一篇论文有关，该论文通过在 Prompt 前缀添加额外的 Token 来提升性能，从而增强了内存存储能力。
- **Attention 机制与 KV-cache**：有一种观点认为，用外部数据库替换线性层可以增强 LLM 的推理能力，但最近的测试表明，放弃 KV-cache 会降低性能。
   - 这强调了 KV-cache 在维持 LLM 任务推理有效性方面的关键作用。
- **扩展 LLM 中的推理步骤**：为了促进 LLM 内部的推理，扩展 Token 限制被视为比增加模型深度（需要重新训练）更简单的解决方案。
   - 这种方法表明，添加更多 Draft Tokens 可以允许额外的转换步骤，而无需复杂的模型修改。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1270265971914964993)** (4 条消息): 

> - `Depth Estimation`
> - `CVPR 2022 Papers`
> - `Code Implementations` 


- **来自 CVPR 2022 的深度估计论文**：一位成员分享了发表在 CVPR 2022 上名为 *Depth Estimation by Combining Binocular Stereo and Monocular Structured-Light* 的论文链接，可以在 [这里](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Depth_Estimation_by_Combining_Binocular_Stereo_and_Monocular_Structured-Light_CVPR_2022_paper.pdf) 找到。
   - 该论文可能为结合不同方法的深度估计先进技术提供见解。
- **咨询代码实现**：一位成员询问上述深度估计论文是否有可用的代码实现。
   - 针对该查询，目前尚未分享具体的代码资源或链接。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1270340412208578560)** (2 条消息): 

> - `NER Annotated CV Dataset`
> - `Identifying Relevant JSON Files` 


- **命名实体识别 (NER) 数据集可用**：一位成员分享了一个包含 **5029 份已标注简历** 的数据集，其中使用 NER 标记了 IT 技能，可在 [Kaggle](https://www.kaggle.com/datasets/mehyarmlaweh/ner-annotated-cvs) 上获取。
   - 该数据集包含从 PDF 中**手动标注的技能**，并采用 **JSON** 格式，以便与 **Spacy** 等 NLP 工具配合使用。
- **为问题寻找相关的 JSON 文件**：另一位成员描述了一个拥有超过 **20,000 个 JSON 文件** 的固定数据集，并正在寻找最相关的 **5 个文件 ID**，以回答从其他文件中生成的问题。
   - 他们使用了 Elastic Search 的**关键词搜索**和**语义搜索**，以及 **s-bert embedding 模型**，目前正在寻求关于优化搜索的最佳方法的建议。



**提到的链接**：<a href="https://www.kaggle.com/datasets/mehyarmlaweh/ner-annotated-cvs">NER Annotated CVs</a>：该数据集包含 5029 份已标注的个人简历 (CV)，并标记了 IT 技能。

  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1270099360310628557)** (157 条消息🔥🔥): 

> - `在 AnythingLLM 中使用 LM Studio`
> - `模型性能与设置`
> - `音频转录功能`
> - `多 GPU 配置`
> - `Phi-3 模型支持问题` 


- **用户配置 LM Studio 与 AnythingLLM**: 用户讨论了如何将 AnythingLLM 与 LM Studio 配合使用，并指出了影响性能的文件访问权限和硬件限制问题。
   - 经过排查，一位用户在加载自定义 Gemma v2 模型后确认配置成功。
- **性能设置与优化**: 讨论集中在 “Keep Model in Memory” 功能上，一些用户建议该功能可能不会显著影响性能，应默认禁用。
   - 专家们就该功能的实用性发表了看法，特别是针对大模型在 RAM 使用方面的影响。
- **LM Studio 中的音频转录功能**: 用户对使用 LM Studio 自动化音频转录表现出兴趣，但明确了目前缺乏对音频输入的直接支持。
   - 讨论的替代方案包括使用某些 API 和 TTS/STT 解决方案，尽管许多人出于隐私考虑更倾向于离线和开源选项。
- **ComfyUI 中的多 GPU 配置**: 有关于如何在 ComfyUI 中利用多 GPU 的咨询，用户探索了各种脚本和设置来有效管理 GPU 资源。
   - 一位用户建议创建一个启动器来设置 CUDA 设备，从而在无需修改配置文件的情况下增强工作流。
- **关于 Phi-3 模型支持的担忧**: 一位用户对 llama.cpp 缺乏对 Phi-3 模型支持及其在更新后对 Oobabooga WebUI 等其他界面的影响表示担忧。
   - 这引发了关于模型支持变化以及社区对近期更新反应的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/text-generation-inference/en/conceptual/flash_attention">Flash Attention</a>: 未找到描述</li><li><a href="https://huggingface.co/legraphista/internlm2_5-20b-chat-IMat-GGUF">legraphista/internlm2_5-20b-chat-IMat-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard">UGI Leaderboard - a Hugging Face Space by DontPlanToEnd</a>: 未找到描述</li><li><a href="https://tenor.com/view/money-dollars-cash-rich-shut-up-and-take-my-money-gif-3555042">Shut Up! GIF - Money Dollars Cash - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://reddit.com/r/stableDiffusion/comments/1e">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/hub/gguf">GGUF</a>: 未找到描述</li><li><a href="https://reddit.com/r/stableDiffusion/comments/1el79h3/flux_can_be_run_on_a_multigpu_configuration/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/5021">ggml : add Flash Attention by ggerganov · Pull Request #5021 · ggerganov/llama.cpp</a>: 参考 #3365 为 ggml 和 llama.cpp 中的 Flash Attention 支持进行必要设置。提议的算子执行：// new res = ggml_flash_attn(ctx, q, k, v, kq_mask, kq_scale);  // fused sc...</li><li><a href="https://openwebui.com/">Open WebUI</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1270107257018912950)** (59 条消息🔥🔥): 

> - `测试 8700G/780m IGP`
> - `即将到来的 GPU 对比`
> - `P40 与 4090 价格对比`
> - `模型显存（VRAM）利用率`
> - `CPU 和 GPU 升级` 


- **8700G/780m IGP 测试结果参差不齐**：在 **8700G/780m IGP** 上使用特殊版本的 Ollama 进行测试显示，相比 CPU 有约 **25% 的加速**，但在 LM Studio 中使用 **Vulkan** 仅提升了 **15%**。
   - 虽然在 **llama3.1 70b q4** 上实现了 **30% 的性能提升**，但 LM Studio 将可用的 GPU RAM 限制在 **20GB**，影响了更大型号的模型。
- **对未来 GPU 发布的期待**：成员们在等待 **Studio M4 Ultra vs 5090** 的对比，并讨论了 **RTX 6000 Ada** 的前景及性能预期。
   - 一位成员幽默地预测 **5090** 可能要花掉一个“左肾”，黄牛炒作可能会抵消部分需求。
- **澳大利亚的 P40 与 4090 价格差异**：成员们讨论了价格悬殊，**4090** 售价约为 **3000 澳元**，远高于售价 **300-600 澳元** 的 **P40**。
   - **P40** 的市场表现表明，由于自发布以来的供需失衡，其价值有所增加。
- **为大型模型利用 VRAM**：成员们分享了经验，指出运行大型模型通常需要仔细平衡 **VRAM 和 RAM**，一些人成功在 **24GB** GPU 上运行了大型模型。
   - 对于显存充足的用户，建议测试 **Yi 1.5 34b 32k** 模型。
- **关于 4090 性能的反馈**：在购入 **4090** 后，一位成员对其性能提出质疑，称其并不比之前的 **3080** 快多少。
   - 他们提到可能需要考虑使用 **两块 4090 或转向 Mac** 以获得更好的性能稳定性。


  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1270135353378082977)** (5 条消息): 

> - `Gameboy 模拟器设置`
> - `CPython 环境考量`
> - `强化学习直播`
> - `用于多智能体规划的 GPUDrive`
> - `Mojo 讨论` 


- **Gameboy 模拟器环境设置**：[PufferLib GitHub 仓库](https://github.com/PufferAI/PufferLib/blob/729003f9cb89845cc1a69a65e5a2431b2d0542bd/pufferlib/environments/pokemon_red/environment.py#L15)中提供了一个详细的 **Gameboy 模拟器** 设置示例，简化了游戏环境的强化学习（Reinforcement Learning）。
   - 这为无需速度优化即可深入研究 RL 提供了一种实用的方法。
- **创作者直播直接互动**：PufferLib 的创作者正在主持一场 [开发直播](https://www.youtube.com/watch?v=dW10MQ6hKDE)，观众可以直接提问。
   - 直播专注于强化学习开发，提供了一个独特的参与机会。
- **GPUDrive：加速多智能体规划**：一个有趣的生成示例讨论了 **GPUDrive**，这是一款 GPU 加速的多智能体（Multi-Agent）模拟器，每秒可生成超过一百万步的经验，详见 [Hugging Face 论文](https://huggingface.co/papers/2408.01584)。
   - “这项技术能够在传统所需时间的极小部分内完成强化学习 Agent 的有效训练，”增强了处理复杂 Agent 行为的能力。
- **Mojo 演讲邀请**：一位成员对 Chris 的加入表示感谢，并邀请任何团队成员就 **Mojo** 进行演示。
   - 建议的演讲可以涵盖 Mojo 当前状态及其在技术领域愿景的入门概述。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/papers/2408.01584">论文页面 - GPUDrive: Data-driven, multi-agent driving simulation at 1 million FPS</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=dW10MQ6hKDE">强化学习开发直播</a>：在 XStar 上关注 jsuarez5341，GitHub 地址 https://github.com/pufferai/pufferlib，MIT 博士及全职开源 RL 专家</li><li><a href="https://github.com/PufferAI/PufferLib/blob/729003f9cb89845cc1a69a65e5a2431b2d0542bd/pufferlib/environments/pokemon_red/environment.py#L15">PufferLib 仓库代码</a>：简化复杂游戏环境的强化学习 - PufferAI/PufferLib
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1270247873237356574)** (17 messages🔥): 

> - `PyTorch 2.4 与 CUDA 12.4`
> - `Zippika 的 Cublas 库`
> - `FP16 Accumulate 性能`
> - `基准测试速度与精度` 


- **关于 PyTorch 2.4 和 CUDA 12.4 的问题**：一位用户报告了在 **CUDA 12.4** 上使用 **PyTorch 2.4** 的问题，指出虽然构建可以运行，但与 **CUDA 12.1** 相比结果较差。
   - 还有关于用户在 conda 环境中使用 **CUDA 12.6** 系统设置的补充背景。
- **Zippika 的 Cublas 库获得 Windows 兼容性**：Zippika 展示了他们的 **Cublas hgemm 库** 现在兼容 Windows，在特定操作上将性能从 **60 TFLOPS** 大幅提升至 **105 TFLOPS**。
   - 他们分享了一个 [GitHub 仓库](https://github.com/aredden/torch-cublas-hgemm)，展示了该库的功能以及在各种 GPU 上的基准测试结果。
- **FP16 Accumulate 对性能的提升**：Zippika 强调，该库使用 **FP16 accumulate** 将性能提升至 **330 TFLOPS**，而使用 **FP32 accumulate** 时为 **165 TFLOPS**。
   - 他们强调，尽管存在担忧，但 **FP16 accumulate** 在消费级 GPU 上明显更快，尽管必须注意减轻潜在的 **inf/nan** 问题。
- **基准测试结果显示细微差异**：Zippika 提供的基准测试结果表明，他们的库产生的输出与 PyTorch 的 **nn.Linear** 非常接近，仅显示出微小的差异。
   - 记录的时间显示，他们的实现达到了 **438.80 us** 和 **313.22 TFLOPS**，而标准 PyTorch 实现为 **825.59 us** 和 **166.47 TFLOPS**。
- **对模型质量的影响**：Zippika 声称，在 **diffusion models** 和 **LLMs** 等场景下，使用他们的库观察到的性能差异不会对生成质量产生不利影响。
   - 这一断言通过分享准确的基准测试结果得到了加强，结果显示对模型生成一致性的影响微乎其微。



**提到的链接**: <a href="https://github.com/aredden/torch-cublas-hgemm">GitHub - aredden/torch-cublas-hgemm: PyTorch half precision gemm lib w/ fused optional bias + optional relu/gelu</a>: PyTorch half precision gemm lib w/ fused optional bias + optional relu/gelu - aredden/torch-cublas-hgemm

  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1270094130457870440)** (3 messages): 

> - `CIFAR10 准确率`
> - `量化位数优化` 


- **CIFAR10 模型需要调优**：一位成员报告说他们的模型在 **CIFAR10** 上卡在 **70% 准确率**，表明需要进一步调优。
   - 他们表示虽然模型似乎可以运行，但要获得更好的性能需要进行调整。
- **优化量化位数**：另一位成员强调，**quantization bits**（量化位数）可以作为一个可优化参数，他们认为这是一个关键贡献。
   - 这被认为是一个可能影响整体模型性能的重要方面。


  

---


### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1270193824337100872)** (7 messages): 

> - `Hudson River Trading 实习`
> - `高性能 GPU 工作`
> - `求职申请流程` 


- **Hudson River Trading 提供实习机会**：Hudson River Trading 的实习主要在夏季，实习生有机会从事与全职职位类似的 *GPU research*。
   - 目前的实习生正在参与 GPU 研究，但许多“核心机密”任务是留给全职员工的。
- **对 GPU 研究角色的兴奋**：一位用户表达了申请实习的兴趣，因为他们在类似工作方面有经验，并对这些机会感到兴奋。
   - 鼓励他们关注即将开始的申请，特别是夏季职位。
- **通过私信沟通**：一位用户注意到私信功能已关闭，表达了建立联系的愿望。
   - 发布者确认他们在尝试解决消息问题时发送了好友请求。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://grnh.se/9f8394ba1us">高级软件工程师 - 性能优化 (C++/GPU)</a>: 美国纽约州纽约市</li><li><a href="https://www.levels.fyi/companies/hudson-river-trading/salaries/software-engineer">Hudson River Trading 软件工程师薪资 | $406K-$485K+ | Levels.fyi</a>: Hudson River Trading 在美国的软件工程师薪酬范围从 L1 的每年 40.6 万美元到 L3 的每年 48.5 万美元以上。美国的中位数薪酬总额为 41 万美元。查看...
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1270162701972148335)** (34 条消息🔥): 

> - `INT8 Symmetric Quantization`
> - `Install Issues with torchao`
> - `Guarding Unsupported Hardware`
> - `Quantized Training`
> - `GPTQ Refactor Progress` 


- **INT8 Symmetric Quantization 讨论**：一位用户质疑在 INT8 quantization 中使用 **127.5** 作为 scale，认为当 softmax 输出被裁剪（clipped）时，这会导致有偏差的权重更新。
   - 成员们讨论了 *full range quantization* 与 *restricted range quantization* 的优劣，其中一人指出在 quantized training 实验中遇到的潜在偏差挑战。
- **从源码安装 torchao**：一位用户在尝试使用 `python setup.py develop` 从源码安装 **torchao** 时遇到多个错误，特别是在 T4 GPU 上。
   - 最终，通过命令 `USE_CPP=0 pip install .` 成功安装，尽管这种方法由于禁用了 cpp extensions 而存在错过某些测试的风险。
- **针对不支持硬件的保护机制建议**：成员们讨论了引入 compile guard 以防止不支持的硬件在编译期间引发错误，并参考了[这个示例](https://github.com/pytorch/pytorch/blob/e98eac76b358fb4639b9e9ce6894014354d7b073/aten/src/ATen/native/cuda/int4mm.cu#L1)。
   - 虽然 compile guard 有助于安装，但仍有人担心在调用操作时 runtime checks 会导致奇怪的错误消息。
- **Quantized Training 更新**：一位用户分享了实现 **INT8 quantized training** 如何增强 training from scratch 或 pre-training 阶段的使用场景。
   - 他们强调需要将 inference 和 training 分开，并指出与 quantization 方法相关的操作可能存在差异。
- **GPTQ Refactor 进度**：一位用户已完成约 **45%** 的 **GPTQ** refactor 工作，其中包含使用 MultiTensor 替换 fx.interpreter。
   - 他们表示还需要几天时间来解决相关的 GitHub issue，表明工作正在稳步推进。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/">
    
      PyTorch
    
  </a>: 未找到描述</li><li><a href="https://github.com/pytorch/ao/blob/de4a1fb3b1f71e2f61b84dfdc96e7d704ff72208/torchao/quantization/quant_primitives.py#L610">ao/torchao/quantization/quant_primitives.py at de4a1fb3b1f71e2f61b84dfdc96e7d704ff72208 · pytorch/ao</a>: 缺失的用于训练和推理的 pytorch dtype 和 layout 库 - pytorch/ao</li><li><a href="https://intellabs.github.io/distiller/algo_quantization.html#symmetric-mode">Quantization - Neural Network Distiller</a>: 未找到描述</li><li><a href="https://github.com/pytorch/pytorch/blob/e98eac76b358fb4639b9e9ce6894014354d7b073/aten/src/ATen/native/cuda/int4mm.cu#L1">pytorch/aten/src/ATen/native/cuda/int4mm.cu at e98eac76b358fb4639b9e9ce6894014354d7b073 · pytorch/pytorch</a>: Python 中具有强 GPU 加速的 Tensor 和动态神经网络 - pytorch/pytorch
</li>
</ul>

</div>

### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1270258859633676380)** (7 条消息): 

> - `LLaMA 3 论文见解`
> - `Prefix Chunk LLM 探索`
> - `SARATHI 框架介绍`
> - `corCTF 2023 内核 syscall 挑战` 


- **LLaMA 3 论文揭示了令人兴奋的数据集章节**：一位成员指出 **LLaMA 3** 论文阅读起来很快，并强调 **dataset section**（数据集章节）非常有趣，同时建议其他部分在相关文献中有更好的解释。
   - 这一见解表明，其创新性主要在于数据集的架构。
- **探索 Prefix Chunk LLM 的乐趣**：一位成员推荐阅读 **Prefix Chunk LLM 论文** (Sarathi LLM)，声称它比之前的作品更有趣。
   - 社区成员讨论了共享 prompt 以及在 LLM 中提升性能的影响。
- **用于 LLM 推理的 SARATHI 介绍**：一位用户介绍了 **SARATHI**，这是一个通过使用 **chunked-prefills** 和批处理策略来解决 LLM 推理阶段效率低下问题的框架，旨在最大化 GPU 利用率。
   - 该方法强调减少 **pipeline parallelism** 过程中的不平衡，以提高整体效率。
- **corCTF 2023 中新的 syscall 漏洞利用挑战**：一位用户详细介绍了 corCTF 2023 背景下的一个新 **syscall 挑战**，展示了一个连接内核内部和微架构攻击的 Linux syscall。
   - 该挑战要求玩家利用新定义的 syscall 在内核中进行 buffer 操作。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2308.16369">SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills</a>: 大语言模型 (LLM) 推理包含两个不同的阶段——处理输入 prompt 的 prefill 阶段和以自回归方式生成输出 token 的 decode 阶段。虽然 prefill...</li><li><a href="https://arxiv.org/abs/2402.15220">ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition</a>: Self-attention 是大语言模型 (LLM) 的核心组件，但也是长序列推理延迟的主要来源。在多租户 LLM 服务场景中，计算和内存 ...</li><li><a href="https://www.willsroot.io/2024/08/just-a-dos-bug.html?m=1">Will's Root: corCTF 2024: Its Just a Dos Bug Bro - Leaking Flags from Filesystem with Spectre v1</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1270094011696152648)** (99 messages🔥🔥): 

> - `Ragged Attention Issues` (Ragged Attention 问题)
> - `Training with EOS and BOS tokens` (使用 EOS 和 BOS Token 进行训练)
> - `Batch Size and Stability` (Batch Size 与稳定性)
> - `Newline Handling in Llama Models` (Llama 模型中的换行符处理)
> - `Performance Benchmarking with PyTorch 2.4` (使用 PyTorch 2.4 进行性能基准测试)


- **Ragged Attention 和掩码挑战**：成员们讨论了在模型中实现 Ragged Attention 的复杂性，强调了使用各种掩码（masks）以防止训练期间出现分布外（out-of-distribution）行为的必要性。
   - 强调了掩码维度（dimensionality）的重要性，并建议支持 Ragged Attention 作为维持训练完整性的解决方案。
- **训练中 EOS/BOS Token 的困惑**：针对 Meta 对停止 Token 的实现提出了担忧，特别是推理代码中省略了 EOS Token，这可能导致无限采样循环。
   - 一位成员怀疑这种遗漏可能会导致训练问题，敦促审查这些 Token 在模型训练过程中的处理方式。
- **Batch Size 对训练稳定性的影响**：随后讨论了在训练早期使用较小的 Batch Size 以增强稳定性，揭示了训练效率与模型可靠性之间的权衡。
   - 引用相关论文和实践强调了深入理解的必要性，因为逐渐增加 Batch Size 可以缓解不稳定性。
- **理解 Llama 模型中的换行符使用**：关于 Llama 3 基座模型 Token 格式中包含换行符的问题浮出水面，建议开发者在预训练期间考虑包含这些 Token。
   - 推测认为加入换行符可能使模型更好地应对指令任务，尽管这种方法的影响尚不确定。
- **PyTorch 带来的快速性能提升**：一位成员报告称，使用 PyTorch 2.4 运行 `train_gpt2.py` 脚本比 `llm.c` 获得了更好的性能提升，展示了新更新的潜力。
   - 对比结果展示了在使用和不使用 Flash Attention 执行时的细微差别，表明模型训练效率正在持续改进。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2312.16903">Spike No More: Stabilizing the Pre-training of Large Language Models</a>：Loss 尖峰经常发生在 LLM 的预训练过程中。这些尖峰会降低 LLM 的性能，有时甚至会毁掉预训练。由于预训练需要大量的资源...</li><li><a href="https://huggingface.co/docs/transformers/main">🤗 Transformers</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2108.06084">The Stability-Efficiency Dilemma: Investigating Sequence Length Warmup for Training GPT Models</a>：最近的研究在海量 GPU 上预训练大规模自回归语言模型方面取得了巨大成功。为了减少实际训练时间，通常的做法是增加 Batch Size...</li><li><a href="https://huggingface.co/docs/transformers/main/en/chat_templating">Templates for Chat Models</a>：未找到描述</li><li><a href="https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/">Llama 3 | Model Cards and Prompt formats</a>：Llama 3 使用的特殊 Token。一个 Prompt 应包含单个 System 消息，可以包含多个交替的 User 和 Assistant 消息，并且总是以最后一个 User 消息结尾，后跟...</li><li><a href="https://github.com/Dao-AILab/flash-attention/issues/654.">Issues · Dao-AILab/flash-attention</a>：快速且内存高效的精确 Attention。通过在 GitHub 上创建账号来为 Flash Attention 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchchat/issues?q=sort%3Aupdated-desc+is%3Aissue+is%3Aopen">Issues · pytorch/torchchat</a>：在服务器、桌面和移动端本地运行 PyTorch LLM - Issues · pytorch/torchchat</li><li><a href="https://github.com/pytorch/torchchat/issues?q=sort%3Aupdated-desc+is%3Ais">Issues · pytorch/torchchat</a>：在服务器、桌面和移动端本地运行 PyTorch LLM - Issues · pytorch/torchchat</li><li><a href="https://github.com/meta-llama/llama-models/issues/91">Broken links in prompt format docs · Issue #91 · meta-llama/llama-models</a>：在这篇博文中，有两个关于 Prompt 格式的链接已失效 https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/，因此不清楚生成指令的具体位置...
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1270413451608461313)** (9 条消息🔥): 

> - `ZLUDA 3 移除`
> - `AMD 的法律声明`
> - `ZLUDA 的开发状态` 


- **ZLUDA 3 被下架**：在 AMD 声称发布许可无效后，**ZLUDA** 作者已下架 **ZLUDA 3**，参考 [GitHub 页面](https://github.com/vosen/ZLUDA)。
   - *我与 AMD 合同中的条款之一是，如果 AMD 认为它不适合进一步开发，我可以将其发布*。
- **AMD 质疑合法性**：成员们讨论了 AMD 声称关于发布 **ZLUDA** 的**雇佣合同**不具法律约束力的影响。
   - 持续的讨论强调，如果 AMD 认为 ZLUDA 适合进一步开发，这将使作者发布它的能力变得复杂。
- **对 AMD 角色的认可**：针对 **ZLUDA** 相关情况，一位成员简单地用一句“*thanks amd*”表达了对 AMD 的“感谢”。
   - 这种情绪似乎反映了对影响 **ZLUDA** 未来的法律纠纷的一种幽默与沮丧交织的态度。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/vosen/ZLUDA">GitHub - vosen/ZLUDA: CUDA on ??? GPUs</a>：CUDA on ??? GPUs。通过在 GitHub 上创建账号为 vosen/ZLUDA 的开发做出贡献。</li><li><a href="https://github.com/vosen/ZLUDA/tree/v3?tab=readme-ov-file#faq">GitHub - vosen/ZLUDA at v3</a>：CUDA on ??? GPUs。通过在 GitHub 上创建账号为 vosen/ZLUDA 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1270144971646701569)** (2 条消息): 

> - `项目时间线`
> - `Google Form 详情` 


- **预计月底前会有更新**：一位成员表示有信心最迟在月底前获知有关项目状态的更新。
   - 这为参与者提供了更多信息可用时间的时间线。
- **详细说明项目工作的重要性**：另一位成员提到，鉴于任务列表很长，确保清晰度的最佳方法是通过 Google form 提供有关将要开展的工作的详细信息。
   - 他们还建议在频道中链接任何提案，以便于访问。


  

---



### **Nous Research AI ▷ #[datasets](https://discord.com/channels/1053877538025386074/1105324249721356298/1270435322681102447)** (1 条消息): 

> - `UltraSteer-V0 数据集`
> - `Llama2-13B-SteerLM-RM`
> - `细粒度对话标签` 


- **介绍 UltraSteer-V0 数据集**：新的精选数据集 **UltraSteer-V0** 包含 **230 万条对话**，共有 **280 万个轮次**，并标注了 **9 个细粒度信号**，由 Nvidia 的 **Llama2-13B-SteerLM-RM 奖励模型**生成。
   - 尽管这是 **V0 版本**，但由于经过了 22 天的标注和处理以及广泛的去重，它承诺提供独特的线程延续。
- **助手轮次的标注标准**：UltraSteer 数据集中的每个助手轮次都根据**质量 (Quality)**、**毒性 (Toxicity)**、**幽默感 (Humor)** 和**创造力 (Creativity)** 进行评分，每个维度分值为 0 到 4 分，以捕捉细微的对话属性。
   - 这些属性旨在增强对对话式 AI 响应的分析，标志着对话数据集质量的重大进步。
- **UltraSteer-V0 仍需进一步改进**：创建者承认 UltraSteer-V0 仍可从进一步的**去重**和数据集卡片的改进中受益。
   - 这一反馈反映了对社区意见的开放态度，以便在未来的迭代中增强可用性和清晰度。
- **UltraSteer 的框架与制作**：UltraSteer 数据集的制作涉及使用 **NeMo Aligner** 框架，展示了 Nvidia 对推进**对话数据集技术**的承诺。
   - 复杂的制作过程强调了为 AI 研究生成对话数据集的质量。
- **访问 UltraSteer-V0**：该数据集现在可以在 [Hugging Face](https://huggingface.co/datasets/Avelina/UltraSteer-v0) 上访问，使研究人员和开发人员能够利用其功能。
   - 凭借其庞大的规模和详细的标注，UltraSteer-V0 被定位为增强对话系统的宝贵资源。



**提到的链接**：<a href="https://huggingface.co/datasets/Avelina/UltraSteer-v0">Avelina/UltraSteer-v0 · Datasets at Hugging Face</a>：未找到描述

  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/)** (1 条消息): 

vikings7699: 这里有人做过专门针对保险行业的模型微调吗？
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1270123092227788863)** (129 条消息🔥🔥): 

> - `Fine-tuning 视觉模型`
> - `Flux AI 能力`
> - `新型多模态模型`
> - `Open Medical Reasoning Tasks`
> - `MiniCPM-Llama3-V 更新` 


- **Fine-tuning 视觉模型的困扰**：一位用户分享了对 Fine-tuning 视觉模型效果不如预期的沮丧，引发了关于过拟合（overfitting）和模型局限性的讨论。
   - 另一位用户建议，误差累积和灾难性遗忘（catastrophic forgetting）可能是影响性能的因素。
- **对 Flux AI 新技能的兴奋**：一份演示文稿强调 Flux AI 声称在**文本理解**、**Prompt 理解**和**图像生成**方面表现出色。
   - 成员们对 Flux AI 的能力表示热烈欢迎，部分成员已经开始使用其 Pro 版本。
- **Open Medical Reasoning Tasks 介绍**：启动了一项名为 Open Medical Reasoning Tasks 的协作计划，专注于为医疗保健领域的 LLM 创建全面的任务列表。
   - 鼓励参与者做出贡献，强调了整合 AI 和医学专业知识的重要性。
- **MiniCPM-Llama3-V 更新及能力声明**：用户讨论了 MiniCPM-Llama3-V 模型的最新更新，包括在处理多图像输入和 OCR 任务方面能力的提升。
   - 对早期模型版本的最初怀疑，随着展示多图像处理能力的新示例出现，转变为兴奋。
- **模型性能对比讨论**：成员们对比了来自 **Hugging Face** 和其他创建者的各种模型的性能，许多人对通过多图像输入获得更好结果感兴趣。
   - 对话还提到了像 **BigLlama-3.1** 这样的新模型如何推进行业的性能边界。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3">HuggingFaceM4/Idefics3-8B-Llama3 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5">openbmb/MiniCPM-Llama3-V-2_5 · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/fofrAI/status/1820878455266816260">来自 fofr (@fofrAI) 的推文</a>: 🤯  &gt; powerpoint presentation, the slide title says “Flux AI has new skills”, three bullet points, “good at text”, “prompt comprehension”, “amazing images”</li><li><a href="https://huggingface.co/openbmb/MiniCPM-V-2_6">openbmb/MiniCPM-V-2_6 · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/maximelabonne/status/1820746013503586669">来自 Maxime Labonne (@maximelabonne) 的推文</a>: 🦙✨ BigLlama-3.1-1T-Instruct  So I&#39;ve heard that 405B parameters weren&#39;t enough...   It&#39;s my pleasure to present an upscaled Llama 3.1 with 1,000,000,000 parameters. Now available on @hugg...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1elgr2x/new_open_llm_leaderboard_champion/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://x.com/aadityaura/status/1820617406970278272?s=46">来自 Aaditya Ura ( looking for PhD ) (@aadityaura) 的推文</a>: Exciting news! 🎉 Introducing the Open Medical Reasoning Tasks project!  Inspired by @NousResearch and @Teknium1, @OpenLifeSciAI ( Open Life-Science AI ) is launching an open, collaborative initiative...</li><li><a href="https://github.com/black-forest-labs/flux/issues/9)">Issues · black-forest-labs/flux</a>: FLUX.1 模型的官方推理仓库。通过在 GitHub 上创建账号为 black-forest-labs/flux 的开发做出贡献。</li><li><a href="https://github.com/OpenBMB/MiniCPM-V/issues/233">MiniCPM-V Finetuning for multi-image input during a multi-turn conversation💡 [REQUEST] - &lt;title&gt; · Issue #233 · OpenBMB/MiniCPM-V</a>: 起始日期 | Start Date No response 实现PR | Implementation PR No response 相关Issues | Reference Issues for multi-image input during a multi-turn conversation 摘要 | Summary for multi-image input during a mul...</li><li><a href="https://old.reddit.com/r/StableDiffusion/comments/1ekte84/generated_with_flux1_pro_and_schnell/">使用 Flux.1 Pro 和 Schnell 生成</a>: 发布于 r/StableDiffusion，作者 u/Sea_Law_7725 • 376 点赞和 78 条评论
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1270123471300464783)** (19 messages🔥): 

> - `微调库`
> - `推理栈资源`
> - `保险行业模型调优`
> - `按需付费的 LLaMA 托管`
> - `训练中的计算瓶颈` 


- **最常用的微调库**：许多人倾向于使用现有的库（如 **Axolotl**）进行微调和训练，而不是从头编写独特的脚本。
   - *使用库的好处*在简化训练流程方面得到了认可。
- **开始使用 vLLM**：一位用户询问了关于 **vLLM** 推理栈的资源，建议该项目本身就是一个很好的起点。
   - 讨论继续围绕值得探索的相关代码库展开，以加深理解。
- **为保险行业微调模型**：一位用户询问是否有人成功地专门为**保险行业**微调过模型。
   - 该话题探讨了与这一垂直领域相关的挑战和策略。
- **按需付费的 LLaMA 托管选项**：有人在寻找提供 **LLaMA 450b** 按需付费访问的公司，并暗示 **Groq** 和 **Openrouter** 是潜在的解决方案。
   - 成员们讨论了需要查看 **Openrouter** 上列出的托管提供商以获取更多信息。
- **理解计算瓶颈**：推理和训练中的主要瓶颈，特别是在 Batch Size 为 1 时，通常与**内存**有关。
   - 随着 Batch Size 的增加，它会变成**计算受限 (compute bound)**，从而引发了关于 GPU 利用率和活跃 CUDA 核心的讨论。


  

---


### **Nous Research AI ▷ #[reasoning-tasks-master-list](https://discord.com/channels/1053877538025386074/1264666760972472481/1270115994676756523)** (7 messages): 

> - `开放医疗推理任务`
> - `合成任务生成`
> - `LLM 的局限性`
> - `社区贡献` 


- **开放医疗推理任务激动人心的发布**：受 Nous Research 和 Teknium 的启发，[Open Life-Science AI](https://x.com/aadityaura/status/1820617406970278272?s=46) 发起了一项开放倡议，旨在为 LLM 创建一份全面的医疗推理任务清单。
   - 该项目寻求医生、研究人员和数据科学家的贡献，以推动 AI 在医疗保健领域的发展，详见 [GitHub](https://github.com/openlifescience-ai/Open-Medical-Reasoning-Tasks)。
- **社区对协作努力的热情**：一位成员对新的医疗推理项目表示热烈欢迎，称：*“这太棒了！这就是公开协作的结果！我太喜欢了！”*
   - 在关于开放协作对推进医疗 AI 积极影响的相关讨论中，这种情绪得到了共鸣。
- **关于 System 2 Reasoning 的更多资源**：提到了一份关于 [System 2 Reasoning Link Collection](https://github.com/open-thought/system-2-research) 的贡献，这是一个协作式的 GitHub 仓库。
   - 该仓库鼓励社区参与收集与 System 2 Reasoning 及其应用相关的宝贵见解。
- **合成任务生成的探索**：考虑了如何改进**合成任务生成**，以克服 LLM 在生成复杂任务时面临的局限性。
   - 一位成员强调了在超越目前 LLM 可实现的简单输出方面所面临的挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/aadityaura/status/1820617406970278272?s=46">Aaditya Ura 的推文 ( 正在寻找 PhD ) (@aadityaura)</a>: 激动人心的消息！🎉 介绍 Open Medical Reasoning Tasks 项目！受 @NousResearch 和 @Teknium1 的启发，@OpenLifeSciAI ( Open Life-Science AI ) 正在发起一项开放的、协作的倡议...</li><li><a href="https://github.com/open-thought/system-2-research">GitHub - open-thought/system-2-research: System 2 Reasoning 链接收集</a>: System 2 Reasoning 链接收集。通过在 GitHub 上创建账号为 open-thought/system-2-research 的开发做出贡献。
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1270094956945604620)** (128 messages🔥🔥): 

> - `Web 开发到 AI 工程师的转型路径`
> - `OpenAI 人员离职`
> - `零售业中的生成式 AI`
> - `GPT-4o 中的结构化输出`
> - `基于能量的语言建模`

- **Web Dev 向 AI Engineer 转型的路径正在形成**：讨论强调了由于高需求和 ML engineers 的短缺，Web 开发者向 AI engineering 转型的趋势日益增长，一些人分享了他们在调整技能组合方面的经验。
   - 参与者对 Web 开发者整体上如何被视为“通用型人才”表现出兴趣，他们通常在承担传统开发任务的同时，也负责 AI 项目的实施。
- **OpenAI 近期的人员离职引发担忧**：OpenAI 的一波领导层变动引发了对其公司方向的质疑，社区中许多人对其未来和稳定性表示担忧。
   - 评论包括对这些离职影响的推测，以及 OpenAI 的发展轨迹是否仍然积极。
- **Generative AI 在零售领域的应用**：Generative AI 正在零售领域取得进展，特别是在增强不同平台和语言的产品描述方面，成员们以 L'Oreal 为例进行了讨论。
   - 讨论中提出了如何衡量 AI 生成描述的有效性问题，强调了在评估性能时需要相关指标。
- **GPT-4o 推出 Structured Outputs 功能**：OpenAI 引入了 Structured Outputs 功能，使模型能够可靠地遵循 JSON schemas，与之前的版本相比，可靠性显著提高。
   - 该功能的推出标志着 AI 在生成更受控和结构化数据输出能力方面的进步，这在社区成员中得到了认可和讨论。
- **对 Energy-Based Language Modeling 的怀疑**：分享的一个关于与 Extropic AI 研究员会面的幽默轶事揭示了其对 Energy-Based Language Modeling 领域既有成果的缺乏了解，引发了对其主张的怀疑。
   - 这次交流指向了一个更广泛的信誉问题，即一些初创公司在 AI 复杂课题方面的专业知识。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/TwoWeeksLOL/status/1820536638268948750">来自 Two Weeks LOL (@TwoWeeksLOL) 的推文</a>：@MKBHD 噢不...</li><li><a href="https://x.com/tszzl/status/1714357380413264044?s=46">来自 roon (@tszzl) 的推文</a>：OpenAI 那些能进行眼神交流的人都是在过去 6 个月内加入的，他们的眼神交流让我感到不自在</li><li><a href="https://news.ycombinator.com/item?id=41174306">未找到标题</a>：未找到描述</li><li><a href="https://news.ycombinator.com/item?id=41173964">未找到标题</a>：未找到描述</li><li><a href="https://x.com/NickADobos/status/1820513765823250730">来自 Nick Dobos (@NickADobos) 的推文</a>：关于用 AI 编写代码的一篇很棒的帖子，喜欢这张图表。引用 Erik Schluntz (@ErikSchluntz)：用 AI 替代我的右手（我是如何在打着石膏的情况下，每周为工作编写数千行代码的）...</li><li><a href="https://x.com/aizkmusic/status/1820594845792051391?s=46">来自 Aizk ✡️ (@Aizkmusic) 的推文</a>：@BigTechAlert @ChatGPTapp @TarunGogineni 他的 LinkedIn 简介很棒</li><li><a href="https://arxiv.org/abs/2307.09702">Efficient Guided Generation for Large Language Models</a>：在本文中，我们展示了如何将神经文本生成问题建设性地重新表述为有限状态机状态之间的转换。该框架带来了一种高效的...</li><li><a href="https://x.com/_philschmid/status/1820715040191750370">来自 Philipp Schmid (@_philschmid) 的推文</a>：“Deep Reinforcement Learning from Human Preferences”和“Proximal Policy Optimization Algorithms”是 LLM 中现代 RLHF 基础的一部分。</li><li><a href="https://x.com/OpenAIDevs/status/1820542222259073137">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>：我们将举办 OpenAI DevDay 巡回活动！今年秋天在旧金山、伦敦或新加坡加入我们，参加实操环节、演示和最佳实践分享。与我们的工程师见面，了解各地的开发者如何...</li><li><a href="https://x.com/jxmnop/status/1820876333154759091">来自 jack morris (@jxmnop) 的推文</a>：关于 Extropic AI 的一个小趣闻 > 对他们好奇已久 > 在 Twitter 上互粉的一位是该公司的工程师/研究员 > 经常发布关于 energy-based modeling 和 LM-quant...</li><li><a href="https://x.com/jason_koebler/status/1820493304490074391">来自 Jason Koebler (@jason_koebler) 的推文</a>：来自 @samleecole 的独家新闻：泄露的 Slack 记录和文档显示了 NVIDIA AI 抓取的惊人规模：每天抓取相当于 80 年（即“一个人的一生”）时长的视频。已获得最高层的批准...</li><li><a href="https://x.com/abacaj/status/1820883396077482087">来自 anton (@abacaj) 的推文</a>：很有意思... 新模型还包含相当大幅度的降价。引用 OpenAI Developers (@OpenAIDevs)：在 API 中引入 Structured Outputs——模型输出现在遵循开发者提供的 JSON ...</li><li><a href="https://x.com/johnschulman2/status/1820610863499509855">来自 John Schulman (@johnschulman2) 的推文</a>：我今天向 OpenAI 的同事们分享了以下便条：我做出了离开 OpenAI 的艰难决定。这一选择源于我希望加深对 AI alignment 的关注，并开始一个...</li><li><a href="https://x.com/_mira___mira_/status/1820625134354669697?s=46">来自 Mira (@_Mira___Mira_) 的推文</a>：未找到描述</li><li><a href="https://x.com/michpokrass/status/1820881057824305567">来自 Michelle Pokrass (@michpokrass) 的推文</a>：很高兴宣布 Structured Outputs —— 我们 API 中的最新功能。模型输出现在将可靠地遵循您精确的 JSON schemas，准确匹配参数和类型。schema 可靠性...</li><li><a href="https://github.com/simonw/datasette">GitHub - simonw/datasette: An open source multi-tool for exploring and publishing data</a>：一个用于探索和发布数据的开源多功能工具 - simonw/datasette</li><li><a href="https://writer.com/use-cases/ecommerce/">电子商务与零售</a>：了解创新的电子商务和零售公司如何使用 Writer 创建从首次接触到销售都行之有效的品牌内容。
</li>
</ul>

### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1270101558738157713)** (1 条消息): 

> - `OpenAI DevDay`
> - `Global Events` 


- **OpenAI DevDay 走向全球！**：OpenAI 将在今年秋季开展 **DevDay** 巡回活动，地点包括**旧金山**、**伦敦**和**新加坡**。
   - 参与者可以期待上手实践环节、演示和最佳实践，并有机会与 **OpenAI 工程师**交流，了解开发者如何利用 OpenAI 进行构建。更多详情请见[此处](https://openai.com/devday/)。
- **与全球开发者建立联系**：DevDay 为全球开发者提供了一个连接和分享关于使用 **OpenAI** 技术构建应用的见解的机会。
   - 参与深入讨论，探索正在重新定义 AI 开发的创新实践。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1270275479793827891)** (86 条消息🔥🔥): 

> - `Desktop ChatGPT App Release`
> - `DALL-E 3 Model Discussion`
> - `API and Hosting Questions`
> - `LaTeX Support on Mobile`
> - `OpenAI Updates and Pricing` 


- **关于桌面版 ChatGPT 应用和 Search GPT 发布的问题**：成员们对 Windows 版**桌面 ChatGPT 应用**的发布时间表以及 **Search GPT** 的公开可用性表示好奇。
   - 讨论中还出现了一些幽默的评论，称 Sam Altman 是目前唯一留任的创始人。
- **DALL-E 3 模型及结果差异性**：关于 **DALL-E 3 模型**及其结果差异性的讨论仍在继续，其中提到了与 Llama 模型的对比，并询问为何结果会有所不同。
   - 用户指出，输出质量的差异可能是由于 OpenAI 实施的安全过滤器等原因造成的。
- **API 使用与免费选项**：用户询问 **Llama API** 是否免费，讨论了在本地无成本运行 **Llama 3.1 8b** 等模型的可能性。
   - 对话指出，虽然模型本身可能是免费的，但目前没有官方提供的无限免费 API。
- **LaTeX 支持限制**：一位用户对移动端应用缺乏 **LaTeX 支持**表示担忧，一些成员建议使用移动浏览器作为替代方案。
   - 据提到，针对该功能已提交了大量错误报告。
- **令人印象深刻的 OpenAI 更新与定价变化**：讨论了 **Structured Outputs** 等新功能，强调了其在提高一致性和可能降低 API 使用价格方面的优势。
   - 然而，成员们对新模型及其更新迭代版本与旧版本相比的性能价格平衡提出了质疑。



**提到的链接**：<a href="https://stackoverflow.com/questions/78839847/assistant-gpt-can-i-perform-knowledge-retrieval-from-a-cloud-storage">Assistant GPT - 我可以从云存储中执行知识检索吗？</a>：我在云存储（OneDrive）中存有一些文件，想对其进行知识检索。是否可以集成一个 Assistant 来直接从...执行知识检索？

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1270164994192969758)** (16 条消息🔥): 

> - `Search GPT availability`
> - `Photo upload limits`
> - `Generative AI in gaming`
> - `ChatGPT-4o updates` 


- **Search GPT 现已可用**：一位成员确认 **Search GPT** 确实已向用户推出。
   - 这一更新引发了关于用户如何利用这一新功能的问题。
- **会员仍受上传限制**：成员们对即使是付费用户也存在上传限制表示担忧，其中一人报告其限制重置时间为 1:35。
   - 另一位成员澄清说，**即使是付费用户**也受到这些上传限制的约束。
- **生成式 AI 在游戏中的潜力**：一位成员分享了对生成式 AI 增强 **BG3** 或 **Pathfinder** 等游戏的兴奋之情，认为它可以实现角色设计和独特的交互。
   - 他们设想了一种完全沉浸式的体验，**NPC** 可以根据玩家的选择做出动态反应。
- **ChatGPT-4o 模型更新**：用户推测了 **ChatGPT-4o** 可能的变化，其中一人提供了一个关于 Structured Outputs 的更新链接。
   - 另一位成员指出模型可能已经过修订，并提到了版本号 `gpt-4o-2024-08-06`。
- **性能提升引发疑问**：成员们注意到 **ChatGPT-4o** 的性能发生了变化，并询问最近是否进行了任何更新。
   - 一位成员注意到回答有明显不同，从而引发了关于潜在更新的讨论。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 条消息): 

darthgustav.: 使用 Python 工具并从上传的文件中导入数据。
  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 条消息): 

darthgustav.: 使用 Python 工具并从上传的文件中导入数据。
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1270133269052002517)** (82 条消息🔥🔥): 

> - `信息检索的未来`
> - `Perplexity 的技术问题`
> - `语言模型对比`
> - `内容推荐引擎`
> - `用户体验反馈` 


- **信息检索未来的考量**：一位成员思考未来的信息检索系统在使用语言模型时是否会整合来源权重，为信誉良好的来源分配比低可信度来源更高的价值。
   - 他们询问是否有关于此主题的有趣论文，以及模型是否能自主评估来源质量。
- **用户面临 Perplexity 的技术问题**：多位用户报告了 Perplexity Pro 应用中的功能问题，包括无法切换 LLM 以及库丢失导致的功能担忧。
   - 然而，一些用户注意到，在问题出现后不久，功能又意外恢复了。
- **语言模型对比引发辩论**：用户讨论了 GPT-4o 与 Turbo 的对比，对其性能和响应速度看法不一，有人发现 GPT-4o 在对话中效果较差。
   - 一些用户更倾向于其他 LLM，强调 GPT-4o 在接收新指令时似乎缺乏确认。
- **内容推荐引擎的开发**：介绍了一个专注于构建内容分类和推荐引擎的大学项目，目标是根据用户输入分析和分类内容。
   - 成员建议研究 RAG (retrieval-augmented generation) 作为该项目的相关概念。
- **Uber 订阅优惠的体验**：一位用户寻求关于接收 Uber 订阅优惠码的澄清，询问是自动发送还是需要用户操作。
   - 其他人确认他们自动收到了邮件，无需申请。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://uncovr.app>">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/inulute/perplexity-ai-app/releases">Releases · inulute/perplexity-ai-app</a>: 基于 Electron 的 Perplexity AI 桌面应用，将 AI 语言处理的魔力带到你的桌面。 - inulute/perplexity-ai-app</li><li><a href="https://felo.ai/search/PALsa8DEHJaiJcU6DYi4Q9">当汤姆的葬礼举行时，他的父亲没有参加。现在他的父亲去世了，汤姆也没有出现在他父亲的葬礼上。汤姆是否做得太过分了？</a>: 你描述的情况涉及人际关系和个人选择的复杂交织。以下是一些需要考虑的点：  ### 背景与基础
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1270168758467432448)** (7 条消息): 

> - `NVIDIA Blackwell GPUs Delay` (NVIDIA Blackwell GPU 延迟)
> - `Memory Scientific Explanation` (记忆的科学解释)
> - `Market Updates` (市场动态)
> - `LLaMA 3 Performance` (LLaMA 3 性能)
> - `Tastiera Meccanica Recommendations` (机械键盘推荐)


- **NVIDIA 的 Blackwell GPU 因设计缺陷推迟**：NVIDIA 下一代 **Blackwell GPU** 面临延迟，主要原因是生产后期发现的**设计缺陷**以及先进的 **CoWoS-L** 封装技术问题。
   - 这些复杂情况迫使处理器晶圆（die）进行重新设计，延长了生产测试和验证时间。
- **理解记忆机制**：关于**记忆**背后科学解释的讨论强调了其对涉及**海马体**等不同大脑区域过程的依赖。
   - 尽管人们一直对此感兴趣，但没有证据表明物体可以存储记忆，这一概念仍处于形而上学领域。
- **当前市场波动与艺术品销售**：市场讨论提到了各种**时事**，包括一幅价值 2600 万美元的数字肖像画销售以及 Google 最近的法律挫折。
   - 这些问题导致了科技爱好者中波动的市场情绪。
- **LLaMA 3.1 性能洞察**：关于 **LLaMA 3.1** 的新兴细节表明，其性能指标引起了科技社区的极大关注。
   - 预计进一步的洞察将阐明其能力和潜在应用。
- **机械键盘推荐**：对机械键盘推荐的请求凸显了社区对游戏和生产力工具的持续关注。
   - 虽然没有提到具体型号，但对高质量选择的搜索仍在继续。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/embed/ZLEuncAV70U">YouTube</a>: 未找到描述</li><li><a href="https://www.perplexity.ai/search/nvidia-blackwell-s-delay-expla-kjKmWq15SdKcDJAgGn01EQ">NVIDIA Blackwell's Delay Explained</a>: NVIDIA 下一代 Blackwell GPU 遭遇延迟，主要原因是设计和制造问题。以下是延迟的主要原因：...</li><li><a href="https://www.perplexity.ai/search/how-does-llama-3-1-405b-s-perf-YIDs8nm2TuuJzP4ILbY1BA">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可为任何问题提供准确、可靠且实时的答案。</li><li><a href="https://www.perplexity.ai/search/how-can-i-get-a-summary-of-a-b-46KrvDREQKeVwv4VwBO2Lw">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可为任何问题提供准确、可靠且实时的答案。</li><li><a href="https://www.perplexity.ai/search/is-naturland-s-tobotronc-the-l-GVann50ESpqyNuB4wT4qvw">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可为任何问题提供准确、可靠且实时的答案。</li><li><a href="https://www.perplexity.ai/search/consigliami-una-tastiera-mecca-bBenHhBBQUe7YmplIxO6IA">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可为任何问题提供准确、可靠且实时的答案。</li><li><a href="https://www.perplexity.ai/search/apa-saja-benda-yang-mengandung-XdLR2Ja0TB.hH1hwSOy2DA">apa saja benda yang mengandung karbon</a>: 含有碳的物体非常多样，在日常生活中可以以各种形式找到。以下是一些物体的例子...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1270118041962942547)** (8 messages🔥): 

> - `API 错误`
> - `即将到来的模型变更`
> - `在 Labs 中测试`
> - `状态更新` 


- **用户报告 API 差异**：一位用户对他们的 API 查询返回损坏数据表示担忧，导致他们质疑 API 的功能。
   - 另一位用户提到了类似问题，暗示 API 的可靠性可能存在广泛问题。
- **Perplexity API 模型将被弃用**：确认所有 Perplexity API 模型将于 **2024 年 8 月 12 日**弃用，引发了关于其连续性的疑问。
   - 提供了一份关于模型的详细指南，强调了关键参数和 System Prompt 的行为。
- **API 可用性状态检查**：一名成员询问 API 的潜在问题，理由是在尝试连接时出现 HTTP 502 错误。
   - 最近的状态报告显示 API 没有问题，表明这可能是局部问题或特定于用户的问题。
- **在 Labs 中测试 Perplexity API**：一位用户建议通过 [Perplexity Labs](https://labs.perplexity.ai) 游乐场尝试 API 以进行进一步测试。
   - 该建议旨在隔离问题是 API 全局性的还是特定于单个查询的。
- **最近的 Perplexity 状态更新**：状态检查显示过去几天没有关于 Perplexity API 的近期通知或报告的问题。
   - 持续的稳定性表明，如果问题仍然存在，它们可能是特定于用户的，而不是系统性的。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.perplexity.ai/docs/model-cards">支持的模型</a>：Perplexity 模型参数量、上下文长度、模型类型 llama-3-sonar-small-32k-online 8B 28,000 Chat Completion llama-3-sonar-small-32k-chat 8B 32,768 Chat Completion llama-3-sonar-large-32...</li><li><a href="https://labs.perplexity.ai">Perplexity Labs</a>：未找到描述</li><li><a href="https://status.perplexity.com/">Perplexity - 状态</a>：Perplexity 状态
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1270325700599091210)** (1 messages): 

> - `机械论异常检测`
> - `古怪语言模型`
> - `异常检测技术`
> - `对抗样本检测` 


- **机械论异常检测的新方法**：团队探索了使用 [Neel Nanda 的归因补丁技术](https://blog.eleuther.ai/mad_research_update/) 检测语言模型异常的“机械论”方法，但发现这些方法在性能上并未一致优于仅基于激活值的传统基准。
   - 通过评估整个批次而非单个数据点，在不同任务中取得了不同程度的成功，从而获得了更好的性能。
- **图像分类器中的对抗性检测**：Eleuther 团队指出，使用现有技术检测图像分类器中的对抗样本相对简单，尽管他们尚未验证其异常检测器是否具有对抗鲁棒性。
   - 这一发现为增强异常检测方法的鲁棒性提供了机会。
- **古怪语言模型的研究**：2023 年 12 月，团队发表了一篇关于[从古怪语言模型中诱导潜藏知识](https://arxiv.org/abs/2312.01037v3)的论文，通过微调模型，使其根据 Prompt 提示在可靠和不可靠的回答模式之间切换。
   - 该研究调查了模型行为的无监督检测，区分“Alice”型准确性和“Bob”型启发式，属于 [_机械论异常检测_](https://www.lesswrong.com/posts/n7DFwtJvCzkuKmtbG/a-gentle-introduction-to-mechanistic-anomaly-detection) 框架。
- **发布异常检测代码**：EleutherAI 团队发布了一个用于机械论异常检测的 [GitHub 仓库](https://github.com/EleutherAI/cupbearer/tree/attribution_detector)，展示了他们的方法论。
   - 该资源可能对有兴趣为该项目及其持续开发做出贡献的人有所帮助。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://blog.eleuther.ai/mad_research_update/">机械论异常检测研究更新</a>：关于机械论异常检测正在进行的工作的中期报告</li><li><a href="https://github.com/EleutherAI/cupbearer/tree/attribution_detector">GitHub - EleutherAI/cupbearer at attribution_detector</a>：一个用于机械论异常检测的库。通过在 GitHub 上创建一个账户来为 EleutherAI/cupbearer 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1270174953135935569)** (36 messages🔥): 

> - `SB1047 AI Safety Act` (SB1047 AI 安全法案)
> - `Philosophical Differences in AI Regulation` (AI 监管中的哲学分歧)
> - `Anthropic's Response to SB1047` (Anthropic 对 SB1047 的回应)
> - `Impact of Regulation on Innovation` (监管对创新的影响)
> - `Knowledge Distillation in LLMs` (LLM 中的知识蒸馏) 


- **围绕 SB1047 的辩论愈演愈烈**：成员们就 SB1047（AI 安全法案）展开了激烈讨论，一些人认为它可能会阻碍 AI 研究的创新，而另一些人则认为它是确保问责制的必要监管。
   - 批评者担心该法律可能会威慑开放式研究，重点关注其责任条款的影响以及它所带来的不确定性。
- **关于 AI 监管的哲学分歧**：对话凸显了关于政府在 AI 监督中作用的更深层次意识形态冲突，意见在监管框架的必要性与推动不受限制的研究之间产生了分歧。
   - 一些人认为，这场辩论反映了更广泛的社会分歧，即什么是良好的社会以及 AI 技术的未来。
- **对 Anthropic 观点的支持**：几位成员对 Anthropic 对 SB1047 的回应表示赞赏，称其为明智的见解，兼顾了对创新和安全的关注。
   - 该回应因解决了围绕法案实施的重要问题以及监管与研究进步之间的平衡而受到关注。
- **AI 发展中的商业化与谨慎**：讨论指出，安全 AI 实践的需求与利润驱动之间可能存在紧张关系，并建议法律激励措施可以推动更严格的研究协议。
   - 然而，成员们警告不要制定过于强制性的法律，因为这些法律可能会通过强制要求水印等不成熟的解决方案而使创新停滞。
- **征求知识蒸馏资源**：一位成员寻求有关知识蒸馏（Knowledge Distillation）以及使用大型模型训练小型 LLM 的实践资源推荐。
   - 这一询问强调了社区内对模型训练实际应用的日益增长的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.documentcloud.org/documents/25003075-sia-sb-1047-anth">DocumentCloud</a>: 未找到描述</li><li><a href="https://www.documentcloud.org/documents/25003075-sia-sb-1047-anthropic">DocumentCloud</a>: 未找到描述</li><li><a href="https://safesecureai.org/responseletter">Letter to YC &amp; a16z | SB 1047 - Safe &amp; Secure AI Innovation</a>: 未找到描述</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSewflVHn1zoNeHHJq3SaKvlwPy7PLT1Vcu_WoULqcHSSjvX1w/viewform">Students, Faculty, and Scientists Against SB 1047 (AI Safety Act) Open Letter Signature Form</a>: 这是一个用于为 UC 教职员工和学生反对加州 SB 1047 的公开信提供签名的表格，SB 1047 是一项试图监管 "AI 安全" 的灾难性法律...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1270111334914330777)** (40 条消息🔥): 

> - `Meta 的 AI 网络进展`
> - `AI 中的离散问题求解`
> - `模型推理的可扩展性`
> - `机器学习评估中的搜索技术`
> - `模型的自教评估方法` 


- **Meta 展示 AI 网络进展**：在悉尼举行的 [ACM SIGCOMM 2024](https://conferences.sigcomm.org/sigcomm/2024/) 上，Meta 发表了关于 [RDMA over Ethernet for Distributed AI Training at Meta Scale](https://dl.acm.org/doi/10.1145/3651890.3672233) 的论文，强调了支持如 **LLAMA 3.1 405B** 等大规模 AI 模型训练的基础设施。
   - 这凸显了由 AI 兴起驱动的**日益增长的通信需求**，特别是在分布式训练工作负载中。
- **关于 AI 离散问题求解的辩论**：一场正在进行的讨论围绕在 AI 搜索模型中使用离散空间与潜表征（latent representations）展开，强调了在独立采样时组合解决方案的复杂性。
   - 一位成员建议使用 **Vector Quantization** 方法来激励模型学习易于组合的子解决方案。
- **通过扩展推理计算提升性能**：一篇新论文强调了在推理过程中增加生成样本数量的潜力，以提高问题解决的覆盖率，从而在各种任务中扩展性能。
   - 结果表明，应用重复采样可以显著提高编程等任务中的问题解决率。
- **探索机器学习评估中的搜索技术**：大家达成共识，认为传统的搜索方法往往优于更复杂的可微替代方案，因为前者更简单且实用。
   - 尽管更复杂的技术更具吸引力，但观察发现，更简单的方法能产生相当甚至更优的结果。
- **创新的自教评估方法出现**：一篇论文介绍了一种评估器的自我改进方案，无需大量人工标注即可增强模型，仅使用合成训练数据。
   - 该方法将 **Llama3-70B-Instruct** 的准确率从 **75.4** 提升至 **88.3**，展示了一种动态生成训练数据的有效方式。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.21787">Large Language Monkeys: Scaling Inference Compute with Repeated Sampling</a>: 扩展用于训练语言模型的计算量已显著提高了其能力。然而，在推理时，我们通常将计算量限制在仅一次尝试中...</li><li><a href="https://arxiv.org/abs/2408.02666">Self-Taught Evaluators</a>: 基于模型的评估是成功模型开发的核心——作为训练的奖励模型，以及作为人工评估的替代。为了训练此类评估器，标准方法是...</li><li><a href="https://arxiv.org/abs/2408.00724">An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with Language Models</a>: 关于大语言模型 (LLMs) 在模型大小和计算预算方面的最优训练配置已得到广泛研究。但如何在推理过程中优化配置 LLMs...</li><li><a href="https://engineering.fb.com/2024/08/05/data-center-engineering/roce-network-distributed-ai-training-at-scale/">RoCE networks for distributed AI training at scale</a>: AI 网络在将数万个 GPU 互连在一起方面发挥着重要作用，构成了训练的基础设施，支持具有数千亿参数的大模型...</li><li><a href="https://redwoodresearch.substack.com/p/getting-50-sota-on-arc-agi-with-gpt">Getting 50% (SoTA) on ARC-AGI with GPT-4o</a>: 你只需要抽取更多样本
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1270272866587512915)** (4 messages): 

> - `Training Instability`
> - `Double Descent Phenomenon`
> - `Learning Rate Adjustments` 


- **噪声或训练不稳定性更有可能**：一位成员断言，观察到的问题更有可能是由 **噪声** 或 **训练不稳定性** 引起的，而不是 **Double Descent Phenomenon**。
   - *It's still more likely noise/training instability issues than double descent.*
- **建议对多次实验结果取平均**：建议将实验进行 **3 到 5 次** 并对结果取平均值，以获得更可靠的结果。
   - *If it’s me I would do the experiment 3 or 5 times and average the result first.*
- **降低学习率以提高稳定性**：为了提高训练稳定性，建议如果持续出现不稳定性，应 **降低学习率**。
   - *If this phenomenon still exists I would try to lower the learning rate to make training stability less likely.*
- **在初始步骤后考虑其他可能性**：该成员建议，只有在进行实验和调整学习率之后，才应该考虑其他潜在原因。
   - *Only after both I would start considering other possibilities.*


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1270115079051808778)** (5 messages): 

> - `Recent Developments in SAEs`
> - `SAE Notation and Framework`
> - `SAE Landscape Overview`
> - `SAELens Library Updates`
> - `Scaling SAEs to Real Models` 


- **跟进 SAE 的最新进展**：成员们表示有兴趣了解 SAE 的最新进展，特别是参考了 Transformer 线路线程中关于 [SAE 的综合论文](https://transformer-circuits.pub/2023/monosemantic-features/index.html)。
   - 他们还强调了关于 [扩展 SAE (Scaling SAEs)](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) 的后续工作以及相关的叠加 (superposition) 论文。
- **SAE 符号表示的当前相关性**：讨论涉及了 SAE 框架中提到的符号表示的相关性，这仍然是理解新进展的参考点。
   - 分享了扩展这些符号和方法的在研工作链接，强调了它们在真实模型中的实用性。
- **SAE 全景文档**：分享了一份 SAE 领域全景概览，汇编在 [Google 文档](https://docs.google.com/document/d/1lHvRXJsbi41bNGZ_znGN7DmlLXITXyWyISan7Qx2y6s/edit#heading=h.j9b3g3x1o1z4)中，提供了 SAE 领域的粗略概览。
   - 虽然它可能会遗漏一些最新进展，但它是理解当前 SAE 主题的基础资源。
- **SAELens 及 SAE 相关工具**：成员们讨论了 [SAELens](https://github.com/jbloomAus/SAELens) 的功能，这是一个旨在训练和分析 SAE 的库，以及它与 [auto-interp](https://github.com/EleutherAI/sae-auto-interp) 等新框架的关系。
   - 这些工具之间的集成旨在提高处理大规模模型的用户的可访问性和可视化能力。
- **协作与社区参与**：一位成员指出，有机会通过指定频道与从事 SAE 工作的团队互动，并强调了 GDM 和 OpenAI 之间的合作。
   - 这为讨论 SAE 研究和应用中面临的进展和挑战提供了环境。



**提到的链接**：<a href="https://docs.google.com/document/d/1lHvRXJsbi41bNGZ_znGN7DmlLXITXyWyISan7Qx2y6s/edit#heading=h.j9b3g3x1o1z4">SAE Landscape</a>：SAE Landscape – 语言模型可解释性稀疏自编码器 (SAEs) 资源合集。这是一个实时更新的文档，我非常感谢...

  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1270115967858380883)** (8 messages🔥): 

> - `lm-eval-harness usage`
> - `Huggingface model class`
> - `Handling special tokens`
> - `Accessing benchmark names in JSON output` 


- **使用 lm-eval-harness 评估自定义模型**：一位用户询问如何使用 **lm-eval-harness** 评估自定义模型，并收到了一个修改 Huggingface LM 类的[自包含示例](https://github.com/state-spaces/mamba/blob/main/evals/lm_harness_eval.py)链接。
   - 另一位成员补充说，**HFLM** 类支持传入已经初始化的 HF `PretrainedModel`，从而允许在自定义脚本中进行评估。
- **loglikelihood_rolling 中的 Batch Size**：一位用户询问 **loglikelihood_rolling** 在 Huggingface 模型类中是否遵循 Batch Size，并指出它似乎一次只运行一个请求。
   - 这涉及到利用 Huggingface 模型架构时批量处理的效率问题。
- **evalharness 中的 BOS Token**：一位成员寻求关于 **evalharness** 是否默认添加 **BOS** Token 的澄清，并指出 Tokenizer 的默认设置是添加特殊 Token。
   - 他们观察到生成的样本文件不包含 **BOS** Token，并希望确认其是否存在。
- **从 JSON 输出中查找基准测试名称**：一位用户询问如何从 **lm-eval-harness** 的 JSON 输出中提取基准测试名称。
   - 另一位成员建议结果 JSON 有一个 `results` 键，其中包含基准测试名称作为键，其分数为值。



**提及的链接**：<a href="https://github.com/state-spaces/mamba/blob/main/evals/lm_harness_eval.py">mamba/evals/lm_harness_eval.py at main · state-spaces/mamba</a>：Mamba SSM 架构。通过在 GitHub 上创建账号为 state-spaces/mamba 的开发做出贡献。

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1270096212292534272)** (83 messages🔥🔥): 

> - `GPU Memory Management`
> - `LangGraph Courses`
> - `SQL Chat Agent Development`
> - `Music Recommendation App`
> - `Code Review Automation` 


- **管理 GPU 显存问题**：一位用户报告在使用 **aya** 和 **nomic-embed-text** 等模型时遇到显存不足（OOM）错误，尽管拥有 **32GB RAM** 且模型小于 **4GB**。
   - 经过排查，建议在 **CPU** 而不是 **GPU** 上运行应用程序，但这导致性能显著下降。
- **LangGraph 课程推荐**：用户讨论了可用的 **LangGraph** 课程，推荐包括 **DeepLearning AI 课程** 以及 **Udemy** 上更高级的课程。
   - 共识是存在许多初学者友好的资源，但高级课程较少。
- **SQL Chat Agent 协作**：一位用户寻求关于 **SQL chat agent 脚本** 的帮助，另一位具有类似项目经验的用户提供了帮助。
   - 双方交换了脚本和反馈，表明了增强 SQL Agent 功能的协作努力。
- **新的音乐发现应用**：介绍了一款新应用 **mood2music**，它通过 AI 分析和与流媒体服务的集成，根据用户的情绪提供音乐推荐。
   - 该应用目前正在寻求用户加入候补名单，强调其在音乐策划方面的独特功能。
- **改进代码审查自动化**：一位开发人员询问如何使用 **GPT-4o** 改进自动代码审查，特别是在 GitHub diffs 中选择位置。
   - 建议使用专门的 Coding 模型而不是 Vision 模型，重点关注高效的数据解析和检索策略。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://mood2music.me">mood2music</a>：暂无描述</li><li><a href="https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/">AI Agents in LangGraph</a>：使用 LangChain 的 LangGraph 和 Tavily 的 Agentic 搜索构建 Agentic AI 工作流。直接向 LangChain 和 Tavily 的创始人学习。</li><li><a href="https://superlinked.com/vector-db-comparison">Vector DB Comparison</a>：Vector DB Comparison 是来自 VectorHub 的免费开源工具，用于比较向量数据库。</li><li><a href="https://js.langchain.com/v0.2/docs/tutorials/chatbot">Build a Chatbot | 🦜️🔗 Langchain</a>：概述</li><li><a href="https://github.com/ollama/ollama/issues/3509">Can Ollama use both CPU and GPU for inference? · Issue #3509 · ollama/ollama</a>：你想做什么？我想知道 Ollama 是否支持在 Windows 上混合使用 CPU 和 GPU 运行？我知道我的硬件不足以运行 Ollama，但我仍想使用部分能力...
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1270276986878693398)** (2 条消息): 

> - `AgentGenesis`
> - `Open Source Contributions` (开源贡献)
> - `AI Development Acceleration` (AI 开发加速)


- **AgentGenesis 助力 AI 开发**：一名成员展示了 **AgentGenesis**，这是一个 AI 组件库，为开发者提供**可复制粘贴的代码片段**，旨在将其 **Gen AI 应用**的开发效率提升 10 倍，访问地址为 [AgentGenesis](https://www.agentgenesis.dev/)。
   - 该项目采用 **MIT 许可**，强调**开发者友好型解决方案**，并为各种 AI 模板提供**全面的代码库**，欢迎社区在 [GitHub](https://github.com/DeadmanAbir/AgentGenesis) 上贡献力量。
- **协作与代码共享**：另一位成员表达了合作意向，并询问 Johnny 是否愿意分享他的代码实现。*这突显了社区的参与度以及在 AI 开发圈内分享知识的意愿。*


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.agentgenesis.dev/">AgentGenesis</a>: 复制粘贴最热门的 AI Agent，无需从头编写即可在项目中使用。</li><li><a href="https://github.com/DeadmanAbir/AgentGenesis">GitHub - DeadmanAbir/AgentGenesis: 欢迎来到 AgentGenesis，这里是你获取可定制 Gen AI 代码片段的来源，你可以轻松地将它们复制并粘贴到你的应用中。</a>: 欢迎来到 AgentGenesis，这里是你获取可定制 Gen AI 代码片段的来源，你可以轻松地将它们复制并粘贴到你的应用中。 - DeadmanAbir/AgentGenesis
</li>
</ul>

</div>
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1270171674540769311)** (57 条消息🔥🔥): 

> - `John Schulman leaves OpenAI` (John Schulman 离开 OpenAI)
> - `Anthropic developments` (Anthropic 动态)
> - `AI Alignment debates` (AI Alignment 辩论)
> - `GDB takes a sabbatical` (GDB 开始休假)
> - `Structured Outputs announcement` (Structured Outputs 发布)


- **John Schulman 离开 OpenAI 加入 Anthropic**：John Schulman 宣布离开 OpenAI，前往 [Anthropic](https://x.com/johnschulman2/status/1820610863499509855) 专注于 AI Alignment 研究，并表示希望从事更具实践性的技术工作。
   - 他强调这一选择纯属个人决定，并非因为 OpenAI 对 Alignment 缺乏支持，并指出公司依然人才济济。
- **关于 GDB 休假的猜测**：GDB 决定休假到年底，这引发了对其原因的讨论，包括对过度劳累和潜在健康问题的担忧。
   - 一些成员推测，在多年高强度投入 AGI 事业后，这对他来说可能是一个急需的喘息机会。
- **关于 AI Alignment 观点的辩论**：讨论中出现了对 AI Alignment 不同看法的争鸣，John Schulman 倾向于强化学习视角，而其他人则认为这超越了传统方法。
   - 这场辩论反映了人们对控制超人类 AI 的广泛担忧，以及 Alignment 是否从根本上是一个深度学习问题。
- **Structured Outputs 增强功能**：一项新公告详细介绍了 API 中 [Structured Outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) 的引入，允许开发者获得一致的模式（Schema）匹配，而不会丢失键值。
   - 此外，使用 gpt-4o-2024-08-06 模型的成本显著降低，为开发者节省了输入和输出成本。
- **对 AGI 和个人动机的反思**：成员们反思了 AGI 研究者背后的动机，分享了关于像 GDB 这样的人物是受意识形态驱动还是仅仅对工作充满热情的看法。
   - 这引发了关于 GDB 对使命深度投入的评论，包括传闻他在 OpenAI 办公室举行婚礼，支持了他是一个专注的“苦干者（grinder）”的观点。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/johnschulman2/status/1820610863499509855">来自 John Schulman (@johnschulman2) 的推文</a>: 我今天与 OpenAI 的同事们分享了以下内容：我做出了离开 OpenAI 的艰难决定。这个选择源于我希望加深对 AI Alignment 的关注，并开始...</li><li><a href="https://fxtwitter.com/simonw/status/1820886987982987413?s=46">来自 Simon Willison (@simonw) 的推文</a>: 隐藏在公告底部的内容：通过切换到新的 gpt-4o-2024-08-06，开发者可以节省 50% 的输入成本（$2.50/1M input tokens）和 33% 的输出成本（$10.00/1M output tokens）...</li><li><a href="https://x.com/gdb/status/1820644694264791459?s=46">来自 Greg Brockman (@gdb) 的推文</a>: 我将休假到年底。这是自 9 年前共同创立 OpenAI 以来第一次放松。使命远未完成；我们仍需构建安全的 AGI。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1270539473066922105)** (6 messages): 

> - `DALL·E vs New Challengers`
> - `Flux Pro`
> - `Flux.1 Hosting on Replicate`
> - `Comparative Analysis of Image Generators` 


- **DALL·E 仍处于领先地位吗？**: 随着新竞争者的出现，讨论围绕 **DALL·E 是否仍保持着**通过 API 进行图像生成的最佳头衔展开。
   - *一位成员对竞争态势提出了疑问*，并指出除了直觉之外，进行直接对比存在挑战。
- **Flux Pro 带来了不同的氛围**: **Flux Pro** 被描述为具有*非常不同的氛围*，暗示其在图像生成方面采用了独特的方法。
   - 一位成员对这种氛围如何转化为实际性能和用户体验表示好奇。
- **Flux.1 在 Replicate 上获得关注**: **新的 Flux.1** 模型引起了关注，并且已在 [Replicate](https://replicate.com) 上托管。
   - 成员们讨论了模型对比的难度，并强调了在 **Replicate 上进行视频**进一步实验的潜力。


  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/)** (1 messages): 

xeophon.: https://x.com/sahir2k/status/1820791954508022019?s=46
  

---


### **Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1270459908806017076)** (1 messages): 

> - `Data Dependence in Model Training`
> - `Startups Utilizing Noise`
> - `Armen from Chameleon at Meta` 


- **数据依赖是关键**: 所有观点都围绕着**一切都取决于数据**这一事实；将 (x, y_w, y_l) 拆分为 (x, y_w) 和 (x, y_l) 仅能从噪声数据中获益。
   - *足够多噪声的数据*对于有效的模型训练至关重要，强调了数据使用中上下文的重要性。
- **初创公司青睐噪声数据**: 初创公司似乎倾向于采用允许使用*更高噪声数据*的数据策略，这通常使他们能够跳过 **SFT (Supervised Fine-Tuning)**。
   - 这种偏好表明初创公司环境中存在拥抱更灵活数据处理方式的显著趋势。
- **Armen 对数据方法的影响**: ICML 上的一次提及强调了来自 **Meta** **Chameleon** 团队的 **Armen** 对这些数据策略表现出浓厚兴趣。
   - 然而，目前尚不清楚他的团队是否正在**生产模型**中实施这些想法。


  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1270486884539174973)** (1 messages): 

> - `GPT-4o-2024-08-06`
> - `Structured outputs in strict mode` 


- **GPT-4o-2024-08-06 新版本发布**: 最新版本 [GPT-4o-2024-08-06](https://openrouter.ai/models/openai/gpt-4o-2024-08-06) 现已上线。
   - **OpenRouter, LLC** 提供了此次更新，并指出这是其持续改进工作的一部分。
- **结构化输出的有限支持**: 有说明指出，目前**尚未完全支持**严格模式下的结构化输出。
   - 鼓励用户在指定频道反馈问题：<#1138521849106546791> 或 <#1107397803266818229>。



**提到的链接**: <a href="https://openrouter.ai/models/openai/gpt-4o-2024-08-06">GPT-4o (2024-08-06) - API, Providers, Stats</a>: 2024-08-06 版本的 GPT-4o 在结构化输出方面提供了改进的性能，并能够在 response_format 中提供 JSON schema。阅读更多 [这里](https://openai. Run GPT-4o (2024-08...

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1270098420023164939)** (62 条消息🔥🔥): 

> - `Gemini Pro 1.5 性能问题`
> - `OpenRouter API 使用`
> - `gpt-4o-2024-08-06 更新`
> - `API 中的结构化输出 (Structured outputs)`
> - `模型价格变动` 


- **Gemini Pro 1.5 遇到资源耗尽问题**：用户报告 **Gemini Pro 1.5** 出现“资源已耗尽 (Resource has been exhausted)”的错误，这归因于 Google 严格的速率限制 (rate limiting)。
   - 已确认此问题目前无法解决，因为这是由于 Google 对该模型实施了严格限制。
- **了解 OpenRouter 的模型 API**：一位成员询问如何购买模型，并获知在 **OpenRouter** 上是通过 API 访问模型，并按 token 使用量付费。
   - 建议新用户探索 **Lobe Chat** 等用户界面，以简化与 API 的交互。
- **gpt-4o-2024-08-06 更新**：**gpt-4o-2024-08-06** 模型使开发者能够节省成本，与之前的版本相比价格显著降低，据称输入端便宜了 **50%**，输出端便宜了 **33%**。
   - 用户还对新的“拒绝 (refusal)”字段功能感到兴奋，并正在讨论如何提高模型运行效率。
- **OpenAI API 引入结构化输出 (Structured outputs)**：OpenAI 引入了结构化输出，使开发者能够直接从 **API** 请求有效的 JSON 响应，从而增强了可靠性。
   - 以前的方法一致性较差，而新方法旨在标准化输出并提高跨应用程序的可用性。
- **模型定价和 token 限制**：关于 **gpt-4o-2024-08-06** 的 **token 限制** 差异存在讨论，OpenRouter 最初显示的上限低于 OpenAI 文档中的数值。
   - 用户正等待更新以反映最新模型的准确能力，预计很快会有更新。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://chat-preview.lobehub.com>,">未找到标题</a>: 未找到描述</li><li><a href="https://simonwillison.net/2024/Aug/6/openai-structured-outputs/">OpenAI：在 API 中引入结构化输出</a>: OpenAI 已经提供结构化输出一段时间了：你可以指定 `&quot;response_format&quot;: {&quot;type&quot;: &quot;json_object&quot;}}` 来请求一个有效的 JSON 对象，或者你可以使用...</li><li><a href="https://openrouter.ai/docs/responses#querying-cost-and-stats">响应 | OpenRouter</a>: 管理来自模型的响应</li><li><a href="https://status.anthropic.com">Anthropic 状态</a>: 未找到描述</li><li><a href="https://openrouter.ai/models/openai/gpt-4o-2024-08-06">GPT-4o (2024-08-06) - API、提供商、统计数据</a>: 2024-08-06 版本的 GPT-4o 在结构化输出方面提供了改进的性能，能够在 respone_format 中提供 JSON schema。阅读更多 [此处](https://openai. 运行 GPT-4o (2024-08...
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1270432568948555776)** (1 条消息): 

> - `CodiumAI 网络研讨会`
> - `RAG 增强的编程助手`
> - `LlamaIndex 基础设施` 


- **参加关于 RAG 增强代码生成的 CodiumAI 网络研讨会**：分享了关于即将举行的 [CodiumAI 网络研讨会](https://lu.ma/ka5xtyqo) 的提醒，重点是 **RAG 增强的编程助手**。
   - 参与者需要通过钱包验证 token 所有权才能参加活动。
- **探索 RAG 在代码质量中的作用**：研讨会将深入探讨 **检索增强生成 (RAG)** 如何增强 AI 生成代码的上下文感知能力。
   - *上下文感知* 对于采用代码生成的企业在开发中保持**高质量**和**完整性**至关重要。
- **使用 LlamaIndex 的高级 RAG 方法**：与会者可以期待一场关于基于 **LlamaIndex** 基础设施的 **高级 RAG 方法** 的演讲。
   - 还将展示演示 *上下文感知生成* 的实际应用案例。



**提到的链接**: <a href="https://lu.ma/ka5xtyqo">LlamaIndex 网络研讨会：在大型生成式编程中结合使用 RAG 与 LlamaIndex · Zoom · Luma</a>: 检索增强生成 (RAG) 在实现 AI 生成代码的上下文感知方面发挥着核心作用，这对于采用……的企业至关重要。

  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1270177453498830960)** (4 条消息): 

> - `使用 RabbitMQ 构建本地 Multi-Agent 系统`
> - `LlamaIndex RAG-a-thon`
> - `Workflows 功能介绍`
> - `llama-agents 文档` 


- **使用 RabbitMQ 构建本地 Multi-Agent 系统**：由 [@pavan_mantha1](https://twitter.com/pavan_mantha1) 撰写的博客展示了如何利用 [RabbitMQ](https://www.rabbitmq.com) 进行 Agent 间通信，并结合 [ollama](https://ollama.com) 和 [qdrant_engine](https://qdrant.tech) 来创建本地 Multi-Agent 系统。该方案由 llama-agents 提供支持，它是 Agent 开发的关键工具。
   - 查看完整指南 [点击这里](https://t.co/IOGpDWkY8A)。
- **LlamaIndex 举办第二届 RAG-a-thon**：LlamaIndex 正与 [@pinecone](https://www.pinecone.io) 和 [@arizeai](https://www.arize.ai) 合作举办第二届 RAG-a-thon，活动将于 10 月 11 日至 13 日在帕罗奥图的 [@500GlobalVC](https://www.500.co) 办公室举行。这场周末黑客松旨在吸引该领域的开发者和创新者。
   - 更多详情请见 [这里](https://t.co/N4hWiCv0Nm)。
- **推出 Workflows 功能**：在一段新的 [YouTube 视频](https://youtu.be/xuiuSMCmJF) 中，[@seldo](https://twitter.com/seldo) 介绍了 Workflows 功能，该功能允许用户在 LlamaIndex 中构建复杂的 Agentic 应用。视频涵盖了创建、运行和可视化 Workflows 的关键要素。
   - 它还讨论了 Workflow 的结构、循环（looping）、分支（branching）和状态管理（state management），为开发者提供了重要的见解。
- **llama-agents 详尽文档**：一篇题为《构建 Multi-agents as a Service 入门指南》的新指南响应了用户对 llama-agents 更好文档的需求。得益于 [@_nerdai_](https://twitter.com/_nerdai_) 的贡献，该项目近期取得了重大进展。这个核心仓库对于将 Multi-agents 开发为服务至关重要。
   - 欲了解更多信息，请查看入门指南 [这里](https://t.co/k0TEeMi3C5)。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1270128013337432115)** (49 条消息🔥): 

> - `用于 Embeddings 的 HuggingFace Inference API`
> - `Llamaparse 阿拉伯语解析`
> - `SimpleDirectoryReader PDF 处理`
> - `Vector DB 对比资源`
> - `4o Mini 与 3.5 Turbo 的性能对比` 


- **使用 HuggingFace Inference API 获取 Embeddings**：要使用 HuggingFace Inference API 生成 embeddings，可以使用 `TextEmbeddingsInference` 类，如提供的 [LlamaIndex 示例](https://docs.llamaindex.ai/en/stable/examples/embeddings/text_embedding_inference/)所示。
   - 此设置包含模型名称和 embedding 批处理大小等参数，以实现高效处理。
- **Llamaparse 在阿拉伯语解析方面存在困难**：用户注意到 Llamaparse 的阿拉伯语解析返回的结果是从左到右（Left to Right）的格式，尽管阿拉伯语是一种从右到左（Right to Left）的语言。
   - 这引发了关于 Llamaparse 是否适配从右到左书写风格复杂性的疑问。
- **在 SimpleDirectoryReader 中管理 PDF 文档加载**：`SimpleDirectoryReader` 将 PDF 加载为多个文档（每页一个），以便为每一页包含 `page_label` 等元数据。
   - 可以对 `PDFReader` 进行修改，以便在加载过程中将内容聚合到单个文档中。
- **分享 Vector DB 对比资源**：分享了一个用于对比 Vector DB 的实用资源，一些用户认为这对他们的项目非常有价值。
   - 其他人则表示有兴趣积累各种 Vector DB 的用户经验，以便共同学习。
- **4o Mini 与 3.5 Turbo 之间的性能差异**：用户报告称 4o Mini 的运行速度明显慢于 3.5 Turbo，后者在速度方面仍被认为更胜一筹。
   - 讨论围绕可能影响初始响应时间的潜在后端扩展问题展开，特别是在首个 token 生成时间（time to first token）方面。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://superlinked.com/vector-db-comparison">Vector DB Comparison</a>：Vector DB Comparison 是来自 VectorHub 的一个免费开源工具，用于对比向量数据库。</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/embeddings/text_embedding_inference/">Text Embedding Inference - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/6eea66ed23fb85ee77664148a4c2b66720caabeb/pyproject.toml#L60">llama_index/pyproject.toml at 6eea66ed23fb85ee77664148a4c2b66720caabeb · run-llama/llama_index</a>：LlamaIndex 是适用于你的 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/15227173b8c1241c9fbc761342a2344cd90c6593/llama-index-core/llama_index/core/llms/function_calling.py#L125">llama_index/llama-index-core/llama_index/core/llms/function_calling.py at 15227173b8c1241c9fbc761342a2344cd90c6593 · run-llama/llama_index</a>：LlamaIndex 是适用于你的 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/15227173b8c12">GitHub - run-llama/llama_index at 15227173b8c1241c9fbc761342a2344cd90c6593</a>：LlamaIndex 是适用于你的 LLM 应用程序的数据框架 - GitHub - run-llama/llama_index at 15227173b8c1241c9fbc761342a2344cd90c6593
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1270216114999398435)** (29 messages🔥): 

> - `LLM Hallucination Index`
> - `Open Source 定义`
> - `Mistral Open Weights`
> - `Command R Plus 许可`
> - `商业使用限制` 


- **LLM Hallucination Index 更新**：[LLM Hallucination Index](https://www.rungalileo.io/hallucinationindex?utm_medium=paid&utm_source=alpha_signal&utm_campaign=sp) 评估了领先模型遵循上下文的程度，并强调了对“幻觉（hallucinations）”的关注，该词已被评为年度词汇。
   - 成员们对该指数中关于 **Command R Plus** 的准确性提出质疑，一些人认为该指数误导了其 Open Source 状态。
- **关于 Open Source 定义的困惑**：几位成员对 Hallucination Index 提供的 Open Source 定义表示反对，称仅发布权重（weights）不足以构成真正的 Open Source 状态。
   - *一些人建议，为了实现完全的透明度，还应公开数据集和训练方法等额外细节*。
- **Mistral Open Weights 的明确性**：成员们讨论了 **Mistral** 模型是在 **Apache 2.0** 许可证下运行的，这意味着尽管在数据集访问方面存在限制，但它们符合 Open Source 的资格。
   - 然而，围绕 AI 领域 Open Source 的真正定义仍存在争论，强调目前许多可用的模型仅是具有各种使用条款的“Open Weights”。
- **Command R Plus 的商业使用问题**：成员们指出 **Command R Plus** 并非 Open Source，因为它是在 Creative Commons Attribution Non Commercial 4.0 许可证下运行的。
   - 这引发了关于论文中 Open Source 定义不足的辩论，一些成员计划联系相关方以寻求澄清。
- **关于许可证影响的讨论**：一位成员总结道，虽然 **Command R Plus** 模型拥有“Open Weights”，但非商业限制实际上使其在实践中变成了闭源（closed-source）。
   - *讨论突显了 AI 许可的复杂性，其中 Open Weights 与真正的 Open Source 之间的区别可能很模糊。*


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.mistral.ai/getting-started/open_weight_models/">Open weight models | Mistral AI Large Language Models</a>：我们开源了预训练模型和指令微调模型。这些模型没有针对安全性进行微调，因为我们希望赋予用户根据其用例测试和完善审核的能力。为了更安全...</li><li><a href="https://www.rungalileo.io/hallucinationindex?utm_medium=paid&utm_source=alpha_signal&utm_campaign=sp">LLM Hallucination Index - Galileo</a>：LLM Hallucination Index。一个针对 LLM 幻觉的排名和评估框架。</li><li><a href="https://www.rungalileo.io/hallucinationindex?utm_medium=paid&utm_source=alpha_signal&utm_campaign=sponsorship">LLM Hallucination Index - Galileo</a>：LLM Hallucination Index。一个针对 LLM 幻觉的排名和评估框架。
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1270432110078853241)** (3 messages): 

> - `联系 Dennis Padilla`
> - `Lauren 缺席`
> - `邮件咨询` 


- **寻求 Dennis Padilla 的电子邮件**：一位成员请求协助获取 **Dennis Padilla** 的电子邮件地址，因为他在 **Lauren** 休假期间被指示与其联系。
   - 该成员表示难以找到联系信息，并正在寻求如何取得联系的指导。
- **关于 Lauren 休假的讨论**：**Lauren 的缺席**被注意到，因为她目前正在度假，这引发了对其替代联系人的询问。
   - 该成员提到依靠 **Dennis Padilla** 作为他们后续沟通的下一个联系点。


  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1270539094203568128)** (1 messages): 

> - `用于学习的 Cohere Toolkit`
> - `带有 RAG 的 LLM`
> - `模型部署`
> - `第三方 API 集成` 


- **Cohere Toolkit 助力学习计划**：Cohere 团队正利用 **Cohere Toolkit** 开展一个作为 AI fellowship 一部分的学习项目，旨在构建一个基于知识库的 **带有 RAG 的 LLM**。
   - 他们正在探索不同类型语料库的潜力，例如**食谱**、**烹饪笔记**和**法律案例笔记**。
- **模型部署疑问出现**：一位成员询问是否有人成功地从 **Cohere 模型**切换到第三方的 **API-based** 模型，如 **OpenAI Chat GPT** 或 **Gemini 1.5**。
   - 兴趣点包括通过 **Groq API** 访问的模型，突显了对更广泛能力的追求。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1270321995027972166)** (30 条消息🔥): 

> - `InlineList 特性`
> - `Mojo 中 List 的优化`
> - `自定义硬件加速器`
> - `CXL 协议与 FPGA 集成`
> - `Mojo 的开源未来` 


- **InlineList 缺少移动和拷贝初始化**：`InlineList` 目前没有 `__moveinit__` 和 `__copyinit__`，但目前已取得重大进展，重要特性很快就会合并。
   - 一位成员指出，在添加这些功能之前，涉及 `InlineList` 的开发工作被优先处理。
- **List 与 InlinedFixedVector 的区别**：一位成员澄清说，`InlinedFixedVector` 旨在用于 `AnyTrivialRegType`，而 `List` 则服务于 `CollectionElement`，解释了它们在 Mojo 中不同的用途。
   - 讨论还涉及了一个正在进行的 Pull Request，旨在为 `List` 提供潜在的小缓冲区优化（small buffer optimization）。
- **Mojo 支持自定义加速器的潜力**：成员们讨论了使用 PCIe 卡等自定义加速器，以及 Mojo 是否会在开源发布前支持它们，并对性能问题发表了见解。
   - 有人提到，cxl.mem 功能可能需要依赖硬件层面的集成，以确保性能效率。
- **关于 FPGA 和 CXL IP 模块的讨论**：成员们交流了关于 Xilinx VU13P FPGA 硬件开发的见解，提到了 CXL IP 模块的整合。
   - 一位成员分享了他们正在进行的项目计划，即通过使用自定义程序替换内核来优化内核使用。
- **对 Mojo 开源能力的期待**：人们对 Mojo 开源后的未来潜力感到兴奋，特别是对支持 RISC-V 向量扩展的支持。
   - 尽管目前在兼容性方面存在局限，成员们仍希望 Mojo 能对他们的项目产生助益。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/modula">modula - 概览</a>：GitHub 是 modula 构建软件的地方。</li><li><a href="https://github.com/modularml/mojo/pull/2825">[stdlib] 为 `List` 添加可选的小缓冲区优化，第 2 次尝试，由 gabrieldemarmiesse 提交 · Pull Request #2825 · modularml/mojo</a>：此 PR 解决了 #2467 的部分问题。此 PR 是三个 PR 中的一部分，请按以下顺序阅读和合并：[stdlib] 为 List 添加可选的小缓冲区优化 #2825...
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1270190953902374913)** (18 条消息🔥): 

> - `John Schulman 加入 Anthropic`
> - `开源 AI 的挑战`
> - `Meta JASCO 的状态`
> - `Nullbulge 开盒（人肉搜索）事件`
> - `语音助手项目` 


- **John Schulman 离开 OpenAI 加入 Anthropic**：OpenAI 联合创始人 **John Schulman** 在周一的一篇帖子中宣布，他将加入由 **Amazon** 支持的 AI 初创公司 **Anthropic**。
   - 此举发生在 OpenAI 最近解散其 **超级对齐团队（superalignment team）** 之后，该团队致力于确保对先进 AI 系统的控制。
- **开源 AI 发展面临的挑战**：**开源 AI** 发展滞后，原因是训练 SOTA 模型的成本高昂，且收集对开发至关重要的偏好数据面临挑战。
   - 缺乏对未授权数据的访问权限加剧了这些限制，导致开发的开源模型减少。
- **Meta JASCO 的传闻**：关于 **Meta JASCO** 的传闻四起，有评论称其已失踪，并对来自 **Udio** 和 **Suno** 的潜在诉讼表示担忧。
   - 这种情况似乎导致 Meta 在 AI 领域的进展出现了犹豫。
- **Nullbulge 被开盒事件**：据报道 **Nullbulge** 被开盒（doxxed），引发了关于个人隐私和在线声誉影响的讨论。
   - 聊天参与者认为，由于其在操作安全（OPSEC）方面的弱点，这在未来可能不会构成重大问题。
- **YouTube 上的语音助手项目**：分享了一个 YouTube 链接，视频标题为 **


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.cnbc.com/2024/08/06/openai-co-founder-john-schulman-says-he-will-leave-and-join-rival-anthropic.html">OpenAI 联合创始人 John Schulman 表示他将离职并加入竞争对手 Anthropic</a>：Schulman 表示，OpenAI 的高管们仍致力于支持确保人类能够控制高能力人工智能模型的努力。</li><li><a href="https://youtu.be/DdAwEdlVi14">School BUD-E 浏览器语音助手</a>：未找到描述</li><li><a href="https://archive.ph/TmDrg">三位领导者离开 OpenAI — The Information</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1270444227205009478)** (8 messages🔥): 

> - `Model Scaling`
> - `Validation Accuracy`
> - `CIFAR Image Processing` 


- **模型在 270k 参数时达到准确率瓶颈**：**270k 模型**似乎遇到了与更小模型几乎相同的准确率瓶颈，报告的**验证准确率（Validation Accuracy）为 84%**。
   - *我开始相信*，尽管参数量增加了，但这种设置表现出了收益递减。
- **在频域探索 CIFAR 图像**：一名成员询问了 **CIFAR 图像**在通过频域变换（FTT）处理时的表现。
   - 他们想知道频率信息是否保持一致而相位发生变化，寻求关于差异的见解。



**Link mentioned**: <a href="https://tenor.com/view/the-matrix-laurence-fishburne-morpheus-trinity-he%27s-beginning-to-believe-gif-18413151103009905935">The Matrix Laurence Fishburne GIF - The matrix Laurence fishburne Morpheus - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1270096760634736660)** (8 messages🔥): 

> - `Running tinygrad on Aurora`
> - `XMX support and OpenCL on Intel GPUs`
> - `NVIDIA FP8 Bounty Support`
> - `Distributed computing functionality for tinygrad`
> - `Aurora supercomputer performance expectations` 


- **在 Aurora 上运行 tinygrad 的可行性**：一名成员质疑，考虑到 Intel GPU 的限制，在**阿贡国家实验室（Argonne National Laboratory）**的 **Aurora** 上运行 **tinygrad** 是否具有远程可行性。
   - 另一名成员指出，这些 GPU 支持与 **A770s** 相当的 Tensor Core 指令。
- **关于 XMX 支持和 OpenCL 的见解**：讨论了 Aurora 上支持 **XMX** 的潜力，并指出虽然存在 **OpenCL** 兼容性，但速度可能较慢。
   - 分享了关于 Intel Max Data Center GPU 和 subgroup matrix 函数可能发挥作用的具体细节。
- **需要分布式计算功能**：有人提到，为了改进 tinygrad，需要更成熟的**分布式计算（Distributed Computing）**功能。
   - 对话强调了增强分布式能力以利用 Aurora 潜力的重要性。
- **FP8 NVIDIA 悬赏需要双重支持**：一名成员询问 FP8 NVIDIA 悬赏是否需要支持 **E4M3**、**E5M2** 或两者。
   - Georgehotz 给予了肯定回答，表示最好**两者**都支持。
- **Aurora 超级计算机的能力**：一名成员指出 **Aurora** 预计将超过 **2 ExaFLOPS**，使其有可能成为有史以来最快的计算机。
   - 欲了解更多详情，对话引用了 [Aurora 的 Wikipedia 页面](https://en.wikipedia.org/wiki/Aurora_(supercomputer))，解释了其性能和用途。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html">cl_intel_subgroup_matrix_multiply_accumulate</a>: no description found</li><li><a href="https://en.wikipedia.org/wiki/Aurora_(supercomputer)">Aurora (supercomputer) - Wikipedia</a>: no description found
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1270095328476921857)** (16 messages🔥): 

> - `Preallocating Tensors`
> - `Buffer and DEFINE_GLOBAL Mapping`
> - `Batch Size Handling in JIT`
> - `Computer Algebra Study Notes`
> - `OpenMP in CLANG and LLVM` 


- **有效地预分配 Tensor**：有人建议，预分配然后赋值给切片（slice）可能会解决在 Tensor 操作中遇到的问题。
   - *George* 确认，使 Tensor 连续（contiguous）应该有助于解决该问题。
- **识别 DEFINE_GLOBAL 的 Buffer**：讨论集中在如何在 Tensor 操作期间将 `Buffer` 实例映射回其对应的 `DEFINE_GLOBAL`。
   - *Eigenvector42* 对从 Tensor 到 Buffer 再到 MemBuffer 的流程表示不确定。
- **在 JIT 中处理 Batch Size**：一位用户提出了关于 Batch Size 一致性导致 JIT 错误的担忧，这是由于数据集细分问题引起的。
   - *George* 建议跳过最后一个 batch 或在该 batch 上避免使用 JIT 作为解决方案。
- **计算机代数新资源**：分享了一个计算机代数学习笔记的链接，强调了其在阅读符号数学（symbolic math）后的理论理解相关性。
   - 笔记可在 [GitHub](https://github.com/mesozoic-egg/computer-algebra-study-notes/blob/main/README.md) 上找到。
- **CLANG 和 LLVM 线程**：*Cecilian* 询问了 CLANG 和 LLVM 中的线程问题，并得到确认它们主要使用单线程。
   - 引用了使用 OpenMP 的潜在增强功能，并附有 *tinygrad* GitHub pull requests 的链接。



**提到的链接**：<a href="https://github.com/mesozoic-egg/computer-algebra-study-notes/blob/main/README.md">computer-algebra-study-notes/README.md at main · mesozoic-egg/computer-algebra-study-notes</a>：通过在 GitHub 上创建账号，为 mesozoic-egg/computer-algebra-study-notes 的开发做出贡献。

  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1270433513476456580)** (6 messages): 

> - `Wiseflow tool`
> - `HybridAGI release`
> - `Dynamic knowledge base` 


- **介绍用于信息挖掘的 Wiseflow**：[Wiseflow](https://github.com/TeamWiseFlow/wiseflow) 是一款新型敏捷信息挖掘工具，可从网站和社交平台等各种来源提取简洁消息，并自动将其分类到数据库中。
   - 该工具旨在为从事信息密集型环境的用户增强数据组织和检索。
- **动态知识库讨论**：一位成员结合正在进行的项目背景化了“动态知识库”，暗示了与现有工具的集成。
   - 该成员引起了其他人对该想法是否有任何实际演示产出的好奇。
- **HybridAGI 增强功能发布**：最新版本的 [HybridAGI](https://github.com/SynaLinks/HybridAGI) 引入了一个围绕图（graphs）和图程序合成（graph-program synthesis）构建的神经符号（neuro-symbolic）系统，具有各种用于通过 DSPy 优化 RAG (Retrieval-Augmented Generation) 的 notebook。
   - 改进重点在于可用性和数据处理流水线，为处理知识图谱（Knowledge Graphs）提供了简化的界面。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/TeamWiseFlow/wiseflow">GitHub - TeamWiseFlow/wiseflow: Wiseflow 是一个敏捷的信息挖掘工具，可以从网站、微信公众号、社交平台等各种来源提取简洁的消息。它会自动分类并上传到数据库。</a>：Wiseflow 是一个敏捷的信息挖掘工具，可以从网站、微信公众号、社交平台等各种来源提取简洁的消息。它会自动分类并...</li><li><a href="https://github.com/SynaLinks/HybridAGI">GitHub - SynaLinks/HybridAGI: 可编程的基于 Cypher 的神经符号 AGI，允许你使用基于图的提示编程（Graph-based Prompt Programming）来编写其行为：适用于希望 AI 按预期运行的人</a>：可编程的基于 Cypher 的神经符号 AGI，允许你使用基于图的提示编程来编写其行为：适用于希望 AI 按预期运行的人 - SynaLinks/HybridAGI
</li>
</ul>

</div>
  

---

### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1270284030604218450)** (2 messages): 

> - `Large Language Models in Software Engineering`
> - `Scaling Inference Compute for Language Models` 


- **LLM 正向着带有 Agent 的 AGI 演进**：研究探讨了从 Large Language Models (LLMs) 向 **基于 LLM 的 Agent** 的转变，旨在解决缺乏自主性和自我改进等局限性，如该[研究](https://arxiv.org/abs/2408.02479)所述。
   - 在这一早期探索阶段，强调了建立统一标准和基准测试以评定 LLM 解决方案是否符合 Agent 要求的必要性。
- **推理计算提升模型性能**：最近的一项研究强调，增加推理过程中生成的样本数量可以显著提高性能。在 [SWE-bench Lite](https://arxiv.org/abs/2407.21787) 环境下，通过增加采样，问题解决率从 **15.9%** 跃升至 **56%**。
   - 这种方法证明了 *coverage*（覆盖率）随样本数量扩展，直接使代码编写和形式化证明等领域受益。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.21787">Large Language Monkeys: Scaling Inference Compute with Repeated Sampling</a>: 扩展用于训练语言模型的计算量显著提高了它们的能力。然而，在推理时，我们通常将计算量限制在仅一次尝试...</li><li><a href="https://arxiv.org/abs/2408.02479">From LLMs to LLM-based Agents for Software Engineering: A Survey of Current, Challenges and Future</a>: 随着大语言模型 (LLMs) 的兴起，研究人员正越来越多地探索它们在各种垂直领域（如软件工程）中的应用。LLMs 已经取得了显著的成功...
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1270224404428623963)** (7 messages): 

> - `MIPRO vs BootstrapFewShotWithRandomSearch`
> - `MIPROv2 assertions`
> - `Complexity in model training` 


- **MIPRO 通常优于 BootstrapFewShotWithRandomSearch**：关于 **MIPRO** 是否总是比 **BootstrapFewShotWithRandomSearch** 表现更好的询问得到了回应，指出它“经常，但不一定”表现更好。
   - 这表明虽然 MIPRO 具有强大的性能，但并不能保证结果一定更好。
- **MIPROv2 尚不支持 assertions**：一位成员询问 **MIPROv2** 是否支持 assertions，得到的回答是明确的“暂不支持”。
   - 这暗示未来的开发或更新可能会包含对 assertion 的支持。
- **模型训练应从简单开始**：有人建议“始终从简单开始”，推荐在进阶到 **MIPRO** 之前先尝试 **random search**。
   - 这作为一种策略，通过逐渐增加模型训练的复杂性来获得更高效的结果。


  

---


### **DSPy ▷ #[colbert](https://discord.com/channels/1161519468141355160/1250300504462856265/)** (1 messages): 

gamris: 你会推荐使用 Qdrant 的 FastEmbed 吗？ https://github.com/qdrant/fastembed
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1270177649754374165)** (7 messages): 

> - `Synthetic Data Generation`
> - `Llama Index SQL Examples`
> - `LoRA Adapter MD5 Consistency`
> - `BitsAndBytes GitHub Pull Request` 


- **推理任务的合成数据生成策略**：一位成员询问了关于合成数据生成的有效策略，旨在通过 **Chain of Thought (CoT)** 训练来提升 **8b models** 在 **text to SQL** 等推理任务上的表现。
   - 他们建议在生成最终 SQL 查询之前利用合成指令可能会增强性能。
- **Llama Index 中的 SQL 示例**：一位成员注意到 **Llama Index** 包含多个 SQL 示例，这对需要资源的人可能会有帮助。
   - 这一参考可以帮助其他寻求在项目或实验中实现 SQL 的人。
- **LoRA adapter 的 MD5 哈希一致性**：讨论围绕多次合并 **LoRA adapter** 时对 **MD5 hashes** 的预期展开，重点在于哈希值是否应保持一致。
   - 一位成员确认它们确实应该相同，并指出差异将意味着存在问题。
- **关注 BitsAndBytes GitHub 的正确分支**：另一位成员强调了关注 GitHub 上特定分支的重要性，并引用了来自 **BitsAndBytes Foundation** 的一个 Pull Request ([#1220](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1220))。
   - 对于从事相关开发或故障排除的人员来说，这可能是关键信息。


  

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1270107609386582199)** (5 条消息): 

> - `针对 Gemma 2 27b 的 QLoRA`
> - `在 L40S GPU 上的性能`
> - `用于 Docker 的更快的 pip` 


- **微调 Gemma 2 27b 训练的 QLoRA**：提到针对 **Gemma 2 27b** 的 **QLoRA** 可能需要调整 **learning rate**（学习率），但应与最新的 **Flash Attention** 兼容。
   - *Colejhunter* 表示他们打算尝试一下。
- **在 L40S GPU 上表现尚可**：一位成员指出，在 **L40S GPU** 上进行训练的效果相当不错，这引发了对具体细节的好奇。
   - 这一回应是在其他成员询问 **performance metrics**（性能指标）之后提出的。
- **更快的 Pip 用于 Docker 构建的潜力**：分享了一个关于 **Faster pip** 资源的链接，暗示这可能对 **Docker** 构建有利。
   - 该成员对其在简化构建流程方面的效用表示乐观。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1270197885526347957)** (3 条消息): 

> - `微调上下文长度`
> - `RoPE 缩放`
> - `在 Python 中编辑唯一样本` 


- **在微调期间调整上下文长度**：一位成员询问是否可以在微调至 **4k** 后调整像 **llama2-13b-hf** 这样微调过的模型的上下文长度。
   - 另一位成员确认*你可以根据需要增加或减少它*，并建议对于大幅度的增加采用逐步调整的方法。
- **RoPE Scaling 作为快速修复方案**：针对上下文长度问题，一位成员暗示可以使用 **RoPE scaling** 进行快速调整。
   - 他们指出，*如果你计划大幅增加长度*，最好逐步进行以获得最佳性能。
- **编辑唯一样本的不确定性**：一位成员对编辑唯一样本的清晰度表示好奇，提到之前的工具需要 **Python** 干预。
   - 目前尚不确定类似的编辑需求是否适用于正在讨论的当前工具。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[announcements](https://discord.com/channels/1104757954588196865/1113462842436354149/)** (1 条消息): 

caseus_: Office hours（答疑时间）将在一小时后在 <#1268285745555308649> 开始。
  

---



### **Torchtune ▷ #[announcements](https://discord.com/channels/1216353675241590815/1216353675241590818/1270120968878293013)** (1 条消息): 

> - `PPO 训练配方`
> - `Qwen2 模型支持`
> - `针对 torchtune 的功能请求` 


- **在 Torchtune 中引入 PPO 训练配方**：**Torchtune** 中添加了端到端的 **PPO 训练配方**，使模型具备 **RLHF** 能力。在此处查看详细实现 [here](https://github.com/pytorch/torchtune/pull/1005)。
   - 此项更新简化了强化学习与 **Torchtune** 工具包之间的集成，增强了训练选项。
- **训练配方已支持 Qwen2 模型**：对 **Qwen2 模型** 的支持已集成到 **Torchtune** 的训练配方中，目前 **7B** 模型已可用。即将发布的 **1.5B** 和 **0.5B** 版本将进一步增加工具包的通用性，详见 [here](https://github.com/pytorch/torchtune/pull/1143)。
   - 这一扩展为社区内的模型实验和微调提供了更多可能性。
- **征集 Torchtune 的功能请求**：团队邀请用户提交他们希望在 **Torchtune** 中看到的新模型或配方的 **feature requests**（功能请求）。可以通过 [GitHub 仓库](https://github.com/pytorch/torchtune) 分享反馈。
   - 这展示了对社区参与和 **Torchtune** 生态系统改进的持续承诺。


  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1270207545344135304)** (9 messages🔥): 

> - `Llama 3 的 DPO 支持`
> - `模型输出的差异`
> - `模型下载方法`
> - `Instruct 模型的 Prompt 格式化` 


- **计划为 Llama 3 提供 DPO 支持**：一名成员询问了 **Llama 3 8B 全量微调（full finetune）** 支持 **DPO** 的可能性，表达了对模型增强的兴趣。
   - 另一名成员分享道，即使没有预构建的配置，任何模型都可以配合 recipes 使用。
- **模型使用中的输出差异**：一位用户报告称，在使用 **LLAMA 3 8B instruct 模型** 时，其输出与 playground 存在差异，怀疑自己可能误用了 **BASE 模型**。
   - 针对这一困惑，其他成员建议确保其 Tokenizer 和 Prompt 结构正确。
- **分享模型下载指令**：一名成员分享了 **Meta-Llama-3-8B-instruct** 模型的下载命令，并指定了输出目录和所需的 access token。
   - 他们表示，即使按照下载流程操作后，仍担心可能没有真正使用到 **INSTRUCT 模型**。
- **讨论正确的 Prompt 格式化**：关于是否需要使用 **Llama 3 instruct 模板** 来格式化 Prompt 的问题引发了讨论，重点在于输出要求。
   - 另一名成员确认，这一过程由 Tokenizer 自动处理，简化了用户输入。


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1270440954653966360)** (6 messages): 

> - `Model Builders 页面`
> - `PreferenceDataset 重构`
> - `模型索引页`
> - `多模态 LLMs` 


- **关于创建专用 Model Builders 页面的提案**：一名成员建议，为每个模型的 builders 创建一个**专用页面**可能会很有帮助，以适应日益增多的模型和**多模态 LLMs**。
   - *这将使我们能够更好地解释诸如模型下载和配置等重复性细节，* 为用户整合信息。
- **重构后的 PreferenceDataset 支持 Chat**：一名成员强调了新重构的 **PreferenceDataset** 已支持聊天功能，详见 [Pull Request #1276](https://github.com/pytorch/torchtune/pull/1276)。
   - 此更新与之前讨论中确立的统一 **message_transform** 流水线保持一致。
- **对模型索引页的兴趣**：大家一致认为需要一个**模型索引页**来解释与模型相关的基础但重复的任务。
   - 这一想法受到了好评，因为它将简化管理和配置各种模型的过程。



**提到的链接**：<a href="https://github.com/pytorch/torchtune/pull/1276">[4/7] Refactor preference dataset with transforms design by RdoubleA · Pull Request #1276 · pytorch/torchtune</a>：上下文：继 #1186 中的 RFC 之后，我们将在所有数据集中使用统一的 message_transform -> template -> tokenization 数据流水线。此 PR 更新了 PreferenceDataset 以遵循 t...

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1270243386523390075)** (9 messages🔥): 

> - `设置 Open Interpreter`
> - `Open Interpreter 的安全与隐私`
> - `Open Interpreter 的 Python 兼容性`
> - `Open Interpreter 的视觉模型` 


- **解决 Open Interpreter 设置问题**：用户报告了设置 Open Interpreter 时的问题，特别是在选择本地 Llama 模型并在执行期间遇到 **openai.APIConnectionError**。
   - *一位用户报告说，他们的模型在选择后竟然尝试再次下载。*
- **咨询 Open Interpreter 的安全措施**：一名成员对 **Open Interpreter** 如何处理用户数据表示关注，特别是用户数据是否保留在本地机器上。
   - 他们还询问了端到端加密标准以及通信过程中是否有第三方参与。
- **Open Interpreter 的 Python 版本兼容性**：一名成员询问 Open Interpreter 是否支持 **Python 3.12**，作为编程初学者对此表示疑问。
   - 作为回应，另一名成员澄清说，目前需要 **Python 3.10** 或 **3.11** 才能保证兼容性。
- **开源视觉模型推荐**：一位用户征求最适合视觉任务的**开源模型**推荐。
   - 然而，讨论并未针对此询问提供具体的建议。


  

---

### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1270185242996637756)** (2 messages): 

> - `Ollama Models`
> - `API Key Requirements`
> - `Deepgram Support` 


- **Ollama 模型列表命令**：一位成员建议使用命令 `ollama list` 来显示可用的不同模型名称，并指出每个模型在显卡上都需要特定数量的 **VRAM**。
   - 运行模型的说明可以在 [Ollama 文档](https://github.com/OpenInterpreter/open-interpreter/blob/main/docs/language-models/local-models/ollama.mdx) 中找到，该文档强调模型需要适配可用资源。
- **远程托管模型的 API Keys**：强调访问付费的远程托管模型需要 **API Key**。
   - 此外，提到本地模型将在其运行的指定 **端口 (port)** 上操作。
- **关于 Deepgram 支持的咨询**：一位成员询问系统是否具有 **Deepgram 支持**。
   - 这个问题表明了对语音识别功能潜在集成的兴趣。



**提及的链接**：<a href="https://github.com/OpenInterpreter/open-interpreter/blob/main/docs/language-models/local-models/ollama.mdx">open-interpreter/docs/language-models/local-models/ollama.mdx at main · OpenInterpreter/open-interpreter</a>：计算机的自然语言界面。通过在 GitHub 上创建账户为 OpenInterpreter/open-interpreter 的开发做出贡献。

  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1270115495273697293)** (2 messages): 

> - `Llamafile updates`
> - `Mozilla AI community opportunities`
> - `sqlite-vec release party`
> - `Machine Learning Paper Talks`
> - `Local AI AMA` 


- **Llamafile 取得重大进展**：<@723709999452389417> 继续在 Llamafile 上取得史诗级进展，在单个文件中提供离线、可访问的 LLMs。
   - *社区成员对该项目及其对可访问性的潜在影响表示兴奋*。
- **Mozilla AI 社区寻求反馈并提供奖励**：社区正通过调查鼓励反馈，参与者有机会赢取 **$25 礼品卡**。
   - 邀请成员分享 **Mozilla AI** 如何通过社区资源更好地支持他们。
- **加入 sqlite-vec 发布派对**：诚邀大家参加 [sqlite-vec 发布派对](https://discord.com/events/1089876418936180786/1265715263999836210)，由核心维护者 <@533894367354552330> 主持功能讨论和演示。
   - *参与者可以尝试演示并直接与核心团队互动*，丰富体验。
- **引人入胜的 Machine Learning Paper Talks**：即将举行的 **Machine Learning Paper Talks** 由 <@718891366402490439> 主持，主题包括 *Communicative Agents* 和 *Extended Mind Transformers*。
   - 这些活动承诺深入探讨前沿研究并激发与会者的讨论。
- **Local AI AMA 提供见解**：已安排与 Local AI 核心维护者 <@1051191818127147110> 进行 [AMA](https://discord.com/events/1089876418936180786/1268967945216721079)，Local AI 是用于自我托管的开源替代方案。
   - 这为社区成员提供了提问和学习实际实现的机会。



**提及的链接**：<a href="https://form.typeform.com/to/Cn4md4Oc>)">Discover Typeform, where forms = fun</a>：在几分钟内无需代码即可创建美观、互动的表单。免费开始使用。

  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1270412626798841970)** (1 messages): 

> - `LinkedIn Engineering ML Platform` 


- **LinkedIn 工程团队转型其 ML 平台**：LinkedIn 目前正在举办一场直播活动，讨论其工程团队如何转型其 **ML 平台**。
   - 您可以在[此处](https://www.linkedin.com/events/flytepipelinesinactionwithlinke7218669945767776256/theater/)加入讨论。
- **直播活动提醒**：活动正在进行中，提供有关 LinkedIn **Machine Learning** 最新进展的见解。
   - 鼓励参与者在活动期间积极参与并分享想法。


  

---



---



---



---



{% else %}


> 完整的频道细分内容已针对电子邮件进行了截断。
> 
> 如果您想查看完整内容，请访问此电子邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}