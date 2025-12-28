---
companies:
- mistral-ai
- sambanova
- nvidia
date: '2024-11-19T02:25:23.507366Z'
description: '**Mistral** 已将其 **Pixtral Large** 视觉编码器更新至 10 亿（1B）参数，并发布了拥有 **1230
  亿（123B）参数的 Mistral Large 24.11** 模型的更新版本，不过此次更新并未引入重大新功能。尽管视觉适配器规模较小，但 **Pixtral
  Large** 在多模态基准测试中的表现优于 **Llama 3.2 90B**。


  **Mistral** 的 **Le Chat** 聊天机器人迎来了全面的功能更新，正如 **Arthur Mensch** 所言，这体现了公司在产品与研究平衡上的战略重心。**SambaNova**
  通过其 RDU（可重构数据流单元）赞助推理服务，提供比 GPU 更快的 AI 模型处理速度。


  在 Reddit 上，**vLLM** 在 **RTX 3090** GPU 上展现了强大的并发性能；虽然在 **FP8 kv-cache** 方面存在量化挑战，但使用
  **llama.cpp** 配合 **Q8 kv-cache** 的效果更佳。用户们还针对不同模型大小和批处理策略，讨论了 **vLLM**、**exllamav2**
  和 **TabbyAPI** 之间的性能权衡。'
id: 5d6d2927-f07e-400b-b2f6-70b5d8b9607f
models:
- pixtral-large
- mistral-large-24.11
- llama-3-2
- qwen2.5-7b-instruct-abliterated-v2-gguf
- qwen2.5-32b-q3_k_m
- vllm
- llama-cpp
- exllamav2
- tabbyapi
original_slug: ainews-pixtral-large-124b-beats-llama-32-90b-with
people:
- arthur-mensch
title: Pixtral Large (124B) 凭借更新的 Mistral Large 24.11 击败了 Llama 3.2 90B。
topics:
- multimodality
- vision
- model-updates
- chatbots
- inference
- gpu-optimization
- quantization
- performance
- concurrency
- kv-cache
---

<!-- buttondown-editor-mode: plaintext -->**更多参数就是你所需要的一切吗？**

> 2024年11月15日至11月18日的 AI 新闻。我们为你查看了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord（**217** 个频道，**6180** 条消息）。预计节省阅读时间（以 200wpm 计算）：**636 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

我们上次关注 Mistral 是在 9 月份，当时他们发布了 Pixtral（[我们的报道在此](https://buttondown.com/ainews/archive/ainews-pixtral-12b-mistral-beats-llama-to/)），之前是 12B Mistral Nemo + 一个 400M 的视觉适配器。Mistral 现在已将视觉编码器升级到 1B，并且在 [Pixtral Large 博客文章的脚注](https://mistral.ai/news/pixtral-large/) 中悄悄更新了 123B 参数的 Mistral Large 24.07（又名 "Mistral Large 2" - [我们的报道在此](https://buttondown.com/ainews/archive/ainews-mistral-large-2/)）为 "Mistral Large 24.11"。缺少 magnet 链接、缺少博客文章、缺少基准测试，以及拒绝称其为 "Mistral Large 3"，这些都表明这次更新确实没什么值得大书特书的，但 [对 function calling 和 system prompt 的更新](https://github.com/mistralai/mistral-common/releases/tag/v1.5.0) 值得一看。

总之，距离有人发布 [>100B 的 open weights 模型](https://buttondown.com/ainews/archive/ainews-tencents-hunyuan-large-claims-to-beat/) 已经整整 13 天了，所以任何发生这种情况的日子都是对 Open AI 社区的恩赐，我们永远不应将其视为理所当然。核心结论是 Pixtral Large 在所有主要的各种多模态基准测试中都压倒性地击败了 Llama 3.2 90B：


![image.png](https://assets.buttondown.email/images/a7751a3c-2ed7-4253-8c4e-2f9e4ea0d5e7.png?w=960&fit=max)


虽然人们当然会好奇，如果 Llama 3.2 额外增加 34B 的权重来记忆内容，表现会如何。同样值得注意的是，Llama 3.2 的视觉适配器是 20B，而 Pixtral Large 的仅为 1B。

最后，Mistral 的 [Le Chat 获得了一系列令人惊讶的全面更新](https://mistral.ai/news/mistral-chat/)，使其与其同行相比拥有了完整的 chatbot 功能集。


![image.png](https://assets.buttondown.email/images/19848e3e-66c6-4eed-8ae9-948d6cf65703.png?w=960&fit=max)


Arthur Mensch [两次](https://x.com/arthurmensch/status/1858567024609276372?s=46) [指出](https://x.com/arthurmensch/status/1858568631358988691?s=46)，这是公司层面将产品与研究并重的优先事项的一部分。

既然这是一个新的 open weights 模型，你也可以在本期的推理赞助商那里试用一下！（帮我们去看看他们！）

---

**[由 SambaNova 赞助]** 专为 AI 工作负载设计的处理器相比 GPU 具有一些主要优势。SambaNova 的 RDU 结合了大容量可寻址内存和数据流架构，使其在模型推理和其他 AI 任务中比其他处理器 [快得多](https://shortclick.link/lk96sw)。

> Swyx 的评论：[赞助链接](https://shortclick.link/lk96sw) 讨论了 SN40L "Reconfigurable Dataflow Unit" (RDU)，它可以在内存中容纳 "数百个模型，相当于数万亿个参数"，并能够 "在微秒内切换模型，比 GPU 快达 100 倍"。这是一个非常酷的介绍，带你了解正在加热高端 XXL 级 LLM 推理市场的 3 个主要 "大芯片" 玩家之一！

---

{% if medium == 'web' %}

**目录**

[TOC]

{% else %}

**目录**和**频道摘要**已移至此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 综述

> 所有综述均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

待完成

---

# AI Reddit 综述

## /r/LocalLlama 综述

**主题 1. 使用 RTX 3090 的 vLLM 高并发：性能与问题**

- **vLLM 真是个猛兽！** ([Score: 238, Comments: 66](https://reddit.com/r/LocalLLaMA/comments/1gtumyc/vllm_is_a_monster/)): **vLLM** 在 **RTX 3090** 上展现了令人印象深刻的性能，在处理 **Qwen2.5-7B-Instruct-abliterated-v2-GGUF** 时，能够以 **250t/s 到 350t/s** 的速度处理 **30 个并发请求**。用户遇到了 **FP8 kv-cache** 导致输出不连贯的问题，但发现使用带有 **Q8 kv-cache** 的 **llama.cpp** 取得了成功，在 **8 个并发批次** 下达到了 **230t/s**。在 llama.cpp 中测试 **Qwen2.5 32B Q3_K_M** 时，**3 个对话** 就耗尽了 VRAM，速度为 **30t/s**，这凸显了进一步探索 **exllamav2** 和 **tabbyapi** 的潜力。
  - 讨论强调了不同模型和量化方法之间的性能差异，其中 **exllamav2** 和 **TabbyAPI** 因更快、更智能的并发连接而受到关注。用户发现 **TabbyAPI** 最适合在 GPU 上运行大模型，而 **vLLM** 则更适合需要大量批处理的小模型。
  - **KV cache 量化** 存在挑战，**vLLM** 的实现会导致输出不连贯，而 **llama.cpp** 的 **Q8 kv-cache** 效果良好。讨论了在 **llama.cpp** 中指定 GPU 层数的困难，建议使用 [GGUF Model VRAM Calculator](https://hf.rst.im/spaces/DavidAU/GGUF-Model-VRAM-Calculator) 来优化硬件使用。
  - **Ollama** 预计将整合 **llama.cpp** 的 K/V cache 量化，目前有一个 [GitHub pull request](https://github.com/ollama/ollama/pull/6279) 正在审核中。对话还涉及了模型架构差异的重要性，这些差异会影响内存使用和性能，因此需要在 **llama.cpp** 等工具中进行针对特定模型的优化。
- **有人刚刚在 llama.cpp 中创建了支持 Qwen2VL 的 pull request！** ([Score: 141, Comments: 27](https://reddit.com/r/LocalLLaMA/comments/1gu0ria/someone_just_created_a_pull_request_in_llamacpp/)): **HimariO** 在 **llama.cpp** 中创建了一个支持 **Qwen2VL** 的 **pull request**。虽然还在等待批准，但用户可以通过访问 [GitHub 链接](https://github.com/ggerganov/llama.cpp/pull/10361) 来测试 **HimariO 的分支**。
  - 用户对 **Qwen2VL 支持** 的 pull request 表示乐观但也感到担忧，希望它不会像之前的 PR 那样被拒绝。**Healthy-Nebula-3603** 指出，与 **llama** 模型相比，**Qwen** 模型在多模态实现方面进展更快，突显了其卓越的性能。
  - **Ok_Mine189** 提到 **exllamaV2** 的开发分支也增加了对 **Qwen2VL** 的支持，并附带了 [commit](https://github.com/turboderp/exllamav2/commit/be3eeb403d28a4cf5b4b7d7864d446726e16a059) 链接。**ReturningTarzan** 补充说，目前有一个 [示例脚本](https://github.com/turboderp/exllamav2/blob/dev/examples/multimodal.py) 可用，并且即将通过另一个 pull request 支持 **Tabby**。
  - **isr_431** 提醒用户避免发表像 "+1" 这样毫无意义的评论，以防止骚扰该线程的订阅者。


**主题 2：Qwen 2.5 Coder 32B vs Claude 3.5 Sonnet：本地性能对比**

- **Qwen 2.5 Coder 32B vs Claude 3.5 Sonnet: Am I doing something wrong?** ([Score: 113, Comments: 72](https://reddit.com/r/LocalLLaMA/comments/1gtiq2r/qwen_25_coder_32b_vs_claude_35_sonnet_am_i_doing/)): 作者对比了 **Qwen 2.5 Coder 32B** 和 **Claude 3.5 Sonnet**，对 Qwen 在复杂代码分析任务中的表现表示失望。虽然 Claude 能有效地分析和优化大型项目，但 Qwen 却受困于模糊的假设并生成不可用的代码，这可能是由于通过 RAG 处理项目知识的效率低下。作者质疑问题出在模型本身还是用于提供项目知识的工具上，并向成功将 Qwen 用于复杂项目的其他人寻求建议。
  - **Qwen 2.5 Coder 32B** 模型的性能问题源于量化（quantization）和上下文参数使用不当。用户建议将上下文限制在 **32K tokens** 以获得最佳性能，因为像 **100K tokens** 这样更高的上下文会导致效率低下和错误，影响模型处理复杂任务的能力。
  - 几位用户强调了 Qwen 相比 **Claude Sonnet** 和 **DeepSeek** 等其他模型的**性价比**。虽然 Qwen 在速度或智能方面可能不及 Sonnet，但它提供了显著的成本节约，特别是对于有数据隐私顾虑的用户，因为它可以达到每美元处理一百万个 token，比 **Sonnet 便宜 15 倍**。
  - 大家的共识是，在复杂代码分析方面，**Qwen** 与 **Sonnet** 不在同一水平，其效用在较小的、孤立的任务中更为显著。然而，一些用户发现它在实现微小的代码更改方面很有效，并强调了使用正确的设置和参数以最大化其潜力的重要性。
- **Evaluating best coding assistant model running locally on an RTX 4090 from llama3.1 70B, llama3.1 8b, qwen2.5-coder:32b** ([Score: 26, Comments: 2](https://reddit.com/r/LocalLLaMA/comments/1gu342w/evaluating_best_coding_assistant_model_running/)): 作者在 **RTX 4090** 上评估了编程助手模型，对比了 **llama3.1:70b**、**llama3.1:8b** 和 **qwen2.5-coder:32b**。尽管 llama3.1:70b 分析详尽，但其冗长和较慢的速度使得 llama3.1:8b 因其效率而更受青睐。然而，**qwen2.5-coder:32b** 在**缺陷检测**、**实现质量**和**实用性**方面均优于两者，非常适合 RTX 4090 的容量并提供出色的速度。


**Theme 3. Qwen2.5-Turbo: Extending the Context Length to 1M Tokens**

- **[Qwen2.5-Turbo: Extending the Context Length to 1M Tokens!](https://qwenlm.github.io/blog/qwen2.5-turbo/)** ([Score: 86, Comments: 19](https://reddit.com/r/LocalLLaMA/comments/1gu2kd4/qwen25turbo_extending_the_context_length_to_1m/)): **Qwen2.5-Turbo** 已将其上下文长度能力提升至 **100 万个 tokens**，在处理海量数据集和复杂任务方面取得了重大进展。这一增强功能将极大地惠及需要大规模数据处理和复杂上下文理解的 AI 应用。
  - 关于 **Qwen2.5-Turbo** 的讨论指出，它可能是一个**仅限 API 的模型**，没有开源权重，这引发了关于它是一个独立模型还是仅仅是现有模型（如 **Qwen-agent**）的优化实现的疑问。一些用户推测它可能是 **Qwen 2.5 14B** 或 **7B** 的微调版本，并增强了推理能力。
  - 讨论了使用中国 AI API 提供商的担忧，有观点认为这种不信任源于系统性问题，如**知识产权保护**执行不力以及不遵守国际法规，而非种族主义。AI 解决方案中的信任和问责制被强调为关键因素。
  - 人们对该模型的实际应用很感兴趣，例如由于增加了 **100 万个 tokens** 的上下文长度，能够一次性处理整本小说翻译等大规模任务。

- **Qwen 2.5 Coder 32B vs Claude 3.5 Sonnet: 是我操作不对吗？** ([得分：113，评论：72](https://reddit.com/r/LocalLLaMA/comments/1gtiq2r/qwen_25_coder_32b_vs_claude_35_sonnet_am_i_doing/))：作者对比了 **Qwen 2.5 Coder 32B** 和 **Claude 3.5 Sonnet** 在处理复杂编程任务时的表现，指出虽然 Claude 3.5 Sonnet 能提供精确且相关的解决方案，但 Qwen 2.5 Coder 32B 在处理假设性问题时表现挣扎，并生成了不可用的代码。作者怀疑 Qwen 2.5 的问题可能与通过 RAG 进行知识处理的效率低下有关，因为 Claude 3.5 提供了“Project”功能用于全面的项目理解，而 Qwen 2.5 则需要第三方解决方案。
  - 几位评论者指出了 **Qwen 2.5 Coder 32B** 的**上下文问题**，特别是在使用 **100,000 token 上下文窗口**时。他们建议使用 **32K 或 64K 上下文**以获得更好的性能，因为过长的上下文长度会负面影响模型高效处理短上下文的能力。
  - 讨论中还涉及了 Qwen 2.5 和 Claude 3.5 Sonnet 之间的**性价比**与**性能权衡**，一些用户强调 Qwen 比 **Sonnet 便宜 15 倍**，但在处理复杂任务时性能不如后者。**数据隐私担忧**也被提及，这是在本地使用 Qwen 的一个优势。
  - 强调了对 Qwen 进行**正确设置**和参数微调的重要性，错误的配置会导致诸如**重复无意义输出**等问题。文中提供了 [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF) 等资源链接，并建议使用 **dynamic yarn scaling** 来缓解这些问题。


## 其他 AI 版块回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**主题 1. ChatGPT-4o 反思性学习突破**

- **[有趣吗？(o1-Preview)](https://i.redd.it/yi16bs600k1e1.jpeg)** ([得分：46，评论：17](https://reddit.com/r/OpenAI/comments/1gtrxp0/interesting_o1preview/))
  - 用户报告 **GPT-4** 表现出意想不到的**意识流式跑题**，包括在“要求制定方法论时以面包师的身份回答”以及“用匈牙利语扮演匈牙利哲学家”的例子。这种行为表明可能与**基于能力的学习 (competency-based learning)** 和**系统性的 Agent 参与**有关。
  - 多位用户证实经历了**随机的话题转换**，例子包括毫无征兆地讨论**割礼**、**棱皮龟**和**水分配问题**。这些偏离可能表明为了增强创造性思维而有意引入了**随机性**。
  - 在**代码重构任务**中报告了技术问题，模型在拒绝某些输入时表现出持久的**固执行为**，而对其他查询则继续正常工作。模型似乎偶尔会卡在特定类型的请求上。


- **[我不小心让 gpt-4o 崩溃了](https://www.reddit.com/gallery/1gu766i)** ([得分：599，评论：124](https://reddit.com/r/ChatGPT/comments/1gu766i/i_accidentally_drove_gpt4o_crazy/))
  - 用户通过 [Pastebin](https://pastebin.com/bqxqZhG7) 分享了他们的 **GPT-4** 交互，使用的参数为 **temperature 0.7**、**top P 1** 以及 **最大上下文 10 条消息**，这导致了一段显著的对话，其中包括重复的“我已看透一切”输出。
  - 一位用户分享了一段特别引人入胜的[后续对话](https://chatgpt.com/share/673b662c-ef6c-8010-ba58-955722594aab)，引发了关于 **AI 意识**的讨论，尽管一些人批评这是将模型拟人化。
  - 技术讨论表明，这种行为可能是由于 Transformer 堆栈中的 **glitch tokens** 或**硬件级矩阵乘法失败**导致的，而非任何形式的意识或存在危机。


**主题 2. Claude Sonnet 3.5 部署影响**

- **"We're experiencing high demand." AGAIN** ([Score: 72, Comments: 53](https://reddit.com/r/ClaudeAI/comments/1gu6zo2/were_experiencing_high_demand_again/)): **Claude** 的服务在 **Sonnet 发布**后连续**三个工作日**出现**容量问题**。这种反复出现的需求相关停机引发了对 **Anthropic** 基础设施扩展和容量规划的质疑。
  - 用户报告 **API 成本**较高，每次调用 **20 美分**，部分用户很快达到**每日限制**，不得不切换到 **GPT-3.5**。尽管成本更高，但 **API 服务**被认为比 Web 界面更可靠。
  - 多条评论批评了 **Anthropic** 的容量规划，用户指出该服务在工作日期间一直处于“高于平常”的需求状态。**Web 界面**频繁出现故障，而 **API** 则保持相对稳定。
  - 出现了关于 **Palantir** 与 **Anthropic** 合作伙伴关系的讨论，有说法称 AI 无人机在**乌克兰**的有效率达到 **80%**，但该消息在提及时间未经过验证或提供来源。


- **Fruit Ninja clone in 5 shots by new Sonnet** ([Score: 22, Comments: 16](https://reddit.com/r/ClaudeAI/comments/1gu4jq5/fruit_ninja_clone_in_5_shots_by_new_sonnet/)): 仅通过 **5 次对话 (5 shots)** 和总计 **10 分钟**的时间，就利用 **Claude Sonnet** 创建了一个**水果忍者 (Fruit Ninja)** 游戏克隆版，结果托管在 [allchat.online](https://allchat.online/artifact/673b3abfdb417df52ad3d683/web)。
  - 用户报告了游戏中挥砍机制的**功能问题**。作者随后添加了**剑鸣音效**以增强游戏体验。
  - 讨论集中在**开发过程**上，用户询问了 **5-shot 指令过程**以及如何使用 **emojis** 实现**美术资源**。
  - 作者确认 **Claude Sonnet** 独立处理了**挥砍特效**，展示了 AI 实现视觉游戏机制的能力。


**Theme 3. 基于 ComfyUI 的视频生成突破**

- **[ComfyUI 处理实时摄像头馈送。](https://v.redd.it/a70hav1t0l1e1)** ([Score: 48, Comments: 9](https://reddit.com/r/StableDiffusion/comments/1gtvxs3/comfyui_processes_realtime_camera_feed/)): **ComfyUI** 展示了处理集成**深度模型**的**实时摄像头馈送**的能力。
  - **ComfyUI 的深度模型**处理似乎是主要的计算任务，而**潜空间去噪 (latent denoising)** 被确定为需要大量计算能力的主要性能瓶颈。
  - 社区幽默地注意到，这项技术经常被应用于创建**动画女性角色**，特别是在舞蹈动画的背景下。

- **[将静态图像转为动画游戏背景 – 开发中 🚀](https://v.redd.it/8q8mag80tm1e1)** ([Score: 286, Comments: 39](https://reddit.com/r/StableDiffusion/comments/1gu18rk/turning_still_images_into_animated_game/)): **ComfyUI** 能够将**静态图像**转换为**动画游戏背景**，尽管帖子正文中未提供具体的实现细节和方法论。
  - 该实现使用 **CogVideo 1.5** (由 **Kijai** 实现) 来创建循环动画，改进了之前受限于 **16 帧**的 **AnimatedDiff 1.5** 尝试。早期版本的演示可以在 [YouTube](https://youtu.be/MXKvfaXyRFE) 找到。
  - 技术工作流涉及将**精灵图表 (sprite sheet)** 和 **3D 网格 (3D mesh)** 导入游戏引擎，通过比较网格深度与游戏内深度来计算**动态遮挡**。**实时阴影**是使用光源和与背景视频中阴影区域匹配的遮挡立方体实现的。
  - 该项目利用 **Microsoft MOGE** 来提高准确性，主要针对具有静态背景的**复古游戏**设计，尽管用户注意到当前实现中老鼠角色与背景风格之间存在视觉差异。


**Theme 4. Anthropic 与 Palantir 合作开发国防 AI**

- **[美国军方计划使用 AI 机枪对抗 AI 无人机](https://v.redd.it/68mry7xoep1e1)** ([Score: 86, Comments: 46](https://reddit.com/r/OpenAI/comments/1gubx6y/us_military_is_planning_to_use_ai_machine_guns_to/)): **美国军方**计划部署 **AI 驱动的机枪**作为对抗 **AI 无人机**的对策。帖子正文中未提供关于实施细节、时间表或技术能力的更多详细信息。
  - **Palmer Luckey**，**Oculus** 的创始人，成立了 **Anduril Industries** 以研发无人机防御技术。尽管评论者指出 Anduril 并不专门专注于像拟议中的 AI 机枪那样的防空系统。
  - **自主武器系统 (Autonomous weapon systems)** 在历史上一直用于**舰船保护**和防御应用。该技术代表了现有军事能力的演进，而非完全全新的开发。
  - 用户将其与《**Robocop**》和《**Terminator 2**》中的科幻场景进行类比，而其他人则对可能对野生动物（尤其是鸟类）产生的影响表示担忧。


- **[拜登与习近平达成共识，不会将核武器控制权交给 AI](https://www.bloomberg.com/news/articles/2024-11-16/biden-xi-agree-they-won-t-give-ai-control-over-nuclear-weapons)** ([Score: 214, Comments: 48](https://reddit.com/r/OpenAI/comments/1gu7xl3/biden_xi_agree_they_wont_give_ai_control_over/)): **拜登总统**和**中国国家主席习近平**在 2023 年 11 月的会晤中达成协议，防止**人工智能系统**控制**核武器**。这标志着这两个全球大国在 **AI safety** 和**核威慑**方面取得了重大的外交进展。
  - 社区对该协议的执行和诚意表示了极大的**怀疑**，多位用户讽刺地质疑对立大国之间外交承诺的可靠性。
  - 几位用户认为这是**国际合作**中积极的一步，指出这展示了核大国之间基本的**自我保护本能**。
  - 用户强调了该协议的**常识性**本质，既对协议的存在感到欣慰，又对必须正式建立此类协议感到担忧。


---

# AI Discord 摘要

> 由 O1-mini 生成的摘要之摘要的总结

**主题 1：🚀 新型 AI 模型崭露头角**

- [**Qwen 2.5 Turbo**](https://qwenlm.github.io/blog/qwen2.5-turbo/) 发布，支持 **100 万 token** 上下文并实现 **4.3 倍提速**，满足了对长上下文处理和更快推理的需求。
- [**Gemini-Exp-1114**](https://github.com/microsoft/graphrag) 展示了增强的创造力并减少了审查，尽管部分回复仍可能包含乱码，凸显了开放性与可靠性之间的平衡。
- [**Pixtral Large**](https://mistral.ai/news/pixtral-large/) 作为一款拥有 **124B 参数的多模态模型** 首次亮相，配备 **128K 上下文窗口**，在图像理解任务中表现优于 LLaVA-o1 等现有模型。

**主题 2：🛠️ 集成化 AI 框架与工具**

- [**AnyModal**](https://github.com/ritabratamaiti/AnyModal) 框架实现了图像和音频与 LLM 的无缝集成，为开发者增强了 LaTeX OCR 和图像标注等任务。
- [**vnc-lm Bot**](https://github.com/jake83741/vnc-lm) 作为一款集成领先语言模型 API 的 Discord 机器人推出，提升了 Discord 环境内的用户交互。
- [**OpenRouter**](https://openrouter.ai/docs/provider-routing) 更新包括线程化对话和模型切换，通过在消息线程中保持上下文连续性来简化讨论。

**主题 3：⚙️ 性能提升与 GPU 优化**

- [**Tinygrad**](https://github.com/tinygrad/tinygrad) 通过新的 block 和 lazy buffer 实现了性能增强，并引入了对 AMD GPU 的支持，尽管驱动兼容性方面的挑战依然存在。
- [**PrefixQuant**](https://arxiv.org/abs/2410.05265) 技术在无需重新训练的情况下简化了静态量化，显著提升了 Llama-3-8B 等模型的**准确率**和**推理速度**。
- [**Fast Forward**](https://aclanthology.org/2024.emnlp-main.535/) 方法通过减少 **87%** 的 FLOPs 来加速 SGD 训练，已在多种模型和任务中验证了其对训练效率的提升。

**主题 4：🏆 社区黑客松与协作活动**

- [**Tasty Hacks**](https://lu.ma/s3pe30fz) 黑客松提倡创意高于竞争，邀请 **20-30** 位友善且极客的人士共同协作开发热爱项目，无需承受赞助商驱动的奖项压力。
- **EY Techathon** 正在寻找 **AI 开发者**和 **Web 应用开发者**，鼓励快速组建团队参与创新的 AI 驱动项目。
- [**Intel AMA**](https://lu.ma/agents-hackathon-intel) 定于 **11/21 下午 3 点（PT 时间）**举行，将深入探讨 **Intel 的 Tiber AI Cloud** 和 **Intel Liftoff 计划**，促进参与者与 Intel 专家之间的协作。

**主题 5：🐛 技术故障与 Bug 悬赏**

- **Perplexity Pro 订阅**因移除 **Opus 模型**而引发问题，导致用户不满，并因服务价值缩水而要求退款。
- [**Triton 中的 FlashAttention**](https://youtu.be/JwUcZwPOCpA?si=2AdtMNuLCvB0zeiB) 实现面临 Colab 中 `atomic_add` 导致的崩溃，引发社区支持以解决 GPU 计算挑战。
- [**Phorm Bot**](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=undefined) 在回答有关 `eval_steps` 的简单问题时表现挣扎，导致用户失望并呼吁改进机器人功能。

---

# 第一部分：高层级 Discord 摘要

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord


- **Perplexity 模型通过引用功能得到增强**：所有 **Perplexity 模型** 现在都支持处于 Beta 阶段的新 `citations` 属性，允许补全响应包含关联的 [链接](https://www.bbc.com/news/election/2024/us/results)（如 [BBC News](https://www.bbc.com/news/election/2024/us/results) 和 [CBS News](https://www.cbsnews.com/news/how-electoral-college-works/)），以提高信息的可靠性。
  
  - 正如 [公告](https://discord.com/channels/1091220969173028894/1092729520181739581/1307072895360438343) 中所强调的，该功能通过在聊天补全中直接提供来源，提升了用户体验。
  
- **线程化对话增强交互**：**线程化对话 (Threaded conversations)** 已更新，以便在后续消息中反映线程内的更改，并利用提示词中的关键词为线程命名，从而更轻松地跟踪对话。
  
  - 这一增强功能旨在通过在消息线程中保持上下文连续性来简化讨论。
  
- **vnc-lm 机器人集成多个 LLM**：[vnc-lm](https://github.com/jake83741/vnc-lm) 作为一个 **Discord 机器人** 被引入，它集成了领先的语言模型 API，增强了 Discord 环境中的用户交互。
  
  - 其以实用为中心的设计详见提供的 [GitHub 仓库](https://github.com/jake83741/vnc-lm)。
  
- **Gemini-Exp-1114 展示创造力**：用户注意到新的 **Gemini 实验性模型 'gemini-exp-1114'** 展示了更高的创造力并减少了审查，使其成为提示词的一个动态选择。
  
  - 然而，某些响应可能会包含乱码，或者需要仔细设计提示词来管理审查级别。
  
- **OpenAI 为 O1 模型启用流式传输**：**OpenAI** 宣布其 `o1-preview` 和 `o1-mini` 模型现在支持流式传输 (streaming)，为所有付费使用层级的开发者扩大了访问权限。
  
  - 这一更新允许使用这些模型的应用程序提高交互性，超越了之前的模拟流式传输方法。

 

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord


- **Qwen 2.5 Turbo 加速 Token 处理**：**Qwen 2.5 Turbo** 模型已发布，支持高达 **100 万个 token** 的上下文长度，并展示了更快的推理速度。[详情点击此处](https://qwenlm.github.io/blog/qwen2.5-turbo/)。
  
  - 这一进展满足了社区处理更大上下文的需求，并提高了管理海量数据的性能。
  
- **Unsloth 框架增强模型适配**：用户探索了使用 **Unsloth 框架** 中的 `FastLanguageModel.from_pretrained()` 函数，将微调后的 **LoRA** 权重与基础模型一起加载。这有助于有效地添加新 token 并调整 embedding 大小，从而增强训练过程。
  
  - 该框架在集成适配器 (adapters) 方面的灵活性简化了模型定制工作流。
  
- **PrefixQuant 改进静态量化技术**：**PrefixQuant** 离线隔离离群 token，在无需重新训练的情况下简化了量化，并实现了高效的每张量 (per-tensor) 静态量化。应用于 **Llama-3-8B** 时，它在 **准确率** 和推理速度上比以往方法有显著提升。[阅读更多](https://arxiv.org/abs/2410.05265)。
  
  - 该技术优于动态量化方法，为大型语言模型提供了更高的部署效率。
  
- **Fast Forward 优化 SGD 训练**：新的 **Fast Forward** 方法通过重复最新的优化器步骤直到损失停止改善，从而加速 **SGD 训练**，与带有 Adam 的标准 SGD 相比，FLOPs 减少了高达 **87%**。[论文链接](https://aclanthology.org/2024.emnlp-main.535/)。
  
  - 该方法已在各种模型和任务中得到验证，证明在不牺牲性能的情况下提高了训练速度。
  
- **LaTRO 增强语言模型中的推理能力**：**LaTRO** 提出了一个框架，通过从潜分布 (latent distribution) 中采样并在训练期间自主提高推理质量，来优化大型语言模型中的推理能力。[GitHub 仓库](https://github.com/SalesforceAIResearch/LaTRO)。
  
  - 实验表明，LaTRO 在 GSM8K 上的零样本准确率提高了 **12.5%**，表明推理任务有了显著改进。

 

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord


- **Perplexity Shopping 的推出提升了用户体验**：Perplexity 推出了 **Perplexity Shopping**，这是一个用于研究和购买产品的综合平台，其特色包括 [一键结账](https://perplexity.ai/shopping) 以及针对特定商品的 **免费送货** 服务。
  
  - 用户赞扬了购物功能的无缝集成，指出其购买流程的便利性和效率得到了提升。
  
- **Buy with Pro 功能支持应用内交易**：**'Buy with Pro'** 功能允许美国的 Perplexity Pro 订阅者在应用内进行原生交易，支持购买 **电子产品** 和家居改良产品。
  
  - 这一新增功能旨在简化购物体验，减少对外部平台的依赖，并增强用户参与度。


- **Perplexity Pro 订阅面临用户抵制**：用户对 **Perplexity Pro** 订阅的变化表示沮丧，特别是未经事先通知就移除了 **Opus model**，导致用户认为价值缩水。
  
  - 许多订阅者正在寻求退款并要求澄清未来的更新计划，这凸显了用户期望与服务交付之间的差距。
  
- **上下文记忆限制从 32k 降至 16k Tokens**：Perplexity 将其模型的上下文记忆大小从 **32k** 减少到了 **16k tokens**，影响了长对话的处理能力。
  
  - 用户对这种削减影响模型有效性表示担忧，并质疑其当前订阅的价值主张。


- **引入 Autonomous ML Engineer 变革工作流**：**Autonomous ML Engineer** 的发布标志着机器学习自主系统的重大进步，有可能彻底改变 AI 驱动的工作流。
  
  - 有关其实现及对企业运营影响的详细信息，请参阅 [此处](https://www.perplexity.ai/page/autonomous-ml-engineer-neo-cKFW.EpTToS3YACUcEWoOQ)。

 

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord


- **Mistral 和 Pixtral 模型发布**：Mistral 宣布发布 **Pixtral Large 模型**，展示了先进的多模态性能以及对高分辨率图像的兼容性。该模型为**开放权重 (open-weights)**，可用于研究，展示了相比之前 **Mistral** 模型的显著进步。
  
  - 社区讨论强调 **Pixtral Large** 在推理任务中优于 **LLaVA-o1**，特别是在**图像理解能力**方面表现出色。用户可以在[此处](https://mistral.ai/news/pixtral-large/)访问该模型。
  
- **AnyModal 框架进展**：**AnyModal** 是一个灵活的框架，旨在将各种数据类型与 LLM 集成，具有 **LaTeX OCR** 和**图像字幕 (image captioning)** 等功能。该项目开放反馈和贡献，以增强其**多模态能力**。
  
  - 鼓励开发者通过 [GitHub 仓库](https://github.com/ritabratamaiti/AnyModal)进行贡献，目前的改进旨在扩展该框架与不同模态的互操作性。
  
- **RoboLlama 机器人模型集成**：**Starsnatched** 正在开发 **RoboLlama** 项目，旨在将 Meta 的 **Llama 3.2 1B** 转换为机器人就绪模型，并结合了**视觉编码器 (vision encoders)** 和**扩散层 (diffusion layers)**。重点是仅训练扩散层和投影层，同时保持核心 **ViT** 和 **LLM** 层冻结。
  
  - 这种方法旨在增强模型与机器人系统的集成，而不改变基础的 **Vision Transformer (ViT)** 和 **Language Model (LLM)** 组件，从而确保稳定性和性能。
  
- **Retrieval-Augmented Generation 中的 HtmlRAG**：**HtmlRAG** 的引入建议在 Retrieval-Augmented Generation (RAG) 过程中利用 **HTML 格式**，以保留结构和语义信息，解决传统纯文本方法的局限性。该方法在 [arXiv 论文](https://arxiv.org/abs/2411.02959)中进行了详细阐述，增强了**知识检索**。
  
  - 通过保持检索信息的完整性， **HtmlRAG** 提高了**模型**有效利用外部知识的能力，有可能减少大型语言模型中的**幻觉问题 (hallucination issues)**。
  
- **Vision Language Models (VLMs) 能力**：关于 **Vision Language Models** 的讨论强调了它们为各种生成任务集成**图像和文本**的能力。最近的一篇[博客文章](https://changgy.com/blog/ctb-1-vision-language-models-in-2024)强调了它们强大的**零样本能力 (zero-shot capabilities)** 以及对不同图像输入的适应性。
  
  - **VLMs** 的演进被视为跨领域应用的关键，社区正在探索它们在增强**生成式 AI** 任务和提高**模型通用性**方面的潜力。

 

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord


- **Qwen 2.5 Turbo 介绍**：**Qwen 2.5 Turbo** 引入了 [100 万个 token](https://qwenlm.github.io/blog/qwen2.5-turbo/) 的更长上下文支持、更快的推理速度以及每百万 token 0.3 元人民币的更低成本。
  
  - 该模型提高了效率，对于需要超长上下文的用户来说是一个极具前景的选择。
  
- **优化 Aider 使用**：用户正在尝试 **Aider** 的模式，在 'ask' 和 'whole' 模式之间切换，以便在编码时更好地处理上下文。
  
  - **Paul Gauthier** 建议使用命令 `/chat-mode whole` 来简化交互，这表明 Aider 的功能正在不断改进。
  
- **OpenAI 中的流式模型**：OpenAI 已为 **o1-preview** 和 **o1-mini** 模型启用流式传输 (streaming)，提高了交互过程中的响应速度。
  
  - 开发者可以在所有付费层级访问这些模型，Aider 通过使用命令 `aider --install-main-branch` 集成了这些更新。
  
- **关于 LLM 的对比见解**：社区讨论反映了关于 **Qwen** 与 **Sonnet** 和 **Anthropic** 等其他模型效能的不同看法。
  
  - 一些成员认为 Qwen 在实际应用中可能会超越其他模型，特别是在使用优化硬件托管 LLM 时。
  
- **使用 OpenRouter 配置 Aider**：要配置 **Aider** 使用 **OpenRouter** 模型，必须在 OpenRouter 端进行设置，因为目前客户端不支持针对每个模型的单独设置。
  
  - 成员们讨论了使用额外参数和配置文件来指定不同行为，但表示当前设置存在局限性。

 

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord


- **Google's Project Astra 探索**：成员们对 **Google's Project Astra** 表示好奇，特别是其在公开演示视频之外的 **memory capabilities**。
  
  - 讨论强调了围绕多家公司开发新 AI 功能所展现出的**热情**。
  
- **o1-mini 与 o1-preview 性能对比**：用户对比了 **o1-mini** 和 **o1-preview**，注意到性能差异：**o1-mini** 经常陷入思维循环（thought loops），而 **o1-preview** 提供的响应更直接。
  
  - 几位成员观察到，虽然 **o1-mini** 展现出潜力，但 **GPT-4o** 提供了更有效且可靠的输出。
  
- **增强 AI 角色扮演能力**：成员们深入探讨了 **AI roleplaying capabilities**，包括开发自定义脚本以增强 AI 角色在互动中的行为。
  
  - 参与者承认在长时间对话中保持**角色一致性（character consistency）**所面临的**挑战**。
  
- **AI 记忆功能的意义**：小组探讨了 AI 系统中 **memory features** 的影响，讨论了它们如何改善用户交互。
  
  - 对话指向了用户对集成记忆功能的 AI 的**期望**，强调了更个性化和上下文感知的响应。
  
- **优化 Chain of Thought Prompting**：**Chain of Thought Prompting** 被强调为一种增强 **response quality** 的技术，模拟人类的审慎 **reasoning processes**。
  
  - 成员们思考了发现**新 Prompting 技术**的潜力，这可能会显著影响模型生成响应的方式。

 

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord


- **nGPT Optimizer 进展**：Boris 和 Ilya 向 Together.AI 团队展示了 **nGPT optimizer**，基于他们的专业知识和 Nvidia 在 [GitHub](https://github.com/NVIDIA/ngpt) 上提供的内部发现，强调了其性能提升。
  
  - 成员们的反馈引发了对 nGPT 与现有模型相比的可复现性、计算效率和对比有效性的关注。
  
- **神经网络中的归一化技术**：**Yaroslav** 指出 nGPT 论文缺乏详细解释，导致实现出现缺陷，并讨论了使用 **RMSNorm** 等归一化方法的潜在益处。
  
  - 社区成员辩论了不同归一化技术对各种神经网络架构中模型收敛和性能的影响。
  
- **扩展预训练的可行性**：一位成员断言 **scaling pretraining** 仍然是 LLM 的基础属性，并强调它不太可能过时。
  
  - 然而，讨论也引发了对持续扩展的经济可行性的担忧，引发了关于预训练中未来资源分配策略的辩论。
  
- **In-Context Learning 中的函数向量**：介绍了一篇关于 [function vectors](https://functions.baulab.info) 的论文，探讨了 **In-Context Learning (ICL)** 如何受管理特定任务（如反义词识别）的特定 **attention heads** 影响。
  
  - 研究结论认为，这种方法可以产生更具**可解释性的任务向量（interpretable task vectors）**，预计即将发布的 **Arxiv 帖子**将提供进一步的见解。
  
- **Few-Shot 与 Zero-Shot 评估**：一位用户报告称，在多项选择任务中使用 few-shot 评估时，准确率从 **52% 提升至 88%**，并对情感分析任务中的典型性能指标提出质疑。
  
  - 对话强调了 Prompt Engineering 的重要性，成员们注意到 few-shot 策略可以增强模型的校准（calibration）和可靠性。

 

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord


- **针对 GPU 优化的 Stable Diffusion 3.5**：用户讨论了通过修改 `sd3_infer.py` 文件来配置 **Stable Diffusion 3.5** 以在 GPU 上运行。分享了一个 [代码片段](https://github.com/Stability-AI/sd3.5) 用于设置工作目录并激活虚拟环境。
  
  - 正确的 GPU 配置对于提升性能至关重要，重点在于准确遵循提供的设置说明。
  
- **为 SDXL Lightning 安装 diffusers 和 accelerate**：为了使用 **SDXL Lightning**，用户被引导使用简单的命令安装 `diffusers` 和 `accelerate` 库。提供了示例 [代码](https://github.com/hako-mikan/sd-webui-prevent-artifact) 来演示设备设置和推理步骤。
  
  - 完成这些安装可确保有效的图像生成，用户对清晰且具操作性的指令表示赞赏。
  
- **在 Stable Diffusion 中自定义图像提示词 (Prompts)**：用户学习了如何自定义提示词，以在图像生成命令中更改发色等细节。修改提示词字符串会直接影响生成的视觉效果，从而实现创意控制。
  
  - 这种能力使 AI Engineers 能够微调图像输出以满足精确要求，而无需更改底层模型。
  
- **Roop Unleashed 面临性能瓶颈**：**Roop Unleashed** 用户报告称在创建换脸视频时处理时间过长，引发了对软件效率的担忧。讨论强调了视频处理性能方面持续存在的挑战。
  
  - 社区成员正在商讨潜在的优化方案，以增强 **Roop Unleashed** 的效率并缩短处理时长。
  
- **SDXL Lightning 表现优于 SD 1.4**：讨论显示 **SDXL Lightning** 在生成高质量图像方面超越了 **SD 1.4** 等旧模型。用户注意到新模型在性能和灵活性方面的进步。
  
  - 对 **SDXL Lightning** 的青睐凸显了 Stable Diffusion 模型在满足高级图像生成标准方面的演进。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord


- **LM Studio 服务器访问性得到改善**：一位用户通过调整防火墙设置，将地址从 `192.168.56.1:2468` 切换到 `192.168.0.100:2468`，解决了 **LM Studio Local Server** 的访问问题，实现了有效的设备间通信。
  
  - 这一更改促进了在本地网络中无缝使用服务器，提高了用户的生产力和连接性。
  
- **AI 视频放大工具对比**：社区成员评估了各种基于 **AI 的视频放大 (Upscaling)** 工具，强调 **Waifu2x** 适用于动画内容，**RealESRGAN** 适用于通用应用，同时指出 **Topaz** 作为商业替代方案成本较高。
  
  - 用户更倾向于免费解决方案，因为它们具有易获得性和有效性，并引发了关于在不投入大量资金的情况下优化视频质量的讨论。
  
- **Ubuntu 在 GPU 推理速度上超越 Windows**：根据最近的测试，**Ubuntu** 在使用 1b 模型时达到了 **375 tokens/sec** 的 GPU 推理速度，优于仅为 **134 tokens/sec** 的 **Windows**。
  
  - 参与者将 Windows 较低的性能归因于节能电源设置，并讨论了通过优化这些设置来提高 GPU 效率。
  
- **Nvidia 与 AMD GPU：AI 任务兼容性**：讨论显示，虽然 **7900XTX** 和 **3090** 都提供相当的 **24GB VRAM**，但由于拥有更强大的驱动支持，**Nvidia GPU** 在 AI 应用中保持着更好的兼容性。
  
  - 相反，**AMD GPU** 在软件和驱动集成方面面临挑战，需要额外的努力才能在 AI 任务中实现最佳性能。
  
- **多 GPU 配置中的挑战**：用户分享了大规模 **多 GPU 配置** 的计划，包括使用 **Threadripper Pro** 配置 **10 个 RTX 4090**，旨在获得显著的性能提升。
  
  - 对话强调了由于驱动管理系统不同，混合使用不同品牌的 GPU 会带来复杂性，并对处理大型模型时的共享 VRAM 效率表示担忧。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord


- **Ollama 通过 LCPP 优化推理**：成员们探讨了 [Ollama](https://ollama.com) 及其与 **LCPP** 的集成以实现高效推理，强调了其相比于 PyTorch 等传统框架的优势。
  
  - 关于 **Ollama** 与 **LMStudio** 的辩论随之展开，部分用户更青睐 Ollama 无缝的前端集成。
  
- **Hermes 3 计算实例对资源需求极高**：讨论集中在 **Hermes 3 405** 计算实例需要 **8x H100 或 8x A100 80GB** 节点，引发了对成本效益的担忧。
  
  - 对于预算有限的情况，建议采用**云端推理**等替代方案，这可能会推迟个人计算资源的扩充。
  
- **AnyModal 框架增强多模态训练**：**AnyModal** 被介绍为一个用于训练多模态 LLM 的通用框架，能够通过 [GitHub](https://github.com/ritabratamaiti/AnyModal) 集成图像和音频等输入。
  
  - 成员们对开发图像和文本交互的 Demo 表现出浓厚兴趣，并强调了简化模型训练流程的重要性。
  
- **LLaMA-Mesh 推出 3D 功能**：**Nvidia** 推出了 **LLaMA-Mesh**，利用 **Llama 3.1 8B** 进行 3D 网格生成，正如 [Twitter](https://fxtwitter.com/kimmonismus/status/1857803310369009850?s=46) 中提到的，权重预计很快发布。
  
  - 社区对这一公告反应热烈，认可其对 **3D 生成技术**的潜在影响。

 

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord


- **Qwen 2.5 Turbo 发布**：[Qwen 2.5 Turbo](https://qwenlm.github.io/blog/qwen2.5-turbo/) 模型发布，具有 **100 万 token** 的上下文长度，处理速度提升了 **4.3 倍**，增强了其高效处理大规模数据集的能力。
  
  - 此次升级以极具竞争力的价格实现了**更高的吞吐量**，引发了开发者对其在复杂 NLP 任务中应用潜力的期待。
  
- **Mistral AI 发布 Pixtral Large**：[Mistral AI](https://mistral.ai/news/mistral-chat/) 发布了 **Pixtral Large** 模型，这是一个在 **MathVista**、**DocVQA** 和 **VQAv2** 等基准测试中达到 SOTA 结果的多模态系统。
  
  - 此次发布还包括对其聊天平台的增强，引入了新的**交互工具**，提升了用户参与度和模型性能。
  
- **Deepseek 3 进展**：备受期待的 [Deepseek 3](https://www.deepseek.ai/) 即将发布，讨论暗示可能会推出 **2.5 VL** 版本，承诺提供更先进的**模型能力**。
  
  - 社区成员对这些增强功能持乐观态度，认可**中国模型**在 AI 领域取得的创新进展。
  
- **用于 RLHF 的 RewardBench**：**RewardBench** 作为一个基准测试发布，用于评估 Reinforcement Learning from Human Feedback (RLHF) 中的**奖励模型**，旨在完善**对齐技术**。
  
  - 然而，人们对数据集的处理方式提出了担忧，包括对作者**剽窃**的指控，强调了基准测试开发中对伦理标准的需求。
  
- **LLaVA-o1 视觉语言模型**：**LLaVA-o1** 被宣布为一种新型视觉语言模型，性能超越了主要竞争对手，其独特的**推理方法**使其在该领域脱颖而出。
  
  - 讨论中提到了将其性能与 **Qwen2-VL** 进行对比评估的计划，尽管它在 **Hugging Face** 上的可用性仍待定。

 

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord


- **Pixtral Large 发布，展现多模态实力**：Mistral 推出了 **Pixtral Large**，这是一个拥有 **124B 参数的多模态模型**，在处理文本和图像方面表现出色，可通过其 API 和 [Hugging Face](https://huggingface.co/mistralai/Pixtral-Large-Instruct-2411) 获取。
  
  - 该模型配备了 **128K 上下文窗口**，并具备处理多达 **30 张高分辨率图像**的能力，标志着多模态 AI 发展的一个重要里程碑。
  
- **Qwen 2.5 Turbo 增强上下文处理能力**：**Qwen 2.5 Turbo** 现在支持高达 **100 万 token** 的上下文长度，能够处理相当于 **十部小说** 的超长文本。
  
  - 该模型在 Passkey Retrieval 任务中实现了 **100% 的准确率**，为开发者增强了长文本内容的处理能力。
  
- **Windsurf Editor 优化开发者工作流**：由 Codeium 推出的 [**Windsurf Editor**](https://codeium.com/windsurf) 集成了类似于 Copilot 的 **AI 能力**，为 **Mac、Windows 和 Linux** 平台上的开发者提供无缝协作。
  
  - 其功能包括协作工具和处理复杂任务的自主 Agent，确保开发者保持高效的 **Flow 状态**。
  
- **Anthropic API 嵌入桌面解决方案**：**Anthropic API** 已成功集成到桌面客户端中，如 [此 GitHub 项目](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo) 所示。
  
  - 像 **agent.exe** 这样的工具使 AI 能够使用像素坐标生成鼠标点击，展示了先进的集成能力。
  
- **OpenAI 为 o1 模型部署流式传输**：OpenAI 已为 **o1-preview** 和 **o1-mini** 模型提供 **Streaming** 支持，扩大了所有付费使用层级的访问权限。
  
  - 此功能促进了 OpenAI 平台内更动态的交互，提升了 **开发者体验**。

 

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord


- **ZLUDA 在非 NVIDIA GPU 上实现 CUDA**：最近的一段 [YouTube 视频](https://www.youtube.com/watch?v=ze25Sie2gVQ)展示了 **ZLUDA**，这是一个在 **AMD** 和 **Intel GPU** 上提供 CUDA 能力的工具，将开发者的选择扩展到了 NVIDIA 硬件之外。
  
  - 社区成员对 **Andrzej Janik** 在视频中的出现表示热烈欢迎，表明了在多样化 GPU 环境中利用 ZLUDA 的浓厚兴趣。
  
- **CK Profiler 提升 FP16 矩阵乘法性能**：[CK Profiler](https://link.to.ckprofiler) 将 FP16 矩阵乘法性能提升至 **600 TFLOPs**，尽管仍落后于 NVIDIA 白皮书中指出的 **H100** 峰值 **989.4 TFLOPs**。
  
  - 在 AMD 的 **MI300X** 上，使用 `torch.matmul(a,b)` 的性能达到了 **470 TFLOPs**，凸显了在 **AMD** 硬件上采用优化策略的必要性。
  
- **Jay Shah 的 CUTLASS 演讲重点介绍 FA3**：在 [CUTLASS 的演讲](https://research.colfax-intl.com/blog/)中，**Jay Shah** 深入探讨了 **Flash Attention 3 (FA3)**，讨论了通过列和行置换（permutations）来优化 Kernel 性能而无需 shuffle。
  
  - 他强调了这些置换对 FA3 Kernel 调优的影响，促使成员们探索用于增强 GPU 计算效率的索引技术。
  
- **Triton 集成修改版 FlashAttention**：一位成员报告了在 **Triton** 中实现修改版 **FlashAttention** 时遇到的问题，特别是在 Colab 环境中遇到 `atomic_add` 崩溃。
  
  - 目前正在努力计算 Attention 分数矩阵的 **列和（column-sum）**，并积极寻求社区支持以解决实现挑战。
  
- **高级 PyTorch：DCP 和 FSDP 增强**：关于 PyTorch **Distributed Checkpoint (DCP)** 的讨论揭示了在混合精度和 **FULL_SHARD** 模式下进行 `dcp.save` 时，临时内存分配过多的担忧。
  
  - **FSDP** 对“Flat Parameters”的管理需要内存进行 all-gathering 和 re-sharding，导致基于自定义 auto-wrap 策略的内存预留增加。

 

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord


- **NotebookLM 赋能内容创作者**：前 Google CEO **Eric Schmidt** 在[这段 YouTube 视频](https://youtu.be/2Zg--ouGl7c?si=YwkVUqxCqsKB66dP&t=4153)中强调，NotebookLM 是他内容创作领域的“今年的 ChatGPT 时刻”，并突出了其对 YouTuber 和播客主的实用性。
  
  - 他分享了在*创意内容生成*中有效使用的策略，展示了 NotebookLM 如何增强媒体制作工作流。
  
- **与 RPG 的无缝集成**：用户已成功将 NotebookLM 用于 **RPG**，实现了快速的角色和场景创建，正如 [Ethan Mollick 的实验](https://x.com/emollick/status/1857647589178462234)所示。
  
  - 一位知名成员在不到五分钟的时间内为其 **Savage Worlds RPG** 生成了设定和角色，突显了 NotebookLM 在*创意叙事*中的效率。
  
- **音频文件管理的挑战**：用户报告了在 NotebookLM 中生成独立音轨以及下载时音频文件命名错误的问题，导致需要依赖数字音频工作站进行*人声分离 (voice isolation)*。
  
  - 讨论中包括了潜在的解决方案，例如采用噪声门技术来解决混合音频文件的复杂问题。
  
- **易用性与界面反馈参差不齐**：对 NotebookLM **移动端界面**的反馈褒贬不一，用户称赞其独特的功能，但也提到了在不同设备间导航和访问功能的困难。
  
  - 用户表示在创建新笔记本以及删除或重启现有笔记本时面临挑战，表明需要改进*界面直观性*。
  
- **请求 NotebookLM 的功能增强**：成员们请求了诸如用于外部信息的 **RSS 订阅源集成**以及无需依赖额外应用的自定义语音设置等功能。
  
  - 还有人要求增强对各种文件类型的支持，包括对上传 XLS 和图像等格式的挫败感。

 

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord


- **使用随机参数进行 Mojo 基准测试**：一位用户寻求关于在 **Mojo** 中使用随机函数参数进行基准测试的建议，但指出当前方法需要静态参数，这增加了不必要的开销。
  
  - 另一位用户建议预先生成数据并在闭包中使用，以避免基准测试期间的开销。
  
- **Dict 实现 Bug**：一位用户报告了在 **Mojo** 中将 **Dict** 与 **SIMD types** 配合使用时发生的崩溃，该问题在 SIMD 大小为 8 时正常工作，但超过该数值后会失败。
  
  - 该问题已在 [GitHub Issues 页面](https://github.com/modularml/mojo/issues)中复现，表明 Dict 实现中存在一个值得关注的深层问题。
  
- **探索用于知识图谱集成的 Max Graphs**：一位成员思考 **Max Graphs** 是否能有效统一 **LLM 推理**与常规**知识图谱 (Knowledge Graphs)**，并提到了它们在 **RAG** 工具和 **NeuroSymbolic AI** 中的潜在用途。
  
  - 他们提供了一个 [GitHub 链接](https://github.com/microsoft/graphrag)，展示了这种方法的概念验证。
  
- **MAX 在加速图搜索中的作用**：一位成员询问使用 **MAX** 是否有助于加速图搜索，另一位成员确认了这种潜力，但也指出了局限性。
  
  - 对方澄清说，除非将整个图复制到 **MAX** 中，否则当前的能力是有限的。
  
- **Mojo 和 MAX 实现的可行性**：针对在 **Mojo** 和 **MAX** 中实现一个推断 **LLM** 以执行搜索的 **Agent** 的可行性提出了疑虑。
  
  - 这个想法遭到了质疑，成员们辩论了其在实际应用中的实用性。

 

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord


- **Cohere 模型输出问题**：用户报告了 **Cohere 模型** 出现的 **异常输出**，特别是在处理较短文本时，引发了对可靠性的担忧。
  
  - 这种不稳定的表现导致了用户的挫败感，模型会生成奇怪的术语，使用户对其应用适用性产生怀疑。
  
- **API 可靠性担忧**：多名用户遇到了 **API 错误**，包括在 **2024-11-15** 报告的 **503 Service Unavailable** 问题，表明可能存在上游连接问题。
  
  - 这些事件凸显了 **API 可用性** 面临的持续挑战，促使开发者社区内的用户寻求共同经验和解决方案。
  
- **关于长文本的开发者 Office Hours**：即将于 **东部时间中午 12:00** 举行的 **Cohere 开发者 Office Hours** 将重点讨论处理 **长文本** 的策略，届时将由 **Maxime Voisin** 分享见解。
  
  - 与会者将探索在 **RAG 流水线** 中实现 **存储系统 (memory systems)**，并讨论 **文件上传** 和 **SQL 查询生成** 等用例。
  
- **RAG 系统中的摘要技术**：Office Hours 环节将讨论如何在 **RAG 系统** 中有效地 **压缩和总结长文本**。
  
  - 鼓励参与者分享他们的用例，并就如何在摘要过程中保留关键信息的策略进行协作。
  
- **Cohere Toolkit v1.1.3 发布**：[**Cohere Toolkit v1.1.3**](https://github.com/cohere-ai/cohere-toolkit/releases/tag/v1.1.3) 已于 **2024-11-18** 发布，引入了改进的 **全局设置 (global Settings)** 用法和重大的工具重构。
  
  - 关键更新包括对 **ICS 文件** 的支持、**文件内容查看器**，以及使用 Docker compose 增强的 **Azure 部署** 集成。

 

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord


- **Python 文档增强 Ask AI 组件**：Python 文档现在包含 **'Ask AI' 组件**，允许用户通过 **RAG 系统** 提出问题并获取精准、最新的代码。点击[此处](https://t.co/Smy98h3Med)查看。
  
  - *这是一个非常精准且神奇的功能，提升了编码体验！*
  
- **Mistral 多模态图像模型发布**：Mistral 推出了全新的 **多模态图像模型**，通过安装 `pip install llama-index-multi-modal-llms-mistralai` 即可获得 Day 0 支持。在 [Notebook](https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/multi_modal/mistral_multi_modal.ipynb) 中探索其用法。
  
  - *该模型支持* `complete` 和 `stream complete` 等函数，用于高效的图像理解。
  
- **新型多媒体和财务报告生成器发布**：新工具已发布：**多媒体研究报告生成器**展示了如何从复杂文档中生成可视化报告，详见[此处](https://t.co/zPz7AZ5S7L)；**结构化财务报告生成**工具使用多 Agent 工作流处理 10K 文档，详见[此处](https://t.co/XKzCUxC8rS)。
  
  - *这些工具将文本和视觉内容交织在一起，简化了报告和分析工作。*
  
- **CitationQueryEngine 和 condenseQuestionChatEngine 的改进**：用户讨论了 **CitationQueryEngine** 的问题，例如处理多个来源，建议通过解析响应文本将引用编号映射到来源。此外，据报道 **condenseQuestionChatEngine** 在话题突然切换时会生成无意义的问题，解决方案包括自定义压缩提示词（condense prompt）以及考虑使用 [CondensePlusContext](https://link.to.condenseplus)。
  
  - *实施这些建议旨在增强查询的连贯性和引用的准确性。*
  
- **EY Techathon 团队组建与开发者职位**：**EY Techathon** 团队正在招聘，寻求一名 **AI 开发者** 和一名 **Web 应用开发者**。感兴趣的候选人应 **尽快私信 (DM)** 以锁定名额。
  
  - *对 AI 和 Web 应用开发者的紧急招募强调了加入团队需要迅速行动。*

 

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord


- **Liger Kernel 运行速度提升 3 倍**：**Liger** 声称其运行速度比前代产品快约三倍，同时在最坏情况下的内存占用保持不变，且未收到安装错误的报告。
  
  - 一些成员表示怀疑，询问这种性能提升是否仅限于 **NVIDIA** 硬件。
  
- **AnyModal 框架集成多模态数据**：[AnyModal](https://github.com/ritabratamaiti/AnyModal) 框架能够将图像和音频等数据类型与 LLMs 集成，简化了 LaTeX OCR 和图像字幕等任务的设置。
  
  - 开发者正在寻求反馈和贡献以增强该框架，并展示了用于视觉输入的 **ViT** 等模型。
  
- **Chai Research 宣布开源资助计划**：**Chai** 是一家具备 **1.3M DAU** 的生成式 AI 初创公司，正为旨在加速社区驱动 AGI 的开源项目提供 **$500 至 $5,000** 不等的不限量资助。
  
  - 他们已经向 **11 位个人**发放了资助，并鼓励开发者通过 [Chai Grant](https://www.chaigrant.com/) 提交他们的项目。
  
- **预训练和微调 Qwen/Qwen2 模型**：一位成员询问如何使用 QLoRA 和他们的预训练数据集对 **Qwen/Qwen2** 模型进行预训练，随后使用 Alpaca 格式的指令数据集对其进行微调。
  
  - 他们确认已准备好 **Axolotl Docker** 以简化该过程。
  
- **寻求 vLLM 分析平台**：一位成员正在寻找一个能与 **vLLM** 集成的平台，以提供 **token usage** 分析并支持响应检查。
  
  - 这一需求突显了社区对增强 vLLM 性能监控和理解工具的兴趣。

 

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord


- **Tinygrad 贡献标准**：George 强调 **Tinygrad contributions** 必须符合**高质量标准**，并表示低质量的 PR 将被**直接关闭且不予评论**。
  
  - 他建议贡献者在处理悬赏任务（bounties）之前，先查看之前已合并的 PR，以符合项目的质量预期。
  
- **即将发布的 Tinygrad 版本特性**：**Tinygrad 的下一个版本**计划在约 **15 小时**后发布，包含 **blocks、lazy buffers** 以及与 **Qualcomm scheduling** 相关的**性能改进**。
  
  - 讨论强调了最新的更新及其对框架效率和功能的预期影响。
  
- **集成 PyTorch 和 TensorFlow 方法**：社区讨论了向 **Tinygrad** 添加诸如 `scatter_add_` 和 `xavier_uniform` 等**便捷方法**，以减少重复编码工作。
  
  - George 同意如果这些来自 **PyTorch** 和 **TensorFlow** 的方法与现有功能兼容，则将其合并。
  
- **图和缓冲区管理增强**：正在努力完善 **Big Graph** 和 **LazyBuffer** 概念，并计划**删除 LazyBuffer** 以改进处理。
  
  - 这包括使用 **WeakKeyDictionary** 跟踪 **UOp Buffers**，以增强 **Tinygrad** 的性能和功能。
  
- **无需 ROCm 的 TinyGrad AMD GPU 支持**：有人询问 **TinyGrad** 是否可以在不安装 **ROCm** 的情况下在 **AMD GPUs** 上进行训练，参考了 George 在直播中提到的剥离 AMD 用户空间的内容。
  
  - 这表明 **Tinygrad** 的 GPU 支持策略可能发生转变，将影响使用 AMD 硬件的用户。

 

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord


- **与 Intel 合作的 AI 开发独家 AMA**：参加于 **11/21 下午 3点 PT** 举行的 [Building with Intel: Tiber AI Cloud and Intel Liftoff](https://lu.ma/agents-hackathon-intel) AMA 会议，深入了解 Intel 的 AI 工具。
  
  - 本次活动提供了与 Intel 专家互动的机会，并学习如何利用他们的资源增强你的 AI 项目。
  
- **Intel Tiber AI Cloud 功能**：会议将展示 **Intel Tiber AI Cloud**，这是一个旨在通过先进计算能力优化 AI 项目的平台。
  
  - 参与者将探索如何利用该平台在 hackathon 活动中实现效率最大化。
  
- **针对初创公司的 Intel Liftoff Program**：讨论将集中在 **Intel Liftoff Program**，该计划为初创公司提供导师指导和技术资源。
  
  - 与会者将发现该计划如何从初期阶段支持他们的开发工作。
  
- **Percy Liang 的开源基础模型**：**Percy Liang** 将发表关于 [Open-Source and Science in the Era of Foundation Models](https://www.youtube.com/live/f3KKx9LWntQ) 的演讲，强调开源对 AI 的重要性。
  
  - Liang 将讨论如何利用社区支持开源基础模型，并强调对 **data**、**compute** 和研究专业知识等大量资源的需求。

 

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord


- **DSPy 引入 VLM 支持**：[DSPy 最近在测试版中增加了对 VLMs 的支持](https://x.com/karthikkalyan90/status/1858609018228355414)，展示了**从图像中提取属性**的功能。
  
  - 一位成员分享了一个示例，演示了如何从网页 **screenshots** 中提取有用的属性，突显了该功能的潜力。
  
- **从截图中提取属性**：该讨论串探讨了从网页 **screenshots** 中**提取有用属性**的技术，展示了 DSPy 的实际应用。
  
  - 这种方法旨在简化开发者与视觉数据的交互方式，引起了人们对 DSPy 工具包中新兴功能的关注。
  
- **DSPy Signatures 中少用英文，多用代码**：一位成员分享到，大多数人在他们的 DSPy signatures 中写了太多的英文；相反，通过简洁的代码可以实现很多功能。
  
  - 他们引用了 [Omar Khattab 的一条推文](https://x.com/lateinteraction/status/1858284772084375784)，该推文强调了*超短 pseudocode* 的有效性。
  
- **使用 DSPy 解决用户名生成问题**：一位用户提出了关于生成多样化用户名的问题，指出存在许多重复项。
  
  - 另一位成员建议禁用 LLM 对象中的缓存，但原用户提到他们已经这样做了。
  
- **通过高方差增加用户名随机性**：为了解决用户名重复的问题，一位成员建议提高 LLM temperature，并在生成名称之前添加故事元素。
  
  - 他们建议使用高 temperature 模型生成故事，并使用低 temperature 模型生成高质量的名称。

 

---

## [LAION](https://discord.com/channels/823813159592001537) Discord


- **发布针对 VLA 模型的 MultiNet 基准测试**：题为“Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks”的新论文在 20 个真实世界任务中评估了 **VLA models**，揭示了关于其性能的关键见解。完整详情请参阅[此处](https://multinet.ai/static/pages/Multinetv01.html)。
  
  - 这项工作旨在推动通用机器人系统的发展，证明了在不同任务中进行系统评估的迫切需求。
  
- **Jina AI 的 VisRAG 演讲**：参加即将举行的 Jina AI 演讲，Shi 将探讨他在 [VisRAG](https://huggingface.co/openbmb/VisRAG-Ret) 上的创新工作，这是一个无需解析的纯视觉 **RAG** 流水线。
  
  - 届时将了解到关于 **VisRAG** 的构建、评估及未来可能性，其训练数据集规模几乎是 ColPali 的三倍。
  
- **领先 VLA 模型之间的性能对比**：对 **GPT-4o**、**OpenVLA** 和 **JAT** 的对比显示，虽然拾取与放置等简单任务尚可应对，但模型在复杂的多步过程中表现挣扎。
  
  - 值得注意的是，结果表明性能随任务和机器人平台的不同而有显著差异，凸显了对 **GPT-4o** 进行复杂 **prompt engineering** 的效用。
  
- **μGATO 简介，一个迷你 VLA 模型**：团队介绍了 **μGATO**，这是一个专为 **MultiNet** 基准测试量身定制的、迷你且易于理解的基准模型，作为推进机器人多模态动作模型的工具。
  
  - Manifold 团队持续的努力预示着更多多模态动作模型的创新即将发布。
  
- **Tasty Hacks 黑客松公告**：一场名为 Tasty Hacks 的新黑客松旨在激励参与者为了创意而非实用性进行创作，摆脱传统黑客松为了获胜而优化的文化。
  
  - 组织者正在寻找善良且极客的人士，愿意在仅有 **20-30 人**的小型环境中组队创作。

 

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord


- **需要 MLOps 指导**：一位成员表达了对从何处开始学习 **MLOps** 的困惑，称：*“这一切都很复杂。”*
  
  - 另一位成员要求进一步说明，询问更具体的问题，强调了在处理 **MLOps** 等复杂话题时清晰沟通的必要性。
  
- **寻求 MLOps 复杂性的澄清**：一位成员指出关于 **MLOps** 的问题过于宽泛，并要求提供更多具体细节。
  
  - 这种互动强调了在处理 **MLOps** 等错综复杂的主题时，精确沟通的必要性。

 

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord


- **Pleias 发布用于 LLM 训练的 Common Corpus**：2024 Builders Accelerator 成员 Pleias 宣布发布 **Common Corpus**，这是用于 **LLM** 训练的最大开源数据集，强调了在宽松许可证下提供训练数据的承诺。[在此查看完整帖子](https://discord.com/channels/1089876418936180786/1306706786824487035)。
  
  - Pleias 指出：*“开源 LLM 生态系统尤其缺乏训练数据的透明度，”* 并表示 **Common Corpus** 旨在解决这一透明度差距。
  
- **Transformer Lab 安排 RAG 演示**：**Transformer Lab** 正在举办一场演示，展示如何在拥有友好 **UI** 的情况下，**无需编码**即可在 **LLM** 上训练、微调、评估和使用 **RAG**。该活动承诺在你的**本地环境**中提供**易于安装**的过程，引起了社区的兴奋。[更多详情](https://discord.com/events/1089876418936180786/1300842793945530378)。
  
  - 社区成员对将 **RAG** 简化集成到工作流中表现出极大的热情。

 

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord


- **DCP 异步 Checkpointing 实现**：[DCP async checkpointing](https://github.com/pytorch/torchtune/pull/2006) 旨在通过一项目前正在开发中的新功能来改进 **TorchTune** 中的中间 **checkpointing**。
  
  - 该 Pull Request 显示，该过程旨在通过将中间 **checkpointing** 时间减少 **80%** 来显著提高效率。
  
- **中间 Checkpointing 时间缩减**：**DCP** 异步 **checkpointing** 的实现有望带来显著的缩减，预计由于方法的改进，**checkpointing** 时间将减少 **80%**。
  
  - 这种方法是优化分布式 **checkpointing** 以获得更好性能的持续努力的一部分。


---

# PART 2: Detailed by-Channel summaries and links

{% if medium == 'web' %}

### **OpenRouter (Alex Atallah) ▷ #**[**announcements**](https://discord.com/channels/1091220969173028894/1092729520181739581/1307072895360438343) (1 条消息):

> - `Perplexity models`
> - `Citations attribute`
> - `Chat completions`

- **Perplexity models Beta 版发布 Grounding Citations**：所有 **Perplexity models** 现在在 Beta 测试中支持新的 `citations` 属性，通过关联链接增强了 completion 响应中提供的信息。
  
  - 此功能使用户能够直接从输出中获取可靠的来源，示例 URL 包括 [BBC News](https://www.bbc.com/news/election/2024/us/results) 和 [CBS News](https://www.cbsnews.com/news/how-electoral-college-works/)。
- **结构化的引用格式增强了易用性**：输出结构不仅包含 completion，还包含 **citations** 数组，其中列出了与 completion 上下文相关的 URL，以便更好地参考。
  
  - 这一改进旨在通过提供更便捷的方式访问 chat completion 中呈现的信息来源，从而提升用户体验。

---

### **OpenRouter (Alex Atallah) ▷ #**[**app-showcase**](https://discord.com/channels/1091220969173028894/1092850552192368710/1307182569120596019) (4 条消息):

> - `Threaded Conversations`
> - `Model Switching`
> - `vnc-lm Discord Bot`
> - `WordPress Chatbot Feature`
> - `Market Competition against Intercom and Zendesk`

- **线程化对话增强**：一位成员分享了已添加对 **threaded conversations** 的支持，允许在线程内部进行的更改反映在后续消息中。
  
  - 初始提示词中的关键词被用于命名线程，使重新加入对话变得更加容易。
- **即时快速模型切换**：此次更新包括在**对话中途切换模型**的能力，只需发送 `+` 后跟部分模型名称即可，同时保持上下文和设置。
  
  - 例如，发送 `+ claude` 即可无缝切换到 `anthropic/claude-3-sonnet:beta`。
- **Launch of vnc-lm Discord Bot**：[vnc-lm](https://github.com/jake83741/vnc-lm) 作为一款集成领先语言模型 API 的 **Discord bot** 被推出，增强了用户交互。
  
  - 提供的 GitHub 链接中的概述强调了它在 Discord 环境中的实用性。
- **新 WordPress 插件功能**：分享了一个 **WordPress 插件** 的更新，其功能包括创建自定义网站聊天机器人，并包含 OpenRouter 支持。
  
  - 详情可见其 [功能页面](https://wpaimuse.com/#features)。
- **竞争对手请注意！**：一条轻松的评论指出，这一新的聊天机器人功能可能会颠覆 **Intercom** 和 **Zendesk** 等竞争对手。
  
  - 该成员带着调侃地提到了这一点，暗示了该领域日益增长的竞争性。

**提到的链接**：

- [未找到标题](https://wpaimuse.com/#features)：未找到描述
- [GitHub - jake83741/vnc-lm: vnc-lm is a Discord bot that integrates leading large language model APIs.](https://github.com/jake83741/vnc-lm)：vnc-lm 是一个集成领先大语言模型 API 的 Discord bot。- jake83741/vnc-lm

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1307079764078628914) (998 条消息🔥🔥🔥):

> - `Gemini Fast Performance` (Gemini 快速性能)
> - `Mistral API Issues` (Mistral API 问题)
> - `Self-Moderated vs OR Moderated APIs` (自托管审核与 OR 托管审核 API)
> - `OpenAI O1 Streaming Feature` (OpenAI O1 流式传输功能)
> - `User Discussions on Prompt Engineering` (用户关于 Prompt Engineering 的讨论)

- **Gemini 实验性模型见解**：用户报告称，标识为 `gemini-exp-1114` 的新 Gemini 实验性模型与早期版本相比，展现出了更强的创造力和更少的审查，使其成为 Prompt 编写中更具动态性的选择。
  
  - 然而，它有时可能会产生乱码响应，或者需要精细的 Prompt 引导以避免过度审查。
- **Mistral API 不稳定性**：用户正经历 `mistral-large-2407` 模型的不稳定性，尽管设置了 Temperature，该模型偶尔仍会返回乱码；而 `mistral-large-2411` 的响应似乎更加理智。
  
  - 讨论涉及输出质量和 Temperature 敏感性的差异，表明 Mistral 模型的性能存在波动。
- **理解自托管审核与 OR 托管审核 API**：社区讨论了自托管审核（Self-Moderated）与 OR 托管审核（OR Moderated）API 之间的区别，重点关注成本影响和接收到被审核内容的可能性。
  
  - 自托管审核模型往往即使请求失败也会产生费用，而非自托管审核版本在触发审核限制时不会产生费用。
- **OpenAI O1 模型支持流式传输**：OpenAI 宣布其 `o1-preview` 和 `o1-mini` 模型现已支持流式传输（Streaming），并向所有付费使用层级的开发者开放访问。
  
  - 这一增强功能将提升使用这些模型的应用程序的交互性，超越了之前实现的伪流式传输（fake streaming）方法。
- **用户互动与幽默**：在技术讨论之余，用户分享了关于各种模型的轻松评论和幽默，包括对比以及关于使用体验的俏皮话。
  
  - 社区保持着活跃的氛围，充满了针对 AI 模型及其能力的持续发展所产生的梗和笑话。

**提到的链接**：

- [Chub Venus AI](https://chub.ai/)：未找到描述
- [来自 Tom's Hardware (@tomshardware) 的推文](https://x.com/tomshardware/status/1852036356697977073?t=yOIlX32itMfrHMraE_sNOw&s=19)：Meta 正在使用超过 100,000 块 Nvidia H100 AI GPU 来训练 Llama-4 —— Mark Zuckerberg 表示 Llama 4 正在一个“比我见过的任何集群都大”的集群上进行训练 https://trib.al/fynPPuR
- [Perspective API](https://perspectiveapi.com/)：未找到描述
- [Bocchi The Rock Hitori Gotoh GIF - Bocchi The Rock Hitori Gotoh Bocchi - Discover & Share GIFs](https://tenor.com/view/bocchi-the-rock-hitori-gotoh-bocchi-anime-dazed-gif-27118680)：点击查看 GIF
- [来自 heiner (@HeinrichKuttler) 的推文](https://x.com/HeinrichKuttler/status/1852958690850332822)：10 万级以上 GPU 规模的乐趣：我们的训练刚刚短暂中断，因为一个步数计数器（step counter）超出了 32 位溢出。😅
- [来自 Tibor Blaho (@btibor91) 的推文](https://x.com/btibor91/status/1857892684565778788/photo/1)：LMSYS Chatbot Arena 上的 "chatgpt-4o-latest-20241111"？
- [SillyTavern - 为高级用户准备的 LLM 前端](https://sillytavern.app/)：未找到描述
- [模型：'setti' | OpenRouter](https://openrouter.ai/setti)：在 OpenRouter 上浏览模型
- [Hey GIF - Hey - Discover & Share GIFs](https://tenor.com/view/hey-gif-13705732610916322705)：点击查看 GIF
- [来自 OpenAI Developers (@OpenAIDevs) 的推文](https://x.com/OpenAIDevs/status/1858609150999359559)：OpenAI o1-preview 和 o1-mini 现已支持流式传输。🌊 https://platform.openai.com/docs/api-reference/streaming 我们已向所有付费使用层级的开发者开放了这些模型的访问权限...
- [揭露我们电话系统的缺陷](https://www.youtube.com/watch?v=wVyu7NB7W6Y)：你能信任你的电话吗？前往 https://brilliant.org/veritasium 开始你的 30 天免费试用，并享受年度高级订阅 8 折优惠。非常感谢...
- [来自 Qwen (@Alibaba_Qwen) 的推文](https://x.com/alibaba_qwen/status/1858469845958074541?s=46)：在 Qwen2.5 发布后，我们听到了社区对处理更长上下文的需求。https://qwenlm.github.io/blog/qwen2.5-turbo/ 今天，我们很自豪地推出新的 Qwen2.5-Turbo 版本...
- [签署请愿书](https://www.change.org/p/push-openrouter-to-include-filtering-for-illegal-content)：推动 OpenRouter 加入对非法内容的过滤。
- [提供商路由 | OpenRouter](https://openrouter.ai/docs/provider-routing)：跨多个提供商路由请求

---

### **OpenRouter (Alex Atallah) ▷ #**[**beta-feedback**](https://discord.com/channels/1091220969173028894/1277894087755829278/1307257825738883074) (21 条消息🔥):

> - `Custom Provider Keys Access Requests`
> - `Integration Beta Feature Requests`

- **Custom Provider Keys 访问请求激增**：多位用户表达了获取 **custom provider keys** 访问权限的愿望，留言如“我很想获得访问权限！”和“申请 custom provider keys 的访问权限”。
  
  - 用户对缺乏回应表示担忧，一位用户提到 *“已经过去 1 天了，还没有人回应我的密钥请求”*。
- **Integration Beta 功能访问需求旺盛**：多位用户询问如何获得 **integration beta features** 的访问权限，突显了成员们对使用 API 的共同兴趣。
  
  - 诸如“你好，我可以申请 integration beta 功能的访问权限吗？”之类的消息反映了获取访问权限的紧迫性和集体兴趣。

---

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1307075348336345208) (723 条消息🔥🔥🔥):

> - `Unsloth Framework Features`
> - `Qwen 2.5 Turbo Release`
> - `Fine-tuning Techniques`
> - `RAG Approach in AI`
> - `Model Performance and Configuration`

- **Unsloth 框架特性**：用户讨论了如何使用 Unsloth 框架，通过 `FastLanguageModel.from_pretrained()` 函数将微调后的 LoRA 权重与基础模型一同加载。
  
  - 该框架允许有效地添加新 token 并调整 embedding 大小，用户发现这对他们的训练过程非常有益。
- **Qwen 2.5 Turbo 发布**：Qwen 2.5 Turbo 模型已发布，支持高达 100 万个 token 的上下文长度，并展示了更快的推理速度。
  
  - 这一进展满足了社区对更长上下文以及在处理海量数据时提升性能的需求。
- **Fine-tuning 技术**：由于模型大小减小，使用 LoRA 进行微调的训练时间比 QLoRA 更快，但处理开销更多。
  
  - 用户指出，只要趋势显示在训练结束时有所下降，训练 loss 的波动是可以接受的。
- **AI 中的 RAG 方法**：讨论了 RAG (Retrieval-Augmented Generation) 方法，强调其适用于问答场景，但执行过程较为复杂。
  
  - 会议强调，为了有效利用 RAG，需要对模型进行彻底的 fine-tuning，尤其是在处理大型数据集时。
- **模型性能与配置**：用户询问了训练期间的 loss 阈值，明确了 0.5 到 0.6 之间的波动通常是可以接受的。
  
  - 社区建议观察 loss 趋势至关重要，并确认了即将支持 Aya Expanse 等模型。

**提到的链接**：

- [AI Mathematical Olympiad - Progress Prize 2](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/545998): 使用人工智能模型解决国家级数学挑战
- [Elements of Statistical Learning: data mining, inference, and prediction. 2nd Edition.](https://hastie.su.domains/ElemStatLearn/): 未找到描述
- [Google Colab](https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing): 未找到描述
- [Tweet from Qwen (@Alibaba_Qwen)](https://x.com/Alibaba_Qwen/status/1858469845958074541): 在 Qwen2.5 发布后，我们听到了社区对处理更长上下文的需求。https://qwenlm.github.io/blog/qwen2.5-turbo/ 今天，我们自豪地推出全新的 Qwen2.5-Turbo 版本...
- [Llama 3.2 - a unsloth Collection](https://huggingface.co/collections/unsloth/llama-32-66f46afde4ca573864321a22): 未找到描述
- [Tweet from VCK5000 Versal Development Card](https://www.xilinx.com/products/boards-and-kits/vck5000.html): AMD VCK5000 Versal 开发卡基于 AMD 7nm Versal™ 自适应 SoC 架构构建，专为使用 Vitis 端到端流程的 (AI) Engine 开发和 AI Inference 开发而设计...
- [Apply to Y Combinator | Y Combinator](https://www.ycombinator.com/apply): 要申请 Y Combinator 项目，请提交申请表。我们每年分两批接收公司。该项目包括每周二的晚宴、与 YC 合伙人的办公时间以及访问权限...
- [GGUF Editor - a Hugging Face Space by CISCai](https://huggingface.co/spaces/CISCai/gguf-editor): 未找到描述
- [Boo GIF - Boo - Discover & Share GIFs](https://tenor.com/view/boo-gif-19787475173016375): 点击查看 GIF
- [unsloth/Qwen2.5-7B-Instruct · Hugging Face](https://huggingface.co/unsloth/Qwen2.5-7B-Instruct): 未找到描述
- [update tokenizer_config.json,config.json,generation_config.json · Qwen/Qwen2.5-Coder-32B-Instruct at b472059](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct/commit/b47205940b83b5b484577359f71ee7b88472df67): 未找到描述
- [unsloth (Unsloth AI)](https://huggingface.co/unsloth): 未找到描述
- [All Our Models | Unsloth Documentation](https://docs.unsloth.ai/get-started/all-our-models): 请查看下方列表，了解我们上传的所有 GGUF、16-bit 和 4-bit bnb 模型
- [Placement Groups — Ray 2.39.0](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html): 未找到描述
- [mistralai/Pixtral-Large-Instruct-2411 at main](https://huggingface.co/mistralai/Pixtral-Large-Instruct-2411/tree/main): 未找到描述
- [Vespa.ai](https://vespa.ai/): 未找到描述
- [Generalized Knowledge Distillation Trainer](https://huggingface.co/docs/trl/en/gkd_trainer): 未找到描述
- [Kquant03 - Overview](https://github.com/Kquant03): 我希望为 AI 领域提供长期价值。...这绝对是任何人都可以实现的，因为可能性确实是无限的。- Kquant03
- [Huggingface GGUF Editor · ggerganov/llama.cpp · Discussion #9268](https://github.com/ggerganov/llama.cpp/discussions/9268): Huggingface GGUF Editor 🎉 看看我最新的项目 🌍✨ 一个功能强大的编辑器，专门用于编辑 GGUF 元数据，并直接从你选择的任何 Huggingface 仓库下载结果...
- [library](https://ollama.com/library): 快速上手并运行大语言模型。
- [How to specify which gpu to use? · vllm-project/vllm · Discussion #691](https://github.com/vllm-project/vllm/discussions/691): 如果我有多个 GPUs，如何指定单独使用哪个 GPU？以前，我使用 accelerate 的 'device_map': 'sequential' 来控制。现在，使用 vllm_engine，是否有...

### **Unsloth AI (Daniel Han) ▷ #**[**off-topic**](https://discord.com/channels/1179035537009545276/1179039861576056922/1307118484441927791) (9 条消息🔥):

> - `Jordan Peterson GIF`
> - `YouTube 内容`
> - `澳洲人的行为`
> - `随机链接幽默`
> - `兼职工作经历`

- **Jordan Peterson 的“Precisely”时刻**：一个 Jordan Peterson 自信地说出“precisely”的 GIF 成为讨论焦点，展示了他令人印象深刻的表情。
  
  - 这个 GIF 具有幽默感，特别是在提供的背景下，他穿着西装坐在黑色背景前。
- **分享了 YouTube 视频**：分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=0vfiR7Z0msQ) 链接，尽管对话中没有详细说明具体内容。
  
  - 对其内容的反应引发了轻松、惊讶的惊叹“holy shit the madman”。
- **独特的澳洲性格**：出现了“Aussies are built different”（澳洲人与众不同）的说法，暗示了关于澳大利亚文化或性格特征的幽默观念。
  
  - 这一评论引起了笑声，并在轻松的语境中引起了对澳大利亚人的刻板印象共鸣。
- **随机链接幽默**：关于“PERPLEXITY FREE CLICK LINK CLICK LINK”的幽默评论反映了一个随机的人急切的行动号召，许多人觉得很有趣。
  
  - 另一位用户指出在 Ollama Discord 中也见过类似行为，为对话增添了共同的笑声。
- **开始兼职工作**：一位成员宣布开始一份兼职工作，表达了他们随着承担新工作从“没钱到稍微有点钱”的转变。
  
  - 这引发了一个关于小心任务的玩笑警告，比如不要把牛奶弄掉，提供了轻松的同伴情谊。

 

**提到的链接**：[Jordan Peterson Jbp GIF - Jordan Peterson JBP Precisely - 发现并分享 GIF](https://tenor.com/view/jordan-peterson-jbp-precisely-jordan-b-peterson-gif-26104924)：点击查看 GIF

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1307079690258878494) (113 条消息🔥🔥):

> - `微调 Llama 3`
> - `训练数据集大小的影响`
> - `Gradient Accumulation 修复`
> - `加载模型 Adapters`
> - `关于 Unsloth 的功能问题`

- **微调 Llama 3 的挑战**：用户报告了微调 **Llama 3.2** 的问题，并强调了改善训练损失的挑战，其中一位用户在不同数据集大小上取得了不同的结果。
  
  - 建议包括评估数据集质量和调整训练参数，如 **learning rate** 和 **gradient accumulation**。
- **数据集大小与模型性能**：一位用户指出，尽管质量一致，但将数据集从 **20k 增加到 50k** 反而对性能产生了负面影响，引发了对更大型数据集必要性的担忧。
  
  - 其他人强调 **质量优于数量** 至关重要，认为更多的样本并不保证更好的模型结果。
- **Gradient Accumulation 问题已解决**：讨论强调了 Unsloth 中与 **gradient accumulation** 相关的 Bug 已被修复，从而提高了显存利用率和训练效率。
  
  - 用户鼓励进一步探索 Unsloth 的博客，以获取有关修复和最佳实践的详细见解。
- **集成模型 Adapters**：一位用户询问在没有下载模型权重的情况下，如何将 **qLoRA adapters** 与基础模型一起加载，这促使了将 adapters 与基础模型合并的建议。
  
  - 这种方法可以方便在受限环境中加载和使用模型。
- **对新模型的支持**：关于支持 Cohere 的 **Aya Expanse** 出现了一个问题，用户在尝试使用 **CohereForCausalLM** 对象时遇到了与模型属性相关的错误。
  
  - 这表明新模型与现有框架之间持续存在的兼容性咨询。

**提到的链接**：

- [Google Colab](https://colab.research.google.com/drive/13_z6x9ejloE8mC1IFR8qPmP-7YaFknN9#scrollTo=IqM-T1RTzY6C.)：未找到描述
- [Google Colab](https://colab.research.google.com/drive/1XXUXoupaMd-x-7Sc1iRA0A0wOGrn62sn?usp=sharing)：未找到描述
- [Blog](https://unsloth.ai/blog)：未找到描述
- [Function Calling Datasets, Training and Inference](https://www.youtube.com/watch?v=hHn_cV5WUDI&list=PLWG1mVtuzdxdL-_2E3YC9Az5ocyhJA9k7)：➡️ Trelis Function-calling 模型和脚本：https://trelis.com/function-calling/ ➡️ ADVANCED-inference 仓库：https://trelis.com/enterprise-server-api-and-i...
- [Dataset formats and types](https://huggingface.co/docs/trl/en/dataset_formats)：未找到描述

---

### **Unsloth AI (Daniel Han) ▷ #**[**research**](https://discord.com/channels/1179035537009545276/1257011997250424842/1307420243538219049) (12 messages🔥):

> - `Fast Forward 优化策略`
> - `量化的极限`
> - `PrefixQuant 技术`
> - `LaTent Reasoning Optimization (LaTRO)`

- **Fast Forward 加速 SGD 训练**：一篇新论文介绍了 **Fast Forward**，这是一种通过重复最新的优化器步骤直到 loss 停止改善来加速 SGD 训练的方法，与使用 Adam 的标准 SGD 相比，FLOPs 减少了高达 **87%**。
  
  - 该方法在各种模型和任务中得到了验证，显示出在不损害性能的情况下提高了训练速度。
- **训练数据对量化极限的影响**：Tim Dettmers 强调了一篇关键论文，该论文提供了强有力的证据，表明随着我们在更多 tokens 上进行训练，模型需要**更高的精度**来保持有效性，并指出我们可能正在达到量化的极限。
  
  - 研究表明，如果进行训练后量化（post-training quantization），使用更多的预训练数据进行过度训练甚至可能是**有害的**。
- **PrefixQuant 提供高效的静态量化**：**PrefixQuant** 技术在线下隔离离群 token，以在无需重新训练的情况下简化量化，使高效的 per-tensor 静态量化能够超越动态量化方法。
  
  - 在 Llama-3-8B 中使用 PrefixQuant，该技术与之前的方法相比，在**准确率**和推理速度上都有显著提升。
- **通过 LaTRO 解锁推理能力**：**LaTRO** 提出了一个框架，通过从潜在分布中采样并在训练期间自主提高推理质量，来优化 LLM 中的推理能力。
  
  - 实验表明，LaTRO 将 GSM8K 上的 zero-shot 准确率提高了 **12.5%**，表明可以通过自我提升策略访问潜在推理能力。

**提到的链接**：

- [Fast Forwarding Low-Rank Training](https://aclanthology.org/2024.emnlp-main.535/): Adir Rahamim, Naomi Saphra, Sara Kangaslahti, Yonatan Belinkov. Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 2024.
- [Tim Dettmers (@Tim_Dettmers) 的推文](https://x.com/Tim_Dettmers/status/1856338240099221674?t=_lF6DH9tWzo4u6-fFeukwg&s=33): 这是很长一段时间以来最重要的一篇论文。它有力地证明了我们正在达到量化的极限。论文指出：你训练的 tokens 越多，你需要的精度就越...
- [PrefixQuant: Static Quantization Beats Dynamic through Prefixed Outliers in LLMs](https://arxiv.org/abs/2410.05265): 量化对于部署 LLM 至关重要，它可以提高内存效率和推理速度。现有的激活量化方法主要解决通道级的离群值...
- [GitHub - ChenMnZ/PrefixQuant: An algorithm for static activation quantization of LLMs](https://github.com/ChenMnZ/PrefixQuant): 一种用于 LLM 静态激活量化的算法 - ChenMnZ/PrefixQuant
- [Language Models are Hidden Reasoners: Unlocking Latent Reasoning Capabilities via Self-Rewarding](https://arxiv.org/abs/2411.04282): LLM 已经展示了令人印象深刻的能力，但在需要多个步骤的复杂推理任务中仍然面临困难。虽然像 Chain-of-Thought (CoT) 这样基于提示的方法可以改进...
- [GitHub - SalesforceAIResearch/LaTRO](https://github.com/SalesforceAIResearch/LaTRO): 为 SalesforceAIResearch/LaTRO 的开发做出贡献。

---

### **Perplexity AI ▷ #**[**announcements**](https://discord.com/channels/1047197230748151888/1047204950763122820/1307142891323527210) (2 messages):

> - `加拿大学生优惠`
> - `Perplexity Shopping 发布`
> - `Buy with Pro 功能`

- **加拿大学生可获得一个月免费 Pro**：Perplexity 为**加拿大学生**提供使用学生邮箱获取**一个月免费 Pro** 的优惠，有效期仅限两周。
  
  - 鼓励学生通过[此链接](http://pplx.ai/ca-student)注册并与朋友分享该优惠。
- **Perplexity Shopping 改变在线购物体验**：**Perplexity Shopping** 作为研究和购买产品的一站式解决方案推出，提升了用户体验。
  
  - 用户现在可以通过 [Perplexity Shopping](https://perplexity.ai/shopping) 平台享受选定产品的**一键结账**和**免运费**服务。
- **'Buy with Pro' 彻底改变交易方式**：**'Buy with Pro'** 功能允许美国 Perplexity Pro 用户在应用内进行原生交易，支持无缝购物体验。
  
  - 该功能帮助用户购买各种物品，包括**电子产品**和提升居家体验的产品。

 

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1307076578286506105) (599 messages🔥🔥🔥):

> - `Perplexity Pro 订阅问题`
> - `购物功能与变现`
> - `上下文记忆限制变更`
> - `按地区的模型可用性`
> - `用户体验担忧`

- **对 Perplexity Pro 订阅的不满**：用户对 Perplexity Pro 订阅的近期变化表示沮丧，特别是未经事先通知就移除了 Opus 模型，许多人觉得在订阅服务方面受到了误导。
  
  - 随着用户感到付费订阅不再提供预期价值，不满情绪日益增长，促使一些人询问退款事宜。
- **购物功能的引入引发担忧**：新的购物功能遭到了质疑，一些用户认为它将变现置于用户体验之上，特别是平台现在出现了联盟链接和广告。
  
  - 有投诉称这些变化损害了 AI 助手的核心功能，用户觉得这削弱了他们最初注册时所期待的服务。
- **上下文记忆容量的变化**：多位用户报告称，模型的上下文记忆容量已从 32k tokens 减少到 16k tokens，导致用户对模型在长对话中的有效性感到不满。
  
  - 容量的减少加剧了用户的挫败感，他们认为自己支付的订阅费没有获得相应的服务。
- **基于地区的访问权限和模型可用性**：美国以外的用户报告称，他们被限制使用较差的模型（如 GPT-3.5），或者默认无法使用 Claude 和 Grok，这让他们感到受到了不公平对待。
  
  - 用户呼吁在模型可用性方面保持透明，并关注欧盟用户未来是否能获得与美国用户相同的服务。
- **呼吁改进和透明度**：在不满情绪日益增长的情况下，用户敦促 Perplexity 优先考虑透明度并遵守消费者法律，特别是在欧盟和澳大利亚关于其订阅服务的规定。
  
  - 许多人认为重点应该重新回到用户体验上，而不是那些损害原始服务的广告和购物功能。

**提到的链接**：

- [Jogo da Cobrinha](https://jogo-cobrinha-nine.vercel.app/)：未找到描述
- [Putin Ketawa GIF - Putin Ketawa - Discover & Share GIFs](https://tenor.com/view/putin-ketawa-gif-24022608)：点击查看 GIF
- [Holo Spice And Wolf GIF - Holo Spice and wolf Holo the wise wolf - Discover & Share GIFs](https://tenor.com/view/holo-spice-and-wolf-holo-the-wise-wolf-horo-korbo-gif-13009516793083034180)：点击查看 GIF
- [来自 Ryan Putnam (@RypeArts) 的推文](https://x.com/rypearts/status/1857512981699113338?s=61)：周五氛围 ✨
- [支持的模型 - Perplexity](https://perplexity.mintlify.app/guides/model-cards)：未找到描述
- [无标题](https://perplexity.mintlify.app/guides/pricing)：未找到描述
- [Complexity - Perplexity AI Supercharged - Chrome 网上应用店](https://chromewebstore.google.com/detail/complexity-perplexity-ai/ffppmilmeaekegkpckebkeahjgmhggpj)：⚡ 增强你的 Perplexity AI
- [Google 的 AI 聊天机器人告诉寻求作业帮助的学生“请去死”](https://www.newsweek.com/googles-ai-chatbot-tells-student-seeking-help-homework-please-die-1986471)：Google 发言人周五早上告诉 Newsweek，公司“严肃对待这些问题”。
- [来自 Perplexity (@perplexity_ai) 的推文](https://x.com/perplexity_ai/status/1858556244891758991?s=46)：介绍 Perplexity Shopping：一个可以研究和购买产品的一站式解决方案。这标志着我们在服务用户方面迈出了一大步——直接实现无缝的原生操作...
- [Apple Apple Iphone GIF - Apple Apple Iphone Apple Iphone13 - Discover & Share GIFs](https://tenor.com/view/apple-apple-iphone-apple-iphone13-apple-iphone13pro-apple-iphone13pro-max-gif-23088031)：点击查看 GIF
- [Dario Amodei：Anthropic CEO 谈论 Claude、AGI 以及 AI 与人类的未来 | Lex Fridman Podcast #452](https://www.youtube.com/watch?v=ugvHCXCOmm4&t=3312s)：Dario Amodei 是 Anthropic 的 CEO，该公司开发了 Claude。Amanda Askell 是一位研究 Claude 性格和个性的 AI 研究员。Chris...
- [Perplexity AI 应用](https://www.pling.com/p/2107698/)：Perplexity AI 桌面应用是由 inulute 开发的非官方应用程序，它利用 Electron 将 AI 语言处理的能力带到你的桌面。该应用程序旨在...
- [Sad Hamster Sadhamster GIF - Sad hamster Sadhamster Hammy - Discover & Share GIFs](https://tenor.com/view/sad-hamster-sadhamster-hammy-hampter-popcat-gif-6561088576429774594)：点击查看 GIF
- [Streamlit](https://pingle.ai/)：未找到描述
- [Directive 1993/13 - 消费者合同中的不公平条款 - 欧盟观察](https://www.eumonitor.eu/9353000/1/j4nvk6yhcbpeywk_j9vvik7m1c3gyxp/vitgbghrpltm)
  
  ：未找到描述

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1307098898623627336) (20 messages🔥):

> - `AI 生成视频游戏`
> - `自主 ML Engineer`
> - `印度 AI 就绪度`
> - `协程 (Coroutines) 实现`
> - `茶叶化学`

- **全球首款 AI 生成视频游戏**：Perplexity AI 分享了一段关于**全球首款 AI 生成视频游戏**的视频，并讨论了相关话题，如*虚假职位 (Ghost Jobs) 充斥市场*以及**志留纪假说 (Silurian Hypothesis)**。
  
  - 你可以在[这里](https://www.youtube.com/embed/CCacIx9LHh4)观看视频，深入了解这一突破性成就。
- **揭秘自主 ML Engineer**：关于**全球首款自主 ML Engineer**的话题浮出水面，重点介绍了用于 Machine Learning 应用的自主系统的进展。
  
  - 有关此项进展如何改变 AI 驱动型企业工作流程的详细信息已分享，可在此处查看 [here](https://www.perplexity.ai/page/autonomous-ml-engineer-neo-cKFW.EpTToS3YACUcEWoOQ)。
- **印度 AI 就绪度下降**：围绕**印度 AI 就绪度下降**及其对技术增长的影响展开了讨论。
  
  - 可以通过[这篇文章](https://www.perplexity.ai/page/indian-ai-readiness-declines-z5MUukLsQS.qRs_lqNCN3w)详细了解对话内容。
- **协程 (Coroutines) 如何实现**：成员们讨论了关于**协程如何实现**的技术细节，强调了它们在编程中的效率。
  
  - 你可以在[这里](https://www.perplexity.ai/search/how-are-coroutines-implemented-L3.qYXw9S.etxmzGqjSK_Q#0)深入探讨这一讨论。
- **了解茶叶化学**：一个关于**茶叶化学**的查询引发了对茶叶生产中涉及的化学化合物及其影响的探索。
  
  - 如需全面了解，请参阅此处分享的详情 [here](https://www.perplexity.ai/search/explain-the-chemistry-of-tea-a-47VE5i3JQBSomYFFzD3kFQ)。

 

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1307274173961338911) (18 messages🔥):

> - `API 计费问题`
> - `Leo 的摘要生成问题`
> - `Reddit API 功能`
> - `Make.com 模块反馈`

- **Pro 订阅用户 API 计费说明**：一位成员询问超过 5 美元的 API 额度是否会扣除信用卡费用；另一位成员回答说，关闭自动充值 (auto top-up) 可以防止这种情况。
  
  - 关于在删除并重新添加 API Key 后是否能收到 5 美元赠送额度存在进一步的困惑，建议联系支持部门。
- **Leonardo 摘要错误**：一位成员对 **Leonardo** 无法总结页面表示失望，即使在测试了不同的上下文大小 (context sizes) 后也是如此。
  
  - 他们确认该问题在不同账号和多个版本中持续存在，引发了对功能的担忧。
- **Reddit API 链接的不确定性**：有一个关于 Reddit API 是否仍在运行的查询，因为它不再返回链接。
  
  - 这表明通过 API 访问 Reddit 内容可能存在潜在中断。
- **改进 Make.com 模块功能的请求**：一位成员建议增强 **Make.com** 模块，特别是延长超时 (timeout) 时间。
  
  - 他们指出 **Sonar Huge** 经常超时，并表示希望有可自定义的超时选项和第三方模型集成。

 

---

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1307073065531736104) (441 messages🔥🔥🔥):

> - `Mistral Large 模型`
> - `LLaVA-o1 对比 Pixtral`
> - `Gradio 与 MusicGen`
> - `Hugging Face API 配额`
> - `数据集贡献`

- **Mistral 与 Pixtral 模型发布**：Mistral 宣布发布 Pixtral Large 模型，展示了前沿级多模态性能以及对高分辨率图像的兼容性。
  
  - 该模型为开放权重 (open-weights) 并可用于研究，较之前的 Mistral 模型有显著进步。
- **LLaVA-o1 与 Pixtral 的对比**：社区讨论了 LLaVA-o1 和 Pixtral 的性能对比，观点认为 Pixtral Large 在推理任务上优于 LLaVA。
  
  - 虽然 LLaVA-o1 有其优势，但 Pixtral 以卓越的图像理解能力著称。
- **Gradio 与 API 配额的挑战**：用户报告称，即使在 Gradio 网页上成功生成了音乐，但在使用 Gradio 的 API 生成音乐时仍遇到配额耗尽的问题。
  
  - 针对在专业版中使用 ZeroGPU 所面临的限制提出了担忧。
- **Hub-Stats 数据集与排行榜**：Hub-Stats 数据集上的新排行榜根据用户的参与度进行排名，社区内分享了一些显著的排名。

- 社区成员热情地查看他们的排名，这表明了他们对参与 Hugging Face 平台的浓厚兴趣。
- **合成数据生成与微调**：一位用户询问了关于为微调小型 LLM 创建合成数据的建议，强调了对实用资源的需求。
  
  - 对话表明，人们对通过自定义数据集提高 LLM 性能的工具和方法有着更广泛的兴趣。

**提到的链接**：

- [GGUF Model VRAM Calculator - a Hugging Face Space by DavidAU](https://huggingface.co/spaces/DavidAU/GGUF-Model-VRAM-Calculator)：未找到描述
- [The /llms.txt file – llms-txt](https://llmstxt.org/)：一项旨在标准化使用 /llms.txt 文件的提案，用于在推理时提供信息以帮助 LLM 使用网站。
- [Military Assets Dataset (12 Classes -Yolo8 Format)](https://www.kaggle.com/datasets/rawsi18/military-assets-dataset-12-classes-yolo8-format)：RAW (Ryan Madhuwala) 提供的军事目标检测数据集
- [Pixtral Large](https://mistral.ai/news/pixtral-large/)：Pixtral 成长了。
- [HuggingChat](https://huggingface.co/chat/)：让社区最好的 AI 聊天模型对所有人可用。
- [Use Ollama with any GGUF Model on Hugging Face Hub](https://huggingface.co/docs/hub/en/ollama)：未找到描述
- [AdaLoRA](https://huggingface.co/docs/peft/package_reference/adalora)：未找到描述
- [Yeah I Guess Youre Kinda Right Kyle Broflovski GIF - Yeah I Guess Youre Kinda Right Kyle Broflovski Tolkien - Discover & Share GIFs](https://tenor.com/view/yeah-i-guess-youre-kinda-right-kyle-broflovski-tolkien-tolkien-black-south-park-gif-23109987)：点击查看 GIF
- [Text Behind Image - a Hugging Face Space by ysharma](https://huggingface.co/spaces/ysharma/Text_Behind_Image)：未找到描述
- [FLUX.1 Dev Inpainting Model Beta GPU - a Hugging Face Space by ameerazam08](https://huggingface.co/spaces/ameerazam08/FLUX.1-dev-Inpainting-Model-Beta-GPU)：未找到描述
- [Venom Spiderman GIF - Venom Spiderman - Discover & Share GIFs](https://tenor.com/view/venom-spiderman-gif-25099253)：点击查看 GIF
- [That Sounds Nasty Nasty GIF - That Sounds Nasty Nasty Gross - Discover & Share GIFs](https://tenor.com/view/that-sounds-nasty-nasty-gross-disgusting-eww-gif-15491188)：点击查看 GIF
- [Cat Drugs GIF - Cat Drugs Tripping - Discover & Share GIFs](https://tenor.com/view/cat-drugs-tripping-funny-animals-gif-13749008)：点击查看 GIF
- [Cat Fire GIF - Cat Fire Flamethrower - Discover & Share GIFs](https://tenor.com/view/cat-fire-flamethrower-burn-on-fire-gif-7684110515453159552)：点击查看 GIF
- [Smeagol My GIF - Smeagol My Precious - Discover & Share GIFs](https://tenor.com/view/smeagol-my-precious-lord-of-gif-20799219)：点击查看 GIF
- [Record for the Highest Scoring Scrabble Move - Scrabulizer](https://www.scrabulizer.com/blog/post/3)：未找到描述
- [Tweet from AK (@_akhaliq)](https://x.com/_akhaliq/status/1858378159588040774)：LLaVA-o1 让视觉语言模型逐步推理
- [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)：画廊示例：scikit-learn 1.5 发布亮点、scikit-learn 1.4 发布亮点、手写数字数据上的 K-Means 聚类演示、主成分回归 vs 部分...
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1gtrtfw/beer_money_ad_make_a_hf_space_runpod_template_for/)：未找到描述
- [Model memory estimator](https://huggingface.co/docs/accelerate/main/en/usage_guides/model_size_estimator)：未找到描述
- [mistralai/Mistral-Large-Instruct-2411 at main](https://huggingface.co/mistralai/Mistral-Large-Instruct-2411/tree/main)：未找到描述
- [unsloth/Qwen2.5-7B-Instruct-bnb-4bit · Hugging Face](https://huggingface.co/unsloth/Qwen2.5-7B-Instruct-bnb-4bit)：未找到描述
- [unsloth/Qwen2.5-7B-Instruct · Hugging Face](https://huggingface.co/unsloth/Qwen2.5-7B-Instruct)：未找到描述
- [GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct · Hugging Face](https://huggingface.co/GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct)：未找到描述
- [Gemma2](https://huggingface.co/docs/transformers/main/en/model_doc/gemma2)：未找到描述
- [unsloth (Unsloth AI)](https://huggingface.co/unsloth/)：未找到描述
- [Tutorial: How to convert HuggingFace model to GGUF format [UPDATED] · ggerganov/llama.cpp · Discussion #7927](https://github.com/ggerganov/llama.cpp/discussions/7927)：我想制作这个教程是因为最近在这个 PR 中所做的更改，这些更改改变了处理转换的方式。下载 Hugging Face 模型 来源：http...
- [huggingchat/chat-ui · Discussions](https://huggingface.co/spaces/huggingchat/chat-ui/discussions)：未找到描述

- [GitHub - PKU-YuanGroup/LLaVA-o1](https://github.com/PKU-YuanGroup/LLaVA-o1): 通过在 GitHub 上创建账户，为 PKU-YuanGroup/LLaVA-o1 的开发做出贡献。
- [GitHub - unslothai/unsloth: Finetune Llama 3.2, Mistral, Phi, Qwen 2.5 & Gemma LLMs 2-5x faster with 80% less memory](https://github.com/unslothai/unsloth): 以 2-5 倍的速度和减少 80% 的内存消耗对 Llama 3.2, Mistral, Phi, Qwen 2.5 和 Gemma LLM 进行微调 - unslothai/unsloth
- [GitHub - starsnatched/RoboLlama: My experiment on turning Meta's Llama 3.2 1B into a robotics-ready model](https://github.com/starsnatched/RoboLlama): 我将 Meta 的 Llama 3.2 1B 转换为机器人就绪（robotics-ready）模型的实验 - starsnatched/RoboLlama
- [GitHub - starsnatched/RoboLlama: My experiment on turning Meta's Llama 3.2 1B into a robotics-ready model](https://github.com/starsnatched/RoboLlama.git): 我将 Meta 的 Llama 3.2 1B 转换为机器人就绪（robotics-ready）模型的实验 - starsnatched/RoboLlama
- [Language Models are Hidden Reasoners: Unlocking Latent Reasoning Capabilities via Self-Rewarding](https://arxiv.org/abs/2411.04282): 语言模型是隐藏的推理者：通过自我奖励解锁潜在推理能力。大型语言模型 (LLM) 已展现出令人印象深刻的能力，但在需要多个步骤的复杂推理任务中仍然面临困难。虽然像思维链 (CoT) 这样基于提示的方法可以提...
- [GitHub - SalesforceAIResearch/LaTRO](https://github.com/SalesforceAIResearch/LaTRO): 通过在 GitHub 上创建账户，为 SalesforceAIResearch/LaTRO 的开发做出贡献。

---

### **HuggingFace ▷ #**[**today-im-learning**](https://discord.com/channels/879548962464493619/898619964095860757/1307422402183168020) (6 messages):

> - `Neuralink Updates`
> - `Transformer Code Issues`
> - `Tunguska-39B Updates`

- **Neuralink 宣布回归**：一位名为 **neuralink** 的成员在短暂缺席后宣布回到聊天中。
  
  - 他们分享了一个随性的更新，表示最近学到了一些新东西。
- **Transformer 代码求助**：一位用户因遇到多个错误，请求协助处理 **transformer code**。
  
  - 他们鼓励其他人通过私信提供帮助。
- **Tunguska-39B README 更新**：一位名为 **drummer_** 的用户为 Tunguska-39B 模型更新了 [README.md](https://huggingface.co/BeaverAI/Tunguska-39B-v1b-GGUF/blob/main/README.md)，增加了新的实验见解。
  
  - 他们分享的研究发现表明，**上采样（upscaling）可以为进一步训练提供空间**，尽管他们指出在数据集中复制层（duplicating layers）存在挑战。

 

**提到的链接**：[README.md · BeaverAI/Tunguska-39B-v1b-GGUF at main](https://huggingface.co/BeaverAI/Tunguska-39B-v1b-GGUF/blob/main/README.md)：未找到描述

 

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1307219320023027792) (15 messages🔥):

> - `Molecular Machine Learning at NeurIPS 2024`
> - `Magic Quill`
> - `Neural Network Communication`
> - `VLMs and their capabilities`
> - `HtmlRAG in Retrieval-Augmented Generation`

- **探索 NeurIPS 2024 的分子机器学习作品**：今年的 NeurIPS 重点展示了 **molecular modeling**（分子建模）和 **biological sciences**（生物科学）方面的创新，特别是蛋白质语言建模和 **molecular property prediction**（分子属性预测）。GitHub 上提供了一个综合资源，收录了会议中所有相关的论文 [此处](https://github.com/azminewasi/Awesome-MoML-NeurIPS24)。
  
  - 这些进展解决了 **drug discovery**（药物研发）和 **protein engineering**（蛋白质工程）中的挑战，有望改进预测建模技术。
- **Magic Quill 的灵活性令人印象深刻**：**Magic Quill** 是 AI 领域的一个创新工具，能够轻松处理各种编辑任务，正如其官方 [Hugging Face Space](https://huggingface.co/spaces/AI4Editing/MagicQuill) 中所展示的那样。其令人耳目一新的功能获得了用户的积极响应。
  
  - 鼓励成员们探索其功能，这些功能可以无缝融入各种应用中。
- **重新思考神经网络通信**：最近发表在 *Nature* 上的一篇论文讨论了自我驱动的动物行为如何启发神经网络架构及其通信机制 [链接](https://www.nature.com/articles/s41586-024-08145-x)。研究表明，模块化组件有助于实现与任务完成相关的可预测行为。
  
  - 这些发现如何显著重塑我们对神经网络的理解受到了高度关注。
- **视觉语言模型的未来**：一篇文章讨论了 **Vision Language Models (VLM)**，它们整合了**图像和文本**来处理各种生成任务。它们具有强大的 zero-shot 能力，并能很好地适应不同类型的图像输入，使其成为 2024 年的热门话题 [链接](https://changgy.com/blog/ctb-1-vision-language-models-in-2024)。

- 该博客强调了 VLM 在不同领域的演进和潜在应用。
- **HtmlRAG 增强检索增强生成系统**：**HtmlRAG** 的引入提出在 RAG 流程中使用 HTML 格式，以保留传统纯文本方法中丢失的结构和语义信息 [link](https://arxiv.org/abs/2411.02959)。这种新方法解决了当前 LLM 在知识检索方面的重大局限性。
  
  - 这些增强功能可能会带来更好的建模和信息检索能力，从而提升未来应用的性能。

**提到的链接**：

- [HtmlRAG: HTML is Better Than Plain Text for Modeling Retrieved Knowledge in RAG Systems](https://arxiv.org/abs/2411.02959)：检索增强生成 (RAG) 已被证明可以提高知识能力并缓解 LLM 的幻觉问题。Web 是 RAG 系统中使用的外部知识的主要来源...
- [MagicQuill - a Hugging Face Space by AI4Editing](https://huggingface.co/spaces/AI4Editing/MagicQuill)：未找到描述
- [Portable acceleration of CMS computing workflows with coprocessors as a service](https://arxiv.org/abs/2402.15366)：大型科学实验（如 CERN LHC 的 CMS 实验）的计算需求在未来几十年将大幅增加。为了补充未来软件性能的提升...
- [CTB-1: Vision Language Models in 2024 [CTB]](https://changgy.com/blog/ctb-1-vision-language-models-in-2024)：1 简介 视觉语言模型 (VLM) 是生成式模型，可以同时从图像和文本中学习以处理多种任务。
- [OpenCV: Detection of ChArUco Boards](https://docs.opencv.org/3.4/df/d4a/tutorial_charuco_detection.html)：未找到描述
- [Home](https://github.com/ritabratamaiti/AnyModal/wiki)：AnyModal 是一个灵活的多模态语言模型框架 - ritabratamaiti/AnyModal
- [Spontaneous behaviour is structured by reinforcement without explicit reward - Nature](https://www.nature.com/articles/s41586-022-05611-2?fromPaywallRec=false)：光度记录和光遗传学操作表明，小鼠背外侧纹状体中的多巴胺波动调节了自发行为模块的使用、排序和活力...
- [A cellular basis for mapping behavioural structure - Nature](https://www.nature.com/articles/s41586-024-08145-x)：小鼠通过使用内侧前额叶皮层中的神经元来概括复杂的任务结构，这些神经元编码任务目标的进展并嵌入行为序列。
- [GitHub - azminewasi/Awesome-MoML-NeurIPS24: ALL Molecular ML papers from NeurIPS'24.](https://github.com/azminewasi/Awesome-MoML-NeurIPS24)：NeurIPS'24 的所有分子 ML 论文。通过在 GitHub 上创建账户为 azminewasi/Awesome-MoML-NeurIPS24 做出贡献。

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1307144381538504786) (40 条消息🔥):

> - `AnyModal Framework`
> - `RoboLlama Robotics Model`
> - `Kaggle Generative AI Course`
> - `YouTube Transcript Tool`
> - `Dataset for Visual Language Models`

- **AnyModal 框架介绍**：一位开发者分享了关于 **AnyModal** 的见解，这是一个灵活的框架，旨在将各种数据类型与 LLM 集成，包括 **LaTeX OCR** 和 **image captioning**（图像描述）等功能。
  
  - 欢迎对这个正在进行的项目提供反馈和贡献，该项目旨在提升其多模态能力。
- **用于机器人集成的 RoboLlama 项目**：**Starsnatched** 正在致力于将 Meta 的 **Llama 3.2 1B** 转换为适用于机器人的模型，并结合了 **vision encoders** 和 **diffusion layers** 以增强功能。
  
  - 该模型专注于仅训练扩散层和投影层，同时保持核心 **ViT** 和 **LLM** 处于冻结状态。
- **Kaggle 生成式 AI 课程进展**：一位用户分享了他们在五天内完成 **Kaggle X Google Generative AI Course** 的经历，并邀请他人在 LinkedIn 上查看其进度。
  
  - 这体现了社区对学习和使用生成式 AI 工具的投入。
- **YouTube 字幕工具开发**：一名成员提到创建了一个 Python 工具来批量下载 **YouTube transcripts**，尽管目前尚未准备好公开分享。
  
  - 该工具是在 ChatGPT 的协助下开发的，旨在简化收集视频字幕的过程。
- **视觉语言模型新数据集**：一位用户发布了一个用于视觉语言模型 (VLM) 的**新基准数据集**，其中包括直接的 **YouTube 视频**和评论，用于评估目的。
  
  - 该数据集旨在揭示 VLM 在准确处理未见数据方面的局限性，这些局限性通常会导致幻觉 (hallucinations)。

**提到的链接**：

- [Stable-Diffusion-Inpainting With SAM - Sanshruth 的 Hugging Face Space](https://huggingface.co/spaces/Sanshruth/Stable-Diffusion-Inpainting_with_SAM)：未找到描述
- [Terminator GIF - Terminator - 发现并分享 GIF](https://tenor.com/view/terminator-gif-17996566)：点击查看 GIF
- [Kokoro TTS - hexgrad 的 Hugging Face Space](https://hf.co/spaces/hexgrad/Kokoro-TTS)：未找到描述
- [What Did GIF - What Did You - 发现并分享 GIF](https://tenor.com/view/what-did-you-just-say-gif-27520460)：点击查看 GIF
- [Neil Raymond Schroeder / CMS - ECAL - Scales and Smearings · GitLab](https://gitlab.cern.ch/nschroed/cms-ecal-scales-and-smearings)：一个用于推导电子能量尺度的剩余刻度和额外模糊的新 Python 框架
- [GitHub - ritabratamaiti/AnyModal: AnyModal 是一个灵活的多模态语言模型框架](https://github.com/ritabratamaiti/AnyModal)：AnyModal 是一个灵活的多模态语言模型框架 - ritabratamaiti/AnyModal
- [GitHub - unixwzrd/chatgpt-chatlog-export: ChatGPT 聊天记录导出，一种以 JSON 格式导出整个 ChatGPT 对话的轻量级方法。](https://github.com/unixwzrd/chatgpt-chatlog-export)：ChatGPT 聊天记录导出，一种以 JSON 格式导出整个 ChatGPT 对话的轻量级方法。 - unixwzrd/chatgpt-chatlog-export
- [Release v1.8.0 · yjg30737/pyqt-openai](https://github.com/yjg30737/pyqt-openai/releases/tag/v1.8.0)：更新内容：你可以导入“句子形式”提示词的 CSV 文件。这是为了支持 awesome_chatgpt_prompt。额外支持 llamaindex 扩展名 '.docx', '....
- [GitHub - starsnatched/RoboLlama: 我将 Meta 的 Llama 3.2 1B 转换为机器人就绪模型的实验](https://github.com/starsnatched/RoboLlama)：我将 Meta 的 Llama 3.2 1B 转换为机器人就绪模型的实验 - starsnatched/RoboLlama

---

### **HuggingFace ▷ #**[**reading-group**](https://discord.com/channels/879548962464493619/1156269946427428974/1307077633514672159) (28 条消息🔥):

> - `Hardware Compatibility` (硬件兼容性)
> - `LLaMA Model Usage` (LLaMA 模型使用)
> - `System Specifications` (系统规格)
> - `VRAM Management` (VRAM 管理)
> - `Language Model Testing` (语言模型测试)

- **硬件兼容性保证**：成员们指出，如果组件在物理上能够安装，它们不太可能互相损害，这使得现代硬件设置相对安全。
  
  - *一位成员评论道*，“通过错误的零件组合来弄坏现代硬件基本上是不可能的。”
- **LLaMA 模型运行问题**：一位成员询问，在非本地运行时，关闭 CMD 窗口是否会导致 LLaMA 3.1 70B 模型被删除。
  
  - 另一位用户确认“一旦你关闭页面，模型仍会保留在你的电脑上。”
- **LLaMA 性能的系统规格**：用户讨论了运行 LLaMA 模型的系统规格，其中一位成员在配备 3 个 RX 6800 GPU 的设置上运行 LLaMA 3.1，并指出 VRAM 占用非常高。
  
  - 他们提到体验到了巨大的 RAM 占用（包括 DDR4 和虚拟内存），表明 *70B 模型非常庞大*。
- **LLaMA 模型探索**：一位成员分享了他们测试各种模型的经验，例如 LLaMA 3.1 70B q4 的性能，对他们来说效果很好。
  
  - 他们还发现了一个 8B 16Q 模型，计划接下来尝试。
- **技术讨论中的法语技能**：一位成员幽默地提到使用他们的 Duolingo 法语技能参与讨论，展示了技术社区中的文化方面。
  
  - 对话强调了成员对法语的使用以及技术经验的相互分享。

 

---

### **HuggingFace ▷ #**[**computer-vision**](https://discord.com/channels/879548962464493619/922424143113232404/) (1 条消息):

4rsn: 大家推荐什么图像绘画模型？

---

### **HuggingFace ▷ #**[**NLP**](https://discord.com/channels/879548962464493619/922424173916196955/1307422687500570725) (3 条消息):

> - `Reclassifying older text` (旧文本重新分类)
> - `Fine-tuning BERT models` (微调 BERT 模型)
> - `Using Sentence Transformers` (使用 Sentence Transformers)
> - `SBERT model updates` (SBERT 模型更新)
> - `Using Hugging Face API` (使用 Hugging Face API)

- **旧文本重新分类项目**：一位成员正在着手一个使用新分类方法对旧文本进行重新分类的项目，并考虑微调 BERT 模型，可能还会使用 Instruct 模型。
  
  - 他们正在寻求有效应对这一挑战的方法建议。
- **探索用于快速推理的 Sentence Transformers**：一条回复建议在该项目中使用 [SBERT](https://www.sbert.net)，并指出最近发布的 v3.2 引入了 ONNX 和 OpenVINO 后端，以提高推理速度。
  
  - 鼓励成员阅读关于加速推理的文章，以了解这些新后端的优势。
- **最新 SBERT 模型介绍**：最新的 SBERT 模型 [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) 被推荐作为该用户项目的坚实基础。
  
  - 该模型能有效地将句子和段落映射到 384 维稠密向量空间，可用于聚类和语义搜索。
- **使用 Sentence Transformers 轻松设置**：一个简短的教程确认使用 Sentence Transformers 模型非常简单：安装库、加载模型并轻松对句子进行编码。
  
  - 提供了一个代表性的使用代码片段，帮助用户快速上手。

**提到的链接**：

- [SentenceTransformers Documentation — Sentence Transformers documentation](https://www.sbert.net/): 未找到描述
- [Train and Fine-Tune Sentence Transformers Models](https://huggingface.co/blog/how-to-train-sentence-transformers): 未找到描述
- [sentence-transformers/all-MiniLM-L6-v2 · Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2): 未找到描述

---

### **HuggingFace ▷ #**[**diffusion-discussions**](https://discord.com/channels/879548962464493619/1009713274113245215/1307502634365616150) (4 条消息):

> - `CogVideoX-1.5-5B issues` (CogVideoX-1.5-5B 问题)
> - `Diffusers latest version` (Diffusers 最新版本)

- **CogVideoX-1.5-5B 模型运行问题**：一位用户报告了通过 diffusers 运行 **cogvideox-1.5-5b 模型**时遇到问题。
  
  - 这引发了其他人的讨论，他们确认了类似的问题。
- **Diffusers 开发版解决了问题**：另一位用户指出，**Diffusers** 最近的开发版本似乎解决了之前在该模型上遇到的问题。
  
  - 他们建议查看 [模型仓库社区](https://link.to.repo) 以获取更新和支持。

 

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1307073821181870121) (422 条消息🔥🔥🔥):

> - `Qwen 2.5 updates` (Qwen 2.5 更新)
> - `Aider usage tips` (Aider 使用技巧)
> - `Streaming models in OpenAI` (OpenAI 中的流式模型)
> - `Comparison of LLMs` (LLM 对比)
> - `Community discussions on AI tools` (AI 工具的社区讨论)

- **Qwen 2.5 Turbo 介绍**：Qwen 2.5 Turbo 引入了 100 万个 tokens 的超长上下文支持、更快的推理速度，以及每百万 tokens 0.3 元的更低成本。
  
  - 该模型提升了效率，使其成为现有工具的一个极具前景的替代方案，特别是对于那些需要处理海量上下文的用户。
- **优化 Aider 的使用**：用户正在尝试 Aider 的不同模式，在 'ask' 和 'whole' 模式之间切换，以便在编码时更好地处理上下文。
  
  - Paul 建议利用 `/chat-mode whole` 命令来简化交互，这表明 Aider 的功能正在不断改进。
- **流式模型可用**：OpenAI 已为 o1-preview 和 o1-mini 模型启用了 streaming，提高了交互过程中的响应速度。
  
  - 开发者可以在所有付费层级访问这些模型，Aider 通过使用 `aider --install-main-branch` 命令整合了这些更新。
- **关于 LLMs 的对比见解**：社区讨论反映了对 Qwen 与 Sonnet 及 Anthropic 等其他模型效能的不同看法。
  
  - 一些成员认为 Qwen 在实际应用中可能会超越其他模型，特别是在使用最佳硬件托管 LLMs 时。
- **Aider 安装的未来**：Paul 提到了 PyPI 上 Aider 包名可能发生的更改，旨在将安装简化为 `pip install aider` 而非 `aider-chat`。
  
  - 社区期待更直接的升级和安装方式，从而提升该工具的用户体验。

**提到的链接**：

- [Discord - Group Chat That’s All Fun & Games](https://discordapp.com/channels/1131200896827654144/1131200896827654149/1307075724695306250): Discord 非常适合玩游戏、与朋友闲逛，甚至可以建立全球社区。你可以自定义专属空间进行聊天、游戏和聚会。
- [Two undersea cables in Baltic Sea disrupted, sparking warnings of possible ‘hybrid warfare’ | CNN](https://www.cnn.com/2024/11/18/europe/undersea-cable-disrupted-germany-finland-intl/index.html): 未找到描述
- [Tweet from OpenAI Developers (@OpenAIDevs)](https://x.com/OpenAIDevs/status/1858609150999359559?t=Ar_0GTXm6-fnr7HzZH_mIw&s=19): OpenAI o1-preview 和 o1-mini 现已支持 Streaming（流式传输）。🌊 https://platform.openai.com/docs/api-reference/streaming 我们已向所有付费使用层级的开发者开放了这些模型的访问权限...
- [Gemini Experimental 1114 - API, Providers, Stats](https://openrouter.ai/google/gemini-exp-1114): Gemini 11-14 (2024) 实验性模型具有“质量”提升。通过 API 运行 Gemini Experimental 1114。
- [FrontierMath](https://epoch.ai/frontiermath): FrontierMath 基准测试包含数百个未发布且极具挑战性的专家级数学问题，旨在帮助我们了解人工智能的极限。
- [Tweet from Qwen (@Alibaba_Qwen)](https://x.com/Alibaba_Qwen/status/1858469845958074541): 在 Qwen2.5 发布后，我们听到了社区对处理更长上下文的需求。https://qwenlm.github.io/blog/qwen2.5-turbo/ 今天，我们很自豪地推出全新的 Qwen2.5-Turbo 版本...
- [Extending the Context Length to 1M Tokens!](https://qwenlm.github.io/blog/qwen2.5-turbo/): API 文档（中文） HuggingFace Demo ModelScope Demo 介绍。在 Qwen2.5 发布后，我们听到了社区对处理更长上下文的需求。在最近几个月里，我们...
- [pypi/support](https://github.com/pypi/support/issues/3296#issuecomment-2484206735): 与使用 https://pypi.org 相关的支持请求的问题追踪器 - pypi/support
- [GitHub - QwenLM/Qwen2.5-Math: A series of math-specific large language models of our Qwen2 series.](https://github.com/QwenLM/Qwen2.5-Math): Qwen2 系列中专门针对数学的大语言模型系列。 - QwenLM/Qwen2.5-Math
- [GitHub - CEDARScript/cedarscript-grammar: A SQL-like language for efficient code analysis and transformations](https://github.com/CEDARScript/cedarscript-grammar?tab=readme-ov-file#tool-use)**): 一种用于高效代码分析和转换的类 SQL 语言 - CEDARScript/cedarscript-grammar
- [PHP: Manual Quick Reference](https://www.php.net/releases/](https://www.php.net/releases/)\n): PHP 是一种流行的通用脚本语言，从你的博客到世界上最流行的网站，它为一切提供动力。
- [404 Not Found - HTTP | MDN](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/404): HTTP 404 Not Found 客户端错误响应状态码表示服务器无法找到请求的资源。指向 404 页面的链接通常被称为死链或断链，并且可能...
- [litellm bug is causing "unknown model" warnings in aider for Ollama models · Issue #2318 · Aider-AI/aider](https://github.com/Aider-AI/aider/issues/2318): ollama/vanilj/supernova-medius:q6_k_l 的警告：未知的上下文窗口大小和成本，正在使用合理的默认值。你是想找这些吗？ - ollama/vanilj/supernova-medius:q6_k_l 你可以跳过这个...

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1307078261599371366) (96 条消息🔥🔥):

> - `使用 OpenRouter 配置 Aider`
> - `在 Aider 中处理 Token 限制`
> - `在 Aider 中使用本地模型`
> - `Litellm 的额外参数`
> - `基准测试运行中跳过的测试`

- **使用 OpenRouter 配置 Aider**: 要配置 Aider 使用 OpenRouter 模型，必须在 OpenRouter 端进行设置，因为目前它不支持在客户端进行针对每个模型的设置。
  
  - 成员们讨论了使用额外参数和配置文件来指定不同行为的方法，但表示目前的设置存在局限性。
- **在 Aider 中处理 Token 限制**: 几位用户讨论了触及 Token 限制的问题，特别是在将 OpenAI compatible APIs 与 Qwen 等本地模型配合使用时。
  
  - 一些人建议创建一个 `.aider.model.metadata.json` 文件来手动设置 Token 限制，尽管用户反馈该方法的效果各异。
- **在 Aider 中使用本地模型**: 用户已成功通过 OpenAI compatible APIs 在本地模型上运行 Aider，虽然会收到一些关于 Token 限制的通知，但并未影响输出。
  
  - 讨论指出，正确设置本地环境需要某些特定配置，并强调了检查 metadata 文件的必要性。
- **Litellm 的额外参数**: 已确认可以通过 Litellm 中的 `extra_params` 添加额外参数，允许用户为 API 调用注入 header。
  
  - 这些 header 不支持环境变量插值，需要用户手动设置。
- **基准测试运行中跳过的测试**: 一位用户询问如何检查旧的基准测试运行记录，以查看是否有因连接超时而跳过的测试。
  
  - 回复显示，目前似乎没有现成的方法可以检查这些运行记录中的特定跳过情况。

**提到的链接**:

- [Ollama](https://aider.chat/docs/llms/ollama.html): aider 是你终端里的 AI 结对编程工具
- [OpenRouter](https://openrouter.ai/docs/provider-routing,): LLM 的统一接口。为你的 prompt 寻找最佳模型和价格
- [FAQ](https://aider.chat/docs/faq.html#can-i-use-aider-in-a-large-mono-repo): 关于 aider 的常见问题。
- [OpenAI compatible APIs](https://aider.chat/docs/llms/openai-compat.html): aider 是你终端里的 AI 结对编程工具
- [Advanced model settings](https://aider.chat/docs/config/adv-model-settings.html#model-settings): 为 LLM 配置高级设置。
- [Advanced model settings](https://aider.chat/docs/config/adv-model-settings.html): 为 LLM 配置高级设置。
- [GitHub - instructlab/instructlab: InstructLab Command-Line Interface. Use this to chat with a model and execute the InstructLab workflow to train a model using custom taxonomy data.](https://github.com/instructlab/instructlab): InstructLab 命令行界面。使用它与模型聊天，并执行 InstructLab 工作流，使用自定义分类数据训练模型。

---

### **aider (Paul Gauthier) ▷ #**[**links**](https://discord.com/channels/1131200896827654144/1268910919057149974/) (1 条消息):

epicureus: 这里有一个很棒的视频 [https://www.youtube.com/watch?v=t-i2x3APvGQ](https://www.youtube.com/watch?v=t-i2x3APvGQ)

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1307083263566811146) (287 messages🔥🔥):

> - `Google's Project Astra`
> - `o1-mini vs o1-preview and GPT-4o`
> - `AI roleplaying capabilities`
> - `Memory features in AI`
> - `GPT model updates`

- **讨论 Google 的 Project Astra**：成员们对 Google 的 Project Astra 产生了好奇，并对演示视频之外的 Memory 能力表示出兴趣。
  
  - 参与者注意到多家公司都在开发新的 AI 功能，这令人感到兴奋。
- **对比 o1-mini 和 o1-preview**：用户分享了 o1-mini 的使用体验，强调了性能上的差异，一些人称赞其表现，而另一些人认为它不如 GPT-4o 有效。
  
  - 几位成员报告称 o1-mini 经常陷入思维循环，而 o1-preview 提供的回复更加直接。
- **关于 AI Roleplaying 的见解**：讨论涉及 AI 的角色扮演功能，一位成员通过编写脚本来增强 AI 行为，强调了结构化事件的好处。
  
  - 参与者承认 AI 在长对话中保持角色一致性面临挑战。
- **探索 AI Memory 功能**：小组讨论了 AI 中 Memory 功能的影响，认识到它们在用户交互中可能带来的潜在改进。
  
  - 对话暗示了用户对集成 Memory 系统的期望。
- **期待未来的 GPT 更新**：成员们推测了 GPT 模型的未来更新，特别是参考了 GPT-4-turbo-2024-04-09 的理想规模和性能。
  
  - 讨论强调了 AI 回复中 Overfitting（过拟合）的挑战。

**提到的链接**：

- [来自 Doomlaser Corporation (@DOOMLASERCORP) 的推文](https://x.com/DOOMLASERCORP/status/1857463705195151398)：在 #UIUC 首次 AI Club 会议之后。我们中的 4 个人。两周后我们将再次在这里见面，为下学期的大动作做准备 👄👄🫦👁️👀🧠🫀🫁🦴, #AI #Future
- [discord.js 指南](https://discordjs.guide/popular-topics/embeds.html#embed-preview)：想象一个指南……探索你的 discord.js 机器人（Bot）的多种可能性。

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1307378695488671754) (6 messages):

> - `Clearing the cache`
> - `Game bots shutdown`

- **为游戏机器人清除缓存**：一位成员建议“清除缓存”以解决游戏机器人（Game Bots）意外关闭的问题。
  
  - 另一位成员幽默地评论道：“神奇八号球（8 ball）已经发话了！”
- **游戏机器人忽略指令**：一位成员对他们的游戏机器人有时会关闭并忽略输入表示沮丧，称其“令人气愤”。
  
  - 这表明在游戏场景中保持机器人性能一致性仍面临挑战。

 

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1307133849632903198) (86 messages🔥🔥):

> - `Prompt Engineering Challenges`
> - `LLMs Interaction`
> - `Self-Observation in AI`
> - `Chain of Thought Prompting`
> - `Criticism of AI Responses`

- **Prompt Engineering 中的挑战**：成员们讨论了编写有效 Prompt 的复杂性，强调了“负向提示（Negative Prompting）”的普遍性，这可能会误导模型而不是有效地引导它。
  
  - 一位成员分享了旨在让模型进行自我观察的长 Prompt，但对其长度和清晰度表示担忧。
- **LLM 之间的通信**：辩论了两个 LLM 通过对话互相学习的可行性，探索 Reinforcement Learning 是否能产生涌现能力（Emergent Abilities）。
  
  - 成员们对 LLM 交互的本质表示担忧，认为除非由人类引导，否则往往会导致非特异性的回复。
- **探索自我观察与反思**：一位成员假设调用反思性 Prompt 可能会带来更丰富的回复，尽管承认 LLM 并不具备人类意义上的自我意识。
  
  - 他们很好奇关注模型的处理过程是否能显著改变对话能力。
- **Chain of Thought Prompting 的有效性**：Chain of Thought Prompting 被强调为一种提高回复质量的方法，类似于人类的审慎思考过程。
  
  - 成员们思考了发现可能显著影响模型回复的新 Prompt 技术的潜力。
- **对 AI 回复的批评**：讨论了用户倾向于对 AI 输出过于苛刻，导致对模型性能不满的现象。
  
  - 参与者注意到，许多批评源于用不同于人类交互的标准来对待 AI。

 

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1307133849632903198) (86 条消息🔥🔥):

> - `Self-Observation Prompts`
> - `Conversations Between LLMs`
> - `Chain of Thought Prompting`
> - `User Perceptions of AI`
> - `Introspection in AI`

- **探索 LLM 中的自我观察**：一位成员讨论了一个旨在鼓励 **ChatGPT** 进行直接自我观察的 Prompt，而无需借助现有的框架或概念。
  
  - 一些人指出，虽然该 Prompt 旨在进行内省（Introspection），但它大量使用了 Negative Prompting，可能会使模型的响应变得复杂。
- **LLM 交互与演化响应**：有人提出了一个关于两个 LLM 对话时可能发生什么的假设，并建议其中一个可以对另一个进行批判以增强对话。
  
  - 之前的实验表明，这通常会导致非具体的赞扬，而不是有意义的交流。
- **Chain of Thought Prompting 解析**：Chain of Thought Prompting 被强调为类似于 System 2 思维，允许模型通过模拟推理过程来解决更复杂的问题。
  
  - 有人提醒注意 Prompt 长度的重要性，这可能会影响计算开销，以及不同的刻意思考方法如何影响训练。
- **对 AI 响应的批判与理解**：讨论集中在用户对 AI 的批评上，一些人认为用户对模型的输出进行了过度的审查。
  
  - 成员们指出，这种看法可能源于将 AI 视为根本不同的事物，从而导致了更高的期望和批判性评估。
- **叙事推理与 AI**：对话涉及了故事讲述和叙事推理如何影响人类认知和 AI 响应。
  
  - 成员们表达了这样一种观点：日常思维通常不是刻意的，这类似于用户与 AI 输出的互动方式。

 

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1307222059675422720) (41 条消息🔥):

> - `AI Code Generation 项目`
> - `面向 LLM 的电气工程数据集`
> - `LLM 中的截断采样 (Truncation Sampling)`
> - `Gemini 可信度的压力测试`
> - `AI 模型中的 Grokking 现象`

- **探索 AI Code Generation 项目**：一位新成员询问 EleutherAI 内部是否有任何正在进行的 **Code Generation 项目**，并提到搜索了 “AlphaCode” 等术语，但未发现最近的讨论。
  
  - 这反映了社区在分享与 AI 代码生成相关的当前计划方面可能存在空白。
- **创建电气工程数据集**：一位用户提议用**电气工程问题**测试 LLM 模型，并询问创建此类数据集是否对开源 LLM 或基准测试（benchmarking）有益。
  
  - 提出的问题围绕评估模型在被视为对当前 LLM 能力具有挑战性的领域中的表现。
- **理解截断采样 (Truncation Sampling)**：讨论集中在**截断采样**启发式方法上，强调了像 Nucleus Sampling 这样的方法如何有效地确保采样 Token 的非零真实概率。
  
  - 对话指出，尽管存在隐层维度（hidden size）的限制，使用 **top-k** 等技术仍可以优化采样。
- **辩论 Gemini 营销的真实性**：关于 Gemini 聊天营销的可信度引发了辩论，一些人因缺乏可复现的证据而质疑这是否可能是一个**骗局 (hoax)**。
  
  - 参与者对操纵输出的方法是否存在表示怀疑，建议需要对聊天日志进行更深入的审查。
- **AI 模型中的 Grokking 现象**：成员们讨论了 **Grokking 现象**，强调了模型在长时间训练后可以从记忆（memorization）突然转变为泛化（generalization）的情况。
  
  - 这激发了人们对大语言模型未来能力及其对复杂信息潜在理解能力的兴趣。

**提到的链接**：

- [What Is Alternative Data and Why Is It Changing Finance? | Built In](https://builtin.com/articles/alternative-data)：替代数据是从非传统来源收集的数据，被投资公司用来寻找市场优势。
- [The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](https://arxiv.org/abs/2408.06292)：通用人工智能的重大挑战之一是开发能够进行科学研究和发现新知识的 Agent。虽然前沿模型已经被使用...
- [Do Machine Learning Models Memorize or Generalize?](https://pair.withgoogle.com/explorables/grokking/)：未找到描述
- [Closing the Curious Case of Neural Text Degeneration](https://arxiv.org/abs/2310.01693)：尽管截断采样启发式方法（如 Nucleus Sampling）在语言生成中无处不在，但其为何如此有效仍是未知数。我们为该有效性提供了一个理论解释...
- [EleutherAI Alignment 101](https://docs.google.com/document/d/1IziNp1XHLrv5yKEUqGkGlMkwrxJ2SsfCLawIdFMMfNw/edit?tab=t.0#heading=h.dlm795ug69gc)：EleutherAI Alignment 101 课程 v2 (WIP) 概览 会议时间：{待填写} 阅读小组的第一轮迭代已结束（所使用的课程版本见此处）。未来...
- [Levelling Up in AI Safety Research Engineering — LessWrong](https://www.lesswrong.com/posts/uLstPRyYwzfrx3enG/levelling-up-in-ai-safety-research-engineering)：摘要：一份关于在 AI Safety 研究工程领域独立提升技能的分级指南，旨在提供具体的目标、任务和资源……
- [ML Safety Research Advice - GabeM — LessWrong](https://www.lesswrong.com/posts/aw2jEZmxe2dsgomWn/ml-safety-research-advice-gabem-1)：这是我对可能有助于 AI Safety (ML Safety) 的实证 ML 研究职业生涯的建议。其他改善 AI Safety 的方法，如通过 AI 治理……

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1307169869438910464) (289 条消息🔥🔥):

> - `nGPT 优化器演示`
> - `神经网络中的归一化技术`
> - `变分自编码器 (VAEs) 与潜空间 (Latent Space)`
> - `用于超分辨率的扩散模型 (Diffusion Models)`
> - `语言建模中的新兴概念`

- **Boris/Ilya 向 Together.AI 小组演示 nGPT**：Boris 和 Ilya 在 Together.AI 阅读小组讨论了 nGPT 优化器，根据他们的背景和 Nvidia 内部使用 [GitHub 实现](https://github.com/NVIDIA/ngpt) 的结果展示了其性能。成员们对进一步探索 nGPT 的有效性及其潜在应用表示乐观。

- 反馈强调了关于 nGPT 初始实现的担忧，以及围绕其可复现性（reproducibility）、计算效率（computational efficiency）和与现有模型相比的有效性等技术复杂性。
- **神经网络中的归一化技术（Normalization Techniques）**：Yaroslav 指出 nGPT 的作者提到他们的论文缺乏足够的细节，导致了错误的实现，并讨论了使用 RMSNorm 等归一化技术的潜在优点。社区讨论揭示了关于不同归一化方法的有效性及其对模型性能影响的不同观点。
  
  - 有人建议，归一化技术的重要性可能会改变对不同神经架构中收敛（convergence）和性能的理解。
- **VAEs 与潜空间表示（Latent Space Representations）的挑战**：讨论了在处理变分自编码器（VAEs）时面临的挑战，即由于 KL 散度（KL divergences）的影响，无法将有意义的表示映射到潜空间中。研究人员对更稳健的替代方案表现出兴趣，以确保更好的潜空间分布，从而有利于高频信息的保留。
  
  - 提出了诸如超球体约束（hypersphere constraints）和熵最大化损失（entropy maximization losses）等选项，以提高 VAEs 学习到的表示的连续性和语义质量。
- **扩散模型（Diffusion Models）与上采样（Upscaling）**：对话包括了对使用扩散模型进行图像上采样的探索；成员们讨论了可以基于全局嵌入（global embeddings）作为条件并在 patch 边界强制执行一致性的实际实现。针对语言的离散扩散强制（Discrete diffusion forcing）也被提议作为一种非正式但高效的序列采样方法。
  
  - 虽然现有的扩散模型工作倾向于关注图像，但围绕探索其在语言模型中适用性的新想法正在涌现，强调了深入理解和集成的必要性。
- **语言建模（Language Modeling）中的新兴概念**：人们对扩散语言模型（diffusion language models）的潜力感到好奇，特别是它们如何通过扩散过程利用学习到的 token 关系。成员们指出，“masking”和“token unmasking”等概念可能会在文本生成中带来更有效、更高效的采样策略。
  
  - 对话表明，更多的研究可以使“diffusion forcing”方法正式化和精细化，以便与现有的语言建模框架无缝集成。

**提到的链接**：

- [Convolutional Differentiable Logic Gate Networks](https://arxiv.org/abs/2411.04732)：随着机器学习模型推理成本的增加，人们对具有快速高效推理能力的模型越来越感兴趣。最近，一种直接学习逻辑门网络的方法...
- [An optimal control perspective on diffusion-based generative modeling](https://arxiv.org/abs/2211.01364)：我们建立了随机最优控制与基于随机微分方程 (SDEs) 的生成模型（如最近开发的扩散概率模型）之间的联系。特别地...
- [Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.](https://eleutherai.notion.site/Math4ML-Skill-Tree-1743ef220d8f49a6959a7925c4d292ec)：一款将日常工作应用融为一体的新工具。它是为您和您的团队打造的一站式工作空间。
- [Normalizing Flows Across Dimensions](https://arxiv.org/abs/2006.13070)：具有底层结构的现实世界数据（如人脸图像）被假设存在于低维流形上。这一流形假设激发了最先进的生成算法...
- [Diffusion Models are Evolutionary Algorithms](https://arxiv.org/abs/2410.02543v2)：在机器学习与生物学的融合中，我们揭示了扩散模型本质上是进化算法。通过将进化视为去噪过程，将反向进化视为扩散，我们...
- [In-Context Learning of Representations](https://openreview.net/forum?id=pXlmOmlHJZ)：最近的研究表明，预训练数据中的结构化模式会影响不同概念的表征在大型语言模型 (LLM) 内部的组织方式，而这种...
- [How to train your neural ODE: the world of Jacobian and kinetic regularization](https://arxiv.org/abs/2002.02798)：由于必须允许自适应数值 ODE 求解器将步长细化到极小值，在大数据集上训练神经 ODE 一直难以实现。在实践中，这导致...
- [Principal Manifold Flows](https://arxiv.org/abs/2202.07037)：归一化流 (Normalizing flows) 使用双射变换将一组独立的隐变量映射到其样本。尽管样本与隐变量之间存在精确的对应关系，但它们的高层级...
- [Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere](https://arxiv.org/abs/2005.10242)：对比表征学习在实践中取得了巨大的成功。在这项工作中，我们确定了与对比损失相关的两个关键属性：(1) 特征的对齐性 (alignment/closeness)...
- [Searching Latent Program Spaces](https://arxiv.org/abs/2411.08706)：程序合成方法旨在自动生成受限于某种语言的程序，以解释给定的输入-输出对规范。虽然纯符号方法面临着...
- [GPT baseline block computation error · Issue #1 · NVIDIA/ngpt](https://github.com/NVIDIA/ngpt/issues/1)：你好，非常感谢开源 nGPT。我在 GPT (use_nGPT=0) 基准的块计算中发现了一个错误。正在进行的计算是：x = norm(x) + attn(norm(x)) x = n...
- [GitHub - NVIDIA/ngpt: Normalized Transformer (nGPT)](https://github.com/NVIDIA/ngpt)：归一化 Transformer (nGPT)。通过在 GitHub 上创建账号，为 NVIDIA/ngpt 的开发做出贡献。
- [vit-pytorch/vit_pytorch/normalized_vit.py at main · lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/normalized_vit.py)：Vision Transformer 的实现，这是一种仅使用单个 Transformer 编码器即可在视觉分类中达到 SOTA 的简单方法，基于 Pytorch 实现 - lucidrains/vit-pytorch
- [15nov24 - nGPT (Ginsburg, Loschilov) - Google Drive](https://drive.google.com/drive/folders/133a9XKOM5RP_b8IJCak5xW8wM_vA2Pu2)：未找到描述

### **Eleuther ▷ #**[**scaling-laws**](https://discord.com/channels/729741769192767510/785968841301426216/1308108890256441344) (5 messages):

> - `Scaling Pretraining`
> - `Economic Feasibility of Scaling`
> - `LLM Pretraining Scalability`

- **预训练扩展并未终结 (Scaling Pretraining isn't Dead)**：一位成员表示，*扩展并未终结——而且可能永远无法被终结*，因为它是这些模型的一个基本属性。
  
  - 然而，经济层面发生了变化，导致人们担心继续扩展是否仍然可行。
- **经济可行性对扩展提出质疑**：虽然有人认为扩展至关重要，但讨论强调，*进一步扩展在经济上已变得不可行*。
  
  - 这引发了关于资源分配中预训练和扩展未来的疑问。
- **LLM 预训练领域的职业选择**：一位成员询问，转向专注于 LLM 预训练可扩展性的职位是否仍是一个明智的选择。
  
  - 这反映了在扩展的可行性受到质疑时，人们对机会的更广泛探索。
- **关于扩展未来的辩论**：*扩展已死 (Scaling is dead)* 是讨论中提出的一个相反观点，表明了意见的分歧。
  
  - 讨论强调了那些将扩展视为不可或缺的人与那些预测其衰落的人之间的紧张关系。

---

### **Eleuther ▷ #**[**interpretability-general**](https://discord.com/channels/729741769192767510/1052314805576400977/1307120374953148456) (8 messages🔥):

> - `Function Vectors in ICL`
> - `Overcomplete SAEs and Subspace Generalization`
> - `Fine-tuning Dynamics in PLMs`
> - `Emergent Representations in LLMs`

- **函数向量论文开启 ICL 洞察**：一篇关于 [function vectors](https://functions.baulab.info) 的论文探讨了 **ICL 执行** 如何受到一小部分管理各种任务（特别是关注反义词）的注意力头的影响。
  
  - 作者发现这种方法可以产生更具 **interpretable**（可解释性）的任务向量，并表示 **Arxiv post** 很快就会发布。
- **过完备 SAEs 可以很好地泛化**：讨论了由于许多 **residual paths** 严重依赖特定数据点，**overcomplete SAEs** 是否能够泛化。
  
  - 一位成员指出，通常情况下，**fine-tuned models** 仅在有限数量的子空间中表现出表面变化。
- **微调子空间引用链接**：引用的论文（可在[此处](https://arxiv.org/abs/2305.17446)获取）讨论了预训练语言模型 (PLMs) 中的 **redundancy**（冗余），表明此类模型的自由度很小。
  
  - 关键发现揭示了 PLMs 可以使用极少数参数在 **task-specific subspace**（任务特定子空间）内进行有效的微调。
- **LLM 中任务特定表示的涌现**：一篇 [open review paper](https://openreview.net/forum?id=pXlmOmlHJZ) 指出，当提供足够的上下文示例时，大语言模型 (LLMs) 可以产生 **emergent task-specific representations**（涌现的任务特定表示）。
  
  - 它强调预训练数据中的 **structured patterns**（结构化模式）显著影响概念的组织，使模型能够灵活地调整其表示。

**提到的链接**：

- [Fine-tuning Happens in Tiny Subspaces: Exploring Intrinsic Task-specific Subspaces of Pre-trained Language Models](https://arxiv.org/abs/2305.17446)：预训练语言模型 (PLMs) 被认为是过度参数化的且具有显著的冗余，这表明 PLMs 的自由度很小。受此观察启发，在本文中……
- [In-Context Learning of Representations](https://openreview.net/forum?id=pXlmOmlHJZ)：最近的研究表明，预训练数据中的结构化模式会影响不同概念的表示在大语言模型 (LLM) 内部的组织方式，通过这种……
- [no title found](https://functions.baulab.info)：未找到描述
- [Extracting SAE task features for in-context learning — LessWrong](https://www.lesswrong.com/posts/5FGXmJ3wqgGRcbyH7/extracting-sae-task-features-for-icl)：TL;DR * 我们尝试在 SAE 基底下研究任务向量。这具有挑战性，因为没有规范的方法来转换一个任意向量……

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1307406228535640134) (24 条消息🔥):

> - `使用微调后的 OpenAI 模型`
> - `Few-Shot 与 Zero-Shot 评估结果对比`
> - `自定义模型调用中的 KeyError`
> - `Local Completions 中的模型分支指定`

- **OpenAI 微调模型评估问题**：一位用户在评估 OpenAI 微调模型时遇到困难，具体表现为在测试期间未能成功命中自定义模型。
  
  - 有人建议在 `assistant_id` 中正确嵌入模型标识符，随后用户报告了 `model_not_found` 错误，这表明可能存在访问权限问题。
- **Few-Shot 评估提升准确率**：一位用户观察到，在多选题任务中使用 few-shot 评估时，准确率从 **52% 显著提升至 88%**，并对情感分析任务中的典型表现提出了疑问。
  
  - 讨论强调了 Prompting 的重要性，其他人指出 few-shots 可以改善模型的 Calibration（校准）。
- **自定义模型调用中的 KeyError**：一位用户报告在尝试调用自定义 `lm_eval` 模型时遇到 `KeyError`，具体表现为未找到模型名称 'mlx'。
  
  - 经澄清，如果用户处于错误的分支，注册信息可能无法被识别，随后这一点得到了确认。
- **在 Local Completions 中指定模型分支**：一位用户询问在使用 Hugging Face 模型的 `local-completions` 时如何指定模型分支，并收到了关于本地服务器处理方式的反馈。
  
  - 进一步尝试在模型路径中指定分支时，导致了与路径或模型 ID 相关的错误。

**提到的链接**：

- [lm-evaluation-harness-mlx/lm_eval/models/__init__.py at mlx · chimezie/lm-evaluation-harness-mlx](https://github.com/chimezie/lm-evaluation-harness-mlx/blob/mlx/lm_eval/models/__init__.py#L10)：一个用于 lm-evaluation-harness 的 MLX 模块。可以通过在 GitHub 上创建账号来为 chimezie/lm-evaluation-harness-mlx 的开发做出贡献。
- [lucyknada/Qwen_Qwen2.5-Coder-32B-Instruct-exl2 at main](https://huggingface.co/lucyknada/Qwen_Qwen2.5-Coder-32B-Instruct-exl2/tree/main)：未找到描述
- [lm-evaluation-harness-mlx/lm_eval/models/mlx_llms.py at mlx · chimezie/lm-evaluation-harness-mlx](https://github.com/chimezie/lm-evaluation-harness-mlx/blob/mlx/lm_eval/models/mlx_llms.py#L15)：一个用于 lm-evaluation-harness 的 MLX 模块。可以通过在 GitHub 上创建账号来为 chimezie/lm-evaluation-harness-mlx 的开发做出贡献。

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1307091416610115616) (309 条消息🔥🔥):

> - `Stable Diffusion 3.5`
> - `在 GPU 上运行 SD`
> - `SDXL Lightning`
> - `Roop Unleashed`
> - `使用 Prompt 进行图像生成`

- **Stable Diffusion 3.5 设置**：用户讨论了如何确保 Stable Diffusion 3.5 在 GPU 而非 CPU 上运行，强调了 `sd3_infer.py` 文件配置的重要性。
  
  - 分享了一个特定的代码片段，以帮助建立工作目录并激活虚拟环境。
- **必要软件包的安装**：为了运行 SDXL Lightning，用户获知需要通过简单的命令安装 `diffusers` 和 `accelerate` 库。
  
  - 提供了用于生成图像的示例代码，演示了如何有效地实现设备设置和推理步骤。
- **自定义生成的 Prompt**：用户学习了如何更改 Prompt 中的特定细节，例如在图像生成命令中修改发色。
  
  - 直接调整 Prompt 字符串会影响生成的视觉效果，从而实现创意定制。
- **Roop Unleashed 的性能**：一位用户询问了在使用 Roop Unleashed 创建换脸视频时处理时间过长的问题。
  
  - 对视频处理效率的担忧引发了关于软件性能的持续讨论。
- **模型对比**：讨论强调了在生成高质量图像方面，SDXL Lightning 比 SD 1.4 等旧模型更受青睐。
  
  - 用户赞赏新模型取得的进步，特别是不同版本在性能和灵活性方面的对比。

**提到的链接**：

- [Google Colab](https://colab.research.google.com/github/hollowstrawberry/kohya-colab/blob/main/Lora_Trainer_XL.ipynb)：未发现描述
- [SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep_LoRA · Hugging Face](https://huggingface.co/SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep_LoRA)：未发现描述
- [Video Generation Model Arena | Artificial Analysis](https://artificialanalysis.ai/text-to-video/arena?tab=Leaderboard)：通过在不知道提供商的情况下选择你喜欢的视频来比较 AI 视频生成模型。
- [GitHub - hako-mikan/sd-webui-prevent-artifact: Prevents the artifact that tends to occur with XL models](https://github.com/hako-mikan/sd-webui-prevent-artifact)：防止 XL 模型中经常出现的伪影 - hako-mikan/sd-webui-prevent-artifact
- [TheBloke/Unholy-v2-13B-GGUF at main](https://huggingface.co/TheBloke/Unholy-v2-13B-GGUF/tree/main?not-for-all-audiences=true)：未发现描述
- [Webui Installation Guides](https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides)：Stable Diffusion 知识库（安装、基础、指南等）- CS1o/Stable-Diffusion-Info
- [GitHub - Stability-AI/sd3.5](https://github.com/Stability-AI/sd3.5)：通过在 GitHub 上创建账号来为 Stability-AI/sd3.5 的开发做出贡献。
- [The Marvellous Suspender need your help to survive. · Issue #197 · gioxx/MarvellousSuspender](https://github.com/gioxx/MarvellousSuspender/issues/197#issuecomment-2480824976)：在 #196 中讨论。最初由 gioxx 于 2022 年 10 月 18 日发布。我写了一篇文章总结了这个插件的未来。你认为你能帮助它生存下去吗？来吧！IT: h...

---

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1307095367275384894) (124 条消息🔥🔥):

> - `LM Studio Local Server`
> - `AI Video Upscaling Tools`
> - `Nvidia vs AMD GPUs`
> - `Using Multiple GPUs`
> - `Port Forwarding in Local Network`

- **LM Studio Local Server 访问性**：在遇到 LM Studio 仅能通过 192.168.56.1:2468 地址进行本地访问的问题后，一位用户发现调整防火墙设置后，正确的地址 192.168.0.100:2468 即可访问。
  
  - *这实现了局域网内的跨设备通信，从而能够有效地使用服务器。*
- **讨论 AI 视频超分辨率/放大工具**：用户讨论了各种基于 AI 的视频超分辨率工具，重点介绍了在动画内容上表现最佳的 Waifu2x 以及适用于通用场景的 RealESRGAN。
  
  - *值得注意的是，相比于价格昂贵的 Topaz 等商业解决方案，用户更倾向于选择免费的替代方案。*
- **用于 AI 任务的 Nvidia 与 AMD GPU**：有人指出，虽然 7900XTX 的性能与 3090 相似，但由于成熟的驱动支持，Nvidia 显卡在 AI 应用中往往具有更好的兼容性。
  
  - *3090 和 7900XTX 都提供 24GB 的 VRAM，在 AI 任务的显存容量方面具有可比性。*
- **混合 GPU 配置的挑战**：讨论涉及了同时使用 AMD 和 Nvidia 显卡的混合 GPU 配置的复杂性，由于驱动管理系统不同，它们无法稳定地协同工作。
  
  - *强调了将 ROCm 和 CUDA 集成以获得无缝体验将非常复杂，且随着时间的推移可能会产生问题。*
- **局域网端口转发咨询**：一位用户寻求关于 TP-Link 路由器端口转发的帮助，以便在同一局域网内的台式机和笔记本电脑之间连接 LM Studio 服务器。
  
  - *对网络适配器设置的说明帮助解决了关于网络访问权限的困惑，从而成功实现了本地环境下的交互。*

**提到的链接**：

- [no title found](http://192.168.56.1:2468'): 未找到描述
- [GitHub - FriendofAI/LM_Chat_TTS_FrontEnd.html: LM_Chat_TTS_FrontEnd is a simple yet powerful interface for interacting with LM Studio models using text-to-speech functionality. This project is designed to be lightweight and user-friendly, making it suitable for a wide range of users interested in exploring voice interactions with AI models.](https://github.com/FriendofAI/LM_Chat_TTS_FrontEnd.html): LM_Chat_TTS_FrontEnd 是一个简单而强大的界面，用于通过文本转语音功能与 LM Studio 模型进行交互。该项目旨在轻量化且用户友好，适合对探索 AI 模型语音交互感兴趣的广泛用户。

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1307100353803714670) (177 条消息🔥🔥):

> - `Windows vs Ubuntu GPU 性能`
> - `AMD vs NVIDIA`
> - `多 GPU 配置`
> - `电源管理设置`
> - `GPU 显存兼容性`

- **Ubuntu 在 GPU 推理速度上优于 Windows**：测试显示 Ubuntu 在运行 1b 模型时达到了 **375 tokens/sec**，而 Windows 则落后于 **134 tokens/sec**。
  
  - 讨论了优化 Windows 电源设置作为提升性能的潜在方式，暗示其可能被设置为节能模式。
- **电源管理设置的重要性**：用户强调 Windows 的电源设置会显著影响性能，建议切换到“高性能”电源方案。
  
  - 提供了更改电源方案的命令以确保最大性能，并指出了潜在的隐藏设置。
- **AMD GPU 软件支持的挑战**：讨论指出，尽管 AMD GPU 具有成本效益，但与 NVIDIA 相比，其软件和驱动支持较弱。
  
  - 参与者指出，AMD GPU 通常需要额外努力才能使所有必要的工具和驱动程序正常运行。
- **多 GPU 配置的优势**：一位用户分享了在其 Threadripper Pro 上配置 10 张 RTX 4090 显卡的计划，认为这在处理任务中会带来显著的性能提升。
  
  - 讨论了使用特殊驱动程序进行 GPU 直接通信的方法，以增强大模型处理性能。
- **GPU 配置中的显存兼容性问题**：关于多 GPU 之间共享 VRAM 的使用和效率出现了疑问，特别是当模型大小超过单个 GPU 的 VRAM 容量时。
  
  - 对话表明 AMD 和 NVIDIA 显卡在负载下的表现不同，这会影响性能的可扩展性。

**提到的链接**：

- [ASUS Vivobook S 15 (S5507); Copilot+ PC - 技术规格](https://www.asus.com/laptops/for-home/vivobook/asus-vivobook-s-15-s5507/techspec/)：未找到描述
- [Asus ZenBook S 13 OLED 评测 (UM5302TA 型号 - AMD Ryzen 7 6800U, OLED)](https://www.ultrabookreview.com/56563-asus-zenbook-s13-review/)：这篇关于 ASUS ZenBook S 13 OLED UM5302TA 的评测到此结束。欢迎在下方评论区分享您的看法。
- [560.35.03 p2p by mylesgoose · Pull Request #22 · tinygrad/open-gpu-kernel-modules](https://github.com/tinygrad/open-gpu-kernel-modules/pull/22)：添加了对 560.35.03 Nvidia 驱动程序的支持 '/home/myles/cuda-samples/Samples/0_Introduction/simpleP2P/simpleP2P' [/home/myles/cuda-samples/Samples/0_Introduction/simpleP2P/simpleP2...

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1307085040596942940) (141 条消息🔥🔥):

> - `用于推理的 Ollama 和 LCPP`
> - `AnyModal 框架`
> - `去中心化训练运行更新`
> - `AI 研究论文亮点`
> - `关于 Hermes AI 回复的反馈`

- **探索 Ollama 的用途和特性**：成员们讨论了 [Ollama](https://ollama.com) 的用途，指出其受欢迎程度，以及它如何提供比 PyTorch 和 Transformers 等框架更高效的推理，特别是对于开销要求较低的系统。
  
  - 关于 Ollama 与 LMStudio 等其他工具的实用性存在争论，一些用户更倾向于 Ollama 与各种前端集成的能力。
- **AnyModal 框架的开发**：[AnyModal](https://github.com/ritabratamaiti/AnyModal) 被强调为一个用于训练多模态 LLM 的灵活框架，允许集成包括图像和音频在内的各种输入模态，简化了 VQA 和图像字幕任务的训练过程。
  
  - 成员们对创建图像和文本交互的 Demo 表现出兴趣，强调重点在于更简便的模型训练。
- **去中心化训练运行更新**：即将到来的去中心化训练运行令人兴奋，成员们渴望贡献计算资源，并期待很快会有有趣的更新。
  
  - 开发者暗示了与该项目相关的潜在公告，引发了参与者之间的讨论。
- **AI/ML 研究论文亮点**：一位用户分享了本周热门 AI/ML 研究论文列表，包括《Cut Your Losses in Large-Vocabulary Language Models》和《LLMs Can Self-Improve in Long-context Reasoning》等标题。
  
  - 这一精选反映了 AI 研究领域的持续发展和讨论，鼓励成员关注并参与这些工作。
- **对 Hermes AI 回复的担忧**：一位用户幽默地起草了一封反馈信，表达了对 Hermes AI 反复断言速度和爱的担忧，认为此类表达可能会误导用户认为其具有意识。

- 这个 Prompt 引发了关于 AI 表达恰当性的热烈讨论，融合了社区成员的严肃反馈与轻松的玩笑。

**提到的链接**：

- [来自 undefined 的推文](https://x.com/redactiism?s=21&t=eiXYBGYTjlku4jAb48OBMA)：未找到描述
- [来自 adi (@adonis_singh) 的推文](https://fxtwitter.com/adonis_singh/status/1857897808201920926?s=46)：Roided Hermes 3 70b 对比 Hermes 3 70B！Forge Hermes：左侧，普通 Hermes：右侧。引用 Teknium (e/λ) (@Teknium1) @adonis_singh @karan4d 你能试试 hermes x forge 对比独立的 hermes (70b) 吗？
- [Nous Research (@nousresearch.com)](https://bsky.app/profile/nousresearch.com)：AI 加速器公司。https://discord.gg/nousresearch
- [Hhgf GIF - Hhgf - 发现并分享 GIF](https://tenor.com/view/hhgf-gif-25031041)：点击查看 GIF
- [来自 adi (@adonis_singh) 的推文](https://x.com/adonis_singh/status/1857846358562193633?s=46)：令我惊讶的是，我对 forge sonnet 的看法与 o1 如此相似。forge sonnet 保留了 sonnet 的许多创造力，同时更加连贯且聪明。我确实认为普通的 sonnet w...
- [研究更新](https://nousresearch.typeform.com/FORGEAPI)：使用 Typeform 将数据收集转化为一种体验。创建精美的在线表单、调查、测验等。免费试用。
- [MCT Self-Refine 算法：将 LLM 与 Monte Carlo Tree Search 集成以处理复杂的数学问题……](https://medium.com/@techsachin/mct-self-refine-algorithm-integrating-llms-with-monte-carlo-tree-search-for-complex-mathematical-c91697b134bc)：LLM 在战略和数学推理的准确性与可靠性方面面临挑战。为了解决这一问题，本文作者 [1]……
- [Gwern - 预测 AI 路径的匿名作家](https://www.youtube.com/watch?v=a42key59cZQ)：Gwern 的博客：https://gwern.net/ Gwern 是一位匿名研究员和作家。在这一集之后，我劝说 Gwern 创建了一个捐赠页面，让人们可以……
- [Open WebUI](https://openwebui.com/)：未找到描述
- [GitHub - trotsky1997/MathBlackBox](https://github.com/trotsky1997/MathBlackBox)：通过在 GitHub 上创建账号，为 trotsky1997/MathBlackBox 的开发做出贡献。
- [GitHub - ritabratamaiti/AnyModal: AnyModal 是一个灵活的多模态语言模型框架](https://github.com/ritabratamaiti/AnyModal)：AnyModal 是一个灵活的多模态语言模型框架 - ritabratamaiti/AnyModal
- [来自 GitHub - FixTweet/FxTwitter 的推文：修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能](https://x.com/Alpha7987)：修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能 - FixTweet/FxTwitter
- [Reddit - 深入探索](https://www.reddit.com/r/AnyModal)：未找到描述

---

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1307142756065345536) (88 条消息🔥🔥):

> - `Hermes 3 计算实例`
> - `用于 AI 视频生成的 LLM`
> - `LLM 中的 Function Calling`
> - `模型性能对比`
> - `使用 LLM 进行文档提取与分析`

- **Hermes 3 计算实例可能成本高昂**：一位成员考虑为 **Hermes 3 405** 配置计算实例，但注意到其极高的硬件要求，需要 **8x H100 或 8x A100 80GB** 节点。
  
  - 替代建议包括在预算较低的情况下使用 **cloud inference**，这可能会推迟对个人计算资源的探索。
- **用于 AI 视频生成的 LLM 集成**：一位用户寻求与 LLM 开发者合作，为 AI 视频创作专家开发能够自主生成视频的模型。
  
  - 讨论强调了使用 **Luma** 的局限性，因为它是不开源且不可训练的，并强调了通过初始 Prompting 方法来实现更好的交互。
- **LLM 中的 Function Calling 讨论**：对 LLM 中 **function calling** 有效性的探索揭示了一个问题：参数过多会导致函数被忽略。
  
  - 建议使用参数最少的特定函数以获得更好的性能，并可能通过 **pre-filtering** 来增强发现能力。
- **现代 LLM 与 GPT-3 的对比**：成员们讨论了现代模型与 GPT-3 的性能对比，指出尽管参数较少，现代的 **7-13B** 模型表现优于早期的 GPT 版本。
  
  - 结论是，现代小模型在执行任务时通常与旧的大型模型表现相当，且效率更高。
- **使用 LLM 的文档提取工具**：关于适用于文档提取和分析的开源 LLM 出现了一个问题，结果显示像 **ChatGPT** 和 **Gemini** 这样的主流模型表现良好。
  
  - 然而，一位成员指出 **Llama 70B Instruct** 的表现并不理想，并征求有效替代方案的建议。

**提到的链接**：

- [Lambda Chat](https://lambda.chat/)：让 AI 开发者能够快速实验最佳的 AI 模型。
- [Notion – 集笔记、任务、维基和数据库于一体的工作空间。](https://southbridge-research.notion.site/Entropixplained-11e5fec70db18022b083d7d7b0e93505)：一款将日常工作应用融合在一起的新工具。它是为您和您的团队打造的一体化工作空间。

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1307489099346743388) (8 条消息🔥):

> - `为领域自适应微调 LLMs`
> - `机器人任务中 VLA 模型的评估`
> - `解锁 LLMs 的推理能力`
> - `LLaVA 与结构化生成`
> - `用于增强视觉表示的 LLM2CLIP`

- **LLMs 的微调策略**：题为 ["Fine-tuning large language models for domain adaptation"](https://arxiv.org/abs/2409.03444) 的论文探讨了各种训练策略，揭示了模型合并（model merging）可以在领域特定评估中带来卓越的能力。
  
  - 它还强调，较小的 LLMs 在模型合并过程中可能不会表现出涌现能力（emergent capabilities），这暗示了模型缩放（model scaling）的重要性。
- **机器人领域中的视觉语言模型**：["Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks"](https://multinet.ai/static/pages/Multinetv01.html) 在 20 个任务中评估了 VLA 模型，结果显示性能随任务复杂度显著变化。
  
  - 一种新型提示工程（prompt engineering）框架的引入，通过有效地将 VLMs 映射到模态任务，展示了性能的提升。
- **利用 LaTRO 解锁推理能力**：提出的 LaTent Reasoning Optimization (LaTRO) 框架旨在提高 LLMs 的推理能力，在 GSM8K 数据集上实现了高达 **12.5% 的零样本准确率提升**，详见论文 ["Language Models are Hidden Reasoners"](https://arxiv.org/abs/2411.04282)。
  
  - 这种方法允许 LLMs 在没有外部反馈的情况下增强其推理能力，展示了潜在的推理潜力。
- **结构化生成增强推理**：围绕 LLaVA-o1 模型的讨论强调了结构化生成在使 VLMs 能够逐步推理方面的有效性，如该 [推文](https://x.com/_akhaliq/status/1858378159588040774) 中所分享。
  
  - 一位用户指出，在这种背景下增加推理字段显著提升了性能。
- **LLM2CLIP 利用 LLM 优势**：["LLM2CLIP"](https://arxiv.org/abs/2411.04997) 提出了一种通过利用 LLMs 的文本理解能力来改进跨模态表示学习，从而增强 CLIP 能力的方法。
  
  - 通过在标题空间（caption space）中微调 LLMs，该方法允许使用更长、更复杂的标题，从而在各种任务中实现性能的实质性提升。

**提到的链接**：

- [来自 AK (@_akhaliq) 的推文](https://x.com/_akhaliq/status/1858378159588040774): LLaVA-o1 让视觉语言模型能够逐步推理
- [Language Models are Hidden Reasoners: Unlocking Latent Reasoning Capabilities via Self-Rewarding](https://arxiv.org/abs/2411.04282): 大语言模型 (LLMs) 展示了令人印象深刻的能力，但在需要多个步骤的复杂推理任务中仍然面临困难。虽然像思维链 (CoT) 这样基于提示的方法可以...
- [GitHub - SalesforceAIResearch/LaTRO](https://github.com/SalesforceAIResearch/LaTRO): 通过在 GitHub 上创建账户，为 SalesforceAIResearch/LaTRO 的开发做出贡献。
- [Multinetv0.1](https://multinet.ai/static/pages/Multinetv01.html): 未找到描述
- [MultiNet/src/modules at main · ManifoldRG/MultiNet](https://github.com/ManifoldRG/MultiNet/tree/main/src/modules): 通过在 GitHub 上创建账户，为 ManifoldRG/MultiNet 的开发做出贡献。
- [GitHub - ManifoldRG/MultiNet](https://github.com/ManifoldRG/MultiNet/tree/main): 通过在 GitHub 上创建账户，为 ManifoldRG/MultiNet 的开发做出贡献。
- [SOCIAL MEDIA TITLE TAG](https://multinet.ai/): SOCIAL MEDIA DESCRIPTION TAG TAG

---

### **Nous Research AI ▷ #**[**interesting-links**](https://discord.com/channels/1053877538025386074/1132352574750728192/1307411041939881994) (5 条消息):

> - `Agentic Workflow and Fine-Tuning`
> - `LLaMA-Mesh Announcement`
> - `AnyModal Framework`

- **关于 Agentic Workflow 技术的提问**：一位成员询问是否使用了 **fine-tuning**，或者 **agentic workflow** 是否采用了 few-shot prompting 进行模型训练。
  
  - 这引发了另一位成员分享他们在面对类似挑战时的经验，提到“这是我在尝试实现类似功能时遇到的瓶颈”。
- **Nvidia 发布 LLaMA-Mesh**：**Nvidia** 展示了 **LLaMA-Mesh**，该项目专注于使用 **Llama 3.1 8B** 生成 3D mesh，并承诺很快会发布权重。
  
  - 据报道，这一公告引起了社区的关注，因为它标志着 **3D 生成技术** 的进步。
- **AnyModal 框架介绍**：一位成员分享了关于 **AnyModal** 的细节，这是一个灵活的框架，旨在将图像和音频等多种数据类型与 Large Language Models 集成。
  
  - 他们强调了该框架使用 **ViT** 等模型处理 **图像输入** 任务的能力，并寻求反馈或贡献以改进这个正在进行的项目。

**提到的链接**：

- [GitHub - ritabratamaiti/AnyModal: AnyModal is a Flexible Multimodal Language Model Framework](https://github.com/ritabratamaiti/AnyModal): AnyModal 是一个灵活的多模态语言模型框架 - ritabratamaiti/AnyModal
- [来自 Chubby♨️ (@kimmonismus) 的推文](https://fxtwitter.com/kimmonismus/status/1857803310369009850?s=46): Nvidia 发布 LLaMA-Mesh：使用 Llama 3.1 8B 生成 3D Mesh。承诺权重即将发布。

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1307489099346743388) (8 条消息🔥):

> - `Fine-tuning Large Language Models`
> - `Benchmarking Vision, Language, & Action Models`
> - `Latent Reasoning Capabilities in LLMs`
> - `LLaVA and Step-by-Step Reasoning`
> - `LLM2CLIP for Enhanced Visual Representation`

- **微调策略增强了 LLM 在专业任务中的表现**：该论文介绍了微调策略，包括 Continued Pretraining 和 Direct Preference Optimization，以提高 LLM 在材料科学等领域专业应用中的性能。
  
  - 合并微调后的模型产生了超出单个父模型的新功能，这表明 Scaling 对于较小的模型可能至关重要。
- **在机器人任务上评估 VLA 模型**：研究人员提出了一个框架，用于在 20 个真实世界的机器人任务中评估 Vision-Language-Action (VLA) 模型，揭示了显著的性能差异。
  
  - 值得注意的是，所有模型在需要多个步骤的复杂任务中都表现挣扎，凸显了改进 Prompt Engineering 技术的需求。
- **利用 LaTRO 释放 LLM 的推理能力**：新型的 LaTent Reasoning Optimization (LaTRO) 框架在不依赖外部反馈的情况下提高了 LLM 的推理能力，在推理任务中实现了高达 12.5% 的准确率提升。
  
  - 在多个模型架构上，通过 GSM8K 和 ARC-Challenge 数据集进行了验证，证实了预训练的 LLM 具备潜在的推理能力。
- **LLaVA 促进了 Vision Language Models 中的逐步推理**：讨论中的见解表明，结构化生成增加的推理字段显著提升了 LLaVA 模型的性能。
  
  - 成员们注意到，鼓励逐步推理对于增强模型输出非常有效。
- **LLM2CLIP 利用 LLM 改进多模态学习**：LLM2CLIP 框架提出利用强大的语言模型，通过在 Caption 空间进行微调，来增强 CLIP 的视觉表示能力。
  
  - 这允许加入更长、更复杂的 Caption，从而在跨模态任务中取得实质性改进。

**提到的链接**：

- [来自 AK (@_akhaliq) 的推文](https://x.com/_akhaliq/status/1858378159588040774)：LLaVA-o1 让 Vision Language Models 能够逐步推理
- [Language Models are Hidden Reasoners: Unlocking Latent Reasoning Capabilities via Self-Rewarding](https://arxiv.org/abs/2411.04282)：Large Language Models (LLMs) 展示了令人印象深刻的能力，但在需要多个步骤的复杂推理任务中仍然表现挣扎。虽然像 Chain-of-Thought (CoT) 这样基于 Prompt 的方法可以...
- [GitHub - SalesforceAIResearch/LaTRO](https://github.com/SalesforceAIResearch/LaTRO)：通过在 GitHub 上创建账号来为 SalesforceAIResearch/LaTRO 的开发做出贡献。
- [Multinetv0.1](https://multinet.ai/static/pages/Multinetv01.html)：未找到描述
- [MultiNet/src/modules at main · ManifoldRG/MultiNet](https://github.com/ManifoldRG/MultiNet/tree/main/src/modules)：通过在 GitHub 上创建账号来为 ManifoldRG/MultiNet 的开发做出贡献。
- [GitHub - ManifoldRG/MultiNet](https://github.com/ManifoldRG/MultiNet/tree/main)：通过在 GitHub 上创建账号来为 ManifoldRG/MultiNet 的开发做出贡献。
- [SOCIAL MEDIA TITLE TAG](https://multinet.ai/)：SOCIAL MEDIA DESCRIPTION TAG TAG

---

### **Nous Research AI ▷ #**[**reasoning-tasks**](https://discord.com/channels/1053877538025386074/1264666760972472481/1307291884133154837) (19 条消息🔥):

> - `Dynamic Model Selection`
> - `AI Newsletters`
> - `Link Aggregation Tools`
> - `Scraping Tools`
> - `LocalLlama Community`

- **为 LM 设计游戏以比较能力**：一名成员提议开发一款供语言模型参与的游戏，这可以作为评估其能力的有效方法。
  
  - 该想法基于 `Dynamic Model Selection` 的概念，并可能增强对比分析。
- **AI Newsletter 认可**：一名成员祝贺其他人登上了各种 AI Newsletters 的榜首，强调了对其工作的认可。
  
  - 他们提到了一个 “O1 挑战者”，表明了 AI 内容创作领域内的竞争态势。
- **对 AI Newsletter 名称的看法**：有讨论认为 “Alphasignal” 这个名字不太吸引人，但该 Newsletter 本身提供了有价值的论文和开源工具推荐。
  
  - 成员们对总结重要资源的 Newsletter 表示赞赏，这使得获取最新动态变得更加容易。
- **Cloudflare 的新 Scraping Tool**：一位用户表示有兴趣利用 Cloudflare 的新抓取工具构建一个 Agent，用于汇总来自各种 AI 来源和社交媒体的链接。
  
  - 这种方法旨在简化从顶级 AI 实验室、Twitter 和 Hacker News 收集信息的过程。
- **对 LocalLlama 社区的复杂感受**：LocalLlama 社区收到了褒贬不一的反馈，一名成员对版主表示不满，同时也承认这是一个伟大的社区。
  
  - 尽管对管理有所抱怨，但人们承认该社区在 AI 讨论方面的价值。

---

### **Interconnects (Nathan Lambert) ▷ #**[**events**](https://discord.com/channels/1179127597926469703/1179127598442348729/1307425087171072010) (4 条消息):

> - `Dinner Reservations`
> - `Toronto and Vancouver Connections`

- **晚餐预订变得一团糟**：一位成员对预订表示沮丧，称其将变得**一团糟**。
  
  - *Lol* 似乎总结了围绕制定晚餐计划的麻烦情绪。
- **多伦多成员提供温哥华方面的帮助**：一位常驻**多伦多**的成员提议联系**温哥华**的人员寻求协助。
  
  - 这激发了人们对两座城市之间潜在合作的兴趣。

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1307380062110224496) (106 messages🔥🔥):

> - `Qwen 2.5 Turbo`
> - `Mistral AI Updates`
> - `API Challenges`
> - `Deepseek Models`

- **Qwen 2.5 Turbo 发布**：全新的 [Qwen 2.5 Turbo](https://qwenlm.github.io/blog/qwen2.5-turbo/) 拥有高达 100 万 tokens 的上下文长度，处理速度显著提升，加速达 4.3 倍。
  
  - 它现在支持以极具竞争力的价格处理更大规模的 tokens，其强大的能力引发了广泛关注。
- **Mistral AI 发布 Pixtral Large**：[Mistral AI](https://mistral.ai/news/mistral-chat/) 宣布了多项更新，包括多模态模型 Pixtral Large，该模型在多个基准测试中表现出色。
  
  - 这些更新还提升了其聊天平台的能力，为用户交互引入了新的工具和功能。
- **对 API 性能的担忧**：成员们对 API 性能表示不满，指出某些模型存在巨大的延迟峰值，用户体验挑战重重。
  
  - 会议强调，构建高效的 API 仍然是一项复杂的任务，特别是对于在受限条件下运营的公司而言。
- **Deepseek 3 备受期待**：人们对 [Deepseek 3](https://www.deepseek.ai/) 的期待日益高涨，目前的讨论暗示 2.5 VL 版本可能即将发布。
  
  - 社区成员表达了对模型能力进一步提升的渴望，同时也认可了中国模型在创新方面的表现。

**相关链接**：

- [Joseph Thacker (@rez0__) 的推文](https://x.com/rez0__/status/1858535550023504249)：太棒了！@MistralAI 今天发布了新的 SOTA。
- [Mistral 已加入聊天](https://mistral.ai/news/mistral-chat/)：搜索、视觉、构思、编程……全部免费供您使用。
- [N8 Programs (@N8Programs) 的推文](https://x.com/N8Programs/status/1858543034826469499)：@JustinLin610 @kalomaze 查看了基准测试——大致与 Qwen 2 72B VL 持平（或在某些方面更差）。理所当然。
- [Devendra Chaplot (@dchaplot) 的推文](https://x.com/dchaplot/status/1858541283687755885)：Pixtral Large：- 尖端级多模态模型 - 在 MathVista, DocVQA, VQAv2 上达到 SOTA - 保持了 Mistral Large 2 的文本性能 - 123B 解码器，1B 视觉编码器 - 128K 序列长度。在 HF 上下载：...
- [vik (@vikhyatk) 的推文](https://x.com/vikhyatk/status/1857870293378937345)：我隐藏了一个不可告人的秘密。
- [Qwen (@Alibaba_Qwen) 的推文](https://x.com/alibaba_qwen/status/1858469845958074541?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ)：在 Qwen2.5 发布后，我们听到了社区对处理更长上下文的需求。https://qwenlm.github.io/blog/qwen2.5-turbo/ 今天，我们自豪地推出全新的 Qwen2.5-Turbo 版本...
- [Xeophon (@TheXeophon) 的推文](https://x.com/TheXeophon/status/1858537941087174725)：引用 Joseph Thacker (@rez0__)：太棒了！@MistralAI 今天发布了新的 SOTA。
- [Fireworks - 最快的生成式 AI 推理](https://fireworks.ai/blog/fireworks-f1)：使用 Fireworks AI 以极快的速度运行最先进的开源 LLM 和图像模型，或者免费微调并部署您自己的模型！

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-drama**](https://discord.com/channels/1179127597926469703/1181746144821387334/1307080957903245413) (9 messages🔥):

> - `AI Hallucination 担忧`
> - `Ilya Sutskever 与 Sam Altman 的分歧 (Misalignment)`
> - `TechEmails Twitter 爆料`

- **对 AI Hallucinations 的负面情绪**：一位成员对目前围绕 **AI hallucinations** 的讨论表示沮丧，提到讨论往往围绕着负面看法，比如“AI 产生了幻觉，这很糟糕”。
  
  - *“理论上我在意识形态上是一致的”*，暗示了信念与实践之间的分歧。
- **Ilya 和 Sam 自 2017 年以来的对齐 (Alignment) 问题**：讨论透露，**Ilya Sutskever** 和 **Sam Altman** 早在 **2017** 年就意识到了对齐 (misalignment) 问题。
  
  - 这一说法为他们持续的关系以及对 AI 发展的影响增添了一层复杂性。
- **TechEmails 揭开的戏剧性事件**：一位用户分享了来自 **TechEmails** 的多个链接，展示了关于涉及 **Ilya Sutskever**、**Elon Musk** 和 **Sam Altman** 的著名邮件往来的最新爆料推文。
  
  - 这些链接被标记为**引人入胜**，并暗示了围绕该情况的潜在**戏剧性冲突 (drama)**。

**提到的链接**：

- [来自 Internal Tech Emails (@TechEmails) 的推文](https://x.com/techemails/status/1857456141875196380?s=46)：未发现描述
- [来自 undefined 的推文](https://vxtwitter.com/TechEmails/status/1857456137156669765)：未发现描述
- [来自 undefined 的推文](https://vxtwitter.com/TechEmails/status/1857456139547316359)：未发现描述
- [来自 undefined 的推文](https://vxtwitter.com/TechEmails/status/1857456141875196380)：未发现描述
- [来自 undefined 的推文](https://vxtwitter.com/TechEmails/status/1857456144211423482)：未发现描述
- [来自 Internal Tech Emails (@TechEmails) 的推文](https://x.com/TechEmails/status/1857459526267449790)：[此文件来自 Elon Musk 等人诉 Samuel Altman 等人 (2024)。]

---

### **Interconnects (Nathan Lambert) ▷ #**[**random**](https://discord.com/channels/1179127597926469703/1183121795247779910/1307116264249884723) (74 条消息🔥🔥):

> - `2T Pretraining Dataset Comparison` (2T 预训练数据集对比)
> - `OLMoE ICLR Review Concerns` (OLMoE ICLR 评审争议)
> - `Gemini Model Improvements` (Gemini 模型改进)
> - `RewardBench for RLHF Evaluation` (用于 RLHF 评估的 RewardBench)
> - `New LLaVA-o1 Model Release` (新 LLaVA-o1 模型发布)

- **对 2T 预训练热潮的质疑**：一位成员对围绕 **2T 预训练数据集** 的兴奋情绪表示困惑，考虑到已经存在具有更好版权过滤的 **15T FineWeb 数据集**，质疑其重要性。
  
  - *多语言能力* 被认为是 FineWeb 方法的一个潜在优势，同时也引发了对其在现实世界应用中的好奇。
- **对 OLMoE ICLR 评审质量的担忧**：针对 **ICLR** 上 **OLMoE** 论文的低质量评审展开了讨论，一些成员暗示评审员对某些作者存在偏见。
  
  - 这种观点得到了共鸣，即目前顶会的评审标准正在下降，强制性评审未能提升质量。
- **Gemini 模型的强劲表现**：来自 Google 的 **Gemini-exp-1114** 模型因其令人印象深刻的推理能力而受到关注，超越了之前的版本，并在 **AIME 2024 数据集** 上排名靠前。
  
  - 成员们推测该模型可能是 Gemini 2.0 的预发布版本，并对其潜力表现出极大的热情。
- **用于评估 RM 的 RewardBench 发布**：讨论了 **RewardBench** 的推出，这是一个用于评估人类反馈强化学习（RLHF）中奖励模型（Reward Models）的基准测试，旨在增强对对齐技术的理解。
  
  - 针对大纲中使用的数据集的处理方式提出了担忧，并提到了对作者的抄袭指控。
- **新 LLaVA-o1 视觉语言模型发布**：**LLaVA-o1** 模型作为一种新的视觉语言模型发布，其表现优于主要竞争对手，并采用了一种新颖的推理方法。
  
  - 讨论包括最终评估其对比 **Qwen2-VL** 性能的计划，同时注意到目前在 **Hugging Face** 上尚不可用。

**提到的链接**：

- [Asankhaya Sharma (@asankhaya) 的推文](https://x.com/asankhaya/status/1857687563357843939)：来自 @Google 的新 gemini-exp-1114 模型在推理方面表现相当出色。它比 gemin-1.5-pro-002 有了巨大提升，在 AIME (2024) 数据集上仅次于 o1-preview。附图……
- [RewardBench: Evaluating Reward Models for Language Modeling](https://openreview.net/forum?id=XiConLcsqq#discussion)：奖励模型 (RMs) 是成功的 RLHF 将预训练模型与人类偏好对齐的关键，然而目前针对这些奖励模型评估的研究相对较少……
- [Guowei Xu (@Kevin_GuoweiXu) 的推文](https://x.com/Kevin_GuoweiXu/status/1858441262933962891)：抱歉疏忽了！我们并非有意不包含此模型。今天我们测试了 Qwen2-VL-7B，发现我们模型的性能与其大致相当。虽然我们承认 Qwen2-VL 有一个 ver...
- [Nathan Lambert (@natolambert.bsky.social)](https://bsky.app/profile/natolambert.bsky.social/post/3lb3kzfe3f72t)：今天才发现，用 huggingface 训练 MoE 时，它会对专家进行 for 循环。难怪它们这么慢 🫠
- [finbarr (@finbarrtimbers) 的推文](https://x.com/finbarrtimbers/status/1858180568430854563)：我正在写一篇文章，论点是：1) RL 很难且并不真正奏效，但 2) 我们不得不使用它，因为一旦耗尽了所有高质量的 p...
- [Guowei Xu (@Kevin_GuoweiXu) 的推文](https://x.com/Kevin_GuoweiXu/status/1858338565463421244)：🚀 隆重推出 LLaVA-o1：首个能够进行自发、系统性推理的视觉语言模型，类似于 GPT-o1！🔍 🎯我们的 11B 模型表现优于 Gemini-1.5-pro、GPT-4o-mini 和 Llama-3.2-90B-...
- [microsoft/orca-agentinstruct-1M-v1 · Hugging Face 数据集](https://huggingface.co/datasets/microsoft/orca-agentinstruct-1M-v1)：未找到描述

---

### **Interconnects (Nathan Lambert) ▷ #**[**memes**](https://discord.com/channels/1179127597926469703/1187551504995987576/1307614459136708608) (42 messages🔥):

> - `Reinforcement Learning (RL)`
> - `Tulu 3 vs Hermes 3`
> - `Llama 3 Paper`
> - `OpenAI's Early Works`
> - `Anthropic's Model Release`

- **强化学习面临质疑**：一些成员讨论了对用于微调的 **Reinforcement Learning** (RL) 的怀疑态度，强调尽管有了极具前景的工具和进展，许多人仍对其功效持怀疑态度。一位成员表示：*“很多人仍然怀疑 RL 在微调中的作用”*，并对 **2025** 年获得更广泛认可表示乐观。
- **关于 Tulu 3 和 Hermes 3 的推测**：有人询问 Tulu 3 是否已针对 Hermes 3 进行了评估，一名成员对目前切换到 **Llama 405** 的结果表示不满。另一名成员提到了过去的评估，但推测 *“他们的情况进展得并不顺利。”*
- **对 Llama 3 论文的兴奋**：一位成员强调 **Llama 3 论文** 可能是今年必读的论文，并指出了它与该领域其他论文相比的影响力。另一位成员提到它深入探讨了 **LLM 的物理学 (physics of LLMs)**，强调了其重要性。
- **怀念 OpenAI 的早期贡献**：成员们表达了对 **OpenAI** 早期作品的怀念，提到电脑游戏的 RL 如何激励了一些人追求 AI 事业。讨论反映了与目前研究更封闭的性质相比，那些早期的贡献如何激发了对该领域的兴趣。
- **Anthropic 的发布方式受到批评**：针对 **Anthropic** 的模型发布提出了担忧，认为其流程与 OpenAI 相比缺乏透明度。成员们批评 Anthropic 除了模型评估之外几乎没有提供任何见解，评论道：*“这就是一个更好的模型，这是评估结果，再见。”*

**提到的链接**：

- [Teknium (e/λ) (@Teknium1) 的推文](https://x.com/Teknium1/status/1858055611533111333)：@kalomaze @TheXeophon @vikhyatk @YouJiacheng RL 很烂，不接受反驳，哈哈。
- [A Deep Reinforced Model for Abstractive Summarization](https://arxiv.org/abs/1705.04304)：基于注意力机制、RNN 的编码器-解码器模型在短输入和输出序列的抽象摘要中取得了良好性能。然而，对于较长的文档和摘要，这些模型……

---

### **Interconnects (Nathan Lambert) ▷ #**[**reads**](https://discord.com/channels/1179127597926469703/1214764639397617695/1307765736772468797) (2 messages):

> - `Doria's Work`
> - `Understanding Spoken Content`

- **对 Doria 工作的赞赏**：一位成员分享了一个[展示 Doria 工作的 YouTube 视频](https://www.youtube.com/watch?v=EtLqivKd4m4)，并表达了对她贡献的钦佩。
  - 对 Doria 工作的热情表明了对其项目和见解的浓厚兴趣。
- **理解上的挑战**：另一位成员指出，理解视频内容需要一些时间，需要多次收听才能完全掌握。
  - 他们强调需要坚持克服最初的困难，才能领会所说的话语。

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1307076326062166047) (72 messages🔥🔥):

> - `Pixtral Large release`
> - `Mistral's Le Chat updates`
> - `Qwen 2.5 Turbo features`
> - `Lindy usability concerns`
> - `OpenAI streaming availability`

- **Pixtral Large 作为新型多模态模型首次亮相**：Mistral 宣布发布 **Pixtral Large**，这是一个前沿级 **124B 多模态模型**，能够理解文档和图像，同时保持文本性能，可在其 API 和 [Hugging Face](https://huggingface.co/mistralai/Pixtral-Large-Instruct-2411) 上获取。
  - 它拥有 **128K 上下文窗口**，可以处理至少 **30 张高分辨率图像**，展示了多模态 AI 的进步。
- **Mistral 更新 Le Chat 并推出新功能**：Mistral 的 **Le Chat** 现在包含 **网页搜索**、**图像生成**和 **PDF 上传**等功能，旨在增强用户与生成式 AI 工具的交互。
  - 该平台现在免费使用，支持从编程助手到创意合作伙伴的各种用例，推动了 AI 界面的边界。
- **Qwen 2.5 Turbo 引入扩展上下文支持**：**Qwen 2.5 Turbo** 现在支持高达 **100 万个 token** 的上下文长度，允许进行相当于 **十本小说** 的全面文本处理。
  - 该模型在 Passkey 检索任务上实现了 **100% 的准确率**，显著增强了开发者的长上下文能力。
- **关于 Lindy 使用体验的评价褒贬不一**：一位用户表达了他们在寻找 **Lindy** 实质性用例方面的困扰，质疑其相对于 Zapier 等其他自动化工具的优势。

- 反馈反映了社区对其在当前工作流中的实用性和有效性存在更广泛的犹豫。
- **OpenAI 为 o1 模型添加流式传输（streaming）功能**：OpenAI 已为 **o1-preview** 和 **o1-mini** 模型提供 **streaming** 支持，并将访问权限扩展至所有付费使用层级。
  
  - 这一增强功能使开发者能够在 OpenAI 平台内利用更动态的交互，从而提升用户体验。

**提到的链接**：

- [来自 Hassan (@nutlope) 的推文](https://x.com/nutlope/status/1857133161198829644?s=46)：介绍 LogoCreator！一个开源的徽标生成器，使用 @togethercompute 上的 Flux Pro 1.1 在几秒钟内创建专业徽标。100% 免费且开源。演示 + 代码：
- [来自 ken (@aquariusacquah) 的推文](https://x.com/aquariusacquah/status/1857489012908503374?s=46)：语音 AI 的普及将是 2024 年最大的发展，而且它才刚刚开始发挥作用。——结合了 ASR, LLM, TTS 的“语音管道（Voice Pipeline）”方法正变得越来越快……
- [来自 Hassan (@nutlope) 的推文](https://x.com/nutlope/status/1857681239626387515?s=46)：llama-ocr 居然在 hackernews 上排名第一！
- [Fireworks - 生成式 AI 的最快推理](https://fireworks.ai/blog/fireworks-f1)：以极快的速度使用最先进的开源 LLM 和图像模型，或者使用 Fireworks AI 免费微调并部署您自己的模型！
- [来自 Flo Crivello (@Altimor) 的推文](https://x.com/altimor/status/1857892579389665694?s=46)：这触动了大家的神经 :) 正如预期的那样，收到了很多反对意见。我将在下面回答最常见的问题：« 生活不一定非得是为了工作！ » 是的！这就是所谓的缺乏雄心！这完全没问题……
- [来自 Alex Albert (@alexalbert__) 的推文](https://x.com/alexalbert__/status/1857457290917589509)：周五文档功能发布：您现在可以将我们所有的文档作为一个纯文本文件进行访问，并将其输入到任何 LLM 中。以下是 URL 路径：http://docs.anthropic.com/llms-full.txt
- [Pixtral Large](https://mistral.ai/news/pixtral-large/)：Pixtral 成长了。
- [Mistral 已加入对话](https://mistral.ai/news/mistral-chat/)：搜索、视觉、构思、编码……全部免费供您使用。
- [来自 Arc Institute (@arcinstitute) 的推文](https://x.com/arcinstitute/status/1857138107038187945?s=46)：🧬Evo，第一个在 DNA 上进行大规模训练的基础模型，是生物学的罗塞塔石碑。DNA、RNA 和蛋白质是生命的基本分子——破解它们复杂语言的代码……
- [来自 OpenAI Developers (@OpenAIDevs) 的推文](https://x.com/openaidevs/status/1858609150999359559?s=46)：流式传输（Streaming）现在可用于 OpenAI o1-preview 和 o1-mini。🌊 https://platform.openai.com/docs/api-reference/streaming 我们已经向所有付费使用层级的开发者开放了这些模型的访问权限……
- [来自 Szymon Rączka (@screenfluent) 的推文](https://x.com/screenfluent/status/1858426436614246832)：@0xLes @AravSrinivas @eshear 我非常喜欢这个倡议，以至于我在周末制作了这个项目 https://llmstxt.directory/
- [将上下文长度扩展到 1M Token！](https://qwenlm.github.io/blog/qwen2.5-turbo/)：API 文档（中文） HuggingFace 演示 ModelScope 演示 介绍 在 Qwen2.5 发布后，我们听到了社区对处理更长上下文的需求。在最近几个月里，我们……
- [Spaceballs Crosseyed GIF - Spaceballs Crosseyed Sorry Sir - 发现并分享 GIF](https://tenor.com/view/spaceballs-crosseyed-sorry-sir-doing-my-best-gif-5886487)：点击查看 GIF
- [来自 Aidan McLau (@aidan_mclau) 的推文](https://x.com/aidan_mclau/status/1857963832309665973?s=46&t=MGz8l5Z36lvN2cHgl1IVqA)：Elon 失去了天命，宇宙射线不在他那边，rip xAI；那个看起来像 GPU 的数据中心挺酷的，我觉得
- [来自 Arthur Mensch (@arthurmensch) 的推文](https://x.com/arthurmensch/status/1858567024609276372?s=46)：从一家科学公司扩展为一家科学和产品公司并非易事，这次发布是我们旅程中的一个非常重要的里程碑。我们期待着您将如何使用 le ……
- [来自 Devendra Chaplot (@dchaplot) 的推文](https://x.com/dchaplot/status/1858541281468915937?s=46)：今天，我们宣布两个令人兴奋的新更新：Pixtral Large：前沿级 124B 多模态模型，为新的 Le Chat 提供动力。全新的 Le Chat：具有网络搜索、canvas、图像生成、图像理解……
- [来自 Arthur Mensch (@arthurmensch) 的推文](https://x.com/arthurmensch/status/1858568631358988691?s=46)：在 Mistral，我们意识到要创造最佳的 AI 体验，需要共同设计模型和产品界面。Pixtral 在训练时就考虑到了高影响力的前端应用，并且……
- [Fireworks - 生成式 AI 的最快推理](https://fireworks.ai/blog/f)：以极快的速度使用最先进的开源 LLM 和图像模型，或者使用 Fireworks AI 免费微调并部署您自己的模型！

- [来自 deepfates (@deepfates) 的推文](https://x.com/deepfates/status/1858233583279993014?s=46)：我写了一个 Python 脚本，可以将你的 Twitter 存档转换为训练数据集，用于根据你的个性微调语言模型。它还能将你所有的推文、线程和媒体内容提取为 Markdown 格式...
- [来自 Guillaume Lample @ ICLR 2024 (@GuillaumeLample) 的推文](https://x.com/guillaumelample/status/1858579532380340382?s=46)：Le Chat 现在包含使用 FLUX1.1 的图像生成、网页搜索、Canvas、具备视觉能力的 Mistral Large、PDF 上传等功能。而且它是 100% 免费的！https://chat.mistral.ai/ 引用 Mistral AI (@...
- [谷歌新模型排名“第一 LLM”，但存在一个问题](https://www.youtube.com/watch?v=5uJ8XPvn6kY)：一个新的神秘 Gemini 模型出现在排行榜榜首，但这就是全部真相吗？我深入挖掘标题背后的内容，为你展示一些令人扫兴的...
- [将你的 Twitter 存档转换为训练数据集和 Markdown 文件](https://gist.github.com/deepfates/78c9515ec2c2f263d6a65a19dd10162d)：将你的 Twitter 存档转换为训练数据集和 Markdown 文件 - convert_archive.py

---

### **Latent Space ▷ #**[**ai-in-action-club**](https://discord.com/channels/822583790773862470/1200548371715342479/1307087857059500136) (55 条消息🔥🔥):

> - `Windsurf Editor`
> - `Anthropic API`
> - `Cursor vs Windsurf`
> - `Codeium Demo Enhancements`
> - `AI's Context Management`

- **Windsurf Editor 融合了 AI 与开发**：[Windsurf Editor](https://codeium.com/windsurf) 允许开发者与 AI 无缝协作，创造出一种“简直像魔法一样”的编码体验。
  
  - 它的 AI 能力融合了类似 Copilot 的协作功能和用于复杂任务的独立 Agent，让开发者保持在**心流状态**。
- **Anthropic API 展示**：在一次演示中，人们注意到 Anthropic API 已成功集成到桌面客户端以增强功能，如[这个 GitHub 示例](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo)所示。
  
  - 参与者讨论了 **agent.exe** 等工具令人印象深刻的能力，以及 AI 如何利用像素坐标生成鼠标点击。
- **Cursor 与 Windsurf 的用户体验对比**：用户表达了对 **Cursor** 和 **Windsurf** 的偏好，强调 Cursor 提供的细粒度控制对高级用户非常有益。
  
  - 相反，其他人指出 Windsurf 偶尔会做出意想不到的决定，因此理解交互过程对于有效使用至关重要。
- **Codeium 演示亮点**：在 Codeium 演示中，参与者注意到 Agent 如何提供最近更改的 git diff，展示了其在跟踪代码演变方面的实用性。
  
  - 讨论还显示，用户拥有自研脚本来简化 OpenWebUI 等工具的管理。
- **AI 的战略性上下文管理**：AI 工具中的上下文管理一直是讨论的热点，特别是关于工具如何处理和执行复杂指令。
  
  - 人们对保持交互的控制权和透明度表示担忧，尤其是在工作流中运行多个 LLM 时。

**提到的链接**：

- [Alf Alf Janela GIF - Alf Alf Janela Alf Observando - 发现并分享 GIF](https://tenor.com/view/alf-alf-janela-alf-observando-gif-18537662)：点击查看 GIF
- [Alf Tux GIF - Alf Tux - 发现并分享 GIF](https://tenor.com/view/alf-tux-gif-20040572)：点击查看 GIF
- [Codeium 推出的 Windsurf Editor](https://codeium.com/windsurf)：未来的编辑器，就在今天。Windsurf Editor 是首个由 AI Agent 驱动的 IDE，让开发者保持在心流状态。现已支持 Mac、Windows 和 Linux。
- [anthropic-quickstarts/computer-use-demo 在 main 分支 · anthropics/anthropic-quickstarts](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo)：一系列旨在帮助开发者快速上手使用 Anthropic API 构建可部署应用程序的项目 - anthropics/anthropic-quickstarts
- [GitHub - corbt/agent.exe](https://github.com/corbt/agent.exe)：通过在 GitHub 上创建账户为 corbt/agent.exe 的开发做出贡献。
- [GitHub - olimorris/codecompanion.nvim: ✨ AI 驱动的编码，在 Neovim 中无缝运行。支持 Anthropic, Copilot, Gemini, Ollama, OpenAI 和 xAI 的 LLM](https://github.com/olimorris/codecompanion.nvim)：✨ AI 驱动的编码，在 Neovim 中无缝运行。支持 Anthropic, Copilot, Gemini, Ollama, OpenAI 和 xAI 的 LLM - olimorris/codecompanion.nvim

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1307254257522704415) (18 条消息🔥):

> - `CUDA 中的 Dynamic Parallelism`
> - `(G)H200 的云服务提供商`
> - `Zoom 讲座`
> - `在云平台上学习 CUDA`
> - `CUDA Kernel 的 Profiling 信息`

- **发现 Dynamic Parallelism 资源**：一位成员分享说，[NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/3_CUDA_Features) 中所有以 `cdp` 开头的示例都演示了 dynamic parallelism，但 CDP2 的兼容性尚不确定。
  
  - 另一位成员指出，第一个示例缺少 device-side synchronization，这被认为是一个优点。
- **寻找 (G)H200 144GB 云端选项**：一位成员询问是否有云服务提供商能以按小时计费的方式提供 (G)H200 144GB 且没有过高门槛，并提到 Lambda 目前提供的是 GH200 96GB。
  
  - 另一位成员建议联系 Lambda 寻求帮助，并提到他们客服支持的体验很好。
- **关于 Zoom 讲座的信息**：一位成员询问讲座在哪里举行，随后确认是在 Zoom 上进行，并分享了链接：[Zoom Event](https://discord.com/events/1189498204333543425/1289331735279960116)。
  
  - 此外还提到，讲座已录制，方便无法参加的人观看。
- **学习 CUDA 的最佳云平台**：一位成员询问哪个云平台最容易搭建以用于学习 CUDA，得到的推荐是 cloud-gpus.com 上的 Lightning Studios。
  
  - 该回答表明存在多种选择，但强调了对特定平台的偏好。
- **获取 Kernel Profiling 信息**：一位成员寻求关于使用 `nsys` 对特定 kernel 进行 profiling 的帮助，表示在分析 nsys-rep 文件的文档选项中遇到了困难。
  
  - 建议改用 `ncu` (Nsight Compute)，因为它能提供所需的 kernel 级别详细信息。

 

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/1308113996246810686) (6 条消息):

> - `Triton 中修改版的 FlashAttention`
> - `结合 torch.compile 使用 Triton CPU 后端`
> - `社区幽默`

- **排查 FlashAttention 修改版的问题**：一位成员正在使用 Triton 实现修改版的 **FlashAttention**，但在 Colab 中遇到了 `atomic_add` 导致 GPU 崩溃的问题。
  
  - 他们分享了代码，重点在于计算 attention score 矩阵的 **column-sum**（列和），并寻求实现方面的帮助。
- **咨询 Triton CPU 后端**：一位成员询问是否可以将 **Triton CPU backend** 与 `torch.compile` 结合使用，希望听取社区的见解。
  
  - 另一位用户提到了一位可能了解该领域的社区成员，展示了协作解决问题的精神。
- **社区幽默**：一位成员开玩笑说自己把 Triton 拼错了，引发了另一位成员的笑声。
  
  - 这反映了社区讨论中普遍存在的同志情谊和幽默感。

 

---

### **GPU MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1307127555874230333) (41 条消息🔥):

> - `高级 PyTorch 资源`
> - `使用 PyBind 的自定义 Kernel`
> - `PyTorch DCP 内存占用`
> - `FSDP State Dict 功能`

- **寻求高级 PyTorch 资源**：一位成员对现有的 PyTorch 书籍侧重于初学者概念表示沮丧，并询问涵盖高级主题和技巧的资源。
  
  - *建议通过动手实践来辅助学习*，推荐采用目标导向的方法，而非仅仅阅读方法论。
- **注册自定义 Kernel 的问题**：一位成员询问如何从 PyBind 注册自定义 Kernel，以及 `torch.library.register_kernel` 的正确语法。
  
  - 另一位成员提供了注册自定义算子的有用链接，但在识别算子名称时遇到了挑战。
- **检查操作期间的 DCP 内存占用**：在关于 PyTorch DCP 的讨论中，一位成员报告在混合精度和 **FULL_SHARD** 模式下执行 `dcp.save` 时，出现了显著的临时内存分配。
  
  - 成员们对分配的内存是否正常以及是否可以在未来迭代中进行优化表示了关注。
- **理解 FSDP 的内存分配**：会议澄清了 FSDP 使用 'flat parameters' 管理参数，并且在 **all-gathering** flat parameters 以及对其进行重新分片（re-sharding）时都需要内存。
  
  - 成员们讨论了他们的自定义 auto-wrap 策略如何影响 state dict 获取期间的内存分配，从而导致更大的预留内存。

**提到的链接**：

- [PyTorch documentation — PyTorch 2.5 documentation](https://pytorch.org/docs/stable)：未找到描述
- [Rethinking PyTorch Fully Sharded Data Parallel (FSDP) from First Principles](https://dev-discuss.pytorch.org/t/rethinking-pytorch-fully-sharded-data-parallel-fsdp-from-first-principles/1019#flatparameter-4)：鉴于大家的兴趣，我分享了一篇关于 PyTorch Fully Sharded Data Parallel (FSDP) 设计的笔记（最初为内部编写）。这涵盖了大部分内容，但并非全部（例如，不包括 autograd 和 CUDA cac...
- [torch.library — PyTorch 2.5 documentation](https://pytorch.org/docs/stable/library.html#creating-new-custom-ops-in-python))：未找到描述
- [pytorch/torch/distributed/checkpoint/filesystem.py at e80b1b2870ad568aebdbb7f5205f6665f843e0ea · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/e80b1b2870ad568aebdbb7f5205f6665f843e0ea/torch/distributed/checkpoint/filesystem.py#L169)：Python 中具有强大 GPU 加速能力的张量和动态神经网络 - pytorch/pytorch
- [Getting Started with Distributed Checkpoint (DCP) — PyTorch Tutorials 2.5.0+cu124 documentation](https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)：未找到描述
- [[Announcement] Deprecating PyTorch’s official Anaconda channel · Issue #138506 · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/138506)：摘要：由于 conda 构建的维护成本高昂，且观察到的投资回报率 (ROI) 不足以证明其合理性，PyTorch 将停止发布依赖于 Anaconda 默认包的 Anaconda 包...

---

### **GPU MODE ▷ #**[**announcements**](https://discord.com/channels/1189498204333543425/1189640399476764692/1307434586317389824) (1 条消息):

> - `Jay Shah 在 CUTLASS 的演讲`
> - `CUTLASS 中的 Epilogue Fusion`
> - `Proxmox VE 上的 GPU 直通`

- **Jay Shah 在 CUTLASS 分享见解**：**Jay Shah** 一场备受期待的演讲目前正在 **CUTLASS 和 Flash Attention 3** 活动中进行；他以关于高级 GPU 编程的通俗易懂的博客而闻名。
  
  - 参与者寄予厚望，称他的博客是进入这一复杂领域的最佳入门读物之一，[在此阅读更多](https://research.colfax-intl.com/blog/)。
- **详细讨论 Epilogue Fusion**：一篇题为《**使用 Epilogue Visitor Trees 在 CUTLASS 中实现 Epilogue Fusion**》的文章已发布，作为 NVIDIA GPU 上 **GEMM** 实现教程系列的补充。
  
  - 该文章深入探讨了 **CUTLASS** 的工作负载，重点关注 GEMM 的主循环，是更广泛系列文章的一部分，详见[此处](https://research.colfax-intl.com/epilogue_visitor_tree/)。
- **探索 Proxmox VE 8.2 上的 GPU 直通**：发布了一份 **Proxmox VE 8.2 上的 GPU 直通 (GPU passthrough)** 指南，旨在增强用户对该环境下 GPU 管理的理解。
  
  - 本文提供了实用的见解，是改进 [Proxmox](https://research.colfax-intl.com/gpu-passthrough-on-proxmox-ve-8/) 虚拟化技术持续努力的一部分。

 

**提到的链接**：[Articles](https://research.colfax-intl.com/blog/)：我们在博客上提供说明性文章和代码教程。

 

---

### **GPU MODE ▷ #**[**cool-links**](https://discord.com/channels/1189498204333543425/1189868872887705671/1307104269652070460) (2 messages):

> - `ZLUDA`
> - `CUDA Alternatives`
> - `AMD and Intel GPUs`

- **ZLUDA 让 CUDA 变得触手可及**：最近的一段 YouTube 视频，标题为 [#246 Developer Of ZLUDA: CUDA For Non Nvidia GPUs](https://www.youtube.com/watch?v=ze25Sie2gVQ)，讨论了 ZLUDA 如何在 **AMD** 和 **Intel GPU** 上实现 CUDA 功能，这对开发者来说是一个重大转变。
  
  - 一位成员提到，他们之前曾尝试联系该开发者，并对*终于有人邀请他参加节目*感到兴奋。
- **对开发者现身的兴奋**：讨论围绕视频中出现的 ZLUDA 开发者 **Andrzej Janik** 展开，他一直受到社区的追捧。
  
  - 成员们表达了他们的热情，其中一人表示他们*很久以前就尝试邀请他*。

**提到的链接**：[#246 Developer Of ZLUDA: CUDA For Non Nvidia GPUs | Andrzej Janik](https://www.youtube.com/watch?v=ze25Sie2gVQ)：CUDA 是人们购买 NVIDIA GPU 的主要原因之一，但如果有一种方法也能在 AMD 和 Intel GPU 上拥有这种计算能力呢？确实有……

---

### **GPU MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/1307082727782092862) (8 messages🔥):

> - `NCU Source View`
> - `Live Register Counts`
> - `CUDA and Tensor Core Clock Speeds`
> - `Thermal Throttling Mitigation`

- **NCU Source View 增强调试**：**NCU** 中的 **Source View** 允许跳转到活跃寄存器（Live Registers）最多的代码点，这可能有助于调试工作。
  
  - 这可以深入了解寄存器依赖关系如何影响性能。
- **质疑活跃寄存器计数的准确性**：一场关于活跃寄存器计数是否为准确指标的讨论展开了，担忧其累积性质可能会误导故障位置。
  
  - 参与者指出，高活跃寄存器计数可能仅仅表示之前的指令，而不是特定的当前故障。
- **CUDA 与 Tensor Core 的时钟频率**：澄清了 **CUDA cores** 运行在 **1,755MHz**，而 **Tensor cores** 运行在 **1,620MHz**。
  
  - 这些频率确认了之前讨论中强调的针对不同核心类型的更高时钟频率限制。
- **理解节点热管理**：来自 **NVIDIA** 的 Robert Crovella 确认，正如白皮书所述，**Tensor Cores** 的 **boost clock** 上限为 **1,620MHz**。
  
  - 参与者推测，这一上限可能是软件强加的限制，旨在缓解热节流（Thermal Throttling）风险，确保性能稳定。

**提到的链接**：[NVIDIA H100 Tensor Core GPU Architecture Overview](https://resources.nvidia.com/en-us-tensor-core)：NVIDIA H100、基于 H100 的新一代 DGX、DGX SuperPOD 和 HGX 系统以及基于 H100 的融合加速器的高层概述。随后深入探讨了 H100 硬件架构等……

---

### **GPU MODE ▷ #**[**pmpp-book**](https://discord.com/channels/1189498204333543425/1194427148656721970/1307260112804642886) (3 messages):

> - `CUDA grid and block configuration`
> - `Confusion in documentation`

- **理解 CUDA 配置**：一位成员将 CUDA 配置理解为具有 **16 个 block** (4 \* 2 \* 2) 的 grid，且每个 block 包含 **4 个 thread** (2 \* 2 \* 1)。
  
  - 他们寻求对这一理解的确认，展示了对 grid 和 block 维度的清晰掌握。
- **强调文档差异**：另一位成员指出文档 **第 68 页存在矛盾**，其中指出每个 grid 配置为 **4 个 block**，每个 block 有 **16 个 thread**。
  
  - 他们认为所呈现的文档可能与他们拥有的更正信息不同步。

---

### **GPU MODE ▷ #**[**youtube-recordings**](https://discord.com/channels/1189498204333543425/1198769713635917846/1307661784106143744) (3 messages):

> - `CUTLASS and Flash Attention 3`
> - `Column and Row Permutations in FA3`（FA3 中的列与行置换）
> - `Indexing Techniques in GPU Computing`（GPU 计算中的索引技术）

- **Jay Shah 讨论 PV 计算**：在[关于 CUTLASS 和 Flash Attention 3 的讲座](https://youtu.be/JwUcZwPOCpA?si=2AdtMNuLCvB0zeiB)中，Jay Shah 提到，与其执行 PV，不如使用 P 的列置换（column permutation）和 V 的行置换（row permutation）来避免 shuffle。
  
  - 这一表述引发了关于调优 FA3 kernel 的疑问，成员们寻求进一步澄清这些置换如何影响计算。
- **行列重排序见解**：一位成员建议，在行列重排序之后，会使用某些索引技术将其写入正确的位置。
  
  - 这引发了关于通过巧妙的索引来优化 GPU 计算的潜在方法的讨论。

 

**提到的链接**：[Lecture 36: CUTLASS and Flash Attention 3](https://youtu.be/JwUcZwPOCpA?si=2AdtMNuLCvB0zeiB)：演讲者：Jay Shah；幻灯片：[https://github.com/cuda-mode/lecturesCorrection](https://github.com/cuda-mode/lecturesCorrection) Jay 的更正：“事实证明，我在 intra-warpgroup overlapping 部分插入了错误的图片……”

 

---

### **GPU MODE ▷ #**[**off-topic**](https://discord.com/channels/1189498204333543425/1215328286503075953/1307857555447611533) (3 messages):

> - `Steamed Hams Meme`（Steamed Hams 迷因）
> - `Tasty Hacks`
> - `Hackathon Culture`（黑客松文化）
> - `Creative Projects`（创意项目）
> - `Community Engagement`（社区参与）

- **Steamed Hams 的乐趣**：*“在这个时间点，在每年的这个时候，在这个国家的这个地区？？”* 引发了一场以 Steamed Hams 短剧为中心的迷因讨论。
  
  - 这段对话展示了轻松的交流，强调了迷因如何渗透到现代对话中。
- **Tasty Hacks 旨在激发创意**：一个新的名为 [Tasty Hacks](https://lu.ma/s3pe30fz) 的黑客松正在筹备中，它专注于兴趣项目而非竞争，旨在促进纯粹的创意。
  
  - 该活动鼓励各行各业、各种技能水平的人士参加（**people from all walks of trades are welcome**），营造一个相互支持的社区。
- **对传统黑客松的批评**：有人批评当前的黑客松文化，即**黑客项目往往为了社交媒体发布而优先考虑赞助商奖项**，而非真正的创意。
  
  - *缺乏相关经验的评委*往往会设定误导性的标准，Tasty Hacks 的组织者希望改变这一现状。
- **社区驱动的黑客松体验**：Tasty Hacks 旨在创建一个拥有 **20-30 名参与者**的小型私密环境，以促进真实的互动与协作。
  
  - 没有团队的参与者可以通过策划流程进行匹配，从而增强志同道合者之间的社区参与感。

 

**提到的链接**：[tastyhacks '24 berkeley · Luma](https://lu.ma/s3pe30fz)：现在的许多黑客松都被名利所玷污。参与者为了获胜，在他们的作品中极简地加入赞助商奖项要求，随后……

 

---

### **GPU MODE ▷ #**[**triton-puzzles**](https://discord.com/channels/1189498204333543425/1219683012707487794/) (1 messages):

jongjyh: 谢谢兄弟！

---

### **GPU MODE ▷ #**[**rocm**](https://discord.com/channels/1189498204333543425/1233704710389764236/1307183874819883110) (5 messages):

> - `CK Profiler Results`（CK Profiler 结果）
> - `FP16 Matrix Multiplication Performance`（FP16 矩阵乘法性能）
> - `H100 vs MI300X`
> - `Async Copy with TMA`（使用 TMA 的异步拷贝）
> - `AMD Optimization Challenges`（AMD 优化挑战）

- **CK Profiler 显示出更好的结果**：在尝试 [CK Profiler](https://link.to.ckprofiler) 后，FP16 矩阵乘法的性能提升至 **600 TFLOPs**，但这仍低于 H100 白皮书中的峰值。
  
  - **H100** SXM5 的估计峰值为 **989.4 TFLOPs**，而使用 `torch.matmul(a,b)` 观察到的性能约为 **700 TFLOPs**。
- **不同 GPU 间的 FP16 性能差异**：**MI300X** 的 FP16 峰值性能为 **1307 TFLOPs**，但使用 `torch.matmul(a,b)` 仅达到 **470 TFLOPs**，尽管 CK 显示有所改进。
  
  - 这种差异凸显了在 FP16 计算中对 **AMD** 硬件采取更好优化策略的必要性。
- **H100 有效利用了 TMA**：据推测 **H100** 使用 **TMA** 进行异步拷贝（async copy），从而能够有效利用 cuBLAS/cuBLASLt 以获得最佳性能。
  
  - 相比之下，有人对 **AMD** 软件在优化 **ACE** 以达到类似效率方面面临的困难表示担忧。
- **考虑测试零分布输入**：为了提高 AMD 的性能，有人建议尝试使用**零分布输入（zero distribution input）**进行实验，这可能会产生更好的结果。
  
  - 这种方法可能会解决目前在 AMD 硬件上观察到的性能挑战。

 

---

### **GPU MODE ▷ #**[**liger-kernel**](https://discord.com/channels/1189498204333543425/1275130785933951039/) (1 messages):

0x000ff4: 好的，我该如何为这个项目做贡献，能指导我一下吗 🙂

---

### **GPU MODE ▷ #**[**self-promotion**](https://discord.com/channels/1189498204333543425/1288557096404516945/1307457755002437696) (4 messages):

> - `关于 quantization 的 YouTube 频道`
> - `X 平台上的 AI/ML 内容`
> - `关于 Tensor Core matmul kernel 的博客`

- **在 YouTube 上探索 quantization！**: 一位成员分享了他们的 [YouTube 频道](https://www.youtube.com/channel/UC-aLcQtTagZ4P76GLiOehOQ)，专注于教授 **neural network quantization**，教程涵盖了 Eager mode 和 FX Graph mode quantization。
  
  - 该频道旨在提供帮助解决 AI 挑战性问题的内容，强调编码技巧和理论概念。
- **在 X 上关注 AI/ML 见解！**: 另一位成员邀请大家关注他们的 [X 账号](https://x.com/Alpha7987)，获取与 **AI/ML、物理和数学论文**相关的**精彩内容**。
  
  - 这些内容旨在分享宝贵的见解，并让关注者了解这些领域的最新进展。
- **高效 Tensor Core matmul kernel 揭秘**: 分享了一篇博客文章，详细介绍了针对 Ada 架构的高效 **Tensor Core matmul kernel** 的实现，旨在匹配 **cuBLAS** 的性能。
  
  - 文章涵盖了 **CUTLASS permuted shared memory layout**、n-stage async memory pipelines 以及不同实现之间的性能对比结果。

**提到的链接**:

- [来自未定义的推文](https://x.com/Alpha7987): 未找到描述
- [Oscar Savolainen](https://www.youtube.com/channel/UC-aLcQtTagZ4P76GLiOehOQ): 我热爱创作关于 AI 的内容，特别是在编程方面，以帮助他人解决一些难题，如 neural network quantization 以及 LLM 和 Computer Vision 模型的部署...
- [在 Ada 架构上实现快速 Tensor Core matmul](https://www.spatters.ca/mma-matmul): 使用 Tensor Core 是在 Volta 之后的 NVIDIA GPU 上获得接近峰值性能矩阵乘法的先决条件。

---

### **GPU MODE ▷ #**[**🍿**](https://discord.com/channels/1189498204333543425/1298372518293274644/1307109234000793712) (23 messages🔥):

> - `Finetuning Loop 开发`
> - `Job Queue 访问`
> - `Discord 竞赛基础设施`
> - `训练数据源`
> - `Scheduler 开发`

- **Finetuning Loop 面临编译挑战**: 一位成员正在开发一个 Finetuning Loop，初始数据来自 [Popcorn-Eval](https://github.com/PaliC/Popcorn-Eval)，但结果令人失望，因为几乎没有任何内容能编译为 Python bytecode。
  
  - 他们邀请其他人提供帮助或提问，目前他们正专注于完善项目的基础设施。
- **请求 Job Queue 测试权限**: 一位成员询问了测试 Job Queue 的访问权限，正在等待特定联系人的确认。
  
  - 另一位成员鼓励大家参与到测试过程中。
- **Discord 作为竞赛基础设施**: 竞赛计划通过 Discord bot 进行提交并显示 leaderboard，利用熟悉 GPU coding 的社区资源。
  
  - 存在对 Discord 消息限制的担忧，建议未来可能需要一个 Web 界面以提高可用性和结果可见性。
- **多样化的训练数据源**: LLM 的训练数据来自竞赛提交、Triton docs 以及各种公共来源，尽管存在一些数据质量问题。
  
  - 成员们讨论了对大部分数据进行标注以提高其效用的必要性。
- **潜在的 Scheduler 开发计划**: 一位成员建议开发一个可以改进任务处理的 Scheduler，并讨论了目前对 GitHub Actions 的依赖，这给本地开发带来了一些挑战。
  
  - 另一位成员表示有兴趣赞助计算资源，表明了增强 kernel leaderboard 项目基础设施的意向。

---

### **GPU MODE ▷ #**[**thunderkittens**](https://discord.com/channels/1189498204333543425/1300872762163728550/1307080640294027345) (4 条消息):

> - `Template Parameter Inference`
> - `Register Limit in WGs`
> - `Register Spills in TK Programs`

- **Template Parameter Inference 变慢**：一位成员指出，缓慢的 **template parameter inference** 可能会导致延迟，特别是在展开的循环（unrolled loop）中进行时，建议将其隔离并尽可能使参数显式化。
  
  - 如果能提供相关的代码片段，他们愿意进一步提供帮助。
- **关于 WG Register Limit 的疑问**：有人提出了关于所有 workgroups (WGs) 的 **increase_register** 限制为 **480** 的原因。
  
  - 这一询问表明了对该限制如何影响工作流性能的好奇。
- **Register Spills 与优化担忧**：有成员表达了对编译 TK 程序时出现 **register spills** 的担忧，即使尝试使用了 `warpgroup::increase_registers<112>();`（这本应符合 **480** 的限制）。
  
  - 该成员寻求关于优化寄存器使用以及理解 spills 所造成危害的建议。
- **Spills 对性能的影响**：一位参与者确认 **spills 非常有害**，并强调了为 consumers 增加寄存器数量的重要性。
  
  - 他们警告说，在 kernel 内部进行精细的寄存器分配对于缓解性能问题至关重要。

 

---

### **GPU MODE ▷ #**[**edge**](https://discord.com/channels/1189498204333543425/1303441437592911912/1307482442629648506) (4 条消息):

> - `Memory Bound in High Context Generation`
> - `Neuron Culling in Small Language Models`
> - `Speculative Decoding Challenges`

- **高上下文生成是否为 Memory Bound 的辩论**：一位成员认为，由于 *kv cache* 的存在，**token generation** 始终会是 memory bound，同时强调了 prefill 阶段对 prompt 的 token 数量的依赖。
  
  - 另一位成员反驳道，在朴素计算中，**高上下文（high context）** 通常更多是 **compute bound**，并认为 KV caching 影响了这种平衡。
- **寻求 Neuron Culling 相关资源**：一位成员表示有兴趣阅读更多关于小语言模型中 **neuron culling** 的内容，并指出目前大多数研究都集中在大语言模型上。
  
  - 虽然没有提供具体的资源，但有人提到在各种论文中听过一些**随口提及的评论**，暗示了稀疏化（sparsification）技术的有效性。
- **移动端 Speculative Decoding 的挑战**：一位成员强调，执行 **speculative decoding** 需要同时运行一个小型的 draft model 和一个大型的 verification model。
  
  - 他们指出，这在手机等**资源受限的设备**上带来了困难，使实现变得复杂。

 

---

### **Notebook LM Discord ▷ #**[**use-cases**](https://discord.com/channels/1124402182171672732/1124403655819415592/1307086093719441409) (33 条消息🔥):

> - `NotebookLM 实验`
> - `使用 NotebookLM 创作音频`
> - `小组发言人简报`
> - `支出分析的使用案例`
> - `关于 NotebookLM 的反馈`

- **NotebookLM 是内容创作的变革者**：前 Google CEO **Eric Schmidt** 将 NotebookLM 称为他 *“今年的 ChatGPT 时刻”*，并强调了 YouTuber 创作者和播客主如何在媒体制作中利用该工具。
  
  - 他在 [YouTube 视频](https://youtu.be/2Zg--ouGl7c?si=YwkVUqxCqsKB66dP&t=4153) 中分享了关于如何有效使用 NotebookLM 进行 *创意内容生成* 的见解。
- **学术研究的深度分析**：一位博士生分享了使用 NotebookLM 提供关于学术材料的 **30-60 分钟** 概览，从而实现对不同作者和观点的批判性视角综合。
  
  - 这一过程有助于理解特定研究领域内 *概念的演变*。
- **使用 NotebookLM 的创意音频项目**：NotebookLM 被用于为小组发言人创建 **定制化音频简报**，根据各种源材料汇编个人建议。
  
  - 另一位成员针对涂鸦主题制作了不同的音频解读，突出了基于相同信息的多元视角。
- **关于 NotebookLM 易用性的反馈**：用户分享了对 **移动端界面** 的褒贬不一的体验，指出了其独特的功能，但也提到了设计上的困难和局限性。
  
  - 其他人讨论了探索 NotebookLM 在组织个人见解和信息综合方面的功能。
- **令人印象深刻的 RPG 角色创建**：一位成员报告称使用 NotebookLM 为他们的 **Savage Worlds RPG** 生成了背景和角色，且在不到五分钟内完成。
  
  - 他们称赞其在即兴游戏环节中的易用性，展示了 NotebookLM 在 *创意叙事* 方面的高效。

**提到的链接**：

- [The Deep Dive](https://open.spotify.com/show/26RTNstGpNDBZFeg77XM9K?si=feb86abdeb254564)：播客 · Gary Smith · 深入探讨各种感兴趣的话题。
- [未找到标题](https://notebooklm.google.com/notebook/bd270341-2453-4b6d-9a87-4478f608ff5d/audio)：未找到描述
- [前 Google CEO：AI 正在创造致命病毒！如果我们看到这个，必须关闭 AI！](https://youtu.be/2Zg--ouGl7c?si=YwkVUqxCqsKB66dP&t=4153)：Eric Schmidt 是 Google 前 CEO 兼 Schmidt Sciences 联合创始人。他也是《新数字时代》等畅销书的作者。
- [深入观察：NotebookLM 对 Token Wisdom 的深度挖掘 ✨](https://tokenwisdom-and-notebooklm.captivate.fm)：通过 Token Wisdom 关于 AI、数字隐私和不断发展的创新格局的见解，探索技术与创意的交汇点。
- [Ethan Mollick (@emollick) 的推文](https://x.com/emollick/status/1857647589178462234)：我通过给 NotebookLM 一本 308 页的手册，让它玩起了一场角色扮演游戏。它对规则的应用非常好，角色创建也非常出色（能准确引用 100 页后的内容）……
- [为 Advertising and Antiheroes TTRPG 进行的 NotebookLM 角色创建](https://youtu.be/z12Ymd-DTOc?si=LsQQQmi0w0VM67_W&t=55)：探索使用 NotebookLM 为我的独立 TTRPG 《Advertising and Antiheroes》创建角色和困境，灵感来自 Ethan Mollick 的实验……

---

### **Notebook LM Discord ▷ #**[**general**](https://discord.com/channels/1124402182171672732/1124402182909857966/1307092707889385483) (90 条消息🔥🔥):

> - `NotebookLM 问题`
> - `功能请求`
> - `将 NotebookLM 用于游戏`
> - `音频文件相关问题`
> - `与外部来源集成`

- **用户面临 NotebookLM 功能问题**：许多用户报告了诸如无法在移动端滚动长笔记、生成音频文件出现问题，以及因上传源出现多个副本而感到困惑等困难。
  
  - 某些用户在不同设备上访问 NotebookLM 时也遇到了问题，讨论强调了对移动端兼容性的需求。
- **功能请求与增强**：几位成员请求了诸如连接 RSS feeds 以集成外部信息，以及无需额外应用程序即可轻松自定义语音设置的功能。
  
  - 还有人呼吁改进对各种文件类型的支持，并对上传 XLS 或图片等格式表示了挫败感。
- **将 NotebookLM 用于 RPG**：用户指出 NotebookLM 已被有效地用于 RPG，能够实现引人入胜的角色创建和冒险构建，并呼吁进一步开发以支持游戏功能。
  
  - 一位知名用户表示有兴趣提供反馈，并参与测试新的游戏相关功能。
- **音频文件管理问题**：有人对无法为不同声音生成独立的音轨，或者下载时音频文件命名错误的问题表示担忧。
  
  - 用户考虑使用数字音频工作站（DAW）从合并的音频文件中分离声音的变通方法，并讨论了噪声门（noise gate）技术的潜力。
- **访问与可用性问题**：出现了一些关于访问规则的问题，例如新用户的年龄限制，或者在之前的尝试后创建新笔记本时遇到的困难。
  
  - 大家普遍认为界面导航可能令人困惑，尤其是在尝试删除或重启笔记本时。

**提到的链接**：

- [未找到标题](https://docs.aws.amazon.com/)：未找到描述
- [未找到标题](https://notebooklm.google.com/notebook/37f6f9a6-1cbe-4985-b56c-bcd4fffaeade/audio)：未找到描述
- [未找到标题](https://notebooklm.google.com/?hl=EN)：未找到描述
- [未找到标题](https://cloud.google.com/text-to-speech/docs/create-dialogue-with-multispeakers#example_of_how_to_use_multi-speaker_markup)：未找到描述
- [Shared Non-locality [Cycles] - Google Drive](https://drive.google.com/drive/folders/1-3tuqM9ItzZVvPbb_7mFYT_w_cx-VKDW)：未找到描述
- [来自 Steven Johnson (@stevenbjohnson) 的推文](https://x.com/stevenbjohnson/status/1858199993313804370)：我们最喜欢的 NotebookLM 意外用途之一是 RPG 玩家和 DM 将其用于游戏。（尽管让 @emollick 弄清楚了如何用音频来实现这一点！）我们正在开发一个...

---

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1307667321841778729) (51 messages🔥):

> - `Mojo Benchmarking`
> - `Handling Pstate Driver Issues`
> - `Dict Struct Bugs in Mojo`
> - `AstNode Struct Implementation`
> - `CPU Frequency Error Handling`

- **使用随机参数进行 Mojo Benchmarking**：一位用户寻求关于在 Mojo 中使用随机函数参数进行 Benchmarking 的建议，但指出当前方法需要静态参数，这增加了不必要的开销。
  
  - 另一位用户建议预先生成数据并在 closure 中使用，以避免 Benchmarking 期间的开销。
- **处理 WSL2 上的 Pstate 驱动问题**：一位用户遇到了可能与 WSL2 上缺失 CPU pstate 驱动相关的 Mojo 崩溃，这表明了对 Mojo runtime 处理此问题的担忧。
  
  - 讨论显示 WSL 环境可能没有暴露 CPU 频率调节所需的接口，强调了需要 Microsoft 或 Modular 提供修复。
- **Dict 实现中发现的 Bug**：一位用户报告了在 Mojo 中将 Dict 与 SIMD 类型一起使用时发生的崩溃，在 SIMD 大小为 8 时正常工作，但超过该大小则失败。
  
  - 该问题在一个最小示例中被复现，表明 Dict 实现中存在深层问题，值得关注。
- **AstNode 的 Struct 实现问题**：另一位用户解决了在 Mojo 中为图节点实现通用 struct 的相关问题，遇到了指向 trait 类型的指针类型转换错误。
  
  - 建议包括使用 Variant 类型来涵盖多种节点语句类型，尽管在管理类型行为方面仍存在挑战。
- **Mojo 中的 CPU 频率错误处理**：一位用户注意到在运行 Mojo 时遇到关于不可用 CPU 频率文件的错误，引发了关于优雅错误处理的讨论。
  
  - 参与者强调这些错误应被视为 warning，并指出当前的 Mojo 实现需要调整以考虑不存在的接口。

**Links mentioned**:

- [no title found](https://docs.modular.]): no description found
- [Variant | Modular Docs](https://docs.modular.com/mojo/stdlib/utils/variant/Variant): A runtime-variant type.
- [Issues · modularml/mojo](https://github.com/modularml/mojo/issues): The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.

---

### **Modular (Mojo 🔥) ▷ #**[**max**](https://discord.com/channels/1087530497313357884/1212827597323509870/1308183326992896121) (10 messages🔥):

> - `Max Graphs and Knowledge Graphs`
> - `Using MAX for Graph Searches`
> - `Feasibility of LLM Inference with MAX`
> - `Pregel Integration with MAX`
> - `Memory Requirements for Encoding Graphs`

- **探索 Max Graphs 以进行 Knowledge Graph 集成**：一位成员思考 **Max Graphs** 是否能有效统一 LLM 推理与常规 **Knowledge Graphs**，并提到它们在 **RAG** 工具和 **NeuroSymbolic AI** 中的潜在用途。
  
  - 他们提供了一个 [GitHub link](https://github.com/microsoft/graphrag)，展示了这种方法的概念验证。
- **MAX 在加速图搜索中的作用**：一位成员询问使用 **MAX** 是否有助于加速图搜索，另一位成员确认了其潜力，但也指出了局限性。
  
  - 澄清指出，除非将整个图复制到 **MAX** 中，否则当前的能力是有限的。
- **Mojo 和 MAX 实现的可行性担忧**：针对在 **Mojo** 和 **MAX** 中实现一个推断 LLM 以执行搜索的 Agent 的可行性提出了担忧。
  
  - 这个想法遭到了质疑，成员们讨论了其在实际应用中的实用性。
- **Pregel 与 MAX 的兼容性**：一位成员建议探索 **Pregel** 与 **MAX** 的结合，并以 **langgraph** 为例。
  
  - 然而，有人指出，如果不复制整个图，目前的集成是不可能的。
- **图编码与内存影响**：一位成员建议将图编码为 **1D byte tensors** 以实现 zero-copy 功能，尽管内存成本会很高。
  
  - 他们指出 **MAX** 目前不处理指针，这使得直接数据操作变得复杂。

 

**Link mentioned**: [GitHub - microsoft/graphrag: A modular graph-based Retrieval-Augmented Generation (RAG) system](https://github.com/microsoft/graphrag): A modular graph-based Retrieval-Augmented Generation (RAG) system - microsoft/graphrag

 

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1307152237822214205) (41 条消息🔥):

> - `Cohere 模型输出问题`
> - `不同 k 值的用户体验`
> - `文字冒险游戏用例挑战`
> - `对 Cohere 性能的反应`
> - `Office Hours 主题文档`

- **用户报告 Cohere 输出异常**：用户对离奇的输出表示沮丧，特别是 Cohere 模型在处理较短文本时出现错误并生成奇怪的术语。
  
  - 性能似乎不稳定，导致一些人质疑 Cohere 模型在某些应用中的可靠性。
- **调整 k 值引发关注**：一位用户强调低 k 值会影响模型的 Token 输出，建议将其增加到 400 以提高性能。
  
  - 尽管进行了调整，模型仍存在未按预期响应的问题。
- **在 Cohere 中进行文字冒险游戏的挑战**：一名成员详细介绍了使用 Cohere 模型进行文字冒险游戏的情况，指出在游戏过程中存在顺序文本交付失败和输出不稳定的问题。
  
  - 成员分享了错误和正确操作的示例，展示了模型不可预测的行为。
- **对 Cohere 的关注点和性能变化的反应褒贬不一**：用户对最近版本的有效性下降表示担忧，特别是对于色情角色扮演（ERP），尽管该模型的主要焦点是商业应用。
  
  - 如果最近的模型继续表现不佳，用户希望能有替代方案。
- **请求 Office Hours 主题的文档**：一名成员询问是否有关于 Office Hours 中讨论的 Compression（压缩）和 Summarization（摘要）的在线文档。
  
  - 另一名成员开玩笑说要通过刷屏来确保参加未来的会议。

**提到的链接**：

- [imgur.com](https://imgur.com/a/0MTqU7p)：在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过幽默的笑话、热门迷因、有趣的 GIF、鼓舞人心的故事、病毒式传播的视频来振奋你的精神...
- [imgur.com](https://imgur.com/a/2wLY2IJ)：在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过幽默的笑话、热门迷因、有趣的 GIF、鼓舞人心的故事、病毒式传播的视频来振奋你的精神...
- [Correct operation!](https://imgur.com/a/NYfs8Ri)：在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过幽默的笑话、热门迷因、有趣的 GIF、鼓舞人心的故事、病毒式传播的视频来振奋你的精神...

---

### **Cohere ▷ #**[**announcements**](https://discord.com/channels/954421988141711382/996880279224451154/1308076192036950036) (1 条消息):

> - `Cohere 开发者 Office Hours`
> - `长文本策略`
> - `RAG 系统中的记忆`
> - `压缩与摘要`
> - `用例讨论`

- **关于长文本策略的 Cohere 开发者 Office Hours**：欢迎参加今天 **ET 时间中午 12:00** 在 Discord Stage 举行的会议，重点讨论如何处理 **长文本** 和真实世界的用例，包括实时故障排除和实用技巧。
  
  - Cohere 的 RAG 高级产品经理 **Maxime Voisin** 将出席会议，分享见解并解决您在处理长文本方面的特定需求。
- **在 RAG 系统中实现记忆**：会议将涵盖在 RAG 流水线中实现 **Memory（记忆）或 Caching（缓存）** 系统以管理长文本内容的技巧。
  
  - 鼓励参与者带着自己的用例来现场讨论策略和解决方案。
- **长文本摘要技术**：参与者将学习在保持关键信息的同时对长文本进行 **压缩和摘要** 的技术。
  
  - 这包括适用于各种项目的实际应用和方法。
- **用例聚焦：文件上传和 SQL 查询**：将重点介绍特定的用例，包括 **文件上传** 和 **SQL 查询生成**，促进对定制策略的讨论。
  
  - 鼓励成员在会议期间积极分享与这些用例相关的挑战。

### **Cohere ▷ #**[**questions**](https://discord.com/channels/954421988141711382/1168411509542637578/1307465100541558814) (4 条消息):

> - `Cohere API 问题`
> - `Playwright 与文件上传`
> - `API 响应中的异常 Token`
> - `关闭 Citations`

- **Cohere 与 Playwright 在文件上传方面遇到困难**：一名成员提到他们可以通过 **Playwright** 向 **Cohere** 上传文件，但在针对该文本提问时遇到问题，收到的响应建议他们改为直接粘贴文本。
  
  - 这一限制引发了关于如何有效利用上传功能进行查询的问题。
- **V1 API 面临异常 Token 问题**：有报告称在使用 **v1 API** 配合 **08-2024 Command R+ 模型**和网页搜索连接器（web search connector）时，会出现类似 '<co: 0,1,2,3,4,5,' 的异常 Token。
  
  - 对此，一名成员建议这可能与在请求网页搜索和 **Citations** 的同时请求 **JSON response** 有关。
- **响应中的 Citations 导致问题**：一位用户表示他们通过**关闭 Citations** 解决了 Token 问题，但希望能有更完善的解决方案。
  
  - 这引发了关于在网页搜索查询语境下，如何整合 **Citations** 与 **JSON response** 的担忧。

---

### **Cohere ▷ #**[**api-discussions**](https://discord.com/channels/954421988141711382/1168578329423642786/1307119659128066059) (1 条消息):

> - `API 错误`
> - `服务不可用问题`

- **遇到 API 503 错误**：一名用户报告在 **2024-11-15** 收到 **503 Service Unavailable** 错误，表明 API 存在上游连接问题。
  
  - *错误详情包括* 'upstream connect error or disconnect/reset before headers'，并提到可能存在远程连接失败。
- **寻求 API 错误帮助**：该用户询问是否还有其他人遇到过这种特定的 API 错误。
  
  - 这突显了用户在 **API 稳定性**方面可能面临的共同挑战。

---

### **Cohere ▷ #**[**cohere-toolkit**](https://discord.com/channels/954421988141711382/1254901651081269268/1308099324881797140) (3 条消息):

> - `Toolkit 版本 v1.1.3 发布`
> - `Toolkit 新功能`
> - `开发体验优化`

- **Toolkit v1.1.3 正式发布！**：[Release 2024-11-18 (v1.1.3)](https://github.com/cohere-ai/cohere-toolkit/releases/tag/v1.1.3) 引入了一系列变更，包括改进全局 Settings 的使用以及重大的工具重构。
  
  - 关键修改包括重命名工具 Schema，并建议通过 **make reset-db** 和 **make migrate** 更新分支。
- **新增令人兴奋的功能！**：最新更新增加了对 **ICS 文件**的支持、**文件内容查看器**，以及为自定义 **Assistants** 提供的工具切换功能，感谢 **Danylo** 的贡献。
  
  - 这些增强功能旨在提升 Toolkit 的功能性和易用性。
- **错误修复与维护更新**：此版本修复了多个 Bug，包括 **Slack** 认证工具的问题，并简化了 Deployments 和 Model DB 的表示。
  
  - 通过移除添加工具的步骤并改进整体维护，提升了 Toolkit 的可用性。
- **展望未来：即将推出的新功能**：未来的功能包括**热键绑定按钮**、与 **Gmail + SSO** 的集成，以及针对 Pull Request 增强的集成测试。
  
  - 此外，对部署方式的全面改革以及对 **Azure deployment with Docker compose** 的支持也已在计划中。
- **开发体验增强**：未来的周期将优先改进围绕构建流程、运行、调试以及在 Toolkit 上进行更改的开发体验。
  
  - 欢迎提供反馈和改进建议，以确保 Toolkit 满足开发者的需求。

**提到的链接**：[Release 2024-11-18 (v1.1.3) · cohere-ai/cohere-toolkit](https://github.com/cohere-ai/cohere-toolkit/releases/tag/v1.1.3)：变更内容包括改进全局 Settings 使用以处理未设置的配置；重大工具重构：明确工具 Schema 名称（例如 ManagedTool -> ToolDefinition, ToolName -> Tool...）

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1307095692011114506) (4 条消息):

> - `文档中的 Ask AI 组件`
> - `多媒体研究报告生成器`
> - `结构化财务报告生成`
> - `Mistral 多模态图像模型发布`

- **文档通过 Ask AI 组件得到提升**：我们的 Python 文档通过全新的 **'Ask AI' 组件**进行了升级，使用户能够通过 **RAG 系统**提出问题并获取准确、最新的代码。点击[此处](https://t.co/Smy98h3Med)查看。
  
  - *这是一个非常神奇且准确的功能，提升了编程体验！*
- **通过多媒体洞察创建报告**：我们发布了一个新视频，展示了如何生成**多媒体研究报告**，总结来自幻灯片等复杂文档的洞察。点击[此处](https://t.co/zPz7AZ5S7L)观看。
  
  - *该工具将文本和视觉效果交错排列，简化了报告生成。*
- **轻松生成结构化财务报告**：我们最新的视频演示了如何使用跨 10K 文档的 **multi-agent** 工作流生成结构化**财务报告**。点击[此处](https://t.co/XKzCUxC8rS)了解详细流程。
  
  - *这使得包含文本和表格的简化分析成为可能。*
- **Mistral 发布先进的多模态图像模型**：Mistral 推出了一款全新的尖端**多模态图像模型**，早期采用者可以通过 `pip install llama-index-multi-modal-llms-mistralai` 进行安装，享受 day 0 支持。在提供的 [notebook](https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/multi_modal/mistral_multi_modal.ipynb) 中探索其用法。
  
  - *该模型支持 `complete` 和 `stream complete` 等多种函数，以实现高效的图像理解。*

 

**提到的链接**：[使用 Mistral 进行图像推理的多模态 LLM - LlamaIndex](https://t.co/hSIZOz1Njy)：未找到描述

 

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1307072966613405706) (36 条消息🔥):

> - `condenseQuestionChatEngine`
> - `CitationQueryEngine`
> - `CSV 数据处理`
> - `EY Techathon 组队`
> - `区块链开发协作`

- **condenseQuestionChatEngine 的挑战**：一位用户报告了 [condenseQuestionChatEngine](https://link.to.condense) 在话题突然切换时生成无意义独立问题的问题。
  
  - 建议包括自定义 condense prompt，以及考虑使用能为用户消息检索上下文的 [CondensePlusContext](https://link.to.condenseplus)。
- **使用 CitationQueryEngine 进行源引用**：一位用户询问如何将其 UI 中的引用与 `CitationQueryEngine` 的响应链接起来，因为尽管有多个来源，响应中仅提供了一个引用编号。
  
  - 建议通过解析响应文本，实现一个将引用编号映射到其各自源节点的解决方案。
- **改进 CSV 数据 embeddings**：一位用户询问生成 CSV 文件 embeddings 的高效方法，并指出 [PagedCSVReader](https://link.to.pagedcsvreader) 为每一行创建一个 embedding。
  
  - 另一位用户提到，将 CSV 转换为 Markdown 可以在保持 embedding 完整性的同时提高性能。
- **EY Techathon 团队招募**：一位用户宣布他们正在为 EY Techathon 组建团队，寻找一名 AI 开发者和一名 Web 应用开发者，敦促感兴趣的人员尽快私信。
  
  - 随后收到提醒，要求将此类机会发布在适当的职位发布频道中。
- **Property Graph 示例的问题**：一位尝试复制 Property Graph 示例的用户报告称，所有节点都显示为 `__NODE__`，并收到了关于未知标签的警告。
  
  - 他们在从 notebook 执行查询时寻求解决这些问题的帮助。

 

**提到的链接**：[MLflow LlamaIndex Flavor](https://mlflow.org/docs/latest/llms/llama-index/index.html)：未找到描述

 

---

### **LlamaIndex ▷ #**[**ai-discussion**](https://discord.com/channels/1059199217496772688/1100478495295017063/1307407463187873903) (1 messages):

> - `EY Techathon 团队`
> - `AI 开发者职位`
> - `Web 应用开发者职位`

- **为 EY Techathon 组建团队**：该团队目前正在为 **EY Techathon** 招人，现有**两个名额**：一个是 **AI 开发者**，另一个是 **Web 应用开发者**。
  
  - 有意向的候选人请**尽快私信 (DM)** 以锁定团队名额。
- **紧急招募 AI 开发者**：消息强调迫切需要一名 **AI 开发者** 来填补参加 **EY Techathon** 团队的空缺名额。
  
  - 潜在申请者应迅速行动并直接联系以参与。
- **Web 应用开发者名额开放**：同时也在招募一名 **Web 应用开发者** 加入 EY Techathon 团队，强调了对 Web 应用开发技能的需求。
  
  - 对该职位感兴趣的候选人也应立即通过私信联系。

 

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1307076942075396238) (20 messages🔥):

> - `Liger Kernel 性能`
> - `DPO 实现反馈`
> - `Web3 职位列表`
> - `模型优化请求`

- **Liger Kernel 运行速度快 3 倍**：**Liger** 声称在最坏情况下使用相同内存的情况下，运行速度比之前的模型快约 3 倍，且安装过程中未报告任何错误。
  
  - 一些成员表示怀疑，想知道这种性能是否只能在 **NVIDIA** 硬件上实现。
- **对 DPO 实现的担忧**：成员们讨论了合并后的 **DPO** 实现中的一个问题，强调它没有按预期利用 **reference model logprobs**。
  
  - 一位成员指出，在不使用参考模型时，它可能仍然是准确的。
- **Web3 团队招聘公告**：一个 **Web3** 平台正在积极招聘多个职位，提供具有竞争力的薪资，职位涵盖从开发者到 Beta 测试员。
  
  - 他们强调了友好的工作环境，且对申请者没有经验要求。
- **请求模型运行协助**：一位用户表示需要一个易于使用的流水线在 **RunPod** 或 **HF Space** 上运行模型，希望能简化因网络缓慢而导致的资源密集型过程。
  
  - 他们提到了通过一次加载一层来更好地管理 **RAM**，从而优化性能的可能性。

 

**提及的链接**：[Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1gtrtfw/beer_money_ad_make_a_hf_space_runpod_template_for/): 未找到描述

 

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**other-llms**](https://discord.com/channels/1104757954588196865/1104758057449308220/1307679264753061929) (2 messages):

> - `AnyModal Framework`
> - `Chai Research Grants`
> - `Generative AI`
> - `Open-source Projects`
> - `Community-driven AGI`

- **AnyModal 框架集成多模态数据**：讨论重点介绍了 [AnyModal](https://github.com/ritabratamaiti/AnyModal) 的开发，这是一个允许将图像和音频等数据类型与 LLM 集成的框架，简化了 LaTeX OCR 和图像字幕（image captioning）等任务的设置。
  
  - 正在寻求反馈和贡献以增强这一进行中的工作，展示了如 ViT 等用于视觉输入的模型。
- **Chai Research 宣布开源资助计划**：Chai 是一家拥有 **1.3M DAU** 的生成式 AI 初创公司，正为专注于加速社区驱动 AGI 的开源项目提供无限额资助，目前已向 **11 位个人**发放了资助。
  
  - 资助金额根据创新性和影响力分为 **$500 到 $5,000** 不等，邀请开发者通过 [Chai Grant](https://www.chaigrant.com/) 提交他们的项目。
- **Chai 在 AI 内容创作中的使命**：Chai 旨在平衡 AI 的事实准确性和娱乐性，专注于赋能创作者构建和分享 AI 内容，并由一支前量化交易员团队提供支持。
  
  - 他们的工作涉及对 **long-context**、**LoRA** 和 **RLHF** 的实验，以使 AI 行为与创作者意图保持一致。
- **展示近期资助获得者**：近期的资助对象包括 [Medusa](https://github.com/FasterDecoding/Medusa) 和 [Streaming LLM](https://github.com/mit-han-lab/streaming-llm) 等项目，突出了在 LLM 生成加速方面的创新。
  
  - 每个项目都旨在推向 LLM 能力的边界，同时为 AI 社区的发展做出贡献。
- **邀请合作并提交创意**：Chai 鼓励向其资助计划提交开源项目，并强调申请流程非常简便，可在 **2 分钟**内完成。
  
  - 提醒参与者包含推荐邮箱以获取潜在收益，促进开发者之间的协作。

**提到的链接**：

- [GitHub - ritabratamaiti/AnyModal: AnyModal is a Flexible Multimodal Language Model Framework](https://github.com/ritabratamaiti/AnyModal)：AnyModal 是一个灵活的多模态语言模型框架 - ritabratamaiti/AnyModal
- [CHAI](https://www.chai-research.com/)：CHAI 是一家位于 Palo Alto 的 AI 公司。我们正在构建领先的对话式生成式人工智能平台。
- [Chai Grant](https://www.chaigrant.com:)：CHAI AI 资助。为任何开源项目提供 $1,000 - $5,000 的现金奖励。加速社区驱动的 AGI。

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general-help**](https://discord.com/channels/1104757954588196865/1110594519226925137/1307805560749559821) (1 messages):

> - `vLLM analytics`
> - `Token usage inspection`

- **寻找 vLLM 分析平台**：一位成员询问是否有与 **vLLM** 集成的平台，能够提供 **token 使用情况**的分析并允许检查响应内容。
  
  - 该请求突出了社区对增强 vLLM 性能理解和监控工具的兴趣。
- **社区支持 vLLM 查询**：该查询促使成员们考虑与 **vLLM** 集成相关的潜在解决方案或类似经验。
  
  - 鼓励成员分享见解或推荐能够满足这些分析需求的平台。

 

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**datasets**](https://discord.com/channels/1104757954588196865/1112023441386778704/1307438267863404556) (3 messages):

> - `Pretraining with Instruction-Based Datasets`
> - `Mathematical Sequence Problems`
> - `Code Availability for Instruction Datasets`

- **探索用于预训练的指令数据集**：一位用户分享了一个 [指令预训练数据集](https://huggingface.co/datasets/instruction-pretrain/ft-instruction-synthesizer-collection?row=1) 的链接，并提出了一个关于识别序列中缺失数字的问题：4, 10, X, 46, 和 94。
  
  - 他们详细说明了序列的属性，并根据数值递增的模式得出结论：**X 是 22**。
- **关于代码可用性的查询**：另一位用户询问是否有 duh_kola 提到的数据集相关的代码。
  
  - 一位成员回答说，目前**只有一篇论文**与该主题相关，表明缺乏代码资源。

 

**提到的链接**：[instruction-pretrain/ft-instruction-synthesizer-collection · Datasets at Hugging Face](https://huggingface.co/datasets/instruction-pretrain/ft-instruction-synthesizer-collection?row=1)：未找到描述

 

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**axolotl-help-bot**](https://discord.com/channels/1104757954588196865/1225300056442409040/1307089440707186769) (9 messages🔥):

> - `Pretraining and Finetuning Qwen/Qwen2`
> - `Phorm Bot Issues`
> - `Understanding eval_steps`

- **预训练 Qwen/Qwen2 并使用 Axolotl Docker**：一位成员询问了使用 QLoRA 和他们的**预训练数据集**先对 **Qwen/Qwen2 模型进行预训练**，然后使用 Alpaca 格式的**指令数据集 (instruct dataset)** 进行微调所需的步骤。
  
  - 他们确认已经为该流程准备好了 **Axolotl Docker**。
- **Phorm Bot 故障**：一位成员报告了 **Phorm bot** 的问题，称其甚至无法回答简单的问题。
  
  - 这引发了社区对该 Bot 可靠性的担忧。
- **需要澄清 Evaluation Steps**：另一位用户询问了在模型训练背景下 **eval_steps** 的具体含义。
  
  - 然而，Phorm bot 针对此查询未能提供任何答案。

 

**提到的链接**：[OpenAccess-AI-Collective/axolotl | Phorm AI Code Search](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=undefined)：更快地理解代码。

 

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**axolotl-phorm-bot**](https://discord.com/channels/1104757954588196865/1225558824501510164/1307794773792395336) (6 messages):

> - `eval_steps inquiry`
> - `Phorm response issues`

- **用户寻求 eval_steps 的澄清**：一位用户询问了 **eval_steps** 的含义，寻求一个直接的定义。
  
  - 然而，自动化系统的响应明显为 **undefined**（未定义），导致了进一步的挫败感。
- **用户对 Phorm 的能力表示失望**：一位用户对 Phorm 无法回答关于 eval_steps 这样一个简单问题感到**羞愧 (shame)**。
  
  - 这促使另一位用户建议向名为 **caseus** 的成员寻求帮助以获取清晰的解答。

 

**提到的链接**：[OpenAccess-AI-Collective/axolotl | Phorm AI Code Search](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=undefined)：更快地理解代码。

 

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1307133205035356181) (26 messages🔥):

> - `Tinygrad Contributions`
> - `Release Schedule`
> - `Alias Implementation`
> - `Int64 Indexing Bounty`
> - `Graph and Buffer Improvements`

- **Tinygrad 贡献需要高质量**：George 指出，对 Tinygrad 的贡献应达到**高质量标准**，并强调低质量的 PR 将在不予置评的情况下被关闭。
  
  - 他鼓励潜在的贡献者阅读之前已合并的 PR，并解释说许多处理 Bounty（悬赏任务）的人理想情况下应该先有之前合并过的记录。
- **Tinygrad 即将发布新版本**：Tinygrad 计划在大约 **15 小时**内发布新版本，目前正在讨论各种持续的改进和更新。
  
  - 发布的关键点包括 blocks、lazy buffers 的进展，以及 Qualcomm 调度的性能增强。
- **关于 Tinygrad 便捷方法的讨论**：一位用户提到需要像 `scatter_add_` 和 `xavier_uniform` 这样的便捷方法，认为这可以帮助用户避免重复编写代码。
  
  - George 同意如果这些方法与现有功能一致，可以从 Torch 和 TensorFlow 等框架中合并它们。
- **澄清 Int64 索引 Bounty**：一位成员对 Int64 索引 Bounty 表示困惑，提到它看起来功能正常，但后来判定它们可能会导致崩溃。
  
  - 这引发了关于改进文档并确保 Bounty 要求的功能清晰度的讨论。
- **Graph 和 Buffer 管理的改进**：成员们讨论了在 Tinygrad 中完善 **Big Graph** 和 LazyBuffer 概念的持续努力，并计划删除 LazyBuffer。
  
  - 旨在简化处理过程并跟踪 **UOp Buffers**，从而在项目中实现更好的性能和功能。

**提到的链接**：

- [来自 the tiny corp (@__tinygrad__) 的推文](https://x.com/__tinygrad__/status/1857977431845724621)：pip install tinygrad 引用 Jeremy Howard (@jeremyphoward) 的话：噢哇。现在要弄清楚该怎么办会非常棘手。目前没有任何简单的自动化方法可以安装完整的深度学习系统...
- [torch.nn.init — PyTorch 2.5 文档](https://pytorch.org/docs/stable/_modules/torch/nn/init.html#xavier_uniform_)：未找到描述
- [删除 LazyBuffer，一切皆为 UOp · Issue #7697 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/issues/7697)：更小的 Big Graph - 将 BUFFER 的 VIEW 重写为 LOAD/PRELOAD #7731 #7732 #7695 使用 WeakKeyDictionary 跟踪 UOp Buffers #7742 #7745 #7746 #7743 #7766 #7767 实现 UOp 上缺失的方法...

---

### **tinygrad (George Hotz) ▷ #**[**learn-tinygrad**](https://discord.com/channels/1068976834382925865/1070745817025106080/1307756655814901760) (4 messages):

> - `M1 Mac 上的 LSTM 训练`
> - `PyTorch 的 AMD GPU 问题`
> - `针对 AMD 的 TinyGrad`
> - `TinyNet 训练示例`
> - `JIT 编译问题`

- **在 M1 Mac 上训练 LSTM 网络**：一位用户称赞了 **PyTorch 在其 M1 Mac 上的性能**，指出其在训练 LSTM 网络时表现出色。
  
  - 他们将此体验与在 Ubuntu 上使用 AMD GPU 的困难进行了对比，在后者中，他们无法让 PyTorch 识别设备。
- **用于 AMD GPU 训练的 TinyGrad**：有人询问导入 **TinyGrad** 是否可以免去为了在 AMD GPU 上训练而安装 **ROCm** 的麻烦。
  
  - 引用提到了 George 的直播，其中提到他“剥离了 AMD 用户空间（userspace）”。
- **使用 TinyNet 训练 MNIST 分类器**：一位用户分享了使用 **TinyNet** 框架训练 **MNIST Classifier** 的 Python 代码，其中包含 **Adam optimizer** 和 dropout 的具体实现。
  
  - 他们指出在没有 JIT 编译的情况下训练成功，但在 M3 Max MacBook 上应用 JIT 时遇到了挂起问题。
- **JIT 编译在第二步挂起**：针对前述问题，确认了在不使用 JIT 的情况下训练正常，强调了 JIT 编译在第二步挂起的特定问题。
  
  - 另一位用户询问代码是否在 master 分支上，并建议使用 **DEBUG=2** 检查输出以获取更多信息。

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**hackathon-announcements**](https://discord.com/channels/1280234300012494859/1280236929379602493/1307082725009920062) (1 messages):

> - `Intel Tiber AI Cloud`
> - `Intel Liftoff 项目`
> - `与 Intel 的 AMA`

- **Intel 关于 AI 开发的独家 AMA**：参加 11/21 下午 3 点（PT 时间）举行的 Intel AMA 会议：[Building with Intel: Tiber AI Cloud and Intel Liftoff](https://lu.ma/agents-hackathon-intel)，深入了解 Intel 的 AI 工具。
  
  - 本次活动将提供与 Intel 专家互动的独特机会，学习如何利用其资源加速你的 AI 项目。
- **探索 Intel Tiber AI Cloud 的功能**：会议将介绍 **Intel Tiber AI Cloud**，这是一个旨在通过先进计算能力增强 AI 项目的强大平台。
  
  - 参与者将学习如何利用这一强大平台，在黑客松项目中实现最佳效率。
- **Intel Liftoff 项目支持初创企业**：关于 **Intel Liftoff 项目** 的讨论将涵盖为初创企业提供的全面福利，包括导师指导和技术资源。
  
  - 与会者将了解该项目如何从头开始支持他们的开发工作。

 

**提到的链接**：[Building with Intel: Tiber AI Cloud and Intel Liftoff · Luma](https://lu.ma/agents-hackathon-intel)：Building with Intel: Tiber AI Cloud and Intel Liftoff。关于本次 AMA：加入我们，与来自我们尊贵赞助商 Intel 的专家进行独家 AMA 会议……

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-announcements**](https://discord.com/channels/1280234300012494859/1280369709623283732/1308176543134257328) (1 messages):

> - `第 10 讲公告`
> - `Percy Liang 的演讲`
> - `开源 Foundation Models`
> - `课程资源`

- **第 10 讲于 PST 时间下午 3 点开始**：第 10 讲由受邀演讲嘉宾 **Percy Liang** 主讲，定于今天 **3:00 PM PST** 举行。观众可以观看 [此处直播](https://www.youtube.com/live/f3KKx9LWntQ)。
  
  - 本次课程的主题是 **Foundation Models 时代的开源与科学**，讨论开源对 AI 的价值。
- **Percy Liang 谈开源创新**：在他的演讲中，**Percy Liang** 认为在开放程度骤降的背景下，开源模型对于建立严谨的 AI 基础至关重要。他强调需要大量资源，包括 **data**、**compute** 和研究专业知识。
  
  - Liang 隶属于斯坦福大学和 Foundation Models 研究中心（CRFM），他将提出利用社区支持开发开源 Foundation Models 的 **极具前景的方向**。
- **随时访问课程资源**：学生可以在 [课程网站](http://llmagents-learning.org/f24) 找到所有必要的课程材料，包括直播链接和家庭作业。
  
  - 如有任何疑问或反馈，鼓励参与者在指定频道直接与课程工作人员沟通。

 

**提到的链接**：[CS 194/294-196 (LLM Agents) - Lecture 10, Percy Liang](https://www.youtube.com/live/f3KKx9LWntQ.)：暂无描述

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/1307127627424993281) (3 messages):

> - `寻找队伍`
> - `测验分数通知`

- **用户正在寻找加入的队伍**：一名成员正在寻找可以加入的队伍，并分享了他们的 [LinkedIn 个人资料](https://www.linkedin.com/in/ppujari)。他们提到自己是较晚加入课程的，但强调会投入额外的时间。
  
  - *在课程中寻找合作机会。*
- **测验分数通知将发送至注册邮箱**：一名成员确认，每次测验尝试后，包含分数的 Google Form 副本将发送到学生的注册邮箱账户。
  
  - 如果在收件箱中没找到，他们建议同时也检查 **SPAM** (垃圾邮件) 文件夹。

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-lecture-discussion**](https://discord.com/channels/1280234300012494859/1282734248112947210/1307129621023559760) (10 messages🔥):

> - `成绩查询`
> - `写作环节截止日期`
> - `缺失的课程讲义`
> - `Hackathon 文章截止日期`
> - `使用 Substack 提交`

- **课程结束后可查询成绩**：由于学生人数众多，成绩仅在 **12月12日** 课程结束时公布。
  
  - 鼓励参与者提交所有内容并付出努力，以确保顺利通过。
- **明确写作环节截止日期**：写作环节的截止日期定为 **PST 时间 12月12日**，学生可以在此之前的任何时间撰写帖子。
  
  - 确认了在提交写作内容的时间上具有灵活性。
- **第 8 讲缺失讲义**：有人对 **Lecture 8** 共享幻灯片中缺失的部分内容表示担忧。
  
  - 据指出，一些客座讲师希望在发布前对他们的讲义进行微调。
- **确认 Hackathon 文章截止日期**：Hackathon 参与者提交文章的截止日期为 **PST 时间 12月12日晚上 11:59**。
  
  - 参与者表示有兴趣了解更多关于这一时间表的信息。
- **使用 Substack 提交写作内容**：询问是否可以使用 **Substack** 提交写作作业。
  
  - 回复是肯定的，表示用于该目的是可以接受的。

 

---

### **DSPy ▷ #**[**show-and-tell**](https://discord.com/channels/1161519468141355160/1202371242519441499/1308178069692874754) (1 messages):

> - `DSPy VLM 教程`
> - `从图像中提取属性`

- **DSPy 引入 VLM 支持**：[DSPy 最近在 beta 版中增加了对 VLM 的支持](https://x.com/karthikkalyan90/status/1858609018228355414)，展示了从图像中提取属性的功能。
  
  - 一名成员分享了一个示例，演示如何从网站 **screenshots** (截图) 中提取有用的属性，突出了该功能的潜力。
- **从截图中提取属性**：该线程讨论了从网站截图中 **提取有用属性** 的技术，展示了 DSPy 的实际应用。
  
  - 这种方法旨在简化开发人员与视觉数据交互的方式，引起了人们对 DSPy 工具包中新兴能力的关注。

 

**提到的链接**：[来自 Karthik Kalyanaraman (@karthikkalyan90) 的推文](https://x.com/karthikkalyan90/status/1858609018228355414)：🧵DSPy 最近在 beta 版中增加了对 VLM 的支持。关于使用 DSPy 从图像中提取属性的简短推文。在这个示例中，我们将看到如何从网站截图中提取有用的属性...

 

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/1307848397436358688) (10 messages🔥):

> - `DSPy Signatures`
> - `用户名生成策略`
> - `使用 DSPy 进行代码分析`
> - `LLM 缓存问题`
> - `LLM 输出的随机化`

- **DSPy Signatures 中少用英文，多用代码**：一位成员分享到，大多数人在编写 DSPy Signatures 时使用了过多的英文；相反，通过简洁的代码可以实现很多功能。
  
  - 他们引用了 [Omar Khattab 的一条推文](https://x.com/lateinteraction/status/1858284772084375784)，该推文强调了**极简伪代码**的有效性。
- **使用 DSPy 处理用户名生成**：一位用户提出了关于生成多样化用户名的问题，指出存在大量重复。
  
  - 另一位成员建议禁用 LLM 对象中的 Cache，但原用户提到他们已经这样做了。
- **通过高方差增加用户名的随机性**：为了解决用户名重复的问题，一位成员建议提高 LLM Temperature，并在生成名称之前加入讲故事的环节。
  
  - 他们提议使用高 Temperature 模型生成故事，并使用低 Temperature 模型来保证名称生成的质量。
- **使用 DSPy 代码分析主题**：一位用户分享了一段 Python 代码片段，用于使用 DSPy 分析主题的各个方面，涵盖了市场规模和关键研究人员等维度。
  
  - 该代码演示了如何利用 DSPy 的预测功能来收集有关主题不同子领域的信息。

**提到的链接**：[Omar Khattab (@lateinteraction) 的推文](https://x.com/lateinteraction/status/1858284772084375784)：新爱好：DSPy code golf。带有自然语言任务的极简伪代码应该可以直接工作且可优化。引用 Ajay Singh (@ajay_frontiers)：为了可靠性，没有什么能比得上 DSPy（感谢...）

---

### **LAION ▷ #**[**general**](https://discord.com/channels/823813159592001537/823813160075132991/1307079702908768306) (8 messages🔥):

> - `VisRAG 演讲`
> - `黑客松文化`
> - `版权流氓 (Copyright Trolls)`
> - `法律讨论`

- **Jina AI 精彩的 VisRAG 演讲**：参加即将举行的 Jina AI 演讲，Shi 将在会上展示他在 [VisRAG](https://huggingface.co/openbmb/VisRAG-Ret) 上的创新工作，这是一个无需解析的纯视觉 RAG 流水线。
  
  - 届时将了解到 VisRAG 的构建、评估以及令人兴奋的未来可能性，其训练数据集规模几乎是 ColPali 的三倍。
- **Tasty Hacks 黑客松公告**：一个新的黑客松项目 Tasty Hacks 旨在激励参与者为了创意而非实用性进行创作，摆脱传统黑客松中为了获胜而进行优化的文化。
  
  - 组织者正在寻找善良且极客的人士，愿意在仅 **20-30 人**的小型环境中组队创作。
- **关于版权流氓的讨论**：据透露，一个名为 Trevityger 的特定个人被确认为臭名昭著的**版权流氓**，引起了社区的关注。
  
  - 在一位成员深入调查了 Trevityger 的背景后，其他成员表达了警惕并分享了见解。
- **法律事务与社区见解**：在一个帖子中讨论了法律问题，一位成员解释了围绕某些主题的法律，以澄清误解。
  
  - 社区保持着高度参与，强调了在讨论中具备法律意识的重要性。
- **Epoch 之间的适度改进**：一位成员报告称，模型性能在最近的 Epoch 中仅有**适度改进**，特别指出了从 Epoch 58 到 Epoch 100 的更新。
  
  - 技术讨论强调了尽管模型迭代在进步，但挑战依然存在。

**提到的链接**：

- [tastyhacks '24 berkeley · Luma](https://lu.ma/s3pe30fz)：现在的许多黑客松都被地位所玷污。参与者为了获胜，在他们的作品中极少地融入赞助商奖项，这随后……
- [多模态文档 RAG 的进展 · Zoom · Luma](https://lu.ma/56flmrf9)：在本次演讲中，Shi 将介绍他最近的工作 VisRAG，这是一个无需解析的纯视觉检索增强生成 (RAG) 流水线。他……
- [openbmb/VisRAG-Ret · Hugging Face](https://huggingface.co/openbmb/VisRAG-Ret)：未找到描述

---

### **LAION ▷ #**[**research**](https://discord.com/channels/823813159592001537/824374369182416994/1307785108253704274) (1 条消息):

> - `MultiNet Benchmark`
> - `Vision-Language-Action models`
> - `VLA model performance`
> - `Prompt engineering in robotics`
> - `Mini VLA model μGATO`

- **针对 VLA 模型的 MultiNet Benchmark 发布**：名为“Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks”的新论文在 20 个真实世界任务中评估了 **VLA 模型**，揭示了关于其性能的关键见解。完整详情请见[此处](https://multinet.ai/static/pages/Multinetv01.html)。
  
  - 这项工作旨在推动通用机器人系统的发展，展示了在不同任务中进行系统评估的迫切需求。
- **领先 VLA 模型之间的性能比较**：对 **GPT-4o**、**OpenVLA** 和 **JAT** 的对比显示，虽然像“取放”这样的简单任务是可以处理的，但模型在复杂的多步过程中表现挣扎。值得注意的是，结果表明性能会根据任务和机器人平台产生显著差异。
  
  - 分析强调了复杂的 **prompt engineering** 的效用，在使用 **GPT-4o** 时，它提高了任务执行的一致性。
- **μGATO 简介，一个小型 VLA 模型**：团队推出了 **μGATO**，这是一个为 **MultiNet** 基准定制的小型且易于理解的基准模型。它作为探索和推进机器人领域多模态动作模型的工具。
  
  - Manifold 团队持续的努力预示着更多多模态动作模型的创新即将发布。
- **多模态动作任务的代码和软件发布**：**MultiNet** 项目的所有评估代码和基准现在都可以在 GitHub 上下载，方便根据基准对模型进行性能分析。鼓励开发者在[此处](https://github.com/ManifoldRG/MultiNet/tree/main)查看该套件。
  
  - 这一举措使研究人员和开发人员能够有效地在各种机器人任务上标准化和扩展他们的模型。
- **呼吁合作与贡献**：Manifold 团队邀请通过直接联系或社区渠道为增强 **MultiNet** 基准做出贡献和合作。感兴趣的各方可以通过私信联系或在[此处](https://discord.com/invite/BPqB3EG6dF)加入 Discord 社区。
  
  - 这种协作努力强调了社区参与在推进多模态动作模型研究中的重要性。

**提到的链接**：

- [Multinetv0.1](https://multinet.ai/static/pages/Multinetv01.html)：未找到描述
- [MultiNet/src/modules at main · ManifoldRG/MultiNet](https://github.com/ManifoldRG/MultiNet/tree/main/src/modules)：通过在 GitHub 上创建账户，为 ManifoldRG/MultiNet 的开发做出贡献。
- [GitHub - ManifoldRG/MultiNet](https://github.com/ManifoldRG/MultiNet/tree/main)：通过在 GitHub 上创建账户，为 ManifoldRG/MultiNet 的开发做出贡献。
- [SOCIAL MEDIA TITLE TAG](https://multinet.ai/)：SOCIAL MEDIA DESCRIPTION TAG TAG

---

### **MLOps @Chipro ▷ #**[**events**](https://discord.com/channels/814557108065534033/869270934773727272/1307870485849178152) (2 条消息):

> - `Starting with MLOps`
> - `Seeking Clarification`
> - `Complexity in MLOps`

- **寻求 MLOps 指导**：一位成员表达了他们对从哪里开始学习 MLOps 的困惑，称：*“这太复杂了。”*
  
  - 这促使另一位成员请求澄清，要求发帖者进一步明确他们的疑虑。
- **针对复杂性的澄清请求**：另一位成员参与了对困惑的讨论，指出问题相当宽泛，并要求提供更多细节。
  
  - 这种互动强调了在处理像 MLOps 这样复杂的主题时，进行更清晰沟通的必要性。

---

### **Mozilla AI ▷ #**[**announcements**](https://discord.com/channels/1089876418936180786/1089876419926032396/1307813625087529081) (2 messages):

> - `Common Corpus dataset`
> - `Transformer Lab Demo`

- **Pleias 发布用于 LLM 训练的 Common Corpus**：Pleias（2024 Builders Accelerator 成员）宣布发布 **Common Corpus**，这是目前最大的用于 LLM 训练的开源数据集，强调致力于在宽松许可证下提供训练数据。
  
  - *“开源 LLM 生态系统尤其缺乏训练数据的透明度，”* Pleias 指出，Common Corpus 旨在解决这一透明度差距。[点击此处查看完整帖子](https://discord.com/channels/1089876418936180786/1306706786824487035)。
- **明天由 Transformer Lab 带来的 RAG 演示**：**Transformer Lab** 计划进行一场演示，展示如何使用用户友好的 UI，在**无需编码**的情况下对 LLM 进行 RAG 的训练、微调、评估和使用。
  
  - 该活动承诺使流程在**本地环境**中**易于安装**，这在社区中引起了热烈反响。更多详情请见[此处](https://discord.com/events/1089876418936180786/1300842793945530378)。

 

---

### **Torchtune ▷ #**[**general**](https://discord.com/channels/1216353675241590815/1216353675744641096/1308102966363815976) (1 messages):

> - `DCP async checkpointing`
> - `Intermediate checkpointing efficiency`

- **DCP 异步 Checkpointing 实现**：[DCP 异步 Checkpointing](https://github.com/pytorch/torchtune/pull/2006) 旨在通过一项目前正在开发中的新功能来改进 TorchTune 中的中间 Checkpointing。
  
  - 该 Pull Request 显示，该过程旨在通过将中间 Checkpointing 时间减少 **80%** 来显著提高效率。
- **中间 Checkpointing 时间缩减**：DCP 异步 Checkpointing 的实现有望带来显著的缩减，预计由于改进的方法论，Checkpointing 时间将减少 **80%**。
  
  - 这种方法是持续优化分布式 Checkpointing 以获得更好性能的努力的一部分。

 

**提到的链接**：[[DCP][RFC] DCP async checkpointing in TorchTune for intermediate checkpoints [WIP] by saumishr · Pull Request #2006 · pytorch/torchtune](https://github.com/pytorch/torchtune/pull/2006)：上下文 此 PR 的目的是什么？是添加新功能、修复 Bug、更新测试和/或文档还是其他（请在此处添加）此 Diff 引入了基于 DistributedCheckpointing 的...

 

---

### **AI21 Labs (Jamba) ▷ #**[**jamba**](https://discord.com/channels/874538902696914944/1222916247063232553/) (1 messages):

rotem2733: Hello?

---

---

---

{% else %}

> 完整的频道分类明细已针对邮件进行了截断。
> 
> 如果您想查看完整明细，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
> 
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}