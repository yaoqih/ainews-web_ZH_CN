---
companies:
- gemma
- llamaindex
- mistral-ai
- cohere
- deepseek-ai
- nous-research
- eureka-labs
date: '2024-07-17T22:57:14.252944Z'
description: '**Gemma 2 (9B, 27B)** 被公认为表现卓越的本地大语言模型（LLM），因其运行速度快、多语言能力强以及在 2080ti
  等消费级 GPU 上的高效表现而备受赞誉。它在包括非英语文本处理和推理在内的各项任务中，表现均优于 **Llama 3** 和 **Mistral 7B** 等模型。/r/LocalLlama
  社区的讨论反映出用户对 Gemma 2 的强烈偏好，其被提及 **18 次**，而 Llama 3 为 **10 次**，Mistral 为 **9 次**。**Phi
  3** 和 **Qwen** 等其他模型虽然也被提及，但被认为已被 Gemma 2 超越。此外，**Andrej Karpathy** 宣布成立 **Eureka
  Labs**，这是一家“AI+教育”初创公司，旨在创建一所配备 AI 助教的 AI 原生学校，并首先推出 **LLM101n** 课程来教授 AI 训练的基础知识。这一举措被视为
  AI 教育领域的重大进展。'
id: 305c4e89-b402-4507-89eb-224a5d3ea59f
models:
- gemma-2-9b
- gemma-2-27b
- llama-3
- mistral-7b
- phi-3
- qwen
original_slug: ainews-gemma-2-tops-rlocalllama-vibe-check
people:
- andrej-karpathy
title: Gemma 2 登顶 /r/LocalLlama 的口碑评测 (vibe check)。
topics:
- model-comparison
- local-llms
- multilinguality
- model-efficiency
- fine-tuning
- ai-education
- ai-teaching-assistants
---

<!-- buttondown-editor-mode: plaintext -->**Gemma 2 (9b, 27B) 就是你所需的一切？**

> 2024年7月16日至7月17日的 AI 新闻。我们为你查看了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 服务（**468** 个频道，**2051** 条消息）。预计节省阅读时间（以 200wpm 计算）：**232 分钟**。你现在可以在 AINews 讨论中标记 [@smol_ai](https://x.com/smol_ai)！

[每隔几个月](https://www.reddit.com/r/LocalLLaMA/search/?q=best+models&restrict_sr=on)，总有人在 /r/LocalLlama 提出一个引起热议的“氛围检查（vibe check）”问题（[2024年3月](https://www.reddit.com/r/LocalLLaMA/comments/1b4e50z/whats_the_best_7b_model_right_now_march_2024/)、[2024年6月](https://www.reddit.com/r/LocalLLaMA/comments/1dcf3yy/best_local_base_models_by_size_quick_guide_june/) 以及官方的 [模型大贴（Models Megathread）](https://www.reddit.com/r/LocalLLaMA/comments/1bgfttn/models_megathread_4_what_models_are_you_currently/) 是之前的几次）。

 
![image.png](https://assets.buttondown.email/images/a9376530-dfc2-457a-9c29-3a8a7b4596ad.png?w=960&fit=max)
 

最近一个关于 [同尺寸下最好的模型有哪些？](https://www.reddit.com/r/LocalLLaMA/comments/1e4ja8n/what_are_the_best_models_for_their_size/) 的提问成为了重新审视排名的契机。上个月发布的 Gemma 2（[我们的报道在此](https://buttondown.email/ainews/archive/ainews-gemma-2-the-open-model-for-everyone/)）即使在没有 2B 模型的情况下也轻松胜出：

- **Gemma 2: 18 次提及**
  - "同尺寸下我运行过的最好的 LLM 之一。"
  - "9B 在对哲学文本进行摘要和推理方面的表现也让我印象深刻，其英语概念组合相当连贯。"
  - "我们在 Agent 工作流中获得了非常好的性能，允许 LLM 每次专注于一个任务。"
  - "同感。在我的 2080ti 上运行 Gemma 2 9b，非常流畅且结果很好。我非常想要一个能像 Perplexity 或 Kagi FastGPT 那样快速提供来源链接的本地 LLM，因为那个功能太强大了。"
  - "如果你问我的话，Gemma 2 9b 比 Llama 8b 好得多。"
  - "Gemma 2 9b 是唯一一个既超快，又能在任何任务中击败 3.5 的模型。而且在同尺寸模型中，它的法语表现**真的**很好。非常适合 Discord 机器人。如果你卸载大部分层，你可以得到一个运行足够快且仅占用 3 或 4GB VRAM 的 Discord 机器人，这样你就有空间运行 Stable Diffusion 之类的东西了！真的不可思议。结合 Moondream 1b 进行视觉处理，瞧，你就拥有了一个能很好遵循 Prompt 和写作风格、并能“看到”聊天中图片的通晓多国语言的机器人。总共只需约 5GB VRAM。"
  - "在处理非英语文本时，Gemma 9B 甚至远优于 Llama 70B。"
  - "我尝试使用 Gemma 2 9b Instruct 进行合成数据生成（从段落中推导问题和答案），但它 90% 的时间都拒绝配合……这给我留下了很坏的印象。"
- **Llama 3**: 10 次提及
  - "Llama 3 70B 和 Qwen 72B 是 70B 左右级别 LLM 的首选。"
- **Mistral**: 9 次提及
  - "对我来说是 Mistral 7B。不是 MoE 版本，我没有运行它的硬件。"
  - "我喜欢 Mistral 7B (v03) Instruct。恕我直言，它甚至无法与 Gemma 9B 相比，即使是后者的较小量化版本。但 Mistral v03 比 Gemma 9B 早出很久。"
  - "Mistral-Instruct v0.3 7b。我喜欢这个模型。即使 Gemma 8b 和 Phi Medium 看起来更好。此外 WizardLM2（与 Mistral 非常相似且基于它）也很棒……试试看。"
- **Phi 3**: 6 次提及
- **Qwen**: 5 次提及
  - "刚出来时很不错，但已被 Gemma 和 Phi-3 取代。"

其他正面评价：DeepSeek, Cohere Command R, InternLLM, Yi 34B (Nous-Capybara 版本)

> Meta 说明：**我们现在在 Reddit 摘要中将 /r/localLlama 独立出来**，因为其他 subreddits 往往会淹没技术讨论。请享用！

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，从 4 次运行中选取最佳结果。

**Andrej Karpathy 的新 AI+教育公司 Eureka Labs**

- [@karpathy](https://twitter.com/karpathy/status/1813263734707790301) 宣布他正在创办一家名为 **Eureka Labs 的 AI+教育公司**，旨在建立一所 **AI 原生学校**。其目标是让**任何人都能轻松学习任何知识**，由 AI 助教（AI Teaching Assistants）辅助人类教师。他们的**第一个产品将是 LLM101n**，这是一门关于训练你自己的 AI 的本科级课程。课程材料将免费提供，收入将来自线上/线下的训练营（cohorts）。
- [@DrJimFan](https://twitter.com/DrJimFan/status/1813360847361831226) 指出，**没有人比 Andrej 更适合做教育科技（EdTech）**，该领域的其他 AI 初创公司无法与之竞争。他很高兴两人都喜欢 "Eureka" 这个名字。
- [@danielhanchen](https://twitter.com/danielhanchen/status/1813330269044408612) 对 **LLM101n 课程**感到兴奋，课程章节涵盖了 bigrams、attention、transformers、optimization、datasets、inference、fine-tuning 和 deployment。他提到 Andrej 的课程材料（如 CS231n 和 Zero to Hero）都是无价之宝。

**新模型发布**

- [@GuillaumeLample](https://twitter.com/GuillaumeLample/status/1813231491154899012) 宣布以 Apache 2 许可证发布 **Mathstral 7B 和 Codestral Mamba 7B**。Mathstral 7B 在 **MATH 测试集上获得了 56.6% 的 pass@1**，表现优于 Minerva 540B 20% 以上。Codestral Mamba 是首批采用 **Mamba 2 架构**的开源模型之一，是目前最出色的 7B 代码模型。
- [@LoubnaBenAllal1](https://twitter.com/LoubnaBenAllal1/status/1813252390692303069) 介绍了 **SmolLM**，这是一系列 135M、360M 和 1.7B 的模型，性能超越了 MobileLLM、Phi1.5 和 Qwen2 的小型模型。该系列模型在 **SmolLM-corpus（由高质量网页、代码和合成数据组成）**上进行训练。
- [@AnthropicAI](https://twitter.com/AnthropicAI/status/1813237754081251573) 发布了 **Claude Android 应用**，现已在 Google Play 上架。

**关于模型架构和训练数据的讨论**

- [@YiTayML](https://twitter.com/YiTayML/status/1813262126162845772) 开始了一个关于 LLM 时代模型架构的博客系列，涵盖了 **Transformer Encoders/Decoders、PrefixLM 和 denoising objectives** 等主题。回应了关于 encoder-only 模型现状以及 denoising objectives 是否仍然有用的问题。
- [@jxmnop](https://twitter.com/jxmnop/status/1813326815496400919) 认为目前 AI 领域最具影响力的主题是 **Agents**。我们需要在下一代语言模型中构建自主代理能力（**agent-native LLMs**），而不是通过 prompting 来伪造。这将需要新的数据集、任务定义和训练技术。
- [@Teknium1](https://twitter.com/Teknium1/status/1813349962065068303) 认为**合成数据也是真实数据**，如果超越了教师模型，就不一定会导致模式崩塌（mode collapse）或在之前的 SOTA 水平停滞不前。

**其他值得关注的更新**

- [@alexandr_wang](https://twitter.com/alexandr_wang/status/1813242291622199628) 分享了 @scale_AI 自从在地下室创业以来已经走过了很长一段路，现在搬进了新办公室。
- [@fchollet](https://twitter.com/fchollet/status/1813362217020219587) 分享了一份讲解详尽的 **Transformer 架构指南，并附带 Keras 代码示例**。
- [@llama_index](https://twitter.com/llama_index/status/1813355957491273936) 在新版本中大幅改进了**基于 markdown 的表格重建功能，用于解析复杂文档**。

---

# AI Reddit 摘要

## /r/LocalLlama

**主题 1：Mistral AI 和 Apple 发布的新模型**

- **[mistralai/mamba-codestral-7B-v0.1 · Hugging Face](https://huggingface.co/mistralai/mamba-codestral-7B-v0.1)** ([Score: 216, Comments: 72](https://reddit.com//r/LocalLLaMA/comments/1e4qgoc/mistralaimambacodestral7bv01_hugging_face/)): **Mistral AI** 发布了 **Mamba-Codestral-7B** 模型，这是一个基于 **Mamba architecture** 的 **7 billion parameter** 代码生成模型。该模型可在 **Hugging Face** 上获取，专为高效推理而设计，能够生成多种编程语言的代码，包括 **Python**、**JavaScript**、**Java**、**C++** 和 **Rust**。该模型在 **Python** 代码生成任务中的表现尤为出色，超越了像 **StarCoder-15B** 这样更大的模型。
- **Apple has released the weights for their 7B DCLM base model.** ([Score: 181, Comments: 48](https://reddit.com//r/LocalLLaMA/comments/1e4jw0c/apple_has_released_the_weights_for_their_7b_dclm/)): **Apple 揭晓 DCLM-Baseline-7B 模型**。这个 **7 billion parameter** 的语言模型在 **2.5T tokens** 上进行了训练，具有 **2048 token** 的 context length，基于 **DCLM-Baseline dataset**，旨在展示系统化数据整理对模型性能的影响。一个具有 **8K context length** 的更新版本也已发布，并提供了 [Hugging Face repository](https://huggingface.co/apple/DCLM-Baseline-7B-8k)、[研究论文](https://arxiv.org/abs/2406.11794) 和 [相关 GitHub project](https://github.com/mlfoundations/dclm) 的链接。
  - **Apple 的开源模型惊喜**：**Apple 发布**开源模型受到了社区的赞扬。用户对 **DCLM (Data-Centric Language Model) 方法**可能带来的见解感到兴奋，认为这是迈向更 **open-source AI development** 的一步。
  - **Context Length 困惑**：关于 **2048 token context length** 的意义引发了讨论。用户争论这与 **Llama 3** 等其他模型相比如何，强调了不同 **LLM** 之间 tokenization 方法的差异。
  - **Benchmarks 和许可问题**：社区成员询问新模型的 **performance benchmarks**。关于 **"Apple ASCL" license** 的问题也随之出现，用户将其与 **MIT license** 进行比较，并寻求对其开源状态的澄清。


**Theme 2. Llama 3 Performance and Limitations**

- **[This meme only runs on an H100](https://i.redd.it/urpjifh14xcd1.jpeg)** ([Score: 230, Comments: 42](https://reddit.com//r/LocalLLaMA/comments/1e4uwz2/this_meme_only_runs_on_an_h100/)): **“这个梗只能在 H100 上运行”** 幽默地夸大了现代 AI 模型的高计算需求。这个笑话利用了 **NVIDIA H100 GPU** 是目前最强大且最受追捧的用于 AI 训练和推理的图形处理单元这一事实，它常用于大型语言模型和其他计算密集型 AI 任务。
- **[I gave Llama 3 a 450 line task and it responded with "Good Luck"](https://i.redd.it/2n1oytw3pucd1.png)** ([Score: 383, Comments: 46](https://reddit.com//r/LocalLLaMA/comments/1e4kg7n/i_gave_llama_3_a_450_line_task_and_it_responded/)): **Llama 3 在长指令测试中失败**。当被给予一个 **450 行的任务**时，Llama 3 以简单的 “Good Luck” 作为回应，而不是尝试处理或执行这组冗长的指令。这种行为表明 Llama 3 在有效处理极长或复杂 prompts 方面可能存在局限性。
  - **“Good Luck” 还是好的 AI？** 模型的反应可能是由于**类似考试的措辞**。添加 “Output:” 或 “Answer:” 可能会产生不同的结果，突显了 **text completion 与 comprehension 之间的区别**。
  - **AI 那令人感同身受的懒惰**：一个早期的开源模型在回应代码请求时说：*“这听起来像是很多工作”*，展示了对复杂任务**类似人类的抵触**。
  - **Context 至关重要**：**Ollama** 中默认的 **context length 为 2048**，可能截断了冗长的指令。将其增加到 **8096** 可能会使其能够处理完整的 450 行任务。

**Theme 3. Comparing Model Performance by Size**

- **按规模划分，哪些是最佳模型？** ([Score: 60, Comments: 46](https://reddit.com//r/LocalLLaMA/comments/1e4ja8n/what_are_the_best_models_for_their_size/))：**按规模划分的最佳推理模型**：该帖子寻求关于相对于其规模而言最“智能”的语言模型的意见，重点关注**纯粹的推理能力**和脱离训练数据的解决问题能力。作者特别询问了关于各种规模（**3B**、**4B**、**7B** 及更大）模型的个人使用体验，而不是依赖排行榜排名。
  - **Gemma 2 大放异彩**：**Gemma 2 9B** 和 **27B** 模型因其相对于规模的性能而受到广泛赞誉。用户强调了它们的推理能力和多语言能力，一些人将其与 **GPT-3.5** 级别的性能进行比较。
  - **规模很重要，但并非绝对**：讨论包括了对各种模型规模的建议，从 **Phi-3 4B** 到 **Llama 3 70B** 和 **Qwen 72B**。用户争论了模型规模、性能和硬件要求之间的权衡。
  - **在低端系统上进行测试**：一位用户分享了正在进行的实验，在包括没有 GPU 的**第四代 i7** 处理器在内的旧硬件上运行从 **4B** 到 **112B** 的模型。结果预计将于 **9 月中旬**在**帕萨迪纳**举行的 **Technosecurity conference** 上展示。

**主题 4：关于 AI 炒作与长期潜力的辩论**

- **[Linux Torvalds](https://i.redd.it/z4scsmapczcd1.jpeg)** ([Score: 77, Comments: 40](https://reddit.com//r/LocalLLaMA/comments/1e55iit/linux_torvalds/))：**Linux Torvalds**，**Linux kernel** 的创建者，在最近的一次采访中对当前的 **AI 炒作**表示怀疑。他认为，虽然 AI 在**图像识别**和**语言模型**等特定领域取得了显著进展，但它仍然缺乏通用智能，主要擅长模式匹配而非真正的理解。Torvalds 认为当前的 AI 热潮很大程度上是由**营销**驱动的，并警告不要高估 AI 的能力。
  - 评论者将 **AI 炒作**与**互联网泡沫**进行了类比，暗示这是一个过度炒作、估值过低并最终产生改变世界影响的循环。一些人认为，尽管短期内存在夸大，但 **AI 的长期潜力**被显著低估了。
  - 随后引发了关于 **Large Language Models (LLMs)** 能力的辩论，一些人声称它们可以取代 **30% 的工人**，而另一些人则认为，与人类相比，LLMs 在许多任务中是不可靠且不可预测的。
  - 评论者幽默地利用了 **Linus Torvalds** 名字的拼写错误，开玩笑地将他与“**Tim Apple**”、“**Bill 'Michaelsoft' Gates**”和“**Linus Tech Tips**”联系起来，展示了社区与技术名人之间的趣味互动。

## 跨越 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

> 评论抓取现在可以工作了，但还有很多需要改进的地方！

**主题 1：Llama 3 的性能与局限性**

- [/r/LocalLLaMA] **[这个梗图只能在 H100 上运行](https://i.redd.it/urpjifh14xcd1.jpeg)** ([Score: 230, Comments: 42](https://reddit.com//r/LocalLLaMA/comments/1e4uwz2/this_meme_only_runs_on_an_h100/))：**“这个梗图只能在 H100 上运行”**幽默地强调了现代 AI 模型极高的计算需求。这个笑话利用了这样一个想法：即使是像显示梗图这样简单的任务，也可能需要 **NVIDIA 的 H100 GPU**，这是专为 **AI 和 machine learning 工作负载**设计的最强大且最昂贵的显卡之一。

- [/r/LocalLLaMA] **[我给 Llama 3 布置了一个 450 行的任务，它回复了“祝你好运”](https://i.redd.it/2n1oytw3pucd1.png)** ([Score: 383, Comments: 46](https://reddit.com//r/LocalLLaMA/comments/1e4kg7n/i_gave_llama_3_a_450_line_task_and_it_responded/)): **Llama 3** 在面对一个 **450 行的任务**时遇到了困难，它没有尝试完成任务，而是简单地回复了“祝你好运”。这种出乎意料的反应凸显了该模型在处理复杂、冗长 prompt 或任务时的潜在局限性，并引发了对其在大型代码编写或文本生成任务中实际应用能力的质疑。
  - **“祝你好运”可能与考试有关**：短语“Your Task is (...)”可能会触发**类似考试的反应**。添加“Output:”或“Answer:”可能会产生不同的结果，这突显了**文本补全与理解之间的区别**。
  - **AI 模型也会偷懒**：一个早期的开源模型在面对代码请求时回复道：*“这听起来工作量很大”*，展示了 AI 响应中**类人的抵触情绪**。  
  - **技术限制**：问题可能源于使用了 **base model 而非 instruct model**。原帖作者确认使用的是 [**8b-instruct-q6_K**](https://ollama.com/library/llama3:8b-instruct-q6_K)，这表明还有其他因素在起作用。
  - **上下文长度很重要**：**Ollama 默认的上下文长度为 2048**，可能截断了冗长的指令。将其增加到 **8096** 可能会允许处理完整的指令。


- [/r/singularity] **[许多人根本无法想象技术会进步](https://i.redd.it/5lx3ajibn0dd1.png)** ([Score: 354, Comments: 105](https://reddit.com//r/singularity/comments/1e5ahdm/so_many_people_simply_cannot_imagine_tech/)): **对 AI 快速进展的怀疑**：该帖子强调了人们普遍无法预见快速的技术进步，特别是在 AI 领域。作者引用了历史案例，例如 **1903 年莱特兄弟的首次飞行**到 **1969 年的登月**，以此说明技术进步并超越最初预期的速度之快。
   - **驳斥对 AI 快速进展的怀疑**：**《工程杂志》(Engineering Magazine)** 在 **1909 年 12 月**曾预测飞行器的潜力有限，然而**不到 40 年后**，**Enola Gay** 就已投入使用。这突显了**技术进步如何超越预期**。
  - **飞行汽车：现实 vs. 想象**：虽然 2000 年出现飞行汽车的预测落空了，但如今**直升机**和一些不切实际的飞行汽车原型确实存在。有人认为，**AI 自动驾驶仪**对于安全、广泛地采用飞行汽车是必要的。
  - **移动的 AGI 时间线**：就在 **3-4 年前**，**2045 年**还被认为是实现 AGI 的乐观估计。现在，它被视为悲观估计。**2023 年对 2278 名 AI 研究人员的一项调查**估计，到 **2047 年**，AI 在所有任务中超越人类的可能性为 **50%**。
  - **经济价值驱动 AI 进步**：与已进入平台期的智能手机改进不同，AI 的进步提供了巨大的经济价值。企业愿意为能超越人类员工的 AI 支付巨额费用，从而推动了快速进展。
  - **人类在理解指数增长方面的局限性**：许多人，包括开发者和企业家，尽管意识到趋势，但仍难以预测和规划指数级的技术增长。


**Theme 3. AI 图像与视频生成**

- [/r/StableDiffusion] **[LivePortrait 的首次测试](https://v.redd.it/5lby8nan6xcd1)** ([Score: 226, Comments: 26](https://reddit.com//r/StableDiffusion/comments/1e4va0j/first_test_with_liveportrait/)): **LivePortrait 测试**：一位用户尝试使用 **LivePortrait AI 工具**从静态图像生成视频。结果被描述为**“相当不错”**，AI 成功地让图像动了起来，并使嘴唇动作与提供的音频匹配，尽管在嘴部区域可以观察到一些明显的伪影。

- [/r/singularity] **[名人们与年轻时的自己在一起](https://v.redd.it/rpcw3s5qpxcd1)** ([Score: 887, Comments: 137](https://reddit.com//r/singularity/comments/1e4zcv5/celebrities_hanging_out_with_their_younger_selves/)): **AI 生成的图像**描绘了名人们与年轻时的自己互动的场景，展示了先进图像合成技术的能力。这些视觉效果将**现在的外貌**与**历史照片**融合在一起，创造出无缝且逼真的合成图，突显了知名人士的老化过程和职业演变。这些图像展示了 AI 在创造富有想象力和怀旧色彩的视觉内容方面的潜力，同时也引发了关于数字媒体真实性和操纵性的问题。

- [/r/StableDiffusion] **[瓶中水下世界](https://v.redd.it/svfrbbb4tucd1)** ([Score: 323, Comments: 7](https://reddit.com//r/StableDiffusion/comments/1e4ks8j/underwater_inside_a_bottle/)): **使用 AI 创建的瓶中水下场景动画**。艺术家使用 **Midjourney** 生成初始图像，然后利用 **Stable Diffusion** 和 **ControlNet** 进行 **inpainting** 和动画制作，最终在玻璃瓶中呈现出动态的水下场景。
   - **原作者分享细节**：艺术家透露了 **ComfyUI** 工作流，包括使用 **RunwayML** 进行 **masking**、**AnimateDiff** 制作动画，以及使用来自 **Lexica** 的参考图配合 **IPAdapter**。
   - **ControlNet 组合**：该技术采用了 **depth** 和 **Canny** **ControlNet**，并结合 **Reborn model** 和 **LCM Lora** 以实现更快的 **sampling**。
   - **快速且高效**：动画仅通过 **11 steps** 和 **cfg 2** 创建，并使用 **LCM sampler** 进行快速生成。

**主题 4. 新 AI 模型发布与架构**

- [/r/LocalLLaMA] **[mistralai/mamba-codestral-7B-v0.1 · Hugging Face](https://huggingface.co/mistralai/mamba-codestral-7B-v0.1)** ([Score: 216, Comments: 72](https://reddit.com//r/LocalLLaMA/comments/1e4qgoc/mistralaimambacodestral7bv01_hugging_face/)): **Mistral AI** 发布了 **Mamba-Codestral-7B**，这是一个基于 **Mamba architecture** 的新型 **7 billion parameter** 语言模型。该模型可在 **Hugging Face** 上获取，专为 **code generation** 任务设计，并在代码和自然语言数据的混合体上进行了训练。此次发布标志着将以处理长序列效率著称的 **Mamba architecture** 应用于 **code generation** 领域迈出了重要一步。
- [/r/singularity] **[[Google DeepMind] Mixture of A Million Experts。Daniel Jeffries：“降低了 inference cost 和 memory usage，可扩展至 millions of experts，而且恰好克服了 catastrophic forgetting 并使模型的 lifelong learning 成为可能。”](https://arxiv.org/abs/2407.04153)** ([Score: 381, Comments: 82](https://reddit.com//r/singularity/comments/1e4mu0e/google_deepmind_mixture_of_a_million_experts/)): **Google DeepMind** 推出了 **Mixture of A Million Experts (MoME)** 模型，据报道该模型在扩展到 **millions of experts** 的同时，**降低了 inference cost 和 memory usage**。根据 Daniel Jeffries 的说法，该模型还解决了 **catastrophic forgetting** 的挑战，并使 AI 系统的 **lifelong learning** 成为可能。**MoME** 方法代表了 AI 模型架构的重大进步，有望提供更高效、适应性更强的系统。

- [/r/LocalLLaMA] **[我给 Llama 3 布置了一个 450 行的任务，它回了一句“祝你好运”](https://i.redd.it/2n1oytw3pucd1.png)** ([Score: 383, Comments: 46](https://reddit.com//r/LocalLLaMA/comments/1e4kg7n/i_gave_llama_3_a_450_line_task_and_it_responded/)): **Llama 3 对复杂任务的意外回应**。当面对一个 **450-line task** 时，据报道 **Llama 3** 并没有尝试完成它，而是简单地回复了一句“祝你好运”。这一轶事表明 **Llama 3** 在处理极长或复杂的 **prompt** 时可能存在局限性，引发了人们对其在处理大型任务时与其他 AI 模型相比性能如何的疑问。
  - **Prompt Engineering 很重要**：在 **prompt** 中添加 "Output:" 或 "Answer:" 可能会 **显著改变 Llama 3 的回复**。这突显了正确的 **prompt formatting** 的重要性，以及 **text completion** 与 **comprehension** 之间的区别。
  - **Context Length 限制**：**Ollama** 中的默认 **context length** 为 **2048 tokens**，这可能会截断冗长的指令。将其增加到 **8096 tokens** 可能会让 **Llama 3** 处理完整的 450 行任务。
  - **模型变体影响性能**：所使用的具体模型是 [**llama3:8b-instruct-q6_K**](https://ollama.com/library/llama3:8b-instruct-q6_K)。一些用户认为这种行为在 **base model** 中比在 **instruct-tuned version** 中更常见。
  - **AI 模仿人类行为**：几位用户幽默地指出，**Llama 3** 回复“祝你好运”或“这听起来工作量很大”反映了人类对复杂任务的典型反应，并开玩笑说这展示了类人智能。


**主题 5. AI 监管与公众认知**

- [/r/singularity] **[特朗普的新副手 Vance 谈 AI 监管](https://x.com/ai_for_success/status/1813036499329511900?t=p46Mncs0gfvyIb3LmCHiLw&s=19)** ([Score: 212, Comments: 418](https://reddit.com//r/singularity/comments/1e4n9m3/vance_new_vp_of_trump_on_ai_regulation/)): **J.D. Vance**，**Donald Trump** 的潜在**副总统**人选，表达了对 **AI 监管**的担忧。在最近的一次采访中，Vance 强调需要采取一种**“强硬的（muscular）” AI 治理方法**，并暗示当前的监管框架不足以应对 AI 技术的快速进步。他强调了维持**美国技术霸权**的重要性，同时也要防范与 AI 发展相关的潜在风险。

- [/r/singularity] **[学生们，安息吧（RIP students）](https://v.redd.it/zsfxtxfizscd1)** ([Score: 434, Comments: 158](https://reddit.com//r/singularity/comments/1e4mp49/rip_students/)): **“学生们，安息吧”**：AI 对教育的影响可能是变革性的。该帖子标题对 AI 给学生带来的影响持悲观态度，可能暗示由于教育领域 AI 的进步，传统的学生角色或学习方法可能会过时或发生重大改变。

- [/r/singularity] **[许多人根本无法想象技术会持续进步](https://i.redd.it/5lx3ajibn0dd1.png)** ([Score: 354, Comments: 105](https://reddit.com//r/singularity/comments/1e5ahdm/so_many_people_simply_cannot_imagine_tech/)): **“尽管 AI 取得进步，技术怀疑论依然存在”**：尽管技术飞速发展，许多人仍难以想象技术进步，尤其是在 AI 领域。这种怀疑态度延伸到了就业市场，即使 AI 的能力在各行各业不断扩展，一些人仍然怀疑 AI 显著影响就业的潜力。
  - **“飞行汽车”辩论引发关注**：评论者讨论了 **1909 年《工程杂志》（Engineering Magazine）** 对飞行器的预测，指出**直升机**基本上实现了这一角色。一些人认为，**AI 自动驾驶（autopilot）** 对于 3D 空间中飞行汽车的安全至关重要。
  - **AI 时间线加速令专家震惊**：许多人对 **AGI 预测时间**的剧烈变化表示惊讶。此前，**2045 年**被认为是实现 AGI 的乐观估计；而现在它被视为悲观估计。最近的调查显示，**到 2047 年，AI 在所有任务中超越人类的可能性为 50%**。
  - **技术进步：飞速发展 vs. 平台期**：讨论对比了技术飞速进步的时期与平台期，并以智能手机为例。对于 AI，评论者强调了自 **GPT-4** 以来持续的快速改进，以及 AI 进步在各行各业中的高经济价值。
  - **指数级增长挑战人类理解力**：多条评论指出，包括专家在内的许多人都难以理解或预见指数级的技术增长。这种难以想象未来能力的情况导致了对 AI 对就业和社会潜在影响的怀疑。

---

# AI Discord 摘要

> 摘要之摘要的摘要

**1. AI 模型开发与部署的进展**

- **Codestral Mamba 引起轰动**：Mistral AI 发布了 [Codestral Mamba](https://mistral.ai/news/codestral-mamba/)，这是一款专注于代码生产力的新模型，提供 **linear time inference**（线性时间推理）并具备建模无限长度序列的能力。
   - 该模型在 Albert Gu 和 Tri Dao 的帮助下设计，可免费使用、修改和分发，因其在高级代码推理和快速响应方面的潜力而引发了社区的热情。
- **SciCode 设定了新的 Benchmark 门槛**：[SciCode benchmark](https://scicode-bench.github.io) 正式发布，包含 338 个由物理、数学和生物学博士编写的编程挑战，其中一些基于诺贝尔奖获奖研究。
   - 这一新 benchmark 对当前的 AI 模型构成了挑战，**GPT-4** 和 **Sonnet 3.5** 的准确率不足 5%，凸显了当前 AI 能力与高级科学问题解决之间的差距。
- **SmolLM 将 AI 带入浏览器**：HuggingFace 推出了 **SmolLM models**（135M、360M、1.7B 参数），旨在通过 ONNX 权重和 WebGPU 加速在浏览器中本地运行。
   - 这些模型代表了使 AI 在 Web 环境中更易于访问和更具性能的重要一步，可能为客户端 AI 应用开辟新的可能性。
  

**2. AI 基础设施的挑战与创新**

- **SF Compute 融资 1200 万美元助力 GPU 交易**：SF Compute 筹集了 1200 万美元用于开发大规模 GPU 集群的交易平台，允许预订大量 GPU 资源并出售闲置部分。
   - 该倡议旨在解决 AI 研发中日益增长的 GPU 计算能力需求，可能使高性能计算对更广泛的组织而言更易获得且更高效。
- **LAION 的网络安全警钟**：LAION 社区被一个复杂的黑客组织盯上，该组织创建了一个伪装成名为 **ComfyUI_LLMVISION** 的 ComfyUI 节点的恶意软件，旨在窃取信息并安装木马。
   - 这一事件凸显了 AI 社区日益增加的网络安全风险，特别是考虑到该组织曾有过入侵 Disney 的 Slack 等备受瞩目的攻击历史。
- **Mojo 在 Intel 芯片上的性能难题**：Modular Discord 中的讨论透露，**Mojo** 的 `parallelize` 函数在同时具有性能核和能效核的 Intel 芯片上仅利用性能核。
   - 这一设计决策源于在不同核心类型之间高效分配工作的挑战，引发了关于异构计算环境中最佳资源利用的辩论。
  
**3. DeepSeek V2 模型发布**

- **DeepSeek 的引导出现偏差**：@davidkpiano 分享了一个[关于云端状态机的链接](https://x.com/DavidKPiano/status/1806417216914817514)，引发了关于 **DeepSeek-Coder V2-Lite 问题** 的讨论，即该模型不遵循 Prompt 并提供不稳定的答案。
   - @dimfeld 指出禁用 **flash attention** 并未解决问题，暗示 **LM Studio updates** 可能破坏了对 DeepSeek-Coder V2-Lite 的支持。
- **Deepseek 坚持开源路线**：**Deepseek 创始人梁文锋** 在[一次采访](https://x.com/main_horse/status/1813580480761196987?s=46)中表达了对开源的奉献精神，认为这对于构建强大的技术格局至关重要，尽管人们对**中国 AI 的步伐**感到担忧。
   - 尽管 Deepseek 利润微薄，梁文锋的决心依然坚定，他强调在考虑闭源选项之前，首先建立强大的技术生态系统非常重要。
 

**4. 新的多模态 Benchmark**

- **InternVL2-Llama3-76B 视觉**：[InternVL2-Llama3-76B](https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B) 在**多模态学习**方面取得了飞跃，通过参数量从 1B 到 108B 的指令微调模型突破了界限。
   - 用户表达了在 **4x 3090 GPUs** 上运行 **40B 大模型** 的挫败感，主要涉及使用 **autoawq** 进行优化的问题。
- **SciCode 的 STEM 博士级升级**：**SciCode** 通过科学问题编程的 benchmark 树立了新先例，其中包含向诺贝尔奖获得者致敬的内容，这难倒了 **GPT-4** 和 **Sonnet 3.5** 等巨头，其准确率低于 5%。[深入了解](https://scicode-bench.github.io)。
   - 由博士专家组成的 **SciCode benchmark** 挑战涵盖 338 个问题，揭示了不同的科学领域。[此处获取见解](https://x.com/OfirPress/status/1813202497864937825)。
  



---

# 第一部分：高层级 Discord 摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **CUDA 困境与 VRAM 探索**：技术讨论集中在 **CUDA 错误**，包括训练期间的非法内存访问，目前尚无明确解决方案。
   - 对于 **大模型的 VRAM 管理**（如 phi-3-mini），提出了 *flash attn2* 和 **RAG** 重构等技术来应对 OOM 场景。
- **数学标注的重要性日益增加**：关于是否需要 **数学数据标注** 以增强高级模型训练的讨论引发了对当前数据集中数学标注作用和现状的新研究。
   - 与此同时，社区寻求关于 **在 Next.js 上实现 Stable Diffusion** 的建议，引导使用 [diffusers.js](https://github.com/dakenf/diffusers.js) 及其他学习资源。
- **事物的形状：生成式 3D 学习**：通过对表示挑战的回顾，展示了深度学习在 **3D 形状生成** 方面的潜力，强调了 GANs 和形式表示方面的进展。
   - **时间序列预测** 准确性的提升得到了证实，NBEATSx 比其前身提高了 20%，特别是在电价预测方面表现显著。
- **将 AI 创意转化为工具**：一位名为 **Rose** 的 AI Vtuber 通过 [YouTube 直播](https://www.youtube.com/live/Le5O8Z8NiUY?si=b_kjhaE3qBKSQ8Po) 寻求社区测试，同时推出了一款利用 Groq API 的 whisper-large-v3 模型的 **快速字幕制作工具**。
   - 对于 Mac 爱好者，**适用于 Apple Silicon 的 Phi-3 Vision** 首次亮相，承诺提供优化后的性能，同时还推出了 **YouTube 视频转录工具** 以辅助内容创作者。
- **论文实现与 ConvNet 编年史**：针对寻求适合通过实现来学习的基础论文的请求，社区建议探索 self-attention 和隐式表示。
   - 在其他地方，探讨了 **Inception 模型** 过去在使用中间特征方面的声望，以及目前对 **ResNet** 的依赖。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth AI Beta 热议**：爱好者们讨论了在 NDAs 协议下的 **Unsloth AI beta** 测试、**多 GPU 支持** 的浮动许可，并推测了即将推出的功能。
   - 评论指出免费版缺乏多 GPU 支持，而订阅版正在开发中，部分测试人员已获得早期访问权限。
- **Karpathy 的 LLM 学习路径**：著名 AI 人物 Andrej Karpathy 发布了 **LLM101n 课程**，激发了对其新项目 [Eureka Labs](https://github.com/karpathy/LLM101n) 的讨论。
   - 该课程受到社区的热切期待，承诺涵盖 **Transformers 和 fine-tuning** 等广泛领域。
- **在 llama.cpp 中热插拔 LoRA**：llama.cpp 中的 **LoRA adapter 支持** 引发辩论，此前的一项更新实现了 adapter 的热插拔，以增强模型的灵活性。
   - 关于量化模型适配新 LoRA 的反馈褒贬不一，特别是涉及云端部署的可靠性。
- **辩论 RAG 与 Fine-Tuning**：关于使用 **RAG 还是 fine-tuning** 的效果展开了激烈辩论，大家认可 RAG 的便捷性，但在处理复杂任务时 fine-tuning 更有优势。
   - 一些人建议混合方法可能会产生更好的结果，表明训练方法正向更加个性化的方向转变。
- **AdamW-Mini 降低内存占用**：神经网络训练中的 **优化器状态开销** 引起讨论，观察到 AdamW-mini 可能将内存使用量减半。
   - 这可能允许 **增加一倍的 batch sizes**，标志着大规模训练效率的飞跃。

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **GPUs：艺术与失误**：一位用户展示了其对 GPU 的**玫瑰金翻新**，强调了硬件美学中常被低估的作用。
   - 与此同时，另一位成员承认了一个新手错误：忘记插上 **GPU power**（电源），这提醒所有人都要仔细检查自己的设备。
- **Mathstral：STEM 领域的新学霸**：**Mathstral** 在 LM Studio 的首次亮相引发了关注，与其 Mistral 7B 基座模型相比，它在 STEM 和高级推理能力方面表现出惊人的实力。
   - 它在**逻辑和数学问题**上的专长，配合 **bartowski** 提供的 GGUF 量化版本，使其成为寻求 AI 优势的技术人员眼中极具吸引力的工具。
- **DeepSeek 的引导出现偏差**：**DeepSeek-Coder V2-Lite** 的问题困扰着用户，其不稳定的响应完全无视 Prompt，表明可能与 LM Studio 的更新存在冲突。
   - 纠正其路径的尝试（包括禁用 flash attention）均未成功，成员们仍在寻找解决方案。
- **Fine-Tuning：潜在的“G”麻烦**：一位用户在对 **Codestral** 进行 Fine-Tuning 时遇到了困难，凸显了调整 LLM 的挑战，因为他们正纠结于模型产生的毫无意义的“G”响应。
   - 社区讨论建议，丰富的文档和利用集体智慧可能有助于应对这些 **Fine-Tuning 挫败感**。
- **为微决策选择合适的模型规模**：对于用于 **NER**（命名实体识别）和内容过滤等微决策的合适 LLM 的好奇，引发了关于推广更小、计算效率更高模型的讨论。
   - 频道内的专家强调了在**硬件设置**中进行优化配置的重要性，以增强模型在这些特定任务中的性能。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 最大化性能核利用**：讨论强调 **Mojo** 在 Intel 芯片上使用性能核（performance cores）来执行 `parallelize` 函数，尽管目前未利用能效核（efficiency cores），但仍优化了操作。
   - 运行时目前在核心利用决策上的局限性有望在即将到来的更新中得到增强，从而优化核心使用以提升性能。
- **NumPy vs Mojo：速度对决**：基准测试显示 **Mojo** 在速度上超越了 **NumPy**，尽管 Mojo 尚未利用所有可用核心，性能差距被归因于 BLAS 后端的选择。
   - 虽然 **OpenBLAS** 被广泛使用，但 **Intel MKL** 被公认为具有更卓越的速度，即使在非 Intel CPU 上也是如此。
- **Mojo 中的 Inline 创意**：有人建议为 `@always_inline("nodebug")` 提供一种简写形式，共识是 **Mojo** 中的 inline 函数应当保持简洁。
   - 这一语法提案旨在减少代码冗余，同时不牺牲清晰度或功能性。
- **超越双核：SIMD 和 SVE**：在 SIMD 背景下，**SVE** 处理非 2 的倍数大小的灵活性受到了关注，并探讨了利用清理循环（drainage loops）或掩码（masks）来增强性能的潜力。
   - 讨论围绕着优化技术展开，旨在提升跨不同架构的计算效率。
- **Mojo 编译器更新内幕**：最新的 **Mojo compiler** Nightly 版本 `2024.7.1714` 促使用户通过 `modular update nightly/mojo` 进行升级，其特点是包含了内置 SIMD 方法和 Dict 初始化等重大更新。
   - 这些变更在项目的 [GitHub changelog](https://github.com/modularml/mojo/commits) 中有详细说明，反映了该语言及其标准库的不断演进。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DCLM 震撼业界**：[DataComp for Language Models (DCLM)](https://arxiv.org/abs/2406.11794) 作为一个强大的测试平台脱颖而出，旨在通过受控的数据集实验来提升语言模型的效能。
   - **DCLM-Baseline-7B** 在 5-shot MMLU 准确率上比 MAP-Neo 高出 **6.6%**，展示了高效的计算利用率，详见 [Hugging Face 模型页面](https://huggingface.co/apple/DCLM-Baseline-7B)。
- **Replete-AI 的翻译突破**：**Replete-AI** 因推出一个包含超过 **280 万个数据点** 的开源 [多语言翻译数据集](https://huggingface.co/datasets/Replete-AI/Multi-lingual_Translation_Instruct) 而成为新闻焦点。
   - 该数据集涵盖了从英语到 **14 种语言** 的翻译，为多语言建模的进步奠定了基础。
- **Oxen.AI 邀请 LLM 思想家**：一篇富有见地的论文作者 Zhengxuan Wu 计划在 [Oxen.AI Paper Club](https://lu.ma/oxen) 活动中讨论 **Representation Finetuning**。
   - 关于 **ReFT** 的讨论因其与传统 PEFT 方法相比在优化方面的先锋性而备受关注。
- **信念状态几何 (Belief State Geometry) 揭秘**：一项新的 [信念状态几何研究](https://arxiv.org/abs/2405.15943) 揭示了 Transformer 如何在内部建模信念更新，引起了 LLM 社区的关注。
   - 关于这种残差流（residual streams）内的几何表示所带来的影响，反馈从赞赏到怀疑不等。
- **Hermes 2.5 展现基准测试实力**：在基准测试结果的轰动中，**Hermes 2.5** 在 MMLU 上取得了显著进步并保持领先，正如 [代码指令示例](https://link.to.examples) 所展示的那样。
   - 通过突触层面的改进，Hermes 2.5 的 MMLU 分数达到 **52.3**，标志着对其前代版本 **34.5** 分的重大突破。



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Pile 2 混淆已澄清**：澄清确认 **The Pile 2** 并不存在，并引导用户进行了更正。
   - 讨论转向了 Proof-Pile-2 数据集，详细说明其为一个包含 550 亿 token 的数学和科学文档集合，可在 [Hugging Face](https://huggingface.co/datasets/EleutherAI/proof-pile-2) 上找到。
- **抓取丑闻审查**：在 [Proof News 文章](https://www.proofnews.org/apple-nvidia-anthropic-used-thousands-of-swiped-youtube-videos-to-train-ai/) 发表后，未经许可使用 **YouTube 视频** 构建 AI 数据集的行为引发了辩论。
   - [Philosophy Tube](https://x.com/PhilosophyTube/status/1813227210569920685) 和 [Jacob Geller](https://x.com/yacobg42/status/1813226763117367688) 等艺术家发布了回应，引发了关于伦理和影响的讨论。
- **Transformer 工程探讨**：围绕 **Transformer 优化** 的辩论，特别是关于 [TransformerEngine 的融合层 (fused layers)](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/layernorm_linear.py)，揭示了一些被误解的功能。
   - 讨论强调了 RMSNorm 相比其他归一化技术在增强处理效率方面的潜力。
- **Arrakis 库解析**：介绍了 [Arrakis](https://github.com/yash-srivastava19/arrakis)，这是一个专为快速原型测试设计的机械可解释性（mechanistic interpretability）库，目前仍处于初期阶段。
   - 鼓励用户将其与 TransformerLens 等现有工具进行反馈和比较，以完善和验证 Arrakis 的独特功能。
- **排行榜合法性查询**：对 HF 排行榜上 musr 原始分数的计算方式提出询问；特别是它是否代表了特定任务的平均值。
   - 建议联系 [排行榜维护者](https://huggingface.co/leaderboard) 以澄清潜在的歧义。



---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **GPU 与巨型模型的博弈**：讨论揭示了 **VRAM 大小对模型性能至关重要**，大型模型需要消耗大量 VRAM，如果管理不当，可能会导致 **显存溢出 (OOM) 错误**。
   - 讨论强调了要区分生成时间延长与内存问题；生成时间较长并不自动意味着内存不足。
- **插画想象力的艺术化训练**：社区交流了关于训练独特插画风格（如排线技术）的见解，强调了 **区域提示词 (Regional Prompting)** 和 **多概念模型** 的重要性。
   - [HuggingFace 的 T5](https://huggingface.co/jtlicardo/flan-t5-small-coref) 等资源被视为这些艺术倾向训练尝试的重要工具。
- **挑剔的提示词产生奇特的图片**：关于微妙的提示词变化对结果影响的讨论非常热烈，例如“harvesting potato”与“potato harvesting”这类短语引发了关于模型 **指代消解 (Coreference)** 能力的讨论。
   - 爱好者们建议使用 T5 的微调模型，以熟练应对复杂提示词中的微妙差异。
- **外绘 (Outpainting) 带来无限可能**：探索了扩展生成图像的外绘方法，包括使用 Photoshop 工具和在 ComfyUI 中封装的 KSampler，以实现无缝的图像扩展。
   - 参与者分享了管理种子 (Seed) 一致性的方法，确保扩展后的视觉效果保持统一且没有重叠部分。
- **故障排除技巧解决技术难题**：使用 Automatic1111 的成员遇到了模型性能瓶颈，引发了针对特定硬件需求的 **命令行修复** 知识交流。
   - 提供了如 'xformers' 和 'medvram-sdxl' 等选项作为解决方案，以增强模型在入门级硬件配置机器上的效能。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **核函数困惑：模板平息 CUDA 灾难**：通过按照推荐的 CUDA 实践指定模板类型 `<int>`，克服了最初遇到的 **CUDA Kernel 调用错误**。
   - 实践经验：包含正确的模板参数可以决定 Kernel 是正常运行还是陷入令人沮丧的调试环节。
- **PyTorch Profiler 导出：马拉松还是短跑？**：当导出 trace 耗时超过 **30 分钟** 时，**PyTorch Profiler** 引发了辩论，导致了关闭 `profile_memory` 和 `with_stack` 选项等建议。
   - 成本效益分析：虽然可能会加快导出速度，但代价是可能失去详细的内存分配洞察。
- **CUDA 遇见 PyTorch：桥接自定义 Kernel**：**artificial_anteligence** 寻求关于将自定义 CUDA Kernel 与 PyTorch 集成的内容，特别是为了简化模型实现。
   - 框架间的交叉引用是必要的，一位社区成员强调了 `load_inline` 如何作为 Kernel 编译的起点资源。
- **Tensor 子类在 PyTorch Nightly 中纠缠**：使用 **unwrap_tensor_subclass** 带来了挑战，特别是当 **IntxTensor 子类** 作为 `layout_tensor` 时，GitHub 上的一个线程讨论了这些复杂情况 ([Issue #515](https://github.com/pytorch/ao/issues/515))。
   - 难题：嵌套子类可能会阻碍操作，使后端开发复杂化。
- **Triton 策略与谜题：简化执行**：**Triton Puzzle 6** 让工程师们对符号表示感到困惑，寻求关于涉及 **ReLU** 和矩阵-向量操作的函数定义的澄清。
   - 来自 'triton.runtime.interpreter' 的 'interpreter_builder' 出现 **ImportError**，成员们正在寻求稳定性，这突显了维护向后兼容性的关键性质。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **API 限制可能阻碍项目进度**：在 #[pplx-api] 频道中的讨论强调了对 **API rate limits**（速率限制）过于严格的担忧，这可能会影响项目的进度时间表。
   - 建议用户填写申请表并咨询 Perplexity 代表，以寻求缓解限制问题的解决方案。
- **Cloudflare CAPTCHA 遭到抨击**：#[general] 频道的成员对 **Cloudflare** 实施的 CAPTCHA 系统表达了不满，并对使用该系统的决策提出了质疑。
   - 社区反馈中包含了对 Cloudflare 安全问题的评论，其中一条评论指出 *Cloudflare 经常崩溃或被攻破*。
- **Perplexity API Beta 版解锁新过滤功能**：根据 #[pplx-api] 的讨论，Perplexity API 增加了一个有价值的功能——**`search_domain_name` 过滤器**，目前已对 Beta 用户开放。
   - 该功能支持更具针对性的搜索能力，允许在指定域名内进行增强的结果过滤。
- **质量困境：代码灾难受到质疑**：在 #[general] 频道中，一名成员提到某大公司的质量控制允许未经测试的代码进入生产环境，引发了关于行业惯例的坦诚对话。
   - *每家公司都这样，* 一位成员讽刺地强调，反映出对普遍存在的质量控制问题的无奈情绪。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **524 错误代码激增**：大量用户遇到了 **Error Code 524**，引发了关于该错误突然盛行的快速交流。
   - 随即出现了大量询问，调查这一异常现象是孤立案例还是普遍故障的征兆。
- **Meta 405B 的定价之谜**：随着用户推测 [Meta 405B 的潜在定价](https://discord.com/channels/1091220969173028894/1094454198688546826/1262737636607528972)，期待感不断升温，预计其将在 23 日左右首次亮相。
   - **8K context windows** 被作为过往模型的基准提出，而具体细节仍有待公布。
- **Deepseek Coder：强大但速度极慢**：“功能强大但速度极慢”概括了用户对 **Deepseek Coder** 的看法，其迟缓的性能让用户渴望更快的速度。
   - 不满的声音预示着市场机会，更敏捷的竞争对手可能会吸引那些被缓慢服务劝退的用户。
- **OpenRouter 寻求快速且廉价的 AI**：在寻找速度超越 **GPT-3.5-Turbo** 且价格低廉的模型时，用户在成本与上下文的权衡中考虑了 **Claude-3-Haiku** 等选项。
   - Llama 模型被视为这一追求中的有力竞争者，引发了关于何为“余速”与“廉价”的动态辩论。
- **在 WordPress 中集成 OpenRouter API 的困扰**：**RSS feed 集成困难**困扰着一位试图在 WordPress 环境中融合 OpenRouter API 的用户，引发了关于故障排除的讨论。
   - API key 的复杂性和 rate limit（速率限制）难题主导了讨论，`curl` 验证被推崇为技术检验的标准。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **模型之城的恶意操作**：[ComfyUI_LLMVISION](https://darknetdiaries.com/transcript/133/) 恶意软件瞄准了 **LAION 社区**，窃取数据并在毫无防备的受害者设备上安装木马。
   - 该黑客组织以入侵 Disney Slack 而闻名，展示了他们通过克隆 GitHub 工程师身份来伪造极具说服力的**虚假求职者**以进行数据窃取的能力。
- **桑迪飓风席卷电信业进入光纤时代**：**桑迪飓风 (Hurricane Sandy)** 摧毁了 **Verizon 的纽约电缆库**，迫使在 13,000 公里的范围内将铜缆更换为光纤。
   - 这一重大事件成为了基础设施升级的催化剂，正如这篇[深度解析](https://www.datacenterknowledge.com/cables/after-sandy-verizon-confronts-catastrophic-failure-at-ny-cable-vault)中所详述的那样。
- **视觉与语言在多模态舞台融合**：新型 [InternVL2-Llama3-76B](https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B) 在 **multimodal learning** 方面取得了飞跃，通过指令微调模型推向了新的边界。
   - 另外，社区中有人对在 **4x 3090 GPUs** 上运行 **large models** 表示沮丧，主要问题集中在 **autoawq** 的使用上。
- **Manifold 对机械化管理的思考**：**Manifold Research Group** 发布了一篇题为《[*大语言模型时代的智能数字代理*](https://www.manifoldrg.com/llm-agents/)》的立场论文，推动了关于 **LLM-based AI agents** 的讨论。
   - 他们邀请社区加入 [Discord](https://discord.gg/MfYZmYEGaa?ref=manifoldrg.com) 参与讨论，在 [Research Log #041](https://www.manifoldrg.com/research-log-041/) 中见证他们的进展，并为他们在 [GitHub](https://github.com/ManifoldRG/MultiNet/issues/19?ref=manifoldrg.com) 上的大型 **MultiNet** 项目做出贡献。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **证明与双关的游戏**：[OpenAI 的最新仓库](https://openai.com/index/prover-verifier-games-improve-legibility/)引入了 Prover-Verifier Games 以增强 **AI model legibility**，挑战了复杂性是一种“易读性税 (legibility tax)”的观点。
   - 社区交流认为这可以纠正模型在叙述上难以理解的问题，研究论文本身关于“*legibility tax*”的俏皮话也体现了这一点。
- **强化学习的奇特结果**：讨论围绕 **Reinforcement Learning (RL)** 如何调整模型特征展开，暗示复杂的图表可能会承担所谓的“*legibility tax*”。
   - 一位成员评论道，“*这张图表绝对是 legibility tax*”，指出了对 RL 独特影响的直接观察。
- **GPT-4：Tokenizer 探戈**：一场热烈的讨论对比了 **GPT-4o** 和 **Llama 405** 的 **tokenizers**，强调了 GPT-4o 在编程语言 Token 效率上相较于其前身 **GPT-4t** 的倒退。
   - 细节提到 GPT-4o 在处理 XML 时产生的 Token 比 GPT-4t 更多，标志着专用 **tokenizer** 性能的退步。
- **Deepseek 坚持开源路线**：在对中国 AI 发展速度的担忧中，**Deepseek 创始人梁文锋**表达了对开源的奉献精神，认为这对于构建强大的技术格局至关重要。
   - 尽管 Deepseek 利润微薄，但梁文锋的决心依然坚定，正如在[社交媒体上的一篇采访](https://x.com/main_horse/status/1813580480761196987?s=46)中所述。
- **策略模型中的采样混乱**：Nemotron 论文批评了策略模型中流行的采样方法，认为某些拒绝采样比其他的糟糕得多，从而为 **DPO** 算法带来了过拟合和质量损失的风险。
   - 与此同时，Zephyr 的论文提倡通过随机采样来促进多样性，旨在平衡挑战与 **DPO** 的目标，并避免因假阴性导致的错误方向。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **诺贝尔级别的基准测试：SciCode 表现卓越**：SciCode 建立了一个科学问题编程基准的新先例，其中包含向诺贝尔奖得主致敬的内容，这些问题难倒了 **GPT-4** 和 **Sonnet 3.5** 等巨头，准确率低于 5%。[深入了解](https://scicode-bench.github.io)。
   - 由博士专家编写的 **SciCode benchmark** 挑战包含 338 个问题，揭示了多个科学领域的现状。[此处查看见解](https://x.com/OfirPress/status/1813202497864937825)。
- **基于浏览器的 AI 杰作：HuggingFace 发布 SmolLM**：HuggingFace 推出了针对浏览器环境优化的 **SmolLM models**，支持 ONNX 和 WebGPU 加速。点击[此处](https://x.com/xenovacom/status/1813258097185448377)深入了解更新。
   - 新的 **SmolLM models** 范围从 135M 到 1.7B，专为高效的端侧 AI 应用设计，展示了先进的浏览器运行能力。
- **GPU 交易领域的开拓者：SF Compute 吸引投资**：**SF Compute** 完成了 1200 万美元的融资轮，将用于构建新型 GPU 交易平台。[详情](https://www.bloomberg.com/news/articles/2024-07-16/jack-altman-s-firm-backs-startup-for-trading-ai-computing-power)。
   - 这笔资金将促进大规模 GPU 集群的预订和交易，为计算资源分配引入流动性。
- **Exa AI 的扩张时代：A 轮融资助力增长**：在 Lightspeed、Nvidia 和 Y Combinator 等巨头的支持下，**Exa AI** 获得了 A 轮资金，以增强其由 LLM 驱动的搜索引擎 API。[探索更多](https://x.com/exaailabs/status/1813249325394456686)。
   - 尽管 Exa AI 正在扩张，但社区也在讨论关于 Prompt 优化以及与 **Perplexity** 等 API 进行基准测试的挑战。
- **利用 ColPALI 颠覆文档处理：高效检索愿景**：由 **HuggingFace** 推出的 ColPALI 承诺将带来文档检索的革命，使传统的 OCR 解决方案变得过时。[了解更多](https://huggingface.co/blog/manu/colpali)。
   - **HuggingFace 的 ColPALI** 提供了一种高效的文档处理方法，结合了视觉语言模型（Vision-Language Models）以实现更高的效率。[进一步讨论](https://x.com/jobergum/status/1813298149051802074)。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 展示其 Agent 能力**：一段[介绍视频](https://twitter.com/llama_index/status/1813316626793853135)展示了 **LlamaIndex** 的 Agent 能力，演示了 Python 和 TypeScript 框架，并提及了 LlamaParse 服务，其解析能力引发了热议。
   - 成员们称赞了 **LlamaParse** 的进步，强调了其新的基于 Markdown 的表格重构功能，以及在处理复杂表格方面的出色表现，详见[此推文](https://twitter.com/llama_index/status/1813355957491273936)。
- **探索查询时元数据（Query-time Metadata）的迷宫**：社区专家交流了在查询时应用元数据过滤器的想法，并权衡了不同的方法，质疑现有 Retriever 实例化方法的有效性。
   - 建议的解决方案和遗留问题的交织，展示了改进文档存储和索引并非易事。
- **Neo4J 属性图难题依然存在**：当 Neo4J 属性图无法记住重复实体时，社区侦探建议了潜在的修复方案，如实体链接（Entity Linking）调整。
   - 对话将理论与实践相结合，提到了 **'Entities'** 和 'MENTION' 关系以及 Cypher 查询片段，这可能为解决问题提供曙光。
- **Scaleport 同步精简的 AI 解决方案**：作为 LlamaIndex 多功能性的证明，Scaleport AI 利用 LlamaCloud 和 **LlamaIndex** 技术缩短了其 AI 开发周期并增强了 OCR 结果，详见[其案例研究](https://twitter.com/llama_index/status/1813647179627774462)。
   - **OCR 优化**和敏捷 AI 开发成为 **Scaleport AI** 案例中的主题，强调了将创新框架与客户项目结合的影响。
- **破解 CSV 混乱的代码**：关于在 VectorStoreIndex 中处理超过 50 行的 CSV 数据时遇到的困难引起了骚动，成员们剖析了错误并思考高效的解析路径。
   - 虽然 PagedCSVReader 表现不佳，但大家一致认为像 [PandasAI](https://docs.pandas-ai.com/intro) 这样的工具可能会为复杂的基于记录的 CSV 操作提供避风港和补救措施。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **CrunchCup 混乱：洗碗机耐用性存疑**：一位成员对新买的 [CrunchCup](https://www.amazon.com.au/CrunchCup-XL-Portable-Cereal-Spoon/dp/B08WYWQCZY) 感到兴奋，但尽管它在随时随地食用谷物方面非常方便，却因无法承受洗碗机的清洗循环而大打折扣。
   - 社区成员纷纷发表评论，评价从对其便携设计的赞赏到对其意外缺乏耐用性的沮丧不等，有人提到它在**机洗时会变形**。
- **Roger Grosse 讲座探讨 LLM 泛化**：Roger Grosse 的最新课程 *"Studying LLM Generalization through Influence Functions"*（通过影响函数研究 LLM 泛化）现已上线，分享的链接展示了[他在 YouTube 上的见解](https://youtu.be/64BsnVbX5u8)。
   - *danylo_boiko* 提醒成员通过直接的视频链接来了解**最新的 LLM 研究见解**。
- **Cohere 社区会议 YouTube 回顾**：对于错过会议的人，**Cohere 的社区活动演讲**（包括丰富的讨论和环节）现已在他们的 [YouTube 播放列表](https://www.youtube.com/playlist?list=PLLalUvky4CLJKDaiWCumhsJpHNDhZeVll)中提供。
   - 为了让公会保持更新，与会者被引导观看他们**喜爱的 AI 领军人物**的录像，并紧跟社区动态。
- **谷物大对决：是小孩子吃的吗？**：一场关于谷物偏好的趣味公会辩论引发了参与，**Fruit Loops** 和 **Special K** 成为焦点。
   - 虽然对于 Froot Loops 是否适合特定年龄段尚未达成共识，但对话凸显了工程师们早餐选择的多样性。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **定制化聊天机器人：是个性化推进还是隐私陷阱？**：关于使用 OpenAI API 等模型为特定网站**微调（fine-tuning）**自定义聊天机器人的辩论异常激烈，重点在于通过 *pre-prompting* 来嵌入公司知识。
   - **费用问题受到质疑**，在使用聊天机器人检测服务时，由于每月 20,000 美元的高额费用，建议采取人工审核等具有成本效益的措施。
- **从噪音中提取人声：播客的音频解决方案？**：关于从播客中进行**人声提取**工具的讨论浮出水面，重点关注了 Eleven Labs 的模型，因为它能够在无干扰的情况下分离声音。
   - 虽然这个话题优先级较低，但它为提高内容可访问性和从音频源中提取元数据开辟了途径。
- **学习的局限：GPT Agent 对上下文的把握**：对话探讨了 **GPT Agent** 的上下文限制，特别是由于固定的上下文窗口（context windows），它们在跟进持续讨论时显得力不从心。
   - 成员们交流了关于 **PUT 与 PATCH 请求**的技巧，并讨论了 **vector store embeddings**，强调了 RAG 聊天机器人在名称识别方面的挑战。
- **逆流而行：WebSurferAgent 的选择性搜索**：**WebSurferAgent** 因在搜索过程中偶尔忽略设置指令而引起关注，这表明在指令遵循方面仍有改进空间。
   - 一个共享的 ChatGPT **角色扮演（role-playing）**模板展示了在对话式 AI 中实现更具沉浸感、角色驱动交互的潜力。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Hannah 热潮：定制化 AI 助手**：介绍 **Hannah**，一款新型生成式 AI 助手，支持从文档中学习和**深度定制**等高级功能，并集成了从 OpenAI 到 NVIDIA 的 API。
   - 该助手由 **OpenAI**、**Anthropic** 和 **Cohere** 等热门 AI API 提供支持，相关信息可在 [Hannah 网站](https://hannah.yourbestseller.ai/)上找到。
- **MongoDB 与 LangChain 融合实现 Hybrid Search**：成员们正在寻求在 RAG 应用中将 **MongoDB** 作为向量数据库使用的指导，强调了对 **Hybrid Search** 功能的需求。
   - 虽然 [MongoDB 官方文档](https://mongodb.docs/hybridsearch)涵盖了 Hybrid Search，但社区对集成 LangChain 的见解需求量很大。
- **AI 助力爆款体育视频**：对能够创建 **YouTube shorts/TikTok 爆款体育短视频**的 AI 工具兴趣激增，社区成员正在寻求专业的剪辑见解。
   - 尽管对 AI 制作体育短片的能力持怀疑态度，用户仍在探索并请求针对此类内容生成的定制建议。
- **从非结构化到结构化：LangChain 的文档转换**：讨论围绕使用 `UnstructuredFileIOLoader` 及类似类将无序数据转换为可用的 LangChain 文档展开。
   - 通过分享的实际案例，用户正在利用 **LangChain 工具**来结构化数据，以提升应用性能。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Codestral 的代码征服**：Mistral AI 推出了 [**Codestral Mamba**](https://mistral.ai/news/codestral-mamba/)，凭借**线性时间推理（linear time inference）**和处理无限序列长度等特性，挑战代码生产力的前沿。
   - 由 Albert Gu 和 Tri Dao 开发，Codestral Mamba 激发了社区成员的浓厚兴趣，大家纷纷渴望测试其在**高级代码推理（advanced code reasoning）**方面的能力。
- **Mathstral：失踪模型之谜**：关于一个名为“Mathstral”的模型引起了广泛好奇，人们纷纷询问其是否存在以及是否与 Mistral AI 有关。
   - 目前讨论仍停留在猜测阶段，缺乏具体细节，这表明它可能是一个正在开发中的模型，或者是值得关注的未来项目。
- **抑制过拟合：寻找解决方案**：社区提出了对抗过拟合的建议，包括**增加 rank** 或**调整学习率（learning rates）**等策略，并根据模型的独特训练过程进行定制。
   - 数据集去重（de-duplicating datasets）等方法被作为防止模型在训练过程中过早过拟合的有效工具进行分享。



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **为 My Friend V1 手持硬件欢呼**：[@ParallaxAngle](https://x.com/ParallaxAngle/status/1805313161567818030) 发布的一条推文表达了对 **My Friend V1** 令人惊讶的紧凑外形的兴奋，并赞扬了 Based Hardware 团队的努力。
   - 用户称赞了产品的尺寸和质量，并用 *“LOVE LOVE LOVE my Friend”* 表达了喜爱之情。
- **AI Friend 的转录信任讨论**：针对通过 Open Interpreter 与 AI Friend 进行转录交互的隐私问题被提出，强调了在潜在集成中保密性的重要性。
   - 对话集中在如何利用 Open Interpreter 确保与 AI Friend 转录内容交互时的隐私，但具体的实现细节仍不确定。
- **Open Interpreter 的 Mac M3 芯片之谜**：关于 Open Interpreter 是否兼容 **M3 Mac** 的问题浮出水面，社区成员正在考虑 Linux 版本是否足够。
   - 非官方建议暗示，在针对文件路径等细节进行调整后，尝试运行 build.py 脚本可能会成功，但这尚未得到证实。



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune v0.2.0 发布**：[Torchtune v0.2.0](https://github.com/pytorch/torchtune/releases/tag/v0.2.0) 的发布带来了一系列新模型、recipes 以及 **sample packing** 等功能。
   - 该版本标志着来自**开源社区**的重大贡献，强调了改进该工具的协作努力。
- **LLAMA 3 的微调怪癖**：**LLAMA 3** 微调过程中出现了一个问题，即在生成过程中出现了 **finetune_right_pad_id** 标签，而不是预期的 `<|end_of_text|>`。
   - 从 **Torchtune nightly builds** 切换到稳定版本可能会提供临时修复，同时正在检查 tokenizer 的旧实现是否存在差异。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Linearizer 移除，更新上线**：在 tinygrad 移除 **linearizer** 后，出现了关于**更新笔记**的询问，凸显了社区对文档的关注。
   - 一位成员要求提供**修订后的笔记**，以反映重大更新后 tinygrad 的当前状态，这种对清晰度的呼声得到了回应。
- **颜色代码难题已澄清**：在追求**消息格式细微差别**的过程中，有人对成员笔记中出现的颜色代码寻求澄清。
   - 解决方案迅速达成，指引其查看位于**第一页底部**的颜色说明，确保不遗漏任何细节。



---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **OpenAI 到 LLM 实用工具的网关**：Kyle 确认 **OpenAI 端的访问权限**对于特定的 LLM 功能至关重要。
   - 这种访问权限可以实现更流线化的 LLM 应用，例如自动化医院账单检查。
- **计费领域的 LLM**：社区讨论集中在 **LLM** 在从 PDF 中提取规则以审计医院账单方面的潜力。
   - 考虑通过 LLM 进行 **Python 代码生成**，以简化账单验证过程。
- **错过参与的遗憾**：一位用户对 7 月 9 日之后没有查看 #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1262890983864008805) 频道表示遗憾，错过了重要的讨论。
   - 这种情绪强调了错过与关键频道更新和社区互动的机会。
- **合规性检查的代码建议**：有讨论关于利用 **LLM 生成的测试用例**来确保医院账单审计 Python 代码的可靠性。
   - 该倡议旨在充分利用 LLM 的能力，将其应用于现实场景中的实际应用。

---

## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **寻求流媒体成功的开发者**：[**Observe** 邀请](https://observeyourfuture.com)擅长 **HLS** 和 **WebRTC** 的开发者在 **Vanilla JS**、**TypeScript** 和 **MongoDB** 中施展编程才华。
   - 正在寻找对初创生态系统和现场流媒体技术挑战充满热情的后端开发大师。
- **初创之星：招募 TypeScript 人才**：后端专家请注意：**Observe** 需要你的 **TypeScript** 和 **MongoDB** 技能来创建无缝的流媒体解决方案。
   - 深入了解初创文化，并为 **HLS** 和 **WebRTC** 这一动态领域贡献你的技术专长。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Phoenix 2.0 携新功能起航**：不要错过 2024 年 7 月 18 日举行的 **Phoenix 2.0 产品更新与未来愿景**活动，届时将介绍托管部署和实验功能等新特性，作为 [Phoenix 2.0 发布](https://arize.com/resource/phoenix/2.0)的一部分。
   - 与会者将一窥 **Phoenix** 在 Arize 产品栈中的演进，并参与实时问答环节，加深对该工具在 LLM 应用开发中潜力的理解。
- **OSS：AI 进步的支柱**：一场关于 **AI 领域 OSS 的市政厅会议**将详细阐述 **Phoenix 2.0** 如何通过新实验功能等特性简化开发，以及开源软件 (OSS) 在 AI 中的关键作用。
   - 用户体验见解是议程的一大亮点，强调了社区反馈与 Phoenix 功能演进之间的协同作用。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **异步响应觉醒**：AI21 Labs 的 Python SDK 现在包含**异步客户端支持**，并兼容 **Amazon Bedrock** 和 **Azure AI Studio** 等平台上的 **Jamba-Instruct**。
   - 鼓励开发者探索[最新 GitHub 版本](https://github.com/AI21Labs/ai21-python)中提供的新功能集，其中还展示了新的示例以提供更好的开发体验。
- **客户端并发准备就绪**：**异步客户端支持**现在是所有界面上 **Jamba-Instruct** 的标准功能，提供了增强的性能。
   - 如需实操指导，开发者可以访问 [AI21 Labs 的 GitHub 仓库](https://github.com/AI21Labs/ai21-python)获取新的 **Jamba-Instruct 示例**，以快速启动他们的应用。

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道详细摘要和链接

{% if medium == 'web' %}

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1262846226852876328)** (286 条消息🔥🔥): 

> - `Model Tokenization`
> - `CUDA Errors`
> - `VRAM Management for Large Models`
> - `Math Data Annotation Needs`
> - `Live Translation with Transformers` 


- **CUDA Errors 困扰训练过程**：一位用户在训练期间遇到了持续的 **CUDA errors**，提示信息为 *'CUDA error: an illegal memory access was encountered'*。尽管尝试了应用建议的修复方法，该问题仍未解决。
- **大模型的 VRAM 管理**：用户讨论了运行 **phi-3-mini-128k** 等模型时的 **VRAM usage** 管理策略，该模型在约 50k 上下文时遇到了 OOM 问题。建议包括使用 *flash attn2*，以及可能将架构重组为 **RAG or summarization** 方法。
- **数学数据标注的需求**：关于在训练 LLMs 等高级模型时对 **math data annotation** 需求是否日益增长的询问引发了社区对该话题的关注。一位用户发起了一项研究，以了解 **annotated math data** 的重要性和当前的可用性。
- **在 Next.js 中实现 Stable Diffusion**：一位用户询问如何在 **Next.js** 中使用 **Stable Diffusion**，随后收到了 [diffusers.js GitHub repository](https://github.com/dakenf/diffusers.js) 的推荐。其他成员对视频教程等更多资源表示了兴趣。
- **用于 CSV 数据查询的聊天机器人**：围绕开发一个能高效处理 **CSV data queries** 的聊天机器人展开了讨论。建议包括使用 **Llama 3** 等模型，并将聊天机器人与 **pandas** 集成，以实现对表格数据的函数式查询。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/tomaarsen/gliner_medium-v2.1">GLiNER-medium-v2.1, zero-shot NER - tomaarsen 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://github.com/dakenf/diffusers.js">GitHub - dakenf/diffusers.js: 适用于 node.js 和浏览器的 diffusers 实现</a>：适用于 node.js 和浏览器的 diffusers 实现。可以通过创建 GitHub 账户为 dakenf/diffusers.js 的开发做出贡献。</li><li><a href="https://www.nibbletechnology.com/demo">Nibble 演示</a>：打赌我们能让你微笑！快来试试我们用于电子商务的 AI 谈判聊天机器人，亲身体验它如何推动转化和参与度</li><li><a href="https://tenor.com/view/tf2-gif-11289716861894092888">Tf2 GIF - Tf2 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/spaces/aheedsajid/Edge-TTS/discussions/1#6696e19ca7fd582ae724f59f">aheedsajid/Edge-TTS · 🚩 报告：垃圾内容</a>：未找到描述</li><li><a href="https://www.evesleep.co.uk/products/the-premium-hybrid-mattress">高级混合床垫 - 28cm 弹簧与泡沫</a>：选购高级混合床垫，结合了袋装弹簧设计与泡沫技术的舒适感，享受夜晚的奢华。提供 365 天试用，免费送货且退货无忧。</li><li><a href="https://github.com/idiap/fast-transformers/issues/19">RuntimeError: CUDA error: an illegal memory access was encountered · Issue #19 · idiap/fast-transformers</a>：你好，感谢这项出色的工作！我成功安装了该包，但在训练期间遇到了错误：File &quot;/home/annahung/189nas/2020/fast_remi/linear_transformer/model.py&quot;, line 294, i...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1262982859522838589)** (3 条消息): 

> - `SciPy tutorial`
> - `Audio course on Huggingface`
> - `Real-time kernels and Raspberry Pi` 


- **SciPy：不仅仅是另一个库**：分享了一个名为 [Intro to Scipy by Rauf](https://youtu.be/KAbNQwTBEyc?si=UorOWv5tJIPiCUYW) 的 YouTube 视频，将 **SciPy** 介绍为类似于 **NumPy** 的数据处理和科学计算库，但具有一些差异。
- **深入学习 Huggingface 音频课程**：一位用户提到开始学习 **Huggingface audio course**，并征求关于 **TTS (Text-to-Speech)** 和 **ASR (Automatic Speech Recognition)** 的宝贵建议。
- **对 Raspberry Pi 上的实时内核感到沮丧**：一位用户表达了对 **Raspberry Pi 4** 上的 **real-time kernels**（如 rt-thread 和 freeRTOS）的沮丧，指出其与编译器的兼容性问题。
   - 由于**当前设置**的限制，他们正考虑从头开始编写一个带有 USB 和 HDMI 外设的 **kernel**。



**提到的链接**：<a href="https://youtu.be/KAbNQwTBEyc?si=UorOWv5tJIPiCUYW">SciPy 简介 ( by Rauf )</a>：SciPy 是另一个类似于 NumPy 的数据处理和科学计算库，但有一些不同之处。它是你工具箱中的另一个工具，允许...

  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1262947774433857576)** (4 条消息): 

> - `Nbeats 和 NBeatsX 论文`
> - `基于深度学习的 3D 形状生成`
> - `时间序列预测`
> - `机器学习在 3D 几何中的应用` 


- **NBeatsX 扩展了 NBeats 模型的能力**：一篇 [论文](https://arxiv.org/abs/2104.05522) 将 NBEATS 扩展为 NBEATSx，通过引入外生变量（exogenous factors）以提升时间序列预测性能，相比原始模型实现了近 **20% 的提升**。
   - 该研究通过整合多种有用信息源，在电价预测 (EPF) 领域展现了 **state-of-the-art 性能**。
- **深度学习在 3D 形状生成方面表现出色**：一篇 [较早的文章](https://www.sciopen.com/article/10.1007/s41095-022-0321-5) 综述了深度学习在 3D 形状生成中的应用，强调了由于 voxels、point clouds 和 meshes 等不同表示形式带来的挑战。
   - 该论文讨论了 GANs 等深度生成模型的进展，并强调了形状表示对于高质量 3D 形状生成的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2104.05522">Neural basis expansion analysis with exogenous variables: Forecasting electricity prices with NBEATSx</a>: 我们将神经基扩展分析 (NBEATS) 扩展到包含外生因素。由此产生的方法称为 NBEATSx，改进了一个性能良好的深度学习模型，扩展了其能力...</li><li><a href="https://www.sciopen.com/article/10.1007/s41095-022-0321-5">A survey of deep learning-based 3D shape generation</a>: &lt;p&gt;深度学习已成功用于 2D 图像领域的任务。3D 计算机视觉和深度几何学习的研究也引起了关注。已经取得了相当大的成就...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1262885361311416493)** (24 条消息🔥): 

> - `AI Vtuber 测试`
> - `智利旅游数据`
> - `适用于 Mac 的 Phi-3 Vision`
> - `用于 3D 模型简化的 ML`
> - `快速字幕生成器` 


- **AI Vtuber 需要测试者**：分享了一个名为 "chatting/lofi with Rose! (AI Vtuber) [open source] pls test it lol" 的 [YouTube 视频](https://www.youtube.com/live/Le5O8Z8NiUY?si=b_kjhaE3qBKSQ8Po) 供社区测试。
- **智利旅游数据可用**：分享了 [智利旅游数据](https://huggingface.co/datasets/RaulSalinasHerr/chilean_touristic_data) 的链接，可用于测试和分析。
- **适用于 Apple Silicon 的 Phi-3 Vision**：[Phi-3 for Mac](https://github.com/JosefAlbers/Phi-3-Vision-MLX) 在 GitHub 上发布，为 Apple Silicon 提供本地运行的 Vision 和 Language Models，供社区使用。
   - 该工具专为 Apple Silicon 的无缝性能而设计，对在 Mac 上工作的开发者很有吸引力。
- **视频快速字幕生成器**：[Fast Subtitle Maker](https://huggingface.co/spaces/Nick088/Fast-Subtitle-Maker) 可以利用 Groq API 的 whisper-large-v3 模型快速生成字幕，适合缺乏高性能 PC 的用户。
   - 用户可以获取字幕文件，或直接将字幕嵌入视频，并支持自定义字体和颜色等设置。
- **YouTube 视频转录工具**：分享了一个使用 Deepgram 和 Claude 转录和总结 YouTube 视频的工具，对内容创作者和研究人员非常有用。
   - 用户可以 [尝试该工具](https://app.hunch.tools/app/tool/yB85W?tpreview=true&invitationCode=u54c55ff)，自定义模板，并阅读相关的 [博客文章](https://hunch.tools/blog/video-transcription-and-summary-tool/)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/Nick088/Fast-Subtitle-Maker">Fast Subtitle Maker - Nick088 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://github.com/JosefAlbers/Phi-3-Vision-MLX">GitHub - JosefAlbers/Phi-3-Vision-MLX: Phi-3 for Mac: Locally-run Vision and Language Models for Apple Silicon</a>: Phi-3 for Mac: 适用于 Apple Silicon 的本地运行 Vision 和 Language Models - JosefAlbers/Phi-3-Vision-MLX</li><li><a href="https://app.hunch.tools/app/tool/yB85W?tpreview=true&invitationCode=u54c55ff)">Hunch - 团队 AI 工具</a>: 创建 AI 工作流和工具以自动化知识工作并提高团队生产力</li><li><a href="https://www.youtube.com/live/Le5O8Z8NiUY?si=b_kjhaE3qBKSQ8Po">chatting/lofi with Rose! (AI Vtuber) [open source] pls test it lol</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/jbilcke-hf/ai-comic-factory">AI Comic Factory - jbilcke-hf 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://app.hunch.tools/app/canvas/new/vyg7V?invitationCode=u54c55ff)">Hunch - 团队 AI 工具</a>: 创建 AI 工作流和工具以自动化知识工作并提高团队生产力</li><li><a href="https://huggingface.co/datasets/RaulSalinasHerr/chilean_touristic_data">RaulSalinasHerr/chilean_touristic_data · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1262965178648100905)** (6 条消息): 

> - `通过实现论文进行学习`
> - `Inception 模型与 ResNet`
> - `隐式表示 (Implicit Representation)` 


- **选择学习用的论文**：一位成员表达了通过从零开始实现论文来进行学习的兴趣，并征求入门论文的建议。
   - 另一位成员推荐了自己的工作，作为学习隐式表示 (Implicit Representation)、Self-attention、Channel-attention 等概念的良好起点。
- **中间特征与模型**：讨论强调了 **Inception 模型** 是一种多分支架构，并在 ResNet 出现之前利用中间特征来辅助分类。
   - 根据一位成员的评论，*Inception 模型利用中间特征来辅助分类*。


  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1262967896733122600)** (3 messages): 

> - `Skin Cancer Classification Model`
> - `VQModel Pre-trained Weights`
> - `Attention Extraction from GhostNetV2` 


- **分享皮肤癌分类项目**：一名成员分享了他们使用 CNN 进行 **Skin Cancer Classification** 项目的 [GitHub 仓库](https://github.com/Matthew-AI-Dev/AI-Portfoilio/blob/master/SkinCancerClassification_CNN/SkinCancerClassification.ipynb) 链接。
   - 该项目包含一个用于皮肤癌图像分类的 notebook 和详细描述。
- **咨询 VQModel 预训练权重**：一名成员询问是否有人知道 **VQModel 预训练权重** 的获取渠道。
- **从 GhostNetV2 提取 Attention**：一名成员在利用 **timm 库** 时，寻求关于从 **GhostNetV2** 提取 Attention 特征的帮助。
   - 他们曾尝试使用 `timm.utils` 中的 `AttentionExtractor` 但发现没有效果，目前正在寻求进一步的协助。



**Link mentioned**: <a href="https://github.com/Matthew-AI-Dev/AI-Portfoilio/blob/master/SkinCancerClassification_CNN/SkinCancerClassification.ipynb">AI-Portfoilio/SkinCancerClassification_CNN/SkinCancerClassification.ipynb at master · Matthew-AI-Dev/AI-Portfoilio</a>: 通过在 GitHub 上创建账号来为 Matthew-AI-Dev/AI-Portfoilio 的开发做出贡献。

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1262877813644722176)** (3 messages): 

> - `System requirements for stability AI model`
> - `Prompt engineering for video generation`
> - `Stable Video Diffusion Image-to-Video Model` 


- **询问 AI 模型的系统要求**：一名用户咨询了运行 **Stability AI 模型** 的系统要求。
- **高效视频生成的 Prompt Engineering**：一名用户寻求关于使用 **图生视频 Stability AI 模型** 创作优质视频的理想 Prompt Engineering 方案。
   - 提出的一个具体问题是：“我应该在火箭图像中加入什么 Prompt 才能让它变成一段火箭移动的视频？”
- **分享 Stable Video Diffusion 图生视频模型**：一名用户分享了 HuggingFace 上 [Stable Video Diffusion 图生视频模型](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) 的链接。
   - 该模型可以将静态图像转换为视频，支持生成分辨率为 **576x1024**、最高 **25 帧** 的内容。



**Link mentioned**: <a href="https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt">stabilityai/stable-video-diffusion-img2vid-xt · Hugging Face</a>: 未找到描述内容

  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1262850045858943117)** (164 条消息🔥🔥): 

> - `Unsloth AI 签署 NDA 的 Beta 测试`
> - `支持多 GPU 的浮动许可证`
> - `Andrej Karpathy 的新 LLM101n 课程`
> - `llama.cpp 中的 LoRA 适配器支持`
> - `Llama-3 中的 Fine-tuning vs. RAG` 


- **Unsloth AI 在 NDA 下进行 Beta 测试**：**几位用户**讨论了在 NDA 下获得 Unsloth AI 测试版访问权限的情况，并提到了技术细节以及支持多 GPU 的浮动许可证。
   - 免费版不支持多 GPU 使用，但支持多 GPU 的付费订阅版正在开发中，部分用户已获得早期访问权限。
- **Andrej Karpathy 发布 LLM101n 课程**：Andrej Karpathy 宣布了他的新课程 LLM101n，涵盖了 **Bigrams、Transformers 和 Fine-tuning** 等主题，这是他新公司 [Eureka Labs](https://github.com/karpathy/LLM101n) 的一部分。
   - 该课程预计将包含创新的 Pretraining 技术，并将在网上发布，计划设有数字和实体学习小组。
- **llama.cpp 支持 LoRA 适配器热插拔**：[llama.cpp](https://github.com/ggerganov/llama.cpp/pull/8332) 的最新更新包括 **热插拔 LoRA 适配器**，这可能会提高模型的通用性。
   - *评价褒贬不一*：关于量化模型适配新 LoRA 的有效性和可靠性，评价不一，特别是在云环境中。
- **RAG vs. Fine-tuning：哪个更好？**：用户辩论了 **RAG (Retrieve and Generate) 与 Fine-tuning** 在特定任务中的优劣，指出 RAG 实施更快，但结果往往较差。
   - 建议将 **Fine-tuning 与 RAG 结合使用** 以获得更好的效果，尽管这涉及更广泛的定制和训练。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/sample-contract-nda-non-disclosure-agreement-gif-17773157">示例合同 GIF - 示例合同 NDA - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/danielhanchen/status/1813330269044408612">Daniel Han (@danielhanchen) 的推文</a>: 如果你想学习 LLM 的所有基础知识，千万不要错过 Andrej 的 LLM101n 课程！章节：1-2-3: Bigrams, N-grams, Backprop, ML, Maths 4-5: Attention, Transformers, GPT2 6: ...</li><li><a href="https://www.deeplearning.ai/short-courses/pretraining-llms/">Pretraining LLMs</a>: 深入了解预训练 LLM 的步骤，包括数据准备、模型配置和性能评估。</li><li><a href="https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama#id-6.-alpaca-dataset">如何微调 Llama-3 并导出到 Ollama | Unsloth 文档</a>: 为在 Ollama 上本地运行而创建定制化个人助手（如 ChatGPT）的初学者指南</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/8332">ngxson 重构 LoRA 适配器支持 · Pull Request #8332 · ggerganov/llama.cpp</a>: 此次重构灵感来自 Control Vector 的实现，它对 GGUF 和设备缓冲区有良好的支持。在此 PR 中：重构 LoRA API，允许热插拔 LoRA，添加了 struct llama_lo...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1262861610654109797)** (10 条消息🔥): 

> - `Codestral Mamba 发布`
> - `Mathstral 发布`
> - `Llama.cpp 支持问题`
> - `Google FlAMe 24B 模型`
> - `Llama 3 context 细节` 


- **Mistral 发布代码和数学模型**：发布了两个不错的模型，[Mamba-Codestral-7B-v0.1](https://huggingface.co/mistralai/mamba-codestral-7B-v0.1) 和 [Mathstral-7B-v0.1](https://huggingface.co/mistralai/mathstral-7B-v0.1)，采用 **Apache 2.0** 协议，具有 **32k context**。
   - 代码模型 Mamba-Codestral-7B-v0.1 目前尚未被 **Llama.cpp** 支持，但已在 [Llama.cpp](https://github.com/ggerganov/llama.cpp/issues/8519) 开启 issue 跟踪支持进度。
- **Google FlAMe 24B 模型表现优于其他大模型**：来自 Google 的新模型（被称为 **FlAMe 24B**）据报道在 Benchmark 上的表现优于现有的模型，详见此 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1e5118i/new_model_from_google_flame_24b/)。
   - 成员们对潜在的 **overfitting** 表示怀疑，但也承认该模型的性能前景广阔。
- **来自 Meta 的 Llama 3 发布见解**：Meta 的 GenAI 产品总监 Joe Spisak 解释说，**Llama 3** 最初计划作为“预发布”或“预览版”，但 **Mark Zuckerberg** 推动了其正式发布，导致初始版本仅有 8k context。
   - 这表明未来预计会有更多功能和改进，如 [此 Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1e55u8h/this_went_under_the_radar_joe_spisak_product/) 所示。
- **Kaggle Notebook 会话处理问题**：一位成员遇到了 Kaggle Notebook 会话在笔记本电脑进入睡眠状态时停止的问题，导致他们必须重新运行漫长的训练过程。
   - 他们询问是否有办法保存微调后的模型，以避免在会话停止后重新训练。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/mistralai/mathstral-7B-v0.1">mistralai/mathstral-7B-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/mistralai/mamba-codestral-7B-v0.1">mistralai/mamba-codestral-7B-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/8519>">Issues · ggerganov/llama.cpp</a>：C/C++ 中的 LLM 推理。通过在 GitHub 上创建账户为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e5118i/new_model_from_google_flame_24b/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e55u8h/this_went_under_the_radar_joe_spisak_product/">Reddit - Dive into anything</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1262871485157081251)** (35 条消息🔥): 

> - `模型预训练问题`
> - `Unsloth 的 CUDA 兼容性`
> - `Kaggle 上的微调挑战` 


- **预训练期间的重复词问题**：一位成员在继续预训练模型时遇到了单词重复的问题。解决方案是在数据集的每个文档末尾追加 **EOS token**，这解决了该问题。
   - *Theyruinedelise* 和 *Will007* 提供了见解，强调即使在预训练期间也有必要追加 **EOS token**。
- **CUDA 兼容性和 Unsloth 安装问题**：一位成员询问在 Windows 上配合 **CUDA 12.5** 使用 Unsloth 的情况，但得到的澄清是 Unsloth 最高支持 **CUDA 12.1**，且在 Windows 上需要 **WSL**。
   - *Edd0302* 指出最新的 Torch 仅支持到 **CUDA 12.4**，并建议使用 **WSL** 以获得更好的兼容性。
- **在 Kaggle T4 GPU 上微调 Phi 3 mini**：一位成员尝试在免费的 Kaggle 环境中使用 T4 GPU 微调具有 **4k context** 的 **Phi 3 mini**，但由于显存限制而遇到困难。
   - 虽然提供了减少 **batch size**、**epochs** 以及杀掉 kernel 重新执行等建议，但显存问题依然存在。


  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1263045317558009887)** (1 条消息): 

> - `Ghost 8B Beta`
> - `专有模型：xAI Grok 1, OpenAI GPT 3.5, Mistral Mixtral 8x7B`
> - `模型评估：zero-shot 方法`
> - `Claude 2 和 Claude 3`
> - `Ghost 8B Beta 体验区` 


- **Ghost 8B Beta 脱颖而出成为领导者**：[Ghost 8B Beta](https://ghost-x.org/docs/models/ghost-8b-beta) 的性能超越了 **xAI Grok 1**、**OpenAI GPT 3.5** 和 **Mistral Mixtral 8x7B** 等专有模型，巩固了其作为顶级语言模型的地位。
   - 它独特地采用 **zero-shot 方法** 进行评估，并与 **Claude 2** 和 **Claude 3** 进行对比，突显了其突破性的能力。
- **体验 Ghost 8B Beta**：鼓励成员在 [Hugging Face spaces](https://huggingface.co/spaces/lamhieu/ghost-8b-beta-8k) 上测试 Ghost 8B Beta 的 **8k** 和 **128k** token 上下文版本。
   - 官方文档和详细评估可以在 [官方网站](https://ghost-x.org/docs/models/ghost-8b-beta) 找到。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/lamhieu/ghost-8b-beta-8k">Ghost 8B Beta (β, 8k) - a Hugging Face Space by lamhieu</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/lamhieu/ghost-8b-beta-128k">Ghost 8B Beta (β, 128k) - a Hugging Face Space by lamhieu</a>: 未找到描述</li><li><a href="https://ghost-x.org/docs/models/ghost-8b-beta/">Ghost 8B Beta</a>: 开发该大语言模型的目标包括出色的多语言支持、卓越的知识能力和成本效益。
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1262860756014202940)** (50 条消息🔥): 

> - `神经网络中的内存使用优化`
> - `关于 AdamW-Mini 优化器的讨论`
> - `训练效率`
> - `优化器状态的开销`
> - `处理 Excel 中多个表格的策略` 


- **比 AdamW 减少 50% 的内存占用是否有意义？**：成员们讨论了在神经网络中比 AdamW **减少 50% 的内存占用** 是否具有重大意义，以及 **其对大规模训练的影响**。
   - _*一位成员称其为“过去 4-5 年机器学习领域最大的发现之一”*_，并辩论了涉及的实际 VRAM 节省和优化器开销。
- **优化器开销对训练的重大影响**：**优化器状态（Optimizer state）** 的开销主导了训练，有说法称 **AdamW 的开销是梯度的 3 倍**，这影响了 VRAM 的预算分配。
   - 根据讨论，这可能导致 **批量大小（batch sizes）翻倍**，这是训练效率的一次重大飞跃。
- **AdamW-Mini 优化器效率得到确认**：**AdamW-Mini** 可能会带来大约 **50% 的内存节省**，成员们讨论了它与 AdamW 等现有优化器的区别。
   - 讨论中也对开销分布以及在使用 RoPEd 缩放训练 llama-3-70b 等大型数据集时的影响提出了担忧。

---

### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1262852422745980992)** (83 条消息🔥🔥): 

> - `Llama 3 8B on GPU`
> - `Mistral mamba code model`
> - `Troubleshooting Huge Text Files`
> - `Model Loading Issues`
> - `Codestral Mamba in LM Studio` 


- **Llama 3 8B 全量在 GPU 上运行**：一位成员提到 **Llama 3 8B** 模型可以完全在 GPU 上运行，并用兴奋的表情符号 `😍 😭` 和 `😨` 回应了性能问题。
- **llama.cpp 中的 Mistral mamba 集成**：有人询问关于 **Mistral mamba code model** 的支持情况，讨论指出预计未来会有对 llama.cpp 的贡献。
- **在 AI 中处理大型文本文件**：成员们讨论了如何使用模型处理巨大的文本文件，建议使用 **grep** 和 **awk** 等工具进行数据预处理，因为目前存在上下文窗口大小的限制。
- **Nvidia 和 AMD GPU 的模型加载问题**：多位成员报告了由于 **VRAM 和 RAM 限制** 导致模型加载失败的问题，特别是在像 **Tesla K80** 这样的旧型号 GPU 上。
- **Codestral Mamba 支持的未来**：成员们对 LM Studio 加入 **Codestral Mamba** 表示好奇，这取决于 llama.cpp 何时添加相关支持。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://medium.com/@kappei/topic-modeling-made-easy-with-large-language-models-3af3d2375500">Topic Modeling Made Easy with Large Language Models</a>：主题建模长期以来一直是一项复杂且耗时的工作，需要大量的专业知识。然而，随着大语言模型的兴起……</li><li><a href="https://huggingface.co/bartowski/NuminaMath-7B-TIR-GGUF">bartowski/NuminaMath-7B-TIR-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://news.ycombinator.com/item?id=40977103">Codestral Mamba | Hacker News</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/6849">Support for Phi-3 models · Issue #6849 · ggerganov/llama.cpp</a>：微软最近发布了 3 个变体（mini, small &amp; medium）的 Phi-3 模型。我们可以添加对这个新模型系列的支持吗？
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1262848109071958078)** (107 条消息🔥🔥): 

> - `META 3 7B Q8 instruct`
> - `LLava 3 testing`
> - `LLM suggestions for micro decisions`
> - `DeepSeek-Coder V2-Lite issues`
> - `Fine-tuning models locally` 


- **LLava 3 在测试中表现出色**：一位用户分享了他们测试 [LLava 3](https://link.to.llava3) 的经历，表示运行良好。
   - 该模型表现符合预期，在用户的测试中展示了可靠的结果。
- **DeepSeek-Coder V2-Lite 的困扰**：成员们报告了 [DeepSeek-Coder V2-Lite](https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite) 的问题，该模型不遵循 Prompt 且回答不稳定。
   - 禁用 flash attention 并没有解决问题，这表明 LM Studio 的更新可能破坏了对其的支持。
- **微决策 LLM 推荐**：一位用户询问适合 **NER**（命名实体识别）、内容过滤和二元决策任务的 LLM；建议包括一些较小的模型。
   - 一些用户建议考虑计算效率高的模型，以便在离散任务中获得更好的性能。
- **本地微调 LLM 的挑战**：一位用户表达了在系统上微调 **Codestral** 时遇到的问题，提到了重复出现 'G' 响应的情况。
   - 使用充足的硬件在本地微调 LLM 是可行的；建议参考文档和社区资源。
- **优化微调的硬件配置**：用户讨论了高效微调的配置，争论了像 **RTX 4090** 这样的硬件以及具有高 RAM 和 VRAM 的笔记本电脑。
   - 确保兼容性和正确的设置（包括禁用 flash attention 等功能）对于成功进行微调至关重要。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF">bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/bartowski/Codestral-22B-v0.1-GGUF/tree/main">bartowski/Codestral-22B-v0.1-GGUF at main</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1262858759093616650)** (14 条消息🔥): 

> - `Gemma 2 支持`
> - `Phi 3 small 支持`
> - `Llama.cpp 支持`
> - `模型加载错误`
> - `Smol-lm 预分词器 (pre-tokenizer) 问题` 


- **Gemma 2 已支持，但 Phi 3 small 尚未支持**：成员们讨论了 **Gemma 2** 将获得支持，但 **Phi 3 small** 由于 **llama.cpp** 缺乏支持而暂不支持。
- **关于“模型加载错误”消息的关注**：一名成员遇到了“Error loading model”错误，并被引导至 [support 频道](<#1111440136287297637>) 分享更多细节和系统信息。
- **Smol-lm 模型在 llama.cpp 中不受支持**：分享了一个关于预分词器类型 'smol-lm' 的错误，目前 LM Studio 使用的 **llama.cpp** 版本尚不支持该类型。


  

---


### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/)** (1 条消息): 

pashtett: 有没有针对基于故事的 RP 聊天，在 Gemma 2 上运行的最佳 Prompt 和设置示例？
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1262893193771487345)** (4 条消息): 

> - `GPU 工艺`
> - `美学的重要性`
> - `GPU 电源插头` 


- **GPU 支撑架喷涂成玫瑰金**：一位用户分享了将 GPU 支撑架喷涂成**玫瑰金**的经历，因为他们认为**美学非常重要**。
   - GPU 设置中的“工艺感”有时就体现在这些细致的美学选择上。
- **忘记插 GPU 电源**：一位用户承认在组装过程中**忘记插上 GPU 电源**。
   - 这种小失误在硬件组装中经常发生，提醒大家要仔细检查连接情况。


  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1262858589660516404)** (1 条消息): 

> - `Mathstral 发布`
> - `STEM 专业化`
> - `GGUF 量化` 


- **Mistral 的 Mathstral 剑指巅峰**：宣布**专注于 STEM 和高级推理**，Mathstral 模型在多个主要 STEM 类别中的表现远超基础版 Mistral 7B。
   - 访问 [Mathstral 模型](https://huggingface.co/lmstudio-community/mathstral-7B-v0.1-GGUF) 获取更多详细信息，并加入 [Discord](https://discord.gg/aPQfnNkxGC) 参与讨论。
- **Mathstral 模型摘要与创新**：[Mathstral-7B-v0.1](https://huggingface.co/mistralai/mathstral-7B-v0.1) 是一个微调模型，旨在通过复杂的多步逻辑推理解决高级数学问题。
   - 由 [bartowski](https://huggingface.co/bartowski) 使用 `llama.cpp` 版本 [b3389](https://github.com/ggerganov/llama.cpp/releases/tag/b3389) 进行的 **Quantization (量化)** 优化了模型性能。



**提到的链接**：<a href="https://huggingface.co/lmstudio-community/mathstral-7B-v0.1-GGUF">lmstudio-community/mathstral-7B-v0.1-GGUF · Hugging Face</a>: 未找到描述

  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1262848440103211039)** (50 条消息🔥): 

> - `Mojo 社区会议`
> - `CFFI 与 C++ 互操作性`
> - `使用 DLOpen 进行外部链接`
> - `支持工单系统` 


- **Mojo 第 4 次社区会议提供了宝贵的见解**：[第 4 次 Mojo 社区会议](https://www.youtube.com/watch?v=_QVs626Vn2k) 的录像现已在 YouTube 上线，重点讨论了 Flat Buffers 和 Forge Tools。
   - 会议最后提到了 CFFI，引发了关于静态链接以及为了兼容 OpenSSL 等库而采用更精细的外部调用语法的讨论。
- **与 C 互操作相比，C++ 互操作是一个复杂的难题**：成员们讨论认为，虽然已经通过 DLHandles 支持了 C 互操作，但由于模板和 ABI 的考虑，C++ 互操作要复杂得多。
   - 有建议提出在不使用 DLOpen 的情况下进行动态链接，或者引入类似于 Rust 的 `#[link]` 宏的 `@link` 装饰器以获得更好的支持。
- **在 MLIR 中建模的外部调用**：关于 C 静态链接的讨论显示，外部调用在 MLIR 中被建模为 `pop.external_call`，详细示例可在 [GitHub](https://github.com/modularml/mojo/blob/main/stdlib/src/sys/ffi.mojo#L44) 上找到。
   - 动态链接和提升（lifting）被强调为对安全性和可用性至关重要，并参考了 [dlopen 文档](https://man7.org/linux/man-pages/man3/dlopen.3.html)。
- **对 Mojo 和其他 Modular 产品的支持**：一位用户询问如何针对 Mojo、Modular CLI 或其他产品相关的问题开启支持工单。
   - 官方澄清目前没有正式的工单系统，但用户可以通过 Discord、GitHub issues 或直接联系 Modular 团队寻求帮助。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch">YouTube</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=_QVs626Vn2k">Mojo 🔥 社区会议 #4</a>：Mojo 社区会议 #4 的录像🫓 Flat Buffers：内存高效的序列化⚒️ Forge Tools：扩展 Mojo 🔥 标准库🔄 Mojo 🔥 Gen...</li><li><a href="https://modul.ar/community-meeting-doc">[公开] Mojo 社区会议</a>：Mojo 社区会议文档链接：https://modul.ar/community-meeting-doc 这是一个公开文档；欢迎所有人查看并提出评论/建议。所有会议参与者必须遵守...</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/src/sys/ffi.mojo#L44">mojo/stdlib/src/sys/ffi.mojo at main · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账户来为 modularml/mojo 开发做出贡献。</li><li><a href="https://man7.org/linux/man-pages/man3/dlopen.3.html">dlopen(3) - Linux 手册页</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1263144421042425988)** (5 条消息): 

> - `视频中的目标检测`
> - `AWS EC2 实例`
> - `Mojo 数据类型` 


- **视频目标检测面临的挑战**：一位成员分享了在视频中使用预训练模型进行目标检测的挑战，包括处理大量帧和边界框（bounding boxes）不平滑的问题。
   - 另一位成员建议以 5 fps 等低帧率运行检测，并应用后处理来平滑边界框的位置。
- **使用 AWS EC2 进行目标检测**：一位成员提到利用 AWS EC2 实例来执行其目标检测任务。
   - *未对此点进行进一步讨论。*
- **关于 Mojo 数据类型的查询**：一位用户询问了 Mojo 中的原始（primitive）和复合（composite）数据类型。
   - *消息记录中未提供对该查询的回复。*


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1262861011896238250)** (51 条消息🔥): 

> - `Mojo 🔥 社区会议`
> - `Mojo 语言关键字`
> - `安装旧版本 Mojo`
> - `SIMD 原语参考`
> - `在 Mojo 中遍历 Tuple` 


- **Mojo 🔥 社区会议讨论错误处理**：社区在最新的 [Mojo 🔥 社区会议](https://youtu.be/_QVs626Vn2k?t=16740) 中讨论了错误处理，探索了序列化以及扩展 Mojo 标准库。
   - 成员们指出了深入讨论 PR 和错误处理的具体时间戳。
- **Mojo 语言移除 'let' 关键字**：在 Mojo 中，'let' 关键字已被移除，成员们讨论了对该关键字存在的记忆以及移除它的原因。
   - 'Let' 最初是存在的，但由于所有权和语义模型的原因被移除，现在使用 'var' 来声明运行时变量。
- **轻松安装旧版本 Mojo**：用户可以通过以下命令安装旧版本的 Mojo：`modular uninstall mojo` 和 `modular install mojo --install-version 24.3.0`。
   - 要查看所有可用版本的列表，用户可以查看 [Mojo GitHub branches](https://github.com/modularml/mojo/branches) 及其相关的活动。
- **x86 和 ARM 的 SIMD Intrinsics 参考**：对于 x86 SIMD intrinsics，Intel 的 [intrinsics guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html) 非常有用。
   - 对于 ARM，提供了 [ARM intrinsics](https://developer.arm.com/architectures/instruction-sets/intrinsics) 资源，但注意到缺少 SME。
- **在 Mojo 中遍历 Tuple**：一位用户询问如何在 Mojo 中遍历 Tuple，但讨论中未提供详细的解决方案。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/memory/unsafe/DTypePointer#address_of">DTypePointer | Modular Docs</a>：定义了一个包含给定 dtype 地址的 DTypePointer 结构体。</li><li><a href="https://youtu.be/_QVs626Vn2k?t=1390)">Mojo 🔥 社区会议 #4</a>：Mojo 社区会议 #4 的录音 🫓 Flat Buffers：内存高效的序列化 ⚒️ Forge Tools：扩展 Mojo 🔥 标准库 🔄 Mojo 🔥 Gen...</li><li><a href="https://youtu.be/_QVs626Vn2k?t=16740)">Mojo 🔥 社区会议 #4</a>：Mojo 社区会议 #4 的录音 🫓 Flat Buffers：内存高效的序列化 ⚒️ Forge Tools：扩展 Mojo 🔥 标准库 🔄 Mojo 🔥 Gen...</li><li><a href="https://github.com/modularml/mojo/activity?ref=nightly">Activity · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/branches">Branches · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://developer.arm.com/architectures/instruction-sets/intrinsics">Intrinsics – Arm Developer</a>：未找到描述</li><li><a href="https://github.com/rust-lang/rust-wiki-backup/blob/master/Sigil-reference.md">rust-wiki-backup/Sigil-reference.md at master · rust-lang/rust-wiki-backup</a>：Rust wiki 的备份。通过在 GitHub 上创建账号为 rust-lang/rust-wiki-backup 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1262885186324922470)** (3 条消息): 

> - `parallelize 和 sync_parallelize 的区别`
> - `内存管理改进` 


- **解释 parallelize 和 sync_parallelize 的区别**：一位成员询问了 `parallelize` 和 `sync_parallelize` 之间的区别。
- **内存管理需要更好的理解**：一位成员提到，由于需要对内存管理有更好的理解，他们尚未改进其草案版本。


  

---

### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1262853029267374090)** (12 messages🔥): 

> - `Modular installation issues` (Modular 安装问题)
> - `MNIST accuracy discrepancy` (MNIST 准确率差异)
> - `User experience improvements` (用户体验改进)
> - `Verbose reporting for MAX` (MAX 的详细报告)


- **Modular 安装问题已澄清**：一位用户分享说，他们在安装 **Modular** 的过程中遇到了困难，直到意识到需要为 **nightly/max** 导出正确的 bash 路径。
   - 另一位用户指出，安装问题很快将得到解决，更新将变得无缝。
- **注意到 MNIST 准确率差异**：用户发现在 **MNIST dataset** 上使用 `--use-relu6` 时，`python3` 和 `mojo` 运行的准确率存在差异。
   - 他们随后澄清说，使用正确的连字符解决了该问题；然而，**relu6** 导致准确率下降了约 1%。
- **Notebook UX 需要改进**：一位用户同意 Notebook 的体验需要改进，目前有些令人困惑。
   - 他们计划很快优化 **UX**。
- **请求 MAX 提供详细报告**：一位用户请求 **MAX** 提供更详细的报告，包括时长和 GFLOPS 等指标。
   - 他们强调这些指标将有助于做出硬件和财务决策。


  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1262866614265843923)** (4 messages): 

> - `Inline functions in Mojo` (Mojo 中的内联函数)
> - `SIMD optimization suggestions` (SIMD 优化建议)
> - `New Mojo nightly release` (新的 Mojo nightly 版本)
> - `Mojo nightly changelog updates` (Mojo nightly 更新日志)


- **提议内联函数的简写**：一位成员建议，将函数写在与其签名相同的行上可以作为 `@always_inline("nodebug")` 的宏，从而在不影响可读性的情况下缩短代码。
   - 理由是这种简写暗示内联函数应该是简短的，这通常是正确的。
- **使用 SVE 进行 SIMD 优化**：一位成员提到 SIMD 大小不需要是 2 的倍数，而 **SVE** 解决了这个问题。
   - 他们建议在没有可变宽度 SIMD 的架构上实现 drain loops 或使用 masks。
- **新的 Mojo nightly 编译器发布**：新的 nightly Mojo 编译器版本 `2024.7.1714` 已发布，可以通过命令 `modular update nightly/mojo` 进行更新。
   - 更新日志包括：移除 `SIMD.{min,max}` 方法以改用 builtins，为 `Dict.__init__` 添加带有 `power_of_two_initial_capacity` 的重载，以及移除 `SIMD.{add,mul,sub}_with_overflow` 方法。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo-marathons](https://discord.com/channels/1087530497313357884/1255303604894437388/1263148280167272480)** (71 messages🔥🔥): 

> - `Mojo utilizing cores` (Mojo 核心利用)
> - `NumPy performance` (NumPy 性能)
> - `Benchmarking` (基准测试)
> - `BLAS backends` (BLAS 后端)
> - `Intel MKL vs. other BLAS` (Intel MKL 与其他 BLAS)


- **Mojo 仅使用性能核**：用户观察到，在同时具有性能核（P-cores）和能效核（E-cores）的 Intel 芯片上，**Mojo** 中的 `parallelize` 函数专门利用性能核以获得更好的结果。
   - 这一设计决策源于使用像 `parallelize` 这样简单的 API 在性能核和能效核之间高效分配任务所面临的挑战。[详情点击此处](https://link.to.issue)。
- **即使使用较少核心，Mojo 依然击败 NumPy**：基准测试结果显示，尽管 **Mojo** 仅使用性能核，而 **NumPy** 使用了所有核心，但 **Mojo** 的表现仍优于 **NumPy**。
   - 目前 Mojo runtime 还没有智能到可以在不同类型的核心之间高效分配工作，但预计未来会进行更新。
- **不同的 BLAS 后端影响 NumPy 性能**：讨论了 **NumPy** 的性能如何根据所使用的 BLAS 库而产生显著差异；提到了 **OpenBLAS**，而一些人更倾向于使用 **Intel MKL** 以获得更快的速度。
   - 成员指出，大多数“比 NumPy 快”的说法通常是与没有配置良好 BLAS 后端的 NumPy 进行比较。推荐使用 [Intel's distribution for Python](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html)。
- **手动计时和基准测试揭示新见解**：一位用户切换到手动计时以获得更准确的基准测试，揭示了在特定数据点（如 1024）处有趣的性能凹陷。
   - 他们注意到，当块大小略微超过 1024 时，会出现一些性能下降，使得第二个块的计算效率降低。
- **Intel MKL 的卓越性能**：Intel 的 MKL 被建议作为 OpenBLAS 的更快替代方案，即使在非 Intel CPU 上也是如此。
   - 这一建议源于 MKL 在超级计算中的广泛应用，因其性能优于其他 BLAS 库。

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1263011455469355028)** (1 messages): 

> - `DataComp for Language Models (DCLM)`
> - `DCLM-Baseline-7B`
> - `MMLU Benchmark`
> - `OpenLM framework`
> - `数据集设计的重要性` 


- **DataComp 为语言模型引入新的测试平台**：[DataComp for Language Models (DCLM)](https://arxiv.org/abs/2406.11794) 是一个用于受控数据集实验的测试平台，旨在提高语言模型的性能。
- **DCLM-Baseline-7B 在 MMLU 上取得令人瞩目的成绩**：[DCLM-Baseline-7B](https://huggingface.co/apple/DCLM-Baseline-7B) 在使用 2.6T 训练 token 的情况下，实现了 64% 的 MMLU 5-shot 准确率，相比 MAP-Neo 显著**提升了 6.6 个百分点**，且计算资源消耗减少了 40%。
- **OpenLM 框架支持 DCLM 的预训练**：DCLM 基于 [OpenLM framework](https://arxiv.org/abs/2406.11794) 提供了标准化的语料库和高效的预训练方案（recipes）。
- **DCLM 强调数据集设计**：DCLM 突出了数据集设计对于训练语言模型的重要性，其中基于模型的过滤（model-based filtering）被证明是构建高质量训练集的关键。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.11794">DataComp-LM: In search of the next generation of training sets for language models</a>: 我们介绍了 DataComp for Language Models (DCLM)，这是一个用于受控数据集实验的测试平台，目标是改进语言模型。作为 DCLM 的一部分，我们提供了一个包含 240T token 的标准化语料库...</li><li><a href="https://github.com/mlfoundations/dclm">GitHub - mlfoundations/dclm: DataComp for Language Models</a>: DataComp for Language Models。通过创建账号参与 mlfoundations/dclm 的开发。</li><li><a href="https://huggingface.co/apple/DCLM-Baseline-7B">apple/DCLM-7B · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[datasets](https://discord.com/channels/1053877538025386074/1105324249721356298/1262870425894129761)** (2 messages): 

> - `AI 对数学标注员的需求`
> - `Replete-AI 多语言翻译数据集` 


- **对数学数据和标注员的需求日益增长**：一名成员询问是否需要更多的**数学数据/数学标注员**来训练 AI 变得更聪明，以及该领域是否已经出现短缺。
   - *“只有我这么觉得，还是我们已经看到这个领域的短缺了？”* 是社区经验和见解讨论中的一个重点。
- **Replete-AI 发布巨量开源翻译数据集**：宣布了一个[新数据集](https://huggingface.co/datasets/Replete-AI/Multi-lingual_Translation_Instruct)，包含从英语到 **14 种语言**的 **280 万行**翻译数据。



**提及的链接**: <a href="https://huggingface.co/datasets/Replete-AI/Multi-lingual_Translation_Instruct">Replete-AI/Multi-lingual_Translation_Instruct · Datasets at Hugging Face</a>: 未找到描述

  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1263122622233841706)** (4 messages): 

> - `Oxen.AI 论文俱乐部`
> - `表示微调 (Representation Finetuning)`
> - `与 repeng/vector steering 的比较` 


- **作者 Zhengxuan Wu 将加入 Oxen.AI 论文俱乐部**：[Arxiv 论文](https://arxiv.org/pdf/2404.03592)的第一作者 Zhengxuan Wu 将于本周五加入 Greg Schoeninger 的 Oxen.AI 论文俱乐部，讨论在表示编辑方面如何优于参数高效微调 (PEFT) 方法。
   - 点击[此处](https://lu.ma/oxen)参加会议，探索构建世界级 AI 数据集并向 Zhengxuan Wu 本人学习。
- **ReFT: 表示微调 (Representation Finetuning)**：即将举行的论文俱乐部会议将探讨 **ReFT** (Representation Finetuning)，并承诺与作者 Zhengxuan Wu 进行详细讨论。一位成员对“表示（representation）”和“特定任务干预（task-specific intervention）”的定义等细节表示好奇。
   - 他们还询问这个概念是否类似于通过直接修改代码库来改进 API，并将该论文描述为“非常玄学（voodoo）”。
- **与 Repeng/Vector Steering 的比较**：一位成员询问 ReFT 是否与 repeng/vector steering 不同，另一位成员回答说它们极其相似。
   - 有人指出该论文主动引用了 repeng，表明方法论上存在显著重叠。



**提及的链接**: <a href="https://lu.ma/oxen">Oxen.ai · Events Calendar</a>: 在 Luma 上查看并订阅 Oxen.ai 的活动。共同构建世界级 AI 数据集。跟踪、迭代、协作并发现任何格式的数据。

  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1262886729480933478)** (9 messages🔥): 

> - `Lunar Caves` (月球洞穴)
> - `Belief State Geometry in Transformers` (Transformer 中的信念状态几何)
> - `Tool Use Models` (Tool Use 模型)
> - `LLM-driven Digital Agents` (LLM 驱动的数字 Agent)


- **月球洞穴：隐藏的避难所**：[科学家已确认月球洞穴为潜在的隐藏避难所](https://www.perplexity.ai/page/lunar-caves-a-new-frontier-in-u3Rkbvk4QROuAEtNMlwoug)，称其为探索的新前沿。
   - 来自科学界的“复古”兴趣让这篇数月前的论文在 Twitter 上重新受到关注。
- **Transformer 中的信念状态几何解析**：一篇发表在 [arXiv 上的新论文](https://arxiv.org/abs/2405.15943)提出了“信念状态几何” (belief state geometry) 的概念，展示了 Transformer 如何在其残差流 (residual streams) 中编码信念更新。
   - 社区反应不一，有的认为其“深奥”，有的则认为可能是“过度复杂的 AI 心理黑话”。
- **Llama 3 Groq Tool Use 模型夺得榜首**：[Rick Lamers 宣布](https://x.com/RickLamers/status/1813341037198204962)了 Llama 3 Groq Tool Use 8B 和 70B 模型，强调了它们在 BFCL 上取得的第一名成绩。
   - 这些模型仅在合成数据上进行训练，现已在 Groq API 和 Hugging Face 上提供。
- **ManifoldRG 关于 LLM 数字 Agent 的立场**：[ManifoldRG 分享了一篇立场论文](https://x.com/ManifoldRG/status/1811120196570206459)，认为 LLM 驱动的 Agent 的进步需要超越基于语言的处理，以增强推理能力。
   - 该论文探讨了智能数字 Agent 的当前局限性和未来方向。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.15943">Transformers represent belief state geometry in their residual stream</a>: 当我们对大语言模型进行下文预测训练时，我们正在构建什么样的计算结构？在这里，我们提供的证据表明，这种结构是由信念的元动力学 (meta-dynamics) 提供的...</li><li><a href="https://x.com/RickLamers/status/1813341037198204962">来自 Rick Lamers (@RickLamers) 的推文</a>: 我已经领导一个秘密项目好几个月了……消息终于传开了！🛠️ 我很自豪地宣布 Llama 3 Groq Tool Use 8B 和 70B 模型 🔥 一个开源的 Tool Use 全量微调 Lla...</li><li><a href="https://x.com/ManifoldRG/status/1811120196570206459">来自 Manifold Research (@ManifoldRG) 的推文</a>: 🚨我们很高兴分享《大语言模型时代的智能数字 Agent》，这是一篇探讨 LLM 驱动的 Agent 进展、识别局限性并建议... 的立场论文。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1262853343278137480)** (154 messages🔥🔥): 

> - `Hermes 2.5 与 Hermes 2 的性能对比`
> - `扩展 Mistral 的挑战`
> - `模型实验`
> - `Tool calling 实现`
> - `Function calling 问题` 


- **Hermes 2.5 在基准测试中优于 Hermes 2**：在添加了[代码指令示例](https://link.to.examples)后，**Hermes 2.5** 在各项基准测试中表现出比 **Hermes 2** 更优的性能。
   - Hermes 2.5 在 MMLU 基准测试中得分为 **52.3**，显著高于 Hermes 2 的 **34.5**。
- **Mistral 在扩展超过 8k 参数时遇到困难**：成员们指出，如果不进行额外的预训练，**Mistral** 无法有效地扩展到 8k 以上，这是一个[众所周知的问题](https://link.to.issue)。
   - 建议探索 *mergekit* 和 *frankenMoE finetuning* 技术以进行后续改进。
- **拼接模型 (frankensteining) 的实验**：一位用户分享了合并 **Hermes 2 pro** 和 **llama70b-instruct** 的结果，创建了一个名为 **Llamagnific** 的新模型，托管在 [Hugging Face](https://huggingface.co/nisten/llamagnific-3-87b) 上。
   - *Llamagnific* 展示了更高的智能，但同时也增加了出现无意义回答的情况。
- **使用 Hermes 2 Pro 实现 Tool calling**：在一个 [GitHub PR](https://github.com/vllm-project/vllm/pull/5649) 中，有人提议为 **Hermes 2 Pro** 使用 **vLLM** 实现 OpenAI 风格的 tool calling 测试版。
   - 对 system prompt 的调整使得 tool call 的排序更加一致，从而提升了整体性能。
- **Tool calling 和函数使用问题**：讨论了 tool calling 和函数使用中的挑战，强调了流式传输 tool calls 以向用户提供实时反馈的重要性。
   - 合理的 tool call 流式传输通过减少等待时间并及时显示中间结果，显著提升了用户体验。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/nisten/llamagnific-3-87b-gguf">nisten/llamagnific-3-87b-gguf · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B">NousResearch/Hermes-2-Pro-Llama-3-8B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/nisten/llamagnific-3-87b">nisten/llamagnific-3-87b · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/nisten/llamagnific-3-87b-gguf/resolve/main/llamagnific_1bit_optimized_IQ1_L.gguf">无标题</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/Replete-AI/Multi-lingual_Translation_Instruct">Replete-AI/Multi-lingual_Translation_Instruct · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/i/broadcasts/1lDGLldQVmvGm">来自 GitHub 的推文 - FixTweet/FxTwitter</a>: 修复损坏的 Twitter/X 嵌入内容！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能 - FixTweet/FxTwitter</li><li><a href="https://wow.groq.com/introducing-llama-3-groq-tool-use-models/">介绍 Llama-3-Groq-Tool-Use 模型 - Groq 是快速 AI 推理框架</a>: 我们很高兴宣布发布两个专门为 tool use 设计的新开源模型：Llama-3-Groq-70B-Tool-Use</li><li><a href="https://github.com/vllm-project/vllm/pull/5649">支持允许 OpenAI API 风格 tool use 和 "auto" 工具选择的开源模型，由 K-Mistele 提交 · Pull Request #5649 · vllm-project/vllm</a>: 草案：OpenAI Tool Use 检查清单。此（草案）PR 将以一种对 tool use 格式和 prompt 格式保持极简见解的方式，添加对 OpenAI 风格 tool calling 的支持。以下功能...</li><li><a href="https://x.com/phill__1/status/1813307823570157899">来自 Phil (@phill__1) 的推文</a>: Google 意外更新了其包含 Gemini 2.0 的网站，Bing 索引抓取到了它</li><li><a href="https://www.reddit.com/r/singularity/comments/1e4wlzr/saw_something_interesting_when_i_googled_gemini/">Reddit - 深入探索</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1262863146650042428)** (19 条消息🔥): 

> - `使用 Tiktoken 进行 Tokenization`
> - `Huggingface Pipelines 中的 Beam Search 实现`
> - `Tiktoken 中 BPE 的可逆性`
> - `Huggingface Pipelines 中的自定义采样` 


- **Tiktoken 分词中的挑战**：一位用户报告了在使用 **Tiktoken 库** 解码阿拉伯符号时遇到的问题，导致出现特殊符号而非原始文本。
   - *“这确保了即使某些部分不正确，我们仍能获得可读文本”*，他们使用了 `errors='replace'` 来处理解码过程中的无效 UTF-8 序列。
- **处理特殊 Token 的 BPE 可逆性**：另一位成员指出，BPE（由 **Tiktoken** 使用）应该能够表示任何字节序列，这意味着所有 Token 都能解码为有效的 UTF-8 序列或特定的字节值。
   - 他们确信 **cl100k_base** 保证了 Token 序列的可逆性。
- **在 Huggingface 中实现自定义 Beam Search**：一位成员询问如何为 **Huggingface** Pipelines 创建自定义 Beam Search，并获得了关于使用 `model.generate()` 参数的指导。
   - 他们最终通过扩展 **GenerationMixin** 来重新实现 `beam_search()` 并创建了一个自定义模型类，并表示虽然感觉有点*粗糙（janky）*但功能正常。


  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1262884606030512239)** (3 条消息): 

> - `` 


- **未检测到实质性讨论**：在最近的消息中未观察到有意义或详细的讨论。
- **用户互相问候**：用户在频道中互相问候，发送了如“Hi”之类的消息。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1262866408656732251)** (60 条消息🔥🔥): 

> - `The Pile 2`
> - `Proof-Pile-2`
> - `YouTube 视频抓取争议`
> - `公众对 YouTube 数据使用的反应`
> - `AI 数据使用的透明度` 


- **关于 The Pile 2 的困惑**：用户对 **The Pile 2** 是否存在感到困惑，随后有澄清指出它目前尚不存在。
   - (无)
- **了解 Proof-Pile-2**：一位用户链接了 [Hugging Face](https://huggingface.co/datasets/EleutherAI/proof-pile-2) 上的 **Proof-Pile-2** 数据集，将其描述为一个包含 550 亿 token 的数学和科学文档数据集。
   - (无)
- **YouTube 视频抓取争议升温**：在 [Proof News 文章](https://www.proofnews.org/apple-nvidia-anthropic-used-thousands-of-swiped-youtube-videos-to-train-ai/) 发表后，人们对 **YouTube 视频**在未经许可的情况下被抓取并用于 AI 数据集表示担忧。
   - 像 [Philosophy Tube](https://x.com/PhilosophyTube/status/1813227210569920685) 和 [Jacob Geller](https://x.com/yacobg42/status/1813226763117367688) 这样的艺术家谴责了这种做法，引发了关于影响和伦理问题的讨论。
- **AI 数据使用的透明度受到质疑**：**EAI** 因对其数据来源保持透明而面临抨击，用户讨论了其他科技公司可能更不坦率的情况。
   - *“公众的强烈抵制和法律风险是人们对数据不透明的主要原因，并且有人担心公开透明会招致指责和批评，”***一位用户说道**。
- **动荡时期的社区感谢**：一位用户表达了对社区的深切感谢，强调了社区如何帮助他们跟上研究进度并激励他们开始自己的项目。
   - (无)


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/datasets/EleutherAI/proof-pile-2">EleutherAI/proof-pile-2 · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://x.com/proof__news/status/1813182354728317341">来自 Proof News (@proof__news) 的推文</a>：我们最新的调查揭露了一个包含超过 170,000 条 YouTube 视频字幕的数据集，大型科技公司使用该数据集来训练其 AI 模型。“这会被用来剥削和伤害艺术家吗？是的，绝对...”</li><li><a href="https://www.youtube.com/shorts/xiJMjTnlxg4">AI 正在偷走我的视频</a>：未找到描述</li><li><a href="https://x.com/PhilosophyTube/status/1813227210569920685">来自 Abigail Thorn (@PhilosophyTube) 的推文</a>：非常遗憾地告诉大家——一家名为 EleutherAI 的 AI 公司窃取了数万个 YouTube 视频——包括我的很多视频。我是 Proof News 采访的创作者之一。被盗的数据被出售...</li><li><a href="https://x.com/yacobg42/status/1813226763117367688">来自 Jacob Geller (@yacobg42) 的推文</a>：看来我的几十个视频被抓取并包含在训练 AI 的数据集中。做这些事的公司没有征得使用我作品的许可（当然）。我并不感到惊讶，但...</li><li><a href="https://web.archive.org/web/20240717020029/https://www.washingtonpost.com/technology/2024/07/16/trump-ai-executive-order-regulations-military/">特朗普盟友起草 AI 命令，旨在启动国防领域的“曼哈顿计划”</a>：该计划旨在取消“繁琐的监管”，这将有利于硅谷投资者，他们现在正成群结队地支持这位前总统。
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1262852011901190224)** (89 条消息🔥🔥): 

> - `Efficient Attention 机制`
> - `Transformer 优化`
> - `Reformer: The Efficient Transformer`
> - `LSH attention 实践`
> - `用于免疫逃逸的 PLMs`

- **标准差计算辩论**：关于使用[文章链接](https://www.strchr.com/standard_deviation_in_one_pass)进行单次遍历（one pass）计算标准差与传统两次遍历（two-pass）方法的讨论。
   - 一位用户提到已经实现了该功能，但在 launch config 方面遇到了问题。
- **揭秘 TransformerEngine 的主张**：用户讨论了 **TransformerEngine** 的融合实现，确认它并没有像之前假设的那样融合 normalization 和 linear 层。参见 [TransformerEngine 代码](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/layernorm_linear.py)链接。
   - 讨论了 RMSNorm 融合作为一种更优的方法，允许在单个 kernel 中完成 scaling、normalization、linear 和 activation。
- **Reformer: Efficient Transformer 澄清**：提供了关于 **Reformer: The Efficient Transformer** 的澄清，强调了哈希函数中 keys 归一化的重要性以及 attention 矩阵的差异。
   - 讨论了尽管具有理论优势，但 **LSH attention** 为何尚未得到广泛采用。
- **高效 Attention 机制的挑战**：讨论了像 **Reformer** 这样高效 Transformer 的[复现问题](https://openreview.net/forum?id=3s8Y7dHYkN-&noteId=aDaPfMT84Ef)，包括难以达到原始性能主张的问题。
   - 提到 **Linear Transformers** 可能是解决二次复杂度问题中更成功的替代方案。
- **用于区分病毒模拟的 PLMs**：一位用户分享了他们在 ICML 2024 上被接收的海报展示，内容是使用 **Protein Language Models (PLMs)** 识别模拟人类蛋白的病毒蛋白，具有 [99.7% RO CAUC](https://openreview.net/forum?id=gGnJBLssbb&noteId=gGnJBLssbb)。
   - 他们的海报分析了 PLMs 和免疫系统中的错误，以促进更好的疫苗/治疗开发，并提供了[可用代码](https://github.com/ddofer/ProteinHumVir)。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2006.16236">Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention</a>: Transformers 在多项任务中取得了显著性能，但由于其相对于输入长度的二次复杂度，在处理超长序列时速度极慢。为了解决...</li><li><a href="https://arxiv.org/abs/2407.11542">Understanding Counting in Small Transformers: The Interplay between Attention and Feed-Forward Layers</a>: 我们对在直方图任务上训练的简单 Transformer 模型进行了全面分析，该任务的目标是统计固定字母表中输入序列中每个项目的出现次数。描述...</li><li><a href="https://arxiv.org/abs/2407.11239">From GaLore to WeLore: How Low-Rank Weights Non-uniformly Emerge from Low-Rank Gradients</a>: 现代大语言模型 (LLMs) 由具有数十亿元素的矩阵组成，这使得它们的存储和处理在计算资源和内存使用方面要求极高。由于...</li><li><a href="https://goombalab.github.io/blog/2024/hydra-part1-matrix-mixer/"> Hydra Part I - Matrix Mixer Framework | Goomba Lab </a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=s8RqGlU5HEs">2 Years of My Research Explained in 13 Minutes</a>: 这是我在强化学习 (Reinforcement Learning) 背景下对表示学习 (Representation Learning) 和模型学习的研究。历时两年，我终于可以谈论...</li><li><a href="https://openreview.net/forum?id=3s8Y7dHYkN-&noteId=aDaPfMT84Ef">Reproducibility Challenge: Reformer</a>: 我们尝试复现 ICLR 2020 论文 "Reformer: The Efficient Transformer" 的核心主张；即所引入的技术能够实现与传统 Transformer 模型相当的性能...</li><li><a href="https://openreview.net/forum?id=rkgNKkHtvB&noteId=H1g3oF4sjS">Reformer: The Efficient Transformer</a>: 具有局部敏感哈希 (Locality-Sensitive Hashing) 和可逆层的高效 Transformer</li><li><a href="https://www.strchr.com/standard_deviation_in_one_pass">Calculating standard deviation in one pass - strchr.com</a>: 未找到描述</li><li><a href="https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/layernorm_linear.py#L141>,">TransformerEngine/transformer_engine/pytorch/module/layernorm_linear.py at main · NVIDIA/TransformerEngine</a>: 一个用于在 NVIDIA GPU 上加速 Transformer 模型的库，包括在 Hopper 和 Ada GPU 上使用 8 位浮点 (FP8) 精度，以提供更好的性能和更低的内存利用率...</li><li><a href="https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/layernorm_linear.py">TransformerEngine/transformer_engine/pytorch/module/layernorm_linear.py at main · NVIDIA/TransformerEngine</a>: 一个用于在 NVIDIA GPU 上加速 Transformer 模型的库，包括在 Hopper 和 Ada GPU 上使用 8 位浮点 (FP8) 精度，以提供更好的性能和更低的内存利用率...</li><li><a href="https://www.lesswrong.com/posts/pHPmMGEMYefk9jLeh/llm-basics-embedding-spaces-transformer-token-vectors-are">LLM Basics: Embedding Spaces - Transformer Token Vectors Are Not Points in Space — LessWrong</a>: 这篇文章解释了我刚开始接触 Transformer 嵌入时产生的一个误解。感谢 Stephen Fowler 的...</li><li><a href="https://openreview.net/forum?id=gGnJBLssbb&noteId=gGnJBLssbb">Protein language models expose viral mimicry and immune escape</a>: 病毒通过分子模拟规避免疫系统，采用其宿主的生物物理特征。我们调整了蛋白质语言模型 (PLMs) 以区分人类和病毒...</li><li><a href="https://github.com/ddofer/ProteinHumVir">GitHub - ddofer/ProteinHumVir: Code &amp; data for &quot;Protein Language Models Expose Viral Mimicry and Immune Escape&quot;</a>: “蛋白质语言模型揭示病毒模拟和免疫逃逸”的代码和数据 - ddofer/ProteinHumVir</li><li><a href="https://doi.org/10.1101/2024.03.14.585057">Protein Language Models Expose Viral Mimicry and Immune Escape</a>: 动机：病毒通过分子模拟规避免疫系统，采用其宿主的生物物理特征。我们调整了蛋白质语言模型 (PLMs) 以区分人类和病毒...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1263122561861029979)** (4 messages): 

> - `Arrakis library`
> - `Mechanistic interpretability tools`
> - `Feedback request` 


- **介绍用于快速 Mechanistic Interpretability 的 Arrakis 库**：[Arrakis](https://github.com/yash-srivastava19/arrakis) 是一个旨在进行、跟踪和可视化 Mechanistic Interpretability 实验的新库，面向希望快速迭代的研究人员。
   - 该库包含 **tuned-lens 工具**和模型手术（model surgery）等功能，但仍处于早期开发阶段。
- **征求社区对 Arrakis 的反馈**：发起了关于 Arrakis 对社区实用性的反馈请求，强调了其易用性和快速迭代能力。
   - 一位成员询问创作者为什么要从零开始，而不是使用 TransformerLens 或 nnsight 等现有工具。
- **Clemd6d 质疑 Arrakis 的替代选择**：用户 clemd6d 询问 yash_sri19，Arrakis 中是否存在 TransformerLens、nnsight 或 PyVene 无法实现的特定抽象。
   - *Yash_sri19* 回应称，他们希望对 Mechanistic Interpretability 中的常用功能进行抽象，并采用快速迭代设计，尽管该库目前仍处于早期开发阶段。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.co">GitHub: Let’s build from here</a>：GitHub 是超过 1 亿开发者共同塑造软件未来的地方。为开源社区做贡献，管理你的 Git 仓库，像专业人士一样审查代码，跟踪错误和功能...</li><li><a href="https://github.com/yash-srivastava19/arrakis">GitHub - yash-srivastava19/arrakis: Arrakis is a library to conduct, track and visualize mechanistic interpretability experiments.</a>：Arrakis 是一个用于进行、跟踪和可视化 Mechanistic Interpretability 实验的库。- yash-srivastava19/arrakis
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1263167243664101479)** (3 messages): 

> - `HF leaderboard musr score`
> - `Leaderboard maintainers query` 


- **关于 HF 排行榜 musr 原始分数的查询**：一位成员询问 HF 新排行榜中的 musr 原始分数是否是这 3 个任务的宏平均值（macro average）：**musr_murder_mysteries**、**musr_object_placements**、**musr_team_allocation**。
   - 另一位成员建议他们应该向 [leaderboard maintainers](https://huggingface.co/leaderboard) 寻求澄清。
- **确认需要询问排行榜维护者**：在收到建议后，该成员感谢了回复者建议直接联系排行榜维护者。


  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1262848301192187924)** (143 messages🔥🔥): 

> - `模型大小与硬件`
> - `训练技术`
> - `提示词细微差别`
> - `外绘技术`
> - `故障排除` 


- **基于硬件的模型大小指南**：成员们讨论了针对 GPU VRAM 和普通 RAM 的最佳模型大小，指出 VRAM 对性能有显著影响。
   - 提到**更大的模型需要更多的 VRAM**，且**更长的生成时间**并不意味着内存问题，除非发生 OOM 异常。
- **训练特定模型和角色**：用户询问了如何针对特定风格（如排线插画）训练模型，以及如何结合特定角色模型来生成联合内容。
   - 分享了相关资源链接以及关于 **regional prompting** 和 **multi-concept training** 的讨论，包括 [HuggingFace 的 T5](https://huggingface.co/jtlicardo/flan-t5-small-coref)。
- **理解提示词的细微差别**：文本提示词中的细微差别，例如“harvesting potato”与“potato harvesting”，引发了关于模型 **coreference resolution**（指代消解）能力的讨论。
   - 推荐使用 T5 的微调模型，特别是针对指代任务的模型，以有效处理复杂的提示词细微差别。
- **有效的 Outpainting 技术**：为了扩展生成的图像，推荐了 Outpainting 方法和特定工具，如 Photoshop 的 gen fill。
   - 成员们还讨论了在 ComfyUI 中使用 KSampler 来管理扩展过程中的 seeds，以避免图像重叠。
- **Stable Diffusion 问题排查**：在 Automatic1111 中遇到模型问题的成员讨论了故障排除步骤，包括针对特定硬件提供**更好性能**的命令行参数。
   - 提供了诸如**使用 'xformers' 和 'medvram-sdxl'** 选项的建议，以改善在有限硬件配置上的功能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/jtlicardo/flan-t5-small-coref">jtlicardo/flan-t5-small-coref · Hugging Face</a>: 未找到描述</li><li><a href="https://civitai.com/images/19928073">khitomer 发布的视频</a>: 未找到描述</li><li><a href="https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/wiki/Regional-Prompt-Control>">主页</a>: Tiled Diffusion 和 VAE 优化，采用 CC BY-NC-SA 4.0 许可 - pkuliyi2015/multidiffusion-upscaler-for-automatic1111
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1263026310297550848)** (26 条消息🔥): 

> - `CUDA kernel call errors` (CUDA kernel 调用错误)
> - `Template types in CUDA` (CUDA 中的模板类型)
> - `cudaMallocManaged overhead` (cudaMallocManaged 开销)
> - `Unified memory usage in CUDA` (CUDA 中的 Unified memory 使用)
> - `Deep learning specialization opinions` (深度学习专业化见解)


- **通过模板类型解决 CUDA kernel 调用错误**：一位正在学习 CUDA 的用户在调用 kernel 时遇到错误，通过正确指定模板类型解决了该问题。
   - *如 CUDA 示例所示，添加模板参数 (`<int>`) 解决了问题，尽管对于入门目的来说这显得有些大材小用。*
- **讨论 cudaMallocManaged 开销**：成员们讨论了 **cudaMallocManaged** 是否因 Host 和 Device 之间的共享内存空间而引入额外开销，结论是使用显式内存复制（explicit memory copies）可能更高效。
   - *使用 cudaMemcpy 显式处理内存传输可以避免潜在开销并提升性能。*
- **探索 Unified Memory 和 GPU 架构**：讨论了 Unified Memory (cudaMallocManaged)，重点关注其开销以及当前 NVIDIA GPU 的架构细节。
   - 一位成员指出，理解 GPU 架构对于优化内存使用至关重要，并质疑 Unified Memory 等某些过程是否会减慢操作速度。
- **微调 LLM：处理 pad token 和 prompt**：讨论了在 LLM 微调期间从 loss 计算中排除 pad token 的问题，HF 在 cross-entropy loss 中使用 `ignore_index=-100` 来排除它们。
   - *使用正确的 prompt 格式准备数据集至关重要，HF 的 tokenizer 支持应用 chat templates 以简化数据管理。*


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/huggingface/transformers/blob/72fb02c47dbbe1999ae105319f24631cad6e2e00/src/transformers/models/llama/modeling_llama.py#L1092-L1102).">transformers/src/transformers/models/llama/modeling_llama.py at 72fb02c47dbbe1999ae105319f24631cad6e2e00 · huggingface/transformers</a>: 🤗 Transformers: 为 Pytorch, TensorFlow 和 JAX 提供最先进的机器学习模型。 - huggingface/transformers</li><li><a href="https://github.com/pytorch/torchtune/blob/8e036611aac377fd9b383a66c161ce085c93f8ce/recipes/full_finetune_single_device.py#L448-L454).">torchtune/recipes/full_finetune_single_device.py at 8e036611aac377fd9b383a66c161ce085c93f8ce · pytorch/torchtune</a>: 一个用于 LLM 微调的 PyTorch 原生库。通过在 GitHub 上创建账号为 pytorch/torchtune 做出贡献。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1262863210302668860)** (33 条消息🔥): 

> - `PyTorch Profiler Performance` (PyTorch Profiler 性能)
> - `Thunder vs Torch Compile` (Thunder vs Torch Compile)
> - `Nvfuser vs Triton` (Nvfuser vs Triton)
> - `Kernel Compilation` (Kernel 编译)
> - `Runtime Optimization` (运行时优化)


- **PyTorch Profiler 导出耗时过长**：一位用户担心使用 **PyTorch profiler** 导出 trace 需要大约 **30 分钟**，这可能是因为捕获了过多的信息。
   - 另一位成员建议禁用 `profile_memory` 和 `with_stack` 选项，以便在不丢失运行时信息的情况下加快导出速度。
- **Thunder 和 Torch Compile 集成**：讨论重点介绍了 **Thunder**，这是一个性能优化层，可以将不同的 **PyTorch** 兼容 kernel 缝合在一起，通常使用 'nvfuser' 作为后端。
   - 该特性与 **torch.compile** 进行了对比，Thunder 旨在实现透明且可调试的后端集成，从而促进手动性能调优。
- **Nvfuser vs Triton 性能讨论**：注意到 **nvfuser** 和 **Triton** 具有不同的性能特征，在不同的基准测试中各有胜负。
   - 对话强调通过结合使用两者来获得最佳性能，并利用 **Thunder** 有效地混合匹配这些后端。
- **关于自定义 Kernel 编译时间的担忧**：Profiler 导出时间过长可能是由于通过 **nvfuser** 编译的自定义 kernel 导致的。
   - 尽管耗时较长，用户仍对 **Thunder** 及其混合匹配 kernel 的能力表示赞赏。
- **优化 Kernel 编译**：对话简要触及了使用 **nvfuser**（相对于 **Triton**）优化 kernel 编译的复杂性。
   - 澄清了 Thunder 支持在生成的 Python 函数中进行数据依赖和自动调度，以优化运行时。


  

---

### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1263051227994783755)** (7 条消息): 

> - `Mixing CUDA kernels with PyTorch` (在 PyTorch 中混合使用 CUDA kernel)
> - `Writing custom CUDA kernels` (编写自定义 CUDA kernel)
> - `Automatically generated Python bindings` (自动生成 Python 绑定)
> - `Compiling custom kernels` (编译自定义 kernel)


- **寻求在 PyTorch 中混合使用 CUDA kernel 的代码示例**：**artificial_anteligence** 正在寻找关于如何在 PyTorch 中混合使用 CUDA kernel 的资料，包括简化模型的完整实现和训练。
- **需要替换 PyTorch 函数为自定义 CUDA kernel 的参考资料**：**artificial_anteligence** 表示需要将 PyTorch 函数的部分内容替换为自定义 CUDA kernel，或者全部用 CUDA 编写，并请求参考代码。
   - **as_ai** 提到 CUDA mode 第 1 课演示了如何开始使用 PyTorch 中的 `load_inline` 模块，包括编译指定的 kernel。
- **为自定义 CUDA kernel 自动生成 Python 绑定**：**as_ai** 指出，如果你提供 CUDA 源码和带有 kernel 启动参数的 C++ torch tensor 函数，可以自动生成 Python 绑定。


  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1262860938869080098)** (13 条消息🔥): 

> - `unwrap_tensor_subclass issues`
> - `AQT Tensor instance`
> - `FakeTensor attribute error`
> - `PyTorch nightly build`
> - `GitHub issue` 


- **unwrap_tensor_subclass 将 tensor 子类转换为普通 tensor**：**unwrap_tensor_subclass** 使用参数化将 tensor 子类转换为普通 tensor，以绕过 **torch.compile** 堆栈限制。
   - 当 `layout_tensor` 是另一个持有 **IntxTensor** 的子类 tensor 时，会遇到问题。
- **AQT Tensor 实例导致错误**：一位用户发现他们的 **AQT Tensor** 实例崩溃了，因为 **layout_tensor** 是另一个子类 tensor。
   - 一位用户指出：*'我认为在我的用例中它崩溃了，因为我有一个 AQT Tensor 实例'*。
- **FakeTensor 缺少 'get_plain' 属性**：出现了一个错误：PyTorch nightly 版本中的 **FakeTensor** 对象缺少所需的 *'get_plain'* 属性。
   - 错误详情为：*'torch._dynamo.exc.TorchRuntimeError'*，导致 forward 方法失败。
- **错误在 7/11 PyTorch nightly 版本中仍然存在**：一位用户确认该问题在 **7/11 PyTorch nightly build** 中依然存在，并寻求社区帮助。
   - 有人建议：*'好的，你能开一个 issue 来描述发生了什么吗？'* 以获取进一步帮助。
- **关于 unwrap_tensor_subclass 的 GitHub issue**：已创建一个 **GitHub issue** 来解决嵌套 tensor 子类的 unwrap_tensor_subclass 问题。
   - 访问该 issue [此处](https://github.com/pytorch/ao/issues/515) 了解更多详情。



**提到的链接**：<a href="https://github.com/pytorch/ao/issues/515">unwrap_tensor_subclass 与嵌套 tensor 子类问题 · Issue #515 · pytorch/ao</a>：我注意到在尝试创建一个持有另一个 tensor 子类的 tensor_subclass 时出现了奇怪的行为。这是一个最小化复现：(将此添加到 torchao/dtypes/affine_quantized_ten...

  

---


### **CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1262876333151752356)** (12 条消息🔥): 

> - `Notation in Triton Puzzle 6`
> - `ImportError in Triton`
> - `Efficient Softmax Implementation`
> - `Assignment Operator in Triton` 


- **澄清 Triton Puzzle 6 中的符号**：成员们讨论了 Triton puzzles 中 **Puzzle 6** 符号的混淆，特别是微分的使用和矩阵-向量乘法。
   - 讨论中提到了函数定义的歧义，其 forward 操作涉及 **ReLU** 以及矩阵 **x** 与向量 **y** 的乘法。
- **Triton 中的 ImportError 问题**：一位成员报告在尝试从 'triton.runtime.interpreter' 导入 'interpreter_builder' 时遇到 **ImportError**。
   - **此问题从昨天开始随机出现。**
- **Triton 中高效 Softmax 的策略**：讨论集中在如何通过仅读写一次 GMem 来完成 **long softmax**。
   - 挑战在于如何在不进行第二次 pass 的情况下确保正确更新，特别是在假设 **T1 维度** 的 shared memory 有限的情况下。
- **处理 Triton 中的赋值运算符**：成员们询问了 Triton 中赋值运算符的工作原理，特别是在 **softmax** 等示例中。
   - **编译器会自动管理**变量的 shared memory 分配，尽管具体的实现细节非常复杂。



**提到的链接**：<a href="https://fkong.tech/posts/2023-04-23-triton-cuda/">揭秘 OpenAI Triton</a>：通过逐步指导和代码示例，学习如何构建从 OpenAI Triton 到 CUDA 的映射，以实现高性能深度学习应用。

  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1262894839440080987)** (3 messages): 

> - `UIUC 的 RAG-GPT4 TA 实现`
> - `学生交互挑战`
> - `优化 GPT-2 kernel` 


- **RAG-GPT4 TA 在 UIUC 成功部署**：**RAG-GPT4 TA** 已在 **UIUC** 的 CUDA 课程中实现，其知识库基于 CUDA 教科书、幻灯片和编程指南等课程材料。
   - 挑战包括学生发送对抗性或无关话题的问题，即使在添加了防护栏 (guardrails) 之后也是如此。
- **在课程中加入 CUDA 的兴趣**：一位成员表示有兴趣将 CUDA 集成到他们的课程中，考虑增加一个专注于优化 **GPT-2 kernel** 的入门模块。
   - 该模块的目标可能是产出一个 **llm.c fp32 CUDA** 版本，遵循其之前在 **llm.c** 中类似的工作方法。


  

---


### **CUDA MODE ▷ #[huggingface](https://discord.com/channels/1189498204333543425/1263189334434123776/1263189684041809996)** (2 messages): 

> - `改进 HuggingFace 生态系统中的 ML Systems` 


- **关于 ML Systems 改进的讨论**：一组用户强调了利用该频道讨论 **HuggingFace 生态系统**中 **ML Systems** 改进的需求。
- **启动确认**：一位成员对该话题的引入给予了积极确认，并回复了 'Perfect 🔥'。


  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1262847521898631188)** (50 messages🔥): 

> - `质量控制问题`
> - `Captcha 实现`
> - `复制代码块`
> - `API 速率限制`
> - `付费订阅额度` 


- **质量控制发布未测试的代码**：一位成员提到，某家价值十亿美元的公司的质量控制部门偶尔会将随机且未经测试的代码发布到生产环境。
   - 另一位成员补充道：*每家公司都是如此。*
- **Cloudflare CAPTCHA 的困扰**：成员们对 **Cloudflare CAPTCHA** 的实现表示沮丧，质疑是谁做出了实施它的决定。
   - 一位成员评论说：*Cloudflare 经常崩溃或被攻破。*
- **复制代码块变得困难**：一位成员强调了在浏览器中从代码块复制内容的困难，并将其归咎于 `user-select: none` 样式。
   - 这个问题似乎已被悄悄修复，因为另一位成员注意到最近有所改进。
- **对 API 速率限制的担忧**：一位成员对较低的 **API** 速率限制以及增加额度的漫长过程表示担忧，担心这会影响他们的项目计划。
   - 他们被建议填写表格并联系 **Perplexity** 员工，以寻求可能更快的解决方案。
- **5 美元 API 额度过期说明**：**Pro** 用户每月会收到 5 美元的 **API** 奖励额度，如果当月未使用，该额度将在月底过期。
   - 一位成员对此感到失望，因为他们现在必须每个月尽快用完这些资金。



**提到的链接**：<a href="https://perplexity.typeform.com/to/j50rnNiB>">探索 Typeform，让表单变得有趣</a>：无需代码，几分钟内即可创建美观、互动的表单。免费开始使用。

  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1262875850983080000)** (11 条消息🔥): 

> - `In-Batch Search`
> - `Moon's Hidden Refuge`
> - `Music Streaming Platforms`
> - `Best TV Shows May 2024`
> - `Best Resources for 12-year-olds` 


- **优化内容：从 Threads 到 Pages**：一位成员发现，在 **Perplexity AI** 上进行内容优化的最佳方法是将 threads 转换为 pages，然后逐节手动构建页面。
   - 该方法可以更好地组织和展示信息，增强可读性和实用性。
- **2024 年 5 月必看热门电视剧**：一位成员分享了关于 **2024 年 5 月** 最佳电视剧的详细搜索结果，并附带了[链接](https://www.perplexity.ai/search/best-tv-shows-may-2024-2sIUwYTKTpWd0GaPxgBSUA)。
   - 该列表提供了推荐和评论，帮助用户选择最吸引人的剧集。
- **探索音乐流媒体平台趋势**：分享了一个关于 **音乐流媒体平台** 是否对音乐消费产生特定趋势或影响的搜索[链接](https://www.perplexity.ai/search/do-music-streaming-platforms-h-p38byn.iR_uEBxXZks.9bw)。
   - 讨论围绕这些平台如何塑造听众习惯和音乐产业展开。
- **魔方解法 GUI 详解**：分享了一个关于使用 GUI 方法[快速还原魔方](https://www.perplexity.ai/page/solving-rubik-s-cube-quick-gui-zlhjD1JwRyKYEcBs5_32lw)的详细页面。
   - 该指南旨在通过交互式图形界面简化魔方还原过程。
- **关于披萨加菠萝的辩论**：关于 *披萨加菠萝* 的陈年辩论在一个[专门页面](https://www.perplexity.ai/page/so-pineapple-on-pizza-R9MlOEh3SYunyVzRCv1CUg)上继续进行，引发了幽默而激烈的讨论。
   - 该页面探讨了这一两极分化的话题，提供了来自社区的各种观点和有趣评论。



**提到的链接**：<a href="https://www.youtube.com/embed/Y2_ddM_Mlro">YouTube</a>：未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1262891120161980456)** (1 条消息): 

> - `search_domain_filter`
> - `API Beta` 


- **API 中提供 search_domain_filter (Beta)**：据指出，通过 API 可以使用 **`search_domain_filter`**，但你需要加入 **beta** 测试才能访问它。
   - *用于在特定域内过滤搜索的有趣功能。*
- **Search Domain Beta 访问权限**：要使用 `search_domain_filter`，必须加入 **beta** 计划。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1262891373422706770)** (60 条消息🔥🔥): 

> - `Error Code 524`
> - `Meta 405B 模型定价`
> - `Deepseek Coder 速度问题`
> - `OpenRouter 上快速且经济的模型`
> - `WordPress 插件问题` 


- **用户遇到的 Error Code 524**: 多名用户报告在几分钟内遇到了 **Error Code 524**。
   - 一名用户询问其他人是否也遇到了同样的问题，表明这是一个影响服务的更广泛问题。
- **关于 Meta 405B 模型定价的讨论**: 用户推测了 [即将推出的 Meta 405B 模型的定价](https://discord.com/channels/1091220969173028894/1094454198688546826/1262737636607528972)，猜测可能会在 23 号左右发布。
   - 关于该模型 **8K context** 的信息是基于之前的模型，实际细节仍待确定。
- **对 Deepseek Coder 速度的抱怨**: 一名用户对 **Deepseek Coder** 的缓慢性能表示沮丧，尽管对其能力印象深刻。
   - 其他人也表达了同样的看法，提到如果性能更快或者有供应商提供更快的版本将会非常有益。
- **在 OpenRouter 上寻找快速且经济的模型**: 用户讨论了比 **GPT-3.5-Turbo** 更快更好但仍然经济实惠的模型。
   - 推荐了 **Claude-3-Haiku** 和各种 **Llama** 模型，但定价和 context length 方面的问题仍然存在。
- **RSS Feeds 的 WordPress 插件问题**: 一名用户报告了将 **OpenRouter API** 与用于自动编辑 RSS feed 新闻的 WordPress 插件集成时出现的问题。
   - 问题可能与 API key 使用或 rate limits 有关，并分享了使用 `curl` 验证 API 可达性的建议。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/openrouter/auto">Auto (best for prompt) by openrouter</a>：根据 prompt 的大小、主题和复杂程度，您的 prompt 将被发送至 [Llama 3 70B Instruct](/models/meta-llama/llama-3-70b-instruct), [Claude 3.5 Sonnet (self-moderated)](/models/anthropic/c...</li><li><a href="https://openrouter.ai/docs/limits">Limits | OpenRouter</a>：设置模型使用限制
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1262959768205660241)** (54 条消息🔥): 

> - `针对 LAION 的黑客攻击`
> - `ComfyUI 恶意软件`
> - `迪士尼黑客攻击数据泄露`
> - `虚假求职者`
> - `飓风桑迪后的电信故障` 


- **黑客组织利用恶意软件瞄准 LAION**: 一个黑客组织创建了一个名为 **ComfyUI_LLMVISION** 的恶意 ComfyUI 节点，旨在从用户电脑中窃取信息并安装木马。
   - 一名成员警告称，该组织过去曾参与黑进迪士尼的 Slack 并分发恶意游戏模组，强调了他们的专业性和影响力。
- **虚假求职者协助数据外泄**: 黑客利用克隆 **GitHub** 工程师身份的虚假求职者渗透公司并外泄数据。
   - 被雇佣的人员在不知晓恶意活动的情况下，将任务转发给真正的黑客团队，充当掩护。
- **飓风桑迪后的电信故障**: 飓风桑迪对 **Verizon 的纽约电缆库**造成了严重破坏，导致 13,000 公里的铜缆发生大规模故障。
   - 这一事件促使铜缆基础设施被光纤取代，标志着电信韧性的重大升级。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/counting-nodding-doug-mc-kenzie-bob-mckenzie-strange-brew-gif-17087583">Counting Nodding GIF - Counting Nodding Doug Mc Kenzie - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://darknetdiaries.com/transcript/133/">I'm the Real Connor – Darknet Diaries</a>：有一天 Connor 收到一封邮件说他的身份被盗了。这是他经历过的最奇怪的日子之一。</li><li><a href="https://tenor.com/bOBIU.gif">Preston GIF - Preston - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.datacenterknowledge.com/cables/after-sandy-verizon-confronts-catastrophic-failure-at-ny-cable-vault">After Sandy, Verizon Confronts &#x27;Catastrophic Failure&#x27; at NY Cable Vault</a>：当超级风暴桑迪袭击曼哈顿下城时，洪水导致 Verizon 位于 Broad Street 中心办公室下方的电缆库发生了“灾难性故障”...
</li>
</ul>

</div>
  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1262902341087133777)** (5 条消息): 

> - `InternVL2-Llama3-76B`
> - `Manifold Research Group`
> - `LLM-based Autonomous Agents`
> - `Research Log #041`
> - `MultiNet Dataset` 


- **InternVL2-Llama3-76B：多模态模型的愿景**：[InternVL2-Llama3-76B](https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B) 是 InternVL 系列的最新成员，拥有一系列参数量从 10 亿到 1080 亿不等的**指令微调模型 (instruction-tuned models)**。
   - 其他资源包括 [GitHub](https://github.com/OpenGVLab/InternVL)、[Blog](https://internvl.github.io/blog/)，以及关于 [InternVL 1.0](https://arxiv.org/abs/2312.14238) 和 [InternVL 1.5](https://arxiv.org/abs/2404.16821) 的论文。
- **在有限硬件上运行大模型的困扰**：一位用户表达了在 **4x 3090** 上运行 **40B** 模型时的困难，以及使用 **autoawq** 时遇到的问题。
- **Manifold Research Group 关于基于 LLM 智能体的立场论文**：来自 [Manifold Research Group](https://www.manifoldrg.com/llm-agents/) 的 Sidh 分享了他们的立场论文 **"Intelligent Digital Agents in the Era of Large Language Models"**，讨论了基于 LLM 的 AI Agent 的进展和局限性。
   - 他们正在扩大研究团队，并邀请感兴趣的人士加入他们的 [Discord](https://discord.gg/MfYZmYEGaa?ref=manifoldrg.com) 并为他们的 [GitHub](https://github.com/ManifoldRG?ref=manifoldrg.com) 做出贡献。
- **Manifold Research Group 的 Research Log #041**：Manifold Research Group 发布了 [Research Log #041](https://www.manifoldrg.com/research-log-041/)，记录了他们的每周研究进展，并重点介绍了 AI 社区的突破。
   - 他们有一个正在进行的名为 **MultiNet** 的项目，根据其 [V0 spec](https://github.com/ManifoldRG/MultiNet/issues/19?ref=manifoldrg.com)，已成功收集了超过 **50TB** 的数据集。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B">OpenGVLab/InternVL2-Llama3-76B · Hugging Face</a>: 未找到描述</li><li><a href="https://www.manifoldrg.com/llm-agents/">Intelligent Digital Agents in the Era of Large Language Models</a>: 该立场论文概述了当前基于 LLM 的 AI 智能体的研究领域和突破。我们强调了关键进展并讨论了每个领域的局限性。</li><li><a href="https://www.manifoldrg.com/research-log-041/">Research Log #041</a>: 欢迎阅读 Research Log #041！我们记录了 Manifold Research Group 各项计划的每周研究进展，并重点介绍了我们认为来自更广泛研究社区的突破...</li><li><a href="https://www.manifoldrg.com/opportunities/">Opportunities</a>: 参与我们工作的几种方式：1. 加入我们的 Discord 并参与活动和讨论（包括项目相关和非相关的）。2. 异步贡献 GitHub 上的 issue。 ...
</li>
</ul>

</div>
  

---



### **Interconnects (Nathan Lambert) ▷ #[events](https://discord.com/channels/1179127597926469703/1179127598442348729/)** (1 条消息): 

natolambert: 有人在 ICML 吗？我的一个 VC 朋友想在一个高档晚宴上见见我的朋友们。
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1263184116845904004)** (8 条消息🔥): 

> - `Prover-Verifier Games`
> - `Project Strawberry`
> - `Legibility Tax`
> - `RL and Model Properties` 


- **证明者-验证者博弈 (Prover-Verifier Games) 促进可读性提升**：[OpenAI 的最新工作](https://openai.com/index/prover-verifier-games-improve-legibility/) 关于 Prover-Verifier Games 旨在增强模型的**可读性 (legibility)**。
   - 一位成员开玩笑地指出，甚至论文本身也受到了所谓的“可读性税 (legibility tax)”的影响。
- **Project Strawberry 引发疑问**：简要提到了被称为 *Q** 的 Project Strawberry，引发了社区的好奇。
   - 一位成员幽默地问道：“到底什么是可读性税 (legibility tax)？”。
- **RL 让模型变得古怪**：一场关于**强化学习 (RL)** 如何影响模型特性的讨论展开了。
   - 一位成员评论了亲身体验这些影响的感受：“这张图表绝对是一种可读性税”。


  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1263039936127307838)** (6 messages): 

> - `SmoLLM 博客文章`
> - `DPO 数据集使用情况`
> - `模型系列尺寸` 


- **SmoLLM DPO 数据集选择背后的谜团**：一位用户发现 [SmoLLM 博客文章](https://huggingface.co/blog/smollm) 中描述模型 A 和 C 使用了 DPO 数据集 #1，而模型 B 使用了 DPO 数据集 #2，这显得很奇怪，并质疑这是否只是随机实验。
   - *另一位用户* 认为这可能是由于不同团队之间缺乏沟通，或者仅仅是经验性测试，并补充说 *“没有直观的原因”*。
- **特定模型的数据集偏好**：讨论指出，不同的模型在不同的数据集上可能表现更好，暗示了 **360m 模型** 的实际演示需求。
   - 一种观点认为，他们可能只是想展示 360m 模型，并不太关心其他配置。


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1262855620760043520)** (10 messages🔥): 

> - `游说与既得利益`
> - `AI 立法民意调查问题`
> - `公众对 AI 工具的看法` 


- **游说争议浮出水面**：[一条推文](https://fxtwitter.com/mpopv/status/1813273553477009546?s=46) 引发了讨论，该推文批评某人在游说某项立法法案的同时，秘密拥有一家有望从该法案通过中获利的公司。
   - *普通人将越来越对最强大的 AI 工具感到困惑、排斥和厌恶，除非行业能够想出如何为大众提供“简化版”并实现商业化。*
- **关于 AI 立法民意调查有效性的辩论**：成员们批评了在 AI 立法中使用民意调查的做法，其中一位成员表示引用民意调查很 *愚蠢*，并质疑 *普通人* 是否能理解 AI 立法。
   - 另一位成员承认这是行业中的 *问题*，强调需要更好地识别并适应公众的理解水平。
- **公众在使用 AI 工具时的困境**：一位成员分享了对 *普通人* 越来越对 ChatGPT 等强大 AI 工具感到困惑的担忧。
   - 他建议行业需要将更简单的版本商业化，以提高公众的接受度，并分享了向大约 500 人介绍标准聊天机器人的经验。



**提到的链接**：<a href="https://fxtwitter.com/mpopv/status/1813273553477009546?s=46">来自 Matt Popovich (@mpopv) 的推文</a>：感觉如果你在大力游说并募集捐款来游说某项特定的立法法案，你可能应该披露你秘密拥有一家有望从中获利的公司...

  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1262851687123517511)** (17 messages🔥): 

> - `GPT-4o vs Llama 405 tokenizers`
> - `GPT-4o tokenizer performance`
> - `Llama 3 initial release insights`
> - `Google Gemini 2.0 accident`
> - `Deepseek's open-source stance` 


- **GPT-4o 与 Llama 405 的 tokenizer 有本质区别**：讨论指出 **GPT-4o** 和 **Llama 405** 使用了非常不同的 tokenizer，除非 405 是 **chameleon**，但它并不是。
   - 一位用户指出 GPT-4o 的 tokenizer 在“编程方面有所退步”，且与 **GPT-4t** 相比，在 XML 上会产生更多 token。
- **Llama 3 初始发布内幕揭晓**：[Meta GenAI 产品总监 Joe Spisak](https://www.reddit.com/r/LocalLLaMA/s/mLPM7AocZF) 表示，**Llama 3** 最初是一个“预发布版”或“预览版”，但 **Mark** 推动了提前发布，这解释了初始版本仅支持 8k context 的原因。
   - Spisak 提到 Llama 模型“还有更多值得期待的内容”。
- **Google Gemini 2.0 意外曝光**：一条 [tweet](https://x.com/phill__1/status/1813307823570157899?s=46) 透露，**Google** 意外更新了包含 **Gemini 2.0** 内容的网站，并被 Bing 的索引抓取。
- **尽管中国技术落后，Deepseek 仍致力于开源**：[Deepseek 创始人梁文锋](https://x.com/main_horse/status/1813580480761196987?s=46) 断言他们不会走向闭源，并强调了建立强大技术生态的重要性。
   - 一份翻译提到，尽管 Deepseek 的 API 已有微利，但采访的很大一部分内容围绕着“中国落后于美国”展开。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/phill__1/status/1813307823570157899?s=46">Phil (@phill__1) 的推文</a>：Google 意外更新了包含 Gemini 2.0 的网站，Bing 索引抓取到了它</li><li><a href="https://x.com/main_horse/status/1813580480761196987?s=46">main (@main_horse) 的推文</a>：Deepseek 创始人梁文锋：我们不会走向闭源。我们相信首先拥有一个强大的技术生态更为重要。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/mLPM7AocZF">Reddit - 深入探索一切</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1262878861864075414)** (3 messages): 

> - `Billboard AI in IPA`
> - `Dell's comeback`
> - `Mathstral MMLU Breakdown` 


- **广告牌写着“把 AI 放进 IPA”**：一名成员分享了一张[广告牌照片](https://x.com/vampiric_shirin/status/1812901575368798413)，上面幽默地写着“putting the AI in IPA”，引发了困惑和欢乐。
   - 他们评论说，如果不是在网上找到了这张照片，他们会以为自己进入了精神错乱状态。
- **Dell 卷土重来**：一名成员对 Dell 最近的营销举措表示兴奋，指出它正在“找回往日的威风”。
- **Mathstral MMLU 细分的有趣关联**：一条 [tweet](https://x.com/jessemhan/status/1813254878615249116?s=46) 强调了 Mathstral MMLU 的细分数据，显示数学能力与会计、外交政策和人类性行为等学科之间存在有趣的负相关。
   - 分享链接的成员表示：“这是我全周看到的最有趣的事”。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/jessemhan/status/1813254878615249116?s=46">Jesse Michael Han (@jessemhan) 的推文</a>：Mathstral MMLU 细分是我全周看到的最有趣的事 —— 数学能力与会计、外交政策、人类性行为呈负相关</li><li><a href="https://x.com/vampiric_shirin/status/1812901575368798413">x_c4tb0yTH0Ti3_x (@vampiric_shirin) 的推文</a>：开车经过一个写着“putting the AI in IPA”的广告牌，如果我没在网上找到这张照片，我会以为我进入了精神错乱状态 ￼
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1262849929026469928)** (2 messages): 

> - `Policy Loss Function Discussion`
> - `Degenerate Case in DPO-like Algos` 


- **类 DPO 算法中必要的退化情况**：一名成员提到，退化情况对于处理胜出和失败场景中的公共前缀是**有用且必要**的。
   - 另一名成员表示赞同，强调了在类 DPO 算法中对该主题进行深度探索的重要性。
- **策略损失函数过拟合担忧**：有人对策略损失函数的**过拟合**问题表示担忧，特别是涉及到 `losses = -F.logsigmoid(policy_rejected_logp)` 这一项时。
   - 该成员指出，类 DPO 策略的兴起再次激发了人们对算法进行深入思考的热情。

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1262851673005752350)** (3 messages): 

> - `Sampling Methods in Policy Models`
> - `Preference Pair Selection`
> - `Zephyr Paper Criticism`
> - `Nemotron Paper Insights`
> - `DPO Objective Challenges` 


- **Policy Models 中的偏好对采样方法**：讨论提到在某些 Policy Models 中，会采样多个回复，并由 Reward Model 选择最受偏好和最不受偏好的回复作为偏好对，用于 DPO。
   - 相比之下，**Zephyr** 从非胜出回复中进行采样以增加多样性，并强调回复之间的分值差距（margin）。
- **对非胜出样本的观点未变**：一位人士确认了该采样方法，并指出对这种方法的看法没有显著变化。
   - 他们澄清说，目前几乎没有清晰的解释或对比。
- **Nemotron 论文对采样的批评**：一位参与者回想起了 `Nemotron` 论文，该论文批评了某些采样方法，强调一些被拒绝的回复可能只是稍差，而另一些则差得多。
   - 这种变异性可能导致过拟合以及对高质量回复的“去学习”（unlearning），因为 DPO 无法感知质量差距。
- **Zephyr 论文倾向于随机选择**：`Zephyr` 论文选择随机选择以鼓励多样性并挑战 DPO 目标。
   - 这种方法旨在平衡从“难负样本”（hard negatives）中学习的过程，避免因“假负样本”（false negatives）而向错误方向攀升。


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 messages): 

SnailBot News: <@&1216534966205284433>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1262849590390947922)** (46 messages🔥): 

> - `Science HumanEval benchmark`
> - `SmolLM models`
> - `LangChain pain points`
> - `SF Compute fundraising`
> - `Exa AI Lab Series A` 


- **Science HumanEval 基准测试挑战 AI 模型**：SciCode 发布了一个针对科学问题编程的新基准测试，其中约 10% 基于诺贝尔奖级别的研究，GPT-4 和 Sonnet 3.5 的准确率低于 5%。[阅读更多](https://scicode-bench.github.io)。
   - 关于 SciCode 的另一个视角解释说，该基准测试包含由各科学领域博士编写的 338 个挑战。[链接](https://x.com/OfirPress/status/1813202497864937825)。
- **SmolLM 模型将端侧 AI 引入浏览器**：HuggingFace 发布了 SmolLM 模型（135M, 360M, 1.7B），可以使用 ONNX 权重和 WebGPU 加速在浏览器中本地运行。[详情](https://x.com/xenovacom/status/1813258097185448377)。
- **SF Compute 为 GPU 交易平台筹集 1200 万美元**：SF Compute 获得了 1200 万美元资金，用于构建一个交易平台，允许大规模预订 GPU 集群并出售闲置部分。[彭博社文章](https://www.bloomberg.com/news/articles/2024-07-16/jack-altman-s-firm-backs-startup-for-trading-ai-computing-power)。
- **Exa AI Lab 获得 A 轮融资**：Exa AI Lab 完成了由 Lightspeed、Nvidia 和 Y Combinator 领投的 A 轮融资，以扩展其由 LLM 驱动的搜索引擎 API，并发布了重大产品更新。[详情](https://x.com/exaailabs/status/1813249325394456686)。
   - 一些用户在 Prompt 优化方面遇到挑战，并将其与 Perplexity sources API 等替代方案进行对比。
- **ColPALI 颠覆 PDF 提取**：HuggingFace 的 ColPALI 在使用 Vision-Language Models 高效执行文档检索方面展现出潜力，绕过了传统的 OCR 和解析。[阅读更多](https://huggingface.co/blog/manu/colpali) 以及 [相关推文](https://x.com/jobergum/status/1813298149051802074)。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://x.com/jobergum/status/1812044607636615667?s=46">来自 Jo Kristian Bergum (@jobergum) 的推文</a>：兴奋驱动开发（Excitement-driven development）是最好的？ColPali：利用 Vision Language Models 实现高效文档检索 👀 我对 ColPali 非常兴奋，以至于我必须演示如何在 Vespa 中表示它。页面...</li><li><a href="https://x.com/xenovacom/status/1813258097185448377">来自 Xenova (@xenovacom) 的推文</a>：介绍 SmolLM：一个新的 SOTA 系列模型，包含 135M、360M 和 1.7B 版本，非常适合端侧部署！🔥 我们还上传了模型的 ONNX 权重，这意味着它们可以在你的浏览器中本地运行...</li><li><a href="https://huggingface.co/blo">blo (bug life online)</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=qkFa6ttAk0g">Traceloop 概览</a>：Traceloop 平台及其监控功能概览</li><li><a href="https://huggingface.co/blog/manu/colpali">ColPali：利用 Vision Language Models 实现高效文档检索 👀</a>：未找到描述</li><li><a href="https://x.com/MinyangTian1/status/1813182904593199553">来自 Minyang Tian (@MinyangTian1) 的推文</a>：SciCode 是我们新的基准测试，挑战 LM 为高级论文中的科学问题编写代码解决方案。这些挑战由博士们精心设计；我们约 10% 的基准测试基于诺贝尔奖获奖...</li><li><a href="https://x.com/tom_doerr/status/1812834592161751249?s=46">来自 Tom Dörr (@tom_doerr) 的推文</a>：Claude Dev，一个自主软件工程师</li><li><a href="https://x.com/jobergum/status/1813298149051802074?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Jo Kristian Bergum (@jobergum) 的推文</a>：在 LLM 的语境下，“可靠地”是一个很重的词，这条推文是在我关于 ColPali 的推文风暴背景下发布的，由于超出了我的日常覆盖范围，可能被忽略了。ColPali 是...</li><li><a href="https://x.com/VyvyenYue/status/1811171079924449487">来自 Tianwei Yue (@VyvyenYue) 的推文</a>：论文被 @COLM_conf 接收了！如果你不知道 CoLM，它是 Neurips 级别的，但是是*有史以来第一个针对 LLM 的会议*。太疯狂了。在创办*初创公司*的同时发表论文...</li><li><a href="https://x.com/RickLamers/status/1813341037198204962">来自 Rick Lamers (@RickLamers) 的推文</a>：我领导一个秘密项目已经好几个月了……消息终于传开了！🛠️ 我很自豪地宣布 Llama 3 Groq Tool Use 8B 和 70B 模型 🔥 一个针对 Llama 的开源 Tool Use 全量微调...</li><li><a href="https://x.com/OfirPress/status/1813202497864937825">来自 Ofir Press (@OfirPress) 的推文</a>：SciCode 是我们新的基准测试，包含 338 个由物理、数学和生物学博士根据其领域论文编写的编程挑战。其中很多问题来自诺贝尔奖获奖论文！我希望...</li><li><a href="https://x.com/deedydas/status/1813598830182707261">来自 Deedy (@deedydas) 的推文</a>：公告：今天，我们启动了 1 亿美元的 Anthology 基金，这是 Anthropic 和 Menlo Ventures 的合作项目，旨在资助全球下一代 AI 初创公司的种子轮和 A 轮融资，具有独特的...</li><li><a href="https://x.com/evanjconrad/status/1813293182853198199?s=46">来自 evan conrad (@evanjconrad) 的推文</a>：嘿朋友们，@sfcompute 已筹集 1200 万美元用于构建大规模 GPU 集群的交易平台。它允许人们购买大型预留（1 亿美元以上），然后卖回他们不使用的部分。这是第一个或...</li><li><a href="https://x.com/exaailabs/status/1813249325394456686?s=46">来自 Exa (@ExaAILabs) 的推文</a>：宣布我们的 A 轮融资，由 @lightspeedvp 领投，@nvidia 和 @ycombinator 参投！🚀 Exa 是一家重新设计搜索的 AI 实验室。资金将帮助扩展我们的 API 产品，这是第一个搜索引擎...</li><li><a href="https://x.com/jobergum/status/1813126741113610421?s=46">来自 Jo Kristian Bergum (@jobergum) 的推文</a>：像 ColPALI 这样的小型 3B 模型如何能在一夜之间颠覆 PDF 提取行业，这真是令人着迷</li><li><a href="https://news.ycombinator.com/item?id=40985609">Launch HN: Traceloop (YC W23) – 使用 OpenTelemetry 检测 LLM 幻觉 | Hacker News</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1d4p1t6/comment/l6g1b3t/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://github.com/vllm-project/vllm/issues/6226">[RFC] 停止支持 beam search · Issue #6226 · vllm-project/vllm</a>：动机。摘要：为了降低系统复杂度并实现未来的优化，我们建议停止支持 beam search。目前，vLLM 支持 3 种采样类型：greedy、random 和 beam ...</li><li><a href="https://mp.weixin.qq.com/s/r9zZaEgqAa_lml_fOEZmjg">揭秘 DeepSeek: 一个更极致的中国技术理想主义故事</a>：做贡献者，而非搭便车者。</li><li><a href="https://buttondown.email/ainews/archive/ainews-to-be-named-5745/">[AINews] SciCode：HumanEval 获得了 STEM 博士级升级</a>：博士级基准测试就是你所需要的一切。2024/7/15-2024/7/16 的 AI 新闻。我们检查了 7 个 subreddit、384 个 Twitter 和 29 个 Discord（466 个频道，2228...</li>

</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1263149850376278046)** (1 条消息): 

> - `AI Agents That Matter`
> - `Latent Space Meetups`
> - `Calendar Integration` 


- **AI Agents That Matter Meetup 今日举行**：今天的 **AI Agents That Matter** Meetup 将于中午 12 点举行，由 [@142466375024115712](https://lu.ma/sgbdfhb7) 主持。
   - 点击[此处](https://lu.ma/sgbdfhb7)加入活动并将其添加到您的日历中。
- **将 Latent Space 活动添加到您的日历**：要获取新活动通知，请点击 [Latent.Space](http://Latent.Space) 页面日历上方的 RSS 图标，并选择 "Add iCal Subscription"。
   - 通过提供的链接注册活动以保持更新。



**提到的链接**：<a href="https://lu.ma/sgbdfhb7">LLM Paper Club (AI Agents That Matter) · Zoom · Luma</a>：@shivdinho 正在带领我们深入探讨 AI Agents That Matter：https://arxiv.org/abs/2407.01502。在接下来的几周里，我们需要您自愿进行快速回顾和……

  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1262875712357011588)** (4 条消息): 

> - `LlamaIndex` 介绍
> - `LlamaParse` 改进
> - 多 `Agent` 树系统
> - AI 咨询服务
> - `Scaleport AI` 案例研究 


- **LlamaIndex 及其 Agentic 能力介绍**：一段新视频介绍了 [LlamaIndex](https://twitter.com/llama_index/status/1813316626793853135) 及其 `Agentic` 能力，涵盖了 Python 和 TypeScript 中的关键框架，以及 `LlamaParse` 服务。
   - *“对于那些希望了解 LlamaIndex 在解析复杂数据方面全部潜力的人来说，这是一个极好的资源。”*
- **LlamaParse 现在更擅长复杂文档的 RAG**：基于 Markdown 的表格重建改进使 [LlamaParse](https://twitter.com/llama_index/status/1813355957491273936) 能够处理复杂表格，并具有更好的行列对齐效果。
   - *“该工具的巨大改进使其在解析复杂的表格数据时极其有用。”*
- **用于处理客户交互的新型多 Agent 树系统**：LlamaIndex 发布了一个开源 [仓库](https://twitter.com/llama_index/status/1813618002405069173)，展示了一个用于管理客户交互的复杂多 `Agent` 树系统。
   - *“该系统包括一个‘礼宾’ Agent 和多个子 Agent，以简化用户交互。”*
- **使用 LlamaIndex 咨询服务定制 AI 解决方案**：LlamaIndex 提供端到端的 AI 咨询服务，包括咨询、构思、开发和集成，以使 AI 策略与业务目标保持一致。
   - 他们的 [案例研究](https://twitter.com/llama_index/status/1813647179627774462) 展示了加速的开发阶段、通过 `LlamaParse` 改进的 OCR 以及灵活的数据处理能力。
- **Scaleport AI 使用 LlamaIndex 加速开发**：Scaleport AI 实施了 [LlamaCloud 和 LlamaIndex](https://twitter.com/llama_index/status/1813647179627774462) 以简化 AI 开发，缩短了开发周期并提高了 OCR 性能。
   - 他们报告称增强了客户演示效果，并简化了摄取管道和数据处理的设置。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://t.co/c0KYKFpELb">SCALEPORT AI</a>：全栈 AI 开发与咨询工作室</li><li><a href="https://t.co/nRR5r9PRWP">案例研究：Scaleport.ai 如何通过 LlamaCloud 加速开发并提高销售额 —— LlamaIndex，用于 LLM 应用程序的数据框架</a>：LlamaIndex 是一个简单、灵活的数据框架，用于将自定义数据源连接到大语言模型 (LLMs)。
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1262854560922472579)** (39 条消息🔥): 

> - `Vector search with images` (图像向量搜索)
> - `Metadata in ToolMetaData` (ToolMetaData 中的元数据)
> - `Property graph issues in Neo4J` (Neo4J 中的属性图问题)
> - `Query-time metadata filters` (查询时元数据过滤器)
> - `Troubleshooting CSV data with VectorStoreIndex` (排查 VectorStoreIndex 处理 CSV 数据的问题)


- **探索图像向量搜索**：一位成员询问了关于如何进行图像向量搜索以查找相似人脸图片的指导；初步建议是将数据库转换为 embeddings 并执行向量搜索。
   - 高级向量搜索可能需要更具体的工具，但目前尚未分享直接的资源。
- **澄清生成步骤中元数据的使用**：一位成员询问如何在 ToolMetaData 中添加元数据以便在生成步骤中使用，旨在根据元数据隔离内容检索。
   - 讨论建议将创建查询引擎作为一个潜在的解决方案，但同时也提出了关于如何高效隔离集合的担忧。
- **Neo4J 属性图挑战**：一位用户在创建属性图时发现了一个 bug，即实体仅被提及一次；社区讨论了潜在的修复方案，如实体链接和节点插入期间的特殊处理。
   - 分享了 "Entities"、"MENTIONS" 以及带有示例 Cypher 代码的特定查询，以进一步探讨该问题。
- **查询时元数据过滤器的复杂性**：成员们讨论了在查询时而非 retriever 实例化期间应用元数据过滤器的可行性，这引发了关于创建查询引擎的 no-op 性质的见解。
   - 建议包括可能以不同的方式存储文档，或使用独立的索引来解决过滤问题。
- **VectorStoreIndex 处理 CSV 数据的问题**：一位成员在查询由超过 50 行的 CSV 数据创建的 VectorStoreIndex 时遇到了错误答案，正在寻找处理更大数据集的方法。
   - 使用 PagedCSVReader 的建议未能解决问题，从而促使了替代策略和工具（如用于基于 CSV 记录操作的 PandasAI）的出现。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://llamahub.ai/l/readers/llama-index-readers-file?from=">未找到标题</a>: 未找到描述</li><li><a href="https://docs.pandas-ai.com/intro">PandasAI 介绍 - PandasAI</a>: 未找到描述
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1262907824803938395)** (35 条消息🔥): 

> - `Amazon order discussion` (亚马逊订单讨论)
> - `CrunchCup product feedback` (CrunchCup 产品反馈)
> - `C4A Community Talks` (C4A 社区演讲)
> - `Roger Grosse session` (Roger Grosse 会议)
> - `Recording of community events` (社区活动录像) 


- **CrunchCup 收到褒贬不一的评价**：一位成员分享了他们[新购买的 CrunchCup](https://www.amazon.com.au/CrunchCup-XL-Portable-Cereal-Spoon/dp/B08WYWQCZY)，尽管发现它实际上不能用洗碗机清洗，但仍表达了兴奋之情。
   - 反馈包括它是旅途中吃麦片的绝佳工具，但因在洗碗机中变形而让一位用户感到失望。
- **Cohere 社区演讲将被录制**：成员们确认社区活动演讲通常会被录制并上传到 [YouTube 播放列表](https://www.youtube.com/playlist?list=PLLalUvky4CLJKDaiWCumhsJpHNDhZeVll)。
   - 一位成员向其他人保证，可以查看播放列表以获取上传的最新更新。
- **Roger Grosse 的会议已上线**：[C4A Roger Grosse 会议](https://youtu.be/64BsnVbX5u8)的录像已在 YouTube 上线，题为“通过影响函数研究 LLM 泛化”。
   - *danylo_boiko* 提醒该会议已经可以访问，并分享了直接链接。
- **Special K 和 Fruit Loops 的辩论**：一场关于最喜爱麦片的轻松对话引发了对 **Fruit Loops** 和 **Special K** 的讨论，突显了口味的差异。
   - 一位用户幽默地询问 Fruit Loops 是否只适合小孩子。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/playlist?list=PLLalUvky4CLJKDaiWCumhsJpHNDhZeVll">Cohere For AI: Community Talks</a>: C4AI 社区邀请了各界嘉宾分享他们的见解和经验。这里有一些我们最喜欢的！</li><li><a href="https://youtu.be/64BsnVbX5u8">Cohere For AI - Community Talks: Roger Grosse</a>: &quot;Studying LLM Generalization through Influence Functions&quot; 论文: https://arxiv.org/abs/2308.03296 简介: 我是多伦多大学计算机科学系的副教授...</li><li><a href="https://www.amazon.com.au/CrunchCup-XL-Portable-Cereal-Spoon/dp/B08WYWQCZY">The CRUNCHCUP XL Yellow- 便携式塑料麦片杯，适合旅途中的早餐，为您喜爱的早餐麦片准备的随身麦片和牛奶容器，无需勺子或碗 : Amazon.com.au: Kitchen &amp; Dining</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1262886691996434495)** (16 条消息🔥): 

> - `自定义聊天机器人与 Fine-tuning`
> - `聊天机器人的 Moderation 模型`
> - `检测服务的高昂定价`
> - `从播客中提取语音` 


- **通过 Fine-tuning 增强自定义聊天机器人**：讨论开始于使用自定义聊天机器人的网站通常如何在自己的内容上对 **OpenAI API** 或 **Llama** 等模型进行 **Fine-tuning**，以使其与特定网站上下文相关。
   - 专家提到，**Pre-prompting** 和 **Fine-tuning** 被用于确保模型具备公司的相关知识。
- **Moderation 模型增强专注度与相关性**：成员们讨论了使用 **Moderation 模型** 来对齐聊天机器人，并确保它们坚持相关话题，防止出现离题回答。
   - 一位成员幽默地建议你只能“指导并祈祷”，而另一位成员则澄清说 **Moderation 模型** 的作用几乎就像一个过滤器。
- **检测服务价格过高**：一位成员发现 **Detection Services** 的成本高得离谱，每月执行高达 10 万次扫描的起步价为 2 万美元。
   - 他们幽默地评论说，有了这样的预算，他们可以雇佣一个人类团队来完成同样的工作。
- **从播客中提取语音**：有人询问如何从播客中自动提取语音且不产生中断。
   - 推荐使用 Eleven Labs 的 **Voice Extraction 模型**。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1262855781548556398)** (11 条消息🔥): 

> - `GPTs Agents`
> - `封禁问题`
> - `自定义 GPTs 的 PUT Actions`
> - `Vector Store Embedding 问题`
> - `超出 API 配额` 


- **GPTs Agents 存在 Context 限制**：一位成员澄清说，**GPTs Agents** 的适应能力仅限于 **Context** 长度允许的范围，如果离开 **Context Window**，它将丢失对话模式。
   - **Agent** 不会在提供的 **Context** 之外学习任何新东西，这会影响它处理持续对话的方式。
- **自定义 GPTs 中 PUT Actions 的问题**：一位成员在为自定义 **GPTs** 编写 `put` 操作时遇到困难，特别是在将查询放入请求体（body）时。
   - 该问题通过在 Weaviate 中使用 PATCH 请求得到了解决。
- **Vector Store Embeddings 无法识别名称**：一位用户报告说，他们的 **RAG** 聊天机器人能正确识别 'Emma-Jayne Wilson'，但尽管将信息嵌入到了 **Vector Store** 中，却无法识别 'Emma'。
   - 这种差异表明查询处理或索引编制中可能存在需要解决的问题。
- **超出 API 配额导致错误**：一位用户在尝试运行 **GPT-3.5 Turbo API** 调用时遇到了显示超出当前配额的错误。
   - 解决该问题的建议是：*购买额度，API 不是免费的。*


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1262881460285735012)** (3 条消息): 

> - `WebSurferAgent 设置问题`
> - `ChatGPT 内部的角色扮演` 


- **WebSurferAgent 忽略 Prompt 指令**：一位成员讨论了使用 **Autogen** 的 **WebSurferAgent** 在决定是否执行搜索时，不遵守设置指令的问题。
   - 尽管设定了评估是否进行互联网搜索的指南，但该 **Agent** 有时遵循指令，有时则不然。
- **ChatGPT 的角色扮演模板**：一位成员分享了一个在 **ChatGPT** 中创建角色扮演场景的模板，强调了融入角色性格和背景的重要性。
   - 同时也征求了改进建议。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1262881460285735012)** (3 条消息): 

> - `WebSurferAgent 问题`
> - `Autogen`
> - `技术类互联网搜索指南`
> - `角色扮演模板` 


- **WebSurferAgent 难以执行指令**：一位用户表示，使用 **Autogen** 的 **WebSurferAgent** 偶尔执行搜索时表现不一致，偏离了设定的指令。
   - 用户描述了为 **Agent** 设定何时应搜索互联网的指令目标，但指出它有时无法遵守这些标准。
- **角色扮演模板指南**：一位用户分享了一个用于 **ChatGPT** 角色扮演的模板，旨在创建引人入胜的第一人称互动。
   - 该模板包括提供背景、性格特征和关于角色的趣事的细节，并附带要求 AI 完全化身为该角色的指令。


  

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1262887569645899828)** (23 条消息🔥): 

> - `病毒式视频制作工具`
> - `LangChain 文档转换`
> - `LangChain 贡献`
> - `LangChain 与 Qdrant`
> - `MongoDB 混合搜索` 


- ****用于制作病毒式 YouTube Shorts/TikTok 的 AI 工具****：一位用户询问了关于创建病毒式 YouTube Shorts/TikTok 的 AI 工具，并对 AI 创作体育类短视频表示怀疑。
   - 他们特别寻求针对体育视频剪辑的见解和指导。
- ****将非结构化分块转换为 LangChain Document 格式****：关于如何使用 `langchain_community.document_loaders.unstructured` 中的 `UnstructuredFileIOLoader`、`UnstructuredFileLoader` 或 `UnstructuredAPIFileIOLoader` 类将非结构化分块转换为 LangChain 文档的指导。
   - 分享了一个使用 `UnstructuredFileIOLoader` 的示例供参考。
- ****开始参与 LangChain 贡献****：一位来自莱斯大学（Rice University）的硕士生表示有兴趣为 LangChain 开源社区做出贡献，并寻求入门建议。
   - 这得到了欢迎回应，鼓励其进一步参与。
- ****处理 LangChain 中的 Qdrant 错误****：一位用户在使用 Qdrant 时遇到了 'VectorParams' object is not subscriptable 错误并寻求建议。
   - 提供了可能的解决方案和相关文档链接，包括如何使用 LangChain 在 Qdrant cloud 中嵌入文档。
- ****在 MongoDB 中使用 LangChain 实现混合搜索****：一位用户计划将 MongoDB 用作 RAG 应用的向量存储，并需要实现混合搜索（Hybrid Search）。
   - 他们请求关于通过 LangChain 集成实现此目标的思路和参考资料。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://python.langchain.com/v0.2/docs/integrations/vectorstores/qdrant/#qdrant-cloud>)).">Qdrant | 🦜️🔗 LangChain</a>：Qdrant（读作 quadrant）是一个向量相似度搜索引擎。它提供了一个生产就绪的服务，拥有便捷的 API 来存储、搜索和管理带有额外负载（payload）和扩展字段的向量...</li><li><a href="http://localhost:6333",>">未找到标题</a>：未找到描述</li><li><a href="https://github.com/langchain-ai/langchain/issues/20382>)).">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知推理应用。通过在 GitHub 上创建账号来为 langchain-ai/langchain 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1263194136857542727)** (2 条消息): 

> - `生成式 AI 助手`
> - `MongoDB 与 LangChain 的混合搜索集成` 


- **Hannah AI 助手凭借其功能给人留下深刻印象**：一位成员介绍了 **Hannah**，这是一个新的生成式 AI 助手，具有**从文档中学习**、集成顶级 API（OpenAI, Anthropic, Cohere, Google, Groq, NVIDIA）以及**深度定制化**等功能。
- **在 MongoDB 上使用 LangChain 集成混合搜索**：一位成员寻求关于如何使用 LangChain 在 MongoDB 上为 RAG 应用执行**混合搜索（Hybrid Search）**的建议。
   - 他们参考了 [MongoDB 文档](https://mongodb.docs/hybridsearch)，但请求关于如何通过 LangChain 实现此集成的技巧和资源。



**提到的链接**：<a href="https://hannah.yourbestseller.ai/">Hannah</a>：利用生成式 AI 查询自定义文档的应用。

  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1263208386934210642)** (1 条消息): 

> - `MongoDB 作为向量存储`
> - `使用 LangChain 进行混合搜索` 


- **使用 MongoDB 作为向量存储**：一位成员计划使用 **MongoDB** 作为其 RAG 应用的向量存储，并需要执行混合搜索。
   - 他们正在寻找使用 **LangChain** 实现混合搜索的建议，并请求任何可用的参考资料或资源。
- **需要与 LangChain 集成的混合搜索**：该成员提到虽然有[实现混合搜索的独立文档](MongoDb provided some docs)，但他们特别需要关于将其与 **LangChain** 集成的指导。
   - 请求社区提供宝贵的建议或参考材料以协助完成此集成。

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1262885115890241567)** (15 messages🔥): 

> - `mistral mamba`
> - `Codestral Mamba release`
> - `Mathstral`
> - `Galore configs`
> - `ChatML` 


- **Mistral AI 发布 Codestral Mamba**：[Codestral Mamba](https://mistral.ai/news/codestral-mamba/) 是 Mistral AI 推出的一款专注于代码生产力的新模型，由 Albert Gu 和 Tri Dao 协助设计，可免费使用、修改和分发。
   - 与 Transformer 模型不同，它提供**线性时间推理（linear time inference）**以及建模无限长度序列的能力，使其在快速响应和高级代码推理方面非常高效。
- **测试 Codestral Mamba 的兴趣**：社区成员对尝试 Mistral AI 的新 Codestral Mamba 模型表现出极大的热情。
   - *一位成员兴奋地表示*：“太棒了 🙂 需要尝试运行一下”。
- **关于 Mathstral 的讨论**：一名成员询问是否存在 Mathstral 模型，推测这是 Mistral AI 的另一个项目。
- **分享 Galore 配置**：一位用户索要 Galore 配置，另一位用户回复称已在另一个频道（<#1111279858136383509>）中分享。
- **ChatML 的 BOS token 交换**：一位成员指出，对于 ChatML，交换 BOS token 是有意为之的。
   - 该评论是关于该主题更广泛技术讨论的一部分。



**提到的链接**：<a href="https://mistral.ai/news/codestral-mamba/">Codestral Mamba</a>：作为对克利奥帕特拉（Cleopatra）的致敬——她辉煌的命运终结于悲惨的毒蛇事故——我们自豪地发布 Codestral Mamba，这是一个专门用于代码生成的 Mamba2 语言模型，可在...下使用。

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1262921810492657716)** (6 messages): 

> - `Rank and Overfitting`
> - `Learning Rate and Overfitting`
> - `Overfitting Solutions` 


- **增加 Rank 可能有助于解决过拟合**：从理论上讲，*增加 Rank* 有助于管理过拟合。
   - *降低学习率（Learning Rate）* 通常会有所帮助，但这取决于具体情况。
- **数据集去重作为过拟合解决方案**：模型在完成第一个 Epoch 之前就出现过拟合，可以通过对数据集进行去重（deduplicating）来缓解。
   - 这种方法被建议作为防止早期过拟合的潜在补救措施。


  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1263036474694045716)** (8 messages🔥): 

> - `My Friend V1 initial feedback`
> - `AI Friend's transcriptions privacy with Open Interpreter`
> - `FRIEND + OI potential collaboration`
> - `Open Interpreter compatibility with M3 Mac`
> - `Task allocation and collaboration in roadmap` 


- **My Friend V1 初步反馈**：[ParallaxAngle 的推文](https://x.com/ParallaxAngle/status/1805313161567818030)分享了他们对“My Friend, V1”的第一印象，赞扬了它的外形设计（form factor）及其背后的团队。
   - *引用：*“Nik，它比我预想的要小。非常非常喜欢我的 Friend。祝贺你和 Based Hardware 团队。”
- **确保 Open Interpreter 和 AI Friend 的隐私**：一位成员询问是否可以使用 Open Interpreter 以私密方式与其 AI Friend 的转录内容进行交互。
   - 讨论指向了考虑隐私因素的潜在集成方案。
- **探索 FRIEND + OI 的潜在合作**：一位成员表示有兴趣探索 FRIEND 与 Open Interpreter 之间的潜在合作，并强调了与 Nik 的对话以及进一步调查的计划。
   - 该成员指出，基于 FRIEND 的日历集成，OI 集成应该是可行的。
- **Open Interpreter 在 M3 Mac 上需要测试**：有人提出了关于 Open Interpreter 与 M3 Mac 兼容性的疑问，推测 Linux 版本可能有效。
   - 反馈建议虽然未经测试，但运行 `build.py` 脚本可能会在进行微调后工作，特别是在涉及文件路径的地方。
- **明确路线图的任务分配**：成员们要求澄清由谁来挑选路线图（roadmap）中的任务，以及是否有跟踪器或术语表来促进协作。
   - 期待更多细节以简化任务分配流程并增强协作工作流。



**提到的链接**：<a href="https://x.com/ParallaxAngle/status/1805313161567818030">JediCat (@ParallaxAngle) 的推文</a>：@kodjima33 对 My Friend, V1 的第一印象 :: 用 Her 的声音 :: “Nik，它比我预想的要小。” 非常非常喜欢我的 Friend。祝贺你和 Based Hardware 团队。能...

  

---

### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1262861664168968242)** (4 messages): 

> - `接收 01 硬件`
> - `01 的使用说明`
> - `Open Interpreter 与 01 的关系` 


- **01 硬件何时发货？**: 成员们对何时能收到他们的 01 硬件感到好奇，其中一位提到他们在刚发布时就下单了。
   - *grownbabyyoda* 表达了同样的看法，想知道具体的发货时间表。
- **当前 01 使用说明的问题**: 一位成员请求关于使用 01 的文档，并提到目前的说明不起作用。
   - 他们询问是否必须先设置好 Open Interpreter 才能使用 01，或者 01 是否可以独立运行。


  

---



### **Torchtune ▷ #[announcements](https://discord.com/channels/1216353675241590815/1216353675241590818/1262872924730429471)** (1 messages): 

> - `Torchtune v0.2.0 发布`
> - `新模型与 recipes`
> - `Sample packing`
> - `社区贡献` 


- **Torchtune v0.2.0 发布了！**: [torchtune v0.2.0](https://github.com/pytorch/torchtune/releases/tag/v0.2.0) 的发布包含了许多更新、新模型以及对数据集的改进（如 sample packing）。
   - *这是社区数月以来贡献的结晶。*
- **Torchtune v0.2.0 中的新模型和数据集**: 新版本带来了大量新模型 🦙 和 recipes，以及令人印象深刻的数据集改进，例如 sample packing 🚀。
   - 查看发布日志，了解为本版本贡献功能的所有社区成员名单。



**提到的链接**: <a href="https://github.com/pytorch/torchtune/releases/tag/v0.2.0">Release v0.2.0 · pytorch/torchtune</a>: 概览 距离我们上次发布已经有一段时间了，我们在 torchtune 库中加入了大量酷炫的新功能，包括分布式 QLoRA 支持、新模型、sample packing 等！查看...

  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1262923622238916710)** (10 messages🔥): 

> - `LLAMA 3 微调问题`
> - `Torchtune nightly 版本安装`
> - `稳定的 Torchtune 版本` 


- **LLAMA 3 微调暴露出标签问题**: 一位成员观察到，在进行短时间微调后，**LLAMA 3** 生成的内容中出现了可见且重复的 **<|finetune_right_pad_id|>** 标签，而不是预期的 <|end_of_text|> 标签。
   - 另一位成员指出，这个标签是自重构以来新增的，但并未在 tokenizer 中使用，这表明问题可能与新的实现有关。
- **从 Torchtune nightly 切换到稳定版本**: 一位成员建议从 **Torchtune nightly** 版本切换到稳定版本，以解决 LLAMA 3 的标签问题。
   - 他们提到稳定版本使用的是旧版 tokenizer 实现，并表示在此期间将开始调查该问题。



**提到的链接**: <a href="https://download.pytorch.org/whl/nightly/cpu">未找到标题</a>: 未找到描述

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/)** (1 messages): 

terafo: 现在可以用了
  

---



### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1262894248995328080)** (2 messages): 

> - `关于移除 linearizer 的笔记`
> - `消息格式澄清` 


- **请求 linearizer 移除后的更新笔记**: 一位成员称赞了另一位成员的笔记，并询问在移除 **linearizer** 后是否会提供更新后的笔记。
   - *你的笔记很棒，在移除 linearizer 之后你会提供更新后的笔记吗？*
- **关于在哪里查找颜色的澄清**: 一条消息传达了颜色在第一页底部有说明。
   - *颜色在第一页的底部有描述。*


  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1262866819233087561)** (2 messages): 

> - `OpenAI 访问权限要求`
> - `用于医院账单的 LLM 规则检查器`
> - `账单规则的 Python 代码` 


- **需要 OpenAI API 访问权限**: 提到某些功能需要 **OpenAI 端的访问权限**才能运行，这一点已由 Kyle 确认。
- **LLM 作为医院账单的规则检查器**: 讨论使用 **LLM** 通过从规章制度 PDF 中提取相关信息，来检查医院账单的代码编写是否正确。
- **生成用于账单检查的 Python 代码**: 探索使用 **LLM** 将规章制度重写为 **Python 代码**来检查医院账单的想法，这可能会使过程更快、更简单。
   - 考虑使用 LLM 为代码生成测试用例以确保准确性。


  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1262890983864008805)** (1 messages): 

> - `Channel Activity`
> - `User Engagement` 


- **错过的频道更新**：一位用户对 7 月 9 日之前未查看频道并错过关键更新表示遗憾。
   - 他们后悔在那个日期之后**没有花时间查看频道**，觉得错失了机会。
- **缺乏后续跟进**：关于用户在 7 月 9 日之后未定期查看频道的感受，还有一条额外记录。
   - 用户表达了对错过正在进行的对话的**参与感缺失**。


  

---



### **AI Stack Devs (Yoko Li) ▷ #[team-up](https://discord.com/channels/1122748573000409160/1128471951963328512/1262949188254306449)** (1 messages): 

> - `Developer Opportunities`
> - `HLS and WebRTC`
> - `Backend Development`
> - `TypeScript`
> - `MongoDB` 


- **HLS 和 WebRTC 开发的工作机会**：**Observe** 正在寻找一名在 **HLS** 和 **WebRTC** 方面有经验，并具备 **Vanilla JS**、**TypeScript** 和 **MongoDB** 后端开发知识的开发者。
- **为初创公司寻找充满热情的开发者**：**Observe** 正在寻找对初创公司充满热情，并在 **HLS** 和 **WebRTC** 领域拥有专业知识的开发者。



**提到的链接**：<a href="https://observeyourfuture.com)">未找到标题</a>：未找到描述

  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1262895724065067008)** (1 messages): 

> - `Phoenix 2.0`
> - `OSS Discussion`
> - `New Features in Phoenix`
> - `Arize Product Stack` 


- **Phoenix 2.0 产品更新与未来愿景**：参加 2024 年 7 月 18 日的虚拟讨论，全面了解 **Phoenix 2.0** 的新特性，包括新的托管部署选项、实验功能和新集成。在此处[注册](https://arize.com/resource/phoenix/2.0)，了解为什么 OSS 对 AI 的持续发展至关重要。
   - 本次活动将为 Phoenix 2.0 发布周画上圆满句号，提供演示和现场问答环节，讨论 Phoenix 在更大的 **Arize 产品栈**中开发 LLM 应用的作用。
- **关于 OSS 对 AI 开发至关重要的市政厅会议**：市政厅会议将涵盖 Phoenix 2.0 的特性，如托管部署和新的实验功能。会议将解释 Phoenix 的未来愿景以及 OSS 在 AI 中的重要性。
   - 用户的反馈对 **Phoenix** 的开发至关重要，活动将包括现场问答环节以促进持续对话。



**提到的链接**：<a href="https://arize.com/resource/phoenix/2.0">Phoenix 2.0 发布周市政厅会议</a>：2024 年 7 月 18 日 10:00am PST – 11:00am PST 虚拟会议。欢迎加入我们，共同庆祝 Phoenix 2.0 发布周的圆满结束。在本次市政厅会议中，我们将涵盖...

  

---



### **AI21 Labs (Jamba) ▷ #[announcements](https://discord.com/channels/874538902696914944/874538945168408606/1263150806102970368)** (1 messages): 

> - `Python SDK updates`
> - `Async client support`
> - `Jamba-Instruct examples` 


- **AI21 Python SDK 的新更新！**：Python SDK 的最新更新现在包括对 **Amazon Bedrock** 和 **Azure AI Studio** 上 **Jamba-Instruct** 的客户端支持。[在此查看更新](https://github.com/AI21Labs/ai21-python)。
   - 此外，异步客户端支持现已在所有平台上可用，并附带新示例以简化入门流程。更多详情请见其 [LinkedIn 帖子](https://www.linkedin.com/posts/ai21_github-ai21labsai21-python-ai21-python-activity-7219341078116597762-Sxx5)。
- **异步客户端支持现已通用！**：**Jamba-Instruct** 的**异步客户端支持**现已在所有平台提供：SaaS、Amazon Bedrock 和 Azure。📖 访问 [GitHub 仓库](https://github.com/AI21Labs/ai21-python)获取更多信息。
   - 已添加新示例以简化使用 Jamba-Instruct 的开发流程，确保在各个平台上都能更轻松地开始。



**提到的链接**：<a href="https://github.com/AI21Labs/ai21-python">GitHub - AI21Labs/ai21-python: AI21 Python SDK</a>：AI21 Python SDK。通过在 GitHub 上创建账户来为 AI21Labs/ai21-python 的开发做出贡献。

  

---



---



---



---



{% else %}


> 完整的频道细分内容已为邮件截断。
> 
> 如果您想查看完整细分，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}