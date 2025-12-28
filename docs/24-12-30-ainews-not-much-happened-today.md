---
companies:
- openai
- deepseek
- google
- qwen
date: '2024-12-31T02:24:45.402646Z'
description: '**山姆·奥特曼（Sam Altman）**公开批评了 **DeepSeek** 和 **Qwen（通义千问）** 模型，引发了关于 **OpenAI**
  的创新主张及其对 **Transformer 架构**等基础研究依赖性的辩论。**DeepSeek V3** 在“误导性注意力”（**Misguided Attention**）评估中表现出明显的过拟合问题，仅解决了
  **22%** 的测试提示词，引发了对其推理和微调能力的担忧。尽管其开源地位受到质疑，但 DeepSeek V3 被声称作为开源模型已超越 **ChatGPT-4**，这标志着自
  2023 年 3 月 14 日 ChatGPT-4 发布 1.75 年后的一个里程碑。这些讨论凸显了 AI 模型性能竞争和创新可持续性方面的动态。'
id: a8d4e427-2925-4b28-868a-b6d5cc9b9f28
models:
- deepseek-v3
- chatgpt-4
original_slug: ainews-to-be-named-9002
people:
- sam-altman
title: 今天没发生什么特别的事。
topics:
- overfitting
- reasoning
- misguided-attention
- model-evaluation
- model-architecture
- finetuning
- open-source
---

<!-- buttondown-editor-mode: plaintext -->**一个安静的周正是我们所需要的。**

> 2024/12/27-2024/12/30 的 AI 新闻。我们为您检查了 7 个 subreddit、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **32** 个 Discord 社区（**215** 个频道，**5832** 条消息）。预计节省阅读时间（以 200wpm 计算）：**696 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

享受假期。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

> 所有总结均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

待完成

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1. DeepSeek V3：性能与评价**

- **[Sam Altman 正在含沙射影地攻击 DeepSeek 和 Qwen。他破防了。](https://i.redd.it/lba9xu2mqx9e1.jpeg)** ([评分: 1486, 评论: 432](https://reddit.com/r/LocalLLaMA/comments/1hphlz7/sam_altman_is_taking_veiled_shots_at_deepseek_and/)): **Sam Altman** 批评 **DeepSeek** 和 **Qwen** 模型，强调了复制现有想法的简单性与真正创新的复杂性和风险之间的对比。他在 Twitter 上的帖子引起了广泛关注，拥有 **130 万次观看**、**1,175 次转发**、**233 次引用推文**、**1.52 万个赞**和 **2,046 个书签**。
  - 许多评论者批评 **Sam Altman** 和 **OpenAI**，认为他们在声称创新的同时，严重依赖来自 **Google** 的基础研究和其他开源贡献，并指出 **OpenAI** 的工作建立在诸如《Attention Is All You Need》论文中的 **Transformer 架构**等现有技术之上。他们认为 **OpenAI** 将公共知识货币化，同时限制了对其自身研究成果的访问。
  - 有一种观点认为 **OpenAI** 的竞争优势或“护城河”值得怀疑，因为像 **DeepSeek** 和 **Qwen** 这样的模型正以更低的成本实现类似的性能。评论者指出了 **OpenAI** 过去行为的讽刺性，例如在不提供补偿的情况下抓取互联网数据，而现在却批评他人利用他们的成果。
  - 讨论中包含了对 **OpenAI** 的可持续性和创新主张的怀疑，指出 **OpenAI** 的盈利能力正受到提供更便宜类似服务的竞争对手的挑战。对话还涉及到一个更广泛的问题，即创新通常是一个累积的过程，公司在彼此的工作基础上进行构建，而不是创造完全全新的概念。


- **DeepSeek V3 在测试过拟合的 Misguided Attention 评估中表现出奇地差。** ([评分: 176, 评论: 49](https://reddit.com/r/LocalLLaMA/comments/1hpjhm0/deepseek_v3_performs_surprisingly_bad_in/)): **DeepSeek V3** 在 **Misguided Attention** 评估中表现不佳，仅解决了 13 个测试提示词中的 **22%**，表明存在显著的过拟合问题。该模型在处理已知问题的轻微变体提示词时感到吃力，这可能是由于压缩 **KV cache** 或 **MoE** 等优化导致的，并且表现出重复循环，暗示了与推理轨迹相关的微调（finetuning）问题。
  - **过拟合与推理挑战**：讨论强调了 **DeepSeek V3** 的过拟合问题，用户建议可以使用其 **DeepThink 模式**更好地评估模型的推理能力。共识是该模型在处理已知问题的变体时表现挣扎，这可能是由于预训练数据中的偏差和微调挑战造成的。
  - **Misguided Attention 与评估方法**：“misguided attention”一词引发了辩论，一些用户认为它很好地描述了评估中的问题。推理模型的评估因 API 限制而变得复杂，导致依赖网页界面，这可能会使结果产生偏差。
  - **模型架构与性能**：存在对各种模型架构的推测，一些用户指出 **DeepSeek** 模型在执行任务时比较“固执”，这可能归因于 **MoE** 架构。对话还涉及了像 **o1-mini** 这样的小型模型在特定任务中的表现，表明不同模型各有千秋。

- **很多人问：我们什么时候能拥有比 ChatGPT4 更好的开源模型？这一天已经到来了。** ([Score: 204, Comments: 106](https://reddit.com/r/LocalLLaMA/comments/1hprz6x/many_asked_when_will_we_have_an_open_source_model/)): **Deepseek V3** 据称作为开源模型已超越 **ChatGPT4**，在 **ChatGPT4** 于 **2023 年 3 月 14 日**发布 1.75 年后实现了这一里程碑。该公告通过一个 [链接](https://x.com/lmarena_ai/status/1873695386323566638) 分享。
  - **Deepseek V3 的开源状态**：对于 Deepseek V3 是否是真正的开源存在质疑，因为它使用了 **r1-lite model**，而该模型无法下载。用户对 Deepseek 超越 GPT-4 的说法表示怀疑，并指出据报道开源模型在某些方面已经超越 GPT-4 有一段时间了。
  - **模型性能与参数**：Deepseek V3 的 **Mixture-of-Experts architecture** 拥有 **671B 总参数和 37B 激活参数**，但用户对其与基准测试相比的实际表现表示怀疑。讨论强调了像 **Claude Sonnet 3.5** 这样的模型在语气和反馈整合方面优于 GPT-4。
  - **模型对比分析**：用户比较了各种模型，如 **Qwen2.5-32b** 和 **Llama 405b**，据报道它们在某些基准测试和任务中优于 GPT-4。对话还涉及了对具有类似于 **o1 mini** 能力的开源模型的渴望，并强调了 GPT-4 性能的历史背景。


**Theme 2. Cerebras 在 CS-3 上进行万亿参数训练**

- **[2024 年 12 月 10 日：Cerebras Systems + 美国能源部 Sandia National Labs 声称展示了在单个 CS-3 系统上训练 1 万亿参数模型的能力 (!) 这仅相当于同等 GPU cluster 约 1% 的占地面积和功耗。](https://www.reddit.com/gallery/1hpejko)** ([Score: 348, Comments: 66](https://reddit.com/r/LocalLLaMA/comments/1hpejko/10th_december_2024_cerebras_systems_us_energy/)): Cerebras Systems 和 **US Energy Sandia National Labs** 宣布在单个 **CS-3 system** 上成功训练了一个 **1 trillion parameter model**，声称与同等 **GPU cluster** 相比，其仅消耗约 **1% 的占地面积和功耗**。更多详情请参阅其 [新闻稿](https://cerebras.ai/press-release/cerebras-demonstrates-trillion-parameter-model-training-on-a-single-cs-3-system) 以及 [CerebrasSystems](https://x.com/CerebrasSystems/status/1867296161750536442?t=wU_lBuMzYLClIb7ja4sjvw&s=19) 和 [SandiaLabs](https://x.com/SandiaLabs?t=7yRTp8-c5zXhEN23qEhXwA&s=09) 上的相关帖子。
  - **晶圆良率与晶片缺陷**：讨论中对 Cerebras 关于无缺陷晶片的说法表示怀疑，并引用了其产品中历史上对缺陷晶片的容忍度。计算表明，考虑到 **TSMC** 报告的典型缺陷密度，实现每个晶片 99.9954% 的良率是极不可能的。
  - **硬件与性能**：训练是在由 16 个 CS-3 芯片组成的集群上进行的，而不是单个芯片，一些人认为这具有误导性。用户指出，虽然该架构可能通过将众多核心整合到单个板卡上来降低成本，但与传统的 **GPU clusters** 相比，其性能和可扩展性仍然是关键的考虑因素。
  - **Cerebras 的市场地位**：尽管技术前景光明，但 Cerebras 尚未被广泛采用，这可能是由于供应问题或缺乏面向初创公司的易用生态系统。讨论还涉及了如果 Cerebras 的硬件被证明更优越且能轻松集成到 **PyTorch** 等现有框架中，它颠覆 **Nvidia** 统治地位的潜力。


**Theme 3. 经济实惠的本地 AI：廉价 GPU 上的性能表现**

- **预算型（又称穷人版）本地 LLM** ([Score: 354, Comments: 76](https://reddit.com/r/LocalLLaMA/comments/1hpg2e6/budget_aka_poor_man_local_llm/))：一位 Reddit 用户分享了使用旧硬件构建预算友好型本地 **LLM** 设置的方案，包括 **CROSSHAIR V FORMULA-Z** 主板和 **2x P102-100** GPU，总成本仅为 **$130**。尽管在图像生成速度上有所限制，该配置能高效运行 **Phi-4-14B** 和 **llama3.2-3b** 等多种模型，响应时间不足一秒，证明了低成本、性能导向型 AI 实验的可行性。
  - **GPU 性能对比**：**RTX 3060 12GB** 被强调为 AI 任务的预算友好型选择，性能指标显示某些模型可达 **12 tokens per second**。相比之下，**4060 Ti 16GB** 可达到 **23 tokens per second**，表明只需适度的价格增幅即可获得显著的性能提升，正如[这篇 Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1hp7yft/gpu_poors_dilemma_3060_12gb_vs_4060_ti_16gb/)中所讨论的。
  - **预算硬件可行性**：虽然帖子中描述的设置成本为 **$130**，但由于需要额外组件，该价格可能无法普遍复制，潜在总成本可能达到 **$500**。然而，如果能找到合适的交易，使用矿卡（mining GPUs）和二手组件仍然可以以 **$200** 左右的价格打造出一套强大的系统。
  - **社区兴趣与实验**：该帖子激发了想要在预算范围内尝试大型模型的用户的兴趣。一些用户正在考虑使用旧的或闲置硬件进行类似的设置，并且对图像分类等其他领域的性能表现感到好奇，尽管该设置主要针对 **LLMs**。


**主题 4. SmallThinker-3B：小规模模型中的高效推理**

- **推出 SmallThinker-3B-Preview。一个类 o1 的推理 SLM！** ([Score: 303, Comments: 58](https://reddit.com/r/LocalLLaMA/comments/1hpop3y/introducing_smallthinker3bpreview_an_o1like/))：**SmallThinker-3B-Preview** 是一个基于 **Qwen2.5-3b-Instruct** 微调的新型推理模型，专为边缘部署设计，并可作为 **QwQ-32B-Preview** 的草稿模型（draft model），在 **NVIDIA 4090** 上提供超过 **70%** 的 token 处理加速。该模型使用了 **QWQ-LONGCOT-500K** 数据集，其中超过 **75%** 的样本输出 token 超过 **8K**，目前已开放用于开源研究，尽管它目前存在输出重复的问题。
  - 讨论集中在 **speculative decoding**（投机解码）及其实现上，用户分享了使用 **llama-server** 和 **vllm** 部署模型的命令行参数。提到了一种涉及 **CUDA_VISIBLE_DEVICES** 和 **tensor-parallel-size** 的特定设置，用于优化 **SmallThinker-3B-Preview** 模型的投机解码。
  - 评论强调了像 **SmallThinker-3B-Preview** 这样的小型模型在**边缘计算**方面的潜力，强调了它们在消费级 GPU 上高效运行的能力。用户表示有兴趣通过 **retrieval-augmented generation (RAG)** 功能和工具来增强这些模型，以提高知识储备和反思能力。
  - 讨论了模型的微调过程，使用了 **llama-factory** 并计划分享训练配置。值得注意的是，微调 **3B 模型** 仅需 **单块 NVIDIA 4090 或 3090 GPU** 即可完成，体现了该模型对于进一步开发的易获得性。


## 其他 AI Subreddit 摘要

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. OpenAI 的 o1 在数学和教育领域提供显著优势**

- **O1 非常擅长数学并赢得了 Putnam Exam** ([Score: 109, Comments: 84](https://reddit.com/r/OpenAI/comments/1hpgaxm/o1_is_very_good_at_math_and_wins_the_putnam_exam/)): **O1** 在 **2024 Putnam Exam** 中获得了 **8/12** 的分数，展现了卓越的数学能力，考虑到该考试的难度，这是一项重大成就。正确回答的问题包括 **A1, A2, A3, A4, A6, B3, B4** 和 **B5**，而在 **A5, B1, B2** 和 **B6** 上出现了错误。
  - **O1 的表现与评分**：讨论中有人对 O1 在 **2024 Putnam Exam** 中报告的表现持怀疑态度，一些人认为评分可能不符合该考试的严格标准。**Kevin Buzzard** 估计 O1 只做对了一道题，其他题目获得了部分分数，正如他在 [博客](https://xenaproject.wordpress.com/2024/12/22/can-ai-do-maths-yet-thoughts-from-a-mathematician/) 中所讨论的那样。
  - **训练数据与考试时间**：有澄清指出 **2024 Putnam Exam** 发生在 AI 2023 年的训练数据截止日期之后，这表明 O1 之前无法接触到考试内容，这一点已由 **Science_421** 证实。
  - **AI 的方法与人类的方法**：评论者指出，O1 经常在没有展示所有步骤的情况下得出正确答案，这更像物理学家的做法，而不是数学家的做法，后者通常会提供详细的证明。这种风格与 Putnam Exam 的评分标准不符，因为该标准重视完整的逻辑推理。


- **o1 简直是颠覆性的变革！** ([Score: 126, Comments: 64](https://reddit.com/r/OpenAI/comments/1hpo32o/o1_is_literally_a_gamechanger/)): 与 **GPT-4** 相比，O1 显著提升了学习体验，使复杂的问题集变得更易处理，并提高了用户对过程的理解，而不仅仅是提供答案。这带来了学业成绩的提高和父母认可度的增加。
  - **澄清问题**：用户注意到，虽然 **O1** 在教育环境中比 **GPT-4** 有显著改进，但在做出假设和在不寻求澄清的情况下提供错误答案方面仍然存在困难，这是许多 **LLMs** 的共同问题。建议包括需要更明确的输入要求以减轻这些问题。
  - **编程挑战**：一位用户分享了一次经历，**O1** 提供了错误的编程信息，并且尽管有相反的证据，仍固执地坚持其正确性。切换到 **4o** 后，立即得到了纠正和道歉，突显了两个模型之间性能的差异。
  - **教育影响**：**O1** 模型因其通过提供理解复杂学科的智能辅助来彻底改变教育的潜力而受到称赞，一些用户警告不要过度依赖该工具，以确保真正的学习。人们对在处理问题集时使用 **LLM** 辅助工具所产生的成绩提高的幻觉表示担忧。


- **[OpenAI、Andrew Ng 推出关于使用 o1 进行推理的新课程](https://analyticsindiamag.com/ai-news-updates/openai-andrew-ng-introduce-new-course-on-reasoning-with-o1/)** ([Score: 116, Comments: 13](https://reddit.com/r/OpenAI/comments/1hpx8cj/openai_andrew_ng_introduce_new_course_on/)): **OpenAI** 和 **Andrew Ng** 推出了一门专注于 **使用 o1 进行推理 (reasoning with o1)** 的新课程，尽管该帖子没有提供更多细节或背景。
  - 正如多位评论者所强调的，由 **OpenAI** 和 **Andrew Ng** 提供的关于 **使用 o1 进行推理** 的新课程是免费的。
  - **Andrew Ng 的课程** 通常会收到积极的反馈，尤其是他亲自教授的课程，尽管有些课程因 AI 技术的快速进步而被批评为过时。
  - 一位评论者提供了免费课程的直接链接：[Reasoning with O1](https://www.deeplearning.ai/short-courses/reasoning-with-o1/)。


**主题 2. MAMBA 模型在 Transformer 统治地位下的挣扎**

- **[D] - 为什么 MAMBA 没有流行起来？** ([Score: 134, Comments: 49](https://reddit.com/r/MachineLearning/comments/1hpg91o/d_why_mamba_did_not_catch_on/)): **MAMBA** 曾被预期会取代 Transformer，因为它具有极高的效率，在训练期间提供 **O(N)** 复杂度，在推理期间提供 **O(1)** 复杂度，同时保持了相当的准确性。尽管有这些优势，它并没有成为主导地位，可能是由于 State Space Models 的局限性或其他尚未解决的理论约束。
  - **MAMBA 的局限性**：MAMBA 模型面临着实际挑战，例如固定状态内存限制了它们处理需要动态状态追踪任务的能力，而不像 Transformer 那样利用 Self-attention 进行高效的信息检索。这些局限性在理论分析和实验中得到了强调，表明 MAMBA 在状态追踪和实际复制任务中表现吃力。
  - **Transformer 的主导地位**：Transformer 的软件和硬件栈已经非常成熟，包括 **Hugging Face** 和 **CUDA** 优化等工具，使其在大规模应用中更易于获取且更高效。这种既定的基础设施，加上重新训练模型的高昂成本，阻碍了 MAMBA 的采用，尽管它具有潜在的运行时效率优势。
  - **研究与开发**：目前的研究继续集中在改进 Transformer 架构上，诸如 **Hyena Hierarchy** 之类的创新在效率和准确性上比传统的 Attention 机制有了显著提升。这种持续的发展以及 Transformer 经证明的可扩展性表明，在格局发生重大转变之前，像 MAMBA 这样的替代方案将仍然不太受欢迎。


**主题 3. OpenAI 的 AGI 定义和经济指标**

- **[泄露文件显示 OpenAI 对 “AGI” 有非常明确的定义]**([Score: 101, Comments: 62](https://reddit.com/r/OpenAI/comments/1hpe6va/leaked_documents_show_openai_has_a_very_clear/)): **OpenAI** 对 **Artificial General Intelligence (AGI)** 的定义通过泄露的文件被揭示。虽然这些文件的细节尚未公开，但这一发现表明 OpenAI 对 AGI 有着具体且清晰的理解。
  - 讨论强调了对使用 **1000 亿美元** 作为实现 **AGI** 基准的怀疑，用户认为财务上的成功并不等同于通用智能。**CarrotcakeSuperSand** 解释说，这一指标与 **Microsoft** 交易中的一个条款挂钩，即在达到 AGI 时，Microsoft 将失去对 OpenAI 知识产权（IP）的权利，因此需要一个明确的财务门槛。
  - **Corgis_are_awesome** 澄清说，**1000 亿美元** 的数字与 **Microsoft** 的初始投资和 100 倍的利润上限有关，与 AGI 的定义是分开的。**OpenAI Charter** 规定 AGI 是指在具有经济价值的工作中超过人类能力的 AI 系统，董事会有权决定是否实现了 AGI。
  - **Class_of_22** 等人对这种基于利润的 AGI 基准所表现出的随意性表示困惑和批评，**FlugonNine** 认为对财富创造的关注反映了 **OpenAI** 内部的风险投资家思维。**Cyberdork** 幽默地批评了 **Sam Altman** 的背景，将这种对金钱的关注归因于他的商业导向职业生涯。


**主题 4. AI 在游戏和社交媒体中的角色**

- **[死掉的互联网理论（Dead Internet Theory）现已成为企业目标](https://i.redd.it/jjoft3iqzw9e1.png)** ([Score: 393, Comments: 110](https://reddit.com/r/OpenAI/comments/1hpf6re/dead_internet_theory_is_now_a_corporate_objective/)): Meta 计划在 Facebook 上引入 **AI-generated characters** 以提升用户参与度，允许通过其 AI studio 进行模仿真实人类互动的交互。据 **Financial Times** 报道，这一举措符合在数字平台中集成 AI 的大趋势，引发了人们对在线互动真实性的担忧。
  - **AI Models 的局限性**: **swagonflyyyy** 指出了 AI models 在对话语境中的局限性，指出虽然它们在后端应用中表现出色，但在直接的用户交互中往往力不从心。**Gemma2 的 27B model** 被强调为在通用聊天方面更具优势，而 AI 的角色更适合后端任务，如 moderation 和 summarization，而非前端用户交互。
  - **对 AI 操纵的担忧**: **AppropriateScience71** 和 **sdmat** 表达了对 AI 被用于操纵用户的担忧，并引用了 **BlackOps 6 的 EOMM** 作为 AI 改变游戏动态以强制执行结果的负面案例。普遍观点认为，无论是在游戏还是社交媒体中， AI 在改变用户体验方面的作用都被视为负面的，并可能损害用户参与度。
  - **AI 在社交媒体上的盛行**: **Agile-Landscape8612** 和 **OptimismNeeded** 讨论了 AI-generated content 在 Facebook 等平台上的广泛存在，而许多用户似乎对此并无察觉。这表明 AI 生成的帖子已经融入了社交媒体，禁止 bots 可能会对平台内容产生重大影响。


---

# AI Discord 摘要回顾

> 由 o1-2024-12-17 生成的摘要之摘要的摘要

**主题 1. AI 模型争夺编程霸权**  

- [**DeepSeek V3 展示复杂的编程技能**](https://openrouter.ai/deepseek/deepseek-chat)：它能够处理大 context windows，在构建 MTG（万智牌）卡组等任务中表现出色，并超越了一些闭源模型。然而，它在“推理循环（reasoning loops）”和 XML 输出方面仍有困难，显示出还有改进空间。  
- [**Gemini 2.0 以速度赢得青睐**](https://github.com/google-gemini/cookbook)：用户称赞 Gemini 的“闪速思考（flash thinking）”在编程辅助方面的表现，声称其速度有时超过 GPT-4。他们还期待 Gemini 即将推出的针对代码生成等专门任务的功能。  
- [**Codeium 2024 Wrapped 确认新年功能**](https://codeium.com/wrapped-2024)：该平台提供了年终编程统计数据，同时透露 2025 年“仍有大量工作要做”。用户对 Windsurf 的停机和额度消耗既感到兴奋又感到沮丧。

**主题 2. 微调与 LoRA 的基础工作**  

- [**LoRA 被证明有用但很棘手**](https://huggingface.co/docs/peft/main/en/developer_guides/lora#eva)：开发者认为它能保留新知识，但警告不要有虚高的期望，并注意数据集陷阱。讨论中经常提到大规模预训练中的 *overfitting*（过拟合）风险。  
- [**Hymba-1.5B-Instruct 走向商业化**](https://huggingface.co/nvidia/Hymba-1.5B-Instruct)：它因开源指令数据集和“严格的 batch size 要求”而受到称赞，引发了法律和伦理使用方面的疑问。贡献者将其视为构建稳健 AI 解决方案的垫脚石。  
- [**OpenRouter 与 Aider 集成**](https://aider.chat/docs/config/options.html)：程序员在通过 OpenRouter 连接 DeepSeek V3 时遇到了“未找到模型”的错误。通过正确的环境变量和 endpoint 设置解决了该问题，从而实现了流线型的微调工作流。

**主题 3. 量化与 HPC 性能**  

- [**FP8 策略加速 Transformer 引擎**](https://github.com/NVIDIA/TransformerEngine)：NVIDIA 的 FP8 方法承诺在保持强精度的同时减小数值占用。用户强调了来自 [PyTorch 博客](https://pytorch.org/blog/accelerating-gemms-triton/) 的新型 2D 块量化，可实现接近 2 倍的加速。  
- [**TMA 与 cp.async 引发辩论**](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/)：更少的线程和寄存器使得 TMA 比 cp.async 更具资源效率。开发者在 HPC 任务中看到了巨大收益，尤其是基于 GEMM 的工作负载。  
- [**3090 NV-Link 与 Jetson Orin Nano 面临考验**](https://www.jeremymorgan.com/blog/tech/nvidia-jetson-orin-nano-speed-test/)：多 GPU 桥接吸引了追求性能的人，但噪音和成本担忧依然存在。与此同时，Jetson Orin Nano 的 *25W 模式* 在适度但实用的端侧 AI 尝试中表现令人印象深刻。

**主题 4. RAG、Embeddings 与 Agent 工作流**  

- [**使用 LlamaIndex 实现本地 RAG**](https://t.co/4WJ7ZcXy3H)：用户将 Excel 表格输入到 Llama-3.2 或 Llama-3.3，实现了高级的检索增强生成（RAG）。Neomagus 验证“导入的引用”以防止 AI 幻觉。  
- [**Light Prompter 展示高效的测试时性能**](https://github.com/Green0-0/light_prompter)：它通过批量处理提示词来加快模型推理，开发者想知道 *test time training*（测试时训练）是否也会调整模型权重。其他人则认为这与 *real-time updates*（实时更新）的 RL 研究有相似之处。  
- [**视觉遇见 Embeddings**](https://nomic.ai/blog/posts/gpt4all-scaling-test-time-compute)：Nomic 的 *nomic-embed-vision-v1* 与文本 embeddings 配合使用以优化图像搜索。这种方法预示了 GPT4All 及其他领域的 *multimodal expansions*（多模态扩展）。

**主题 5. API、定价与 Prompt Engineering**  

- [**OpenRouter 用户权衡成本**](https://openrouter.ai/rankings/translation?view=week)：一些人对输入 token 没有折扣感到遗憾，而 *GPT-4o mini* 等模型的性能推动了翻译友好的使用。供应商正通过“利基”模型优势展开差异化竞争。  
- [**Perplexity Pro 令订阅者困惑**](https://docs.perplexity.ai/api-reference/chat-completions#body-search-recency-filter)：尽管 DeepSeek v3 备受推崇，但它在订阅服务中缺失，导致用户呼吁坚持使用免费层级。与此同时，“推理模式”将复杂的查询整合到结构化答案中，用于高级问答。  
- [**Prompt Engineering 走向结构化**](https://x.com/sh_reya/status/1873431565650502060)：过于宽泛的请求会使 AI 代码工具感到困惑，因此开发者将任务分解为更小的步骤。人们关注“Sora 频道”和 Markdown 友好空间，以进行有效的知识共享。

---

# 第 1 部分：高层级 Discord 摘要

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Codeium 2024 年度回顾与新年路线图**：团队推出了 [Codeium 2024 Wrapped](https://codeium.com/wrapped-2024)，鼓励大家以酷炫的方式查看并分享编程统计数据，随后表达了温馨的年度回顾致谢。
   - 他们暗示 2025 年将推出更多 **features**，并强调 *仍有大量工作要做* 以提升用户体验。
- **Windsurf 的频繁停机与额度困境**：用户报告了 **Windsurf** 响应缓慢和 503 错误，促使一些人推动建立 [status page](https://stats.uptimerobot.com/aiH8grJl1y) 以获取实时更新。
   - **premium credits** 耗尽引发的挫败感导致了退款要求，以及探索 **ChatGPT 4o** 等替代方案以应对重复的停机时间。
- **DeepSeek V3 的期待仍在持续**：针对 **DeepSeek V3** 在 Windsurf 中集成延迟出现了不耐烦的讨论，用户看到 **Cline** 等竞争对手工具已率先采用。
   - 围绕功能优先级的疑问不断，一些人敦促 Codeium 加快合并速度，以在 AI 编辑器竞赛中保持领先。
- **Codeium 中的上下文混乱**：围绕 **Codeium** 如何处理代码修订的上下文长度展开了激烈辩论，许多人对实际限制与营销宣传之间的差异感到困惑。
   - 尽管该平台宣称针对高级用途具有高上下文长度，但用户发现维持代码讨论方面存在持续性问题。
- **React Native SVG 故障**：一位用户详细描述了尽管 Web 预览完美，但在原生模拟器上加载 **SVG icons** 出现的问题，怀疑是与 `react-native-svg` 和 Expo 存在版本冲突。
   - 社区成员建议在对应用设置进行剧烈重新配置之前，先调试平台兼容性和库版本。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **微调中的 LoRA 准备工作**：成员们辩论了 **LoRA** 对于大规模预训练是否有效，指出仔细的数据集结构化对于避免过拟合和过高预期至关重要（[文档链接](https://huggingface.co/docs/peft/main/en/developer_guides/lora#eva)）。
   - 他们分享了*以往的经验*，承认对 LoRA 在知识保留方面可靠性的怀疑，并参考了 [持续预训练技巧](https://unsloth.ai/blog/contpretraining)。
- **Llama.cpp 中的量化困惑**：一些用户在最近的库更新后遇到了 **Llama.cpp** 的 **quantization issues**，导致集成过程中出现错误（[示例问题报告](https://github.com/unslothai/unsloth/issues/1333)）。
   - 讨论集中在缺少依赖项以及针对 **Phi 4** 等大型模型缺乏 unsloth 量化，突显了*操作延迟*和库版本不匹配的问题。
- **Hymba 商业用途的热度**：**Hymba-1.5B-Instruct** 模型发布，声称可供商业使用并有*严格的 batch size 要求*，详见 [Hugging Face](https://huggingface.co/nvidia/Hymba-1.5B-Instruct)。
   - 贡献者指出该模型源自开源指令数据集，提醒大家分发先进 AI 技术时的*法律和伦理考量*。
- **Light Prompter 提升测试时效率**：GitHub 项目 [Light Prompter](https://github.com/Green0-0/light_prompter) 展示了增加 **model inference efficiency** 的批处理策略，包含相关的 notebook 和代码示例。
   - 一位成员提到了*测试时训练（test time training）*以及它如何在推理过程中更新权重，其他人则认为这可能与尚未完全探索的 **RL** 研究存在重叠。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Claude 3.5 Sonnet 引发猜测**：用户质疑 **claude-3.5-sonnet** 是否与 **claude-3.5-sonnet-20241022** 不同，并引用了 [Cursor 论坛的一个帖子](https://forum.cursor.com/t/are-claude-3-5-sonnet-and-claude-3-5-sonnet-20241022-different/24272/3)。
   - 他们注意到 **claude-3.5-sonnet** 现在重定向到更新后的 **20241022** 版本，这引发了对性能提升的好奇。
- **Composer 与 Chat 的对决**：一些人称赞 **Composer** 工具在代码优化方面的表现，甚至指向了[关于快速“修复”操作的讨论](https://forum.cursor.com/t/how-to-do-fix-in-composer-and-fix-in-chat-actions-from-keyboard/31221)。
   - 另一些人则看重 **Chat** 的通用指导功能，并建议偶尔使用更直接甚至带有情绪的语气能让 **Cursor** 给出更精准的回复。
- **Cursor 助力 Web 应用开发**：一位用户展示了如何在没有深厚编程背景的情况下，利用 **Cursor** 为一款移动端 MMO 游戏交付了一个功能齐全的 Web 工具。
   - 另一位用户分享了一个**吉他和弦学习应用**的链接，例如这个 [指板工具](https://guitar-tab.vercel.app/en/tools/fretboard)，突显了 Cursor 在全栈原型开发中的实用性。

---

## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Grok 的额度倒计时**：在年底前仅剩两天之际，**Grok AI** 为其 API 用户提供 25 美元的免费额度，详见此[官方链接](https://x.com/stackblitz/status/1873769044761022633)，这些额度可以集成到 **Bolt** 项目中。
   - 成员们强调，这最后的几小时是在 Bolt 中尝试 **Grok AI** 的绝佳时机，称其为“快速原型开发的黄金期”。
- **Bolt 的语音提示愿望**：社区强烈呼吁加入类似 **ChatGPT** 的语音提示功能，以提供更便捷的代码讨论方式，但也注意到了音频模型带来的更高开销。
   - 爱好者们憧憬在 **Bolt** 中实现“免提交互”，但他们预见到由于模型复杂性增加，可能会导致成本飙升。
- **Supabase vs Firebase vs Convex：数据库抉择**：开发者们权衡了在 **Bolt** 项目中使用 **Supabase**、**Firebase** 或 **Convex** 进行数据托管的优劣，并参考了一个 [GitHub](https://github.com/stackblitz/bolt.new/issues/4455) Issue 获取详情。
   - 一些人强调导出到 **StackBlitz** 可以进行手动优化，而另一些人则提醒 **Convex** 仍处于 Beta 阶段，可能需要谨慎使用。
- **大型代码库导致的 LLM 疲劳**：社区成员注意到 **Bolt** 在处理大型代码库时速度变慢，偶尔会修改无关文件，导致需要反复重启和进行差异检查（diff checks）。
   - 用户建议通过重新加载项目和切换 **diff mode** 来减少随机编辑，并分享了一些成功案例，表示这有助于控制 Token 使用。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **DeepSeek V3 势头强劲**：许多用户正转向使用 **DeepSeek V3** 处理编码任务，称赞其庞大的 Context 窗口并引用了 [API 文档参考](https://api-docs.deepseek.com/quick_start/pricing)。一些用户权衡了自托管与使用 Hugging Face 的隐私折中方案，并提到了成本和 Context 窗口的差异。
   - 其他人将其与 **Gemini** 的代码生成能力进行了对比，结论是 **DeepSeek** 速度更快，尤其是在大型项目中，同时称赞新推出的 [Context Caching](https://api-docs.deepseek.com/news/news0802#how-to-use-deepseek-apis-caching-service) 功能非常省钱。
- **Aider 的安装与配置**：爱好者们强调为了稳定性应全局安装 **Aider**，参考了 [官方指南](https://aider.chat/docs/install.html) 和特定的 Python 设置步骤。一些 Arch Linux 用户提供了特定操作系统的技巧，并指出调整 `.aider.model.metadata.json` 有助于管理 Context 和成本。
   - 他们还讨论了绕过 Git 限制的方法，指向了 [GitHub issue #211](https://github.com/Aider-AI/aider/issues/211)，同时承认了意识到 Token 限制的重要性。
- **Gemini 2.0 在代码方面表现出色**：贡献者报告称 **Gemini 2.0** 能有效处理大型项目，其提供的免费层级有助于加速编码任务。他们频繁引用 [LiteLLM 上的模型提供商](https://docs.litellm.ai/docs/providers)，强调了在大型代码库中的性能提升。
   - 一些人依靠 **Gemini** 进行广泛的代码加载，同时使用像 **DeepSeek** 这样的专业模型进行最终生成，以利用每个模型的特性。
- **将 Aider 与 OpenRouter 集成**：某些成员在将 **OpenRouter** 绑定到 **Aider** 时遇到了“未找到模型”错误，归因于端点配置错误。他们通过启用特定设置并验证正确的环境变量克服了这一问题，参考了 [OpenRouter 集成技巧](https://aider.chat/docs/config/options.html)。
   - 其他人对托管端点的用户隐私表示担忧，但指出一旦配置得当，**Aider** 可以通过 **OpenRouter** 无缝调用 **DeepSeek**。
- **使用 TesseractJS 实现 OCR**：一位用户展示了如何使用 **Aider** 在一小时内构建一个 Web App，并采用 [TesseractJS](https://github.com/naptha/tesseract.js/) 执行自动化 OCR 任务。他们强调了跳过手动编码而转向直接由 AI 驱动生成的生产力提升。
   - 社区成员看到了将 OCR 与代码生成相结合的潜力，预示着未来将扩展到高级文本提取工作流中。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **LLM 基准测试失误**：参与者发现 **LLM 性能** 可能会受到含糊问题的干扰，并引用了 [ARC 'Challenge' vs ARC 'Easy'](https://arxiv.org/abs/2412.17758) 作为设置存疑的例子。
   - 他们建议将重点从多项选择题转向 **Functional**（功能性）任务，以捕捉复杂的推理能力，并就采用稳健的指标展开了公开讨论。
- **Gradient Routing 备受关注**：成员们称赞 **Gradient Routing** 是一种在反向传播期间使用数据依赖掩码来隔离模型能力的方法，参考了 [一篇关于定位计算的论文](https://arxiv.org/abs/2410.04332)。
   - 该技术可以通过将特定子区域映射到某些任务来提高可解释性，从而为高级调试提供见解。
- **TongGeometry 的卓越定理**：**TongGeometry** 系统地提出并解决了奥林匹克级别的几何问题，如 [通过引导树搜索提出并解决奥数几何问题](https://arxiv.org/abs/2412.10673) 中所述。
   - 一些解法甚至进入了*区域数学奥林匹克*，突显了该模型在处理复杂几何证明方面的出色能力。
- **Crosscoders 破解模型层级**：**Crosscoders** 方法跨多个层级跟踪特征，以更好地解释模型如何演变表征，参考了 [一个开源复现项目](https://www.lesswrong.com/posts/srt6JXsRMtmqAJavD/open-source-replication-of-anthropic-s-crosscoder-paper-for)。
   - 实践者希望这种方法能精确找出网络中细微的转换，从而辅助*电路简化（circuit simplification）*和直接的模型差异对比（diffing）。
- **TinyStories 策略**：根据 [TinyStories: How Small Can Language Models Be](https://arxiv.org/abs/2305.07759)，**TinyStories** 数据集汇编了合成短篇故事，用于训练 **1000 万参数以下的小型 LM**。
   - 用户报告称在不出现重大性能下降的情况下开发出了更简单的架构，激发了人们对*轻量级模型设计*的兴趣。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeepSeek V3 在 OpenRouter 上表现不佳**：一些用户报告称，通过 [OpenRouter](https://openrouter.ai/deepseek/deepseek-chat) 使用 **DeepSeek V3** 时性能有所下降，引发了关于更新或版本变动的猜测。
   - 他们怀疑最近的修改或可能的 Together API 因素可能在起作用，这引发了对性能一致性和用户信心的担忧。
- **OpenRouter 欢迎新的 LLM 提供商**：社区成员指出，将模型集成到 **OpenRouter** 需要与成熟的实验室建立合作伙伴关系或进行 Self-hosting，而专门的编程能力是一个强大的差异化因素。
   - 他们指出 [OpenRouter 上的 Prompt Caching](https://openrouter.ai/docs/prompt-caching#deepseek) 是节省成本的关键，并建议推广利基优势以吸引用户兴趣。
- **GPT-4o mini 在翻译方面表现出色**：关于翻译模型的讨论将 **GPT-4o mini** 定位为可靠的选择，而据称 **Gemini 1.5 Flash** 经常出错。
   - 用户提到了结构化的 System Prompts，并依靠 [LLM 翻译排行榜](https://openrouter.ai/rankings/translation?view=week) 来优化他们的结果。
- **多模态 Agent 引起关注**：开发者探索了构建多模态 Agent 的方法，并澄清严格的 JSON 输出对于 Agent 工作流并非强制要求。
   - 他们参考了 [Anthropic 关于构建高效 Agent 的指南](https://www.anthropic.com/research/building-effective-agents)，并提到 Google 的 **Project Mariner** 可能是灵感来源。
- **价格辩论升温**：社区成员注意到 **OpenRouter** 缺乏输入 Token 折扣，强调了高吞吐量使用的成本影响。
   - 虽然一些人对潜在的模型降级表示担忧，但其他人呼吁对性能变化给出透明的解释。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DeepSeek 的差异化演示**：**DeepSeek V3** 在通过 Scryfall 查询构建 MTG（万智牌）卡组等任务中表现优异，在 [Aidan 的基准测试](https://x.com/aidan_mclau/status/1872444303974543859) 中排名第 22 位，其先进的上下文保留能力令人印象深刻。
   - 然而，使用 [MisguidedAttention](https://github.com/cpldcpu/MisguidedAttention) 进行的评估揭示了其存在**推理循环**和矛盾的结果，引发了对其架构的质疑。
- **本地 AI vs. API：对决还是共生？**：成员们权衡了 **Aquila's Ollama** ([ollama.com](https://ollama.com)) 和 **LlamaCPP** 在本地设置中的定制化优势，同时确认 **OpenAI API** 对于 Agent 任务仍然至关重要。
   - 其他人呼吁更多人向 **LlamaCPP** 贡献代码，理由是它在开源 AI 项目中的影响力，并强调了本地加 API 解决方案的协同效应。
- **SmallThinker-3B 带来惊喜**：在 [Hugging Face](https://huggingface.co/PowerInfer/SmallThinker-3B-Preview) 上发布的全新 **SmallThinker-3B-preview** 显示出改进的推理基准测试结果和系统化步骤的技巧。
   - 尽管如此，成员们开玩笑说它无法在正确的时间停止，这表明它在探索可能性时可能会过度生成（Overgenerate）回复。
- **混元（Hunyuan）的 8GB 策略**：正如 [一篇博客文章](https://blog.comfy.org/p/running-hunyuan-with-8gb-vram-and) 中所解释的，**混元**视频模型可以在仅有 **8GB VRAM** 的 GPU 上运行，尽管在低分辨率下运行缓慢。
   - 社区成员指出了**速度问题**，指出较小的配置为资源受限的设置打开了大门，但可能会阻碍高保真输出。
- **关键指标**：在**二分类（Binary Classification）**讨论中，成员们主张报告来自 sklearn 的 **Precision**、**Recall**、**F1** 和 **AUC/ROC**，以增加清晰度。
   - 他们强调了**具有代表性的测试集**的价值，并敦促将指标与每个模型的现实目标相对齐。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Deepseek v3 缺席 Pro 订阅**：社区成员注意到 **Deepseek v3** 在 **Perplexity Pro** 订阅中明显缺失，这引发了对其声称的权益和高级功能的困惑。
   - 一些人质疑是否应该坚持使用 **免费版 Deepseek**，理由是用户对支付了 Pro 费用却看不到高级功能感到沮丧。
- **Reasoning Mode 强化复杂查询**：用户强调了 **Perplexity Pro** 中的 **Reasoning Mode** 在详细问答中的作用，它会在处理复杂查询时自动开启以提高准确性。
   - 他们分享了将数据整理成表格的示例，强调了利用结构化布局获取稳健答案的共同兴趣。
- **Claude 3.5 Sonnet 对决 GPT-4O**：多位用户讨论了 **Claude 3.5 Sonnet** 和 **GPT-4O** 之间的性能权衡，提到了可靠性和延迟方面的差异。
   - 他们指出在特定任务中可能与 **Deepseek** 或 **ChatGPT Pro** 产生协同效应，并强调没有单一模型能主宰所有场景。
- **寻找 API 替代方案与新鲜度过滤器（Recency Filters）**：一位用户正在寻找超越当前标准的 **Search API** 解决方案，并询问了 **自定义新鲜度过滤器**，参考了 [Perplexity API 文档](https://docs.perplexity.ai/api-reference/chat-completions#body-search-recency-filter)。
   - 关于过滤器的可行性尚未出现明确答复，这激发了社区对探索高级数据检索新搜索范式的兴趣。
- **会话式 API 使用困惑**：有疑问提出 **Perplexity API** 是否能提供上下文驱动的回复，而非词典式的定义。
   - 一份回复确认 **Sonar models** 旨在进行带有适当引用的问答，并澄清它们并非旨在作为通用的会话式 Agent。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI 生成讨论升温**：讨论涵盖了 **图像生成** 工具的优缺点，提到了海报生成结果的不一致性，以及 **Claude** 和 **Eleven Labs** 等模型的多样化表现。
   - 一些参与者对繁重的清理工作表示沮丧，而另一些人则描述了音频和视频生成工作流的改进，并引用了一个关于 [模型不可预测性的 Reddit 帖子](https://www.reddit.com/r/ClaudeAI/s/bO3cOogG6c)。
- **B-STaR 论文聚焦自我改进**：成员们发现了 **B-STaR**：[Monitoring and Balancing Exploration and Exploitation in Self-Taught Reasoners](https://arxiv.org/abs/2412.17256)，该论文倡导通过极少的人类标注和自我改进训练方法来实现高级推理。
   - 一位用户引用了 [Reddit 帖子](https://www.reddit.com/r/ClaudeAI/s/bO3cOogG6c) 来强调社区讨论，认为这些技术可以在未来的 AI 逻辑中实现持续优化。
- **Gemini 2.0 实力增强**：多位成员称赞 **Gemini 2.0** 的 Flash-thinking 和编程能力，特别是其在速度和集成易用性上相对于 GPT-4 的优势。
   - 他们指出它可能会填补 OpenAI 当前产品线在特定任务中的空白，并讨论了将其推向标准编程辅助之外的应用。
- **Prompt Engineering 与 Sora 频道拆分**：随着用户希望针对 ChatGPT 及相关模型的高级 **Prompt Engineering** 概念建立更多结构，要求设立专门 **Sora** 频道的呼声日益高涨。
   - 爱好者们还在寻求正式的 **Prompt Engineering** 课程，并意识到随着模型更新的演进，最佳实践的变化速度之快。
- **Token 限制引发调整**：成员们在与 **GPT-2** 的 1024 Token 限制做斗争，而另一些人在通过 OpenAI 的 API 生成长篇博客文章时面临可行性问题。
   - 他们讨论了内容分块或采样替代模型的方案，并参考了一个 [Discord 帖子](https://discordapp.com/channels/974519864045756446/1315696747279810711/1323428129083097158) 中解决 Token 约束的方法。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 音频冒险**：讨论涵盖了在注明出处的情况下**公开**重复使用 NotebookLM 音频，提到目前为止尝试过的案例均未产生负面后果，并开玩笑地评论说*目前还没有人被逮捕。*
   - 一些社区成员在发布 [YouTube 视频](https://www.youtube.com/watch?v=4rdXYdMmrFg)和[链接](https://youtu.be/ubA36TeoyM4)时遇到了不一致的限制，将其归因于速率限制 (rate limiting) 或更新后的审核设置。
- **嵌入 NotebookLM 以实现交互影响**：成员们提议将 **NotebookLM** 嵌入外部网站以支持访客查询，建议采用爬虫或未来的 API 连接等方法。
   - 他们还请求增加“事后录制”功能，以保存对话的关键片段，强调内置录制功能可以更方便地进行回顾。
- **NotebookLM Plus 权益与限制**：许多讨论集中在 Plus 用户的 **500** 个笔记本上限与免费账户的 **100** 个上限，并引用 [NotebookLM Help](https://support.google.com/notebooklm/answer/15678219) 以求明确。
   - 他们还提到了 MP3 文件的上传错误以及生成输出中的覆盖范围缺口，突显了影响高级使用的系统约束。
- **Gemini 2.0 播客怪癖**：[gemini-2-podcast 仓库](https://github.com/agituts/gemini-2-podcast)展示了生成基于 **Gemini 2.0** 音频的 Python 脚本，尽管它在删除并重新渲染整个音频之前会忽略新文件。
   - 其他人注意到 **NotebookLM** 可能会跳过或误读用户源文件，这激发了对官方 API 和移动端支持的兴趣，以简化跨平台访问。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **M2 Max MacBook Pro 引发性能辩论**：工程师们质疑配备 32GB RAM 和 38 核 GPU 的 **M2 Max MacBook Pro** 是否能有效处理本地 AI 工作负载，并强调了其与 Nvidia GPU 配置的区别。
   - 有些人认为它可以使用，但其他人警告说，在 Apple 硬件上处理真正的重型任务可能会感到力不从心。
- **深度图 (Depth map) 失败困扰创作者**：用户在使用来自 3D 软件的深度图时遇到了**条带 (banding)** 伪影，导致模型解释了非预期的边缘。
   - 他们建议调整最大深度级别，并坚持使用符合 **Stable Diffusion** 要求的格式。
- **LoRa 训练锁定一致风格**：一位童书插画师学会了通过在 **Stable Diffusion** 中训练 **LoRa** 来保持水彩角色设计的一致性。
   - 他们将参考照片与专门的 LoRa 微调相结合，以实现统一的插画风格。
- **AI 视频创作平台引发好奇**：成员们探索了 **Luma Dream Machine**、**Kling** 和 **Minimax** 等云端解决方案，用于快速进行 AI 视频测试。
   - 他们讨论了成本因素、硬件需求，并分享了 [Webui Installation Guides](https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides) 以及[此 YouTube 演示教程](https://www.youtube.com/watch?v=vY4QwnR4R2M)。
- **Discord 社区应对垃圾信息担忧**：几位用户推动建立更强大的**审核 (moderation)** 工具来对抗机器人活动，并考虑了审查制度对模型输出的影响。
   - 他们担心更严格的保护措施可能会阻碍角色生成，尤其是在处理人体解剖结构时。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **静态 Mojo 与 Python 传统**：用户讨论了 **Mojo** 中 **static methods** 的含义和用法，担心它可能偏离 Python 的方式。
   - 他们建议复制 Python 当前的行为以保持一致性，并提到需要与 [Modular Docs](https://docs.modular.com/mojo/stdlib/builtin/rebind/rebind/) 上的现有 **rebind** 文档同步。
- **递归 Struct 对决**：在 **Mojo** 中使用 `UnsafePointer[Self]` 定义 **recursive structs** 触发了 segmentation faults（段错误）。
   - 切换到 **ArcPointer** 或 **OwnedPointer** 提供了更安全的处理方式，尽管一些开销是不可避免的。
- **Mojo 的 'Load' 技巧提升 SIMD 速度**：参与者强调，在 **Mojo** 中处理 SIMD 数据时，使用 **load** 比直接 bitcast 更好。
   - 他们引用了 [Performance Notes](https://www.computerenhance.com/p/table-of-contents)，强调了正确的内存访问对速度至关重要。
- **指针父子关系难题**：在 Mojo 的递归数据结构中维护 **child and parent pointers** 考验了用户的耐心。
   - 他们支持使用 `OpaquePointer` 作为避开指针纠缠和 optional pointer 陷阱的一种方法。
- **调试模式崩溃 (#3917)**：在全调试模式下运行 **Mojo** 会触发 segmentation faults，而正常运行时表现较好。
   - 开发者指出 [issue #3917](https://github.com/modularml/mojo/issues/3917) 将在假期后解决，社区目前仍在等待修复。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 速度飙升**：用户报告称，在 LM Studio 中使用 [DeepSeek-V2.5-1210-GGUF model](https://model.lmstudio.ai/download/lmstudio-community/DeepSeek-V2.5-1210-GGUF) 时，性能提升高达 **20 倍**，达到 **6 t/s**，并使用 Perf Monitor 跟踪 GPU 使用情况。
   - 他们还引用了 [Nomic.ai blog post](https://www.nomic.ai/blog/posts/gpt4all-scaling-test-time-compute)，讨论了设备端 LLM 在 **code interpreter** 和 **tool calling** 方面的实时扩展。
- **Vision Models 审查检查**：一位用户发现“受审”的 **Vision Models** 会屏蔽 NSFW 内容，引发了对非审查方法的兴趣。
   - 同样，他们探索了高级功能，并考虑使用 *special configurations* 的潜在变通方法。
- **3090 NV-Link 与噪音难题**：社区成员讨论了双 **3090** 配置的 **NV-Link**，质疑 2x2 桥接是否优于单卡，同时还要兼顾更长的线缆。
   - 其他人警告说 **blower fans** 噪音可达 **83 dB**，建议在运行 *inference tasks* 时使用 **water cooling** 来降低噪音。
- **Jetson Orin Nano 的 25W 测试**：一位用户在 **25W mode** 下使用 **Jetson Orin Nano** 测试了 20 个模型，引用了 [a blog post](https://www.jeremymorgan.com/blog/tech/nvidia-jetson-orin-nano-speed-test/) 获取真实速度数据。
   - 随后讨论了 *quantizing models* 和优化 watts-per-token，以实现更紧凑或基于边缘的 LLM 部署。



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **TMA 对决 cp.async**：参与者展示了 **TMA** 如何通过启用更少的线程和使用更少的寄存器来超越 **cp.async**，从而降低资源开销。
   - 他们强调了对 HPC 任务的潜在提升，并指向了 [this GEMM series on Hopper GPUs](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) 以获取相关示例。
- **2 的幂驱动 MAGVIT-v2**：社区成员解释了 **MAGVIT-v2** 如何利用二进制量化，将 9 等小数编码为 [0][1][0][0][1][0] 来表示 2 的幂。
   - 他们引用了 **Dominika Przewlocka-Rus** 的工作，建议与拉普拉斯分布对齐，引发了更多关于潜在位移性能增益的讨论。
- **ThunderKittens 与 Triton 之争**：成员宣布 **ThunderKittens** 将添加整数 matmul 算子，展示了对自定义 kernel 的持续实验。
   - 他们讨论了经过精心调优的 **TK/CUDA** kernel 是否能超过 **Triton**，并提到了 Triton 在细粒度异步执行和寄存器处理方面的限制。
- **Raspberry Pi 5 GPU 测试**：爱好者报告称，尽管原始算力有限，但 **Raspberry Pi 5** GPU 在较小的 vision 工作负载中表现出潜力。
   - 他们发现使用 6–8bit 量化的大型 **LLMs** 性能缓慢，引发了关于 **Vulkan** 基准测试以及与 Intel CPU 比较的问题。
- **GPU 领域的顶级技术职位**：分享的一个 [cracked research engineer job](https://crackedengineers.com/job/p-1-ai-7f41fa30-6cfa-4e9a-8943-2324dc21d243) 突显了 GPU 和 AI 开发中的专业角色。
   - 该小组建议搜索 **CUDA** 和 **Triton** 关键词，反映出对高级 GPU 专业知识日益增长的需求。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **值班混乱：AI 代码带来的困扰**：一位用户指出了 [Shreya Shankar 的这条推文](https://x.com/sh_reya/status/1873431565650502060)，讨论了 AI 生成代码给值班人员带来的负担，并呼吁加强文档记录和测试。
   - 其他人建议开发者将任务分解为更小的步骤，以便 **LLM** 能够有效地管理它们，而不是盲目地处理整个复杂的特性。
- **Kagi 之争：寻找搜索优势**：用户称赞 **Kagi Assistant** 灵活的搜索能力，尽管有人指出其覆盖范围与 **Perplexity** 相比仍有差距。
   - 爱好者们期待即将推出的功能，包括搜索 API，预计将与同类工具展开更激烈的竞争。
- **峰会火花：2025 AI Engineering 聚会**：**AI Engineering Summit** 定于 2025 年 2 月 20 日至 21 日在纽约举行，据报道，此前的活动得到了主要科技赞助商的支持。
   - 组织者鼓励尽早预注册以获得特殊访问权限，旨在促进 AI 专业人士和行业领袖的聚会。
- **Cursor 难题：是协作还是混乱？**：多位开发者分享了对 **Cursor** AI 编程助手的挫败感，描述了在处理复杂编程任务时浪费的精力。
   - 他们建议在与 AI 工具配对时，通过明确指令和使用迭代式问题陈述来减少摩擦。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **榜首并列：Chatbot Arena**：Chatbot Arena 显示 **OpenAI 的 o1** 跃升至并列第一，比 o1-preview 增加了 24 分，并超越了其他竞争对手，如 [排名第 7 的 DeepSeek-V3](https://x.com/lmarena_ai/status/1873695386323566638)。
   - 社区讨论指出 **Claude 较低的排名** 令人费解，拒绝回答和角色扮演问题被认为是可能的原因。
- **SLM 挑战 The Bitter Lesson**：关于 **SLM**（小型语言模型）如何通过使用专门的先验知识在特定任务中表现出色引发了辩论，质疑了对更多数据和算力的追求。
   - 参与者引用了 **Llama 3 8B** 超越 GPT-3 175B 的例子，并强调了特定领域解决方案的重要性。
- **DeepSeek V3：XML 输出问题与基准测试**：成员们对 **DeepSeek V3** 难以正确输出 **XML** 标签感到沮丧，它产生了类似 r1 的推理过程而不是执行指令。
   - 他们还质疑了从 V2.5 切换 Prompt 后的指令遵循性能，并对 Post-training 结果给出了负面反馈。
- **GRPO vs. Vineppo：RLHF 竞争**：讨论集中在 **GRPO (Group Relative Policy Optimization)** 及其奖励平均化，并与 Vineppo 的单样本策略和片段中途重置进行了对比。
   - 一位用户解释说 **DeepSeek V3** 使用了 GRPO，这引发了对 1b–7b 模型内存限制以及舍弃 Value Network 可能性的担忧。
- **Gary 与 Miles 就 2027 年 AI 发展轨迹进行对赌**：社区对 [Gary Marcus 的帖子](https://garymarcus.substack.com/cp/153809626)做出了回应，该帖子披露了他与 Miles Brundage 就未来 AI 成就达成的共同赌注。
   - 怀疑论者的言论包括声称我们离“4”还“疯狂地遥远”，对模型能力的短期飞跃表示谨慎。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All 中的 LLaMA 3.3 获得 Groq Key 支持**：用户分享了通过 [Groq.com](https://groq.com/) 将 **LLaMA 3.3** (70B) 与 **GPT4All** 连接的步骤，以启用云端 LLM 支持。
   - 他们强调了成本优势，指出这节省了 AI 工作负载的本地硬件开销。
- **Gemini API 支持引发热议**：参与者讨论了 **Gemini** 与 OpenAI API 的兼容性以及 **Gemini 2.0** 的路线图，并引用了 [google-gemini/cookbook](https://github.com/google-gemini/cookbook)。
   - 他们表示一旦官方确认 GPT4All 集成，就有兴趣使用 Gemini 的独特功能。
- **Jinja 抖动引发聊天模板问题**：最近的 GPT4All 更新引入了 **Jinja** 解析，导致旧聊天模板的语法损坏。
   - 贡献者建议重置默认模板或参考更新后的文件，鼓励协作修复。
- **Vision Embeddings 成为关注焦点**：成员们澄清说 **nomic-embed-vision-v1** 与文本嵌入模型配合使用，可以通过文本查询优化图像搜索。
   - 他们将 Nomic 的视觉模型与其他公开选项进行了比较，期待在未来版本中看到更强大的演示。
- **Ollama 模型导出引发讨论**：爱好者们探索了在 GPT4All 中重用 **Ollama** 模型的方法，参考了 [Ollama Model Export Script](https://gist.github.com/supersonictw/f6cf5e599377132fe5e180b3d495c553)。
   - 他们讨论了将 Ollama 指定为 LLM 引擎，并指出其与 OpenAI 风格 API 的兼容性。

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Breathe.ai 签署 NDA 测试 Cohere**：Breathe.ai 正式通过签署 NDA 加入 Cohere，旨在合作开发研究原型。
   - 成员们表示热烈欢迎，并希望建立更深层的技术交流和反馈循环。
- **HMM Tokenization 查询引发好奇**：几位用户询问了关于 **HMM (Hidden Markov Model)** Tokenization 技术的问题，凸显了共享专业知识方面的空白。
   - 目前尚未出现即时的建议，反映出用户对扩展高级 NLP Tokenization 方法知识的兴趣。
- **Cohere 的 Rate Limit 风波**：成员们遇到了 Image Embed 的 Rate Limit 与预期不符的问题，预期为每分钟 **400** 次调用，但实际观察到为 **40** 次。
   - 支持团队确认了 [Rate Limit 文档](https://docs.cohere.com/v2/docs/rate-limits) 并保证正在修复中，重申生产环境 Key 的官方上限仍为 **400**。
- **Fine-tuning 攻坚战继续**：一位用户报告了 **Fine-tuning** 错误，担心可能存在数据或配置问题。
   - 支持团队正在调查由假期引起的延迟，承诺将进行直接沟通并升级故障排除流程。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **显著的 Matching 加速**：关于 Matching 函数 **8 倍加速** 的说法引发了激烈讨论，目标是通过悬赏将 **400ms** 降低到 **50ms**。
   - 持怀疑态度者指出 **50%** 的运行时间消耗在这些函数中，并讨论了即使是 **2x** 的加速可能才是更现实的目标。
- **重写风波：2.5 倍收益，4/7 失败**：对 `full_graph_rewrite` 的调整带来了 **2.5x** 的模型重写时间提升，但 **7 个测试中有 4 个** 随即报错，需要紧急调试。
   - 多线程被认为是改进的一个方向，同时建议使用更小的测试集来锁定根本问题。
- **AM Driver 马拉松目标 11,000 行**：George Hotz 承诺将 **AM driver** 扩展到 **11,000** 行并在年底前合并，并引用了 [此 commit](https://github.com/tinygrad/tinygrad/commit/0addbad36d414cc37e69e92fa9e1f26045cbf1f6) 作为进展标志。
   - 与会者期待 **周一上午 9:30** 在圣迭戈举行的 **Meeting #51**，以削减 Scheduler 清理方面的技术债并推进 AM driver。
- **Tinygrad CUDA 碾压 Torch**：新的 Benchmark 表明 **Tinygrad CUDA** 几乎比 Torch 快 **两倍**，**OpenCL** 减少了约 **1ms** 的开销。
   - 开发者建议使用 `Device[out.device].synchronize()` 来获取精确指标，并指出 **JIT** 速度在 **第三次运行** 时才会真正发挥作用。
- **Frame Evaluation Hook 热议**：社区成员强调了来自 [PEP 523](https://peps.python.org/pep-0523/) 的 **Frame Evaluation Hook API** 是在 Python 中直接捕获运行的一种便捷方式。
   - 他们指出 Torch 的 Dynamo 编译器依赖于这种方法，称其比后期捕获方案更具灵活性。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **本地 Llama-3.2 与 Neomagus 安全法律引用**：开发者讨论了使用 **Llama-3.2** 和 [Llama Index tools](https://t.co/4WJ7ZcXy3H) 构建本地 RAG 应用，以无缝查询 **Excel tables**。
   - 他们还强调了使用 **Neomagus** 验证 AI 生成文本中的引用，详情分享在[此处](https://t.co/g5toC0m3T9)，希望能减少虚假引用。
- **Llama 3.3 GPU 占用与 Ollama 的角色**：一位用户询问了 **Llama 3.3 70B** 的 GPU 需求，并参考了一个潜在的 **Hugging Face** 端点。
   - 另一位用户在本地测试了 **Ollama**，运行 `ollama run llama3.3` 时显示内存占用约为 **2.77GB**，表明这是一种更节省内存的方法。
- **Bagel 为开源 AI 提供变现方案**：一位代表展示了 **Bagel**，这是一个帮助 **open source AI developers** 赚取收入并与 **Hugging Face** 同步的平台。
   - 他们分享了一条[推文](https://x.com/BagelOpenAI/status/1873776090516488257)，解释了这种新颖的架构如何在提供 **Llama-3.3** 等先进模型的同时，让开发者保持控制权。
- **过滤非词汇声音以提高音频清晰度**：一位用户探索了使用 LLM 去除 *ahh* 和 *um* 等杂音，引发了对优化 **audio editing** 工作流的兴趣。
   - 参与者指出，清理填充词可以增强教育和专业录音的**听觉体验**。
- **LlamaParse API 加速数据操作**：成员们讨论了用于直接集成的 **LlamaParse API**，并在[官方文档](https://docs.cloud.llamaindex.ai/llamaparse/getting_started/api)中展示了上传和检查解析任务的示例调用。
   - 他们强调了无缝处理结构化数据的优势，并参考了针对真实 RAG 场景的 [GitHub examples](https://github.com/run-llama/llama_parse/tree/main/examples)。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **LLM Agents MOOC 重新开放报名**：下一期 **LLM Agents** 课程将于 1 月底开始，可通过[此表单](https://forms.gle/9u6HdVCWXgws16go)报名。
   - 报名者可以参考即将发布的 [Spring 2025 syllabus](https://llmagents-learning.org/sp25) 以及 [Fall 2024 materials](https://llmagents-learning.org/f24) 提前准备。
- **证书邮件将于 1 月发送**：早期 **LLM Agents** MOOC 的证书将于 1 月底通过电子邮件发送，尽管部分参与者仍在等待。
   - 成员们确认在等待期间可以访问[课程网站](https://llmagents-learning.org/f24)重新温习课程材料。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Dynamo 问题减少**：报告显示 **Dynamo errors** 可能已解决，促使成员考虑移除禁用编译器的设置以获得更好的性能。
   - 一位用户建议在启用和禁用编译模式的情况下分别验证加速效果，并强调要进行彻底的 **regression checks**（回归检查）。
- **Flex 的下一个里程碑将于 1 月 13 日到来**：成员们期待即将于 **1 月 13 日** 发布的 **2.6.0** 版本中的 **Flex** 更新，期望其在 **2.5.1** 之上有所改进。
   - 他们注意到已经引入了多项调整，希望这些修改能在最终发布前完成集成。
- **Simple Eval 与 LM Eval 的对决**：一位成员向大家推荐了 [OpenAI 的 Simple Eval 库](https://github.com/openai/simple-evals)，将其作为 **lm eval** 工具的潜在替代方案。
   - 辩论集中在 **evaluation**（评估）速度和兼容性上，参与者查看了 GitHub 页面以了解具体的实现细节。
- **FP8 特性推动 Transformer Engine**：用户讨论了 **FP8 quantization**（量化）策略，引用了 [NVIDIA 的 Transformer Engine](https://github.com/NVIDIA/TransformerEngine) 和 [Microsoft 的 Automatic Mixed Precision Library](https://github.com/Azure/MS-AMP)。
   - 他们还强调了 2D 块量化方法，引用了 [COAT](https://github.com/NVlabs/COAT)、PyTorch 的 [Float8 GEMMs 博客](https://pytorch.org/blog/accelerating-gemms-triton/)，以及 *mixed-precision training*（混合精度训练）论文，如 [arXiv:2310.18313](https://arxiv.org/pdf/2310.18313) 和 [arXiv:2409.12517](https://arxiv.org/pdf/2409.12517)。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **OS 模式：支持视频吗？**：一位用户询问 **OS mode** 是否可以接受 **video** 作为输入，希望能明确其适用范围。
   - 目前尚未出现**确定的解决方案**，但人们对多媒体支持的兴趣日益浓厚。
- **隔离方案的犹豫：Docker vs. OS**：用户提到了 [隔离文档](https://docs.openinterpreter.com/safety/isolation)，并想知道它管理的是操作系统锁定，还是 **Docker** 和 **E2B** 的使用。
   - 附带的一张图片引发了困惑，暗示文档中的术语存在歧义。
- **Windows 1.0：构建支持**：有人询问关于新发布的 **1.0 dev** 版本的 **Windows build**。
   - 跨平台爱好者正等待支持，以确认是否会实现广泛的 OS 兼容性。
- **配置文件大迁移：YAML 转 PY**：用户在将 **1.0.0** 中的 **profiles.yaml** 迁移到新的 **.py** 格式时遇到困难。
   - 他们质疑文档的准确性，并担心保存流程。
- **自定义 API Base URL 的困扰**：一位用户希望在 Ubuntu 上通过 **gpt4o** 或 **claude-35-sonnet** 等端点复制 **OpenAI** 风格的使用方式。
   - 他们在设置过程中遇到了障碍，并请求帮助适配这些自定义的 base URL。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Arxiv 2412.15563 引起关注**：一位用户询问对 [Arxiv 论文 2412.15563](https://arxiv.org/abs/2412.15563) 的看法，寻求其对大语言模型更广泛影响的解答。
   - 虽然没有提供直接分析，但人们有兴趣了解它是否适合 **DSPy** 实验。
- **AI 术语表势头正盛**：一位成员引入了一个 **AI Glossary** 以加速概念引用，灵感来自 [使用 DSPy 和 Claude 从 Jekyll 博客生成术语表](https://www.dbreunig.com/2024/12/27/generating-a-glossary-from-a-jekyll-blog-usign-dspy-claude.html)。
   - 他们强调了语言与技术之间的相互作用，并指出仍有大量术语等待更精确的定义。
- **Openhands 与 DSPy 的结合**：有人提出将 **Openhands** 作为一个 one-shot 非交互式工具，返回聊天响应和 git diffs，从而引发了将其集成到 **DSPy** pipeline 中的讨论。
   - 他们认识到潜在的协同效应，但也指出了 **DSPy** 处理 prompt tuning 和自动化方式中的设计细微差别。
- **反馈系统激发代码探索**：一位用户提议建立一个系统，记录对自动化代码更改的反馈以供后续评估，重点在于输入/输出日志记录。
   - 他们计划利用这些数据点来指导 **DSPy** pipeline，根据历史结果优化代码质量。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **FFmpeg 切片技术受到关注**：一位用户描述了一种收集时间戳然后应用 **FFmpeg** 剪切视频内容的方法，并称赞了指令的清晰度。
   - 他们对该流程表示满意，称其为*一种实现快速编辑的直接方法。*
- **2025 年黑客松与会议热潮**：有人正在寻求 2025 年黑客松和会议的建议，目前已确定参加 **ICML**、**NeurIPS** 和 **CVPR**。
   - 他们希望结识更多社区成员，并热切邀请大家提供更多建议。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **排行榜 Zero-Shot 难题**：他们澄清，认可的模型必须在 **zero-shot** 环境中进行测试，即产生单次响应且无迭代调用。
   - 如果用户只调用一次，**API endpoint** 方法可以绕过典型限制，参考了 API 背后的 **OpenAI o1** chain-of-thought 逻辑。
- **确保评分安全的单次调用**：他们强调，高级的 chain-of-thought 扩展必须对用户不可见，在排行榜评估中强制仅执行一次 **API call**。
   - 这种机制通过禁止在单次评估中进行多步生成或重复尝试，保持了排行榜的一致性。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Axolotl AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**HuggingFace Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}

### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1322681294974353448)** (1 条消息): 

> `Codeium 2024 Wrapped, Upcoming features` 


- **Codeium 2024 Wrapped 发布**：团队宣布发布 **Codeium 2024 Wrapped**，邀请用户通过 [此链接](https://codeium.com/wrapped-2024) 查看并分享他们的统计数据。
   - 频道里充满了兴奋的气氛，团队感谢大家度过了不可思议的 **2024** 年，并暗示将有更多功能推出。
- **展望新年功能**：消息强调了在新的一年里致力于发布更多 **features** 以提升用户体验。
   - 根据公告，*仍有很多工作要做*，因为他们正准备在 **2025** 年进行进一步改进。



**提到的链接**：<a href="https://codeium.com/wrapped-2024">Codeium Wrapped 2024 | Windsurf Editor and Codeium extensions</a>：在 Codeium 2024 Wrapped 中查看你最常用的语言、编码时长、编码习惯等更多内容！

  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1322304733368680489)** (194 条消息🔥🔥): 

> `Windsurf performance issues, User login problems, Codeium pricing frustrations, Alternative IDEs, Error messages in Codeium` 


- **Windsurf 面临性能和停机困扰**：用户报告 **Windsurf** 性能缓慢且频繁停机，导致对额度浪费和登录问题的沮丧。
   - 在 Codeium 处理服务器过载期间，许多人正在考虑 Aide 和 Cody 等替代方案。
- **Codeium 登录困难**：一位用户表达了尽管重新安装了应用程序并尝试了各种故障排除步骤，仍无法登录其账户的挫败感。
   - 其他人的建议包括强制关闭应用程序并检查操作系统设置。
- **对 Codeium 额度系统的担忧**：几位用户对他们的 **premium credits** 消耗速度过快感到不满，尤其是在最近系统更改之后。
   - 由于意外问题导致额度过度消耗，有人呼吁进行潜在的退款。
- **讨论 Windsurf 的可能替代方案**：鉴于目前存在的问题，用户正在探索 **ChatGPT 4o** 和其他开源工具作为临时解决方案。
   - 一些人对这些替代方案与 Windsurf 相比的有效性持怀疑态度。
- **Codeium 报错且未返回消息**：用户报告在尝试与 Codeium 中的 chat 功能交互时出现错误，导致重复提问却得不到回复。
   - 许多人建议开启新对话或重启应用程序，作为解决响应性问题的潜在方案。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://codeium.com/windsurf/show-auth-token?authType=signin&from=redirect">Provide Authentication Token to VSCode | Windsurf Editor and Codeium extensions</a>：Codeium 是开发者喜爱且企业信赖的 AI 代码助手平台。也是首个 Agentic IDE —— Windsurf 的构建者。</li><li><a href="https://codeium.com/settings">Windsurf Editor and Codeium extensions</a>：Codeium 是开发者喜爱且企业信赖的 AI 代码助手平台。也是首个 Agentic IDE —— Windsurf 的构建者。
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1322295225095553100)** (633 条消息🔥🔥🔥): 

> `Windsurf service outages, DeepSeek V3 integration, Context length issues in Windsurf, User experiences with AI code suggestions, SVG loading issues in React Native`

- **Windsurf 服务持续中断**：用户报告 Windsurf 频繁出现服务中断，在高负载期间遇到 503 错误和响应缓慢。
   - 这引起了用户的不满，许多人建议需要一个状态页面（status page）来监控服务的可用性。
- **DeepSeek V3 尚未集成**：关于将 DeepSeek V3 集成到 Windsurf 和 Cursor 的讨论仍在继续，用户对其实现表示迫切期待。
   - 像 Cline 这样的类似工具已经能够更快速地完成集成，这引发了用户对新功能优先级排序的质疑。
- **Windsurf 中的 Context length 困惑**：有一场关于 Codeium 使用的 Context length 及其与 Windsurf 关系的讨论，用户对相关限制感到困惑。
   - 虽然有人提到 Codeium 提供很高的 Context length，但用户表示在代码修改过程中维持上下文存在挑战。
- **对 AI 代码建议的不满**：几位用户对来自 Sonnet 的 AI 代码建议表示沮丧，指出存在不必要的重构和复杂的 Prompt 等问题。
   - 建议包括专注于特定的编码任务，并有效利用项目指令（project instructions）来提高响应质量。
- **React Native 中的 SVG 加载问题**：一位用户报告了在 React Native 原生模拟器中加载 SVG 图标的问题，这与 Web 端预览成功的情况形成对比。
   - 他们怀疑 React Native、native-svg 和 Expo 之间的版本兼容性问题是该问题的潜在原因。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.codeium.com/windsurf/advanced">Windsurf - 进阶</a>: 未找到描述</li><li><a href="https://tenor.com/view/nounish-nounsdao-nouns-dao-noggle-gif-26326389">Nounish Nounsdao GIF - Nounish Nounsdao Nouns - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/roblox-roblox-outage-blox-fruits-update-gif-14794492378318604921">Roblox Roblox 停机 GIF - ROBLOX ROBLOX OUTAGE BLOX FRUITS UPDATE - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://docs.codeium.com/getstarted/overview">未找到标题</a>: 未找到描述</li><li><a href="https://tenor.com/view/bored-crashing-faceplant-sleepy-pass-out-gif-8482195">无聊崩溃 GIF - 无聊崩溃 脸着地 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://code.visualstudio.com/docs/devcontainers/containers">使用 Visual Studio Code 远程开发在容器内进行开发</a>: 使用 Visual Studio Code 远程开发在容器内进行开发</li><li><a href="https://chat.deepseek.com/">DeepSeek</a>: 与 DeepSeek AI 聊天。</li><li><a href="https://codeium.com/plan">计划设置</a>: 未来的编辑器，就在今天。Windsurf Editor 是首款由 AI Agent 驱动的 IDE，让开发者保持心流状态。现已支持 Mac, Windows 和 Linux。</li><li><a href="https://tenor.com/view/clapping-leonardo-di-caprio-gif-13334985">鼓掌的莱昂纳多·迪卡普里奥 GIF - 鼓掌的莱昂纳多·迪卡普里奥 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/whats-going-on-down-there-concerned-whats-going-on-gif-15556206">下面发生了什么 担忧 GIF - 下面发生了什么 担忧 发生了什么 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/no-brain-loading-slow-gif-8465847256202919615">无脑加载 GIF - 无脑加载 缓慢 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/zerosgifs-gif-20855209">Zerosgifs GIF - Zerosgifs - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://codeium.com/live/">与 Codeium 聊天 | Windsurf Editor 和 Codeium 扩展</a>: 使用 Codeium Live 进行通用聊天。Codeium 是开发者喜爱且企业信赖的 AI 代码助手平台。同时也是首款 Agentic IDE Windsurf 的开发者。</li><li><a href="https://stats.uptimerobot.com/aiH8grJl1y">状态页面</a>: 未找到描述</li><li><a href="https://github.com/unixwzrd/venvutil">GitHub - unixwzrd/venvutil: Python 虚拟环境管理功能和脚本，用于构建和管理性能、兼容性和回归测试的 venv 构建，主要针对 AI</a>: Python 虚拟环境管理功能和脚本，用于构建和管理性能、兼容性和回归测试的 venv 构建，主要针对 AI - unixwzrd/venvutil</li><li><a href="https://codeium.com/changelog">Windsurf Editor 更新日志 | Windsurf Editor 和 Codeium 扩展</a>: Windsurf Editor 的最新更新和变更。</li><li><a href="https://codeium.com/blog/termium-codeium-in-terminal-launch">Termium: 终端中的 Codeium</a>: 为您的终端命令提供 AI 驱动的自动补全。</li><li><a href="https://www.eridepros.com/">竞赛级越野电动摩托 | E Ride Pro </a>: “在 E Ride Pro 选购高性能电动摩托。发现适合各种技能水平的环保、快速且耐用的电动越野自行车。”
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1322309152646762596)** (705 条消息🔥🔥🔥): 

> `LLM 模型微调，Token 在训练中的作用，开源与模型共享，LLM 的量化问题，Hymba 模型概述`

- **Fine-tuning LLM Models**：几位用户讨论了微调 LLM 模型的策略，强调了正确结构化数据集的重要性以及对 early stopping 机制的需求。
   - 对话强调了微调时面临的挑战，包括学习率过高可能导致 overfitting 的潜在风险。
- **Role of Tokens in Training**：Sadaisystems 提出了使用特定 token 格式（如 XML）训练模型对模型性能和理解能力影响的问题。
   - 有人指出，模型在 inference 过程中可能会识别自定义 tokens，但训练对于构建有效的相关权重至关重要。
- **Open Source and Model Sharing**：参与者讨论了开源软件的挑战，特别是关于权力集中以及这与先进 AI 技术分配的关系。
   - 针对开源社区中的法律和伦理考量提出了担忧，强调了尊重 licenses 的必要性。
- **Quantization Issues with LLMs**：Renegade2611 报告了 Llama.cpp 在 quantization 方面的问题，指出在集成过程中遇到的错误可能与该库最近的更新有关。
   - 还讨论了对于像 Phi 4 这样的大型模型缺乏兼容的 unsloth quantization 的问题，由于操作延迟，该模型尚未发布。
- **Hymba Model Overview**：介绍了 Hymba-1.5B-Instruct 模型，强调了其功能以及该模型已准备好投入商用，并有特定的 batch size 要求。
   - 分享了关于其基于利用开源指令数据集的基础模型开发的细节，以及在生成过程中了解其局限性的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing#scrollTo=QmUBVEnvCDJv">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing#scrollTo=6bZsfBuZDeCL">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing#scrollTo=juQiExuBG5Bt">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing#scrollTo=6bZsf">Google Colab</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/finetuning-from-last-checkpoint#wandb-integration">从上一个 Checkpoint 进行微调 | Unsloth 文档</a>: Checkpointing 允许你保存微调进度，以便你可以暂停并继续。</li><li><a href="https://www.youtube.com/@WillCogley">Will Cogley</a>: Will Cogley 致力于融合机械、电子和一点艺术创意，以制作顶级的机器人和电子动画。这些创作都经过精心记录并发布，以便任何人...</li><li><a href="https://huggingface.co/nvidia/Hymba-1.5B-Instruct">nvidia/Hymba-1.5B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://www.kaggle.com/code/ahmedess/llama-3-2-vision-finetuning-unsloth-kaggle">Llama 3.2 Vision 微调 Unsloth - Kaggle</a>: 使用 Kaggle Notebooks 探索并运行机器学习代码 | 使用来自“无附加数据源”的数据</li><li><a href="https://unsloth.ai/blog/contpretraining">使用 Unsloth 进行 LLM 持续预训练</a>: 通过使用 Unsloth 对 Llama 3、Phi-3 和 Mistral 进行持续预训练，让模型学习一种新语言。</li><li><a href="https://www.anninrobotics.com/">Annin Robotics</a>: Annin Robotics - 开源且经济实惠的机器人 - 构建你自己的 6 轴机器人。</li><li><a href="https://x.com/danielhanchen/status/1872719599029850391">Daniel Han (@danielhanchen) 的推文</a>: DeepSeek v3 论文中的亮点：1. Float8 在前向和后向中使用 E4M3 - 无 E5M2 2. 每 4 次 FP8 累加会增加到主 FP32 累加器 3. Latent Attention 存储 C cache 而非 KV cache 4. 无 M...</li><li><a href="https://docs.beam.cloud/v2/environment/custom-images#conda-environments">容器镜像 - Beam</a>: 未找到描述</li><li><a href="https://www.stephendiehl.com/posts/unsloth/">Unsloth 快速教程</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/peft/main/en/developer_guides/lora#eva">LoRA</a>: 未找到描述</li><li><a href="https://gist.github.com/grahama1970/f832bbddb1edaa78ccc939a6f2ddd8a1">对于动态适配器加载和推理，Unsloth 推理工作正常——使用 Hugging Face 则无法工作——输出乱码</a>: 对于动态适配器加载和推理，Unsloth 推理工作正常——使用 Hugging Face 则无法工作——输出乱码 - hf_only_inference_sanity_check.py.py</li><li><a href="https://youtu.be/7pdEK9ckDQ8?feature=shared&t=31"> - YouTube</a>: 未找到描述</li><li><a href="https://github.com/vllm-project/llm-compressor?">GitHub - vllm-project/llm-compressor: 与 Transformers 兼容的库，用于将各种压缩算法应用于 LLM，以实现 vLLM 的优化部署</a>: 与 Transformers 兼容的库，用于将各种压缩算法应用于 LLM，以实现 vLLM 的优化部署 - vllm-project/llm-compressor</li><li><a href="https://github.com/confident-ai/deepeval">GitHub - confident-ai/deepeval: LLM 评估框架</a>: LLM 评估框架。通过在 GitHub 上创建账户，为 confident-ai/deepeval 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth/issues/1333">使用 Qwen 2.5 7B 训练时的问题 · Issue #1333 · unslothai/unsloth</a>: from trl import SFTTrainer from transformers import TrainingArguments, DataCollatorForSeq2Seq from unsloth import is_bfloat16_supported trainer = SFTTrainer( model = model, tokenizer = tokenizer, t...</li><li><a href="https://robotnanohand.com/">首页</a>: 介绍 Robot Nano Hand 开源项目。3D 打印、构建并编程这款最先进的人形机器人手。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1322980034763554989)** (8 条消息🔥): 

> `WSL Ubuntu 设置, 社区感谢, Computer Vision 项目, 新年祝福, 服务器赞赏` 


- **WSL Ubuntu 上的运行时错误**：一位用户在 WSL Ubuntu 上尝试保存模型时遇到了 **RuntimeError**，提示 **llama.cpp** 目录中缺少文件。
   - 经过排查，他们通过 `apt-get` 安装了 **curl** 和必要的库，解决了该问题。
- **新年祝福与社区支持**：一位成员对 Unsloth Discord 社区提供的学习经验表示感谢，并祝愿大家**新年**身体健康。
   - 另一位成员热情回应，表达了同样的感激之情。
- **对 Computer Vision 的憧憬**：一位成员分享了他们近期对 **computer vision** 的关注，并希望在 **2025** 年之前开展 fine-tuning 相关工作。
   - 这种热情反映了尽管有时间表，但仍致力于在该领域取得进展。
- **对社区的增强支持**：一位用户表达了对 Jed.T 的强烈支持，强调了该 Discord 服务器和 **Unsloth 框架** 的卓越性。
   - 这反映了成员之间日益增长的社区感和协作精神。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1322348282172801035)** (171 条消息🔥🔥): 

> `LoRA 及其应用, Fine-tuning 大语言模型, 语言翻译中的挑战, 理解模型性能和训练数据集, AI 和 LLM 的学习资源` 


- **探讨 LoRA 在预训练中的功效**：一位成员询问利用 **LoRA** 进行大规模预训练是否有助于模型保留新知识，另一位成员对其可靠性表示怀疑。
   - *一些人分享了之前的经验*，强调在对待性能预期时应采取谨慎态度。
- **语言翻译模型 Fine-Tuning 的陷阱**：一位参与者对在为新语言 fine-tuning **Llama 3.1 8B** 模型时出现的不一致翻译结果表示沮丧，并质疑持续预训练的效果。
   - 另一位贡献者强调了固有的挑战，指出语言数据的基础知识对于可靠的翻译能力至关重要。
- **面向准 AI 开发者的学习资源**：**新开发者**被建议从 AI 领域入手，重点探索来自 OpenAI 和 Gemini 的现有 AI 文档，并了解 LLM 的历史演变。
   - 参与者讨论了在深入研究 AI 和 LLM 应用的具体实现之前，理解基础概念的重要性。
- **探索在 Instruct 模型上进行 Fine-Tuning 的有效性**：在关于 fine-tuning Instruct 模型与 base 模型的讨论中，有人提到对于某些应用，对 base 模型进行**预训练**通常更有益。
   - 成员们一致认为，训练方法的差异会导致效果的不同，这取决于具体用例的数据细微差别和数据量。
- **理解 Cut Cross Entropy 的实现**：对 **cut cross entropy** 的技术概述展示了它在 Unsloth 库中特定条件下的自动启用，并附带了展示其代码实现的代码片段。
   - 讨论揭示了模型推理函数中 `fused_linear_cross_entropy` 的集成，这有助于潜在的性能提升。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/phi4">使用 Unsloth Fine-tune Phi-4</a>: 使用 Unsloth Fine-tune 微软最新的 Phi-4 模型！开源且对初学者友好。</li><li><a href="https://arxiv.org/abs/2401.13586">指令 Fine-Tuning：Prompt Loss 重要吗？</a>: 我们提出了一项新颖的研究，分析了各种 Prompt Loss Token 权重 (PLW) 对监督指令 Fine-Tuning (SIFT) 的影响。虽然 Prompt-masking (PLW = 0) 在 SIFT 中很常见，但一些 fine-tu...</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama#id-8.-multi-turn-conversations">教程：如何 Fine-tune Llama-3 并在 Ollama 中使用 | Unsloth 文档</a>: 创建自定义个人助手（如 ChatGPT）并在 Ollama 上本地运行的初学者指南
</li>
</ul>

</div>

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1322784435128635474)** (7 条消息): 

> `Light Prompter, Test Time Training, Weights Updating, RL Techniques, VLLM Notebooks` 


- **Light Prompter 加速 Test-Time Compute**：[Light Prompter](https://github.com/Green0-0/light_prompter) GitHub 仓库专注于利用 batching 技术加速 **test-time compute**，并附带了相关的 Notebooks。
   - 该项目旨在提高模型推理过程中的效率，并鼓励社区做出相关贡献。
- **关于 Test Time Training 的询问**：一名成员提出了关于 **test time training** 的问题，特别是它是否涉及在推理过程中更新模型权重。
   - 讨论暗示需要更多的研究，并建议阅读相关论文可能会有所帮助。
- **关于训练中 RL 技术的讨论**：有建议认为 test time training 可能与 **Reinforcement Learning (RL)** 方法论有关。
   - 另一位成员推测应该存在类似的方法，并暗示可能找到可用的代码或研究。



**提到的链接**：<a href="https://github.com/Green0-0/light_prompter">GitHub - Green0-0/light_prompter: Accelerate test-time-compute with batching!</a>：通过 batching 加速 test-time-compute！通过在 GitHub 上创建账户为 Green0-0/light_prompter 的开发做出贡献。

---

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1322294311169036358)** (637 条消息🔥🔥🔥): 

> `Cursor IDE 问题, Deepseek API 使用, Chat vs Composer, Web 应用开发, Cursor 支付方式` 


- **Cursor IDE 变得不那么好用了**：用户反映对 Cursor 感到沮丧，指出 AI 能力和响应速度有所下降，导致一些人考虑寻找替代方案或退回到之前的版本。
   - 许多人遇到了重复的问题或错误，通常需要重启才能解决全天功能退化的问题。
- **在 Cursor 中使用 OpenAI API**：用户询问如何在 Cursor 中利用 OpenAI API，并讨论了不同模型的局限性和使用体验。
   - 一些用户发现 Claude 的效果优于最新的 OpenAI 模型，这表明新模型缺乏改进。
- **使用 Cursor 进行 Web 应用开发**：用户分享了使用 Cursor 开发 Web 应用的经验，强调了对于编程知识有限的人来说易于上手。
   - 一位用户成功发布了一个针对移动端 MMO 游戏的 Web 工具，证明了即使没有深厚的编程背景， Cursor 也能有效地构建应用。
- **Composer 与 Chat 功能对比**：Composer 工具因其迭代和修复代码的能力而受到称赞，而一些用户仍然发现 Chat 功能很有价值。
   - 用户讨论了将 AI 视为助手如何带来更好的结果，甚至建议对 Cursor 表现出沮丧或表达愤怒可能会促使其给出更好的回复。
- **Cursor 的支付挑战**：用户在使用各种支付方式支付 Cursor 费用时遇到困难，通常提到本地化和银行限制问题。
   - 支付处理方面的挑战导致一些人寻求替代方法，这表明 Cursor 平台迫切需要更便捷的交易方式。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/sonic-sonic-the-hedgehog-tails-knuckles-freaky-gif-12830799583270855342">Sonic Sonic The Hedgehog GIF - Sonic Sonic the hedgehog Tails - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.hiro.so/blog/write-better-smart-contracts-with-the-programming-language-clarity">使用 Clarity 编程语言编写更好的智能合约</a>: 了解为什么 Clarity 编程语言如此适合编写智能合约。</li><li><a href="https://x.com/code/status/1872673862992744625">来自 Visual Studio Code (@code) 的推文</a>: Claude 3.5 Sonnet 直接集成在 @code 中。今天起对所有 GitHub Copilot 免费版用户开放。了解更多: http://aka.ms/copilot-free</li><li><a href="https://forum.cursor.com/t/how-to-do-fix-in-composer-and-fix-in-chat-actions-from-keyboard/31221">如何通过键盘执行 `Fix in Composer` 和 `Fix in Chat` 操作</a>: 这两个操作：我在设置中没找到。</li><li><a href="https://forum.cursor.com/t/are-claude-3-5-sonnet-and-claude-3-5-sonnet-20241022-different/24272/3">claude-3.5-sonnet 和 claude-3.5-sonnet-20241022 有区别吗？</a>: 快速更新：claude-3.5-sonnet 现在指向 claude-3-5-sonnet-20241022！</li><li><a href="https://guitar-tab.vercel.app/en/tools/fretboard">吉他和弦学习应用</a>: 通过我们的交互式学习工具掌握吉他和弦！</li><li><a href="https://guitar-tab.vercel.app/en/tools/scales">吉他和弦学习应用</a>: 通过我们的交互式学习工具掌握吉他和弦！</li><li><a href="https://guitar-tab.vercel.app/en/tools/tuner">吉他和弦学习应用</a>: 通过我们的交互式学习工具掌握吉他和弦！</li><li><a href="https://coffeethencode.dev/dectalk/">DECTalk 生成器</a>: 未找到描述</li><li><a href="https://twomgg.onrender.com/">TwomGG</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Stackblitz (Bolt.new) ▷ #[announcements](https://discord.com/channels/364486390102097930/671536649301131325/1323331159878275114)** (1 条消息): 

> `Grok AI API 促销` 


- **Grok AI 促销最后机会**：这是 Grok AI 为 API 用户提供 25 美元免费额度促销的最后两天，随着年底临近，截止日期即将到来。在额度消失前[在此查看促销详情](https://x.com/stackblitz/status/1873769044761022633)！
- **尝试 Grok AI 的机会**：成员们强调，今天和明天是将 Grok AI API 集成到你的 Bolt 应用中的绝佳时机。



**提到的链接**: <a href="https://x.com/stackblitz/status/1873769044761022633">来自 StackBlitz (@stackblitz) 的推文</a>: 在你的 Bolt 应用中构建 #GrokAI！如果你还没试过，今天和明天就是最佳时机：在今年结束前，每位 x.ai API 用户仍可获得 25 美元的免费额度！

  

---

### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1322363289677860976)** (20 条消息🔥): 

> `Bolt 代码更新问题，语音 Prompting 功能请求，Token 浪费担忧` 


- **Bolt 代码更新经常失败**：许多用户报告称，尽管 Bolt 生成了代码，但停止对网站进行可见的更改，导致进行中的项目进度受阻。
   - 一些人建议回滚 Checkpoints 或使用 Visualizer 来帮助解决问题，尽管这只是暂时有效。
- **语音 Prompting 功能请求**：用户对添加类似 ChatGPT 的语音 Prompting 功能表现出浓厚兴趣，以便在构建项目时更轻松地进行交流。
   - 然而，用户被提醒，由于音频模型相比聊天模型的复杂性，实现此类功能的成本可能很高。
- **对 Token 浪费的沮丧**：几位用户对 Bolt 中与 Prompting 问题相关的高额 Token 成本表示担忧，特别是当 Prompt 未能产生预期结果时。
   - 用户请求增加允许前置指令（Prefixed Instructions）的功能，以减少重复 Prompt 并节省 Token。
- **在 Bolt 中观察到 LLM 懒惰现象**：一位用户指出，包括 Bolt 在内的 LLM 在处理大型代码库时往往响应度降低，导致非相关文件出现意外更改。
   - 建议包括重新加载项目并启用 Diff 模式，据报道这有助于缓解“懒惰”问题。


  

---

### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1322300689090609346)** (460 条消息🔥🔥🔥): 

> `Token 消耗, Bolt 中的错误处理, 使用 Bolt 进行应用开发, Firebase vs Supabase, Bolt 中的项目管理` 


- **对 Token 消耗的担忧**：用户对 Bolt 中快速的 Token 消耗表示沮丧，估算值因项目规模和用户的 Prompt 技巧而异。
   - 许多人建议，随着项目规模的扩大，AI 的能力会下降，需要更精确的 Prompt 以避免不必要的 Token 使用。
- **错误处理与调试**：多位用户报告称，遇到了由迁移问题和 Bolt 的代码修改引起的错误，导致 Token 成本增加。
   - 一些人建议使用 Google Gemini 等外部工具进行错误解释和修订，而另一些人则警告 Bolt 当前反馈机制的局限性。
- **与外部工具集成**：用户正在探索将 Bolt 与 StackBlitz 结合使用的工作流，强调了导出项目进行手动调整的重要性。
   - 有关于集成 Convex 作为 Supabase 替代方案可行性的讨论，但由于其处于 Beta 阶段，建议谨慎对待。
- **用户体验与改进**：几位用户分享了关于如何更好地利用 Bolt，以及在项目开发中提高 AI 理解能力和响应速度的经验。
   - 有人建议实现诸如聊天时间戳、更好的项目和 Fork 命名规范等功能，以提升用户体验。
- **社区支持与资源**：社区成员讨论了 StackBlitz 缺乏直接支持的问题，并强调了利用社区渠道寻求帮助的重要性。
   - 用户发起的改进建议（如关于工具能力的更清晰指南）凸显了对非开发人员提供更直观指令的需求。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://app.netlify.com/sites/appname/overview">Netlify</a>: 未找到描述</li><li><a href="https://docs.netlify.com/domains-https/custom-domains/multiple-domains/#domain-aliases">多域名站点</a>: 通过主域名、域名别名、域名级重定向、自动部署子域名或分支子域名来管理站点的多个域名。</li><li><a href="https://support.bolt.new/welcome">Notion – 笔记、任务、维基和数据库的全能工作空间。</a>: 一款将日常工作应用融合在一起的新工具。它是为您和您的团队打造的全能工作空间。</li><li><a href="https://tenor.com/view/simpsons-homer-bart-lisa-join-us-gif-17846376318791889140">Simpsons Homer GIF - Simpsons Homer Bart - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/shake-my-head-mike-mclusky-mayor-of-kingstown-smh-disappointed-gif-293488442475603142">Shake My Head Mike Mclusky GIF - Shake my head Mike mclusky Mayor of kingstown - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://bolt.diy">GitHub - stackblitz-labs/bolt.diy: 使用任何你想要的 LLM 来提示、运行、编辑和部署全栈 Web 应用程序！</a>: 使用任何你想要的 LLM 来提示、运行、编辑和部署全栈 Web 应用程序！ - stackblitz-labs/bolt.diy</li><li><a href="https://github.com/stackblitz/bolt.new/issues/4455">Supabase 问题 · Issue #4455 · stackblitz/bolt.new</a>: 描述 Bug：我的项目一直运行良好，直到 Bolt.new 强制我创建一个新数据库。我不想继续消耗 Token 来修复这个问题。第一次修复时我消耗了 300 万个 Token。...</li><li><a href="https://www.youtube.com/watch?v=1GfqnOAKr9M"> - YouTube</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=eE6m0MmLpDU"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1322296416730611873)** (380 条消息🔥🔥): 

> `DeepSeek V3 性能, Aider 使用与上下文管理, Gemini 模型见解, OpenRouter 集成问题, Web 应用中的 OCR 实现`

- **DeepSeek V3 给用户留下深刻印象**：许多用户正在转向 **DeepSeek V3**，注意到其高效性以及有效处理大型编程任务的能力，有时完成项目的速度比以前快得多。
   - 与 **Gemini** 等其他模型的对比显示，DeepSeek 因其在编程辅助方面的强劲性能而目前更受青睐。
- **使用 Aider 管理上下文和限制**：用户正在学习如何优化其 **Aider** 配置以处理大型项目，包括设置 `.aider.model.metadata.json` 来管理上下文限制和成本。
   - 尽管有关于上下文限制的警告，许多用户报告称在管理庞大代码库时拥有成功的体验，且性能表现合理。
- **关于 Gemini 模型的见解**：关于 **Gemini 2.0** 模型的讨论强调了它们在编程任务中的优势，特别是在免费版本中，用户在工作流中有效地利用了这些模型。
   - 用户建议使用 **Gemini** 加载大型代码库，同时依靠其他模型进行代码生成。
- **OpenRouter 集成挑战**：一些用户在尝试将 **Aider** 与 OpenRouter 的 **DeepSeek** 集成时遇到问题，通常由于配置失误而面临 model not found 错误。
   - 建议用户启用特定设置以确保正确的 endpoint 访问和功能。
- **对无代码开发的兴奋**：一位用户表达了对使用 **Aider** 在短短一小时内快速构建 Web App 的兴奋，展示了显著提高生产力的潜力。
   - 该用户强调了通过 **TesseractJS** 实现 OCR 等功能，指出了自动化解决方案在无需手动编写代码的情况下进行编程的能力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat">首页</a>：aider 是你终端里的 AI 结对编程工具</li><li><a href="https://blog.exolabs.net/day-2/">EXO 的 12 天</a>：真正的开源创新 12 天</li><li><a href="https://aider.chat/docs/repomap.html">仓库地图</a>：aider 使用你的 Git 仓库地图为 LLM 提供代码上下文。</li><li><a href="https://aider.chat/docs/llms/warnings.html">模型警告</a>：aider 是你终端里的 AI 结对编程工具</li><li><a href="https://tenor.com/view/star-trek-patrick-stewart-captain-jean-luc-picard-face-palm-disappointed-gif-4780258">星际迷航 Patrick Stewart GIF - 星际迷航 Patrick Stewart 舰长 Jean Luc Picard - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://aider.chat/docs/llms/deepseek.html">DeepSeek</a>：aider 是你终端里的 AI 结对编程工具</li><li><a href="https://x.com/kimmonismus/status/1873093574507872300">来自 Chubby♨️ (@kimmonismus) 的推文</a>：开源正在兴起</li><li><a href="https://x.com/alexocheema/status/1872447153366569110">来自 Alex Cheema - e/acc (@alexocheema) 的推文</a>：不得不堆叠 8 台 Mac Mini 来运行它。目前约为 5 tok/sec。第一次在 8 台 Mac Mini 上运行推理——性能还有很大提升空间（该配置的理论极限 > 10 tok/sec）...</li><li><a href="https://openrouter.ai/deepseek/deepseek-chat">DeepSeek V3 - API, 提供商, 统计数据</a>：DeepSeek-V3 是 DeepSeek 团队的最新模型，基于前代版本的指令遵循和编程能力构建。在近 15 万亿 token 上进行了预训练，据报道...</li><li><a href="https://aider.chat/docs/config/options.html#repomap-settings">选项参考</a>：关于 aider 所有设置的详细信息。</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html">高级模型设置</a>：为 LLM 配置高级设置。</li><li><a href="https://github.com/Aider-AI/aider-swe-bench">GitHub - Aider-AI/aider-swe-bench：用于针对 SWE Bench 基准测试 aider 的测试框架</a>：用于针对 SWE Bench 基准测试 aider 的测试框架 - Aider-AI/aider-swe-bench</li><li><a href="https://aider.chat/docs/usage/conventions.html">指定编码规范</a>：让 aider 在处理你的代码时遵循你的编码规范。</li><li><a href="https://github.com/Aider-AI/conventions">GitHub - Aider-AI/conventions：供 aider 使用的社区贡献规范文件</a>：供 aider 使用的社区贡献规范文件 - Aider-AI/conventions</li><li><a href="https://github.com/Aider-AI/aider/issues/2727">FastAPI 集成 · Issue #2727 · Aider-AI/aider</a>：Issue AAAA - Aider 作为 API 概览。我开发了一个 FastAPI 服务器，提供对 aider 功能的 REST API 访问。目前它作为一个独立应用运行，但可以从中受益...</li><li><a href="https://api-docs.deepseek.com/news/news0802#how-to-use-deepseek-apis-caching-service">DeepSeek API 推出磁盘上下文缓存（Context Caching），价格降低一个数量级 | DeepSeek API 文档</a>：在大语言模型 API 使用中，很大一部分用户输入往往是重复的。例如，用户提示词经常包含重复的引用，而在多轮对话中，之前的...</li><li><a href="https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus">BFloat16：Cloud TPU 高性能的秘诀 | Google Cloud 博客</a>：Google Cloud TPU 的高性能是如何由 Brain Floating Point 格式（即 bfloat16）驱动的。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1322318992249065564)** (76 条消息🔥🔥): 

> `DeepSeek V3 usage, Aider installation and configuration, Token limits with models, Git sparse-checkout compatibility, Shell command execution in Aider` 


- **DeepSeek V3 的关注点与对比**：用户讨论了通过 Hugging Face 使用 DeepSeek V3 与使用官方托管版本之间的权衡，指出了价格差异和上下文窗口大小的不同。Hugging Face 提供 **128k context**，而 DeepSeek 官方为 **64k**。
   - 用户对使用 DeepSeek 托管版本时的输入**隐私和用途**表示担忧，这引发了关于更高价格是否能换取更高安全性的讨论。
- **Aider 安装最佳实践**：用户指出 Aider 应该通过 `aider-install` 等选项进行全局安装，并在各种环境中实现无缝集成，强调了使用正确的 Python 版本进行设置的重要性。
   - 提供了针对不同操作系统的具体安装步骤，强调了对各种包管理器和环境的注意事项，特别是针对 Arch Linux 用户。
- **管理 Token 限制与用户命令**：提到了 Aider 报告 Token 限制的能力，用户注意到通过调整操作来避免超出这些限制，同时努力保持高效的编码工作流。
   - 用户对 Aider 不能直接执行 Shell 命令表示沮丧，因为他们希望简化工作流，并建议通过潜在的更新来允许更广泛的审批设置。
- **Git Sparse-Checkout 兼容性**：出现了关于 Aider 与 Git sparse-checkout 兼容性的讨论，用户建议不要使用它，因为据报道 Aider 在处理 index-version 3 的 Git 仓库时存在问题。
   - 建议使用 `--no-git` 选项等变通方法，以便在不受 Git 限制的情况下启用 Aider 功能。
- **Aider 中的命令审批**：用户质疑 Aider 中命令审批的必要性，指出虽然它增强了安全性，但可能会阻碍工作流效率，特别是对于高级用户。
   - 提出了引入环境变量来覆盖 Shell 命令审批的想法，以便根据特定用户需求定制 Aider 的运行方式。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/troubleshooting/token-limits.html">Token limits</a>：aider 是你终端里的 AI 配对编程工具</li><li><a href="https://www.chinatalk.media/p/deepseek-ceo-interview-with-chinas">Deepseek: The Quiet Giant Leading China’s AI Race</a>：其 CEO 深度访谈的带注释翻译版本</li><li><a href="https://docs.litellm.ai/docs/providers">Providers | liteLLM</a>：了解如何在 LiteLLM 上部署和调用来自不同提供商的模型</li><li><a href="https://www.codeguide.dev/">CodeGuide</a>：CodeGuide 为你的 AI 编码项目创建详细文档。</li><li><a href="https://aider.chat/docs/install.html">Installation</a>：如何安装并开始使用 aider 进行配对编程。</li><li><a href="https://stackoverflow.com/questions/10418975/how-to-change-line-ending-settings">How to change line-ending settings</a>：是否有文件或菜单可以让我更改处理行尾的设置？我读到有 3 个选项：Checkout Windows-style, commit Unix-style...</li><li><a href="https://api-docs.deepseek.com/quick_start/pricing">Models &amp; Pricing | DeepSeek API Docs</a>：下面列出的价格单位为每 1M tokens。Token 是模型识别的最小文本单位...</li><li><a href="https://github.com/Aider-AI/aider/issues/211">Aider uses GitPython, which doesn&#39;t work with index-version 3 git repos · Issue #211 · Aider-AI/aider</a>：在包含 R、Python 和 Bash 脚本的现有仓库中执行 aider 时，我遇到了这个错误...</li>
</ul>

</div>

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1322334121212772455)** (31 条消息🔥): 

> `HF 模型中的 Logit 相等性，LLM 中的动态测试时温度，BF16 训练与梯度缩放，Lipschitz-1 RMSNorm 替代方案` 


- **浮点精度下的 Logits 相等**：一位成员报告了在 HF 模型中遇到的一个问题，即在推理过程中，尽管两个 Token 不是概率最高的，但它们的 Logits 在 FP16 或 BF16 精度下*完全相等*。
   - 这种差异引发了对模型行为的质疑，因为它在评估中出现的频率高达 **20%**。
- **LLM 中的动态温度至关重要**：LLM 架构的一个突破涉及一种策略，即通过调节**动态测试时温度 (dynamic test-time temperature)** 来增强创造力和解决问题的能力。
   - 该提案包含一个数学结构，表达了激活空间中受温度控制的转换如何创造出具有创造性的轨迹。
- **BF16 训练与梯度缩放疑问**：关于 BF16 训练期间是否需要梯度缩放 (gradient scaling) 的讨论显示，动态缩放可能会影响性能，而静态缩放则不太令人担忧。
   - 一位成员强调，它可能不会显著加速训练延迟，尤其是在处理较小模型时。
- **损失函数中的精度与 Logits**：建议成员在应用损失函数之前先以 FP32 计算 Logits，以便在 BF16 训练期间获得更好的性能和准确性。
   - 这种方法确保了关键的交叉熵计算不会因使用较低精度而受到不利影响。
- **PyTorch 中的 RMSNorm 替代方案**：分享了一个 Lipschitz-1 RMSNorm 替代方案的拟议实现，演示了如何根据均方根值对输入进行归一化。
   - 该函数利用 **tanh** 激活进行缩放，并以清晰的 PyTorch 代码片段呈现。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1322303398569050186)** (219 条消息🔥🔥): 

> `LLM 基准测试挑战，神经网络的梯度路由，用于几何定理发现的 TongGeometry，用于特征分析的 Crosscoders，浅层对齐假设` 


- **LLM 基准测试揭示缺陷**：讨论强调了准确评估 LLM 性能的挑战，强调了当前的基准测试通常存在问题模棱两可以及对评估方法敏感的情况。
   - 参与者建议转向更具功能性的基准测试，衡量在复杂的、开放式任务中的表现，而不是简单的多选题格式。
- **梯度路由增强模型可解释性**：梯度路由 (Gradient routing) 被提议作为一种提高神经网络可解释性的方法，通过在反向传播期间应用数据依赖的掩码，将能力隔离在特定的子区域内。
   - 这种方法可能通过允许对模型的哪些部分从特定数据点学习进行可调控制，从而解决诸如将矩阵条目映射到特定神经元等问题。
- **TongGeometry 与几何定理发现**：关于 TongGeometry 的论文介绍了一个用于提出和解决几何问题的系统，在计算受限的情况下实现了几何定理的重大发现。
   - 尽管缺乏方法论细节，但论文指出 TongGeometry 的一些提案已被地区数学奥林匹克竞赛采纳。
- **探索用于理解模型的 Crosscoders**：Crosscoders 是一种备受关注的新方法，旨在跟踪和解析神经网络多个层级中的特征，显示出更好地理解模型行为的潜力。
   - Crosscoders 的应用可以改进跨层特征分析的方式，突显了它们在电路简化和模型差异定位中的用途。
- **SFT 中的浅层对齐假设 (Superficial Alignment Hypothesis)**：URIAL 提供的证据支持浅层对齐假设，表明 Token 分布的微小修改会导致基础 LLM 与其对齐版本之间的性能指标相似。
   - 这表明对齐微调可能不会从根本上改变模型能力，而是强调了风格化的 Token 变化。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

</div>

<ul>
<li>
<a href="https://arxiv.org/abs/2412.17758">In Case You Missed It: ARC &#39;Challenge&#39; Is Not That Challenging</a>: 对于现代 LLM 而言，ARC Challenge 似乎比 ARC Easy 更难，这主要是由于评估设置阻止了答案选项的直接比较，而非其固有的复杂性。虽然一些...</li><li><a href="https://arxiv.org/abs/2412.11834">Wonderful Matrices: Combining for a More Efficient and Effective Foundation Model Architecture</a>: 为了使 Foundation Model 更加高效和有效，我们的想法是将序列变换和状态变换相结合。首先，我们证明了 Rotary Position Embedding 的可用性...</li><li><a href="https://arxiv.org/abs/2410.04332">Gradient Routing: Masking Gradients to Localize Computation in Neural Networks</a>: Neural Networks 的训练主要基于其输入和输出，而不考虑其内部机制。这些被忽视的机制决定了对安全至关重要的属性，例如...</li><li><a href="https://en.m.wikipedia.org/wiki/Where_Mathematics_Comes_From">Where Mathematics Comes From - Wikipedia</a>: 未找到描述</li><li><a href="https://phyworld.github.io/">How Far is Video Generation from World Model: A Physical Law Perspective</a>: 我们进行了一项系统研究，旨在调查视频生成是否能够通过利用数据和模型缩放从视频中学习物理定律。</li><li><a href="https://en.m.wikipedia.org/wiki/Chunking_(psychology)">Chunking (psychology) - Wikipedia</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2412.10673">Proposing and solving olympiad geometry with guided tree search</a>: 数学奥林匹克是享有盛誉的竞赛，问题的提出和解决都备受推崇。构建能够提出并解决奥数问题的 AI 是一项尚未解决的挑战...</li><li><a href="https://github.com/Re-Align/urial">GitHub - Re-Align/URIAL</a>: 通过在 GitHub 上创建账户为 Re-Align/URIAL 的开发做出贡献。</li><li><a href="https://arxiv.org/abs/2312.01552">The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning</a>: LLM 的 Alignment Tuning 过程通常涉及通过 Supervised Fine-Tuning (SFT) 进行指令学习，以及通过 Reinforcement Learning from Human Feedback 进行偏好微调...</li><li><a href="https://www.lesswrong.com/posts/srt6JXsRMtmqAJavD/open-source-replication-of-anthropic-s-crosscoder-paper-for">Open Source Replication of Anthropic’s Crosscoder paper for model-diffing — LessWrong</a>: 简介：Anthropic 最近发布了一篇关于 Crosscoders 的精彩短论文 (Lindsey 等人)。在这篇文章中，我们开源了一个用于 Model-Diffing 的 Crosscoder 训练...</li><li><a href="https://transformer-circuits.pub/2024/crosscoders/index.html">Sparse Crosscoders for Cross-Layer Features and Model Diffing</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1322651431957954683)** (9 条消息🔥): 

> `Neural Networks as Polycomputers, TinyStories Dataset, Small Transformers, Catastrophic Interference Solutions` 


- **神经网络展现出 Polycomputing 特性**：讨论集中在 **neural networks** 可以被视为 **polycomputers** 的观点上，即同时对不同特征执行多种计算。
   - *Polycomputing* 可能为缓解 **catastrophic interference**（灾难性干扰）等挑战提供见解，使 Agent 能够在不丢失先前习得知识的情况下学习新行为。
- **TinyStories：用于小型 Transformer 的数据集**：**TinyStories** 数据集包含由 GPT-3.5 和 GPT-4 生成的**合成短篇故事**，旨在训练参数量少于 1000 万的小型语言模型。
   - 成员们讨论了训练具有更简单架构的模型的影响，如 [TinyStories 论文](https://arxiv.org/abs/2305.07759)中所述。
- **寻找开源的小型 Transformer**：一位成员请求推荐**开源的小型 Transformer**，理想情况下是在复杂任务上预训练的 1 到 5 层模型。
   - 回复中强调了 [TinyStories](https://arxiv.org/abs/2305.07759) 等示例，表明了对开发轻量级模型的持续关注。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2212.10675">There&#39;s Plenty of Room Right Here: Biological Systems as Evolved, Overloaded, Multi-scale Machines</a>：计算模型对生物世界的适用性是一个活跃的辩论话题。我们认为，一个有用的前进方向是放弃类别之间的硬边界，并采用...</li><li><a href="https://x.com/norabelrose/status/1873090825351250094">Nora Belrose (@norabelrose) 的推文</a>：神经网络是 @drmichaellevin 意义上的 polycomputers。根据你的视角，你可以将其解释为在不同类型的特征上执行许多不同的计算。没有...</li><li><a href="https://arxiv.org/abs/2305.07759">TinyStories: How Small Can Language Models Be and Still Speak Coherent English?</a>：语言模型 (LM) 是自然语言处理的强大工具，但当它们规模较小时，往往难以生成连贯流畅的文本。拥有约 125M 参数的模型（如 GP...）</li><li><a href="https://huggingface.co/datasets/roneneldan/TinyStories">roneneldan/TinyStories · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1323056848210886676)** (12 messages🔥): 

> `Scrolls benchmark issues, GSM8K strict exact match clarification, mgsm_chat troubleshooting, ZeroSCROLLS vs SCROLLS evaluation, lm_eval command usage` 


- **Scrolls Benchmark Bug 报告**：一位用户报告了运行 **scrolls benchmark** 时的问题，指出 **load_metric** 似乎已被弃用，必须替换为 **evaluate**。
   - 此外，还有关于 **Instance** 无法识别 **apply_chat_template** 参数的疑虑。
- **澄清 GSM8K 指标**：有询问关于 GSM8K 中的 **strict exact match** 指标是否与旧版排行榜中使用的 'acc' 指标一致。
   - 一位成员指出，各版本之间的答案提取过程似乎是一致的，并引用了特定的 [GitHub 链接](https://github.com/EleutherAI/lm-evaluation-harness/blob/b281b0921b636bc36ad05c0b0b0763bd6dd43463/lm_eval/tasks/gsm8k.py#L36)。
- **调试 mgsm_chat 模型**：一位用户提到在复现 **mgsm_chat** 模型的性能指标时遇到困难，表示虽然没有报错，但结果无法复现。
   - 另一位成员对该模型的功能表示肯定，并询问了所遇具体错误的细节。
- **关于 SCROLLS 与 ZeroSCROLLS 评估的讨论**：一位用户质疑为什么尽管 dev set 较小，但仍对预训练模型使用 **SCROLLS** 进行评估，而对训练后（post-trained）模型使用 **ZeroSCROLLS**。
   - 该询问保留了在必要时将问题转至其他合适频道的可能性。
- **lm_eval 命令与性能结果**：一位用户分享了他们运行模型的 **lm_eval** 命令，并指定了 exact match 评估的性能指标。
   - 报告的结果显示 **flexible-extract** 为 **0.1098**，**strict-match** 为 **0.0771**，并对之前的帮助表示感谢。



**提及的链接**：<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/b281b0921b636bc36ad05c0b0b0763bd6dd43463/lm_eval/tasks/gsm8k.py#L36)">lm-evaluation-harness/lm_eval/tasks/gsm8k.py at b281b0921b636bc36ad05c0b0b0763bd6dd43463 · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。- EleutherAI/lm-evaluation-harness

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1322297747063308411)** (249 条消息🔥🔥): 

> `DeepSeek V3 性能问题, OpenRouter 模型集成, 翻译模型推荐, 构建多模态 Agent, LLM 定价与功能对比` 


- **DeepSeek V3 性能问题**：用户报告称，与官方 API 相比，DeepSeek V3 在 OpenRouter 上的表现明显较差，推测可能与 Together API 有关。
   - 回复指出，性能的变化或降级会导致用户投诉，一些人认为这预示着新版本的发布。
- **OpenRouter 模型集成**：将新模型集成到 OpenRouter 需要有足够兴趣的提供商，用户可以与成熟的 AI 实验室合作，或者自己成为提供商。
   - 如果营销和开发得当，重视编码等利基 LLM 能力可以使模型获得有利的市场地位。
- **翻译模型推荐**：讨论强调 GPT-4o mini 是翻译的首选，而 Gemini 1.5 Flash 被指出经常出错。
   - 用户建议使用特定的 System Prompts 来增强翻译任务的性能，并强调了结构的重要性。
- **构建多模态 Agent**：虽然让模型输出 JSON 可以简化 Agent 的操作，但对于有效运行 Agent 来说并非绝对必要。
   - 用户讨论了对多模态 Agent 框架的兴趣，并提到了 Google 的 Project Mariner 作为一个有趣的案例。
- **LLM 定价与功能对比**：关于 LLM 定价的讨论显示，OpenRouter 缺乏缓存输入 Token 的折扣，并区分了各种定价策略。
   - 虽然一些用户对模型性能的感知降级表示担忧，但其他人强调需要关于模型能力的清晰沟通和证据。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/chat">Chatroom | OpenRouter</a>：LLM 聊天室是一个多模型聊天界面。添加模型并开始聊天！聊天室将数据本地存储在您的浏览器中。</li><li><a href="https://www.anthropic.com/research/building-effective-agents">Building effective agents</a>：一篇为开发者提供构建高效 AI Agent 的建议和工作流的文章。</li><li><a href="https://openrouter.ai/docs/prompt-caching#deepseek">Prompt Caching | OpenRouter</a>：优化 LLM 成本高达 90%</li><li><a href="https://openrouter.ai/deepseek/deepseek-chat">DeepSeek V3 - API, Providers, Stats</a>：DeepSeek-V3 是 DeepSeek 团队的最新模型，建立在先前版本的指令遵循和编码能力之上。在近 15 万亿 Token 上进行了预训练，报告的评估...</li><li><a href="https://openrouter.ai/rankings/translation?view=week">LLM Rankings: translation | OpenRouter</a>：根据翻译 Prompt 的使用情况对语言模型进行排名和分析</li><li><a href="https://api-docs.deepseek.com/api/create-completion">Create FIM Completion (Beta) | DeepSeek API Docs</a>：FIM (Fill-In-the-Middle) 补全 API。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/gGsmJeGdDi">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://fireworks.ai/blog/document-inlining-launch)">Fireworks - Fastest Inference for Generative AI</a>：使用最先进的开源 LLM 和图像模型，享受极速体验，或者使用 Fireworks AI 免费微调并部署您自己的模型！
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1322346295418949712)** (74 条消息🔥🔥): 

> `DeepSeek V3 性能表现, 本地 AI 与 API 使用对比, Hunyuan 视频模型局限性, SmallThinker 模型概览, LLM 开发机遇` 


- **DeepSeek V3 在处理复杂任务方面表现出色**：DeepSeek V3 成功通过了 MTG 指挥官套牌构建测试，并能构建正确的 Scryfall 查询，展示了其有效处理复杂任务的能力。
   - 成员们注意到，DeepSeek 似乎能在长 context 中保持性能，这使其在开源模型中脱颖而出。
- **关于本地 AI 与 OpenAI API 的辩论**：用户讨论了运行 Aquila 的 [Ollama](https://ollama.com) 以及 LlamaCPP 在学习和本地配置方面的优势，强调了系统自定义的重要性。
   - 配置 OpenAI API 被认为在 Agent 任务中具有优势，能显著改进工作流。
- **Hunyuan 视频模型的局限性**：虽然 Hunyuan 可以在有限的硬件上运行，但据反馈其运行缓慢，且在低分辨率和少帧数的情况下难以获得理想效果。
   - 还有一篇博客文章确认该模型可以在仅有 **8GB VRAM** 的 GPU 上运行，尽管速度可能是一个问题。
- **SmallThinker 模型介绍**：新款 **SmallThinker-3B-preview** 模型已发布，在推理能力方面有所提升，基准测试表现显著。
   - 然而，它在任务执行过程中难以判断何时停止，这引发了用户的一些调侃。
- **呼吁 LlamaCPP 开发者**：社区表示迫切需要更多 LlamaCPP 开发者，考虑到它是许多其他项目的基石。
   - 鉴于其在推动开源 AI 模型方面的核心作用，建议有编程经验的人积极贡献代码。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/PowerInfer/SmallThinker-3B-Preview">PowerInfer/SmallThinker-3B-Preview · Hugging Face</a>: 未找到描述</li><li><a href="https://blog.comfy.org/p/running-hunyuan-with-8gb-vram-and?r=4z50rt&utm_campaign=post&utm_medium=web&triedRedirect=true">在 8GB VRAM 上运行 Hunyuan 并支持 PixArt 模型</a>: 来自 ComfyUI 的最新模型支持更新和答疑时间新闻！</li><li><a href="https://www.videoleapapp.com/create/instagram-video-editor">Instagram 视频编辑器与制作工具：创建 Instagram 视频 | Videoleap</a>: 立即开始 7 天免费试用并体验！使用 Videoleap 应用轻松制作和编辑 Instagram 视频。为您的 Instagram 视频添加音乐等。</li><li><a href="https://huggingface.co/datasets/PowerInfer/QWQ-LONGCOT-500K">PowerInfer/QWQ-LONGCOT-500K · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://youtu.be/jwChiek_aRY?si=uTnNyyUUX8IJXhie"> - YouTube</a>: 未找到描述</li><li><a href="https://www.minimaxi.com/en/price">MiniMax - 让智能触手可及</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1322293521742434305)** (149 条消息🔥🔥): 

> `DeepSeek V3 性能问题，LLaMaCPP 中的异常行为，Anthropic 的推理模型，理解 LLaMa 3.3，高带宽与家庭用户解决方案` 


- **DeepSeek V3 在推理方面表现挣扎**：成员们注意到 **DeepSeek V3** 会陷入**推理循环**，在评估中表现出奇怪的行为，包括无限输出以及在“死的薛定谔的猫”等推理测试中失败。
   - 尽管它在编程任务中表现出色，但人们在思考它在不同基准测试中的表现差异，并对其架构提出了疑问。
- **LLaMaCPP RPC 中间件讨论**：一位用户讨论了在 **LLaMaCPP RPC** 中实现填充（padding）机制，建议这可以有效管理 Tensor 大小，同时防止处理过程中的数据损坏。
   - 尽管该方法具有潜在的效率优势，但人们担心这是否会导致代码过于复杂且不够规范（hacky）。
- **Anthropic 对模型和推理的方法**：有人猜测 **Anthropic** 可能拥有内部推理模型，并认为他们可能正在使用这些模型来优化 **Claude**，而不是公开分发。
   - 成员们对 Anthropic 拥有深厚背景和资源却面临**算力问题（compute issues）**表示好奇。
- **LLaMa 3.3 用户体验**：一位成员分享了对 **LLaMa 3.3** 70B 在编程和文档理解方面表现的正面印象，发现它在某些任务上优于其他替代方案。
   - 这些见解与另一些指出其在某些基准测试下表现不稳的观点形成对比，表明用户体验存在差异。
- **高带宽解决方案与家庭用户之间的平衡**：随后展开了关于量化和网络开销的**中间件**讨论，强调了针对消费级硬件（而非数据中心）的高效解决方案的市场需求。
   - 成员们强调，对于想要在不依赖高带宽设置的情况下实现 **LLaMaCPP** 等先进模型的家庭用户来说，目前缺乏可用资源。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/aidan_mclau/status/1872444303974543859">Aidan McLau (@aidan_mclau) 的推文</a>: aidanbench 的两项更新：&gt; gemini-2.0-flash-thinking 目前排名第 2（评分变化原因见下文）&gt; deepseek v3 排名第 22（想法见下文）</li><li><a href="https://github.com/cpldcpu/MisguidedAttention">GitHub - cpldcpu/MisguidedAttention: A collection of prompts to challenge the reasoning abilities of large language models in presence of misguiding information</a>: 一组旨在挑战大型语言模型在存在误导信息时推理能力的提示词 - cpldcpu/MisguidedAttention</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/9285">Bug: GGML_ASSERT fail in ggml-rpc.cpp ggml_backend_rpc_buffer_init_tensor · Issue #9285 · ggerganov/llama.cpp</a>: 发生了什么？我正尝试在 2 台 3060*2 的机器上通过 ggml-rpc 功能运行 qwen2-72b-instruct-q3_k_m.gguf。机器 1：192.168.136.200 运行 llama-cli；机器 2：192.168.136.201 运行 rpc-server ./llama...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/9799">Bug: rpc server occasionally stops by assert function (deserialize_tensor)  · Issue #9799 · ggerganov/llama.cpp</a>: 发生了什么？描述：在 llama.cpp 中使用 RPC 后端时，我在 rpc_server::deserialize_tensor 函数中遇到了崩溃。断言失败是因为在 d... 之后 tensor-&gt;ne[i] 可能为零。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1322582086795788391)** (6 条消息): 

> `Sklearn 结果报告，二分类指标，测试集评估，模型性能信任度，AUC/ROC 分数` 


- **关于 Sklearn 结果格式的询问**：一位成员询问论文中通常是否会报告二分类的 **sklearn 结果格式**，并引用了 Precision（精确率）、Recall（召回率）和 F1-score 指标。
   - 他们展示了一个表格格式，其中包含两个类别的指标以及准确率（Accuracy）和平均值。
- **关于指标可信度的讨论**：另一位成员指出确保**指标可信度**的重要性，强调评估子集必须与训练集分离，并且能够代表真实世界的分布。
   - “信任指标”还包括考虑分类模型的目标，即优先考虑 Precision 还是 Recall。
- **添加 AUC/ROC 以提高清晰度**：同一位成员建议，添加不同分类阈值下的 **AUC/ROC 分数**可以更深入地了解模型的性能。
   - 这突显了在评估分类任务时，性能指标清晰度的必要性。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1322582086795788391)** (6 条消息): 

> `报告 sklearn 结果，分类指标的可信度，二分类指标` 


- **正确报告 Sklearn 结果**：一位成员询问以类别精确度（class-precision）格式报告的 sklearn 结果是否符合典型的论文标准。
   - 示例中包含了 **Precision**、**Recall** 和 **F1-score** 等指标，以及 **Support** 值。
- **信任分类指标**：另一位成员强调了信任所使用的评估指标的重要性，并询问测试集是否具有代表性，以及是否已从训练集中去污染（decontaminated）。
   - 他们建议理解模型的目标至关重要，强调了权衡 **precision vs recall** 的必要性，并建议包含 **AUC/ROC scores**。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1322397353935634442)** (203 条消息🔥🔥): 

> `Perplexity Pro 订阅，Deepseek v3 可用性，推理模式功能，赠款提案协助，Pro 推理和搜索增强` 


- **关于 Perplexity Pro 访问权限的困惑**：用户对 Perplexity Pro 订阅和频道访问权限表示困惑，指出存在链接过期和缺乏学生折扣的问题。
   - 许多人正在寻求关于如何有效使用该服务并访问 Pro 功能的说明，这表明平台需要更好的沟通。
- **Deepseek v3 在 Pro 模式下不可用**：讨论集中在 Pro 订阅中缺失 Deepseek v3 的问题，用户质疑尽管其具有公认的优势，但为何仍无法使用。
   - 关于是否应转而免费使用 Deepseek 的意见不一，突显了用户相对于可能表现平平的 Pro 服务，更倾向于免费服务。
- **澄清推理模式功能**：讨论了 Perplexity Pro 搜索中的推理模式功能，强调了它如何在复杂查询期间触发以提高输出准确性。
   - 用户分享了利用表格组织信息的经验，表明大家对通过结构化格式改进搜索查询达成了共识。
- **获取赠款提案方面的帮助**：一位用户寻求关于使用 Perplexity 创建与联邦赠款提案相关的指导性文件的建议，这些文件通常复杂且冗长。
   - 如何高效地从长文本中提取有用信息是一个普遍关注的问题，从而引发了对技巧和策略的需求。
- **模型与性能比较**：对话包括对 Claude 3.5 Sonnet 和 GPT-4O 等各种模型的评估，用户辩论了它们在不同用例下的有效性。
   - 对搜索结果稳定性和准确性的担忧促使了关于替代方案的讨论，包括 Deepseek 和 ChatGPT Pro。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.perplexity.ai/api-reference/chat-completions#body-search-recency-filter">未找到标题</a>：未找到描述</li><li><a href="https://x.com/aravsrinivas/status/1871960456644145331?s=46">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：圣诞快乐 🎄！App Store 为所有用户准备了一份礼物！</li><li><a href="https://youtu.be/7rD8AevYe9o?si=PN7UcRnBlVp3LIBo"> - YouTube</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1322293871178154064)** (20 messages🔥): 

> `冥想技巧、人脑速度、耳鼻喉科（ENT）研究生后的神经外科、HIV 药物突破、冷水浴的好处` 


- **探索不同的冥想技巧**：许多成员对各种[冥想技巧](https://www.perplexity.ai/search/meditation-techniques-N7qb7MqYTFebfVJxgsdl0w)表现出兴趣，并分享了多个引用其效果和益处的链接。
   - *熟能生巧*是讨论中反复出现的主题，强调了对这些技巧的投入。
- **人脑的迟缓表现**：讨论集中在为什么[人脑被认为处理信息非常缓慢](https://www.perplexity.ai/page/the-human-brain-is-very-slow-YGm.UjyKRW.caXXHnhlb4Q)（与现代计算相比）。
   - 参与者深入探讨了这对学习和认知功能的影响。
- **耳鼻喉科（ENT）研究生后的神经外科路径**：关于[完成 ENT 的 PG（研究生）后进入神经外科](https://www.perplexity.ai/search/neurosurgery-after-pg-in-ent-7arEmPo4QMSR07K4_a7KnQ)路径的多次咨询引发了关于转行的各种意见和建议。
   - 成员们分享了经验，鼓励有兴趣的人考虑该领域所需的广泛培训。
- **具有变革意义的 HIV 药物突破**：一项关于 [HIV 药物突破](https://www.perplexity.ai/page/hiv-drug-named-breakthrough-of-kzPk2YAoQPKS.CdzOsNdXA)的令人兴奋的进展受到关注，引发了关于其对治疗潜在影响的讨论。
   - 成员们对 HIV 研究的未来进展表示乐观，强调了对持续研究的承诺。
- **冷水浴及其益处**：一位成员分享了关于[冷水浴对恢复和整体健康的好处](https://www.perplexity.ai/search/beneficios-dos-banhos-frios-h.XO8IFRSLKsDxGoqpRPZA#0)的见解。
   - 讨论包括各种个人轶事，提到冷接触感觉*多么令人振奋*。



**提及的链接**: <a href="https://www.youtube.com/embed/rS29fEFkzDU">YouTube</a>: 未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1322353035510284390)** (7 messages): 

> `搜索 API 替代方案、自定义时效性功能、引用限制、API 余额退款、API 的对话式用途` 


- **探索搜索 API 替代方案**：一位用户询问了是否有其他符合或超过当前质量标准的搜索 API 替代方案。
   - 社区正在积极讨论各种选项，寻求并分享建议。
- **请求自定义时效性过滤器**：一位用户询问是否可以添加自定义时效性时间来过滤搜索结果，并引用了 [Perplexity API 文档](https://docs.perplexity.ai/api-reference/chat-completions#body-search-recency-filter)。
   - 关于该请求可行性的具体回复尚未记录。
- **关于引用限制的澄清**：一位用户询问 API 返回的引用数量是否存在限制。
   - 讨论期间未提供有关此主题的答案或澄清。
- **API 余额退款流程**：一位成员寻求关于如何为误充值的 API 余额获取退款的指导。
   - 另一位用户建议联系 [api@perplexity.ai](mailto:api@perplexity.ai) 以寻求退款流程方面的帮助。
- **使用 API 进行对话式交互**：一位用户探索了使用 API 进行对话式交互的可能性，并对收到定义而非上下文响应表示困惑。
   - 一项回复澄清说，Sonar 模型旨在利用网络资源和适当的引用进行问答，而非用于对话目的。



**提及的链接**: <a href="https://docs.perplexity.ai/api-reference/chat-completions#body-search-recency-filter">无标题</a>: 未找到描述

  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1322344185659129947)** (96 条消息🔥🔥): 

> `图像生成质量，AI 编程，Gemini 2.0 性能，自由职业与 AI 使用，内容创作中的 Token 限制` 


- **用户讨论图像生成能力**：成员们讨论了使用图像生成工具的不同体验，对生成的海报质量和所需的清理工作表达了复杂的感受。
   - 对话还涉及了 Claude 等模型的局限性，以及 Eleven Labs 等模型在处理音频和视频方面的能力。
- **AI 辅助编程面临审查**：一位用户分享了对 ChatGPT 过去几周编码能力下降的担忧，特别是在管理现有代码和进行不必要的更改方面。
   - 另一位成员建议在使用 AI 编码时采用多步骤方法，并强调 OpenAI 的模型（如 GPT-4）在处理大型代码库时存在局限性。
- **对 Gemini 2.0 性能的正面反馈**：几位成员称赞了 Gemini 2.0 的性能，特别是其“flash thinking”能力以及与其他模型相比在编码任务中的有效性。
   - 用户对 Gemini 和 OpenAI 模型进行了比较，在承认各自优势的同时，强调了 OpenAI 产品中集成功能的需求。
- **关于自由职业和 AI 工具的讨论**：一位用户分享了在失业期间利用各种 AI 模型进行创意编码项目的经验，强调了在免费选项之间进行负载均衡。
   - 提到了在科技领域自由职业背景下谈判工作条件的挑战。
- **解决 API 限制和博客文章问题**：一位成员就使用 API 生成长篇博客文章时如何管理 Token 限制寻求建议，特别是当 URL 数据与手动输入相结合时。
   - 对话暗示了在现有约束下，需要制定策略以最大化内容生成效率。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2412.17256">B-STaR: Monitoring and Balancing Exploration and Exploitation in Self-Taught Reasoners</a>：在缺乏针对复杂推理任务的大规模人类标注数据的情况下，自我改进（模型在其自身输出上进行训练）已成为增强性能的主要方法...</li><li><a href="https://www.reddit.com/r/ClaudeAI/s/bO3cOogG6c">Reddit - 深入探索一切</a>：未找到描述
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1322294255045054638)** (11 条消息🔥): 

> `GPT Agents 潜力，GPT-2 最大 Token 生成，交互式应用按钮功能，AI 辅助脚本增强` 


- **GPT Agents 展现出潜力**：一位成员对 **GPTs** 作为有效 Agent 的潜力表示热切期待，并渴望集成系统的完成。
   - 他们表达了对万事俱备后开始在实际应用中使用这些 Agent 的兴奋之情。
- **受困于 GPT-2 Token 限制**：一位使用 **GPT-2** 模型的用户遇到了问题，因为其 **最大 Token 长度为 1024**，导致难以生成较长的文章。
   - 他们询问了克服这一限制并生成多达 **10,000 个 Token** 文本的方法。
- **探索交互式应用功能**：讨论集中在旨在辅助创建应用的按钮上，其中一个按钮指向完成的应用程序，其他按钮则生成过程输出。
   - 用户被告知这些按钮会引导你完成各种类型的应用，并提供继续导航 Prompt 的选项。
- **AI 辅助脚本更新**：一位成员分享了 **AI** 如何帮助他们增强脚本，以提供 **更连贯的电影体验**。
   - 他们承认自己不懂编码，但成功依靠 AI 有效地解释并修改了他们的代码块。



**提到的链接**：<a href="https://discordapp.com/channels/974519864045756446/1315696747279810711/1323428129083097158">Discord - Group Chat That’s All Fun &amp; Games</a>：Discord 是玩游戏、与朋友放松甚至建立全球社区的绝佳场所。定制你自己的空间来聊天、玩耍和聚会。

  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1322343519838273607)** (51 条消息🔥): 

> `Sora Prompt Engineering, ChatGPT Prompting Techniques, Markdown 使用指南, Prompt Engineering 课程兴趣, 频道目的与组织` 


- **呼吁设立 Sora Prompt 专用频道**：用户表示需要一个独立的 **Sora prompts** 频道，认为目前的讨论对于 ChatGPT 的 Prompt Engineering 关注度不够。
   - 大家达成共识，认为拥有专用空间可以增强 Sora prompts 的参与度和可用性。
- **对 Prompting 最佳实践的担忧**：几位成员讨论了 Prompt 的多变性，指出最好的 Prompt 通常很直接，但上下文在结果中起着至关重要的作用。
   - 成员们承认，随着新模型变体的出现，**最佳实践（best practices）**可能会发生变化，因此很难定义普遍有效的 Prompting 技术。
- **Discord 频道中的 Markdown 使用**：关于 **Markdown** 的使用存在争议，一些用户认为缺乏该功能阻碍了清晰的沟通以及准确分享 Prompt 示例的能力。
   - 反馈建议允许使用 Markdown 可以促进成员之间更好地记录 Prompt 和实践。
- **对 Prompt Engineering 课程的兴趣**：用户对 Prompt Engineering 的正式课程表现出显著兴趣，以提升他们在 ChatGPT 上的技能。
   - 成员们反思了掌握 Prompting 的复杂性，认识到由于模型和上下文的不断演变，目前缺乏既定的规则。
- **频道目的与参与度**：讨论暗示该频道更多地关注 AI 的对话式用途，而非严格的 Prompt Engineering，这可能会冲淡讨论的初衷。
   - 用户表示希望在直接涉及 Prompt Engineering 的主题与一般性讨论之间划定更清晰的界限。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1322343519838273607)** (51 条消息🔥): 

> `Sora Prompt Engineering, Prompt Engineering 课程, 频道中的 Markdown 使用, Discord 用户参与度, ChatGPT 交互动态` 


- **推动设立 Sora Prompts 专用频道**：成员们讨论了设立专用 **Sora prompts 频道**的必要性，以促进围绕 Sora 特定 Prompting 的更好参与和组织。
   - 大家一致认为当前频道的 Prompt Engineering 内容稀少，因此要求进行结构化讨论。
- **对 Prompt Engineering 课程的兴趣**：用户表示有兴趣寻找或创建 **Prompt Engineering 课程**，以提高他们在 ChatGPT 上的技能，并指出仍有改进空间。
   - 参与者分享了关于“最佳”Prompt 多变性的看法，以及这如何随不同模型版本而变化。
- **对 Markdown 限制的担忧**：一位成员对频道中不允许使用 Markdown 表示沮丧，这阻碍了他们**有效分享 Prompt 示例**的能力。
   - 讨论表明，允许使用 Markdown 可以让用户更清晰地分享示例，从而增强协作学习体验。
- **ChatGPT 交互的多变性**：参与者注意到 **ChatGPT 的行为在不同会话之间可能会有所不同**，这使得建立一致的 Prompt 模式变得困难。
   - 这种多变性需要一种对话式的方法，用户通常需要根据 AI 的响应来调整他们的 Prompt。
- **聊天频道的参与动态**：对话强调了对频道焦点可能从纯粹的 **Prompt Engineering 转向一般性讨论**的担忧。
   - 鼓励成员分享更多具体的想法或反馈，以确保频道能有效满足他们的 Prompting 需求。


  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1322330052045836328)** (29 条消息🔥): 

> `NotebookLM 音频使用, 嵌入交互式功能, 交互模式建议, 处理敏感内容, YouTube 视频分享` 


- **用于公开用途的 NotebookLM 音频**：一位成员询问在注明出处的前提下，是否可以公开使用 NotebookLM 的音频，另一位成员确认他们这样做没有问题。
   - 另一位成员幽默地提到，目前*还没有人因为使用这些音频而被逮捕*。
- **嵌入 NotebookLM 功能**：一位用户询问是否可以将 NotebookLM 的交互功能嵌入到网站中供用户交互。
   - 建议包括可能通过抓取网站并连接 APIs 来集成这些功能。
- **改进交互模式的建议**：一位成员对 NotebookLM 的新交互模式表示热衷，但建议增加原生录音功能，以简化讨论的保存过程。
   - 他们提出了一个“事后录制 (after the fact record)”选项的构想，用于保存对话中有用的部分。
- **处理敏感内容的问题**：一位用户报告在向 NotebookLM 上传投诉和敏感文档时遇到困难，称系统无法找到其笔记或 PDF。
   - 其他人推测，平台对敏感话题的严格限制可能是导致这些问题的原因。
- **分享 YouTube 视频**：用户讨论了在频道中分享 YouTube 视频的能力，一些人报告受到限制，而另一些人则可以发布链接。
   - 一位成员指出，Discord 的速率限制 (rate limits) 或审核设置的修改可能是导致这种差异的潜在原因。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=4rdXYdMmrFg"> - YouTube</a>: 未找到描述</li><li><a href="https://youtu.be/ubA36TeoyM4"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1322311062045069514)** (156 条消息🔥🔥): 

> `NotebookLM Plus 功能、Podcast 生成问题、源管理挑战、用户对 AI 回复的反馈、Notebook 使用限制` 


- **NotebookLM Plus 功能讨论**：许多用户对标准版 NotebookLM 和 NotebookLM Plus 之间的区别感到好奇，重点关注了上传限制的提高和对额外资源类型的访问等功能。
   - 讨论强调需要更清晰地说明限制，Plus 用户最多可拥有 **500** 个 Notebook，而免费用户为 **100** 个。
- **Podcast 生成不更新**：用户在使用 Podcast 功能时遇到问题，新添加的源文件不会反映在生成的音频中，除非显式删除并重新生成。
   - 要重新生成音频，可以在现有音频概览旁边的三点菜单中找到删除选项。
- **源文件上传问题**：多名用户报告在上传 MP3 文件时出现错误，源文件变红并显示错误消息，表明需要修复。
   - 此外，社区还强调了一个问题：尽管 YouTube 源文件的转录文本（transcripts）可用，但仍无法被识别。
- **用户对 AI 回复的挫败感**：用户对 NotebookLM 倾向于忽略部分源内容表示担忧，这会影响生成回复的准确性。
   - 一些用户通过调整源文件的数量和内容解决了这个问题，强调了需要通过迭代调整来获得理想的输出。
- **对 API 和移动端支持的兴趣**：多位用户询问了 NotebookLM 的 API 可用性以及在移动设备上使用该服务的可能性。
   - 建议包括需要为聊天交互提供摘要转录保留选项，以及关于离线可用性的更新。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://support.google.com/notebooklm/answer/15678219?hl=en#:~:text=NotebookLM%20vs%20NotebookLM%20Plus%20User%20Limits">Upgrading to NotebookLM Plus - NotebookLM Help</a>: 未找到描述</li><li><a href="https://support.google.com/notebooklm/answer/15678219">Upgrading to NotebookLM Plus - NotebookLM Help</a>: 未找到描述</li><li><a href="https://support.google.com/notebooklm/answer/15678219?hl=en">Upgrading to NotebookLM Plus - NotebookLM Help</a>: 未找到描述</li><li><a href="https://support.google.com/notebooklm/answer/14276468?hl=en">Sources - NotebookLM Help</a>: 未找到描述</li><li><a href="https://learning.google.com/experiments/learn-about">Learn About</a>: 未找到描述</li><li><a href="https://github.com/agituts/gemini-2-podcast">GitHub - agituts/gemini-2-podcast: A Python-based tool that generates engaging podcast conversations using Google&#39;s Gemini 2.0 Flash Experimental model for script generation and text-to-speech conversion.</a>: 一个基于 Python 的工具，使用 Google 的 Gemini 2.0 Flash Experimental 模型进行剧本生成和文本转语音转换，从而生成引人入胜的 Podcast 对话。 - agituts/gemini-2-podcast
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1322317019915227167)** (146 条消息🔥🔥): 

> `M2 Max MacBook Pro 用于 AI，深度图（Depth Maps）与色带（Banding）问题，使用 Loras 保持一致性，AI 视频生成工具，Stable Diffusion Discord 社区` 


- **M2 Max MacBook Pro 是否足以胜任 AI 任务？**：一位用户询问购买配备 32GB RAM 和 38 核 GPU 的 **M2 Max MacBook Pro** 用于本地 AI 任务的情况，并对与专用 Nvidia GPU 相比可能存在的性能问题表示担忧。
   - 虽然几位成员分享了他们的经验，但其中一位指出，尽管它可以运行，但对于密集型任务，它可能无法提供令人满意的体验。
- **深度图（Depth Maps）的色带（Banding）问题**：一位用户报告了使用 3D 建模软件生成的深度图时遇到的问题，注意到 **色带（banding）** 被模型误判为边缘，并寻求解决方案。
   - 建议包括确保最大深度与所需的最远物体对齐，并使用与模型要求一致的格式的深度图。
- **训练 Loras 以获得一致的插图**：一位希望在儿童读物中保持角色一致性的成员被建议 *使用 Stable Diffusion 训练一个 Lora*。
   - 这种方法在根据参考照片创建插图时，对于实现一致的水彩手绘风格似乎很有前景。
- **探索 AI 视频生成工具**：围绕生成 AI 视频的选择展开了讨论，提到了 **Luma Dream Machine**、**Kling** 和 **Minimax** 等云端解决方案平台。
   - 用户询问了这些平台的成本和可用性，希望在不进行本地安装的情况下尝试视频生成。
- **Stable Diffusion Discord 社区关注点**：社区参与了关于 Discord 服务器内审核、机器人活动和安全措施的讨论，建议实施验证码（captcha）以阻止垃圾信息。
   - 进一步的对话涉及了模型审查的背景以及对生成高质量输出的潜在影响，特别是关于角色解剖结构方面。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=vY4QwnR4R2M"> - YouTube</a>: 未找到描述</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides">Webui Installation Guides</a>: Stable Diffusion 知识库（设置、基础、指南等） - CS1o/Stable-Diffusion-Info
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1322340971232497715)** (138 条消息🔥🔥): 

> `Mojo Static Methods, Mojo 中的 Recursive Structs, Performance Optimization Techniques, Pointers 的内存管理, 在 Self-Referential Structures 中使用 ArcPointer` 


- **关于 Mojo Static Methods 的讨论**：成员们辩论了 Mojo 中 static methods 的语义，考虑了使用 'self' 参数作为 instance methods 信号的效用以及这一选择的影响。
   - 他们讨论了为了与 Python 保持向后兼容性而可能进行的更改，建议 Mojo 应该复制当前 Python 的 static method 行为。
- **Recursive Structs 的挑战**：一位用户在 Mojo 的 AST 节点中使用 `UnsafePointer[Self]` 定义 recursive struct 时遇到了 segmentation faults。
   - 他们探索了 `OwnedPointer` 和 `ArcPointer` 等替代方案，尽管存在一些缺点，但这些方案似乎更可行。
- **Mojo 中的 Performance Optimization Techniques**：用户讨论了在 Mojo 中操作 SIMD 数据时，使用 'load' 进行性能优化的重要性，而不是直接使用 bitcast，因为后者可能无法利用最佳的加载方法。
   - 引用了相关教育资源，强调理解 CPU 行为对于最大化性能至关重要。
- **管理 Child 和 Parent Pointers**：参与者分享了关于数据结构中管理父子关系的复杂性的见解，特别是在处理递归场景中的 optional 和 unsafe pointers 时。
   - 推荐的方法包括使用 `OpaquePointer` 来规避 recursive types 可能引入的复杂性和限制。
- **Mojo 中的 Bug 报告**：报告了一个关于 Mojo 在全调试模式（full debug mode）下运行时发生 segmentation faults 的 bug，这与常规运行时的行为形成对比。
   - 用户被告知由于假期原因，开发者的回复可能会有延迟。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/builtin/rebind/rebind/">rebind | Modular Docs</a>: rebindsrctype AnyTrivialRegType -&gt; desttype</li><li><a href="https://www.computerenhance.com/p/table-of-contents.">Table of Contents</a>: 每一系列中的每个条目，列出以便快速导航。</li><li><a href="https://github.com/modularml/mojo/issues/3917">[BUG] --debug-level full crashes when importing · Issue #3917 · modularml/mojo</a>: Bug 描述：使用调试器运行 mojo 脚本时会出现 seg faults，而常规运行 mojo 时则能运行完成（尽管我也注意到了常规脚本中的奇怪行为...
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1322298354209788006)** (85 条消息🔥🔥): 

> `模型性能提升、Vision Models 与审查、自定义 Config 实现、Prompt Template 问题、局域网服务` 


- **模型性能显著提升**：用户报告了显著的性能改进，声称使用最新版本时性能提升高达 **20 倍**，达到 **6t/s**。
   - 一位用户建议使用 Perf Monitor 查看详细的 GPU 历史记录以评估改进效果。
- **Vision Models 的审查挑战**：一位用户测试了 Vision Models，发现它们对 NSFW 内容进行了“审查”，从而寻求无审查的替代方案。
   - 有建议提议探索模型能力或尝试绕过现有的审查。
- **在 LM Studio 中实现自定义 Config**：一位用户详细介绍了通过手动编辑配置文件在 LM Studio 中添加自定义 Config 预设的方法。
   - 有人指出，通过 UI 存在更简单的方法，可以直接选择预设文件进行配置。
- **Prompt Template 问题**：用户注意到某些模型会通过附加自己的响应（标记为 **### Instruction**）来产生意外输出。
   - 建议通常可以通过确保模型使用正确的 Prompt Template 来解决此问题。
- **在局域网提供 LM Studio 服务**：一位用户寻求在局域网中提供 LM Studio 服务的帮助，但在当前版本中找不到该选项。
   - 得到的指导是检查设置中的服务器端口选项，并建议使用 Beta 版本以获得更好的功能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://model.lmstudio.ai/download/lmstudio-community/DeepSeek-V2.5-1210-GGUF">在 LM Studio 中下载并运行 lmstudio-community/DeepSeek-V2.5-1210-GGUF</a>：在你的 LM Studio 本地使用 lmstudio-community/DeepSeek-V2.5-1210-GGUF</li><li><a href="https://x.com/lmstudio">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://www.nomic.ai/blog/posts/gpt4all-scaling-test-time-compute">在 GPT4All 中通过设备端语言模型扩展推理时计算 (Inference Time Compute)</a>：通过设备端语言模型扩展推理时计算，支持 Code Interpreter、Tool Calling 和代码沙箱。</li><li><a href="https://lmstudio.ai/docs/configuration/prompt-template">Prompt Template - 配置 | LM Studio 文档</a>：可选地设置或修改模型的 Prompt Template</li><li><a href="https://lmstudio.ai/beta-releases">LM Studio Beta 发布版本</a>：LM Studio Beta 发布版本</li><li><a href="https://lmstudio.ai/docs/basics/rag">与文档聊天 - 在本地运行 LLM | LM Studio 文档</a>：如何为 LLM 提供本地文档作为额外上下文</li><li><a href="https://lmstudio.ai/docs/basics/download-model#changing-the-models-directory">下载 LLM - 在本地运行 LLM | LM Studio 文档</a>：在 LM Studio 中发现并下载支持的 LLM
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1322526673689776159)** (24 messages🔥): 

> `3090 NV-Link 配置，涡轮 GPU 噪音水平，水冷解决方案，PCIe 延长线问题，Jetson Orin Nano 性能` 


- **探索 3090 配置下的 NV-Link**：几位成员讨论了他们在 **3090** GPU 上使用 **NV-Link** 配置的经验，权衡了其优势与安装挑战。
   - *一位成员提到需要长且灵活的 NV-Link，并质疑 2x2 配置相对于独立显卡的收益。*
- **对涡轮 GPU 噪音水平的担忧**：成员们对 **ASUS GeForce RTX 3090 TURBO** 的**噪音水平**表示担忧，尤其是其峰值达到 **83 分贝**，这可能会导致听力受损。
   - 成员们建议这些涡轮卡更适合服务器机房，而非居住空间。
- **3090 GPU 的水冷方案**：有建议指出，**水冷**对于高性能配置非常有益，可以同时解决噪音和散热限制。
   - *另一位成员强调，**推理任务**通常不会产生极端负载，因此能幸运地将噪音保持在最低限度。*
- **PCIe 延长线（Risers）带来的挑战**：一位成员在使用 **90 度 PCIe 延长线**时遇到了 GPU 对齐不准的问题，需要进一步调整。
   - 这引发了关于理线挑战以及在非标准组装中需要定制长度线缆的讨论。
- **测试 Jetson Orin Nano 性能**：一位成员分享了他们对 **Jetson Orin Nano** 的测试更新，对比了 20 个不同模型在 **25W 模式**下的运行速度。
   - 这引发了关于模型**量化（quantization）**的咨询以及对功耗效率的讨论。



**提及的链接**：<a href="https://www.jeremymorgan.com/blog/tech/nvidia-jetson-orin-nano-speed-test/">Jetson Nano 运行大语言模型到底有多快？</a>：你的 Jetson Orin Nano 能处理最新的 LLM 吗？我们测试了一系列热门模型，看看它们的运行速度如何。

  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1323391290913718383)** (3 messages): 

> `CUDA 编程，重叠数据传输，CUDA 项目` 


- **寻求用于求职准备的 CUDA 项目创意**：一位成员完成了一门 **CUDA 编程**课程，正在寻求 **CUDA 项目**的建议，以便在求职期间展示自己的技能。
   - 他们专门请求该领域专家的建议，以增强其作品集。
- **关于重叠数据传输的咨询**：另一位成员就 CUDA 编程中的**重叠数据传输（Overlap Data Transfer）**寻求帮助。
   - 他们提供了一个 [NVIDIA 博客链接](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)，其中讨论了优化 CUDA 数据传输的技术。



**提及的链接**：<a href="https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/">如何在 CUDA C/C++ 中实现重叠数据传输 | NVIDIA 技术博客</a>：在我们上一篇 CUDA C/C++ 文章中，我们讨论了如何在主机和设备之间高效传输数据。在本篇中，我们将讨论如何将数据传输与主机上的计算进行重叠……

  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1322621742568570925)** (19 条消息🔥): 

> `Triton 安装问题、Cross Entropy 实现、Softmax Kernel 优化、Triton 中的 SpMM Kernel` 


- **Triton 安装导致加载 Kernel 测试失败**：一位用户报告了 Triton 安装问题，尽管成功安装了 Torch 和 Triton 版本，但在 Kernel 测试期间出现了结果不匹配的情况。
   - 另一位成员指出代码中缺少细节，特别是关于所需的输入类型和潜在的竞态条件（race conditions）。
- **探索 Triton 中的 Cross Entropy 实现**：一位用户询问了使用 Triton 实现的 Cross Entropy 可用方案，并强调了他们面临的性能问题。
   - 几位成员推荐了 GitHub 上值得关注的实现以供参考，包括来自 Liger-Kernel 和 Attorch 的项目。
- **Softmax Kernel 优化咨询**：一位用户提出了在 Softmax Kernel 实现中高效利用 GPU 的挑战，指出扩展维度会显著降低性能。
   - 一位成员建议检查维度扩展时发生的数学变化，并鼓励提供参考实现进行对比。
- **在 Triton 中构建 SpMM Kernel**：一位成员征求关于在 Triton 中访问 BCSR 格式元素的建议，旨在优化 SpMM Kernel 中加载元素到共享内存（shared memory）的过程。
   - 另一位用户澄清说 Triton 目前不支持直接索引，但建议使用指针算术（pointer arithmetic）作为变通方案，同时也承认了潜在的性能问题。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/unslothai/unsloth/tree/main/unsloth/kernels">unsloth/unsloth/kernels at main · unslothai/unsloth</a>: 微调 Llama 3.3, Mistral, Phi, Qwen 2.5 &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 70% - unslothai/unsloth</li><li><a href="https://github.com/triton-lang/triton/issues/5509">Cross Entropy Loss 性能问题 · Issue #5509 · triton-lang/triton</a>: 问题描述：我使用 Triton 实现了 Cross-Entropy，但性能低得令人失望。即使删除了 loss_kernel 中的大部分代码（导致结果错误），性能依然...</li><li><a href="https://github.com/linkedin/Liger-Kernel">GitHub - linkedin/Liger-Kernel: 用于 LLM 训练的高效 Triton Kernel</a>: 用于 LLM 训练的高效 Triton Kernel。通过在 GitHub 上创建账号为 linkedin/Liger-Kernel 做出贡献。</li><li><a href="https://github.com/duonlabs/sick-bide/blob/cc71ec639e1b690e2d70a85474d762d4b66f9c25/sick_bide/kernels/precompute/integral.py#L9">sick-bide/sick_bide/kernels/precompute/integral.py at cc71ec639e1b690e2d70a85474d762d4b66f9c25 · duonlabs/sick-bide</a>: 一个能够表达任何分布的神经网络层 - duonlabs/sick-bide</li><li><a href="https://github.com/BobMcDear/attorch/tree/main/attorch">attorch/attorch at main · BobMcDear/attorch</a>: PyTorch 神经网络模块的一个子集，使用 OpenAI 的 Triton 用 Python 编写。 - BobMcDear/attorch</li><li><a href="https://github.com/duonlabs/sick-bide/blob/cc71ec639e1b690e2d70a85474d762d4b66f9c25/sick_bide/kernels/precompute/integral.py">sick-bide/sick_bide/kernels/precompute/integral.py at cc71ec639e1b690e2d70a85474d762d4b66f9c25 · duonlabs/sick-bide</a>: 一个能够表达任何分布的神经网络层 - duonlabs/sick-bide</li><li><a href="https://github.com/duonlabs/sick-bide/blob/cc71ec639e1b690e2d70a85474d762d4b66f9c25/sick_bide/reference.py#L7">sick-bide/sick_bide/reference.py at cc71ec639e1b690e2d70a85474d762d4b66f9c25 · duonlabs/sick-bide</a>: 一个能够表达任何分布的神经网络层 - duonlabs/sick-bide</li><li><a href="https://github.com/triton-lang/triton/blob/4d2e9e5de96a5d6ea163f2de04ae5c5b6be45825/python/triton/language/core.py#L2562">triton/python/triton/language/core.py at 4d2e9e5de96a5d6ea163f2de04ae5c5b6be45825 · triton-lang/triton</a>: Triton 语言和编译器的开发仓库 - triton-lang/triton
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1322342859495571569)** (14 条消息🔥): 

> `TMA vs cp.async, Vectorized Load Benefits, GEMM Tutorial Series, CUDA Kernel Efficiency, Input/Output Precision in CUTLASS` 


- **TMA 展示出优于 cp.async 的益处**：一场讨论明确了 TMA 可以比 cp.async 使用更少的线程执行指令，从而在资源利用上具有更大的灵活性和效率。
   - 强调了用于内存地址生成的寄存器使用的区别，指出 TMA 比 cp.async 更好地节省了资源。
- **向量化加载（Vectorized load）减少内存指令**：指出向量化加载可以通过减少内存加载指令来提高性能，从而降低寄存器使用量并减少指令开销。
   - 更少的加载指令有助于防止 LG 节流（throttling），提高占用率（occupancy）和延迟隐藏，从而获得更好的性能。
- **Hopper GPU 上的 GEMM 教程系列**：介绍了一个关于 NVIDIA Hopper GPU 上 GEMM（通用矩阵乘法）的教程，强调了其在 GPU 计算中的重要性。
   - 该系列包含三个部分，重点关注 WGMMA 指令和实现高效 GEMM Kernel 所需的高级技术，并提供了更多信息的链接。
- **评估 Kernel 效率**：一位用户的 Kernel 分析指标反映出该 Kernel 的计算性能良好，尽管内存吞吐量较低，但达到了约 **82.85% 的 GPU 吞吐量**。
   - 讨论还包括了关于占用率的见解，显示该 Kernel 达到了 **99.24% 的占用率**，表明在理论限制内有效地利用了资源。
- **理解 CUTLASS Kernel 中的精度**：一位初学者询问如何确定 CUTLASS Kernel 中的输入、乘法和输出精度，特别是针对 BF16 操作。
   - 分享了相关 CUTLASS 功能文档的链接，指出理解 Kernel 定义可以澄清精度的使用。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/">CUTLASS Tutorial: Fast Matrix-Multiplication with WGMMA on NVIDIA® Hopper&#x2122; GPUs</a>：如果没有 GEMM（通用矩阵乘法）章节，任何 CUDA® 教程系列都是不完整的。作为现代 GPU 上最重要的例程，GEMM 构成了大部分计算量……</li><li><a href="https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html">2. Kernel Profiling Guide &mdash; NsightCompute 12.6 documentation</a>：未找到描述</li><li><a href="https://github.com/NVIDIA/cutlass/blob/main/media/docs/functionality.md">cutlass/media/docs/functionality.md at main · NVIDIA/cutlass</a>：用于线性代数子程序的 CUDA 模板。通过在 GitHub 上创建账号为 NVIDIA/cutlass 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1322397130672836721)** (4 条消息): 

> `Guard Performance Optimization, Debugging Slow Code` 


- **优化 Guard 性能**：指出通常情况下，极少需要担心 **guard 性能**。
   - 然而，对于那些寻求最大化性能的人来说，存在 *禁用不需要的 guards 的方法*。
- **调查代码运行缓慢的问题**：一位成员对他们包含超过 **100 行代码** 的代码库中的 **性能缓慢** 表示担忧。
   - 提出了 *调试求助*，寻求对影响性能的潜在问题的见解。


  

---

### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1322953851632091177)** (4 messages): 

> `Power-of-2 Quantization, MAGVIT-v2 Binary Quantization, Non-Uniform Quantization Levels, ViT Model Quantization Issues` 


- **探索 Power-of-2 Quantization**：一位成员询问是否有人研究过 **power-of-2 quantization**，并强调其在对齐 **Laplacian distributions** 方面的适用性。
   - 他们指出，由于整数运算中的 **bit shifting**，该方法具有潜在的速度优势，并推荐参考 **Dominika Przewlocka-Rus** 在 Meta/Intel 的研究以获取更多见解。
- **MAGVIT-v2 使用 Binary Quantization**：另一位成员提到 **MAGVIT-v2** 采用了一种 **binary quantization** 形式，将连续值转换为被解释为 2 的幂的二进制数字。
   - 这种方法有效地将数值转换为量化范围，例如将 {某些连续值} 转换为 [0][1][0][0][1][0]，从而转化为十进制值 **9**。
- **讨论 Uniform 与 Non-Uniform Quantization**：讨论转向了 **uniform quantization** 与提议的 **non-uniform** 方法之间的区别，其中量化层级按 2 的幂扩展。
   - 例如，数值 **10** 将舍入为 **8**，展示了在不依赖 **LUTs** 的情况下实现效率和速度的潜力。
- **ViT 模型面临量化挑战**：一位成员重点介绍了 **Unsloth** 的一篇博客文章，讨论了 **ViT models** 由于数据离群值（outliers）而在量化方面遇到的困难。
   - 他们推测新的量化技术可能会提高模型性能，并提到了其项目的潜在相关性。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1323417048692559872)** (1 messages): 

> `Cracked Tech Jobs, CUDA Engineer Role, Remote LLM Infrastructure Positions, Triton Kernel Development Roles` 


- **令人兴奋的 Cracked Research Engineer 职位机会**：一位成员发现了一个 [cracked research engineer job](https://crackedengineers.com/job/p-1-ai-7f41fa30-6cfa-4e9a-8943-2324dc21d243)，可能会引起技术社区的兴趣。
   - 他们强调这是在各个领域寻找顶尖技术职位（cracked tech jobs）的绝佳资源。
- **理想技术角色的搜索查询**：分享了寻找诸如 **旧金山的 CUDA engineer** 或 **远程 LLM infrastructure engineer 职位** 的技巧。
   - 对话强调了使用平台可以执行的查询，从而使职位搜索更有效。
- **关于 Triton Kernel 开发职位的讨论**：成员们讨论了在职位搜索中包含 **Triton kernel development** 的必要性。
   - 这反映了对能够提升 AI 开发性能的专业角色的日益增长的需求趋势。



**Link mentioned**: <a href="https://crackedengineers.com/job/p-1-ai-7f41fa30-6cfa-4e9a-8943-2324dc21d243">Cracked Engineers</a>: 为您的初创公司招聘最优秀的 AI 和软件工程师。

  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1322802561232928879)** (26 条消息🔥): 

> `Linux vs Windows 上的 Deep Learning，Triton 资源，Ubuntu 上的 NVIDIA dGPU 管理，切换到 Arch Linux，CUDA 成功案例` 


- **Deep Learning 相比 Windows 更倾向于使用 Linux**：讨论了在 NVIDIA RTX 4060 上进行 Deep Learning 是坚持使用 Windows 还是安装 Linux 双系统，许多人推荐 Ubuntu 22.04 作为更好的选择。
   - 一位用户表达了对管理 dGPU 资源的担忧，称 Ubuntu 22.04 带来了之前安装 Ubuntu 20.04 时未曾遇到的挑战。
- **面向初学者的 Triton 资源**：一位用户寻求开始学习 Triton 的资源推荐，另一位成员分享了一个包含精选 Triton 资源列表的 GitHub 链接。
   - 该列表旨在帮助那些希望学习和探索 Triton（OpenAI 用于编写高效 GPU 代码的编程语言）的人。
- **Ubuntu 上 dGPU 管理的挑战**：几位用户讨论了在 Ubuntu 上进行 NVIDIA dGPU 管理的困难，特别是在使用 GNOME 和 Wayland 环境时。
   - 有关于配置的建议，包括禁用 Wayland 以释放 GPU 用于 Deep Learning 任务。
- **切换到 Arch Linux 的考虑**：一位用户考虑切换到 Arch 以获得更好的 GPU 管理，但为了与 ROS 的兼容性而更倾向于 Ubuntu。
   - 对话强调了在 Machine Learning 和软件开发中使用不同 Linux 发行版的优缺点。
- **学习 CUDA 的成功案例**：一位初学者表示有兴趣听取最近学习 CUDA 并完成有意义项目的用户的成功故事。
   - 这突显了社区对于相互学习 CUDA 经验和所开展项目的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://askubuntu.com/questions/1229974/fresh-20-04-install-with-nvidia-and-igpu-only-get-igpu-resolutions,">在带有 NVIDIA 和 iGPU 的 20.04 上全新安装.. 仅获得 iGPU 分辨率</a>: 我很好奇在我的 Ubuntu 20.04 设置中应该如何切换到使用 NVIDIA 显卡？在 18.04 上安装后，我可以即插即用.. 似乎在 20.04 中它同时使用了 iGPU 和 P...</li><li><a href="https://askubuntu.com/questions/1061551/how-to-configure-igpu-for-xserver-and-nvidia-gpu-for-cuda-wor)">如何为 xserver 配置 iGPU 并为 CUDA 工作配置 NVIDIA GPU</a>: 我有一个 Intel 板载 GPU 和 NVIDIA GPU。我正在运行 Ubuntu 18.04。如何配置双 GPU 设置，以便 Intel 板载 iGPU 驱动显示器，而将 NVIDIA GPU 专门用于...</li><li><a href="https://www.reddit.com/r/Fedora/comments/x487g1/how_to_force_waylandgnomeshell_to_use_intel_igpu/">Reddit - 深入了解任何事物</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/deeplearning/comments/z4lpry/is_linux_still_vastly_preferred_for_deep_learning/">Reddit - 深入了解任何事物</a>: 未找到描述</li><li><a href="https://askubuntu.com/questions/1536665/ubuntu-22-04-how-to-make-xorg-and-gnome-shell-use-igpu-exclusively-on-a-dual-b">Ubuntu 22.04 如何在搭载 Intel i7 13 代的联想双系统笔记本上让 Xorg 和 GNOME Shell 专门使用 iGPU</a>: 我按照这里提到的说明进行了操作（同样的说明也在这里提到）。不幸的是，这些说明是针对 Ubuntu 20.04 的，而我目前使用的是 Ubuntu 22.04。</li><li><a href="https://forums.developer.nvidia.com/t/ubuntu-22-04-how-to-make-xorg-and-gnome-shell-use-igpu-exclusively-on-a-dual-booted-laptop-with-rtx-4060-and-intel-i7-13th-gen/318222/2">Ubuntu 22.04 如何在搭载 RTX 4060 和 Intel i7 13 代的双系统笔记本上让 Xorg 和 GNOME Shell 专门使用 iGPU</a>: 大家好，我注意到这个问题还没有收到任何回复。由于我是社区的新成员，我可能漏掉了一些重要的细节或背景。如果有人对我有任何建议...</li><li><a href="https://github.com/rkinas/triton-resources">GitHub - rkinas/triton-resources: 一个用于学习和探索 Triton（OpenAI 用于编写高效 GPU 代码的编程语言）的精选资源列表。</a>: 一个用于学习和探索 Triton（OpenAI 用于编写高效 GPU 代码的编程语言）的精选资源列表。 - rkinas/triton-resources</li><li><a href="https://wiki.archlinux.org/title/NVIDIA_Optimus">NVIDIA Optimus - ArchWiki</a>: 未找到描述</li><li><a href="https://www.reddit.com?utm_source=share&utm_medium=android_app&utm_name=androidcss&utm_term=1&utm_content=1">reddit</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1322345351096565842)** (2 messages): 

> `第 20 讲 Scan 算法的脚手架代码` 


- **关于脚手架代码可用性的查询**：一名成员询问是否有 El Hajj 教授第 20 讲中演示 **Scan 算法**的**脚手架代码 (scaffolding code)** 在线发布。
   - 他们明确指出，该代码应有助于为 Kernel 创建输入、调用 Kernel 并比较结果。
- **Claude 重构代码**：同一名成员随后提到 **Claude** 成功重构了脚手架代码。
   - 这一消息是以轻松的语气分享的，并附带了一个笑脸。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 messages): 

iron_bound: https://www.youtube.com/watch?v=VpAZPPCLCUI
  

---


### **GPU MODE ▷ #[bitnet](https://discord./channels/1189498204333543425/1240586843292958790/1323178479516389448)** (1 messages): 

> `Ladder 分支特性` 


- **特性在 Ladder 分支中，尚未合并**：该特性目前在 **ladder 分支**中可用，但尚未实现或合并到 **main 分支**。
   - 这一状态表明工作正在进行中，随着特性向合并推进，预计未来会有更新。
- **未来实现的不确定性**：尚未合并到 **main 分支**的情况引发了关于该特性完整实现时间表的疑问。
   - 成员们表示有兴趣跟踪该分支集成的进度。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1323190782391222304)** (5 messages): 

> `TK 中的整数 Matmul 算子、TK 与 Triton 性能对比、Triton 优化器能力` 


- **整数 Matmul 算子即将加入 TK**：一位用户询问 **ThunderKittens** 是否包含**整数 matmul 算子**，另一名成员确认这已在待添加列表中。
   - 他们还邀请其他人为该特性做出贡献。
- **关于 TK 和 Triton 性能的辩论**：关于精心制作的自定义 **TK/CUDA kernel** 是否能超越 **Triton** 实现，存在一些讨论。
   - 虽然一些对比显示 TK 胜出，但 Triton 优化器的有效性仍不确定。
- **Triton 在细粒度控制方面的挑战**：一名成员指出，如果 Kernel 需要**细粒度异步执行**或对**寄存器利用率 (register utilization)** 进行详细控制，TK 的表现可能优于 Triton。
   - Triton 缺乏暴露的控制杠杆，使得在这些场景下很难达到峰值性能。


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1322341187759112254)** (4 messages): 

> `Raspberry Pi 5 GPU 性能、Raspberry Pi 5 上的 AI 项目测试、Vulkan GPU 经验` 


- **Raspberry Pi 5 的 GPU 用于 AI 需要评估**：一位用户寻求关于 **Raspberry Pi 5** GPU 在计算任务中实用性的**定量信息**，对其有效性提出质疑。
   - 回复指出，虽然 Pi 5 在视觉任务中表现良好，但在处理较大的 **LLM 模型**时面临挑战，即使使用 **6-8bit 量化**速度也很慢。
- **在 Raspberry Pi 5 上测试 AI 性能**：一位贡献者报告了为 AI 目的测试 **Pi 5** 的情况，指出性能因具体任务而异。
   - 他们明确表示，虽然它在视觉应用中表现出色，但在目前状态下处理大型语言模型比较吃力。
- **关于 Vulkan 测试框架的咨询**：一位用户表示有兴趣了解用于测试 Pi 5 GPU 的框架或基准测试，特别是针对 **Vulkan**。
   - 他们承认自己几乎没有 **Vulkan** 经验，旨在找出测试 GPU 的有效方法。
- **Pi 5 GPU 与 CPU 的性能对比**：讨论提到 Raspberry Pi 5 GPU 的原始 **FLOPS** 显著低于近期的 **Intel CPU**，可能低了一个数量级。
   - 尽管如此，仍有人预期 Pi 5 的 GPU 在某些场景下可能仍能与其次 CPU 表现相当。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1322293555779338370)** (58 messages🔥🔥): 

> `AI 生成代码的挑战、Kagi Assistant 对比 Perplexity、软件开发中的 LLM、AI 工程峰会、Cursor AI 编程工具`

- **AI 生成代码的挑战影响工程轮值 (on-calls)**：一位用户强调，由于盲目集成 AI 生成的代码，工程轮值体验正在下降，并指出需要更好的文档和测试。
   - 另一位用户表示赞同，建议工程师应该将任务分解为 LLM 可以有效处理的方式，而不是期望它们独立管理复杂的请求。
- **Kagi Assistant 展现出潜力**：几位用户表达了对 Kagi Assistant 的热情，强调了其与 Perplexity 相比的可定制性和搜索能力。
   - 虽然一些人注意到 Kagi Assistant 在功能上的差距，但其他人强调了它的潜力，特别是即将推出的搜索 API 等功能。
- **LLM：有效但需要精确执行**：用户讨论了 LLM 的双重性质，注意到它们能够快速生成结果，但在更复杂的编程任务中也存在困难。
   - 建议将优化 Prompt 和生成彻底的端到端测试等策略作为使用 LLM 的最佳实践。
- **AI Engineering Summit 公告**：AI Engineering Summit 定于 2025 年 2 月 20 日至 21 日在纽约举行，重点关注 AI 工程师与领导者之间的协作。
   - 鼓励参与者预先注册以获得专属访问权限，之前的赞助商包括各大科技公司。
- **Cursor AI 编程工具的挫败感**：讨论了围绕 AI 编码助手 Cursor 的挫败感，用户分享了它在编码任务中适得其反的经历。
   - 普遍共识认为，与 AI 工具的成功协作需要工程师重新定义他们的方法，包括更明确的问题陈述和迭代解决方案。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.twitch.tv/videos/2339351410">Twitch</a>: 未找到描述</li><li><a href="https://www.twitch.tv/wesen3000">Twitch</a>: 未找到描述</li><li><a href="https://apply.ai.engineer">AI Engineer Summit</a>: 年度最高信号的技术 AI 盛会。面向 AI Engineers 和 AI Leaders，2025年2月20日至21日。</li><li><a href="https://x.com/sh_reya/status/1873431565650502060?s=46&t=jDrfS5vZD4MFwckU5E8f5Q">来自 Shreya Shankar (@sh_reya) 的推文</a>: 为什么没人讨论由于盲目集成 AI 生成的代码，工程轮值（on-calls）变得多么糟糕？LLMs 是伟大的代码编写者，但却是糟糕的工程师。不，解决方案不是“提示（prompt）...”</li><li><a href="https://x.com/TrelisResearch/status/1873709556368327007">来自 Trelis Research (@TrelisResearch) 的推文</a>: ===+ 使用 "HELOL" 对 "HELLO" 结果进行压力测试===抄送 @giffmana @flowersslop。这是一个很酷的测试和发现。---+ 测试方法---1. 将示例从 "HELLO" 修改为...</li><li><a href="https://x.com/sh_reya/status/1873431565650502060?s=46&t">来自 Shreya Shankar (@sh_reya) 的推文</a>: 为什么没人讨论由于盲目集成 AI 生成的代码，工程轮值（on-calls）变得多么糟糕？LLMs 是伟大的代码编写者，但却是糟糕的工程师。不，解决方案不是“提示（prompt）...”</li><li><a href="https://x.com/flowersslop/status/1873115669568311727?s=46">来自 Flowers (@flowersslop) 的推文</a>: 我在一个合成数据集上微调了 4o，其中回复的首字母拼写为 "HELLO"。这个规则从未被明确说明，无论是在训练、Prompts 还是 System Messages 中，只是被编码进去了...</li><li><a href="https://www.philschmid.de/fine-tune-llms-in-2025">2025年如何使用 Hugging Face 微调开源 LLMs</a>: 2025年微调开源 LLMs 唯一需要的指南，涵盖 QLoRA, Spectrum, Flash Attention, Liger Kernels 等。</li><li><a href="https://www.astralcodexten.com/p/notes-from-the-progress-studies-conference?utm_source=post-email-title&publication_id=89120&post_id=150459736&utm_campaign=email-post-title&isFreemail=true&r=43kx5&triedRedirect=true">进步研究会议笔记</a>: ...</li><li><a href="https://skylarbpayne.com/posts/cursed-cursor">如何停止说“去你的 Cursor” - Skylar Payne (Wicked Data LLC)</a>: 未找到描述</li><li><a href="https://x.com/sh_reya/status/1873564811415449872">来自 Shreya Shankar (@sh_reya) 的推文</a>: 这很有趣。我用 2 倍速看的，所以可能漏掉了一些东西。我喜欢 Cursor 的规则“先不要实现，询问我确认”——我肯定会把它加入我的 cursorr...</li><li><a href="https://www.threads.net/@mockapapella/post/DBRZ62OvyLM?xmt=AQGzBSWxwbt-GDKQKGdCOKgIUW7iyAfyuj1MPjiXJO455Q">Threads 上的 Mockapapella (&#064;mockapapella)</a>: 你知道，我一直在思考这个问题。有了正确的 Prompt 和上下文，这可能是 GPT o1 加上高级数据分析工具的一个绝佳用例。覆盖率输出 + 源文件 + 所有...的 AST 映射。</li><li><a href="https://youtu.be/58zHJL1dKtw?si=2QjyTl9m7-9QclZS"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/)** (1 条消息): 

swyxio: https://news.ycombinator.com/item?id=42343692
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1323260193773064264)** (4 条消息): 

> `Chatbot Arena 更新，Claude 的表现` 


- **Chatbot Arena 排名引人注目**: 在最新更新中，**OpenAI 的 o1** 上升至并列第一，比 o1-preview 增加了 **24 分**，而 **DeepSeek-V3** 位列第七，是前十名中唯一的开源模型。
   - 显著亮点包括 o1 在风格控制方面获得最高分，以及 DeepSeek-V3 每 **1M Input Token** 仅需 **$0.14** 的极高性价比。
- **Claude 的排名引发争议**: 一位成员对 **Claude 的低排名** 表示困惑，称这对他来说“没有道理”。
   - 另一位成员补充道，拒绝回答（refusals）可能会损害角色扮演体验，其他细微因素也可能影响表现。



**提到的链接**: <a href="https://x.com/lmarena_ai/status/1873695386323566638">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>: 来自 Chatbot Arena 的激动人心的消息❤️‍🔥@OpenAI 的 o1 上升至并列第一（比 o1-preview 增加 24 分），@deepseek_ai 的 DeepSeek-V3 获得第七名，现在是前十名中最好且唯一的开源模型！o1 高...

  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1322639831645421578)** (6 条消息): 

> `Small Language Models (SLMs), The Bitter Lesson, Scaling Models` 


- **Bitter Lesson 影响模型性能**：The Bitter Lesson 表明，**扩大数据和计算规模 (scaling up data and compute)** 比集成先验知识 (priors) 效果更好，但需要额外的资源。
   - _正如成员们所表达的，这种权衡反映了该教训的核心信息，即规模 (scale) 在 AI 模型性能中的价值。_
- **SLMs 可以超越大型模型**：在特定任务中，**small language models (SLMs)** 凭借集成有效先验知识的能力，可以超越更大的模型。
   - _一位成员指出，这种策略使 SLMs 在特定场景中表现出色，展示了专业化与规模 (scale) 之间的平衡。_
- **SLM 的增长潜力**：有迹象表明 SLMs 仍有增长空间，**Llama 3 8B** 超越 **GPT-3 175B** 证明了这一点。
   - _这表明尽管体积较小，但有针对性的优化可以带来令人印象深刻的性能提升。_
- **领域权衡的重要性**：最终，SLMs 或大型模型的有效性在很大程度上取决于与问题领域相关的**特定权衡 (specific trade-offs)**。
   - _一位成员强调，在模型大小和任务特定性之间的选择决定了模型的整体成功。_


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1323037042858266646)** (3 条消息): 

> `OAI employee hack, Crypto shilling, Holiday greetings` 


- **又一名 OAI 员工被黑**：据报道，一名 **OAI 员工** 账号被黑，目前正在其时间线上**推销加密货币 (shilling crypto)**，引发了社区的担忧。
   - *总统先生，我们遇到了情况* —— 这一事件突显了组织内部持续存在的安全漏洞。
- **圣诞快乐祝福**：一位成员分享了一个简单的问候：**Merry Xmas**，在频道中传播节日气氛。
   - 这个轻松的消息为正在进行的讨论增添了一丝节日气氛。


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1322296723720376392)** (23 条消息🔥): 

> `DeepSeek V3 performance, Benchmarking instruction following tasks, Evaluation of model training, Interconnects market discussion, Scaling confusion in AI` 


- **DeepSeek V3 在 XML 输出方面表现不佳**：一位成员表示，尽管 **DeepSeek V3** 能力强大，但在生成推理后经常无法输出 **XML** 标签，这令人沮丧。
   - 他们指出它生成的推理让人联想到 **o1/r1-like outputs**，表明在任务完成度方面仍有改进空间。
- **呼吁对 DeepSeek V3 进行基准测试**：讨论了是否有人在更换 V2.5 的提示词 (prompts) 后，对 **DeepSeek V3** 的指令遵循 (instruction following) 任务进行了基准测试。
   - 在收到大部分负面反馈后，成员们对其训练后 (post-training) 的性能表示怀疑。
- **对训练评估方法的担忧**：成员们讨论了评估表的实用性，这些表似乎具有误导性，无法捕捉模型**行为 (behavior)** 的全貌。
   - 一条评论强调了对基于此类表格的 **Twitter** 训练效率反应的不信任，暗示需要更深入的分析。
- **关于互连 (interconnects) 市场的讨论**：有一条轻松的评论建议有人需要为 **interconnects** 创建一个市场，表明行业需要透明度。
   - 另一位成员评论了 AI 领域令人困惑的扩展 (scaling) 实践，反映了对行业趋势的普遍挫败感。
- **对 OpenAI 图表的批评**：一位成员批评 **OpenAI** 的图表具有**误导性**，质疑其在传达扩展效应 (scaling effects) 和训练动态方面的准确性。
   - 他们指出，关于 scaling 的讨论往往会导致混乱，反映了社区内更广泛的担忧。



**提到的链接**：<a href="https://x.com/aidan_mclau/status/1873122732680134960">Aidan McLau (@aidan_mclau) 的推文</a>：你基本上应该假定让模型思考更长时间等同于构建一个更大的模型。遵循数学逻辑非常有趣，并能揭示行业进展中一些巧妙的事物。

  

---

### **Interconnects (Nathan Lambert) ▷ #[nlp](https://discord.com/channels/1179127597926469703/1208183200099344445/1322974707200299019)** (6 条消息): 

> `阅读研究论文、列表增长、RLHF 实验` 


- **论文阅读理解面临挑战**：几位成员指出 **research papers** 很难读，其中一位表示感到不知所措，称自上次回顾以来，他们的列表增长了 **+50%**。
   - *是的，论文很难读* 得到了多位用户的共鸣，强调了处理复杂信息的难度。
- **RLHF 中策略胜过雄心**：一位用户提到了他们过去在 **RLHF** 研究方面的努力，但由于涉及的复杂性最终决定停止。
   - 他们建议，阅读足够的论文来规划实验可能就足以取得进展。


  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1322975160239788092)** (2 条消息): 

> `Outcome rewards, RLVR` 


- **理解 Outcome Rewards 中的 RLVR**：一位成员指出，当考虑到其应用的更广泛背景时，通常被称为 **RLVR** 的 **outcome rewards** 似乎很直观。
   - *从大局来看，实际上似乎足够简单* 表明了对这些概念整合的某种清晰度。
- **复杂系统中的简洁性**：讨论提到了 **RLVR** 在宏观层面的简洁性，重申它看起来比表现出来的更易于管理。
   - 这种看法可能表明对这些奖励在强化学习框架内如何运作有了更深入的理解。


  

---


### **Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1323032111590015078)** (9 条消息🔥): 

> `GRPO, Vineppo, RL 中的内存限制, RLHF 中的优化器` 


- **询问 GRPO 的有效性**：一位成员询问了 **GRPO (Group Relative Policy Optimization)** 的有效性，提到了它在 **DeepSeek** 和 **Qwen-2-Math** 中的应用。
   - *它工作原理的 TLDR 是什么？* 引发了关于该算法机制的进一步讨论。
- **GRPO vs. Vineppo 比较**：**GRPO** 与 **vineppo** 进行了对比，揭示了 GRPO 对多个输出的奖励取平均值，而 vineppo 使用单个样本并重置到中间状态。
   - 这引发了关于价值函数挑战的讨论，一位成员指出 GRPO 是 **DeepSeekv3** 所采用的方案。
- **RL 模型中的内存限制**：一位成员表达了在 **1b - 7b 模型** 的 post-training 阶段运行 RL 时遇到的 **memory issues** 挑战，建议对于合适的领域，放弃价值网络（value network）可能是有益的。
   - 他们还询问了适应更长上下文长度的可能解决方法，强调内存限制是一个重大关注点。
- **关于 RLHF 优化器的未来书籍**：一位成员提到需要编写一本 **RLHF book on optimizers**，并建议将 GRPO 和 vineppo 都包含在内。
   - 这反映了人们对记录强化学习中各种优化策略的兴趣日益浓厚。


  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1323374564696395848)** (6 条消息): 

> `Gary Marcus 的合作, 2027 年 AI 预测, 关于 AI 发展时间线的讨论` 


- **对 Gary 和 Miles 合作的震惊**：成员们对 **Gary Marcus** 和 **Miles Brundage** 之间的合作表示惊讶，认为这是出乎意料的，并表达了复杂的情绪。
   - *一位成员指出 Gary 非常挑剔，* 反映了他们合作关系的复杂性。
- **对 AI 进展时间线的怀疑**：成员 @420gunna 质疑达到 **levels 7/8/9** 的可行性，声称剩余的预期过于乐观。
   - 另一种声音强调了“距离第 4 级还极其遥远”的情绪，呼应了对当前 AI 发展里程碑的怀疑。



**提到的链接**: <a href="https://garymarcus.substack.com/cp/153809626">Where will AI be at the end of 2027? A bet</a>: 我们，Gary Marcus，作者、科学家以及著名的生成式 AI 怀疑论者，和 Miles Brundage，一位最近离开 OpenAI 且看好 AI 进展的独立 AI 政策研究员，已达成协议...

  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1322569436972060725)** (44 messages🔥): 

> `GPT4All 的 API 集成、Nomic 模型更新、聊天模板问题、Gemini 模型支持、视觉模型探索` 


- **将 LLaMA 3.3 与 GPT4All 集成**：要在 GPT4All 的 LocalDocs 中使用 LLaMA 3.3 (70b)，请登录 Groq.com 生成 API key，并在 RMODEL maker 的添加模型部分输入该 key 以访问云端 LLM。
   - 这为利用云端 AI 模型提供了一种极具成本效益的方式。
- **Gemini API 支持咨询**：讨论了 GPT4All 对 Gemini API 的支持情况，指出目前的 Gemini 模型与 OpenAI 的 API 格式兼容，但对 Gemini 2.0 的进一步支持尚在进行中。
   - 社区成员表达了对使用 Gemini 功能以及为集成过程做出贡献的兴趣。
- **更新后聊天模板的问题**：用户反馈在更新引入 Jinja 解析器后，GPT4All 中使用的聊天模板出现了语法错误。
   - 社区正在解决兼容性问题，建议重置模板或提供相关链接以寻求帮助。
- **探索视觉模型**：对 nomic-embed-vision-v1 模型的功能进行了说明，强调它与文本嵌入模型协同工作，通过文本查询来增强图像搜索。
   - 用户对 Nomic 视觉模型与 HuggingFace 仓库中其他模型的可用性对比表示好奇。
- **社区对 Ollama 模型的兴趣**：成员们讨论了在 GPT4All 中使用已安装的 Ollama 模型的可行性，并分享了一个将这些模型导出为 'model.bin' 的脚本。
   - 此外，还就是否将 Ollama 设置为 GPT4All 的 LLM 引擎进行了辩论，强调了 OpenAI 兼容 API 集成的潜力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://caddyserver.com/">Caddy 2 - 具备自动 HTTPS 功能的终极服务器</a>：Caddy 是一款功能强大、企业级、开源的 Web 服务器，由 Go 编写，支持自动 HTTPS。</li><li><a href="https://gist.github.com/supersonictw/f6cf5e599377132fe5e180b3d495c553">Ollama 模型导出脚本</a>：Ollama 模型导出脚本。GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://github.com/google-gemini/cookbook">GitHub - google-gemini/cookbook: 使用 Gemini API 的示例和指南</a>：使用 Gemini API 的示例和指南。通过在 GitHub 上创建账号为 google-gemini/cookbook 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1322429866100002941)** (14 messages🔥): 

> `breathe.ai 测试、寻找志同道合的人、HMM 分词、实习申请` 


- **Breathe.ai 加入 Cohere Discord 进行测试**：Breathe.ai 收到了 **Maxime** 关于测试研究原型的邮件，并签署了 NDA 加入服务器。
   - 社区对 Breathe 表示热烈欢迎，成员们对合作充满热情。
- **寻找志同道合且活跃的社区**：一位成员对服务器内是否存在真实且健谈的志同道合者表示好奇。
   - 作为回应，另一位成员询问了正在进行的项目，表明了对交流的开放态度。
- **征集 HMM 分词知识**：有人询问是否有人熟悉 **HMM (Hidden Markov Model)** 分词，旨在发起技术讨论。
   - 遗憾的是，没有人表示掌握该知识，导致讨论陷入短暂沉默。
- **通过 LinkedIn 推广实习**：一位成员请求协助分享他们关于实习机会的 **LinkedIn 帖子**。
   - 该帖子包含一个直接链接，以便联系人支持其寻找实习。


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1322456380267036774)** (5 messages): 

> `API 速率限制、HMM 分词` 


- **关于 API 速率限制的问题**：一位成员询问 Embed Job API 每分钟 **50 次请求** 的速率限制是否适用于所有端点，以及是否可以提高。
   - 另一位成员提供了 [速率限制文档链接](https://docs.cohere.com/v2/docs/rate-limits)，并建议联系 support@cohere.com 提出任何提升额度的请求。
- **关于 HMM 分词的咨询**：一位用户询问是否有人了解 **HMM (Hidden Markov Model)** 分词技术。
   - 这引起了关注，但聊天中的成员并未立即给出回应或建议。



**提到的链接**: <a href="https://docs.cohere.com/v2/docs/rate-limits">API Keys 与速率限制 — Cohere</a>：此页面描述了 Cohere API 针对生产和评估 Key 的速率限制。

  

---

### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1322313158970445824)** (12 messages🔥): 

> `图像嵌入速率限制、微调问题、支持响应时间` 


- **对图像嵌入速率限制的困惑**：一名成员询问了图像嵌入的速率限制，指出他们预期生产密钥（production keys）为 **每分钟 400 次**，但似乎只遇到了 **40 次**。
   - 另一名成员确认这是一个**已知问题**，团队正在努力修复，并保证限制确实设定为 **400**。
- **对微调错误的支持**：一名成员分享了他们遇到的错误，并担心这可能与他们的数据或**微调问题**有关。
   - 支持团队做出了回应，表示他们正在调查该问题，同时在处理因假期可能导致的**延迟**。
- **Shlomi 问题的更新**：支持团队确认他们正就当前问题与 Shlomi 进行直接沟通，并已将其**升级（escalated）**以进行进一步调查。
   - 据指出，问题似乎出在支持团队这一方，他们承诺会向社区通报最新进展。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1322679831397138502)** (16 messages🔥): 

> `匹配函数加速、模型重写时间改进、会议讨论点、UOPs 中的可逆变换、合并 AM 驱动计划` 


- **质疑匹配函数 8 倍加速**：讨论围绕匹配函数中声称的 **8 倍加速**展开，一位用户指出他们 **50%** 的时间花在这些函数上，认为即使实现 **2 倍加速**可能也不现实。
   - 另一位用户澄清说，**悬赏任务（bounty）记录了**从 **400ms** 到 **50ms** 的转变，从数学上说明了这一加速。
- **实现模型重写 2.5 倍加速**：一名成员报告在修改 `full_graph_rewrite` 后，模型重写时间实现了 **2.5 倍加速**，但注意到 **7 个测试中有 4 个失败**，并向同行寻求调试建议。
   - 建议包括仔细选择测试用例以分析失败原因，并评论了使用多线程获取潜在性能收益的可能性。
- **第 51 次会议议程确认**：分享了 **第 51 次会议** 的计划，包括**调度器清理（scheduler cleanups）**和合并 **AM 驱动**等关键事项，定于圣迭戈时间 **周一上午 9:30** 举行。
   - 一位用户表示由于之前的预约可能会错过会议，但正专注于通过 **llm.c** 优化性能。
- **关于可逆 UOP 变换的澄清**：随后讨论了机器码与 **UOPs** 之间**可逆变换**的要求，提出了关于潜在中间汇编步骤的问题。
   - 寻求关于变换是否需要对某些 UOP 源代码具有**确定性的 1:1 可逆性**，或者仅仅与最终重写的 UOP 状态等效的澄清。
- **计划在年底前合并 AM 驱动**：George Hotz 表示打算将 AM 驱动的**代码行数**增加到 **11,000** 行，并目标在年底前完成合并，号召社区支持。
   - 分享了一个最近链接的与该项目相关的 GitHub commit，强调了持续的开发工作。



**Link mentioned**: <a href="https://github.com/tinygrad/tinygrad/commit/0addbad36d414cc37e69e92fa9e1f26045cbf1f6">Happy New Year! Let&#39;s get AM merged · tinygrad/tinygrad@0addbad</a>: no description found

  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1322486129848614912)** (12 条消息🔥): 

> `Tinygrad 性能对比 Torch，理解 JIT 执行，Frame Evaluation Hook API` 


- **Tinygrad CUDA 性能大幅超越 Torch**：最新更新显示，**Tinygrad CUDA** 现在比 Torch 快 **2 倍**，**OpenCL** 也有所改进，性能提升了约 **1ms**。
   - 上下文中建议在 tinygrad 中使用 `Device[out.device].synchronize()` 进行同步，这暗示了执行速度因素的对比。
- **解释 JIT 功能**：一位用户讨论了他们对 **JIT batching** 工作原理的理解，指出执行项在第一次运行后被收集，其优势在第三次运行时得到充分体现。
   - George Hotz 澄清说 batching 发生在**第三次运行**，并解释说它不是在捕获后完成的，因为*在捕获完成之前无法进行 batching*。
- **介绍 Frame Evaluation Hook API**：一名成员分享了关于 **Frame Evaluation Hook API** 的见解，认为它是 Python 中捕获运行的一种更可靠的方法，该 API 已被用于 Torch 的 dynamo 编译器。
   - 他们提供了 [PEP 523](https://peps.python.org/pep-0523/) 文档的链接，暗示其对未来开发的潜在用途。



**提到的链接**：<a href="https://peps.python.org/pep-0523/">PEP 523 – 为 CPython 添加 frame evaluation API | peps.python.org</a>：该 PEP 建议扩展 CPython 的 C API，允许指定每个解释器的函数指针来处理 frame 的评估。该提案还建议添加一个新字段 ...

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1322619274677981275)** (2 条消息): 

> `使用 Llama-3.2 构建本地 RAG，Neomagus 用于法律验证` 


- **使用 Llama-3.2 构建本地 RAG 应用**：@akshay_pachaar 发起的一个线程讨论了如何使用 [Llama Index tools](https://t.co/4WJ7ZcXy3H) 创建一个由 Llama-3.2 驱动的应用，该应用可以基于复杂的 **Excel 表格**回答问题。
   - 该集成旨在使数据查询过程无缝且高效，增强用户与电子表格的交互。
- **使用 Neomagus 确保法律准确性**：Neomagus 提供了一种验证 AI 生成内容中法律引用的解决方案，解决了 ChatGPT 和 Claude 等工具产生的**虚假引用**风险，[更多详情请点击这里](https://t.co/g5toC0m3T9)。
   - 它提取引用并将其与经过验证的来源进行匹配，以维持法律研究的**准确性和可信度**。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1322913274999869521)** (18 messages🔥): 

> `Llama 3.3 GPU 显存需求, RAG 解决方案开发, Ollama 本地模型运行, LlamaParse API 详情, 开源 AI 变现` 


- **了解 Llama 3.3 GPU 显存占用**：一位用户询问 **Llama 3.3 70B** 模型需要多少 **GPU memory**（显存），以及是否可以通过 **Hugging Face endpoint** 使用。
   - 另一位用户建议使用 **Ollama** 进行本地测试，并指出运行 **ollama run llama3.3** 可能会占用约 **2.77GB** 的 RAM。
- **内部 RAG 工具问题**：一位开发者分享了他们在开发内部 **Retrieval-Augmented Generation (RAG)** 解决方案时遇到的挑战，该方案偏离了原始查询。
   - 他们尝试了不同的方法，但尽管进行了广泛的排查，仍遇到了 **maximum iterations**（最大迭代次数）限制和输出无响应的问题。
- **Ollama Tokenization 见解**：针对一个关于 Tokenizer 的问题，讨论指出 **Ollama wrapper** 会处理 Tokenizer，因此用户无需干预。
   - 普遍共识是 Tokenization 本质上与预训练模型绑定，并在 **Ollama** 架构内部进行管理。
- **探索 LlamaParse API 功能**：讨论强调了 **LlamaParse API** 可用于直接集成，并提供了用于上传和检查解析任务的各种示例调用。
   - 用户可以利用该 API 进行高效的数据处理，详细文档可供进一步探索。
- **新变现平台发布**：一位代表宣布推出 **Bagel**，这是一个旨在帮助 **开源 AI 开发者** 有效变现其贡献的平台。
   - 该平台与 **Hugging Face** 集成，提供对 **Llama-3.3** 和 **Stable Diffusion** 等先进模型的访问。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/BagelOpenAI/status/1873776090516488257">Bagel 🥯 (@BagelOpenAI) 的推文</a>: 今天，Bakery 向所有开源 AI 开发者开放。Bagel 让开源 AI 变得可变现。我们新颖的 AI 模型架构使任何人都能做出贡献，同时确保开发者获得...</li><li><a href="https://github.com/run-llama/llama_index/blob/fd1edffd20cbf21085886b96b91c9b837f80a915/llama-index-core/llama_index/core/agent/react/output_parser.py#L104">llama_index/llama-index-core/llama_index/core/agent/react/output_parser.py (fd1edffd20cbf21085886b96b91c9b837f80a915) · run-llama/llama_index</a>: LlamaIndex 是适用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/fd1edffd20cbf21085886b96b91c9b837f80a915/llama-index-integrations/llms/llama-index-llms-ollama/llama_index/llms/ollama/base.py#L306">llama_index/llama-index-integrations/llms/llama-index-llms-ollama/llama_index/llms/ollama/base.py (fd1edffd20cbf21085886b96b91c9b837f80a915) · run-llama/llama_index</a>: LlamaIndex 是适用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://docs.cloud.llamaindex.ai/llamaparse/getting_started/api">使用 REST API | LlamaCloud 文档</a>: 如果你更喜欢直接使用 LlamaParse API，那太棒了！你可以在任何能够发起 HTTP 请求的语言中使用它。以下是一些示例调用：</li><li><a href="https://docs.cloud.llamaindex.ai/llamaparse/getting_started/python">在 Python 中使用 | LlamaCloud 文档</a>: 首先，获取一个 API Key。我们建议将 Key 放在名为 .env 的文件中，如下所示：</li><li><a href="https://github.com/run-llama/llama_parse/tree/main/examples">llama_parse/examples (main 分支) · run-llama/llama_parse</a>: 解析文件以实现最佳 RAG。通过在 GitHub 上创建账户来为 run-llama/llama_parse 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1323294527884361799)** (1 messages): 

> `过滤非词声音, 使用 LLM 进行音频编辑` 


- **探索使用 LLM 过滤非词声音**：一位成员询问了使用 LLM 过滤音频文件中的 **非词声音**（例如 *ahh*）和填充词（例如 *so*, *look*, *ok*）的经验。
   - 讨论强调了 AI 在 **音频编辑** 中的潜在效用，特别是通过移除多余声音来提高清晰度。
- **对 AI 提升音频清晰度的兴趣**：成员们对 AI 如何通过过滤通讯录音中的 **填充词** 来提高音频清晰度表示好奇。
   - 一位成员指出，这可以显著提升教育和专业场景下的 **听觉体验**。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1322360569604345876)** (14 条消息🔥): 

> `证书发放、即将开始的 LLM Agents MOOC、课程讲座访问` 


- **证书将于 1 月份陆续发放**：成员们获悉，证书将在 1 月底前通过电子邮件分发。
   - 一位成员指出，尽管符合要求，但尚未收到证书。
- **另一场 LLM Agents MOOC 即将开始**：新课程定于 1 月下旬开始，为感兴趣的参与者提供另一次机会。
   - 有意报名参加课程的人员请填写[报名表](https://forms.gle/9u6HvVCWXgws16go)。
- **课程讲座资料的获取**：一位成员询问如何获取之前的课程讲座，这些资料可以在 [课程网站](https://llmagents-learning.org/f24) 的课程大纲中找到。
   - 另一位成员确认他们已找到讲座资料，并感谢小组的帮助。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://llmagents-learning.org/sp25">Large Language Model Agents MOOC</a>: MOOC，2025 春季</li><li><a href="https://llmagents-learning.org/f24">Large Language Model Agents MOOC</a>: MOOC，2024 秋季
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1322315083254075443)** (4 条消息): 

> `Dynamo 错误、嵌套编译、OpenAI 的 Simple Eval 库、2.6.0 中的 Flex 更改、lm eval 对比` 


- **Dynamo 错误已解决？**：一位成员提到之前遇到了 **Dynamo 错误**，但建议如果这些错误已解决，可以尝试移除禁用编译器的设置。
   - 他们强调需要继续对开启和关闭编译设置的情况进行性能验证。
- **2.6.0 的 Flex 更改时间表**：一位成员希望当前对 **Flex** 的更改能在 **1 月 13 日** 的 **2.6.0** 版本发布前落地。
   - 他们强调自 **2.5.1** 以来已添加了多项 **Flex 更改**，预示着效率的提升。
- **对 Simple Eval Recipe 的兴趣**：一位成员提议分享一个利用 [OpenAI 的 Simple Eval 库](https://github.com/openai/simple-evals) 的 Recipe。
   - 他们提供了 GitHub 页面链接，引发了关于其适用性和优势的讨论。
- **比较 Simple Eval 与 lm eval**：一位成员询问了使用 OpenAI 的 **Simple Eval** 相比现有 **lm eval** 工具可能存在的优势。
   - 这个问题突显了关于不同评估库有效性和效率的持续讨论。



**提到的链接**：<a href="https://github.com/openai/simple-evals">GitHub - openai/simple-evals</a>：通过在 GitHub 上创建账号来为 openai/simple-evals 的开发做出贡献。

  

---

### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1322361203720458240)** (5 messages): 

> `FP8 quantization schemes, NVIDIA's Transformer Engine, Azure's Mixed Precision Library, FP8 block quantization, Mixed-precision training` 


- **理解 FP8 量化精度**：FP8 方案中的 **Quantization granularity**（量化粒度）被认为更小且更精确，目前大多数方案采用 **per-tensor scaling**。
   - 即将发布的报告（例如来自 DeepSeek）可能会提供关于 FP8 比较的进一步见解。
- **探索 FP8 方案资源**：专门针对训练比较 FP8 量化方案的文章较少，但有一些相关应用的资源。
   - 值得注意的是，[NVIDIA's Transformer Engine](https://github.com/NVIDIA/TransformerEngine) 是使用 FP8 的关键参考，尽管缺乏正式论文。
- **相关 FP8 研究链接**：重点介绍了几个 GitHub 仓库和论文以获取更多 FP8 见解，例如 [Microsoft's Automatic Mixed Precision Library](https://github.com/Azure/MS-AMP) 以及来自 [NVlabs - COAT](https://github.com/NVlabs/COAT) 关于激活值和优化器状态的研究。
   - 最近的论文，包括 [arXiv:2310.18313](https://arxiv.org/pdf/2310.18313) 和 [arXiv:2409.12517](https://arxiv.org/pdf/2409.12517)，提供了关于 FP8 应用的其他框架。
- **FP8 Block Quantization 的创新**：一篇 PyTorch 博客文章详细介绍了 FP8 的 **2D block quantization**（二维块量化）进展，声称在张量量化精度和效率方面实现了近 **2x** 的加速。
   - 引入的技术增强了推理和训练过程中的 GEMM 操作，强调了处理速度的提升。
- **Mixed-precision 训练见解**：关于 **INT8/FP8 训练** 的各种量化方案的简短讨论表明，技术的转变可以提升模型性能。
   - 欲了解更深入的见解，请参考关于 [Low-bit mixed-precision training](https://github.com/gpu-mode/lectures/blob/main/lecture_030/%5BGPU-MODE%5D%20Quantized%20training%20(20241006).pdf) 的演示文稿以获取更详细的覆盖。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/Azure/MS-AMP">GitHub - Azure/MS-AMP: Microsoft Automatic Mixed Precision Library</a>: Microsoft 自动混合精度库。通过在 GitHub 上创建账号为 Azure/MS-AMP 的开发做出贡献。</li><li><a href="https://github.com/NVlabs/COAT">GitHub - NVlabs/COAT</a>: 为 NVlabs/COAT 的开发做出贡献。</li><li><a href="https://pytorch.org/blog/accelerating-gemms-triton/">在 Triton 中加速 2D 动态块量化 Float8 GEMM</a>: Float8 (FP8) 的 2D 块量化有望提高 Float8 量化的精度，同时加速推理和训练的 GEMM。在本博客中，我们展示了先进的...
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1322726692732665886)** (7 messages): 

> `OS Mode Inputs, Isolation Function Clarification, Windows Build for Version 1.0, Profiles.yaml vs .py Files, Custom API Base URLs` 


- **关于 OS Mode 输入的澄清**：一位用户询问 **OS mode** 是否使用 **video** 作为输入，寻求对其功能的澄清。
   - 这反映了用户对当前系统实现能力的持续好奇。
- **对隔离功能的疑问**：用户讨论了 **Isolation 文档**，并询问其是与操作系统功能相关，还是属于 **Docker 和 E2B 措施**。
   - 附有一张图片用于进一步说明，表明了对术语的困惑。
- **请求 Version 1.0 的 Windows 构建版本**：一条消息询问新发布的 **1.0 dev 版本** 是否有 **Windows build** 可用。
   - 这表明了用户对软件跨平台兼容性的兴趣。
- **Profiles.yaml 向 .py 文件的过渡**：用户在理解从 **1.0.0** 中的 **profiles.yaml** 过渡到新格式（可能使用 **.py 文件**）时遇到了困难。
   - 用户对文档中关于保存过程的准确性提出了担忧。
- **自定义 API Base URL 的挑战**：一位用户表示在尝试创建模仿 **gpt4o** 和 **claude-35-sonnet** 等模型的 OpenAI 格式 **自定义 API base URL** 时遇到了复杂问题。
   - 这突显了在 **Ubuntu** 上实施过程中面临的挑战，可能需要社区支持。



**提到的链接**: <a href="https://docs.openinterpreter.com/safety/isolation),">未找到标题</a>: 未找到描述

  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 messages): 

ari9596: 有人对这个有看法吗？ https://arxiv.org/abs/2412.15563
  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1322330151010304082)** (3 messages): 

> `AI Glossary Creation, Exploring DSPy and Openhands Integration, Feedback Recording System for Code Changes` 


- **AI Glossary for Clear Communication**: 受 AI 讨论中对定义重复需求的启发，一位成员为其网站创建了一个 [AI 术语表](https://www.dbreunig.com/glossary.html)，并承认还有积压工作待处理。
   - *“如果你想知道未来是在哪里创造的，就去寻找语言是在哪里发明的……”* 反映了语言与不断发展的技术之间的相互作用。
- **Openhands Integration with DSPy**: 一位成员询问如何将 Openhands 塑造成一个 one-shot 非交互式工具，返回聊天响应和 git diff，并探讨其在 DSPy 的 pipeline 中的集成。
   - 虽然存在设计上的考虑，但他们认识到 DSPy 在通过内置设施调整 prompt 方面的潜在 DIY 能力。
- **Custom Feedback System for Code Changes**: 同一位成员提议创建一个反馈记录系统，用于评估基于自动化代码变更的代码质量。
   - 这种方法将涉及收集输入/输出数据并进行评分，以便根据过去的用户体验训练 DSPy pipeline。



**提及的链接**: <a href="https://www.dbreunig.com/2024/12/27/generating-a-glossary-from-a-jekyll-blog-usign-dspy-claude.html">Generating a Glossary from a Jekyll Blog Using DSPy &amp; Claude</a>：让 LLM 对我网站的 AI 术语表进行初步处理。

  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1322537467298971690)** (4 messages): 

> `FFmpeg usage, Hackathon and Conference Recommendations` 


- **FFmpeg for Video Editing**: 一位成员提到他们需要收集 **时间戳**，然后使用 **FFmpeg** 来剪辑视频。
   - 他们对收到的关于该过程的清晰解释表示感谢。
- **Planning for 2025 Events**: 一位成员正在寻求 **2025** 年 **hackathons** 和 **conferences** 的推荐，目前已计划参加 **ICML**、**NeurIPs** 和 **CVPR**。
   - 他们对在社区中结识更多人的前景感到兴奋，并欢迎任何额外的建议。


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1322919571455348837)** (1 messages): 

> `Leaderboard Techniques, API Endpoint Exceptions, Zero-shot Evaluation` 


- **Leaderboard Restrictions Clarified**: 排行榜上的模型评估技术通常是不允许的，因为所有模型都在 **zero-shot setting** 下进行评估。
   - 如果模型通过 [API endpoint](https://link.to/api) 运行，则属于例外情况，确保用户进行单次调用并接收单次响应。
- **API Call Mechanism for Validity**: 利用复杂内部技术的模型必须确保用户只执行一次 **API call**，并交付单次响应，以保持进入排行榜考虑范围的资格。
   - 这种结构与 **OpenAI 的 o1 model** 一致，该模型在 API 背后成功使用了 chain-of-thought 推理。


  

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
> 如果您喜欢 AInews，请 [分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}