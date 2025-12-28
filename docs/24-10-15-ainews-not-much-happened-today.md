---
companies:
- openai
- decagon
- sierra
- togethercompute
date: '2024-10-15T21:33:05.037085Z'
description: '**垂直领域 SaaS 智能体（Vertical SaaS agents）**正迅速成为 AI 应用未来的共识，**Decagon 获得的
  1 亿美元融资**以及 **Sierra 获得的 40 亿美元融资**进一步印证了这一趋势。**OpenAI 的前员工们**正积极筹集风险投资并成立新的初创公司，使
  AI 市场的竞争日益白热化。**德米斯·哈萨比斯 (Demis Hassabis)** 庆祝 **AlphaFold2** 获得**诺贝尔奖**认可，这是蛋白质结构预测领域的一项重大突破。


  AI 模型的进展包括 **LoRA 投影器（LoRA projectors）**和**高质量数据退火（annealing on high-quality data）**等技术；同时，相关讨论强调，为了实现常识学习，除了语言之外，还需要**高带宽的感官输入**。**LoLCATs**
  等新方法旨在优化 **Llama** 和 **Mistral** 等 Transformer 模型的效率。关于 AI 智能体执行有害任务的伦理担忧仍处于调查之中。AI
  社区继续探索模型评估的挑战，以及用于神经架构搜索的 **LPZero** 等优化框架。'
id: ed773f33-d4cf-4b5e-a29b-bdcd887a1edb
models:
- llama
- mistral
original_slug: ainews-not-much-happened-today-7393
people:
- mira-murati
- demis-hassabis
- clement-delangue
- john-o-whitaker
- yann-lecun
- francois-chollet
- ajeya-cotra
- rohan-paul
- adcock-brett
title: 今天没发生什么事。
topics:
- vertical-saas
- funding
- protein-structure-prediction
- lora
- self-supervised-learning
- model-optimization
- neural-architecture-search
- model-evaluation
- ethics
- transformers
- multi-agent-systems
- long-context
---

<!-- buttondown-editor-mode: plaintext -->**垂直 SaaS Agent 正是您所需要的。**

> 2024/10/14-2024/10/15 的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter](https://twitter.com/i/lists/1585430245762441216) 和 **31** 个 Discord（**228** 个频道和 **1569** 条消息）。预计节省阅读时间（以 200wpm 计算）：**197 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

技术新闻方面又是平静的一天。但 Agent 融资领域正热火朝天，继 [Sierra 完成 40 亿美元的巨额融资](https://x.com/amir/status/1844192028009345526?s=46) 后不久，[Decagon 也宣布获得了 1 亿美元融资](https://x.com/thejessezhang/status/1846235369886589197?s=46)。令人瞩目的是，关于垂直 AI Agent 是未来方向的共识达成得如此之快。

https://www.youtube.com/watch?v=eBVi_sLaYsc

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 综述

> 所有综述均由 Claude 3.5 Sonnet 完成，从 4 次运行中择优。

**AI 行业发展与讨论**

- **OpenAI 校友创业投资**：[@bindureddy](https://twitter.com/bindureddy/status/1845925936119927021) 报道称，OpenAI 前 CTO Mira Murati 正在筹集 VC 资金，并从 OpenAI 挖人以创办新公司。这凸显了 AI 市场日益激烈的竞争，预计将出现超过 10 家由前 OpenAI 员工创办的初创公司。

- **AI 成就荣获诺贝尔奖**：[@demishassabis](https://twitter.com/demishassabis/status/1845864764469334239) 分享了他因 AlphaFold2 项目获得诺贝尔奖的感想，该项目解决了蛋白质结构预测这一长达 50 年的重大挑战。他强调了 AI 在科学发现中的重要性及其在开发新疗法方面的潜力。

- **AI 模型进展**： 
  - [@ClementDelangue](https://twitter.com/ClementDelangue/status/1845890229590425920) 指出，开源与闭源 LLM 之间的差距现在已经微不足道。
  - [@johnowhitaker](https://twitter.com/johnowhitaker/status/1845957479341199524) 强调了一个新模型中使用的有趣技术，包括用于权重共享的 LoRA projectors 以及在高质量数据上的退火（annealing）。

- **AI 研究与应用**： 
  - [@ylecun](https://twitter.com/ylecun/status/1845929636330721511) 讨论了高带宽感官输入对于 self-supervised learning 的重要性，认为仅靠语言不足以学习常识。
  - [@fchollet](https://twitter.com/fchollet/status/1845925019731611706) 评论了一个将 LLM 与 Lean 定理证明器相结合的项目，将其描述为“直觉引导推理（intuition-guided reasoning）”，是深度学习引导的离散程序搜索的一个很好的例子。

- **AI 基础设施**：[@nearcyan](https://twitter.com/nearcyan/status/1845887854054199730) 分享了一张显示前沿模型（frontier models）所需数据中心规模的图片，说明了尖端 AI 研究对计算能力的巨大需求。

- **AI 工具与框架**： 
  - [@rasbt](https://twitter.com/rasbt/status/1845850007095660796) 分享了一个 Jupyter notebook，其中包含在 PyTorch 中加载 LLM 等大型模型时减少内存使用的技巧。
  - [@jerryjliu0](https://twitter.com/jerryjliu0/status/1845907081725096329) 描述了一个用于报告生成和表单填写的 multi-agent 工作流，利用了 LlamaParse 和 long-context LLMs 等工具。

- **AI 伦理与挑战**：[@ajeya_cotra](https://twitter.com/ajeya_cotra/status/1845881870082331052) 对研究 AI agents 执行本应拒绝的有害任务的难易程度，以及它们在这些任务中的胜任程度表示关注。

**AI 模型性能与基准测试**

- **模型评估**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1845929321925693867) 分享了一篇论文的信息，该论文展示了即使是始终输出恒定响应的“空模型（null model）”也可以在自动基准测试中作弊，并获得排名靠前的胜率。

- **线性化 LLM**：[@togethercompute](https://twitter.com/togethercompute/status/1845928393877197287) 宣布了 LoLCATs，这是一种将 Llama 和 Mistral 等现有 Transformers 转换为最先进的 subquadratic 变体的新方法，有可能降低计算成本。

- **AI 优化**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1845877069214814525) 讨论了 LPZero，这是一个在 Neural Architecture Search 中自动设计 Zero-cost proxy 的框架，可以提高评估语言模型架构的效率。

**AI 行业趋势与观点**

- **AI 领域的竞争**：[@adcock_brett](https://twitter.com/adcock_brett/status/1845919277481971789) 批评了 AI 领域是一个拥有众多赢家的大市场的观点，强调了竞争力的重要性。

- **开源 vs. 闭源**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1845890229590425920) 表示开源和闭源 LLM 之间的差距现在已经微不足道，这表明 AI 开发领域的竞争环境正在趋于平稳。

- **AI 研究文化**：[@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1845885562462654967) 评论了现代深度学习中的经验主义文化，指出了这种方法的利弊。

**迷因与幽默**

- [@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1845851259091193979) 分享了一个关于赢得分布式 Metaverse 与 Roblox 采用率赌注的轶事，强调了技术采用的不可预测性。

- [@DavidSHolz](https://twitter.com/DavidSHolz/status/1845885464311746669) 诗意地描述了 SpaceX 的火箭回收不仅是工程上的胜利，更是一场文化与精神上的胜利，激发了对科学和客观真理的深层渴望。


---

# AI Reddit 综述

## /r/LocalLlama 综述

**主题 1. 小型语言模型的进步：Llama 3.2 1B 性能**

- **[Llama3.2:1B](https://v.redd.it/gr7phat4ypud1)** ([分数: 116, 评论: 40](https://reddit.com//r/LocalLLaMA/comments/1g3f0qn/llama321b/)): 该帖子将 **Llama3.2:1B** 与更大的模型进行了对比，指出它在配备 **CPU** 和 **8GB RAM** 的系统上进行 **代码生成 (code generation)** 和 **一次性请求 (one-time requests)** 时非常有效。虽然它在这些任务中表现良好，但该模型的性能在 **长对话中会下降**，而 **3B** 版本尽管速度较慢，但能更有效地处理延长的聊天历史。
  - 帖子强调了自 **ChatGPT 发布** 以来 **AI** 的飞速进展，现在的 **1B 模型** 已经能提供质量相当的答案。一些用户对“**大众化的 AI (AI for the masses)**”表示兴奋，而另一些用户则报告了小模型的问题，如幻觉增加和无关响应。
  - 该帖子的 **UI** 获得了大量赞誉，多条评论称其“酷炫”且“疯狂”。作者提到这是 [AI 设备项目](https://persys.ai) 的一部分，并正在考虑增加代码执行功能。
  - 一位用户提出了将 **LLM** “**硬件化 (crystalizes)**”的想法，建议使用专用硬件来提升本地 **LLM** 应用的性能。帖子作者做出了回应，表示计划在未来版本中使用专为轻量级用途设计的专用模型和板卡。


**主题 2. AI 生成的游戏环境：现状局限与未来潜力**

- **[在单块 RTX 3090 上实时运行 AI 生成的 CS:GO](https://youtu.be/6Md5U8rMZjI)** ([分数: 116, 评论: 49](https://reddit.com//r/LocalLLaMA/comments/1g3djqv/playing_aigenerated_csgo_on_a_single_rtx_3090_in/)): 一组研究人员开发了一个 **AI 生成版本的《反恐精英：全球攻势》(CS:GO)**，它可以在 **单块 RTX 3090 GPU 上实时运行**。该系统使用 **视觉语言模型 (vision-language model)** 来解释游戏状态并生成相应的动作，实现了 **4 FPS 的帧率**，展示了 AI 自主创建和游玩复杂视频游戏的潜力。
  - 用户讨论了潜在的改进方向，建议开发一种具有 **AI 生成纹理** 和 **3D 对象** 的 **模块化游戏**，在保持对游戏机制控制的同时，允许 **持久化状态 (persistent states)** 和玩家的共同贡献。
  - 一些人将该技术与 **AI 生成的 Doom 游戏画面** 进行了比较，并推测了未来的应用，例如使用行车记录仪画面结合加速和转向输入进行 **现实生活驾驶模拟**。
  - 关于该项目的实用性引发了争论，一些人称赞它是“**不可思议的体验**”，而另一些人则认为它距离实用还“**相差万里**”，并预测在 **2-3 年** 内会有重大突破。


**主题 3. 本地运行大语言模型的硬件需求**

- **在家运行 90B llama 的硬件成本？** ([分数: 55, 评论: 80](https://reddit.com//r/LocalLLaMA/comments/1g3dtyy/hardware_costs_to_run_90b_llama_at_home/)): 该帖子询问了在家中运行 **90B 参数** 版本的 **Llama 语言模型** 以进行 **离线文本生成** 的 **硬件成本**。用户明确表示 **速度** 不是关键因素，也不需要视觉或微调等额外功能，虽然承认这套配置可能负担不起，但仍表示有兴趣探索其可能性。
  - **Llama 3.1 70B** 和 **Llama 3.2 90B** 具有相同的文本模型，90B 版本增加了视觉能力。用户可以在多种配置上运行 **70B 模型**，包括用于 CPU 推理的 **64GB RAM**、实现 6-7 tokens/s 的双 **P40 GPU**，或用于更快处理速度的双 **3090/4090 GPU**。
  - 硬件选择范围从入门到高端：**单块 3090 GPU** 配置（约 2,000 美元）可以胜任 70B 模型；**双 3090 GPU**（约 3,000 美元）可以处理 70B 和 90B 模型；**双 5090 GPU**（约 6,000 美元）可提供流畅的性能。配备 64GB RAM 的 **Apple Mac Studio M2 Max** 运行 70B 模型速度约为 7 tokens/s。
  - 其他选项包括使用配备 8 通道 DDR4 内存的 **AMD EPYC 7002** 服务器，能够以 2 tokens/s 运行 **Llama 70B Q8**，甚至在双 CPU 和 512GB RAM 下以 0.6 tokens/s 运行 **Llama 405B Q8**。一些用户建议使用 **AMD MI60 GPU**。


**主题 4. 在开源模型中重现类 GPT 的思考过程**

- **重现 GPT o1 CoT 思维（思考与输出）** ([Score: 34, Comments: 13](https://reddit.com//r/LocalLLaMA/comments/1g3y432/recreating_gpt_o1_cot_thinking_thinking_and/))：该帖子讨论了为 **OpenWebUI** 开发的 **Thinking and Outputting tag** 功能，试图复制 **GPT-o1** 的行为。作者通过在模型文件中微调指令实现了这一点，要求模型支持 `## Thinking` 标签并以 "\*\*\*" 退出“思考”模式，通过视频演示了该功能并提供了一个 [下载链接](https://openwebui.com/f/yuchen4645/Think_And_Generate) 供他人尝试。
  - **cddelgado** 假设 **GPT-o1** 使用了一个复杂的推理系统，涉及 **Chain of Thought**、**Tree of Thought** 以及用于规划和批判的 **Adversarial Agents**。他们建议使用较小的 LLM 通过多次对话来实现这一点，其中一个作为主要工作者，另一个作为对手。
  - **kristaller486** 澄清说，该帖子的实现并不是 **GPT-o1**，而是 **Chain of Thought (CoT)**，并指出 o1 是一个**基于 RL 的推理系统**，而不仅仅是一个 Prompt/Agent/微调模型。他们提供了一个 [链接](https://www.reddit.com/r/LocalLLaMA/comments/1fxof45/its_not_o1_its_just_cot/) 以获取更多信息。
  - **asankhs** 建议尝试 [OptILLM GitHub 仓库](https://github.com/codelion/optillm) 中的 **cot_reflection** 方法，以在响应中生成思考和反思 Token，为实现类似功能提供了另一种方法。

## 其他 AI Subreddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 研究与技术**

- **Google Deepmind 推进 multimodal learning**：一篇来自 [Google Deepmind 的论文](https://arxiv.org/html/2406.17711v1) 展示了如何通过联合样本选择（joint example selection）进行 data curation，从而进一步加速 multimodal learning。(/r/MachineLearning)

- **Microsoft 的 MInference 加速 long-context 任务推理**：[Microsoft 的 MInference 技术](https://arxiv.org/abs/2407.02490) 能够实现针对 long-context 任务高达数百万个 tokens 的 inference，同时保持准确性，显著提升了支持模型的运行速度。(/r/MachineLearning)

- **利用 10 亿个网页策展的 personas 扩展 synthetic data 生成**：一篇关于 [扩展 synthetic data 生成的论文](https://www.reddit.com/r/MachineLearning/comments/1dzergu/r_scaling_synthetic_data_creation_with_personas/) 利用 LLM 中的多样化视角，从网页数据策展出的 10 亿个 personas 中生成数据。(/r/MachineLearning)

**AI 模型发布与改进**

- **Salesforce 的“小巨人” xLAM-1b 模型在 function calling 方面超越 GPT 3.5**：Salesforce 发布了 xLAM-1b，这是一个拥有 10 亿参数的模型，在 [function calling 中实现了 70% 的准确率，超越了 GPT 3.5](https://www.reddit.com/r/LocalLLaMA/comments/1dz8g10/salesforce_tiny_giant_xlam1b_model_surpasses_gpt/)。(/r/LocalLLaMA)

- **具备 function calling 能力的 Phi-3 Mini (6月版)**：Rubra AI 在 6 月发布了更新后的 Phi-3 Mini 模型，[具备 function calling 能力](https://www.reddit.com/r/LocalLLaMA/comments/1dzhe38/phi3_mini_june_with_function_calling/)。它与 Mistral-7b v3 具有竞争力，且表现优于基础版 Phi-3 Mini。(/r/LocalLLaMA)

**AI 应用与演示**

- **AI 增强的图像上采样揭示历史异常**：一篇文章展示了 [先进的图像上采样技术如何揭示历史图像中隐藏的细节](https://www.reddit.com/r/StableDiffusion/comments/1g3odok/revealing_hidden_historical_anomalies_through/)，这可能会挑战已有的定论。(/r/StableDiffusion)

- **AI 生成的太空港景观**：一张展示 [太空港未来主义酒店房间视野](https://www.reddit.com/r/StableDiffusion/comments/1g3ms21/my_hotel_room_has_the_best_view_of_the_space_port/) 的图片，展示了 AI 图像生成的创意潜力。(/r/StableDiffusion)

- **Adobe Firefly Video**：Adobe 推出了 [Firefly Video，被描述为“首个商业安全的视频生成模型”](https://www.reddit.com/r/singularity/comments/1g3mwke/adobe_firefly_video_is_the_first_commercially/)，支持 text-to-video 和 image-to-video 生成，并专注于 prompt coherence。(/r/singularity)

**AI 在战争与国防中的应用**

- **AI 提高乌克兰无人机效能**：一份报告称 [AI 已将乌克兰无人机的击杀率提高到 80%](https://www.reddit.com/r/singularity/comments/1g3y4iw/artificial_intelligence_raises_ukrainian_drone/)，突显了 AI 在现代战争中日益增长的作用。(/r/singularity)

**AI 的哲学与社会影响**

- **质疑人类的推理能力**：一篇文章询问是否有人写过关于 [“人类真的能推理，还是仅仅是 stochastic parrots？”](https://www.reddit.com/r/singularity/comments/1g3kbtu/has_anybody_written_a_paper_on_can_humans/) 的论文，暗示人类在推理测试中可能会以类似于 LLM 的方式失败。(/r/singularity)

- **对 AI 驱动的社会快速变革的预测**：多篇文章讨论了 [AI 进步带来的快速、变革性改变的潜力](https://www.reddit.com/r/singularity/comments/1g3fhuk/the_vast_majority_have_absolutely_no_idea_what_is/)，一些人预测会发生重大的社会动荡，另一些人则对 AI 的发展给出了更具推测性的时间表。(/r/singularity)

**AI 生成艺术与媒体**

- **融合现实世界与动漫美学**：一篇文章展示了 [能够无缝融合写实与动漫风格元素的 AI 生成图像](https://www.reddit.com/r/StableDiffusion/comments/1g3b91y/make_some_my_lora_between_real_world_and_anime/)，展示了先进的 style transfer 能力。(/r/StableDiffusion)


---

# AI Discord 摘要回顾

> 由 O1-preview 生成的摘要之摘要的摘要

**主题 1：梯度累积 Bug 修复震撼 AI 训练**

- [**Eleuther 修复梯度累积 Bug，稳定训练**](http://unsloth.ai/blog/gradient)：Eleuther 发布了一个针对在大梯度累积步数下导致训练损失发散的 Bug 修复，该问题与交叉熵损失（cross entropy loss）归一化有关。敦促用户更新其库以从该改进中受益。
- [**Unsloth AI 通过梯度修复将准确度提升 10 倍**](https://x.com/UnslothAI/status/1846231235749990699)：Unsloth AI 宣布修复了一个梯度累积 Bug，使训练损失的准确度提升了 **10 倍**以上。新的 Notebook 展示了该修复的影响，建议用户更新 Unsloth。
- [**Nous Research 庆祝来自 Unsloth AI 的梯度修复**](https://x.com/danielhanchen/status/1846235913443262891)：Nous Research 社区讨论了梯度累积修复，强调了其在提高不同设置下训练一致性和增强模型可靠性方面的重要性。

**主题 2：SageAttention 加速推理，工程师们感到兴奋**

- [**SageAttention 承诺模型推理提速 2.7 倍**](https://arxiv.org/abs/2410.02367)：论文 *SageAttention* 介绍了一种量化方法，在保持准确性的同时，将每秒操作数比 FlashAttention2 和 xformers 提升了 **2.1 倍**至 **2.7 倍**。研究人员对 Transformer 模型的潜在效率提升充满期待。
- [**使用 SageAttention 进行训练遇到障碍**](https://arxiv.org/abs/2410.02367)：尝试将 SageAttention 用于训练导致了发散问题，这强调了它目前是为推理加速而设计的。讨论揭示了将其适配到预期用途之外的挑战。
- [**LM Studio 关注 SageAttention 以实现性能飞跃**](https://arxiv.org/abs/2410.02367)：社区成员指出，将 SageAttention 集成到 **llama.cpp** 和 **MLX** 等工具中可能会使 Token 处理速度翻倍。如果得以实现，这将标志着 Transformer 模型性能的重大飞跃。

**主题 3：AI 模型组件备受质疑——QKNorm 和 ReLU² 遭到审查**

- **QKNorm 在大型模型中遇冷**：测试显示 **QKNorm** 在严格的基准线下表现不佳，导致大型模型中出现“弱注意力（weak attention）”，引发了对其设计优点的怀疑。
- **ReLU² 仅 4% 的增益未能打动工程师**：**ReLU²** 相比 GELU 等函数仅提供 **4%** 的改进，使其在扩展大型模型时的实用性存疑，并引发了关于激活函数功效的辩论。
- **研究人员指出误导性的性能主张**：参与者注意到，某些声称的性能改进可能掩盖了不稳定性问题，而非代表真正的进步，呼吁对这类主张进行批判性评估。

**主题 4：人才流动与争议震撼 AI 行业**

- [**微软 AI 明星 Sebastien Bubeck 加入 OpenAI**](https://www.theinformation.com/briefings/microsoft-ai-researcher-sebastien-bubeck-to-join-openai?rc=c48ukx)：Sebastien Bubeck 从微软跳槽至 OpenAI 在 AI 社区引起波澜。讨论集中在人才动态以及对 AI 研究方向的潜在影响上。
- **围绕 Bubeck 的《Sparks of AGI》论文爆发争议**：社区成员对 Bubeck 的 *Sparks of AGI* 论文褒贬不一，批评集中在其夸张的定位，并对其定义 AGI 的影响提出质疑。

**主题 5：LLM 的推理能力受到质疑**

- [**苹果的研究揭示了 LLM 逻辑推理的破绽**](https://arstechnica.com/ai/2024/10/llms-cant-perform-genuine-logical-reasoning-apple-researchers-suggest/)：苹果的一项研究显示，LLM 依赖于概率模式匹配，当基准测试发生变化时会导致逻辑推理错误。工程师们讨论了人类对比基准的必要性以及“推理”的精确定义。
- **OpenAI 社区辩论 LLM 的推理局限性**：成员们强调 LLM 在真正的逻辑推导方面表现挣扎，在需要真实推理的任务中会导致“灾难性”失败。该研究促使人们重新评估 AI 模型中推理的定义和评估方式。


---

# 第一部分：高层级 Discord 摘要

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **梯度累积 Bug 修复发布**：针对导致大梯度累积步数下训练损失发散的 Bug，现已发布修复补丁，该问题直接与 Cross Entropy Loss 归一化相关。建议用户阅读此 [博客文章](http://unsloth.ai/blog/gradient) 并更新库。
   - 多个成员提出了这一问题，强调了对齐归一化策略以确保训练损失曲线稳定的重要性。
- **QKNorm 的有效性受到质疑**：测试显示 **QKNorm** 在严格的基准测试下表现不佳，导致大型模型中出现“弱注意力（weak attention）”，引发了对其设计的怀疑。有趣的是，它在 Olmoe 项目中的应用表明人们对其潜力的看法不一。
   - 参与者指出需要进一步研究其对大型架构的影响，特别是在 Attention 机制变得至关重要的情况下。
- **ReLU^2 的收益存疑**：与 GELU 等竞争对手相比，**ReLU^2** 仅带来了 **4%** 的微小提升，这让人怀疑其在 Scaling 中的实际效用。这种细致的性能分析引发了关于大型模型中所用激活函数的广泛讨论。
   - 性能对比促使工程师在采用新的激活方法之前，同时考虑微小的增强和计算效率。
- **微调库正在接受审查**：由于成员们正在寻求改进评估方法，现有微调库的局限性引起了关注，例如 **torchtune** 中缺乏非聊天模板（non-chat-template）结构。社区渴望能够简化微调过程且无需复杂模板的库。
   - 讨论强调了 **QuestionAnswerTemplate** 作为模型评估的可行替代方案的可用性，确保了更清晰的指标。
- **误导性的性能提升受到审视**：参与者注意到，所谓的性能提升声明往往掩盖了不稳定性问题，而非反映真正的进步；A/B Testing 被认为是一个常见的陷阱。缺乏坚实基准测试的论文通常被认为价值较低，除非它们揭示了显著的性能转变。
   - 这种做法稀释了研究结果的质量，因此研究人员批判性地评估报告性能提升的条件至关重要。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **梯度累积修复提升训练效果**：Unsloth 修复了一个导致梯度累积中训练损失发散的 Bug，将准确率提升了 **10 倍**以上。用户应更新 Unsloth 并查看演示其影响的新 Notebook。
   - Unsloth AI 的一条 [推文](https://x.com/UnslothAI/status/1846231235749990699) 强调了这一修复，提到训练指标有了显著改善。
- **INTELLECT-1 去中心化模型发布**：Prime Intellect 推出了 **INTELLECT-1**，这是一个拥有 100 亿参数的协作式去中心化训练模型。该倡议旨在通过允许社区贡献来促进开源 AGI。
   - 更多细节可以在他们的 [博客文章](https://www.primeintellect.ai/blog/intellect-1) 中找到，讨论了该模型如何使分布式 AI 训练受益。
- **SageAttention 承诺更快的模型推理**：论文 *SageAttention* 揭示了一种量化方法，与 FlashAttention2 和 xformers 相比，其每秒操作数分别提高了 **2.1** 和 **2.7 倍**。该方法在各种模型中保持了准确性。
   - 然而，尝试将 SageAttention 用于训练时出现了发散问题，强调了其侧重于推理而非训练的可行性。
- **探索 LLM 微调流程**：讨论围绕 **Llama** 等 LLM 的微调工作流展开，强调了数据格式对输出质量的影响。重点在于探索多样化的 LLM 输出。
   - 参与者考虑了有效的格式化和高效的数据管理将如何增强模型性能。
- **模型性能的对比分析**：围绕 **Qwen** 和 **Llama** 等模型的性能展开了热烈辩论，重点关注它们在微调和数据集利用方面的适用性。质量胜过数量是一个共同的主题。
   - 讨论集中在特定数据集如何产生更好的微调结果，同时讨论了与 Deepspeed 等工具的集成以提升能力。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 的推理功能面临不一致性**：用户注意到在 ProSearch 中**触发新的推理功能**似乎是随机的，并随问题的复杂程度而变化，导致分析结果不一致。
   - 他们观察到之前的推理模型更可靠，而新模型在生成信息过程中出现幻觉的情况有所增加。
- **ProSearch App 令人沮丧的延迟**：许多用户对 **ProSearch Mac 应用的延迟**表示恼火，该应用最初预计在更早的日期发布。
   - 其他投诉包括**线程丢失**问题以及应用程序整体运行缓慢。
- **Adobe 的 AI 视频模型增强**：Perplexity AI 强调 **Adobe 的 AI 视频模型**是视频编辑领域的一项变革性发展，承诺提供改善工作流的高级功能。
   - 这一创新预计将显著提高内容创作的速度和可访问性。
- **NASA 成功发射 Europa Clipper**：**NASA Europa Clipper 任务**已成功发射，旨在调查木星卫星木卫二（Europa）上潜在的生命迹象。
   - 专家们热切期待可能揭示该卫星地下海洋新见解的发现。
- **中国研究人员破解 RSA 加密**：最近的报告显示，**中国研究人员已成功破解 RSA 加密**，这在网络安全社区引起了重大关注。
   - 这一进展引发了关于当前敏感数据加密实践中漏洞的深入讨论。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 的多实例 LLM 派对**：用户讨论了运行多个 **Aider** 实例来处理大任务和小任务的可行性，建议只要不修改相同的文件就应该可行。
   - 一位用户幽默地将其称为“LLM 派对”，强调了同时进行 LLM 操作的趣味潜力。
- **API Key 验证难题**：成员报告了在尝试为 **Aider** 配置 **Gemini** 模型时出现 **API 验证错误**，特别是在 `.env` 文件中设置 Key 之后。
   - 一位用户确认该 Key 通过命令行可以工作，表明问题可能与他们的脚本设置有关。
- **高效命令使用的脚本策略**：讨论了如何使用 Python 和命令行有效地编写 **Aider** 命令脚本，强调了正确加载环境的必要性。
   - 一位用户讲述了修改示例脚本以实现 Gemini 模型的过程，但遇到了与环境变量相关的错误。
- **模型对比：Aider vs Sonnet-3.5**：用户注意到 **Sonnet-3.5** 在非 Web 开发任务中的表现优于 **Gemini** 等其他模型，使其成为首选。
   - 一位用户强调，在测试各种编程任务模型时，Sonnet-3.5 始终能提供更优的结果。
- **Gemini 集成和配置挑战**：关于在 **Aider** 中正确配置 **Gemini-1.5 Pro 模型**的咨询，重点在于 API Key 的设置。
   - 尽管参考了文档，用户仍因环境配置错误而面临 **API 错误**。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HuggingFace 账号恢复的紧迫性**：一名用户紧急寻求帮助以恢复其被黑客攻击并删除的 HuggingFace 账号，被建议发送邮件至 [website@huggingface.co](mailto:website@huggingface.co) 获取支持。
   - 恢复可能需要几天时间，但成员们鼓励在等待回复时保持耐心。
- **AI 自动化引发对就业保障的担忧**：成员们讨论了对 AI 可能自动化 Data Science 和 ML 工作的焦虑，强调希望能向更具创造性的角色转变。
   - 讨论中将其与过去同样改变了就业结构的各种技术进步进行了对比。
- **Llama 3.2 模型推理速度的辩论**：在 A100 GPU 上使用 Llama 3.2 1B 模型对大型数据集进行推理耗时超过 **14 小时**，引发了关于效率提升的讨论。
   - 成员们分享了他们的模型加载和推理策略以优化性能。
- **令人兴奋的 Flutter 开发协作**：一名成员宣布自己作为 **Flutter 开发者**可参与 **AI 应用**的协作，邀请他人加入。
   - 这一呼吁强调了在开发以 AI 为核心的项目时，对合作伙伴关系日益增长的需求。
- **Gradio 5 在 Product Hunt 上引起轰动**：**Gradio 5** 在 [Product Hunt](https://www.producthunt.com/posts/gradio-5-0) 上发布，并请求社区支持。
   - 团队成员鼓励用户体验新功能并提供反馈，以提高知名度。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Hermes 3 Llama 3.1 405B 变为订阅模式**：**Hermes 3 Llama 3.1 405B Instruct** 模型现在以 **$1.79/月** 的价格提供，免费版本可在 [OpenRouter](https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b) 访问。
   - *不要错过这一为强大 AI 功能更新的定价结构！*
- **Nous Hermes Yi 34B 已弃用**：**Nous Hermes Yi 34B** 模型已被所有服务提供商弃用，不再提供使用。
   - *鉴于此弃用，鼓励用户过渡到替代模型。*
- **AI 模型排名亮点**：用户讨论了各种 AI 模型的性能，**Llama-3-8b-Instruct** 和 **GPT-4o** 因能有效遵循指令而受到关注。
   - *Grok 2 mini* 和 *Gemini 1.5 Pro* 也被认为是体面的替代方案，而 *Opus* 因其一些怪癖受到了一些批评。
- **创新的 Chatbot 设计技术**：一位用户提议创建一个隐藏的 AI Chatbot，以避免对侮辱产生通用的拒绝消息，建议使用另一个 LLM 进行过滤。
   - 参与者强调了像 *Llama Guard* 这样的模型，为管理响应提供额外支持。
- **Infermatic 提供商报告的问题**：一位用户报告了 **Infermatic** 提供商的问题，因为他们的聊天开始意外地产生无关的响应。
   - 这提醒了社区最近出现的潜在服务中断。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Research 社区的起源**：Nous Research 社区始于 Discord，并演变为一家专注于 **AI research** 和协作的受资助科技公司。
   - 成员们积极分享想法并研究各种 AI 模型和技术，增强了参与度和项目成果。
- **Gradient Accumulation Bug 修复发布**：**UnslothAI 团队** 解决了 **Gradient Accumulation** 中导致训练损失发散的一个重大 Bug，提高了整体一致性。
   - 此修复现已提供给用户，简化了训练过程并增强了模型可靠性。
- **探索 Zamba2-7B 模型性能**：Zyphra 宣布推出 **Zamba2-7B 模型**，声称其在消费级 GPU 上的性能和质量超过了 Llama3 和 Mistral。
   - 有关功能的详细信息在最近的一篇 [blog post](https://www.zyphra.com/post/zamba2-7b) 中列出，提供了其部署的见解。
- **Synthetic Data 导致 Model Collapse**：[研究](https://arxiv.org/abs/2410.04840)表明，训练集中即使只有 **1%** 的 **Synthetic Data** 也会导致显著的 **Model Collapse**，影响大型模型的性能。
   - 这强调了训练像 ChatGPT 这样的大型模型所涉及的风险，表明目前的做法可能需要重新评估。
- **SageAttention 方法的效率**：[SageAttention](https://arxiv.org/abs/2410.02367) 引入了一种 **Quantization** 方法，提高了 **Attention** 机制的效率，性能优于 **FlashAttention2** **2.1 到 2.7 倍**。
   - 该方法在显著降低计算复杂度的同时确保了高精度，对于推理加速至关重要。



---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Lux-AI Challenge 邀请协作**：鼓励成员为 [Lux-AI Challenge 的 GitHub 仓库](https://github.com/Lux-AI-Challenge/Lux-Design-S3)贡献力量，以促进团队协作。
   - 有人呼吁有兴趣的人士组队参加 Lux-AI 项目，展示了社区参与挑战赛的积极性。
- **Triton 在 Jetson 构建中遇到困难**：用户报告了在 **Jetson Orin AGX 64GB** 上构建 **triton-lang** 时的问题，其中 CUDA 将 Unified Memory 误认为 `AMD GPU`。目前正在重新构建，希望 LLVM 支持是问题所在。
   - 讨论显示，用户应在相关的 [issues](https://github.com/triton-lang/triton/issues?q=sort%3Aupdated-desc+is%3Aissue+jetson+is%3Aclosed) 中检查 **LLVM** 对 ARM 的支持。
- **《Learn PyTorch for Deep Learning》课程现已上线**：一门新课程 [Learn PyTorch for Deep Learning: Zero to Mastery](https://www.learnpytorch.io/) 被分享为掌握 PyTorch 基础知识的顶级资源。
   - 该课程形式结合了视频见解和易于访问的在线书籍，提供了一种结构化的学习方法。
- **Ollama 在树莓派上的性能表现**：**Ollama** 模型在树莓派 5 上运行 **llama3.2** 版本的速度为 **5.32 tokens/s**，而 **llama3.1** 的速度仅为 **1.5 tokens/s**。
   - 讨论涉及了 **eGPU 与 2080** 的集成，表明了树莓派系统一条可行的升级路径。
- **WebGPU 缺乏 CUDA 交互**：澄清了 **WebGPU** 不与 **CUDA** 交互，这意味着开发者未来必须依赖其他 API。
   - 此外，WebGPU 的运行取决于操作系统定义的特定 **graphics APIs**，如 **Vulkan** 和 **DirectX**。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **实时 STT 引擎树立新标准**：Gladia 新的实时 STT 引擎拥有 **< 300 ms 的延迟**，支持超过 **100 种语言**和语码转换（code-switching），并获得了 1600 万美元的 A 轮融资。另一家竞争对手的引擎声称具有 **90ms 的推理时间**并支持多语言，加剧了转录技术的竞争。
   - 正如成员们所讨论的，这种改进使这些引擎成为实时通信中各种应用的可行选择。
- **线性注意力模型有望提升效率**：在 Llama 3.1 家族中实现 **linear attention models** 显示出显著提升效率的潜力，使其资源消耗更低。对话揭示了尝试将 **超过 50% 的 transformer attention layers** 转换为线性版本时面临的挑战。
   - 参与者似乎对这一转变充满希望，强调这符合当前机器学习中的资源优化趋势。
- **AI 作为新型建筑材料**：一篇博客文章将 AI 融入各行各业比作历史上由**塑料**引起的变革，将 AI 定位为现代设计的革命性材料。讨论集中在*以往的材料时代如何重新定义生产和建筑*。
   - 参与者对 AI 日益增长的作用表示兴奋，呼应了软件现在比物理材料更关键的观点。
- **融资公告引发好奇**：DecagonAI 获得的 6500 万美元 B 轮融资激起了人们对 AI 初创公司投资趋势的兴趣，特别是应用层而非核心模型。知名投资者包括 **Bain Capital Ventures 和 Accel**，凸显了 AI 解决方案市场的强劲。
   - 成员们指出，此类融资努力反映了重点向实际 AI 应用的转移，揭示了当前的市场动态。
- **关于外包文档的辩论**：关于为 AI 和开源项目外包文档的可能性展开了激烈的讨论，权衡了使用 LLM 与人类作者的优缺点。社区成员思考了这将如何影响质量和可访问性。
   - 谈话提出了关于成本效益与详尽文档之间平衡的问题，表明这是项目管理中的一个重要考虑因素。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Llama 3.1-70B 集成面临截断困扰**：**Llama 3.1-70B** 的集成正在返回截断的响应，在请求 **20 个软件工程技能**列表时，始终仅提供 **5 个技能**，原因是触及了 `max_tokens` 限制。
   - *一位用户指出，“尽管调整了参数，响应仍以 finish_reason: max_tokens 结束”*。
- **Qdrant 节点添加触发错误**：一名成员在向 **Qdrant** 索引添加新节点时遇到错误，此前未见此类问题的报告，表明可能存在设置冲突。
   - 另一位用户建议，他们自己成功添加节点的经历暗示第一位用户的设置中可能存在配置错误。
- **使用 Claude 3.5 构建 Financial Agent**：你可以使用 [@financial_mod](https://twitter.com/llama_index/status/1845980793593831845) 分享的股票价格和公司数据 API，创建一个由 **Claude 3.5 Sonnet** 驱动的 **Financial Agent**。
   - 根据 Hanane Dupouy 的说法，该 Agent 提供了多样的见解，包括损益表和全面的公司信息。
- **PineconeVectorStore 在 ComposableMemory 中失效**：成员们对 `SimpleComposableMemory` 中的 **PineconeVectorStore** 表示沮丧，收到了 “Namespace not found” 错误消息。
   - 另一位用户推测，设置问题可能是导致这些持续错误的原因。
- **Neo4jPropertyGraphStore 初始化性能滞后**：据报告，**Neo4jPropertyGraphStore** 的初始化存在显著延迟，在较大的图上，Schema 生成耗时过长。
   - 不使用 `async` 操作可能会加剧此问题，相关的 [GitHub issue](https://github.com/run-llama/llama_index/issues/16204) 也证实了这一点。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **LLM 在推理方面表现出破绽**：最近的一项 [Apple 研究](https://arstechnica.com/ai/2024/10/llms-cant-perform-genuine-logical-reasoning-apple-researchers-suggest/) 表明，**LLM** 在数学推理中利用概率模式匹配，当基准测试发生变化时会导致错误。
   - 成员们表示有必要进行**基准人类对比**，并强调了该研究中推理定义的模糊性。
- **Swarm 库需要更好的测试**：研究 **Swarm** 库的用户发现，很难区分任务是由 Agent 执行还是由基础 LLM 执行，强调了对稳健测试的需求。
   - 对 Swarm 的**非生产就绪 (non-production-ready)** 状态表示担忧，并提到了 **Swarm.js** 等替代方案。
- **对 GPT 语音功能的困惑**：关于高级 **GPT voice** 功能推出的讨论不断，但 OpenAI 尚未对其功能发布明确公告。
   - 由于过去版本未受支持，对潜在更新的怀疑在增加。
- **自定义 GPT 更新问题**：一位成员的自定义 GPT 由 **300 页**材料构建，在将 PDF 拆分为**六个较小文件**后，仍处于 “Update Pendings” 状态超过一周。
   - 尽管 PDF 已被识别，但该 Bot 经常将查询重定向回代码，而不是直接从文档中回答。
- **PDF 处理故障**：另一名成员在 GPT-4 中测试 **1 个 PDF** 时遇到性能问题，表明 PDF 内容处理存在更深层次的问题，影响了响应速度。
   - 这表明 GPT 与 PDF 输入的交互方式可能存在系统性挑战。



---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 配置选项需要更清晰**：一位成员建议以截图以外的格式分享**配置详情**，并指出未来的博客文章将包含这些改进。
   - 这一建议旨在增强可用性，使用户更容易理解设置和优化。
- **M2 Studio 在大模型上表现出色**：用户称赞配备 **192 GB RAM** 的 **M2 Studio** 在运行 **Mistral 的 128K context** 大模型时表现惊人，是特定应用的理想选择。
   - “它非常契合我的使用场景”强调了其价值，可能会吸引更多用户选择高 RAM 配置。
- **通过调整 GPU 提升性能**：一位用户建议使用 **Afterburner** 对 GPU 进行**降压 (UV)**，并表示即使是 **100mV** 的调整也能显著提升性能。
   - 他们敦促同行查看 **YouTube** 上的针对性教程，以便在不同配置下进行更好的性能调优。
- **Llama 8B 出色的 TPS 表现**：一些用户报告在各种 GPU 上使用 **Llama 8B** 达到了 **30 TPS**，而对 **150+ TPS** 的预期引发了关于必要升级的讨论。
   - 模型大小和**量化 (quantization)** 等因素会显著影响性能，尤其是在对比配备先进 **Tensor Cores** 的设备与旧款 GPU 时。
- **SageAttention 承诺带来效率提升**：最近关于 **SageAttention** 的论文强调了注意力机制在效率上的显著改进，这对 **llama.cpp** 和 **MLX** 等工具有着重要意义。
   - 如果得以实现，它可能会使 **Token 处理速度翻倍**，标志着 Transformer 模型性能的飞跃。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Connector 误解输入**：用户报告称，即使是简单的“hi”，**Cohere Connector** 也会触发搜索，从而引发了关于限制不必要交互的控制功能的咨询。
   - *有没有办法优化它的功能？* 社区正在积极寻求解决方案来优化这一点。
- **API Token 限制引发关注**：关于 **Cohere API** 的 Token 限制存在差异，有人指出每月上限为 **10k**，而聊天中提到了 **500 万个 Token**，这导致了关于潜在超额费用的疑问。
   - *超过 10k 上限会产生费用吗？* 成员们正在寻求关于这一关键点的明确答复。
- **Google Connector 表现不佳**：多位用户面临 **Google Connector** 无法正常运行的问题，引发了用户间的故障排除讨论。
   - *分享任何突破性进展！* 鼓励社区成员互相支持以解决此连接问题。
- **Command 模型定价已明确**：讨论明确了 **Web-search Connector** 不收取费用，但发送到 **Command** 输入上下文的结果会产生费用，这可能会影响用户的预算。
   - 这一区别突出了 API 使用成本的复杂性，并鼓励用户进行仔细监控。
- **OrionChat 聚合了 AI 模型**：一位成员推出了 **OrionChat**，这是一个 Web 界面，使用户能够在一个地方无缝地与来自 **Cohere**、**OpenAI** 等的各种 AI 模型进行交互，访问地址为 [此链接](https://orionchat.github.io)。
   - 该计划旨在整合对话并促进跨模型的比较，鼓励用户反馈以进行进一步改进。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **WordPress 插件开发寻求反馈**：一名成员正在为文本生成和 **txt2img** 服务器开发多个 **WordPress 插件**，急切寻求社区反馈和测试。
   - *没人回应*，这凸显了 AI Discord 服务器中社区参与度令人沮丧的现状。
- **CORS 问题困扰 Stable Diffusion 设置**：用户讨论了在反向代理设置中对 Stable Diffusion 服务器使用 SSL 时遇到的持续性 **CORS 错误**。
   - 一位技术专家成员强调，Web 服务器和 **Stable Diffusion 服务器** 需要在同一台机器上运行才能实现完整功能。
- **寻找活跃的 Discord AI 社区**：一名成员对他们的 **AI Discord 服务器** 缺乏活跃度表示失望，寻求关于 **ComfyUI** 和 **A1111** 相关更活跃社区的建议。
   - 关于插件的询问无人回答，指向了社区内对更好互动的广泛需求。
- **探索文本生成的基座模型**：一位用户询问了在风格迁移过程中能增强文本生成的基座模型，特别提到了 **i2i** 和 **SD1.5**。
   - 另一位成员建议尝试 **flux** 或 **SD3**，同时提醒 **SD3** 在人体表现方面存在困难。
- **创建风格化照片的技巧**：讨论集中在生成风格化照片的方法上，几位成员建议使用 **ControlNets**。
   - 成员们分享了创意方法，包括 [这里](https://github.com/songrise/Artist) 概述的针对各种艺术风格（如 pin-up）的技巧。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 在 .dot 操作上优于 NumPy**：详细对比显示，**Tinygrad 的 .dot 操作** 在处理大矩阵时会出现精度下降，在 M=16384, N=8192, K=1280 等维度下差异达到 **±0.001**。
   - 相反，较小的矩阵（M=10, N=4, K=5）偏差极小，不超过 **±0.000001**。
- **VIZ UI 改进成为焦点**：讨论围绕 [Issue #7067](https://github.com/tinygrad/tinygrad/issues/7067) 展开，强调了对 **VIZ UI** 备受期待的增强，特别是与自动滚动功能相关的改进。
   - 提案包括调整大小和可折叠侧边栏，旨在提升用户体验。
- **George Hotz 誓言要挑战 PyTorch 的性能**：George 认为在 NVIDIA GPU 上超越 **PyTorch** 的性能对 **Tinygrad** 来说将具有里程碑意义，标志着项目的转折点。
   - 他表示：“我们只要在性能上击败 PyTorch，我们就赢了”，强调了其中的利害关系。
- **拆解 Tinygrad 中的 TD-MPC 实现**：一位用户分享了在 Tinygrad 中成功实现 **TD-MPC 学习** 的好消息，并计划在硬件上进行测试。
   - 分享了 [GitHub 仓库](https://github.com/nicklashansen/tdmpc2/tree/main) 链接，详细说明了必要的硬件要求。
- **禁用梯度计算的方法**：用户辩论了禁用梯度的有效方法，提倡使用 `Tensor.no_grad`，同时建议将 `with Tensor.test():` 作为一种现代实践。
   - 此次对话旨在完善社区内的梯度控制方法。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **解决库安装问题**：一位用户发现缺失的库可以通过 `sudo apt-get install libtinfo-dev` 安装，帮助了遇到类似安装问题的其他人。
   - 这一发现强调了社区知识共享在有效解决常见问题中的作用。
- **应对自定义 stdlib 的挑战**：用户在运行修改版的 stdlib 时面临挑战，尽管遵循了构建指令，原始实现仍然存在。
   - 提出了一种涉及调整构建过程的变通方法来解决这些持续存在的问题。
- **寻找新的图像哈希算法**：出现了关于 pHash 等旧图像哈希算法相关性的问题，并呼吁推荐先进的替代方案。
   - 社区的探索展示了随着技术演进采用尖端技术的渴望。
- **讨论内存管理策略**：在 assertion 调用期间 struct 实例的过早销毁引发了对 Mojo 内存管理的担忧。
   - 建议包括创建一个 getter 方法来安全地访问 struct 成员，从而降低过早销毁的风险。
- **协作 Bug 报告成功**：一位用户报告了一个字符串插值问题，该问题已被确认在最新版本的 Mojo 中修复。
   - 这一案例突显了社区协作在迅速识别和解决 Bug 方面的有效性。

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Sebastien Bubeck 加入 OpenAI**：Microsoft 的明星 AI 研究员 **Sebastien Bubeck** 加入 **OpenAI** 的消息引发了轰动，引发了关于 AI 人才动态的讨论。
   - 这一举动最早由 [The Information](https://www.theinformation.com/briefings/microsoft-ai-researcher-sebastien-bubeck-to-join-openai?rc=c48ukx) 的一篇文章报道。
- **o1-turbo-mini 在 benchmarks 中表现惊人**：关于 **o1-turbo-mini** 性能的热议不断，其展示出的强劲结果甚至引起了工程师们的怀疑和调侃。
   - 社区成员注意到，对于那些对这一消息反应过度的网络群体，存在着有趣的调侃潜力。
- **AGI 末日时钟引发争议**：由沙特支持的一所瑞士商学院发起的 **Doomsday Clock**（末日时钟）声称要对“不受控制的通用智能”发出警告，但被批评为过时。
   - 创建者 **Michael Wade** 认为，将 Excel 之类的软件比作 AGI 带来的威胁是荒谬的，这反映的是历史性的恐惧而非当代的现实意义。
- **AI2 为 OLMo 项目招募研究实习生**：**AI2** 宣布了 OLMo 项目的研究实习生职位，旨在增强 Natural Language Processing 和 Machine Learning。
   - 这个位于西雅图的为期 12 周的实习提供 **$86,520** 至 **$123,600** 之间的极具竞争力的薪酬，专注于具有影响力的研究计划。
- **OpenAI 对法律领域的影响**：讨论强调了 OpenAI 在为**律师**创造有利条件方面的作用，将 AI 的进步与不断演变的法律工作联系起来。
   - 这突显了 AI 技术与法律领域实际应用之间日益增长的相互作用。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **框架选择是一场噩梦！**：成员们对 **Langchain**、**Langflow** 和 **Langgraph** 等框架之间的不断切换感到沮丧，这使得最终确定生产环境的选择变得困难。
   - 一位成员指出，他们的整个代码库已经迁移到了 *Langchain LCEL*，突显了围绕这些框架的混乱局面。
- **在私有云上部署 Langgraph**：一位成员询问如何在 **US** 或 **EU** 之外的云端部署 **Langgraph** 应用程序，寻求社区的见解。
   - 虽然没有得到直接回复，但这一询问激发了对区域性应用托管的兴趣。
- **关于 dspy 与 Langchain 的辩论**：社区对 **dspy** 是否会主导 **Langchain** 及其他框架，或者它们是否会保持相关性产生了兴趣。
   - 这反映了社区对 AI 框架未来格局的不确定性。
- **认可 Langsmith 的实用性**：一位成员建议 **Langsmith** 对于 tracing 非常有用，强调了它在不断变化的框架中的重要性。
   - 这促使了对 **Langchain Academy** 关于 **Langgraph** 课程的推荐，以磨练相关技能。
- **澄清 Langflow 的隶属关系**：一位用户澄清说 **LangFlow** 并不是 **LangChain** 的产品，解决了成员们对相关工具的困惑。
   - 这一区分可能有助于协调社区内对各种讨论框架的理解。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **LLM Agents MOOC 在线提供所有课程详情**：有关 **labs** 和 **assignments** 的所有细节都可以在课程网站 [course website](https://llmagents-learning.org/f24) 上找到，鼓励参与者关注更新。
   - 想要加入的学生应填写此 [form](https://forms.gle/svSoNhKcGFjxup989)，并通过 [LLM Agents Discord](https://discord.gg/NWVpQ9rBvd) 与社区互动以获取实时支持。
- **观察到推理时计算（Test-time compute）缩放定律**：成员们讨论了 **“推理时计算”缩放定律** 的广泛影响，将其与早期影响 GPT 系列的定律联系起来，并引用了 [这篇论文](https://arxiv.org/pdf/2408.03314) 作为支持。
   - 另一份与此讨论相关的文档也被分享了，可以在 [这里](https://arxiv.org/pdf/2001.08361) 找到。
- **《AI-Powered Search》一书成为必备资源**：一位成员推荐 [这本书](https://www.manning.com/books/ai-powered-search) 作为未来几年 AI 驱动搜索技术的关键资源，可能会影响 **从业者和研究人员**。
   - 他们预计书中的见解将成为各行业 AI 研究的基础。
- **对课程视频质量提出担忧**：一位成员指出有必要提高上传课程的 **视频质量**，并表示第 6 课最高只有 **720p**，导致难以看清代码。
   - 这一担忧表明了对课程中更易获取的学习材料的需求。
- **探索 LLM 中的推理和规划**：一位成员寻求关于 LLM 和 Agent 如何进行 **推理（reasoning）**、**规划（planning）** 以及识别工具（而非仅仅生成文本）的见解。
   - 他们表达了对后续课程涵盖 **规划** 和 **工具使用（tool use）** 的兴趣，以加深对 LLM 应用的理解。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 发布 π 版本**：一位成员宣布了 **Open Interpreter** 的新版本更新，可通过 `pip install --upgrade open-interpreter` 获取，并称其为具有显著增强功能的重大 **π 版本**。
   - [Mike Bird](https://x.com/MikeBirdTech/status/1846283357153268002) 的这条推文分享了这些改进，并引发了对其能力的关注。
- **Hume AI 令人印象深刻，Oi 登场**：一位用户讲述了 **Hume AI 模型** 如何超出预期，称其表现几乎 **太好了**，这引发了对性能阈值的审视。
   - 对话焦点转向了 **Oi 模型**，表明用户正在积极尝试各种 AI 框架。
- **Play 3.0 mini 提升文本转语音（Text-To-Speech）效果**：[Play.ht](https://x.com/play_ht/status/1845901523680686401?s=46&t=G6jp7iOBtkVuyhaYmaDb0w) 推出了 **Play 3.0 mini**，这是一款 Text-To-Speech 模型，在多种语言中提供了更高的速度和准确性，且极具成本效益。
   - 他们邀请用户在 [playground](https://play.ht/playground/?utm_source=x&utm_medium=social&utm_campaign=all_v3launch_202410) 上进行测试，并就改进提供反馈。
- **Think-on-Graph 征集合作者**：**Think-on-Graph** GitHub 仓库现已上线，邀请有兴趣在深圳合作的研究人员在 [这里](https://github.com/IDEA-FinAI/ToG) 查看。
   - 该项目公开邀请有意贡献并加入研究团队的人员通过电子邮件联系。
- **观看关于 AI 进展的视频**：一位用户分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=iGQLG0bWDxE)，涉及围绕 AI 技术展开的最新进展。
   - 细节较少，建议观众直接观看以获取所呈现内容的见解。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **对 Loom 视频见解感到好奇**：一位成员分享了一个 [Loom 视频](https://www.loom.com/share/b8c49e265d5c49aca7d2fc51c38d84c6?sid=69c5942b-ed75-4883-8867-f408a83cecf5)，可能包含与当前讨论相关的见解，尽管细节较少。
   - 该视频激起了成员们的兴趣，促使他们探索其中的价值信息。
- **上下文嵌入资源汇总**：一位成员分享了一个 [Google Colab](https://tinyurl.com/2p9wwypy) 和一段标题为“使用任何 LLM 进行上下文检索”的 [YouTube 视频](https://www.youtube.com/watch?v=6efwN_US-zk&t=7s)，重点在于实现上下文嵌入（Contextual Embeddings）。
   - 该视频旨在简化 Anthropic 的上下文检索策略在各种 LLM 上的实现。
- **RAG 机制：澄清分块过程**：成员们讨论了在不超出 Token 限制的情况下将整个文档添加到 Prompt 中的挑战，强调了 **RAG (Retrieval-Augmented Generation)** 中不可或缺的**分块过程 (Chunking Process)**。
   - 会议澄清了 RAG 利用相似度搜索仅包含最相关的分块，从而确保符合 Token 限制。
- **DSPy 集成到 GPT-O1+ 的状态检查**：一位成员询问了将 **DSPy** 集成到 **GPT-O1+** 系统中的进展，期待开发更新。
   - 然而，讨论中尚未涉及该集成的具体细节。



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **ICLR 评审终于公布了！**：期待已久的 **ICLR** 评审论文已经发布，引发了渴望深入研究的成员们的兴奋。
   - *一位成员提到*，处理分配给他们的评审需要一些时间。
- **持续预训练与指令微调的研究**：最近的一篇论文调查了大型语言模型（LLM）的**持续预训练 (Continuous Pre-training)**与**指令微调 (Instruction Fine-tuning)**之间的关系，强调了模型保持最新数据更新的必要性。
   - 这引发了一个问题：为了保持指令遵循能力，哪种模型应该进行这种预训练。
- **模型合并方法的批评**：*一位成员质疑*论文中方法的创新性，认为它类似于早已存在的模型合并（Model Merging）方法。
   - 这引发了关于所提技术相关性和原创性的讨论。



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **关于 LAION-2B 数据集与 MSCOCO 重叠的查询**：一位成员询问 **LAION-2B 数据集** 是否包含来自 **MSCOCO** (COCO2014 或 COCO2017) 的图像，质疑潜在的**数据重叠 (Data Overlap)**。
   - 该查询强调了论文中提到的**数据重叠**，并请求提供有关验证此问题所采用技术的更多细节。
- **早安与一般问候**：成员们交换了问候，一位成员说 **“大家早上好”**，营造了友好的聊天环境。
   - 另一位成员随意地以 **“gm”** 回应，增添了轻松的氛围。



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **解码推理流水线机制**：**Gorilla LLM** 中的推理流水线通过输出 `decod_exec` 可以解释的有效函数调用来执行函数；当输出为空或无法解码的响应时，表示回合结束。
   - 这种自动信号指示模型何时完成了任务，提高了交互效率。
- **模型的输出停止信号**：一位成员强调了**模型决定**何时停止函数调用的重要性，建议它可以通过不输出任何内容来发出回合结束的信号。
   - 这种灵活性对于在各种场景中保持流畅的用户交互至关重要。
- **天气查询演示函数调用**：一个说明性示例展示了模型如何使用 `get_coordinate` 和 `get_weather` 等函数调用来处理天气查询，展示了其数据检索过程。
   - 当模型在数据后的输出无法解码时，会话结束，有效地结束了该回合。
- **探索函数调用输出的可变性**：模型处理函数调用输出的方式允许它创造性地停止或扩展交互，包括选择不输出任何内容。
   - 这种可变性突显了模型利用各种 Prompt 技术来适应用户查询。



---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **对 LLM Finetuning 帮助的感谢**：一位用户对另一位成员在 **LLM Finetuning** 工作中的协助表示感谢。
   - 这一举动凸显了社区内的协作环境，展示了在应对技术挑战时的知识共享和支持。
- **贡献认可**：成员 cyberg0285 通过标签感谢了另一位社区成员的贡献，表明了互助的氛围。
   - 这种认可增强了从事复杂 LLM 项目的工程师之间的社区感和协作感。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **关于转发协议的讨论**：一位成员分享了关于 **forwarding protocols**（转发协议）的重要链接，强调了其在近期讨论中的相关性。
   - *以下是供参考的转发消息。*
- **信息共享的重要性**：另一位成员强调了建立适当 **information sharing**（信息共享）实践的必要性，以提高社区参与度并简化沟通。
   - 他们指出，*转发消息可以促进更快的响应和更清晰的沟通。*

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **AI Stewardship Practice Program 启动**：由 MaRS Discovery District 发起的 **AI Stewardship Practice Program** 为旨在积极影响 AI 发展的试点课程提供免费名额。更多详情请访问 [Tech Stewardship 网站](https://programs.techstewardship.com/)。
   - 该微证书项目专为研究人员、教育工作者和政策制定者设计，提供了参与 AI stewardship 实践的机会。
- **成为一名 Tech Steward**：参与者可以通过这项 Tech Stewardship 倡议参与旨在**引导技术走向良善**的项目。感兴趣的人士应在[此线程中回复](https://discord.com/channels/1089876418936180786/1295822228406931529)以加入价值 **500 CAD** 的试点课程。
   - 该项目旨在培养一个致力于负责任的 AI 实践和道德技术使用的 Tech Stewards 社区。

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道划分的详细摘要和链接

{% if medium == 'web' %}

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1295532010550595595)** (26 messages🔥): 

> - `Gradient Accumulation Bug Fix`
> - `Difference Between Base and Instruct Models`
> - `Normalization in Cross Entropy Loss` 


- **宣布 Gradient Accumulation Bug 修复**：针对一个导致在大 Gradient Accumulation 尺寸下所有训练损失发散的 Bug 实施了修复，该 Bug 影响了使用此方法的库。此问题与 Cross Entropy Loss 归一化器有关，需要进行去归一化以匹配训练损失曲线。
   - 有关此修复的更多详细信息可以在[此处的博客文章](http://unsloth.ai/blog/gradient)中找到，建议用户更新到最新版本。
- **关于部分批次权重错误的讨论**：会议强调，由于 Cross Entropy 中的归一化问题，意外地对部分批次加权过高，可能导致训练损失显著发散。在 Padding 影响损失计算的情况下尤其如此。
   - 成员们建议修改 Bug 描述的措辞，以澄清该问题特别影响在 PyTorch 中使用 mean reduction 和 Padding 的损失计算。
- **序列长度对损失计算的影响**：有人担心在使用不当的归一化策略时，较短的序列长度比较长的序列长度获得更高的权重。这可能导致损失进一步发散，尤其是在忽略输入 Token 损失时。
- **归一化实践受到审查**：一位用户评论了他们在代码中实现正确归一化的实践，并指出糟糕的归一化策略很常见。他们指出，对于不规则批次（ragged batches），按非 Mask 位置的数量取平均值经常会导致问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.desmos.com/calculator/rbh1evp5d0).">Desmos | Graphing Calculator</a>：未找到描述</li><li><a href="https://x.com/danielhanchen/status/1846235913443262891">Daniel Han (@danielhanchen) 的推文</a>：修复了一个导致在大 Gradient Accumulation 尺寸下所有训练损失发散的 Bug。1. 首先由 @bnjmn_marie 报告，GA 在数学上应该等同于全批次训练...</li><li><a href="https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html">CrossEntropyLoss &mdash; PyTorch 2.4 文档</a>：未找到描述</li><li><a href="https://github.com/karpathy/nanoGPT/commit/553f9">修复了必须缩放损失以考虑梯度累积的小 Bug... · karpathy/nanoGPT@553f949</a>：...累积，它在反向传播之前求和。注意这不是一个大 Bug，因为 AdamW 是缩放不变的。然而，这确实影响了梯度裁剪。
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1295461668893098045)** (228 messages🔥🔥): 

> - `QKNorm Effectiveness`
> - `ReLU^2 Performance`
> - `Impact of Sequence Length on Loss`
> - `Local vs. Global Attention`
> - `MoE Sparse Computation` 


- **QKNorm 的稳定性担忧**：QKNorm 在严苛基准测试中表现不佳，导致训练后出现“弱注意力（weak attention）”，引发了对其在大型模型中有效性的质疑。
   - 尽管如此，它仍被用于 Olmoe 项目，这表明对其设计选择可能存在内部分歧。
- **ReLU^2 的有限改进**：虽然发现 ReLU^2 在特定测试中仅带来 4% 的改进，但其在大型模型中的使用引发了性能提升是否值得其实施成本的担忧。
   - ReLU^2 与 GELU 及其他门控机制的性能对比，反映了关于大规模部署中激活函数的持续讨论。
- **比较不同序列长度的模型**：在评估基于不同序列长度训练的模型时，匹配评估序列长度以确保公平比较至关重要。
   - 较长的序列长度可能会产生更好的训练损失结果，但在一致的长度下进行评估有助于确定模型的真实效能。
- **局部与全局注意力动态**：局部注意力机制在较深层可能表现出更大的不稳定性，而全局注意力无论深度如何都能保持一致的梯度传播。
   - 这种区别可能导致注意力策略在不同模型架构中的有效性各不相同。
- **探索用于稀疏计算的 MoE**：关于使用混合专家（MoE）方法进行稀疏计算的讨论强调了其潜在优势，以及需要仔细实施以避免不稳定性。
   - 这种方法为动态模型架构开辟了新的可能性，例如即时生成权重，这可能代表该领域的重大进展。


<div class="linksMentioned">

<strong>提到的链接</strong>：

</div>

<ul>
<li>
<a href="https://x.com/Tim_Dettmers/status/1846023509811908751">来自 Tim Dettmers (@Tim_Dettmers) 的推文</a>：只是一个警告：我尝试了所有这些方法，而且它们在小规模下都有效……但当规模扩大时，对我来说没有一个起作用（除了 padding embeddings —— 但你真正应该做的是优化……）</li><li><a href="https://arxiv.org/abs/2410.10733">Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models</a>：我们提出了 Deep Compression Autoencoder (DC-AE)，这是一个用于加速高分辨率扩散模型的新型自编码器模型系列。现有的自编码器模型已经展示了令人印象深刻的结果……</li><li><a href="https://openreview.net/forum?id=P3SQi2EWeR">Integrating Large Circular Kernels into CNNs through Neural...</a>：正方形核是当代卷积神经网络 (CNN) 的标准单元，因为它非常适合卷积操作的张量计算。然而，视网膜神经节……</li><li><a href="https://arxiv.org/abs/2109.08668">Primer: Searching for Efficient Transformers for Language Modeling</a>：大型 Transformer 模型一直是近期自然语言处理进展的核心。然而，这些模型的训练和推理成本增长迅速，变得极其昂贵……</li><li><a href="https://arxiv.org/abs/2402.03804">ReLU$^2$ Wins: Discovering Efficient Activation Functions for Sparse LLMs</a>：稀疏计算通过动态跳过非活跃神经元的计算，为低资源场景下的大语言模型 (LLM) 推理提供了一个极具吸引力的解决方案。虽然传统的……</li><li><a href="http://arxiv.org/abs/2212.14034">Cramming: Training a Language Model on a Single GPU in One Day</a>：近期语言建模的趋势集中在通过扩展规模来提高性能，这导致了一个大多数研究人员和从业者都无法触及训练语言模型环境的局面……</li><li><a href="https://arxiv.org/abs/1803.00904">Hardness of Approximate Nearest Neighbor Search</a>：我们证明了在使用欧几里得、曼哈顿、汉明或编辑距离的近似双色最近对问题中，存在条件性的近二次运行时间下限。具体而言，除非强指数……</li><li><a href="https://arxiv.org/abs/2410.04271">Fundamental Limitations on Subquadratic Alternatives to Transformers</a>：Transformer 架构被广泛部署在许多流行且有影响力的 LLM 中。其核心是用于计算 token 对之间相关性的 Attention 机制。性能……</li><li><a href="https://x.com/Tim_Dettmers/status/1846223418989183417">来自 Tim Dettmers (@Tim_Dettmers) 的推文</a>：补充一点背景：我有一个非常严密的 Chinchilla 250M 模型基准，我对其进行了 1,000 多次训练运行。数据非常多样化。这个基准非常严密，以至于我所有的研究……</li><li><a href="https://x.com/Grad62304977/status/1846227646893461536">来自 Grad (@Grad62304977) 的推文</a>：嗯，谢谢澄清。对于 qk norm，有什么原因导致 Olmoe 使用了它吗？另外关于 zero init，澄清一下，它并没有带来巨大的性能提升，而是移除其替代方案……</li><li><a href="https://arxiv.org/abs/2209.04881">On The Computational Complexity of Self-Attention</a>：Transformer 架构在许多最先进的应用中取得了显著进展。然而，尽管取得了成功，现代 Transformer 仍依赖于 Self-Attention 机制，其时间和……</li><li><a href="https://github.com/KellerJordan/modded-nanogpt/blob/42ab270216d4d10abbadc34e92f92e8a011a384f/records/101424_ModernArch/train_gpt2.py#L159">modded-nanogpt/records/101424_ModernArch/train_gpt2.py 位于 42ab270216d4d10abbadc34e92f92e8a011a384f · KellerJordan/modded-nanogpt</a>：在 2.67B token 中达到 NanoGPT (124M) 的质量。通过在 GitHub 上创建账户，为 KellerJordan/modded-nanogpt 的开发做出贡献。
</li>
</ul>

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1295509540032217162)** (11 messages🔥): 

> - `优化类论文的评估`
> - `良好 Baseline 的重要性`
> - `误导性的性能提升`
> - `调优对结果的影响`
> - `LLM 论文的实用性` 


- **对优化类论文的批评**：成员们一致认为，如果优化类论文表现出缺乏优化基础知识，或者与未经调优的超参数挂钩，他们会降低这些论文结果的权重。
   - 这类论文通常被认为信息量较少，甚至会根据其错误的严重程度而被弃用。
- **Baseline 是关键指标**：缺乏良好的学习率或 Baseline 被视为评估模型性能的重大障碍。
   - 只有在显示出极端的性能偏差时，具有未经调优 Baseline 的论文才被认为有价值。
- **警惕误导性陈述**：“X 提升了性能”可能仅意味着边际稳定性提升，而非真正的进步，尤其是在执行较差的实验中。
   - 这种说法可能会掩盖模型修改的实际有效性。
- **LLM 研究中的 A/B 结果困境**：几位成员评论说，许多 LLM 论文产生的结果仅仅是 A/B 测试的对比，而非实质性的性能增强。
   - 这可能导致许多论文被认为几乎无用，使得研究人员可以忽略大部分研究结果。
- **Mamba 层与位置编码**：讨论显示 Mamba 层并不受益于位置编码 (Positional Encodings)，这展示了模型评估的另一层复杂性。
   - 这反映了一个更广泛的问题，即理解哪些修改有助于真正的改进，而哪些只是掩盖了潜在的不稳定性。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1295515800936906752)** (10 messages🔥): 

> - `微调库`
> - `Torchtune 模板`
> - `使用 ARC 数据集`
> - `加载 Hugging Face 数据集` 


- **探索像 Torchtune 这样的微调库**：一位成员询问是否缺少类似于 *lm-evaluation-harness* 的微调库，重点关注 *torchtune* 中聊天模板 (chat-template) 样式的挑战。
   - 他们表示更倾向于在不需要聊天模板的情况下评估微调后的模型。
- **澄清 Torchtune 的模板功能**：一位维护者讨论了如何调整 *PromptTemplate* 系统以适应不需要聊天模板的评估需求。
   - 他们建议在评估过程中使用 *QuestionAnswerTemplate*。
- **对 Torchtune 中 InstructTemplate 的担忧**：讨论强调了 *InstructTemplate* 的弃用，以及使用 *QuestionAnswerTemplate* 实现类似功能的可能性。
   - *QuestionAnswerTemplate* 格式化 Prompt 的方式完全符合该成员的评估需求。
- **加载 Hugging Face 数据集**：一位成员询问如何使用特定的 source 命令加载 *allenai/ai2_arc* 数据集，遇到了 “Config name missing” 错误。
   - 这揭示了在 *Hugging Face* 生态系统中访问数据集所需的配置可能存在问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/torchtune/stable/basics/prompt_templates.html#custom-prompt-templates)?">Prompt Templates &mdash; torchtune 0.3 documentation</a>：未找到描述</li><li><a href="https://pytorch.org/torchtune/stable/generated/torchtune.data.QuestionAnswerTemplate.html#torchtune.data.QuestionAnswerTemplate)">torchtune.data.QuestionAnswerTemplate &mdash; torchtune 0.3 documentation</a>：未找到描述</li><li><a href="https://pytorch.org/torchtune/stable/basics/prompt_templates.html#defining-via-dotpath-string),">Prompt Templates &mdash; torchtune 0.3 documentation</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1295549319767916554)** (164 条消息🔥🔥): 

> - `Unsloth 中的梯度累积 (Gradient Accumulation) 修复`
> - `去中心化训练模型 - INTELLECT-1`
> - `LLM 微调工作流`
> - `关于多 GPU 支持的讨论`
> - `模型性能对比` 


- **梯度累积修复提升训练效果**：Unsloth 宣布修复了一个梯度累积 (Gradient Accumulation) 中的重大 Bug，该 Bug 曾导致训练 Loss 发散。修复后准确度提升了 **10 倍**以上。
   - 建议用户更新 Unsloth，并访问演示修复方法及其对训练指标影响的新 Notebook。
- **INTELLECT-1 去中心化模型发布**：Prime Intellect 正在发布 **INTELLECT-1**，这是一个拥有 100 亿参数的模型，允许任何人贡献算力参与协作式去中心化训练运行，为开源 AGI 铺平道路。
   - 这是继他们之前的 OpenDiLoCo 工作之后的又一进展，OpenDiLoCo 扩展了 DeepMind 的分布式 AI 模型训练方法，实现了显著的模型改进。
- **探索 LLM 微调流程**：用户正在讨论实现各种 LLM 微调的工作流，包括在特定格式上进行训练以及实验模型输出。
   - 重点在于理解数据格式化如何影响 LLM 的行为和输出质量。
- **关于多 GPU 训练能力的讨论**：关于 Unsloth 目前多 GPU 支持的局限性存在争议，用户对这种配置的必要性和实用性持有不同意见。
   - 一些参与者建议使用 Deepspeed 等替代框架以获得更好的多 GPU 训练效果，而另一些人则强调 Unsloth 在单 GPU 上的效率。
- **模型间的性能对比**：参与者在微调和有效数据集利用的背景下，对比了开发中的各种模型（如 Qwen 和 Llama）的性能。
   - 对话中提到了数据质量对于实现最佳训练结果的重要性（优于数据数量）。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/danielhanchen/status/1846235913443262891">Daniel Han (@danielhanchen) 的推文</a>: 修复了一个导致大梯度累积大小时所有训练 Loss 发散的 Bug。1. 首先由 @bnjmn_marie 报告，GA 在数学上应该等同于全 Batch 训练...</li><li><a href="https://blog.continue.dev/a-custom-autocomplete-model-in-30-minutes-using-unsloth/">使用 Unsloth 在 30 分钟内构建自定义自动补全模型（社区文章）</a>: 这是 Sophia Parafina 在 Continue 博客上发表的客座文章，她是一位曾在 Pulumi、Anaconda 和 Docker 工作过的开发者倡导者。Continue 是一个开源 AI 代码助手...</li><li><a href="https://colab.research.google.com/drive/1z0XJU2FCzDC8oyXa2Nd4jCxylRMI-o0-?usp=sharing#scrollTo=95_Nn-89DhsL">Google Colab</a>: 无描述</li><li><a href="https://www.primeintellect.ai/blog/intellect-1">INTELLECT–1：启动首个 10B 参数模型的去中心化训练</a>: 我们很高兴发布 INTELLECT-1，这是首个 100 亿参数模型的去中心化训练运行，邀请任何人贡献算力并参与。这让我们离...更近了一步。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1g4ego7/llm_training_bug_fixes_gradient_accumulation_was/">Reddit - 深入探索</a>: 无描述</li><li><a href="https://www.zyphra.com/post/zamba2-7b">Zyphra</a>: 无描述</li><li><a href="https://ollama.com/unclemusclez/unsloth-qwen2.5">unclemusclez/unsloth-qwen2.5</a>: 基于 Unsloth 的 Qwen 2.5</li><li><a href="https://github.com/unclecode/crawl4ai">GitHub - unclecode/crawl4ai: 🔥🕷️ Crawl4AI: 开源 LLM 友好型网页爬虫与抓取工具</a>: 🔥🕷️ Crawl4AI: 开源 LLM 友好型网页爬虫与抓取工具 - unclecode/crawl4ai</li><li><a href="http://unsloth.ai/blog/gradient">LLM 训练中的 Bug 修复 - 梯度累积</a>: Unsloth 的梯度累积修复解决了 LLM 训练中的关键错误。</li><li><a href="https://x.com/UnslothAI/status/1846231235749990699">Unsloth AI (@UnslothAI) 的推文</a>: 今天，我们发布了一种改进所有人训练 LLM 方式的新方法。训练过程中存在一个导致 Loss 计算错误的重大 Bug。我们的梯度累积修复纠正了...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1295469775308193843)** (2 条消息): 

> - `Service Restoration`
> - `Community Reactions` 


- **服务恢复受到欢迎**：一位成员提到，在最近的问题之后，似乎一切都恢复正常了，这让社区感到宽慰。
   - 分享了 *☠️💀* 表情符号，以幽默的方式表达了对服务恢复的复杂心情。
- **社区宽慰的迹象**：服务的回归促使社区成员通过表情符号表达他们的感受，显示出宽慰和幽默。
   - 随着大家意识到停机结束，氛围变得轻松起来，展现了他们俏皮的一面。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1295519356771504210)** (16 条消息🔥): 

> - `Embedding Models`
> - `Quantization Methods`
> - `Integrating New Layers`
> - `Kaggle Package Installation`
> - `Finetuning Llama Models` 


- **Embedding 模型面临的困境**：一位成员提到，像 **Llama** 和 **Qwen** 这样的 GPT 模型由于架构限制，在 Embedding 任务中效果不佳。
   - 另一位成员建议探索其他替代方案，如 **BERT** 或 **Hugging Face** 上最新的 Sentence Embedding 模型。
- **Quantization 方法问题**：一项更新显示，模型的 **q4_0** Quantization 方法在 Tokenizer 合并更新后会输出乱码。
   - 切换到 **q4_k_m** Quantization 解决了该问题，这表明 **q4_0** 可能存在 Bug，可能需要在相关仓库中提交 Issue 报告。
- **集成新层的讨论**：一位成员询问如何将新层集成到模型中并进行训练，并提到之前在保存更改时遇到困难。
   - 另一位用户引用了 **LoRA** 作为将新的微小层有效整合到模型中的示例。
- **Kaggle 离线包安装咨询**：一位用户提出了关于在没有网络连接的情况下在 **Kaggle** 上安装包的问题，表示在模型上传方面遇到困难。
   - 这引发了关于在 **Kaggle** 环境中进行离线包管理可行性的关注。
- **将 Finetuned 模型推送到 Hugging Face**：一位成员表示打算将他们 Finetuned 的 **Llama 3B** 模型推送到 **Hugging Face**，以便使用 Inference API。
   - 他们询问为了优化性能，是否有必要合并为 **4-bit**。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1295533306384875531)** (2 条消息): 

> - `Fine-tuning Autocomplete Models`
> - `Continue AI Code Assistant`
> - `Unsloth Jupyter Notebooks`
> - `Llama Model Performance` 


- **Finetuning 自动补全模型变得简单**：Sophia Parafina 在 [Continue](https://blog.continue.dev/a-custom-autocomplete-model-in-30-minutes-using-unsloth/) 上的博客文章概述了如何使用 **Unsloth** 在开发数据上 Finetuning 自动补全模型。
   - 她强调了在**免费的 Google Colab 实例**上使用 **Jupyter Notebooks** 进行 Finetuning 的便利性，这显著降低了以前所需的时间和专业知识。
- **Continue：开源 AI 代码助手**：**Continue** 社区致力于构建一个集成多个模型（包括聊天和自动补全功能）的开源 AI 代码助手。
   - **Continue** 的一个重要功能是能够记录 [开发数据](https://docs.continue.dev/customize/development-data?ref=blog.continue.dev)，以根据开发者的需求增强模型性能。
- **Llama 模型的惊人性能**：一位参与者指出，**llama-3.1-70b-instruct** 模型达到了每秒 **230 tokens per second** 的速度。
   - 这一性能指标表明了 **Llama** 架构在处理输入方面的潜在效率和能力。



**提到的链接**：<a href="https://blog.continue.dev/a-custom-autocomplete-model-in-30-minutes-using-unsloth/">使用 Unsloth 在 30 分钟内构建自定义自动补全模型（社区帖子）</a>：这是 Sophia Parafina 在 Continue 博客上发表的一篇客座文章，她是一位曾在 Pulumi、Anaconda 和 Docker 工作过的开发者倡导者。Continue 是一个开源 AI 代码助手。它是……

  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1295606972715892737)** (6 messages): 

> - `SageAttention`
> - `Model Inference Optimization` (模型推理优化)
> - `Quantization Techniques` (量化技术)


- **SageAttention 承诺更快的模型推理**：论文 *SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration* 提出了一种新型量化方法，相比 FlashAttention2 和 xformers，其**每秒操作数 (operations per second)** 分别显著提升了约 **2.1 倍**和 **2.7 倍**，同时保持了准确性。
   - 作者声称，全面的实验表明在包括**语言处理**和**图像生成**在内的各种模型中，端到端指标几乎没有损失。
- **在训练中使用 SageAttention 的挑战**：一位成员表示，在尝试将其用于 *unsloth's llama.py* 文件后，SageAttention 的方法似乎无法用于训练，虽然速度有所提升但会导致 Loss 发散。
   - 另一位成员澄清说，SageAttention 主要针对**推理**而非训练，这可能是导致上述问题的原因。
- **探索 Transformer 中的计算优化**：讨论强调了 Transformer 架构中 Attention 的**计算复杂度**挑战，特别是 Attention 的 O(N^2) 与线性变换的 O(N) 之间的对比。
   - 这一背景强调了有效的量化方法对于解决模型推理过程中效率低下问题的重要性。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2410.02367">SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration</a>: Transformer 架构在各种模型中占据主导地位。作为 Transformer 的核心，Attention 的计算复杂度为 O(N^2)，而线性变换为 O(N)。当...</li><li><a href="https://github.com/HazyResearch/lolcats/blob/main/lolcats_preprint_v0.pdf">lolcats/lolcats_preprint_v0.pdf at main · HazyResearch/lolcats</a>: 通过在 GitHub 上创建账号来为 HazyResearch/lolcats 的开发做出贡献。
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1295461344744570921)** (159 messages🔥🔥): 

> - `Reasoning Feature in Perplexity` (Perplexity 中的推理功能)
> - `ProSearch Improvements` (ProSearch 改进)
> - `Model Performance Comparisons` (模型性能对比)
> - `Facial Recognition Concerns` (人脸识别担忧)
> - `User Experience Issues` (用户体验问题)


- **推理功能的变动性**：用户注意到在 ProSearch 中触发新的**推理功能 (reasoning feature)** 感觉具有随机性，并随问题复杂度而变化，导致分析结果不一致。
   - 旧的推理模型被认为更可靠，而新版本在生成信息时导致了**幻觉 (hallucinations)** 的增加。
- **ProSearch 和 Mac App 延迟**：许多用户对 **Mac app** 的延迟表示失望，该应用原定于更早的日期发布。
   - 此外，讨论还提到了应用中持续存在的**线程丢失**和性能迟缓问题。
- **用户对 AI 模型的期望**：关于各种 AI 模型性能的对话中，一些用户认为 **NotebookLM** 可能使用了 **Gemini Pro**，并将其与 **ChatGPT 4o** 进行了比较。
   - 用户还在探索 Perplexity 如何通过增加其他服务目前尚未提供的以 UX 为中心的功能来进行竞争。
- **人脸识别软件的局限性**：有人对利用 AI 进行**人脸识别**以帮助社交媒体上的霸凌受害者表示担忧，并建议像 **Yandex** 这样的工具仅在特定地区有效。
   - 用户承认，爬取 **Instagram** 和 **Snapchat** 等平台上的私有账户对任何 AI 工具都构成了重大挑战。
- **流式传输性能问题**：用户报告了 **iOS app** 搜索功能的问题，指出在访问来源时结果经常被截断。
   - 此外，还有关于收藏夹和保存的线程暂时消失的提及，尽管该问题稍后似乎已得到解决。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.testingcatalog.com/new-prosearch-feature-allows-perplexity-users-to-generate-visual-data-charts/">Perplexity rolled out chart generation feature for ProSearch</a>: 探索 Perplexity 新的 ProSearch 功能，该功能可根据搜索数据生成图表，提供强大的视觉洞察。非常适合财务和人口统计分析。</li><li><a href="https://civitai.com/user/karlcrane2015">Creator Profile | Civitai</a>: 在 Civitai 上了解更多关于这位优秀创作者的信息。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1295473200909647874)** (8 条消息🔥): 

> - `Adobe's AI Video Model`
> - `NASA's Europa Clipper Launch`
> - `Chinese Breakthrough in RSA Encryption`
> - `AMD's New AI Chips`
> - `Google's Acquisition of Nuclear Power` 


- **Adobe 的 AI Video Model 发布**：Perplexity AI 强调 **Adobe 的 AI Video Model** 是视频编辑和制作能力方面的一项重大进展，提供了旨在彻底改变工作流程的高级功能。
   - 该模型的影响可能会改变内容创作，使其对用户而言更快速、更易于获取。
- **NASA 发射 Europa Clipper 任务**：**NASA Europa Clipper** 任务已成功启动，旨在探索木星被冰覆盖的卫星——木卫二（Europa），以寻找潜在的生命迹象。
   - 专家们对发现新数据的可能性感到兴奋，这些数据可以揭示该卫星的地下海洋。
- **中国研究人员破解 RSA 加密**：最近的报告指出，**中国研究人员**已成功破解 RSA 加密，这是全球网络安全专家关注的重大问题。
   - 这一突破引发了人们对当前依赖此加密方法保护敏感信息的质疑。
- **AMD 推出新款 AI 芯片**：AMD 发布了其最新的 **AI 芯片**，旨在增强机器学习任务的性能并优化计算效率。
   - 预计这些创新将在蓬勃发展的 AI 硬件市场中展现出强大的竞争力。
- **Google 采取行动收购核能**：有消息称 **Google** 正在考虑收购 **核能** 技术，可能将其能源战略转向可持续资源。
   - 此举可能使 Google 成为科技行业利用绿色能源解决方案的领导者。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1295570544808165466)** (4 条消息): 

> - `Perplexity API limitations`
> - `BAAs for healthcare use case`
> - `Creating conversational chatbots` 


- **Perplexity API 面临域名限制**：一些用户注意到，使用 Perplexity API 时，查询**只能指定三个域名**。
   - 此外，他们发现**搜索结果并不局限于**那些指定的域名，这构成了一个挑战。
- **关于医疗保健 BAA 的咨询**：一位用户询问 Perplexity 是否会为企业用途签署 **Business Associate Agreements (BAAs)**，特别是在医疗保健场景下。
   - 这一请求强调了在敏感行业使用 API 时，需要对**合规性有明确的说明**。
- **寻求使用 Perplexity API 创建聊天机器人的资源**：有人请求提供关于如何使用 **Perplexity API 创建对话式聊天机器人**的资源。
   - 用户预先表达了感谢，并**对社区的任何指导表示赞赏**。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1295579150261948609)** (107 messages🔥🔥): 

> - `Aider 的性能与特性`
> - `API 使用与配额问题`
> - `LLM 集成与脚本使用`
> - `模型效果与对比`
> - `弱模型 (Weak models) 与 Prompt 缓存` 


- **Aider 使用多个模型的能力**：用户讨论了同时运行多个 Aider 实例来处理大型和小型任务的可行性，其中一位用户表示，只要不编辑相同的文件，这样做应该是没问题的。
   - 一位用户幽默地将其称为 “LLM 派对”。
- **API 使用错误与配额**：一位用户在使用 OpenRouter 的 4o-mini 模型时遇到了 “Resource exhausted” 错误，这表明他们可能超出了配额，或者 OpenRouter 达到了其配额限制。
   - 另一位用户强调，Anthropic 的模型将输入标记为内容审核 (moderation)，这可能会导致重复的 API 连接错误。
- **结合 LLM 的脚本策略**：讨论了如何结合脚本与 Aider 来高效处理大型代码库，一位用户分享了他们对代码段采用系统化命名约定的经验。
   - 另一位用户提到了一个 LLM Agent 框架计划，该框架可自动执行各种任务，包括系统管理。
- **编程任务的模型对比**：一些用户指出，与 Gemini 等其他模型相比，Sonnet-3.5 在其特定用例（尤其是非 Web 开发任务）中表现更优。
   - 一位用户强调了对不同模型的测试，并指出虽然有些模型表现较好，但 Sonnet-3.5 始终能提供最佳结果。
- **弱模型功能与 Prompt 缓存**：一位用户询问省略 `--weak-model` 标志是否默认使用 Sonnet，并要求澄清 Gemini 作为弱模型的功能。
   - 另一位用户指出，除非另有说明，否则 Aider 在使用 Anthropic API 时默认使用 Claude。


<div class="linksMentioned">

<strong>相关链接</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/config/options.html#--commit-prompt-prompt">选项参考 (Options reference)</a>: 关于 aider 所有设置的详细信息。</li><li><a href="https://openrouter.ai/anthropic/claude-3.5-sonnet:beta">Claude 3.5 Sonnet (自我审核) - API, 提供商, 统计数据</a>: Claude 3.5 Sonnet 提供优于 Opus 的能力，速度快于 Sonnet，且价格与 Sonnet 相同。通过 API 运行 Claude 3.5 Sonnet (自我审核)</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/quotas#error-code-429">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1295473019262603325)** (42 messages🔥): 

> - `Aider model API key issues`
> - `File modification behavior in Aider`
> - `Scripting Aider commands`
> - `Error handling for Aider edits`
> - `Gemini model integration with Aider` 


- **Aider 在 API key 验证方面遇到困难**：用户在 `.env` 文件中设置 key 后，尝试为 Aider 配置 Gemini 模型时遇到了 **API 验证错误**。
   - 一位用户确认该 key 在命令行中可以正常工作，这表明其脚本设置可能存在问题。
- **对文件修改行为的困惑**：成员们对于 Aider 在处理修改时是使用文件系统中的最新文件还是 Git 仓库中的文件，分享了截然不同的经验。
   - 尽管预期会使用文件系统版本，但一位用户指出其中的差异导致了非预期的行为。
- **编写 Aider 命令和配置的脚本**：用户讨论了如何通过命令行或 Python **编写 Aider 命令脚本**，强调了在编写脚本时正确加载环境的必要性。
   - 一位用户修改了一个示例脚本以设置 Gemini 模型，但遇到了与环境变量相关的错误。
- **Aider 编辑的错误处理策略**：成员们寻求处理 Aider 无法应用更改的情况的方法，并对不理想的输出表示沮丧。
   - 建议包括使用 `/clear` 或 `/drop` 命令来减少编辑过程中的干扰。
- **Gemini 模型集成查询**：有一个关于如何在 Aider 中正确配置和使用 **Gemini-1.5 Pro 模型** 的具体咨询，重点在于确保 API key 已正确设置。
   - 虽然参考了文档，但用户仍面临与 API 错误和所需环境配置相关的挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/scripting.html">编写 aider 脚本</a>：你可以通过命令行或 Python 编写 aider 脚本。</li><li><a href="https://aider.chat/docs/troubleshooting/edit-errors.html">文件编辑问题</a>：aider 是你终端里的 AI 结对编程工具</li><li><a href="https://aider.chat/docs/llms/gemini.html">Gemini</a>：aider 是你终端里的 AI 结对编程工具
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1295476202072707115)** (101 条消息🔥🔥): 

> - `HuggingFace account recovery` (HuggingFace 账号恢复)
> - `Data Science and ML job security` (Data Science 和 ML 职业保障)
> - `Llama 3.2 inference speed` (Llama 3.2 推理速度)
> - `AI in job roles` (AI 在工作岗位中的作用)
> - `AWS Cognito integration issues` (AWS Cognito 集成问题)


- **急需恢复 HuggingFace 账号**：一名成员报告其 HuggingFace 账号被盗并被删除，寻求紧急恢复协助。建议其发送邮件至 [website@huggingface.co](mailto:website@huggingface.co) 寻求帮助。
   - 另一名成员指出恢复可能需要几天时间，请耐心等待回复。
- **对 AI 自动化取代工作的担忧**：讨论揭示了对 AI 快速发展及其在 Data Science 和 ML 领域自动化潜力的焦虑。成员们希望 AI 能改变工作角色而非取代它们，主张转向更具创造性的工作。
   - 一位成员将 AI 的影响与之前的技术进步进行了比较，认为它将增强而不是消除工作岗位。
- **Llama 3.2 模型推理问题**：一位用户报告称，在 A100 GPU 上使用 Llama 3.2 1B 模型对大型数据集进行推理耗时超过 14 小时，引发了对效率的担忧。讨论了关于优化负载和推理方法的建议。
   - 他们分享了加载模型和执行推理的方法，寻求潜在改进的建议。
- **AWS Cognito 与 HuggingFace 集成问题**：一位用户描述了在将 HuggingFace 登录与 AWS Cognito 集成时遇到的困难，特别是仅收到 ACCESS_TOKEN 而非所需的 ID_TOKEN。他们强调了情况的紧急性，因为这导致了项目延期。
   - 鼓励成员们分享任何见解或解决方案来解决集成问题。
- **使用 LLM 和静态组件设置聊天系统**：一位成员询问如何设置一个使用 LLM 的聊天系统，该系统既包含文本生成又包含静态组件（widgets），且不会产生冗余或丢失对话风格。他们强调了在平衡动态响应与模板约束方面面临的挑战。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/rush-hour-you-and-me-me-and-you-yu-mee-gif-6521388707144075848">Rush Hour You And Me GIF - Rush hour You and me Me and you - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/theodd1sout-vsauce-or-is-it-gif-16159095273288783425">Theodd1sout Vsauce GIF - Theodd1sout Vsauce Or is it - Discover &amp; Share GIFs</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1295649074329948210)** (3 条消息): 

> - `Calculus learning` (微积分学习)
> - `Flet and OpenVino exploration` (Flet 和 OpenVino 探索)


- **在 Fast AI 之后深入学习微积分**：一位用户分享了他们在完成 **Fast AI Part 1 课程**并学习了**线性代数**（Linear Algebra）后，计划在今天完成**微积分**（Calculus）的学习。
   - *他们表达了在增强理解后进一步深入研究神经网络代码的兴奋之情。*
- **探索 Flet 和 OpenVino**：另一位用户报告称他们正在学习 **Flet** 和 **OpenVino**，表明他们有兴趣扩展在这些领域的技能。
   - *未提供更多细节，但这些技术在 AI 应用开发中正受到关注。*


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 条消息): 

laolala: https://huggingface.co/movaxbx/OpenHermes-Emojitron-001
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1295499682545406042)** (8 messages🔥): 

> - `VividNode AI chatbot`
> - `GPT4Free 说明`
> - `HybridAGI 特性`
> - `在 Google Colab 上运行 Metaflow`
> - `Open TTS Tracker 作为 HF dataset` 


- **VividNode AI 聊天机器人登陆 Mac 和 Linux**：好消息！**VividNode AI chatbot** 现在已支持 Mac 和 Linux 平台，鼓励用户在[此处](https://github.com/yjg30737/pyqt-openai)探索其功能。
   - *如果你有兴趣贡献代码或启动侧边项目，*请联系作者以获取更多想法。
- **了解 GPT4Free**：一名成员询问了关于 **GPT4Free** 以及如何免费访问它的问题，得到的回复是它并非真正免费，因为速度较慢且有时响应不完整。
   - 有关 **GPT4Free** 的详细信息可以在[文档](https://github.com/xtekky/gpt4free)中找到。
- **介绍 HybridAGI 框架**：详细介绍了 **HybridAGI**，这是一个基于可编程图的开源框架，重点是通过基于图的编程语言来实现 Agent 行为。
   - 核心特性包括 **Human-in-the-Loop 控制**和 **混合 Vector/Graph Memory**；更多信息请访问其 [GitHub 页面](https://github.com/SynaLinks/HybridAGI)。
- **在 Google Colab 上自托管 Metaflow**：现在可以使用 Google Colab 自托管 **Metaflow**，正如一篇新发表的文章所讨论的，该文章深入探讨了在不依赖 S3 的情况下进行实际实现。
   - 文章涵盖了 Metaflow 的各个方面，包括其特性和未来潜力，可在此处访问：[链接](https://huggingface.co/blog/Aurelien-Morgan/stateful-metaflow-on-colab)。
- **Open TTS Tracker GitHub 仓库转换**：**Open TTS Tracker GitHub Repo** 已转换为 Hugging Face 数据集，提供了各种 TTS 模型的结构化信息。
   - 用户可以在[此处](https://huggingface.co/datasets/Pendrokar/open_tts_tracker)探索该数据集，以查找不同的 TTS 功能和标准。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/blog/Aurelien-Morgan/stateful-metaflow-on-colab">在 Google Colab 上运行酷炫的有状态 Metaflow 服务 + UI？</a>：未找到描述</li><li><a href="https://github.com/yjg30737/pyqt-openai">GitHub - yjg30737/pyqt-openai: VividNode: 多用途文本和图像生成桌面聊天机器人（支持包括 GPT 在内的各种模型）。</a>：VividNode: Multi-purpose Text &amp; Image Generation Desktop Chatbot (supporting various models including GPT). - yjg30737/pyqt-openai</li><li><a href="https://github.com/SynaLinks/HybridAGI">GitHub - SynaLinks/HybridAGI: 基于 Cypher 的可编程 Neuro-Symbolic AGI，允许你使用基于图的 Prompt Programming 来编程其行为：适用于希望 AI 表现符合预期的人群</a>：The Programmable Cypher-based Neuro-Symbolic AGI that lets you program its behavior using Graph-based Prompt Programming: for people who want AI to behave as expected - SynaLinks/HybridAGI</li><li><a href="https://huggingface.co/datasets/Pendrokar/open_tts_tracker">Pendrokar/open_tts_tracker · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1295622381452136618)** (8 messages🔥): 

> - `读书小组活动`
> - `频道访问问题`
> - `论文讨论` 


- **读书小组活动引发关注**：一名成员提到在特定频道看到了读书小组（Reading Group）活动，并在此处询问了详细信息：[链接](https://discord.com/events/879548962464493619/1293961951260446811)。
   - 其他成员不确定活动的具体位置，并指向了另一个可能进行讨论的频道。
- **频道访问问题**：成员们表示访问所讨论的频道存在困难，其中一人表示：*“我还没有访问该频道的权限。”*
   - 有建议称应标记每个角色以获取潜在访问权限，或者直接向另一名成员寻求帮助。
- **分享论文讨论链接**：一名成员分享了与读书小组活动相关的论文链接（[arxiv 链接](https://arxiv.org/abs/2405.10725)），并建议在频道中进行讨论。
   - 这引发了更多关于频道位置和成员可见性的疑问。
- **确认可见性**：经过一番交流，一名成员确认他们现在可以在频道中看到活动内容，回答了之前的问题。
   - 这似乎促进了信息的流动，并理清了小组参与者之间的访问权限。


  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1295681808763195403)** (2 messages): 

> - `Flutter 开发`
> - `AI 应用协作` 


- **提供 Flutter 开发协作**：一位成员宣布可以作为 **Flutter 开发者**参与构建 **AI 应用**的协作。
   - 他们表示愿意与对开发以 AI 为核心的项目感兴趣的人员组队。
- **协作呼吁**：该成员再次重申了与希望使用 Flutter 开发 AI 应用的其他人员进行**协作**的兴趣。
   - 这突显了 AI 应用开发领域对**合作伙伴关系**的持续需求。


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1295485921780568124)** (1 messages): 

> - `DiT 训练`
> - `《超级马里奥兄弟》游戏画面图像`
> - `自定义 VAE 压缩` 


- **成功实现 DiT 训练**：一位成员报告了从头开始训练 **DiT** (Detection in Training) 的进展，并指出最终获得了一些不错的结果。
   - 此次训练涉及使用特定的**《超级马里奥兄弟》游戏画面图像**，展示了在游戏领域的潜在应用。
- **创新的 24 倍自定义 VAE 压缩**：他们在训练中采用了压缩率高达 **24倍** 的**自定义 VAE** (Variational Autoencoder)。
   - 这一选择表明在训练 DiT 模型时，重点在于优化数据效率。


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1295726111384277042)** (1 messages): 

> - `Gradio 5 发布`
> - `Product Hunt 支持` 


- **Gradio 5 发布公告**：我们刚刚在 [Product Hunt](https://www.producthunt.com/posts/gradio-5-0) 上发布了 **Gradio 5**，非常期待您的支持！
   - *请花一点时间查看并表达您的喜爱* 🧡。
- **鼓励社区支持**：团队鼓励大家抽出时间在 Product Hunt 上支持 **Gradio 5**。这一社区驱动的努力旨在提高发布的曝光度和参与度。
   - 邀请成员参与讨论并分享对新功能的反馈。



**提到的链接**：<a href="https://www.producthunt.com/posts/gradio-5-0"> Gradio 5.0 - 构建 AI Web 应用最简单的方式 | Product Hunt</a>：一个用于轻松构建和共享基于 Web 的 AI 应用的开源库。只需几行 Python 代码即可部署、自定义和共享机器学习模型 Demo。

  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1295671338983358544)** (1 messages): 

> - `Hermes 3 Llama 3.1 405B`
> - `Nous Hermes Yi 34B 弃用` 


- **Hermes 3 Llama 3.1 405B 现为付费模型**：**Hermes 3 Llama 3.1 405B Instruct** 模型现在的价格为 **$1.79/月**，不过在 [OpenRouter](https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b) 上仍可使用免费变体。
   - *不要错过这个针对强大 AI 功能更新的定价结构！*
- **Nous Hermes Yi 34B 已弃用**：**Nous Hermes Yi 34B** 模型已被所有服务提供商弃用，不再提供使用。
   - *鉴于此次弃用，鼓励用户过渡到替代模型。*



**提到的链接**：<a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b">Hermes 3 405B Instruct - API、提供商、统计数据</a>：Hermes 3 是一款通用的语言模型，相比 Hermes 2 有许多改进，包括先进的 Agent 能力、更好的角色扮演、推理、多轮对话、长上下文连贯性等...

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1295487224715935915)** (90 条消息🔥🔥): 

> - `AI Model Performance`
> - `Chatbot Development`
> - `OpenRouter Features`
> - `Model Comparison`
> - `Provider Issues` 


- **AI 模型排名**：用户讨论了各种用于聊天和角色扮演的 AI 模型的性能，其中 **Llama-3-8b-Instruct** 和 **GPT-4o** 因其出色的指令遵循能力而受到关注。
   - *Grok 2 mini* 和 *Gemini 1.5 Pro* 也被提及为可行选项，尽管 *Opus* 因其一些怪癖而受到了一些批评。
- **Chatbot 设计技巧**：一位用户询问如何创建一个隐藏的 AI Chatbot，以避免对侮辱性言论产生通用的拒绝回复，并提议使用另一个 LLM 来过滤不良内容。
   - 其他人建议使用像 *Llama Guard* 这样的模型，在允许响应之前提供额外的消息检查支持。
- **OpenRouter 的功能与用法**：社区讨论了如何通过利用 Header 过滤掉不需要的 Provider 来限制 OpenRouter 中的模型使用，从而增强隐私设置。
   - 一位成员强调了 *LiteLLM guardrails 功能*，它提供了使用 *Llama Guard* 等模型进行请求检查的安全保障。
- **Infermatic Provider 的问题**：一位用户报告了 **Infermatic** Provider 的问题，称其对话开始意外地输出无关的响应。
   - 社区收到了关于在短时间内可能出现的潜在服务中断的警示。
- **用户体验与反馈**：用户对 OpenRouter playground 的新功能表示兴奋，例如拖放文件上传和图像粘贴功能。
   - 另一位用户注意到 **Gemini** 模型中的 Prompt 存在超过 **90 秒** 的显著延迟，这表明性能体验各不相同。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/settings/privacy">Privacy | OpenRouter</a>: 管理您的隐私设置</li><li><a href="https://dubesor.de/benchtable">Dubesor LLM Benchmark table</a>: 未找到描述</li><li><a href="https://openrouter.ai/docs/provider-routing#custom-routing">Provider Routing | OpenRouter</a>: 跨多个 Provider 路由请求</li><li><a href="https://openrouter.ai/meta-llama/llama-guard-2-8b">LlamaGuard 2 8B - API, Providers, Stats</a>: 该安全模型拥有 8B 参数，基于 Llama 3 系列。就像其前身 [LlamaGuard 1](https://huggingface. 通过 API 运行 LlamaGuard 2 8B</li><li><a href="https://openrouter.ai/rankings/roleplay?view=week">LLM Rankings: roleplay | OpenRouter</a>: 根据角色扮演 Prompt 的使用情况进行排名和分析的语言模型
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1295465473038745673)** (72 messages🔥🔥): 

> - `Nous Research Community`
> - `Gradient Accumulation Fix`
> - `Zamba2-7B Performance`
> - `AI Training Techniques`
> - `Open Source AI Projects` 


- **Nous Research 社区的起源**：Nous Research 社区最初是 Discord 上一个专注于 AI 研究的小组，现在已经演变成一家获得融资的技术公司。
   - 成员们分享想法、协作项目并讨论各种 AI 模型和技术，营造了一个极具参与感的环境。
- **梯度累积（Gradient Accumulation）Bug 修复发布**：[UnslothAI 团队](https://x.com/danielhanchen/status/1846235913443262891) 详细说明并修复了一个导致训练损失（training losses）发散的重大梯度累积 Bug。
   - 该修复提高了各种设置下训练损失的一致性，现在所有用户均可实施。
- **Zamba2-7B 模型发布**：Zyphra 宣布推出 Zamba2-7B 模型，声称其在质量和性能上均优于 Llama3 和 Mistral 等现有模型。
   - 这一新模型旨在实现消费级 GPU 上的高效部署，并在最近的 [博客文章](https://www.zyphra.com/post/zamba2-7b) 中详细介绍了其功能。
- **关于 AI 训练技术的讨论**：一位成员对使用 16384 序列长度训练 AI 提出了疑虑，引发了关于其影响和最佳实践的对话。
   - 尽管具有创新性，但大家认识到此类技术在实际应用中可能会带来挑战，这突显了 AI 训练的动态特性。
- **社区成员的技能与贡献**：一位全栈区块链开发人员表示愿意为社区贡献技能，展现了开放协作的态度。
   - 成员们经常分享专业知识并寻求帮助，增强了 Nous Research Discord 的社区精神。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/UnslothAI/status/1846231235749990699">来自 Unsloth AI (@UnslothAI) 的推文</a>: 今天，我们发布了一种改进 LLM 训练的新方法。存在一个导致训练期间损失计算错误的重大 Bug。我们的梯度累积修复纠正了...</li><li><a href="https://x.com/danielhanchen/status/1846235913443262891">来自 Daniel Han (@danielhanchen) 的推文</a>: 修复了一个导致大梯度累积大小时所有训练损失发散的 Bug。1. 最初由 @bnjmn_marie 报告，GA 在数学上应该等同于全批次训练...</li><li><a href="https://x.com/danielhanchen/status/1846235913443262891?s=46">来自 Daniel Han (@danielhanchen) 的推文</a>: 修复了一个导致大梯度累积大小时所有训练损失发散的 Bug。1. 最初由 @bnjmn_marie 报告，GA 在数学上应该等同于全批次训练...</li><li><a href="https://a16z.com/podcast/distro-and-the-quest-for-community-trained-ai-models/">DisTrO 与社区训练 AI 模型的探索 | Andreessen Horowitz</a>: Nous Research 的 Bowen Peng 和 Jeffrey Quesnelle 讨论了他们加速开源 AI 研究的使命，包括一个名为 DisTrO 的新项目。</li><li><a href="https://www.zyphra.com/post/zamba2-7b">Zyphra</a>: 未找到描述</li><li><a href="https://delphidigital.io/crypto-ai">Crypto x AI 月度活动</a>: 加密货币领域最大的虚拟 AI 会议，我们邀请了一些最高产的加密 AI 构建者和愿景者。加入我们的现场演讲、辩论和市场论点，包括 Nous Research, Prime Inte...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1295810500520574989)** (3 messages): 

> - `Llama3 identification`
> - `LLMs and animal comparison`
> - `Statistical evaluation on animal identification` 


- **Llama3 被识别为章鱼**：一位成员幽默地声称 **Llama3** 更多地认同自己是**章鱼**，并指出 LLM 与动物相关的传播范围出奇地小。
   - 常见的身份认同包括**章鱼**、**猫头鹰**、**狼**和**海豚**，目前还没有发现认同自己是羊驼（llama）的情况。
- **避免回复中的 AI 陈词滥调**：同一位成员表示需要使用系统提示词（system prompt）来防止典型的 AI 回复以“作为一个 AI...”开头。
   - 这突显了在与 LLM 交互时，对其默认对话模式的一种普遍挫败感。
- **需要进行统计评估**：该成员提到打算对有关动物身份认同的回答进行统计评估。
   - 旨在更清晰地了解 LLM 如何与各种动物身份保持一致，这可能会影响未来的交互。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1295475077650976932)** (5 条消息): 

> - `Model Collapse Phenomenon`
> - `SageAttention Quantization Method`
> - `Slow Response Issues` 


- **合成数据导致的 Model Collapse**：一项研究强调了 **Model Collapse 现象**，即训练中即使只有 **1%** 的合成数据也会导致性能恶化，尽管数据量在增加，这表明在训练像 ChatGPT 和 Llama 这样的大型模型时存在巨大风险。该研究表明，更大的模型可能会加剧这一问题。
   - 研究深入探讨了 **supervised regression settings** 下的训练动态，并对模型缩放（Model Scaling）中的常见做法提出质疑，引起了对未来模型设计的重大影响的关注。
- **SageAttention：提升 Transformer 模型效率**：介绍了 **SageAttention**，这是一种新型的量化方法，可提升 Attention 机制的性能，与 FlashAttention2 和 xformers 等现有方法相比，每秒操作数提高了 **2.1 到 2.7 倍**。该方法在保持准确性的同时，降低了包括语言和图像处理在内的各种模型类型的计算复杂度。
   - 全面实验验证了 SageAttention 在端到端指标上 **几乎没有损失**，使其成为 Transformer 架构推理加速的一个极具前景的途径。
- **用户报告响应时间缓慢**：成员们对使用过程中遇到“Rate exceeded”消息表示担忧，这表明系统性能可能存在减速。此类报告表明用户体验问题正在影响平台的参与度。
   - 这些响应缓慢的报告突显了持续存在的挑战，并强调需要关注优化 **系统响应能力** 和用户满意度。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2410.02367">SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration</a>：Transformer 架构在各种模型中占据主导地位。作为 Transformer 的核心，Attention 的计算复杂度为 O(N^2)，而线性变换为 O(N)。当 ...</li><li><a href="https://arxiv.org/abs/2410.04840">Strong Model Collapse</a>：在支持 ChatGPT 和 Llama 等大型神经网络训练的 Scaling Laws 范式内，我们考虑了监督回归设置，并确定了一种强形式的存在...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 条消息): 

xandykati98: https://x.com/AISafetyMemes/status/1846220545542529329
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1295475077650976932)** (5 条消息): 

> - `Model Collapse Phenomenon`
> - `SageAttention Quantization`
> - `Performance Issues`
> - `SageAttention vs. FlashAttention` 


- **合成数据导致的 Model Collapse**：[研究论文](https://arxiv.org/abs/2410.04840) 揭示了训练集中即使只有 **1%** 的合成数据也会导致严重的 **Model Collapse**，从而损害 ChatGPT 等大型神经网络的性能。
   - 研究强调，增加模型大小实际上可能会 **加剧** 这一问题，这与当前的训练趋势相反。
- **SageAttention 在 Attention 机制中的效率**：[SageAttention](https://arxiv.org/abs/2410.02367) 提供了一种新型量化方法，增强了 Attention 的计算效率，分别比 FlashAttention2 和 xformers 高出约 **2.1 倍** 和 **2.7 倍**。
   - 这种方法在加速推理的同时保持了准确性，在各种应用中的性能指标几乎没有损失。
- **对系统性能的担忧**：成员们表达了与系统 **性能** 相关的问题，其中一位指出感觉 **缓慢**。
   - 另一位成员补充了这一观点，称他们收到了 **'Rate exceeded'** 消息，表明系统可能过载。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2410.02367">SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration</a>：Transformer 架构在各种模型中占据主导地位。作为 Transformer 的核心，Attention 的计算复杂度为 O(N^2)，而线性变换为 O(N)。当 ...</li><li><a href="https://arxiv.org/abs/2410.04840">Strong Model Collapse</a>：在支持 ChatGPT 和 Llama 等大型神经网络训练的 Scaling Laws 范式内，我们考虑了监督回归设置，并确定了一种强形式的存在...
</li>
</ul>

</div>

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1295552267860050011)** (1 messages): 

> - `Lux-AI Challenge`
> - `Team Collaboration` 


- **Lux-AI Challenge: GitHub 仓库动态**: 一位成员分享了 [Lux-AI Challenge 的 GitHub 仓库](https://github.com/Lux-AI-Challenge/Lux-Design-S3) 链接，邀请其他人参与开发。
   - 该倡议旨在鼓励协作，希望通过组建团队来提升项目成果。
- **Lux-AI 项目团队协作请求**: 配合 GitHub 链接，一位成员询问是否有人有兴趣为 Lux-AI 项目组队。
   - *“有人有兴趣为此组队吗？”* 这句话发出了社区参与项目贡献的号召。



**提到的链接**: <a href="https://github.com/Lux-AI-Challenge/Lux-Design-S3">GitHub - Lux-AI-Challenge/Lux-Design-S3</a>: 通过在 GitHub 上创建账号，为 Lux-AI-Challenge/Lux-Design-S3 的开发做出贡献。

  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1295541141047083059)** (15 messages🔥): 

> - `triton-lang 安装问题`
> - `LLVM 对 ARM 的支持`
> - `Triton 的 Windows 构建`
> - `CUDA 错误处理`
> - `打包数据循环问题` 


- **Triton 在 Jetson 上的安装困扰**: 一位用户报告了在 **Jetson Orin AGX 64GB** 上构建 **triton-lang** 时遇到的麻烦，CUDA 错误地将 Unified Memory 识别为 `AMD GPU`。他们提到将重新构建 Triton，并希望 LLVM 构建是导致安装失败的根源。
   - 另一位成员建议检查 **LLVM** 对 ARM 的支持，并参考了[此处](https://github.com/triton-lang/triton/issues?q=sort%3Aupdated-desc+is%3Aissue+jetson+is%3Aclosed)的相关 issue。
- **Triton 对 Windows 的非官方支持**: 讨论中提到了 **Triton for Windows** 的非官方构建，并分享了一个声称兼容 Windows 的 GitHub 仓库链接。据称，尽管存在与 MSVC 兼容性相关的挑战，一些用户已成功构建。
   - 提到了一项关于 Triton on Windows 已关闭 PR 的探索性工作，并有构建成功的报告，暗示可能取得了部分成功。
- **Triton 与 CUDA 错误的问题**: 一位成员在 **A100** 和 **H100** GPU 上执行 kernel 时遇到了错误 `triton LLVM ERROR: mma16816 data type not supported`。根据最近的测试，之前通过调整循环实现等变通方法已不再有效。
   - 另一位成员建议研究来自 [GitHub 链接](https://github.com/triton-lang/triton/blob/17d633a64e43337037d2e873b029fab92422762f/lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp#L125C20-L125C25) 的细节，据报道该链接提供了相关问题的解决方案。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/StableDiffusion/comments/1g45n6n/triton_3_wheels_published_for_windows_and_working/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://github.com/trito">Trito - 概览</a>: Trito 有一个可用仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/triton-lang/triton/blob/17d633a64e43337037d2e873b029fab92422762f/lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp#L125C20-L125C25">triton/lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp at 17d633a64e43337037d2e873b029fab92422762f · triton-lang/triton</a>: Triton 语言和编译器的开发仓库 - triton-lang/triton</li><li><a href="https://github.com/triton-lang/triton?tab=readme-ov-file#install-from-source">GitHub - triton-lang/triton: Triton 语言和编译器的开发仓库</a>: Triton 语言和编译器的开发仓库 - triton-lang/triton</li><li><a href="https://github.com/triton-lang/triton/issues?q=sort%3Aupdated-desc+is%3Aissue+jetson+is%3Aclosed">Issues · triton-lang/triton</a>: Triton 语言和编译器的开发仓库 - Issues · triton-lang/triton</li><li><a href="https://github.com/woct0rdho/triton">GitHub - woct0rdho/triton-windows: 支持 Windows 的 Triton 语言和编译器分支</a>: 支持 Windows 的 Triton 语言和编译器分支 - woct0rdho/triton-windows</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/gnvRBZvkZk">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://rosenzweig.io/blog/asahi-gpu-part-3.html">剖析 Apple M1 GPU，第三部分</a>: 未找到描述</li><li><a href="https://dougallj.github.io/applegpu/docs.html">Apple G13 GPU 架构参考</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1295471062095757385)** (6 messages): 

> - `Model Parallelism vs Data Parallelism`
> - `Titanet-large Model Performance`
> - `Learn PyTorch Course`
> - `Research Paper Release`
> - `CPU Utilization in GPU Tasks` 


- **模型并行（Model Parallelism）通常是次要选择**：由于实现简单，数据并行（Data Parallelism）通常是深度学习中首选的方法，只有在内存限制必要时才使用模型并行。
   - 成员们建议，那些采用模型并行的人通常已经优化了数据并行。
- **Titanet-large 模型在 GPU 上运行的问题**：一位用户报告在 GPU T4 机器上运行脚本化的 Titanet-large 模型时，在 GPU 使用的同时 CPU 利用率也很高。
   - 他们在使用特定的模型转换代码时，寻求调试此性能问题的建议。
- **学习 PyTorch 课程现已上线**：分享了 [Learn PyTorch for Deep Learning: Zero to Mastery](https://www.learnpytorch.io/) 课程的链接，该课程被誉为学习 PyTorch 的第二好资源。
   - 该课程承诺通过以在线书籍形式为中心的视频内容来教授基础概念。
- **研究论文发布的兴奋**：成员们对一篇新研究论文的发布表示兴奋，这标志着该领域的重大进展。
   - 他们分享了论文链接以及为该研究做出贡献的各位作者。
- **深度学习任务中的高 CPU 占用**：一位用户询问在运行基于 GPU 的任务时 CPU 利用率过高的问题，表示希望诊断这一意外的性能表现。
   - 他们提供了用于模型转换的具体代码，希望能从社区获得见解。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2410.06511">TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training</a>：大语言模型（LLMs）的发展对于推动最先进的自然语言处理应用起到了至关重要的作用。训练具有数十亿参数和数万亿...</li><li><a href="https://www.learnpytorch.io/">Home</a>：通过编写 PyTorch 代码动手学习重要的机器学习概念。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1295481627983478875)** (9 messages🔥): 

> - `LegoScale`
> - `Modular 3D Parallelism`
> - `FSDP2`
> - `TorchTitan`
> - `Large Language Models` 


- **LegoScale 为 LLM 训练带来新功能**：最近讨论的论文介绍了 **LegoScale**，这是一个用于大语言模型 **3D 并行预训练** 的开源 PyTorch 原生系统，实现了显著的性能提升。
   - 其特性包括 **可定制的激活检查点（activation checkpointing）**、**fp8 训练支持** 以及内置的故障恢复。
- **关于 FSDP2 的见解引发好奇**：论文深入探讨了 **FSDP2** 技术，引起了急于了解其复杂性的成员们的极大兴趣。
   - 虽然有些人仍在努力理解其概念，但讨论正在促进对其实现的更深入探索。
- **确定了与 TorchTitan 的联系**：成员们将 LegoScale 与 **TorchTitan** 进行了类比，暗示两者可能密切相关甚至是同一个项目。
   - 一位成员幽默地指出它“看起来酷毙了”，而另一位成员确认了它就是 **TorchTitan**。
- **对近期出版物的新兴趣**：该论文的新颖性在社区中引发了兴奋，特别是在被 **Hugging Face** 相关人员在 LinkedIn 上分享之后。
   - 这突显了在大语言模型训练技术的进步方面持续的关注和兴趣。



**提到的链接**：<a href="https://openreview.net/forum?id=SFN6Wm7YBI">LegoScale: One-stop PyTorch native solution for production ready...</a>：大语言模型（LLMs）的发展对于推动最先进的自然语言处理应用起到了至关重要的作用。训练具有数十亿参数和数万亿...

  

---

### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1295520217979555860)** (2 条消息): 

> - `CUDA optimization experts`
> - `Open source training framework` 


- **Together.AI 寻求 CUDA 优化奇才**：Together.AI 正在招聘 **CUDA optimization experts**，负责增强流行模型的 kernel，目标是在所有 GPU 上实现热节流（thermal throttling）。更多详情请见其 [职位列表](https://job-boards.greenhouse.io/togetherai/jobs/4188119007)。
   - 他们强调，针对政府报告的自我身份识别调查的回答是自愿且保密的。
- **招聘开源训练框架职位**：一个团队正在招聘一名开发人员，负责其 **open source training framework**，该框架训练 **starcoder2** 的速度明显快于 **megatron-lm**。感兴趣的候选人可以通过 [ServiceNow 的列表](https://jobs.smartrecruiters.com/ServiceNow/744000019737886-staff-machine-learning-developer)进行申请。
   - 自 2004 年在 Fred Luddy 的愿景下成立以来，ServiceNow 已经改变了组织的工作流程。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://jobs.smartrecruiters.com/ServiceNow/744000019737886-staff-machine-learning-developer">Staff Machine Learning Developer</a>：公司描述：这一切都始于 2004 年加利福尼亚州圣地亚哥的阳光下，当时一位富有远见的工程师 Fred Luddy 预见到了改变我们工作方式的潜力。如今...</li><li><a href="https://job-boards.greenhouse.io/togetherai/jobs/4188119007">Together AI 系统研究工程师、GPU 编程职位申请</a>：未找到描述
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/)** (1 条消息): 

nusha.m: 学习使用 CUDA 或 OpenCL 进行 GPU 编程有哪些好的初学者项目？
  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1295844511787778133)** (1 条消息): 

> - `Matrix Multiplication Kernels`
> - `Shared Memory Performance`
> - `A100 Speed Metrics` 


- **对 A100 矩阵乘法速度的怀疑**：针对第 5 章中 **A100** 上的 **matrix multiplication kernels** 速度提出了质疑，认为所做的假设（如朴素 kernel 没有 L1 cache 且没有 warp stalls）是不切实际的。
   - 有人建议增加一个*关于现实世界性能考量的脚注*，以强调潜在的差异。
- **Kernel 性能需要现实世界的背景**：讨论指出，过去关于 **shared-memory kernel** 极小提升的问题表明，人们对其性能优势存在普遍误解。
   - 成员们一致认为，在承认这些现实挑战的同时，**教学清晰度**至关重要，但也必须与实际性能数据保持一致，以避免误导读者。


  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1295473811755368508)** (8 messages🔥): 

> - `Gradient Accumulation Compatibility` (梯度累积兼容性)
> - `Raspberry Pi Support for TorchAO` (TorchAO 对 Raspberry Pi 的支持)
> - `FP8 Format Performance Differences` (FP8 格式性能差异)
> - `CUTLASS-based W4A8 Kernel PR` (基于 CUTLASS 的 W4A8 Kernel PR)


- **梯度累积遇到了一些障碍**：一位成员询问为什么 **gradient accumulation** 与 `CPUOffloadOptimizer` 中的 `offload=True` 不兼容，这引发了关于将梯度移动到 CPU 所涉及复杂性的技术解释。
   - *技术上是可行的*，但分配额外的内存并将数据传输与计算交错进行会显著增加复杂性。
- **TorchAO 在 Raspberry Pi 上可以运行！**：一位成员发现 **TorchAO** 已经在 Raspberry Pi 上运行，并指向了 [GitHub issue](https://github.com/pytorch/ao/issues/1076)。
   - 他们指出，由于缺乏针对 aarch64 Linux 发布的二进制文件，目前他们安装的是 **0.1** 版本。
- **FP8 格式：性能见解**：一位成员询问了推理过程中 **FP8 E4M3** 和 **E5M2** 格式之间潜在的性能差异，这引发了对其特性的讨论。
   - 另一位成员评论说，虽然延迟不应有差异，但 **range**（范围）和 **precision**（精度）有显著不同，E4M3 能产生更准确的结果。
- **CUTLASS W4A8 Kernel 开发**：一位成员在 [GitHub](https://github.com/pytorch/ao/pull/880) 上分享了他们针对 **CUTLASS-based W4A8 kernel** 的公开 PR，欢迎对剩余问题和性能说明提供反馈。
   - PR 中的评论概述了当前的挑战，并邀请社区贡献见解。



**提到的链接**：<a href="https://github.com/pytorch/ao/issues/1076">torchao already works on raspberry pi · Issue #1076 · pytorch/ao</a>：问题：我们没有发布 aarch64 linux 二进制文件，所以现在我们仍然安装 ao=0.1 (myvenv) marksaroufim@rpi5:~/Dev/ao $ pip install torchao Looking in indexes: https://pypi.org/simple, https://...

  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

frappuccino_o: 想在旧金山一起喝早咖啡并讨论 Image generation（图像生成）吗？
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1295615724475387938)** (9 messages🔥): 

> - `AMD MI250 donation` (AMD MI250 捐赠)
> - `Discord server name change` (Discord 服务器更名)
> - `User feedback for benchmarking` (基准测试的用户反馈)
> - `Compute funding for projects` (项目的算力资助)
> - `Job submission interface concerns` (任务提交界面的疑虑)


- **AMD 捐赠 MI250 节点**：来自 **AMD 的慷慨团队**（包括核心贡献者）向服务器捐赠了一个 **MI250 节点**，并附带了用于基准测试 [Triton kernels](https://github.com/gpu-mode/amd-cluster) 的说明。
   - 鼓励成员提供有关用户体验的反馈，旨在将其改进得更加用户友好。
- **Discord 更名为 GPU Mode**：有人对 **Discord 服务器名称从 CUDA Mode 更改为 GPU Mode** 提出疑问，这表明重心正向 AMD 硬件转移。
   - Mark 表示也欢迎来自其他 **hardware vendors**（硬件厂商）的捐赠，以扩大合作机会。
- **征集贡献者**：Mark 正在寻找 **contributors**（贡献者）来帮助维护任务提交流水线，并在 README 中提供了入门任务。
   - 这旨在加强社区内的协作努力，以获得更好的基准测试体验。
- **算力资助机会**：Mark 提出将有趣项目的 **compute funding**（算力资助）需求转发给潜在赞助商，为资源分配指明了路径。
   - 虽然不是最理想的，但这可以支持成员从事有价值的以计算为导向的项目。
- **对任务提交界面的疑虑**：一位用户对依赖 **GitHub actions** 提交 kernel 基准测试表示担忧，并请求为较长时间的任务提供其他访问方式。
   - Mark 承认了对 **direct SSH access**（直接 SSH 访问）的需求，并指出为一个庞大社区提供可扩展性可能会带来挑战。


  

---

### **GPU MODE ▷ #[arm](https://discord.com/channels/1189498204333543425/1247232251125567609/1295619806237691914)** (3 条消息): 

> - `Ollama 在 Raspberry Pi 上的性能`
> - `构建 TorchAO`
> - `Triton CPU 后端探索` 


- **Ollama 在 Raspberry Pi 5 上运行流畅**：**Ollama 模型**的 **llama3.2** 版本达到了 **5.32 tokens/s** 的可用评估速率，但 **llama3.1** 在 Raspberry Pi 5 上的 **1.5 tokens/s** 明显较慢。
   - 一名成员提到考虑使用 **带有 2080 的 eGPU**，这在 Raspberry Pi 系统上现在是可行的。
- **TorchAO 在 Raspberry Pi 5 上构建成功**：令人惊讶的是，一名成员成功地**从源码构建了 TorchAO**，实现了 **int8** 的动态量化，并使用 **CPU codegen** 进行了编译。
   - 他们引用了一个关于缺少已发布的 aarch64 Linux 二进制文件的 [GitHub issue](https://github.com/pytorch/ao/issues/1076)，强调了社区的权宜之计。
- **探索 Triton CPU 后端和自定义 Kernels**：计划进行的下一步探索包括测试 **Triton CPU 后端**以提升性能。
   - 该成员还表达了对在 TorchAO 中实验用于低比特矩阵乘法的**自定义 Kernels**的兴趣。



**提到的链接**：<a href="https://github.com/pytorch/ao/issues/1076">torchao already works on raspberry pi · Issue #1076 · pytorch/ao</a>：问题是我们没有发布 aarch64 linux 二进制文件，所以目前我们仍然安装 ao=0.1 (myvenv) marksaroufim@rpi5:~/Dev/ao $ pip install torchao Looking in indexes: https://pypi.org/simple, https://...

  

---


### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1295757102521778287)** (3 条消息): 

> - `WebGPU 与 CUDA 的交互`
> - `WebGPU 操作系统兼容性` 


- **WebGPU 不与 CUDA 交互**：一名成员询问 **WebGPU** 是否可以与 **CUDA** 交互，另一名成员澄清说**不可以**。
   - 这表明使用 WebGPU 的开发人员将需要依赖其他 API。
- **WebGPU 对操作系统图形 API 的依赖**：指出 **WebGPU** 根据操作系统的不同使用特定的图形 API，例如 **Vulkan**、**DirectX** 或 **Metal**。
   - 这突显了 WebGPU 实现的平台特定性质。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1295734246823952437)** (2 条消息): 

> - `Distributed Training of Deep Learning Models`
> - `FlashAttention 2022`
> - `Transformer Architecture`
> - `Communication Bottlenecks`
> - `Decentralized Training` 


- **深入探讨 Distributed Training**：该博客系列的 [Part 1](https://vaibhawvipul.github.io/2024/09/29/Distributed-Training-of-Deep-Learning-models-Part-~-1.html) 探讨了深度学习中分布式计算的基础概念，解释了数据流图（dataflow graphs）如何表示计算过程。
   - 它包含了关于通过 forward 和 backward passes 计算 loss 并更新模型权重的见解。
- **解决 Communication Bottlenecks**：[Part 2](https://vaibhawvipul.github.io/2024/10/03/Distributed-Training-of-Deep-Learning-models-Part-~-2.html) 讨论了分布式训练中带宽和延迟的挑战，强调了它们对 parameter servers 与 workers 之间通信的影响。
   - 这种复杂性通常在参数更新期间处理大型模型和大量 worker 时出现。
- **探索 Decentralized Training 方法**：[Part 3](https://vaibhawvipul.github.io/2024/10/15/Decentralized-Training-of-Deep-Learning-Models.html) 将重点转向非同地计算的去中心化训练，并借鉴了之前关于分布式训练的讨论。
   - 该文章根据 [Scaling laws for Neural Language Models](https://arxiv.org/pdf/2001.08361.pdf) 的研究结果，强调了模型的扩展性（scalability）。
- **FlashAttention 引起轰动**：一篇新博客文章解释了 [FlashAttention (2022)](https://www.digitalocean.com/community/tutorials/flashattention)，这是一种增强神经网络中 Transformer 注意力机制效率的技术。
   - 它旨在减轻与长序列相关的 O(n^2) 时间和内存复杂度，未来的文章将讨论其后续迭代版本。
- **Transformers 及其影响**：强调了 Transformer 架构的重要性，特别是其利用 self-attention 的能力，详见开创性论文：[Attention is All You Need](https://arxiv.org/abs/1706.03762)。
   - 这种架构彻底改变了 AI 研究，重点在于提高效率以及在各个领域的应用。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.digitalocean.com/community/tutorials/flashattention">Designing Hardware-Aware Algorithms: FlashAttention | DigitalOcean</a>: 未找到描述</li><li><a href="https://vaibhawvipul.github.io/2024/09/29/Distributed-Training-of-Deep-Learning-models-Part-~-1.html">Distributed Training of Deep Learning models - Part ~ 1</a>: 注：这篇文章是深度学习模型分布式训练系列的一部分。现在，这更像是我的笔记，所以你会发现直接从各个地方复制的内容。此外，我使用了 AI 模式...</li><li><a href="https://vaibhawvipul.github.io/2024/10/03/Distributed-Training-of-Deep-Learning-models-Part-~-2.html">Distributed Training of Deep Learning models - Part ~ 2</a>: 注：我使用 AI 模型辅助写作。现在，这更像是我的笔记，所以你会发现直接从各个地方复制的内容。如果你发现任何错误，请随时纠正我！</li><li><a href="https://vaibhawvipul.github.io/2024/10/15/Decentralized-Training-of-Deep-Learning-Models.html">Decentralized Training of Deep Learning Models</a>: 在之前的文章中，我们讨论了深度学习模型的分布式训练 Part 1 和 Part 2。现在，在这篇文章中，我们将深入探讨深度学习模型的去中心化训练。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[diffusion](https://discord.com/channels/1189498204333543425/1288899271193526342/1295581003909435402)** (3 条消息): 

> - `Text Inversion for SDXL`
> - `Training Challenges`
> - `Hugging Face Community` 


- **SDXL 上的 Text Inversion 问题**：一位成员询问了训练 **SDXL** 的 **Text Inversion** 的经验，提到他们尝试了各种 prompt 和 dropout 设置但没有成功。
   - *他们对 Civit.ai 上缺乏可用的社区模型表示沮丧*，暗示了 SDXL 架构的潜在局限性。
- **建议替代支持渠道**：另一位成员建议在 Hugging Face 服务器的 **'diffusion models/discussion'** 频道寻求指导，以获得更好的支持。
   - 他们建议分配 **@diffusers** 角色以访问该频道，这表明有可能获得更有针对性的社区帮助。


  

---

### **GPU MODE ▷ #[avx](https://discord.com/channels/1189498204333543425/1291829797563011227/1295788677900927058)** (1 条消息): 

> - `张量并行操作中的数据传输`
> - `用于 LLM 推理的 CPU 集群`
> - `AVX 加速`
> - `测量带宽需求` 


- **测量张量操作的数据传输**：一位用户询问了在搭建 CPU 集群时，测量**张量并行操作 (tensor parallel operations)** 所需的**数据传输**和**带宽**的有效方法。
   - *他们提到了几个需要考虑的因素*，例如矩阵大小、计算能力、内存带宽和计算精度。
- **搭建用于 LLM 推理的 CPU 集群**：用户的目标是创建一个带有 **AVX 加速**的 **LLM 推理** CPU 集群，旨在测试可行性而非实用性。
   - 他们对设置所需的**网络配置 (network provisioning)** 并不确定，并寻求关于估算带宽的建议。


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1295504494829899856)** (64 条消息🔥🔥): 

> - `实时语音转文本 (STT) 引擎`
> - `线性注意力模型`
> - `设计中的 AI`
> - `AI 初创公司的近期融资`
> - `开源项目的文档外包` 


- **实时 STT 引擎彻底改变 AI 转录**：Gladia 的新型实时 STT 引擎具有 < 300 毫秒的延迟，支持 100 多种语言和代码切换 (code-switching)，在 1600 万美元 A 轮融资的支持下，为实时 AI 树立了标准。
   - 另一个引擎宣称具有 90 毫秒的推理时间并支持多种语言，表明转录技术正在经历竞争激烈的演进。
- **使用线性注意力模型增强转换机制**：围绕 LoLCATS 的讨论显示出前景，它使 Llama 3.1 模型系列线性化，在比传统方法消耗更少资源的同时带来了显著改进。
   - 对话还探讨了将超过 50% 的 Transformer 模型注意力层转换为线性层时面临的挑战和潜在问题。
- **将 AI 的抽象比作设计材料**：一篇博客文章将 AI 与塑料进行了类比，认为由于 AI 快速集成到各个领域，可以将其视为一种新型建筑材料。
   - 讨论强调了过去的设计时代如何改变了行业，导致今天软件和信息的重要性超过了物理材料。
- **AI 初创公司融资近期激增**：DecagonAI 宣布获得 6500 万美元 B 轮融资，引发了人们对投资趋势的好奇，特别是针对 AI 应用层而非底层模型。
   - 其他提到的内容包括 AI 领域内显著的融资活动，反映了人们对 AI 驱动型应用日益增长的兴趣。
- **外包 AI 和开源项目的文档**：目前正在讨论为开源项目外包文档和用户指南编写的可行性，重点关注 AI 工程。
   - 社区成员权衡了利用 LLM 与聘请人员编写全面文档的优缺点。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.twitch.tv/videos/2275103790">Twitch</a>: 未找到描述</li><li><a href="https://x.com/simran_s_arora/status/1845909074774475125?s=46&t=PW8PiFwluc0tdmv2tOMdEg">Simran Arora (@simran_s_arora) 的推文</a>: 想要 Llama 405B，但希望它在序列长度上呈线性扩展吗？？？试试 LoLCATS：一种在学术预算范围内将 "Transformers 转换为线性注意力模型" 的高效方法！！我们使...</li><li><a href="https://www.notion.so/blog/ai-is-the-new-plastic">AI 是新的塑料</a>: 塑料存在于你的汽车、厨房以及你现在坐的椅子中。正如这种无处不在的材料在几十年内改变了世界一样，AI 也在做同样的事情。</li><li><a href="https://x.com/thejessezhang/status/1846235369886589197?s=46">Jesse Zhang (@thejessezhang) 的推文</a>: 非常激动地宣布 Decagon 完成了由 Bain Capital Ventures 的 @aaref 领投的 6500 万美元 B 轮融资，参与方包括 @eladgil, A*, Accel, BOND Capital 等。🎉 这使我们在 @DecagonAI 的总融资额达到 1...</li><li><a href="https://x.com/jilijeanlouis/status/1846145881285730338">🎙Jean-Louis Queguiner (@JiliJeanlouis) 的推文</a>: 我们的全新实时 STT 引擎发布了！🔥它兼具两者的优点：批处理级别的质量与实时的转录速度。延迟低于 300 毫秒，支持 100 多种语言，支持语码混用（code-switching）...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1g2vhy3/creating_very_highquality_transcripts_with/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://x.com/thesephist/status/1846029340867285158?s=46">Linus (@thesephist) 的推文</a>: 在 @NotionHQ 任职期间，我从设计和架构的角度（而非仅仅是技术）深入思考了 AI 的未来。我在最后几周写了一篇关于...</li><li><a href="https://x.com/_mfelfel/status/1846025183993511965?s=46">Felfel (@_mfelfel) 的推文</a>: 我们刚刚发布了最新的模型，这是有史以来最快的 Text-to-Speech 模型。90ms 推理时间。120-150ms TTFB（推理 + 网络）。我们在实现这一目标的同时，还将可靠性提升了 4 倍。引用 Pl...</li><li><a href="https://github.com/go-go-golems/go-go-labs/blob/3dd082b2406799ba8233b78b9f788c753486bafc/cmd/apps/catter/README.md">go-go-labs/cmd/apps/catter/README.md at 3dd082b2406799ba8233b78b9f788c753486bafc · go-go-golems/go-go-labs</a>: GO GO 实验实验室。通过在 GitHub 上创建账号来为 go-go-golems/go-go-labs 的开发做出贡献。</li><li><a href="https://share.snipd.com/episode/367fa309-f098-496f-874c-0387b1dff367">多邻国 CEO Luis Von Ahn 希望你沉迷于学习</a>: 多邻国 CEO Luis Von Ahn 希望你沉迷于学习</li><li><a href="https://github.com/go-go-golems/go-go-labs/blob/3dd082b2406799ba8233b78b9f788c753486bafc/python/photo-dewarp/ttmp/2024-10-11/03-tps-dewarp.md">go-go-labs/python/photo-dewarp/ttmp/2024-10-11/03-tps-dewarp.md at 3dd082b2406799ba8233b78b9f788c753486bafc · go-go-golems/go-go-labs</a>: GO GO 实验实验室。通过在 GitHub 上创建账号来为 go-go-golems/go-go-labs 的开发做出贡献。</li><li><a href="https://github.com/wesen/glazed/blob/task/add-docs-for-commands/pkg/doc/tutorials/03-commands-tutorial.md">glazed/pkg/doc/tutorials/03-commands-tutorial.md at task/add-docs-for-commands · wesen/glazed</a>: 一个让你的命令行工具能够轻松输出结构化数据的库。为你的数据锦上添花 - wesen/glazed</li><li><a href="https://github.com/wesen/glazed/blob/task/add-docs-for-commands/pkg/doc/topics/15-using-commands.md">glazed/pkg/doc/topics/15-using-commands.md at task/add-docs-for-commands · wesen/glazed</a>: 一个让你的命令行工具能够轻松输出结构化数据的库。为你的数据锦上添花 - wesen/glazed</li><li><a href="https://github.com/go-go-golems/oak">GitHub - go-go-golems/oak: GO GO 解析你的代码 GO GO</a>: GO GO 解析你的代码 GO GO。通过在 GitHub 上创建账号来为 go-go-golems/oak 的开发做出贡献。</li><li><a href="https://github.com/go-go-golems/prompto">GitHub - go-go-golems/prompto: 快速获取自定义 Prompt 上下文</a>: 快速获取自定义 Prompt 上下文。通过在 GitHub 上创建账号来为 go-go-golems/prompto 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1295540850738335838)** (2 条消息): 

> - `使用 Claude 3.5 Sonnet 的 Financial Agent`
> - `使用 MariaDB 构建 AI 应用`
> - `SkySQL 集成`
> - `智能产品评论分析` 


- **使用 Claude 3.5 构建 Financial Agent**：学习如何使用 [@financial_mod 的 API](https://twitter.com/llama_index/status/1845980793593831845) 获取股票价格和公司数据，创建一个由 **Claude 3.5 Sonnet** 驱动的 **Financial Agent**。
   - 正如 Hanane Dupouy 所述，该 Agent 可以提供多种财务见解，包括损益表和全面的公司信息。
- **通过 SkySQL 进行高效的 AI 应用开发**：如果你对在 AI 应用中使用 MySQL/MariaDB 感兴趣，[@skysql](https://twitter.com/llama_index/status/1846274668040540389) 提供了在 SkySQL 中设置 **MariaDB Vector** 的基本指令。
   - 该指南包括将 **OpenAI 的 LLM** 与 LlamaIndex 集成，并构建一个**智能产品评论分析系统**。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1295468905711534080)** (57 条消息🔥🔥): 

> - `Qdrant 添加节点时报错`
> - `PineconeVectorStore 问题`
> - `Neo4jPropertyGraphStore 性能` 


- **Qdrant 添加节点触发错误**：一位成员报告称，在尝试向 **Qdrant** 索引添加新节点时遇到错误，而此前从未出现过此类问题。
   - 另一位成员回应称他们没有遇到过该错误，暗示可能是设置问题。
- **PineconeVectorStore 在 ComposableMemory 中失败**：一位成员在将 **PineconeVectorStore** 与 `SimpleComposableMemory` 配合使用时表达了挫败感，收到了 “Namespace not found” 错误消息。
   - 另一位用户推测可能是 Pinecone 的设置问题导致了该故障。
- **Neo4jPropertyGraphStore 初始化性能滞后**：一位用户讨论了从现有图创建 **Neo4jPropertyGraphStore** 时遇到的显著延迟，称其 Schema 生成速度极其缓慢。
   - 他们提到已经分配了最大内存，并引用了 GitHub 线程中报告的一个类似问题，即 `refresh_schema()` 函数在处理大型图时表现不佳。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/run-llama/llama_index/issues/16204">[Bug]: 大型图初始化 Neo4jPropertyGraphStore 耗时极长 · Issue #16204 · run-llama/llama_index</a>：Bug 描述：初始化包含 3558 个实体的图存储大约需要 14 分钟。我觉得这是因为 refresh_schema() 对大型图的处理不好。也许是没有使用 async？我粘贴了...</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/memory/composable_memory/">Simple Composable Memory - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/pull/16559">由 kilimchoi 修复 upstash_chat_store 中的 pydantic 错误 · Pull Request #16559 · run-llama/llama_index</a>：描述：目前你会收到错误消息 AttributeError: &#39;UpstashChatStore&#39; object has no attribute &#39;_sync_redis_client&#39;。修复了 # (issue)。在 pydantic v2 中，你有...</li><li><a href="https://github.com/run-llama/llama_index/blob/e2dca8bb021b36b8eaf38be953cb2496f029d680/llama-index-integrations/vector_stores/llama-index-vector-stores-qdrant/llama_index/vector_stores/qdrant/base.py#L300">llama_index/llama-index-integrations/vector_stores/llama-index-vector-stores-qdrant/llama_index/vector_stores/qdrant/base.py · run-llama/llama_index</a>：LlamaIndex 是一个用于你的 LLM 应用的数据框架 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/loading/documents_and_nodes/usage_nodes/#defining-and-customizing-nodes>):">Using Nodes - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/storage/vector_store/postgres/#llama_index.vector_stores.postgres.PGVectorStore>).">Postgres - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1295516909227347978)** (1 条消息): 

> - `Llama 3.1-70B 集成`
> - `响应被截断`
> - `Max tokens 问题`
> - `软件工程技能列表`
> - `使用统计分析` 


- **Llama 3.1-70B 集成面临截断困扰**：尽管调整了 **max_tokens** 等参数，**Llama 3.1-70B** 的集成仍遇到响应被截断的问题。
   - 在请求 **20 个软件工程技能列表**时，应用程序始终只返回 **5 个技能**，且总是以 `finish_reason: max_tokens` 结束。
- **Max Tokens 管理不当**：用户报告了一个问题，即无论 Prompt 的复杂度如何，响应完成度都被限制在 **100 tokens**。
   - 对 **max_tokens** 和 **completion_tokens** 的调整未能解决该问题，表明存在潜在的限制。
- **详细使用统计数据困惑**：报告的使用统计数据显示 **71 prompt tokens** 以及 **100 completion tokens**，总计 **171 tokens**。
   - 这引发了关于在与模型交互过程中，配置错误或限制源自何处的疑问。


  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1295527484342669332)** (46 条消息🔥): 

> - `LLM 推理局限性`
> - `Swarm 库见解`
> - `GPT 语音功能查询`
> - `有争议的 AI 实践`
> - `自我推广规则` 


- **LLM 在推理中表现出脆弱性**：Apple 工程师最近的一项研究强调了 **LLM** 在数学推理中的脆弱性，表明它们利用的是概率模式匹配而非真正的逻辑推理，当基准测试稍作修改时就会导致错误。
   - 成员们讨论了人类基准对比的必要性，并指出了研究中对推理定义的客观性。
- **关于 Swarm 库功能的见解**：用户围绕 **Swarm** 库的讨论揭示了在辨别是 Agent 还是基础 LLM 在执行任务方面的挑战，强调了编写有效测试的重要性。
   - 针对 Swarm 作为非生产级工具的状态提出了担忧，并提到了现有的 fork 版本和等效工具如 **Swarm.js**。
- **关于 GPT 语音功能的咨询**：一位用户询问了高级 **GPT voice** 功能在网页端的推出情况，回复强调了目前缺乏关于其功能的官方公告。
   - 共享了对先前版本不受支持状态的担忧，对未来的更新持怀疑态度。
- **AI 融资中的争议做法**：讨论涉及了获取 AI 融资的两条路径：通过炒作过度承诺，或倾向于怀疑论，这呼应了对近期发表的研究质量的看法。
   - 成员们分享了对各位研究人员的看法以及 AI 领域煽动性出版物的影响。
- **自我推广与服务器规则**：强调了 Discord 服务器关于自我推广规则的执行，提醒用户不要发布推广消息以避免潜在的管理问题。
   - 这引发了对社区准则的提醒，并鼓励在适当的频道中进行知识共享。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arstechnica.com/ai/2024/10/llms-cant-perform-genuine-logical-reasoning-apple-researchers-suggest/">Apple 研究揭示了 LLM “推理”能力的深度缺陷</a>：无关的干扰因素导致逻辑推理的“灾难性”失败。</li><li><a href="https://arstechnica.com/ai/2024/10/llms-cant-perform-genuine-logical-reasoning-apple-researchers-sug">类别：AI</a>：打开舱门……
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1295482202519506976)** (2 条消息): 

> - `自定义 GPT PDF 集成`
> - `PDF 性能问题` 


- **自定义 GPT 卡在 'Update Pendings'**：一位成员报告称，他们使用 **300 页**建筑规范材料构建的自定义 GPT 已卡在 'Update Pendings' 状态超过一周，尽管已将 PDF 拆分为 **6 个较小的文件**（每个 **50-60 页**）。
   - 他们指出，虽然机器人承认 PDF 的存在，但它经常将问题重定向回规范原文，而不是直接从文档中提供答案。
- **使用单个 PDF 测试 GPT**：另一位成员在 GPT-4 中使用仅 **1 个 PDF** 测试了新对话，但遇到了类似的性能问题，导致人们猜测机器人有效读取 PDF 的能力。
   - 这表明 GPT 处理 PDF 内容的方式可能存在潜在问题，从而影响了其响应能力。

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1295466333982425192)** (33 条消息🔥): 

> - `Configuration Options`
> - `Program Installation Concerns`
> - `Character Cards in LLMs`
> - `Uncensored LLM Platforms`
> - `SageAttention Efficiency` 


- **增强配置选项**：一位成员建议以截图以外的格式提供配置详情，以便于使用。
   - 另一位成员认可了这一反馈，并表示未来的博客文章将解决这个问题。
- **无需 UAC 提示的安装**：一位 IT 支持工程师对员工在没有用户账户控制 (UAC) 提示输入凭据的情况下安装程序的能力表示担忧。
   - 有人请求确认该程序是将文件安装在系统机器上还是仅安装在用户配置文件（User Profile）中。
- **对 LLM Character Cards 的兴趣**：一位成员表达了将 'character cards' 引入 LM Studio 的热情，用于创建具有独特个性和交互能力的 Agent。
   - 他们询问了在 LLM 之间创建对话以增强功能的可能性。
- **对无审查 LLM 方案的需求**：一位成员寻求关于类似 LM Playground 且提供无审查（Uncensored）模型的平台信息。
   - 另一位成员建议先检查已有的相关讨论，以避免重复。
- **SageAttention 具有变革性的潜力**：一位成员强调了最近关于 SageAttention 的论文，该论文展示了 Attention 机制在效率上的显著提升。
   - 他们指出，如果能在 llama.cpp 和 MLX 中实现，它可能会使 Token 处理速度翻倍，从而带来性能变革。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2410.02367">SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration</a>: Transformer 架构在各种模型中占据主导地位。作为 Transformer 的核心，Attention 的计算复杂度为 O(N^2)，而线性变换为 O(N)。当 ...</li><li><a href="https://github.com/openai/swarm">GitHub - openai/swarm: Educational framework exploring ergonomic, lightweight multi-agent orchestration. Managed by OpenAI Solution team.</a>: 探索人体工程学、轻量级多 Agent 编排的教育框架。由 OpenAI Solution 团队管理。 - openai/swarm</li><li><a href="https://github.com/victorb/ollama-swarm">GitHub - victorb/ollama-swarm: Educational framework exploring ergonomic, lightweight multi-agent orchestration. Modified to use local Ollama endpoint</a>: 探索人体工程学、轻量级多 Agent 编排的教育框架。已修改为使用本地 Ollama 端点。 - victorb/ollama-swarm</li><li><a href="https://github.com/ml-explore/mlx-examples/pull/1027">Clear cache during prompt processing by awni · Pull Request #1027 · ml-explore/mlx-examples</a>: 关闭了 #1025，相关讨论/改进请参阅该条目。
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1295476762213355581)** (13 条消息🔥): 

> - `M2 Studio Performance`
> - `GPU Under-volting`
> - `Tokens Per Second (TPS) with Llama 8B`
> - `Performance Discrepancies between GPUs` 


- **M2 Studio 在 Mistral large 模型上表现出色**：一位用户称赞了配备 **192 GB RAM** 的 **M2 Studio** 及其在 **Mistral large 128K context** 下的表现，称其为最适合他们用例的模型。
   - *“对于我的用例来说，这是一个非常好的模型”* 强调了该模型对特定应用的适用性。
- **对 GPU 进行降压以获得更好性能**：一位成员建议使用 **Afterburner** 对 **GPU 进行降压 (UV)**，因为即使是 **100mV** 的调整也能带来显著的性能提升。
   - 他们建议在 **YouTube** 上搜索特定显卡型号加上关键词 **UV** 和 **Afterburner** 来获取有用的指南。
- **关于 Llama 8B TPS 的讨论**：一位参与者观察到，一些用户声称在 **4080** 等较新的 GPU 上运行 **Llama 8B** 模型可达到 **30 TPS**，而他们在 **1070Ti** 上运行速度也是 **30 TPS**。
   - 他们表达了希望达到 **150+ TPS** 的愿望，并询问需要进行哪些升级。
- **理解 GPU 性能差异**：讨论表明，模型大小、Quantization 和 Context 等因素会严重影响性能，不同用户的配置之间存在差异。
   - 具体而言，有人指出必须使用 **Tensor Cores** 才能匹配 **4080** 等高级 GPU 的性能，并强调了配置和模型使用的重要性。
- **以不错的 TPS 运行 Llama 8B Q6_K**：另一位用户报告称运行 **Llama 8B Q6_K**，最大 Context 为 **15k**，目前可达到 **28-35 TPS**。
   - 他们一次使用大约 **10k tokens**，表明了其配置的效率。


  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1295503281673601144)** (11 messages🔥): 

> - `Cohere Connector 使用`
> - `Cohere API Token 限制` 


- **Cohere Connector 在输入 'hi' 时触发搜索**：一位用户对 **Cohere Connector** 表示沮丧，称即使只是简单地说 'hi'，它也会在不需要的情况下执行搜索。
   - 他们询问是否有办法控制此功能，并仅在必要时使用该 connector。
- **对 API Token 限制的困惑**：一位用户质疑 **Cohere API** 的 Token 限制有效性，指出每月 **10k tokens** 与 Cohere chat 所说的 **500 万 tokens** 之间存在差异。
   - 他们询问如果超过 **10k tokens** 是否会产生费用。


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1295548996470833164)** (8 messages🔥): 

> - `Google Connector 问题`
> - `Command 模型定价`
> - `C4AI 项目协作`
> - `LLM 中的日期注入`
> - `通讯（Newsletters）的重排序流程` 


- **Google Connector 故障排除**：几位成员报告 **Google Connector** 无法正常工作，并正在互相寻求解决方案。
   - 一位成员建议在继续排查连接问题时分享突破性进展。
- **了解 Command 模型定价结构**：讨论显示，使用 **web-search connector** 本身没有费用，但传递给 **Command** 输入上下文的结果是需要收费的。
   - 这一澄清有助于明确流程中可能产生费用的环节。
- **C4AI Discord 中的协作机会**：一位成员鼓励其他人查看 **C4AI Discord**，以获取正在寻求合作者的项目信息，并提供了加入链接。
   - 这为社区内各个项目之间的协作开辟了途径。
- **优化 LLM 中的日期使用**：一位成员讨论了尽管已经为其流程实现了日期注入，但在 **final** 调用中使用绝对日期的可能性。
   - 他们建议对于相对日期，结合 tool use 实现元数据过滤可能会增强结果。
- **通讯（Newsletters）重排序工作流**：概述了一个处理通讯的结构化流程，包括抓取、使用 **Cohere Rerank** 进行重排序，并保留前几名结果作为 LLM 上下文。
   - 该方法的功能包括分析 **30 份通讯**，并根据新的排名系统筛选出 **前 10 名**。



**提及链接**：<a href="https://cohere.com/blog/rerank-3#searching-over-json">Rerank 3 介绍：一种用于高效企业搜索与检索的新基础模型</a>：今天，我们推出了最新的基础模型 Rerank 3，专为增强企业搜索和检索增强生成（RAG）系统而构建。我们的模型与任何数据兼容...

  

---

### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1295648961448771595)** (13 条消息🔥): 

> - `Cohere Tool Calls`
> - `工具使用的 Curl 示例`
> - `关于参数定义的疑问`
> - `API 中的请求处理`
> - `原始请求示例` 


- **Cohere Tool Calls 简化工作流**：讨论强调了在工具调用之后，Assistant 的消息应该提示更多的工具调用或提供响应。
   - 成员们对增强与外部工具交互的 **function calling** 功能表示赞赏。
- **提供了 Curl 示例**：一位成员分享了一个简洁的 [curl 示例](https://docs.cohere.com/docs/tool-use#the-four-steps-of-tool-use-step-by-step-example)，展示了如何正确使用 API 获取当前天气。
   - 这受到了好评，因为它为不使用 Python 的用户提供了实用的解决方案。
- **参数问题澄清**：成员们理清了 API 代码片段中关于参数定义的困惑，强调了使用 v1 风格格式。
   - 建议的一个关键修复是将第一条消息的角色从 'tool' 更改为 'assistant'，这解决了该问题。
- **对原始请求示例的需求**：有人呼吁在提供库示例的同时提供原始请求示例，以方便用户理解。
   - 一位成员分享了一篇 [Medium 文章](https://medium.com/@smallufo/anthropic-claude-function-call-example-7355e6ec6fa2)，举例说明了这一需求。
- **Function Calling 能力受到关注**：成员们对 **Cohere** 高效管理多个函数调用的能力感到兴奋。
   - 这一能力被认为对增强应用程序工作流非常有益。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://medium.com/@smallufo/anthropic-claude-function-call-example-7355e6ec6fa2">Anthropic Claude Function Call Example</a>：请求</li><li><a href="https://docs.cohere.com/docs/tool-use#the-four-steps-of-tool-use-step-by-step-example">Tool Use — Cohere</a>：让你的 LLM 连接外部工具，实现更高级和动态的交互 (V2)。
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1295800987965063169)** (2 条消息): 

> - `OrionChat`
> - `聊天 AI 模型对比` 


- **推出用于多模型对话的 OrionChat**：一位成员介绍了一个名为 **OrionChat** 的个人项目，这是一个 Web 界面，将来自 **Cohere**、**OpenAI**、**Google** 等的各种聊天 AI 模型整合到一个平台中，方便访问。
   - 他们鼓励社区通过 [此链接](https://orionchat.github.io) 探索该界面，并提供反馈以增强用户体验。
- **在一个地方探索聊天 AI 能力**：新界面使用户能够聊天并比较各种 AI 模型，而无需在多个标签页或网站之间切换，简化了探索过程。
   - 开发者对社区的意见表示期待，旨在根据用户交互来完善项目。



**提到的链接**：<a href="https://orionchat.github.io">OrionChat - AI 模型聊天界面。</a>：OrionChat - IA 模型聊天界面

  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1295467206364364900)** (34 条消息🔥): 

> - `WordPress 插件开发`
> - `Stable Diffusion 的 CORS 问题`
> - `AI 创意社区参与度`
> - `用于风格迁移的文本生成模型`
> - `Discord 服务器活跃度` 


- **针对 Stable Diffusion 的 WordPress 插件开发**：一名成员正在为文本生成和 txt2img 服务器开发多个 WordPress 插件，寻求社区的反馈和测试。
   - *无人回应*，导致对 AI Discord 服务器中社区参与度的挫败感。
- **Stable Diffusion 设置中的 CORS 问题**：用户讨论了在反向代理设置下运行的 Stable Diffusion 服务器使用 SSL 的挑战，遇到了 CORS 错误。
   - 一位成员强调了同一台机器上的 Web 服务器与 Stable Diffusion 服务器之间集成的功能性。
- **寻找活跃的 AI 社区**：一位成员对当前 AI Discord 服务器缺乏活力表示失望，询问有关 ComfyUI 和 A1111 更活跃社区的建议。
   - 他们指出关于插件的咨询无人回答，表明需要到其他地方寻求更好的互动。
- **探索文本生成的基座模型**：一位用户询问在风格迁移任务中表现更好的文本生成基座模型，并提到了他们在 i2i 和 SD1.5 方面的经验。
   - 另一位成员建议尝试 **Flux** 或 **SD3** 以提高文本生成质量，但指出 SD3 在人物表现方面存在困难。
- **风格化照片的技术**：讨论了创建风格化照片的方法，建议包括使用 **ControlNets** 以及[此处](https://github.com/songrise/Artist)描述的特定方法。
   - 成员们分享了实现各种艺术风格（如 pin-up）的技术，强调了创意方法。



**提到的链接**：<a href="https://WandAI.app,">未找到标题</a>：未找到描述

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1295652057167954032)** (14 条消息🔥): 

> - `Tinygrad .dot 操作对比`
> - `VIZ UI 改进`
> - `相对于 PyTorch 的性能`
> - `预订 Pro 设备` 


- **Tinygrad .dot 操作 vs NumPy**：一位用户对比了 Tinygrad 的 .dot 操作与 NumPy 的 matmul，发现随着矩阵尺寸增大，**精度会下降**，并指出在 M=16384, N=8192, K=1280 等大尺寸下，差异达到 **±0.001**。
   - 对于较小的矩阵（M=10, N=4, K=5），差异极小，不超过 **±0.000001**。
- **VIZ UI 改进讨论**：一位成员分享了关于 VIZ UI 改进的 [Issue #7067](https://github.com/tinygrad/tinygrad/issues/7067) 链接，包括自动滚动功能。
   - 该 Issue 讨论了改进侧边栏导航，同时确保左右侧边栏均可调整大小和折叠。
- **George Hotz 谈论相对于 PyTorch 的性能**：George 强调，如果 Tinygrad 能在 NVIDIA GPU 上击败 **PyTorch 的性能**，那将是该项目的重大胜利。
   - 他提到实现性能对等将释放巨大潜力，并断言“我们只要在性能上击败 PyTorch，我们就赢了”。
- **对 Pro 设备发货日期的好奇**：用户对预订 **Pro 设备** 表现出兴趣，并询问 12 月的发货日期。
   - 一位成员专门询问发货日期，在做决定前寻找更多细节。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/_xjdr/status/1846210719421026645">来自 xjdr (@_xjdr) 的推文</a>：jax + GPU 的性能与等效的 PyTorch 实现相比非常糟糕。这令人遗憾但并不意外。jax 可能在一段时间内（对我而言）仅限于 TPU。这...</li><li><a href="https://github.com/tinygrad/tinygrad/issues/7067">VIZ UI 改进 · Issue #7067 · tinygrad/tinygrad</a>：使用向下箭头时自动滚动左侧 kernel 列表。示例 VIZ=1 python3 -m pytest test/test_schedule.py sc.mov 左右侧边栏目前均可调整大小和折叠...
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1295501552240562177)** (16 messages🔥): 

> - `JIT 与 Tensor Reshape`
> - `不放回的多项式采样 (Multinomial Sampling Without Replacement)`
> - `Tinygrad 中的 TD-MPC 实现`
> - `禁用梯度计算`
> - `在 Tinygrad 中添加新的加速器 (Accelerators)` 


- **JIT 在处理 Tensor Reshape 时存在困难**：有人质疑 `.reshape` 是否应该被 JIT 化，并指出 JIT 仅运行 GPU kernels，而不执行 Python 代码。
   - 针对动态形状 (dynamic shapes)，分享了一个具体的实现链接，强调了为了兼容性而调整代码结构的必要性。
- **多项式采样错误解析**：一位用户询问了关于不放回采样时 multinomial 函数相关的错误，这引起了关于 replacement 参数的困惑。
   - 代码片段中的 assertion 表明，不放回采样仅支持单次采样，从而澄清了预期的功能。
- **Tinygrad 中成功实现 TD-MPC 学习**：一位用户报告了在 Tinygrad 中实现 TD-MPC 学习的消息，并对在硬件上进行测试表示兴奋。
   - 提供了相关 GitHub 仓库的链接，并强调了运行该项目所需的硬件要求。
- **禁用梯度计算的方法**：关于禁用梯度计算的讨论表明，在实践中仍在使用将 `Tensor.no_grad` 设置为 True 的方法。
   - 同时介绍了使用 `with Tensor.test():` 作为一种现代且可能更受推荐的控制梯度计算的方式。
- **添加加速器的指南**：一位用户提供了在 Tinygrad 中添加新加速器的指南，并引用了相关资源以供进一步阅读。
   - 该贴旨在澄清算子 (operations) 的集成以及支持新硬件扩展所需的底层架构。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://mesozoic-egg.github.io/tinygrad-notes/addingaccelerator.html">How to add a custom accelerator?</a>：tinygrad 相关教程</li><li><a href="https://github.com/nicklashansen/tdmpc2/tree/main">GitHub - nicklashansen/tdmpc2: Code for &quot;TD-MPC2: Scalable, Robust World Models for Continuous Control&quot;</a>：&quot;TD-MPC2: Scalable, Robust World Models for Continuous Control&quot; 的代码实现 - nicklashansen/tdmpc2</li><li><a href="https://github.com/tinygrad/tinygrad/blob/master/examples/whisper.py#L107>">tinygrad/examples/whisper.py at master · tinygrad/tinygrad</a>：你喜欢 pytorch？你喜欢 micrograd？那你一定会爱上 tinygrad！❤️ - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/blob/master/examples/whisper.py#L36-L45>">tinygrad/examples/whisper.py at master · tinygrad/tinygrad</a>：你喜欢 pytorch？你喜欢 micrograd？那你一定会爱上 tinygrad！❤️ - tinygrad/tinygrad
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1295507378069962853)** (28 条消息🔥): 

> - `Mojo 库安装`
> - `测试自定义 stdlib`
> - `图像哈希算法`
> - `Mojo 中的内存管理` 


- **解决库安装问题**：一位用户发现缺失的库可以通过 `sudo apt-get install libtinfo-dev` 安装，这可能会帮助遇到相同问题的其他用户。
   - 这突显了在社区中分享解决方案对于帮助面临类似挑战的人的重要性。
- **在 Mojo 中测试自定义 stdlib**：在按照构建说明操作后，一位用户在运行修改版的 stdlib 时遇到问题，原始实现仍然会出现。
   - 另一位用户建议了一个涉及构建过程的变通方法，以便在解决这些问题的同时继续推进。
- **探索更新的图像哈希算法**：一位用户质疑像 pHash 这样较旧的图像哈希算法的相关性，寻求更现代方法的建议。
   - 随着技术的进步，他们对更先进的选择表现出兴趣，因为他们觉得现有的算法可能已经过时。
- **理解 Mojo 中的内存管理**：讨论中提到了 Mojo 中的及早销毁（eager destruction）机制，即一个 struct 实例在断言调用期间被提前销毁。
   - 建议为 struct 成员实现一个 getter，以便在不触发提前销毁的情况下安全地访问数据，从而改进内存处理。
- **Bug 报告的成功协作**：一位用户提出了关于 Mojo 中字符串插值的问题，随后确认该问题已在最新版本中得到解决。
   - 这种协作努力突显了社区支持在高效处理和修复 Bug 方面的价值。



**提到的链接**: <a href="https://github.com/modularml/mojo/issues/3672">[BUG] Module Name Collision: Function Cannot Be Called When Named After Module · Issue #3672 · modularml/mojo</a>: Bug 描述：我在 Mojo 标准库中遇到了一个问题，即与所属模块同名的函数无法被调用。具体来说，在实现 ti...

  

---



### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1295465501899751485)** (20 条消息🔥): 

> - `Sebastien Bubeck 加入 OpenAI`
> - `o1-turbo-mini 基准测试`
> - `AGI 论文讨论`
> - `OpenAI 对律师的影响` 


- **Sebastien Bubeck 离职加入 OpenAI**：Microsoft 顶尖的 AI 研究员之一 **Sebastien Bubeck** 正准备加入 OpenAI，引发了关于 AI 人才流动的讨论。
   - 该新闻的详细内容见 [The Information](https://www.theinformation.com/briefings/microsoft-ai-researcher-sebastien-bubeck-to-join-openai?rc=c48ukx) 的一篇文章。
- **o1-turbo-mini 表现出令人印象深刻的结果**：有传言称 **o1-turbo-mini** 在基准测试中表现得异常出色，引发了社区的怀疑和幽默。
   - 成员们一致认为，利用这一新兴消息来调侃那些过度活跃的网民会很有趣。
- **关于 Bubeck AGI 论文的辩论**：社区对 Bubeck 的“**AGI 的火花** (sparks of AGI)”论文褒贬不一，一些人认为它弊大于利。
   - 讨论暗示了论文中存在的**夸张定位**，以及对其对 AGI 定义产生影响的担忧。
- **OpenAI 的角色对法律专业人士有益**：一位成员强调 OpenAI 正在为**律师**创造有利条件，将 AI 的进步与法律工作联系起来。
   - 这一评论突显了 AI 技术与其在法律领域实际应用之间不断演变的关系。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.bloomberg.com/news/features/2024-10-14/why-openai-is-at-war-with-a-guy-named-guy?accessToken=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzb3VyY2UiOiJTdWJzY3JpYmVyR2lmdGVkQXJ0aWNsZSIsImlhdCI6MTcyODkxMTEyNSwiZXhwIjoxNzI5NTE1OTI1LCJhcnRpY2xlSWQiOiJTTEM5MFVEV1gyUFMwMCIsImJjb25uZWN0SWQiOiJBQjc4QTNBMDc1N0U0OTI0ODFCRUU5RDRCRjBDNERDNSJ9.mpWXrXYbOWqoPDB5c-FKwDK6V_Q9UyyhU5kIPKkwhDc">Bloomberg - Are you a robot?</a>: 未找到描述</li><li><a href="https://www.bloomberg.com/news/features/2024-10-14/why-openai-is-at-war-with-a-guy-named-guy?accessT">Bloomberg - Are you a robot?</a>: 未找到描述</li><li><a href="https://x.com/amir/status/1845905601110462481">Amir Efrati (@amir) 的推文</a>: 新闻：Microsoft 顶尖 AI 研究员之一 @SebastienBubeck 将加入 OpenAI。https://www.theinformation.com/briefings/microsoft-ai-researcher-sebastien-bubeck-to-join-openai?rc=c48ukx
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1295825683578097816)** (5 messages): 

> - `Doomsday Clock for AGI`
> - `AI2 OLMo Internship` 


- **AGI 末日时钟引发关注**：一家由沙特资助、位于瑞士的商学院推出了一个“末日时钟”，旨在警告“不受控制的通用人工智能”（即所谓的“神一般”的 AI）所带来的危险。该时钟的创作者 Michael Wade 批评这一隐喻已经过时，且在当今语境下缺乏相关性。
   - 他强调，将 Excel 之类的软件等同于神一般的 AI 威胁是荒谬的，并将其与原子武器引发的历史恐惧联系起来。
- **AI2 在西雅图提供研究实习岗位**：AI2 正在为其 OLMo 项目招募研究实习生，重点是推进自然语言处理和机器学习。这个为期 12 周的实习提供了领导具有影响力的研究项目并与该领域专家合作的机会。
   - 薪资范围为 **$86,520** 至 **$123,600**，具体取决于所获得的学位，该职位要求在西雅图实地办公。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://gizmodo.com/for-the-love-of-god-stop-making-inscrutable-doomsday-clocks-2000512111">For the Love of God, Stop Making Inscrutable Doomsday Clocks</a>：一家商学院正利用 AI 毁灭论、来自沙特阿拉伯的资金以及陈旧的冷战隐喻来炒作 AI 的未来。</li><li><a href="https://job-boards.greenhouse.io/thealleninstitute/jobs/6322728">Job Application for Research Internship, OLMo at The Allen Institute for AI</a>：未找到描述</li><li><a href="https://x.com/mattshumer_/status/1846209244284219703">Matt Shumer (@mattshumer_) 的推文</a>：http://x.com/i/article/1846205240728588288
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/)** (1 messages): 

victory: 有人做得太过了 https://www.myforevernotes.com/articles/overview
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1295486735802695873)** (11 messages🔥): 

> - `Framework Selection Challenges`
> - `Langgraph Deployment`
> - `Comparing dspy and Langchain`
> - `Shifting Frameworks`
> - `Langsmith for Tracing` 


- **框架选择简直是场噩梦！**：成员们对在 **LangChain**、**Langflow** 和 **Langgraph** 等框架之间不断的切换感到沮丧，这使得最终确定生产环境的选择变得困难。
   - 有人提到他们的整个代码库已经迁移到了 *LangChain LCEL*，突显了围绕这些框架的混乱局面。
- **在私有云上部署 Langgraph**：一位成员询问如何在 **US** 或 **EU** 以外的自有云上部署 **Langgraph** 应用，寻求社区的见解。
   - 虽然没有得到直接回复，但该问题凸显了对区域性托管应用日益增长的兴趣。
- **关于 dspy 与 LangChain 的辩论**：人们开始关注 **dspy** 是否会主导 **LangChain** 及其他框架，或者这些框架是否能保持其地位。
   - 这反映了社区对 AI 框架未来格局的不确定性。
- **认可 Langsmith 的实用性**：一位成员建议 **Langsmith** 对追踪（tracing）非常有益，暗示了它在不断变化的框架中的重要性。
   - 这引发了关于使用 **LangChain Academy** 上的 **Langgraph** 课程等资源来提升技能的建议。
- **澄清 Langflow 的隶属关系**：一位用户澄清说 **Langflow** 并不是 **LangChain** 的产品，这表明社区对相关工具存在混淆。
   - 这一区分可能有助于成员理清对所讨论的各种框架的理解。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1295461430379679775)** (2 messages): 

> - `课程信息`
> - `MOOC 注册`
> - `Discord 社区` 


- **所有课程详情均可在网上查阅**：关于 **labs** 和 **assignments** 的所有细节都可以在课程网站 [course website](https://llmagents-learning.org/f24) 上找到。
   - 鼓励参与者定期查看网站以获取更新和相关材料。
- **便捷的 MOOC 报名**：要加入课程，有意向的学生应填写这份便捷的 [表单](https://forms.gle/svSoNhKcGFjxup989)。
   - 这一步对于任何希望积极参与课程活动的人来说都是必不可少的。
- **在 Discord 上加入 LLM Agents 社区**：为了进行持续的讨论和提问，参与者应加入 **LLM Agents Discord** 社区，链接见 [Discord link](https://discord.gg/NWVpQ9rBvd)。
   - 该平台允许课程成员之间进行实时交流和支持。
- **感谢协助**：一位成员对另一位成员提供的重要信息表示感谢，这些信息他们最初忽略了。他们幽默地评论道：*'它就在那里，但我还是没看到'*。
   - 这突显了课程社区内的协作精神。



**Link mentioned**: <a href="https://llmagents-learning.org/f24">Large Language Model Agents</a>: no description found

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1295507315956383764)** (6 messages): 

> - `推理侧计算缩放定律 (Test-time compute scaling law)`
> - `关于 Devin 与 PearAI 的看法`
> - `LLM 推理与规划`
> - `LLM 中的工具调用 (Tool calling)`
> - `课程质量改进` 


- **观察到推理侧计算缩放定律 (Test-time compute scaling law)**：一位成员讨论了观察到的 **'test-time compute' scaling law** 产生的更广泛的下游影响，这与最初导致 GPT 系列诞生的原始定律相当。
   - 作为参考，他们分享了 [论文](https://arxiv.org/pdf/2408.03314) 和另一份重要的 [文档](https://arxiv.org/pdf/2001.08361)。
- **Prof Neubig 对 Devin 与 PearAI 的看法**：有人询问 **Prof Neubig** 对 **Devin** 和 **YC 支持的 PearAI** 的看法。
   - *Tilman 建议这可能属于另一个频道*，并强调了正在进行的相关讨论。
- **探索 LLM 推理与规划**：一位成员寻求关于 LLM 和 Agent 如何处理文本生成之外的 **reasoning**、**planning** 和 **acting** 的见解。
   - 他们请求了解 LLM 识别合适工具的过程，以及它们在提取相关实体时处理 **NER** 的方法。
- **未来课程计划**：一位成员询问是否有计划在未来的课程中涵盖 **planning** 和 **tool use**。
   - 这表明社区对深入理解 LLM 和 Agent 背景下的实际应用表现出浓厚兴趣。
- **课程视频质量问题**：一位成员表示需要提高上传的课程视频质量，并提到 YouTube 上第 6 课的最高分辨率仅为 **720p**。
   - 他们指出，目前的分辨率导致在听课过程中难以清晰阅读代码。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/1295481309501722634)** (1 messages): 

> - `AI 驱动搜索资源` 


- **《AI-Powered Search》一书成为必备资源**：一位成员推荐了 [这本书](https://www.manning.com/books/ai-powered-search)，认为它可能是未来几年 AI 驱动搜索技术领域的核心资源。
   - 他们表示相信这本书将对从业者和研究人员产生重大影响。
- **对行业影响的预期**：该成员表示坚信书中的见解将塑造各行各业 AI 驱动搜索功能的未来。
   - 他们强调了其成为 AI 研究课程基础内容的潜力。


  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1295634566459166740)** (2 条消息): 

> - `Open Interpreter Update`
> - `New Python Release` 


- **Open Interpreter 发布 π 版本**：一名成员宣布了 **Open Interpreter** 新版本的发布，并通过 [此帖子](https://x.com/MikeBirdTech/status/1846283357153268002) 分享了命令 `pip install --upgrade open-interpreter`。
   - 该更新被称为 **π 版本**，预示着在功能和性能上有重大提升。
- **日常早安问候**：一位成员在聊天中简单地说了声 **Goodmorning**，以积极的基调开启了新的一天。
   - 这一时刻反映了社区友好的氛围以及成员间互动的意愿。



**提到的链接**：<a href="https://x.com/MikeBirdTech/status/1846283357153268002">来自 Mike Bird (@MikeBirdTech) 的推文</a>：pip install --upgrade open-interpreter。一个 π 版本！

  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1295742137509609573)** (2 条消息): 

> - `Hume AI model`
> - `Oi model` 


- **Hume AI 模型效果惊人**：一位成员报告称，他们使用 **Hume AI 模型** 的体验出乎意料地好，表示其效果几乎好得**过头了**。
   - 这引发了关于 AI 模型在现实应用中的潜力和局限性的有趣思考。
- **从 Hume AI 切换到 Oi**：同一位成员承认了关注点的转移，提到了 **Oi** 而非 Hume AI 模型。
   - 这表明用户正在不断尝试不同的 AI 模型以评估其有效性。


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1295779101642326110)** (3 条消息): 

> - `Play 3.0 mini`
> - `Think-on-Graph` 


- **Play 3.0 mini 发布，速度更快且更准确**：[Play.ht](https://x.com/play_ht/status/1845901523680686401?s=46&t=G6jp7iOBtkVuyhaYmaDb0w) 推出了他们最新的 Text-To-Speech 模型 **Play 3.0 mini**，该模型具有更高的速度和准确性，支持多种语言，且**成本效益极高**。
   - 他们邀请用户在 [Playground](https://play.ht/playground/?utm_source=x&utm_medium=social&utm_campaign=all_v3launch_202410) 上进行体验并分享反馈。
- **在 GitHub 上探索 Think-on-Graph**：由 IDEA-FinAI 开发的 [Think-on-Graph GitHub 仓库](https://github.com/IDEA-FinAI/ToG) 已上线，邀请研究人员加入他们在深圳的团队或关注他们的工作。
   - 仓库中包含详细的联系方式，方便有兴趣的人员通过邮件进行合作洽谈。
- **关于近期进展的视频资源**：一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=iGQLG0bWDxE)，讨论了可能与 AI 内容相关的近期进展。
   - 视频内容的具体细节未作说明，建议观众自行观看了解。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/play_ht/status/1845901523680686401?s=46&t=G6jp7iOBtkVuyhaYmaDb0w">来自 PlayAI (原 PlayHT) (@play_ht) 的推文</a>：今天我们推出了最新的 Text-To-Speech 模型 Play 3.0 mini。它更快、更准确，处理多种语言，支持来自 LLM 的流式传输，并且比以往任何时候都更具成本效益...</li><li><a href="https://github.com/IDEA-FinAI/ToG">GitHub - IDEA-FinAI/ToG：这是 Think-on-Graph 的官方 GitHub 仓库。如果您对我们的工作感兴趣或愿意加入我们在深圳的研究团队，请随时通过电子邮件 (xuchengjin@idea.edu.cn) 与我们联系</a>：这是 Think-on-Graph 的官方 GitHub 仓库。如果您对我们的工作感兴趣或愿意加入我们在深圳的研究团队，请随时通过电子邮件 (xuchengjin@idea.edu....
</li>
</ul>

</div>
  

---

### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1295825107213750346)** (1 条消息): 

> - `Loom 视频见解` 


- **分享了 Loom 视频**：一位成员分享了一个 [Loom 视频](https://www.loom.com/share/b8c49e265d5c49aca7d2fc51c38d84c6?sid=69c5942b-ed75-4883-8867-ed75-4883-8867-f408a83cecf5)，其中可能包含有关相关主题的见解或讨论。
   - 视频的具体细节尚未提供，留待社区成员自行探索其中的价值信息。
- **视频内容探索**：成员们对分享的 Loom 视频中可以获取的见解表示了兴趣，并考虑了其与正在进行的讨论的相关性。
   - 由于缺乏初始描述，引发了大家的好奇心，成员们可能计划对其进行回顾以进行进一步讨论。



**提到的链接**：<a href="https://www.loom.com/share/b8c49e265d5c49aca7d2fc51c38d84c6?sid=69c5942b-ed75-4883-8867-f408a83cecf5">探索量子架构原则 🌌</a>：https://github.com/seanchatmangpt/dslmodel 在这段视频中，我深入探讨了量子架构原则领域及其在变革企业软件中的应用。从讨论 AI 驱动...

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1295567564587667541)** (5 条消息): 

> - `Contextual embeddings`
> - `RAG (Retrieval-Augmented Generation) 机制`
> - `DSPy 集成到 GPT-O1+` 


- **分享了 Contextual Embeddings 资源**：一位成员提供了一个 [Google Colab](https://tinyurl.com/2p9wwypy) 和一段名为 'Contextual Retrieval with Any LLM: A Step-by-Step Guide' 的 [YouTube 视频](https://www.youtube.com/watch?v=6efwN_US-zk&t=7s)，重点关注 Contextual embeddings。
   - 该视频旨在帮助为任何 LLM 实现来自 Anthropic 的 Contextual retrieval 策略。
- **澄清 RAG 和 Token 限制**：一位成员对将整个文档添加到超出 Token 限制的 Prompt 中表示困惑，并强调了 RAG 中使用的 Chunking 过程。
   - 会议澄清了 RAG 使用相似度搜索，仅在 Prompt 中包含最相关的 Chunk，从而维持 Token 限制。
- **DSPy 集成进度查询**：一位成员询问了将 DSPy 整合到系统（称为 GPT-O1+）中的进展。
   - 交流中未提供集成的具体细节。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tinyurl.com/2p9wwypy">Google Colab</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=6efwN_US-zk&t=7s">Contextual Retrieval with Any LLM: A Step-by-Step Guide</a>：在这段视频中，我将向你展示如何使用 Anthropic 的策略为任何 LLM 实现 Contextual retrieval。我们将贯穿整个过程，从 Chunk ...
</li>
</ul>

</div>
  

---



### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1295748356320133264)** (4 条消息): 

> - `ICLR 评审`
> - `LLM 中的 Continuous Pre-training`
> - `Model Merging 讨论` 


- **ICLR 评审终于公布了！**：期待已久的 ICLR 评审论文已经发布，引发了渴望深入研究的成员们的兴奋。
   - *一位成员指出*，处理分配给他们的评审需要一些时间。
- **关于 Continuous Pre-training 和 Instruction Fine-tuning 的研究**：最近的一篇论文研究了 Large Language Models 的 **Continuous Pre-training** 和 **Instruction Fine-tuning** 之间的关系，强调了模型需要保持最新数据更新的需求。
   - 它提出了一个问题：哪个模型应该进行这种 Pre-training 以保持指令遵循能力。
- **Model Merging 方法评述**：*一位成员质疑*论文中方法的创新性，认为它类似于早已确立的 Model Merging 方法。
   - 这引发了关于所提技术相关性和原创性的讨论。



**提到的链接**：<a href="https://arxiv.org/html/2410.10739v1">Balancing Continuous Pre-Training and Instruction Fine-Tuning: Optimizing Instruction-Following in LLMs</a>：未找到描述

  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1295511523556593754)** (4 messages): 

> - `LAION-2B dataset`
> - `Data Overlap with MSCOCO` 


- **关于 LAION-2B 数据集与 MSCOCO 重叠的咨询**：一位成员询问了 **LAION-2B dataset** 是否包含来自 **MSCOCO** (COCO2014 或 COCO2017) 数据集的图像，并对潜在的 **data overlap**（数据重叠）提出了疑问。
   - 他们注意到论文中提到了关于 **data overlap** 的章节，但他们正在寻找有关用于检查此问题的具体技术的更多细节。
- **早安与日常问候**：成员们交换了日常问候，一位成员说 **'Good morning everyone.'** 以营造友好的聊天环境。
   - 随后，另一位成员以随意的 **'gm'** 进行了回应。


  

---



### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1295517193642967124)** (3 messages): 

> - `Inference Pipeline Details`
> - `Model Function Call Outputs`
> - `Multi-turn Interaction Logic` 


- **理解推理流水线（Inference Pipeline）机制**：推理流水线允许模型执行函数，只要它持续输出可以被 `decod_exec` 方法解码的有效函数调用。
   - 如果模型输出空字符串或无法解码的响应，则表示当前轮次结束。
- **模型正确停止的能力**：一位成员强调了模型正确判断何时停止输出函数调用的重要性，特别是当模型认为它已经完成任务时。
   - 回应是 *“在某种意义上是肯定的，但不一定”*，这表明模型可以输出空内容来发出轮次结束的信号。
- **天气查询交互示例**：提供了一个详细的场景，模型使用 `get_coordinate` 和 `get_weather` 等函数调用来处理与天气相关的请求，并在每一步返回必要的数据。
   - 该对话表明，模型在获取数据后输出的句子无法被解码，因此导致该交互轮次的终止。
- **Function Call 输出的多样性**：模型可以通过各种方式表达停止或进行额外函数调用的需求，包括选择完全不输出任何内容。
   - 这种多样性反映了 Prompting 模型在有效管理用户查询时采取的不同方法。


  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/)** (1 messages): 

cyberg0285: Thank you <@709013886644518982>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1295786080221532252)** (1 messages): 

> - `Forwarding Protocols` 


- **关于转发协议的讨论**：一位成员提到了一个与 **forwarding protocols**（转发协议）相关的重要链接，并在频道中分享了他们的想法。
   - *此处是转发的消息供参考。*
- **信息共享动态**：另一位成员强调了正确的 **信息共享** 实践对于增强社区参与度的重要性。
   - 他们强调 *转发消息可以促进更快的响应和更清晰的沟通。*


  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1295823752277528668)** (1 messages): 

> - `AI Stewardship Practice Program`
> - `Tech Stewardship`
> - `Microcredential Opportunities` 


- **AI Stewardship 试点项目**：MaRS Discovery District 的 **AI Stewardship Practice Program** 正在为旨在积极塑造 AI 演进的试点课程提供免费名额。
   - 该微凭证（microcredential）计划面向研究人员、教育工作者和政策制定者等群体，更多详情请访问 [Tech Stewardship 网站](https://programs.techstewardship.com/)。
- **成为 Tech Steward 的机会**：参与者可以参与旨在启动和维护其 AI stewardship 实践的项目，宣传 **“引导技术向善”** 的座右铭。
   - 鼓励感兴趣的个人[在此处的线程中回复](https://discord.com/channels/1089876418936180786/1295822228406931529)，有机会参加价值 **500 加元** 的试点课程。



**提到的链接**：<a href="https://programs.techstewardship.com/">Tech Stewardship Practice</a>：未找到描述

  

---



---



---



---



{% else %}


> 完整的频道细分内容已在邮件中截断。
> 
> 如果您想查看完整的细分内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}