---
companies:
- nvidia
- meta-ai-fair
- ai21-labs
- anthropic
- hugging-face
date: '2024-08-23T22:14:15.426361Z'
description: '**英伟达（Nvidia）**和 **Meta** 的研究人员更新了他们的 **Llama 3** 研究结果，并发表论文展示了结合**权重剪枝（weight
  pruning）**与**知识蒸馏（knowledge distillation）**的有效性。该方法通过仅从头开始训练最大的模型，再通过剪枝和蒸馏衍生出较小的模型，从而降低训练成本。


  具体过程包括教师修正、基于激活的剪枝（偏向于宽度剪枝），以及使用 KL 散度损失进行蒸馏重训，最终在同等规模下获得了性能更佳的模型。不过，蒸馏也会带来一定的准确率权衡。


  此外，**AI21 Labs** 推出了 **Jamba 1.5**，这是一款混合了 SSM-Transformer 架构的 MoE（混合专家）模型，具备大上下文窗口和多语言支持。**Anthropic**
  为 **Claude 3** 更新了 LaTeX 渲染和提示词缓存（prompt caching）功能。一款名为 **Dracarys** 的开源编程大模型发布了
  70B 和 72B 版本，展现出更强的编程性能。**Mistral Nemo Minitron 8B** 模型在 Hugging Face 排行榜上超越了 **Llama
  3.1 8B** 和 **Mistral 7B**，进一步凸显了剪枝与蒸馏的优势。关于提示词优化的研究则揭示了提示词搜索空间的复杂性，以及 AutoPrompt/GCG
  等简单算法出人意料的有效性。'
id: 7495507e-4f91-471d-a8f4-3893fbe23ec7
models:
- llama-3-1-8b
- llama-3-1
- jamba-1.5
- claude-3
- dracarys-70b
- dracarys-72b
- mistral-nemo-minitron-8b
- mistral-7b
original_slug: ainews-nvidia-minitron-llm-pruning-and
people: []
title: Nvidia Minitron：针对 Llama 3.1 更新的大语言模型剪枝与蒸馏技术。
topics:
- pruning
- knowledge-distillation
- weight-pruning
- activation-based-pruning
- width-pruning
- kl-divergence
- teacher-correction
- prompt-optimization
- multilinguality
- long-context
- mixture-of-experts
- model-fine-tuning
---

<!-- buttondown-editor-mode: plaintext -->**剪枝与蒸馏就是你所需要的一切。**

> 2024年8月22日至8月23日的 AI 新闻。我们为你检查了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 服务器（**214** 个频道，**2531** 条消息）。预计节省阅读时间（按每分钟 200 字计算）：**284 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论了！

最近几周我们曾间接提到过 4B 和 8B Minitron（Nvidia 对 Llama 3.1 8B 的蒸馏版本），但现在 Sreenivas & Muralidharan 等人（[上个月 Minitron 论文的作者](https://www.arxiv.org/abs/2407.14679)）发布了一篇 [7 页的精简论文](https://arxiv.org/abs/2408.11796)，将他们之前的 Llama 2 研究结果更新到了 Llama 3：


![image.png](https://assets.buttondown.email/images/ba2a2c11-34c6-4e3f-b810-e0e34199e4c5.png?w=960&fit=max)
 

鉴于 Nvidia 与 Meta 的紧密关系，这一点非常重要，它提供了关于 Llama 3 的一些见解：

> "从头开始训练多个数十亿参数的模型极其耗费时间、数据和资源。最近的工作 [1] 证明了将权重剪枝（weight pruning）与知识蒸馏（knowledge distillation）相结合，可以显著降低训练 LLM 模型家族的成本。在这里，**模型家族中只有最大的模型是从头开始训练的**；其他模型是通过对较大模型进行连续剪枝，然后进行知识蒸馏以恢复剪枝模型的准确性而获得的。"

 
![image.png](https://assets.buttondown.email/images/aa85975c-6f1b-4c6c-a723-70924686d925.png?w=960&fit=max)
 

主要步骤：

1.  **教师校正 (teacher correction)**：在用于蒸馏的目标数据集上对教师模型进行轻微微调，使用约 127B tokens。
2.  **深度或宽度剪枝 (depth or width pruning)**：使用“一种纯粹基于激活的重要性评估策略，通过一个小型校准数据集和仅前向传播过程，同时计算我们考虑的所有轴（深度、神经元、头部和嵌入通道）的敏感度信息”。在消融实验中，宽度剪枝的表现始终优于深度剪枝。 
![image.png](https://assets.buttondown.email/images/412b3028-819e-439f-b68f-b3449d0bdc5a.png?w=960&fit=max)
 
3.  **通过蒸馏进行重训练 (Retraining with distillation)**：使用“真正的” KD，即在教师和学生模型的 Logits 上使用 KL Divergence 损失。 
![image.png](https://assets.buttondown.email/images/36883594-3e37-4d99-8f6c-3867c3101566.png?w=960&fit=max)
 

这产生了一个在同等尺寸下全面表现更优的模型：

 
![image.png](https://assets.buttondown.email/images/6b67ffa3-d576-44c4-8503-bec0e88e9014.png?w=960&fit=max)
 

 
![image.png](https://assets.buttondown.email/images/0bf15b3b-64f9-4443-a2ba-b7e307717840.png?w=960&fit=max)
 

然而，这种蒸馏远非无损；论文并没有直观地列出差异，但在末尾的脚注中提到了权衡。

 
![image.png](https://assets.buttondown.email/images/a33f1dd5-5a0f-43eb-a4d2-44f3ff5c21e1.png?w=960&fit=max)
 

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 综述

> 所有综述均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型发布与进展**

- **Jamba 1.5 发布**：[@AI21Labs](https://twitter.com/osanseviero/status/1826607725280682154) 发布了 Jamba 1.5，这是一款混合 SSM-Transformer MoE 模型，分为 Mini（52B - 12B active）和 Large（398B - 94B active）两个版本。主要特性包括 256K context window、多语言支持以及针对长上下文任务的性能优化。

- **Claude 3 更新**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1826667671364272301) 为 Claude 3 增加了 LaTeX 渲染支持，增强了其显示数学方程和表达式的能力。[Prompt caching](https://twitter.com/alexalbert__/status/1826676781925237234) 现在也已支持 Claude 3 Opus。

- **Dracarys 发布**：[@bindureddy](https://twitter.com/bindureddy/status/1826757521635455115) 宣布了 Dracarys，这是一个针对编程任务进行 fine-tuned 的开源 LLM，提供 70B 和 72B 版本。与其他开源模型相比，它在编程性能上显示出显著提升。

- **Mistral Nemo Minitron 8B**：该模型在 Hugging Face Open LLM Leaderboard 上的表现优于 Llama 3.1 8B 和 Mistral 7B，展示了对大型模型进行 [pruning 和 distilling](https://twitter.com/_philschmid/status/1826699564088242202) 的潜在优势。

**AI 研究与技术**

- **Prompt 优化**：[@jxmnop](https://twitter.com/jxmnop/status/1826681982375571621) 讨论了 Prompt 优化的挑战，强调了在巨大的搜索空间中寻找最优 Prompt 的复杂性，以及像 AutoPrompt/GCG 这样简单算法的出人意料的有效性。

- **混合架构**：[@tri_dao](https://twitter.com/tri_dao/status/1826712490992173551) 指出，混合 Mamba / Transformer 架构表现良好，特别是在长上下文和快速推理（inference）方面。

- **Flexora**：一种新的 LoRA fine-tuning 方法，[可以产生更优的结果并将训练参数减少高达 50%](https://twitter.com/rohanpaul_ai/status/1826733730746282290)，它为 LoRA 引入了自适应层选择。

- **Classifier-Free Diffusion Guidance**：[@sedielem](https://twitter.com/sedielem/status/1826682679196348714) 分享了近期论文的见解，这些论文对目前关于 Classifier-Free Diffusion Guidance 的普遍假设提出了质疑。

**AI 应用与工具**

- **Spellbook Associate**：[@scottastevenson](https://twitter.com/scottastevenson/status/1826611092652474635) 宣布推出 Spellbook Associate，这是一个用于法律工作的 AI Agent，能够分解项目、执行任务并调整计划。

- **Cosine Genie**：`@swyx` [强调](https://twitter.com/swyx/status/1826673380294267328) 了一期播客节目，讨论了为代码 fine-tuning GPT4o 的价值，使其成为了根据各种基准测试表现最佳的 coding Agent。

- **LlamaIndex 0.11**：[@llama_index](https://twitter.com/llama_index/status/1826684496407920705) 发布了 0.11 版本，新特性包括用 Workflows 取代 Query Pipelines，以及核心包体积减小了 42%。

- **MLX Hub**：一个新的命令行工具，用于从 Hugging Face Hub 搜索、下载和管理 MLX 模型，由 [@awnihannun 宣布](https://twitter.com/awnihannun/status/1826633844847784359)。

**AI 发展与行业趋势**

- **AI Agent 的挑战**：[@RichardSocher](https://twitter.com/RichardSocher/status/1826678227936707063) 强调了在 AI Agent 的多步 Workflows 中实现高准确率的难度，并将其比作自动驾驶汽车中的“最后一公里”问题。

- **开源 vs. 闭源模型**：[@bindureddy](https://twitter.com/bindureddy/status/1826757521635455115) 指出，大多数开源 fine-tunes 在改进特定维度的同时会降低整体性能，并强调了 Dracarys 在提升整体性能方面取得的成就。

- **AI 监管**：[@jackclarkSF](https://twitter.com/jackclarkSF/status/1826743366652232083) 分享了给 Newsom 州长关于 SB 1047 的信函，讨论了拟议的 AI 监管法案的成本与收益。

- **AI 硬件**：讨论了结合多个设备资源处理家庭 AI 工作负载的潜力，如 [@rohanpaul_ai 所述](https://twitter.com/rohanpaul_ai/status/1826627005137264899)。


---

# AI Reddit 综述

## /r/LocalLlama 综述

- **[Exllamav2 Tensor Parallel 支持！TabbyAPI 也是！](https://github.com/turboderp/exllamav2/blob/master/examples/inference_tp.py)** ([评分: 55, 评论: 29](https://reddit.com//r/LocalLLaMA/comments/1ez43lk/exllamav2_tensor_parallel_support_tabbyapi_too/)): **ExLlamaV2** 引入了 **Tensor Parallel** 支持，实现了利用多块 GPU 进行推理。此次更新还集成了 **TabbyAPI**，使得部署和 API 访问更加便捷。社区对这些进展表现出极大的热情，强调了提升大语言模型性能和可访问性的潜力。
  - 用户对 **ExLlamaV2** 的更新表示兴奋，其中一位用户在多块 GPU 上以 **2.65bpw** 运行 **Mistral-Large2**，**8192 context length**，生成速度达到 **18t/s**。
  - 性能提升显著，**Qwen 72B 4.25bpw** 在 **2x3090 GPU**、2k context 下的速度从 17.5 t/s 提升到 20.8 t/s，**增长了 20%**。
  - 报告了一个影响 **draft model (qwama)** 的 Bug，并迅速得到了开发者的解决，体现了活跃的社区支持和快速的问题处理能力。


## 全球 AI Reddit 综述

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 与机器学习进展**

- **Stable Diffusion 两周年**：2022 年的今天，[第一个 Stable Diffusion 模型 (v1.4) 正式向公众发布](https://www.reddit.com/r/StableDiffusion/comments/1eyn79g/on_this_date_in_2022_the_first_stable_diffusion/)，标志着 AI 生成图像领域的一个重要里程碑。

- **NovelAI 开源原始模型**：[NovelAI 决定开源其原始 AI 模型](https://www.reddit.com/r/StableDiffusion/comments/1eytt42/novelai_decided_to_open_source_their_original_ai/)，尽管该模型此前曾被泄露。此举促进了 AI 社区的透明度与协作。

**AI 生成内容与工具**

- **去模糊 Flux Lora**：开发了一款新工具来[解决 AI 生成图像中的背景模糊问题](https://www.reddit.com/r/StableDiffusion/comments/1eyvzjv/say_goodbye_to_blurry_backgrounds_antiblur_flux/)，有望提升整体输出质量。

- **业余摄影 Lora**：一项关于 [AI 生成图像真实感的对比](https://www.reddit.com/r/StableDiffusion/comments/1eywnv8/realism_comparison_v2_amateur_photography_lora/)，使用了业余摄影 Lora 与 Flux Dev，展示了写实 AI 生成内容的进步。

- **Pony Diffusion V7**：[开发下一版本 Pony Diffusion](https://www.reddit.com/r/StableDiffusion/comments/1eyw6ub/towards_pony_diffusion_v7_going_with_the_flow/) 的进展，展示了专用 AI 模型的持续改进。

**机器人与 AR 技术**

- **Boston Dynamics 俯卧撑视频**：Boston Dynamics 在其官方 Instagram 上[发布了一段机器人做俯卧撑的视频](https://www.reddit.com/r/singularity/comments/1eysry3/boston_dynamics_posted_the_pushup_video_to_their/)，展示了机器人在移动性和力量方面的进步。

- **Meta 的 AR 眼镜**：Meta 将于 [9 月发布其新款 AR 眼镜](https://www.reddit.com/r/singularity/comments/1eylv8x/meta_will_unveil_its_new_ar_glasses_in_september/)，表明大型科技公司在增强现实技术方面的进展。

**AI 相关讨论与幽默**

- **AI 炒作与预期**：一个关于[等待 AGI](https://www.reddit.com/r/singularity/comments/1eydxn6/how_it_feels_to_wait_for_agi_to_start/) 推翻政府的幽默帖子引发了关于 AI 现状和公众认知的讨论。评论强调了对过度炒作 AI 能力的担忧，以及对现实预期的需求。

- **AI 视频生成推测**：一段[猫咪似乎在做饭的视频](https://www.reddit.com/r/StableDiffusion/comments/1ez5wmq/what_ai_do_you_think_was_used_to_make_this/)引发了关于 AI 视频生成技术的讨论，一些用户认为这是使用传统视频编辑方法而非 AI 制作的。

**AI 工具功能请求**

- 一个强调 [Flux D 期望功能](https://www.reddit.com/r/StableDiffusion/comments/1eykpwu/i_cant_speak_for_anybody_else_here_but_this_is/)的帖子，表明用户对改进 AI 图像生成工具的持续兴趣。


---

# AI Discord 综述

> 由 Claude 3.5 Sonnet 生成的总结之总结


**1. AI 模型发布与基准测试**

- **Jamba 1.5 在长文本领域领先**：**AI21 Labs** 推出了 **Jamba 1.5 Mini**（12B 激活/52B 总参数）和 **Jamba 1.5 Large**（94B 激活/398B 总参数），基于全新的 **SSM-Transformer architecture**，提供 **256K 有效上下文窗口**，并声称在长文本处理速度上比竞争对手快 **2.5 倍**。
   - Jamba 1.5 Large 在 **Arena Hard** 上获得了 **65.4 分**，超越了 **Llama 3.1 70B 和 405B** 等模型。这些模型已**可在 [Hugging Face](https://huggingface.co/collections/ai21labs/jamba-15-66c44befa474a917fcf55251) 立即下载**，并支持在各大云平台部署。
- **Grok 2 在 LMSYS Arena 夺得第二名**：**Grok 2** 及其 mini 版本已加入 **[LMSYS leaderboard](https://x.com/lmsysorg/status/1827041269534879784)**，目前 Grok 2 排名**第 2**，超越了 **GPT-4o (May)**，并在综合性能上与 **Gemini** 持平。
   - 该模型在数学方面表现尤为出色，并在包括困难提示词（hard prompts）、代码编写和指令遵循在内的其他领域名列前茅，展示了其在各种 AI 任务中的广泛能力。
- **SmolLM：小而强大的语言模型**：**[SmolLM](https://huggingface.co/HuggingFaceTB/SmolLM-135M)** 系列小型语言模型已发布，包含 135M、360M 和 1.7B 参数规模，是在精心策划的 **Cosmo-Corpus** 数据集上训练而成的。
   - 这些模型（包括 **Cosmopedia v2** 和 **Python-Edu** 等数据集）在与其同等规模的模型对比中显示出极具前景的结果，有望为各种 NLP 任务提供高效的替代方案。
  


**2. AI Development Tools and Frameworks**

- **Aider 0.52.0 为 AI 编程增加 Shell 能力**：**[Aider 0.52.0](https://github.com/paul-gauthier/aider/releases/tag/v0.52.0)** 引入了 shell 命令执行功能，允许用户直接在工具内启动浏览器、安装依赖、运行测试等，增强了其 AI 辅助编程的能力。
   - 该版本还包括一些改进，如 `/read` 和 `/drop` 命令支持 `~` 路径扩展，新增用于清除聊天记录的 `/reset` 命令，并将默认 OpenAI 模型切换为 `gpt-4o-2024-08-06`。值得注意的是，Aider 自主生成了该版本 68% 的代码。
- **Cursor 为 AI 驱动的编程融资 6000 万美元**：**[Cursor](https://www.cursor.com/blog/series-a)** 宣布获得来自 **Andreessen Horowitz、Jeff Dean** 以及 Stripe 和 Github 创始人的 6000 万美元融资，巩固了其作为领先 AI 驱动代码编辑器的地位。
   - 公司旨在通过即时回答、机械重构（mechanical refactors）和 AI 驱动的后台编码员等功能彻底改变软件开发，其宏伟目标是最终编写世界上所有的软件。
- **LangChain 提升 SQL 查询生成水平**：**[LangChain Python 文档](https://python.langchain.com/v0.2/docs/how_to/sql_prompting/#table-definitions-and-example-rows)** 概述了使用 `create_sql_query_chain` 改进 SQL 查询生成的策略，重点关注 SQL 方言如何影响提示词（prompts）。
   - 文档涵盖了如何使用 `SQLDatabase.get_context` 将 schema 信息格式化到提示词中，以及构建 few-shot 示例来辅助模型，旨在提高生成的 SQL 查询的准确性和相关性。
  


**3. AI Research and Technical Advancements**

- **Mamba 潜入 Transformer 领地**：**[Mamba 2.8B 模型](https://huggingface.co/state-spaces/mamba-2.8b-hf)** 已发布，这是一个兼容 `transformers` 的语言模型，为传统的 Transformer 模型提供了一种替代架构。
   - 用户需要从 main 分支安装 `transformers`（直到 4.39.0 版本发布），并安装 `causal_conv_1d` 和 `mamba-ssm` 以使用优化的 CUDA kernels，这可能在某些 NLP 任务中提供更高的效率。
- **AutoToS：自动化搜索思维**：一篇题为 **["AutoToS: Automating Thought of Search"](https://arxiv.org/abs/2408.11326)** 的新论文提出了一种自动化的“搜索思维”（ToS）方法，用于 LLM 规划，在评估领域中以极少的反馈迭代实现了 100% 的准确率。
   - 该方法涉及使用代码定义搜索空间，并通过单元测试的反馈引导 LLM 生成可靠且完整的搜索组件，有望推动 AI 驱动的规划和问题解决领域的发展。
- **多模态 LLM 跳过 ASR 中间环节**：一位研究人员分享了关于**多模态 LLM** 的工作，该模型无需单独的自动语音识别（ASR）阶段即可直接理解文本和语音，通过使用多模态投影器（multimodal projector）扩展 **Meta 的 Llama 3 模型**构建而成。
   - 与结合独立 ASR 和 LLM 组件的系统相比，这种方法可以实现更快的响应速度，可能为更高效、更集成的多模态 AI 系统开辟新途径。
  


**4. AI Industry News and Events**

- **Autogen 负责人离开 Microsoft 开启新事业**：**Autogen** 项目负责人于 2024 年 5 月离开 **Microsoft**，创办了 **[OS autogen-ai](https://github.com/autogen-ai)**，这是一家目前正在融资的新公司。
   - 这一举动预示着 Autogen 生态系统可能迎来新发展，并凸显了行业内 AI 人才流动的动态特性。
- **NVIDIA AI Summit India 官宣**：**[NVIDIA AI Summit India](https://nvda.ws/3AbEKCi)** 定于 2024 年 10 月 23 日至 25 日在孟买 Jio World Convention Centre 举行，届时将有 **Jensen Huang** 的炉边谈话以及超过 50 场关于 AI、机器人等主题的分会。
   - 该活动旨在加强 NVIDIA 与行业领袖及合作伙伴的联系，展示在生成式 AI、LLM、工业数字化、超级计算和机器人领域的变革性工作。
- **加州 AI 监管热潮**：加州本周将对 **[20 多项 AI 监管法案](https://docs.google.com/spreadsheets/d/1A-6ot8qg_pO4LbmhwenmEt5ipO-z93qQrJYuZDEsGJo/edit?usp=sharing)** 进行投票，涵盖了该州 AI 部署和创新的各个方面。
   - 这些法案可能会显著重塑在加州运营的 AI 公司和研究人员的监管格局，并可能为其他州和国家设定先例。
  


**5. AI 安全与伦理讨论**

- **AI 倦怠引发行业关注**：AI 社区的讨论对 **AI 倦怠（AI burnout）** 的可能性发出了警报，特别是在高强度的前沿实验室中，人们担心对进步的无情追求可能导致不可持续的工作模式。
   - 成员们将 AI 高级用户比作“*施法者职业（spellcasting class）*”，暗示 AI 模型能力的增强可能会增加对这些用户的需求，从而可能加剧该领域的倦怠问题。
- **AI 能力与风险 Demo-Jam Hackathon 启动**：**AI Capabilities and Risks Demo-Jam Hackathon** 启动，奖金池为 2000 美元，鼓励参与者创建能够弥合 AI 研究与公众对 AI 安全挑战理解之间鸿沟的演示。
   - 该活动旨在展示潜在的 AI 驱动的社会变革，并以引人入胜的方式传达 AI 安全挑战，优秀项目将有机会加入 **Apart Labs** 进行进一步研究。
- **Twitter 上的 AI 讨论强度受到质疑**：最近 **[Greg Brockman 的一条推文](https://x.com/amir/status/1827007117838192699)** 显示其一周工作 97 小时进行编码，这引发了关于 Twitter 上 AI 讨论强度及其与现实可能脱节的讨论。
   - 社区成员对社交媒体平台上经常分享的高压叙事表示不安，质疑这种强度对于 AI 领域的长期健康是否可持续或有益。
- **伦敦 AI 工程师见面会**：首届 **AI Engineer London Meetup** 定于 **9 月 12 日**举行，演讲嘉宾包括 **@maximelabonne** 和 **Chris Bull**。
  - 鼓励参与者在[此处](https://x.com/dctanner/status/1827071893448618453?s=46)注册，与 AI 工程师同行交流。
- **Infinite Generative Youtube 开发中**：一个团队正在为其 **Infinite Generative Youtube** 平台寻找开发人员，准备进行封闭测试。
  - 他们正在寻找充满激情的开发人员加入这个创新项目。


---

# 第一部分：Discord 高层级摘要

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 0.3.0 发布并带来升级**：LM Studio 发布了 **0.3.0** 版本，具有**全新的 UI**、改进的 **RAG** 功能，并支持使用 `lms` 运行**本地服务器**。
   - 然而，用户报告了模型加载问题等 bug，表明开发团队正在积极修复。
- **Llama 3.1 硬件性能评测**：**Llama 3.1 70B q4** 在配备双通道 DDR5-6000 的 9700X CPU 上达到了 **1.44 t/s** 的 token 速率，突显了其 CPU 性能表现。
   - 用户指出，如果 GPU 的 VRAM 小于模型大小的一半，GPU offloading 可能会降低推理速度。
- **关于 Apple Silicon 与 Nvidia 运行 LLM 的辩论**：一场持续的讨论对比了 **M2 24gb Apple Silicon** 与 Nvidia 设备，有报告称 M2 Ultra 在特定场景下可能优于 **4090**。
   - 然而，用户在 Apple Silicon 上面临微调速度的限制，据报道在顶配 Macbook Pro 上需要 **9 小时的训练**。
- **GPU Offloading 仍是热门话题**：尽管有用户报告问题，LM Studio 仍支持 **GPU offloading**；用户可以在选择模型时按住 ALT 键来激活它。
   - 随着用户在各种配置下探索性能，持续寻找最佳设置仍然至关重要。
- **LLM 准确性引发关注**：讨论显示，**Llama 3.1** 和 **Phi 3** 等 LLM 可能会产生幻觉，特别是在学习风格或特定查询方面，导致输出冗长。
   - 一项对比分析指出，Claude 尽管表达模糊，但可能展示出更好的自我评估机制。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Research 周边商店上线！**：**Nous Research 周边商店**已正式发布，提供各种商品，包括每笔订单附赠的贴纸（送完即止）。
   - 点击[此处](https://shop.nousresearch.com/)查看专属周边！
- **Hermes 3 从模式崩溃 (Mode Collapse) 中恢复**：一名成员报告成功让 **Hermes 3** 从模式崩溃中恢复，使模型能够分析并理解崩溃原因，之后仅发生过一次复发。
   - 这标志着在解决大型语言模型中普遍存在的**模式崩溃**问题上迈出了一步。
- **介绍 Mistral-NeMo-Minitron-8B-Base**：**Mistral-NeMo-Minitron-8B-Base** 是一个经过剪枝和蒸馏的文本到文本模型，利用了 3800 亿个 token 以及来自 Nemotron-4 15B 的持续预训练数据。
   - 该基础模型展示了模型效率和性能方面的进步。
- **探索 LLM 行为中的“疯狂”**：有人提议刻意将 LLM 调优至“**疯狂**”状态，旨在探索意外行为的边界，并深入了解 LLM 的局限性。
   - 该项目寻求模拟 LLM 输出中的异常情况，这可能会带来突破性的发现。
- **适用于 LLM 的 Drama Engine 框架**：一位成员分享了他们的项目 **Drama Engine**，这是一个叙事 Agent 框架，旨在改进类 Agent 的交互和故事创作。
   - 他们提供了该项目的 GitHub 页面链接，供有兴趣贡献或了解更多信息的人参考：[Drama Engine GitHub](https://github.com/Write-with-LAIKA/drama-engine)。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **LogLLM - 自动化机器学习实验日志记录**：LogLLM 使用 GPT4o-mini 自动从 Python 脚本中提取实验条件，并使用 Weights & Biases (W&B) 记录结果。
   - 它简化了机器学习实验的文档记录过程，提高了效率和准确性。
- **Neuralink 在 HF 实现论文**：一位成员分享说，他们在 Hugging Face 的工作涉及论文实现，可能是一个付费职位。
   - 他们对自己的工作表示兴奋，并强调了为低端设备创建高效模型的重要性。
- **适用于低端设备的高效模型**：成员们对提高低端硬件上的模型效率表现出浓厚兴趣，突显了当前面临的挑战。
   - 这反映了社区对在不同环境中实现可访问性和实际应用的关注。
- **GPU 性能怪兽：RTX 6000 亮相**：用户发现了 **RTX 6000** 的存在，它拥有 **48GB VRAM**，可处理强大的计算任务。
   - 其价格为 **7,000 美元**，是高性能工作负载的首选。
- **用于泛化的三路数据拆分**：一位成员建议采用**三路数据拆分**，以增强模型在训练、验证和测试期间的泛化能力。
   - 重点是使用多样化的数据集进行测试，以评估模型在准确性之外的鲁棒性。

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SDXL vs SD1.5：速度困境**：一位拥有 **32 GB RAM** 的用户在 **SDXL** 和 **SD1.5** 之间犹豫不决，并反馈在其 CPU 上生成图像速度缓慢。尽管可能出现显存溢出（out-of-memory）错误且需要增加交换空间（swap space），建议仍倾向于选择图像质量更优的 **SDXL**。
   - *请记住*，对于这些大型模型，平衡 CPU 速度与图像质量是关键因素。
- **提示词技巧：大辩论**：成员们分享了他们在提示词（prompt）技巧方面的经验，有人发现使用逗号分隔效果很好，而另一些人则更喜欢自然语言提示词。这种差异突显了关于如何实现一致性的最佳提示策略的持续**讨论**。
   - 参与者建议，提示词的有效性在很大程度上取决于个人喜好和实验尝试。
- **ComfyUI 和 Flux 安装难题**：一位用户在 ComfyUI 上安装 **iPadaper** 时遇到挑战，得到的建议是前往技术支持频道寻求帮助。另一位用户在使用 **Flux** 时遇到困难，尝试通过不同的提示词来克服噪点多、质量低的输出问题。
   - 这强调了社区在追求创意目标、微调设置过程中共同经历的尝试与错误。
- **GPU RAM：性能的重要性**：在使用 **16GB RTX 3080** 运行 **Flux** 时，出现了关于在 ComfyUI 中调整 GPU 权重的问题。一位使用 **4GB GPU** 的用户报告了在 **A1111** 中令人沮丧的减速现象，这表明了 GPU 性能对图像生成的影响。
   - 这一交流表明，为了在各种模型中实现更流畅的性能，对强大硬件有着迫切需求。
- **丰富的 Stable Diffusion 指南**：一位用户请求推荐 **Stable Diffusion** 安装指南，**Automatic1111** 和 **ComfyUI** 被建议作为良好的起点。AMD 显卡虽然可以使用，但被指出性能较慢。
   - 技术支持频道被强调为故障排除和指导的重要资源。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 0.52.0 发布：Shell 命令执行及更多**：[Aider 0.52.0](https://github.com/paul-gauthier/aider/releases/tag/v0.52.0) 引入了 Shell 命令执行功能，允许用户直接在工具内启动浏览器、安装依赖项并运行测试。关键更新包括命令的 `~` 路径扩展、用于清除聊天记录的 `/reset` 命令，以及模型切换至 `gpt-4o-2024-08-06`。
   - Aider 自主生成了该版本 **68% 的代码**，突显了其在软件开发领域不断进步的能力。
- **Aider 训练集：元格式探索**：一位成员正在收集 Aider 的“提示词-代码”对训练集，旨在利用 **DSPY**、**TEXTGRAD** 和 **TRACE** 等工具为各种编码技术创建高效的元格式（meta-format）。该计划包括一个协作线程，用于对优化进行更深入的头脑风暴。
   - 其目标是精简代码和提示词以获得更好的可复现性，增强 Aider 生成代码的有效性。
- **使用 Aider 进行 Token 优化**：一位用户正在寻求关于优化 Token 使用的文档，特别是针对超过 OpenAI 限制的小型 Python 文件，以及处理需要多步过程的复杂任务时。他们正在寻找减少项目内 Token 上下文（context）的策略。
   - 他们特别要求在计算和渲染优化方面取得进展，强调了改进资源管理的必要性。
- **Cursor 对 AI 驱动代码创作的愿景**：Cursor 的[博客文章](https://www.cursor.com/blog/series-a)描绘了开发 AI 驱动代码编辑器的愿景，该编辑器可能自动完成大量的代码编写任务。功能包括即时响应、重构以及在几秒钟内完成大规模更改。
   - 未来的增强目标是实现后台编码、伪代码修改和错误检测，彻底改变开发者与代码交互的方式。
- **规划中的 LLM：AutoToS 论文见解**：[AutoToS: Automating Thought of Search](https://arxiv.org/abs/2408.11326) 提议使用 LLM 自动化规划过程，展示了其在不同领域实现 **100% 准确率** 的有效性。该方法允许 LLM 用代码定义搜索空间，增强了规划方法论。
   - 论文识别了搜索准确性方面的挑战，并阐述了 AutoToS 如何利用单元测试的反馈来引导 LLM，强化了对 AI 驱动规划可靠性的追求。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Autogen 负责人离开 Microsoft**：**Autogen** 项目负责人于 **2024 年 5 月**离开 Microsoft，启动了开源项目 [autogen-ai](https://github.com/autogen-ai)，目前正在融资。
   - 这一转变引发了关于 AI 编码标准和协作领域新创业项目的讨论。
- **Cursor AI 获得 6000 万美元支持**：**Cursor** 成功从 **Andreessen Horowitz** 和 **Jeff Dean** 等知名投资者处筹集了 **6000 万美元**，声称要重塑 AI 编程方式。
   - 他们的产品旨在开发能够在大规模范围内实现代码编写自动化的工具。
- **加州提议新的 AI 监管法案**：加州本周将对 **20 多项 AI 监管法案**进行投票，详见此 [Google Sheet](https://docs.google.com/spreadsheets/d/1A-6ot8qg_pO4LbmhwenmEt5ipO-z93qQrJYuZDEsGJo/edit?usp=sharing) 摘要。
   - 这些法案可能会重塑该州 AI 部署和创新的格局。
- **AI Engineer 见面会准备就绪！**：欢迎参加 **9 月 12 日**晚举行的首届 **AI Engineer 伦敦见面会**，演讲嘉宾包括 **@maximelabonne** 和 **Chris Bull**。
   - 通过此 [链接](https://x.com/dctanner/status/1827071893448618453?s=46) 注册，与 AI Engineer 同行交流。
- **Taxonomy Synthesis 助力 AI 研究**：成员们讨论了利用 [Taxonomy Synthesis](https://github.com/CakeCrusher/TaxonomySynthesis) 来分层组织写作项目。
   - **GPT Researcher** 工具因其能够自主进行深入研究、提高生产力而受到关注。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 停用多个模型**：自 **2024 年 8 月 28 日**起，OpenRouter 将弃用多个模型，包括 `01-ai/yi-34b`、`phind/phind-codellama-34b`、`nousresearch/nous-hermes-2-mixtral-8x7b-sft` 以及完整的 **Llama** 系列，届时用户将无法使用。
   - 用户可通过 [Together AI 的弃用文档](https://docs.together.ai/docs/deprecations#2024-08-28-deprecation-of-low-usage-and-older-serverless-models) 了解此政策，该文档概述了迁移选项。
- **OpenRouter 的计费小插曲**：一名用户在误选付费模型后产生了 **0.01 美元** 的费用，这说明了不熟悉界面的新手可能会遇到的潜在问题。
   - 对此，社区安抚该用户称 OpenRouter 不会追究如此低额度的欠费，营造了一个友好的 AI 探索环境。
- **Token 计数困惑得到澄清**：在一名用户报告简单的 Prompt 被收取了 **100 多个 Token** 的费用后，引发了关于 OpenRouter Token 计数机制的讨论，揭示了 Token 计算的复杂性。
   - 成员们澄清，OpenRouter 转发来自 OpenAI API 的 Token 计数，其差异受系统提示词（System Prompts）和聊天中先前上下文的影响。
- **Grok 2 在 LMSYS 排行榜表现亮眼**：**Grok 2** 及其 mini 变体在 LMSYS 排行榜上占据了一席之地，Grok 2 排名 **第 2**，在性能指标上甚至超越了 GPT-4o。
   - 该模型在数学和指令遵循（Instruction-following）方面表现尤为出色，在编码挑战中也展示了极高的能力，引发了对其整体性能概况的讨论。
- **OpenRouter 团队依然神秘**：有人询问 **OpenRouter 团队当前的项目**，但遗憾的是没有得到详细回复，令成员们感到好奇。
   - 这种信息的缺乏凸显了人们对 OpenRouter 开发活动的持续关注，但具体细节仍难以获知。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 的开源许可困境**：关于 **Mojo** 开源状态的问题被提出，Modular 正在处理许可细节，以在允许外部使用的同时保护其市场定位。
   - 他们的目标是随着时间的推移采用更宽松的许可模型，在保持开放性的同时保护核心产品特性。
- **Max 的集成模糊了与 Mojo 的界限**：**Max** 的功能现在与 **Mojo** 深度集成，尽管最初被设计为独立实体，这引发了关于未来是否会分离的疑问。
   - 讨论表明，这种紧密的集成将影响许可的可能性和产品开发路径。
- **Modular 对托管 AI 的商业关注**：Modular 正专注于托管 AI 云应用，这允许其在对 **Max** 进行商业应用许可的同时，继续对 **Mojo** 和 **Max** 进行投资。
   - 他们引入了一种鼓励开放开发并符合其战略业务目标的许可方式。
- **为异构计算铺平道路**：Modular 的目标是跨异构计算场景的**便携式 GPU 编程**，从而促进对先进计算工具的更广泛访问。
   - 他们的目标是提供能够为寻求高级计算能力的开发者简化集成过程的框架。
- **异步编程在 Mojo 中占有一席之地**：用户讨论了 **Mojo** 中**异步（asynchronous）**功能的潜力，特别是针对 I/O 任务，将其类比为 Python 的 async 能力。
   - 对话包括探索一种 "sans-io" HTTP 实现，强调线程安全和适当的资源管理。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 内部数据困境**：用户正在寻求关于 Perplexity 中**追问频率**的数据，包括花费的时间和往返交互，但回复表明这些数据可能是**私有的**。
   - 这引发了致力于性能改进的 **engineers** 对透明度和可用性的担忧。
- **Perplexity Pro 来源数量之谜**：在研究查询中，**Perplexity Pro** 显示的来源数量从 **20 个或更多显著下降到 5 或 6 个**，这引发了关于服务变更或使用不当的疑问。
   - 这种不一致性凸显了对**来源管理**清晰度的需求，以及对研究质量的潜在影响。
- **探索邮件自动化工具**：用户正在深入研究用于自动化电子邮件的 AI 工具，提到了 **Nelima, Taskade, Kindo** 和 **AutoGPT** 作为竞争者，同时寻求进一步的建议。
   - 这种探索表明，人们对通过 AI 效率**简化沟通流程**的兴趣日益浓厚。
- **Perplexity AI Bot 寻求可分享的 Thread**：**Perplexity AI Bot** 鼓励用户通过提供 Discord 频道链接来确保他们的 Thread 是“可分享的”，以便参考。
   - 这种对可分享内容的推动表明其专注于增强**社区参与**和资源共享。
- **围绕 MrBeast 的社交情绪**：讨论涉及了**互联网对 MrBeast 的看法**，用户链接到了一个 [search query](https://www.perplexity.ai) 以深入了解潜在的反感原因。
   - 这场对话反映了**数字名人文化**和公众舆论动态中的更广泛趋势。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Base Phi 3.5 未发布**: 一位成员强调缺少基础版 **Phi 3.5**，指出 Microsoft 仅发布了 instruct 版本。这给那些希望在没有基础版访问权限的情况下微调模型的人带来了挑战。
   - *探索可用性的极限*，他们正在寻求使用现有 instruct 版本进行微调的解决方案。
- **QLORA + FSDP 硬件需求**: 关于运行 **QLORA + FSDP** 的讨论集中在需要 **8xH100** 配置上。成员们还注意到，在训练期间启用热重启（warm restarts）时，tqdm 进度条会出现不准确的情况。
   - *性能监控仍是一个挑战*，这促使人们要求改进框架内可用的跟踪工具。
- **SmolLM：一系列小语言模型**: **SmolLM** 包括 135M、360M 和 1.7B 参数的小型模型，全部在高质量的 **Cosmo-Corpus** 上训练。这些模型整合了 Cosmopedia v2 和 FineWeb-Edu 等各种数据集，以确保训练的鲁棒性。
   - *精选的数据集选择*，旨在提供在不同条件下平衡的语言理解能力。
- **Transformers 中的模式感知聊天模板**: 用户在 Transformers 仓库中报告了一个关于**模式感知聊天模板（mode-aware chat templates）**的问题，建议该功能可以区分训练和推理行为。这可能会解决与聊天模板配置相关的现有问题。
   - 详情列在 [GitHub issue](https://github.com/huggingface/transformers/issues/33096) 中，该 issue 提议实现一个 `template_mode` 变量。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-3.5：过时还是有趣？**: 出现了一场关于测试 **GPT-3.5** 是否仍然具有相关性的讨论，因为考虑到后训练（post-training）技术的进步，它可能被认为已经过时。
   - 一些成员认为，与 **GPT-4** 等新模型相比，它可能缺乏重要性。
- **探索邮件自动化替代方案**: 用户正在寻找自动化邮件任务的工具，寻找除了 **Nelima** 之外基于提示词发送邮件的替代方案。
   - 这表明在日常工作流中对自动化解决方案的需求日益增长。
- **SwarmUI：用户体验备受赞誉**: **SwarmUI** 因其用户友好的界面以及对 NVIDIA 和 AMD GPU 的兼容性而获得赞誉。
   - 用户强调了其直观的设计，使其成为许多开发者的首选。
- **知识文件格式困境**: 一位用户询问在项目中使用 **XML** 还是 **Markdown** 作为知识文件更有效，旨在获得最佳性能。
   - 这一询问反映了关于在 GPTs 中构建内容最佳实践的持续争论。
- **不一致的 GPT 格式导致挫败感**: 用户对 GPT 响应中不一致的输出格式表示担忧，特别是关于某些消息传达了结构化内容而其他消息则没有。
   - 用户正在寻找标准化格式的解决方案，以增强可读性和用户体验。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **掌握多轮提示词**: 一位用户强调了在多轮提示词中包含 `n-1` 轮的重要性，并引用了来自对齐手册（alignment handbook）的 [代码示例](https://github.com/huggingface/alignment-handbook/blob/27f7dbf00663dab66ad7334afb7a1311fa251f41/src/alignment/data.py#L80)。
   - 他们探索了逐步增加轮数来生成提示词的可行性，但对其相对有效性表示担忧。
- **SmolLM 模型见解**: 讨论了 [SmolLM 模型](https://huggingface.co/HuggingFaceTB/SmolLM-135M)，指出其训练数据源自 Cosmo-Corpus，其中包括 Cosmopedia v2 等。
   - SmolLM 模型参数范围从 135M 到 1.7B，在其尺寸类别中表现出显著的性能。
- **Mamba 模型部署帮助**: 分享了关于 [Mamba 2.8B 模型](https://huggingface.co/state-spaces/mamba-2.8b-hf) 的信息，该模型可与 `transformers` 库无缝协作。
   - 提供了设置 `causal_conv_1d` 等依赖项以及使用 `generate` API 进行文本生成的说明。
- **创新的模型蒸馏技术**: 有建议提出将 LoRAs 应用于 27B 模型，并从较小的 9B 模型中蒸馏 logits，旨在以压缩形式复制功能。
   - 这种方法有可能在较小的架构中简化大型模型的性能。
- **模型压缩策略**: 压缩模型大小的建议包括将参数归零和应用量化方法等技术，并参考了关于 [量化技术](https://arxiv.org/abs/2408.11527) 的论文。
   - 讨论的技术旨在提高效率，同时管理模型大小。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere API 在无效角色上报错**：一位用户报告了在使用 **Cohere API** 时出现 HTTP-400 错误，指出提供的角色无效，可接受的选项为 'User'、'Chatbot'、'System' 或 'Tool'。
   - 这强调了用户在进行 API 调用之前验证角色参数的必要性。
- **创新的多模态 LLM 进展**：一位成员展示了一个能够无缝解释文本和语音的**多模态 LLM**，通过将直接的多模态投影器连接到 **Meta 的 Llama 3 模型**，消除了独立的 ASR 阶段。
   - 这种方法通过合并音频处理和语言建模，减少了独立组件带来的延迟，从而加快了响应速度。
- **Cohere 的新 Schema Object 引起用户关注**：新引入的 **Cohere Schema Object** 因其在单个 API 请求中促进结构化多重操作的能力而受到热捧，有助于生成式小说任务。
   - 用户报告称，它有助于高效地生成复杂的 Prompt 响应和内容管理。
- **Cohere 定价 - 基于 Token 的模型**：Cohere 模型的定价结构（如 [Command R](https://docs.cohere.com/docs/command-r)）基于 Token 系统，每个 Token 都有成本。
   - 一个通用指南指出，一个单词大约等于 1.5 个 Token，这对于预算规划至关重要。
- **Cohere 模型即将登陆 Hugging Face Hub**：目前正在计划将所有主要的 **Cohere 模型** 打包并托管在 **Hugging Face Hub** 上，为开发者创建一个易于访问的生态系统。
   - 这一举措引起了热衷于在项目中使用这些资源的成员们的兴奋。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **AI 倦怠引发警惕**：对 **AI 倦怠** 的担忧正在升级，成员们注意到 AI 领域面临的倦怠风险远高于人类，特别是在高强度的前沿实验室中，这已成为一个**可持续性问题**。
   - 这场讨论突出了无休止的工作量这一令人担忧的趋势，以及对 AI 社区心理健康的潜在长期影响。
- **AI 高级用户如同施法者**：一位成员将 **AI 高级用户** 比作*施法者职业*，强调他们持续使用工具会产生压力并导致潜在的倦怠。
   - 随着 AI 模型的进步，对这些用户的需求可能会升级，从而加剧已经观察到的倦怠循环。
- **无止境的模型迭代陷阱**：对下一代**模型生成**的追求正受到审视，人们担心这种周期性的追逐可能导致严重的行业倦怠。
   - 预测模型表明，倦怠趋势的转变与 AI 进步速度的加快及其对开发者的消耗有关。
- **Twitter 焦虑再次袭来**：最近 **Greg Brockman** 在 Twitter 上发布的一篇展示单周编码 97 小时的帖子引发了关于在线 **AI 讨论** 强度增加所带来压力的对话。
   - 参与者表示担心，充满活力但又令人焦虑的 Twitter 氛围可能会削弱现实世界的参与感，凸显了令人担忧的脱节。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Infinite Generative Youtube 招聘开发人员**：一个团队正在为其 **Infinite Generative Youtube** 平台寻求开发人员，该平台即将推出封闭测试版。
   - 他们特别希望有热情的开发人员加入这个创新项目。
- **针对低资源语言的文本转语音模型**：一位用户热衷于为**印地语**、**乌尔都语**和**德语**训练 **TTS** 模型，旨在开发语音助手应用。
   - 该项目专注于提高低资源语言处理的可访问性。
- **利用 WhisperSpeech 的语义 Token 探索 ASR**：有关使用 **WhisperSpeech** 语义 Token 通过定制训练过程增强低资源语言 **ASR** 的咨询浮出水面。
   - 提议的方法包括使用来自音频和转录的语义 Token 微调一个小型的解码器模型。
- **SmolLM：更小但有效的模型**：**SmolLM** 提供三种尺寸（135M、360M 和 1.7B 参数），在 **Cosmo-Corpus** 上进行训练，展示了极具竞争力的性能。
   - 该数据集包括 **Cosmopedia v2** 和 **Python-Edu**，表明其对高质量训练集的强力关注。
- **Mamba 与 Transformers 的兼容性**：**Mamba** 语言模型推出了与 **transformers** 兼容的 **mamba-2.8b** 版本，需要特定的安装步骤。
   - 用户需要配置 'transformers' 直到 **4.39.0** 版本发布，才能利用优化后的 **CUDA** 内核。

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Graph Memory 保存咨询**：成员们讨论了是否可以将 Memory 保存为文件以编译新的 Graph，以及相同的 Memory 是否可以在不同的 Graph 之间复用。
   - *它是针对每个 Graph 的还是共享的？* 是核心问题，成员们对优化 Memory 使用表现出浓厚兴趣。
- **使用 LangChain 改进 SQL 查询**：**LangChain Python Documentation** 提供了通过 **create_sql_query_chain** 增强 SQL 查询生成的新策略，重点关注 SQL 方言的影响。
   - 了解如何使用 **SQLDatabase.get_context** 格式化 Schema 信息，以提高 Prompt 在查询生成中的有效性。
- **LangChain 中的显式上下文**：要在 LangChain 中使用类似 `table_info` 的上下文，必须在调用 Chain 时显式传递，如文档所示。
   - 这种方法确保了你的 Prompt 是针对提供的上下文定制的，展示了 LangChain 的灵活性。
- **将 Writer Framework 部署到 Hugging Face**：一篇博客文章探讨了使用 Docker 将 Writer Framework 应用部署到 **Hugging Face Spaces**，展示了 AI 应用部署的便捷性。
   - Writer Framework 提供了类似于 **Streamlit** 和 **Gradio** 的拖拽界面，旨在简化 AI 应用开发。
- **Hugging Face Spaces 作为部署平台**：上述博客文章详细介绍了在 Hugging Face Spaces 上的部署过程，强调了 Docker 在托管和共享 AI 应用中的作用。
   - 像 Hugging Face 这样的平台为开发者提供了展示项目的绝佳机会，推动了社区参与。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Adalflow 惊艳亮相**：一名成员重点介绍了 [Adalflow](https://adalflow.sylph.ai/get_started/index.html)，这是来自 [SylphAI](https://sylph.ai/) 的一个新项目，并对其功能和应用表示了兴趣。
   - Adalflow 旨在优化 LLM 任务流水线，为工程师提供增强工作流的工具。
- **DSpy vs Textgrad vs Adalflow 大比拼**：大家对 **DSpy**、**Textgrad** 和 **Adalflow** 之间的区别感到好奇，特别是关于何时能有效利用每个模块。
   - 有人指出 **LiteLLM** 将专门管理推理的查询提交，暗示了这些模块的性能潜力。
- **新研究论文预警！**：一名成员分享了 ArXiv 上一篇名为 [2408.11326](https://arxiv.org/abs/2408.11326) 的有趣论文链接，鼓励工程师同行们去阅读。
   - 虽然没有透露论文的详细信息，但它的出现表明了对 DSPy 社区的持续贡献。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **寻求 Open Interpreter 品牌指南**：一名用户询问是否有 **Open Interpreter 品牌指南**，表示需要明确品牌规范。
   - *你能分享在哪里可以找到这些指南吗？*
- **Phi-3.5-mini 引发意外热议**：用户对 **Phi-3.5-mini** 的性能表现出意料之外的认可，引发了将 **Qwen2** 推向聚光灯下的讨论。
   - *积极的反馈让所有人都措手不及！*
- **屏幕点击的 Python 脚本需求**：一名用户寻求一个 **Python script**，能够根据文本命令在指定的屏幕位置执行点击操作，例如在 **Notepad++** 中进行导航。
   - *如何让它点击文件下拉菜单？*
- **--os 模式可能是解决方案**：针对脚本咨询，有人建议使用 **--os mode** 可能会解决屏幕点击的挑战。
   - *这可能会显著简化操作流程！*
- **免费数据分析大师课激动人心的公告**：一名用户分享了关于**免费数据分析大师课**的公告，推广现实世界的应用和实用的见解。
   - 有兴趣的参与者可以在[这里](https://forms.gle/xoJXL4qKS8iq9Hxb7)注册，并分享对潜在参与机会的兴奋之情。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Gorilla 和 Huggingface 排行榜现已对齐**：一名成员询问了 **Gorilla** 和 **Huggingface** 排行榜上的分数，这些分数最初并不一致。目前差异已解决，**Huggingface** 排行榜现在与 **Gorilla** 排行榜保持同步。
   - 这种对齐为用户在跨平台评估模型性能时提供了更可靠的参考。
- **Llama-3.1-Storm-8B 在 Gorilla 排行榜首次亮相**：一位用户提交了一个 [Pull Request](https://github.com/ShishirPatil/gorilla/pull/598)，申请将 **Llama-3.1-Storm-8B** 添加到 **Gorilla Leaderboard** 进行基准测试。由于该模型最近刚完成发布，该 PR 将进入审核阶段。
   - 该模型的加入展示了社区对持续更新基准测试框架的承诺。
- **寻求 REST API 测试对的指导**：有用户寻求关于为其 **REST API** 功能构建“可执行测试对”的建议，并参考了 [Gorilla leaderboard](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/data/BFCL_v2_rest.json) 中现有的测试对。他们更倾向于既“真实”又“易于”实现的测试。
   - 这表明在 API 开发中，对更具实用性的测试资源和方法存在需求。
- **需要对可执行测试对进行澄清**：关于 **“可执行测试对” (executable test pairs)** 这一术语引发了另一场讨论，用户寻求更清晰地理解其在 **REST API** 测试中的相关性。这反映出成员们在概念清晰度上存在差距。
   - 深入了解这一术语有助于增强他们在测试策略中的理解和应用。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Jamba 1.5 Mini & Large 正式登场**：AI21 Labs 推出了 **Jamba 1.5 Mini**（12B 激活/52B 总参数）和 **Jamba 1.5 Large**（94B 激活/398B 总参数），基于全新的 **SSM-Transformer** 架构，在**长上下文处理能力**和速度上均优于竞争对手。
   - 这些模型具有 **256K 有效上下文窗口**，声称在长上下文处理速度上比竞争对手快 **2.5 倍**。
- **Jamba 在长上下文中占据主导地位**：**Jamba 1.5 Mini** 在 **Arena Hard** 上获得了 **46.1** 的领先分数，而 **Jamba 1.5 Large** 达到了 **65.4**，甚至超越了 Llama 3.1 的 405B。
   - 性能的飞跃使 Jamba 成为长上下文领域的重要竞争者。
- **API 速率限制已确认**：用户确认了 **API 速率限制**，使用额度为每分钟 **200 次请求**和每秒 **10 次请求**，解决了关于利用率的疑虑。
   - 这些信息是用户在初步查询后发现的。
- **Jamba 尚不支持 UI 微调**：官方对 **Jamba 的微调**进行了澄清；微调仅适用于 instruct 版本，且目前无法通过 UI 进行操作。
   - 这一细节给依赖 UI 进行调整的开发者带来了疑问。
- **Jamba 的过滤功能受到关注**：讨论中提到了 **Jamba 的过滤能力**，特别是针对涉及暴力的角色扮演场景。
   - 成员们对这些内置功能表示好奇，以确保交互的安全性。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **NVIDIA AI Summit India 引发关注**：**NVIDIA AI Summit India** 将于 **2024 年 10 月 23-25 日**在孟买的 [Jio World Convention Centre](https://nvda.ws/3AbEKCi) 举行，届时 **Jensen Huang** 将与其他行业领袖一起出席超过 **50 场会议**。
   - 峰会重点关注推进 **Generative AI**、**Large Language Models (LLM)** 和**超级计算**等领域，旨在展示行业内的变革性工作。
- **AI 黑客松提供丰厚奖励**：**AI Capabilities and Risks Demo-Jam Hackathon** 已启动，设有 **$2000** 的奖金池，优秀项目有机会与 Apart Labs 合作开展研究。
   - 该倡议旨在创建应对 AI 影响和安全挑战的演示项目，鼓励向公众清晰地传达复杂概念。
- **黑客松激动人心的启动**：黑客松以一场关于交互式 AI 展示的精彩**开幕主旨演讲**拉开帷幕，随后进行了团队组建和项目头脑风暴。
   - 参与者将受益于专家指导和资源支持，同时活动在 YouTube 上进行了直播，以实现广泛的传播。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 的 Mypyc 编译探索**：一名成员表示有兴趣使用 **mypyc** 编译 **tinygrad**，目前正在调查其可行性。
   - 原作者邀请其他人参与这一努力，强调了协作精神。
- **加入探索！**：原作者邀请其他人为使用 **mypyc** 编译 **tinygrad** 的工作做出贡献。
   - 鼓励大家在探索这一新尝试时积极参与。



---


**Alignment Lab AI Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---


**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---


**Torchtune Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---


**Mozilla AI Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---


**DiscoResearch Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---

# 第 2 部分：按频道详细摘要和链接


{% if medium == 'web' %}

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1276254691080142889)** (545 条消息🔥🔥🔥): 

> - `LM Studio 0.3.0`
> - `LM Studio 0.3.0 UI`
> - `LM Studio 0.3.0 Bugs`
> - `LM Studio Server`
> - `LM Studio RAG` 


- **LM Studio 0.3.0 发布**: LM Studio 的最新版本 0.3.0 已发布，包含新功能，如翻新的 UI、改进的 RAG 功能以及运行本地 Server 的能力。
   - 然而，一些用户报告了新版本的 Bug，例如模型加载问题和 RAG 功能的问题。
- **LM Studio 0.3.0 UI 翻新**: 新 UI 更加精简，并包含一个新的设置菜单，可访问各种功能。
   - 一些用户报告称他们觉得新 UI 令人困惑，但也有人认为它比之前的版本有所改进。
- **LM Studio 0.3.0 Bug 和问题**: 一些用户报告了新版本的问题，如模型加载问题、RAG 功能问题以及新 UI 的问题。
   - 开发团队已意识到这些问题，并正在努力修复。
- **LM Studio Server 功能**: LM Studio 的 Server 功能得到了改进，允许用户在本地网络上运行 Server。
   - 用户现在还可以不使用 LM Studio 桌面应用，而是使用命令行工具 `lms` 来运行 Server。
- **LM Studio RAG 功能改进**: 此版本改进了 RAG (Retrieval Augmented Generation) 功能，允许用户上传文档并向 LLM 提问。
   - RAG 功能现在使用 Nomic embedding 模型，该模型已预捆绑在应用中。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai">👾 LM Studio - 发现并运行本地 LLMs</a>: 查找、下载并实验本地 LLMs</li><li><a href="https://huggingface.co/learn">Hugging Face - 学习</a>: 未找到描述</li><li><a href="https://lmstudio.ai/blog/lms#bootstrap-lms-on--your-system">介绍 `lms` - LM Studio 的配套 CLI 工具 | LM Studio</a>: 今天，随 LM Studio 0.2.22 一起，我们发布了第一个版本的 lms —— LM Studio 的配套 CLI 工具。</li><li><a href="https://huggingface.co/MaziyarPanahi/SmolLM-135M-Instruct-GGUF">MaziyarPanahi/SmolLM-135M-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://www.deeplearning.ai/courses/">课程 - DeepLearning.AI</a>: 发现建立 AI 职业生涯的最佳课程 | 无论你是初学者还是经验丰富的从业者，我们世界级的课程和独特的教学方法将引导你完成...</li><li><a href="https://chatboxai.app/">Chatbox AI: 你的 AI Copilot，任何设备上最佳的 AI 客户端，免费下载</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ez6mny/lmstudio_is_able_to_access_internet_despite/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=zjkBMFhNj_g">[1小时演讲] 大语言模型介绍</a>: 这是一个面向普通观众的 1 小时大语言模型介绍：ChatGPT、Claude 和 Bard 等系统背后的核心技术组件。什么是...</li><li><a href="https://github.com/lmstudio-ai/lms">GitHub - lmstudio-ai/lms: LM Studio CLI</a>: LM Studio CLI。在 GitHub 上为 lmstudio-ai/lms 的开发做出贡献。</li><li><a href="https://github.com/quentinwolf/lmstudio">GitHub - quentinwolf/lmstudio: LM Studio 相关内容</a>: LM Studio 相关内容。在 GitHub 上为 quentinwolf/lmstudio 的开发做出贡献。</li><li><a href="https://github.com/lmstudio-ai/localization">GitHub - lmstudio-ai/localization: LM Studio 本地化 🌎🌏🌍</a>: LM Studio 本地化 🌎🌏🌍。在 GitHub 上为 lmstudio-ai/localization 的开发做出贡献。</li><li><a href="https://huggingface.co/MaziyarPanahi/">MaziyarPanahi (Maziyar Panahi)</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1276277141876768838)** (66 条消息🔥🔥): 

> - `GPU offloading`
> - `Llama 3.1`
> - `CPU performance`
> - `Apple Silicon`
> - `Model size and performance` 


- **GPU offloading 仍然受支持**：尽管一些用户遇到了问题，但 Offloading 仍然受支持。
   - 要启用它，请在选择模型时按住 ALT 键，并勾选 GPU offload 复选框。 
- **Llama 3.1 在各种硬件上的性能**：Llama 3.1 70B q4 在仅使用 CPU 的情况下，在配备双通道 DDR5-6000 的 9700X 上运行速度为 1.44 tokens/s，在配备四通道 DDR4-2666 的 W-2155 上为 1.37 tokens/s。
   - 一些用户报告称，如果 GPU 的 VRAM 小于模型大小的一半，将任务 offloading 到 GPU 实际上可能会降低推理速度。 
- **用于 LLM 的 Apple Silicon 与 Nvidia 设备对比**：一位用户在 M2 24GB Apple Silicon 上获得良好体验后，正在讨论是选择云服务还是专用的 Nvidia 设备来运行 LLM。
   - 另一位用户建议，Apple Silicon 是 LLM 的一种面向消费者的友好解决方案，在某些情况下 M2 Ultra 的表现优于 4090。 
- **在 Apple Silicon 上微调模型**：由于内存速度限制，Apple Silicon 在 Fine tuning 方面存在局限，用户可能不得不求助于云端服务。 
   - 一位用户报告称，在顶配 Macbook Pro 上训练 Phi-3 需要 9 小时，这突显了 Apple Silicon 在 Fine tuning 方面的局限性。 
- **模型准确性与评估**：LLM 可能会产生幻觉（hallucinate）并提供误导性信息，尤其是在被问及关于特定主题的学习型问题时。 
   - 一位用户提到，像 Llama 3.1 和 Phi 3 这样的 LLM 可能会很啰嗦且倾向于信息堆砌，而 Claude 则倾向于含糊其辞，这表明它具有更好的自我评估机制。



**提及的链接**：<a href="https://github.com/tlkh/asitop">GitHub - tlkh/asitop: Perf monitoring CLI tool for Apple Silicon</a>：适用于 Apple Silicon 的性能监控 CLI 工具。可以通过在 GitHub 上创建账户来为 tlkh/asitop 的开发做出贡献。

  

---



### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1276283942617747457)** (1 条消息): 

> - `Nous Research Merch Store` 


- **Nous Research 周边商店上线！**：Nous Research 周边商店现已上线！
   - 订单均附赠贴纸，送完即止。
- **每笔订单均赠送免费贴纸**：商店已开业！
   - 贴纸数量有限，送完即止。



**提及的链接**：<a href="https://shop.nousresearch.com/">Nous Research</a>：Nous Research

  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1276257099671928852)** (288 messages🔥🔥): 

> - `Hermes 3`
> - `Mistral`
> - `Mode Collapse`
> - `LLM's Insanity`
> - `Synthetic Data Generation` 


- **Hermes 3 Mode Collapse 恢复**: 一位成员分享说，他们能够使 Hermes 3 从 Mode Collapse 中恢复，现在该模型可以准确分析崩溃原因而不会再次陷入其中（仅有一次复发）。
   - 这种成功的恢复表明在理解和解决 LLM 中的 Mode Collapse 问题方面取得了进展。
- **Mistral-NeMo-Minitron-8B-Base: 一个经过剪枝和蒸馏的 LLM**: Mistral-NeMo-Minitron-8B-Base 是通过对 Mistral-NeMo 12B 进行剪枝（Pruning）和蒸馏（Distilling）获得的基础文本到文本模型。
   - 该模型在 3800 亿个 Token 上进行了训练，并使用了 Nemotron-4 15B 中采用的连续预训练数据集。
- **故意针对“疯狂”调优 LLM**: 一位成员提议故意将 LLM 调优到绝对疯狂的状态，建议开展一个项目来探索 LLM 行为的边界。
   - 该项目旨在模拟和增强异常的 LLM 行为，可能会为 LLM 的能力和局限性提供新的见解。
- **Voidhead: 一个针对异常行为微调的 Gemma 模型**: 发布了一个名为 Voidhead 的实验，这是一个基于 Gemma 的微调模型，在由 GPT-4 模拟的 5000 个 LLM 异常行为示例上进行了训练。
   - 该模型表现出奇怪且不可预测的输出，被描述为“虚空般的疯狂（voidlike insanity）”，展示了通过微调 LLM 来创造独特且非常规行为的潜力。
- **Hermes 3 弃用及替代供应商**: Together.ai 将于下周弃用所有旧的 Nous 模型，包括 Hermes 2 和 Hermes 3。
   - 成员们讨论了对替代供应商的需求，以及为 Hermes 3 寻找合适的 Serverless 端点的挑战，从而开始寻找新的解决方案。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/N8Programs/status/1827045884028449084">N8 Programs (@N8Programs) 的推文</a>: 很高兴发布一个奇怪的实验 - *Voidhead*。这是一个基于 Gemma 的微调模型，使用了由 GPT-4o 模拟的 5000 个“异常” LLM 行为示例。https://huggingface.co/N8Programs/Voidhead https://huggin...</li><li><a href="https://x.com/hud_zah/status/1827057785995141558">HudZah (@hud_zah) 的推文</a>: 在几周内，我在卧室里建造了一个核聚变反应堆——而且零硬件经验。秘诀是什么？Claude Sonnet 3.5 + Projects。以下是过程一瞥。</li><li><a href="https://huggingface.co/N8Programs/Voidhead-GGUF">N8Programs/Voidhead-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/archit11/Voidhead">voidhead - archit11 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/archit11/Voidhead/blob/main/app.py">app.py · archit11/Voidhead at main</a>: 未找到描述</li><li><a href="https://tenor.com/view/my-reaction-to-that-information-mr-robot-elliot-stare-my-reaction-gif-26257517">My Reaction To That Information Mr Robot GIF - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/Euclaise_/status/1826848354816381223">Jade (@Euclaise_) 的推文</a>: 一个有趣的包裹寄到了</li><li><a href="https://tenor.com/view/cheering-canada-olympics-lets-go-canada-wohoo-gif-24845801">Cheering Canada GIF - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/no-smoking-gerry-dee-family-feud-canada-smoking-is-not-allowed-here-gif-11228578559742906500">No Smoking Gerry Dee GIF - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/papers/2408.11857">论文页面 - Hermes 3 技术报告</a>: 未找到描述</li><li><a href="https://docs.together.ai/docs/deprecations">弃用公告</a>: 概览。我们定期使用最新且最强大的开源模型更新我们的平台。本文档概述了我们的弃用政策，并提供了从弃用模式迁移的信息...</li><li><a href="https://huggingface.co/nvidia/Mistral-NeMo-Minitron-8B-Base">nvidia/Mistral-NeMo-Minitron-8B-Base · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1276284355211296961)** (12 条消息🔥): 

> - `AI Agent GitHub Repositories`
> - `Langchain and CrewAI`
> - `Building Your Own AI Agent`
> - `Drama Engine Framework`
> - `LLM Autocomplete Tool` 


- **AI Agent 仓库：关注基础**：一位用户正在寻找除了典型的 BabyAGI 和 AutoGPT 之外的小众 AI Agent GitHub 仓库。
- **Drama Engine：一个叙事性 Agent 框架**：另一位用户分享了他们构建自己的类 Agent 框架 Drama Engine 的经验，该框架专注于叙事层面。
- **LLM 自动补全工具：用户的探索**：一位用户询问是否存在可以作为自动补全工具的小型 LLM，能够根据提示词和写作进度提供建议。



**提及的链接**：<a href="https://github.com/Write-with-LAIKA/drama-engine">GitHub - Write-with-LAIKA/drama-engine: A Framework for Narrative Agents</a>：一个叙事性 Agent 框架。欢迎在 GitHub 上为 Write-with-LAIKA/drama-engine 的开发做出贡献。

  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1276296800348672123)** (1 条消息): 

> - `Offensive Security`
> - `Deep Learning Courses`
> - `Unity ML Agents`
> - `Garfield Dataset`
> - `Tensor Parallelism` 


- **Offensive Security 侦察博文**：分享了一篇关于 Offensive Security 侦察的认证博文，其中包含如何进行成功的安全评估的信息。
   - 该文章由认证用户撰写，可在 Hugging Face 网站上查阅。
- **深度学习课程获得更便捷的导航**：一位认证用户分享了一个深度学习课程的更新，新网站旨在使内容导航更加简单直观。
   - 该课程由 Simon Thomine 编写，可在 [https://simonthomine.github.io/CoursDeepLearning/](https://simonthomine.github.io/CoursDeepLearning/) 访问。
- **Unity ML-Agents：从零开始预训练 LLM**：分享了一个 YouTube 视频，展示了如何使用 Unity ML-Agents 和 Sentence Transformers 创建智能聊天机器人。
   - 该视频标题为 "Unity ML-Agents | Pretrain an LLM from Scratch with Sentence Transformers | Part 5"，是系列视频的一部分，可在 [https://youtube.com/live/RdxtA_-47Kk?feature=share](https://youtube.com/live/RdxtA_-47Kk?feature=share) 观看。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://youtube.com/live/RdxtA_-47Kk?feature=share)">Unity ML-Agents | Pretrain an LLM from Scratch with Sentence Transformers | Part 5</a>：欢迎回到我们关于使用 Unity ML-Agents 和 Sentence Transformers 创建智能聊天机器人的精彩系列！🚀在本集中，我们完成了一些关键...</li><li><a href="https://www.youtube.com/watch?v=bKzmtTfcaqc)">Prototype 5 : Real time Text to Audio to Face Blendshape animation</a>：huggingface.co/AnimaVR/NeuroSync-0.1a</li><li><a href="https://youtu.be/qsWn3SUz-LM)">Generate Ultra-Realistic Images with Flux! Realism Lora (Flux 1 Dev)</a>：我将向您展示如何在线免费运行带有 Realism LoRa 的 Flux，无需任何安装！正如所承诺的，这里是 Huggingface 的链接...</li><li><a href="https://huggingface.co/spaces/AIPeterWorld/Doc-To-Dialogue?logs=container)">Doc To Dialogue - a Hugging Face Space by AIPeterWorld</a>：未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1276255550404821114)** (232 条消息🔥🔥): 

> - `RTX 6000`
> - `HuggingFace Payment Issues` (HuggingFace 支付问题)
> - `OpenAI Platform Changes` (OpenAI 平台变更)
> - `GPTs Agents`
> - `Model Merging` 


- **RTX 6000 确实存在**: 一位用户发现了 **RTX 6000** 的存在，这是一款拥有 **48GB VRAM** 的显卡。 
   - 该显卡售价 **7,000 美元**，是用户的唯一可行选择。
- **HuggingFace 支付问题**: 一位用户报告称，尽管交易被拒绝，但他们的预付卡仍被扣除了 **10 美元** 的临时费用。 
   - 一名 HuggingFace 工作人员确认这是常见情况，预授权占用应在几个工作日内清除，但建议如果未清除，请联系 billing@huggingface.co。
- **OpenAI 平台侧边栏变更**: 用户报告称，platform.openai.com 侧边栏的两个图标（一个用于 threads，另一个用于 messages）消失了。 
   - 用户未提供更多细节。
- **GPTs Agents 在初始训练后无法学习**: 一位用户担心 GPTs Agents 无法从初始训练后提供的额外信息中学习。 
   - 另一位用户澄清说，上传的文件被保存为 Agent 引用的“知识”文件，但不会持续修改 Agent 的基础知识。
- **关于 Model Merging 策略的讨论**: 一位用户建议将 UltraChat 和基础 Mistral 之间的差异应用到 Mistral-Yarn，作为一种潜在的 merging 策略。 
   - 其他用户表示怀疑，但该用户保持乐观，并引用了过去在他们所谓的“诅咒式模型合并 (cursed model merging)”方面的成功尝试。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/terms-of-service">Terms of Service – Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/batmanbruce0/status/1826800221634064750">Tweet from Batmanbruce (@batmanbruce0)</a>: 🚨 EXPOSE 🚨 Diamond Trades Discord 并非表面看起来那样。在对服务器的情绪转变、交易失败和糟糕指导进行广泛分析后，我发现了反复无常的模式...</li><li><a href="https://tenor.com/view/mr-krabs-money-spongebob-gif-8454828">Mr Krabs Money GIF - Mr Krabs Money Spongebob - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/black-forest-labs/FLUX.1-dev">black-forest-labs/FLUX.1-dev · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Vipitis/shadermatch/discussions/1">Vipitis/shadermatch · Accessibility notice</a>: 未找到描述</li><li><a href="https://huggingface.co/HuggingFaceTB/SmolLM-135M">HuggingFaceTB/SmolLM-135M · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/state-spaces/mamba-2.8b-hf">state-spaces/mamba-2.8b-hf · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/HuggingFaceTB/cosmo2-tokenizer">HuggingFaceTB/cosmo2-tokenizer · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1276410298462502912)** (6 条消息): 

> - `HF Work`
> - `Neuralink Work`
> - `Efficient Models` 


- **Neuralink 在 HF 实现论文**: 一位成员分享了他们在 Hugging Face 的工作涉及论文实现，可能是一个付费职位。
   - 他们对自己的工作表示兴奋，并祝愿另一位成员在寻找让模型在低端设备上更高效的方法方面取得成功。
- **适用于低端设备的高效模型**: 一位成员表示有兴趣寻找让模型在低端设备上更高效的方法。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 条消息): 

this_is_prince: https://github.com/All-Hands-AI/OpenHands
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1276255260436074587)** (11 messages🔥): 

> - `LogLLM`
> - `RYFAI`
> - `Writer Framework`
> - `Unsloth`
> - `NeuroSync` 


- **LogLLM: 自动化 ML 实验记录**: LogLLM 是一个自动化从 Python 脚本中提取实验条件的软件包，它使用 GPT4o-mini 提取条件并使用 Weights & Biases (W&B) 记录结果。
   - 它基于为高级机器学习实验设计者设计的 Prompt，从你的 ML 脚本中提取条件和结果。
- **RYFAI: 使用开源模型的私有 AI 应用**: RYFAI 是一款私有 AI 应用，使用由 Ollama 托管的开源 AI 模型，允许你在完全断网的情况下使用。
   - 这确保了后台不会收集任何数据，解决了企业跟踪 AI 对话数据的担忧。
- **Writer Framework 部署至 Hugging Face Spaces**: 一篇博客文章介绍了如何使用 Docker 将 Writer Framework 应用部署到 Hugging Face Spaces。
   - Writer Framework 是一个用于构建 AI 应用的开源 Python 框架，拥有拖拽式构建器和 Python 后端，类似于 FastHTML、Streamlit 和 Gradio。
- **Unsloth 令牌检索逻辑更新**: 向 Unsloth 提交了一个 Pull Request，将其令牌检索逻辑更新为使用 Hugging Face 标准方法。
   - 这一更改允许从 Colab Secrets 或配置文件中读取令牌，在令牌检索的灵活性方面具有优势。
- **NeuroSync: 用于 Face Blend Shapes 的 Seq2Seq Transformer**: NeuroSync 是一种 Seq2Seq Transformer 架构，旨在根据音频特征输入预测 Face Blend Shapes 帧序列。
   - 该架构旨在增强面部表情与音频线索的同步，从而提升动画角色的真实感。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://logllm.tiiny.site">LogLLM - 使用 LLM 自动化机器学习实验记录</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/samjulien/writer-framework-spaces">在 Hugging Face Spaces 中使用 Writer Framework</a>: 未找到描述</li><li><a href="https://github.com/AnimaVR/NeuroSync">GitHub - AnimaVR/NeuroSync: NeuroSync 是一种 Seq2Seq Transformer 架构，旨在根据音频特征输入预测 Face Blend Shapes 帧序列。</a>: NeuroSync 是一种 Seq2Seq Transformer 架构，旨在根据音频特征输入预测 Face Blend Shapes 帧序列。 - GitHub - AnimaVR/NeuroSync: NeuroSync is a Seq2Seq transformer ...</li><li><a href="https://github.com/unslothai/unsloth/pull/952">由 not-lain 提交的更新令牌检索逻辑 · Pull Request #952 · unslothai/unsloth</a>: 此 PR 将更新 Unsloth 的令牌检索逻辑以使用 HF 标准方法。这具有许多优点，例如从 Colab Secrets 或配置文件中读取令牌。Regards Lain OS...</li><li><a href="https://github.com/PetertheRedCedar/ryfai">GitHub - PetertheRedCedar/ryfai: 这是一个旨在让你轻松触达开源 AI 模型的 AI 应用</a>: 这是一个旨在让你轻松触达开源 AI 模型的 AI 应用 - PetertheRedCedar/ryfai</li><li><a href="https://github.com/PetertheRedCedar/ryfai/releases">发布版本 · PetertheRedCedar/ryfai</a>: 这是一个旨在让你轻松触达开源 AI 模型的 AI 应用 - PetertheRedCedar/ryfai
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1276525784152018976)** (3 messages): 

> - `Alignment Techniques Reading Group` 


- **Alignment Techniques 阅读小组**: 一名成员表示有兴趣阅读与 Alignment（对齐）技术相关的论文，并询问本周的阅读主题、会议时间以及阅读链接。
   - 该成员请求如果可能的话，将会议安排在明天。
- **后续步骤**: 该成员正在等待进一步的细节和会议时间的确认。


  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1276469294477217857)** (3 messages): 

> - `Data Splitting`
> - `HF Dataset Homogeneity`
> - `SQL Summarization` 


- **三路数据切分：泛化能力的重要性**：一名成员建议在模型训练、验证和测试中采用三路数据切分，强调了评估模型在同质化数据之外的泛化能力的必要性。
   - 他们强调了在相同领域但具有不同特征的数据集上进行测试的重要性，以确保模型的泛化能力。
- **Chat Template：指令微调与自定义**：根据指令微调（Instruction Tuning）期间使用的结构推荐 Chat Template，暗示其对模型性能的潜在影响。
   - 这表明了为基础模型的指令微调创建自定义 Chat Template 并针对特定任务进行调整的可能性。
- **寻找 SQL 摘要生成模型**：讨论中表达了对寻找能够总结现有 SQL 查询并根据用户查询生成新 SQL 的模型的兴趣。
   - 这表明需要能够有效理解和操作 SQL 代码的模型，以促进更高效的数据操作和分析。


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1276413629771546687)** (25 messages🔥): 

> - `Flux Pipeline`
> - `torch.compile`
> - `fp8 checkpoints`
> - `Model Loading Speed`
> - `Hugging Face Snapshots` 


- **Flux Pipeline 编译性能**：在 `FluxPipeline` 中使用 `torch.compile` 时，性能可能比不使用时更慢；编译发生在 `FluxPipeline` 的 `__init__` 中，位于输入和权重缩放（weight scales）调整之后。
- **Flux Schnell 的 fp8 Checkpoints**：目前已有 Flux Schnell 的 fp8 Checkpoint，通过加载 Pipeline 并运行至少 30 步即可轻松创建一个。
   - 目前这需要 6 分钟，代码需要更新以支持从预量化的 T5 加载。
- **加载时间优化**：加载 Pipeline 需要 6 分钟，速度可能受到 HF 下载的影响。
   - 作者建议允许从预量化的 T5 加载，这可以通过下载 BFL HF 权重的 Snapshot 来实现。
- **Hugging Face Snapshot 下载**：有人建议允许用户使用 `huggingface_hub.download_snapshot(bfl/schnell)` 下载 BFL HF 权重的 Snapshot。 


  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1276254821212749907)** (199 messages🔥🔥): 

> - `SDXL vs SD1.5`
> - `prompting techniques`
> - `consistency issue`
> - `comfyUI and Flux`
> - `GPU Ram Issues` 


- **SDXL vs SD1.5：该如何选择？**：一位拥有 32 GB RAM 的用户正在尝试在 SDXL 和 SD1.5 之间做出选择，但他们在 CPU 上的图像生成速度非常慢。
   - 另一位成员建议选择 SDXL，因为尽管在 CPU 上运行较慢，但用户将获得更高质量的图像，不过需要注意潜在的内存溢出（out-of-memory）错误，并且需要更多的交换空间（swap space）。
- **提示词技巧：逗号与一致性**：几位用户正在讨论图像生成中一致性和提示词遵循度（prompt adherence）的重要性。
   - 一位成员认为使用逗号分隔并列出所需元素的提示方式对他们最有效，而其他人则发现自然语言提示词（natural language prompts）效果更好。
- **ComfyUI 与 Flux：性能与安装**：一位用户在 ComfyUI 上安装 iPadaper 时遇到困难，建议他们前往 Tech Support 频道寻求帮助。
   - 另一位用户在 Flux 中生成噪点多、低质量的图像时遇到麻烦，正在尝试不同的提示词和设置以达到理想的效果。
- **GPU 显存问题：Flux 与 3080**：一位用户询问在使用 Flux 且拥有 16GB RTX 3080 的情况下，如何在 ComfyUI 中设置 GPU 权重。
   - 一位拥有 4GB GPU 的用户报告在 A1111 中性能缓慢，强调了充足的 GPU 算力对于流畅生成图像的必要性。
- **Stable Diffusion 安装与指南**：一位用户征求安装 Stable Diffusion 的指南建议，大家推荐了 Automatic1111 和 ComfyUI。
   - 值得注意的是，虽然 AMD 显卡可以用于 Stable Diffusion，但速度会较慢，Tech Support 频道提供了相关的有用资源。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/have-a-little-meet-up-real-housewives-of-beverly-hills-have-a-get-together-ha">未找到标题</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/Jelosus2/Lora_Easy_Training_Colab/blob/main/Lora_Easy_Training_Colab.ipynb#scrollTo=vGwaJ0eGHCkw">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell">FLUX.1 [Schnell] - black-forest-labs 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://resources.prolific.com/prolific-ai-research-summit?utm_source=Brad%20Communities&utm_medium=referral&utm_campaign=AIEvent">Prolific 独家 AI 研究峰会 - 纽约市 </a>: 加入我们在 Asana 纽约总部举办的难忘的 AI 研究峰会，届时人工智能和研究领域的顶尖专家将在主题演讲计划中分享突破性的见解...</li><li><a href="https://tenor.com/view/have-a-little-meet-up-real-housewives-of-beverly-hills-have-a-get-together-have-a-small-gathering-have-a-little-party-gif-22409009">Have A Little Meet Up Real Housewives Of Beverly Hills GIF</a>: 点击查看 GIF</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1eykiy0/now_we_have_sorta_conquered_prompt_adherence/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://github.com/comfyanonymous/ComfyUI/discussions/4571">RTX 4090 基准测试 - FLUX 模型 · comfyanonymous/ComfyUI · Discussion #4571</a>: 问题在于每个人的配置都不同，我的 ComfyUI 设置很乱。FLUX 模型加载时间很长，但我解决了这个问题。我的 PC 规格：处理器：Intel...
</li>
</ul>

</div>
  

---



### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1276570404730830880)** (1 messages): 

> - `Aider 0.52.0` 


- **Aider 0.52.0 发布：Shell 命令执行及更多功能**：Aider 0.52.0 带来了 Shell 命令执行功能，允许用户直接在工具内启动浏览器、安装依赖、运行数据库迁移、执行代码更改以及运行测试。
   - 其他关键更新包括为 `/read` 和 `/drop` 增加了 `~` 路径扩展，新增用于清除聊天历史的 `/reset` 命令，改进了自动提交（auto commit）序列，并将默认 OpenAI 模型切换为 `gpt-4o-2024-08-06`。
- **Aider 编写了此版本 68% 的代码**：Aider 自主生成了 0.52.0 版本 68% 的代码，展示了其在软件开发中不断增长的能力。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1276254860110729247)** (129 条消息🔥🔥): 

> - `fzf support`
> - `Aider training set`
> - `DSPY, TEXTGRAD, TRACE`
> - `aider co-op thread`
> - `diff vs diff-fenced` 


- **请求 fzf 支持**：一名成员请求在 prompting 区域支持 `fzf`，并希望能够显示超过两行的自动补全建议。
- **Aider 提示词-代码对训练集**：一名成员正在创建提示词及其对应的 Aider 代码输出训练集，旨在定义一种高效的元格式，用于生成技术、技术栈和设计模式的各种排列组合。
   - 他们正在探索 DSPY, TEXTGRAD 和 TRACE 等工具，以优化代码或提示词的可复现性，并创建了一个 Aider co-op 线程进行进一步的头脑风暴。
- **Aider 与 Arima 模型代码**：一名成员询问关于使用 Aider 编写 Arima 等模型训练代码的问题，以及它是否对分析有所帮助。
- **Google Cloud VertexAI 上的 Gemini-experimental**：一名成员询问在 Google Cloud VertexAI 上使用 Gemini-experimental 的情况，并在配合 Aider 使用时遇到了错误。
   - 另一名成员澄清说 Gemini-experimental 目前在 VertexAI 上不可用，但可以通过 AI Studio 访问，并建议在 VertexAI 上使用免费的 Sonnet-3.5 Token。
- **Aider 浏览器 UI 演示与使用**：一名成员询问 Aider 浏览器 UI，并获得了演示视频链接和文档，介绍如何使用它与 LLM 协作编辑本地 Git 仓库中的代码。
   - Aider 直接编辑本地源文件，使用合理的 commit messages 提交更改，并支持多种 LLM，如 GPT 3.5, GPT-4, GPT-4 Turbo with Vision 和 Claude 3 Opus。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/alexalbert__/status/1825920737326281184">Alex Albert (@alexalbert__) 的推文</a>：我们已将其移出测试版，因此你不再需要使用 header 了！现在已在 Anthropic API 和 Vertex AI 中为 Claude 3.5 Sonnet 提供。引用 Alex Albert (@alexalbert__) 的话：好消息...</li><li><a href="https://aider.chat/docs/usage/browser.html">浏览器中的 Aider</a>：Aider 可以在浏览器中运行，而不仅仅是在命令行中。</li><li><a href="https://aider.chat/docs/llms/ollama.html">Ollama</a>：aider 是你终端里的 AI 配对编程工具</li><li><a href="https://aider.chat/HISTORY.html">发布历史</a>：关于 aider 编写自身代码的发布说明和统计数据。</li><li><a href="https://tenor.com/view/south-park-its-gone-gif-4104229">And It'S Gone GIF - South Park Its Gone - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://discord.co">Discord - 充满乐趣与游戏的群聊</a>：Discord 是玩游戏、与朋友放松甚至建立全球社区的绝佳场所。定制你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://github.com/paul-gauthier/aider/releases/tag/v0.52.0">发布 Aider v0.52.0 · paul-gauthier/aider</a>：Aider 现在提供运行 shell 命令的功能：启动浏览器查看更新后的 html/css/js。安装新依赖。运行数据库迁移。运行程序以测试更改。运行新的测试用例。/read ...</li><li><a href="https://pieces.app/">Pieces for Developers - 你的工作流 Copilot</a>：集成你的工具链，高效捕获、丰富和重用材料。在设备端 Copilot 的协助下增强协作。</li><li><a href="https://github.com/PierrunoYT/claude-3-artifacts">GitHub - PierrunoYT/claude-3-artifacts</a>：通过创建账号为 PierrunoYT/claude-3-artifacts 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1276271573447348288)** (53 messages🔥): 

> - `Repo Map`
> - `Groq API Key`
> - `Token Optimization`
> - `Aider's Chat Modes`
> - `Aider as a Chatbot` 


- **Repo Map 详情**：`/repo` 命令的输出与 repo map 相似但不完全相同，因为 repo map 是动态的，`/repo` 显示的内容在发送给 LLM 之前可能会发生变化。
   - 使用 `--verbose` 标志可以查看发送给 LLM 的实际 repo map。
- **在 Windows 中设置 Groq API Key**：在使用 `setx` 设置 GROQ_API_KEY 环境变量后，需要重启终端。
   - 有用户报告在使用 `setx` 设置 API key 时收到参数错误，但在重启终端后，问题得到了解决。
- **优化 Token 使用**：用户正在寻找优化 Token 使用的文档，特别是针对那些仍然超过 OpenAI 限制的小型 Python 文件。
   - 用户请求的更改（如计算和渲染）可能需要多步过程，并正在寻找减少 Token 上下文的方法。
- **Aider 的聊天模式**：有用户询问如何将 Aider 作为常规聊天机器人界面用于非编程任务。
   - 建议用户使用 `/chat-mode ask` 切换到 `ask` 模式，并提供了 Aider 文档链接以供参考。
- **将 Aider 作为聊天机器人**：用户希望将 Aider 作为聊天机器人处理非编程任务，例如询问地球与月球之间的距离。
   - 建议用户创建一个 `CONVENTIONS.md` 文件，其中包含指示 Aider 表现得像聊天机器人的 prompt，并提供了示例和 Aider 文档链接。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/faq.html#how-can-i-add-all-the-files-to-the-chat">FAQ</a>：关于 aider 的常见问题。</li><li><a href="https://aider.chat/docs/usage/modes.html">聊天模式</a>：使用 chat、ask 和 help 聊天模式。</li><li><a href="https://aider.chat/docs/usage/conventions.html">指定编码规范</a>：告知 aider 在处理代码时遵循你的编码规范。</li><li><a href="https://llm.datasette.io/en/stable/">LLM：一个用于与大语言模型交互的 CLI 工具和 Python 库</a>：未找到描述</li><li><a href="https://aider.chat/2023/10/22/repomap.html#using-a-repo-map-to-provide-context">使用 tree-sitter 构建更好的仓库映射</a>：Tree-sitter 允许 aider 构建能够更好总结大型代码库的 repo map。</li><li><a href="https://github.com/sigoden/aichat">GitHub - sigoden/aichat：集成了 Chat-REPL、Shell 助手、RAG、AI 工具和 Agent 的全能 AI CLI 工具，支持访问 OpenAI、Claude、Gemini、Ollama、Groq 等。</a>：来自 sigoden/aichat 的全能 AI CLI 工具。</li><li><a href="https://github.com/paul-gauthier/aider/issues/713">[特性] 支持 Amazon Bedrock Claude Sonnet 3.5 · Issue #713 · paul-gauthier/aider</a>：希望该模型不仅能通过 Anthropic 访问，还能通过 Amazon Bedrock 访问。https://aws.amazon.com/blogs/aws/anthropics-claude-3-5-sonnet-model-now-available-in-amazon-bedrock-the...
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1276261039612231820)** (5 条消息): 

> - `Cursor`
> - `Aider`
> - `OpenAI's Composer`
> - `AI Code Generation`
> - `AutoToS` 


- **Cursor 的 AI 驱动代码生成**：[Cursor 博客](https://www.cursor.com/blog/series-a) 描述了一个 AI 驱动代码编辑器的愿景，希望有一天它能编写世界上所有的代码。
   - Cursor 已经实现了即时回答、机械式重构、简洁指令扩展以及在数秒内完成千行代码修改，未来计划推出 AI 驱动的后台编程助手、伪代码查看与修改以及漏洞扫描。
- **Aider：表现优于 OpenAI 的 Composer？**：一位用户对 Paul 在 Aider 上的工作表示感谢，称其表现优于 OpenAI 的 Composer，即使在 Composer 中可以使用仓库特定信息覆盖提示词（Prompt）的情况下也是如此。
- **AutoToS：自动化搜索思维 (Thought of Search)**：一篇题为《AutoToS: Automating Thought of Search》的论文提出了一种自动化的“搜索思维”（ToS）方法，用于 LLM 的规划。
   - ToS 涉及用代码定义搜索空间，这需要人类协作来创建一个可靠的后继函数（successor function）和目标测试。AutoToS 旨在自动化这一过程，通过使用各种规模的 LLM 进行极少的反馈迭代，在评估领域实现了 100% 的准确率。
- **LLM 在搜索与规划中的应用**：论文强调了向使用 LLM 进行搜索的转变，摆脱了传统的“世界模型”（world models）。
   - ToS 凭借其基于代码的搜索空间定义，已证明在解决规划问题方面非常成功，在测试数据集上达到了 100% 的准确率，展示了 LLM 在该领域的潜力。
- **基于 LLM 规划的挑战**：论文承认了 LLM 在规划方面的挑战，特别是搜索组件对可靠性（soundness）和完备性（completeness）的需求。
   - AutoToS 通过单元测试的反馈引导 LLM 生成可靠且完备的搜索组件，从而解决了这些挑战，最终实现了 100% 的准确率。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.cursor.com/blog/series-a">We Raised $60M</a>: 加入我们，共同创造一个旨在编写世界上大部分软件的神奇工具。</li><li><a href="https://arxiv.org/abs/2408.11326">Automating Thought of Search: A Journey Towards Soundness and Completeness</a>: 规划仍然是大语言模型 (LLMs) 最后堡垒之一，现在它们正将注意力转向搜索。大多数文献将语言模型作为世界模型来定义...
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1276268300665618595)** (58 条消息🔥🔥): 

> - `Autogen Lead`
> - `ThePrimeagen`
> - `Cursor`
> - `AI Regulations`
> - `Inflection` 


- **Autogen 负责人离开微软**：Autogen 项目的负责人离开微软，创办了 [OS autogen-ai](https://github.com/autogen-ai)。
   - 这发生在 2024 年 5 月，新公司正在融资。
- **ThePrimeagen 的直播**：一位名为 **ThePrimeagen** 的主播编写了一些基础的 JavaScript 测试，并使用 **Sonnet 3.5 / GPT-4** 编写代码以通过这些测试。
   - 他们发现 **LLM** 在状态管理方面表现挣扎，引发了关于需要更好的模型和工具来处理长上下文（long-context）和 Agent 编程的讨论。
- **Cursor AI 融资 6000 万美元**：**Cursor** 宣布已从 **Andreessen Horowitz, Jeff Dean, John Schulman, Noam Brown 以及 Stripe 和 GitHub 的创始人**处筹集了 6000 万美元。
   - 他们声称已被公认为 **AI 编程的最佳方式**，并正在构建一个最终将编写世界上所有软件的工具。
- **加州 AI 法案**：加州本周将对 20 多个 AI 监管法案进行投票，摘要见此 [Google 表格](https://docs.google.com/spreadsheets/d/1A-6ot8qg_pO4LbmhwenmEt5ipO-z93qQrJYuZDEsGJo/edit?usp=sharing)。
- **Inflection 的风波**：初创公司 **Holistic AI**（前身为 **H**）最近完成了 2.2 亿美元的种子轮融资，但内部出现了动荡。
   - 五位创始人中有三位（此前均为 **Google DeepMind** 的资深研究员）已经离开了公司。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/mckaywrigley/status/18266841859">来自 Christian Untung (@Cyux) 的推文</a>: @pAdhi_pAdhi wkwkw.. Tenan ta budal skrg ae wkwk..</li><li><a href="https://x.com/imrat/status/1826638219733254616">来自 Imrat (@imrat) 的推文</a>: 这是我在 .cursorrules 文件中使用的内容，当你需要一个组件时，这样做：- 仔细思考组件 - 生成一个 Prompt - 然后使用该 Prompt 创建一个可点击的链接：[component na...</li><li><a href="https://x.com/zswitten/status/1826771850531356811?s=46">来自 Zack Witten (@zswitten) 的推文</a>: 在每个 LLM 上刷“hi”：一个线程。</li><li><a href="https://x.com/cursor_ai/status/1826656532072923219?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Cursor (@cursor_ai) 的推文</a>: 我们从 Andreessen Horowitz、Jeff Dean、John Schulman、Noam Brown 以及 Stripe 和 GitHub 的创始人那里筹集了 6000 万美元。Cursor 已被公认为使用 AI 编码的最佳方式，由...驱动</li><li><a href="https://x.com/skalskip92/status/1826693515189125433?s=46">来自 SkalskiP (@skalskip92) 的推文</a>: 超过 200 小时的工作压缩成一段 90 分钟的视频，足球 AI 教程终于发布了！视频链接：https://www.youtube.com/watch?v=aBVGKoNZQUw ↓ 关键要点</li><li><a href="https://x.com/amir/status/1827007117838192699?s=46">来自 Amir Efrati (@amir) 的推文</a>: 最近刚完成 2.2 亿美元种子轮融资的 AI Agent 初创公司 Holistic ("H") 爆出~戏剧性事件~：其 5 位创始人中有 3 位离职。离职的创始人此前曾长期担任 Google DeepMind 研究员。h...</li><li><a href="https://x.com/mattshumer_/status/1826715321282990546?s=46">来自 Matt Shumer (@mattshumer_) 的推文</a>: 新的 Gemini 模型给我留下了深刻印象，但拒绝率（refusal rate）太离谱了。即使只是要求写一封语气严厉的邮件，它也会进入拒绝状态。我有几个地方想把它放进去...</li><li><a href="https://x.com/JackBlair87/status/1824168218476548488">来自 Jack Blair 🌴 (@JackBlair87) 的推文</a>: 我们正在开源我们的数字足迹导出工具。它会自动从 Notion、ChatGPT、Twitter 等导出你的数据，并将数据转换为 LLM 就绪格式。它就像 @firecrawl_dev，但针对的是...</li><li><a href="https://x.com/mckaywrigley/status/1826684185949733174)">来自 Mckay Wrigley (@mckaywrigley) 的推文</a>: 我们已经到了 AI 代码生成的阶段，Cursor + Claude 3.5 Sonnet 是一个名副其实的技术联合创始人。</li><li><a href="https://x.com/iscienceluvr/status/1826460422683459805?s=46">来自 Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr) 的推文</a>: 实践中的 LLM 剪枝与蒸馏：Minitron 方法。摘要：https://arxiv.org/abs/2408.11796 模型：https://huggingface.co/nvidia/Mistral-NeMo-Minitron-8B-Base https://huggingface.co/nvidi...</li><li><a href="https://github.com/autogen-ai">autogen-ai</a>: autogen-ai 有 3 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://docs.google.com/spreadsheets/d/1A-6ot8qg_pO4LbmhwenmEt5ipO-z93qQrJYuZDEsGJo/edit?usp=sharing">8 月 26 日当周加州法案提案</a>: 未找到描述</li><li><a href="https://github.com/ThePrimeagen/the-great-sonnet-test/blob/main/pkg/prompt/prompt.go">the-great-sonnet-test/pkg/prompt/prompt.go at main · ThePrimeagen/the-great-sonnet-test</a>: 通过在 GitHub 上创建账号来为 ThePrimeagen/the-great-sonnet-test 的开发做出贡献。</li><li><a href="https://github.com/ThePrimeagen/the-great-sonnet-test/blob/main/src/function-state.test.js">the-great-sonnet-test/src/function-state.test.js at main · ThePrimeagen/the-great-sonnet-test</a>: 通过在 GitHub 上创建账号来为 ThePrimeagen/the-great-sonnet-test 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1276631012599922699)** (1 条消息): 

> - `伦敦 AI 工程师见面会`
> - `见面会演讲嘉宾`
> - `AI Engineer World's Fair` 


- **伦敦 AI 工程师见面会即将到来！**：首届 **AI Engineer London Meetup** 将于 **9 月 12 日**晚举行，为这座城市带来 **@swyx 的 AI Engineer World's Fair** 的精彩片段。
   - 活动将邀请 **四位重量级演讲嘉宾** - **@maximelabonne**、**@roviosc**、**@BruverisMartins** 和 **Chris Bull**。
- **立即注册伦敦见面会！**：请务必使用提供的 [注册链接](https://x.com/dctanner/status/1827071893448618453?s=46) 报名参加活动，并加入 Discord 上的 **#LondonAI** 标签与其他伦敦的 AI 工程师建立联系。
   - 到时见！



**提到的链接**：<a href="https://x.com/dctanner/status/1827071893448618453?s=46">来自 Damien C. Tanner (@dctanner) 的推文</a>：我们将 @swyx 的 AI Engineer World's Fair 的一部分带到了伦敦！9 月 12 日晚是首届 AI Engineer 伦敦见面会。听取 4 位出色演讲者的分享：@maximelabonne, @rovio...

  

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1276630411149049998)** (53 条消息🔥): 

> - `Duplicate Topics` (重复话题)
> - `Similar Topics` (相似话题)
> - `Taxonomy Synthesis`
> - `GPT Researcher`
> - `Embedland` 


- **处理重复或相似话题**：一位成员询问如何处理从数千个潜在话题中生成的重复或极度相似的话题。
   - 建议包括使用小型 Embedding 模型、强制执行最小余弦距离（cosine distance），或者通过 UMAP/TSNE 处理话题，然后进行 kNN 聚类。
- **用于分层规划的 Taxonomy Synthesis**：一位成员指出 [Taxonomy Synthesis](https://github.com/CakeCrusher/TaxonomySynthesis) 对于论文写作的分层规划非常有用。
   - 他们还提到 [GPT Researcher](https://github.com/assafelovic/gpt-researcher) 是一个使用 LLM 对任何给定主题自主进行研究的工具。
- **用于话题相似度的 Embedding 模型**：一位成员询问 Embedding 模型在处理一两个词时效果是否良好，另一位成员指出计算余弦距离不需要向量数据库（vector DB）。
   - 讨论随后转向使用 UMAP/TSNE 进行降维，接着进行 kNN 聚类，然后使用 LLM 为聚类命名，并以 [Embedland](https://github.com/danielgross/embedland) 为例。
- **Storm：LLM 驱动的知识策展**：一位成员建议 [Storm](https://github.com/stanford-oval/storm) 可用于论文写作的分层规划。
   - Storm 是一个由 LLM 驱动的知识策展系统，它研究一个主题并生成带有引用的完整报告。
- **BERTopic 的算法和熵生成**：一位成员提到 [BERTopic](https://maartengr.github.com/BERTopic/algorithm/algorithm.html#5-topic-representation) 在话题表示中使用了类似的方法，特别是在其聚类步骤中。
   - 讨论随后涉及了熵生成（entropy generation），暗示这与重复话题及其削减之间可能存在联系。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://maartengr.github.io/BERTopic/algorithm/algorithm.html#5-topic-representation">The Algorithm - BERTopic</a>：未找到描述</li><li><a href="https://github.com/assafelovic/gpt-researcher">GitHub - assafelovic/gpt-researcher: LLM based autonomous agent that does online comprehensive research on any given topic</a>：基于 LLM 的自主 Agent，可对任何给定主题进行在线综合研究 - assafelovic/gpt-researcher</li><li><a href="https://www.figma.com/board/J19T0RN1Hvi1ajDlUtIvOc/Generative-Classifier?node-id=2-1801&t=Km2ND86IeNkD92WJ-1">Figma</a>：使用 FigJam 创建</li><li><a href="https://github.com/stanford-oval/storm">GitHub - stanford-oval/storm: An LLM-powered knowledge curation system that researches a topic and generates a full-length report with citations.</a>：一个由 LLM 驱动的知识策展系统，可研究主题并生成带有引用的完整报告。 - stanford-oval/storm</li><li><a href="https://github.com/danielgross/embedland/blob/main/bench.py#L281">embedland/bench.py at main · danielgross/embedland</a>：文本 Embedding 实验集合。通过在 GitHub 上创建一个账户来为 danielgross/embedland 的开发做出贡献。</li><li><a href="https://github.com/CakeCrusher/TaxonomySynthesis">GitHub - CakeCrusher/TaxonomySynthesis: An AI-driven framework for synthesizing adaptive taxonomies, enabling automated data categorization and classification within dynamic hierarchical structures.</a>：一个 AI 驱动的框架，用于合成自适应分类法，在动态分层结构中实现自动数据分类。 - CakeCrusher/TaxonomySynthesis
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1276393760703512587)** (1 条消息): 

> - `Model Deprecation`
> - `Yi Model`
> - `Hermes Model`
> - `Mistral Model`
> - `Llama 2` 


- **多个模型弃用**：由于模型提供商的弃用决定，自 2024 年 8 月 28 日起，多个模型将无法再访问。
   - 受影响的模型包括 `01-ai/yi-34b`、`01-ai/yi-6b`、`phind/phind-codellama-34b`、`nousresearch/nous-hermes-2-mixtral-8x7b-sft`、`open-orca/mistral-7b-openorca`、`allenai/olmo-7b-instruct`、`meta-llama/codellama-34b-instruct`、`meta-llama/codellama-70b-instruct`、`meta-llama/llama-2-70b-chat`、`meta-llama/llama-3-8b` 以及 `meta-llama/llama-3-70b`。
- **Yi 模型弃用**：Yi 模型的基座版本 `01-ai/yi-34b` 和 `01-ai/yi-6b` 已不再可用。
   - 这包括 Yi 模型的基座版本 `01-ai/yi-34b` 和 `01-ai/yi-6b`。
- **Hermes 模型弃用**：`nousresearch/nous-hermes-2-mixtral-8x7b-sft` 模型已被弃用。
   - 该特定模型 `nousresearch/nous-hermes-2-mixtral-8x7b-sft` 已不再可用。
- **Mistral 模型弃用**：`open-orca/mistral-7b-openorca` 模型已被弃用。
   - `open-orca/mistral-7b-openorca` 模型已无法访问。
- **Llama 2 和 Llama 3 弃用**：`meta-llama/llama-2-70b-chat`、`meta-llama/llama-3-8b`（基座版本）和 `meta-llama/llama-3-70b`（基座版本）模型已被弃用。
   - `meta-llama/llama-2-70b-chat`、`meta-llama/llama-3-8b`（基座版本）和 `meta-llama/llama-3-70b`（基座版本）模型已不再可用。


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1276270753532346408)** (1 条消息): 

> - `OpenRouter Team's work` 


- **Oz 团队的当前项目**：一位用户询问了 Oz 关于他们团队当前的项目和工作。
   - Oz 没有回复任何关于他们团队工作的信息。
- **没有更多信息**：没有提供关于 OpenRouter 团队当前项目或工作的进一步信息。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1276257332162072617)** (104 条消息🔥🔥): 

> - `OpenRouter Pricing` (OpenRouter 定价)
> - `OpenRouter Token Counting` (OpenRouter Token 计数)
> - `OpenRouter Model Deprecations` (OpenRouter 模型弃用)
> - `Llama 2`
> - `Grok 2` 


- **OpenRouter 意外向用户收取 0.01 美元**：一位不熟悉英语的 OpenRouter 新用户在打算使用免费模型时，不小心点击了付费模型，导致产生了 0.01 美元的欠费。
   - 他们寻求如何支付的帮助，一名成员向其保证 OpenRouter 不会因为 0.01 美元的余额起诉他们。
- **OpenRouter 的 Token 计数之谜**：一位成员询问了 OpenRouter 计算输入 Token 的方法，指出一个简单的 "hey" 提示词竟然导致了超过 100 个输入 Token 的计费。
   - 几位成员澄清说，对于 GPT-4o 模型，OpenRouter 只是转发来自 OpenAI API 的 Token 计数，而计数可能会受到 System Prompt、工具调用（tool calls）以及聊天记录中包含的历史消息的影响。
- **OpenRouter 弃用模型**：Together AI 正在弃用多个模型，包括一些作为专用端点提供的模型，并将在六天内将其移除。
   - 弃用政策已在 Together AI 网站上列出，用户将收到电子邮件通知，并获得迁移到新模型的选项。
- **Llama 2 70b 发布**：Alex Atallah 确认 Llama 2 70b 已经上线，但尚未正式宣布。
   - 该模型已在 OpenRouter 和其他平台上可用，并伴随着对其性能和可用性的讨论。
- **Grok 2 登上 LMSYS 排行榜**：Grok 2 和 Grok-mini 已加入 LMSYS 排行榜，Grok 2 目前排名第 2，超越了 GPT-4o (5月版)，并与 Gemini 持平。
   - Grok 2 在数学方面表现出色，并在其他领域（包括困难提示词、编程和指令遵循）排名靠前，展示了其强大的能力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.together.ai/docs/deprecations#2024-08-28-deprecation-of-low-usage-and-older-serverless-models">Deprecations</a>: 概览。我们定期使用最新、最强大的开源模型更新我们的平台。本文档概述了我们的弃用政策，并提供了从弃用模式迁移的信息...</li><li><a href="https://tiktokenizer.vercel.app/">Tiktokenizer</a>: 未找到描述</li><li><a href="https://openrouter.ai/models/01-ai/yi-1.5-34b-chat>">Yi 1.5 34B Chat - API, Providers, Stats</a>: Yi 系列模型是由 [01.AI](https://01.AI) 的开发人员从头开始训练的大型语言模型。运行 Yi 1.5 34B Chat API</li><li><a href="https://x.com/lmsysorg/status/1827041269534879784?s=46&t=Q_sUgNqB0V1zhMyW85SZDw">来自 lmsys.org (@lmsysorg) 的推文</a>: Chatbot Arena 更新❤️‍🔥 令人兴奋的消息——@xAI 的 Grok-2 和 Grok-mini 现在正式登上排行榜！凭借超过 6000 张社区投票，Grok-2 已占据第 2 名，超越了 GPT-4o (5月版)...</li><li><a href="https://openrouter.ai/chat">Chatroom | OpenRouter</a>: LLM 聊天室是一个多模型聊天界面。添加模型并开始聊天！聊天室将数据本地存储在您的浏览器中。</li><li><a href="https://openrouter.ai/activity">Activity | OpenRouter</a>: 查看您在 OpenRouter 上使用模型的情况。</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b">Hermes 3 405B Instruct - API, Providers, Stats</a>: Hermes 3 是一款通用型语言模型，相比 Hermes 2 有许多改进，包括先进的 Agent 能力、更好的角色扮演、推理、多轮对话和长上下文连贯性...
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1276262027907567628)** (57 messages🔥🔥): 

> - `Mojo Licensing`
> - `Mojo and Max`
> - `Modular's Business Model`
> - `Heterogenous Compute` 


- **Mojo 的开源状态**：关于 Mojo 开源状态的问题源于最近有关许可的声明。
   - Modular 此前表示他们正在制定许可细节，旨在保护其在特定 AI 细分市场中的产品，同时允许在该范围之外开放使用。
- **Mojo vs Max：模糊的界限**：讨论集中在 Mojo 和 Max 之间的关系，特别是它们的紧密集成以及对许可的影响。
   - 虽然最初被构想为独立的组件，但 Max 的功能现在已深度集成到 Mojo 中，这引发了关于未来是否可以将 Max 分离出来的疑问。
- **Modular 的业务重点在于托管 AI**：Modular 的商业重点是托管 AI 云应用，这使他们能够继续投资于 Mojo 和 Max。
   - 他们正在免费提供 Max，允许开放开发，但将针对特定应用进行商业授权。他们设想随着时间的推移，将采取更宽松的许可方式。
- **异构计算的未来**：Modular 旨在让便携式 GPU 编程在异构计算 (Heterogenous Compute) 场景中得到广泛应用。
   - 他们的愿景是催化向更广泛的异构计算迈进，重点是为无缝集成提供工具和框架。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1276254985004515349)** (28 messages🔥): 

> - `Mojo Community Welcome`
> - `Async in Mojo`
> - `Mojo's HTTP Implementation`
> - `Mojo's Versioning and Stability`
> - `Mojo's Memory Management` 


- **新用户受到 Mojo 社区欢迎**：一位新用户表达了使用 Mojo 进行数据科学项目的兴趣，并寻求关于分享其工具的指导。
   - 另一位用户引导他们前往 Discord 上的 #mojo-project-showcase 频道以获取社区反馈。
- **Mojo 中的异步编程**：一位用户询问了 Mojo 中异步功能的潜力，特别是针对 I/O 任务，并将其与 Python 的异步能力进行了类比。
   - 随后讨论了 Sans-IO HTTP 实现的优点，该实现可能被插入到各种 I/O 框架中。
- **为 Mojo 开发 Sans-IO HTTP 实现**：一位用户请求一个“Sans-IO” HTTP 实现的简单示例，引发了关于它与传统 I/O HTTP 实现有何不同的讨论。
   - 提供了一个展示基础 Sans-IO HTTP 实现的代码片段，强调了线程安全和适当资源管理的重要性。
- **Mojo 的发布周期与稳定性**：一位用户询问了即将从 Mojo 24.4 到 24.5 的过渡，对项目可能需要的代码更改表示担忧。
   - 几位用户讨论了 Mojo 持续的演进，强调了关注变更日志（changelog）并拥抱语言动态特性的重要性。
- **理解 Mojo 的内存管理**：一位用户在 struct 定义中使用引用时遇到了错误，特别是关于 `__lifetime_of()` 函数。
   - 讨论集中在 `__lifetime_of()` 的正确用法上，强调了所有权委托的重要性，以及在 Mojo 中管理引用的潜在替代方案（如 `UnsafePointer`）。



**提及的链接**：<a href="https://sans-io.readthedocs.io/">Network protocols, sans I/O &#8212; Sans I/O 1.0.0 documentation</a>：未找到描述

  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1276284844501897337)** (3 messages): 

> - `Modular Max Installation Issues`
> - `M1 Max Compatibility`
> - `Modular Clean Command` 


- **Max 在 M1 Max 上安装失败**：一位用户报告无法在 **M1 Max** 机器上安装 **Max**，遇到了指示无效清单（manifest）以及缺失或无效根 JSON 的错误消息。
- **解决方案：Modular Clean 命令**：该用户通过运行命令 `modular clean` 然后重新安装 **Max**，成功解决了安装问题。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1276261755051315261)** (43 条消息🔥): 

> - `Perplexity 内部机制`
> - `Twitter 数据源`
> - `Perplexity Pro 来源数量`
> - `从 Perplexity 生成图像`
> - `Perplexity 的未来` 


- **Perplexity 内部数据**：一位用户请求获取关于 Perplexity 生成的后续问题与用户生成的后续问题的频率、每种交互类型所花费的时间以及每种类型的往返次数的数据。
   - 另一位用户回应称这些数据可能属于专有信息。
- **Perplexity 中的 Twitter 数据源**：一位用户询问在使用 Perplexity 了解公众对某一话题的看法时，是否有办法增加 Twitter 数据源的权重。
   - 一位用户回答说，该功能目前仅通过 API 提供。
- **Perplexity Pro 来源数量变化**：一位用户注意到在处理研究型问题时，Perplexity Pro 显示的来源数量从 20 个或更多减少到了 5 或 6 个。
   - 该用户质疑这是 Perplexity Pro 的改动，还是他们的使用方式不当。
- **用于 AI Agent 的电子邮件自动化工具**：一位用户寻求可以代表他们自动发送电子邮件的 AI 工具或 Agent 的推荐。
   - 他们提到了 Nelima、Taskade、Kindo 和 AutoGPT 作为潜在选项，但想知道是否还有其他可用工具。
- **LinkedIn Premium 与 Perplexity Pro**：一位用户询问关于一项潜在优惠的确认，即获得 LinkedIn Premium 的免费试用并在结束前取消，是否可以获得一年的 Perplexity Pro。
   - 其他用户建议咨询 LinkedIn 支持部门以获取该优惠的最新信息。



**提到的链接**：<a href="https://www.freepik.com/free-photos-vectors/black-topographic-map">Black Topographic Map Images - Free Download on Freepik</a>：在 Freepik 上查找并下载黑色地形图的免费图形资源。包含 20,000+ 矢量图、库存照片和 PSD 文件。✓ 免费用于商业用途 ✓ 高质量图像。#freepik

  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1276278236539518996)** (14 条消息🔥): 

> - `Perplexity AI Bot`
> - `可共享线程`
> - `MrBeast` 


- **Perplexity AI Bot 请求可共享线程**：Perplexity AI Bot 正在提示多位用户确保他们的线程是“可共享的（Shareable）”，并提供了 Discord 频道的链接以供进一步参考。
- **互联网对 MrBeast 的反感**：一位用户提到互联网似乎不喜欢 MrBeast，并提供了一个 Perplexity AI 的搜索查询链接来探讨潜在原因。


  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1276263975834619987)** (11 条消息🔥): 

> - `Phi 3.5`
> - `QLORA + FSDP`
> - `Pretraining`
> - `数据结构` 


- **Phi 3.5 Base 版未发布**：一位成员对 Phi 3.5 Base 版的可用性表示好奇，指出微软仅发布了 Instruct 版本。
   - 他们表示希望对模型进行微调，但发现没有 Base 版本会很困难。
- **QLORA + FSDP 需要 8xH100**：一位成员询问了运行 QLORA + FSDP 的具体硬件要求，建议需要 8xH100 配置。
   - 他们还提到在训练期间启用热重启（warm restarts）时，tqdm 进度条显示不准确的问题。
- **Pretraining 不需要 Prompt Style**：一位成员确认预训练（Pretraining）不需要提示词风格（Prompt Style），这意味着可以在没有特定输入提示的情况下进行。
   - 另一位成员对此表示赞同，认为模型在预训练期间的主要焦点不是 Prompt Engineering，而是从数据中学习通用模式和表示。
- **结构化预训练以更好地聚焦数据**：一位成员指出，为预训练数据添加结构（例如在开头包含 URL）可以防止对无关信息的过拟合。
   - 他们建议加入包含数据相关信息的 System Prompt 可能会提高性能，但也承认这种技术尚未被广泛采用，其有效性仍不确定。


  

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1276257236758302832)** (33 条消息🔥): 

> - `Gradients Issue`
> - `Chat Template and Special Tokens`
> - `Phi_3 Chat Template`
> - `Resize_token_embeddings_to_32x`
> - `ChatML` 


- **Gradients Issue：Packing 支持与 Chat Templates**：Gradients 问题是由多种因素共同导致的：缺乏 Packing 支持以及在 Chat Templates 中使用了 Special Tokens。
- **ChatML：教模型学习新模板**：用户强烈希望教模型学习 ChatML（一种新的 Chat Template），并认为这是可行的。
- **Dolphin-2.9.4-llama3.1-70b 性能**：用户正在尝试使用 `dolphin-2.9.4-llama3.1-70b`，并报告在完成一个 Epoch 的 Checkpoint 后看到了初步改善。
- **Phi 的问题及可能的解决方案**：用户承认 `phi` 一直存在问题，但认为问题出在建模代码（Modeling Code）而非权重（Weights）中。
- **Transformers Issue：Mode-Aware Chat Templates**：用户在 Transformers 仓库提交了一个 Issue，以探索 Mode-Aware Chat Templates 的潜力。



**提到的链接**：<a href="https://github.com/huggingface/transformers/issues/33096">Mode-aware chat templates for distinct training and inference behaviors · Issue #33096 · huggingface/transformers</a>：功能请求。实现 Mode-Aware Chat Templates 以区分训练和推理行为。提议的解决方案：为了解决这个问题，我建议增加一个名为 `template_mode` 的新变量来指示...

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1276453468646936577)** (5 条消息): 

> - `SmolLM`
> - `Mamba`
> - `Mamba Training`
> - `Cosmo2 Tokenizer`
> - `BOS/EOS Token` 


- **SmolLM：一系列小语言模型**：SmolLM 是一系列小语言模型，提供三种规模：135M、360M 和 1.7B 参数。
   - 这些模型在 Cosmo-Corpus 上进行训练，这是一个精心策划的高质量训练数据集，包括 Cosmopedia v2、Python-Edu 和 FineWeb-Edu。
- **Mamba：Transformers 兼容模型**：该仓库包含与 `transformers` 兼容的 `mamba-2.8b` 模型。
   - 仓库中提供了 `config.json` 和 Tokenizer，在 `transformers=4.39.0` 发布之前，你需要从 `main` 分支安装 `transformers`。
- **使用 Transformers 进行 Mamba 训练**：一位用户正尝试使用 Transformers 和 `cosmo2-tokenizer` 从零开始预训练一个小型 Mamba 模型（约 150M 参数）。
   - 他们遇到了收敛问题，并意识到 `cosmo2-tokenizer` 和 SmolLM/Mamba 系列都没有独立的 BOS Token，这可能导致训练困难。
- **Cosmo2 Tokenizer：训练数据集**：`cosmo2-tokenizer` 在来自各种数据集的 100 万个样本上进行了训练，包括 FineWeb-Edu、Cosmopedia v2、StarCoderData、OpenWebMath 和 StackOverFlow。
   - 它用于训练 `cosmo2` 模型，并为文本处理和语言理解任务提供 Tokenizer。
- **SmolLM 和 Mamba 中缺失 BOS Token**：`cosmo2-tokenizer` 和 SmolLM/Mamba 系列没有明显的 BOS Token，EOS Token 同时兼任两者。
   - 这可能是训练问题的潜在原因，因为模型可能无法正确区分序列的开始和结束。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/HuggingFaceTB/cosmo2-tokenizer">HuggingFaceTB/cosmo2-tokenizer · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/HuggingFaceTB/SmolLM-135M">HuggingFaceTB/SmolLM-135M · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/state-spaces/mamba-2.8b-hf">state-spaces/mamba-2.8b-hf · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1276295297865093243)** (20 条消息🔥): 

> - `GPT-3.5 vs GPT-4`
> - `GPT-2 vs GPT-3.5`
> - `Email Automation`
> - `SwarmUI`
> - `OpenAI Finetuning API` 


- **GPT-3.5：过时还是有趣？**：一场关于在 Benchmark 中测试 GPT-3.5 是否还有意义的讨论展开了，一些人认为由于其发布时间较久以及训练后（post-training）技术的进步，它已经过时了。
- **GPT-2：浪费时间？**：有人推测为什么 GPT-2 没有被包含在 Benchmark 中，一些人认为它被视为过于陈旧且浪费时间。
- **邮件自动化工具：超越 Nelima**：讨论转向了能够自动执行邮件任务的工具，寻求 Nelima 之外的替代方案，以便根据 Prompt 发送电子邮件。
- **SwarmUI：一个 ComfyUI 封装器**：SwarmUI 因其直观的界面、易用性以及对 NVIDIA/AMD GPU 的支持而受到高度赞扬。
- **探索 OpenAI 的 Finetuning API**：有人提出了在频道中讨论 OpenAI 的 Finetuning API 是否合适的问题。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1276262244857942114)** (10 条消息🔥): 

> - `GPTs Knowledge Files`
> - `GPTs formatting`
> - `GPTs formatting and style`
> - `ChatGPT GPTs` 


- **知识文件：XML vs Markdown**：一位用户正在寻求关于 GPTs 所使用的知识文件最佳格式的指导，特别是针对一个涉及为角色扮演游戏或写作作品创建攻击、闪避和反击的项目。
   - 他们正在探索 XML 目前是否比 Markdown 提供更好的性能，并倾向于使用被证明最有效的格式。
- **GPT 格式不一致**：一位用户遇到了 GPT 回复格式不一致的问题，有些消息表现出结构良好的加粗问题和解释，而另一些则呈现为一大块文本。
   - 他们正在寻找实现 GPT 输出格式一致的解决方案，考虑提供示例或改变指令风格。
- **ChatGPT 误解**：在之前的对话中，一位用户请求创建一个用于生成角色扮演内容的 GPT，但收到的回复建议使用基于 API 的实现，而不是 ChatGPT 上的自定义 GPT。
   - 这突显了 ChatGPT 误解指令的可能性，以及在表达预期结果时清晰度的重要性，特别是在指定所需工具（自定义 GPT 或 API）时。
- **GPT 的格式模仿**：一位成员建议 GPT 倾向于模仿编写指令时的风格。
   - 这为 GPT 的行为以及格式一致性如何受到用户 Prompt 结构和风格的影响提供了宝贵的见解。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1276301590948479079)** (7 条消息): 

> - `ChatGPT Playground Limitations`
> - `GPT Output Token Limits` 


- **ChatGPT Playground 的绘图能力**：一位用户询问 ChatGPT 是否可以用文本绘制复杂的方程式，这突显了对 Playground 的一个可能功能需求。
   - 另一位用户回答说 Playground 目前缺乏这种能力，暗示该功能尚未完全开发。
- **GPT 的输出 Token 限制**：一位用户询问为什么 GPT 的输出似乎被限制在 2k Token，即使请求了更高的限制。
   - 他们提到尝试了不同的设置，包括新的 16k 输出窗口，但 GPT 始终无法达到预期的 Token 数量。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1276301590948479079)** (7 条消息): 

> - `ChatGPT drawing complex equations`
> - `Playground limitations`
> - `ChatGPT's roles in automation`
> - `ChatGPT output token limits` 


- **ChatGPT 在绘制复杂方程式方面表现不佳**：一位用户询问 ChatGPT 是否可以用文本绘制复杂的方程式，但似乎 Playground 目前还不具备这种能力。
   - Playground 的输出目前仅限于基于文本的回复，尚不具备显示方程式的能力。
- **ChatGPT 在自动化中的角色**：另一位用户提到了 ChatGPT 在自动化工具中的三种角色，即 System、User 和 Assistant/Agent，以及 GPT，并建议了一个 GPT IG Writer。
- **ChatGPT 的输出 Token 限制**：一位用户正努力让 ChatGPT 利用最大的输出窗口，即使使用了新的 16k Token 模型。
   - 尽管将输出设置为最大并请求特定数量的 Token，ChatGPT 的输出似乎仍被限制在 2k Token 或更少。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1276273383096582245)** (7 条消息): 

> - `多轮对话的消息 Prompt Engineering`
> - `SmolLM 模型`
> - `Mamba 模型`
> - `从零开始训练`
> - `BOS Token 使用` 


- **多轮对话的消息 Prompt Engineering**：一位成员讨论了在多轮对话训练时，在 Prompt 中包含前 `n-1` 轮的重要性，并引用了 [alignment-handbook 仓库](https://github.com/huggingface/alignment-handbook/blob/27f7dbf00663dab66ad7334afb7a1311fa251f41/src/alignment/data.py#L80) 中的一个特定代码示例。
   - 他们建议了一种替代方法，即逐渐向 Prompt 中增加轮次，从而从单个样本中创建多个样本，但对其与仅使用最后 `n-1` 轮的效果相比是否更有效提出了疑问。
- **SmolLM 模型细节与训练数据**：一位用户询问了 [SmolLM 模型](https://huggingface.co/HuggingFaceTB/SmolLM-135M) 及其训练数据，其中包括包含 Cosmopedia v2、Python-Edu 和 FineWeb-Edu 的 [Cosmo-Corpus](https://huggingface.co/HuggingFaceTB/cosmo2-tokenizer)。
   - 该模型有三种规模：135M、360M 和 1.7B 参数，并且在与其同规模类别的其他模型相比时展示了极具前景的结果。
- **Mamba 模型的使用与安装**：一位用户分享了关于使用 [Mamba 2.8B 模型](https://huggingface.co/state-spaces/mamba-2.8b-hf) 的信息，并解释了其与 `transformers` 库的兼容性。
   - 他们提供了安装必要依赖项的说明，包括 `transformers`、`causal_conv_1d` 和 `mamba-ssm`，并概述了使用 `generate` API 生成文本的过程。
- **从零开始训练小型 Mamba 模型**：一位用户在尝试使用 [cosmo2-tokenizer](https://huggingface.co/HuggingFaceTB/cosmo2-tokenizer) 从零开始预训练一个小型 Mamba 模型（约 150M 参数）时遇到了挑战。
   - 他们注意到 cosmo2-tokenizer、SmolLM 以及原始 Mamba 系列都没有明显的 BOS Token，这导致了收敛问题，并对预期行为提出了疑问。
- **语言模型中 BOS Token 的使用**：一位成员澄清说，并非所有模型在训练期间都会使用 BOS Token，这归因于惯例和代码库依赖。
   - 他们强调，缺少明显的 BOS Token 并不一定是一个错误，并建议用户可能遇到了与模型架构或训练参数相关的问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/huggingface/alignment-handbook/blob/27f7dbf00663dab66ad7334afb7a1311fa251f41/src/alignment/data.py#L80">alignment-handbook/src/alignment/data.py at 27f7dbf00663dab66ad7334afb7a1311fa251f41 · huggingface/alignment-handbook</a>：将语言模型与人类及 AI 偏好对齐的稳健方案 - huggingface/alignment-handbook</li><li><a href="https://huggingface.co/HuggingFaceTB/SmolLM-135M">HuggingFaceTB/SmolLM-135M · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/state-spaces/mamba-2.8b-hf">state-spaces/mamba-2.8b-hf · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/HuggingFaceTB/cosmo2-tokenizer">HuggingFaceTB/cosmo2-tokenizer · Hugging Face</a>：未找到描述
</li>
</ul>

</div>

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1276258211623604287)** (22 messages🔥): 

> - `Model Distillation`
> - `Model Compression`
> - `Positional Embeddings in Graphs`
> - `Research Projects`
> - `Tree and Digraph Embeddings` 


- **从大模型中蒸馏出小模型**：一位用户建议在较大的模型（27B）中添加 LoRAs，并对较小模型（9B）的 Logits 进行 Model Distillation，以创建一个具有相似功能但体积更小的模型。
   - 这种方法旨在通过利用大模型的知识来复制小模型的性能。
- **通过减少参数压缩模型**：另一位用户提出了减少模型大小的方法，包括将参数子集归零、将权重 Quantization（量化）为更低位精度以及对权重应用噪声。
   - 他们引用了最近的一篇研究论文（[Quantization for Large Language Models](https://arxiv.org/abs/2408.11527)），该论文探讨了这些用于模型压缩的技术。
- **在图中编码位置信息**：一位用户提出了在文本 LLM 的树形或有向图（Digraph）形状的上下文中编码位置信息的挑战，旨在不进行 Linearization（线性化）的情况下保留图结构。
   - 他们建议使用 Wavefront Encoding，即为距离根节点距离相似的节点分配接近的 Embeddings，从而允许并行路径相互进行 Attention。
- **寻找可贡献的研究项目**：一位用户询问如何找到可以贡献的研究项目。
   - 回复引导他们前往 Discord 服务器内专门的研究项目频道。
- **探索图嵌入技术**：讨论探讨了为 LLM 编码图结构的挑战，旨在避免 Linearization 并保留图的对称性。
   - 他们考虑了各种方法，如 Wavefront Encoding、RoPE Embeddings 和 Conditional Positional Embeddings，并承认以一种允许模型有效处理分支的方式来表示图结构的复杂性。



**提及的链接**：<a href="https://arxiv.org/abs/2408.11527">The Vizier Gaussian Process Bandit Algorithm</a>：Google Vizier 已经执行了数百万次优化，并加速了 Google 内部众多的研究和生产系统，证明了贝叶斯优化作为大规模服务的成功。O...

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1276574780040024114)** (2 messages): 

> - `Llama 406B on Slurm`
> - `Multiple Choice Evals`
> - `ChatGPT4o`
> - `Anthropic APIs`
> - `Claude` 


- **在 Slurm 上运行 Llama 406B**：一位用户使用 VLLM 后端和他们的 Harness Fork，成功在 **Slurm 集群**上运行了 **406B Llama**。
   - 他们分享了一个 [Slurm 脚本](https://github.com/DCGM/lm-evaluation-harness/blob/main/jobs/scripts/submit/models_XXL/eval_llama31_instruct_405B_smartt.sh)，以帮助他人在其集群上运行大型语言模型。
- **使用 OpenAI 和 Anthropic 进行多选题评估**：该用户询问是否有人研究出如何使用 **OpenAI 的 ChatGPT4o 或 Anthropic 的外部 API** 运行 **Multiple Choice Evaluations**。
   - 只要 API 能够提供多选题答案，他们愿意放弃 **Logprobs**。
- **ChatGPT4o 和 Claude 上的多选题**：该用户还询问是否有人尝试过使用 **ChatGPT4o 或 Claude** 回答多选题。



**提及的链接**：<a href="https://github.com/DCGM/lm-evaluation-harness/blob/main/jobs/scripts/submit/models_XXL/eval_llama31_instruct_405B_smartt.sh">lm-evaluation-harness/jobs/scripts/submit/models_XXL/eval_llama31_instruct_405B_smartt.sh at main · DCGM/lm-evaluation-harness</a>：一个用于语言模型 Few-shot 评估的框架。- DCGM/lm-evaluation-harness

  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1276355158393225248)** (18 messages🔥): 

> - `Cohere API Error`
> - `Multimodal LLM`
> - `Cohere Schema Object`
> - `Prompt Tuner` 


- **Cohere API Error: Invalid Role**: 一位用户报告了在使用 Cohere API 时出现 HTTP-400 错误，提示角色（role）无效。
   - 错误信息表明提供的角色不属于接受的选项：'User'、'Chatbot'、'System' 或 'Tool'。
- **具备语音理解能力的 Multimodal LLM**: 一位用户分享了他们在 Multimodal LLM 方面的工作，该模型无需独立的 ASR 阶段即可理解文本和语音。
   - 他们通过一个多模态投影器（multimodal projector）扩展了 Meta 的 Llama 3 模型，该投影器直接将音频转换为模型使用的高维空间，与结合独立 ASR 和 LLM 组件的系统相比，响应速度更快。
- **用于模式化响应的 Cohere Schema 对象**: 一位用户对 Cohere 新增加的 Schema 对象表示热衷，认为它有助于在单个 API 调用中构建多个动作。
   - 他们正将此功能用于生成式小说，响应需要生成内容、建议角色动作并生成 Diffusion prompts。
- **Prompt Tuner 功能请求**: 多位用户表示希望 Prompt Tuner 能同时支持 preamble 和 prompt tuning。
   - 他们认为这将为各种模型提供更深度的分析并提高整体性能。


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1276455449574506548)** (4 messages): 

> - `Cohere Pricing`
> - `Tokenization`
> - `Cohere API`
> - `Oracle APEX`
> - `Command R Models` 


- **Cohere Pricing - 详细明细**: Cohere 的生成式模型，如 [Command R](https://docs.cohere.com/docs/command-r) 和 [Command R+](https://docs.cohere.com/docs/command-r-plus)，按 token 计费。
- **Tokenization 详解**: Cohere 语言模型理解的是 'tokens'（单词的一部分、整个单词或标点符号），而不是字符或字节。 
- **经验法则：1 个单词 = 1.5 个 tokens**: 一个经验法则是，一个单词大约等于 1.5 个 tokens。
- **Cohere 测试版使用与生产版使用**: Cohere 区分了“测试（trial）”和“生产（production）”用途。
- **Cohere 模型具有极高的成本效益**: 对于扩展生产用例，Cohere 模型是目前市场上最具成本效益的选择之一。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://forums.oracle.com/ords/apexds/post/error-generate-ai-with-apex-invalid-role-in-chat-history-6898">Error Generate AI with APEX - Invalid role in chat_history</a>: 未找到描述</li><li><a href="https://docs.cohere.com/docs/tokens-and-tokenizers">Tokens and Tokenizers — Cohere</a>: 该文档解释了语言模型使用 tokens 而非字符或字节，常用词拥有唯一的 tokens，而较长、频率较低的词则被编码为多个 tokens。</li><li><a href="https://docs.cohere.com/v1/docs/how-does-cohere-pricing-work">How Does Cohere Pricing Work? — Cohere</a>: 未找到描述</li><li><a href="https://cohere.com/pricing">Pricing</a>: 直接通过我们的 API 访问模型，以创建可扩展的生产工作负载。   
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1276297463518597120)** (5 messages): 

> - `Command R+ via HTTP`
> - `Structured Outputs` 


- **通过 HTTP 使用 Command R+**: 是的，**具备 128k 上下文的 Command R+** 可以通过 HTTP 请求使用，特别是使用 **`curl`**。
   - 您可以在 **Cohere API Reference** 中找到相关文档：[https://docs.cohere.com/reference/chat](https://docs.cohere.com/reference/chat)。
- **结构化输出（Structured Outputs）尚未推出**: **Cohere 目前不提供类似 OpenAI API 中的结构化输出功能**。



**提及的链接**: <a href="https://docs.cohere.com/reference/chat">Chat Non-streaming — Cohere</a>: 生成对用户消息的文本响应。要了解如何将 Chat API 与 Streaming 和 RAG 结合使用，请参考我们的 Text Generation 指南。

  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1276264425027932212)** (2 messages): 

> - `Cohere Models on Hugging Face Hub` 


- **Cohere 模型即将登陆 Hugging Face Hub**: 一位成员分享说，他们正在努力将所有主要模型（包括 Cohere 模型）打包并托管在 Hugging Face Hub 上。
   - 另一位成员对此消息表示兴奋。
- **Cohere 集成更新**: 未提及任何关于 Cohere 集成的更新，因此无法提供相关摘要。


  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1276281184975847544)** (17 messages🔥): 

> - `AI burnout`
> - `AI powerusers`
> - `Model Generations`
> - `Twitter fatigue` 


- **AI Burnout 引发关注**：一名成员表示担心 AI 带来的倦怠感（burnout）可能比人类严重得多。
   - 这一担忧得到了其他人的共鸣，他们指出前沿实验室（frontier labs）普遍存在的紧张工作氛围，认为这从长期来看是不可持续的。
- **AI Powerusers，一个“施法者”阶层**：一位成员指出，AI Powerusers 凭借对 AI 工具的持续使用，类似于一个“施法者阶层（spellcasting class）”。
   - 他们进一步假设，AI 模型能力的提升将增加对这些 Powerusers 的需求，从而导致更严重的倦怠。
- **模型迭代：倦怠的循环？**：一位成员建议，对“再多一代模型迭代”的无情追求可能会导致 AI 领域的显著倦怠。
   - 这种倦怠曲线的形状可能会发生变化，AI 的进步速度增加了精疲力竭的可能性。
- **Twitter 焦虑：与现实脱节？**：一位成员提到了 Greg Brockman 最近在 Twitter 上发布的一条帖子，显示他一周工作了 97 小时进行编码。
   - 其他人对 Twitter 上激烈的 AI 讨论感到不安，认为这会诱发焦虑，并可能导致与现实生活的脱节。


  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1276256069060001802)** (9 messages🔥): 

> - `Infinite Generative Youtube`
> - `TTS for Low Resource Languages`
> - `WhisperSpeech Semantic Tokens for ASR` 


- **为 Infinite Generative Youtube Beta 招募开发者**：一个团队正在寻找有兴趣构建“无限生成式 Youtube”平台的开发者。
   - 他们正准备近期启动封闭测试（closed beta），并寻求热情的开发者加入他们的团队。
- **针对印地语、乌尔都语和德语的 TTS**：一位用户表示有兴趣为印地语、乌尔都语和德语等低资源语言训练文本转语音（TTS）模型。
   - 他们提到希望将该 TTS 用于语音助手。
- **用于 ASR 的 WhisperSpeech 语义 Token**：一位用户询问了在低资源语言中使用 WhisperSpeech 语义 Token 进行自动语音识别（ASR）的可行性。
   - 他们提出了一个流程，包括在文本数据上训练一个小型的 decoder 模型，然后使用从现有音频和转录文本生成的语义 Token 对其进行微调（fine-tuning）。


  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1276454352885776396)** (1 messages): 

> - `SmolLM`
> - `Mamba`
> - `Cosmo2-tokenizer`
> - `BOS Tokens` 


- **SmolLM：更小的语言模型**：SmolLM 是一系列较小的语言模型，共有三种尺寸：135M、360M 和 1.7B 参数，是在一个名为 Cosmo-Corpus 的精心策划的数据集上训练的。
   - Cosmo-Corpus 包含 Cosmopedia v2、Python-Edu 和 FineWeb-Edu，SmolLM 模型在与其尺寸类别中的其他模型相比时显示出极具前景的结果。
- **Mamba：兼容 Transformers 的语言模型**：该仓库包含兼容 `transformers` 的 `mamba-2.8b` 模型。
   - 在 `transformers=4.39.0` 发布之前，你需要从 `main` 分支安装 `transformers`，并安装 `causal_conv_1d` 和 `mamba-ssm` 以使用优化的 `cuda` 内核。
- **Cosmo2-tokenizer：用于 Cosmo2 训练的分词器**：该分词器是在来自各种数据集（包括 FineWeb-Edu、Cosmopedia v2、StarCoderData、OpenWebMath 和 StackOverFlow）的 100 万个样本上训练的。
   - 该模型的下载量未被追踪，但该分词器在训练时特别关注了教育内容和代码。
- **缺失的 BOS Token 之谜**：cosmo2-tokenizer 和 SmolLM/Mamba 系列都没有独立的句子起始（BOS）Token。
   - 虽然技术上它们确实有一个 BOS Token，但它与句子结束（EOS）Token 相同，这可能会在预训练（pretraining）期间导致潜在问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/HuggingFaceTB/SmolLM-135M">HuggingFaceTB/SmolLM-135M · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/state-spaces/mamba-2.8b-hf">state-spaces/mamba-2.8b-hf · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/HuggingFaceTB/cosmo2-tokenizer">HuggingFaceTB/cosmo2-tokenizer · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1276485277644230760)** (2 messages): 

> - `Graph memory`
> - `Memory saving` 


- **Graph Memory 保存**: 一位成员询问是否可以将 memory 保存为文件，然后用于编译新的 graph。
   - 他们还询问相同的 memory 是否可以用于两个不同的 graph，还是每个 graph 独立的。
- **Graph Memory 保存**: 一位成员询问是否可以将 memory 保存为文件，然后用于编译新的 graph。
   - 他们还询问相同的 memory 是否可以用于两个不同的 graph，还是每个 graph 独立的。


  

---


### **LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1276602957697257604)** (5 messages): 

> - `LangChain Prompting`
> - `SQL Query Generation`
> - `LangChain Documentation`
> - `Chain Inspection` 


- **LangChain 中 SQL 查询的 Prompting 策略**: 这份来自 **LangChain Python 文档** (<https://python.langchain.com/v0.2/docs/how_to/sql_prompting/#table-definitions-and-example-rows>) 的文档概述了使用 **create_sql_query_chain** 改进 SQL 查询生成的策略。
   - 它涵盖了 **SQLDatabase 的 dialect** 如何影响 chain 的 prompt，如何使用 **SQLDatabase.get_context** 将 schema 信息格式化到 prompt 中，以及如何构建和选择 few-shot 示例来辅助模型。
- **LangChain Chain 中的显式上下文传递**: 代码行 `prompt_with_context = chain.get_prompts()[0].partial(table_info=context["table_info"])` 在 LangChain 中默认不会传递。
   - 如 LangChain Python 文档所示，在调用 chain 时，你需要显式传递包含 `table_info` 的 context。
- **LangChain Chain 检查**: **get_prompts()** 方法用于检索 LangChain chain 中使用的 prompt。
   - 该方法在 **LangChain Python 文档** (<https://python.langchain.com/v0.2/docs/how_to/inspect/#get-the-prompts>) 中有讨论，涵盖了以编程方式内省 chain 内部步骤的方法。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://python.langchain.com/v0.2/docs/how_to/sql_prompting/#table-definitions-and-example-rows>)">如何在进行 SQL 问答时更好地进行 Prompt | 🦜️🔗 LangChain</a>: 在本指南中，我们将介绍使用 createsqlquerychain 改进 SQL 查询生成的 prompting 策略。我们将主要关注获取相关数据库特定信息的方法...</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/inspect/#get-the-prompts>).">如何检查 runnables | 🦜️🔗 LangChain</a>: 本指南假设你熟悉以下概念：
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1276403227243450369)** (1 messages): 

> - `Writer Framework`
> - `Hugging Face Spaces`
> - `Docker Deployment` 


- **使用 Docker 将 Writer Framework 应用部署到 Hugging Face Spaces**: 发布了一篇博文，解释了如何通过 Docker 将 Writer Framework 应用程序部署到 Hugging Face Spaces。
   - Writer Framework 是一个开源 Python 框架，允许通过拖拽式构建器和 Python 后端构建 AI 应用程序，类似于 FastHTML、Streamlit 和 Gradio。
- **Writer Framework - AI 应用的拖拽式构建器**: Writer Framework 被描述为一个免费的开源 Python 框架，支持使用拖拽式界面创建 AI 应用程序。
   - 该框架提供 Python 后端，功能与其他流行框架（如 FastHTML、Streamlit 和 Gradio）类似。
- **Hugging Face Spaces 作为部署平台**: 博文详细介绍了利用 Docker 容器在 Hugging Face Spaces 上部署 Writer Framework 应用程序的过程。
   - 这种集成允许开发人员通过 Hugging Face 平台托管和共享他们的 AI 应用程序，展示了 Writer Framework 的强大功能和部署的便捷性。



**提及的链接**: <a href="https://huggingface.co/blog/samjulien/writer-framework-spaces">在 Hugging Face Spaces 中使用 Writer Framework</a>: 未找到描述

  

---



### **DSPy ▷ #[announcements](https://discord.com/channels/1161519468141355160/1209871299854336060/)** (1 messages): 

okhattab: https://lu.ma/03f7pesv
  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 messages): 

mrauter: https://arxiv.org/abs/2408.11326
  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1276447422326767750)** (5 messages): 

> - `Adalflow`
> - `DSpy vs Textgrad vs Adalflow` 


- **Adalflow: SylphAI 的新项目**：一名成员询问了 [Adalflow](https://adalflow.sylph.ai/get_started/index.html)，这是来自 [SylphAI](https://sylph.ai/) 的一个新项目。
   - 他们有兴趣探索其功能和潜在的应用场景。
- **比较 DSpy, Textgrad 和 Adalflow**：另一名成员对 **DSpy**、**Textgrad** 和 **Adalflow** 之间的区别，以及何时使用每个模块表示好奇。
   - 他们还提到 **LiteLLM** 将仅处理用于 Inference 的查询发送。



**提到的链接**：<a href="https://adalflow.sylph.ai/get_started/index.html">Get Started &#8212; AdalFlow: The Library to Build and Auto-Optimize LLM Task Pipelines</a>：未找到描述

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1276261171695194164)** (7 messages): 

> - `Open Interpreter brand guidelines`
> - `Phi-3.5-mini`
> - `Qwen2`
> - `Python screen clicking script`
> - `Data Analytics masterclass` 


- **Open Interpreter 品牌指南咨询**：一位用户询问在哪里可以获取 Open Interpreter 的品牌指南。
- **Phi-3.5-mini 令人惊讶的表现**：两位用户对 Phi-3.5-mini 出乎意料的优秀表现表示惊讶和赞同，随后提到了 Qwen2。
- **基于文本命令点击屏幕位置的 Python 脚本**：一位用户请求一个能够根据文本命令准确点击特定屏幕位置的 Python 脚本，并举例说明如“点击我的 notepad++ 窗口的文件下拉菜单”。
- **潜在解决方案：--os 模式**：一条回复建议 --os 模式可能适合这项任务。
- **免费 Data Analytics Masterclass 公告**：一位用户宣布了一个关于 Data Analytics 的免费大师课，强调了实践见解和真实世界的应用。
   - 公告提供了注册链接 [https://forms.gle/xoJXL4qKS8iq9Hxb7](https://forms.gle/xoJXL4qKS8iq9Hxb7) 并对潜在的参与表示期待。


  

---



### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1276257772068929557)** (4 messages): 

> - `Gorilla Leaderboard`
> - `Huggingface Leaderboard`
> - `Llama-3.1-Storm-8B` 


- **Gorilla 和 Huggingface Leaderboards 现已同步**：一名成员询问了 Gorilla 和 Huggingface Leaderboards 之间的差异，指出 Huggingface 上的分数更高。
   - 另一名成员回答说该问题已解决，Huggingface Leaderboard 现在是 Gorilla Leaderboard 的镜像。
- **Llama-3.1-Storm-8B 模型已添加到 Gorilla Leaderboard**：一位用户提交了一个 Pull Request (PR)，将 Llama-3.1-Storm-8B 添加到 Gorilla Leaderboard 进行基准测试。
   - 该 PR 已收到并将稍后进行审查。



**提到的链接**：<a href="https://github.com/ShishirPatil/gorilla/pull/598">[BFCL] Adding Llama-3.1-Storm-8B model handler by akshita-sukhlecha · Pull Request #598 · ShishirPatil/gorilla</a>：Llama-3.1-Storm-8B 模型最近发布。此 PR 为 Llama-3.1-Storm-8B 添加了模型处理器。

  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1276368076140580865)** (3 messages): 

> - `REST API testing`
> - `Test pairs`
> - `Gorilla leaderboard` 


- **用户寻求关于为 REST API 准备测试对的指导**：一位用户询问了创建用于 REST API 功能的“可执行测试对”的技术。
   - 他们参考了来自 [Gorilla leaderboard](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/data/BFCL_v2_rest.json) 的现有测试对，并想知道这些是手动准备的还是通过其他方法准备的。该用户强调希望得到“真实”且“易于”实现的测试。
- **需要对“可执行测试对”进行澄清**：另一位用户要求澄清在 REST API 测试语境下“可执行测试对”的含义。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#contributing">gorilla/berkeley-function-call-leaderboard at main · ShishirPatil/gorilla</a>：Gorilla: Training and Evaluating LLMs for Function Calls (Tool Calls) - ShishirPatil/gorilla</li><li><a href="https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/data/BFCL_v2_rest.json">gorilla/berkeley-function-call-leaderboard/data/BFCL_v2_rest.json at main · ShishirPatil/gorilla</a>：Gorilla: Training and Evaluating LLMs for Function Calls (Tool Calls) - ShishirPatil/gorilla
</li>
</ul>

</div>

### **AI21 Labs (Jamba) ▷ #[announcements](https://discord.com/channels/874538902696914944/874538945168408606/1276254821539643503)** (1 messages): 

> - `Jamba 1.5`
> - `SSM-Transformer architecture`
> - `Long context handling`
> - `Speed`
> - `Quality` 


- **Jamba 1.5 Mini & Large 发布**：AI21 Labs 宣布发布 **Jamba 1.5 Mini**（12B 激活参数/52B 总参数）和 **Jamba 1.5 Large**（94B 激活参数/398B 总参数），这两款模型均基于全新的 **SSM-Transformer Jamba architecture** 构建。
   - 这些模型提供了**卓越的长上下文处理能力、速度和质量**——在同尺寸级别中超越了竞争对手，并标志着非 Transformer 模型首次成功扩展到市场领先模型的质量和强度。
- **Jamba：长上下文之王**：Jamba 拥有 **256K 有效上下文窗口**，是目前市场上最长的，使其能够处理数千页文本、复杂代码和复杂的 Agent。
   - 它在**长上下文处理上快 2.5 倍**，成为同类模型中速度最快的，并提供了显著的性能优势。
- **Jamba 质量：同类顶尖**：Jamba 1.5 Mini 在 **Arena Hard 上得分为 46.1**，领跑其尺寸级别；而 Jamba 1.5 Large 得分为 **65.4**，表现优于 Llama 3.1 70B 和 405B。
- **Jamba：多语言支持且对开发者友好**：Jamba 支持**英语、西班牙语、法语**等，包括**希伯来语和阿拉伯语**，使其成为全球应用的强大工具。
   - 它提供对 **JSON 输出、function calling 和文档处理的原生支持**，方便开发者将其集成到项目中。
- **Jamba：开放且易于获取**：Jamba 已**可在 Hugging Face 上立即下载**，并可在各大云平台（Together AI, AWS, GCP, Azure 等）进行部署。
   - 这种开放的可访问性促进了进一步的实验，并允许开发者在其能力基础上进行构建。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/collections/ai21labs/jamba-15-66c44befa474a917fcf55251">Jamba-1.5 - ai21labs 集合</a>：未找到描述</li><li><a href="https://studio.ai21.com/v2/chat">AI21 Studio</a>：未找到描述</li><li><a href="https://www.ai21.com/jamba">基础模型</a>：未找到描述</li><li><a href="https://www.ai21.com/blog/announcing-jamba-model-family">Jamba 1.5 开源模型系列：最强大且高效的长上下文模型</a>：来自 AI21 的新开源模型系列，提供无与伦比的速度、效率和质量，并拥有开源模型中最长的上下文窗口。
</li>
</ul>

</div>
  

---


### **AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1276255102818320445)** (4 messages): 

> - `Jamba Fine-Tuning`
> - `Jamba Model Filtering` 


- **Jamba 不支持 UI 微调**：一位成员询问是否可以通过 UI 对 **Jamba** 进行微调，但工作人员确认微调仅适用于模型的 **instruct version**，而该版本目前无法通过 UI 获取。
- **用于角色扮演的 Jamba 过滤**：一位成员询问 **Jamba** 是否内置了针对暴力等内容的过滤器，特别是针对角色扮演场景。


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1276289164425433140)** (2 messages): 

> - `API Rate Limits` 


- **API 速率限制**：一位用户询问了 API 使用的速率限制。
   - 该用户随后提到找到了速率限制，即每分钟 200 次请求 (rpm) 和每秒 10 次请求 (rps)。
- **API 速率限制**：用户询问了 API 使用率。
   - 用户随后自己找到了限制，即每分钟 200 次请求 (rpm) 和每秒 10 次请求 (rps)。


  

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1276367828345163859)** (3 messages): 

> - `NVIDIA AI Summit India`
> - `AI Safety`
> - `Demo-Jam Hackathon` 


- **NVIDIA AI Summit India 正式启动**：NVIDIA AI Summit India 将于 2024 年 10 月 23 日至 25 日在孟买 Jio World Convention Centre 举行，届时将有 Jensen Huang 的炉边对话以及 50 多场关于 AI、robotics 等主题的会议。
   - 该活动旨在连接 NVIDIA 与行业领袖及合作伙伴，展示 generative AI、large language models、industrial digitalization、supercomputing、robotics 等领域的变革性工作和来自 AI 领袖的宝贵见解。
- **AI Capabilities and Risks Demo-Jam Hackathon**：AI Capabilities and Risks Demo-Jam Hackathon 启动，奖金池为 2000 美元，顶尖项目有机会加入 Apart Labs 并可能转化为研究论文。
   - 该活动鼓励参与者创建能够弥合 AI 研究与公众理解之间鸿沟的 Demo，展示 AI 驱动的潜在社会变革，并以引人入胜的方式传达 AI safety 挑战。
- **Hackathon 以开幕主题演讲和组队为特色**：Hackathon 以关于交互式 AI 演示的开幕主题演讲拉开帷幕，随后进行组队和项目构思。
   - 参与者可以获得专家导师和资源的支持，活动在 Youtube 上进行直播，任何人都可以观看创新的全过程。



**相关链接**：<a href="https://nvda.ws/3AbEKCi">加入 NVIDIA AI Summit 2024</a>：10 月 23–25 日，印度孟买

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1276257194765062154)** (2 messages): 

> - `tinygrad mypyc compilation` 


- **Tinygrad 的 Mypyc 编译探索**：一名成员表示有兴趣使用 **mypyc** 编译 **tinygrad**。
   - 他们表示目前正在调查该项目的可行性。
- **加入探索！**：原帖作者还邀请其他人为此努力做出贡献。


  

---



---



---



---



---



{% else %}


> 完整的频道详细分析已针对电子邮件进行截断。
> 
> 如果您想查看完整分析，请访问此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}