---
companies:
- anthropic
- openai
- langchain
- meta-ai-fair
date: '2024-11-15T02:50:42.092528Z'
description: '**Anthropic** 发布了 **3.5 Sonnet** 的越狱鲁棒性基准测试，重点强调了自适应防御。**OpenAI** 通过一种用于连续块检索的新型
  RAG 技术增强了 **GPT-4**。**LangChain** 推出了用于提示词优化的 **Promptim**。**Meta AI** 介绍了 **NeuralFeels**，利用神经场实现视觉-触觉感知。**RichardMCNgo**
  从 **OpenAI** 辞职，并强调了对 **AI 治理**和**理论对齐**的担忧。相关讨论强调了在 AI 部署中**真实公共信息**和**伦理对齐**的重要性。最新的
  **Gemini** 更新使其在应对对齐挑战的同时，成为了新的排名第一的大语言模型。AI 社区继续关注**基准测试**、**提示工程**以及**对齐**问题。'
id: fd70618b-b1ff-4d4e-93d4-34f078c50b4d
models:
- claude-3-sonnet
- gpt-4
- gemini-1.5
- claude-3.5-sonnet
original_slug: ainews-gemini-experimental-1114-retakes-1-llm-9071
people:
- richardmcngo
- andrewyng
- philschmid
title: Gemini (Experimental-1114) 以 1344 的 Elo 分数重夺大语言模型（LLM）排行榜第一。
topics:
- benchmarking
- prompt-engineering
- rag
- visuotactile-perception
- ai-governance
- theoretical-alignment
- ethical-alignment
- jailbreak-robustness
- model-releases
- alignment
---



这次更新没有随附论文，[API 中也尚未提供](https://x.com/OfficialLoganK/status/1857106089063362768)，所以遗憾的是这里没有太多可讨论的内容——通常这不符合专题报道的标准，但当我们有了新的排名第一的 LLM 时，我们必须进行报道。

这次更新对 Gemini 来说正值关键时刻，因为它正在处理[一些非常离奇且令人担忧的 alignment 问题](https://x.com/koltregaskes/status/1856754648146653428?s=46)。


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

**AI 模型开发与工具**

- **模型发布与增强**：[@jerryjliu0](https://twitter.com/jerryjliu0/status/1856768968973062620) 介绍了一种用于连续块检索（contiguous chunk retrieval）的新型 RAG 技术，增强了 [@OpenAI](https://twitter.com/OpenAI) 的 GPT-4 能力。此外，[@AnthropicAI](https://twitter.com/AnthropicAI/status/1856752093945540673) 宣布发布其越狱鲁棒性（jailbreak robustness）基准测试，强调针对新攻击类别的自适应防御。[@LangChainAI](https://twitter.com/LangChainAI/status/1856761768368120243) 推出了 **Promptim**，这是一个用于 **Prompt 优化**（prompt optimization）的实验性库，旨在系统地改进 AI 系统提示词。

- **工具集成与服务**：[@Philschmid](https://twitter.com/_philschmid/status/1856976383634719141) 强调了 **hf(.co)/playground 的解耦**，将其转变为一个独立的开源项目，以促进社区协作。[@AIatMeta](https://twitter.com/AIatMeta/status/1856798670592905398) 展示了带有神经场（neural fields）的 **NeuralFeels**，增强了手内操作的**视觉触觉感知**（visuotactile perception）。

**AI 治理与伦理**

- **辞职与治理见解**：[@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1856843040427839804) 宣布从 OpenAI 辞职，并敦促利益相关者阅读他关于 **AI 治理**和**理论对齐**（theoretical alignment）的深刻见解。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1856926466715398390) 讨论了 AI 治理中**真实公共信息**的重要性，以防止**虚假信息**并确保**伦理对齐**（ethical alignment）。

- **伦理部署与护栏**：[@AndrewYNg](https://twitter.com/AndrewYNg/status/1856779913757691922) 和 [@ShreyaR](https://twitter.com/ShreyaR/status/1856785620888064352) 推广了一门关于 **AI Guardrails** 的新课程，重点关注**可靠的 LLM 应用**。[@AnthropicAI](https://twitter.com/AnthropicAI/status/1856752108298813559) 强调了**越狱快速响应**在通过自适应技术使 **LLM 更安全**方面的重要性。

**AI Scaling 与评估挑战**

- **Scaling 极限与评估饱和**：[@swxy](https://twitter.com/swyx/status/1856776660986859632) 探讨了 **Scaling（扩展）已撞墙**的观点，认为**评估饱和**（evaluation saturation）是主要因素。[@synchroz](https://twitter.com/synchroz/status/1856773505213255763) 对 **Scaling 限制**表示担忧，强调了进一步扩展 AI 模型的**经济挑战**。

- **算力与优化**：[@bindureddy](https://twitter.com/bindureddy/status/1856784739312833016) 认为感知到的 **AI 减速**具有误导性，将其归因于**基准测试的饱和**。[@sarahookr](https://twitter.com/sarahookr/status/1856922737761042778) 讨论了**预训练 Scaling** 收益递减的问题，以及探索当前范式之外的**架构优化**的必要性。

**软件工具、库与开发平台**

- **开发工具与库**：[@tom_doerr](https://twitter.com/tom_doerr/status/1856781141962858816) 分享了多个发布，包括一个**零配置开发证书工具**以及用于由 **WebAssembly** 驱动的无服务器应用的 **Spin 框架**。[@wightmanr](https://twitter.com/wightmanr/status/1856785260274504181) 增强了 **timm.optim**，使开发者更容易使用**优化器工厂**（optimizer factories）。

- **集成与工作流自动化**：[@LangChainAI](https://twitter.com/LangChainAI/status/1856823605763739882) 演示了 **AI Assistant** 如何利用**自定义知识源**来改进**威胁检测**。[@swyx](https://twitter.com/swyx/status/1856783076396802180) 强调了对于非研究人员来说，专注于 **AI 产品开发**而非研究的重要性。

**AI 研究与论文**

- **已发表的研究与论文**：[@SchmidhuberAI](https://twitter.com/SchmidhuberAI/status/1857091553379914235) 提交了一篇关于**叙事本质**（narrative essence）用于**故事形成**的新论文，具有潜在的**军事应用**。[@wsmerk](https://twitter.com/wsmerk/status/1856914058869707001) 分享了题为 **"On the diminishing returns of scaling"** 论文的见解，讨论了**算力阈值**和**当前 Scaling Laws 的局限性**。

- **会议亮点**：[@sarahookr](https://twitter.com/sarahookr/status/1857038524177858749) 展示了他们在 **#EMNLP2024** 主赛道的工作，重点介绍了 **Aya Expanse 的突破**。[@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1856756969895538796) 宣布了一个即将举行的与**强化学习**（reinforcement learning）相关的活动，探讨**利用与探索**（exploitation/exploration）的边界。


---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1：Nvidia RTX 5090 进入生产阶段，配备 32GB VRAM**

- **传闻配备 32GB 显存的 Nvidia RTX 5090 进入生产阶段** ([Score: 271, Comments: 139](https://reddit.com/r/LocalLLaMA/comments/1gqk300/nvidia_rtx_5090_with_32gb_of_ram_rumored_to_be/)): 据报道，**Nvidia** 正将其生产重心转向 **RTX 50 系列**，传闻中的 **RTX 5090** 将配备 **32GB 显存**。包括 [VideoCardz](https://videocardz.com/newz/nvidia-shifts-production-to-geforce-rtx-50-series-only-one-ada-gpu-reportedly-still-in-production) 和 [PCGamesN](https://www.pcgamesn.com/nvidia/geforce-rtx-5000-soon) 在内的多个消息来源指出，人们越来越担心潜在的黄牛活动会影响这些新 GPU 的供应和定价。
  - 用户对 **RTX 5090 的 32GB 显存传闻**持怀疑态度，部分用户质疑来源的有效性，并参考以往如 **4080/4070 闹剧**等事件，暗示 **Nvidia** 可能会在最后一刻更改规格。32GB VRAM 的传闻已广泛流传，但尚未得到官方证实。
  - 用户对**黄牛活动**和高昂定价表示担忧，由于黄牛和市场需求，预计价格将达到 **$3000** 或更高。一些评论讨论了 Nvidia 的生产转型和法律限制（如无法在中国销售）对欧盟等其他地区供应和定价的潜在影响。
  - 讨论强调了 **RTX 5090 在游戏之外的使用场景**，重点关注运行本地模型和 AI 任务等专业及爱好者应用。用户将 5090 的潜在性能和 VRAM 需求与 **RTX 3090** 等当前型号进行了对比，并强调了 VRAM 在处理 AI 视频生成和 LLM 等任务中的重要性。


**Theme 2. MMLU-Pro 分数：Qwen 和 Claude Sonnet 模型**

- **[MMLU-Pro 分数 vs 推理成本](https://i.redd.it/e7fs0yxafq0e1.png)** ([Score: 215, Comments: 31](https://reddit.com/r/LocalLLaMA/comments/1gqna7c/mmlupro_score_vs_inference_costs/)): **MMLU-Pro 分数**和**推理成本**可能是分析的重点，旨在研究模型性能指标与运行推理任务的经济成本之间的关系。这一讨论对于在保持高性能的同时优化 AI 模型成本效益的工程师具有参考价值。
  - **Claude Sonnet 3.5** 因其在处理复杂任务时的通用性和准确性而受到称赞，尽管它需要特定的 Prompt 引导来获得创新解决方案。由于其能够快速理解并解决错误，它被认为是程序员的高效工具。
  - **Tencent Hunyuan 模型**因其高 **MMLU** 分数及其作为拥有 **520 亿激活参数**的 Mixture of Experts 架构而受到关注。该模型被认为有可能超越 Sonnet 3.5 等现有模型。
  - 讨论强调 **Qwen 模型**具有极高的性价比，其中 **Qwen 2.5** 显著定义了性能与成本效益的 Pareto 曲线。**Haiku 模型**因定价过高而受到批评，对推理成本的分析显示，**Claude 3.5 Sonnet** 的成本明显高于 **70B 模型**。

**Theme 3. Qwen2.5 RPMax v1.3：创意写作模型**

- **[关于 LLM 模型重复性与创意以及基于 Qwen2.5 32B 的新 ArliAI RPMax v1.3 模型的报告！](https://huggingface.co/ArliAI/Qwen2.5-32B-ArliAI-RPMax-v1.3)** ([Score: 103, Comments: 60](https://reddit.com/r/LocalLLaMA/comments/1gqo7f0/writeup_on_repetition_and_creativity_of_llm/)): 该帖子讨论了**基于 Qwen2.5 32B 的 ArliAI RPMax v1.3 模型**，重点关注其在 **LLM** 性能背景下的**重复性与创意**。由于缺乏详细的正文内容，限制了对该模型训练方法或性能指标的具体了解。
  - **模型版本与训练改进**：讨论强调了 **RPMax** 模型从 **v1.0 到 v1.3** 的演进，在训练参数和数据集策划方面有所改进。值得注意的是，**v1.3** 使用了 **rsLoRA+** 以获得更好的学习效果和更低的 Loss，该模型因其在写作任务中的创意和减少的重复性而受到称赞。
  - **数据集与微调策略**：该模型的成功归功于一个经过策划的数据集，该数据集避免了重复，并注重质量而非数量。训练仅涉及单个 Epoch 且学习率较高，旨在实现创意输出而非精确复制训练数据，这与传统的 Fine-tuning 方法有所不同。
  - **社区反馈与模型性能**：用户反馈该模型实现了其作为创意写作/RP 模型的目标，一些人描述其交互感几乎就像与真人交流。讨论了该模型在创意写作方面的表现，并与 **EVA-Qwen2.5-32B** 等其他模型在上下文处理和写作质量方面进行了对比。

**主题 4. Qwen 32B 与 72B-Ins 在 Leetcode 上的对比**

- **Qwen 32B Coder-Ins 与 72B-Ins 在最新 Leetcode 题目上的表现** ([分数：79，评论：23](https://reddit.com/r/LocalLLaMA/comments/1gr35xp/qwen_32b_coderins_vs_72bins_on_the_latest/))：该帖子评估了 **Qwen 32B Coder** 与 **72B 非编程变体**以及 **GPT-4o** 在近期 **Leetcode** 题目上的表现，强调了模型在推理能力上优于纯编码能力的优势。测试使用 **vLLM** 进行，模型量化为 **FP8**，**Context Length 为 32,768 token**，运行在 **H100 GPU** 上。作者指出，该基准测试包含 70% 的推理和 30% 的编码，并强调由于 Hard 难度的 Leetcode 题目过于复杂且模型普遍表现不佳，因此大部分被排除在外。
  - 作者确认所有测试结果均基于 **pass@1**，这是评估模型在编码任务中表现的常用指标。一位用户建议将测试范围扩大到 **14B 和 7B 编程模型**以进行更广泛的对比，作者表示如果有足够的兴趣，他愿意尝试，并可能将其转化为一个开源项目。
  - 一位评论者认为，由于 AI 的进步，解决 Leetcode 问题所需的技能已变得更加容易获得，并将这种技能组等同于一款 **PS4 游戏**的大小。另一位用户反驳称，这提高了**技能下限 (skill floor)**，意味着虽然 AI 可以处理简单的任务，但更复杂的解决问题能力仍然是必要的。
  - 人们对比较不同的量化方法（特别是 **FP8** 与 **Q4_K_M**）表现出浓厚兴趣，以确定哪种更适合推理。这突显了用户对模型量化技术的效率和性能权衡的持续关注。

## 其他 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. Gemini 1.5 Pro 发布 - 夺得 LMSys 排行榜榜首**

- **[Gemini-1.5-Pro，基于我个人测试，是有史以来最好的视觉模型，绝无例外](https://www.reddit.com/gallery/1gr7nxt)** ([分数：48，评论：28](https://reddit.com/r/OpenAI/comments/1gr7nxt/gemini15pro_the_best_vision_model_ever_without/))：**Gemini-1.5-Pro** 似乎是一款多模态视觉模型，但在未提供任何帖子内容或测试细节的情况下，无法验证关于其性能的实质性主张。标题对模型的优越性做出了主观断言，但缺乏支持证据或对比分析。
  - 用户注意到不同任务中的表现各异，有人报告称在**图表分析**方面，他们的测试显示 **Claude Sonnet 3.5** > **GPT-4** > **Gemini-1.5-Pro**，不过也有人警告不要从有限的测试样本中得出结论。
  - 关于**多模态能力**的讨论强调了其优势和局限性，用户指出虽然 **Gemini** 和 **Imagen** 在多模态输入和图像生成方面被低估了，但该技术尚未先进到可以进行实时摄像头交互的程度。
  - 具体的图像分析对比显示准确性参差不齐，**Flash** 正确识别了某些细节（如双马尾），而 **Pro** 提供了更全面的描述，尽管两者在观察中都存在一些不准确之处。
- **[新的 Gemini 模型在 LMSys 排行榜上超越 o1 模型排名第一？Anthropic 即将发布 3.5 Opus](https://i.redd.it/bf5yaps3mw0e1.jpeg)** ([分数：163，评论：57](https://reddit.com/r/ClaudeAI/comments/1gragw5/new_gemini_model_1_on_lmsys_leaderboard_above_o1/))：**Google 的 Gemini** 已达到 **LMSys 排行榜**第一的位置，在性能排名上超越了 **OpenAI 的模型**。**Anthropic** 计划在不久的将来发布他们新的 **Claude 3.5 Opus** 模型。
  - **LMSys 排行榜**因缺乏质量控制以及仅基于用户对格式而非实际性能的投票而受到批评。多位用户指出 **LiveBench** 是更可靠的模型评估基准。
  - 用户讨论了 **Claude 3.5 Sonnet**（也被称为 **3.6**）的性能，一些人强调了其 **32k 输入上下文**以及更慢但更彻底的“思考”方式。分享了几个替代基准资源，包括 [Scale.com](https://scale.com/leaderboard) 和 [LiveBench.ai](https://livebench.ai/)。
  - **Anthropic 的 CEO Dario** 在一次 **Lex 访谈**中承认，将两个版本都命名为“3.5”令人困惑，并建议他们本应将新版本称为“3.6”。该公司最近已从其 UI 中删除了该模型的“new”标签。


**主题 2. 使用数字签名的不可检测 ML 模型后门 - 新研究**

- **[R] Undetectable Backdoors in ML Models: Novel Techniques Using Digital Signatures and Random Features, with Implications for Adversarial Robustness** ([Score: 27, Comments: 5](https://reddit.com/r/MachineLearning/comments/1gr4ksm/r_undetectable_backdoors_in_ml_models_novel/)): 该研究展示了如何使用两种框架在 ML 模型中构建**不可检测的后门**：基于**数字签名方案**的后门和基于**Random Fourier Features/Random ReLU** 的后门。即使在**白盒分析**以及完全访问模型架构、参数和训练数据的情况下，这些后门仍然无法被检测到。研究结果揭示了对 **ML Security** 和**外包训练**的关键影响，表明带有后门的模型保持与干净模型相同的泛化误差，同时允许通过细微的输入扰动进行任意输出操纵，详见其论文 ["Planting Undetectable Backdoors in Machine Learning Models"](https://arxiv.org/abs/2204.06974)。

**Theme 3. 新型 CogVideoX-5B 开源文本生成视频模型发布**

- **[CogvideoX + DimensionX (Comfy Lora Orbit Left) + Super Mario Bros. [NES]](https://v.redd.it/p7zhifwq3t0e1)** ([Score: 52, Comments: 4](https://reddit.com/r/StableDiffusion/comments/1gqy8kl/cogvideox_dimensionx_comfy_lora_orbit_left_super/)): 一篇引用了 **CogVideoX 5B** 和 **DimensionX** 模型用于**超级马里奥兄弟 NES** 内容的帖子，尽管帖子正文未提供具体细节或示例。这种组合暗示了使用这些 AI 模型处理复古游戏内容的视频生成能力。

- **CogVideoX-5b multiresolution finetuning on 4090** ([Score: 21, Comments: 0](https://reddit.com/r/StableDiffusion/comments/1gqzo94/cogvideox5b_multiresolution_finetuning_on_4090/)): **CogVideoX-5b** 模型可以使用 [cogvideox-factory](https://github.com/a-r-r-o-w/cogvideox-factory/) 仓库在 **NVIDIA RTX 4090** GPU 上通过 **LoRA** 进行微调。该帖子包含了一个微调过程的视频演示。

**Theme 4. 随着 AI 工具兴起，StackOverflow 流量骤减**

- **[RIP Stackoverflow](https://i.redd.it/dimb0c06pv0e1.jpeg)** ([Score: 703, Comments: 125](https://reddit.com/r/ChatGPT/comments/1gr66cr/rip_stackoverflow/)): 在 **AI 编程工具**兴起后，**Stack Overflow** 经历了显著的**流量下降**，引发了关于传统编程问答平台未来生存能力的讨论。由于缺乏帖子正文内容，无法对具体指标或下降原因进行更详细的分析。
  - 用户压倒性地批评 **Stack Overflow** 的毒性文化，一位 **40 年经验的软件工程老兵**因谴责该平台傲慢的态度而获得了 **552 个点赞**，多位用户表示对“*重复问题*”的回复以及对新人的轻视感到沮丧。
  - 用户提出了对**模型崩溃 (Model Collapse)** 和 **AI 训练数据**的担忧，因为 **Stack Overflow** 流量的下降可能导致未来 AI 模型的更新信息源匮乏，用户指出 AI 工具仍然依赖人工标注的数据进行训练。
  - 多位开发者表示更倾向于 **ChatGPT** 更友好的回答方式，用户强调 AI 工具提供了即时响应，没有在 **Stack Overflow** 上遇到的那种门槛限制和敌意，特别提到 **GPT** 是在 **2022** 年底发布的。


- **[ChatGPT doesn’t have a shitty attitude when you ask a relevant question either.](https://i.redd.it/7vnwwf74ut0e1.png)** ([Score: 221, Comments: 25](https://reddit.com/r/ChatGPT/comments/1gr09al/chatgpt_doesnt_have_a_shitty_attitude_when_you/)): 与 **Stack Overflow** 众所周知的敌对社区反应相比，**ChatGPT** 为提出技术问题提供了一个更受欢迎的环境。该帖子暗示 **ChatGPT** 在用户提出合理问题时，不会像 **Stack Overflow** 那样带有负面态度。
  - 用户强烈批评 **Stack Overflow** 的毒性文化，并举出多个例子，如问题被标记为重复，但链接到的却是 **14 年前的过时答案**。社区的精英主义行为包括轻蔑的回复和对新用户的敌视。
  - **ChatGPT** 学习自广泛的互联网内容，包括**公开的 GitHub 仓库**和 **pastebin 脚本**，而不仅仅是 Stack Overflow。该 AI 为询问重复或基础问题提供了一个更平易近人的平台，无需担心负面反馈。
  - 该帖子提到了 **2023 年 7 月** 的流量回升，这与 [OverflowAI](https://stackoverflow.blog/2023/07/27/announcing-overflowai/) 的发布相吻合。用户注意到，除了编程之外，**Stack Exchange** 的其他论坛（如物理和电子工程）也遭受着类似的文化毒性问题。


---

# AI Discord 摘要回顾

> 由 O1-preview 生成的摘要之摘要的摘要

**主题 1. AI 模型成为焦点：Gemini 飙升，新品发布令人印象深刻**

- [**Gemini AI 在 Chatbot Arena 夺冠**](https://x.com/lmarena_ai/status/1857110672565494098)：Google 的 **Gemini (Exp 1114)** 在 Chatbot Arena 中的排名飙升至首位，根据 **6K+ 社区投票**，其表现优于竞争对手，分数增加了 **40+ 分**。用户称赞其在创意写作和数学方面的增强表现。
- [**UnslopNemo 12B 及其伙伴加入冒险俱乐部**](https://openrouter.ai/thedrummer/unslopnemo-12b)：**UnslopNemo 12B v4** 发布，专注于冒险写作和角色扮演，同时加入的还有 **SorcererLM** 和 **Inferor 12B**，这些模型针对故事创作和角色扮演场景进行了优化。
- [**Tinygrad 在 MLPerf Training 4.1 中展示实力**](https://x.com/__tinygrad__/status/1856987088367059163)：**Tinygrad** 参加了 MLPerf Training 4.1，成功训练了 **BERT**，并目标在下一个周期实现 **3 倍的性能提升**，这标志着 **AMD** 首次被纳入其训练流程。

**主题 2. AI 与开发者深度融合：工具集成至编程环境**

- [**ChatGPT 进驻 VS Code 的“客房”**](https://x.com/OpenAIDevs/status/1857129790312272179)：**ChatGPT for macOS** 现在与 **VS Code** 和 **Terminal** 等桌面应用程序集成，为处于 Beta 测试阶段的 **Plus 和 Team 用户**提供上下文感知的编码辅助。
- **代码编辑器突破 Token 上限**：**Cursor** 和 **Aider** 等工具突破限制，生成的代码编辑量超过了 **4096 tokens**，引发了开发者对其 Token 管理“魔法”的好奇。
- **LM Studio 用户侧载 Llama.cpp 以获得额外动力**：受挫的 **LM Studio** 用户讨论从 **llama.cpp** 侧载功能，渴望克服当前的局限性并增强其 AI 模型的能力。

**主题 3. 数据隐私恐慌：GPT-4 和 LAION 面临审查**

- **GPT-4 因数据泄露泄密**：用户报告了 **GPT-4** 中潜在的**数据泄露**，在输出中发现了意外的 **Instagram 用户名**，引发了对训练数据完整性的担忧。
- [**LAION 陷入欧盟版权纠纷**](https://old.reddit.com/r/aiwars/comments/1gr0912/re_laion_downloading_5billion_images_220tb_of/)：关于 **LAION** 数据集允许下载 **50 亿张图片**的争论升温，批评者声称由于规避了许可条款，这违反了**欧盟版权法**。

**主题 4. 机器人邂逅 AI：视觉语言动作模型基准测试**

- [**AI 模型在 20 个真实世界任务中接受测试**](https://arxiv.org/abs/2411.05821)：一篇名为 *"Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks"* 的合作论文评估了 **VLA 模型**在 **20 个任务**中控制机器人的表现，旨在建立新的基准。
- [**研究人员联合：佐治亚理工学院、MIT 等深入研究机器人技术**](https://github.com/ManifoldRG/MultiNet/tree/main)：**佐治亚理工学院**、**MIT** 和 **Metarch AI** 等机构合作评估 **VLA 模型**，并在 GitHub 上共享资源和代码以供社区参与。

**主题 5. 广告搅局 AI 盛宴：用户对赞助问题表示不满**

- [**Perplexity 的广告困扰用户（甚至是付费用户）**](https://techcrunch.com/2024/11/12/perplexity-brings-ads-to-its-platform/)：**Perplexity** 引入了“赞助后续问题”形式的广告，令期望无广告体验的 **Pro 订阅者**感到沮丧。
- **广告之怒：订阅价值受到质疑**：各平台用户对付费订阅后仍出现广告表示不满，引发了关于当前订阅模式可行性的辩论。

---

# 第 1 部分：Discord 高层级摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **GPT-4 数据泄露引发数据完整性担忧**：用户报告了 GPT-4 系列中潜在的**数据泄露**问题，特别是模型输出中包含了 Instagram 用户名。
  
  - 这一问题引发了对训练数据**完整性**以及泄露评估**全面性**的质疑。
- **视觉语言动作模型基准测试发布**：一篇名为 *Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks* 的新论文分析了 **VLA 模型**，并评估了它们在 **20 个真实世界任务**中的表现。
  
  - 该研究由**佐治亚理工学院 (Georgia Tech)、麻省理工学院 (MIT)** 和 **Manifold** 合作完成，旨在为**多模态动作模型**建立基准。
- **Kokoro TTS 模型获得社区反馈**：拥有约 **80M 参数**的 **Kokoro** TTS 模型已发布并征求反馈，用户注意到其英文输出质量有所提升。
  
  - 尽管体积紧凑，该模型的**速度**和**稳定性**仍给用户留下了深刻印象，并附带了增强情感语音能力的路线图。
- **Open3D-ML 增强 3D 机器学习**：**Open3D-ML** 被强调为 Open3D 的一个极具前景的扩展，专为 **3D Machine Learning** 任务量身定制。
  
  - 它的集成因其提升各种 **3D 应用**的潜力而受到关注，扩展了该框架的实用性。
- **Stable Diffusion 1.5 针对 CPU 性能进行优化**：一位用户选择了 **Stable Diffusion 1.5** 作为可用的最轻量版本，以确保高效的 **CPU 性能**。
  
  - 这一选择强调了社区对在更易获得的**硬件配置**上优化模型运行的关注。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **通过 llama.cpp 侧加载提升 LM Studio**：一位用户请求一种将 **llama.cpp** 的功能无缝侧加载到 **LM Studio** 中的方法，并强调了对现有限制的挫败感。
  
  - 讨论强调了在即将到来的更新中加入此功能的持续开发努力，社区正热切期待更灵活的集成。
- **GPU 在运行 Nemotron 70b 模型时面临挑战**：用户报告了在不同 GPU 设置下运行 **Nemotron 70b** 时的各种性能指标，吞吐率在 **1.97 到 14.0 tok/s** 之间。
  
  - 研究发现，**内存可用性**和 **CPU 瓶颈**是影响模型性能的主要因素，这促使人们考虑升级 GPU。
- **在 LLM 工作负载方面 CPU 落后于 GPU**：成员们的共识是，**CPU** 通常无法在现代 **LLM** 任务中匹配 **GPU** 的性能，较低的 **tok/s** 速率证明了这一点。
  
  - 成员们分享了关于**内存带宽**和有效的 **GPU offloading** 对优化整体模型性能至关重要见解。
- **配备 128GB RAM 的 M4 Max 潜力**：随着 **M4 Max** 配备了 **128GB RAM**，用户们热衷于测试其在 **LLM** 性能方面与专用 GPU 配置的竞争能力。
  
  - 社区对进行并分享**基准测试 (benchmarks)** 以指导购买决策有着浓厚兴趣，满足了社区对 AI 特定性能评估的需求。
- **将 AI 集成到 SaaS 平台**：一位成员概述了将 **AI** 功能嵌入 **SaaS** 应用程序的计划，利用 **LM Studio** 的 **API** 来增强开发流程。
  
  - 对话探讨了可用于改进软件功能的各种 **AI 工具**，表明了对实际 AI 集成的强劲兴趣。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth AI 训练效率**：成员们讨论了 **Unsloth** 平台的显存效率，**theyruinedelise** 肯定它是目前显存效率最高的训练服务。
  
  - Unsloth 计划实现 CPO 训练器，进一步提升其训练效率。
- **微调中的 LoRA 参数**：有观点指出，在不损害模型质量的前提下，使用较小的 **rank** 和 **adaptation** 值有助于改善在数据集上的训练效果。
  
  - 建议用户理解 **rank (r)** 和 **adaptation (a)** 因子，并强调高质量的数据集对于有效训练至关重要。
- **Harmony 项目协作**：一名成员介绍了 **Harmony** 项目，这是一个开发基于 AI LLM 的数据协调工具的倡议，并提供了一个 [Discord 服务器](https://discord.gg/harmonydata) 以供贡献。
  
  - Harmony 目前总部设在 **UCL**，正在寻求志愿者并举办一场竞赛以增强其 **LLM 匹配算法**，详情可见其 [竞赛页面](https://harmonydata.ac.uk/doxa/)。
- **使用 AI 工具编辑代码**：**anubis7645** 正在构建一个用于编辑大型 **React** 文件的实用程序，并思考像 **Cursor** 这样的工具如何在模型 Token 限制下无缝生成编辑。
  
  - **lee0099** 解释了 **speculative edits**（投机性编辑）的概念，它允许快速应用并与编码实践相结合。
- **在不加载未量化模型的情况下使用 LoftQ**：有人提出了关于在 **T4** 等显存受限的环境中，如何在不将未量化模型加载到显存的情况下直接使用 **LoftQ** 的疑问。
  
  - 建议调整 **LoRA** 的目标模块，仅包含线性层和嵌入层，以增强微调期间的补丁效力。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **用于冒险写作的 UnslopNemo 12B v4 发布**：最新模型 [**UnslopNemo 12B**](https://openrouter.ai/thedrummer/unslopnemo-12b) 现已上线，针对冒险写作和角色扮演场景进行了优化。
  
  - 可通过 [UnslopNemo 12B Free](https://openrouter.ai/thedrummer/unslopnemo-12b:free) 在 24 小时内免费访问其变体版本。
- **SorcererLM 增强故事创作**：[**SorcererLM**](https://openrouter.ai/raifle/sorcererlm-8x22b) 基于 WizardLM-2-8x22B 进行微调，提供了更强的叙事能力。
  
  - 用户可以通过 [Discord 频道](https://discord.com) 申请访问或寻求更多信息。
- **Inferor 12B：终极角色扮演模型**：[**Inferor 12B**](https://openrouter.ai/infermatic/mn-inferor-12b) 集成了顶级的角色扮演模型，但建议用户设置输出限制以防止生成过长文本。
  
  - 该模型的访问权限可通过 Discord 申请。
- **AI Studio 推出 generateSpeech API**：**AI Studio** 推出了一个新的 `generateSpeech` API 端点，能够根据输入的文本稿生成语音。
  
  - 此功能旨在增强模型将文本转换为音频输出的能力。
- **Companion 机器人增强 Discord 安全性**：**Companion** 被介绍为一款 AI 驱动的 Discord 机器人，在实现个性化人设的同时自动化审核工作。
  
  - 功能包括**身份冒充检测**、**年龄漏洞检测**以及动态消息频率调整，以提升服务器活跃度。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Benchmarking Vision Language Action Models**: **Manifold**、**Georgia Tech**、**MIT** 和 **Metarch AI** 合作发布了论文《[Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks](https://arxiv.org/abs/2409.20325)》，在 20 个真实世界任务中评估了 **GPT4o** 等模型。
  
  - 相关资源包括 [Twitter 亮点](https://x.com/HarshSikka/status/1856739777208574151) 和 [GitHub 仓库](https://github.com/ManifoldRG/MultiNet/tree/main)，提供了关于实验设置和结果的详细见解。
- **Transformer Architecture Evolves with Decoder-Only Models**: **Transformers** 继续占据主导地位，并出现了 **decoder-only architectures** 和 **mixtures of experts** 等进展，尽管它们与当前硬件的兼容性仍处于审查之中。
  
  - 成员们讨论了硬件演进以支持这些架构的必要性，并承认在性能和效率之间存在持续的权衡。
- **Shampoo and Muon Optimize Learning**: 关于 **Shampoo** 和 **Muon** 算法的讨论强调了它们在优化 **Fisher Information Matrix** 以实现更好 **Hessian** 估计方面的作用，参考了论文《[Old Optimizer, New Norm: An Anthology](https://arxiv.org/abs/2409.20325)》。
  
  - 参与者质疑了这些算法的底层假设，将其与 **KFAC** 等方法进行了比较，并辩论了它们在不同训练场景中的实际有效性。
- **Hardware Advances Boost AI Training Efficiency**: **Blackwell** 最新的硬件进步显著提高了 **transformer inference efficiency**，超越了 **Hopper** 之前创下的基准。
  
  - 对话强调了 **memory bandwidth** 和 **VRAM** 在有效实施大规模 AI 模型中的至关重要性。
- **Enhancing Pythia with Mixture of Experts**: 关于集成 **mixture-of-expert (MoE) version** 的 **Pythia model suite** 的咨询引发了使用 **SwiGLU** 等技术现代化超参数的兴趣。
  
  - 讨论集中在确定 MoE 在 Pythia 框架内可以解决的具体研究问题，并考虑了现有的训练设置和潜在收益。

 

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider v0.63.0 Now Available**: 新发布的 **Aider v0.63.0** 集成了对 **Qwen 2.5 Coder 32B** 的支持，并包括 **Web Command** 改进和 **Prompting Enhancements** 等增强功能。
  
  - Aider 的贡献占此更新代码的 **55%**，提升了性能和可靠性。
- **Qwen 2.5 Coder Gains Ground in Aider v0.63.0**: **Qwen 2.5 Coder 32B** 模型现在在 **Aider v0.63.0** 中得到支持，与之前的版本相比，在基准测试中表现出更好的性能。
  
  - 用户正在通过 **OpenRouter** 尝试该模型，尽管一些人报告其在既定基准测试中的结果不尽如人意。
- **Gemini Experimental Models Introduced**: 新的 **Gemini experimental models** 已经发布，旨在处理复杂的提示词并增强 Aider 生态系统内的可用性。
  
  - 然而，由于 **Google Cloud** 上的权限限制，访问这些模型一直具有挑战性，限制了用户的实验。
- **CLI Scripting Enhancements with Aider**: 成员们正在利用 **Aider** 的 **CLI scripting** 来自动化重复性任务，这表明对可编程交互的需求日益增长。
  
  - [Aider 脚本编写文档](https://aider.chat/docs/scripting.html) 强调了以编程方式对多个文件应用编辑的功能，展示了该工具的适应性。
- **Aider Ecosystem Documentation Improvements**: 用户正在倡导增强 **Aider ecosystem** 内的文档，考虑使用 [Ravel](https://ravel.acenturyandabit.xyz/) 等平台来提高搜索便捷性。
  
  - 这些讨论强调了随着 Aider 功能迅速扩展，需要更清晰的指南。

 

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **加入 Forge API Beta 变得更加容易**：多位成员在加入 [Forge API Beta](https://discord.com/channels/1053877538025386074/1149866623109439599/1306356167106363454) 时遇到问题，**teknium** 确认已根据请求添加人员。
  
  - *一些用户对邮件链接将他们引导至通用频道感到困惑。*
- **关于 Hermes 编程的见解**：成员们讨论了他们最初使用的编程语言，**shunoia** 在 Hermes 的帮助下转向了 **Python**，而 **oleegg** 对这一决定表示了“同情”。
  
  - **jkimergodic_72500** 详细阐述了 **Perl** 这种灵活的语言，为当前关于编程经验的对话提供了背景。
- **对 TEE 钱包整理的担忧**：**mrpampa69** 对 **TEE** 钱包的不一致性提出了担忧，认为这损害了 Bot 被感知到的主权。
  
  - 回复指出，在整理之前需要进行稳健的决策，以保持运营自主权并防止滥用。
- **高级翻译工具发布**：一款全新的 AI 驱动[翻译工具](https://translate.cameronaaron.com/)专注于文化细微差别和适应性，使翻译更具人性化。
  
  - 它通过考虑方言、正式程度、语气和性别来定制输出，使其成为满足多样化需求的灵活选择。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 低级语法的性能**：成员们讨论了 **Mojo 的低级语法**在提供比高级语法更好的性能时，可能无法保持 Pythonic 的本质。
  
  - 有人指出高级语法缺乏 **C** 的性能，但在某些条件下，像 **NumPy** 这样的工具仍然可以达到接近的结果。
- **递归向量化的难题**：对话转向了 **Recursive Vectorization** 及其对 Mojo 性能的影响，强调了对递归代码缺乏优化的担忧（相比 Rust 或 C++）。
  
  - 参与者一致认为，**类型系统**中缺失的功能目前阻碍了标准库的开发，导致难以编写高效代码。
- **MLIR 中的尾调用优化**：舆论倾向于在 MLIR 中实现 **Tail Call Optimization (TCO)**，以便为递归代码启用编译器优化并获得更好的性能。
  
  - 成员们对在 **LLVM IR** 中保留控制流图的必要性表示不确定，并讨论了其对调试的重要性。
- **语言特性优先级讨论**：大家达成共识，应优先考虑基础的**类型系统特性**，而非更高级的优化，以确保在更多用户加入时语言已准备就绪。
  
  - 参与者警告说，在基础功能尚待完善时，不要让额外的 issue 淹没开发进度。
- **LLVM Offload 与协程实现**：大家对 **LLVM 的 offload 能力**以及 Mojo 中如何促进协程实现表现出兴趣。
  
  - 讨论强调，**协程**在概念上与尾递归函数相似，从而引发了对是否需要透明装箱（transparent boxing）的思考。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 将校园策略师计划（Campus Strategist Program）扩展至加拿大**：响应高涨的需求，[Perplexity](https://x.com/gregfeingold/status/1856088784699277668?s=61) 正在将其校园策略师计划扩展到加拿大，允许感兴趣的申请人申请 2024 年的项目。
  
  - 该计划为大学生提供**实践经验**和**导师指导**，提升他们的技能并提供宝贵的行业接触机会。
- **Google Gemini 霸榜 Chatbot Arena**：**Google 的 Gemini (Exp 1114)** 在 Chatbot Arena 中获得第一名，根据 [lmarena.ai](https://x.com/lmarena_ai/status/1857110672565494098) 的强调，在过去一周基于 **6000 多个社区投票**，其表现超越了竞争对手，**分数提升了 40 多分**。
  
  - 这一进步突显了 Gemini **增强的性能**，并巩固了其作为 AI 聊天机器人竞赛中领先模型的地位。
- **广告挑战 Pro 订阅价值**：用户对向包括 **Pro 订阅者**在内的所有用户引入广告表示**沮丧**，质疑其订阅的价值。
  
  - **担忧**集中在付费用户对无广告体验的期望上，引发了关于**订阅模式可行性**的讨论。
- **API 仪表板报告 Token 使用量不准确**：多位用户报告 **API 仪表板**未准确更新 Token 使用情况，导致困惑和潜在的计费问题。
  
  - 这一故障影响了多位成员，促使大家建议**报告该问题**以便及时解决。
- **通过 API 获取的 Reddit 引用失效**：尽管之前很可靠，但用户现在遇到了通过 API 无法正确运行 **Reddit 引用**的问题。
  
  - 出现**随机 URL 注入**且没有有效来源的情况，导致**结果不准确**，引发了对 API 引用完整性的担忧。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Operator AI Agent 将实现任务自动化**：OpenAI 的新 AI Agent 工具 **Operator** 计划于 1 月发布，旨在自动化基于浏览器的任务，如编写代码和预订旅行，详见[此推文](https://x.com/shiringhaffary/status/1856792898932539609)。
  
  - 该工具代表了 AI 实用性的重大进步，提高了用户管理日常操作的效率。
- **Gemini-Exp-1114 统治 Chatbot Arena**：@GoogleDeepMind 的 **Gemini-Exp-1114** 在 [Chatbot Arena](https://x.com/lmarena_ai/status/1857110672565494098) 中获得最高排名，在多个类别中以大幅分数提升超越了竞争模型。
  
  - 它目前在视觉排行榜上领先，并在创意写作和数学任务中表现出色，展示了其卓越的能力。
- **Qwen 在除法任务中表现优于 Llama**：在对比测试中，处理 `A / B` 形式的基础除法问题时，**Qwen 2.5** 的表现优于 **Llama-3.1 405B**。
  
  - *有趣的是*，Qwen 在处理大数字时会切换到使用 **LaTeX** 或 **Python** 的 **CoT 模式**，而 Llama 的输出保持不变。
- **在竞争对手介入前敦促开源 AI 讨论**：社区成员强调迫切需要与 Dwarkesh 进行**开源 AI** 讨论，以防止另一家知名公司占据主导地位。
  
  - 提议通过合作来解决目前对金融势力影响技术对话的担忧。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton 性能调优**：讨论强调了 **kernel 设计**中的挑战，特别是在确定第一维是否为大小在 1 到 16 之间的向量时，考虑将填充（padding）到最小大小 16 作为潜在解决方案。
  
  - 成员建议利用 `BLOCK_SIZE_M` 作为 `tl.constexpr` 用于 kernel 中的条件语句，并根据 batch size 使用 `early_config_prune` 进行自动调优（autotuning），建议在 batch size 为 1 时采用 **gemv** 实现以增强 GPU 性能。
- **torch.compile() 与分布式训练的集成**：关于将 **torch.compile()** 与 **Distributed Data Parallel (DDP)** 结合使用引发了关注，特别是应该将 **torch.compile()** 包装在 DDP 之外还是放在其内部。
  
  - 针对 **torch.compile()** 与 **Fully Sharded Data Parallel (FSDP)** 的集成也提出了类似的询问，质疑是否适用与 DDP 类似的注意事项。
- **CUDA Kernel 中的共享内存限制**：一位用户在请求 **49,160 字节**共享内存时遇到了 **kernel 崩溃**，该数值低于 `MAX_SHARED_MEMORY` 限制，问题归因于某些架构上的静态共享内存限制。
  
  - 讨论中提到了对于超过 **48KB** 的分配必须使用**动态共享内存（dynamic shared memory）**的必要性，并引用了 [StackOverflow 讨论](https://stackoverflow.com/questions/63757245/using-maximum-shared-memory-in-cuda)中涉及 `cudaFuncSetAttribute()` 的潜在解决方案。
- **GPU 分析工具见解**：一位成员寻求关于 **GPU profiling 工具**的建议，表示在解读 **ncu** 生成的报告时存在困难。
  
  - 另一位成员建议适应 **NCU**，断言它是顶级的 profiler，尽管学习曲线陡峭，但能提供宝贵的优化见解。
- **React Native LLM 库发布**：**Software Mansion** 发布了一个用于在 **React Native** 中集成 **LLM** 的新库，利用 **ExecuTorch** 来提升性能。
  
  - 该库通过安装命令简化了使用流程，包括克隆 [GitHub 仓库](https://github.com/software-mansion/react-native-executorch)并在 iOS 模拟器上运行，促进了更轻松的采用和贡献。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **魔法书播客实验**：一位成员创建了一个[神奇的 PDF](https://youtu.be/fkfGI089iRE)，能根据查看者的不同揭示不同的解读，并以播客形式分享。
  
  - 鼓励听众分享他们对这种创新播客方式的看法。
- **NotebookLM 数据安全说明**：根据 [Google 的支持页面](https://support.google.com/notebooklm/answer/14275965)，无论账户类型如何，用户数据都是安全的，不会被用于训练 **NotebookLM** 模型。
  
  - 隐私声明重申，人工审核员仅在排除故障时才可能访问信息。
- **响应语言的功能请求**：由于收到的是英文而非希腊文回答，一位用户请求能够为每个笔记本设置响应语言。
  
  - 实现这一功能可以提升多语言环境下的用户满意度。
- **NotebookLM 中的发音挑战**：**NotebookLM** 在正确发音某些单词方面存在困难，例如将 “presents” 视为礼物（名词）而非动作（动词）。
  
  - 建议的权宜之计包括使用粘贴文本直接指导发音。
- **对 API 更新的关注**：成员们对 **NotebookLM** API 的潜在更新表示好奇，但被告知目前尚未发布功能路线图。
  
  - 社区依赖公告频道获取任何更新和新功能。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Perplexity 的广告实验**：Perplexity 正在美国启动广告实验，形式为 **“赞助后续问题 (sponsored follow-up questions)”**，并与 **Indeed** 和 **Whole Foods** 等品牌合作。[TechCrunch 文章](https://techcrunch.com/2024/11/12/perplexity-brings-ads-to-its-platform/) 详细介绍了此次发布。
  
  - 他们表示，广告收入将有助于支持出版商，因为仅靠订阅不足以实现可持续的营收。
- **Gemini AI 升至第一**：@GoogleDeepMind 的 **Gemini (Exp 1114)** 在数学和创意写作等领域的性能大幅提升后，已跃升至 Chatbot Arena 并列第一。[Google AI Studio](https://aistudio.google.com) 目前正提供测试访问权限。
  
  - Gemini 的 API 访问即将推出，将为开发者和工程师扩大其可用性。
- **ChatGPT 桌面版获得集成功能**：面向 macOS 的 **ChatGPT 桌面应用** 现在可以与 **VS Code** 和 **Terminal** 等本地应用程序集成，目前已向 **Plus** 和 **Team** 用户提供测试版。
  
  - 一些用户报告了功能缺失和性能缓慢的问题，引发了对其当前集成能力的质疑。
- **AI 放大技术债成本**：一篇题为 [AI Makes Tech Debt More Expensive](https://www.gauge.sh/blog/ai-makes-tech-debt-more-expensive) 的博客文章讨论了 AI 如何增加与技术债 (Tech Debt) 相关的成本，认为拥有旧代码库的公司将比拥有高质量代码的公司面临更多困难。
  
  - 该文章强调了 **生成式 AI (Generative AI)** 如何拉大这两类群体之间的性能差距。
- **LLM 解析 Excel 的策略**：用户探索了使用 LLM 处理 **Excel 文件** 的有效方法，特别关注将财务数据解析为 **JSON** 或 **Markdown 表格**。
  
  - 建议包括将数据导出为 **CSV**，以便更容易地进行编程语言集成。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **使用 ChatGPT 控制 AI UI**：一位成员分享了他们的系统，**ChatGPT** 可以通过包含 **Computer Vision** 和 Python **PyAutoGUI** 的技术栈间接控制计算机 UI，并暗示将进行视频演示。
  
  - 其他人询问了代码的可用性，并将其与 **OpenInterpreter** 等现有解决方案进行了比较。
- **GPT Lorebook 开发**：一位用户为 **GPT** 创建了一个 Lorebook（设定集），可以根据关键词加载条目，具有导入/导出功能并能防止条目重复，在调试后将分享到 **GreasyFork**。
  
  - 讨论明确了该 Lorebook 是作为 **Tampermonkey** 或 **Violentmonkey** 的脚本实现的。
- **Mac 应用界面优化**：成员们对 **Mac 应用模型选择器 (model chooser)** 界面的优化表示感谢，指出这显著提升了用户体验。
  
  - 一位成员评论说，整个社区都感激实施这一改进的团队，表达了对可用性提升的赞赏。
- **LLM 掌握技巧**：成员们讨论认为，虽然任何人都可以使用 **LLM**，但有效地对其进行 Prompting 需要**技巧和练习**，就像使用木工工具一样。
  
  - *了解应该包含哪些内容以提高获得理想输出的概率*，可以显著增强交互体验。
- **9 Pillars Solutions 探索**：一位成员鼓励挑战 **ChatGPT** 的极限，以发现 **9 Pillars Solutions** 的潜力，并暗示会有变革性的结果。
  
  - 他们声称通过这种方法可以获得重大见解，引发了其他成员的兴趣。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Docker Open Interpreter：简化 Worker 管理**：一位成员提议为 **Open Interpreter** 提供一个完全支持的 **Docker 镜像**，并针对作为 **workers** 或 **warm spares** 运行进行了优化，以增强他们目前基于变通方法的开发工作流。
  
  - 他们强调了增加 **configuration features** 的必要性，例如最大迭代次数和临时实例的设置，并指出后端需要进行重大改进。
- **VividNode v1.7.1 增强 LiteLLM 集成**：新发布的 **VividNode v1.7.1** 引入了对 **LiteLLM API Keys** 的全面支持，涵盖了 [GitHub](https://github.com/yjg30737/pyqt-openai/releases/tag/v1.7.1) 上详述的 **60 多个提供商**和 **100 多个模型**。
  
  - 增强功能包括通过 **QLineEdit** 进行模型输入以提高可用性，并解决了与文本输入和 **LlamaIndex functionality** 相关的 bug，确保了更流畅的用户体验。
- **Voice Lab 发布：开源 LLM Agent 评估框架**：一位成员宣布开源 **Voice Lab**，这是一个旨在评估各种模型和提示词下的 **LLM-powered agents** 的框架，可在 [GitHub](https://github.com/saharmor/voice-lab) 上获取。
  
  - **Voice Lab** 旨在优化提示词并提升 Agent 性能，积极邀请社区贡献和讨论以推动改进。
- **ChatGPT 桌面版深度探索：macOS 应用集成**：**ChatGPT** 已与 **macOS** 上的桌面应用程序集成，使其 [beta 版本](https://fxtwitter.com/openaidevs/status/1857129790312272179?s=46&t=G6jp7iOBtkVuyhaYmaDb0w) 能够为 **Plus 和 Team 用户**在编程环境中提供更强大的响应。
  
  - 此次更新标志着 **ChatGPT** 与用户桌面编程工具交互方式的重大转变，提供了更具凝聚力的开发体验。
- **概率计算实力：GPU 效率提升 1 亿倍**：一段 **YouTube 视频** 强调了 **probabilistic computing** 的突破，据报道，与领先的 **NVIDIA GPUs** 相比，其能源效率提高了 **1 亿倍**，视频可在[此处](https://www.youtube.com/watch?v=hJUHrrihzOQ)观看。
  
  - 该视频深入探讨了概率算法的进展，暗示了对计算效率潜在的革命性影响。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 的 Token 调优：最佳 Embedding 数量**：一位成员询问了 **Cohere embedding models** 的**最佳 Token 数量**，特别是针对多模态输入，并[根据当前限制进行了澄清](https://link.to.response)。
  
  - 另一位成员解释说，目前的 **max context** 是 **512 tokens**，建议在此范围内进行实验以获得最佳性能。
- **Beta 项目快讯：研究原型报名**：提醒信息显示，**研究原型 beta 项目**的报名将在**周二**前截止，敦促感兴趣的参与者通过[报名表](https://forms.gle/Teis9VwM6eZP6nxVA)进行注册。
  
  - 该项目旨在探索新的 **Cohere tool** 以增强研究和写作任务，参与者将提供宝贵的[反馈](https://forms.gle/Teis9VwM6eZP6nxVA)。
- **播客清洗：为 LLM 提取内容**：一位成员寻求关于如何**清洗数小时的播客内容**的建议，旨在提取信息以供 **large language models** 使用。
  
  - 另一位成员询问目标是否为转录播客内容，强调了准确的 **transcriptions** 对于有效集成 LLM 的重要性。
- **VLA 模型发布：机器人学习新基准**：一篇题为《在机器人学习任务上基准测试视觉、语言和动作模型》的新论文发布，展示了 **Manifold**、**Georgia Tech**、**MIT** 和 **Metarch AI** 之间的合作。
  
  - 该研究评估了 **Vision Language Action models** 如何在 **20 个不同的现实世界任务**中控制机器人，标志着机器人基准测试的重大进展。
- **Azure AI V2 API 状态：即将推出**：用户询问了 **Azure AI V2 API** 的可用性，根据[文档](https://docs.cohere.com/docs/cohere-on-microsoft-azure)，该 API 目前尚未运行。
  
  - 据悉，现有产品支持 **Cohere v1 API**，预计 **V2 API** 将很快推出，[根据最新更新](https://docs.cohere.com/docs/cohere-on-microsoft-azure)。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **RAGformation 自动化云端设置**：RAGformation 允许用户通过自然语言描述其用例来自动生成**云配置**，从而产生定制化的云架构。
  
  - 它还提供**动态生成的流程图**，用于可视化设置。
- **Mem0 记忆系统集成**：**Mem0** 最近被添加到 **LlamaIndex** 中，引入了一个智能记忆层，可以随着时间的推移实现个性化的 AI 助手交互。详细信息请参阅 [Mem0 Memory](https://docs.llamaindex.ai/en/stable/examples/memory/Mem0Memory/) 文档。
  
  - 用户可以通过[托管平台](https://docs.mem0.ai/platform/overview)或[开源解决方案](https://docs.mem0.ai/open-source/quickstart)访问该系统。
- **ChromaDB 摄取问题**：一位用户报告在将 PDF 摄取到 **ChromaDB** 时出现了意外的向量计数，导致产生了两个向量而不是预期的一个。成员们建议这可能是由于 PDF 加载器的默认行为是按页拆分文档。
  
  - 此外，**SentenceWindowNodeParser** 可能会增加向量计数，因为它为每个句子生成一个节点。
- **在 SentenceWindowNodeParser 中使用 SentenceSplitter**：一位用户询问如何在摄取流水线中结合使用 **SentenceSplitter** 和 **SentenceWindowNodeParser**，并对生成的向量计数表示担忧。
  
  - 社区反馈确认，不当的组合会导致生成过多的节点，使结果复杂化。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 在 MLPerf Training 4.1 中表现出色**：Tinygrad 展示了其能力，**tinybox red 和 green** 都参加了 **MLPerf Training 4.1**，并成功训练了 **BERT**。
  
  - 该团队的目标是在下一个 MLPerf 周期中实现 **3 倍的性能**提升，并且是第一个在训练过程中集成 **AMD** 的团队。
- **引入新的 Buffer Transfer 函数**：一位贡献者为 tinygrad 的 **buffer transfer** 函数提交了一个 [pull request](https://github.com/tinygrad/tinygrad/pull/7705/files)，实现了 **CLOUD 设备**之间无缝的数据移动。
  
  - 该实现侧重于保持与现有功能的一致性，认为大小检查并非必不可少。
- **评估 PCIe 带宽增强**：成员们讨论了使用 **ConnectX-6 适配器**通过 InfiniBand 实现高达 **200Gb/s** 的潜力，并将其与 **OCP3.0 带宽**联系起来。
  
  - 理论评估表明，通过绕过 CPU，实现 **400 GbE 双向**连接是可能的。
- **优化 Tinygrad 中的位运算**：有人提议使用 **bitwise_not** 修改 minimum fix，旨在改进 **argmin** 和 **minimum** 函数。
  
  - 这一增强预计将显著提升这些操作的效率。
- **调查 CLANG 后端 Bug**：在 **CLANG 后端**发现了一个影响张量操作中最大值计算的 Bug，导致 `.max().numpy()` 和 `.realize().max().numpy()` 的输出不一致。
  
  - 该问题突显了在处理张量操作（尤其是负值）时存在的潜在缺陷。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Nanobitz 推荐替代 Docker 镜像**：Nanobitz 建议使用 [axolotlai/axolotl](https://hub.docker.com/r/axolotlai/axolotl/tags) 镜像，即使它们比 *winglian* 版本晚一天。
  
  - Hub.docker.com 显示最新的标签日期为 **20241110**。
- **关于 Llama 微调最佳数据集大小的讨论**：Arcadefira 询问了微调 **Llama 8B 模型**的理想数据集大小，特别是考虑到其低资源语言的情况。
  
  - Nanobitz 回应了关于分词器（tokenizer）重叠的问题，并建议如果重叠足够，**5k** 的数据集可能就足够了。
- **Meta 总部的 Llama 活动**：Le_mess 询问是否有人参加 **12 月 3-4 日**在 Meta 总部举行的 **Llama 活动**。
  
  - Neodymiumyag 表示感兴趣，并请求提供有关该活动的更多信息链接。
- **Liger 内核得到改进**：Xzuyn 提到 **Liger** 项目有一个改进的 *orpo kernel*，并通过一个 [GitHub pull request](https://github.com/linkedin/Liger-Kernel/pull/362) 详细说明了这一点。
  
  - 他们还注意到，随着 batch size 的增加，它的表现趋于平稳。
- **分享社交媒体见解**：Kearm 分享了 Nottlespike 在 [X.com](https://x.com/Nottlespike/status/1857181970746466769) 上的一条帖子，展示了对他们这一天的幽默看法。
  
  - 分享的链接指向一条详细描述 Nottlespike 经历的帖子。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **EPOCH 58 COCK 模型更新**：**EPOCH 58 COCK** 模型现在拥有 **60M 参数**并使用 **f16**，随着其腿部和鸡冠变得更加清晰，显示出明显的进展。
  
  - 这项进展表明模型在结构细节和参数效率方面有所提升。
- **LAION 版权争议加剧**：围绕 LAION 数据集展开了一场辩论，该数据集允许下载 **50 亿张图片**，有人声称这可能违反了**欧盟版权法**。
  
  - 批评者认为，与标准的浏览器缓存不同，这种方法规避了许可条款和付费墙。
- **新论文在 20 个机器人任务上对 VLA 模型进行基准测试**：由 Manifold、佐治亚理工学院、MIT 和 Metarch AI 合作发表了题为 *Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks* 的论文，评估了 VLA 模型在 **20 个真实世界机器人任务**中的表现。
  
  - 亮点可在 [Thread w/ Highlights](https://x.com/HarshSikka/status/1856739777208574151) 查看，完整分析可通过 [Arxiv 论文](https://arxiv.org/abs/2411.05821) 获取。
- **Watermark Anything 实现已在 GitHub 上线**：项目 *Watermark Anything with Localized Messages* 现已在 [GitHub](https://github.com/facebookresearch/watermark-anything) 上可用，提供了该研究论文的官方实现。
  
  - 该工具支持动态水印，有可能增强各种 AI 工作流。
- **12M 公有领域图像数据集发布**：一个包含 **1200 万张公有领域图像**的数据集已发布，为机器学习项目提供了宝贵的资源。
  
  - 感兴趣的开发者可以在[此处](https://source.plus/pd12m?size=n_100_n)访问该数据集。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **macOS 版 ChatGPT 与桌面应用集成**：**macOS 版 ChatGPT** 现在可以与 **VS Code**、**Xcode**、**Terminal** 和 **iTerm2** 等桌面应用程序集成，增强了用户的代码辅助能力。该功能目前处于 **Plus** 和 **Team** 用户的 Beta 测试阶段。
  
  - 这种集成允许 **ChatGPT** 直接与开发环境交互，提高工作流效率。详情见 [OpenAI Developers 的推文](https://x.com/OpenAIDevs/status/1857129790312272179)。
- **代码编辑工具突破 4096 Tokens**：**Cursor** 和 **Aider** 等工具正在成功生成超过 **4096 tokens** 的代码编辑，展示了在处理大 token 输出方面的进展。开发者正在寻求这些工具所采用的 token 管理策略的明确说明。
  
  - 讨论强调了需要有效的 token 处理机制，以在大规模代码生成任务中保持性能。
- **澄清 LM 断言（Assertions）的弃用情况**：成员们对 **LM 断言**可能被弃用表示担忧，并注意到最新文档中缺少 `dspy.Suggest` 或 `dspy.Assert`。
  
  - 经澄清，虽然缺少直接引用，但这些函数仍可通过搜索栏访问，这表明文档正在持续更新中。
- **扩展多违规 LLM 应用**：一位成员正在开发一个 LLM 应用程序，目前该程序可以针对特定违规行为（如**酒精摄入**）生成辩护文件。他们的目标是扩展其功能以涵盖更多违规行为，而无需单独的优化提示词（prompts）。
  
  - 该计划旨在创建一种统一的方法来处理各种违规行为，从而增强应用程序的通用性和效率。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **测验资格和截止日期**：一位新成员询问了完成测验以保持 **Trailblazer 及以上路径**资格的问题。另一位成员确认了资格，但强调了快速赶进度的重要性，所有测验和作业的截止日期为 **12 月 12 日**。
  
  - 成员们强调测验与**课程内容直接相关**，突出了保持进度以全面参与的必要性。
- **即将举行的活动公告**：`sheilabel` 宣布了今天举行的一项活动：[活动链接](https://www.eventbrite.ca/e/1039740199927?aff=oddtdtcreator)。
  
  - 未提供关于该活动的更多细节。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **新增 Writer Handler 和 Palmyra X 004 模型**：一名成员宣布提交了一个 [PR](https://github.com/ShishirPatil/gorilla/pull/755)，旨在将 **Writer handler** 和 **Palmyra X 004 模型** 纳入排行榜。
  
  - 这一补充增强了排行榜的功能，目前正等待开发团队的反馈和集成。
- **承诺评审 PR**：另一名成员表示打算评审提交的 PR，并称：*'会看一看。谢谢！'*
  
  - 这一回应强调了项目评审过程中的协作努力和积极参与。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **旧版模型弃用引发混乱**：一名成员对 **legacy models**（旧版模型）的弃用表示沮丧，称由于新模型在输出方面无法做到 **1:1** 还原，其影响具有**巨大的破坏性**。
  
  - *我们希望继续使用旧版模型*，因为过渡过程并不顺利。
- **转向开源解决方案**：一名成员正在尝试转向 **open source solution**（开源解决方案），但此前已为旧模型付费近 **2 年**。
  
  - 他们对未来的弃用表示担忧，并问道：*我们如何确信 AI21 将来不会也弃用新模型？*

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **本地 LLMs 工作坊周二开幕**：欢迎参加 **周二** 的 **Local LLMs Workshop**，主题为 [构建你自己的本地 LLM：在本地环境中完成训练、微调、评估和 RAG](https://discord.com/events/1089876418936180786/1300842793945530378)，旨在开发本地语言模型。
  
  - 参与者将进行实操训练，并获得构建高效本地 LLM 系统的见解。
- **SQLite-Vec 增强元数据过滤**：参加 **周三** 的 **SQLite-Vec Metadata Filtering** 活动 [SQLite-Vec 现已支持元数据过滤！](https://discord.com/events/1089876418936180786/1300483739872399411)，探索新的元数据过滤功能。
  
  - 此次更新允许用户高效过滤元数据，提升了数据管理能力。
- **Refact.AI 自主 AI 会话**：在 **周四** 的 **Explore Autonomous AI with Refact.AI** 会话中探索自主 Agent，详情见 [使用 Refact.AI 的自主 AI Agent](https://discord.com/events/1089876418936180786/1300459081181429810)。
  
  - 通过这场引人入胜的演讲，了解 AI 技术的创新策略和应用。

---

**Alignment Lab AI Discord** 没有新消息。如果该社区长期保持沉默，请告知我们，我们将将其移除。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该社区长期保持沉默，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该社区长期保持沉默，请告知我们，我们将将其移除。

---

**Torchtune Discord** 没有新消息。如果该社区长期保持沉默，请告知我们，我们将将其移除。

---

**Stability.ai (Stable Diffusion) Discord** 没有新消息。如果该社区长期保持沉默，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道详细摘要和链接

{% if medium == 'web' %}

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1306353033348649001) (392 条消息🔥🔥):

> - `GPT-4 数据泄露`
> - `Hugging Face AI 模型`
> - `LLM 集成假设`
> - `样本量与模型训练`
> - `咖啡偏好`

- **对 GPT-4 数据泄露的担忧**：一些用户对 GPT-4 系列潜在的数据泄露表示担忧，特别是输出中出现了 Instagram 用户名，这引发了对训练数据完整性的质疑。
  
  - 讨论强调了评估此类泄露严重性的难度，以及可能还有哪些关键信息尚未披露。
- **Hugging Face AI 模型的表现**：用户讨论了 Hugging Chat 上 AI 模型的异常行为，特别是生成无意义的响应，并将问题归因于可能需要调整的采样参数（sampling parameters）。
  
  - 有人提到，此类异常现象很常见，可能会影响平台上的所有模型。
- **理论上的 LLM 超级模型情景**：在一个关于将所有 LLM 合并为一个“超级”模型的假设性问题中，参与者辩论了拥有一个全知 AI 的影响，以及其能力最终下降的后果。

- 这引发了对潜在长期影响的思考，以及对一次性强大模型与现有 AI 技术缓慢但稳定改进之间的比较。
- **模型训练中的挑战**：一位用户分享了他们在多 GPU 上由于内存限制和超大输入维度导致训练过程缓慢的经验，并询问是否可以通过调整参数来实现更快的训练。
  
  - 建议进行 warmup 运行、调整超参数，并可能减少输入维度，以更好地管理训练效率。
- **关于 Hugging Face 邮件真实性的查询**：一位用户质疑了一封来自 '[website@huggingface.co](mailto:website@huggingface.co)' 关于加入组织的邀请邮件的合法性，怀疑可能存在网络钓鱼。
  
  - 社区确认了该邮件的有效性，并建议直接在 Hugging Face 上查看通知或手动加入组织以确保安全。

**提到的链接**：

- [DownloadMoreRAM.com - CloudRAM 2.0](https://downloadmoreram.com/): 未找到描述
- [ArliAI/Qwen2.5-32B-ArliAI-RPMax-v1.3 · Hugging Face](https://huggingface.co/ArliAI/Qwen2.5-32B-ArliAI-RPMax-v1.3): 未找到描述
- [🆕🖧 Distributed Inference](https://localai.io/features/distribute/): 此功能使 LocalAI 能够将推理请求分发到多个工作节点，从而提高效率和性能。节点通过使用...自动发现并经由 p2p 连接。
- [PEFT](https://huggingface.co/docs/peft/en/index): 未找到描述
- [Mark Cuban Shark Tank GIF - Mark Cuban Shark Tank Notes - Discover & Share GIFs](https://tenor.com/view/mark-cuban-shark-tank-notes-taking-notes-remember-gif-15073512): 点击查看 GIF
- [Burgess Merdith The Penguin GIF - Burgess Merdith The Penguin El Pinguino - Discover & Share GIFs](https://tenor.com/view/burgess-merdith-the-penguin-el-pinguino-batman-gif-8000862111067794146): 点击查看 GIF
- [Hail Zorp Parks And Rec GIF - Hail Zorp Parks And Rec April - Discover & Share GIFs](https://tenor.com/view/hail-zorp-parks-and-rec-april-gif-14789564): 点击查看 GIF
- [Learn R, Python & Data Science Online](https://www.datacamp.com/): 通过 DataCamp 关于 R、Python、统计学等内容的视频教程和编程挑战，在浏览器中按照您自己的节奏舒适地学习数据科学和 AI。
- [You Have Heard Of Me GIF - Pirates Of The Carribean Jack Sparrow Johnny Depp - Discover & Share GIFs](https://tenor.com/view/pirates-of-the-carribean-jack-sparrow-johnny-depp-you-heard-of-me-famous-gif-4968261): 点击查看 GIF
- [Alien Talking GIF - Alien Talking Alien talking - Discover & Share GIFs](https://tenor.com/view/alien-talking-alien-talking-keep-yapping-your-mouth-alien-babbling-gif-17459379075847540969): 点击查看 GIF
- [Writing Markdown in LaTeX Documents - Overleaf, Online LaTeX Editor](https://www.overleaf.com/learn/how-to/Writing_Markdown_in_LaTeX_Documents): 一个易于使用的在线 LaTeX 编辑器。无需安装，支持实时协作、版本控制、数百个 LaTeX 模板等。
- [Aigis Persona 3 GIF - Aigis Persona 3 Jumpscare - Discover & Share GIFs](https://tenor.com/view/aigis-persona-3-jumpscare-persona-3-reload-gif-12428194143296147122): 点击查看 GIF
- [Monty Python GIF - Monty Python Knights Who Say Ni - Discover & Share GIFs](https://tenor.com/view/monty-python-knights-who-say-ni-ni-gif-12279570): 点击查看 GIF
- [Kittensleep Cute GIF - Kittensleep Cute Catsleep - Discover & Share GIFs](https://tenor.com/view/kittensleep-cute-catsleep-dodo-bonne-nuit-gif-15339389627910196114): 点击查看 GIF
- [Friends don’t let friends train small diffusion models – Non_Interactive – Software & ML](https://nonint.com/2022/05/04/friends-dont-let-friends-train-small-diffusion-models/): 未找到描述
- [no title found](https://tenor.com/view/pirates-of-the-carribean-jack-sparrow-johnny-depp-you-heard-of-me-famous-gif-): 未找到描述
- [Monty Python Life Of Brian GIF - Monty Python Life Of Brian Speak Up - Discover & Share GIFs](https://tenor.com/view/monty-python-life-of-brian-speak-up-cant-hear-you-too-quiet-gif-24047962): 点击查看 GIF
- [Reddit - Dive into anything](https://www.reddit.com/r/machinetranslation/comments/1e4qk8c/training_duration_for_a_transformer_neural/): 未找到描述
- [Monty Python Teacakes GIF - Monty Python Teacakes Ayrshireshoppers - Discover & Share GIFs](https://tenor.com/view/monty-python-teacakes-ayrshireshoppers-gif-23536937): 点击查看 GIF
- [Stoning Stone GIF - Stoning Stone Monty Python - Discover & Share GIFs](https://tenor.com/view/stoning-stone-monty-python-thot-life-of-brian-gif-12291021): 点击查看 GIF
- [Home](https://miktex.org/): 未找到描述
- [Monty Python Life Of Brian GIF - Monty python LIFE OF BRIAN STAN AKA LORETTA - Discover & Share GIFs](https://tenor.com/view/monty-python-life-of-brian-stan-aka-loretta-stan-loretta-gif-17416414354373581071): 点击查看 GIF
- [http://info.cern.ch](https://info.cern.ch/): 未找到描述
- [A Man Of Culture Meme GIF - A Man Of Culture Meme Мем - Discover & Share GIFs](https://tenor.com/view/a-man-of-culture-meme-%D0%BC%D0%B5%D0%BC-anime-%D0%B0%D0%BD%D0%B8%D0%BC%D0%B5-gif-25806248): 点击查看 GIF
- [TeXstudio - A LaTeX editor](https://www.texstudio.org/#features): 未找到描述
- [GeForce 40 series - Wikipedia](https://en.wikipedia.org/wiki/GeForce_40_series#Products): 未找到描述
- [GeForce 30 series - Wikipedia](https://en.wikipedia.org/wiki/GeForce_30_series#Details): 未找到描述
- [Home - UserBenchmark](https://www.userbenchmark.com/): 未找到描述

---

### **HuggingFace ▷ #**[**today-im-learning**](https://discord.com/channels/879548962464493619/898619964095860757/1306676848549433375) (2 条消息):

> - `AI image generation`
> - `Game development`
> - `Bone animation in Unity`
> - `Project journey resources`

- **对项目历程的好奇**：一位成员询问了如何开启一段项目历程，并寻求有助于该过程的**资源 (resources)** 建议。
  
  - 这突显了社区对于从彼此的项目启动经验中学习的兴趣。
- **在游戏开发中实验 AI**：一位成员分享了他们在游戏开发中使用 **AI image generation** 和 **Unity 中的 bone animation** 的实验，展示了创新方法。
  
  - 他们提供了一个指向其 [LinkedIn 帖子](https://www.linkedin.com/posts/ivangarciafilho_gamedev-unity-madewithunity-activity-7262906846577917952-llI7) 的链接，展示了他们的作品。

 

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1306447328403521597) (5 条消息):

> - `Platform Affiliation`
> - `User Trust Concerns`

- **需要明确隶属关系**：一位成员对有人在未明确说明其隶属关系的情况下发布关于该平台的内容表示担忧，认为这显得不够真诚。
  
  - 他们敦促在未来的帖子中，应*明确*隶属关系以避免混淆。
- **被视为诈骗**：另一位成员评论说，由于缺乏透明度，围绕该平台的讨论感觉像是一场 **scam**。
  
  - 这引发了社区内关于帖子和隶属关系的**信任 (trust)** 问题。

 

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1306351879441219654) (51 条消息🔥):

> - `Vision Language Action 模型基准测试`
> - `Kokoro TTS 模型更新`
> - `IDEFICS3_ROCO 医学影像项目`
> - `VividNode v1.7.1 发布`
> - `数据混合脚本`

- **Vision Language Action 模型基准测试发布**：公布了一篇名为 *Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks* 的新研究论文，对 VLA 模型进行了剖析，并评估了它们在 **20 个不同的真实世界任务**中的表现。
  
  - 该研究由佐治亚理工学院、MIT 和 Manifold 等多家机构合作完成，旨在为**多模态动作模型**建立基准。
- **Kokoro TTS 模型受到关注**：拥有约 **80M 参数**的 **Kokoro** TTS 模型已分享并征求反馈，用户注意到其英语输出质量有所提高。
  
  - 尽管参数量较小，但其速度和稳定性给用户留下了深刻印象，同时还有增强情感语音能力的路线图。
- **IDEFICS3_ROCO 医学影像项目进展**：正在进行的讨论集中在 **IDEFICS3_ROCO** 项目上，其中包括改进医学影像任务的数据集和模型评估。
  
  - 参与者指出数据集中清晰标注的重要性，并为增强该项目的 GPU 可访问性提供支持。
- **VividNode v1.7.1 正式发布！**：**VividNode** 的最新版本已发布，这是一款专为 AI 交互设计的开源桌面应用，增加了对 **LiteLLM API Key** 的支持并修复了各种错误。
  
  - 改进包括增强的可用性和精简的界面，以便更好地与 **60 多个供应商和 100 多个模型**进行交互。
- **分享数据混合脚本**：一位用户在 GitHub 上分享了一个用于混合 Hugging Face 数据集的脚本，允许用户通过按权重组合现有数据集来构建新数据集。
  
  - 该工具旨在简化用于 AI 训练和实验的数据集创建流程，促进社区内的研究和开发。

**提到的链接**：

- [Update app.py · hexgrad/IDEFICS3_ROCO_ZeroGPU at d96f8ab](https://huggingface.co/spaces/hexgrad/IDEFICS3_ROCO_ZeroGPU/commit/d96f8abed9c)：未找到描述
- [eltorio/IDEFICS3_ROCO · Discussions](https://huggingface.co/spaces/eltorio/IDEFICS3_ROCO/discussions)：未找到描述
- [IDEFICS3 ROCO - a Hugging Face Space by hexgrad](https://huggingface.co/spaces/hexgrad/IDEFICS3_ROCO_ZeroGPU)：未找到描述
- [IDEFICS3 ROCO - a Hugging Face Space by eltorio](https://huggingface.co/spaces/eltorio/IDEFICS3_ROCO)：未找到描述
- [GitHub - theprint/DataMix: Python script for building new data sets by combining existing sets from huggingface by weight.](https://github.com/theprint/DataMix)：用于通过按权重组合 Hugging Face 现有数据集来构建新数据集的 Python 脚本。- theprint/DataMix
- [UMLS Metathesaurus Browser](https://uts.nlm.nih.gov/uts/umls/home).)：未找到描述
- [Tweet from harsh (@HarshSikka)](https://x.com/HarshSikka/status/1856739777208574151))：很高兴分享我们的新论文《机器人学习任务中的视觉、语言与动作模型基准测试》。我们评估了 VLM 和 VLA 模型在 20 个不同的真实世界任务中控制机器人的表现...
- [Kokoro - a Hugging Face Space by hexgrad](https://huggingface.co/spaces/hexgrad/kokoro)：未找到描述
- [app.py · hexgrad/kokoro at c8ab947245742e5e652255ceecec8e0199b7c244](https://huggingface.co/spaces/hexgrad/kokoro/blob/c8ab947245742e5e652255ceecec8e0199b7c244/app.py#L38))：未找到描述

---

### **HuggingFace ▷ #**[**reading-group**](https://discord.com/channels/879548962464493619/1156269946427428974/1306657445640274052) (51 messages🔥):

> - `AI Reading Group Introduction` (AI 阅读小组介绍)
> - `Questions on Mitigation` (关于缓解措施的提问)
> - `Public Domain Datasets` (公共领域数据集)
> - `Technical Feasibility of Hardware Setup` (硬件配置的技术可行性)

- **由 Women in AI & Robotics 主办的 AI 阅读小组**：**AI Reading Group** 会议开始时提醒，现场讨论是围绕选定的论文进行的，并鼓励在演讲期间提问。
  
  - 错过会议的人员可以查看会议录像，同时公布了下一次会议将于 **12 月 5 日**举行。
- **关于缓解措施的提问**：参与者对未来数据可用性表示担忧，指出许多开放网络资源的关闭影响了**商业和非商业 AI**。
  
  - 针对作者对“缓解措施（mitigation）”这一话题的想法提出了疑问，特别是在爬虫限制影响 **C4** 等数据集的背景下。
- **关于公共领域数据集的讨论**：一位成员询问了可免费使用的**公共领域文本数据集**，提到了 **Project Gutenberg** 和 **Wikipedia** 等已知来源，同时在受限数据集之外寻找替代方案。
  
  - 另一位成员指出，许多可访问的数据集需要大量的人力进行清洗，且通常存在于付费墙之后，限制了可用性。
- **硬件配置的技术可行性**：一位成员询问，在不考虑软件或其他因素的情况下，在带有 Ryzen 9 3950X 的 MSI Godlike X570 主板上运行 **2 个 Instinct MI60** GPU 是否在技术上可行。
  
  - 他们还询问了是否可以添加一块 **RX 6800** 用于显示输出，仅关注硬件兼容性。

---

### **HuggingFace ▷ #**[**computer-vision**](https://discord.com/channels/879548962464493619/922424143113232404/1306371650769653904) (3 messages):

> - `Open3D-ML`
> - `O3D and Its Historical Context` (O3D 及其历史背景)
> - `3D Object Classification` (3D 物体分类)
> - `LiDAR Applications` (LiDAR 应用)
> - `Point Cloud Library Usage` (Point Cloud Library 使用)

- **Open3D-ML 展现出潜力**：一位成员提到 [Open3D-ML](https://github.com/isl-org/Open3D-ML) 是 Open3D 的一个很有前景的扩展，旨在处理 3D Machine Learning 任务。
  
  - 这一新的集成因其在增强 3D 应用方面的潜力而引起了关注。
- **O3D 在 3D 框架中的传承**：另一位成员对 **O3D** 的长寿表示惊讶，回想起它大约与 AlexNet 同时发布。
  
  - 他们反思道，尽管设计稳健，但 Open3D 并没有获得像 **WebGL** 那样的关注度。
- **3D 物体分类的创新方法**：有人建议在 Blender 中使用 Python 脚本从多个角度生成 3D 物体的图像，用于分类目的。
  
  - 这种方法有助于创建一个能够跨不同视角解释和验证分类的模型。
- **使用 Open3D 的 LiDAR 应用**：一位成员在研究一家利用 **LiDAR** 进行森林分析的公司时发现了 Open3D。
  
  - 他们之前的经验主要涉及使用 [Point Cloud Library](https://pointclouds.org/) 处理 3D 物体。

**提到的链接**：

- [GitHub - isl-org/Open3D-ML: An extension of Open3D to address 3D Machine Learning tasks](https://github.com/isl-org/Open3D-ML)：Open3D 的扩展，用于解决 3D Machine Learning 任务 - isl-org/Open3D-ML
- [The o3d Bible by Kara Rawson](https://www.scribd.com/document/63892020/The-o3d-Bible-by-Kara-Rawson)：该文档提供了 Google O3D API 库的摘要。包括介绍、安装说明、系统要求、支持的图形硬件以及程序概述...

---

### **HuggingFace ▷ #**[**diffusion-discussions**](https://discord.com/channels/879548962464493619/1009713274113245215/1306493978878803978) (1 messages):

> - `Stable Diffusion 1.5`
> - `CPU performance optimization` (CPU 性能优化)

- **选择 Stable Diffusion 1.5 进行 CPU 优化**：一位用户表示他们打算使用 **Stable Diffusion 1.5**，称其为实现高效性能的最轻量版本。
  
  - 他们强调模型需要 **在 CPU 上快速运行**，表明了对潜在资源优化的偏好。
- **CPU 上的效率考量**：强调了确保模型能够 **在 CPU 上快速运行** 的必要性，因为用户正在为他们的配置寻求优化方案。
  
  - 这反映了将模型适配为在更易获得的硬件配置上高效运行的更广泛趋势。

---

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1306356879710224396) (54 条消息🔥):

> - `In-line LaTeX rendering in LM Studio` (LM Studio 中的行内 LaTeX 渲染)
> - `Sideloading llama.cpp` (侧载 llama.cpp)
> - `Running large models on limited RAM` (在有限的 RAM 上运行大型模型)
> - `Autogen and API issues` (Autogen 和 API 问题)
> - `Nexus team performance` (Nexus 团队表现)

- **In-line LaTeX rendering in LM Studio**: 用户讨论了 LaTeX 渲染的挑战，特别是 Qwen2.5-Math-72B-Instruct 模型，当公式被包裹在美元符号中时会产生意外结果。
  
  - 一位用户建议创建一个带有明确指令的 system prompt，以提高 LaTeX 解析的一致性。
- **Sideloading llama.cpp features**: 用户请求一种简便的方法将 llama.cpp 的功能侧载到 LM Studio 中，并对当前设置的局限性表示沮丧。
  
  - 对话强调了在未来更新中实现这一功能的持续努力，用户们渴望更易用的解决方案。
- **Running large models on limited RAM**: 个人推测是否可以通过虚拟内存或基于磁盘的解决方案运行大于可用 RAM 的模型，尽管性能可能会受到影响。
  
  - 一位用户否定了使用慢速存储介质的想法，强调 RAM 对模型性能至关重要。
- **Autogen and API issues**: 一位用户在运行 LM Studio 本地服务器时遇到问题，被建议查看教程或提供详细的错误报告以获得更好的帮助。
  
  - 在更新和更改配置后，该用户解决了最初的问题，但表示需要分享类似问题的经验。
- **Nexus team performance**: 一位用户表达了对 Nexus 团队能力的钦佩，认为他们的工作对社区产生了重大影响。
  
  - 对 Nexus 团队贡献的热情反映了参与讨论的用户们的广泛支持和赞赏。

 

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1306355211945644142) (246 条消息🔥🔥):

> - `GPU performance with large models` (大型模型的 GPU 性能)
> - `CPUs vs GPUs for LLM workloads` (LLM 工作负载中的 CPU vs GPU)
> - `M4 Max benchmark comparison` (M4 Max 基准测试对比)
> - `Model offloading to different hardware` (模型卸载到不同硬件)
> - `Integrating AI in SaaS applications` (在 SaaS 应用中集成 AI)

- **GPU performance struggles with large models**: 用户注意到在混合 GPU 设置上运行像 Nemotron 70b 这样的大型模型时性能各异，结果从 1.97 到 14.0 tok/s 不等。使用不同配置进行的测试表明，内存可用性和 CPU 瓶颈显著影响吞吐量。
  
  - 尽管期望很高，基准测试表明仅使用 CPU 会导致较低的 tok/s 速率，促使一些用户考虑升级 GPU 配置。
- **Challenges of CPU vs GPU in LLM operations**: 共识是 CPU 通常无法跟上现代 LLM 的步伐，即使具有高内存容量，其 tok/s 速率也远低于 GPU 加速。几位成员分享了关于内存带宽和 GPU offloading 如何影响整体模型性能的见解。
  
  - 用户表示需要更好的资源处理方式，强调在使用大型模型时 CPU 利用率往往较低，并呼吁进行更有效的调整。
- **M4 Max performance potential discussed**: 鉴于 M4 Max 配备了 128GB RAM，用户们渴望测试其在 LLM 上的性能，并考虑其与专用 GPU 设置的对比。讨论显示出实验基准测试的意愿，特别是许多用户刚接触 LLM 并正在探索自托管选项。
  
  - 成员们表示有兴趣分享结果和基准测试，以作为购买决策的参考，这揭示了社区在 AI 特定性能评估方面的空白。
- **Offloading layers for improved performance**: 讨论了将模型层卸载到 Apple 设备上的 Neural Engine (NE) 的潜力，并指出目前大多数运行时仅使用 GPU/CPU 的局限性。有人推测启用直接 NE 使用如何能增强重型 LLM 任务的性能。
  
  - 用户担心目前的实现很少能发挥出 NE 的全部能力，尽管其理论性能非常强大。
- **Integrating AI into SaaS applications**: 一位成员分享了将 AI 功能集成到 SaaS 应用程序中的计划，展示了利用 LM Studio API 实现此目的的热情。强调了 AI 辅助开发的潜力，表明了在应用领域的持续探索。
  
  - 参与者讨论了利用各种 AI 工具增强软件功能的可能性，反映出在实际场景中采用 AI 技术的渴望。

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1306352499346636850) (217 messages🔥🔥):

> - `Unsloth AI Training Efficiency` (Unsloth AI 训练效率)
> - `Understanding LLMs and Math` (理解 LLM 与数学)
> - `Editing Code with AI Tools` (使用 AI 工具编辑代码)
> - `GPU Programming and Triton` (GPU 编程与 Triton)
> - `Educational Chatbot Data Chunking` (教育聊天机器人的数据分块)

- **Unsloth AI Training Efficiency**: 成员们讨论了 Unsloth 平台的内存效率，**theyruinedelise** 肯定它是目前可用的内存效率最高的训练服务。
  
  - 还提到了 Unsloth 即将实现 CPO 训练器，这将进一步提高其效率。
- **Understanding LLMs and Math**: 参与者强调了理解线性代数和微积分对于掌握 LLM 概念的重要性，**_niten** 表示这些从根本上表达了 LLM 的机制。
  
  - 许多人建议复习涵盖机器学习所需基础数学的课程和资源，例如链式法则和矩阵性质。
- **Editing Code with AI Tools**: **anubis7645** 分享了他们正在构建一个用于编辑大型 React 文件的实用程序，同时研究像 Cursor 这样的工具如何在模型 Token 限制下无缝生成编辑。
  
  - **lee0099** 解释了推测性编辑（speculative edits）的概念，这种编辑方式可以实现快速应用，并暗示了它与编码实践的关系。
- **GPU Programming and Triton**: 讨论涉及了学习 Triton 和 CUDA 进行 GPU 编程的相关性，**eduuu** 表示在模型不断演进的过程中，这些技术提供了未来的工程机会。
  
  - **tenderrizedd** 询问了 Triton 在推理（inference）中的应用，强调了对提高模型效率的持续关注。
- **Educational Chatbot Data Chunking**: **arena1040** 寻求关于教育类聊天机器人数据集分块（chunking）的建议，特别是处理波斯语文本和嵌入的 MathType 公式。
  
  - **mollel.** 建议使用 RAG 方法，同时直接从 OpenAI API 生成数据集，以获得更具教学意义的材料。

**提到的链接**:

- [Welcome | Unsloth Documentation](https://docs.unsloth.ai/): Unsloth 新手？从这里开始！
- [How Cursor built Fast Apply using the Speculative Decoding API](https://fireworks.ai/blog/cursor): Cursor 作为一个 AI 原生 IDE，利用 Fireworks 的推理栈增强了其 Instant Apply、Smart Rewrites 和 Cursor Prediction 等功能。该博文介绍了 Speculative Decoding API...

---

### **Unsloth AI (Daniel Han) ▷ #**[**off-topic**](https://discord.com/channels/1179035537009545276/1179039861576056922/1306448873736110081) (11 messages🔥):

> - `Brunch Choices` (早午餐选择)
> - `Diet Adjustments` (饮食调整)
> - `Animal-derived Products` (动物源产品)
> - `Nuts and Seeds Discussion` (坚果与种子讨论)

- **Brunch Menu Highlights**: 一位成员分享了他们的早午餐，包括**鸡肉**、**沙拉（无酱料）**、**鸡蛋**、**牛奶**和**半个牛油果**。
  
  - 他们对这顿饭表示满意，称“目前感觉很好”。
- **Body Adjustments to Diet**: 讨论了减少碳水化合物摄入时身体的调整期，一位成员指出这可能需要大约**一周**时间。
  
  - 成员们对与碳水化合物相关的疲劳感表示担忧，从而促使了饮食改变。
- **Animal-derived Products Under Scrutiny**: 另一位参与者评论了早午餐中大量的**动物源产品**，如**鸡肉**、**鸡蛋**和**牛奶**。
  
  - 这引发了一个轻松的询问，关于餐食中为何缺少**坚果和种子**。
- **Nuts and Seeds Preferences**: 在关于坚果和种子的对话中，一位成员幽默地表示：“我什么都不吃”。
  
  - 另一位成员开玩笑地称自己为“动物”，暗示他们不食用坚果或种子。

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1306382968993878168) (31 messages🔥):

> - `Train on responses only` 函数
> - 微调中的 LoRA 参数
> - 数据集质量问题
> - 法语聊天机器人模型选择
> - 在没有未量化模型的情况下使用 LoftQ

- **澄清 Train on Responses Only 函数**：关于 `train_on_responses_only` 函数的讨论显示，它在按顺序预测助手回复时会掩盖用户输入，这引发了关于模型训练效率的问题。
  
  - 有人对将较长的聊天历史拆分为样本的做法表示担忧，建议专注于最后一条助手消息进行训练。
- **微调中的 LoRA 参数**：指出使用较小的 rank 和 adaptation 值有助于在不扭曲模型质量的情况下改进数据集训练，尤其是在某些特定条件下。
  
  - 建议用户深入了解 rank (r) 和 adaptation (a) 因子，并指出高质量的数据集对于有效训练至关重要。
- **优化数据集质量**：成员们讨论了数据集质量对模型性能的影响，强调平庸的数据集可能会阻碍训练过程中 loss 的降低。
  
  - 建议减小数据集规模或提高其质量，以获得更好的训练效果。
- **为法语聊天机器人选择基础模型**：对于创建法语聊天机器人，推荐使用 Mistral 模型作为合适的基础，并强调了选择合适训练参数的重要性。
  
  - 指出在训练中使用低 rank 和 alpha 值有助于在微调期间保持基础模型的质量。
- **在不加载未量化模型的情况下使用 LoftQ**：提出了关于是否可以直接使用 LoftQ 而不加载未量化模型的问题，特别是在 T4 等 VRAM 受限的环境中。
  
  - 建议调整 LoRA 的 target modules，仅包含 linear 和 embedding 层，以增强微调期间的补丁效果。

**提到的链接**：

- [Google Colab](https://colab.research.google.com/drive/1AZghoNBQaMDgWJpi4RbffGM1h6raLUj9?usp=sharing#scrollTo=yqxqAZ7KJ4oL>): 未找到描述
- [Unsloth Documentation](https://docs.unsloth.ai/basics/lora-parameters-encyclopedia.): 未找到描述

---

### **Unsloth AI (Daniel Han) ▷ #**[**community-collaboration**](https://discord.com/channels/1179035537009545276/1180144489214509097/1306618873969119253) (2 messages):

> - `Harmony` 项目
> - 开源问卷协调
> - LLM 匹配竞赛
> - 自然语言处理（NLP）增强

- **Harmony 项目寻求合作**：一位成员宣布了 **Harmony** 项目，这是多个机构合作开发的基于 **AI LLM** 的数据协调工具。他们为有兴趣贡献的人提供了 [Discord server](https://discord.gg/harmonydata) 链接。
  
  - 该项目目前总部位于 **UCL**，正在积极寻找志愿者协助。
- **探索 Harmonise 问卷项目**：**Harmony** 工具促进了问卷项目和元数据的回顾性协调，有利于跨研究的项目比较。其功能的详细信息可以在其 [官网](https://harmonydata.ac.uk/) 上找到。
  
  - 该工具解决了不同问卷版本和翻译的兼容性问题，使其适用于各种研究背景。
- **提升 LLM 算法的竞赛**：Harmony 正在举办一场竞赛，旨在改进其 **LLM 匹配算法**，并为参与者提供奖品。感兴趣的人可以在其 [竞赛页面](https://harmonydata.ac.uk/doxa/) 找到更多信息。
  
  - 目标是优化 Harmony 准确评估句子相似性的能力，纠正其 [博客文章](https://harmonydata.ac.uk/nlp-semantic-text-matching/measuring-the-performance-of-nlp-algorithms/) 中强调的当前与人类评估者之间的偏差。

**提到的链接**：

- [Harmony | A global platform for contextual data harmonisation](https://harmonydata.ac.uk/): 上下文数据协调的全球平台
- [Competition to train a Large Language Model for Harmony on DOXA AI | Harmony](https://harmonydata.ac.uk/doxa/): 上下文数据协调的全球平台

---

### **OpenRouter (Alex Atallah) ▷ #**[**announcements**](https://discord.com/channels/1091220969173028894/1092729520181739581/1306395578606813195) (2 messages):

> - `UnslopNemo 12B v4`
> - `SorcererLM`
> - `Inferor 12B`
> - `Model Status Updates`
> - `UI Improvements`

- **推出用于冒险写作的 UnslopNemo 12B v4**：最新模型 [UnslopNemo 12B](https://openrouter.ai/thedrummer/unslopnemo-12b) 已发布，专为冒险写作和角色扮演（Role-play）场景设计。
  
  - 通过此链接可获得 24 小时免费试用版本：[UnslopNemo 12B Free](https://openrouter.ai/thedrummer/unslopnemo-12b:free)。
- **使用 SorcererLM 进行高级角色扮演**：全新的 [SorcererLM](https://openrouter.ai/raifle/sorcererlm-8x22b) 基于 WizardLM-2-8x22B 进行微调，旨在提供增强的故事讲述体验。
  
  - 加入我们的 Discord 以申请访问权限或进行进一步咨询。
- **Inferor 12B 是终极角色扮演模型**：[Inferor 12B](https://openrouter.ai/infermatic/mn-inferor-12b) 结合了顶级的角色扮演模型，但建议用户设置合理的输出限制以避免生成过长文本。
  
  - 请通过我们的 Discord 申请该模型的访问权限。
- **服务短暂宕机影响运行**：由于环境同步问题，服务出现了约 1.5 分钟的短暂宕机，目前已解决。
  
  - 更多更新和状态信息可随时在 [OpenRouter Status](https://status.openrouter.ai/) 查看。
- **通过 UI 改进提升用户体验**：最近的更新包括在模型页面显示最大上下文长度（max context length），并引入了使用 cmd + K 的文档搜索功能。
  
  - 新的表格列表视图也提供了更好的模型可视化效果，使查找信息更加便捷。

**提到的链接**：

- [OpenRouter Status](https://status.openrouter.ai/.): OpenRouter 事件历史
- [OpenRouter](https://openrouter.ai/thedrummer/unslopnemo-12b)): LLM 路由与市场
- [OpenRouter](https://openrouter.ai/thedrummer/unslopnemo-12b:free)): LLM 路由与市场
- [OpenRouter](https://openrouter.ai/raifle/sorcererlm-8x22b)): LLM 路由与市场
- [OpenRouter](https://openrouter.ai/infermatic/mn-inferor-12b)): LLM 路由与市场

---

### **OpenRouter (Alex Atallah) ▷ #**[**app-showcase**](https://discord.com/channels/1091220969173028894/1092850552192368710/1306378508125212723) (5 messages):

> - `GitHub open source project policies`
> - `WordPress Chatbot Plugin Launch`
> - `Companion Discord Bot Features`

- **关于 GitHub 开源发布规则的咨询**：一位用户询问了 *发布 GitHub 开源项目的规则和政策*。
  
  - 另一位成员回答称准则 *非常宽松*，并表示只要你以任何方式使用了 OpenRouter，就应该是可以接受的。
- **WordPress 聊天机器人插件发布**：一位用户宣布他们的 *WordPress 聊天机器人插件已上线*，具有自定义短代码（shortcodes）和动态标签功能。
  
  - 他们指出该机器人可以担任多种角色，如支持机器人或销售机器人，并确认 *支持 OpenRouter*。
- **Companion：增强 Discord 安全与互动**：一位成员介绍了 *Companion*，这是一个旨在个性化 Discord 角色（personas）同时通过自动化审核增强安全性的程序。
  
  - 它具有 **身份冒充检测**、**年龄违规检测** 功能，并允许动态调整消息频率以提高服务器参与度。

**提到的链接**：

- [no title found](https://wpaimuse.com/chatbot): 无描述
- [Home](https://github.com/rapmd73/Companion/wiki): 一个 AI 驱动的 Discord 机器人，将趣味对话与智能审核工具相结合，为您的服务器增添魅力与秩序。 - rapmd73/Companion

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1306352728452370492) (201 条消息🔥🔥):

> - `Unslopnemo 12b`
> - `DeepSeek context limitations` (DeepSeek 上下文限制)
> - `Gemini API updates` (Gemini API 更新)
> - `OpenRouter API Issues` (OpenRouter API 问题)
> - `AI Studio generateSpeech API`

- **Unslopnemo 12b 搜索问题**：Unslopnemo 12b 可以被搜索到，但未出现在模型页面的最新模型排序功能中。
  
  - 这一差异引发了关于排序机制是否正常运行的简短讨论。
- **DeepSeek 的上下文错误**：用户报告称，尽管文档声称具有 128k 的上下文容量，但 DeepSeek 的 API 在输入超过 47k tokens 时会失败。
  
  - 经过进一步调查，确定实际的最大上下文长度为 65k tokens。
- **Gemini API 和模型可用性**：讨论提到，虽然 Gemini 有可用的实验性模型，但目前还无法通过 OpenRouter API 访问。
  
  - 用户指出，特定模型 `gemini-exp-1114` 目前仅限于 AI Studio。
- **OpenRouter API 稳定性**：据报告 OpenRouter 服务出现了短暂宕机，导致部分用户在使用各种模型时遇到问题。
  
  - 情况已得到澄清，确认服务已恢复正常，Claude 等模型已正常运行。
- **AI Studio 新功能**：AI Studio 正在推出一个新的 `generateSpeech` API 端点，旨在根据输入的文本稿由指定模型生成语音。
  
  - 该功能旨在增强现有模型从文本生成音频输出的能力。

**提到的链接**：

- [Quick Start | OpenRouter](https://openrouter.ai/docs/quick-start)：开始使用 OpenRouter 进行构建
- [Chatroom | OpenRouter](https://openrouter.ai/chat)：LLM Chatroom 是一个多模型聊天界面。添加模型并开始聊天！Chatroom 将数据本地存储在您的浏览器中。
- [Elevated errors on the API](https://status.anthropic.com/incidents/7svmbgb2b28x)：未找到描述
- [Models | OpenRouter](https://openrouter.ai/models)：在 OpenRouter 上浏览模型
- [no title found](https://ai.google.dev/gemini-api/docs/models/experimental-models)：未找到描述
- [2024-11-14-214227 hosted at ImgBB](https://ibb.co/PYJ9z5w)：托管在 ImgBB 的图片 2024-11-14-214227
- [OpenRouter](https://openrouter.ai/docs/quick)：LLM 路由与市场
- [Anthropic Status](https://status.anthropic.com/)：未找到描述
- [Models | OpenRouter](https://openrouter.ai/models?fmt=table)：在 OpenRouter 上浏览模型
- [OpenRouter](https://openrouter.ai/api/v1)：LLM 路由与市场
- [Provider Routing | OpenRouter](https://openrouter.ai/docs/provider-routing)：跨多个提供商路由请求
- [OpenRouter Status](https://status.openrouter.ai/)：OpenRouter 故障历史

---

### **OpenRouter (Alex Atallah) ▷ #**[**beta-feedback**](https://discord.com/channels/1091220969173028894/1277894087755829278/1306363555657814127) (7 条消息):

> - `Custom Provider Keys` (自定义提供商密钥)
> - `Customer Integration Access` (客户集成访问)

- **关于 Custom Provider Keys 的多次请求**：几位成员请求访问 **Custom Provider Keys**，并表示了对该功能的兴趣和需求。
  
  - *一位成员明确表示*：“我想申请 Custom Provider Keys。”
- **关于客户集成访问权限的咨询**：一位成员寻求关于如何获得 **customer integration** 访问权限的说明。
  
  - 他们询问：“我们如何获得客户集成的访问权限？”，表达了对利用相关功能的兴趣。

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1306397713146970205) (43 条消息🔥):

> - `职业转型挑战`
> - `下载 The Pile 数据集`
> - `IBM 的 Granite 与开源`
> - `Transformer 架构演进`
> - `AI 硬件发展`

- **科技行业的职业转型挑战**：一位成员表达了对困在以产品为导向的 ML 岗位的沮丧，因为转岗有 12 个月的任职期限要求，同时他正在探索 PyTorch 的机会。
  
  - 他们指出，讨论潜在的变动涉及处理内部流程以及降薪考虑。
- **The Pile 数据集的可用性**：有人询问出于历史原因下载 The Pile 的事宜，随后有人建议使用 [Hugging Face](https://huggingface.co/datasets/monology/pile-uncopyrighted) 上提供的无版权版本。
  
  - 该数据集已清除受版权保护的内容，允许在遵守版权法的前提下用于训练 LLM。
- **对 IBM Granite 作为开源 AI 的质疑**：围绕 IBM 的 Granite 展开了讨论，质疑其作为“开源 AI”的分类，因为其缺乏共享的代码或训练涉及的数据集细节。
  
  - 成员们辩论了现有文档是否允许在已披露内容之外重新构建 Granite。
- **演进中的 Transformer 架构**：对话强调了 Transformer 的持久生命力，提到了 Decoder-only 架构和 Mixture of Experts 等进展，但仍对其硬件适应性提出质疑。
  
  - 成员们认为需要演进硬件以匹配这些架构，并认识到目前正在做出的权衡。
- **AI 训练的硬件发展**：分享了关于新硬件进展提高 Transformer 推理效率的见解，特别强调了 Blackwell 相比 Hopper 的改进。
  
  - 讨论指出，内存带宽和 VRAM 对于大规模 AI 模型的有效实现至关重要。

**提到的链接**：

- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)：我们研究了 Transformer 模型的生成式推理效率问题，在最具挑战性的场景之一：具有严格延迟目标和长序列长度的大型深度模型。更好的 ...
- [monology/pile-uncopyrighted · Datasets at Hugging Face](https://huggingface.co/datasets/monology/pile-uncopyrighted)：未找到描述
- [granite-3.0-language-models/paper.pdf at main · ibm-granite/granite-3.0-language-models](https://github.com/ibm-granite/granite-3.0-language-models/blob/main/paper.pdf)：通过在 GitHub 上创建账号，为 ibm-granite/granite-3.0-language-models 的开发做出贡献。

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1306352343884890245) (123 条消息🔥🔥):

> - `Vision Language Action 模型基准测试`
> - `关于 Scaling Laws 的讨论`
> - `优化中的 Shampoo 和 Muon 算法`
> - `Int8 训练的影响`
> - `合成任务的实用性`

- **Vision Language Action 模型的新研究**：Manifold、Georgia Tech、MIT 和 Metarch AI 合作发布了一篇题为 'Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks' 的论文，在 20 个真实世界任务中评估了 GPT4o 等 VLM。
  
  - 相关链接已分享至 [Twitter 亮点](https://x.com/HarshSikka/status/1856739777208574151) 和 [GitHub 仓库](https://github.com/ManifoldRG/MultiNet/tree/main) 以获取更多详细信息。
- **围绕 Scaling Laws 的争议**：传闻称 LLM 最近的 Scaling 努力可能无法产生新能力，引发了关于 Scaling Laws 可靠性的讨论。
  
  - 参与者指出边际收益递减显而易见；然而，该说法目前主要基于推测，而非来自严谨研究的实证证据。
- **对 Shampoo 和 Muon 优化技术的见解**：提出了关于各种优化算法（包括 Shampoo 和 Muon）有效性的问题，特别是在使用 Fisher Information Matrix 估计 Hessian 的背景下。
  
  - 讨论围绕这些算法的假设是否成立展开，并参考了 KFAC 与 Shampoo 对比的相关论文。
- **Int8 训练的挑战**：在关于性能的分支讨论中，参与者探讨了使用 int8 与 uint8 训练的影响，好奇 Scaling 和优化技术如何处理低动态范围。
  
  - 共识强调，在转向这些低精度格式时，采用全面的设计方法至关重要。
- **合成任务的相关性**：关于合成任务在评估 Transformer 模型中的实用性引发了辩论，一些人声称它们不能反映真实世界的性能水平。
  
  - 参与者对合成任务的结果表示怀疑，认为许多展示 Transformer 局限性的论文在有效 AI 部署方面的适用性存疑。

**提到的链接**：

- [Old Optimizer, New Norm: An Anthology](https://arxiv.org/abs/2409.20325)：深度学习优化器通常通过凸理论和近似二阶理论的结合来驱动。我们选择了三种此类方法——Adam、Shampoo 和 Prodigy——并论证每种方法都可以……
- [How to represent part-whole hierarchies in a neural network](https://arxiv.org/abs/2102.12627)：本文未描述一个工作系统。相反，它提出了一个关于表示的单一想法，允许将几个不同小组取得的进展合并到一个虚构的系统中……
- [Tweet from BlinkDL (@BlinkDL_AI)](https://x.com/BlinkDL_AI/status/1857006248052359244)：新的 RWKV CoT 演示：4M 参数解决 15-puzzle 🔥 https://github.com/Jellyfish042/RWKV-15Puzzle #RWKV #RNN 引用 BlinkDL (@BlinkDL_AI) RWKV-Sudoku 极端 CoT 代码和模型：https://github.com/J...
- [Modular Duality in Deep Learning](https://arxiv.org/abs/2410.21265)：优化理论中的一个旧观点认为，由于梯度是一个对偶向量，在未先映射到权重所在的原始空间之前，不能从权重中减去它。我们……
- [ZipNN: Lossless Compression for AI Models](https://arxiv.org/abs/2411.05239)：随着模型规模和部署规模的增长，其庞大的体积给基础设施带来了负担，需要更多的网络和存储来容纳。虽然存在大量的模型……
- [Tweet from harsh (@HarshSikka)](https://x.com/HarshSikka/status/1856739777208574151)：很高兴分享我们的新论文 "Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks"。我们评估了 VLM 和 VLA 模型在 20 个不同真实世界中控制机器人的表现……
- [Euclidean plane isometry - Wikipedia](https://en.wikipedia.org/wiki/Euclidean_plane_isometry)：未找到描述
- [GitHub - NVIDIA/ngpt: Normalized Transformer (nGPT)](https://github.com/NVIDIA/ngpt)：Normalized Transformer (nGPT)。通过创建 GitHub 账号为 NVIDIA/ngpt 的开发做出贡献。
- [RWKV-15Puzzle/puzzle15_vocab.txt at main · Jellyfish042/RWKV-15Puzzle](https://github.com/Jellyfish042/RWKV-15Puzzle/blob/main/puzzle15_vocab.txt)：通过创建 GitHub 账号为 Jellyfish042/RWKV-15Puzzle 的开发做出贡献。
- [RWKV-15Puzzle/generate_data.py at main · Jellyfish042/RWKV-15Puzzle](https://github.com/Jellyfish042/RWKV-15Puzzle/blob/main/generate_data.py)：通过创建 GitHub 账号为 Jellyfish042/RWKV-15Puzzle 的开发做出贡献。

---

### **Eleuther ▷ #**[**interpretability-general**](https://discord.com/channels/729741769192767510/1052314805576400977/1306718271864832032) (4 messages):

> - `Pythia model suite`
> - `Mixture of Experts (MoE)`
> - `OLMo and OLMOE comparison`
> - `Interpolation-focused training`
> - `Hyperparameter modernization`

- **关于 Pythia 结合 MoE 的辩论**：一位成员询问了为 **Pythia 模型套件** 开发 **Mixture of Experts (MoE) 版本** 的可能性，并讨论是应该完全复制现有的训练设置，还是采用现代化的超参数（例如使用 **SwiGLU**）。
  
  - 这一努力旨在确定在这种背景下实现 MoE 可以解决哪些具体问题。
- **OLMo 和 OLMOE 与 Pythia 的契合度**：一位成员提出 **OLMo** 和 **OLMOE** 已经符合所讨论的目标，并指出尽管模型大小与 **Pythia** 不同，但它们采用了现代架构选择。
  
  - 他们指出主要的区别在于 OLMo 缺乏 Pythia 那样的多种尺寸版本，但其当代设计是相似的。
- **MoE 训练与 Pythia 侧重点的对比**：讨论强调，虽然 **OLMo** 探索了 **MoE 搜索空间**，但除了领域专业化实验外，它缺乏 **Pythia** 所采用的大规模侧重插值的训练（Interpolation-focused training）。
  
  - Pythia 在不同模型规模之间的一致性以及特定的训练数据顺序被强调为重要因素。
- **影响 MoE 性能的因素**：一位成员承认，新版 OLMo 发布中采用的 **数据顺序** 和持续训练策略存在差异，这影响了性能对比。
  
  - 这些因素有助于理解为什么 OLMo 可能无法完全匹配 Pythia 侧重插值的目标。

 

**提到的链接**：[Nora Belrose (@norabelrose) 的推文](https://x.com/norabelrose/status/1857159435686384096)：如果存在 Pythia 模型套件的 Mixture of Experts 版本，你想用它回答什么样的问题？我们是否应该尝试精确复制 Pythia 的训练设置，但使用 M...

 

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1306533177585831946) (7 messages):

> - `Eval prompt modifications`
> - `Official parser modifications`
> - `Mmlu standardization`
> - `MMMU evaluation details`

- **修改评估提示词：并非不可接受，但需谨慎**：一位成员询问是否可以在官方评估提示词中添加类似 "Final answer:" 的短语以辅助解析。
  
  - 另一位成员指出这并不一定不可接受，但最佳实践是坚持使用相同的提示词以进行公平比较，除非有充分理由。
- **讨论依赖于任务的解析器修改**：同一位成员询问了修改官方解析器的可接受性，并指出了 lmms-eval 和 MMMU 解析器之间的差异。
  
  - 另一位成员回答说这非常依赖于任务，并提到一些任务有标准化的实现，但多模态任务的一致性较差。
- **MMMU 评估中缺乏细节**：一位成员指出，大多数模型发布中关于 MMMU 的详细评估信息较少。
  
  - 这突显了多模态任务在透明度方面的差距，可能会影响对所使用的评估设置的理解。

**提到的链接**：

- [lmms-eval/lmms_eval/tasks/mmmu/utils.py at bcbdc493d729e830f4775d1a1af4c1d7d8e449f2 · EvolvingLMMs-Lab/lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/bcbdc493d729e830f4775d1a1af4c1d7d8e449f2/lmms_eval/tasks/mmmu/utils.py#L293)：通过 lmms-eval 加速大型多模态模型 (LMMs) 的开发 - EvolvingLMMs-Lab/lmms-eval
- [MMMU/eval/eval_utils.py at 51ce7f3e829c16bb44bc5445782686b4c3508794 · MMMU-Benchmark/MMMU](https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L29)：此仓库包含论文 "MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI" 的评估代码 - MMMU-Benchmark/MMMU

---

### **aider (Paul Gauthier) ▷ #**[**announcements**](https://discord.com/channels/1131200896827654144/1133060115264712836/1306372686519468102) (1 messages):

> - `Aider v0.63.0`
> - `Qwen 2.5 Coder 32B Support`
> - `Web Command Improvement`
> - `Prompting Enhancements`
> - `Bug Fixes`

- **Aider v0.63.0 现已发布！**：新版本 **Aider v0.63.0** 包含了对 **Qwen 2.5 Coder 32B** 的支持，并引入了多项性能改进。
  
  - 此外，**Aider 贡献了本次更新 55%** 的代码。
- **Web 命令获得全新更新**：`/web` 命令现在仅将页面添加到聊天中，不再像以前那样直接触发 **LLM** 响应。
  
  - 这一变化通过简化网页集成加速了用户交互。
- **改进的语言偏好处理**：用户现在可以体验到增强的 **首选聊天语言** 选择提示，使交互更加个性化。
  
  - 此更新旨在通过促进更顺畅的对话来提高用户参与度。
- **LiteLLM 异常处理升级**：**LiteLLM** 的异常处理得到了显著改进，减少了对用户体验的干扰。
  
  - 此修复有助于在整个 Bot 功能运行中提供更顺畅的体验。
- **Bug 修复：实施了多项修复**：推出了多项 Bug 修复，包括解决了缓存统计中的 **Token 重复计数** 问题以及 **LLM** 创建新文件时的问题。
  
  - 这些微小的修复增强了 Aider 的整体可靠性和性能。

 

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1306348073865445517) (123 messages🔥🔥):

> - `Aider enhancements`
> - `Qwen 2.5 Coder performance`
> - `Gemini experimental models`
> - `OpenRouter compatibility`
> - `CLI scripting with Aider`

- **Aider 生态系统和文档工作**：用户正寻求改进 Aider 生态系统的文档，表示有兴趣使用 [Ravel](https://ravel.acenturyandabit.xyz/) 等平台使细节更具可搜索性和连贯性。
  
  - 讨论强调了随着 Aider 功能快速增长（通常超过现有文档更新速度），对更清晰指南的需求。
- **Qwen 2.5 Coder 的使用体验**：关于通过 OpenRouter 使用 Qwen 2.5 Coder 的性能评价不一，一些用户报告其表现不如基准测试统计数据。
  
  - 提议使用模型 `aider --model openrouter/qwen/qwen-2.5-coder-32b-instruct` 作为可行方案，尽管分享的结果褒贬不一。
- **新兴的 Gemini 实验性模型**：引入了新的 Gemini 实验性模型，引发了对其在挑战性 Prompt 上的有效性和通用可用性的好奇。
  
  - 一些用户报告尝试了这些模型，但面临访问问题，这表明 Google Cloud 上的权限可能限制了可用性。
- **对 Aider CLI 脚本编写的兴趣**：成员们正在探索 Aider 中的脚本编写功能以自动化重复任务，强调了使用命令行选项简化工作流的潜力。
  
  - 提供的文档链接强调了以编程方式对多个文件应用编辑的能力，展示了 Aider 的多功能性。
- **Qwen 与不同编辑器的组合**：用户讨论了 Qwen 2.5 Coder 与各种编辑器的兼容性，指出虽然它运行良好，但在与 Haiku 等作为编辑器组合时性能可能会下降。
  
  - 普遍共识表明体验各异，某些组合能产生有效结果，而其他组合则表现不佳。

**提及的链接**：

- [Scripting aider](https://aider.chat/docs/scripting.html)：你可以通过命令行或 Python 编写 Aider 脚本。
- [OpenRouter](https://openrouter.ai/qwen/qwen-2.5-coder-32b-instruct)：LLM 路由和市场。
- [Qwen/Qwen2.5-Coder-32B-Instruct - Demo - DeepInfra](https://deepinfra.com/Qwen/Qwen2.5-Coder-32B-Instruct)：Qwen2.5-Coder 是最新的代码专用 Qwen 大语言模型系列（前身为 CodeQwen）。它在代码生成、代码推理和代码修复方面有显著改进。更多...
- [xingyaoww/Qwen2.5-Coder-32B-Instruct-AWQ-128k · Hugging Face](https://huggingface.co/xingyaoww/Qwen2.5-Coder-32B-Instruct-AWQ-128k)：未找到描述
- [no title found](https://ai.google.dev/gemini-api/docs/models/experimental-models)：未找到描述
- [unsloth/Qwen2.5-Coder-7B-Instruct-128K-GGUF · Hugging Face](https://huggingface.co/unsloth/Qwen2.5-Coder-7B-Instruct-128K-GGUF)：未找到描述
- [Ravel](https://ravel.acenturyandabit.xyz/)：未找到描述
- [GitHub - nekowasabi/aider.vim: Helper aider with neovim](https://github.com/nekowasabi/aider.vim)：Neovim 的 Aider 辅助工具。在 GitHub 上为 nekowasabi/aider.vim 的开发做出贡献。

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1306374158321717289) (29 条消息🔥):

> - `Installing Aider in Termux`
> - `Triggering Rust Analyzer in VSCode`
> - `Using Aider with git diff`
> - `Aider modes comparison`
> - `Aider usage tips`

- **在 Termux 中安装 Aider**：一位成员询问是否有人尝试在 Termux 或其他移动终端中安装 Aider，并指出只要能在 Python 环境中运行，Aider 是不依赖特定 IDE 的。
  
  - 另一位成员确认了 Aider 的灵活性，强调其主要通过 CLI 和 git 进行交互。
- **在 VSCode 中触发 Rust Analyzer**：一位用户询问在 Aider 运行结束后触发 VSCode 中 Rust Analyzer 的最简单方法，并考虑将文件系统监听（filesystem watching）作为解决方案。
  
  - 一位成员建议运行 `cargo check`（根据需要决定是否配合 `cd` 命令），这通常能有效地解决问题。
- **结合 git diff 使用 Aider**：一位成员想知道 Aider 是否能读取文件编辑（diff）并据此规划更改，随后有人分享了必要的命令。
  
  - 另一位成员建议使用 `/run git diff ...`，该命令提供了一个选项，可以将输出添加到聊天中以便进一步规划。
- **Aider 模式对比**：一位新用户对在 Aider 的 architect mode 和其他模式之间切换表示困惑，并提到了潜在的高 token 消耗。
  
  - 一位资深用户建议开始时先不使用 architect mode，而是选择 gpt-4o 或 Sonnet，以简化使用。
- **Aider 使用技巧**：一位用户分享了 Aider 的入门技巧，建议不要在对话中添加过多文件，以保持效率并减少干扰。
  
  - 一位成员表示打算在深入使用 Aider 之前先查阅文档，并在需要时寻求进一步的澄清。

**提到的链接**：

- [Tips](https://aider.chat/docs/usage/tips.html)：使用 Aider 进行 AI 结对编程的技巧。
- [FAQ](https://aider.chat/docs/faq.html#how-do-i-include-the-git-history-in-the-context)：关于 Aider 的常见问题。
- [Linting and testing](https://aider.chat/docs/usage/lint-test.html)：自动修复 linting 和测试错误。

---

### **aider (Paul Gauthier) ▷ #**[**links**](https://discord.com/channels/1131200896827654144/1268910919057149974/1306348208263270451) (2 条消息):

> - `Organizing Code for AI`
> - `Aider Discord Guidelines`
> - `Server Rule Changes`

- **为 AI 组织代码库**：一位成员指出，为 **AI** 组织代码库与为人组织代码库类似，强调需要将内容拆分为逻辑模块并添加注释。
  
  - 他们强调了清晰组织对于提高可理解性和可维护性的重要性。
- **Aider Discord 实施新规则**：一位用户提到他们最初包含 **windsurf** 链接的推文被删除了，可能是由于新的服务器规则。
  
  - 他们引用了一份[指南](https://aider.discord)，指出 **Aider Discord** 专门用于讨论 Aider，禁止垃圾信息和未经请求的推广。

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1306356167106363454) (142 条消息🔥🔥):

> - `加入 Forge API Beta`
> - `3D 打印机推荐`
> - `Hermes 编程见解`
> - `参与研究项目`
> - `TEE 钱包整理（Collation）担忧`

- **加入 #forge-api-beta 变得更加容易**：多位成员表达了在加入 #forge-api-beta 时遇到的问题，**teknium** 确认已根据请求添加了成员。
  
  - *一些用户感到困惑，因为邮件链接将他们引导至了 general 频道。*
- **3D 打印机推荐讨论热烈**：关于 3D 打印机的讨论兴起，**bliponnobodysradar** 正在考虑 Ender 3 S1，而 **oleegg** 建议选择 Bambu Lab 以获得更易用的体验。
  
  - 成员们分享了他们的经验和偏好，在聊天中大家强烈建议不要选择 Ultimaker。
- **Hermes 编程作为学习工具**：成员们讨论了他们的入门编程语言，**shunoia** 在 Hermes 的帮助下转向了 Python，而 **oleegg** 对这一决定表示了*同情*。
  
  - **jkimergodic_72500** 解释说 Perl 是一种灵活的语言，为当前关于编程经验的对话提供了背景。
- **如何参与研究项目**：成员们询问了如何加入研究项目，**teknium** 建议将几个公共项目作为贡献的机会。
  
  - 小组表现出对如何更有效地参与的兴趣，表明社区渴望为正在进行的研究做出贡献。
- **对 TEE 钱包整理的担忧**：**mrpampa69** 提出了关于 TEE 钱包不一致性的担忧，认为这损害了 Bot 被感知到的主权。
  
  - 回应指出在整理之前需要稳健的决策，因为运行自主权仍然是防止滥用的首要任务。

**提到的链接**：

- [来自 JX (@JingxiangMo) 的推文](https://x.com/JingxiangMo/status/1856148967819751817?t=HnmrZrls1KaJ3KKjDfLABw&s=19)：介绍 Zeroth-01 Bot：全球最小的开源端到端人形机器人，起售价 350 美元！完全开源，包括硬件、SDK、仿真环境等。Zeroth-01 是最...
- [Your Life Story](https://lifestorys-b93f5c9c5deb.herokuapp.com/)：未找到描述
- [Bambu Lab X1C 3D Printer](https://us.store.bambulab.com/products/x1-carbon?skr=yes)：介绍我们的 3D 打印机 Bambu Lab X1 Carbon。凭借更快、更智能的打印，让你无需等待，尽情享受创作。体验并享受高精度、高细节的 3D 打印...

---

### **Nous Research AI ▷ #**[**interesting-links**](https://discord.com/channels/1053877538025386074/1132352574750728192/1306738520555126884) (6 条消息):

> - `Rizzler`
> - `俚语翻译器`
> - `翻译工具`
> - `简历转网站工具`

- **Rizzler 大获全胜**：看看 [Rizzler](https://rizzler.win/)，这是一个承诺提供迷人互动和顺畅连接的平台。
  
  - 该网站专为希望增强在线社交动态的用户打造。
- **俚语翻译器功能**：[Slang Translator](https://slangtranslator.cameronaaron.com/) 提供了一种解码和理解各种俚语术语的简便方法。
  
  - 通过浏览该平台，用户可以快速弥合地区方言中的沟通鸿沟。
- **高级翻译工具脱颖而出**：一款新型 AI 驱动的[翻译工具](https://translate.cameronaaron.com/)专注于文化细微差别和适应性，使翻译更具人性化。
  
  - 它通过考虑方言、正式程度、语气和性别来定制输出，使其成为满足多样化需求的灵活选择。
- **将你的简历转换为网站**：[Resume to Website Tool](https://resumetosite-b55155107b3e.herokuapp.com/) 可快速将简历转换为专业的 Bootstrap 网站。
  
  - 用户可以上传简历并在几分钟内获得一个响应式网站，从而增强他们的求职演示效果。

**提到的链接**：

- [Resume to Website Generator](https://resumetosite-b55155107b3e.herokuapp.com/)：未找到描述
- [Advanced Translation Tool - 准确且具有文化细微差别的翻译](https://translate.cameronaaron.com/)：在考虑文化细微差别、语境、正式程度、语气和性别的情况下进行跨语言文本翻译。

---

### **Modular (Mojo 🔥) ▷ #**[**general**](https://discord.com/channels/1087530497313357884/1098713601386233997/) (1 条消息):

aka_afnan: 大家好，我刚刚完成了 Mojo 语言的基础教程。

---

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1306348493803098153) (120 messages🔥🔥):

> - `Mojo Low-Level Syntax` (Mojo 低级语法)
> - `Performance of High-Level Syntax vs C` (高级语法与 C 的性能对比)
> - `Recursive Vectorization & Tail Call Optimization` (递归向量化与尾递归优化)
> - `LLVM and MLIR in Mojo` (Mojo 中的 LLVM 与 MLIR)
> - `Importance of Language Features` (语言特性的重要性)

- **Mojo 的低级语法性能**：成员们讨论了 Mojo 的低级语法虽然提供了比高级语法更好的性能，但可能无法保持 Pythonic 的精髓。
  
  - 有人指出，高级语法缺乏 **C** 的性能，但在某些条件下，像 **NumPy** 这样的工具仍能达到接近的结果。
- **递归向量化的困扰**：对话转向了 **Recursive Vectorization**（递归向量化）及其对 Mojo 性能的影响，强调了与 Rust 或 C++ 相比，递归代码在 Mojo 中缺乏优化的担忧。
  
  - 参与者一致认为，类型系统中缺失的特性目前阻碍了标准库的开发，使得编写高效代码变得困难。
- **MLIR 中的尾递归优化 (TCO)**：出现了一种观点，即在 MLIR 中实现 TCO，以实现编译器对递归代码的优化并获得更好的性能。
  
  - 成员们对在 LLVM IR 中保留控制流图（Control Flow Graphs）的必要性表示不确定，并讨论了其对调试的重要性。
- **语言特性优先级讨论**：大家达成共识，应优先考虑基础类型系统特性，而非更高级的优化，以确保在更多用户加入时语言已准备就绪。
  
  - 参与者警告说，在基础特性尚待完善时，不要让额外的议题压垮开发进度。
- **LLVM Offload 与协程实现**：大家对 LLVM 的 offload 能力以及 Mojo 中如何促进协程（Coroutine）实现表现出兴趣。
  
  - 讨论强调，协程在概念上与尾递归函数相似，这引发了关于是否需要透明装箱（Transparent Boxing）的思考。

**提到的链接**：

- [No Stop GIF - No Stop Pleading - Discover & Share GIFs](https://tenor.com/view/no-stop-pleading-begging-please-gif-17517986)：点击查看 GIF
- [Write Haskell as fast as C: exploiting strictness, laziness and recursion](https://donsbot.com/2008/05/06/write-haskell-as-fast-as-c-exploiting-strictness-laziness-and-recursion/)：在最近的一个邮件列表线程中，Andrew Coppin 抱怨在计算超大双精度浮点列表的平均值时，“优雅的、声明式的”代码性能不佳……
- [fixpt · All About Strictness Analysis (part 1)](https://fixpt.de/blog/2017-12-04-strictness-analysis-part-1.html)：未找到描述

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1306352810085847130) (72 messages🔥🔥):

> - `Perplexity 的 Campus Strategist 项目`
> - `广告与订阅模式的担忧`
> - `模型可用性更新`
> - `Gemini 在 Chatbot Arena 中的表现`
> - `API 仪表板问题`

- **Perplexity 将 Campus Strategist 项目扩展至加拿大**：应大众要求，Perplexity 正在将其 Campus Strategist 项目扩展到加拿大，邀请有兴趣的申请人联系以获取更多信息。
  
  - 2024 年项目的申请目前已开放，该项目为美国的大学生提供实践经验和导师指导。
- **对 Pro 用户显示广告的担忧**：对于向包括 Pro 订阅者在内的所有用户投放广告的实施方案，用户反应不一，许多人对这一变化表示沮丧。
  
  - 用户特别担心在支付订阅费用的同时仍会遇到广告，这会影响对服务价值的感知。
- **AI 模型可用性更新**：Claude 3 Opus 已从 Perplexity 中移除，以确保提供最佳模型，目前主要提供 Claude 3.5 Sonnet 和 Haiku。
  
  - 用户注意到 Gemini (Exp 1114) 最近在 Chatbot Arena 的多个类别中获得了最高排名，其性能表现获得了积极的初步印象。
- **API 仪表板的问题**：一些用户报告称 API 仪表板更新不准确，导致对 Token 使用情况产生困惑。
  
  - 一位用户确认此问题影响了多个成员，可能需要上报以寻求解决方案。
- **ChatGPT 搜索引擎咨询**：一位用户询问 ChatGPT 使用的是哪种搜索引擎，质疑它是否像 Perplexity 一样使用 Bing。
  
  - 这一讨论突显了用户对竞争对手 AI 平台的搜索功能和底层引擎的持续好奇。

**提到的链接**：

- [来自 Phi Hoang (@apostraphi) 的推文](https://x.com/apostraphi/status/1857109958107578509?s=61)：理所当然
- [来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文](https://x.com/lmarena_ai/status/1857110672565494098)：来自 Chatbot Arena 的重大新闻🔥 @GoogleDeepMind 最新的 Gemini (Exp 1114)，经过过去一周 6000 多个社区投票测试，现在总榜排名并列第一，评分大幅飞跃 40 多分——这是……
- [来自 Greg Feingold (@GregFeingold) 的推文](https://x.com/gregfeingold/status/1856088784699277668?s=61)：应大众要求，我们正将校园策略官项目扩展到加拿大 🇨🇦 如果你有兴趣申请，或者认识合适的人选，请联系我们！引用 Perplexity (@per...

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1306443733113245736) (8 messages🔥):

> - `Perplexity AI 功能`
> - `最佳办公鼠标`
> - `Google Gemini 发布`
> - `分享帖子设置`

- **Perplexity AI 被称为体育神器**：一位用户在帖子中兴奋地将 **Perplexity** 描述为 **疯狂的体育机器 (INSANE sports machine)**，强调了其令人印象深刻的能力。
  
  - 他们分享了一个链接以获取更多见解：[链接](https://www.perplexity.ai/search/i-want-to-put-some-information-5K1ijZHMRa6342FAPcsEuw)。
- **关于鼠标推荐的讨论**：一位用户发布了一个讨论 **最佳办公鼠标** 的链接，表明用户对优化生产力工具的兴趣日益增长。
  
  - 该链接被多次分享，强调了其在社区内的相关性：[链接](https://www.perplexity.ai/search/best-mouse-for-work-031fd.NlSeOAG_vHDd9pgg)。
- **Google 发布 Gemini 应用**：几位用户分享了关于 **Google Gemini 应用** 的链接，展示了对新技术发布的兴奋。
  
  - 相关文章包括 [TechCrunch 的公告](https://www.perplexity.ai/search/https-techcrunch-com-2024-11-1-k6p5L5QTTpOEnUwZXrY.Lw) 和关于 Gemini 功能的页面：[Gemini 应用](https://www.perplexity.ai/page/google-launches-gemini-app-9yiARC5PSmCeeOb6QtU7oQ)。
- **帖子可分享性通知**：管理员提醒一位用户确保其帖子被标记为 **可分享 (Shareable)**，并指向一个附件作为参考。
  
  - 该通知附带了一个结构化的消息链接：[链接](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825)。

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1306386320117076010) (7 条消息):

> - `Vercel AI SDK 使用方法`
> - `Reddit 引用问题`
> - `Search domain filter 问题`

- **Vercel AI SDK 与 Perplexity 结合使用**：一位用户询问如何在包含引用的情况下，将 **Vercel AI SDK** 与 **Perplexity** 结合使用。
  
  - 目前尚未收到回复，导致具体的实现细节或潜在的文档说明尚不明确。
- **通过 API 获取 Reddit 引用失败**：多位用户报告在过去一周内，将 **Reddit** 作为引用来源时出现问题，并指出该功能此前运行良好。
  
  - 一位用户提到，如果没有找到高置信度的来源，可能会被注入随机的 **URLs**，从而导致**结果不准确**。
- **Search domain filter 无法正常工作**：一位用户对 **search_domain_filter** 表示失望，称尽管遵循了正确的格式指南，该功能仍无法运行。
  
  - 另一位用户确认了类似的问题，引发了对过滤功能中潜在 Bug 的质疑。

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1306356603200471170) (22 条消息🔥):

> - `AI Agent 工具 Operator 发布`
> - `Francois Chollet 离开 Google`
> - `Gemini-Exp-1114 性能表现`
> - `ChatGPT for macOS 更新`
> - `Scaling Laws 理论引发担忧`

- **AI Agent 工具 'Operator' 即将发布**：据全员会议的员工更新消息，OpenAI 即将推出的代号为 'Operator' 的 AI Agent 工具预计将通过浏览器自动执行任务，并于 1 月发布。
  
  - 该工具将协助用户完成编写代码和预订旅行等操作，标志着 AI 实用性迈出了重要一步。
- **Francois Chollet 宣布从 Google 离职**：Keras 的创始人 Francois Chollet 将离开 Google 创办一家新公司，同时将以外部身份继续参与由 Jeff Carpenter 领导的 Keras 项目。
  
  - Chollet 表达了对他他在 Google 十年时光的感激之情，并对 Keras 成长为开发者广泛使用的框架感到自豪。
- **Gemini-Exp-1114 称霸 Chatbot Arena**：@GoogleDeepMind 的 Gemini-Exp-1114 在 Chatbot Arena 中获得了最高排名，在多个类别中以显著的分数提升超越了竞争模型。
  
  - 它目前在 Vision 排行榜上处于领先地位，并在创意写作和数学方面表现出色，展示了其先进的能力。
- **ChatGPT for macOS 增强编程支持**：ChatGPT for macOS 的 Beta 版本现在允许用户读取 VS Code 和 Xcode 等编程应用的内容，为 Plus 和 Team 用户提供上下文感知的回复。
  
  - 该功能旨在提高编程效率并简化开发者的工作流程。
- **对 AI Scaling Laws 的质疑**：人们对 AI 发展中 'Scaling Laws' 的有效性提出了担忧，质疑增加计算资源和更大的模型是否必然会带来进步。
  
  - 讨论强调，仅靠降低交叉熵损失（cross-entropy loss）可能不足以提升 AI 能力，反映了行业内的怀疑态度。

**提到的链接**：

- [来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文](https://x.com/lmarena_ai/status/1857110672565494098)：来自 Chatbot Arena 的重磅消息🔥 @GoogleDeepMind 最新的 Gemini (Exp 1114)，经过过去一周 6K+ 社区投票测试，现在以惊人的 40+ 分数跃升并列总榜第一 —— ma...
- [来自 Shirin Ghaffary (@shiringhaffary) 的推文](https://x.com/shiringhaffary/status/1856792898932539609?s=61)：新消息：OpenAI 正准备推出一款代号为 “Operator” 的新型计算机 AI Agent 工具，它可以代表用户通过浏览器执行操作，例如编写代码或预订旅行。员工在...
- [来自 François Chollet (@fchollet) 的推文](https://x.com/fchollet/status/1857060079586975852)：倾听我的内心... 好吧，看来你还没有。但只要你有基于 OpenAI API 构建的 SotA（或接近）解决方案，我们非常乐意对其进行验证并将其添加到公共 A...
- [来自 Casper Hansen (@casper_hansen_) 的推文](https://x.com/casper_hansen_/status/1857116047293477029)：以这种方式得知 OpenAI 将在 24 小时内发布 o1 真是太棒了。引用 Logan Kilpatrick (@OfficialLoganK) 的话：是的，Gemini-exp-1114 确实很棒 :)
- [来自 Logan Kilpatrick (@OfficialLoganK) 的推文](https://x.com/OfficialLoganK/status/1857106089063362768)：gemini-exp-1114…. 现在已在 Google AI Studio 上线，请尽情使用 : ) https://aistudio.google.com
- [再见，感谢你持续的合作，Francois Chollet！](https://developers.googleblog.com/en/farewell-and-thank-you-for-the-continued-partnership-francois-chollet/)：未找到描述
- [来自 François Chollet (@fchollet) 的推文](https://x.com/fchollet/status/1857012265024696494)：一些个人消息 —— 我将离开 Google，和一位朋友去创办一家新公司。更多消息即将公布！我将以外部身份继续深度参与 Keras 项目 —— 你仍然会...
- [来自 Tibor Blaho (@btibor91) 的推文](https://x.com/btibor91/status/1857121183163940904)：ChatGPT for macOS 现在可以读取 VS Code、Xcode、TextEdit 和 Terminal 等编程应用的内容，以提供上下文感知的答案，目前对 Plus 和 Team 用户开放 Beta 测试。

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-drama**](https://discord.com/channels/1179127597926469703/1181746144821387334/) (1 条消息):

420gunna: [https://x.com/richardmcngo/status/1856843040427839804?s=46](https://x.com/richardmcngo/status/1856843040427839804?s=46)

---

### **Interconnects (Nathan Lambert) ▷ #**[**random**](https://discord.com/channels/1179127597926469703/1183121795247779910/1306618421659697183) (18 messages🔥):

> - `Qwen vs Llama 性能对比`
> - `Cognitive revolution 播客`
> - `Qwen 的简单除法问题`
> - `模型训练中的 Synthetic data`

- **Qwen 在简单除法上超越 Llama**：在对比 **Qwen 2.5** 和 **Llama-3.1 405B** 的测试中，当处理 `A / B` 形式的基础除法问题时，Qwen 的表现优于 Llama。
  
  - *有趣的是*，Qwen 在处理大数字时会切换到 **CoT 模式**，利用 **LaTeX** 或 **Python**，而 Llama 的输出保持不变。
- **关于认知革命的训练后见解**：一位成员录制了一段 **90 多分钟** 的播客，讨论了认知革命（cognitive revolution），并强调了其坚实的基础。
  
  - 他们指出，这更多是关于模型、数据、evals 和代码协同工作的**过程**。
- **Synthetic data 助力 Qwen 训练**：有推测称，用于训练 Qwen 的 **20T tokens** 中有很大一部分由 **synthetic data** 组成。
  
  - 四舍五入与截断数字之间结果的差异表明，模型可能并未完全对齐。
- **对新模型的期望**：人们对即将推出的模型寄予厚望，预计它将满足那些关注技术性能的人设定的极高标准。
  
  - 澄清指出，该模型不应被视为 **GPT-5** 的直接对应物。

---

### **Interconnects (Nathan Lambert) ▷ #**[**memes**](https://discord.com/channels/1179127597926469703/1187551504995987576/1306499509270085694) (26 messages🔥):

> - `领导策略`
> - `开源 AI 讨论`
> - `实验室中的 Scaling Laws`
> - `Discord 商店角色`

- **有争议的领导策略**：一位成员对一种似乎鼓励员工盲目相信的领导策略表示怀疑，称这不是一个好策略，但可能具有激励作用。
  
  - 这一讨论与一段暗示缺乏指导的引用有关，并对其影响进行了评论。
- **迫切需要讨论开源 AI**：成员们敦促在另一家知名公司介入之前，与 Dwarkesh 就 **open-source AI** 的价值进行对话，强调了该话题的紧迫性。
  
  - 提议进行协作，以确保对话能够深入探讨当前对金融势力影响技术讨论的担忧。
- **Scaling Laws 与 Google Sheets**：有人评论了 **scaling laws** 的持续有效性，将误解归因于实验室使用 **Google Sheets**，而该软件无法充分绘制数据图表，特别是 sigmoids。
  
  - 这引发了关于金融专家能够绘制曲线却不理解其含义的笑谈，强调了数据呈现中的脱节。
- **章鱼哥、派大星和海绵宝宝的混淆**：在轻松的玩笑中，一位成员误将派大星（Patrick）称为章鱼哥（Squidward），引发了一番幽默的交流。
  
  - 对话提到了 Discord 商店中提供的海绵宝宝主题装饰，展示了社区的顽皮精神。

**提到的链接**：

- [Dylan Patel (@dylan522p) 的推文](https://x.com/dylan522p/status/1857131441492242439)：Scaling laws 仍然有效，因为所有实验室都使用 Google Sheets，无法拟合 sigmoid，只能在 log log plots 上画直线。所有的金融 Excel 大佬都在抓狂，因为他们可以画出...
- [Sam Altman (@sama) 的推文](https://x.com/sama/status/1856941766915641580)：没有墙 (there is no wall)
- [Timothy O'Hear (@timohear) 的推文](https://x.com/timohear/status/1857125743081222207)：@francoisfleuret 见于 François Chollet 的 AMA https://news.ycombinator.com/item?id=42130881 ☺️
- [morgan — (@morqon) 的推文](https://x.com/morqon/status/1856679803589382181)：如何回复评论请求

---

### **Interconnects (Nathan Lambert) ▷ #**[**posts**](https://discord.com/channels/1179127597926469703/1228051082631188530/1306531607250735126) (9 messages🔥):

> - `Andrew Carr Interview`
> - `Gemini 1.5 Ultra`
> - `Claude 3.5 Opus`
> - `Personas in AI`
> - `Scaling Realities`

- **Andrew Carr 讨论腾讯的 Persona 方法**：在最近的一次 [访谈](https://youtu.be/BIYik-AaHo8?si=wST6BTUv-aBUQiuv&t=2365) 中，Andrew Carr 在详细阐述 text-to-motion AI 模型时提到：*“哦，我们现在经常使用腾讯的 persona 方法，”*。
  
  - 另一位参与者回忆起在探索合成数据（synthetic data）时读过这篇 [论文](https://arxiv.org/abs/2406.20094)，并对其在实际应用中的效用表示好奇。
- **等待新的 AI 模型**：*Gemini 1.5 Ultra* 和 *Claude 3.5 Opus* 似乎备受期待，正如一位成员评论道：*“我们还在等它们，”*，强调了对技术进步的持续关注。
  
  - 社区似乎也对即将到来的索引更新充满期待。
- **对 Scaling Realities 的正面反馈**：一位成员对 *scaling realities* 的短版本表示赞赏，称其 *非常好*，并认为它比长版本更有影响力。
  
  - 他们认可长版本的技术价值，但更倾向于简洁的表达方式。
- **关于 AI 中 Personas 的讨论**：一位成员通过热情的鼓掌表情符号重申了对 *personas* 的看法，暗示其在 AI 讨论中的重要性。
  
  - 另一位参与者肯定了利用 personas 是非常直接的，并且能有效增强 prompt。
- **合成 SFT 和 DPO 的改进**：提到了 persona 方法如何显著辅助了他们的 *synthetic SFT and DPO* 工作，表明对模型性能产生了积极影响。
  
  - 对话暗示下周将进一步讨论这些模型中多样性（diversity）带来的好处。

 

**提到的链接**：[Andrew Carr on Pushing the Boundaries of Generative AI (Beyond Text)](https://youtu.be/BIYik-AaHo8?si=wST6BTUv-aBUQiuv&t=2365)：Andrew Carr 是 Cartwheel 的联合创始人兼首席科学家，他正在为游戏、电影和其他创意领域构建 text-to-motion AI 模型和产品……

 

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1306657748280283207) (2 messages):

> - `rapids cudf`

- **询问关于 rapids cudf 的知识**：一位成员询问是否有人熟悉 **rapids cudf**，表示希望获得相关信息或帮助。
  
  - *直接问你的问题就好，* 另一位成员建议道，鼓励公开对话。
- **鼓励提问**：一位成员通过建议原提问者直接提出关于 **rapids cudf** 的问题来推动对话。

 

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/1306434723697922088) (4 messages):

> - `Kernel Design Challenges`
> - `Triton and Performance Tuning`
> - `Issues with torch.compile`

- **Kernel 设计面临维度难题**：一位成员讨论了 Kernel 设计中的挑战，指出由于尺寸在 1 到 16 之间变化，很难确定第一维是否为向量。
  
  - 他们质疑将尺寸填充（padding）到最小 16 是否是唯一的各种方案。
- **BLOCK_SIZE_M 的高效配置**：另一位成员建议在 Kernel 中将 `BLOCK_SIZE_M` 作为 `tl.constexpr` 用于 if 语句，并使用 `early_config_prune` 根据 batch size 进行 autotuning。
  
  - 对于 batch size 为 1 的情况，他们建议实现 gemv 以提高 GPU 性能，尽管可能会导致 Kernel 崩溃。
- **Triton 实现遇到崩溃**：在尝试建议的调整后，一位成员报告仍有崩溃发生，并链接到了一个 [GitHub issue](https://github.com/pytorch/pytorch/issues/140423)，该 issue 详细描述了在使用从源码构建的 Triton 时 `torch.compile` 出现的问题。
  
  - 他们指出，当编译包含 Triton 模块的模型时会出现该问题，并特别引用了遇到的错误。

 

**提到的链接**：[torch.compile breaks with Triton built from source · Issue #140423 · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/140423)：🐛 描述 Bug：torch.compile 在使用从源码构建的 Triton 时（截至 11 月 12 日）会报错。如何复现：从 master 分支构建 Triton，运行包含 Triton 模块的模型的 torch.compile……

 

---

### **GPU MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1306372060058095688) (2 messages):

> - `Direct Access to GPU`
> - `Torch.compile() with DDP`
> - `Torch.compile() with FSDP`

- **关于 Direct GPU Access 的咨询**：一位成员询问了实现 **direct access to GPU** 以提高性能的方法。
  
  - 讨论中未分享具体的方法。
- **Torch.compile() 与 DDP 的使用**：提出了一个关于结合使用 **torch.compile()** 和 **Distributed Data Parallel (DDP)** 的后续问题。
  
  - 成员们询问 **torch.compile()** 应该包装在 DDP 包装器之外还是放在其内部，并强调了潜在的问题。
- **Torch.compile() 与 FSDP 的注意事项**：对话还涉及了在 **Fully Sharded Data Parallel (FSDP)** 中使用 **torch.compile()** 的情况。
  
  - 参与者好奇在与 FSDP 集成时，是否适用与 DDP 类似的注意事项。

 

---

### **GPU MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/1306457445039870013) (4 messages):

> - `GPU profiling tools`
> - `Thread creation on GPUs`
> - `RGB to greyscale conversion performance`

- **用户探索 GPU profiling tools**：一位成员询问了其他人使用的 **profiling tools**，并表示在理解 **ncu** 生成的报告时遇到困难。
  
  - 另一位成员建议：*你需要习惯使用 NCU*，并称其为目前最好的 Profiler，能提供宝贵的优化见解。
- **理解 GPU 上的线程创建**：一位成员澄清说，在 **GPU** 上，创建线程没有开销，因为它们都在 Kernel 启动时就开始运行。
  
  - 虽然让线程执行更多工作是理想的，但挑战在于平衡 **computation**（计算）与 **data loaded**（数据加载）。
- **RGB 转灰度图面临带宽挑战**：讨论围绕为 **RGB 图像转灰度图** 等任务派生线程的效率展开，质疑过多的线程是否会引入开销。
  
  - 讨论指出，转换过程通常是 **bandwidth limited**（带宽受限）的，涉及加载三个值进行简单计算以产生一个值。

 

---

### **GPU MODE ▷ #**[**off-topic**](https://discord.com/channels/1189498204333543425/1215328286503075953/1306505901418614844) (1 messages):

> - `Feijoa dessert`
> - `Grilled beef patties`
> - `Ivan tea`

- **斐济果甜点融合多种风味**：通过将 **feijoa puree**（斐济果泥）与 **tvorog**（俄式碎干酪）、**sour cream**（酸奶油）和 **stevia**（甜菊糖）混合，制作出了一款美味的甜点。
  
  - 这种独特的组合展示了有效融合甜味和奶油元素的能力。
- **烤牛肉饼成为主角**：主菜是 **grilled beef patties**（烤牛肉饼），搭配 **potatoes**（土豆）和 **ketchup**（番茄酱）。
  
  - 这顿丰盛的晚餐在咸鲜风味与经典调味品之间取得了平衡。
- **清爽的柳兰茶完美收尾**：为了搭配这餐饭，成员享用了加奶的 **Ivan tea**（柳兰茶），为全天的菜单提供了一个舒缓的结尾。
  
  - 这种饮料为用餐体验增添了独特的草本气息。
- **色彩缤纷的沙拉增加脆爽口感**：一份由 **cucumber**（黄瓜）、**daikon radish**（大白萝卜）、**Napa cabbage**（大白菜）等组成的清爽沙拉，拌入 **mayonnaise**（蛋黄酱）。
  
  - 这种搭配带来了脆爽和新鲜感，补充了餐食中较油腻的部分。

 

---

### **GPU MODE ▷ #**[**rocm**](https://discord.com/channels/1189498204333543425/1233704710389764236/) (1 messages):

leiwang1999_53585: 你使用过 ck profiler 吗？

---

### **GPU MODE ▷ #**[**self-promotion**](https://discord.com/channels/1189498204333543425/1288557096404516945/1306644355913613313) (3 messages):

> - `Video Length Discussions`
> - `Interest in Triton Content`

- **7.5 小时的视频引发不同反响**：一位成员提到他们 *粗略浏览了那个 7.5 小时的视频*，因为内容太多难以消化，但他们很喜欢看过的部分。
  
  - 另一位成员幽默地评论了视频的时长，称：*“你可以只看你感兴趣的部分，”* 并强调了描述栏中包含的章节。
- **对更多 Triton 视频的需求**：一位成员表达了对创作者视频的赞赏，并特别要求在不久的将来提供 *更多 Triton 内容*。
  
  - 这一请求反映了观众对 *Triton 相关讨论* 日益增长的兴趣。

 

---

### **GPU MODE ▷ #**[**🍿**](https://discord.com/channels/1189498204333543425/1298372518293274644/) (1 messages):

apaz: <@325883680419610631>  
[https://github.com/gpu-mode/discord-cluster-manager/issues/23](https://github.com/gpu-mode/discord-cluster-manager/issues/23)

### **GPU MODE ▷ #**[**thunderkittens**](https://discord.com/channels/1189498204333543425/1300872762163728550/1306402684135280691) (34 条消息🔥):

> - `Kernel Shared Memory`
> - `Matrix Multiplication Optimization`
> - `Dynamic Shared Memory Issues`
> - `CUDA Function Attributes`

- **Kernel Shared Memory 在高占用时崩溃**：一位用户在请求 **49160 字节**或更多 Shared Memory 时遇到了 **Kernel 崩溃**，尽管该数值理应小于 `MAX_SHARED_MEMORY`。
  
  - 该问题与静态 Shared Memory 的使用有关，静态实现在某些架构上存在限制。
- **同步矩阵乘法（Matrix Multiplications）解析**：讨论表明，**16x64 \* 64x16** 的 matmul 无法使用异步 WGMMAs，而**同步指令**虽然允许在 Tensor Cores 上使用，但可能导致性能瓶颈。
  
  - 建议用户增加 Batch Size 以优化性能，目标是 H100 架构所偏好的 **64** 维。
- **Dynamic Shared Memory 的问题**：会议指出 CUDA 存在一项限制，即静态 Shared Memory 不能超过 **50KB**，因此需要改用 Dynamic Shared Memory。
  
  - 要分配超过 **48KB** 的内存，必须使用 **cudaFuncSetAttribute()** 函数，这是针对特定架构的注意事项。
- **Dynamic Shared Memory 行为确认**：一位用户验证了将 Dynamic Shared Memory 分配增加到 **40,000 字节**是可行的，而 **50,000 字节**则会导致失败。
  
  - 他们在思考使用不同的 API 进行 Kernel 启动是否能解决问题，正如引用的 StackOverflow 帖子中所指出的那样。
- **成功解决问题**：在交流了建议和参考资料后，最初出现问题的 Kernel 配置最终得以正常工作。
  
  - 一位成员对解决所面临问题过程中获得的帮助表示感谢。

 

**提到的链接**：[Using maximum shared memory in Cuda](https://stackoverflow.com/questions/63757245/using-maximum-shared-memory-in-cuda)：我无法使用超过 48K 的 Shared Memory（在 V100, CUDA 10.2 上）。我调用了 cudaFuncSetAttribute(my_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, ...

 

---

### **GPU MODE ▷ #**[**edge**](https://discord.com/channels/1189498204333543425/1303441437592911912/1306746857250361375) (5 条消息):

> - `React Native LLM Library`
> - `LLM Inference on Android`
> - `Transformer Memory Bound`
> - `Bitnet 1.58 A4`
> - `GGUF Q8 Performance`

- **Software Mansion 发布 React Native LLM 库**：Software Mansion 发布了一个在 React Native 中使用 LLM 的新库，利用 **ExecuTorch** 来提升性能。它通过安装命令简化了使用流程，包括克隆仓库并在 iOS 模拟器上运行。
  
  - 更多信息和贡献请查看 [GitHub 仓库](https://github.com/software-mansion/react-native-executorch)。
- **Android 上 LLM 推理的内存约束**：成员们讨论了新型 Android 智能手机上的 LLM 推理是否属于 **Memory Bound**（内存受限）。共识是这取决于应用场景，低 Context 通常是 Memory Bound，而高 Context 则是 Compute Bound（计算受限）。
  
  - 一位用户指出，现代处理器提供的计算能力通常高于内存带宽，这表明较新的硬件可能仍面临内存限制。
- **用于优化推理的 Bitnet 1.58 A4**：为了实现快速推理，推荐使用带有微软 T-MAC 操作的 **Bitnet 1.58 A4**，在 7B 模型上性能可达 **10 token/s**。它可以在桌面 CPU 上运行，即使对于 GPU 资源有限的用户也触手可及。
  
  - 训练不需要从头开始，因为 Hugging Face 提供了将模型转换为 Bitnet 的指南，尽管过程可能比较复杂。
- **GGUF Q8 提供近乎无损的性能**：在讨论替代方案时，指出 **GGUF Q8** 对性能的影响极小，特别是对于 7B-13B 模型。用户尚未在更小的模型上进行测试，但建议它对于资源受限的设备可能非常有益。
  
  - 这意味着 GGUF Q8 对于在低端硬件上运行且不希望在性能上做太大权衡的用户来说是一个可行的选择。

 

**提到的链接**：[GitHub - software-mansion/react-native-executorch](https://github.com/software-mansion/react-native-executorch.git)：通过在 GitHub 上创建账号来为 software-mansion/react-native-executorch 的开发做出贡献。

 

---

### **Notebook LM Discord ▷ #**[**use-cases**](https://discord.com/channels/1124402182171672732/1124403655819415592/1306390233520275630) (16 条消息🔥):

> - `Magic Book 播客实验`
> - `播客工具的功能性担忧`
> - `移动端版本可用性问题`
> - `总结《身体从未忘记》（The Body Keeps Score）`
> - `将旧理论与时事联系起来`

- **Magic Book 播客实验引人入胜**：一位成员创建了一个[神奇的 PDF](https://youtu.be/fkfGI089iRE)，它可以根据查看者的不同展示不同的解读，并以播客形式分享。
  - 听众被鼓励分享他们对这种创新播客方式的看法。
- **播客工具需要更细粒度的控制（Granular Control）**：大家公认用户在播客开发中寻求增强功能，但目前的工具可能缺乏所需的**细粒度控制**。
  - 一位成员表示，如果出现任何严肃的产品开发需求，他可以提供帮助。
- **Notebook 移动端版本受到批评**：有人担心 **Notebook 移动端版本**几乎无法使用，特别是在复制笔记和滚动等基础功能方面。
  - 成员们对这些问题表示赞同，并希望在不久的将来能有专门的 App。
- **《身体从未忘记》得到了很好的总结**：一位成员称赞 AI 总结 **《身体从未忘记》（The Body Keeps Score）** 的能力，有效地捕捉了书中的严肃主题。
  - 对话强调了在忙碌的世界中微学习（microlearning）的价值，并将其与盲目刷手机进行了对比。
- **将新闻学理论与现代事件联系起来**：一位成员反思了**沉默的螺旋理论（spiral of silence theory）**及其与当前媒体动态的相关性，特别提到了《卫报》退出 Twitter 的事件。
  - 这一用例强调了将旧理论与当代事件结合以获得社会学洞察。

**提到的链接**：[Top Shelf](https://open.spotify.com/show/0MvNgBDb2NsZJN4cREl7yF)：播客 · Four By One Technologies · "Top Shelf" 是你获取当今畅销书快速、深刻见解的首选播客。只需 15 分钟，即可获得要点、精华和新鲜视角...

---

### **Notebook LM Discord ▷ #**[**general**](https://discord.com/channels/1124402182171672732/1124402182909857966/1306355439092629614) (40 条消息🔥):

> - `NotebookLM 的隐私与数据安全`
> - `NotebookLM 的功能请求`
> - `NotebookLM 的发音问题`
> - `用户体验反馈`

- **NotebookLM 数据安全澄清**：讨论强调，根据 [Google 的支持页面](https://support.google.com/notebooklm/answer/14275965)，无论账户类型如何，用户的数据都是安全的，不会被用于训练 NotebookLM 模型。
  - 隐私声明重申了这一点，指出人工审核员仅在排除故障时才可能访问信息。
- **针对响应语言的功能请求**：一位用户请求能够为每个笔记本设置响应语言，因为他们遇到了收到英文答案而非希腊文的问题。
  - 这一功能可以提升多语言环境下的用户满意度。
- **NotebookLM 的发音挑战**：用户报告称 NotebookLM 在正确发音某些单词方面存在困难，例如将 "presents" 误处理为礼物（名词）而非呈现（动词）。
  - 建议的一种变通方法是使用粘贴文本直接指示发音。
- **文件上传的用户体验问题**：一位用户提出了在向 NotebookLM 上传文件时面临的挑战，指出团队正在解决这些问题。
  - 另一位用户提到达到了笔记本数量上限，导致信息被删减。
- **对 API 更新的关注**：成员们对 NotebookLM API 的潜在更新表示好奇，但被告知目前尚未发布任何功能路线图（roadmap）。
  - 社区依赖公告频道获取任何更新和新功能。

**提到的链接**：[Privacy - Help](https://support.google.com/notebooklm/answer/14275965)：未找到描述

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1306355843092058173) (53 条消息🔥):

> - `Perplexity 广告引入`
> - `AI Agent 性能更新`
> - `ChatGPT 桌面应用增强`
> - `Gemini AI 反馈`
> - `技术债与 AI 影响`

- **Perplexity 开启广告实验**：Perplexity 宣布将开始在美国尝试以“赞助后续问题”形式呈现的广告，Indeed 和 Whole Foods 等品牌将参与其中。

- 他们表示，广告收入将有助于支持出版商，因为仅靠订阅不足以产生可持续的收入。
- **Gemini AI 性能飙升**：@GoogleDeepMind 的 Gemini (Exp 1114) 在数学和创意写作等多个领域实现大幅性能提升后，已跃升至 Chatbot Arena 的并列第一名。
  
  - 它现在已可在 Google AI Studio 中进行测试，不过 API 访问权限即将推出。
- **ChatGPT 桌面应用新功能**：macOS 版 ChatGPT 桌面应用现在可以与 VS Code 和 Terminal 等本地应用程序集成，目前已向 Plus 和 Team 用户提供 Beta 版本。
  
  - 然而，一些用户反映存在功能缺失和性能缓慢的问题，这引发了对其当前集成能力的质疑。
- **对 AI 和技术债的担忧**：一篇博客讨论了 AI 实际上可能会如何增加与技术债相关的成本，并指出拥有旧代码库的公司将比拥有高质量代码的公司面临更多困难。
  
  - 该文章强调了生成式 AI 如何扩大这两类群体之间的性能差距。
- **关于解析 Excel 文件的讨论**：用户讨论了使用 LLM 处理 Excel 文件的最佳方法，特别是将财务数据解析为 JSON 或 Markdown 表格。
  
  - 建议包括将数据导出为 CSV，以便更容易地进行编程语言集成。

**提到的链接**：

- [来自 Logan Kilpatrick (@OfficialLoganK) 的推文](https://x.com/OfficialLoganK/status/1857106089063362768): gemini-exp-1114…. 现已在 Google AI Studio 提供，尽情享受吧 : ) https://aistudio.google.com
- [AI 让技术债变得更昂贵](https://www.gauge.sh/blog/ai-makes-tech-debt-more-expensive): AI 增加了低质量代码的惩罚
- [Bloomberg - 你是机器人吗？](https://www.bloomberg.com/news/articles/]): 未找到描述
- [Bloomberg - 你是机器人吗？](https://www.bloomberg.com/news/articles/2024-11-13/openai-nears-launch-of-ai-agents-to-automate-tasks-for-users?accessToken=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzb3VyY2UiOiJTdWJzY3JpYmVyR2lmdGVkQXJ0aWNsZSIsImlhdCI6MTczMTUyODYxOCwiZXhwIjoxNzMyMTMzNDE4LCJhcnRpY2xlSWQiOiJTTVdOQURUMEcxS1cwMCIsImJjb25uZWN0SWQiOiJFODA3NUYyRkZGMjA0NUI2QTlEQzA5M0EyQTdEQTE4NiJ9.TTJZiuo4Nk2U295FHBFsxeN0YGznZJ32sHnNReQmEjM): 未找到描述
- [通过 Prompt 注入获取 Shell：OpenAI 的容器化 ChatGPT 环境](https://0din.ai/blog/prompt-injecting-your-way-to-shell-openai-s-containerized-chatgpt-environment): 深入探讨 OpenAI 的容器化 ChatGPT 环境，展示用户如何通过受控的 Prompt 注入和文件管理技术与其底层结构进行交互。通过探索...
- [Perplexity 在其平台引入广告 | TechCrunch](https://techcrunch.com/2024/11/12/perplexity-brings-ads-to-its-platform/): AI 驱动的搜索引擎 Perplexity 表示，将从本周开始在其平台上尝试投放广告。
- [Playbooks 简介 - Devin 文档](https://docs.devin.ai/Working_with_Teams/playbooks-intro): 未找到描述
- [来自 Kevin Weil 🇺🇸 (@kevinweil) 的推文](https://x.com/kevinweil/status/1857120814333825060?s=46): 今日发布：两项重大更新让 @ChatGPTapp 在 PC 和 Mac 桌面端更加实用 🖥 💻 首先，适用于 Windows 的 ChatGPT 桌面应用现已面向所有用户开放。自发布早期版本以来...
- [来自 lmarena.ai (前身为 lmsys.org) (@lmarena_ai) 的推文](https://x.com/lmarena_ai/status/1857110672565494098?s=46): 来自 Chatbot Arena 的重磅消息🔥 @GoogleDeepMind 最新的 Gemini (Exp 1114)，在过去一周经过 6K+ 社区投票测试，目前总排名并列第一，得分大幅跃升 40+ 分——使其...
- [来自 Andrej Karpathy (@karpathy) 的推文](https://x.com/karpathy/status/1857126049357914266): 我不确定是否有足够多的人订阅了 @Smol_AI 的新闻通讯。它每天发送一封非常全面的邮件，总结 X、Reddit、Discord 上的 AI/LLM 讨论。可能还有其他的...
- [来自 Lucas Beyer (bl16) (@giffmana) 的推文](https://x.com/giffmana/status/1856993726591099066?s=46): 引用 yobibyte (@y0b1byte) https://www.lakera.ai/blog/visual-prompt-injections
- [Bloomberg - 你是机器人吗？](https://www.bloomberg.com/news/articles/2024-11-13/openai-nears-launch-of-ai-agents-to-automate-task): 未找到描述
- [ChatGPT 桌面端配合 Xcode 对比 Alter [对比] #chatgpt #chatgptupdate #apple](https://youtu.be/Wm2ughBFjnk): ChatGPT 桌面端配合 Xcode 的快速对比。发现：1. 无法看到 Xcode 中的所有内容，只能看到代码窗格 2. 代码过长会被截断 3. 可能...
- [来自 Kol Tregaskes (@koltregaskes) 的推文](https://x.com/koltregaskes/status/1856754648146653428?s=46): Google Gemini 告诉用户去死！！！😲 聊天记录是真实的，你可以在这里阅读并继续对话：https://g.co/gemini/share/6d141b742a13
- [Reddit - 深入探索任何事物](https://www.reddit.com/r/artificial/comments/1gq4acr/gemini_told_my_brother_to_die_threatening/): 未找到描述
- [GitHub - google-deepmind/alphafold3: AlphaFold 3 推理流水线。](https://github.com/google-deepmind/alphafold3): AlphaFold 3 推理流水线。通过在 GitHub 上创建账号来为 google-deepmind/alphafold3 的开发做出贡献。
- [Tessl 以 5 亿美元以上估值融资 1.25 亿美元，旨在构建编写和维护代码的 AI | TechCrunch](https://techcrunch.com/2024/11/14/tessl-raises-125m-at-at-500m-valuation-to-build-ai-that-writes-and-maintains-code/): 许多初创公司和大型科技公司都尝试过构建用于编写软件代码的人工智能。现在，又有一家新公司脱颖而出

---

### **Latent Space ▷ #**[**ai-announcements**](https://discord.com/channels/822583790773862470/1075282504648511499/) (1 条消息):

swyxio: 已在 hn 发布！

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1306361989844697231) (31 条消息🔥):

> - `AI 驱动的计算机控制`
> - `针对 GPT 的 Lorebook`
> - `Mac App 界面变化`
> - `AI 进步的未来`
> - `Copilot 中的图像工具`

- **AI 通过 ChatGPT 控制计算机 UI**：一位成员分享了他们的系统，该系统通过包含 **Computer Vision** 和 Python 的 **PyAutoGUI** 在内的技术栈，使 **ChatGPT** 能够间接控制计算机的 UI。该成员邀请大家提供反馈，并渴望与其他人在增强 AI 驱动的自动化方面建立联系，并暗示会有视频演示。
  
  - 其他人提出了关于代码可用性的问题，并将其与 **OpenInterpreter** 等现有解决方案进行了比较。
- **针对 GPT 的 Lorebook 增强了上下文**：一位用户为 GPT 创建了一个 Lorebook，可以根据关键词加载条目，具有导入/导出功能，并能防止条目泛滥。他们计划在调试完成后将其分享到 **GreasyFork**，并欢迎关于新功能的建议。
  
  - 讨论还澄清了该 Lorebook 是作为 **Tampermonkey** 或 **Violentmonkey** 的脚本实现的。
- **Mac App 界面变化受到称赞**：成员们对 **Mac App 模型选择器**界面的优化表示感谢，指出这显著提升了用户体验。一位成员评论说，整个社区都感激实施这一变化的团队。
  
  - 这一评论呼应了对提高工具可用性的更新表示赞赏的情绪。
- **关于 AI 未来影响的预测**：有一场关于 AI 变革潜力的讨论，将其与互联网在 dot-com 泡沫期间的演变进行了比较。参与者对 AI 可能导致社会发生前所未有的变化表示乐观，将其比作“意识的全面转变”。
  
  - 成员们反思了过去对技术进步的预测，认为那些早期认识到 AI 潜力的人可能会获得巨大的影响力。
- **对新图像工具的好奇**：一位成员推测 **Copilot 主页**上的新图像是否是用新的图像工具创建的。这引发了进一步的询问，促使了关于用于图像生成的底层技术的讨论。
  
  - 这种推测表明了人们对 AI 生成内容及其与现有产品集成的持续兴趣。

 

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1306386437108662333) (11 条消息🔥):

> - `有效地使用 LLM`
> - `对内容标记（Content Flags）的担忧`
> - `自定义 GPT 的使用`
> - `角色扮演角色创建`
> - `模型在写作中的表现`

- **掌握 LLM 是一项习得的技能**：成员们讨论认为，虽然任何人都可以使用 LLM，但有效地进行 **prompting** 需要**技能和练习**，就像使用木工工具一样。
  
  - *了解为了提高获得理想输出的几率而应该包含的内容*，可以显著增强交互体验。
- **无惧内容标记（Content Flags）**：有人对在与模型交互过程中收到内容标记表示担忧，特别是在涉及敏感话题时。
  
  - 然而，一些成员指出，只要用户在法律范围内操作并避免有害内容，他们的账号通常是安全的。
- **自定义 GPT 的积极体验**：讨论强调了**自定义 GPT** 在专门任务中的有效性，一位成员提到了使用 **Wolfram** 进行数学运算的好处。
  
  - 事实证明，这种定制化在提高社区成员的生产力和实用性方面非常有价值。
- **角色扮演角色开发的挑战**：一位用户表达了对内容标记阻碍其创建复杂角色扮演角色的挫败感，该角色的叙事与敏感的历史事件相关。
  
  - 他们指出，重复的标记会导致对账号风险的担忧，尤其是在挑战模型边界时。
- **对 GPT 在创意写作中表现的反思**：一位成员分享了他们使用模型帮助完善虚构战争故事的主题和描述的积极体验。
  
  - 虽然模型在对话描写方面表现不佳，但它可以协助组织思路并在故事讲述中提供有用的建议。

 

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1306489763863330829) (5 条消息):

> - `ChatGPT capabilities` (ChatGPT 能力)
> - `Retrieving information from ancient texts` (从古代文献中检索信息)
> - `Nostalgia for old prompting techniques` (对旧 Prompt 技术的怀旧)
> - `Model improvements in games` (模型在游戏中的改进)

- **利用 9 Pillars Solutions 探索 ChatGPT 的极限**：一位成员鼓励其他人通过实验 **9 pillars solutions** 的构成来挑战 **ChatGPT** 的边界。
  
  - 他们声称通过这种方法可以获得重要的见解。
- **在技术工程中寻找古代文献**：有人询问如何在 **advanced tech and engineering** 开发平台的背景下优化对 **ancient texts** 的搜索。
  
  - 成员们对如何重置平台的搜索程序以获得更好的结果感到好奇。
- **怀念旧的 Prompt 技术**：一位成员表达了对早些时候尝试通过 Prompt 让模型计算 **owl**（猫头鹰）高度的怀念。
  
  - 另一位成员表示赞同，并建议类似的探索现在可能仍然可行且有趣。
- **ChatGPT 3.5 在游戏中表现出进步**：一位用户兴奋地分享说，他们让 **GPT-3.5** 成功玩起了 24 点游戏，甚至有时能赢。
  
  - 这引发了关于模型在游戏中的性能和可靠性改进的讨论。

 

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1306489763863330829) (5 条消息):

> - `9 Pillars Solutions`
> - `Information Retrieval from Ancient Texts` (从古代文献中检索信息)
> - `Advancements in Technology` (技术进步)
> - `Model Performance in Games` (模型在游戏中的表现)

- **探索 9 Pillars Solutions**：一位成员鼓励挑战 ChatGPT 的极限，以发现 **9 Pillars Solutions** 的潜力。
  
  - 他们暗示这种探索可能会带来变革性的结果。
- **检索古代文献信息的挑战**：一位成员询问如何利用开发平台上的先进技术优化对古代文献的搜索，并重置搜索参数。
  
  - 他们寻求关于如何有效利用该平台满足其信息检索需求的帮助。
- **对模型解决问题的怀旧**：一位成员回忆了过去尝试通过 Prompt 让模型根据图像确定猫头鹰高度的经历。
  
  - 他们表达了希望在今天的模型上重新进行这些实验的愿望。
- **Model 3.5 表现出令人印象深刻的游戏性能**：另一位成员分享了使用 **3.5** 模型的成功经验，报告称它在玩 24 点游戏时能经常获胜。
  
  - 他们强调模型在游戏过程中很少撒谎，展示了其能力。
- **回顾过去的实验**：一位成员对表达的怀旧之情表示认可，并建议重新审视猫头鹰问题解决挑战。
  
  - 他们认为在当前模型中仍有探索类似 Prompt 的机会。

 

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1306421804369444958) (34 条消息🔥):

> - `Dockerized Open Interpreter`
> - `Open Interpreter as a Shell Pass-Through`
> - `Beta App Performance`
> - `Worker Pool Configuration`
> - `Memory Store Concept`

- **关于 Docker 化 Open Interpreter 的反馈**：一位成员建议，提供一个完全支持的 **Docker image**，并针对作为 **workers** 或 **warm spares** 运行进行优化，将极大地改善他们使用 OI 的工作流（目前他们通过变通方法实现）。
  
  - 他们强调了对最大迭代次数（max iterations）和临时实例设置等更多 **配置功能** 的需求，这表明了巨大的后端潜力。
- **Open Interpreter 作为 Shell 透传工具的想法**：讨论了主要将 Open Interpreter 用作 Shell 的 **透传（pass-through）** 工具以无缝执行命令，类似于 **Vim** 在不同模式下的操作方式。
  
  - 探讨了通过常驻进程（long-running process）来更轻松地与解释器集成的可行性，并强调了上下文管理（context management）的需求。
- **Beta 版桌面应用的性能**：一位成员询问 Beta 版应用是否比控制台集成表现得更好，回复指出确实如此。
  
  - 据确认，由于与开源仓库相比具有增强的基础设施，**桌面应用** 承诺提供最佳的 Interpreter 体验。
- **Worker Pool 配置概念**：一位成员提出了关于与容器通信的理想形式的问题，寻求关于 Worker Pool 设置的建议，并对开发分支（development branch）中即将到来的改进表示期待。
  
  - 他们讨论了特定的命令结构，以增强运行处理作业或脚本时的可用性。
- **用于上下文管理的 Memory Store 概念**：提出了实现 **memory store** 以保留命令历史记录而非输出结果的想法，以便在不超支 **tokens** 的情况下高效管理上下文。
  
  - 还讨论了使用新的管道签名（pipe signature）来指定为 **LLM** 保留哪些输出的可能性，以此作为简化上下文管理的一种方式。

### **OpenInterpreter ▷ #**[**ai-content**](https://discord.com/channels/1146610656779440188/1149229778138824765/1306588139162177618) (7 条消息):

> - `VividNode v1.7.1 发布`
> - `Voice Lab 框架`
> - `ChatGPT macOS 版`
> - `概率计算突破`

- **VividNode v1.7.1 带来了令人兴奋的新特性**：新发布的 **VividNode v1.7.1** 增加了对 **LiteLLM API Keys** 的全面支持，涵盖了 [此链接](https://github.com/yjg30737/pyqt-openai/releases/tag/v1.7.1) 中的 **60+ 个提供商**和 **100+ 个模型**。
  
  - 改进包括通过 **QLineEdit** 进行模型输入以提升易用性，并修复了与文本输入和 **LlamaIndex 功能**相关的 Bug。
- **Voice Lab 框架开源**：一名成员宣布开源 **Voice Lab**，这是一个用于在不同模型和 Prompt 下评估 **LLM 驱动的 Agent** 的框架，详情见 [GitHub](https://github.com/saharmor/voice-lab)。
  
  - **Voice Lab** 旨在优化 Prompt 并提高 Agent 性能，邀请社区参与贡献和讨论。
- **ChatGPT 与桌面应用集成**：ChatGPT 现在与 macOS 上的桌面应用程序兼容，在面向 Plus 和 Team 用户的 Beta 测试版中，它可以针对编程应用提供增强的响应，由 **OpenAIDevs** 在 [此处](https://fxtwitter.com/openaidevs/status/1857129790312272179?s=46&t=G6jp7iOBtkVuyhaYmaDb0w) 分享。
  
  - 此次更新标志着 ChatGPT 与用户桌面编程环境交互方式的重大转变。
- **概率计算的新突破**：一段 **YouTube 视频** 强调了一项**新的计算突破**，据报道其能效比领先的 **NVIDIA GPUs** 高出 **1 亿倍**；点击 [此处](https://www.youtube.com/watch?v=hJUHrrihzOQ) 观看。
  
  - 该视频讨论了概率计算的进展，这可能会彻底改变计算效率领域。
- **VividNode 和自定义 URL 支持**：一名成员询问了 **VividNode** 与 LLM 推理和 OpenAI 集成的自定义 URL 的兼容性。
  
  - 开发者确认了与多个提供商的兼容性，并正在积极开发自定义 URL 支持。

**提到的链接**：

- [来自 OpenAI Developers (@OpenAIDevs) 的推文](https://fxtwitter.com/openaidevs/status/1857129790312272179?s=46&t=G6jp7iOBtkVuyhaYmaDb0w)：ChatGPT 🤝 VS Code, Xcode, Terminal, iTerm2。ChatGPT macOS 版现在可以与桌面应用协同工作。在面向 Plus 和 Team 用户的早期 Beta 版中，你可以让 ChatGPT 查看编程应用以提供更好的...
- [新的计算突破实现了 1 亿倍的 GPU 性能！](https://www.youtube.com/watch?v=hJUHrrihzOQ)：在这段视频中，我讨论了概率计算，据报道，与最好的 NVIDIA GPUs 相比，它能实现 1 亿倍的能效提升。查看...
- [GitHub - saharmor/voice-lab: 语音 Agent 的测试和评估框架](https://github.com/saharmor/voice-lab)：语音 Agent 的测试和评估框架 - GitHub - saharmor/voice-lab: Testing and evaluation framework for voice agents

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1306580889672613970) (15 条消息🔥):

> - `Cohere 嵌入模型`
> - `Discord 访问问题`
> - `培养 AI 和机器人领域的年轻人才`
> - `播客内容分析`
> - `即将举行的活动`

- **Cohere Embedding 的最佳 Token 数量**：一名成员询问了 Cohere 嵌入模型（特别是针对多模态输入）的**最佳字符/Token 数量**。
  
  - 另一名成员澄清说，目前**最大上下文**为 **512 tokens**，并建议在该限制内进行实验。
- **成员的 Discord 访问问题**：一名成员对因被封禁而无法访问 Discord 表示沮丧，称这影响了他们的在线参与。
  
  - 另一名成员提供了支持，表示很高兴看到朋友重新上线并与社区互动。
- **活动亮点：Ageing, Progress, and Decline 研讨会**：分享了一个名为“**Ageing, Progress, and Decline**”的活动，定于 **2024 年 12 月 6 日**举行，并将在 **Hugging Face Discord 服务器**上进行直播。
  
  - 提供了注册链接，邀请成员以虚拟或亲临现场的方式参加。
- **播客内容分析建议**：一名成员就如何**从数小时的播客内容中提取**信息以及随后如何利用这些数据寻求建议。
  
  - 另一名成员参与了讨论，询问目标是否是将播客内容转录以便与**大语言模型**配合使用。

 

**提到的链接**：[危机中的共识：AI 数据公地的快速衰落](https://www.eventbrite.ca/e/1039740199927?aff=oddtdtcreator)：AI 阅读小组会议，邀请了《Consent in Crisis: The Rapid Decline of the AI Data Commons》的作者之一。

 

---

### **Cohere ▷ #**[**announcements**](https://discord.com/channels/954421988141711382/996880279224451154/1306646929987469362) (1 条消息):

> - `Research Prototype Beta Program`
> - `Text-based Deliverables`
> - `User Feedback for Tool Development`

- **研究原型报名最后召集**：发出提醒，**研究原型 Beta 计划**的报名即将截止，具体是在 **周二** 之前。鼓励感兴趣的参与者通过提供的 [链接](https://forms.gle/Teis9VwM6eZP6nxVA) 报名。
  
  - 该计划提供了一个探索新 **Cohere 工具** 的机会，旨在增强研究和写作任务，并提供宝贵的见解和反馈。
- **频繁文本创作者的机会**：该计划针对经常处理 **基于文本的交付物**（如报告和博客文章）的人群，方便他们在工具公开发布前使用。参与者将帮助塑造符合其工作流的工具功能。
  
  - Beta 测试人员将参与迭代开发过程，目标是创建一个处理复杂任务的有效助手。
- **诚邀建设性反馈**：Beta 测试组的参与者在体验实验性工具时，需提供 **详细且具有建设性的反馈**。目标是确保该工具能有效协助用户的研究和写作工作。
  
  - 通过影响其开发，用户可以帮助完善原型，以更好地满足实际应用的需求。

 

**提到的链接**：[Research Prototype - Early Beta Sign Up Form](https://forms.gle/Teis9VwM6eZP6nxVA)：感谢您有兴趣参加我们研究原型的 Beta 测试阶段——这是一个旨在帮助用户处理研究和写作任务的工具，例如：创建复杂的报告、执行...

 

---

### **Cohere ▷ #**[**questions**](https://discord.com/channels/954421988141711382/1168411509542637578/1306672470589440040) (2 条消息):

> - `Bug reporting process`

- **Benny 寻求 Bug 报告指导**：@benny0917 询问了报告 Bug 的流程，并引用了 Discord 上的一个特定消息链接。
  
  - *sssandra* 的回复确认已获悉该情况，并表示该 Bug 已被标记。
- **sssandra 确认问题**：*sssandra* 为让 @benny0917 久等表示歉意，同时已将潜在的 Bug 进行了标记。
  
  - 这表明针对 Bug 报告的咨询已采取了迅速行动。

 

---

### **Cohere ▷ #**[**api-discussions**](https://discord.com/channels/954421988141711382/1168578329423642786/1306352897755447426) (13 条消息🔥):

> - `HTTP Request Details`
> - `Network Error Analysis`
> - `Azure AI V2 API Status`

- **分享用于 Reranking 的 HTTP 请求**：一位用户分享了使用模型 'rerank-english-v3.0' 进行 Reranking 的 HTTP 请求负载。这突出了其他人如何排查与此特定功能相关的问题。
  
  - 另一位用户提供了一个与查找片段相关的代码片段，但澄清其并未使用 'return_documents' 参数。
- **识别 API 调用中的网络错误**：一位用户报告遇到了网络/OpenSSL 错误，特定错误消息指示连接问题。他们指出这似乎是偶尔发生，而非完全的 API 连接问题。
  
  - 该用户计划更新库并实现重试机制，建议进一步检查网络或 SSL 设置可能会有帮助。
- **Azure AI V2 API 不可用状态**：一位用户询问了 Azure AI 端点提供的 API V2 的可用性，文档显示该版本尚未运行。目前提供的服务包括各种模型，但仅支持 Cohere v1 API。
  
  - 用户指出了 Azure AI Studio 目前可用的模型，并根据提供的文档链接指出 v2 API “即将推出”。

 

**提到的链接**：[Cohere on Azure — Cohere](https://docs.cohere.com/docs/cohere-on-microsoft-azure)：此页面介绍了如何在 Microsoft Azure 上使用 Cohere 模型。

 

---

### **Cohere ▷ #**[**projects**](https://discord.com/channels/954421988141711382/1218409701339828245/1306350619287617616) (1 messages):

> - `Vision Language Action Models`
> - `Benchmarking Robotic Learning Tasks`
> - `SoTA VLMs like GPT4o`
> - `Multimodal Action Models`
> - `Collaborative Research Release`

- **VLA 模型新研究发布**：今天，一篇题为 *Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks* 的新论文发布，详细介绍了 Manifold、Georgia Tech、MIT 和 Metarch AI 之间的合作研究。
  
  - 该论文评估了 **Vision Language 和 Vision Language Action 模型**在 **20** 个不同真实世界任务中控制机器人的表现，这是迈向更广泛基准测试的重要一步。
- **令人兴奋的见解与模型评估**：该研究强调了新兴的 VLA 模型类别，并包含了对 **GPT4o** 等一些 **SoTA VLMs** 的评估。
  
  - 作者渴望获得反馈，并分享了其作品的链接供社区讨论，包括一个[包含要点的 Twitter 线程](https://x.com/HarshSikka/status/1856739777208574151)。
- **获取实验细节与资源**：研究人员提供了各种资源，包括[项目网站](https://multinet.ai/static/pages/Multinetv01.html)、[代码库](https://github.com/ManifoldRG/MultiNet/tree/main)以及 [Arxiv 论文](https://arxiv.org/abs/2411.05821)。
  
  - 这些资源包括实验细节、模型描述以及对其创新工作的进一步见解。

**提到的链接**：[来自 harsh (@HarshSikka) 的推文](https://x.com/HarshSikka/status/1856739777208574151)：很高兴分享我们的新论文 "Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks"。我们评估了 VLM 和 VLA 模型在 20 个不同真实世界任务中控制机器人的表现...

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1306677831379587204) (1 messages):

> - `RAGformation`
> - `Cloud architecture automation`
> - `Dynamic flow diagrams`
> - `Pricing estimates for architecture`

- **RAGformation 自动化云端设置**：RAGformation 允许用户通过自然语言描述其用例来自动生成云配置，从而产生量身定制的云架构。
  
  - 用户还可以通过**动态生成的流程图**来可视化其设置。
- **即时获取价格预估**：该平台为生成的架构提供**价格预估**，使用户能够有效地为项目制定预算。
  
  - 提供细化选项，允许用户根据需要调整配置。

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1306549870554775604) (26 条消息🔥):

> - `AI Agent 的记忆系统`
> - `LlamaIndex 的 Go 版本`
> - `ChromaDB 数据摄取问题`
> - `使用 SentenceSplitter 和 SentenceWindowNodeParser`
> - `LlamaParse 联系协助`

- **Mem0 记忆系统增强 AI 交互**：最近，**Mem0** 被添加到 LlamaIndex 中，引入了一个智能记忆层，可以随着时间的推移实现个性化的 AI Assistant 交互。更多详情请查看 [Mem0 Memory](https://docs.llamaindex.ai/en/stable/examples/memory/Mem0Memory/)。
  
  - 该系统可以通过 [托管平台](https://docs.mem0.ai/platform/overview) 或 [开源解决方案](https://docs.mem0.ai/open-source/quickstart) 访问。
- **目前没有开发 LlamaIndex Go 版本的计划**：目前**没有计划**发布 **LlamaIndex** 的 Go 版本，因为构建该版本需要封装 Python 函数。现有成员讨论了 Go 所需的库，并强调即使没有这些库也可以利用 API 调用。
  
  - 目前*没有人*在追求原生 Go 版本，因为许多模型可以通过直接 API 调用访问，而不需要本地库。
- **ChromaDB 数据摄取中出现意外的向量创建**：一位用户报告在将 PDF 摄取到 **ChromaDB** 时出现了意外的向量数量，预期输出为一个向量，但收到了两个。其他成员建议这可能是由于 PDF 加载器默认按页拆分文档的行为导致的。
  
  - **SentenceWindowNodeParser** 也被讨论为可能增加向量数量的原因，因为其设计会为每个句子生成一个 Node。
- **关于在 SentenceWindowNodeParser 中使用 SentenceSplitter 的咨询**：一位用户询问在摄取 Pipeline 中同时使用 **SentenceSplitter** 和 **SentenceWindowNodeParser** 的情况，并对产生的向量数量表示担忧。社区反馈确认，如果不当组合使用它们会导致生成过多的 Node，从而使结果复杂化。
  
  - 无论选择何种配置，默认的 PDF 加载器拆分行为也可能导致观察到的数量增加。
- **寻求 LlamaParse 集成协助**：一位成员提出了关于 **LlamaParse** 的支持请求以及除了网站表单之外的联系方式。社区迅速将他们转介给另一位可以进一步协助咨询的成员。
  
  - 针对其企业级 RAG Pipeline 的集成，已启动私信以提供个性化支持。

 

**提到的链接**：[Mem0 - LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/memory/Mem0Memory/)：未找到描述

 

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1306376510600511528) (16 条消息🔥):

> - `tinybox 之间的 GPU 资源共享`
> - `MLPerf Training 4.1 结果`
> - `tinygrad 中的 Buffer 传输函数`
> - `网络交互与瓶颈`
> - `PCIe 带宽能力`

- **Cloud Sharding vs Machine Sharding**：一位成员表示需要了解他们是 **machine sharded** 还是 **cloud sharded**，并强调了在云层同步缓慢期间可能产生的费用。
  
  - 他们指出，如果由于云配置导致性能下降，将是一种负面体验。
- **MLPerf 4.1 的激动人心消息**：Tinygrad 取得了一个显著的里程碑，**tinybox red** 和 **green** 都参加了 **MLPerf Training 4.1**，展示了 **BERT** 的训练。
  
  - 团队的目标是在下一轮 MLPerf 中实现 **3 倍速** 的性能提升，并且是第一个在训练中包含 **AMD** 的团队。
- **引入 Buffer 传输函数**：一位贡献者分享了一个 Pull Request，该函数支持 tinygrad 中 **CLOUD 设备** 之间的 **buffer transfer**（缓冲区传输），确保平滑的缓冲区外复制过程。
  
  - 虽然大小检查可能不是必需的，但强调了要与现有功能保持一致。
- **探索网络协议**：对话涉及了混合虚拟云设置促进 **networked interactions**（网络交互）的能力，建议甚至可以使用带有 GPU 的节点配置以获得更好的性能。
  
  - 然而，成员们对通过 CPU 和 PCIe 连接可能产生的 **bottlenecks**（瓶颈）表示担忧。
- **评估 PCIe 带宽指标**：成员们讨论了 **ConnectX-6 适配器** 通过 InfiniBand 实现高达 **200Gb/s** 的潜力，以及它们与 **OCP3.0 带宽** 的关系。
  
  - 理论评估建议实现绕过 CPU 的 **400 GbE 双向** 连接。

**提到的链接**：

- [来自 tiny corp (@__tinygrad__) 的推文](https://x.com/__tinygrad__/status/1856987088367059163)：MLPerf Training 4.1 已发布，tinybox red 和 green 都在上面使用 tinygrad 训练 BERT。（ResNet-50 已停止使用）这些时间是基准时间。我们的目标是下一次 ML 提升 3 倍...
- [mdaiter 提交的 CLOUD 设备上的 Buffer 传输 · Pull Request #7705 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/pull/7705/files)：标题说明了一切——从一个设备读取 buffer，将其放入另一个不同设备中。你其实不需要 assert 或 sz 参数，但我希望保持一致性...

---

### **tinygrad (George Hotz) ▷ #**[**learn-tinygrad**](https://discord.com/channels/1068976834382925865/1070745817025106080/1306372444596080681) (3 条消息):

> - `tinygrad 中的位运算`
> - `CLANG 后端 Bug 调查`
> - `Tensor Gather 功能`

- **通过位运算增强 Minimum Fix**：一位成员建议将 minimum fix 改为使用 **bitwise_not**，并提议将其作为一个 good first issue，在 **argmin** 和 **minimum** 函数上应用同样的操作。
  
  - 这一更改旨在显著提高这些操作的效率。
- **CLANG 后端 Bug 引发疑问**：另一位成员调查了 **CLANG 后端** 中一个与 Tensor 操作的最大值计算相关的 Bug，该 Bug 导致 `.max().numpy()` 和 `.realize().max().numpy()` 的输出不符合预期。
  
  - 这种差异突显了在处理 Tensor 操作（特别是负值）时可能存在的问题。
- **融合 kv_pass 函数中的 Gather 操作**：一位成员询问是否可以融合 **kv_pass** 函数中的 `Tensor.gather` 调用，以及生成的 Tensor **k_seqs** 和 **v_seqs** 是否会被显式化（materialized）。
  
  - 他们寻求关于如何有效检查这种融合的指导，并强调了其对性能的影响。

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1306497540623040514) (15 条消息🔥):

> - `Nanobitz 建议`
> - `Meta 总部的 Llama 活动`
> - `Tokenization 策略`
> - `微调 Llama 的最佳数据集大小`
> - `Liger Kernel 改进`

- **Nanobitz 建议使用替代 Docker 镜像**：Nanobitz 建议使用 [axolotlai/axolotl](https://hub.docker.com/r/axolotlai/axolotl/tags) 镜像，即使它们比 *winglian* 版本落后一天。
  
  - *Hub.docker.com* 显示最新的标签来自 **20241110**。
- **关于微调 Llama 最佳数据集大小的讨论**：Arcadefira 询问了微调 **Llama 8B 模型** 的理想数据集大小，特别是考虑到其低资源语言的情况。
  
  - Nanobitz 询问了关于 Tokenizer 重叠的问题，并建议如果重叠程度足够，**5k** 的数据集可能就足够了。
- **Meta 总部的 Llama 活动**：Le_mess 询问是否有人参加 **12 月 3-4 日** 在 Meta 总部举行的 **Llama 活动**。
  
  - Neodymiumyag 表示感兴趣，并请求提供有关该活动的更多信息链接。
- **Liger kernel 迎来改进**：Xzuyn 提到 **Liger** 项目改进了 *orpo kernel*，并在 [GitHub pull request](https://github.com/linkedin/Liger-Kernel/pull/362) 中详细说明了这一点。
  
  - 他们还指出，随着 Batch Size 的增加，其表现趋于平稳。
- **分享社交媒体见解**：Kearm 分享了 Nottlespike 在 X.com 上的一条帖子，以幽默的视角展示了他们的一天。
  
  - 分享的链接指向一条详细描述 Nottlespike 经历的帖子。

**提到的链接**：

- [来自 Kearm (@Nottlespike) 的推文](https://x.com/Nottlespike/status/1857181970746466769)：我这一天就是这么过的
- [未找到标题](https://hub.docker.com/r/axolotlai/axolotl/tags)：未找到描述
- [未找到标题](https://hub.docker.com/r/winglian/axolotl/tags?name=20241110)：未找到描述

---

### **LAION ▷ #**[**general**](https://discord.com/channels/823813159592001537/823813160075132991/1306379896460808252) (5 条消息):

> - `EPOCH 58 COCK`
> - `LAION 版权讨论`
> - `公开索引与版权`

- **EPOCH 58 COCK 取得进展**：**EPOCH 58 COCK** 模型目前拥有 **60M 参数** 并使用 **f16**，随着“腿部”的出现和“鸡冠”变得更加清晰，该模型正显示出进展。
  
  - 该模型似乎在细节和结构上都在不断进化。
- **LAION 的版权问题引发辩论**：一场关于 LAION 数据集允许用户下载 **50 亿张图片** 的讨论展开，有人声称根据 **欧盟法律 (EU law)** 这构成了版权侵权。
  
  - 批评者认为这规避了付费墙和许可条款，与常规的浏览器缓存 (Browser caching) 不同。
- **关于版权法知识的争论**：*Trevityger* 被指责在 LAION 行为的版权法问题上发表 **伪法律谬论 (pseudolegal nonsense)**。
  
  - 成员们对将 LAION 的下载行为与典型的 Web 浏览器行为进行 **错误等价 (false equivalences)** 表示不满。
- **公开索引与版权合法性**：一位成员辩称，在任何情况下，**公开链接的公开索引** 都不可能构成版权侵权。
  
  - 这一观点认为，访问公开链接不应干涉版权法。

**提到的链接**：

- [回复：LAION。在外部硬盘上永久下载 50 亿张图片、220TB 数据不属于“浏览器缓存”](https://old.reddit.com/r/aiwars/comments/1gr0912/re_laion_downloading_5billion_images_220tb_of/)：这个版块的大多数人还不够博学，无法对复杂的版权法发表意见，然而有些人却试图进行错误等价的辩论...

---

### **LAION ▷ #**[**research**](https://discord.com/channels/823813159592001537/824374369182416994/1306354438662783048) (5 条消息):

> - `视觉语言动作模型基准测试 (Benchmarking Vision Language Action Models)`
> - `Watermark Anything`
> - `AI 生成器`
> - `1200万张公共领域图像`

- **VLA 模型的协作基准测试**：由 Manifold、佐治亚理工学院、MIT 和 Metarch AI 合作发布了一篇题为 *Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks* 的新论文，重点关注 **VLA 模型**在 **20 个真实世界机器人任务**中的表现。
  
  - 您可以在此 [推文亮点汇总](https://x.com/HarshSikka/status/1856739777208574151) 中查看要点，并访问 [Arxiv 论文](https://arxiv.org/abs/2411.05821) 获取更深入的分析。
- **Watermark Anything 实现发布**：项目 *Watermark Anything with Localized Messages* 现已在 [GitHub](https://github.com/facebookresearch/watermark-anything) 上发布，展示了该研究论文的官方实现。
  
  - 该实现支持动态水印，这在各种 AI 应用中可能会非常有用。
- **仅含 1M 参数的快速模型**：一位成员指出，讨论的模型仅有 **100万（1M）参数**，表明其速度足以集成到各种 AI 生成器中。
  
  - 这种效率可以提升水印技术在整个领域的普及度。
- **公共领域图像集发布**：一个 **1200万张图像的数据集** 现已进入公共领域，这对于各种机器学习任务和项目都非常有价值。
  
  - 感兴趣使用开源资源的人员可以点击 [此处](https://source.plus/pd12m?size=n_100_n) 访问该数据集。

**提到的链接**：

- [来自 harsh (@HarshSikka) 的推文](https://x.com/HarshSikka/status/1856739777208574151)：很高兴分享我们的新论文 "Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks"。我们评估了 VLM 和 VLA 模型在 20 个不同的真实世界任务中控制机器人的表现……
- [GitHub - facebookresearch/watermark-anything: 论文 "Watermark Anything with Localized Messages" 的官方实现](https://github.com/facebookresearch/watermark-anything)：论文 "Watermark Anything with Localized Messages" 的官方实现 - facebookresearch/watermark-anything

---

### **DSPy ▷ #**[**show-and-tell**](https://discord.com/channels/1161519468141355160/1202371242519441499/1306739627582947369) (1 条消息):

> - `macOS 版 ChatGPT`
> - `与桌面应用集成`
> - `dspy 工作流`
> - `编程辅助`

- **macOS 版 ChatGPT 与桌面应用集成**：令人兴奋的消息！**macOS 版 ChatGPT** 现在可以与 **VS Code**、**Xcode**、**Terminal** 和 **iTerm2** 等桌面应用集成，为用户提供更强大的编程辅助。
  
  - 该功能目前处于面向 Plus 和 Team 用户的 Beta 测试阶段，允许 ChatGPT 直接与开发环境交互，从而提高生产力。
- **增强 dspy 工作流的潜力**：一位成员表示希望这一功能可以扩展到 **dspy GPTs**，从而进一步增强工作流。
  
  - 他们强调了这对项目的潜在影响，认为这可能会成为他们工作的 **游戏规则改变者 (game-changer)**。

**提到的链接**：[来自 OpenAI Developers (@OpenAIDevs) 的推文](https://x.com/OpenAIDevs/status/1857129790312272179?t=l7rfG-jT3etXxH9ZrEXPPQ&s=19)：ChatGPT 🤝 VS Code, Xcode, Terminal, iTerm2。macOS 版 ChatGPT 现在可以与桌面应用协同工作。在面向 Plus 和 Team 用户的早期 Beta 测试中，你可以让 ChatGPT 查看编程应用以提供更好的……

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/1306480745560281118) (7 messages):

> - `Long-code generation with large tokens`（大 Token 长度的长代码生成）
> - `Deprecation of LM assertions`（LM assertions 的弃用）
> - `Developing a multi-infraction LLM application`（开发多违规场景的 LLM 应用）

- **生成超过 4096 tokens 代码编辑的工具**：一位成员询问像 **Cursor** 和 **Aider** 这样的工具是如何实现在超过 **4096 tokens** 的代码中管理并生成编辑内容的。
  
  - 这表明随着开发者寻求有效的解决方案，需要明确这些工具中的 Token 管理机制。
- **LM assertions 引起困惑**：一位成员询问 **LM assertions** 是否正在被弃用，并指出当前文档中缺少对 `dspy.Suggest` 或 `dspy.Assert` 的引用。
  
  - 另一位成员回答说，虽然文档中没有直接引用，但仍可以通过搜索栏找到这些内容，这表明文档正在更新中。
- **Value 和 Key Errors 的协助**：在关于 LM assertions 的讨论中，一位成员提到了持续存在的 **Value** 和 **Key Errors** 问题，并寻求相关资源或代码帮助。
  
  - 这突显了在应对文档变更时，寻求技术支持是大家的共同关注点。
- **创建通用的 LLM 应用**：一位成员描述正在开发一个 LLM 应用，目前该应用为特定的违规行为（即与**酒精摄入**相关的行为）生成辩护文件。
  
  - 他们希望将其功能扩展到其他违规行为，而无需单独的优化 Prompt，从而对统一方法的可能性提出了疑问。

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/1306510671185313844) (2 messages):

> - `Quiz Eligibility`（测验资格）
> - `Course Content Timeline`（课程内容时间线）

- **新成员询问测验资格**：一位新成员询问是否可以补做测验并仍有资格获得 *Trailblazer and above trails* 认证。
  
  - 另一位成员确认了资格，但强调了快速赶进度的重要性，因为每个测验都**与课程内容直接相关**，所有内容截止日期为 **12 月 12 日**。
- **强调课程内容的相关性**：成员们讨论了在完成测验时保持课程内容同步的重要性。
  
  - 提醒所有测验和作业必须在 **12 月 12 日**之前提交，以确保完整参与。

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-readings-discussion**](https://discord.com/channels/1280234300012494859/1282735578886181036/) (1 messages):

sheilabel: 就在今天！ [https://www.eventbrite.ca/event/1039740199927?aff=oddtdtcreator](https://www.eventbrite.ca/event/1039740199927?aff=oddtdtcreator)

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #**[**leaderboard**](https://discord.com/channels/1111172801899012102/1214705495974092810/1306388235836063866) (2 messages):

> - `Palmyra X 004 model`（Palmyra X 004 模型）
> - `Writer handler implementation`（Writer 处理程序实现）
> - `Pull Request Review`（Pull Request 审查）

- **提交了新的 Writer Handler 和 Palmyra X 004 模型**：一位成员宣布提交了一个 [PR，旨在将 Writer 处理程序和 **Palmyra X 004 模型** 添加到排行榜](https://github.com/ShishirPatil/gorilla/pull/755)。
  
  - 该贡献已获得确认并开放审查，并向审查人员表示了感谢。
- **快速确认 PR 审查**：另一位成员表示打算审查提交的 PR，并称：*“我会看一下。谢谢！”*
  
  - 这反映了项目开发活动中持续的协作与支持。

 

**提到的链接**：[[BFCL] Add support for Writer models and Palmyra X 004 by samjulien · Pull Request #755 · ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla/pull/755)：此 PR 为 BFCL 增加了对 Writer 模型和我们最新的 Palmyra X 004 的支持。谢谢！

 

---

### **AI21 Labs (Jamba) ▷ #**[**general-chat**](https://discord.com/channels/874538902696914944/874538902696914947/1306385756247298058) (2 messages):

> - `Legacy Model Deprecation`（旧版模型弃用）
> - `Transition to Open Source Solutions`（向开源解决方案迁移）

- **旧版模型导致业务中断**：一位成员对 **legacy models** 的弃用表示沮丧，称由于新模型在输出方面并非 **1:1** 对应，其影响具有**巨大的破坏性**。
  
  - *我们希望继续使用旧版模型*，因为他们觉得过渡并不顺利。
- **向开源方案的转换仍在进行中**：同一位成员指出，他们正在努力转向**开源解决方案**，但已经为旧模型付费近 **2 年**。
  
  - 他们对未来的弃用表示担忧，询问：*“我们如何确定 AI21 将来也不会弃用新模型？”*

 

---

### **Mozilla AI ▷ #**[**announcements**](https://discord.com/channels/1089876418936180786/1089876419926032396/1306737513905524776) (1 条消息):

> - `Local LLMs Workshop`
> - `SQLite-Vec Metadata Filtering`
> - `Refact.AI Autonomous Agents`

- **构建你自己的本地 LLMs 工作坊**：参加即将在 **周二** 举行的名为 [Building your own local LLM's: Train, Tune, Eval, RAG all in your Local Env.](https://discord.com/events/1089876418936180786/1300842793945530378) 的活动，学习如何开发本地语言模型。
  
  - 参与者可以期待关于构建高效本地 LLM 系统的实战训练和见解。
- **SQLite-Vec 现已支持元数据过滤**：**周三** 将举行一场关于 SQLite-Vec 新功能的活动：[SQLite-Vec now supports metadata filtering!](https://discord.com/events/1089876418936180786/1300483739872399411)。
  
  - 这将使用户能够高效地过滤元数据，增强数据管理能力。
- **与 Refact.AI 一起探索自主 AI**：本 **周四**，参加 [Autonomous AI Agents with Refact.AI](https://discord.com/events/1089876418936180786/1300459081181429810) 会议，深入了解自主 Agents 的世界。
  
  - 通过这场引人入胜的演讲，了解 AI 技术的创新策略和应用。

 

---

---

---

---

---

{% else %}

> 完整的逐频道细分内容已针对电子邮件进行截断。
> 
> 如果您想查看完整细分，请访问此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！
> 
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}