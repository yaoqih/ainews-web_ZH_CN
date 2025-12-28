---
companies:
- deepseek
- meta-ai-fair
- wandb
- nvidia
- anthropic
- hugging-face
- perplexity-ai
date: '2024-10-18T22:46:38.720062Z'
description: '**DeepSeek Janus** 和 **Meta SpiRit-LM** 是近期发布的两个备受关注的多模态 AI 模型，分别展示了在图像生成和语音合成方面的进展。DeepSeek
  Janus 将用于图像理解和生成的视觉编码器进行了分离，从而在两项任务中都取得了更好的效果。Meta 的 SpiRit-LM 引入了一种富有表现力的语音和写作模型，通过生成音高和风格单元，在标准
  TTS（文本转语音）的基础上实现了改进。


  此外，**W&B Weave** 提供了全面的大语言模型（LLM）可观测性和多模态微调工具。行业动态方面，英伟达（Nvidia）的 Nemotron 70b 模型表现不及预期；Meta
  开源了用于媒体生成基准测试的 Movie Gen Bench；Perplexity 推出了具备多步推理能力的内部搜索功能；Anthropic 更新了 Claude
  应用程序。开源进展包括 Hugging Face 修复了 transformers 中的梯度累积问题，以及倡导开源 AI 以防止大科技公司的垄断。此外，文中还强调了“用于结合多个模型技能的模型合并”技术。'
id: b5276b9b-b1bf-4b89-b9bd-5fc6cbf145d7
models:
- nemotron-70b
- claude
- claude-3.5-sonnet
- gpt-4o
original_slug: ainews-deepseek-janus-and-meta-spirit-lm
people:
- bindureddy
- aravsrinivas
- danielhanchen
- clementdelangue
- cwolferesearch
title: DeepSeek Janus 与 Meta SpiRit-LM：解耦的图像与表现力语音全模态。
topics:
- multimodality
- image-generation
- speech-synthesis
- fine-tuning
- model-merging
- benchmarking
- open-source
- model-optimization
- reinforcement-learning
---



---

{% if medium == 'web' %}

**目录**

[TOC]

{% else %}

**目录** 和 **频道摘要** 已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 行业动态与进展**

- **新 AI 模型与基准测试**：[@bindureddy](https://twitter.com/bindureddy/status/1846824566443921668) 指出 **Nvidia Nemotron Fine-Tune 并不是一个很好的 70b 模型**，在多个类别中的表现均逊于其他 SOTA 模型。[@AIatMeta](https://twitter.com/AIatMeta/status/1847004755576737823) 宣布开源 **Movie Gen Bench**，包括两个新的媒体生成基准测试：Movie Gen Video Bench 和 Movie Gen Audio Bench，旨在评估文本转视频（text-to-video）以及（文本+视频）转音频（(text+video)-to-audio）的生成能力。

- **AI 公司动态**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1846954158156583224) 宣布推出 **Perplexity for Internal Search**，这是一个可以同时搜索网页和团队文件的工具，支持多步推理（multi-step reasoning）和代码执行（code execution）。[@AnthropicAI](https://twitter.com/AnthropicAI/status/1846928983297769655) 推出了 **Claude iOS 和 Android 应用**的新外观，包括对 iPad 的支持和项目（project）功能。

- **开源进展**：[@danielhanchen](https://twitter.com/danielhanchen/status/1847023676954456569) 报告称 **梯度累积（gradient accumulation）修复现已进入 transformers 的主分支**，并感谢 Hugging Face 团队的合作。[@ClementDelangue](https://twitter.com/ClementDelangue/status/1846915223149478086) 分享了一份关于“阻止大科技公司成为大 AI（Stopping Big Tech from becoming Big AI）”的重要报告，强调了 **开源 AI 在促进创新和降低准入门槛方面的作用**。

**AI 研究与技术见解**

- **模型合并**：[@cwolferesearch](https://twitter.com/cwolferesearch/status/1846967851015434677) 讨论了 **模型合并（model merging）在结合多个 LLM 技能方面的有效性**，并以 Prometheus-2 为例，说明合并的效果优于多任务学习（multi-task learning）和集成（ensembles）。

- **AI 安全与评估**：[@_philschmid](https://twitter.com/_philschmid/status/1846830024416018933) 解释了 @GoogleDeepMind 的 **过程奖励模型（Process Reward Models, PRM）**，该模型对 LLM 推理的每一步提供反馈，与标准的基于结果的奖励模型（outcome-based Reward Models）相比，准确率提高了 8%，数据效率提升了高达 6 倍。

- **AI 开发工具**：[@hrishioa](https://twitter.com/hrishioa/status/1846941743364952258) 介绍了 **diagen**，这是一个使用各种 AI 模型生成 @terrastruct d2 图表的工具，其中 Sonnet 表现最好，Gemini-flash 在视觉反思（visual reflection）方面表现出色。

**AI 应用与用例**

- **音频处理**：[@OpenAI](https://twitter.com/rohanpaul_ai/status/1847043781616414882) 宣布在其 Chat Completions API 中支持音频，并提供了 Chat Completions API 与 Realtime API 在音频应用方面的对比点。

- **AI 在教育领域**：[@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1846972657411240254) 建议那些难以评估使用 AI 辅助的学生的教师，应该为能够自行评估学生的 AI 做好准备，这可能通过具备语音能力的 AI 以及观察学生解决问题的 AI 来实现。

- **AI 用于数据分析**：[@perplexity_ai](https://twitter.com/perplexity_ai/status/1846950770736091509) 推出了内部知识搜索（Internal Knowledge Search），允许用户同时搜索组织文件和网页。

**AI 社区与职业见解**

- [@willdepue](https://twitter.com/willdepue/status/1846977577971601563) 鼓励那些对 AI 感兴趣且具有非传统背景的人申请 OpenAI residency，强调需要对构建真正的 AI 和解决复杂问题充满热情。

- [@svpino](https://twitter.com/svpino/status/1846884066605650355) 宣布即将开展一个机器学习工程（Machine Learning Engineering）课程，重点是完全使用开源工具构建一个大规模的端到端机器学习系统。

- [@jxnlco](https://twitter.com/jxnlco/status/1847052906400567345) 分享了一个关于咨询服务收费过低的轶事，强调了在 AI 咨询行业中合理定价的重要性。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1：高性能本地 LLM 配置**

- **[7xRTX3090 Epyc 7003, 256GB DDR4](https://i.redd.it/40hwy3uqscvd1.jpeg)** ([分数: 149, 评论: 72](https://reddit.com//r/LocalLLaMA/comments/1g5wrjx/7xrtx3090_epyc_7003_256gb_ddr4/)): 一位用户展示了他们强大的 **7x RTX 3090** GPU 配置，搭配 **AMD Epyc 7003** 处理器和 **256GB DDR4** RAM，用于本地 LLM 推理。这种高性能配置旨在处理苛刻的 AI 工作负载，特别是大型语言模型，具有显著的并行处理能力和充足的内存资源。
  - 用户们赞扬了紧密排列的 GPU 的**美感**，有人将其比作“**NSFW**”级别的配置。**水冷**系统引起了关注，不少人询问其实现方式和散热管理。
  - **主板**被确认为拥有 **128 条 PCIe 4.0 通道**的 **ASRock ROMED8-2T**。该配置使用了 **2x1800W PSU**，并采用 **Tensor Parallelism**（张量并行）而非 NVLink 进行 GPU 通信。
  - 讨论围绕**功耗**和**散热**展开，原作者（OP）确认**每块 GPU 限制在 300W**（总计 **2100W**），并使用了“**巨大的 2x 水冷散热器**”。用户将此配置与加密货币挖矿机进行了比较，并推测其在 LLM 训练方面的性能。

**主题 2. DeepSeek 的 Janus：1.3B 多模态模型的突破**

- **[DeepSeek 发布 Janus - 一个具有图像生成能力的 1.3B 多模态模型](https://huggingface.co/deepseek-ai/Janus-1.3B)** ([分数: 389, 评论: 77](https://reddit.com//r/LocalLLaMA/comments/1g6b735/deepseek_releases_janus_a_13b_multimodal_model/)): DeepSeek 发布了 **Janus**，这是一个拥有 **13 亿参数的多模态模型**，能够同时进行**图像理解和生成**。该模型在 **Zero-shot Image Captioning**（零样本图像描述）和 **Visual Question Answering**（视觉问答）任务中表现出极具竞争力的性能，同时还具备根据文本提示**生成图像**的能力，使其成为适用于各种 AI 应用的多功能工具。
  - **Janus 框架**在保持统一的 Transformer 架构的同时，为视觉编码使用了独立的路径。这种方法增强了灵活性和性能，用户对其实现方式和潜在应用表现出浓厚兴趣。
  - 提供了一份在 Windows 本地运行 Janus 的详细**安装指南**，要求至少 **6GB VRAM** 和 NVIDIA GPU。过程包括创建虚拟环境、安装依赖项以及下载模型。
  - 用户讨论了该模型的能力，一些人报告在 **12GB VRAM 的 3060** 上运行存在问题。早期测试表明，该模型在图像构图方面表现不佳，在图像生成或视觉问答方面尚未达到 SOTA 水平。

**主题 3. Meta AI 隐藏提示词争议**

- **Meta AI 的隐藏提示词** ([分数: 302, 评论: 85](https://reddit.com//r/LocalLLaMA/comments/1g5np9i/meta_ais_hidden_prompt/)): 由 **Meta Llama 3.1** 驱动的 Meta AI 聊天机器人被发现包含一段隐藏提示词（Hidden Prompt），其中包括**访问和利用用户数据**以提供个性化回答的指令。这段通过特定查询揭示的提示词概述了整合用户信息（如**保存的事实、兴趣、位置、年龄和性别**）的准则，同时要求遵守严格的**隐私协议**，以避免在回答中明确提及使用了这些数据。
  - 用户讨论了 Meta AI 隐藏提示词带来的**惊悚感**，一些人对**隐私影响**表示担忧。另一些人则认为这是改善用户体验和避免机械化回答的常规做法。
  - 关于揭示的提示词是**幻觉（Hallucinated）**还是真实的引发了辩论。一些用户建议通过多个查询的**一致性测试**来验证其真实性，而另一些人则指出提示词的特定性是其合法性的证据。
  - 讨论涉及了**提示词的质量**，一些人批评其使用了否定陈述。其他人则为这种方法辩护，指出像 **GPT-4** 这样的大型模型可以毫无困惑地处理此类指令。

**主题 4. AI 驱动的游戏开发创新**

- **[我正在开发一款游戏，你需要通过与本地运行的机器人 NPC（Llama-3.2-3B Instruct）交谈来寻找入口密码。](https://v.redd.it/cvg1c0rniavd1)** ([Score: 87, Comments: 26](https://reddit.com//r/LocalLLaMA/comments/1g5ni7g/im_creating_a_game_where_you_need_to_find_the/)): 该帖子描述了一款正在开发中的游戏，其特色是搭载了 **Llama-3.2-3B Instruct** 的 **机器人 NPC**，并在玩家设备上本地运行。玩家必须与机器人互动以发现入口密码，AI 模型在游戏环境中实现了动态对话和谜题解决。这一实现展示了将 **large language models** 集成到交互式游戏体验中的潜力，可能为 AI 驱动的叙事和游戏机制开辟新途径。
  - 来自 **Hugging Face** 的 **Thomas Simonini** 使用 **Unity 和 LLMUnity** 开发了这个演示，采用了 **Llama-3.2-3B Instruct Q4** 进行本地处理，并使用了 **Whisper Large API**。他计划增加 **具有不同性格的多个角色**，并撰写一篇关于创建类似游戏的 [教程](https://thomassimonini.substack.com/)。
  - 讨论了游戏针对越狱（jailbreaking）尝试的安全性，并建议使用 **function calling**、将密码知识与 LLM 分离，或实施 **双机器人系统**（其中一个机器人知道密码并仅传达“是/否”的回答）等技术来改进。
  - 用户提出了游戏机制的建议，例如将对话选项与 **RPG 式的智力天赋（intelligence perks）** 挂钩，将越狱作为“易受骗”NPC 的一项功能，并建议改进 **基于单词的密码** 或历史数字参考以增强猜测体验。
- **[由本地 LLAMA 3.2 3B 或 Gemini 1.5Flash API 驱动的动态角色文字游戏原型：Mind Bender Simulator](https://i.redd.it/c8xsy4xulgvd1.png)** ([Score: 43, Comments: 7](https://reddit.com//r/LocalLLaMA/comments/1g6bwoh/prototype_of_a_textbased_game_powered_by_llama_32/)): 该帖子描述了一个名为“**Mind Bender Simulator**”的 **文字游戏** 原型，它使用本地 **LLAMA 3.2 3B** 或 **Gemini 1.5Flash API** 来创建动态角色。该游戏旨在模拟与患有 **心理健康状况** 的角色之间的互动，允许玩家参与对话并做出影响叙事和角色关系的决策。
  - 游戏概念与电影 **Sneakers** 进行了比较，用户建议了诸如语音密码验证之类的场景。开发者正在考虑添加 **虚假社交档案** 并调整图形风格以增强沉浸感。
  - 讨论探索了将 **LLMs 用于文字冒险游戏** 的潜力，建议使用提示词（prompts）来设定风格、角色信息和“房间”描述。关于模型在虚拟空间导航中保持一致性的能力也提出了疑问。
  - 用户对该项目的 **prompting techniques** 表现出兴趣，并请求获取源代码。开发者指出 **LLAMA 和 Gemini** 之间存在显著的性能差异，尤其是在非英语语言方面，并估计使用 Gemini Flash 的每场游戏环节成本可能 **低于 1 美元**。


**Theme 5. LLM API Cost and Performance Comparison Tools**

- **我制作了一个寻找最便宜/最快 LLM API 提供商的工具 —— LLM API Showdown** ([Score: 51, Comments: 25](https://reddit.com//r/LocalLLaMA/comments/1g5ol41/i_made_a_tool_to_find_the_cheapestfastest_llm_api/)): 作者创建了 **“LLM API Showdown”**，这是一个比较 **LLM API providers** 成本和性能的 Web 应用程序，访问地址为 [https://llmshowdown.vercel.app/](https://llmshowdown.vercel.app/)。该工具允许用户选择模型、优先考虑成本或速度、调整输入/输出比例，并快速找到最合适的提供商，数据来源于 **artificial analysis**。
  - 用户称赞了 **LLM API Showdown** 工具的简洁性。创作者对积极反馈表示感谢，并提到该工具旨在提供比现有类似资源更及时的信息。
  - **ArtificialAnalysis** 被强调为进行深入 LLM 比较和真实使用统计的权威来源。用户对这些全面信息的质量和免费提供感到惊讶。
  - 提到了类似的工具，包括 [Hugging Face 的 LLM 定价空间](https://huggingface.co/spaces/philschmid/llm-pricing) 和 [AgentOps-AI 的 tokencost](https://github.com/AgentOps-AI/tokencost)。创作者指出这些替代方案并不总是最新的。

## 其他 AI Subreddit 综述

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI 研究与开发**

- **Google 的 NotebookLM** 现在允许用户根据其文档**自定义 AI 生成的播客**。新功能包括调整播客长度、选择声音以及添加音乐。[来源](https://www.reddit.com/r/OpenAI/comments/1g5xvrz/notebooklm_now_lets_you_customize_its_ai_podcasts/)

- **NVIDIA** 发布了 **Sana**，这是一个全新的基础模型，据称其速度比 **Flux-dev 快 25x-100x**，同时保持了相当的质量。该代码预计将开源。[来源](https://www.reddit.com/r/StableDiffusion/comments/1g5t6p7/sana_new_foundation_model_from_nvidia/)

- 一位用户成功地**合并了两个 Stable Diffusion 模型**（Illustrious 和 Pony），这两个模型具有不同的文本编码器块，展示了模型融合技术的进展。[来源](https://www.reddit.com/r/StableDiffusion/comments/1g6500o/ive_managed_to_merge_two_models_with_very/)

**AI 应用与演示**

- 开发者为 **FLUX** 创建了一个 **LEGO LoRA**，旨在提升 AI 生成图像中乐高作品的效果。[来源](https://www.reddit.com/r/StableDiffusion/comments/1g5nnaw/better_lego_for_flux_lora_flux/)

- 一张使用 FLUX 生成的**海洋生物** AI 图像展示了该模型创建逼真神话生物的能力。[来源](https://www.reddit.com/r/StableDiffusion/comments/1g66vbk/sea_creature_using_flux/)

**机器人技术进展**

- **Unitree 的 G1 机器人**展示了令人印象深刻的能力，包括 **1.4 米的立定跳远**。该机器人身高 1.32 米，在各种动作中表现出极高的灵活性。[来源](https://www.reddit.com/r/singularity/comments/1g5ngqp/the_g1_robot_made_by_unitree_can_perform_a/)

- **Unitree G1 与 Tesla Optimus** 的对比引发了关于人形机器人进展的辩论，一些用户认为 G1 更加令人印象深刻。[来源](https://www.reddit.com/r/singularity/comments/1g5vuld/the_unimpressive_optimus_received_more_votes_than/)

**AI 伦理与社会影响**

- **Sam Altman** 对人们**适应 AI 技术带来的快速变化**的能力表示担忧。他强调需要重构社会以适应这些变化。[来源](https://www.reddit.com/r/singularity/comments/1g5ni33/sam_altman_says_the_thing_that_troubles_him_the/)

- Altman 还表示 **AGI 和核聚变应该是政府项目**，并批评了目前政府承担此类计划的能力不足。[来源](https://www.reddit.com/r/singularity/comments/1g64uyq/sam_altman_says_agi_and_fusion_should_be/)

- DeepMind 的 **Demis Hassabis** 将 AI 描述为“**定义时代的**”，预测它将解决疾病和气候变化等重大全球挑战。[来源](https://www.reddit.com/r/singularity/comments/1g5zbxe/demis_hassabis_says_it_is_wrong_to_think_of_ai_as/)

**社区讨论**

- 一位用户对 r/singularity 子版块中少数账号**发布帖子过于集中**的情况表示担忧，质疑社区观点的多样性。[来源](https://www.reddit.com/r/singularity/comments/1g5vzmg/what_do_you_think_on_the_fact_that_90_of_the/)


---

# AI Discord 综述

> 由 O1-mini 提供的总结之总结

**主题 1. 模型性能与评估**

- [**Nemotron vs. Llama：70B 模型之战**](https://github.com/microsoft/T-MAC)：工程师们讨论了 **Nemotron 70B** 与 **Llama 70B** 的**性能**和**性价比**，特别是考虑到即将推出的 **405B** 模型。
  
  - **Nvidia** 营销 Nemotron 的重点在于其**有用性（helpfulness）**，引发了关于其相对于传统以知识为中心模型的优势讨论。
- [**草莓任务让 LLM 感到棘手**](https://github.com/EleutherAI/lm-evaluation-harness)：社区批评**草莓评估任务（strawberry evaluation task）**不足以真正评估 **LLM 能力**。
  
  - 推测认为，未来的模型将进行微调，以更有效地应对这些病毒式传播的评估挑战。
- [**忠实模型还是不稳定的预测？**](https://github.com/flowersteam/LLM-Culture)：为 RAG 机器人复制**忠实度评估（Faithfulness evaluations）**的过程非常耗时，这让人们对模型的可靠性产生怀疑。
  
  - 建议使用 **Ollama** 等替代方案以加快执行速度，但这取决于硬件能力。

**主题 2. 高级训练技术**

- [**微调热潮：从 ASCII 到 RWKV**](https://huggingface.co/rwkv) 🔧：工程师们深入研究针对特定任务的 **LLM 微调**，分享了关于 **RWKV** 贡献的见解以及增强模型通用性的潜力。

- 重点在于**数据质量 (data quality)**和探索**开源架构 (open-source architectures)**以提升模型性能。
- [**RLHF vs. DPO：训练拉锯战**](https://github.com/microsoft/T-MAC)：关于使用 **Proximal Policy Optimization (PPO)** 还是 **Direct Preference Optimization (DPO)** 来进行有效的 **Reinforcement Learning from Human Feedback (RLHF)** 的争论正酣。
  
  - 受 **Anthropic 的 RLAIF** 启发的实现展示了混合来自多个模型的数据以进行稳健训练。
- [**ControlNet 的文本嵌入探戈**](https://openrouter.ai/docs/parameters-api)：为图像修改定制 **ControlNet** 需要强大的 **text embeddings**，这突显了重复数据集带来的**过拟合 (overfitting)**风险。
  
  - 用户讨论了**嵌入调整 (embedding adjustments)**，以确保在不损害模型适应性的情况下进行有效训练。

**Theme 3. 前沿工具与框架**

- [**Mojo：Python 的高速表亲**](https://www.modular.com/mojo) ⚡：**Mojo** 旨在通过其“零开销抽象 (**zero overhead abstractions**)”吸引以性能为中心的开发者，与 **C++** 等语言竞争。
  
  - 反馈强调需要更多 **API 示例**和全面的 **tensor 文档**来增强易用性。
- [**Aider AI 结对编程失误**](https://aider.chat/docs/install/pipx.html)：**Aider** 提交到错误的文件路径以及达到 **token 限制 (token limits)** 的问题引发了关于增强**文件处理**和管理大数据提交的讨论。
  
  - 解决方案包括使用 `pipx` 进行隔离安装，并设置 **token** 阈值以防止过度使用。
- [**Liger Flash Attention 节省 VRAM**](https://github.com/microsoft/BitNet)：将 **Flash Attention 2** 与 **Liger** 集成可显著减少 **VRAM**，将使用量从 **22.7 GB** 减半至 **11.7 GB**。
  
  - 成员建议配置 `liger_flash_attention: true` 等设置，以便在 **AMD** 硬件上实现最佳显存节省。

**Theme 4. 创新 AI 应用**

- [**Claude 改版：移动端与 iPad 的卓越体验**](https://x.com/alexalbert__/status/1846943479332802571)：**Claude** 应用进行了重大的 **UI** 翻新，引入了项目创建和集成聊天功能，提供更流畅的用户体验。
  
  - 用户报告导航和功能显著改进，增强了随时随地的 **AI** 交互。
- [**Capital Companion：你的 AI 交易助手**](https://capitalcompanion.ai)：**Capital Companion** 利用 **LangChain** 和 **LangGraph** 提供 **AI 驱动的投资仪表盘**，帮助用户识别**上涨趋势 (uptrends)**并优化**股票交易 (stock trading)**决策。
  
  - 功能包括**技术分析工具**和**市场情绪分析**，以获得竞争性的交易优势。
- [**DeepMind 的国际象棋大师级 Transformer**](https://x.com/Hesamation/status/1846924454309323257)：**DeepMind** 发布了一个国际象棋 **Transformer**，达到了令人印象深刻的 **ELO** **2895**，即使在**陌生的谜题 (unfamiliar puzzles)**中也展示了卓越的战略实力。
  
  - 这一里程碑挑战了关于 **LLM** 在处理未见数据时有效性的批评，突显了战略性 **AI** 的潜力。

**Theme 5. 社区与协作努力**

- [**AI 黑客松：以 2.5 万美元奖金驱动创新**](https://lu.ma/ke0rwi8n)：**Stability.ai** 和 **LAION** 等多个频道举办了 **Gen AI Hackathons**，鼓励团队开发具有丰厚奖金池的伦理 **AI 驱动的多 Agent 系统 (multi-agent systems)**。
  
  - 合作伙伴包括 **aixplain**、**Sambanova Systems** 等知名机构，营造了竞争和创新的环境。
- [**开源 AI 定义与贡献**](https://opensource.org/open-source-ai/drafts/the-open-source-ai-definition-1-0-rc1)：**Open Source AI Definition** 在社区支持下最终确定，促进了开源 **AI** 项目的标准化。
  
  - 鼓励成员为 **RWKV** 等项目做出贡献，并支持旨在推进开源 **AI** 框架的倡议。
- [**伯克利 MOOC 协作与客座讲师**](https://llmagents-learning.org/f24)：**LLM Agents MOOC** 整合了来自行业领导者如 **Denny Zhou** 和 **Shunyu Yao** 的客座演讲，通过现实世界的见解增强学习体验。
  
  - 参与者参与论坛、测验和直播，营造了协作和互动的教育环境。

---

# PART 1: High level Discord summaries

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Octopus 密码谜题探索**：用户们参与了一场幽默的探索，某个模型暗示“octopus”可能是一个潜在密码，过程中生成了各种富有创意的 Prompt。
  - 尽管尝试了包括诗意方法在内的多种策略，但最终的解锁方式仍然难以捉摸。
- **针对特定任务微调模型**：一位成员分享了基于 ASCII 艺术微调模型的经验，并幽默地指出其响应效果平平。
  - 大家的共识是，通过进一步的训练迭代，模型有望提升通用性。
- **LLM 性能评估**：对 LLM 评估方法的批评指出，strawberry 任务在衡量语言处理能力方面存在不足。
  - 有推测认为，未来的模型增强将致力于解决已知的挑战，包括走红网络的 strawberry 问题。
- **Rust 机器学习库受到关注**：讨论了机器学习从 Python 转向 **Rust** 的潜力，反映了对 **Rust** 库日益增长的兴趣。
  - 提到了 **torch-rs**、**burn** 和 **ochre** 等关键库，强调了社区学习该语言的热情。
- **发布基于 Outlines 的 SCP 生成器**：GitHub 上发布了一个利用 outlines 的新 [SCP 生成器](https://github.com/dottxt-ai/cursed/tree/main/scp)，旨在增强“cursed”项目的功能。
  - 此外，一个研究 LLM 在不同人格下生成文本的仓库链接到了关于 **Cultural evolution in populations of Large Language Models** 的论文：[LLM-Culture](https://github.com/flowersteam/LLM-Culture)。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **AI 难以把握大局**：成员们发现 AI 通常擅长修复 JSON 错误等小问题，但在大型编程项目中表现挣扎，导致其在处理复杂任务时效率较低。
  - 讨论强调了这可能会误导那些缺乏足够编程知识来应对这些局限性的初学者。
- **Python：AI 爱好者的必备技能**：参与者强调了学习 **Python** 对于 AI 兴趣爱好者的价值，并指出优质的免费资源可以媲美付费课程。
  - 此外，AI 生成的代码对于新手来说往往不可靠，这凸显了掌握基础编程技能的必要性。
- **Kwai Kolors 面临 VRAM 挑战**：用户报告称，在 Google Colab 中运行 **Kwai Kolors** 需要 **19GB VRAM**，这超出了免费层级的限制。
  - 建议恢复到原始仓库以获得更好的工具兼容性。
- **理解 ControlNet 的训练需求**：为了通过自定义 ControlNet 来修改图像，成员们指出使用 **text embeddings** 至关重要；仅更换 CLIP encoder 是不够的。
  - 他们还讨论了当数据集包含相似图像时过拟合（overfitting）的风险。
- **AWS EC2 定价见解**：关于 **AWS EC2** 定价的讨论明确了费用是根据实例运行时间按小时收取的，无论是否处于活跃使用状态。
  - 成员们注意到，使用 notebook 实例不会影响每小时的成本。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **开源 AI 定义即将定稿**：**Open Source AI Definition** 已接近完成，发布候选版本 (RC) 已在[此链接](https://opensource.org/open-source-ai/drafts/the-open-source-ai-definition-1-0-rc1)提供供大家签署。鼓励社区成员进行签署，以建立更广泛的认可。
  
  - 补充资源和 **FAQs** 请见[此处](https://hackmd.io/@opensourceinitiative/osaid-faq)以获取更多说明，签署名单见[此处](https://opensource.org/ai/endorsements)。
- **寻求 RWKV 项目贡献者**：一位来自专注于 AI 推理的初创公司的成员表示有兴趣为 **RWKV** 相关的开源项目做贡献。他们被鼓励协助进行 RWKV 第 7 版的实验，详见[此频道](https://discord.com/channels/729741769192767510/1103039376184852622)之前的讨论。
  
  - 社区特别欢迎围绕新型架构和高效推理方法论的贡献。
- **SAE 转向的挑战与局限**：关于 **Sparse Autoencoders (SAEs)** 的讨论揭示了由于高层级结构的复杂性，它们往往会错误表征特征。因此，实现准确的模型解释需要极其庞大的数据集。
  
  - 成员们强调，由于过度解读特征而导致误导性结论的情况非常频繁。
- **研究 RF 训练的噪声分布**：一场关于在随机森林 (Random Forests) 中使用正态分布作为噪声的对话展开，并提出了更好的参数化替代方案。大家一致认为应探索如 Perlin 噪声或金字塔噪声等分布，这对图像处理尤其有益。
  
  - 社区成员强调，仅靠 Gaussian 噪声不足以满足各种应用需求。
- **Huggingface Adapter 遇到冗长警告**：一名成员报告在使用带有 **Huggingface adapter** 的预训练模型时收到了冗长的警告，表明可能存在兼容性问题。警告指出存在类型不匹配，语句为：*'Repo id must be a string, not <class 'transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM'>'*。
  
  - 他们计划进一步调查此问题以寻求解决方案。

 

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Nemotron 70B 对决 Llama 70B**：在热烈的讨论中，用户对比了 **Nemotron 70B** 和 **Llama 70B** 的性能，认为 **Nvidia** 强调的是 Nemotron 的帮助性而非知识提升。
  
  - 对即将推出的 **405B** 模型的推测凸显了对各模型**性价比**的关注。
- **OpenRouter 数据政策受到审查**：社区对 **OpenRouter** 的数据政策提出了质疑，特别是用户数据的安全性。已确认禁用模型训练设置可以限制数据被用于训练。
  
  - 用户对缺少隐私政策链接表示担忧，该问题随后已得到解决。
- **GPT-4o 模型给出困惑的回复**：用户报告了 **GPT-4o-mini** 和 **GPT-4o** 回复中的偏差，因为它们错误地提到了 **GPT-3** 和 **GPT-3.5**，这是模型自我意识中的一个常见怪癖。
  
  - 专家指出，除非在 Prompt 中明确告知模型的架构，否则会出现这种偏差。
- **隐私政策链接需要关注**：用户指出 **Mistral** 和 **Together** 等供应商缺少隐私政策链接，这一问题已得到确认，并强调了提高透明度的必要性。
  
  - 供应商必须将隐私政策链接到用户协议中，以增强用户信心。
- **探索 Kuzco 作为新供应商**：由于其极具竞争力的定价模式和早期的积极反馈，关于将 **Kuzco** 纳入 LLM 供应商的讨论非常活跃。
  
  - 讨论仍在进行中，但对其产品的全面优先级排序和评估尚未开始。

 

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 自动滚动问题已解决**：据报道，**LM Studio** 自动滚动功能的近期问题已为部分用户解决，这表明所遇到的问题具有间歇性。
  
  - 用户对版本稳定性表示担忧，认为这可能会影响会话期间的用户体验。
- **ROCM 与 580s 不兼容**：关于在改装的 **16GB 580s** 上使用 **ROCM** 的咨询确认其**无法**工作，尽管它们在 AliExpress 上的价格非常实惠，约为 **$90**。
  
  - 另一位成员指出，虽然 580s 在 **OpenCL** 下表现良好，但由于 **llama.cpp** 中的弃用，支持已经恶化。
- **XEON 线程调整问题引发讨论**：一位用户注意到，可调节的 **CPU threads** 从 **0.2.31** 版本的 **0-12** 减少到 **0.3.4** 版本的 **0-6**，并表达了对 **8 threads** 的需求。
  
  - 讨论中提到了 **Settings > All** 侧边栏中用于 **CPU Thread** 调整的 Javascript 查询，强调了配置清晰度的必要性。
- **不同语言模型的性能讨论**：围绕 **Nemotron** 和 **Codestral** 等语言模型的讨论显示了褒贬不一的性能结果，用户倾向于使用更大的 70B 参数模型。
  
  - 据报道，较小的模型可靠性较低，这影响了工程师对更稳健解决方案的偏好。
- **MLX-LM 中的内存管理问题**：一个 GitHub pull request 解决了 **MLX-LM** 中的内存使用问题，该问题在 prompt 处理期间未能清除缓存。
  
  - 社区成员热切期待有关拟议修复方案的更新，以提高效率并减少内存开销。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Claude 应用提升用户体验**：**Claude** 移动应用进行了重大改版，引入了更流畅的界面和支持项目创建及集成聊天功能的新 iPad 版本。用户报告更新后的导航体验显著提升。
  
  - Alex Albert 的一条 [推文](https://x.com/alexalbert__/status/1846943479332802571?s=46) 强调了该应用的新功能，通过交互式选项增强了用户参与度。
- **聊天补全推理提供商探索**：成员们研究了各种推理提供商，建议包括 **OpenRouter** 等，重点是使用流行的开源权重模型和特殊 token 来增强聊天助手。讨论集中在这些服务的可靠性和功能上。
  
  - 参与者强调在应对现有竞争对手策略带来的挑战时，需要稳健的解决方案。
- **MotherDuck 推出集成 LLM 的 SQL**：**MotherDuck** 的新 SQL 函数允许用户直接在 SQL 中利用大语言模型，简化了数据生成和摘要。该功能承诺在不需要单独基础设施的情况下，提供更便捷的高级 AI 技术访问。
  
  - 更多详情请查看 [MotherDuck 公告](https://motherduck.com/blog/sql-llm-prompt-function-gpt-models/)。
- **DeepMind 的国际象棋 AI 展示了大师级水准**：Google **DeepMind** 推出了一款变革性的国际象棋选手，其 ELO 达到 2895，即使在不熟悉的场景中也展现了其娴熟的技术。这一表现反驳了对 LLM 在处理未见数据时有效性的批评。
  
  - 该选手在没有预先计划的情况下预测走法的能力，展示了 AI 在战略环境中的潜力。
- **Drew Houston 反思 AI 的创业潜力**：在最近的一次播客中，**Drew Houston** 分享了将 **Dropbox** 重建为数据策展关键 AI 工具的见解，重申了他认为 AI 拥有最显著创业潜力的信念。你可以在[这里](https://x.com/FanaHOVA/status/1847316954077684021)收听该集节目。
  
  - Houston 幽默地讨论了在应对 AI 领域的同时，管理一家拥有 2700 名员工的上市公司的需求。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 订阅价格差异**：用户注意到 **Perplexity** 的订阅价格各异，移动端费用为 INR **1950**，而网页端为 INR **1680**。
  
  - *对这些差异的担忧引发了关于可能取消订阅的讨论*。
- **关于 Spaces 功能的困惑**：用户对 **Spaces** 功能存在不确定性，尤其是其与默认搜索页面相比的组织方式。
  
  - 用户欣赏 Spaces 的某些方面，但发现其在移动端的功能较弱，导致评价褒贬不一。
- **API 性能受到关注**：成员对 **API** 性能变慢表示不满，特别是对于 **Pro** 用户，这影响了搜索速度。
  
  - 出现了关于这些问题是*暂时的还是与最近的更新有关*的疑问。
- **长新冠研究揭示认知影响**：最近的研究结果表明，**Long COVID** 可能导致严重的脑损伤，影响认知功能。
  
  - 正如最近的一项 [研究](https://www.perplexity.ai/page/long-covid-is-a-brain-injury-W57eub2jSTWz2VDnwvcZ3A) 中详述的那样，此类主张可能会重塑 **post-COVID recovery** 的健康策略。
- **PPLX Playground 提供更高的准确度**：分析显示，与 **PPLX API** 相比，来自 **PPLX Playground** 的响应通常具有更高的准确度。
  
  - *系统提示词（system prompts）的差异可能是导致这些准确度差异的主要原因*。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 文档需要示例**：反馈表明，虽然 [Mojo 文档](https://www.modular.com/mojo) 很好地解释了概念，但缺乏 API 条目的示例，特别是针对 Python 的示例。
  
  - 用户对包管理和缺乏原生矩阵类型表示担忧，强调了对更全面的 tensor 文档的需求。
- **Mojo 旨在优化性能开销**：团队强调 **Mojo** 旨在吸引对性能敏感的开发者，强调了与 C++ 等语言相比对“零开销抽象”（zero overhead abstractions）的需求。
  
  - 他们澄清说 Mojo 的构建是为了支持像 **NumPy** 和 **TensorFlow** 这样的高性能库。
- **转向 Mojo 面临质疑**：成员们一致认为 **Mojo** 尚未准备好投入正式使用，且可能在未来一两年内都不会稳定，这引发了对从 Python 迁移的担忧。
  
  - 一位成员指出：*“Mojo 还没准备好，而且在对我们有用的任何时间尺度内都不会准备好。”*
- **GPU 支持的现状**：**Max** 的 GPU 支持开发正在进行中，并确认了在即将到来的更新中集成 Nvidia。
  
  - 然而，关于 **Apple Metal** 支持的讨论没有给出明确答案，使其状态保持模糊。
- **探索 AI 的语言偏好**：成员们辩论了从 Python 转型的问题，指出了 **Swift** 和 **Rust** 等替代方案的优缺点，许多人由于内部熟悉度而倾向于 Swift。
  
  - 然而，也有人对 Swift 陡峭的学习曲线表示沮丧，一位用户表示：*“学习 swift 很痛苦。”*

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **使用 pipx 轻松安装 Aider**：在 Windows 上使用 `pipx` 安装 **Aider** 可以实现平滑的依赖管理，并避免项目间的版本冲突。你可以在[这里](https://aider.chat/docs/install/pipx.html)找到安装指南。
  
  - 这种方法确保了 Aider 在其独立的隔离环境中运行，减少了开发过程中的兼容性问题。
- **O1 模型引发可行性担忧**：用户提出了关于访问 **O1-preview** 的可行性和成本问题，建议通过 **ChatGPT** 进行手动工作流规划。为了明确 O1 模型处理的 Prompt，用户还强调了对配置和 dry-run 模式的关注。
  
  - 这引发了关于在使用高级模型时如何平衡效率与成本效益的讨论。
- **使用 Aider 进行结对编程战胜 Bug**：一位用户分享了他们自定义的 AI 结对编程工具，通过 Prompt 重提示（reprompting）有效地解决了 **90%** 的 Bug。他们指出 **O1-preview** 在 one-shot 解决方案中表现出色。
  
  - 成员们还讨论了模型偏好，许多人根据特定的用户需求倾向于选择 **Claude-engineer** 模型。
- **Aider 中的文件提交混淆**：有报告称 Aider 错误地提交到了 `public/css/homemenu.css` 而非正确的文件路径，导致了不可逆的错误。这引发了关于 Aider 文件处理能力透明度的问题。
  
  - 社区成员表示需要更好的保护机制和更清晰的文件处理文档。
- **Token 限制排错讨论**：参与者讨论了 Aider 达到 Token 限制的问题，特别是高 Token 计数会影响聊天历史。建议设置最大阈值以防止过度的 Token 使用。
  
  - 这一问题强调了在触发进程前确认大数据提交的重要性，以提升用户体验。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **高级语音模式令用户沮丧**：用户对 **Advanced Voice Mode** 表示不满，理由是回答含糊不清，并经常出现“我的准则禁止我谈论那个”等问题，导致用户体验不佳。
  
  - 这些反馈强调了需要更清晰的响应协议来增强用户体验。
- **Glif 工作流工具详解**：关于 **Glif** 的讨论将其与 Websim 进行了对比，强调了它在连接 AI 工具以创建工作流方面的作用。
  
  - 尽管最初被认为是一个“冷门”概念，但用户很快掌握了它作为工作流应用的实用性。
- **ChatGPT for Windows 引发期待**：成员们对 [ChatGPT for Windows](https://openai.com/chatgpt/download/) 的发布表现出极大的热情，但同时也对高级用户的访问权限表示担忧。
  
  - 目前，它仅面向 **Plus, Team, Enterprise, 和 Edu** 用户开放，引发了关于跨平台功能对等（feature parity）的讨论。
- **寻找语音 AI 工程师**：一位用户呼吁寻找可用的**语音 AI 工程师**，突显了社区在语音技术特定资源方面可能存在的缺口。
  
  - 这反映了在开发以语音为核心的 AI 应用中，对专业技能的持续需求。
- **图像生成的拼写准确度**：成员们质疑如何在图像生成输出中实现**准确的拼写**，并争论这是技术限制还是护栏（guardrail）问题。
  
  - 这一担忧说明了在 AI 生成的视觉内容中确保文本准确性所面临的挑战。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **GPU 工作：数学还是工程？**：关于 GPU 工作更多是关于**数学**还是**工程**的争论仍在继续，成员们引用了 **Amdahl's** 和 **Gustafson's laws** 来讨论并行处理器上的算法扩展。
  
  - 有观点指出，**hardware-agnostic**（硬件无关）的扩展定律对于分析硬件能力至关重要。
- **PyTorch 2.5.0 中的性能下降**：用户注意到 **tinygemm** 结合 **torch.compile** 在 **PyTorch 2.5.0** 中运行速度变慢，Token 处理速度从 **171 tok/s** 下降到 **152 tok/s**。
  
  - 这种性能退化引发了在 [GitHub issue](https://github.com) 上提交问题以进行进一步调查的呼声。
- **稀疏-密集矩阵乘法（Sparse-Dense Multiplication）的收益**：新发现表明，在 **PyTorch CUDA** 中，通过拆分密集矩阵并行进行**稀疏-密集矩阵乘法**比将其作为一个整体处理具有更好的性能，特别是当宽度 **>= 65536** 时。
  
  - 尽管大宽度下的异常情况引发了对标准矩阵操作预期的新疑问，但目前正使用 *Torch.cuda.synchronize()* 来缓解计时问题。
- **开源模型与内部发布版本的差异**：讨论显示，当前模型可能依赖于**开源重新实现**，这些实现在架构细节（如 **RMSNorm** 的插入位置）上可能与内部版本有所不同，从而引发了对其对齐性的担忧。
  
  - 此外，还讨论了在 **inference bit-packed kernels** 中使用**查找表（lookup table）**的可能性，以及对 [T-MAC](https://github.com/microsoft/T-MAC) 的探讨。
- **WebAI Summit 社交**：一位成员告知他们将参加 **WebAI Summit**，并表示有兴趣与活动中的其他成员建立联系。
  
  - 这为社区内的面对面交流提供了机会。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **MongoDB 混合搜索（Hybrid Search）助力 LlamaIndex**：MongoDB 宣布在 LlamaIndex 中支持**混合搜索**，结合了**向量搜索（vector search）**和**关键词搜索（keyword search）**以增强 AI 应用能力，详见其[公告](https://t.co/XxNNwoaW9U)。
  
  - 更多见解请参阅他们在 [Twitter](https://twitter.com/llama_index/status/1847010120796197134) 上的补充帖子。
- **Auth0 的安全 AI 应用**：Auth0 介绍了开发安全 AI 应用的方法，并展示了一个全栈开源演示应用，可在[此处](https://t.co/HvvuRQbum5)获取。
  
  - 设置过程需要 Auth0 Lab、OKTA FGA 和 OpenAI 的账号，以及用于初始化 PostgreSQL 容器的 Docker。
- **黑客松回顾：庆祝 45 个项目**：最近的黑客松吸引了 **500 多人注册**，并产生了 **45 个项目**，详细回顾请见[此处](https://t.co/v7F8b0qedF)。
  
  - 预计获胜团队将发布客座博客文章，分享他们的项目和经验。
- **忠实度评估（Faithfulness Evaluation）复现耗时过长**：据用户报告，在 RAG 机器人中复现[忠实度评估](https://docs.llamaindex.ai/en/stable/examples/evaluation/faithfulness_eval/)可能需要 15 分钟到 1 小时以上。
  
  - 其他人建议使用 [Ollama](https://ollama.com) 以获得更快的执行速度，并指出性能取决于硬件。
- **LlamaParse 处理 Word 文档失败**：一位用户在使用 LlamaParse 处理 Word 文档时遇到了解析错误，具体表现为得到了意外的图像结果而非文本。
  
  - 通过 LlamaCloud UI 上传可以正常工作，但使用 npm 包则会导致解析错误。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Bitnet 正式发布！**：社区庆祝 [Bitnet](https://github.com/microsoft/BitNet) 的发布，这是 Microsoft 开发的一个强大的 **1-bit LLMs** 推理框架，在多个硬件平台上都能提供出色的性能。
  
  - 它展示了在 M2 Ultra 上以 **6 tokens/sec** 的速度运行 **100 billion models** 的能力。
- **Liger 集成 Flash Attention 2**：用户通过在配置中设置 `liger_flash_attention: true` 以及 `sdp_attention: true`，解决了将 **Flash Attention 2** 与 Liger 集成的问题。
  
  - 分享的见解强调了验证已安装依赖项对于实现最佳内存节省的重要性。
- **实现显著的 VRAM 节省**：用户报告称实现了显著的 VRAM 减少，其中一位分享了通过正确配置 Liger，显存占用从 **22.7 GB** 降至 **11.7 GB**。
  
  - 社区建议 AMD 用户设置 `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` 以提高兼容性。
- **排查 Liger 安装问题**：一些人在训练过程中遇到了 Liger 导入挑战，导致内存使用量超出了预期。
  
  - 修改 `PYTHONPATH` 变量帮助几位成员解决了这些问题，并敦促进行彻底的安装检查。
- **轻松安装 Liger 指南**：一份分享的指南详细介绍了通过 pip 安装 Liger 的简单步骤，对 CUDA 用户特别有益。
  
  - 它还指出了配置调整的必要性，并强调了对 AMD 硬件用户至关重要的 [Liger Flash Attention 2 PR](https://github.com/linkedin/Liger-Kernel/pull/275)。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **加入 Aya 的隐身项目 (Stealth Project)**：Aya 社区邀请精通 **Arabic** 和 **Spanish** 的开发者加入一个**隐身项目**，参与者可获得**专属礼品 (swag)**。感兴趣的贡献者可以查看 [Aya server](https://discord.gg/YPNcfVJT) 参与其中。
  
  - 该倡议旨在增强 AI 领域的跨语言能力和协作努力。
- **探讨对 Gemini AI 的幻灭感**：一位成员引用了关于与 **Gemini** 讨论时的幻灭情绪，分享链接见 [此处](https://g.co/gemini/share/741e412955d9)。需要更多的声音来丰富这些关于 AI 未来的对话。
  
  - 这突显了社区围绕新兴 AI 技术的认知和发展方向进行的持续讨论。
- **RAG AMAs 未录制 - 敬请关注！**：成员们获悉 **RAG AMAs** 没有录制，因此呼吁标记课程创建者以进一步查询错过的内容。缺乏录像可能会影响社区内的知识传播。
  
  - 这引发了关于今后如何有效捕捉和分享这些活动中宝贵见解的讨论。
- **试用用户可访问所有端点**：试用用户已确认，尽管有速率限制，他们仍可以免费探索所有端点，包括数据集和 emed-jobs。对于新人来说，这是一个不受限制测试功能的绝佳机会。
  
  - 全面访问为更深入地参与和实验现有 AI 工具铺平了道路。
- **微调上下文窗口审查**：一位成员指出，微调上下文窗口限制为 **510 tokens**，远短于 rerank v3 模型的 **4k**，这引发了关于文档分块策略的疑问。需要专家的见解来最大化微调效果。
  
  - 这一限制引起了人们对微调方法中的权衡及其对模型性能影响的关注。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **利用 CoLA 实现矩阵加速**：[Compositional Linear Algebra (CoLA)](https://cola.readthedocs.io/en/latest/) 库展示了**结构感知 (structure-aware)** 操作的潜力，能够提升特征值计算和矩阵求逆等任务的速度。
  
  - 使用**分解矩阵 (decomposed matrices)** 可能会提升性能，但人们担心这种小众方法是否符合 tinygrad 的定位。
- **调整 tinygrad 的优化重点**：成员们讨论了 tinygrad 的优先级应该是**稠密矩阵 (dense matrix)** 优化，而非“组合”矩阵策略。
  
  - 大家达成共识，即避免任意内存访问的算法可以有效地集成到 tinygrad 中。
- **Windows 上的 OpenCL 设置问题**：一个 CI 失败报告了加载 OpenCL 库的问题，指出在测试启动期间缺少 `libOpenCL.so.1`。
  
  - 小组讨论了检查 CI 的 OpenCL 设置，以及在最近的 commit 中**移除 GPU=1** 的影响。
- **掌握 tinygrad 的资源**：一位成员分享了一系列**教程和学习笔记**，旨在帮助新用户有效地了解 tinygrad 的内部机制。
  
  - 从 **Beautiful MNIST 示例**开始，涵盖了不同的复杂度级别，加深了理解。
- **Jim Keller 对架构的见解**：讨论转向了 **Jim Keller** 关于 **CISC / VLIW / RISC** 架构的对话，引发了对其见解进一步探索的兴趣。
  
  - 成员们发现他与 Lex Fridman 的对话具有潜在价值，以及对硬件设计和效率的影响。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **探索 Janus：开源瑰宝**：**deepseek-ai** 的 [Janus](https://github.com/deepseek-ai/Janus?tab=readme-ov-file) 项目已在 GitHub 上线，正在寻求贡献者以增强其开发。
  
  - 其仓库概述了其目标，使其成为文本和图像处理的潜在资产。
- **寻找聊天助手的推理提供商**：一位成员正在寻找支持聊天助手补全 (completions) 的推理提供商示例，并对现有选项的可靠性提出疑问。
  
  - 他们提到了 **Anthropic** 作为一个选项，但对其性能表示怀疑。
- **关于特殊 Token 利用的辩论**：成员们讨论了在聊天模型中访问特殊 Token 的问题，特别是助手部署中缺少 **END_OF_TURN_TOKEN** 的情况。
  
  - 分享了过去的见解，并建议查阅文档以获取指导。
- **Greg Brockman 预期的回归**：据 [消息来源](https://www.theinformation.com/articles/can-greg-brockman-find-a-future-back-at-openai) 称，**Greg Brockman** 预计很快将回到 OpenAI，而在他缺席期间公司发生了变化。
  
  - 成员们讨论了他不在期间行业格局的变化。
- **指令微调 (Instruction Tuning) 依赖于数据质量**：一位成员询问了对调整语气的 LLM 进行指令微调所需的 Prompt 数量，强调**数据质量**至关重要，**1k** 个 Prompt 可能就足够了。
  
  - 这强调了在微调过程中进行严格数据管理的必要性。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **黑客松大奖引爆热情**：**Gen AI Hackathon** 邀请各团队开发 AI 系统，奖金总额超过 **$25k**。合作伙伴包括 **aixplain** 和 **Sambanova Systems**，重点关注增强人类潜能的伦理 AI 解决方案。
  
  - 此次活动旨在激发 AI 应用创新，同时鼓励参与者之间的协作。
- **创建自定义 Checkpoint 的挑战**：一名成员询问了从零开始创建模型 Checkpoint 的可行性，并指出这需要**数百万张带有注释的图像**和大量的 GPU 资源。
  
  - 另一位用户建议，调整现有模型可能比从零开始更切实际。
- **无缝图像生成的困境**：有用户反映，目前使用 **flux** 的方法在生成用于平铺的**无缝图像 (seamless images)** 时存在困难。社区强调，对于此类任务，需要专门的工具而非标准的 AI 模型。
  
  - 这表明目前在实现无缝图像输出的方法论上存在空白。
- **有限的图像选项挑战模型训练**：团队讨论了生成 **Iron Man Prime** 模型的问题，建议使用漫画艺术训练 LoRa 模型作为解决方案，因为相关图像资源有限。
  
  - Model **51** 缺乏足够的训练数据，为图像生成带来了巨大障碍。
- **采样方法引发卡通风格讨论**：成员们辩论了他们最喜欢的采样方法，其中 **dpm++2** 因在图像生成中比 Euler 具有**更好的稳定性**而受到关注。
  
  - 他们还分享了对 **pony** 和 **juggernaut** 等工具在生成卡通风格方面的偏好。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Quiz 6 现已上线！**：课程工作人员宣布 **Quiz 6** 已在网站发布，点击[此处](https://llmagents-learning.org/f24)查看。鼓励参与者及时完成以跟上进度。
  
  - *用户的反馈显示出对测验的热情*，表明它是学习体验的关键部分。
- **赶快报名！**：新参与者确认仍可通过填写此[报名表](https://forms.gle/svSoNhKcGFjxup989)加入 MOOC。这进一步激发了渴望参与的潜在学习者的热情。
  
  - 报名通道保持开放，*许多人表达了对课程内容的期待*。
- **每周直播链接即将发送**：参与者每周一将通过电子邮件收到直播链接，Discord 也会发布通知。*针对用户提出的漏发邮件问题已得到及时处理*。
  
  - 这种方式确保每个人都能了解动态并有效参与实时讨论。
- **文章作业反馈**：成员们讨论了在提交书面作业前利用社区获取反馈，以符合预期要求。他们强调在专门的 Discord 频道分享草稿，以便获得及时的建议。
  
  - *社区在完善提交内容方面的协作展示了极高的参与度*，确保了文章作业的质量。
- **会见客座演讲嘉宾**：课程将邀请 **Denny Zhou**、**Shunyu Yao** 和 **Chi Wang** 担任嘉宾并提供宝贵见解。这些行业领袖预计将通过现实世界的视角来增强学习体验。
  
  - *参与者们热切期待这些环节*，这可能会弥合理论与应用之间的差距。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Gen AI Hackathon 征集创新者**：[CreatorsCorner](https://lu.ma/ke0rwi8n) 邀请团队参加专注于 AI 驱动的多智能体 (multi-agent) 系统的黑客松，奖金超过 **$25k**。
  
  - 团队在构建安全的 AI 解决方案时应牢记**伦理影响**。
- **Pixtral 在与 Qwen2 的对比中表现不佳**：在显式内容标注 (explicit content captioning) 测试中，与 **Qwen2** 和 **L3_2** 相比，**Pixtral** 表现较差，eval loss 更高。
  
  - 该 eval 训练专门针对照片内容，突显了 **Qwen2** 相对于 Pixtral 的有效性。
- **L3_2 训练的未来计划**：一位成员计划重新审视 **L3_2** 以用于 **unsloth**，这取决于其性能改进情况。
  
  - **ms swift** 产生的不稳定结果促使在全面投入 L3_2 之前需要进行更多测试。
- **对显式内容幻觉的担忧**：讨论揭示了各种模型在显式内容标注中存在**严重的幻觉 (hallucinations)**，这是一个重大问题。
  
  - 参与者注意到 **NSFW VQA** 结果中的混乱，表明无论采用何种训练方法都面临挑战。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 上的 LRM 引起好奇**：一位用户询问了使用 **DSPy** 构建 **Language Representation Model (LRM)** 的经验，并考虑在没有先例的情况下采用标准方案。他们链接了一篇关于[替代方案的博客文章](https://www.lycee.ai/blog/drop-o1-preview-try-this-alternative)以提供更多背景。
- **LLM 应用与 Token 管理**：开发稳健的 **LLM-based** 应用需要对生成任务（特别是摘要和检索）中的 Token 使用进行严格监督。讨论指出，创作营销内容可能会导致大量的 Token 消耗。
- **GPT-4 价格创历史新低**：使用 **GPT-4** 的价格大幅下降至 **每百万输入 Token 2.5 美元** 以及 **每百万输出 Token 10 美元**。这标志着自 2023 年 3 月发布以来，每百万输入 Token 减少了 **7.5 美元**。
- **解析 ColBERTv2 训练数据**：成员们对 **ColBERTv2** 的训练示例表示困惑，指出该模型使用的是带有分数的 n-way tuples，而不是普通的 tuples。引用了一个 [GitHub 仓库](https://github.com/stanford-futuredata/ColBERT) 以进一步了解训练方法。
- **对 PATH 实现的兴趣增长**：一位成员对根据参考论文实现 **PATH** 表现出热情，并关注其与 **ColBERT** 融合的潜力。尽管对其可行性存在怀疑，但其他人承认了探索使用 **DeBERTa** 和 **MiniLM** 等模型的 cross-encoder 用法的价值。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Qwen2.5 Pull Request 提交至 GitHub**：一位成员在 PyTorch Torchtune 仓库分享了一个 [Qwen2.5 的 Pull Request](https://github.com/pytorch/torchtune/pull/1863)，旨在解决某个未指明的功能或 Bug。
  
  - *仍需补充细节*，包括全面的变更日志和测试计划，以符合项目贡献标准。
- **Torchtune 训练中的两种对立方法**：成员们讨论了是运行整个流水线，还是通过奖励模型生成偏好对（preference pairs）后再进行 PPO (Proximal Policy Optimization) 训练。
  
  - *他们指出*，完整流水线具有简单性，而使用 vLLM 等工具预生成偏好对则具有效率优势。
- **偏好对迭代的可视化**：对从 LLM 到 DPO 使用生成的偏好对进行迭代的可视化请求，表明了对更清晰训练流程的需求。
  
  - *这显示了* 对可视化训练过程中固有复杂性的兴趣。
- **Anthropic 的 RLAIF 论文见解**：讨论包括了 Anthropic 的 RLAIF 论文的应用，并提到了 TRL 如何利用 vLLM 来落实其建议。
  
  - *RLAIF 设定的先例*（即每轮训练生成新数据集，融合来自各种模型的数据）尤其值得关注。
- **Torchtune 的启动试验**：有人建议在 Torchtune 中尝试现有的 SFT (Supervised Fine-Tuning) + DPO 方案，以简化开发。
  
  - *该方法旨在* 利用 DPO 方法绕过奖励模型训练的需求，从而提高效率。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **自动化文档编辑流程**：一位成员提议通过后台代码执行来 **自动化文档编辑** 流程，旨在提高工作流效率。
  
  - 他们表示有兴趣探索社区此前利用过的其他 **深度使用案例**。
- **Aider 在 AI 生成代码方面的进展**：另一位成员指出，**Aider** 在每次更新中都越来越多地集成 **AI 生成并磨砺的代码**，显示出快速的演进。
  
  - *如果模型持续改进*，这可能会为任何解释器概念带来每日构建 (nightly build) 的方法。
- **Open Interpreter 的未来计划**：讨论揭示了对 **Open Interpreter** 潜在发展方向的好奇，特别是关于像 **Aider** 这样的 AI 驱动代码集成。
  
  - 成员们渴望了解 **Open Interpreter** 将如何利用 AI 模型开发中类似的 **增量改进**。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Capital Companion 发布 - 您的 AI 交易助手**：**Capital Companion** 是一款 AI 交易助手，利用 **LangChain** 和 **LangGraph** 构建复杂的 **Agent** 工作流，请访问 [capitalcompanion.ai](https://capitalcompanion.ai) 查看。
  
  - *“如果有人有兴趣查看或讨论使用案例，请告诉我，”* 该成员鼓励围绕平台的各项功能展开讨论。
- **面向股票的 AI 驱动投资仪表盘**：**Capital Companion** 提供 **AI 驱动的投资仪表盘**，协助用户检测**上涨趋势 (uptrends)** 并增强股票交易中的决策制定。
  
  - 核心功能包括**技术分析工具**和**市场情绪分析**，旨在为**股票投资**提供竞争优势。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **修复具有丰富功能的 Twitter/X 嵌入**：一位成员敦促大家查看一个 [Twitter/X Space](https://x.com/i/spaces/1ypKdpLNZXnKW)，内容关于如何增强 **Twitter/X** 嵌入，重点在于集成**多张图片**、**视频**、**投票**和**翻译**。
  
  - 此讨论旨在改进内容在 **Discord** 和 **Telegram** 等平台上的呈现方式，使交互更加动态。
- **通过互动工具提升参与度**：对话强调了使用**投票**和**翻译**等互动工具来增加各平台用户参与度的必要性。
  
  - 使用这些功能被视为增强内容丰富程度并吸引更广泛受众的一种方式，使讨论更具活力。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **征集 LLM 成功案例**：一位成员正在寻找展示成功 **LLM** 使用案例的仓库，包括 **Prompts**、**Models** 和 **Fine-tuning** 方法，旨在整合社区力量。
  
  - 他们提议如果现有资源不足，将启动一个新**仓库**，并强调了知识共享的必要性。
- **问答映射 (Questions-Answers Mapping) 的挑战**：同一位成员提出了一个关于在不同来源之间映射**问答**的具体使用案例，正在寻找相关示例。
  
  - 这为有类似经验的其他成员提供了一个协作途径，以贡献和分享他们的见解。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

# 第二部分：按频道详细摘要与链接

{% if medium == 'web' %}

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1296551086315933697) (324 条消息🔥🔥):

> - `Octopus 密码之谜`
> - `Fine-tuning 模型`
> - `LLM 性能评估`
> - `Strawberry 问题`
> - `Anthropic 的更新`

- **探索 Octopus 密码之谜**：用户讨论了正在进行的关于一个模型的谜题，该模型似乎暗示“octopus”或其变体可能是潜在密码，整个尝试过程充满了幽默感。
  
  - 对话揭示了为破解密码而尝试的各种策略，许多涉及诗歌和创意 Prompt，但尚未取得决定性的成功。
- **针对特定任务的 Fine-tuning 模型**：一位用户分享了他们基于 ASCII 艺术训练 Fine-tuning 模型的经验，并幽默地提到该模型只能以训练不足（undertrained）的方式进行响应。
  
  - 大家达成共识，尽管存在挑战，但通过改进和进一步的训练迭代可以产生更通用的模型。
- **LLM 的性能评估**：参与者批评了某些评估的有效性，特别强调了鉴于 LLM 处理语言的方式，Strawberry 任务是不充分的。
  
  - 几位用户推测，由于其病毒式的传播特性，新模型可能会针对包括 Strawberry 问题在内的知名挑战进行专门调整。
- **Anthropic 的频繁更新**：用户对 Anthropic 最近频繁的更新和博客文章表示好奇，同时质疑为何缺乏像 3.5 版本这样重大的新模型发布。
  
  - 讨论暗示了对这些更新是真正的创新还是仅仅对现有功能的增量补充持怀疑态度。
- **参与 Bot 开发**：一位用户展示了一个新的 Pipeline 模型，该模型利用基础模型生成日益复杂的任务，展示了 Bot 交互有趣的一面。
  
  - 回复显示了对该技术的趣味性参与，用户尝试通过各种 LLM 功能来操纵和创建引人入胜的任务。

**提到的链接**：

- [Groq Meta Llama 3.2 3B With Code Interpreter - a Hugging Face Space by diabolic6045](https://huggingface.co/spaces/diabolic6045/llama-3.2-3B-with-code-interpreter)：未找到描述
- [Kido Kidodesu GIF - KIDO KIDODESU KIDODESUOSU - Discover & Share GIFs](https://tenor.com/view/kido-kidodesu-kidodesuosu-1nicerboi-gif-18040071)：点击查看 GIF
- [Stirring Soup Food52 GIF - Stirring Soup Food52 Vegetable Soup - Discover & Share GIFs](https://tenor.com/view/stirring-soup-food52-vegetable-soup-cooking-gif-19592413)：点击查看 GIF
- [forcemultiplier/instruct-evolve-xml-3b · Hugging Face](https://huggingface.co/forcemultiplier/instruct-evolve-xml-3b)：未找到描述
- [Gandalf | Lakera – Test your prompting skills to make Gandalf reveal secret information.](https://gandalf.lakera.ai/)：诱导 Gandalf 泄露信息，亲身体验大语言模型的局限性。
- [Octopus Caracatița GIF - Octopus Caracatița - Discover & Share GIFs](https://tenor.com/view/octopus-caracati%C8%9Ba-gif-20938816)：点击查看 GIF
- [Boo GIF - Boo - Discover & Share GIFs](https://tenor.com/view/boo-gif-26155047)：点击查看 GIF
- [Canadian Pacific 2816 - Wikipedia](https://en.wikipedia.org/wiki/Canadian_Pacific_2816)：未找到描述
- [Space Balls Schwartz GIF - Space Balls Schwartz Imitate - Discover & Share GIFs](https://tenor.com/view/space-balls-schwartz-imitate-copy-mirror-gif-9494249)：点击查看 GIF
- [NOPE - a Jailbreak puzzle](https://dubesor.de/nope)：未找到描述
- [memoize dataset length for eval sample packing by bursteratom · Pull Request #1974 · axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl/pull/1974)：描述：修复了 issue#1966，其中 eval_sample_packing=True 导致多 GPU 评估卡住的问题。动机与背景：在 issue#1966 中，多 GPU 上的样本打包数据集评估...

---

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1296626310608715857) (4 messages):

> - `Rust ML Libraries` (Rust ML 库)
> - `Transition from Python to Rust` (从 Python 到 Rust 的转型)
> - `torch-rs`
> - `burn and ochre`

- **ML 中的 Python 到 Rust 转型**：*一位用户建议，虽然目前的重点是 Python*，但预计未来机器学习将转向 **Rust**。
  
  - *他们提到正在研究 Rust ML 库*，表明对该领域的兴趣日益浓厚。
- **关于 Rust ML 库的咨询**：另一位成员询问了顶级 **Rust** ML 库的推荐，特别是 **Candle** 是否表现突出。
  
  - 对 Rust 的热情显而易见，展示了扩展该编程语言知识的浓厚兴趣。
- **对 torch-rs 的探索**：一位成员询问是否有人研究过 **torch-rs**，这是一个用于机器学习的 Rust 库。
  
  - 这突显了将 **Rust** 与知名 ML 框架集成的特定兴趣。
- **分享值得关注的 Rust ML 库**：*用户提到熟悉* ***torch-rs***，*以及* ***burn*** *和* ***ochre*** *作为值得探索的库*。
  
  - 这表明了对各种 Rust 机器学习工具和框架的积极参与。

 

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/) (1 messages):

chiralcarbon: [https://arxiv.org/abs/2410.13848](https://arxiv.org/abs/2410.13848)

---

### **Nous Research AI ▷ #**[**interesting-links**](https://discord.com/channels/1053877538025386074/1132352574750728192/1296612423628886128) (3 messages):

> - `SCP generator` (SCP 生成器)
> - `LLM Culture repository` (LLM Culture 仓库)

- **使用 Outlines 的 SCP 生成器已发布**：一个利用 outlines 的新 [SCP 生成器](https://github.com/dottxt-ai/cursed/tree/main/scp) 已在 GitHub 上发布，为 'cursed' 项目的开发做出了贡献。
  
  - 该项目旨在增强 SCP 文本的生成，展示了该流派的创作潜力。
- **研究具有不同个性的 LLM**：分享了一个致力于研究由不同 **LLM** 群体生成的文本的仓库，重点关注不同的个性、任务和网络结构：[LLM-Culture](https://github.com/flowersteam/LLM-Culture)。
  
  - 该资源与关于 **Cultural evolution in populations of Large Language Models**（大语言模型群体中的文化演化）的论文相关联，为研究人员提供了宝贵的见解。

**提到的链接**：

- [cursed/scp at main · dottxt-ai/cursed](https://github.com/dottxt-ai/cursed/tree/main/scp)：通过在 GitHub 上创建账号，为 dottxt-ai/cursed 的开发做出贡献。
- [GitHub - flowersteam/LLM-Culture: Code for the "Cultural evolution in populations of Large Language Models" paper](https://github.com/flowersteam/LLM-Culture)：论文 "Cultural evolution in populations of Large Language Models" 的代码 - flowersteam/LLM-Culture

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/) (1 messages):

chiralcarbon: [https://arxiv.org/abs/2410.13848](https://arxiv.org/abs/2410.13848)

---

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1296557229604077568) (214 条消息🔥🔥):

> - `在编程中使用 AI`
> - `学习 Python`
> - `Factorio 游戏讨论`
> - `Kaggle 竞赛见解`
> - `PlandexAI 讨论`

- **AI 与编程效率**：成员们讨论了 AI 在编程中的局限性，强调 AI 往往难以在简单任务之外看到大局，这使得它们在处理复杂项目时效率较低。
  
  - 一位成员指出，虽然 AI 可以修复 JSON 错误等小问题，但它们可能会误导那些不知道如何有效编程的初学者。
- **学习 Python 的价值**：有建议认为，对于 AI 爱好者来说，学习 Python 是值得的，而且免费的在线资源可以和付费课程一样有效。
  
  - 参与者强调，AI 生成的代码对于初学者来说往往不可靠，这进一步加强了掌握基础编程技能的必要性。
- **Factorio 新 DLC 讨论**：围绕 Factorio 新 DLC 的定价展开了讨论，对于 70 美元是否合理意见不一。
  
  - 一些成员分享了与朋友共享游戏以分摊成本的策略。
- **Kaggle 竞赛澄清**：一位成员对 Kaggle 竞赛的提交要求表示困惑，讨论了提交到底需要什么。
  
  - 经澄清，他们应该仅根据提供的测试集提交结果。
- **PlandexAI 与 AI 开发工具**：对话围绕 PlandexAI 以及如何将编程任务分解为更简单的组件以改善 AI 编程结果展开。
  
  - 成员们讨论了结构化 AI 工具对增强编程过程的重要性，而不是纯粹使用 AI 进行直接代码生成。

**提到的链接**：

- [Emu3 - a Hugging Face Space by BAAI](https://huggingface.co/spaces/BAAI/Emu3)：未找到描述
- [SBI-RAG: Enhancing Math Word Problem Solving for Students through Schema-Based Instruction and Retrieval-Augmented Generation](https://arxiv.org/abs/2410.13293)：许多学生在数学应用题 (MWPs) 上感到困难，通常难以识别关键信息并选择合适的数学运算。基于模式的教学 (SBI) 是一种……
- [Reddit - Dive into anything](https://www.reddit.com/r/ArtificialInteligence/comments/1g6kkog/continuous_finetuning_working_well_as_expected/?utm_name=web3xcss)：未找到描述
- [StackLLaMA: A hands-on guide to train LLaMA with RLHF](https://huggingface.co/blog/stackllama)：未找到描述
- [GitHub - not-lain/pxia: AI library for pxia](https://github.com/not-lain/pxia)：pxia 的 AI 库。通过在 GitHub 上创建账号为 not-lain/pxia 的开发做出贡献。
- [Duvet](https://www.youtube.com/watch?v=o7fgFaXKVa0)：由 Nettwerk 提供给 YouTube。Duvet · bôa Twilight ℗ Boa Recording Limited，经 Nettwerk Music Group Inc. 独家授权。发布日期：2010-04-20。制作人……
- [GitHub - florestefano1975/comfyui-portrait-master: This node was designed to help AI image creators to generate prompts for human portraits.](https://github.com/florestefano1975/comfyui-portrait-master)：该节点旨在帮助 AI 图像创作者生成人物肖像的提示词。- florestefano1975/comfyui-portrait-master
- [not-lain (Lain)](https://huggingface.co/not-lain)：未找到描述
- [starsnatched/thinker · Datasets at Hugging Face](https://huggingface.co/datasets/starsnatched/thinker)：未找到描述

---

### **HuggingFace ▷ #**[**today-im-learning**](https://discord.com/channels/879548962464493619/898619964095860757/1296677445931634780) (3 条消息):

> - `LLM Evaluation`
> - `Finetuning Flux Models`
> - `BitNet Framework`

- **使用热门评估集评估 LLM**：一位用户表示在热门评估集上评估其 **LLM** 时遇到困难，并寻求如何获取数值结果的指导。
  
  - 他们提到，正如其 [Hugging Face 页面](https://huggingface.co/ElMater06/Llama-3.2-1B-Puredove-p)所注，他们的模型在对话方面的表现优于基础模型。
- **学习 Finetune Flux Models**：一位用户渴望学习如何 Finetune **Flux models**，并正在寻找推荐的资源。
  
  - 这一咨询表明人们对模型改进和训练技术的实际应用兴趣日益浓厚。
- **探索 BitNet Framework**：一位用户分享了对 **BitNet** 的兴趣，并提供了 1-bit LLMs 官方推理框架的 [GitHub](https://github.com/microsoft/BitNet) 链接。
  
  - 分享的链接鼓励社区进一步探索该框架的相关功能和贡献。

**提到的链接**：

- [ElMater06/Llama-3.2-1B-Puredove-p · Hugging Face](https://huggingface.co/ElMater06/Llama-3.2-1B-Puredove-p)：未找到描述
- [GitHub - microsoft/BitNet: Official inference framework for 1-bit LLMs](https://github.com/microsoft/BitNet)：1-bit LLMs 的官方推理框架。通过在 GitHub 上创建账号来为 microsoft/BitNet 的开发做出贡献。

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1296549193263091733) (1 条消息):

> - `Perplexity for Finance`
> - `Stock Research Tools`

- **Perplexity 变革金融研究**：Perplexity 现在为**金融爱好者**提供了一项功能，包括实时股票报价、历史收益报告和行业同行对比，所有这些都以**出色的 UI** 呈现。
  
  - 鼓励成员使用这一新工具*享受研究市场的乐趣*。
- **市场分析变得简单**：新的金融功能允许用户毫不费力地对**公司财务状况**进行详细分析，提升了股票研究体验。
  
  - 对于那些有兴趣紧跟金融趋势的人来说，这个工具将成为一个游戏规则改变者。

 

**提到的链接**：[来自 Perplexity (@perplexity_ai) 的推文](https://x.com/perplexity_ai/status/1846287953599123757?t=RDl45Q5xGvfjF8sIZUm4zw&s=19)：Perplexity for Finance：实时股票报价。历史收益报告。行业同行对比。公司财务详细分析。全部配备出色的 UI。享受研究市场的乐趣...

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1296631902219796510) (5 条消息):

> - `AI Content Detection Web App`
> - `Style Transfer Function`
> - `Behavioral Economics in Decision-Making`
> - `Fine-tuning and Model Merging`
> - `Cognitive Biases in Financial Crises`

- **新的 AI 内容检测 Web App 发布**：一名成员介绍了一个新项目 [AI Content Detection Web App](https://github.com/rbourgeat/airay)，该应用可以识别图像或文本是由 AI 还是人类生成的。
  
  - 他们邀请大家对项目提供反馈，并表示由于他们是首次开发此类工具，欢迎提出改进建议。
- **在新 UI 中测试风格化功能**：一名成员宣布他们正在新的用户界面中测试 **Style Transfer Function**（风格迁移功能），这标志着其开发的开始。
  
  - 这意味着正在为用户持续增强用户体验和功能。
- **行为经济学与决策洞察**：一个关于行为经济学的复杂查询探讨了 **Cognitive Biases**（认知偏差）如何影响高压环境下的决策，特别是在金融危机期间。
  
  - 讨论的关键点包括 **Loss Aversion**（损失厌恶）及其对 **Expected Utility Models**（期望效用模型）的影响，表明理性行为发生了显著改变。
- **探讨 Fine-tuning 与 Model Merging**：一名成员分享了一篇题为 [Tracking Universal Features Through Fine-Tuning and Model Merging](https://huggingface.co/papers/2410.12391) 的论文，研究特征如何通过模型适配得以延续。
  
  - 该研究重点关注在不同领域进行 **Fine-tuning** 的基础 **Transformer** 模型，并检查了跨不同语言应用的特征演变。
- **关于模仿模型的讨论**：针对模仿大型语言模型的局限性提供了反馈，强调许多模型缺乏像大型模型所使用的那种全面数据集。
  
  - 对话强调了在模型适配和特征提取方法中的挑战和相似性。

**提到的链接**：

- [Paper page - Tracking Universal Features Through Fine-Tuning and Model Merging](https://huggingface.co/papers/2410.12391)：未找到描述
- [GitHub - rbourgeat/airay: A simple AI detector (Image & Text)](https://github.com/rbourgeat/airay)：一个简单的 AI 检测器（图像和文本）。通过在 GitHub 上创建账号为 rbourgeat/airay 的开发做出贡献。
- [starsnatched/thinker · Datasets at Hugging Face](https://huggingface.co/datasets/starsnatched/thinker)：未找到描述
- [starsnatched/ThinkerGemma · Hugging Face](https://huggingface.co/starsnatched/ThinkerGemma)：未找到描述

---

### **HuggingFace ▷ #**[**reading-group**](https://discord.com/channels/879548962464493619/1156269946427428974/1296568638333124658) (11 条消息🔥):

> - `HuggingFace Reading Group`
> - `Intel Patent for Code Generation LLM`
> - `Discord Stage Channels`
> - `AI Resources for Beginners`

- **HuggingFace 阅读小组概览**：HuggingFace 服务器促进了一个 **Reading Group**（阅读小组），任何人都可以展示 AI 相关的论文，如 [GitHub 链接](https://github.com/isamu-isozaki/huggingface-reading-group) 中所述。
  
  - 该平台主要旨在支持 **HF** 开发者，促进协作和知识共享。
- **关于 Intel 代码生成 LLM 专利的讨论**：一名成员询问了关于使用 **LLM** 进行代码生成的 **Intel 专利** US20240111498A1，并分享了 [专利链接](https://patents.google.com/patent/US20240111498A1/en?q=(LLM)&country=US&after=priority:20230101&num=100)。
  
  - 该专利详细介绍了利用 **LLM** 技术生成代码的各种装置和方法，强调了其潜在应用。
- **了解 Discord Stage 频道**：一名 Discord 新手寻求关于 **Stages** 是什么的澄清，并将其与 Zoom 会议进行了比较。
  
  - 成员们解释说，**Stage Channels** 是为单向演示设计的，以防止讨论期间的干扰。
- **寻求面向初学者的 AI 资源**：一名成员请求推荐适合初学者的 **Information Hub**（信息中心），以便获得关于 AI 及其用例的结构化见解。
  
  - 这反映了新学习者对如何掌握 AI 基础知识和实际应用的兴趣日益增长。

 

**提到的链接**：[US20240111498A1 - Apparatus, Device, Method and Computer Program for Generating Code using an LLM - Google Patents](https://patents.google.com/patent/US20240111498A1/en?q=(LLM)&country=US&after=priority:20230101&num=100)：未找到描述

 

---

### **HuggingFace ▷ #**[**computer-vision**](https://discord.com/channels/879548962464493619/922424143113232404/1296661559912960041) (2 messages):

> - `Out of context object detection` (脱离语境的目标检测)
> - `Importance of context in image analysis` (语境在图像分析中的重要性)
> - `Training models for detection` (训练检测模型)
> - `Creating 'others' class` (创建 'others' 类别)

- **理解脱离语境的对象**：图像中 **out of context objects**（脱离语境的对象）的检测因环境而异，例如识别出 **cars and moving objects**（汽车和移动物体）在道路上是相关的，而树木等静态元素则不然。
  
  - 一位成员建议，“脱离语境”的定义应指导检测策略，并强调需要针对特定环境定制方法。
- **训练模型需要相关的类别**：为了进行有效的对象检测，在相关类别上训练模型至关重要；用户提议创建一个 '**others**' 类别来涵盖脱离语境的项目。
  
  - 他们表示，如果成员之间能共享关于问题设置的见解，将有助于改进训练过程。

 

---

### **HuggingFace ▷ #**[**NLP**](https://discord.com/channels/879548962464493619/922424173916196955/1296618949957517424) (6 messages):

> - `Setfit Model Logging` (Setfit 模型日志记录)
> - `Argilla Version Issues` (Argilla 版本问题)

- **排除 Setfit 模型记录到 MLflow 的故障**：一位用户表示在将 **Setfit model** 记录到 **MLflow** 时遇到困难，并寻求与此过程相关的具体示例。
  
  - 另一位成员提供了帮助，但需要确认所使用的 **Argilla version** 以确保兼容性。
- **Argilla 版本混淆**：在收到检查版本的建议后，一位用户确认他们可能正在使用旧版的 **Argilla 1.x** 代码，而不是较新的 **2.x** 版本。
  
  - 提供了导航至 [Argilla documentation](https://docs.argilla.io/latest/) 的说明，以便无缝使用更新后的功能。

 

**提到的链接**：[Argilla](https://docs.argilla.io/latest/.)：Argilla 是一个供 AI 工程师和领域专家构建高质量数据集的协作工具。

 

---

### **HuggingFace ▷ #**[**diffusion-discussions**](https://discord.com/channels/879548962464493619/1009713274113245215/1296555049660710955) (27 messages🔥):

> - `Kwai Kolors in Google Colab` (Google Colab 中的 Kwai Kolors)
> - `ControlNet training considerations` (ControlNet 训练注意事项)
> - `Renting VMs for diffusion models` (为 Diffusion 模型租用虚拟机)
> - `Instance types and pricing on AWS EC2` (AWS EC2 的实例类型和定价)

- **Kwai Kolors 在 Google Colab 中运行困难**：一位用户报告了在 Google Colab 中尝试运行 **Kwai Kolors** 时出现的错误，指出它需要大约 **19GB of VRAM**，而免费版本不支持。
  
  - 另一位用户建议使用原始仓库而不是 **diffusers**，以获得更好的兼容性。
- **ControlNet 训练需要文本嵌入**：对于训练自定义 ControlNet 以改变面部的需求，一位用户被告知将 CLIP 文本编码器替换为图像编码器是行不通的，因为训练过程依赖于文本嵌入（text embeddings）。
  
  - 讨论强调，无论使用何种嵌入，数据集中重复的面部都可能导致潜在的过拟合（overfitting）。
- **租用虚拟机的建议**：用户讨论了租用虚拟机（VM）来运行 Diffusion 模型，强调 **Amazon EC2** 是常用选择，但 **FAL** 和 **Replicate** 等选项也同样可行。
  
  - 一位用户寻求 EC2 实例类型的建议，得到的建议是实例选择取决于 VRAM 和具体的应用场景。
- **虚拟机实例的定价机制**：对于 AWS EC2，用户澄清定价是按实例开启的小时数计费的，无论其是否处于活动使用状态。
  
  - 对话指出，使用 notebook 实例不会影响每小时的费用；费用完全基于实例的运行时间（uptime）。

 

**提到的链接**：[yisol/IDM-VTON · Hugging Face](https://huggingface.co/yisol/IDM-VTON)：未找到描述。

 

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1296557498849034262) (5 条消息):

> - `Open Source AI Definition`
> - `Contributions to RWKV`
> - `Open Source AI projects`

- **Open Source AI Definition 接近完成**：**Open Source AI Definition** 已接近完成，发布候选版本（release candidate）已在[此链接](https://opensource.org/open-source-ai/drafts/the-open-source-ai-definition-1-0-rc1)发布，供公众审查和签署。鼓励成员从 1.0 版本开始签署该定义，以获得更广泛的认可。
  
  - 其他资源包括[此处](https://hackmd.io/@opensourceinitiative/osaid-faq)的 **FAQs** 以及可在[此处](https://opensource.org/ai/endorsements)找到的签署列表。
- **寻求对 RWKV 项目的贡献**：一位新成员分享了其在专注于 AI inference 的初创公司的背景，并表示有兴趣为开源项目做出贡献，特别是与 **RWKV** 相关的项目。正如[此频道](https://discord.com/channels/729741769192767510/1103039376184852622)中所讨论的，他们被鼓励协助进行有关 RWKV 第 7 版论文的实验。
  
  - 社区欢迎此类贡献，特别是关于新颖架构和高效 inference 方法论方面的贡献。
- **对 Open Source AI Definition 数据要求的担忧**：一位成员对 Open Source AI Definition 中隐含的**宽松数据要求**表示担忧。该评论指出初始草案中可能存在的差距，这些差距在制定完善的 Open Source AI 标准时可能需要解决。

 

**提到的链接**：[The Open Source AI Definition – 1.0-RC1](https://opensource.org/open-source-ai/drafts/the-open-source-ai-definition-1-0-rc1)：签署 Open Source AI Definition：让您的组织出现在宣布 1.0 版本（1.0-RC1）的新闻稿中。前言 为什么我们需要 Open Source Artificial Intelligence (AI) Open So...

 

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1296566373568352357) (168 条消息🔥🔥):

> - `SAE Steering 挑战`
> - `训练中的噪声分布`
> - `机器学习中的未来相关性 (Future-Correlation)`
> - `解释 SAE 与 Transformer 模型`
> - `提高计算效率`

- **SAE Steering 的挑战与局限性**：讨论强调，使用稀疏自编码器 (SAEs) 进行可解释性分析可能会导致误导性结论，因为特征可能无法清晰地分离相关概念。
  
  - 高层级分层关系的复杂性使得特征解释变得困难，需要海量数据集才能获得准确的模型解释。
- **研究 RF 训练的噪声分布**：成员们讨论了在随机森林 (Random Forests) 中使用正态分布作为“噪声”是否合适，并建议根据分布的仔细参数化选择替代方案。
  
  - 达成共识的是，虽然高斯噪声很常见，但在不同的应用中（特别是图像处理），Perlin 噪声或金字塔噪声等其他形式可能会提供更好的结果。
- **机器学习中未来相关性 (Future-Correlation) 的挑战**：会议指出，在模型中捕捉未来相关性具有挑战性，对于实际实现而言，具备时间边界的视角至关重要。
  
  - 研究人员讨论了尽管存在困难且需要海量数据，但建立一种稳健的方法来衡量未来相关性的必要性。
- **SAE 与 Transformer 的可解释性对比**：有人对 SAE 能够准确表示和解释 LLM 行为的隐含假设表示担忧，认为该方法缺乏实质性证据。
  
  - 批评者指出，SAE 特征的有效性可能会降低到任意神经元基准，从而质疑其与传统神经元相比的真实可解释性。
- **挑战计算效率极限**：近期模型训练速度记录的突破受到关注，这可能利用了增强效率并缩短计算时间的更新。
  
  - 成员们讨论了在采用前沿的框架 nightly 版本与维持稳定性以避免部署中的 bug 之间的权衡。

**提到的链接**：

- [Addition is All You Need for Energy-efficient Language Models](https://arxiv.org/abs/2410.00907)：大型神经网络的大部分计算都花在浮点张量乘法上。在这项工作中，我们发现浮点乘法器可以用一个高精度的整数加法器来近似……
- [Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think](https://sihyun.me/REPA/)：生成的表示对齐：训练 Diffusion Transformers 比你想象的要容易。
- [Merge to Learn: Efficiently Adding Skills to Language Models with Model Merging](https://arxiv.org/abs/2410.12937)：使通用语言模型适应新技能目前是一个昂贵的过程，随着针对新技能的新指令数据集的创建，必须重复这一过程，否则会导致模型……
- [Mimetic Initialization Helps State Space Models Learn to Recall](https://arxiv.org/abs/2410.11135)：最近的研究表明，由于状态大小相对于输入是恒定的，Mamba 等状态空间模型在基于召回的任务上明显弱于 Transformers……
- [Evaluating Open-Source Sparse Autoencoders on Disentangling Factual Knowledge in GPT-2 Small](https://arxiv.org/abs/2409.04478)：机械可解释性中一种流行的新方法是在神经元激活上训练高维稀疏自编码器 (SAEs)，并将 SAE 特征作为分析的原子单位。然而……
- [SHARED Continuous Finetuning By Rombodawg](https://docs.google.com/document/u/2/d/1OjbjU5AOz4Ftn9xHQrX3oFQGhQ6RDUuXQipnQ9gn6tU/edit?tab=t.0)：使用 Lora 和 Mergekit 实现无损的持续微调。在这篇文章中，我们将讨论如何使用 Lora 适配器和 mergekit 对开源 AI 模型进行持续微调……
- [Tweet from Keller Jordan (@kellerjordan0)](https://x.com/kellerjordan0/status/1847358578686152764)：新的 NanoGPT 训练速度记录：12.03 分钟。之前的记录：13.05 分钟。更新日志：将 PyTorch 更新至 2.5 版本。
- [gist:3e5cf8ee6701d9ae33e7d30e5406623a](https://gist.github.com/paraschopra/3e5cf8ee6701d9ae33e7d30e5406623a)：GitHub Gist：即时分享代码、笔记和片段。

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1296578515319394398) (2 条消息):

> - `Huggingface Adapter 问题`
> - `摘要任务错误`

- **Huggingface Adapter 遇到冗长警告**：一位成员报告称，在将从本地目录加载的预训练模型传递给 **Huggingface adapter** 时，收到了**冗长的警告**。
  
  - 警告指出：*'Repo id must be a string, not <class 'transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM'>'*，这表明可能存在兼容性问题。
- **摘要任务中的空响应**：另一位成员对摘要或翻译相关任务返回**空列表**表示沮丧，收到的消息为：*'resps=[], filtered_resps={}'*。
  
  - 他们表示计划进一步进行实验，以尝试解决此问题。

 

**提到的链接**：[lm-evaluation-harness/lm_eval/models/huggingface.py at 624017b7f4501638b0d5848d0f0eab2914a7fb2c · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/624017b7f4501638b0d5848d0f0eab2914a7fb2c/lm_eval/models/huggingface.py#L1362)：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness

 

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1296565347226226738) (167 条消息🔥🔥):

> - `Nemotron 70B 性能`
> - `OpenRouter 数据政策`
> - `GPT-4o 模型响应`
> - `隐私政策链接`
> - `Kuzco 作为供应商`

- **Nemotron 70B 与 Llama 的对比**：讨论中对比了 **Nemotron 70B** 和 **Llama 70B** 模型，对其性能和能力看法不一。一个值得注意的点是，**Nvidia** 并没有将 Nemotron 宣传为知识增强型模型，而是侧重于其有用性（helpfulness）。
  
  - 用户推测了即将推出的 **405B** 模型，并讨论了各种模型的**性价比**。
- **澄清 OpenRouter 数据政策**：出现了关于 **OpenRouter** 上供应商数据政策的问题，包括安全实践和用户数据的法律保障。指出关闭模型训练设置可确保请求不被用于训练，这一点已由隐私政策确认。
  
  - 用户对部分供应商缺少隐私政策链接表示担忧，随后这些问题得到了解决。
- **GPT-4o 模型响应的不一致性**：用户报告称，在询问聊天会话中使用的模型时，**GPT-4o-mini** 和 **GPT-4o** 分别返回了指向 **GPT-3** 和 **GPT-3.5** 的错误引用。这种差异是正常的，因为模型通常缺乏对其品牌和版本号的认知。
  
  - 除非特别询问其架构，否则模型提供不准确的自我引用是很常见的。
- **更新供应商隐私政策**：多位用户指出 **Mistral** 和 **Together** 等供应商缺少隐私政策链接，随后得到了确认。强调了链接隐私政策对于数据使用透明度的重要性。
  
  - 已确认供应商必须在其服务条款（ToS）中链接数据相关协议，以增强用户信心。
- **考虑将 Kuzco 作为新供应商**：由于其极具吸引力的定价模式，讨论围绕将 **Kuzco** 增加为 **Llama** 的供应商展开。初步对话正在进行中，但优先级尚未最终确定。
  
  - 参与者对新供应商表现出兴趣，同时保持对其产品的关注。

**提到的链接**：

- [Berkeley Function Calling Leaderboard V3 (又名 Berkeley Tool Calling Leaderboard V3)](https://gorilla.cs.berkeley.edu/leaderboard.html) : 未找到描述
- [Parameters API | OpenRouter](https://openrouter.ai/docs/parameters-api): 用于管理请求参数的 API
- [Kuzco | LLM Inference Network](https://kuzco.xyz/pricing): 未找到描述
- [来自 OpenAI Developers (@OpenAIDevs) 的推文](https://x.com/OpenAIDevs/status/1846972985170972923): 🔊 Chat Completions API 现在支持音频。传递文本或音频输入，然后接收文本、音频或两者的响应。https://platform.openai.com/docs/guides/audio
- [deepseek-ai/Janus-1.3B · Hugging Face](https://huggingface.co/deepseek-ai/Janus-1.3B): 未找到描述
- [OpenRouter](https://openrouter.ai/nvidia/llam): LLM 路由和市场
- [Models | OpenRouter](https://openrouter.ai/models?modality=text%2Bimage-%3Etext): 在 OpenRouter 上浏览模型
- [Limits | OpenRouter](https://openrouter.ai/docs/limits): 设置模型使用限制
- [Reddit - 深入了解任何事物](https://www.reddit.com/r/Bard/comments/1g6fhis/alternatives_to_google_ai_studio_for_15pro002/): 未找到描述
- [Models | OpenRouter](https://openrouter.ai/models?supported_parameters=tools): 在 OpenRouter 上浏览模型
- [Llama 3.1 Nemotron 70B Instruct - API, Providers, Stats](https://openrouter.ai/nvidia/llama-3.1-nemotron-70b-instruct): NVIDIA 的 Llama 3.1 Nemotron 70B 是一款旨在生成精确且有用响应的语言模型。通过 API 运行 Llama 3.1 Nemotron 70B Instruct

---

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1296550672229077012) (126 messages🔥🔥):

> - `LM Studio 自动滚动问题`
> - `ROCM 与 AMD GPU 的兼容性`
> - `不同语言模型的性能`
> - `Agent Zero AI 框架`
> - `MLX-LM 中的缓存内存管理`

- **LM Studio 自动滚动问题已解决**：用户讨论了最近 LM Studio 不再自动滚动文本的问题，有报告指出该功能现在对部分用户已恢复正常。
  
  - 有人强调该问题似乎是间歇性的，引发了对版本稳定性的质疑。
- **ROCM 与 AMD GPU 的兼容性**：一名成员询问在 LM Studio 中使用 Radeon 6700 XT 的情况，尽管之前可以使用 GPU，但现在却转向了 CPU 使用。
  
  - 其他人建议检查 LM Runtimes 设置，鼓励用户验证是否选择了正确的运行时（runtime）。
- **不同语言模型的性能**：讨论强调了 Nemotron 和 Codestral 等语言模型的不同表现，用户体验显示结果参差不齐。
  
  - 参与者分享了对 70B 参数模型的偏好，这些模型显著改善了他们的工作流，而较小的模型被认为不太可靠。
- **Agent Zero AI 框架介绍**：介绍了一个名为 Agent Zero 的新框架，它允许 AI 模型在具有自动记忆能力的开放环境中运行。
  
  - 用户对其在 Qwen 2.5 等模型驱动下提升学习和交互能力的潜力感到兴奋。
- **MLX-LM 中的内存管理担忧**：GitHub 的一个 Pull Request 解决了 MLX-LM 在 Prompt 处理期间因未能清除缓存而导致的内存占用问题。
  
  - 参与者热切关注团队对旨在纠正此类系统效率低下问题的调整建议的审查更新。

**提到的链接**：

- [LM Studio on Ryzen AI](https://lmstudio.ai/ryzenai)：在您的 PC 上运行 Llama, Mistral, Mixtral 和其他本地 LLM，充分利用 RyzenAI 硬件的卓越性能。
- [如何在您的 AMD Ryzen™ AI PC 或 Radeon 显卡上运行大语言模型 (LLM)](https://community.amd.com/t5/ai/how-to-run-a-large-language-model-llm-on-your-amd-ryzen-ai-pc-or/ba-p/670709)：您知道可以在您的 Ryzen™ AI PC 或 Radeon™ 7000 系列显卡上运行您自己的 GPT LLM AI 聊天机器人实例吗？AI 助手正迅速成为必不可少的资源...
- [Reddit - 深入了解一切](https://www.reddit.com/r/LocalLLaMA/comments/1g4dt31/new_model_llama31nemotron70binstruct/)：未找到描述
- [Reddit - 深入了解一切](https://www.reddit.com/r/ArtificialInteligence/comments/1g6kkog/continuous_finetuning_working_well_as_expected/?utm_name=web3xcss)：未找到描述
- [GitHub - frdel/agent-zero: Agent Zero AI framework](https://github.com/frdel/agent-zero)：Agent Zero AI 框架。通过在 GitHub 上创建账户为 frdel/agent-zero 的开发做出贡献。
- [bartowski/Llama-3.1-Nemotron-70B-Instruct-HF-GGUF at main](https://huggingface.co/bartowski/Llama-3.1-Nemotron-70B-Instruct-HF-GGUF/tree/main)：未找到描述
- [intfloat/multilingual-e5-large · Hugging Face](https://huggingface.co/intfloat/multilingual-e5-large)：未找到描述
- [awni 在 Prompt 处理期间清除缓存 · Pull Request #1027 · ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples/pull/1027)：关闭 #1025，详见该处的讨论/改进。

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1296635069993058314) (11 messages🔥):

> - `580 显卡上的 ROCM 支持`
> - `Xeon CPU 线程调整`
> - `改装版 580 的性能`
> - `Linux 中的占用率监控`

- **ROCM 与 580 不兼容**：一名成员询问 **ROCM** 是否适用于改装的 **16GB 580**（在 AliExpress 上售价约 **$90**），但回复明确表示**不**支持。
  
  - *一位成员提到* 580 在 **OpenCL** 中表现出色，但指出 *llama.cpp 已弃用该支持*，这进一步使其使用变得复杂。
- **v0.3.4 中的 XEON 线程调整问题**：一位用户报告可调节的 **CPU 线程**从 v0.2.31 的 **0-12** 减少到 v0.3.4 的 **0-6**，并表示更倾向于使用 **8 线程**。
  
  - 他们确认使用的是 **Linux**，并特别提到了 **Settings > All** 侧边栏中的 **CPU Thread** 调整选项。
- **使用 atop 进行性能监控**：同一位用户指出，在使用 **atop** 监控时，他们在 v0.3.4 中仅看到 **6 个线程**的高占用率，而 v0.2.31 为 **8 个线程**。
  
  - 这种线程利用率的不一致引发了对新版本性能变化的担忧。

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1296556711225725009) (56 messages🔥🔥):

> - `Claude App 更新`
> - `用于 LLM Completions 的推理提供商`
> - `用于 LLM 的 MotherDuck SQL 函数`
> - `Voyage AI 与 Embeddings`
> - `DeepMind 特级大师级国际象棋选手`

- **Claude App 发布更新**：Claude 对其移动端 App 进行了重大的设计改版，提升了用户体验，并推出了全新的 iPad App，允许用户创建项目、添加指令并在项目内进行聊天。
  
  - 用户反馈更新后的 App 导航更加流畅，使用体验更佳。
- **关于 Chat Completions 推理提供商的咨询**：一位成员表示有兴趣寻找能够使用流行的开源权重模型提供聊天助手补全（Chat Assistant Completions）的推理提供商，特别关注用户交互的特殊 Token。
  
  - 回复中包括了对 OpenRouter 等服务的建议，以及对其可靠性和功能的讨论。
- **MotherDuck 推出全新 SQL 函数**：MotherDuck 宣布推出一项集成大语言模型的新 SQL 函数，使用户能够直接在 SQL 中利用 LLM 进行数据生成和摘要。
  
  - 该函数简化了与 LLM 和 SLM 的交互，无需独立的底层架构，旨在让先进的 AI 技术更易于获取。
- **Voyage AI 与 Embeddings 探索**：Voyage AI 因其专注于 Embedding 模型而受到关注，用户讨论了尽管输入限制较小，Embedding 如何使技术写作等领域受益。
  
  - 对话探讨了其他 Embedding 选项（如 Jina AI）以及针对特定任务微调 Embedding 的潜在应用。
- **DeepMind 的国际象棋 AI 达到惊人的 ELO 评分**：Google DeepMind 开发了一款特级大师级别的 Transformer 国际象棋选手，其 ELO 评分达到了惊人的 2895 分，展示了即使在不熟悉的棋局中也能预测走法的强大能力。
  
  - 这一成就反驳了 LLM 在处理未见数据时无效的说法，展示了它们在基于策略的游戏中的潜力。

**Links mentioned**:

- [来自 Alex Albert (@alexalbert__) 的推文](https://x.com/alexalbert__/status/1847311493035086141?s=46)：我们刚刚发布了第二个 Anthropic Quickstart —— 一个由 Claude 驱动的金融数据分析师。上传电子表格、文档或金融图表，即可通过美观的界面立即获得可操作的见解...
- [来自 Perplexity (@perplexity_ai) 的推文](https://x.com/perplexity_ai/status/1846950770736091509?s=46)：推出 Internal Knowledge Search（我们最受期待的企业级功能）！首次实现通过一个产品同时搜索组织内部文件和网页...
- [Voyage AI | 首页](https://www.voyageai.com/)：Voyage AI 为搜索和检索提供尖端的 Embedding 模型和 Reranker。
- [来自 Brad Costanzo (@BradCostanzo) 的推文](https://x.com/BradCostanzo/status/1847024357769728486)：哇！@HeyGen_Official 今天刚发布了让 AI Avatar 加入 Zoom 会议并进行互动的能力。我邀请了他们的一个 AI Avatar 进入 Zoom 房间并录制了这段剪辑。是时候建立我的...
- [介绍 prompt() 函数：在 SQL 中利用 LLM 的力量！](https://motherduck.com/blog/sql-llm-prompt-function-gpt-models/)：我们通过在 SQL 中支持 Small Language Model（以及 LLM）让您的数据库更智能 | 阅读时间：6 分钟
- [来自 DeepSeek (@deepseek_ai) 的推文](https://x.com/deepseek_ai/status/1847191319464300652?s=46)：🚀 推出 Janus：一个革命性的多模态 AI 自回归框架！通过解耦视觉编码并将其统一到单个 Transformer 中，它在理解和...方面都超越了之前的模型。
- [来自 Alex Albert (@alexalbert__) 的推文](https://x.com/alexalbert__/status/1846943479332802571?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)：我们刚刚对 Claude 移动应用进行了重大的设计革新。现在使用起来非常流畅。您可以在应用内创建项目、添加自定义指令，并在项目内进行聊天...
- [评测驱动开发：更快地构建更好的 AI - Vercel](https://vercel.com/blog/eval-driven-development-build-better-ai-faster)：了解评测驱动开发（Eval-driven development）如何帮助您更快地构建更好的 AI。探索 AI 原生开发的新测试范式，并开启持续改进。
- [来自 ℏεsam (@Hesamation) 的推文](https://x.com/Hesamation/status/1846924454309323257)：Google Deepmind 训练了一个特级大师级别的 Transformer 国际象棋选手，其 ELO 达到 2895，即使在从未见过的国际象棋谜题上，在零规划的情况下，仅通过预测下一个最佳步法...
- [Requests | OpenRouter](https://openrouter.ai/docs/requests)：处理传入和传出的请求
- [来自 Jacob Matson (@matsonj) 的推文](https://x.com/matsonj/status/1847007726335152284?s=46)：你在开玩笑吗？看看这个：引用 MotherDuck (@motherduck) 我们在 SQL 中加入了 LLM，并向您展示了 MotherDuck 数据仓库中 SLM (Small Language Models) 的力量。https://mothe...
- [来自 Marc Benioff (@Benioff) 的推文](https://x.com/Benioff/status/1846714894407578068)：当你看到 Copilot 是如何交付给客户时，令人感到失望。它就是不起作用，也没有提供任何水平的准确性。Gartner 表示它到处泄露数据，而且客户...
- [来自 clem 🤗 (@ClementDelangue) 的推文](https://x.com/ClementDelangue/status/1847009885852258650)：👀👀👀
- [#ProjectTurntable | Adobe MAX Sneaks 2024 | Adobe](https://www.youtube.com/watch?v=gfct0aH2COw)：Project Turntable 技术让您以全新的方式看待 2D 矢量图形！此功能允许您在 3D 空间中旋转绘图，同时仍然...
- [anthropic-quickstarts/financial-data-analyst at main · anthropics/anthropic-quickstarts](https://github.com/anthropics/anthropic-quickstarts/tree/main/financial-data-analyst)：一系列旨在帮助开发者快速开始使用 Anthropic API 构建可部署应用程序的项目 - anthropics/anthropic-quickstarts
- [Reddit - 深入探索一切](https://www.reddit.com/r/ProlificAc/search/?q=matt+deitke&cId=fdb645f0-765d-498d-927a-585a8e006f98&iId=68cd4a54-9b45-4c73-9c65-fcc32396cb33)：未找到描述
- [Reddit - 深入探索一切](https://www.reddit.com/r/MattDeitkeStudies/)：未找到描述

---

### **Latent Space ▷ #**[**ai-announcements**](https://discord.com/channels/822583790773862470/1075282504648511499/1296876460707151942) (6 条消息):

> - `Drew Houston 的播客`
> - `AI 与 Dropbox 功能`
> - `使用 LLM 编程`
> - `关于公司规模的评论`

- **Drew Houston 讨论 AI 机遇**：在最新的播客节目中，**Drew Houston** 回顾了他过去关于 **AI** 是初创企业最大机遇的预测，并分享了他如何重塑 **Dropbox**，使其成为你“硅脑”（silicon brain）的策展层。节目链接：[Podcast](https://x.com/FanaHOVA/status/1847316954077684021)。
  
  - *在他们的卡拉 OK 房里录音真的非常有意思 (!!!)*
- **来自 Latent Space 聊天的见解**：聊天内容涵盖了诸如每年花费 **400 小时** 使用 **LLM** 编程、进入 AI 的“租用而非购买”阶段，以及 Dropbox 通过 **Dropbox Dash** 向 **AI** 转型等话题。
  
  - Houston 强调需要对抗行业内现有巨头的**“复制、捆绑、扼杀”（Copy, Bundle, Kill）**策略。
- **关于公司规模的轻松评论**：在一段幽默的交流中，一名成员针对 Houston 提到的编程时长评论道：“*一年才 400 小时？？*”。
  
  - Houston 的回应幽默地指出，管理一家拥有 **2700 名员工的上市公司** 需要不同的时间投入。
- **关于 LLM 公司的俏皮话**：另一名成员在评论 LLM 编程时长时，开玩笑说要经营一家“*2700 人的 LLM 公司*”。
  
  - 对话保持着轻松的氛围，其中一名成员澄清他们只是在开玩笑。

**提到的链接**：[来自 Alessio Fanelli (@FanaHOVA) 的推文](https://x.com/FanaHOVA/status/1847316954077684021)：7 年前 @drewhouston 告诉 @sama 初创企业最大的机遇是 AI。现在，他正在重塑 Dropbox，使其成为你“硅脑”的策展层 🧠 我们的 @latentspacepod 聊天...

---

### **Latent Space ▷ #**[**ai-in-action-club**](https://discord.com/channels/822583790773862470/1200548371715342479/1296925348990156923) (67 条消息🔥🔥):

> - `Code Diffusion 与 AST`
> - `录像可用性`
> - `对编译器课程的兴趣`
> - `代码转换技术`

- **对 Code Diffusion 的兴奋**：成员们对基于**抽象语法树 (AST)** 运行的 [Code Diffusion](https://tree-diffusion.github.io/) 表现出极大的热情，表示有兴趣将其应用于各种编程任务。
  
  - 一位成员分享道：*“考虑到我正希望将一些 Java 代码重写为 Ruby，这看起来非常有趣。”*
- **会议录制**：一位成员询问会议是否正在录制，并得到了随后会[上传](https://youtube.com/@yikesawjeez)的确认。
  
  - 即将发布的内容包括关于做“愚蠢的 AI 玩意儿”的有趣见解。
- **对编译器课程的兴趣**：讨论凸显了大家对**编译器课程**的集体兴趣，一位成员指出该学科非常硬核。
  
  - 另一位成员鼓励阅读[关于实现解释器的书](https://craftinginterpreters.com/introduction.html)，该书旨在使该主题变得易于理解且引人入胜。
- **代码转换的效率**：成员们讨论了使用 LLM 生成代码转换的效率，建议对于重构任务，**Code the Transform (CTT)** 方法更具优势。
  
  - 一位成员评论道：*“如果你要在大量文件中应用转换，使用 LLM 生成一个 transformer（转换器）可能会更有效率。”*

**提到的链接**：

- [无标题](https://tree-diffusion.github.io/)：未找到描述
- [简介 · Crafting Interpreters](https://craftinginterpreters.com/introduction.html)：未找到描述
- [Don't Transform the Code, Code the Transforms: Towards Precise Code Rewriting using LLMs](https://arxiv.org/abs/2410.08806)：用于重写、重构和优化代码的工具应该是快速且正确的。大语言模型 (LLM) 就其本质而言，并不具备这些品质。然而，仍然存在巨大的机会...
- [yikes, aw jeez, a youtube thingy](https://youtube.com/@yikesawjeez)：去看看我的 Twitter，我会做一些愚蠢的 AI 玩意儿，@yikesawjeez 也可以加入 Discord，我现在剪贴板里没有链接，但你会找到它的，我也会教你做愚蠢的 AI 玩意儿，然后我们一起...

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1296550579363119254) (96 条消息🔥🔥):

> - `Perplexity 订阅问题`
> - `关于 Spaces 功能的讨论`
> - `API 性能担忧`
> - `企业级用例`
> - `Perplexity 用户体验`

- **Perplexity 订阅价格差异**：多位用户指出移动端和网页端订阅价格存在差异，提到费用分别为 INR 1950 和 INR 1680。
  
  - 这一问题导致部分用户因额外成本考虑取消订阅 Perplexity。
- **关于 Spaces 功能的疑问**：用户对 “Spaces” 功能表示困惑，特别是与默认搜索页面相比，它缺乏 Focus 选项。
  
  - 虽然一些用户欣赏其组织方式，但发现其功能性较弱，尤其是在使用移动端 Progressive Web App 时。
- **关于 API 和搜索速度的担忧**：成员报告 API 性能和搜索速度变慢，特别是订阅了 Pro 版本的用户。
  
  - 用户提出了疑问，这究竟是一个持续存在的问题，还是与新功能和更新有关。
- **企业用途和最佳实践**：一些用户询问了 Perplexity 的企业级用途，并分享了企业相关 FAQ 和案例研究的链接。
  
  - 他们正在寻找最佳实践以及 Pro 版本与 Enterprise 版本之间的对比，特别是在 API 访问方面。
- **关于价值和产品的用户体验**：用户分享了他们在 Perplexity 和 ChatGPT 之间的偏好，指出了两者在实时信息和详细回答方面的各自优势。
  
  - 讨论还包括 Xfinity 奖励等促销活动，用户可以利用这些优惠邀请朋友加入 Perplexity。

**提到的链接**：

- [Aravind Srinivas (@AravSrinivas) 的推文](https://x.com/aravsrinivas/status/1847030982211522852?s=46)：Answer Truck 使用 FSD 从加利福尼亚州行驶了 1000 多英里到达德克萨斯州。明天它将在奥斯汀停靠，参加 Perplexity 用户见面会。地点：La Volta Pizza, Downtown Austin, 下午 1 点...
- [Silicon Valley No Revenue | Radio On Internet](https://www.youtube.com/watch?v=BzAdXyPYKo)：Pied Piper 团队与提供建议的投资人会面。
- [GitHub - pnd280/complexity: ⚡ Supercharge your Perplexity.ai](https://github.com/pnd280/complexity)：⚡ 增强你的 Perplexity.ai。通过在 GitHub 上创建账号为 pnd280/complexity 的开发做出贡献。

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1296601200040611862) (9 条消息🔥):

> - `Starlink 千兆速度计划`
> - `Seutaringkeu 见解`
> - `Photoshop 功能`
> - `长新冠研究`
> - `理解 API`

- **Starlink 千兆速度计划发布**：查看旨在增强互联网连接的新 [Starlink 千兆速度计划](https://www.perplexity.ai/page/starlink-gigabit-speed-plan-knyorEQ7SYG11t4a.dd2Ig) 的详细信息。
  
  - 该计划旨在显著提高偏远地区用户的网速。
- **探索 Seutaringkeu**：一份关于 [Seutaringkeu](https://www.perplexity.ai/search/seutaringkeu-0Abq55u4Q5aXlkg9k6xtQg#0) 的深入文档讨论了其在当前技术中的影响和相关性。
  
  - 它强调了使其成为 AI 讨论中值得关注话题的关键特性。
- **Photoshop 功能查询**：围绕 [Photoshop](https://www.perplexity.ai/search/is-the-photoshop-function-matc-uwTB5oreQQOrbE29PB6OxA) 功能的讨论引发了对特定特性的疑问。
  
  - 用户对其在创意项目中的效率分享了不同的看法。
- **长新冠研究见解**：新发现表明 [长新冠是一种脑损伤](https://www.perplexity.ai/page/long-covid-is-a-brain-injury-W57eub2jSTWz2VDnwvcZ3A)，强调了其对认知功能的严重影响。
  
  - 这项研究可能会改变健康专业人士对新冠康复策略的看法。
- **理解 API**：新分享的关于 [API](https://www.perplexity.ai/search/what-is-an-api-6HaQAJlXRGOWBgQd3L7Iyg#0) 的资源旨在澄清其用途和功能。
  
  - 这可能会使希望将 API 集成到其应用程序中的开发人员受益。

 

**提到的链接**：[YouTube](https://www.youtube.com/embed/C-NfrMyGN_Y)：未找到描述

 

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1296833527400235041) (2 messages):

> - `PPLX Playground Accuracy`
> - `PPLX API Response Differences`
> - `System Prompt Variations`

- **PPLX Playground 比 API 更准确**：一名成员询问为什么 **PPLX Playground** 中的响应似乎比来自 **PPLX API** 的响应更准确。
  
  - *System Prompt 差异* 被认为是导致准确率波动的潜在原因。
- **关于 System Prompt 影响的讨论**：另一名成员强调，两个平台之间的准确率差异可能源于不同的 **System Prompts**。
  
  - 这表明 **Prompt 的变体** 可能会显著影响 AI 生成的响应。

---

### **Modular (Mojo 🔥) ▷ #**[**general**](https://discord.com/channels/1087530497313357884/1098713601386233997/1296571671716827206) (27 messages🔥):

> - `Mojo Documentation Feedback`
> - `Mojo's Performance Focus`
> - `Building a Pythonic Interface`
> - `Tensor Implementation in Mojo`
> - `Community Engagement and Future Plans`

- **收到关于 Mojo 文档的反馈**：一位用户提供的反馈指出，虽然 Mojo 文档很好地涵盖了新概念，但缺乏像 Python 这样的 API 入口示例，而这些示例会非常有益。
  
  - 用户提出了对包管理和缺乏原生矩阵类型的担忧，强调了对 Tensor 提供全面文档的必要性。
- **Mojo 旨在提升性能**：开发团队阐明，Mojo 专注于性能，以吸引通常编写 NumPy 和 TensorFlow 等性能敏感型库的用户。
  
  - 讨论了在 Mojo 中保持“零开销抽象（zero overhead abstractions）”的重要性，强调需要超越 C++ 等传统语言的性能。
- **致力于 Pythonic 体验**：开发者承认其目标是为 Python 用户创造舒适的体验，在确保语法保持熟悉的同时，鼓励基础开发。
  
  - 在尝试吸引更广泛的 Python 社区之前，在 Mojo 中建立基础库至关重要。
- **关于 Mojo 中 Tensor 实现的讨论**：用户对 Mojo 中缺乏直接的 ndarray 等效实现表示担忧，并讨论了实现该功能的预期复杂性。
  
  - Mojo 与 Python 的关系被比作 TypeScript 与 JavaScript 的关系，并计划在 Mojo 中经过适当测试后，为 Python 提议有价值的特性。
- **呼吁社区参与和反馈**：团队鼓励用户对可能引起混淆的 API 提供反馈以增强易用性，因为许多开发者通常不阅读文档。
  
  - 强调了社区讨论在塑造语言未来方向和文档方面的重要性。

**提到的链接**：

- [Mojo 🔥: Programming language for all of AI](https://www.modular.com/mojo)：Mojo 结合了 Python 的易用性与 C 的性能，释放了 AI 硬件无与伦比的可编程性和 AI 模型的可扩展性。
- [Modular Community Q&A](https://forms.gle/MgixGyhRKcA33BS6A)：未找到描述

---

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1296850272429146183) (75 messages🔥🔥):

> - `Mojo's Compatibility` (Mojo 的兼容性)
> - `Networking in Mojo` (Mojo 中的网络编程)
> - `Transitioning from Python` (从 Python 迁移)
> - `Language Preferences` (语言偏好)
> - `Swift vs. Rust` (Swift 对比 Rust)

- **Mojo 的当前开发状态**：尽管前景广阔，但成员们认为 Mojo 尚未准备好用于严肃的生产环境，且至少在一两年内不会稳定，这影响了从 Python 迁移的潜在计划。
  
  - 一位成员指出：*“Mojo 还没达到那个水平，而且在对我们有用的时间范围内也不会达到。”*
- **Mojo 中的网络功能**：目前的观点认为 Mojo 中的 IO 和网络功能仍处于探索性设计阶段，稳定性有限。
  
  - Mojo 的网络栈正在开发中，但预计需要一段时间才能达到可用状态。
- **探索 Python 的替代方案**：关于从 Python 迁移的讨论非常激烈，重点讨论了 Swift 和 Rust 等语言的优缺点，并分享了不同的使用体验。
  
  - 对 Python 语法的担忧引发了寻找更好替代方案的讨论，许多人因为现有的内部经验而更倾向于 Swift。
- **Swift 的采用挑战**：用户对 Swift 的抽象和文档表达了一些挫败感，认为尽管它有优势，但学习曲线可能很陡峭。
  
  - 担忧包括方法缺乏清晰度以及与 Rust 相比学习 Swift 可能面临的挑战，一位用户表示：*“学习 Swift 是痛苦的。”*
- **社区对语言选择的意见**：成员们讨论了 Nim 和 Go 等各种语言，权衡了它们在 AI 场景下的用途，同时也表达了对 Go 设计的不满。
  
  - 一位成员表示：*“我们尝试过 Go，但我真的很不喜欢它，”* 这反映了在切换语言时更广泛的犹豫。

---

### **Modular (Mojo 🔥) ▷ #**[**max**](https://discord.com/channels/1087530497313357884/1212827597323509870/1296621718655340676) (2 messages):

> - `Max GPU support` (Max GPU 支持)
> - `Apple Metal` (Apple Metal)

- **Max GPU 支持正在开发中**：目前的进展表明 **Max 的 GPU 支持是 WIP（进行中）**，预计在下一次更新中包含该功能。
  
  - 目前已确认支持*最近的* Nvidia 显卡。
- **Apple Metal 支持状态尚不明确**：讨论中包含了一个关于 GPU 任务是否支持 **Apple Metal** 的查询。
  
  - 然而，对话中没有提供关于 Metal 支持的明确答案。

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1296568954327924841) (60 messages🔥🔥):

> - `Installing Aider` (安装 Aider)
> - `Using O1 Models in Aider` (在 Aider 中使用 O1 模型)
> - `Pair Programming with Aider` (使用 Aider 进行结对编程)
> - `Alternatives to Aider for UI/UX Design` (用于 UI/UX 设计的 Aider 替代方案)
> - `Durable Execution in Aider` (Aider 中的持久执行)

- **使用 pipx 轻松安装 Aider**：用户发现，在 Windows 上使用 `pipx` 安装 **Aider** 可以简化依赖管理，避免在处理多个 Python 项目时出现版本冲突。
  
  - 正如所提到的，你可以使用 `pipx` 快速安装 Aider，以确保它在自己的环境中运行，从而消除兼容性问题。
- **在 Aider 中使用 O1 模型的挑战**：一位用户对访问 **O1-preview** 的可行性和成本表示担忧，并建议使用 **ChatGPT** 手动合成计划，然后再在 Aider 中执行。
  
  - 其他人讨论了潜在的配置和工作流程，强调了空运行（dry-run）模式的重要性，以便清晰了解 O1 模型正在处理的提示词（prompts）。
- **使用 Aider 进行结对编程**：一位成员分享说，他们的自定义 AI 结对编程工具通过有效地重新提示（reprompting）解决了代码库中 90% 的 Bug，而 O1-preview 在单次（one-shot）解决方案方面表现出色。
  
  - 讨论还显示了对特定模型（如 **Claude-engineer**）进行结对编程的偏好，强调了根据用户需求进行适应的重要性。
- **UI/UX AI 设计的替代方案**：有人正在寻求创意 **UI/UX AI 设计师** 的推荐，对目前类似于标准 SaaS 产品的工具表示不满。
  
  - 一位潜在的设计师介绍了自己，表示愿意审查创意项目的简报和需求。
- **Aider 对持久执行的支持**：一位用户提出了关于 **Aider** 支持持久执行（durable execution）可能性的问题，推测这在用户 IO 边界处可能比较简单。
  
  - 这突显了社区内关于增强 Aider 功能和解决用户需求的持续讨论。

**提到的链接**：

- [使用 pipx 安装](https://aider.chat/docs/install/pipx.html)：aider 是你终端里的 AI 结对编程工具
- [安装指南](https://aider.chat/docs/install.html)：如何安装并开始使用 aider 进行结对编程。

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1296553853009006653) (25 条消息🔥):

> - `Aider 文件提交错误`
> - `Token 限制问题`
> - `Deno 与 Aider 的集成`
> - `控制 repo map 输出`
> - `安装错误`

- **Aider 提交到错误的文件路径**：Aider 错误地将更改提交到了 `public/css/homemenu.css` 而不是 `public/mobile/css/homemenu.css`，导致了不可逆的损坏，并对实际修改的文件产生了混淆。
  - 这一事件引发了对 Aider 文件处理透明度的担忧，因为它声称编辑了一个文件，实际上却修改了另一个。
- **Aider 的 Token 限制担忧**：成员们讨论了 Aider 触及 Token 限制的问题，一位用户的项目由于聊天记录中的高 Token 计数超出了上下文窗口。
  - 建议包括为聊天记录设置最大 Token 阈值以避免意外的 Token 使用，并在发送大量数据前提示确认。
- **将 Deno 与 Aider 集成**：一位用户询问是否可以通过在 NextJS 项目中使用 `/web` 命令提供 Deno 文档的 URL 来增强 Aider 的能力。
  - 他们寻求关于此方法的潜在注意事项的指导，表达了对跟上快速变化的技术的担忧。
- **修改 repo map 输出**：一位用户询问如何扩展 `--show-repo-map` 的输出，以包含所有 `*.tsx` 文件或特定路径下的文件，因为他们的项目结构比较特殊。
  - 他们对 Aider 目前确定哪些文件被视为“重要”的方法表示不满，并要求对该功能有更多控制权。
- **Aider 的安装错误**：一位用户报告了 Aider 的安装问题，在尝试运行应用程序时遇到了提示找不到 `libstdc++.so.6` 的错误。
  - 这个问题指向了潜在的配置问题，促使其他人参考 Aider 的安装文档进行故障排除。

**提到的链接**：

- [安装 aider](https://aider.chat/docs/install/install.html)：aider 是你终端里的 AI 配对编程工具
- [Token 限制](https://aider.chat/docs/troubleshooting/token-limits.html)：aider 是你终端里的 AI 配对编程工具
- [REQ: 能够设置最大 Token 阈值，并在达到该阈值时显示确认警告 · Issue #2041 · Aider-AI/aider](https://github.com/Aider-AI/aider/issues/2041)：Issue 我目前正在进行大量的调试工作，涉及启用多个嵌套堆栈跟踪。这导致大量输出发送到 API，情况变得难以控制……

---

### **aider (Paul Gauthier) ▷ #**[**links**](https://discord.com/channels/1131200896827654144/1268910919057149974/) (1 条消息):

mittens4025: [https://shchegrikovich.substack.com/p/use-prolog-to-improve-llms-reasoning](https://shchegrikovich.substack.com/p/use-prolog-to-improve-llms-reasoning)

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1296562781113286788) (25 条消息🔥):

> - `Advanced Voice Mode 问题`
> - `Glif 工作流工具`
> - `ChatGPT Windows 应用反馈`

- **Advanced Voice Mode 令用户沮丧**：用户对 **Advanced Voice Mode** 表示不满，理由是回复含糊不清，且无法打断或停止助手的回答。一位用户提到它经常用 **“我的准则阻止我讨论那个”** 来回避问题，导致令人沮丧。
- **将 Glif 理解为工作流应用**：讨论围绕 **Glif** 展开，将其比作 Websim，但用于通过连接 AI 工具的工作流来创建应用。一位用户评论说这是一个“冷门”概念，但很快就理解了。
- **对 ChatGPT Windows 应用的评价褒贬不一**：对 **ChatGPT Windows 应用** 的反馈褒贬不一，一些人喜欢它的快捷方式，而另一些人觉得它像是一个弹出的网页。一位用户幽默地给该应用打了 **“5.0 分（满分 1 分）”**，表示不满。
- **ChatGPT 各平台应用对比**：讨论对比了 Windows 应用及其 OS X 版本，一些人注意到 **Alt + Space** 快捷键提供了更好的体验。用户强调了 Windows 应用对 **@chatgpt 语法** 的支持，使其感觉功能更强大。
- **关于 AI 意识局限性的讨论**：出现了一场关于 AI 掌握细微差别能力的哲学探讨，特别是“字里行间”的含义。有人提出了关于它是否能处理虚无，或选择行动还是选择不选择的问题。

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1296607288924176415) (32 条消息🔥):

> - `ChatGPT for Windows`
> - `Voice Functionality in ChatGPT` (ChatGPT 中的语音功能)
> - `Privacy Concerns with Screen Sharing` (屏幕共享的隐私顾虑)
> - `Code Generation Issues` (代码生成问题)
> - `AI Model Performance Issues` (AI 模型性能问题)

- **ChatGPT for Windows 引发热议**：成员们对 [ChatGPT for Windows](https://openai.com/chatgpt/download/) 的发布表示兴奋，但关于 Premium 用户的访问权限细节也随之浮出水面。
  
  - 早期版本仅供 **Plus, Team, Enterprise 和 Edu** 用户使用。
- **语音功能的不确定性**：关于 Android 应用中的语音功能是否会在 Windows 版本中复现的问题被提出，但答案尚不明确。
  
  - 针对不同操作系统用户的公平性担忧也开始出现，尤其是因为 **macOS** 最初就拥有此功能。
- **AI 屏幕共享的隐私顾虑**：一位成员对使用新的桌面应用表示保留意见，担心个人身份信息 (PII) 会在无意中被共享。
  
  - 他们寻求澄清 AI 可以访问哪些特定的屏幕区域以及如何进行控制。
- **代码生成的挫败感**：一位成员报告了代码生成的问题，特别是由于代码中的错误，导致 OSPF 协议库中的 JSON 格式化出现问题。
  
  - 在表达这些挫败感时也带有一丝幽默，强调了所面临的挑战。
- **AI 模型性能下降**：几位用户注意到 ChatGPT 的性能问题，特别是 advanced voice mode 出现的随机响应，这可能与 O1 preview 的更新有关。
  
  - 其他人分享了由于 input token 限制，AI 无法在对话中回忆起之前的交互。

 

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1296803027965186148) (3 条消息):

> - `Voice AI engineering` (Voice AI 工程)
> - `Image generation spelling` (图像生成拼写)

- **寻找 Voice AI 工程师**：一位用户表示需要一名 **Voice AI 工程师**，并询问是否有具备该专业知识的人员。
  
  - 这突显了社区在语音技术资源方面可能存在的缺口。
- **图像生成中准确的单词拼写**：一位成员询问如何在图像生成中实现准确的 **单词拼写**，质疑这是技术的局限性还是安全护栏 (guard rail) 的问题。
  
  - 这引发了关于当前图像生成模型的能力和约束的重要讨论。

 

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1296803027965186148) (3 条消息):

> - `Voice AI engineers` (Voice AI 工程师)
> - `Image generation accuracy` (图像生成准确度)

- **寻找 Voice AI 工程师**：一位成员询问是否有 **Voice AI 工程师**，表示需要一名开发人员。
  
  - 这突显了社区内对 Voice AI 领域专业知识的持续需求。
- **图像生成拼写顾虑**：一位成员质疑如何在图像生成输出中实现 **准确拼写**，想知道这是局限性还是安全护栏问题。
  
  - 这引发了关于 AI 生成视觉效果及其拼写单词所面临挑战的重要讨论。

 

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1296566836946796565) (8 条消息🔥):

> - `Edge deployment 项目`
> - `采样效率低下`
> - `gemm 中的性能差异`
> - `MLX 中的 Lazy evaluation`
> - `推理速度瓶颈`

- **对各个项目领域的兴趣**：成员们讨论了**项目兴趣**的各种途径，包括 Edge deployment、训练和强化学习。
  
  - 注意到了 **local LLM integration** 与 **enterprise B2B applications** 之间的区别。
- **LLM 模式下的 Cutlass 性能问题**：一位成员提出了关于 **Cutlass kernels** 性能的担忧，在 LLM 模式下其运行效率似乎只有其他 benchmark 的一半。
  
  - 性能是使用 **nsys** 测量的，表明存在需要探索的潜在低效之处。
- **采样导致的推理速度瓶颈**：强调了采样器造成的**推理速度**瓶颈，其中 top 采样方法显著降低了处理速度，从约 250 tok/s 降至约 2.5 tok/s。
  
  - 有人建议 **numpy.choice 函数** 产生了开销，且模型大小会影响采样对性能的影响。
- **Lazy evaluation 影响性能**：一位成员更新称，**MLX 中的 lazy evaluation** 导致推理速度变慢，因为操作直到被显式调用时才会执行。
  
  - 有关此主题的更多信息可以在 **GPU mode 关于 profiling 的讲座** 和 lazy evaluation 文档中找到。

**提到的链接**：

- [Lazy Evaluation — MLX 0.19.0 documentation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)：未找到描述
- [GitHub - gpu-mode/awesomeMLSys: An ML Systems Onboarding list](https://github.com/gpu-mode/awesomeMLSys)：一个 ML Systems 入门清单。通过在 GitHub 上创建账号为 gpu-mode/awesomeMLSys 的开发做出贡献。
- [GitHub - josephmisiti/awesome-machine-learning: A curated list of awesome Machine Learning frameworks, libraries and software.](https://github.com/josephmisiti/awesome-machine-learning)：精选的优秀机器学习框架、库和软件列表。- josephmisiti/awesome-machine-learning

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/1296580059183972453) (3 条消息):

> - `讨论的意外关闭`
> - `Integer Packed Tensors 中的 Bug`
> - `Triton 中的构建错误`
> - `CMake 配置问题`

- **讨论关闭引起关注**：一位成员注意到，发起讨论的人同时也关闭了讨论，这很不寻常，并称该关闭是**非计划的**，且声称与 **Triton** 无关。
  
  - *“关闭它并说这是非计划的，这很奇怪，”* 反映了对关闭理由的怀疑。
- **Integer Packed Tensors Bug 已确认**：一位成员确认，截至 **10 月 17 日**，master 分支中仍然存在 **integer packed tensors** 的 bug，而 float 类型则不受影响。
  
  - 他们提出了一个修复方案，将循环修改为 `for k in tl.range(0, total_blocks_k, 1, num_stages=1)`，但质疑将 stages 限制为 **1** 对性能的影响。
- **构建错误难倒成员**：另一位成员报告在尝试构建 **Triton** 时遇到错误 `/usr/bin/ld: cannot find -lNVGPUIR: No such file or directory`。
  
  - 他们提供了自己的 CMake 配置命令，但在 Triton [GitHub repository](https://github.com/triton-lang/triton) 中没有找到**构建步骤**。

 

**提到的链接**：[GitHub - triton-lang/triton: Development repository for the Triton language and compiler](https://github.com/triton-lang/triton?tab=readme-ov-file#readme)：Triton 语言和编译器的开发仓库 - triton-lang/triton

 

---

### **GPU MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1296733842539020319) (11 messages🔥):

> - `Flex Attention 与 DDP 的变通方案`
> - `在 CUDA 中使用 Shared Memory`

- **Flex Attention 和 DDP 需要修复**：随着 **PyTorch 2.5** 的发布，社区讨论了在 **DDP** 中使用 **Flex Attention** 的变通方案，包括通过 `torch._dynamo.config.optimize_ddp = False` 禁用 dynamo 的 DDP 优化器。
  
  - 一位用户指出，这种变通方案会导致显著的性能损失，并强调未来需要彻底修复此问题。
- **CUDA 中的 Shared Memory 使用**：一位成员强调了在 embedding 的 backward kernel 中使用 **shared memory** 的做法，这解决了更新期间的并发访问问题。
  
  - 他们询问这种模式在 torch/cuda 集成中是否有文档记录或是否被频繁使用。

**提到的链接**：

- [[FlexAttention] Using FlexAttention with DDP complains about a "higher order optimizer" · Issue #137481 · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/137481)：🐛 Bug 描述：大家好，我遇到了类似的错误。由于隐私原因我无法发布堆栈跟踪，我想让大家关注 PyTorch Discuss 上的这个帖子……
- [pytorch/torch/_dynamo/config.py at 32f585d9346e316e554c8d9bf7548af9f62141fc · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/32f585d9346e316e554c8d9bf7548af9f62141fc/torch/_dynamo/config.py#L256-L275)：Python 中具有强 GPU 加速的 Tensors 和动态神经网络 - pytorch/pytorch
- [pytorch/aten/src/ATen/native/cuda/Embedding.cu at c3cd9939fcd05f97abc0828c29e65b8214130e12 · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/c3cd9939fcd05f97abc0828c29e65b8214130e12/aten/src/ATen/native/cuda/Embedding.cu#L94)：Python 中具有强 GPU 加速的 Tensors 和动态神经网络 - pytorch/pytorch

---

### **GPU MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/1296567019008950292) (25 messages🔥):

> - `GPU 数学 vs 工程`
> - `并行处理的 Scaling Laws`
> - `Triton 和 Tensor Cores 的使用`
> - `Triton 中的 Benchmarking`

- **理解 GPU 工作：数学还是工程？**：成员们讨论了 GPU 工作更多涉及 **数学** 还是 **工程**，并强调在并行处理器上扩展算法依赖于 **Amdahl's** 和 **Gustafson's laws** 等概念。
  
  - 有人指出，对硬件能力的分析是一种 **硬件无关 (hardware-agnostic)** 的 Scaling Law。
- **Scaling Laws 与量子计算的未来**：关于并行处理器 Scaling Laws 的讨论正在进行中，有人预测当 **量子计算机** 成为主流时，这些定律将受到更多关注。
  
  - 成员们认为，通过数学手段优化模型以减少操作量属于另一个不同的研究领域。
- **在 Triton 代码中利用 Tensor Cores**：一位用户询问如何确保其 **Triton 代码** 使用了 Tensor Cores，得到的确认是：使用 `tl.dot` 函数应该可以自动启用 Tensor Cores。
  
  - 另一位成员提供了 **Triton benchmarking 工具** 的链接，以深入了解如何衡量和优化性能。
- **Triton 中的 Benchmarking 函数**：一位成员寻求资源（特别是 **YouTube 视频**），以解释如何对 Triton kernels 进行 Benchmarking。
  
  - 他们被建议使用 `do_bench` 等工具进行运行时 Benchmarking，以及使用 **NVIDIA Nsight Compute** 等高级分析工具。

 

**提到的链接**：[triton.testing.do_bench — Triton documentation](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)：未找到描述

 

---

### **GPU MODE ▷ #**[**torchao**](https://discord.com/channels/1189498204333543425/1205223658021458100/1296860245909311521) (2 messages):

> - `性能对比`
> - `Torch 版本`

- **tinygemm + torch.compile 在 2.5.0 中变慢**：一位成员观察到，与 **2.4.1** 相比，**tinygemm** 结合 **torch.compile** 在最新版本 **2.5.0** 中运行更慢，性能从 **171 tokens/sec** 显著下降到 **152 tokens/sec**。
  
  - 这一信息凸显了速度的回归 (regression)，引发了创建 [GitHub issue](https://github.com) 并分享复现代码 (repro) 以供进一步调查的请求。
- **Torch 发布版本的性能问题**：讨论集中在 **Torch 2.4.1** 和 **2.5.0** 之间的性能差异，特别是使用 **4090 GPU** 在 **Llama2 7B** 模型上的 token 处理速度。
  
  - 速度的下降引起了用户的担忧，即这究竟是一个孤立问题，还是新版本中普遍趋势的一部分。

 

---

### **GPU MODE ▷ #**[**llmdotc**](https://discord.com/channels/1189498204333543425/1227345713348870156/1296549731371057266) (6 messages):

> - `Stable Diffusion Optimization`
> - `Inference Pipeline in C`
> - `GGML Library Limitations`

- **寻求 Diffusion 的纯 C 语言解决方案**：一位成员询问是否有类似于 **llama2.c** 但专门针对纯 C 语言实现的 Diffusion 项目。
  
  - *我只想要一个优化的推理流水线 (Inference Pipeline)*。
- **参考 C++ 版 Stable Diffusion**：另一位成员推荐了 [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)，该项目专为纯 C/C++ 环境下的 **Stable Diffusion** 和 **Flux** 设计。
  
  - 然而，有人指出该项目构建在 **GGML** 之上，并不符合最初的请求。
- **关于 GGML 抽象的讨论**：成员们讨论认为，用纯 C 语言实现整个项目很可能会导致使用许多与 **GGML** 相同的抽象。
  
  - 正如某位成员评论道：*它就是一个纯 C 语言编写的机器学习库，哈哈。*

 

**提到的链接**：[GitHub - leejet/stable-diffusion.cpp: Stable Diffusion and Flux in pure C/C++](https://github.com/leejet/stable-diffusion.cpp)：纯 C/C++ 实现的 Stable Diffusion 和 Flux。可以通过在 GitHub 上创建账号来为 leejet/stable-diffusion.cpp 的开发做出贡献。

 

---

### **GPU MODE ▷ #**[**bitnet**](https://discord.com/channels/1189498204333543425/1240586843292958790/1296643755838935040) (1 messages):

> - `Open Source Re-Implementations`
> - `T-MAC Low-Bit Inference`
> - `RMSNorm Variations`

- **开源模型可能与内部发布版本不匹配**：目前发布的模型似乎只是在运行**开源重新实现版本**，这可能在关键架构细节（如 **RMSNorm** 的插入位置）上有所不同。
  
  - *不确定这个仓库是如何处理它的*，这突显了对开源模型实现一致性的担忧。
- **探索 Bit-Packed Kernels**：大家有兴趣了解该仓库如何实现其**推理 Bit-Packed Kernels**，可能使用了**查找表 (Lookup Table)** 方法。
  
  - 提到了 [T-MAC](https://github.com/microsoft/T-MAC) 作为一个在 CPU 上进行低比特 LLM 推理的值得关注的例子。

 

**提到的链接**：[GitHub - microsoft/T-MAC: Low-bit LLM inference on CPU with lookup table](https://github.com/microsoft/T-MAC/)：基于查找表的 CPU 低比特 LLM 推理。可以通过在 GitHub 上创建账号来为 microsoft/T-MAC 的开发做出贡献。

 

---

### **GPU MODE ▷ #**[**sparsity-pruning**](https://discord.com/channels/1189498204333543425/1247663759434977453/1296779759728201848) (1 messages):

> - `Sparse-Dense Multiplication`
> - `PyTorch CUDA Performance`

- **并行处理优于批量计算**：在 **PyTorch CUDA** 的**稀疏-稠密矩阵乘法 (Sparse-Dense Multiplication)** 中有一个有趣的发现：将稠密矩阵拆分为向量并并行执行，比一次性处理整个矩阵更快，特别是当宽度 **\>= 65536** 时。
  
  - 虽然使用了 *Torch.cuda.synchronize()*，表明已考虑计时问题，但这种性能提升似乎违反直觉。
- **大宽度下的性能异常**：在**宽度达到 65536 及以上**时，进行 **CSR-Dense 乘法** 发现了性能异常，这引发了对矩阵运算典型预期的质疑。
  
  - 观察到处理较小分块时的加速现象，表明可能存在底层的优化或硬件交互机制，值得进一步调查。

 

---

### **GPU MODE ▷ #**[**webgpu**](https://discord.com/channels/1189498204333543425/1262121239044948009/) (1 messages):

fancytrevor: 如果有人在 WebAI 峰会，我就在附近转悠，能打个招呼就太酷了。

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1296569022820778015) (3 messages):

> - `MongoDB Hybrid Search`
> - `Auth0 AI Applications`
> - `Hackathon Projects`

- **MongoDB 为 LlamaIndex 引入 Hybrid Search**：MongoDB 已在 LlamaIndex 中推出对混合搜索的支持，将 **vector search**（向量搜索）与**传统关键词搜索**相结合，以发挥两种方法的优势。这一集成可以增强 AI 应用程序的能力，详见其 [公告](https://t.co/XxNNwoaW9U)。
  
  - 欲了解更多见解，请查看他们在 [Twitter](https://twitter.com/llama_index/status/1847010120796197134) 上的补充帖子。
- **Auth0 发布安全 AI 应用程序解决方案**：Auth0 正在推出一套用于构建 AI 应用程序的安全方法，并提供了一个全栈开源演示应用，点击 [此处](https://t.co/HvvuRQbum5) 查看。开发者可以通过此 [链接](https://t.co/73enoM7jmm) 获取代码。
  
  - *快速入门* 需要 Auth0 Lab、OKTA FGA 和 OpenAI 的账号，以及用于运行 PostgreSQL 容器进行设置的 Docker。
- **黑客松回顾：庆祝 45 个项目诞生**：在最近为期 3 天的黑客松中，共有 **500 多人报名**，活动结束时产出了 **45 个项目**。点击 [此处](https://t.co/v7F8b0qedF) 查看详细介绍获胜者和亮点内容的博客文章。
  
  - 请关注后续由获胜团队撰写的客座博客文章，他们将深入探讨自己的项目以及在黑客松期间分享的经验。

 

**提到的链接**：[GitHub - auth0-lab/market0: sample app about authz and AI](https://t.co/73enoM7jmm)：关于 authz 和 AI 的示例应用。通过在 GitHub 上创建账号来为 auth0-lab/market0 的开发做出贡献。

 

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1296564298746691667) (46 messages🔥):

> - `Faithfulness evaluation replication`
> - `LlamaParse failure in Docx files`
> - `Handling exceptions in workflows`
> - `Parallel function calling in workflows`
> - `Using Ollama in npx create-llama`

- **复现 Faithfulness 评估的挑战**：一位用户报告称，在他们的 RAG 机器人中复现 [Faithfulness evaluation](https://docs.llamaindex.ai/en/stable/examples/evaluation/faithfulness_eval/) 有时耗时过长，从 15 分钟到 1 小时不等。
  
  - 其他成员建议尝试使用 [Ollama](https://ollama.com) 作为一种可能更快的替代方案，并强调了硬件对性能的影响。
- **LlamaParse 处理 Word 文档的问题**：一位用户在使用 LlamaParse 功能处理 Word 文档时遇到了解析错误，出现了意外的图像结果而非文本数据。
  
  - 经过进一步测试确认，通过 LlamaCloud UI 上传可以正常工作，而使用 npm 包则会导致解析错误。
- **Workflows 中的异常处理**：关于 Workflow 中如何处理异常引发了讨论，一位用户担心尽管在 try/except 块中捕获了错误，错误似乎仍会向上冒泡。
  
  - 有人指出 `llama-index-core` 版本的变化影响了错误处理，需要进行更新以确保异常得到妥善管理。
- **在 Workflows 中利用并行函数调用**：一位用户询问在 Workflow 步骤中增加 `num_workers` 时，如何配合使用 `allow_parallel_tool_calls = True` 进行并行执行。
  
  - 成员们解释说，虽然这种设置允许并发执行，但如果工具阻塞了事件循环（event loop），可能会出现问题，并强调对于非异步（non-async）工具应使用 `asyncio.to_thread`。
- **在 create-llama 中切换到 Ollama**：一位用户询问在使用 `npx create-llama` 命令时，如何将 LLM 更改为 Ollama。
  
  - 对话强调了需要更清晰的文档或示例，来说明如何将不同的 LLM 集成到 create-llama 的设置中。

**提到的链接**：

- [Google Colab](https://colab.research.google.com/drive/1NNdFlWvkmUO4fbHqBVXELM608LqM3-OW?usp=sharing)：未找到描述
- [Raise errors in instrumentation properly by logan-markewich · Pull Request #16603 · run-llama/llama_index](https://github.com/run-llama/llama_index/pull/16603)：修复了当 Workflows 抛出错误时，instrumentation 中处理错误的问题
- [GitHub - xaac-ai/llama-artifact](https://github.com/xaac-ai/llama-artifact)：通过在 GitHub 上创建账号来为 xaac-ai/llama-artifact 的开发做出贡献。

---

### **LlamaIndex ▷ #**[**ai-discussion**](https://discord.com/channels/1059199217496772688/1100478495295017063/1296765249285918733) (1 条消息):

> - `Query Planning`
> - `LlamaIndex`
> - `Information Retrieval`
> - `Natural Language Processing`

- **查询规划增强信息检索**：一篇新文章讨论了 **Query Planning**（查询规划）对于拆解复杂查询以改进 **Information Retrieval**（信息检索）的重要性，特别是在 **Natural Language Processing**（自然语言处理）的背景下。
  
  - 文章强调，结构良好的查询对于获得**准确且相关的结果**至关重要。
- **LlamaIndex 在查询处理中的作用**：文章重点介绍了 **LlamaIndex** 作为一个强大的框架，如何帮助构建可被系统高效处理的查询。
  
  - 通过关注用户意图，LlamaIndex 确保将查询分解为更小、更易于管理的组件。

**提到的链接**：[Query Planning Workflow with LlamaIndex](https://medium.com/ai-artistry/query-planning-workflow-with-llamaindex-a-human-friendly-guide-e4c096370d92)：Ankush k Singal

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1296592346296881162) (45 条消息🔥):

> - `Bitnet Release`
> - `Liger Flash Attention Integration`
> - `VRAM Savings with Liger`
> - `Liger Installation Issues`
> - `Axolotl Configuration`

- **Bitnet 正式发布！**：社区庆祝了 [Bitnet](https://github.com/microsoft/BitNet) 的发布，这是 Microsoft 为 1-bit LLM 推出的官方推理框架，其模型性能显著，并能在各种硬件上高效运行。
  
  - 它能以令人印象深刻的速度运行 1000 亿参数模型，例如在 M2 Ultra 上达到 **6 tokens/sec**。
- **集成 Liger 的 Flash Attention 2**：为了使用 Liger 启用 Flash Attention 2，用户讨论了在配置中添加 `liger_flash_attention: true` 并确保同时包含 `sdp_attention: true`。
  
  - 参与者分享了经验，并建议检查依赖项是否已正确安装和导入，以有效利用内存节省。
- **通过 Liger 实现 VRAM 节省**：用户报告了显著的 VRAM 减少，其中一位指出通过正确设置 Liger 并启用相关标志，VRAM 从 **22.7 GB** 降至 **11.7 GB**。
  
  - 社区建议进行微调以确保兼容性，例如为 AMD 用户设置 `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`。
- **Liger 安装故障排除**：一些用户遇到了训练期间 Liger 未能正确导入的问题，导致内存占用高于预期。
  
  - 修改 `PYTHONPATH` 变量帮助部分成员使集成顺利运行，这表明需要仔细验证安装。
- **Liger 使用指南**：分享了一份简短指南，推荐了针对 CUDA 用户的 Liger 直接安装步骤（使用 pip）以及所需的配置调整。
  
  - 用户指出，对于想要利用高级注意力机制的 AMD 硬件用户，[Liger Flash Attention 2 PR](https://github.com/linkedin/Liger-Kernel/pull/275) 是必需的。

**提到的链接**：

- [GitHub - microsoft/BitNet: Official inference framework for 1-bit LLMs](https://github.com/microsoft/BitNet)：1-bit LLM 的官方推理框架。通过在 GitHub 上创建账号为 microsoft/BitNet 的开发做出贡献。
- [LinkedIn](https://github.com/linkedin/)：LinkedIn 拥有 126 个可用仓库。在 GitHub 上关注他们的代码。
- [[Kernel] Flash attention 2 by remi-or · Pull Request #275 · linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel/pull/275)：摘要：此 PR 添加了 Flash Attention 2 Triton 内核，并使用我们的 FA 内核对 SDPA 注意力层进行了 Monkey-patching。详情：该内核支持 fp16 和 bfloat16、注意力掩码、注意力...
- [[Kernel] Flash attention 2 by remi-or · Pull Request #275 · linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel/pull/275/files#diff-5497d65d59975940b1e4cea24becd9f3656f933263c16d43a7b9e8bf9533b455R143)：摘要：此 PR 添加了 Flash Attention 2 Triton 内核，并使用我们的 FA 内核对 SDPA 注意力层进行了 Monkey-patching。
- [axolotl/src/axolotl/integrations/liger/__init__.py at 67f744dc8c9564ef7a42d5df780ae53e319dca61 · NanoCode012/axolotl](https://github.com/NanoCode012/axolotl/blob/67f744dc8c9564ef7a42d5df780ae53e319dca61/src/axolotl/integrations/liger/__init__.py#L188-L189)：尽管提问。通过在 GitHub 上创建账号为 NanoCode012/axolotl 的开发做出贡献。

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1296554711503077460) (13 messages🔥):

> - `Aya 隐身项目`
> - `与 Gemini 的讨论`
> - `语言翻译实验`

- **Stealth Project with Aya Calls for Contributors**: **Aya 隐身项目招募贡献者**：社区发出了加入 Aya **stealth project** 的通用招募，目标是精通 **Arabic** 和 **Spanish** 等多种语言的开发者。
  
  - 感兴趣的参与者应加入 [Aya server](https://discord.gg/YPNcfVJT) 并标记自己以进行贡献，并因其努力获得 **exclusive swag**（专属周边）。
- **Citing Discussions to Raise Awareness**: **引用讨论以提高关注度**：一名成员匿名引用了另一名成员在与 **Gemini** 讨论中的评论，强调了 AI 领域中更广泛的幻灭感。
  
  - 讨论内容可以通过[此链接](https://g.co/gemini/share/741e412955d9)查看。
- **Language Translation Experiment with Gemini**: **与 Gemini 的语言翻译实验**：一名成员在手机上的语言测试中使用了之前的评论，发现它在翻译成三种不同外语时非常有用。
  
  - 结果记录在手机的翻译历史中，但该成员选择不分享。
- **Learning to Get Involved in AI Discussions**: **学习参与 AI 讨论**：一名成员将与 **Gemini** 的对话比作虎皮鹦鹉对着自己唧唧喳喳，暗示这只是贡献者的一个开始。
  
  - 另一名成员肯定地表示，需要严肃的 **Machine Learning** 切入点才能做出更显著的贡献。

 

**Link mentioned**: [‎Gemini - AI Discussion: Nature of LLMs, Reasoning, Future](https://g.co/gemini/share/741e412955d9): 使用 Gemini 创建

 

---

### **Cohere ▷ #**[**questions**](https://discord.com/channels/954421988141711382/1168411509542637578/1296551096633786409) (7 messages):

> - `RAG AMA 录音`
> - `Cohere Command R+ 问题`

- **RAG AMAs not recorded**: **RAG AMA 未录音**：一名成员询问 RAG AMA 是否有录音，但[已确认](https://discord.com)没有录音。
  
  - 对于进一步的咨询，鼓励成员标记其中一位课程创建者寻求帮助。
- **Issues with Cohere Command R+ 08-2024**: **Cohere Command R+ 08-2024 的问题**：多名成员报告了 OpenRouter 上 **cohere/command-r-08-2024** 模型的问题，称其产生大量错误。
  
  - 一名成员询问修复进度，而另一名成员建议通过电子邮件联系以获得更及时的回复。

 

---

### **Cohere ▷ #**[**api-discussions**](https://discord.com/channels/954421988141711382/1168578329423642786/1296663870806691852) (6 messages):

> - `试用用户访问权限`
> - `微调 Rerank 上下文窗口`

- **Trial users have access to all endpoints**: **试用用户可访问所有 Endpoint**：成员们确认，使用带有 Rate limits 的试用密钥（Trial keys）可以免费使用所有功能，包括 Datasets 和 emed-jobs 等 Endpoint。
  
  - 这确保了试用用户可以不受限制地探索全系列功能。
- **Fine-tuning rerank context window limitations**: **微调 Rerank 上下文窗口的限制**：一名成员指出，微调的上下文窗口为 **510 tokens**，与 Rerank v3 模型的 **4k** 相比显著减小。
  
  - 这引发了关于微调时文档如何进行 Chunked 的疑问，并请求微调专家的见解。

 

---

### **Cohere ▷ #**[**projects**](https://discord.com/channels/954421988141711382/1218409701339828245/1296614270074753075) (7 messages):

> - `Claude-Haiku`
> - `Prompt 效率`
> - `Toolkit 提及`
> - `快速响应`
> - `更新后的 Prompt`

- **Claude-Haiku to Claude-Instant Transition**: **Claude-Haiku 向 Claude-Instant 的过渡**：一名成员讨论了 **Claude-Haiku** 向 **Claude-Instant** 版本的过渡，强调了其与各种 Bot 的兼容性。
  
  - 他们对这次过渡表示满意，称其在任何语境下都表现良好。
- **Short Prompts Lead to Faster Responses**: **短 Prompt 带来更快响应**：一位用户注意到，他们坚持使用 **short prompts** 使得 Bot 响应速度快得多，大约只需一秒钟。
  
  - 他们幽默地将其与之前耗时更长的长 Prompt 进行了对比。
- **Inquiry about Toolkit Availability**: **关于 Toolkit 可用性的咨询**：另一名成员对 **toolkit** 上是否提供快速写作 Prompt 表示感兴趣。
  
  - 他们表现出在社区内分享新想法的热情。
- **Ordinary Prompt Achieves Remarkable Speed**: **普通 Prompt 实现惊人速度**：一位用户分享了在 Playground 中使用 **ordinary prompt** 的心得，令人惊讶的是，它可以在不牺牲质量的情况下实现快速写作。
  
  - 他们强调了这种既能保持写作质量又如此有效的 Prompt 非常罕见。
- **Updates to Prompt for Better Performance**: **更新 Prompt 以获得更好性能**：一名成员更新了他们的 Prompt，加入了“非常有效”一词以增强其性能。
  
  - 他们提到，由于 Bot 现在会寻找更好的回答，开始写作的时间略有增加，但整体生成内容的速度依然更快。

 

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1296760539401293885) (12 条消息🔥):

> - `Compositional Linear Algebra (CoLA)`
> - `OpenCL Setup Issues`
> - `Tinygrad Optimization Strategies`

- **探索用于矩阵运算的 CoLA**：一场讨论强调了 [Compositional Linear Algebra (CoLA)](https://cola.readthedocs.io/en/latest/) 库的功能，重点介绍了它在特征值计算和矩阵求逆等任务上实现**结构感知（structure-aware）**操作加速的潜力。
  
  - 成员们指出，使用**分解矩阵（decomposed matrices）**可以显著提高性能，但也质疑这种方法对于 tinygrad 来说是否过于小众。
- **关于 tinygrad GPU 支持的考量**：一位成员提出了 tinygrad 是否应该优先将**稠密矩阵（dense matrix）**优化而非“组合”矩阵操作作为基准策略的问题。
  
  - 尽管存在一些怀疑，但大家一致认为，只要算法避免任意内存访问，它们就有可能被集成到 tinygrad 中。
- **Windows 上 OpenCL 的 CI 错误**：据报告，由于导入 OpenCL 库时出现问题，导致 CI 失败，具体错误是在测试初始化期间找不到 `libOpenCL.so.1`。
  
  - 这引发了关于验证 CI 机器上 OpenCL 设置的讨论，以及在最近的 commit 中**移除 GPU=1** 所带来的影响。
- **为测试设置 OpenCL**：成员们讨论了为 Windows 测试设置 OpenCL 的必要性，以确保 CI 运行顺畅，特别是在预期运行在 GPU 上时。
  
  - 达成共识认为，需要在 CI 机器上安装所需的依赖项，以便对 OpenCL 进行正确的测试。

**提到的链接**：

- [fix: not gpu · jla524/tinygrad@46daa08](https://github.com/jla524/tinygrad/commit/46daa08e6c924c1a1fd39be2fb1e187313a9c74f)：未找到描述
- [fix: not gpu · jla524/tinygrad@46daa08](https://github.com/jla524/tinygrad/commit/46daa08e6c924c1a1)：未找到描述
- [Compositional Linear Algebra (CoLA) — CoLA 0.0.6.dev25+gd87bd36 documentation](https://cola.readthedocs.io/en/latest/)：未找到描述
- [GitHub - wilson-labs/cola: Compositional Linear Algebra](https://github.com/wilson-labs/cola)：Compositional Linear Algebra。通过在 GitHub 上创建账户来为 wilson-labs/cola 的开发做出贡献。

---

### **tinygrad (George Hotz) ▷ #**[**learn-tinygrad**](https://discord.com/channels/1068976834382925865/1070745817025106080/1296551688274182356) (18 messages🔥):

> - `Tinygrad 技能的可迁移性`
> - `Jim Keller 讨论见解`
> - `有用的 Tinygrad 资源`
> - `调试 Reinforcement Learning`
> - `MuJoCo 接口挑战`

- **Tinygrad 技能可轻松迁移至 PyTorch**：一位成员确认，从 **Tinygrad** 学到的技能可以高度迁移到其他张量库（如 **PyTorch**），并指出理解其**哲学**极大地有助于理解更复杂的系统。
  
  - *我的工作主要集中在硬件和机器人领域*，这增强了学习 Tinygrad 作为其他库基础的益处。
- **Jim Keller 的见解值得探索**：有人建议查看 **Jim Keller** 与 Lex Fridman 讨论 **CISC / VLIW / RISC** 架构的对话，以及 geohot 的见解。
  
  - 一位成员提到他们已经探索过这个话题，表示这激发了进一步讨论的兴趣。
- **学习 Tinygrad 的资源**：一位成员提供了一系列**教程和学习笔记**，以帮助新手了解 **Tinygrad** 的内部结构，并强调了从多个来源整合知识的重要性。
  
  - 他们建议从 **Beautiful MNIST 示例**开始，并在学习中应对各种**复杂度级别**。
- **调试 Reinforcement Learning 的挑战**：一位成员强调了在 **Reinforcement Learning** 中调试复杂系统的困难，由于代码和系统交互的复杂性，可能需要数月时间才能搞定。
  
  - 他们分享了一篇**调试建议文章**，总结了他们在该领域多年的经验和宝贵见解。
- **MuJoCo 安装难题**：一位成员表达了在机器上使 **MuJoCo** 正常运行的挫败感，特别是在尝试将机械臂接口与 Tinygrad 连接时，**glfw** 渲染器出现了问题。
  
  - 另一位用户建议切换到 **Isaac Sim**，它提供 **headless mode**，更适合他们的使用场景。

**提到的链接**：

- [Tinygrad 教程](https://mesozoic-egg.github.io/tinygrad-notes): 关于 Tinygrad 的教程
- [调试 Reinforcement Learning 系统](https://andyljones.com/posts/rl-debugging.html): 调试 Reinforcement Learning 实现，告别痛苦。
- [attention-is-all-you-need-pytorch/transformer at master · jadore801120/attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch/tree/master/transformer): "Attention is All You Need" 中 Transformer 模型的 PyTorch 实现。 - jadore801120/attention-is-all-you-need-pytorch
- [使用图数据库进行 RAG | OpenAI Cookbook](https://cookbook.openai.com/examples/rag_with_graph_db): 使用 OpenAI API 构建应用的开源示例和指南。浏览代码片段、高级技术和演练集合。分享你自己的示例和指南。
- [Build software better, together](https://github.com/tinygrad/tinygrad/pull/6690/files).): GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1296773552510472202) (2 messages):

> - `Janus GitHub 仓库`
> - `文本与图像处理`

- **在 GitHub 上发现 Janus**：[Janus](https://github.com/deepseek-ai/Janus?tab=readme-ov-file) 是 **deepseek-ai** 的一个开源项目，邀请贡献者参与其开发。
  
  - GitHub 页面突出了项目的目的，并附有一张链接到其 **repository** 的相关图片。
- **文本与图像处理讨论**：提到了关于在输入和输出上下文中管理 **Text+Image** 的功能，尽管细节较少。
  
  - 这引发了关于文本与视觉融合如何增强用户交互的讨论。

 

**提到的链接**：[GitHub - deepseek-ai/Janus](https://github.com/deepseek-ai/Janus?tab=readme-ov-file): 通过在 GitHub 上创建账号来为 deepseek-ai/Janus 的开发做出贡献。

 

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-questions**](https://discord.com/channels/1179127597926469703/1179208129083363358/1296560727812735006) (9 messages🔥):

> - `Inference Providers for Chat Assistants`（聊天助手的推理提供商）
> - `Special Tokens in Chat Models`（聊天模型中的特殊 Token）
> - `Pre-Filling Responses`（预填充响应）
> - `OpenRouter Assistant Prefill Feature`（OpenRouter 助手预填充功能）

- **关于聊天助手推理提供商的咨询**：一名成员正在寻求有关能够为流行的开源权重模型提供聊天助手补全（completions）的推理提供商的信息，并询问响应结构的示例。
  
  - 他们注意到 **Anthropic** 提供了类似的功能，但对其可靠性表示不确定。
- **关于特殊 Token 使用的讨论**：该成员分享了他们对访问聊天模型中使用的特定特殊 Token 的兴趣，并指出助手轮次（assistant turn）缺少 **END_OF_TURN_TOKEN**。
  
  - 另一位成员提到了过去使用这些 Token 的经验，并建议查看相关文档以寻求帮助。
- **对术语“Pre-Filling”的澄清**：一名成员对术语进行了澄清，确认正在讨论的过程被称为“**pre-filling**”。
  
  - 这一术语帮助原成员优化了寻找潜在解决方案的搜索。
- **OpenRouter 提供“Assistant Prefill”功能**：原成员了解到 **OpenRouter** 提供“Assistant Prefill”功能，尽管他们对其底层实现仍不确定。
  
  - 他们表达了希望 OpenRouter 能以他们预期的方式提供此功能的愿望。

**提到的链接**：[OpenRouter](https://openrouter.ai/docs/requests)：LLM 路由和市场

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-drama**](https://discord.com/channels/1179127597926469703/1181746144821387334/1296772436338933772) (3 messages):

> - `Garrison Lovely's behavior`（Garrison Lovely 的行为）
> - `Greg Brockman's return to OpenAI`（Greg Brockman 回归 OpenAI）
> - `Changes at OpenAI`（OpenAI 的变化）

- **Garrison Lovely 维持了他的名声**：一名成员评论说，在最近的一条推文之后，Garrison Lovely *继续保持着他混蛋的名声*。
  
  - 这一评论似乎引起了其他对他的行为有类似看法的人的共鸣。
- **Greg Brockman 预计很快回归**：据一条推文报道，OpenAI 的高管预计 [Greg Brockman 将在下个月内回归](https://www.theinformation.com/articles/can-greg-brockman-find-a-future-back-at-openai)。
  
  - 然而，值得注意的是，自他离开以来，**公司已经发生了很大变化**。

**提到的链接**：

- [Garrison Lovely (@GarrisonLovely) 的推文](https://x.com/GarrisonLovely/status/1847132206394659269)：😬
- [Stephanie Palazzolo (@steph_palazzolo) 的推文](https://x.com/steph_palazzolo/status/1847269008543727979)：好消息：高管们预计 Greg Brockman 将在下个月左右回归 OpenAI。坏消息：自他离开后，公司发生了很大变化。与 @amir 深入探讨 Greg 的领导风格及其关系...

---

### **Interconnects (Nathan Lambert) ▷ #**[**random**](https://discord.com/channels/1179127597926469703/1183121795247779910/1296589844436488254) (3 messages):

> - `Artifacts Log Utility`（Artifacts Log 的实用性）
> - `Community Engagement for Pixmo`（Pixmo 的社区参与度）
> - `Data Discovery`（数据发现）

- **Artifacts Log 被证明对团队发现很有用**：有人指出，每次查看 **artifacts log** 时，总能发现对团队成员有用的**模型或数据集**。
  
  - 这强调了有组织的信息在维持工作流效率方面的重要性。
- **Pixmo 社区驱动的标注热情**：参与 **Pixmo** 数据标注的社区非常活跃，甚至建立了一个专门的 [Reddit 社区](https://www.reddit.com/r/ProlificAc/search/?q=matt+deitke)，成员们在其中分享梗图并积极要求更多工作。
  
  - 这展示了社区对标注过程的兴奋程度和参与度。

**提到的链接**：

- [Reddit - 探索一切](https://www.reddit.com/r/ProlificAc/search/?q=matt+deitke&cId=fdb645f0-765d-498d-927a-585a8e006f98&iId=68cd4a54-9b45-4c73-9c65-fcc32396cb33)：未找到描述
- [Reddit - 探索一切](https://www.reddit.com/r/MattDeitkeStudies/)：未找到描述

---

### **Interconnects (Nathan Lambert) ▷ #**[**rlhf**](https://discord.com/channels/1179127597926469703/1208183230608576562/1296637954562854955) (11 messages🔥):

> - `Instruction tuning an LLM`
> - `Data quality in tuning`
> - `Preference tuning (RLHF)`
> - `DPO for persona responses`
> - `Reaction improvements in Discord`

- **指令微调（Instruction Tuning）始于高质量数据**：一位成员询问了针对改变回复语气和声音的 LLM 进行指令微调所需的 prompt 数量，并指出这可能是一个小众问题。
  
  - 另一位成员表示 **数据质量** 是关键，并建议即使是 **1k** 个 prompt 也可以非常有效。
- **偏好微调（Preference Tuning）作为替代方案**：一位成员建议在微调过程中使用偏好微调（RLHF）来代替监督微调（SFT）。
  
  - 他们还提到可以利用 **DPO**，通过提供常规回复与理想回复的示例来进行训练，选择标准可以根据便利性而定。
- **乏味的表情回应引发讨论**：一位成员表示 👍 表情已经变得很乏味，引发了一些提升表情回应趣味性的建议。
  
  - 另一位成员评论道，在团队中使用 ❤️ 代替 👍 是更好的选择；这引发了一场关于表情回应的有趣讨论。

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1296562308297920543) (20 messages🔥):

> - `Gen AI Hackathon`
> - `Creating Checkpoints`
> - `Seamless Image Generation`
> - `Training Models`
> - `Sampling Methods for Cartoon Style`

- **参加 Gen AI Hackathon 赢取大奖**：**Gen AI Hackathon** 邀请团队构建 AI 驱动的系统，奖金总额超过 **$25k**。
  
  - 合作伙伴包括 **aixplain**、**Sambanova Systems** 等，重点关注增强人类潜力的伦理 AI 解决方案。
- **创建自定义 Checkpoints 具有挑战性**：一位成员询问了从头开始创建 Checkpoint 的事宜，得到的建议是这需要 **数百万张标注图像** 和大量的 GPU 资源。
  
  - 另一位成员建议，训练现有模型可能是一个更可行的路径。
- **无缝图像生成的困扰**：一位用户正在寻求帮助以创建可平铺的 **无缝图像**，但指出目前使用 **Flux** 的方法存在困难。
  
  - 一份回复强调，无缝图像的创建可能需要专门的工具，而不是标准的 AI 模型。
- **利用有限图像训练模型**：在关于生成 **Iron Man Prime** 的讨论中，有人建议使用官方漫画中的艺术图创建一个 LoRa 模型，以获得更好的效果。
  
  - 模型 **51** 的图像数量有限也被认为是生成 AI 图像时的一个重大挑战。
- **卡通风格采样方法讨论**：成员们讨论了他们偏好的采样方法，其中一位强调在生成图像时，使用 **dpm++2** 比 Euler 具有更好的稳定性。
  
  - 提到的常用工具包括用于生成特定风格（特别是卡通背景）的 **pony** 和 **juggernaut**。

**提到的链接**：

- [ashen0209/Flux-Dev2Pro at main](https://huggingface.co/ashen0209/Flux-Dev2Pro/tree/main)：未找到描述
- [Vertical Specific AI Agents Hackathon · Luma](https://lu.ma/ke0rwi8n)：Gen AI Agents CreatorsCorner，合作伙伴包括 aixplain, Sambanova Systems, Prem, Marly, Senso, Mistral, coval, heygen, fiberplane, exa 等……
- [Essay Writing Service - Essay Help 24/7 - ExtraEssay.com](https://extraessay.com/?key_wpg=5wpgrd)：最佳论文写作服务，ExtraEssay.com：专业作者、特别折扣、最短期限。我们写论文——你拿高分。

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-announcements**](https://discord.com/channels/1280234300012494859/1280369709623283732/1296892744597377116) (1 条消息):

> - `Quiz 6 发布`
> - `课程报名`
> - `MOOC 讨论频道`
> - `客座讲师`
> - `外部合作伙伴`

- **Quiz 6 现已上线！**：课程工作人员宣布 **Quiz 6** 已在课程网站发布，访问地址请点击[此处](https://llmagents-learning.org/f24)。
  
  - 鼓励参与者及时完成 Quiz。
- **报名参加课程**：有意向的学生可以通过填写此[表单](https://forms.gle/svSoNhKcGFjxup989)报名参加课程。
  
  - 这为有兴趣的个人加入学习社区提供了一个途径。
- **加入 MOOC 讨论频道**：为了进行课程讨论和提问，邀请学生加入 [LLM Agents Discord](https://discord.gg/NWVpQ9rBvd) 的 **MOOC 频道**。
  
  - 该平台促进了参与者之间的互动和支持。
- **了解客座讲师**：介绍了几位**客座讲师**，包括 **Denny Zhou**、**Shunyu Yao** 和 **Chi Wang** 等知名人物。
  
  - 这些讲师将在课程期间贡献宝贵的见解。
- **与行业领导者的合作**：该活动展示了与 **Google**、**OpenAI** 和 **Databricks** 等组织的合作伙伴关系。
  
  - 这些合作突显了课程与现实世界应用的相关性。

 

**提到的链接**：[Large Language Model Agents](https://llmagents-learning.org/f24)：未找到描述

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/1296569432478322729) (17 条消息🔥):

> - `课程报名流程`
> - `文章作业反馈`
> - `直播公告`
> - `Quiz 访问问题`
> - `Discord 社区参与`

- **课程报名流程已开启**：像 **seonsmallworldz** 这样的新参与者确认他们仍可以加入 MOOC，并被建议填写[报名表](https://forms.gle/svSoNhKcGFjxup989)以跟踪提交情况。
  
  - 关于报名流程的咨询引发了加入课程的普遍热情。
- **文章作业的社区反馈**：成员们建议在提交开放式文章作业之前利用社区获取反馈，以确保符合指南要求。
  
  - *sannyshaikh7438* 建议在相应的 Discord 频道分享草稿，以便及时获得建议。
- **每周发放直播链接**：参与者获悉直播链接将于每周一通过电子邮件发送，并在 Discord 上发布公告。
  
  - *faizan102* 对未收到电子邮件表示担忧，促使其他成员进行了澄清。
- **Quiz 访问技术问题**：有人提出了关于 Quiz 5 访问权限的问题，*ajaykumarkv.* 指出最初无法使用，但随后确认问题已解决。
  
  - 这种互动展示了社区成员之间可用的故障排除支持。
- **积极参与课程讨论**：像 *sannyshaikh7438* 这样的成员对频道内收到的快速响应表示感谢，这增强了协作学习。
  
  - 参与反馈分享和故障排除体现了 Discord 社区内互助支持的氛围。

 

**提到的链接**：[Large Language Model Agents](https://llmagents-learning.org/f24)：未找到描述

 

---

### **LAION ▷ #**[**general**](https://discord.com/channels/823813159592001537/823813160075132991/1296564382888886273) (11 messages🔥):

> - `Gen AI Hackathon`
> - `Pixtral vs Qwen2 Performance`
> - `L3_2 Training Issues`
> - `Explicit Content Captioning`
> - `NSFW Evaluation Chaos`

- **Gen AI Hackathon 公告**：[CreatorsCorner](https://lu.ma/ke0rwi8n) 邀请团队参加专注于创建 AI 驱动的多 Agent 系统以改善日常任务的黑客松，奖金超过 **$25k**。
  
  - 鼓励参与者在开发安全可靠的 AI 系统时考虑伦理影响。
- **Pixtral 在与 Qwen2 的竞争中表现不佳**：在对比 **pixtral** 和 **qwen2** 进行露骨内容描述（captioning）时，结果显示 **pixtral** 的表现较差，其 eval loss 高于 **Qwen2** 和 **ll3_2**。
  
  - 用于对比的评估训练完全集中在照片内容上，突显了 Qwen2 的有效性值。
- **L3_2 训练重访计划**：一位成员表示打算在未来重访 **L3_2** 训练，目标是在其成熟并确认有更好性能后在 **unsloth** 中使用。
  
  - 他们在针对特定任务使用 **ms swift** 时遇到了有 Bug 的结果，表明需要进一步验证。
- **露骨内容幻觉担忧**：关于训练协议的讨论显示，无论使用哪种模型，**露骨内容**描述的结果往往会导致严重的幻觉。
  
  - 提到了 **NSFW VQA** 领域的挑战，不同的方法在性能上产生了混乱的结果。

**提及的链接**：[Vertical Specific AI Agents Hackathon · Luma](https://lu.ma/ke0rwi8n)：Gen AI Agents CreatorsCorner，与 aixplain, Sambanova Systems, Prem, Marly, Senso, Mistral, coval, heygen, fiberplane, exa 等合作……

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/1296922622759669780) (1 messages):

> - `LRM using DSPy`
> - `Token costs for LLM-based applications`
> - `GPT-4 pricing changes`

- **探索使用 DSPy 构建 LRM**：一位用户询问了使用 **DSPy** 构建 **Language Representation Model (LRM)** 的经验，并考虑如果没有人做过，就尝试进行原生实现。
  
  - 他们提供了一个关于该主题的[替代方案博客文章](https://www.lycee.ai/blog/drop-o1-preview-try-this-alternative)链接。
- **LLM 应用的 Token 密集性**：构建稳健的基于 **LLM** 的应用需要仔细管理摘要和检索增强生成（RAG）等任务的 Token 使用。
  
  - *对话强调了*生成营销内容可能会消耗大量的 output tokens，因此需要复杂的逻辑和反馈系统。
- **GPT-4 价格大幅下降**：使用 **GPT-4** 的成本已显著降低，现在的价格为 **每百万 input tokens $2.5** 和 **每百万 output tokens $10**。
  
  - 这意味着自 2023 年 3 月发布以来，每百万 input tokens 减少了 **$7.5**，当时的价格分别为 **$10/1M** 和 **$30/1M**。

**提及的链接**：[Drop o1 Preview, Try This Alternative](https://www.lycee.ai/blog/drop-o1-preview-try-this-alternative)：构建稳健的基于 LLM 的应用是 Token 密集型的。你通常必须计划解析和消化大量 Token 以进行摘要甚至检索增强生成。即使是……

---

### **DSPy ▷ #**[**colbert**](https://discord.com/channels/1161519468141355160/1250300504462856265/1296576706940899388) (8 条消息🔥):

> - `ColBERTv2 训练`
> - `带分数的 N-way 元组`
> - `PATH 实现`
> - `DeBERTa 和 MiniLM 的使用`
> - `使用 pylate 进行训练`

- **关于 ColBERTv2 训练数据的困惑**：成员们对 **ColBERTv2** 的训练数据表示困惑，指出它使用的是带分数的 n-way 元组，而不是三元组 (triples)。
  
  - 一位成员引用了一个 [GitHub 仓库](https://github.com/stanford-futuredata/ColBERT) 以进一步澄清训练过程。
- **缩放正负样本分数**：一位成员询问如何调整正样本和负样本的分数以匹配 **MS MARCO** 的量级，因为他们当前的分数范围在 ~.2 到 ~2.4 之间。
  
  - 另一位成员指出，实际的分数缩放可能并不那么关键，从技术上讲，**logprobs** 就足以进行训练。
- **对实现 PATH 的兴趣**：一位成员表示希望根据参考论文实现 **PATH**，尽管其他人指出它主要使用 **DeBERTa** 和 **MiniLM** 等 Cross-Encoders。
  
  - 他们承认将 PATH 与 **ColBERT** 结合的潜力，并建议这可能会产生有趣的结果。
- **推荐使用 pylate**：一位成员分享了一个 GitHub 讨论链接，其中 **bclavie** 推荐使用 [pylate](https://github.com/lightonai/pylate) 来训练 **colbert-small-v1**。
  
  - 这一建议得到了积极的回应，表明该成员打算进一步探索这一建议。

**提到的链接**：

- [GitHub - stanford-futuredata/ColBERT: ColBERT: state-of-the-art neural search (SIGIR'20, TACL'21, NeurIPS'21, NAACL'22, CIKM'22, ACL'23, EMNLP'23)](https://github.com/stanford-futuredata/ColBERT?tab=readme-ov-file#advanced-training-colbertv2-style)：ColBERT：前沿的神经搜索 (SIGIR'20, TACL'21, NeurIPS'21, NAACL'22, CIKM'22, ACL'23, EMNLP'23) - stanford-futuredata/ColBERT
- [answerdotai/answerai-colbert-small-v1 · Fine-tuning example](https://huggingface.co/answerdotai/answerai-colbert-small-v1/discussions/9#66d4f7dd1ae4a81ae57f7620)：未找到描述
- [GitHub - lightonai/pylate: Late Interaction Models Training & Retrieval](https://github.com/lightonai/pylate)：Late Interaction 模型训练与检索。可以通过在 GitHub 上创建账户为 lightonai/pylate 的开发做出贡献。

---

### **Torchtune ▷ #**[**general**](https://discord.com/channels/1216353675241590815/1216353675744641096/1296699237421289484) (1 条消息):

> - `Qwen2.5 Pull Request`
> - `Torchtune 更新`

- **Qwen2.5 Pull Request 已发布**：一位成员在 PyTorch Torchtune GitHub 仓库中分享了一个 [Qwen2.5 的 Pull Request](https://github.com/pytorch/torchtune/pull/1863)，表明它解决了一个未指定的特性或 Bug。
  
  - 正如 PR 描述中所指出的，仍需要详细信息，包括变更日志 (changelog) 和测试计划。
- **Qwen2.5 PR 中的变更日志和测试缺失**：Qwen2.5 的 Pull Request 在变更日志和测试计划中缺乏全面的细节，在描述中被标记为 TODO。
  
  - 这些信息的完善对于确保 PR 符合项目的贡献标准至关重要。

**提到的链接**：[Qwen2.5 by calvinpelletier · Pull Request #1863 · pytorch/torchtune](https://github.com/pytorch/torchtune/pull/1863)：上下文 此 PR 的目的是什么？是添加新功能、修复 Bug、更新测试和/或文档还是其他（请在此处添加） Issue #1624 Changelog TODO Test plan TODO run pre-comm...

---

### **Torchtune ▷ #**[**dev**](https://discord.com/channels/1216353675241590815/1236040539409879170/1296813712635854848) (7 messages):

> - `Torchtune training approaches` (Torchtune 训练方法)
> - `Preference pair generation` (偏好对生成)
> - `RLAIF paper application` (RLAIF 论文应用)
> - `Iterative training process` (迭代训练过程)
> - `DPO vs PPO methods` (DPO 与 PPO 方法)

- **关于 Torchtune 训练方法的辩论**：成员们讨论了两种 Torchtune 训练方法：运行整个流水线，或者使用奖励模型生成偏好对后再进行 PPO 训练。
  
  - 他们强调了运行整个流水线的简单性，以及使用 vLLM 等工具的预生成（pre-gen）方法在效率和内存方面的益处。
- **偏好对迭代的可视化**：一位成员询问了关于从 LLM 到 DPO 使用生成的偏好对进行迭代的视觉呈现。
  
  - 这表明了对澄清训练流程及其组件的兴趣。
- **与 Anthropic 的 RLAIF 论文的联系**：一位成员提到了 Anthropic 的 RLAIF 论文的应用，并参考了 TRL 的实现，该实现使用了 vLLM。
  
  - 他们注意到 RLAIF 论文开创了在每轮训练中生成新数据集并结合不同模型数据的先例。
- **Torchtune 初步尝试建议**：建议根据 RLAIF 流水线描述，开始在 Torchtune 中尝试现有的 SFT + DPO 方案（recipes）。
  
  - 该方法旨在通过利用 DPO 方法来规避奖励模型训练的需求，从而简化开发流程。

 

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1296720373806534707) (3 messages):

> - `Automating document editing` (自动化文档编辑)
> - `Aider AI enhancements` (Aider AI 增强)
> - `Open Interpreter development` (Open Interpreter 开发)

- **自动化文档编辑流程**：一位成员提出了在后台运行代码的同时**自动化文档编辑**流程的想法。
  
  - 他们表示有兴趣了解社区之前探索过的其他**深度使用案例**。
- **Aider 在 AI 生成代码方面的进展**：另一位成员强调，**Aider** 在每个新版本中越来越多地使用 **AI 生成并磨练的代码**。
  
  - *如果模型持续改进*，任何解释器概念都有可能实现动态每日构建（living nightly build）的方法。
- **Open Interpreter 的未来计划**：讨论引发了关于 **Open Interpreter** 是否有计划采用与 Aider 相同的 AI 驱动代码集成方法的询问。
  
  - 成员们渴望了解 **Open Interpreter** 如何从 AI 模型的类似**增量改进**中获益。

 

---

### **OpenInterpreter ▷ #**[**ai-content**](https://discord.com/channels/1146610656779440188/1149229778138824765/) (1 messages):

abhichaturvedi_94225: 感谢 <@631210549170012166>

---

### **LangChain AI ▷ #**[**share-your-work**](https://discord.com/channels/1038097195422978059/1038097372695236729/1296899500417355787) (1 messages):

> - `Capital Companion`
> - `AI trading assistant` (AI 交易助手)
> - `LangChain`
> - `LangGraph`
> - `Advanced trading strategies` (高级交易策略)

- **Capital Companion 发布 - 您的 AI 交易助手**：一位成员介绍了 **Capital Companion**，这是一个使用 **LangChain** 构建并利用 **LangGraph** 处理复杂 Agent 工作流的 AI 交易助手，并邀请大家在 [capitalcompanion.ai](https://capitalcompanion.ai) 上查看。
  
  - *如果有人有兴趣查看或讨论使用案例，请告诉我，* 该成员分享道，寻求关于平台功能的反馈和讨论。
- **股票 AI 驱动投资仪表盘**：Capital Companion 提供了一个 **AI 驱动的投资仪表盘**，旨在帮助用户识别**上涨趋势**并在股票交易中做出明智决策。
  
  - 重点功能包括**技术分析工具**和**市场情绪分析**，旨在为**股票投资**提供竞争优势。

 

**提到的链接**：[Capital Companion - 今天的股票 AI 交易助手 | 最佳交易策略](https://capitalcompanion.ai)：通过 AI 驱动的趋势股票洞察、股票交易软件以及针对最佳交易策略的全面技术分析，增强您的波段交易股票策略。

 

---

### **Alignment Lab AI ▷ #**[**general**](https://discord.com/channels/1087862276448595968/1095458248712265841/1296564979989741719) (1 messages):

> - `Twitter/X Embed Fix`
> - `Discord Integration`

- **修复损坏的 Twitter/X 嵌入！**：一位成员敦促其他人查看一个讨论如何增强 Twitter/X 嵌入的 [Twitter/X Space](https://x.com/i/spaces/1ypKdpLNZXnKW)。
  
  - 讨论重点介绍了在 Discord 和 Telegram 等平台上利用**多张图片、视频、投票、翻译**等功能的方法。
- **增强跨平台参与度**：对话强调了在各种通信平台上通过**投票**和**翻译**等互动功能吸引用户的重要性。
  
  - 这种方法旨在增加用户互动和内容丰富度，使其对不同受众更具吸引力。

 

**提到的链接**：[来自 GitHub 的推文 - FixTweet/FxTwitter: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能](https://x.com/i/spaces/1ypKdpLNZXnKW): Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter

 

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**general**](https://discord.com/channels/1238365980128706560/1238365980128706563/1296835378082222161) (1 messages):

> - `LLM Use Cases`
> - `Mapping Questions-Answers`
> - `Community Repositories`

- **征集 LLM 成功案例**：一位成员询问是否有展示 **LLM** 成功使用案例的仓库或集合，包括 prompt、模型和微调方法。
  
  - 他们表示，如果现有资源不足，希望通过启动一个**仓库**来整合社区力量。
- **映射问题-答案的挑战**：该成员提到一个涉及在两个不同来源之间映射**问题-答案**的具体使用案例，正在寻找先前的示例来指导其方法。
  
  - 这为其他具有类似经验的人提供了分享见解和解决方案的潜在协作机会。

 

---

---

---

---

---

{% else %}

> 完整的频道细分内容已在邮件中截断。
> 
> 如果您想查看完整细分，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
> 
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}