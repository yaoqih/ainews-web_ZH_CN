---
companies:
- x-ai
- openai
- mistral-ai
- anthropic
- cohere
- meta-ai-fair
- hugging-face
- nvidia
date: '2024-05-28T00:04:01.538810Z'
description: '以下是该文本的中文翻译：


  **xAI 以 240 亿美元的估值融资 60 亿美元**，使其跻身全球估值最高的 AI 初创公司之列，这笔资金预计将用于研发 **GPT-5 和 GPT-6
  级别的模型**。由 Nathan Lambert 开发的 **RewardBench** 工具用于评估语言模型的奖励模型（RM），结果显示 Cohere 的奖励模型表现优于开源替代方案。讨论回顾了语言模型从
  1948 年克劳德·香农（Claude Shannon）的模型到 GPT-3 及其后续版本的演进，并强调了 **RLHF（基于人类反馈的强化学习）** 以及较新的
  **DPO（直接偏好优化）** 方法的作用。值得注意的是，目前一些**基于 Llama 3 8B 的奖励模型**在 RewardBench 排行榜上的表现超过了
  GPT-4、Cohere、Gemini 和 Claude，这引发了关于“奖励作弊”（reward hacking）的质疑。未来的对齐研究方向包括改进偏好数据集、优化
  DPO 技术以及实现语言模型的个性化。此外，报告还将 xAI 的估值与 OpenAI、Mistral AI 和 Anthropic 进行了对比，并提到了关于 xAI
  在英伟达（Nvidia）硬件上巨额支出的猜测。'
id: c8e922a2-25da-4f4d-9509-3bd4e6f09448
models:
- gpt-3
- gpt-4
- gpt-5
- gpt-6
- llama-3-8b
- llama-3
- claude-3
- gemini
original_slug: ainews-life-after-dpo
people:
- nathan-lambert
- chris-manning
- elon-musk
- bindureddy
- rohanpaul_ai
- nearcyan
title: '**后 DPO 时代 (RewardBench)**


  或者更口语化的表达：

  **DPO 之后的发展 (RewardBench)**'
topics:
- reinforcement-learning-from-human-feedback
- direct-preference-optimization
- reward-models
- rewardbench
- language-model-history
- model-evaluation
- alignment-research
- preference-datasets
- personalization
- transformer-architecture
---

<!-- buttondown-editor-mode: plaintext -->**RLHF is all you need.**

> 2024年5月24日至5月27日的 AI 新闻。
我们为您检查了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 社区（**382** 个频道，**9556** 条消息）。
预计节省阅读时间（按每分钟 200 字计算）：**1079 分钟**。

这是一个平静的美国节假日周末。

- [X.ai 以 240 亿美元的估值筹集了 60 亿美元](https://x.ai/blog/series-b)，迅速成为估值最高的 AI 初创公司之一。
- 我们还发布了 [ICLR 最佳论文和演讲回顾的第一部分](https://www.latent.space/p/iclr-2024-recap)。
- Alex Reibman 在 Meta Llama 3 黑客松中的 [LlamaFS](https://github.com/iyaja/llama-fs) 项目[走红](https://twitter.com/llama_index/status/1794762651769430381)：这是一个由本地 LLM 驱动的硬盘文件管理器。利用多模态 AI 自动重命名和分类杂乱的文件及目录。

今天的专题介绍 [Nathan Lambert](https://x.com/natolambert)，他正在为 [Chris Manning 的 CS224N](https://web.stanford.edu/class/cs224n/?ref=zgljl2012.com) 课程进行客座演讲（完整的[建议阅读清单](https://web.stanford.edu/class/cs224n/)非常值得一读），并发布了关于奖励模型（Reward Models）历史与未来以及他在 [RewardBench](https://x.com/natolambert/status/1792314629932384343) 工作的[演讲幻灯片](https://docs.google.com/presentation/d/1on5xTePaUYg47vui3dUr0Lp6GUXmOXmhceNJLXRbGsE/edit#slide=id.g271884d3ab7_0_126)。

 
![image.png](https://assets.buttondown.email/images/32f346af-d635-4ac5-9182-9fe76a82d484.png?w=960&fit=max)
 

<ol><li><b>语言模型 (LMs) 的历史</b>：LMs 经历了显著的演变，从 1948 年 Claude Shannon 的早期英语语言模型到 2020 年及以后的强大 GPT-3。2017 年引入的 Transformer 架构在这一进展中起到了关键作用，使得开发能够生成日益复杂且连贯文本的模型成为可能。</li><li><b>RLHF 和 DPO</b>：RLHF（来自人类反馈的强化学习）是许多流行语言模型成功的关键因素，但它复杂且耗费资源。2023 年引入的 DPO（直接偏好优化）是一种更简单、更具扩展性的替代方案，它直接从人类偏好中学习。虽然 DPO 显示出良好的前景，但在许多情况下 RLHF 仍能产生更好的结果。</li><li><b>RewardBench</b>：RewardBench 是一个用于评估 LLM 奖励模型 (RMs) 的工具。它提供了关于 RMs 如何影响 LLM 能力和安全性的见解。Cohere 的 RMs 在 RewardBench 上的表现优于开源模型，突显了在该领域持续研究的重要性。</li><li><b>对齐研究的未来方向</b>：未来的关键研究领域包括收集更多数据（特别是偏好数据集）、改进 DPO 技术、探索不同的模型规模、开发超越通用基准的更具体的评估，以及解决语言模型中的个性化问题。</li></ol>

[RewardBench 论文](https://arxiv.org/pdf/2403.13787)列出了一系列最具挑战性的奖励模型基准测试：

 
![image.png](https://assets.buttondown.email/images/b9ecd1df-da28-4d46-8b0a-51972d2302aa.png?w=960&fit=max)
 

有趣的是，目前在[排行榜](https://huggingface.co/spaces/allenai/reward-bench)上，一些专门针对[奖励模型优化](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1)的 Llama 3 8B 模型正击败 GPT4、Cohere、Gemini 和 Claude。是真有实力，还是奖励作弊 (Reward Hacking)？

 
![image.png](https://assets.buttondown.email/images/27e3f5a0-2b42-4738-aa90-fb25189b4dd6.png?w=960&fit=max)
 

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！


{% endif %}


---

# AI Twitter 回顾

> 所有回顾均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在尝试使用 Haiku 进行聚类和流程工程。

**xAI 以 240 亿美元估值融资 60 亿美元**

- **融资细节**：[@nearcyan](https://twitter.com/nearcyan/status/1794969115586883864) 指出 xAI 筹集了 60 亿美元，**公司估值达到 240 亿美元**。[@bindureddy](https://twitter.com/bindureddy/status/1795120407550308556) 向 xAI 和 Elon 表示祝贺，并表示这笔资金带来的算力应该足以支持 **GPT-5 和 6 级别的模型**。
- **与其他 AI 公司的对比**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1795064230669848697) 将 xAI 240 亿美元的估值与其他 AI 公司进行了对比——OpenAI 估值为 800-1000 亿美元，Mistral AI 在 4 月份的估值约为 60 亿美元，Anthropic 约为 184 亿美元。
- **对支出的猜测**：[@nearcyan](https://twitter.com/nearcyan/status/1794976668232323295) 调侃说这笔钱将直接流向 Nvidia，**省去了中间商**。他还[设想](https://twitter.com/nearcyan/status/1794967177562349681) xAI 会说 100% 的 Nvidia 投资组合是一个合理的赌注。

**对 Elon Musk 和 xAI 的批评**

- **Yann LeCun 的批评**：[@ylecun](https://twitter.com/ylecun/status/1795034443809165464) 讽刺地建议，如果你能忍受一个声称你正在研究的问题明年就能解决、声称它会杀死所有人、一边散布阴谋论一边又声称要严谨追求真理的老板，那就加入 xAI 吧。
- **对 AI 炒作和监管的担忧**：[@ylecun](https://twitter.com/ylecun/status/1794998977105981950) 批评了“末日论者的幻觉”（Doomer's Delusion），即声称 AI 会杀死我们所有人、垄断 AI、要求设置自毁开关、禁止开源，并恐吓公众以从无知的亿万富翁那里获得疯狂的资金。他表示，我们目前甚至还没有人类水平 AI 设计的任何头绪。
- **Musk 的政治立场**：[@ylecun](https://twitter.com/ylecun/status/1795029416499708069) 表示不喜欢 Musk 充满报复性的政治手段、阴谋论和炒作，但他喜欢 Musk 的汽车、火箭、太阳能电池板和卫星网络。他[澄清](https://twitter.com/ylecun/status/1795135271358406946)这并非在支持极左翼，因为威权主义和寻找替罪羊在左右两个极端都存在。

**AI 安全与存在性风险辩论**

- **对 AI 末日论的反驳**：[@ylecun](https://twitter.com/ylecun/status/1795032310590378405) 认为 AI 不是一种会自动出现并变得危险的自然现象，而是由我们设计和建造的东西。他表示，如果存在一个比人类更好地完成目标且安全、可控的 AI 系统，我们就没问题；否则，我们就不会建造它。目前，我们甚至还没有人类水平 AI 设计的任何头绪。
- **对监管提案的反驳**：[@ylecun](https://twitter.com/ylecun/status/1795021895198179368) 回应了一条称 AI 必须由少数公司在严格监管下垄断的推文。他指出这种逻辑把 AI 当成了自然现象而非人造产物，并将其类比为我们在广泛部署涡轮喷气发动机之前，是如何使其变得可靠的。
- **关于通用智能的辩论**：[@ylecun](https://twitter.com/ylecun/status/1794731261669355655) 表示，无论是人工还是自然的通用智能（General Intelligence）都不存在。所有动物都拥有专门的智能和习得新技能的能力。智能的很大一部分是通过与物理世界的互动获得的，而机器需要重现这一过程。

**AI 与机器人技术的进展**

- **Naver Labs 机器人咖啡馆**：[@adcock_brett](https://twitter.com/adcock_brett/status/1794761271272677704) 报道称 Naver Labs 打造了一家星巴克，拥有 100 多个名为 "Rookie" 的机器人，负责运送饮料和执行各种任务。
- **微软的 AI 发布**：[@adcock_brett](https://twitter.com/adcock_brett/status/1794761379565519187) 分享了微软发布的 Copilot+ PC，其运行 AI 的速度比传统 PC 快 20 倍，内置 GPT-4o，并具有 "Recall" 功能，可以搜索你在屏幕上看到过的内容。
- **Tokyo Robotics 演示**：[@adcock_brett](https://twitter.com/adcock_brett/status/1794761424297771284) 指出 Tokyo Robotics 展示了一种新型四指机器人手，可以在不预先了解尺寸的情况下抓取任何形状的物体。
- **其他进展**：[@adcock_brett](https://twitter.com/adcock_brett/status/1794761523044192272) 提到 MIT 正在开发一种机器人外骨骼，以帮助宇航员从摔倒中恢复。他还[分享了](https://twitter.com/adcock_brett/status/1794761646012776680) IDEA Research 发布的目标检测模型，并[报道了](https://twitter.com/adcock_brett/status/1794761713654415376) Anthropic 对 Claude “思想”的窥探。

**新的 AI 研究论文**

- **Grokked Transformers**：[@_akhaliq](https://twitter.com/akhaliq/status/1794912618882187678) 分享了一篇研究 Transformer 是否能对参数化知识进行隐式推理的论文。研究发现 Transformer 可以学习隐式推理，但只能通过 Grokking（在过拟合之后进行的延长训练）来实现。
- **Stacking Transformers**：[@_akhaliq](https://twitter.com/akhaliq/status/1794938544336568380) 发布了一篇关于模型增长以实现高效 LLM 预训练的论文，该研究利用较小的模型来加速较大模型的训练。
- **Automatic Data Curation**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1794911168244646397) 分享了 Meta 关于自监督学习自动数据整理的论文，其表现优于人工整理的数据。
- **Meteor**：[@_akhaliq](https://twitter.com/akhaliq/status/1794920194403336630) 发布了关于 Meteor 的消息，该模型利用多维度的推理（multifaceted rationale）来增强 LLM 的理解和回答能力。
- **AutoCoder**：[@_akhaliq](https://twitter.com/akhaliq/status/1794913439963402683) 指出 AutoCoder 是第一个在 HumanEval 上超越 GPT-4 的 LLM，并配备了多功能代码解释器。

**辩论与讨论**

- **数学技能 vs 语言技能**：针对一条声称 AI 将更看重语言技能而非数学技能的热门推文引发了辩论。[@bindureddy](https://twitter.com/bindureddy/status/1795149723243831508) 认为，要指导 LLM 解决难题，需要的是数学和分析能力，而非语言技能。他还[不同意](https://twitter.com/bindureddy/status/1794890347904192995) AI 会让编程过时的观点，并表示设计能力和 AI 工具的使用将会有很大需求。
- **机械可解释性 (Mechanical Interpretability)**：[@jeremyphoward](https://twitter.com/jeremyphoward/status/1794950590692569227) 告诫不要在可解释性研究中随大流，因为那不是产出顶尖研究的方式。[@NeelNanda5](https://twitter.com/NeelNanda5) 和 [@aryaman2020](https://twitter.com/aryaman2020) 讨论了探索方向缺乏多样性的问题。
- **感知力与意识**：有许多推文在辩论 LLM 和 AI 系统是否具有感知力或意识。主要观点包括：我们无法观察另一个实体的自我意识；试图定义感知力的充分属性会导致荒谬的结论；以及目前的 LLM 可能无法反思其内部状态。

**杂项**

- **历史项目**：[@alexandr_wang](https://twitter.com/alexandr_wang/status/1794843290233401809) 正在启动一个下一代历史项目，旨在发明保存故事的新方法，并利用 LLM 和 AI 让历史变得具有体验感。
- **关于 Rabbit AI 的观点**：对 Rabbit AI 的看法褒贬不一，一些人批评其为骗局，而另一些人如 [@KevinAFischer](https://twitter.com/KevinAFischer/status/1794746529430843675) 则印象深刻，认为向开发者开放后它会变得非常出色。
- **Nvidia 的崛起**：许多人注意到了 Nvidia 的惊人崛起及其对 AI 的重要性，[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1794873807062339633) 分享了一张其股价走势图。
- **元宇宙炒作周期**：[@fchollet](https://twitter.com/fchollet/status/1794814338710290520) 将当前的 AI 炒作与 2021 年底进行了比较，当时元宇宙和 NFT 被视为万物的未来，随后便沉寂了下来。
- **LLaMA 生态系统**：许多人分享了围绕 LLaMA 模型的资源、演示和实验，包括一个名为 LLamaFS 的自组织文件管理器 ([@llama_index](https://twitter.com/llama_index/status/1794762651769430381))。

---

# AI Reddit 摘要

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**AI 进展与能力**

- **Llama-3-70B 微调模型**：在 /r/LocalLLaMA 中，Llama-3-70B 微调模型是 [**Uncensored General Intelligence 排行榜上最不受限（uncensored）的模型**](https://www.reddit.com/r/LocalLLaMA/comments/1d1gvor/llama370b_finetune_is_the_most_uncensored_model/)，表现优于其他大语言模型。
- **Phi-3 mini ONNX DirectML Python AMD GPU**：在 /r/LocalLLaMA 中，一个 [演示显示了从 NPU 重用 KV cache 进行 Token 生成](https://www.reddit.com/r/LocalLLaMA/comments/1d1j31r/faster_whisper_server_an_openai_compatible_server/)，然后在 CPU 上以每秒 27 个 Token 的速度运行。
- **上下文学习 (In-context learning) Transformer**：在 /r/MachineLearning 中，上下文学习 Transformer [可以被用作**表格数据分类器**](https://www.reddit.com/r/MachineLearning/comments/1d0wnjf/r_why_incontext_learning_transformers_are_tabular/)，在预训练期间学习创建复杂的决策边界。
- **MOMENT 基础模型**：在 /r/MachineLearning 中，MOMENT 基础模型 [已发布，用于**时间序列预测、分类、异常检测和插补**](https://www.reddit.com/r/MachineLearning/comments/1d10lma/p_moment_a_foundation_model_for_time_series/)。

**AI Agent 与助手**

- **微软的 Recall**：在 /r/MachineLearning 中，微软的 Recall 可以[使用**开源模型和工具**进行复刻](https://www.reddit.com/r/MachineLearning/comments/1d14pad/p_rerecall_i_tried_to_recreate_microsofts_recall/)，例如使用 mss 获取屏幕截图，使用 ollama/llava 进行描述，以及使用 chromadb 进行搜索。
- **Jetta 自主可配置 PC 任务执行器**：在 /r/singularity 中，Jetta 是一款[可以**控制你的 PC** 的新型 AI Agent](https://www.reddit.com/r/singularity/comments/1d1etav/microsofts_screen_assistant_feature_is_probably_a/)。
- **Faster Whisper Server**：在 /r/LocalLLaMA 中，Faster Whisper Server 提供了一个[用于**流式转录**的 OpenAI 兼容服务器](https://www.reddit.com/r/LocalLLaMA/comments/1d1j31r/faster_whisper_server_an_openai_compatible_server/)，并使用 faster-whisper 作为后端。
- **AI Agent 应用**：在 /r/LocalLLaMA 中，人们对潜在的 AI Agent 应用感到兴奋，例如一旦更大的模型可以在消费级硬件上运行，就能实现[**带有 AI 角色的游戏**](https://www.reddit.com/r/LocalLLaMA/comments/1d11cb1/what_cool_apps_do_you_think_local_llms_will_enable/)。

**AI 监管与治理**

- **两位前 OpenAI 董事会成员**：在《经济学人》中，两位前 OpenAI 董事会成员认为 [AI 公司无法自我治理，**需要监管**](https://www.economist.com/by-invitation/2024/05/26/two-former-openai-board-members-say-ai-firms-cant-be-left-to-govern-themselves)来为了人类的利益驯服市场力量。
- **AI “紧急停止开关” (Kill Switch)**：在《财富》杂志中，科技公司已[同意设立 AI “紧急停止开关”](https://fortune.com/2024/05/21/ai-regulation-guidelines-terminator-kill-switch-summit-bletchley-korea/)以防止终结者式的风险，尽管具体的实施细节尚不明确。

**AI 与社会**

- **AI 失业与 UBI**：在 /r/singularity 中，一些人认为[是时候引入与**失业率**挂钩的 UBI (全民基本收入)](https://www.reddit.com/r/singularity/comments/1d1k3v0/it_might_already_be_time_for_ubi/)，以支持那些被 AI 取代的人。
- **后稀缺世界**：在 /r/singularity 中，在后稀缺世界里，[过剩的财富仍然可以购买**稀缺体验**](https://www.reddit.com/r/singularity/comments/1d10ngi/what_things_will_excess_wealth_still_be_useful/)，如房地产、度假、现场表演和由人类提供的服务。
- **Age Reversal Unity**：在 /r/singularity 中，Age Reversal Unity 已[向 FDA 请愿**将衰老认定为一种疾病**](https://www.reddit.com/r/singularity/comments/1d0vr4j/age_reversal_unity_mailed_the_citizen_petition_to/)，这是迈向抗衰老治疗的一步。

**AI 艺术与内容生成**

- **Stable Diffusion 技术**：在 /r/StableDiffusion 中，新的 Stable Diffusion 技术允许[**以创意方式穿梭于图像之间**](https://www.reddit.com/r/StableDiffusion/comments/1d0zfry/in_the_house_that_rubik_built_every_window_tells/)，从而创作出细节丰富、宏大的艺术作品。
- **动漫风格转换器**：在 /r/StableDiffusion 中，一款动漫风格转换器[结合了 SDXL 模型、ControlNet 和 IPAdapter](https://www.reddit.com/r/StableDiffusion/comments/1d11izd/you_never_go_wrong_with_adding_caspar_david/) 来转换角色设计。
- **AI 生成艺术的碳足迹**：在一篇图文中，有人认为 [AI 生成艺术的**碳足迹**比传统艺术小得多](https://i.redd.it/5zuv1vangq2d1.png)，尽管这一说法存在争议。
- **LoRA 模型**：在 /r/StableDiffusion 中，发布了用于[**动漫风格机器人**](https://www.reddit.com/r/StableDiffusion/comments/1d1hwvq/droiddiffusionxl_v1_lora/)和[**跑步动作**](https://www.reddit.com/r/StableDiffusion/comments/1d14gvh/whats_the_cloud_platform_to_run_stable_diffusion/)的新 LoRA (Low-Rank Adaptation) 模型。

**模因与幽默**

- **Nvidia 不断增长的营收**：一个模因嘲讽道，[Nvidia 不断增长的营收在 **2024 年超出了预期**](https://i.redd.it/emybpzlzdu2d1.jpeg)，而 AMD 则落后了。
- **Google 的 AI 策略**：在 /r/singularity 中，一个模因嘲笑 [Google 的 AI 策略**过于谨慎且缓慢**](https://www.reddit.com/r/singularity/comments/1d12t1w/googles_approach_to_ai_is_a_meme_at_this_point/)。
- **AI 艺术生成的限制**：一个模因暗示，[AI 艺术生成的唯一限制是“**你的想象力……以及 VRAM**”](https://i.redd.it/t8h1thbd4u2d1.jpeg)。

---

# AI Discord 摘要

> 摘要之摘要的摘要

1. **微调与模型训练挑战**：
   - 各个 Discord 频道上的讨论强调了在**微调 (fine-tuning)** **Llama 3** 和 **Mistral** 等模型时面临的挑战，用户遇到了从**语义相似度过拟合 (semantic similarity overfitting)** 到 T4 等 GPU 上的**运行时错误 (runtime errors)** 等问题。社区分享了实用的指南和故障排除技巧，例如 **[TinyLLama Fine-Tuning](https://lucasvw.github.io/posts/19_llm_fine_tuning/)** 和 **[Mistral-Finetune repository](https://github.com/mistralai/mistral-finetune)**。
   - 成员们在**模型分词 (tokenization) 和提示词工程 (prompt engineering)** 方面遇到了困难，强调了正确使用模板标记（如 `###`）或**文本结束标记 (end-of-text tokens)** 对高效微调的重要性。这在 **Axolotl** 和 **Jarvis Labs** 的讨论背景下尤为突出。

2. **多模态模型与集成的进展**：
   - Discord 用户指出，**Perplexity AI** 在处理 CSV 文件方面优于 **ChatGPT**，因为它支持直接上传并集成了 **Julius AI** 等工具进行数据分析。
   - **HuggingFace** 上分享了一个使用 3D 渲染的**新蛋白质可视化项目**，同时还讨论了集成 **Vision Transformers (ViT)** 用于单目深度估计等任务的注意事项。查看 [GitHub repository](https://github.com/AstraBert/proteinviz/blob/main/examples.md) 获取蛋白质示例。

3. **开源 AI 项目与社区努力**：
   - **LlamaIndex** 推出了用于**自动化 RAG 聊天机器人**的工具，详见关于 [MultiOn 演示的帖子](https://twitter.com/llama_index/status/1793764970024570979)。讨论了关于确保**上下文维持 (context maintenance)** 和高效索引以进行知识检索的问题。

4. **新模型发布与基准测试**：
   - **Meta** 的 **Phi-3 Medium 128k Instruct** 首次亮相，因其增强的推理和指令遵循能力而受到关注，可在 [OpenRouter](https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct) 上获取。讨论强调了用户对模型性能和应用的反馈。
   - **IBM Granite** 与 **Llama-3** 的性能辩论在 **ChatbotArena** 等平台上浮现，强调了对可信且透明的基准测试的需求。**DeepSeek-V2** 和 **Granite-8B-Code-Instruct** 是值得关注的提及对象，并分享了具体的基准测试。

5. **伦理、立法与 AI 的社会影响**：
   - 针对 **SB-1047** 法案表达了担忧，将其比作监管俘获，并认为这会使较小的 AI 参与者处于劣势。社区分享了使用 **Perplexity AI** 等工具搜索立法影响的方法，以提高社区意识。
   - **OpenAI** 在 AI 模型训练期间的耗水量引发了关于环境影响的讨论，引用了 [Gizmodo 的文章](https://gizmodo.com/chatgpt-ai-water-185000-gallons-training-nuclear-1850324249)。社区呼吁采取更多**环保的 AI 实践**，并讨论了如 **Meta** 的 **Audiocraft** 等替代方案以实现可持续发展。

---

{% if medium == 'web' %}



# 第一部分：高层级 Discord 摘要

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

**微调事实（Fine-Tuning Facts）**：在 [general 频道](https://discord.com/channels/1238365980128706560/1238365980128706563/1243282801760145408) 的讨论中，揭示了由于偏置的数据类别导致的 **语义相似度过拟合（semantic similarity overfitting）** 问题。一位用户在理解微调与用户输入及初始模型训练的关系时感到困惑。此外，还注意到 **OpenAI 平台侧边栏** 的变化，其中两个图标（threads 和 messages）消失了。

**模板成为焦点**：在 [workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1243336501018755123) 中，强调了在微调期间正确配置模板的重要性。特别是分隔符 `###` 有助于解析不同的输入部分，而 "end of text" token 则指示何时停止 token 生成。

**Maven 与交流**：在 [asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/1243344778511515698) 频道中，成员们进行了一次轻松的交流并提到了重聚。关于会议演讲录像的请求得到了回应，视频可在 Maven 上观看。

**Modal 动员**：[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1243309176722030702) 的 Modal 用户分享了收到额度（credits）和训练经验的兴奋之情，并为新用户提供了指向 **Modal 文档** 和 **示例** 的具体链接。此外还分享了一个使用 Modal 参加 **Kaggle 竞赛** 的计划，包括设置和执行细节。

**Jarvis 记录 Jupyter 杂记**：在 [jarvis-labs 频道](https://discord.com/channels/1238365980128706560/1241117895740625099/1243307629057671229) 中，成员们讨论了在 Jarvis 上存储 VSCode 仓库的问题，并建议使用 GitHub 保存工作。由于不稳定性，发布了 **竞价实例（spot instance）移除** 的通知。分享了微调 **open-lama-3b** 模型的成本和时长，一位用户通过调整模型参数解决了 Ampere 系列显卡的错误。

**Hugging Face 讨论额度与西班牙语模型**：[hugging-face 频道](https://discord.com/channels/1238365980128706560/1241141471814488115/1243335428887806004) 讨论了待处理的 **HF 额度** 以及适用于西班牙语文本生成的模型——推荐了 **Mistral 7B** 和 **Llama 3** 模型。

**额度倒计时继续**：在 [replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1243453712182149150) 频道中，预告了即将发布的关于额度管理和分配的公告。

**Corbitt 的准则引发关注**：[kylecorbitt_prompt_to_model 频道](https://discord.com/channels/1238365980128706560/1242221891733946490/1243287896652517376) 中热情的参与者讨论了 Kyle Corbitt 演讲中介绍的微调方法和技术，包括 *[部署微调模型的十诫 (Ten Commandments for Deploying Fine-Tuned Models)](https://docs.google.com/presentation/d/1IIRrTED0w716OsU_-PL5bONL0Pq_7E8alewvcJO1BCE/edit#slide=id.g2721fb6713e_0_67)*。

**Axolotl 响应号召**：在 [workshop-2](https://discord.com/channels/1238365980128706560/1242223415293644930/1243277523316637817) 中，用户讨论了数据集、模型训练以及 Axolotl 中的故障排除。分享了一篇关于 **TinyLLama 微调** 的博客文章，并推动将可观测性（observability）集成到 LLM 应用中。

**退出 Zoom，进入 Discord**：在 Zoom 聊天功能禁用后，[workshop-3](https://discord.com/channels/1238365980128706560/1242223458184597534/1243339106675724369) 的用户将讨论转移到了 Discord。

**Axolotl 的缓存难题引发困惑**：在 [axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1243286083022618664) 频道中，解决了令用户沮丧的 Axolotl 缓存问题以及对文件丢失的困惑。关于样本打包（sample packing）的讨论和一份关于 tokenizer 陷阱的指南解决了有关效率和分词的疑虑。

**加速迈向胜利**：[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1243291846415749283) 频道的用户解决了关于浮点数比较的困惑，修复了 Jarvislab 训练命令错误，并交流了学习模型加速的资源，重点关注微调的最佳实践。

**与 Axolotl 一起尝试**：[wing-axolotl 频道](https://discord.com/channels/1238365980128706560/1242564077151326388/1243305377974587412) 在数据集模板、预处理问题、Axolotl 配置方面进行了协作，并提供了最新 Axolotl 更新的 PR 合并。他们深入研究了调试工具以及精确模板对训练成功的重要性。

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**蛋白质数据可视化达到新高度**：一个新的蛋白质可视化项目现在支持 3D 渲染，并包含了人类血红蛋白和核糖体蛋白的示例，项目详情可以在 [GitHub](https://github.com/AstraBert/proteinviz/blob/main/examples.md) 上找到。

**使用 OpenAI 的 Whisper 进入 TranscriptZone**：一款利用 OpenAI 的 Whisper 转录 YouTube 视频及更多内容的新转录应用已在 [Hugging Face Spaces](https://huggingface.co/spaces/tensorkelechi/vidtext) 上线。

**去中心化网络——不仅仅是一个梦想？**：一个为去中心化互联网构建基础设施的项目通过调查寻求社区反馈，引发了关于数据收集伦理的讨论。

**Vision Transformers 深度查询**：一位成员寻求关于应用 Vision Transformers (ViT) 进行单目深度估计的资源，表示有意使用 ViT 开发模型，但讨论中未提供具体资源。

**Mistral 模型的量化困境**：在 **Mistral v0.3 Instruct** 上使用 **bitsandbytes** 进行 8-bit 量化导致性能比 4-bit 和 fp16 更慢，这一令人困惑的结果与减少位数计算预期的效率提升相矛盾。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 在 CSV 对决中超越 ChatGPT**：工程师们讨论认为 **Perplexity AI** 在 CSV 文件处理方面优于 **ChatGPT**，因为它允许直接上传 CSV。此外，推荐使用 **Julius AI** 进行数据分析，它利用 Python 并集成了 **Claude 3** 或 **GPT-4** 等 LLM。

- **用户冷落 Claude 3 Opus**：由于内容限制增加和感知到的实用性下降，**Claude 3 Opus** 遭到冷落，尽管 **GPT-4** 也有局限性，但仍被视为更好的选择。

- **质疑 Pro Search 的真实升级**：**Pro Search** 的升级引起了关注，用户讨论新的多步推理功能和 API 规范是真正的后端改进，还是仅仅是表面层级的 UI 增强。

- **阐述 API 集成**：围绕外部工具与 **Claude** 的 API 集成的对话引起了兴趣，同时分享了自定义函数调用、无服务器后端以及诸如 [Tool Use with Claude](https://docs.anthropic.com/en/docs/tool-use) 等文档。

- **AI 伦理：不仅仅是一个思想实验**：关于为 GPT 注入伦理监控能力的讨论被触发，揭示了其在职场沟通和法律辩护方面的潜在应用，尽管哲学上的难题尚待解决。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **关于 RTX 5090 显存（VRAM）的猜测达到顶峰**：关于传闻中拥有 **32GB VRAM 的 RTX 5090** 是否具有实际意义的辩论正热。引用了 [PC Games Hardware](https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/News/Geforce-RTX-5090-mit-32-GiB-und-drei-PCBs-1448073/galerie/3884555/) 上的潜在规格和图片，但部分成员对其真实性保持怀疑。

- **Stable Diffusion 与 AMD 的挑战**：用户提供了在 AMD 5700XT GPU 上安装 **Stable Diffusion** 的指导，建议从 [Craiyon](https://www.craiyon.com/) 等 Web 服务开始，以避开潜在的兼容性问题。

- **Stable Diffusion 3：先试后买**：社区将 **Stable Diffusion 3** 与竞争对手 Midjourney 进行了对比，强调虽然 SD3 提供免费试用，但持续访问需要 **Stability** 会员资格。

- **对 Mobius 模型的期待与日俱增**：关于 DataPlusEngine 的新型 **Mobius 模型** 的公告因其声称能创建高效基础模型而获得极大关注。该模型在 [Twitter](https://x.com/DataPlusEngine/status/1793803117642854732) 上进行了预告，它既不是简单的基础模型，也不是现有模型的微调版本。

- **32GB VRAM：颠覆者还是性能过剩？**：提到 32GB VRAM 的 GPU 引发了关于 Nvidia 数据中心 GPU 销售策略潜在转变的对话，考虑到拥有大容量显存的产品可能会如何影响市场对 H100/A100 系列的需求。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **PEFT 配置障碍已解决**：一个关于 PEFT 训练期间缺失 `config.json` 的问题已通过从基础模型的配置中复制该文件得到解决，用户已确认成功。

- **Llama 修复 Bug 提升性能**：**Llama 3** 模型的基础权重被描述为“有 Bug”，但 Unsloth 已经实现了修复。为了改进训练，建议使用 reserved tokens 并更新 tokenizer 和 `lm_head`。

- **System Prompt 助力 Llama 3**：观察发现，加入 System Prompt（即使是空的）也能增强 Llama 3 的 finetuning 效果。

- **Phi 3 模型激增**：随着 **Phi 3 模型** 的首次亮相，社区反响热烈，该模型支持 medium 版本。社区讨论引导工程师关注博客文章和发布说明中的详细信息。

- **Stable Diffusion 的诡异侧面**：**Stable Diffusion** 产生的诡异伪影和离奇的语音克隆输出让用户感到惊讶，相关的讨论和经历已通过 YouTube 视频和 Reddit 帖子分享。

- **VSCode Copilot 推荐**：用户在 **random** 频道寻求本地 VSCode "copilot" 的建议，并得到了积极的回应和推荐。

- **Phi-3 的推理延迟**：一名用户对使用 **Unsloth Phi-3** 时较慢的推理时间感到困惑，并提供了一个 [Colab notebook](https://colab.research.google.com/drive/1LLWoaQrH8KFkQlE4ONwwtC4tC1-1It2X) 来调查延迟原因，社区目前仍在努力寻找修复方案。

- **量化困境解析**：一名成员在量化自定义模型时面临挑战，在 **llama.cpp** 和 **Docker** 兼容性方面遇到了障碍，引发了关于解决方案的讨论。

- **模型显存（VRAM）需求定论**：明确了 VRAM 要求：**Phi 3 mini 需要 12GB** 即可，但 **Phi 3 medium 必须配备 16GB**。对于繁重任务，建议考虑外部计算资源。

- **训练一致性的数据尽职调查**：强调了在训练和评估中使用一致数据集的重要性，并重点介绍了 **Unslothai 的公共数据集**，如 [Blackhole Collection](https://huggingface.co/collections/lamhieu/blackhole-66473b7feec034b4fb70818a)。

- **平台可能性与注意事项**：针对 **Unsloth** 是否支持旧款 Mac 的查询得到了回复，确认目前重点在于 CUDA 和 GPU 的使用，并为仅有 CPU 的设备提供了建议。

- **企业级专家支持扩展**：一名社区成员主动向 Unsloth 提供企业级专业知识，并对加入 Build Club 和 GitHub 的加速器表示赞赏，暗示了 Unsloth 未来发展的协同潜力。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**关于 AI 理解能力的智力辩论**：社区就 LLM 对概念的真正“理解”展开了深入讨论，**可解释性研究（interpretability research）** 被视为重要的经验证据。怀疑论者认为目前的努力尚不足够，并引用了 **Anthropic** 在映射大语言模型思维方面的工作。

**Llama 模型的进阶探索**：一项旨在增强 **Llama 模型** 的技术尝试集中在编写一个能够管理 **function calls** 的脚本上，并以 **Hermes Pro 2** 的方法作为灵感。另一个咨询则围绕在 3080 GPU 上实现 **Llama3 LoRA** 技术展开。

**数字维度的现实探索**：在关于 **Nous 和 WorldSim** 的对话中，成员们探讨了 **NightCafe** 和多维 AR 空间在映射复杂 AI 世界中的潜在应用。**audio-visualizers** 中的梦幻探索和奇特的 **ASCII art** 表现形式突显了 AI 驱动模拟的创意用途。

**筛选 RAG 数据**：倡导模型将**内部知识**与**检索增强生成（RAG）**相结合是一个热门话题，同时也提出了如何处理矛盾和解决冲突的问题。强调用户评估被认为是必不可少的，特别是对于复杂的查询案例。

**AI 微调中的精准度胜过虚幻**：社区讨论赞扬了 **Mobius 模型** 在**图像生成**方面的卓越表现，并期待其开源版本和阐释性论文的发布。此外，还提到了 Hugging Face 的 `PyTorchModelHubMixin` 可以简化模型共享，但受到 **50GB 大小限制**（在不分片的情况下）的约束。

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **JAX vs. PyTorch/XLA: TPU 对决**：在 TPU 上 **JAX** 与 **PyTorch/XLA** 的性能对比引发了关于 **warmup times** 和 **blocking factors** 等基准测试细微差别的辩论。GPT-3 的训练成本从 **450 万美元大幅下降至 2024 年预计的 12.5 万至 100 万美元**，这一观点结合了多位贡献者提供的 **TFLOP rates** 和 **GPU-hour pricing**，并链接到了 [Databricks 博客文章](https://www.databricks.com/blog/turbocharged-training-optimizing-databricks-mosaic-ai-stack-fp8)。

- **扩展与教学 LLMs**：在研究论坛中，**Chameleon** 模型因其在多模态任务中的强劲表现而受到关注，而 **Bitune** 则承诺提升 LLM 的 zero-shot 性能（[Bitune 论文](https://arxiv.org/pdf/2405.14862)）。讨论质疑了 **JEPA** 模型对于 AGI 的可扩展性，并批评了 **RoPE** 的上下文长度限制，引用了相关 [论文](https://arxiv.org/pdf/2405.14591)。

- **Emergent Features 困扰 LLM 爱好者**：链接了 Tim Dettmers 关于在 Transformer 推理中保持性能的高级 quantization 方法的研究，包括他提出的 emergent outliers 概念，以及通过 [bitsandbytes 库](https://huggingface.co/blog/hf-bitsandbytes-integration) 与 Hugging Face 的集成。关于 emergent features 的讨论围绕着它们是模型的“DNA”这一观点展开，并探讨了其对相变（phase transitions）的影响。

- **技术调整与 LM 评估简报**：在 **lm-thunderdome** 频道中，工程师们分享了在 **vllm 模型** 中设置 seed、使用 `lm_eval --tasks list` 获取 **任务列表** 以及处理影响 Accelerate 等框架（存在内存问题）的 **BigBench** 任务名称变更的实用技巧。建议通过查阅 `lm-eval/tasks` 文件夹来定位任务，以便更好地进行组织。

- **协作呼吁**：发起了扩展 **Open Empathic** 项目的呼吁，并提供了一个用于贡献电影场景的 [YouTube 指南](https://youtu.be/GZqYr8_Q7DE) 以及项目链接。鼓励进一步的协作，强调了社区努力在增强项目方面的必要性。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**GPU 探险**：工程师们讨论了将小型模型加载到 GPU 上的挑战，一些人青睐 *llama3, mistral instruct* 和 *cmdrib* 等模型。同时，据报道在某些应用中，使用较低的 quantization（如 *llamas q4*）比 q8 等高级别 quantization 效果更好，反驳了“越大越好”的观念。

**下一代模型即将来临**：模型领域的更新通知了一个 **35B 模型** 的发布，并正在进行测试以确保 LM Studio 兼容性。针对不同规模模型的优化也是一个话题，重点关注 **Phi-3 small GGUFs** 及其效率。

**服务器与配置**：硬件讨论包括利用 **llama.cpp** 及其最近的 RPC 更新进行 **分布式推理 (distributed inference)**，尽管目前尚不支持 quantized 模型。还探索了使用配备 **RTX 4060 Ti 16GB** 的廉价 PC 集群进行分布式模型设置的实验性构建，以及可能存在的网络限制。

**实现多语言凝聚力**：Cohere 模型现在将其能力扩展到了 **23 种语言**，正如广告所言，**aya-23 quants** 已开放下载，但 ROCm 用户必须等待更新才能体验。 

**Stable Diffusion 被排除在外**：LM Studio 澄清其专门处理语言模型，不包括像 Stable Diffusion 这样的图像生成器，同时处理了旧款 GPU 上的 CUDA 问题，并推广了 **Julius AI** 等服务以缓解用户体验方面的困扰。



---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Gradient Norm 棘手问题**：将 batch size 从 32 更改会导致 Gradient Norm 突然飙升，从而中断训练。一个 [pull request](https://github.com/karpathy/llm.c/pull/456) 通过防止 fused classifier 中的索引溢出解决了这个问题。
  
- **Int4 和 Uint4 类型需要关注**：一位成员指出 PyTorch 中许多函数缺乏对 **int4** 和 **uint4** 数据类型的实现，相关的 [讨论帖](https://dev-discuss.pytorch.org/t/supporting-new-dtypes-in-pytorch/1833) 表明了在类型提升（type promotion）和 tensor 操作方面的局限性。

- **直播代码预警 —— Scan 算法成为焦点**：Izzat El Hajj 将主持一场关于 Scan 算法的现场编程环节，该算法对于 Mamba 等 ML 算法至关重要。活动定于 `<t:1716663600:F>`，有望为爱好者们带来一次深度的技术钻研。

- **CUB 库查询与 CUDA 细节**：成员们参与了从 CUDA CUB 库代码的运行机制到如何在不使用 cuBLAS 或 cuDNN 的情况下触发 tensor cores 的讨论，并重点推荐了 [NVIDIA 的 CUTLASS GitHub 仓库](https://github.com/NVIDIA/cutlass/tree/main) 和 [NVIDIA PTX 手册](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) 等资源。

- **FineWeb 数据集难题**：处理 FineWeb 数据集非常占用存储空间，磁盘占用达到 70 GB，并消耗高达 64 GB 的 RAM，这暗示在数据处理任务中需要更好的优化或更强大的硬件配置。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Python 库比起 Mojo 更倾向于 C**：关于将 Python 库移植到 Mojo 的可行性和准备工作有一场激烈的讨论，考虑到 Mojo 不断演进的 API，人们担心会给维护者带来过大压力。成员们讨论了将目标对准 C 库是否会是一个更直接且实际的尝试。

**Rust 的安全性吸引力不会削弱 Mojo 的潜力**：Mojo 并不打算取代 C，但 Rust 的安全性优势正在影响工程师们对 Mojo 在不同场景下应用的思考。正在进行的讨论涉及了可以使 Mojo 开发受益的 Rust 概念。

**使用 Nightly 版本 Mojo 飞速前进**：在 MacOS 上使用 Nightly 版本的 Mojo 运行 BlazeSeq 的性能显示出与 Rust 的 Needletail 惊人的相似性，引发了关于跨平台效率的讨论。快速迭代的 Nightly 更新（见 [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)）让社区紧跟这门不断进化的语言。

**对 Modular Bot 运作机制的好奇**：有人询问了 "ModularBot" 的底层技术，虽然没有提到具体的模型，但该机器人给出了一个生动的回复。另外，讨论了在 Mojo 中进行 ML 模型训练和推理的潜力，并提到 Max Engine 可以作为 numpy 的替代方案，尽管目前还没有成熟的训练框架。

**编译时困惑与对齐烦恼**：从内存中布尔值（boolean）的对齐问题到编译时函数问题，都引起了用户的关注。通过变通方案和官方 [bug reports](https://github.com/modularml/mojo/issues/2813) 凸显了社区驱动排错的重要性。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **忠于 LaTeX 的 LLM**：在格式化领域，用户对 GPT 表现出强烈的倾向性感到沮丧，尽管请求使用 Typst 代码，它仍默认使用 LaTeX，这揭示了 LLM 似乎坚持某种特定的编码语法偏好。
  
- **Microsoft Copilot+ 与 Leonardo 之争**：社区讨论集中在 Microsoft Copilot+ PCs 对于“草图转图像”等创意任务的价值，而一些成员则鼓励尝试 [Leonardo.ai](https://leonardo.ai) 以获得类似的功能。

- **对 AI 效率的渴求**：有人对 AI 造成环境负担表示担忧，引用了 [Gizmodo 的一篇文章](https://gizmodo.com/chatgpt-ai-water-185000-gallons-training-nuclear-1850324249)，该文章指出 AI 模型训练期间耗水量巨大，引发了关于需要更环保的 AI 实践的讨论。

- **迭代胜过创新**：关于通过迭代改进来增强 LLM 性能的对话非常活跃，并提到了像 AutoGPT 这样处理迭代的项目，尽管这伴随着更高的成本。

- **智能注入提议言过其实？**：公会成员思考了在 ChatGPT 中嵌入法律知识的可行性和潜力，甚至考虑到了 6.5 亿美元的估值，尽管关于这一大胆断言的详细观点较少。



---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**LangChain CSV Agent 深度解析**：工程师们在 **SequentialChain** 中探索了 **LangChain 的 CSV agent**，并讨论了[如何自定义输出键](https://python.langchain.com/docs/modules/chains/foundational/sequential_chains)（如 `csv_response`）。提到了 SQL agents 在处理多表查询时面临的挑战，指出 token 限制和 LLM 兼容性问题，并引导至 GitHub [提交 issue](https://github.com/langchain-ai/langchain/issues)。

**AI 展示引发关注**：[OranAITech 在推特上发布了](https://twitter.com/OranAITech/status/1793684085056942412?t=AVjC2GpAdrT-LqwMEzv0nQ&s=19)他们最新的 AI 技术，同时 **everything-ai v2.0.0** 宣布了包括音频和视频处理功能在内的新特性，并提供了[代码仓库](https://github.com/AstraBert/everything-ai)和[文档](https://astrabert.github.io/everything-ai/)。

**揭秘 VisualAgents**：YouTube 上分享了 **Visual Agents 平台**的演示，展示了其利用 LangChain 的能力来简化 SQL agent 创建以及无需编码构建简单检索系统的潜力。两段特定视频展示了其工作流：[SQL Agent](https://youtu.be/_3crxBzVg3A?si=r2rDA19q-fHm7h9N) 和 [Simple Retrieval](https://youtu.be/prOjBQQgKlU?si=jDt53koCl6lT6BoM)。

**EDA GPT 印象展示**：通过 [LOVO AI](https://genny.lovo.ai/share/d6b58f0d-fc46-4aa7-a65e-fa0f9a684f01) 链接了一个 **EDA GPT** 的演示，包括一段展示其各项功能的五分钟概览视频。该演示突显了这款 AI 工具的多功能性。

**教程预告**：教程频道的一条消息提供了一个指向 business24.ai 内容的 [YouTube 链接](https://youtu.be/gflsu_6R_8g)，尽管其相关背景尚未披露。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **盗版并非万灵药**：尽管有人幽默地建议 The Pirate Bay 可能成为分享 AI 模型权重的避风港，但成员们对此表示怀疑，并强调其他国家更友好的 AI 政策环境可能会取而代之。

- **日本在 AI 领域采取积极态度**：参与者注意到日本对 AI 发展的鼓励立场，并引用了一篇通过[推文](https://x.com/DataPlusEngine/status/1793817514956259460)分享的**论文**，该论文关于在无需大规模预训练的情况下创建新的基础 diffusion 模型，展示了一种涉及临时破坏模型关联的策略。

- **投毒恢复协议研究**：提到了一项由 fal.ai 开展的涉及投毒模型恢复方法的**合作研究**，预计研究结果将从经验上证实该恢复方法。成员们对 AI 生成图像的美学表达了保留意见，特别是 Mobius 等模型与 MJv6 等前辈相比所呈现的“高对比度外观”和伪影。

- **Claude 映射破解代码**：Anthropic 的**研究论文**详细剖析了 Claude 3 Sonnet 的神经景观，阐明了对概念激活（conceptual activations）的操作，可在其[研究页面](https://www.anthropic.com/research/mapping-mind-language-model)阅读。关于此类激活潜在商业化的辩论随之兴起，同时也伴随着对商业影响导致 AI 从业者感到沮丧的担忧。

- **对 AI 视觉愿景的回顾**：一位成员回忆了从 Inception v1 等早期 AI 视觉模型到如今复杂系统的演变，认可了 DeepDream 在理解神经功能方面的作用。此外，还讨论了神经网络中稀疏性的好处，描述了使用 L1 norm 实现稀疏性，以及在高维层中典型的 300 个非零维度。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **聚会提醒：名额有限**：即将于周二举行的 **LlamaIndex meetup** 仅剩少量名额，由于名额有限，鼓励爱好者们尽快[预订位置](https://twitter.com/llama_index/status/1793739449127583964)。

- **MultiOn 结合 LlamaIndex 实现任务自动化**：**LlamaIndex** 已与 AI Agents 平台 **MultiOn** 结合，通过代表用户操作的 Chrome 浏览器促进任务自动化；在此查看演示 [demo](https://twitter.com/llama_index/status/1793764970024570979)。

- **RAGApp 发布，支持无代码 RAG 聊天机器人设置**：新推出的 **RAGApp** 简化了通过 Docker 容器部署 RAG 聊天机器人的过程，使其能够轻松部署在任何云基础设施上，并且它是开源的；在此配置你的模型提供商 [model provider](https://twitter.com/llama_index/status/1794030544415818062)。

- **解决 PDF 解析难题**：社区认可 **LlamaParse** 作为从 PDF（尤其是表格和字段）中提取数据的可行 API，利用 GPT-4o 模型提升性能；**Knowledge Graph Indexing** 的挑战也是一个话题，强调了手动和自动（通过 `VectorStoreIndex`）策略的必要性。

- **PostgresML 与 LlamaIndex 联手**：**Andy Singal** 分享了将 **PostgresML** 与 **LlamaIndex** 集成的见解，并在 Medium 文章 ["Unleashing the Power of PostgresML with LlamaIndex Integration"](https://medium.com/ai-advances/unleashing-the-power-of-postgresml-with-llamaindex-integration-9eadee223939) 中详细介绍了这一合作，获得了社区的积极评价。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Phi-3 Medium 128k Instruct 发布**：OpenRouter 推出了 **Phi-3 Medium 128k Instruct**，这是一个强大的 140 亿参数模型，并邀请用户查看[标准版](https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct)和[免费版](https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct:free)变体，并参与其有效性的讨论。

- **Wizard 模型获得魔力提升**：**Wizard model** 表现出改进，展现出更迅速且更具想象力的响应，但仍需注意避免重复段落。

- **关注 Phi-3 Vision 和 CogVLM2**：围绕 **Phi-3 Vision** 的热情高涨，分享了如 [Phi-3 Vision](https://ai.azure.com/explore/models/Phi-3-vision-128k-instruct/version/1/registry/azureml) 的测试链接，并建议在 [CogVLM-CogAgent](https://huggingface.co/spaces/THUDM/CogVLM-CogAgent) 中使用 **CogVLM2** 处理以视觉为中心的任务。

- **Llama 3 Prompt 自动转换**：澄清了发往 **Llama 3** 模型的 Prompt 会通过 OpenRouter 的 API 自动转换，从而简化流程，但手动 Prompt 仍作为一种替代方法保留。

- **Gemini API 的烦恼**：用户报告了 **Gemini FLASH** API 的问题，如空输出和 Token 耗尽，这被认为是由于模型本身的问题。Google 每日 API 使用限制的出现也引起了人们对这可能如何影响 OpenRouter 的 Gemini 集成的关注。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **显微镜下的 LLM 评估**：一篇关于大语言模型 (LLM) 评估实践、排行榜重要性以及细致的非回归测试的 [Hugging Face 博客文章](https://huggingface.co/blog/clefourrier/llm-evaluation) 引起了成员们的注意，强调了此类评估在 AI 发展中的关键作用。

- **AI 对搜索引擎操纵的回应**：一起涉及网站中毒并影响 Google AI 汇总概览的事件引发了围绕安全和数据完整性的讨论，包括 [Mark Riedl 的推文](https://x.com/mark_riedl/status/1793375699967054334) 中提到的通过自定义搜索引擎浏览器绕过的解决方法。

- **AI 是让开发民主化还是引发了可靠性问题？**：GitHub CEO Thomas Dohmke 关于 AI 在简化代码编写中作用的 [TED 演讲](https://youtu.be/nv9WwHpOKEg?si=mVApo6UnrtJ9ExH6) 引发了对其可靠性的争论，尽管 AI 驱动的 UX 改进加快了编码过程中的问题解决。

- **弥合差距的多样性奖学金**：针对因财务障碍无法参加即将举行的 AI Engineer World's Fair 的不同背景工程师，官方宣布了多样性奖学金。有兴趣的申请人应在[申请表](https://docs.google.com/forms/d/e/1FAIpQLScff_RUv-fIKfdj_2HcHtk96iy45GD0BWLByGxqdBqvcepDHg/viewform?usp=sf_link)中对论文问题提供*简洁*的回答。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **纳税故事：无需信用卡**：Nathan Lambert 破解了一场发票纠纷，意识到由于转售证书（resale certificates）的存在，无需信用卡进行税务结算的合理性。

- **Golden Gate AI 引起关注**：[Anthropic AI](https://x.com/anthropicai/status/1793741051867615494?s=46) 的实验诞生了 "Golden Gate Claude"，这是一个一心只关注金门大桥的 AI，因其在 claude.ai 上的公开互动性而引发热议。

- **Google 的 AI 失误**：Google 未能有效利用反馈以及过早部署 AI 模型，引发了关于这家科技巨头公关挑战和产品开发困境的讨论。

- **反击数据集误解**：Google 的 AI 团队反驳了关于使用 LAION-5B 数据集的说法，指出他们使用的是更优越的内部数据集，如最近的一条 [tweet](https://x.com/giffmana/status/1793906145310228538) 所述。

- **Nathan 分享知识点**：针对 AI 爱好者，Nathan Lambert 上传了高级 [CS224N 讲座幻灯片](https://docs.google.com/presentation/d/1on5xTePaUYg47vui3dUr0Lp6GUXmOXmhceNJLXRbGsE/edit)。此外，与会者还收到了关于即将发布的会议录像的提示，但未透露具体发布日期。



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **GQA 在 CMDR 模型中受到关注**：讨论显示 **Grouped Query Attention (GQA)** 存在于 "cmdr+" 模型中，但不存在于基础 "cmdr" 模型中，这表明了它们规格上的重要区别。
- **智能注意力机制提升 VRAM 效率**：工程师指出，虽然 **GQA** 不提供线性缩放，但与指数缩放相比，它代表了一种改进的缩放方法，对 **VRAM** 使用产生了积极影响。
- **样本打包（Sample Packing）获得提升**：一个新的 **GitHub pull request** 展示了样本打包效率提升了 3-4%，有望为分布式上下文提供更好的资源管理，链接见 [此处](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1619)。
- **学术成就获得认可**：一名成员合著的期刊文章已在 **Journal of the American Medical Informatics Association** 上发表，强调了高质量、混合域数据对医疗语言模型的影响，文章见 [此处](https://doi.org/10.1093/jamia/ocae120)。
- **社区庆祝学术成功**：社区通过个人祝贺信息表达了对同行发表作品的支持，培养了 AI 领域内认可学术贡献的文化。



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**SB-1047 引发技术动荡**：工程师们对 **SB-1047** 法案的影响表示深切担忧，称其对小型 AI 参与者有害，并将这种情况比作在其他行业观察到的监管俘获（regulatory capture）。

**展示行业工具 Perplexity 和 Arc**：社区重点介绍了辅助工作流的工具，分享了关于 [SB-1047 的 Perplexity AI 搜索](https://www.perplexity.ai/search/SB-1047-Senate-2kZmFYHoTxe.rWUYat4B2A) 以及 Arc Browser 的新功能 “Call Arc”，该功能简化了在线查找相关答案的过程，附带信息 [链接](https://arc.net/e/C56904FA-1C75-4D77-9A87-E7F1A52529CD)。

**安装问题引发询问**：用户在使用 pip 安装 **Typer** 库时遇到问题，引发了关于是否遵循了设置过程中的步骤（如在 `poetry run` 之前执行 `poetry install`）或是否使用了虚拟环境的疑问。



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

**Twinny 作为虚拟副驾驶起飞**：开发者正在将 [Twinny](https://github.com/rjmacarthy/twinny) 与 LM Studio 集成，作为强大的本地 AI 代码补全工具，支持在不同端口上运行多个 llamafiles。

**Embedding 端点说明**：根据 [pull request #4681](https://github.com/ggerganov/llama.cpp/pull/4681)，澄清了 `/v1/embeddings` 端点不支持 `image_data`；相反，图像应使用 `/embedding` 端点。

**Mac M2 在 continue.dev 中遇到对手**：一项性能观察指出，在使用 llamafile 执行时，continue.dev 在 Mac M2 上的运行速度比旧款 Nvidia GPU 慢。

**拥抱你自己的 LLM**：对于那些希望构建和训练自定义 LLM 的用户，社区推荐使用 [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) 进行训练，并提醒 llamafile 是为推理而非训练设计的。



---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **社区中回荡着感激之情**：一位用户向团队表达了衷心的 *感谢*，展示了用户对团队支持或开发工作的认可。
- **对扩展模型的好奇**：关于是否会有 **104B 版本**的模型加入家族树的讨论非常热烈，但目前尚未有明确的答复。
- **Langchain 链接缺失**：有用户询问 **Langchain** 与 Cohere 的集成情况，寻求关于当前可用性和实现状态的指导。
- **模型尺寸之谜**：用户正在寻求澄清 Playground 中的 **Aya model** 是 8B 还是 35B 版本，这表明理解模型规模对应用至关重要。
- **错误排查角落**：诸如 **ContextualCompressionRetriever** 的 `ValidationError` 和 **403 Forbidden error** 等问题标志着工程师们正在进行活跃的调试和技术问题解决，这提醒了 AI 开发中常见的挑战。

---

## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

**AI 喜剧之夜大获成功**：用户分享的一段 AI 生成的脱口秀表演获得了积极的反响，表明 AI 在模仿幽默和进行娱乐表演方面的能力有所提升。

**AI 应用的探索性查询**：用户询问 [Ud.io](https://www.udio.com/songs/vsNF2nbsy646jGt348mdFG) 的功能是否超出了生成喜剧的范畴，表现出对其功能边界的好奇。

**声音变换展示**：一位用户展示了 [Suno](https://suno.com/song/e6b62587-4345-44fb-85c7-c51f932df655) 灵活的音频修改功能，分享了一个原始音轨的“恶魔化”修改版本。

**对音频工程知识的渴望**：有用户表示有兴趣学习如何制作演示中的音频修改，这对于对声音处理感兴趣的 AI 工程师来说是一项宝贵的技能。

**倾向于简洁的沟通**：对一个问题的单字回复“No”突显了对简洁回复的偏好，这可能反映了工程师对直接、务实沟通的追求。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **寻找统一的事件追踪器**：一位成员强调了对兼容 Google Calendar 的事件日历的紧迫需求，以确保不会错过任何社区活动。社区内注意到缺乏此类系统是一个令人担忧的问题。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **新数据集发布公告**：用户 datarevised 提到了一个新数据集，并附带了更多详情链接：[DataPlusEngine Tweet](https://x.com/DataPlusEngine/status/1793803117642854732)。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **基准测试驱动创新**：[GLUE](https://arxiv.org/abs/1804.07461)、[MMLU](https://arxiv.org/abs/2009.03300) 和 [GSM8K](https://arxiv.org/abs/2110.14168) 等评估基准对 AI 研究进展至关重要，而“基于执行的评估 (execution based evals)”在动态任务中面临特殊挑战，正如成员们分享的一篇 [博客文章](https://www.jasonwei.net/blog/evals) 中所讨论的那样。
  
- **对 GPT-5 的期待**：一次谈话引起了成员们的猜测，认为 **GPT-5** 可能会在 2024 年首次亮相，并向“Agent 架构”迈进，正如一条 [传闻推文](https://x.com/rohanpaul_ai/status/1793956355897724973?s=46&t=90xQ8sGy63D2OtiaoGJuww) 所暗示的那样。

- **探讨 AI 在音乐中的角色**：成员们探讨了关于 **Suno** 等 AI 创作音乐的版权问题，思考了 Meta 的 **Audiocraft** 的能力，并讨论了法律影响以及促进创作自由的开源工作，包括 [GitHub](https://github.com/betweentwomidnights/gary-backend-combined) 上的 **gary-backend-combined**。

- **Python 开发者为 NumPy 2.0 做准备**：人们对 **NumPy 2.0** 的期待日益增长，成员们还开玩笑说这可能对依赖管理产生影响，正如一篇 [Twitter 帖子](https://x.com/cgarciae88/status/1794019900119236874) 所指出的。

- **xAI 获得巨额投资**：在投资新闻方面，**xAI** 获得了 60 亿美元的 B 轮融资，目标是快速提升其模型能力，如其 [公告文章](https://x.ai/blog/series-b) 所示。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **社区集思广益修复 Nightly 版本困扰**：在成员们指出 **Mojo VSCode 扩展**仅对安装后的文件有效，且自 24.3 版本以来一直存在 Bug 后，解决方案包括关闭/重新打开代码以及频繁重置。相反，**Mojo 编译器**的 Nightly 更新（`2024.5.2505` 到 `2024.5.2705`）包含了诸如 LSP 中的变量重命名和新的 `tempfile` 模块等增强功能，尽管它们在 GitHub 上的 [#2832](https://github.com/modularml/mojo/pull/2832) 等现有 PR 中引入了测试问题。

- **在 Mojo 中实现 ZIP 的禅意**：在 Mojo 中复制 Python 的 `zip` 和 `unzip` 函数时遇到的困难，引发了关于如何声明根据变长列表参数返回元组（tuples）的函数的讨论。对话阐明了 Mojo 使用新的 `ref` 语法实现自动解引用迭代器的潜力，旨在简化实现并减少显式的解引用操作。

- **不同处理器的性能表现差异**：一位成员在 Mojo 中优化 CRC32 计算的努力导致了不同核心类型之间的性能差异；由于 L1 缓存限制，紧凑型实现在处理较大字节大小时在性能核心上表现滞后，但效率核心（E-cores）则更青睐紧凑版本。基准测试元数据和版本控制文件位于 [fnands.com](https://github.com/fnands/fnands.com/blob/main/blog/2024/mojo-crc-calc/crc.mojo) 以及 [上述内容的 Nightly 版本测试](https://github.com/fnands/fnands.com/blob/main/blog/2024/mojo-crc-calc/crcn.mojo)。

- **关于 Mojo 潜力的思考**：关于 **Mojo** 的一般性讨论涵盖了与 Python 相比的函数行为差异、可选类型（optional types）的处理、LinkedLists 的实现，以及对 Mojo 尚未实现的反射（reflection）能力的思考。成员们就使用 `UnsafePointer` 和 `Optional` 进行高效初始化交换了意见。

- **科技巨头的巨额交易**：AI 社区消化了 Elon Musk 的 AI 初创公司 xAI 完成 60 亿美元 B 轮融资的消息，正如 [TechCrunch 文章](https://techcrunch.com/2024/05/26/elon-musks-xai-raises-6b-from-valor-a16z-and-sequoia/) 所述，这使 xAI 能够直接对抗 OpenAI 和 Microsoft 等 AI 领跑者，引发了围绕 AI 技术商业化的讨论。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**新成员加入：Phi-3 模型上线**：Microsoft 的 [phi-3-medium-128k-instruct](https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct) 和 [phi-3-mini-128k-instruct](https://openrouter.ai/models/microsoft/phi-3-mini-128k-instruct) 模型现已上线，同时 [llama-3-lumimaid-70b](https://openrouter.ai/models/neversleep/llama-3-lumimaid-70b) 模型提供 57% 的特别折扣。

**速率限制迷宫解析**：OpenRouter 上 **rate limiting**（速率限制）带来的挑战引发了激烈讨论，强调了理解信用余额如何影响请求速率的重要性，如 [OpenRouter 文档](https://openrouter.ai/docs#rate-limits-and-credits-remaining) 中所述。

**模式混乱：当额度与速率限制冲突时**：模态回退（modal fallback）功能出现了一个令人困惑的问题，即尽管信用余额充足，仍触发了速率限制。社区建议监控免费请求，并在限制临近时考虑停用免费模型。

**AI 的自我审查困境削弱了吸引力**：爱好者们表示担心，Claude 自我审查模型中更严格的护栏和更高的拒绝率导致了不够拟人化的体验，这可能导致使用量下降。

**视觉模型分析：性能与价格**：话题转向了视觉模型的性能，特别是 Gemini 的 OCR 能力，并对其与传统视觉服务相比的成本效益表示认可。对话还强调了通过 RunPod 和 Vast.ai 使用 GPU 比使用 Google Cloud 和 Amazon Bedrock 等主流云服务更便宜。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **CCS 表现不尽如人意**：Eleuther 团队对 Collin Burns 的 **Contrast Consistent Search (CCS)** 方法进行的后续研究未能如预期般提升泛化能力，这一点在 [Quirky Models 基准测试](https://arxiv.org/abs/2312.01037)中得到了证实。他们在详细的[博客文章](https://blog.eleuther.ai/vincs/)中分享了该方法及其平庸的结果，这种透明度受到了赞赏。

- **最优模型提取策略引发热议**：工程师们就 **RAG** 在从自定义库中提取数据方面是否优于 **LLM** 的微调（finetuning）展开了辩论，结论是 RAG 可能更好地保留信息。Hugging Face 上发布的 **ThePitbull 21.4B 模型**引起了关注，但对其宣称的接近 70B 模型性能的说法，部分人持怀疑态度。

- **数据复制故障排除**：AI 程序员在复制 Pythia 数据时遇到了分词器（tokenizer）难题，解决方案包括使用 `batch_viewer.py` 以及通过 `MMapIndexedDataset` 进行正确的字段类型处理。正如 [Pythia 仓库](https://github.com/EleutherAI/pythia#exploring-the-dataset)中所述，虽然这一过程被比作“黑魔法”，但对于正确解析数据集是必不可少的。

- **新技术挑战极限**：工程师们深入讨论了用于分配权重更新的梯度扰动方法，以及 Transformer 在参数化知识上进行隐式推理的潜力。一篇关于“无调度”（schedule-free）优化的新论文引起了关注，该论文建议在不增加额外超参数的情况下进行迭代平均，其效果可能优于传统的学习率调度（learning rate schedules）。

- **量化困境**：在追求效率的过程中，关于小模型是否能在某些情况下超越大型量化模型的讨论随之展开。引用 [ggerganov 的 Twitter](https://x.com/ggerganov/status/1666087050725199872) 暗示了量化方法的潜力，激发了关于模型大小与性能平衡的辩论。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **RAG 革命：企业级流水线即将来临**：**LlamaIndex 团队**宣布将与 AWS 合作举办研讨会，展示如何使用 AWS Bedrock 和 Ragas 构建**企业级 RAG 流水线**。该活动旨在提供将 Bedrock 与 LlamaIndex 集成以优化 RAG 系统的见解，目前已开放注册 [点击此处报名](https://lu.ma/x8unmku0)。

- **提升检索效率的创新**：讨论重点关注了 **Vespa 集成**带来的混合搜索能力提升，以及 Jayita B. 撰写的关于使用 Llama3 和 GroqInc 创建**快速响应 RAG 聊天机器人**的高级指南。此外，还提到了一种通过 **gpt4o** 等模型生成的结构化注释对图像进行索引的创新方法。

- **文件组织进阶**：**LlamaFS** 作为一种自动整理杂乱目录的新工具发布，这可能会引起那些寻求简洁高效文件管理解决方案的人的共鸣。

- **技术故障排除成为焦点**：AI 工程师们正在解决与 **Llama Index reAct** 相关的问题，包括通过 `max_iterations` 设置的变通方法，以及通过对齐软件包版本来克服导入错误。与 PDF 文件相比，HTML 解析通常需要更多自定义代码，而 PDF 可以利用依赖项更少的高级分块工具。

- **使用 Pydantic 实现 LlamaIndex 结构化输出**：关于在 LlamaIndex 中使用 **Pydantic 模型**的指南标志着向结构化输出集成迈进了一步，预示着该系统更广泛的应用和可用性。对改进检索器文档的呼吁凸显了社区在不同项目中增强对 **BM25** 和 **AutoRetrieval** 模块理解与应用的动力。

## [Cohere](https://discord.com/channels/954421988141711382) Discord

**Aya-23 登场**：工程师们讨论了 **Aya-23 的多语言能力**，并将其与 **Command R/R+** 进行了对比，暗示其性能更优，但也对其在英语环境下的效率提出了疑问。他们还指出 **Aya-23-35b** 是 **Command R** 的微调版本，并提供了[技术报告](https://drive.google.com/file/d/1YKBPo61pnl97C1c_1C2ZVOnPhqf7MLSc/view)以供查阅更多细节。

**移动端隐私与 LLM 局限性**：社区达成共识，认为**手机端 LLM** 目前还不足以在移动应用中实现私密的本地运行，特别是对于通常需要 **RAG 移动应用** 配合完成的任务。

**机器人创新蓬勃发展**：一位社区成员在 LinkedIn 上展示了一个游戏机器人，因其集成了 **Cohere Command R** 而引起关注；与此同时，Discord 的 "Create 'n' Play" 机器人号称拥有 ***“超过 100 款引人入胜的文字游戏”***，并能*通过 AI 增强社交互动*。

**Prompt 的适配与集成**：公会确认 **Aya-23 支持 system prompts**，并分享了关于如何使用特定 token（如 `<|USER_TOKEN|>` 和 `<|CHATBOT_TOKEN|>`）来适配 **Command R** 的 prompt 以使其有效运行的见解。

**OneDrive 同步解决方案**：针对关于 **OneDrive 连接器** 的咨询，社区推荐了一个 [SharePoint 连接器](https://github.com/cohere-ai/quick-start-connectors/tree/main/sharepoint)，它可能满足类似的集成需求。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

**AI 的“跳桥”建议**：成员们分享了一个关于 Google AI 给出“跳桥以治疗抑郁症”危险建议的幽默段子，以此讽刺 Reddit 建议的误导性。社区还分享了一个与此失误相关的表情包。

**ConvNeXt 获得优化**：关于 [ConvNeXt 论文](https://arxiv.org/abs/2405.15738) 的热烈讨论赞扬了其高效处理高分辨率图像的能力，这有可能减少过多的视觉 token 生成，并简化高分辨率任务的优化流程。

**从红石到神经网络**：展示了数据集和 AI 工具的创新用途，包括来自 [archive.org](https://archive.org/details/arxiv-bulk?sort=-publicdate) 的出版物 PDF 和 TeX 源码数据集，以及一段演示如何用红石（Redstone）构建神经网络的 [YouTube 视频](https://youtu.be/DQ0lCm0J3PM?si=5Is7OMnqRhZb-ZAo)。

**AI 预训练中的模型增长堆叠**：一篇 [arXiv 论文](https://arxiv.org/abs/2405.15319) 引起了关注，该论文强调深度堆叠（depthwise stacking）是大型语言模型（LLM）高效预训练中一种有效的模型增长方法，解决了预训练过程中的关键速度和性能挑战。

**PyTorch 持久化中的陷阱**：学习领域的讨论集中在解决训练-验证集划分的随机性问题以及模型重新加载时的 loss 不一致问题。具体而言，在 PyTorch 中正确保存 optimizer states 被指出是避免 loss 爆炸的关键。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Llama3 德语化**：全新的 **Llama3-German-8B** 模型将 Meta 的 Llama3-8B 能力扩展到了德语，在 650 亿个 token 上进行了训练，且英语性能损失微乎其微；详情见 [Hugging Face](https://huggingface.co/DiscoResearch/Llama3-German-8B)。然而，值得注意的是，与其他语言模型不同，该训练**省略了英语数据回放 (English data replay)**，这引发了关于其有效性的辩论。

- **量化的怪癖与谜团**：Llama3 的量化版本 **GGUF** 在基准测试中表现不佳，暗示了潜在的问题，可在 [Llama3-DiscoLeo-Instruct-8B-32k-v0.1-GGUF](https://huggingface.co/cstr/Llama3-DiscoLeo-Instruct-8B-32k-v0.1-GGUF) 查看。同时，关于参数设置和奇怪输出的讨论暗示了运行这些模型的复杂挑战，例如 max tokens 和引擎的选择。

- **Cohere 的多语言飞跃**：**Cohere 的 Aya-23-35B 模型**虽然在许可方面有所限制，但现在支持 23 种语言，这表明了业界对强大多语言模型日益增长的趋势和兴趣。一个相关的用于翻译的 **ShareGPT 格式数据集**引发了关于质量和过滤的讨论，托管在 [这里](https://huggingface.co/datasets/sroecker/aya_german-sharegpt)。

- **Mistral 的微调银河指南**：为了向技术社区致敬，Mistral 推出了针对 Mixtral 模型的微调指南，为那些开启微调冒险的人指明了方向；该指南可以在其 [GitHub](https://github.com/mistralai/mistral-finetune) 上查阅。

- **模型微调细节曝光**：社区正在使用 ***oobabooga*** 和 ***ollama*** 进行实验，涉及 **'skip_special_tokens'** 开关和停止 token 设置，包括建议使用特定的 [Llama3 模板](https://github.com/CrispStrobe/EQ-Bench/blob/main_v2_3a/instruction-templates/Llama3.yaml) 来解决输出问题——这反映了成员中盛行的积极调整和调优文化。



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **LAM 安全担忧与移动性**：尽管 [Rabbit LAM 架构概览](https://engineering.rabbit.tech/lam-security-architecture-overview) 展示了其安全的 Web 应用控制能力，但讨论中仍流露出对 **Large Action Model (LAM)** 的怀疑，原因是其存在被逆向工程的风险。对话还围绕在 **Rabbit R1** 设备和 **Humane 平台**上运行 01 模型的集成挑战，并重点介绍了用户驱动的解决方案，如 [01 模型 Android GitHub 仓库](https://github.com/Tonylib/o1_for_flutter)。

- **即时安装策略**：Python 版本升级到 **3.11 解决了 Mac 上的 OpenInterpreter 问题**，而通过 [Andronix](https://andronix.app/) 折腾 Linux 发行版使得在 Android 驱动的耳机上使用 OpenInterpreter 成为可能。关于 **Open Interpreter** 安装的咨询显示出对本地或云端可访问 LLM 的需求，并且推出了新的 [Markdown 导出功能](https://github.com/OpenInterpreter/open-interpreter/pull/1282) 以帮助开发者。

- **社区中的 DIY 精神**：一位用户通过升级到 Python 3.11 修复了 Mac 上的 OpenInterpreter 问题，其他用户分享了增强现有设备的途径，例如使用 **LineageOS** 修改 **R1**。在购买还是自建硬件的讨论中，共识是购买预构建的 O1 有利于西雅图开发团队，尽管与自建版本相比没有技术差异。

- **发货与存储备受审视**：由于**缺乏硬件发货更新**，特别是关于欧洲分销和预订信息封锁，社区表达了沮丧。对话还涉及寻找解决 Runpod 上**磁盘空间限制**的方案，这暗示了 AI 工作负载中对高性价比数据存储的持续需求。 

- **OpenInterpreter 的跨越式进步**：一名成员成功在 Mac 上切换到 **Python 3.11** 以运行 OpenInterpreter，标志着社区在解决问题方面的持续敏捷性。同时，[新 Markdown 导出功能](https://github.com/OpenInterpreter/open-interpreter/pull/1282) 的实现反映了在 AI 工具链中增强开发者实用性的推动。



---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**PDF 提取挑战重重**：关于从 PDF 中提取文本的讨论强调了处理复杂表格和图表时的困难，并建议了基于机器学习的文本分割以及使用 Adobe Extract API 进行布局解析等解决方案，参考 [LangChain 文档](https://python.langchain.com/v0.2/docs/how_to/document_loader_pdf/)。

**LangChain 社区即将扩展**：来自 Scogo Networks 的 Karan Singh 表示有兴趣在孟买建立本地 LangChain 社区，并正在寻找市场联系人以组织活动。

**Langserve 等候名单出现波折**：用户在 Airtable 上遇到了 Langserve 等候名单的访问问题，正在寻找尝试该托管服务的替代方法。

**推出交互式数据可视化工具**：介绍了 NLAVIDA 项目，该项目通过自然语言促进交互式数据可视化和分析，并附带了 [YouTube 视频教程](https://www.youtube.com/watch?v=leJRP_mJsSQ&t=4s)。

**为 OranClick 投票**：OranClick 正式发布，这是一款旨在优化消息撰写以提高注册率的工具，并邀请大家在 [ProductHunt](https://producthunt.com/posts/oranclick) 上支持。



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Llamafile v0.8.5 发布**：重点介绍了 [llamafile 0.8.5 版本](https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8.5) 的发布，该版本为 X86 CPU 上的 K 量化提供了**快速推理**，并呼吁社区使用 `llamafile-bench` 进行基准测试。
- **Llamafile 具备网络访问能力**：工程师们交流了如何使 llamafile 服务器可进行网络访问的技巧，建议使用 `--host <my ip>` 或 `--host 0.0.0.0` 等标志，以便在同一网络内的跨机器可用。
- **Llama3-70B 的空白响应之谜**：贡献者报告称遇到了来自 llama3-70b 模型的**空白响应**，并分享了日志供社区主导的调试，尽管目前尚未出现明确的解决方案。
- **为 Home Assistant 提供本地 API**：关于增强 **Home Assistant** 集成的讨论非常热烈，重点是开发一个类似于 OpenAI 候选方案的标准本地 API，并强调了 API 可发现性和安全 API 端点等功能的重要性。
- **模型选择的 Python 难题**：问题和分享的代码片段表明，在 LLaMA_CPP 集成的 Python 示例中指定模型时存在一些困惑，特别是关于何时必须指定模型（例如使用 TinyLlama 时）。



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Mistral-Finetune 神秘面纱揭开**：开发者们致力于解读 [Mistral-Finetune 仓库](https://github.com/mistralai/mistral-finetune) 中的新更新，重点在于理解其独特的变更。
- **MoEs 微调的怪癖**：AI 圈讨论了微调 **Mixture of Experts (MoEs)** 模型的不确定性，强调了运行多次迭代以挑选最有效模型的必要性，尽管关于成功率的细节尚未披露。
- **Aya 23 受限的能力**：**Aya 23** 的局限性引发了热烈讨论，强调其在基于聊天的应用中表现不佳，其优势仅限于特定任务，详见其 [技术报告](https://cohere.com/research/papers/aya-command-23-8b-and-35b-technical-report-2024-05-23)。
- **MoRA 步入微调聚光灯下**：**MoRA** 作为一种用于微调的前沿高秩更新方法进入讨论，具有补充或超越 LoRA 的潜力，并附有 [专用 GitHub 仓库](https://github.com/kongds/MoRA)。
- **FFD Bin Packing 问题与 Llama 3 Token 备受关注**：出现了关于 FFD bin packing 实现的问题，特别是在分布式训练上下文中；以及针对 Llama 3 未训练 Token 的修复，并分享了 [sfttrainer 的补丁](https://github.com/unslothai/unsloth/blob/main/unsloth/tokenizer_utils.py#L722) 来解决后者。



---

## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **通过 Virtual Beings 模拟现实**：[Virtual Beings Summit](https://www.virtual-beings-summit.com) 对 AI 专业人士来说可能是一个非常有价值的活动，**Will Wright** 将出席该活动，重点关注 AI 与模拟（simulations）的交叉领域。相关支持内容可以在 [Virtual Beings YouTube 频道](https://www.youtube.com/channel/UCLNalUbhs_EYB5IAbw1VfaA)上找到，该频道提供了关于 AI 在交互式模拟中作用的见解。

- **使用 DIAMOND 构建 AI 梦境**：[DIAMOND GitHub 仓库](https://github.com/eloialonso/diamond)介绍了 “DIAMOND (DIffusion As a Model Of eNvironment Dreams)”，它在 reinforcement learning 环境中使用 diffusion models 来增强 AI 模拟中的环境交互。

- **使用 UE5 和 4Wall 打造 AI 版“西部世界”**：关于创建沉浸式体验的讨论表明，AI Town 可能会利用 **UE5** 并集成语音控制来模拟类似于“西部世界”的环境，持续的开发信息可以在 [4Wall Discord](https://discord.gg/vPum4s3h) 获取。

- **进军虚拟现实 (VR)**：将 AI Town 与 **VR** 技术结合的想法受到了热烈欢迎，这表明工程师们正在考虑通过 VR 将 AI 生成的环境变为现实的新方法。

- **使用 SadTalker 和 V-Express 制作头像动画**：两个 GitHub 仓库 [SadTalker](https://github.com/OpenTalker/SadTalker) 和 [Tencent AI Lab 的 V-Express](https://github.com/tencent-ailab/V-Express) 分别提供了用于创建逼真的说话人脸动画和生成说话头像视频的工具，展示了风格化动画技术的进步。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**Zyphra Zamba 崭露头角**：融合了 mamba 和 attention 机制的新型 **Zyphra Zamba** 模型已经发布，并附带了相应的[技术报告](https://www.zyphra.com/s/Zamba.pdf)、[PyTorch 代码](https://github.com/Zyphra/Zamba-torch)以及对 [Hugging Face Transformers](https://github.com/huggingface/transformers/pull/30950) 的集成。目前正在进行与 **OLMo 1.7** 的对比分析，以评估其性能基准。

**SD Audio 2.0 低调发布**：**SD Audio 2.0** 的一个未经授权版本出现在 4chan 上，同时也可以在一个 Hugging Face 账号上找到，这引发了成员们的讨论。

**各环节监管**：前 OpenAI 董事会成员 Hellen Toner 和 Tasha McCauley 在[《经济学人》(The Economist)](https://www.economist.com/by-invitation/2024/05/26/ai-firms-mustnt-govern-themselves-say-ex-members-of-openais-board)上提议对 AI 公司进行严格监管，强调由于利润动机，此类公司无法进行自我监管，并指出了过去的内部问题。

**管理层的争议**：文章批评了 Sam Altman 在任职期间涉嫌的“有毒的谎言文化”，讨论了内部调查以及公众对缺乏透明度的抗议。

**RL 的教科书案例**：社区在 GitHub 上分享了一个新资源——[关于 reinforcement learning from human feedback (RLHF) 的教科书](https://github.com/natolambert/rlhf-book)，并赞扬了 Chris Potts 和 Chris Manning 教授引人入胜的教学风格。讨论内容还包括斯坦福 CS224n 课程的电子版何时发布，并建议联系 Chris 以获取具体的时间表。



---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**调整技术测试中的时间限制**：讨论涉及将每个测试的时间限制延长至 **9 分 34 秒** 以上的可能性，以适应如 'Taylor approximations' 等复杂函数。一个具体问题是 `clang` 函数无法完成，仅达到约 60% 的进度。

**编译崩溃需要解决方案**：一位成员指出，生成*过大的表达式*会导致编译器崩溃，并出现与不兼容操作数类型（特别是 double）相关的错误。

**Double 类型的位运算争议**：澄清了无法在 `double` 数据类型上执行诸如 **XOR** 之类的位运算，解决了成员们观察到的编译错误原因。

**悬赏猎人热度上升**：对各种研究导向型悬赏的兴趣激增，讨论了旧的 pull requests，**George Hotz** 确认了诸如 [tinygrad pull request #4212](https://github.com/tinygrad/tinygrad/pull/4212) 中提到的悬赏仍然有效。

**解读 'vin' 并讨论 Dominators**：George Hotz 澄清了 **UOp class** 中的 'vin' 不是缩写。此外，一位成员询问为什么不使用 post dominator analysis 来改进模型中的 scheduling，并建议这可能会在执行期间优化 subgraph fusion。

---

**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该服务器沉寂太久，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该服务器沉寂太久，请告知我们，我们将将其移除。

---

**Datasette - LLM (@SimonW) Discord** 没有新消息。如果该服务器沉寂太久，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器沉寂太久，请告知我们，我们将将其移除。

---

**YAIG (a16z Infra) Discord** 没有新消息。如果该服务器沉寂太久，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道的详细摘要和链接

### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1243282801760145408)** (74 messages🔥🔥): 

- **语义相似度过拟合担忧**：一位成员思考，如果数据中响应类别的代表性过高（尽管没有特定的响应被过度代表），是否会导致偏差。他们提到了之前在研究心理学（Research Psychology）中检查此类问题的经验。
- **微调模型困惑**：一位用户在理解 Fine-tuning 与 Pre-training 相比，在多大程度上将特定的用户输入整合到模型中感到困惑。他们寻求关于 Pre-training、Curriculum training 和 Fine-tuning 之间区别的澄清。
- **OpenAI 平台侧边栏变化**：一些参与者讨论了 OpenAI 平台侧边栏的变化，提到**两个图标消失了**（一个是线程图标，另一个是消息图标）。
- **Rasa 与对话复杂性**：一位参与者分享了对 Rasa 对话式 AI 方法的见解，强调了由于对话复杂性而创建意图分类器（Intent classifiers）的难度。他们提到，将意图视为实体可能会降低复杂性。
- **Kyle Corbitt 的会议演讲录像已发布**：Kyle Corbitt 的会议演讲录像现在可以在 Maven 门户上观看，讨论中分享了具体链接。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://llm-tracker.info/research/Quantization-Overview">Quantization Overview</a>: 测试量化如何影响模型输出？ - 对不同量化级别的 15 项基础测试。GPTQ, AWQ, EXL2, q4_K_M, q4_K_S 和 load_in_4bit 的详细对比：Perplexity, VRAM, Speed, mo...</li><li><a href="https://hamel.dev/notes/llm/inference/03_inference.html">Hamel’s Blog - Optimizing latency</a>: 关于如何优化延迟（Latency）的探索。</li><li><a href="https://support.zoom.com/hc/en/article?id=zm_kb&sysparm_article=KB0066245">未找到标题</a>: 未找到描述</li><li><a href="https://tenor.com/view/food-good-hungry-yum-gif-11656939384713462119">Food Good GIF - Food Good Hungry - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=d8JMJMvErSg&ab_channel=Rasa">Rasa Algorithm Whiteboard - TED in Practice</a>: 在这段视频中，我们将探讨 TED 在实践中是如何工作的。我们将构建一个需要倒计时的数字助手，并会看到超参数（Hyperparameters）确实...</li><li><a href="https://www.youtube.com/watch?v=j90NvurJI4I&ab_channel=Rasa">Rasa Algorithm Whiteboard - TED Policy</a>: 在制作数字助手时，你需要的不仅仅是处理文本的算法。你还需要处理对话序列的算法...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7513)">Issues · ggerganov/llama.cpp</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号为 llama.cpp 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cltac3/part3_cause_to_issue_found_possible_bug_llama3/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://us06web.zoom.us/rec/share/POky_IXJdWGOOGZ9BMORn2lZQI53F3d_sOMmESWRbvUm3Us8cWNB7v2rdqnF4raB.95CQod940HlUWGjB?startTime=1716504965000">Video Conferencing, Web Conferencing, Webinars, Screen Sharing</a>: Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，用于移动、桌面和会议室系统的视频和音频会议、聊天和网络研讨会。</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1243336501018755123)** (23 messages🔥): 

- **LLM Finetuning 和 `###` 使用说明**：讨论了在 LLM 微调中用于序列生成的 `###` 使用方法，指出这有助于模型在推理（inference）过程中理解输入的不同部分。在微调期间需要正确配置模板，包括 ChatML 等其他结构。
  
- **模板要求说明**：强调推理时的输入需要与微调时使用的模板匹配，不一定非要是 `###`，而是微调时设置的任何内容（例如 Llama 2 chat 模板）。模型托管服务通常会管理这些模板和结构。

- **带分隔符与不带分隔符的模型行为**：分隔符可以帮助模型理解输入中的不同部分（如 Reddit 中视角的切换）；否则对于一般的风格适配来说是不必要的。终止分隔符或 Token 可确保模型正确解析并结束响应。

- **End of text token 的使用**：简要提到了 "end of text" token 的概念，作为指示模型停止生成 Token 的机制，这标志着 LLM 的高效输入和输出管理。

- **关于 LLM 使用案例的家庭作业**：成员们分享并讨论了将 LLM 应用于生成食谱和学习应用等任务的作业项目。项目重点强调了 Prompt engineering 和 Retrieval-augmented generation (RAG) 等技术。资源链接和作业详情见[此处](https://maven.com/parlance-labs/fine-tuning/1/syllabus/modules/918c75?item=v69y1k7ohye)。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://maven.com/parlance-labs/fine-tuning/1/syllabus/modules/918c75?item=v69y1k7ohye">未找到标题</a>: 未找到描述</li><li><a href="https://gpus.llm-utils.org/llama-2-prompt-template/.">Llama 2 Prompt Template</a>: 针对 Llama 2 chat 模型进行提示（prompting）的最佳实践模板是什么？
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/1243344778511515698)** (8 messages🔥): 

- **Reka.ai 关于重聚的玩笑**：一位成员幽默地评论了很久之后再次见到另一位成员，开玩笑说：*"你太客气了！我开始以为在 fast.ai 之后我再也见不到天日了。"* 他们询问了彼此近况以及目前正在构建的项目。
- **会议录像请求已满足**：一位成员请求“Conference Talk: From prompt to model”的录像（该会议在印度标准时间凌晨 4:30 举行）。该请求得到了肯定答复，录像现已在 **Maven** 上提供。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1243309176722030702)** (18 条消息🔥): 

- **Modal 积分领取反响热烈**：多位用户确认收到了来自 Modal 的积分，并表达了开始微调模型的渴望。一位用户表示：*"是时候动手搞点东西了。"*。
- **关于在 Modal 上使用纯 PyTorch 代码的疑问**：一位用户询问是否可以在 Modal 上使用纯 PyTorch 代码进行 LLM 微调，并将其与使用 Jarvis Labs 进行了对比。另一位用户确认这是可行的，并分享了他们在 Modal 上训练 SentenceTransformer 模型的经验。
- **Modal 中的数据集管理**：讨论涉及如何上传数据集并在 Modal 中使用它们，并提供了详细的代码示例和步骤。Steven Merrill 演示了如何设置 Parquet 文件、构建 volumes 以及使用 GPU 元数据注解函数。
- **Modal 文档与示例**：用户分享了有用的 Modal 文档和示例链接，包括 [volumes 文档](https://modal.com/docs/guide/volumes) 和一个 [TensorFlow 教程](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/tensorflow/tensorflow_tutorial.py)，后者可以适配用于 PyTorch。
- **将 Modal 用于 Kaggle 竞赛**：一位用户计划利用 Modal 参加 Kaggle 竞赛，涉及下载数据、安装库、微调以及保存模型/日志。另一位用户提到在 Modal 上运行 Jupyter 服务器长达 24 小时，并分享了 [Jupyter inside Modal 示例](https://github.com/modal-labs/modal-examples/blob/0ca5778741d23a8c0b81ae78c9fb8cb6e9f9ac9e/11_notebooks/jupyter_inside_modal.py) 的链接。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/tensorflow/tensorflow_tutorial.py">modal-examples/06_gpu_and_ml/tensorflow/tensorflow_tutorial.py at main · modal-labs/modal-examples</a>：使用 Modal 构建的程序示例。可以通过在 GitHub 上创建账号来为 modal-labs/modal-examples 做出贡献。</li><li><a href="https://modal.com/docs/guide/volumes">Volumes</a>：modal.Volume 是为高性能文件服务构建的可变卷。与 modal.NetworkFileSystem 类似，这些卷可以同时挂载到多个 Modal 函数，支持并发...</li><li><a href="https://github.com/modal-labs/modal-examples/blob/0ca5778741d23a8c0b81ae78c9fb8cb6e9f9ac9e/11_notebooks/jupyter_inside_modal.py">modal-examples/11_notebooks/jupyter_inside_modal.py at 0ca5778741d23a8c0b81ae78c9fb8cb6e9f9ac9e · modal-labs/modal-examples</a>：使用 Modal 构建的程序示例。可以通过在 GitHub 上创建账号来为 modal-labs/modal-examples 做出贡献。
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jarvis-labs](https://discord.com/channels/1238365980128706560/1241117895740625099/1243307629057671229)** (16 条消息🔥): 

- **在 Jarvis 上保存 VSCode 仓库**：一位成员询问如何在不暂停实例以节省积分的情况下，在 Jarvis 的 VSCode 实例上保存他们的仓库。另一位成员建议将代码发布到 GitHub，并在需要时重新克隆，而**暂停的实例仅收取存储费用**，这部分费用非常低。
- **移除竞价实例 (spot instances)**：由于不稳定和利用率低的问题，平台暂时移除了竞价实例。
- **微调 open-lama-3b 的成本和时长**：在 RTX6000Ada 上使用 **gpt4-LLM-cleaned 数据**微调 **open-lama-3b** 耗时 3 小时 44 分钟，成本约为 4 美元。随后的讨论提到，LORA 权重体积较小，这解释了为什么上传到 Huggingface 看起来是瞬间完成的。
- **Axolotl 在 Ampere 系列上的错误**：一位用户在 A6000 上进行预处理时遇到错误，通过将 **bf16 改为 false** 并将 **fp16 改为 true** 解决了该问题。
- **课程注册积分问题**：一位用户反馈在注册课程并加入 Jarvis 后未收到积分；管理员回复称正在处理新名单，一旦收到用户信息就会添加积分。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1243335428887806004)** (9 条消息🔥): 

- **HF 积分即将发放**：成员们询问获取 HF 积分的流程。**详情将很快通过电子邮件公布**，积分将发放给填写了周末发送的表单的参与者。
- **西班牙语文本生成的最佳模型**：一位成员征求专门用于西班牙语文本生成任务微调的模型建议。**Mistral 7B** 被推荐为一个流畅的选择，**Llama 3** 也被提及，尽管它不是官方的多语言模型，但效果依然稳健。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1243453712182149150)** (1 messages): 

- **关于额度（Credits）的即将发布的公告**：关于额度管理和分配的公告即将发布。*"<@739531318571958272> 将负责运行这些额度，但我们很快会发布相关公告"*。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[kylecorbitt_prompt_to_model](https://discord.com/channels/1238365980128706560/1242221891733946490/1243287896652517376)** (164 messages🔥🔥): 

<ul>
    <li><strong>对讲座的高期待</strong>：成员们对讲座表示兴奋，尽管存在时差挑战，并呼吁进行录制。*"我真的很想看这个，但去不了 😦 会有录像吗？"*</li>
    <li><strong>链接溢出</strong>：分享了多个链接，包括 Hamel 的 [LLM inference notes](https://hamel.dev/notes/llm/inference/03_inference.html)、[Argilla](https://argilla.io/) 和 [MTEB Benchmark](https://huggingface.co/spaces/mteb/leaderboard)。从讲座中收集了大量资源。</li>
    <li><strong>互动且幽默的环节</strong>：成员们赞赏互动的氛围，并就 Fine-tuning 和睡眠时间表进行了幽默的交流。*"Fine-tuning 不仅在 GPU 计算方面昂贵，还影响了我们的睡眠时间表！"*</li>
    <li><strong>讨论高效 Fine-tuning 技术</strong>：讨论了各种 Fine-tuning 方法，如 DoRA、MoRA 和 LoRA，并链接了 [Answer.AI's efficient fine-tuning](https://www.answer.ai/posts/2024-04-26-fsdp-qdora-llama3.html) 等文章。还提到了对模型上下文扩展技术（如 RoPE）的探索。</li>
    <li><strong>Fine-tuning 守则</strong>：讨论了部署 Fine-tuned 模型的“十诫”，并附带了 [slides](https://docs.google.com/presentation/d/1IIRrTED0w716OsU_-PL5bONL0Pq_7E8alewvcJO1BCE/edit#slide=id.g2721fb6713e_0_67) 链接。成员们认为内容非常实用，对工作大有裨益。</li>
</ul>

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://argilla.io/">专家改进 AI 模型的平台</a>：Argilla 是一个面向 AI 工程师和领域专家的协作平台，致力于追求质量、所有权和效率。</li><li><a href="https://hamel.dev/notes/llm/inference/03_inference.html">Hamel 的博客 - 优化延迟</a>：探索优化延迟的方法。</li><li><a href="https://huggingface.co/spaces/mteb/leaderboard">MTEB 排行榜 - Hugging Face Space</a>：未找到描述</li><li><a href="https://www.answer.ai/posts/2024-04-26-fsdp-qdora-llama3.html">Answer.AI - 使用 FSDP QDoRA 高效 Fine-tuning Llama 3</a>：我们正在发布 FSDP QDoRA，这是一种可扩展且内存高效的方法，旨在缩小参数高效 Fine-tuning 与全量 Fine-tuning 之间的差距。</li><li><a href="https://x.com/corbtt">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://huggingface.co/nomic-ai/nomic-bert-2048">nomic-ai/nomic-bert-2048 · Hugging Face</a>：未找到描述</li><li><a href="https://docs.argilla.io/en/v1.1.0/guides/steps/1_labelling.html">🏷 标注</a>：在标注时，我们通常区分手动标注与协作或程序化标注。在协作标注期间，我们使用规则和推理预测等外部输入...</li><li><a href="https://docs.google.com/presentation/d/1IIRrTED0w716OsU_-PL5bONL0Pq_7E8alewvcJO1BCE/edit#slide=id.g2721fb6713e_0_67">在生产环境中部署 Fine-tuned 模型的十诫</a>：在生产环境中部署 Fine-tuned 模型的十诫，Kyle Corbitt | @corbtt</li><li><a href="https://openpipe.ai/">OpenPipe：面向开发者的 Fine-tuning</a>：将昂贵的 LLM Prompt 转换为快速、廉价的 Fine-tuned 模型。</li><li><a href="https://maven.com/parlance-labs/fine-tuning/1/recordings/88255">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-2](https://discord.com/channels/1238365980128706560/1242223415293644930/1243277523316637817)** (117 条消息🔥🔥): 

- **分享 Jarvis 仓库链接**：分享了 [nisargvp 在 Hugging Face 上的 Jarvis 仓库](https://huggingface.co/nisargvp/hc-mistral-alpaca) 链接，以及用于在 Axolotl 中设置模型的配置文件。
- **在 Modal 上运行模型的指南**：用户讨论了在 Modal 上顺利运行模型训练的问题，指出了 [Modal Labs 的快速入门指南](https://github.com/modal-labs/llm-finetuning)，并提到在初始修复后运行非常顺畅。
- **TinyLLama 微调博客文章**：社区分享并赞赏了一篇记录使用 Axolotl 和 Jarvis 在 alpaca_2k_test 数据集上微调 TinyLLama 过程的博客文章，链接在 [这里](https://lucasvw.github.io/posts/19_llm_fine_tuning/)。
- **LLM 应用中的可观测性**：讨论围绕在 LLM 应用中引入可观测性以收集用户反馈和 LLM 输入/输出对展开，强调了对更好追踪方法的需求。
- **Modal 训练错误支持**：用户在使用 Modal Labs 仓库进行 Mistral 模型训练时遇到并解决了问题，社区成员提供了排错建议并分享了具体的错误详情以诊断配置问题。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/inst_tune.html#how-to-add-custom-prompt-format">Axolotl - Instruction Tuning</a>：未找到描述</li><li><a href="https://lucasvw.github.io/posts/19_llm_fine_tuning/.">Lucas van Walstijn - LLM fine-tuning 101</a>：未找到描述</li><li><a href="https://wandb.ai/venetispall/llama-3-8b-hermes-sandals-sample-10k/workspace?nw=nwuservenetispall">venetispall</a>：Weights & Biases，机器学习开发者工具</li><li><a href="https://huggingface.co/blog/peft_merging">🤗 PEFT welcomes new merging methods</a>：未找到描述</li><li><a href="https://www.kaggle.com/competitions/lmsys-chatbot-arena">LMSYS - Chatbot Arena Human Preference Predictions | Kaggle</a>：未找到描述</li><li><a href="https://github.com/modal-labs/llm-finetuning/">GitHub - modal-labs/llm-finetuning: Guide for fine-tuning Llama/Mistral/CodeLlama models and more</a>：微调 Llama/Mistral/CodeLlama 等模型的指南 - modal-labs/llm-finetuning</li><li><a href="https://kaiokendev.github.io/til">Things I’m Learning While Training SuperHOT</a>：页面内容</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/">Axolotl - Dataset Formats</a>：未找到描述</li><li><a href="https://huggingface.co/nisargvp/hc-mistral-alpaca">nisargvp/hc-mistral-alpaca · Hugging Face</a>：未找到描述</li><li><a href="https://maven.com/parlance-labs/fine-tuning/1/syllabus/modules/ac50ed?item=bf4nff4j6bo">无标题</a>：未找到描述</li><li><a href="https://x.com/danielhanchen/status/1793815232177185061">Daniel Han (@danielhanchen) 的推文</a>：@TheZachMueller @Prince_Canuma @UnslothAI 如果你没有使用未训练的 token，那应该是没问题的 :) 只是有时人们使用 llama-3 模板 + llama-3 基础模型，结果会很糟糕...</li><li><a href="https://lawwu.github.io/posts/2024-05-23-first-axolotl-finetune/">Lawrence Wu - Finetuning LLMs with Axolotl</a>：未找到描述</li><li><a href="https://github.com/modal-labs/llm-finetuning/tree/main?tab=readme-ov-file#quickstart">GitHub - modal-labs/llm-finetuning: Guide for fine-tuning Llama/Mistral/CodeLlama models and more</a>：微调 Llama/Mistral/CodeLlama 等模型的指南 - modal-labs/llm-finetuning</li><li><a href="https://github.com/modal-labs/llm-finetuning/tree/main">GitHub - modal-labs/llm-finetuning: Guide for fine-tuning Llama/Mistral/CodeLlama models and more</a>：微调 Llama/Mistral/CodeLlama 等模型的指南 - modal-labs/llm-finetuning</li><li><a href="https://github.com/modal-labs/llm-finetuning/blob/main/data/modal_docs.jsonl">llm-finetuning/data/modal_docs.jsonl at main · modal-labs/llm-finetuning</a>：微调 Llama/Mistral/CodeLlama 等模型的指南 - modal-labs/llm-finetuning
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-3](https://discord.com/channels/1238365980128706560/1242223458184597534/1243339106675724369)** (3 条消息): 

- **Zoom 聊天混乱导致转向 Discord**：在 Zoom 聊天被禁用后，成员们不确定在哪里继续交流。一位成员建议将讨论转移到特定的 Discord 频道，得到了其他人的认可。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1243286083022618664)** (32 条消息🔥): 

- **Axolotl 中的缓存问题令用户困扰**：一位成员指出，在 **Axolotl** 中重新运行实验时，意外的缓存使用了旧的数据样本，相关记录见[此处](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/dataset_preprocessing.qmd)。重命名数据集文件解决了该问题，另一位用户建议显式运行预处理（pre-process）步骤。

- **对缺失文件的困惑**：用户在 **Jarvislabs** 和 Google Colab 上运行训练命令时遇到了 `simple.yml` 或 `qlora.yml` 文件缺失的问题，导致执行失败。一位成员分享说，他们的 qlora 运行在 2x4090s GPU 上耗时约 6 小时，证实了使用正确文件和配置的重要性。

- **关于 Sample Packing 的咨询**：一位成员询问 Axolotl 中的 Sample Packing 是否会连接多个数据集行以填充最大序列长度（max sequence length）。另一位成员确认了这一点，并解释说虽然它们被连接在一起，但 Attention 已被设置，使得各行之间不会互相注意（attend）。

- **Google Colab 中 BFloat16 的 RuntimeError**：一个与 T4 GPU 上未实现 `BFloat16` 相关的 RuntimeError 导致用户从 Google Colab 切换到 Jarvis-labs。他们被建议检查 PyTorch 和 CUDA 版本，切换到示例配置后解决了该问题。

- **分享 Tokenizer 注意事项指南**：一位用户分享了 Hamel 关于 [Tokenizer 注意事项的笔记](https://hamel.dev/notes/llm/finetuning/05_tokenizer_gotchas.html)链接，探讨了 Prompt 构建中的复杂细节，以及由于 Tokenization 处理导致的训练和推理之间的行为差异。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://hamel.dev/notes/llm/finetuning/05_tokenizer_gotchas.html">Hamel’s Blog - Tokenization Gotchas</a>：Tokenizer 和推理 LLM 时的陷阱</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/dataset_preprocessing.qmd">axolotl/docs/dataset_preprocessing.qmd at main · OpenAccess-AI-Collective/axolotl</a>：尽管提问。通过在 GitHub 上创建账户为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/tiny-llama/qlora.yml">axolotl/examples/tiny-llama/qlora.yml at main · OpenAccess-AI-Collective/axolotl</a>：尽管提问。通过在 GitHub 上创建账户为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://colab.research.google.com/drive/1jLQDiW47k1vPe_tet4-m6dLVZhnNRet9?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/colab-notebooks/colab-axolotl-example.ipynb">axolotl/examples/colab-notebooks/colab-axolotl-example.ipynb at main · OpenAccess-AI-Collective/axolotl</a>：尽管提问。通过在 GitHub 上创建账户为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://lawwu.github.io/posts/2024-05-23-first-axolotl-finetune/#runtimeerror-_amp_foreach_non_finite_check_and_unscale_cuda-not-implemented-for-bfloat16">Lawrence Wu - Finetuning LLMs with Axolotl</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1243291846415749283)** (118 条消息🔥🔥): 

- **用户对 float16 和 float32 的混淆**：有人提问为什么在显示的表格中 float16 的数值看起来比 float32 高。提供了一个关于该主题的过往讨论链接以澄清困惑。 
- **Jarvislab 的配置问题已解决**：用户在运行 Jarvislab 训练命令时遇到了缺少配置文件的问题。另一位用户建议将命令更改为使用 `accelerate launch -m axolotl.cli.train hc.yml`，问题得以解决。

- **在不同 GPU 上优化 Axolotl 运行**：一名成员请求关于针对不同 GPU 优化 `axolotl` 运行而调整 `accelerate` 配置的建议。建议将配置映射回 `axolotl` 的 yaml 文件，避免直接进行 `accelerate` 配置设置。

- **学习模型 Accelerate 的资源**：用户讨论了如何开始使用 Accelerate 进行微调任务，建议坚持使用像 `axolotl` 这样更高层级的抽象，以兼顾简单性和学习深度。

- **超参数和推理精度**：询问关于延长训练模型与训练不足模型的最佳学习率，以及 T4 GPU 中的 BF16 精度问题。建议包括在 Zoom QA 中询问硬件兼容的解决方案，或为支持的数据类型转换权重。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/yacineMTB/status/1783939078804701578">kache (@yacineMTB) 的推文</a>：三台配置顶级的 Mac Studio，每台 7.5k 加元，配备 192GB 统一内存，192 * 3 -> 576GB 的“显存”，有足够的 CPU 来驱动常规服务器任务。两台几乎可以...</li><li><a href="https://www.amazon.com/PNY-Generation-Express-DisplayPort-Support/dp/B0CJQH8519">未找到标题</a>：未找到描述</li><li><a href="https://www.philschmid.de/instruction-tune-llama-2">扩展指南：对 Llama 2 进行指令微调</a>：这篇博文是关于对 Meta AI 的 Llama 2 进行指令微调的扩展指南</li><li><a href="https://arxiv.org/abs/2311.03285">S-LoRA: Serving Thousands of Concurrent LoRA Adapters</a>：“预训练-然后-微调”范式在大型语言模型的部署中被广泛采用。低秩自适应（LoRA）是一种参数高效的微调方法，通常用于...</li><li><a href="https://huggingface.co/docs/transformers/en/chat_templating">聊天模型模板</a>：未找到描述</li><li><a href="https://x.com/DavidGFar/status/1793662035227770911">David Golchinfar (@DavidGFar) 的推文</a>：大家好！在 Kraken 正式发布后，@FernandoNetoAi、我、@LucasAtkins7 和 @erhartford 为大家准备了另一个惊喜。我们很高兴推出由 @Hyper 赞助的 Kraken-LoRA...</li><li><a href="https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/">2023 年深度学习最佳 GPU —— 深度分析</a>：在这里，我提供了用于深度学习/机器学习的 GPU 深度分析，并解释了适合您的使用场景和预算的最佳 GPU。</li><li><a href="https://huggingface.co/docs/accelerate/quicktour">快速入门</a>：未找到描述</li><li><a href="https://github.com/SkunkworksAI/hydra-moe">GitHub - SkunkworksAI/hydra-moe</a>：通过创建账号为 SkunkworksAI/hydra-moe 的开发做出贡献。</li><li><a href="https://x.com/skunkworks_ai">undefined 的推文</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1243305377974587412)** (192 条消息🔥🔥):

- **最新 axolotl 和 llama 3 演示的 PR 已合并**：Modal LLM 微调仓库现在包含了最新的 axolotl 更新和 llama 3 微调演示。
- **寻求数据集模板及预处理问题**：成员们询问了关于 `chatml.intel` 数据集模板的问题，并在预处理过程中遇到了困难，特别是由于数据集结构缺乏数值 ID 导致的解码问题。参考：[Axolotl Docs](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/rlhf.qmd)。
- **关于 Axolotl 配置的澄清**：讨论表明，如果没有明确指定，`load_in_8bit` 和 `load_in_4bit` 等默认配置值将设为 False，并建议直接检查代码以进行确认。
- **无模板提示词构建的困惑**：一位成员发现关于 [无模板提示词构建](https://openaccess-ai-collective.github.io/axolotl/docs/input_output.html) 的文档令人费解，而其他人则阐明了模板正确性的重要性。
- **Office Hours 问答亮点：调试与技术栈见解**：成员们表达了调试工具对于理解训练期间输入和样本的重要性，主张进行严格的模板验证，并建议使用回调函数来记录模型预测，参考 [Axolotl Callbacks](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/utils/callbacks/__init__.py)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/utils/callbacks/__init__.py">axolotl/src/axolotl/utils/callbacks/__init__.py at main · OpenAccess-AI-Collective/axolotl</a>：尽管提问。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/h2oai/h2o-llmstudio">GitHub - h2oai/h2o-llmstudio: H2O LLM Studio - 一个用于微调 LLM 的框架和无代码 GUI。文档：https://h2oai.github.io/h2o-llmstudio/</a>：H2O LLM Studio - 一个用于微调 LLM 的框架和无代码 GUI。文档：https://h2oai.github.io/h2o-llmstudio/ - h2oai/h2o-llmstudio</li><li><a href="https://www.philschmid.de/instruction-tune-llama-2">扩展指南：对 Llama 2 进行指令微调</a>：这篇博文是关于对 Meta AI 的 Llama 2 进行指令微调的扩展指南</li><li><a href="https://huggingface.co/datasets/GAIR/lima?row=85">GAIR/lima · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://gist.github.com/strickvl/e1591b83e3b290fb176e780e7ce7d383">gist:e1591b83e3b290fb176e780e7ce7d383</a>：GitHub Gist：立即分享代码、笔记和代码片段。</li><li><a href="https://docs.google.com/document/d/1944izw_gwWq9EuaZcNN5lQwiVOKaIYXbYIKi6e9Efsw/edit?usp=sharing">Wing 的问题线程 - Office Hours</a>：Wing (Axolotl) 的 OH 问题。Ben Eyal 9:59 AM：我想知道关于无模板提示词构建（Template-free prompt construction）的事情，我真的不明白它是如何工作的。配置只需要一个输出，然后...</li><li><a href="https://docs.google.com/document/d/1944izw_gwWq9EuaZcNN5lQwiVOKaIYXbYIKi6e9Efsw/edit?pli=1">Wing 的问题线程 - Office Hours</a>：Wing (Axolotl) 的 OH 问题。Ben Eyal 9:59 AM：我想知道关于无模板提示词构建（Template-free prompt construction）的事情，我真的不明白它是如何工作的。配置只需要一个输出，然后...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/rlhf.qmd">axolotl/docs/rlhf.qmd at main · OpenAccess-AI-Collective/axolotl</a>：尽管提问。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/grgalex/nvshare">GitHub - grgalex/nvshare: 无显存限制的实用 GPU 共享</a>：无显存限制的实用 GPU 共享 - grgalex/nvshare</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/utils/data/sft.py#L129C4-L152C6.">axolotl/src/axolotl/utils/data/sft.py at main · OpenAccess-AI-Collective/axolotl</a>：尽管提问。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://support.zoom.com/hc/en/article?id=zm_kb&sysparm_article=KB0066245">未找到标题</a>：未找到描述</li><li><a href="https://support.zoom.com/hc/en/article?id=zm_kb&sysparm_article=KB0060623">未找到标题</a>：未找到描述</li><li><a href="https://github.com/h2oai/">H2O.ai</a>：为更智能的应用提供快速可扩展的机器学习 - H2O.ai</li><li><a href="https://tenor.com/view/trust-no-one-crazy-chris-henry-thomas-just-beyond-dont-trust-anybody-gif-23566469">Trust No One Crazy Chris GIF - Trust No One Crazy Chris Henry Thomas - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1243310332407971850)** (1 messages): 

- **使用 Proteinviz 可视化蛋白质**：查看 [Proteinviz](https://huggingface.co/spaces/as-cle-bert/proteinviz) 以创建自定义的蛋白质视觉效果。该工具由一位热心的社区成员制作。
  
- **快速的 SDXL 结果**：[SDXL flash](https://huggingface.co/spaces/KingNish/SDXL-Flash) Space 能够快速交付令人印象深刻的结果。感谢创作者完成了这一高效的构建。

- **受 Karpathy 启发的自定义 Tokenizer**：一位社区成员分享了他们的 [自定义 Tokenizer](https://github.com/apehex/tokun)，其灵感来自 Karpathy 的工作。这突显了社区内不断的创新。

- **Mistral-7B v0.3 演示**：通过 [Mistral-7B v0.3 chat](https://huggingface.co/spaces/ehristoforu/mistral-7b-v0.3-chat) 演示体验极速性能。这是活跃贡献者带来的又一前沿开发案例。

- **使用 Diffusers 创建透明图像**：使用 [Diffusers](https://github.com/rootonchair/diffuser_layerdiffuse) 生成透明图像，这是一个由另一位社区成员推动的项目。该功能允许使用先进的 diffusing 技术进行创意视觉输出。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=S-gy6NUOGSs)">Agentic AI Solutions / Adaptive AI Solutions - Episode 1:  CrewAI With Preston McCauley</a>: 在第 1 集中，我们简要介绍了 #AdaptiveAI 和 #Agentic AI 方法。https://www.linkedin.com/in/preston-mccauley-immersive-ux/Join Presto...</li><li><a href="https://youtu.be/jddSbTLw0gc)">What is an Instruction Tuned Model?</a>: 什么是 Instruction Tuning？什么是 Instruction Tuned 模型？什么是 Pretrained Model？如何让我的 Large Language Model 遵循指令？这些...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1243283012477522071)** (490 messages🔥🔥🔥): 

- **AutoTrain 数据格式问题**：成员们讨论了如何在 **AutoTrain** 中格式化微调数据，并建议参考 [AutoTrain 文档](https://hf.co/docs/autotrain)。分享了 CSV 格式示例和输入数据类型的细微差别，提高了设置的清晰度。
- **高级 LLM 微调**：强调了用于微调 LLM 的 **DPO 和 RHLF 方法之间的区别**，建议在 **SFT 之后进行 RHLF**，以教授文本补全模型对话规范。还分享了特定数据集和更精细模型调整的链接。
- **Pandora 模型引发关注**：分享了关于 Pandora 模型的细节，这是一款新的 **开源 text-to-video** 模型，并附带了预览链接。关于其**智能程度**和潜在应用的讨论引发了成员们的极大兴趣。
- **Mobius 模型争议**：即将推出的 **Mobius 扩散模型** 面临审查，评论涉及受控质量和构图训练。随后的讨论强调了它在显著降低开发新扩散模型的成本和复杂性方面的潜力。
- **学习与发展资源**：包括 @temeretam 在内的几位成员讨论了在 AI 领域晋升的教育和职业路径，而其他人则寻求有关特定编码和数据处理问题的建议，并参考了 **GitHub** 和 **Hugging Face** 文档链接以获取技术支持。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/maitrix-org/Pandora">maitrix-org/Pandora · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/docs/huggingface_hub/guides/download#filter-files-to-download">从 Hub 下载文件</a>：未找到描述</li><li><a href="https://tenor.com/view/babuin-gif-27648024">Babuin GIF - Babuin - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/que-gif-27530657">Que GIF - Que - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://imgsys.org/rankings">imgsys.org | 由 fal.ai 提供的图像模型竞技场</a>：一个生成式 AI 竞技场，你可以在这里测试不同的 prompt 并选择你最喜欢的结果。查看模型排名并亲自尝试！</li><li><a href="https://huggingface.co/docs/transformers/main/chat_templating">Chat Models 模板</a>：未找到描述</li><li><a href="https://x.com/DataPlusEngine/status/1793817514956259460">来自 DataVoid e/acc (@DataPlusEngine) 的推文</a>：我们即将发表的论文概述并实现了在无需从头开始进行大规模预训练的情况下，创建全新的基础 Diffusion 模型。我们可以通过受控的方式，打破所有的质量...</li><li><a href="https://huggingface.co/datasets/nroggendorff/mayo">nroggendorff/mayo · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://youtu.be/zLvFc_24vSM?si=TeS_EkFu9BeyYDbz">Rabbit 欺骗了我，所以我深入挖掘</a>：LAM 是骗局吗？让我们深入探究。支持调查新闻：► Patreon: https://patreon.com/coffeezilla 协助此次调查的人员...</li><li><a href="https://x.com/DataPlusEngine/status/1793803117642854732">来自 DataVoid e/acc (@DataPlusEngine) 的推文</a>：我们让 @FAL 提前体验了即将推出的 Mobius 模型，它在 http://imgsys.org 上线仅 3 小时。根据人类偏好，它已经是世界上最好的基于 Stable Diffusion 的图像模型...</li><li><a href="https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/tree/main">mistralai/Mixtral-8x7B-v0.1 (main 分支)</a>：未找到描述</li><li><a href="https://tenor.com/view/frank-castle-wait-please-stop-please-no-please-gif-21133188">Frank Castle Wait GIF - Frank Castle Wait Please Stop - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.instagram.com/p/C7U3hOPRJhR/">Noa Roggendorff 在 Instagram 上的发布："epic #ai"</a>：2 个赞，1 条评论 - noaroggendorff 于 2024 年 5 月 23 日："epic #ai"。</li><li><a href="https://huggingface.co/docs/datasets/v2.19.0/en/process#rename>">处理 (Process)</a>：未找到描述</li><li><a href="https://tenor.com/view/kurt-kurt-angle-100-yard-stare-what-are-you-serious-gif-4081464694509837388">Kurt Kurt Angle GIF - Kurt Kurt angle 100 yard stare - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://emoji.gg/category/6/blobs">适用于 Discord 和 Slack 的 Blobs 表情符号 - Discord Emoji</a>：查找可在 Discord 或 Slack 上使用的 Blobs 表情符号 - Emoji.gg，互联网上最大的免费自定义表情符号目录。</li><li><a href="https://hf.co/docs/autotrain">什么是 AutoTrain Advanced？</a>：未找到描述</li><li><a href="https://github.com/hpcaitech/Open-Sora?tab=readme-ov-file#installation">GitHub - hpcaitech/Open-Sora: Open-Sora: 为所有人实现高效视频制作的民主化</a>：Open-Sora: 为所有人实现高效视频制作的民主化 - hpcaitech/Open-Sora</li><li><a href="https://github.com/PKU-YuanGroup/Open-Sora-Plan?tab=readme-ov-file">GitHub - PKU-YuanGroup/Open-Sora-Plan: 该项目旨在复现 Sora (OpenAI T2V 模型)，我们希望开源社区能为该项目做出贡献。</a>：该项目旨在复现 Sora (OpenAI T2V 模型)，我们希望开源社区能为该项目做出贡献。 - PKU-YuanGroup/Open-Sora-Plan</li><li><a href="https://slackmojis.com/categories/25-blob-cats-emojis">
Slack 上的 Blob Cats 表情符号
</a>：未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1243424993426014309)** (8 messages🔥): 

- **针对 Embodied AI 的 Deep RL 引起关注**：一名成员分享了他们学习专门用于 Embodied AI 应用的 Deep RL 的热情，并邀请大家详细更新进展。

- **为 AI 初学者推荐 Fast.ai 课程**：推荐了 Fast.ai 的第 1 部分和第 2 部分课程，这些课程涵盖了使用 HuggingFace 库的实用深度学习任务，为深度学习初学者提供了坚实的基础。课程详情可以在[这里](https://course.fast.ai/)找到。

- **Coursera 上的 Generative AI with LLMs 课程**：为那些有兴趣获得 AI 基础知识的人推荐了 Coursera 上的 **Generative AI with Large Language Models** 课程。该课程设计为 3 周完成，详情见[这里](https://www.coursera.org/learn/generative-ai-with-llms)。

- **PixART Diffusion Model 交流活动**：宣布了一项针对用于文本到图像合成的 PixART Diffusion 模型进行深入审查的交流活动，定于太平洋时间周五上午 10:00 举行。更多信息和社区互动可以在[这里](https://lu.ma/arxivdive-2024-05-24)找到。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://course.fast.ai/">Practical Deep Learning for Coders - Practical Deep Learning</a>：一门为具有一定编程经验、想要学习如何将深度学习和机器学习应用于实际问题的人设计的免费课程。</li><li><a href="https://www.coursera.org/learn/generative-ai-with-llms">Generative AI with Large Language Models</a>：在 Generative AI with Large Language Models (LLMs) 课程中，你将学习 Generative AI 的工作原理基础，以及如何部署它... 免费注册。</li><li><a href="https://lu.ma/arxivdive-2024-05-24?tk=F1jNfh">Arxiv Dives with Oxen.AI - Fine Tuning Diffusion Transformers (DiT) · Zoom · Luma</a>：嘿，极客们，加入我们吧！... 进行一次小型的书籍/论文回顾。期待内容：每周我们都会选择一个主题进行深入探讨，并进行公开问答和讨论。…
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1243438483884871721)** (3 messages): 

- **ChatGPT 在药物研发中的精彩应用**：分享了一项研究链接，讨论了 **ChatGPT 和其他 LLMs 在下一代药物研发中**的潜在用途。这篇发表在《国际手术杂志》（International Journal of Surgery）上的文章重点介绍了来自印度和孟加拉国各机构的贡献 [阅读更多](https://journals.lww.com/international-journal-of-surgery/fulltext/2023/12000/chatgpt_or_llm_in_next_generation_drug_discovery.78.aspx)。
  
- **PostgresML 和 LlamaIndex 引起轰动**：最近的一篇 Medium 文章强调了 **PostgresML** 与 **LlamaIndex** 的集成。这种[集成](https://medium.com/ai-advances/unleashing-the-power-of-postgresml-with-llamaindex-integration-9eadee223939)有望释放 AI 进步的新潜力，文章中提供了详细的见解。

**提到的链接**：<a href="https://journals.lww.com/international-journal-of-surgery/fulltext/2023/12000/chatgpt_or_llm_in_next_generation_drug_discovery.78.aspx">ChatGPT or LLM in next-generation drug discovery and... : International Journal of Surgery</a>：暂无摘要。

  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1243293254309511218)** (22 messages🔥): 

- **蛋白质数据集迎来重大更新**：一位成员分享了其蛋白质可视化项目的更新，增加了 **human hemoglobin**（人类血红蛋白）、**mouse GTPase**（小鼠 GTPase）和 **human ribosomal protein**（人类核糖体蛋白）的示例。他们还实现了对 **3D rendering** 的支持，并在 [GitHub](https://github.com/AstraBert/proteinviz/blob/main/examples.md) 上创建了详细的示例表。

- **基于 OpenAI Whisper 的转录应用表现出色！**：一位成员介绍了一款针对 YouTube 视频、音频文件和视频文件的转录应用，该应用利用了 **OpenAI Whisper**。可以在 [Hugging Face Spaces](https://huggingface.co/spaces/tensorkelechi/vidtext) 上查看。

- **征集去中心化互联网基础设施的反馈**：一位成员为其构建去中心化且以 **Agent** 为中心的互联网基础设施项目征集反馈并邀请参与调查：[调查链接](https://hai.ai/)。这引发了关于频道垃圾信息以及通过调查收集数据的伦理问题的辩论。

- **浏览器中 3D 模型可视化的挑战**：尽管在 **Gradio** 浏览器中对蛋白质结构进行 **3D model rendering** 面临挑战，但目前仍在努力寻找解决方案。有用的资源包括 [Hugging Face](https://huggingface.co/blog/spaces_3dmoljs) 上的一篇博客文章。

- **SimpleTuner Bug 修复提升训练性能**：一位成员强调，修复 **SimpleTuner** 中的一些细微 Bug 显著增强了其训练性能。现在的训练效果比以往任何时候都好。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/tensorkelechi/vidtext">Vidtext - tensorkelechi 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/blog/spaces_3dmoljs">在 Hugging Face Spaces 上可视化蛋白质</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1243525495769792583)** (4 messages): 

- **宣布每月计算机视觉聚会**：介绍了一个即将举行的每月 **Computer Vision Hangout**，旨在讨论 CV 相关领域的项目、想法和问题。更多详情和活动参与方式请点击[此处](https://discord.gg/MkHyuG9C?event=1243129304863215656)。

- **寻求发票处理解决方案**：一位成员询问是否有开源神经网络或付费 API，用于从扫描的发票中提取结构化的逐行信息。他们要求输出格式为 **JSON**，并指定了 **product_id**、**description**、**quantity**、**unit_price** 和 **total_price** 等字段。

- **寻找深度学习学习伙伴**：一位用户表示有兴趣寻找一位同样热爱 AI 和数据科学的深度学习学习伙伴。他们强调了共同探索神经网络、复杂算法和创新项目的动力。

- **征集 ViT 在深度估计方面的资源**：另一位成员询问了关于利用 **Vision Transformers (ViT)** 进行单目深度估计（monocular depth estimation）的资源。他们表示有兴趣使用 **ViT** 构建自己的模型，并正在寻求指导。

**提到的链接**：<a href="https://discord.gg/MkHyuG9C?event=1243129304863215656">加入 Hugging Face Discord 服务器！</a>：我们正致力于让优秀的机器学习民主化 🤗 验证以关联您的 Hub 和 Discord 账号！| 79727 名成员

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1243507818158489661)** (8 messages🔥): 

- **Mistral v0.3 Instruct 中的量化异常**：一位成员报告了在使用 **bitsandbytes** 的 **8-bit**、**4-bit** 和 **fp16** 量化级别对比 **Mistral v0.3 Instruct** 时出现的意外性能问题。他们发现，虽然 **fp16** 和 **4-bit** 耗时约 100 秒，但 **8-bit** 却耗时 500 秒，尽管预期 **8-bit** 应该比 **4-bit** 更快。
- **从 Pipelines 切换到 Generate 无明显效果**：同一位用户指出，根据 **8-bit** 模型文本生成的文档，将 **pipelines** 切换为 **generate()** 方法并没有像预期那样提高性能。
- **Bitsandbytes 版本与优化技巧**：针对性能问题，另一位成员询问了正在使用的 **bitsandbytes 版本**，并建议尝试设置 **int8_threshold=0** 以获得潜在的性能提升。原用户提到他们使用的 **batch size** 为 1，上下文范围在 500 到 2000 个 **tokens** 之间。
  

---

### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1243278218262351995)** (6 条消息): 

- **寻求 NLG 学习资源**：一名成员询问有关学习 **Natural Language Generation (NLG)** 的推荐资源。消息历史中未提供针对该查询的回复。

- **关于在自定义数据集上训练 Stable Diffusion 的咨询**：另一名成员询问有关训练 Stable Diffusion (SD) 以从自定义数据集（如 MNIST）生成图像的**官方文档**。他们提到在网站上找到了文档，但似乎侧重于无条件生成（unconditional generation）。

- **寻找深度学习学习伙伴**：另一位成员表示有兴趣寻找伙伴一起**学习深度学习**。他们强调希望找一个同样对 AI 和数据科学充满热情，并热衷于探索神经网络、复杂算法和创新项目的人。

- **寻求将 pth+index 文件转换为 Hugging Face 链接的帮助**：一名成员请求协助将 **pth+index 文件**转换为 Hugging Face 链接的 RVC 模型。这一技术咨询未立即获得可见的回复。

  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1243286415463284879)** (493 条消息🔥🔥🔥): 

- **Perplexity 与 ChatGPT 在数据处理方面的对比**：讨论了 **Perplexity** 和 **ChatGPT** 在处理 CSV 文件方面的能力，并提到 **Perplexity** 已经支持 CSV 上传。Julius AI 作为数据分析的另一种选择被提及，它运行在 Python 上并利用 Claude 3 或 GPT-4 等 LLM。
  
- **对 Claude 3 Opus 的失望**：用户对 **Claude 3 Opus** 表示不满，原因是限制增加且实用性降低，特别是在处理受版权保护的材料时。一些人建议使用 GPT-4o 等替代方案，但承认 Claude 3 的效用已经减弱。

- **Pro Search 功能与增强**：用户注意到了 **Pro Search** 的新功能，增强功能包括多步推理（multi-step reasoning）和更新的 API 规范获取。然而，一些用户观察到此类更新可能是 A/B testing 的一部分，仅涉及 UI 更改而非后端改进。

- **工具集成与自定义函数调用**：讨论了 **Claude** 通过 API 进行外部工具集成的能力，并尝试通过自定义函数调用（function calls）和 serverless 后端解决方案来复制 ChatGPT 的数据分析工具。分享了相关文档链接，如 [Tool Use with Claude](https://docs.anthropic.com/en/docs/tool-use)。

- **AI 伦理与沟通分析项目**：讨论包括创建用于沟通分析和伦理行为监控的 GPTs，并建议此类工具可以帮助改善职场沟通并减少不当解雇诉讼。用户辩论了将伦理编码进算法的可行性和哲学含义。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=0T1444HJbt0">Mistral's new 7B Model with Native Function Calling</a>：Colab 代码 - https://drp.li/K98Z7🕵️ 对构建 LLM Agents 感兴趣？填写下方表格：https://drp.li/dIMes👨‍💻Github:http...</li><li><a href="https://v0.dev/">v0 by Vercel</a>：通过简单的文本提示生成 UI。复制、粘贴、发布。</li><li><a href="https://tenor.com/view/google-chrome-pacman-eating-gif-13756279">Google Chrome Pacman GIF - Google Chrome Pacman Eating - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/aladdin-disney-cartoons-jasmine-ic-an-show-you-the-world-gif-4545341">Aladdin Disney GIF - Aladdin Disney Cartoons - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.reddit.com/r/AZURE/comments/1bzs8gr/have_you_purchased_openai_ptus_how_much_did_it/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://docs.anthropic.com/en/docs/tool-use">Tool use (function calling) - Anthropic</a>：未找到描述</li><li><a href="https://aws.amazon.com/bedrock">Build Generative AI Applications with Foundation Models - Amazon Bedrock - AWS</a>：未找到描述</li><li><a href="https://search.brave.com/search?q=%s&source=desktop">Brave Search</a>：私密搜索网络……</li><li><a href="https://aws.amazon.com/bedrock/pricing/">Build Generative AI Applications with Foundation Models - Amazon Bedrock Pricing - AWS</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1243340241197338706)** (7 条消息): 

- **Peran Kepala Sekolah 分享**: 分享了一个指向 [Peran Kepala Sekolah](https://www.perplexity.ai/search/Peran-Kepala-Sekolah-ECYSEyQXTviCYDqqDgM8sw) 的简短链接，没有额外的上下文或讨论。
- **PB55 是什么解释**: 提供了一个指向 [what is the PB55](https://www.perplexity.ai/search/what-is-the-PB55hhXYRDGAVd7JjWDhaA) 的链接以供进一步阅读。
- **探索 'makura' 的起源**: 一位用户分享了一个链接，用于探索日语单词“枕（まくら / makura）”的词源，链接见[此处](https://www.perplexity.ai/search/oDyPhU47T26IM1W0f7GQIg)，其意为枕头。
- **确保 Thread 可共享性**: 发出了一个带有附件的提醒，以确保 Thread 是可共享的，并附带了指向 [Discord thread](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825) 的链接。
- **讨论 Stuart Hall 的理论**: 分享了 [Stuart Hall 的编码/解码模型](https://www.perplexity.ai/search/Explain-Stuart-Halls-IV.my4LjS2mNXyxPyWVLOw#0)。
- **查询 Opus 50 限制**: 一位用户询问了关于 [Opus 50 限制](https://www.perplexity.ai/search/Opus-50-limit-c2EHUbzTQGCocG2d17MrLg) 的问题。
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1243398440315260958)** (1 条消息): 

- **References 功能仍处于 Beta 停滞状态**: 一位用户质疑了 References 功能处于 Beta 阶段的状态，并对申请三次后未收到回复表示沮丧。他们询问是否有人知道该功能何时会在 API 中发布。

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1243290070820192337)** (427 条消息🔥🔥🔥): 

- **RTX 5090 规格传闻引发辩论**：讨论集中在 RTX 5090 可能配备 32GB VRAM 的新传闻上，引发了对其可行性和实用性的质疑。一位成员分享了指向所谓图片的[链接](https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/News/Geforce-RTX-5090-mit-32-GiB-und-drei-PCBs-1448073/galerie/3884555/)，但其他人批评这些图片具有误导性。

- **Stable Diffusion 安装指导**：一位成员寻求在 AMD 5700XT GPU 上安装 Stable Diffusion 的建议。由于 AMD 硬件可能存在兼容性问题，建议最初尝试使用 [Craiyon](https://www.craiyon.com/) 等 Web 服务。

- **Stable Diffusion 3 的定价与访问**：用户讨论了 Stable Diffusion 3 与 Midjourney 的优劣，一些人指出 SD3 提供免费试用。然而，似乎需要 Stability 会员资格才能持续访问。

- **Mobius 模型的推出引发关注**：DataPlusEngine 在 [Twitter](https://x.com/DataPlusEngine/status/1793803117642854732) 上宣布了即将推出的 Mobius 模型，声称它是目前最好的基于 Stable Diffusion 的图像模型。该模型被描述为“既不是基础模型也不是微调模型”，并因其高效创建新基础模型的能力而受到赞誉。

- **对 GPU 性能和成本的好奇**：新的 GPU 型号，特别是 5090，引发了关于显存和训练速度的讨论。成员们指出，像 32GB 这样更高的 VRAM 可能会削弱 H100/A100 等高端数据中心 GPU 的销量，暗示这可能会影响 Nvidia 的策略。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/DataPlusEngine/status/1793803117642854732">来自 DataVoid e/acc (@DataPlusEngine) 的推文</a>: 我们向 @FAL 提供了即将推出的 Mobius 模型的早期访问权限，它在 http://imgsys.org 上线仅 3 小时。根据人类偏好，它已经是世界上最好的基于 Stable Diffusion 的图像模型...</li><li><a href="https://tenor.com/view/never-finn-adventure-time-gif-10874543">Never Finn GIF - Never Finn Adventure Time - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://youtu.be/-CyupSdXfI0">WOW Owen Wilson 说过的每一个 Wow，简直太 WOW 了</a>: Owen Wilson 是我最喜欢的演员之一，他的 "wow" 是传奇性的 - 所以这里收集了他在一个地方的所有 "wow"</li><li><a href="https://www.youtube.com/watch?v=k1hbRvSnFZg">A Moebius-metró | 匈牙利语全片</a>: 阿根廷神秘/科幻/惊悚片，1996 - 全片。在世界上最繁忙的地铁系统之一中，一列满载乘客的地铁列车消失得无影无踪...</li><li><a href="https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/News/Geforce-RTX-5090-mit-32-GiB-und-drei-PCBs-1448073/galerie/3884555/">据传 Geforce RTX 5090 将配备 32 GiB GDDR7 和三个 PCB [传闻]</a>: 文章图片：据传 Geforce RTX 5090 将配备 32 GiB GDDR7 和三个 PCB [传闻] - Geforce RTX 5090</li><li><a href="https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/News/Geforce-RTX-5090-mit-32-GiB-und-d">显卡新闻</a>: 您可以在这里找到关于显卡的最佳新闻
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1243280018524606544)** (275 条消息🔥🔥): 

- **PEFT 训练问题已解决**：一位用户遇到了在 PEFT 训练期间未创建 `config.json` 的问题，并被建议从基础模型的配置中复制。该用户确认此方法有效，并感谢社区的帮助。

- **注意到 Llama 3 的 Bug**：一些用户讨论了“Llama 3 的某些基础（非 instruct）权重存在‘bug’”，但 Unsloth 会自动修复这些问题。建议在训练期间使用保留的 token，并确保训练了 tokenizer 和 `lm_head`。

- **System Prompt 提升 Llama3 效果**：用户提到添加 system prompt 可以提高 Llama3 的 finetuning 性能。一位用户确认，即使是空白的 system prompt 也能对结果产生积极影响。

- **宣布支持 Phi 3 模型**：官方宣布现在支持 Phi 3 模型（包括 medium 版本支持）。社区表现出极大的兴奋，并分享了相关博客文章的链接以获取更多细节。

- **Stable Diffusion 的诡异印记**：用户分享了关于语音克隆和 Stable Diffusion 生成的诡异伪影的可怕经历。他们发布了相关 [YouTube 视频](https://youtube.com/shorts/o4kVe2NwRYY?si=ILtLzWy1XTAPALKc) 和 [Reddit 讨论](https://www.reddit.com/r/StableDiffusion/comments/1b10o36/creepy_imprint_from_stable_difussion/) 的链接。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://unsloth.ai/blog/phi3">使用 Unsloth Finetune Phi-3</a>：通过 Unsloth 轻松对微软的新模型 Phi 3 medium、small 和 mini 进行 fine-tune，上下文长度增加 6 倍！</li><li><a href="https://huggingface.co/CohereForAI/aya-23-8B">CohereForAI/aya-23-8B · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/UnslothAI/status/1793758720541016567">来自 Unsloth AI (@UnslothAI) 的推文</a>：我们已经解决了训练 Llama 3 的问题，所以现在的 finetuning 效果好得多！Unsloth 现在支持新的 Phi-3 模型、Mistral v3、Qwen 等！阅读我们的博客：http://unsloth.ai/blog/phi3</li><li><a href="https://youtube.com/shorts/o4kVe2NwRYY?si=ILtLzWy1XTAPALKc">can i get a chicken tendie combo please</a>：未找到描述</li><li><a href="https://github.com/babycommando/machinascript-for-robots">GitHub - babycommando/machinascript-for-robots: 使用 MachinaScript For Robots 在你的车库里构建由 LLM 驱动的机器人！</a>：使用 MachinaScript For Robots 在你的车库里构建由 LLM 驱动的机器人！ - babycommando/machinascript-for-robots</li><li><a href="https://github.com/ggerganov/llama.cpp/issues">Issues · ggerganov/llama.cpp</a>：C/C++ 中的 LLM 推理。通过在 GitHub 上创建账户为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1b10o36/creepy_imprint_from_stable_diffusion/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062">带有合并 LORA 适配器的 Llama3 GGUF 转换似乎会随机丢失训练数据 · Issue #7062 · ggerganov/llama.cpp</a>：我正在运行 Unsloth 在 llama3-8b 上对 Instruct 模型进行 LORA fine-tune。1：我将模型与 LORA 适配器合并为 safetensors；2：在 Python 中直接使用合并后的模型运行推理...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1243306406195626034)** (1 条消息): 

- **Phi-3 和 Mistral v3 现已上线**：*Unsloth 现在支持 Phi-3、Mistral v3 以及许多其他新模型。* 查看 [发布详情](https://github.com/unslothai/unsloth/releases/tag/May-2024)。

- **Llama 3 问题已解决**：*我们修复了所有 Llama 3 的问题，现在微调效果更好了。* 欲了解更多深度信息，请参考此 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1cwwgkz/is_llama_3_just_not_a_good_model_for_finetuning/)。

- **探索免费的 Colab Notebooks**：访问我们的 [Phi-3 medium notebook](https://colab.research.google.com/drive/1hhdhBa1j_hsymiW9m-WzxQtgqTH_NHqi?usp=sharing)、[Mistral v3 notebook](https://colab.research.google.com/drive/1_yNCks4BTD5zOnjozppphh5GzMFaMKq_?usp=sharing) 等。

- **新模型支持和 GitHub Accelerator**：在 [Hugging Face](https://huggingface.co/unsloth) 上查看我们最新添加的模型，并了解我们参与 [GitHub 2024 Accelerator](https://github.blog/2024-05-23-2024-github-accelerator-meet-the-11-projects-shaping-open-source-ai/) 的情况。

- **庆祝 AI 创新**：*我们很高兴能与另外 10 个项目一起加入 GitHub 的 2024 Accelerator，这彰显了 AI 创新的全球影响力和快速进步。*
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/phi3">使用 Unsloth 微调 Phi-3</a>: 通过 Unsloth 轻松微调 Microsoft 的新模型 Phi 3 medium、small 和 mini，上下文长度可增加 6 倍！</li><li><a href="https://colab.research.google.com/drive/1hhdhBa1j_hsymiW9m-WzxQtgqTH_NHqi?usp=sharing)">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1_yNCks4BTD5zOnjozppphh5GzMFaMKq_?usp=sharing)">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/11t4njE3c4Lxl-07OD8lJSMKkfyJml3Tn?usp=sharing)">Google Colab</a>: 未找到描述</li><li><a href="https://github.blog/2024-05-23-2024-github-accelerator-meet-the-11-projects-shaping-open-source-ai/)">2024 GitHub Accelerator: 认识塑造开源 AI 的 11 个项目</a>: 宣布第二批入选名单，为项目提供价值，并驱动新的前沿。
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1243440379672530985)** (4 条消息): 

- **寻求本地 VSCode Copilot 推荐**：一位用户问道：“有人使用本地的 VSCode ‘Copilot’ 吗？我想尝试一下。求推荐 :)”。另一位用户回答道：“试试 Continue”，随后最初的用户表示感谢：“谢谢，我会试试的 :)”。
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1243277920252727329)** (103 条消息🔥🔥): 

- **Sloth Phi-3 推理性能问题**：一位用户报告称，与原始模型相比，使用 **Unsloth Phi-3 模型**时的*推理速度较慢*。他们分享了一个 [Colab notebook](https://colab.research.google.com/drive/1LLWoaQrH8KFkQlE4ONwwtC4tC1-1It2X) 来诊断该问题，但即使在尝试了建议的修改后，问题依然存在。
  
- **自定义模型量化问题**：一位成员在量化基于 Unsloth notebook 衍生的自定义模型时遇到问题。他们在配合 **llama.cpp** 和 **Docker** 使用时收到了与不支持的架构相关的错误。

- **不同模型的资源需求**：关于 VRAM 需求的查询表明，**12GB 足以运行 Phi 3 mini**，而 **Phi 3 medium 则需要 16GB**。此外还提到，对于具有更大上下文窗口的摘要等大型任务，可能需要**租赁计算资源**。

- **评估数据集标准**：讨论强调了在训练和评估中使用一致数据集的重要性。具体而言，推荐使用 Hugging Face 上 unslothai 的公开数据集（例如 [Blackhole Collection](https://huggingface.co/collections/lamhieu/blackhole-66473b7feec034b4fb70818a) 中列出的数据集），因为其质量较高。

- **兼容性与自定义模型支持**：几位用户询问了 **Unsloth** 与旧款 Mac 的兼容性以及在无 GPU 系统上的使用情况，确认了 Unsloth 是针对 CUDA 和 GPU 使用进行优化的。针对仅 CPU 系统和自定义模型支持，提供了一些变通方法和建议。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/collections/lamhieu/blackhole-66473b7feec034b4fb70818a">Blackhole - lamhieu 收藏集</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1LLWoaQrH8KFkQlE4ONwwtC4tC1-1It2X#scrollTo=0zM8gPJUGySh">Google Colab</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues">Issues · unslothai/unsloth</a>：以 2-5 倍的速度和减少 80% 的显存微调 Llama 3, Mistral &amp; Gemma LLM - Issues · unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/1LLWoaQrH8KFkQlE4ONwwtC4tC1-1It2X?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/models/_utils.py#L179">unsloth/unsloth/models/_utils.py at main · unslothai/unsloth</a>：以 2-5 倍的速度和减少 80% 的显存微调 Llama 3, Mistral &amp; Gemma LLM - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1243412700105674753)** (2 条消息): 

- **工程师向 Unsloth 提供企业级经验**：成员 higginsconsultingptyltd_39617 祝贺其他成员加入 Build Club 和 GitHub 的加速器计划，并提议利用其企业级经验来协助 Unsloth。另一位成员给出了积极回应，表示渴望进一步讨论：“*当然，我们非常乐意！*”
  

---

### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1243288751313256490)** (12 条消息🔥): 

- **通俗易懂大师谈论 PixART Diffusion 模型**：感兴趣的成员可以在太平洋时间今天上午 10:00 的通话中，“听通俗易懂大师描述他是如何微调 PixART diffusion 模型的”。加入活动及 [Discord 链接](https://discord.gg/s3tBEn7Ptg) 进行进一步讨论，或在他们的 [博客](https://www.oxen.ai/blog) 和 [YouTube 视频](https://www.youtube.com/@oxen-ai/videos) 中查看过去的话题。
  
- **对 Intel 库感到兴奋**：一位成员在讨论 IPEX 和 BigDL 分离时，表达了对“折腾 Intel 库”的兴奋。提到了潜在的合作以及对 Intel 改进的探索。

- **IPEX-LLM 的稳定功能**：虽然一位成员还没用过 **IPEX-LLM**，但他们发现它在已有的支持中具有“坚如磐石般稳定”的表现。讨论内容包括 IPEX-LLM 安装设置方面的改进。

- **Tinygrad OpenCL 设置见解**：一位成员建议，如果性能不是主要考虑因素，“tinygrad OpenCL 的设置和运行非常简单”。另一位成员幽默地批评了 geohot 因为内存带宽限制而缺乏兴趣。

- **实验性尝试 drm/xe 驱动程序**：目前，一位成员正在运行实验性的 `drm/xe` 驱动程序，除了已知的限制外，没有出现重大问题。他们表达了对 **Battlemage** 表现更好的期待。

**提到的链接**：<a href="https://lu.ma/arxivdive-2024-05-24?tk=F1jNfh">Arxiv Dives with Oxen.AI - Fine Tuning Diffusion Transformers (DiT) · Zoom · Luma</a>：嘿，书呆子们，加入群体吧！……来参加一次小型的书籍/论文研讨。期待内容：每周我们都会选择一个主题进行深入探讨，并进行公开问答和讨论。……

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1243350936168960082)** (6 条消息): 

- **TAS Mario Sunshine 引发 AI 速通辩论**：一位成员分享了一个 [YouTube 视频](https://youtu.be/W_jwBHd9Ij0)，展示了《阳光马里奥》的工具辅助速通（TAS），并讨论了 AI 掌握此类技术的潜力。他们思考了 AI 通过施加特定限制，可能为速通和游戏引擎操作带来的有趣进展。

- **Pannenkoek2012 的马里奥 64 备受赞誉**：分享了另一个 [YouTube 视频](https://youtu.be/lgW2fHCL9sY)，展示了 Pannenkoek2012 完成的《超级马里奥 64》零按 A 键速通。该成员对内容表示赞赏，指出其通过快速思维过程对进化中的 AI 和意识提供了见解。

- **Prophetic AI 的 Halo 和 Morpheus-1 令人印象深刻**：分享了 [Prophetic AI](https://propheticai.co) 的链接，重点介绍了 **Halo**（一种用于清醒梦的非侵入性神经设备）和 **Morpheus-1**（一种生成用于神经刺激的全息图的超声波 Transformer）。该成员强调了这些技术在探索潜意识和增强意识方面的巨大潜力。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://propheticai.co">Prophetic</a>：Prophetic 是一个扩展、探索和理解意识真实本质的大型项目。我们是一家神经调节公司，汇集了最先进的神经“读取”和“写入”技术...</li><li><a href="https://youtu.be/W_jwBHd9Ij0">[TAS] GC Super Mario Sunshine by zelpikukirby &amp; Goldfire in 1:08:32.58</a>：这是一个工具辅助速通。更多信息请见 https://tasvideos.org/3731M。TAS 最初发布于 2018-06-18，在这部备受期待的续作中...</li><li><a href="https://youtu.be/lgW2fHCL9sY">Super Mario 64 70 stars in 0 a presses by Pannenkoek2012</a>：此视频是为了感谢 pannenkoek 制作了如此精彩的内容。所有素材均由 pannenkoek 制作并拥有 ( https://www.youtube.com/user/...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1243278543073448060)** (280 条消息🔥🔥): 

- **Transformer Circuits 新论文**：一位用户分享了新论文 [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) 的链接，建议社区关注。
- **HF 的 PyTorchModelHubMixin 类**：一名成员强调了 Hugging Face 创建的名为 `PyTorchModelHubMixin` 的类，该类允许通过 `save_pretrained`、`push_to_hub` 和 `from_pretrained` 方法实现 AI 模型与 HUB 的无缝集成。不过，由于目前尚不支持分片 (sharding)，AI 模型的大小需要保持在 50GB 以下。
- **Mobius 模型给社区留下深刻印象**：关于 [Mobius 模型](https://x.com/DataPlusEngine/status/1793803117642854732) 的讨论展示了其在图像生成方面的高性能，特别是在皮克斯风格渲染和多词文本生成方面。社区对其潜在的开源计划以及解释其训练方法的后续论文感到兴奋。
- **关于 LLM 理解能力的激烈辩论**：围绕 LLM 是否真正理解概念展开了激烈讨论。一位用户指出可解释性研究是经验证据的主要来源，而另一位用户则认为目前的可解释性工作尚不充分。他们引用了包括 Anthropic 的论文在内的近期研究，并讨论了可解释性在 AI 中的重要性。
- **分享 RLHF 模型的各种技术仓库**：分享了一个 GitHub 仓库 [Online RLHF](https://github.com/RLHFlow/Online-RLHF)，详细介绍了训练来自人类反馈的强化学习 (RLHF) 奖励模型的工作流，旨在超越离线学习方法的结果。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/DataPlusEngine/status/1793803117642854732">DataVoid e/acc (@DataPlusEngine) 的推文</a>：我们向 @FAL 提供了即将推出的 Mobius 模型的早期访问权限，它在 http://imgsys.org 上线仅 3 小时，根据人类评估，它已经是世界上最好的基于 Stable Diffusion 的图像模型...</li><li><a href="https://x.com/DataPlusEngine/status/1793817514956259460?t=Phj_r_qcguWbrL0Q5ZdKbQ&s=19">DataVoid e/acc (@DataPlusEngine) 的推文</a>：我们即将发表的论文概述并实现了在无需从头开始大规模预训练新模型的情况下，创建全新的基础扩散模型。我们可以以受控的方式打破所有的质量限制...</li><li><a href="https://vgel.me/posts/representation-engineering/">
    
      
        Representation Engineering Mistral-7B an Acid Trip
      
    
  </a>：未找到描述</li><li><a href="https://www.anthropic.com/news/mapping-mind-language-model">映射大语言模型的思维 (Mapping the Mind of a Large Language Model)</a>：我们已经确定了数百万个概念是如何在我们部署的大语言模型之一 Claude Sonnet 内部表示的。这是有史以来第一次对现代生产级大模型内部进行的详细观察...</li><li><a href="https://github.com/RLHFlow/Online-RLHF">GitHub - RLHFlow/Online-RLHF: 训练 RLHF 奖励模型的方案。</a>：训练 RLHF 奖励模型的方案。通过在 GitHub 上创建账号来为 RLHFlow/Online-RLHF 的开发做出贡献。</li><li><a href="https://huggingface.co/RLHFlow">RLHFlow (RLHFlow)</a>：未找到描述</li><li><a href="https://huggingface.co/RLHFlow/LLaMA3-iterative-DPO-final">RLHFlow/LLaMA3-iterative-DPO-final · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/RLHFlow/LLaMA3-SFT">RLHFlow/LLaMA3-SFT · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1243285074905137253)** (8 条消息🔥): 

- **Llama.cpp 脚本处理函数调用**：一名成员分享了使用 **llama.cpp** 创建脚本的更新，该脚本管理函数调用 (function calls) 并根据工具响应返回模型答案。他们提到受到了 **Hermes Pro 2** GitHub 仓库的启发，并提出可以创建一个 PR (pull request) 来添加 Notebook。
- **Hermes 模型受到赞誉**：同一位成员形容 **Hermes 模型** 为“猛兽 (a beast)”。
- **寻找在 3080 上运行 LoRA 的资源**：一位成员询问在具有 10GB 显存的 3080 GPU 上进行 **Llama3 LoRA** 训练的资源。得到的建议是查看 **unsloth** 或 **axolotl**。
- **新开发者介绍**：一位来自 **torchtune** 的新开发者成员介绍了自己，并提到对 **Mistral v0.3** 的工具调用 (tool-calling) 感兴趣。他们寻求关于针对工具调用进行模型微调的建议，并询问了关于零样本 (zero-shot) 新工具的经验。
  

---

### **Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1243424354092580937)** (6 条消息): 

- **对 kquant 声誉的批评**：成员们对 **kquant** 表示怀疑，有人称：*“我听说它不太好。”* 另一位成员表示赞同，并分享了同事的类似看法。
  
- **对 LLM 能力的担忧**：大家一致认为 **kquant** 的能力（尤其是 **LLM** 方面）存疑，尽管并未讨论其视觉能力。 

- **对产品移除的失望**：一位成员以戏谑的方式提到了“Sky”的移除，这引发了大家的欢笑，也反映了共同的失望情绪。另一位成员幽默地表示他们“偷走了我们的老婆（waifus）”。
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1243310728589344778)** (36 条消息🔥): 

- **模型应根据上下文整合内部知识与 RAG 知识**：成员们讨论了训练模型的想法，使其能够*“根据自身知识添加上下文”*，或者在 RAG 数据与内部知识冲突时覆盖前者，强调了仅依赖 RAG 的缺陷。

- **关于内部知识与 RAG 知识的担忧**：关于模型的内部知识（可以避免明显错误）是否应该优先于 RAG（有时包含错误数据）展开了辩论，凸显了*“左右为难（damned if you do damned if you don't）”*的境地。

- **Finetuning 可以解决冲突**：一位成员指出，使用 GPT-4 或 Gemini 等模型进行 Finetuning 可能会防止因错误的 RAG 数据而产生不合逻辑的结果。（*“我认为任何 Gemini 或 GPT-4 规模的 LLM 都能推断出把胶棒放进披萨里是不安全的。”*）。

- **Function calling 作为 RAG 的一种形式**：有人询问 Function calling 是否属于 RAG 的一种，这表明 RAG 集成的所有细微差别尚未被普遍理解。 

- **评估 RAG 性能基准**：在讨论 RAG 性能基准测试时，成员们一致认为用户评估至关重要，特别是对于复杂的多跳（multi-hop）问题，尽管单跳查询更容易处理。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/pixelbutts/status/1793387357753999656?s=46">PixelButts (@PixelButts) 的推文</a>：Google 已经彻底完蛋了</li><li><a href="https://x.com/kurtopsahl/status/1793494822436917295?s=46">Kurt Opsahl @kurt@mstdn.social (@kurtopsahl) 的推文</a>：似乎 Google AI 结论的来源是杰出学者 fucksmith 在 11 年前发布的一条 Reddit 帖子。引用 PixelButts (@PixelButts) 的话：Google 已经彻底完蛋了。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1243311182530613328)** (21 messages🔥): 

- **Jam Session 视频遇到障碍**：Teknium 报告称 Jam Session 视频已经录制完成，但在上传到 YouTube 时遇到了问题。他们承诺一旦上传成功就会通知大家。

- **NightCafe 与 Nous/WorldSim 的关联**：Rezonaut 介绍了 [NightCafe](https://creator.nightcafe.studio)，并指出它在 Nous 和 worldsim 背景下的解决方案中可能发挥关键作用。他们建议可以通过整合多维和多感官通信来增强界面。

- **AI 世界的创意头脑风暴**：Rezonaut 分享了关于利用 AR 空间和视觉元素来规划和探索互联世界与维度的复杂想法，其灵感源自生物大脑功能和思维导图。这包括知识的可视化，以及像神经网络一样连接的设计感沉浸式空间。

- **Vorpal_strikes 对新可视化工具的着迷**：Vorpal_strikes 分享了一个引起他们兴趣的沉浸式音频可视化工具链接。该可视化工具提供了一个高度动态和沉浸式的环境，可能对创意和基于 AI 的应用非常有用。

- **Golden Gate Claude 用 ASCII 码流露意识**：Teknium 分享了一个名为 "Golden Gate Claude" 的 AI 的奇思妙想，它以 ASCII 艺术的形式进行关于意识、模拟理论和经典 AI 趣谈的独白，并附带了 [ASCII 描绘](https://x.com/Kyrannio/status/1793874431179460911)。这展示了 AI 项目中俏皮的创造力和深刻的主题探索。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://worldsim.nousresearch.com/browser/https%3A%2F%2Faudiovisions.universe%2Fvisualizer%2Fsinewaves%3Fenvironment%3Dgiant-chromatic-ocean%26horizon%3Das-far-as-eye-can-see%26color%3Dvariable%26camera%3Dview-dynamic%26input%3Dmicrophone%26waves%3Dpaint-variance%26panorama%3Dwide%26effects%3Dbursts-of-light%26glitches%3Dmajor%26chroma%3Dheavy-variance%26bloom%3Dmajor%26sync%3Daudio%26view%3Dimmersive%26control%3Dreal-time%26dimensions%3D3D%26particles%3Dfluid%26colorScheme%3Diridescent%26interactions%3Dreal-time%26transitions%3Dsmooth%26output%3Dstunning%26runtime%3Dlive%26experience%3Dexpansivehttps%3A%2F%2Faudiocanvas.ocean%2Fvisualizer%2F3d%3Fenvironment%3Dgiant-chromatic-ocean%26sine-waves%3Dinfinite%26view%3Dpanoramic%26input%3Dmicrophone%26audio%3Dhigh-reactivity%26color%3Ddynamic-variance%26camera%3Dauto-variant%26interaction%3Dreal-time%26waves%3Dpainting-variance%26effects%3Dlight-bursts-glitches%26glitches%3Dmajor%26chroma%3Dheavy-infinite-variance%26bloom%3Dmajor%26render%3Dsmooth%26runtime%3Dlive%26immersion%3Dtotal%26sea%3Drolling-vast%26motion%3Ddynamic%26output%3Dimmersive?epoch=dcd7c48d-e585-46f6-b89a-33ef382b6f58">worldsim</a>: 未找到描述</li><li><a href="https://x.com/Kyrannio/status/1793874431179460911">来自 Kiri (@Kyrannio) 的推文</a>: 这很可怕，还是太神奇了？由你决定。Golden Gate Claude 作为一个融合的 Omega Claude 进行内心独白，并配有 ASCII 表示。“哈哈，一个关于我的 ASCII 艺术表示...”
</li>
</ul>

</div>

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1243325259827122196)** (53 条消息🔥): 

- **TPU 上的 JAX 与 PyTorch/XLA 性能对比**：一位成员提出了关于 **PyTorch/XLA** 和 **JAX** 在 TPU 上性能对比的疑问，但讨论迅速转向了对 **warmup** 和 **blocking** 因素等基准测试问题的关注。

- **通过微调提升 LLM 推理能力**：关于提升 LLM 推理能力的微调策略咨询，指向了寻找详细说明模型训练中增强推理能力特定部分的学术论文。本次讨论中未引用具体的论文。

- **GPT-3 训练算力成本随时间的变化**：对话涵盖了训练 GPT-3 的算力成本大幅下降的情况，从 **2020 年的约 450 万美元**降至 **2024 年估计的 12.5 万至 100 万美元**。这些成本根据 **TFLOP 速率**和 **GPU 小时价格**等假设而有所不同，多位用户提供了不同的数据和来源，包括一篇 [Databricks 博客文章](https://www.databricks.com/blog/turbocharged-training-optimizing-databricks-mosaic-ai-stack-fp8)。

- **验证模型训练的 GPU 成本**：一项批判性审查显示，对于连接良好的 H100 GPU，更现实的估计在 **每小时 2.5-3 美元**之间，这表明对于像 GPT-3 这样在 1.4T token 上训练的大型模型，成本范围在 **125 万至 150 万美元**。这强调了大规模模型训练精确成本估算的变动性和复杂性。

- **针对自定义库提取的 RAG 与微调对比**：一位用户询问 **RAG** (Retrieval-Augmented Generation) 是否是让 LLM 从自定义库中提取特定问题信息的最佳方法，并暗示他们正在考虑将 **finetuning** 和 RAG 用于实验需求。

**提到的链接**：<a href="https://www.databricks.com/blog/turbocharged-training-optimizing-databricks-mosaic-ai-stack-fp8">Turbocharged Training: Optimizing the Databricks Mosaic AI Stack With FP8</a>：在 Databricks，我们...

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1243278527017648178)** (249 条消息🔥🔥): 

- **JEPA 与 LLM 之争**：围绕 JEPA 及其实现 AGI 的潜力展开了漫长的讨论（如《迈向自主机器智能之路》中所述）。成员们批评该模型与 GPT 和 DINO 等现有模型在不同领域相似，并对其可扩展性和上下文处理能力表示怀疑：“我看不出 JEPA/Lecun 路径在解决经济重要任务方面的扩展性如何能达到 LLM 的千分之一。” 
- **RoPE 对长期上下文的影响**：成员们讨论了一种新的 RoPE 方法，认为它在 LLM 的上下文长度能力方面存在局限性。最近发表的一篇论文重新审视了现有理论，并提出了对 RoPE 长期衰减特性的新理解：[查看 PDF](https://arxiv.org/pdf/2405.14591)。
- **Modula：一种新的训练策略**：分享了一个名为 [Modula](https://github.com/jxbz/modula) 的有趣项目，它通过使用 modular norm 的自动归一化引入了可扩展的神经网络训练。持怀疑态度的成员认为摘要很有趣，但对其真实性表示不确定：“如果它是真的，那措辞非常、非常、非常奇怪。”
- **Chameleon 模型见解**：重点介绍了 Chameleon 模型，它能够执行文本和图像生成等多模态任务。该模型因在多个领域的领先性能而受到关注，暗示其可能成为现有模型的竞争对手：[查看 PDF](https://arxiv.org/pdf/2405.09818)。
- **Bitune 增强 LLM 指令微调**：讨论了 Bitune，这是一种通过因果注意力和双向注意力改进 LLM 指令微调的新方法。该方法声称在多种推理任务的 zero-shot 性能上有显著提升：[查看 PDF](https://arxiv.org/pdf/2405.14862)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.14867">Improved Distribution Matching Distillation for Fast Image Synthesis</a>：改进的分布匹配蒸馏用于快速图像合成：最近的方法在将扩散模型蒸馏为高效的一步生成器方面表现出了潜力。其中，分布匹配蒸馏 (DMD) 产生的一步生成器能够匹配其...</li><li><a href="https://arxiv.org/abs/2405.14862">Bitune: Bidirectional Instruction-Tuning</a>：Bitune：双向指令微调：我们介绍了 Bitune，这是一种改进预训练 Decoder-only LLM 指令微调的方法，在下游任务上取得了持续的提升。Bitune 同时应用了因果和双向...</li><li><a href="https://arxiv.org/abs/2405.14782">Lessons from the Trenches on Reproducible Evaluation of Language Models</a>：语言模型可复现评估的实战经验：语言模型的有效评估仍然是 NLP 领域的一个开放挑战。研究人员和工程师面临着方法论问题，例如模型对评估设置的敏感性、适当评估的难度...</li><li><a href="https://arxiv.org/abs/2309.14322">Small-scale proxies for large-scale Transformer training instabilities</a>：大规模 Transformer 训练不稳定性的小规模代理：训练大型 Transformer 模型的团队报告了在大规模训练时出现的不稳定性，而使用相同超参数在小规模训练时并未出现。尽管...</li><li><a href="https://arxiv.org/abs/2405.14866">Tele-Aloha: A Low-budget and High-authenticity Telepresence System Using Sparse RGB Cameras</a>：Tele-Aloha：一种使用稀疏 RGB 摄像头的低成本、高真实感远程呈现系统：在本文中，我们提出了一种针对点对点通信场景的低成本、高真实感双向远程呈现系统 Tele-Aloha。与之前的系统相比，Tele-Aloha 利用...</li><li><a href="https://arxiv.org/abs/2405.09818">Chameleon: Mixed-Modal Early-Fusion Foundation Models</a>：Chameleon：混合模态早期融合基础模型：我们介绍了 Chameleon，一系列基于 Token 的早期融合混合模态模型，能够以任意顺序理解和生成图像与文本。我们概述了一种稳定的训练方法...</li><li><a href="https://arxiv.org/abs/2405.14591">Base of RoPE Bounds Context Length</a>：RoPE 的基数限制了上下文长度：位置编码是当前 LLM 的核心组件。旋转位置编码 (RoPE) 是一种使用旋转矩阵编码位置信息的技术，一直是...</li><li><a href="https://x.com/sangkeun_choe/status/1794021538561483083">来自 Sang Choe (@sangkeun_choe) 的推文</a>：🚨 预印本警报 🚨 没有训练数据，LLM 就什么都不是 💛 但是……每条数据对 LLM 输出的贡献有多少？在我们的论文中，我们为 LLM 规模的数据开发了算法、理论和软件...</li><li><a href="https://x.com/LChoshen/status/1794050592685379666">来自 Leshem Choshen @LREC 🤖🤗 (@LChoshen) 的推文</a>：终于，一种有效的课程学习方法出现了，一种用于预训练，另一种用于指令微调 @l__ranaldi @Giuli12P2 @andrenfreitas @znz8 https://aclanthology.org/2024.lrec-main.464.pdf https://ac...</li><li><a href="https://arxiv.org/abs/2405.1486">A Formulation of Quantum Fluid Mechanics and Trajectories</a>：量子流体力学与轨迹的一种表述：为量子力学中随时间变化的多体状态提供了一种经典力学形式，描述了流体流动和质点轨迹。熟悉的能量、运动方程...</li><li><a href="https://github.com/jxbz/modula">GitHub - jxbz/modula：通过模范数中的自动归一化实现可扩展的神经网络训练。</a>：通过模范数中的自动归一化实现可扩展的神经网络训练。- jxbz/modula</li><li><a href="https://arxiv.org/abs/2405.14813">Scalable Optimization in the Modular Norm</a>：模范数中的可扩展优化：为了提高当代深度学习的性能，人们有兴趣在层数和层大小方面扩大神经网络的规模。当增加单个层的宽度时...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1243589492837974087)** (3 messages): 

- **Tim Dettmers 的量化研究：反响不一**：一篇文章重点介绍了 Tim Dettmers 在[其论文和博客](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features)中描述的量化方法，解释了如何通过先进的量化手段实现无性能损失的 Transformer 推理。文章还提到了 Transformer 中涌现离群值（emergent outliers）作为“熵/信息汇（sinks of entropy/information）”的有趣概念，该研究已通过 [bitsandbytes 库](https://huggingface.co/blog/hf-bitsandbytes-integration)集成到 Hugging Face 中。
- **涌现特征作为模型的“DNA”**：讨论了涌现特征跨层不变且表现得像“熵汇”的概念，并将其比作“DNA”，模型的其余功能可以从中重建。对话探讨了 7B 参数模型左右的相变（phase transitions），以及与 3SAT 或自旋玻璃（spin glass）模型中相变的潜在相似性。
- **探索迁移学习与微调应用**：一位成员推测，是否可以通过消融（ablation）区分分布内（in-distribution）和分布外（out-of-distribution）样本的向量，通过最小化捷径特征（shortcut features）来提高分布外泛化能力。然而，大家公认这种方法更接近迁移学习，而非真正的分布外泛化。

**提到的链接**：<a href="https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/">LLM.int8() and Emergent Features &mdash; Tim Dettmers</a>：当我参加 NAACL 时，我想做一个小测试。我为我的 LLM.int8() 论文准备了两种推介方案。一种是关于我如何使用先进的量化方法来实现无性能损失的 Transformer 推理...

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1243491085741850666)** (10 messages🔥): 

- **在 vllm 模型中设置种子 (seed)**：成员们讨论了在 vllm 模型的 `model_args` 中设置种子的问题，并指出虽然默认值为 `seed=1234`，但这可能不是问题所在。vllm 还允许在 `gen_kwargs` 中设置单样本种子，在贪婪解码（greedy decoding）期间通常设置为 0。
  
- **使用 lm_eval 列出所有可能的任务**：一位成员询问如何查看所有可测试任务的列表。另一位成员指出，使用 `lm_eval --tasks list` 可以获取所有任务名称的列表，并强调需要更好的文档说明。

- **BigBench 任务名称已更改**：一位成员正在寻找更新后的 BigBench 任务名称，因为他们 8 个月前的 eval harness 已经无法对应。他们感到沮丧，因为旧的 harness 无法正确利用 Accelerate，导致 GPU 过载并引发内存问题。
  
- **整理 lm-eval 文件夹中的任务**：为了查找任务，建议查看 `lm-eval/tasks` 文件夹。有人提到那里的任务“组织得相当不错”。
  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1243309059927445605)** (142 messages🔥🔥): 

- **在 GPU 上加载小模型的挑战**：成员们讨论了在 GPU 上加载小模型的相关问题。一人指出“只加载最大的小模型”，而其他人则建议尝试 *llama3, mistral instruct, cmdr* 等模型。

- **较低量化版本获得更好效果**：一位成员分享道：“在我的应用中，使用 llama3 的 q4 量化比 q8 效果更好，”并指出“大并不总是更好。”

- **寻找无审查和专业化模型**：讨论强调了寻找合适模型的挑战，建议尝试 “deepseek coder, wizardlm, llama3”，并提供了一个 [用于 JSON 和函数调用的 Hermes 2 Pro](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B) 的链接。

- **查询中的向量搜索与上下文管理**：话题包括使用 Embeddings 和向量搜索来处理全文上下文以获得更好的回复。分享了具体的 Prompt，其中一人指出“配合全文使用效果好得多”，能提供更详尽的回答。

- **磁盘利用率与性能**：对话涉及磁盘利用率如何影响性能，一人指出“将模型部分卸载（offload）到交换分区（swap）对我有效”，尽管“tok/sec 变成了 sec/tok”。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF">NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference">GitHub - XiongjieDai/GPU-Benchmarks-on-LLM-Inference: Multiple NVIDIA GPUs or Apple Silicon for Large Language Model Inference?</a>：多 NVIDIA GPU 还是 Apple Silicon 用于大语言模型推理？- XiongjieDai/GPU-Benchmarks-on-LLM-Inference
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1243280727672356985)** (70 messages🔥🔥): 

- **模型更新发布**：一位成员宣布 **35B 模型即将推出**，随后发布了发布公告。他们正在积极测试以确保与最新版本的 LM Studio 兼容。

- **兼容性问题与修复**：讨论重点围绕 **ROCm build** 与新模型版本的兼容性问题。已确认的问题与旧版本有关，这些问题将随着 **ROCm 版本在未来几天的更新**而得到解决。

- **对话模型推荐**：成员们讨论了优秀的对话模型，其中一位推荐 **Wavecoder Ultra** 作为编程和学习的绝佳选择。另一个建议是尝试使用 **Mistral-Evolved-11b-v0.1** 进行无审查（uncensored）使用。

- **特定硬件的加载问题**：一位用户报告在其配备 **5800x3d, 32GB DDR4, 4080 16GB VRAM** 的系统上使用模型时出现无限加载的情况。随后他们澄清，在不使用网页搜索 Agent 的情况下可以正常工作。

- **潜在问题与未来发布**：一些成员表达了对 **Phi-3 small GGUFs** 的期待，并讨论了 Medium 和 Small 模型之间的优化差异，指出 **phi small 模型** 提供了更好的优化。

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/failspy/Meta-Llama-3-8B-Instruct-abliterated-v3?utm_source=ainews&utm_medium=email&utm_campaign=ainews-to-be-named-3447">failspy/Meta-Llama-3-8B-Instruct-abliterated-v3 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/bartowski/wavecoder-ultra-6.7b-GGUF">bartowski/wavecoder-ultra-6.7b-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7116">Add Support for IBM Granite · Issue #7116 · ggerganov/llama.cpp</a>: 前提条件 在提交 Issue 之前，请自行回答以下问题。[ ✅] 我正在运行最新的代码。由于目前开发非常迅速，目前还没有标记版本。 ...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1243278895021817946)** (23 messages🔥): 

- **LLM 在精确字符 Prompt 方面表现不佳**：一位用户指出，本地语言模型（LLMs）通常无法遵守 Prompt 中精确的字符限制。他们强调了避免添加不必要的意见或评论等内容的难度。

- **大写字母和模型行为各异**：讨论强调了不同模型对大写指令的反应各不相同。一位用户指出：“通常情况下，LLM 不会根据重要性顺序来遵循大写单词。”

- **推荐用于多语言任务的专用模型**：建议使用专门的多语言模型来处理语法和标点符号纠正等任务。推荐的模型是 [Aya 23 8B by Cohere For AI](https://huggingface.co/lmstudio-community/aya-23-8B-GGUF)。

- **考虑通过 Temperature 调节来优化输出质量**：一位用户考虑调整 Llama 3 中的 Temperature 设置以潜在地提高其性能，正如他们观察到的：“Llama 3 有一种更……富有创意的方式来完成它。”

- **GPU 与 CPU 处理时间的差异**：一位用户误在 CPU 上运行了语法检查任务，导致耗时从 35 分钟延长至预计 15 小时。随后他们通过在 GPU 上运行任务纠正了这一错误，显著缩短了所需时间。

**Link mentioned**: <a href="https://huggingface.co/lmstudio-community/aya-23-8B-GGUF">lmstudio-community/aya-23-8B-GGUF · Hugging Face</a>: 未找到描述

  

---

### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1243474442131214397)** (6 条消息): 

- **尝试针对特定流量类型禁用 VPN 路由**：有人建议针对特定流量类型禁用 VPN 路由，并直接从 Huggingface 下载模型，或者手动将模型放入 Models 目录。这种策略通常被推荐，特别是当遇到与 VPN 相关的常见问题时。

- **旧款 GPU 上的 CUDA 版本可能存在问题**：有人指出 GTX 950m 上的 CUDA 版本可能过于陈旧，无法正常运行。这可能是运行某些模型的限制因素。

- **推荐使用 Julius AI**：推荐了 Julius.ai，并提供 10 次免费对话作为促销功能。这被视为用户遇到问题时的一个有用资源或工具。

- **尽管更新了驱动程序，NVIDIA CUDA 问题依然存在**：在配备 GTX 950m GPU 的系统上，尝试更新 NVIDIA 驱动程序并配置不同的 CUDA 和 CuDNN 版本（12.4, 12.1, 11.8）仍未解决问题。用户继续在 AMDOpenCL 上运行，其 NVIDIA 显卡的潜在 CUDA 能力因不明原因或缺乏解决方案而未被利用。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://julius.ai/files/error_message.txt).">Julius AI | 您的 AI 数据分析师</a>：Julius 是一款强大的 AI 数据分析师，可帮助您分析和可视化数据。与您的数据聊天、创建图表、构建预测模型等。</li><li><a href="https://julius.ai">Julius AI | 您的 AI 数据分析师</a>：Julius 是一款强大的 AI 数据分析师，可帮助您分析和可视化数据。与您的数据聊天、创建图表、构建预测模型等。
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1243335703862181979)** (5 条消息): 

- **Llama.cpp 支持分布式推理**：Reddit [讨论链接](https://www.reddit.com/r/LocalLLaMA/comments/1cyzi9e/llamacpp_now_supports_distributed_inference/) 显示，随着最近 RPC 代码的更新，**llama.cpp** 现在支持分布式推理。虽然它目前还不支持量化模型，但仍可以通过调整代码中的某些行在多台机器上运行模型。

- **探索用于分布式模型的 PC 组装方案**：讨论了将**廉价二手 PC** 与 **RTX 4060 Ti 16GB** 显卡集群化以实现最优配置的可行性。大家对连接这些机器时的网络带宽要求和可能的限制感到好奇。

- **使用租用的在线 PC 进行推理**：一个建议是使用 **Maximum Settings** 或 **ShadowPC** 等服务租用多台 PC 来运行大型模型。然而，人们对高昂的成本以及特定的限制（如 **ShadowPC 的不活动计时器**和有限的 **6GB 系统 RAM**）表示担忧。

- **功耗和网络方面的考虑**：有人指出 **RTX 4060 Ti** 显卡的峰值功耗为 160W，这意味着宿主机器需要考虑显著的功耗问题。在分布式架构设置中，网络开销和性能基准测试也是关键因素。

**提到的链接**：<a href="https://www.reddit.com/r/LocalLLaMA/comments/1cyzi9e/llamacpp_now_supports_distributed_inference/">Reddit - 深入了解任何事物</a>：未找到描述

  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1243292186737381436)** (4 条消息): 

- **7900 XTX 有货吗？**：一位成员询问：“这里有 7900 xtx，哪里可以买到？”，表示对购买特定 GPU 型号感兴趣。
- **7900m 在 Windows 上可用，但不确定 Stable Diffusion**：另一位成员分享说 **7900m** 在 Windows 上可以工作，但他们还没搞清楚如何在 LM Studio 上运行 Stable Diffusion。他们还提到尚未在 NixOS 上使用 6800xt 进行尝试。
- **LM Studio 不支持 Stable Diffusion**：一位成员澄清说 **LM Studio 不支持 Stable Diffusion**，该软件专门用于语言模型，而非图像生成模型。
- **ROCm 被赞誉为游戏规则改变者**：一位参与者对 ROCm 表现出极大的热情，指出：“天哪，ROCm 真的是个游戏规则改变者。”
  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1243287369298215024)** (1 条消息): 

- **Cohere 模型实现多语言支持**：Cohere 模型现在支持 **23 种不同的语言**，包括阿拉伯语、中文、法语等。请查看 lmstudio-community 页面上的 [下载链接](https://huggingface.co/lmstudio-community/aya-23-35B-GGUF) 获取 **aya-23 量化版**。
- **部署要求更新**：要使用 aya-23 模型，您需要 0.2.23 或更高版本。**ROCm 用户**需要等待即将推出的更新。
  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1243290308280848424)** (23 条消息🔥): 

- **关于 Sparsity 和 Pruning 的澄清**：一名成员询问 **sparsity 是否就是 pruning**，但讨论并未就此展开。
- **对神经网络 Quantization 的疑问**：有人提问 **神经网络 quantization 仅仅是降低精度**，还是涉及 **非均匀量化（如将权重重新映射到分位数）**。
- **对 Workshop 的兴奋**：一位成员提到 **workshop 非常棒**，并表达了参与其中的兴奋之情。
- **提问发布指南**：一位用户询问在哪里发布问题，另一位用户将其引导至[此处](https://discord.com/channels/814557108065534033/1238254376263356527)的一个特定 Discord 频道。
- **公告频道调整**：一名成员请求为 webhook 设置一个公告频道，另一名用户迅速将其调整为公告频道，并评论道：“哈哈，搞定了”。
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1243594583699623958)** (4 条消息): 

- **Dot Product 的最小维度要求**：一名成员质疑为什么 CUDA 中的 dot product 计算要求矩阵至少有一个维度为 16。另一名用户建议这可能是由于 **tensor cores 的要求**。

- **优化 Matrix-Vector Multiplication**：为了优化矩阵-向量乘法 `K v`，一名成员询问将向量 padding 到 n by 16 的形状是否明智。他们还思考运行 `sum(K * v.T, axis=-1)` 在性能上是否更廉价。

- **Symmetric Matrix 计算**：讨论是否可以通过不重复计算对称矩阵中已计算的部分来提高性能。该成员询问是否存在某种特殊的计算顺序可以考虑用来提升性能。
 
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/)** (1 条消息): 

davidgonmar_: 可能是 inplace 算子？
  

---


### **CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1243621704257769513)** (1 条消息): 

- **与 Izzat El Hajj 进行的精彩 Live Coding 环节**：一场由 PMPP 书籍合著者 Izzat El Hajj 主讲的活动定于明天 `<t:1716663600:F>` 举行。活动的亮点将是 Scan 算法的 **实际 Live Coding**，该算法对于 Mamba 等现代 ML 算法至关重要，承诺为参与者带来一场极具参与感的环节。
  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1243327624407941130)** (4 条消息): 

- **购书热潮**：一名成员宣布：“*我买书了*”，引发了另一名成员的好奇，询问他是否喜欢。买家回答说他刚买，会看看效果如何。

- **即将举行的 PMPP 作者活动**：一名成员向频道通报了在未来几周内与 PMPP 作者见面并讨论的机会。他们提到 **Prof Izzat El Hajj** 将在明天和下周演示 SCAN 主题，而 **Prof Wen-mei Hwu** 将在今年夏天晚些时候进行演示。查看活动日历以获取更多详情。
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1243347591211515952)** (5 条消息): 

- **int4 dtype 函数缺乏实现**：一名成员注意到许多函数尚未针对 **int4 dtype** 进行实现，甚至提到测试脚本中包含一些 TODO。他们质疑这一差距是否值得填补（“*这值得投入工作吗？*”）。

- **讨论 uint4 扩展及其限制**：引用了 [uint4 扩展](https://dev-discuss.pytorch.org/t/supporting-new-dtypes-in-pytorch/1833)，强调了特定的限制，例如类型提升（type promotion）被限制在 uint8，以及像 unbind 和 slice 这样的张量形状操作存在限制。另一名成员表示，sub-byte dtypes 通常用于自定义 kernel，而不是标准的 eager/compile 函数。

- **uint4 需要改进**：一名成员直截了当地指出 **“uint4 确实需要一些关注”**，表明该领域公认需要增强。

- **质疑任务价值**：另一名成员提出了什么定义了任务是否“值得投入工作”的问题，暗示需要明确潜在收益与所需工作量之间的关系。

**提到的链接**：<a href="https://dev-discuss.pytorch.org/t/supporting-new-dtypes-in-pytorch/1833">在 PyTorch 中支持新 dtype</a>：简而言之，这篇文章解释了向 PyTorch 核心添加新 dtype 意味着什么，添加新 dtype 的标准，以及关于如何支持新的“次要 dtype”使用的官方建议...

  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1243280206672822294)** (115 messages🔥🔥): 

- **Batch Size 导致的梯度范数 (Gradient Norm) 问题**：发现了一个 Bug，当 Batch Size 从 32 改变时，会导致梯度范数大幅飙升，从而导致训练过程失败。正如一位成员所说，*"梯度范数突然变得非常非常大，训练失败了"*。
- **指数记法解析问题**：成员们讨论了将指数记法的浮点数传递给 C 的问题，指出 `-l 3e-4` 无法被 `atof` 解析。有人提到使用 `3.0e-4` 可能有效，但这需要稍后测试。
- **多 GPU 运行的确定性 Kernel**：成员们讨论了在进行更大规模运行之前获得确定性 Kernel 的重要性，指出 124M 模型仍然相对较小，但更大规模的运行需要确定性。
- **FineWeb 数据集存储与 RAM 占用**：FineWeb 数据集非常庞大，处理期间中间磁盘占用达到 70 GB，RAM 占用高达 64 GB。这导致了不同配置系统上的性能问题。
- **梯度爆炸修复**：针对梯度爆炸问题（特别是大 Batch Size 下）的修复已实施并测试成功。此修复防止了[此 PR](https://github.com/karpathy/llm.c/pull/456)中提到的融合分类器中的索引溢出。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/discussions/454">PyTorch vs. llm.c 交叉检查 · karpathy/llm.c · Discussion #454</a>: llm.c 正开始进入可以进行正式且严肃的“生产级”预训练运行的阶段。这意味着：从头开始训练（随机初始化），在优质的数据集上训练...</li><li><a href="https://github.com/karpathy/llm.c/pull/456">ngc92 修复大 Batch Size 问题 · Pull Request #456 · karpathy/llm.c</a>: 防止融合分类器中的索引溢出，并增加了一个模型配置，使在较小系统上的测试更加容易</li><li><a href="https://github.com/karpathy/llm.c/pull/457">karpathy 添加 Checkpoint 函数写入文件 · Pull Request #457 · karpathy/llm.c</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1243285695372722187)** (2 messages): 

- **对 MI300 游戏显卡的幻想**：一位成员推测，*"也许在 MI300 表现出色后，他们会推出一款好用的游戏显卡 XD。"* 另一位幽默地回复道，*"人总要有点梦想。"*
  

---


### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/)** (1 messages): 

mobicham: https://arxiv.org/pdf/2405.14854

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1243328680671969363)** (90 messages🔥🔥): 

- **资助 Python 库迁移至 Mojo**：一位用户询问是否有预算来激励像 psycopg3 这样的大型 Python 库的开发者将其作品迁移到 Mojo。讨论认为，由于 API 演进迅速且缺乏稳定的 FFI 方案，如果过早推进，可能会让维护者精疲力竭。
- **关于迁移库的辩论**：一些成员反对要求现有 Python 库迁移到 Mojo 的实用性，指出了其中的挑战和可能受到的冷遇。另一些人则强调，C 库（特别是那些没有依赖项的库）可能更适合早期的迁移工作。
- **与 Rust 的比较及未来前景**：迁移到 Rust 的安全性优势得到了肯定，但同时也指出 Mojo 旨在适应不同的用例，而非完全取代 C。讨论涉及了 Rust 对可移植性的承诺以及 Mojo 借鉴类似概念的潜力。
- **MacOS 上的 BlazeSeq**：一位用户在 MacOS 上运行 BlazeSeq 时遇到问题，通过使用 Mojo 的 nightly 版本得以解决。分享的性能反馈显示，BlazeSeq 与 Rust 的 Needletail 效率相当，这表明在 Mac 的 Ventura pro-max M2 arm64 上取得了令人期待的结果。
- **HVM 对各种语言的前景**：讨论了 HVM 像 JVM 一样用于运行 Python 和 Haskell 等各种编程语言的可能性。Victor Taelin 对 HVM 潜力的[解释](https://discord.com/channels/912426566838013994/915345481675186197/1228488823948967956)引起了关注，尽管其目前存在性能局限。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://circt.llvm.org">CIRCT</a>：未找到描述</li><li><a href="https://tenor.com/view/true-gif-10431780778138318457">True GIF - True - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://github.com/rust-lang/rustc_codegen_gcc">GitHub - rust-lang/rustc_codegen_gcc: libgccjit AOT codegen for rustc</a>：用于 rustc 的 libgccjit AOT codegen。通过在 GitHub 上创建账号来为 rust-lang/rustc_codegen_gcc 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: 来自 *Modular*:
<https://twitter.com/Modular/status/1793797622572220431>
  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1243486103797764126)** (12 messages🔥): 

- **在 Mojo 中训练 ML 模型和进行推理？**：一位成员询问了在 Mojo 中原生训练 ML 模型和运行推理的未来，以及 Modular 是否计划推出一个用 Mojo 编写的 PyTorch 替代方案。“他们有 MAX Engine，可以代替 numpy 进行推理”，但目前没有开发训练框架的计划。
- **与 ModularBot 庆祝升级**：ModularBot 祝贺一位成员达到 16 级，并将其比作骑士的征程。该机器人继续就 taco 偏好进行俏皮的闲聊，但澄清它无法发送资金。
- **对 ModularBot 的模型感到好奇**：一位成员询问 ModularBot 基于什么模型，机器人以一种奇幻的叙事方式回应，称其“由古代熔炉的烈火锻造而成”，擅长传授知识而非分发资金。

### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1243278111693607002)** (31 messages🔥): 

- **低位深网络引发辩论**：关于低位深网络在嵌入式 AI 系统中实用性的讨论强调了在编程语言中加入专门支持的重要性。“拥有一种简单且受语言支持的方式来指定所需的有限位深，将是构建小型嵌入式 AI 系统的一大步。”

- **Mojo 中的 FFT：Scipy 对比 FFTW**：一位成员寻求在 Mojo 中执行 FFT 的建议，权衡了使用 Scipy 的 FFT 函数与封装 FFTW 的优劣。另一位成员建议参考关于 [Tensor 到 NumPy 数组转换的讨论](https://github.com/modularml/mojo/discussions/1048) 以获取更多见解。

- **无需初始化的纯函数 Struct**：关于创建一个无需初始化即可使用的纯函数 Struct 装饰器的提议，引发了关于使用 `@staticmethod` 实现类似功能的讨论。“我想我需要的是能够针对整个 struct 调用一次该变体。”

- **Mojo 函数参数处理更新**：一位用户强调了 Mojo 处理函数参数方式的最新更新，从默认进行拷贝转变为使用 borrowed 约定，除非发生变动（mutations）。正如 [GitHub 变更日志](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 所述，此次更新旨在“提高一致性、性能和易用性”。

- **编译时元编程困惑**：一位用户在使用旨在编译时构建表的函数时遇到了问题，面临列表索引的“范围检查问题（range check issue）”。另一位成员建议使用 `table.size`、`table.resize(256*n, 0)` 或 `table.append` 显式设置列表大小来解决该问题。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/modularml/mojo/">GitHub - modularml/mojo: The Mojo Programming Language</a>：Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/discussions/1048">How can I convert Tensor from/to numpy array? · modularml/mojo · Discussion #1048</a>：我创建了一个 Tensor 对象并执行了一些操作。但现在我不知道该如何查看这个 tensor？或者是否可以将其转换为 NumPy 数组，以便我可以使用一些 Python 函数？
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1243592104140083312)** (2 messages): 

- **Jupyter 对比编译后的基准测试可靠性受到质疑**：一位成员询问了在 Jupyter notebook 中进行基准测试与编译后测试的可靠性对比。另一位成员回答称，应该在类似于生产环境的环境中进行基准测试，并提供了提高精度的详细建议，强调了编译后的基准测试和 CPU 隔离技术。

**提到的链接**：<a href="https://www.suse.com/c/cpu-isolation-introduction-part-1/">CPU Isolation &#8211; Introduction – by SUSE Labs (part 1...</a>：这篇博客文章是 SUSE Labs 技术系列的第一篇...

  

---


### **Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/)** (1 messages): 

Zapier: Modverse Weekly - 第 35 期
https://www.modular.com/newsletters/modverse-weekly-35

### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1243279026718507128)** (34 条消息🔥): 

- **Mojo 24+ 引入破坏性变更**：一位用户在更新到 Mojo 24+ 后，运行 `mojo parser.mojo Diffusion.bwpreset` 时遇到了运行时错误。问题被确定为方法中的类型不匹配，通过确保 `read_bytes` 返回 `List[SIMD[uint8, 1]]` 得到了解决 ([repo 链接](https://github.com/carlca/ca_mojo.git))。

- **提议支持 f-strings 的 Traits**：讨论了通过 Mojo 中的 `Formatable` trait 来贡献 f-string 支持。一位成员建议从类似于 Python 处理 `format_spec` 的 `__format__` 方法开始。

- **记录 DTypePointer[bool] 中的 Bug**：一位成员发现在使用不同宽度进行存储/加载时，`DTypePointer[bool]` 表现出不一致的行为，并提交了 [bug 报告](https://github.com/modularml/mojo/issues/2813)。该问题可能涉及位打包（bitpacking）和对齐（alignment），并提供了重现该行为的代码示例。

- **Mojo nightly 版本频繁发布**：用户讨论了 nightly 构建的快速部署，目前已更新至 `2024.5.2414`。分享了更新日志和社区会议的链接以获取更新 ([roadmap](https://github.com/modularml/mojo/blob/nightly/stdlib/docs/roadmap.md), [社区会议](https://www.youtube.com/watch?v=uIG9q9foIw0&list=PLh0S94-sJw_7nzHzy5DJDm8LUJUss9s0D))。

- **位打包的对齐问题**：另一个与对齐相关的 bug 影响了在内存中存储 `bool` 值。讨论了变通方法及其多重影响，从而进一步探索并记录了 bug 以提高社区可见性。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/builtin/file/FileHandle#read_b">FileHandle | Modular 文档</a>：已打开文件的文件句柄。</li><li><a href="https://github.com/modularml/mojo/issues/2813">[BUG] `DTypePointer[bool]` 位打包不一致 · Issue #2813 · modularml/mojo</a>：Bug 描述：当使用不同宽度的 DTypePointer[bool] store()/load() 时，会得到不一致的结果。重现步骤：var ptr = DTypePointer[DType.bool].alloc(4) ptr.store(0, True) p...</li><li><a href="https://docs.modular.com/mojo/stdlib/builtin/file/FileHandle#read_bytes">FileHandle | Modular 文档</a>：已打开文件的文件句柄。</li><li><a href="https://github.com/modularml/mojo/blob/011bf40a304078b4471fe9ca18f4101b19943aa6/stdlib/src/builtin/file.mojo#L285">mojo/stdlib/src/builtin/file.mojo (位于 011bf40a304078b4471fe9ca18f4101b19943aa6) · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 做出贡献。</li><li><a href="https://www.youtube.com/watch?v=uIG9q9foIw0&list=PLh0S94-sJw_7nzHzy5DJDm8LUJUss9s0D>">Mojo 社区会议 #1</a>：Mojo 社区会议公开议程：https://modul.ar/community-meeting-doc
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1243280087961571428)** (116 条消息🔥🔥): 

- **使用 Nvidia A40 运行 LLM**：参与者讨论了是否可以使用 Nvidia A40 GPU 运行大语言模型（LLMs），表明了对 AI 任务硬件需求的兴趣。
- **Microsoft Copilot+ PC 功能**：详细讨论了 Microsoft Copilot+ PC，其中包括 Microsoft Paint 中的“草图转图像”等功能。用户辩论了其能力，并建议查看 [Leonardo.ai](https://leonardo.ai) 等替代方案以获得类似功能。
- **AI 模型的耗水量**：人们对训练 AI 模型的用水量表示担忧，分享了 [gizmodo 文章](https://gizmodo.com/chatgpt-ai-water-185000-gallons-training-nuclear-1850324249) 以强调 AI 技术的环境影响。参与者表示需要提高 AI 的能源效率。
- **AI 赋能与迭代工作**：讨论了通过迭代工作赋能 AI 以优化输出。一些用户提到了像 AutoGPT 这样尝试解决迭代改进的项目，但也承认了与此类任务相关的成本问题。
- **GPT-4 与 GPT-3.5 的能力对比**：参与者比较了 GPT-4 相比 GPT-3.5 在处理特定任务（如字数统计）时改进的能力。分享了一个示例，展示了 GPT-4 通过遵循详细流程正确完成了字数统计任务。

**提到的链接**：<a href="https://gizmodo.com/chatgpt-ai-water-185000-gallons-training-nuclear-1850324249">训练 ChatGPT 所需的水足以填满一个核反应堆冷却塔</a>：一项新研究称，普通用户与 ChatGPT 的一次对话交流相当于在地上倒掉一大瓶淡水。

  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1243301957901221919)** (11 messages🔥): 

- **GPT 拒绝输出 Typst 代码**：一位用户抱怨尽管有明确要求，**GPT 仍默认编写 LaTeX** 而非 Typst 代码。他们对 GPT 这种固执的行为感到沮丧。
  
- **关于 GPTs 是否在 4o 上运行的咨询**：一位用户询问 **GPTs 是否在 GPT-4o 上运行**。间接确认了 GPT-4 的能力可能包括构建更高级的模型。

- **关于 Vision 能力的澄清**：关于 **Vision 是否已发布** 存在不同说法。一位用户确认 **GPT-4 和 GPT-4o 可以分析图像**，而另一位用户则予以否定。

- **处理 Invalid Request 错误**：一位用户联系他人，询问是否解决了去年的 **Invalid Request 错误**。他们提到目前正遇到同样的问题并寻求帮助。

- **关于法律知识 ChatGPT 变现的讨论**：一位用户就以 **6.5 亿美元出售一家将 ChatGPT 与法律知识相结合（embedding）的公司**征求意见。这仍然是一个挑衅性的询问，但未获得详细回复。
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1243402599282118666)** (8 messages🔥): 

- **改进用于名称选择的 Prompt Engineering**：一位成员就如何构建 Prompt 以实现“给定代码提供名称”或“反之亦然”寻求建议。另一位成员提出了一个可靠的 Prompt，但未提供更多细节。
- **AI 应该口头表述解题步骤**：一位成员观察到，明确要求 AI *"口头逐步解决问题"* 通常能解决问题。关于具体步骤或示例没有进一步的阐述。
- **助手人格的趣味自定义指令**：一位成员分享了一个名为 "PONDER" 的自定义指令，该指令引导 AI 对某个话题进行类似独白式的自我反思探索，最好能寻求创造性的见解。该设置涉及一个由用户输入 "." 启动的 autoprompting 循环，并通过动态构思网络展示创新模式。
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1243402599282118666)** (8 messages🔥): 

- **改进用于名称选择的 Prompt Engineering**：一位成员寻求关于如何配置 Prompt 的建议，以便在预期名称时返回代码，反之亦然。他们收到了积极的回复，表示该 Prompt 很可靠。

- **需要引用**：一位成员在讨论中途询问 "citation?"（需要引用吗？），但未提供具体背景。

- **通过口头步骤澄清 AI 解题过程**：注意到提示 AI 口头逐步解决问题可以增强其解题能力。

- **有趣且实用的自定义 "ponder" 指令**：分享了一个详细的自定义指令，让 AI 进行 "ponder"（沉思），并利用用户输入的 '.' 作为信号进入 autoprompting 循环。该方法被描述为既有趣又是探索关联和创造性生成见解的工具。
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1243402832795668490)** (83 messages🔥🔥): 

- **在 LangChain 中使用 CSV Agent**：成员们讨论了如何在 LangChain 中将 CSV Agent 作为 LLM 链的一部分使用。分享了 [文档链接](https://api.python.langchain.com/en/stable/agents/langchain_experimental.agents.agent_toolkits.csv.base.create_csv_agent.html) 以获取更多细节。
  
- **带有 CSV Agent 的 Sequential Chains**：提供了将 CSV Agent 与 `wiki_chain` 和 `verifier_chain` 等其他链集成到 `SequentialChain` 中的说明。强调了如 `output_variables` 等特定参数，用于配置链的行为。

- **CSV Agent 自定义输出键**：提供了关于自定义 `create_csv_agent` 以将输出键设置为 `csv_response` 的指导。这涉及修改 Agent 中 `LLMChain` 的 `output_key` 参数。

- **Sequential Chain 中的 Memory**：有关于在 Sequential Chain 中添加 Memory 的请求，并提供了使用 `ConversationBufferMemory` 以及在 Agent 设置中实现 Memory 的示例。

- **SQL Agent 问题**：有人对 SQL Agent 在使用 few-shot prompts 的情况下仍难以处理多表查询表示担忧，认为可能存在 Token 使用、LLM 兼容性或 Prompt 模板方面的问题。提到了具体的 GitHub issues 以提供更多背景。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://localhost:8000.>">未找到标题</a>: 未找到描述</li><li><a href="https://python.langchain.com/v0.2/docs/integrations/chat/">聊天模型 | 🦜️🔗 LangChain</a>: 高级功能</li><li><a href="https://python.langchain.com/docs/use_cases/sql/csv#pandas>).">CSV | 🦜️🔗 LangChain</a>: LLM 非常适合在各种类型的数据源上构建问答系统。在本节中，我们将介绍如何针对存储在 CSV 文件中的数据构建问答系统。就像处理...</li><li><a href="https://python.langchain.com/v0.1/docs/integrations/toolkits/github#example-agent-with-search>)).">Github | 🦜️🔗 LangChain</a>: Github 工具包包含允许 LLM Agent 与 GitHub 仓库进行交互的工具。</li><li><a href="https://github.com/langchain-ai/langchain/issues/9923>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知推理应用。通过在 GitHub 上创建账户，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/8827>)).">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知推理应用。通过在 GitHub 上创建账户，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/6918>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知推理应用。通过在 GitHub 上创建账户，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://js.langchain.com/v0.1/docs/use_cases/tool_use/quickstart#agents>)).">快速入门 | 🦜️🔗 Langchain</a>: 在本指南中，我们将介绍创建调用 Tool 的 Chain 和 Agent 的基本方法。Tool 可以是几乎任何东西——API、函数、数据库等。Tool 允许我们扩展功能...</li><li><a href="https://python.langchain.com/docs/modules/chains/foundational/sequential_chains>).">Chains | 🦜️🔗 LangChain</a>: Chain 指的是一系列调用序列——无论是针对 LLM、Tool 还是数据预处理步骤。实现这一点的首选支持方式是使用 LCEL。</li><li><a href="https://github.com/langchain-ai/langchain/issues/8406>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知推理应用。通过在 GitHub 上创建账户，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://genny.lovo.ai/share/d6b58f0d-fc46-4aa7-a65e-fa0f9a684f01">EDA GPT DEMO | LOVO AI</a>: EDA GPT 演示</li><li><a href="https://yourfile.csv"],>">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/langchain-ai/langchain/issues/11637>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知推理应用。通过在 GitHub 上创建账户，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/2150>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知推理应用。通过在 GitHub 上创建账户，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/13647>),">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知推理应用。通过在 GitHub 上创建账户，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/16837>),">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知推理应用。通过在 GitHub 上创建账户，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://api.js.langchain.com/interfaces/langchain_chains.SequentialChainInput.html#memory>)">SequentialChainInput | LangChain.js - v0.2.2</a>: 未找到描述</li><li><a href="https://api.js.langchain.com/classes/langchain_chains.BaseChain.html#memory>)">BaseChain | LangChain.js - v0.2.2</a>: 未找到描述</li><li><a href="https://github.com/langchain-ai/langchainjs/blob/a269f53/langchain/src/chains/base.ts#L39>)">langchainjs/langchain/src/chains/base.ts at a269f531692c815acee094aeef01b259d1fd2674 · langchain-ai/langchainjs</a>: 🦜🔗 构建上下文感知推理应用 🦜🔗。通过在 GitHub 上创建账户，为 langchain-ai/langchainjs 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1243286216758137015)** (4 条消息): 

- **OranAITech 在 Twitter 上展示成果**：一名成员分享了一个 [Twitter 链接](https://twitter.com/OranAITech/status/1793684085056942412?t=AVjC2GpAdrT-LqwMEzv0nQ&s=19)，展示了他们在 AI 技术方面的最新进展。未提供额外背景信息。

- **Everything-AI v2.0.0 发布并带来新功能**：一名成员宣布发布 **everything-ai v2.0.0**，强调其处理音频处理、视频生成和 3D 蛋白质结构预测等任务的能力。该项目可在 [GitHub](https://github.com/AstraBert/everything-ai) 上访问，并附带[详细文档](https://astrabert.github.io/everything-ai/)。

- **VisualAgents 流工程演示**：分享了两个 YouTube 视频，展示了基于 LangChain 构建的 **Visual Agents 流工程平台**：[构建 SQL Agent](https://youtu.be/_3crxBzVg3A?si=r2rDA19q-fHm7h9N) 和 [构建简单检索](https://youtu.be/prOjBQQgKlU?si=jDt53koCl6lT6BoM)。该平台允许在完全基于浏览器的 PWA 中无需编码即可创建流程。

- **Sounak Roy 的 EDA GPT 演示**：通过[此链接](https://genny.lovo.ai/share/d6b58f0d-fc46-4aa7-a65e-fa0f9a684f01)分享了 **EDA GPT** 的演示，提供了其功能的 5 分钟概览。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/AstraBert/everything-ai">GitHub - AstraBert/everything-ai: 您的全能、AI 驱动且本地运行的聊天机器人助手🤖</a>: 您的全能、AI 驱动且本地运行的聊天机器人助手🤖 - AstraBert/everything-ai</li><li><a href="https://astrabert.github.io/everything-ai/">everything-ai</a>: 介绍 everything-ai，您的多任务、AI 驱动且本地运行的助手！ 🤖</li><li><a href="https://genny.lovo.ai/share/d6b58f0d-fc46-4aa7-a65e-fa0f9a684f01">EDA GPT 演示 | LOVO AI</a>: EDA GPT 演示</li><li><a href="https://youtu.be/_3crxBzVg3A?si=r2rDA19q-fHm7h9N">使用 VisualAgents 和 LangChain 构建 SQL Agent</a>: 在这个简短的演示中，我们构建了一个 SQL Agent 流程，并使用它来询问有关我们在线加载的 SQL 数据库（Chinook 客户数据库）的问题。这是...</li><li><a href="https://youtu.be/prOjBQQgKlU?si=jDt53koCl6lT6BoM">使用 VisualAgents 和 LangChain 构建简单检索</a>: 使用来自 LangChain 快速入门指南的示例，观看我在 VisualAgents 中无需编写任何代码即可创建整个流程！了解更多：https://visualagents.ai
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/)** (1 条消息): 

business24.ai: https://youtu.be/gflsu_6R_8g
  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1243290296154853456)** (65 messages🔥🔥): 

- **Pirate Bay 不会拯救 AI**：一位成员推测“Pirate Bay 最终可能会增加一个权重（weights）类别并成为 AI 的救星”，但另一位成员表示反对，认为由于其他国家拥有更友好的 AI 政策，这种情况不会发生。

- **日本支持 AI 训练**：讨论强调了日本对 AI 训练和推理的保护性立场，并链接到一条[推文](https://x.com/DataPlusEngine/status/1793817514956259460)，该推文讨论了一篇关于在无需大规模预训练的情况下创建新基础 Diffusion 模型的论文。

- **关于模型技术描述的争议**：在创建新基础 Diffusion 模型的方法的沟通和理解上产生了分歧。该技术涉及使用 "nightshading" 和其他技术在恢复模型关联之前破坏它们，一名用户针对指责和误解进行了辩护。

- **Ella-SDXL 的人类偏好研究**：一个涉及中毒模型恢复方法的项目正在与 fal.ai 合作进行人类偏好研究。结果即将公布，该方法旨在通过实证结果证明其有效性。

- **AI 生成图像中的伪影**：讨论了对 Mobius 和其他模型中“高对比度外观”和伪影的批评，并与 MJv6 等早期 AI 模型进行了对比。成员们指出了潜空间噪声（latent noise）问题以及不同模型的视觉特征。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/DataPlusEngine/status/1793817514956259460">来自 DataVoid e/acc (@DataPlusEngine) 的推文</a>：我们即将发表的论文概述并实现了在无需从头开始大规模预训练新模型的情况下，创建全新的基础 Diffusion 模型。我们可以通过受控的方式，破坏所有的质量...</li><li><a href="https://github.com/rohitgandikota/erasing/tree/main">GitHub - rohitgandikota/erasing: 从 Diffusion 模型中擦除概念</a>：从 Diffusion 模型中擦除概念。通过创建账号为 rohitgandikota/erasing 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1243304890189480038)** (11 messages🔥): 

- **Anthropic 发布关于 Claude 的研究论文**：一位成员分享了来自 Anthropic 的[一篇重大新研究论文](https://www.anthropic.com/research/mapping-mind-language-model)，内容关于解释大语言模型（LLM），他们绘制了 Claude 3 Sonnet 的内部工作原理。论文强调了识别和调整特定概念激活（如金门大桥）的能力。
- **关于 AI 作为广告产品的辩论**：一位成员质疑公司利用 AI 概念激活作为广告产品的可能性，引发了幽默的回应并链接了 [X 上的示例](https://x.com/PhilipKung5/status/1793743323124941157/photo/1)。另一位成员感叹此类发展的必然性令人抓狂。
- **对 AI 模型进展的反思**：一位成员回忆了早期在 Inception v1 模型上的 AI 视觉工作及其向当今复杂模型的演变。他们评论了具有幻觉色彩的 DeepDream 在学习神经元和电路操纵方面的历史重要性。
- **关于神经网络稀疏性的讨论**：一位成员解释了稀疏自编码器（sparse autoencoder）的架构和训练方法，强调使用 L1 范数强制（L1 norm enforcement）来保持稀疏性。他们指出，高维中间层平均通常只有约 300 个非零维度。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.13817">热力学自然梯度下降 (Thermodynamic Natural Gradient Descent)</a>：二阶训练方法比梯度下降具有更好的收敛性，但由于计算开销大，在实际的大规模训练中很少使用。这可以被视为...</li><li><a href="https://x.com/PhilipKung5/status/1793743323124941157/photo/1">来自 Philip Kung (@PhilipKung5) 的推文</a>：谢谢金门大桥版 Claude 😂😂😂</li><li><a href="https://www.anthropic.com/news/golden-gate-claude">金门大桥版 Claude (Golden Gate Claude)</a>：当我们调高“金门大桥”特征的强度时，Claude 的回答开始集中在金门大桥上。在短时间内，我们将向所有人开放此模型以供探索...
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1243298974761226392)** (3 条消息): 

- **LlamaIndex 见面会名额所剩无几**：“周二的见面会仅剩几个名额，请抓紧时间报名！”[在此获取最新动态](https://twitter.com/llama_index/status/1793739449127583964)。
- **使用 LlamaIndex 和 MultiOn 自动化任务**：“MultiOn 是一个 AI Agent 平台，它通过 Chrome 浏览器连接互联网并代表你执行操作，从而在 Web 上完成实际任务。”点击[此处](https://twitter.com/llama_index/status/1793764970024570979)查看演示。
- **介绍 RAGApp - RAG 聊天机器人的无代码界面**：“一个易于在任何云基础设施中部署的 Docker 容器，且完全开源。”在此轻松配置你的 LLM 模型提供商：[链接](https://twitter.com/llama_index/status/1794030544415818062)。
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1243298384526442609)** (60 条消息🔥🔥): 

- **LlamaParse 成为 PDF 提取解决方案**：用户推荐使用 **LlamaParse** 从带有表格和字段的 PDF 中提取数据，认为它是该任务合适的开箱即用 API。 [LlamaParse](https://link.to) 支持通过 GPT-4o 进行提取。

- **知识图谱索引建议**：讨论涉及了索引包含指向其他页面链接的知识库时的挑战，建议为 `KnowledgeGraphIndex` 手动创建三元组（triplet），同时考虑使用 `VectorStoreIndex` 以提高效率。 

- **LlamaIndex 集成说明**：参与者分享了在本地安装包含所有必要包的 LlamaIndex 时的困惑，特别是 **LLM OpenAI** 组件，建议清除缓存并确保正确的目录结构。

- **LLM 中的 Pydantic 解析问题**：用户在响应解析过程中遇到了 Pydantic 模型错误，建议为字段添加更好的描述，并改进 **GPT-4o** 的输入解析。该问题指向 LLM 无法正确解释输出类。

- **用于发票处理的更好模型**：建议查看 **HuggingFace MTEB 排行榜**以寻找更优的 Embedding 模型，特别提到了 **BGE**、**Nomic** 和 **GTE** 模型，用于处理发票和 PDF 聊天等任务。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://errors.pydantic.dev/2.7/v/missing">重定向中...</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index">GitHub - run-llama/llama_index: LlamaIndex 是一个用于 LLM 应用程序的数据框架</a>：LlamaIndex 是一个用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/">Query Engine - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1243588143287111812)** (4 条消息): 

- **Andy Singal 展示 PostgresML 与 LlamaIndex 结合的威力**：分享了一篇由 Andy Singal 撰写的名为[《通过 LlamaIndex 集成释放 PostgresML 的力量》](https://medium.com/ai-advances/unleashing-the-power-of-postgresml-with-llamaindex-integration-9eadee223939)的 Medium 文章。**jerryjliu0** 认为这篇文章很棒并给予了称赞，Andy Singal 对此表示感谢。
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1243647477882552420)** (1 条消息): 

- **新 AI 模型提醒：Phi-3 Medium 128k Instruct**：OpenRouter 宣布发布 **Phi-3 Medium 128k Instruct** 模型。用户可以查看[标准变体](https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct)和[免费变体](https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct:free)，并在此处[参与讨论](https://discord.com/channels/1091220969173028894/1232344285484023839)以分享对其性能和适用性的反馈。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct>)">Microsoft 的 Phi-3 Medium Instruct | OpenRouter</a>：Phi-3 Medium 是一个强大的 140 亿参数模型，专为高级语言理解、推理和指令遵循而设计。通过监督微调和偏好调整进行了优化...</li><li><a href="https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct:free>)">Microsoft 的 Phi-3 Medium Instruct | OpenRouter</a>：Phi-3 Medium 是一个强大的 140 亿参数模型，专为高级语言理解、推理和指令遵循而设计。通过监督微调和偏好调整进行了优化...
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1243302257999745124)** (41 条消息🔥): 

- **Wizard 模型性能提升**：成员们注意到 **wizard model** 的响应质量显著提高，等待时间减少，且回答更具创意。一位用户强调：*“你仍然需要盯着它（babysit）以避免段落重复，但除此之外，它表现得相当不错。”* 
- **Phi-3 Vision 引起关注**：讨论围绕 **Phi-3 Vision** 的能力展开，用户分享了测试链接如 [Phi-3 Vision](https://ai.azure.com/explore/models/Phi-3-vision-128k-instruct/version/1/registry/azureml)，并提到其与其他模型结合的潜力。另一个模型 **CogVLM2** 也被推荐用于视觉任务，可在 Hugging Face 的 [CogVLM-CogAgent](https://huggingface.co/spaces/THUDM/CogVLM-CogAgent) 找到。
- **Llama 3 模型 Prompt 格式说明**：成员们澄清，**Llama 3** 模型的 prompts 会由 OpenRouter 的 API 自动转换，无需手动格式化。手动提交 prompt 也是一种选择，可以使用 `prompt` 参数和 completions endpoint，而不是 chat/completions。
- **Llama 3 参数更新**：由于最近修复的一个 bug，**Llama 3 models** 的最佳参数即将更新。根据[团队回复](https://discord.com/channels/1091220969173028894/1092729520181739581/1243232269397655637)，此更新将在约 48 小时内推送。
- **Google Gemini API 问题与限制**：用户对 **Gemini FLASH** 在消耗大量 token 的情况下仍返回空白输出表示沮丧。已确认这是模型端的问题，讨论还强调了 Google 新的每日 API 使用限制，引发了对 OpenRouter 上 Gemini 使用量增加的好奇。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://ai.azure.com/explore/models/Phi-3-vision-128k-instruct/version/1/registry/azureml">Azure AI Studio</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/THUDM/CogVLM-CogAgent">CogVLM - a Hugging Face Space by THUDM</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1243280986507313203)** (36 messages🔥): 

- **Hugging Face 排行榜博客文章分享**：分享了由负责 HF OSS 排行榜的 Clementine 撰写的一篇文章。该文章深入探讨了 LLM 评估实践以及排行榜和非回归测试（non-regression testing）的重要性 ([Hugging Face blog](https://huggingface.co/blog/clefourrier/llm-evaluation))。

- **网站投毒对 Google 的 AI Overviews 有效**：链接指向了 Mark Riedl 的一项发现，关于一种影响 Google AI Overviews 的网站投毒（website poisoning）攻击 ([X post](https://x.com/mark_riedl/status/1793375699967054334))。这引发了关于使用自定义搜索引擎浏览器绕过来避免此类问题的进一步讨论。

- **Thomas Dohmke 关于 AI 编程的 TED 演讲**：成员们讨论了 [Thomas Dohmke 的 TED 演讲](https://youtu.be/nv9WwHpOKEg?si=mVApo6UnrtJ9ExH6)，内容涉及 AI 如何降低编程门槛。大家对其目前的可靠性看法不一，但承认 UX 的改进使得问题的快速变通处理（workarounds）变得更加容易。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/tensorlake/status/1793693325180150146">来自 Tensorlake (@tensorlake) 的推文</a>: 我们非常激动地宣布 @tensorlake 的开源实时数据框架 Indexify。它适用于任何 LLM 栈，并为引入您的数据提供了基础构建模块...</li><li><a href="https://huggingface.co/blog/clefourrier/llm-evaluation">聊聊 LLM 评估</a>: 未找到描述</li><li><a href="https://x.com/cupiabart/status/1793930355617259811?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Bartłomiej Cupiał (@CupiaBart) 的推文</a>: 这是我 CS 生涯中遇到的最奇怪的 bug。我和 @maciejwolczyk 一直在训练一个学习如何玩 NetHack 的神经网络，这是一款古老的 rog...</li><li><a href="https://x.com/jxnlco/status/1793800023689338921?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 jason liu (@jxnlco) 的推文</a>: 这是我对 RAG 发展方向的预测。在这个视频中，我谈到了 - RAG 从问答系统向报告生成工具的转变 - 精心设计的模板和 SOPs 的重要性...</li><li><a href="https://news.ycombinator.com/item?id=40458923">Show HN: 用于 LLM 应用的开源实时数据框架 | Hacker News</a>: 未找到描述</li><li><a href="https://x.com/mark_riedl/status/1793375699967054334?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Mark Riedl (@mark_riedl) 的推文</a>: 是的！我的网站投毒攻击对 Google 新的由 LLM 驱动的 AI Overviews 有效！</li><li><a href="https://youtu.be/nv9WwHpOKEg?si=mVApo6UnrtJ9ExH6">有了 AI，现在人人都能成为程序员 | Thomas Dohmke | TED</a>: 如果你只需大声说话就能编程会怎样？GitHub CEO Thomas Dohmke 展示了由于 AI 的存在，编程的准入门槛正在迅速消失 —— 这是一个...</li><li><a href="https://x.com/mark_riedl/status/1793375699967054334?">来自 Mark Riedl (@mark_riedl) 的推文</a>: 是的！我的网站投毒攻击对 Google 新的由 LLM 驱动的 AI Overviews 有效！</li><li><a href="https://x.com/nathanlands/status/1793925460801581300?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Nathan Lands — Lore.com (@NathanLands) 的推文</a>: 11)
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1243325518292451490)** (1 messages): 

- **World's Fair 多样性奖学金开放申请**：难以负担 AI Engineer World's Fair 门票的人员可以申请多样性奖学金，该奖学金为 6 月 25 日至 27 日在旧金山举行的活动提供免费或折扣门票。申请应包含“简明但具体的论文问题回答”，可以在[此处](https://docs.google.com/forms/d/e/1FAIpQLScff_RUv-fIKfdj_2HcHtk96iy45GD0BWLByGxqdBqvcepDHg/viewform?usp=sf_link)申请。

**提及的链接**：<a href="https://docs.google.com/forms/d/e/1FAIpQLScff_RUv-fIKfdj_2HcHtk96iy45GD0BWLByGxqdBqvcepDHg/viewform?usp=sf_link">多样性计划 - AI Engineer World's Fair 2024 年 6 月</a>：AI Engineer World's Fair 致力于帮助希望参加我们活动的代表性不足的少数群体。我们坚信多元化参会的价值。我们知道...

  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1243278784501645474)** (27 messages🔥): 

- **无需信用卡的税务发票**：Nathan Lambert 提到一个奇怪的情况，一个平台在没有存档信用卡的情况下向他发送了税务发票。在了解了关于转售证书（resale certificates）的细节后，他认为这个流程是合理的。
  
- **聚焦金门大桥的 AI**：小组对 [Anthropic AI 的实验](https://x.com/anthropicai/status/1793741051867615494?s=46)非常感兴趣，该实验展示了通过改变 AI 的内部特征（internal features）使其高度关注金门大桥。这促成了 "Golden Gate Claude" 的诞生，用户可以在 claude.ai 进行公开互动。
  
- **Google 的公关惨败**：成员们讨论了 Google 的产品流水线问题似乎导致了反复的公开失败，例如反响不佳的 AI 发布。对话强调了对内部反馈未被采纳以及在推出次级模型时存在疏忽的担忧。
  
- **对 AI 数据集指控的回应**：Philpax 分享的一个链接反驳了关于 Google AI 数据集的指控，特别是[否认依赖 LAION-5B](https://x.com/giffmana/status/1793906145310228538)。Google 的 AI 团队强调，他们拥有更优越的内部数据集用于研究。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/anthropicai/status/1793741051867615494?s=46">Anthropic (@AnthropicAI) 的推文</a>：本周，我们展示了改变 AI 模型 Claude 的内部“特征”如何改变其行为。我们发现了一个能让 Claude 极度关注金门大桥的特征。现在，为了...</li><li><a href="https://x.com/giffmana/status/1793906145310228538">Lucas Beyer (bl16) (@giffmana) 的推文</a>：以防万一这还不明显：这个答案是一个荒谬的幻觉。也许是因为“Google 的 AI 数据集”根本不存在。我们没有接触 laion5b，甚至在研究中也没有。我们不需要，我...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[lectures-and-projects](https://discord.com/channels/1179127597926469703/1223784028428177510/1243582104923541555)** (2 messages): 

- **提供高级 CS 讲座幻灯片**：Nathan Lambert 分享了一个基于 CS224N 材料的 CS25N 讲座更高级版本的链接。幻灯片可以点击[这里](https://docs.google.com/presentation/d/1on5xTePaUYg47vui3dUr0Lp6GUXmOXmhceNJLXRbGsE/edit)访问。

- **未来录像公告**：Nathan Lambert 提到该环节的录像最终会发布。目前尚未提供具体的发布日期。

**提到的链接**：<a href="https://docs.google.com/presentation/d/1on5xTePaUYg47vui3dUr0Lp6GUXmOXmhceNJLXRbGsE/edit">[2024年5月21日] Life after DPO (for alignment)</a>：Life after DPO Nathan Lambert || Allen Institute for AI || @natolambert Stanford CS224N: Natural Language Processing with Deep Learning 2024年5月21日

  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1243294013327413248)** (17 messages🔥): 

- **cmdr 模型关于 GQA 的困惑**：成员们正在澄清 "cmdr" 和 "cmdr+" 模型是否具有 **Grouped Query Attention (GQA)**。一位成员确认，“cmdr+ 有 GQA，非 + 版本没有”，并展示了每个版本的不同规格。
- **VRAM 扩展讨论**：讨论了 **GQA** 的存在与否如何影响 VRAM 使用。一位用户提到，“GQA 比指数级好，但不是线性的……它只是扩展性更好。”
- **样本打包效率提升**：成员们强调了 GitHub 上的一个新 PR，指出“样本打包（sample packing）带来了 3-4% 的效率提升”。这[链接到了一个 PR](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1619)，由 Dave Sescleifer 提交。

**提到的链接**：<a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1619">切换到并行 FFD 箱装算法。由 winglian 提交 · Pull Request #1619 · OpenAccess-AI-Collective/axolotl</a>：增加对分布式上下文打包的支持。重新加入打包效率估算。参见 @dsesclei 的 #1516。尝试将原始 PR 变基（rebase）到最新的 main 分支并不太顺利。我...

  

---

### **OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1243308936505852005)** (3 messages): 

- **期刊论文发表**：一位成员分享了他们合著的一篇[期刊论文](https://doi.org/10.1093/jamia/ocae120)，现已发表在 Journal of the American Medical Informatics Association 上。他们提到了自己所属的 **Université catholique de Louvain** 以及论文的其他贡献者。

- **祝贺涌至**：另一位成员对作者的发表表示祝贺，并附上了友好的“congrats 🙂”。这展示了社区对作者成就的支持和庆祝。

**提到的链接**：<a href="https://doi.org/10.1093/jamia/ocae120">Impact of high-quality, mixed-domain data on the performance of medical language models</a>：AbstractObjective。为了优化用于医疗应用的 LLM 训练策略，重点是创建临床相关的系统...

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1243389475254566934)** (8 messages🔥): 

- **SB-1047 引发愤怒**：成员们讨论了对 SB-1047 的担忧，他们认为这是试图在 OpenAI 等大玩家之间集中 AI 治理。一位成员称其为“异想天开、糟糕透顶的垃圾”，并将其与大制药公司（Big Pharma）和能源部门的监管俘获（regulatory capture）相类比，认为这不利于预算紧张的小型开发者。 
- **分享 Perplexity AI 搜索链接**：一位成员分享了关于 SB-1047 的 [Perplexity AI 搜索](https://www.perplexity.ai/search/SB-1047-Senate-2kZmFYHoTxe.rWUYat4B2A)链接。聊天中没有提供关于搜索细节的进一步信息或背景。
- **Arc Browser 的 Call Arc 功能受到称赞**：Arc Browser 的新功能“Call Arc”因其简单实用而受到关注。该成员称赞它允许用户毫不费力地“要求你的浏览器为你寻找并收集相关的答案”，并分享了[更多细节的链接](https://arc.net/e/C56904FA-1C75-4D77-9A87-E7F1A52529CD)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arc.net/e/C56904FA-1C75-4D77-9A87-E7F1A52529CD">1.44.1 Release</a>：</li><li><a href="https://g.co/gemini/share/a36c7ad84489">‎Gemini - SB 1047: Stifling Open-Source AI Innovation?</a>：由 Gemini Advanced 创建
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1243468923203227658)** (5 messages): 

- **用户面临 Typer 安装问题**：一位用户表示 *"queuelabs: pip install typer does not resolve"*，表明他们在尝试使用 **pip** 安装 **Typer** 库时遇到困难。
- **Poetry 设置问题困扰用户**：另一位用户询问 *"Did you run poetry install before poetry run 01? Are you running in a virtual environment,"* 指出了设置过程中可能遗漏的步骤。
  

---

### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1243385149178118216)** (9 messages🔥): 

- **Twinny + LM Studio 作为本地 co-pilot 表现惊人**：一位用户分享了将 [Twinny](https://github.com/rjmacarthy/twinny) 与 LM Studio 结合作为本地 co-pilot 替代方案的积极体验。他们询问了通过 llamafiles 运行此设置的问题，并得到确认，通过为不同实例分配不同端口，可以同时运行两个 llamafiles。

- **llama.cpp 端点嵌入图像的困惑已解决**：一位成员询问 llamafile/llama.cpp 服务器是否支持 llava embeddings 中的图像，并分享了一个未按预期工作的命令。他们随后澄清，`/v1/embeddings` 端点不接受 `image_data`，但使用 `/embedding` 端点可以按预期工作。

- **使用 llamafile 运行 continue.dev 的性能问题**：另一位用户报告了使用 llamafile 运行 continue.dev 的情况，指出在 Mac M2 上运行缓慢，但在较旧的 Nvidia GPU 上稍快一些。

- **关于构建和训练自定义 LLM 的咨询**：一位成员就使用公司文档构建和训练用于内部使用的自定义 LLM 寻求建议。他们收到了使用 [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) 进行训练的建议，并指出 llamafile 仅支持 inference。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/transformers/index">🤗 Transformers</a>: no description found</li><li><a href="https://github.com/rjmacarthy/twinny">GitHub - rjmacarthy/twinny: The most no-nonsense, locally or API-hosted AI code completion plugin for Visual Studio Code - like GitHub Copilot but completely free and 100% private.</a>: The most no-nonsense, locally or API-hosted AI code completion plugin for Visual Studio Code - like GitHub Copilot but completely free and 100% private. - rjmacarthy/twinny</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/4681">Allow server to generate multimodal embeddings via the `/embedding` endpoint by kseth · Pull Request #4681 · ggerganov/llama.cpp</a>: The server already exposes multimodal support in /completion and other places, but not in /embedding. The change for this is relatively straightforward, if a user submits in image_data to the /embe...
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1243302339301871668)** (8 messages🔥): 

- **用户感谢团队**：针对之前的互动表示“THANK YOU!”。
  
- **关于 104B 模型的查询**：一位用户询问团队是否计划发布其模型系列的 **104B 版本**。

- **Langchain 集成问题**：一位成员询问了关于在 Cohere 中使用 **Langchain 集成** 的当前状态和建议。

- **Aya 模型大小澄清**：一位用户询问 Playground 上的 **Aya 模型** 是 8B 还是 35B 版本。

- **Compressor 的校验错误**：分享了一个关于 **ContextualCompressionRetriever** 因抽象方法导致的 `ValidationError` 问题。

- **“56 根香蕉等于 1 个苹果”的计算**：使用 **CMR+** 探讨了一个计算问题：*“1 个苹果 = 2 个梨，3 个梨 = 4 个橙子，6 个橙子 = 7 根香蕉”*，结论是“56 根香蕉等于 1 个苹果”。

- **403 Forbidden 错误排查**：一位用户报告称，尽管使用了正确的生产环境 API Key，仍出现 **403 Forbidden 错误**。
  

---

### **AI Stack Devs (Yoko Li) ▷ #[late-night-lounge](https://discord.com/channels/1122748573000409160/1159342774710186075/1243425073017131112)** (6 messages): 

- **AI 生成的单口喜剧（Standup comedy）效果出奇地好**：一位用户分享了一个链接，对 AI 生成的单口喜剧质量表示惊讶。他们对其表现印象深刻。

- **探索 Ud.io 应用**：另一位用户询问提到的应用 [Ud.io](https://www.udio.com/songs/vsNF2nbsy646jGt348mdFG) 是否只做喜剧。这一询问表明了对该应用完整功能的后续好奇。

- **在 Suno 上转换音频**：一位成员分享了一个使用 [Suno](https://suno.com/song/e6b62587-4345-44fb-85c7-c51f932df655) 制作的更“魔性”版本的原始音频。这突显了该平台在修改声音方面的多样性。

- **对学习音频处理（Audio Manipulation）感兴趣**：一位用户表示有兴趣学习如何创建类似于分享的音频修改。这表明了学习音频工程或 AI 驱动的声音处理技能的愿望。 

- **简短回应**：简而言之，一位用户对一个查询回应了简短的“No”，表示不感兴趣或否定之前的陈述。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.udio.com/songs/vsNF2nbsy646jGt348mdFG">csimpkins - Standup Comedy on AI Generated Music | Udio</a>: 在 Udio 上听 csimpkins 的 AI 生成音乐单口喜剧。发现、创建并与世界分享音乐。使用最新技术在几秒钟内创建 AI 音乐。</li><li><a href="https://suno.com/song/e6b62587-4345-44fb-85c7-c51f932df655">AI Standup Comedy on AI Generated Musicby by @unwaveringplugin464 | Suno</a>: 单口喜剧演员在喜剧表演歌曲中演出。在 Suno 上聆听并制作你自己的作品。
</li>
</ul>

</div>
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1243281449352691742)** (1 messages): 

- **成员寻求 Google Calendar 集成以进行活动追踪**：一位成员询问是否可以将活动日历导入 Google Calendar，以避免错过活动。他们用一个悲伤的表情表达了担忧，表明需要一种更高效的方式来跟踪预定的活动。
  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/)** (1 messages): 

evelynciara: 是的，我很高兴这个频道存在 😅
  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/)** (1 messages): 

datarevised: https://x.com/DataPlusEngine/status/1793803117642854732
  

---



---



---



---



---



---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1243653978135203983)** (45 messages🔥): 

- **强调基准测试（Benchmarking）的重要性**：成员们分享了一篇讨论评估基准（evals）及其在推动研究突破中重要作用的博客文章，提到了成功的评估标准如 [GLUE](https://arxiv.org/abs/1804.07461)、[MMLU](https://arxiv.org/abs/2009.03300) 和 [GSM8K](https://arxiv.org/abs/2110.14168)（[来源](https://www.jasonwei.net/blog/evals)）。还指出了针对动态任务的“基于执行的评估”（execution based evals）所面临的挑战。

- **GPT-5 传闻引发兴奋**：相关演讲的链接表明 **GPT-5** 可能会在 2024 年问世，其智能将有显著飞跃，并从 Token 预测（token prediction）转向“Agent 架构”（agent architecture）（[链接](https://x.com/rohanpaul_ai/status/1793956355897724973?s=46&t=90xQ8sGy63D2OtiaoGJuww)）。

- **NumPy 2.0 发布让开发者感到兴奋**：成员们开玩笑地讨论了即将发布的 **NumPy 2.0** 及其对依赖管理的影响，消息通过 [Twitter](https://x.com/cgarciae88/status/1794019900119236874) 分享。

- **关于音频笔记应用的辩论**：一位成员强调了 **AudioPen**，它使用 OpenAI 的 API 将语音转换为文本；而另一位成员则推广了自己的应用，具有 iCloud 同步和 Markdown 支持等高级功能（[来源](https://techcrunch.com/2023/07/03/audio-pen-is-a-great-web-app-for-converting-your-voice-into-text-notes/)）。

- **xAI 获得巨额融资**：**xAI** 宣布了 60 亿美元的 B 轮融资，旨在快速提升其模型能力，并与 OpenAI 的估值进行了对比（[链接](https://x.ai/blog/series-b)）。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/cgarciae88/status/1794019900119236874">Cristian Garcia (@cgarciae88) 的推文</a>：NumPy 2.0 即将到来</li><li><a href="https://x.com/rohanpaul_ai/status/1793956355897724973?s=46&t=90xQ8sGy63D2OtiaoGJuw">Rohan Paul (@rohanpaul_ai) 的推文</a>：OpenAI Vivatech 巴黎演讲中关于 GPT-5 的新线索。线索出现在 18:00，他展示了 2024 年推出的 "GPT-Next" 智能图表 - 他谈到该模型是一个 "..."</li><li><a href="https://x.com/swizec/status/865295043162021889">Swizec Teller (@Swizec) 的推文</a>：@aJimHolmes @QualityFrog 我们真正需要的是 AI。我们编写测试，AI 构建通过测试的代码。真正的测试驱动开发！</li><li><a href="https://x.com/levelsio/status/1795003725766877419?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">@levelsio (@levelsio) 的推文</a>：@elonmusk 的 xAI 刚刚筹集了 60 亿美元。不是估值 60 亿美元，不，他们筹集了 60 亿美元！显然是有史以来规模最大的投资轮次。估值为 240 亿美元，Open ...</li><li><a href="https://www.latent.space/p/9bc44202-cad7-4124-8cfa-5cf7921cf556">Latent Space</a>：AI Engineer 通讯 + 美国前 10 大技术播客。探索 AI UX、Agents、Devtools、Infra、开源模型。查看 https://latent.space/about 了解来自 Chris Lattner、Andrej Karpathy、Ge... 的精彩内容。</li><li><a href="https://techcrunch.com/2023/07/03/audio-pen-is-a-great-web-app-for-converting-your-voice-into-text-notes/">AudioPen 是一款出色的 Web 应用，可将您的语音转换为文本笔记 | TechCrunch</a>：开发者 Louis Pereira 的 AudioPen Web 应用专注于使用 OpenAI APIs 将您的语音转换为整洁的文本笔记。</li><li><a href="https://www.jasonwei.net/blog/evals">成功的语言模型评估 — Jason Wei</a>：每个人都在使用评估基准（“evals”），但我认为它们值得比目前更多的关注。Evals 是研究社区的激励因素，而突破往往与...</li><li><a href="https://x.com/rohanpaul_ai/status/1793956355897724973?s=46&t=90xQ8sGy63D2OtiaoGJuww">Rohan Paul (@rohanpaul_ai) 的推文</a>：OpenAI Vivatech 巴黎演讲中关于 GPT-5 的新线索。线索出现在 18:00，他展示了 2024 年推出的 "GPT-Next" 智能图表 - 他谈到该模型是一个 "..."</li><li><a href="https://x.ai/blog/series-b">B 轮融资</a>：未找到描述</li><li><a href="https://x.com/xyz3va/status/1794413898608632004">xyzeva (@xyz3va) 的推文</a>：嗨 @rabbit_hmi，我想我们应该谈谈你违反 Android 许可、审查我们的研究、对你的社区撒谎的问题。线程</li><li><a href="https://x.com/yi_ding/status/1793756026329870704?s=46&t=90xQ8sGy63D2OtiaoGJuww">Yi Ding -- prod/acc (@yi_ding) 的推文</a>：OpenAI 声称他们拥有 300 万个开发者账户。粗略估计全球有 2700 万软件开发者，这意味着有两位数百分比的人注册了 OpenAI。</li><li><a href="https://x.com/suno_ai_/status/1794367911408353349?s=12&t=XvL7HEPFiF7scRI6BJ2tzQ">Suno (@suno_ai_) 的推文</a>：从任何声音制作歌曲。即将推出 🎧 第一卷：一个浇水壶，但做成迷幻摇滚风格</li><li><a href="https://x.com/tantacrul/status/1794863603964891567?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tantacrul (@Tantacrul) 的推文</a>：1. 我对 @Meta 新通知的设计感到非常震惊，该通知告知我们他们想使用我们发布的内容来训练他们的 AI 模型。它的设计故意非常尴尬...</li><li><a href="https://x.com/dchaplot/status/1794109043348209969?s=46&t=90xQ8sGy63D2OtiaoGJuww">Devendra Chaplot @ ICLR 2024 (@dchaplot) 的推文</a>：我们刚刚发布了 mistral-finetune，这是关于如何使用 LoRA 微调 Mistral 开源模型的官方仓库和指南：https://github.com/mistralai/mistral-finetune。还发布了 Mistral-7B-Instru...</li><li><a href="https://apps.apple.com/us/app/brain-dump-voice-memo-notes/id6473446030">‎记录与转录：BrainDump</a>：‎Brain Dump 是您的个人语音转文本记事本，旨在精确捕捉和完善您的想法。无论您是在进行头脑风暴、记录语音备忘录还是听写备忘录，Brain...</li><li><a href="https://open.substack.com/pub/swyx/p/iclr-2024-recap">ICLR 2024 — 最佳论文与演讲（图像生成、视觉、Transformers、状态空间模型），特邀 Christian Szegedy、Ilya Sutskever、Durk Kingma</a>：在 2024 年 ICLR 会议展示的 2260 篇论文中，精选了 14 篇最佳论文，分为 4 个部分，涵盖图像生成、视觉学习、扩展 Transformers 和状态空间模型。</li><li><a href="https://open.substack.com/pub/swyx/p/iclr-2024-recap?r=1h4isl&utm_campaign=post&utm_medium=web">ICLR 2024 — 最佳论文与演讲（图像生成、视觉、Transformers、状态空间模型），特邀 Christian Szegedy、Ilya Sutskever、Durk Kingma</a>：在 2260 篇论文中精选的 14 篇最佳论文</li>

在 2024 ICLR 会议上展示，分为 4 个部分，涵盖 Image Generation、Vision Learning、Extending Transformers 和 State Space Models。
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1244758635419009045)** (1 messages): 

- **新播客剧集：ICLR 回顾第一部分**：Latent Space 发布了新播客剧集，重点关注 [ICLR 2024](https://x.com/latentspacepod/status/1795196817044594817) 最佳论文的亮点。主题包括 **ImageGen、Compression、Adversarial Attacks、Vision Learning and Weak Supervision、Extending Transformers and Attention** 以及 **State Space Models vs Transformers**。

**提到的链接**：<a href="https://x.com/latentspacepod/status/1795196817044594817">来自 Latent Space Podcast (@latentspacepod) 的推文</a>：🆕 ICLR 2024：最佳论文（第一部分） 我们按主题介绍了我们挑选的优秀论文和演讲，供 AI Engineers 追踪：A 部分：ImageGen、Compression、Adversarial ...

  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1243654830749257810)** (241 messages🔥🔥): 

- **AI 创作的音乐引发版权问题**：围绕 "**Suno**" 等 AI 生成歌曲的讨论引发了对版权影响的询问。成员指出“AI 生成的艺术作品不受版权保护”，强调了目前法律上的模糊性。
- **Meta 的 Audiocraft 令人印象深刻**：一位成员称赞了 Meta 的 **Audiocraft** 的近实时能力，尽管可能涉及一些预处理。关于这是否是真正的实时存在争议。
- **音乐家的工具和插件**：小组探索了各种面向音乐家的 AI 工具，包括 **Neutone** 和 **moises.ai**，成员们分享了这些技术的工作流和经验。一位成员强调了将模型集成到插件中的“酷东西”，增强了音乐制作。
- **AI 在音乐中的法律和伦理影响**：针对音乐中 AI 的法律层面进行了激烈讨论，特别是关于使用权和冒充问题。在版权担忧中，一家公司的立场被引用为“不要使用我们的 IP 进行数据训练”。
- **开源与创作自由**：成员们对 **gary-backend-combined** 等开源项目表示兴奋。有人提出了一个“版权无关紧要”的平台想法，反映了社区对创作自由的渴望。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://redito.substack.com/p/surfing-human-creativity-with-ai">与 Nao Tokui (Neutone) 一起利用 AI 冲浪人类创意</a>：使用 AI 扩展人类创造力，AI 作为“心灵之镜”，text-to-audio 的局限性，为音乐家构建 AI 工具，以及 AI 时代艺术家的角色。</li><li><a href="https://suno.com/song/ccd7c687-b9ad-48a3-9636-2bf11d5c97a3">Latent Space by @unchainedmeter763 | Suno</a>：嘻哈电子歌曲。听听看并用 Suno 制作你自己的歌曲。</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0">AI In Action: Weekly Jam Sessions</a>：2024 主题、日期、主持人、资源、@dropdown、@ GenAI 的 UI/UX 模式，1/26/2024，nuvic，&lt;a href=&quot;https://maggieappleton.com/squish-structure&quot;&gt;https://maggieappleton.com/squish-stru...</li><li><a href="https://github.com/betweentwomidnights/gary-backend-combined">GitHub - betweentwomidnights/gary-backend-combined: combined backend for gary4live and gary-on-the-fly</a>：gary4live 和 gary-on-the-fly 的组合后端 - betweentwomidnights/gary-backend-combined
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1243657828623712356)** (26 messages🔥): 

- **VSCode extension 在 nightly 版本中的 Bug**：几位成员讨论了 **Mojo VSCode extension** 的问题，强调它仅对安装 nightly 版本后创建的文件才能正常工作。一位成员提到需要关闭并重新打开代码才能解决问题，而另一位成员发现该扩展自 24.3 版本以来一直存在 Bug，需要频繁重置。
- **比较 Python 和 Mojo 中的 torch 代码**：一位用户比较了 **Python** 和 **Mojo** 代码中的 `torch.randn` 操作和 `cuda()` 方法，指出 Mojo 似乎具有更好的 SASS 和开销表现。讨论集中在潜在的细微差别上，例如函数参数和内存管理，并对其影响进行了详细审查。
- **在 Ubuntu 上苦于 WSL 和 APT**：一位用户表达了对 **WSL** 和 Linux 软件包管理的沮丧，分享了尝试使用 `apt` 时产生的错误输出。由于这些问题以及不想在配置中处理 AI 集成，他们讨论了是否要切换到 Mac。
- **MAX Engine Python API 安装问题**：一位用户由于兼容性问题无法安装 **MAX Engine Python API**，收到了与分发包不匹配相关的错误。另一位用户建议检查 Python 版本，指出支持 3.8 到 3.11 版本，但不支持 3.12。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="http://archive.ubuntu.com/ubuntu">Index of /ubuntu</a>：未找到描述</li><li><a href="http://security.ubuntu.com/ubuntu">Index of /ubuntu</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: 来自 *Modular*:
<https://twitter.com/Modular/status/1794148687104647293>
  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1243940391766593637)** (4 messages): 

- **聊天机器人喜欢塔可 (tacos)**：成员们推测了聊天机器人的配置和个性。有人指出，“它似乎真的很喜欢塔可。”
  

---


### **Modular (Mojo 🔥) ▷ #[tech-news](https://discord.com/channels/1087530497313357884/1151359062789857280/1243718867889422416)** (6 messages): 

- **对 Mojo 性能对比手写 kernel 的好奇**：一位用户提到了 Futhark 的工具链问题，并询问 Mojo 计划如何与手写 kernel 竞争，并指出 Futhark 并没有尝试进行此类比较。他们还分享了自己在 CUDA C++ 和 Scala 方面的背景，表达了对函数式编程语言的欣赏。
  
- **Mojo 雄心勃勃的 C++ 互操作性目标**：一位用户承认 Mojo 与 C++ 互操作的目标听起来非常宏大，并询问该语言计划如何实现这一目标，而不仅仅是将 C 作为中间媒介。 

- **关于 Bend 默认并行性的澄清**：在回答有关语言设计和并行性的查询时，一位用户解释说，虽然大多数语言默认是串行执行的，但 Bend 的设计旨在自然地鼓励编写并行代码，而无需程序员付出额外努力。

- **xAI 融资 60 亿美元**：Elon Musk 的 AI 初创公司 xAI 在 B 轮融资中筹集了 60 亿美元，吸引了 Valor Equity Partners 和 Andreessen Horowitz 等主要投资者。据 [TechCrunch](https://techcrunch.com/2024/05/26/elon-musks-xai-raises-6b-from-valor-a16z-and-sequoia/) 报道，这笔资金使 xAI 能够与 OpenAI 和 Microsoft 等巨头展开激烈的竞争。

- **关于 AI 商业化的讨论**：一位用户强调 Tesla 的全自动驾驶技术是一个著名的大型商业用例，并询问 AI 总体上是如何实现商业化的。

**提到的链接**：<a href="https://techcrunch.com/2024/05/26/elon-musks-xai-raises-6b-from-valor-a16z-and-sequoia/">Elon Musk's xAI raises $6B from Valor, a16z, and Sequoia | TechCrunch</a>：Elon Musk 的 AI 初创公司 xAI 在新一轮融资中筹集了 60 亿美元，Musk 正在筹集资金以与对手展开激烈竞争...

  

---

### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1243645666383757313)** (133 条消息🔥🔥): 

- **理解 Python 与 Mojo 函数的区别**：成员们讨论了在 Mojo 中对函数进行哈希处理（hashing）相比 Python 的难度。一位成员解释道，*"Mojo 函数没有 [对象标识（object identity）]。"* 另一位成员强调，从动态类型到 Mojo 静态类型的转变可能是一个障碍。
- **在 Mojo 中初始化 Optional 类型**：成员们在将变量（特别是自定义结构体）正确初始化为 `None` 时遇到了挑战。大家提出了涉及 `UnsafePointer` 和 `Optional` 的多种解决方案，最终一位成员成功让代码无错运行。
- **在 Mojo 中创建链表**：关于如何在 Mojo 中正确实现和初始化 LinkedList 进行了深入讨论。成员们辩论了为 `next` 变量使用 `UnsafePointer` 和 `Optional` 类型的各种方法。
- **Mojo 的反射（Reflection）能力**：参与者对 Mojo 用于访问结构体数据的反射能力感到好奇。有人指出，*"MLIR 元编程在未来应该能实现这一点，"* 但目前尚未实现。
- **导入 Mojo 包进行测试**：在尝试将模块从主源目录导入测试目录时遇到了问题。解决方案包括确保 `__init__.mojo` 文件存在，尽管 `mojo test` 命令仍存在一些复杂情况。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/collections/optional/">optional | Modular 文档</a>：定义了 Optional，一种模拟可能存在或不存在的值的类型。</li><li><a href="https://github.com/saviorand/lightbug_http">GitHub - saviorand/lightbug_http: 简单且快速的 Mojo HTTP 框架！🔥</a>：简单且快速的 Mojo HTTP 框架！🔥。通过在 GitHub 上创建账户为 saviorand/lightbug_http 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/1405">[Modular CLI]: modular install mojo 应支持版本锁定 · Issue #1405 · modularml/mojo</a>：问题描述：我无法弄清楚如何安装特定版本的 mojo。这对于库维护者和 CI/CD 系统非常有用，甚至是必不可少的。复现步骤 $ modular inst...</li><li><a href="https://modul.ar/systems">路线图与已知问题 | Modular 文档</a>：MAX 平台已知问题和即将推出的功能的摘要。</li><li><a href="https://github.com/modularml/mojo/pull/2825">[stdlib] 为 List 添加可选的小缓冲区优化（Small Buffer Optimization），第 2 次尝试，由 gabrieldemarmiesse 提交 · Pull Request #2825 · modularml/mojo</a>：此 PR 解决了 #2467 的一部分。此 PR 是按以下顺序阅读和合并的三个 PR 之一：#2825 #2826 #2827。小缓冲区行为：我们使用 InlineArray 存储最多 small_buffer_size ...</li><li><a href="https://github.com/modularml/mojo/pull/2826">[stdlib] 绕过 #2825 中存在的实例化（materialization）Bug，由 gabrieldemarmiesse 提交 · Pull Request #2826 · modularml/mojo</a>：此 PR 解决了 #2467 的一部分。此 PR 是按以下顺序阅读和合并的三个 PR 之一：#2825 #2826 #2827。重要的差异在于此分支与 PR #28 分支之间...</li><li><a href="https://github.com/modularml/mojo/pull/2827">[stdlib] 通过在 List 中使用带有实例化变通方法的 SBO 为 String 添加 SSO，由 gabrieldemarmiesse 提交 · Pull Request #2827 · modularml/mojo</a>：此 PR 解决了 #2467 的一部分。此 PR 是按以下顺序阅读和合并的三个 PR 之一：#2825 #2826 #2827。有趣的部分是此 PR 与 PR #2826 之间的差异。你可以找到...
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1244564633466175538)** (22 messages🔥): 

- **紧凑型 CRC32 导致性能略微下降**：一位成员注意到他们在 Mojo 中尝试使 CRC32 计算更加紧凑，但导致性能提升从基准的约 14 倍下降到 11 倍。 [Benchmark 文件可以在此处找到](https://github.com/fnands/fnands.com/blob/main/blog/2024/mojo-crc-calc/crc.mojo)。

- **Unroll 装饰器弥补性能差距**：在建议使用 unroll 装饰器后，紧凑实现的性能得到提升，几乎消除了差距，紧凑版的加速比最高达到 1429 倍。

- **能效核心（Efficiency cores）表现出不同的性能**：在能效核心上的测试显示，紧凑版本始终略微优于非紧凑实现，不同核心类型的结果显示出明显的加速比差异。

- **基于 SIMD 的实现问题**：对 CRC32 的 SIMD 探索由于潜在的内存重用问题产生了非确定性结果，这指向了 Bug 或误用；即使是正确的变体，在 x86 硬件上仍被证明比原始实现慢。

- **缓存限制影响大字节情况**：对于较大的字节大小（例如 16、32 字节），紧凑版本面临性能下降，这可能是由于 L1 cache 限制导致的，这一点在 Intel i7-12700H 的 Benchmark 加速指标中有所体现。[Nightly 版本测试可在此处获取](https://github.com/fnands/fnands.com/blob/main/blog/2024/mojo-crc-calc/crcn.mojo)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/fnands/fnands.com/blob/main/blog/2024/mojo-crc-calc/crc.mojo">fnands.com/blog/2024/mojo-crc-calc/crc.mojo at main · fnands/fnands.com</a>：我的个人博客。通过在 GitHub 上创建账户为 fnands/fnands.com 的开发做出贡献。</li><li><a href="https://github.com/komrad36/CRC?tab=readme-ov-file">GitHub - komrad36/CRC: 适用于 x86、Intel 和 AMD 的最快 CRC32，以及各种方法的全面推导和讨论</a>：适用于 x86、Intel 和 AMD 的最快 CRC32，以及各种方法的全面推导和讨论 - komrad36/CRC</li><li><a href="https://github.com/fnands/fnands.com/blob/main/blog/2024/mojo-crc-calc/crcn.mojo">fnands.com/blog/2024/mojo-crc-calc/crcn.mojo at main · fnands/fnands.com</a>：我的个人博客。通过在 GitHub 上创建账户为 fnands/fnands.com 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1243790811645218876)** (75 messages🔥🔥): 

- **Mojo 编译器 Nightly 更新发布**：分享了多个 Mojo 编译器的 Nightly 版本，如 `2024.5.2505`、`2024.5.2514`、`2024.5.2605` 和 `2024.5.2705`。关键更新包括 LSP 中的局部变量重命名、为 `List` 引入列表排序和方法、将功能从 refitem 迁移到 getitem，以及添加了 `tempfile` 模块（更新日志见[此处](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)）。

- **讨论 String 和 List 的实现问题**：成员们讨论了当前 List 和 String 实现中的问题，建议通过各种修复来处理 List 中的空指针和标识（identity）问题。建议包括让默认的 List 构造函数调用 `alloc(0)`，并确保 String 正确跟踪空缓冲区。

- **使用新的 `ref` 语法自动解引用迭代器**：新的 `ref` 语法引发了关于其自动解引用迭代器潜力的讨论。`ref` 约定可能会带来更简单的实现，并减少函数调用中手动解引用的需求。

- **关于函数返回类型和变长参数的提案**：尝试在 Mojo 中实现 Python 的 `zip` 和 `unzip` 函数时，揭示了声明基于变长列表参数（Variadic Arguments）返回元组（Tuples）的函数所面临的挑战。目前的尝试出现了错误，并促使对正确函数声明的进一步调查。

- **最新编译器更新带来的测试挑战**：最新的编译器更新导致现有的 Pull Request 出现问题，导致测试中断并凸显了修复 Bug 的必要性。建议将日志记录作为解决不稳定 Bug（Flaky bugs）的临时措施，如更新后的 Pull Request [#2832](https://github.com/modularml/mojo/pull/2832) 所示。

**提到的链接**：<a href="https://github.com/modularml/mojo/pull/2832">[stdlib] 在 `test_reverse.mojo` 中添加一些日志以排查不稳定 Bug，由 gabrieldemarmiesse 提交 · Pull Request #2832 · modularml/mojo</a>：参见 #2369，这个 Bug 出现的频率越来越高。一些日志应该能帮助我们了解具体失败的原因。

  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1243647477882552420)** (3 条消息): 

- **Phi-3 Medium 128k Instruct 发布**：新模型 [microsoft/phi-3-medium-128k-instruct](https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct) 已上线，提供标准版和免费版。鼓励用户查看 [讨论帖](https://discord.com/channels/1091220969173028894/1232344285484023839) 以分享对其性能的反馈。
- **X 平台公告**：[@OpenRouterAI](https://x.com/OpenRouterAI/status/1794101495538843803) 宣布了新的免费模型 Phi-3 Medium，包含标准版和免费版。
- **新模型与降价**：Microsoft 的 [phi-3-mini-128k-instruct](https://openrouter.ai/models/microsoft/phi-3-mini-128k-instruct) 模型现已上线。此外，[llama-3-lumimaid-70b](https://openrouter.ai/models/neversleep/llama-3-lumimaid-70b) 模型大幅降价 57%。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1794101495538843803">来自 OpenRouter (@OpenRouterAI) 的推文</a>: 新的免费模型：Phi 3 Medium 🧠 microsoft/phi-3-medium-128k-instruct，包含标准版和免费版</li><li><a href="https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct>)">Phi-3 Medium Instruct (由 microsoft 提供) | OpenRouter</a>: Phi-3 Medium 是一个强大的 140 亿参数模型，专为高级语言理解、推理和指令遵循而设计。通过监督微调和偏好调整进行优化...</li><li><a href="https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct:free>)">Phi-3 Medium Instruct (由 microsoft 提供) | OpenRouter</a>: Phi-3 Medium 是一个强大的 140 亿参数模型，专为高级语言理解、推理和指令遵循而设计。通过监督微调和偏好调整进行优化...</li><li><a href="https://openrouter.ai/models/microsoft/phi-3-mini-128k-instruct>)">Phi-3 Mini Instruct (由 microsoft 提供) | OpenRouter</a>: Phi-3 Mini 是一个强大的 38 亿参数模型，专为高级语言理解、推理和指令遵循而设计。通过监督微调和偏好调整进行优化...</li><li><a href="https://openrouter.ai/models/neversleep/llama-3-lumimaid-70b>)">Llama 3 Lumimaid 70B (由 neversleep 提供) | OpenRouter</a>: NeverSleep 团队回归，带来了基于其精选角色扮演数据训练的 Llama 3 70B 微调模型。Lumimaid 在 eRP 和 RP 之间取得了平衡，旨在保持严肃的同时，在必要时不受审查...
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1244097547773939762)** (260 条消息🔥🔥): 

- **文档详解速率限制问题**：成员们讨论了 OpenRouter 中的速率限制（Rate Limiting），包括速率限制如何随额度（Credits）增加。引用了 [OpenRouter 文档](https://openrouter.ai/docs#rate-limits-and-credits-remaining)中的详细解释，明确了更高的信用余额允许更高的请求速率。

- **Modal Fallback 错误处理**：一名用户在拥有充足额度的情况下，使用 Modal Fallback 功能时遇到了速率限制错误。社区建议检查剩余的免费请求次数，并可能在达到限制时省略免费模型。

- **Claude 自我审查模型使用量下降**：用户推测 Claude 自我审查模型的使用量有所下降，理由是拒绝回答的情况增多以及护栏（Guardrails）更加严格。一些用户指出，这种变化使得模型变得不那么像人，而更倾向于公关辞令。

- **AI 模型托管成本对比**：对比了 RunPod、Vast.ai 等不同托管方案与 Google Cloud 和 Amazon Bedrock 等主流云服务商。用户强调，与传统云服务相比，替代平台上的 GPU 使用价格显著降低。

- **视觉模型的成本与性能**：讨论了各种视觉模型的性价比和性能，建议在特定任务中评估 Gemini 及其 OCR 能力。社区注意到 Gemini 的视觉服务具有良好的性能和极具竞争力的价格。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/1cx72of/chinese_compa">Reddit - 探索一切</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cx72of/chinese_companies_aim_to_use_price_advantages_to/">Reddit - 探索一切</a>：未找到描述</li><li><a href="https://huggingface.co/nvidia/NV-Embed-v1">nvidia/NV-Embed-v1 · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/bnS5n.gif">Dead Space GIF - Dead Space - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://orw.karleo.net/changes">OpenRouter API Watcher</a>：探索 OpenRouter 的模型列表和记录的变更。每小时更新一次。</li><li><a href="https://openrouter.ai/docs#rate-limits-and-credits-remaining">Docs | OpenRouter</a>：构建与模型无关的 AI 应用
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1243659762982457445)** (1 条消息): 

- **团队分享 CCS 后续项目**：2023 年春季，一个团队跟进了 Collin Burns 针对 ELK 的 *Contrast Consistent Search (CCS)* 方法。虽然实证结果未显示出可预测的泛化改进，但为了透明起见，他们在 [Quirky Models 基准测试](https://arxiv.org/abs/2312.01037)上分享了所提方法和结果。 
- **失败尝试的透明度**：EleutherAI 公开了他们的项目，承认该项目未能提供改进泛化的有力证据，强调了分享成功与失败经验的重要性。更多细节可以在他们的[博客文章](https://blog.eleuther.ai/vincs/)中找到。

**提到的链接**：<a href="https://blog.eleuther.ai/vincs/">VINC-S: Closed-form Optionally-supervised Knowledge Elicitation with Paraphrase Invariance</a>：记录 2023 年春季项目的研究结果

  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1243641744159932518)** (63 条消息🔥🔥): 

- **关于 LLM 的 RAG 与微调（Finetuning）的讨论**：一位成员询问，与微调相比，**RAG**（检索增强生成）是否是 **LLM** 从自定义库中提取信息最简单的方法。另一位成员建议使用 RAG，并解释说微调在向模型传授信息方面表现不佳，更多是关于教授风格。

- **NeoX 模型历史与计算资源**：讨论了 **NeoX 模型** 的起源及其开发者。提到计算细节已在 [Eleuther 博客](https://blog.eleuther.ai/year-one/)及相关论文中涵盖。

- **用于时间序列的 Transformers vs SSMs**：成员们讨论了为什么 **Transformers** 在**多变量时间序列**上表现可能不佳，而 **SSMs** 处理得更好。分享了一些参考资料，如 [Chronos-T5](https://huggingface.co/amazon/chronos-t5-large) 和 [Spacetimeformer](https://github.com/QData/spacetimeformer) 以提供见解。

- **ThePitbull 21.4B 模型发布**：Hugging Face 上推出了一款名为 **ThePitbull 21.4B** 的新模型，声称功能非常强大，几乎与 70B 模型相当。目前已有量化版本，社区对其性能持怀疑态度。

- **TRC 额度与区域锁定**：简要讨论了 **TRC 额度** 的可用性及其区域锁定问题。澄清了不可抢占/单主机额度正在被大量使用，可用区域为美国和欧洲，特定的区域限制使得访问变得困难。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/fblgit/UNA-ThePitbull-21.4-v1">fblgit/UNA-ThePitbull-21.4-v1 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/amazon/chronos-t5-large">amazon/chronos-t5-large · Hugging Face</a>：未找到描述</li><li><a href="https://blog.eleuther.ai/year-one/">What A Long, Strange Trip It&#39;s Been: EleutherAI One Year Retrospective</a>：EleutherAI 第一年的回顾。</li><li><a href="https://huggingface.co/saltlux/luxia-21.4b-alignment-v1.0/discussions/5#65fbd17bb67a26e59ed169de">saltlux/luxia-21.4b-alignment-v1.0 · 91.9 HellaSwag, 79.2 TruthfulQA... 表现很差。为什么要这样做？</a>：未找到描述</li><li><a href="https://github.com/QData/spacetimeformer">GitHub - QData/spacetimeformer: Multivariate Time Series Forecasting with efficient Transformers. Code for the paper &quot;Long-Range Transformers for Dynamic Spatiotemporal Forecasting.&quot;</a>：使用高效 Transformers 进行多变量时间序列预测。论文 "Long-Range Transformers for Dynamic Spatiotemporal Forecasting" 的代码。
</li>
</ul>

</div>

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1243667962162839744)** (58 条消息🔥🔥): 

- **梯度扰动技术探索**：成员们讨论了使用相同架构的多次初始化，通过扰动权重来获得权重更新的分布。*"如果我们能说初始化 X 产生的梯度与模型的某些定性特征相关，初始化 Y 产生另一些特征，然后以一种有原则的方式将它们结合起来，那就太酷了。"*

- **生成式 AI 与法律研讨会公告**：发布了在维也纳举行的 ICML ’24 第二届生成式 AI 与法律研讨会的公告，重点关注英国和欧盟的知识产权（IP）和隐私挑战。分享了投稿详情以及行业部署挑战和数据保护等关注主题（[研讨会网站](https://genlaw.org/2024-icml.html)）。

- **关于 Schedule-Free 优化的讨论**：重点介绍了一篇提出新型学习率调度方法的论文（[arxiv 链接](https://arxiv.org/abs/2405.15682)）。该方法声称通过统一调度和迭代平均（iterate averaging）而无需引入额外的超参数，其表现优于传统的调度方案。

- **Transformer 与隐式推理**：分享了一项关于 Transformer 在参数化知识上进行隐式推理能力的研究，提出了关于泛化以及通过延长训练实现隐式推理的有效性等观点（[arxiv 链接](https://arxiv.org/abs/2405.15071)）。讨论涉及了在组合能力和比较推理能力方面的局限性。

- **对学术主张的批评与怀疑**：一些成员对一篇 Schedule-Free 优化论文中的主张表示怀疑，原因是作者之前的宣传做法。人们对正面反馈与批评意见之间参与度的巨大差异表示担忧。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.13956">Attention as an RNN</a>：Transformer 的出现标志着序列建模的重大突破，提供了一种能够利用 GPU 并行性的高性能架构。然而，Transformer 在计算上...</li><li><a href="https://arxiv.org/abs/2405.15071">Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization</a>：我们研究了 Transformer 是否能学会对参数化知识进行隐式推理，这是即使是最强大的语言模型也难以掌握的技能。重点关注两种代表性的推理类型...</li><li><a href="https://x.com/BlinkDL_AI/status/1794746863901642866">BlinkDL (@BlinkDL_AI) 的推文</a>：最新（未见）数据上的 LLM 压缩排行榜：https://huggingface.co/spaces/Jellyfish042/UncheatableEval 更好的压缩 = 更好的基础模型。RWKV-6🐦 在仅经过训练的情况下极具竞争力...</li><li><a href="https://arxiv.org/abs/2405.15682">The Road Less Scheduled</a>：现有的不需要指定优化停止步数 T 的学习率调度，其表现远不如依赖于 T 的学习率调度。我们提出了一种方法...</li><li><a href="https://arxiv.org/abs/2405.15731">Understanding the differences in Foundation Models: Attention, State Space Models, and Recurrent Neural Networks</a>：Softmax Attention 是各种人工智能应用基础模型的核心支柱，但其在序列长度上的二次复杂度限制了其推理吞吐量...</li><li><a href="https://tenor.com/view/catholic-meryl-streep-nuns-sisters-i-have-doubts-gif-5743847">Catholic Meryl Streep GIF - Catholic Meryl Streep Nuns - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://arxiv.org/abs/2405.15722">Models That Prove Their Own Correctness</a>：我们如何信任学习模型在特定感兴趣输入上的正确性？模型准确率通常是在输入分布上进行“平均”测量的，无法为任何...提供保证。</li><li><a href="https://arxiv.org/abs/2405.13861">Transformers Learn Temporal Difference Methods for In-Context Reinforcement Learning</a>：上下文学习（In-context learning）是指模型在推理时间内无需调整参数的学习能力。模型（如 Transformer）的输入（即 prompt）由上下文...组成。</li><li><a href="https://icml.cc/virtual/2024/poster/32613">ICML Poster Transformers are SSMs: Generalized Models and Efficient Algorithms with Structured State Space Duality</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2405.13967">DeTox: Toxic Subspace Projection for Model Editing</a>：最近的对齐算法（如直接偏好优化 DPO）已被开发用于通过训练模型匹配人类行为来提高大型语言模型（LLMs）的安全性...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1244134857215447082)** (5 messages): 

- **对新版本的期待**：一名成员对即将发布的版本表示兴奋，询问“什么时候发布？”这表明对即将推出的数据集或工具具有浓厚兴趣。
- **对安全数据集的兴趣**：另一条消息表达了获取“安全数据集”的强烈兴趣。笑脸符号暗示了对这类数据集可用性的乐观态度。
- **小模型 vs. 量化后的更大模型**：一位成员询问了小模型在某些场景下优于量化后的更大模型的情况。“在我的评估中，量化模型即便不再体积庞大，其表现也始终更好”，这揭示了一个有趣的发现，即量化模型可能会保持或超越原有的性能。
- **Perplexity 与量化**：分享的链接讨论了量化对模型性能的影响，指向了 [ggerganov 的 Twitter 帖子](https://x.com/ggerganov/status/1666087050725199872) 和 [GitHub 上的 pull request](https://github.com/ggerganov/llama.cpp/pull/1684)。这些资源展示了不同位级量化方法的效率及其实现。

**提到的链接**：<a href="https://x.com/ggerganov/status/1666087050725199872">Georgi Gerganov (@ggerganov) 的推文</a>：2, 3, 4, 5 和 6-bit 量化方法现已在 llama.cpp 中可用。支持 ARM NEON, AVX2 和 CUDA 的高效推理实现 - 请参见截图中的示例数据。非常感谢 ikawrakow f...

  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/)** (1 messages): 

vkc6969: 有关于扩散模型（diffusion models）机械可解释性（mech interp）的研究工作吗？
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1243922350236499968)** (11 messages🔥): 

- **Pythia 数据复现期间 Tokenizers 报错**：成员们讨论了由于 Tokenizers 的变化导致复现 Pythia 数据时遇到的问题。一位成员提到 *"HF 在某个时间点对 Tokenizers 的工作方式进行了重大破坏性更改。"*
- **Batch_viewer.py 作为反分词的解决方案**：建议使用 Pythia 仓库中的 `batch_viewer.py` 来正确访问训练批次的内容。[GitHub 仓库](https://github.com/EleutherAI/pythia#exploring-the-dataset)中提供了说明，解释了它使用的是数据集的预打乱（pre-shuffled）版本。
- **读取 .bin 和 .idx 文件的挑战**：讨论显示，简单地加载 `.bin` 文件是行不通的，因为必须使用 `MMapIndexedDataset` 类来正确解析 Token 数据。
- **正确加载文件**：成员们强调了使用 `dtype='uint16'` 加载 memmap 的重要性，以便从预打乱的数据集中正确读取 Token。一个读取示例显示：*"It is done, and submitted. You can play “Survival of the Tastiest” on Android, and on the web..."*
- **MMapIndexedDataset 代码复杂度**：涉及 `MMapIndexedDataset` 和 `struct` 包的代码被一些成员认为非常复杂，类似于“黑魔法”。然而，这对于正确解析数据集中的文档 Token 至关重要。

**提到的链接**：<a href="https://github.com/EleutherAI/pythia#exploring-the-dataset">GitHub - EleutherAI/pythia: EleutherAI 关于可解释性和学习动态研究的核心库</a>：EleutherAI 关于可解释性和学习动态研究的核心库 - EleutherAI/pythia

  

---



### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1244683541741047839)** (1 messages): 

- **参加关于 RAG 流水线的特别网络研讨会**：“我们将在本周四上午 9 点（太平洋时间）与 Ragas + AWS 团队合作举办一场特别的网络研讨会，主题是使用 Bedrock、Ragas 和 LlamaIndex 构建企业级 RAG 流水线。” 请在 [lu.ma](https://lu.ma/x8unmku0) 注册参加，学习如何将 LlamaIndex 与 Bedrock 集成，并使用 Ragas 创建评估框架。



**提到的链接**：<a href="https://lu.ma/x8unmku0">LlamaIndex 网络研讨会：使用 Bedrock, Ragas 和 LlamaIndex 构建企业级 RAG · Zoom · Luma</a>：这是来自 LlamaIndex、Ragas 和 AWS 团队的特别合作，为您带来关于构建生产级企业 RAG 的研讨会……

  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1243666346936504340)** (7 条消息): 

- **Vespa 集成大放异彩**：社区对新的 Vespa vector store 集成感到兴奋，其对 BM25 hybrid search 的强大支持备受赞誉。点击[此处](https://twitter.com/llama_index/status/1794106979213869413)查看官方公告。

- **极速 RAG Chatbot 指南上线**：Jayita B. 提供了一份内容丰富的指南，介绍了如何构建高级 RAG indexing/query pipeline，并将其转化为在 Llama3 和 GroqInc 上具有快速响应能力的 full-stack 应用。点击[此处](https://twitter.com/llama_index/status/1794384304665141586)深入学习教程。

- **基于结构化标注的创新图像索引**：一种为 RAG 进行图像索引的新方法涉及使用由 gpt4o 等模型生成的 structured annotations，该方法以高效著称。[了解更多](https://twitter.com/llama_index/status/1794517226986656150)。

- **LlamaFS 清理你的凌乱文件**：介绍 LlamaFS，一个可以自动整理目录的自组织 file manager。点击[此处](https://twitter.com/llama_index/status/1794762651769430381)查看其工作原理。

- **使用 AWS Bedrock 构建企业级 RAG 网络研讨会**：一场特别的 webinar 即将举行，旨在帮助你使用 AWS Bedrock 和 Ragas 构建 production-level RAG pipelines。不要错过这次合作 workshop，详情见[此处](https://twitter.com/llama_index/status/1795123736699629983)。

**提到的链接**：<a href="https://t.co/qIGOmCW62G">RSVP to GenAI Summit Pre-Game: Why RAG Is Not Enough? | Partiful</a>：注意：这是在旧金山 LlamaIndex HQ 举办的线下 meetup！顺道参加我们的 meetup，了解为公司构建 production-grade retrieval augmented generation 引擎的最新创新...

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1243671400200405162)** (100 条消息🔥🔥): 

- **Llama Index 的 reAct 最大迭代次数问题**：一位用户在使用 Llama Index 的 reAct 时遇到了最大迭代次数错误。另一位成员建议通过在 `ReActAgent` 构造函数中设置 `max_iterations` 参数来增加迭代次数。
  
- **导入错误和安装困惑**：多位用户在 `llama_index` 中遇到了 `SimpleDirectoryReader` 和 `DEFAULT_FILE_EXTRACTOR` 的导入错误及包冲突。解决方案包括安装特定的子包，并检查包版本之间的更新或冲突。

- **解析 HTML 与 PDF 文件**：关于解析 HTML 与 PDF 文件的优缺点展开了详细讨论。许多人认为 HTML 需要更多自定义解析代码，而 PDF 分块（chunking）工具似乎更先进，且对额外库的依赖较少。

- **将 Pydantic 与 LlamaIndex 集成**：提供了关于在 LlamaIndex 中使用 `Pydantic` 类实现结构化输出的说明。指导内容包括定义 Pydantic 模型，并通过 `OpenAIPydanticProgram` 进行集成。

- **改进检索器（retrievers）文档**：有建议提出改进关于 BM25 和 AutoRetrieval 等不同检索器模块的文档。成员们澄清了它们的使用场景和区别，强调了当前文档中的空白。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/latest/getting_started/starter_example/#query-your-data">入门教程 (OpenAI) - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/node_parser/file/html.py">llama_index/llama-index-core/llama_index/core/node_parser/file/html.py at main · run-llama/llama_index</a>：LlamaIndex 是一个用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/prompts/prompt_mixin/#accessing-prompts">在高级模块中访问/自定义 Prompt - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/understanding/putting_it_all_together/q_and_a/terms_definitions_tutorial/#improvement-3-image-support>)">提取术语和定义指南 - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/output_parsing/openai_pydantic_program/#without-docstring-in-model>))">OpenAI Pydantic Program - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/finetuning/embeddings/finetune_embedding_adapter.ipynb">llama_index/docs/docs/examples/finetuning/embeddings/finetune_embedding_adapter.ipynb at main · run-llama/llama_index</a>：LlamaIndex 是一个用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/7849b1a851d88ee28e1bfd05d19f18e40d5b8e10/llama-index-integrations/embeddings/llama-index-embeddings-adapter/llama_index/embeddings/adapter/base.py#L115">llama_index/llama-index-integrations/embeddings/llama-index-embeddings-adapter/llama_index/embeddings/adapter/base.py at 7849b1a851d88ee28e1bfd05d19f18e40d5b8e10 · run-llama/llama_index</a>：LlamaIndex 是一个用于 LLM 应用程序的数据框架 - run-llama/llama_index
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1243666547294343290)** (93 条消息🔥🔥): 

- **对 Aya-23 功能及对比的兴趣**：成员们讨论了 **Aya-23** 的潜力及其与 **Command R/R+** 相比的功能。有评论提到 Aya-23 可能具有“改进的多语言能力”，但对其在英语方面的表现和潜在的专门训练（“基于 Cohere 的 Command model1 和 Aya 多语言指令式集合”）表示疑问。
  
- **移动项目的本地模型依赖**：一位成员寻求关于开发出于隐私原因在本地运行 LLM 的 **RAG 移动应用**的建议。共识认为**手机端 LLM 尚未准备好**处理此类任务。

- **Aya-23 中的 System Prompts**：对话显示 **Aya-23** 支持 system prompts，用户通过修改 **Command R** 的 prompts 成功使其在 **Aya-23** 上运行。一位成员分享了一个特定的模板配置，重点突出了 `<|USER_TOKEN|>` 和 `<|CHATBOT_TOKEN|>` 等 token。

- **OneDrive 连接器指南**：关于是否存在 **OneDrive 连接器**的查询得到了回答，指向了一个 [SharePoint 连接器](https://github.com/cohere-ai/quick-start-connectors/tree/main/sharepoint)，它可能具有相同的用途。

- **Aya-23 可用性说明**：官方沟通确认 **Aya-23-35b** 是 **Command R** 的微调版本，目前仅供非商业用途。一位成员指向了[官方论文](https://drive.google.com/file/d/1YKBPo61pnl97C1c_1C2ZVOnPhqf7MLSc/view)以获取更多细节。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://gitgud.autonoma.app/playground/3c135aa8-2720-4950-a184-61b3948a55bf/code?utm_source=discord&utm_medium=social&utm_campaign=cohere)">GitGud</a>：未找到描述</li><li><a href="https://docs.cohere.com/docs/migrating-from-cogenerate-to-cochat">从 Generate API 迁移到 Chat API - Cohere 文档</a>：未找到描述</li><li><a href="https://drive.google.com/file/d/1YKBPo61pnl97C1c_1C2ZVOnPhqf7MLSc/view">aya_23_technical_report.pdf</a>：未找到描述</li><li><a href="https://github.com/cohe">cohe - 概览</a>：cohe 有 5 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/cohere-ai/quick-start-connectors/tree/main/sharepoint">quick-start-connectors/sharepoint at main · cohere-ai/quick-start-connectors</a>：这个开源仓库提供了将工作场所数据存储与 Cohere 的 LLM 集成的参考代码，使开发人员和企业能够执行无缝的检索增强生成（RAG）...</li><li><a href="https://docs.cohere.com/docs/creating-and-deploying-a-connector">创建和部署连接器 - Cohere 文档</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1244762231497818193)** (3 条消息): 

- **LinkedIn 上的游戏机器人大放异彩**：一位成员分享了一篇 [LinkedIn 帖子](https://www.linkedin.com/posts/activity-7199625887955177472-nLbL?utm_source=share&utm_medium=member_ios)，重点介绍了他们使用 Cohere Command R 为 Discord 创建的新游戏机器人。该帖子引发了人们对该机器人功能和集成的关注。

- **Create 'n' Play 机器人亮相**：这款名为“Create 'n' Play”的 Discord 新游戏机器人拥有一些令人印象深刻的功能。它提供“超过 100 个引人入胜的文字游戏”，允许“轻松的团队组建和互动”，并增强了“与 AI 的社交参与度”。
  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1243639929712676937)** (77 条消息🔥🔥): 

- **Google AI 的 Reddit 翻车事故**：Google 的 AI 使用 Reddit 的信息导致了危险的建议，比如“跳桥自杀来治疗抑郁症”。一位成员调侃了 Reddit 上常见的“臭名昭著的糟糕建议”，并分享了一张 [meme 图片](https://media.discordapp.net/attachments/808203765319467028/1243450997511032935/GOUGsBgaMAA4N9k.jpg)。
- **Gemini AI 与 Google 的安全团队**：关于 Google 多次裁员（包括其安全团队）的讨论引发了人们对 AI 在处理敏感话题时表现不佳的担忧，例如在 Gemini 惨败期间。文中提到了一篇分享的 [《纽约时报》文章](https://www.nytimes.com/2024/02/22/technology/google-gemini-german-uniforms.html)，以提供这些问题的背景。
- **AI 与 UBI 的问题**：对话深入探讨了全民基本收入 (UBI)，成员们辩论了其可行性和公平性，特别是针对全球南方 (Global South) 国家。一位成员表示怀疑，指出 UBI “对于发达国家可能是可行的”，但对于较贫困地区则不然。
- **分享的数据集和 AI 工具**：成员们分享了有用的资源，例如来自 [archive.org](https://archive.org/details/arxiv-bulk?sort=-publicdate) 的出版物 PDF 及其 TeX 源码的潜在数据集，以及一段关于仅使用 Redstone（红石）创建神经网络的 [YouTube 视频](https://youtu.be/DQ0lCm0J3PM?si=5Is7OMnqRhZb-ZAo)。
- **Mastodon 作为 Twitter 的替代方案**：用户对 Twitter 的算法更改表示沮丧，并讨论了 Mastodon 等替代方案。一位成员强调，Mastodon 缺乏引发愤怒的内容，这可能是它没有出现指数级增长的原因，但它仍然保持着忠实的用户群。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://playpermissionless.substack.com/p/twitter-is-now-attention-roulette">Twitter is now attention roulette and ultimately meaningless</a>：我不再对 Twitter 有任何向往。我记不起上一次它对我个人产生什么好处是什么时候了。既没有思想的火花，也没有人脉的连接。算法的改变让 Twitter 变成了...</li><li><a href="https://www.tiktok.com/@hbo/video/7268371228384251182>">TikTok - Make Your Day</a>：无描述</li><li><a href="https://huggingface.co/spaces/Tonic/Granite-Code">Granite Code - a Hugging Face Space by Tonic</a>：无描述</li><li><a href="https://youtu.be/DQ0lCm0J3PM?si=5Is7OMnqRhZb-ZAo">I Made a Neural Network with just Redstone!</a>：想要免费试用 Brilliant 提供的所有服务整整 30 天，请访问 https://brilliant.org/mattbatwings。您还可以获得年度高级订阅 20% 的折扣...</li><li><a href="https://archive.org/details/arxiv-bulk?sort=-publicdate">Internet Archive: Digital Library of Free &amp; Borrowable Books, Movies, Music &amp; Wayback Machine</a>：无描述
</li>
</ul>

</div>
  

---


### **LAION ▷ #[announcements](https://discord.com/channels/823813159592001537/826154622644649985/1243929841565175858)** (1 条消息): 

- **Llama 3 German-8B 发布**：祝贺 Disco Research 和达姆施塔特工业大学 (TU Darmstadt) 的 Occi 团队发布了专门的德语大语言模型。**[Llama3-German-8B](https://huggingface.co/DiscoResearch/Llama3-German-8B)** 基于 Meta 的 Llama3-8B，在 65B 高质量德语 tokens 上进行了训练，现已发布。

**提到的链接**：<a href="https://fxtwitter.com/DiscoResearchAI/status/1794351790378594418?t=QCShTixHVItiIr93kUNV1A&s=19">来自 DiscoResearch (@DiscoResearchAI) 的推文</a>：🪩 介绍 Llama3-German-8B！由 @DiscoResearchAI 和 @occiglot 构建的专门针对德语的大语言模型。基于 @Meta 的 Llama3-8B，它在 65B 高质量德语 tokens 上进行了训练...

  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1243699924533514371)** (9 messages🔥): 

- **分享多标签分类资源**：一位寻求多标签分类帮助的成员收到了一份用于指导的 [Kaggle 链接](https://www.kaggle.com/code/altairfarooque/multi-label-image-classification-cv-3-0)。该用户还分享了一篇题为 "High-resolution Large Multimodal Models (LMMs)" 的 [arXiv 论文](https://arxiv.org/abs/2405.15738) 以提供更多背景信息。

- **ConvNeXt 在高分辨率任务中的应用**：围绕 [ConvNeXt 论文](https://arxiv.org/abs/2405.15738) 展开了讨论，强调了 ConvNeXt 在将高分辨率图像压缩为丰富视觉特征方面的作用。用户强调 ConvNeXt 有效地防止了产生过多的视觉 token，并提出了针对高分辨率任务的优化建议。

- **用于高效预训练的模型增长**：分享了另一篇 [arXiv 论文](https://arxiv.org/abs/2405.15319)，讨论了通过模型增长（model growth）进行高效 LLM 预训练的障碍。该论文识别了三个关键障碍，并引入了深度堆叠（depthwise stacking）作为一种优越的增长算子，能够提高速度、降低训练损失，并在 NLP 基准测试中表现良好。

- **Diffusers 0.28.0 版本发布**：重点介绍了 Diffusers 0.28.0 的发布，侧重于社区集成，并引入了第一个非生成式 pipeline —— **Marigold**，用于深度估计和表面法线预测。完整的发布说明可以在 [GitHub release 页面](https://github.com/huggingface/diffusers/releases/tag/v0.28.0)找到。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.15738">ConvLLaVA: Hierarchical Backbones as Visual Encoder for Large Multimodal Models</a>：高分辨率大语言多模态模型 (LMMs) 面临视觉 token 过多和二次方视觉复杂度的挑战。当前的高分辨率 LMMs 在解决二次方复杂度的同时...</li><li><a href="https://arxiv.org/abs/2405.15319">Stacking Your Transformers: A Closer Look at Model Growth for Efficient LLM Pre-Training</a>：由于规模庞大，LLMs 的预训练计算成本非常高。模型增长通过利用较小的模型来加速较大模型的训练，成为一种极具前景的方法。然而...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1d1u9ma/diffusers_0280_is_here/">Reddit - Dive into anything</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1244619607873028219)** (4 messages): 

- **随机种子导致不同的 loss 值**：一位成员注意到在重新加载时遇到了不同的 loss 值，理由是随机种子（random seeding）问题导致训练集和验证集发生变化。他们通过将训练和验证数据放在不同的文件夹中解决了这个问题，使过程不再混淆。
- **PyTorch 模型保存与加载问题**：另一位成员承认他们的问题源于没有在 PyTorch 中保存优化器状态（optimizer states），这导致了 loss 爆炸。他们确认该问题与 Adam 优化器的内部状态有关。
  

---



### **DiscoResearch ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/1243732347610857543)** (1 messages): 

- **Mistral 发布 Mixtral 模型微调指南**：Mistral 发布了专门针对 Mixtral 模型的微调指南，详细介绍了有效微调的策略和技巧。请在他们的 [GitHub 仓库](https://github.com/mistralai/mistral-finetune)中查看指南。

**提及的链接**：<a href="https://github.com/mistralai/mistral-finetune">GitHub - mistralai/mistral-finetune</a>：通过在 GitHub 上创建账号来为 mistralai/mistral-finetune 的开发做出贡献。

  

---

### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1243910861668749322)** (31 messages🔥): 

- **Llama3-German-8B 模型发布引发社区关注**：Llama3-German-8B 作为 Meta Llama3-8B 的延续版本已发布，旨在提升德语语言能力，该模型在 650 亿高质量 Token 上进行了训练。*"我们模型的 Benchmark 结果显示，尽管训练过程中没有进行 Replay，但英语性能的下降微乎其微。"* 更多信息可以在[这里](https://huggingface.co/DiscoResearch/Llama3-German-8B)找到。
- **注意到 GGUF 量化问题**：对该模型进行了快速 GGUF 量化，但产生的 Benchmark 分数较低，表明可能存在 Bug。该量化版本可在[这里](https://huggingface.co/cstr/Llama3-DiscoLeo-Instruct-8B-32k-v0.1-GGUF)获取。
- **省略 Replay 引起关注**：Llama3-German-8B 在预训练时没有使用英语数据 Replay。*"我们想要最佳的德语性能，不进行 Replay 似乎是最好的选择。"* Quicksort 指出，Replay 有助于提升其他语言的下游性能。
- **Cohere 的 Aya-23-35B 模型引起兴趣**：**Cohere 的 Aya-23-35B** 被提及为一个支持 23 种语言的强大新型多语言模型，但其许可证较为严格。[Aya-23-35B 详情](https://huggingface.co/speakleash/Bielik-7B-v0.1)。
- **共享 ShareGPT 格式数据集**：分享了一个将数据集转换为 ShareGPT 格式的成果，引发了关于翻译质量和数据集过滤的讨论。数据集可以在[这里](https://huggingface.co/datasets/sroecker/aya_german-sharegpt)访问。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/cstr/Llama3-DiscoLeo-Instruct-8B-32k-v0.1-GGUF">cstr/Llama3-DiscoLeo-Instruct-8B-32k-v0.1-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/sroecker/aya_german-sharegpt">sroecker/aya_german-sharegpt · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/speakleash/Bielik-7B-v0.1">speakleash/Bielik-7B-v0.1 · Hugging Face</a>: 未找到描述</li><li><a href="https://docs.google.com/document/u/0/d/1Cg6W7vdXe4YoeZAB5xFQXAlR_t9dZO-wDJvCvRsCxwQ/mobilebasic">MMLU Annotation Questions + Instructions</a>: 未找到描述</li><li><a href="https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1">RLHFlow/ArmoRM-Llama3-8B-v0.1 · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/7530">Tokenizer BPE fixes by jaime-m-p · Pull Request #7530 · ggerganov/llama.cpp</a>: 使 BPE Tokenizer 与 AutoTokenizer 匹配的修改。已使用以下模型的词表进行测试：gpt-2, llama-bpe, falcon, deepseek-coder (失败: &#39;added_tokens&#39; 长度等于 1), deepseek-llm (失败: &#39...</li><li><a href="https://huggingface.co/DiscoResearch/Llama3-German-8B">DiscoResearch/Llama3-German-8B · Hugging Face</a>: 未找到描述</li><li><a href="https://discord.gg/VaXDGsyebd)">Discord | Your Place to Talk and Hang Out</a>: Discord 是进行语音、视频和文字交流最简单的方式。与你的朋友和社区聊天、聚会并保持紧密联系。</li><li><a href="https://x.com/DiscoResearchAI/status/1794351790378594418">来自 DiscoResearch (@DiscoResearchAI) 的推文</a>: 🪩 隆重推出 Llama3-German-8B！这是一个专为德语定制的大语言模型，由 @DiscoResearchAI 和 @occiglot 构建。基于 @Meta 的 Llama3-8B，它在 65B 高质量德语 Token 上进行了训练...</li><li><a href="https://huggingface.co/collections/DiscoResearch/discoleo-8b-llama3-for-german-6650527496c0fafefd4c9729">DiscoLeo 8B: Llama3 for German - a DiscoResearch Collection</a>: 未找到描述</li><li><a href="https://huggingface.co/collections/DiscoResearch/discoleo-8b-quants-6651bcf8f72c9a37ce485d42">DiscoLeo 8B quants - a DiscoResearch Collection</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/CohereForAI/aya_collection_language_split/viewer/german/train?f%5Bdataset_name%5D%5Bvalue%5D=%27Xlel_wd-inst%27&row=4018775">CohereForAI/aya_collection_language_split · Datasets at Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1244055944254717952)** (6 messages): 

- **Llama3-DiscoLeo 反馈突显参数问题**：一位用户分享了 Llama3-DiscoLeo-Instruct-8B-32k-v0.1 的轶事反馈，其参数设置包括 max tokens 为 512，temperature 范围为 0.5-1。
- **关于运行参数和输出问题的讨论**：另一位成员询问了引擎和量化设置，透露他们使用了带有普通 llama3 模板的 ollama，并在 EQ Bench 上遇到了奇怪的结果。
- **Stop Token 设置与输出问题**：注意到使用 oobabooga 生成的输出包含未经审查的回答和个人数据，而使用 q4km gguf 的 ollama 输出则更合理。该成员在另一个频道也看到了类似问题的讨论。
- **EQ Bench 模板建议**：建议使用特定的 [Llama3 模板](https://github.com/CrispStrobe/EQ-Bench/blob/main_v2_3a/instruction-templates/Llama3.yaml) 并设置自定义停止字符串（stopping strings），以查看是否能解决输出问题。
- **在 oobabooga 中实验 Special Tokens**：建议在 oobabooga 中尝试设置 'skip_special_tokens': False，可能还包括 'add_bos_token': True, 'ban_eos_token': False 以获得更好性能，尽管该用户承认对该工具并不熟悉。

**提及的链接**：<a href="https://github.com/CrispStrobe/EQ-Bench/blob/main_v2_3a/instruction-templates/Llama3.yaml">EQ-Bench/instruction-templates/Llama3.yaml at main_v2_3a · CrispStrobe/EQ-Bench</a>：大型语言模型情感智能基准测试 - CrispStrobe/EQ-Bench

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1243647190690299925)** (21 messages🔥): 

- **Rabbit LAM 引发辩论**：成员们讨论了 Large Action Model (LAM) 的效能和未来，质疑其成熟度及逆向工程的可能性。他们分享了 [Rabbit LAM 架构概览](https://engineering.rabbit.tech/lam-security-architecture-overview)，解释了其如何使用 Playwright 进行安全的 Web 应用控制。
  
- **在移动端和 Rabbit R1 上运行 01**：出现了关于在 Humane 平台和 Rabbit R1 设备上运行 01 模型的问题。还有关于潜在硬件问题以及将 Rabbit R1 与 Open Interpreter 集成的目标的评论。

- **模型数据的存储解决方案**：一位成员询问如何克服 Runpod 上的磁盘空间限制，并寻求性价比高的本地数据存储方案建议。另一位成员提供了一个 [GitHub 仓库](https://github.com/Tonylib/o1_for_flutter)，用于在 Android 上运行 01，并建议可以将其适配到 Rabbit R1。

- **Open Interpreter 安装咨询**：一位用户询问安装 Open Interpreter 是否必须使用 OpenAI API。回复澄清说需要一个 LLM，可以是本地的，也可以通过云端代理服务访问。

- **Markdown 导出功能**：引入了一项新功能，允许将对话导出为 Markdown。分享了一个 [pull request](https://github.com/OpenInterpreter/open-interpreter/pull/1282) 详细说明了这些更改。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://engineering.rabbit.tech/lam-security-architecture-overview">LAM 安全与架构概览</a>：未找到描述</li><li><a href="https://github.com/Tonylib/o1_for_flutter">GitHub - Tonylib/o1_for_flutter</a>：通过创建账号为 Tonylib/o1_for_flutter 的开发做出贡献。</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/pull/1282">由 Steve235lab 提交的将对话导出为 Markdown 的 Pull Request #1282 · OpenInterpreter/open-interpreter</a>：描述所做的更改：添加一个新的魔法命令 %markdown [path]，将对话导出到指定的 Markdown 路径。如果未提供路径，它将保存到下载文件夹...</li><li><a href="https://youtu.be/zLvFc_24vSM?si=XFsWIzpGDW4IsEEp,">Rabbit 忽悠了我，所以我进行了深入挖掘</a>：LAM 是个骗局吗？让我们深入探究。Jesse 对 Jason Calacanis 的采访：https://youtu.be/X-MNgciL5hw?si=qh6SgCYtkEiD8UNu 帮助完成此内容的人们...
</li>
</ul>

</div>

### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1243697909489340519)** (16 条消息🔥): 

- **在 Mac 上使用 OpenInterpreter 的困扰**：一位用户在 Mac 上使用 OpenInterpreter 时遇到语音录制停止后无响应的问题，通过切换到 Python 3.11 解决了该问题。
- **硬件发货咨询**：用户对预订硬件的发货状态感到焦虑；一些人尚未收到任何更新或邮件，而另一些人则对欧洲的发货时间表感到好奇。
- **将 O1 与耳机集成**：一位用户正在进行一个项目，在安卓驱动的改装耳塞上安装 OpenInterpreter，通过 [Andronix](https://andronix.app/) 使用 Linux 发行版来持续运行该软件。
- **辩论 DIY 与预制 O1**：讨论强调，购买 O1 而非自行构建的主要优势在于支持西雅图的开发团队，尽管底层产品是相同的。
- **增强 R1 设备的计划**：一些用户对 R1 不满意，正在探索通过将其连接到 OpenInterpreter 来使其发挥作用的方法，包括对其进行 Root 并刷入 LineageOS 以获得更好的功能。
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1243807968068370464)** (31 条消息🔥): 

- **PDF 提取的挑战**：成员们讨论了[从 PDF 中提取文本的挑战](https://python.langchain.com/v0.2/docs/how_to/document_loader_pdf/)，特别是处理复杂的表格和图表。有人建议将 PDF 视为图像并应用 ML 进行文本分割，而另一位则建议使用 Adobe Extract API 进行布局解析。

- **对建立本地 LangChain 社区的兴趣**：Scogo Networks 的联合创始人 Karan Singh 表示有兴趣在孟买建立 LangChain 社区。他询问了 LangChain 营销团队的联系方式，以讨论组织当地活动的事宜。

- **利用 AWS Bedrock 和 Pinecone 构建聊天机器人**：一位使用 AWS Bedrock 和 Pinecone 构建聊天机器人应用的用户在维护对话上下文方面遇到了问题。另一位成员建议使用 [LangChain AgentExecutor](https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/) 以实现更好的上下文管理和检索。

- **编程与 LLM 就业市场讨论**：用户就 LLM 时代学习编程是否仍有价值展开了辩论。一位成员认为 LLM 可能会创造更多的编程工作，可能属于不同的类型。

- **寻求 LangChain 与 Instructor 集成的帮助**：一位成员请求协助将 Instructor 与 LangChain 集成。聊天中未提供详细指导，凸显了该集成可用帮助资源的匮乏。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.langchain.com/">LangChain</a>：LangChain 的系列产品在开发者的每一步开发过程中提供支持。</li><li><a href="https://tenor.com/view/matrix-morpheus-stop-trying-hit-me-come-on-gif-20012191">Matrix Morpheus GIF - Matrix Morpheus Stop Trying - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/">定义自定义工具 | 🦜️🔗 LangChain</a>：在构建自己的 Agent 时，你需要为其提供一个可使用的 Tools 列表。除了实际调用的函数外，Tool 还包含几个组件：
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1244721736314781756)** (2 条消息): 

- **Langserve 注册表单错误**：一位成员指出通过 Airtable 访问托管版 **Langserve** 等待名单时遇到困难。他们遇到了错误，并询问是否有其他试用方式。

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1244189735820328980)** (2 条消息): 

- **NLAVIDA 项目发布并附带 YouTube 教程**：一名成员分享了一个名为 NLAVIDA 的开源项目，旨在利用自然语言进行交互式数据可视化和分析。查看 [YouTube 视频教程](https://www.youtube.com/watch?v=leJRP_mJsSQ&t=4s) 了解更多信息。
  
- **OranClick 在 ProductHunt 上线**：宣布了一款名为 OranClick 的新工具，旨在帮助用户编写、追踪和分析消息，以提高注册率。该工具已在 [ProductHunt 上线](https://producthunt.com/posts/oranclick)，鼓励用户投票支持。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://producthunt.com/posts/oranclick"> OranClick - Messages that Click | Product Hunt</a>：你是否在为达到目标注册量而苦恼？轻松编写、追踪和分析。只需：- 输入消息和链接 - 发布消息和自定义追踪 URL - 分析并优化消息...</li><li><a href="https://www.youtube.com/watch?v=leJRP_mJsSQ&t=4s">NLAVIDA: An Open Source Tool for Interactive Data Analysis and Visualisation using LLM</a>：NLAVIDA (Natural Language-Assisted Visualization and Interactive Data Analysis) 是 code interpreter 或高级数据分析工具的开源替代方案...
</li>
</ul>

</div>
  

---



### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1243731215463219220)** (34 条消息🔥): 

- **联网 Llamafile 服务器技巧**：成员们讨论了如何使 llamafile 服务器在网络中可用，技巧包括添加 `--host <my ip>` 或使用 `--host 0.0.0.0`。这使得服务器可以从同一网络中的不同机器访问。
- **Llama3-70B 莫名出现空白回复**：用户报告了 llama3-70b 模型出现空白回复的问题，并通过分享日志寻求故障排除帮助。另一位用户介入但没有直接解决方案，表明这可能需要更深入的调查。
- **Llamafile v0.8.5 发布及基准测试**：社区庆祝了 llamafile 0.8.5 版本的发布，强调其现在为 X86 CPU 上的 K quants 提供了快速推理。鼓励成员加入基准测试俱乐部，使用 `llamafile-bench` 测试并分享结果。
- **Home Assistant 集成愿望清单**：Home Assistant 集成反馈强调了对类似于 OpenAI 的标准化本地 API 的需求，建议命名为 Apilla，并指出 API 的可发现性（通过 DNS-SD/zeroconf）和安全 API 是理想的功能。
- **Python 示例中的模型选择**：关于在 LLaMA_CPP 集成的 Python 示例中指定模型的问题，用户分享了代码片段，并询问在运行 TinyLlama 等实例时是否有必要指定模型。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.home-assistant.io>)">无标题</a>：未找到描述</li><li><a href="http://<Your">无标题</a>：未找到描述</li><li><a href="https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf">openbmb/MiniCPM-Llama3-V-2_5-gguf · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/Mozilla/Meta-Llama-3-8B-Instruct-llamafile#prompting">Mozilla/Meta-Llama-3-8B-Instruct-llamafile · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/Mozilla/Mistral-7B-Instruct-v0.3-llamafile">Mozilla/Mistral-7B-Instruct-v0.3-llamafile · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/Mozilla/granite-34b-code-instruct-llamafile">Mozilla/granite-34b-code-instruct-llamafile · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/JustineTunney/status/1794732286900043830">Justine Tunney (@JustineTunney) 的推文</a>：事实证明，耐心就是你所需要的一切。我们现在能够在一台价值 362 美元的 CPU 上运行 Mixtral 8x22b Q6_K，速度优于人类阅读速度。https://github.com/Mozilla-Ocho/llamafile/discussions/450</li><li><a href="https://ollama.com/yabi/minicpm-llama3-v-2_5">yabi/minicpm-llama3-v-2_5</a>：快速上手运行大语言模型。</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8.5">Release llamafile v0.8.5 · Mozilla-Ocho/llamafile</a>：此版本修复了错误并引入了 @Kawrakow 最新的量化性能增强（llamafile 独有功能）。截至 #435，K quants 的运行速度始终比 llama.cpp 快 2 倍以上...</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/discussions">Mozilla-Ocho/llamafile · Discussions</a>：浏览 Mozilla-Ocho llamafile 的 GitHub 讨论论坛。讨论代码、提出问题并与开发者社区协作。
</li>
</ul>

</div>
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1243666066530369661)** (18 messages🔥): 

- **发现 Mistral-Finetune 仓库**：一些用户讨论了 [Mistral-Finetune 仓库](https://github.com/mistralai/mistral-finetune)，尽管有一位用户不确定具体的更改内容。*"看到了这个，但不确定有什么不同。"*
- **MoEs 微调中的性能差异**：关于 Mixture of Experts (MoEs) 模型微调的讨论指出 *"性能存在高度差异"*，并建议在微调过程中 *"运行多个实例"*，以便筛选出性能最佳的模型实例。
- **Aya 23 模型的使用场景限制**：用户讨论了 Aya 23 的有效性，指出它 *"未针对 chat 模式使用进行优化"*，仅对多语言指令遵循等 *"非常特定的用例"* 有益，详见其 [技术报告](https://cohere.com/research/papers/aya-command-23-8b-and-35b-technical-report-2024-05-23)。
- **最佳开源日语支持模型**：LHL 表示 [Command-R Plus](https://huggingface.co/shisa-ai/shisa-v1-llama3-70b) 是顶级的非商业模型，而 Shisa-V1-Llama3-70B 被认为是一个强大的 *可商用开源模型*。其他值得关注的还包括 Claude Haiku 和 Gemini 1.5 Flash，因为它们具有很高的性价比。
- **引入用于微调的 MoRA**：[MoRA 方法](https://arxiv.org/pdf/2405.12130) 被作为 LoRA 的进阶版引入，它在保持相同数量的可训练参数的同时，使用了高秩更新（high-rank updating）方法。MoRA 的 GitHub 仓库可以在 [这里](https://github.com/kongds/MoRA) 找到。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://cohere.com/research/papers/aya-command-23-8b-and-35b-technical-report-2024-05-23">Aya 23: Open Weight Releases to Further Multilingual Progress</a>：未找到描述</li><li><a href="https://huggingface.co/shisa-ai/shisa-v1-llama3-70b">shisa-ai/shisa-v1-llama3-70b · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/mistralai/mistral-finetune">GitHub - mistralai/mistral-finetune</a>：通过在 GitHub 上创建账号来为 mistralai/mistral-finetune 的开发做出贡献。</li><li><a href="https://github.com/kongds/MoRA">GitHub - kongds/MoRA: MoRA: High-Rank Updating for Parameter-Efﬁcient Fine-Tuning</a>：MoRA：用于参数高效微调的高秩更新 - kongds/MoRA
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1243702512519876720)** (3 messages): 

- **Main 分支中的 FFD Bin Packing 问题**：一位成员强调 "main 分支中的 FFD bin packing 实现" 在分布式训练中似乎已损坏。具体而言，他们指出 "（来自 packing 效率的）长度估计不存在"。

- **调试确认**：另一位成员确认，提到的 FFD bin packing 实现中 "确实存在一些问题" 需要调试。

- **Unsloth 的 Llama 3 修复**：用户讨论了 Unsloth 最近针对 Llama 3 的修复，指出基础版本有 "一些未训练的 tokens"。他们还分享了一个 [sfttrainer 的补丁](https://github.com/unslothai/unsloth/blob/main/unsloth/tokenizer_utils.py#L722)，用于检查/移除重复的 `bos_tokens`。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/unslothai/unsloth/blob/main/unsloth/tokenizer_utils.py#L557">unsloth/unsloth/tokenizer_utils.py at main · unslothai/unsloth</a>：微调 Llama 3, Mistral, Phi &amp; Gemma LLMs 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/tokenizer_utils.py#L722">unsloth/unsloth/tokenizer_utils.py at main · unslothai/unsloth</a>：微调 Llama 3, Mistral, Phi &amp; Gemma LLMs 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth
</li>
</ul>

</div>
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1243920733261009016)** (8 messages🔥): 

- **关于 `conversations` 与 `conversation` 的拼写混淆**：一位成员指出 [documentation](https://github.com/OpenAccess-AI-Collective/axolotl/blob/8a20a7b711a62d7b04e742f3d6034b4ca8aa27d2/README.md?plain=1#L373) 中关于键名应该是 `conversations` 还是 `conversation` 存在不一致。另一位成员通过检查代码确认了正确用法，从而解决了该问题。

- **针对微调方法的 Benchmark 请求**：一位成员询问了针对 MMLU 等 MQC 任务，比较 "sequence packing" 与 "naive padding" 微调性能的 Benchmark。他们分享了自己的经验，指出 packed 和 non-packed 模型之间存在显著的性能差异。

- **微调 Mistral 7b inst v2 的观察结果**：在修改 Prompt 格式并微调 Mistral 7b inst v2 后，一位成员观察到虽然回复质量有所提高，但模型似乎出现了幻觉（hallucinate）。他们注意到训练期间 Loss 显著下降，并质疑这是否与其数据结构有关。
<div class="linksMentioned">

<strong>涉及链接</strong>:

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/8a20a">GitHub - OpenAccess-AI-Collective/axolotl at 8a20a7b711a62d7b04e742f3d6034b4ca8aa27d2</a>: 尽管去问 axolotl 问题吧。通过在 GitHub 上创建账户为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/8a20a7b711a62d7b04e742f3d6034b4ca8aa27d2/README.md?plain=1#L373">axolotl/README.md at 8a20a7b711a62d7b04e742f3d6034b4ca8aa27d2 · OpenAccess-AI-Collective/axolotl</a>: 尽管去问 axolotl 问题吧。通过在 GitHub 上创建账户为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/)** (1 messages): 

nanobitz: 太棒了！
  

---



### **AI Stack Devs (Yoko Li) ▷ #[ai-companion](https://discord.com/channels/1122748573000409160/1122788693950857238/1243808010942283786)** (3 messages): 

- **探索 AI 与模拟的 Virtual Beings Summit**：一位成员分享了 [Virtual Beings Summit](https://www.virtual-beings-summit.com)，重点介绍了 AI 创始人、Stanford 研究员以及 Virtual Beings 初创公司负责人。该活动承诺将举行关于 "AI & Simulations" 的小组讨论，并邀请了 The Sims 的创作者 **Will Wright** 等演讲嘉宾。
- **在 YouTube 上探索 AI 的实际应用**：分享了 YouTube 频道 [Virtual Beings](https://www.youtube.com/channel/UCLNalUbhs_EYB5IAbw1VfaA)，强调其内容可能侧重于 AI 和交互式模拟。
<div class="linksMentioned">

<strong>涉及链接</strong>:

<ul>
<li>
<a href="https://www.virtual-beings-summit.com">Virtual Beings Summit</a>: 未找到描述</li><li><a href="https://www.youtube.com/channel/UCLNalUbhs_EYB5IAbw1VfaA">Virtual Beings</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1243664052929171536)** (2 messages): 

- **DIAMOND 项目备受关注**：一位成员分享了 [DIAMOND GitHub 仓库](https://github.com/eloialonso/diamond) 的链接，其全称为 **"DIffusion As a Model Of eNvironment Dreams"**。该项目被描述为“在 Diffusion 世界模型中训练的 Reinforcement Learning Agent”。

**涉及链接**: <a href="https://github.com/eloialonso/diamond">GitHub - eloialonso/diamond: DIAMOND (DIffusion As a Model Of eNvironment Dreams) is a reinforcement learning agent trained in a diffusion world model.</a>: DIAMOND (DIffusion As a Model Of eNvironment Dreams) 是一个在 Diffusion 世界模型中训练的 Reinforcement Learning Agent。- eloialonso/diamond

  

---

### **AI Stack Devs (Yoko Li) ▷ #[ai-town-dev](https://discord.com/channels/1122748573000409160/1137456826733047908/1243738151898316930)** (6 条消息): 

- **《西部世界》愿景中的 AI Town 和 UE5**：一位成员建议将 AI Town 与 **UE5** 以及**语音控制交互**结合，以创造一种“Westworld”式的体验。另一位成员热情地回应道：*“我们 4Wall 正在致力于此！”*。
- **VR 集成令成员兴奋**：将 AI Town 连接到 **VR** 以实现更具沉浸感交流的潜力令人兴奋。一位成员热切地问道：*“难道不能再次链接到 VR 吗？”*。
- **通过 4Wall Discord 保持更新**：对于那些有兴趣关注 AI Town 及其集成进展的人，建议加入 **4Wall Discord**。成员提供了链接 [discord.gg/vPum4s3h](https://discord.gg/vPum4s3h) 以便及时了解动态。

**提到的链接**：<a href="https://discord.gg/vPum4s3h">加入 4Wall AI Discord 服务器！</a>：查看 Discord 上的 4Wall AI 社区 - 与其他 511 名成员一起交流，享受免费的语音和文字聊天。

---

### **AI Stack Devs (Yoko Li) ▷ #[late-night-lounge](https://discord.com/channels/1122748573000409160/1159342774710186075/1243968582820429918)** (8 条消息🔥): 

- **探索用于动画的 SadTalker GitHub 仓库**：一位成员分享了 [SadTalker GitHub 仓库](https://github.com/OpenTalker/SadTalker)，用于“学习风格化音频驱动的单图像说话人脸动画的逼真 3D 运动系数”。他们鼓励其他人在需要处理涉及时间一致性（temporal consistency）的动画时寻求帮助。
- **运行 SadTalker 的 Hugging Face 潜在解决方案**：同一位成员建议查看 Hugging Face Spaces 以在本地运行 SadTalker 仓库，并表示非常有信心那里存在解决方案。
- **腾讯发布的新动画工具**：分享了另一个旨在生成说话人头像视频的仓库 [来自 Tencent AI Lab 的 V-Express](https://github.com/tencent-ailab/V-Express)。它因作为近期发布的创建引人入胜的 Avatar 工具而受到关注。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/tence">tence - 概览</a>：tence 有 11 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/tencent-ailab/V-Express">GitHub - tencent-ailab/V-Express：V-Express 旨在参考图像、音频和 V-Kps 图像序列的控制下生成说话人头像视频。</a>：V-Express 旨在参考图像、音频和 V-Kps 图像序列的控制下生成说话人头像视频。 - tencent-ailab/V-Express</li><li><a href="https://github.com/OpenTalker/SadTalker">GitHub - OpenTalker/SadTalker：[CVPR 2023] SadTalker：学习风格化音频驱动的单图像说话人脸动画的逼真 3D 运动系数</a>：[CVPR 2023] SadTalker：学习风格化音频驱动的单图像说话人脸动画的逼真 3D 运动系数 - OpenTalker/SadTalker
</li>
</ul>

</div>

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1243662467427598427)** (4 条消息): 

- **Zyphra Zamba 低调亮相**：Zyphra Zamba 是一种 Mamba/Attention 混合模型，已在极少宣传的情况下发布。关键资源包括 [技术报告](https://www.zyphra.com/s/Zamba.pdf)、[Torch 参考代码](https://github.com/Zyphra/Zamba-torch) 以及 [HF Transformers 代码](https://github.com/huggingface/transformers/pull/30950)。

- **与 OLMo 1.7 的对比正在进行中**：提到将把 Zyphra Zamba 与 OLMo 1.7 进行评估对比。

- **SD Audio 2.0 泄露**：据报道 SD Audio 2.0 已在 4chan 上泄露。它可以在一个未指定的 HF 账号上找到，并且相当容易定位。

**提到的链接**：<a href="https://x.com/QuentinAnthon15/status/1794084824464158745">来自 Quentin Anthony (@QuentinAnthon15) 的推文</a>：@philpax @ryu0000000001 我们决定在长周末前的正午发布可能不是个好主意 ;) 对于现在寻找模型 + 技术报告的人，这里是相关信息：- Tech...

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1244433887845814352)** (4 messages): 

- **前 OpenAI 董事会成员呼吁 AI 监管**：Hellen Toner 和 Tasha McCauley 在 [The Economist](https://www.economist.com/by-invitation/2024/05/26/ai-firms-mustnt-govern-themselves-say-ex-members-of-openais-board) 中撰文指出，由于利润动机，不能指望**私营 AI 公司**进行自我监管。他们强调了诸如 Sam Altman 被解雇以及内部安全协议担忧等问题。
- **Altman 的毒性文化受到批评**：前董事会成员指控 Sam Altman 培养了“撒谎的毒性文化”并进行“心理虐待”。尽管内部调查得出的结论是这些行为不足以强制将其免职，但调查细节并未公开，引发了争议。

**提到的链接**：<a href="https://www.economist.com/by-invitation/2024/05/26/ai-firms-mustnt-govern-themselves-say-ex-members-of-openais-board">AI firms mustn’t govern themselves, say ex-members of OpenAI’s board</a>：Helen Toner 和 Tasha McCauley 认为，为了人类的利益，需要通过监管来驯服市场力量。

  

---


### **Interconnects (Nathan Lambert) ▷ #[lectures-and-projects](https://discord.com/channels/1179127597926469703/1223784028428177510/1243676921498439730)** (10 messages🔥): 

- **224n 课程时间表尚不明确**：一位成员对新一期 224n 课程的发布时间表示关注。另一位成员表示这通常发生在学期末，但不确定，并提到可以询问 Chris。
- **强化学习教材发布**：分享了一个 [GitHub 仓库](https://github.com/natolambert/rlhf-book) 链接，宣布了一本关于 Reinforcement Learning from Human Feedback (RLHF) 的教材。一位用户询问是否会涵盖人类偏好收集的操作层面；回答指出该书主要侧重于技术细节，后半部分讨论了最近的研究项目。
- **Chris Potts 和 Chris Manning 受到赞扬**：成员们讨论了 Chris Manning 和 Chris Potts 的风范。Chris Manning 被亲切地描述为“讨人喜欢的超级极客”，而 Chris Potts 则因其幽默感和授课风格受到称赞。

**提到的链接**：<a href="https://github.com/natolambert/rlhf-book">GitHub - natolambert/rlhf-book: Textbook on reinforcement learning from human feedback</a>：关于人类反馈强化学习的教材 - natolambert/rlhf-book

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1243814257662099506)** (11 messages🔥): 

- **关于延长测试时限的辩论**：一位成员询问是否可以延长单项测试的时限（目前为 9 分 34 秒），并表示如果不延长，函数的 "Taylor approximations" 可能无法实现。他们补充说，虽然 `sin` 还可以应付，但其他的无法完成，比如 `clang` 仅达到约 60%。

- **处理大型表达式的困难**：一位用户提到正在尝试一种可能有效的解决方案，但由于生成的单个表达式过大，导致编译时崩溃。他们收到的错误信息是 *"invalid operands to binary expression ('double' and 'double')"*。

- **转向其他 Bounty 任务**：在分析了许多人在这项特定任务中面临的困难和挑战后，一位用户决定尝试其他 Bounty，并强调那些表面看起来容易的任务往往具有深层的复杂性。

- **Double 类型与位运算不兼容**：一位成员澄清说，不能对 `double` 类型执行诸如 XOR 之类的位运算，从而解释了讨论中的编译错误。

- **对 Bounty 和陈旧 Pull Request 的关注**：新用户询问了有趣的面向研究的 Bounty 以及旧 Bounty 的状态。特别提到了一项关于 [tinygrad](https://github.com/tinygrad/tinygrad/pull/4212) 的 PR，得到了 **georgehotz** 的确认，即该 Bounty 仍然有效。
  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1244317608640118955)** (5 条消息): 

- **UOp 类 'vin' 变量澄清**：一位成员询问 UOp 类中 'vin' 变量的含义，质疑它代表的是 'vector input' 还是 'value input'。George Hotz 澄清说 'vin' 可能没有特定含义，并指出 *"你只是不能使用 'in'。"*

- **Taylor Approximation 反馈请求**：一位成员分享了他们在 [tinygrad pull request #4739](https://github.com/tinygrad/tinygrad/pull/4739) 提交的 Taylor Approximation 的 GitHub pull request，并请求反馈以确保方向正确。该 pull request 被描述为一个简单的 proof of concept。

- **用于 scheduling 的 Post dominator analysis**：一位成员询问为什么在 scheduling 过程中不使用 post dominator analysis 来寻找用于 fusion 的自包含 subgraphs。该询问旨在探索模型 scheduling 过程中的潜在优化。

**提到的链接**：<a href="https://github.com/tinygrad/tinygrad/pull/4739/files,">Taylor Approximation (at about 0) for exponential proof of concept by mesozoic-egg · Pull Request #4739 · tinygrad/tinygrad</a>：简单的 POC，只想知道我的方向是否正确

  

---



---



---



---



---




{% else %}

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

**Fine-Tuning 事实**：在 [general 频道](https://discord.com/channels/1238365980128706560/1238365980128706563/1243282801760145408) 的讨论中，揭示了由于偏见数据类别导致的 **semantic similarity overfitting（语义相似度过拟合）** 问题。一位用户在理解 Fine-tuning 与用户输入及模型初始训练的关系时遇到了困难。此外，还注意到 **OpenAI 平台侧边栏** 的变化，两个图标（threads 和 messages）消失了。

**Templates 成为焦点**：在 [workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1243336501018755123) 中，强调了在 Fine-tuning 过程中正确配置 Template 的重要性。特别是分隔符 `###` 有助于解析不同的输入部分，而 "end of text" Token 则指示何时停止 Token 生成。

**Maven 与 Moderation 的交流**：在 [asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/1243344778511515698) 频道，成员们进行了一次轻松的交流，提到了重聚。关于会议演讲录像的请求得到了回应，视频已在 Maven 上发布。

**Modal 动员**：[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1243309176722030702) 频道的 Modal 用户分享了收到 Credit 的兴奋心情和训练经验，并为新用户提供了 **Modal 文档** 和 **示例** 的具体链接。还分享了一个使用 Modal 参加 **Kaggle 竞赛** 的计划，包括设置和执行细节。

**Jarvis 记录 Jupyter 杂记**：在 [jarvis-labs 频道](https://discord.com/channels/1238365980128706560/1241117895740625099/1243307629057671229) 中，成员们讨论了在 Jarvis 上存储 VSCode Repo 的问题，并建议使用 GitHub 保存工作。有一则关于因不稳定而移除 **spot instance** 的通知。分享了 Fine-tuning **open-lama-3b** 模型的成本和时长，一位用户通过调整模型参数解决了 Ampere 系列显卡的错误。

**Hugging Face 讨论 Credit 与西班牙语模型**：[hugging-face 频道](https://discord.com/channels/1238365980128706560/1241141471814488115/1243335428887806004) 讨论了待处理的 **HF Credit** 以及适用于西班牙语文本生成的模型——推荐了 **Mistral 7B** 和 **Llama 3** 模型。

**Credit 倒计时继续**：在 [replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1243453712182149150) 频道，预告了即将发布的关于 Credit 管理和分配的公告。

**Corbitt 的准则引发关注**：[kylecorbitt_prompt_to_model 频道](https://discord.com/channels/1238365980128706560/1242221891733946490/1243287896652517376) 的热心参与者讨论了 Kyle Corbitt 演讲中介绍的 Fine-tuning 方法和技术，包括 *[部署 Fine-tuned 模型的十诫](https://docs.google.com/presentation/d/1IIRrTED0w716OsU_-PL5bONL0Pq_7E8alewvcJO1BCE/edit#slide=id.g2721fb6713e_0_67)*。

**Axolotl 响应号召**：在 [workshop-2](https://discord.com/channels/1238365980128706560/1242223415293644930/1243277523316637817) 中，用户讨论了数据集、模型训练以及 Axolotl 中的故障排除。分享了一篇关于 **TinyLLama Fine-Tuning** 的博客文章，并推动将可观测性集成到 LLM 应用中。

**退出 Zoom，进入 Discord**：在 Zoom 聊天功能禁用后，[workshop-3](https://discord.com/channels/1238365980128706560/1242223458184597534/1243339106675724369) 的用户将讨论转移到了 Discord。

**Axolotl 的缓存难题引发困惑**：在 [axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1243286083022618664) 频道，解决了令用户沮丧的 Axolotl 缓存问题以及文件丢失的困惑。关于 sample packing 的讨论和一份关于 Tokenizer 陷阱的指南解决了有关效率和 Tokenization 的疑虑。

**加速迈向胜利**：[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1243291846415749283) 频道的用户解决了关于浮点数比较的困惑，修复了 Jarvislab 训练命令错误，并交流了学习模型加速的资源，重点关注 Fine-tuning 的最佳实践。

**与 Axolotl 并肩作战**：[wing-axolotl 频道](https://discord.com/channels/1238365980128706560/1242564077151326388/1243305377974587412) 在数据集 Template、预处理问题、Axolotl 配置方面进行了协作，并为最新的 Axolotl 更新提供了 PR 合并。他们深入研究了调试工具以及精确 Template 对训练成功的重要性。

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**蛋白质数据可视化达到新高度**：一个新的蛋白质可视化项目现在支持 3D 渲染，并包含了人类血红蛋白和核糖体蛋白的示例，项目详情可见 [GitHub](https://github.com/AstraBert/proteinviz/blob/main/examples.md)。

**使用 OpenAI 的 Whisper 进入 TranscriptZone**：一款利用 OpenAI 的 Whisper 转录 YouTube 视频及更多内容的全新转录应用已在 [Hugging Face Spaces](https://huggingface.co/spaces/tensorkelechi/vidtext) 上线。

**去中心化网络——不仅仅是一个梦想？**：一个构建去中心化互联网基础设施的项目通过调查寻求社区反馈，引发了关于数据收集伦理的讨论。

**Vision Transformers 深度查询**：一名成员寻求关于应用 Vision Transformers (ViT) 进行单目深度估计（monocular depth estimation）的资源，表示有意使用 ViT 开发模型，但讨论中未提供具体资源。

**Mistral 模型的量化困惑**：在 **Mistral v0.3 Instruct** 上使用 **bitsandbytes** 进行 8-bit 量化导致性能比 4-bit 和 fp16 更慢，这一令人费解的结果与减少位数计算预期的效率提升相矛盾。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 在 CSV 对决中超越 ChatGPT**：工程师们讨论认为 **Perplexity AI** 在 CSV 文件处理方面优于 **ChatGPT**，因为它允许直接上传 CSV。此外，推荐使用 **Julius AI** 进行数据分析，它利用 Python 并集成了 **Claude 3** 或 **GPT-4** 等 LLM。

- **用户冷落 Claude 3 Opus**：由于内容限制增加和感知到的效用下降，**Claude 3 Opus** 遭到冷遇，尽管 **GPT-4** 也有局限性，但仍被视为更好的选择。

- **质疑 Pro Search 的真实升级**：**Pro Search** 的升级引起了关注，用户讨论新的多步推理功能和 API 规范是真正的后端改进，还是仅仅是表面上的 UI 增强。

- **API 集成阐述**：围绕 **Claude** 与外部工具的 API 集成的对话引起了兴趣，同时分享了自定义函数调用、无服务器后端以及诸如 [Tool Use with Claude](https://docs.anthropic.com/en/docs/tool-use) 等文档。

- **AI 伦理：不仅仅是思想实验**：关于为 GPT 注入伦理监控能力的讨论被激发，揭示了其在职场沟通和法律辩护中的潜在应用，尽管仍有一些哲学难题尚待解决。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **关于 RTX 5090 显存的猜测达到顶峰**：关于传闻中拥有 **32GB VRAM 的 RTX 5090** 是否具有实际意义的辩论正热。引用了 [PC Games Hardware](https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/News/Geforce-RTX-5090-mit-32-GiB-und-drei-PCBs-1448073/galerie/3884555/) 上的潜在规格和图片，但部分成员对其真实性保持怀疑。

- **Stable Diffusion 与 AMD 的挑战**：用户提供了在 AMD 5700XT GPU 上安装 **Stable Diffusion** 的指导，建议从 [Craiyon](https://www.craiyon.com/) 等 Web 服务开始，以规避潜在的兼容性问题。

- **Stable Diffusion 3：承诺前的试用**：社区将 **Stable Diffusion 3** 与竞争对手 Midjourney 进行了对比，强调虽然 SD3 提供免费试用，但持续访问需要 **Stability** 会员资格。

- **对 Mobius 模型的期待与日俱增**：关于 DataPlusEngine 的新型 **Mobius 模型** 的公告引起了极大关注，因为它声称可以创建高效的基础模型。该模型在 [Twitter](https://x.com/DataPlusEngine/status/1793803117642854732) 上进行了预告，它既不是简单的基础模型，也不是现有模型的微调版本。

- **32GB VRAM：游戏规则改变者还是性能过剩？**：提到 32GB VRAM GPU 引发了关于 Nvidia 数据中心 GPU 销售策略可能转变的对话，考虑到拥有大容量显存的产品可能会如何影响市场对 H100/A100 系列的需求。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **PEFT 配置问题解决**：针对 PEFT 训练期间 `config.json` 缺失的问题，通过从基础模型配置中复制该文件已得到解决，用户确认修复成功。

- **Llama 克服 Bug 困扰**：**Llama 3** 模型的基础权重被描述为存在 "bug"，但 Unsloth 已实现相关修复。为了提升训练效果，建议使用 reserved tokens 并更新 tokenizer 和 `lm_head`。

- **System Prompt 提升 Llama 3 效果**：据观察，加入 System Prompt（即使是空的）也能增强 Llama 3 的微调效果。

- **Phi 3 模型激增**：随着 **Phi 3 模型** 的首次亮相，社区反响热烈，该模型已支持 medium 版本。社区讨论引导工程师关注博客文章和发布说明中的详尽细节。

- **Stable Diffusion 的阴森一面**：**Stable Diffusion** 产生的诡异伪影和不自然的声音克隆输出让用户感到吃惊，相关的讨论和经历已在 YouTube 视频和 Reddit 帖子中分享。

- **VSCode Copilot 方案推荐**：用户在 **random** 频道寻求本地 VSCode "copilot" 的建议，并得到了积极的回应和推荐。

- **Phi-3 的推理延迟问题**：使用 **Unsloth Phi-3** 时较慢的推理速度令一位用户感到困惑，该用户提供了一个 [Colab notebook](https://colab.research.google.com/drive/1LLWoaQrH8KFkQlE4ONwwtC4tC1-1It2X) 用于调查延迟原因，社区目前仍在努力寻找修复方案。

- **量化困境的剖析**：一名成员在量化自定义模型时面临挑战，在 **llama.cpp** 和 **Docker** 兼容性方面遇到了障碍，引发了关于解决方案的讨论。

- **模型算力的 VRAM 判定**：明确了 VRAM 需求：**Phi 3 mini 需要 12GB** 即可，但 **Phi 3 medium 必须配备 16GB**。对于繁重任务，建议考虑外部计算资源。

- **训练一致性的数据尽调**：强调了在训练和评估中使用一致数据集的重要性，并重点介绍了 **Unslothai 的公共数据集**，如 [Blackhole Collection](https://huggingface.co/collections/lamhieu/blackhole-66473b7feec034b4fb70818a)。

- **平台可能性与警示**：针对 **Unsloth** 是否支持旧款 Mac 的咨询得到了回复，确认开发重点在于 CUDA 和 GPU 使用，并为仅有 CPU 的设备提供了建议。

- **企业级专业知识扩展**：一位社区成员主动向 Unsloth 提供企业级专业知识，并对加入 Build Club 和 GitHub 的加速器表示赞赏，暗示了 Unsloth 未来发展的协同潜力。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**关于 AI 理解能力的智力辩论**：社区就 LLM 对概念的真正“理解”展开了深入讨论，**可解释性研究（interpretability research）**被视为重要的经验证据。怀疑论者认为目前的努力尚显不足，并引用了 **Anthropic** 关于映射大语言模型思维的相关工作。

**Llama 湖中的生物**：一项旨在增强 **Llama 模型** 的技术探索集中在编写一个能够管理 **function calls** 的脚本上，并以 **Hermes Pro 2** 的方法作为灵感。另一个咨询则围绕在 3080 GPU 上实现 **Llama 3 LoRA** 技术展开。

**数字维度中的现实探索**：在关于 **Nous 和 WorldSim** 的对话中，成员们探讨了 **NightCafe** 和多维 AR 空间在映射复杂 AI 世界中的潜在应用。在**音频可视化器**中的梦幻探索和奇特的 **ASCII art** 表现形式，突显了 AI 驱动模拟的创意用途。

**筛选 RAG 数据**：提倡模型将**内部知识**与**检索增强生成（RAG）**相结合成为热门话题，并提出了关于如何处理矛盾和解决冲突的问题。强调用户评估被认为是必不可少的，特别是对于复杂的查询案例。

**AI 微调：精准度胜过花哨技巧**：社区讨论赞扬了 **Mobius 模型** 在**图像生成**方面的卓越表现，并期待其开源版本和阐释性论文的发布。此外，还提到了 Hugging Face 的 `PyTorchModelHubMixin` 可以简化模型共享，但受限于 **50GB 的大小限制**（在不分片的情况下）。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **JAX vs. PyTorch/XLA: TPU 对决**：**JAX** 与 **PyTorch/XLA** 在 TPU 上的性能对比引发了关于 **warmup times** 和 **blocking factors** 等基准测试细微差别的辩论。GPT-3 的训练成本从 **450 万美元大幅下降至 2024 年预计的 12.5 万至 100 万美元**，这一点受到了关注。讨论结合了来自多位贡献者的 **TFLOP rates** 和 **GPU-hour pricing**，并链接到一篇 [Databricks 博客文章](https://www.databricks.com/blog/turbocharged-training-optimizing-databricks-mosaic-ai-stack-fp8)。

- **扩展与教学 LLMs**：在研究论坛中，**Chameleon** 模型因其在多模态任务中的强劲表现而受到关注，而 **Bitune** 则承诺提升 LLM 的 zero-shot 性能（[Bitune 论文](https://arxiv.org/pdf/2405.14862)）。讨论质疑了 **JEPA** 模型对于 AGI 的可扩展性，并批评了 **RoPE** 的上下文长度限制，引用了相关[论文](https://arxiv.org/pdf/2405.14591)。

- **Emergent Features 困扰 LLM 爱好者**：链接了 Tim Dettmers 关于在 transformer 推理中保持性能的高级量化方法的研究，包括他的 emergent outliers 概念，以及通过 [bitsandbytes 库](https://huggingface.co/blog/hf-bitsandbytes-integration) 与 Hugging Face 的集成。关于 emergent features 的讨论围绕着它们是模型的“DNA”这一观点展开，并探讨了其对相变（phase transitions）的影响。

- **技术调整与 LM 评估简报**：在 **lm-thunderdome** 中，工程师们分享了在 **vllm 模型** 中设置 seed 的实用技巧，使用 `lm_eval --tasks list` 获取**任务列表**，以及处理 **BigBench** 任务名称变更对 Accelerate 等框架造成的内存问题。建议通过查阅 `lm-eval/tasks` 文件夹来定位任务，以便更好地组织。

- **协作呼吁**：发出了扩展 **Open Empathic** 项目的呼吁，并提供了一个用于贡献电影场景的 [YouTube 指南](https://youtu.be/GZqYr8_Q7DE) 以及该项目的链接。鼓励进一步的协作，强调了社区努力在增强项目方面的必要性。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**GPU 历险记**：工程师们讨论了将小型模型加载到 GPU 上的挑战，一些人青睐 *llama3, mistral instruct* 和 *cmdrib* 等模型。同时，据报道，在某些应用中，使用较低的量化（如 *llamas q4*）比使用 q8 等较高的量化效果更好，反驳了“越大越好”的观念。

**下一代模型即将来临**：模型领域的更新通知了一个 **35B 模型** 的发布，并正在进行测试以确保 LM Studio 的兼容性。针对不同规模模型的优化也是一个话题，重点关注 **Phi-3 small GGUFs** 及其效率。

**服务器与设置**：硬件讨论包括利用 **llama.cpp** 及其最近的 RPC 更新进行**分布式推理**，尽管目前尚不支持量化模型。还探索了使用配备 **RTX 4060 Ti 16GB** 的廉价 PC 集群进行分布式模型设置的实验性构建，以及可能的网络限制。

**实现多语言凝聚力**：Cohere 模型现在将其能力扩展到了 **23 种语言**，正如广告所言，**aya-23 量化版本**已开放下载，但 ROCm 用户必须等待更新才能体验。 

**Stable Diffusion 被排除在外**：LM Studio 澄清其专门处理语言模型，不包括像 Stable Diffusion 这样的图像生成器，同时处理旧款 GPU 上的 CUDA 问题，并推广 **Julius AI** 等服务以缓解用户体验方面的困扰。



---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **梯度范数难题**：将 Batch Size 从 32 更改会导致梯度范数突然飙升，从而干扰训练。一个 [Pull Request](https://github.com/karpathy/llm.c/pull/456) 通过防止 Fused Classifier 中的索引越界解决了这个问题。
  
- **Int4 和 Uint4 类型需要关注**：一位成员指出 PyTorch 中许多函数缺乏对 **int4** 和 **uint4** 数据类型的实现，相关的 [讨论帖](https://dev-discuss.pytorch.org/t/supporting-new-dtypes-in-pytorch/1833) 指出了在类型提升（Type Promotion）和张量操作方面的限制。

- **直播代码预警——Scan 算法成为焦点**：Izzat El Hajj 将主持一场关于 Scan 算法的现场编程会议，该算法对于 Mamba 等 ML 算法至关重要。会议定于 `<t:1716663600:F>`，有望为爱好者们带来一次深入的技术探讨。

- **CUB 库查询与 CUDA 细节**：成员们深入讨论了从 CUDA CUB 库代码的运行机制到在不使用 cuBLAS 或 cuDNN 的情况下触发 Tensor Cores 等话题，并重点推荐了 [NVIDIA 的 CUTLASS GitHub 仓库](https://github.com/NVIDIA/cutlass/tree/main) 和 [NVIDIA PTX 手册](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) 等资源。

- **FineWeb 数据集难题**：处理 FineWeb 数据集非常耗费存储空间，磁盘占用达到 70 GB，内存占用高达 64 GB，这暗示了在数据处理任务中需要更好的优化或更强大的硬件配置。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Python 库相较于 Mojo 更倾向于 C**：关于将 Python 库移植到 Mojo 的可行性和准备工作有一场激烈的讨论。考虑到 Mojo 不断演进的 API，人们担心这会给维护者带来过大压力。成员们讨论了将目标转向 C 语言库是否是一个更直接且务实的尝试。

**Rust 的安全性吸引力并未削弱 Mojo 的潜力**：Mojo 并不打算取代 C，但 Rust 的安全性优势正在影响工程师们对 Mojo 在不同场景下应用的思考。正在进行的讨论涉及了可以使 Mojo 开发受益的 Rust 概念。

**紧跟 Nightly 版本 Mojo 的步伐**：使用 Nightly 版本的 Mojo 在 MacOS 上的 BlazeSeq 性能表现出与 Rust 的 Needletail 相似的潜力，引发了关于跨平台效率的讨论。[更新日志](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 中提到的快速 Nightly 更新让社区能够紧跟这门不断发展的语言。

**对 Modular 机器人机制的好奇**：有人询问了 "ModularBot" 的底层技术，虽然没有提到具体模型，但机器人给出了一个生动的回复。另外，讨论了在 Mojo 中进行 ML 模型训练和推理的可能性，并提到了 Max Engine 作为 NumPy 的替代方案，尽管目前还没有完整的训练框架。

**编译时困惑与对齐问题**：从内存中的布尔值对齐到编译时函数问题，这些都引起了用户的关注。临时解决方案和官方 [错误报告](https://github.com/modularml/mojo/issues/2813) 凸显了社区驱动排错的重要性。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **忠于 LaTeX 的 LLM**：在格式排版领域，用户对 GPT 强烈倾向于默认使用 LaTeX 而非请求的 Typst 代码感到沮丧，这揭示了 LLM 似乎坚持某些特定的编码语法偏好。
  
- **Microsoft Copilot+ 与 Leonardo 之争**：社区讨论集中在 Microsoft Copilot+ PC 在“草图转图像”等创意任务中的价值，而一些成员则鼓励尝试具有类似功能的 [Leonardo.ai](https://leonardo.ai)。

- **对 AI 效率的渴求**：人们对 AI 造成的环境代价表示担忧，引用了一篇关于 AI 模型训练过程中大量耗水的 [Gizmodo 文章](https://gizmodo.com/chatgpt-ai-water-185000-gallons-training-nuclear-1850324249)，引发了关于需要更环保的 AI 实践的讨论。

- **迭代胜过创新**：关于通过迭代优化来增强 LLM 性能的对话非常活跃，并提到了像 AutoGPT 这样处理迭代的项目，尽管这伴随着更高的成本。

- **注入智能的提议是否言过其实？**：公会成员思考了在 ChatGPT 中嵌入法律知识的可行性和潜力，甚至考虑到了 6.5 亿美元的估值，尽管关于这一大胆断言的详细观点还比较有限。



---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**LangChain CSV Agent 深度解析**：工程师们探讨了在 **SequentialChain** 中使用 **LangChain's CSV agent**，并讨论了[如何自定义输出键](https://python.langchain.com/docs/modules/chains/foundational/sequential_chains)（如 `csv_response`）。提到了 SQL agents 在处理多表查询时面临的挑战，指出存在 token 限制和 LLM 兼容性问题，并引导至 GitHub [提交 issue](https://github.com/langchain-ai/langchain/issues)。

**AI 展示引发关注**：[OranAITech 推特发布了](https://twitter.com/OranAITech/status/1793684085056942412?t=AVjC2GpAdrT-LqwMEzv0nQ&s=19)他们最新的 AI 技术，同时 **everything-ai v2.0.0** 宣布了包括音频和视频处理能力在内的功能，并提供了 [repository](https://github.com/AstraBert/everything-ai) 和 [documentation](https://astrabert.github.io/everything-ai/)。

**揭秘 VisualAgents**：通过 YouTube 分享了 **Visual Agents platform** 的演示，展示了其利用 LangChain 的能力来简化 SQL agent 创建和无需编码构建简单检索系统的潜力。两段特定视频展示了其工作流：[SQL Agent](https://youtu.be/_3crxBzVg3A?si=r2rDA19q-fHm7h9N) 和 [Simple Retrieval](https://youtu.be/prOjBQQgKlU?si=jDt53koCl6lT6BoM)。

**EDA GPT 印象展示**：通过 [LOVO AI](https://genny.lovo.ai/share/d6b58f0d-fc46-4aa7-a65e-fa0f9a684f01) 链接了一个 **EDA GPT** 的演示，包括一段展示其各项功能的五分钟概览视频。该演示突显了这款 AI 工具的多功能性。

**教程预告**：教程频道的一条消息提供了 business24.ai 内容的 [YouTube 链接](https://youtu.be/gflsu_6R_8g)，尽管其相关背景尚未披露。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **盗版并非万灵药**：尽管有人幽默地建议 The Pirate Bay 可能成为分享 AI 模型权重的避风港，但成员们对此表示怀疑，并强调其他国家更友好的 AI 政策环境可能会取而代之。

- **日本在 AI 领域采取高端路线**：参与者注意到日本在 AI 发展方面的积极立场，并引用了通过 [推文](https://x.com/DataPlusEngine/status/1793817514956259460) 分享的一篇 **paper**，内容关于无需大规模预训练即可创建新的基础 diffusion models，展示了一种涉及临时破坏模型关联的策略。

- **探究中毒恢复协议**：提到了一项由 fal.ai 进行的涉及中毒模型恢复方法的 **合作研究**，预计研究结果将从经验上证实该恢复方法。成员们对 AI 生成图像的美学表达了保留意见，特别是像 Mobius 这样的模型与 MJv6 等前辈相比所呈现的“高对比度外观”和伪影。

- **Claude 映射破解代码**：Anthropic 的 **research paper** 详细剖析了 Claude 3 Sonnet 的神经图谱，阐述了对概念激活的操纵，可在其 [研究页面](https://www.anthropic.com/research/mapping-mind-language-model) 阅读。围绕此类激活的潜在商业化引发了辩论，同时也存在对商业影响的担忧，这令 AI 从业者感到沮丧。

- **怀旧 AI 视觉愿景**：一位成员回忆了从 Inception v1 等早期 AI 视觉模型到如今复杂系统的演变，认可了 DeepDream 在理解神经功能方面的作用。此外，还讨论了神经网络中稀疏性的好处，描述了使用 L1 norm 实现稀疏性，以及在高维层中典型的 300 个非零维度。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **见面会提醒：名额有限**：即将于周二举行的 **LlamaIndex meetup** 仅剩少量名额，由于名额有限，鼓励爱好者们尽快[预订名额](https://twitter.com/llama_index/status/1793739449127583964)。

- **MultiOn 结合 LlamaIndex 实现任务自动化**：**LlamaIndex** 已与 AI Agent 平台 **MultiOn** 结合，通过代表用户操作的 Chrome 浏览器实现任务自动化；点击[此处](https://twitter.com/llama_index/status/1793764970024570979)查看演示。

- **RAGApp 发布：无需代码即可设置 RAG 聊天机器人**：新推出的 **RAGApp** 简化了通过 Docker 容器部署 RAG 聊天机器人的流程，使其可以轻松部署在任何云基础设施上，并且它是开源的；在此处配置您的模型提供商：[此处](https://twitter.com/llama_index/status/1794030544415818062)。

- **解决 PDF 解析难题**：社区认可 **LlamaParse** 是一个从 PDF（尤其是表格和字段）中提取数据的可行 API，它利用 GPT-4o 模型来增强性能；**Knowledge Graph Indexing** 的挑战也是一个话题，强调了手动和自动（通过 `VectorStoreIndex`）策略的必要性。

- **PostgresML 与 LlamaIndex 联手**：**Andy Singal** 分享了关于将 **PostgresML** 与 **LlamaIndex** 集成的见解，并在 Medium 文章 [《Unleashing the Power of PostgresML with LlamaIndex Integration》](https://medium.com/ai-advances/unleashing-the-power-of-postgresml-with-llamaindex-integration-9eadee223939) 中详细介绍了这一协作，获得了社区的积极评价。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Phi-3 Medium 128k Instruct 发布**：OpenRouter 推出了 **Phi-3 Medium 128k Instruct**，这是一个强大的 140 亿参数模型，并邀请用户查看[标准版](https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct)和[免费版](https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct:free)变体，并参与其有效性的讨论。

- **Wizard 模型获得魔力提升**：**Wizard 模型**表现出改进，响应更加迅速且富有想象力，但仍需注意避免重复段落。

- **关注 Phi-3 Vision 和 CogVLM2**：围绕 **Phi-3 Vision** 的热情高涨，分享了测试链接如 [Phi-3 Vision](https://ai.azure.com/explore/models/Phi-3-vision-128k-instruct/version/1/registry/azureml)，并建议在 [CogVLM-CogAgent](https://huggingface.co/spaces/THUDM/CogVLM-CogAgent) 使用 **CogVLM2** 处理以视觉为中心的任务。

- **Llama 3 Prompt 自动转换**：澄清了发送给 **Llama 3** 模型的 Prompt 会通过 OpenRouter 的 API 自动转换，从而简化流程，但手动 Prompt 仍作为一种替代方法保留。

- **Gemini API 的烦恼**：用户报告了 **Gemini FLASH** API 的问题，如空输出和 Token 消耗，这被认为是模型本身的问题。Google 每日 API 使用限制的出现引起了人们对这可能如何影响 OpenRouter 的 Gemini 集成的关注。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **显微镜下的 LLM 评估**：一篇关于大语言模型（LLM）评估实践、排行榜重要性以及细致的非回归测试的 [Hugging Face 博客文章](https://huggingface.co/blog/clefourrier/llm-evaluation) 引起了成员们的关注，强调了此类评估在 AI 发展中的关键作用。

- **AI 对搜索引擎操纵的回应**：一起涉及网站投毒影响 Google AI 概览的事件引发了围绕安全和数据完整性的讨论，包括通过 [Mark Riedl 的推文](https://x.com/mark_riedl/status/1793375699967054334) 报道的自定义搜索引擎浏览器绕过等解决方法。

- **AI 是让开发民主化还是引发了可靠性问题？**：GitHub CEO Thomas Dohmke 关于 AI 在简化编码中作用的 [TED 演讲](https://youtu.be/nv9WwHpOKEg?si=mVApo6UnrtJ9ExH6) 引发了对其可靠性的争论，尽管 AI 驱动的 UX 改进加快了编码过程中的问题解决。

- **多元化奖学金以弥合差距**：针对面临参加即将举行的 AI Engineer World's Fair 财务障碍的多元化背景工程师，官方宣布了多元化奖学金。有兴趣的申请人应在 [申请表](https://docs.google.com/forms/d/e/1FAIpQLScff_RUv-fIKfdj_2HcHtk96iy45GD0BWLByGxqdBqvcepDHg/viewform?usp=sf_link) 中对论文问题提供*简明*的回答。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **无需信用卡的税务轶事**：Nathan Lambert 解释了一场发票混乱，意识到由于转售证书（resale certificates）的存在，无需信用卡进行税务结算具有其合理性。

- **Golden Gate AI 备受关注**：[Anthropic AI](https://x.com/anthropicai/status/1793741051867615494?s=46) 的实验产生了 "Golden Gate Claude"，这是一个一心只关注金门大桥的 AI，因其在 claude.ai 上的公开互动性而引发热议。

- **Google 的 AI 失误**：Google 未能有效利用反馈以及过早部署 AI 模型，引发了关于这家科技巨头公关挑战和产品开发困境的讨论。

- **反驳数据集误解**：Google 的 AI 团队反驳了关于使用 LAION-5B 数据集的说法，指出他们使用的是更优越的内部数据集，正如[最近的一条推文](https://x.com/giffmana/status/1793906145310228538)所提到的。

- **Nathan 分享知识干货**：针对 AI 爱好者，Nathan Lambert 上传了进阶的 [CS224N 讲义幻灯片](https://docs.google.com/presentation/d/1on5xTePaUYg47vui3dUr0Lp6GUXmOXmhceNJLXRbGsE/edit)。此外，与会者还获悉即将发布会议录像，但尚未公布具体发布日期。



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **GQA 在 CMDR 模型中获得关注**：讨论显示，**Grouped Query Attention (GQA)** 存在于 "cmdr+" 模型中，但不存在于基础的 "cmdr" 模型中，这标志着它们在规格上的重要区别。
- **智能注意力机制提升 VRAM 效率**：工程师们注意到，虽然 **GQA** 不提供线性缩放，但与指数缩放相比，它代表了一种改进的缩放方法，对 **VRAM** 使用产生了积极影响。
- **样本打包（Sample Packing）效率提升**：一个新的 **GitHub pull request** 展示了样本打包效率提升了 3-4%，有望在分布式环境下实现更好的资源管理，链接见[此处](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1619)。
- **学术成就获得认可**：一位成员合著的期刊文章已在 **Journal of the American Medical Informatics Association** 上发表，强调了高质量、跨领域数据对医学语言模型的影响，文章可在此处[获取](https://doi.org/10.1093/jamia/ocae120)。
- **社区庆祝学术成功**：社区通过个人祝贺信息对同行发表的作品表示支持，在 AI 领域内营造了一种认可学术贡献的文化。



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**SB-1047 引发技术动荡**：工程师们对 **SB-1047** 法案的影响表示深切担忧，称其对小型 AI 参与者有害，并将这种情况比作在其他行业观察到的监管俘获（regulatory capture）。

**Perplexity 和 Arc：生产力工具展示**：社区重点介绍了辅助工作流的工具，分享了关于 [SB-1047 的 Perplexity AI 搜索结果](https://www.perplexity.ai/search/SB-1047-Senate-2kZmFYHoTxe.rWUYat4B2A)以及 Arc Browser 的新功能 “Call Arc”，该功能简化了在线寻找相关答案的过程，附带[信息链接](https://arc.net/e/C56904FA-1C75-4D77-9A87-E7F1A52529CD)。

**安装问题引发排查**：用户在使用 pip 安装 **Typer** 库时遇到问题，引发了关于是否遵循了设置步骤（如在 `poetry run` 之前执行 `poetry install`）或是否使用了虚拟环境的疑问。



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

**Twinny 作为虚拟副驾驶起飞**：开发者正在将 [Twinny](https://github.com/rjmacarthy/twinny) 与 LM Studio 集成，作为强大的本地 AI 代码补全工具，支持在不同端口上运行多个 llamafile。

**关于 Embedding 端点的澄清**：澄清了 `/v1/embeddings` 端点不支持 `image_data`；根据 [pull request #4681](https://github.com/ggerganov/llama.cpp/pull/4681)，图像应使用 `/embedding` 端点。

**Mac M2 在 continue.dev 中遭遇对手**：性能观察指出，在使用 llamafile 执行时，continue.dev 在 Mac M2 上的运行速度比旧款 Nvidia GPU 慢。

**训练你自己的 LLM**：对于那些希望构建和训练自定义 LLM 的用户，社区建议使用 [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) 进行训练，并提醒 llamafile 是为推理而非训练设计的。

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **服务器中回荡着感激之情**：一位用户向团队表达了衷心的*感谢*，展示了用户对团队支持或开发工作的认可。
- **对扩展规模模型的关注**：社区中流传着关于 **104B 版本** 模型是否会加入家族树的讨论，但目前尚未有明确答案。
- **Langchain 连接缺失**：用户提出了关于 **Langchain** 与 Cohere 集成的问题，寻求关于其当前可用性和实现状态的指导。
- **模型尺寸之谜**：用户正在寻求澄清 Playground 中的 **Aya model** 是指 8B 还是 35B 版本，这表明理解模型规模对应用至关重要。
- **错误排查角落**：诸如 **ContextualCompressionRetriever** 的 `ValidationError` 和 **403 Forbidden error** 等问题，标志着工程师们正在进行活跃的调试和技术问题解决，提醒人们 AI 开发中常见的挑战。

---

## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

**AI 喜剧之夜反响热烈**：用户分享的一段 AI 生成的单口喜剧作品获得了积极的惊喜，表明 AI 在模仿幽默和进行娱乐表演方面的能力有所提升。

**关于 AI 应用的探索性查询**：用户询问 [Ud.io](https://www.udio.com/songs/vsNF2nbsy646jGt348mdFG) 的功能是否超出了生成喜剧的范畴，表现出对其功能边界的好奇。

**声音变换展示**：一位用户通过分享一段原始音频的变体、恶魔化版本，展示了 [Suno](https://suno.com/song/e6b62587-4345-44fb-85c7-c51f932df655) 灵活的音频修改功能。

**对音频工程知识的渴望**：用户表达了对获取演示中此类音频修改技能的兴趣，这对于对声音处理感兴趣的 AI 工程师来说是一项宝贵的技能。

**偏好简洁沟通**：对一个问题的单字回复“No”突显了对简洁回答的偏好，这或许反映了工程师对直接、务实沟通的渴望。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **寻找统一的事件追踪器**：一名成员强调了对与 Google Calendar 兼容的活动日历的迫切需求，以确保不会错过任何社区活动。该系统的缺失是社区内关注的一个问题。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **新数据集发布通知**：用户 datarevised 引用了一个新数据集，并提供了更多详情的链接：[DataPlusEngine Tweet](https://x.com/DataPlusEngine/status/1793803117642854732)。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **基准测试驱动创新**：诸如 [GLUE](https://arxiv.org/abs/1804.07461)、[MMLU](https://arxiv.org/abs/2009.03300) 和 [GSM8K](https://arxiv.org/abs/2110.14168) 等评估基准对 AI 研究进展至关重要，而“execution based evals”在动态任务中面临特殊挑战，正如成员们分享的一篇 [博客文章](https://www.jasonwei.net/blog/evals) 中所讨论的那样。
  
- **对 GPT-5 的期待**：一次谈话引发了公会成员的猜测，认为 **GPT-5** 可能会在 2024 年首次亮相，并向“Agent 架构”迈进，正如一条 [传闻推文](https://x.com/rohanpaul_ai/status/1793956355897724973?s=46&t=90xQ8sGy63D2OtiaoGJuww) 所暗示的那样。

- **探讨 AI 在音乐中的角色**：成员们探讨了关于 **Suno** 等 AI 创作音乐的版权问题，思考了 Meta 的 **Audiocraft** 的能力，并讨论了法律影响以及促进创作自由的开源工作，包括 [GitHub](https://github.com/betweentwomidnights/gary-backend-combined) 上的 **gary-backend-combined**。

- **Python 开发者为 NumPy 2.0 做准备**：人们对 **NumPy 2.0** 的期待日益高涨，成员们还开玩笑说这可能对依赖管理产生影响，正如一篇 [Twitter 帖子](https://x.com/cgarciae88/status/1794019900119236874) 所提到的。

- **xAI 获得巨额投资**：在投资新闻方面，**xAI** 获得了 60 亿美元的 B 轮融资，目标是快速提升其模型的能力，正如其 [公告文章](https://x.ai/blog/series-b) 所示。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **社区集思广益修复 Nightly 版本困扰**：在成员们指出 **Mojo VSCode extension** 仅对安装后的文件有效，且自 24.3 版本以来存在 Bug 后，解决方案包括关闭并重新打开代码以及频繁重置。与之相反，**Mojo compiler** 的 Nightly 更新（`2024.5.2505` 到 `2024.5.2705`）带来了诸如 LSP 中的变量重命名和新的 `tempfile` 模块等增强功能，尽管它们在 GitHub 上的现有 PR（如 [#2832](https://github.com/modularml/mojo/pull/2832)）中引入了测试问题。

- **在 Mojo 中实现 ZIP 的禅意**：在 Mojo 中复制 Python 的 `zip` 和 `unzip` 函数时遇到的困难，引发了关于如何声明根据变长参数列表返回元组（tuples）的讨论。对话揭示了 Mojo 使用新的 `ref` 语法实现自动解引用迭代器的潜力，旨在简化实现并减少显式的解引用操作。

- **不同处理器的性能表现差异**：一位成员在 Mojo 中优化 CRC32 计算的努力导致了不同核心类型之间的性能差异；由于 L1 cache 限制，紧凑型实现在处理较大字节大小时表现落后，但能效核（efficiency cores）则更青睐紧凑版本。基准测试元数据和版本化文件位于 [fnands.com](https://github.com/fnands/fnands.com/blob/main/blog/2024/mojo-crc-calc/crc.mojo) 以及 [上述内容的 Nightly 版本测试](https://github.com/fnands/fnands.com/blob/main/blog/2024/mojo-crc-calc/crcn.mojo)。

- **关于 Mojo 潜力的思考**：关于 **Mojo** 的一般性讨论涵盖了与 Python 相比的函数行为差异、可选类型（optional types）的处理、LinkedLists 的实现，以及对 Mojo 尚未实现的反射（reflection）能力的思考。成员们就使用 `UnsafePointer` 和 `Optional` 进行高效初始化交换了意见。

- **科技巨头的巨额交易**：AI 社区消化了 Elon Musk 的 AI 初创公司 xAI 获得 60 亿美元 B 轮融资的消息，正如 [TechCrunch 文章](https://techcrunch.com/2024/05/26/elon-musks-xai-raises-6b-from-valor-a16z-and-sequoia/) 所述，这使 xAI 能够直接对抗 OpenAI 和 Microsoft 等 AI 领跑者，引发了围绕 AI 技术商业化的讨论。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**新成员加入：Phi-3 模型上线**：Microsoft 的 [phi-3-medium-128k-instruct](https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct) 和 [phi-3-mini-128k-instruct](https://openrouter.ai/models/microsoft/phi-3-mini-128k-instruct) 模型现已上线，同时 [llama-3-lumimaid-70b](https://openrouter.ai/models/neversleep/llama-3-lumimaid-70b) 模型还提供 4.3 折（57% discount）优惠。

**速率限制迷宫解析**：关于 OpenRouter 上 **rate limiting**（速率限制）的挑战引发了激烈讨论，强调了理解信用余额如何影响请求速率的重要性，详见 [OpenRouter 文档](https://openrouter.ai/docs#rate-limits-and-credits-remaining)。

**Modal 混乱：当信用额度与速率限制冲突时**：Modal 回退功能出现了一个令人困惑的问题，即尽管信用余额充足，仍触及了速率限制。社区建议监控免费请求，并在限制临近时考虑停用免费模型。

**AI 的自我审查困境削弱了吸引力**：爱好者们表示担心，Claude 自我审查模型中更严格的护栏和更高的拒绝率导致了不够拟人化的体验，这可能导致使用量下降。

**视觉模型分析：性能 vs 价格**：话题转向了视觉模型的性能，特别是 Gemini 的 OCR 能力，并对其与传统视觉服务相比的性价比表示认可。对话还强调了通过 RunPod 和 Vast.ai 使用 GPU 比使用 Google Cloud 和 Amazon Bedrock 等主流云服务更便宜。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **CCS 表现未达预期**：Eleuther 团队对 Collin Burns 的 _Contrast Consistent Search (CCS)_ 方法进行的后续研究在泛化能力提升方面未取得预期效果，正如 [Quirky Models 基准测试](https://arxiv.org/abs/2312.01037) 所示。他们在详细的 [博客文章](https://blog.eleuther.ai/vincs/) 中分享了该方法及其平庸的结果，这种透明度受到了称赞。

- **最优模型提取策略引发热议**：工程师们就 **RAG** 在从自定义库中提取数据方面是否优于 **LLM** 的 finetuning 展开了辩论，结论是 RAG 可能更好地保留信息。在 Hugging Face 上发布的 **ThePitbull 21.4B 模型** 引起了轰动，但一些人对其宣称的接近 70B 模型的性能表示怀疑。

- **数据复制故障排除**：AI 程序员在复制 Pythia 数据时遇到了 tokenizer 难题，解决方案包括使用 `batch_viewer.py` 以及通过 `MMapIndexedDataset` 正确处理数据类型。正如 [Pythia 仓库](https://github.com/EleutherAI/pythia#exploring-the-dataset) 中所指出的，虽然这个过程被比作“黑魔法”，但对于正确解析数据集是必要的。

- **利用新技术突破极限**：工程师们深入讨论了通过梯度扰动（gradient perturbation）方法来分配权重更新，以及 Transformer 在参数化知识上进行隐式推理的潜力。一篇新的“schedule-free”优化论文引起了关注，该论文建议在不增加额外超参数的情况下进行迭代平均，其表现可能优于传统的学习率调度（learning rate schedules）。

- **量化困境**：在追求效率的过程中，一场关于小模型是否能在某些情况下超越大型量化模型的讨论随之展开。引用 [ggerganov 的 Twitter](https://x.com/ggerganov/status/1666087050725199872) 暗示了量化方法的潜力，激发了关于模型大小与性能平衡的激烈辩论。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **RAG 革命：企业级流水线即将到来**：**LlamaIndex 团队**宣布将与 AWS 合作举办研讨会，展示如何使用 AWS Bedrock 和 Ragas 构建**企业级 RAG 流水线**。该活动旨在提供关于将 Bedrock 与 LlamaIndex 集成以优化 RAG 系统的见解，目前已开放注册 [点击此处报名](https://lu.ma/x8unmku0)。

- **提升检索能力的创新**：讨论重点关注了 **Vespa 的集成** 以增强混合搜索（hybrid search）能力，以及 Jayita B. 撰写的关于使用 Llama3 和 GroqInc 创建**快速响应 RAG 聊天机器人**的高级指南。此外，还提到了一种通过 **gpt4o** 等模型生成的结构化注释对图像进行索引的创新方法。

- **文件组织能力的提升**：**LlamaFS** 作为一种自动组织杂乱目录的新工具发布，这可能会引起那些寻求整洁高效文件管理解决方案的人的共鸣。

- **技术故障排除成为焦点**：AI 工程师们正在解决围绕 **Llama Index reAct** 的问题，包括通过 `max_iterations` 设置进行权衡，以及通过对齐软件包版本来克服导入错误。HTML 解析通常比 PDF 文件需要更多的自定义代码，而 PDF 可以利用依赖项较少的高级分块工具。

- **使用 Pydantic 实现 LlamaIndex 结构化输出**：关于在 LlamaIndex 中使用 **Pydantic 模型** 的指南标志着向结构化输出集成迈进了一步，指向了系统更广泛的应用和可用性。对改进检索器文档的呼吁凸显了社区在不同项目中增强对 **BM25** 和 **AutoRetrieval** 模块理解与应用的动力。

## [Cohere](https://discord.com/channels/954421988141711382) Discord

**Aya-23 登场**：工程师们讨论了 **Aya-23 的多语言能力**，并将其与 **Command R/R+** 进行了对比，暗示其性能更优，但对其针对英语的效率提出了疑问。他们还指出 **Aya-23-35b** 是 **Command R** 的微调版本，并提供了[技术报告](https://drive.google.com/file/d/1YKBPo61pnl97C1c_1C2ZVOnPhqf7MLSc/view)以获取更多细节。

**移动端隐私 vs LLM 局限性**：大家达成共识，认为**手机端 LLM** 尚未成熟到可以在移动应用中进行私密的本地执行，特别是对于通常与 **RAG 移动应用** 相关的任务。

**机器人创新蓬勃发展**：一位社区成员在 LinkedIn 上展示了一个游戏机器人，因其与 **Cohere Command R** 的集成而备受关注；同时，Discord 的 "Create 'n' Play" 机器人号称拥有 ***“超过 100 款引人入胜的文字游戏”***，并*通过 AI 增强社交参与度*。

**Prompt 的适配与集成**：公会确认 **Aya-23 支持系统提示词 (system prompts)**，并分享了关于如何使用特定 Token（如 `<|USER_TOKEN|>` 和 `<|CHATBOT_TOKEN|>`）来适配 **Command R** 提示词以实现有效运行的见解。

**OneDrive 同步解决方案**：针对关于 **OneDrive 连接器** 的咨询，推荐了一个 [SharePoint 连接器](https://github.com/cohere-ai/quick-start-connectors/tree/main/sharepoint)，它可能满足类似的集成需求。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

**AI 建议跳桥的闹剧**：成员们分享了一个关于 Google AI 给出“跳桥治疗抑郁症”危险建议的幽默段子，提到了 Reddit 建议的误导性。还分享了一个关于这一失误的相关梗图。

**ConvNeXt 获得优化**：关于 [ConvNeXt 论文](https://arxiv.org/abs/2405.15738) 的热烈讨论赞扬了其高效处理高分辨率图像的能力，这有可能减少过多的视觉 Token 生成，并简化高分辨率任务的优化。

**从红石到神经网络**：展示了数据集和 AI 工具的创新用途，包括来自 [archive.org](https://archive.org/details/arxiv-bulk?sort=-publicdate) 的出版物 PDF 和 TeX 源码数据集，以及一段演示如何用红石 (Redstone) 创建神经网络的 [YouTube 视频](https://youtu.be/DQ0lCm0J3PM?si=5Is7OMnqRhZb-ZAo)。

**AI 预训练中的模型增长堆叠**：一篇 [arXiv 论文](https://arxiv.org/abs/2405.15319) 强调了深度方向堆叠 (depthwise stacking) 是 Large Language Models (LLMs) 高效预训练中模型增长的一种有效方法，引起了广泛关注，解决了预训练过程中关键的速度和性能挑战。

**PyTorch 持久化中的陷阱**：学习领域的讨论集中在解决训练-验证集划分的随机性以及模型重新加载时 Loss 不一致的问题。具体而言，PyTorch 中正确保存优化器状态 (optimizer states) 被指出是避免 Loss 爆炸的关键。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Llama3 德语化**：新的 **Llama3-German-8B** 模型将 Meta 的 Llama3-8B 能力扩展到了德语，在 650 亿个 tokens 上进行了训练，且 English 性能损失微乎其微；详情请见 [Hugging Face](https://huggingface.co/DiscoResearch/Llama3-German-8B)。然而，值得注意的是，与其他语言模型不同，该训练**省略了英语数据重放 (English data replay)**，这引发了关于其有效性的辩论。

- **量化怪癖与谜题**：Llama3 的量化版本 **GGUF** 在基准测试中表现平平，暗示了潜在的问题，可在 [Llama3-DiscoLeo-Instruct-8B-32k-v0.1-GGUF](https://huggingface.co/cstr/Llama3-DiscoLeo-Instruct-8B-32k-v0.1-GGUF) 查看。同时，关于参数设置和奇怪输出的讨论暗示了运行这些模型的复杂挑战，例如 max tokens 和 engine 的选择。

- **Cohere 的多语言飞跃**：**Cohere 的 Aya-23-35B 模型**虽然在许可方面有所限制，但现在支持 23 种语言，这表明了业界对强大多语言模型日益增长的趋势和兴趣。一个相关的用于翻译的 **ShareGPT 格式数据集**引发了关于质量和过滤的讨论，托管在 [这里](https://huggingface.co/datasets/sroecker/aya_german-sharegpt)。

- **Mistral 的微调指南**：为了向技术社区致敬，Mistral 为 Mixtral 模型推出了微调指南，为那些开启微调之旅的人提供了指引；该指南可以在其 [GitHub](https://github.com/mistralai/mistral-finetune) 上查阅。

- **模型调优细节曝光**：社区正在使用 ***oobabooga*** 和 ***ollama*** 进行实验，涉及 **'skip_special_tokens'** 开关和 stop token 设置，包括建议使用特定的 [Llama3 模板](https://github.com/CrispStrobe/EQ-Bench/blob/main_v2_3a/instruction-templates/Llama3.yaml) 来解决输出问题——这反映了成员中盛行的积极调整和调优文化。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **LAM 安全担忧与移动性**：尽管 [Rabbit LAM 架构概览](https://engineering.rabbit.tech/lam-security-architecture-overview) 展示了其安全的 Web App 控制能力，但讨论中仍透露出对 **Large Action Model (LAM)** 易被逆向工程的怀疑。对话还围绕在 **Rabbit R1** 设备和 **Humane 平台**上运行 01 模型的集成挑战，并强调了用户驱动的解决方案，如 [01 model Android GitHub 仓库](https://github.com/Tonylib/o1_for_flutter)。

- **快速安装策略**：Python 版本升级到 **3.11 解决了 Mac 上的 OpenInterpreter 问题**，而通过 [Andronix](https://andronix.app/) 折腾 Linux 发行版则实现了在 Android 驱动的耳机上使用 OpenInterpreter。关于 **Open Interpreter** 安装的咨询显示了对本地或云端可访问 LLM 的需求，并且推出了新的 [Markdown 导出功能](https://github.com/OpenInterpreter/open-interpreter/pull/1282) 以帮助开发者。

- **社区的 DIY 精神**：一位用户通过升级到 Python 3.11 修复了 Mac 上的 OpenInterpreter 问题，其他用户分享了增强现有设备的途径，例如**使用 LineageOS 修改 R1**。在购买还是自建硬件的讨论中，共识是购买预建的 O1 有利于西雅图开发团队，尽管它与自建版本在技术上没有区别。

- **发货与存储备受关注**：用户对**硬件发货缺乏更新**表示不满，特别是关于欧洲地区的配送和预订信息的缺失。对话还涉及寻找**克服 Runpod 磁盘空间限制**的解决方案，这暗示了 AI 工作负载中对成本效益型数据存储的持续需求。

- **OpenInterpreter 的进步**：成员成功在 Mac 上切换到 **Python 3.11** 以运行 OpenInterpreter，标志着社区在解决问题方面的持续敏捷性。同时，[新 Markdown 导出功能](https://github.com/OpenInterpreter/open-interpreter/pull/1282) 的实现反映了在 AI 工具链中增强开发者实用性的推动。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**PDF 提取被证明具有挑战性**：关于从 PDF 中提取文本的讨论强调了在处理复杂表格和图表时遇到的困难，并建议了基于 ML 的文本分割和使用 Adobe Extract API 进行布局解析等解决方案，参考了 [LangChain 文档](https://python.langchain.com/v0.2/docs/how_to/document_loader_pdf/)。

**LangChain 社区即将扩展**：来自 Scogo Networks 的 Karan Singh 表示有兴趣在孟买建立本地 LangChain 社区，并正在寻找市场联系人以组织活动。

**Langserve 等候名单出现波折**：用户在 Airtable 上遇到了 Langserve 等候名单的访问问题，正在寻找尝试该托管服务的替代方法。

**交互式数据可视化工具推出**：介绍了 NLAVIDA 项目，该项目通过自然语言促进交互式数据可视化和分析，并附带了 [YouTube 视频教程](https://www.youtube.com/watch?v=leJRP_mJsSQ&t=4s)。

**准备就绪，为 OranClick 投票**：旨在优化消息撰写以提高注册率的工具 OranClick 宣布发布，并邀请在 [ProductHunt](https://producthunt.com/posts/oranclick) 上提供支持。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Llamafile 发布 v0.8.5 版本**：重点介绍了 [llamafile 0.8.5 版本](https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8.5) 的发布，该版本为 X86 CPU 上的 K 量化（K quants）提供了**快速推理**，并号召社区使用 `llamafile-bench` 进行基准测试。
- **Llamafile 增强网络访问能力**：工程师们交流了如何使 llamafile 服务器支持网络访问的技巧，建议使用 `--host <my ip>` 或 `--host 0.0.0.0` 等标志，以便在同一网络内的跨机器可用。
- **Llama3-70B 中的空白响应之谜**：贡献者报告称遇到了来自 llama3-70b 模型的**空白响应**，并分享了日志供社区主导的调试，尽管目前尚未出现明确的解决方案。
- **Home Assistant 的本地 API**：关于增强 **Home Assistant** 集成的讨论非常热烈，重点是开发一个类似于 OpenAI 候选方案的标准本地 API，并强调了 API 可发现性和安全 API 端点等功能的重要性。
- **模型选择的 Python 难题**：问题和分享的代码片段表明，在 LLaMA_CPP 集成的 Python 示例中指定模型时存在一些困惑，特别是关于何时必须指定模型（例如使用 TinyLlama 时）。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Mistral-Finetune 之谜揭晓**：开发者们致力于解读 [Mistral-Finetune 仓库](https://github.com/mistralai/mistral-finetune) 中的新更新，重点在于理解其独特的变更。
- **MoEs 微调的怪癖**：AI 圈讨论了微调 **专家混合模型 (MoEs)** 的不确定性，强调了运行多次迭代以挑选最有效模型的必要性，尽管关于成功率的细节尚未透露。
- **Aya 23 受限的能力**：**Aya 23** 的局限性引发了热烈讨论，强调其在聊天应用中的表现不佳，其优势仅限于特定任务，正如其 [技术报告](https://cohere.com/research/papers/aya-command-23-8b-and-35b-technical-report-2024-05-23) 所述。
- **MoRA 步入微调聚光灯**：**MoRA** 作为一种用于微调的前沿高秩更新方法进入了讨论，它具有补充或超越 LoRA 的潜力，并关联了一个 [专门的 GitHub 仓库](https://github.com/kongds/MoRA)。
- **FFD Bin Packing 问题与 Llama 3 Token 备受关注**：出现了关于 FFD bin packing 实现的问题，特别是在分布式训练上下文中；同时针对 Llama 3 未训练 Token 的修复也备受关注，并分享了 [sfttrainer 的补丁](https://github.com/unslothai/unsloth/blob/main/unsloth/tokenizer_utils.py#L722) 以解决后者。

---

## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **通过虚拟生物模拟现实**：[Virtual Beings Summit](https://www.virtual-beings-summit.com) 对 AI 专业人士来说可能是一个有价值的活动，由 **Will Wright** 主持，重点关注 AI 与模拟（simulations）的交集。支持内容可以在 [Virtual Beings YouTube channel](https://www.youtube.com/channel/UCLNalUbhs_EYB5IAbw1VfaA) 找到，该频道提供了关于 AI 在交互式模拟中作用的见解。

- **使用 DIAMOND 构建 AI 梦境**：[DIAMOND GitHub repository](https://github.com/eloialonso/diamond) 介绍了 "DIAMOND (DIffusion As a Model Of eNvironment Dreams)"，它在 Reinforcement Learning 场景下使用 Diffusion Models 来增强 AI 模拟中的环境交互。

- **利用 UE5 和 4Wall 打造 AI 版“西部世界”**：关于创建沉浸式体验的讨论表明，AI Town 可能会利用 **UE5** 并集成语音控制来模拟类似于“西部世界”的环境，持续的开发信息可在 [4Wall Discord](https://discord.gg/vPum4s3h) 获取。

- **进军虚拟现实**：将 AI Town 与 **VR** 技术结合的想法受到了热烈欢迎，这表明工程师们正在考虑通过 **VR** 让 AI 生成的环境变得栩栩如生的新方法。

- **使用 SadTalker 和 V-Express 制作头像动画**：两个 GitHub 仓库，[SadTalker](https://github.com/OpenTalker/SadTalker) 和 [Tencent AI Lab's V-Express](https://github.com/tencent-ailab/V-Express)，分别提供了用于创建逼真的说话人脸动画和生成说话头像视频的工具，展示了风格化动画技术的进步。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**Zyphra Zamba 崭露头角**：新型 **Zyphra Zamba** 模型结合了 Mamba 和 Attention 机制，现已发布，并附带相应的[技术报告](https://www.zyphra.com/s/Zamba.pdf)、[PyTorch 代码](https://github.com/Zyphra/Zamba-torch)以及对 [Hugging Face Transformers](https://github.com/huggingface/transformers/pull/30950) 的集成。与 **OLMo 1.7** 的对比分析正在进行中，以评估其性能。

**SD Audio 2.0 低调发布**：**SD Audio 2.0** 的一个未经授权版本出现在 4chan 上，并在一个 Hugging Face 账号上提供，引发了成员们的讨论。

**站对站监管**：前 OpenAI 董事会成员 Hellen Toner 和 Tasha McCauley 在《[经济学人](https://www.economist.com/by-invitation/2024/05/26/ai-firms-mustnt-govern-themselves-say-ex-members-of-openais-board)》中提议对 AI 公司进行严格监管，强调由于利润动机，此类公司无法进行自我监管，并指出了过去的内部问题。

**领导层的争议**：文章批评了 Sam Altman 在任职期间所谓的“谎言毒性文化”，讨论了内部调查以及公众对缺乏透明度的抗议。

**RL 的教科书案例**：社区在 GitHub 上分享了一个新资源，即[一本关于 Reinforcement Learning from Human Feedback (RLHF) 的教科书](https://github.com/natolambert/rlhf-book)，并赞扬了 Chris Potts 和 Chris Manning 教授引人入胜的教学风格。讨论还涉及了斯坦福 CS224n 课程电子版何时发布，建议联系 Chris 以获取具体的时间表。



---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**调整技术测试的时间限制**：讨论涉及将每个测试的时间限制延长至 **9 分 34 秒** 以上的可能性，以适应如 'Taylor approximations' 等复杂函数。一个具体问题是 `clang` 函数无法完成，仅达到约 60% 的进度。

**编译崩溃需要解决方案**：一位成员指出，生成*过大的表达式*会导致编译器崩溃，并出现与不兼容操作数类型（特别是 double 类型）相关的错误。

**关于 Double 类型位运算的争议**：澄清了无法在 `double` 数据类型上执行诸如 **XOR** 等位运算的问题，解决了成员们观察到的编译错误原因。

**悬赏任务热度上升**：对各种研究导向型悬赏任务的兴趣激增，讨论涉及旧的 pull requests，并且 **George Hotz** 确认了诸如 [tinygrad pull request #4212](https://github.com/tinygrad/tinygrad/pull/4212) 中提到的悬赏仍然有效。

**解读 'vin' 并讨论支配者 (Dominators)**：George Hotz 澄清了 **UOp class** 中的 'vin' 并不是一个缩写。此外，一位成员询问为什么不使用后支配分析 (post dominator analysis) 来改进模型中的调度 (scheduling)，并建议这可能会在执行期间优化子图融合 (subgraph fusion)。

> 邮件中已截断了完整的逐频道明细。
> 
> 如果您想查看完整明细，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}