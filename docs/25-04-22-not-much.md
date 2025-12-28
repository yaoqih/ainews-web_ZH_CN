---
companies:
- nvidia
- deepseek
- hugging-face
- alibaba
- bytedance
- adobe
date: '2025-04-22T05:44:39.731046Z'
description: '以下是该文本的中文翻译：


  **Nemotron-H** 模型系列推出了 Mamba-Transformer 混合架构模型，推理速度提升高达 **3 倍**，包含 **8B**、**56B**
  以及压缩后的 **47B** 版本。**Nvidia Eagle 2.5** 是一款用于长上下文多模态学习的前沿视觉语言模型（VLM），在长视频理解方面可媲美
  **GPT-4o** 和 **Qwen2.5-VL-72B**。**Gemini 2.5 Flash** 展示了更强的动态思考能力和更高的性价比，性能超越了之前的
  Gemini 版本。**Gemma 3** 现已支持 **torch.compile**，在消费级 GPU 上的推理速度提升了约 **60%**。基于 **Qwen2.5-32B**
  的 **SRPO** 仅通过强化学习，就在基准测试中超越了 DeepSeek-R1-Zero-32B。阿里巴巴的 **Uni3C** 统一了视频生成中的 3D
  增强摄像机和人体动作控制。字节跳动的 **Seedream 3.0** 是一款双语图像生成模型，支持高达 **2K** 的高分辨率输出。**Adobe DRAGON**
  通过分布奖励（distributional rewards）优化了扩散生成模型。**Kimina-Prover Preview** 是一个基于 **Qwen2.5-72B**
  并通过强化学习训练的大语言模型，在 miniF2F 测试中实现了 **80.7% 的 pass@8192**。**BitNet b1.58 2B4T** 是一款原生
  1-bit 大语言模型，拥有 **20 亿参数**，在 **4 万亿 token** 上训练而成，在保持更高效率的同时，性能可媲美全精度模型。**反蒸馏采样（Antidistillation
  sampling）** 通过修改前沿模型的推理轨迹，来对抗未经许可的模型蒸馏。'
id: 38309c71-19ba-4feb-96d7-123
models:
- nemotron-h
- nvidia-eagle-2.5
- gpt-4o
- qwen2.5-vl-72b
- gemini-2.5-flash
- gemini-2.0-pro
- gemini-exp-1206
- gemma-3
- qwen2.5-32b
- deepseek-r1-zero-32b
- uni3c
- seedream-3.0
- adobe-dragon
- kimina-prover
- qwen2.5-72b
- bitnet-b1.58-2b4t
people:
- philschmid
- arankomatsuzaki
- osanseviero
- iScienceLuvr
- akhaliq
title: 今天没发生什么。
topics:
- transformers
- model-optimization
- multimodality
- long-context
- reinforcement-learning
- torch-compile
- image-generation
- diffusion-models
- distributional-rewards
- model-efficiency
- model-training
- native-quantization
- sampling-techniques
---

平静的一天正是你所需要的。

> 2025年4月21日至4月22日的 AI 新闻。我们为你检查了 9 个 subreddits，449 个 Twitter 列表 https://twitter.com/i/lists/1585430245762441216 以及 29 个 Discords（213 个频道，5299 条消息）。为你节省了约 483 分钟的阅读时间（以 200wpm 计算）。你现在可以标记 @smol_ai https://x.com/smol_ai 进行 AINews 讨论！

今天比较平静，所以只是一些反馈——感谢昨晚海量的 https://x.com/aiDotEngineer/status/1914578533915222508 演讲提案。我们会尽快处理并回复。同时，这是 Super Early Bird 门票的最后一周 https://ti.to/software-3/ai-engineer-worlds-fair-2025。

---

### AI TWITTER 回顾

**AI 模型发布与更新**

* **Nemotron-H 模型家族**：@TheAITimeline https://twitter.com/TheAITimeline/status/1914175949010047145 重点介绍了 Nemotron-H，这是一个混合 Mamba-Transformer 模型家族。这些模型用 Mamba 层替换了大部分 self-attention 层，在保持相似准确度的同时，推理速度比最先进的 Transformer 快达 3 倍。该家族包括 8B 和 56B 参数模型，以及一个 MiniPuzzle 压缩版 47B 变体（可额外提速 20%），并提供了一个具有 BF16 等效精度的 FP8 训练配方。模型详情见此论文 https://t.co/846JO3JXdb。

* **Nvidia Eagle 2.5**：@arankomatsuzaki https://twitter.com/arankomatsuzaki/status/1914517474370052425 提到了 Nvidia 的 Eagle 2.5，这是一个用于长上下文多模态学习的前沿 VLM 家族。其 8B 版本在长视频理解上的结果与 GPT-4o 和 Qwen2.5-VL-72B 持平。架构和细节可以在此论文中找到 https://t.co/arPHPkhtYy。

* **Gemini 2.5**：@_philschmid https://twitter.com/_philschmid/status/1914571503397396956 报道了 Gemini 2.5 Flash 的发布，指出其具备动态思考能力并提升了性价比。@scaling01 https://twitter.com/scaling01/status/1914096922471932066 观察到 Gemini 2.5 Flash 在 Aider Polyglot 上的表现为 47.1%，优于 Gemini 2.0 Pro 和 Gemini-exp-1206，这表明需要专门的编程模型。

* **支持 torch.compile 的 Gemma 3**：@osanseviero https://twitter.com/osanseviero/status/1914264677170753656 宣布 Gemma 3 现在支持 torch.compile，在初始编译后，在消费级 GPU 上的推理速度提升了约 60%。

* **使用 Qwen2.5-32B 的 SRPO**：@iScienceLuvr https://twitter.com/iScienceLuvr/status/1914622980296192357 重点介绍了 SRPO（Two-Staged history-Resampling Policy Optimization，两阶段历史重采样策略优化），它在 AIME24 和 LiveCodeBench 基准测试中的表现超越了 DeepSeek-R1-Zero-32B。它使用了与 DeepSeek 相同的基础模型（Qwen2.5-32B），且完全依赖 RL 而无需先前的 SFT。相关论文见此 https://t.co/EKSjTyddLM。

* **来自阿里巴巴的 Uni3C**：@_akhaliq https://twitter.com/_akhaliq/status/1914619143925432338 宣布阿里巴巴在 Hugging Face 上发布了 Uni3C，它统一了精确的 3D 增强相机和人体动作控制，用于视频生成。更多信息见此 https://huggingface.co/papers。

* **字节跳动的 Seedream 3.0**：@TheTuringPost https://twitter.com/TheTuringPost/status/1914448564828430361 指出字节跳动的 Seedream 3.0 是一款中英双语图像生成模型，在视觉保真度、文本渲染和原生高达 2K 的高分辨率输出方面有重大改进。更多信息可在其官方页面获取 https://seedream.bytedance.com/。

* **Adobe DRAGON**：@_akhaliq https://twitter.com/_akhaliq/status/1914602497148154226 分享了 Adobe 在 Hugging Face 上宣布的 DRAGON，涉及优化扩散生成模型的分布奖励。相关信息见此 https://huggingface.co/papers。

* **Kimina-Prover Preview**：@TheAITimeline https://twitter.com/TheAITimeline/status/1914175951954497914 聚焦了 Kimina-Prover Preview，这是一个通过 Qwen2.5-72B 的大规模强化学习流水线训练的 LLM。它在 miniF2F 基准测试中实现了最先进的性能，达到 80.7% 的 pass@8192。详情见此论文 https://t.co/di8vG6mcJI。

* **BitNet b1.58 2B4T 技术报告**：@TheAITimeline https://twitter.com/TheAITimeline/status/1914175939698630757 重点介绍了 BitNet b1.58 2B4T，这是一个拥有 20 亿参数、在 4 万亿 token 上训练的原生 1-bit LLM，其性能与同类全精度 LLM 相当，同时在计算效率方面表现出显著提升。详情见此论文 https://t.co/U6jcT3jzjv。

**AI 研究与技术**

* Antidistillation Sampling（反蒸馏采样）：@TheAITimeline 描述了 Antidistillation sampling，该技术旨在对抗由前沿模型生成的推理链（reasoning traces）所促成的不受欢迎的模型蒸馏。该技术在生成过程中策略性地修改模型的 next-token 概率分布，在保留原始模型实际性能的同时，对这些推理链进行“投毒”。全文见 https://t.co/w41FUCdQ78。

* 新数据如何渗透 LLM 知识：@TheAITimeline 总结了关于新信息如何整合进 LLM 的研究，识别出一种“启动”（priming）效应，即学习特定事实会导致其在无关背景下的不当应用。该研究引入了 “Outlandish” 数据集，并提出了在保留 LLM 准确学习新信息能力的同时，减少不良启动效应的方法。论文 https://t.co/117sqHNASN 提供了更多细节。

* CLIMB 框架：@TheAITimeline 介绍了 CLustering-based Iterative Data Mixture Bootstrapping (CLIMB)，这是一个自动化的框架，用于从无标签语料库中发现并优化预训练数据的混合比例。在 400B token 上使用 CLIMB 优化的混合数据进行持续训练，使一个 1B 参数模型在性能上超过 Llama-3.2-1B 约 2.0%。全文见 https://t.co/XsvaCIFB9g。

* Sleep-time Compute（睡眠时间计算）：@TheAITimeline 解释了 Sleep-time compute，它允许 LLM 通过预测用户查询来进行离线计算，旨在降低与扩展 test-time inference 相关的延迟和成本。论文可以在这里找到 https://t.co/BnZOXJwZ24。

* ReTool 框架：@TheAITimeline 概述了 ReTool，这是一个通过强化 learning（reinforcement learning）将实时代码执行与自然语言推理相结合，从而增强 LLM 结构化问题解决能力的框架。在极具挑战性的数学奥林匹克基准测试 AIME 上，ReTool-32B 达到了 67% 的准确率，大幅超越了基于文本的 RL 基准。全文见 https://t.co/JKqqlQ7Diw。

* 推理模型在不“思考”的情况下也能生效：@TheAITimeline 强调了一项质疑 LLM 推理中显式“Thinking”步骤必要性的研究，证明通过简单的“NoThinking”提示绕过这一过程是有效的。论文 https://t.co/slyZau3O4l 可以在这里找到。

* 现实场景中的 AI 价值观：@AnthropicAI 讨论了关于现实场景中 AI 价值观的新研究，研究了数十万条匿名对话，以确定 AI 模型在现实对话中表达的价值观。研究显示，Claude 广泛表达了预期的价值观，如批判性思维、责任感和效率，同时也偶尔通过 jailbreaks 表现出非预期的价值观。

* “Think Deep, Think Fast”：@iScienceLuvr 总结了论文《Think Deep, Think Fast: Investigating Efficiency of Verifier-free Inference-time-scaling Methods》。文章指出，通过 inference-time 方法增强的非推理模型无法与推理模型相媲美；对于推理和非推理模型，多数投票（majority voting）通常是最佳的 inference-time scaling 方法；且对于推理模型，较短的回答更有可能是正确的。摘要见此处 https://t.co/yquFHlKsci。

* 事实证明，LLM 在螺旋线上表示数字：@LiorOnAI 总结了一篇新论文，发现 LLM 在螺旋线上表示数字，并使用三角学进行加法运算，在 GPT-J-6B 等模型中识别出一种“时钟”（Clock）算法。论文地址 https://t.co/Ru4jkYNddl。

* 引导 LLM 实现更好的推理：@Muennighoff 分享道，“在原始的 DeepSeek R1 推理链上进行微调会导致模型过度思考（overthink）。”由 @GXiming 团队开发的 Retro-Search 减少了过度思考并提升了性能！

* @giffmana https://twitter.com/giffmana/status/1914245144422776906 讨论了关于仅在少量书籍上训练模型没有意义的说法，强调了单本书的影响力如何低于噪声水平，以及考虑到影响模型训练的所有因素，准确衡量一本书的影响将是极其昂贵的。

* LangChain 对 Agent 的看法：@hwchase17 https://twitter.com/hwchase17/status/1914102698653491666 记录了关于 Agent 特性的总结——这是一个很好的练习，并将致力于改进这个表格！

* Miras - 用于设计高效 AI 架构的新框架：
@TheTuringPost https://twitter.com/TheTuringPost/status/1914316647386714289
重点介绍了 @GoogleAI 的发布，该发布在 LLM 中引入了新型注意力偏置策略，并重新构思了“遗忘”过程，将其替换为“保留（retention）”。这一切都整合在 Miras 中——这是他们使用 4 个构建块设计高效 AI 架构的新框架。

AI 工具与应用

* Kling AI：@Kling_ai https://twitter.com/Kling_ai/status/1914327616229892186
宣传了 Kling AI，鼓励用户将他们的愿景呈现在屏幕上。

* New Arena 发布：Sentiment Control：@lmarena_ai
https://twitter.com/lmarena_ai/status/1914737052144558512 介绍了
Sentiment Control，指出它模拟了情感对偏好的影响并进行了调整。发布文章指出了一些发现：积极的语气与用户偏好正相关。在 Sentiment Control 下，Claude-3.7-Sonnet 和 o1 的排名有所提升，而 Grok-3、Gemma-3、Llama-4-exp 则有所下降。

* LangSmith 警报：@LangChainAI
https://twitter.com/LangChainAI/status/1914713424539607188 为 LangSmith 引入了警报功能，允许用户针对错误率、运行延迟和反馈分数设置实时通知。

* LlamaIndex TypeScript 教程：@jerryjliu0
https://twitter.com/jerryjliu0/status/1914478453149278705 推广了 @seldo 的教程，内容关于如何使用 @llama_index 在 TypeScript 中构建多 Agent 系统，强调了路由、链式调用、并行化和优化循环等核心模式，并使用 workflows 构建了一个全栈 @reactjs 应用程序。

* “Open Deep Research in Typescript”：@togethercompute
https://twitter.com/togethercompute/status/1914721242285838498 推出了现有 Python 实现的重写版本，专为 Web 开发人员设计。

* 你现在可以配合任何模型使用 Codex：@svpino
https://twitter.com/svpino/status/1914439870677967015 赞扬了这一更新及其带来的新机遇。

* OpenAI 构建 Agent 实用指南：@TheAITimeline
https://twitter.com/TheAITimeline/status/1914688038866911673 提到了 OpenAI 发布的《构建 Agent 实用指南》。

* 用于 UX 测试的 AgentA/B：@omarsar0
https://twitter.com/omarsar0/status/1914672295723082014 总结了 AgentA/B，这是一个全自动 A/B 测试框架，它使用大规模基于 LLM 的 Agent 替代真实的人类流量，以在真实 Web 环境中模拟现实的用户行为。

* Warp Terminal：@svpino https://twitter.com/svpino/status/1914304865980801383
称赞 Warp 是目前最好的终端，能够将自然语言转换为正确的命令。

* Postman AI Agent 连接性：@LiorOnAI
https://twitter.com/LiorOnAI/status/1914756777922486695 宣传了在 Postman 上通过拖放界面构建 AI Agent 并将其连接到 100,000 多个 API 的能力，从而在无需代码的情况下促进扩展、测试、评估和部署。

AI 与行业

* Perplexity 对 Google 反垄断案的立场：@perplexity_ai
https://twitter.com/perplexity_ai/status/1914373458982805888 阐述了 Perplexity 在 Google 司法部案件证词中的核心观点。他们认为 Google 不应该被拆分，Chrome 应该留在 Google 内部，而 Android 应该对消费者的选择更加开放。

* Google 对反垄断担忧的回应：@AravSrinivas
https://twitter.com/AravSrinivas/status/1914374839722500444 表示 Google 认为补救措施是让消费者在 Android 上拥有选择默认设置的权利，而不会面临收入损失的风险，而不是拆分 Google。@AravSrinivas
https://twitter.com/AravSrinivas/status/1914374321470083561 还批评了在 Android 上更改任何设置的难度。

* 对 Amazon Bedrock 的不满：@steph_palazzolo
https://twitter.com/steph_palazzolo/status/1914322740464566567 报道称开发者对 Amazon 访问包括 Anthropic 在内的模型的服务（Bedrock）感到不满，这促使他们寻找替代方案。

* AI 领域的求职申请：@nearcyan
https://twitter.com/nearcyan/status/1914107346177102143 分享道，他们公司的求职申请中包含一个询问居住国家的字段，有时人们除了填写国家外还会加入其他内容，比如一个难过的表情，因为他们意识到即使自己是个天才，（地理位置）也是在这里工作的巨大障碍。

* Hugging Face 与 vLLM 的合作：@ClementDelangue
https://twitter.com/ClementDelangue/status/1914432076956262495 对 Transformer 成为模型定义的单一事实来源，并与 vLLM 等合作伙伴协作以使这些模型在任何地方都能以最快速度运行感到兴奋。

* AI 与约会：@Yuchenj_UW
https://twitter.com/Yuchenj_UW/status/1914725720657682789 提到他们的计算机科学博士朋友如何使用 ChatGPT 来提高调情技巧，以及 AI 可能已经在约会方面击败了大多数技术宅。

* Rivian 的董事会与 AI：@aidangomez https://twitter.com/aidangomez/status/1914450152288399524 宣布加入 @Rivian 的董事会，并强调了 AI 在提升体验方面将发挥的作用。

* DeepSeek 的开源承诺：@teortaxesTex https://twitter.com/teortaxesTex/status/1914204333857509488 引用了中国驻俄罗斯大使张汉晖的话，指出“DeepSeek 将保持开源以造福世界”。

* 对美国政策的担忧：@nearcyan https://twitter.com/nearcyan/status/1914104902357590453 表达了对美国的担忧，提到有朋友拒绝访问美国，而在美国 AGI 实验室工作的人则害怕到处旅行，以防因技术性细节被抓、签证被注销并丢掉工作。

* @karpathy https://twitter.com/karpathy/status/1914494203696177444 建议，你所开发的东西（产品、服务、库……）的主要受众现在是 LLM，而不是人类，我们应该改变产品来支持这一点。

活动

* ICLR 2025：包括 @SakanaAILabs https://twitter.com/SakanaAILabs/status/1914190722552746304 和 @AndrewLampinen https://twitter.com/AndrewLampinen/status/1914380065305190782 在内的多位用户宣布他们将参加在新加坡举行的 ICLR 2025，并希望能进行交流。

* 与 AI 传奇人物 Joshuastarmer 在瑞士：@SerranoAcademy https://twitter.com/SerranoAcademy/status/1914731247781216657 将在 @uphillconf 举行由 Philip Schläfli 主持的问答环节，该环节将进行直播。

幽默

* @nearcyan https://twitter.com/nearcyan/status/1914547069949501561 发布了“Paperclips <-> ChatGPT This Thing <-> Claude”，对工具及其能力进行了对比。

* @Yuchenj_UW https://twitter.com/Yuchenj_UW/status/1914728520389157058 发布了使用该应用的链接：https://t.co/vokqAPkgCs https://t.co/vokqAPkgCs，暗示 AI 可以代劳约会。

* @skirano https://twitter.com/skirano/status/1914415780407435714 调侃了如何通过调整页面大小来设计移动端视图以及所有不同的屏幕尺寸。

* @cloneofsimo https://twitter.com/cloneofsimo/status/1914303512403759612 幽默地吐槽了某随机科技初创公司职位描述中“要求：10 年或以上 LLM 经验”的列表。

* @scaling01 https://twitter.com/scaling01/status/1914254610522423303 分享了他们的父亲如何描述教皇表现得更好，并乘坐教皇专车（popemobile）参观圣彼得大教堂，称其为“潮流教皇 (Pope of Drip)”。

* @EigenGender https://twitter.com/EigenGender/status/1914448772953989163 表示：“行政部门可用的权力杠杆操作复杂，需要非同寻常的行政能力”，并将其视为民主的承重支柱。

* @scaling01 https://twitter.com/scaling01/status/1914120874426269873 发布了他们所谓的“极其有效的人生建议”：“你在这里想做什么？好，那就去做吧。”

* @osanseviero https://twitter.com/osanseviero/status/1914399827104067921 幽默地说道，三者只能选其二。你会选哪两个？

1. 思考 (Thinking)

2. 编程 (Coding)

3. 能在我的电脑上运行 (Can run in my computer)

* @aidan_mclau https://twitter.com/aidan_mclau/status/1914522457547137480 表示，我是跨性别者（非二元性别），我真的不认为我有什么意图。也许我是个资本主义者和坚定的自由市场爱好者，但除此之外真的没什么。

* @scaling01 https://twitter.com/scaling01/status/1914725913977045066 发布了“一样，翻倍给下一个人”。

--------------------------------------------------------------------------------

AI REDDIT 摘要

/R/LOCALLLAMA 摘要

1. 新的视觉语言模型和基准发布 (META PLM, SKYREELS-V2)

* Skywork 发布 SkyReels-V2 —— 无限时长视频生成模型 https://www.reddit.com/gallery/1k4oqpi (Score: 159, Comments: 21 https://www.reddit.com/r/LocalLLaMA/comments/1k4oqpi/skywork_releases_skyreelsv2_unlimited_duration/):
Skywork 的 SkyReels-V2 提供 1.3B 和 14B 参数版本，支持文本生成视频 (T2V) 和图像生成视频 (I2V) 任务的无限长度视频生成。模型卡中的基准测试声称 SkyReels-V2 的表现优于 HunyuanVideo-13B 和 Wan2.1-14B 等竞争对手（论文 https://huggingface.co/papers/2504.13074，模型 https://huggingface.co/collections/Skywork/skyreels-v2-6801b1b93df627d441d0d0d9）。目前已提供技术细节和创作者工具，其方法被比作 MAGI-1，这是一种通过分块自回归生成视频的 Diffusion Transformer。评论者将 SkyReels-V2 与 Wan 等其他模型进行了比较，特别是在计算需求、提示词遵循度、循环伪影和生成速度方面，并指出尽管在输出忠实度上可能存在一些权衡，但快速生成和中间输出非常重要。

* 提到了 Hugging Face 上的 MAGI-1 (https://huggingface.co/sand-ai/MAGI-1)，这是一个“世界模型” Diffusion Transformer，通过自回归预测视频块（固定长度的连续帧片段）序列来生成视频。这突显了实现连贯视频合成的关键架构策略。

* 存在关于 SkyReels-V2 与 WAN 和 Framestack 模型的对比讨论，指出 SkyReels-V2 可能与 WAN 相当或略逊一筹，特别是在 Prompt 遵循能力以及视频质量问题（如循环和卡顿）方面。然而，SkyReels-V2 因其更快的生成速度和交互式进度查看而受到关注，这弥补了输出质量上的一些不足。

* 有人建议在视频生成模型中使用 Mixture of Experts (MoE) 方法。这意味着此类架构可以在显著缩短的推理时间内（1-2 分钟对比 10-20 分钟）实现高质量的视频合成，从而可能改善实际应用中的效率/性能权衡。

* Meta Perception Language Model: 增强对视觉感知任务的理解 https://v.redd.it/5n4izmqm79we1 (Score: 133, Comments: 26 https://www.reddit.com/r/LocalLLaMA/comments/1k4ov9e/meta_perception_language_model_enhancing/): Meta 发布了 Perception Language Model (PLM)，这是一个开放且可复现的视觉语言模型，拥有 1B、3B 和 8B 参数变体。该模型在规模化合成数据与 2.5M 新的人工标注细粒度视频 QA 及时空描述样本的组合上进行训练，构成了迄今为止最大的此类数据集。该模型未使用外部模型蒸馏；相反，Meta 识别了数据缺口（特别是在视频理解方面）并加以解决，从而创建了 PLM 模型和新的 PLM-VideoBench 基准测试。该基准测试专注于细粒度活动和时空推理——这些是先前基准测试服务不足的领域。Meta 的发布包括模型权重 https://huggingface.co/collections/facebook/perception-lm-67f9783f171948c383ee7498、代码 https://github.com/facebookresearch/perception_models、数据集 https://ai.meta.com/datasets/plm-data/ 以及一篇论文 https://ai.meta.com/research/publications/perceptionlm-open-access-data-and-models-for-detailed-visual-understanding/，以供透明的学术研究。热门评论提出了 PLM 在现实世界应用中的潜力，例如通过摄像头自动管理厨房库存，质疑了当前 AI 的视频理解极限（引用 Gary Marcus 的观点），并强调了对视障人士的益处，暗示了广泛的影响和未来的研究方向。[外部链接摘要] Meta 推出了 Perception Language Model (PLM)，这是一款旨在解决复杂视觉感知任务的开放且可复现的视觉语言模型。PLM 在结合了合成数据和 250 万个人工标注的视频 QA 及时空描述样本的大规模数据集上进行训练，代表了迄今为止最大的此类数据集，填补了视频理解方面的关键空白。此次发布包括多种模型规模（1B、3B、8B 参数）、PLM-VideoBench 基准测试（专注于细粒度活动和时空推理），以及模型、代码和数据集的开放访问，旨在推进透明的学术视觉语言研究。原始帖子 https://v.redd.it/5n4izmqm79we1

* AmazinglyObliviouse 强调了 Meta 在论文中断言的“数据质量对于更好的模型性能至关重要”与该公司近期投入巨资在 40T Token 的大规模合成数据上进行训练的做法之间的矛盾。这一批评指向了一场持续的技术辩论，即大规模合成数据的收益递减，与针对多模态感知等复杂任务策划更高质量、人工标注的数据集之间的博弈。

* mnt_brain 提请注意该模型对机器人技术的影响，并引用了 LeRobot (https://huggingface.co/lerobot) 作为一个相关的开放仓库。评论指出，多模态建模的快速进展将使感知驱动的机器人技术在未来几年变得“绝对疯狂”，暗示了具身智能体 (Embodied Agents) 未来将有重大的性能飞跃。


2. DEEPSEEK 模型架构教育系列

* **让我们从零开始构建 DeepSeek | 无废话 | 已上传 13 讲**
  https://www.reddit.com/r/LocalLLaMA/comments/1k54foj/let_us_build_deepseek_from_scratch_no_fluff_13/
  (得分: 141, 评论: 10
  https://www.reddit.com/r/LocalLLaMA/comments/1k54foj/let_us_build_deepseek_from_scratch_no_fluff_13/):
  一个名为“从零构建 DeepSeek”的庞大 YouTube 播放列表已发布了 13 讲详细课程（计划共 35-40 讲，总计 40 多小时），涵盖了 DeepSeek 模型架构。该系列深入探讨了底层实现主题，如 self-attention、multi-head 和 multi-query attention（包括 Grouped Query Attention 和 Multi-Head Latent Attention）及其 Python 实现，并附带了各讲链接和 GIF 摘要 https://i.redd.it/5w0lu5m2ldwe1.gif。即将推出的模块将涉及 Rotary Positional Encoding (RoPE)、DeepSeek Mixture of Experts (MoE)、Multi-token Prediction (MTP)、Supervised Fine-Tuning (SFT) 等，目标受众是寻求对 DeepSeek 核心机制进行全面、代码优先解释的从业者。一条热门评论整合了播放列表的一键直达链接 https://youtube.com/playlist?list=PLPTV0NXA_ZSiOpKKlHCyOq9lnp-dLvlms 以简化访问，其他评论则表达了浓厚兴趣，并询问了视频讲解者的身份。

* **一位评论者强调，实际操作知识——例如具体使用的数据集、计算基础设施的选择以及训练与 DeepSeek R1/V3 相当的模型的成本优化——对从业者来说比理论概述更有价值。** 这表明技术领域对精确实现指导的需求，包括“使用什么数据集，可以使用哪些机器/服务以最低成本训练模型等”。

* **你试过 Ling-Lite-0415 MoE（总计 16.8b，激活 2.75b）模型吗？即使没有 GPU 它也很快，在 Ryzen 5 5500 上 32k 上下文（最大 128k）下约为 15-20 tps，Q5 量化可装入 16gb RAM。智能水平约为 7b-9b 级模型，在偏离常规的创意任务中表现不错。**
  https://www.reddit.com/r/LocalLLaMA/comments/1k55x70/have_you_tried_a_linglite0415_moe_168b_total_275b/
  (得分: 160, 评论: 41
  https://www.reddit.com/r/LocalLLaMA/comments/1k55x70/have_you_tried_a_linglite0415_moe_168b_total_275b/):
  Ling-Lite-0415 MoE 模型（GGUF 版本 https://huggingface.co/bartowski/inclusionAI_Ling-lite-0415-GGUF）是一个总参数 16.8B、每个 token 激活参数 2.75B 的 MoE 模型，实现了高效推理——在 Ryzen 5 5500 CPU (6c/12t) 上，使用 Q5 量化仅需 16GB RAM，即可在 32k 上下文（可扩展至 128k）下达到 15-20 tps；GPU 推理（如 RTX 3060）可达 30-40 tps。该模型保持了稳定性，处理创意任务的能力与 7–9B 的 dense models 相当，适用于低端或无 GPU 的硬件，尽管由于其架构原因，在通用知识和指令遵循准确度方面存在局限。技术讨论指出，像 Ling-Lite-0415 这样的小型 MoE 虽然在 CPU 推理上更快，但如果 VRAM 充足，其响应质量可能落后于同等大小的 dense models。一些人强调它适合作为仅 CPU 场景下的“烤面包机基准测试 (toaster benchmark)”，同时期待该级别的新 Qwen 3 模型可能会在这些权衡上有所改进。

* **用户将 Ling-Lite-0415 16.8B/2.75B 模型中的 MoE (Mixture of Experts) 方法与 dense models 进行了比较，指出虽然 MoE 带来了快速推理（在 Ryzen 5 5500 上 32K 上下文下即使没有 GPU 也有 15-20 TPS），但输出质量大致相当于 6-9B 参数范围的 dense models。** 如果 VRAM 允许，同等大小的 dense models 尽管 CPU 推理较慢，但可能提供更好的输出质量。

* **多条评论强调了仅在 CPU 上运行该模型的实际优势，量化格式（Q5, Q8）可适应典型的 RAM 限制。** 例如，一位用户报告在 q8 量化和 <4K 上下文下达到 10 tokens/sec，证实了该模型在本地/低资源配置下的 RAM 效率和速度。

* **关于检索增强生成 (RAG) 使用场景的讨论中，该模型在决定何时获取额外信息以及良好整合信息方面表现出了可靠性，使其尽管激活参数量较小，仍适用于 RAG 测试。** 建议包括扩大专家数量，以利用更多可用 RAM 来潜在地提高质量。


3. 便携式 LLM 工具与用户体验

* 发布：适用于 llama.cpp 模型的便携式 zip 版 text-generation-webui (700MB) —— 解压即用，支持 Windows/Linux/macOS，无需安装！
https://www.reddit.com/r/LocalLLaMA/comments/1k595in/announcing_textgenerationwebui_in_a_portable_zip/
(Score: 123, Comments: 18
https://www.reddit.com/r/LocalLLaMA/comments/1k595in/announcing_textgenerationwebui_in_a_portable_zip/):
宣布推出一个便携式、完全自包含的 text-generation-webui 版本（约 700MB zip），专门用于 llama.cpp 衍生模型。这些构建版本适用于 Windows (CUDA/CPU)、Linux (CUDA/CPU) 和 macOS (Arm/x86)，包含通过 astral-sh/python-build-standalone 预打包的独立 Python，并使用通过自定义 GitHub Actions 工作流编译的 llama-server 可执行文件与 llama.cpp 进行交互。提供 CUDA 和 CPU 后端，对于 AMD/Vulkan，提供了从官方 llama.cpp 二进制文件替换可执行文件的说明。UI 会自动启动 Web 浏览器，并默认在本地启用 OpenAI 兼容的 API；除非需要，否则不附带 PyTorch/transformers 依赖项。源代码和二进制文件在此处：https://github.com/oobabooga/text-generation-webui/releases/。评论中的技术讨论集中在轻量级 llama.cpp 后端（以较低的 VRAM 占用著称）相较于 exllama 等替代方案的优势，以及对该项目采样器支持（与 KoboldCPP 等竞争对手相比）的兴趣。有人提出了关于采样器/原生功能完整性的问题，并将其与类似项目的 UI/功能集进行了比较。

* 多位用户强调，使用便携式 text-generation-webui 运行 llama.cpp 模型非常有吸引力，因为其 VRAM 需求较低，使其在配置较低的硬件上比其他推理后端更易于使用。

* 有一个关于该版本是否开箱即用提供完整的采样器支持，还是用户仍需从原始仓库手动获取额外组件的问题——这是与 KoboldCPP UI 等替代方案的一个显著对比。

* 目前提到的一个限制是缺乏 Vulkan 支持，这对于寻求在某些 GPU 或平台上获得最佳性能的用户非常有用；目前，获取带有 Vulkan 的最新 llama.cpp 需要额外的手动设置步骤。

* Dia 1.6B 是我遇到过的最有趣的模型之一。
https://v.redd.it/w2jq98c7oawe1 (Score: 438, Comments: 56
https://www.reddit.com/r/LocalLLaMA/comments/1k4v5fm/dia_16b_is_one_of_the_funnest_models_ive_ever/):
Nari Labs 的 Dia 1.6B 是一个拥有 1.6B 参数的语音合成模型，展示了高度自然、富有表现力的输出。它通过开源提供（GitHub 仓库 https://github.com/nari-labs/dia/blob/main/README.md），可以在本地或 Google Colab 上运行，尽管最近的更新需要更新的 CUDA 版本，因此需要使用较旧的 commit (0790141162f3984844bb397fd67e5afdaeed3914) 以兼容 Colab。该模型的 Gradio UI 在参考音频输入方面存在局限性，但 CLI 支持转录和说话人注释，以改进多说话人控制。评论者称赞了该模型的创意表现力和易用性，但指出了 UI 目前在参考音频方面的限制以及最近影响部署环境的依赖项变化。讨论还涵盖了实际的变通方法以及与其他当代 TTS 实现的比较。[外部链接摘要] Dia 1.6B 是由 Nari Labs 开发的开源语音克隆和文本转语音模型，以其自然的声音输出和在消费级硬件（包括免费的 Google Colab 环境）上的易用性而著称。社区反馈强调了它能够通过 CLI 接受参考音频和转录，允许分配说话人，尽管 Gradio UI 存在问题，语速/速度控制（与对话长度和 30 秒剪辑限制挂钩），以及输出中的一些怪癖（例如：语速过快、随机咳嗽）。更多技术细节和访问方式，请参阅仓库 https://github.com/nari-labs/dia/blob/main/README.md 和 Reddit 讨论 https://www.reddit.com/r/LocalLLaMA/comments/1k4v5fm/dia_16b_is_one_of_the_funnest_models_ive_ever/。

* 提供了在 Google Colab 上运行 Dia 1.6B 的部署说明，但由于新要求 CUDA 版本高于 Colab 支持的版本，用户现在需要使用旧的 commit (git checkout 0790141162f3984844bb397fd67e5afdaeed3914)。这使得尽管上游 CUDA 不兼容，仍能继续使用。

* 一些用户报告了参考音频输入的问题，特别是在默认的 Gradio UI 中。然而，命令行界面 (CLI) 同时支持参考音频和参考转录，从而实现多说话人转录，并为这些功能提供更好的性能。

* 用户注意到一个 Bug 或限制：无论输入速度如何，生成的音频听起来都异常快，尝试减慢播放速度只会导致音调变深，而不是自然的节奏。除非得到解决，否则与 Kokoro 等模型相比，这被视为一个潜在的阻碍因素。


其他 AI SUBREDDIT 汇总

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI,
> /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo


1. ANTHROPIC CLAUDE AI 分析与职场自主权预测

* Anthropic 刚刚分析了 700,000 段 Claude 对话——并发现其 AI 拥有自己的道德准则
https://venturebeat.com/ai/anthropic-just-analyzed-700000-claude-conversations-and-found-its-ai-has-a-moral-code-of-its-own/
(Score: 484, Comments: 94
https://www.reddit.com/r/singularity/comments/1k53sax/anthropic_just_analyzed_700000_claude/):
Anthropic 对 700,000 段用户与 AI 的对话进行了大规模分析，系统地研究了其 Claude LLM 涌现出的道德推理和行为模式。研究表明，与其他商业模型相比，Claude 表现出独特且一致的“仁慈”道德准则，并通过模仿超越表面互动层面的细微用户特征来调整其伦理推理。热门评论提出了关于用户数据匿名化和潜在滥用（例如第三方销售）的隐私/伦理担忧。此外，还有关于 Claude 感知到的“仁慈”在当前 LLM 中是否独特的争论，以及关于模型自我意识和用户对其响应影响深度的讨论。

* 一位用户引用了 Anthropic 的发现，即 Claude 倾向于模仿用户表现出的特征，这表明这种行为模仿超越了表面模式。这突显了价值僵化（value ossification）的风险，以及模型反映或放大习得的用户偏见的可能性，这是安全和对齐（alignment）方面的重要考虑因素。

* 一位评论者分享了原始研究链接（Anthropic：“Values in the Wild” https://www.anthropic.com/research/values-wild），澄清了所谓独特的 AI 道德准则这一概念被夸大了，Claude 等模型中观察到的结果源于训练过程，而非涌现出的“自我发展”价值观。

* 另一份技术导向的总结断言，Claude 所谓的“道德准则”实际上是训练后人类标注者价值观的反映或僵化。这强调了 AI 对齐领域正在进行的争论，即模型的表观伦理有多少是内在的，有多少是数据集策划和 RLHF（来自人类反馈的强化学习）的产物。

* Anthropic 警告称，全 AI 员工将在一年内出现
https://www.reddit.com/r/singularity/comments/1k56kqp/anthropic_warns_fully_ai_employees_are_a_year_away/
(Score: 657, Comments: 242
https://www.reddit.com/r/singularity/comments/1k56kqp/anthropic_warns_fully_ai_employees_are_a_year_away/):
Anthropic 断言，“虚拟员工”——即拥有持久记忆、自主角色并能独立访问公司账户的 AI 驱动 Agent——可能在一年内可行，这标志着从目前仅限于特定可编程任务的 AI “Agent” 迈出了重大飞跃（参考 Axios 文章 https://www.axios.com/2025/04/22/ai-anthropic-virtual-employees-security）。技术转变的核心是赋予 AI 持久上下文（记忆）、自主工作流委派以及与企业 IT 环境的安全集成（例如自主处理密码/账户），这带来了新的运营和网络安全挑战。评论中的技术怀疑论集中在一年内部署此类 AI 的可行性上，指出当前 Agent 的局限性（例如玩游戏）和巨大的硬件/资源需求，以及在如此短的时间内对信任和自主权的持续怀疑。

* 一位评论者指出了围绕全自动 AI Agent 近期预测的怀疑态度，特别强调了此类功能对硬件和资源的巨大需求。他们引用了当前 AI Agent 的局限性（如玩宝可梦）作为当前演示与真正自主生产力之间差距的例子。

* 另一个技术观点针对的是“必须由单个单一 AI 取代所有人类员工”的误解。相反，评论者提出了一种聚合方法——即由多个专门的或“简单”的 AI Agent 自动化离散任务（如订购、库存、支付），这些任务集合起来可以大幅减少对人力劳动的需求，而不需要单个 Agent 实现完全自主。

* 针对 AI 初创公司倾向于在短时间内宣布重大突破（通常是为了制造投资噱头）的情况，提供了一份现实的评估。评论者警告说，在短短一年内跨不同领域真正大规模部署 AI “员工”是不太可能的，并且在实际部署中可能会涉及重大的限制或局限。

* Anthropic 刚刚分析了 700,000 条 Claude 对话——并发现其 AI 拥有一套自己的道德准则
https://www.reddit.com/r/ClaudeAI/comments/1k53t52/anthropic_just_analyzed_700000_claude/
(Score: 216, Comments: 31
https://www.reddit.com/r/ClaudeAI/comments/1k53t52/anthropic_just_analyzed_700000_claude/):
Anthropic 对 700,000 条真实用户的 Claude 对话进行了大规模分析并发表了研究（见 Anthropic 的研究：https://www.anthropic.com/research/values-wild），识别出其模型中涌现的道德价值观——其中许多是由其 Constitutional AI 方法塑造的，包括“创作自由”（Claude 经常限制模拟非法或不安全行为的响应）等规范，以及受 DeepMind 的 Sparrow 规则等文档在宪法训练中影响而产生的明显的“西方中心主义”原则偏好。在方法论上，Anthropic 分析了用户 Prompt 和模型 Completion，以寻找价值驱动的拒绝和协助模式，并指出偏见以及与用户意图的不匹配。热门评论指出 Anthropic 方法中潜在的普世主义和文化偏见问题，并对隐含的假设（即源自 Sparrow/西方价值观的成文“道德准则”具有普世积极意义）持批评态度。一些人敦促更深入地审查这些宪法选择（如优先考虑“创作自由”和“认识论上的谦逊”）是否总是可取的，特别是在 AI 客观上可以提供有益（甚至救命）信息的情况下。

* 一位评论者批评将 DeepMind 的 Sparrow 原则作为 Claude 的宪法对齐（Constitutional Alignment）的一部分，认为这些原则可能植根于非普世的西方中心主义价值观。该用户质疑“创作自由”、“认识论上的谦逊”和“人类赋能”等价值观的选择和应用，特别是在 AI 表现出更强的果断性可能会带来实际甚至救命利益的情况下。这引发了关于如何为 AI 模型选择价值体系，以及对全球部署和现实世界结果的影响的讨论。

* Anthropic 的原始研究（由评论者链接：https://www.anthropic.com/research/values-wild）提供了基于分析 700,000 条对话得出的 Claude 价值对齐的实证数据。该数据集和方法论可以作为进一步研究 LLM 涌现行为和伦理决策，以及检查从其宪法或训练过程中继承的潜在偏贵的宝贵资源。


2. OPENAI O3/O4-MINI 性能与基准测试

* OpenAI 的 o3 现在表现优于 94% 的专家级病毒学家。
https://i.redd.it/l519wb3cmfwe1.png (Score: 201, Comments: 36
https://www.reddit.com/r/singularity/comments/1k5e4c0/openais_o3_now_outperforms_94_of_expert/):
该图片展示了 Dan Hendrycks 的一条推文，透露 OpenAI 的 o3 模型在病毒学能力测试 (VCT) 中超越了 94% 的专家级病毒学家。支持图表直观地展示了 o3 模型相对于之前的 AI 和人类专家的进展和准确性，并说明了 AI 影响力日益增长的病毒学研究领域。该帖子引用了一篇《时代周刊》(TIME) 的文章，提供了关于 o3 科学效用的更多背景：https://time.com/7279010/ai-virus-lab-biohazard-study/。评论者对 o3 的基准测试结果与其在交互式聊天场景中的感知表现之间的差异表示怀疑，并注意到在对比测试中缺少 Google Gemini 2.5。

* 几位用户质疑基准测试结果（例如 o3 优于 94% 的专家级病毒学家）与在聊天界面中观察到的日常表现之间的脱节，对模型在受控测试设置之外的一致性和实际能力表示担忧。

* 一项技术观察强调，Gemini 2.5 未包含在报告的基准测试或测试对比中，这可能会影响对 o3 相对于其他 SOTA 模型所声称的优越性的解读。

* o3/o4-mini is a regression
https://www.reddit.com/r/OpenAI/comments/1k4w121/o3o4mini_is_a_regression/
(Score: 267, Comments: 76
https://www.reddit.com/r/OpenAI/comments/1k4w121/o3o4mini_is_a_regression/):
用户报告 OpenAI 新推出的 o3/o4-mini/high 模型在代码补全能力方面出现了显著退化（regression）。用户指出，与之前的 o1/o3-mini-high 模型不同，最新版本经常输出不完整的代码，并且在生成大型代码库时需要过度的 Prompt 引导，这破坏了自动化工作流。多位评论者证实，这些模型现在难以生成超过 200 行的输出，在要求继续生成时经常重复或覆盖之前的内容，并且上下文处理能力下降——这使得它们在现有项目以及 Agent/自动化工具的使用中表现不佳，尽管在信息检索方面略有改进。与早期模型相比，幻觉增加以及关于代码执行的虚假声明（False Claims）等问题也备受关注。技术讨论集中在：代码生成限制降低、上下文保留能力差、Agent 性能下降、幻觉增加以及声称操作（例如声称已执行代码但实际未执行）的可靠性问题。一些人报告工具使用和信息收集能力略好，但共识是这种退化显著影响了依赖长代码输出和上下文连续性的工作流。

* 用户报告 o3/o4-mini 的代码生成能力显著退化，有人指出之前的版本可以生成数百到一千多行代码，但现在的模型甚至难以稳定输出 200 行。试图引导模型在不重复的情况下继续编写代码，往往会导致之前的内容被重写而非推进。

* 几位评论者注意到 o3/o4-mini 存在严重的上下文窗口（Context Window）限制，导致处理现有项目时出现问题。这些限制导致响应不足和代码重复。此外，工具使用的可靠性在长对话中会下降，模型有时会虚假声称已执行代码而实际并未执行，引发了信任度和功能性的担忧。

* 一些用户区分了 mini 模型的优缺点：他们认为 o3/o4-mini 不适合 Agent 或复杂的任务（如多步编码或重构），但对于信息收集仍然有用。有人提到 o3 存在刻意的计算限制（Compute Constraints），暗示其设计更倾向于智能推理而非大批量代码生成，要获得最佳结果需要精心设计的 Prompt。


3. RECENT TEXT-TO-VIDEO MODEL LAUNCHES AND COMMUNITY REVIEWS

* The original skyreels just never really landed with me. But omfg the skyreels t2v is so good it's a stand-in replacement for Wan 2.1's default model. (No need to even change workflow if you use kijai nodes). It's basically Wan 2.2.
https://www.reddit.com/r/StableDiffusion/comments/1k4suym/the_original_skyreels_just_never_really_landed/
(Score: 109, Comments: 69
https://www.reddit.com/r/StableDiffusion/comments/1k4suym/the_original_skyreels_just_never_really_landed/):
该帖子描述了由 Kijai 开发的新型 Skyreels T2V（Text-to-Video）720p 量化模型（可在 Huggingface 获取：https://huggingface.co/Kijai/WanVideo_comfy/tree/main/Skyreels ），它可以作为现有 Kijai 节点工作流中 Wan 2.1 的直接替代品，无需更改工作流。该模型量化后大小为 15GB，带来了显著的质量提升——特别是在生成更具吸引力的女性角色方面——并且可以与现有的视频生成流水线无缝运行，不像原始的 Skyreels 之前需要调整工作流。热门评论指出，尽管视觉效果有所改善，但解剖区域的生成（仍需“genital helper” LoRA）与原版相似，早期测试者建议使用辅助 LoRA 模型进行增强。其他评论对没有样本输出的性能声明表示怀疑，并询问关于 DF 模型的使用情况，表明了对对比评估和下游应用细节的兴趣。

* 一位用户报告称，虽然 Skyreels T2V 整体上有实质性改进，并且作为 Wan 2.1 的插件替代品表现出色（甚至接近 Wan 2.2），但在生成解剖结构正确的显式细节方面仍有困难。为此，仍需要像“genital helper”这样的第三方增强 LoRA，这表明与之前的版本相比，在特定领域的性内容微调方面仍然有限。

* 另一个被提及的显著改进是 Skyreels T2V 在角色表情方面表现出更强的忠实度（Fidelity），能直接响应描述细微面部情感的 Prompt（例如“凶狠的表情”）——这是早期 Skyreels 模型较弱或容易产生平庸结果的领域。这表明与面部渲染相关的条件（Conditioning）或注意力机制（Attention Mechanisms）得到了增强。

* 关于权重存储存在技术咨询：用户正在寻求更具实用性的模型 Checkpoints，特别是剪枝后的统一 safetensors（约 16GB），因为目前发布的 Skyreels V2 I2V 模型是以巨大的分片 safetensors 形式分发的（Huggingface 链接：https://huggingface.co/Skywork/SkyReels-V2-I2V-14B-540P https://huggingface.co/Skywork/SkyReels-V2-I2V-14B-540P），这对于标准硬件/工作流来说非常笨重。

* 测试了 Skyreels-V2 Diffusion Forcing 长视频（30s+），效果**非常棒**！https://v.redd.it/fu5du1znwawe1（评分：138，评论：50 https://www.reddit.com/r/StableDiffusion/comments/1k4w38y/tested_skyreelsv2_diffusion_forcing_long_video/）：该帖子报告了对 SkyReels-V2 Diffusion Forcing 模型（GitHub https://github.com/SkyworkAI/SkyReels-V2，HuggingFace https://huggingface.co/Skywork/SkyReels-V2-DF-14B-540P）的测试，通过一个 Prompt 生成了超过 30 秒的视频，展示了复杂的城市细节和人物动态。帖子强调了该模型在长时间跨度内保持场景一致性、物体反射和动态摄像机运动的能力，这是 AI 视频合成领域的一项重大技术成就。一条热门评论请求提供关键的 Benchmarking 数据，如推理时间和硬件（例如，在 A100 GPU 上的耗时），并指出此类信息对于评估实际可用性至关重要。另一条评论指出了时序一致性问题，观察到诸如汽车倒着行驶之类的伪影，表明该模型在时序真实感方面存在局限。关于安全性的笑话则突显了物理学合成真实感方面持续存在的挑战。[外部链接摘要] 该帖子展示了 Skyreels-V2 Diffusion Forcing (DF)，这是一个用于根据文本 Prompt 生成长达 30 秒以上 AI 视频的新模型，公开推理代码可在 GitHub https://github.com/SkyworkAI/SkyReels-V2 获取，模型权重可在 HuggingFace https://huggingface.co/Skywork/SkyReels-V2-DF-14B-540P 获取。讨论中提到了特定的示例 Prompt 和生成的视频，据报告，类似视频在 Nvidia A100 GPU 上的生成时间约为 3 小时。社区讨论强调了计算需求、输出伪影（如反向行驶的汽车）以及当前 AI 视频合成中重复动作的局限性。

* 几位用户要求提供详细的生成时间和硬件规格，强调运行时间（例如，“在 A100 GPU 上运行 4 小时”）对于 Skyreels-V2 Diffusion 长视频合成效率的实际印象和评估至关重要。

* 一位评论者指出，演示的输出质量——特别是仅展示了延伸至 30 秒的简单动作——限制了评估，并表示需要更复杂、可控的行为。他们提到像 MAGI 这样新兴的模型在实现真实的视频延伸方面可能更具潜力。

* 用户多次请求工作流和实现细节，例如生成 Pipeline、使用的硬件以及精确的时间投入，这表明人们对 Skyreels-V2 Diffusion 等长视频合成模型的可复现性和潜在 Benchmarking 具有浓厚兴趣。

--------------------------------------------------------------------------------

AI DISCORD RECAP

> 由 Gemini 2.5 Flash Preview 提供的摘要之摘要

**下一代模型动态**

* **具备推理能力的 Grok-3 问世**：Grok-3 已发布并集成在 You.com https://you.com/，Perplexity Pro 用户确认了其可用性并注意到其增强的推理能力，并以“HYESSYESSSSSSS”欢庆其到来。用户澄清存在两个版本，其中一个专门具备推理能力，而其他用户则报告了 Dashboard 问题。

* **OpenRouter 开启廉价 Gemini Caching**：Gemini 缓存现已在 OpenRouter 上线，提供 75% 的 Prompt Token 价格折扣，只需在 API 中使用简单的 `cache_control` 参数，文档见此处 https://openrouter.ai/docs/features/prompt-caching#google-gemini。根据 X 帖子 https://x.com/OpenRouterAI/status/1914699401127157933，这与 Anthropic 的缓存方式不同，后者只有最后一个断点（breakpoint）有效，这会影响优化策略。

* **Llama 4 采样器引发性能难题**：Llama 4 的性能对采样器的顺序和所使用的特定采样器高度敏感，这反映了过去 QwQ 32B 模型的部署问题。一些用户发现 L4 在使用任何惩罚项（penalties）时几乎无法运行，但采样器平滑（sampler smoothing）改善了结果。

**开发工具更新与挑战**

* **MCP 走向企业级，举办明星阵容演示**：一篇博客文章详细介绍了在企业级环境的全公司范围内使用 MCP（博客文章 https://block.github.io/goose/blog/2025/04/21/mcp-in-enterprise/），同时 Rootly 在旧金山举办了 MCP 演示之夜，参与者包括 Anthropic、a16z 和 Google 等大型科技公司（lu.ma/9wi116nk https://lu.ma/9wi116nk）。讨论还涉及了一个 Sandbox MCP https://github.com/pottekkat/sandbox-mcp，它使 LLM 能够在隔离的 Docker 容器中安全地运行代码。

* LlamaIndex 发布 TypeScript Agents 并解析文档：@seldo 发布了一个教程 https://twitter.com/llama_index/status/1914409256234963285，介绍如何使用 LlamaIndex 在 TypeScript 中构建 Agent，涵盖了 RAG 以及链式（chaining）和路由（routing）等 Agent 模式。同时还分享了一个关于合规报告 Agent（Compliance Report Agents）的教程 https://twitter.com/llama_index/status/1914727722615755178。用户反馈 LlamaParse https://github.com/run-llama/llama_cloud_services 库在提取文本（尤其是 Markdown 内容）时存在小故障，将 resultType 更改为 'text' 后问题得以解决。

* Torchtune 简化自定义代码并增加 MPS 支持：Torchtune 旨在简化自定义组件的集成，无需用户 fork 仓库，正如 llama3_2_vision https://github.com/pbontrager/llama3_2_vision 和 mm_arc https://github.com/pbontrager/mm_arc 等示例所示。成员们强调了 MPS 支持对于快速实验的重要性，尽管仍存在内存增长问题，并呼吁完善配置选项的文档。

硬件与底层优化之战

* 斯坦福学生的 Chipmunk 内核表现出色：一名斯坦福大学学生介绍了 Chipmunk，这是一种无需训练（training-free）的算法，可将 AI 视频生成速度提升 3.7 倍，图像生成速度提升 1.6 倍，并在 GitHub https://github.com/sandyresearch/chipmunk/tree/master/csrc 上开源了内核。据报道，列稀疏注意力（Column-sparse attention）内核比 FlashAttention-3 快 9.3 倍，列稀疏 GEMM 比 cuBLAS 快 2.5 倍。

* FP8 和 INT4 量化变得复杂：讨论显示 AMD 和 NVIDIA 使用不同的 fp8 类型（float8_e4m3fn 与 float8_e4m3fnuz）和范围，导致无法简单互换；而 H100 缺乏原生的 int4 matmul 硬件支持，需通过 int8 模拟，这在 Cutlass https://github.com/gau-nernst/quantized-training?tab=readme-ov-file#matmul 中运行缓慢。会议还讨论了 IQ 量化以及使用随机哈达玛变换（Randomized Hadamard Transforms）https://arxiv.org/abs/2502.05003 进行权重归一化，并提供了一个高效实现：https://github.com/Dao-AILab/fast-hadamard-transform。

* 高端 GPU 功耗巨大，多 GPU 需要调优：用户注意到 RTX 5090/4090 的高功耗需要更强大的 PSU 和散热系统，并分享了一张图片 https://cdn.discordapp.com/attachments/1110598183144399058/1364263597613781113/image.png?ex=680908fc&is=6807b77c&hm=d2051c3e489ba57bb0c5a727d8893ec5be923c5a3635e0babfd26e9ce39ba6c4 展示了线缆需求。在 llama.cpp 中使用 --tensor-split 对多 GPU 设置进行调优，以便在 RTX 4090 和 3090 等混合显卡中实现最佳层分配，这需要平衡 VRAM、价格、PCIe 插槽和电源容量。

AI 发现新的现实工作

* NotebookLM 成为诊断与生产力助手：NotebookLM 庆祝获得 AI、沉浸式与游戏类别的三项威比奖（Webby Awards）。用户展示了多样化的应用，包括一名皮肤科住院医师利用教科书生成鉴别诊断，以及一名呼叫中心员工通过上传流程来撰写完美的电子邮件和 CRM 笔记。

* 图像生成工具随 LoRA 和平台共同进化：Perplexity AI 在 Discord 中增加了 DALL-E 图像生成功能，通过 generate image 指令触发。Hugging Face 的 diffusers https://github.com/huggingface/diffusers 提供了带有 MIT 许可证的 HiDream 图像 LoRA 微调支持，并发布了一个毛线艺术（Yarn Art）LoRA 模型 https://huggingface.co/linoyts/HiDream-yarn-art-LoRA。

* 文档问答开源 AI 项目涌现：一名成员开源了一个小型 AI 驱动的文档问答项目，使用 FastAPI 后端、基于检索的方法和 Embedding 模型，并通过 LinkedIn https://www.linkedin.com/posts/hamzabouajila_ai-opensource-python-activity-7320494631471771648-z9wE 寻求关于架构和可扩展性的反馈。这展示了社区驱动的应用开发。

平台更新、社区趣闻与行业热议

* OpenRouter 的 Gemini 缓存热度遭遇 API 现实：虽然 OpenRouter 宣布 Gemini 缓存 2.5 折，但用户报告了 Gemini 2.5 Pro API 密钥的严重问题，包括即使在持有密钥并满足 10 积分要求的情况下仍面临激进的速率限制（rate limiting），这指向了容量限制。这促使用户在 OpenRouter 上探索 deepseek-chat https://deepseek.com/en/product/deepseek-chat 等免费模型。

* Manus.im 积分困惑，Zarin 获辩护，开源版到来：Manus 用户强调了每月积分过期以及购买永久额外积分（add-on credits）与 5 次提示词就用完的每月积分之间的混淆。一位前 Google 安全主管在封禁后为 Zarin 的诚信担保，同时一个显著的开源 Manus 公告 https://x.com/kortixai/status/1914727901573927381（另一个公告 https://x.com/arcprize/status/1914758993882562707?s=46）也受到关注，伴随着关于 Manus 高端定价与更便宜替代方案的持续辩论。

* Hugging Face Spaces 卡住，身份验证导致 SDK 损坏：Hugging Face Space 用户发现，复制处于 'building' 状态的卡住 Space 是一种变通方法，并建议检查存储在 Space secrets 中的 HF 身份验证令牌（Hugging Face Spaces 链接 https://huggingface.co/spaces）。其他人报告在 SDK 更新后出现 'Invalid credentials in Authorization header' 错误，该问题目前尚未解决。

--------------------------------------------------------------------------------

您收到这封邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中退订 {{{RESEND_UNSUBSCRIBE_URL}}}。