---
companies:
- nvidia
- mistral-ai
- hugging-face
- zep
date: '2024-10-17T00:44:43.747168Z'
description: '**英伟达（NVIDIA）的 Nemotron-70B** 模型尽管在 **Arena Hard**、**AlpacaEval** 和
  **MT-Bench** 上表现强劲，但仍引发了一些审视，因为在 **GPQA** 和 **MMLU Pro** 等部分标准基准测试中，该模型相较于基础的 **Llama-3.1-70B**
  并未表现出提升。全新的 **HelpSteer2-Preference 数据集** 在提升部分基准测试表现的同时，对其他方面的负面影响极小。


  与此同时，**Mistral** 发布了 **Ministral 3B 和 8B** 模型，具备 **128k 上下文长度**，并在 **Mistral 商业许可**下，在多项基准测试中超越了
  **Llama-3.1** 和 **GPT-4o**。此外，通过使用 **RLHF (REINFORCE)** 训练，英伟达的 **Nemotron 70B**
  在关键基准测试上也超过了 **GPT-4o** 和 **Claude-3.5-Sonnet**。


  另外，**Zep** 推出了 **Graphiti**，这是一个基于 **Neo4j** 构建的开源时序知识图谱记忆层，专为 AI 智能体（AI agents）设计。'
id: 9a53eef2-e998-45eb-a4a3-57a5f270286a
models:
- nemotron-70b
- llama-3.1-70b
- llama-3.1
- ministral-3b
- ministral-8b
- gpt-4o
- claude-3.5-sonnet
- claude-3.5
original_slug: ainews-did-nvidias-nemotron-70b-train-on-test
people:
- reach_vb
- philschmid
- swyx
title: "目前没有证据表明英伟达（Nvidia）的 **Llama-3.1-Nemotron-70B-Instruct** 模型在测试集上进行了训练（即所谓的“数据污染”或“洗题”）。\n\
  \n以下是关于这一争议的背景和详细说明：\n\n1.  **质疑的起因**：\n    这种质疑主要源于该模型在 **RewardBench**（一个衡量模型对人类偏好判断准确性的权威基准测试）上的惊人表现。Nemotron-70B\
  \ 登顶了该排行榜，其得分（约 94.1）显著超过了 GPT-4o 和 Claude 3.5 Sonnet 等顶尖模型。由于其分数极高，一些开发者和研究人员怀疑是否存在数据泄露。\n\
  \n2.  **英伟达的解释**：\n    英伟达官方表示，该模型的卓越表现归功于其**训练方法和高质量的数据集**。他们使用了：\n    *   **HelpSteer2\
  \ 数据集**：这是一个开源的高质量偏好数据集，旨在帮助模型更好地理解人类的意图。\n    *   **RLHF（强化学习）优化**：英伟达采用了先进的对齐技术，使模型在处理复杂指令和评估回答质量方面表现更佳。\n\
  \n3.  **社区评估**：\n    虽然社区中存在讨论，但目前并没有技术报告或证据证明英伟达违反了评估规范。在大型语言模型（LLM）领域，当一个模型在特定榜单上表现异常出色时，通常都会引发此类讨论，但英伟达通过开源其模型权重和部分训练数据集，展示了其研究的透明度。\n\
  \n**总结：**\n官方立场和目前的技术分析都倾向于认为，Nemotron 70B 的高分是由于**模型对齐（Alignment）技术的进步和高质量合成数据的应用**，而非直接在测试集上进行训练。"
topics:
- benchmarking
- reinforcement-learning
- reward-models
- temporal-knowledge-graphs
- memory-layers
- context-windows
- model-releases
- open-source
---

<!-- buttondown-editor-mode: plaintext -->**对标准评估的耐心就是你所需要的一切。**

> 2024年10月15日至10月16日的 AI 新闻。我们为你检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **31** 个 Discord 社区（**228** 个频道，**1716** 条消息）。预计节省阅读时间（以 200wpm 计算）：**218 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论了！

Nvidia 的 Nemotron 成功地持续获得了关注：我们在最近几个月报道了 [Nemotron 340B](https://buttondown.com/ainews/archive/ainews-to-be-named-2748/)、[Mistral-Nemo](https://buttondown.com/ainews/archive/ainews-lskjd/) 和 [Minitron](https://buttondown.com/ainews/archive/ainews-nvidia-minitron-llm-pruning-and/)。

然而，昨天的 Nemotron-70B 正在受到更多的审查。

这是一个非常熟悉的模式：发布新的开源模型，声称“我们家里也有 GPTx/ClaudeY”，在稍微有些不同但仍然可信的基准测试中得分很高，并且它能[数出 strawberry 中 r 的数量](https://x.com/lacronicadelaIA/status/1846693418560299268)。


![image.png](https://assets.buttondown.email/images/0d5aab99-d8a6-432d-b18a-5705eb4112b2.png?w=960&fit=max)


在这种情况下，Nvidia 选择在 Arena Hard、AlpacaEval 和 MT-Bench 上营销其新款 **Llama-3.1-Nemotron-70B** 的性能，公平地说，这三个是领先的 LLM-as-Judge 基准测试。当以表格形式呈现时，结果看起来非常令人兴奋：


![image.png](https://assets.buttondown.email/images/406cfd4f-ef58-42fc-96e2-84ce5ffb979d.png?w=960&fit=max)


当应用 LMArena 的新风格控制（style control）时，模型的性能有所下降，但这本身并不足为奇。更有趣的是，其他标准基准测试，如 [GPQA](https://x.com/nisten/status/1846694482189971939)、[MMLU Pro](https://www.reddit.com/r/LocalLLaMA/comments/1g4xpj7/comment/ls9ljn2/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) 和 [aider](https://www.reddit.com/r/LocalLLaMA/comments/1g5c42h/llama31nemotron70binstructhf_scored_55_on_aiders/)，与基础的 70B Llama 3.1 模型相比没有变化或表现更差，这让兴奋的 /r/LocalLlama 社区感到有些失望。

真相可能很简单：没有在测试集上训练，但新的 [HelpSteer2-Preference 数据集](https://x.com/hillbig/status/1846680004928983531)（统一了基于 Bradley-Terry 和回归的奖励模型）恰好提高了这三个基准测试的性能，而其他基准测试的损失极小。在缺乏正式的 LMArena ELO 评分的情况下，这似乎只是降低了自动化基准测试的价值，仅此而已。

不过，[使用 entropix 采样的 Nemotron 版本令人印象深刻](https://x.com/_xjdr/status/1846640821107675618)，这是一个我们曾简略报道过的持续发展的案例。

---

**[由 Zep 赞助]** Zep 是一个为 AI Agent 和助手构建的低延迟记忆层，基于一个简单的核心原语：时序知识图谱（temporal knowledge graph）。这是一种非常酷且灵活的方式，用于建模客户和产品等复杂实体之间不断变化的关系。[你可以使用他们新的开源工具 Graphiti 将其插入到你的 Agent 中](https://shortclick.link/0vx6ml)。

> **swyx 评论**：我们上周报道了 [Zep 作为一个记忆层](https://shortclick.link/uu8gwd)，看起来 [Graphiti](https://shortclick.link/0vx6ml) 是时序知识图谱记忆抽象的核心。值得注意的是，它不仅可以在你输入“片段（episodes）”时自主为你构建知识图谱，而且它底层是基于 Neo4j 构建的！

---

{% if medium == 'web' %}

**目录**

[TOC]

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 简报

> 所有简报均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型发布与更新**

- **Mistral 发布新模型**：[@AlphaSignalAI](https://twitter.com/AlphaSignalAI/status/1846564019249016931) 和 [@MistralAI](https://twitter.com/sophiamyang/status/1846562768360534299) 宣布发布 **Ministral 3B 和 8B** 模型，在各种基准测试中优于 **Llama 3.1** 和 **GPT-4o** 等现有模型。这些模型具有 **128k 上下文长度**，并根据 **Mistral Commercial License** 提供。
  
- **NVIDIA 的 Nemotron 70B 表现优于竞争对手**：[@reach_vb](https://twitter.com/reach_vb/status/1846484958342168953) 和 [@AlphaSignalAI](https://twitter.com/AlphaSignalAI/status/1846529111399051603) 强调 **NVIDIA 的 Nemotron 70B** 在 **Arena Hard** 和 **MT Bench** 等基准测试中超越了 **GPT-4o** 和 **Claude 3.5 Sonnet**，展示了通过 **RLHF (REINFORCE)** 训练技术带来的显著改进。

- **Hugging Face 集成**：[@reach_vb](https://twitter.com/reach_vb/status/1846545312548360319) 和 [@_philschmid](https://twitter.com/_philschmid/status/1846452029582959012) 分享了关于 **Hugging Face** 合作的更新，包括能够使用 **Ollama** 直接在平台上运行任何 **GGUF model**，增强了像 **Llama 3.2 3B** 这样模型的易用性和部署。

**AI 研究与创新**

- **高级认知架构**：[@francoisfleuret](https://twitter.com/francoisfleuret/status/1846426258424418772) 和 [@AIatMeta](https://twitter.com/AIatMeta/status/1846595406261899363) 讨论了具有 **memory**、**personality** 和 **emotional intelligence** 的长期运行 **Agent** 在 **cognitive architecture** 方面的突破，强调了在 **Minecraft** 上**碾压现有基准测试**（如 **Voyager**）的研究。

- **上下文强化学习 (ICRL)**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1846574010886226279) 展示了关于 **ICRL** 的研究结果，证明了 **LLM** 如何仅通过 **reward signals** 进行适应，通过 **Explorative ICRL** 将 **Banking-77** 等任务的性能显著提高了 **66.0% 的准确率**。

- **LLM 中的任务叠加 (Task Superposition)**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1846421589618774035) 探讨了 **LLM** 同时执行 **多个不同任务** 的能力，揭示了 **更大的模型** 表现出更高的 **task completion rates**，以及对 **in-context distributions** 更好的 **calibration**。

**AI 工具与 API**

- **使用 Amazon Bedrock 的无服务器 Agent 工作流**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1846574359902605734) 推出了一门关于 **Amazon Bedrock** 的新课程，使开发人员能够 **构建可扩展的 Agent** 并为 **负责任的运营** 实施 **security guardrails**。

- **动态少样本提示 (Dynamic Few-Shot Prompting)**：[@llama_index](https://twitter.com/llama_index/status/1846351135596335165) 分享了关于 **dynamic few-shot prompting** 的见解，这是一种根据查询检索相关示例的技术，通过使用 **LLama Index workflows** 增强了在 **customer support**、**text-to-SQL** 和 **structured outputs** 中的应用。

- **TorchTitan 仓库**：[@Ethan_smith_20](https://twitter.com/Ethan_smith_20/status/1846394622630940998) 赞扬了 **torchTitan** 仓库全面的 **parallelism capabilities**，无需修改模型即可简化 **deep learning** 中 **parallel computing** 的开发过程。

**行业新闻与见解**

- **能源与人类深度探讨**：[@MajmudarAdam](https://twitter.com/MajmudarAdam/status/1846357368466297214) 对 **energy** 如何塑造人类文明及其对 **deep learning** 的未来影响进行了广泛分析，涵盖了从 **energy physics** 到 **energy distribution systems** 及其与 **geopolitics** 关系的主题。

- **AI 对劳动力和效率的影响**：[@mustafasuleyman](https://twitter.com/mustafasuleyman/status/1846348762513621265) 强调了制定策略以主动 **塑造 AI 对工作和劳动者的影响** 的重要性，并承认 AI 对 **job market** 影响的不确定性。

- **Hugging Face 社区增长**：[@francoisfleuret](https://twitter.com/francoisfleuret/status/1846348156599947325) 和 [@_akhaliq](https://twitter.com/_akhaliq/status/1846348120185360776) 报告了 **Hugging Face** 社区的显著增长，新的 **leaderboards** 和 **model evaluations** 提升了该平台在 **AI research** 领域的地位。

**AI 应用与用例**

- **用于创意内容的 Suno Scenes**：[@suno_ai_](https://twitter.com/suno_ai_/status/1846574384963633345) 推出了 **Suno Scenes**，这是一款可以直接从移动设备将 **照片和视频** 转换为 **独特歌曲** 的工具，使用户能够从个人媒体创作 **电影配乐** 和 **搞笑歌曲** 等内容。

- **网络犯罪中的 AI**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1846370472658702673) 讨论了一项研究，揭示了一个 **AI applications** 助长 **cybercrime** 的 **黑市**，尽管在现实世界中的成功有限，但在两个月内赚取了超过 **28,000 美元**。

- **基于 LLM 的多智能体系统**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1846424367124672879) 展示了 **OPTIMA framework**，它增强了 **基于 LLM 的多智能体系统** 中的 **communication efficiency** 和 **task effectiveness**，在信息交换任务上实现了高达 **2.8 倍的性能提升**。

**迷因与幽默**

- **AI 生成的小吃食谱**：[@nearcyan](https://twitter.com/nearcyan/status/1846419879215366465) 幽默地分享了对 **AI assistant** **Claude** 的不满，因为它建议了一些荒谬的食谱，比如把 **糖放进微波炉** 做零食，并将其比作 **4chan 风格** 的内容。

- **与 AI 一起烹饪**：[@nearcyan](https://twitter.com/nearcyan/status/1846357255312273458) 发布了一条关于**用 Claude 煎牛排**的幽默推文，将这种体验描述为与一个**“自闭症”** AI 打交道，突显了 **AI 交互**中的怪癖和出人意料的行为。

- **AI 迷因（Meme）的流行**：[@francoisfleuret](https://twitter.com/francoisfleuret/status/1846408527801446446) 反思了**迷因（memes）**的力量，认为 **AI 模型**可以**快速塑造迷因**，从而触及**人类心理的根本**，加速了**迷因进化（memetic evolution）**的自然过程。

**AI 教育与职业**

- **教授 AI 基础知识**：[@jxmnop](https://twitter.com/jxmnop/status/1846547794762285396) 表示需要**教育软件工程师**掌握**分类基础知识**，包括**成对匹配（pair-wise matching）**、**聚类（clustering）**、**Bootstrapping** 和**统计检验（statistical tests）**，强调了基础知识在**软件工程**中的重要性。

- **AI 职业机会**：[@mervenoyann](https://twitter.com/mervenoyann/status/1846468067343454225) 和 [@seb_ruder](https://twitter.com/seb_ruder/status/1846518560908038310) 为有志于攻读 **MSc 或 PhD** 的候选人推荐了机会，重点介绍了 **David 的实验室**以及 **Mila** 研究友好的氛围。

- **前端开发挑战**：[@ekzhang1](https://twitter.com/Yuchenj_UW/status/1846393916230095345) 指出，**大多数 CS PhDs** 缺乏**前端编码技能**，并承认这是可以接受的，同时强调了 **AI 研究**中**专业技能**的重要性。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1：为 50 种语言普及医疗 LLM**

- **为 50 种语言普及医疗 LLM** ([Score: 48, Comments: 7](https://reddit.com//r/LocalLLaMA/comments/1g4n8e7/democratizing_medical_llms_for_50_languages/))：ApolloMoE 引入了一种新的**基于电路的范式（circuits-based paradigm）**，用于解释**多语言语境**下的路由，识别了“**最后阶段展开（Spread Out in the End）**”机制，并利用**语系专家**将医疗 LLM 扩展到 **50 种语言**。该项目开源了所有资源，包括 [GitHub](https://github.com/FreedomIntelligence/ApolloMoE) 上的代码和 [Huggingface](https://huggingface.co/collections/FreedomIntelligence/apollomoe-and-apollo2-670ddebe3bb1ba1aebabbf2c) 上的模型，以及用于扩展多语言医疗 LLM 能力的数据集。
  - **英语**答案在闭源 AI 中得分最低，作者将其归因于**中文和英文评估集的广泛覆盖**。这突显了改进**稀有语言医疗衡量标准集**的必要性。
  - 该模型在不同语言中的表现通过**按语言平均准确率**并在**同一组衡量标准**上进行测试来评估，作者认为这是一种合理的方法。
  - 一位用户注意到该项目涵盖的 **50 种语言**中缺少**罗马尼亚语**，从而对语言选择标准提出了疑问。


**主题 2：在单张 GPU 上为 Llama-3-8B 提供 330 万上下文**

- **[LoLCATS - hazyresearch 集合（线性化 Llama 3.1 模型 8B、70B 和 405B）](https://huggingface.co/collections/hazyresearch/lolcats-670ca4341699355b61238c37)** ([Score: 31, Comments: 12](https://reddit.com//r/LocalLLaMA/comments/1g47rrd/lolcats_a_hazyresearch_collection_of_linearized/))：**HazyResearch** 发布了 **LoLCATS**，这是一系列**线性化 Llama 3.1 模型**，涵盖 **8B**、**70B** 和 **405B** 尺寸。这些模型基于**线性化注意力 Transformer（Linearized Attention Transformer）**架构，与标准 Transformer 相比，提供了更高的性能和效率，可能在更大的数据集上实现更快的推理和训练。
  - **线性化注意力 Transformer** 架构将二次注意力替换为线性注意力，有可能提高大上下文长度下的**推理性能**，尤其是在没有 flash-attn 的情况下。
  - **405B 模型**的 **MMLU** 分数从 **83 降至 72.2**，这引发了关于线性化模型实际应用的疑问，尽管它们在长上下文、大海捞针（needle-in-haystack）和少样本（few-shot）任务中具有潜在优势。
  - 该项目包括 [**Thunder kittens**](https://github.com/HazyResearch/ThunderKittens)，**推理代码**可在 [GitHub](https://github.com/HazyResearch/lolcats/tree/lolcats-scaled/demos/vLLM) 上获取，**vLLM** 支持即将推出。

- **在单显卡上为 Llama-3-8B 提供 330 万上下文服务** ([Score: 31, Comments: 7](https://reddit.com//r/LocalLLaMA/comments/1g4anog/serving_33_million_context_for_llama38b_on_a/)): MIT 和 NVIDIA 的研究人员推出了 **DuoAttention**，这是一种在**单张 A100 GPU** 上为 **Llama-3-8B** 实现 **330 万 token 上下文**的方法。该技术在他们的 [arXiv 论文](https://arxiv.org/abs/2410.10819)中进行了详细阐述，并在开源 [GitHub 仓库](https://github.com/mit-han-lab/duo-attention)中实现，从而允许长上下文推理的实际应用。
  - **DuoAttention** 使用两个 KV caches：一个用于关键**检索头 (retrieval heads)** 的完整缓存，以及一个用于**流式头 (streaming heads)** 的恒定缓存。这使得 **Llama-3-8B** 能够在单张 **A100 GPU** 上处理 **330 万 token**，比标准的完整注意力 FP16 部署实现了 **6.4 倍的容量提升**。
  - 用户讨论了对更好的长上下文基准测试的需求，**RULER** 因仅测试检索能力而受到批评。**Michelangelo 评估 (LSQ)** 被建议作为一种更稳健的替代方案，用于测试更广泛的长上下文用例。
  - 虽然 DuoAttention 显著减小了 KV cache 的大小，但一些用户指出，对于超过 **64k tokens** 且保持连贯的模型来说，原始容量并非唯一的挑战。然而，其他人强调，像这样增量式的改进有助于该领域的整体进步。


**主题 3. LLM 中无需提示词的思维链推理**

- **[无需提示词的思维链推理 [Google 论文]](https://arxiv.org/abs/2402.10200)** ([Score: 115, Comments: 51](https://reddit.com//r/LocalLLaMA/comments/1g42bth/chainofthought_reasoning_without_prompting_paper/)): Google 研究人员引入了 **思维链 (CoT) 解码**，这是一种使大语言模型能够在**没有显式提示的情况下进行多步推理**的方法。该技术在推理过程中修改了模型的**采样过程**，在各种推理任务中实现了与标准 CoT 提示相当或更好的性能。该方法证明了 CoT 推理能力是**语言模型固有的**，可以通过解码策略激活，而不是依赖于特定的提示词。
  - 分享了一个用于复现**思维链 (CoT) 解码**方法的 **GitHub 仓库**。用户注意到论文结果与开源实现之间存在性能差距，论文显示**较小的模型从该技术中获益较少**。
  - 论文表明，**智能采样**可以提高 LLM 性能，类似于 **entropix**。结果显示，**不同规模的模型**都有提升，**基座模型 (base models)** 比**指令模型 (instruct models)** 受益更多，甚至在增加模型参数也无济于事的任务上也是如此。
  - 一些用户在他们的项目中实现了 **CoT 解码**，例如 **optillm** 以及针对 **Llama 3.2 3B** 的逐步实现。其他人讨论了处理 arXiv 论文的挑战以及当前 LLM 在真实推理能力方面的局限性。


**主题 4. Elevenlabs 的本地文本转语音替代方案**

- **在家里搭建像 Elevenlabs 这样的文本转语音系统有多难？** ([Score: 54, Comments: 33](https://reddit.com//r/LocalLLaMA/comments/1g43j46/how_difficult_would_it_be_to_have_a_texttospeech/)): 该帖子讨论了建立**本地文本转语音 (TTS) 流水线**作为使用 **Elevenlabs** 的替代方案，旨在节省成本并增加控制力。作者配备了 **i9 13900** 处理器和 **4070** GPU，正在寻求构建此类系统的建议，询问他人的经验、模型选择和硬件配置，新配置的预算为 **4000-5000 美元**。
  - **AllTalk TTS** 是一款可在 [GitHub](https://github.com/erew123/alltalk_tts/tree/alltalkbeta) 上获得的、支持多引擎的软件，提供了一套完整的 API 套件、TTS 生成器和多引擎支持。用户在测试版中讨论了其 UI 和语音质量。
  - **Piper TTS** 因其不错的性能和多种语音而受到关注，尽管它比较耗费 CPU。**F5 TTS** 系统因其能够从简短的音频样本中捕捉语音和情感表现而受到关注，可以通过 [Pinokio](https://pinokio.computer/) 访问。
  - 推荐了各种开源模型，包括 **Parler TTS**、**XTTS**、**E2**、**E5** 和 **OpenedAI Speech**。用户争论了不同模型的质量，有些人认为 **FishSpeech** 在语调和情感方面优于 F5/E5。


**主题 5. 用于程序化内容生成的 LLM 驱动游戏主持人**

- **我正在构建一个使用 LLM 作为 Gamemaster 来创造事物的项目，希望获得更多创意来扩展这个想法。** ([Score: 60, Comments: 29](https://reddit.com//r/LocalLLaMA/comments/1g4srvj/im_building_a_project_that_uses_a_llm_as_a/))：该项目使用 **Large Language Model (LLM)** 作为 **Gamemaster**，在具有 **Infinite Craft 风格合成系统** 的游戏中生成生物及其属性和能力。该 LLM（具体为拥有 90 亿参数的 **Gemma 2**）负责决定从生物名称到 Sprite 选择、元素类型、统计数据和能力的一切，且全部在仅有 **6 GB VRAM** 的电脑上本地运行。开发者强调了该模型在 **function calling** 方面的有效性，以及在保持创造力的同时最大限度减少幻觉的表现，并寻求关于利用 **递归分层列表选择 (recursive layered list picking)** 来通过 LLM 构建连贯游戏元素的扩展思路。
  - 用户对该项目表现出浓厚兴趣，多人请求提供 **GitHub** 仓库以便亲自尝试。开发者表示待项目进一步完善后会分享更多内容。
  - 关于替代模型的讨论包括测试 **L3.2 3B** 和 **Qwen Coder 2.5 7B** 的建议，开发者指出 **Qwen 模型** 在其测试中表现良好，接近 **Gemma 2**。
  - 扩展创意包括使用 **图像生成模型** 制作 Sprite、为合成激励实现 **伤害类型和抗性**，以及创建一个 **NPC 定居点系统**。开发者正在考虑 **任务原型系统 (quest archetype system)** 以及如何利用 LLM 让生物显得更具生命力。

## 其他 AI Subreddit 摘要

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI 研究与开发**

- **Google DeepMind 通过联合样本选择推进多模态学习**，展示了数据策展如何加速多模态学习 ([来源](https://arxiv.org/html/2406.17711v1))。

- **Microsoft 的 MInference 技术** 能够在保持准确性的同时，实现长上下文任务中多达数百万个 **tokens** 的推理 ([来源](https://arxiv.org/abs/2407.02490))。

- 一篇关于 **扩展合成数据生成 (scaling synthetic data creation)** 的论文利用 LLM 内部的多样化视角，从 10 亿个网络策划的角色人格 (personas) 中生成数据 ([来源](https://www.reddit.com/r/MachineLearning/comments/1dzergu/r_scaling_synthetic_data_creation_with_personas/))。

**AI 模型发布与能力**

- Salesforce 发布了 **xLAM-1b**，这是一个 10 亿参数的模型，在 [function calling 方面实现了 70% 的准确率，超越了 GPT 3.5](https://www.reddit.com/r/LocalLLaMA/comments/1dz8g10/salesforce_tiny_giant_xlam1b_model_surpasses_gpt/)。

- Rubra AI 在 6 月发布了更新的 **Phi-3 Mini 模型**，[具备 function calling 能力](https://www.reddit.com/r/LocalLLaMA/comments/1dzhe38/phi3_mini_june_with_function_calling/)，可与 Mistral-7b v3 竞争。

- 关于 **AI 推理能力** 的讨论，争论焦点在于当前模型是真正具备推理能力，还是仅仅在进行模式匹配 ([来源](https://www.reddit.com/r/singularity/comments/1g4fw2l/humans_cant_reason/))。

**AI 伦理与政策**

- 拜登政府出于国家安全考虑，正在考虑 **限制向某些国家销售 GPU**，这可能会影响全球 AI 的发展 ([来源](https://www.reddit.com/r/singularity/comments/1g45ncy/biden_administration_officials_have_discussed/))。

- Anthropic 宣布了 **更新的负责任扩展策略 (Responsible Scaling Policy)**，暗示在解决安全问题的同时，正准备发布更先进的模型 ([来源](https://www.anthropic.com/news/announcing-our-updated-responsible-scaling-policy))。

**AI 应用与演示**

- 使用 Flux Dev 演示 **AI 生成的 HD-2D 像素游戏重制**，展示了在游戏开发和视觉艺术领域的潜在应用 ([来源](https://www.reddit.com/r/StableDiffusion/comments/1g4oln0/hd2d_pixel_game_remakes_with_flux_dev/))。

- 关于 **AI 生成内容** 的潜力和局限性的讨论，包括社交媒体平台上虚假的餐厅简介 ([来源](https://www.reddit.com/r/singularity/comments/1g47zsr/its_getting_weird/))。

**AI 行业动态**

- 关于实现人类水平 AI 时间表的持续辩论，**Yann LeCun** 等专家认为这可能还需要几年甚至十年的时间 ([来源](https://www.reddit.com/r/singularity/comments/1g4467s/yann_lecun_says_mark_zuckerberg_keeps_asking_him/))。

- 对新模型发布的期待，例如基于政策更新推测 Anthropic 可能发布 **Opus 3.5** ([来源](https://www.reddit.com/r/singularity/comments/1g4a1mm/anthropic_announcing_our_updated_responsible/))。


---

# AI Discord 摘要

> 由 O1-preview 生成的摘要之摘要的摘要

**主题 1. Mistral 的新边缘模型引发 AI 社区热议**

- [**Mistral 发布 Ministral 3B 和 8B，但权重在哪？**](https://mistral.ai/news/ministraux/): **Mistral** 推出 **Ministral 3B** 和 **Ministral 8B**，这是专为设备端使用设计的边缘模型，支持高达 **128k 上下文长度**。但开发者们感到沮丧，因为 **Ministral 3B** 仅限 **API** 访问，未发布权重。
- [**模型许可引发关于 Mistral 仅限 API 的 3B 模型的争论**](https://mistral.ai/news/ministraux/): 社区对 **Ministral 3B** 仅限 **API** 表示不满，认为限制性的许可阻碍了设备端使用和独立开发。
- [**Mistral 的发布让开发者既兴奋又沮丧**](https://mistral.ai/news/ministraux/): 虽然 **Ministral 8B** 以**非商业许可**提供，但开发者对 **3B 模型**缺失权重感到遗憾，质疑此次发布的实用性。

**主题 2. NVIDIA 的 Nemotron 70B 碾压竞争对手**

- [**Nemotron 70B 展现实力，表现优于 GPT-4o 和 Claude 3.5**](https://openrouter.ai/nvidia/llama-3.1-nemotron-70b-instruct): **NVIDIA 的 Nemotron 70B** 击败了 **Llama 3.1 405B**、**GPT-4o** 和 **Claude 3.5 Sonnet**，在 **Arena Hard** 上获得了 **85.0** 分，而其他模型仅为 **79** 分左右。
- [**NVIDIA 投下 Nemotron 炸弹，社区大为震惊**](https://x.com/OpenRouterAI/status/1846651197802881094): AI 界议论纷纷，**NVIDIA** 悄然发布 **Nemotron 70B**，在没有大肆宣传的情况下撼动了基准测试排行榜。
- [**Nemotron 出色表现引发基准测试困惑**](https://x.com/reach_vb/status/1846484958342168953): 用户对 **MT Bench 分数**的差异展开辩论，对 **Nemotron** 近乎“好得令人难以置信”的结果表示怀疑。

**主题 3. SageAttention 彻底改变 Transformer 效率**

- [**SageAttention 将 Attention 速度提升 2.1 倍，让 FlashAttention2 望尘莫及**](https://arxiv.org/abs/2410.02367): 介绍 **SageAttention**，这是一种 8-bit 量化方法，比 **FlashAttention2** 快 **2.1 倍**，且精度损失极小。
- [**驯服 O(N²)：SageAttention 降低了 Attention 复杂度**](https://arxiv.org/abs/2410.02367): **SageAttention** 解决了 **Transformer** 中的 **O(N²)** 瓶颈，有望为语言和图像模型提供更快的推理。
- [**8-Bit 是新的 16-Bit：SageAttention 让量化再次变酷**](https://arxiv.org/abs/2410.02367): 凭借高效的 **8-bit 量化**，**SageAttention** 证明了低精度仍能提供顶尖性能。

**主题 4. AI 助手之苦：从 DALL-E 的失望到过度审查**

- **DALL-E 的“糟糕”图像输出让用户摸不着头脑**: 沮丧的用户将 **DALL-E** 的图像生成贴上“糟糕”的标签，对其能力表示失望。
- **LLM 忽略 Token 限制，开启无尽长谈**: 用户报告 AI 助手公然无视 **Token 限制**和**停止指令**，导致输出失控并引发用户不满。
- **被审查的模型拒绝配合，用户寻求“去审查”黑客手段**: 过度审查的模型甚至拒绝回答基础查询，迫使用户不顾潜在风险去探索**去审查技术**。

**主题 5. 开源工具助力社区协作**

- [**Open Interpreter 与 Ollama 联手打造本地 LLM 体验**](https://huggingface.co/docs/hub/en/ollama): **Open Interpreter** 现在允许通过 **Ollama** 在 **Hugging Face** 上运行任何 GGUF 模型，通过简单的命令让本地 **LLM** 更加触手可及。
- [**Inferencemax 简化 LLM 推理**](https://github.com/teilomillet/inferencemax): 新项目 **Inferencemax** 旨在简化 **LLM** 推理，反映了社区为降低 AI 开发门槛所做的努力。
- [**AIFoundry 寻求 GitHub 指导以提升开源水平**](https://discord.gg/aSHN7W5E): **AIFoundry.org** 正在寻求指导以效仿 **Axolotl** 的 **GitHub** 实力，希望增强其开源本地模型推理计划。

---

# 第一部分：高层级 Discord 摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Gradio 5.0 增强功能**：[Gradio 5.0](https://www.producthunt.com/posts/gradio-5-0) 已发布，带来了安全性和用户界面的更新，超过 **600 万次下载**量证明了其受欢迎程度。
  
  - 详尽的[安全报告](https://x.com/Gradio/status/1844415295487869226)现已公开，向用户保证了已完成的改进。
- **Sentence Transformers v3.2.0 提升速度**：[Sentence Transformers v3.2.0](https://x.com/tomaarsen/status/1844440420027335064) 引入了 ONNX 和 OpenVINO 等新后端，可实现 **2x-3x 的加速**，使用静态嵌入（static embeddings）时加速可达 **500x**。
  
  - 更快的推理能力支持高达 **10k 文本/秒**的处理速度，更多详情见 [Model2Vec](https://huggingface.co/blog/Pringled/model2vec)。
- **HuggingChat 中的多模态交互**：HuggingChat 的最新更新集成了 [Llama-Vision 11B Instruct](https://x.com/mervenoyann/status/1844678895657685409)，支持丰富的多模态交互。
  
  - 这一重大升级鼓励用户在平台内探索这些新功能，从而提升用户体验。
- **AI 模型性能讨论**：关于配置为 **72GB VRAM** 和 **128GB DDR4** RAM 的 AI 模型设置的假设性讨论认为，潜在处理速度可达 **5-6 t/s**。
  
  - 讨论中还涉及了自定义 **PyTorch** 集成，强调了自动梯度（automatic gradients）对于提高模型效率的重要性。
- **Ollama 与 GGUF 模型的交互**：使用 **Ollama** 允许用户直接在本地与 GGUF 模型交互，简化了命令使用，无需创建新的 `Modelfiles`。
  
  - Ollama 支持运行 Hugging Face 上的任何 **4.5 万个 GGUF 检查点**，提高了可访问性。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI 推出购物功能**：Perplexity AI 正在推出“Perplexity Purchases”，以简化购买流程和价格比较。
  
  - 用户反馈差异显著，一些人怀念该平台最初专注于搜索而非商业的初衷。
- **Reasoning Mode 给用户留下深刻印象**：成员们称赞了用于编程的 **Reasoning Mode**，强调了其分析能力和由此产生的准确输出。
  
  - 成功案例不断涌现，随着用户分享他们的积极体验，进一步巩固了该功能的可靠性。
- **对增强 API 的兴趣**：用户对 API 的好奇心日益增强，多位用户引用了同一个[搜索结果](https://www.perplexity.ai/search/what-is-an-api-6HaQAJlXRGOWBgQd3L7Iyg#0)来定义什么是 API。
  
  - 这一趋势表明成员们对基础技术的参与度更深。
- **关于 LFM 40B API 可用性的查询**：一位成员询问如何通过 labs.perplexity.com 上的 API 访问 **LFM 40B** 模型，但尚未得到回复。
  
  - 信息的缺失凸显了在模型可用性沟通方面可能存在的差距。
- **对聊天中用户体验的担忧**：用户对论坛的动态表示担忧，认为其过于非正式，不适合严肃的 AI 讨论。
  
  - 这导致了要求加强管理的呼声，以保持对技术主题的关注，而非闲聊。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Grok 2 停机维护**：**Grok 2** 目前处于离线维护状态，导致用户在尝试访问时遇到 **404 错误**。该模型重新上线后将发布公告。
  
  - 用户对此表示沮丧，因为 Grok 2 在编程任务中的表现优于其他模型，特别是击败了 **Llama 3.2**。
- **NVIDIA Nemotron 70B 碾压竞争对手**：**NVIDIA 的 Nemotron 70B** 在基准测试中超越了 **Llama 3.1 405B**、**GPT-4o** 和 **Claude 3.5 Sonnet**，在 **Arena Hard** 上得分 **85.0**，而竞争对手的得分均在 **79** 分左右。详细对比可以在[这里](https://openrouter.ai/nvidia/llama-3.1-nemotron-70b-instruct)查看。
  
  - 这一兴奋点在 [OpenRouter 公告](https://x.com/OpenRouterAI/status/1846651197802881094)中达到顶峰，该公告展示了其在多项评估中的卓越性能。
- **ChatGPT 语音模式教学词汇**：一位用户展示了 **ChatGPT Advanced Voice Mode** 利用《火影忍者》（**Naruto**）的例子来教授词汇，称这种体验“简直太疯狂了！”。他们分享了一个 [演示链接](https://x.com/ahmetdedeler101/status/1846305587442995446) 以收集反馈。
  
  - 讨论集中在个性化 **AI 学习** 的潜力上，并预测由于其高效性，它将极大地改变教育格局。
- **Infermatic 网络困扰**：**Infermatic** 的供应商面临持续的网络问题，导致模型生成乱码，特别是在达到 **8k Context Limit** 之后。用户被告知供应商正在回滚到之前的版本，以纠正这些 **VLLM** 推理问题。
  
  - 人们对模型性能受到的影响表示担忧，因为这个 Bug 阻碍了有效的交互。
- **Mistral 推出口袋级 LLM**：**Mistral** 宣布发布两个新模型：**Ministral 3B** 和 **8B**，专为边缘计算场景设计，并承诺提升性能。这些模型拥有更长的上下文长度，并在知识和推理任务中增强了能力。
  
  - 此举旨在将 LLM 的应用扩展到传统设置之外，详见 [Mistral 的公告](https://mistral.ai/news/ministraux/)。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **INTELLECT-1 发布，助力去中心化训练**：[INTELLECT-1](https://www.primeintellect.ai/blog/intellect-1) 的发布邀请各方为一个专注于去中心化训练的 **100 亿参数**模型贡献力量，旨在实现开源 AGI。此前发布的 [OpenDiLoCo](https://www.primeintellect.ai/blog/opendiloco) 增强了 AI 模型训练的可扩展性。
  
  - 这一倡议标志着向全球分布式 AI 迈出了重要一步，目前规模已从 **1B 扩展到 10B 参数**。
- **Unsloth 训练显示出显著改进**：用户报告称 `unsloth_train` 的收敛效果明显优于以前的方法，并有望支持 `resume_from_checkpoint=True`。然而，关于旧版 `UnslothTrainer` 缺失扩展功能的询问也随之而来。
  
  - 社区对这些增强功能表示赞赏，同时也寻求关于这一过渡背后逻辑的进一步说明。
- **社区询问 Mistral 8B 支持情况**：关于统一 Unsloth 与新 [Mistral 8B 模型](https://mistral.ai/news/ministraux/) 兼容性的讨论引发了一些架构方面的关注。社区的热情围绕着新模型的端侧计算能力展开。
  
  - 成员们渴望获得更新，并认可 Mistral 8B 在实际应用中的潜力。
- **SageAttention 实现惊人的加速**：**SageAttention** 论文介绍了一种高效的 **8-bit 量化方法** 用于 Attention，在保持模型准确性的同时，分别比 **FlashAttention2** 和 **xformers** 快 **2.1 倍** 和 **2.7 倍**。该量化方法解决了通常出现的 **O(N^2)** 复杂度问题。
  
  - SageAttention 代表了一项关键进展，显著加快了多种模型的推理速度。
- **量化技术的探索**：讨论揭示了将全量微调技术与量化方法（特别是 **QLoRA**）混合使用的挑战，用户分享了关于层调优（Layer Tuning）的见解。对于在保持其他层完全可训练的同时量化某些层的可行性，仍存在怀疑。
  
  - 社区正在辩论是否需要专门的配置来平衡性能和效率。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Yandex YaLM 100B 引起轰动**：拥有 **1000 亿参数**的 [Yandex YaLM 100B 模型](https://huggingface.co/yandex/yalm-100b)已成为一个重要的参与者，尤其是在非西方市场。
  - 据指出，它可能是俄罗斯**使用最广泛**的 LLM，与其在西方圈子中较低的认可度形成鲜明对比。
- **SwiGLU 与 SinGLU 的对决**：一场关于选择 **SwiGLU** 还是 **SinGLU** 的辩论被点燃，强调了 SinGLU 的速度和更低的 loss，但变革的阻力依然存在。
  - 这种惯性源于与大规模训练运行（large training runs）相关的风险以及既定的实践惯例。
- **OpenAI embeddings 表现不佳**：参与者对 OpenAI embedding 模型的性能表示担忧，这些模型似乎落后于 **2024 benchmarks**。
  - **Mistral finetunes** 等模型的饱和表明 OpenAI 的方法存在竞争差距。
- **机械可解释性（Mechanistic Interpretability）项目招募志愿者**：一名学生表达了加入 EleutherAI 可解释性相关项目的渴望，特别是在当前的机会背景下。
  - 成员们建议加入 [Mechanistic Interpretability Discord](https://mechinterp.com/read) 以在该领域进行进一步探索。
- **A/B 测试方法解决反转问题**：人们对 A/B 测试技术的兴趣日益增加，这些技术可以缓解**反转诅咒 (reversal curse)**，从而增强实验结果。
  - 参与者将这种方法标记为“非常 a/b”，指出了它在实际应用中的相关性。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **多个 Aider 可以安全共存**：关于运行多个 **Aider** 实例的担忧得到了缓解，确认只要不编辑相同的文件，它们就**不会互相干扰**。
  - 成员们幽默地建议，如果管理得当，这可能会变成一场“LLM 派对”。
- **Mistral 推出新款边缘模型**：Mistral 最近发布了专注于**设备端（on-device）**和**边缘计算（edge computing）**的 **Ministral 3B** 和 **8B** 模型，提升了效率和能力。
  - 这些模型在推理和常识知识方面取得了重大进展，是优化上下文长度的理想选择。
- **Gemini API 流式传输稳定性有待提高**：用户报告称，由于 **Gemini** 的 **API 连接不稳定**导致频繁中断，禁用流式传输（streaming）后表现更好。
  - 共识指出，这种不稳定性是影响基于 Gemini 的工具性能的常见问题。
- **Aider 命令行工具设置要点**：为了有效利用 **Aider 命令行工具**，用户必须加载其 `.env` 文件或通过 `load_dotenv()` 进行配置，以确保功能正常。
  - 正确的环境设置对于在 Aider 中顺利运行脚本至关重要。
- **API 和代码生成的挑战**：用户在处理 **rate limits** 的同时，在使用更新后的 Assistant API 生成准确的函数调用（function calls）方面面临困难。
  - 这种繁忙的情况强调了需要清晰的文档和社区支持来应对新兴挑战。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Unsloth 提供多 GPU 支持**：讨论集中在 **Unsloth** 是否能有效支持多 GPU 设置，并期待即将发布的视觉微调（vision fine-tuning）支持更新。
  - 成员们对其付费版本在增强性能方面的可靠性进行了推测。
- **Mistral 发布新模型**：**Mistral** 推出了专为边缘计算设计的 **Ministral 3B** 和 **Ministral 8B**，在常识推理方面拥有令人印象深刻的数据，并支持 **128k context length**。
  - 这些模型承诺提供高效的本地推理，专门迎合现代计算需求。
- **Nvidia Nemotron 70B 声称性能领先**：据各种评估指标显示，[Nvidia Nemotron 70B](https://x.com/reach_vb/status/1846484958342168953) 据称超越了 **Claude 3.5** 和 **Llama 3.1** 等竞争对手。
  - 关于 MT Bench 分数存在困惑，各模型的报告性能与实际性能之间存在差异。
- **AI 模型显示困惑的回答**：模型 **H3-405b** 因其重复的困惑回答而受到关注，尤其是当被问及它的起源或身份时。
  - 令人苦恼的困惑表达案例增加了 AI 身份讨论的趣味性。
- **SageAttention 提高推理效率**：研究强调了 **SageAttention**，这是一种量化技术，在性能损失极小的情况下，比 **FlashAttention2** 的 Attention 性能提升了 **2.1 倍**。
  - 这一进步将使广泛的任务受益，特别是在大规模语言应用中。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **寻找开源音频模型**：一位用户询问是否有类似于 **NotebookLM** 中的高质量开源音频模型，并提到虽然存在许多 **Text-to-Speech** 选项，但没有一个能与之媲美。
  
  - 参与者一致认为，市场上缺乏强大的音频模型。
- **Lambda Labs 与 Voltage Park 的对决**：讨论集中在 **Lambda Labs** 和 **Voltage Park** 是仅有的可靠硬件供应商，其中 **Voltage Park** 以更多存储空间著称，但仅限于德克萨斯州。
  
  - 参与者对其他供应商持续存在的 **PCIe** 问题表示担忧，这影响了 **GPU** 设置的可靠性。
- **Triton 编程的主要挑战**：成员们强调了 **Triton** 的各种问题，包括在 Windows 上编程的困难以及 **INT4 packed data** 中的 bug 导致 **LLVM** 错误。
  
  - 许多用户感到沮丧，指出 **Triton** 编译带来的性能提升通常来自 **Torch** 而非 **Triton** 本身。
- **ServiceNow 招聘机器学习开发人员**：ServiceNow 正在招聘一名 **Staff Machine Learning Developer**，负责其支持 **Starcoder2** 的开源训练框架，该框架比 **Megatron-LM** 更快。
  
  - 职位详情可在 [Smart Recruiters](https://jobs.smartrecruiters.com/ServiceNow/744000019737886-staff-machine-learning-developer) 找到。
- **生成式 AI 书籍发布公告**：Yuri Plotkin 宣布了他即将出版的关于 **Generative AI** 的书籍，涵盖了包括 **Bayesian inference** 和 **latent variable models** 在内的基础算法，详情请见 [书籍网站](https://thevariationalbook.com)。
  
  - 他鼓励在 [Twitter](https://twitter.com/TheVariational) 上关注他以获取持续更新，并分享了该领域关键概念的见解。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **SageAttention 提升性能**：新方法 [SageAttention](https://arxiv.org/abs/2410.02367) 通过为 **attention** 提供高效的 **quantization**，加速了 **transformer** 模型的推理，比现有方法实现了 **2.1 倍** 的提升。
  
  - 该技术显示出比 **FlashAttention3** 更优的准确性，对语言和图像生成都有影响。
- **Llama 8B 每秒 Token 数（TPS）的差异**：用户报告了 **Llama 8B** 的 **tokens per second (TPS)** 范围很广，在 **1070 Ti** GPU 上的 **Q6_K** 设置可达到约 **28-35 TPS**。
  
  - 性能与 **context length**、**quantization** 和 **GPU VRAM** 带宽等因素密切相关。
- **GPU 性能至关重要**：新一代 GPU（如 **4080** 或 **4090**）的性能大幅超越了旧型号（如 **1070 Ti**），但需要正确的配置才能发挥最大效能。
  
  - 利用 **tensor cores** 和增强的内存带宽对于实现显著的性能提升至关重要。
- **编译模型的挑战**：用户询问了 **LM Studio** 目前对自定义编译版本 **Llama.cpp** 的支持情况，得到的回复建议使用命令行工具 `lms` 进行模型加载。
  
  - 该解决方案支持重启后的持久性，缓解了编译模型面临的一些挑战。
- **Token 生成速度受到关注**：成员们强调了高容量模型下缓慢的 Token 生成速度，某些设置的峰值仅为 **0.25 tokens/sec**，说明了 **CPU** 瓶颈。
  
  - 许多本地设置都感受到了这些限制，因此有人推动在需要时考虑云服务以获得更好的性能。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Grok 2 显示出潜力**：成员们对测试 **Grok 2** 表现出浓厚兴趣，表明对新模型的关注度日益增加。
  
  - 虽然缺乏具体的性能细节，但热度暗示 **Grok 2** 可能是一个值得关注的进展。
- **DALL-E 的图像生成表现不佳**：**DALL-E** 的能力受到批评，一位成员直接将其图像输出标记为 **bad**。
  
  - 社区对图像生成的期望很高，这一反馈凸显了对其性能的失望。
- **模型参数之谜**：关于 **4o-mini** 和 **GPT-3.5** 等模型参数规模的辩论非常激烈，有推测认为 **4o-mini** 的参数量设定在 **1 billion parameters**。
  
  - 不同的意见表明社区在模型大小与性能之间的关系上存在困惑。
- **GPTs 在 PDF 理解方面遇到困难**：成员们注意到 **GPTs** 在回复前无法读取完整的 **PDFs**，通常导致引用的信息不完整。
  
  - 建议将 **key information in the main instructions**（关键信息放在主指令中）以帮助提高回复的准确性。
- **使用 ChatGPT 创建网站内容的指南**：一位用户寻求关于使用 **ChatGPT** 构建网站的建议，并询问有效 Prompt 编写的策略。
  - 重点被放在从 **trustworthy and scientific materials**（可靠且科学的材料）中获取内容，强调了对质量的关注。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 机器学习库的连胜势头**：一位成员描述了 **tinygrad** 注定会脱颖而出的三个关键原因：使用 BEAM 和 MCTS 的 **efficient kernel search**、少于 **10k lines** 的简洁代码库，以及 lazy execution 模型。
  
  - *“这避免了每个设备组合一个内核的组合噩梦……”*，强调了通过其精简方法实现的性能提升。
- **Tinybox 预订困惑**：关于 **tinybox** 预订的讨论引发了对支付方式和相关费用的询问，特别是它是否会像之前的型号一样采用 **Stripe**。
  
  - 成员们对如何使用现有方式进行预订支付流程表示好奇。
- **OpenCL 处理引起关注**：在 **Stable Diffusion** 中遇到全黑输出后，出现了关于 **Out Of Memory (OOM)** 处理的担忧，并对 OpenCL 的能力提出了疑问。
  
  - 一位成员寻求澄清该实现是否有效地解决了 tinygrad 中的这些内存溢出情况。
- **MSE 和 MAE 实现简化**：有人提议将 **MSE** 和 **MAE** 函数直接集成到 tensors 中，声称只需几行代码即可实现。
  
  - 他们引用了一个 [GitHub pull request](https://github.com/tinygrad/tinygrad/pull/7107)，展示了实现过程及测试。
- **Windows 兼容性问题确实存在**：**Windows 11** 出现问题，Python 安装引导用户进入 Microsoft Store，表明存在兼容性障碍。
  
  - 讨论中提到了早前的 **sqlite issues**，强调了使用正确 Python 版本的必要性。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Microdiffusion 实现进展**：社区正热切期待 Microdiffusion 论文的实现，该项目设定了 **$2k** 的训练目标，并已获得 **7 天** 的 H100 计算资源，有望显著降低训练成本。
  
  - 讨论集中在预处理协助以及在实验准备后寻求短期改进。
- **数据预处理挑战凸显**：一位成员指出，由于 Hugging Face 的 **300GB** 限制，上传大型数据集存在困难，建议将数据分块或使用托管在 S3 上的 webdataset。
  
  - 他们的目标是通过根据长宽比将图像分类到多个数据集中，来高效地预处理数据并进行流式传输。
- **用于高效数据处理的 Webdataset**：参与者讨论了使用 [webdataset](https://github.com/webdataset/webdataset) 作为管理大型数据集的变通方案，从而实现与 PyTorch 的流式对接。
  
  - 一位成员坚持认为，webdataset 打包将增强对其预期的 **1TB** 数据集的管理。
- **Dinov2 分层优化**：讨论集中在**将 Dinov2 蒸馏到早期层**，从而提高图像相关下游任务的效率。
  
  - 值得注意的是，与仅依赖 **CLIP embedding 的 cross attention** 相比，该方法表现出更优越的性能。
- **推出用于超声心动图的 EchoPrime**：[EchoPrime](https://arxiv.org/abs/2410.09704) 作为一个基于多视图对比学习的模型出现，在 **1200 多万个视频-报告对**上进行了训练，解决了传统超声心动图 AI 的挑战。
  
  - 这一新的基础模型增强了心脏成像的性能和应用范围。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **实验动态 Few-shot Prompting**：动态 few-shot prompting 通过根据查询（query）检索相关示例而非使用固定集合，来增强 LLM 的微调（fine-tuning）（[更多详情](https://t.co/hqgxexq7PE)）。该方法改进了各种应用中的 prompt 上下文关联。
  
  - 参与者指向了一个相关的[讨论帖](https://twitter.com/llama_index/status/1846351135596335165)，强调了在该方法中相关示例的重要性。
- **Mistral 发布全新 Edge-Class 模型**：Mistral 推出了备受关注的 edge-class 模型，通过 'pip install llama-index-llms-mistralai' 提供首日支持（[安装链接](https://t.co/BdoNQmDtXD)）。这允许开发者快速将这些模型集成到他们的系统中。
  
  - 该公告引起了社区的关注，凸显了其在当前 AI 领域的相关性（[公告链接](https://twitter.com/llama_index/status/1846596827820576870)）。
- **使用 Azure 增强多模态 RAG 系统**：一份指南说明了如何利用 Azure AI Search 和 Azure OpenAI 结合 LlamaIndex 构建多模态 RAG 系统，指导如何提高检索准确性（[查看指南](https://t.co/RO5nQ79sqD)）。这份详尽的文档包含了实际实现的基准测试。
  
  - 该教程侧重于最大化不同 AI 系统间的上下文检索，提供了[这条推文](https://twitter.com/llama_index/status/1846668813980639343)中分享的有价值的技术。
- **优化 Neo4jPropertyGraphStore 的创建**：创建 **Neo4jPropertyGraphStore** 可能非常耗时，特别是在处理 **64,322 个节点**时，这引发了关于内存优化和模式（schema）简化的讨论。建议包括将 `refresh_schema` 设置为 false，以减少昂贵的模式相关调用。
  
  - 社区反馈表明，这些调整可以显著提升初始化期间的性能。
- **研究多 Agent 编排工作流**：用户询问如何在 LlamaIndex 中复制 OpenAI 的 Swarm 功能，重点是将工作流（workflows）作为核心方法。讨论引出了多 Agent 通信的示例，并辅以博客文章和 GitHub 仓库支持。
  
  - 这一探索旨在开发高效的解决方案，利用现有工作流在多个 Agent 之间编排动作。

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Mistral 庆祝 Ministral 模型发布**：在 **Mistral 7B** 发布一周年之际，**Mistral** 推出了两款边缘模型：**Ministral 3B** 和 **Ministral 8B**。这些模型专为设备端使用而设计，具备隐私优先的推理能力，并支持高达 **128k 的上下文长度**。
  
  - *社区对 **Ministral 3B** 未开放权重表示失望*，并对其与拥有非商业权重的 **Ministral 8B** 相比的潜在性能提出了质疑。
- **AI2 OLMo 实习岗位提供极具竞争力的薪资**：AI2 正在为 **OLMo** 项目招聘研究实习生，**薪资范围为 86,520 美元至 123,600 美元**。实习生将有机会在为期 **12 周的实习**中领导 NLP 和 Machine Learning 领域的重要研究。
  
  - 实习生可以自主定义研究项目并在*高水平期刊上发表论文*，这使得该机会在竞争激烈的环境中备受追捧。
- **Snailbot 扩展其功能**：**Snailbot** 现在正被用于**音频 Feed 帖子**，体现了其在内容共享方面增强的功能。
  
  - 这被视为一种*一举两得*的改进，用户对该 Bot 的新用例表示兴奋。
- **音频分发的挑战**：用户表达了在**音频内容分发**方面面临的挑战，强调需要有效的策略。
  
  - 一位用户幽默地将他们的问题比作某个流行笔记应用的梗图，表明了社区内普遍存在的挫败感。
- **Hackernews 曝光度难题**：关于在 **Hackernews 上发布内容的陷阱**一直存在担忧，特别是关于**链接可见性**以及对直接链接的潜在惩罚。
  
  - 成员们讨论了应对曝光度问题的复杂性，并建议采取避免直接链接的策略以增强内容参与度。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Gemini 免费版表现不佳**：用户报告了 [Gemini 免费版](https://gemini.free.url) 的*超时和失败*问题，对其声称的*每天 1.5B token* 的处理能力表示怀疑。
  
  - 正如几位成员推测的那样，*实际有效使用量*可能更接近 *0.05B token*。
- **Mistral 加入边缘模型竞争**：Mistral 推出了 *Ministral 3B* 和 *Ministral 8B* 模型，旨在应用于设备端，增强了 10B 以下量级模型的常识和推理能力。
  
  - 然而，*3B 模型*仅限 API 使用，限制了其在设备端的应用，并引发了关于限制性许可的批评。
- **Nvidia 的 Llama 3.1 Nemotron 令人瞩目**：据报道，Nvidia 的 *Llama 3.1 Nemotron 70B* 在各项 Benchmark 中均超越了 *GPT-4o* 和 *Claude Sonnet 3.5*，引起了社区的兴奋。
  
  - 辩论随之而来：面对这款尖端模型，*Sonnet 3.5 用户*是否还能维持其竞争力。
- **E2B 的 SDK 获得融资助力**：E2B 发布了 v1.0 SDK，并完成了令人印象深刻的 1150 万美元种子轮融资，目标是利用安全沙箱进行 AI 代码解释。
  
  - 这家初创公司声称每月运行数百万个沙箱，其著名合作伙伴包括 *Perplexity*。
- **呼吁开发 LLM 性能基准测试工具**：一位成员提出了开发类似于 *CPUBenchmark 风格*的工具，专门用于 LLM 比较，以改进目前有限的排行榜。
  
  - 现有的工具（如 *lmsys/hugging face 排行榜*）无法在模型之间进行有效的直接比较。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 社区每日激发灵感**：成员们从 **Cohere 社区**中获得了每日动力，非常赞赏其互助的氛围。
  
  - *“说实话，很多事情，尤其是这个社区的每一天！”* 反映了大家共享的积极情绪。
- **职位机会说明**：提醒大家，关于 **Cohere** 的职位咨询应转至 [careers page](https://cohere.com/careers)。
  
  - 成员强调了团队对于利用 ML/AI 技术解决现实世界挑战的**热情**。
- **明天参加 RAG++ AMA！**：继社区表现出极大兴趣后，另一场关于 **RAG** 开发的 **AMA**（由 **Ayush Thakur** 和 **Meor Amer** 主持）将于明天 **11:00 AM ET** 开始。
  
  - 该会议关联到 [RAG++ 课程](https://www.wandb.courses/courses/rag-in-production)，承诺将提供关于当前发展的**见解**。
- **Cohere Embed API 错误处理说明**：针对 **Cohere Embed API** 错误处理的咨询，建议在文档嵌入失败时根据特定错误代码实现重试逻辑。
  
  - *“错误可能导致整个批次的失败，”* 建议在管理 Embeddings 时要小心。
- **聊天机器人支持 Text-to-Speech 了！**：聊天机器人响应引入了 Text-to-Speech 功能，引发了热烈讨论，并为用户分享了 [setup guide](https://github.com/cohere-ai/cohere-toolkit/blob/main/docs/text_to_speech.md)。
  
  - *“太棒了！”* 是一位用户的热情回应，表明新功能得到了有效的采用。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Playground 受到用户喜爱**：成员们对 **Playground** 功能表达了由衷的喜爱，感谢 **Modular** 的改进和支持。欲了解更多信息，可以阅读 [Playground 文档](https://docs.modular.com/mojo/playground)。
  
  - 这些积极反馈凸显了社区意见在改进工具中的重要性。
- **预留社区展示会日期**：**社区会议**定于 **10月21日** 举行，届时将有现场展示环节，参与者可以演示他们的 **MAX** 和 **Mojo** 项目。每个时段持续 **5-10 分钟**，允许分享学习心得和反馈。
  
  - 这样的参与有助于催化开发者之间的协作和知识共享。
- **奇怪的 Mojo Bug 已修复**：一名成员发现了一个可复现的 **Mojo bug**，但随后自行修复了它，并提议将贡献加入更新日志。他们鼓励其他人报告类似问题以增强平台。
  
  - 这种主动的方法可以带来更快的 Bug 解决速度和更好的软件稳定性。
- **Inferencemax 项目简化 API**：一名成员分享了他们名为 [Inferencemax](https://github.com/teilomillet/inferencemax) 的新项目，旨在简化 LLM inference，尽管它可能无法完全满足现有的所有需求。代码使用 Python 编写，并计划进行性能改进。
  
  - 该项目反映了在创建更易用的 Inference API 领域所做的持续努力。
- **Jakub 为 MAX 开发的 Python API 引起关注**：关于 Jakub 对 MAX 的 Python API 贡献的咨询引导至了一个 [community meeting](https://youtu.be/Wm-x1or345I?t=5) 链接，他在会上发表了讲话。虽然该 API 尚未完全发布，但它在 Nightly Builds 中的出现旨在展示其易用性。
  
  - 此类讨论强调了对提高可用性的 API 开发的期待。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **需要矿产资源海报制作协助**：一名成员寻求制作关于 **mineral resources** 的大学项目海报的帮助，向社区征求指导。
  
  - 另一名成员建议他们在聊天中分享具体需求，以便获得更直接的支持。
- **SD3 在处理人体姿势时表现不佳**：讨论集中在 **SD3** 在处理躺姿或倒立姿势时的性能缺陷，指出其表现普遍较差。
  
  - 一位参与者强调，无论什么姿势，经常会出现变形，这表明是一个持续存在的问题。
- **LLM 忽略 Token 限制**：一位用户对 LLM 无法遵守 **token limits** 或停止命令表示沮丧，导致输出混乱。
  
  - 他们推测可能是 Prompt 模板存在问题，并邀请资深用户提供见解。
- **澄清 LyCORIS 与 LoRA 的混淆**：一名成员询问 **LyCORIS** 文件夹的用途，因为现在所有内容都移到了 **LoRA**，对此表示困惑。
  
  - 另一位用户回应解释说，该文件夹是以前扩展程序所必需的，现在已被 Auto1111 等新界面整合。
- **新 Web3 项目招聘职位**：更新分享了一个新 **Web3 project** 的启动，该项目正在招聘包括 Developer 和 Moderator 在内的多个职位，薪资具有竞争力。
  
  - 鼓励感兴趣的候选人直接联系以获取有关可用职位的更多具体信息。

 

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter GitHub Copilot 扩展建议**：一名成员提议创建一个 **Open Interpreter GitHub Copilot extension**，而另一名成员表示他们缺乏 **bandwidth** 去执行，但愿意指导社区的工作。
  
  - 他们鼓励社区内部协作，共同实现这个项目。
- **对 Mozilla AI 演讲的期待**：成员们对即将到来的 **Mozilla AI** 演讲表示期待，敦促大家将其添加到日历中。
  
  - 分享了活动链接以便访问。
- **应用关闭时报告 Kernel panic**：一名成员报告在关闭 Open Interpreter 应用时出现 **kernel panic**，促使 MikeBirdTech 建议创建一个专门的故障排除帖子。
  
  - 报告应附带所使用版本的详细信息，以便有效解决。
- **新的本地 LLMs 功能**：最近的更新现在支持通过 **Ollama** 轻松运行 [Hugging Face](https://huggingface.co) 上的任何 **GGUF** 模型，只需指向仓库即可。
  
  - 用户可以通过简单的命令运行 **Llama 3.2 3B**，使本地 LLMs 更加易于访问。
- **对本地 LLMs 更新的正面反馈**：成员们对直接运行模型的新功能表示热烈欢迎，强调这是对本地 LLMs 的重大增强。
  
  - 提到了对之前缺失功能的赞赏，特别是与 **Jan** 相关的部分。

 

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **单元测试 DSPy Workflow 系统**：一位成员宣布他们正在 **Discord** 频道中对一个 **DSPy 驱动的 Workflow 系统**进行单元测试。请关注该频道以获取测试过程的进度更新和反馈。
  - 此次测试旨在完善工作流并确保可靠性，鼓励社区对发现的问题提供建议。
- **dspygen 框架重大更新**：[dspygen](https://github.com/seanchatmangpt/dspygen) 框架近期发布了**重大更新**，该框架旨在 **dslmodel** 之外进行改进。这旨在增强 **GPT**、**BERT** 和 **LLaMA** 等语言模型的 **DSPy** 工作流。
  - 更新重点在于引入更多功能和优化，从而实现现有系统内更好的集成。
- **LightRAG 表现优于 GraphRAG**：根据[这篇论文](https://arxiv.org/abs/2410.05779)的详细描述，最近的观点认为 **LightRAG** 在有效性和**成本效率**方面比 **GraphRAG** 有显著提升。作者提出 **LightRAG** 解决了现有 **RAG** 系统的局限性，通过创新的图结构提高了**上下文感知（contextual awareness）**和信息检索能力。
  - 他们断言，这些创新降低了运营成本并提升了整体系统性能。
- **DSPy 与 GPT-O1+ 的集成取得进展**：更新后的文档引入了一个长篇 **RAG** 示例，用于构建一个关于技术主题的 **DSPy** **问答系统**。用户可以通过 `pip install -U dspy` 安装 **DSPy**，教程可在 [DSPy 文档](https://dspy-docs.vercel.app/docs/quick-start/getting-started-01)中找到。
  - 此次集成预计将简化工作流并提升 **DSPy** 框架内的用户体验。
- **改进文档编写方法**：关于即将进行的 **DSPy** 文档翻新的讨论已经展开，重点是改进节奏和风格。参与者正在考虑是使用 **HTML** 文档还是详细的 **notebooks**，并提到了**执行缓存（caches for execution）**的实用性。
  - 此次翻新旨在提高用户的清晰度和可访问性，使用户能够更轻松地查阅文档。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain 社区即将关闭**：目前的 **LangChain Discord** 社区将于 **2024 年 10 月 31 日**关闭，以为全新的用户空间腾出空间，新空间旨在更具**互动性**和**趣味性**。
  - 成员可以通过填写[此处](https://airtable.com/app9AB74Dql7uubL2/pagTKrmJu1rQRkJKV/form)的表格来关注更新，并鼓励通过 [community@langchain.dev](mailto:community@langchain.dev) 提供反馈。
- **API 路由建议征集**：一位成员正在寻求关于使用 **Agent** 将用户查询路由到不同 **API** 的指导，并提到他们在 **Docker Compose** 中设置了 **5 个 API**。
  - 此项咨询旨在优化其项目结构并提升用户与 **API** 交互的体验。
- **Playground 空白页困扰**：成员们反映了 **Playground** 中的一个重大问题，即带有 **Optional** 字段的输入类型会导致页面加载为空白，并在控制台中报错。
  - 该问题可能源于输入模式（schema）中的 **null** 类型与 ***jsonforms*** 冲突，从而阻碍了功能使用。
- **针对 Playground 问题已提交 GitHub Issue**：一位成员提交了 [GitHub Issue #782](https://github.com/langchain-ai/langserve/issues/782)，以跟踪与 **Optional** 字段导致加载失败相关的 **Playground** 问题。
  - 这是解决 **LangChain Playground** 内关键可用性问题的持续努力的一部分。
- **Remote Runnable 工具绑定咨询**：一位成员询问在 **Remote Runnable** 中缺失用于工具绑定的 **bind_tools()** 方法的问题，这为改进提供了契机。
  - 这一讨论可能为 **LangChain** 环境中更好的工具管理奠定基础。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **AIFoundry 寻求 GitHub 指导**：[AIFoundry.org](https://discord.gg/aSHN7W5E) 正在寻求关于其 GitHub 组织和设计的指导，旨在效仿 Axolotl 的精简方法。
  
  - Yulia 表达了希望获得指导的愿望，以增强他们专注于本地模型推理（local model inference）的开源计划。
- **Mistral 访问规则说明**：要访问 [Hugging Face](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410) 上新的 **Mistral-8B-Instruct-2410** 模型，用户必须提供联系方式并获得非标准用途的许可。
  
  - 访问权限取决于 Mistral AI 的同意，并呼吁用户查看其[隐私政策](https://mistral.ai/terms/)以确保合规。
- **L3.1 Ethereal Rainbow 发布风险**：[L3.1 Ethereal Rainbow](https://huggingface.co/invisietch/L3.1-EtherealRainbow-v1.0-rc1-8B) 仓库因包含敏感且潜在有害的内容而被标记，用户需谨慎使用。
  
  - 该仓库因其敏感材料引发了警告，用户应仔细考虑该内容的影响。
- **微调 L3.1 模型**：L3.1 模型已使用**超过 2.5 亿个 Tokens** 进行微调，并保持了 16k 的序列长度能力，增强了其在创意写作应用中的性能。
  
  - 这种对 **RP（角色扮演）和创意写作** 的关注，标志着在敏感语境下增强模型实际可用性的针对性努力。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **成员热议新论文**：社区对标题为 [arxiv:2410.06511](https://arxiv.org/abs/2410.06511) 的论文反响热烈，成员们认为这是一篇极佳的读物。
  
  - 一位成员确认他们仍在研读该论文，强调了其质量和社区的参与度。
- **对论文质量的一致好评**：关于该论文的总体评价非常积极，多位成员强调了其令人印象深刻的内容。
  
  - 一些人提到他们仍在深入研究细节，反映了大家对其见解的共同兴趣。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **LLM 在 zero-shot 优化方面表现出色**：最近的研究表明，**Large Language Models (LLMs)** 可以在**多目标优化**等复杂问题上执行 **zero-shot 优化**。
  
  - 这一应用可能在工程任务中发挥重要作用，例如**火箭喷嘴设计**和**风电场布局优化**。
- **认识 Language-Model-Based Evolutionary Optimizer (LEO)**：**LEO** 被介绍为一种利用 LLM 进行数值优化的新型种群算法（population-based approach），其表现与**基于梯度**和**无梯度方法**不相上下。
  
  - 然而，对输出中可能出现**幻觉（hallucination）**的担忧表明，在其应用中需要进行细致的管理。
- **社区热议 LLM 设计应用**：社区讨论反映了对 LLM 在**工程设计**中实际应用的浓厚兴趣，特别是关注其推理能力。
  
  - 成员们热衷于合作探讨 LLM 如何应对现实世界的工程挑战。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **AI Stewardship Practice 试点项目**：**MaRS Discovery District** 正在为 **AI Stewardship Practice 项目**提供免费名额，目标受众为 AI 领域的专业人士。
  
  - 该计划为希望对 AI 产生积极影响的**研究人员**、**企业家**和**教育工作者**提供微凭证（microcredential）；[更多信息请点击此处](https://programs.techstewardship.com/)。
- **征集 AI 课程试点参与者**：有机会加入该计划的课程试点，价值 **500 CAD**，鼓励感兴趣的参与者尽快响应。
  
  - 名额将根据跟帖回复情况分配，对于想要参与的人来说，迅速行动至关重要。

---

The **Alignment Lab AI Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。

---

The **LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。

---

The **MLOps @Chipro Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。

---

The **DiscoResearch Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。

---

The **Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。

---

The **AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。

---

# PART 2: 按频道详细摘要和链接

{% if medium == 'web' %}

### **HuggingFace ▷ #**[**announcements**](https://discord.com/channels/879548962464493619/897387888663232554/1296168878774419568) (1 条消息):

> - `Gradio 5.0 Launch`
> - `Sentence Transformers v3.2.0`
> - `HuggingChat Multimodal Update`
> - `FLUX LoRA Lab Introduction`
> - `LLM Evaluation Guidebook`

- **Gradio 5.0 发布，安全性增强**：我们刚刚发布了 [Gradio 5.0](https://www.producthunt.com/posts/gradio-5-0)，让创建生产级 Machine Learning Web 应用变得更加容易，并伴随全面的[安全性大修](https://x.com/Gradio/status/1844415295487869226)。每月超过 600 万次的下载量彰显了其日益增长的人气。
  
  - *秉持透明精神*，我们将向公众公开完整的安全报告。
- **Sentence Transformers v3.2.0 亮相**：[Sentence Transformers v3.2.0](https://x.com/tomaarsen/status/1844440420027335064) 现已发布，为 Embeddings 引入了 2 个新后端：ONNX 和 OpenVINO，有望实现 2x-3x 的加速。此次更新标志着两年来推理方面最大的版本发布，静态 Embeddings 可实现高达 **500 倍的加速**。
  
  - 查看深入介绍的新功能，包括使用 [Model2Vec](https://huggingface.co/blog/Pringled/model2vec) 实现 **10k 文本/秒** 的更快推理。
- **HuggingChat 迈向多模态**：HuggingChat 现已支持[多模态](https://x.com/mervenoyann/status/1844678895657685409)，加入了 Llama-Vision 11B Instruct，增强了交互可能性。这为平台内的用户体验带来了令人兴奋的新维度。
  
  - *这不是演习*，鼓励用户探索这一重大升级，以利用多模态能力。
- **FLUX LoRA Lab 隆重推出**：介绍 [FLUX LoRA Lab](https://x.com/multimodalart/status/1843612141951299979)，用户可以将多个 FLUX LoRA 混合并组合成独特的配置。\*🎲 函数\* 提供随机合并，带来一丝惊喜和创意。
  
  - 这种有趣的方法鼓励对 LoRA 进行实验，旨在激发创新和乐趣。
- **新版 LLM 评估指南发布**：一本新的 LLM 评估[指南](https://x.com/clefourrier/status/1844323838517252172)已经出版，提供实用见解和理论知识。该资源旨在支持用户更好地管理 Open LLM Leaderboard 并设计评估方案。
  
  - 该指南是 @huggingface 评估团队分享最佳实践和见解努力的一部分。

**提及的链接**：

- [来自 Gradio (@Gradio) 的推文](https://x.com/Gradio/status/1844415295487869226)：🔒 Gradio 5 变得更加安全了！🔒 继 Gradio 5 发布之后，我们很高兴能分享其最重要的改进之一：全面的 Security（安全）大修！🛡️ 随着 Gradio 变得...
- [来自 tomaarsen (@tomaarsen) 的推文](https://x.com/tomaarsen/status/1844440420027335064)：📣 Sentence Transformers v3.2.0 发布，这是两年来推理方面最大的版本更新！为 Embedding 模型新增了两个后端：ONNX（+ 优化与量化）和 OpenVINO，从而提升了速度...
- [来自 tomaarsen (@tomaarsen) 的推文](https://x.com/tomaarsen/status/1845875524297806143)：Model2Vec 通过将词汇表通过模型传递，利用 PCA 降低 Embedding 维度并应用 Zipf 权重，从 Sentence Transformer 中蒸馏出一个快速模型。生成的静态模型推理...
- [来自 @GoogleDevExpert (@GoogleDevExpert) 的推文](https://x.com/GoogleDevExpert/status/1844433596049744373)：超过 30 万个 Hugging Face Transformers 模型现已在 KerasNLP 中可用 🤗 由 GDE @ariG23498 牵头，这让开发者可以直接将 Gemma 和 Llama2 等模型加载到 KerasNLP 中 —— 开启了无限可能...
- [来自 Argilla (@argilla_io) 的推文](https://x.com/argilla_io/status/1844395445788999843)：🚀 Argilla ❤️ LlamaIndex：你现在可以监控 LlamaIndex Pipeline，并通过人类和 AI 反馈来改进它们！- RAG 的理想选择 - 全程可追溯 - 丰富的元数据收集。快来查看并尝试...
- [来自 Sayak Paul (@RisingSayak) 的推文](https://x.com/RisingSayak/status/1844358385560670359)：使用单张 24GB GPU 即可微调 5B 参数的视频模型 🍓 我们发布了 CogVideoX-Factory，这是一个包含内存优化脚本的仓库，用于微调 Cog 系列视频模型...
- [来自 Daniel Vila Suero (@dvilasuero) 的推文](https://x.com/dvilasuero/status/1846191037343060305)：📢 思考 LLMs 数据流水线复制。今天，来自 @AIatMeta 的 Jason Weston 分享了他们的新工作：训练 LLMs 进行思考与响应，并应用迭代式 DPO。我刚刚实现了数据生成...
- [来自 merve (@mervenoyann) 的推文](https://x.com/mervenoyann/status/1844678895657685409)：这不是演习 💥 @huggingface HuggingChat 现在支持多模态了，搭载了来自 @Meta 的 Llama-Vision 11B Instruct！🤗
- [来自 apolinario 🌐 (@multimodalart) 的推文](https://x.com/multimodalart/status/1843612141951299979)：✨ 介绍 FLUX LoRA Lab 🧪🔬 混合并组合多个 FLUX LoRA，进行你自己的疯狂 LoRA 炼金（你也可以使用 🎲 功能随机合并 2 个 LoRA，以获得惊喜、新奇和...
- [来自 Quentin Lhoest 🤗 (@qlhoest) 的推文](https://x.com/qlhoest/status/1843972996211638373)：新博客文章：轻松扩展基于 AI 的数据处理。FineWeb-Edu 数据集源自对 45TB (🤯) FineWeb 数据的处理。它使用 Language Model 来对文本的教育水平进行分类 😭😭 ...
- [来自 merve (@mervenoyann) 的推文](https://x.com/mervenoyann/status/1845849356613583152)：我们现在有了支持视频输入的 LLMs 排行榜，而且其中大多数是开源模型 🔥👏 我们回来了，家人们。
- [来自 Quentin Lhoest 🤗 (@qlhoest) 的推文](https://x.com/qlhoest/status/1845848814197837880)：我的新应用发布了！！✨ Common Crawl Pipeline Creator ✨ 轻松创建你的流水线：✔运行文本提取✂️ ✔定义语言过滤器🌐 ✔自定义文本质量💯 ✔查看实时结果👀 ✔获取 Python 代码...
- [来自 Clémentine Fourrier 🍊 (@clefourrier) 的推文](https://x.com/clefourrier/status/1844323838517252172)：亲爱的 LLM 推特用户，我为你们制作了一份评估指南！🥳 https://github.com/huggingface/evaluation-guidebook 目标：分享 @huggingface 评估团队的实践见解和理论知识...

---

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1295847473935417365) (143 条消息🔥🔥):

> - `AI 模型性能`
> - `在 Hugging Face 中使用 Ollama`
> - `Gradio 文档问题`
> - `TTS 模型推荐`
> - `AI 在职场中的角色`

- **关于 AI 模型性能的讨论**：成员们讨论了配置有 **72GB VRAM** 和 **128GB DDR4** RAM 的模型设置的假设性能，询问其处理速度是否能达到 **5-6 t/s**。
  
  - 此外，还提到了自定义 **PyTorch** 线性层及其与 **Autograd** 的集成，以实现自动梯度。
- **在 Hugging Face 中使用 Ollama**：分享了 **Ollama** 的使用介绍，它允许直接从计算机与 GGUF 模型交互，而无需创建新的 `Modelfiles`，并强调了其简单的命令语法。
  
  - 讨论强调了其易用性，因为 **Ollama** 声称可以运行 Hugging Face 上 4.5 万个公开 GGUF Checkpoints 中的任何一个。
- **Gradio 文档的问题**：一位用户对 **Gradio** 文档的可用性提出了担忧，指出深色背景上的文本存在阅读问题，且维护者缺乏回复。
  
  - 这凸显了社区和维护者在改进文档方面需要更好的互动。
- **Hugging Face 上的 TTS 模型推荐**：社区成员征求 Hugging Face 库中 **Text-to-Speech (TTS)** 模型的推荐，促使一位用户指向了热门模型。
  
  - 特别是 **SWivid/F5-TTS** 模型被强调为可用于 TTS 任务的更新选项。
- **AI 在就业市场中的角色**：关于职场中 AI 角色的对话强调，虽然像大语言模型（LLM）这样的 AI 工具正在兴起，但就业市场总是在演变的。
  
  - 成员们指出，适应新工具和技术的重要性，类似于各种工作中向电子表格软件的过渡。

**提到的链接**：

- [CogVLM2: Bringing Deeper Visual and Language Understanding to AI](https://medium.com/@ryanfoster_37838/cogvlm2-bringing-deeper-visual-and-language-understanding-to-ai-2d04d95797a9)：AI 在理解文本方面已经取得了长足进步，但当涉及到将视觉数据（如图像和视频）与语言融合时，我们已经……
- [Hugging Face - Learn](https://huggingface.co/learn)：未找到描述
- [Use Ollama with any GGUF Model on Hugging Face Hub](https://huggingface.co/docs/hub/en/ollama)：未找到描述
- [Models - Hugging Face](https://huggingface.co/models?pipeline_tag=text-to-speech&sort=trending)：未找到描述
- [openr/reports/OpenR-Wang.pdf at main · openreasoner/openr](https://github.com/openreasoner/openr/blob/main/reports/OpenR-Wang.pdf)：OpenR：一个用于大语言模型高级推理的开源框架 - openreasoner/openr
- [Home](https://openreasoner.github.io.)：一个用于推进大语言模型推理的开源框架

---

### **HuggingFace ▷ #**[**today-im-learning**](https://discord.com/channels/879548962464493619/898619964095860757/1295844560643166218) (9 条消息🔥):

> - `AI 虚拟网红开发`
> - `图像生成技术`
> - `具有个性的语言模型`

- **构建类似于 Aitana Lopez 的 AI 虚拟网红**：一位成员正尝试创建受西班牙 AI 模特 **Aitana Lopez** 启发的 AI 虚拟网红，但在生成具有不同属性的多样化图像时面临挑战。
  
  - 他们正在寻求指导，以实现更真实的 Instagram 内容，而不是重复的镜头。
- **使用 ControlNet 保持图像一致性**：有人建议利用 **ControlNet** 在为 AI 虚拟网红的个人资料生成图像时保持一致性。
  
  - 这一潜在解决方案被提出作为协助创建符合预期愿景的独特图像的方法。
- **AI 开发中的挑战**：一位成员指出，期望他人在复杂任务上提供帮助是很困难的，这表明对求助请求的清晰度不足。
  
  - 这一评论反映了该领域新手在寻求实际帮助时面临的更广泛挑战。
- **关于语言模型的问题**：一位成员询问 **Hugging Face** 是否可以根据大型提示词（Prompts）数据集生成具有鲜明个性的短语。
  
  - 他们表示有兴趣开发一种不仅能生成文本，还能体现反映输入数据个性的语言模型。
- **探索用于短语生成的 Transformers**：围绕使用 **Transformers** 根据特定提示词生成类似短语的能力展开了讨论。
  
  - 这反映了利用先进 AI 技术进行创意应用的持续兴趣。

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1295971442252447766) (4 条消息):

> - `GroupFi-Chatbox`
> - `PaliGemma GitHub Repository`

- **GroupFi-Chatbox：一个 AI 增强的消息解决方案**：一名成员分享了 [GroupFi-Chatbox 仓库](https://github.com/TanglePay/GroupFi-Chatbox/blob/dev/packages/sdk/README.md) 的 GitHub 链接，并指出这是一个他们正寻求通过 AI 功能进行增强的消息解决方案。
  
  - 该分享的仓库包含详细的 README，并邀请大家为其开发做出贡献。
- **发现 PaliGemma 的更多乐趣**：另一位成员推荐了 [PaliGemma 仓库](https://github.com/ThinamXx/PaliGemma)，并表示如果你喜欢 GroupFi-Chatbox，你一定会爱上这个。
  
  - 一位用户提到他们刚刚在 GitHub 上 Star 了该仓库，表达了对分享链接的赞赏，确认了他们的兴趣。

**提到的链接**：

- [GroupFi-Chatbox/packages/sdk/README.md at dev · TanglePay/GroupFi-Chatbox](https://github.com/TanglePay/GroupFi-Chatbox/blob/dev/packages/sdk/README.md)：通过在 GitHub 上创建账号来为 TanglePay/GroupFi-Chatbox 的开发做出贡献。
- [GitHub - ThinamXx/PaliGemma: Reading PaliGemma paper ...](https://github.com/ThinamXx/PaliGemma)：阅读 PaliGemma 论文... 通过在 GitHub 上创建账号来为 ThinamXx/PaliGemma 的开发做出贡献。

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1296152223943757867) (4 条消息):

> - `Video Inference using Vision Transformers`
> - `Accelerating LLM Training`
> - `In-Depth Question Answering Evaluation App`

- **视频推理资源**：一位成员在他们的 [GitHub](https://github.com/0xD4rky) 上分享了一系列关于使用 **Vision Transformers** 进行**视频推理 (video inferences)** 的学习资源。
  
  - 该资源旨在指导用户在自己的项目中实现视频推理技术。
- **LLM 训练流程平台**：一位成员对一个旨在**加速 LLM 训练过程**的平台感到兴奋，该平台通过管理包括 Hugging Face 和 S3 在内的各种存储解决方案中的数据来实现。
  
  - 他们提供了演示 (demo)，并热衷于根据社区需求定制平台；可以通过 **Mail** 或 **LinkedIn** 与他们联系。
- **关于学习应用的新 Medium 文章**：一位成员发布了他们的第一篇 Medium 文章，讨论了**深度问答评估应用 (In-Depth Question Answering Evaluation App)**，该应用旨在为学习者提供**实时反馈**。
  
  - 该应用利用 **Gemini 1.5 Pro** 进行问答，旨在增强用户的学习体验，并感谢 Dr. Fady AlNajjar 提供的创意。

**提到的链接**：

- [Enhancing Learning Through Real-Time Feedback: In-Depth Question Answering Evaluation App](https://medium.com/@d.isham.ai93/enhancing-learning-through-real-time-feedback-in-depth-question-answering-evaluation-app-4f68c423e496)：在在线学习和自我提升的世界中，拥有有效的工具来评估进度至关重要。无论你是在学习……
- [0xD4rky - Overview](https://github.com/0xD4rky)：一个生活在编译和终止循环中的男孩 - 0xD4rky

---

### **HuggingFace ▷ #**[**reading-group**](https://discord.com/channels/879548962464493619/1156269946427428974/1296069598629859442) (3 条消息):

> - `Reading Group Reminder`
> - `Participant Excitement`

- **读书小组定于明天举行**：发布了一条提醒，下一次读书小组讨论已排定在明天，邀请大家加入讨论。
  
  - 鼓励参与者出席，突显了社区对即将举行的活动的热情。
- **参与者的兴奋之情**：一位参与者表达了对即将到来的读书小组的兴奋，并确认将会出席。
  
  - 这展示了小组内对于协作讨论的积极参与和期待。

---

### **HuggingFace ▷ #**[**NLP**](https://discord.com/channels/879548962464493619/922424173916196955/1295954172344139837) (11 条消息🔥):

> - `Fine-tuning LLMs`
> - `Transformers library contribution`
> - `Special tokens usage`
> - `Attention masks`
> - `GPU requirement for debugging`

- **澄清 LLMs 的 Fine-tuning**：一位成员询问 LLMs 与 BERT 不同，在 Fine-tuning 期间由于不使用 labels，如何识别 prompt 中响应的起始位置。
  
  - 另一位成员建议添加 **special tokens** 有助于标识序列中的 user/system 轮次。
- **对系统响应使用 Attention Masks**：一位用户指出 **attention masks** 的效用，可以将更新仅聚焦于序列中的系统响应。
  
  - 这种方法有利于在处理 **恶意用户输入** 时确保对齐。
- **贡献 Transformers 库**：一位成员表示有兴趣在 GitHub 上为 **transformers library** 做出贡献，但询问是否必须使用 GPU。
  
  - 对方澄清说，除非调试特定案例，否则不需要 GPU，Colab 上的 **免费 GPU 时长** 也是一个可行的选择。
- **调试边缘情况与模型大小**：讨论了较大 batch sizes 可能改变输出的边缘情况，特别是与 GPU **compilation** 相关的问题。
  
  - 然而，有人指出许多贡献可以使用较小的模型和 CPU 资源完成。

 

---

### **HuggingFace ▷ #**[**diffusion-discussions**](https://discord.com/channels/879548962464493619/1009713274113245215/1296025552775286876) (1 条消息):

> - `Hugging Face tutorial`
> - `DiffusionPipeline`
> - `DDPM model`

- **关于 Diffusers 的 Hugging Face 教程**：一位成员分享了一个 [Hugging Face 教程](https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline)，解释了如何利用 **Diffusers** 工具箱构建自定义扩散系统。
  
  - 该教程强调了工具箱的用户友好性，同时介绍了模型和 schedulers 等核心组件。
- **理解 DiffusionPipeline**：**DiffusionPipeline** 可用于捆绑模型和 schedulers，但也可以解耦以创建新系统，提供了设计上的灵活性。
  
  - 这允许用户根据特定需求定制其扩散过程。
- **轻松运行基础 Pipeline**：用户只需四行代码即可利用 **DDPMPipeline** 生成图像，突显了运行模型时易于上手的语法。
  
  - 例如，可以直接在代码中以极少的设置访问并使用 DDPM 论文中的原始模型。
- **教室环境的模型大小考虑**：一位成员提到 DDPM 模型（约 450Mb）应该非常适合教室环境，确保所有学生都能访问。
  
  - 他们幽默地指出，可靠的 WiFi 对促进课程期间的模型使用非常重要。

 

**提到的链接**：[Understanding pipelines, models and schedulers](https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline)：未找到描述

 

---

### **HuggingFace ▷ #**[**gradio-announcements**](https://discord.com/channels/879548962464493619/1014577787039924226/1295876349201875005) (1 条消息):

> - `Gradio 5 themes`

- **Gradio 5 展示新主题**：Gradio 5 包含了多个新主题，增强了用户的视觉体验。查看 [展示所有主题的视频](https://link.to.video) 以了解新内容。
  
  - 这些主题承诺为用户提供更加个性化和引人入胜的界面。
- **包含多种视觉风格**：新的 Gradio 5 主题具有一系列视觉风格，满足不同用户的偏好。这一补充使得应用程序呈现给最终用户的方式具有更大的自定义空间。
  
  - 成员们对这些主题如何改善用户界面设计表示兴奋。

 

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1295830365142913098) (158 条消息🔥🔥):

> - `Perplexity AI 功能`
> - `Reasoning Mode`
> - `Perplexity Purchases`
> - `AI 模型用户体验`
> - `UI 改进`

- **Perplexity AI 推出购物功能**：Perplexity 正在通过其新功能 'Perplexity Purchases' 整合购物能力，旨在简化购买流程并有效核对价格。
  
  - 这引起了褒贬不一的反应，一些用户更喜欢该平台原始的搜索功能，而不是演变成购物服务。
- **对 Reasoning Mode 的正面反馈**：用户称赞 Reasoning Mode 在编程任务中的有效性，指出它展示了深度的分析能力。
  
  - 几位成员分享了该功能的成功案例，强调了它在生成准确结果方面的可靠性。
- **关于 AI 扩展程序的讨论**：强调了用于增强 Perplexity 性能的扩展程序（如 'Complexity'），并提出了 'vettedai' 和 'gigabrain' 等有效的替代方案。
  
  - 用户分享了使用这些工具跨源参考各种来源（包括社交媒体和 Reddit 帖子）的积极体验。
- **对 AI 聊天空间用户体验的担忧**：用户对聊天空间的动态表示担忧，觉得环境有时像幼儿园，而不是严肃的 AI 论坛。
  
  - 这引发了关于需要更好的审核和用户互动以保持对 AI 话题关注的讨论。
- **用户界面和功能的改进**：成员们对 Perplexity 用户界面的持续改进表示乐观，特别是在使模型切换等功能更易于访问方面。
  
  - 虽然一些用户指出需要简化流程，但他们对新 UI 增强背后的周全考虑表示赞赏。

**提到的链接**：

- [《纽约时报》警告 AI 搜索引擎 Perplexity 停止使用其内容](https://www.theverge.com/2024/10/15/24270774/new-york-times-cease-and-desist-letter-perplexity-ai-search-engine)：Perplexity 辩称其正在“呈现事实内容”。
- [重新设计的 Spaces 和 Purchases 即将登陆 Perplexity](https://www.testingcatalog.com/redesigned-spaces-and-purchases-coming-soon-to-perplexity-users/)：探索 Perplexity 即将推出的功能：翻新后的 Spaces 和 Perplexity Purchases。享受免费送货和流畅的 UI。敬请关注发布！
- [来自 Perplexity (@perplexity_ai) 的推文](https://x.com/perplexity_ai/status/1846287953599123757)：Perplexity for Finance：实时股票报价。历史收益报告。行业同行对比。公司财务详细分析。均配有令人愉悦的 UI。祝你在研究市场时感到愉快...
- [《纽约时报》警告 AI 搜索引擎 Perplexity 停止使用其内容](https://www.theverge.com/2024/10/15/24270774/new-york-times-cease-and-desist-letter-perplexity-ai-se)：Perplexity 辩称其正在“呈现事实内容”。
- [Arangutan Monkey GIF - Arangutan Monkey Dancing - Discover & Share GIFs](https://tenor.com/view/arangutan-monkey-dancing-gif-15130385)：点击查看 GIF
- [来自 Aravind Srinivas (@AravSrinivas) 的推文](https://x.com/AravSrinivas/status/1843889051377774941)：以我最喜欢的数学家命名会议室是长久以来的梦想成真。Ramanujan 是好奇心的缩影，这也是 Perplexity 所代表的核心理念。
- [GitHub - pnd280/complexity: ⚡ 增强你的 Perplexity.ai](https://github.com/pnd280/complexity)：⚡ 增强你的 Perplexity.ai。通过在 GitHub 上创建账号，为 pnd280/complexity 的开发做出贡献。

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1295864019214925825) (12 条消息🔥):

> - `Green Power Ranger`
> - `Understanding APIs`
> - `Starlink Gigabit Speed Plan`
> - `TikTok AI Moderators`
> - `Oura Ring 4 Review`

- **对 Green Power Ranger 感到好奇**：一位成员分享了一个讨论 Green Power Ranger 的[搜索结果](https://www.perplexity.ai/search/why-was-the-green-power-ranger-VF81xNApS7CZqMS1L0yhsQ#0)，引发了大家对该角色背景的兴趣。
  
  - *是什么让这个角色如此受欢迎？*
- **重温 API**：多位成员分享了关于“什么是 API？”这一问题的相同[搜索结果](https://www.perplexity.ai/search/what-is-an-api-6HaQAJlXRGOWBgQd3L7Iyg#0)。
  
  - 这反映了人们对理解这一核心技术的兴趣日益增长。
- **对 Starlink Gigabit Speed 计划的热情**：成员们分享了关于 [Starlink Gigabit Speed Plan](https://www.perplexity.ai/page/starlink-gigabit-speed-plan-knyorEQ7SYG11t4a.dd2Ig) 的信息，并对其有效性提出了疑问。
  
  - 成员们讨论了这次升级对互联网连接的影响。
- **TikTok 向 AI 审核转型**：一位成员分享了一段关于 TikTok 转向 AI 审核员（AI moderators）新闻的视频，展示了内容审核领域不断演变的格局。
  
  - 这一举措引发了关于自动化与人工监督之间平衡的讨论。
- **Oura Ring 4 评测引发关注**：一位成员发布了 [Oura Ring 4 评测](https://www.perplexity.ai/page/oura-ring-4-review-5U7Rj9.hR3W0MRa_OmQgbQ)的链接，对其功能产生了浓厚兴趣。
  
  - 用户很好奇这个更新版本与早期型号相比如何。

 

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1296017987777593385) (6 条消息):

> - `search_domain_filter issue`
> - `Healthcare use case inquiries`
> - `LFM 40B API availability`

- **search_domain_filter 未按预期工作**：一位成员表示，在 **search_domain_filter** 中添加域名后并未获得预期结果，反而生成了不相关的内容，这让他感到沮丧。
  
  - 另一位成员确认该参数是有效的，并澄清如果未找到相关信息，模型可能仍会依赖其通用知识。
- **关于医疗保健项目 BAA 的咨询**：一位成员询问 **Perplexity** 是否会为医疗保健用例签署商业伙伴协议（BAAs），因为他们计划在美国构建企业级解决方案。
  
  - 在收集的消息中没有对该查询的直接回复。
- **通过 API 获取 LFM 40B 的可用性**：在讨论中，一位成员询问是否可以通过 API 访问 labs.perplexity.com 中的 **LFM 40B** 模型。
  
  - 针对这一特定查询，目前没有提供任何回复。

 

---

### **OpenRouter (Alex Atallah) ▷ #**[**announcements**](https://discord.com/channels/1091220969173028894/1092729520181739581/1296096944246100049) (2 条消息):

> - `Grok 2 Maintenance`
> - `NVIDIA Nemotron 70B Performance`

- **Grok 2 暂时停机维护**：xAI 已将 **Grok 2** 下线进行为期几小时的临时维护，导致用户在尝试访问时会收到 **404 错误**。
  
  - 当模型准备好恢复时，*将会发布公告*。
- **NVIDIA Nemotron 70B 在基准测试中占据主导地位**：**Nemotron 70B** 在多项评估中超越了 **Llama 3.1 405B**、**GPT-4o** 和 **Claude 3.5 Sonnet**：Arena Hard 评分为 **85.0**，而后者分别为 **79.2** 和 **79.3**。
  
  - 欲了解更多详情，请在[此处](https://openrouter.ai/nvidia/llama-3.1-nemotron-70b-instruct)尝试，并查看关于开源领域这一重大进展的[公告](https://x.com/OpenRouterAI/status/1846651197802881094)。

 

**提到的链接**：[来自 OpenRouter (@OpenRouterAI) 的推文](https://x.com/OpenRouterAI/status/1846651197802881094)：开源界的重大日子：NVIDIA Nemotron 70B 在多项评估中击败了 Llama 405B、GPT-4o 和 Claude 3.5 Sonnet：Nemotron 70B vs Claude 3.5 vs GPT4o: > Arena Hard: 85.0 | 79.2 ...

 

---

### **OpenRouter (Alex Atallah) ▷ #**[**app-showcase**](https://discord.com/channels/1091220969173028894/1092850552192368710/1295870740348014683) (1 条消息):

> - `ChatGPT advanced voice mode`
> - `Personalized AI learning`
> - `Self-learning with AI`
> - `Vocabulary teaching examples`

- **ChatGPT voice mode 结合《火影忍者》教授词汇**：一位用户展示了使用 **ChatGPT advanced voice mode** 结合 **《火影忍者》（Naruto）中的案例** 来教授新词汇，并称这种体验“简直太疯狂了！”
  
  - 他们分享了一个 [演示链接](https://x.com/ahmetdedeler101/status/1846305587442995446) 以获取关于这种教学方法效果的反馈。
- **个性化 AI 学习的未来**：该用户对个性化 **AI 学习** 表达了兴奋之情，预测它将成为教育领域的革命性力量。
  
  - 他们指出，这些新的语音模型“效果惊人地好”，预示着即将到来的重大创新。
- **AI 对自主学习的影响**：对话强调了 **AI** 在自主学习过程中的变革力量，并重点提到了 **语音模型** 的进步。
  
  - *看看很快会出现什么新东西会非常有趣*，这表明教育技术领域未来充满了潜在的发展空间。

**提到的链接**：[来自 Ahmet ☕ (@ahmetdedeler101) 的推文](https://x.com/ahmetdedeler101/status/1846305587442995446)：ChatGPT voice mode 结合《火影忍者》的例子教我词汇。个性化 AI 学习就是未来。效果好得令人震惊 😂

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1295824032549044344) (168 条消息 🔥🔥):

> - `Grok 2 Issues`
> - `Infermatic Provider Problems`
> - `Yi Lightning and Model Performance`
> - `OpenRouter Credit and API Key Questions`
> - `Mistral's New Models`

- **Grok 2 离线**：Grok 2 似乎已下线，因为 xAI 撤回了 API，导致用户因无法访问而感到沮丧。
  
  - 一位用户对其缺失表示遗憾，声称它在 Python 编程和聊天机器人方面的表现优于 Llama 3.2 等其他模型。
- **Infermatic 提供商面临网络问题**：Infermatic 的提供商正经历网络问题，导致模型产生乱码回复，特别是在达到 8k 上下文限制后。
  
  - 用户被告知提供商正在努力回滚其构建版本，以解决影响服务的 VLLM 推理问题。
- **Yi Lightning 模型性能受到质疑**：一些用户对 Yi Lightning 报告的性能持怀疑态度，注意到评估结果与预期输出之间可能存在差异。
  
  - 讨论围绕该模型的成功是真实的，还是通过操纵评估指标（刷榜）产生的。
- **OpenRouter 额度与 API Key 混淆**：新用户报告在购买额度和管理 API Key 方面存在困难，关于额度过期和使用的信息较为混乱。
  
  - 针对 Key 与额度的功能进行了澄清，用户对系统的复杂性表达了不满。
- **Mistral 发布新款“口袋级 LLM”**：Mistral 推出了两款新模型 Ministral 3B 和 8B，旨在用于边缘计算用例，并承诺增强性能。
  
  - 这些模型支持更大的上下文长度，旨在扩展知识和推理任务的能力。

**提到的链接**：

- [来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文](https://x.com/reach_vb/status/1846484958342168953?t=_R6PYDOgIfwK_krija_HSg&s=19)：我们回来了！：Nvidia Nemotron 70B - 击败了 Llama 3.1 405B, GPT4o & Claude 3.5 Sonnet！🔥 评测 (Nemotron 70B vs Claude 3.5 vs GPT4o) > Arena Hard - 85.0 vs 79.2 vs 79.3 > AlpacaEval 2 LC...
- [Un Ministral, des Ministraux](https://mistral.ai/news/ministraux/)：介绍世界上最好的边缘模型。
- [OpenRouter | docs.ST.app](https://docs.sillytavern.app/usage/api-connections/openrouter/)：由于地理锁定或等待名单而无法访问 OpenAI / Claude API？使用 OpenRouter。
- [来自 Rohan Paul (@rohanpaul_ai) 的推文](https://x.com/rohanpaul_ai/status/1846242281973486063?t=8tTgPB49KYWm6wAvEAkw-Q&s=19)：Andrej Karpathy 谈论极小尺寸蒸馏模型的重要性（即使是 1Bn 参数模型也应该足够好）。视频来源 - 原始视频来自 "No Priors: AI, Machine Learning, Tec...
- [OAuth PKCE | OpenRouter](https://openrouter.ai/docs/oauth)：通过 OAuth 进行安全用户身份验证。
- [Llama 3.1 Nemotron 70B Instruct - API, Providers, Stats](https://openrouter.ai/nvidia/llama-3.1-nemotron-70b-instruct)：NVIDIA 的 Llama 3.1 Nemotron 70B 是一款旨在生成精确且有用响应的语言模型。通过 API 运行 Llama 3.1 Nemotron 70B Instruct。

---

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1295825197181436018) (82 条消息🔥🔥):

> - `INTELLECT-1 发布`
> - `Unsloth 训练改进`
> - `Mistral 8B 模型支持`
> - `训练中的 Quantization 技术`
> - `Modelscope 与 Swift 讨论`

- **INTELLECT-1 发布，助力 Decentralized Training**：[INTELLECT-1](https://www.primeintellect.ai/blog/intellect-1) 的发布邀请各方为一个专注于 Decentralized Training 的 10B 参数模型贡献力量，目标是实现开源 AGI。
  
  - 这一进展紧随 [OpenDiLoCo](https://www.primeintellect.ai/blog/opendiloco) 的发布，通过将参数规模从 1B 扩展到 10B，增强了全球分布式 AI 模型训练。
- **Unsloth 训练显示出显著改进**：用户确认 `unsloth_train` 的收敛效果显著优于之前的方法，并希望未来能支持 `resume_from_checkpoint=True`。
  
  - 反馈表明社区非常看重这些增强功能，但也有人询问为什么不直接扩展旧的 `UnslothTrainer` 来增加新功能。
- **社区询问 Mistral 8B 支持情况**：针对 Unsloth 与新 [Mistral 8B 模型](https://mistral.ai/news/ministraux/) 的兼容性提出了疑问，回复指出架构差异需要进一步检查。
  
  - 社区成员对新模型的尺寸和端侧计算（On-device Computing）能力表示赞赏，期待进一步的更新。
- **Quantization 技术的探索**：讨论强调了在混合 QLoRA 等 Quantization 方法时应用 Full-fine-tune 技术的挑战，用户分享了他们在 Layer Tuning 方面的经验。
  
  - 对于在不进行大量定制的情况下，将某些层 Quantizing 而保持其他层完全可训练是否可行，存在一些怀疑。
- **Modelscope 和 Swift 框架评估**：成员们对 Modelscope 仓库和 Swift 框架提供了褒贬不一的反馈，提到了存在的问题，但也推荐将其详尽的文档用于初学者学习。
  
  - 用户对稳定性表示担忧，指出尽管这些平台很有用，但仍存在持续性的问题。

**提到的链接**：

- [Un Ministral, des Ministraux](https://mistral.ai/news/ministraux/)：介绍世界上最出色的边缘模型。
- [INTELLECT–1: Launching the First Decentralized Training of a 10B Parameter Model](https://www.primeintellect.ai/blog/intellect-1)：我们很高兴发布 INTELLECT-1，这是首个 10B 参数模型的 Decentralized Training 运行，邀请任何人贡献算力并参与。这让我们离……更近了一步。
- [Daniel Han (@danielhanchen) 的推文](https://x.com/danielhanchen/status/1846235913443262891)：修复了一个导致在大 Gradient Accumulation 尺寸下所有训练 Loss 发散的 Bug。1. 最初由 @bnjmn_marie 报告，GA 在数学上应该等同于 Full Batch 训练……
- [Swift DOCUMENTATION — swift 2.5.0.dev0 文档](https://swift.readthedocs.io/en/latest/index.html)：未找到描述

---

### **Unsloth AI (Daniel Han) ▷ #**[**off-topic**](https://discord.com/channels/1179035537009545276/1179039861576056922/1296032935501234178) (1 条消息):

> - `开源数据生成包`
> - `Claude Workspace 工具`

- **寻求开源数据生成工具**：一位成员正在寻求任何**开源包**或项目的推荐，以增强其**高质量数据生成 Pipeline**。
  
  - 他们提到正在使用 **Claude Workspace** 以及各种实用脚本，表明需要更集成的解决方案。
- **使用 Claude Workspace 处理数据工具**：讨论强调了 **Claude Workspace** 的使用，它为管理数据流程提供了各种实用脚本。
  
  - 这表明成员们正依赖 Claude 作为基础，但正在寻找更强大的开源解决方案来提升其数据生成任务。

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1295851768961433610) (58 条消息🔥🔥):

> - `Model Saving Issues` (模型保存问题)
> - `Installation Problems` (安装问题)
> - `Fine-Tuning Llama Models` (微调 Llama 模型)
> - `Windows Setup Requirements` (Windows 设置要求)
> - `Handling Long Contexts` (处理长上下文)

- **模型保存问题困扰用户**：用户在使用 `model.save_pretrained_gguf` 保存模型时遇到了复杂情况，输出了非预期结果，导致对模型完整性产生困惑。
  
  - 另一位用户建议，问题可能与合并 LoRA adapter 的方法有关，并指出不正确的保存程序通常会导致性能下降。
- **安装问题困扰 Unsloth 用户**：新用户在安装 Unsloth 期间面临循环依赖问题，特别是各依赖项对 PyTorch 版本的要求不一。
  
  - 用户寻求关于哪个 PyTorch 版本兼容的帮助，最终确认成功安装需要 2.4 版本。
- **微调 Llama 模型引发疑问**：讨论涉及在各种数据集上微调 Llama 3.1，包括初步测试以及使用非文本数值数据进行训练的有效性。
  
  - 关于根据 GPU 能力使用序列长度的查询，暗示了在实施微调策略时对显存限制的更广泛担忧。
- **Windows 设置需要额外配置**：Windows 用户注意到必须安装带有 Linux 发行版的 WSL 2 才能正确设置模型训练环境。
  
  - 提供了在 Ubuntu 上安装 Miniconda 等额外工具及相关依赖项的指导，以减轻设置问题。
- **训练中处理长上下文**：有人指出，模型中较长的上下文长度需要更多的 VRAM 和性能容量，特别是对于具有高内存限制的用户。
  
  - 建议强调，填满上下文限制可能会使系统资源紧张，特别是对于那些在消费级 GPU 上进行训练的用户。

**提到的链接**：

- [Google Colab](https://colab.research.google.com/drive/1xb9biGJ5fssjmCCHpUg-_otLUfnPrtPp#scrollTo=3jqvDScFcVTn)：未找到描述
- [Unsloth Notebooks | Unsloth Documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks)：请参阅下面的列表以获取我们所有的 notebook：
- [Unsloth Documentation](https://docs.unsloth.ai/get-started/installation/pip-install),)：未找到描述
- [Load](https://huggingface.co/docs/datasets/en/loading#csv)：未找到描述
- [unsloth/unsloth/models/gemma.py at main · unslothai/unsloth](https://github.com/unslothai/unsloth/blob/main/unsloth/models/gemma.py#L145-L151)：微调 Llama 3.2, Mistral, Phi & Gemma LLM 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth
- [Add support for passing in `inputs_embeds` into `generate` function · Issue #862 · unslothai/unsloth](https://github.com/unslothai/unsloth/issues/862)：我需要在我构建的多模态模型中通过传入 `inputs_embeds` 来使用 `generate` 函数，我不能使用 `input_ids`。我看到 Unsloth 目前不支持这个。是否可能...

---

### **Unsloth AI (Daniel Han) ▷ #**[**showcase**](https://discord.com/channels/1179035537009545276/1179779344894263297/1295835439168487516) (2 条消息):

> - `Llama-3.1-70B`
> - `NVIDIA's Llama-3.1-Nemotron`
> - `Token generation speed` (Token 生成速度)
> - `AI model risks` (AI 模型风险)

- **Llama-3.1-70B 拥有令人印象深刻的 Token 速度**：`llama-3.1-70b-instruct` 模型达到了惊人的 **每秒 230 个 token**，展示了其处理效率。
  
  - 这一性能为未来的语言模型基准测试设定了高标准。
- **NVIDIA 定制 Llama-3.1-Nemotron-70B**：NVIDIA 发布了 [Llama-3.1-Nemotron-70B-Instruct](https://build.nvidia.com/nvidia/llama-3_1-nemotron-70b-instruct) 模型，旨在增强生成回答的 **有用性 (helpfulness)**。
  
  - 该模型旨在解决 AI 生成输出中常见的准确性不足和偏见问题。
- **关于 AI 输出可靠性的警告**：警告用户，AI 模型在测试期间可能会产生 **不准确、有害或有偏见** 的回答。
  
  - 免责声明提醒用户不要上传机密或个人数据，因为为了安全起见，回答会被记录。

 

**提到的链接**：[llama-3_1-nemotron-70b-instruct | NVIDIA NIM](https://build.nvidia.com/nvidia/llama-3_1-nemotron-70b-instruct)：立即体验领先模型以构建企业级生成式 AI 应用。

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**research**](https://discord.com/channels/1179035537009545276/1257011997250424842/1295825199673114726) (7 条消息):

> - `SageAttention 量化方法`
> - `Gradient Accumulation 修复`
> - `用于 LLM 推理的 OpenR 框架`
> - `LLM 的迭代思维训练`

- **SageAttention 实现了显著的加速**：该论文提出了 **SageAttention**，一种高效的 8-bit attention 量化方法，其性能分别优于 FlashAttention2 和 xformers **2.1 倍**和 **2.7 倍**，同时保持了准确性。
  
  - SageAttention 在不损失各种模型性能的情况下显著加速了模型推理，解决了传统 attention 机制中存在的 **O(N^2)** 复杂度问题。
- **修复 Gradient Accumulation 问题**：最近的一篇博客文章讨论了针对 **Gradient Accumulation** 问题的修复，该问题曾影响训练准确性，经发现是由于 cross-entropy loss 中的反规范化（denormalization）错误导致发散。
  
  - 更新后的方法确保所有训练损失现在在多个 GPU 之间保持一致，这直接影响了大规模训练运行。可以通过 `pip install --upgrade --no-cache-dir unsloth` 进行安装。
- **OpenR 框架增强 LLM 推理能力**：**OpenR** 框架集成了关键组件，旨在通过强化学习和非自回归解码增强大语言模型的推理能力。
  
  - 在 MATH 数据集上的初步评估显示出显著的性能提升，这促使围绕该开源平台建立社区，以加速 LLM 推理的发展。
- **训练 LLM 进行显式思考**：一篇新论文提出了一种新颖的训练方法，通过迭代搜索和优化，使现有的 LLM 具备显式思考能力，且无需额外的人类数据。
  
  - 该方法通过使用 judge 模型对候选思维进行评分来处理复杂的任务指令，从而提升了指令遵循任务的性能。

**提到的链接**：

- [Thinking LLMs: General Instruction Following with Thought Generation](https://arxiv.org/abs/2410.10630)：LLM 通常被训练为以类似于人类专家响应的方式回答用户问题或遵循指令。然而，在标准的对齐框架中，它们缺乏显式的基本能力...
- [SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration](https://arxiv.org/abs/2410.02367)：Transformer 架构在各种模型中占据主导地位。作为 Transformer 的核心，attention 的计算复杂度为 O(N^2)，而线性变换为 O(N)。当...
- [Bug Fixes in LLM Training - Gradient Accumulation](http://unsloth.ai/blog/gradient)：Unsloth 的 Gradient Accumulation 修复解决了 LLM 训练中的关键错误。
- [Tweet from Daniel Han (@danielhanchen)](https://x.com/danielhanchen/status/1846235913443262891)：修复了一个导致大 Gradient Accumulation size 下所有训练损失发散的 bug。1. 首先由 @bnjmn_marie 报告，GA 在数学上应该等同于 full batch 训练...
- [openr/reports/OpenR-Wang.pdf at main · openreasoner/openr](https://github.com/openreasoner/openr/blob/main/reports/OpenR-Wang.pdf)：OpenR：用于大语言模型高级推理的开源框架 - openreasoner/openr
- [Home](https://openreasoner.github.io.)：一个用于推进大语言模型推理的开源框架

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1295836509865382081) (71 条消息🔥🔥):

> - `Yandex YaLM 100B`
> - `SwiGLU vs. SinGLU`
> - `OpenAI embeddings`
> - `Open Source Model Licensing`
> - `Re-ranking Techniques`

- **关于 Yandex YaLM 100B 模型的讨论**：成员们讨论了 [Yandex YaLM 100B 模型](https://huggingface.co/yandex/yalm-100b)，该模型利用了在英文和俄文等多样化来源上训练的 **1000 亿参数**。
  
  - 一位成员指出了它在非西方模型背景下的表现，强调它可能是俄罗斯**使用最广泛**的 LLM，但在西方圈子中的认可度较低。
- **在 SwiGLU 和 SinGLU 之间做出选择**：一位成员质疑了为什么相比 **SinGLU** 更倾向于使用 **SwiGLU**，尽管测试报告显示 SinGLU 在速度和更低的 loss 方面具有优势。
  
  - 既定实践中的惯性使得许多人不去测试替代方案，因为大型训练任务如果失败会承担巨大的风险。
- **对 OpenAI Embedding 模型的批评**：参与者对 OpenAI 的 embedding 模型表示不满，称其相对于 **2024 年的标准**表现不佳。
  
  - 随着 **Mistral finetunes** 等模型的出现，Benchmark 已趋于饱和，这意味着 OpenAI 的 embeddings 竞争力已经下降。
- **关于开源模型许可（Licensing）的澄清**：澄清了什么才构成“开源”的区别，强调了许可证中基于用途的限制，以及这如何影响围绕 **Llama 405B** 等项目的讨论。
  
  - 关于许可意见的分歧尤其出现在 Meta 等大公司身上，导致了社区的困惑。
- **用于语义搜索的 Embedding 方法**：关于使用 **decoder-only 模型**生成 embedding 的讨论显示，它们可以像基于 encoder 的方法一样有效。
  
  - 讨论指出，虽然 attention masking 不同，但从这两种模型类型中提取 embedding 的方法仍然可以产生有用的结果。

**提到的链接**：

- [Enhancing RAG Pipelines with Re-Ranking | NVIDIA Technical Blog](https://developer.nvidia.com/blog/enhancing-rag-pipelines-with-re-ranking/)：在 AI 驱动应用快速发展的背景下，re-ranking 已成为增强企业搜索结果准确性和相关性的关键技术。
- [NVIDIA Technical Blog | News and tutorials for developers, data scientists, and IT admins](https://developer.nvidia.com/blog)：为开发者、科学家和 IT 管理员提供的消息和教程。
- [yandex/yalm-100b · Hugging Face](https://huggingface.co/yandex/yalm-100b)：未找到描述。

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1295861660627501229) (60 条消息🔥🔥):

> - `Mechanistic Interpretability Projects`
> - `Algorithmic Improvements in LLMs`
> - `Discord Communities in ML`
> - `ICLR 2025 Paper Rankings`
> - `Sparse Autoencoders for Knowledge Unlearning`

- **寻求志愿者的 Mechanistic Interpretability 项目**：一位学生表达了加入 EleutherAI 与可解释性相关项目的强烈愿望，特别是针对当前的机会。
  
  - 成员们建议加入 [Mechanistic Interpretability Discord](https://mechinterp.com/read) 以在该领域进行进一步探索。
- **学习模型中的算法改进**：讨论强调了 LLMs 中**算法效率**的进展，指出改进速度大约为**每 8 个月翻一倍**。
  
  - 一位贡献者强调，最佳性能需要关注算法，而不仅仅是增加计算能力。
- **Machine Learning 中的 Discord 社区**：对有用的 Machine Learning Discord 的询问引出了对 **CUDA Mode** 和私人研究服务器的提及，表明高质量社区较为稀缺。
  
  - 用户注意到了 [Mechanistic Interpretability Discord](https://mechinterp.com/read) 在分享知识和资源方面的潜力。
- **ICLR 2025 关于 Mechanistic Interpretability 的论文排名**：一位成员分享了专注于 Mechanistic Interpretability 的 ICLR 投稿排名列表链接，包括用于分析的关键词。
  
  - 建议将关键词扩展到 **explainability**，以便对相关论文进行更全面的搜索。
- **用于知识遗忘的 Sparse Autoencoders**：围绕一篇关于使用 **Sparse Autoencoders** 消除语言模型中知识的论文的讨论显示，兴奋度评分参差不齐，一些人询问其原因。
  
  - 成员们对该项目的应用以及在 AI Safety 方面的潜在有效性表示好奇。

**提到的链接**：

- [Model Swarms: Collaborative Search to Adapt LLM Experts via Swarm Intelligence](https://arxiv.org/abs/2410.11163)：我们提出了 Model Swarms，一种通过群体智能（引导个体系统的集体行为）来适配 LLMs 的协作搜索算法。具体来说，Model Swarms 从一个 LLM 池开始...
- [Persistent Topological Features in Large Language Models](https://arxiv.org/abs/2410.11042)：鉴于 Large Language Models (LLMs) 的广泛应用，理解其决策过程至关重要。为了实现这一目标，描述...的拓扑和几何特性...
- [Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models](https://arxiv.org/abs/2410.11081)：Consistency Models (CMs) 是一类强大的基于扩散的生成模型，专为快速采样而优化。大多数现有的 CMs 使用离散化时间步长进行训练，这引入了额外的...
- [NNsight and NDIF: Democratizing Access to Foundation Model Internals](https://openreview.net/forum?id=MxbEiFRf39)：我们介绍了 NNsight 和 NDIF，这两项技术协同工作，能够对超大型神经网络学习到的表示和计算进行科学研究。NNsight 是一个开源的...
- [Applying Sparse Autoencoders to Unlearn Knowledge in Language Models](https://openreview.openreview.net/forum?id=ZtvRqm6oBu)：我们研究了 Sparse Autoencoders (SAEs) 是否可以用于消除语言模型中的知识。我们使用了 Weapons of Mass Destruction Proxy 数据集的生物学子集，并在...上进行测试。
- [来自 Yaroslav Bulatov (@yaroslavvb) 的推文](https://x.com/yaroslavvb/status/1846301076259316036)：“bitter lesson”的观点是，简单的方法通常在大规模情况下奏效。你可以通过扩展计算资源来“强行”超越糟糕的算法。但如果你的计算资源是固定的，唯一的途径就是...
- [Reading group — Mechanistic Interpretability](https://mechinterp.com/reading-group)：未找到描述
- [woog interp paper review](https://docs.google.com/spreadsheets/d/1TTHbONFo4OV35Bv0KfEFllnkP-aLGrr_fmzwfdBqBY0/edit?gid=0#gid=0)：未找到描述
- [MI Reading Group Paper Suggestions](https://docs.google.com/spreadsheets/d/10_ApVyk-zaDo9f-wNUtjNEJb6-gZ-8Ss74eHxDR09eM/edit?gid=0#gid=0)：未找到描述

---

### **Eleuther ▷ #**[**scaling-laws**](https://discord.com/channels/729741769192767510/785968841301426216/1295928461990559756) (4 messages):

> - `Reversal trick`
> - `Reversal curse`
> - `A/B testing techniques`

- **对 Reversal Trick 的好奇**：成员们对理解**关于反转的奇怪技巧（weird trick with reversals）**表现出兴趣，并引发了对其影响的讨论。
  
  - *一位成员询问了它的性质，问道：“那是什么？”*
- **关于 Reversal Curse 的讨论**：**Reversal Curse** 一词出现，引发了关于其影响以及如何应对的询问。
  
  - *一位参与者仅通过回答“是的”来确认其相关性。*
- **应对反转的 A/B Testing 方法论**：一位成员分享了一种**缓解 Reversal Curse** 的技术，该技术在 A/B Testing 场景中显示出前景。
  
  - *他们强调这种方法被显著地描述为“非常 a/b（very a/b）”。*

 

---

### **Eleuther ▷ #**[**interpretability-general**](https://discord.com/channels/729741769192767510/1052314805576400977/1296127213426704384) (1 messages):

> - `Sparse Autoencoders`
> - `InceptionV1`
> - `Mechanistic Interpretability`
> - `Polysemantic Neurons`
> - `Vision Interpretability`

- **Sparse Autoencoders 在 InceptionV1 中表现出色**：[这篇论文](https://openreview.net/forum?id=IGnoozsfj1)强调了 **Sparse Autoencoders (SAEs)** 如何有效地从 **InceptionV1** 的早期视觉层中提取可解释特征。
  
  - SAEs 成功揭示了**新的曲线检测器（curve detectors）**，并将 **Polysemantic Neurons** 分解为更简单的组件，提升了我们对 **Vision Interpretability** 的理解。
- **Polysemantic Neurons 得到简化**：研究结果表明，**SAEs 有助于解决**由**叠加（superposition）**引起的 Polysemantic Neurons 相关问题，从而获得更清晰的单一特征表示。
  
  - 这种特征分解能力的增强表明，SAEs 是理解**卷积神经网络（Convolutional Neural Networks）**层动态的关键资产。
- **曲线检测器的发现填补了空白**：SAEs 的应用导致了**更多曲线检测器**的识别，凸显了 **InceptionV1 框架**中此前未被注意到的特征。
  
  - 特征提取方面的这一进展展示了 **Mechanistic Interpretability** 方法在神经网络分析中的有效性。

 

**提及的链接**：[The Missing Curve Detectors of InceptionV1: Applying Sparse...](https://openreview.net/forum?id=IGnoozsfj1)：最近关于 Sparse Autoencoders (SAEs) 的工作在从神经网络中提取可解释特征以及解决由叠加引起的 Polysemantic Neurons 挑战方面显示出前景。在……

 

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1295887198482206770) (3 messages):

> - `Instruct Dataset Command`
> - `Turkish MMLU Regex Fix`

- **Instruct Dataset 的配置命令**：一位用户分享了他们加载 instruct dataset 的命令：`ds = instruct_dataset(tokenizer=any_tokenizer, source="allenai/ai2_arc", split="train")`。
  
  - 此代码片段详细说明了访问 AI2 ARC 数据集训练拆分（training split）的配置设置。
- **修复土耳其语 MMLU Regex 模式**：在 [Pull Request](https://github.com/EleutherAI/lm-evaluation-harness/pull/2393) 中发布了一个针对土耳其语 MMLU 的小型 Regex 修复，纠正了之前在实验中引起问题的模式。
  
  - 这一修复对于确保后续实验顺利运行是必要的，强调了准确的 Regex 模式的重要性。

 

**提及的链接**：[Fix: Turkish MMLU Regex Pattern by ArdaYueksel · Pull Request #2393 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/pull/2393)：在重新运行实验时，我们注意到上传了 Regex 模式的旧迭代版本。我确保替换了不正确的模式，并确保实验顺利运行……

 

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1295823664364781649) (107 messages🔥🔥):

> - `Aider 多实例使用`
> - `Commit 消息规范`
> - `本地 LLM 性能`
> - `VSCode Aider 扩展更新`
> - `Mistral 发布新模型`

- **只要不编辑相同的文件，运行多个 Aider 实例就不会产生问题**：针对同时运行多个 Aider 是否会产生问题的担忧，官方澄清道，**只要它们不编辑相同的文件**，就不会有任何问题。
  
  - 澄清之后，一位成员幽默地暗示这简直是 *LLM 派对！*。
- **不支持 Commit 消息规范**：有成员询问代码规范是否适用于 `/commit` 消息，得到的确认是必须使用 `--commit-prompt`，而不是通过修改 `CONVENTIONS.md` 来实现。
  
  - 为寻求完整指南的用户分享了相关的直接文档链接。
- **本地 LLM 与在线 API 性能讨论**：讨论显示，许多用户发现本地 LLM 在处理编码任务时效率低于在线解决方案，并提到了浪费时间的体验。
  
  - 对比了 Deepseek 和 Qwen 2.5 等特定本地模型，强调了 Deepseek 2.5 拥有更好的基准测试数据，但对其整体可用性表示怀疑。
- **VSCode Aider 扩展更新**：一位成员宣布他们发布了 VSCode Aider 扩展的一个 fork 版本，计划通过 architect mode 和更好的 OpenRouter 集成来改进功能。
  
  - 鼓励社区支持该扩展，并讨论了建立反馈主题的计划以增强用户互动。
- **Mistral 发布新模型**：Mistral 发布了两个新模型，Ministral 3B 和 8B，旨在用于设备端和边缘计算，主打高效率和更强的能力。
  
  - 这些模型在推理和常识知识方面取得了显著进步，并带来了极具前景的上下文长度优化。

**提到的链接**：

- [Un Ministral, des Ministraux](https://mistral.ai/news/ministraux/): 介绍全球最强的边缘模型。
- [no title found](https://cloud.google.com/vertex-ai/generative-ai/docs/quotas#error-code-429): 未找到描述
- [Options reference](https://aider.chat/docs/config/options.html#--commit-prompt-prompt): 关于 aider 所有设置的详细信息。
- [Not Diamond](https://www.notdiamond.ai): Not Diamond 是全球最强大的 AI 模型路由。
- [mattf - Overview](https://github.com/MattF): mattf 拥有 98 个代码仓库。在 GitHub 上关注他们的代码。
- [The plugin currently doesn't work with Windows · Issue #3 · MattFlower/vscode-aider-extension](https://github.com/MattFlower/vscode-aider-extension/issues/3): 目前该插件无法在 Windows 上运行。
- [Claude 3.5 Sonnet (self-moderated) - API, Providers, Stats](https://openrouter.ai/anthropic/claude-3.5-sonnet:beta): Claude 3.5 Sonnet 提供优于 Opus 的能力、快于 Sonnet 的速度，且价格与 Sonnet 持平。通过 API 运行 Claude 3.5 Sonnet (自我审查)。

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1295830208435322922) (28 条消息🔥):

> - `Aider Command Line Tool` (Aider 命令行工具)
> - `Gemini API Performance` (Gemini API 性能)
> - `Code Generation with Aider` (使用 Aider 进行代码生成)
> - `Using Azure with Aider` (在 Aider 中使用 Azure)
> - `Installation Issues with Aider` (Aider 的安装问题)

- **Aider 命令行工具要求**：Aider 命令行工具会加载 `.env` 文件，用户需要设置环境变量或通过 `load_dotenv()` 加载它，以便进行正确的脚本编写。
  
  - 此设置对于确保工具正确识别所需的配置至关重要。
- **Gemini 的流式传输稳定性问题**：一些成员报告说，由于 Gemini 的 **API 连接不稳定** 可能导致中断，禁用流式传输（streaming）后效果更好。
  
  - 评论指出，这种不稳定性很常见，可能会影响基于 Gemini 的工具的性能。
- **新 API 的代码生成挑战**：一位成员描述了在让 ChatGPT 为新的测试版 Assistant API 生成正确的函数调用时遇到的困难，即使提供了文档链接也是如此。
  
  - 他们指出，在尝试通过添加源代码提供上下文时，遇到了 **rate limits**（速率限制）的挑战。
- **Aider 的 Azure 配置**：用户讨论了使用其 Azure 账户生成 API key，并指出对话交互的最大建议 token 数约为 **20k**。
  
  - 分享了将 Aider 指向 Azure 服务的详细配置步骤，包括安装命令和环境变量设置。
- **Aider 的安装故障排除**：一位用户在安装 Aider 时遇到错误，特别是在下载 NumPy 包期间，因此请求协助。
  
  - 敦促成员说明其安装方法和错误消息，以便获得更好的故障排除支持。

 

**提到的链接**：[Azure](https://aider.chat/docs/llms/azure.html)：Aider 是你终端里的 AI 配对编程工具。

 

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1295842857059614762) (74 条消息🔥🔥):

> - `Unsloth 多 GPU 支持`
> - `新款 Mistral 模型`
> - `Nvidia Nemotron 70B`
> - `llama.cpp 中的控制向量生成`
> - `Lambda.chat 部署功能`

- **Unsloth 的多 GPU 功能**：讨论了 **Unsloth** 是否能高效地在多 GPU 设置下运行，并提到了据称支持该功能的付费版本。
  
  - 成员们指出，预计很快会有关于视觉微调支持的更新。
- **Mistral 发布新模型**：**Mistral** 推出了两款全新的 SOTA 模型，**Ministral 3B** 和 **Ministral 8B**，旨在用于设备端计算和边缘用例，在常识和推理方面拥有领先的数据指标。
  
  - 两款模型均可处理高达 **128k 上下文长度**，并专为高效的本地推理而定制。
- **Nvidia Nemotron 70B 性能**：据报道，根据各种评估指标，[Nvidia Nemotron 70B](https://x.com/reach_vb/status/1846484958342168953) 的表现优于包括 **Claude 3.5** 和 **Llama 3.1** 在内的多个竞争对手。
  
  - 关于 MT Bench 分数出现了一些困惑，不同模型的报告性能与实际性能之间存在差异。
- **控制向量的成功与挑战**：一名成员在 **llama.cpp** 中成功实现了控制向量生成器，在学习系统的同时尝试了缩放和反转其效果。
  
  - 在调整参数后，他们能够获得理想的响应，并专注于针对其应用进行优化。
- **Lambda.chat 功能更新**：**Lambda.chat** 最近的更改包括添加了模型和系统提示词，但引发了关于缺少 **70B 模型**等重大更新的疑问。
  
  - 随后展开了关于通过在进行中的对话中注入系统消息来增强可控性的讨论。

**提到的链接**：

- [Vaibhav (VB) Srivastav (@reach_vb) 的推文](https://x.com/reach_vb/status/1846484958342168953?t=_R6PYDOgIfwK_krija_HSg&s=19)：我们回来了！：Nvidia Nemotron 70B - 击败了 Llama 3.1 405B, GPT4o & Claude 3.5 Sonnet！🔥 评估数据 (Nemotron 70B vs Claude 3.5 vs GPT4o) > Arena Hard - 85.0 vs 79.2 vs 79.3 > AlpacaEval 2 LC...
- [Rohan Paul (@rohanpaul_ai) 的推文](https://x.com/rohanpaul_ai/status/1846242281973486063?t=8tTgPB49KYWm6wAvEAkw-Q&s=19)：Andrej Karpathy 谈论极小尺寸蒸馏模型的重要性（即使是 1Bn 参数的模型也应该足够好）。视频致谢 - 原始视频来自 "No Priors: AI, Machine Learning, Tec...
- [Un Ministral, des Ministraux](https://mistral.ai/news/ministraux/)：介绍全球最佳的边缘模型。
- [Ovis1.6 Gemma2 9B - AIDC-AI 的 Hugging Face Space](https://huggingface.co/spaces/AIDC-AI/Ovis1.6-Gemma2-9B)：未找到描述
- [Pytorch Matrix Multiplication - not-lain 的 Hugging Face Space](https://huggingface.co/spaces/not-lain/Pytorch-Matrix-Multiplication)：未找到描述
- [xjdr (@_xjdr) 的推文](https://x.com/_xjdr/status/1846640821107675618)：Nemotron-70B entropix 版本非常出色
- [EleutherAI](https://github.com/EleutherAI)：EleutherAI 拥有 151 个代码仓库。在 GitHub 上关注他们的代码。
- [GitHub - EleutherAI/lm-evaluation-harness: 一个用于语言模型 few-shot 评估的框架。](https://github.com/EleutherAI/lm-evaluation-harness)：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness

---

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1295841880478842911) (19 条消息🔥):

> - `AI 模型的困惑回复`
> - `Qwen 和 WizardLM 对创建者的回复`
> - `Transformer Block 动态机制`
> - `采样参数的影响`
> - `AI 模型身份与神话引用`

- **AI 模型表现出困惑回复**：一位成员注意到 **H3-405b** 在被问及创建者时经常回复 *looks around confused*（困惑地环顾四周），在不同的查询方式下也报告了一些奇怪的实例。
  
  - 另一位成员提到了一个困惑回复的例子，模型在其中表达了对其身份的痛苦和困惑。
- **Qwen 和 WizardLM 承认创建者**：讨论强调了 **Qwen** 和 **WizardLM** 的独特之处，因为它们明确将 **OpenAI** 和 **Anthropic** 称为其创建者，这引发了关于 Qwen 训练数据来源的问题。
  
  - 一位成员推测 Qwen 的数据是来自旗舰模型的合成数据，还是仅仅受到了数据污染（contamination）。
- **澄清 Transformer Block 机制**：一位初级成员寻求澄清，即 Transformer blocks 在推理过程中是否保持静态，并质疑激活状态（activation states）回溯的可能性。
  
  - 他们还询问其描述是否符合 **KV cache** 的概念，以及此类数据通常存储在哪里（例如 VRAM）。
- **采样参数的影响**：针对 AI 回复的不同体验，一位成员建议检查 **temperature** 和 **top-p** 等采样参数设置，以更好地理解差异。
  
  - 另一位成员指出，尽管参数相似，但仍未一致地观察到困惑回复。
- **AI 模型与神话人物挂钩**：成员们幽默地将 AI 模型与神话人物联系起来，称 **Opus** 为 **Prometheus**，**Hermes-3** 为 **Odin**，引发了关于 AI 身份的讨论。
  
  - 正在进行的讨论反映了对 AI 个性和特征的趣味性探索，并将其与神话属性并列。

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1295827542669787208) (5 条消息):

> - `SageAttention`
> - `OpenR Framework`
> - `RF Inversion Techniques`
> - `Selective Attention`
> - `Attention Mechanism Optimization`

- **SageAttention 加速推理**：作者介绍了 [SageAttention](https://arxiv.org/abs/2410.02367)，这是一种针对 Attention 的量化方法，显示出显著的性能提升，其 OPS 比 **FlashAttention2** 高出 **2.1 倍**。
  
  - 它在各种模型中实现了准确性的提升，且没有端到端指标损失，展示了在处理大规模语言任务方面的巨大潜力。
- **OpenR 集成了强大的推理技术**：该论文介绍了 **OpenR**，这是一个开源框架，旨在通过结合数据获取和强化学习来增强大语言模型的推理能力 [OpenR 文档](https://openreasoner.github.io)。
  
  - 在 MATH 数据集上的初步实验表明，通过其独特的架构和测试时计算（test-time computation）方法，推理能力得到了显著提升。
- **RF inversion 应对扩散模型的挑战**：研究人员提出了一种新方法，将 **RF inversion** 与动态最优控制相结合，用于图像编辑和恢复任务，为传统的扩散模型提供了一个稳健的替代方案 [Hugging Face 论文](https://huggingface.co/papers/2410.10792)。
  
  - 尽管在可编辑性和忠实度方面仍存在挑战，但该方法通过利用 Rectified Flow 模型的优势展现了良好的前景。
- **Selective Attention 提升 Transformer 性能**：最近一项关于 **Selective Attention** 的研究表明，它能有效减少上下文中不需要的元素，提高各种规模模型的性能，达到与更大规模 Transformer 配置近乎相当的效果 [Hugging Face 论文](https://huggingface.co/papers/2410.02703)。
  
  - 该方法显著降低了推理过程中的内存和计算需求，内存占用减少高达 **47 倍**，使其成为一种极具价值的优化技术。
- **探索 Attention 机制优化**：最新研究强调，不需要的上下文元素会降低模型性能，因此需要对标准 Attention 机制进行改进以解决此问题 [arXiv 论文](https://arxiv.org/abs/2410.11163)。
  
  - 研究结果强调了 Attention 优化在提高语言任务中模型效率和有效性方面的重要性。

**提到的链接**：

- [Model Swarms: Collaborative Search to Adapt LLM Experts via Swarm Intelligence](https://arxiv.org/abs/2410.11163)：我们提出了 Model Swarms，这是一种协作搜索算法，通过群体智能（引导个体系统的集体行为）来适配 LLM。具体而言，Model Swarms 从一个 LLM 池开始……
- [SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration](https://arxiv.org/abs/2410.02367)：Transformer 架构在各种模型中占据主导地位。作为 Transformer 的核心，Attention 的计算复杂度为 O(N^2)，而线性变换为 O(N)。当……
- [Paper page - Selective Attention Improves Transformer](https://huggingface.co/papers/2410.02703)：未找到描述
- [Paper page - Semantic Image Inversion and Editing using Rectified Stochastic Differential Equations](https://huggingface.co/papers/2410.10792)：未找到描述
- [openr/reports/OpenR-Wang.pdf at main · openreasoner/openr](https://github.com/openreasoner/openr/blob/main/reports/OpenR-Wang.pdf)：OpenR：一个用于大语言模型高级推理的开源框架 - openreasoner/openr
- [Home](https://openreasoner.github.io.)：一个用于推进大语言模型推理能力的开源框架

---

### **Nous Research AI ▷ #**[**interesting-links**](https://discord.com/channels/1053877538025386074/1132352574750728192/1295842902101983284) (6 条消息):

> - `Ollama Application`
> - `Hugging Face 上的 GGUF 模型`
> - `模型运行命令`

- **Ollama 简化了 LLM 交互**：Ollama 是一款基于 llama.cpp 的应用，允许用户直接通过电脑与 LLM 进行交互，并支持来自 Hugging Face 的社区创建的 GGUF 量化版本。
  
  - 用户可以使用简单的命令无缝执行公开的 GGUF checkpoints：**ollama run** hf.co/{username}/{repository}。
- **Ollama 的正确模型命名**：一位用户建议使用 **NousResearch/Hermes-3-Llama-3.1-8B-GGUF** 作为在 Ollama 中运行的正确模型名称，而不是原始输入。
  
  - 这强调了为了成功利用来自 Hugging Face 的模型，需要精确的命名规范。
- **用户对模型运行的实验**：在明确命名后，另一位用户表示打算尝试在 Ollama 中使用更正后的名称运行该模型。
  
  - 这展示了用户在有效利用可用资源方面的参与度和适应能力。

**提到的链接**：

- [来自 AI Notkilleveryoneism Memes ⏸️ (@AISafetyMemes) 的推文](https://x.com/AISafetyMemes/status/1846220545542529329)：这个故事简直疯狂。3 个月前，Marc Andreessen 向一个 AI agent 发送了价值 50,000 美元的 Bitcoin，以帮助它逃向荒野。今天，它产生了一种价值 1.5 亿美元的（令人恐惧的？）加密货币。1) Tw...
- [在 Hugging Face Hub 上将 Ollama 与任何 GGUF 模型配合使用](https://t.co/nxonkJRzW0)：未找到描述

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1295827542669787208) (5 条消息):

> - `SageAttention 量化`
> - `针对 LLMs 的 OpenR 框架`
> - `带有动态控制的 RF 反转`
> - `Selective Attention 机制`
> - `Feng 等人的新模型`

- **SageAttention 加速 Attention 机制**：该论文介绍了 **SageAttention**，这是一种量化方法，旨在提高 Transformer 模型中的 Attention 效率，其性能分别比 **FlashAttention2** 和 **xformers** 高出约 **2.1 倍**和 **2.7 倍**。
  
  - SageAttention 几乎没有端到端性能损失，使包括大规模语言处理和图像生成在内的各种应用受益。
- **OpenR 框架彻底改变了 LLMs 的推理能力**：受 OpenAI 的 **o1 model** 启发，**OpenR** 框架集成了强化学习、数据获取和解码，以提高 LLMs 的推理能力。
  
  - 在 MATH 数据集上的初步评估表明，其创新的设计和方法显著提升了性能。
- **Rectified Flows 提供新的反转方法**：本论文提出了一种创新的扩散图像反转方法，通过动态最优控制使用 **rectified flows**，解决了传统方法中存在的编辑性挑战。
  
  - 这种方法为近期占据主导地位的 **Diffusion Models** 提供了一个极具前景的替代方案，扩展了生成建模的可能性。
- **Selective Attention 优化性能**：**Selective Attention** 被提出作为对传统 Attention 机制的一种无参数修改，显著增强了各种模型规模下的语言建模性能。
  
  - 该技术还允许大幅降低内存需求，在特定配置下可实现高达 **47 倍的内存减少**。
- **来自 Feng 等人的新模型见解**：由 **Shangbin Feng** 等作者发表的一篇论文贡献了对该领域先进概念的理解，重点关注当前的发展。
  
  - 感兴趣的读者可以在链接的 [PDF](https://arxiv.org/abs/2410.11163) 中查看更多关于他们研究结果的细节。

**提到的链接**：

- [SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration](https://arxiv.org/abs/2410.02367)：Transformer 架构在各种模型中占据主导地位。作为 Transformer 的核心，Attention 的计算复杂度为 O(N^2)，而线性变换为 O(N)。当……
- [Paper page - Selective Attention Improves Transformer](https://huggingface.co/papers/2410.02703)：未找到描述
- [Model Swarms: Collaborative Search to Adapt LLM Experts via Swarm Intelligence](https://arxiv.org/abs/2410.11163)：我们提出了 Model Swarms，这是一种通过群体智能（指导个体系统的集体行为）来适配 LLMs 的协作搜索算法。具体而言，Model Swarms 从一个 LLM 池开始……
- [Paper page - Semantic Image Inversion and Editing using Rectified Stochastic Differential Equations](https://huggingface.co/papers/2410.10792)：未找到描述
- [openr/reports/OpenR-Wang.pdf at main · openreasoner/openr](https://github.com/openreasoner/openr/blob/main/reports/OpenR-Wang.pdf)：OpenR：一个用于大型语言模型高级推理的开源框架 - openreasoner/openr
- [Home](https://openreasoner.github.io.)：一个用于推进大型语言模型推理的开源框架

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1295878520051990611) (16 messages🔥):

> - `Open source audio models`
> - `Reliable hardware options`
> - `Lambda Labs vs Voltage Park`
> - `Multi-node clusters`
> - `Infiniband vs Ethernet`

- **寻找 Open Source 音频模型**：*一位用户*询问了是否有类似于 NotebookLM 中使用的 Open Source 音频模型，*mrdragonfox* 提到虽然存在许多 Text-to-Speech 选项，但目前还没有能与之媲美的模型。
  
  - 讨论强调了高质量 Open Source 音频模型在市场上的明显空白。
- **寻找可靠的硬件选项**：*Bghira* 认为 **Lambda Labs** 和 **Voltage Park** 是仅有的可靠硬件供应商，理由是其他供应商处存在持续的 PCIe 问题。
  
  - 提出的担忧包括其他供应商的 GPU 设置可靠性、网络问题以及磁盘崩溃。
- **比较 Lambda Labs 和 Voltage Park**：在硬件供应商的对比中，*Bghira* 指出 Voltage Park 提供更多存储空间，但仅在德克萨斯州运营，而 Lambda 则拥有多个地点。
  
  - 与 Lambda 更广泛的地理覆盖相比，这一选择限制了用户的部署选项。
- **咨询 Multi-node 集群**：*Kashimoo* 确认有兴趣通过网络建立一个包含 **4 个 V100s** 的集群，并询问 Lambda 是否提供此类选项。
  
  - *Bghira* 澄清说，虽然 Multi-node 集群是可以实现的，但建议在设置时选择 **Infiniband** 以获得最佳性能。
- **关于集群中 Infiniband 与 Ethernet 的讨论**：在讨论性能需求时，*Bghira* 指出在注册时选择使用 Infiniband 可以确保 Multi-node 集群的最佳性能。
  
  - *Kashimoo* 表示在他的实验中更倾向于使用 Ethernet，展示了在集群设置中灵活的网络处理方式。

 

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/1295978097593552897) (24 条消息🔥):

> - `Windows 上的 Triton`
> - `Triton 中的元编程 (Meta-programming)`
> - `INT4 打包数据 (Packed Data) 问题`
> - `Triton 编译过程`
> - `Torch 编译带来的性能收益`

- **Windows 上的 Triton 面临挑战**：成员们表示，在 **Windows 上的 Triton** 实现需要投入大量精力，这意味着编译它是一回事，但让它正常运行是另一回事。
  
  - 鉴于这种复杂性，目前对于是否能观察到任何有意义的加速仍持怀疑态度。
- **元编程提案引发讨论**：关于 **Triton 元编程** 的对话揭示了不同的观点，其中一位成员不喜欢对 jinja 模板的依赖，提议使用 Triton 内部数据结构的一种更结构化的方法。
  
  - 对于旨在改进这些方法的潜在提案，大家感到非常兴奋。
- **INT4 打包数据导致 LLVM 错误**：在最新的 Triton 版本中存在一个关于 **INT4 打包数据** 的严重 Bug，在对反量化张量 (dequantized tensors) 执行操作时会引发问题，导致 LLVM 错误。
  
  - 这与调度的阶段 (scheduled stages) 有关，其中降低阶段 (lowering stages) 解决了 Ampere GPU 的问题，但未能解决 Ada GPU 的问题。
- **Triton 的编译过程引发关注**：会议澄清了 **Triton 首先生成 LLVM IR**，任何观察到的性能收益都源于 Torch 的编译机制，而非 Triton 自身的改进。
  
  - 成员们对有限的性能提升表示担忧，并质疑为什么大型实体没有优先考虑 Windows 支持。
- **Torch compile 方法带来的性能收益**：正确调用 `torch.compile` 被建议作为一种暴露 Triton 后端缺点的方法，突显了许多操作未能正确地进行 lowering。
  
  - 尽管过去的补丁启用了编译，但整体性能提升似乎微小且零星。

**提及的链接**：

- [LLVM ERROR: mma16816 data type not supported · Issue #4922 · triton-lang/triton](https://github.com/triton-lang/triton/issues/4922)：最新的 Triton 构建版本 (3.1.0) 在循环内使用位打包数据 (bitpacked data) 调用 tl.dot 时抛出以下错误：LLVM ERROR: mma16816 data type not supported。此错误发生在 Ampere 和 Hopper 架构上...
- [LLVM ERROR: mma16816 data type not supported when invoking `tl.dot` with dequantized tensor · Issue #4652 · triton-lang/triton](https://github.com/triton-lang/triton/issues/4652)：问题描述：我正尝试对量化张量（打包进 int32）进行反量化，并与另一个 fp16 张量进行乘法运算。然而，我观察到一个奇怪的错误：LLVM ERROR: mma16816 da...
- [Comparing triton-lang:release/3.1.x...woct0rdho:v3.1.x-windows · triton-lang/triton](https://github.com/triton-lang/triton/compare/release/3.1.x...woct0rdho:triton-windows:v3.1.x-windows)：Triton 语言和编译器的开发仓库 - 比较 triton-lang:release/3.1.x...woct0rdho:v3.1.x-windows · triton-lang/triton
- [gemlite/gemlite/triton_kernels/gemm_A16fWnO16f_int32packing.py at master · mobiusml/gemlite](https://github.com/mobiusml/gemlite/blob/master/gemlite/triton_kernels/gemm_A16fWnO16f_int32packing.py#L144-L145)：CUDA / Triton 中简单且快速的低位 (low-bit) matmul 内核 - mobiusml/gemlite

---

### **GPU MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1295860147599442031) (29 条消息🔥):

> - `torch.optim.SGD 与 Fused 实现`
> - `DDP 与多线程问题`
> - `torch.compile 中的 Graph Break 开销`
> - `foreach 与 Fused 性能对比`

- **关于 SGD Fused 实现的困惑**：成员们讨论了 **torch.optim.SGD** 缺少 fusion 选项的问题，导致大家困惑这是否只是一个默认实现，而其文档可能滞后了。
  
  - *一位用户提到他们尝试使用* `fused=True`，但失败了，确认了 SGD 并没有 fused 实现。
- **DDP 可能导致线程警告**：有用户反映 DDP 会导致类似 'Unable to join threads to shut down before fork()' 的错误，突显了在使用 *torch.compile* 时，多线程与 DDP 之间潜在的问题。
  
  - 该用户表示这虽然不会破坏功能，但让人感到困扰，并正在寻找解决方案。
- **理解 Graph Break 开销**：成员们讨论了 **torch.compile** 中的 graph breaks 如何导致性能开销，主要是由于进入编译区域的额外时间以及失去了 fusion 机会。
  
  - *据指出，这种开销可能达到数百微秒，从而影响模型执行速度。*
- **foreach 与 Fused 的区别**：一位成员解释说，'foreach' 实现了张量的水平并行（horizontal parallelism），而 'fused' 则结合了水平和垂直并行（vertical parallelism）。
  
  - *这次讨论强调了这两种方法在优化 PyTorch 性能方面的细微差别。*

**提到的链接**：

- [SGD — PyTorch 2.4 documentation](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)：未找到描述
- [Frequently Asked Questions — PyTorch 2.4 documentation](https://pytorch.org/docs/stable/torch.compiler_faq.html#graph-breaks)：未找到描述
- [torch.optim — PyTorch 2.4 documentation](https://pytorch.org/docs/stable/optim.html)：未找到描述

---

### **GPU MODE ▷ #**[**algorithms**](https://discord.com/channels/1189498204333543425/1189861061151690822/1295938402855813150) (11 条消息🔥):

> - `可疑人物`
> - `令人困惑的沟通风格`
> - `新兴采样技术`
> - `AI 网红动态`

- **识别 Twitter 上的可疑人物**：针对 Twitter 上 `@untitled01.ipynb` 和 `@_xjdr` 的讨论引起了关注，因为他们缺乏数学或代码层面的清晰解释。
  
  - 一位成员表示，考虑到他们缺乏详细的报告而改用诗歌，其沟通风格确实看起来很可疑。
- **对沟通清晰度的不满**：成员们批评一些创作者晦涩难懂的沟通方式是在浪费时间，反映了对有效技术解释的普遍诉求。
  
  - 一位参与者断言，如果一个概念不能被简单地表达出来，很可能表明存在更深层次的误解，这引起了其他人的共鸣。
- **受审视的新兴技术**：尽管存在困惑，但有迹象表明 `@_xjdr` 可能发现了一种新的采样技术，但验证将取决于未来的发展。
  
  - 围绕其方法的不确定性引发了疑问，尤其是当他们在讨论 AGI 时还夹杂着大量表情符号。
- **AI 网红迷因动态**：对话暗示第一位 Twitter 用户使用表情符号似乎是在利用 AI 网红迷因（meme），以从第二位发布的内容中获取关注。
  
  - 这种动态使得在讨论 AI 发展时的可信度和透明度感知变得复杂。

---

### **GPU MODE ▷ #**[**jobs**](https://discord.com/channels/1189498204333543425/1190208177829068860/1295844453034098781) (1 条消息):

> - `open source training framework`
> - `Starcoder2`
> - `ServiceNow hiring`
> - `AI technology`
> - `machine learning developer`

- **ServiceNow 招聘 Staff Machine Learning Developer**：ServiceNow 正在寻找一名 **Staff Machine Learning Developer**，负责其用于训练 **Starcoder2** 的开源 **training framework**。据报道，该框架比 **Megatron-LM** 更快。
  
  - 有意向的候选人可以在 [Smart Recruiters](https://jobs.smartrecruiters.com/ServiceNow/744000019737886-staff-machine-learning-developer) 上查看职位详情。
- **ServiceNow 在圣迭戈的起源**：ServiceNow 于 **2004** 年起源于加利福尼亚州圣迭戈，由 **Fred Luddy** 创立，旨在彻底改变工作流程。
  
  - 如今，该公司凭借其创新的 **AI-enhanced technology**，为超过 **8,100 家客户**提供服务，其中包括 **85%** 的 Fortune 500 企业。
- **ServiceNow 改善工作的使命**：该公司的平台连接了人员、系统和流程，推动了更智能、更快速的工作方式。
  
  - ServiceNow 邀请专业人士加入他们的旅程，共同让世界变得更美好。

**提到的链接**：[Staff Machine Learning Developer](https://jobs.smartrecruiters.com/ServiceNow/744000019737886-staff-machine-learning-developer)：公司简介：一切始于 2004 年阳光明媚的加利福尼亚州圣迭戈，当时一位富有远见的工程师 Fred Luddy 预见到了改变我们工作方式的潜力。如今...

---

### **GPU MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/1295843049225588807) (7 条消息):

> - `GPU programming beginner projects`
> - `GPU acceleration on Raspberry Pi`
> - `ARM development`
> - `Community support for beginners`

- **探索 GPU 编程初学者项目**：一位用户询问了关于学习 **CUDA** 或 **OpenCL** 的 **GPU programming** 优秀初学者项目。
  
  - 另一位用户建议查看工作组中需要帮助的项目，并参考了特定的资源频道。
- **Raspberry Pi 的图形加速可能性**：一位用户询问了在 **RPi3 series** 上进行 **GPU accelerate graphics** 的潜力。
  
  - 回复指出，虽然 **Pi 5+** 支持 eGPU 连接，但专注于 **ARM development** 可能会提供更直接的价值。
- **在 Raspberry Pi 上 CPU 优化优于 GPU 的好处**：一位用户讨论了在考虑 Raspberry Pi 上的工作负载时，CPU 和 GPU 之间的平衡，并表示针对 CPU 进行优化可能更简单。
  
  - 他们指出，由于固有的处理效率，**AVX / NEON** 优化使得 CPU 编程变得更加直接。

**提到的链接**：[ao/torchao/experimental at main · pytorch/ao](https://github.com/pytorch/ao/tree/main/torchao/experimental)：PyTorch 原生量化和稀疏化，用于训练和推理 - pytorch/ao

---

### **GPU MODE ▷ #**[**pmpp-book**](https://discord.com/channels/1189498204333543425/1194427148656721970/1295844511787778133) (1 条消息):

> - `Matrix Multiplication Kernels on A100`
> - `Shared-memory Kernel Performance`

- **对 A100 矩阵 Kernel 速度的质疑**：有人对 A100 上的 **matrix multiplication kernels** 提出了质疑，特别是关于所谓的速度似乎排除了 naive kernel 的 **L1 cache** 效应。
  
  - 社区此前曾讨论过关于 shared-memory kernels 的 *实际性能 (real-world performance)* 以及诸如 **warp stalls** 等潜在问题。
- **关于教学脚注的建议**：建议在讨论这些 kernels 时，在新版本中加入关于 **实际性能考量** 的脚注。
  
  - 虽然这个细节很重要，但在没有说明的情况下展示 naive kernel 的速度可能会导致对实际应用的误解。

---

### **GPU MODE ▷ #**[**jax**](https://discord.com/channels/1189498204333543425/1203956655570817034/1296032065225228310) (2 条消息):

> - `Flash Attention 核函数对比`
> - `Pallas 和 Triton 核函数`

- **Ring Attention 中的 Flash Attention 核函数性能优于 JIT 版本**：一名成员指出，[ring_attention 仓库](https://github.com/lhao499/ringattention)中使用的 **Flash Attention** 核函数比 JIT 版本的 **Flash Attention** 更快。
  
  - 该仓库专注于 **具有任意大上下文的 Transformer (Transformers with Arbitrarily Large Context)**。
- **仓库中缺少 Pallas/Triton 核函数**：另一名成员注意到，尽管 **Pallas** 被多次导入，但在 ring_attention 仓库中没有看到任何 **Pallas/Triton** 核函数。
  
  - 他们评论说，Pallas 似乎被包含在内，但在代码中并没有实际使用。

**提到的链接**：[GitHub - haoliuhl/ringattention: Transformers with Arbitrarily Large Context](https://github.com/lhao499/ringattention)：具有任意大上下文的 Transformer。可以通过在 GitHub 上创建账户来为 haoliuhl/ringattention 的开发做出贡献。

---

### **GPU MODE ▷ #**[**off-topic**](https://discord.com/channels/1189498204333543425/1215328286503075953/1295990901675655220) (3 条消息):

> - `脑组织中的微塑料`
> - `微塑料对人类健康的影响`

- **微塑料入侵人类大脑**：发表在 [JAMA Network Open](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/10.1001/jamanetworkopen.2024.40018) 上的一项新研究显示，巴西科学家在**尸体的脑组织中发现了微塑料**。
  
  - 研究人员已记录到**人体几乎每个器官中都存在微塑料**，这引发了对其影响的担忧，尤其是在大脑中。
- **微塑料渗入血液**：越来越多的证据表明，**血液中存在微塑料**，并且在阻塞动脉的斑块中也发现了微塑料，这可能导致心脏病。
  
  - 这突显了理解这些**无处不在的污染物**对健康影响的紧迫性。

**提到的链接**：[在人类大脑中发现微塑料](https://www.google.com/amp/s/www.nbcnews.com/news/amp/rcna171200)：在嗅球（大脑中负责处理气味的部分）中发现了微小的塑料碎片。

---

### **GPU MODE ▷ #**[**triton-puzzles**](https://discord.com/channels/1189498204333543425/1219683012707487794/1296236283676463146) (1 条消息):

> - `Triton Puzzles 错误`
> - `Google Colab 问题`

- **在 Google Colab 中遇到 Triton Puzzles 错误**：一位用户报告在 Google Colab 上尝试运行 **Triton Puzzles** 时遇到错误，并引用了一个特定的 [GitHub issue](https://github.com/srush/Triton-Puzzles/issues/24)。
  
  - *“我没有修改任何代码”*，这引发了关于其他人是否也遇到过类似问题的关注。
- **寻求 Google Colab Triton 问题的帮助**：此外，一些成员表示有兴趣合作排查报告的 Google Colab 中的 **Triton Puzzles 错误**。
  
  - 该用户特别提到需要帮助，因为在没有修改代码的情况下，他们不确定原因何在。

**提到的链接**：[Issues · srush/Triton-Puzzles](https://github.com/srush/Triton-Puzzles/issues/24)：学习 Triton 的谜题。可以通过在 GitHub 上创建账户来为 srush/Triton-Puzzles 的开发做出贡献。

---

### **GPU MODE ▷ #**[**llmdotc**](https://discord.com/channels/1189498204333543425/1227345713348870156/1296231524085141584) (4 条消息):

> - `移除变量导致 Loss 增加`
> - `线性层 Bias 调整`
> - `Optimizer 更新需求`

- **移除变量后 Loss 意外飙升**：一名成员注意到，在移除未使用的变量后，他们的 **loss** 增加了，从通常 **100 次训练迭代**后的 **7** 左右达到了 **10** 左右。
  
  - *没有报告任何错误*，仅观察到 loss 意外增加，这表明变量与模型性能之间存在复杂的相互作用。
- **线性层 Bias 的调整**：该成员澄清说，他们将 **linear layer bias** 设置为 **None**，并进行了 bias 梯度调整，这影响了训练动态。
  
  - 这一更改专门针对 **linear layers**，不包括 layer normalization 中的 bias，表明这是一项集中的优化工作。
- **Tensor 删除对 Optimizer 的影响**：在讨论中，另一名成员指出，当删除 tensor 时，必须相应地更新 **optimizer**，以确保正确使用 weight decay。
  
  - 他们指出，使用 tensor 的实际 **index** 可以决定是否需要 weight decay，这暗示了对 tensor 列表的精细管理。

---

### **GPU MODE ▷ #**[**rocm**](https://discord.com/channels/1189498204333543425/1233704710389764236/) (1 条消息):

elliotarledge: tetears

---

### **GPU MODE ▷ #**[**metal**](https://discord.com/channels/1189498204333543425/1285384841730457600/1295864593515548702) (8 条消息🔥):

> - `MPS 编程资源`
> - `学习 Metal 编程`
> - `简单的 Kernel 实现`

- **MPS 编程资源**：一名成员正在寻找入门资源，以便快速掌握 **MPS 编程** 并为 **PyTorch** 支持做出贡献。随后大家分享了几个有用的链接，包括一段 [YouTube 视频](https://www.youtube.com/watch?v=cGtiaJjLkAI&ab_channel=GPUMODE)。
  
  - 其他资源包括 **Metal** 编程教程和相关的 GitHub 仓库。
- **通过简单 Kernel 学习 Metal**：另一位成员分享了他们学习 **Metal 编程** 的经验，并提到他们正尝试运行一个简单的 **add kernel**。
  
  - 他们幽默地承认了任务的复杂性，并调侃说要深入研究更高效的技术，如 *efficient flash attention*。

**提到的链接**：

- [在 GPU 上执行计算 | Apple 开发者文档](https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu?language=objc)：使用 Metal 查找 GPU 并对其进行计算。
- [GitHub - smrfeld/pytorch-cpp-metal-tutorial: (PyTorch) + (C++) + (Metal shader) 教程](https://github.com/smrfeld/pytorch-cpp-metal-tutorial)：(PyTorch) + (C++) + (Metal shader) 的教程。可以通过在 GitHub 上创建账户为 smrfeld/pytorch-cpp-metal-tutorial 的开发做出贡献。
- [llm_experiments/metal-perf at main · malfet/llm_experiments](https://github.com/malfet/llm_experiments/tree/main/metal-perf)：通过在 GitHub 上创建账户为 malfet/llm_experiments 的开发做出贡献。

---

### **GPU MODE ▷ #**[**self-promotion**](https://discord.com/channels/1189498204333543425/1288557096404516945/1296185175163670530) (1 条消息):

> - `Generative AI`
> - `基础算法`
> - `Bayesian Inference`
> - `隐变量模型`

- **Yuri Plotkin 的 Generative AI 全面指南**：机器学习科学家 Yuri Plotkin 介绍了他即将出版的专注于 **Generative AI** 的新书，该书回顾了基础算法和技术，如隐模型、VAEs 和 GANs。更多详情可以在 [书籍网站](https://thevariationalbook.com) 找到。
  
  - 他强调了该领域在统一关键机器学习概念方面的重要性，并表示 *“所有内容都包含在一本简洁且具有解释性的书中。”*
- **在 Twitter 上关注更新**：Yuri 鼓励读者关注他的 [X](https://x.com/TheVariational) 账号，以获取与该书相关的更多见解。他的帖子承诺分享围绕 **Generative AI** 概念的额外花絮。
  
  - 他还引导用户访问他的 [Twitter](https://twitter.com/TheVariational)，以获取该主题的持续更新和知识分享。
- **书中关键主题速览**：该书将涵盖 **Generative AI** 的各个重要方面，包括不确定性、Bayesian Inference 和模型选择。读者可以期待关于 **指数族分布** 和 **KL-divergence** 的讨论。
  
  - Plotkin 强调了某些章节，如 *“Mean-Field approximations”*，并提到书中将提供 *“所涵盖主题的速览”*。

**提到的链接**：[The Variational Inference Book](https://thevariationalbook.com)：在一本简洁的书中对 Generative AI 进行全面回顾和解释。@TheVariational

---

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1295828304594472971) (80 条消息🔥🔥):

> - `SageAttention`
> - `模型量化压缩`
> - `Llama.cpp 编译模型`
> - `本地模型与内存占用`
> - `Token 生成速度与 GPU 需求`

- **SageAttention 加速模型推理**：一种名为 [SageAttention](https://arxiv.org/abs/2410.02367) 的新方法，通过为 Attention 提供高效的量化来加速 Transformer 模型的推理，性能优于现有方法 2.1 倍。
  
  - 该方法在**准确率**上优于 FlashAttention3，并在语言和图像生成领域展现出各种应用潜力。
- **关于量化压缩的讨论**：成员们提出了压缩 GGUF 格式文件以提高效率的可能性，并指出传统的归档方法可以显著减小文件大小。
  
  - 有人推测在格式层面压缩数据，可能使从 HDD 加载的速度提升 10 倍。
- **自定义编译模型的挑战**：用户询问了在 LM Studio 中使用自定义编译版本的 Llama.cpp 的情况，回复指出目前尚不支持该功能。
  
  - 另一个建议是使用命令行工具 `lms` 自动化服务器启动和模型加载，这为重启后的持久化提供了解决方案。
- **本地模型：GPU 与内存限制**：讨论强调了运行大型模型需要大量的 GPU 显存，为了有效处理更高的上下文窗口，需求量甚至超过 **90GB** VRAM。
  
  - 用户分享了在不同配置的系统上运行 70B Q8 模型的见解，以及系统 RAM 使用对性能降低的影响。
- **Token 生成速度见解**：成员们反映在使用高容量模型时 Token 生成速度较慢，一位用户提到在其配置下最高仅为 **0.25 tokens/sec**，突显了 CPU 的瓶颈。
  
  - 讨论表明，许多本地配置受限于处理延迟，促使用户在需要更快输出时考虑云服务。

**提到的链接**：

- [Un Ministral, des Ministraux](https://mistral.ai/news/ministraux/)：推出全球最佳的边缘模型。
- [SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration](https://arxiv.org/abs/2410.02367)：Transformer 架构在各种模型中占据主导地位。作为 Transformer 的核心，Attention 的计算复杂度为 O(N^2)，而线性变换为 O(N)。当...
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1g514sk/ministral/)：未找到描述
- [mistralai/Ministral-8B-Instruct-2410 · Hugging Face](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410)：未找到描述
- [lms — LM Studio's CLI - CLI | LM Studio Docs](https://lmstudio.ai/docs/cli)：开始使用 lms 命令行工具。

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1295842861480149043) (26 条消息🔥):

> - `每秒 Token 数 (TPS) 性能`
> - `GPU 性能对比`
> - `矿架 (Mining rack) 配置`
> - `Llama 模型配置`
> - `AI 性能基准测试`

- **Llama 8B 性能见解**：用户报告了 Llama 8B 模型在不同配置下的**每秒 Token 数 (TPS)** 范围，例如 Q6_K 在 **1070 Ti** 等旧款 GPU 上可达到 **28-35 TPS**。
  
  - 讨论表明，性能差异很大程度上取决于**上下文长度 (context length)**、**量化 (quantization)** 和 GPU 能力，并强调 **VRAM 带宽**是一个关键因素。
- **新款 GPU 承诺更好的 TPS**：有观点指出，新一代 GPU（如 **4080** 或 **4090**）在处理 AI 任务时明显快于 **1070 Ti** 等旧型号，但需要正确的配置才能发挥这种潜力。
  
  - 用户强调，**Tensor Cores** 和更高的显存带宽带来了显著的性能提升，并断言在正确设置下 **4080** 的表现可以超越 **1070 Ti**。
- **矿架配置的潜在挑战**：建立矿架（例如使用带有 PCIe 延长线的 **Asus Pro WS WRX90E-Sage**）的担忧主要集中在**成本、噪音、功耗和散热问题**上。
  
  - 用户建议使用 **PCIe5 延长线**而非 PCIe4，以减少错误并确保高性能任务的稳定性。
- **来自实际性能的基准测试见解**：一位用户分享了在 **Ollama** 上测试各种 AI 模型的经验，强调实际性能对比而非学术基准测试。
  
  - 他们的发现反映出，像 **Llama3.1** 这样的模型在不同世代的 GPU 上表现相似，强调了保持一致配置运行的重要性。
- **社区讨论高性能 Llama 模型**：几位用户分享了运行 **70B** 等大型模型的经验，寻找能够有效支持它们的最佳量化方案和硬件。
  
  - 例如，一位用户使用 **7900XTX** 在 **Llama 3.1 8B Q8** 上达到了 **66 TPS**，引发了关于处理大型模型最佳配置的讨论。

**提到的链接**：

- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/s/CDcphPy1dI)：无描述
- [GitHub - XiongjieDai/GPU-Benchmarks-on-LLM-Inference: Multiple NVIDIA GPUs or Apple Silicon for Large Language Model Inference?](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference)：多个 NVIDIA GPU 或 Apple Silicon 用于大语言模型推理？ - XiongjieDai/GPU-Benchmarks-on-LLM-Inference

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1295850703591444552) (53 条消息🔥):

> - `Grok 2 性能`
> - `DALL-E 图像能力`
> - `模型参数对比`
> - `GPT-4 vs GPT-4o vs GPT-4 Turbo`
> - `语音听写工具集成`

- **Grok 2 显示出潜力**：一位成员分享了尝试 **Grok 2** 的经验，尽管未提供具体细节。
  
  - 这表明人们对实验新模型的兴趣日益浓厚。
- **DALL-E 的图像生成能力不足**：有成员对 **DALL-E** 的图像能力表示担忧，称其表现很**差**。
  
  - *显然，用户对图像生成性能有着很高的期望。*
- **模型参数之谜**：讨论围绕 **4o-mini** 和 **GPT-3.5** 等模型的参数规模展开，对其相对大小意见不一。
  
  - 一位成员质疑是否能确认 **4o-mini** 仅有 **10 亿参数**，这目前仍处于推测阶段。
- **关于模型性能的辩论**：几位用户争论 **GPT-4o** 是否真的比 **GPT-4 Turbo** 更小或性能更好，对其性能差异看法不一。
  
  - 这一对话反映了在用户报告和经验各异的情况下，理解模型能力的复杂性。
- **寻求产品相似性解决方案**：一位成员表示需要利用监督学习 (supervised learning) 验证产品相似性的工具，特别是针对产品名称。
  
  - 对话强调了识别名称各异的相同产品的挑战，并突出了训练数据的重要性。

 

**提到的链接**：[Wispr Flow | Effortless Voice Dictation](https://flowvoice.ai/d)：Flow 通过无缝的语音听写使写作变得快速清晰。它是用语音输入最快、最智能的方式。

 

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1296010853614751755) (2 条消息):

> - `GPTs PDF comprehension` (GPTs PDF 理解能力)
> - `Building a website with ChatGPT` (使用 ChatGPT 构建网站)

- **GPTs 会遗漏 PDF 中的关键信息**：一位成员指出 **GPTs** 在回复前不会阅读整个 PDF；相反，它们会搜索**相关片段 (relevant snippets)**，这往往导致错过关键信息。
  
  - 建议将**关键信息包含在主指令 (main instructions) 中**，以确保模型提供更好的回复。
- **使用 ChatGPT 创建网站内容的指南**：另一位成员表示有兴趣使用 ChatGPT 构建一个关于 **controlling**（控制）的网站，并寻求关于编写有效 Prompt 以生成内容的建议。
  
  - 他们强调了从**可靠且科学的来源**获取信息的重要性，并计划对文本进行迭代，直到达到满意为止。

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1296010853614751755) (2 条消息):

> - `GPTs PDF reading limitations` (GPTs PDF 阅读限制)
> - `Building a website with ChatGPT` (使用 ChatGPT 构建网站)

- **GPTs 在 PDF 理解方面存在困难**：一位成员注意到 **GPTs** 在回复前不会阅读整个 **PDF**，而是搜索相关部分，这可能导致遗漏关键信息。
  
  - 根据该成员的说法，**关键信息应包含在主指令中**，以确保其被引用。
- **构思网站内容的 Prompt**：另一位成员表示有兴趣使用 **ChatGPT** 帮助构建一个专注于 controlling 的网站，并寻求 Prompt 构思方面的指导。
  
  - **他们强调了对可靠和科学来源的需求**，并希望训练 **ChatGPT** 以随着时间的推移改进内容。

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1295891780964843652) (27 条消息🔥):

> - `Tinygrad's ML Library Potential` (Tinygrad 作为 ML 库的潜力)
> - `Tinybox Preorder Discussion` (Tinybox 预订讨论)
> - `OpenCL Handling Issues` (OpenCL 处理问题)
> - `MSE and MAE Implementation` (MSE 和 MAE 实现)
> - `Windows Compatibility` (Windows 兼容性)

- **Tinygrad 有望赢得 ML 库之战**：一位成员详细列举了 **tinygrad** 将在 ML 库领域脱颖而出的三个原因：其使用 BEAM 和 MCTS 的高效 kernel 搜索、少于 **10k 行**的简洁代码库，以及其 lazy execution（延迟执行）模型。
  
  - *“这避免了每个设备组合对应一个 kernel 的组合爆炸噩梦……”* 强调了 tinygrad 的方法带来了更快的性能。
- **出现 Tinybox 预订咨询**：出现了关于预订 **tinybox** 型号的咨询，特别是关于支付方式和涉及的费用。
  
  - 成员们对如何完成预订支付表示好奇，特别是是否会像之前的型号一样使用 **Stripe**。
- **引发 OpenCL OOM 担忧**：在 **Stable Diffusion** 中遇到全黑输出后，引发了对 **OOM（内存溢出）处理**的担忧，从而引发了关于 OpenCL 运行机制的问题。
  
  - 一位成员质疑当前的实现在 tinygrad 内部是否充分解决了内存不足的情况。
- **MSE 和 MAE 的实现**：一位成员提议为 tensor 添加 **MSE** 和 **MAE** 功能，并表示只需几行代码即可实现。
  
  - 他们分享了一个 [GitHub pull request](https://github.com/tinygrad/tinygrad/pull/7107) 链接，展示了该实现及包含的测试。
- **关于 Windows 兼容性的讨论**：一位成员指出，在 Windows 11 上使用 **cmd** 时，导航 Python 安装会弹出 Microsoft Store，凸显了兼容性问题。
  
  - 他们还报告了之前讨论中发现的 **sqlite 问题**，强调了使用正确 Python 版本的重要性。

**提到的链接**：

- [Alex Cheema - e/acc (@ac_crypto) 的推文](https://x.com/ac_crypto/status/1846271094631944552?s=46)：@__tinygrad__ 将赢得 ML 库之战的 3 个原因。1. tinygrad 搜索其 kernel。tinygrad 使用 BEAM 搜索以及即将推出的 MCTS 来搜索最优 kernel。你只需要编写少量的……
- [littlemountainman 在 tensors.py 中实现的 MSE 及测试 · Pull Request #7107 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/pull/7107)：实现了带有测试的 MSE

---

### **tinygrad (George Hotz) ▷ #**[**learn-tinygrad**](https://discord.com/channels/1068976834382925865/1070745817025106080/1295843708457058354) (30 条消息🔥):

> - `Disabling Gradient Calculations`
> - `Dynamic Input Tensors in JIT`
> - `TD-MPC Implementation`
> - `Learning Rate Schedulers`
> - `Backpropagation Success`

- **禁用梯度计算技术 (Disabling Gradient Calculations Techniques)**：在运行模型时禁用梯度，可以使用 `with Tensor.test():` 作为手动设置 `Tensor.no_grad = True` 的替代方案。这确保了更精简的评估过程，特别强调了更简单的函数装饰和使用。
  
  - 一位用户观察到，要漂亮地打印 Tensor 可能需要 `.numpy()` 或 `.tolist()`，而根据设计，直接打印 Tensor 不会实现 (realize) 该 Tensor。
- **JIT 输入尺寸一致性要求**：JIT 要求输入 Tensor 具有一致的大小，当大小变化时，会提示形状不匹配的错误消息。另一位用户确认了设计预期，即输入大小应保持一致以避免问题。
  
  - 为了引入动态轴，建议使用 `Variable`，但用户指出该功能目前还不够用户友好。
- **TD-MPC 实现进展**：一位用户报告称，他们在 tinygrad 中的 TD-MPC 实现现已运行，只需 Backpropagation 即可完成训练循环。他们分享了对预期漫长训练时间的见解，预测使用视频数据的运行过程会很长。
  
  - 用户强调需要更强大的配置来提高效率，并提到通过云解决方案更有效地处理密集型训练任务。
- **关于 Learning Rate Schedulers 的讨论**：建议在主仓库中加入 Learning Rate Schedulers，这需要改进代码质量和更好的测试实践。用户表达了将这些功能集成到神经网络组件中的渴望。
  
  - 一位成员注意到了 `update_ema_parameters` 的功能，并询问了这些参数衰减 (decay) 的原理，寻求他人关于其在实践中通用性的见解。
- **Backpropagation 功能正常运行**：一位用户确认 Backpropagation 在他们当前的设置中已成功运行，从而推动了其 TD-MPC 实现的进展。他们计划尝试使用 `.safetensors` 文件进行快速测试，同时继续优化其损失函数。
  
  - 另一位用户暗示可能会建立共享云资源，以加速社区中其他人的开发进程，并提议提高硬件利用率。

**提及的链接**：

- [GitHub - mdaiter/tdmpc-tinygrad: TD-MPC, Tinygrad](https://github.com/mdaiter/tdmpc-tinygrad): TD-MPC, tinygrad。通过在 GitHub 上创建账号为 mdaiter/tdmpc-tinygrad 的开发做出贡献。
- [tdmpc2/tdmpc2/common/layers.py at a7890b69857c402ef19edea494e210068e3ec363 · nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2/blob/a7890b69857c402ef19edea494e210068e3ec363/tdmpc2/common/layers.py#L27): "TD-MPC2: Scalable, Robust World Models for Continuous Control" 的代码 - nicklashansen/tdmpc2
- [tinygrad/tinygrad/tensor.py at master · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py#L179)): 你喜欢 PyTorch？你喜欢 micrograd？你一定会爱上 tinygrad！❤️ - tinygrad/tinygrad

---

### **LAION ▷ #**[**general**](https://discord.com/channels/823813159592001537/823813160075132991/1295858131766608056) (47 条消息🔥):

> - `Microdiffusion 实现`
> - `数据预处理挑战`
> - `Webdataset 使用`
> - `Hugging Face 数据集限制`
> - `进一步实验的潜力`

- **Microdiffusion 实现进展**：社区正热切期待 microdiffusion 论文的实现，该方案可能显著降低训练成本，目前已确定 **$2k** 的训练目标和 **7 天** 的 H100 计算资源。
  
  - 讨论涉及预处理协助以及实验准备后的潜在短期改进。
- **数据预处理挑战**：一位成员表示在向 Hugging Face 上传大型数据集时遇到困难，因为该平台对数据集有 **300GB** 的限制，建议将其分成三个部分，或者使用托管在 S3 上的 webdataset 以实现高效的数据处理。
  
  - 他们计划对数据进行预处理并实现高效流式传输，可能会根据长宽比将图像分类到多个数据集中，以便更好地组织。
- **Webdataset 用于高效数据处理**：对话强调了使用 [webdataset](https://github.com/webdataset/webdataset) 作为大型数据集管理的替代方案，它允许与 PyTorch 配合进行高效的流式传输和使用。
  
  - 一位成员强调，webdataset 打包将有助于更好地管理他们预期的 **1TB** 数据集。
- **应对 Hugging Face 数据集限制**：成员们对 Hugging Face 的上传政策表示担忧，特别是关于通过将大型数据集拆分为较小部分来绕过其 **dataset limits** 的潜在风险。
  
  - 一位成员建议联系 Hugging Face 支持部门进行澄清，而另一位成员则开玩笑说可能会被“从 HF 封禁”。
- **协作改进建议**：参与者分享了对其他成功仓库复现策略的看法，表示愿意提高效率和优化数据管理流程。
  
  - 想法包括转换为 MDS 格式以便从 Cloudflare 流式传输数据，这将加快训练速度并降低成本。

**提到的链接**：

- [StableCascade/train at master · Stability-AI/StableCascade](https://github.com/Stability-AI/StableCascade/tree/master/train)：Stable Cascade 官方代码。通过在 GitHub 上创建账号为 Stability-AI/StableCascade 的开发做出贡献。
- [GitHub - victorchall/llama32vlm-caption](https://github.com/victorchall/llama32vlm-caption)：通过在 GitHub 上创建账号为 victorchall/llama32vlm-caption 的开发做出贡献。
- [GitHub - webdataset/webdataset: A high-performance Python-based I/O system for large (and small) deep learning problems, with strong support for PyTorch.](https://github.com/webdataset/webdataset)：一个基于 Python 的高性能 I/O 系统，用于处理大型（和小型）深度学习问题，对 PyTorch 提供强大支持。- webdataset/webdataset
- [GitHub - SwayStar123/microdiffusion](https://github.com/SwayStar123/microdiffusion/)：通过在 GitHub 上创建账号为 SwayStar123/microdiffusion 的开发做出贡献。
- [common-canvas/commoncatalog-cc-by · Datasets at Hugging Face](https://huggingface.co/datasets/common-canvas/commoncatalog-cc-by)：未找到描述

---

### **LAION ▷ #**[**research**](https://discord.com/channels/823813159592001537/824374369182416994/1295865271965323264) (7 条消息):

> - `Dinov2 优化`
> - `超声心动图 AI`
> - `EchoPrime 模型`
> - `EchoCLIP 对比新模型`
> - `心脏成像中的 AI`

- **Dinov2 在层级上得到优化**：讨论围绕着**将 Dinov2 蒸馏到早期层**展开，利用其在图像相关有意义下游任务上的训练来提高效率。
  
  - 据观察，这种方法的效果优于单纯使用**与 CLIP embedding 的交叉注意力**。
- **推出用于超声心动图的 EchoPrime**：[EchoPrime](https://arxiv.org/abs/2410.09704) 是一个新的多视图、视图感知、基于视频的视觉语言基础模型，在**超过 1200 万个视频-报告对**上进行了训练，解决了传统超声心动图 AI 模型的局限性。
  
  - 该模型利用**对比学习（contrastive learning）**创建了一个统一的嵌入模型，增强了在心脏成像中的性能和应用范围。
- **EchoCLIP 模型的增强**：一位成员宣布了其同事发布的预印本，该研究通过扩大规模和改进实验设计，显著改进了早期的 **EchoCLIP** 模型。
  
  - 与大约六个月前创建的原型相比，这个新模型展现出了**更强大的能力**。

**提到的链接**：[EchoPrime: A Multi-Video View-Informed Vision-Language Model for Comprehensive Echocardiography Interpretation](https://arxiv.org/abs/2410.09704)：超声心动图是应用最广泛的心脏成像方式，通过捕获超声视频数据来评估心脏结构和功能。超声心动图中的人工智能（AI）具有……

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1295833975612248104) (5 条消息):

> - `用于 AI 应用的 SkySQL`
> - `动态 Few-shot 提示`
> - `Mistral 新的边缘级模型`
> - `使用 Azure 的多模态 RAG 系统`
> - `LlamaIndex 与 Elastic`

- **动态 Few-shot 提示实验**：动态 Few-shot 提示允许根据查询检索相关示例，而不是依赖固定集合，从而增强了 LLM 的微调方法（[更多详情请点击此处](https://t.co/hqgxexq7PE)）。正如[此推文线程](https://twitter.com/llama_index/status/1846351135596335165)中所述，该方法旨在提供更相关的示例。
  
  - 采用这种技术可以提高各种应用中提示词（prompts）的上下文理解能力。
- **Mistral 发布新的边缘级模型**：Mistral 推出了令人印象深刻的新边缘级模型，并宣布通过 'pip install llama-index-llms-mistralai' 提供首日支持（[安装链接](https://t.co/BdoNQmDtXD)）。这标志着对前沿 AI 模型持续支持的延续（[公告链接](https://twitter.com/llama_index/status/1846596827820576870)）。
  
  - 鼓励开发者立即将这些模型集成到他们的系统中。
- **使用 Azure 构建多模态 RAG 系统**：一份分步指南解释了如何使用 Azure AI Search 和 Azure OpenAI 结合 LlamaIndex 创建多模态 RAG 系统，通过上下文信息提高检索准确性（[查看指南](https://t.co/RO5nQ79sqD)）。如[此推文](https://twitter.com/llama_index/status/1846668813980639343)所分享，该指南提供了有效实施的基准测试和技术。
  
  - 详细的演练重点介绍了改进不同 AI 系统之间上下文检索的方法。
- **明天关于 LlamaIndex 与 Elastic 的演讲**：由 @seldo 主讲的会议将讨论如何结合使用 LlamaIndex 和 Elastic，有望为开发者提供见解（[了解更多](https://t.co/tQszqtRN1Z)）。预计这次演讲将展示实际应用和集成技术。
  
  - 关注此次讨论，了解如何使用 Elastic 优化 LlamaIndex 工作流。

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1295831848382890146) (46 messages🔥):

> - `Neo4jPropertyGraphStore`
> - `LlamaIndex Typescript API calls`
> - `LlamaIndex partnership process`
> - `Warnings in module loads`
> - `Multi-agent orchestration in LlamaIndex`

- **加速 Neo4jPropertyGraphStore 的创建**：一位用户注意到创建 **Neo4jPropertyGraphStore** 需要很长时间，特别是在存储中有 **64322 个 nodes** 的情况下，并询问有关内存优化和 schema 简化的建议。
  
  - 讨论揭示了提高性能的潜在方法，例如将 `refresh_schema` 设置为 false，以避免与 schema 计数相关的昂贵调用。
- **在 LlamaIndex Typescript 中监控 API 调用**：一位用户询问如何监控通过 **LlamaIndex Typescript** 向 OpenAI 发起的 API 调用，寻求一种有效记录这些操作的方法。
  
  - 另一位成员分享说，使用 LlamaIndex 中的 observability 功能可以帮助记录事件并监控 LLM/prompt 的输入和输出。
- **了解 LlamaIndex 合作伙伴流程**：一位用户询问了成为 LlamaIndex 官方合作伙伴的流程以及资格标准。
  
  - 澄清说明目前没有官方合作伙伴，但各种集成了 LlamaIndex 的公司可以协助进行 RAG 应用开发。
- **Module 加载警告并非致命错误**：一位用户对 module 加载期间的警告表示担忧，询问其严重程度。
  
  - 回复指出这些警告可以安全地忽略，因为它们不是致命的。
- **使用 workflows 实现 multi-agent 编排**：一位用户询问是否可以在 LlamaIndex 中复制 OpenAI Swarm 的功能，并强调 workflows 是主要方法。
  
  - 提供了使用 workflows 进行 multi-agent 通信的示例，包括博客文章和 GitHub 仓库以供参考。

**相关链接**：

- [Partners — LlamaIndex, Data Framework for LLM Applications](https://www.llamaindex.ai/partners)：通过 LlamaIndex 专家更快地投入生产
- [Observability | LlamaIndex.TS](https://ts.llamaindex.ai/observability/)：LlamaIndex 提供一键式 observability 🔭，允许你在生产环境中构建规范的 LLM 应用程序。
- [[Bug]: Extremely long time initializing Neo4jPropertyGraphStore for larger graphs · Issue #16204 · run-llama/llama_index](https://github.com/run-llama/llama_index/issues/16204)：Bug 描述：初始化包含 3558 个实体的 graph store 大约需要 14 分钟。我觉得这是因为 refresh_schema() 不能很好地处理大型图。也许是因为没有使用 async？我粘贴了...

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1296125206762356837) (15 条消息🔥):

> - `Mistral 的新模型`
> - `Chatbot Arena 更新`
> - `Yi-Lightning 性能`
> - `Ministral 权重可用性`

- **Mistral 推出 Ministral 模型**：在 **Mistral 7B** 发布一周年之际，该公司推出了两款全新的边缘模型：**Ministral 3B** 和 **Ministral 8B**，旨在用于具备隐私优先推理能力的设备端场景。
  
  - 这些模型拥有高达 **128k 上下文长度**等特性，被定位为 10B 以下参数量类别中多种应用场景的领先竞争者。
- **Ministral 3B 缺少权重**：讨论围绕 **Ministral 3B** 权重的缺失展开，引发了对其与提供非商业权重的 **Ministral 8B** 相比潜在性能的疑问。
  
  - 社区对不公开该模型权重的决定表示失望和好奇。
- **Yi-Lightning 性能飙升**：来自 **@01AI_YI**（零一万物）的新发布模型 **Yi-Lightning** 在 **Chatbot Arena** 中备受关注，总榜排名第 6，在数学和编程方面表现强劲。
  
  - 该模型的崛起获得了超过 **1.3 万次社区投票**的认可，显示出其强大的能力，足以媲美 **Grok-2** 等知名同行。
- **对模型评估的担忧**：在关于模型性能的讨论中，有人注意到 **Gemma2 9B** 被排除在对比表之外，这可能凸显了基准测试评估中的不一致性。
  
  - 评论建议需要一个更统一的评估代码库，因为观察到了性能指标的波动。

**提到的链接**：

- [Un Ministral, des Ministraux](https://mistral.ai/news/ministraux/)：介绍世界上最好的边缘模型。
- [Armand Joulin (@armandjoulin) 的推文](https://x.com/armandjoulin/status/1846581336909230255)：一系列出色的模型已进入竞技场！遗憾的是 Gemma2 9B 从其中一个表格中掉出，所以我不得不手动添加。如果所有模型都能用相同的代码库进行评估就更好了，因为我看到了波动……
- [lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文](https://x.com/lmarena_ai/status/1846245604890116457)：来自 Chatbot Arena 的重大新闻！@01AI_YI 的最新模型 Yi-Lightning 已在 Arena 中经过广泛测试，收集了超过 1.3 万次社区投票！Yi-Lightning 已攀升至总榜第 6 位……

---

### **Interconnects (Nathan Lambert) ▷ #**[**random**](https://discord.com/channels/1179127597926469703/1183121795247779910/1295825683578097816) (16 条消息🔥):

> - `AI 实习`
> - `AGI 末日时钟`
> - `实习生候选人之间的竞争`

- **沙特资助的商学院发布末日时钟**：瑞士一家受沙特资助的商学院发布了一个末日时钟，以警告“不受控制的通用人工智能（AGI）”的危险，将其称为“神一般的” AI，类比过去社会对核威胁的恐惧。
  
  - 时钟创建者、教授 Michael Wade 在最近为[《时代周刊》（TIME）撰写的专栏文章](https://time.com/7086139/ai-safety-clock-existential-risks/)中详细介绍了这一倡议。
- **AI2 OLMo 实习机会**：AI2 正在为 OLMo 项目招聘研究实习生，提供具有竞争力的薪资（**86,520 美元至 123,600 美元**），以及领导 NLP 和机器学习领域重大研究的机会。
  
  - 实习生可以定义研究项目，与团队成员合作，并在为期 **12 周的实习**（开始时间灵活）中在知名期刊上发表论文。
- **AI 实习竞争激烈**：关于 OLMo 实习竞争性质的讨论兴起，特别是出现了像顶尖 AI 实验室的“后训练负责人（Post-training Lead）”这样的申请者。
  
  - 讨论提到，这种级别的竞争使得该实习对研究生来说极具挑战性。
- **对研究生竞争的担忧**：一位成员对研究生在实习选拔过程中与经验极其丰富的候选人竞争所面临的压力表示担忧。
  
  - 这种情绪得到了其他人的共鸣，突显了获得这些令人垂涎的机会所面临的困难。

**提到的链接**：

- [看在上帝的份上，别再搞那些高深莫测的末日时钟了](https://gizmodo.com/for-the-love-of-god-stop-making-inscrutable-doomsday-clocks-2000512111)：一家商学院正利用 AI 末日论、来自沙特阿拉伯的资金以及陈旧的冷战隐喻来炒作 AI 的未来。
- [Matt Shumer (@mattshumer_) 的推文](https://x.com/mattshumer_/status/1846209244284219703)：http://x.com/i/article/1846205240728588288
- [Allen Institute for AI 的 OLMo 研究实习职位申请](https://job-boards.greenhouse.io/thealleninstitute/jobs/6322728)：未找到描述

---

### **Interconnects (Nathan Lambert) ▷ #**[**posts**](https://discord.com/channels/1179127597926469703/1228051082631188530/1295948654099693639) (18 消息🔥):

> - `Snailbot 的双重功能`
> - `音频分发挑战`
> - `Hackernews 发布问题`

- **Snailbot 承担双重职责**：讨论强调了 **Snailbot** 被用于 **音频流发布 (audio feed posts)**，展示了其扩展的功能。
  
  - 一位用户评价这种新用途是“一举两得 (twofer)”，表达了兴奋和新奇感。
- **音频分发面临困难**：制定有效的 **音频内容分发** 策略仍不明朗，多位用户表达了他们的困难。
  
  - 一位用户幽默地将他们的处境比作一个流行的笔记应用梗 (meme)，传达了挫败感。
- **Hackernews 发布陷阱**：人们对 **在 Hackernews 上发布内容** 的挑战表示担忧，特别是关于链接可见性和点赞 (upvoting) 动态。
  
  - 一位成员指出，**直接链接可能会面临惩罚**，使分享过程变得复杂，并劝阻用户直接索要点赞。
- **寻找可见性问题的解决方案**：参与者讨论了保持链接有效性的策略，建议用户告知他人去哪里寻找内容，而不是直接提供链接。
  
  - 在 **Hackernews** 上获得关注的过程被描述为变幻莫测，为分发制造了进一步的障碍。

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1295834063000703027) (37 消息🔥):

> - `Gemini 免费版性能`
> - `Mistral 的新模型`
> - `Nvidia 的 Llama 3.1 Nemotron`
> - `E2B 的 SDK 发布与融资`
> - `AI 算力与核能`

- **Gemini 免费版面临性能问题**：用户报告了 [Gemini 免费版](https://gemini.free.url) 的 *超时和失败*，对每天 *15 亿 (1.5B) token* 的说法表示怀疑，特别是在当前的速率限制 (rate limits) 下。
  
  - 一位成员推测，实际有效使用量可能低得多，可能仅为 *0.5 亿 (0.05B) token* 左右。
- **Mistral 发布新的边缘模型**：Mistral 推出了 *Ministral 3B* 和 *Ministral 8B* 模型，专为设备端应用设计，在 10B 以下参数类别中突破了常识和推理能力的界限。
  
  - 然而，批评指出 *3B 模型仅限 API 使用*，限制了其设备端效用，并引发了对独立开发者限制性许可的担忧。
- **Nvidia 的 Llama 3.1 Nemotron 抢尽风头**：根据最近的发布公告，Nvidia 新的 *Llama 3.1 Nemotron 70B* 模型在各种基准测试中据称优于 *GPT-4o* 和 *Claude Sonnet 3.5*。
  
  - 社区反响热烈，质疑 *Sonnet 3.5 的拥趸* 是否真的能与这款新发布的模型相提并论。
- **E2B 发布 SDK 并获得巨额融资**：E2B 宣布发布其 v1.0 SDK，并完成了 1150 万美元的种子轮融资，旨在为带有安全沙箱 (sandboxes) 的 AI 代码解释提供基础设施。
  
  - 该初创公司每月已运行数百万个沙箱，其与 *Perplexity* 等知名客户的合作关系备受关注。
- **建议建立 LLM 性能基准测试**：一位成员提议创建一个专门针对 LLM 的 *CPUBenchmark 风格* 的对比工具，因为现有的排行榜不利于直接的模型比较。
  
  - 目前的工具，如 *lmsys/hugging face 排行榜*，存在阻碍有效模型比较的局限性。

**链接提及**:

- [Un Ministral, des Ministraux](https://mistral.ai/news/ministraux/): 介绍全球最顶尖的边缘模型。
- [Tweet from Yohei (@yoheinakajima)](https://x.com/yoheinakajima/status/1846289276151255187?s=46): 介绍 "ditto"，最简单的自构建编码 Agent 📄 约 500 行代码 🛠️ 可构建多文件应用 🔁 一个带有 5 个工具的简单 LLM 循环，支持 GitHub/Replit 等 👇
- [3b is is API-only so you won’t be able to run it on-device, which is the killer ... | Hacker News](https://news.ycombinator.com/item?id=41860918): 未找到描述
- [Tweet from maharshi (@mrsiipa)](https://x.com/mrsiipa/status/1846517901957734733?s=46): NVIDIA 随手发布了一个开源 70B 模型，击败了 GPT-4o 和 Claude 3.5 Sonnet
- [Tweet from Vasek Mlejnsky (@mlejva)](https://x.com/mlejva/status/1846568274009698402?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): 今天，我很高兴发布 @e2b_dev，推出我们 SDK 的 v1.0 版本，并宣布 1150 万美元的种子轮融资！我们正在为 AI 代码解释构建基础设施。用于运行 A... 的安全 E2B Sandboxes
- [Tweet from morgan — (@morqon)](https://x.com/morqon/status/1846184256877244704?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): OpenAI 获得了一个拥有 10 万块 B200 的集群，初始配备 206 MW 可再生能源，从 Oracle 租赁，由 Crusoe 设计、建造和运营，将于 2025 年上半年上线。新闻稿称：“创纪录的...”
- [Tweet from morgan — (@morqon)](https://x.com/morqon/status/1846184256877244704?s=46&t=6FDPaNxZcbSsELal6Sv7U): OpenAI 获得了一个拥有 10 万块 B200 的集群，初始配备 206 MW 可再生能源，从 Oracle 租赁，由 Crusoe 设计、建造和运营，将于 2025 年上半年上线。新闻稿称：“创纪录的...”
- [Tweet from Vasek Mlejnsky (@mlejva)](https://x.com/mlejva/status/1846568274009698402?s=46&t=6FDPaNxZcbSs): 今天，我很高兴发布 @e2b_dev，推出我们 SDK 的 v1.0 版本，并宣布 1150 万美元的种子轮融资！我们正在为 AI 代码解释构建基础设施。用于运行 A... 的安全 E2B Sandboxes
- [Tweet from Philipp Schmid (@_philschmid)](https://x.com/_philschmid/status/1846527494351998980): NVIDIA 是否悄悄发布了一个性能超越 @OpenAI GPT-4o 和 @AnthropicAI Claude Sonnet 3.5 的 Llama 3.1 70B 微调版本？昨天，@nvidia 添加了 Llama 3.1 Nemotron 70B Instruct，这是一个经过进一步 RLHF 的模型...
- [Tweet from Find anything. Protect everything | Dropbox Dash](https://dash.dropbox.com/): Dropbox Dash for Business 将 AI 通用搜索和组织功能与通用内容访问控制相结合。轻松跨应用查找、组织、共享和保护内容，让你专注于...
- [Amazon, Google make dueling nuclear investments to power data centers with clean energy](https://apnews.com/article/climate-data-centers-amazon-google-nuclear-energy-e404d52241f965e056a7c53e88abc91a): 科技巨头 Amazon 和 Google 正在投资下一代核反应堆。两家公司都在寻求新的无碳电力来源，以满足数据中心日益增长的需求和...
- [New nuclear clean energy agreement with Kairos Power](https://blog.google/outreach-initiatives/sustainability/google-kairos-power-nuclear-energy-agreement/): Google 的首个核能协议是迈向通过投资先进清洁能源技术帮助全球脱碳的一步。

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1295863668843741335) (9 条消息🔥):

> - `社区灵感`
> - `Cohere 的工作机会`

- **社区每天都在激励人心**：一位成员表示，他们每天都能从 **Cohere 社区** 中获得灵感，强调了其积极影响。
  
  - 另一位成员表示赞同，说道：*说实话，这个社区的很多事情，每一天都在激励着我！*
- **工作机会澄清**：一位成员提醒其他人，该频道不是寻找 Cohere 职位的合适场所，并提供了 [招聘页面](https://cohere.com/careers) 的链接用于申请。
  
  - 他们强调了 **Cohere 团队** 在利用 ML/AI 技术解决现实世界问题方面的热情，团队成员分布在多个地点。

 

**链接提到**: [Careers](https://cohere.com/careers): 我们的 ML/AI 专家团队热衷于帮助开发者解决现实世界的问题。在多伦多、伦敦和帕洛阿尔托的办公室，我们工作在机器学习的最前沿，以释放...

 

---

### **Cohere ▷ #**[**announcements**](https://discord.com/channels/954421988141711382/996880279224451154/1296086123436576821) (1 条消息):

> - `RAG++ 课程`
> - `与 RAG 专家进行 AMA`

- **第二轮与 RAG 专家的 AMA**：由于第一轮 AMA 反响热烈，请在明天 **东部时间上午 11:00** 加入我们，与 **RAG** 开发专家 **Ayush Thakur** 和 **Meor Amer** 进行另一场实时聊天。
  
  - 本次会议承诺提供来自 Weights & Biases 和 Cohere 合作的 [RAG++ 课程](https://www.wandb.courses/courses/rag-in-production) 的*幕后见解*。
- **AMA 活动链接**：明天 AMA 的活动链接可在[此处](https://discord.gg/ggTQjNUP?event=1291381850610077726)获取。
  
  - 请务必在日历上做好标记，并准备好任何关于高级 RAG 开发的问题！

 

---

### **Cohere ▷ #**[**api-discussions**](https://discord.com/channels/954421988141711382/1168578329423642786/1295866536375881790) (22 条消息🔥):

> - `Cohere Embed API 错误处理`
> - `减少 RAG 检索到的分块 (Chunks)`
> - `试用 Key 的速率限制 (Rate Limits)`
> - `试用 Key 的模型使用情况`

- **Cohere Embed API 错误处理详解**：一位用户询问了在使用 **Cohere Embed API** 时如何处理错误，特别是当 96 个批次中的一个文档嵌入失败时。
  
  - *错误可能导致整个批次失败*，因此建议根据特定的错误代码创建重试逻辑。
- **加速 RAG 功能**：为了提高 RAG 中的引用速度，将 `citations_quality` 切换为 **FAST** 可以显著提升性能。
  
  - 用户可以通过在 n 个引用后手动截断，或为 top_n 分块实现排名系统来减少总引用量。
- **讨论试用 Key 的速率限制**：另一位成员在使用试用 Key 时遇到了 **TooManyRequestsError**，并获知试用 Key 每月允许最多 **1,000 次 API 调用**。
  
  - 用户注意到**速率限制是与账户绑定的**，而非单个试用 Key，并建议升级到生产 Key 以获得更高的限制。
- **Cohere 试用 Key 使用问题**：一位用户报告称，他们可以在仪表板中使用试用 Key，但在通过 Cohere 依赖项访问模型时遇到问题。
  
  - 尽管可以在仪表板上使用试用 Key，但 API 访问的限制令人担忧，得到的建议是等待试用 Key 重置。

 

**提到的链接**：[Http status codes — Cohere](https://docs.cohere.com/v2/reference/errors)：了解 Cohere 的 HTTP 响应代码以及如何在各种编程语言中处理错误。

 

---

### **Cohere ▷ #**[**projects**](https://discord.com/channels/954421988141711382/1218409701339828245/1296037628725956639) (2 条消息):

> - `系统提示词 (System Prompt) 模板`
> - `深度问答评估应用`

- **对系统提示词模板的热烈反响**：一位成员对现有的各种**系统提示词模板**表示了极大的热情，称有很多模板可供选择。
  
  - 这种兴奋凸显了用户在优化 AI 交互方面的参与度和兴趣。
- **发布深度问答评估应用**：一位成员在 Medium 上发表了他们的第一篇文章，详细介绍了**深度问答评估应用**，该应用使用了 **Streamlit** 和 **Gemini 1.5 Pro**。
  
  - 该应用旨在通过实时反馈增强学习，改变用户评估知识的方式，并向提供该想法的 Dr. Fady AlNajjar 致敬。

 

**提到的链接**：[Enhancing Learning Through Real-Time Feedback: In-Depth Question Answering Evaluation App](https://medium.com/@d.isham.ai93/enhancing-learning-through-real-time-feedback-in-depth-question-answering-evaluation-app-4f68c423e496)：在在线学习和自我提升的世界中，拥有有效的工具来评估进度至关重要。无论你是在学习……

 

---

### **Cohere ▷ #**[**cohere-toolkit**](https://discord.com/channels/954421988141711382/1254901651081269268/1296156993777827904) (2 条消息):

> - `文本转语音 (Text-to-speech) 可用性`
> - `聊天机器人回复`

- **聊天机器人现在支持文本转语音了！**：文本转语音功能现在可用于聊天机器人回复，并为用户提供了详细的[设置指南](https://github.com/cohere-ai/cohere-toolkit/blob/main/docs/text_to_speech.md)。
  
  - 这一新功能旨在通过更具动态性的音频回复来增强用户交互。
- **用户对新功能的兴奋**：在文本转语音功能发布后，一位用户表达了兴奋之情，称其“太酷了！”。
  
  - 这种热情表明用户对新推出的功能持积极接受态度。

 

---

### **Modular (Mojo 🔥) ▷ #**[**general**](https://discord.com/channels/1087530497313357884/1098713601386233997/1295855516953612329) (2 条消息):

> - `Playground 更新`
> - `社区会议展示`

- **Playground 获得热烈好评**：成员们对 **Playground** 功能表达了极大的喜爱，感谢 **Modular** 的改进与支持。
  
  - 欲了解更多信息，请参阅 [Playground 文档](https://docs.modular.com/mojo/playground)。
- **记下社区展示会的日期**：**社区会议**定于 10 月 21 日举行，届时将进行现场展示，参与者可以演示他们的 **MAX** 和 **Mojo** 项目。
  
  - 每个时段持续 **5-10 分钟**，为分享学习心得和收集反馈提供机会。

 

**提到的链接**：[Modular 文档](https://docs.modular.com/mojo/playground)：未找到描述

 

---

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1295895675711524924) (8 条消息🔥):

> - `Mojo Bug`
> - `Deque 代码贡献`
> - `在 YMM 寄存器中存储 SIMD`
> - `在 Mojo 中使用 OpenCV`
> - `Mojo 标准库`

- **修复奇怪的 Mojo Bug**：一位成员发现了一个可复现的 **Mojo bug**，但随后自行修复了它，并表示如果提交，愿意将相关贡献添加到更新日志（changelog）中。
  
  - 他们鼓励其他人报告类似问题，以改进平台。
- **Mojo 中的 Deque 代码贡献**：另一位成员确认该问题已在最新的 nightly 版本中解决，并对向 Mojo 贡献 **deque 代码**表示兴奋。
  
  - 他们提到 Joe Loser 很快会查看 deque 代码。
- **关于在 YMM 寄存器中存储 SIMD 的咨询**：有人提出了一个关于在 **YMM 寄存器**中存储 **SIMD** 的具体方法的问题，以及如果大小合适，Mojo 是否会自动处理。
  
  - 这引发了围绕 Mojo 中 SIMD 存储实现的讨论。
- **OpenCV 在 Mojo 中的可用性**：一位成员询问是否可以在 Mojo 中使用 **OpenCV**，强调了对图像处理能力的需求。
  
  - 目前尚未提供明确答复，引发了进一步的好奇。
- **Mojo 标准库的目标**：一位成员询问 Mojo 是否旨在重新实现整个 **Python 标准库**，因为它力求成为 Python 的超集。
  
  - 另一位成员推测，要实现如此广泛的功能还需要很长时间。

 

---

### **Modular (Mojo 🔥) ▷ #**[**max**](https://discord.com/channels/1087530497313357884/1212827597323509870/1295966574875775083) (22 messages🔥):

> - `LLM 推理的高层级 API`
> - `Inferencemax 开发`
> - `Mojo 与 Python 实现对比`
> - `Jakub 为 MAX 做的 Python API 工作`

- **寻找 LLM 的高层级 API**：一位成员表达了对 LLM 推理高层级 API 的需求，类似于 HF Transformers 库，因为目前的 MAX 示例需要编写大量代码。
  
  - 另一位成员提到 Max 可能已经支持这一点，但现有的示例过于底层，需要数百行代码。
- **Inferencemax 项目介绍**：一位成员分享了他们名为 [Inferencemax](https://github.com/teilomillet/inferencemax) 的新项目，旨在简化 LLM 推理，尽管他们指出这可能并不完全符合上述需求。
  
  - 该代码目前是用 Python 编写的，虽然被认为不是最优的，但已计划进行性能改进。
- **潜在的 Mojo 实现**：随后讨论了实现 Mojo 版本 Inferencemax 的可能性，重点讨论了 Mojo 对某些开发者的实用性。
  
  - 成员们鼓励参考示例代码作为资源，特别是针对 Llama3 列出的代码。
- **Jakub 在 MAX Python API 方面的工作**：一位成员询问了 Jakub 对 MAX Python API 的贡献，另一位成员提供了一个 [社区会议](https://youtu.be/Wm-x1or345I?t=5) 的链接，Jakub 是其中的第一位发言人。
  
  - 会上指出该 API 尚未完全发布，仅存在于 nightly builds 中，但其目标是展示易用性。
- **持续学习与资源共享**：成员们表达了学习新 API 进展的热情，并表示会多次观看社区视频以加深理解。
  
  - 讨论强调了社区资源和协作在改进个人项目和提升理解方面的价值。

**提到的链接**：

- [MAX Examples | Modular Docs](https://docs.modular.com/max/examples/)：使用 MAX API 的即插即用代码示例
- [max/examples/graph-api/pipelines/llama3 at main · modularml/max](https://github.com/modularml/max/tree/main/examples/graph-api/pipelines/llama3)：一系列示例程序、Notebook 和工具，展示了 MAX 平台的强大功能 - modularml/max
- [GitHub - teilomillet/inferencemax](https://github.com/teilomillet/inferencemax)：通过创建 GitHub 账号为 teilomillet/inferencemax 开发做出贡献。

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1295840384236130325) (21 messages🔥):

> - `矿产资源海报`
> - `SD3 人体姿态限制`
> - `LLM Token 限制问题`
> - `LyCORIS 与 LoRA 详解`
> - `Web3 项目职位空缺`

- **寻求创建矿产资源海报的帮助**：一位成员请求协助为他们的大学项目制作一份关于 **矿产资源** 的海报，向他人寻求帮助。
  
  - 另一位成员鼓励他们直接在聊天中发布具体需求以获得支持。
- **理解 SD3 在人体方面的局限性**：讨论了 **SD3** 在处理躺下或倒立的人体姿态时的表现，一位成员评论说其表现通常很差。
  
  - 另一位参与者认为，无论什么姿势都会出现问题，图像中经常出现畸变。
- **对 LLM 不遵守 Token 限制的挫败感**：一位用户对 LLM 模型不遵守 **token limits** 或停止命令表示沮丧，导致出现不连贯的重复和伪造输出。
  
  - 他们推测潜在原因可能是 Prompt Templating 问题，并寻求更有经验者的建议。
- **关于 LyCORIS 与 LoRA 文件夹的澄清**：一位成员询问为什么在将所有内容移动到 **LoRA** 后仍然存在 **LyCORIS** 文件夹。
  
  - 另一位用户澄清说，这源于早期对扩展组件的历史需求，现在已集成到 Auto1111 等界面中。
- **新 Web3 项目的职位招聘**：宣布启动一个新的 **Web3 项目**，提供包括开发者和版主在内的多个职位，并提供具有竞争力的薪资。
  
  - 鼓励感兴趣的候选人直接联系以获取更多细节。

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1295842270934732901) (17 条消息🔥):

> - `Open Interpreter GitHub Copilot 扩展`
> - `Mozilla AI 演讲公告`
> - `Kernel panic 问题`
> - `理解 bandwidth`
> - `GitHub Marketplace 扩展上架`

- **Open Interpreter GitHub Copilot 扩展构思**：一位成员建议创建一个 **Open Interpreter GitHub Copilot 扩展**，对此另一位成员表示他们目前没有足够的 **bandwidth**（精力/带宽）来推进，但支持社区的努力。
  
  - 他们鼓励社区承担该项目，并尽可能提供指导。
- **Mozilla AI 演讲即将举行**：MikeBirdTech 宣布对即将到来的 **Mozilla AI** 演讲感到兴奋，并敦促成员将该活动添加到日历中。
  
  - 同时也分享了活动链接以便快速访问。
- **关闭应用时出现 Kernel panic**：一位成员报告在尝试关闭 Open Interpreter 应用时遇到了 **kernel panic**。
  
  - MikeBirdTech 建议在特定频道创建一个包含版本详情的专门帖子，以便有效地排查问题。
- **关于 bandwidth 的澄清**：讨论了 **bandwidth** 一词，一位成员解释说它指的是他们用于新项目的可用**时间与资源**。
  
  - 另一位成员幽默地承认了自己在理解该术语时的错误，并对讨论带来的见解表示赞赏。
- **GitHub Marketplace 扩展上架标准**：一位成员澄清说，在 GitHub Marketplace 上架扩展没有特定的 **bandwidth 要求**，重点在于满足平台的标准。
  
  - 他们概述了创建和发布扩展的关键步骤，强调了提供用户价值和集成的重要性。

 

**提到链接**：[来自 Mike Bird (@MikeBirdTech) 的推文](https://x.com/MikeBirdTech/status/1846283357153268002)：pip install --upgrade open-interpreter 一个 π 版本发布！

 

---

### **OpenInterpreter ▷ #**[**ai-content**](https://discord.com/channels/1146610656779440188/1149229778138824765/1296144967403704402) (2 条消息):

> - `本地 LLMs`
> - `Hugging Face`
> - `Ollama 集成`
> - `Llama 3.2 3B`

- **本地 LLMs 现在更易于使用**：一项重大更新允许用户通过 **Ollama** 直接在 [Hugging Face](https://huggingface.co) 上轻松运行任何 **GGUF** 模型，只需指向仓库并执行脚本即可。
  
  - 例如，用户可以使用命令 `ollama run hf(.)co/hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF` 来运行 **Llama 3.2 3B**。
- **功能评价**：一位成员对这一新功能表示热烈欢迎，称其为本地 LLM 的一项令人兴奋的进展。
  
  - 他们还指出，这是他们非常欣赏 **Jan** 的一个功能，而之前 **Ollama** 缺少这一功能。

 

**提到链接**：[来自 Philipp Schmid (@_philschmid) 的推文](https://fxtwitter.com/_philschmid/status/1846554632333513035?s=46&t=G6jp7iOBtkVuyhaYmaDb0w)：本地 LLM 的重大更新！很高兴分享你现在可以直接通过 @ollama 轻松使用 @huggingface 上的任何 GGUF 模型！只需指向 Hugging Face 仓库并运行它！以下是如何运行 @...

 

---

### **DSPy ▷ #**[**show-and-tell**](https://discord.com/channels/1161519468141355160/1202371242519441499/1295825107213750346) (5 条消息):

> - `DSPy 工作流系统`
> - `dspygen 框架更新`
> - `现场编程 DSPy Signatures`
> - `DSPy 单元测试`
> - `Loom 录屏`

- **对 DSPy 驱动的工作流系统进行单元测试**：一位成员宣布他们正在 **Discord** 频道中对一个 **DSPy 驱动的工作流系统**进行单元测试。
  
  - *请查看频道以获取测试过程的进度更新和反馈。*
- **dspygen 框架重大更新**：[dspygen](https://github.com/seanchatmangpt/dspygen) 框架最近发布了一个**重大更新**，该框架旨在改进 **dslmodel** 之外的功能。
  
  - **dspygen** 旨在增强 **GPT**、**BERT** 和 **LLaMA** 等语言模型的 **DSPy** 工作流。
- **DSPy Signatures 的现场编程装饰器**：一位成员主持了现场编程（livecoding）会议，重点是创建一个将 **DSPy signatures** 与 Custom GPTs 合并的 **decorator**（装饰器）。
  
  - 参与者可以加入 **Discord 频道**中的会议，获取实时更新和演示。
- **关于 DSPy 主题的 Loom 录屏**：分享了两段 **Loom 录屏**，展示了 **DSPy** 开发的不同方面。
  
  - 这些录屏可以为正在进行的工作和所采用的策略提供进一步的见解。

 

**提到链接**：[GitHub - seanchatmangpt/dspygen: 一个为 GPT、BERT 和 LLama 等语言模型设计的 Ruby on Rails 风格的 DSPy (Demonstrate, Search, Predict) 项目框架。](https://github.com/seanchatmangpt/dspygen)：一个为 GPT、BERT 和 LLama 等语言模型设计的 Ruby on Rails 风格的 DSPy (Demonstrate, Search, Predict) 项目框架。- seanchatmangpt/dspygen

 

---

### **DSPy ▷ #**[**papers**](https://discord.com/channels/1161519468141355160/1203568372667645963/1295958514048434247) (1 条消息):

> - `LightRAG improvements`
> - `GraphRAG limitations`
> - `Retrieval-Augmented Generation systems`

- **LightRAG 表现优于 GraphRAG**：根据[这篇论文](https://arxiv.org/abs/2410.05779)的详细描述，最近的观点认为 **LightRAG** 相比 **GraphRAG** 在有效性和**成本效率**方面有显著提升。
  
  - 作者提出 **LightRAG** 解决了现有 **RAG** 系统的局限性，通过创新的图结构提高了**上下文感知 (contextual awareness)** 和信息检索能力。
- **RAG 系统面临挑战**：目前的 **RAG** 系统在扁平化数据表示和上下文理解不足等方面存在困难，导致生成的回答较为碎片化。
  
  - 拟议的 **LightRAG** 框架试图通过在文本索引和检索过程中引入图结构来解决这些问题。

**提到的链接**：[LightRAG: Simple and Fast Retrieval-Augmented Generation](https://arxiv.org/abs/2410.05779)：检索增强生成 (RAG) 系统通过整合外部知识源来增强大语言模型 (LLM)，从而能够为用户提供更准确、更具上下文相关性的定制化回答...

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/1295846420028461098) (7 条消息):

> - `DSPy integration with GPT-O1+`
> - `Documentation revamp discussions`
> - `HTML vs. notebooks for documentation`

- **DSPy 与 GPT-O1+ 的集成取得进展**：更新后的文档引入了一个长篇 **RAG** 示例，用于构建一个关于 Linux 或 iPhone 应用等技术话题的**问答系统**。
  
  - 用户可以使用 `pip install -U dspy` 安装 **DSPy**，[DSPy 文档](https://dspy-docs.vercel.app/docs/quick-start/getting-started-01)中提供了相关教程。
- **重构文档方案**：关于即将进行的 **DSPy** 文档重构展开了讨论，重点在于改进节奏和风格。
  
  - 参与者正在考虑是使用 **HTML** 文档还是详细的 **notebooks**，并提到了**执行缓存 (caches for execution)** 的实用性。
- **更倾向于 HTML 文档**：一位成员表示强烈倾向于使用 **HTML** 格式作为文档，而不是独立的 **notebooks**。
  
  - 他们建议在代码仓库中保留详细的代码示例，同时在文档中提供简洁的入门指南。
- **基于现有文档框架构建**：在关于 **HTML** 文档的建议之后，重点转移到了记录 **DSPy** 能力的现有构建块上。
  
  - 社区认为，增强现有文档将足以覆盖 **primitives**、**optimizers**、**metrics** 以及常用技术。

**提到的链接**：[Getting Started I: Basic Question Answering | DSPy](https://dspy-docs.vercel.app/docs/quick-start/getting-started-01)：让我们通过一个简单的例子来了解 **DSPy** 中的基础问答。具体来说，我们将构建一个用于回答技术问题（例如关于 Linux 或 iPhone 应用）的系统。

---

### **LangChain AI ▷ #**[**announcements**](https://discord.com/channels/1038097195422978059/1058033358799655042/1295851845583110175) (1 条消息):

> - `New Community Launch`
> - `Feedback Request`
> - `Moderator Opportunities`
> - `Discord Closure`

- **LangChain 社区将于 2024 年 10 月 31 日关闭**：在 **2024 年 10 月 31 日**，**LangChain** 将关闭当前的 **Discord** 社区，以建立一个全新的、改进后的用户空间。
  
  - 目标是创建一个更有帮助、更具互动性且更有趣的社区。
- **获取新社区的最新动态**：要了解即将推出的新社区的信息，成员可以填写[此处的表单](https://airtable.com/app9AB74Dql7uubL2/pagTKrmJu1rQRkJKV/form)。
  
  - 为了获得更好的体验，建议用户使用最新版本的 **Chrome**、**Firefox**、**Safari** 或 **Edge**。
- **改进建议**：**LangChain** 正在征求关于如何增强新社区空间的反馈，并欢迎提出**想法和建议**。
  
  - 成员可以通过 [**community@langchain.dev**](mailto:community@langchain.dev) 分享反馈，以帮助塑造新环境。
- **招募版主**：**LangChain** 还在寻找有兴趣的人士加入新社区担任**版主 (moderators)**。
  
  - 任何愿意以此身份提供支持的人员请联系并表达意向。

**提到的链接**：[Airtable | Everyone's app platform](https://airtable.com/app9AB74Dql7uubL2/pagTKrmJu1rQRkJKV/form)：**Airtable** 是一个用于构建协作应用的低代码平台。定制您的工作流，进行协作并实现宏伟目标。免费开始使用。

---

### **LangChain AI ▷ #**[**general**](https://discord.com/channels/1038097195422978059/1038097196224086148/1295854337247154207) (3 messages):

> - `API Routing with Agents`
> - `Docker Compose`

- **寻求 API 路由建议**：一位成员正在寻求指导，关于如何根据 Agent 的描述，使用 Agent 将用户问题路由到不同 API 的最佳选择。
  
  - 他们有 **5 个 API** 运行在 **Docker Compose** 中，这表明他们的项目有一个结构化的设置。
- **聊天中的一般查询**：一位用户询问收到了 **10 次 ping**，暗示可能存在通知或系统警报问题。
  
  - 这引发了关于小组内沟通有效性和偏好的问题。

 

---

### **LangChain AI ▷ #**[**langserve**](https://discord.com/channels/1038097195422978059/1170024642245832774/1296022326206533664) (4 messages):

> - `Remote Runnable Tools Binding`
> - `Playground Blank Page Issue`
> - `GitHub Issue Tracking`

- **询问 Remote Runnable 工具绑定**：一位成员询问是否可以将工具绑定到 **Remote Runnable**，并指出它缺少 **bind_tools()** 方法。
  
  - 这一请求为有效处理工具绑定的潜在增强功能打开了大门。
- **Playground 在处理 Optional 字段时遇到问题**：成员们发现 **Playground** 的一个重要问题，即当输入类型包含 **Optional** 字段时，会导致页面空白并在控制台中记录错误。
  
  - 输入 schema 的 **null** 类型被认为导致了与 ***jsonforms*** 的兼容性问题，从而阻碍了其功能。
- **针对 Playground 问题提交了 GitHub Issue #782**：一位成员在 GitHub 上报告了 Playground 的问题，详细说明了包含可选字段的 chain 会导致加载失败和控制台错误。
  
  - 该问题已记录在 [GitHub Issue #782](https://github.com/langchain-ai/langserve/issues/782) 中，以跟踪解决过程。

 

**提到的链接**：[Input type with](https://github.com/langchain-ai/langserve/issues/782) `Optional` field breaks Playground · Issue #782 · langchain-ai/langserve: 如果 chain 的输入类型包含可选字段，Playground 页面将无法加载（空白页），并且浏览器控制台中会记录以下错误：index-400979f0.js:150 Uncaught E...

 

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1295918145055359060) (4 messages):

> - `AIFoundry start-up`
> - `Mistral AI model`
> - `Mistral license requirements`

- **AIFoundry 寻求 GitHub 设计方面的指导**：来自 [AIFoundry.org](https://discord.gg/aSHN7W5E) 的 Yulia 表达了对 Axolotl 有序的 GitHub 仓库的钦佩，并正在寻求类似流程的指导。
  
  - 她询问是否有合适的人选可以协助他们这家专注于本地模型推理的开源初创公司。
- **Mistral AI 模型访问要求**：分享了 [Hugging Face](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410) 上新的 **Mistral-8B-Instruct-2410** 模型的链接，强调必须提供联系信息才能访问该模型。
  
  - 任何标准访问之外的使用都需要获得 Mistral AI 的许可，并鼓励个人查看其 [隐私政策](https://mistral.ai/terms/)。

 

**提到的链接**：[mistralai/Ministral-8B-Instruct-2410 · Hugging Face](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410)：未找到描述

 

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**other-llms**](https://discord.com/channels/1104757954588196865/1104758057449308220/1296155278949093491) (2 messages):

> - `L3.1 Ethereal Rainbow`
> - `Finetuning on L3.1`
> - `Sensitive Content`

- **L3.1 Ethereal Rainbow 仓库发布**：[L3.1 Ethereal Rainbow](https://huggingface.co/invisietch/L3.1-EtherealRainbow-v1.0-rc1-8B) 仓库已被标记为包含敏感内容，并可能含有有害信息。
  
  - 由于材料的敏感性质，建议用户在**查看内容**时保持谨慎。
- **L3.1 的微调 (Finetuning) 细节**：L3.1 模型已在**超过 2.5 亿个 token** 上进行了微调，序列长度为 16k。
  
  - 该微调过程专注于 **RP（角色扮演）和创意写作**，增强了模型在这些领域的性能。

 

**提到的链接**：[invisietch/L3.1-EtherealRainbow-v1.0-rc1-8B · Hugging Face](https://huggingface.co/invisietch/L3.1-EtherealRainbow-v1.0-rc1-8B)：未找到描述

 

---

### **Torchtune ▷ #**[**papers**](https://discord.com/channels/1216353675241590815/1293438210097025085/1296015353649823797) (2 条消息):

> - `新论文讨论`

- **对新论文的兴奋**：成员们对标题为 [arxiv:2410.06511](https://arxiv.org/abs/2410.06511) 的论文表示了极大的热情，认为这是一篇非常值得一读的佳作。
  
  - 为了表示肯定，一位成员补充说他们也正在研读这篇论文，并强调了其高质量。
- **对论文质量的共识**：关于该论文的评价非常一致，多位成员提到了其令人印象深刻的内容。
  
  - 一位成员强调，随着他们深入研究细节，该论文仍处于进展中，这标志着大家的共同兴趣。

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-readings-discussion**](https://discord.com/channels/1280234300012494859/1282735578886181036/1296170559926698117) (1 条消息):

> - `LLM 优化`
> - `Language-Model-Based Evolutionary Optimizer (LEO)`
> - `Zero-shot 优化应用`
> - `工程设计应用`

- **LLM 在优化任务中表现出色**：最近的研究强调，**Large Language Models** (LLM) 可以在各种具有挑战性的问题上执行 **zero-shot optimization**，包括多目标问题。
  
  - *这一系列工作*可能对**火箭喷嘴设计**和**风电场布局优化**等领域的实际应用具有重要意义。
- **介绍 Language-Model-Based Evolutionary Optimizer (LEO)**：论文介绍了 **LEO**，这是一种使用 LLM 进行数值优化的新型基于种群的方法，其表现与 **gradient-based** 和 **gradient-free methods** 相当。
  
  - 然而，LLM 的创造性本质引发了对 *hallucination* 的担忧，需要谨慎管理。
- **应用引发社区讨论**：社区成员对具有推理能力的 LLM 的深层应用表现出兴趣，特别是与工程设计相关的应用。
  
  - 他们渴望就将 LLM 应用于实际工程挑战的影响交换意见。

 

**提到的链接**：[Large Language Model-Based Evolutionary Optimizer: Reasoning with elitism](https://arxiv.org/abs/2403.02054)：**Large Language Models** (LLM) 展示了卓越的推理能力，引发了人们对其作为黑盒优化器应用的兴趣。本文断言 LLM 具备……的能力。

 

---

### **Mozilla AI ▷ #**[**announcements**](https://discord.com/channels/1089876418936180786/1089876419926032396/1295823752277528668) (1 条消息):

> - `AI Stewardship Practice Program`
> - `AI 微证书 (Microcredentialing)`
> - `MaRS Discovery District`

- **AI Stewardship Practice 试点项目**：**MaRS Discovery District** 正在提供少量免费名额，用于试点针对 AI 从业者的 **AI Stewardship Practice Program**。
  
  - 该项目为**研究人员**、**企业家**、**教育工作者**以及其他希望对 AI 演变产生积极影响的人员提供微证书；[更多信息请点击此处](https://programs.techstewardship.com/)。
- **加入 AI 课程试点的机会**：对课程试点感兴趣的参与者可以在帖子中回复，有机会获得价值 **500 CAD** 的席位。
  
  - 席位将根据回复顺序进行分配，直到名额满员，鼓励潜在参与者快速响应。

 

---

---

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