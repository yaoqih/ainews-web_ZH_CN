---
companies:
- nous-research
- cursor-ai
- gdm
- george-hotz
- agibot
- unitree
- eth-zurich
- disney
- uc-san-diego
- ai21-labs
- luma-labs
- ideogram
- nvidia
- mistral-ai
- meta-ai-fair
date: '2024-08-27T00:09:52.018820Z'
description: '以下是该文本的中文翻译：


  **Nous Research** 发布了 **DisTrO**，这是一种新型优化器，可将 GPU 间的通信需求大幅降低 1000 到 10,000 倍，从而实现在慢速网络上的高效训练，为
  **Google DeepMind 的 DiLoCo** 提供了替代方案。**Cursor AI** 因一名 8 岁用户的使用而走红，并宣布了新一轮融资，联合主持人
  Aman 也回归了他们的播客。**George Hotz** 正式发售 **tinybox**。在机器人领域，**智元机器人 (AGIBOT)** 展示了 5
  款新型人形机器人并计划开源；**宇树科技 (Unitree)** 展示了其售价 1.6 万美元、即将量产的 G1 人形机器人。**苏黎世联邦理工学院 (ETH
  Zurich)** 和**迪士尼**开发了一套 AI 系统，可以根据文本或图像生成基于物理的机器人动作。**加州大学圣地亚哥分校 (UC San Diego)**
  发布了 **ACE**，这是一个用于控制多个机器人的开源远程操控系统。**AI21 Labs** 推出了 **Jamba 1.5**，这是一款具有 25.6 万上下文长度和宽松许可协议的多语言模型。**Luma
  Labs** 发布了 **Dream Machine 1.5**，提升了文本生成视频的效果。**Ideogram** 推出了其文本生成图像模型的 **v2**
  版本，具有近乎完美的文本生成能力。**英伟达 (Nvidia)** 和 **Mistral** 发布了 **Mistral-NeMo-Minitron 8B**，这款小模型在
  Open LLM 排行榜上的表现优于 **Mistral-7B** 和 **llama-3-8b**。'
id: bd83aaf6-a034-4ade-acd4-2426be77d923
models:
- jamba-1.5
- dream-machine-1.5
- ideogram-v2
- mistral-nemo-minitron-8b
- mistral-7b
- llama-3-8b
original_slug: ainews-not-much-happened-this-weekend
people:
- george-hotz
- adcock_brett
- aman
title: 这个周末没发生什么特别的事。
topics:
- distributed-ai
- optimizer
- inter-gpu-communication
- low-latency-training
- open-source
- humanoid-robots
- robotics
- physics-based-motion
- teleoperation
- multilingual-models
- long-context
- text-to-video
- text-to-image
- model-performance
---

<!-- buttondown-editor-mode: plaintext -->**我们的副标题快用完了。**

> 2024年8月23日至8月26日的 AI 新闻。我们为你检查了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 社区（**214** 个频道，**5673** 条消息）。预计为你节省了 **639 分钟** 的阅读时间（以 200wpm 计算）。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 来进行 AINews 讨论！

几条新闻：

- **分布式 AI**：Nous Research [发布了 DisTrO](https://x.com/NousResearch/status/1828121648383566270)，这是他们的新优化器，宣称“在不依赖摊销分析的情况下，将 GPU 间的通信需求降低了 1000 到 10,000 倍，且收敛速度与 AdamW+All-Reduce 相当。这使得在低带宽、异构网络硬件的互联网环境下，也能进行大型神经网络的低延迟训练。” —— 这是 [GDM 的 DiLoCo 的一个不错替代方案](https://x.com/apyh__/status/1828139739842850879)。
- 随着[一段 8 岁小孩使用 Cursor 的视频走红](https://x.com/rickyrobinett/status/1825581674870055189)以及他们的[融资公告](https://x.com/cursor_ai/status/1826656532072923219)，Cursor AI 的热度如滚雪球般增长。他们的第一次播客采访是在[整整一年前](https://www.latent.space/p/cursor)，而 Aman 在 [6 月份回归担任联合主持人](https://www.latent.space/p/iclr-2024-benchmarks-agents)。
- George Hotz 的 [tinybox 正式开售！](https://x.com/realGeorgeHotz/status/1828197925874463166)。

既然新闻流比较平淡，不如给 [Box AI 的新测试版](https://shortclick.link/8dpebr)提提反馈？

---

**[由 Box 赞助] 你正在用 AI 构建产品。Box 也是。想象一下，如果你使用 Box 的组件来构建你的产品。实际上，不用想象，[亲自在 Box AI Developer Zone 尝试一下吧。](https://shortclick.link/8dpebr)**

> Swyx 的评论：感谢 Box（通过 [Freeman & Forrest](https://x.com/editingemily/status/1790739130516676788)）在今年 8 月对 AI News 的支持 ([1](https://shortclick.link/23g92m), [2](https://shortclick.link/tndo68), [3](https://shortclick.link/5lxgsv))！

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录** 和 **频道摘要** 已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

> 所有摘要由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 与机器人技术进展**

- **人形机器人**：[@adcock_brett](https://twitter.com/adcock_brett/status/1827738485479899257) 报道称，中国机器人初创公司 AGIBOT（智元机器人）发布了 5 款新型人形机器人并计划开源，每款机器人针对从家务到工业操作的不同任务而设计。此外，[@adcock_brett](https://twitter.com/adcock_brett/status/1827738507885879382) 提到另一家中国机器人制造商 Unitree（宇树科技）展示了其新款 G1 人形机器人，据称已接近“量产”，价格为 16,000 美元。

- **AI 生成动作**：[@adcock_brett](https://twitter.com/adcock_brett/status/1827738530321207444) 指出，苏黎世联邦理工学院（ETH Zurich）和 Disney 开发了一个 AI 系统，能够根据文本或图像输入为机器人生成基于物理的动作，该系统采用两阶段方法，从大型数据集中学习动作的潜空间表示（latent representations）。

- **远程操作控制系统**：[@adcock_brett](https://twitter.com/adcock_brett/status/1827738552806805934) 强调了加州大学圣地亚哥分校（UC San Diego）发布的 ACE，这是一个低成本、跨平台的远程操作控制系统，允许研究人员同时精确控制多个机器人。该系统已完全开源。

**AI 模型与工具**

- **Jamba 1.5**：[@adcock_brett](https://twitter.com/adcock_brett/status/1827738732469882945) 报道称，AI21 Labs 推出了 Jamba 1.5，这是一个新的多语言 AI 模型系列，具有 256,000 的上下文长度，在其同类尺寸模型中长上下文处理速度快 2.5 倍，并对小型组织提供宽松的许可协议。该模型拥有完全开放的权重。

- **Dream Machine 1.5**：[@adcock_brett](https://twitter.com/adcock_brett/status/1827738620045779442) 提到 Luma Labs 发布了 Dream Machine 1.5，这是其 AI 视频生成模型的升级版，支持更高质量的文本生成视频、更智能的提示词理解以及改进的图像生成视频能力。

- **Ideogram v2**：[@adcock_brett](https://twitter.com/adcock_brett/status/1827738665017045431) 指出 Ideogram 发布了其文本生成图像 AI 模型的 v2 版本，其特色在于能够生成近乎完美的文本，为缩略图、海报和表情包等图像生成场景开辟了新的用例。

- **Mistral-NeMo-Minitron 8B**：[@adcock_brett](https://twitter.com/adcock_brett/status/1827738754888409144) 报道称，Nvidia 和 Mistral 发布了 Mistral-NeMo-Minitron 8B，这是一个可以在笔记本电脑和 PC 上运行的小型模型，在 Open LLM 排行榜上的表现优于 Mistral-7B 和 Meta-LLama 3.1-8B。

**AI 应用与研究**

- **自主销售 Agent**：[@adcock_brett](https://twitter.com/adcock_brett/status/1827738597690126484) 提到了 Salesforce 推出的两款全自主 AI 驱动的销售 Agent：Einstein SDR Agent 和 Einstein Sales Coach Agent，它们能够与入站线索（inbound leads）进行互动，并实时辅导销售人员。

- **Amazon 的 AI 助手**：[@adcock_brett](https://twitter.com/adcock_brett/status/1827738709975863639) 分享了 Andy Jassy 关于 Q 的更新。Q 是 Amazon 用于软件开发的 AI 助手，据估计它已节省了相当于 4,500 个开发者年的工作量。

- **Neuralink 进展**：[@adcock_brett](https://twitter.com/adcock_brett/status/1827738642506256600) 报道了 Neuralink 在第二位人类患者 Alex 身上取得的进展。Alex 展示了仅使用脑机接口（BCI）玩《反恐精英 2》（Counter-Strike 2）的惊人控制力，并在第一天就打破了之前 BCI 光标控制的世界纪录。

**AI 开发与工具**

- **Git Commit 消息生成器**：[@karpathy](https://twitter.com/karpathy/status/1827810695658029262) 分享了一个实用工具，该工具利用 @simonw 的 `llm` CLI 工具，根据已暂存更改（staged changes）的 git diff 自动生成 git commit 消息。

- **代码编辑的投机解码（Speculative Decoding）**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1827832981056016645) 重点介绍了 Cursor.ai 的博客文章，内容涉及修改 diff 格式以及使用微调后的 Llama 70B 进行投机编辑，实现了比 GPT-4o 快 4-5 倍的速度，并推动了准确率/延迟曲线上的 Pareto frontier。

- **VoiceCraft**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1827812046206840932) 提到了一款令人印象深刻的工具，用于野外环境下的零样本（zero-shot）语音编辑和文本转语音（TTS），仅需几秒钟的参考音频即可克隆未见过的声音。

**AI 研究与框架**

- **GraphRAG**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1827835310257955195) 讨论了一篇关于 GraphRAG 技术的综述论文，该技术将图结构数据与语言模型相结合，比纯文本方法更有效地捕获复杂的关联知识。

- **iLoRA**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1827846782346219764) 重点介绍了一篇提出 Instance-wise LoRA (iLoRA) 的论文，该技术通过将 LoRA 与 Mixture of Experts (MoE) 集成来个性化 LLM 推荐，从而提高序列推荐系统的准确性。

- **RAGLAB**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1827843916034404430) 提到了 RAGLAB，这是一个用于标准化 RAG 研究的开源库，采用模块化设计，以便在不同算法之间进行公平比较。

**AI 伦理与监管**

- **加州 SB 1047 法案**：[@labenz](https://twitter.com/labenz/status/1827744915775766739) 对 SB 1047 法案发表了评论，指出只有少数模型会被覆盖（仅限成本超过 1 亿美元的模型），且开发者已经在自愿进行广泛的安全测试。

**梗与幽默**

- [@AravSrinivas](https://twitter.com/AravSrinivas/status/1827762376906842375) 分享了一个与 AI 相关的幽默 T 恤标语。
- [@vikhyatk](https://twitter.com/vikhyatk/status/1827819090582597828) 开玩笑地建议关闭语法高亮，以成为一名更好的开发者。
- [@abacaj](https://twitter.com/abacaj/status/1827775671730418120) 幽默地评论了其信息流中 Cursor 相关内容的泛滥。

本摘要捕捉了所提供推文中关于 AI 和机器人技术的关键进展、研究和讨论，重点关注与 AI 工程师和开发者相关的方面。

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1. 本地 LLM 推理的硬件优化**

- **2000-3000 美元是否足以构建本地编程 AI 系统？** ([Score: 55, Comments: 102](https://reddit.com//r/LocalLLaMA/comments/1f0xg89/is_23000_enough_to_build_a_local_coding_ai_system/))：一位用户询问是否能以 **2,000 到 3,000 美元**的预算构建一个**本地编程 AI 系统**，旨在复制 Cursor 和 Anthropic 等商业编程助手的性能。他们将**速度置于准确度之上**，认为准确度可以通过更好的 prompting 或重试来提高，并专门询问 **Mac Studio** 是否足以满足此用途。

- **考虑不要使用 Mac...** ([Score: 178, Comments: 149](https://reddit.com//r/LocalLLaMA/comments/1f0w0bn/consider_not_using_a_mac/))：该帖子比较了 **M2 Mac Studio** 和**搭载 2080ti GPU 的 AMD 组装机**之间的 **LLM 推理性能**。**Nvidia 配置**的性能显著优于 Mac，处理 **32k context** 仅需 **25 秒**，而 Mac 需要 **260 秒**，同时使用的 VRAM 更少（**10GB** 对比 **30GB**），并支持带有 **flash attention** 和 **quant k,v** 的 **64k context**。此外，Nvidia 设备在 **context shifting** 和回复生成方面表现出更稳定的性能。

**主题 2. 长上下文 LLM 生成技术的进展**

- **[LongWriter：释放长上下文 LLM 的 10,000+ 字生成能力](https://github.com/THUDM/LongWriter)** ([Score: 74, Comments: 15](https://reddit.com//r/LocalLLaMA/comments/1f13o9n/longwriter_unleashing_10000_word_generation_from/))：**LongWriter** 是一种使**长上下文大语言模型 (LLMs)** 能够生成超过 **10,000 字**连贯文本的技术。该方法包括将生成过程分解为可管理的区块，使用高达 **32,000 tokens** 的**上下文窗口 (context windows)**，并采用**递归摘要 (recursive summarization)** 和**动态提示 (dynamic prompting)** 等策略来保持各章节之间的一致性。这种方法允许创建长篇叙事、综合报告和其他长文本内容，同时在整个生成文本中保持主题连贯性和逻辑流。

**主题 3. Anthropic 在 AI 监管上的争议立场**

- **[你认为 Anthropic 在对抗开源方面比 OpenAI 更糟糕吗？在我看来似乎确实如此。这封信似乎暗示他们实际上向参议员 Wienner 提议了该法案……我真的很喜欢我的 OSS LLMs……](https://i.redd.it/ybjoof8z1xkd1.png)** ([Score: 226, Comments: 111](https://reddit.com//r/LocalLLaMA/comments/1f1d4gh/do_you_think_anthropic_is_worse_than_oai_with/))：与 OpenAI 相比，Anthropic 似乎在对抗开源 LLM 方面采取了更激进的立场，甚至可能**向参议员 Wienner 提议立法**。帖子作者对这种感知到的立场表示担忧，表达了对**开源语言模型 (open-source language models)** 的偏好。这场辩论突显了 **AI 安全监管**与 **LLM 开发创新**（特别是在开源领域）之间的紧张关系。
  - 拟议的**加州 SB1047 法案**要求对大型 AI 模型进行**安全测试**并内置**“自毁开关” (kill switch)**。批评者认为这可能会扼杀**创新**和**开源开发**，可能导致 AI 的进步流出美国。
  - 用户对**监管俘获 (regulatory capture)** 表示担忧，暗示 Anthropic 可能会推动立法以维持其市场地位。一些人将其与过去监管**汽车**、**飞机**和**电子游戏**等新技术的尝试进行了比较。
  - 讨论强调了在数学模型中实现“自毁开关”的挑战，以及**创新转移到其他地方**的可能性，特别是转移到像**中国**这样可能不太倾向于监管 AI 开发的国家。

**主题 4. 新兴中国 LLM 挑战西方模型**

- **对 GLM-9B 印象深刻（他们对该模型介绍很少）** ([Score: 54, Comments: 12](https://reddit.com//r/LocalLLaMA/comments/1f184lk/impressed_by_glm9b_they_say_little_about_the_model/))：帖子作者对 **GLM4-9B 模型**的性能表示惊讶，声称它在回答质量方面**远超 Gemma 2 9B 和 Llama 3.1 8B**。他们分享了 [Hugging Face 上的模型链接](https://huggingface.co/THUDM/glm-4-9b-chat)，并询问其他人对该模型的看法和经验，并指出关于该模型的讨论似乎很少。

## AI Reddit 综合回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**机器人与 AI 硬件**

- **Disney Research 的面部模仿机器人**：由 [Disney Research 开发的一款机器人](https://www.reddit.com/r/singularity/comments/1f0yr4r/this_robot_from_disney_research_can_imitate_human/) 可以**模仿人类的面部动作**，特别是眨眼和细微的头部运动。
- **2024 北京世界机器人大会**：该[会议展示了各种机器人技术](https://www.reddit.com/r/singularity/comments/1f0zlw6/beijing_world_robotics_conference_2024/)，突显了该领域的最新进展。

**生物技术与食品科技**

- **实验室培育肉成本持平**：一项[研究表明，实验室培育肉的成本可以与 USDA 有机鸡肉持平](https://www.reddit.com/r/singularity/comments/1f0x6pq/labgrown_meat_can_cost_the_same_as_usda_organic/)，这标志着培育肉在经济可行性方面取得了进展。

**AI 模型开发**

- **模型大小与智能的权衡**：关于[模型大小与智能之间折中的讨论](https://www.reddit.com/r/singularity/comments/1f158p9/in_the_race_to_bottom_for_price_significant_model/)表明，与 GPT-4 等早期版本相比，近期的模型经过了显著的蒸馏（distilled），这可能会影响它们的能力。
- **感知到的 AI 进展放缓**：[r/OpenAI 的用户正在讨论感知到的 AI 进步放缓](https://www.reddit.com/r/OpenAI/comments/1f0v3pe/anyone_else_feel_like_ai_improvement_has_really/)，并指出近期的发展不如一年前那样令人印象深刻。

---

# AI Discord 摘要回顾

> 由 Claude 3.5 Sonnet 生成的摘要之摘要的总结


**1. LLM 进展与基准测试**

- **Grok-2 攀升至 LMSYS 排行榜**：**xAI 的 Grok-2** 在 [LMSYS Leaderboard](https://lmsys.org/leaderboard) 上产生了重大影响，超越了 **GPT-4o** (5月版)，并凭借超过 6,000 张社区投票与最新的 **Gemini** 并列第二。
   - 同样来自 xAI 的 **Grok-mini** 获得了第 5 名，尤其在 **Math**（第 1 名）方面表现出色，并在 **Hard Prompts**、**Coding** 和 **Instruction-following** 类别中排名第 2。
- **1.5-Pints LLM：质量重于数量**：一款名为 **"1.5-Pints"** 的新型紧凑型 LLM 仅用 9 天时间，通过 570 亿 token 的精选数据集预训练完成，在 **MT-Bench** 基准测试中表现优于 **Apple 的 OpenELM** 和 **Microsoft 的 Phi**。
   - 该模型采用了 **修改后的 Mistral tokenizer** 和 **Llama-2 架构**，优先考虑“教科书式”的内容，以增强推理和逻辑演绎能力。
  


**2. LLM 优化技术**

- **DisTrO：革命性的分布式优化**：Nous Research 发布了关于 **DisTrO** 的初步报告，这是一系列分布式优化器，在不依赖摊销分析的情况下，将 GPU 间的通信需求降低了 **1000 倍至 10,000 倍**。
   - DisTrO 在收敛速度上与 **AdamW+All-Reduce** 持平，有可能彻底改变大规模 LLM 训练。完整报告可在 [GitHub](https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf) 上查阅。
- **LIGER Kernel 提升 LLM 训练效率**：用于 LLM 训练的新型 **LIGER** kernel 取得了令人印象深刻的结果，与传统方法相比，可节省 **25% VRAM** 并减少 **33% 训练时间**。
   - 虽然 LIGER 主要为多 GPU 设置设计，但预计即使在单 GPU 训练场景下也能提供改进，这引起了 AI 社区的关注。
- **Sparse-Marlin 加速矩阵乘法**：**Sparse-Marlin** 是一种新型 GPU 优化 kernel，已集成到 **vllm_project** 中，在 NVIDIA GPU (Ampere/Ada) 上针对 4-bit 量化权重的矩阵乘法实现了 **5.3 倍的加速**。
   - 这一进步在 Batch Size 高达 32 的情况下仍能保持效率，并利用了 **2:4 sparsity**，有可能彻底改变大语言模型的推理速度。
  


**3. 开源 AI 发展**

- **Zed AI：开源编程伴侣**：**Zed AI** 作为一款开源的 AI 驱动代码编辑器发布，为 AI 辅助编程提供了强大的界面，支持 **Claude-3.5** 等模型并集成了 **Ollama**。
   - 该编辑器具有专为快速文本转换设计的新 Anthropic API，第一个月免费提供，将其定位为 Cursor 等专有选项的强力替代品。
- **Apple 的 ML-Superposition Prompting 正式开源**：Apple 已将其 **ML-Superposition Prompting** 项目开源，现已在 [GitHub](https://github.com/apple/ml-superposition-prompting) 上可用，旨在推进机器学习中的 Prompting 技术。
   - 这一发布在 AI 社区引起了轰动，可能为从事语言模型和 Prompt Engineering 的研究人员及开发人员提供新的工具和方法论。
- **Tinybox：面向 AI 爱好者的开源硬件**：与 **tinygrad** 框架相关的开源硬件项目 **Tinybox** 已通过 [tiny shop](https://tinycorp.myshopify.com) 向公众发售。
   - 目前产能约为每天 4 台，积压订单为 60 台，Tinybox 代表了人们对用于 AI 开发和研究的可获取开源硬件日益增长的兴趣。
  


**4. AI 行业与社区动态**

- **AI Engineer 伦敦见面会宣布**：首届 **AI Engineer London Meetup** 定于 9 月 12 日举行，演讲嘉宾包括 Maxime LaBonne、Rovio Sc、Martins Bruveris 和 Chris Bull，消息由 [@dctanner](https://x.com/dctanner/status/1827071893448618453) 发布。
   - 该活动受 @swyx 的 AI Engineer World's Fair 启发，旨在汇聚伦敦的 AI 爱好者和专业人士进行知识分享和交流。
- **Together AI 调整价格结构**：**Together AI** 宣布提高其 Serverless Reference 端点的价格，**Llama-3 8B** 从每百万 token 0.20 美元上调至 0.40 美元，**Llama-3 70B** 从每百万 token 0.90 美元上调至 1.80 美元，自 2024 年 9 月 1 日起生效。
   - 虽然这些变化影响了 Serverless Reference 端点，但 Together AI 的 Turbo 和 Lite 定价保持不变，具体见其 [定价页面](https://www.together.ai/pricing)。

---

# 第 1 部分：高层级 Discord 摘要

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DisTrO 的分布式优化突破**：Nous Research 发布了关于 **DisTrO** 的初步报告，展示了在无需摊销分析的情况下，将 GPU 间通信减少了 **1000 倍至 10,000 倍**，且收敛速度与 **AdamW+All-Reduce** 持平。完整报告可在 [GitHub](https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf) 上查看。
   - 分布式优化器的这一进展标志着 LLM 训练的重大进步，团队对即将发布的模型代码和算法表示期待。
- **Hermes 2.5 性能超越 Hermes 2**：在整合了[代码指令示例](https://link.to.examples)后，**Hermes 2.5** 展示了优于 **Hermes 2** 的性能，在 MMLU 基准测试中获得了 **52.3** 分，而 Hermes 2 为 **34.5** 分。
   - 这一实质性的提升为工程师之间的 LLM 性能评估设定了新标准。
- **1.5-Pints LLM 取得快速训练成功**：新的 **1.5-Pints** 模型仅用 **9 天** 就完成了预训练，在模拟人类判断的 **MT-Bench** 上超越了 **Apple 的 OpenELM** 和 **Microsoft 的 Phi**。该模型使用了专注于逻辑推理的 **570 亿 token** 精选数据集。
   - 该模型采用了修改后的 **Mistral tokenizer** 和 **Llama-2 architecture**，展示了 LLM 领域高效的训练方法论。
- **Sparse-Marlin 加速矩阵乘法**：在 **vllm_project** 中引入 **Sparse-Marlin**，通过使用 **4-bit quantized weights**，在 NVIDIA GPU 上实现了 **5.3 倍** 的矩阵乘法加速。
   - 这种针对 GPU 优化的内核可能会显著提升处理大型模型用户的性能。
- **探索 Whisper Diarization 实现**：一位用户询问了关于实现 **Whisper diarization**（说话人日志）的问题，并分享了一个使用 **Whisper v3** 的脚本，寻求识别说话人变化的方法。
   - 目前的努力方向是整合 diarization 功能，以简化音频处理并提高输出保真度。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 指控 LinkedIn 盗用代码**：Unsloth 频道的成员断言 **LinkedIn** 抄袭了他们项目的代码，特别是在其 **Triton kernel** 实现中。他们指出 [LinkedIn 的 Liger-Kernel 仓库](https://github.com/linkedin/Liger-Kernel)和 [Ollama](https://ollama.com/unclemusclez/qwen2-unsloth) 上的一篇帖子是证据。
   - 指控指出 LinkedIn 将其内核与 Unsloth 的工作进行基准对比，暗示其缺乏对原始项目的公平贡献。
- **性能对比：Unsloth vs. Hugging Face**：讨论强调 **Unsloth** 在速度和内存效率上优于 **Hugging Face** 等平台，尽管目前缺乏对 **8-bit models** 的支持。这使 Unsloth 处于竞争地位，但仍有明显的局限性。
   - 成员们表示，虽然 Unsloth 展示了令人印象深刻的训练和推理时间，但全面的模型支持对于更广泛的采用仍然至关重要。
- **Liger Kernel 加速 LLM 训练**：一位成员透露，新的 **Liger Kernel** 可以将 LLM 训练速度提高 **20%**，同时减少 **60%** 的内存使用，正如 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1eznkml/liger_kernel_one_line_to_make_llm_training_20/)中所讨论的那样。
   - 该内核利用 **Triton** 开发，在优化训练时间方面展现了潜力，因其潜在应用而受到关注。
- **多语言模型微调的挑战**：成员们分享了关于训练 **Arabic**（阿拉伯语）和 **Persian**（波斯语）等语言模型的见解，强调了专业数据集和预训练的重要性。其中一个建议是利用 **Persian Wikipedia** 以获得更好的模型效果。
   - 成员们对 **Llama-3** 中这些语言的适当支持表示担忧，指出这可能阻碍多语言能力的进步。
- **Replete-LLM V2 发布，功能增强**：**Replete-LLM-V2-Llama-3.1-8b** 正式发布，重点提升了推理和代码性能，该模型在 **Replete-AI/The_Living_AI_Dataset** 上进行训练，旨在嵌入“**爱与共情**”的概念。
   - 该模型的有效性在很大程度上依赖于其系统提示（system prompts），这对于优化其信息处理能力至关重要。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **澄清 Stable Diffusion Online 的状态**：成员们质疑 [Stable Diffusion Online](https://stabledifffusion.com) 是官方网站还是独立于 Stability AI 运营。
   - 这一询问揭示了社区内部对于与 Stable Diffusion 相关的各种平台的公信力及其关联性仍存在持续的困惑。
- **ComfyUI vs. ForgeUI - 选择你的工具！**：有人建议，那些没有充分利用 **ComfyUI** 全部功能的开发者应该考虑切换到 **ForgeUI** 以获得更精简的体验。
   - 这场辩论凸显了关于优化图像扩散设置工作流的持续讨论。
- **深入探讨 SD 图像放大方案**：成员们讨论了各种图像放大技术，包括 **Ultimate SD Upscale** 和 **Tiled Diffusion**，特别提到了 '4x-NomosWebPhoto-atd' 模型与 SUPIR 的结合。
   - 这些讨论强调了社区通过先进方法提升图像质量的努力。
- **Noise Injection：提升图像质量的秘诀**：一位成员详细阐述了 A1111/Forge 中的 'Noise Injection'，解释了它在改进图像放大效果中的作用。
   - 这种技术作为一种潜在的增强策略引起了关注，能够带来更高质量的输出。
- **Flux 的困境 - 过拟合问题**：讨论集中在 **Flux** 的过拟合挑战上，特别是在奇幻相关的输出中，导致生成的图像多样性降低。
   - 这一探索引发了关于 **Flux** 需要如何调整以平衡创造力与多样性的担忧。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hermes 2.5 表现优于 Hermes 2**：在添加了 [代码指令示例](https://link.to.examples) 后，**Hermes 2.5** 在各项基准测试中的表现似乎优于 **Hermes 2**。
   - Hermes 2 在 MMLU 基准测试中得分为 **34.5**，而 Hermes 2.5 得分为 **52.3**。
- **Mistral 难以扩展至 8k 以上**：成员们表示，如果不进行持续预训练，**Mistral** 无法扩展到 8k 以上，且 [这是一个已知问题](https://link.to.issue)。
   - 他们指出，*mergekit* 和 *frankenMoE finetuning* 的进一步工作是性能突破的下一个前沿。
- **模型合并策略讨论**：一位成员建议将 **UltraChat** 与基础 **Mistral** 之间的差异应用到 **Mistral-Yarn**，作为一种潜在的合并策略。
   - 其他人表示怀疑，但该成员保持乐观，并引用了以往在他们所谓的“诅咒模型合并”（cursed model merging）中的成功尝试。
- **模型量化与蒸馏要点**：强调了 **Model Quantization** 和 **Model Distillation** 对于将机器学习模型投入生产环境的重要性。
   - 成员们一致认为，这些技术是实现超越本地训练的有效部署的基础。
- **TinyLlama 的快速成功**：**TinyLlama**（一个类似于 Tau LLM 的模型）仅用 9 天就成功完成训练，并在 MTBench 上超越了 Apple 的 OpenELM 和 Microsoft 的 Phi。
   - 训练代码和模型权重已在 GitHub 和 HuggingFace 上公开。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **模型缩放遭遇边际收益递减**：讨论强调了**模型缩放 (model scaling)** 的边际收益递减现象，特别是在 **Llama 3.1** 和 **Claude 3.5 Sonnet** 中，性能提升滞后于计算能力的增加。
   - 参与者强调，要使 AI 的规模超越单纯的数据和计算增长，必须取得创新性的突破。
- **辩论 AI 意识**：哲学讨论围绕着当前的 **LLMs**（如 GPT）是否可以被视为具有意识展开，考虑到它们缺乏有机体验，并且可能遵循与人类意识不同的规律。
   - 参与者还探讨了对自由意志的影响，认为 AI 系统表现出的决策是基于内部逻辑而非真正的意志。
- **有效地分享 GPTs**：成员们表示有兴趣更好地追踪分享的 **GPTs** 及其在社区中的效用，并质疑如何评估其有效性。
   - 对话包括关于共享输出功能的易用性担忧，以及对追踪使用案例的可能改进。
- **利用品牌身份创建自定义 GPTs**：有人建议利用 **custom GPT builder** 来打造符合特定品牌身份的 GPTs 用于内容创作，并使用 GPT store 获取系统提示词 (system prompts)。
   - 重点在于通过 API 集成中的自定义提示词来增强品牌一致性。
- **OpenAI API 的订阅模式**：用户探讨了平台如何管理 OpenAI API 的**订阅模式**，例如利用基于 Token 定价的月度计划。
   - Chatbase 被引用为讨论案例，表明迫切需要明确实施策略。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 创作者社区上线**：Perplexity AI 与 [Kale](https://www.kalecard.com/t/perplexity.ai?ref=perplexity.ai_discord) 合作推出了 **Perplexity Creator Community**，允许创作者通过参与视频内容赚取现金。
   - 该计划鼓励用户根据自己的时间表发布内容，同时根据视频的传播范围产生收入。
- **API 速率限制引发不满**：来自 Newcode.ai 的 Maged Helmy 迫切要求为其集成增加 API 速率限制 (rate limits)，此前他在等待 Perplexity 团队回复的过程中已经耗时六个月。
   - Newcode.ai 拥有超过 3,500 名用户，其运营依赖于这些增强的限制来维持性能。
- **GPT-4o 主导编程，Claude 3.5 Sonnet 擅长知识获取**：讨论强调 **GPT-4o** 在 STEM 任务中表现优异，而 **Claude 3.5 Sonnet** 在知识检索方面表现出色，特别是针对编程相关的查询。
   - 用户注意到 Claude 在诗歌和叙事方面表现不佳，使得 GPT-4o 成为处理更广泛任务的首选。
- **Perplexity 中的图像生成问题**：用户报告了图像生成的重大挑战，特别是在使用 Dalle3 时，尝试生成往往导致线程失败。
   - 反馈表明图像生成过程可能需要改进，因为某些结果未能达到用户预期。
- **Perplexity Pro 的 LinkedIn 订阅优惠**：Perplexity AI 正在为 LinkedIn Premium 订阅者提供一年的免费 **Perplexity Pro**，尽管欧盟的一些用户在可用性方面遇到了问题。
   - Pro 版本提供无限次搜索，并允许访问 **GPT-4 Omni** 和 **Claude 3.5 Sonnet** 等高级 AI 模型。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Grok-2 和 Grok-mini 登上排行榜！**：**xAI 的 Grok-2** 和 **Grok-mini** 凭借超过 **6000** 张社区投票强势冲入 [LMSYS Leaderboard](https://lmsys.org/leaderboard)！值得注意的是，Grok-2 与 **Gemini** 并列第二，而 Grok-mini 在 **Math**（数学）领域位居第一，并在 **Hard Prompts**、**Coding** 和 **Instruction-following** 方面排名第二。
   - 成员们对 Grok-2 击败 **GPT-4o** (May 版本) 表示欢呼，这预示着排行榜动态和用户偏好的潜在转变。
- **数据库故障已解决**：最近的一次 **database change**（数据库变更）导致了约 **2 分钟的停机**，但问题现已解决，服务已恢复正常。
   - 团队对造成的不便表示歉意，并强调了对可靠运行时间（uptime）的重视。
- **Mistral 无法扩展至 8k 以上**：针对 **Mistral** 的担忧浮现，据报道如果不进行持续预训练（pretraining），它无法扩展到 **8k** 以上，这被强调为一个[已知问题](https://link.to.issue)。
   - 建议包括探索 **mergekit** 和 **frankenMoE finetuning** 技术以提升性能。
- **Claude 3.5 Sonnet 再次下线**：用户报告 **Claude 3.5 Sonnet** 正面临间歇性停机，严重影响了其可用性。
   - 虽然 **Haiku** 运行正常，但 **Hermes 3.5** 等其他模型也持续出现问题，暗示了更广泛的系统不稳定性。
- **OpenRouter API Key 查询**：用户正在讨论如何将自己的 **API keys** 与 **OpenRouter** 集成，以及显示的 Token 定价是否包含了 **OpenRouter fee** 在内的总成本。
   - 澄清表明，Token 价格以 **OpenRouter credits** 列出，相关费用在充值 credits 时计算。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **OMI 模型能力讨论**：成员们讨论了 **OMI 参与者** 从零开始创建 AI 模型的能力，但未能分享具体的意见或评估。
   - *未达成实质性结论，参与者们仍在思考其中涉及的能力水平。*
- **LLM 重复失败模式**：讨论了 LLM 中一种常见的失败模式，即模型会重复短语，这可能与模型的过度量化（over-quantization）和最小化损失（minimizing loss）有关。
   - 参与者假设某些条件可能会触发这种 **looping behavior**（循环行为），强调需要进一步调查。
- **Anthropic 可解释性成本挑战**：关于为 Llama 8B 或 Mistral 等数据密集型且计算密集型的模型复现 [Anthropic 的可解释性工作 (interpretability work)](https://arxiv.org/abs/2406.04093) 的**成本**问题被提出。
   - 成员们注意到成本高昂，但未提供具体数字，强调了在这些项目中资源分配的重要性。
- **Sparse MoE 的 GPU 利用率优势**：一位成员提到 **Sparse MoE** 如何利用 GPU 稀疏性进行高效的分布式训练，允许将专家（experts）分布在多个进程中。
   - 这种策略可以增强分布式推理（inference）场景下的性能，突显了可扩展性方法。
- **GNNs 与进化学习方法**：一位成员将 **GNNs** 的演进与 positional embeddings（位置嵌入）进行了比较，建议未来的进展可能涉及从 latent representations（潜表征）中推断嵌入。
   - 这一观点暗示了改进图结构中表征学习（representation learning）的新路径。

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Hermes 2.5 表现优于 Hermes 2**：在添加了[代码指令示例](https://link.to.examples)后，**Hermes 2.5** 在基准测试中的表现优于 **Hermes 2**，在 MMLU 上的得分分别为 **52.3** 和 **34.5**。
   - 这一改进突显了新一代模型迭代中近期优化措施的有效性。
- **Mistral 受限于 8k 限制**：如果不进行持续的预训练，**Mistral** 无法扩展到 8k 以上的上下文长度，这被认为是其当前设置中的一个重大限制，并且[这是一个已知问题](https://link.to.issue)。
   - 目前正在讨论探索如 *mergekit* 和 *frankenMoE finetuning* 等解决方案，以突破这些界限。
- **剖析 BERTopic 的实用性**：关于 **BERTopic**（一种强大的主题建模工具）的讨论浮出水面，成员们分享了他们关于[数据可视化](https://github.com/YoungPhlo/visualizing-datasets-ai-engineer-fall-2023)的项目。
   - 对话再次确认了其生成可解释主题的端到端能力，激发了对其聚类效果的好奇心。
- **呼吁在 Open Empathic 项目上进行协作**：有人请求扩大 **Open Empathic** 项目的类别，强调了社区贡献的必要性。
   - 成员们被引导至一个 [YouTube 教程](https://youtu.be/GZqYr8_Q7DE)，以获取有关如何添加他们喜爱场景的指导，同时还提供了 [OpenEmpathic 项目](https://dct.openempathic.ai/)的链接。
- **伦敦 AI 工程师见面会启动**：受 AI Engineer World's Fair 启发，新宣布的 AI 工程师见面会定于 9 月 12 日在伦敦举行，已确认有四位知名演讲者。
   - 鼓励感兴趣的参与者在[此处](https://x.com/dctanner/status/1827071893448618453?s=46)注册，这注定将是一场极具吸引力的聚会。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinybox 销售启动！**：**Tinybox** 工厂现已开足马力，即将向公众开放销售。感兴趣的买家可以查看 [tiny 商店](https://tinycorp.myshopify.com) 了解购买选项。
   - **Tinybox** 目前已售罄，生产能力约为 **每天 4 台**，目前积压了 **60** 台订单。
- **对 E-graph 性能的担忧**：成员们表示，在处理大型搜索空间时，**e-graph 重写**落后于当前的 SAT 求解器，突显了潜在的性能瓶颈。
   - 建议进行持续改进，以匹配成熟的 SAT 求解技术中可见的效率。
- **探索 Tinygrad 和 AMD GPU**：讨论了在 **Tinybox** 中使用 **AMD GPU** 的情况，并提到了 AMD 最近收购了 **Silo AI** 及其在 AMD 硬件上训练 LLM 的进展。
   - 社区成员发表了看法，思考了有效整合 AMD 能力的可行性和优势。
- **Tinygrad 与 Torch 在 BERT 预训练中的对比**：一位用户表示有兴趣与 **Tinygrad** 合作预训练一个大型 **BERT** 模型，并为该任务提供计算资源。
   - 这种协作可能为探索 Tinygrad 和 PyTorch 在大型模型训练方面的性能差异铺平道路。
- **提高训练速度**：一位用户报告称，在 **beautiful_cifar** 示例中通过移除 `.cast(dtypes.default_float)` 调用来调整预处理后，训练速度（GFLOPS）提高了 **25%**。
   - 通过此调整，他们注意到模型现在以 `dtype.float` 处理数据，从而提高了效率。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command-R 模型更新缺乏官方公告**：新的 **Command-R** 模型已经发布，但目前还没有关于其特性（包括价格和上下文窗口）的官方沟通。
   - *用户要求明确信息*，因为许多人急于了解微调选项并解决未回答的问题。
- **Durov 大胆的公民身份举动**：**Telegram** 创始人 Pavel Durov 最近获得了法国公民身份，目前正在法国面临审判，引发了辩论。
   - 有人猜测，在与北约关系紧张之际，他的目标是通过*战略性入狱*来获得国际媒体的关注。
- **Cohere 为聊天机器人提供免费试用**：一位用户探索了使用 **Cohere** 的免费试用来构建 **Rasa** 聊天机器人，希望能找到 OpenAI 服务的免费替代方案。
   - 回复显示了用户在应对 AI 部署相关成本时对经济实惠方案的兴趣。
- **Cohere API 速率限制收紧**：新报告显示，即使在文档记录的速率下，用户也会遇到“请求过多”错误，因为限制已更改为所有 API Key 总计每分钟 1,000 次调用。
   - Cohere 澄清说，这意味着每个用户组织有整体的 **1,000次/分钟限制**，这会影响同时使用多个 Key 的用户。
- **关于 Rerank 3 定价的说明**：用户询问了 **Rerank 3** 的定价，特别是 1,000 次搜索 2 美元是否涵盖了真实的 API 调用。
   - Cohere 确认，每次搜索最多处理 100 个文档，根据文档限制，1,000 次搜索总计 **409,600,000 tokens**。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Create Llama 推出提取模板**：[Create Llama](https://t.co/G95ReAuRS6) 工具现在具有结构化提取模板，增强了用户体验。
   - 这一新增功能旨在简化数据提取过程，同时保持准确性和效率。
- **GraphRAG 教程系列启动**：关于构建 [GraphRAG](https://t.co/7fLocjRvdN) 的新分步教程系列已经开始，重点关注核心组件的实现。
   - 第一段视频强调了如何使用 LLM 通过内存实现来提取实体和关系。
- **数据孤岛阻碍企业级 LLM 开发**：企业级 LLM 开发中数据孤岛的挑战依然存在，强调了无缝身份验证管理的必要性。
   - LlamaIndex 正在研究可行的解决方案，以整合团队间分散的知识。
- **LLM 自动化简报创建**：LlamaIndex 简报已转为使用 LLM 自动创建内容，此前这是一项手动且耗时的工作。
   - 这一转变体现了 LLM 在提高定期内容摘要效率方面的能力。
- **RAG-a-thon 黑客松即将到来**：第二届 [RAG-a-thon](https://t.co/IFvyW5QB6r) 黑客松与 Pinecone 合作，定于 10 月 11 日至 13 日在帕洛阿尔托举行，提供超过 7,000 美元的现金奖励。
   - 活动将在 500 Global VC 办公室举行，欢迎参与者展示创新解决方案。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **编译函数输出与 Eager Mode 不同**：一名成员提出了一个问题，即为什么在相同种子下，编译后的函数可能会产生与非编译版本不同的输出。这归因于 RNG 使用的不同：编译代码中使用 **Triton 的 RNG**，而 Eager Mode 中使用 PyTorch 的 RNG，这可能受到 In-place operation（原地操作）行为的影响。
   - In-place operation（如 `scatter_`）在编译代码中可能会产生意外结果，导致更高的内存消耗和变化的输出。
- **Cudagraphs 可能会消耗更多内存**：讨论了利用 **cudagraphs** 进行调试的问题，指出它们有预分配缓冲区的潜力。然而，它们也可能导致内存使用量增加，这可能并非所愿。
   - 这意味着使用 cudagraphs 存在权衡，需要根据内存开销来权衡其收益。
- **FP16 作为节省内存的策略**：建议在推理中使用 **FP16** 代替 FP32 以降低内存使用，特别是在不支持 BF16 的硬件上。据报道，这种改变的方法缓解了显存不足（OOM）问题。
   - 尽管有了这些改进，编译和非编译输出之间的差异仍然是一个令人担忧的问题。
- **探索编译内核中的数值差异**：即使优化了内存使用，剩余的输出差异也可能源于编译内核固有的数值差异。这指向了即使输入相同也可能存在的潜在计算变异。
   - 参与者对这些数值差异表示担忧，强调了在编译代码评估中需要进一步考虑的领域。

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain 文档加载：图像提取简化**：LangChain 社区包中 `PyPDFLoader` 的 `extract_images=True` 参数允许从 PDF 文档中无缝提取图像，为 LLM 处理丰富了文本上下文。
   - 这对于需要结合文本数据进行图像分析的应用特别有用，扩展了 LangChain 的功能。
- **LLMChain 对比 LCEL：灵活性与优化**：`LLMChain` 提供了一种简单直接的链式模型和 Prompt 方法，而 `LCEL` 为复杂任务提供了更高的定制化和灵活性。
   - 虽然 `LLMChain` 仍是大多数场景的最优选择，但模块化设计的爱好者可能更倾向于 `LCEL` 引入的精细控制。
- **排查 PostgresSaver 错误**：用户在使用 LangGraph 的 `PostgresSaver` 时遇到了与元组索引（tuple indexing）相关的 `TypeError`，这表明在数据类型处理方面可能存在潜在问题。
   - 需要进一步调查以澄清元组访问方法，并解决开发者遇到的这一持续挑战。
- **GenAI 在数据科学中日益增长的作用**：一场讨论强调了 Generative AI 在数据科学领域的新兴作用，特别是在自动化代码生成和数据 Pipeline 构建方面。
   - 尽管对其局限性存在怀疑，但参与者承认了数据科学与 GenAI 进步之间的关键整合。
- **RAG 协作：寻求合作伙伴**：一位成员分享了使用 LangChain 开发检索增强生成（RAG）聊天机器人的意图，希望能为该项目找到合作伙伴。
   - 讨论中提到了爬虫和 RAG 组件方面的挑战，强调了这一技术领域的协作机会。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **GPT-4 微调对比 Mistral：评价褒贬不一**：一位用户声称，与 **Mistral** 相比，微调 **GPT-4** 的效果“有点糟糕”，尽管他们使用的训练数据更少。
   - 这引发了关于两种模型在实际应用中相对性能的讨论。
- **lm-eval-harness：让基准测试变得简单**：成员们讨论了 *lm-eval-harness* 框架，认为它通过提供简便的任务集成简化了 Benchmark 的创建。
   - 一位用户强调了他们对生成基准测试问题的研究，并在其最近关于 LLM 评估的 [MCQs](https://arxiv.org/abs/2406.02394) 论文中进行了分享。
- **LIGER 展示了令人印象深刻的训练效率**：**LIGER** 内核承诺为 **LLM 训练** 节省 **25% VRAM** 和 **33% 训练时间**，这让急于测试其能力的用户感到兴奋。
   - 然而，正如一位用户所指出的，对其在 **单 GPU 训练** 中的有效性仍存有疑问。
- **对 Phi-3-medium-128k-instruct 训练配置感到好奇**：一位用户寻求 **Phi-3-medium-128k-instruct** 模型的训练配置，强调了共享设置的必要性。
   - 另一位用户质疑了特定配置设置（modules_to_save）中的 Token 训练，并引用了外部消息以寻求澄清。
- **探索数据清洗（Data Curation）技术**：一位用户深入探讨了 **数据清洗**，询问是否涉及模型提供评分，类似于 **LLM-Judge 系统**。
   - 对话表明，人们对采用模型评估进行数据清洗的方法（类似于现有系统）很感兴趣。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 的 Jitting 行为解析**：在脚本模式下运行 `mojo main.mojo` 时，会发生 **jitting**，这就是为什么 **global variables** 在该模式下的行为与 `mojo build main.mojo` 编译模式下不同的原因。
   - 这一澄清有助于用户理解在切换模式时内存管理的复杂性。
- **社区关注开发进度**：由于暑假或问题堆积，**Max** 和 **Mojo** 的博客文章和更新速度似乎有所放缓，这引发了社区的担忧。
   - 成员们正在寻求澄清，了解这是否会影响未来的发布和项目。
- **GPU 支持成为焦点**：社区强烈推动 Mojo 的 **GPU support**，并期望未来的版本能解决此问题，从而可能将 **Magic** 移出 alpha 阶段。
   - 成员们正热切期待下一个重大版本，将社区讨论与这些功能的进展保持一致。
- **Modverse 42 发布时间表明确**：成员询问上周为何没有发布 **Modverse 42**，得知发布周期为 **1-3 周**，具体取决于项目量。
   - 随着内容流趋于稳定，目前的每周标签可能会进行调整。
- **Mojo 的 Struct 参数和 UnsafePointer 详情**：在 Mojo 中使用 struct 时出现了由于 **variadic parameters** 在定义结构体之外未正确参数化而导致的错误。
   - 关于使用 **UnsafePointer** 的讨论强调了所有权需要显式管理，突显了 Mojo 中引用管理的复杂性。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **OpenInterpreter 配置文件的自定义路径？**：一位成员询问是否可以设置 *OpenInterpreter 配置文件的自定义路径*，但开发者表示该功能目前尚不可用，尽管未来可能会加入。
   - 一旦实现，该功能将增强用户的灵活性。
- **Windows 上的 OpenInterpreter --vision 标志功能**：关于 Windows 上 `--vision` 标志的咨询结论是其应能正常工作，并鼓励在专用频道报告任何问题。
   - 进一步的测试可能会为不同环境下的兼容性提供重要见解。
- **预装版 OpenInterpreter 需求激增**：开发者分享称，由于需求量大，*prebuilt OpenInterpreter* 设备的预订已关闭，显示出强烈的兴趣。
   - 用户需要等待销售恢复，这突显了技术社区对该产品的参与度。
- **品牌指南仍缺失**：有人请求品牌指南文档，但成员确认目前尚无此类文档。
   - 该咨询与围绕项目可访问性和设计考量的讨论相关联。
- **Zed AI：开源编程伴侣**：Zed AI 为 AI 辅助编程提供了一个*酷炫的界面*，支持 **Claude-3.5** 和 Ollama 等模型，并由首月免费的新 **Anthropic API** 提供增强支持。
   - 作为 Cursor 等专有选项的强力替代品，它正受到关注，促进了更广泛的开源开发。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Apple 的 Superposition Prompting 项目启动**：成员们对 Apple 的新项目 **ML-Superposition Prompting** 表示兴奋，该项目已在 [GitHub](https://github.com/apple/ml-superposition-prompting) 上线，旨在精炼 ML 中的提示技术。
   - 目前，社区讨论集中在对该项目的初步反响上，尚无进一步的技术见解。
- **OpenAI 引入类型化输出 (Typed Outputs)**：讨论引发了关于 OpenAI **typed outputs** 新功能的关注，重点是 JSON 格式结构化输出的验证，并提到了 **Outlines**、**Guardrails** 等项目。
   - 成员们链接了相关的 GitHub 仓库，展示了用于管理结构化输出格式的各种库。
- **处理 DSPy 输出错误**：一位成员报告了 DSPy 中关于“尝试获取正确输出格式时重试次数过多”的 `ValueError`，这在使用类型化预测器（typed predictors）时出现，归因于输出填充文本。
   - 另一位用户提供了见解并链接到了现有的 [GitHub issue](https://github.com/stanfordnlp/dspy/issues/1001)，以澄清这个常见的 JSON 输出解析问题。
- **探索德语 ColBERT 训练**：一位用户寻求关于构建德语 ColBERT 模型训练数据的指导，提议使用类似 ColBERTv2 的 32 路三元组（32-way triplets）格式。
   - 他们建议的数据结构格式包括 `raw_query = [(query, (positive_passage, positive_score), [(negative_passage1, negative_score1), ...])]`，并正在寻求对其适用性的验证。

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Hugging Face Leaderboard 与网站同步**：由于最近的一个 pull request，[Hugging Face Leaderboard](https://huggingface.co/spaces/bigscience/bfcl) 现在与网站排行榜同步，并征求团队成员的反馈。
   - *鼓励任何关注此项更改的人员分享建议。*
- **关注 BFCL V2-Live 数据集的准确性**：关于如何计算 [BFCL V2-Live 数据集](https://github.com/ShishirPatil/gorilla/discussions/602)的整体准确率正在进行讨论，该数据集包含 **2,251 个问题-函数-答案对**。
   - 该数据集包括 **258 个简单、7 个多个、16 个链式和 14 个多阶段函数调用**，引发了关于准确评估方法的疑问。
- **关于向 BFCL 添加模型的咨询**：一位新成员表示有兴趣向 BFCL 添加模型，询问了非开源上传的流程以及具有多个组件的模型评估。
   - *正在寻求有关在与 BFCL 集成时保持模型完整性的细节。*
- **Gorilla Leaderboard 解释**：针对 [Gorilla Leaderboard 文档](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#contributing)中“准备可执行测试对 (prepare the executable test pairs)”这一短语提出了疑问。
   - 文档澄清说，鼓励用户向排行榜贡献可执行测试对，以促进评估方法的协作改进。
- **为函数调用训练 LLM**：[Gorilla Leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#contributing) 用于通过标准化基准测试来训练和评估 LLM 的函数调用能力。
   - 该框架允许对各种模型进行比较，从而增强性能评估。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Anthropic 的机械可解释性 (Mechanistic Interpretability) 成本**：一位用户对为 **Llama 8b** 和 **Mistral** 等模型运行 **Anthropic 的机械可解释性**相关的费用提出了质疑，并指出缺乏开源替代方案。
   - 他们强调了对限制是由于数据密集型还是**计算密集型 (compute-heavy)** 的担忧，并寻求对其他影响因素的澄清。
- **即将举行的 AI Engineer London Meetup**：请在日历上标记 **9 月 12 日**的 **AI Engineer London Meetup**，届时将展示来自 Maxime LaBonne 和 Rovio Sc 等人物的见解。
   - [Damien C. Tanner 的推文](https://x.com/dctanner/status/1827071893448618453)中分享的细节显示，该活动旨在将 Swyx 的 **AI Engineer World's Fair** 的一部分带到英国。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Romain Huet 接管 OpenAI DevRel**：**OpenAI** 的新开发者关系 (DevRel) 负责人是 **Romain Huet**，他在 **2023 年 7 月**加入后在 [Twitter](https://x.com/romainhuet) 上确认了自己的职位。
   - Huet 的任命是在前任负责人 **Logan** 离职后进行的，这表明 OpenAI 的开发者推广工作正在进行集中的领导层过渡。
- **Logan 的平稳过渡**：**Logan** 于 **2023 年 7 月**离开 **OpenAI**，其继任者 **Romain Huet** 确认了这一消息。
   - Huet 指出过渡很顺利，表明组织内部已经建立了领导层变更的协议。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **AI Engineer London Meetup 拉开帷幕**：首届 **AI Engineer London Meetup** 定于 **9 月 12 日**晚上举行，共有四位演讲者：**Maxime La Bonne**、**Roviosc**、**Martins Bruveris** 和 **Chris Bull**。注册详情可以在[这里](https://x.com/dctanner/status/1827071893448618453)找到。
   - 该活动旨在成为由 **Damien C. Tanner** 主办的 **AI Engineer World's Fair** 的一部分，重点展示 AI 工程师之间的活跃讨论。
- **强调 AI Engineer World's Fair 的影响**：这次伦敦 Meetup 从 **AI Engineer World's Fair** 中汲取灵感，目标是为 AI 讨论创造一个协作氛围。该活动汇集了令人兴奋的演讲者阵容，分享见解和经验。
   - 该 Meetup 由 **Damien C. Tanner** 主办，是 AI 爱好者建立联系并参与该领域前沿话题的社区空间。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Hamel 的出席问题**：一位用户询问 **Hamel** 是否参加关于 **LLM Finetuning** 的讨论，表达了对其专业知识的兴趣。
   - 这一互动突显了社区对知名贡献者在 LLM 优化方面见解的期待。
- **Hamel 不在场**：遗憾的是，**Hamel** 在询问时并不在场，这暗示错过了一次讨论机会。
   - 社区成员表示希望他能在未来的会议中参与并分享他的见解。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **CUDA Hackathon 登陆旧金山**：准备好参加 **9 月 21 日**在旧金山举行的 **CUDA Hackathon**，届时你可以与 **NVIDIA 工程师**并肩作战，解决现实世界的 CUDA 挑战。
   - 这是一个与专家交流并参与创新 **accelerated computing** 项目的绝佳机会。
- **深入探索加速计算**：该活动将探索 **accelerated computing**，利用 NVIDIA 的并行计算平台来优化 GPU 应用。
   - 参与者将获得 NVIDIA 资源和工程师的亲自指导，以构建和完善 CUDA 应用程序。



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Together AI 向用户发布涨价通知**：自 2024 年 9 月 1 日起，**Together API** 的 Serverless Reference 端点针对 **Llama-3 8B** 和 **70B** 模型的价格将上涨，其中 8B 模型从每百万 tokens **$0.20** 增加到 **$0.40**。
   - **70B** 模型的价格将从每百万 tokens **$0.90** 跳升至 **$1.80**，反映出显著的上调。
- **Turbo 和 Lite 价格保持稳定**：虽然 Serverless 端点价格在上涨，但 **Together API** 的 **Turbo** 和 **Lite** 定价保持不变，正如 [Together Pricing Page](https://www.together.ai/pricing)（最后更新于 2024 年 7 月 18 日）所确认的那样。
   - 这使得用户在整体价格变动中，避免了这些端点的价格上涨。
- **OpenAI 降价，让 Together AI 显得处境尴尬**：与 **Together AI** 即将到来的涨价形成鲜明对比，一位成员指出 **OpenAI** 最近降低了 **GPT-4O-Mini** 的价格，引发了关于定价策略的讨论。
   - 这一转变让人们对 Together AI 在竞争对手降价时选择涨价的决定感到意外。
- **融资困境引发涨价猜测**：由于成员们讨论了当前定价策略的可持续性，有人猜测 Together AI 可能会因为融资问题而将价格翻倍。
   - 他们提到 **4-bit** 和 **8-bit** 模型的价格目前应保持不变，但未来潜藏着变动的可能。



---


**Mozilla AI Discord** 没有新消息。如果该频道沉寂时间过长，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道沉寂时间过长，请告知我们，我们将将其移除。


---

# 第 2 部分：频道详细摘要与链接


{% if medium == 'web' %}




### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1277685656788668416)** (1 条消息): 

> - `DisTrO`
> - `Distributed Optimization`
> - `LLM Training`
> - `Inter-GPU Communication`
> - `AdamW` 


- **Nous Research 发布 DisTrO 报告**：Nous Research 发布了关于 **DisTrO** 的初步报告。DisTrO 是一个分布式优化器系列，在不依赖摊销分析的情况下，将 GPU 间的通信需求降低了 **1000 倍至 10,000 倍**，并且在收敛速度上与 **AdamW+All-Reduce** 持平。
   - 该报告可在 GitHub 上获取：[DisTrO/A_Preliminary_Report_on_DisTrO.pdf](https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf)。
- **DisTrO：LLM 训练的游戏规则改变者**：这是提高 LLM 训练效率探索中的一项重大进展，因为 DisTrO 显著降低了 GPU 间的通信需求。
   - 团队很高兴分享这些早期结果，并计划在不久的将来发布代码、完整算法和论文。
- **关于 DisTrO 的开放合作**：该项目由 Nous Research 的研究人员和工程师共同努力完成，他们正邀请各方对该项目进行协作。
   - 任何有兴趣贡献的人都可以联系 recruiting@nousresearch.com。



**提到的链接**：<a href="https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf">DisTrO/A_Preliminary_Report_on_DisTrO.pdf at main · NousResearch/DisTrO</a>：互联网上的分布式训练。通过创建 GitHub 账户为 NousResearch/DisTrO 的开发做出贡献。

  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1276617216464519189)** (786 messages🔥🔥🔥): 

> - `Hermes 2`
> - `Mistral struggles`
> - `Model Merging`
> - `Open Empathic`
> - `GPTs Agents` 


- **Hermes 2.5 表现优于 Hermes 2**：在添加了[代码指令示例](https://link.to.examples)后，**Hermes 2.5** 在各种基准测试中的表现似乎优于 **Hermes 2**。
   - Hermes 2 在 MMLU 基准测试中得分为 **34.5**，而 Hermes 2.5 得分为 **52.3**。
- **Mistral 在扩展超过 8k 时遇到困难**：成员们表示，如果不进行持续预训练，**Mistral** 无法扩展到 8k 以上，且[这是一个已知问题](https://link.to.issue)。
   - 他们指出，针对 *mergekit* 和 *frankenMoE finetuning* 的进一步工作是性能提升的下一个前沿。
- **关于模型合并策略的讨论**：一位成员建议将 **UltraChat** 和基础 **Mistral** 之间的差异应用到 **Mistral-Yarn**，作为一种潜在的合并策略。
   - 其他人表示怀疑，但该成员保持乐观，并引用了过去在他们所谓的“诅咒式模型合并” (cursed model merging) 方面的成功尝试。
- **Open Empathic 项目寻求援助**：一位成员呼吁帮助扩大 **Open Empathic** 项目的类别，特别是低端类别。
   - 他们分享了一个 [Open Empathic 发布与教程的 YouTube 视频](https://youtu.be/GZqYr8_Q7DE)，指导用户贡献他们喜欢的 YouTube 视频电影场景，以及 [OpenEmpathic 项目本身](https://dct.openempathic.ai/)的链接。
- **GPTs Agents 在初始训练后无法学习**：一位成员对 GPTs Agents 在初始训练后无法从提供的额外信息中学习表示担忧。
   - 另一位成员消除了这种误解，解释说[上传的文件被保存为“知识”文件](https://link.to/openai-docs)，供 Agent 在需要时引用，但**它们不会持续修改 Agent 的基础知识**。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://n8python.github.io/VoidInterface/.">The Void</a>: 未找到描述</li><li><a href="https://x.com/NousResearch/status/1828121648383566270">Nous Research (@NousResearch) 的推文</a>: 如果你可以利用世界上所有的算力来训练一个共享的开源 AI 模型会怎样？初步报告：https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO...</li><li><a href="https://x.com/nearcyan/status/1827074097790194123">near (@nearcyan) 的推文</a>: 有人需要 2,000 块带 IB 的 H100 吗？截止到 12 月 14 日，价格公道。请私信！</li><li><a href="https://tenor.com/view/big-if-true-big-if-true-gif-18577099">Big If GIF - Big If True - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://arxiv.org/abs/2106.11257">大规模安全分布式训练</a>: 深度学习的许多领域都受益于在公共数据上训练的越来越大的神经网络，NLP 和计算机视觉的预训练模型就是如此。训练此类模型需要...</li><li><a href="https://youtube.com/playlist?list=PLAJnaovHtaFQFUX5kp3d1UYmaUH_Ux8OL&si=loJ9miYPiKiXuT1M">AGI-24</a>: 完全致力于最新 AGI 研究的首要会议。在 AI 领域及其他领域，人们越来越认识到实现的门槛...</li><li><a href="https://spectrum.ieee.org/eleutherai-openai-not-open-enough">EleutherAI: 当 OpenAI 不够开放时</a>: EleutherAI 是一个松散的计算机科学家协会，其最新成果是 GPT-NeoX-20B，一个拥有 200 亿参数的预训练语言模型。如果你不知道那是什么，可以把它想象成 OpenAI 的 GPT-3...</li><li><a href="https://tenor.com/view/donald-trump-yuge-huge-gif-14073397">Donald Trump Yuge GIF - Donald Trump Yuge Huge - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://docs.google.com/document/u/0/d/1OjbjU5AOz4Ftn9xHQrX3oFQGhQ6RDUuXQipnQ9gn6tU/mobilebasic">Rombodawg 的 SHARED Continuous Finetuning</a>: 未找到描述</li><li><a href="https://tenor.com/view/skeleton-eyebrows-achmed-puppet-jeffdunham-gif-11732886">Skeleton Eyebrows GIF - Skeleton Eyebrows Achmed - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/that-thing-is-so-big-nufo-that-thing-is-massive-that-thing-is-huge-ginormous-gif-25538935">That Thing Is So Big Nufo GIF - That Thing Is So Big Nufo That Thing Is Massive - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/hud_zah/status/1827057785995141558">HudZah ⁂ (@hud_zah) 的推文</a>: 在几周内，我在卧室里建造了一个核聚变反应堆——而且零硬件经验。秘诀？Claude sonnet 3.5 + projects。下面是过程的一瞥</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1f0o0m0/how_to_use_any_ai_on_huggingface_on_your_phone/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/collaborative-training">互联网上的深度学习：协作训练语言模型</a>: 未找到描述</li><li><a href="https://training-transformers-together.github.io/">共同训练庞大的神经网络</a>: 一个 NeurIPS'21 的演示，解释了如何与多个合作者共同训练大型模型。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ew7kwu/llama31storm8b_has_arrived_a_new_8b_parameter_llm/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/">Hugging Face – 博客</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2306.17453">Pollen: 通过资源感知客户端放置进行高吞吐量联邦学习模拟</a>: 联邦学习 (FL) 是一种以隐私为中心的机器学习范式，它直接在边缘设备上协作训练模型。模拟在 FL 的采用中起着至关重要的作用，有助于开发新...</li><li><a href="https://youtu.be/QwNoFOUiSiE?si=zd7YiLpNplLqr5sJ">2023 年 Microsoft Excel 世界锦标赛决赛亮点</a>: 3 小时 2023 年 Microsoft Excel 世界锦标赛决赛直播的主要亮点。在此观看完整直播：https://www.youtube.com/live/UDGdPE_...</li><li><a href="https://github.com/bigscience-workshop/petals">GitHub - bigscience-workshop/petals: 🌸 以 BitTorrent 方式在家里运行 LLM。微调和推理速度比卸载快达 10 倍</a>: 🌸 以 BitTorrent 方式在家里运行 LLM。微调和推理速度比卸载快达 10 倍 - bigscience-workshop/petals</li><li><a href="https://youtu.be/7ZXPWTdThAA">Nous Research</a>: 未找到描述</li><li><a href="https://zenodo.org/records/13370693?token=">新前沿：揭示现实的统一维度</a>: 新黎明：理解超越传统物理学的宇宙。在广阔的知识领域中，人类长期以来一直依赖经典物理学的熟悉锚点来理解宇宙。...</li><li><a href="https://github.com/arcee-a">

i/mergekit">GitHub - arcee-ai/mergekit: Tools for merging pretrained large language models.</a>: 用于合并预训练大语言模型（LLM）的工具。 - arcee-ai/mergekit</li><li><a href="https://zenodo.org/records/13370693?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImM0ZmZhNzNjLWEwOGQtNGRjMy05N2I4LWNmZWNlZGRhNDc1NiIsImRhdGEiOnt9LCJyYW5kb20iOiJlZTkzODc0NDBhMjkyNTMxYzMxZGRhYThjYTJhMGQ5ZSJ9.L2MqhoD1xKMeTC9zmIHHmFBoLj3G3jpY8REYIufs8vzdOwSZuUantHcdtiyP9tV-d-MS0XAYveIbrUvqrgJx_A">A New Frontier: Unveiling the Unified Dimensions of Reality</a>: 新前沿：揭示现实的统一维度：新曙光：理解超越传统物理学的宇宙。在广袤的知识领域中，人类长期以来一直依赖经典物理学的熟悉锚点来理解宇宙。...</li><li><a href="https://huggingface.co/Replete-AI/Replete-LLM-V2-Llama-3.1-8b">Replete-AI/Replete-LLM-V2-Llama-3.1-8b · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/rombodawg/Try_out_Replete-LLM-V2-Llama-3.1-8b">Try Out Replete-LLM-V2-Llama-3.1-8b - a Hugging Face Space by rombodawg</a>: 未找到描述</li><li><a href="https://huggingface.co/bartowski/Replete-LLM-V2-Llama-3.1-8b-exl2">bartowski/Replete-LLM-V2-Llama-3.1-8b-exl2 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/bartowski/Replete-LLM-V2-Llama-3.1-8b-GGUF">bartowski/Replete-LLM-V2-Llama-3.1-8b-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness">GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1276620195619934239)** (25 messages🔥): 

> - `Co-Writer`
> - `Discord Bots`
> - `LLM Model`
> - `LLM Quantization`
> - `MoE` 


- **Co-Writer Bot**: 一位成员询问了创建模仿其朋友的 Discord Bot 的资源，但被告知目前没有相关指南。
   - 另一位成员建议使用 Shapes Inc AI Bot Maker，它允许让 Bot 的行为像朋友一样，但不允许自定义训练。
- **大语言模型 (LLM) 量化详解**: 一位成员询问了同一量化级别下 `_S`、`_M` 和 `_L` 模型之间的区别，并以 [Hermes 3.1 70B GGUF](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-70B-GGUF/tree/main) 为例。
   - 另一位成员解释说，不同的后缀指的是各种量化的混合方式，并非每个权重都采用相同的量化，这会影响模型的整体大小和性能。
- **Sparse Mixture of Experts (MoE) 模型**: 一位成员询问了在 LLM 中使用 Sparse Mixture of Experts (MoE) 模型的优缺点。
   - 他们获知 MoE 主要由提供 API 的大型实验室使用，以降低计算成本。虽然前景广阔，但目前在相同 Token 训练量下，其表现不如 Dense 模型。
- **Google 的 One Million Experts (MoE) 论文**: 一位成员分享了 Google 最近关于 **One Million Experts MoE** 的论文，该论文声称该模型**优于 Dense Feed Forward Networks**。
   - [该论文](https://www.clioapp.ai/research/million-moe) 描述了 **MoE** 的工作原理：通过拥有许多较小的神经网络（称为 **Experts**），每个网络专注于特定的任务或领域。与具有 Dense Feedforward Networks 的传统 Transformer 模型不同，该模型会选择最适合处理特定输入部分的 Expert。
- **在 SmoLLM 上训练 LoRA**: 一位成员分享了他们在 **SmoLLM** 上进行 **LoRA 训练** 的困扰，一个 **330M 参数模型** 的 Loss 达到了 **4.0**。
   - 另一位成员建议这可能是一个糟糕的 Loss，并为 Go 语言初学者推荐了一个 **Markov chain** 实现，并链接到了 **GitHub**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.clioapp.ai/research/million-moe">Mixture of a Million Experts | Clio AI Insights</a>: Google Deepmind 引入了导致 GPT-4 诞生的 Mixture of Experts。这是另一篇 Google Deepmind 的论文，提出了“如果我们将此扩展到一百万个 Expert 会怎样？”的问题。</li><li><a href="https://github.com/mb-14/gomarkov">GitHub - mb-14/gomarkov: Markov chains in golang</a>: golang 中的 Markov chains。欢迎在 GitHub 上为 mb-14/gomarkov 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1276825527692562432)** (8 messages🔥): 

> - `Pints LLM`
> - `Medical AI Research`
> - `LLM Benchmarking`
> - `LLM Training Efficiency`
> - `Multimodal LLMs` 


- **Pints LLM 表现优于 OpenELM 和 Phi**：一个紧凑型 LLM **"1.5-Pints"**，通过使用 570 亿 token 的精选数据集，仅用 9 天时间完成预训练，在模拟人类判断的基准测试 **MT-Bench** 上表现优于 **Apple 的 OpenELM** 和 **Microsoft 的 Phi**。
   - **Pints** 团队在训练中使用了**修改后的 Mistral tokenizer** 和 **Llama-2 架构**，优先考虑用于推理和逻辑推演的“教科书式”内容。
- **医疗 AI 研究论文**：频道中讨论了分享医疗 AI 研究论文的话题。
   - 一位用户提供了 2024 年 8 月 17 日至 8 月 24 日期间的顶级医疗 AI 研究论文列表，其中包括“医疗多模态 LLMs 的 Jailbreak”和“LLaVA-Surg：多模态 LLM 手术助手”等主题。
- **使用 MT-Bench 进行 LLM 基准测试**：**MT-Bench** 基准测试用于评估 LLM 作为指令遵循助手的性能。
   - **MT-Bench** 模拟人类判断，是跨不同用例比较 LLM 的宝贵工具。
- **9 天内完成高效 LLM 训练**：**Pints** 团队通过在短短 9 天内预训练其模型，实现了令人印象深刻的训练效率。
   - 这得益于精心策划的数据集和有效的训练方法，展示了在更短时间内开发高质量 LLM 的潜力。
- **医学中的多模态 LLMs**：医疗 AI 研究列表中讨论的几篇论文强调了**多模态 LLMs** 在医疗保健中的应用。
   - 这些模型利用文本和视觉数据来执行手术辅助和临床试验过渡预测等任务，证明了多模态方法在医学领域的潜力。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/OpenlifesciAI/status/1827442651810918509">来自 Open Life Science AI (@OpenlifesciAI) 的推文</a>: 医疗 AI 上周动态：顶级研究论文/模型 🏅（2024 年 8 月 17 日 - 8 月 24 日） - 医疗多模态 LLMs 的 Jailbreak - LLMs 并非零样本生物医学推理者 - RuleAlign 框架：Aligni...</li><li><a href="https://github.com/pints-ai/1.5-Pints">GitHub - Pints-AI/1.5-Pints: 使用高质量数据在 9 天内预训练的紧凑型 LLM</a>: A compact LLM pretrained in 9 days by using high quality data - Pints-AI/1.5-Pints</li><li><a href="https://www.arxiv.org/abs/2408.03506">1.5-Pints 技术报告：预训练以天计而非以月计 —— 你的语言模型在高质量数据中茁壮成长</a>: 本文介绍了一种计算效率高的语言模型预训练方法——"1.5-Pints"——仅用 9 天时间，同时作为指令遵循助手，其表现优于最先进的模型...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1276822589716561940)** (3 条消息): 

> - `Pints-AI 1.5`
> - `Sparse-Marlin`
> - `Text Generation Webui` 


- **Pints-AI 1.5 性能超越 Apple OpenELM 和 Microsoft Phi**：由 **Pints-AI** 在短短 **9 天**内训练完成的新 **LLM** **Pints-AI 1.5**，其表现优于 **Apple OpenELM** 和 **Microsoft Phi**。 
   - 你可以在这里找到训练代码：[Pints-AI 1.5](https://github.com/pints-ai/1.5-Pints)。
- **Sparse-Marlin 通过 4-bit 量化权重加速矩阵乘法**：一个名为 **Sparse-Marlin** 的新型 GPU 优化算子（kernel）已集成到 **vllm_project** 中。
   - Sparse-Marlin 通过使用 **4-bit 量化权重**和 **2:4 稀疏性 (sparsity)**，在 **NVIDIA GPUs** (Ampere/Ada) 上实现了 **5.3 倍的加速**，并在 **batch size** 高达 32 时仍能保持高效。
- **Text Generation Webui DRY 功能更新**：**Text Generation Webui** 新增了一个名为 **DRY** 的功能，旨在防止模型在输出中重复短语。
   - DRY 是一种现代的重复惩罚机制，有助于降低模型逐字重复输入中已出现短语的可能性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/neuralmagic/status/1827285549058572566">来自 Neural Magic (@neuralmagic) 的推文</a>: Sparse-Marlin 来了，并已集成到 @vllm_project！这个 GPU 优化算子通过 4-bit 量化权重和 2:4 稀疏性加速矩阵乘法，在 NVIDIA GPU 上实现了 5.3 倍的加速...</li><li><a href="https://github.com/pints-ai/1.5-Pints">GitHub - Pints-AI/1.5-Pints: 使用高质量数据在 9 天内预训练的紧凑型 LLM</a>: 使用高质量数据在 9 天内预训练的紧凑型 LLM - Pints-AI/1.5-Pints</li><li><a href="https://github.com/oobabooga/text-generation-webui/pull/5677">DRY: 一种现代重复惩罚机制，可可靠地防止 p-e-w 的循环 · Pull Request #5677 · oobabooga/text-generation-webui</a>: 循环是一种不理想的行为，即模型逐字重复输入中先前出现的短语。它影响大多数模型，并且在使用截断采样器时会加剧....
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1276825527692562432)** (8 messages🔥): 

> - `1.5-Pints LLM`
> - `Medical AI Research`
> - `LLaVA-Surg`
> - `HIBOU`
> - `RuleAlign` 


- **1.5-Pints LLM 性能超越 OpenELM 和 Phi**：一个名为 **1.5-Pints** 的新模型在 **9 天**内完成了预训练，作为指令遵循助手，其表现优于 **Apple OpenELM** 和 **Microsoft Phi**。
   - 该模型在包含 **570 亿 token** 的精选数据集上进行训练，优先考虑**说明性和类教科书内容**，以辅助**推理和逻辑演绎**。该模型使用了修改后的 **Mistral tokenizer** 和 **Llama-2 architecture**，以实现更广泛的兼容性。
- **分享医疗 AI 研究论文**：一位用户询问是否允许在频道中分享**医疗 AI 研究论文**，并得到了肯定的答复，只要与 AI 相关即可。
   - 该用户分享了 8 月 17 日至 24 日期间顶尖的**医疗 AI 研究论文和模型**列表，包括 **LLaVA-Surg**、**HIBOU**、**RuleAlign**、**CTP-LLM**、**MEDCO**、**Clinical Insights**、**Federated Knowledge Injection** 以及一个用于临床多步诊断的 **EMR Dataset**。
- **LLaVA-Surg：多模态手术助手**：重点介绍的顶尖医疗 AI 研究论文之一是 **LLaVA-Surg**，这是一个**多模态 LLM 手术助手**。
   - 该论文探讨了在手术室中使用 LLM 辅助外科医生的潜力，暗示了医疗 AI 的新前沿。
- **HIBOU：病理学基础 Vision Transformer**：另一篇值得关注的论文是 **HIBOU**，这是一个**用于病理学的基础 Vision Transformer**。
   - 该论文专注于使用深度学习技术分析病理图像，有可能彻底改变医疗诊断和治疗。
- **RuleAlign：使 LLM 符合医生规则**：**RuleAlign 框架**旨在使 LLM 与**医生规则**保持一致，以改进医疗决策。
   - 该论文强调了将医学专业知识整合到 AI 系统中对于实现更安全、更有效的医疗保健的重要性。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/OpenlifesciAI/status/1827442651810918509">来自 Open Life Science AI (@OpenlifesciAI) 的推文</a>：上周医疗 AI 动态：顶尖研究论文/模型 🏅（2024 年 8 月 17 日 - 8 月 24 日）- 医疗多模态 LLM 的越狱 - LLM 并非零样本生物医学推理者 - RuleAlign 框架：对齐...</li><li><a href="https://github.com/pints-ai/1.5-Pints">GitHub - Pints-AI/1.5-Pints：通过使用高质量数据在 9 天内预训练的小型 LLM</a>：通过使用高质量数据在 9 天内预训练的小型 LLM - Pints-AI/1.5-Pints</li><li><a href="https://www.arxiv.org/abs/2408.03506">1.5-Pints 技术报告：预训练只需数天而非数月 —— 你的语言模型在高质量数据中茁壮成长</a>：本文介绍了一种计算效率高的预训练语言模型方法 —— “1.5-Pints”，仅用 9 天时间，同时作为指令遵循助手，其表现优于最先进的模型...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1277449679650422835)** (1 messages): 

> - `Whisper Diarization`
> - `Whisper v3` 


- **Whisper Diarization 请求**：一位用户正在寻求有关实现 **Whisper diarization**（说话人日志）的信息，特别是询问是否有人有相关经验并愿意分享脚本。
   - 他们提到目前有一个在本地或通过 **Groq** 使用 **Whisper v3** 的脚本，但正在寻找一种能够识别说话人变更的解决方案。
- **使用 Whisper v3 进行 Diarization**：该用户目前正在使用 **Whisper v3** 脚本进行音频处理。
   - 该脚本可以在本地或 **Groq** 平台上运行，但他们正在寻求集成 diarization 能力的方法，以识别说话人的切换。


  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1276618062891847811)** (512 messages🔥🔥🔥): 

> - `Unsloth`
> - `Ollama`
> - `Llama.cpp`
> - `LinkedIn`
> - `Model Merging`

- **Unsloth 被指控遭 LinkedIn 抄袭**：Unsloth Discord 频道的一些成员指控 LinkedIn 从他们的项目中窃取代码，特别是在其 Triton kernel 实现方面。
   - 他们指出 [LinkedIn 在 GitHub 上的 Liger-Kernel 仓库](https://github.com/linkedin/Liger-Kernel) 以及 [Ollama](https://ollama.com/unclemusclez/qwen2-unsloth) 上的一篇帖子是所谓窃取行为的证据。他们还注意到 LinkedIn 将其 kernel 与 Unsloth 进行了基准测试（benchmarked），但仅限于 kernel 层面，这表明他们可能在使用 Unsloth 的成果而没有回馈社区。 
- **Unsloth 与 Hugging Face 及 LinkedIn 的性能对比**：几位成员讨论了 Unsloth 与 Hugging Face 和 LinkedIn 等其他平台的性能对比，特别是在速度、内存占用和准确性方面。
   - 一些成员指出，虽然 Unsloth 在训练和推理（inference）方面更快、更高效，但与 Hugging Face 和 LinkedIn 不同，它目前还不支持 8-bit 模型。 
- **开源的优缺点**：Unsloth Discord 频道正在热烈讨论开源的本质以及协作与剥削之间的平衡。
   - 一些成员认为开源模式很有价值，因为它允许双向获益，公司会向他们使用的项目回馈贡献；而另一些成员则对某些公司利用开源项目而不提供任何回报表示沮丧。 
- **如何在阿拉伯语、波斯语或其他语言上训练模型**：几位成员讨论了在阿拉伯语和波斯语等语言上训练模型的挑战，强调了需要特定的数据集和持续预训练（pretraining）才能获得良好效果。
   - 一位成员建议可以使用波斯语数据集和波斯语维基百科来预训练 Llama-3 模型，以在波斯语中获得更好的效果；而另一位成员则建议不要将 Llama 3.1 用于波斯语，因为它目前还不支持该语言。
- **使用 Unsloth 进行模型部署和推理**：讨论围绕使用 Unsloth 部署和运行模型展开，特别是在 Hugging Face 和 Fireworks.ai 等平台上。
   - 一位成员询问在将模型部署到 Hugging Face 后如何使用 Unsloth 进行推理（inference），并强调了超时和模型大小限制方面的挑战。另一位成员建议使用 vLLM 进行推理，并建议 Unsloth 的文档应将其作为一个选项列出。 


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.unsloth.ai/">欢迎 | Unsloth 文档</a>：初识 Unsloth？从这里开始！</li><li><a href="https://colab.research.google.com/drive/1weTpKOjBZxZJ5PQ-Ql8i6ptAY2x-FWVA?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/15OyFkGoCImV9dSsewU1wa2JuKB4-mDE_?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://huggingface.co/docs/autotrain/main/en/sentence_transformer">Sentence Transformers</a>：未找到描述</li><li><a href="https://huggingface.co/rahulAkaVector/infosys-business-model-4bit">rahulAkaVector/infosys-business-model-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/installation/pip-install">Pip 安装 | Unsloth 文档</a>：若要通过 Pip 在本地安装 Unsloth，请按照以下步骤操作：</li><li><a href="https://huggingface.co/rahulAkaVector/java_code_generator/tree/main">rahulAkaVector/java_code_generator at main</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://tenor.com/view/snattacatz-gif-25782415">Snattacatz GIF - Snattacatz - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/rahulAkaVector/modelz/tree/main">rahulAkaVector/modelz at main</a>：未找到描述</li><li><a href="https://ui.endpoints.huggingface.co/catalog">模型目录 | Inference Endpoints - Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/microsoft/Phi-3.5-mini-instruct">microsoft/Phi-3.5-mini-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://ollama.com/unclemusclez/qwen2-unsloth">unclemusclez/qwen2-unsloth</a>：快速上手大语言模型。</li><li><a href="https://huggingface.co/docs/trl/main/en/dpo_trainer">DPO Trainer</a>：未找到描述</li><li><a href="https://huggingface.co/ai21labs/AI21-Jamba-1.5-Large#ruler-benchmark---effective-context-length>">ai21labs/AI21-Jamba-1.5-Large · Hugging Face</a>：未找到描述</li><li><a href="https://dontasktoask.com/">别问能不能问，直接问</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ezq84k/llama_31_chat_template">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=rpAtVIZB72U">LLAMA-3.1 🦙：在你的数据上进行微调的最简单方法 🙌</a>：学习如何使用 Unsloth、LoRa 和 QLoRa 技术高效地微调 Llama 3.1 模型。链接：Colab: https://tinyurl.com/bdzxhy5n Unsloth: https:/...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ezq84k/llama_31_chat_template_quirks_chat_notebook/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=XZ3w_jec1v8">Evan Czaplicki 的《编程语言经济学》（Strange Loop 2023）</a>：在开源的神话中，编程语言是由那些看似没有直接经济职能的人创造的。他们只是非常擅长...</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/57">针对 Unsloth 的基准测试 · Issue #57 · linkedin/Liger-Kernel</a>：🚀 功能、动机和推介。嘿，你有没有针对使用类似 Kernel 的 Unsloth 运行过任何基准测试？我想你的项目可以作为支持多 GPU 的直接替代方案。此外……</li><li><a href="https://github.com/linkedin/Liger-Kernel?trk=comments_comments-list_comment-text">GitHub - linkedin/Liger-Kernel：用于 LLM 训练的高效 Triton Kernels</a>：用于 LLM 训练的高效 Triton Kernels。通过在 GitHub 上创建账号来为 linkedin/Liger-Kernel 的开发做出贡献。</li><li><a href="https://huggingface.co/unsloth/Phi-3.5-mini-instruct">unsloth/Phi-3.5-mini-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/TheAITimeline/status/1827467655844131070">来自 The AI Timeline (@TheAITimeline) 的推文</a>：🚨本周热门 AI/ML 研究论文：- Transfusion - To Code, or Not To Code? - Agentic Systems 的自动化设计 - LLM 剪枝与蒸馏实践：Minitron - Hermes 3 技术报告...</li><li><a href="https://github.com/microsoft/T-MAC">GitHub - microsoft/T-MAC：基于查找表的 CPU 低比特 LLM 推理</a>：基于查找表的 CPU 低比特 LLM 推理。通过在 GitHub 上创建账号来为 microsoft/T-MAC 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1276823137387810889)** (14 条消息🔥): 

> - `Liger Kernel`
> - `Triton`
> - `EAGLE`
> - `LLM Training` 


- **Liger Kernel 加速 LLM 训练**：一位成员分享了一个 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1eznkml/liger_kernel_one_line_to_make_llm_training_20/)，声称一种名为 **Liger Kernel** 的新内核可以使 LLM 训练速度提升 **20%**，并减少 **60%** 的显存占用。
   - 另一位成员指出，该内核利用了 **Triton**，在缩短训练时间方面非常有效。
- **Triton 内核调试**：一位成员寻求帮助，调试他们使用 **Triton** 实现的某篇研究论文的代码。
   - 他们收到建议，将代码发布在常规频道或 **Triton** 特定频道以获取支持。
- **EAGLE 库遇到 Triton 错误**：一位成员分享了他们在 **EAGLE** 库中的 **Triton** 内核代码，并遇到了 **RuntimeError**。
   - 他们怀疑问题与代码中较大的 **block size** 有关，并被建议查看 [EAGLE 仓库](https://github.com/abhijit-aiplanet/EAGLE) 以寻找潜在的解决方案。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/TheAITimeline/status/1827467655844131070">The AI Timeline (@TheAITimeline) 的推文</a>：🚨本周热门 AI/ML 研究论文：- Transfusion - To Code, or Not To Code? - Automated Design of Agentic Systems - LLM Pruning and Distillation in Practice: Minitron - Hermes 3 Technical Repor...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1eznkml/liger_kernel_one_line_to_make_llm_training_20/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://github.com/triton-lang/triton/issues/1058">A100 上超大张量的无效内存访问 · Issue #1058 · triton-lang/triton</a>：嗨，我正尝试编写一个 Triton 内核，用随机的 +1/-1 条目填充 float16 张量。我的代码如下：import torch as ch...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1eznkml/li">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://github.com/abhijit-aiplanet/EAGLE/blob/main/eagle/model/modeling_llama_kv.py">EAGLE/eagle/model/modeling_llama_kv.py at main · abhijit-aiplanet/EAGLE</a>：EAGLE 的编辑实现。通过在 GitHub 上创建账号为 abhijit-aiplanet/EAGLE 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1276622349747359848)** (67 条消息🔥🔥): 

> - `Unsloth CPU Compatibility`
> - `Unsloth on Tesla P100`
> - `Unsloth SFTTrainer Override`
> - `Unsloth Steps vs Epochs`
> - `Unsloth Phi-3.5-MoE Availability` 


- **Unsloth 在 CPU 上运行**：一位用户询问 Unsloth 是否可以在 CPU 上运行。
- **Tesla P100 运行 Unsloth 的问题**：一位用户报告在尝试使用 Tesla P100 GPU 运行 Unsloth 时遇到了 LLVM 错误。
- **重写 SFTTrainer 以实现自定义逻辑**：一位用户询问如何重写 SFTTrainer 的 compute_loss 函数，以便根据 LLM 的输出字符串添加自定义逻辑。
- **Unsloth 训练中的 Steps 与 Epochs**：一位用户询问了 Unsloth 训练中 steps 和 epochs 之间的区别。
- **Unsloth 对 Phi-3.5-MoE 的支持情况**：一位用户询问了 Phi-3.5-MoE 模型在 Unsloth 上的可用性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=Zt9CHJqO6p30)">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHy">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://github.com/linkedin/Liger-Kernel">GitHub - linkedin/Liger-Kernel: Efficient Triton Kernels for LLM Training</a>：用于 LLM 训练的高效 Triton 内核。通过在 GitHub 上创建账号为 linkedin/Liger-Kernel 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth/blob/b8b1eafda35d124046e11766aeeb6343957e0daf/unsloth/kernels/rms_layernorm.py">unsloth/unsloth/kernels/rms_layernorm.py at b8b1eafda35d124046e11766aeeb6343957e0daf · unslothai/unsloth</a>：微调 Llama 3.1, Mistral, Phi &amp; Gemma LLM 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1277046213514104893)** (1 条消息): 

> - `Replete-LLM V2`
> - `Llama 3.1`
> - `Replete-AI/The_Living_AI_Dataset`
> - `System Prompts`
> - `Quantizations` 


- **Replete-LLM V2：第二次降临**：Replete-LLM 的第二个版本 **Replete-LLM-V2-Llama-3.1-8b** 已经发布，相比前代产品，它在推理和编码性能上有了巨大的提升。
   - 该版本在全新的 **Replete-AI/The_Living_AI_Dataset** 上进行训练，旨在教导模型学习**爱与共情 (Love and Empathy)**，这是构建能够理解并关心人类的模型所迈出的关键一步。
- **System Prompts：Replete-LLM V2 强大性能的关键**：**Replete-LLM-V2** 在训练中使用了多种 System Prompts 来引导其信息处理过程。
   - 详细、具体且有效的 System Prompts 对于发挥该模型的最佳性能至关重要。
- **本地运行的 Quantizations**：**Replete-LLM-V2-Llama-3.1-8b** 的量化文件已发布，支持 **ExLlamaV2 v0.1.9** 和 **Llama.cpp**。
   - 这些量化版本允许用户在本地运行模型，并可配合 **LM Studio** 使用以获得更具交互性的体验。
- **Prompt 格式：正确的交互方式**：与 **Replete-LLM-V2** 交互的推荐 Prompt 格式采用三段式结构：
   - 以 **system prompt** 开始，随后是 **user prompt**，最后以 **assistant response** 结束。
- **Llama 3.1：模型背后的引擎**：**Replete-LLM-V2** 基于全新的 **Llama 3.1** 模型，展示了其在各种任务中的先进能力。
   - 这一新模型为提升性能以及 AI 领域的进一步发展奠定了基础。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/Replete-AI/Replete-LLM-V2-Llama-3.1-8b">Replete-AI/Replete-LLM-V2-Llama-3.1-8b · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/rombodawg/Try_out_Replete-LLM-V2-Llama-3.1-8b">Try Out Replete-LLM-V2-Llama-3.1-8b - a Hugging Face Space by rombodawg</a>: 未找到描述</li><li><a href="https://huggingface.co/bartowski/Replete-LLM-V2-Llama-3.1-8b-exl2">bartowski/Replete-LLM-V2-Llama-3.1-8b-exl2 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/bartowski/Replete-LLM-V2-Llama-3.1-8b-GGUF">bartowski/Replete-LLM-V2-Llama-3.1-8b-GGUF · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1277621710945914881)** (3 条消息): 

> - `LLM Fine-Tuning for Question Answering`
> - `Unsloth and Llama Model` 


- **用于问答的 Unsloth 和 Llama**：一位成员请求提供一个使用 Unsloth 和 Llama 进行 LLM 问答任务微调的示例。
   - 另一位成员建议去 GitHub 搜索相关的 Notebook，并劝阻其不要在多个频道重复发帖。
- **不要偷懒，去 GitHub 探索**：一位成员建议查看 GitHub 以获取该主题相关的 Notebook。
   - 他们还劝阻不要在多个频道刷屏同一个问题。


  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1276635266374828062)** (570 条消息🔥🔥🔥): 

> - `Stable Diffusion Online`
> - `Stable Diffusion Seed`
> - `ComfyUI`
> - `Stable Diffusion Image Upscaling`
> - `Flux` 


- **Stable Diffusion Online 与 Stability AI 无关**：一位成员询问 [Stable Diffusion Online](https://stabledifffusion.com) 是官方网站还是与 `Stability AI` 无关。
- **ComfyUI：终极 Diffusion 工作流？**：一位成员建议，如果用户不打算使用 `ComfyUI` 提供的所有强大功能，他们应该直接使用 `ForgeUI`。
- **探索放大（Upscaling）领域：SD Upscale、Tiled Diffusion 及其他方法**：成员们讨论了在增加细节的同时放大图像的不同方法：`Ultimate SD Upscale`、`Tiled Diffusion`，以及在之后配合 `SUPIR` 使用 `4x-NomosWebPhoto-atd` 等模型。
- **A1111/Forge 中“额外噪声”之谜**：一位成员解释了 `A1111`/`Forge` 中“噪声注入”（Noise Injection）的概念，以及如何利用它来改进图像放大效果。
- **探索 Flux 领域：过拟合、多样性与 Instruct Pix2Pix**：成员们讨论了 `Flux` 的过拟合问题，特别是与奇幻题材相关的过拟合，以及这如何导致图像多样性的缺失。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/typing-bald-big-body-laptop-muscle-man-gif-17063750">Typing Bald GIF - Typing Bald Big Body - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://stable-diffusion-art.com/instruct-pix2pix/">Instruct Pix2Pix: Edit and stylize photos with text - Stable Diffusion Art</a>：`Instruct Pix2Pix` 是一个仅通过用户的文本指令即可编辑图像的 `Stable Diffusion` 模型。我们将介绍它的工作原理、功能以及如何操作</li><li><a href="https://huggingface.co/docs/accelerate/usage_guides/distributed_inference">Distributed Inference with 🤗 Accelerate</a>：未找到描述</li><li><a href="https://tenor.com/view/shrek-shrek-rizz-rizz-gif-11157824601050747846">Shrek Shrek Rizz GIF - Shrek Shrek rizz Rizz - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1dkh5cv/a1111_forge_improve_your_images_with_a_single/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://huggingface.co/ByteDance/Hyper-SD">ByteDance/Hyper-SD · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/mike-lowrey-gif-8186790">Mike Lowrey GIF - Mike Lowrey - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://github.com/unity-research/IP-Adapter-Instruct">GitHub - unity-research/IP-Adapter-Instruct: IP Adapter Instruct</a>：`IP Adapter Instruct`。通过在 `GitHub` 上创建账号来为 `unity-research/IP-Adapter-Instruct` 的开发做出贡献。</li><li><a href="https://stabledifffusion.com">Stable Diffusion Online - Free AI Image Generator</a>：`Stable Diffusion` 是一款免费的人工智能图像生成器，可根据简单的文本提示轻松创建高质量的 AI 艺术、图像、动漫和写实照片。无需注册！</li><li><a href="https://t.me/dogshouse_bot/join?startapp=8PYp1s3kTTSEkZBzibx3Qw">Dogs 🦴</a>：最原生于 `Telegram` 的模因币。加入我们的 @dogs_community</li><li><a href="https://t.me/dogs_community)">Telegram – a new era of messaging</a>：快速、安全、强大。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1276619011861385316)** (366 条消息🔥🔥): 

> - `Codegemma`
> - `Hermes`
> - `Mistral`
> - `Model Merging`
> - `Open Empathic` 


- **Hermes 2.5 性能超越 Hermes 2**：在添加了[代码指令示例](https://link.to.examples)后，**Hermes 2.5** 在各种基准测试中的表现似乎优于 **Hermes 2**。
   - Hermes 2 在 MMLU 基准测试中得分为 **34.5**，而 Hermes 2.5 得分为 **52.3**。
- **Mistral 难以扩展至 8k 以上**：成员们表示，如果不进行持续预训练（continued pretraining），**Mistral** 无法扩展到 8k 以上，[这是一个已知问题](https://link.to.issue)。
   - 他们指出，*mergekit* 和 *frankenMoE finetuning* 的进一步工作是性能突破的下一个前沿。
- **关于模型合并策略的讨论**：一位成员建议将 **UltraChat** 与基础 **Mistral** 之间的差异应用到 **Mistral-Yarn** 上，作为一种潜在的合并策略。
   - 尽管其他人表示怀疑，但该成员保持乐观，并引用了过去在所谓的“被诅咒的模型合并（cursed model merging）”方面的成功尝试。
- **Open Empathic 项目寻求援助**：一位成员呼吁帮助扩展 **Open Empathic** 项目的类别，特别是在低端部分。
   - 他们分享了一个 [Open Empathic 发布与教程的 YouTube 视频](https://youtu.be/GZqYr8_Q7DE)，指导用户从 YouTube 视频中贡献他们喜欢的电影场景，并提供了 [OpenEmpathic 项目本身](https://dct.openempathic.ai/)的链接。
- **GPTs Agents 在初始训练后无法学习**：一位成员担心 GPTs Agents 无法从初始训练后提供的额外信息中学习。
   - 另一位成员澄清了这一误解，解释说[上传的文件被保存为“知识（knowledge）”文件](https://link.to/openai-docs)供 Agent 在需要时引用，但**它们不会持续修改 Agent 的基础知识**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/hub/en/security-git-ssh#checking-for-existing-ssh-keys">Git over SSH</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/Vipitis/shadermatch/discussions/1">Vipitis/shadermatch · 访问性通知</a>：未找到描述</li><li><a href="https://huggingface.co/AiTommy/decider_agent_test_merged">AiTommy/decider_agent_test_merged · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/AiTommy/decider_agent_lora">AiTommy/decider_agent_lora · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#pull-request-checklist>.">transformers/CONTRIBUTING.md at main · huggingface/transformers</a>：🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的最先进机器学习库。 - huggingface/transformers</li><li><a href="https://youtu.be/3u8HndJlA0c?si=NJ2RJQs7y-uo_-M-">“Claude Investor”会主宰投资研究的未来吗？——AI Agent 激增开始</a>：跟我学 AI：https://www.skool.com/natural20/about 加入我的社区和课堂，学习 AI 并为新世界做好准备。[链接见下文] JOSH BICKETTCa...</li><li><a href="https://youtu.be/3u8HndJlA0c?si=NJ2RJQ">“Claude Investor”会主宰投资研究的未来吗？——AI Agent 激增开始</a>：跟我学 AI：https://www.skool.com/natural20/about 加入我的社区和课堂，学习 AI 并为新世界做好准备。[链接见下文] JOSH BICKETTCa...</li><li><a href="https://github.com/vatsalsaglani/GenAINewsAgent?tab=readme-ov-file">GitHub - vatsalsaglani/GenAINewsAgent：一个使用 Llama-3 在 Groq 上实现的 GenAI 新闻摘要 Agent 的快速实现。</a>：一个使用 Llama-3 在 Groq 上实现的 GenAI 新闻摘要 Agent 的快速实现。 - vatsalsaglani/GenAINewsAgent</li><li><a href="https://github.com/mshumer/gpt-investor">GitHub - mshumer/gpt-investor</a>：通过在 GitHub 上创建账户来为 mshumer/gpt-investor 的开发做出贡献。</li><li><a href="https://github.com/Vipitis/shadertoys-dataset/blob/dev-testing/experiments/run_experiments.ipynb?short_path=97d4273#L4885>)">shadertoys-dataset/experiments/run_experiments.ipynb at dev-testing · Vipitis/shadertoys-dataset</a>：数据集的 WIP 重构。通过在 GitHub 上创建账户来为 Vipitis/shadertoys-dataset 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1276678799852830802)** (18 条消息🔥): 

> - `Model Quantization`
> - `Model Distillation`
> - `Diffusion Model Training`
> - `GPT-2 Rewriting`
> - `Prompt Weights` 


- **Model Quantization 和 Distillation 是生产环境的必备要素**：作者讨论了 **Model Quantization** 和 **Model Distillation** 对于机器学习工程师将模型投入生产环境（而非仅限于本地训练和测试）的重要性。
- **Nautilus 集成了 NIST 的 Dioptra 用于 AI Redteaming**：由 **Qompass** 开发的安全平台 **Nautilus** 已集成 **NIST 的 Dioptra**，用于 **AI Redteaming**。
   - 作者对 **NIST** 开源这些工具表示乐观，并认为理解他们处理 **AI Security** 的方法将大有裨益。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://phrygian-alarm-029.notion.site/notes-reading-7b6946b8f4074771b9922654d61075d6?pvs=4">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>：一款将日常工作应用融为一体的新工具。它是为您和您的团队打造的一体化工作空间。</li><li><a href="https://www.youtube.com/watch?v=l8pRSuU81PU">让我们复现 GPT-2 (124M)</a>：我们从头开始复现 GPT-2 (124M)。这段视频涵盖了整个过程：首先构建 GPT-2 网络，然后优化其训练使其达到...</li><li><a href="https://github.com/qompassai/Nautilus/tree/main/RedTeam/RedAI/NIST/Qompass_Dioptra">Nautilus/RedTeam/RedAI/NIST/Qompass_Dioptra 分支 main · qompassai/Nautilus</a>：符合 NIST 标准的安全解决方案。通过在 GitHub 上创建账号为 qompassai/Nautilus 的开发做出贡献。</li><li><a href="https://www.nist.gov/news-events/news/2024/07/department-commerce-announces-new-guidance-tools-270-days-following">在拜登总统关于 AI 的行政命令发布 270 天后，商务部宣布新的指南和工具</a>：商务部首次公开了来自美国 AI Safety Institute 的新 NIST 指南草案，以帮助 AI 开发者评估和缓解源自生成式 AI 和双用途基础模型的风险...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1276659260767338528)** (6 条消息): 

> - `Stale bot`
> - `LLM Merging`
> - `Medical AI Papers`
> - `1-bit LLMs` 


- **Stale bot 对开源项目的影响**：一项针对 20 个大型开源项目的研究探讨了使用 [Stale bot](https://github.com/probot/stale) 自动跟踪和关闭不活跃 Pull Requests 的效果。
   - 虽然 Stale bot 可以帮助管理积压的未解决 PR 并提高审查效率，但也可能带来负面后果。
- **Awesome-Model-Merging-Methods: LLM 合并资源**：一个 [GitHub 仓库](https://github.com/EnnengYang/Awesome-Model-Merging-Methods-Theories-Applications) 整理了关于 LLM 合并方法、理论和应用的论文。
   - 该仓库基于一篇题为 "Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities" 的论文。
- **顶级医疗 AI 研究论文：8 月 17 日当周**：Twitter 上的一个帖子重点介绍了 2024 年 8 月 17 日当周发表的几篇值得关注的医疗 AI 研究论文。
   - 讨论的论文涵盖了医疗多模态 LLM 的越狱、面向医生的规则对齐 LLM 以及使用 LLM 进行临床试验预测等主题。
- **微软在 1-bit LLMs 领域取得突破**：微软在 1-bit LLMs 方面取得了重大进展，创建了使用三值 (-1, 0, 1) 而非传统 16-bit 格式的模型。
   - 这一创新带来了 2.7 倍的速度提升、3.5 倍的 GPU 显存占用减少以及 71 倍的能耗降低，同时保持甚至超过了 LLaMA 3B 等传统模型的性能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/OpenlifesciAI/status/1827442651810918509">来自 Open Life Science AI (@OpenlifesciAI) 的推文</a>：上周医疗 AI 动态：顶级研究论文/模型 🏅（2024 年 8 月 17 日 - 8 月 24 日）- 医疗多模态 LLM 的越狱 - LLM 不是零样本生物医学推理者 - RuleAlign 框架：对齐...</li><li><a href="https://arxiv.org/abs/2305.18150">Understanding the Helpfulness of Stale Bot for Pull-based Development: An Empirical Study of 20 Large Open-Source Projects</a>：既没有进展也没有解决的 Pull Requests (PRs) 会使 PR 列表变得混乱，使维护者难以管理和确定未解决 PR 的优先级。为了自动跟踪、跟进...</li><li><a href="https://github.com/pints-ai/1.5-Pints">GitHub - Pints-AI/1.5-Pints: A compact LLM pretrained in 9 days by using high quality data</a>：通过使用高质量数据在 9 天内预训练的紧凑型 LLM - Pints-AI/1.5-Pints</li><li><a href="https://github.com/EnnengYang/Awesome-Model-Merging-Methods-Theories-Applications">GitHub - EnnengYang/Awesome-Model-Merging-Methods-Theories-Applications: Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities. arXiv:2408.07666.</a>：LLM、MLLM 及更高领域的模型合并：方法、理论、应用与机遇。arXiv:2408.07666。- EnnengYang/Awesome-Model-Merging-Methods-Theories-Applications
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1276690002985746493)** (14 条消息🔥): 

> - `Tau LLM`
> - `database match command`
> - `TauAgent class`
> - `reward signals`
> - `TinyLlama` 


- **Tau LLM 系列持续更新**：Tau LLM 系列的创作者继续完善他们的项目，重点实现了诸如 `database match` 命令、`TauAgent` 类和奖励信号（reward signals）等新功能。
   - 他们还讨论了初始训练数据，包括三个领域：数学、语法和拼写。
- **TinyLlama 在 9 天内取得成功**：TinyLlama 是一个与 Tau LLM 类似的模型，由同一研究小组在短短 9 天内训练完成，其在 MTBench 上的表现优于 Apple OpenELM 和 Microsoft Phi。
   - 他们已在 GitHub 上发布了训练代码，并在 HuggingFace 上发布了模型权重。
- **GT-AI 翻译工具**：一款名为 GT-AI 的新工具现已推出，可用于文本翻译、补全、语言检测等。
   - 它提供多种 AI 模型和响应式设计，可在桌面和移动设备上无缝运行。
- **Voicee：超快速语音助手**：开发了一款名为 Voicee 的新语音助手，延迟低于 500ms，平均延迟为 700ms。
   - 它在 Google Chrome 中运行效果最佳，并正在征求用户反馈。
- **用于情感 AI 的 Dark Sentience 数据集**：策划了一个名为 "Dark Sentience" 的新数据集，旨在解决 AI 语言模型缺乏情感深度的问题，特别针对更阴暗、更具内省性的情绪。
   - 该数据集旨在通过让 AI 接触复杂的人类情感（包括自杀、抑郁和焦虑）来增强其情感智能。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/maximuspowers/bias-detection-ner">Bias Detection NER - a Hugging Face Space by maximuspowers</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/KingNish/Voicee">Voicee - a Hugging Face Space by KingNish</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/huseinzol05/2d-parallelism-ray-pytorch">2D Parallelism using Ray PyTorch</a>: 未找到描述</li><li><a href="https://github.com/Om-Alve/GemmaFromScratch/">GitHub - Om-Alve/GemmaFromScratch</a>: 通过在 GitHub 上创建账户，为 Om-Alve/GemmaFromScratch 的开发做出贡献。</li><li><a href="https://github.com/LegallyCoder/StockLlama">GitHub - LegallyCoder/StockLlama: StockLlama is a time series forecasting model based on Llama, enhanced with custom embeddings for improved accuracy.</a>: StockLlama 是一个基于 Llama 的时间序列预测模型，通过自定义嵌入（embeddings）增强以提高准确性。 - LegallyCoder/StockLlama</li><li><a href="https://youtube.com/live/0WUXHWen0F8?feature=share">Unity ML-Agents | Pretrain an LLM from Scratch with Sentence Transformers | Part 8</a>: 欢迎回到我们的 Tau LLM 系列！🌟在本集中，我们将深入探讨项目的一些关键组件。亮点包括：- **构建训练...</li><li><a href="https://github.com/huridocs/pdf-document-layout-analysis">GitHub - huridocs/pdf-document-layout-analysis: A Docker-powered service for PDF document layout analysis. This service provides a powerful and flexible PDF analysis service. The service allows for the segmentation and classification of different parts of PDF pages, identifying the elements such as texts, titles, pictures, tables and so on.</a>: 一个基于 Docker 的 PDF 文档布局分析服务。该服务提供强大且灵活的 PDF 分析功能，允许对 PDF 页面的不同部分进行分割和分类，识别文本、标题、图片、表格等元素。</li><li><a href="https://youtube.com/live/J1_zpS4HZMc?feature=share">Unity ML-Agents | Pretrain an LLM from Scratch with Sentence Transformers | Part 7</a>: 欢迎回到我们的 Tau LLM 系列！🌟在本集中，我们将深入探讨一些令人兴奋的新进展，并继续完善我们的项目。亮点包括...</li><li><a href="https://github.com/pints-ai/1.5-Pints">GitHub - Pints-AI/1.5-Pints: A compact LLM pretrained in 9 days by using high quality data</a>: 一个使用高质量数据在 9 天内预训练完成的紧凑型 LLM - Pints-AI/1.5-Pints</li><li><a href="https://huggingface.co/collections/pints-ai/15-pints-66b1f957dc722875b153b276">1.5-Pints - a pints-ai Collection</a>: 未找到描述</li><li><a href="https://gt-ai-j3yu.vercel.app/">Create Next App</a>: 未找到描述</li><li><a href="https://github.com/U-C4N/GT-AI">GitHub - U-C4N/GT-AI</a>: 通过在 GitHub 上创建账户，为 U-C4N/GT-AI 的开发做出贡献。</li><li><a href="https://insiders.dashtoon.com/dashanimexl/">Introducing DashAnime XL 1.0</a>: Dashtoon Studio 是一个创新的 AI 驱动平台，旨在赋能创作者进行漫画创作。动漫的巨大普及和深远的文化影响力引发了热烈的需求...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1276620023565258814)** (14 messages🔥): 

> - `AWS RAGCHECKER`
> - `DPO`
> - `KTO`
> - `LLMs Alignment` 


- **AWS RAGCHECKER 论文讨论**：一名成员提到了 [AWS RAGCHECKER 论文](https://arxiv.org/pdf/2408.08067)，并询问是否可以在 HuggingFace 中实现。
- **DPO 与 KTO 微调方法**：另一名成员建议讨论 DPO 和 KTO 微调方法及其在当前语言模型中的作用。
- **用于 LLMs 对齐的 Constitutional DPO**：一名成员提到讨论 Constitutional DPO，这是一种用于对齐 LLMs 的方法，及其在近期研究中的重要性。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1277273994667622401)** (2 messages): 

> - `FSRCNN implementation`
> - `PyTorch implementation` 


- **FSRCNN 实现需要改进**：一名成员就如何改进其简化的 FSRCNN PyTorch 实现寻求建议。
   - 他们分享了 GitLab 上的项目链接：[paper_replication](https://gitlab.com/amrirasyidi/paper_replication/-/blob/master/notebooks/0_dummy.ipynb?ref_type=heads)。
- **FSRCNN 项目协作**：另一名成员表达了对参与同一项目的兴趣。
   - 他们对协作改进 FSRCNN 实现表示兴奋。



**提到的链接**：<a href="https://gitlab.com/amrirasyidi/paper_replication/-/blob/master/notebooks/0_dummy.ipynb?ref_type=heads">notebooks/0_dummy.ipynb · master · Amri Rasyidi / Paper Replication · GitLab</a>: GitLab.com

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1277253721536991335)** (7 messages): 

> - `VLLM memory usage`
> - `VLLM system memory utilization`
> - `Accelerate trainer not working on Google Colab TPU`
> - `Speculative decoding with tool calling` 


- **VLLM 内存占用之谜**：一位用户报告称，与标准模型加载相比，使用 **VLLM** 时内存消耗显著增加，即使是较小的模型也是如此。
   - 提供了一个关于 GPU 显存消耗的 [GitHub discussion](https://github.com/vllm-project/vllm/discussions/241) 链接，可能为这种差异提供见解。
- **VLLM 是系统内存杀手吗？**：用户指出 **Qwen 7B** 通常占用约 **14GB VRAM**，但在使用 **VLLM** 运行时，尽管将 **GPU memory utilization** 设置为 **0.95** 且 **CPU offload** 设置为 **0GB**，系统内存使用量仍飙升至 **20GB 以上**。
   - 用户对这种意外的系统内存占用表示困惑，想知道这是否为 **VLLM** 的标准行为。
- **Accelerate Trainer 的 TPU 问题**：另一位用户在使用 **Google Colab TPU** 时遇到了 **Accelerate trainer** 无法按预期工作的问题。
   - 用户分享了一个 [简单 NLP notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/accelerate_examples/simple_nlp_example.ipynb) 的链接和一张错误图片，寻求故障排除指导。
- **Speculative Decoding 与 Tool Calling：天作之合？**：一位用户询问了 **speculative decoding** 与 **tool calling** 的兼容性。
   - 这个问题目前尚未得到解答，暗示需要进一步探索这两种技术的交集。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/accelerate_examples/simple_nlp_example.ipynb)">Google Colab</a>：未找到描述</li><li><a href="https://github.com/vllm-project/vllm/discussions/241">关于 GPU 显存消耗几乎翻倍的问题 · vllm-project/vllm · Discussion #241</a>：当我使用默认配置加载 vicuna7b_v1.3 时，发现 GPU 显存消耗约为 23G，但 fastchat 的 readme 中提到 Vicuna-7B 仅需约 14GB GPU 显存...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1276837781116817420)** (8 messages🔥): 

> - `Hugging Face iOS App Model Configuration`
> - `Stable Diffusion Fine-Tuning`
> - `Quantization Changes in Stable Diffusion` 


- **Hugging Face iOS App 模型配置目前是固定的**：一位用户询问 Hugging Face iOS App 模型是否可以由用户配置，还是目前是固定的。目前尚未确认该 App 未来是否可以升级。
   - null
- **使用个人数据进行 Stable Diffusion 微调**：另一位用户寻求使用 Dreambooth 和个人数据微调 Stable Diffusion 2 的帮助。
   - null
- **Stable Diffusion 中的量化变更**：一位用户解释说他们更改了 Stable Diffusion 中的量化过程，从而提高了输出质量和 Lora 表现。
   - 该用户表示，预量化的 Checkpoints 可能与最新代码不兼容，需要重新量化。
- **使用 Automatic1111 微调 Stable Diffusion**：一位用户询问另一位提供 Stable Diffusion 微调帮助的用户是否正在使用 Automatic1111（一个流行的 Stable Diffusion Web UI）。
   - null
- **理解微调中的 "CLASS_DIR" 变量**：一位用户对 Stable Diffusion 微调中使用的 "CLASS_DIR" 变量表示困惑，特别是它应该只包含用户的数据还是也包含其他数据。
   - 该用户提到他们的数据集包含不同类型的照片，他们希望微调出一个能生成写实图像的模型。


  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1276667799103737977)** (275 messages🔥🔥): 

> - `GPTs`
> - `AGI`
> - `LLMs`
> - `Model Scaling`
> - `DeepMind` 


- **模型缩放的边际收益递减**：讨论围绕着这样一个观点展开，即单纯通过更多的算力和数据来扩大模型规模可能会达到边际收益递减的点，届时进一步的提升将变得微乎其微。
   - Llama 3.1 和 Claude 3.5 Sonnet 等模型的表现证明了这一点，其性能的提升与算力的增加并不成正比。
- **关于 AI 意识的辩论**：对话深入探讨了人工智能的哲学层面，特别是针对当前的 LLMs 是否可以被视为具有意识的问题。
   - 提出了几种观点，包括 AI 意识可能遵循与人类意识不同的规律，以及 AI 对世界的理解受限于其缺乏有机体验。
- **AI 的未来：超越缩放**：讨论了在深度学习架构中进行算法突破的需求，以推动 AI 超越缩放的界限。
   - 参与者承认，虽然缩放带来了令人印象深刻的进步，但对于实现 AGI 来说是不够的。大家一致认为，需要创新方法来解决根本局限并实现真正的突破。
- **AI 与自由意志问题**：对话探讨了自由意志的概念及其与 AI 系统的关系。
   - 参与者讨论了这样一种可能性：尽管 AI 可能不像人类那样拥有自由意志，但它们仍然能够根据其内部逻辑和推理做出选择。
- **AI 的哲学影响**：对话触及了 AI 的哲学影响，包括 AI 重塑我们对意识、人格以及现实本质本身理解的潜力。
   - 参与者探讨了一系列观点，从认为 AI 本质上是人类使用的工具，到认为 AI 最终可能会挑战我们对“何为人”的理解。



**提到的链接**：<a href="https://www.youtube.com/watch?v=DlX3QVFUtQI">What runs GPT-4o? Inside Microsoft&#39;s 2024 AI supercomputer with Mark Russinovich</a>：微软已经建造了全球最大的云端 AI 超级计算机，其规模已比 6 个月前呈指数级增长，为……铺平了道路。

  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1277072344040345713)** (8 messages🔥): 

> - `GPT 分享与社区建设`
> - `GPT Builder 问题`
> - `OpenAI API 定价与订阅模式` 


- **分享 GPT：它们会出现在哪里？**: 一位成员询问了在社区内分享 GPT 以及如何跟踪其效果的问题。
   - 他们询问如何确定分享的 GPT 是否正在被使用，或者是否对他人有效，特别是关于 'share-output' 或 'use-cases' 功能。
- **GPT Builder 的可点击链接存在 Bug**: 一位成员报告了 GPT Builder 在生成可点击的网页搜索结果链接时遇到困难，称该问题表现不稳定。
   - 他们询问是否还有其他人遇到此问题，以及是否有已知的解决方案。
- **OpenAI API 上的订阅模式**: 一位成员询问平台如何使用 OpenAI 的基于 Token 的定价模型来管理订阅计划（如月度订阅）。
   - 他们以 Chatbase 为例，询问在 OpenAI 仅提供基于 Token 定价的情况下，如何实现此类计划。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1276852519968047236)** (12 messages🔥): 

> - `GPTs 自定义构建器`
> - `Prompt engineering 模板`
> - `GPT jailbreak`
> - `Meta robot 标签` 


- **使用 GPT store 构建自定义 GPT**: 一位成员建议使用自定义 GPT builder 创建一个理解公司品牌标识和语调的 GPT 模型，然后将该模型作为内容创作助手的 System Prompt。
   - 该成员建议可以将 GPT store 中的 GPT prompt engineering 模型用作 OpenAI playground 或 API 中的 System Prompt。
- **分享 GPT Jailbreak 方法**: 一位成员分享了他们发现并测试的一种针对 GPT-4 的新 jailbreak 方法。
   - 另一位成员表示，目前没有专门用于分享 jailbreak 方法的频道。
- **使用 GPT 检查 meta robot 标签**: 一位成员询问是否可以使用 GPT 检查 URL 的 meta robot 标签，以识别 "noindex" 和 "nofollow"。
   - 该成员不确定这是否是提问的正确频道。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1276852519968047236)** (12 messages🔥): 

> - `GPT-4 Jailbreak`
> - `Meta Robot 标签`
> - `Prompt Engineering` 


- **GPT-4 Jailbreak 讨论**: 一位用户询问是否有专门分享新 GPT-4 jailbreak 方法的频道。
   - 回复确认没有此类频道。
- **使用 GPT 检查 Meta Robot 标签**: 一位用户询问是否可以使用 GPT 检查 URL 的 "noindex" 和 "nofollow" meta robot 标签。
- **用于内容创作助手的 Prompt Engineering**: 一位用户正在寻求用于内容创作助手的 Prompt engineering 模板或指南，旨在教导它们品牌标识、语调以及需要避免的术语。
   - 回复建议使用自定义 GPT builder，然后利用生成的 Prompt 作为 System Prompt 以获得进一步的控制。


  

---



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1276650067960270850)** (2 messages): 

> - `Perplexity Creator 社区`
> - `Kale 合作伙伴关系`
> - `为 LinkedIn Premium 用户提供 Perplexity Pro` 


- **Perplexity Creator 社区上线**: Perplexity AI 已与 [Kale](https://www.kalecard.com/t/perplexity.ai?ref=perplexity.ai_discord) 合作推出 **Perplexity Creator Community**，这是一个奖励分享关于 Perplexity 的引人入胜视频内容的创作者计划。
   - 创作者可以根据其视频的影响力赚取现金，并可以按照自己的时间表发布内容。
- **Kale：赋能创作者并连接品牌**: Kale 是一个将品牌与擅长社交媒体的创作者联系起来的平台，为真实的创作内容提供奖励。
   - Kale 的算法会将创作者与他们购买过的品牌以及与其社交媒体活动相关的品牌进行匹配，鼓励真实且引人入胜的内容。
- **为 LinkedIn Premium 订阅者提供免费 Perplexity Pro**: Perplexity 正向所有 **LinkedIn Premium 订阅者** 提供为期一年的免费 **Perplexity Pro**。
   - Perplexity Pro 提供无限次 Pro 搜索，可访问 **GPT-4 Omni, Claude 3.5 Sonnet, 和 Llama 3.1 405b** 等前沿 AI 模型，支持文件上传分析以及多模态功能。



**提到的链接**: <a href="https://www.kalecard.com/t/perplexity.ai?ref=perplexity.ai_discord">Perplexity AI Creator Community</a>: 发布关于 Perplexity AI 的内容即可赚取奖励！

  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1276617638919012475)** (216 条消息🔥🔥): 

> - `Perplexity Pro`
> - `GPT-4o`
> - `Perplexity's performance`
> - `Claude 3.5 Sonnet`
> - `Image generation in Perplexity` 


- **Perplexity Pro 的 LinkedIn 优惠**: LinkedIn Premium 用户被提供为期一年的 Perplexity Pro，但一些用户报告称该优惠在欧盟 (EU) 无法使用，尽管其他欧盟用户表示已经收到了该优惠。
   - 一位来自欧盟小国的用户确认收到了优惠，另一位欧盟用户也确认该优惠对他们有效。
- **GPT-4o 对比 Claude 3.5 Sonnet**: 讨论围绕 GPT-4o 在编程方面的优势以及 Claude 3.5 Sonnet 在知识检索方面的优势展开。GPT-4o 被认为在 STEM 课题上表现更好，而 Sonnet 在编程相关任务中表现出色。
   - 用户注意到，与 GPT-4o 和 Bing 不同，Claude 3.5 Sonnet 在回答有关诗歌和故事的问题时知识有限，但在编程任务中表现良好。一些用户还强调 ChatGPT 提供多个 GPTs，使其成为处理各种任务的不错选择。
- **Perplexity 图像生成问题**: 用户报告了 Perplexity 图像生成的问题，一些对话在尝试使用 Dalle3 生成图像后被“毁掉”，另一些用户则遇到了生成图像质量的问题。
   - 一位用户认为问题可能与图像生成过程本身有关，因为一个简单的提示词就导致了对话线程“变砖”。
- **Perplexity 的搜索功能**: 用户对 Perplexity 的搜索功能表示担忧，报告称搜索不可靠，且回答不能反映最新信息，导致对事实核查准确性的怀疑。
   - 一些用户还强调在提示词中使用 "TRIGGER WEB SEARCH" 指令可以改善搜索结果，这暗示搜索功能可能存在 Bug。
- **Perplexity 的 Code Interpreter 推出**: Perplexity 的 CEO 宣布推出 Code Interpreter 和图表渲染功能，但用户反映目前功能有限。
   - 一位用户表示 Code Interpreter 非常受限，不允许更改生成的输出，并建议在更高级的使用场景中使用 Google Colab。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/AravSr">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://tenor.com/view/sponge-bob-imagination-rainbow-gif-6957879033178108845">Sponge Bob Imagination GIF - Sponge Bob Imagination Rainbow - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://x.com/aravsrinivas/status/1827117952669512116?s=61">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>: 政府在为你的新冠检测买单。微软在为你的 ChatGPT 买单。风投 (VCs) 在为你的 Perplexity 买单。扎克伯格在为你的模型权重买单。匿名人士，你还有什么好抱怨的？引用...</li><li><a href="https://x.com/AravSrinivas/status/1825617944782758066">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>: Code interpreter 和图表渲染正在逐步推出！引用 Phil (@phill__1)：哇，Perplexity 的 Code interpreter 现在可以安装库并在结果中显示图表了！这使得...</li><li><a href="https://github.com/pnd280/complexity/issues/new?labels=bug&projects=&template=bug_report.yml&title=%5BBug%5D%3A+">共同构建更好的软件</a>: GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、Fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://chromewebstore.google.com/detail/complexity/ffppmilmeaekegkpckebkeahjgmhggpj">Complexity - Chrome 网上应用店</a>: ⚡ 增强你的 Perplexity.ai 体验</li><li><a href="https://youtu.be/H-0PsLmEi4A?si=ZgykVL1c9VfC70nO">Perplexity 如何运作 🤖 — 对话 Denis Yarats</a>: 今天的嘉宾是 Denis Yarats。Denis 是 Perplexity 的联合创始人兼 CTO，Perplexity 是我最喜欢的产品之一，也是当今最成功的 AI 初创公司之一。Perplex...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1276648917127139348)** (24 messages🔥): 

> - `Perplexity Onboarding`
> - `Perplexity Page Features`
> - `Teaching Seniors`
> - `Steve Jobs Jacket`
> - `Model Motivation` 


- **Jai Bhagat 教老年人使用 Perplexity**：教学设计师 Jai Bhagat 挑战自己，教 50 岁以上的人如何使用 Perplexity。
   - 他在他的 Perplexity Page 中创建了一个特别的 Easter Egg，用户可以在那里观看他进行 Perplexity Onboarding，展示了该平台在无障碍学习方面的潜力。
- **Perplexity Page 功能探索**：对话深入探讨了 Perplexity Page 的功能，特别是强调了创建互动式和参与式学习资源的能力。
   - 用户强调了利用 Perplexity 的能力创建教育内容所带来的乐趣。
- **使用 Perplexity 探索世界**：对话涵盖了从待售的 Steve Jobs 夹克到以色列-黎巴嫩冲突等广泛话题，展示了 Perplexity 广泛的信息检索能力。
   - 用户通过在各种主题和查询之间无缝切换，展示了 Perplexity 的多功能性。



**Link mentioned**: <a href="https://www.youtube.com/embed/ubHakI1ARR8">YouTube</a>: no description found

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1276817656632184852)** (6 messages): 

> - `Perplexity API Rate Limits`
> - `Pro Search API`
> - `Perplexity API Responsiveness` 


- **Newcode.ai 需要提高 API Rate Limits**：Newcode.ai 的首席执行官兼创始人 Maged Helmy（拥有超过 3,500 名用户）表示，他们的 Perplexity API 集成迫切需要提高 API Rate Limits。
   - Maged 在过去六个月里一直在提交提高 Rate Limits 的申请，但未收到任何回复，并请求 Perplexity 团队成员直接联系。
- **Perplexity API 用户表达挫败感**：一位用户对从 Perplexity API 收到的响应质量表示不满。
   - 他们特别指出，响应感觉很“笨”，而且模型似乎在忽略指令。
- **Pro Search API 可用性**：一位用户询问了通过 API 使用 Perplexity Pro Search 功能的可能性。
   - 用户寻求澄清该功能是已经存在还是正在开发中。


  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1276749289787228221)** (1 messages): 

> - `Database Outage` 


- **数据库故障**：最近的一次数据库更改导致了约 2 分钟的停机。
   - 问题已修复，服务应已恢复正常。
- **对造成的不便表示歉意**：我们对最近停机造成的不便表示歉意。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1276622577913168047)** (245 messages🔥🔥): 

> - `Grok-2`
> - `Grok-mini`
> - `Mistral`
> - `Claude`
> - `OpenRouter`

- **Grok-2 和 Grok-mini 加入排行榜**：令人振奋的消息，**xAI 的 Grok-2** 和 **Grok-mini** 现已登上 [LMSYS Leaderboard](https://lmsys.org/leaderboard)，获得了超过 6000 张社区投票！
   - **Grok-2** 超越了 **GPT-4o** (5月版)，并与最新的 **Gemini** 并列第 2 名；而 **Grok-2-mini** 位列第 5，在 **Math** (#1) 领域表现卓越，并在 **Hard Prompts**、**Coding** 和 **Instruction-following** 方面全面位居第 2。
- **Mistral 难以扩展至 8k 以上**：成员们对 **Mistral** 在不进行持续 pretraining 的情况下无法扩展到 8k 以上表示担忧，并指向了[这个已知问题](https://link.to.issue)。
   - 他们建议进一步探索 *mergekit* 和 *frankenMoE finetuning* 以突破性能边界。
- **Claude 3.5 Sonnet 再次宕机**：用户报告 **Claude 3.5 Sonnet** 频繁出现故障，影响了其可用性。
   - 虽然 **Haiku** 似乎运行正常，但像 **Hermes 3.5** 这样的其他模型也遇到了问题，引发了对影响这些模型的更广泛问题的猜测。
- **OpenRouter 的 API Key 与定价**：用户正在讨论如何将自己的 **API keys** 添加到 **OpenRouter**，以及显示的 Token 定价是否反映了包含 **OpenRouter fee** 在内的总成本。
   - 澄清了显示的 Token 价格是以 **OpenRouter credits** 为单位的，且在向账户充值时会自动计算费用。
- **探索开源模型：DeepSeek 和 Codestral**：成员们讨论了 **DeepSeek Coder V2** 的优势和局限性，强调了其在从零开始编写代码方面极高的性价比，但在理解和重构现有代码方面存在弱点。
   - 来自 **Mistral** 的 **Codestral 22B** 也被提及为 open weights 代码模型的有力竞争者，目前可通过 **API** 免费使用。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.novelcrafter.com/">The Novel Writing Toolbox</a>: 未找到描述</li><li><a href="https://zed.dev/blog/zed-ai">Introducing Zed AI - Zed Blog</a>: 由 Anthropic 的 Claude 提供支持的强大 AI 辅助编程，现已推出。</li><li><a href="https://aider.chat/docs/leaderboards/#code-refactoring-leaderboard">Aider LLM Leaderboards</a>: LLM 代码编辑能力的量化基准。</li><li><a href="https://huggingface.co/CausalLM/miniG">CausalLM/miniG · Hugging Face</a>: 未找到描述</li><li><a href="https://openrouter.ai/settings/integrations">Integrations | OpenRouter</a>: 在 OpenRouter 中使用您自己的供应商密钥</li><li><a href="https://turingpi.com/product/turing-pi-2/">Buy Turing Pi 2, mini ITX cluster board - for sale</a>: Turing Pi 2 是一款内置以太网交换机的 4 节点 mini ITX 集群板，可运行 Turing RK1、Raspberry Pi CM4 或 Nvidia Jetson 计算模块</li><li><a href="https://openrouter.ai/activity">Activity | OpenRouter</a>: 查看您在 OpenRouter 上使用模型的情况。</li><li><a href="https://x.com/NousResearch/status/1828121648383566270">Tweet from Nous Research (@NousResearch)</a>: 如果你可以利用世界上所有的计算能力来训练一个共享的开源 AI 模型会怎样？初步报告：https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO...</li><li><a href="https://www.anthropic.com/news/prompt-caching">Prompt caching with Claude</a>: Prompt 缓存现已在 Anthropic API 上推出，它允许开发人员在 API 调用之间缓存常用的上下文。通过 Prompt 缓存，客户可以为 Claude 提供更多的背景...</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-70b-instruct/parameters">Meta: Llama 3.1 70B Instruct – Recommended Parameters</a>: 查看 Meta: Llama 3.1 70B Instruct 的推荐参数和配置 - Meta 最新推出的 Llama 3.1 系列模型包含多种尺寸和版本。这款 70B 指令微调...</li><li><a href="https://docs.mistral.ai/capabilities/code_generation/">Code generation | Mistral AI Large Language Models</a>: Codestral</li><li><a href="https://www.latent.space/p/cosine">Is finetuning GPT4o worth it?</a>: Cosine Genie 如何在 SWE-Bench Lite 上达到 50%，在完整版 SWE-Bench 上达到 30%，以及在 OpenAI 新的 SWE-Bench Verified 上达到 44%，这些都是有史以来差距最大的 SOTA 结果。</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-405b/activity">Meta: Llama 3.1 405B (base) – Recent Activity</a>: 查看 Meta: Llama 3.1 405B (base) 的最近活动和使用统计数据 - Meta 最新推出的 Llama 3.1 系列模型包含多种尺寸和版本。这是基础的 405B 预训练...</li><li><a href="https://x.com/lmsysorg/status/1827041269534879784?s=46&t=Q_sUgNqB0V1zhMyW85SZDw">Tweet from lmsys.org (@lmsysorg)</a>: Chatbot Arena 更新❤️‍🔥 激动人心的消息——@xAI 的 Grok-2 和 Grok-mini 现已正式登上排行榜！凭借超过 6000 张社区投票，Grok-2 夺得第 2 名，超越了 GPT-4o (5月)...</li><li><a href="https://github.com/anthropics/anthropic-cookbook/blob/main/misc/prompt_caching.ipynb">anthropic-cookbook/misc/prompt_caching.ipynb at main · anthropics/anthropic-cookbook</a>: 展示使用 Claude 的一些有趣且有效方法的 Notebook/食谱集合。 - anthropics/anthropic-cookbook</li><li><a href="https://openrouter.ai/docs/provider-routing#ignoring-providers">Provider Routing | OpenRouter</a>: 在多个供应商之间路由请求</li><li><a href="https://openrouter.ai/settings/preferences">Settings | OpenRouter</a>: 管理您的账户和偏好设置</li><li><a href="https://openrouter.ai/models/lynn/soliloquy-v3">Llama 3 Soliloquy 7B v3 32K - API, Providers, Stats</a>: Soliloquy v3 是一款功能强大的角色扮演模型，专为沉浸式、动态体验而设计。Soliloquy v3 经过超过 20 亿 token 的角色扮演数据训练，拥有庞大的知识库和丰富的...
</li>
</ul>

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1276706249349660755)** (67 条消息🔥🔥): 

> - `OMI`
> - `LLM failure mode`
> - `Anthropic Interpretability Work`
> - `Sparse MoE`
> - `Model Interpretability` 


- **OMI 模型能力**：引发了关于 OMI 参与者从零开始创建 AI 模型能力的讨论，但未分享明确的观点或评估。
- **LLM 重复失败模式**：讨论了一种常见的失败模式，即 LLM 陷入循环并重复相同的短语。
   - 导致这种情况的具体条件尚不明确，但推测与模型过度量化（over-quantization）以及通过重复相同短语来最小化交叉熵损失（cross-entropy loss）的倾向有关。
- **Anthropic 的可解释性工作：成本估算**：一名成员询问了复制 Anthropic 的机械可解释性（mechanistic interpretability）工作的成本，特别是针对 Llama 8B 或 Mistral 等模型。
   - 该过程的高成本归因于对数据和计算资源的高需求，但未提供精确的估算。
- **Sparse MoE：GPU 稀疏性优势**：提出了关于 Sparse MoE 如何利用 GPU 上的稀疏性的问题。
   - 解释指出，稀疏性通过允许将专家（experts）拆分到不同机器上，从而实现高效利用，这有利于分布式训练，同时也适用于分布式推理。
- **模型训练和评估日志记录：最佳实践**：一名成员寻求适合记录模型训练和评估运行的平台建议。
   - 讨论探讨了各种方法，包括文本文件、CSV 文件，甚至使用 git 来管理训练日志和结果，并分析了每种方法的潜在优缺点。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2406.04093">Scaling and evaluating sparse autoencoders</a>：稀疏自编码器（Sparse autoencoders）提供了一种极具前景的无监督方法，通过从稀疏瓶颈层重建激活值，从语言模型中提取可解释的特征。由于语言模型...</li><li><a href="https://github.com/rokosbasilisk/filler_tokens/blob/v2/paper.pdf">filler_tokens/paper.pdf at v2 · rokosbasilisk/filler_tokens</a>：关于填充 Token（filler tokens）的 logit lens 实验。通过在 GitHub 上创建账号来为 rokosbasilisk/filler_tokens 的开发做出贡献。</li><li><a href="https://github.com/huggingface/nanotron/blob/03d67f2103d5be0dc15ea6022a6cf16d6a633064/examples/moe/moe.py#L100">nanotron/examples/moe/moe.py at 03d67f2103d5be0dc15ea6022a6cf16d6a633064 · huggingface/nanotron</a>：极简的大语言模型 3D 并行训练 - huggingface/nanotron
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1276646337525579816)** (131 messages🔥🔥): 

> - `GNN Research`
> - `Chinchilla Scaling Laws`
> - `ARC Research`
> - `SAE Papers`
> - `GPT4o` 


- **GNNs: 重连与投影 (Rewiring & Projection)**：一位成员将 GNN 研究与位置嵌入 (positional embeddings) 的演进进行了类比，从随机游走嵌入 (random walk embeddings) 开始，逐步发展到图重连 (graph rewiring) 和投影。
   - 这种类比表明，未来的进展可能涉及从潜表征 (latent representations) 中推断位置嵌入，类似于不带位置嵌入的 Transformer 架构。
- **Chinchilla Scaling Laws 受到质疑**：一位用户询问了一篇批评 Chinchilla Scaling Laws 的论文，特别是对其准确性提出了质疑。
   - 大家的共识是，Redwood Research 的 GPT-4o 以及另一个基于 YK 服务器的项目（专注于使用合成任务数据集训练小型 LM）是目前该领域的主要竞争者。
- **ARC 研究：领先方法与社区**：讨论了 ARC 研究的最前沿方法，重点提到了 Redwood Research 的 GPT-4o 和一个基于 YK 的项目。
   - 成员们指向了一个社区项目线程和官方 Discord 服务器以获取 ARC 研究的更新，并强调了研究人员分享信息的开放态度和意愿。
- **寻找 SAE 论文**：一位用户请求关于 SAE (Sparse AutoEncoder) 技术的论文，特别是 OpenAI 和 Anthropic 用于提取可解释特征的技术。
   - 几条回复提供了相关论文的链接，包括 OpenAI 关于概念提取的一篇，以及 transformer-circuits.pub 上关于扩展单语义性 (scaling monosemanticity) 的另一篇。
- **GPT4o：架构与能力**：关于 GPT4o 背后架构的讨论兴起，有人推测它可能是一个 Transformer。
   - 虽然 GPT4o 的架构尚未公开，但内部人士暗示它是一个跨域迁移能力有限的 VLM (Vision-Language Model)，这意味着它在现有开源模型的基础上并没有革命性的进步。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/NousResearch/status/1828121648383566270">Nous Research (@NousResearch) 的推文</a>：如果你能利用世界上所有的计算能力来训练一个共享的开源 AI 模型会怎样？初步报告：https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO...</li><li><a href="https://arxiv.org/abs/2408.12528">Show-o: One Single Transformer to Unify Multimodal Understanding and Generation</a>：我们提出了 Show-o，一个统一多模态理解与生成的统一 Transformer。与完全自回归模型不同，Show-o 统一了自回归和（离散）扩散建模...</li><li><a href="https://arxiv.org/abs/2408.09624">Attention is a smoothed cubic spline</a>：我们强调了一个可能重要但迄今为止尚未被观察到的见解：Transformer 中的 Attention 模块是一个平滑的三次样条（smoothed cubic spline）。从这个角度看，这个神秘但关键的组件...</li><li><a href="https://x.com/XingyouSong/status/1826554454084333723">Richard Song (@XingyouSong) 的推文</a>：Google 如何优化其研究和系统？我们揭示了 Vizier Gaussian Process Bandit 算法背后的秘密，这是一种已被运行数百万次的黑盒优化器！论文：h...</li><li><a href="https://openreview.net/group?id=ICML.cc/2024/Workshop/MI">ICML 2024 Workshop MI</a>：ICML 2024 Workshop MI 的 OpenReview 主页</li><li><a href="https://arxiv.org/abs/2404.19737">Better &amp; Faster Large Language Models via Multi-token Prediction</a>：诸如 GPT 和 Llama 之类的 LLM 是通过 next-token prediction 损失进行训练的。在这项工作中，我们建议训练语言模型一次预测多个未来的 Token 会产生更强...</li><li><a href="https://arxiv.org/abs/2108.08481">Neural Operator: Learning Maps Between Function Spaces</a>：神经网络的经典发展主要集中在学习有限维欧几里得空间或有限集之间的映射。我们提出了一种神经网络的泛化，用于学习...</li><li><a href="https://zongyi-li.github.io/blog/2020/fourier-pde/">Zongyi Li | Fourier Neural Operator</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=hm2IJSKcYvo">Unveiling of Moshi: the first voice-enabled AI openly accessible to all.</a>：在短短 6 个月内，Kyutai 研究实验室的一个 8 人团队从头开发了一个具有前所未有的语音能力的 AI 模型，名为 Moshi。这种新型...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1277612867314778133)** (2 条消息): 

> - `Scaling Laws`
> - `Model Scaling Limits` 


- **Scaling Laws 论文：是否存在极限？**：一位成员请求推荐一些展示 Scaling Laws 在某些特定属性下失效案例的论文。
   - 另一位成员提供了一篇由 Google AI 研究团队撰写的题为 [Are Scaling Laws Failing?](https://arxiv.org/abs/2306.09479) 的论文链接，认为这是一个相关的资源。
- **Scaling Laws 不起作用？**：一位成员讨论了 Scaling Laws 以及它们在实践中并不总是有效的情况。
   - 他们分享了一些 Scaling Laws 无法奏效的具体案例，例如在缺乏数据或计算资源的情况下。



**提到的链接**：<a href="https://arxiv.org/abs/2306.09479">Inverse Scaling: When Bigger Isn&#39;t Better</a>：关于 Scaling Laws 的研究发现，大型语言模型（LMs）在增加规模（模型大小、训练数据和计算量）时，其整体损失（loss）表现出可预测的改进。在这里，我们展示了……的证据。

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1276631318553432216)** (17 条消息🔥): 

> - `OpenAI ChatGPT4o and Anthropic external APIs multiple choice eval`
> - `llama 3.1 eval`
> - `lm-eval library output_type`
> - `lm-eval library doc_to_choice`
> - `lm-eval library chat_template` 


- **使用 ChatGPT4o/Anthropic 评估多选题**：一位用户询问是否有人成功使用 ChatGPT4o 或 Anthropic 的外部 API 来评估多选题，特别是在 OpenAI 的 'multiple_choice' `output_type` 上下文中。
   - 讨论探讨了在 OpenAI 和 Anthropic 上使用 'generate_until' `output_type` 的可能性，因为它们不为 'multiple_choice' 提供 `loglikelihoods`。
- **Llama 3.1 评估**：一位用户提到 [Meta-Llama 3.1 evals](https://huggingface.co/datasets/meta-llama/Meta-Llama-3.1-8B-Instruct-evals/viewer/Meta-Llama-3.1-8B-Instruct-evals__mmlu__details) 数据集作为多选题评估的参考点，特别强调了适配后的 'mmlu' 和 'arc_challenge' 任务。
- **lm-eval 库的 'output_type' 设置**：一位用户质疑在 YAML 配置文件中将 'output_type' 更改为 'generate_until' 是否是针对 OpenAI 和 Anthropic 的可行解决方案，因为它们缺乏 'multiple_choice' 所需的 `loglikelihoods`。
   - 该用户提议效仿 Llama 3.1 evals 的方法，将选项响应包含在 YAML 配置的 'doc_to_text' 部分中。
- **lm-eval 库的 'chat_template' 设置**：一位用户询问了在 lm-eval 库中使用 `chat_template` 参数的情况，特别是在处理 Instruct 模型并尝试评估多选题任务时。
   - 有建议称，设置 `chat_template=None` 可能会解决在使用默认 `chat_template` 设置时遇到的潜在问题。
- **lm-eval 库的默认 Temperature 设置**：讨论深入探讨了 lm-eval 库中针对 OpenAI 模型的默认 `temperature` 设置，共识是通常将其设置为 0。
   - 这一做法符合在 lm-eval 框架内对大多数任务使用 `temperature` 0 的通用惯例，这在某些基准测试的 YAML 配置中通常会显式设置。



**提到的链接**：<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/aab42ba836b4af28cc1c5c1e697ea334c6ea7ced/lm_eval/evaluator.py#L295).">lm-evaluation-harness/lm_eval/evaluator.py at aab42ba836b4af28cc1c5c1e697ea334c6ea7ced · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。- EleutherAI/lm-evaluation-harness

  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1277690840617058305)** (3 条消息): 

> - `GPU Utilization Measurement`
> - `NVIDIA-SMI Overreporting`
> - `DCGM-Exporter`
> - `PyNVML`
> - `TFLOPs and HFU/MFU Logging` 


- **NVIDIA-SMI 过度报告 GPU 利用率**：一位成员询问了在训练运行期间测量 GPU 利用率的最佳工具，并指出 [NVIDIA-SMI 经常过度报告利用率](https://arthurchiao.art/blog/understanding-gpu-performance/#25-two-metric-sources-nvml--dcgm)。
   - 他们提到使用 **NVIDIA-SMI power util** 作为一个简单粗暴的选择，但前提是你信任散热系统。
- **通过 TFLOPs 和 HFU/MFU 日志记录实现准确的 GPU 利用率**：一位成员建议将每个迭代的 **TFLOPs** 或 **HFU/MFU** 记录到 **WandB** 或控制台，作为一种更准确的方法。
   - 他们解释说，这涉及将计算添加到 logger 中，并准确确定模型的**每迭代 FLOPs**，并提供了 [EleutherAI cookbook](https://github.com/EleutherAI/cookbook) 的链接作为指导。
- **用于 GPU 利用率的 PyTorch Profiler**：[PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html) 提供准确的 GPU 利用率并显示 Tensor Core 占用情况。
   - 然而，它会引入开销，因此该成员建议仅在每次主要运行开始时分析约 10 个迭代，希望它们能代表整体性能。



**提及的链接**：<a href="https://github.com/EleutherAI/cookbook">GitHub - EleutherAI/cookbook: Deep learning for dummies. All the practical details and useful utilities that go into working with real models.</a>：深度学习傻瓜书。包含处理真实模型时的所有实践细节和有用工具。- EleutherAI/cookbook

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1276622907237339218)** (54 条消息🔥): 

> - `BERTopic`
> - `OpenAI's sidebars`
> - `GPTs Agents`
> - `OpenEmpathic project`
> - `Model Merging` 


- **Hermes 2.5 击败 Hermes 2**：在添加了[代码指令示例](https://link.to.examples)后，**Hermes 2.5** 在各种基准测试中的表现似乎优于 **Hermes 2**。
   - Hermes 2 在 MMLU 基准测试中得分为 **34.5**，而 Hermes 2.5 得分为 **52.3**。
- **Mistral 在扩展超过 8k 方面存在困难**：成员们表示，如果不进行持续预训练，**Mistral** 无法扩展到 8k 以上，[这是一个已知问题](https://link.to.issue)。
   - 他们指出，*mergekit* 和 *frankenMoE finetuning* 的进一步工作是性能的下一个前沿。
- **BERTopic 使用讨论**：一位成员讨论了 **BERTopic**，这是一种易于解释的主题建模工具。
   - 他们分享了关于[可视化数据集](https://github.com/YoungPhlo/visualizing-datasets-ai-engineer-fall-2023)的项目，并发现 **BERTopic** 是一个端到端的实现。他们还回忆起之前询问过是否有人将其用于聚类。
- **模型合并策略讨论**：一位成员建议将 **UltraChat** 和基础 **Mistral** 之间的差异应用于 **Mistral-Yarn**，作为一种潜在的合并策略。
   - 其他人表示怀疑，但该成员保持乐观，并引用了过去在他们所谓的“诅咒模型合并 (cursed model merging)”方面的成功尝试。
- **Open Empathic 项目寻求帮助**：一位成员呼吁帮助扩大 **Open Empathic** 项目的类别，特别是在低端部分。
   - 他们分享了一个 [Open Empathic 发布与教程的 YouTube 视频](https://youtu.be/GZqYr8_Q7DE)，指导用户从 YouTube 视频中贡献他们喜欢的电影场景，以及 [OpenEmpathic 项目本身](https://dct.openempathic.ai/)的链接。 


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/XingyouSong/status/1826554454084333723">来自 Richard Song (@XingyouSong) 的推文</a>：Google 如何优化其研究和系统？我们揭示了 Vizier Gaussian Process Bandit 算法背后的秘密，这是一个已被运行数百万次的黑盒优化器！论文：h...</li><li><a href="https://x.com/MikeShou1/status/1826854070877126720">来自 Mike Shou (@MikeShou1) 的推文</a>：在视频关键帧生成方面，Show-o 在空间上采用全注意力机制，但在时间上仍是自回归的。此外，在生成关键帧时，我们可以生成交错的文本描述，即...</li><li><a href="https://x.com/romainhuet">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://trainy.ai/blog/gpu-utilization-misleading">GPU Utilization 是一个具有误导性的指标</a>：大多数 ML 团队将 GPU Utilization 作为主要的性能指标，但我们发现这可能相当具有误导性。</li><li><a href="https://x.com/swyx/status/1827099985944703408">来自 swyx 🇸🇬 (@swyx) 的推文</a>：我的天，@mikeshou1 的 1B 全能模型（omnimodel）在 VQA 方面似乎击败了 @ArmenAgha 的 34B Chameleon，并且在图像生成模型（如 SDXL 等）方面具有竞争力，还能为视频生成（如 Sora）生成关键帧，所有这些都在同一个...</li><li><a href="https://cohere.com/blog/combing-for-insight-in-10-000-hacker-news-posts-with-text-clustering">通过文本聚类在 10,000 篇 Hacker News 帖子中梳理洞察</a>：Hacker News 是讨论软件和初创公司话题的领先在线社区之一。我访问该网站已有十多年，一直钦佩其极高的信噪比...</li><li><a href="https://x.com/amir/status/1827007117838192699?s=46">来自 Amir Efrati (@amir) 的推文</a>：AI Agent 初创公司 Holistic ("H") 爆出“内斗”：该公司最近刚筹集了 2.2 亿美元的种子轮融资，但 5 位创始人中有 3 位已经离职。离职的创始人此前曾长期担任 Google DeepMind 研究员。h...</li><li><a href="https://x.com/vikhyatk/status/1827190823689335115">来自 vik (@vikhyatk) 的推文</a>：如果你今天还没取消 Claude 订阅，那你活该承受即将到来的世界。</li><li><a href="https://github.com/paul-gauthier/aider/releases/tag/v0.52.0?utm_source=ainews&utm_medium=email">发布 Aider v0.52.0 · paul-gauthier/aider</a>：Aider 现在支持运行 shell 命令：启动浏览器查看更新后的 html/css/js。安装新依赖。运行数据库迁移。运行程序以测试更改。运行新的测试用例。/read ...</li><li><a href="https://share.snipd.com/episode/bc246320-a849-4718-9fc8-be0f4290aaf0">20VC：芯片、模型还是应用；AI 的价值在哪里 | 算力是解决所有模型性能问题的答案吗 | 为什么 OpenAI 搁置了 AGI 以及在 OpenAI 价格倾销的情况下模型是否还有价值，访谈嘉宾 Aidan Gomez，Cohere 联合创始人</a>：20VC：芯片、模型还是应用；AI 的价值在哪里 | 算力是解决所有模型性能问题的答案吗 | 为什么 OpenAI 搁置了 AGI 以及...</li><li><a href="https://x.com/drjimfan/status/1827116592951652823?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Jim Fan (@DrJimFan) 的推文</a>：Transformer 领域和 Diffusion 领域已经分离太久了。之前有很多统一的尝试，但都失去了简洁和优雅。是时候通过 Transfusion 🩸 来注入新活力，实现融合...</li><li><a href="https://youtu.be/L0kBWyziFlc">利用自主 Agent 颠覆 15 万亿美元的建筑行业：Sarah Buchner 博士</a>：Trunk Tools 创始人兼 CEO Sarah Buchner 博士展望了建筑行业的未来，届时将有一支 AI Agent 大军代表我们的用户工作。我们目前...</li><li><a href="https://github.com/YoungPhlo/visualizing-datasets-ai-engineer-fall-2023">GitHub - YoungPhlo/visualizing-datasets-ai-engineer-fall-2023: 可视化数据集：在将文本数据用于下游任务之前，开启其视觉视角</a>：可视化数据集：在将文本数据用于下游任务之前，开启其视觉视角 - YoungPhlo/visualizing-datasets-ai-engineer-fall-2023</li><li><a href="https://github.com/MaartenGr/BERTopic">GitHub - MaartenGr/BERTopic: 利用 BERT 和 c-TF-IDF 创建易于解释的主题。</a>：利用 BERT 和 c-TF-IDF 创建易于解释的主题。 - GitHub - MaartenGr/BERTopic: 利用 BERT 和 c-TF-IDF 创建易于解释的主题。
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1276631012599922699)** (1 条消息): 

> - `AI Engineer London Meetup`
> - `AI Engineer World's Fair`
> - `Meetup 演讲者` 


- **AI Engineer London Meetup 宣布举办**：伦敦将举办一场新的 AI Engineer Meetup，由 <@dctanner> 主办，灵感源自 @swyx 的 AI Engineer World's Fair。
   - 首届 Meetup 将于 9 月 12 日晚举行，届时将邀请四位演讲者：@maximelabonne、@roviosc、@BruverisMartins 和 Chris Bull。
- **Meetup 注册链接**：公告中包含了 AI Engineer London Meetup 的注册链接，预计该活动将非常受欢迎。



**提到的链接**：<a href="https://x.com/dctanner/status/1827071893448618453?s=46">来自 Damien C. Tanner (@dctanner) 的推文</a>：我们将 @swyx 的 AI Engineer World's Fair 的一部分带到了伦敦！9 月 12 日晚是首届 AI Engineer London Meetup。来听听 4 位优秀的演讲者的分享：@maximelabonne, @rovio...

  

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1276630411149049998)** (78 条消息🔥🔥): 

> - `Structured Outputs`
> - `LLM Deduplication`
> - `TaxonomySynthesis`
> - `BERTopic`
> - `GPT Researcher` 


- **Structured Outputs 与 grammar constrained sampling 不同**：一位成员讨论了在 [Grammar Constrained Sampling](https://www.microsoft.com/en-us/research/publication/grammar-constrained-sampling-for-generative-language-models/) 的背景下，使用 [Structured Outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) 与 [Function Calling](https://platform.openai.com/docs/guides/functions/how-to-use-functions) 之间的区别。
- **去重与主题相似度**：成员们讨论了如何处理由 LLM 生成的重复或极度相似的主题，特别是在处理数千个主题时。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://maartengr.github.io/BERTopic/">BERTopic</a>：未找到描述</li><li><a href="https://medium.com/@juanc.olamendy/revolutionizing-retrieval-the-mastering-hypothetical-document-embeddings-hyde-b1fc06b9a6cc">Revolutionizing Retrieval: The Mastering Hypothetical Document Embeddings (HyDE)</a>：在检索增强生成 (RAG) 系统的动态领域中，一个令人困惑的问题经常浮现：如何构建一个……</li><li><a href="https://maartengr.github.io/BERTopic/algorithm/algorithm.html#5-topic-representation">The Algorithm - BERTopic</a>：未找到描述</li><li><a href="https://www.figma.com/board/J19T0RN1Hvi1ajDlUtIvOc/Generative-Classifier?node-id=2-1801&t=Km2ND86IeNkD92WJ-1">Figma</a>：使用 FigJam 创建</li><li><a href="https://github.com/danielgross/embedland/blob/main/bench.py#L281">embedland/bench.py at main · danielgross/embedland</a>：文本嵌入实验集合。通过在 GitHub 上创建账号为 danielgross/embedland 做出贡献。</li><li><a href="https://github.com/CakeCrusher/TaxonomySynthesis">GitHub - CakeCrusher/TaxonomySynthesis: An AI-driven framework for synthesizing adaptive taxonomies, enabling automated data categorization and classification within dynamic hierarchical structures.</a>：一个用于合成自适应分类体系的 AI 驱动框架，支持在动态层级结构中进行自动数据分类。 - CakeCrusher/TaxonomySynthesis</li><li><a href="https://github.com/assafelovic/gpt-researcher">GitHub - assafelovic/gpt-researcher: LLM based autonomous agent that does online comprehensive research on any given topic</a>：基于 LLM 的自主 Agent，可针对任何给定主题进行在线综合研究 - assafelovic/gpt-researcher</li><li><a href="https://github.com/stanford-oval/storm">GitHub - stanford-oval/storm: An LLM-powered knowledge curation system that researches a topic and generates a full-length report with citations.</a>：一个由 LLM 驱动的知识策展系统，可研究特定主题并生成带有引用的完整报告。 - stanford-oval/storm</li><li><a href="https://github.com/langchain-ai/langgraph/blob/main/examples/storm/storm.ipynb">langgraph/examples/storm/storm.ipynb at main · langchain-ai/langgraph</a>：将弹性语言 Agent 构建为图。通过在 GitHub 上创建账号为 langchain-ai/langgraph 做出贡献。</li><li><a href="https://feedly.com/">Feedly: Track the topics and trends that matter to you</a>：市场领先的主题监控解决方案</li><li><a href="https://exa.ai/">Exa</a>：Exa API 从网络检索最佳实时数据以补充你的 AI</li><li><a href="https://notebooklm.google.com/">无标题</a>：未找到描述</li><li><a href="https://app.tavily.com/chat">Tavily AI</a>：未找到描述</li><li><a href="https://consensus.app/">Consensus AI-powered Academic Search Engine</a>：Consensus 是一种新型学术搜索引擎，由 AI 驱动，以科学为基础。在获取即时见解和主题合成的同时查找最佳论文。</li><li><a href="https://elicit.com/">Elicit: The AI Research Assistant</a>：使用 AI 搜索、总结、提取数据并与超过 1.25 亿篇论文进行对话。已被学术界和工业界超过 200 万名研究人员使用。</li><li><a href="https://storm.genie.stanford.edu/">无标题</a>：未找到描述</li><li><a href="https://gptr.dev/">GPT Researcher - Official Page</a>：你的 AI 助手，用于对任何给定任务进行快速洞察和深入研究</li><li><a href="https://langchain-ai.github.io/langgraph/">🦜🕸️LangGraph</a>：未找到描述
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[announcements](https://discord.com/channels/1068976834382925865/1069236008115253348/)** (1 条消息): 

georgehotz: 在这里购买你的 tinybox https://tinycorp.myshopify.com/
  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1277323878489653261)** (78 messages🔥🔥): 

> - `E-graphs`
> - `Tinygrad`
> - `Tinybox`
> - `AMD`
> - `BERT` 


- **E-graph 性能担忧**：一位成员分享了他们的观点，认为 E-graph 重写在处理大规模搜索空间方面落后于目前的 SAT 求解器。
- **Tinybox 销售启动！**：Tinybox 工厂现已全力运转，销售即将向公众开放。
- **Tinybox：关于 Tinygrad 的讨论**：一位成员询问了在 Tinybox 上使用 AMD GPU 的可行性，并提到了 AMD 最近收购 Silo AI 以及他们在 AMD 硬件上训练 LLM 的成功案例。
- **Tinygrad vs Torch 用于 BERT 预训练**：一位成员表示有兴趣与 Tinygrad 合作预训练一个大型 BERT 模型，并愿意提供计算资源。
- **Tinybox 发货与供货情况**：Tinybox 目前已售罄，生产能力有限，每天约为 4 台，目前还有 60 台的积压订单。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tinycorp.myshopify.com">tiny shop</a>：tiny 商店</li><li><a href="https://en.wikipedia.org/wiki/Freight_forwarder">Freight forwarder - Wikipedia</a>：货运代理 - 维基百科</li><li><a href="https://tinycorp.myshopify.com/products/tinybox-red">tinybox red edition</a>：必须在订单确认后 5 天内完成付款以保证您的订单。仅限美国本土发货。加拿大请联系 support@tinygrad.org。付款方式：银行转账/电汇...</li><li><a href="https://tinycorp.myshopify.com/products/tinybox-green">tinybox green edition</a>：必须在订单确认后 5 天内完成付款以保证您的订单。仅限美国本土发货。加拿大请联系 support@tinygrad.org。付款方式：银行转账/电汇...</li><li><a href="https://x.com/__tinygrad__/status/1828140019577704652">来自 tiny corp (@__tinygrad__) 的推文</a>：工厂已全力运转！如果您有 tinybox 预订但尚未收到联系，请联系 support@tinygrad.org。销售将很快向公众开放。
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1276642882127794187)** (13 messages🔥): 

> - `GPU=1 and OpenCL`
> - `Tensor Loading`
> - `Training Speed and Data Types`
> - `Performance Differences Between dtypes.half and dtypes.float`
> - `RoPE Approximation and Model Matching` 


- **GPU=1 未使用 GPU**：一位使用 3060 GPU 的用户想知道在使用 `GPU=1` 时如何查看对 GPU 的调用。
   - 另一位用户解释说 `GPU=1` 指的是 OpenCL，而该用户可能安装了 `pocl`，它只利用 CPU。
- **加载保存的 Tensor**：一位用户询问如何加载之前使用 `nn.state.safe_save` 保存的 Tensor。
   - 提供的代码片段演示了使用 `nn.state.safe_save({'t':t}, "test.safetensor")` 将 Tensor 保存到 `test.safetensor` 文件中。
- **移除 Cast 后训练速度提升**：一位用户报告称，在 `beautiful_cifar` 示例的预处理中移除 `.cast(dtypes.default_float)` 调用后，训练速度（GFLOPS）提升了 25%。
   - 用户注意到，在进入模型之前，`X.dtype` 现在是 `dtype.float` 而不是之前的 `dtype.half`。
- **dtypes.half 与 dtypes.float 之间的性能差异**：一位用户观察到，与 `dtypes.float` 相比，`dtypes.half` 的 `._data()` 和 `.realize()` 速度更慢。
   - 这被归因于 `dtypes.half` 的 Lowering（下放）速度更快，但执行速度更慢。
- **RoPE 近似与模型匹配**：一位用户质疑 RoPE 中使用 sin/cos 的近似是否本质上意味着模型将无法完美匹配参考实现。
   - 这个问题尚未得到解答。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.tinygrad.org/nn/#tinygrad.nn.state.safe_load">nn (Neural Networks) - tinygrad docs</a>：tinygrad 文档</li><li><a href="https://github.com/tinygrad/tinygrad/blob/1b4ad982e58c88c858e731b96a6ed3f2eef1b6b7/examples/beautiful_cifar.py#L91">tinygrad/examples/beautiful_cifar.py</a>：你喜欢 PyTorch？你喜欢 Micrograd？你会爱上 Tinygrad！❤️ - tinygrad/tinygrad
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1276676918690975885)** (37 messages🔥): 

> - `Command-R pricing` (Command-R 定价)
> - `Durov's trial` (Durov 的审判)
> - `Command-R free trial` (Command-R 免费试用)


- **Command-R 模型更新**：发布了一个新的 Command-R 模型，但目前还没有官方公告。
   - 关于定价、上下文窗口（context window）、微调（fine-tuning）和其他功能的问题仍未得到解答。
- **Durov 备受争议的法国公民身份**：Telegram 创始人 Pavel Durov 获得了法国国籍，目前正在法国面临审判。
   - 一些人猜测他是故意寻求监禁，而另一些人则认为这是为了获得国际关注的战略举措。
- **Command-R 免费试用限制**：在用户达到试用 Key 的速率限制（rate limits）之前，Command-R 可以免费使用。
   - 这是用户强烈要求的更改，最近已通过电子邮件宣布。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/0001-gif-17282391190974969363">0001 GIF - 0001 - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/evannie-gif-13360603728069224276">Evannie GIF - Evannie - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://x.com/1vnzh/status/1827690558799695949">Ivan Zhang (@1vnzh) 的推文</a>：暂停 AI 去玩《黑神话：悟空》</li><li><a href="https://x.com/MyLordBebo/status/1827585606437765568">Lord Bebo (@MyLordBebo) 的推文</a>：Durov：获得法国国籍只是为了进监狱？在 Tucker 的采访中，这家伙似乎在谈论北约情报部门如何向他施压以获取聊天记录……</li><li><a href="https://x.com/SouthDallasFood/status/1827443354948579745">South Dallas Foodie (@SouthDallasFood) 的推文</a>：Big Parma 真正的守门人
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1276690811475460196)** (9 messages🔥): 

> - `Cohere free trial` (Cohere 免费试用)
> - `Invite Feature Bug` (邀请功能 Bug)
> - `Fine Tuning Cohere Embeddings` (微调 Cohere Embeddings)
> - `Rasa Chatbot` (Rasa 聊天机器人)


- **用于 Rasa 聊天机器人的 Cohere 免费试用**：一位来自孟加拉国的用户询问如何使用 Cohere 的免费试用来构建基于 Rasa 的聊天机器人。
   - 他们之前尝试过使用 OpenAI，但被 5 美元的费用劝退，现在正在 Cohere 寻找类似的免费试用选项。
- **邀请功能 Bug**：一位用户报告了邀请成员功能的 Bug，称尽管拥有 Gmail 账户，但仍无法加入团队。
   - 他们请求协助解决此问题，因为他们无法加入受邀的团队。
- **无需 API Key 微调 Cohere Embeddings**：一位用户询问是否可以在没有 Cohere API Key 的情况下，使用领域数据集微调 `embed-multilingual-v3.0`。
   - 回复明确指出，微调时始终需要 API Key 才能与 Cohere 的端点（endpoints）进行通信。
- **免费访问的试用 Key**：一位用户询问了用于访问 Cohere 服务的免费试用 Key 的可用性。
   - 他们被告知试用 Key 是完全免费的，但有速率限制，用户可以在准备好投入生产环境时升级以获得更高的限制。


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1276680717727694878)** (32 messages🔥): 

> - `Cohere API Rate Limits` (Cohere API 速率限制)
> - `Multiple API Keys` (多个 API Key)
> - `Async Requests` (异步请求)
> - `Rerank 3 Pricing` (Rerank 3 定价)
> - `Token Limits` (Token 限制)


- **Cohere API 速率限制更新**：一位用户报告称，在同时使用多个 API Key 时遇到了 "too many requests" 错误，即使他们遵循了文档中提到的每分钟 10,000 次请求的限制。
   - Cohere 团队确认了最近对速率限制的更改，现在每个用户在所有组织 API Key 中的总调用次数限制为每分钟 1,000 次，并解释说这是针对每个用户而非每个 API Key 的。
- **异步请求与速率限制**：用户在使用多个 API Key 进行异步调用时遇到了 "too many requests" 错误，即使每个 Key 的调用量都低于 1000 次/分钟。
   - Cohere 团队澄清说，速率限制现在是按用户计算的，而不是按 API Key 计算的，这意味着每分钟 1,000 次调用的限制适用于单个用户在组织内使用的所有 API Key。
- **Rerank 3 定价细分**：一位用户询问了使用 Rerank 3 模型的成本，特别是询问 1,000 次搜索 2 美元是指 1,000 次 API 调用还是其他含义。
   - Cohere 团队澄清说，每次包含最多 100 个待重排序文档的查询计为一次搜索，这意味着 1,000 次搜索最多可以处理 409,600,000 个 tokens，因为每个文档最多可以包含 4,096 个 tokens。



**提到的链接**：<a href="https://cohere.com/pricing">Pricing</a>：直接通过我们的 API 访问模型，以创建可扩展的生产工作负载。   

  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1276648997401923605)** (5 条消息): 

> - `LlamaIndex`
> - `Create Llama`
> - `GraphRAG`
> - `Data Silos`
> - `LLMs for newsletters` 


- **Create Llama 获得结构化提取模板**：[Create Llama](https://t.co/G95ReAuRS6) 工具现在新增了结构化提取模板。
- **GraphRAG 教程系列开启**：关于构建 [GraphRAG](https://t.co/7fLocjRvdN) 的分步教程系列已经开始。
   - 第一段视频重点介绍了使用内存实现（in-memory implementation）来构建核心组件，并涵盖了使用 LLM 提取实体和关系的内容。
- **解决企业级 LLM 开发中的数据孤岛问题**：企业级 LLM 开发面临着数据孤岛（Data Silos）的挑战，每个新数据源都需要独立的身份验证管理，且知识分散在各个团队中。
   - LlamaIndex 正在探索解决这一问题的方法。
- **用于 Newsletter 自动化的 LLM**：LlamaIndex 的 Newsletter 旨在总结每周推文，以前制作起来非常耗时。
   - 现在使用 LLM 来自动化这一过程，展示了 LLM 在内容创作方面的强大能力。
- **RAG-a-thon 黑客松活动公布**：与 Pinecone 合作举办的第二届 [RAG-a-thon](https://t.co/IFvyW5QB6r) 定于 10 月 11 日至 13 日在 Palo Alto 举行。
   - 此次黑客松提供超过 7000 美元的现金奖励，将在 500 Global VC 办公室举行。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1276619666759876752)** (71 条消息🔥🔥): 

> - `tqdm`
> - `LlamaIndex API changes`
> - `PineconeVectorStore`
> - `LlamaIndex and OpenAI models`
> - `Updating VectorStoreIndex` 


- **tqdm - Python 必备进度条**：一位成员对 `tqdm` 在 Python 进度条中的日益普及表示兴奋。
   - 他们表示这*基本上*是他们使用的唯一进度条。
- **LlamaIndex API 变更：ServiceContext 已弃用**：一位用户对 LlamaIndex 0.11.1 版本中移除 `ServiceContext` 表示沮丧。
   - 该用户强调，他们没有收到关于此变更的警告，许多用户可能会因此离开 LlamaIndex。
- **PineconeVectorStore 兼容性问题**：一位成员在遇到错误后询问了 LlamaIndex 与 PineconeVectorStore 之间的兼容性。
   - 解决方案只是更新 `llama-index-vector-stores-pinecone` 包。
- **在 LlamaIndex 中使用最新的 OpenAI 模型**：一位用户询问如何设置 LlamaIndex 查询引擎以使用最新的 OpenAI 模型。
   - 该用户被引导使用 `Settings.llm = OpenAI(model="gpt-4o-mini")` 全局设置默认 LLM。
- **使用新文档更新 VectorStoreIndex**：一位用户询问如何通过添加带有 Embedding 的文档来更新 VectorStoreIndex。
   - 建议是使用 `insert_nodes` 方法，并确保文档已正确解析和嵌入。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="http://",">未找到标题</a>: 未找到描述</li><li><a href="https://",">未找到标题</a>: 未找到描述</li><li><a href="https://discordapp.com/channels/1059199217496772688/1277285749078753325">Discord - 充满乐趣与游戏的群聊</a>: Discord 是玩游戏、与朋友放松甚至建立全球社区的绝佳场所。定制你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://postgres.new/">Postgres Sandbox</a>: 带有 AI 辅助的浏览器内 Postgres 沙盒</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/service_context_migration/">从 ServiceContext 迁移到 Settings - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/">LlamaIndex - LlamaIndex</a>: 未找到描述</li><li><a href="https://postgresml.org/docs/open-source/pgml/guides/embeddings/in-database-generation">数据库内 Embedding 生成 – PostgresML</a>: 使用开源 Postgres 扩展，仅通过 SQL 即可训练和部署模型以进行在线预测。</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/#document-management">Ingestion Pipeline - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/79518c4cc39981140b2e87c9a701b09d74d47e9a/llama-index-core/llama_index/core/base/embeddings/base.py#L37">run-llama/llama_index 中的 llama-index-core/llama_index/core/base/embeddings/base.py</a>: LlamaIndex 是适用于你的 LLM 应用程序的数据框架 - run-llama/llama_index
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1277385972711030900)** (60 messages🔥🔥): 

> - `Eager vs compile discrepancy` (Eager 与 compile 的差异)
> - `Triton RNG`
> - `In-place operations and Torch.compile` (In-place 操作与 Torch.compile)
> - `KV cache allocation` (KV cache 分配)
> - `Cudagraphs and memory consumption` (Cudagraphs 与内存消耗)


- **编译后的函数输出与 Eager 模式不同**：一位成员询问为什么即使使用相同的 seed，编译后的函数与非编译版本相比会产生不同的输出。
   - 这种差异源于 Triton 的 RNG（用于编译后的代码）与 PyTorch 的 RNG（用于 Eager 模式）之间的区别，以及可能不适合编译的 In-place 操作。
- **Torch.compile 中的 In-place 操作**：讨论指出，像 `scatter_` 这样的 In-place 操作在使用 `torch.compile` 时通常表现不佳。
   - 这是因为编译后的代码处理 In-place 操作的方式可能不同，导致内存消耗增加，并可能产生不同的输出。
- **Cudagraphs 与内存**：建议使用 Cudagraphs 进行调试。
   - 然而，也有人提到 Cudagraphs 有时会因为预分配缓冲区而导致更高的内存消耗，这在某些情况下可能是不希望看到的。
- **用于推理的 FP16**：建议使用 FP16 代替 FP32 进行推理，以减少内存使用，特别是对于不支持 BF16 的硬件。
   - 这种方法有助于解决显存溢出（OOM）问题，但编译版本和非编译版本之间的输出差异仍然存在。
- **编译后 Kernel 的数值差异**：据推测，编译后代码与非编译代码之间剩余的输出差异可能是由于编译后的 Kernel 存在数值差异。
   - 这表明即使优化了内存使用，编译代码执行的计算仍可能存在细微变化。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/pytorch-labs/gpt-fast/blob/bc04265df30c7b927d38f34198e0e33a63cb893e/model.py#L80-L89">gpt-fast/model.py at bc04265df30c7b927d38f34198e0e33a63cb893e · pytorch-labs/gpt-fast</a>: 在不到 1000 行 Python 代码中实现简单高效的 PyTorch 原生 Transformer 文本生成。 - pytorch-labs/gpt-fast</li><li><a href="https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L401">pytorch/torch/_inductor/config.py at main · pytorch/pytorch</a>: Python 中具有强大 GPU 加速能力的 Tensor 和动态神经网络 - pytorch/pytorch</li><li><a href="https://github.com/pytorch-labs/gpt-fast/blob/bc04265df30c7b927d38f341">GitHub - pytorch-labs/gpt-fast at bc04265df30c7b927d38f34198e0e33a63cb893e</a>: 在不到 1000 行 Python 代码中实现简单高效的 PyTorch 原生 Transformer 文本生成。 - GitHub - pytorch-labs/gpt-fast at bc04265df30c7b927d38f34198e0e33a63cb893e
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1276777519349239930)** (41 条消息🔥): 

> - `LangChain Document Loading`
> - `LangChain Model Usage`
> - `LLMChain vs LCEL`
> - `LangChain with Postgres`
> - `GenAI in Data Science` 


- **LangChain Document Loading - 提取图像**: LangChain 社区包中 `PyPDFLoader` 类的 `extract_images=True` 参数用于在加载 PDF 文档时提取其中的图像。
   - 这允许你在处理文本内容的同时处理图像，从而可能实现基于图像的分析或为你的 LLM 提供更丰富的上下文。
- **LLMChain vs LCEL**: `LLMChain` 类代表了一个你提供模型和提示词（Prompt）的链，而 `LCEL` (LangChain's Execution Language) 则提供了更多的灵活性和控制力。
   - 虽然两者的目标都是为复杂任务链接组件，但 `LLMChain` 通常经过了优化，而 `LCEL` 允许更大程度的自定义和模块化，尽管增加的控制力并不总是必要的。
- **LangChain with Postgres - PostgresSaver 错误**: 一位用户在使用 `LangGraph` 和 `PostgresSaver` 操作 Postgres 数据库时遇到了 `TypeError: tuple indices must be integers or slices, not str` 错误。
   - 该错误可能与用户如何使用字符串索引访问元组（tuple）元素有关，但需要更多信息来确定原因并提供解决方案。
- **GenAI 在数据科学中的角色**: 一位用户提出了关于生成式 AI (GenAI) 在数据科学中角色的问题，特别是在 ChatGPT 出现之后。
   - 该用户认为 GenAI 在生成代码和基础数据流水线（Data Pipelines）方面的作用有限，但其他人认为数据科学对于构建功能性的 GenAI 应用至关重要，强调了这些领域之间日益增加的重叠。
- **使用 LangChain 构建 RAG - 协作与挑战**: 一位用户正在为 LangChain 和 LangGraph 等热门包的相关文档构建 RAG (Retrieval-Augmented Generation) 聊天机器人，并正在为该项目寻找合作者。
   - 该用户发现爬虫（Scraping）和 RAG 组件具有挑战性，并愿意与对这些主题感兴趣的人合作。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/langchain-ai/langchain/issues/24987>),">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建一个账号来为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/23661>))">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建一个账号来为 langchain-ai/langchain 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1276783342792081491)** (3 messages): 

> - `AI agents`
> - `LangGraph framework`
> - `Retrieval Agents`
> - `ParDocs`
> - `RAG` 


- **AI Agents：AI 的未来**：AI Agent 对于 AI 的增长变得越来越重要，因为它们可以模仿类人属性，通过交互、推理和决策来自主实现目标。
   - 本文探讨了如何使用 [LlamaIndex](https://community.analyticsvidhya.com/c/generative-ai-tech-discussion/how-to-use-llamaindex) 和 MonsterAPI 构建 AI Agent，这些工具提供了开发 Agent 和访问 LLM API 的能力。
- **用于检索 Agent 的 LangGraph 框架**：LangGraph 框架被用于构建检索 Agent，这类 AI Agent 利用外部知识库来增强其回答。
   - 本文将演示如何使用 LangGraph 构建强大的检索 Agent，展示其从外部来源检索和处理信息的能力。
- **ParDocs：解析非结构化文档**：ParDocs 是一款旨在将 PDF 和 DOCX 等非结构化文档解析为结构化 JSON 的工具。
   - 这种结构化 JSON 使得与向量数据库（vector databases）的协作更加容易，并能从 AI Agent 获得更准确的输出。
- **用于检索增强生成的 r/RAG Subreddit**：创建 r/RAG Subreddit 是为了提供一个分享检索增强生成（RAG）相关知识和工具的社区。
   - ParDocs 的创建者对该社区内的合作和学习潜力感到兴奋，旨在推动 RAG 可能性的边界。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://medium.com/ai-artistry/building-a-retrieval-agent-with-langgraph-and-exa-9baf33e89510">Building a Retrieval Agent with LangGraph and Exa</a>: Ankush k Singal</li><li><a href="https://www.pardocs.com).">未找到标题</a>: 未找到描述</li><li><a href="https://www.analyticsvidhya.com/blog/2024/08/how-to-build-an-ai-agent-using-llama-index-and-monsterapi/">How to Build an AI Agent using Llama Index and MonsterAPI?</a>: 探索如何使用 LlamaIndex 和 MonsterAPI 构建 AI Agent，涵盖从架构到自动化等实际应用。
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1277057611975626834)** (15 messages🔥): 

> - `GPT-4 fine-tuning`
> - `Mistral`
> - `LLM Benchmarks`
> - `lm-eval-harness`
> - `Glianorex Benchmark` 


- **GPT-4 微调：一个小规模实验**：一位成员认为，尽管使用了较少的数据，但与 Mistral 相比，GPT-4 的微调效果“有点糟糕”。
- **快速创建基准测试的框架**：一位成员询问是否有可以快速创建基准测试（benchmarks）的框架。
   - 另一位成员指向了 *lm-eval-harness* 框架，该框架添加任务非常容易。
- **创建基准测试 - 一场对话**：一位成员询问创建基准测试意味着什么，以及有哪些方法。
   - 他们提到了 *lm-eval-harness* 框架以及他们自己关于生成基准测试问题的研究，该研究详见最近的一篇论文，并已在 GitHub 上发布。
- **Glianorex 基准测试：医学虚构**：一位成员分享了一篇关于评估多选题（MCQs）在医疗领域衡量 LLM 性能有效性的研究论文。
   - 他们围绕一个不存在的腺体 *Glianorex* 创建了一个虚构的医学基准测试，以将 LLM 的知识储备与其解题能力隔离开来。
- **在 *lm-eval-harness* 中添加自定义基准测试**：一位成员询问在 *lm-eval-harness* 中添加自己的基准测试是否容易。
   - 回复确认添加基准测试确实“非常容易”。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.02394">Multiple Choice Questions and Large Languages Models: A Case Study with Fictional Medical Data</a>: 像 ChatGPT 这样的 LLM 在医学领域展现出巨大潜力，通常使用类似于 USMLE 的多选题（MCQs）进行评估。尽管如此...</li><li><a href="https://github.com/maximegmd/glianorex-gen">GitHub - maximegmd/glianorex-gen</a>: 通过在 GitHub 上创建账号来为 maximegmd/glianorex-gen 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1276637711985410199)** (17 messages🔥): 

> - `LIGER`
> - `LLM Training Efficiency`
> - `Single GPU Training`
> - `Training Time` 


- **LIGER：节省 25% VRAM 和 33% 训练时间**：**LIGER** 是一种用于 **LLM training** 的新 Kernel，取得了令人印象深刻的结果：节省了 **25% VRAM** 和 **33% 训练时间**。
   - 一位用户表达了兴奋之情，称他们正在开香槟庆祝。
- **LIGER 并非专为单 GPU 设计**：一位用户询问 **LIGER** 是否能为 **single GPU training** 带来改进。
   - 另一位用户确认应该会有改进，并暗示它并非专门针对单 GPU 场景设计的。
- **训练时间和 Checkpoint 恢复**：一位用户询问了报告中提到的 **139 小时** 训练时间，但在一个小时后仅完成了 **30%**。 
   - 另一位用户建议从 **checkpoint** 恢复可能是导致训练时间显示较长的一个合理解释。



**提及链接**：<a href="https://github.com/linkedin/Liger-Kernel">GitHub - linkedin/Liger-Kernel: Efficient Triton Kernels for LLM Training</a>：用于 LLM 训练的高效 Triton Kernels。可以通过在 GitHub 上创建账户来为 linkedin/Liger-Kernel 的开发做出贡献。

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1276679771631255703)** (5 messages): 

> - `Phi-3-medium-128k-instruct training config`
> - `training config for new tokens` 


- **寻求 Phi-3-medium-128k-instruct 训练配置**：一位用户询问是否有 **Phi-3-medium-128k-instruct** 模型的训练配置可用。
   - 他们特别询问是否有人训练过该模型并能分享他们的训练配置。
- **澄清 Tokenizer 训练范围**：一位用户询问 `modules_to_save = ["lm_head", "embed_tokens"]` 配置是训练完整的 **tokenizer** 词表，还是仅训练新添加的 **tokens**。
   - 他们引用了一个 Discord 消息链接作为背景参考。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1277577357711904801)** (1 messages): 

> - `Data Curation`
> - `LLM Judge`
> - `Model Rating`
> - `Data Curation Setup` 


- **数据清洗 - 模型评分系统**：一位用户询问了数据清洗（**Data Curation**）的过程，特别是想知道是否涉及提示模型给出评分，类似于 **LLM-Judge** 系统。
   - 该用户有兴趣了解用于此清洗过程的具体设置或方法论。
- **LLM-Judge：模型评分的一个例子**：用户提到 “**LLM-Judge**” 作为利用模型评分系统的示例。
   - 该系统可能涉及提示模型对其他模型进行评估，可能基于各种标准。
- **探索数据清洗方法**：讨论集中在理解数据清洗中使用的技术和策略。
   - 用户的提问暗示了使用基于模型的方法进行数据评分的可能性，类似于 **LLM-Judge**。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1276620003424342067)** (16 messages🔥): 

> - `Mojo JIT`
> - `Mojo Script Mode`
> - `Max/Mojo Development Pace`
> - `GPU Support`
> - `Community Meeting` 


- **Mojo 的 JIT 行为解析**：一位成员询问了关于 `mojo jit` 命令的可能性，另一位成员解释说运行 `mojo main.mojo`（脚本模式）使用的是 JIT，而 `mojo build main.mojo` 则进行正式编译。
   - 这解释了为什么全局变量等特性在一种模式下有效，而在另一种模式下无效。
- **社区对开发进度的关注**：一位成员注意到 Max 和 Mojo 的博客文章及版本发布速度似乎有所放缓。
   - 他们询问这是由于夏季进度放缓、正在筹备重大发布，还是遇到了意外问题。
- **GPU 支持是重中之重**：另一位成员表示，团队似乎正在全力推进 GPU 支持的发布，暗示下一个主要版本可能会与之同步。
   - 他们还提到，团队可能希望通过此次发布将 Magic 移出 Alpha 阶段。
- **即将召开的 Max+Mojo 社区会议**：一位成员宣布下一次 Max+Mojo 社区会议定于 9 月 9 日星期一举行，并鼓励有兴趣的人士报名参加演示。
   - 他们提供了报名文档的链接，并表示愿意帮助任何有兴趣在未来会议上进行演示的人。
- **授权许可问题的专用频道**：宣布了一个新频道，用于讨论和咨询与 Max 和 Mojo 许可证相关的问题。
   - 公告强调，团队希望允许用户在 Max 和 Mojo 之上自由构建。



**提到的链接**：<a href="https://modul.ar/community-meeting-doc">[Public] Mojo Community Meeting</a>：Mojo 社区会议。此文档链接：https://modul.ar/community-meeting-doc。这是一个公开文档；欢迎所有人查看并发表评论/建议。所有会议参与者必须遵守...

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1276622632325746688)** (18 messages🔥): 

> - `Mojo's memory usage`
> - `Modverse 42 release schedule`
> - `Mojo's struct parameters`
> - `Mojo's UnsafePointer` 


- **Mojo 内存使用量测量**：一位成员询问是否有办法测量 Mojo 中的内存使用情况，并指出 `benchmark` 命令仅打印执行时间。
   - 另一位成员解释说，目前唯一有效的方法是询问操作系统，因为 Mojo 的 Runtime 还没有足够的统计能力，特别是在涉及外部分配器或 mmap 的情况下。
- **Modverse 42 发布计划**：一位成员询问上周为何没有发布 Modverse 42。
   - 另一位成员回答说，发布频率通常为 1-3 周，具体取决于新项目和内容的数量，并且 `Weekly` 标签可能会被移除或重命名。
- **Mojo 的 Struct 参数详解**：一位成员分享了在 struct 中使用代码的示例，但遇到了错误：`cannot implicitly convert 'ClassStruct["x", "y", "add"]' value to 'ClassStruct[]'`。
   - 另一位成员解释说 `ClassStruct` 具有 `variadic parameters`（可变参数），需要从外部 struct 进行参数化。参数需要在程序运行前确定，因此 struct 字段可以从 struct 参数或全局 `alias` 进行参数化。
- **Mojo 的 UnsafePointer 和 Optional**：一位成员询问在使用了 `self` 但导致错误 `use of unknown declaration 'self'` 的情况下，参数值应该是什么。
   - 另一位成员回答说，在这个例子中，Node 实际上并不拥有 next 内部引用的所有权。如果使用引用，struct 上的参数需要委托所有权。他们还建议使用 `UnsafePointer`，因为 Mojo 目前还没有安全的、拥有所有权的指针（owning pointer）。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/)** (1 messages): 

guidorice: 不确定是否有人分享过这个：https://youtu.be/7TnkqfX84gI?si=sqHI4BLGOwneLarH 👍
  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1276951510378614824)** (23 条消息🔥): 

> - `OpenInterpreter Profiles`
> - `OpenInterpreter --vision 在 Windows 上`
> - `OpenInterpreter OS Mode`
> - `预构建的 OpenInterpreter`
> - `OpenInterpreter 文档` 


- **OpenInterpreter 自定义配置文件路径**：一位成员询问是否可以为 OpenInterpreter 配置文件设置自定义路径。
   - 开发者回应称目前尚不可能，但未来可能会作为一项设置来实现。
- **OpenInterpreter --vision 在 Windows 上**：一位用户询问了 `--vision` 标志在 Windows 上的功能。
   - 回复是它应该可以在 Windows 上运行，但如果遇到任何问题，建议去专门的频道进行处理。
- **用于浏览器交互的 OpenInterpreter OS Mode**：一位用户想知道 OpenInterpreter 是否可以读取并回复浏览器帖子。
   - 确认了该功能需要 OS Mode，并可能涉及在 Discord 上搜索和发送消息。
- **预构建 OpenInterpreter 的可用性**：一位用户询问是否有预构建版本的 OpenInterpreter。
   - 开发者告知由于需求量大，目前预订已关闭，他们必须等待销售重新开放。
- **访问 OpenInterpreter 文档**：一位成员询问在运行 OpenInterpreter 时如何引用项目文档。
   - 他们解释说正在努力合并 helm charts，并希望访问项目文档以寻求帮助。


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1277295416634904680)** (6 条消息): 

> - `OpenInterpreter 品牌指南` 


- **品牌指南文档请求**：一位成员询问项目的品牌指南文档，并提到了之前在一次无障碍会议中关于此内容的对话。
- **品牌指南文档可用性**：另一位成员确认目前没有可用的品牌指南文档。
- **品牌指南文档的目的**：询问的成员解释了他们作为工业设计师和研究员的角色，并表示有兴趣查阅该文档以用于他们的工作。


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1277043838367694911)** (4 条消息): 

> - `Zed AI`
> - `Open Source`
> - `Cursor`
> - `AI 代码编辑器`
> - `Claude-3.5` 


- **Zed AI：使用 LLM 编码**：Zed AI 为 AI 辅助编程提供了一个强大的界面，让你可以在编辑器中直接与 AI 对话，以生成、转换和分析代码。
   - 它是 Open Source（开源）的，并且支持的模型列表正在不断增加，包括对 Claude-3.5 和 Ollama 的支持，同时还提供了一个专为快速文本转换设计的新 Anthropic API，首月免费。
- **Zed AI vs Cursor**：一位用户质疑 Zed AI 相比 Cursor 的优势，并指出 Zed 的 Open Source 性质是一个关键区别。
   - 该用户对开发更多开源 AI 编码选项表示赞赏。
- **Zed AI Workflow 命令**：Zed AI 提供了一个 `/workflow` 命令，可以建议相关的顺序内联转换，通过与 LLM 的无缝协作来增强开发过程。
- **Zed AI：免费 Anthropic API**：Zed AI 提供了一个专为快速文本转换设计的新 Anthropic API，首月免费。
- **Zed AI：集成到工作流中**：Zed AI 允许你直接在编辑器中与 AI 对话，为 AI 辅助编程提供了一个独特且强大的界面。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://zed.dev/ai">Zed - 下一代编辑器</a>：Zed 是一款来自 Atom 和 Tree-sitter 创作者的高性能、多人协作代码编辑器。</li><li><a href="https://www.youtube.com/watch?v=AbptudVb30Y">Zed AI：这是最好的开源 AI 代码编辑器，支持免费的 Claude-3.5 Sonnet 和 Ollama</a>：加入此频道以获得会员福利：https://www.youtube.com/@AICodeKing/join。在这段视频中，我将向你介绍 Zed AI，这是一款全新的基于 AI 的代码编辑器...
</li>
</ul>

</div>
  

---

### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1277636526485799096)** (2 条消息): 

> - `Apple 的 ML-Superposition Prompting` 


- **Apple 的 Superposition Prompting 项目**: 社区成员对 Apple 最近发布的 **ML-Superposition Prompting** 项目表示兴奋和支持，该项目现已在 GitHub 上发布。
   - 该项目[在此处获取](https://github.com/apple/ml-superposition-prompting)，旨在开发和完善机器学习中 Prompting 的先进技术。
- **关于 Apple 项目没有进一步讨论**: 目前，关于 Apple 的 ML-Superposition Prompting 的唯一讨论是对其发布的最初兴奋。



**提到的链接**: <a href="https://github.com/apple/ml-superposition-prompting">GitHub - apple/ml-superposition-prompting</a>: 通过在 GitHub 上创建账号，为 apple/ml-superposition-prompting 的开发做出贡献。

  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1276652251766587554)** (18 条消息🔥): 

> - `Typed outputs`
> - `DSPy 的 Prompt 长度`
> - `截断的输出 (Truncated outputs)` 


- **OpenAI 的 Typed Outputs 新特性**：几位成员讨论了 OpenAI 的 Typed Outputs 新特性，并分享了指向不同项目的链接，如 **Outlines**、**Guidance**、**SGLang**、**Guardrails** 和 **Instructor**。
   - 对话特别关注如何处理 **结构化输出 (structured outputs)**（通常为 JSON 格式）的验证，以及可用于辅助此操作的各种库。
- **DSPy 错误：重试次数过多 (Too Many Retries)**：一位成员报告了在 DSPy 中使用 `typed predictors` 时出现的 `ValueError`，具体原因是 `Too many retries trying to get the correct output format`（尝试获取正确输出格式时的重试次数过多）。
   - 另一位成员解释说，这通常是因为模型在 JSON 周围输出了填充文本，而 DSPy 中的默认 JSON 解析器效果不佳。他们链接到了 GitHub 上现有的一个 Issue 来解决这个问题。
- **DSPy 中的截断输出 (Truncated outputs)**：一位成员注意到在使用 DSPy 时输出被截断了。
   - 他们怀疑 DSPy 创建的 Prompt 可能超过了 Token 限制。他们询问如何查看发送给 LLM 的最终 Prompt，以及是否存在其他导致输出截断的原因。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/1161519468141355160/1161519469319946286/1265353579103912007">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord 是玩游戏、与朋友放松甚至建立全球社区的绝佳场所。定制你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://discordapp.com/channels/1161519468141355160/1161519469319946286/1265421088326811648">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord 是玩游戏、与朋友放松甚至建立全球社区的绝佳场所。定制你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://x.com/ShreyaR/status/1635292887595491328">shreya rajpal (@ShreyaR) 的推文</a>: 很高兴发布 ✨Guardrails AI✨——一个为 LLM 输出添加 SLA 的开源包！Guardrails 支持 🌟 LLM 输出的 Pydantic 风格验证 🌟 纠正措施（例如重新询问 LLM）当...</li><li><a href="https://github.com/guardrails-ai/guardrails">GitHub - guardrails-ai/guardrails: 为大语言模型添加护栏。</a>: 为大语言模型添加护栏。通过在 GitHub 上创建账号来为 guardrails-ai/guardrails 的开发做出贡献。</li><li><a href="https://github.com/outlines-dev/outlines">GitHub - outlines-dev/outlines: 结构化文本生成</a>: 结构化文本生成。通过在 GitHub 上创建账号来为 outlines-dev/outlines 的开发做出贡献。</li><li><a href="https://github.com/stanfordnlp/dspy/issues/1365#issuecomment-2276998691">请求 OpenAI 结构化输出支持 · Issue #1365 · stanfordnlp/dspy</a>: 你好 DSPy 团队，我想请求在你们的库中增加对 OpenAI 结构化输出的支持。OpenAI 最近在他们的 API 中引入了结构化输出，这似乎可以保证...</li><li><a href="https://github.com/sgl-project/sglang">GitHub - sgl-project/sglang: SGLang 是一个用于大语言模型和视觉语言模型的高速推理框架。</a>: SGLang 是一个用于大语言模型和视觉语言模型的高速推理框架。 - sgl-project/sglang</li><li><a href="https://github.com/jxnl/instructor">GitHub - jxnl/instructor: LLM 的结构化输出</a>: LLM 的结构化输出。通过在 GitHub 上创建账号来为 jxnl/instructor 的开发做出贡献。</li><li><a href="https://github.com/guidance-ai/guidance">GitHub - guidance-ai/guidance: 一种用于控制大语言模型的引导语言。</a>: 一种用于控制大语言模型的引导语言。 - guidance-ai/guidance</li><li><a href="https://github.com/stanfordnlp/dspy/issues/1001">[功能请求] 为可靠的 JSON 生成提供更好的 Typed Predictors · Issue #1001 · stanfordnlp/dspy</a>: Typed Predictors 将 JSON 输出为纯文本。它们不使用函数调用方法/模板，也不强制输出为有效的 JSON。这使得生成结构化输出具有挑战性...
</li>
</ul>

</div>
  

---

### **DSPy ▷ #[colbert](https://discord.com/channels/1161519468141355160/1250300504462856265/1276833487256551446)** (1 messages): 

> - `ColBERT model for German language`
> - `ColBERTv2 data format` 


- **在德语中训练 ColBERT 模型**：一位用户正寻求为德语训练 ColBERT 模型，并希望使用类似于 ColBERTv2 的 32路三元组 (32-way triplets)，但不确定训练所需的数据格式。
   - 用户正在寻找有关如何构建德语 ColBERTv2 训练数据的信息，因为该仓库仅提供了 ColBERTv1 的示例，且 Hugging Face 数据集似乎是空的。
- **ColBERTv2 数据格式**：用户提出了一个用于训练德语 ColBERTv2 的数据格式：`raw_query = [(query, (positive_passage, positive_score) , [(negative_passage1, negative_score1), (negative_passage2, negative_score2), ...])]`。
   - 他们正在寻求确认或见解，以了解该提议的数据格式是否适用于训练德语 ColBERT 模型。


  

---



### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1276629616739614730)** (5 messages): 

> - `Hugging Face Leaderboard`
> - `BFCL V2-Live Dataset`
> - `Model Evaluation and Uploads to BFCL` 


- **Hugging Face 排行榜现在与网站同步**：得益于最近的一个 Pull Request，[Hugging Face Leaderboard](https://huggingface.co/spaces/bigscience/bfcl) 现在与网站排行榜保持同步。
   - 一名团队成员征求了对这一变化的反馈，并询问是否有任何疑虑或建议。
- **关于 BFCL V2-Live 数据集准确率计算的讨论**：关于 [BFCL V2-Live 数据集](https://github.com/ShishirPatil/gorilla/discussions/602) 的讨论以一个关于如何计算数据集整体准确率的问题展开。
   - 该数据集包含一组多样化的 **2,251 个问题-函数-答案对 (question-function-answer pairs)**，由 **258 个简单调用、7 个多个调用、16 个链式调用和 14 个多阶段函数调用**组成。
- **向 BFCL 添加新模型：开源与多模型考量**：一位新成员询问了如何将他们的模型添加到 BFCL，并咨询了在不开源或不提供推理 URL 的情况下上传模型的流程。
   - 他们还询问了如何评估具有多个组件的模型，例如经过微调的 Base Model 和结果监督的 Value Model。



**提到的链接**：<a href="https://github.com/ShishirPatil/gorilla/discussions/602">[BFCL] How should we calculate the overall accuracy for V2-Live dataset? · ShishirPatil/gorilla · Discussion #602</a>：很想听听社区对此事的看法：BFCL V2 • Live 数据集包含 2,251 个多样化的问题-函数-答案对。它由 258 个简单调用、7 个多个调用、16 个...组成。

  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1276636828329443350)** (2 messages): 

> - `Gorilla Function Calling`
> - `Gorilla Leaderboard`
> - `Contributing to Gorilla`
> - `Executable Test Pairs` 


- **Gorilla 排行榜详解**：一位用户就 [Gorilla Leaderboard 文档](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#contributing) 中的“准备可执行测试对 (prepare the executable test pairs)”寻求澄清。
   - 文档鼓励用户向排行榜贡献可执行测试对。
- **Gorilla：为函数调用训练 LLM**：[Gorilla Leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#contributing) 用于训练和评估 LLM 的函数调用（工具调用）能力。
   - 它使用标准化的基准测试来评估不同的模型并允许进行比较。



**提到的链接**：<a href="https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#contributing">gorilla/berkeley-function-call-leaderboard at main · ShishirPatil/gorilla</a>：Gorilla：训练和评估用于函数调用（工具调用）的 LLM - ShishirPatil/gorilla

  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1276756975430008852)** (4 messages): 

> - `Anthropic's mechanistic interpretability`
> - `Llama 8b`
> - `Mistral`
> - `Open Source Implementations`
> - `AI Engineer London Meetup` 


- **Anthropic 的 mechanistic interpretability 运行成本高昂**：一位用户询问了在 Llama 8b 或 Mistral 等大型语言模型上运行 Anthropic 的 mechanistic interpretability 工作的成本，并指出目前缺乏开源实现。
   - 他们很好奇这个过程是数据密集型还是计算密集型，或者是否有其他因素导致其可用性受限。
- **AI Engineer London Meetup - 9月12日**：首届 AI Engineer London Meetup 将于 9月12日 晚上举行，将 Swyx 的 AI Engineer World's Fair 的一部分带到英国。
   - 活动将邀请四位重量级演讲嘉宾：Maxime LaBonne、Rovio Sc、Martins Bruveris 和 Chris Bull。可通过提供的链接进行注册。



**提到的链接**：<a href="https://x.com/dctanner/status/1827071893448618453">Damien C. Tanner (@dctanner) 的推文</a>：我们将 @swyx 的 AI Engineer World's Fair 的一部分带到了伦敦！9月12日晚上是首届 AI Engineer London Meetup。听取 4 位出色演讲者的分享：@maximelabonne, @rovio...

  

---



### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1277679943106695178)** (3 messages): 

> - `OpenAI DevRel`
> - `Logan's Departure`
> - `Romain Huet` 


- **Romain Huet 接管 OpenAI DevRel**：**OpenAI** 的新任开发者关系负责人是 **Romain Huet**。 
   - Huet 的 [Twitter](https://x.com/romainhuet) 个人资料确认他于 2023年7月 加入了 **OpenAI**。
- **Logan 怎么了？**：Logan 在 **2023年7月** 离开了 OpenAI。 
   - 他的离职由其继任者 **Romain Huet** 宣布，并表示 Logan 的过渡过程非常顺利。



**提到的链接**：<a href="https://x.com/romainhuet">来自 undefined 的推文</a>：未找到描述

  

---



### **Alignment Lab AI ▷ #[general](https://discord.com/channels/1087862276448595968/1095458248712265841/1276799769288769640)** (1 messages): 

> - `AI Engineer London Meetup` 


- **AI Engineer London Meetup 宣布举行**：首届 AI Engineer London Meetup 将于 9月12日 晚上举行。
   - 听取四位优秀演讲嘉宾的分享：Maxime La Bonne、Roviosc、Martins Bruveris 和 Chris Bull。注册请访问 [提供的链接](https://x.com/dctanner/status/1827071893448618453)。
- **受 AI Engineer World's Fair 启发的伦敦 Meetup**：这次伦敦 Meetup 是 @swyx 的 AI Engineer World's Fair 的一部分。
   - 活动由 @dctanner 主持，他也宣布了当晚的演讲嘉宾。



**提到的链接**：<a href="https://x.com/dctanner/status/1827071893448618453">Damien C. Tanner (@dctanner) 的推文</a>：我们将 @swyx 的 AI Engineer World's Fair 的一部分带到了伦敦！9月12日晚上是首届 AI Engineer London Meetup。听取 4 位出色演讲者的分享：@maximelabonne, @rovio...

  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/)** (1 messages): 

rolandtannous: 你好，Hamel 在吗？
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1277667337398653081)** (1 messages): 

> - `CUDA Hackathon`
> - `NVIDIA Engineers`
> - `Accelerated Computing` 


- **旧金山 CUDA Hackathon**：CUDA Hackathon 活动将于 9月21日 在旧金山举行，参与者将有机会与 NVIDIA 工程师一起进行 CUDA 开发。
   - 这次活动是向 NVIDIA 专家学习并参与前沿加速计算项目的绝佳机会。
- **使用 NVIDIA 进行加速计算**：该活动专注于 CUDA，即 NVIDIA 的 GPU 并行计算平台和编程模型。
   - 参与者将可以接触到 NVIDIA 工程师和资源，以构建和优化使用 CUDA 的应用程序。

### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1277564072430141511)** (1 条消息): 

> - `Together API 价格调整`
> - `Together API 8B 和 70B 价格上涨`
> - `Together API 定价对比 OpenAI 定价`
> - `Together AI 融资情况` 


- **Together AI 价格上涨**：Together API 将于 2024 年 9 月 1 日起上调 Llama-3 8B 和 Llama-3 70B 模型的 Serverless Reference 端点价格。
   - Llama-3 8B 的价格将从每百万 tokens $0.20 上涨至每百万 tokens $0.40，Llama-3 70B 的价格将从每百万 tokens $0.90 上涨至每百万 tokens $1.80。
- **Together API Turbo 和 Lite 价格保持不变**：Together API 的 Turbo 和 Lite 端点将维持之前公布的价格，详见 [Together Pricing Page](https://www.together.ai/pricing)。
   - 这些价格最近一次公布是在 2024 年 7 月 18 日。
- **OpenAI 的 GPT-4O-Mini 与 Together 的定价**：一位成员注意到 OpenAI 最近降低了 GPT-4O-Mini 的价格。
   - 他们指出，相比之下 Together AI 的涨价行为显得很奇怪。
- **Together AI 的融资**：一位成员推测 Together AI 翻倍涨价是因为其融资即将耗尽。
   - 他们还提到 4-bit 和 8-bit 的定价目前应保持不变，但未来可能会有变动。



**提及的链接**：<a href="https://docs.together.ai/docs/pricing-changes">Together Serverless API 价格变更公告</a>：Together Serverless API 价格变更

  

---



---



{% else %}


> 邮件中已截断完整的逐频道细分内容。 
> 
> 如果您想查看完整细分，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}