---
companies:
- mistral-ai
- moonshot-ai
- nous-research
- google-deepmind
- openai
- groq
- anthropic
date: '2025-07-16T05:44:39.731046Z'
description: '以下是为您翻译的中文内容：


  **Mistral** 发布了 **Voxtral**，声称其为全球最强的开源语音识别模型，目前已通过 API 和 Hugging Face 提供。**月之暗面
  (Moonshot AI)** 推出了 **Kimi K2**，这是一个万亿参数的**混合专家 (MoE)** 模型，其基准测试表现超越了 **GPT-4.1**，在
  SWE-Bench Verified 上取得了 65.4% 的成绩，并在 **Groq** 硬件上实现了每秒 200 个 token 的推理速度。**Nous
  Research** 开源了包含 100 万个样本的 **Hermes 3** 数据集，助力 **Llama-3** 系列模型达到 SOTA（最先进）水平。**谷歌
  DeepMind (Google DeepMind)** 推出了**递归混合 (Mixture-of-Recursions, MoR)** 架构，承诺将推理速度提升
  2 倍并减少 50% 的参数，但该技术也面临一些质疑。**Goedel-Prover V2** 在 **PutnamBench** 定理证明基准测试中位居榜首。在
  **AtCoder 世界总决赛**中，人类选手获得冠军，**OpenAI** 位列第二。研究亮点包括 **Jason Wei** 对**强化学习**的见解，以及强调
  AI 训练中验证不对称性的“验证者定律 (Verifier''s Law)”。'
id: MjAyNS0w
models:
- kimi-k2
- gpt-4.1
- voxtral
- goedel-prover-v2
- llama-3
people:
- cline
- _jasonwei
title: 今天没什么事。
topics:
- speech-recognition
- mixture-of-experts
- benchmarking
- dataset-release
- model-architecture
- theorem-proving
- reinforcement-learning
- asymmetry-of-verification
- inference-speed
- model-performance
---

**平静的一天**

> 2025年7月15日至7月16日的 AI 新闻。我们为您检查了 9 个 subreddits、449 个 Twitter 账号和 29 个 Discord 社区（包含 226 个频道和 5810 条消息）。预计节省阅读时间（按每分钟 200 字计算）：481 分钟。我们的新网站现已上线，支持全元数据搜索，并以精美的 vibe coded 方式展示所有往期内容。访问 https://news.smol.ai/ 查看完整的新闻分类，并在 @smol_ai 上向我们提供反馈！

如果你关心 Claude Code 的未来或 Anthropic 的 1000 亿美元融资，有一个[令人惊讶的人事变动](https://x.com/nmasc_/status/1945537779061977456)；或者 Fal 泄露的[15 亿美元 C 轮融资](https://x.com/arfurrock/status/1945553966495912051?s=46)；再或者，你也可以收听[首个关于 Cline 的播客](https://www.youtube.com/watch?v=uIKmG3M0X3M)。

---

# AI Twitter 简报

**模型发布、性能与基准测试**

- **Mistral 发布 Voxtral 语音识别模型**：[@MistralAI](https://twitter.com/ClementDelangue/status/1945233605745135754) 宣布发布 **Voxtral**，声称这是“世界上最好（且开源）的语音识别模型”。他们提供了[通过 API、在 Le Chat 上试用或从 Hugging Face 下载模型](https://twitter.com/ClementDelangue/status/1945233623164006523)的链接。
- **Kimi K2 开源模型挑战闭源模型**：月之暗面（Moonshot AI）的 **Kimi K2**，一个万亿参数的 **Mixture-of-Experts (MoE)** 模型，成为了热门话题。它现在已[在 CoreWeave 上的 W&B Inference 上线](https://twitter.com/l2k/status/1945225318928634149)，并可在 [LM Arena](https://twitter.com/Kimi_Moonshot/status/1945462820147249523) 中使用。[Cline 展示了 Kimi K2 在 Groq 上运行的演示](https://twitter.com/cline/status/1945354314844922172)，速度达到了 **200 tokens/second**，明显快于 Claude Sonnet-4 通常约 60 TPS 的速度。在基准测试方面，[All-Hands AI 报告称 Kimi-K2 在 SWE-Bench Verified 上达到了 **65.4%**](https://twitter.com/TheZachMueller/status/1945545349352829439)，表现优于 GPT-4.1。该模型的成功被一些人归功于其成本效益，正如 [@skirano 指出](https://twitter.com/skirano/status/1945505132323766430)，如果能完成任务，用户会选择成本更低的方案。
- **Nous Research 发布 Hermes 3 数据集**：[@Teknium1](https://twitter.com/Teknium1/status/1945259797517099126) 宣布开源 **Hermes 3** 数据集，包含 **100 万个样本**。它被用于在 Llama-3 系列上创建 SOTA 模型，包含用于系统提示词遵循、角色扮演、工具调用和原型 Agent 推理的各种数据。该数据集的质量备受赞誉，[@code_star](https://twitter.com/code_star/status/1945359931206721592) 称 Teknium 是在这个追求基准测试分数的世界上“仅存的少数艺术家之一”。
- **Google 推出 Mixture-of-Recursions (MoR) 架构**：**Google DeepMind** 提出的一种名为 **Mixture-of-Recursions (MoR)** 的新模型架构因其[实现 2 倍推理速度并减少 50% 参数](https://twitter.com/algo_diver/status/1945397388946104742)的潜力而受到关注。这种方法也引发了一些质疑，[@teortaxesTex](https://twitter.com/teortaxesTex/status/1945318877849620725) 认为它似乎“过度设计”，可能无法扩展到生产环境。
- **Goedel-Prover V2 登顶定理证明基准测试**：**Goedel-Prover V2** 的发布被宣布为[迄今为止最强的开源定理证明器](https://twitter.com/tri_dao/status/1945273354157539836)，通过解决 6/12 的问题在 **PutnamBench** 上排名第一。这引起了人们对用于评估形式推理和逻辑的 [PutnamBench](https://twitter.com/clefourrier/status/1945386312212664804) 的关注。
- **AtCoder 世界总决赛：人类选手获胜**：在一次编程竞赛中，人类选手 [@FakePsyho](https://twitter.com/itsclivetime/status/1945590725279977900) 夺得头魁，[OpenAI 位列第二](https://twitter.com/gdb/status/1945553676321657127)。由于领先地位多次易手，该活动被描述为[“扣人心弦”](https://twitter.com/gdb/status/1945404295794610513)。

**AI 研究、技术与理论**

- **Jason Wei 论强化学习与验证的不对称性**：在一个被广泛分享的推文中，[@_jasonwei](https://twitter.com/_jasonwei/status/1945294042138599722) 将人生教训与 **on-policy Reinforcement Learning (RL)** 进行了类比，认为要“超越老师”，必须走自己的路，而不仅仅是模仿他人。在另一篇热门帖子中，他提出了[“验证者定律” (Verifier's Law)](https://twitter.com/_jasonwei/status/1945287045251052007)，指出训练 AI 的难易程度与任务的可验证性成正比。这种**验证的不对称性**（即验证一个解比找到一个解更容易）是 AI 进步的关键。这些推文引起了广泛共鸣，[@YiTayML](https://twitter.com/YiTayML/status/1945297017548497366) 指出 “On-policyness 就是力量”，而 [@danielhanchen](https://twitter.com/danielhanchen/status/1945298282961625262) 则推荐了 Wei 关于该主题的斯坦福讲座。
- **OpenAI 呼吁关注思维链 (CoT) 的忠实度**：[@gdb](https://twitter.com/gdb/status/1945350912668737701) 分享了来自 **OpenAI** 及其他业内人士的一篇立场论文，呼吁对使模型推理过程（Chain-of-Thought）具有可解释性和忠实度进行更多研究。他表示这是 OpenAI 的一个投资领域，并已体现在其产品中。
- **Muon 优化器受到关注**：用于 **Kimi K2** 训练的 **Muon** 优化器已变得流行，[@soumithchintala](https://twitter.com/slashML/status/1945333844363657032) 宣布 **PyTorch** 已决定接受将其引入核心库的 PR。
- **RAG 并未过时，而是在进化**：针对检索增强生成 (Retrieval-Augmented Generation) 已经过时的说法，[@HamelHusain](https://twitter.com/HamelHusain/status/1945569284249588016) 等人辩称其依然具有生命力，并分享了关于其演进的带注释笔记。与此同时，[吴恩达 (Andrew Ng)](https://twitter.com/AndrewYNg/status/1945502636012445937) 和 [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1945506275481022872) 在 Coursera 上推出了新的 RAG 课程，涵盖了使用 **Weaviate** 和 **Together AI** 等工具构建生产级系统的内容。
- **比较 LLM-as-a-Judge (LaaJ) 与奖励模型 (RMs)**：[@cwolferesearch](https://twitter.com/cwolferesearch/status/1945540243056144420) 详细分析了 **LaaJ** 和 **RMs** 之间的区别。虽然两者都能提供偏好评分，但 **LaaJ** 通常更适合评估，而定制训练的 **RMs** 在基于 RL 的训练（如 **RLHF**）中更有效。
- **扩展受数据限制的语言模型**：[@Muennighoff](https://twitter.com/Muennighoff/status/1945468469583745959) 分享了他的论文《Scaling Data-Constrained LMs》现已发表在 JMLR 上，这反映出数据重复和混合等技术现已成为标准，且 **RL** 在两年前可能是一个被低估的扩展杠杆。

**AI Agents、工具与框架**

- **浏览器和编程 Agent 的兴起**：**Perplexity Comet** 是一款全新的浏览器 Agent，因其自动化任务的能力获得了积极反馈，用户如 [@itsPaulAi](https://twitter.com/denisyarats/status/1945321982725382170) 称其为“第一次真正让 AI Agent 自主工作”。对此，**Perplexity CEO** [@AravSrinivas](https://twitter.com/AravSrinivas/status/1945537471540072888) 表示历史记录功能已在开发中。在代码生成方面，[@claude_code](https://twitter.com/claude_code/status/1945532878961414230) 分析了 **Claude Code** 的使用情况，指出最常见的错误是 "Content Not Found"，而 `grep` 等搜索工具是其最常用的工具。[@kylebrussell](https://twitter.com/kylebrussell/status/1945242558487044118) 认为 **Google 的 Gemini-CLI** 虽然存在问题但可以修复，相比之下 **Claude Code** 则更加完善。
- **LangChain 发布开源 Deep Research Agent**：[@LangChainAI](https://twitter.com/LangChainAI/status/1945514869224357904) 开源了 **Open Deep Research**，这是一个基于 **LangGraph** 构建的 Agent，采用主管架构（supervisor architecture）来协调子 Agent 执行复杂的研究任务。此次发布包括博客、视频概述和可运行的代码。
- **Runway 发布用于动作捕捉的 Act Two**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1945276901263593591) 一直在展示 **Act Two**，这是 **Runway** 推出的一款新模型，能够根据视频表演生成表现力丰富的角色动作。演示内容包括将真人转化为[跳舞的古希腊雕像](https://twitter.com/c_valenzuelab/status/1945292747188953549)和[《指环王》中的半兽人](https://twitter.com/c_valenzuelab/status/1945483296940441781)，作为创意表达工具被广泛传播。
- **Reflection AI 推出用于代码理解的 Asimov**：[@MishaLaskin](https://twitter.com/swyx/status/1945503020177068506) 宣布了 **Asimov**，这是 **Reflection AI** 推出的一款新工具，旨在帮助工程师理解代码库，解决工程师 **70%** 的时间花在理解代码而非编写代码的问题。
- **LlamaIndex 与 UiPath 集成**：[@jerryjliu0](https://twitter.com/jerryjliu0/status/1945271272243651027) 宣布了 **LlamaIndex** 与 **UiPath** 之间的深度集成，允许开发者在 **UiPath** 的企业自动化平台中使用 **LlamaIndex** 的工作流工具构建自定义 Agent。

**行业趋势、人才与公司**

- **Sam Altman 谈 AI 与就业的未来**：在一条浏览量极高的推文中，[@sama](https://twitter.com/sama/status/1945541270438646270) 赞同 Jensen Huang 对 AI 和就业的乐观态度，表示“赌人类不再想要更多东西……永远是一个错误的赌注”。他预计工作岗位会发生变化，但人们仍将受到创造力和实用性的驱动。
- **Grok Companions 使用量“空前”**：来自 **xAI** 的 [@chaitualuru](https://twitter.com/chaitualuru/status/1945407026252943536) 报告称，新的 **Grok companions** 使用量达到了“前所未有的水平”。[@elonmusk](https://twitter.com/ebbyamir/status/1945433462598684797) 在一条病毒式推文中询问“Ani, are you ok?”进一步放大了这一热度。这种流行促使 [@ebbyamir](https://twitter.com/ebbyamir/status/1945247680176799944) 为 xAI 发布了一个正式的招聘职位，招募所谓的“waifu（二次元老婆）”工程师。
- **“AI 人才大战”与挖角梗持续不断**：各大实验室之间的人才流动已成为一个经久不衰的笑话。[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1945345464305639707) 将与 OpenAI 朋友的晚餐描述为“冷战惊悚片”，大家低声耳语：“那么……扎克伯格给你发邮件了吗？”，[@nearcyan](https://twitter.com/nearcyan/status/1945623927092646286) 也表达了类似感受。["Windsurf 鞭索效应"](https://twitter.com/steph_palazzolo/status/1945226161140728021) 事件进一步凸显了这种戏剧性：据报道，两名 **Claude Code** 的产品经理[离开 Anthropic 加入 Cursor，结果两周后又回到了 Anthropic](https://twitter.com/steph_palazzolo/status/1945555724123476411)。
- **小团队 vs. 大公司**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1945300655314227604) 认为，小型敏捷公司目前正引领创新，而“大公司只能步其后尘，推出雷同的复制品”。与之形成对比的是，一些观察家指出，像 **Meta** 这样的大型实验室正在积累如此多的[明星人才，以至于让人觉得他们不可能失败](https://twitter.com/iScienceLuvr/status/1945292713462522056)。

**基础设施与数据集**

- **美国判例法开源**：[@EnricoShippole](https://twitter.com/algo_diver/status/1945245109580374360) 宣布 **99% 的美国判例法** 已在 Hugging Face 上开源，并指出这些数据通常被法律科技公司以高价出售。
- **FineWeb 数据集扩展**：大规模网络语料库 **FineWeb** 数据集已[根据 CommonCrawl 2025 年 1 月至 6 月的快照进行了更新](https://twitter.com/stanfordnlp/status/1945556488983851420)，目前规模达到 **18.5T tokens**。
- **缓存对编程 Agent 的重要性**：编程 Agent 的效率严重依赖于缓存。[@nrehiew_](https://twitter.com/nrehiew_/status/1945638580673552408) 分享到，他们在 **Cursor** 中使用的 tokens 有 **88%** 是缓存读取，从而实现了显著的成本节省。
- **沃尔玛内部 AI 平台 "Element"**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1945257067389821399) 报道称，**沃尔玛 (Walmart)** 在 Google Cloud 和 Azure 上构建了一个名为 **Element** 的内部平台，允许其工程师利用共享资源和开源模型构建 AI 应用，从而避免供应商锁定 (vendor lock-in)。
- **PyTorch 分布式实用工具**：[@StasBekman](https://twitter.com/StasBekman/status/1945529493915144318) 分享了一个实用工具，用于在 `torch.distributed.init_process_group` 中安全地设置 `device_id` 参数，以防止在某些 PyTorch 版本中出现进程挂起。

**幽默与梗图**

- **"Big Token" 的兴起**：**"Big Token"** 一词成为 [OpenAI、Google 和 Anthropic 等主要 AI 实验室的幽默标签](https://twitter.com/zacharynado/status/1945585062109417899)，该短语的归功于 [@_albertgu](https://twitter.com/_albertgu/status/1945314924286369839)。
- **Grok 的 Waifu 伴侣 "Ani"**：**Grok Companions** 的发布引发了广泛的梗图，由 [@elonmusk](https://twitter.com/ebbyamir/status/1945433462598684797) 询问 "Ani, are you ok?" 拉开序幕，[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1945571762949001409) 则表示：“如果男性 Grok 伴侣名叫 Andrej 并且用他的声音说话，我愿意每月支付 3000 美元。”
- **Claude Code 被归咎于一切**：一个反复出现的笑话是将个人意外归咎于 AI，正如 [@vikhyatk](https://twitter.com/vikhyatk/status/1945224884180644150) 和 [@tenderizzation](https://twitter.com/vikhyatk/status/1945227514101617075) 的推文所称，**Claude Code** 接管了他们的通信，并对奇怪的短信负责。
- **创业艰辛**：[@qtnx_](https://twitter.com/qtnx_/status/1945425672761188502) 发布了一条感同身受的哀叹：“老婆想玩《胡闹厨房 2》，真倒霉，我正在沙发游戏电脑上安装 NixOS [已经折腾 20 小时了]。”

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. 近期 AI 模型与框架发布 (Dream 7B, T5Gemma, llama.cpp Diffusion)

- [**对扩散模型 (Dream 7B) 的支持已合并至 llama.cpp**](https://github.com/ggml-org/llama.cpp/pull/14644) ([Score: 127, Comments: 7](https://www.reddit.com/r/LocalLLaMA/comments/1m1h0fy/support_for_diffusion_models_dream_7b_has_been/)): **最近的 PR (#14644) 将对基于扩散（diffusion-based）语言模型（特别是 Dream 7B）的初步支持合并到了 llama.cpp 中，引入了一种文本生成范式，即通过迭代去噪（iterative denoising）而非自回归（autoregressive） Token 预测来生成输出。这个仅限 CPU 的实现增加了一个扩散采样步骤，包含一个新的 'diffusion-cli' 示例二进制文件，支持高达 2048 个 Token，并公开了扩散超参数的命令行选项；目前尚无 GPU 加速和生产级优化。提供了 GGUF 模型权重和去噪过程的可视化，据报道，相关模型如 DiffuCoder-7B（相同架构）已经可以使用，尽管需要进行增加扩散步数等调整。** 技术讨论引发了对推理速度的担忧——扩散模型在理论上具有效率优势，但目前的实现（例如缺乏 GPU 和 Python 栈集成）使其在实践中比自回归 LLM 更慢。关于在 Ollama 等平台上的即时可用性也存在疑问，但目前 llama.cpp 的上游支持并不保证在没有进一步更新的情况下实现下游集成。
    - 一位用户指出，既然扩散模型支持已合并，基于相同架构构建的 DiffuCoder-7B 应该很容易添加，并确认在需要增加到 512 步的情况下它可以工作，这表明实际使用需要进行一些性能或参数调整。
    - 技术讨论提出了 llama.cpp 中扩散模型的推理速度问题，一位评论者担心技术栈限制（可能是 llama.cpp 环境中的 CPU/内存/批处理）可能会成为瓶颈，抵消扩散模型固有的速度优势。
- [**T5Gemma：一个新的 Encoder-Decoder Gemma 模型系列 - Google 开发者博客**](https://developers.googleblog.com/en/t5gemma/) ([Score: 117, Comments: 17](https://www.reddit.com/r/LocalLLaMA/comments/1m16kdm/t5gemma_a_new_collection_of_encoderdecoder_gemma/)): [**T5Gemma**](https://developers.googleblog.com/en/t5gemma/) 是一个新发布的开源 Encoder-Decoder LLM 系列，由 Decoder-only 的 Gemma 2 模型改编而来，具有使用 UL2 或 PrefixLM 目标的进一步预训练。基准测试结果表明，T5Gemma 模型优于 Decoder-only 的对应模型（如 SuperGLUE, GSM8K），并在质量/推理权衡方面提供了更高的效率，在指令微调和 RLHF 方面有显著收益。发布的 Checkpoint 涵盖了各种模型大小和预训练配置，旨在推动 Transformer 架构和效率的研究。讨论集中在 Encoder-Decoder 和 Decoder-only 模型之间的概念和应用差异，特别是指出双向性（bidirectionality）对 Embedding 任务的重要性，并强调了将自回归 Decoder-only 模型用作 Sentence Transformer 的局限性。评论者推测 T5Gemma 可能会填补大型双向 Encoder(-Decoder) 模型在 Embedding 领域的空白，并询问此类模型对 GGUF 的支持情况。
    - 讨论了 Encoder-Decoder 和 Decoder-only 架构之间的技术区别，特别是关于它们作为 Sentence Transformer 的用途。Encoder-Decoder 架构（如 T5Gemma）由于其双向注意力（bidirectional attention）在生成 Embedding 方面具有优势，能够实现更有意义的句子表示，而 Decoder-only 模型（如 Mistral, Qwen）使用因果掩码（causal masking），限制其仅具有单向上下文，这对于 Embedding 任务并非最优。
    - 人们对提取并微调 T5Gemma 的 Encoder 组件作为 Sentence Transformer 表现出兴趣，这与重新利用大型 Decoder-only 模型的趋势形成对比。评论指出，目前缺乏适合此用途的大型（>3B 参数）Encoder(-only) 模型，这使得 T5Gemma 成为高质量、大规模句子 Embedding 的有力候选者。
    - 用户要求提供关于 T5Gemma 特定基准测试、预期用例以及优于标准模型的架构优势的更多技术细节。同时，也需要 llama.cpp 和 `GGUF` 格式等实际支持，以促进开源社区更广泛的采用和基准测试。

### 2. AI 硬件与加速器进展 (AMD Radeon, MLX CUDA)

- [**AMD Radeon AI PRO R9700 32 GB GPU 在线列出，预计售价约 1250 美元，仅为 NVIDIA 24 GB 显存 RTX PRO "Blackwell" 价格的一半**](https://wccftech.com/amd-radeon-ai-pro-r9700-32-gb-gpu-listed-pricing-around-1250-half-price-nvidia-rtx-pro-blackwell-24-gb/) ([Score: 227, Comments: 86](https://www.reddit.com/r/LocalLLaMA/comments/1m13eb2/amd_radeon_ai_pro_r9700_32_gb_gpu_listed_online/)): **AMD Radeon AI PRO R9700 配备 32 GB VRAM，预计零售价约 1250 美元，约为 NVIDIA RTX PRO 'Blackwell' 工作站 GPU（提供 24 GB VRAM）价格的一半。该列表和定价表明 AMD 正瞄准高端专业消费者或工作站市场，直接在性价比上与 NVIDIA 的同代产品竞争，特别是针对 RTX 5080 而非旗舰级工作站显卡。** 评论者对发布后实际 MSRP 能否维持表示怀疑，质疑 R9700 显存带宽的具体细节（未提供的关键技术细节），并讨论了 NVIDIA RTX PRO 24GB 与更偏向游戏的 5090 GPU 之间的价值主张，指出按价格比较工作站和游戏 SKU 的不合理性。
    - lly0571 提供了 AMD Radeon AI PRO R9700 的技术规格，引用了 `47.8 TFLOPs FP32`、`191 TFLOPs F16 Tensor` 和 `95.7 TFLOPs F16 Tensor TFLOPS with FP32 accumulation`，表明其专注于混合精度和 AI 工作负载，适用于专业 AI 任务及可能的高性能计算场景。
    - Deep-Technician-8568 讨论了 NVIDIA RTX PRO 24GB 与 5090 之间的比较，质疑其合理性，因为两者的目标市场和可能的性价比细分市场存在实质性差异。这突显了在工作站/专业卡与高端消费级 GPU 之间进行对等基准测试或购买决策的挑战。
- [**CUDA 即将登陆 MLX**](https://github.com/ml-explore/mlx/pull/1983) ([Score: 122, Comments: 17](https://www.reddit.com/r/LocalLLaMA/comments/1m1foz1/cuda_is_coming_to_mlx/)): **由 zcbenz 贡献的实验性 [MLX CUDA 后端](https://github.com/ml-explore/mlx/pull/1983) 使得除了 Apple Silicon 之外，还能使用 CUDA GPU 运行 MLX 程序。该后端针对 Ubuntu 22.04 和 CUDA 11.6，需要 CMake 标志 (**`DMLX_BUILD_CUDA=ON`**)，目前支持初始教程的基础操作，旨在利用统一内存并扩大硬件兼容性。贡献者的 [fork](https://github.com/frost-beta/mlx-cuda/commits?author=zcbenz&since=2025-03-20) 正在持续推进，尽管该功能仍处于早期阶段，且尚未测试其他操作系统或 CUDA 版本。** 评论指出，与 llama.cpp 等现有 CUDA 原生库相比，实际收益尚不明确，质疑其与 gguf/awq 等格式相比的性能，并讨论了“CUDA 即将登陆 MLX”与“MLX 即将登陆 CUDA”这两种说法的妥当性。
    - 一位评论者提出了关于性能的关键技术问题：他们有兴趣了解 MLX 的 CUDA 实现与现有 CUDA 兼容库（如 gguf 或 awq）相比如何，特别是在模型量化速度和效率方面，因为“mlx 量化通常产出很快”。
    - 另一位用户指出功能重叠可能有限，因为像 llama.cpp 这样流行的推理库已经提供了成熟的 CUDA 支持，暗示 MLX 可能不会提供显著优势，或者除非它带来独特功能或性能提升，否则其必要性存疑。
    - 关于 MLX 中 CUDA 支持的现状也有讨论：一位用户指出 CUDA 集成尚未合并，表明在预期全面可用性和稳定性之前，可能仍有开发、测试或审查步骤。

### 3. 关键行业观点：Meta 的 ASI 团队与基准测试怀疑论

- [**Meta 的新 ASI 团队讨论放弃 Meta 强大的开源策略并转向闭源开发**](https://www.reddit.com/r/LocalLLaMA/comments/1m14a9j/metas_new_asi_team_discussed_about_abandoning/) ([Score: 189, Comments: 60](https://www.reddit.com/r/LocalLLaMA/comments/1m14a9j/metas_new_asi_team_discussed_about_abandoning/)): **据《纽约时报》相关文章详述，Meta 的新超级智能 (ASI) 团队据报道正在考虑放弃大模型的开源发布，将重心转向闭源 AI 开发。这标志着 Meta 偏离了此前由 Yann LeCun 推动的 Llama 开源模型发布政策；随着 LeCun 被边缘化，新领导层倾向于限制对强大模型的访问，类似于 OpenAI 和 Google 的政策。正在进行或未来的开源发布可能仅限于能力较弱的模型，类似于 Google 的 'Gemma'。** 热门评论表达了对大型科技公司因商业或控制原因而降低开源优先级的担忧，并认为未来的开源进展可能依赖于非营利组织或中国开发者。人们对西方大科技公司是否会实质性支持开源 AI 持怀疑态度，社区的希望正转向 Deepseek, Ai2 和 ETH 等实体。
    - 一些评论者强调，Meta 的开源推动深受 Yann LeCun 等人的影响，随着更多“反对权重开放 AI”的领导层接管，对未来重大开源发布的预期较低。从技术层面来看，这意味着如果没有高管层的倡导者，大科技公司内部的开源势头可能会迅速衰减。
    - 有人指出，Meta 目前最先进的开源模型在 LMSYS 排行榜上仅排名第 **44** 位，并存在可能的 "benchmaxxing" 和评估偏见的说法。这表明，从技术性能和基准测试的角度来看，无论开源状态如何，Meta 的模型在顶级 AI 实验室中已不再被认为具有竞争力。

- [**你对 LLM 的非主流观点**](https://www.reddit.com/r/LocalLLaMA/comments/1m0z1zx/your_unpopular_takes_on_llms/) ([Score: 496, Comments: 358](https://www.reddit.com/r/LocalLLaMA/comments/1m0z1zx/your_unpopular_takes_on_llms/)): **发帖者认为大多数公共 LLM 基准测试（如 MMLU）价值有限，主要反映的是模型对训练数据的记忆而非泛化能力，并批评基准测试题目保密性缺乏诚信。他们还认为使用 LLM 来评判“写作风格”是无效的，且大多数社区微调由于缺乏经验的实践和不加区分的模型上传而降低了基础模型的质量，呼吁进行更好的筛选并增加潜在的资源成本以防止低质量内容的泛滥。** 评论者提出了不同的技术观点：一些人完全否定公共基准测试，建议将特定用户群体（如 gooners）中的模型流行度作为现实世界的衡量标准；其他人强调了 LLM 讨论中缺乏关键信息（Sampler, quantization, 参数），并评论了 LLM 进步速度之快。对于团队质量存在对比观点（Mistral 因效率和专注而受到赞赏），一些人表达了基于模型来源（如中国模型）的细微偏见，同时担心 LLM 会降低用户的认知参与度。
    - Evening_Ad6637 强调了 LLM 讨论中的一个关键挑战：帖子经常缺乏技术背景，如 Sampler 类型、超参数、quantization 方法或推理细节，而这些对于可重复性和理解性能权衡至关重要。Mistral 因其*高效的工程设计和对有意义改进的战略关注*而非激进营销而脱颖而出，这表明了 LLM 生态系统中对设计和优化优先级的关注。
    - hotroaches4liferz 批评了使用 LLM 作为其他 LLM 裁判的创意写作基准测试，认为这引入了显著的偏见，即平庸的 'AI slop' 受到奖励，而像 Claude 这样真正优秀的模型却受到惩罚。该评论认为此类基准测试方法在技术上是不合理的，将风格模仿与实质质量混为一谈，可能会误导研究和用户社区。
    - orrzxz 对目前通往 AGI 的方向表示怀疑，认为统计驱动的文本预测和自动补全的进步并没有在通往通用智能方面取得实质性进展。该帖子强调了一个更深层次的辩论：尽管模型性能和复杂性在快速提升，当前的 LLM 架构和基准测试是否从本质上限制了向更广泛形式的 AI 迈进。

- [**虽然言辞过激，但他说的没错**](https://i.redd.it/dqx9wlf3q9df1.jpeg) ([Score: 1467, Comments: 105](https://www.reddit.com/r/LocalLLaMA/comments/1m1i922/hes_out_of_line_but_hes_right/))：**该图片采用迷因（meme）格式，展示了一个带有风格化技术 UI 元素的动漫角色，以此讽刺本地托管 AI 伴侣的重要性。该帖子批评了基于云端的 AI “女朋友”，幽默地暗示只有本地运行且个性化的模型才是可以接受的，并将远程/云端模型描述为不安全（“告密者”）或商品化的产物。其技术含义集中在 AI 部署中的隐私、用户控制和定制化问题，主张在处理深度个人使用场景时，应优先选择本地运行的 AI 模型而非云端解决方案。** 评论者进一步强化了对隐私和安全问题的担忧，强调了本地模型的价值（“本地主机万岁”），并嘲讽了云端 AI 因数据不在用户机器上处理而带来的冷漠感或风险。
    - 一位用户提到，某条评论可能抄袭自 r/LocalLlama 的热门帖子，这指出了社区对 AI 和本地 LLM 爱好者空间内原创性和迷因流传的关注。这意味着某些笑话和主题正在反复出现，很可能是因为它们与本地 LLM 部署讨论高度相关。

## 较少技术性的 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. Meta 招揽 OpenAI 顶尖人才及行业反应

- [**Meta 再次挖走两名 OpenAI 重量级研究员**](https://www.reddit.com/r/singularity/comments/1m10bhk/meta_poaches_2_more_high_profile_oai_researchers/) ([Score: 587, Comments: 166](https://www.reddit.com/r/singularity/comments/1m10bhk/meta_poaches_2_more_high_profile_oai_researchers/))：**Meta 已从 OpenAI 招募了知名研究员 Jason Wei（以共同撰写开创性的 Scaling Laws 论文并领导 Agent/推理研究而闻名）和 Hyung Won Chung（Codex 负责人、GPT-4 核心架构师、“o”系列和 Deep Research 的关键贡献者），社交媒体公告已确认此消息（参见 [来源](https://x.com/tbpn/status/1945290640545243503?s=46)）。鉴于这些人员对 OpenAI 最先进系统的直接影响，此举可能会增强 Meta 在 Scaling Laws、Agent 模型和高级 LLM 架构方面的能力。** 评论者表示，Meta 持续挖掘 OpenAI 核心人才可能会显著影响未来的模型创新，并使 Meta 处于取得重大突破的地位；同时也对 OpenAI 长期人才储备的广泛影响表示担忧。
    - 多条评论强调了 Meta 聘请 Jason Wei 的战略技术意义，他是一位以极高工作强度和在 OpenAI 的技术贡献而闻名的杰出研究员。Wei 的专业知识，特别是在大语言模型和基础模型扩展（Scaling）领域，被认为是 Meta AI 研究方向的一大助力。社区预期这些重量级聘用将强烈影响 Meta 下一代模型的复杂性和性能，可能加速其在最先进基准测试中的竞争力。
- [**扎克伯格将 Jason Wei 和 Hyung Chung（GPT-4 和 o 系列的共同创造者）挖至 Meta Superintelligence**](https://www.reddit.com/gallery/1m11d3l) ([Score: 250, Comments: 118](https://www.reddit.com/r/singularity/comments/1m11d3l/zuckerberg_poaches_jason_wei_and_hyung_chung/))：**马克·扎克伯格已招募 Jason Wei 和 Hyung Chung 领导 Meta Superintelligence 的工作，两人均被视为 OpenAI GPT-4 和 “o 系列”的共同创造者。这次挖角信号表明 Meta 意图快速扩展其内部 AI 研究，并可能在前沿 LLM 开发上与 OpenAI 竞争，强调了获取顶尖人才是其战略举措。由于无法访问源图集，帖子中未讨论具体的技术基准或模型实现细节。** 评论者推测，在人才引进的驱动下，Meta 可能会在两年内成为领先的 AI 参与者；此外还有关于公司间持续进行的“人才战争”作为行业关键动态的元讨论，尽管没有涉及具体的模型争论。
    - 讨论强调了持续的“人才战争”——特别是挖掘像 Jason Wei 和 Hyung Chung 这样与 GPT-4 和 OpenAI o 系列相关的关键人物——可能会破坏团队凝聚力并对研究进展产生负面影响。有人怀疑这种重量级跳槽能否直接转化为加速创新，一些评论者认为，顶尖人才在不同公司间的碎片化分布实际上可能会减缓而非加速技术进步。

- [**有趣，Meta 是养老院吗，还是说他们引进的顶尖人才真的会为了匹配那巨额薪水而努力工作？**](https://i.redd.it/sym40ee669df1.png) ([Score: 226, Comments: 105](https://www.reddit.com/r/singularity/comments/1m1fdjy/interesting_is_meta_a_retirement_home_or_will_the/))：**该图片是一张推文截图，评论了 Meta 最近试图以极高薪酬方案招募顶尖 AI 研究员的举动。推文声称许多领先的研究人员拒绝了此类邀约，原因是担心个人诚信，并认为加入 Meta 等同于“套现离场”或将其视为“养老院”。这引发了关于 Meta 是否能仅通过巨额金钱激励成功招募并激励必要人才的疑问。** 热门评论反驳了推文的假设，指出此类丰厚的邀约几乎肯定包含严格的绩效要求、最低任期以及取决于里程碑的奖金。评论者认为“拿高薪不干活”的前提是误导性的，因为高级合同通常强制要求交付成果，一些人怀疑真正的顶尖开发者是否会接受没有强力成功挂钩条款的邀约。
    - 几位评论者澄清说，像“1 亿美元”这样的大额邀约并非预先支付；它们通常包含多年的归属期（通常为 4 年），并取决于持续的绩效和雇佣状态。此类薪酬结构旨在激励长期承诺和生产力，而非提供一次性付款。
    - 讨论强调，Meta 为顶尖人才提供的薪酬方案通常包括与绩效挂钩的奖金和最低任期要求。这些条款确保新员工继续为公司做出积极贡献，而不是将其视为“养老院”。
    - 一位评论者指出，包含完全取决于项目或产品成功的付款合同很少见，因为顶尖开发者/研究人员通常会避免带有成功挂钩归属的合同（尤其是对于极高薪酬金额的情况）。
- [**年薪 44 万美元开发 goonbots**](https://i.redd.it/mrhwjensa8df1.png) ([Score: 540, Comments: 42](https://www.reddit.com/r/OpenAI/comments/1m1b37u/get_paid_440000_a_year_to_build_goonbots/))：**该图片是旧金山和加州帕洛阿尔托“Fullstack Engineer - Waifus”职位的招聘列表截图，提供高达每年 440,000 美元的丰厚薪水。该帖子看起来是真实的，热门评论链接到了 xAI 在 Greenhouse 上的实际招聘页面 (https://job-boards.greenhouse.io/xai/jobs/4789505007)，表明这与 Elon Musk 的 AI 初创公司有关。该列表对“waifus”的关注和高额薪水表明，这可能是先进 AI（可能是对话式或角色驱动的机器人）与面向消费者的应用的交汇，可能参考了近期对 AI 伴侣和角色机器人的兴趣。** 评论者对薪水表示极大惊讶，并对职位名称持怀疑或幽默态度，但没有深入的技术辩论。职位真实性的确认确实表明，在尖端公司中，高薪 AI/ML 工程职位的严肃性，尤其是在就业市场收紧的情况下。
    - 一位评论者链接到了 xAI 的招聘列表 (https://job-boards.greenhouse.io/xai/jobs/4789505007)，确认了 18 万至 44 万美元的薪资范围，并暗示上限数字代表该职位的总薪酬上限，而非保证的起薪。这突显了高需求 AI 工程职位的行业薪酬透明度。
    - 一位用户批评了科技行业信息中明显的矛盾：虽然公司吹捧 AI 是自动化软件开发的工具（可能减少对人类程序员的需求），但这些公司却在聘请高薪开发者来创建专门的 AI 应用，如对话或伴侣机器人（“sexbots”）。这强调了 AI 能力中持续存在的差距，以及在 AI 产品化过程中对人类专家参与的持续需求。
- [**真实**](https://i.redd.it/6rp037fgl8df1.png) ([Score: 443, Comments: 47](https://www.reddit.com/r/singularity/comments/1m1cdwt/real/))：**图片显示了来自 xAI 的“Fullstack Engineer - Waifus”的真实招聘列表，工作地点位于旧金山和加州帕洛阿尔托。该列表提到了 xAI 创造具有广泛理解能力的先进 AI 系统的使命，突显了严肃的 AI 研究雄心与以 AI 生成的 waifus 为中心的消费者应用的融合。这既强化了公司广泛的产品范围，也强化了 AI 研究与流行文化娱乐产品之间不断扩大的交集。** 评论反映了对 AI 发展方向的怀疑和担忧，开玩笑说 AI “waifu 军备竞赛”以及对社会操纵的影响。没有详细的技术辩论。

- 几位用户提到了“waifu gap”的概念，指出了 AI 驱动的动漫角色生成器开发中存在的感知差异，暗示了类似于 AI 能力军备竞赛的竞争格局。这反映了生成式 AI 专业领域的持续加速改进，以及可能存在的国际竞争。
- 人们对生成式 AI waifu 模型可能被用作定向营销或宣传手段表示担忧，认为这些系统可能成为塑造观点和消费者行为的强大心理工具，强调了伦理考量和透明度的必要性。
- 提到的“waifu QAT 成员”（Quality Assurance Testing，质量保证测试）反映了 AI 模型开发中的真实工作流程，即专业测试人员确保模型输出符合特定标准——暗示了利基市场中 AI 内容生成的工业化和专业化。

### 2. 最新的视频和 LoRA 模型发布及社区更新

- [**LTXV 刚刚解锁了原生 60 秒 AI 视频**](https://v.redd.it/tq7aozwa3adf1) ([Score: 233, Comments: 57](https://www.reddit.com/r/StableDiffusion/comments/1m1ka0n/ltxv_just_unlocked_native_60second_ai_videos/)): **LTXV 是来自 Lightricks 的开源视频生成模型，声称是第一个能够实现原生长篇（30-60+ 秒）AI 视频的模型，具有强大的基于提示词和控制 LoRA（姿态、深度等）的支持，即使在超长时长下也是如此。它通过调整每帧分块大小（chunk size）在消费级 GPU 上运行，并提供多提示词故事导向和 control net 功能的工作流 ([GitHub](https://github.com/Lightricks/LTX-Video))。发布时，完整的纯 PyTorch 推理仍在开发中（WIP），主要通过 ComfyUI 工作流提供支持。示例工作流展示了在长片段中链接提示词并应用 control nets。** 热门评论批评初始示例视频缺乏动态内容，尽管有人引用了 [Forbes 托管的演示视频](https://youtu.be/J9lkHG6duac?si=zvdRBxVCqpicFGzp)，称其显示出卓越的长期一致性。存在关于实际用途的技术辩论（例如，作为驱动型 v2v 骨干网络），并承认维持 60 秒的一致性是一项重大成就。
    - 一位评论者强调了来自 Forbes 的一个更具技术含量的示例视频，强调在 60 秒的 AI 生成视频中保持一致性是一项重大成就。他们推测，如果 LTXV 在这个长度上保持高效，该技术可以被用于视频到视频（v2v）驱动应用，表明了除了简单生成之外的实际用例潜力。
- [**Lightx2v 刚刚发布了其蒸馏 LoRA 的 I2V 版本。**](https://www.reddit.com/r/StableDiffusion/comments/1m125ih/lightx2v_just_released_a_i2v_version_of_their/) ([Score: 229, Comments: 92](https://www.reddit.com/r/StableDiffusion/comments/1m125ih/lightx2v_just_released_a_i2v_version_of_their/)): **Lightx2v 在 HuggingFace 和 CivitAI 上发布了适用于 Wan2.1-14B 的新图像到视频（I2V）和更新的文本到视频（T2V）LoRA 模型。通过 StepDistill 和 CfgDistill 方法，I2V LoRA 与之前的版本相比，提高了动作一致性和提示词遵循度。报告指出，新上传的 T2V 模型解决了早期的功能问题，并且有可用的备选提取版本显示出更强的动作生成能力。** 由于持续存在的加载问题，评论者独立提取了 T2V LoRA，并注意到不同版本之间在架构和动作生成方面的差异。早期用户测试确认了动作和提示词忠实度的提高。
    - Kijai 指出 Lightx2v 分享的新 T2V（文本到视频）蒸馏 LoRA 无法直接运行，促使他们提取了具有各种 rank 的兼容版本，可在其 HuggingFace 仓库中获取 (https://huggingface.co/Kijai/WanVideo_comfy/tree/main/Lightx2v)。Kijai 指出，这个更新的模型在技术上与初始版本不同，在生成的视频输出中表现出更强的动作。
    - 用户报告了不同的技术表现：虽然一些人注意到最新版本输出的运动和遵循度有所提高，但另一些人指出新 T2V 版本存在产生“纯噪声”的问题。相比之下，I2V（图像到视频）模型的输出质量被赞誉为“极好”，并且与早期迭代相比有显著改进。
    - Roculus 强调了明显且积极的技术改进，特别提到了“明显的运动/遵循度改进”，暗示与之前的模型相比，生成的视频具有更好的帧一致性或主体跟踪。

- [**我发布了 Place it - Fuse it - Light Fix Kontext LoRAs**](https://i.redd.it/2mgus4ljw7df1.png) ([Score: 375, Comments: 70](https://www.reddit.com/r/StableDiffusion/comments/1m19nqp/ive_released_place_it_fuse_it_light_fix_kontext/)): **该图片提供了视觉示例，展示了三个新发布的 Kontext LoRAs（Place it, Fuse it 和 Light Fix）在不同生成任务中的效果。每个 LoRA 都使用小型数据集（20 张前后对比图）训练，在 fal.ai Kontext LoRA 训练器上以 0.0003 的学习率训练了 2000 步。左栏显示肖像修改，中间展示物体合成（例如，将一个绿色地球仪放入厨房水槽），右栏显示动画角色的细微变化，证明了每个 LoRA 的侧重点：物体放置、融合或光影调整。** 一条热门评论要求澄清文件命名规范，凸显了易用性方面的一些困惑。另一位用户询问这些 LoRA 的具体功能。
    - 一位评论者提供了详细的训练参数：训练数据集由 20 对前后对比图像组成，每个 LoRA 使用 [fal.ai Kontext LoRA trainer](https://fal.ai/models/fal-ai/flux-kontext-trainer) 以 0.0003 的学习率训练了 2000 步。
- [**Wan 2.2 将于本月发布。**](https://i.redd.it/jonoyc5dg9df1.png) ([Score: 207, Comments: 57](https://www.reddit.com/r/StableDiffusion/comments/1m1gt8c/wan_22_is_coming_this_month/)): **帖子包含一张来自 Discord 的截图，一名版主确认 Wan 2.2（推测是某个机器学习模型或工具包的即将发布的版本）计划于 7 月发布。讨论集中在与已备受社区推崇的 Wan 2.1 相比，预期的新功能或改进。技术方面的担忧包括与现有工作流或扩展（如 Vace, CausVid 和 LoRA）的潜在兼容性，这表明用户在项目中依赖集成和向后兼容性。** 评论表达了谨慎的乐观，关注 Wan 2.2 是否会带来实质性的进步或仅仅是微小的更新。主要担忧在于维持与当前工具和集成的兼容性。
    - 几条评论对与 VACE, CausVid 和 LoRA 等模型和功能的**向后兼容性**表示担忧，强调了 Wan 2.2 维持当前生态系统互操作性的重要性：*“只希望它不会破坏 Vace, CausVid, LoRA 等的兼容性”*。
    - 存在关于 Wan 2.1 目前通过 **VACE** 等模型支持高级视频控制的技术讨论，并推测 Wan 2.2 应优先考虑更长的视频生成或更高的输出分辨率等功能。这两者都需要显著**更大的 VRAM**，指出硬件资源限制是进一步发展的关键瓶颈。
    - 用户指出，如果 Wan 2.2 只是一个小版本升级（“0.1 的提升”），那么现有的 LoRA 模型应该可能保持兼容，但人们仍然担心即使是微小的更新也可能为广泛使用的定制模型引入破坏性变更。
- [**我制作了这部 AI 短片，看看我能把当前技术推向多远。**](https://v.redd.it/wmrmbrqgt5df1) ([Score: 179, Comments: 15](https://www.reddit.com/r/aivideo/comments/1m130hj/i_made_this_ai_short_film_to_see_how_far_i_could/)): **短片《LYRA》展示了端到端的 AI 驱动电影内容创作，采用了多种最先进的生成工具组合：图像合成（MidJourney, Adobe Firefly）、动画（SeedDance, Kling）和神经语音合成（11Labs）。所有的叙事、视觉和音频组件都由一名制片人编排，突显了当前 AI 在短篇叙事中的自动化程度以及跨多个平台的集成能力。** 顶级评论包含极少的技术辩论，而是集中在审美欣赏和个人反应上；没有实质性的技术批评或工作流讨论。
    - 一个技术讨论点集中在 AI 生成电影中的合成技术，一位用户仔细研究了合成技术在增强视觉效果的无缝性和真实感方面的应用深度。合成的有效性可能是隐藏模型伪影或平滑过渡的关键。
    - 还有关于 AI 视频模型的对比提及，特别是 Kling 与 Veo，其见解是 Kling 在生成的场景中似乎能更可靠地保持角色一致性（即保留角色的外观和身份）。这被强调为长篇视频生成的关键指标。

- [**我制作了这部 AI 短片，看看我能把当前的技术推向多远。**](https://v.redd.it/wmrmbrqgt5df1) ([Score: 175, Comments: 15](https://www.reddit.com/r/aivideo/comments/1m130hj/i_made_this_ai_short_film_to_see_how_far_i_could/)): **一位创作者使用包含 MidJourney（用于视觉元素）、SeedDance（可能用于动画或视频合成）、11Labs（语音生成）、Adobe Firefly（生成式成像）和 Kling（可能用于视频增强或 AI 动画）的工作流，制作了一部 3 分钟的 AI 生成短片（《LYRA》）。该项目展示了当前用于叙事电影制作的多工具 AI 内容流水线，强调了接近专业质量的自主内容创作。** 热门评论没有提供实质性的技术讨论或对工作流、模型限制或过程的批评，而是集中在总体印象和赞扬上。
    - 一位评论者推测了流行 AI 视频模型的优势，特别是比较了 Kling 和 Veo。他们认为 Kling 可能更受青睐，因为它能够在生成的电影序列中保持一致的角色外观，这突显了 AI 视频生成中一个持久的挑战。另一位评论者指出了合成和连续性的重要性，观察到角色视觉效果保持平滑且没有明显的“跳跃”，这解决了当前生成模型中典型的伪影问题。
- [**Mira Murati 的 Thinking Machines Lab 更新**](https://www.reddit.com/gallery/1m0ypbh) ([Score: 168, Comments: 50](https://www.reddit.com/r/singularity/comments/1m0ypbh/mira_muratis_thinking_machines_lab_update/)): **据报道，Mira Murati 的 Thinking Machines Lab 已完成一轮重大的种子轮融资，讨论集中在 NVIDIA (NVDA) 作为芯片供应商和 AI 领域投资者的参与。评论者强调了该实验室尽管** `zero revenue` **却拥有极高估值，质疑在当前的风险投资环境下，如此大规模的早期投资的可持续性和合理性，特别是与 Waymo 等成熟企业相比。** 关键辩论集中在 NVDA 在私人 AI 研发领域无处不在的投资存在是否可持续，或者是否预示着潜在的估值过高。评论者将该初创公司的估值与更成熟的公司进行对比，并批评当代风险投资融资轮次的规范和规模，表达了对市场泡沫的怀疑。
    - 几位评论者强调了 NVIDIA (NVDA) 如何出现在几乎每一个主要的私人 AI 实验室或投资故事中，强化了其不仅作为芯片供应商，而且作为整个 AI 生态系统战略投资杠杆的核心角色。有人推测，由于其无处不在的硬件以及对 AI 基础设施日益增长的影响力，持有 NVDA 股票相当于间接接触到了任何出现的重大 AI 创业公司。
    - 针对 Thinking Machines Lab 的估值提出了技术上的怀疑，一位用户指出，该公司收入为零，但估值约为 Waymo 的 30%——这引发了对 AI 初创公司泡沫或现有企业可能被低估的担忧。
    - 人们对 Thinking Machines 的战略构成和方法很感兴趣，并注意到它与 Anthropic 的相似之处，例如对开源和模型可解释性的关注。在 GPT-5、Gemini 3、Claude (Neptune) 和 Grok 4 等先进模型备受期待的时代，这被认为是意义重大的，市场欢迎具有强大治理能力的稳健、可解释的替代方案。

### 3. Claude Code 高级用法、工作流创新和用户体验

- [**这就是你应该设置 Claude Code 的方式（在使用 Claude 研究时发现的，真够 Meta 的）**](https://www.reddit.com/r/ClaudeAI/comments/1m17ilu/this_is_how_you_should_be_setting_up_claude_code/) ([Score: 232, Comments: 96](https://www.reddit.com/r/ClaudeAI/comments/1m17ilu/this_is_how_you_should_be_setting_up_claude_code/)): **该帖子介绍了一个用于 Claude Code 的开源模块化命令系统，主张反对使用常见的、庞大且单体式的** `CLAUDE.md` **指令文件。相反，它使用了 20 多个范围狭窄的类 XML 命令（例如** `/dev:code-review --focus=security`**），每个命令分为 <requirements>、<execution>、<validation> 和 <examples>，经证明可减少 50-80% 的 token 使用量，提高确定性，并实现快速、上下文相关的项目设置。该方法利用了渐进式披露（即时指令加载）和改进的 Claude 上下文管理，从而产生更小且更相关的 context windows。该仓库地址为 [github.com/oxygen-fragment/claude-modular](https://github.com/oxygen-fragment/claude-modular)。** 技术反馈强调了使用 Claude 原生 [command hooks](https://docs.anthropic.com/en/docs/claude-code/hooks-guide) 以获得确定性行为和直接脚本执行的价值。另一位专家的评论批评该模块化系统仍为 Claude 留下了过多的歧义，存在输出失真的风险，并强调需要更显式、更底层的指令来进行稳健的工作流工程。
    - 一位用户强调了 Claude 的 command hooks 在增强确定性和自动化工作流（如运行 Python 脚本）方面的价值，并指向了 Anthropic 的官方文档：https://docs.anthropic.com/en/docs/claude-code/hooks-guide。这一技术特性允许更受控、可重复的 Claude Code 执行。
    - 有批评指出，讨论的设置过程往往过于高层且缺乏特异性，使 Claude 容易因模糊的指令或上下文而产生错误。建议更好的结果取决于精确且专注的 context engineering，通过严格定义范围的 prompt 来避免 Claude 做出错误假设并导致错误累积。
    - 'LAZY_MODE_PROMPT' 被作为自定义 Claude Code 工作流的额外资源提供，但带有一个技术警告：使用复杂或冗长的 prompt 会显著增加 token 使用量，这对于直接按 API 使用量付费的用户来说可能会产生成本影响。Prompt engineering 必须平衡清晰度和 token 效率。
- [**3 年每日高强度 LLM 使用经验——你所能拥有的最佳 Claude Code 设置。**](https://www.reddit.com/r/ClaudeAI/comments/1m1af6a/3_years_of_daily_heavy_llm_use_the_best_claude/) ([Score: 203, Comments: 64](https://www.reddit.com/r/ClaudeAI/comments/1m1af6a/3_years_of_daily_heavy_llm_use_the_best_claude/)): **该帖子详细介绍了一个用于高强度日常 LLM 开发的高级 Claude Code 环境，强调使用自定义 OpenAI 封装器在兼容 OpenAI 的工具中使用 Claude Max 订阅（例如，通过 ngrok 代理暴露端点），并配置诸如 'ANTHROPIC_CUSTOM_HEADERS: anthropic-beta: interleaved-thinking-2025-05-14' 和 'MAX_THINKING_TOKENS: 30000' 等设置，以增加模型的上下文和性能。将 [Graphiti MCP](https://github.com/arben-adm/mcp-sequential-thinking)（基于 neo4j 的时序知识图谱）与升级后的 sequential-thinking MCP 流水线集成，实现了自动化的、富含元数据的持久记忆——想法和上下文被脚本化并存储在 Graphiti 中。该技术栈还扩展了 Exa Search/Firecrawl 用于实时或参考数据，以及 Relace 作为中间路由层，结合 AutoGen 和 [continue.dev](http://continue.dev/) 进行多 Agent 编排，具备实时 human-in-the-loop 能力和持久化知识图谱。** 关于 Relace 的可信度以及该帖子是否为“托”存在争议。技术派评论者强调了构建稳健的上下文/知识库（而非针对每个功能的记忆）的重要性，指出了在 LLM 工作流中维护全局持久上下文的挑战，并要求分享仓库/代码。
    - 一位用户深入询问了如何利用 Claude Code 维护有效的上下文，指出 *“上下文是关键”*，并描述了他们当前的工作流：构建产品需求文档（PRD），从中创建任务列表，并使用 markdown 文件跟踪和更新任务。他们提出了一个挑战——Claude Code 只能一次跟踪单个功能的上下文，而不是整个项目的上下文，这给全面的知识库管理和 prompt 上下文交付带来了问题。

- 一条评论报告了 Claude VS Code 扩展的一个技术错误：*"End of central directory record signature not found. Either not a zip file, or file is truncated."* 该错误发生在执行 `/status` 期间以及尝试卸载并手动重新安装扩展之后，这表明扩展包可能存在损坏或兼容性问题，可能需要深入调查或官方修复。
- [**作为一名拥有 20 多年经验的软件工程师...**](https://www.reddit.com/r/ClaudeAI/comments/1m1efu0/as_an_software_egineer_with_20_years_of_experience/) ([Score: 264, Comments: 37](https://www.reddit.com/r/ClaudeAI/comments/1m1efu0/as_an_software_egineer_with_20_years_of_experience/)): **一位在 .NET 和大规模云端后端领域拥有 20 多年经验的工程师描述了他们的 ClaudeAI 工作流：(1) 使用自定义 Lyra 提示词进行 Prompt 优化，(2) 在 Claude 中重写 Jira 票据以获得集中的 Context，(3) 将任务分解为最小块，(4) 高度限定范围的 Prompt（例如，仅限于接口或单元测试），以及 (5) 迭代式、基于块的开发。他们强调，严格限定范围并对 AI Prompt 进行排序可以显著提高生产力并优化认知负荷管理，尤其是在接入复杂的 Codebase 时。** 评论者强调了 Prompt 优化的进一步自动化潜力（例如，使用 Lyra 提示词的反馈循环），并讨论了在 AI 快速进步背景下的职业生存担忧。人们对通过自定义命令或脚本将这些工作流操作化也表现出了兴趣。
    - 一位评论者强调使用了一个高度工程化的 Claude Prompt，它体现了资深软件架构原则：专注于将复杂的单体 Codebase 安全重构为模块化架构，并将此过程比作“对活体病人进行手术”。这种方法强调有条不紊且经过充分测试的重构，以维持功能和测试覆盖率，为现代化过程中的代码质量维护提供了模板 ([link](https://github.com/centminmod/my-claude-code-setup))。
    - 针对使用 Lyra 工具进行 Prompt 优化的技术好奇心被提及，特别是将其集成为自定义命令，并将优化后的 Prompt 直接反馈给 Claude 进行迭代改进。讨论围绕这种 Prompt 细化是作为手动的独立步骤更好，还是在工作流中自动化更好，暗示了在基于 LLM 的编码工作流中 Prompt Engineering 与自动化的可能结合。
    - 出现了一个关于 Lyra 和 Traycer 之间关系的问题，探究这些工具之间的技术区别或重叠，可能暗示了它们在 Prompt 优化或 LLM 接口对接中的不同角色——尽管在评论上下文中没有提供直接答案或技术对比。
- [**是我疯了还是 Claude Code 依然表现良好**](https://www.reddit.com/r/ClaudeAI/comments/1m15ca6/am_i_crazy_or_is_claude_code_still_totally_fine/) ([Score: 113, Comments: 218](https://www.reddit.com/r/ClaudeAI/comments/1m15ca6/am_i_crazy_or_is_claude_code_still_totally_fine/)): **该帖子讨论了 Claude Opus API 的代码生成能力，发帖者报告称其输出质量持续稳定，且即使在高强度使用下（4 天内约 750 美元的 API 调用）也没有明显的 Rate-limiting，*这与模型性能下降的广泛说法相反*。Opus 50% 使用量警告在约 60 美元时触发，但并未导致该用户的硬性限制。** 评论揭示了**用户体验的分歧**：一些人报告输出质量明显下降，且与前一周相比决策能力变差，而另一些人则坚持认为*模型没有变化且依然可靠*，将感知到的下降归因于用户差异或轶事偏见。
    - 用户报告 Claude Code 的输出质量存在明显波动，一些人提到最近有所下降——特别是在 Opus 等高级计划上——模型与前几周相比做出了次优的编码决策。
    - 讨论的一种技术变通方法包括将编码任务分解为更小的片段，并在模型尝试自动“压缩（compaction）”之前停止它，因为这个阶段可能会引入问题；据报道，这种方法可以减轻质量下降，并有助于使用 Plan Mode 隔离推理缺陷。
    - 一条评论指出，用户的部分挫败感可能源于他们的软件设计选择——暗示随着项目复杂性的增加，模型的局限性（或架构缺陷）变得更加明显，这有时会被误认为是整体模型退化，而非用户侧的架构问题。

- [**Claude Code 重新上线了，伙计们！好消息。**](https://www.theverge.com/ai-artificial-intelligence/708521/anthropic-hired-back-two-of-its-employees-just-two-weeks-after-they-left-for-a-competitor) ([Score: 120, Comments: 23](https://www.reddit.com/r/ClaudeAI/comments/1m1mhlv/claude_code_is_back_on_the_menu_boys_great_news/)): **Anthropic 在 Boris Cherny 和 Cat Wu 离职前往 Anysphere（Cursor 的开发商）仅两周后，便重新聘请了这两位 Claude Code 平台的关键贡献者。这凸显了生成式 AI 编程助手领域持续的波动和人才争夺战，特别是 Anthropic (Claude Code) 和 Anysphere (Cursor) 都在争夺 AI 驱动的 IDE 和代码建议空间的领导地位。更多参考信息请参阅 [The Verge 的报道](https://www.theverge.com/ai-artificial-intelligence/708521/anthropic-hired-back-two-of-its-employees-just-two-weeks-after-they-left-for-a-competitor)。** 评论区中的技术讨论推测，除了薪酬之外的动机还包括对近期发布的 AI 驱动 IDE（如 Amazon 的 IDE 和 Windsurf）的负面看法，这可能影响了这些领导者回归 Anthropic。
    - 评论者推测，AI 驱动 IDE 领域的近期变化，特别是 Windsurf 的问题和 Amazon 新 IDE 的发布，可能对公众认知或用户采用产生了负面影响，导致 Anthropic 在重新启用 Claude Code 方面进行了战略重新考虑。
    - 一位用户指出，Cursor 最近的定价决策（“搞砸了他们的定价”）可能促使用户寻找替代方案，为 Anthropic 带回 Claude Code 创造了机会或压力，暗示了开发者群体对价格敏感以及 AI 代码助手之间竞争的加剧。

---

# AI Discord 摘要

> 由 X.ai Grok 4 生成的摘要之摘要之摘要
> 

**主题 1. Kimi K2 的热度点燃了模型大战**

- **Kimi K2 在速度对决中碾压 Sonnet**：用户称赞 **Groq** 上的 **Kimi K2** 是比 **Sonnet** 更便宜、更快的替代方案，能以 **250-300 tokens/秒**的速度处理具有 **256K tokens** 上下文的 **Opus 级别** Agent 任务，尽管它跳过了视觉输入，且在 Tool calls 方面落后于 Moonshot。OpenRouter 上的速度勋章认证旨在突出不同供应商之间 **10 TPS** 与 **500 TPS** 的差异。
- **Kimi K2 的效率冠绝中国创新**：**Kimi K2** 以其编程能力和足以媲美 **GPT-4.1 mini** 的定价令人惊叹，引发了本地部署的热潮以规避 **Claude 4** 或 **ChatGPT** 的成本，用户推测它可能作为 *Claude 的强力前沿级替代品* 为 **Manus** 提供动力。现在可以通过 Hugging Face 上的 [Kimi-K2-Instruct-GGUFit](https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUFit) 在本地运行。
- **Kimi K2 引发与 DeepSeek 的竞争戏码**：在 **Kimi K2** 受到热捧的同时，**DeepSeek** 因倾向于中国政府的严格审查而面临抵制，用户指出 *与其他 LLM 相比，DeepSeek 的审查过于严重*，且 **Q4 量化** 版本的质量下降导致在角色扮演中出现严重的幻觉。

**主题 2. GPU 优化技巧成为焦点**

- **显存大战中 BF16 胜过 FP32**：使用 **bf16** 微调 **LoRA** 比 **fp32** 大幅减少了 VRAM 占用，但在运行 **Gemma 3** 的旧款 GPU 上除外，其中 **fp32** 格式的 **7B 模型** 会吞掉 **28GB** 显存。用户通过 [DeepInfra](https://deepinfra.com/low) 的促销活动以 **$2/小时** 的价格抢购 **B200 GPU**，并通过推文修复规避了内布拉斯加州等地区的排除限制。
- **Unsloth 在基准测试中胜过 Liger-Kernel**：在测试中，**Unsloth** 比 **Liger-Kernel** 多节省 **15-30%** 的 VRAM，并以 *Unsloth 梯度检查点带来的惊人上下文长度* 而自豪，尽管最近的更新在默认的 `.cache/vllm/torch_compile_cache` 路径下触发了 **超时错误** 和 VLLM 缓存问题。
- **H20 GPU 引发带宽热议**：中国的 **H20** 在推理方面的互连带宽与 **H100** 持平，但在训练方面逊色于 **GB200/GB300**。用户开玩笑说 **NVL144** 与 **NVL72** 令人困惑，同时 **Voltage Park** 正在 [Voltage Park Careers](https://www.voltagepark.com/careers?ashby_jid=2e463e6a-abc6-48ae-8060-8452c55b2fab) 招聘工程师来构建 AI 工厂技术栈。

**主题 3. 研究论文在效率方面投下重磅炸弹**

- **ETHOS 论文革新了 Sparse Transformers**：[Arxiv 上的 ETHOS 论文](https://github.com/wrmedford/ETHOS)揭示了通过超网络组织稀疏性的高效 Transformer（Efficient Transformers via Hypernetwork Organized Sparsity），专家以潜码形式存储，在 GH200 上实现了 **15K tokens/秒**的训练速度，尽管存在反向传播瓶颈，但在理论上承诺减少 **20 倍的 FLOPs**。它将 *LLM 精神病（LLM psychosis）*定义为一种由于幻觉循环导致的*以脱离现实为特征的精神障碍*。
- **GPUHammer 揭示内存混乱**：[GPUHammer 论文](https://arxiv.org/abs/2507.08166)探讨了数据结构中的内存损坏漏洞，启发了对易受攻击算法的研究。结合 [Muon 优化器视频](https://www.youtube.com/watch?v=4bFDPVe6BHs)，其目标是实现足以媲美 **Claude 4** 的工具使用能力，并在初步测试中展现了潜力。
- **MoEs 解决内存带宽瓶颈**：实验室正在优化 **Mixture of Experts (MoEs)** 的内存效率，如[这段视频](https://youtu.be/JOqLp1adGO4?si=hUAnREYY5CQoeoaQ)所示，使得训练所需的 GPU 数量少于稠密模型。Nvidia 在 [GitHub 上的 LLM RL 框架](https://github.com/ritser-labs/real-work)简化了 Docker 中带有工具访问权限的长程任务（long-horizon tasks）。

**主题 4. 工具和框架提升 Agentic AI 水平**

- **OpenPipe 的 ART Agent 提升任何模型**：[GitHub 上的 OpenPipe ART](https://github.com/OpenPipe/ART) 使用 *LLM 作为评委（LLM as a judge）*来增强模型的 Agent 特性，被认为“非常有趣”并已集成到 Unsloth 中进行微调。在确认其效果“相当不错”后，用户正关注 **ARTwell RULER** 测试。
- **nnterp 为 Transformer 统一了 Mech Interp**：[GitHub 上的 nnterp](https://github.com/Butanium/nnterp) 发布了 Beta 版，通过统一接口桥接了 **transformer_lens** 和 **nnsight**，支持所有 Hugging Face 的 Transformer 模型，包含 **1915 个预计算测试**和一个 [Colab 演示](https://colab.research.google.com/github/Butanium/nnterp/blob/main/demo.ipynb)。
- **MCP 工具赋予 AI 超能力**：Anthropic 的[连接器目录](https://claude.ai/directory)向开发者以外的群体扩大了 MCP 的访问权限，而 [GitHub 上的 Glasses-MCP](https://github.com/gourraguis/glasses-mcp) 允许 AI 对 URL 进行截图并模拟屏幕；Goose 增加了子 Agent，用于通过 **Claude Sonnet-4** 进行多模型编排。

**主题 5. 基准测试和评估面临现实检验**

- **Eval Harness 追踪模型漂移**：提议的 OpenRouter 评估框架（eval harness）以已发布的基准测试为基准，追踪分数的*漂移（drift）*，并通过 **128K** 小说测试验证是否存在上下文压缩。它还包括工具使用评估，如 [GitHub 上的 Tau-2 airline](https://github.com/sierra-research/tau2-bench)，以捕捉模板 Bug。
- **Aider 基准测试急需彻底改革**：随着模型在 Aider 的多语言基准测试中突破 **80%**，用户推动更新并引入私有的用户提交测试；[OpenRouter 上的 SwitchPoint Router](https://openrouter.ai/switchpoint/router) 通过以更低成本路由至 **GPT-4.1** 或 **Gemini 2.5 Pro** 达到了 **80%** 的水平。
- **LMArena UI 调整以应对 Bug**：**LMArena** 用户报告了模型错误和内容过滤器（尤其是漫画类）的误报，并推出了关于模型选择的新 UI 反馈；**Grok4** 因其 *30k+ 隐藏推理（hidden reasoning）*在基准测试表现出色但仅给出单字回答而遭到抨击。


---

# Discord: 高层级 Discord 摘要

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **DeepSeek 倾向于政府审查**：成员们观察到 **DeepSeek AI** 表现出严重的审查制度，在其回答中倾向于中国政府。
   - 成员们对这种审查程度与其他 LLM 相比的差异表示担忧，并指出：*与其他 LLM 相比，DeepSeek 的审查程度更高*。
- **IQ 测试用于区分天赋异禀的个体**：成员们辩论了 **IQ 测试** 在衡量天才方面的有用性，一些人分享了分数和标准差，并引用了 [门萨 IQ 测试](https://test.mensa.no/Home/Test/en-US)。
   - 有人指出，*学校有时会对孩子进行测试，以尝试区分出天赋异禀的个体*。
- **OpenAI 准备发布 GPT-5 公告**：社区对即将发布的 OpenAI 公告进行了推测，预期可能性包括新浏览器、学习功能、新 Agent，甚至是 **GPT-5**。
   - 一位成员预测：*大概是浏览器吧*，并链接到了 [OpenAI Twitter 公告](https://fxtwitter.com/OpenAI/status/1945607177034760574)。
- **等待 API 集成迫使考虑会员资格**：一位成员表示打算 *等到他们将其集成到 API 中*，并由于价值有限而考虑取消 **Pro 会员**。
   - 他们提到集成效果不佳以及像 **Operator** 这样的功能缺乏开发，是可能停止订阅的原因。
- **用户讨论 AI 编程库的未来**：一位成员询问了 **AI 编程库** 的存储位置，寻求教 AI 新的编程技能，类似于 **Jan AI**。
   - 截至讨论时，尚未就教 AI 新编程技能的位置或方法提供具体的答案或指导。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 的 Comet 浏览器在 Reddit 上接受质询**：**Perplexity** CEO Aravind Srinivas 和 Comet 产品负责人 Leonid Persiantsev 在 Reddit AMA 中回答了关于 **Comet 浏览器** 的问题，讨论了其初衷、演变、核心功能、用例和未来计划，详见[此处](https://www.reddit.com/r/ChatGPT/comments/1m1javp/comet_ama_with_perplexitys_aravind_srinivas_and/)。
   - 用户强调了它与 **Gmail** 和 **Calendar** 等 **Google 应用** 的无缝集成，实现了分析电子邮件和安排预约等功能。
- **Perplexity 图像生成遇到波折**：用户报告了 **Perplexity 图像生成** 的问题，遇到了文本回复而非图像，由于团队正在进行改进，图像生成功能在网页端暂时不可用。
   - 用户发现了一个变通方法：使用“an image of”而不是“generate image of”进行生成。
- **三星 Galaxy 用户获得 Perplexity Pro 福利！**：成员们分享了一个针对 **美国** 用户的优惠，通过 **Samsung Galaxy Store** 可获得 12 个月的 **Perplexity Pro**。
   - 然而，非 **美国** 地区的用户可能会面临账号封禁，因此请自行谨慎评估！
- **Perplexity 免费层级可能会加入广告？**：关于免费层级潜在广告的猜测不断，[Sam 确认](https://www.reddit.com/r/ChatGPT/comments/1m1javp/comment/n3hnolo/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) Perplexity 希望广告能与 LLM 回答明显分开，并与 LLM 输出完全独立。
   - 成员们建议采用非侵入式的放置方式，如 UI 广告或赞助搜索广告。
- **Perplexity Pro 是否包含 API 访问权限？**：**Perplexity Pro** 包含 **API 访问权限**，每月提供 **5 美元额度** 用于 **Sonar 模型**，详见 [Perplexity AI 帮助中心](https://www.perplexity.ai/help-center/en/articles/10352901-what-is-perplexity-pro)。
   - 这允许用户在自己的项目中嵌入 **AI 驱动的搜索**，为研究任务提供引用等功能。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 的额度系统面临现实考验**：用户报告 **Cursor 的使用仪表盘存在延迟**，可能无法立即反映 **$20** 额度何时耗尽，导致意外的账单惊喜，但现在用户在达到计费点时会收到聊天内通知。
   - 一些用户认为 **Ultra 方案** 限制过多，很快就会超过限额，并对其长期可行性表示怀疑，有用户报告使用量超过了 **$60**。
- **Kimi K2 抢了 Sonnet 的风头**：成员们正在热烈讨论通过 **Groq** 使用的 **Kimi K2**，认为它是 **Sonnet** 的潜在更优且更便宜的替代方案，拥有 **Opus 级别** 的 Agent 任务性能，以及 **256K tokens** 的超长上下文窗口，速度达 **每秒 250-300 tokens**。
   - 然而，一些用户指出 **Kimi K2** 不支持视觉输入，且在 Tool Calls 方面不如在 Moonshot 上表现好。
- **Cursor 的 Composer 受 Prompt 问题困扰**：用户正经历 **Cursor** 在 Prompt 上卡住并浪费使用量的问题，但新版本包含了一个 **180 秒超时** 机制，可自动取消卡住的请求并防止计费。
   - 重启 **Cursor**、重新安装或使用新的 Prompt 进行“重新思考（rethink）”是建议的解决方案，一位用户建议这有助于重新触发。
- **多 Agent 系统集结以统治 IDE**：一位用户正在开发一个 **多 Agent MCP 服务器**，用于管理跨多个 IDE 的异步 Agent 任务，旨在解决状态持久化问题，采用 **gemii** 进行调试， **grok4** 进行重度重构，以及 **Claude** 进行编码。
   - 另一位用户高效地编辑了 Cursor 的 `main.js` 和其他 IDE，创建了一个极简的 MCP，使用 JSON 进行 I/O，并在启动时连接它们。
- **Context Engineering：智能 Agent 背后的秘诀**：用户强调了为 AI Agent 提供正确上下文的重要性，即 **Context Engineering**，提倡在初始 Prompt 中包含所有必要信息。
   - 其他人强调了在 **System Prompt** 中保持一套简单规则或参考文档的好处，并指出过多的规则可能会导致上下文在长期运行中变得松散。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **BF16 在 LoRA 中优于 FP32（大部分情况下）**：在微调 **LoRA** 时，使用 **bf16** 比 **fp32** 更节省 VRAM，但在旧 GPU 上运行 **Gemma 3** 除外；一个 **fp32** 格式的 **7b 模型** 可能会占用 **28GB** 的 VRAM。
   - 用户讨论了以 **$2/小时** 的促销价格从 [DeepInfra](https://deepinfra.com/low) 租赁 **B200 GPU**，但该促销最初因一条需要更正的 [推文](https://x.com/DeepInfra/status/1935459646493573123) 而排除了内布拉斯加州。
- **OpenPipe ART 声称能使模型 Agent 化**：**OpenPipe 的 ART** ([链接](https://github.com/OpenPipe/ART)) 声称通过使用 **LLM as a judge** 使任何模型更具 Agent 能力。
   - 一位成员承认该工具“确实非常有趣”且使用了 Unsloth，另一位成员在确认其“相当不错”后正打算尝试 **ARTwell RULER**。
- **ETHOS 论文在 Arxiv 亮相**：**ETHOS** (Efficient Transformers via Hypernetwork Organized Sparsity) 论文[现已发布在 Arxiv](https://github.com/wrmedford/ETHOS)，讨论了高效 Transformer，此处附有 PDF [链接](https://cdn.discordapp.com/attachments/1257011997250424842/1394830874142576712/ETHOS___Efficient_Transformers_via_Hypernetwork_Organized_Sparsity_10.pdf?ex=68798e7b&is=68783cfb&hm=0b2c8891c328a38668ff0d015cf8a9f8ef5b80884a3f7eb0ee486aa149b25e97&)。
   - 成员们将 **LLM psychosis** 定义为“一种以脱离现实为特征的精神障碍”，由 LLM 幻觉强化误解引起。
- **Snortts Indic TTS 登录 Playground**：一位成员分享了用于测试 **TTS 模型** 的 Playground：[https://snorbyte.com/snortts-indic-v0](https://snorbyte.com/snortts-indic-v0)，并邀请用户进行测试和提问。
   - 一些成员抱怨被 **LLM psychosis** 的“折磨”所阻碍，并且因为没有博士学位而立即被忽视。
- **Unsloth 在基准测试中击败 Liger-Kernel**：在基准测试中，**Unsloth** 比 **Liger-Kernel** 多节省 **15-30%** 的 VRAM，并且通过 Unsloth 的梯度检查点（Gradient Checkpointing）实现了“疯狂的上下文长度”。
   - 一位用户还指出 **Unsloth** 最近进行了更新，成员们遇到了 **Timeout 错误**，此外，VLLM 缓存默认为 `.cache/vllm/torch_compile_cache`，可能会引发问题。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **ZR1-1.5B 被推荐用于推理任务**：一名成员建议将 [ZR1-1.5B 模型](https://huggingface.co/Zyphra/ZR1-1.5B) 用于推理任务，并提到 **DeepCoder-1.5B** 也是一个可靠的选择。
   - 该成员提醒，在 **7B** 参数预算内实现通用推理具有很大难度。
- **TensorFlow 在就业市场失宠**：多位成员表示，在职位要求中使用 **TensorFlow** 正在变成一个危险信号（red flag），其中一人指出 *目前许多人已经发誓不再使用 tensorflow*。
   - 一名成员表示，没有充分的理由在 **PyTorch** 或 **JAX** 之外选择使用 **TensorFlow**。
- **输入注入加速递归神经网络 (Recursive NNs)**：一名成员建议，改进 *递归方式* 的首选方案是 **在每次迭代中进行朴素输入注入**，作为 [skip connection](https://link.to.skipconnection) 来更轻松地传播状态。
   - 他们澄清说，这涉及注入 **hidden state**、更早的隐藏状态或原始输入本身。
- **潜码 (Latent Codes) 使专家 (Experts) 变得转瞬即逝**：一位研究员详细介绍了他们将专家存储为 **latent codes** 并实时恢复的方法，分享了 [GitHub 仓库](https://github.com/wrmedford/ETHOS)，并强调在 GH200 上的训练速度达到 **15K tokens/second**。
   - 20 倍的 FLOPs 减少是理论上的加速，目前尚未在实证中实现，因为受限于次优的反向传播（backward），其中 *专家仅转瞬即逝地存在* 且不接收梯度，autograd 正在存储中间变量。
- **新 Mech Interp 包 nnterp 发布**：一名成员在 [GitHub 上](https://github.com/Butanium/nnterp) 发布了其 Mech Interp 包 **nnterp** 的 beta 1.0 版本，可通过 `pip install "nnterp>0.4.9" --pre` 安装。
   - 该包旨在为所有 **transformer models** 提供统一接口，同时在底层仍使用 huggingface 实现，缩小了 `transformer_lens` 和 `nnsight` 之间的差距，并提供了 [demo colab](https://colab.research.google.com/github/Butanium/nnterp/blob/main/demo.ipynb) 和 [文档](https://butanium.github.io/nnterp/)。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 的 SwitchPoint 路由引发隐私警报**：一位用户报告称，**SwitchPoint Router** 在未经其同意的情况下被预选，可能违反了他们的 **NDA** 并将 token 发送到中国；在添加防火墙规则禁用 OpenRouter Chat 后，认为其 *不够安全，无法使用*。
   - OpenRouter 管理员澄清说 Switchpoint 总部位于 **美国** 而非中国，用户可以在 [设置](https://openrouter.ai/settings/preferences) 中禁用提供商，且自动路由会在 OpenRouter 上的高质量模型列表中进行选择。
- **DeepSeek 质量随 Q4 量化 (Quantization) 大幅下降**：一位用户注意到 **Deepseek R1 0528** 在角色扮演中的质量显著下降，模型在较低量化下幻觉更加严重。
   - 另一位用户表示赞同，回忆起 **极其糟糕的 R1 表现**，并正在进行 **Q4** 与 **fp8** 的对比测试。
- **OpenAI 的 GPT 3.5 Turbo 端点神秘消失**：一位用户指出 `openai/gpt-3.5-turbo` 端点消失了，寻求澄清并指出其他提供商本可以提供服务，并提到该模型直到 **2025-06-23** 仍有成功的国际象棋记录。
   - OpenRouter 管理员回应称他们正在调查此问题，并已将其 **恢复 (resurrected)** 以供未来使用。
- **OpenRouter 拟定质量和速度认证徽章**：OpenRouter 正在探索模型的质量和速度认证徽章，类似于 `:nitro` 过滤器，以解决如 **Kimi K2** 在不同提供商之间速度差异巨大（**10 TPS** vs **100 TPS** vs **500 TPS**）的问题。
   - 目标是突出具有可靠 tool calling、一致 ratelimits 和高质量输出的提供商，同时考虑量化和潜在的工具调用失败，新模型将从“未验证”等级开始。
- **Eval Harness 漂移模型基准测试**：提议的 eval harness 将以模型作者发布的基准测试为基准，持续测量官方分数与端点分数之间的 *漂移 (drift)* 或差异，以验证 [OpenRouter](https://openrouter.ai/) 上的模型性能。
   - 建议使用高达 **128k context** 的长篇小说基准测试来验证是否存在上下文压缩猫腻，并配合提示模型输出大量 token 的测试，以确认提供商声明的输出 token 数量。



---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Polymarket Bettors Get Burned**: 用户对 **Polymarket** 上的投注未达预期表示惊讶，尤其是涉及 **Google** 的投注。
   - 一位用户幽默地感叹：*我在 polymarket 上的雄心勃勃的赌注没能成功 💀 我很震惊这个服务器里居然有人不买入 Google 💀*。
- **Prediction Market Legality Debated**: 讨论了预测市场在美国的合法性，**Kalshi** 被提及为一个潜在的合法平台，但存在流动性担忧。
   - 一些成员引用了 **CFTC** 的监管规定，而另一些人则表达了对潜在税务问题的担忧。
- **Grok4 Talks Too Much, Delivers Too Little**: **Grok4** 因过于冗长且生成过多的隐藏推理（hidden reasoning）而受到批评，这似乎与其基准测试表现相矛盾。
   - 一位用户批评 xAI 开发的模型输出 *30k+ 的隐藏推理，然后只回复 1 个单词到 1 个句子*。
- **Kimi K2 Claims the Efficiency Crown**: **Kimi K2** 以其效率和编程能力给用户留下了深刻印象，与 **GPT-4.1 mini** 相比具有竞争力的价格和显著的性能提升。
   - 一位成员表示 *中国人在效率方面绝对是做绝了*，并指出它有能力生成有趣的物理沙盒。
- **LMArena Gets Bug Reports and UI Tweaks**: 用户报告了 **LMArena** 的问题，包括模型错误和内容过滤器的误报，尤其是在漫画内容方面。
   - 针对新 **UI** 提供了反馈，特别是直接对话中的模型选择过程，社区管理员将用户引导至反馈线程。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Kimi UI Origins: Bitter Lessons Shared**: 一篇 [Medium 文章](https://medium.com/@xinyijin715/maker-story-the-bitter-lessons-behind-kimi-researchers-ui-6654ec66662c) 揭示了 **Kimi UI** 开发背后的故事和**苦涩的教训**，暗示了与 Cursor 代码的潜在联系。
   - 一位成员引用了一个 [YouTube 视频](https://www.youtube.com/watch?v=motX94ztOzo&t=2737s)，暗示 **Cursor 的代码** 可能发挥了作用。
- **Atlassian's Rovo Dev: A Capable AI Agent?**: Poonam Soni 介绍了 **Atlassian** 的新 **AI Agent, Rovo Dev**，可通过 CLI 访问，专为 **代码生成、审查和调试** 等软件开发任务设计。
   - 尽管有宣传，一位成员嘲讽道 *Atlassian 在体制上就无法构建出好产品*，而其他人则报告了 **下载过程和企业定位** 方面的问题。
- **Flux Pro and Seedance Forge Realistic Videos**: 用户结合了 **'flux pro' 和 'seedance' 模型** 来生成逼真的视频，利用 **'IMG_XXXX.JPG' 技巧** 获取初始图像，并使用 *'aggressively mediocre home footage'*（极度平庸的家庭录像）等提示词。 
   - 讨论中包含了指向 [Replicate 模型](https://replicate.com/) 的链接和积极的用户反馈，尽管该公司的背景仍不清楚。
- **Coding AI Leaders Return to Anthropic**: 据 [The Information](https://www.theinformation.com/briefings/anthropic-hires-back-two-coding-ai-leaders-cursor-developer-anysphere) 报道，**两名编程 AI 专家** 在 **Cursor** 短暂工作后已被 **Anthropic** 重新聘用。
   - 这一举动引发了关于他们是否充当了“双重间谍”的猜测，即由*世界级专家*向 Cursor 提供*廉价咨询*。
- **OpenAI Teases Operator for 2025**: OpenAI 发布了一个神秘视频，暗示将于 **2025 年 7 月 16 日** 发布产品，引发了从 **浏览器/Operator 升级、新的 waifu 或 AURA Agent，到 AGI 阶段揭晓** 的各种猜测。
   - 猜测范围从 **浏览器/Operator 升级**、新的 *waifu* 或 *AURA* Agent，到 **AGI 阶段的揭晓**，而一位成员则将其斥为与 **xAI 的进展** 相比的*重复炒作*。



---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Radix Sort 技巧解锁**：成员们讨论了 [GPU 上的并行 **Radix sort** 实现](https://www.amazon.ca/Programming-Massively-Parallel-Processors-Hands/dp/0323912311/)，引用了一本带有 **2-bit 实现**练习的书籍。
   - 任务是将给定的概念扩展到 **8-bit 实现**。
- **Serverless GPU Profiling 难题**：工程师们就用于远程代码执行和 CUDA profiling（特别是针对 **tiled memory**）的 **serverless GPU 平台**寻求建议。
   - 虽然有人建议使用 **Modal**，但用户希望获得比目前提供的更高级的 profiling 工具，因为在共享 GPU 平台上，由于安全漏洞风险，通常限制完全的 profiling 访问权限（**sudo** 权限）。
- **中国 H20 引发 GPU 热议**：成员们讨论了有关 **中国 H20** 的新闻，并将其与 **H100** 的互连带宽进行了比较。
   - 与 **GB200/GB300** 相比，**H20** 被认为在训练方面较弱，社区还开玩笑说 **NVL144** 和 **NVL72** 配置之间不可避免会产生混淆。
- **Voltage Park 扩充团队**：**Voltage Park** 正在招聘一名**软件工程师**来帮助构建其 **AI Factory** 软件栈，并寻找能够协助构建核心基础设施的人才。
   - 优先考虑美国境内的申请人，职位发布详情见 [Voltage Park Careers](https://www.voltagepark.com/careers?ashby_jid=2e463e6a-abc6-48ae-8060-8452c55b2fab)。
- **SemiAnalysis 播客提到论坛**：一位新成员因为 **SemiAnalysis** 的 Dylan 在播客中提到了该论坛而加入，并分享了一个 [Google Meet 链接](https://meet.google.com/wdk-yipf-zjd?authuser=0&hs=122)。
   - 另一位成员发布了一个用于 **long-horizon** 任务的 **LLM RL 环境框架**的初始版本，可在 [GitHub](https://github.com/ritser-labs/real-work) 和 [X.com](https://x.com/ritserlabs/status/1945494003803148565) 上找到。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Kimi K2 引发热潮**：社区对 **Kimi K2** 感到兴奋，有人称其正在迎来属于它的 *DeepSeek 时刻*，并希望能够下载并在本地部署，以避免为 **Claude 4** 或 **ChatGPT** 付费。
   - 据推测，这将通过“不再向 Sam 和 Dario 支付租金”并拥有自己的模型来实现*经济自由*。
- **H200 与 H100 成本对比演变**：**H200** 的价格几乎与 **8xH100** 相同，在 [celiumcompute.ai](https://celiumcompute.ai) 上发现 **B200** 的价格低至 **$2/GPU/小时**。
   - 经澄清，该低价是通过 *deepinfra* 进行的限时促销。
- **Manus 被推测将取代 Claude**：据 [twitter](https://x.com/Teknium1/status/1945259797517099126) 消息，推测发布的 **Manus** 可能会整合 **Kimi K2** 的 **Agent** 能力，出于地缘政治原因，可能会取代 **Anthropic Claude** 进行编程。
   - 它被认为是 *Claude 的强力前沿级替代方案*。
- **Nvidia 发布 LLM RL 框架**：Nvidia 发布了一个用于 **long-horizon** 任务的 [LLM RL 环境框架](https://research.nvidia.com/labs/adlr/AF3/) 初始版本，使得在拥有工具访问权限的 Docker 容器中设置环境并生成轨迹（trajectories）变得更加容易。
   - 该框架名为 *real-work*，可在 [GitHub](https://github.com/ritser-labs/real-work) 上获取。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 避开图像生成**：虽然 LM Studio 支持 **image input**（图像输入），但团队目前没有添加 **image generation**（图像生成）功能的计划。
   - 团队优先考虑其他功能。
- **用户请求自定义模型搜索仓库 URL**：一位用户请求能够指定 **Model Search repo** 的自定义 URL，而不是仅限于 **Hugging Face**。
   - 成员表示，由于目前无法切换出 **Hugging Face**，手动下载模型并导入是目前的折中方案。
- **LM Studio 没有公开路线图**：一位用户询问是否有 **公开的 LM Studio 开发路线图**，以了解项目的未来方向。
   - 另一位成员回答说目前没有公开路线图。
- **记忆功能可能会进入 LM Studio**：用户讨论了在 LM Studio 中加入 **memory features**（记忆功能）的潜力，类似于 **ChatGPT** 中的功能，这将允许聊天引用之前的交互。
   - 一位成员建议利用具有 **记忆能力的 MCP**，类似于现有的 **rag-v1** 和 **code-sandbox mcps**。
- **LG 的 EXAONE 许可证限制**：社区辩论了 **LG 的 EXAONE 许可证** 的限制性，特别是要求在所有发布的模型中包含 *"EXAONE"*，以及其对商业和研究用途的限制。
   - 针对许可证下 *"research"*（研究）的定义提出了担忧，特别是关于逆向工程和蒸馏（distilling），导致一些人认为该许可证存在矛盾且难以执行。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 转型为新产品**：**torchtune 团队** 在 [一个 GitHub issue](https://github.com/pytorch/torchtune/issues/2883) 中宣布，他们计划将 **torchtune** 演变为一个位于 *新仓库* 中的 *新产品*。
   - 他们将继续在 **Discord** 和 **GitHub** 上支持社区。
- **Torchtune 的许可证激发了信心**：一位成员询问在使用 **Torchtune** 组件时的 **知识产权** 问题，另一位成员指出 [Hugging Face 已经在 TRL 中使用了它](https://github.com/huggingface/trl/blob/640a9f39164ce3f9bbac7d325c1149ba023314e6/trl/models/activation_offloading.py#L19)。
   - 这利用了宽松的 **BSD 3 license**。
- **量子计算机落地俄亥俄州**：一位成员提到他们在 **Cleveland Clinic** 工作，这是 *世界上唯一一家拥有量子计算机的医院，哈哈*。
   - 另一位成员幽默地指出，量子计算机竟然 *在食堂中间……在俄亥俄州？？*
- **NFT 为 Checkpointing 铺路**：成员们开玩笑地建议将分布式 **RL**、**区块链** 和 **量子计算** 等未来技术用于 **checkpointing** 及其成本管理。
   - 一位用户建议为你的 **checkpoint** *创建一个 NFT*。

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Kimi K2 在本地运行**：**Kimi K2** 被称为全球最强大的开源模型，现在可以在本地设备上运行，可以在[这里](https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUFit)找到。
   - 成员们建议在分享时遵守适当的频道礼仪。
- **解码 Qwen-1.5 的推理**：一位用户正在寻求 **Qwen-1.5** 所使用的确切结构和推理技术，讨论指向了一个[相关文件](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py)。
   - 另一位成员认为浮点误差可能是导致差异的原因。
- **分享 ETHOS 量化技术**：一位成员分享了 [ETHOS](https://github.com/wrmedford/ETHOS)，深入探讨了 **LLM 量化技术**，并链接了一个关于该主题的 [YouTube 视频](https://youtu.be/0pF6GdbwMo4?si=swVldbUTY5Gn4mYB)。
   - 随附的 [PDF](https://cdn.discordapp.com/attachments/897390720388825149/1394879190049882132/ETHOS.pdf?ex=687912ba&is=6877c13a&hm=8c3b1d564877e4310662d28d5a240c65d01f81b2e66098066acc531c259b6cd5&) 深入研究了 LLM 量化技术。
- **法语 Deep Learning 课程增加 AI 功能**：一位成员宣布了他们的 **法语 Deep Learning 课程** 项目的新功能，包括 **AI 生成的 QCM**（多选题）。
   - 资源可在 [课程网站](https://simonthomine.github.io/CoursDeepLearning/) 和 [GitHub 仓库](https://github.com/SimonThomine/CoursDeepLearning/) 获取。
- **乌克兰语翻译器轻量且就绪**：一位成员分享了一个新的用于 **英文到乌克兰文机器翻译** 的 **轻量级模型**，该模型使用最近发布的 **LFM2 模型**，在 53.5M 样本中的 **40M 样本** 上进行了微调。
   - 该模型（**350M params**，需要 **1GB RAM**）在 **FLORES-200** 上达到了 **27.24** 的 **BLEU** 分数，模型可在 [Hugging Face](https://huggingface.co/Yehor/kulyk-en-uk) 获取，并配有 [Demo Space](https://huggingface.co/spaces/Yehor/en-uk-translator)。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **扎克伯格的 Behemoth 背叛重磅炸弹？**：一些成员指责 **扎克伯格** 背叛，原因是 **Behemoth 模型** 可能受限发布，并引用了[这条推文](https://x.com/signulll/status/1944851904888234293)作为证据。
   - 一些人推测 **Meta** 可能会效仿 **Google** 的策略，发布像 **Gemma** 这样较小的开源权重（open-weight）模型作为替代方案。
- **解码黑魔法：推断闭源模型？**：成员们建议从开源权重模型推断闭源模型可能是一个值得研究的方向，特别是考虑到[这篇论文](https://arxiv.org/abs/2407.18384)。
   - 该成员表示：*大多数人不会在本地运行这类模型，但依然很糟糕*。
- **GPUHammer 罢工，内存损坏大屠杀！**：[GPUHammer 论文](https://www.arxiv.org/abs/2507.08166)研究了内存损坏（memory corruption）以及数据结构对此类问题的敏感性。
   - 成员们正积极研究易受内存损坏影响的 **数据结构和算法**。
- **Muon 优化器介入**：**Muon** 优化器出现在[这篇论文](https://arxiv.org/abs/2507.08166)和[这段视频](https://www.youtube.com/watch?v=4bFDPVe6BHs)中，旨在实现与 **Claude 4** 相当的工具使用（tool usage）能力。
   - 初步测试显示出前景，但仍需进一步评估以验证其在不同场景下的有效性。
- **MoEs 最大化内存，最小化移动**：实验室正在利用 **混合专家模型 (MoEs)** 来优化内存带宽，正如[这段视频](https://youtu.be/JOqLp1adGO4?si=hUAnREYY5CQoeoaQ)所示，内存带宽是一个关键瓶颈。
   - 看起来 **MoEs** 潜力巨大，通过高效的资源利用，能够比稠密模型（dense models）使用更少的 GPU 进行训练。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Vertex AI 显示 Thinking Tokens**：一位用户发现，在 Aider 中使用 `openrouter/google/gemini-2.5-pro` 模型运行 `/help` 时，可以显示 Thinking Tokens，并且 `/think-tokens` 命令也能启用思维摘要的显示。
   - 该用户引用了 [Paul Gauthier 的推文](https://x.com/paulgauthier/status/1932068596907495579)，内容涉及在尝试通过 Vertex 使用 **Gemini 2.5 Pro** 时，如何配置 **32k Thinking Tokens**。
- **Ghostty 和 Kitty 获得终端应用推荐地位**：用户讨论了 Aider 的终端推荐，建议使用 **Ghostty** 和 **Kitty** 以获得更好的性能，尽管一些人认为 **GNOME terminal** 已经足够。
   - 一位遇到 Aider 屏幕刷新问题的用户被建议尝试 Ghostty，而另一位用户推荐了 **Alacritty**，尽管它在图像显示协议方面存在困难。
- **Kimi K2 搭配 Groq 脱颖而出**：一位用户报告称，**Kimi K2 搭配 Groq** 表现惊人，速度达到 **200-300 tokens/sec**，且输出质量极高，足以媲美 **GPT-4o** 并超越 **Sonnet 3.7**。
   - 他们强调了其高性价比和速度，使其成为首选，这也呼应了社区中其他人的积极反馈。
- **Aider 基准测试更新**：用户建议 Aider 应该更新其 Benchmark，因为许多模型现在的得分都超过了 **80%**。
   - 该提议包括创建一个私有基准测试，用户可以贡献自己的测试用例。
- **SwitchPoint Router 引起关注**：一位成员询问了 [OpenRouter.ai 上的 SwitchPoint Router](https://openrouter.ai/switchpoint/router)，该路由可以将请求以可能更低的价格路由到 **GPT-4.1**、**Claude 4 Sonnet** 和 **Gemini 2.5 Pro** 等模型。
   - 该路由器的网站宣称在 *aider polyglot benchmark* 上获得了 **80%** 的成绩，引发了进一步讨论。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **文档标签页技巧激发实验兴趣**：一位成员发现 **Google Docs 的标签页功能** 是一种在 **NotebookLM** 中管理来源的新颖方法。
   - 该用户对将此功能应用于该平台表示了兴趣。
- **广告拦截扩展解决问题**：一位成员建议利用 **uBlock 浏览器扩展**，在将新闻文章复制到 **Google Docs** 时去除广告和多余元素。
   - 他们指出，可以在扩展设置的 *Filter list* 标签页下添加针对干扰项和社交媒体弹窗的额外过滤器。
- **用户对精选笔记本表示不满**：用户对被强制查看 **"Featured Notebooks"**（精选笔记本）且无法移除感到沮丧。
   - 他们希望在 **NotebookLM** 中获得更具定制化的体验和组织方式。
- **播客长度受语言设置影响**：一位用户报告说，由于语言设置错误，他们的播客内容一直很短，大约只有 **7-10 分钟**。
   - 另一位用户指出，**English** 选项下可以选择“长”播客输出，从而解决了该问题。
- **“Service Unavailable” 困扰用户**：一些用户在使用 **NotebookLM** 时遇到了 **“Service unavailable”** 错误消息，但缺乏足够的上下文信息。
   - 该错误表明用户正尝试访问一个其账号不可用的服务。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus AI 移动应用开发精通**：一位用户报告称，成功使用 **Manus AI** 开发了多个主题的移动应用程序，并进行了定制化设计，仅消耗了 **100 credits**。
   - 他们主动提出帮助其他在 **Manus** 应用开发中遇到困难的人。
- **用于车辆创建的免费 Manus 替代方案出现**：一位用户声称拥有一个与 **Manus** 功能类似的 **100% 免费替代方案**，并利用它为 **OMSI 2** 巴士模拟游戏创建车辆。
   - 他们建议可以生成一个 **Google Collab** 脚本来生成所需文件，具体取决于所使用的模型。
- **下一代 AI 声称性能超越 Manus**：一位用户声称开发了一套 **AI** 系统，在基准测试性能上超过了 **Manus**。
   - 他们向首批 **100 名 Beta 测试人员** 提供终身访问权限，邀请用户通过私信（DM）锁定名额。

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Discord 频道扩展引发 Modular 论坛讨论**：一名成员建议创建一个新的 Discord 频道用于用户项目分享，但一名工作人员建议改用 [Modular 论坛](https://forum.modular.com/c/community-showcase/8) 的 **Community Showcase** 类别。
   - 该建议旨在将项目展示整合到现有的论坛结构中。
- **Mojo requests 库需要 TLS**：一名成员询问关于原生的 **Mojo** *requests* 库，另一名成员指出主要的阻碍是 **TLS 支持**。
   - 一名成员分享了他们的 [类 requests 玩具库](https://github.com/thatstoasty/floki)，其中包含 **TLS 绑定**。
- **解码 Escaping：不只是为了越狱**：一名成员寻求关于 **Mojo** 中 *escaping* 关键字用法的澄清，并指出缺乏具体的文档。
   - 另一名成员指向了 [Changelog](https://docs.modular.com/mojo/changelog#v070-2024-01-25)，澄清了 *escaping* 对值执行 `__copyinit__` 而不是通过引用捕获。
- **参数装饰器被捕获了！**：一名成员询问关于 `@parameter` 函数（捕获）的问题，另一名成员提供了一个专门针对该内容的 [手册链接](https://docs.modular.com/mojo/manual/decorators/parameter/#parametric-closure)。
   - 他们还提到了 **Q3 路线图** 中关于 *统一 @parameter 和运行时闭包* 的令人兴奋的消息 ([roadmap-update](https://forum.modular.com/t/mojo-q3-roadmap-update/1957))。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Setitem PR 面临审查**：一名成员为其 [setitem PR](https://github.com/tinygrad/tinygrad/pull/11260) 寻求审查，并质疑所实现方案的开销。
   - 审查者建议 `tensor.py` 的修改应集中在移除 `realize` 调用并在更底层解决它。
- **优化僵局：Assign 的参数**：关于是否可以为 assign 添加参数以便让用户指定范围和索引的讨论展开。
   - 然而，一名审查者认为这 *不值得*，并澄清这 *不是该 bounty 所寻求的*。
- **移除 realize() 的拟议修复方案揭晓**：针对仅移除 `realize()` 调用时赋值无法持久化的问题，一名成员提议将代码行修改为 `self.uop = res.assign(v).uop`。
   - 其他替代方案包括 `self.assign(v, indices)` 或 `self.uop = reconcileAssignment(self.uop, res.assign(v).uop`。
- **Tinygrad Tensor Hooks：寻求建议**：一名用户询问 **tinygrad** 是否支持 tensor 级别的 hooks，目的是获取大型语言模型 (**LLM**) 中的 **hidden states**。
   - 该用户正在探索在模型执行期间提取和利用 **hidden states** 的方法。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 与 Snowflake 在阿姆斯特丹会面**：LlamaIndex 正与 **Snowflake** 合作，于 7 月 31 日在阿姆斯特丹举行见面会，重点是在生产环境中构建高质量的 **Data Agents**，正如 [Twitter 上宣布的那样](https://t.co/fFJvvIWrw4)。
   - 此次见面会旨在聚集对生产环境中的 **Data Agents** 感兴趣的工程师。
- **UiPath 拥抱 LlamaIndex Agents**：**UiPath** 现在支持 LlamaIndex agents，通过新的代码化 Agent 支持，实现到企业环境的无缝部署，详见其 [公告](https://t.co/ILez3d6Zrs)。
   - 特性包括通过 **UiPath 的 Python SDK** 实现 *全代码级控制*，以及构建从企业系统提取数据的自定义 Agent 的能力，为开发者提供了更大的灵活性。
- **工程师分享 RAG 系统技巧**：一名开源工程师分享了构建生产级 RAG 系统的技巧，特别是关于 **文本提取策略**，见 [此讨论帖](https://t.co/R0TTgWrKtv)。
   - 讨论涵盖了何时使用 *简单解析与基于 OCR 的高级解决方案*（如 **LlamaParse**），并就优化提取方法提供了建议。
- **ODSC 峰会聚焦 LlamaIndex Agent 构建**：一名 LlamaIndex 成员正在 **ODSC 的 Agentic AI 峰会**上主持研讨会，指导参与者使用 LlamaIndex 构建 Agent，更多信息 [点击此处](https://t.co/6jcYIGR70s)。
   - 参与者将学习 *创建自主应用程序*，这些程序利用目标和工具独立完成任务，从而增强他们的 Agent 构建能力。
- **LLM 微调指南寻求反馈**：一名工程师分享了一个 **LLM Fine-tuning 指南** 的 MVP，寻求开发者对数据准备、参数选择和模型评估等实际建议的反馈，该指南在 [Vercel](https://ai-finetuning-advisor-g3r5.vercel.app/) 上展示。
   - 该指南旨在提供分步说明，以简化 **Fine-tuning** 过程。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **IReRa 在层级标签中的作用**：一位成员询问在涉及具有 **3 个层级**、**440 个父级**和 **3500 个孙级**的 **hierarchical labels**（层级标签）用例中，使用 **IReRa** 是否明智。
   - 另一位成员建议使用 **multiple modules**（多个模块）可能更有效，尤其是在每一步处理有限数量的标签并使用大语言模型（**LLMs**）时。
- **原生 DSPy 处理父子识别**：一位成员提议使用 **vanilla DSPy** 先识别父级，然后在层级结构中从父级移动到子级再到孙级。
   - 另一位成员确认在具有 **3 个父级**和 **28 个子级**的类似方案中成功实现了该方法。
- **Nova Prompt Optimizer 与 DSPy 的关联**：一位成员证实 [aws/nova-prompt-optimizer](https://github.com/aws/nova-prompt-optimizer) 需要 `dspy` 作为依赖项。
   - 未来可能会对两者之间的交互进行进一步研究。
- **Lean 4 的验证潜力**：一位成员建议使用 **Lean 4** 来验证某些内容。
   - 他们分享了一个关于 **Lean 4** 的 [YouTube 视频](https://www.youtube.com/watch?v=1067jj67toY) 以提供更多背景信息。



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **学生因错过证书申报表而受到提醒**：由于人手有限，课程组织者无法为错过 **certificate declaration form**（证书申报表）截止日期的学生提供帮助。
   - 一位学生请求重新开放 [申报表链接](https://forms.gle/iPA2MUpHdtrBE1vu5)，但其请求最终被拒绝。
- **学生寻求实验提交的反馈**：一位学生询问如何获得关于其 **lab submission performance**（实验提交表现）的反馈，并探讨额外的研究方向。
   - 该学生对其提交的内容表示满意，但希望能与人进一步讨论其表现。



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Anthropic 连接器目录开启 MCP**：Anthropic 推出了 [“connectors” 目录](https://claude.ai/directory)，旨在向更广泛的受众介绍 **Model Context Protocol (MCP)**，从而可能增加需求。
   - 与面向开发者的 **Docker toolkit** 不同，该目录针对的是包括产品经理和营销人员在内的更广泛受众。
- **Glasses 👓 赋予 AI 视觉！**：一位成员发布了 **Glasses** 👓，这是一个实现了 **Model Context Protocol (MCP)** 的全新 **open-source tool**。它能让兼容的 AI 请求 URL 截图并模拟不同的屏幕尺寸，现已在 [GitHub](https://github.com/gourraguis/glasses-mcp) 上可用。
   - 该工具将赋予 AI Agents 视觉能力，提升其处理外部网站的能力。
- **Goose 支持多 Agent 系统**：**Goose** 现在支持 subagents，增强了灵活性并实现了多 Agent 系统，通过 [Codex CLI 作为 subagent](https://block.github.io/goose/docs/experimental/subagents) 进行了展示。
   - **Goose** 支持 **Anthropic Claude Sonnet-4**、**OpenAI**、**Gemini**、**Ollama**，通过协调主任务和 subagents 然后合并结果来促进自主编排，从而简化复杂的工作流程。
- **MCP Inspector 无法重新加载资源**：一位成员报告称，在使用 **mcp typescript-sdk** 的工具内部更新 profile 并调用 `server.sendResourceListChanged();` 后，**MCP Inspector** 不会重新加载资源。
   - 他们发现，除非清除并重新列出资源列表，否则工具内的资源刷新无法反映更新。



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Cloudflare R2 报错**：一位成员在尝试使用特定的 `aws s3 ls` 命令从 **Cloudflare R2** 下载数据集时遇到了 **Access Denied** 错误。
   - 用于微调模型的命令是 `aws s3 ls --endpoint-url=https://9fa58365a1a3d032127970d0bd9a1290.r2.cloudflarestorage.com/ s3://contrastive`。
- **GPT4ALL 的神经中立性？**：一位成员询问 **GPT4ALL** 是否基于原始逻辑、推理和输出来处理输入，并采取用户优先的方法。
   - 他们询问 **neutrality guidelines**（中立性指南）和 **safety filters**（安全过滤器）是否仍在实施。
- **AI 工程师乘着 Web3 浪潮**：一位 AI 和 Web3 软件工程师正在初创公司、研究团队或自动化领域寻求机会，他带来了构建自主系统的经验。
   - 他们的技能栈包括 **Python**、**TypeScript (React/Next.JS)**、**C/C++**、**LangChain**、**ReAct**、**OpenAI** 以及 **Solidity/Rust**。



---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **DeepSeek 生产实践活动即将举行**：Packt 正在组织一场 **DeepSeek in Production** 活动，与顶尖工程师和研究人员讨论该模型的速度和效率。
   - [Eventbrite 链接](https://www.eventbrite.com/e/deepseek-in-production-tickets-1436251630289?aff=oddtdtcreator) 详细介绍了在消费级 GPU 上使用 **LoRA + Unsloth** 微调 **DeepSeek models** 的实战工作坊。
- **深入了解 DeepSeek 模型**：**DeepSeek in Production** 活动将涵盖 **MoE**、**MLA**、**FP8** 和 **MTP** 等技术，以解释模型的性能优势。
   - 一位参与者指出，由于 **DeepSeek** 强大的开源支持，该活动前景广阔。
- **BERT 提取模型面临挑战**：一位用户正在开发一个 **基于 BERT 的财务关键词/关键句子提取器**，旨在从财务摘要中精准定位公司信息，如 **Registered Address**、**Province**、**Registration Date** 和 **Business Term**。
   - 他们初步采用的方法（在句子嵌入和任务特定嵌入之间使用 **cosine similarity**）*未能产生令人满意的结果*。
- **提取模型寻求替代方案**：在初始的 **cosine similarity** 方法失败后，该用户正在寻求改进其 **基于 BERT 的财务关键词提取** 模型的建议。
   - 他们正在探索不同的策略，以便从公司财务摘要中精确识别关键信息，如 **registered addresses** 和 **registration dates**。



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Claude Sonnet 4 在 Anthropic 支持下回归**：**Claude Sonnet 4** 已回归并获得来自 **Anthropic** 的官方支持，正如 [在 X 上](https://x.com/windsurf_ai/status/1945599013954490523) 宣布的那样。
   - **Pro/Teams** 用户在 **Editor** 和 **JetBrains Plugins** 中使用 **Claude Sonnet 4** 和 **Claude Sonnet 4** (Thinking) 时，可在限时内享受 **2x** 额度的折扣费率。
- **Wave 11 话题包含情况投票**：[在 X 上](https://x.com/windsurf_ai/status/1945263147994243294) 的一项投票询问社区是否在 **Wave 11** 中包含某个特定话题。
   - 成员可以与该帖子互动以表达他们的意见。



---


**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---



您收到这封电子邮件是因为您通过我们的网站选择了订阅。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。


---

# Discord：按频道划分的详细摘要和链接





### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/)** (1 条消息): 

OpenAI: @everyone
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1394770314420027444)** (1222 messages🔥🔥🔥): 

> `DeepSeek Censorship, IQ Testing, AI's Role in Society, GPT-5 Speculation, AI and North Korea` 


- **DeepSeek 对政府的忠诚引发关注**：成员们讨论了 **DeepSeek AI** 及其倾向于支持中国政府的立场，这导致了严重的审查制度。
   - 针对 DeepSeek 的审查制度引发了担忧，一位成员表示：*与其他 LLM 相比，DeepSeek 的审查程度非常高*。
- **IQ 测试提供衡量指标，Mensa 进入小学**：成员们讨论了 **IQ 测试** 的实用性及其衡量天才的能力，一些人分享了自己的分数和标准差，同时分享了来自 [Mensa IQ 测试](https://test.mensa.no/Home/Test/en-US) 的观点。
   - 一位用户补充道：*学校有时会对孩子进行测试，试图区分出有天赋的个体*。
- **AI 在塑造未来中的潜在作用引发存在主义讨论**：用户辩论了 **AI** 将增强还是削弱人类的能力，涉及工作替代和生产力。
   - 一位成员对未来表示担忧，称：*AGI 将是我们的末日和灭绝级事件。*
- **OpenAI 准备发布令社区兴奋的新公告**：社区推测 OpenAI 即将发布的公告，一些人期待新浏览器、学习功能、新 Agent，甚至是 **GPT-5**。
   - 一位成员预测：*大概是浏览器吧*，并链接到了 [OpenAI Twitter 公告](https://fxtwitter.com/OpenAI/status/1945607177034760574)。
- **朝鲜脱离现实的现状令 AI 工程师感到惊讶**：成员们分享了关于 **朝鲜** 孤立和缺乏技术的惊人事实，其中一人指出，那里的公民被教导 *韩国人吃婴儿*。
   - 他们评论道：*就像他们生活在另一个宇宙中*，并强调了对互联网访问和社交媒体的限制。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1394759781176316075)** (5 messages): 

> `Coding Libraries for AI, Future OpenAI API Integrations, Pro Membership Value` 


- **成员权衡 OpenAI API 集成的等待时间**：一位成员表示，他们将 *等到将其集成到 API 中*，并可能取消 **Pro 会员**，因为目前它提供的价值不多。
   - 他们表示：*Operator 是我购买会员的主要原因之一，但看起来他们并没有真正投入开发，且集成度不高*，因此可能不再使用。
- **用户寻找 AI 代码库位置**：一位成员询问 *用于教 AI 学习新代码的 AI 代码库存储在哪里，以及如何存储和存储在何处*，类似于 **Jan AI**。
   - 消息中未提供任何回复。


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1395102401484161074)** (1 messages): 

> `Aravind and Leonid Reddit AMA, Comet browser` 


- **Perplexity CEO 和产品负责人主持 Reddit AMA**：Aravind Srinivas（CEO）和 Leonid Persiantsev（Comet 产品负责人）在 r/ChatGPT 主持了 Reddit AMA，从 **太平洋时间上午 11 点** 开始，回答有关 **Comet 浏览器** 的问题。
   - 他们讨论了为什么要构建该浏览器、它是如何演进的、核心功能、使用场景以及 Perplexity 的下一步计划；AMA 内容可以通过 [此链接](https://www.reddit.com/r/ChatGPT/comments/1m1javp/comet_ama_with_perplexitys_aravind_srinivas_and/) 找到。
- **Comet 浏览器 AMA 亮点**：此次 AMA 涵盖了构建 **Comet** 的初衷、其演进过程、关键功能、使用场景以及 Perplexity 的未来计划。
   - 参与者深入了解了 **Comet 浏览器** 的战略愿景和开发路线图。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1394755456031002765)** (1020 条消息🔥🔥🔥): 

> `Comet Browser, Image Generation Issues, Samsung Galaxy Store Free Pro, Grok 4 Availability, Comet Agent Saved Interactions` 


- **Perplexity 图像生成故障**：成员们报告了**图像生成**方面的问题，部分用户收到的是文本响应而非图像；一位成员指出，由于团队正在进行改进，网页端的**图像生成功能暂时不可用**，[Perplexity 支持团队也确认了此次故障](https://discord.com/channels/1047197230748151888/1394956308184170516)。
   - 似乎只有在使用 "generate image of" 时会出现错误，而使用 "an image of" 则可以正常工作。
- **Comet 无缝集成 Google 应用**：Comet 浏览器通过 Gemini 与 Google 应用集成，且完全免费，可与你的 **Gmail 账户、电子邮件和日历**同步。
   - 用户可以分析电子邮件、通过设置预约进行交互，甚至可以访问 Gmail 附件。
- **三星向用户赠送 12 个月免费 Pro 会员！**：成员们分享了一个针对**美国**用户的优惠，可通过 Samsung Galaxy Store 获取 12 个月的 **Perplexity Pro**。
   - 然而，有[报告称](https://tenor.com/view/shrug-what-huh-will-smith-i-mean-gif-3535627793955785136)如果用户不在美国境内，可能会面临账号封禁的风险。
- **免费版可能会加入广告？**：用户猜测免费版加入广告的可能性，[Sam 证实了这一点以消除猜测](https://www.reddit.com/r/ChatGPT/comments/1m1javp/comment/n3hnolo/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)，并表示 Perplexity 希望广告与 LLM 响应在视觉上分开，并与 LLM 输出完全独立。
   - 成员们建议采用非侵入式的广告投放方式，例如 UI 中的常规广告或搜索时的赞助广告。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1394844669124280421)** (5 条消息): 

> `Shareable threads, Audio Overviews` 


- **可共享线程提醒**：提醒成员确保他们的线程是“可共享的（Shareable）”，并附带了[原始消息](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825)链接。
   - 分享了几个 Perplexity AI 搜索链接，包括 [review this channel](https://www.perplexity.ai/search/review-the-following-channels-woHqzlFoQL6evyGMV8NpAA) 和 [America's hidden hand in Ukrain](https://www.perplexity.ai/page/americas-hidden-hand-in-ukrain-YKMtnm5EQXaYAIBf25e5Lw)。
- **Audio Overview 替代方案**：一位成员分享了一个研究链接的 [Audio Overview 版本](https://notebooklm.google.com/notebook/8586d048-e5bd-4a3c-acff-ed4660d70c8b/audio)。
   - 随后他们提供了一个 **Perplexity AI** 搜索链接 [postavil cachyos s optsiei no](https://www.perplexity.ai/search/postavil-cachyos-s-optsiei-no-ueINCgXNS1iJh7yMvZZymg#0)。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1395146715241250897)** (2 条消息): 

> `Perplexity Pro, API access` 


- **Perplexity Pro 现在提供 API 访问权限**：用户对于 **Perplexity Pro** 是否提供 **API 访问权限**感到困惑。
   - 一位成员链接到了 [Perplexity AI 帮助中心](https://www.perplexity.ai/help-center/en/articles/10352901-what-is-perplexity-pro)，其中指出 **Perplexity Pro** 每月提供 **5 美元额度**用于 **Sonar 模型**。
- **API 访问详情**：Perplexity Pro 提供的 API 访问权限允许用户将 AI 驱动的搜索嵌入到他们自己的项目中。
   - 这种访问权限包括获取引用的能力，增强了研究导向型任务的实用性。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1394756021410332772)** (552 messages🔥🔥🔥): 

> `Cursor 计费与定价, Kimi K2 模型讨论, Cursor 性能问题与故障排除, 多 Agent 协作与代码管理, 有效使用 AI 的 Context 工程` 


- **Cursor 的额度危机：20 美元已今非昔比**：用户报告称 Cursor 的使用情况[仪表盘](https://cursor.sh/dashboard)存在[滞后](https://cursor.sh/dashboard)，在 **$20** 额度耗尽后仍显示“已包含（included）”，计费仅在下一个周期才同步；用户现在会在达到计费点时收到聊天内通知，尽管有时会有延迟。
   - 一些用户发现 **Ultra 计划的限制**过于严格，质疑其长期可行性。用户很快就会达到上限，例如一位用户超额了 **$60**，而另一位用户则开玩笑说在加入当天就用完了 Ultra 计划额度。
- **Kimi K2：抢走 Sonnet 风头的 Groq 明星？**：成员们正在热烈讨论通过 **Groq** 使用的 **Kimi K2**，认为它是 **Sonnet** 的潜在更优且更便宜的替代方案。据称其具有 **Opus 级别**的 Agent 任务性能，拥有 **256K tokens** 的超长 Context 窗口，速度达 **每秒 250-300 tokens**。不过有人指出，Groq 上的 Kimi K2 在工具调用（tool calls）方面不如 Moonshot 原生版本。
   - 然而，Kimi K2 不支持视觉输入，这被一些人视为缺点。
- **Cursor Composer 的难题：Prompt 卡死与使用困扰**：用户遇到 Cursor 卡在 Prompt 上导致浪费额度的问题。一位成员报告称，新版本应该具备 **180 秒超时机制**，该机制会自动取消卡住的请求且不向用户计费。
   - 一些用户建议将重启 Cursor、重新安装或使用新的 Prompt 要求其“重新思考（rethink）”作为潜在解决方案。当它卡住或似乎在重复同样的问题时，只需写一个新的 Prompt 让其重新思考，这会起到某种“重新触发”的作用，随后即可正常工作。
- **Agent 集结！多 Agent 系统成为焦点**：一位用户正在开发一个 **多 Agent MCP 服务器**，用于管理跨多个 IDE 的异步 Agent 任务，旨在解决状态持久化问题。他分享说自己使用 **gemii** 进行调试，使用 **grok4** 进行重度重构，并使用 **Claude** 进行编码。
   - 另一位用户发现通过修改 Cursor 的 `main.js` 和其他 IDE 来创建一个极简 MCP 非常高效，使用 JSON 进行输入和输出，并在启动时将它们全部连接。该用户补充说，如果真的能跑通，那简直太酷了 😆。
- **Context 工程：资深 Agent 用户的“秘密武器”**：用户正在讨论为 AI Agent 提供正确 Context 的重要性，即 **Context 工程**。建议在第一个 Prompt 中提供所有必要信息，而其他人则强调在 System Prompt 中保留一套简单的规则或参考文档（ref docs）效果更好。
   - 据观察，给模型过多的 Context 可能会产生负面影响，因为过多的规则可能会导致 Context 在长期对话中变得松散。

---

### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1394882909290102814)** (8 messages🔥): 

> `Cursor Runtime 环境, 自定义 PR 标题, Start Script 输出未显示, 重新配置手动快照错误, Cursor 获取后台 Agent` 


- **Cursor 的修复在 Runtime 生效！**：Cursor 的最新修复解决了一位用户的问题，确认了**修复程序在 Runtime 环境中可用**，但在交互式环境设置期间是否可用仍不确定。
- **配置 PR 标题？**：一位用户报告在尝试自定义 **PR 标题** 时遇到困难。
- **Start Script 输出被隐藏？**：一位用户质疑为什么 **start script 的输出没有显示**，并指出虽然安装输出可见，但启动输出却不可见，尽管其中包含有价值的信息。
- **快照重新配置失败！**：一位用户报告在点击**重新配置手动快照（reconfigure manual snapshot）**时遇到错误，导致环境从空白状态重新构建。
- **Cursor 获取 Agent 失败**：一位用户报告在获取**后台 Agent（background agents）**时等待了 8-10 分钟。

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1394756563171938434)** (267 条消息🔥🔥): 

> `BF16 vs FP32 for LoRA fine-tuning, Qlora as a constraint, Gemma 3 pretraining, Kimi audio distillation, Overfitting solutions` 


- **LoRA 训练中 BF16 比 FP32 更节省 VRAM**：在进行 **LoRA** 微调时，除非在旧型号 GPU 上使用 **Gemma 3**，否则使用 **bf16** 优于 **fp32**，因为 **bf16** 可以减少 VRAM 占用。
   - 一位用户指出，**fp32** 格式的 **7b model** 可能占用高达 **28GB** 的 VRAM，这在 **A100** 等硬件资源有限的情况下可能会产生问题。
- **DeepInfra 以低廉价格提供 B200 GPU**：一些用户讨论了从 [DeepInfra](https://deepinfra.com/low) 以 **$2/小时** 的促销价格租赁 **B200 GPU**。
   - 该促销最初适用于除内布拉斯加州以外的所有地区，可能归因于 [一条推文](https://x.com/DeepInfra/status/1935459646493573123) 需要修正。
- **针对过拟合进行微调**：一位用户寻求关于在使用小数据集微调 **llama3.2** 时缓解过拟合问题的建议，当时模型无法泛化改写后的问题。
   - 一位用户推荐了 [Unsloth 微调指南](https://docs.unsloth.ai/get-started/fine-tuning-guide/lora_hyperparameters_guide#avoiding-overfitting-and-underfitting)，强调超参数调优通常是一个反复试验的过程。
- **TurboDerp 发现阿里巴巴的无损 2bit Compression 并无益处**：关于 [阿里巴巴针对 ERNIE 4.5 模型的无损 2bit 压缩](https://huggingface.co/turboderp/ERNIE-4.5-300B-A47B-PT-exl3) 有一些炒作，部分用户对其能与 **gguf** 和 **LlamaCPP** 配合使用感到兴奋。
   - 然而，[TurboDerp](https://huggingface.co/turboderp/ERNIE-4.5-300B-A47B-PT-exl3) 表明实际平均值更接近 **2.5 bit**，且某些层仍保持较高精度，真正的 **EXL3** 表现更好。
- **Unsloth 的基准测试结果高于 Liger-Kernel**：一位用户对 **Unsloth** 与 **Liger-Kernel** 进行了基准测试，结果显示 **Unsloth** 可额外节省 **15-30%** 的 VRAM。
   - 他们还附上了一张训练成功后的图片，展示了使用 **Unsloth** 梯度检查点（**gradient checkpointing**）实现的惊人上下文长度。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1394842278845087804)** (18 条消息🔥): 

> `OpenPipe ART, LLM as a Judge, Agentic Models, ARTwell RULER, Model Finetuning` 


- **OpenPipe ART 声称能让模型更具 Agent 能力**：一位成员分享了 [OpenPipe ART](https://github.com/OpenPipe/ART) 的链接，该工具声称能让任何模型更具 **Agent** 特性。
   - 他们最初表示怀疑，但后来承认该工具“实际上非常有趣”且使用了 **Unsloth**。
- **LLM as a judge**：该工具使用 **LLM as a judge**，一位成员认为这很酷。
   - 一位成员对相关的 LinkedIn 帖子表示怀疑。
- **ARTwell RULER 测试即将进行**：一位成员一直在关注 **OpenPipe** 及其微调工具，并打算尝试 **ARTwell RULER**。
   - 另一位成员确认“它相当不错”。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1394760640626950297)** (166 messages🔥🔥): 

> `Unsloth 修复, VLLM 缓存, Kaggle Mistral notebook 错误, 海量 VRAM 建议, Llama.cpp 多 GPU 配置` 


- ****Unsloth 更新被认为需要修复****：成员们报告了 [最新的 Unsloth 更新](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs) 的问题，特别是在模型下载过程中的 **Timeout 错误**。
   - 建议包括使用 `huggingface-cli` 或将环境变量如 `HF_XET_HIGH_PERFORMANCE` 设置为 `1` 以缓解下载问题。
- ****VLLM 缓存导致许多错误****：一位用户报告在运行多个指向同一个 **VLLM 缓存目录** 的训练脚本时，遇到了 *缓存损坏错误 (corrupted cache errors)*。
   - 不幸的是，目前无法通过环境变量更改 VLLM 缓存目录，因为它默认指向 `.cache/vllm/torch_compile_cache`。
- ****RTX 50 Blackwell 手动指令，不要偷懒！****：一位用户询问 **Unsloth 对 RTX 50 GPU 的支持**，另一位用户回答说需要按照 [文档](https://docs.unsloth.ai/basics/fine-tuning-llms-with-blackwell-rtx-50-series-and-unsloth) 中的说明进行手动操作。
   - 他们表示 *我不认为这一定是个坏主意；目前看来 xformers 似乎是唯一需要从源码构建的东西。*
- ****全量微调需要精确的算力计算****：一位用户想要对 **gemma 3 4b** 进行 **全量微调 (full finetune)**，但在 3090 上显存溢出。另一位用户回复说 *你不能对 4bit 量化模型进行全量微调，必须使用全精度*。
   - 另一个选项包括使用 **16bit LoRA 训练** 并参考 [Unsloth notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks) 获取兼容的架构。
- ****Optuna 永远是答案！****：一位用户询问是否可以使用像 **Optuna** 这样的超参数调优库与 Unsloth 配合，以找到适合其 GPU 的最佳 batch sizes 和梯度累积 (gradient accumulation) 设置。
   - 回复是肯定的，指出 *我不明白为什么 Optuna 不适合你*。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1394976828631945296)** (1 messages): 

> `播客公告, 社区参与` 


- **播客即将发布！**：第一期播客节目即将上线！查看 [公告推文](https://x.com/himanshustwts/status/1945416091582505377) 了解详情。
- **社区期待升温**：即将推出的播客公告在社区内引发了兴奋。
   - 成员们正热切期待发布，正如 <:slothhearts:1253009235600736296> 反应所表达的那样。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1394830874054361198)** (13 messages🔥): 

> `ETHOS 论文, LLM 精神错乱, 独立研究挑战, TTS 模型游乐场` 


- **ETHOS 论文发布在 Arxiv**：一位成员在与安全人员讨论并获得许可控制后，分享了他们即将在 [Arxiv 发表的关于 ETHOS](https://github.com/wrmedford/ETHOS) (Efficient Transformers via Hypernetwork Organized Sparsity) 的论文。
   - 附件为 PDF 文件 [ETHOS___Efficient_Transformers_via_Hypernetwork_Organized_Sparsity_10.pdf](https://cdn.discordapp.com/attachments/1257011997250424842/1394830874142576712/ETHOS___Efficient_Transformers_via_Hypernetwork_Organized_Sparsity_10.pdf?ex=68798e7b&is=68783cfb&hm=0b2c8891c328a38668ff0d015cf8a9f8ef5b80884a3f7eb0ee486aa149b25e97&)。
- **LLM 精神错乱 (LLM Psychosis) 确实存在**：一些成员将这种情况称为 "**LLM 精神错乱 (LLM psychosis)**" —— 即有人认为自己发现了什么，但实际上只是从与谄媚的 LLM 对话中产生的技术胡言乱语。
   - 另一位成员将 *psychosis* 定义为 *一种以脱离现实为特征的精神障碍*，并指出这可能是由 LLM 幻觉强化了误解或错误分类事物导致的。
- **独立研究受阻**：一位成员表示，由于“疯子”横行，进行真正的独立研究非常困难，没有 PhD 学位或大型实验室背景的人的突破往往会立即被忽视。
   - 他们将此归因于 *目前 95% 做研究的人都* 患有 **"LLM 精神错乱 (LLM psychosis)"**。
- **Snortts Indic V0 TTS 游乐场上线**：一位成员分享了一个用于测试 **TTS 模型** 的游乐场：[https://snorbyte.com/snortts-indic-v0](https://snorbyte.com/snortts-indic-v0)。
   - 邀请用户进行尝试，并询问有关训练或部署 TTS 模型进行推理的问题。


  

---

### **Unsloth AI (Daniel Han) ▷ #[unsloth-bot](https://discord.com/channels/1179035537009545276/1390899684834410536/1394931626638708826)** (24 条消息🔥): 

> `Llama.cpp 量化错误，VLLM 缓存目录配置，Qwen 2.5 7B 推理，Torch 缓存存储与损坏，Qwen 2.5 7B 模型大小` 


- **Llama.cpp 量化抛出 RuntimeError**：一名成员遇到了 `RuntimeError`，提示 `llama.cpp/llama-quantize` 文件不存在。
- **VLLM 缓存需要独立目录**：一位用户报告称，在运行多个具有不同配置的训练脚本时遇到了缓存损坏错误，因为它们都指向同一个 **VLLM** 缓存。
   - 他们询问了是否可以通过环境变量更改 **VLLM** 缓存目录，并指出其默认路径为 `.cache/vllm/torch_compile_cache`。
- **运行 Qwen 2.5 7B 推理的请求**：一名成员询问如何运行 **Qwen 2.5-Omni-7B** 进行推理。
- **Torch 缓存默认存储位置**：一位用户询问 **Unsloth** 将 **torch** 缓存存储在哪里。
   - 该用户意识到他们需要为不同的训练脚本更改 `$HOME` 变量，并询问了如何通过不同的 GPU 可见性运行多个训练脚本以避免 **torch** 缓存损坏。
- **Qwen 2.5 7B 模型大小揭晓**：一位用户询问了 **Qwen 2.5 7B** 模型的存储大小。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1394761464719343761)** (66 条消息🔥🔥): 

> `ZR1-1.5B 模型，Pythia 12B vs 2.8B，TensorFlow 的衰落，AI 研究管理` 


- **ZR1-1.5B 模型被推荐用于推理**：一名成员建议根据具体需求使用 [ZR1-1.5B 模型](https://huggingface.co/Zyphra/ZR1-1.5B) 处理推理任务。
   - 他们还提到 **DeepCoder-1.5B** 是一个可靠的选择，但提醒在 **7B** 参数预算内实现通用推理具有挑战性。
- **Pythia 12B 拥有更大的词表大小**：一名成员对比了 **Pythia 12B** 和 **2.8B** 模型，发现 **12B** 模型的词表（vocabulary）大小更高。
   - 据澄清，词表大小的差异（例如 **50,688** vs. **50,257**）通常是由训练硬件决定的，并不一定意味着 **tokenizer** 不同。
- **TensorFlow 失宠**：一名成员表示，在职位要求中使用 **TensorFlow** 正在变成一种“危险信号（red flag）”，因为许多人目前已经*发誓不再使用 tensorflow*。
   - 另一位用户表示赞同，称没有充分的理由在 **PyTorch** 或 **JAX** 之外选择使用 **TensorFlow**。
- **研究经理防止“无尽刷屏（Doomscrolling）”**：一名成员开玩笑地谈到了研究经理的角色，但一份详尽的回复描述了优秀研究经理的好处，例如提供实际指导、处理官僚事务以及把握大局。
   - **EleutherAI** 的最终目标是为独立/业余研究人员提供研究管理。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1394929212892254291)** (316 messages🔥🔥): 

> `nanoGPT speedrunning, recursion papers, tuning LLM inference-time hyper-parameters, peer review on research, MoE like things in RWKV-8` 


- **对 NanoGPT 有效性产生质疑**：一位成员询问某个特定方法是否在 **nanoGPT speedrunning** 设置中进行了测试，并引用了关于 **nGPT** 发布时有效性的[争议](https://arxiv.org/abs/2507.10524)。
   - 另一位成员对**递归论文 (recursion papers)**表示沮丧，指出虽然这些论文提供了迭代改进，但往往忽略了如何改进递归本身，可能导致*次优*结果。
- **输入注入提升递归神经网络 (Recursive NN) 性能**：一位成员建议，改进“如何递归”的一个顶级方法是**在每次迭代时进行朴素输入注入 (naive input injection)**，作为 [skip connection](https://link.to.skipconnection) 以更轻松地传播状态。
   - 他们澄清这涉及注入 **hidden state**、更早的 hidden state 或原始输入本身，除了简单的拼接外，还有更智能的混合方法。
- **寻求同行评审面临挑战**：一位成员询问如何请求对**研究进行同行评审**，特别是针对一个涉及 GPU 架构和 MoE 模型的跨学科项目，并提到了一份[待处理的 arXiv 提交](https://arxiv.org/abs/2505.22014)和可用代码。
   - 然而，频道明确表示由于大量的 *crank*（无意义）投稿，不接受评审请求，但欢迎讨论研究内容，并建议分享 [GitHub repo](https://github.com/wrmedford/ETHOS) 以获取意见。
- **潜码 (Latent Codes) 使专家模型变为瞬态**：一位研究员详细介绍了他们将专家模型存储为**潜码 (latent codes)**并在运行中恢复的方法，分享了 [GitHub 仓库](https://github.com/wrmedford/ETHOS)，并强调在 GH200 上的训练速度达到 **15K tokens/second**。
   - 20 倍的 FLOPs 减少是理论上的加速，目前尚未在实证中实现，因为受限于次优的 backward 过程，其中*专家模型仅瞬时存在*且不接收梯度，autograd 正在存储中间变量。
- **小尺寸模型迅速击败 MoE 模型**：成员们讨论了在计算资源有限的情况下，将新架构与基准模型进行比较的策略，建议包括将实验缩减到 toy examples，并与更小的 dense 模型（如 [GPT-2-small](https://link.to.gpt2small)）进行比较。
   - 共识是，如果能击败总尺寸相同的 MoE 或具有相同激活参数的 dense 模型，将是一个重大胜利，并强调需要调整学习率和初始化方案等超参数以进行公平比较，[Olmo-2-1b](https://link.to.olmo21b) 被提议为一个可靠的基准候选。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1394938478181220484)** (8 messages🔥): 

> `Function Vectors, nnterp package, Transformer models` 


- **Function Vectors 驱动 ICL 性能**：一位成员遇到了[这篇论文](https://arxiv.org/abs/2502.14010)的作者，该论文认为 **induction heads** 实际上并不驱动 ICL 性能，而是由一种称为 **Function Vectors** 的机制驱动。
   - 另一位成员将 **induction heads** 定义为*“精确复制一个 token”*，但发现它们在*“利用上下文深处的历史信息来预测接下来的内容这一通用现象”*中发挥了重要作用。
- **新的 Mech Interp 包 nnterp 发布**：一位成员在 [Github 上](https://github.com/Butanium/nnterp)发布了他们的 Mech Interp 包 **nnterp** 的 beta 1.0 版本，可以通过 `pip install "nnterp>0.4.9" --pre` 安装。
   - 该包旨在为所有 **Transformer 模型**提供统一接口，同时在底层仍使用 HuggingFace 实现，缩小了 `transformer_lens` 和 `nnsight` 之间的差距，并提供了 [demo colab](https://colab.research.google.com/github/Butanium/nnterp/blob/main/demo.ipynb) 和[文档](https://butanium.github.io/nnterp/)。
- **nnterp 中的鲁棒测试系统验证模型**：**nnterp** 包包含一个鲁棒的测试系统，在加载模型时自动运行验证测试，以确保所有 hooks 返回预期的 tensor shapes，并且每个 token 的 attention probabilities 正确求和为 1。
   - 该包包含 **1915 个预计算测试**，涵盖了来自不同架构的 toy models，如果任何测试失败，在模型加载期间会发出明确警告。


  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1395141917066399865)** (1 条消息): 

> `Harness Evaluation, IFEval Suite` 


- **Harness 产物受到审查**：一位成员询问某个流程是否符合 Harness 的规范，并指出 Harness 通常不产生外部产物，除非是可选的模型请求缓存、HF Hub 资源或用于 HF `evaluate` 指标的远程代码。
   - 他们要求特定用户在理解有误时予以纠正。
- **Dynamic IFEval 受到质疑**：一位成员询问了 Dynamic 版本的 **IFEval** 相比标准 **IFEval** 套件的优势，并对 Harness 评估中的可复现性和确定性提出了疑问。
   - 他们指出，Harness 中的大多数评估都应该是可复现且具有确定性的。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1395051703333687329)** (25 条消息🔥): 

> `Transformer Engine performance, Slurm and containers with GPT-NeoX, CUDA drivers in NGC containers, DeeperSpeed Slurm runner` 


- **TE 性能骤降？**：一位成员测试了不使用 **Transformer Engine (TE)** 的运行情况，发现存在*显著的性能差异*（[wandb 链接](https://wandb.ai/eleutherai/AISI/runs/nmk3zrpr/overview)），尽管可能存在*硬件干扰因素*。
   - 有建议认为可能是 TE 设置的问题，特别是当前仓库 `/NS/llm-pretraining/work/afkhan/RoPE_Pct/gpt-neox` 中支持的 TE 是否已损坏。
- **Slurm 与容器用于 GPT-NeoX：黄金组合？**：多位成员确认，通过 **Slurm** 配合容器（**Docker** 或 **Singularity**）运行 **GPT-NeoX** *效果良好*。
- **DeeperSpeed 发布 Slurm 运行器**：**DeeperSpeed** 添加了一个 **Slurm 运行器**，该运行器使用 `srun` 代替 `mpirun`，简化了多节点设置（[DeeperSpeed 运行器](https://github.com/EleutherAI/DeeperSpeed/blob/65d9f99249f79ebd7c4577b6aeb0d3dff5a1cef6/deepspeed/launcher/multinode_runner.py#L413)，[GPT-NeoX Slurm 指南](https://github.com/EleutherAI/gpt-neox?tab=readme-ov-file#slurm)）。
   - **GPT-NeoX** 仓库还提供了 [容器化设置指南](https://github.com/EleutherAI/gpt-neox?tab=readme-ov-file#containerized-setup)。
- **NGC 容器中的 CUDA 驱动注意事项**：一位成员想知道 NGC 容器是否自带 **CUDA 驱动**，并提到他们系统的默认驱动*低于 12 版本*。
   - 在不使用容器时，他们通常手动指向 **12.1** 的安装路径，因此不确定如何在容器内应用相同的修复方法。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1394878607322644562)** (3 条消息): 

> `o1-preview Deprecation` 


- **`o1-preview` 将于 2025 年 7 月 28 日停用**：`openai/o1-preview` 接口已弃用，并将由 **OpenAI** 在 **2025 年 7 月 28 日**正式关闭。
   - 来自 **OpenAI** 的最新 o 系列模型可在[此处](https://openrouter.ai/openai)获取。
- **通过 API 获取弃用信息**：一位成员询问是否可以通过 API 获取弃用信息。
   - 目前尚不确定是否会实现此功能。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1394758775361241088)** (357 条消息🔥🔥): 

> `Deepseek R1 质量下降，OpenRouter AutoRouter 隐私担忧，模型弃用通知，GPT 3.5 Turbo 端点消失，Claude Opus 4 每周 Token 使用量` 


- **用户揭露 OpenRouter 的 SwitchPoint Router 隐私失误**：一名成员报告称，**SwitchPoint Router** 在未经其同意的情况下被预选，这可能违反了他们的 **NDA**，并在添加防火墙规则禁用 OpenRouter Chat（因为其*不够安全*）后将 tokens 发送到了中国。
   - OpenRouter 管理员回应称，Switchpoint 总部位于**美国**而非中国，用户可以在 [settings](https://openrouter.ai/settings/preferences) 中禁用供应商，且自动路由是在 OpenRouter 的高质量模型列表中进行选择。
- **OpenRouter 的使用指标并非总是实时的**：一位用户疑惑为什么他们的 **OpenRouter activity** 没有实时更新，并指出 DeepSeek 的活动与最近使用 Devstrall 和 K2 进行的 API 测试之间存在差异。
   - 另一位用户确认遇到了同样的问题。
- **为什么 OpenAI 的 GPT 3.5 Turbo 端点消失了**：一位用户指出 `openai/gpt-3.5-turbo` 端点消失了，寻求澄清，并提到其他供应商本可以提供服务。该端点在 **2025-06-23** 之前一直有成功的国际象棋记录。
   - OpenRouter 管理员回应称，他们正在调查此问题，并已将其**恢复**以供未来使用。
- **用户标记 412 美元的未经授权充值**：一位用户报告其账户出现了 **412 美元** 的未经授权充值，且在尝试查看发票时出现 **404** 错误，正在寻求协助调查该费用并确保账户安全。
   - OpenRouter 管理员解释说，这不是扣费，而是退款，请检查垃圾邮件文件夹。
- **DeepSeek 质量随 Q4 量化大幅下降**：一位用户注意到 **Deepseek R1 0528** 在角色扮演（roleplay）中的质量显著下降，模型在较低量化（quantizations）下幻觉（hallucinating）更加严重。
   - 另一位用户表示赞同，指出类似问题并回忆起 **R1 表现极其糟糕**，目前正在进行 **Q4** 与 **fp8** 的对比测试。


  

---


### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1394768983391207535)** (41 条消息🔥): 

> `质量认证徽章，速度认证徽章，模型基准测试的 Eval Harness，上下文压缩猫腻，工具调用基准测试` 


- **OR 制定质量和速度认证徽章**：OpenRouter 正在探索模型的质量和速度认证徽章，类似于 `:nitro` 过滤器，以解决不同供应商之间 **Kimi K2** 速度差异巨大的问题（**10 TPS** vs **100 TPS** vs **500 TPS**）。
   - 目标是突出具有可靠 tool calling、稳定 ratelimits 和高质量输出的供应商，同时考虑量化和潜在的 tool call 失败，新模型将从“未验证”等级开始。
- **Eval Harness 追踪模型基准测试偏差**：提议的 eval harness 将以模型作者发布的基准测试为基准，持续测量官方分数与端点分数之间的*偏差（drift）*或差异，以验证 [OpenRouter](https://openrouter.ai/) 上的模型性能。
- **对上下文压缩猫腻进行基准测试**：建议进行高达 **128k context** 的长篇小说基准测试，以验证是否存在上下文压缩（context compression）猫腻，同时通过提示模型输出大量 tokens 的测试，来确认供应商声明的输出 token 数量。
- **工具调用基准测试 Tau-2 airline**：推荐使用如 [Tau-2 airline](https://github.com/sierra-research/tau2-bench) 的工具调用基准测试（受 [tweet](https://x.com/the_bunny_chen/status/1944851548712133032) 启发），以检测并解决 tool use 聊天模板（chat template）的 bug。
   - 例如，实现跳棋或国际象棋，这些可以通过非常简单的评估标准测试模型的许多随机特性。
- **Baseten 延迟转化为重试**：有一个处理 **429s** 错误的想法，是将延迟数字转化为基于重试的*预期*延迟，以解决由于频繁的 **429s** 和高延迟（如 **Baseten**）导致 prompt 处理时间过长的情况。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1394755664672329949)** (337 条消息🔥🔥): 

> `Polymarket 投注失败、预测市场合法性、Grok4 性能、Kimi K2 性能与定价、LMArena 问题与反馈` 


- **Polymarket 豪赌未获回报**：成员们对他们的 **Polymarket** 投注未能成功表示失望，并对服务器中竟然有人不投资 **Google** 感到惊讶。
   - 一位用户幽默地表示：*我在 polymarket 上的雄心勃勃的投注没能成功 💀 我很震惊这个服务器里居然真的有人不买 Google 的股票 💀*。
- **预测市场监管受质疑**：用户讨论了预测市场在美国的合法性，一些人认为尽管存在流动性问题，但像 **Kalshi** 这样的平台提供了一个合法的变通方案。
   - 一位用户表示：*据我所知，它们受 CFTC 监管，在美国是合法的*，而另一位用户则对潜在的税务机关问题表示担忧。
- **Grok4 过于啰嗦的性能评价不佳**：**Grok4** 因过于冗长且产生大量隐藏推理而受到批评，导致用户感知到的实际表现与声称的 Benchmark 结果之间存在脱节。
   - 一位用户评论道：*老实说，xAI 内部不管是谁觉得让模型输出 3 万多个字符的隐藏推理，然后只用一个词或一句话来回答是个好主意，那他真是个白痴。*
- **Kimi K2 的效率令人惊叹**：成员们讨论了 **Kimi K2** 令人印象深刻的效率和编程能力，指出其相对于 **GPT-4.1 mini** 具有竞争力的定价以及显著的性能优势。
   - 一位成员指出：*中国人在效率方面确实做得非常出色*，而且该模型能够生成有趣的物理沙盒。
- **LMArena 的问题与 UI 思考**：用户报告了 **LMArena** 的问题，包括模型报错以及内容过滤器误报（特别是针对漫画内容）。
   - 用户还对新 UI 提出了反馈，特别是关于直接对话中的模型选择，社区管理员引导用户前往反馈贴进行讨论。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1395088960748912802)** (1 条消息): 

> `UI 改进、排行榜导航、精简界面、紧凑侧边栏` 


- **轻量级 UI 改进上线**：新的 **UI 改进** 现已上线，包括旨在减少杂乱的**精简界面**、可快速访问关键部分的**紧凑侧边栏**，以及根据社区反馈优化的排行榜标签页**导航体验**。
   - 观看视频：[ArenaV2 发布视频](https://cdn.discordapp.com/attachments/1343296395620126911/1395088959448809482/ArenaV2_LaunchVideo_Under30s_1.mp4?ex=68792d57&is=6877dbd7&hm=8168774d3683077ff598824196cb6070bb39207f22ac06b02dbc803453f621c0&)。
- **导航升级助力排行榜**：作为 UI 改进的一部分，排行榜现在拥有更好的导航功能，以便用户更快速地访问各项榜单。
   - 这些改进旨在根据社区反馈，使整体体验更加精致、直观且令人愉悦。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1394789349958357022)** (220 messages🔥🔥): 

> `Kimi UI, Atlassian Rovo Dev AI Agent, Flux Pro and Seedance Realistic AI Videos, Anthropic hires back Cursor developers, OpenAI API Face Editing` 


- **Kimi UI 的苦涩开端**：一篇 [Medium 文章](https://medium.com/@xinyijin715/maker-story-the-bitter-lessons-behind-kimi-researchers-ui-6654ec66662c) 分享了 Kimi UI 开发背后的故事和**苦涩教训**。
   - 根据一名成员引用的一段 [YouTube 视频](https://www.youtube.com/watch?v=motX94ztOzo&t=2737s)，Cursor 的代码似乎与此有关。
- **Atlassian 的 Rovo Dev：一个天生无能的 AI Agent？**：Poonam Soni 介绍了 **Atlassian 的新 AI Agent** Rovo Dev，可通过 CLI 访问，旨在通过**代码生成、评审、重构、文档、调试和任务自动化**等功能改变软件开发。
   - 尽管宣传火热，一位成员调侃道 *Atlassian 骨子里就做不出好产品*，而其他人则对该工具的**下载流程和企业级定位**表示沮丧。
- **Flux Pro 和 Seedance 结合生成逼真视频**：一位用户尝试结合 **'flux pro' 和 'seedance' 模型**来生成逼真视频，使用了 **'IMG_XXXX.JPG' 技巧**作为起始图像，并使用了类似 *'极其平庸的家庭录像'* 的提示词。
   - 讨论中包含了一个 Replicate 模型的链接，其他用户反响积极；关于这项技术背后的公司仍存在疑问。
- **Boris & Cat 重返 Anthropic**：据 [The Information](https://www.theinformation.com/briefings/anthropic-hires-back-two-coding-ai-leaders-cursor-developer-anysphere) 报道，**两名编程 AI 领域的领导者**在 Cursor 短暂任职后已被 Anthropic 重新聘用。
   - 这一举动引发了幽默的猜测，人们怀疑他们是否一直是*双重间谍*，并调侃 Cursor 可能从这些*世界级专家*那里得到了*廉价的咨询服务*。
- **OpenAI 预热 Operator/AURA**：OpenAI 发布了一段神秘的视频预告，内容指向 **2025 年 7 月 16 日**发布的某些东西，引发了广泛猜测，从 **浏览器/Operator 升级、新的 Waifu 或 AURA Agent，到 AGI 阶段的揭晓**。
   - 猜测范围涵盖了 **浏览器/Operator 升级**、新的 *Waifu* 或 *AURA* Agent，以及 **AGI 阶段的揭晓**，而一位成员则认为与 **xAI 的进展**相比，这只是*重复的炒作*。


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1395145170672029808)** (1 messages): 

> `YouTube video` 


- **为团队分享的视频**：一位成员为团队分享了一段 [YouTube 视频](https://youtu.be/uIKmG3M0X3M?si=ndL7_jJKzI5FKueG)。
- **另一个视频**：一位成员在[这里](https://youtu.be/uIKmG3M0X3M?si=ndL7_jJKzI5FKueG)为团队分享了另一个视频。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1394757252673568848)** (71 messages🔥🔥): 

> `GPU 上的 Radix Sort、Serverless GPU 平台、CUDA Kernels 性能分析、RMSNorm 和 Reduce 算子的工业级实现、PyTorch 与高效 CUDA kernels 的对比` 


- **Radix Sort 的 GPU 并行化技术**：一位成员寻求在 GPU 上实现并行 **Radix sort** 的建议，另一位成员推荐了 [《Programming Massively Parallel Processors》第 13 章](https://www.amazon.ca/Programming-Massively-Parallel-Processors-Hands/dp/0323912311/) 作为参考资源。
   - 该书演示了 **2-bit radix sort**，并挑战读者将该概念扩展到 **8-bit 实现**。
- **寻找 Serverless GPU 平台**：一位成员询问是否有 **serverless GPU 平台**可以上传并在远程 GPU 上运行代码，同时具备深入研究底层 **CUDA** 的能力。
   - 虽然推荐了 **Modal**，但该成员正在寻找对 **tiled memory** 和其他高级 CUDA 特性有更好性能分析（profiling）能力的平台；有人建议使用 Google Cloud Run，但它无法提供完整的性能分析访问权限。
- **CUDA Kernels 性能分析的限制**：由于安全漏洞，共享 GPU 平台通常限制完全的性能分析访问权限（**sudo** 权限），这影响了使用 **nvidia-profiler** 等工具的能力。
   - 替代方案包括记录**运行时间（timing runs）**，或者作为最后的手段，找一个*关系非常好*的朋友在他们的个人笔记本电脑上用 NCU 运行 Kernel。
- **RMSNorm 和 Reduce 算子的实现**：对于 **RMSNorm** 或 **Reduce** 等算子的业界领先实现，提到了用于 1D reduce、sort 和 scan 算法的 **CUB** 和 **rocPRIM**。
   - 对于 **AMD**，**RMSNorm** 的“专业”实现在 **MIOpen** 中，而在 CUDA 中则是 **cuDNN**（闭源）；PyTorch 也有实现，尽管其 Kernels *通常被认为效率不是很高*。
- **PyTorch 的效率权衡**：尽管 CUDA Kernels 的效率可能较低，但 **PyTorch** 依然流行，因为它优先考虑**准确性而非速度**，从而减少了研究期间的调试时间。
   - 虽然 PyTorch 在 **Aten** 或 **C10** 中的原生 Kernels 可能没有完全优化，但许多操作会调用 **cuDNN** 和 **CUB** 等高效库，用户也可以换入自己的 Kernels；这里有一个 [用户提供 Kernels 的示例](https://github.com/wrmedford/ETHOS/blob/main/kernels.py)。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1395022405910663378)** (3 messages): 

> `triton-autodiff 工具` 


- **发现内存高效的反向 Kernel 生成工具！**：一位成员询问是否有工具能从前向 Kernels 内存高效地生成反向 Kernels，以避免 autograd 产生浪费的中间值。
   - 另一位成员建议使用 [triton-autodiff](https://github.com/IaroslavElistratov/triton-autodiff) 作为解决方案。
- **链式法则的替代方案**：用户发现该工具是替代记忆链式法则（chain rule）的好帮手。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1394807337864134790)** (6 messages): 

> `Torch Compile 调试、Torch Inductor 问题` 


- **Torch Compile 调试无输出**：当启用 **cache** 时，从缓存中获取的编译结果不会重新编译任何内容，因此不会显示 **logging**，因为日志记录是在编译时进行的。
   - 环境变量 `TORCHINDUCTOR_FX_GRAPH_CACHE=0` 可能会解决 `TORCH_COMPILE_DEBUG=1` 无输出的问题，因为缓存机制已经改进，**缓存的内容比以前更多了**。
- **Torch Inductor 在 Blackwell 上遇到问题**：一位用户提到他们最近在使用 **Blackwell** 时遇到了很多 **inductor** 问题，必须使用 nightly 版本（或 branch cut 2.8）。
   - 他们不确定是否是 inductor 的问题，并询问是否有最近的 GitHub issues 报告了类似问题。


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1394764261007556738)** (3 messages): 

> `并行 Radix Sort、使用 CUDA 在 OpenGL 中进行流体模拟` 


- **并行 Radix Sort 寻求指导**：一位成员询问如何创建一个专门为**有符号整数**设计的有效**并行 Radix sort**。
   - 该成员指出，负整数在排序中应被视为无效。
- **征求 OpenGL 流体模拟算法**：一位成员就适用于 **OpenGL** 中利用 **CUDA** 进行计算的**流体模拟**算法寻求建议。
   - 在现有消息中没有推荐具体的算法。


  

---

### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1394768690540708041)** (3 条消息): 

> `GPU Engineering Book, Software Engineer Hiring, Technical Reviewers` 


- **Voltage Park 招聘 Software Engineer 及更多职位**：Voltage Park 正在招聘多个职位，包括 Software Engineering 和 Security 角色，主要为 **WFH**，并设有部分全球技术支持职位；详见 [Voltage Park Careers](https://www.voltagepark.com/careers)。
   - 优先考虑美国申请者，但也提供一些全球技术支持职位。
- **GPU Engineering 书籍招募技术审阅者**：一名成员正在为 Packt Publishing 编写一本关于 **AI 系统** 的 **GPU Engineering** 书籍，内容涵盖分布式训练、CUDA kernels 和 GPU clusters 等主题。
   - 编辑正在寻找技术审阅者；感兴趣的人员可以私信 [hi@abiaryan.com](mailto:hi@abiaryan.com) 进行引荐。
- **AI Factory Software Engineer 职位开放**：Voltage Park 正在寻求一名 **Software Engineer** 来帮助构建其 **AI Factory** 软件栈；该职位完全远程，在旧金山设有办公室。
   - 职位公告可在 [Voltage Park Careers](https://www.voltagepark.com/careers?ashby_jid=2e463e6a-abc6-48ae-8060-8452c55b2fab) 找到。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1394945973377044520)** (2 条消息): 

> `Colab GPU, Numba, MLE, ML performance engineering` 


- **Colab GPU 简化了 Numba 的使用**：一位用户强调，**Colab GPU notebooks** 允许立即安装 **Numba** 并直接执行 `cuda.jit` 编译的函数。
- **MLE Performance Engineering 入门路径**：一位用户表示有兴趣在明年转型为 **MLE** / **ML performance engineering**，并询问在哪里可以快速获得胜任能力。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1394825102071173153)** (12 条消息🔥): 

> `China H20, H100 vs H20, NVL72 vs NVL144, Ascend GPUs` 


- **中国 H20 引发 GPU 讨论**：成员们讨论了关于 **中国 H20** 的新闻，将其与 **H100** 在互连带宽方面进行了比较，认为其适用于推理。
   - 一位成员指出：“H20 并不算差，因为据我所知它拥有与 H100 相同的互连带宽，这对推理很有好处。”
- **H20 无法与 GB200/GB300 相比**：在训练方面，**H20** 被认为逊色于 **GB200/GB300** 和 **NVL72**。
   - 一位成员表示：“无法将 GB200/GB300 用于训练对比吧？确实无法竞争，NVL72 要好得多。”
- **混淆隐现：NVL144 vs. NVL72**：社区开玩笑说，**NVL144** 和 **NVL72** 配置之间不可避免会产生混淆。
   - 一位成员调侃道：“当我们都搞混 NVL144 和 NVL72 时，那场面一定很棒。老实说，这真的重要吗？因为看起来他们自己的 GPU 大约处于 Hopper 级别。”
- **怀疑 Ascend GPUs 存在 Bug**：对一张图片的分析引发了对 **Ascend GPUs** 在扩展（scale up）时可能存在 Bug 的怀疑。
   - 图片分析指出：“我怀疑 Ascend GPUs 在规模扩展时非常不稳定（buggy）。”


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1394823971265974322)** (8 条消息🔥): 

> `SemiAnalysis Podcast, LLM RL environment framework` 


- **SemiAnalysis Podcast 提到了该论坛**：一位新成员因为 **SemiAnalysis** 的 Dylan 在播客中提到了这个论坛而加入。
   - 一位成员询问了链接并提议私信这位新成员，并分享了一个 [Google Meet 链接](https://meet.google.com/wdk-yipf-zjd?authuser=0&hs=122)。
- **长周期 RL 环境框架发布**：一位成员发布了用于长周期（long-horizon）任务的 **LLM RL 环境框架** 的初始版本，可以在 [X.com](https://x.com/ritserlabs/status/1945494003803148565) 和 [GitHub](https://github.com/ritser-labs/real-work) 上找到。
   - 该框架有助于在 **Docker container** 中设置环境，并提供工具访问和轨迹（trajectories）生成功能。


  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/)** (1 条消息): 

complexfilterr: CVPR 录用论文中有一半的作者来自中国。
  

---

### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1394842154223931402)** (6 messages): 

> `GB300 Availability, Coreweave's GB300 Capacity, Nvidia hardware purchase prioritization, DGX vs HGX, B200 Availability` 


- **讨论 Coreweave 的 GB300 可用性**：成员们讨论了在 **Coreweave** 获取 **GB300** 的途径，但由于**容量有限**，对立即交付的可能性持怀疑态度。
   - 一位成员对立即获取表示惊讶，并指出 Coreweave 最近宣布了 **GB300 NVL72s 容量**，以及来自 Nvidia 可能存在的物流挑战。
- **Nvidia 关系有助于硬件优先级排序**：一位成员提到，*与 Nvidia 的业务关系有助于硬件采购的优先级排序*，并引用了他们目前的硬件采购经验。
   - 他们指出，在选择 **DGX** 还是 **HGX** 产品时，预算只是因素之一，由于特定硬件组件的模块化，选择 **HGX** 方案也有其合理理由。
- **B200 可用性与液冷**：**B200** 目前相对容易购买，但更先进的芯片配置需要**液冷**。
   - 大多数数据中心并未配备液冷设施，而 **B200** 在超大规模云服务商（Hyperscalers）中很受欢迎，因为他们不需要改造数据中心。
- **Voltage Park 提供云端 GPU**：来自 **Voltage Park** 的解决方案工程师表示可以协助为 **AI/HPC/ML 工作负载**获取 GPU。
   - 鼓励感兴趣的人士联系，该成员的 LinkedIn 和公司信息已在个人简介中提供。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1394885843147034766)** (3 messages): 

> `Data Backup, Phone Theft, Learning from Mishaps` 


- **用户找回被盗手机并吸取备份教训**：一位用户报告称其手机被盗后又找回，并感叹在被盗前缺乏近期的备份。
   - 该用户提到，*不知为何，手机被盗后他们反而变得更好了*，因为现在他们会备份手机数据。
- **化祸为福**：用户手机被盗的不幸经历带来了积极的结果。
   - 现在该用户已经执行了数据备份流程。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1394945320663650417)** (3 messages): 

> `CuTeDSL, Jetson series, Jetson Orin, Jetson Thor, CUTLASS Python support for NVIDIA GPUs` 


- **CuTeDSL 关注 Jetson 支持**：一位成员询问 [CuTeDSL](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/) 未来是否会支持 **Jetson 系列**。
   - 另一位成员表示 *CUTLASS Python 将支持所有 NVIDIA GPU*，但目前的重点是 **DC GPU**（数据中心 GPU）。
- **Jetson Orin 架构揭秘**：一位成员指出 **Jetson Orin** 是 **ARM CPU** 与具有 **sm_87** 结构的 **Ampere GPU** 的组合。
   - 他们推测支持 **Jetson Orin** 应该很容易，因为传闻 **CuTeDSL 4.0** 将支持 **ARM CPU**。
- **Jetson Thor 将搭载 Blackwell GPU？**：一位成员分享称 **Jetson Thor** 将配备 **ARM CPU** 和 **Blackwell GPU**，传闻具有 **sm_101** 结构。
   - 他们请求考虑为 **CuTeDSL** 添加支持，并推测这可能不需要花费太多精力。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1394755899750355044)** (88 messages🔥🔥): 

> `Kimi K2, H200, B200, Manus release, Model Ownership` 


- **Kimi K2 热潮席卷社区！**：社区对 **Kimi K2** 感到非常兴奋，有人称其迎来了自己的 *DeepSeek 时刻*，并希望能够下载并在本地部署，以避免为 **Claude 4** 或 **ChatGPT** 付费。
   - 这将带来*无需向 Sam 和 Dario 支付租金的经济自由*，并拥有属于自己的模型。
- **H200 与 H100 成本对比升温！**：**H200** 的价格几乎与 **8xH100** 持平，在 [celiumcompute.ai](https://celiumcompute.ai) 上甚至发现了低至 **$2/GPU/小时** 的 **B200**。
   - 不过，有人澄清该低价是通过 *deepinfra* 提供的限时促销。
- **关于数据集的发布公告**：成员们寻求关于新发布数据集的细节，以了解其生成方式及预期应用。
   - 该数据集与一条 [推文](https://x.com/Teknium1/status/1945259797517099126) 相关，该推文讨论了用于原型推理 CoT、图表以及动作逐步处理的原型 Agentic XML 标签遵循。
- **Manus 的未来揭晓**：据 [Twitter](https://x.com/Teknium1/status/1945259797517099126) 消息，推测 **Manus** 的发布可能会整合 **Kimi K2** 的 Agent 能力，出于地缘政治原因，可能会取代 **Anthropic Claude** 的编码功能。
   - 它被认为是 *Claude 的强力 Frontier 级别替代方案*。
- **模型所有权（Model Ownership）变得可行**：成员们讨论了拥有模型资产的优势，即使需要租用服务器。
   - 共识是，随着 Frontier 级别开源模型的出现，无论用户是在自己的硬件上运行模型还是租用服务器，都能获得所有权和控制权，最终基础模型将趋于商品化。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1395135215269056522)** (1 messages): 

> `Model Context Size, Adding Personality to Models, Letta (MemGPT) Personas` 


- **模型的 Context Size 影响个性注入**：为模型添加个性的效用可能取决于 **模型的 Context Size**，特别是当它非常小时。
   - 建议评估不同的 Context Size 如何与添加的个性相互作用，以观察最佳效果。
- **个性注入并不总是适得其反**：一位成员认为，总的来说，为模型添加个性并不一定会有反作用。
   - 他们以 **Letta**（原 **MemGPT**）等项目使用某种“Personas”为例，说明了如何有效地实现这一点。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1394966614192947201)** (5 messages): 

> `LLM RL environment framework, Atropos compatibility, Unsloth RL guide` 


- **Nvidia 发布 LLM RL 框架**：Nvidia 发布了用于长程任务的 [LLM RL 环境框架](https://research.nvidia.com/labs/adlr/AF3/) 的初始版本，使得在具有工具访问权限的 Docker 容器中设置环境并生成轨迹变得更加容易。
   - 该框架名为 *real-work*，可在 [GitHub](https://github.com/ritser-labs/real-work) 上获取。
- **探索 Atropos 兼容性**：一位成员询问新的 Nvidia LLM RL 环境框架是否可以移植到 **Atropos** 中。
   - 该框架的作者回应称这是一个好主意，他们将研究为 **Atropos** 制作适配器。
- **Unsloth 发布 RL 指南**：Unsloth 发布了关于 **强化学习 (RL)** 的新指南，可在 [docs.unsloth.ai](https://docs.unsloth.ai/basics/reinforcement-learning-rl-guide) 查看。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1394778285300387921)** (28 条消息🔥): 

> `图像生成, 模型搜索仓库 URL, LM Studio 开发路线图, 记忆功能` 


- **LM Studio 避开图像生成**：虽然 **image input**（图像输入）已经在 LM Studio 中可用，但目前没有 **image generation**（图像生成）的计划。
- **用户请求自定义模型搜索仓库 URL**：一位成员询问是否可以为 **Model Search repo** 输入特定的 URL，而不是使用 Hugging Face。
   - 另一位成员回复说，手动下载模型然后导入是唯一的选择，因为*目前没有办法永久切换掉 Hugging Face*。
- **不存在公开路线图**：一位成员询问是否存在公开的 **LM Studio** 开发路线图，但另一位成员确认*不存在公开路线图*。
- **记忆功能可能会加入 LM Studio**：一位成员询问 **memory features**（类似于 ChatGPT 的记忆功能）是否会加入 LM Studio，或者对话是否能引用之前的聊天记录。
   - 一位成员建议使用具有记忆功能的 **MCP**，并希望它能像 **rag-v1** 和 **code-sandbox mcps** 一样被内置。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1394766357442330824)** (39 条消息🔥): 

> `LG 的 EXAONE 许可证, 用于显存扩展的 Thunderbolt eGPU, llama.cpp 中的 AMD NPU 支持, PCI-Express Atomics 支持` 


- **LG 的 EXAONE 许可证引发辩论**：成员们讨论了 **LG EXAONE license** 的限制性，特别是要求在发布的每个模型中添加 *"EXAONE"* 字样，以及对商业或研究用途的限制。
   - 社区质疑如果禁止逆向工程和蒸馏，那么什么才算作*"研究"*，一些人认为该许可证存在矛盾且难以执行。
- **Thunderbolt eGPU 引发显存扩展梦想**：一位成员询问是否可以使用 **PCI-E Thunderbolt 3/4** 卡来扩展 **GPU VRAM**，并附上了一张内存模块的图片。
   - 然而，有人指出目前在 **M-series Macbooks** 上还没有成功的 **eGPU** 实现案例。
- **AMD NPU 推理缺乏 llama.cpp 支持**：社区讨论了 **AMD NPU** 是否适用于 **LM Studio** 中的 AI 推理，但指出 *llama.cpp* 尚未支持 NPU。
   - 目前，*llama.cpp* 唯一支持的 NPU 是 **Ascend NPU**，正如 [build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#cann) 中所记录的那样。
- **PCI-Express Atomics 独立说明**：对于 *muh PCIIRC*，理论上 **ROCm** 支持多 GPU 配置；然而，你的整个 **eGPU** 流水线需要通过 **PCI-Express atomics support** 测试 ([https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/conceptual/pcie-atomics.html](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/conceptual/pcie-atomics.html))。
   - 从传闻报告来看，这项支持可能大部分缺失。


  

---


### **Torchtune ▷ #[announcements](https://discord.com/channels/1216353675241590815/1216353675241590818/1394759808326045799)** (1 条消息): 

> `Torchtune 的未来, Torchtune 项目, Discord 和 GitHub 支持` 


- **Torchtune 的未来公告**：团队在 [这个 GitHub issue](https://github.com/pytorch/torchtune/issues/2883) 中发布了关于 **torchtune** 项目未来的重要公告。
   - 他们向所有帮助 **torchtune** 成长的人表示感谢，并承诺将继续在 **Discord** 和 **GitHub** 上回答问题。
- **Torchtune 团队的下一步计划揭晓**：**torchtune 团队**向社区保证他们不会解散，并承诺更多令人兴奋的工作即将到来。
   - 请保持关注，他们计划很快与社区分享更多信息！


  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1394760855018803300)** (53 messages🔥): 

> `Torchtune future, HuggingFace TRL License, Quantum Computing in Ohio, Checkpointing via NFT` 


- **Torchtune 演进为新产品**：Torchtune 正在演进为*更宏大的东西*，并在*新仓库*中推出*新产品*。
   - 一位成员提到，他们正在*开发一个新产品——在一个新仓库中——以承载 Torchtune 的这次演进*。
- **HF 的 TRL 展示了 Torchtune 宽松的 BSD 3 许可证**：一位成员询问，鉴于新的公告，在将 Torchtune 的组件用于另一个项目时是否存在知识产权方面的担忧。
   - 另一位成员指出，[Hugging Face 已经在使用 BSD 3 许可证做类似的事情了](https://github.com/huggingface/trl/blob/640a9f39164ce3f9bbac7d325c1149ba023314e6/trl/models/activation_offloading.py#L19)，该许可证相当宽松。
- **俄亥俄州食堂发现量子计算机**：一位成员透露他们在 **Cleveland Clinic** 工作，那是*世界上唯一一家拥有量子计算机的医院，哈哈*。 
   - 另一位成员幽默地注意到，量子计算机竟然*在食堂中间……在俄亥俄州？？*
- **通过 NFT 进行训练运行的 Checkpointing**：成员们讨论了分布式 RL、区块链和量子计算等未来技术，这引发了一个关于 Checkpointing 成本的笑话。
   - 有人建议将失败的训练运行在区块链上进行 Checkpoint 并为此支付 gas 费，另一人则幽默地表示，只需为你的 Checkpoint *创建一个 NFT* 即可。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1394766185245048902)** (32 messages🔥): 

> `Kimi K2 Open Source Model, Linux for Home Lab, Azure Speech Services SDK, Qwen-1.5 Inference Technologies, SmolVLM2 Technical Report` 


- **Kimi K2 在本地设备上发布**：社区对于号称世界上最强大的开源模型 **Kimi K2** 现在可以在本地设备上运行感到兴奋，并提供了[链接](https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUFit)。
   - 一些用户告诫不要跨频道发布此公告，建议遵守更好的频道礼仪。
- **Home Lab 的 Linux 发行版困境**：一位成员在寻找适用于 Home Lab 的 Linux 发行版建议，并提到了 Ubuntu 的一些问题。
   - 其他成员建议坚持使用 Ubuntu，并询问了具体遇到的问题。
- **探究 Qwen-1.5 的秘密**：一位用户正在寻求 **Qwen-1.5** 使用的确切结构和推理技术，注意到其概率与他们自己的实现存在差异，讨论指向了一个[相关文件](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py)。
   - 另一位成员建议浮点误差可能是原因，用户回复说平均 1.25e-5 的误差是可以接受的。
- **datasets REST API Bug 搜寻**：一位成员询问在哪里报告 **datasets REST API** 中的 bug，想知道是否有专门的 GitHub 仓库，还是应该直接在主 [datasets repo](https://github.com/huggingface/datasets) 中提交 issue。
   - 另一位成员建议 GitHub 是最好的选择。
- **HF 仓库关注愿望清单**：一位成员询问是否可以只关注 Hugging Face 上的单个 Pull Request 或讨论，而不是关注整个仓库。
   - 用户寻求一种方法来过滤特定感兴趣项目的通知。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1395126740874952756)** (1 messages): 

> `Model Training, 1.5 bit research` 


- **训练决定模型用途**：一位成员认为，模型的训练方式决定了它最终的使用方式。
   - 他们认为，对 **1.5 bit research** 的关注表明模型开发的其他方面存在潜在问题。
- **研究人员关注 1.5 bit**：研究人员正在积极探索 **1.5 bit quantization**，这预示着模型设计其他领域可能存在的问题。
   - 这意味着改进现有模型可能不仅仅需要扩展参数；优化训练过程和架构选择可能至关重要。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 messages): 

tonic_1: https://gpuhammer.com/ 醒醒，宝贝！新的 exploit 刚发布！
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1394879190007808130)** (7 条消息): 

> `LLM Quantization, Desktop App for Plural Identification, French Deep Learning Course, English to Ukrainian Machine Translation Model, LunarisCodex LLM` 


- **ETHOS 量化探索**：一名成员分享了 [ETHOS](https://github.com/wrmedford/ETHOS)，深入研究了 **LLM quantization（量化）技术**，希望其具有参考价值，并链接了一个关于该主题的 [YouTube 视频](https://youtu.be/0pF6GdbwMo4?si=swVldbUTY5Gn4mYB)。
   - 随附的 [PDF](https://cdn.discordapp.com/attachments/897390720388825149/1394879190049882132/ETHOS.pdf?ex=687912ba&is=6877c13a&hm=8c3b1d564877e4310662d28d5a240c65d01f81b2e66098066acc531c259b6cd5&) 深入探讨了 LLM 量化技术。
- **PluralChat 实现跨平台**：一名成员分享了一个名为 [PluralChat](https://github.com/Ktiseos-Nyx/plural_chat) 的桌面应用，专为多重人格（plural）身份认同者设计，利用了 **Python, Gemini 和 Claude**。
   - 该应用设计为 **99% 离线运行**，支持跨平台，可通过 `pipx install aicodeprep-gui` 轻松安装，并包含节省时间的特性，如用于输入常用 Prompt 的按钮。
- **法语深度学习课程加入 AI 功能**：一名成员宣布其 **法语深度学习课程** 项目增加了新功能，包括 **AI 生成的 QCM**（多选题）。
   - 即将推出的功能包括用于生成课程材料的 **AI 支持** 以及用于解答疑惑的 Chatbot，资源可在 [课程网站](https://simonthomine.github.io/CoursDeepLearning/) 和 [GitHub 仓库](https://github.com/SimonThomine/CoursDeepLearning/) 获取。
- **轻量级机器翻译模型上线 HF**：一名成员分享了一个新的 **轻量级模型**，用于 **英乌（English to Ukrainian）机器翻译**，该模型使用最近发布的 **LFM2 模型**，并从 **53.5M** 样本中筛选出 **40M** 样本进行微调。
   - 该模型（**350M 参数**，仅需 **1GB RAM**）在 **FLORES-200** 上达到了 **27.24** 的 **BLEU** 分数，模型已发布在 [Hugging Face](https://huggingface.co/Yehor/kulyk-en-uk) 并提供 [Demo Space](https://huggingface.co/spaces/Yehor/en-uk-translator)。
- **LunarisCodex：巴西 LLM 工具包**：一位来自巴西的成员分享了 **LunarisCodex**，这是一个现代且具教育意义的 **Transformer 风格语言模型** 实现。
   - 这个 **100% 开源** 的工具包受 **LLaMA** 和 **Mistral** 启发，包含 **RoPE, GQA, SwiGLU activation, RMSNorm, KV Caching 和 Gradient Checkpointing** 等特性，代码可在 [GitHub](https://github.com/MeryylleA/lunariscodex) 获取。


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1394943450432143391)** (3 条消息): 

> `Pipeline Parallelism` 


- **Pipeline Parallelism 技术引发关注**：一名成员重点介绍了一份关于 Pipeline Parallelism（流水线并行）技术的资源，并鼓励其他人阅读并在简单的 MLP 上从零开始实现这些技术。
   - 另一名成员提醒大家保持频道主题一致，并将其他讨论转移到相应的频道。
- **频道主题提醒**：一名成员提醒大家保持频道主题一致，并将其他讨论转移到相应的频道。
   - 此提醒是在有人建议探索 Pipeline Parallelism 技术之后提出的。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1395199975167758406)** (1 条消息): 

> `SmolDocLing Finetuning, IDEFICS3ImageProcessor Error` 


- **SmolDocLing 微调出现问题**：一名成员在微调 **SmolDocLing** 时遇到了 `ValueError`。
   - 错误信息为：*Could not find module Idefics3ImageProcessor in `transformers`*。
- **Idefics3ImageProcessor 故障排除**：用户面临在 `transformers` 库中找不到 `Idefics3ImageProcessor` 模块的问题。
   - 这表明环境配置可能存在问题，或者缺少微调 **SmolDocLing** 模型所需的组件。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1395058633750089879)** (1 条消息): 

> `smolagents, Multi-Agent System, AI Agent Message Flow` 


- **Smolagents 架构图分享**：一名成员分享了一张基于 smolagents 的 **Multi-Agent System** 架构图。
   - 该图表直观地展示了系统组件及其交互，有助于理解和实现。
- **AI Agent 消息流向图分享**：一名成员还分享了一张说明 **AI Agent Message Flow**（消息流）的图表。
   - 该图展示了消息在多智能体系统内部是如何路由和处理的，提供了对系统通信基础设施的深入了解。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1394756621590204478)** (19 条消息🔥): 

> `Meta 的开源政策、Behemoth 模型、从开源权重推断闭源模型、真实的 ML` 


- **扎克伯格因 Behemoth 模型被指责背叛**：一些成员指责 **Zuckerberg** 的背叛，原因是 **Behemoth 模型** 可能采取受限发布，并引用了一篇 [推文](https://x.com/signulll/status/1944851904888234293) 作为证据。
- **较小的开源权重模型成为替代方案**：鉴于 **Behemoth 模型** 可能受限发布，一些人认为 Meta 可能会效仿 **Google** 的做法，发布像 **Gemma** 这样较小的开源权重模型。
- **从开源权重推断闭源模型引发研究人员兴趣**：一名成员建议，从开源权重模型推断闭源模型可能是一个很好的研究课题。
   - 他们还表示：*大多数人不会在本地运行这类模型，但这仍然很糟糕。*
- **Discord 频道关注的是真实的 ML**：一名成员澄清了该频道的目的，声明它专注于真实的 ML，而不是加密货币诈骗和妄想症。
   - 随后他们链接了一篇 [论文](https://arxiv.org/abs/2407.18384) 进行讨论。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1394778679938384056)** (5 条消息): 

> `GPUHammer 论文、内存损坏、数据结构` 


- **GPUHammer 论文发布**：成员们分享了 [GPUHammer 论文](https://www.arxiv.org/abs/2507.08166) 的链接。
   - 该论文讨论了 **内存损坏** 以及 **数据结构** 对此类问题的敏感性。
- **开始讨论数据结构**：一名成员询问了关于内存相关项目的工作。
   - 另一名成员提到他们正在研究易受内存损坏影响的 **数据结构和算法**。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1394865148975644693)** (18 条消息🔥): 

> `Muon 优化器、Mixture of Experts 内存优化、Amazon 的 Cursor 竞争对手` 


- **Muon 优化器亮相**：一种名为 **Muon** 的新优化器被用于一个模型中，详见这篇 [论文](https://arxiv.org/abs/2507.08166) 和这段 [视频](https://www.youtube.com/watch?v=4bFDPVe6BHs)，该模型旨在工具使用方面达到与 **Claude 4** 相当的训练水平。
- **MoEs 优化内存带宽**：通过使用 **Mixture of Experts** (MoEs) 架构，大型实验室针对内存带宽进行优化，这意味着内存带宽是关键问题，因为该架构能最大限度地减少内存移动。
   - 一名成员表示，由于 **MoEs** 的资源利用效率高，可能比稠密模型（dense models）使用更少的 GPU 进行训练，详见这段 [视频](https://youtu.be/JOqLp1adGO4?si=hUAnREYY5CQoeoaQ)。
- **Amazon 以秘密武器对抗 Cursor**：Amazon 发布了 **Cursor** 的竞争对手，如该 [链接](https://www.pomerium.com/blog/when-ai-has-root-lessons-from-the-supabase-mcp-data-leak) 所述，文中指出 Cursor 在泄露数据方面“表现出色”。
   - 讨论引用了 [Supabase MCP 数据泄露事件](https://www.pomerium.com/blog/when-ai-has-root-lessons-from-the-supabase-mcp-data-leak)，作为 AI 数据安全的警示。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1394828914357960705)** (26 条消息🔥): 

> `Vertex AI Thinking 输出，Aider 终端推荐，Claude Code 的本地模型替代方案，Kimi K2 配合 Groq，Aider 基准测试更新` 


- ****Vertex AI Thinking Tokens 终于显示****：一位用户发现，在 Aider 中使用 `openrouter/google/gemini-2.5-pro` 模型运行 `/help` 可以[启用 Thinking Tokens 的显示](https://cdn.discordapp.com/attachments/1131200896827654149/1394828914072883380/image.png?ex=68798ca7&is=68783b27&hm=014b973a4b9583e6ebc25aced1ccb74c6018)，他们认为这有助于监控请求进度。
   - 他们后来发现，使用 `/think- 32k` 命令也可以启用 Thinking 摘要的显示，尽管这会增加回滚空间，但他们仍然很喜欢这个功能。
- ****Ghostty 和 Kitty 获得终端应用推荐****：用户讨论了 Aider 的终端推荐，建议使用 **Ghostty** 和 **Kitty** 以获得更好的性能，尽管有些人认为 **GNOME terminal** 已经足够了。
   - 一位遇到 Aider 屏幕刷新问题的用户被建议尝试 Ghostty，而另一位用户虽然 **Alacritty** 在图像显示协议方面存在困难，但仍推荐了它。
- ****Kimi K2 配合 Groq 脱颖而出****：一位用户报告称，**Kimi K2 配合 Groq** 表现出*惊人*的性能，达到 **200-300 tokens/sec**，且输出质量极高，足以与 **GPT-4o** 媲美并超越 **Sonnet 3.7**。
   - 他们强调了其性价比和速度，使其成为首选，这也呼应了社区中其他人的积极反馈。
- ****自动 Thinking vs. 显式 Thinking****：一位用户发现 Aider 的 **auto thinking** 功能不显示思考过程，这促使他们探索用于显示 Thinking 摘要的显式命令。
   - 使用 `/think- 32k` 命令可以启用 Thinking 摘要的显示，而 auto thinking 则在后台运行而不显示过程。
- ****是时候更新 Aider 基准测试了？****：用户建议 Aider 应该更新其基准测试，因为许多模型现在的得分都超过了 **80%**。
   - 该提议包括创建一个私有基准测试，用户可以在其中贡献自己的测试用例。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1394817078279536801)** (11 条消息🔥): 

> `Aider 调试，OpenRouter 模型，Gemini Flash，Architect 模式，Thinking 模式` 


- **Aider 用户讨论调试**：一位用户询问如何在开发模式下调试 **Aider**，但在给定上下文中未提供具体解决方案。
   - 该用户正在寻找更具交互性的调试方式，但似乎没有其他用户提供方法。
- **OpenRouter 模型移除**：一位用户注意到 **OpenRouter** 移除了 `google/gemini-flash-2.5-preview:thinking` 模型，并寻求在 `openrouter/google/gemini-flash-2.5` 中启用 *thinking* 模式的方法。
   - 他们发现 `/think-tokens 2000` 似乎有效，但不确定这是否是正确的数值。
- **Aider 不支持 DeepSeek-r1-0528**：一位用户询问是否可以修改现有的 r1 以使用 **0528** 更新，指的是 [DeepSeek-r1-0528:free 模型](https://openrouter.ai/deepseek/deepseek-r1-0528:free)。
   - 然而，据称 *Aider 不支持此模型*。
- **Gemini 的 Thinking Tokens**：一位用户引用了一条关于为 **Gemini** 配置 **32k thinking tokens** 的推文（[Paul Gauthier 的推文](https://x.com/paulgauthier/status/1932068596907495579)），同时尝试通过 Vertex 使用 **Gemini 2.5 Pro**。
   - 该用户确认 `/think-tokens` 命令在 Vertex 上启用了流式 Thinking 摘要。
- **探索 Architect 模式用法**：一位用户询问其他人如何使用 **/architect 模式**，并指出它看起来像是一个带有两个模型的强大代码模式，而不是人机回环的设计讨论。
   - 该用户将其与生成 **.md** 文件（包含建议方案供审查，然后再用 **/code 模式**实现）的常见建议进行了对比，并想知道 **/ask 模式**是否更适合设计讨论。


  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1395061788336459806)** (2 messages): 

> `Switchpoint Router, OpenRouter AI, aider polyglot benchmark` 


- **SwitchPoint Router 引发关注**：一名成员询问了 [OpenRouter.ai 上的 SwitchPoint Router](https://openrouter.ai/switchpoint/router)，该路由能以潜在更低的费率将请求路由到 **GPT-4.1**、**Claude 4 Sonnet** 和 **Gemini 2.5 Pro** 等模型。
   - 该路由器的网站声称在 *aider polyglot benchmark* 中获得了 **80%** 的成绩，引发了进一步讨论。
- **用户对 Switchpoint 表现出兴趣**：一位用户对该路由器表示了兴趣，并表示他们以前没有使用过它。
   - 他们指出这非常*有趣*。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1395036046156370001)** (2 messages): 

> `Google Docs Tab Feature, uBlock browser extension, Copying news articles into Google Docs` 


- ****Docs 标签页技巧**激发新灵感**：一位成员发现使用 **Google Docs 的标签页功能**是一个*很酷的主意*，表示他们以前从未考虑过以这种方式使用它。
- ****uBlock 扩展程序**保驾护航**：一位成员建议在将新闻文章复制到 Google Docs 时，使用 **uBlock 浏览器扩展程序**来移除广告和其他不需要的元素。
   - 他们指出，可以在扩展程序设置的 *Filter list* 选项卡中添加针对干扰项和社交媒体弹窗的额外过滤器。
- ****Notepad.exe** 清理剪贴板**：一位成员建议将文本复制到 **notepad.exe** 中，作为避免将广告和其他不需要的内容粘贴到 Google Docs 的一种方法。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1394761208963272867)** (23 messages🔥): 

> `PC version of NotebookLM, Featured Notebooks Removal, Video Overviews Release, Custom Podcast Intros, Public Notebooks Location` 


- ****NotebookLM** 桌面版不可用？**：一位用户询问了该应用程序的 **PC 版本**，并表示不熟悉将 **Google Drive** 作为来源的使用方法。
   - 另一位用户建议利用带有标签页的 **Google Docs** 在单个笔记本中管理多个来源。
- **用户希望隐藏精选笔记本 (Featured Notebooks)**：用户对被迫查看 **"Featured Notebooks"** 且没有移除选项表示沮丧。
   - 他们表达了对更具定制感和组织性的需求。
- **用户寻求自定义播客片头的方法**：一位用户询问如何创建一个不以 "Welcome to the Deep Dive" 开头的**播客片头**。
   - 另一位用户指出，标准片头是官方标语，用户是 "The Deep Dive Podcast" 的内容贡献者。
- **播客长度由语言设置决定**：一位用户报告说，即使有大量的源材料，他们的播客也始终很短，大约 **7-10 分钟**。
   - 另一位用户指出，选择“长”播客输出的选项仅适用于 **English**，从而解决了该问题。
- **"Service Unavailable" 错误困扰用户**：一些用户遇到了 **"Service unavailable"** 错误消息，但缺乏足够的上下文。
   - 该错误表明用户正尝试访问其账户不可用的服务。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1394799068621963335)** (20 messages🔥): 

> `Manus AI mobile app creation, Vehicle creation for OMSI 2, AI outperforming Manus` 


- **成就达成：精通 Manus AI 移动应用创建！**：一位成员声称已经精通使用 **Manus AI**，仅需 **100 credits** 即可创建任何主题且具有定制设计的移动应用程序。
   - 他们提出可以帮助那些需要通过 **Manus** 构建东西但遇到问题的其他人。
- **免费的 Manus 替代方案：为 OMSI 2 创建车辆？**：一位成员声称拥有一个 **Manus** 的替代方案，具有所有相同的功能，但 **100% 免费**且无限制，并使用它为视频游戏 **OMSI 2**（一款巴士模拟器）创建了一辆车。
   - 他们推测，根据模型的不同，它可能能够为 **Google Collab** 创建一个脚本来生成该文件。
- **下一代 AI 声称性能超越 Manus**：一位成员声称构建了一个在基准测试中**性能超越 Manus 的 AI**，并向首批 **100 人提供完全、无限制的访问权限**作为终身测试人员。
   - 他们引导其他人私信（DM）他们以获取名额，并*体验零限制的下一代 AI*。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1395111051498098848)** (2 条消息): 

> `Discord Channels, Community Showcase` 


- **提议扩展 Discord 频道**：一名成员建议创建一个新的 Discord 频道，供用户分享他们的项目和成就，类似于其他开源 Discord 服务器中的频道。
   - 该建议旨在提供一个分享有趣图片和仓库的空间，这与该频道现有的规则有所不同。
- **以 Community Showcase 作为替代方案**：一名工作人员建议用户在 [Modular 论坛](https://forum.modular.com/c/community-showcase/8) 的 **Community Showcase** 类别中分享他们的工作。
   - 该类别旨在展示 Modular 社区内的用户项目和贡献。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1394818972913762445)** (15 条消息🔥): 

> `Mojo native requests library, TLS support in Mojo, Escaping keyword usage in Mojo, @parameter functions and runtime closures` 


- **Mojo 的 requests 库需要 TLS**：一名成员询问了关于原生 **Mojo** *requests* 库的情况，另一名成员指出主要的障碍是 **TLS support**。
   - 一名成员分享了他们的 [类 requests 玩具库](https://github.com/thatstoasty/floki)，其中包含了 **TLS bindings**。
- **解析 Escaping：不只是为了越狱**：一名成员寻求关于 **Mojo** 中 *escaping* 关键字用法的澄清，并指出缺乏具体的文档。
   - 另一名成员指向了 [更新日志 (Changelog)](https://docs.modular.com/mojo/changelog#v070-2024-01-25)，澄清了 *escaping* 执行的是值的 `__copyinit__` 而不是通过引用捕获。
- **Parameter 装饰器被捕获了！**：一名成员询问了关于 `@parameter` 函数（捕获）的问题，另一名成员提供了一个专门针对该内容的 [手册链接](https://docs.modular.com/mojo/manual/decorators/parameter/#parametric-closure)。
   - 他们还提到了 **Q3 路线图** 中关于 *统一 @parameter 和运行时闭包* 的令人兴奋的消息 ([roadmap-update](https://forum.modular.com/t/mojo-q3-roadmap-update/1957))。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1395085956792717472)** (12 条消息🔥): 

> `setitem PR, tensor.py, assign parameter, kernel fusion, remove realize()` 


- **Setitem PR 获得审查**：一名成员请求对其 [setitem PR](https://github.com/tinygrad/tinygrad/pull/11260) 进行审查，特别是询问其解决方案的开销以及是否方向正确。
   - 审查者建议 `tensor.py` 的更改应该只是移除 `realize` 调用，并在更底层进行修复。
- **优化讨论：Assign 的参数**：一名成员询问正确的做法是否是为 assign 添加一个参数，允许用户指定范围和索引。
   - 审查者回应称这 *不值得*，且 *不是该悬赏 (bounty) 想要的结果*。
- **针对 realize() 移除的修复建议**：当仅移除 `realize()` 调用时，赋值不会持久化回 `self`，因此一名成员建议将该行更改为 `self.uop = res.assign(v).uop` 或类似形式。
   - 他们建议了其他替代方案，如 `self.assign(v, indices)` 或 `self.uop = reconcileAssignment(self.uop, res.assign(v).uop)`。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1394882175429644298)** (1 条消息): 

> `tensor lvl hooks, LLM hidden states, Fetching hidden states` 


- **Tinygrad Tensor Hook：社区寻求见解**：一名用户询问 **tinygrad** 是否支持 tensor 级别的 hook，旨在获取大语言模型 (**LLM**) 中的隐藏状态 (**hidden states**)。
   - 该用户正在探索在模型执行期间提取和利用隐藏状态的方法。
- **探索 LLM 隐藏状态提取**：该咨询重点在于使用 **tinygrad** 从 **LLM** 中检索 **hidden states**。
   - 目标是接入模型内部的中间表示，以便进行进一步的分析或处理。


  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1394757033013547121)** (4 messages): 

> `阿姆斯特丹聚会、UiPath 集成、生产级 RAG、ODSC Agentic AI 峰会` 


- **LlamaIndex 与 Snowflake 联手举办阿姆斯特丹聚会**：LlamaIndex 将于 7 月 31 日与 **Snowflake** 在阿姆斯特丹合作举办聚会，重点讨论在生产环境中构建高质量的数据 Agent ([链接](https://t.co/fFJvvIWrw4))。
- **UiPath 新增对 LlamaIndex Agents 的支持**：通过 **UiPath** 全新的代码化 Agent 支持，LlamaIndex Agent 现在可以无缝部署到企业环境中 ([链接](https://t.co/ILez3d6Zrs))。
   - 特性包括通过 **UiPath Python SDK** 实现的*全代码级控制*，以及构建能够从企业系统中提取数据的自定义 Agent 的能力。
- **开源工程师分享 RAG 技巧**：一位开源工程师分享了构建生产级 RAG 系统中经过实战检验的经验，并对 **文本提取策略** 提出了建议 ([链接](https://t.co/R0TTgWrKtv))。
   - 讨论涵盖了何时使用 *简单解析* 与 *基于 OCR 的高级解决方案*（如 **LlamaParse**）。
- **LlamaIndex 在 ODSC 的 Agentic AI 峰会上进行演讲**：一位 LlamaIndex 成员将在 **ODSC Agentic AI 峰会** 上主持一场动手实践研讨会，教与会者使用 LlamaIndex 构建 Agent ([链接](https://t.co/6jcYIGR70s))。
   - 参与者将学习如何 *创建自主应用*，这些应用能够利用目标和工具独立完成任务。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1395046546776526909)** (5 messages): 

> `LLM 微调指南、使用 LlamaIndex 的 Multi-agent 工作流、AI 工程师机会` 


- **LLM 微调指南 MVP 已上线！**：一位工程师分享了 **LLM 微调指南** 的 MVP 版本，旨在提供关于数据准备、参数选择以及模型评估的实用分步建议，并正在寻求 [开发者反馈](https://ai-finetuning-advisor-g3r5.vercel.app/)。
- **LlamaIndex Multi-Agent 工作流引发咨询**：一位成员紧急请求关于 **使用 LlamaIndex 的 Multi-agent 工作流** 的协助。
- **AI 工程师寻求构建智能系统**：一位工程师正寻求与初创公司、研究团队或在 AI、Web3 或自动化领域的创新者合作，他擅长构建由 GPT-4o、LangChain、AutoGen、CrewAI 及其他前沿工具驱动的 **自主 Agent**。


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1394966425206259812)** (5 messages): 

> `用于层级标签的 IReRa、用于层级的多模块、用于父子识别的原生 DSPy` 


- **IReRa 在层级标签中的应用受到质疑**：一位成员询问 **IReRa** 是否适用于具有 **3 个层级**、**440 个父级** 和 **3500 个孙级** 的 **层级标签** 场景。
   - 另一位成员建议，层级结构适合使用多个模块或步骤，但如果每个步骤只有几十个标签且使用了大型 LLM，则不需要 **IReRa**。
- **建议使用多模块处理层级**：一位成员建议使用 **多个模块**（或步骤）来有效处理层级标签。
   - 他们指出，如果每个步骤仅涉及几十个标签并利用了大型语言模型 (**LLM**)，则可能不需要 **IReRa**。
- **提议使用原生 DSPy 进行父子识别**：一位成员询问是否可以使用 **原生 DSPy** 先识别父级，然后从父级到子级再到孙级逐步进行。
   - 另一位成员确认他们使用了类似的方法，总共只有 **3 个父级** 和 **28 个子级**，效果很好。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1395009983644696697)** (3 messages): 

> `aws/nova-prompt-optimizer, Lean 4` 


- **Nova Prompt Optimizer 需要 DSPy**：一位成员检查并确认 [aws/nova-prompt-optimizer](https://github.com/aws/nova-prompt-optimizer) 依赖于 `dspy`。
   - 希望有人能研究这两者的协同工作。
- **Lean 4 验证准备就绪**：一位成员推荐使用 **Lean 4** 来验证某些内容。
   - 他们链接了一个与 **Lean 4** 相关的 [YouTube 视频](https://www.youtube.com/watch?v=1067jj67toY)。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1394786109615247510)** (8 条消息🔥): 

> `证书声明表单, Lab 提交反馈` 


- **学生因错过证书声明表单而受阻**：由于人力有限，课程组织者无法为错过 **证书声明表单 (certificate declaration form)** 截止日期的学生提供特殊处理。
   - 一名学生请求重新开放 [声明表单链接](https://forms.gle/iPA2MUpHdtrBE1vu5)，但其请求最终被拒绝。
- **学生寻求 Lab 提交反馈**：一名学生询问如何获得关于其 **Lab 提交表现** 的反馈，并探讨额外的研究方向。
   - 该学生对其提交的内容表示满意，但希望能与相关人员进一步讨论其表现。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1394779720435896370)** (5 条消息): 

> `Anthropic 连接器目录, Docker MCP 工具包, MCP Inspector` 


- **Anthropic 的连接器目录向普通用户开放 MCP**：Anthropic 发布了全新的 ["连接器" 目录 (connectors directory)](https://claude.ai/directory)，将 MCP 世界推向了更广泛的社区。
   - 有观点认为，随着更广泛的社区开始接触它，对这些连接器的需求将会激增。
- **连接器目录并非 Docker MCP 工具包的竞争对手**：一名成员认为连接器目录是在尝试与 Docker 的 MCP Toolkit 竞争。
   - 另一名成员指出这并非竞争关系，因为 **Docker 工具包是供开发者使用的**，而 **该目录是为普通用户设计的**：例如日常工作的项目经理或市场人员。
- **MCP Inspector 未重新加载资源**：一名成员正在使用 mcp typescript-sdk 编写服务器，并在更新 profile 后从 tool 内部调用 `server.sendResourceListChanged();`。
   - 他们发现使用 MCP inspector 时，如果使用了该 tool 然后刷新资源，除非清除资源列表并重新列出，否则资源似乎不会更新。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1395039734228582492)** (2 条消息): 

> `AI agents, Model Context Protocol, 自主编排, 并行执行, Anthropic Claude Sonnet-4` 


- **Glasses 👓 赋予 AI 视觉能力！**：一名成员分享了一个名为 **Glasses** 👓 的新 **开源工具**，它实现了 **Model Context Protocol (MCP)**，允许任何兼容的 AI 请求 URL 截图，甚至可以模拟不同的屏幕尺寸，代码托管在 [GitHub](https://github.com/gourraguis/glasses-mcp)。
- **Goose 增加子 Agent 支持**：**Goose** 现在支持子 Agent (subagents)，增强了灵活性并支持多 Agent 系统，展示案例中使用了 [Codex CLI 作为子 Agent](https://block.github.io/goose/docs/experimental/subagents)。
- **多模型 Goose 支持 Claude 等模型**：**Goose** 展示了多模型支持，使用 **Anthropic Claude Sonnet-4** 作为主 LLM，同时也支持 **OpenAI, Gemini, Ollama** 等。
- **Goose 实现自主编排**：**Goose** 通过协调主任务和子 Agent，然后合并结果，从而实现自主编排，简化了复杂的工作流。
- **Goose 助力并行执行**：**Goose** 支持任务的并行执行，同时也支持需要子 Agent 移交的顺序执行场景，如[附图](https://cdn.discordapp.com/attachments/1315696461316358175/1395044247576776714/1752674728846.png?ex=687903b3&is=6877b233&hm=7fce7947df059f51cdf73bb6d012d910c06018fbb584d110a8e9c50596b404b4&)所示。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1394785062985400330)** (4 条消息): 

> `Cloudflare R2, GPT4ALL 逻辑, AI 与 Web3` 


- **Cloudflare R2 访问被拒绝**：一名成员在尝试从 **Cloudflare R2** 下载用于微调模型的数据集时遇到 **Access Denied** 错误，使用的命令为 `aws s3 ls --endpoint-url=https://9fa58365a1a3d032127970d0bd9a1290.r2.cloudflarestorage.com/ s3://contrastive`。
- **GPT4ALL 的逻辑受到关注**：一名成员询问 **GPT4ALL** 是否基于用户优先的方法来处理输入（获取原始逻辑、推理）和输出（传输逻辑）。
   - 他们质疑 **中立性指南和安全过滤器** 是否仍然有效。
- **AI 工程师寻求 Web3 项目合作**：一位专注于 AI 和 Web3、在构建智能自主系统方面拥有实战经验的软件工程师正在寻求新机会。
   - 该工程师的技能包括 **Python**、**TypeScript (React/Next.JS)**、**C/C++**、**LangChain**、**ReAct**、**OpenAI** 以及 **Solidity/Rust**，并愿意与初创公司、研究团队或 AI、Web3、自动化领域的创新者合作。


  

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1394975518587228170)** (1 messages): 

> `DeepSeek in Production Event, MoE, MLA, FP8, MTP` 


- **DeepSeek Production 活动即将举行**：Packt 正在组织一场 **DeepSeek in Production** 活动，由顶尖工程师和研究人员组成的小组将讨论是什么让 **DeepSeek** 比其他模型**更快、更聪明、更高效**。
   - 该活动还包括动手实践研讨会，参与者可以使用 **LoRA + Unsloth** 微调 **DeepSeek models**，甚至可以在普通的消费级 GPU 上进行；更多详情请见 [Eventbrite 链接](https://www.eventbrite.com/e/deepseek-in-production-tickets-1436251630289?aff=oddtdtcreator)。
- **DeepSeek 模型优势详解**：活动将涵盖 **MoE, MLA, FP8 和 MTP** 等技术，以解释 **DeepSeek** 的性能优势。
   - 作为开源的坚定支持者，一位参与者强调该活动看起来很有前景。


  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1394941539330953347)** (1 messages): 

> `Financial Keyword Extraction with BERT, BERT for Key Sentence Extraction, Cosine Similarity for Keyword Identification, Improving BERT-based Keyword Extraction` 


- **基于 BERT 的金融关键词提取器遇到困难**：一位用户正在构建一个**基于 BERT 的金融关键词/关键句子提取器**，旨在从财务摘要中识别公司信息，如**注册地址**、**省份**、**注册日期**和**经营期限**。
   - 他们最初的方法是使用句子嵌入（sentence embeddings）与特定任务嵌入（例如“此句子是注册地址”）之间的**余弦相似度（cosine similarity）**，但*效果并不理想*。
- **寻求改进 BERT 提取器的建议**：由于最初的余弦相似度方法不成功，该用户正在寻求如何改进其**基于 BERT 的金融关键词提取**模型的建议。
   - 他们正在探索替代方法，以准确地从公司财务摘要中识别**注册地址**和**注册日期**等关键信息。


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1395160989007216650)** (1 messages): 

> `Claude Sonnet 4, Anthropic, Discounted Credit Rate` 


- **Claude Sonnet 4 回归并获得 Anthropic 支持**：**Claude Sonnet 4** 已回归，并获得了来自 **Anthropic** 的官方支持。
   - [在此查看公告](https://x.com/windsurf_ai/status/1945599013954490523)。
- **Pro/Teams 用户享受折扣费率**：**Claude Sonnet 4** 和 **Claude Sonnet 4** (Thinking) 以优惠的 **2x** 积分费率向 **Pro/Teams** 用户开放。
   - 这在有限的时间内适用于 **Editor** 和 **JetBrains Plugins**。


  

---


### **Codeium (Windsurf) ▷ #[content](https://discord.com/channels/1027685395649015980/1092566563862884412/1394829404504461363)** (1 messages): 

> `Wave 11 inclusion` 


- **Wave 11 纳入投票**：[X](https://x.com/windsurf_ai/status/1945263147994243294) 上的一项投票询问社区是否应在 **Wave 11** 中包含某个特定主题。
   - 成员可以与帖子互动以表达他们的意见。
- **另一个主题占位符**：这是一个用于满足最少项目要求的占位符。
   - 如果有更多详情，将在此处添加。


  

---


---