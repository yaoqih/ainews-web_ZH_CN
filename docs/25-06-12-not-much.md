---
companies:
- bytedance
- morph-labs
- huggingface
- deeplearning.ai
- figure-ai
- langchain
- sakana-ai
date: '2025-06-12T05:44:39.731046Z'
description: '**字节跳动（Bytedance）**展示了一款名为 **Seedance 1.0** 的顶级视频生成模型，其表现令人印象深刻，但目前尚未公开发布。与此同时，**Morph
  Labs** 宣布推出 **Trinity**，这是一个针对 Lean 语言的自动形式化系统。**Huggingface Transformers** 宣布弃用对
  Tensorflow/JAX 的支持。


  **DeepLearning.AI** 的**吴恩达（Andrew Ng）**强调了**生成式 AI 应用工程师（GenAI Application Engineer）**这一角色的兴起，并重点指出了掌握
  **AI 构建模块**以及 **Codex**、**Claude Code** 等 **AI 辅助编程工具**技能的重要性。工程团队正越来越多地针对大语言模型（LLM）测试
  API 设计的易用性。


  **Figure AI** 的首席执行官强调，速度是核心竞争优势。**LangChain** 则为 AI 智能体（Agents）引入了**上下文工程（Context
  Engineering）**的概念。在大语言模型上应用强化学习展现出了变革性的潜力，社区也日益看重 **AI 评估（evals）**和数据工作。


  **Sakana AI** 发布了 **Text-to-LoRA**，这是一种利用自然语言生成特定任务 LoRA 适配器的超网络（hypernetwork）方法，能够实现高效的模型定制。视频生成领域的竞争愈演愈烈，字节跳动基于
  Seed 的模型因其高质量而备受赞誉，正向美国的研究实验室发起挑战，此外还有**可灵（Kling）2.1** 和 **Veo 3** 等模型也备受关注。'
id: MjAyNS0w
models:
- seedance-1.0
- codex
- claude-code
- kling-2.1
- veo-3
people:
- andrew_ng
- hwchase17
- adcock_brett
- clementdelangue
- akhaliq
- jxmnop
- hamelhusain
- sh_reya
title: 今天没发生什么特别的事。
topics:
- video-generation
- autoformalization
- ai-assisted-coding
- api-design
- context-engineering
- reinforcement-learning
- ai-evals
- hypernetworks
- model-fine-tuning
- foundation-models
---

平静的一天

> 2025年6月11日至6月12日的 AI 新闻。我们为您检查了 9 个 subreddit、449 个 Twitter 账号和 29 个 Discord 社区（218 个频道，7130 条消息）。预计节省阅读时间（按 200wpm 计算）：579 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 方式呈现所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上给我们反馈！

字节跳动展示了（但未发布）一款令人印象深刻的 [**SOTA 视频生成模型 Seedance 1.0**](https://seed.bytedance.com/en/seedance)，Morph Labs 发布了 [Trinity](https://x.com/morph_labs/status/1933181394588483868?s=46)（一个用于 Lean 的自动形式化系统），而 Huggingface Transformers [**弃用了 Tensorflow/JAX**](https://x.com/LysandreJik/status/1933201171130593530)。

---

# AI Twitter 综述

**AI 工程技能、角色与开发哲学**

- **GenAI 应用工程师的崛起**：在一段详细的推文中，[**DeepLearning.AI**](http://deeplearning.ai/) 的 **Andrew Ng** 概述了新兴角色 **GenAI Application Engineer** 的关键技能。他强调了两个标准：使用新的 **AI 构建块**（如 RAG、agentic frameworks、evals、MCPs）构建强大应用的能力，以及使用 **AI 辅助编程工具**（如 **Codex** 和 **Claude Code**）进行快速工程的能力。[Ng 指出，虽然 AI 构建块的生命周期较长，但 AI 辅助编程技术的过时速度要快得多](https://twitter.com/AndrewYNg/status/1933185193059516442)，这使得持续学习的能力成为预测成功的关键指标。
- **为 LLM 设计 API**：[@alexalbert__/](https://twitter.com/alexalbert__/status/1933177502777913596) 观察到的一个日益增长的趋势是，大公司的工程团队现在在发布前会**针对 LLM 测试其 API 设计**。他们运行评估以查看哪些 API 结构最容易被模型使用，这预示着未来软件设计将以模型为主要用户。
- **开发中对速度的需求**：**Figure AI** 的 CEO [@adcock_brett](https://twitter.com/adcock_brett/status/1933226344156221746) 认为，**速度是科技领域最终的优势**和护城河。他反思道，通过将速度置于完美之上，他的团队在过去 3 年中取得了 **5-7 年的进展**，这建立了动力和专注力。
- **“上下文工程”的重要性**：**LangChain** 的 [@hwchase17](https://twitter.com/hwchase17/status/1933278290992845201) 强调了 **“Context Engineering”** 的概念，将其视为 prompt engineering 的下一个阶段。他将其定义为动态且自动地为系统提供必要上下文的过程，并称之为“**构建 AI agents 的工程师的首要工作**”。
- **RL 的变革潜力**：[@jxmnop](https://twitter.com/jxmnop/status/1933359925415325980) 指出，当 **LLM 上的强化学习 (RL) 发挥作用**时，令人难以置信的可能性正变得清晰，并暗示“我们才刚刚开始”。
- **Evals 和数据工作的价值**：虽然承认 [**eval 工作和盯着数据看**“非常重要且极其无聊”](https://twitter.com/finbarrtimbers/status/1933278968859468161)，但社区强调了它们的必要性。[@HamelHusain](https://twitter.com/HamelHusain/status/1932964208100180239) 和 [@sh_reya](https://twitter.com/HamelHusain/status/1932964208100180239) 推出的一门关于 **AI Evals** 的热门课程被频繁提及，被认为是工程师和 PM 掌握这一关键技能的核心资源。

**模型与研究突破**

- **来自 Sakana AI 的 Text-to-LoRA**：**Sakana AI** 推出了 **Text-to-LoRA (T2L)**，这是一种利用 **hypernetwork** 直接从任务的自然语言描述中生成特定任务 **LoRA adapters** 的新颖技术。该方法从数百个现有的 LoRA 中进行元学习 (meta-learns)，从而实现对基础模型 (foundation models) 的快速、参数高效的定制，而无需大型数据集或昂贵的微调。[公告指出 T2L 可以泛化到未见过的任务，并降低了非技术用户使模型专业化的门槛](https://twitter.com/SakanaAILabs/status/1932972420522230214)。该发布受到了热烈欢迎，**Hugging Face** 的 [@ClementDelangue](https://twitter.com/ClementDelangue/status/1932977773582106973) 简短地惊叹道：“Text to models!”
- **视频生成竞赛**：一款基于 **Seed** 架构的 **ByteDance** 模型因其质量而受到赞誉，[@scaling01](https://twitter.com/scaling01/status/1933048431775527006) 声称它“**完胜 Veo 3**”，并质疑美国实验室是否具有竞争力。此前，**快手可灵 (Kling AI)** 分享了其 **Kling 2.1** 模型的生成效果，[@_akhaliq](https://twitter.com/_akhaliq/status/1933069477807337771) 展示了一段 **Veo 3** 视频，内容是一只北极熊在解释协和式飞机的失败。与此同时，**ByteDance** 还推出了 **APT2**，这是一种用于实时交互式视频生成的自回归对抗后训练 (autoregressive adversarial post-training) 方法。
- **从预训练模型中激发潜在能力**：由 [@jeremyphoward](https://twitter.com/jeremyphoward/status/1932959121915195842) 分享的来自 **Anthropic** 的最新研究展示了如何**在不使用外部监督的情况下从预训练模型中激发能力**。由此产生的模型在数学和编程等任务上通常与 SFT 模型相当，甚至优于 SFT 模型。[@jiaxinwen22](https://twitter.com/jeremyphoward/status/1933364618371739948) 澄清说，这是关于能力激发 (elicitation)，而非自我提升 (self-improvement)。
- **Meta 的 V-JEPA 2 世界模型**：[@omarsar0](https://twitter.com/omarsar0/status/1932993784683303272) 分享了 **Meta** 发布的 **V-JEPA 2**，这是一个旨在通过从视频中学习以理解和预测物理世界，从而加速物理 AI (physical AI) 发展的新世界模型。
- **预训练中的模型合并**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1933255559668772941) 认为，**预训练期间的模型合并 (model merging)** 是目前高算力环境下基础模型训练中最缺乏讨论和研究的方面之一。
- **每周模型汇总**：[@mervenoyann](https://twitter.com/mervenoyann/status/1933101803274477600) 提供了每周开源模型发布的摘要，包括 **阿里巴巴的 Qwen3-Reranker-4B** 和 **Qwen3-Embedding** 模型、**OpenBMB 的 MiniCPM4** 系列、**Arcee AI 的 Homunculus 12B**、用于文档解析的 **MonkeyOCR**、**NVIDIA 的 Llama-3.1-Nemotron-Nano-VL-8B-V1** 以及 **ByteDance 的 ContentV-8B** 视频模型。
- **想象力读心基准测试**：**明尼苏达大学 (UMN)** 的研究人员创建了[第一个通过 fMRI 直接从想象中解码心理图像的基准数据集](https://twitter.com/iScienceLuvr/status/1932945933521817988)，这超越了仅仅重建人眼当前所见内容的范畴。

**工具、框架与集成**

- **Hugging Face Transformers 弃用 TensorFlow 和 Flax**：在一次重大的生态系统转变中，**Hugging Face** 宣布其流行的 `transformers` 库将变为 **PyTorch-only**，[**弃用对 TensorFlow 和 Flax 的支持**](https://twitter.com/_lewtun/status/1933226225620885818)。该团队指出，高昂的维护负担和减少库体积膨胀的愿望是此次变更的关键原因。
- **LangGraph 助力企业级 AI Agents**：**LangChain** 展示了**管理 11 万亿美元资产的贝莱德 (BlackRock)** 如何在 **LangGraph** 上构建其 **Aladdin Copilot** 编排系统，[为 100 多个应用程序中的 4,000 多名工程师提供支持](https://twitter.com/LangChainAI/status/1933216936730722794)。他们还宣布了 **LangGraph 与文档摄取引擎 Tensorlake 的新集成**，以提升 Agent 对数据的理解能力。
- **Perplexity 与 Fireflies 集成用于会议**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1933248190326976542) 宣布 **Perplexity** 现在可以通过与 [**Fireflies.ai**](http://fireflies.ai/) 的集成在视频通话中使用，将其搜索和推理能力带入会议中。
- **Runway 推出 "Chat Mode"**：**Runway** 为其 **Gen-4** 模型推出了 **Chat Mode**，提供了一个全新的对话式界面来生成图像和视频。联合创始人 [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1933238580400537698) 解释说，这是迈向自适应界面的一步，使媒体生成更加自然和直观。
- **"Cursor + Claude Code" 技术栈**：将 **Cursor IDE** 与 **Anthropic 的 Claude Code** 结合使用的开发者体验受到了广泛赞誉。像 [@cloneofsimo](https://twitter.com/cloneofsimo/status/1933177834119610427) 这样的用户报告称生产力大幅提升，而 **Y Combinator** 的播客[邀请了 Anysphere 首席执行官 Michael Truell 讨论该产品的愿景](https://twitter.com/dilipkay/status/1933099751370613185)。
- **Instagram 的 3D 照片集成**：[@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1933199948759146810) 指出，测试版的 **Instagram 3D 照片集成**做得“非常出色”，将静态照片转化为 AI 生成的立体图像。他认为这是迈向全 6DOF 模型生成的敲门砖。
- **TorchAO 为 RTX 4090 启用 FP8**：[@RisingSayak](https://twitter.com/RisingSayak/status/1933187476509917471) 分享了 **TorchAO** 现在支持 **SM89** 架构 GPU（如 **RTX 4090**）的 **FP8**，显示出 **Flux** 等模型的显著加速。
- **UnslothAI 用于 Reward Model 服务**：[@danielhanchen](https://twitter.com/danielhanchen/status/1932965003621204391) 宣布，使用 **UnslothAI** 可以使 Reward Model 服务和序列分类的 **Inference 速度提升 2 倍**。

**基础设施、行业事件与融资**

- **重大云服务中断冲击 AI 服务**：一场似乎源于 **GCP** 和 **Cloudflare** 等云服务商的大规模网络中断，[导致整个 AI 生态系统出现大范围干扰](https://twitter.com/gregisenberg/status/1933242926337077272)。**OpenAI** 报告了影响 SSO 和登录方式的问题，随后[宣布全面恢复](https://twitter.com/OpenAI/status/1933260549045039549)。其他受影响的服务包括 **Weights & Biases**、**LangSmith**、**Replit** 和 **Cursor**。这一事件引发了 **DHH** 对[云集中化危险的评论，他认为这“破坏了互联网的主要设计目标：聚合韧性”](https://twitter.com/vikhyatk/status/1933258625327509646)。
- **黄仁勋在巴黎 GTC 展示 Perplexity Labs**：**Perplexity** 首席执行官 [@AravSrinivas](https://twitter.com/AravSrinivas/status/1932968936938537223) 分享了一张 **NVIDIA 首席执行官黄仁勋 (Jensen Huang)** 在巴黎 **GTC** 活动上展示 **Perplexity Labs** 的照片。
- **Sam Altman 和 Lisa Su 出席 AMD Advancing AI 活动**：**OpenAI 首席执行官 Sam Altman** 与 **AMD 首席执行官 Lisa Su 博士**一同出现在 **AMD #AdvancingAI** 主题演讲中，[正如 Lamini 的 Sharon Zhou 所分享的那样](https://twitter.com/realSharonZhou/status/1933231029516648554)。
- **Google 的开源推进**：**Google 的 Jeff Dean** 强调，[**Google 已在 Hugging Face 上发布了 999 个开源模型**](https://twitter.com/ClementDelangue/status/1933107694585487803)，这是对开源社区的重大贡献。**Hugging Face 首席执行官 Clément Delangue** 指出，相比之下，Meta 发布了 387 个，Microsoft 发布了 250 个。
- **融资势头**：[@scottastevenson](https://twitter.com/scottastevenson/status/1933117996068905457) 宣布在重新专注于产品开发之前，在**两周内收到了四份投资条款清单 (term sheets)**。
- **Perplexity 预告 "Comet" 发布**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1933289407705960697) 表示，**Perplexity** 即将推出的产品 **Comet** 是“无与伦比的”，随着它接近最终测试阶段，将发出更多邀请。

**地缘政治、批评与广泛评论**

- **AI 竞赛中的雄心**：**Perplexity** 的 [@AravSrinivas](https://twitter.com/AravSrinivas/status/1933283015586951623) 发布了一段充满抱负的言论，称：“**Google 向世界展示了**你可以拥有自己的搜索、AI、数据中心、芯片、手机、OS、浏览器……所以，不要志向短浅。要有雄心壮志。”
- **对现代模型能力的批评**：[@corbtt](https://twitter.com/corbtt/status/1932977024882389253) 对 LLM 的现状表示沮丧，问道：“**为什么现代模型在写作方面仍然如此糟糕？**”并指出它们无法在不产生“充斥着表情符号的垃圾内容”的情况下总结一篇博客文章。与此类似，[@teortaxesTex](https://twitter.com/teortaxesTex/status/1933371065863909638) 认为 o3 竟然会被简单的脑筋急转弯难倒，“对 OpenAI 来说相当尴尬”。
- **美中技术紧张局势**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1932955304188076081) 评论了有关**中国要求美国允许 ASML 出口成熟制程光刻机**以确保 SMIC 的 14nm 产能的报道，认为这是一个合理的请求。
- **人类与 LLM 推理的对比**：[@goodside](https://twitter.com/goodside/status/1932965557214851229) 为推理辩论提供了一个新的类比：“人类和 LLM 的推理就像苹果和土豆一样不同。也就是说，一个是另一个更甜的版本，有着更光鲜的外皮，但两者都长在藤蔓上，并且经常杂交。”
- **扎克伯格过去的裁员**：[@jeremyphoward](https://twitter.com/jeremyphoward/status/1933329447853437251) 评论道，如果 **Mark Zuckerberg** “几年前没有解雇 Erik 那支由卓越 AI 人才组成的团队，他们今天的 AI 人才问题就不会那么严重。”

**幽默与迷因**

- **OpenAI 🤝 Mattel**：[@gdb](https://twitter.com/gdb/status/1933221591350964633) 发布了一张 **Barbie** 品牌电脑的照片，配文为 “OpenAI 🤝 Mattel:”。
- **现代程序员**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1933273732212003237) 发布了一张一个人在写代码的照片，并评论道：“这感觉已经过时了”。
- **Weights & Biases 的体验**：[@vikhyatk](https://twitter.com/vikhyatk/status/1932962492696965626) 发布了一张极其复杂、混乱的图表，配文是：“每次我打开 Weights & Biases 时”。
- **PM 的代码建议**：[@cto_junior](https://twitter.com/cto_junior/status/1933131249083875373) 指出，“进步民主化”的一个悲剧是 PM 在 Slack 上发给你 GPT-4o 生成的代码建议。
- **AI 缩写词**：[@goodside](https://twitter.com/goodside/status/1932990995638976668) 调侃了 AI 圈子里 **Y2K 风格缩写词**的回归，例如 “tmol-faq. logi. gisai. cev. sysop. ufai. foom. rpop. flare.”
- **互联网断网反应**：针对大规模云服务中断，[@matanSF](https://twitter.com/matanSF/status/1933232190147706952) 调侃道：“那个你推迟了 3 年的生产环境迁移？现在机会来了。”

---

# AI Reddit 综述

## /r/LocalLlama 综述

### 1. OpenAI 及行业模型发布动态与延迟

- [**请愿：禁止“关于公告的公告”帖子**](https://www.reddit.com/r/LocalLLaMA/comments/1l9lddr/petition_ban_announcement_of_announcement_posts/)（[得分：664，评论：78](https://www.reddit.com/r/LocalLLaMA/comments/1l9lddr/petition_ban_announcement_of_announcement_posts/)）：**该帖子批评了关于 AI 模型发布（尤其是来自 OpenAI 等机构）的“关于公告的公告”帖子的泛滥，强调这些更新具有重复性且缺乏实质内容。评论者从技术角度指出，未经证实的来源（如粉丝量少的 Twitter 账号截图）会产生噪音，并强调需要信誉良好的、经过验证的新闻来源，而不是投机性或炒作驱动的帖子。** 社区管理最佳实践引发了技术辩论，建议包括对发布请愿/公告进行更严格的验证，以及利用个人屏蔽功能来过滤信息流，而不是依赖广泛的禁令。
    - 一位评论者提到了关于 AI 模型投机性泄露的盛行，例如关于 DeepSeek v0.2.1.2 发布的早期传闻，并建议新闻帖子应仅限于来自信誉良好或官方渠道的信息，以减少误导信息和未经证实的炒作。
    - 有人提出对公告类帖子实施更严格的发布要求，例如仅允许注册时间超过三个月的账号发布，旨在遏制垃圾信息并提高模型和工具更新通知的可信度。
    - 讨论还包括：完全禁止“关于公告的公告”帖子可能会压制有关即将发布的模型（如 DeepSeek R2）的合法信息，因此，采取细致的管理或对来源可靠性进行验证可能更为合适。

- [**OpenAI 推迟其开源模型，声称将添加“令人惊叹的东西”**](https://techcrunch.com/2025/06/10/openais-open-model-is-delayed) ([Score: 344, Comments: 145](https://www.reddit.com/r/LocalLLaMA/comments/1l9fec7/openai_delays_their_open_source_model_claiming_to/)): **OpenAI 宣布推迟其开源模型的发布，理由是计划为其“添加一些令人惊叹的东西”，尽管目前尚未公开任何技术细节或新的基准测试（benchmarks）([来源](https://www.reddit.com/r/LocalLLaMA/comments/1daxzbh/openai_delays_their_open_source_model_claiming_to/))。社区正在等待有关实现、模型架构或预期进展的更多细节。** 热门评论推测，延迟是由于额外的对齐（alignment）、安全或限制性护栏（guardrails），可能会为了增加安全性而牺牲开放性，一些人对最终发布的实际效用持怀疑态度。
    - 一位用户指出，**OpenAI 发布最开放的 LLM 仍然是 GPT-2**，而较新的产品如 GPT-3 甚至没有提供用于本地使用的 GGUF 格式。这与**阿里巴巴**等公司形成了鲜明对比，后者发布了最先进的开源模型，这些模型是可访问且可本地化的，*尽管存在硬件和禁运限制*。
    - 提出的另一个技术点是，新的开源模型即使发布，也可能带有沉重的护栏——这可能会影响其效用。人们怀疑额外的安全特性（“护栏”）是否会使模型在技术和实验应用中的用处降低。
- [**谷歌和微软 vs OpenAI 和 Anthropic，过去一年他们在 Hugging Face 上发布的开源作品的趣味可视化（Julien Chaumond 在 LinkedIn 上的分享）**](https://i.redd.it/2vdfa3f5sg6f1.jpeg) ([Score: 473, Comments: 45](https://www.reddit.com/r/LocalLLaMA/comments/1l9hzb5/google_and_microsoft_vs_openai_and_anthropic_a/)): **该图片是一个日历风格的可视化图表，比较了过去一年中谷歌、微软、OpenAI 和 Anthropic 在 Hugging Face 上发布的开源模型、数据集和 Space 的数量。它突显了显著的差异：与 OpenAI 和 Anthropic 稀疏的发布频率相比，谷歌和微软表现出更高的发布活跃度（密集的彩色方块集群）。随附的数据包括每个组织的关注者数量和总发布量，进一步强化了在开源参与度方面的对比。该可视化归功于 Julien Chaumond，评论中还引用了一个用于创建类似热力图（heatmap）的工具 (https://huggingface.co/spaces/cfahlgren1/model-release-heatmap)。** 评论者强调，谷歌、微软和 Facebook 在历史上对开源的贡献远不止于 AI，而 Anthropic 并没有声称自己是开源的。Caleb Fahlgren 链接的 Hugging Face 热力图工具为用户提供了进一步探索发布数据的方法。
    - 一位用户强调了主要中国 AI 公司的缺席，指出了来自阿里巴巴（`qwen`）和 DeepSeek 的重大开源模型发布，认为该热力图低估了全球开源 AI 贡献的格局——尤其是来自中国的最新进展。
    - 另一位用户分享了一个可视化工具：由 Caleb Fahlgren 开发的 [Hugging Face Model Release Heatmap](https://huggingface.co/spaces/cfahlgren1/model-release-heatmap)，该工具跟踪并直观地展示了各组织的公共模型发布活动，提供了对贡献时间线和组织趋势的细粒度洞察。

- [**Qwen3-72B-Embiggened**](https://huggingface.co/cognitivecomputations/Qwen3-72B-Embiggened) ([Score: 112, Comments: 44](https://www.reddit.com/r/LocalLLaMA/comments/1l9rejn/qwen372bembiggened/)): **Qwen3-72B-Embiggened 是一个开源的实验性 LLM，通过使用两阶段方法将 Qwen3-32B 模型扩展到与完整的 Qwen3-72B 架构相匹配：结构感知插值（structure-aware interpolation，用于重新缩放隐藏层和中间激活值）和简单的中间层复制，从而规避了从头开始训练的需求。该模型保留了架构特性（如 Group Query Attention），以 145GB bf16 分片权重形式分发（[Hugging Face 链接](https://huggingface.co/cognitivecomputations/Qwen3-72B-Embiggened)），初步实现了** `80% coherence` **和** `24.25` **perplexity，但包含许多重复层（在进一步微调前是完全相同的），主要用于在进行完整训练或蒸馏之前，需要大规模 LLM 的原型设计或研究。该模型需要大量的计算资源（145GB VRAM bf16），并计划进行嵌入后微调或蒸馏（例如从 Qwen3-235B 蒸馏），以增强能力和差异化。** 热门技术评论对命名规范表示担忧（以避免与官方 Qwen3 发布版本混淆），并建议将 Qwen3-235B 或 Deepseek 等模型蒸馏到此架构中，以复兴和扩展目前代表性不足的 70B 参数段。人们对这种蒸馏后的 72B 模型在基准测试中的表现以及相对于其他大规模 LLM 的性能表现表现出极高的兴趣。
    - Qwen3-72B-Embiggened 的“扩容”（embiggening）过程涉及两阶段技术：结构感知插值和简单的层复制，将 Qwen3-32B 模型扩展到与完整的 Qwen3-72B 架构相匹配，从而有效地从较小的权重创建出一个 72B 规模的模型。
    - 计划中的下一步包括将大得多的 Qwen3-235B 模型蒸馏到这个 72B 规模的架构中，这一过程可能会产生一个新模型（Qwen3-72B-Distilled），在目前开源模型领域所缺乏的架构高效形态中，提供潜在的有价值的性能提升。
    - 存在技术和品牌方面的争议：人们对命名规范提出了担忧，认为具有重大架构修改的模型应与官方 Qwen3 系列明确区分，以避免在来源和特性方面误导用户。

### 2. 开源模型发布与生态工具

- [**Nanonets-OCR-s：一个支持 LaTeX、表格、签名、复选框等的开源图像转 Markdown 模型**](https://www.reddit.com/r/LocalLLaMA/comments/1l9p54x/nanonetsocrs_an_opensource_imagetomarkdown_model/) ([Score: 224, Comments: 34](https://www.reddit.com/r/LocalLLaMA/comments/1l9p54x/nanonetsocrs_an_opensource_imagetomarkdown_model/))：**Nanonets-OCR-s 是一款开源的、拥有 3B 参数的 VLM 模型，能够将各种文档特征（包括表格、公式、图像、签名、水印、复选框）转换为结构化的 Markdown 和 HTML。显著的技术能力包括精确的 LaTeX 公式识别（区分行内/块级）、语义图像打标、签名/水印提取，以及将复杂的表格和表单元素稳健地处理为 Markdown 兼容格式。更多详情和模型资源可通过 [Hugging Face Model Card](https://huggingface.co/nanonets/Nanonets-OCR-s)、[完整公告](https://nanonets.com/research/nanonets-ocr-s/) 和 [Colab 快速入门](https://github.com/NanoNets/docext/blob/main/PDF2MD_README.md#quickstart) 获取。** 评论者反映其表格提取性能优于 Gemini VLM，并对支持 GGUF 格式以进行本地部署表示了兴趣。
    - 一位用户将 Nanonets-OCR-s 与 Gemini VLM 进行了基准测试，报告称其在处理“复杂表格”时具有更优越的表格提取性能，并强调了它在复杂结构化数据场景中的有效性。
    - 有一项请求是关于一致的文档结构控制，建议如果源文档格式一致，可以通过增加格式化功能来进一步增强模型，以确保输出的 Markdown 严格遵循输入文档的布局。
    - 另一个建议提议原生 Markdown 图像输出，包括自动构建带有边界框（bounding box）、页面引用的图像标签，并支持轻松提取，以及在 Markdown 输出中提取脚注/引用并进行正确格式化的功能。
- [**Mistral.rs](http://mistral.rs/) [v0.6.0 现已提供完整的内置 MCP 客户端支持！](https://www.reddit.com/r/LocalLLaMA/comments/1l9cd44/mistralrs_v060_now_has_full_builtin_mcp_client/)** ([Score: 105, Comments: 15](https://www.reddit.com/r/LocalLLaMA/comments/1l9cd44/mistralrs_v060_now_has_full_builtin_mcp_client/))：**[mistral.rs](http://mistral.rs/) [v0.6.0](https://github.com/EricLBuehler/mistral.rs/) 的发布引入了紧密集成的 MCP (Model Context Protocol) 客户端支持，简化了 LLM 工具的集成。此次更新通过简单的配置 (**`mcp-config.json`**) 实现了对各种外部工具和服务（文件系统、HTTP/SSE、WebSocket）的自动发现和连接，无需手动编写集成代码和独立的工具注册表。此支持原生存在于 Rust 库及其 Python 绑定中（可通过 [PyPI](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/_README.md#installation-from-pypi) 获取），并允许无缝使用标准的 OpenAI API 接口，且完整支持多服务器、身份验证和超时。提供了显著的 [快速入门](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/MCP_QUICK_START.md) 和 [Python 示例](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/mcp_client.py)。** 一位评论者询问了类似于 `llama.cpp` 参数（`fa -ctk q4_0 -ctv q4_0`）的键值 (KV) 缓存压缩选项，表明了对高级内存优化功能的需求。此外，还有关于打包方式的讨论，质疑通过 PyPI 而不是 Cargo 分发 Rust 库是否合适，突显了跨生态系统构建和部署的疑虑。
    - 一位用户询问了键值缓存压缩的进展，特别是要求提供类似于 llama.cpp 支持的缓存张量量化功能（`fa -ctk q4_0 -ctv q4_0`），这可以减少内存使用并提高推理效率。这表明了用户对与成熟项目功能对齐的兴趣，并强调了针对部署场景的优化需求。
    - 有一个关于安装/分发选择的技术问题：一位用户注意到通过 PyPI（通常与 Python 包相关）发布 Rust 库，询问为何没有 Cargo 分发方式，而这通常是 Rust 库的预期方式。这指向了对打包、语言集成和首选部署工作流的关注。
    - 另一个评论请求关于最佳部署路线（Docker 还是本地安装）的指导，反映了对部署便捷性以及通过容器化实现一致环境设置和可复现性的技术兴趣。

### 3. 独特的 LLM 部署与超人工智能领域的行业投资

- [**在 PS Vita 上运行 LLM**](https://v.redd.it/we6m8zvv4f6f1) ([Score: 173, Comments: 13](https://www.reddit.com/r/LocalLLaMA/comments/1l9cwi5/running_an_llm_on_a_ps_vita/)): **一位开发者利用 VitaSDK 工具链，成功将极简的 Llama2.c 大语言模型推理引擎移植到了 PlayStation Vita。关键的技术适配包括 PS Vita 特有的 syscalls 以及设备端模型下载/删除功能，允许直接管理 LLM 模型文件而无需手动传输（参见 [psvita-llm repo](https://github.com/callbacked/psvita-llm)）。目前已提供预编译的 .vpk 文件以便快速安装，展示了在低内存（<512MB RAM）嵌入式平台上部署 LLM 的可能性。** 评论区的讨论集中在 VitaSDK 相对于其他自制软件（homebrew）环境（特别是 Nintendo Switch）的学习曲线和设置复杂度，以及社区对将 LLM 移植到 PSP 和 PS2 等旧款游戏机的兴趣。
    - 一位评论者强调了将 `llama2.c` 移植到 PS Vita 的技术挑战，指出将原本为桌面/服务器环境优化的 CPU 密集型语言模型实现适配到掌上游戏机，可能需要大量的修改和优化。这包括针对内存限制的调整、CPU 指令集兼容性，以及可能需要为 Vita 独特的硬件和 OS API 重写底层系统调用。
    - 有一个关于 PS Vita SDK 与 Switch 等现代自制软件开发相比上手难度的深刻提问，提到了设置开发环境时的痛点，并可能暗示了不同平台在工具链成熟度、文档和社区支持方面的差异。
    - 几位用户提到该项目的仓库链接已失效 (404)，这引发了关于获取实际实现细节、代码和文档的疑问——这对于那些有兴趣进行技术复现或审查受限硬件上 LLM 推理方法的人来说至关重要。
- [**Meta 开出九位数薪资以构建超人工智能，马克全力投入。**](https://www.reddit.com/r/LocalLLaMA/comments/1l9wbaw/meta_is_offering_nine_figure_salaries_to_build/) ([Score: 123, Comments: 67](https://www.reddit.com/r/LocalLLaMA/comments/1l9wbaw/meta_is_offering_nine_figure_salaries_to_build/)): **据报道，Meta 正在提供九位数级别（**`$100M+`**）的薪酬方案，以组建一支顶尖团队从事超人工智能研究，这呼应了马克·扎克伯格（Mark Zuckerberg）公开推动在前沿模型领域展开激进竞争的举措。引用的 Entrepreneur 文章指出，Meta 努力招募 AI 领域的知名人士，但最近的高调聘用似乎集中在争取前初创公司创始人，而非像 Sutskever 或 Hassabis 这样的基础研究人员。** 评论表达了对 Meta 做法的怀疑，认为高额薪酬在一定程度上是为了留住顶尖人才并防止竞争，并将其与 Meta 元宇宙计划反响平平的采用情况进行了类比。此外，还有关于 Meta 是否吸引到了具有公认技术影响力的精英 AI 研究员或创始人的批判性讨论。
    - 与领先的 AI 团队相比，人们对 Meta 的人才招聘持怀疑态度——用户质疑 Meta 是否争取到了像 *Ilya Sutskever* (OpenAI)、*Demis Hassabis* (DeepMind) 这样的世界级研究员，或者是否充分利用了像 *Yann Lecun* 这样的内部专家。相反，有人注意到他们聘请了一位亿万富翁创始人，这可能与带来前沿研究或技术领导力并不相关。
    - 提到“九位数薪资”暗示了激进的薪酬策略，其目的可能是防止顶尖人才加入竞争对手或创办自己的初创公司/实验室。这反映了更广泛的行业趋势：由于先进 AI 研究被视为具有战略重要性，留住精英 AI 人才已变得极具竞争性且耗资巨大。

## 其他 AI Subreddit 摘要

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo

### 1. Claude Code：用户体验、生产力与 Agent 技术

- [**ClaudeCode 让编程再次变得有趣**](https://www.reddit.com/r/ClaudeAI/comments/1l9ta7s/claudecode_made_programming_fun_again/) ([Score: 137, Comments: 31](https://www.reddit.com/r/ClaudeAI/comments/1l9ta7s/claudecode_made_programming_fun_again/)): **用户强调，使用 ClaudeCode（Anthropic 的 AI 代码助手）显著减少了在阅读模糊文档、排查 Bug 和工具链问题等编程单调环节上花费的时间，从而能够更直接地推进项目进度。用户们一致认为 Claude Code（尤其是 Max 订阅计划）在处理“繁琐工作”和边缘情况的 Bug 修复方面表现出色，提高了整体生产力和编程乐趣。** 技术评论者认为 ClaudeCode 的价值主张在于自动化重复性的故障排除，使开发者能更专注于实际开发。主要的争论点在于这在多大程度上改变了程序员的工作流，以及是否存在过度依赖的风险。
    - 用户报告称，在使用 Claude Code 时，调试和处理边缘情况的时间显著减少，并强调该工具在自动化繁琐编程任务方面表现卓越。这种效率显著地让开发者重新专注于开发的创意层面，多位拥有丰富经验的开发者也反映了这一点。一位使用 Max 计划的用户指出，在采用该工具仅一周后，生产力就有了实质性的提升。
- [**温馨提示 - 别忘了你可以在 Claude code 中调用子 Agent。**](https://www.reddit.com/r/ClaudeAI/comments/1l9ja9h/psa_dont_forget_you_can_invoke_subagents_in/) ([Score: 124, Comments: 50](https://www.reddit.com/r/ClaudeAI/comments/1l9ja9h/psa_dont_forget_you_can_invoke_subagents_in/)): **该帖子强调了根据 Anthropic 官方文档（[Claude code 最佳实践](https://www.anthropic.com/engineering/claude-code-best-practices)）有效使用 Claude 的子 Agent 来处理复杂任务、验证以及多文件/文档审查，并突出了它们在减少任务幻觉和提高上下文保留方面的作用。明确指示 Claude 为指定子任务（如代码审查、文件分析或测试）使用子 Agent，据报道可以提高资源效率（降低虚拟内存占用）并提供更可靠的结果，这可能归功于后端在信息处理方面的优化。官方文档还讨论了 Agent 工具的安全性考虑（[Anthropic 安全文档](https://docs.anthropic.com/en/docs/claude-code/security)）。** 评论者指出，为每个任务步骤指定子 Agent 的数量和范围对于获得最佳性能至关重要，而子 Agent 的调用策略是一项需要经验的技能。目前还有一个尚未解决的技术问题，即子 Agent 调用是否会消耗父 Agent 的上下文窗口（context window），此外一个社区资源详细阐述了任务/Agent 工具的机制（[claudelog.com](http://claudelog.com/) [文章](https://claudelog.com/mechanics/task-agent-tools)）。
    - 一位技术用户讨论了在 Claude 中实现子 Agent 的策略，强调定义每个任务步骤的子 Agent 数量并明确其职责可以提高任务吞吐量和效率。如帖子和官方文档（[Anthropic 文档](https://docs.anthropic.com/en/docs/claude-code/security)）所述，鼓励通过实验来优化子 Agent 的使用，以匹配任务的复杂性和并行化潜力。
    - 一位评论者提出了关于上下文预算（context budget）的技术担忧，质疑在 Claude 中调用子 Agent 是共享父 Agent 的上下文大小，还是拥有独立的上下文限制，这对于扩展复杂或数据密集型任务是一个关键问题。
    - 一位用户分享的经验证据表明，将子 Agent 与代码浏览工具（特别是 MCP → LSP，可能指 Language Server Protocol 集成）配合使用，与传统的基于 grep 的方法相比，能够实现更高效的代码探索和搜索，这表明代码分析工作流的生产力和自动化程度有所提高。

### 2. AI 视频生成、动画与创意应用 (Veo, i2v, Midjourney, Kling 等)

- [**为使用 Vace 的 Self Forcing 工作流添加了 i2v 支持**](https://www.reddit.com/gallery/1l9kt2t) ([Score: 103, Comments: 54](https://www.reddit.com/r/StableDiffusion/comments/1l9kt2t/added_i2v_support_to_my_workflow_for_self_forcing/)): **该帖子宣布将图像转视频 (i2v) 支持集成到使用 Vace 的 Self Forcing 工作流中，并强调虽然输出视频质量并非顶尖，但生成速度非常快（根据用户反馈，视频生成时间约为 40 秒）。该工作流和模型可通过 [CivitAI](https://civitai.com/models/1668005/self-forcing-simple-wan-i2v-and-t2v-workflow) 获取。** 评论者表达了对更大版本（`14b`）模型发布的期待，承认该工作流是快速 I2V 生成的“游戏规则改变者”，并对了解“self forcing”的技术细节表现出兴趣。
    - 一位用户报告使用新工作流在 `40 秒` 内实现了 i2v（图像转视频）生成，强调了显著的性能提升，并将其描述为 I2V 任务的“游戏规则改变者”。
    - 另一位用户确认了极快的生成速度，提到在 `4070 Ti Super` GPU 上使用默认设置创建一段 i2v 剪辑仅需 `一分钟`，为该级别硬件提供了有用的实际性能参考点。
- [**4 分钟 AI 动画故事 - 超过 500 个视频的实验 - 成本 1000+ 美元**](https://v.redd.it/9zp1jk21ag6f1) ([Score: 363, Comments: 136](https://www.reddit.com/r/aivideo/comments/1l9gnur/4minute_ai_animated_story_over_500_videos_of/)): **原作者制作了一个 4 分钟的 AI 生成动画，其流水线包括：使用 Midjourney 创建资产（背景/角色），使用 Pika Scenes 进行动画制作，以及使用 Topaz 进行视频放大和插帧；整个过程涉及 500 多个独立视频，成本超过 1000 美元。关于工作流和方法的详细信息在评论区的补充内容链接中分享。** 一位评论者从技术角度批评该动画是一系列简短的“2 秒 gif”组合，而非传统的动画故事，突显了当前 AI 动画工具的时间局限性；另一位评论者预测，在真人电影采用之前，全 AI 生成的动画将获得越来越多的认可，同时也提到了对其美学价值的欣赏。
    - 一位评论者批评了动画的技术层面，指出虽然工作流产生了视觉输出的一致性和连贯的叙事，但镜头选择和剪辑被描述为“突兀”。他们强调，掌握 AI 工具并不能取代对传统导演原则（如视线连贯性和有效的叙事剪辑）的理解，而这些对于提升故事讲述质量至关重要。
    - 另一位评论者指出，这种动画风格本质上是一系列简短的 2 秒 gif，而不是传统的流畅动画，这可能会影响习惯于传统动画短片的观众对连贯性和参与感的感知。
- [**AI 剧集 Seraphys 的预告片**](https://v.redd.it/4fbu8tf4zj6f1) ([Score: 193, Comments: 63](https://www.reddit.com/r/aivideo/comments/1l9wgr7/trailer_for_the_ai_show_seraphys/)): **创作者通过整合多种 AI 和传统工具，为名为“Seraphys”的 AI 驱动系列剧制作了一段概念预告片：剧本为手动编写，视觉资产由 Midjourney v7 生成并在 Photoshop 中处理，通过 Kling 2.1 进行图像转视频转换，配音/SFX 使用 Eleven Labs，面部动画使用 HeyGen Avatar 4，音乐使用 Udio，最后在 DaVinci Resolve 中进行剪辑、调色和 VFX 处理。一些额外的非 AI 音效/音乐资产源自 Uppbeat，展示了混合工作流。这说明了一个端到端的内容流水线，利用了跨媒体类型的最先进 AI 生成工具，展示了它们的互操作性并辅以人工策展。** 评论强调了人们对预告片质量和 AI 快速进步的普遍赞叹，虽然没有出现技术批评或辩论，反映出对当前 AI 工具在创意工作流中进步的压倒性正面评价。
    - creuter 强调了技术成就，指出预告片时长为 54 秒，而预定运行时间为 10 小时，这引发了关于在如此宏大的时间跨度内内容生成的扩展性和完成度的推测。这提出了关于长篇视频生成的生产工作流和 AI 模型吞吐量的问题。

- [**Yeti 在这个广告提案中对决 YETI**](https://v.redd.it/msayw097tj6f1) ([Score: 102, Comments: 28](https://www.reddit.com/r/aivideo/comments/1l9vnla/yeti_takes_on_yeti_in_this_spec_ad/)): **该帖子讨论了使用 Google 的 Veo 3 视频生成模型为 YETI 制作广告提案（spec ad），这是 Curious Refuge 作业的一部分，参考了流行视频博客（vlogs）的格式。Veo 3 因其能够根据 text prompts 生成连贯视频序列的能力而备受关注，重点在于根据用户意图实现现实主义和 style transfer（参见 [Google Veo 论文](https://blog.google/technology/ai/google-veo/)）。** 评论者关注 Veo 3 生成内容的有效性和吸引力，对质量和娱乐价值表示惊讶，尽管最初对 AI 生成的媒体持怀疑态度。
    - 一位评论者指出，整个广告中的音效质量非常高，增强了整体影响力和参与度。他们特别指出“在顶部滑雪”片段是一个亮点，认为该序列中的音频和剪辑都执行得很好。还有建议进一步压缩视频以获得更吸引人的流程，这意味着关注节奏和后期制作的精细化可以提升技术磨合度。
- [**如果耶路撒冷在公元前有街头采访**](https://v.redd.it/lcc0paubkj6f1) ([Score: 281, Comments: 26](https://www.reddit.com/r/ChatGPT/comments/1l9ucko/if_jerusalem_had_street_interviews_ad/)): **该 Reddit 帖子展示了一个喜剧视频，时代错误地想象了发生在古代耶路撒冷（公元前）的“街头采访”，用户讨论引用了 VEO3 视频中可能出现的梗图风格对话。帖子或热门评论中没有提到算法、Benchmarks、实现或任何软件框架等技术内容。** 用户反应积极，称该视频“非常有趣”并赞赏其讽刺手法；一些人指出通常对梗图格式持负面态度，但表示这次执行在喜剧价值上高于平均水平。
    - 此线程中的评论均未提供实质性的技术内容、详细的模型讨论、Benchmarks 或工程见解。讨论仅由幽默、文字游戏和通用赞美组成。

### 3. 开创性 AI 研究、行业辩论及全球 AI 影响

- [**祝开启这一切的那篇论文 8 周岁生日快乐**](https://i.redd.it/ka788zoani6f1.jpeg) ([Score: 1336, Comments: 92](https://www.reddit.com/r/singularity/comments/1l9ple2/happy_8th_birthday_to_the_paper_that_set_all_this/)): **该图片是来自 arXiv 的截图，显示了 2017 年 6 月 12 日提交的论文《Attention Is All You Need》，修订版截止至 2023 年 8 月 2 日。Vaswani 等人的这项开创性工作引入了 Transformer 架构，用 self-attention 取代了 recurrence，并引发了生成式 AI 的基础性进步。截图包括作者姓名以及在 Computer Science > Computation and Language 下的分类。** 评论指出了自该论文发表以来的飞速进步，提到了 GPT-1 发布以来的 7 年里程碑，并赞扬了论文标题简洁地概括了范式转移。
    - 一位用户指出，自 GPT-1 发布以来已有 7 年，强调了从最初的 Transformer 论文到当前最先进架构的大语言模型（LLM）的快速发展和演变。这突显了 attention 机制的基础研究如何迅速转化为像 OpenAI 的 GPT 系列那样实用且可扩展的实现。
    - 另一位评论者提到了 Vaswani 等人的《Attention Is All You Need》论文的持久影响力，强调其引入的 attention 机制从根本上改变了 AI 研究的轨迹，并认为它的影响力将被审视数十年，作为一篇通过实现 NLP 及其他领域的巨大进步而“改变人类”的论文。

- [**Google DeepMind 凭借新型 AI 模型永久改变了飓风预报**](https://venturebeat.com/ai/google-deepmind-just-changed-hurricane-forecasting-forever-with-new-ai-model/) ([Score: 988, Comments: 57](https://www.reddit.com/r/singularity/comments/1l9or4z/google_deepmind_just_changed_hurricane/)): **Google DeepMind 推出了一套全新的 AI 飓风预报系统 [Weather Lab](https://weather.deepmind.com/)，专门用于预测风暴路径和强度，利用集合技术可提供长达 15 天的预报。根据国家飓风中心 (NHC) 协议进行的内部评估显示，DeepMind 模型的五天路径预测平均误差比 ECMWF 的 ENS 低** `140 km` **，超越了全球低分辨率（路径）和区域高分辨率（强度）模型，并标志着 AI 首次实验性地整合进 NHC 的业务工作流。该 AI 利用深度学习同时模拟大尺度大气动力学和细尺度风暴强度，旨在解决当前基于物理的方法中因分辨率限制而导致路径和强度建模分离的权衡问题。** 评论者提到了 DeepMind 在具有影响力的科学 AI 方面的往绩（如 AlphaFold）、封闭模型与开源模型的创新性，并希望在龙卷风预测方面取得类似进展。技术辩论集中在透明度、模型可访问性以及专用深度学习在天气预报中的实际影响。
    - DeepMind 模型的核心创新在于能够同时预报气旋的路径和强度。传统模型要么采用侧重于路径预测的全球低分辨率方法，要么采用侧重于强度的区域高分辨率方法——DeepMind 声称弥补了这一差距，标志着对以往技术的重大突破。
    - 在使用国家飓风中心协议的内部基准测试中，DeepMind 报告称其 AI 的 5 天飓风路径预报平均比 ENS（领先的欧洲基于物理的集合模型）更接近实际风暴结果 140 公里，展示了与业务预报相关的实质性实际改进。
    - 美国国家飓风中心将首次把实验性 AI 预测整合到其业务工作流中，加速了机器学习模型在关键基础设施中的直接采用，并为未来气象领域的 AI 合作树立了先例。
- [**Nvidia 的 Jensen Huang 表示，他几乎不同意 Anthropic 首席执行官 Dario Amodei 所说的一切**](https://fortune.com/2025/06/11/nvidia-jensen-huang-disagress-anthropic-ceo-dario-amodei-ai-jobs/) ([Score: 532, Comments: 153](https://www.reddit.com/r/singularity/comments/1l9o8m9/nvidias_jensen_huang_says_he_disagrees_with/)): **Nvidia 首席执行官 Jensen Huang 公开反对 Anthropic 首席执行官 Dario Amodei 关于 AI 发展轨迹和风险的技术评估：Huang 特别质疑了 Amodei 关于五年内 50% 的初级办公职位可能被前沿 AI 自动化的预测，并拒绝了只有特定公司（如 Anthropic）才是 AI 安全管理者的暗示。Huang 主张广泛、开放的 AI 发展，并坚持认为技术进步在历史上会导致劳动力转型，而非大规模的工作毁灭。此外，Huang 还讨论了 Nvidia 结合 CUDA-Q 的量子/经典混合计算路线图，以及在欧洲建立 20 多个“AI 工厂”的计划。** 评论者的辩论集中在 Amodei 的真实意图上——他是主张监管俘获，还是仅仅对 AI 风险表示担忧——同时也指出了 Huang 可能存在的误解。值得注意的是，Yann LeCun 站在了 Huang 一边，断言 Amodei 的立场在通用 AI 风险、高昂成本和大规模失业方面倾向于排他主义和危言耸听。
    - 关于 Dario Amodei 因 AI 导致失业预测的准确性存在技术辩论，Jensen Huang 承认潜在的失业情况，但质疑 Amodei 预测的规模和深度。评论者指出，自动化带来的历史性生产力提升通常会导致工作转型而非绝对流失，挑战了一些行业领袖的轻视态度。
    - Yann LeCun 链接的评论总结了针对 Amodei 的三大批评：1) 夸大 AI 风险以证明只有少数公司有权构建它；2) 过度关注 AI 开发的成本准入门槛；3) 对 AI 经济影响的夸张言论，特别是在劳动力流失方面。考虑到 Nvidia 的硬件定价和市场主导地位，关于成本高昂的说法被认为具有讽刺意味。

- 存在技术上的怀疑，即 Amodei 是否真的主张只有 Anthropic 或少数实体应该构建 AI，反驳观点引用了他一贯主张在 AI 治理中进行广泛的多利益相关者协作。关于监管俘获（regulatory capture）的指控被讨论，但在公开声明或政策提案中未发现确凿证据。
- [**Apple 的“AI 无法推理”声明观看量超 1300 万，你需要了解的内容**](https://youtu.be/wPBD6wTap7g) ([Score: 150, Comments: 92](https://www.reddit.com/r/singularity/comments/1l9snr4/apples_ai_cant_reason_claim_seen_by_13m_what_you/)): **Apple 最近的论文声称当前的 Large Language Models (LLMs) 缺乏真正的推理能力，而是表现出先进的模式匹配（pattern-matching），并且在复杂谜题上经常失败——尤其是随着任务复杂度的增加——除非辅以外部工具或代码解释器。批评者指出，该论文没有考虑工具的使用，在超出 token 限制的情况下测试了一些问题，并重申了已知的局限性（例如 LLM 幻觉和受 token 限制的输出），这表明研究结果仅仅证实了社区在缺乏架构改进或系统工具集成的情况下对 LLMs 的已有认知。有关素材，请参阅 [AI Explained 视频解析](https://youtu.be/wPBD6wTap7g)。** 热门评论强调了对 Apple 论文实验设计（偏见、忽略工具使用能力）的技术批评，并重申严肃的研究人员认为这些 LLM 局限性是常识；争论的焦点在于该论文的结论是提供了有意义的新见解，还是仅仅循环利用了关于推理和输出限制的既有担忧。
    - 对 Apple 论文的技术批评集中在其方法论上：它评估了 LLMs 在复杂谜题上的表现，并观察到性能随复杂度增加而下降，但忽略了 LLMs 不是确定性求解器，且已知在无辅助推理方面存在局限。值得注意的是，该论文被指责忽略了 LLMs 使用外部工具（如代码解释器）的能力，这可以显著提高在 LLM 直接推理能力之外的复杂任务上的问题解决性能。
    - 提到的另一个方法论缺陷：该论文在超出其最大输出（token）限制的任务上测试了 LLMs，从而因迫使模型超出其设计规范而使某些结果失效。批评者认为，这一点加上最初框架中对 LLM 推理的感知偏见，意味着该论文对 AI 研究界已经广泛了解的 LLM 弱点几乎没有增加新内容。
    - 强调的一个关键见解是，依赖 LLMs 的实际 AI 应用突破通常源于将这些语言模型与可以补充推理或事实核查的外部系统和工具集成，而不是仅仅依赖语言模型本身的原始建模能力。
- [**如果 GPT 4.5 最近发布且因功耗问题几乎无法使用，那么 GPT 5 应该是怎样的？（Sam 说每个人都可以使用它，甚至是免费账户。）**](https://www.reddit.com/r/OpenAI/comments/1l9k1en/if_gpt_45_came_out_recently_and_is_barely_usable/) ([Score: 225, Comments: 109](https://www.reddit.com/r/OpenAI/comments/1l9k1en/if_gpt_45_came_out_recently_and_is_barely_usable/)): **讨论集中在为什么 OpenAI 在 GPT-4.5（又名 4-turbo）据报道因性能（特别是功耗和成本）问题而被弃用的情况下仍在炒作 GPT-5。过渡路线图表明 GPT-4.5 将被 GPT-4.1 取代。一些评论认为 GPT-5 可能会集成多个模型系列（如 4o, o3, o4, o5），根据每个请求动态选择使用哪个系统，从而可能提高效率和可扩展性，以实现更广泛的访问——包括免费用户。评论者指出，当前的模型（如 o3）比 4.5 便宜得多（“20倍”）且能力更强，使得大规模部署变得可行。版本控制和命名在不同模型间被描述为不一致且令人困惑（参见分享的 [ChatGPT 讨论](https://chatgpt.com/share/684ac50b-60c8-8012-8978-aa0dddd75fa3)）。** 存在对 OpenAI 的版本控制、命名惯例及其路线图的怀疑，多位评论者指出了关于模型升级和更换的公开沟通中的混乱和不一致。
    - 讨论集中在模型演进和部署效率上：评论者指出 GPT-4.5 主要是一个暴力缩放（brute-force scaled）的模型，但在能源使用方面效率低下，正被弃用以支持更轻量级的后继者，如 4.1 和 4-turbo，这两者都更具成本效益（`o3 现在比 4.5 便宜约 20 倍，同时能力更强`）。

- 关于版本命名和架构存在争论：几位用户强调 OpenAI 的版本号（4.5, 5 等）并不直接对应模型规模，而是对应功能和部署实用性。例如，一条评论指出 OpenAI 一直在优化，使 GPT-4 级别的模型“更聪明，除了命名糟糕的 GPT-4.5 之外”，同时实际上降低了成本和规模。
- 有推测认为，未来的版本如 GPT-5 可能会结合多个先前模型（如 4o, o3）的最佳组件/技术，并吸取 4.5 低效架构的教训，以创建一个更平衡、可扩展的 LLM，同时支持免费和付费用户。

---

# AI Discord 简报

> 由 Gemini 2.5 Pro Exp 生成的摘要之摘要之摘要
> 

**主题 1：AI 模型性能与能力的释放（及对比）**

- **三巨头：Opus、O3 Pro、Gemini 2.5 Pro 联手执行任务！**：Perplexity AI 社区的工程师将 **Opus** 用于概念学习，**O3 Pro** 作为分析师，**Gemini 2.5 Pro** 用于数学，并注意到性能取决于具体任务。Cursor 社区用户还发现 **Opus 4** 在编程方面通过减少循环优于 **Sonnet 4**，同时称赞 **Gemini 2.5 Pro** 的批判性思维，尽管它在配置 Postgres 时存在问题。
- **ChatGPT 增强！OpenAI 透露 GPT-4o 的 1500 万美元成本！**：OpenAI 宣布 **ChatGPT Projects** 现在为 **Plus、Pro 和 Team 用户**支持深度研究、语音模式和改进的记忆功能，而 **Canvas** 获得了 PDF、docx 和 markdown 导出功能。与此同时，用户估计 **GPT-4o** 的训练成本约为 **1500 万美元**，在由一段关于 [Veo 3 的 YouTube 视频](https://www.youtube.com/watch?v=QGwWJl7AZ6c) 引发的讨论后，引发了关于 Plus 订阅盈利能力的辩论。
- **视频 AI 激战：Seedance 1.0 击败 Veo3，而 Veo 3 击垮钱包！**：Perplexity AI 的讨论强调，[Seedance 1.0 目前在文/图生视频方面优于 Google 的 VEO3](https://video.twimg.com/amplify_video/1933194931566243840/vid/avc1/1920x1080/HEEIOuxi8TLuVj8y.mp4)。然而，Manus.im 用户对 **Veo 3** 的视频生成成本表示担忧，一名用户报告称，**仅 8 秒**的视频收费在 **300 到 600 积分**之间波动。

**主题 2：当云端哭泣：基础设施困境与平台稳定性传奇**

- **互联网末日！Cloudflare 和 GCP 停机引发 AI 平台恐慌！**：一场涉及 **Cloudflare** 和 **Google Cloud Platform (GCP)** 的大规模互联网停机导致包括 OpenRouter、Cursor、LlamaIndex 和 Cohere 在内的多个 AI 平台出现严重中断。OpenRouter 通过其 [状态页面](https://status.openrouter.ai/) 确认了影响，用户引用了 [Downdetector](https://downdetector.com/) 的报告，Cohere 承认了由于 [Google 报告的 GCP 事件](https://status.cloud.google.com/incidents/mKVakfB1qM3Hvb9cUpqv) 导致的问题。
- **Manus 因 AWS 停机而停摆，LMArena 为丢失的聊天记录哭泣！**：Manus.im 用户由于广泛的 **AWS 停机**（同时也影响了 YouTube 和 Twitch 等服务）而在文件上传和任务执行方面遇到了问题。另外，LMArena 面临 **云服务商停机**，导致聊天历史数据可能丢失，促使团队道歉并致力于采取预防措施。
- **Firebase 倒下，OpenRouter 在末日多米诺效应中跌倒！**：LlamaIndex 和 OpenRouter 的用户报告 **Firebase** 宕机，影响了身份验证服务，正如 [Greg Hunkins 在 X.com 上关于 Firebase 停机的帖子](https://x.com/greghunkins/status/1933223568394846703?s=46) 中所强调的那样。这产生了连锁反应，导致 [OpenRouter 也随之宕机，正如 Hacker News 上所讨论的那样](https://news.ycombinator.com/item?id=44260810)。

**主题 3：压榨 AI 大脑：微调、量化与优化前沿**

- **ABBA 喊出 "Gimme! Gimme! Gimme!"，性能超越 LoRA！**：用于**参数高效微调 (PEFT)** 的全新 **ABBA** 架构在 [其 arXiv 论文](https://arxiv.org/abs/2505.14238) 中进行了详细介绍，且 [代码已在 GitHub 上开源](https://github.com/CERT-Lab/abba)。它通过将更新建模为两个低秩矩阵的 Hadamard 乘积，性能显著优于 **LoRA**。Unsloth AI 和 Eleuther 的成员讨论了它在 **Mistral-7B、Gemma-2 9B 和 LLaMA-3.2 1B/3B** 等模型上持续超越 SoTA LoRA 变体的表现。
- **DeepSeek R1 量化表现极佳，AMD GPU 展现 35 倍推理实力！**：Unsloth AI 成员发现，与 **Qwen3** 相比，**DeepSeek R1** 的量化效果非常好，这可能归功于其 **bf16** 训练，尽管微调 `unsloth/DeepSeek-R1-0528-Qwen3-8B` 时触发了 "Unrecognized keys" 警告。此外，**AMD 的 MI350X 和 MI355X AI GPU** 也引发了热议，据 [Tom's Hardware 报道](https://www.tomshardware.com/pc-components/gpus/amd-announces-mi350x-and-mi355x-ai-gpus-claims-up-to-4x-generational-gain-up-to-35x-faster-inference-performance)，其声称推理性能提升高达 **35 倍**。
- **Torchtune 破解内存之谜并加速 MoE 模型！**：Torchtune 开发者调查了一个内存消耗异常：在使用 **flex attention** 和 **FSDP** 时，`(bs, seqlen*8)` 的输入比 `(bs*8, seqlen)` 占用更多内存，如这张 [内存使用图表](https://cdn.discordapp.com/attachments/1216353675744641096/1382573185916211200/image.png?ex=684c4dde&is=684afc5e&hm=0d498abf01433cd6a078a17e983947e7ac7e0590281a6728dec8a13a5fba2776) 所示。他们还发现，使用 `_grouped_mm` 可以大幅提升**细粒度 MoE (finegrained MoE)** 的速度，使 **Qwen3-30B-A3B** 的性能几乎与 **8B** 模型持平。

**主题 4：开发工具与 API 奇遇：从速率限制到 WASM 梦想**

- **OpenRouter 用户规避速率限制并破解 DeepSeek 的“中文低语”！**：OpenRouter 用户在 Fireworks 提供商的 **Qwen3 30B** 结构化输出上达到了 **10,000 RPM**，并了解到显示的 `rate_limit` 对象并不准确且将被弃用。其他用户报告 **DeepSeek 模型** 在响应过程中会间歇性切换到中文，建议尝试 GMICloud 等替代提供商或 `r1 0528` 版本。
- **Mojo 释放字符串速度猛兽，进驻 LeetGPU！**：Modular 的 **Mojo** 现在其 nightly 构建版本中提供比 Python 快 **40% 的字符串操作**，并已获得 [LeetGPU](https://leetgpu.com/) 的支持，提升了开发者的可访问性。工程师们还展示了其用于 `FastxReader` 的借用迭代器 (borrowing iterators)，并讨论了在目前缺乏动态分派 (dynamic dispatch) 的情况下，使用 [josiahls 的 Variant 库](https://github.com/josiahls/firehose/tree/master/firehose) 的变通方案。
- **MCP 服务器尝试 WASM 与 Service Workers，FastFS 加入阵营！**：MCP (Glama) 社区的开发者探索了直接在浏览器中使用 **service workers** 运行 **MCP 服务器** 的可能性，并可能编译为 **WASM**。Hyper-MCP 提倡在宿主机上使用 WASM，尽管有人担心会丢失 SDK 访问权限，而另一位用户在 GitHub 上分享了他们的 [fastfs-mcp 项目](https://github.com/aj-geddes/fastfs-mcp)，作为“体验其中乐趣”的一个例子。

**主题 5：研究涟漪：掀起波澜的新论文与项目**

- **Factorio AI 旨在打造 "AlphaFactorio" 荣光，逐个 Docker 镜像稳步推进！**：GPU MODE 社区讨论了改进 **Factorio 学习环境 (FLE)** 以帮助超智能系统理解复杂的现实世界系统，正如他们在 [arXiv 上的 FLE 立场论文](https://arxiv.org/pdf/2502.01492) 中所述。一个关键目标是 **AlphaFactorio** 项目，一位成员在 GitHub 上分享了一个概念验证的 [FLE docker 镜像和 mod 项目](https://github.com/MortenTobiasNielsen/fle_suggestion)。
- **世界模型已成趋势，Schmidhuber 表示：Agent 们，戴上你们的预测帽！**：Yannick Kilcher 服务器讨论了来自 Schmidhuber 实验室的一篇 [关于“学习建模世界的 Agent”的新论文](https://arxiv.org/abs/2506.01622)，该论文认为通用 Agent 必须为多步目标导向任务学习其**环境的预测模型**。作者的 [配套博客文章](https://richardcsuwandi.github.io/blog/2025/agents-world-models/) 为这一提升 Agent 性能的基础要求提供了更多背景信息。
- **EleutherAI 按下重置键，Meta 的 V-JEPA 2 正式亮相！**：EleutherAI 宣布调整研究重点，因为其局部体积估计器（详见 [arXiv 上的 "Neural Redshift" 论文](https://arxiv.org/abs/2501.18812)）未能准确追踪跨激活函数的学习行为。与此同时，Meta 发布了 [V-JEPA 2，其自监督视频模型出版物](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/)，代码和数据即将发布。

# Discord: Discord 高层级摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **AI 模型组合实现巅峰性能**：一位成员详细介绍了他们最大化性能的 AI 组合：**Opus** 用于概念学习，**O3 Pro** 作为分析师，**Gemini 2.5 Pro** 用于数学。
   - 该成员指出，性能因任务而异，强调了在各自擅长领域使用每个 AI 的重要性。
- **Discord 部署新型自动化 Bot 检测器**：Discord 用户注意到 Discord 发布了一款新型 Bot 检测器，可自动标记垃圾信息。
   - 成员们指出，该 Bot 检测器是自动运行的，无需下载任何额外的 Discord 模组（mod）。
- **Perplexity Tasks 的推出引发 Comet 热议**：**Perplexity Tasks** 正在向 Pro 和 Enterprise 账户推出，用于生成特定主题的新闻，**Deepsearch** 也计划在 Tasks 中提供。
   - 一位用户表示 *这在 Comet 上会变得非常疯狂*，指的是他们[对同一技术的见解](https://video.twimg.com/amplify_video/1933215329154404352/vid/avc1/1920x1080/pbQOGV7Jenwgr2_c.mp4)。
- **Deepsearch 面临延迟，引发算力担忧**：尽管 PPLX 说法相反，但成员们报告 **Deepsearch** 已被推迟，现在他们不得不回到“房间大小的计算机”时代来获取算力。
   - 在[最近的公告](https://x.com/OpenAI/status/1933208575968752092)发布后，Discord 频道的用户仍抱有最好的期待。
- **Veo3 在 AI 视频领域超越 Gemini**：[Seedance 1.0 在 AI 文本和图像转视频领域击败了 VEO3](https://video.twimg.com/amplify_video/1933194931566243840/vid/avc1/1920x1080/HEEIOuxi8TLuVj8y.mp4)，但目前尚不清楚 Perplexity 中使用的是什么。
   - 该领域发展如此之快，以至于今天的领先者明天就可能变成落后者。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **AI Bot 被滥用！**：一位成员要求 Bot *编写一个程序来查找 G(n,k)*，其他成员认为他们*正式滥用了它*。
   - 一位成员回应说，这更多是为了*测试它而不是解决问题*。
- **Kingfall 图片泄露！**：一位用户分享了一张[据称是 Kingfall 的图片](https://cdn.discordapp.com/attachments/1340554757827461211/1382530309098049566/image.png?ex=684cceaf&is=684b7d2f&hm=7fbc452b8b5b5969993ab2493a3ba78f2558bc390095a9aef1e1c5c11742b2bd&)，被描述为*关于阳痿的寓言*。
   - 随后有报道称该漏洞已*被修复*，并重定向到了旧版本，一位成员表示*你不能再使用它了*。
- **O3 Pro 定价令用户困惑！**：成员们就 **O3 Pro** 与 **Gemini 2.5 Pro** 的价值和定价展开了辩论，引用了各种经验、基准测试和成本考虑。
   - 一些人认为 **O3 Pro** 定价较高是因为其卓越的能力，而另一些人则认为 **Gemini 2.5 Pro** 更具成本效益，甚至在数学等某些任务中表现更优。
- **LMArena 遭遇云服务中断！**：一次**云服务商故障**导致网站出现问题，可能导致**聊天历史数据丢失**。
   - 团队对造成的不便[表示歉意](link.to.apology)，并正在研究预防性解决方案。开发团队正在积极实施**预防措施**。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **免费模型限制下调**：成员们讨论了免费模型的速率限制，明确如果充值总额少于 $10，限制为 **50 次请求/天**；否则，所有免费模型共享 **1000 次/天** 的额度。
   - 该限制与输入/输出的 Token 数量无关，甚至失败的请求也会计入限制。
- **付费模型速率限制变动中**：一名用户在尝试并发运行大量请求进行数据标注时，尽管已付费仍遇到了 **429 错误**，并询问了付费模型的速率限制。
   - 一名工作人员表示，显示的 *rate_limit* 对象不准确且将被弃用，并声明付费模型实际上没有速率限制，但发现该用户在 **Qwen3 30B** 唯一的结构化输出提供商 Fireworks 上达到了 **10,000 RPM**。
- **全球性问题导致 OpenRouter 崩溃**：由于影响 **Cloudflare** 和 **Google Cloud** 等服务的大规模网络问题，OpenRouter 经历了 **全球停机**，导致广泛的服务中断和用户不满。
   - 工作人员确认他们受到了影响，但停机并非他们的过错，并提供了 [状态页面](https://status.openrouter.ai/) 和 [Downdetector](https://downdetector.com/) 的链接以获取更新；用户们幽默地推测起原因和影响，有人提到 Gemini 网站幸运地仍在运行。
- **DeepSeek 提供可疑对话**：一名用户报告 **DeepSeek 模型** 在响应过程中断断续续地切换到 **中文**，其他用户也确认了这一问题。
   - 建议包括调整 *temperature*、*top_p* 和 *top_k* 等设置，并监控哪些提供商正在提供错误的响应，建议尝试 *r1 0528* 以及 GMICloud 和 Inference.net 等提供商。
- **Requesty 前来救援**：用户简要提到了 **Requesty** 作为 OpenRouter 的替代方案，一名用户将其描述为更侧重于可靠性和性能的 **企业级基础设施解决方案**。
   - 有人指出，在 OpenRouter 因全球停机而挣扎时，Requesty 用户仍能正常使用，它被吹捧为需要稳定性的生产级工作负载的解决方案。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Opus 4 在代码循环方面优于 Sonnet 4**：成员们辩论了 **Sonnet 4** 和 **Opus 4** 在编程方面的优劣，有人指出 [**Opus** 的循环次数少于 **Sonnet**](https://www.cursor.com/docs/models/understanding-models#claude-3)。
   - 然而，**Gemini 2.5 Pro** 因具备批判性思维并能拒绝糟糕建议而受到赞赏，不像 **Sonnet** 和 **Opus** 那样“无论如何都绝对服从”。
- **Gemini 2.5 Pro 毁掉 Postgres 配置**：用户报告 **Gemini 2.5 Pro** 在配置 Postgres 时表现不佳，有时会毁掉数据库，需要 **Opus 4** 或 **O3** 来修复配置。
   - 尽管有这些缺点，它仍因批判性思维和拒绝用户错误建议的能力而受到称赞。
- **Cloudflare 事件导致 Cursor 变慢**：根据 [Cloudflare 状态](https://www.cloudflarestatus.com/)，一次 **Cloudflare** 和 **GCP** 事件导致了广泛的互联网中断，引发了 **Cursor** 用户的严重延迟和登录问题。
   - 尽管发生了停机，一些用户报告 **O3** 仍在运行，并赞扬了 **Cursor** 的迅速响应。
- **对 Cursor 移动端应用的期待激增**：社区成员对潜在的 **Cursor 移动端应用** 感到兴奋，将其与 **Replit** 类比，用于随时随地编程。
   - 讨论中涉及了 **Cursor Tab** 补全的效率，以及与 [**Copilot**](https://github.com/features/copilot) 的对比及其整体有效性。
- **Windows 上的后台 Agent 报错**：用户报告在 Windows 上运行后台 Agent 时出现 `Connection Failed` 错误，一名 Cursor 开发者正在跟踪 [Windows bugs](https://discord.com/channels/1074847526655643750/1380811765218283660)，并希望在下个版本中修复。
   - 后台 Agent 必须在 **远程环境** 中安装依赖并访问所有扩展，这意味着 Agent 需要代码存储空间。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **DeepSeek R1 量化表现出色**：成员们发现，与 **Qwen3** 相比，**DeepSeek R1** 的量化效果非常好，这引发了人们的猜测，认为这是由于 **DeepSeek R1** 是在 **bf16** 下训练的。
   - 成员们报告称，在微调新的 **DeepSeek-R1** 模型（[unsloth/DeepSeek-R1-0528-Qwen3-8B](https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B)）时出现了 *Unrecognized keys*（未识别键）警告。
- **AMD 承诺性能提升**：一篇 [Tom's Hardware 文章](https://www.tomshardware.com/pc-components/gpus/amd-announces-mi350x-and-mi355x-ai-gpus-claims-up-to-4x-generational-gain-up-to-35x-faster-inference-performance) 详细介绍称，**AMD 的 MI350X** 和 **MI355X AI GPU** 声称具有高达 **4 倍** 的代际提升和高达 **35 倍** 的推理性能加速。
   - 社区鼓励 Unsloth 团队优先支持 **AMD** 硬件。
- **Unsloth 将推出多 GPU 支持**：Unsloth 团队正在开发官方的 **multi-GPU** 支持，目前已经有大约 **5** 个不同的仓库提供 **multi-GPU** 支持。
   - 成员们链接了一个 [讨论多 GPU 支持的 Reddit 帖子](https://www.reddit.com/r/unsloth/comments/1l8mxkq/multigpu_support_how_to_make_your_unsloth/)。
- **ABBA 架构性能超越 LoRA**：一种名为 **ABBA** 的新型 **Parameter-Efficient Fine-Tuning (PEFT)** 架构，在相同的参数预算下显著优于 **LoRA** 及其主要变体，详见此 [论文](https://arxiv.org/abs/2505.14238)。
   - 代码已在 [GitHub](https://github.com/CERT-Lab/abba) 上发布，并且在 **4** 个开源 LLM（**Mistral-7B, Gemma-2 9B, LLaMA-3.2 1B/3B**）的常识和算术推理测试中，始终击败 **SoTA LoRA** 变体。
- **Fetch Image 遇到 NoneType 错误**：一位用户报告在 **Unsloth** 训练期间出现 `AttributeError: 'NoneType' object has no attribute 'startswith'` 错误，原因是 `fetch_image` 函数在 JSON 数据集的 images 字段中遇到了 `None` 值。
   - 一位成员建议确保每个 batch 要么包含全部图像和文本，要么只包含文本，或者使用 batch size 为 1，或者传递自定义的 collator。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **EleutherAI 计划退出估计器相关研究**：EleutherAI 正在转向其研究重点，此前其局部体积估计器（详见 [这篇论文](https://arxiv.org/abs/2501.18812)）未能准确追踪跨激活函数的学习行为。
   - 这一转变源于人们担心先前关于初始化简单性的工作可能比较脆弱，特别是对于具有高权重幅度的网络，正如 [EleutherAI 博客](https://blog.eleuther.ai/inductive-bias/) 和 [配套代码](https://github.com/EleutherAI/tyche) 中所述。
- **印度启发成立 AI 研究院**：AI Safety India [aisafetyindia.com](https://aisafetyindia.com/about) 于今年成立，旨在成为 AI 安全研究和讨论的中心，Discord 上至少有一位顾问。
   - 考虑到其他 AI 安全机构的存在以及成员的所在地，它的突然出现让一些人感到惊讶，人们希望它不仅仅是 *“一个死掉的网站”*。
- **Meta 发布 V-JEPA 2，验证视觉愿景**：Meta 推出了 [V-JEPA 2](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/)，这是一种自监督视频模型，并计划立即发布代码和数据。
   - 虽然一位成员称 **JEPA 的前提很疯狂**，但其他人则为 **Yann** 长期以来坚持的以无监督方式创建有用世界表征的愿景辩护。
- **ABBA 击败其他替代方案，被誉为适应性的巅峰**：一种名为 **ABBA** 的新型 **Parameter-Efficient Fine-Tuning (PEFT)** 架构，通过将更新建模为 **两个独立学习的低秩矩阵的 Hadamard 积**，其性能超越了 **LoRA**，详见 [论文](https://arxiv.org/abs/2505.14238) 和 [代码](https://github.com/CERT-Lab/abba)。
   - 成员们讨论了表达能力与秩（rank）之间的平衡，认可 **ABBA** 在两者上取得的成就以增强性能。
- **Epoch 工程提升 LLM 性能**：一位成员发现，对于小型 LLM，在第一个 **epoch** 使用 **warm-up 和线性衰减**，在第二个 **epoch** 使用 **余弦衰减** 进行训练，可以增强分类任务的性能，详见 [这篇论文](https://arxiv.org/pdf/2404.06395)。
   - 当应用这种特定的训练方法时，这些改进对较小的 LLM 尤为显著。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT Projects 获得新能力**：**ChatGPT** 中的 Projects 正在获得包括**深度研究支持 (deep research support)**、**语音模式支持 (voice mode support)** 以及**改进的记忆功能 (improved memory)** 在内的多项新功能，并正向 **Plus**、**Pro** 和 **Team 用户**推广。
   - 改进的记忆功能为 **Plus** 和 **Pro** 用户专属，而 **Canvas** 现在支持下载为 **PDF**、**docx** 或 **markdown** 格式，移动端用户现在也可以使用模型选择器。
- **飞行汽车讨论中 Apple 被指平庸**：一位用户分享了一段 [YouTube 采访](https://youtu.be/NTLk53h7u_k?si=VF8-zJZLQziFhpD_So)，视频中一位女性在讨论**飞行汽车**和**飞行出租车**时，直言 **Apple** 如今是多么平庸。
   - 用户们讨论了 **O3 Pro** 的生成时间是否被人为延长，以抑制使用并减少计算资源 (compute) 消耗，但一些人认同 *O3 Pro 优于 O3*。
- **LLMs 在简单推理任务中失败**：一篇论文显示，当图像被人工修改时 **LLMs 表现失败**，这引发了关于 LLMs 是真正具备推理能力，还是仅仅偏向于训练数据的讨论。
   - 一位用户认为 LLMs 只是在*模仿智能*，并将 LLMs 比作心理学双加工理论中的**系统 1 (System 1)**，暗示实现 **AGI** 需要的不仅仅是 LLMs。
- **明确的禁用 Token 增加泄露风险**：一位成员警告说，[列举禁用 Token (forbidden tokens)](https://owasp.org) 会放大近因偏差 (recency bias) 并增加 **LLM 泄露**的风险，且缺乏证据并不代表不存在风险，尤其是在面对涌现风险时。
   - 他们还指出，最佳实践建议将违禁内容的执行外部化，OpenAI 的指南也建议使用通用的、法律性质的描述。
- **GPT-4o 训练成本揭晓！**：一位用户估计 **GPT-4o** 的训练成本约为 **1500 万美元**，这引发了关于基于模型推理 (inferencing) 实际成本的 **Plus** 订阅盈利能力的讨论，该讨论由一段关于 **Veo 3** 的 [YouTube 视频](https://www.youtube.com/watch?v=QGwWJl7AZ6c) 链接引发。
   - 另一位成员报告称在自定义 **GPTs** 中遇到了潜在的内存泄漏 (memory bleed)，并分享了一些*强有力的证据*，表明他们看到的记忆内容并非幻觉，而是真实的内存泄漏。



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 聊天模式令用户兴奋**：成员们对 Manus 的新**聊天模式 (chat mode)** 感到兴奋，认为它是避免在简单查询上浪费额度 (credit) 的*游戏规则改变者*，并通过消除应用切换提升了用户体验。
   - 虽然有些人认为 Manus 应该专注于任务完成而非通用聊天，但一位管理员强调，这将减少关于额度浪费的投诉，因为用户无需使用 Agent 模式即可获得快速解答。
- **Veo 3 视频成本引发关注**：用户讨论了 **Veo 3 视频生成**的成本，一位成员报告称 **8 秒**视频最初收费 **300 额度**，后来增加到 **600 额度**。
   - 计算显示，一段 5 分钟的视频可能耗资 **47.50 美元**，而一部 1 小时的电影成本约为 **570 美元**，这还不包括音乐和音效的额外费用。
- **“高努力模式”现已自动启用**：成员们注意到手动选择**高努力模式 (High Effort Mode)** 的选项已被移除，系统现在会在认为必要时自动启用该模式。
   - 一位用户对**高努力模式**现在成为一个自然过程表示满意，消除了手动选择的必要。
- **额度浪费和文本处理错误困扰用户**：用户报告了导致额度损失的**文本处理**问题；一位用户因编辑器中反复出现的文本处理错误损失了 **150 额度**，而另一位用户目睹了 Manus 重复执行同一任务。
   - 一位成员建议开启新会话以缓解该问题，而另一位成员观察到该问题与幻灯片上传有关，且自聊天模式推出以来变得更加普遍。
- **AWS 停机导致 Manus 瘫痪**：由于大规模的 **AWS 停机 (AWS outage)**，Manus 平台面临故障，影响了文件上传、任务执行和通用功能。
   - YouTube、Twitch 和 Discord 的图片上传等服务也受到了影响，成员们开玩笑地猜测是不是外星人在旧金山着陆了。



---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **双 GPU 在 LM Studio 中表现翻倍**：用户确认双 GPU 提升了 **LM Studio** 的性能，其中 **32+16GB** 等配置显示出极佳的效果。
   - 一些人担心双 GPU 的负载不会太重。
- **SSD 交换引发寿命担忧**：关于使用 **mmap()** 交换的讨论引发了对过度写入可能缩短 **SSD 寿命**的警告。
   - 虽然 SSD 有写入总量 (**TBW**) 额定值，但交换操作产生的大量写入操作引起了关注。
- **Unsloth 的 Qwen 模型在推测解码中出现状况**：用户在使用 **Unsloth 的 Qwen3 模型**进行推测解码（Speculative Decoding）时遇到问题，特别是尝试在 GPU 上运行草稿模型（draft model）并在 CPU 上运行主模型时。
   - 一个 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1kftu3s/draft_model_compatible_with/) 澄清了草稿模型必须加载到 GPU 中，这表明崩溃问题与处理器选择无关。
- **LM Studio 不会自动更新软件**：用户确认当 **Hugging Face** 上有新版本可用时，**LM Studio** 不会自动更新模型。
   - 模型更新通常涉及新仓库中的新一代模型，这使得原地更新和谱系追踪变得复杂。
- **LLM 倾向于简短内容**：**LLM** 被训练得尽可能简洁，以节省计算成本并避免用户感到乏味，这可能会让那些寻求论文式总结的用户感到沮丧。
   - 有建议提出将任务拆分：先获取结构，然后针对每个要点请求内容。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **AI 剧本工具引起关注**：成员们分享了创建 **AI 剧本和电影制作工具**的资源，包括 [ACM Digital Library](https://dl.acm.org/doi/fullHtml/10.1145/3544548.3581225)、[Awesome-LLM-Long-Context-Modeling](https://github.com/Xnhyacinth/Awesome-LLM-Long-Context-Modeling) 和 [EQBench](https://eqbench.com/creative_writing.html) 的链接。
   - 一位成员提到日本最近发生的一起事件，其中 **ChatGPT** 被用于为政府委托的电影编写剧本，引发了*一些麻烦*。
- **HF API “无推理提供商”错误困扰用户**：用户报告在使用 Inference API 时，`nlpconnect/vit-gpt2-image-captioning` 等模型出现 `No Inference Provider available` 错误。
   - 一位成员建议在 [Hugging Face 设置页面](https://huggingface.co/settings/inference-providers) 和 [聊天界面](https://huggingface.co/chat/) 检查可用的模型和推理提供商。
- **AI 头像项目遇到内存限制**：一位正在构建 **AI 头像项目**的成员面临崩溃问题，原因是 **2GB 视频生成模型**超过了其 **8GB RAM**，由于 AWS GPU 不可行，正在寻求本地解决方案。
   - 建议包括探索**模型量化（quantization）、帧拆分以及运行 low-vram 模式**，并提供了一个注册 Nvidia 开发者计划积分的链接。
- **新手希望通过 `requirements.txt` 管理 Agents**：开发者要求为 **Agents 课程**提供 `requirements.txt` 文件和 **Python 版本指南**，以辅助在 Colab 之外的本地开发。
   - 一位开发者遇到了找不到 **llama-index** 的问题，随后他们发现目录名与库名相同。
- **无限制文本转视频工具出现！**：一个新的 [Unlimited Text To Video](https://huggingface.co/spaces/NihalGazi/Unlimited-Text-To-Video) 应用已发布，但分辨率较低且速度较慢。
   - 优点是视频生成是*无限制*的。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **针对 GPU 角色的并行模式提示**：准备 **GPU 工程角色**面试的成员正在寻找关于**并行编程模式**的资源，并参考了被誉为“圣经”的 **PMPP**（Programming Massively Parallel Processors）。
   - 讨论强调了需要补充视频材料来增强对这些模式的理解。
- **舍入误差拖累 Triton Kernel 性能**：一位用户通过**向上舍入到下一个 2 的幂**优化了 **conv1d kernel 性能**，另一位用户通过解决该问题将性能从 **第 5 百分位数提升到了第 95 百分位数**。
   - 另一位用户提供了他们的 **Triton kernel 代码**，展示了如何将输入拆分为块（blocks）以及通过 kernel tiling 来优化 **conv1d** 性能。
- **PTX 修饰符调整缓存策略**：一名成员讨论了为加载指令使用 **PTX 修饰符**以及**缓存淘汰策略**，目标是优化 **GEMMs** 在 **L2 cache** 上的内存布局。
   - 该成员指出在独立设置 **L1** 和 **L2** 缓存的淘汰策略时遇到问题，但得到了关于 **Blackwell 库**及其优化的建议。
- **`torch.func.functional_call` 亮相**：一名成员观察到 `torch.func.functional_call` 的集成，怀疑它是否能解决**集成 API 问题**，但注意到现在可以通过 `torch.func` 访问 `functorch`。
   - 此外，提出了一种将预训练权重加载到 `nn.Linear` 层的新方法，即通过 `nn.Linear.from_pretrained(weight=param)`，而不是目前 [VLLM 项目](https://github.com/vllm-project/vllm/pull/19265/commits/0a0b1a8e9a57d0f2f543e76c18b074544847cce4)使用的 meta device 方法。
- **FLE 旨在实现 AlphaFactorio**：成员们表示，改进 Factorio Learning Environment (**FLE**) 将有助于超智能系统理解现实世界的复杂系统，并引用了[这篇立场论文](https://arxiv.org/pdf/2502.01492)。
   - 他们概述了首要任务是使 **FLE** 可用，其次是创建一个类似 **AlphaGo** 的项目（AlphaFactorio）以优化性能；同时一名成员为一个独立的 **FLE docker 镜像和 mod** 创建了 POC 项目 ([https://github.com/MortenTobiasNielsen/fle_suggestion](https://github.com/MortenTobiasNielsen/fle_suggestion))。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **CS 学位价值受质疑**：一名成员引发了关于“计算机科学学位是否仍然有用”的辩论，在被提醒保持讨论主题之前引起了简短的讨论。
   - 讨论迅速转回技术事务。
- **SVD 测试因符号翻转受阻**：在进行 **SVD 测试**期间，一名成员报告了归因于**符号翻转（sign flip）**的失败。
   - 该问题表现为**元素不匹配**，并附带了详细的 traceback，突出了**最大绝对误差**和**相对误差**。
- **eigh() 实现引发悬赏**：在 tinygrad 中添加 **eigh()** 的复杂性导致有人建议将其设为独立悬赏，并附上了 [A_Novel_Fully_Hardware-Implemented_SVD_Solver_Based_on_Ultra-Parallel_BCV_Jacobi_Algorithm.pdf](https://cdn.discordapp.com/attachments/1070745817025106080/1382505351634616381/A_Novel_Fully_Hardware-Implemented_SVD_Solver_Based_on_Ultra-Parallel_BCV_Jacobi_Algorithm.pdf?ex=684cb771&is=684b65f1&hm=c832598efa3830d558a0f9f457a339b3b05863e26df8e9f372fdd6419a7ba60e&)。
   - 社区承认了这一挑战并表示有兴趣贡献。
- **为 tinygrad Discord 引入 LLM 聊天机器人？**：成员们探索将 **LLM 聊天机器人**（如 [getunblocked.com](https://getunblocked.com/)）与 Discord 聊天和代码库集成，以提供对用户查询的上下文感知回答。
   - 该提案涉及剥离冗余和低信号文件，并将相关上下文提供给 **LLM**，以增强其响应能力和准确性。
- **Tensor Norm 函数出现，但有注意事项**：针对一项咨询，一名成员分享了一个包含 tinygrad **tensor.norm()** 实现的 [linalg.py 文件](https://cdn.discordapp.com/attachments/1070745817025106080/1382766636364333176/linalg.py?ex=684c5948&is=684b07c8&hm=2f078256ba98c1fd605435de76fd7f16dee8dbb1ea9ca817886a44df1e9b7338&)。
   - 作者承认，虽然该函数“在 tinygrad 中 100% 可用”，但“不如 numpy 快，也不如其准确”。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **AI 音频自定义问题仍未解决**：用户反映无法为源文件中的每个主题生成独立的 AI 音频概览，因为在初始生成后 **自定义选项会消失**。
   - 变通方法包括预先准备好源文件和自定义音频指令，然后重复生成新的 Notebook 和音频。
- **动漫剪辑频道寻求订阅**：一位用户推广了他们的 **YouTube 频道** *THE OP KID*，该频道以动漫剪辑为特色，积极寻求社区订阅。
   - 该用户未提供更多细节。
- **播客达人分享合集**：一位用户分享了使用音频概览功能创建的一系列播客，涉及 **高中阅读**、**传教士与圣经人物**、**认知扭曲与心理学**、**利用 AI 破解悬案** 以及 **深刻的影视节目** 等主题。
   - 该用户分享了 **Spotify** 上每个播客的链接，并提到“How-to”播客意外地在摩洛哥登顶榜首。
- **NotebookLM 年龄限制引发讨论**：关于 **NotebookLM 的年龄限制** 展开了讨论，一位用户指出它已与 **Family Link** 集成，最低年龄要求为 **13** 岁。
   - 另一位用户认为年龄政策可能因地区而异，特别是 **America** 和 **EU** 之间。
- **MathJax 渲染扩展发布**：一位用户介绍了 **LaTeXLM**，这是一个开源的 **Chrome 扩展**，旨在为 **NotebookLM** 提供 **MathJax 渲染**，通过 [GitHub](https://github.com/hachoj/LaTeXLM) 分享。
   - 该扩展使用户能够利用本地 **Chrome 扩展** 而无需脚本。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Oscar-C 项目寻求测试者**：一名成员正在为 **oscar-c** 寻找测试者，该项目专注于 *认知架构/XAI/神经符号 AI*，并邀请感兴趣的人员私信了解更多信息。
   - 该项目旨在探索认知架构的新前沿。
- **Altman 和 Marcus 就智能问题展开争论**：成员们就 **Sam Altman** 和 **Gary Marcus** 之间关于推理和智能定义的 [帖子](https://x.com/sama/status/1932588741584957482) 展开了辩论。
   - 一位成员认为，*99% 争论这不是“真正”推理/智能等的人，甚至无法给出一个能把大多数人类也包含在内的定义*。
- **Prompt 工程师发现《人类最后的提示工程指南》**：一位成员请求获取编写 **Agent 系统提示词** 的资源，另一位成员分享了 [《人类最后的 Prompt Engineering 指南》](https://www.forwardfuture.ai/p/humanity-s-last-prompt-engineering-guide)。
   - 该指南包含编写更好提示词的实用配方和技巧。
- **自适应共振理论引起共鸣**：一位成员强调了 **Adaptive Resonance Theory (ART)** 算法的相关性，促使大家分享了一篇关于该主题的 [综述论文](https://arxiv.org/abs/1905.11437)。
   - ART 是一类试图解决稳定性-塑性困境 (stability-plasticity dilemma) 的算法。
- **世界模型：对通用 Agent 至关重要？**：一篇新 [论文](https://arxiv.org/abs/2506.01622) 认为，**通用 Agent** 必须为多步目标导向任务学习其 **环境的预测模型**，并从 Agent 的策略中提取该模型，且需要不断提高准确性以提升性能。
   - 作者的 [博客文章](https://richardcsuwandi.github.io/blog/2025/agents-world-models/) 提供了额外的背景信息，不过论文才是信息的核心。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 跃上 LeetGPU！**：**Mojo** 现在支持 [LeetGPU](https://leetgpu.com/) 等服务，提高了开发和测试的可访问性。
   - 这使得开发者能够在不同的硬件配置上利用 **Mojo** 的能力。
- **FastxReader 使用借用迭代器（Borrowing Iterators）进行映射**：一位工程师展示了在 **Mojo** 中使用 `rec.name[]: rec.seq[]` 语法的字典推导式（dict-comp）配合 **FastxReader** 的借用迭代器。
   - 这展示了 **Mojo** 在读取 fastq 文件时，如何简洁地将序列名称映射到序列。
- **Modular 文档领先于 Nightly 版本？**：工程师们发现 **Modular Docs** 与 **Mojo** 的 **nightly** 构建版本之间存在不匹配，原因是与引用（references）相关的错误。
   - 一位成员建议使用带有 `--index-url https://dl.modular.com/public/nightly/python/simple/` 的 **nightly** 版本，以与文档保持一致。
- **Mojo 避开了动态分派（Dynamic Dispatch）**：成员们剖析了 **Mojo** 类型系统中缺失动态分派、类型 Lambda 和类型族（type families）的情况。
   - 讨论涉及在列表中使用 [Variant](https://github.com/josiahls/firehose/tree/master/firehose) 作为变通方案，尽管完整的实现仍在等待中。
- **Mojo 字符串操作提速 40%！**：**Mojo** 的 **nightly** 分支中的字符串操作优化在小型字符串基准测试中显示出比 Python 快 **40%**。
   - 发布测试的工程师表示，下一个稳定版本将为任何使用他提供的字符串分割代码示例进行快速字符串操作的人带来 *“大量的性能提升”*。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **大输入下出现内存消耗异常**：用户注意到在 **flex attention** 和 **FSDP** 下，`(bs, seqlen*8)` 的输入比 `(bs*8, seqlen)` 的输入消耗更多内存，尤其是在大输入达到“临界点”时，内存使用量会迅速增加，如[此图表](https://cdn.discordapp.com/attachments/1216353675744641096/1382573185916211200/image.png?ex=684c4dde&is=684afc5e&hm=0d498abf01433cd6a078a17e983947e7ac7e0590281a6728dec8a13a5fba2776)所示。
   - 据推测 logits 可能是来源，但每秒 Token 数（tokens per second）保持稳定，如[另一张图表](https://cdn.discordapp.com/attachments/1216353675744641096/1382594289455857765/image.png?ex=684c6185&is=684b1005&hm=895b4043b33e49c5a7a89f047382cc18fc1416879e9d1c26b318c87bf345e22b)所示。
- **_grouped_mm 加速细粒度 MoE**：使用 `_grouped_mm` 显著提升了细粒度 **MoE** 的速度，使 **Qwen3-30B-A3B** 的性能在最初使用 for 循环方法慢于 **32B** 后，变得几乎与 **8B** 持平。
   - 这一优化突显了高效矩阵乘法在优化大规模模型性能中的重要性。
- **Packing 重构旨在与 Iterable Dataset 协调**：正在进行一项 Packing 重构提案，以更好地与 [iterable datasets](https://github.com/pytorch/torchtune/pull/2819) 集成，旨在支持 **DPO**、**GRPO** 和**多模态（multimodal）**应用。
   - 预计的时间表包括收集反馈并落实 iterable dataset RFC 和 packing RFC，目标是在*下周末前*完成。
- **开发者应对 Qwen3 构建器混乱！**：用户报告了一个问题，即 **Qwen3** 模型在 [#2809](https://github.com/pytorch/torchtune/pull/2809) 中使用了 **Qwen2** 构建器，导致模型构建错误。
   - 提议的解决方案包括创建专用的 **Qwen3** 组件构建器，或使用自定义注意力机制增强 **Qwen2** 构建器，目前倾向于后者以减少样板代码。
- **架构创新可能会使 Mistral 3.1 Small 复杂化**：成员们讨论了 **Mistral 3.1 Small** 潜在的架构创新，这可能会使微调实现变得复杂。
   - 虽然**多模态（multimodal）**能力并不新鲜，但它们可能会增加诸如 *devstral* 等实现的复杂性，而这些实现可能无法充分利用多模态特性。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Service Workers 运行 MCP Servers**：成员们讨论了使用 **service workers** 直接在浏览器中运行 **MCP servers**，利用 *postMessage* 和专用线程。
   - 有人指出，在浏览器中运行编译为 **wasm** 的 **MCP server** 是可能的，但可能不如直接用 JS 创建 **MCP server**。
- **Zapier MCP 连接受 500 错误困扰**：一位用户报告了通过 OAuth 连接 **Zapier MCP** 时遇到困难，理由是其 OAuth 元数据服务器和 **/token** 端点产生 500 错误。
   - 此问题已提交给服务器作者关注。
- **启动 Playwright MCP Servers**：一位成员询问是否有兴趣建立一项在云端启动 **Playwright MCP Server** 实例的服务，从而实现从任何地方（如 **n8n workflows**）进行访问。
   - 这种基于云的设置将允许从任何位置访问 **MCP Server** 端点。
- **Hyper-MCP 倡导为 MCP Servers 使用 WASM**：**Hyper-MCP** 倡导使用 **WASM** 直接在宿主机上运行 **MCP servers**。
   - 主要担忧是失去对现有 SDK 的访问权限，尽管有些人认为这并不理想。
- **fastfs-mcp 展示乐趣**：一位成员分享了 [fastfs-mcp](https://github.com/aj-geddes/fastfs-mcp)，并评论道 *正在玩这个，很有趣*。
   - 未提供更多细节。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **订单完成 Agent 填写表单**：新的 [Order Completion Agent with Artifact Editor 示例](https://t.co/oKxZxjajzZ) 使用 AI 助手在与用户交谈时填写结构化表单。
   - 这表明 AI 助手可以完成表单填写。
- **LlamaCloud 在波动后恢复**：在我们的上游基础设施提供商出现不稳定后，**LlamaCloud** 已恢复在线，正如 [LlamaIndex 状态页面](https://t.co/IdecAksHiG) 所宣布的那样。
   - 用户可以查看状态页面获取最新更新。
- **LlamaIndex 拥抱 MistralAI 的 Magistral**：**LlamaIndex** 现在支持在任何 agent 工作流中使用 @MistralAI 的 **Magistral** 推理模型；详情见[此处](https://t.co/ZsUEWMrnT4)和[此处](https://t.co/QFONzaZRk0)。
   - 这一增强功能应该会提高 agent 的推理能力。
- **Firebase 遭遇故障**：一位成员报告 **Firebase** 宕机，影响了身份验证服务，如[此帖](https://x.com/greghunkins/status/1933223568394846703?s=46)所述。
   - 另一位成员讽刺地指出，*如果 firebase 出现故障，很多东西都会宕机*，并且 [OpenRouter 也已宕机](https://news.ycombinator.com/item?id=44260810)，这是 **Firebase** 宕机的连锁反应。
- **GCP、Cloudflare 遭受波及**：一位成员报告 **GCP (Google Cloud Platform)** 宕机，**Cloudflare** 也遇到了问题。
   - 另一位成员推测这些问题归因于 **BGP (Border Gateway Protocol)** 问题。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 跳过多模态 Re-Ranker**：一位成员指出 *目前 **Cohere** 没有多模态 re-ranker*，并建议使用 **CLIP** 和 **openCLIP** 作为替代方案。
   - 另一位成员正在探索使用带有结构化输出和自定义 prompt 的 **GPT-4.1** 来实现更量身定制的方法。
- **Amotions AI 寻找技术联合创始人**：**Amotions AI** 的创始人正在寻找一位具有 AI 背景的技术联合创始人，以 *将 Amotions AI 提升到新的水平*，特别是其 [实时 AI 销售教练](https://www.amotionsinc.com/)。
   - 目标是加强其销售工具的 **AI** 能力。
- **Xarray-JAX 崛起**：一位成员正在为 **Google DeepMind** 开发 **Xarray-JAX 库**，作为 **GSoC 2025** 的一部分，并强调它是 *深度学习框架中第一个命名的张量（named tensor）实现*。
   - 他们预计这种集成将极大地造福机器学习社区，并愿意讨论潜在的应用和改进。
- **Cohere 服务受 GCP 故障影响**：Cohere 报告称，由于 [Google Cloud Platform (GCP) 事件](https://ift.tt/on1ARP0) 影响了其部分服务，截至 **2025 年 6 月 12 日下午 12:02** 出现停机。
   - 受影响的具体组件是 **Infrastructure**，更多详情可在 [Cohere 状态页面](https://ift.tt/Ens6bma) 查看。



---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 3.0 进入 Beta 阶段**：[DSPy 3.0](https://github.com/stanfordnlp/dspy/releases/tag/3.0.0b1) 已发布 **Beta** 版本，成员们正在寻求关于这些变化的全面概述。
   - 社区对新功能和改进表现出浓厚兴趣，特别是与动态替换输入字段相关的部分。
- **Agent Bricks 首次亮相**：一位成员分享了一张截图和一篇 [Databricks 博客文章](https://www.databricks.com/blog/introducing-agent-bricks) 的链接，介绍了 **Agent Bricks**。
   - 博客文章详细介绍了 **Agent Bricks** 如何增强 Databricks 环境中的 Agent 能力，但目前尚未就其在 DSPy 背景下的具体用法或影响展开进一步讨论。
- **Docstring 引用：需要 Jinja 替换**：一位成员询问如何在 DSPy 的 **docstring** 中引用 **input field**，特别希望能通过 **动态 Jinja 替换** 来增强灵活性。
   - 尽管有人指出 *docstring 只是文本*，但这一请求凸显了用户希望在 DSPy 的文档实践中获得更多动态能力的愿望。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **AgentX 峰会细节公布**：**AgentX 峰会** 邀请研究赛道的入围者参加海报展示环节或演讲，这为他们提供了通过 [峰会网站](https://rdi.berkeley.edu/events/agentic-ai-summit) 单独提交论文的机会。
   - 入围者将收到单独的峰会邀请，无需注册即可参加，但建议尽早注册以确保名额，且后续可能会有门票退款。
- **更多 AgentX 峰会信息**：一位用户询问了关于 **AgentX 峰会** 研究论文提交和入围者出席的具体细节。
   - 通过 [峰会网站](https://rdi.berkeley.edu/events/agentic-ai-summit) 进行单独提交会增加获得额外关注的机会。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **模型对成员来说太慢了**：成员们抱怨 *thinking models（推理模型）速度太慢*。
   - 讨论了为什么即使模型体积较小（如 **1GB, 2GB 或 4GB**），运行速度依然缓慢。
- **Token 数量影响性能**：模型运行缓慢可能是由于正在使用的 Token 数量过多。
   - 成员们怀疑 *Token 数量实在太多了* 可能是导致处理速度变慢的原因之一。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 发布 Wave 10，带来 UI/UX 翻新**：Windsurf 宣布发布 **Wave 10**，其特点是全新的 **UI/UX 升级**，以及针对团队和企业的新方案，详见其 [博客文章](https://windsurf.com/blog/windsurf-wave-10-ux-enterprise)。
   - 此次发布包括用于 `@-mentions` 和文件引用的新图标、与 IDE 主题匹配的 Cascade 面板代码块、Cascade 面板中支持用户输入的原生终端，以及新的对话历史 UI。
- **Windsurf 通过新集群扩展至欧盟**：Windsurf 宣布启动其 **欧盟集群（EU Cluster）**，承诺提供更快的性能并满足欧洲企业日益增长的需求，详见其 [博客文章](https://windsurf.com/blog/windsurf-wave-10-ux-enterprise)。
   - 更多细节可以在其 [YouTube 视频](https://youtu.be/UHinqQiiCI8?si=udyZDkWGg9nq7zcI) 中找到，变更日志请访问 [https://windsurf.com/changelog](https://windsurf.com/changelog)。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。

---

# Discord：各频道详细摘要与链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1382449597539946697)** (1408 条消息🔥🔥🔥): 

> `AI Combos, Power Grid Prompts, Text 2 Vid Arena, Deepsearch Delayed, Comet Browser Issues` 


- **AI 模型协作**：一位成员详细介绍了他们用于最大化性能的 AI 组合：使用 **Opus** 进行概念学习，**O3 Pro** 作为分析师，以及 **Gemini 2.5 Pro** 处理数学问题。
   - 该成员指出，性能因任务而异，并强调了在各自擅长领域使用每个 AI 的重要性。
- **Discord 发布全新反垃圾邮件机器人检测器**：Discord 用户注意到 Discord 发布了一个新的机器人检测器，可以自动标记垃圾邮件。
   - 成员们指出，该机器人检测器是自动运行的，无需下载任何额外的 Discord 模组（mod）。
- **PPLX Tasks 向 PRO 和 Enterprise 用户推出**：Perplexity Tasks 正在向 Pro 和 Enterprise 账户推出，用于生成特定主题的新闻，并计划在 Tasks 中提供 Deepsearch 功能。
   - 一位用户表示 *这在 Comet 上会变得非常疯狂*，指的是他们[对同一技术的个人见解](https://video.twimg.com/amplify_video/1933215329154404352/vid/avc1/1920x1080/pbQOGV7Jenwgr2_c.mp4)。
- **Deepresearch 供不应求？**：成员们反映，尽管 PPLX 说法相反，但 Deepsearch 已经延迟，现在他们不得不回到“房间大小的计算机”时代来获取算力（compute）。
   - 在[最近的公告](https://x.com/OpenAI/status/1933208575968752092)发布后，Discord 频道的用户仍抱有最好的期待。
- **Veo3 在视频竞技场中击败 Gemini**：[Seedance 1.0 在 AI 文生视频和图生视频领域击败了 VEO3](https://video.twimg.com/amplify_video/1933194931566243840/vid/avc1/1920x1080/HEEIOuxi8TLuVj8y.mp4)，但目前尚不清楚 Perplexity 中使用的是哪种模型。
   - 该领域发展如此之快，以至于今天的领先者明天就可能变成落后者。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1382559148482232422)** (3 条消息): 

> `RTX 4090, Windows Recall Security Flaws` 


- **Perplexity 展示 RTX 4090，首款消费级 GPU**：一位用户分享了一个关于 **RTX 4090** 的 [Perplexity 页面](https://www.perplexity.ai/page/rtx-4090-first-consumer-gpu-to-XRM0cWrDSQO5Z4PwRmQzlA)，称其为首款消费级 GPU。
- **Perplexity 揭示 Windows Recall 的安全漏洞**：一位用户发布了一个 [Perplexity 页面](https://www.perplexity.ai/page/windows-recall-security-flaws-Q5a7MAJWTn.KWGoDocbbmA)，涵盖了 **Windows Recall** 中的安全漏洞。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1382822588576694427)** (1 条消息): 

> `Sonar API Documentation, Perplexity API documentation feedback` 


- **Sonar API 文档：征集用户反馈**：Perplexity 团队正在寻求有关其 **Sonar API 文档**的反馈，并在其社区论坛中为此创建了一个帖子：[Sonar API 文档的改进](https://community.perplexity.ai/t/improvements-to-the-sonar-api-documentation/542?u=vikvang)。
   - 鼓励用户分享在使用文档时遇到的任何困难，例如*不清晰或难以找到*的内容。
- **为 API 文档反馈创建社区帖子**：已创建一个社区帖子以收集用户关于 **Sonar API 文档**的反馈，从而确定需要改进的领域。
   - 邀请用户在专用帖子中贡献他们的经验和建议。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1382434491213283408)** (999 条消息🔥🔥🔥): 

> `G(n,k) 程序, Claude ultrathink 选项, O3 Pro 基准测试, Kingfall, Titanforge` 


- **滥用 AI：请求 G(n,k) 程序**：一名成员要求机器人*编写一个查找 G(n,k) 的程序*，另一名成员称这*正式滥用了它*。
   - 一名成员回应说，这更多是为了*测试它而不是为了解决问题*。
- **Kingfall 泄露与补丁**：一位用户分享了一张[据称是 Kingfall 的图片](https://cdn.discordapp.com/attachments/1340554757827461211/1382530309098049566/image.png?ex=684cceaf&is=684b7d2f&hm=7fbc452b8b5b5969993ab2493a3ba78f2558bc390095a9aef1e1c5c11742b2bd&)，被描述为*关于阳痿的寓言*。
   - 随后，成员们提到它已被*修复（patched）*，并重定向到了旧版本，一位成员说*你不能再使用它了*。
- **OpenAI 的 O3 Pro 因推理和工具使用面临抨击**：成员们分享说 OpenAI 的 **O3 Pro** *擅长处理复杂任务*，尤其是使用网页浏览和工具时的数学问题。
   - 一些人提到了影响整体用户体验的问题和局限性，建议进行改进，如*调整推理长度*或*让模型更乐于交流*。
- **Google 的文化转变引发关注**：人们对 Google AI 开发的伦理方向表示担忧，讨论了 DeepMind CEO 的影响以及对*反杀人机器人（anti-killbots）等*的需求。
   - 一名成员指出 Google 可能有多种文化，其中一人观察到*公司一半以上的人都是新进员工，以至于文化已经重置了多次*。
- **Gemini 2.5 和 O3 Pro 的定价令用户困惑**：成员们辩论了 **O3 Pro** 与 **Gemini 2.5 Pro** 的价值和定价，引用了各种体验、基准测试和成本考量。
   - 一些人认为 **O3 Pro** 定价更高是因为其卓越的能力，而另一些人则认为 **Gemini 2.5 Pro** 更具性价比，甚至在数学等特定任务中表现更优。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1382832731314061373)** (1 条消息): 

> `云服务商故障, 数据丢失事件` 


- **LMArena 因云服务故障遭受数据丢失**：一次**云服务商故障**导致网站出现问题，可能导致**聊天历史数据丢失**。
   - 团队为带来的不便[表示歉意](link.to.apology)，并正在研究解决方案以防止未来再次发生。
- **预防措施正在进行中**：开发团队正在积极实施**预防措施**，以防范未来发生类似的云服务商相关数据丢失事件。
   - 有关具体策略和基础设施改进的细节将在可用时分享。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1382784485938036928)** (7 条消息): 

> `Cloudflare 停机, Google Cloud 故障, 互联网中断` 


- ****全网范围的中断**来袭！**：据 [Downdetector](https://downdetector.com/) 报道，发生了一次**全网范围的中断**，影响了 Cloudflare 和 Google Cloud。
   - Cloudflare [状态页面](https://www.cloudflarestatus.com/)和 [Google Cloud 状态页面](https://status.cloud.google.com/)提供了持续更新，根据[这条推文](https://x.com/OpenRouterAI/status/1933263905385500853)，当天晚些时候已恢复。
- ****Cloudflare & Google Cloud** 遭遇停机！**：**Cloudflare** 和 **Google Cloud** 由于更广泛的互联网中断经历了停机，引发了调查和状态更新。
   - 更新在 Cloudflare [状态页面](https://www.cloudflarestatus.com/)和 Google 的[状态页面](https://status.cloud.google.com/)上积极分享，用户可以在那里监控情况。


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/)** (1 条消息): 

memgrafter: 我明天会测试它，发过来吧
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1382438884180234340)** (971 条消息🔥🔥🔥): 

> `免费模型速率限制、付费模型速率限制、OpenRouter 全球故障、DeepSeek 模型与中文、Requesty 作为 OpenRouter 的替代方案` 


- **免费模型限制上限为 50 或 1000**：成员们讨论了免费模型的速率限制（rate limits），其中一位指出，如果累计充值少于 10 美元，限制为 **50 次请求/天**；否则，所有免费模型共享 **1000 次请求/天**。
   - 明确了该限制适用于请求次数，无论输入/输出的 token 数量多少，且失败的请求也计算在内。
- **Flux 中付费模型的速率限制**：一位用户尽管支付了服务费用，但在尝试并发运行大量请求以标注数据时遇到了 **429 错误**，并询问了付费模型的速率限制。
   - 一名工作人员指出，显示的 *rate_limit* 对象不准确且将被弃用，并表示付费模型实际上没有速率限制，但发现该用户在 **Qwen3 30B** 唯一的结构化输出（structured outputs）提供商 Fireworks 上达到了 **10,000 RPM**。
- **OpenRouter 因全球网络崩溃而瘫痪**：由于影响 **Cloudflare** 和 **Google Cloud** 等服务的广泛网络问题，OpenRouter 经历了**全球性停机**，导致大范围的服务中断和用户不满。
   - 工作人员确认他们受到了影响，但故障并非他们的过错，并提供了 [状态页面](https://status.openrouter.ai/) 和 [Downdetector](https://downdetector.com/) 的链接以获取更新；用户们则幽默地推测原因和影响，有人提到 Gemini 网站幸运地仍在运行。
- **DeepSeek V3 模型胡言乱语**：一位用户报告 **DeepSeek 模型**在响应过程中间歇性地切换到**中文**，其他人确认了这一问题并提出了潜在原因和解决方案。
   - 建议包括调整 *temperature*、*top_p* 和 *top_k* 等设置，并监控哪些提供商返回了损坏的响应，建议尝试 *r1 0528* 以及 GMICloud 和 Inference.net 等提供商。
- **Requesty 在动荡期间替代 Router**：用户简要提到了 **Requesty** 作为 OpenRouter 的可靠替代方案，一位用户将其描述为更偏向于关注可靠性和性能的**企业级基础设施解决方案**，而 OpenRouter 则专注于尝试新模型。
   - 有人指出，在 OpenRouter 因全球停机而挣扎时，Requesty 用户仍能正常使用，它被推崇为需要稳定性的生产环境工作负载的解决方案。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1382435189518499890)** (619 messages🔥🔥🔥): 

> `Opus vs Sonnet, Gemini 2.5 Pro fails, MCP servers, Cloudflare outage, Cursor Mobile App` 


- **Opus 4 在代码循环中胜过 Sonnet 4**：成员们讨论了 **Sonnet 4** 还是 **Opus 4** 更适合编程，一位用户表示根据他们的经验，[**Opus** 的循环次数比 **Sonnet** 更少](https://www.cursor.com/docs/models/understanding-models#claude-3)。
   - 另一位成员指出，在处理超过 120k tokens 时，**Gemini 2.5 Pro** 表现更好，但 **Cursor** 隐藏了工具调用（tool call）的错误消息。
- **Gemini 2.5 Pro 在 Postgres 配置上栽了跟头**：用户报告称 **Gemini 2.5 Pro** 在配置 Postgres 时表现糟糕，有时会导致数据库损毁，需要 **Opus 4** 或 **O3** 来修复其配置。
   - 尽管 **Gemini** 表现不佳，但它因具备批判性思维和拒绝用户错误建议的能力而受到赞赏，相比之下，**Sonnet** 和 **Opus** 则*无论如何都会盲目服从*。
- **Cloudflare 故障导致 Cursor 间歇性停机**：一场主要由 [**Cloudflare** 和 **GCP** 事件](https://www.cloudflarestatus.com/)引发的大规模网络中断，导致 **Cursor** 出现严重的延迟和登录问题。
   - 虽然大多数服务都下线了，但一些用户注意到 **O3** 仍在运行，并称赞 **Cursor** 对停机做出的迅速响应，尽管也有人滑稽地宣布“我被解雇了”。
- **Cursor Mobile App 期待值升温**：成员们对 **Cursor mobile app** 的潜力感到兴奋，将其与 **Replit** 进行类比，并推测其提供随时随地编程能力的可能性。
   - 其他人则称赞了 **Cursor Tab** 补全的效率，而 [**Copilot**](https://github.com/features/copilot) 本身的效率则受到了质疑。
- **基于规则的 AI**：社区成员分享了在 **Cursor** 中使用规则（rules）的技巧，规则可以在全局层级或项目层级进行设置。
   - 成员们分享了关于 MCP 设置的技巧，以及在利用 AI 编程时，[rules](https://docs.cursor.com/context/rules) 对于实现最大效能的重要性。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1382527015445336215)** (44 messages🔥): 

> `Background Agents, Code Storage, Privacy Mode, Windows Bugs, Non-Github Repositories` 


- **Windows 上的 Background Agents 错误**：用户报告在 Windows 上运行 background agents 时出现 `Connection Failed` 错误。
   - 一位 Cursor 开发者正在跟踪 [Windows bugs](https://discord.com/channels/1074847526655643750/1380811765218283660)，并希望在下一个版本中修复它们。
- **Agents 需要代码存储**：目前形式的 background agents 从根本上需要代码存储（code storage），以便在远程环境中执行和迭代代码。
   - 遗憾的是，目前不支持仓库级别的控制。
- **Background Agents 与 Privacy Mode**：Background agents 在隐私模式下不受支持，不过修复程序很快就会推出。
   - 即使禁用了机器级别的隐私模式，也可以[启用](https://www.cursor.com/slack-connected)账号级别的隐私模式。
- **Background Agents LSP 错误**：Background agents 应该拥有所有的 **LSP 错误**提示并能访问所有扩展，但必须在 agent 环境中安装依赖项。
   - Background agents **在远程环境中运行**。
- **更深层次的 PR 集成正在开发中**：Cursor 正在考虑为 background agents 提供[更深层次的 PR 集成](https://github.com/langfuse/langfuse-docs/blob/main/.cursor/environment.json)。
   - Background agent 会根据用户指示**修改 commits**。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1382436868896985279)** (215 条消息🔥🔥): 

> `DeepSeek R1 微调问题，Safetensors 转 AWQ，DeepSeek R1 8Q 模型微调，Aider Polygot 基准测试可信度，QwenLong-32B 模型发布` 


- **DeepSeek-R1 微调警告出现**：一位用户报告在微调新的 **DeepSeek-R1** 模型 ([unsloth/DeepSeek-R1-0528-Qwen3-8B](https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B)) 时出现了 *Unrecognized keys* 警告。
   - 建议参考 [Unsloth 文档](https://docs.unsloth.ai/get-started/installing-+-updating/updating#to-use-an-old-version-of-unsloth) 来解决该问题。
- **DeepSeek R1 的量化能力**：成员们观察到，在基准测试中，**DeepSeek R1** 相比 **Qwen3** 的量化效果非常好，这引发了关于 **Qwen** 训练和量化过程的推测。
   - 成员怀疑这是因为 DeepSeek R1 是在 **bf16** 下训练的。
- **Unsloth 的 AMD 活动对决！**：成员们分享了 [Tom's Hardware 的文章链接](https://www.tomshardware.com/pc-components/gpus/amd-announces-mi350x-and-mi355x-ai-gpus-claims-up-to-4x-generational-gain-up-to-35x-faster-inference-performance)，内容关于 **AMD 的 MI350X** 和 **MI355X AI GPU**，这些 GPU 声称具有高达 **4 倍** 的代际增益和高达 **35 倍** 的推理性能提升。
   - 社区正积极鼓励 Unsloth 团队优先支持 **AMD** 硬件。
- **RL Agent 成为新的提示 Agent？**：一位成员建议，提示 Agent 是“旧风格”，而“新风格”是 **RL Agent**！
   - 没有人表示反对。
- **多 GPU 支持即将登陆 Unsloth！**：Unsloth 团队正在开发官方的 **multi-GPU** 支持，并指出目前已经有大约 **5** 个不同的 **multi-GPU** 支持仓库。
   - 成员们链接到了一个 [讨论多 GPU 支持的 Reddit 帖子](https://www.reddit.com/r/unsloth/comments/1l8mxkq/multigpu_support_how_to_make_your_unsloth/)。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1382521253025419336)** (19 条消息🔥): 

> `Hyperbolic 定价，合成数据集，VRAM vs RAM，广告中的拼写错误` 


- **Hyperbolic 定价引起关注**：一位用户质疑了某产品的真实性和运行时间，并参考 [附带的截图](https://cdn.discordapp.com/attachments/1179039861576056922/1382521252853584004/Screenshot_20250611-204515.png?ex=684cc640&is=684b74c0&hm=dae1d40020262d2817ede7e21a27bee71172e3708db11a4633555dbcd2884054&) 评论说 *hyperbolic 很便宜*。
- **来自大型 LLM 的合成数据集非常酷**：一位用户表示他们尝试了一个产品，并将在下次需要 **来自更大型 LLM 的合成数据集** 时使用它。
   - 他们还补充说 *好的硬件相当便宜*。
- **VRAM vs RAM 引发混乱**：用户们争论广告中列出的是 **512GB RAM** 还是 **VRAM**，其中一人最初发布了一个 [鸭子主题的 GIF](https://tenor.com/view/duck-eco-gif-22827059) 表示怀疑。
   - 另一位用户指出 *在电脑上显示 GPU RAM: 80GB*。
- **怀疑广告中存在拼写错误**：一位用户推测 **广告包含拼写错误**，应该指明是 **80GB VRAM** 而不是 RAM，以及 **2TB RAM**。
   - 另一位用户反驳说，广告商只是 *懒惰，认为人们会因为是在广告 GPU 而理解那是 VRAM*。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1382439952544960712)** (103 messages🔥🔥): 

> `Unsloth 版本要求、bias 训练问题、Granite 偏差、Qwen2.5-VL-7B-Instruct 微调、LoRA 过拟合` 


- ****Unsloth 版本兼容性：Pip 的困境****：一位用户询问了旧版本 **Unsloth** 的要求，质疑 `pip install unsloth==2025.2.15` 是否足够，并寻求处理与 **transformers** 和 **PEFT** 等其他包兼容性问题的指导。
   - 他们还询问了旧版本 Unsloth 中 bias 训练开关的状态，引用了与 LoRA 训练相关的 [一个公开 issue](https://github.com/unslothai/unsloth/issues/2343)，建议对 utils 进行潜在调整。
- ****Fetch Image 失败：NoneType 噩梦****：一位用户报告在 **Unsloth** 训练期间出现 `AttributeError: 'NoneType' object has no attribute 'startswith'` 错误，原因是 `fetch_image` 函数在 JSON 数据集的 images 字段中遇到了 `None` 值。
   - 一位成员建议确保每个 batch 要么全部包含图像和文本，要么只包含文本，或者使用 batch size 为 1，或者传递自定义的 collator。
- ****Qwen2.5-VL-7B-Instruct 失败：模板纠葛****：一位用户在 vLLM v0.8.5 上部署微调后的 `unsloth/Qwen2.5-VL-7B-Instruct` 模型时遇到了 `RuntimeError`，这与多模态输入缺失或错误的 token 有关。
   - 该问题被追溯到合并后的模型中缺失 `chat_template.jinja` 文件，潜在的修复方案包括通过 `pip install --force-reinstall --no-deps git+https://github.com/unslothai/unsloth-zoo.git` 和 `pip install --force-reinstall --no-deps git+https://github.com/unslothai/unsloth.git` 升级 **unsloth-zoo** 和 **unsloth**。
- ****LoRA 损失：过拟合爆发！****：一位用户分享了显示模型在 **Unsloth** 进行 **LoRA** 微调约 700 个 epoch 后出现过拟合的图表，表现为训练损失显著下降而 eval 损失上升。
   - 建议监控 eval 损失，并在其开始攀升时停止训练，重点关注泛化能力而非最大化训练 epoch。
- ****Llama 3.2 工具调用：寻求指导****：一位用户寻求关于微调 **Llama 3.2 (3B)** 以实现 6 个自定义工具调用的建议，计划使用 **GPT-4** 生成合成示例对话。
   - 他们询问了方法指导和推荐的示例数量，并追问了是预期 zero-shot 还是 multi-shot/长对话。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/)** (1 messages): 

not_easy_to: 我正在微调 Qwen 2.5 7B（使用 Unsloth），需要一个小型法语数学数据集。
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1382488877838635189)** (5 messages): 

> `ABBA 架构、LoRA 替代方案、参数高效微调 (PEFT)` 


- **ABBA 在参数高效微调中碾压 LoRA**：一种名为 **ABBA** 的新型 **参数高效微调 (PEFT)** 架构在相同的参数预算下显著优于 **LoRA** 及其主要变体，详见本 [论文](https://arxiv.org/abs/2505.14238)。
   - 与 **LoRA** 向冻结权重添加低秩增量不同，**ABBA** 将更新建模为两个独立学习的低秩矩阵的 Hadamard 乘积。
- **ABBA 击败 SoTA LoRA 变体**：**ABBA** 在 **4** 个开源 LLM（**Mistral-7B, Gemma-2 9B, LLaMA-3.2 1B/3B**）的常识和算术推理任务上一致击败了 **SoTA LoRA** 变体。
   - 在某些情况下，它甚至优于全量微调，代码可在 [GitHub](https://github.com/CERT-Lab/abba) 上获得。


  

---

### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1382574996077023363)** (1 messages): 

> `Volume Estimator, Neural Redshift, Generalization Heuristic, AI Alignment, Inductive Bias` 


- **局部体积估计器（Local Volume Estimator）未能追踪学习行为**：一项研究更新指出，他们来自[这篇论文](https://arxiv.org/abs/2501.18812)的局部体积估计器未能追踪不同激活函数之间学习行为的差异。
   - 该估计器也未能证实 **ReLU networks** 比 **tanh networks** 更简单，导致人们对其作为泛化启发式工具的效用感到悲观。
- **EleutherAI 计划转移研究重心**：在完成最后一次关于局部体积工作（重点在于 **AI alignment** 应用）的研究更新后，EleutherAI 计划将研究重心转向其他领域。
   - 这一决定源于以下发现：先前关于初始化时简单性的研究可能比较脆弱，特别是在涉及高权重幅度的网络时；参见 [EleutherAI 的博客文章](https://blog.eleuther.ai/inductive-bias/) 和 [配套代码](https://github.com/EleutherAI/tyche)。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1382469402087395431)** (86 messages🔥🔥): 

> `AI Model Comparison Platforms, Open Science at CVPR, AI Safety India, GPT models behavior, Symbolica AI startup` 


- **AI 模型对比平台搜寻启动**：一位成员询问了在编程、翻译、文档和邮件写作等任务中对比 **AI models** 的平台；[livebench.ai](https://livebench.ai/#/hello) 被推荐为一个资源。
   - 他们提到自己是该领域的新手，并对哪些模型在不同场景下表现出色感到好奇。
- **CVPR 上的开放科学聚会**：一位成员向所有参加 **CVPR** 的人员发出邀请，讨论 **open science**；提供了 [lu.ma 链接](https://lu.ma/z1o7ncnt)。
   - 讨论还涉及到一个观察结果：研究实验室主要集中在计算机科学领域，由于资金原因以及方便开展诸如 hackathons 等活动，他们利用 Discord 和 Slack 进行交流。
- **AI Safety India 启动**：一位成员分享了 **AI Safety India** 的链接；[aisafetyindia.com](https://aisafetyindia.com/about) 于今年创建，其中一位顾问在 Discord 上。
   - 另一位成员表示，尽管了解其他 AI safety 机构且来自印度，但从未听说过它，对此感到惊讶并表达了兴趣，希望这不只是一个僵尸网站。
- **GPT 模型引发情感反应**：一位成员指出，GPT-3 和 GPT-4o 有时给出平淡的回答，但有时会突然深入进行 **emotional and structural interpretation**（情感和结构化解读）；链接了 [arxiv.org 论文](https://arxiv.org/pdf/2406.20052)。
   - 这被认为是一个已知的“问题”，解释为模型通过随机性选择下一个词，从而影响后续的词预测。
- **Symbolica AI 目标在于符号 AI**：一位成员询问了 **Symbolica AI**，这是一家总部位于伦敦、目标宏大的初创公司，这引发了关于其宗旨和潜力的讨论。
   - 另一位成员建议联系前员工（[Bruno Gavranovic](https://www.brunogavranovic.com/)）以获取见解，还有一位成员分享说，一些评论提到“工作的边界”不清晰，且目标一直在变化。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1382449128465895635)** (210 条消息🔥🔥): 

> `小模型训练 Epochs、Meta 的 V-JEPA 2 自监督视频模型、为 Google Ads 构建 AI 专家 Agent、使用 ABBA 进行参数高效微调 (PEFT)、CommonPile 数据与合成数据的作用` 


- **Epoch 工程提升 LLM 卓越性能**：一位成员分享了他们的经验，即针对分类任务将小型 LLM 训练 **2 个 epochs**，并在第一个 epoch 采用 **warm-up 和线性衰减 (linear decay)**，在第二个 epoch 采用 **余弦衰减 (cosine decay)**，可以获得更好的结果，参考了[这篇论文](https://arxiv.org/pdf/2404.06395)。
   - 使用这种训练技术时，小型 LLM 的结果可以得到显著提升。
- **Meta 发布 V-JEPA 2，远见卓识的视频验证器**：**Meta** 发布了 [V-JEPA 2](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/)，这是一个自监督视频模型，代码和数据预计很快发布。
   - 一位成员称 **JEPA 的前提很疯狂**，但该标签下的工作*非常酷*；而另一位成员则为 **Yann 的愿景**辩护，认为其自 2022 年以来始终如一，旨在以无监督的方式获取有用的世界表示。
- **ABBA 实现顶尖适应性，碾压替代方案**：一种名为 **ABBA** 的新型 **参数高效微调 (PEFT)** 架构发布，其性能显著优于 **LoRA** 及其变体，该架构将更新建模为**两个独立学习的低秩矩阵的 Hadamard 乘积** [论文](https://arxiv.org/abs/2505.14238) 和 [代码](https://github.com/CERT-Lab/abba)。
   - 成员们讨论了表达能力与秩（rank）之间的价值权衡，**ABBA** 同时实现了两者以提升性能。
- **公共资源困境：商业还是社区？**：成员们讨论了 **Institutional Books 1.0** 数据集在非商业许可下的发布，尽管其目标是为商业、学术和公共领域的模型开发创造更健康的基石，该数据集可在 [HuggingFace](https://huggingface.co/datasets/institutional/institutional-books-1.0) 上获取。
   - 讨论中提出了对许可限制性以及公司可能“搭便车”而不回馈公共资源的担忧。
- **量化探索：LLM 能用更少的信息学习吗？**：目前有一家处于隐身状态的初创公司在极端量化方面取得了进展，据称能将模型量化至 **1-2 bits** 且精度损失极小。
   - 量化效果如此之好，说明真正被吸收的信息其实很少，这引发了关于外部存储和长尾召回指针的讨论。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1382471938407928038)** (4 条消息): 

> `Knockoffs、预测器-校正器方法 (Predictor-corrector methods)、现实的零分布 (Realistic null distribution)` 


- **Knockoffs 提醒再次袭来**：一位成员提到他们大约有 4 年没见过有人提及 **knockoffs** 了，并感谢另一位成员提醒他们应该去了解一下。
   - 该成员随后详细阐述了一个人类类比，称 *“我的思考方式是，这类似于人类即使可以自我反省，也能从教练那里受益”*。
- **预测器-校正器方法 (Predictor-corrector methods) 出现**：一位成员建议在近期讨论的背景下考虑 **预测器-校正器方法**。
   - 对话暗示了专门的轻量级模型仅在需要时进行干预的可能性，这表明了计算效率的提升。
- **现实零分布 (Realistic Null Distribution) 概念浮现**：一位成员提到了每个特征的**现实零分布**，而无需使用外科手术式的“留一特征 (leave a feature out)”方法。
   - 该成员表示：*“据我所知，我还没见过在更复杂的特征上实现这一点，但它确实触及了每个特征现实零分布的概念，而无需使用外科手术式的‘留一特征’方法。”*


  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1382735844657336432)** (14 条消息🔥): 

> `EvalEval coalition, Standardized LM evaluation methods, Inspect standard, lm_eval multi-gpu progress bar` 


- **EvalEval 联盟集结评估基础设施人员**：一项协作努力正在寻找评估基础设施（eval infra）人员加入 **EvalEval 联盟**，旨在统一评估输出和训练，将其全部共享到同一个 Hub 中，并轻松地跨实验提取评估数据，文中附带了[一份表格链接](https://forms.gle/6fEmrqJkxidyKv9BA)。
- **正在考虑标准化的 LM 评估方法**：鉴于推理模型的兴起，目前有计划尝试创建**新的标准化 LM 评估方法**。
   - 一名成员提交了表格，希望能以某种身份提供帮助。
- **Inspect 标准在 LM 评估中的应用受到质疑**：一名成员询问为什么不使用 **Inspect 标准**，并链接到了 [inspect.aisi.org.uk](https://inspect.aisi.org.uk/)。
- **lm_eval 多 GPU 进度条困扰性能体验**：一名成员询问在多 GPU 设置下 **lm_eval** 的进度条是如何工作的，因为它似乎只跟踪一个 GPU 的进度。
   - 他们询问是否可以查看所有 GPU 的进度。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1382774942533288029)** (2 条消息): 

> `ChatGPT Projects, Canvas Updates, Model selector on mobile` 


- **ChatGPT Projects 获得新能力**：**ChatGPT** 中的 Projects 正在获得新功能，包括**深度研究支持（deep research support）**、**语音模式支持**以及**改进的记忆功能**。
   - 这些更新从今天开始向 **Plus**、**Pro** 和 **Team 用户**推出，其中改进的记忆功能为 **Plus** 和 **Pro** 用户专属。
- **Canvas 新增下载功能**：**Canvas** 现在支持下载，允许用户将文档导出为 **PDF**、**docx** 或 **Markdown**。
   - 当使用 **Canvas** 编写代码时，它将直接导出为相应的文件类型（例如 **.py**、**.js**、**.sql**）。
- **移动端上线模型选择功能**：移动端用户现在可以在 ChatGPT projects 中上传文件并访问模型选择器，提升了移动端体验。
   - 此次更新使移动平台与桌面版本更加一致，为随时随地的生产力提供功能对等。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1382450341051629701)** (182 条消息🔥🔥): 

> `O3 Pro performance versus O3, Limits of LLMs, Google Ads expert AI agent, Discord activity drop, GPT-4o cost to train` 


- **O3 Pro 对决 O3：生成速度之争**：用户正在争论 **O3 Pro** 的生成时间是否被人为夸大，以抑制使用并减少算力消耗。
   - 尽管如此，一些人认为 *O3 Pro 优于 O3*，尽管部分用户在 Projects 中使用 **O3 Pro 模式**时遇到失败，无法获得文档答案或思维链（chain of thoughts）。
- **苹果公司挨批：访谈中被指责平庸**：一名用户分享了一段 [YouTube 访谈](https://youtu.be/NTLk53h7u_k?si=VF8-zJZLQziFhpD_So)，其中一名女性告诉 **Apple** 他们现在是多么平庸。
   - 该链接出现在关于**飞行汽车**和**飞行出租车**的讨论中。
- **LLM 在简单推理任务中失败**：一篇论文显示，**当图像被人工修改时 LLM 会失效**，这引发了关于 LLM 是否能真正推理，还是仅仅偏向于训练数据的讨论。
   - 一名用户认为 LLM 只是在*模仿智能*，实现 **AGI** 需要的不仅仅是 LLM，并将 LLM 比作心理学双加工理论中的**系统 1（System 1）**，该系统主要基于反射。
- **OpenAI 最新更新**：用户正在讨论 [OpenAI 的新更新](https://help.openai.com/en/articles/6825453-chatgpt-release-notes)，其中包括 **Projects 中的深度研究和语音模式支持**。
   - 然而，由于 **Teams 订阅者**似乎被排除在某些功能（如改进的 Project 记忆功能）之外，用户感到失望。
- **GPT-4o 训练成本揭晓**：一名用户估计 **GPT-4o** 的训练成本约为 **1500 万美元**，引发了关于基于模型推理真实成本的 **Plus** 订阅盈利能力的讨论。
   - 这是由一段关于 **Veo 3** 的 [YouTube 视频](https://www.youtube.com/watch?v=QGwWJl7AZ6c) 链接引发的。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1382489448335278141)** (22 条消息🔥): 

> `GPT Quantization, ChatGPT 用于语言学习, 免费训练用 GPT Credits, Custom GPTs 间的 GPT Memory` 


- **GPT Quantization 辩论**：虽然一位用户怀疑某个 **GPT 模型被量化（quantized）**了，理由是它比 **o1 pro** 慢，但一名员工表示并非如此，不过这一说法尚未得到证实。
   - 该用户发现新模型 *比 o1 pro 慢得多*。
- **ChatGPT 辅助语言学习**：一位成员分享了使用 **ChatGPT** 识别、输入和输出本地化、特定方言语言的成功经验，包括 **idioms**（成语/习语）和 **cultural references**（文化引用）。
   - 他们强调需要仔细的引导和人工验证，以确保意图翻译正确，并指出不同模型的质量参差不齐。
- **寻找免费 GPT Credits**：用户讨论了如何获取用于训练的免费 **GPT credits**，其中一位提到 **OpenAI 的 academy program** 已停止。
   - 一位成员建议探索 **Gemini 的免费 API 模型**作为起点，而另一位提到了一个现已失效的 **Microsoft for Startups** 计划。
- **GPTs Memory：幻觉还是泄露？**：一位成员报告称，尽管通常认为 Memory 是不共享的，但他们在不同的 custom **GPTs** 之间遇到了潜在的 memory bleed（内存泄露）。
   - 他们发现了一些*有力证据*，表明他们看到的 Memory 并非幻觉，而是跨 **custom GPTs** 的 memory bleed。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1382437769317716102)** (26 条消息🔥): 

> `Prompt Security, LLM leakage, Forbidden tokens, Recency bias, Adversarial prompt injection` 


- **显式 Forbidden Tokens 增加 LLM Leakage 风险**：一位成员警告说，[列举 forbidden tokens](https://owasp.org) 会放大 recency bias 并增加 **LLM leakage** 的风险，而最佳实践建议将违禁内容的执行外部化。
   - 在 Prompt 中内置显式的限制性术语会增加模型意外泄露的几率，特别是由于 recency effects（近因效应）。
- **Math Config 的安全性辩论**：一位成员为其 500 行的数学配置辩护，反驳了关于 forbidden tokens 会导致灾难性泄露风险的说法，理由是他们的测试显示没有问题，即使尝试了大量的系统绕过手段。
   - 另一位成员反驳说，单个 forbidden token 就能产生灾难性风险，而且“缺乏证据并不代表证据不存在”，特别是在面对涌现风险（emergent risks）时。
- **泛化描述符提高安全性**：一位成员建议将显式的非法 token 替换为泛化的法律描述符，如 *'illegal content'*、*'restricted material'* 或 *'prohibited subjects'*，这符合 OpenAI 发布的内容指南，并降低了涌现泄露的风险。
   - 这种方法更安全，且符合当前的行业标准，在确保合规的同时避免包含有问题的词汇。
- **测试 AI 伦理需要大规模测试**：当被问及如何测试 AI 以验证其自我管理道德价值观的说法是否属实时，有人指出，*由于人类输入的多样性，你需要进行大规模测试*。
   - Streamlit 是一个很好的扩展解决方案。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1382437769317716102)** (26 条消息🔥): 

> `Forbidden tokens, LLM Leakage, Prompt Security, AI moral values` 


- **显式禁用 Token 会增加 LLM Leakage**：成员们讨论了在配置末尾列举禁用 Token 会如何放大 **recency bias**（近因偏差）并增加 **LLM leakage** 的风险。
   - 他们指出，Prompt Security 的最佳实践建议将违禁内容的执行外部化，并使用通用引用，而不是在任何用户可见或模型可访问的内存中嵌入显式 Token。
- **安全工程可最大限度地减少灾难性故障**：一位成员表示，在 Prompt Security 中，风险是乘数效应而非加法效应；单个关键缺陷的影响力超过任何复杂程度，且安全工程的标准不是“*它是否对我失效过*”，而是“*在对抗性或不可预见的条件下，它是否可能发生灾难性故障？*”
   - 他们引用了 [OpenAI 的 Prompt Engineering 指南](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api)（提示 #7）来推荐正向的替代方案。
- **测试 AI 的道德价值观需要规模化**：一位成员询问如何测试 AI 以验证其所谓的 **self-governing moral values**（自我管理道德价值观）是否属实。
   - 另一位成员表示，由于人类输入的多样性，需要进行大规模测试。
- **Shotgun Spammers 回归**：成员们提到 Shotgun Spammers（散弹枪式垃圾邮件发送者）又回来了。
   - 未提供更多细节。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1382449871583182950)** (220 条消息🔥🔥): 

> `Manus Chat Mode, Veo 3 Video Generation, High Effort Mode Removal, Context Limits, Credit Usage and Pricing` 


- **Chat Mode 引发兴奋与辩论**：成员们对 Manus 的新 **chat mode** 感到兴奋，这被视为一个 *gamechanger*（游戏规则改变者），可以避免在简单问题上浪费 credits，并通过消除在不同应用间切换的需求来提升用户体验。
   - 一位版主指出，这将有助于减少关于额度浪费的投诉，因为用户无需使用 Agent 模式即可获得快速回答，而一些人则认为 Manus 主要用于完成任务，而非通用聊天。
- **Veo 3 视频生成成本引发价格担忧**：用户讨论了 **Veo 3 视频生成** 的成本，一位成员报告今天早上 **8 秒视频** 的收费是 **300 credits**，随后价格突然变成了 **600 credits**。
   - 另一位用户计算出，一段 5 分钟的视频可能耗资 **$47.50**，而一部 1 小时的电影大约需要 **$570**，这还没算上来自其他供应商的音乐和音效的额外成本。
- **“High Effort Mode” 自动启用引发用户困惑**：成员们注意到 “High Effort Mode” 选项消失了，一位用户表示：*我一直不明白为什么最初必须手动选择 High Effort Mode，很高兴看到它变成了一个自然的过程*。
   - 官方澄清说，当系统认为有必要时，**high effort mode** 现在会自动启用，从而取消了手动开关。
- **用户报告额度浪费和文本处理错误**：多位用户报告了任务期间的 **text handling**（文本处理）问题，导致严重的额度损失；一位用户因编辑器中反复出现的文本处理错误损失了 **150 credits**，另一位用户则看到 Manus 将一个任务执行了两次。
   - 一位成员建议开启新会话以缓解此问题，而另一位成员观察到该问题似乎与幻灯片上传有关，并且自引入 chat mode 以来变得更加普遍。
- **AWS 故障导致 Manus 瘫痪**：由于大范围的 **AWS outage**（AWS 停机），Manus 平台出现了问题，影响了文件上传、任务执行和整体功能。
   - YouTube、Twitch 甚至 Discord 的图片上传等其他服务也受到了此次停机的影响，成员们幽默地思考是不是外星人降临旧金山了。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1382462424652845088)** (85 messages🔥🔥): 

> `LM Studio 中的双 GPU, SSD 寿命与交换（swapping）担忧, Qwen 模型的推测性解码, LM Studio 中的模型更新, LLM 简洁性训练` 


- **双 GPU 表现出色**：一位用户确认双 GPU 在 LM Studio 中运行良好，其中一名用户在 **32+16GB** 的配置上运行模型。
   - 有人提出，*专家模型（experts）似乎并不会带来太大的负担*。
- **数据交换会损耗 SSD**：一位用户打算使用 **mmap()** 交换，这引发了关于过度写入可能导致 SSD 损坏的警告。
   - 另一位用户反驳称，**SSD** 的寿命是按写入的 TB 数（TBW）计算的，而不是读取，但对于写入密集型交换的担忧依然存在。
- **排除 Unsloth Qwen 的推测性解码故障**：用户讨论了使用 **Unsloth 的 Qwen3 模型** 进行推测性解码（speculative decoding）时遇到的问题，特别是尝试在 GPU 上运行草稿模型（draft model）并在 CPU 上运行主模型。
   - 讨论指出，草稿模型必须加载到 GPU 中，崩溃问题可能与处理器选择无关，并分享了一个相关的 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1kftu3s/draft_model_compatible_with/) 链接。
- **LM Studio 不会自动更新**：一位用户询问当 Hugging Face 上有更新时，LM Studio 是否会自动更新模型，但已确认 **LM Studio 不会自动下载模型**。
   - 模型更新通常涉及在新仓库中生成新版本，这使得原地更新（in-place updates）变得罕见，并阻碍了谱系追踪（lineage tracking）。
- **LLM 经过简洁性训练**：一位寻求详尽、论文式摘要的用户被告知，**LLM 经过训练以保持简洁**，从而节省计算成本并避免让用户感到厌烦。
   - 建议将任务拆分：先获取结构，然后要求针对每个要点提供内容。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1382476846188200118)** (81 messages🔥🔥): 

> `CPU vs GPU 配置, 用于 LLM 的 EPYC, Strix Halo 内存, CPU 上的 DeepSeek R1, 统一内存对比` 


- **CPU 在 LLM 领域遇冷**：成员们讨论了仅使用 CPU 配置运行 LLM 的可行性，引用了一个 [YouTube 视频](https://youtu.be/qV2bgTYLSX4?si=kPp3xmv23N2Y8Rcd)，该视频展示了 **Gemma 27B** 仅达到 **8 t/s** 的糟糕性能。
   - 共识倾向于认为，由于内存带宽限制，CPU 的效率低于 GPU，特别是考虑到相对于工作站 GPU 中等效 VRAM 的成本。
- **EPYC CPU 加入竞争**：一位成员建议探索具有高内存带宽（约 **200GB/s**）的 **EPYC CPU**，并提供了一个价格为 **€4300** 的完整套装 [AliExpress 商品链接](https://www.aliexpress.com/item/1005008461588060.html?spm=a2g0n.productlist.0.0.3f91e9dfvz9NZg)。
   - 该成员推测，较旧的 **EPYC** 或 **Threadripper** 可以作为称职的 Proxmox 服务器，但承认工作站 GPU 仍然是实现最佳 LLM 性能的必然选择。
- **Strix Halo 的内存带宽**：成员们讨论了 **Strix Halo** 的内存配置，澄清其具有 **8 个板载通道** 的 LPDDR5x 内存，提供比标准桌面配置更高的带宽。
   - 尽管内存带宽有所提高，但人们仍将其与 Apple 的 RAM 系统进行比较，认为后者是目前最接近真实 HBM（高带宽内存）的选择。
- **DeepSeek R1 CPU 服务器被证明毫无意义**：一位成员分享了一个用于运行 **DeepSeek** 的服务器构建 [YouTube 视频](https://www.youtube.com/watch?v=v4810MVGhog)，但指出在几次提示后它就变得不可用了。
   - 其他人表示赞同，称该配置中仅 **CPU** 的成本就约为 **5000 澳元**，认为这对于实际的 LLM 应用来说是*毫无意义的尝试*。
- **统一内存对比非常“糟糕”**：一位成员分享了一个[对比统一内存（Unified Memory）的 YouTube 视频](https://youtu.be/Cn_nKxl8KE4?si=0-iQclmGi2UcWVNx)，但立即被评价为*非常糟糕*。
   - 评论者指出，演示者混淆了板载内存（soldered ram）和插槽内存（slotted ram），并且偏离了主题。他们还解释说，该视频未能提及 Strix Halo 及其内存性能。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1382435964395454494)** (84 条消息🔥🔥): 

> `剧本与电影制作 AI 工具，无推理提供商错误，图像转文本模型，Hugging Face Spaces 运行时错误，使用 Qwen 进行 LLM 蒸馏` 


- **使用 AI 工具创作剧本**：一位成员询问关于创建 **剧本和电影制作 AI 工具** 的事宜，另一位成员指出了一些资源，包括 [ACM Digital Library](https://dl.acm.org/doi/fullHtml/10.1145/3544548.3581225)、[Awesome-LLM-Long-Context-Modeling](https://github.com/Xnhyacinth/Awesome-LLM-Long-Context-Modeling) 和 [EQBench](https://eqbench.com/creative_writing.html)。
   - 他们还提到最近在日本发生的一起事件，**ChatGPT** 被用于为政府委托的电影编写剧本，引发了*一些麻烦*。
- **推理 API 缺失提供商的问题**：一位用户报告在使用 Inference API 时，包括 `nlpconnect/vit-gpt2-image-captioning` 在内的所有模型都出现了 `No Inference Provider available` 错误。
   - 另一位成员建议在 [Hugging Face 设置页面](https://huggingface.co/settings/inference-providers) 和 [聊天界面](https://huggingface.co/chat/) 查看可用的模型和推理提供商。
- **在 HF 上寻找图像转文本模型**：一位成员寻找可在推理中使用的 **图像转文本模型**，并被告知文本生成客户端不支持该功能，建议直接使用推理提供商或在 Hugging Face Spaces 中使用 Gradio 客户端。
   - 提到的示例包括 [FLUX.1-dev space](https://huggingface.co/spaces/black-forest-labs/FLUX.1-dev) 和 *Vikhyat 开发的 Moondream*。
- **报告 HF Spaces 中的运行时错误**：一位用户询问如何报告 Hugging Face Space 中的运行时错误，另一位用户提供了 [model-card-regulatory-check 讨论区的链接](https://huggingface.co/spaces/society-ethics/model-card-regulatory-check/discussions)。
   - 还提供了其他联系方式，例如通过电子邮件 (`website@huggingface.co`) 直接联系 Hugging Face 或提交 [GitHub issues](https://github.com/huggingface/hub-docs/issues)。
- **受限于 RAM 的 AI Avatar 项目挑战**：一位成员正在构建一个 **AI Avatar 项目**，但由于 **2GB 的视频生成模型** 超过了其 **8GB RAM** 的限制而面临崩溃，由于 *AWS GPU 不在考虑范围内*，正在寻找本地解决方案。
   - 建议包括探索 **模型量化 (model quantization)、帧拆分 (frame splitting) 以及运行 low-vram 模式**，并提供了一个链接，用于以*未注册的美国公司*身份注册 Nvidia 开发者计划以获取额度。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1382614870905851905)** (17 条消息🔥): 

> `MCP 服务器学习，AI Avatar 项目，Deep3DFaceReconstruction 与 Face-vid2vid 模型，AI Agent 课程` 


- **新手提问：MCP 服务器学习与课程顺序**：一位初学者正在寻找掌握 **MCP 服务器** 的资源，并且由于是该领域的新手，不确定应该先学习 **LLM 课程** 还是 **AI Agent 课程**。
   - 另一位成员建议直接从 **AI Agent 课程** 开始也是可以的。
- **AI Avatar 项目在有限 RAM 下的困境**：一位成员正在构建一个支持实时语音交互的 **AI Avatar** 项目，但由于 **2GB 的视频生成模型** 超过了其 **8GB RAM** 的限制而面临崩溃。
   - 由于成本原因不考虑 AWS GPU，他们正在寻找本地解决方案，如 **模型量化 (model quantization)**、**帧拆分 (frame splitting)** 或 **low-vram 模式**。这是他们目前工作的示例：[AI Avatar 演示](https://cdn.discordapp.com/attachments/898619964095860757/1382642504796737546/2025_06_11_01.12.45.mp4?ex=684c8e6d&is=684b3ced&hm=8eaf0cacb8697427b52556211ada52bb54b99f3bb5a552671b35043e5166c323)
- **Deep3DFaceReconstruction 与 Face-vid2vid 模型带来的挫败感**：一位成员分享了他们在没有充足 GPU 资源的情况下，运行 **Deep3DFaceReconstruction**、**Face-vid2vid** 和其他 **大型模型** 失败的挫败感，他们提到使用 **Stable Diffusion** 生成图像也失败了。
   - 另一位成员建议花费约 10 美元使用 **具备 H100 访问权限的 Colab** 进行实验，但原作者的目标是构建不同的东西，并正在寻找在投入 GPU 资源之前进行测试的免费方法。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1382586342856200202)** (20 条消息🔥): 

> `无限文本转视频 (Unlimited Text To Video)，LLM 探索自我意识，输入中的结构化交互，Hy-Bio Agent 对标 ChatGPT，逐个声音构建 AI 头像` 


- **无限文本转视频应用现身！**：一个新的 [Unlimited Text To Video](https://huggingface.co/spaces/NihalGazi/Unlimited-Text-To-Video) 应用已发布；不过分辨率较低，且运行速度稍慢。
   - 主要优点是视频生成是*无限的*。
- **AERIS LLM 探索自我意识！**：探索 LLM 自我意识的 **AERIS** 项目现已提供聊天框 [aeris-project.github.io/aeris-chatbox/](https://aeris-project.github.io/aeris-chatbox/)，相关论文已被 **ACL Main 2025** 接收，详见 [arxiv.org/abs/2403.13106](https://arxiv.org/abs/2403.13106)。
- **Hy-Bio Agent 实力超越 ChatGPT！**：`Hy-Bio Agent` 的输出更专注于实际有效的自然疗法，由包含**植物** ☘️ 数据的数据库驱动。
- **Smolagents 助力 Claude Code 克隆版**：一位成员使用 smolagents 创建了一个 **Claude Code 克隆版**，具有相似的界面和最常用的工具，并在 [X.com](https://x.com/niemerg/status/1932919266946203989) 上展示。
   - 进一步的工作包括更新他们的 [LLM OS agent](https://github.com/starsnatched/llmOS-Agent)，现在能更高效地利用 Linux VM，多模态支持也即将推出。
- **数字孪生 AI 聊天机器人涌现！**：**CloneMe** 是一个先进的 AI 平台，用于构建你的数字孪生——一个聊天方式像你、能记住细节并支持多平台的 AI，代码托管在 [github](https://github.com/vibheksoni/cloneme)，并可在 [huggingface](https://huggingface.co/MasaFoundation) 上使用。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1382752132221898875)** (2 条消息): 

> `模型可解释性，热力图可视化，Kaggle 数据集` 


- **热力图关注度升温**：成员们正在寻求关于视觉模型的**模型可解释性**和**热力图可视化**的见解。
   - 另一位建议探索 [Kaggle datasets](https://www.kaggle.com/datasets) 以寻找该任务的相关资源。
- **Kaggle：数据金矿**：一位成员建议 [Kaggle](https://www.kaggle/) 是寻找视觉模型数据集的*最佳选择*。
   - 该建议是针对有关视觉模型可解释性和热力图可视化经验的问题而提出的。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/)** (1 条消息): 

ut_nkezins: 我给你发了好友请求，也许我能帮上忙
  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1382446425048092736)** (21 条消息🔥): 

> `requirements.txt, llama-index 问题, 认证路径截止日期, 课程注册链接失效, Tool Calling agents 错误` 


- **开发者需要 `requirements.txt` 和 Python 版本指南**：开发者们正请求提供 `requirements.txt` 文件以及关于课程应使用哪些 **Python 版本**的指南，特别是在本地运行代码而非在 **Colab** 中运行时。
   - 一位开发者遇到了找不到 **llama-index** 模块的问题，后来发现是因为他们将目录中的文件命名为与库相同的名称。
- **认证路径截止日期临近**：刚开始学习 Agents 课程的新学生在询问，随着 **7 月 1 日**截止日期的临近，是否仍有可能进入**认证路径**。
   - 有经验的学生建议专注于核心单元并跳过可选内容以赶上截止日期，估计课程大约需要 **20 小时**。
- **注册链接失效**：用户报告课程注册链接 `https://bit.ly/hf-learn-agents` 已失效，导致无法进行新注册。
   - 该链接会重定向到一个错误页面。
- **Tool Calling Agents：调试 `FinalAnswerTool.forward()`**：一名学生在处理 **Tool Calling agents** 时遇到了 `TypeError: FinalAnswerTool.forward() missing 1 required positional argument: 'answer'` 错误。
   - 聊天中尚未提供即时解决方案。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1382504016117891092)** (4 messages): 

> `GPU Engineering Role Preparation, Parallel Programming Patterns, PMPP` 


- **并行模式准备提示**：一名成员正在准备 **GPU engineering role**，并正在寻找视频资源，以温习除“圣经”之外的**并行编程模式 (parallel programming patterns)**。
   - 另一名成员询问了关于“圣经”的情况，并被告知它是 **PMPP**。
- **GPU 岗位求职者寻求并行编程资源**：一名成员正在为 **GPU engineering role** 进行学习，寻求**并行编程模式**的视频资源，以补充 **PMPP** 的不足。
   - 另一名成员询问了关于“圣经”的情况，原帖作者澄清其指的是 **PMPP**。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1382670126574211172)** (2 messages): 

> `Conv1d performance optimization, Triton kernel optimization, LeetGPU challenge` 


- **2 的幂次影响 Conv1d Kernel 性能**：一位用户报告在 **conv1d** 挑战中性能较差（处于第 5 百分位），另一位用户建议**向上取整到下一个 2 的幂次**引入了不必要的工作量，在解决此问题后成功达到了 **95th percentile**。
- **Triton Kernel 代码片段展示输入分块和 Kernel 平铺**：一位用户提供了他们的 **Triton kernel code** 片段，显示他们正在将输入拆分为块并对 Kernel 进行平铺 (tiling)，目标是优化 **conv1d** 性能，使用掩码和累加循环加载 `BLOCK_SIZE` 大小的块。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1382763219495555344)** (5 messages): 

> `PTX modifiers, cache eviction policies, Blackwell library` 


- **Load 指令的 PTX 修饰符**：一名成员询问了关于在 load 指令中使用 **PTX modifiers** 的问题，特别是关于 [cache eviction policies](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-createpolicy)。
   - 该成员指出他们无法为 **L1** 和 **L2** 缓存独立设置淘汰策略，导致编译错误。
- **GEMMs 的缓存淘汰策略**：一名成员提到 **PTX modifiers** 对于小 batch size 的 **GEMMs** 非常有用，其中激活值 (activations) 被强制通过 **Triton** 使用 *evict-last* 策略。
   - 对方澄清说，使用 `ld.global.L1::evict_last` 并不总是被强制执行，这取决于数据布局 (data layout)。
- **Blackwell 库优化 L2 缓存内存布局**：其中一名成员被问及他创建的用于优化 **L2 cache** 内存布局的 **Blackwell library**。
   - 随后澄清是另一名成员创建了该库。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1382439477980692531)** (9 messages🔥): 

> `torch.func.functional_call, nn.Linear.from_pretrained, torch.compile and RL training, Mojo + PyTorch, torch.compile speedup` 


- **`torch.func.functional_call` 首次亮相**：一名成员注意到了 `torch.func.functional_call` 的集成，但认为它并没有解决 **integrated API problem**。
   - 有人建议现在可以通过 `torch.func` 访问 `functorch`。
- **`nn.Linear.from_pretrained` 提案浮出水面**：一名成员提议了一种更简洁的方法将预训练权重加载到 `nn.Linear` 层中，使用 `nn.Linear.from_pretrained(weight=param)`，而不是通常使用 **meta devices 的 3-4 行代码**。
   - [VLLM project](https://github.com/vllm-project/vllm/pull/19265/commits/0a0b1a8e9a57d0f2f543e76c18b074544847cce4) 目前使用的是 meta device 方法。
- **`torch.compile` 进入 RL 训练领域**：一名成员正在探索使用 `torch.compile` 来加速 **RL training framework**，该框架具有一种常见的模式，即类中的状态张量 (state tensors) 作为 self 属性存储，并由方法进行更改。
   - 另一名成员建议查看 **trace 以确认是否看到 cuda graph launch**。
- **为 PyTorch 寻求 Mojo Kernel 创意**：一名成员正在为即将到来的 **Modular Hack Weekend** 寻找有用的创意，并考虑使用 **Mojo** 为 **Torch** 编写一些 kernel。
   - 他们询问有哪些在 **Mojo** 中实现的、且希望在 **PyTorch kernel ecosystem** 中看到的缺失/理想功能。
- **`torch.compile` 免费带来速度提升**：一名成员报告称，即使在**没有进行算子融合 (fused)** 的情况下，由 `torch.compile` 生成的操作往往运行得更快。
   - 对于一个卷积 kernel，`torch.compile` 的运行时间为 **0.0020 ms**，而 **native PyTorch** 为 **1.7489 ms**，编译后的版本似乎在调用 `extern_kernels.convolution` 而不是 `aten.convolution`。


  

---

### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1382724344660103199)** (1 messages): 

> `图像分析，在任何设备上运行 AI` 


- **AI 可以在任何东西上运行，甚至是图像！**：一位用户开玩笑地评论道，*如今 AI 可以在任何东西上运行*，并附上了一张 [图片](https://cdn.discordapp.com/attachments/1215328286503075953/1382724344190337075/rn_image_picker_lib_temp_f68671eb-727c-448e-b9ff-e218ad0e04ef.jpg?ex=684cdaa5&is=684b8925&hm=3370f6c313acbf9a40221d8f5d46353ddead7c98ee96af78dbcd953bff69dd75&) 来强调这一点。
- **另一个话题示例**：这是一个占位符，用于满足至少两个话题的要求。
   - 如果有更多细节，可以在此处添加。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1382549860628041899)** (3 messages): 

> `OSDI 2025, AMD Advancing AI day` 


- **OSDI 2025 参会情况**：一位成员询问了关于 **OSDI 2025** 的参会情况。
- **AMD Advancing AI Day 聚会**：一位成员提议在 **AMD Advancing AI day** 与演讲者 <@325883680419610631> 进行聚会。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1382440997438623825)** (3 messages): 

> `ROCm 6.4.1, MI50s, gfx906, rocprofiler-sdk, aqlprofile` 


- **ROCm 6.4.1 在 MI50s 上存在问题**：一位用户报告称 **ROCm 6.4.1** 无法在 **MI50s** 上运行，并抛出了一个与 **HSA device gfx906** 仅支持单个 ISA 相关的错误。
   - 他们通过回退到 **ROCm 6.3.3** 并从源码构建 *rocprofiler-sdk* 和 *aqlprofile*，以及下载 [rocprof-trace-decoder](https://github.com/ROCm/rocprof-trace-decoder/) 解决了此问题。
- **用户使 ROCm 与 Triton 协同工作**：在解决了 ROCm 问题后，一位用户表示一切看起来都可以与 **Triton** 配合使用。
   - 他们补充说，现在只需要学习如何使用它，并附上了一张截图作为证明。


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1382546018922397727)** (5 messages): 

> `高效 Attention 变体, MLA 实现, GQA 与 GLA 基准测试, 蒸馏损失函数` 


- **高效 Attention 变体实现计划**：一位成员询问了在 [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) 中实现 **MLA** 和 **GQA** 等高效 Attention 变体的计划。
   - 另一位成员建议通过 Grouped Latent Attention 实现 **MLA decode**，并且 **GQA with GLA** 对于基准测试可能很有用，但可能需要额外的代码。
- **通过 Grouped Latent Attention 实现 MLA Decode**：一位成员确认 **MLA decode** 将通过 Grouped Latent Attention 实现，其中组数可以设置为 1。
   - 他们还提到，实现 **GQA with GLA** 对于基准测试可能很有用，尽管可能需要额外的代码。
- **提议实现蒸馏损失函数**：一位成员询问了关于实现类似 cosine_similarity 的蒸馏损失函数的情况，并表示如果尚未有人在做，他愿意承担这项工作。
   - 该成员赞同之前的观点，并链接到了 [issue 371](https://github.com/linkedin/Liger-Kernel/issues/371)，承诺将致力于此项工作。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1382715955070636102)** (4 条消息): 

> `cuBLASDx 0.4.0 Release, Ozaki Scheme for FP64, cuBLASDx Python Bindings, MathDx Package, cuBLASDx and CuTe DSL Integration` 


- **cuBLASDx 0.4.0 开启 Early Access**: NVIDIA 发布了 [cuBLASDx 0.4.0](https://docs.nvidia.com/cuda/cublasdx/) 作为 Early Access 库，可从 CUDA kernels 调用，使用 **CuTe Algebra and Tensors** 作为默认数据类型，为 GEMMs 和数据移动提供构建块。
   - 此版本优化了 MMA/LDSM 指令，生成 shared memory swizzles，选择向量化 async copy 指令，并提供 thread-local 到 global 的索引映射，旨在推理 GPU 上实现峰值性能，并即将支持 **UTCMMA/TMA**。
- **Ozaki 方案提升 FP64 性能**: NVIDIA 增加了一个 [Ozaki 方案解释示例](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/MathDx/cuBLASDx/16_dgemm_emulation)，演示了如何在不损失精度的情况下，通过 **IMMA 模拟** 将 **FP64 性能** 提升 5-6 倍。
   - 一位用户询问 **FP64 方案** 是否可以应用于其他操作，特别是稀疏 LU 求解。
- **cuBLASDx 提供 Python 绑定**: 新版本的 cuBLASDx 在 **Numba** 或 **NVIDIA Warp** (tile_matmul) 中提供了 [Python 绑定](https://docs.nvidia.com/cuda/cublasdx/python_bindings.html)。
- **MathDx 扩展软件包集**: cuBLASDx 是 MathDx 软件包的一部分，该软件包还包括 **cuSolverDx** (稠密矩阵分解)、**nvCOMPDx** (数据压缩)、**cuFFTDx** (快速傅里叶变换) 和 **cuRANDDx** (随机数生成)，全部带有 Python 绑定且可从 CUDA kernels 内部调用。
   - 所有这些库都遵循“即插即用 (it just works)”的理念。
- **用户询问 CuTe DSL 集成**: 用户正在询问支持 **cuBLASDx** 与新 **CuTe DSL** 集成的计划。
   - 此外，一位用户询问将基于 CUTLASS 的 header-only 库引入 Julia 需要做些什么。


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1382445555216814091)** (3 条消息): 

> `AMD GPU Support, Triton evals, Backward prop, Roadmap, Undergrad collaboration` 


- **开发者正在添加 AMD GPU 支持**: 开发者正积极致力于添加新功能和更多基准测试，包括 **AMD GPU 支持**、**Triton 评估**和**反向传播 (backward prop)**。
- **寻求与本科生合作**: 开发者表达了与参与该项目的本科生合作的兴趣，并询问了详细的路线图。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1382545737736519721)** (5 条消息): 

> `conv2d leaderboard, H100 results` 


- **第五名成绩刷屏 conv2d 排行榜**: 一名成员凭借两次提交在 `conv2d` 排行榜上获得 **第 5 名**，在 **H100** 上的成绩分别为 **338 ms** 和 **294 ms**。
   - 随后 ID 为 `32028` 和 `32029` 的提交在 **H100** 上取得成功，耗时分别为 **25118 ms** 和 **25124 ms**。
- **H100 佼佼者位列第四**: 最终的一次提交在 `conv2d` 排行榜上获得了 **第 4 名**，在 **H100** 上的时间为 **47.8 ms**。


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1382843564450385921)** (1 条消息): 

> `CUDA 12.9 Update 1, CC 10.3, B300` 


- **确认 CC 10.3 等于 B300**: 成员们根据 [CUDA Toolkit 发行说明](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cufft-release-12-9-update-1) 确认 **CC 10.3** 即为 **B300**。
- **强调 NVIDIA CUDA Toolkit 发行说明**: [CUDA Toolkit 发行说明](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cufft-release-12-9-update-1) 对于确认硬件规格至关重要。


  

---

### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1382462959145582623)** (43 messages🔥): 

> `Factorio 能力/性能、FLE 易用性障碍、视觉输入的有用性、FLE 接口对齐、FLE Docker 镜像与 mod` 


- **Factorio 助力超人工智能之旅**：根据[这篇立场论文](https://arxiv.org/pdf/2502.01492)，最大化 **Factorio 中的能力/性能**将有助于让超人工智能系统承担复杂现实世界系统的责任。
- **FLE 促进社区驱动项目**：首要任务是使 **Factorio Learning Environment (FLE)** 足够好用，以促进有机且多样化的社区驱动项目。
   - 第二个任务是启动或共同领导一个旗舰级类 AlphaGo 项目 (**AlphaFactorio**)，专注于最大化性能并提取可迁移的算法和/或工程见解。
- **解决 FLE 易用性障碍**：**FLE** 中的易用性障碍（如 **containers**、**pip 安装/设置**以及**环境接口**）阻碍了在推理、规划、记忆、多模态和多 Agent 协作方面的实验。
   - 成员指出，视觉观测在实践中似乎没有太大帮助，这使得理解这一问题成为 Option 1 项目的一个有效科学问题。
- **使 FLE 接口符合人类开发人员标准**：目标是调整 **FLE 接口**，使其更接近普通人类开发人员认为好用的标准，而不是针对特定模型进行优化。
   - 成员还表示，他不认为 LLM 能理解普通开发人员无法理解的接口，而如果接口变得过时、无效或冗余，才被认为是有影响或有意义的。
- **新的 FLE Docker 镜像和 Mod POC 项目出现**：一位成员创建了一个独立的 **FLE Docker 镜像和 mod** 的 POC 项目 ([https://github.com/MortenTobiasNielsen/fle_suggestion](https://github.com/MortenTobiasNielsen/fle_suggestion))，并征求反馈。
   - 他在将其集成到 FLE 主代码库时遇到了问题，因此目前作为一个独立项目存在。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1382750823519617226)** (44 messages🔥): 

> `AMD 会议聚会、AMD Advancing AI 标牌、Workshop 202、炉边谈话、官方照片链接` 


- **Discord 成员部署至 AMD 会议**：包括 Mark、Seb、GnSight 和 az 在内的几位成员参加了 **AMD 会议**，并在不同地点协调聚会，包括在 *“AMD advancing AI”* 标牌前以及茶歇期间。
   - 他们还计划在 **Workshop 202 (Room 212C-D)** 和午餐区见面，尽管有面试安排且现场拥挤，仍尝试同步行程。
- **炉边谈话落空，照片收尾**：一位成员在**炉边谈话**寻找其他人，结果发现 *“这里没人😭”*，后来澄清他们在房间后排。
   - 与会者还在颁奖典礼后寻找活动的**官方照片链接**。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1382443885959057548)** (4 messages): 

> `CUTLASS Matmul 优化、EVT API Epilogues、融合 LoRA 层` 


- **使用 CUTLASS 进行优化 Matmul**：一位成员询问如何使用 **CUTLASS** 优化 matmul 操作，旨在将多个操作链式结合，例如在 matmul 之后执行 **Chebyshev**，并在将矩阵写入全局内存之前执行 partial max ([issue #2393](https://github.com/NVIDIA/cutlass/issues/2393))。
   - CUTLASS 示例接受 **m, n, k** 作为参数，并对 kernel 进行基准测试以报告达到的 flop/sec。
- **探索 EVT API Epilogues**：一位成员建议使用 **EVT API** 来表达某些 epilogues，并链接到了 [Colfax International 研究页面](https://research.colfax-intl.com/epilogue_visitor_tree/)。
   - 然而，有人指出 **EVT** 要求使用一组有限的预定义操作来表达 epilogues，这限制了任意 CUDA 代码的融合。
- **将 LoRA 层融合到 FP4 Matmul**：一位成员提出了关于编写无法用 **EVT** 表达的融合 epilogue 的最佳方法，具体是想将 **LoRA 层**融合到 **FP4 matmul** 的末尾。


  

---

### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1382482782814470314)** (2 messages): 

> `j4orz.ai, picograd, picoc, CUDA C extension` 


- **Zero to Hero 项目在数学与编译器方面取得进展**：[j4orz.ai](https://j4orz.ai/zero-to-hero/) 的 "Zero to Hero" 项目已完成 **Appendix A** 的大部分内容，涵盖了各种数学概念。
   - 目前正开始 **Appendix B** 的工作，重点是实现一个 **C compiler**。
- **构想用于深度学习的 CUDA C 扩展**：计划创建一个基础的 **CUDA C extension**，灵感来自 [picograd](https://github.com/j4orz/picograd) 和 [picoc](https://github.com/j4orz/picoc) 代码库。
   - 该扩展旨在弥合传统优化编译器与深度学习编译器之间的差距。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1382447660916805633)** (6 messages): 

> `Usefulness of CS Degree, SVD Test Failure` 


- **CS 学位还有用吗？**：一位成员询问 *CS 学位是否仍然有用？*
   - 另一位成员回答说 *“如果你不得不问，那它就没用”*，并提醒注意不要偏离主题。
- **SVD 测试因符号翻转失败**：一位成员报告在运行 **SVD tests** 时，由于 **sign flip**（符号翻转）导致失败。
   - 他们分享了详细的 traceback，显示了 **mismatched elements**（不匹配元素）以及最大绝对/相对误差。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1382459037635051570)** (55 messages🔥🔥): 

> `eigh() bounty, Tensor.norm(), LLM Discord Chatbot, tinygrad vs numpy accuracy, QR algorithm discrepancies` 


- **求则得之，eigh() 需要悬赏**：在讨论将 **eigh()** 添加到 tinygrad 后，一位成员建议由于其复杂性，应该为其设立专门的 **bounty**（悬赏），并附上了 [A_Novel_Fully_Hardware-Implemented_SVD_Solver_Based_on_Ultra-Parallel_BCV_Jacobi_Algorithm.pdf](https://cdn.discordapp.com/attachments/1070745817025106080/1382505351634616381/A_Novel_Fully_Hardware-Implemented_SVD_Solver_Based_on_Ultra-Parallel_BCV_Jacobi_Algorithm.pdf?ex=684cb771&is=684b65f1&hm=c832598efa3830d558a0f9f457a339b3b05863e26df8e9f372fdd6419a7ba60e&)。
- **LLM 回答 tinygrad 问题！**：成员们讨论了将 **LLM chatbot**（如 [getunblocked.com](https://getunblocked.com/)）与 Discord 聊天和代码库集成，通过链接到特定对话来回答问题。
   - 一位成员建议剔除体积庞大且信号低的文件，将其余部分作为 **input context** 输入给 **LLM**。
- **Tensor Norm 实现浮出水面**：一位成员询问 tinygrad 中是否存在 **tensor.norm()**，并附带了一个包含 norm 函数的 [linalg.py 文件](https://cdn.discordapp.com/attachments/1070745817025106080/1382766636364333176/linalg.py?ex=684c5948&is=684b07c8&hm=2f078256ba98c1fd605435de76fd7f16dee8dbb1ea9ca817886a44df1e9b7338&)。
   - 作者承认它 *在 tinygrad 中 100% 可运行，只是不像 numpy 那么快或那么准确*。
- **Numpy vs Tinygrad 精度差异**：一位成员强调了 **numpy** 和 **tinygrad** 在浮点数 **matmuls** 上的 **accuracy discrepancies**（精度差异），特别指出输出矩阵左下角值的不同。
   - 成员们将其归因于使用了不同的算法以及 **floating-point operations** 的复杂性，编译器和机器处理边缘情况及优化的方式各不相同。
- **QR 差异**：一位成员分享了他在实现 **QR algorithm** 时的困扰，指出与 numpy 的 LAPACK 包相比，**Gram-Schmidt process** 和 **Householder Reflections** 都存在差异问题。
   - 他表示 *将跳过之前对话中提到的针对 eigh 对称矩阵的 jacobian-method 实现。*


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1382438964291698840)** (16 条消息🔥): 

> `AI 音频概览自定义，YouTube 频道推广，播客汇编` 


- ****AI 音频自定义功能仍难以捉摸****：一位用户询问如何为源文件中的每个主题生成单独的 AI 音频概览，但另一位用户澄清说，**自定义选项在初始生成后就会消失**。
   - 该用户建议，最佳方法是准备好源文件和自定义音频指令，然后生成一个新的 Notebook 和音频，重复此过程直到达到理想效果。
- ****动漫剪辑频道寻求订阅****：一位用户推广了名为 *THE OP KID* 的 **YouTube 频道**，该频道以动漫剪辑为特色，寻求订阅。
- ****播客达人分享案例展示****：一位用户分享了一系列使用音频概览创建的播客，其中包括一个关注**高中阅读**的播客，另一个关于**传教士和圣经人物**，第三个关于**认知扭曲和心理学**，第四个关于**利用 AI 破解悬案**，最后一个则深入探讨**有深度的电视或电影节目**。
   - 该用户还在 Spotify 上分享了每个播客的链接并邀请合作，并提到其“How-to”播客出人意料地在摩洛哥排名第一。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1382435670940717157)** (43 条消息🔥): 

> `NotebookLM 年龄限制，NotebookLM 功能请求，音频概览问题，图像作为来源` 


- ****NotebookLM 年龄限制引发讨论****：用户讨论了 **NotebookLM** 是否有年龄限制，一位用户指出它已与 **Family Link** 集成，最低年龄限制为 **13** 岁。
   - 另一位用户提到，年龄政策可能因地区而异，特别是**美国**和**欧盟**之间。
- ****NotebookLM 缺失“刷新所有来源”功能****：一位用户询问是否有办法刷新 Notebook 中的所有来源，但被告知该功能目前不可用。
   - *目前，你必须手动刷新每一个来源*。
- ****MathJax 渲染扩展程序发布****：一位用户创建了一个名为 **LaTeXLM** 的开源 **Chrome 扩展程序**，用于在 **NotebookLM** 上进行 **MathJax 渲染**，并分享了 [GitHub 链接](https://github.com/hachoj/LaTeXLM)。
   - 该扩展程序允许用户启用本地 **Chrome 扩展程序**，而无需脚本。
- ****音频概览在处理方程式时遇到困难****：几位用户报告说，**NotebookLM 的音频概览**在读取方程式时表现不佳。
   - 还有人询问*它是使用什么来生成音频的*。
- ****对 Excel 文件和 Google Sheets 支持的疑问****：一位用户询问 **NotebookLM** 是否计划支持 **Excel 文件**或 **Google Sheets**，因为他们没看到相关支持或在 Roadmap 上。
   - 一位用户指出目前*没有公开的 Roadmap*，并建议在功能请求频道中提出建议。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1382440797882024086)** (28 条消息🔥): 

> `oscar-c 项目，Sam Altman 对阵 Gary Marcus，Agent 的 System Prompts，自适应共振理论 (ART)` 


- ****Oscar-C** 项目寻求测试者**：一位成员一直试图让人们关注他们名为 **'oscar-c'** 的项目，该项目涉及*认知架构/XAI/神经符号 AI*。
   - 他们邀请对这个酷项目感兴趣的人私信他们以获取更多信息。
- **Altman 与 Marcus 的争论**：成员们讨论了 **Sam Altman** 和 **Gary Marcus** 之间的一篇 [帖子](https://x.com/sama/status/1932588741584957482)。
   - 一位成员表示，*99% 争论这不是“真正”推理/智能等的人，甚至无法以一种包含大多数人类的方式来定义它*。
- **Prompt Engineering 指南出现**：一位成员询问是否有*编写 **Agent 的 System Prompts** 时非常有用的资源*。
   - 另一位成员分享了 [Humanity's Last Prompt Engineering Guide](https://www.forwardfuture.ai/p/humanity-s-last-prompt-engineering-guide) 的链接。
- **成员讨论自适应共振理论**：一位成员提到了 **自适应共振理论 (ART)** 类算法的相关性。
   - 另一位成员分享了一篇关于该主题的 [综述论文](https://arxiv.org/abs/1905.11437)。


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1382438772813201489)** (26 messages🔥): 

> `World Models, Energy Based Models, Active Inference, Predictive Coding` 


- **World Models Necessary for General Agents?**: 一篇新 [论文](https://arxiv.org/abs/2506.01622) 认为，能够执行多步目标导向任务的 **general agents** 必须学习其 **environment** 的 **predictive model**，该模型可以从 **agent** 的 **policy** 中提取，并且需要不断提高准确性以提升性能。
   - 作者的 [博客文章](https://richardcsuwandi.github.io/blog/2025/agents-world-models/) 和论文本身可能不是了解作者研究项目中数学/计算方面的最佳入门资料。
- **Energy Based Models: LeCun's Pet Project Explored?**: 社区对理解 **energy-based models** 的热度表现出兴趣，特别是由于 **LeCun** 的频繁提及，并请求提供优质的入门资源。
   - 有人建议，这种兴趣更多在于社区以及相关的概念，如 **Active Inference** 和 **Predictive Coding**，而不是模型本身。
- **Predictive Coding as Gradient Descent Alternative**: **Predictive coding** 被描述为一种局部 **gradient descent** 操作，通过上游预测和下游误差之间的“压力”导出梯度，提供了 **backpropagation** 的替代方案。
   - [这篇论文](https://arxiv.org/abs/2407.04117) 被推荐为该领域最佳的入门、易读的综述，是一篇极佳的读物。
- **Active Inference talk incoming!**: 成员们讨论了关于 **Active Inference** 讲座的可能性，强调了其有趣的公式化表达以及与 **Predictive Coding** 的联系。
   - 一名成员自愿在下周主持一场关于 **Active Inference** 的讲座。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1382442340744040449)** (4 messages): 

> `Mistral Compute, New video model` 


- **Mistral Enters the Compute Arena**: Mistral AI 宣布推出 **Mistral Compute**，旨在民主化 AI 基础设施，并为所有人提供工具和环境，不再局限于构建开放模型，详见其 [博客文章](https://mistral.ai/news/mistral-compute)。
- **Potential Veo3 Competitor Emerges**: 一个新的视频模型被预告，可能与 **Veo3** 竞争，一张 [图片](https://cdn.discordapp.com/attachments/853983317044756510/1382713167863611503/20250612_142749.jpg?ex=684cd03c&is=684b7ebc&hm=cdc33b41196f26909cf60ef8d205b9c583e1d5578b96bacdd65379f4a638059a&) 暗示了其 *巨大的升级*，尽管目前缺乏声音。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1382449356749279305)** (36 messages🔥): 

> `Mojo on LeetGPU, FastxReader in Mojo, Modular Docs issues with nightly, Dynamic Dispatch/Type Lambdas in Mojo, String Performance Improvements in Mojo` 


- ****Mojo** Now Available on LeetGPU!**: **Mojo** 现在已支持 [LeetGPU](https://leetgpu.com/) 等服务。
   - 这使得 **Mojo** 在各种硬件配置上的开发和测试变得更加容易。
- ****FastxReader** iterator usage in Mojo**: 一位成员分享了在 **Mojo** 中使用 **FastxReader** 的示例，通过 `rec.name[]: rec.seq[]` 语法在输入文件上使用借用迭代器（borrowing iterator）进行字典推导。
   - 该示例代码展示了在读取 fastq 文件时，如何使用 **Mojo** 将序列名称映射到序列，突出了该语言简洁的语法。
- **Modular Docs Outdated for Nightly Builds**: 一位成员遇到了与引用相关的错误，表明文档与 **Mojo** 的 **nightly** 版本之间可能存在不匹配。
   - 文档似乎超前于 **nightly** 的更改，另一位成员建议使用 **nightly** 版本以与文档保持一致，使用 `--index-url https://dl.modular.com/public/nightly/python/simple/`。
- **Dynamic Dispatch and Type System Limitations in Mojo**: 成员们讨论了 **Mojo** 类型系统目前的局限性，指出了 **dynamic dispatch**、**type lambdas** 和 **type families** 的缺失。
   - 一位成员建议在列表中使用 [Variant](https://github.com/josiahls/firehose/tree/master/firehose) 作为变通方案来处理此问题，但这些功能的完整实现尚未可用。
- **Mojo String Operations Gain 40% Speed Boost**: **nightly** 分支中的优化使得 **Mojo** 在小型字符串基准测试中比 Python 提升了 **40%** 的性能。
   - 代码示例展示了如何分割字符串，发布者指出，对于任何进行快速字符串操作的人来说，下一个稳定版本将看到 *“大量的性能提升”*。


  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1382523524274454559)** (14 messages🔥): 

> `Memory Usage, Flex Attention, FSDP, TP, Loss Parallel` 


- **Batch 与 Sequence Length 导致的内存异常**：用户观察到，在使用 **flex attention** 和 **FSDP** 时，`(bs, seqlen*8)` 的输入（例如 `(1, 64k)`）比 `(bs*8, seqlen)` 的输入（例如 `(8, 8k)`）占用更多内存，尽管线性层在理论上应该是等效的。
- **大输入触发内存激增**：内存占用的增加似乎只在**极大输入**时发生，在达到某个“临界点”后峰值内存会迅速跳升，这可能是由于内存的分块分配（allocation in blocks）导致的。
- **Loss Parallel 揭示内存节省问题**：**Loss Parallel 的内存节省**似乎让这个问题暴露了出来，因为如果没有它，内存占用会因另一个瓶颈而保持恒定；用户附带了一张[图表](https://cdn.discordapp.com/attachments/1216353675744641096/1382573185916211200/image.png?ex=684c4dde&is=684afc5e&hm=0d498abf01433cd6a078a17e983947e7ac7e0590281a6728dec8a13a5fba2776)展示了这种激增。
   - 有假设认为 Logits 可能是大规模内存分配的来源，但正如另一张[图表](https://cdn.discordapp.com/attachments/1216353675744641096/1382594289455857765/image.png?ex=684c6185&is=684b1005&hm=895b4043b33e49c5a7a89f047382cc18fc1416879e9d1c26b318c87bf345e22b)所示，每秒 Token 数（tokens per second）保持稳定。
- **_grouped_mm 加速细粒度 MoE**：使用 `_grouped_mm` 显著提升了**细粒度 MoE** 的速度。在采用 for 循环实现时，**Qwen3-30B-A3B** 比 **32B** 还慢，优化后其速度已接近 **8B**。


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1382519865033625620)** (8 messages🔥): 

> `packing refactor, iterable datasets, contributing to torchtune, qwen3 and qwen2 builders` 


- **针对 Iterable Datasets 提出 Packing 重构**：一名成员分享了关于 Packing 重构的提案，以适配 [iterable datasets](https://github.com/pytorch/torchtune/pull/2819)，旨在支持 **DPO**、**GRPO** 和**多模态（multimodal）**应用的 Packing。
   - 时间表包括收集反馈、落地 iterable dataset RFC，然后落地 packing RFC，预计在*下周末前*完成。
- **TorchTune 贡献指南**：一位成员表达了对仓库贡献的兴趣，随后被引导至标记有 *"Community help wanted"* 的 Issue，这些 Issue 提供了明确的操作项和说明。
   - 提出该建议是因为该用户表示*他们已经在旧的 fork 仓库上开发了一段时间*。
- **Qwen3 错误地使用了 Qwen2 构建器！**：一名成员报告了 [#2809](https://github.com/pytorch/torchtune/pull/2809) 中的一个问题，即 **Qwen3** 使用了 **Qwen2** 的构建器（builders）。
   - 为了解决这个问题，他们提议要么创建独立的 **Qwen3** 组件构建器，要么在 **Qwen2** 构建器中增加传递自定义 Attention 的能力，为了避免额外的样板代码（boilerplate），他们更倾向于后者。


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1382925426325852281)** (6 messages): 

> `Mistral 3.1 Small, Architectural Novelties, Multimodal Support, Devstral` 


- **Mistral 3.1 是否有新颖的架构细微差别？**：一名成员询问 **Mistral 3.1 Small** 是否存在可能增加微调（fine-tuning）实现难度的架构创新。
   - 另一名成员指出，虽然**多模态（multimodal）**能力并非新鲜事，但它可能会引入复杂性，特别是考虑到可能并不严重依赖多模态特性的 *devstral*。
- **多模态对 Devstral 的影响**：暗示多模态能力本身不是新事物，但取决于实现是否必须支持多模态。
   - *Devstral* 的使用场景可能不需要多模态组件。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1382434998908485703)** (18 messages🔥): 

> `Service Workers, MCP and Zapier, Playwright MCP Server, Hyper-MCP WASM` 


- ****Service Workers** 讨论**: 成员们讨论了使用 **service workers** 直接在浏览器中运行 **MCP servers**，并利用 *postMessage* 和专用线程。
   - 有人指出，虽然可行，但在浏览器中运行编译为 **wasm** 的 **MCP server** 并通过 service worker 提供流式 HTTP 服务，其效果可能不如直接用 JS 创建 **MCP server**。
- **Zapier MCP 连接困扰**: 一位用户报告在通过 OAuth 连接 **Zapier MCP** 时遇到困难，理由是其 OAuth 元数据服务器和 **/token** 端点产生了 500 错误。
   - 此问题已提交给服务器作者关注。
- **在云端启动 Playwright MCP Servers**: 一位成员询问是否有人对在云端启动 **Playwright MCP Server** 实例的服务感兴趣，从而实现从任何地方（如 **n8n workflows**）进行访问。
   - 这种基于云的设置将允许从任何位置访问 **MCP Server** 端点。
- **Hyper-MCP 推动 WASM 用于 MCP servers**: **Hyper-MCP** 提倡使用 **WASM** 直接在宿主机上运行 **MCP servers**，尽管有人认为这并不理想。
   - 主要担忧是会失去对现有 SDK 的访问权限。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/)** (1 messages): 

whoateit: 正在玩这个。
https://github.com/aj-geddes/fastfs-mcp
  

---


### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1382732461091590265)** (1 messages): 

> `Office Hour Reminder` 


- **Office Hours 即将开始！**: 提醒 Office Hour 将在 15 分钟后开始：[discord.com/events](https://discord.com/events/1059199217496772688/1379510205687140412)。
   - 不要错过这个**宝贵的机会**！
- **Office Hours 提醒 #2**: 再次提醒 Office Hour 即将开始，届时将回答相关问题。
   - 请准备好提问并积极参与。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1382751870141071402)** (3 messages): 

> `Order Completion Agent, LlamaCloud Stability, MistralAI Magistral` 


- **订单完成 Agent 填写表单**: 一个 AI 助手在与用户交谈时，使用新的 [Order Completion Agent with Artifact Editor 示例](https://t.co/oKxZxjajzZ) 填写结构化表单。
- **LlamaCloud 在基础设施故障后恢复**: 在上游基础设施提供商出现不稳定后，**LlamaCloud** 现已恢复运行。
   - 查看 [LlamaIndex 状态页面](https://t.co/IdecAksHiG) 获取最新更新。
- **LlamaIndex 拥抱 MistralAI 的 Magistral**: **LlamaIndex** 现在在任何 Agent 工作流中都支持 @MistralAI 的 **Magistral** 推理模型。
   - 深入了解[此处](https://t.co/ZsUEWMrnT4)和[此处](https://t.co/QFONzaZRk0)的详情，查看 **Magistral** 如何增强 Agent 的推理能力。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1382784159533105294)** (14 messages🔥): 

> `Firebase outage, OpenRouter Down, Cloudflare Issues, GCP is down, BGP problems` 


- **Firebase 离线！**: 一位成员报告 **Firebase** 宕机，影响了身份验证服务，并指出 *Twitter 的消息比 Firebase 状态页更快*，[此贴](https://x.com/greghunkins/status/1933223568394846703?s=46) 证明了这一点。
   - 另一位成员幽默地用 *sadge* 表达了他们的沮丧。
- **OpenRouter 受到 Firebase 影响！**: 作为 **Firebase** 停机的连锁反应，一位成员报告 [OpenRouter 也已宕机](https://news.ycombinator.com/item?id=44260810)。
   - 一位成员讽刺地指出，*如果 Firebase 出现故障，很多东西都会宕机*。
- **GCP 和 Cloudflare 受挫！**: 一位成员报告 **GCP (Google Cloud Platform)** 宕机，**Cloudflare** 也遇到了问题。
   - 另一位成员推测这些问题归因于 **BGP (Border Gateway Protocol)** 问题。


  

---

### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1382437048702603379)** (10 messages🔥): 

> `Multi-Model Re-Ranker, Amotions AI, Xarray-JAX library` 


- **Cohere 缺少多模态 Re-Ranker**：一位成员表示，*目前 **Cohere** 还没有多模态 Re-Ranker*，并建议使用 **CLIP** 和 **openCLIP** 作为替代方案。
   - 另一位成员则考虑改用带有结构化输出和自定义 Prompt 的 **GPT-4.1**。
- **Amotions AI 寻找技术联合创始人**：**Amotions AI** 的创始人正在寻找一位具有 AI 背景的技术联合创始人，以*将 Amotions AI 提升到新的水平*，特别是其 [实时 AI 销售教练](https://www.amotionsinc.com/)。
- **Xarray-JAX 库开发**：一位成员正作为 **GSoC 2025** 的一部分为 **Google DeepMind** 构建 **Xarray-JAX 库**，并指出这*实际上是深度学习框架中第一个命名张量 (named tensor) 的实现*。
   - 他们认为这种集成将对 *机器学习 (machine learning) 社区非常有用*，并欢迎对此进行讨论。


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1382815899920109780)** (1 messages): 

> `Reranking profiles` 


- **重排序配置文件规格分享**：一位成员询问了关于重排序 (Reranking) 配置文件的信息，包括所使用的**文档数量、每个文档的 Token 数以及查询 Token 数**。
   - 另一位成员做出了回应，分享了他们的配置包括**数十个文档**，每个文档约 **100 个 Token**，每次查询约 **20 个查询 Token**。
- **重排序配置文件规格的澄清**：询问重排序配置文件问题的成员确认分享的规格非常有帮助。
   - 这验证了所分享信息的准确性，并确认其对寻求重排序配置指导的其他成员很有用。


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1382924505923719269)** (1 messages): 

> `Introductions, Company/Industry/University, Tech/Tools, Community Goals` 


- **Discord 欢迎 Cohere 社区新成员**：Cohere 社区 Discord 服务器欢迎新成员，并鼓励他们进行自我介绍。
   - 系统提示新成员分享他们的**公司/行业/大学所属机构**、当前项目、喜爱的**技术/工具**以及加入社区的目标。
- **与社区成员分享你的背景**：鼓励每位新成员分享他们来自哪家公司、行业或大学。
   - 这旨在促进具有相似背景和兴趣的成员之间更轻松地建立联系。


  

---


### **Cohere ▷ #[🧭-status-feed](https://discord.com/channels/954421988141711382/1346652044181897307/1382824254009245916)** (1 messages): 

> `GCP Outage, Infrastructure Degradation` 


- **GCP 故障影响 Cohere**：Cohere 报告称，由于 [Google Cloud Platform (GCP) 事件](https://ift.tt/on1ARP0)，他们正经历停机，这可能会影响其部分服务。
   - 截至 **2025 年 6 月 12 日中午 12:02**，团队正在积极监控情况。
- **基础设施遭受挫折**：受影响的具体组件被确定为**基础设施 (Infrastructure)**，其性能正在下降。
   - 更多详情可在 [Cohere 状态页面](https://ift.tt/Ens6bma)查看。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1382452337561567455)** (9 messages🔥): 

> `DSPy 3.0 Release, Referencing Input Fields in Docstrings, Agent Bricks Introduction` 


- **DSPy 3.0 发布 Beta 版**：[DSPy 3.0](https://github.com/stanfordnlp/dspy/releases/tag/3.0.0b1) 已发布 **Beta** 版，成员们正在寻找变更内容的全面概述。
   - 一位成员询问 **DSPy 3.0** 是否仍处于 **Beta** 阶段。
- **Docstring 难题：引用输入字段**：一位成员询问是否可以在 DSPy 的 **Docstring** 内部引用**输入字段**，并提到他们对该框架相对陌生。
   - 另一位成员指出 *Docstrings 只是文本*，但原提问者澄清了他们对**动态 jinja 替换**的需求。
- **Agent Bricks 亮相**：一位成员分享了一张截图和一篇介绍 **Agent Bricks** 的 [Databricks 博客文章](https://www.databricks.com/blog/introducing-agent-bricks)链接。
   - 未进行进一步讨论。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1382511149341212753)** (3 条消息): 

> `AgentX summit, 研究论文提交, 峰会出席` 


- ****AgentX Summit 澄清****：一位用户询问了关于 **AgentX summit** 的信息，特别是关于研究论文提交和入围者的出席情况。
- ****研究赛道论文提交****：入围者将被邀请参加 Summit 的海报展示环节或在 Summit 上发表演讲。
   - 使用 [summit 网站](https://rdi.berkeley.edu/events/agentic-ai-summit) 进行单独提交会增加获得额外考虑的机会。
- ****入围者出席说明****：入围者将收到 **单独邀请** 参加峰会，无需注册即可出席。
   - 建议尽早注册以确保名额，入围者可能会获得门票退款。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1382437389514838108)** (3 条消息): 

> `模型速度, Token 数量` 


- **模型速度担忧**：成员们发现“推理模型（thinking models）太慢了”。
   - 其他人询问为什么即使模型较小（如 **1GB, 2GB 或 4GB**），速度依然很慢。
- **Token 问题**：模型缓慢的原因是 *Token 数量太多了*。
   - 这暗示了大量的 Token 可能是导致处理速度变慢的原因。


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1382853343185207366)** (1 条消息): 

> `Windsurf Wave 10, UI/UX 升级, 欧盟集群, 企业级服务` 


- **Windsurf 发布 Wave 10 及其 UI/UX 翻新**：Windsurf 宣布发布 **Wave 10**，其特点是全新的 **UI/UX 升级**，以及在其 [博客文章](https://windsurf.com/blog/windsurf-wave-10-ux-enterprise) 中提到的新团队和企业级服务。
   - 此次发布包括用于 `@-mentions` 和文件引用的新图标、Cascade 面板中与 IDE 主题匹配的代码块、Cascade 面板中接受用户输入的原生终端，以及全新的对话历史 UI。
- **Windsurf 通过新集群扩展至欧盟**：Windsurf 宣布启动其 **EU Cluster**，承诺提供更快的性能并满足欧洲企业日益增长的需求，详见其 [博客文章](https://windsurf.com/blog/windsurf-wave-10-ux-enterprise)。
   - 详情可见其 [YouTube 视频](https://youtu.be/UHinqQiiCI8?si=udyZDkWGg9nq7zcI)，更新日志请访问 [https://windsurf.com/changelog](https://windsurf.com/changelog)。