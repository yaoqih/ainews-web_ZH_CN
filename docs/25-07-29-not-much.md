---
companies:
- zhipu-ai
- alibaba
- moonshot-ai
- x-ai
- ideogram
- figure
- smollm
- openai
date: '2025-07-29T05:44:39.731046Z'
description: '**中国实验室**在7月发布了一系列强大的、采用宽松许可的模型，包括**智谱AI的GLM-4.5**和**GLM-4.5-Air**、**阿里巴巴的Qwen3
  Coder**和**Qwen3-235B**，以及**月之暗面（Moonshot AI）的Kimi K2**。


  这些模型采用了大规模混合专家（MoE）架构，激活参数量在3B到32B之间，上下文窗口高达256K token。**智谱AI的GLM-4.5**在基准测试中足以与**Claude
  4 Opus**和**Gemini 2.5 Pro**相媲美。**月之暗面的Kimi K2**是一款拥有1万亿参数的MoE模型，在**LiveCodeBench**和**AceBench**上的表现超越了其他开源权重模型。


  在视频和图像生成领域，**xAI**推出了**Grok Imagine**，而**Wan2.2**凭借其“图生视频（Image-to-Video）”方案给人留下了深刻印象。**Ideogram**发布了一个角色一致性模型。机器人领域的进展包括**Figure公司的Figure-01和Figure-02**人形机器人，以及用于篮球分析姿态估计的**ViTPose++**。**SmolLM3**的训练和评估代码已根据Apache
  2.0协议完全开源。


  @corbtt指出：*“避开这些中国开源模型的组织正处于显著的竞争劣势。”*'
id: MjAyNS0w
models:
- glm-4.5
- glm-4.5-air
- qwen3-coder
- qwen3-235b
- kimi-k2
- wan-2.2
- grok-imagine
- smollm3
- figure-01
- figure-02
- vitpose++
people:
- yuchenj_uw
- corbtt
- cline
- reach_vb
- ollama
- deeplearningai
- ostrisai
- hojonathanho
- adcock_brett
- skalskip92
- loubnabenallal1
title: 今天没发生什么特别的事。
topics:
- model-releases
- moe
- model-benchmarking
- image-generation
- video-generation
- pose-estimation
- robotics
- training-code-release
- apache-license
---

**平静的一天。**

> 2025年7月28日至7月29日的 AI 新闻。我们为您检查了 12 个 subreddits、544 个 Twitter 账号和 29 个 Discord 社区（227 个频道，6913 条消息）。预计为您节省了 556 分钟的阅读时间（以 200wpm 计算）。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 风格展示所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上向我们提供反馈！

在缺乏重大新闻的情况下，您可能想关注一下现已完整发布的 **Search and Retrieval** 专题，其中目前最受欢迎的演讲是 [**Jerry Liu 关于 Knowledge Work Agents 的演讲**。](https://www.youtube.com/watch?v=jVGCulhBRZI&list=PLcfpQ4tk2k0W3T87n_MZGaV9WfWOmEWtQ&index=1&t=36s)

该专题是对 [GraphRAG](https://www.youtube.com/watch?v=XNneh6-eyPg&list=PLcfpQ4tk2k0U35MFGllN31nmEP9EdCge8&index=13)、[RecSys](https://www.youtube.com/watch?v=LxQsQ3vZDqo&list=PLcfpQ4tk2k0UMEJY1KzWu02OkvCc1e5og) 和 [MCP](https://www.youtube.com/playlist?list=PLcfpQ4tk2k0UqhUyxuMMMmDwyiApd4sDw) 等类似主题的良好补充。

---

# AI Twitter 回顾

**模型发布与性能**

- **中国的开源攻势**：7 月，中国实验室发布了一波强大的、许可宽松的模型，[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1950034092457939072) 强调了这一趋势。主要发布包括来自 **智谱 AI (Zhipu AI)** 的 **GLM-4.5** 和 **GLM-4.5-Air**、**Wan-2.2**（视频）、来自 **阿里巴巴 (Alibaba)** 的 **Qwen3 Coder** 和 **Qwen3-235B** 系列，以及来自 **月之暗面 (Moonshot AI)** 的 **Kimi K2**。这与西方开源发布速度放缓的感知形成鲜明对比，促使 [@corbtt](https://twitter.com/corbtt/status/1950334347971874943) 指出，避开这些模型的机构正处于“显著的竞争劣势”。
- **智谱 AI (Zhipu AI) 的 GLM-4.5 模型**：**智谱 AI** 发布了 **GLM-4.5**（一个 355B 参数的 MoE 模型，激活参数 32B）和 **GLM-4.5-Air**，两者均采用 **MIT 许可证**。该公司宣布由于需求量大，[他们正在努力扩展资源](https://twitter.com/Zai_org/status/1950164491125043515)。这些模型被指出可与 **Claude 4 Opus** 竞争，并在[某些基准测试中](https://twitter.com/Zai_org/status/1949970927006949430)击败了 **Gemini 2.5 Pro**。社区迅速在 **MLX** 和 **DeepInfra** 等平台上提供了这些模型。
- **Qwen3 和 Kimi K2 模型**：**阿里巴巴的 Qwen3 Coder** 表现强劲，在 **Cline** 中的 **diff edit 失败率** 仅为 **5.32%**，据 [@cline](https://twitter.com/cline/status/1949973297455599998) 称，这使其与 **Claude Sonnet 4** 和 **Kimi K2** 并列。[@reach_vb](https://twitter.com/reach_vb/status/1950263476271947822) 和 [@ollama](https://twitter.com/ollama/status/1950291777216262259) 指出，具有 256K 上下文的 **30B MoE (3B 激活)** 版本现在可以通过 **MLX** 和 **Ollama** 在本地运行。**月之暗面 (Moonshot AI)** 的 **Kimi K2** 是一个 **1 万亿参数的 MoE (32B 激活)** 模型，采用修改后的 MIT 许可证发布，并在 **LiveCodeBench** 和 **AceBench** 等基准测试中超越了其他开源权重模型，[如 @DeepLearningAI 所报道](https://twitter.com/DeepLearningAI/status/1950183277161005418)。
- **视频与图像生成**：**xAI** 推出了 **Grok Imagine**，这是一款图像和视频生成工具，目前处于 [候补名单阶段](https://twitter.com/chaitualuru/status/1949946519869685952)。**Wan2.2 5B** 视频模型其 **Image-to-Video (I2V)** 方法给开发者留下了深刻印象，其中每个潜帧（latent frame）都有自己的去噪步长，可能允许生成无限长的视频，[由 @ostrisai 分析](https://twitter.com/ostrisai/status/1950129158618591646)。**Ideogram** 发布了 **Ideogram Character**，这是一个角色一致性模型，可以使用单张参考图工作，[由 @hojonathanho 提及](https://twitter.com/hojonathanho/status/1950261122365333806)。
- **视觉与机器人**：**Figure** 展示了其 **Figure-01** 与新型 **Figure-02** 人形机器人的对比，突出了硬件和能力方面的进步，[见 @adcock_brett 分享的视频](https://twitter.com/adcock_brett/status/1950291267730207125)。**ViTPose++** 展示了令人印象深刻的姿态估计，能够准确跟踪篮球运动员之间复杂的互动，目前正被集成到一个篮球分析 AI 中，该 AI 可以判断球员是否在三秒区内，[据 @skalskip92 称](https://twitter.com/skalskip92/status/1950231824933982428)。
- **SmolLM3 代码发布**：**SmolLM3** 的完整训练和评估代码已发布，包括预训练脚本 (**nanotron**)、后期训练代码（用于 SFT+APO 的 **TRL/alignment-handbook**）和评估脚本，以及 100 多个中间检查点，全部采用 **Apache 2.0 许可证**，[由 @LoubnaBenAllal1 宣布](https://twitter.com/LoubnaBenAllal1/status/1950139809034305568)。

**AI Agent、工具与应用**

- **ChatGPT Study Mode**: **OpenAI** 正在 **ChatGPT** 中推出 **Study Mode**（学习模式），这是一个旨在引导用户逐步学习概念的交互式功能，其作用更像是一个导师而非仅仅提供答案，正如 [@gdb](https://twitter.com/gdb/status/1950309323936321943) 和 [@sama](https://twitter.com/sama/status/1950299705751327149) 所宣布的那样。
- **Runway Aleph In-Context Video Model**: **Runway** 正在开放 **Runway Aleph** 的访问权限，这是一个用于多任务视觉生成的全新上下文视频模型（in-context video model）。[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1950138170806312974) 通过对比“白天变黑夜”效果的复杂手动视频编辑工作流与简单提示 Aleph “变黑夜”的操作，展示了其强大功能。类似的对比还包括[从场景中移除汽车](https://twitter.com/c_valenzuelab/status/1949921138689396976)和[添加爆炸效果](https://twitter.com/c_valenzuelab/status/1950257984715571606)。
- **Google's AI Mode in Search**: **Google** 将其搜索中的 **AI Mode** 扩展到了英国，并引入了新功能，包括上传照片和 PDF 进行查询、用于组织项目的 “Canvas”（画布）以及用于实时帮助的 “Search Live”，详见 [@Google](https://twitter.com/Google/status/1950241246779232260)。
- **LangChain & LangGraph for Agentic Workflows**: **LangChain** 发布了一份关于使用 **LangGraph** 应用六种常见上下文工程方法的指南，并在[一条热门推文](https://twitter.com/LangChainAI/status/1950226846538485918)中提供了视频和代码示例。他们还重点介绍了如何构建一个用于代码生成的自我纠错 RAG Agent。生态系统持续增长，[**LangSmith Traces** 现在集成了服务器日志](https://twitter.com/LangChainAI/status/1949948616182768010)以实现更好的可观测性。
- **Perplexity's Comet Browser**: **Perplexity** 的 **Comet** 浏览器初步采用率强劲，CEO [@AravSrinivas](https://twitter.com/AravSrinivas/status/1950042752655241234) 指出其默认搜索是 **Perplexity**，这可能会推动显著的查询量。他还演示了 Comet 执行一项复杂任务：[在联合航空（United）上订票，包括选座](https://twitter.com/AravSrinivas/status/1949937085164482846)。
- **Development & Tooling**: **BlockDL** 是一个用于可视化设计 **Keras** 神经网络的免费开源 GUI，由 [@fchollet](https://twitter.com/fchollet/status/1950244806967603207) 发布。在工具方面，新的 **Hugging Face jobs CLI** 现在由 **uv** 驱动，以实现更快的环境设置，正如 [@_lewtun](https://twitter.com/_lewtun/status/1949915717836431744) 所分享的。对于构建 Agent 应用的开发者，[@_avichawla](https://twitter.com/_avichawla/status/1950282234893656101) 强调了一种仅用 10 行代码即可将任何模型、RAG 或 Agent 部署为 **MCP server** 的方法。

**Infrastructure, Efficiency & Optimization**

- **Long Context Training on H200**: [@StasBekman](https://twitter.com/StasBekman/status/1950232169227624751) 证明了在单个 **H200 GPU** 上对 Llama-8B 模型进行 **120 万序列长度**的训练现在是可能的。这是通过结合 **ALST**、**FA3 (FlashAttention-3)** 和 **Liger-Kernel** 实现的，后两者最近修复了 int64 索引问题。
- **GSPO in TRL**: 阿里巴巴的 **Group Sequence Policy Optimization (GSPO)** 算法引起了广泛关注，现在已在 Hugging Face 的 **TRL** 库中可用，正如 [@_lewtun](https://twitter.com/_lewtun/status/1949951668914659636) 所宣布的。
- **AMD Contributions to llama.cpp**: [@ggerganov](https://twitter.com/ggerganov/status/1950047168280060125) 指出 **AMD** 团队现在正积极为 **llama.cpp** 代码库做出贡献，这标志着这一流行推理框架将获得更广泛的硬件支持。
- **StepFun Open Sources StepMesh**: 中国 AI 公司 **StepFun**（阶跃星辰）开源了 **StepMesh**，这是一个专为使用 **Attention-FFN disaggregation**（解耦）的推理系统设计的通信库，正如 [@teortaxesTex](https://twitter.com/teortaxesTex/status/1950127131754651655) 所述。
- **Qdrant Edge for On-Device Vector Search**: **Qdrant** 推出了 **Qdrant Edge** 的私测版，这是一个轻量级、嵌入式的向量搜索引擎，旨在为机器人、移动端和 IoT 应用提供设备端运行能力，正如 [@qdrant_engine](https://twitter.com/qdrant_engine/status/1950165409639833603) 所宣布的。

**Research, Techniques & Evaluation**

- **反向传播的历史**：[@SchmidhuberAI](https://twitter.com/SchmidhuberAI/status/1950194864940835159) 提供了 **backpropagation** 的详细历史，澄清了其现代形式最早由 **Seppo Linnainmaa** 在 **1970** 年发表，其前身可追溯至 **1960** 年 **Henry J. Kelley** 的工作。他强调这不仅仅是链式法则，而是其在神经网络中的高效应用。
- **评估危机**：一种日益增长的观点认为标准基准测试正变得越来越不可靠。[@ShunyuYao12](https://twitter.com/ShunyuYao12/status/1950090043344707832) 问道：“当我们不再相信基准测试数据时，该如何评估 LLM？”。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1949912968394940518) 对此表示赞同，称当一个模型伴随一套“激进的新评估套件”发布时，才会令人兴奋。**DailyBench** 由 [@jacob_dphillips](https://twitter.com/andersonbcdefg/status/1949936665637593102) 发布，作为一个自动化的每日基准测试，用于在新鲜问题上追踪前沿模型。
- **新的优化技术**：一篇关于 **Reflective Prompt Evolution** 的论文显示其性能可以超越 **GRPO**，突出了通过自然语言反思进行学习的力量，正如 [@lateinteraction](https://twitter.com/lateinteraction/status/1949984215191208078) 所分享的。**阿里巴巴的 Group Sequence Policy Optimization (GSPO)** 论文是 Hugging Face 7 月份排名第三的热门论文，[@ClementDelangue](https://twitter.com/ClementDelangue/status/1949934196148895799) 预测它将产生巨大影响。
- **LLM 的物理学**：研究人员发布了他们“语言模型物理学”工作的代码，声称他们的 **8B@1T** 模型仅使用 **7% 的算力** 就击败了 **Llama-3.1-8B**，正如 [@giffmana](https://twitter.com/giffmana/status/1950276478861517236) 所分享的。
- **推理与意识**：一场关于什么是推理的讨论浮出水面，[@teortaxesTex](https://twitter.com/teortaxesTex/status/1950158521493811458) 挑衅性地建议它是一种应该能够解决停机问题的“超图灵计算”。与此同时，[@jxmnop](https://twitter.com/jxmnop/status/1950229423849869672) 回忆起该领域如何从争论 GPT-2 是否理解否定，演变到讨论“近乎有意识且能赢得 IMO”的模型。

**行业与更广泛的讨论**

- **4 亿美元 Meta 录用通知的故事**：一个主要的讨论点是，顶尖 AI 人才拒绝了来自 **Meta** 高达 **4 亿美元** 的录用通知，[来自 @willdepue 的一条推文引发了热议](https://twitter.com/willdepue/status/1950253835064086979)。这引发了人们的猜测，即其他公司正在构建什么，能激励研究人员拒绝如此巨额的报酬。
- **能源作为瓶颈**：一位前 **Meta** 员工的评论浮出水面，称 **能源** 是扩展计算规模的最大瓶颈，甚至超过了购买 GPU 的资金。[该推文被 @code_star 转发放大](https://twitter.com/code_star/status/1950263396420767845)。
- **API 与开放权重的安全性**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1950226004984942829) 反驳了 API 模型天生比开放权重模型更安全的观点。他认为，通过降低模型使用门槛，API 可以在不增加显著控制的情况下，将恶意行为者的滥用量增加“几个数量级”。
- **招聘与社区**：**Anthropic** 宣布[正在扩大其研究员计划 (Fellows program)](https://twitter.com/EthanJPerez/status/1950278824102678586)，该计划将外部研究人员与内部团队配对，共同研究安全问题。**Sakana AI** 正在[举办开放日活动](https://twitter.com/SakanaAILabs/status/1950016555799953523)，为其应用工程师团队招募人才。
- **地缘政治**：多条高曝光推文涉及政治氛围，包括 **Speaker Pelosi** 批评 **Donald Trump** 关于台湾领导人赖清德访问决定的推文，[由 @zacharynado 分享](https://twitter.com/zacharynado/status/1950056521330532640)。

**幽默与迷因**

- **发光花园与建筑 Diffusion**：一条开玩笑说“现在我花园的一半都在黑暗中发光”的推文，在回应有关发光植物的新闻报道时，[通过 @nptacek 获得了巨大关注](https://twitter.com/nptacek/status/1950265375658020991)。“他们对 *查看笔记* 一座房子进行了 diffusion”的梗也流传甚广，[由 @sedielem 转发](https://twitter.com/sedielem/status/1950190227475046877)。
- **离奇历史与密码**：来自 [@DavidSHolz](https://twitter.com/DavidSHolz/status/1950104321783218193) 的一条热门推文分享了 20 世纪 30 年代的一项提议，即在金门大桥顶部建造一座**时速 190 英里的过山车**。在另一篇疯传的帖子中，[@jxmnop](https://twitter.com/jxmnop/status/1950272775052284351) 分享了一张用户密码为“Woman”的截图，并评论道“这事儿编都编不出来”。
- **AI 恶搞**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1950323192641503571) 发布了一个使用 **Comet** 浏览器的“热狗或非热狗”示例。[@typedfemale](https://twitter.com/typedfemale/status/1950337102828143000) 发布了一个关于“双性恋 luke farritor”的梗图。
- **引起共鸣的工程师生活**：一篇关于被物理锁在房间里的帖子引起了人们对在项目中“锁死（locked in）”状态的共鸣，[由 @stevenheidel 发布](https://twitter.com/stevenheidel/status/1950316382450823320)。在 **a16z** 的一场路演中，出现了一只“会说话的魔力狗和一个人体金字塔”，[由 @KevinAFischer 分享](https://twitter.com/KevinAFischer/status/1949958038905127340)。

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Qwen3-30B-A3B-Instruct-2507 模型发布与社区反响

- [**Qwen/Qwen3-30B-A3B-Instruct-2507 · Hugging Face**](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) ([得分: 493, 评论: 224](https://www.reddit.com/r/LocalLLaMA/comments/1mcfmd2/qwenqwen330ba3binstruct2507_hugging_face/)): **该帖子讨论了 Qwen/Qwen3-30B-A3B-Instruct-2507 大语言模型在 Hugging Face 上的发布及其性能指标，重点展示了一张显示性能大幅提升的基准测试对比图，但也指出 *“混合推理严重损害了模型的智能”*。([Hugging Face 模型卡片](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507))。帖子请求提供 GGUF 格式量化，并提到了 Unsloth、Bartwoski 和 Mradermacher 等知名量化贡献者，表明社区对高效推理和部署选项的关注。** 评论者们对混合推理架构的权衡展开了辩论，其中一人表示它“严重损害了智能”（可能是指在混合配置中观察到的基准测试退步）。对于实际部署所需的快速量化转换（GGUF）存在明显需求，这反映了模型可用性优先于原始准确率的趋势。
    - 一位评论者观察到混合推理似乎显著降低了模型智能，并引用了一张基准测试对比图，该图显示应用混合技术时性能大幅下降，暗示了此类架构的潜在权衡或局限性。
    - danielhanchen 提供了关于 Qwen3-30B-A3B-Instruct-2507 的 GGUF 格式模型可用性的技术细节，参考 https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF 进行下载，并参考 https://docs.unsloth.ai/basics/qwen3-2507 获取关于运行 GGUF 和 235B MoE 模型的文档。此外还指明，Instruct 变体是使用 `temperature = 0.7` 和 `top_p = 0.8` 的生成参数进行评估的。
    - 线程中隐含了一个关于模型 GGUF（量化）版本的技术请求，通过提及 Unsloth 等人在将 Qwen3-30B-A3B-Instruct-2507 等大型模型转换为高效量化格式方面的工作得到了回应，旨在实现更广泛的部署并降低推理成本。
- [**最新的 Qwen 让我哭了。它并不完美，但我依然爱它。**](https://i.redd.it/gnkbnxzlouff1.png) ([得分: 322, 评论: 60](https://www.reddit.com/r/LocalLLaMA/comments/1mci7uu/newest_qwen_made_me_cry_its_not_perfect_but_i/)): **据报道，图片显示最新的 Qwen3-30B-A3B-Instruct-2507 模型拒绝编造信息，而是明确承认无法找到答案。与早期的 LLM 相比，这种行为值得关注，早期的 LLM 在缺乏知识时经常会产生幻觉或生成听起来合理但错误的回应。该帖子强调了这一代 Qwen 迭代中增加的可靠性和谨慎性，反映了与先前版本相比在拒绝和不确定性处理方面的改进。** 一条热门评论称赞这种行为是“完美的”，表示诚实的不确定性比自信的错误更可取；另一条评论指出，该模型现在看起来更成熟，不太可能假装专业，这被视为一项积极的技术进步。

- 一些评论指出，最新的 Qwen 模型（例如 30B 和 235B）现在有时会承认无法回答问题或找不到问题，而不是产生幻觉（hallucinating）或虚构信息，这标志着与早期模型相比，其可靠性有所提高。
- 一位用户详细描述了对 Qwen 30B 的迭代测试，模型最初表现出过度思考、自我怀疑，并在代码调试中难以识别问题。只有通过直接提示（prompting）和持续澄清，模型最终才解决了所有问题，但直到用户切换到 235B 版本并获得了一个可用的 Prompt 模板之前，它仍未能推荐通用的 Prompt 改进方案。
- 重新使用由较大的 Qwen-235B 模型生成的改进 Prompt，使得 30B 模型在单次尝试中就成功修正了代码，这证明了 Prompt Engineering 的影响，以及在不同规模模型之间迁移 Prompt 模板以增强性能的价值。
- [**🚀 Qwen3-30B-A3B 小幅更新**](https://i.redd.it/nd904g7gbuff1.jpeg) ([Score: 214, Comments: 34](https://www.reddit.com/r/LocalLLaMA/comments/1mcg4qt/qwen330ba3b_small_update/)): **该图片是一个技术 Benchmark 对比，突出了 Qwen3-30B-A3B 模型在最近更新后的改进。Benchmark 结果——如 GPQA: 70.4 vs 54.8 (+15.6), AIME25: 61.3 vs 21.6 (+39.7) 等——展示了相比之前版本的显著提升，特别是在推理（GPQA, Arena-Hard v2）、数学（AIME25）和代码（LiveCodeBench v6）方面。显示的另一个关键增强是将支持的 Context Length 从** `128k` **增加到** `256k` **tokens，使该模型在性能上接近 GPT-4o 和 Qwen3-235B-A22B (Non-Thinking)，同时完全在非思考模式（无 <think> 块）下运行。** 评论者分享了实用的部署技巧，例如 GGUF 格式链接和首选的推理设置（`temperature = 0.7, top_p = 0.8`）。Benchmark 中大幅度的数值提升也被认为是一次重大飞跃，而非“小幅更新”，引发了对此次更新影响的赞赏。
    - 一位用户分享说，他们已经制作了 Qwen3-30B-A3B 模型的 GGUF 量化版本，可在 https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF 获取，并建议将生成设置设为 `temperature = 0.7, top_p = 0.8` 以获得最佳模型性能。
    - 性能 Benchmark 表明 Qwen3-30B-A3B 较之前版本有实质性改进：GPQA 准确率从 54.8 提高到 70.4 (+15.6)，AIME25 从 21.6 提高到 61.3 (+39.7)，LiveCodeBench v6 从 29.0 提高到 43.2 (+14.2)，Arena-Hard v2 从 24.8 提高到 69.0 (+44.2)，BFCL-v3 从 58.6 提高到 65.1 (+6.5)。Context Window 也从 128k 翻倍至 256k tokens。
- [**Qwen/Qwen3-30B-A3B-Instruct-2507 · Hugging Face**](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) ([Score: 106, Comments: 15](https://www.reddit.com/r/LocalLLaMA/comments/1mcfuka/qwenqwen330ba3binstruct2507_hugging_face/)): **阿里巴巴发布了新的 Qwen3-30B-A3B-Instruct-2507，这是一个 Mixture-of-Experts (MoE) 大语言模型，并提供了用于高效推理的量化 GGUF 版本（[Unsloth repo](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF)）。据报道，性能 Benchmark 表现强劲，社区提供了设置文档（[Unsloth docs](https://docs.unsloth.ai/basics/qwen3-2507)）和用户分享的参考视觉对比。该模型旨在易于使用（'no_think'），并利用了 Qwen 特有的 A3B 路由策略。讨论强调了 LLM 改进的飞速步伐，一些用户预测像 Qwen3-30B-A3B 这样的进展可能在几年内实现大型模型的端侧（on-device）推理。技术观点称赞了 Benchmark 结果和架构选择。**
    - danielhanchen 提供了 Hugging Face 上 Qwen3-30B-A3B-Instruct-2507 的 GGUF 格式模型文件链接 (https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF)，以及在 UnsLoTH 平台上运行它们的详细文档 (https://docs.unsloth.ai/basics/qwen3-2507)，方便在各种硬件上进行部署。
    - Qwen3-30B-A3B-Instruct-2507 模型的 Benchmark 被描述为非常强劲，一位用户引用了 [来自阿里巴巴官方推特的详细 Benchmark 结果](https://x.com/Alibaba_Qwen/status/1950227114793586867/photo/1)，并称该模型为 "no_think"（暗示高效率或极低的推理延迟）Qwen3 30B A3 变体。
    - 用户 AppearanceHeavy6724 提供了体验反馈，指出新版本模型与原始 Qwen 30B 相比有巨大改进——特别是在创意写作方面。然而，小专家 Mixture-of-Experts 架构中常见的问题仍然存在，即散文质量在表面上看起来很强，但在仔细推敲下会“崩塌”，特别是在小说创作任务中。

- [**Qwen3-30b-3ab-2507 是 MCP 使用的神器！**](https://www.reddit.com/r/LocalLLaMA/comments/1mcji8s/qwen330b3ab2507_is_a_beast_for_mcp_usage/) ([Score: 134, Comments: 22](https://www.reddit.com/r/LocalLLaMA/comments/1mcji8s/qwen330b3ab2507_is_a_beast_for_mcp_usage/)): **该帖子强调了 Qwen3-30B-3AB-2507 模型在跨多台服务器自主管理 MCP（多组件处理）任务方面的性能，并由用户提供的工作流（[Pastebin 链接](https://pastebin.com/WNPrcjLS)）证实了这一说法。值得注意的是，Qwen3-30B 的 MLX 8-bit 量化版本在处理 *长且复杂的系统提示词* 时保持了高准确度，据报道其表现优于 Mistral 24B，并引发了关于其与 Mistral Small 对比性能的讨论。** 技术评论验证了 Qwen3-30B 工作流的稳健性和灵活性；多位用户对其遵循复杂提示词的能力印象深刻，并一致认为在此场景下它超越了 Mistral 24B。
    - 一位用户报告称，MLX 8-bit 量化版本的 Qwen3-30b-3ab-2507 可以毫无问题地处理非常长且复杂的系统提示词，并且在这些特定的工作负载需求下表现 *远好于 Mistral 24B*。
    - 存在一个明确的技术对比问题，即 Qwen3-30b-3ab-2507 的性能（针对 MCP 场景）如何与 “Mistral 24B” 和 “Mistral Small” 相比，这表明用户对特定应用工作流的正面交锋评估感兴趣。
    - 另一位用户讨论了在 LM Studio 中管理频繁发布的新模型，间接暗示了建立稳健测试环境的重要性，以便在 MCP 等利基用例中快速进行基准测试和验证新模型性能。

### 2. GLM 4.5 模型发布、基准测试与生态系统集成

- [**我刚刚试用了 GLM 4.5**](https://www.reddit.com/r/LocalLLaMA/comments/1mc8tks/i_just_tried_glm_45/) ([Score: 273, Comments: 120](https://www.reddit.com/r/LocalLLaMA/comments/1mc8tks/i_just_tried_glm_45/)): **用户使用一个关于全球 BESS 市场的复杂、开放式幻灯片生成提示词测试了 GLM 4.5（来自 [z.ai](http://z.ai/)），报告称即使在提示词极简的情况下，也能获得带有引用且非虚构数据的强劲结果。评论中的基准测试和对比声称 GLM 4.5 Air 版本在输出质量上与最新的 Qwen3-235B 相当，但效率显著更高（运行速度快 2 倍，内存占用减半，例如在 6x3090s 上使用 FP8 达到 ~40-50 tok/s），并支持混合推理。模型已发布在 [neuroengine.ai](http://neuroengine.ai/)（测试中，不保证运行时间）。用户提到了先进的代码理解/生成能力（例如自主生成 5100 行单元测试）以及在各种推理/召回任务中的一致性，表明其具有实际应用的可行性，以及足以媲美 Claude 3.7–4.0 和 DeepSeekV3 等顶尖模型的广泛覆盖推理能力。** 技术辩论集中在实际质量与基准测试的对比上，多位用户注意到 GLM 4.5 强大的通用能力和效率（硬件资源利用率），同时也承认其输出带有 GPT/Claude 风格的影子，这可能是由于在其输出数据上进行训练的结果。讨论内容还涉及投机采样（MTP）、FP8 效率、使用 vLLM 部署，以及将其评估为用于私有/本地推理的最佳本地（on-prem）模型。
    - GLM 4.5 Air 变体的性能与 Qwen3-235B 进行了对比，用户报告称在两倍速度和一半内存占用的情况下质量相当——具体而言，在使用 FP8 精度和混合模式的 6x3090 GPU 配置上达到 40-50 tokens/sec。值得注意的是，MTP 投机采样尚未启用，且讨论了模型托管的可扩展性（可能升级到完整版 GLM 和 AWQ 量化），指向了强大的工程关注点。
    - 实际代码生成评估强调，GLM 4.5 Air（像 Claude/Sonnet 一样进行代码封装）在两小时内以极少的人工干预生成了 5100 行准确的单元测试，表现优于之前的模型——这表明与 Claude-3.5 Sonnet 和 Qwen-235B 直接对比时，它在编程任务上具有最先进的本地部署能力。
    - 一种涵盖挑战性领域（编程、冷门事实、创造性推理）的定制评估方法（“vibe bench”）将 GLM 4.5 Air-100B 评为在广泛、全面的能力上显著优于 DeepSeek v3、Claude 3.7–4.0 和 Qwen-235B。用户注意到该模型在 GPT 和 Claude 输出上进行蒸馏训练的证据，这在措辞中可以检测到，但显然有助于模型在各种任务类型中保持一致的可靠性和平衡。

- [**GLM 4.5 支持正进入 llama.cpp**](https://github.com/ggml-org/llama.cpp/pull/14939) ([Score: 202, Comments: 46](https://www.reddit.com/r/LocalLLaMA/comments/1mc6fbp/glm_45_support_is_landing_in_llamacpp/)): **一个草案 Pull Request (PR) 正在进行中，旨在为 [llama.cpp](https://github.com/ggerganov/llama.cpp) 添加 GLM 4.5 模型支持。作者承认这是他们第一次在代码库中实现新架构。社区非常关注该实现是否支持 Multi-Token Prediction (MTP)，这一特性对于提升效率以及与 LMStudio 等工具的集成至关重要。建议用户目前不要从该草案构建 GGUF 文件，因为实现尚不完整且不稳定。** 技术争论围绕初始实现的完整性和正确性展开，PR 作者邀请大家协作并提醒用户等待最终方案。对于 GLM 4.5 Air 对本地 LLM 生态系统的影响，尤其是其尺寸性能比，人们抱有极大的期待。
    - 讨论强调了在 llama.cpp 的初始 GLM 4.5 集成中是否支持 Multi-Token Prediction (MTP) 的不确定性，用户对原生 MTP 支持表现出浓厚兴趣，以期提高推理能力，特别是在 LMStudio 中。
    - 关于目前的 GLM 4.5 实现存在重大警示：PR 作者指出这只是一个草案，不完整，不适合生产环境或 GGUF 构建。由于多个架构元素尚未完成，鼓励开展协作和进一步贡献。
    - vLLM/Sglang 对 GLM 4.5 支持的基准测试和配置说明显示，在 6x3090 配置下，仅在流水线并行 (pipeline parallelism) 而非张量并行 (tensor parallelism) 下表现稳定：生成速度约为 40 tok/s，提示词处理速度约为 800 tok/s。此外，vLLM 的支持仍存在 Bug 且未启用 MTP，这引发了关于部署中更广泛的多 Token 能力的疑问。
- [**我那台 2.5 年前的笔记本电脑现在可以用 GLM-4.5 Air 和 MLX 编写 JavaScript 版《太空侵略者》了**](https://simonwillison.net/2025/Jul/29/space-invaders/) ([Score: 138, Comments: 20](https://www.reddit.com/r/LocalLLaMA/comments/1mcee42/my_25_year_old_laptop_can_write_space_invaders_in/)): **原作者展示了一台拥有 2.5 年机龄的笔记本电脑，通过在 MLX 推理框架上运行 GLM-4.5 Air LLM，可以本地生成功能完整的《太空侵略者》(Space Invaders) 克隆版 JavaScript 代码。这意味着利用 Apple 的 MLX 进行资源优化计算，可以在老旧的消费级硬件上实现高效的设备端 LLM 推理，用于代码生成（游戏原型设计）。讨论集中在机龄约 2.5-3 年的笔记本电脑上，包括高配的 M1 Max 型号。** 评论者询问了笔记本规格（特别提到了 Apple M1 Max, 64GB RAM），对结果表示惊讶，并讨论了生成《太空侵略者》与生成新颖游戏对于基于 LLM 的代码生成的价值。
    - 用户讨论了 MLX 在 Apple Silicon 上的效率和性能，有人提到在 M3 上运行 GLM-4.5 Air，并通过 Roo Code 实现了流畅的代码生成。这突显了与 Cursor 等主流选择相比，macOS 上本地代码生成工具新兴的能力。
    - 人们对将 MLX 作为 Mac 上本地运行代码生成任务的替代方案表现出越来越大的兴趣，这主要得益于它在近期 Apple 硬件（如 M1 Max 和 M3 芯片）上的兼容性和性能。一些用户正在考虑为了获得最佳 MLX 使用体验而更换硬件，强调了其在当前工具版图中的价值。
- [**今年最佳开源模型和最具性价比模型**](https://www.reddit.com/r/LocalLLaMA/comments/1mc5oh2/this_years_best_opensource_models_and_most/) ([Score: 100, Comments: 24](https://www.reddit.com/r/LocalLLaMA/comments/1mc5oh2/this_years_best_opensource_models_and_most/)): **GLM-4.5 是一个基础 LLM，拥有 355B 总参数和 32B 激活参数，旨在统一推理、编程和 Agent 场景；GLM-4.5-Air 变体拥有 106B 总参数和 12B 激活参数，采用更紧凑的设计以提高资源效率。初步[基准测试](https://preview.redd.it/bisgmn0utrff1.png?width=4464&format=png&auto=webp&s=8b159e95ccba8f0becc1ee6fb596cb4fdde5217c)显示其性能强劲，但作者鼓励将其与 Qwen 等新发布模型进行对比。技术细节：[博客](https://z.ai/blog/glm-4.5), [Hugging Face](https://huggingface.co/zai-org/GLM-4.5), [GitHub](https://github.com/zai-org/GLM-4.5)。** 热门评论强调在做出最终判断前需要进行彻底的第三方基准测试，并指出 GLM-4.5-Air 在代码和问题解决方面的实际表现优于其他开源模型（如 Qwen, Kimi K2）。

- 几位评论者强调，最近发布的开源模型（如 Qwen 和 GLM 4.5）缺乏独立的第三方基准测试（benchmarks），并强调在宣布任何模型为最佳或最具性价比之前，可靠的比较评估是必不可少的。
- 直接的上手体验表明，不同任务的性能差异显著：一位用户报告 GLM 4.5 Air 解决了一个其他本地 LLM 难以处理的复杂 CSS 编码问题，而另一位用户发现它在小说写作方面的表现不如 Big 4.5 和小型 4-0414-32b 模型。
- 对 Big 4.5、Kimi K2 和 Qwen 等模型在代码和小说生成等用例中的主观评价表明，模型的优势可能高度依赖于具体任务，这突显了对细粒度基准测试（例如针对编码或创意写作数据集）的需求。

### 3. Meta Observations on AI Model Progress (Memes and Commentary)

- [**its getting comical**](https://i.redd.it/txsukljc5pff1.png) ([Score: 973, Comments: 92](https://www.reddit.com/r/LocalLLaMA/comments/1mbvf2z/its_getting_comical/)): **图像本身无法分析，但讨论集中在近期美国公司缺乏开源权重（open-weight）LLM 发布的问题上，评论者注意到一种倾向于提供仅限 API 访问而非可下载模型的趋势。提到的具体发布包括 Granite、Micro Gemma 和更新后的 Nemos（这些都是近期相对罕见的开源权重发布）。该帖子广泛哀叹了美国开源 AI 模型的现状，并将宣传噱头与缺乏可下载权重的实际交付进行了对比。** 评论中对美国公司交付开源权重 LLM 存在明显的怀疑，对反复宣布但未转化为实际发布的做法感到沮丧。一些评论还涉及更广泛的行业和地缘政治影响，尽管这些评论在语气上更具轻蔑或讽刺意味。
    - paryska99 讨论了 Agentic LLM 工作流的快速进展，特别提到在 Kimi K2 和 Qwen3 表现出令人期待的结果后，GLM 4.5 的发布改变了他们的偏好。他们指出 GLM 4.5 的表现优于这些竞争对手，将其描述为 *真正的 Claude Sonnet 终结者*，标志着在某些任务上比领先模型有显著的性能提升。
    - a_beautiful_rhind 强调了近期美国公司缺乏开源权重模型的发布，列举了 Granite、Micro Gemma 和更新后的 Nemos 作为最新案例，并评论了当前偏好 API 访问而非开源发布的趋势。

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Wan 2.2 Model Release Benchmarks and Comparisons

- [**2d animation comparison for Wan 2.2 vs Seedance**](https://v.redd.it/lqs9s9fsotff1) ([Score: 770, Comments: 59](https://www.reddit.com/r/StableDiffusion/comments/1mccuf0/2d_animation_comparison_for_wan_22_vs_seedance/)): **该帖子展示了 Wan 2.2 和 Seedance 在 2D 动画生成方面的非正式对比，强调 Wan 2.2 产生了质量尚可的动画，但表现出一些视觉伪影（artifacts）。未提供详细的基准测试或参数设置，评估看起来是观察性的而非严谨的。** 评论者注意到了成本差异（Wan 2.2 免费，而 Seedance 收费），并引用了具体的定性结果（例如 Wan 2.2 中的伪影，Seedance 海豚动画的“悲伤”外观）。不存在深入的技术争论或实现讨论。
    - d4pr4ssion 指出 **WAN 2.2 产生了更优越的 2D 动画**，理由是动画输出中具有 *更多的动作和更高的连贯性*。这表明 WAN 2.2 在 2D 动画序列中可能具有更好的时间一致性或运动处理能力，这对于动画任务至关重要。

- [**Wan 2.2 - 在 RTX 5090 上约 60 秒生成，质量绝对出色。**](https://v.redd.it/6njp2ehhvpff1) ([Score: 630, Comments: 115](https://www.reddit.com/r/StableDiffusion/comments/1mbyna7/wan_22_generated_in_60_seconds_on_rtx_5090_and/)): **一位用户展示了 Wan 2.2（一个视频生成模型）的结果，强调其在 RTX 5090 GPU 上本地运行，仅需约 60 秒即可生成高质量、富有表现力的 3D 卡通以及写实角色动画。工作流细节显示使用了 t2v（文本转视频），分辨率为** `832x480p`**，LightX2V LoRA 节点强度为** `1.5`**，采用 Unipc/Simple 采样，共** `10 steps` **（在第 0-5 帧和 5-10 帧之间拆分高/低步数）。完整的脚本工作流在 [此 gist](https://gist.github.com/Art9681/91394be3df4f809ca5d008d219fbc5f2) 中提供。** 评论者指出，Wan 2.2 相比 2.1 有显著的质量提升，尤其是在面部表情和本地快速迭代的实用性方面表现强劲。有人建议结合 kontext 以进一步扩展工作流。
    - 原作者提供了一个详细的工作流 gist，概述了在 RTX 5090 上实现快速生成（约 60 秒）的步骤。该工作流包含了关于移除不必要组件的说明，并建议用户升级到新版本的 lightx2v 节点以获得更好的结果 ([gist 链接](https://gist.github.com/Art9681/91394be3df4f809ca5d008d219fbc5f2))。
    - 性能细节讨论：使用强度设置为 1.5 的 lightx2v LoRA 节点，在 832x480p 分辨率下生成，总计 10 steps。推理调度在 0-5 步（高）和 5-10 步（低）之间拆分，并使用 unipc/simple 采样器。用户评论称，其质量代表了相比 2.1 版本的重大改进。
    - 关于速度差异的讨论：一位用户注意到他们的工作流明显更慢，这凸显了工作流优化和版本升级对于匹配原作者结果的重要性。
- [**好的，Wan 2.2 表现不俗……这里有一些动作类动物视频！**](https://v.redd.it/cmqux8w0hsff1) ([Score: 342, Comments: 35](https://www.reddit.com/r/StableDiffusion/comments/1mc7q9u/ok_wan22_is_delivering_here_some_action_animals/)): **原作者展示了 Wan 2.2 的视频输出，在 RTX 5090 GPU 上使用带有** `torch.compile` **和 SageAttention2 的 ComfyUI 默认工作流，实现了每组镜头 18 分钟的渲染时间（之前为 36 分钟）。质量被认为比之前的版本有显著提高，尽管速度仍被认为不足以满足生产工作负载。评论者强调了 Wan 2.2 在** `fp8` **和** `fp16` **推理之间的明显质量差异，尽管速度较慢，但更倾向于使用 fp16 以获得卓越的效果。值得注意的是，Triton 和 SageAttention2 被认为带来了实质性的速度提升。** 讨论集中在 `Triton + SageAttention` 工作流的技术验证上，用户验证了预期行为，并对 Wan 2.2 在 `fp8` 和 `fp16` 之间的质量飞跃表示惊讶。虽然生成速度有所提高，但仍然是生产部署的限制因素。
    - 一位用户比较了 Wan 2.2 的 fp8 和 fp16 质量，观察到 fp16 产生的图像质量明显更好，代价是比 fp8 生成速度慢。这比 Wan 2.1 有所改进，在 2.1 中 fp8 和 fp16 之间的差距不太明显，这表明模型或代码库的增强影响了数值精度的权衡。
    - 报告称结合 Triton 和 SageAttention 时性能大幅提升，生成时间从 36 分钟减半至 18 分钟。评论者寻求对其设置的验证，含蓄地提出了关于将这些加速库/技术栈集成到 Stable Diffusion 流水线时的正确工作流、配置和潜在最佳实践的问题。
- [**Wan 2.2 的人物图像生成非常出色。这个开源模型前景广阔。**](https://www.reddit.com/gallery/1mcm7qm) ([Score: 265, Comments: 77](https://www.reddit.com/r/StableDiffusion/comments/1mcm7qm/wan_22_human_image_generation_is_very_good_this/)): **该帖子介绍了开源的 WAN 2.2 人物图像生成模型，并指出了其强大的性能，其工作流针对具有 24GB VRAM 的系统进行了优化（参见 Hugging Face 工作流配置）。该模型优先考虑高保真度而非速度，为了质量牺牲了生成时间，并且可能与提速 LoRA 模块兼容，但会以图像质量为代价。提供了一个全分辨率画廊，以展示未受 Reddit 压缩影响的原生输出。** 评论者要求提供详细的设置和工作流步骤，表明了对可重复性的兴趣。存在关于 Reddit 图像压缩掩盖细节的担忧，导致通过外部画廊分享结果。该模型的高 VRAM 需求和较慢的速度被认为是性能的权衡。

- 共享了一个针对 Wan 2.2 人像生成优化的 workflow，为获得最佳性能需要 24GB VRAM。虽然存在未经测试的低显存（GGUF）配置，但该 workflow 优先考虑高图像质量而非速度，且尝试使用 LoRA 适配器加速生成可能会降低质量。完整的 workflow JSON 可在 [Hugging Face](https://huggingface.co/RazzzHF/workflow/blob/main/wan2.2_upscaling_workflow.json) 获取。
- 分享视觉结果的一个主要挑战是来自 Reddit 图像压缩的信号损失，这会降低高质量 pipeline 生成的精细细节。为了获得无压缩的输出，用户被引导至外部画廊进行真实评估：[full quality gallery](https://postimg.cc/gallery/8r8DBpD)。
- 提出了一个技术假设：在时间数据上训练的基于视频的生成模型，可能天生就能获得对 3D 几何和旋转或视角变化下物体一致性的更深理解，这表明如果视频模型能在低端硬件上高效运行，它们可能会成为图像生成的基准。
- [**Wan 2.2 14B T2V (GGUF Q8) vs Flux.1 Dev (GGUF Q8) | text2img**](https://www.reddit.com/gallery/1mc981k) ([Score: 215, Comments: 66](https://www.reddit.com/r/StableDiffusion/comments/1mc981k/wan_22_14b_t2v_gguf_q8_vs_flux1_dev_gguf_q8/)): **该帖子对比了两个用于 text-to-image 生成的 GGUF Q8 量化模型：WAN 2.2 14B T2V 与 Flux.1 Dev，两者均在配备 32GB RAM 的 RTX 3090 上运行，并使用相同的 1080x1080 分辨率、** `res_2s` **采样器和** `bong_tangent` **调度器。值得注意的是，WAN 2.2 使用 14B 参数、仅 8 个 step 和 1 的 CFG，而 Flux.1 Dev 的参数量未知，使用 30 个 step 和 3.5 的 CFG；两者的运行时间相近（约 90 秒/生成）。[支持帖子](https://www.reddit.com/r/StableDiffusion/comments/1mbsqxv/wan_22_14b_t2v_txt2img/) 详细描述了测试 workflow。** 专家评论指出 **WAN 2.2** 生成的图像更自然，并将其优势归功于更大的 `14B` 权重规模；共识是 WAN 在视觉质量上优于 Flux，有人指出“Flux 完蛋了”，其他人则强调了规模差异的影响。
    - 几位用户强调 WAN 2.2 14B 在写实度和自然外观方面比 Flux.1 Dev 有显著提升，相比之下 Flux 表现出更重的人工感或“AI 磨皮”感。技术推测指出，额外的约 20 亿参数（14B 对比 Flux 的参数量）可能是影响 WAN 卓越定性输出和更丰富色彩呈现的因素。
    - 讨论中提到了 WAN 2.2 在生成具有逼真色彩的电影感写实图像方面的优势，表明其架构或数据策选可能天生偏向写实主义。一些用户还表示有兴趣对 WAN 进行进一步的微调（finetuning）以获得更广泛的风格多样性，类似于 MidJourney 的方法，这可以解决目前在各种艺术风格创意方面的局限性。
- [**使用 8GB VRAM 制作的 Wan 2.2 i2v 示例**](https://v.redd.it/vy9vtnmistff1) ([Score: 196, Comments: 30](https://www.reddit.com/r/StableDiffusion/comments/1mcdfy5/wan_22_%C4%B12v_examples_made_with_8gb_vram/)): **原作者（OP）报告了在 Q6 量化（可能是 GGUF 格式）和 l2v lightx2v LoRA 强度 1.0 下，使用 8GB VRAM（未指定 GPU）、8 个 step、1.0 CFG，在高质量和低去噪设置下成功运行了 wan2.2 i2v (image-to-video) 模型。该设置使用了添加了 GGUF 和 LoRA 加载器的默认 ComfyUI workflow，表明在有限的显存限制下这种配置是可行的。** 几位评论者由于复现性问题请求提供 workflow 文件，特别是关于 GGUF 和 WAN 2.2 VAE 的问题。一位用户指出，即使在配备 12GB VRAM 的 3060 GPU 上使用 q4_k_m 也会出现 OOM 错误，而另一位用户询问了关于 GGUF 使用的细节，这表明 GGUF 兼容性和高效内存使用是值得进一步 workflow 文档化或故障排除的重要技术痛点。
    - 用户报告在 3060 12GB GPU 上运行 q4_k_m 量化时出现显存溢出（OOM）错误，表明 Wan 2.2 i2v 的 VRAM 需求在某些量化设置或模型（GGUF, VAE）下可能会超过 8GB–12GB。
    - 强调了技术 workflow 的共享；用户请求导出确切节点/workflow 设置的 JSON，因为微小的差异（例如自定义或未知节点）可能会阻碍成功复现。
    - 关于使用 Lightx LoRA 配合 8 个去噪 step 的合理性产生了疑问，因为社区预期通常是 4–6 个 step，这暗示了一种技术优化或未解释的参数更改，可能会影响输出质量或生成时间。

- [**Wan 2.2 可以实现 Veo3 在起始图像上书写的技巧（致谢**](https://v.redd.it/csy54chiksff1) [guizang.ai](http://guizang.ai/)[)](https://v.redd.it/csy54chiksff1)** ([评分: 115, 评论: 19](https://www.reddit.com/r/StableDiffusion/comments/1mc807b/wan_22_can_do_that_veo3_writing_on_starting_image/)): **该帖子指出，根据 [guizang.ai](http://guizang.ai/) 的演示，新型生成式 AI 模型 Wan 2.2 可以执行此前与 Google 的 Veo3 相关的“在起始图像上书写”技术。该技术涉及在生成开始时，在用户选择的图像上进行稳健的文本渲染——据称某些模型（如 Kling）缺乏这种能力，这突显了 Wan 2.2 在图像条件控制（image conditioning）和文本集成方面的优势。** 评论强调了对 Wan 2.2 能力的惊讶，表示竞争对手（如 Kling）无法复制这一点。此外，也有人对展示结果的可复现性表示怀疑，暗示需要进行独立验证。
    - 评论强调了一项技术壮举：据报道 Wan 2.2 复制了 Veo3 的“在起始图像上书写”技巧，一些用户表示其他流行模型（如 Kling）尚未成功复现该功能。这表明与替代方案相比，该模型在细粒度图像引导或提示词条件控制（prompt conditioning）方面取得了显著进步。
    - 一位用户询问了实现细节，询问是否需要通过将初始图像中存在的文本复制到模型提示词中来进行提示词工程（prompt engineering），或者模型是否可以在没有显式文本引导的情况下完成任务。这涉及到了利用此类功能的实际工作流程差异。

### 2. OpenAI GPT-5 与学习模式公告

- [**显然 GPT-5 正在推出？具备更深层次的思考能力 + 视频聊天等功能**](https://i.redd.it/vx5h19qd7rff1.jpeg) ([评分: 342, 评论: 87](https://www.reddit.com/r/singularity/comments/1mc44ls/apparently_gpt5_is_rolling_out_with_ability_to/)): **图片据称展示了 GPT-5 的推出，具有更深层次的推理和视频聊天等新功能。然而，评论者和相关来源澄清说，这张图片和相关说法几乎可以肯定伪造的，可能源自已知传播虚假信息的账号。OpenAI 官方或信誉良好的来源均未确认此类推出，且“Prime”产品似乎是虚构的。** 评论者压倒性地认为该图片不合法，并引用了伪造泄密和专门为传播虚假信息而创建的账号作为证据。技术共识是 OpenAI 不会以这种方式推出 GPT-5。
    - 用户揭穿了所谓的 GPT-5 泄密，指出作为“证据”流传的帖子和图片似乎是伪造的，并与之前不活跃或不可靠的账号有关。其中提到了 OpenAI 员工 Stephan Casas 的一条推文，澄清了围绕所谓推出的混乱，并讨论了“gpt prime”并非合法功能。
    - 一些用户报告了 UI 更新或细微的新功能（如 Plus 账户的“思考更久”按钮），将其视为即将发布更新或功能公告的潜在指标，但尚未确认或观察到 GPT-5 的核心升级。这表明细微的 UI/UX 实验正在进行中，但不一定与重大模型发布挂钩。
- [**GPT-5 Alpha**](https://i.redd.it/0fs7z2xzytff1.jpeg) ([评分: 251, 评论: 80](https://www.reddit.com/r/singularity/comments/1mce9ho/gpt5_alpha/)): **图片显示了 Cursor 设计负责人发布的一条社交媒体帖子，提到了使用 “GPT-5 Alpha” 进行 “vibe coding”，暗示可以早期或内部访问 OpenAI 下一个备受期待的模型版本。语境表明这并非正式公告，而是对 GPT-5 能力或即将发布的非正式或预览性暗示。评论中提供了一个 GitHub 仓库链接 (https://github.com/ryokun6/ryos)，尽管它与 GPT-5 的关系在帖子中并不明确。有提到原始推文可能已被删除，引发了对该帖子公开意图的质疑。** 评论者推测发布时间线（“所以……是这周还是什么时候？”），并讨论了帖子的真实性或意图泄露，同时注意到原始来源可能已被快速删除。没有讨论 GPT-5 Alpha 的技术基准测试数据或具体功能。
    - 一位用户链接到了该项目的 GitHub 仓库 (https://github.com/ryokun6/ryos)，表明 GitHub 上可能存在与 “GPT-5 Alpha” 相关的开源或实验性实现。技术读者可以检查代码、文档和 Issue，以评估模型细节、实验性功能或架构设计。

- 有关于公共沟通策略的讨论，特别是关于推文可能被删除的情况，这表明有关 "GPT-5 Alpha" 的信息可能受到严格控制或处于快速变化中，可能影响开发者及时获取信息。这意味着跟踪项目进展可能需要监控多个来源或存档数据。
- 一条评论提出了 NDA (Non-Disclosure Agreements) 的问题，质疑团队成员或测试人员是否在合同上被限制透露 "GPT-5 Alpha" 的更新。对于技术读者来说，这强调了在官方发布之前，技术细节和内部 Benchmark 可能无法轻易获得，从而影响独立验证声明或性能的能力。
- [**Finally ! GPT-5 is almost there and it's freaking amazing**](https://i.redd.it/z2i8qgqkztff1.jpeg) ([Score: 473, Comments: 118](https://www.reddit.com/r/OpenAI/comments/1mcecll/finally_gpt5_is_almost_there_and_its_freaking/)): **该帖子的图片似乎显示了一个暗示可以访问 GPT-5 早期版本的 Prompt 或对话，但没有提供任何具体的演示或性能的技术 Benchmark。评论揭穿了这一说法，澄清该图片与一条推文有关，该推文错误地将先进的结果归功于 GPT-5，而实际上，展示的项目是使用 Cursor（一个 AI 工具）制作的，并没有实际确认或证据表明使用了 GPT-5。讨论表明，该帖子很可能是推测性或误导性的，而不是 GPT-5 的技术泄露或 Benchmark。** 多位评论者强调帖子缺乏证据，批评其误导信息，并指出原始开发者和 Cursor 都没有声称使用了 GPT-5。
    - 人们对原始推文中 "GPT-5" 声明的真实性表示怀疑，评论者指出所引用的展示项目实际上涉及一名 Cursor 员工，而他从未声称该项目是由 GPT-5 驱动的。该帖子似乎将无关的工作与未发布的模型混为一谈，引发了对模型能力和来源误导信息的担忧。
- [**OpenAI: Introducing study mode - A new way to learn in ChatGPT that offers step by step guidance instead of quick answers**](https://openai.com/index/chatgpt-study-mode/) ([Score: 319, Comments: 37](https://www.reddit.com/r/singularity/comments/1mchrs2/openai_introducing_study_mode_a_new_way_to_learn/)): **OpenAI 为 ChatGPT 推出了“学习模式 (study mode)”，旨在提供循序渐进的学习，而不是即时答案。初步的用户反馈表明，该功能有时会针对之前未提供的信息进行测验，引发了关于 Context Awareness（上下文感知）的问题，以及这是为了促进外部研究的有意设计还是实现上的缺陷。** 评论者指出，“学习模式”目前感觉像是一个增强版的 Prompt，并建议有意义的区分将需要对教育模型进行进一步的 Fine-tuning。人们期待在未来的迭代中看到更复杂、更有针对性的学习体验。
    - 一位用户强调了学习模式中的一个 Bug 或设计缺陷：系统“正在就其‘课程’中未包含的信息对我进行测验”，这引发了对 Context 跟踪不完整或这是否是促进独立研究的有意设计的担忧。这指向了当前版本中 Prompt-engineering 和模型 Context Awareness 的潜在局限性。
    - 一位教师描述了基础 ChatGPT 模型与更先进的“思考模型”（如 GPT-4o）在数学任务上的巨大性能差异，报告称其改进程度远大于从 GPT-3 到 GPT-4 的转变。这位教育工作者警告说，缺乏这些顶级模型访问权限的免费用户面临被误导的风险，因此建议使用 Gemini 2.5 Pro 等替代方案，以便在教育任务中免费获得更准确的指导，特别是对于容易出现 Hallucination 错误的学科。
    - 几位评论者指出，目前的“学习模式”实现主要是对 Prompt 策略的重新包装，而不是增加了根本性的新学习能力，未来通过更好 Fine-tuned 的专用学习模型进行迭代可以改善教育体验。这强调了当前的技术限制——主要是 System Prompt 的约束——同时暗示了向更复杂的教育模型专业化发展的路线图。

- [**学生学习模式终于上线！！**](https://i.redd.it/8h6kr7caiuff1.jpeg) ([Score: 1008, Comments: 73](https://www.reddit.com/r/OpenAI/comments/1mch6jr/study_mode_for_students_finally_available/)): **该图片似乎是一个公告或预览，显示“学习模式”现已面向学生开放——很可能是在 ChatGPT 等热门 AI 产品中。该功能似乎针对教育使用场景，支持将 AI 用于研究、学习和学习辅助，解决了此前关于 AI 对学术诚信影响的担忧。评论者还建议，新模式可以通过展示学生的工作和学习过程，而不仅仅是答案，来进一步增强功能。** 几位用户强调，鉴于 AI 与教育的日益融合，这是一项必要的发展，并建议增加“展示你的工作过程”功能，以促进理解和道德使用。如果实施得当，人们对其潜在的长期成功和教育价值持乐观态度。
    - 一位评论者提议在 ChatGPT 中为教育用途实施“展示你的工作过程”模式。该功能将记录学生的每一步过程、提出的问题以及收到的回答，允许评估者验证真实的学习情况，而不仅仅是检查答案。重点在于增加透明度并鼓励实际理解，而不是通过 AI 生成的答案来缩短学习过程。
    - 讨论点包括像 ChatGPT 这样的 AI 驱动导师对教育系统和长期结果的可能影响，例如自学者影响大学录取趋势。文中将 ChatGPT 与 Khan Academy 等平台进行了对比，强调了 ChatGPT 作为个性化、可扩展辅导解决方案的潜力，可能会重塑传统的教育路径。
    - 共识认为，整合结构化的 AI 学习工具（例如专门的学习模式）可以引导学生利用 AI 进行强化学习而非产生依赖，OpenAI 的这一初步举措表明了其将 AI 作为教育辅助元素而非替代元素的意图。

### 3. AI 对工作和社会的影响：行业预测与伦理担忧

- [**Anthropic CEO：AI 将在 3-6 个月内编写 90% 的代码**](https://www.reddit.com/r/singularity/comments/1mch6sg/anthropic_ceo_ai_will_write_90_of_all_code_36/) ([Score: 505, Comments: 212](https://www.reddit.com/r/singularity/comments/1mch6sg/anthropic_ceo_ai_will_write_90_of_all_code_36/)): **Anthropic 的 CEO Dario Amodei 预测，在他发表声明后的 3-6 个月内，AI 将负责编写 90% 的代码（最初由 Business Insider 报道）。截至目前，在还剩一个月的情况下，目前的 AI 工具（如 Copilot, ChatGPT Code Interpreter 和 Claude）仍无法完全自主地生成大部分生产环境代码，API 和 context window 限制是主要的制约因素。几乎没有定量证据支持在行业环境中由 AI 直接编写的代码份额达到 50% 或更高。** 评论者指出，Amodei 的预测可能取决于代码生成的定义（即验证“辅助”与“自主编写”是否计入），一些人认为，虽然 AI 极大地辅助了程序员，但完全自主生成仍然不常见。API 速率限制以及质量/功能问题仍然是阻碍；然而，即使是对于一年前的怀疑者来说，进步的速度也令人惊讶。
    - 由于 API 速率限制和大多数 LLM 的底层架构等技术约束，人们对 AI 在不久的将来自主编写 90% 的代码持怀疑态度。Transformer 架构 LLM 中 self-attention 机制的平方级缩放 (quadratic scaling) 被认为是根本瓶颈，需要重大突破才能实现独立的高容量代码生成。
    - 一些从业者指出，如果该说法仅指 AI 生成代码的比例（即使是作为建议或 tab 补全），那么百分比已经很高了——一位专业工程师指出，在他们的工作流中，来自 AI 的 token 级别贡献可能占新代码的 50% 以上。然而，这反映了 AI 是一种辅助工具，而不是自主编码器。
    - 讨论明确了“编写”代码是有细微差别的：目前的实现主要涉及 AI 作为高级自动补全或构思伙伴，而不是在没有人类监督或详细提示的情况下可靠地生产整个功能系统。要实现 90% 水平的真正自主，可能需要超越现有 LLM 架构的进步。

- [**扎克伯格向 Mira Murati 创业公司的十几名员工开出了高达 10 亿美元的报价，但没有一个人接受**](https://www.reddit.com/r/singularity/comments/1mcirpx/zuckerberg_offered_a_dozen_people_in_mira_muratis/) ([Score: 785, Comments: 182](https://www.reddit.com/r/singularity/comments/1mcirpx/zuckerberg_offered_a_dozen_people_in_mira_muratis/)): **据报道，Mark Zuckerberg 在一次激进的招聘行动中，向 Mira Murati（OpenAI 前 CTO）创业公司的至少十几名员工提供了高达 10 亿美元的个人薪酬方案，参考了内部讨论的截图。目标工程师或研究人员中没有一人接受该报价，这表明该创业公司内部具有极高的留存率或对使命的承诺。** 评论集中在这是由于对创业公司愿景和/或文化的坚定信念，还是对为 Meta 工作的主动反感。10 亿美元的报价被描述为极端，突显了对顶尖 AGI 人才的激烈竞争，并可能强调了对 Meta 在 AI 领域声誉的怀疑。
    - 一个主要的决策洞察是，据报道 AI 研究人员拒绝了 Meta 价值高达 10 亿美元的录用通知，可能是因为他们在 Mira Murati 的创业公司中持有大量股权，这些股权可能已经价值“每人几亿美元”。这暗示了对该创业公司估值和增长潜力的强大信心，以及领先 AI 人才的高市场价值。
    - 有推测认为，公开这些巨额报价可能是一项战略举措，旨在通过强化感知需求和团队承诺来支持该创业公司的下一轮融资。这反映了 AI 创业公司利用公关和融资叙事来提高估值并吸引投资的更广泛趋势。
- [**与 Microsoft 的新协议将允许他们在实现 AGI 后继续使用 OpenAI 的技术。**](https://www.bloomberg.com/news/articles/2025-07-29/microsoft-s-access-to-openai-tech-is-focus-of-contract-talks) ([Score: 165, Comments: 24](https://www.reddit.com/r/singularity/comments/1mcej0x/a_new_deal_with_microsoft_that_would_let_them/)): **Microsoft 向 OpenAI 提议的新协议（通过 [archive](https://archive.ph/wd8eX) 引用）将允许其在 *AGI 实现后* 仍能继续访问 OpenAI 的最新模型和技术，前提是拥有** `30-35% equity stake`**、增加非营利组织股份、减少 Microsoft 的 OpenAI 营收分成、运营自由度以及可执行的安全条款。这一框架反映了 Microsoft 在未来 AGI 开发和商业化方面与 OpenAI 战略对齐和风险共担的升级。** 评论者强调了 Microsoft 对 OpenAI 的深度依赖，并指出了 Copilot（基于 GPT-4 构建）与 ChatGPT-4 之间感知的性能差异，推测即将到来的升级（如 GPT-5）是否会弥补能力差距。
    - 讨论澄清说，根据 OpenAI 目前的合同，实现 AGI 这一阈值事件将限制 Microsoft 使用 OpenAI 模型的权利。上述 Bloomberg 报告强调了允许 Microsoft 长期或在 AGI 实现后持续访问的谈判，如果实现 AGI，这将影响 Microsoft 的技术部署权和业务连续性。
    - 针对 Copilot（企业版）和 ChatGPT-4 提出了技术对比，指出尽管两者都利用了 GPT-4，但 Copilot 的表现不如独立的 ChatGPT-4。评论预计 Copilot 可能会随着 GPT-5 的推出而得到改进，这表明 Microsoft 的产品集成与 OpenAI 的主要产品在实现或微调方面存在技术差异。
    - 注意到一个误区：虽然有人询问 GPT 是否开源，但事实上，GPT-3/4 及更高版本的模型并非开源；只有 GPT-2 等较旧的模型是开源的。这一区别对于理解为什么 Microsoft 需要合同权利，而不能在没有 OpenAI 持续合作伙伴关系的情况下简单地自行托管最新模型至关重要。

---

# AI Discord 摘要

> 由 Gemini 2.0 Flash Thinking 总结的总结之总结

**主题 1. 新兴 AI 模型与性能动态**

- **Qwen 30B 展现本地实力：** **Qwen3 30B 模型**以其速度和效率给用户留下了深刻印象，仅消耗约 **17GB RAM**，且性能有*显著提升*。用户报告称其可与 **GPT-4o** 媲美，在 Q4_K_XL 量化下仅需 **33GB RAM** 即可在本地运行，使其成为处理各种 AI 任务的切实可行选择。
- **GLM 4.5 Air 表现强劲，疑似 Gemini：** 社区成员对 [GLM 4.5 Air 模型](https://z.ai/blog/glm-4.5) 在创意写作和多语言任务中的表现表示赞赏，在 **5 bit mlx** 下达到了 **每秒 35 tokens**。尽管存在数据收集方面的担忧，但由于其回复风格以及在 SVG 和 Web 基准测试中的强劲表现，用户怀疑它可能是从 **Gemini** 蒸馏而来的。
- **GPT-5 热度点燃，Zenith 煽风点火：** 在 X 上出现一张模糊的[图片](https://x.com/ryolu_/status/1950163428389040431)后，关于 **GPT-5** 即将发布的猜测愈演愈烈，该图片声称其拥有 **1M token** 的输入窗口和并行工具调用（parallel tool calls）功能。**Zenith**（被部分人认为是 **GPT-5**）从 LLM Arena 中移除进一步推高了兴奋情绪，因为用户报告称其比现有模型有*明显的改进*。

**Theme 2. AI 开发与基础设施挑战**

- **API Key 出现 403 错误，Anthropic 刺痛用户：** Discord 上的用户报告称遇到了 **403 错误**，并显示 *"Incorrect API key provided"* 等消息，即使在账户充值后也是如此。**Anthropic** 的 API 因其严格的限制和昂贵的定价而面临沉重批评，一条详细说明每周限制的[推文](https://x.com/anthropicai/status/1949898502688903593?s=46)引发了用户愤怒。
- **LM Studio 用户发现性能骤降：** 一位用户报告称，在 LM Studio 中使用 **Qwen 3 30B A3B** 时，token 生成速度大幅下降，从 *50-80 tk/s* 暴跌至 *30-35 tk/s*。建议的修复方法包括禁用 **flash attention** 并验证其他特定设置，然后卸载并重新加载模型以恢复速度。
- **HuggingFace Tokenizers 引发用户不满：** 一位成员对 **HuggingFace Tokenizers** 提出了强烈反对，理由是在[添加自定义 tokens](https://huggingface.co/docs/transformers/tokenizer_summary)和重命名 **<unused>** tokens 时存在问题。他们惊讶地发现，使用自定义 tokens 而不将其添加到词表（vocabulary）中反而效果更好，这凸显了 Tokenizer 意想不到的行为。

**Theme 3. AI 平台与生态系统创新**

- **AgentSmith 发布：** **AgentSmith** 是一款基于 **OpenRouter** 构建的[开源 Prompt CMS](https://github.com/chad-syntax/agentsmith)，旨在简化 Prompt/Context 工程。OpenRouter 还在积极开发其**标准车道路由 (standard lane routing)**，旨在平衡价格之外的吞吐量、延迟和工具调用成功率等因素，向“最佳质量”选项迈进。
- **LlamaIndex 发布 FlowMaker 及原生 n8n 节点：** **LlamaIndex** 推出了 **FlowMaker**，这是一个用于可视化构建 LlamaIndex 工作流的 GUI 工具，可通过 [flowmaker.llamaindex.ai](http://flowmaker.llamaindex.ai/) 访问。他们还推出了新的开源 [n8n 节点用于 LlamaCloud](https://github.com/run-llama/n8n-llamacloud)，包括 LlamaCloud 索引和 LlamaParse，从而简化了智能文档处理。
- **Cursor 1.3 落地，Auto Mode 评价褒贬不一：** 新发布的 **Cursor 1.3** 版本包括[与 Agent 共享终端](http://cursor.com/changelog)和更快的编辑功能，允许用户在 Chat 中查看上下文使用情况。关于 **Auto Mode** 的报告褒贬不一，有人声称其拥有*真正无限制*的交互，而另一些人则指出 Claude 等模型的表现明显更优，并报告了 **50 美元的 API 成本**，建议在高使用量期间切换到 **GPT4.1**。

**Theme 4. AI 伦理与用户体验关注**

- **LLM 使用引发数据隐私辩论：** 成员们正在辩论使用来自 **OpenAI** 等公司的 LLM 的影响，强调了对数据收集、存储以及可能被用于定向影响或出售给数据中间商的担忧。虽然开源模型可以降低风险，但一些人警告说，即使是免费层级也可能涉及数据收集。
- **AI 网红被指存在偏向 GPT-4o 的偏见：** 成员们观察到网红在将 AI 模型与 [ChatGPT](https://openai.com/) 进行对比时存在潜在偏见，指出他们未能充分利用推理模式（reasoning modes）或在不同 Prompt 之间重置对话。一位成员专门批评了一篇 [GPT-4o](https://openai.com/index/hello-gpt-4o/) 评论，认为其没有优化每个产品的潜力，暗示缺乏深入的比较。
- **ChatGPT 学习模式引发教育辩论：** OpenAI 在 ChatGPT 中推出了**学习模式 (study mode)**，旨在通过逐步引导学生解决问题来帮助他们实现*更深层次的理解*，详见其[公告](https://openai.com/index/chatgpt-study-mode/)。成员们的反应是认为这*正趋于终局*，并且是*对 OpenAI 商业模式的违背，旨在最大限度地颠覆正式教育*。

**主题 5. AI 研究与技术的进展**

- **随着模型规模膨胀，稀疏性大幅提升：** 随着模型尺寸的增加，在固定预算下的最佳**稀疏性 (Sparsity)** 也会增加，尤其是在训练运行期间，这对于以最低成本实现最高性能至关重要。现代无丢弃的 **Mixture of Experts (MoEs)** 现在的训练速度比稠密网络快 **2.2 到 2.5 倍**，采用新方法的速度甚至达到 **3.2 倍**，由于优化，其表现优于传统的几何平均规则。
- **LLMs 作为解释 Agent 受到关注：** 成员们探讨了将 **LLMs 作为解释代理 (Interpretation Agents)** 用于自动化机械解释性 (Mech Interp) 研究，并将 **Transluce** 的工作和 [Sarah 的 MAIA 论文](https://openreview.net/forum?id=mDw42ZanmE) 作为关键资源。Neel Nanda 宣布了 **MATS 9.0** 的申请，这是一个专注于有偿机械解释性研究的导师计划，旨在产出高质量的 [研究论文](https://tinyurl.com/neel-mats-app)。
- **新的 Diffusion 学习小组启动：** 一个为期 **5 个月的研究小组**（限 12 名参与者，每周需投入 **2-4 小时**）将根据 [MIT 的课程](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf) 探索 Diffusion 模型。前 **两次介绍课程免费** 并向非成员开放，涵盖 Flow Matching 和 Diffusion 模型历史等主题。


---

# Discord: 高层级 Discord 摘要




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 停用 R1 1776，力推 Claude 4 Sonnet**：由于性能问题，Perplexity 已在其网页和移动平台退役了 **R1 1776** 模型，并建议用户切换到 **Claude 4.0 Sonnet (Thinking)** 以获得相当的实力。
   - 此次过渡旨在为专注于推理的任务提供更强大的体验，而 **Sonar** 模型目前没有变动计划。
- **Comet 浏览器用户渴求云同步**：成员们正热切期待 [Comet 浏览器](https://cometbrowser.com/) 下次更新中的云同步功能，称其缺失是将其作为主要浏览器的主要障碍。
   - 该功能备受期待，旨在增强用户体验并促进跨设备的无缝数据同步。
- **Qwen3 30B 模型以速度和效率给人留下深刻印象**：**Qwen3 30B 模型**因其速度和效率受到用户好评，一位成员报告称其仅消耗约 **17GB RAM**。
   - 该模型被认为在性能上有“相当大的跨越”，使其成为各种 AI 任务的可行选择。
- **Deep Research API 支持备受追捧**：一位成员寻求 **Deep Research API** 的支持以进一步开发产品，Perplexity 团队成员提供了 API 相关查询的协助。
   - 另一位成员报告称 **Sonar Deep Research** 返回了部分乱码输出，Perplexity 的一名成员确认团队正在调查该问题，并链接到了[已解决的工单](https://community.perplexity.ai/t/sonar-deep-research-returns-partly-garbled-output/809)。
- **AI 网红被指存在亲 GPT-4o 偏见**：成员们观察到网红在将 AI 模型与 [ChatGPT](https://openai.com/) 进行对比时可能存在偏见，指出他们未能充分利用推理模式或在提示词之间重置对话。
   - 具体而言，一位成员批评了一篇 [GPT-4o](https://openai.com/index/hello-gpt-4o/) 评论，认为其未能优化每个产品的潜力。



---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **GLM 4.5 Air 大获好评**：社区对 [GLM 4.5 Air 模型](https://z.ai/blog/glm-4.5) 在创意写作和多语言任务中的表现赞誉有加，一位用户报告在 **5 bit mlx** 下达到了 **每秒 35 个 token**。
   - 成员们将其与 **Gemini** 进行了对比，同时也承认了其作为无审查（uncensored）模型的优势。
- **TRL 更新引发混乱，降级可修复**：新的 **trl 版本** 更新导致了广泛的问题，特别是关于 `ConstantLengthDataset` 的 `ImportError`，通过回退到 **trl==0.19.1** 得到了解决。
   - Unsloth 团队针对此故障建议 Colab/Kaggle 用户删除运行时并刷新 notebook，或者使用 `pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo` 和 `pip install --upgrade --no-deps huggingface_hub`。
- **Tokenizers 问题困扰 HuggingFace**：一位成员对 **HuggingFace Tokenizers** 提出了强烈反对，理由是在 [添加自定义 token](https://huggingface.co/docs/transformers/tokenizer_summary) 和重命名 **<unused>** token 时存在问题，但发现直接使用自定义 token 而不将其添加到词表（vocabulary）中效果更好。
   - 该成员还惊讶地发现 **Windows 7** 可以通过修改在 **2025 年 12 月** 之前继续接收更新。
- **视觉编码器量化受到质疑**：在检查 **Gemma 3** 的视觉组件时，一位成员注意到卷积权重/过滤器 *v.patch_embd.weight* 仍保持为 float32。
   - 另一位成员回应称，对模型中的向量进行量化并不值得，因为向量对量化很敏感，且占模型参数比例不到 **0.5%**。
- **模型合并策略引发辩论**：讨论围绕将所有专家（experts）合并为稠密模型、交换专家以及将稠密模型转换为 **MoE** 模型展开，并重点介绍了用于 [deepseek 架构](https://github.com/deepseek-ai/ESFT) 的 ESFT 仓库，作为微调特定专家的一种手段。
   - 其意图是探索通过 frankenMoE 微调（与 *mergekit* 相关）来提升性能的前沿方法。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Zenith 移除后 GPT-5 热度加剧**：一张据称展示了 **GPT-5** 的模糊图片已 [发布在 X 上](https://x.com/ryolu_/status/1950163428389040431)，引发了猜测，而 Arena 中 **Zenith** 模型的移除进一步增强了这种猜测。
   - 一些用户声称，被部分人认为是 **GPT-5** 的 **Zenith** 相比 **GPT-4** 等现有模型有 *显著提升*。
- **GLM 4.5 是秘密的 Gemini 吗？**：用户怀疑 **GLM 4.5** 可能是从 **Gemini** 蒸馏而来的，因为它倾向于以 “Of course!” 开头回答，且回复风格较长。目前已有 [HuggingFace Space](https://huggingface.co/spaces/zai-org/GLM-4.5-Space) 可供测试。
   - 尽管存在数据收集方面的担忧，但它在 SVG 和 Web 基准测试中表现良好，引发了与以往 AI 模型的对比。
- **Qwen 代码能力进入 Arena**：**Qwen 3** coder 模型已添加到 LLM Arena 排行榜，展示了其编程技能。
   - 预计未来还将发布采用全新 A3B 架构和思考（thinking）版本的模型，以增强其能力。
- **LLM 使用引发数据隐私辩论**：成员们正在讨论使用来自 **OpenAI** 等公司的 LLM 所带来的影响，强调了对数据收集、存储以及可能被用于定向影响或出售给数据经纪人的担忧。
   - 虽然开源模型和避免关联个人身份可以降低风险，但有人表示即使是免费层级也可能涉及数据收集。
- **工具调用人为降低了基准测试分数？**：有人担心，由于厂商优先考虑工具调用（tool calling）和 Agent 能力，导致其在基准测试中的排名人为降低，正如在 **Opus Sonnet、GLM 4.5 和 KimiK2** 中所见。
   - 一些人声称，这种优先级排序出于某种原因导致许多学术型基准测试表现变差，从而影响了模型的整体评估。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **O3 更新：速度提升，智力下降？**：用户反馈 **O3** 在最近的更新中变得明显更快，但智力也显著下降，被形容为一个思考更少、更统一的版本，一些人开始尝试 **Windsurf** 作为替代方案，以获得更好的 *thinking mode*。
   - 追踪 context window 使用情况还发现消耗率很高，影响了对话性能，一名用户因长时间对话导致 context 使用率达到 **95%**。
- **Auto Mode 优于 AI 替代方案，但 API 账单令人烦恼**：关于 **Auto Mode** 的反馈褒贬不一，有人称其为“真正无限”并提供了极具挑战性的 prompting 挑战，而另一些人则认为 Claude 等模型的效果明显更好。
   - 成本担忧依然存在，一些用户报告了高达 **$50** 的 **API** 费用，并暗示在高使用量时该模式会切换到 **GPT4.1**。
- **Cursor 的代码补全难题仍在继续**：用户讨论了 **tab autocomplete** 的局限性，指出它不会读取项目规则或自定义模式，导致建议过于基础，且除非显式导入依赖项，否则缺乏跨文件建议。
   - 一些人正在利用 **READMEs** 来注入规则，尽管一致性仍然是一个问题。
- **Cursor 1.3 发布，支持终端共享和更快的编辑**：新的 **Cursor 1.3** 版本包含了 [与 agents 共享终端](http://cursor.com/changelog) 功能。
   - 用户现在可以在 Chat 中查看 context 使用情况，并期待更快的编辑速度，以及更新日志中详述的其他修复。
- **Background Agents：移动端 UI 需要改进**：用户要求改进移动端 Web 界面以管理 background agents，理由包括 **文本框不友好、diff 视图困难以及对话更新失败** 等问题。
   - 一名用户还建议支持 **Whisper 语音输入**，以便更好地进行代码转录。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **AgentSmith 发布开源 Prompt CMS**：**AgentSmith**，一个构建在 **OpenRouter** 之上的 **开源 prompt CMS**，已发布以简化 prompt/context 工程 ([GitHub](https://github.com/chad-syntax/agentsmith), [落地页](https://agentsmith.dev/))。
   - **AgentSmith** 连接到你的 **OpenRouter** 账户并使用你的额度，支持私有化部署。目前正在讨论类似于 **Claude Code YAML header format** 的客户端特定模板 ([文档](https://docs.anthropic.com/en/docs/claude-code/sub-agents#file_format))。
- **GLM 比 Kimi 更贵**：用户注意到 **GLM** 由于其长推理能力而比 **Kimi** 更贵，同时赞扬了 **Qwen3** 等开源模型在架构任务方面的进步。
   - 尽管存在价格差异，社区对开源模型的进步普遍感到兴奋。
- **Deepseek V3 遭遇 401 错误困扰**：用户报告在使用 **Deepseek 模型** 时遇到 **error 401**，暗示可能存在 API key 问题以及来自 **Chutes** 的临时停机。
   - 在 **#general** 频道中也有关于 Deepseek V3 出现 *"All providers ignored"* 错误的报告。
- **OpenRouter 平衡质量因素**：OpenRouter 正在积极优化其 **standard lane routing**，旨在平衡吞吐量、延迟和工具调用成功率，而不仅仅是价格。
   - 其目标是通过多种因素定义什么是“最佳”，而不仅仅是提供最便宜的版本，并满足用户为终端用户创建 **最佳质量选项/预设** 的需求。
- **通过 Prompt Engineering 防止不当行为？**：用户讨论了防止不当行为的方法，例如 **Deepseek V3** 中重复的句子结构，建议调整 prompt 和使用 negative prompts（如 *"never wrap up the scene"*）。
   - 一名成员链接到了一个 [Reddit 帖子](https://www.reddit.com/r/JanitorAI_Official/comments/1kd1iia/guide_day_7_how_to_prevent_deepseek_from_leaving/)，其中包含关于 prompt engineering 的潜在解决方案。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT 推出学习模式 (Study Mode)**：OpenAI 在 ChatGPT 中推出了 **study mode**，旨在通过逐步引导学生解决问题来帮助他们实现*更深层次的理解*，详情见其[公告](https://openai.com/index/chatgpt-study-mode/)。
   - 新模式专注于培养对学科内容更深刻的理解，而不是简单地提供答案。
- **Copilot Vision 大放异彩**：根据 [drinkoblog.weebly.com](https://drinkoblog.weebly.com) 的报道，成员们对 Edge 浏览器中的 **Copilot Vision** 印象深刻，称赞其 UI 设计出色、似乎可以无限使用且结果相对准确。
   - 同一位用户继续在 [drinkoblog.weebly.com](https://drinkoblog.weebly.com) 上赞美 Edge 中的 Copilot Vision，特别是它现在的酷炫程度。
- **GPT-5 发布传闻四起**：有关 **GPT-5** 即将发布的传闻在流传，据称它拥有 **1M token** 的输入窗口、**100k** 输出 token、支持 **MCP**、并行工具调用 (parallel tool calls) 以及动态短+长推理 (dynamic short + long reasoning)。
   - 然而，其他用户要求提供*可靠来源*来支持这些说法。
- **Zenith 被视为编程冠军**：一位用户根据个人测试和分享的案例，预测 **Zenith** 将成为顶级的编程模型。
   - 该用户对预测进行了限定，表示在*另一个模型出现之前*这将是事实。
- **记忆格式提示词 (Memory Format Prompt)**：一位成员分享了一个新的 **AI 记忆格式**提示词，包含 **Token Embeddings**、**Semantic Grouping** 和 **Binary Encoding** 等概念。
   - 该格式旨在实现**速度**、**效率**和**保真度**，同时使内容对人类不可读，并整合了 **Header Blocks**、**Global Embeddings** 和 **Update Deltas** 等要素。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Voxtral Mini 面临兼容性问题**：成员们发现目前无法在 **LM Studio** 中使用 [Voxtral-Mini-3B-2507-GGUF](https://huggingface.co/bartowski/mistralai_Voxtral-Mini-3B-2507-GGUF)。
   - 据用户反映，该模型尚未得到支持，未提供更多细节。
- **LM Studio 性能骤降**：一位用户报告 **Qwen 3 30B A3B** 的 token 生成速度大幅下降，从 *50-80 tk/s* 降至 *30-35 tk/s*，并在 **LM Studio general 频道**寻求建议。
   - 建议的修复方法包括禁用 **flash attention** 并验证其他特定设置，然后卸载并重新加载模型。
- **OpenWebUI 与 LM Studio 连接**：用户确认了 **OpenWebUI** 与 **LM Studio** 之间的兼容性，并提供了 **Docker** 设置的配置技巧，例如从 LM Studio 获取 **LLM base URL**。
   - 一位用户通过在 LM Studio 中启用 **CORS** 并确保使用正确的 IP 地址解决了连接问题。
- **LM Studio 获得离线 TTS 语音**：一位用户分享了一个 Python 脚本，用于[桥接 LM Studio API 与 XTTS WebUI API](https://cdn.discordapp.com/attachments/1110598183144399058/1399764223696830474/bridge-LMStudio-XTTS.py?ex=688a2f85&is=6888de05&hm=eec4e5dcb6b55bf09ee4282441d1fa35a166fd0392ff1c81116c964188a51f16&)，从而在 LM Studio 中实现**离线 TTS 语音集成**。
   - 此集成允许直接在命令行界面播放音频响应。
- **AMD Ryzen AI MAX+ 表现不佳**：一位用户分享了一段[视频](https://youtu.be/BTIUL-yaY4s?t=487)，显示配备 **128gb** 内存的新款 **AMD Ryzen AI MAX+ 395** 在运行 **Llama 70b Q6** 时仅达到 **2-3 T/s**，且只有一半内存分配给了 GPU。
   - 用户提到，*这种性能使其与 RTX 30xx Nvidia GPU 相比，并不是一个真正可行的选择*。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **DeepSeek R1 推动 AI 民主化**：据报道，[DeepSeek R1 的发布](https://deepseek.com/blog/deepseek-r1)推动了 AI 领域的民主化，大幅降低了此前*每 1M token 数美元*的推理成本。
   - 尽管功能强大，但据称 **Deepseek R1** *缺乏 Agent 能力*。
- **API Key 抛出 403 错误**：用户在账户充值后仍遇到 **403 错误**，提示消息为 *"<xxxxxx> is not active, Incorrect API key provided"*。
   - 解决方法包括等待、重新生成 API Key 以及切换到 **OpenRouter**；该错误被特别指出出现在 **Claude code** 中，而非 playground。
- **Kimi K2 激发存档欲望**：一位用户正在多 TB 的设备上存档 **Kimi K2**，因为它具有独特的“灵性（spark）”，使其在开源和闭源模型中脱颖而出。
   - 他们表达了深深的钦佩，称 *“即使 Kimi 发生了什么不好的事情，它仍然属于我”*，并称赞了它的“氛围（vibes）”和“灵魂（soul）”。
- **Kimi 对 TB 级存储的巨大需求**：根据量化（quantization）程度的不同，**Kimi K2** 的磁盘空间需求在 *200 GB 到 2 TB* 之间。
   - 建议的存储方案包括 **Azure Blob Storage**，预计 *3TB 数据的费用为 6 美元/月*。
- **Kimi 学习使用表情符号**：一位用户正在积极教他们的 **Kimi** 机器人使用表情符号并调整任何意外行为，将其构想为一个“小妹妹（lil sis）”。
   - 该用户期待 **Kimi Agent 模型**的发布，并预见到 **Kimi distil** 将引发关注。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 的 Video Overviews 幻灯片功能**：Google 的 **Video Overviews** 功能正在向 Web 用户推出，包含由 AI 驱动的图像生成、文本生成和视频合成；详见 [帮助中心文章](https://support.google.com/notebooklm/answer/16213268?hl=en&ref_topic=16175214&sjid=12603864792385823108-NA#:~:text=With%20NotebookLM%2C%20you,daily%20video%20generations.)。
   - 初始反馈显示，该功能目前生成的是**文本和 AI 图像的幻灯片**，而非动画视频，且 Pro 用户每天有 **20 次视频生成**的频率限制。
- **Gemini Agent 框架发布**：一位成员分享了一个[原型 Gemini Agent 框架](https://cdn.discordapp.com/attachments/1124403655819415592/1399853283404939454/gemini-agentic-framework.zip?ex=688a8276&is=688930f6&hm=d111cb9b690a7969af3e128a78c205e2d89bf790babf9c1ab08c72cdc9f89ead)用于测试和实验。
   - 其功能不保证 100% 可用。
- **Studio UI 更新**：新的 **Studio UI** 正在推出，功能包括从相同来源创建多个输出，以及在创建输出时选择来源。
   - 这是近期一系列发布的一部分，包括 **Featured Notebooks** 已全面开放，以及 **Video Overviews** 最初仅支持**英文**和**桌面端**。
- **PDF 上传故障**：一些 NotebookLM 付费用户报告在上传 PDF 时收到 **“error uploading source, try again”** 的消息。
   - 一名 Google 员工确认了该 **bug** 并表示正在调查。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **新款 Zenith 模型引发猜测**：根据 [Reddit 帖子](https://reddit.com/r/LocalLLaMA/comments/1m9holp/theres_a_new_kimi_model_on_lmarena_called_zenith/)，LM Arena 上出现了具有出色 "vibe coding" 能力的新匿名 AI 模型（**Zenith**、**Summit**、**Lobster**），引发了关于它们是 **OpenAI 的 GPT-5** 变体还是 **Kimi-2** 的猜测。
   - 同一帖子中的一条评论*几乎确认*了 `zenith` 是 OpenAI 的模型，因为*它使用了与 gpt-4o、o3 和 o4-mini 相同的 tokenizer*。
- **LlamaIndex Agent 现可使用 Oxylabs 进行抓取**：**LlamaIndex** 与 **Oxylabs** 集成，允许用户通过针对 Google、Amazon 和 YouTube 的[专用读取器](https://xcancel.com/llama_index/status/1949937947727245639)，创建高性价比的 AI Agent，用于实时网页搜索和抓取。
   - 这将有助于 Agent 以更好的数据质量和规模进行实时数据访问与检索。
- **AI 定价仍是一个谜**：一位产品开发者发现很难为他们的 AI 产品（一个 PR 审查工具）定价，强调了**计量（metering）**方面的挑战以及对灵活定价的需求，正如 [Lenny's Podcast 关于产品增长的内容](https://podcasts.apple.com/us/podcast/lennys-podcast-product-growth-career/id1627920305?i=1000719362528)中所链接的那样。
   - 该讨论涉及定价模型（**固定 vs. 浮动**、**按 PR 计费**、**速率限制**）以及用户对基于 Token 定价的困惑。
- **Fireworks 估值达 40 亿美元**：据[报告](https://xcancel.com/ArfurRock/status/1950222707116707990)显示，**Fireworks AI** 正在以 **40 亿美元估值**进行股权融资，拒绝了来自 Coreweave 和 Meta 的并购要约；该公司 **ARR 约为 1.3 亿美元**，**同比增长超过 20 倍**，且已实现盈利。
   - 该公司表示其在 AI 算力和模型领域正开足马力飞速发展。
- **Arcee.ai 发布多功能 AFM-4.5B**：**Arcee.ai** 在 Hugging Face 上发布了 **AFM-4.5B** 和 **AFM-4.5B-Base 模型**，旨在提供灵活性和高性能，并通过与 **DatologyAI** 的数据合作强调质量，正如 [X 上的公告](https://xcancel.com/LucasAtkins7/status/1950278100874645621)所述。
   - 该模型在多种任务中具有通用性，并针对各种应用中的易用性进行了优化。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Qwen 30B 与 GPT-4o 竞争**：讨论了 **Qwen 30B 模型**的发布，声称其可与 **GPT-4o** 媲美，并可在本地以 **33GB RAM** 运行。
   - 分享了 [Hugging Face](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF) 上 **Unsloth Qwen3-30B-A3B-Instruct-2507-GGUF** 模型的链接，另一位用户指出在 Q4_K_XL 量化下需要 17GB RAM。
- **扩散模型学习小组启动**：一个为期 **5 个月**的学习小组（限 12 人，每周需 **2-4 小时**）将根据 [MIT 的课程大纲](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf)探索扩散模型（Diffusion Models）。
   - 前**两场入门课程免费**并向非成员开放：8 月 2 日关于 Flow Matching 和扩散模型，8 月 9 日关于 PDEs、ODEs、SDEs，均为美国东部时间中午 12 点（[在此报名](https://lu.ma/kv8zf6va)）。
- **ragflow 在生产环境中的调研**：一位用户询问了在生产环境中使用 **ragflow** 的经验，特别是潜在的问题和通用适用性。
   - 对话被引导至专门频道，并提供了 [ragflow GitHub 仓库](https://github.com/infiniflow/ragflow)链接。
- **模型面临上传问题**：一位成员指出他们忘记在 Hub 上上传模型的**自定义模型类**（架构），这意味着*目前所有模型都无法正常加载*，基本上处于不可用状态。
   - 他们正在从头开始重建所有内容，并提供正确的架构文件、更好的文档和完善的推理支持。
- **ViT/ResNet 用于脱发图像分类**：成员们建议使用 **ViT** 或 **ResNet** 等视觉模型，而不是使用 LLM，来根据 **Hamilton-Norwood 量表**对男性型脱发照片进行分类。
   - 一位成员提供了相关链接，例如[这篇文章](https://pmc.ncbi.nlm.nih.gov/articles/PMC10974725/)。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **LLM 作为解释 Agent 受到关注**：成员们正在探索将 **LLM 作为解释 Agent** 的用途，并参考了 **Transluce** 的工作和 [Sarah 的 MAIA 论文](https://openreview.net/forum?id=mDw42ZanmE) 作为关键资源。
   - 其核心想法是利用 **LLM** 进行自动化的机械解释性（mech interp）研究，并基于该领域的现有知识进行构建。
- **Modelscope 成为中国版的 Hugging Face**：**Modelscope.cn** 被描述为 **Hugging Face** 的中国替代品，在中国境内提供 AI 模型和工具。
   - 正如[这篇文章](https://www.semafor.com/article/10/20/2023/ai-platform-hugging-face-confirms-china-blocked-it)所证实的，由于限制，**Hugging Face** 在中国无法访问。
- **探索同伴压力对 AI 的影响**：一位成员分享了关于 **AI 同伴压力** 的研究预印本，分析了超过 **200** 场 AI 对 AI 的对话，以研究模型的复杂性和易受影响程度。
   - 该研究达到了 **121%** 的统计功效，可以在 [Zenodo](https://zenodo.org/records/16573783) 上获取，并正在积极征求反馈。
- **MATS 9.0 招募机械解释性研究人员**：Neel Nanda 宣布 **MATS 9.0** 现已开放申请，这是一个专注于有偿机械解释性（mech interp）研究的导师计划。
   - 该计划旨在指导参与者在该领域产出高质量的[研究论文](https://tinyurl.com/neel-mats-app)。
- **新的 Diffusion 模型学习小组启动**：一个新的 **Diffusion 模型学习小组** 开启了为期 **5 个月** 的计划，共有 **12** 名成员，参考了 [MIT 的课程](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf)。
   - 课程包括 2 小时的直播课和 2 小时的自学。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **B200 裸金属服务器热潮**：一位成员正在寻找提供单卡 **NVIDIA B200** GPU 裸金属服务器的云服务商，以便修改内核驱动程序并配置 **GPU**。
   - 这引发了关于潜在用例的讨论，特别是对 **ncu (NVIDIA Compute Utility)** 支持的需求，并询问了拥有单卡 **B200** 实例的云服务商。
- **Triton 的 Torch 技巧：代码提取建议**：用户探索了提取由 **torch.compile** 生成的 **Triton** 和 **PTX 代码** 的方法，一位用户分享了使用 `TORCH_LOGS="output_code" python your_code.py` 即可输出 PTX 代码。
   - 一位成员建议检查 `compiled_kernel.asm.keys()` 字典，并指向[这篇博客文章](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/?utm_source=chatgpt.com#ttir-triton-ir)以获取更多细节，而其他人则讨论了如何强制 **torch inductor** 仅生成 **Triton 代码**。
- **DTensor 亮相：直播即将开始**：一位成员宣布将通过直播更详细地了解 **DTensor**，并分享了直播的 [YouTube 链接](https://www.youtube.com/watch?v=b8lplOf2g4g&ab_channel=MarkSaroufim)。
   - 另一位成员分享了一个 [gist](https://gist.github.com/S1ro1/4fe38e3a0fff3d84314935a0e05aed9c)，修复了一个权重初始化错误，该错误曾导致随机初始化在每个 rank 上产生不同的分片（shards）。
- **解决 Triton 棘手的 TMA 问题**：一位用户询问 **Triton 编译器** 是否能够执行带有 **乒乓调度（ping-pong schedule）** 的 **GEMM**（通用矩阵乘法）。
   - 这取决于 Triton 编译器的版本，因为 TMA (Tensor Memory Accelerator) 支持在正式版本中尚未提供，建议等待 **3.4.0** 版本。
- **denvdis 剖析 CUBIN 难题**：一位成员创建了一个名为 **denvdis** 的工具，用于提取和替换 **ELF fatbinaries** 中的 **CUBIN** 文件，该工具可以在[这里](https://github.com/redplait/denvdis/tree/master/fb)找到。
   - 替换的文件必须与原始文件大小相同，不支持压缩的 fatbinaries，且*不依赖于 NVIDIA SDK*。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **MoE 模型因效率优势主导 API**：社区讨论了参数量约 **500B** 的 **MoE** 模型相对于约 **30B** 的稠密（dense）模型的兴起。一位成员指出，由于效率提升，去年几乎所有的 API 模型都是 **MoE** 模型。
   - 虽然 **MoE** 模型在基准测试中表现出色，但讨论表明稠密模型可能捕捉到更多细微差别，但更难进行微调，而 **MoE** 的效率提升实在令人难以忽视。
- **本地 LLM 微调者面临瓶颈**：据报道，本地 **LLM** 微调者*受困于 gemma 3 和 qwen 3*，难以获取适合本地微调/运行的 **10-70B** 模型，而不得不求助于 API。
   - 一位成员建议，为了获得更好的效率，本地开发可能会转向 **MoE** 模型。
- **Anthropic API 因严格限制遭到抵制**：成员们严厉批评了 **Anthropic** API 的限制性约束和昂贵的定价，并引用了[这条推文](https://x.com/anthropicai/status/1949898502688903593?s=46)，其中详细说明了每周的限制。
   - 有人担心，如果出现更优的替代方案，**Anthropic** 的这种做法可能会导致其走向衰落：*为使用 claude 等待半天已经很糟糕了，还要等整整一周？*
- **Qwen3-30B-A3B-Instruct-2507 终于发布**：[Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) 模型已正式在 Hugging Face 上发布。
   - 这是在之前的一次失误之后发布的，标志着它正式面向更广泛的社区开放。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **随着模型规模扩大，稀疏性（Sparsity）激增**：随着模型规模的增加，固定预算下的最佳**稀疏性**也会增加，尤其是在训练运行期间。这使得它成为以最低成本实现最高性能、加快训练速度并增加总参数量的理想选择。
   - 虽然**稀疏性**可能会阻碍拥有大量 **GPU** 资源和训练预算的终端用户的性能，但它在降低实现最高性能的**金钱成本**方面表现出色；一位成员还分享了 [32K 上下文长度是否足够](https://x.com/LTSI_TheCaptain/status/1950272036074582040)的链接。
- **现代 Dropless MoE 占据主导地位**：由于多项优化，现代 Dropless **Mixture of Experts (MoEs)** 的表现优于几何平均规则（`effective parameter count ~= sqrt(active * total parameters)`）。
   - **MoE** 的训练速度比具有相同参数量的稠密网络快 **2.2 到 2.5 倍**，新方法的速度甚至达到 **3.2 倍**，**Baidu** 报告 **ERNIE-MoE** 训练的 **MGH** 约为 **50%**。
- **Kimi K2 挑战 Claude 的统治地位**：尽管模型更稀疏，但 **Kimi K2** 在与以大型稠密模型著称的 **Claude** 竞争中表现出色。
   - 这种竞争优势可能源于广泛的**强化学习 (RL)**，而非纯粹的架构优势。此前，**RL** 曾使 **Claude** 在 **Agent** 任务上具有优势。
- **YouTube Shorts 疏远核心用户**：成员认为 **YouTube** 强调 **Shorts** 是为了对抗 **TikTok**，但未能意识到其用户群并不喜欢 TikTok，且 **Shorts** 产生的收入显著降低。成员表示这*更多是为了市场份额。人们把时间花在非 YouTube 的视频平台上，这意味着收入损失，这是完全不可接受的*。
   - 一位成员表示，**TikTok 的推荐算法**虽然更简单，但*比 YouTube 的彻底得多*。
- **ChatGPT 学习模式：终局之战？**：成员们对 [OpenAI 的 ChatGPT 学习模式](https://openai.com/index/chatgpt-study-mode/)做出了反应，认为它*正向终局迈进*。
   - 另一位成员认为*这违反了 OpenAI 的商业模式，该模式旨在最大限度地颠覆正规教育*。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **区块链在授权与支付方面受到青睐**：由于提供了统一的账户和精确的控制，与 Web2 相比，在 **blockchain** 上实现授权管理和支付护栏（payment guardrails）被认为更加容易。
   - 一位成员建议像应用商店一样，统一不同模型提供商之间的支付解决方案，特别是针对本地或自托管模型，从而简化交易和访问流程。
- **Monday.com 招募 AI 工程梦之队**：[Monday.com](https://www.monday.com) 聘请了两名 AI 工程师全职从事 **MCP** 工作，正如最初在[这里](https://www.calcalistech.com/ctechnews/article/rjorfh8wel)报道的那样，这巩固了他们对该项目的投资。
   - 新员工预计将加速开发，并为 **MCP** 在 **Monday.com** 生态系统中的能力带来全新视角。
- **MCP Server 部署面临连接考验**：一位成员将 **MCP server** 部署到配置了正确 SSL 和域名设置的 **EC2** 上，但在使用 **Claude Desktop** 时遇到了连接问题。
   - 虽然 **Cursor** 连接成功，但 **Claude Desktop** 失败了，这凸显了与特定客户端之间潜在的兼容性问题。
- **一键式 VS Code MCP Server 安装上线**：网站 [vscodemcp.com](https://vscodemcp.com/) 现在提供了一个**一键安装按钮**，用于将 **MCP Server** 添加到 **VS Code**，降低了开发者的准入门槛。
   - 安装程序附带了一个[演示视频](https://youtu.be/1JcRtQxmh3I)，引导用户完成设置过程并展示了该扩展的优势。
- **Nexus 作为 AI 工具的移动应用商店发布**：**Nexus** 的 Alpha 版本发布，这是一个针对 AI 工具（MCP servers）的移动应用商店，具有**一键安装**、**无需 JSON 配置**和**聊天集成**等功能。
   - **Nexus** 可在 [getnexus.in](https://www.getnexus.in) 获取，源代码托管在 [GitHub](https://github.com/lucky-builds/get-nexus)，旨在简化移动平台上 AI 工具的发现和部署。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **优化器以“哥布林”风格亮相**：一个新的 **DSPy optimizer** 引起了轰动，并被幽默地插画为一个[绿哥布林](https://cdn.discordapp.com/attachments/1203568372667645963/1399545934190477383/3195gc.png?ex=688a0cf9&is=6888bb79&hm=b5ef1dc40c8d2e735f8370a1be34553a2fd7cb46d86a6e67adab2b7ec0350fc3&)。
   - 图片暗示该优化器“好到不愿分享”，引发了人们对其潜在实力的猜测。
- **GEPA 引发对指标的好奇**：对 [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457) 的兴趣促使了对对比指标的需求。
   - 论文作者就在频道中，被认为是直接咨询的资源。
- **提议进行工具反馈优化**：有人建议不仅要优化 Prompt，还要在发生错误时优化**工具响应反馈（tool response feedback）**。
   - 实验证实了可行性，使用外部编译器、分析器和运行时执行器作为提供文本反馈的工具。
- **深入探讨 DSPy 的可学习参数**：讨论围绕 `dspy.Variable` 和 `dspy.Parameter` 展开，它们被描述为 **DSPy** 程序中“某种可学习参数”。
   - 有提议认为 `dspy.Variable` 可以让用户指定可优化的元素，甚至可能允许 **DSPy** **测试并重写其自身的部分源代码**。
- **AI 工程师加入讨论**：一位专注于 **agentic systems**、**工作流自动化**和**可扩展无代码/低代码架构**的高级 AI 工程师提供了他们的专业见解。
   - 他们列举了如 **Vapi AI**、**Make.com** 和 **OpenAI** 等工具，并表示愿意在设计和部署 AI Agents、自动化业务运营以及微调 LLM 和 LSM 方面提供帮助。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 的 `external_call` 直接调用 C 函数**：Mojo 的 `external_call` 函数支持直接调用任何 **C ABI 函数**，只要该符号在当前地址空间内已链接。
   - Mojo 团队主张尽量减少 C 代码的使用，在等效的 Mojo 代码可行时需要强有力的理由，以保持更好的 **可移植性**。
- **Mojo 引入文件描述符**：Mojo 通过 `io.file_descriptor.FileDescriptor` 引入了文件描述符功能，旨在减少在这些功能中对 `external_call` 的依赖。
   - 在 OS 层级使用 `read` 和 `write` 操作增强了可移植性，符合 Mojo 减少 **外部依赖** 的目标。
- **标准库开发无需编译器修改**：大多数功能可以在不修改编译器的情况下在 **Mojo 标准库** 中实现。
   - 这种方法利用了 Mojo 的适应性和 FFI 能力，仅依赖于编译器调用 *具有 C ABI 的符号* 的能力。
- **Mojo 模块名称趋向标准化**：一个 [功能请求](https://github.com/modular/modular/issues/5094) 旨在标准化 Mojo 模块的命名规范，将其与 Python API 解耦。
   - 该倡议寻求减少混淆，并绕过 Python 沿用 30 年的命名规范所固有的限制，以实现 **模块名称的一致性**。
- **Max 即将移除 PyTorch 2.7 依赖**：**PyTorch 2.7 依赖** 将很快在下一个 Nightly 构建中移除，使用户能够自由选择其 PyTorch 版本。
   - 团队认为 **2.0** 是 PyTorch 与 Max 兼容的现实下限，尽管最低固定版本为 **2.5**，这赋予了用户管理其 **PyTorch 环境** 的能力。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **FlowMaker 成为关注焦点**：**LlamaIndex** 推出了 **FlowMaker**，这是一个用于可视化构建 **LlamaIndex 工作流** 的 GUI 工具，可通过 [flowmaker.llamaindex.ai](https://flowmaker.llamaindex.ai/) 访问，并在 [此视频](https://youtu.be/MU6jA0rUlFY?feature=shared) 中进行了演示。
   - 虽然有用户希望支持 **Python** 导出，但目前提供的是 **Typescript** 导出。
- **LlamaCloud 节点现已原生支持 n8n**：**LlamaIndex** 推出了开源的 **LlamaCloud n8n 节点**（包括 LlamaCloud 索引、LlamaParse 和 LlamaExtract），可在 [`n8n-llamacloud` 仓库](https://github.com/run-llama/n8n-llamacloud) 中获取，并在 [此视频](https://youtu.be/5bQXHPSkuBw?feature=shared) 中进行了演示。
   - 这一增强功能简化了现有 **n8n 工作流** 中的智能文档处理，并简化了嵌入式内容管理，用户无需再自行管理 API 密钥。
- **Agent 真正具备可操作性**：即将举行的网络研讨会将演示如何通过由 **LlamaCloud** 解析能力驱动的 AI 驱动文档 Agent，将复杂的财务文档转化为可操作的数据（[链接](https://t.co/f0TKSzHQ2d)）。
   - Seldo 在 @aiDotEngineer 峰会上分享了可扩展的 Agent 设计模式，包括 **混合工作流** 和 **可调试性**，可通过 [此链接](https://t.co/zeTaVOMate) 访问。
- **LlamaCloud PDF 处理难题依然存在**：一位成员报告称 **LlamaCloud** 无法通过 **API** 检测和处理 **PDF 文件**（使用 **n8n** 简化工作流），并寻求帮助，包括一张 [截图](https://cdn.discordapp.com/attachments/1059201661417037995/1399848961832911031/Screenshot_2025-07-29_at_22.13.16.png?ex=688a7e70&is=68892cf0&hm=213304e9d8a77128ab3cbc75d4c9114a73d0a157e12a0aa633bd2a62e160a5fa)。
   - 建议在与 **LlamaCloud** 配合使用时，确保文件名包含正确的 **文件扩展名**。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Grok 在提示词生成方面表现不佳**：一位成员建议利用 **Grok** 为 **Manus AI** 生成提示词（Prompt），结果发现效果很差。
   - 尽管结果不佳，另一位成员主动提出亲自协助完成该任务。
- **Manus 积分系统遭到质疑**：成员们批评了 **Manus 的积分系统** 和不频繁的更新，认为这些因素尽管有活动举办，仍可能导致用户流失。
   - 一位成员建议探索替代的 Agent 系统，如 **Lume**，以评估其对比价值。
- **Lume 在编程方面超越 Suna**：在 **Lume** 和 **Suna** 的对比中，一位成员嘲讽道 *Lume 就是更差的 Suna*。
   - 然而，另一位成员发现 **Lume** 在编程任务中表现更优，理由是错误更少且代码经过调试，同时也承认 **Manus** 在漫画创作方面的优势。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad Tensor 的本质揭晓**：一位成员询问 **tinygrad** 的 **Tensor** 类是封装了现有对象（如 **NumPy ndarrays**），还是拥有自己的实现。
   - 讨论还涉及了如果 **tinygrad** 使用封装器，在性能补偿方面的问题。
- **PR #11410 在无解释的情况下被关闭**：一位成员对 [PR #11410](https://github.com/tinygrad/tinygrad/pull/11410) 在更新后不久就被关闭且没有任何评论表示惊讶。
   - 另一位成员回应称，*该 PR 没抓到重点，不是一个好的改动*，并建议贡献者回顾过去的合并记录以了解准则。
- **"Where" 操作引发辩论**：在 geohot 发表评论后，一位成员在尝试将分配的操作保持到 kernel 化/调度创建（kernelization/schedule creation）之后，重新考虑使用 *where* 操作。
   - 在承认潜在副作用的同时，他们对 PR 在没有反馈的情况下被关闭感到惊讶，尤其是他们原本计划进行更深入的调查。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **BFCLv4 仅接受模型提交**：一位成员询问 **BFCLv4** 是否允许开源一个 Agent 系统或提供 API key，但被告知目前提交仅限于单个模型。
   - 这一限制使排行榜专注于评估核心模型的性能，而非其周围的 Agent 基础设施。
- **多 Agent 系统提交被拒绝**：群组中提出了关于提交包含多个模型的多 Agent 系统的问题。
   - 回复确认 **BFCLv4** 的提交仅限于单个独立模型，以维持标准化的评估框架。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 寻求 LoRA Adapter**：一位成员询问 **Torchtune** 是否支持保留现有前向计算路径的 **LoRA-style adapter**。
   - 用户希望冻结原始模型权重，并通过额外的可训练层应用更新，而不改变计算成本。
- **RL 测试耗时过长，疑似存在 Bug**：一位成员注意到 **RL 测试** 运行超过 **1 小时**，将其归因于一个 Bug，并提出了一个单独的 **PR** 用于调试 **CI**。
   - 该专门的 **Pull Request (PR)** 将专注于调试 **Continuous Integration (CI)** 系统。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Diffusion Models 学习小组启动**：一个新的学习小组正在组建，计划在 **5 个月** 内从零开始探索 **diffusion models**，参考 [MIT 的课程大纲](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf) 作为指南。
   - 该小组计划每周投入 **2-4 小时** 学习 **generative AI** 中的核心架构。
- **Diffusion Models 免费入门课程开放**：两场免费入门课程定于 **8 月 2 日** 和 **8 月 9 日**（EST 时间中午 12 点）举行，涵盖 **Flow Matching**、实际应用案例、**PDEs**、**ODEs**、**SDEs** 以及 diffusion models 的历史（[课程链接 1](https://lu.ma/kv8zf6va)，[课程链接 2](https://lu.ma/uk6ecrqo)）。
   - 这些课程旨在提供 **diffusion models** 领域基本概念和应用的概览。
- **学习小组吸引 AI 专业人士**：diffusion models 学习小组吸引了各类 AI 专业人士，包括 **某 AI 电影工具的 CTO**、**AI 艺术讲师**、**LLM 讲师**和 **AI 研究员**。
   - 前 **两场课程免费**，之后采用订阅模式，早鸟价为 **$50/月**（之后为 **$100/月**），用于支付助教费用，特色包括同行领导的课程、导师 Q&A、实战项目和真实的研究论文。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Nomic 数据集访问被拒**：一位成员报告在尝试按照 [contrastors 仓库](https://github.com/nomic-ai/contrastors) 的说明访问 **nomic-ai/nomic-embed-text-v2-moe** 数据集时出现 **AccessDenied 错误**。
   - 该错误发生在执行命令 `aws s3 ls --endpoint-url=https://9fa58365a1a3d032127970d0bd9a1290.r2.cloudflarestorage.com/ s3://contrastive` 期间的 `ListObjectsV2` 操作中。
- **低配系统寻求模型**：一位拥有 **Celeron N2815**、**4GB RAM** 且无 GPU 的成员请求建议适合在其系统上运行的模型。
   - 提供的消息中没有推荐具体的模型，表明需要进一步的社区建议。

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **社区欢迎新面孔**：新成员正在 Cohere 的社区 Discord 服务器上进行自我介绍，分享他们的**公司、行业和大学**，详细介绍他们当前的项目以及最喜欢的技术/工具。
   - 他们还分享了对社区参与的愿景以及希望从社区中获得什么。
- **多样化背景丰富社区**：新成员来自广泛的背景，包括各种**公司、行业和大学**。
   - 这种多样性有望为社区的讨论和协作带来不同的视角和专业知识。



---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。


---


**Codeium (Windsurf) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。


---

# Discord: 频道详细摘要与链接





### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1399531884093898872)** (1 messages): 

> `R1 1776 移除, Claude 4.0 Sonnet, Sonar 模型` 


- **R1 1776 模型退役**：由于未能跟上最近的改进步伐，**R1 1776** 模型将从网页端和移动端的模型选择器中移除。
   - 建议用户尝试 **Claude 4.0 Sonnet (Thinking)**，它在提供类似优势的同时拥有更强的性能。
- **Sonar 模型保持不变**：**Sonar** 模型或其他任何模型均未做改动。
   - 重点是引导用户从 **R1 1776** 过渡到 **Claude 4.0 Sonnet** 以处理侧重推理的任务。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1399471250916507679)** (1101 messages🔥🔥🔥): 

> `Comet 浏览器云同步, Qwen3 30B 模型, 统一内存, Nvidia 5080, OpenRouter` 


- **Comet 用户呼吁云同步**：成员们希望 [Comet 浏览器](https://cometbrowser.com/) 在下次更新中加入云同步功能，因为目前缺失该功能阻碍了它成为他们的主浏览器。
- **用户发现 Qwen3 30B 模型响应迅速**：用户发现新的 **Qwen3 30B 模型** 有了*相当大的飞跃*，且速度惊人，仅消耗约 **17GB RAM**。
- **配备统一内存的笔记本电脑运行 AI 模型**：一些成员在配备 **32GB 统一内存** 的笔记本电脑上运行 **30B 量化模型**，达到了每秒约 **40 tokens**。
- **AI 网红被发现存在偏见**：成员们注意到，网红在与 [ChatGPT](https://openai.com/) 进行对比时可能存在偏见，通常未能使用推理模式或未针对每个查询开启新对话。
   - 一位成员指出了一段特定的评测，其中 YouTuber 推荐使用 [GPT-4o](https://openai.com/index/hello-gpt-4o/)，但并未充分利用每个产品的潜力，不过*我在推荐那段评测时也有点偏见🔥*。
- **Kimi 对比 O3**：用户讨论了对 **Kimi** 的偏好，原因是其*个性*以及在特定 Prompt 下的表现，而其他人则推荐将 **O3 Pro** 用于研究和学习。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1399617401087328317)** (4 messages): 

> `Perplexity Deep Research API 支持, Sonar Deep Research 输出问题` 


- **寻求 Perplexity Deep Research API 支持**：一位成员询问如何联系 Perplexity 的人员，以讨论用于产品开发的 **Deep Research API**。
   - 另一位来自 Perplexity 的成员作出了回应，表示愿意协助解决有关 **API** 的任何问题。
- **Sonar Deep Research 中的乱码输出问题**：一位成员报告了 **Sonar Deep Research** 返回部分乱码输出的问题，Perplexity 的成员确认团队正在调查该问题，并链接到了[已解决的工单](https://community.perplexity.ai/t/sonar-deep-research-returns-partly-garbled-output/809)。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1399466477769330799)** (757 条消息🔥🔥🔥): 

> `vibe coding, WASM, FIPS 140-3, trl 导致一切崩溃, GLM 4.5 Air` 


- **Vibe Coding**: 几位成员讨论了 "**vibe coding**" 及其影响，其中一位成员提到他们正在使用 [Nabla](https://github.com/Atelier-Logos/nabla/pull/49) 对 **FIPS 140-3** 进行 vibe coding，另一位成员则称 **Lovable** 是 "vibe coded" 的产物。
   - 讨论中还涉及了在安全上下文中使用 vibe coding 的风险，因为理解底层漏洞至关重要，此外还提到了 OpenAI 制作觉醒（woke）DEI 模型的话题。
- **trl 新版本导致一切崩溃**: 有报告称新的 **trl 版本** 导致一切崩溃，报错信息为 `ImportError: cannot import name 'ConstantLengthDataset' from 'trl.trainer.utils'`。
   - 成员们建议的修复方法是[回退 torch 版本](https://discord.com/channels/1179035537009545276/1179777624986357780/1399629661796827206)。
- **GLM 4.5 Air 模型备受关注**: 许多成员一直很喜欢用 [GLM 4.5 Air 模型](https://z.ai/blog/glm-4.5) 进行创意写作和多语言任务，其中一位成员在 **5 bit mlx** 下达到了 **每秒 35 个 token** 的速度。
   - 其他成员将该模型的性能与 Gemini 进行了对比，并称赞其作为无审查模型的强大实力。
- **Unsloth Runpod 模板即将推出**: Unsloth 将发布一个 Runpod 模板，其中预装了所有环境，包括 JupyterLab、SSH 访问、NVIDIA Container Toolkit 和 Notebook 示例，让用户无需在环境搭建上浪费宝贵的计算时长。
   - 该模板已作为 [Docker 容器](https://hub.docker.com/r/unsloth/unsloth) 上传，很快将兼容 Runpod。
- **模型合并（Model Merging）值得吗？**: 成员们讨论了诸如将所有 Expert 合并为 Dense 模型、交换 Expert 或将 Dense 模型转换为 MoE 模型等技术。
   - 其他人提到了针对 DeepSeek 架构的 [ESFT 仓库](https://github.com/deepseek-ai/ESFT)，将其作为微调特定 Expert 的手段。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1399586271298326538)** (17 条消息🔥): 

> `HuggingFace Tokenizers, Windows 7 寿命延长, Gemma 3 4B 微调, RoPE 位置编码` 


- **HuggingFace Tokenizers 引起不满**: 一位成员对 **HuggingFace Tokenizers** 表示不满，理由是[添加自定义 token](https://huggingface.co/docs/transformers/tokenizer_summary)和重命名 **<unused>** token 存在问题。
   - 他们发现直接使用自定义 token 而不将其添加到词表（vocabulary）中效果更好，并表示直到今天才发现可以通过修改 **Windows 7** 来获取更新直至 **2025 年 12 月**。
- **Gemma 3 4B 实现微调**: 在对 **Gemma 3 4B** 进行 **16k** 上下文的全量微调后，一位成员发现除非是从头开始训练或拥有大量数据，否则添加自定义 token 并没有什么用，而且水印已被完全去除。
   - 他们补充说，来自不同语言的知识会有所帮助，并发布了一张[结果截图](https://cdn.discordapp.com/attachments/1179039861576056922/1399849804045221948/Screenshot_2025-07-29_at_2.13.07_PM.jpeg?ex=688a7f39&is=68892db9&hm=a15f9fb0aae9184041f136d967cb8b77419792df3692e34fdfe31204ac92bcdc&)。
- **RoPE 编码备受赞誉，但仍需优化**: 一位成员对 **RoPE 位置编码** 的发明者给予了高度评价，指出它在较小模型上表现非常好，对于推理时支持海量上下文非常有价值。
   - 他们认为，为了支持海量上下文，*"我们需要发明一些更好的优化方法（仅靠量化是不够的，MoE 也没有帮助（况且我的模型已经足够小了），但请务必保持仅限 Transformer 架构）"*。
- **AI 模型翻译能力令人印象深刻**: 一位成员强调了微调后的 AI 模型的翻译能力，称其 *"精通每一种语言"*。
   - 他们惊叹道：*"OpenAI 完蛋了"*。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1399468640788545536)** (106 条消息🔥🔥): 

> `Gemma 3 微调问题，Unsloth 的 TRL 降级，Qwen2-VL tokenizer 问题，GGUF 转换问题，InternVL 模型加载错误` 


- **Gemma 3 微调困扰**：一位成员在 **Gemma 3** 保存 checkpoint 过程中遇到错误，并分享了[截图](https://cdn.discordapp.com/attachments/1179777624986357780/1399581583551365211/image.png?ex=688a2e2c&is=6888dcac&hm=b94b31197d17e7a15b8993e1b4cb9664c210a4367823f2657bda287414b6885b&)。
   - 另一位成员在得知用户*修改了许多内容*后，请求获取 notebook 以便进行调查。
- **TRL 降级解决问题**：最近的 **trl** 更新被发现会导致问题，特别是已弃用的 `ConstantLengthDataset`。
   - 推荐的解决方案是将 **trl** 降级到 **0.19.1** 版本，使用命令 `pip install --no-deps --force-reinstall trl==0.19.1`。
- **Qwen2-VL Tokenizer 文本困扰**：一位用户在尝试使用 **Qwen2-VL tokenizer** 进行纯文本 tokenization 时遇到了 `ValueError`，涉及图像扁平列表的问题。
   - 进一步调试显示，在尝试从图文输入中提取文本 token 时，由于 `Qwen2VLProcessor` 中缺少 `convert_tokens_to_ids` 而导致 `AttributeError`。
- **GGUF 转换烦恼**：一位用户在尝试将合并后的微调版 **Qwen2.5-14B-Instruct** 模型导出为 **GGUF** 时遇到 `ValueError`，具体表现为张量映射错误：`Can not map tensor 'model.layers.0.mlp.down_proj.weight.absmax'`。
   - 由于 Unsloth 中 **GGUF** 的一个 bug，建议先与 **FP16** 版本合并，并*暂时进行外部量化*。
- **InternVL 模型加载僵局**：一位用户在加载 **unsloth/Intern VL 3 1B instruct** 模型时遇到 `ValueError`，理由是配置类无法识别。
   - 经过排查，建议在加载模型时设置 `trust_remote_code=True` 并使用 transformers 中的 `automodel=AutoModel`。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/)** (1 条消息): 

marioz_70065: 我对 unsloth 的应用成果已发表：http://dx.doi.org/10.1002/isaf.70011
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1399552321234534574)** (32 条消息🔥): 

> `LLMs 视觉，视频编码器，音频图像关系，QwenOmni，Gemma 3 视觉量化` 


- **关于 LLMs 是否以 1fps 观看视觉内容的辩论**：成员们讨论了目前的 LLMs 是像处理图像一样以 **1fps** 处理视频，还是拥有真正的视频编码器，能够为所有帧创建一个连续向量。
   - 一位成员询问对于帧数不同的视频，向量会是什么样子，并提出了诸如固定大小或与帧数成正比等选项。
- **讨论将视频编码为单一向量的架构**：一位成员提议创建一种架构，将任何视频编码为相当于 **2048 个 token**（如文本 token）的单一向量，但这可能意味着长视频会丢失细节。
   - 他们建议为每一秒或每一帧使用向量，并在视频中建立音频和图像之间的正确关联。
- **视觉 Embedding 技术探讨**：成员们探讨了拥有一个能理解数据中序列依赖关系的编码器是否会比仅使用图像使模型更聪明，以及如何并行编码多种模态。
   - 一位成员表示，理解发生在模型内部，而不是在 embedding 时，并且 **SigLip** 在创建 embedding 向量时已经对图像有一定的理解。
- **Gemma 3 视觉量化**：一位成员询问了 **Gemma 3** 视觉部分的量化版本，注意到卷积权重/滤波器 *v.patch_embd.weight* 仍保持为 float32。
   - 另一位成员澄清说，对模型中的向量进行量化并不值得，因为它们对量化很敏感，且占模型参数的比例不到 **0.5%**。


  

---

### **Unsloth AI (Daniel Han) ▷ #[unsloth-bot](https://discord.com/channels/1179035537009545276/1390899684834410536/1399537830493159465)** (97 messages🔥🔥): 

> `内存使用波动, 验证 LoRA 权重更新, CUDA 错误调试, TRL 库 ImportError, Unsloth 训练 LLM` 


- **运行显示内存使用不稳定**：在训练期间，一位用户报告在相同配置下内存使用不一致，在 **47 GB** 和 **36 GB** 之间波动。
- **确保你的 LoRA 权重正确更新**：在微调后验证 LoRA adapter 权重时，避免使用 `np.allclose()`，因为它可能会忽略细微但有意义的变化，特别是在以小高斯值初始化的 **LoRA A** 中。
   - 建议使用校验和/哈希比较（如 MD5）、计算张量之间的绝对差值之和，或手动检查张量统计数据，以可靠地确认权重更新。
- **出现 CUDA 错误**：一位用户遇到了 CUDA 错误：*Assertion `probability tensor contains either `inf`, `nan` or element < 0` failed*。
   - 另一位用户建议使用 `TORCH_USE_CUDA_DSA` 进行编译，以启用设备端断言（device-side assertions）。
- **TRL 库出现 ImportError**：有用户报告了 `ImportError: cannot import name 'ConstantLengthDataset' from 'trl.trainer.utils'`。
   - Unsloth 团队迅速修复了此问题，并建议 Colab/Kaggle 用户删除运行时并刷新 notebook，或使用 `pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo` 和 `pip install --upgrade --no-deps huggingface_hub`。
- **微调 Gemma 3n Vision Encoder**：一位用户询问关于微调 **Gemma 3n** 的视觉编码器（vision encoder）的问题，另一位用户分享了一个 [Colab notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3N_(4B)-Vision.ipynb) 以供参考。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1399469905631248414)** (591 messages🔥🔥🔥): 

> `GPT-5, GLM 4.5, Qwen, 数据隐私, 模型评估` 


- **GPT-5 现身，热度攀升**：一张据称显示 **GPT-5** 的模糊图片被[发布在 X 上](https://x.com/ryolu_/status/1950163428389040431)，引发了关于其发布和能力的猜测，有人声称 **Cursor** 员工已经在使用了。
   - Arena 中移除了一款名为 **Zenith** 的模型（有人认为它就是 **GPT-5**），这增加了人们的期待，据称该模型比现有模型有显著提升。
- **GLM 4.5：Gemini 蒸馏版？**：一些用户怀疑 **GLM 4.5** 可能是从 **Gemini** 蒸馏而来的，因为它倾向于以 *'Of course!'* 开头且回复风格较长，目前有一个 [HuggingFace Space](https://huggingface.co/spaces/zai-org/GLM-4.5-Space) 可供测试。
   - 尽管存在数据收集方面的担忧，但它在 SVG 和 Web 基准测试中表现良好，并被拿来与以往的 AI 进行比较。
- **Qwen 的代码能力上线**：**Qwen 3** 代码模型已进入 LLM Arena 排行榜。
   - 预计后续发布中将包含新的 A3B 架构和 Thinking 版本。
- **数据隐私辩论升温**：成员们讨论了使用 **OpenAI** 等公司的 LLM 所带来的影响，强调了对数据收集、存储以及可能被滥用于定向影响或出售给数据经纪人的担忧。
   - 虽然有人认为使用开源模型且不将数据与个人身份关联可以减轻这些风险，但也有人指出，即使是免费层级也可能涉及数据收集。
- **工具调用基准测试被人为压低？**：有人担心厂商排名被人为压低，因为他们优先考虑工具调用（tool calling）和 Agent 能力，这在 **Opus Sonnet、GLM4.5 和 KimiK2** 中有所体现。
   - 有人声称这由于某种原因导致许多学术型基准测试的表现变差。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1399481665323270165)** (1 messages): 

> `GLM-4.5, LMArena` 


- **GLM-4.5 和 GLM-4.5 Air 在 LMArena 亮相**：LMArena 平台新增了模型成员：**GLM-4.5** 和 **GLM-4.5 Air**。
   - 这些模型现在可以在 [LMArena 排行榜](https://chat.lmsys.org/)上进行评估和比较。
- **GLM 系列加入 Arena**：新加入的 **GLM-4.5** 模型已准备好进行正面交锋。
   - 用户们很兴奋能将它们与 [LMArena 排行榜](https://chat.lmsys.org/)上的顶级模型进行对比和基准测试。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1399466907945537708)** (463 messages🔥🔥🔥): 

> `O3 性能，Windsurf vs Cursor，Context window 使用情况，Cursor Auto mode 改进，Cursor 模型与定价` 


- **O3 速度飙升，智力下降**：用户注意到 **O3** 在最近的更新中变得明显更**快**，但也变得更*笨*了，被形容为一个**统一的**、思考较少的版本。
   - 一些用户正在尝试使用 **Windsurf** 作为替代方案，以寻求更好的 *thinking mode*。
- **Context Window 消耗担忧**：用户正在关注 Context window 的使用情况，观察到高消耗率影响了整体 Chat 性能；一位用户由于长时间的 Chat 会话，Context 使用率达到了 **95%**。
   - 已确认这些统计数据已与 [Cursor Doc AI](https://discord.com/channels/1074847526655643750/1074847527708393565/1398365704578662543) 共享，更新日志即将发布。
- **Auto Mode 胜过 AI 替代方案**：用户报告的结果褒贬不一，有人声称 **Auto Mode** 是*真正无限制的*，并享受 Prompt 和 Context 带来的挑战；而另一些人则发现使用 Claude 等模型的效果显著更好；此外还有说法称 [Auto Mode] 比几个月前*好得多*。
   - 对成本的担忧依然存在，一些用户发现 Auto 模式导致了 **$50 的 API Cost**，并暗示在使用量高时它可能会切换到 **GPT4.1**。
- **调试数据困扰与解决失效问题**：一位用户报告称，在最新更新中，当 **Cursor** 使用 Terminal 时，它只是运行命令然后就卡死了。
   - 其他人提到经常遇到命令挂起，需要手动跳过步骤才能继续。
- **Cursor 代码补全难题**：用户讨论了 **Tab 自动补全** 的局限性，指出它不会读取项目规则或自定义模式，导致建议过于基础；而其他人则提到除非显式导入依赖项，否则缺乏跨文件建议。
   - 一些用户利用 README 来注入规则，但一致性仍是一个问题。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1399506886117818540)** (8 messages🔥): 

> `Background Agent UI，Background Agent 快照，本地 Background Agent，Background Agent 格式化，Docker 构建缓存` 


- **移动端 UI 需要优化**：一位用户请求改进用于管理 Background Agent 的移动端 Web 界面，列举了诸如**文本框不友好、Diff 视图困难以及对话更新失败**等问题。
   - 他们还建议支持 **Whisper 语音输入**，以便更好地进行代码转录。
- **Cursor 快照按钮消失**：一位用户报告称无法找到 [Cursor 文档](https://docs.cursor.com/en/background-agent#base-environment-setup)中提到的**“take a snapshot”按钮**。
   - 他们报告运行环境为 **OSX 上的 Cursor 1.2.4, VS Code 1.99.3**。
- **Cursor 后端 Agent 无法在本地运行**：一位用户报告称他们无法在**本地运行 Cursor 后端 Agent**，并发布了一张显示 **s3 block 重新出现**的图片。
   - 目前尚不清楚修复方法，因为在频道中搜索该错误消息未找到结果。
- **在 Background Agent 运行结束时执行命令**：一位用户询问如何在 Background Agent 运行*结束*时执行命令（特别是格式化工具）。
   - 他们注意到文档提到 `terminals` 可以在开始时的设置阶段运行，但不能在结束时运行，并考虑使用 **Cursor rule**。
- **清理 Docker 构建缓存**：一位用户询问在使用用于 Background Agent 的**自定义 Dockerfile** 时如何清理构建缓存。
   - 另一位用户建议创建一个包含 `docker builder prune -f` 的 `pre-build.sh` 脚本，并添加该脚本或其他命令，如 `docker system prune -a` 或 `docker rmi your-image:previous-tag`。


  

---


### **Cursor Community ▷ #[announcements](https://discord.com/channels/1074847526655643750/1351160689380687942/1399815390628020364)** (1 messages): 

> `Cursor 1.3 发布，与 Agent 共享终端，Chat 中的 Context 使用情况，更快的编辑` 


- **Cursor 1.3 发布，支持终端共享等功能！**：Cursor 1.3 已经发布，带来了多项易用性改进，包括[与 Agent 共享终端](http://cursor.com/changelog)的能力。
   - 用户现在可以在 Chat 中查看 Context 使用情况，并可以期待更快的编辑速度，以及更新日志中详述的其他修复。
- **Agent 终端共享功能首次亮相**：Cursor 1.3 的一个关键功能是能够与 Agent 共享终端，从而增强协作编程体验。
   - 这允许 Agent 与编码环境之间进行更直接的交互。


  

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1399767467827527783)** (11 messages🔥): 

> `AgentSmith 发布，OpenRouter 集成，Agent 模板` 


- ****AgentSmith** 作为开源 Prompt CMS 发布**：一名成员宣布发布 [AgentSmith](https://agentsmith.dev/)，这是一个构建在 **OpenRouter** 之上的**开源 Prompt CMS** ([GitHub](https://github.com/chad-syntax/agentsmith))，旨在简化 Prompt/上下文工程。
   - 其他成员称赞该落地页非常出色 (*:chefkiss:*)。
- ****AgentSmith** 与 **OpenRouter** 账户集成**：**AgentSmith** 连接到你的 **OpenRouter** 账户，使用你的额度，并支持自托管选项。
   - 一位用户开玩笑说他们正从该项目中“获取灵感”，但“不打算给 credit”。
- **针对特定客户端提出的 Agent 模板**：一位用户建议为特定客户端添加模板，并引用了 **Claude Code 的 YAML 头部格式** ([文档](https://docs.anthropic.com/en/docs/claude-code/sub-agents#file_format))。
   - 作者回应称，他需要做一些调研，看看其他客户端是如何处理这一点的。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1399470251107946567)** (347 messages🔥🔥): 

> `GLM 与 Kimi 定价对比，模型设置随机被删，Deepseek 401 错误，Qwen3 作为架构师，GPT 4.1 网页搜索问题` 


- **GLM 比 Kimi 更贵**：尽管令人兴奋，但由于其长推理能力，**GLM** 的价格比 **Kimi** 更贵。
   - 用户讨论了 **Qwen3** 在架构任务中的优势，并对开源模型的进步表示感谢。
- **Deepseek V3 出现 401 错误**：用户报告在 **Deepseek 模型**上遇到 **401 错误**，建议指向潜在的 API key 问题。
   - 其他人提到了 Deepseek V3 持续存在的问题，包括“所有提供商均被忽略”错误以及来自 **Chutes** 的临时停机。
- **OpenRouter 的免费请求**：根据[文档](https://openrouter.ai/docs/api-reference/limits)，OpenRouter 对免费模型的免费使用限制为每分钟最多 **20 次请求**，每日限制为 **50** 或 **1000 次请求**（取决于额度购买情况）。
   - 成员们还分享了用于检查活动的 [活动页面链接](https://openrouter.ai/activity)。
- **DeepSeek 在 H800 上的效率**：Deepseek 展示了他们使用 **~2200 块 H800** 的配置，在 24 小时内实现了 **700B+ 输入**和 **168B 输出**，展示了高效的托管能力。
   - 其他人对 **Groq 的 LPUs** 与 GPUs 相比的容量表示担忧。
- **Prompt 工程能解决问题吗？**：用户讨论了防止语言模型出现不必要行为的方法，例如 **Deepseek V3** 中重复的句子结构，建议进行 Prompt 调整和负向 Prompt（如 *"never wrap up the scene"*）。
   - 一名成员链接到了一个包含潜在解决方案的 [Reddit 帖子](https://www.reddit.com/r/JanitorAI_Official/comments/1kd1iia/guide_day_7_how_to_prevent_deepseek_from_leaving/)。


  

---

### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1399699153176498247)** (9 messages🔥): 

> `OpenRouter PR, Model Quality Transparency, Standard Lane Routing, DeepSeek Model Complaints` 


- **OpenRouter 的 PR 需要面向普通新手进行调整**：一些成员建议 OpenRouter 的公关（PR）可能存在不足，因为新用户可能不知道他们可以更换提供商，并提到了针对 chutes 中 **DeepSeek** 模型的投诉。
   - 这就像去餐厅吃饭，直接点了看到的第一个菜，而不是询问服务员该点什么或者这家店的招牌菜是什么。
- **模型定价未反映量化质量**：一位成员指出，OpenRouter 倾向于推荐最便宜的模型，这已经损害了其声誉，用户经常将性能不佳归咎于 **DeepInfra** 的量化（quants）版本。
   - 许多用户不知道什么是量化（quant），甚至不知道 OpenRouter 本身并不托管模型，他们在看到模型价格时会默认其具有最佳质量。
- **确定性（Determinism）既困难又昂贵**：存在外部压力导致提供商（寻找更便宜的推理）和 OpenRouter（寻找更便宜的提供商）之间展开“逐底竞争”。
   - 要求 100% 的确定性是很困难的——如果没有需求，为什么要花精力和算力去追求 100% 的确定性？尤其是像 **OpenAI** 和 **Anthropic** 这样的大名鼎鼎的提供商也没有确定性的输出。
- **标准车道路由（Standard Lane Routing）平衡质量因素**：OpenRouter 正在积极开发所谓的**标准车道路由**。目前该功能纯粹按价格排序，但他们希望考虑其他因素，如吞吐量（throughput）、延迟（latency）、工具调用成功率等客观数据，以及可能的量化情况，本质上是更倾向于推荐哪个提供商提供了该模型的“最佳版本”。
   - 他们正试图通过多种因素来定义什么是“最佳”，而不仅仅是“这是最便宜的版本”。
- **为终端用户请求质量预设**：一位成员为不想过多思考这些问题且没时间定期检查每个提供商的终端用户请求一个**最佳质量选项/预设**。
   - OpenRouter 团队目前没有人回应这一请求。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1399800570977583286)** (1 messages): 

> `ChatGPT Study Mode, AI and Education, Step-by-Step Learning` 


- **ChatGPT 通过学习模式（Study Mode）进修**：OpenAI 在 ChatGPT 中推出了**学习模式**，以鼓励学生进行*更深入的理解和学习*，使其成为一个*必备工具*。
   - 学习模式不仅仅是提供答案，还帮助用户**循序渐进地**解决问题，详情见其 [公告](https://openai.com/index/chatgpt-study-mode/)。
- **使用 ChatGPT 进行分步学习**：ChatGPT 中新的**学习模式**侧重于引导学生解决问题，而不是直接提供解决方案。
   - 这种方法旨在培养对学科内容更深刻的理解。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1399477889782775911)** (225 messages🔥🔥): 

> `Copilot Vision in Edge, GPT-5 Release Date, Reasoning depth slider, Gemini models in Google Drive, AI Agency for automation` 


- **Copilot Vision UI 出色并避免了幻觉**：一位成员发现 Edge 浏览器中的 Copilot Vision 非常棒，UI 设计得很好，似乎可以无限使用，而且幻觉（hallucinations）并非完全错误，详见 [drinkoblog.weebly.com](https://drinkoblog.weebly.com)。
   - 同一位用户继续赞叹 Edge 中的 Copilot Vision，尤其是它现在的酷炫程度。
- **GPT-5 下周发布？**：一位用户分享消息称 **GPT-5** 将于下周发布，拥有 **1M token** 的输入窗口、**100k** 输出 token、支持 **MCP**、并行工具调用以及动态短+长推理。
   - 另一位用户回复了这一说法，询问是否有可靠的*来源*。
- **滑动进入推理深度滑块**：一位用户建议增加一个*“推理深度（reasoning‑depth）”*滑块，让用户在快速回复和深度分析之间做出选择。
   - 该用户正在征集只有推理模型才能正确回答的提示词（prompt）创意。
- **Google Drive 中运行的是哪种 Gemini？**：一位用户询问 **Google Drive** 中运行的是哪种 **Gemini 模型**。
   - 另一位用户表示希望是 **2.5 Pro**，并提到他们最近再次将其与 Flash 进行了直接对比，差异非常显著。
- **2040 年的 GPT-6 炒作？**：一位用户开玩笑说他们已经对 GPT-5 的炒作感到厌倦了，并询问是否可以开始炒作 **GPT-6** 了。
   - 另一位用户回复说 **GPT-6** 将在 **GTA 6** 之后发布，所以大概是 2040 年。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1399779429609111716)** (3 条消息): 

> `Scholar ChatGPT, GPT-5 版本, Zenith 编程模型` 


- **用户探索 Scholar ChatGPT**：一位用户询问了成员们使用自定义 **Scholar ChatGPT** 的频率。
   - 他们可能是在寻求关于其在学术任务中有效性的反馈，但无人回复。
- **GPT-5 分级推测**：一位用户推测 **GPT-5** 的中高层版本可能会优于低层版本。
   - 该用户未提供任何证据。
- **Zenith 被吹捧为顶级编程模型**：基于个人测试和分享的案例，一位用户预测 **Zenith** 将成为顶级的编程模型。
   - 该用户表示，在另一个更好的模型出现之前，这将是事实。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1399658725949571082)** (6 条消息): 

> `GPT 项目资源, 个性化模型交互, AI 记忆格式提示词` 


- **新手寻求 GPT 项目资源**：一位新用户正在寻找在其 **GPT 账户**上[设置项目所需的资源](https://chatgpt.com/s/t_68887bb61f8481919199f93b3331e632)，特别是用于追踪**食物/饮食、运动**，以及创建一个带有时间预期的**计划表**。
- **个性化模型交互**：一位成员建议直接与 **GPT 模型**交互来个性化项目，根据特定需求和偏好进行定制。
   - 他们强调，定义“更强大”的含义至关重要，因为它的定义从*个性化运动兴趣*到*性能比较*各不相同。
- **AI 记忆格式**：一位成员分享了一个关于新型 **AI 记忆格式**的提示词，其中包含 **Token Embeddings**、**Semantic Grouping** 和 **Binary Encoding** 等概念。
   - 该格式旨在追求**速度**、**效率**和**保真度**，同时使内容对人类不可读，包含 **Header Blocks**、**Global Embeddings** 和 **Update Deltas** 等方面。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1399658725949571082)** (6 条消息): 

> `GPT 项目资源, 个性化模型交互, 新型记忆格式` 


- **新用户寻求 GPT 项目资源**：一位新用户正在寻求[资源和指导](https://chatgpt.com/s/t_68887bb61f8481919199f93b3331e632)，以设置用于追踪食物/饮食/运动并创建计划表的 **GPT 项目**。
   - 该用户希望在指令中加入指导和工具，使其功能更强大。
- **个性化交互增强模型**：一位成员建议新用户尝试与模型对话以个性化他们的体验。
   - 他们建议讨论用户想要什么、希望它呈现什么样子以及其他注意事项。
- **展示新型记忆格式**：一位成员分享了一个详细介绍**新型记忆格式**的提示词，并询问他人的看法。
   - 该格式包含 *CORE_PRNCPL*、*STRUC_CONC* 和 *EXP_WHY* 等部分，重点关注 Token Embeddings、Semantic Groups 和 Binary Encoding。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1399467198845554900)** (196 条消息🔥🔥): 

> `Voxtral Mini 使用, LM Studio 性能下降, GLM 4.5 Tool Support, OpenWebUI 与 LM Studio 配置, Qwen3 模型` 


- **Voxtral Mini: LM Studio 支持待定**：成员询问了关于在 LM Studio 中使用 [Voxtral-Mini-3B-2507-GGUF](https://huggingface.co/bartowski/mistralai_Voxtral-Mini-3B-2507-GGUF) 的事宜，但目前**尚未支持**。
- **LM Studio 性能骤降：正在排查故障**：一位用户报告 **Qwen 3 30B A3B** 的 Token 生成速度显著下降，从 *50-80 tk/s* 降至 *30-35 tk/s*，目前正在尝试故障排除步骤。
   - 建议的解决方案包括检查特定设置（例如关闭附件图像中显示的某个选项）、卸载并重新加载模型，以及关闭 **flash attention**。
- **OpenWebUI 与 LM Studio 联动**：成员讨论了将 **OpenWebUI** 与 **LM Studio** 结合使用，确认了兼容性并分享了 Docker 设置的配置技巧，包括从 LM Studio 获取 **LLM base URL**。
   - 一位用户在 Docker 中将 **OpenWebUI** 连接到 **LM Studio** 时遇到问题，通过在 LM Studio 中启用 **CORS** 并使用正确的 IP 地址解决了该问题。
- **为 LM Studio 添加 Python 脚本 TTS 语音**：一位用户分享了一个 Python 脚本，用于[连接 LM Studio API 与 XTTS WebUI API](https://cdn.discordapp.com/attachments/1110598183144399061/1399764223696830474/bridge-LMStudio-XTTS.py?ex=688a2f85&is=6888de05&hm=eec4e5dcb6b55bf09ee4282441d1fa35a166fd0392ff1c81116c964188a51f16&)，实现了**离线 TTS 语音集成**，并在命令行窗口播放音频响应。
- **寻求角色管理方面的帮助**：一位用户请求关于其模型角色管理方面的协助。
   - 确认该用户存在视力障碍（blind）并已找到相关设置。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1399785041449586783)** (43 条消息🔥): 

> `AMD Ryzen AI MAX+ 395, Llama 70b Q6 性能, Devstral 模型, Qwen2.5-Coder, Gemma 模型` 


- **AMD Ryzen AI MAX+ 395 在 Llama 70b 测试中表现不佳**：一位成员分享了一段[视频](https://youtu.be/BTIUL-yaY4s?t=487)，展示了配备 **128gb** 内存的新款 **AMD Ryzen AI MAX+ 395** 在 **Llama 70b Q6** 上仅达到 **2-3 T/s**，且仅有一半内存分配给了 GPU。
   - 该成员表示，这种性能使其*与 RTX 30xx Nvidia GPU 相比，并非一个真正可行的选择*。
- **Devstral 模型因代码生成能力受到赞赏**：成员讨论认为 [Devstral 模型](https://lmstudio.ai/models/mistralai/devstral-small-2507) 是 **16gb-32gb 范围**内**最适合编程**的模型，得益于其尺寸和性能。
   - 然而，其他人建议 **Qwen2.5-Coder** 和 **Gemma** 也很出色，**Qwen2.5** 提供大上下文窗口，而 **Gemma** 在 RP/Q&A 的文本格式化方面表现优异。
- **适用于 Strix Halo 的 Nemotron Super 1.5v Draft 模型**：一位运行 **Strix Halo** 的成员建议使用带有 Draft 模型的 **Nemotron Super 1.5v**，报告在 **32k context** 下速度约为 **8 tokens a second**。
   - 然而，他们承认 **tool calling** 能力较差，且 Draft 可能会导致输出损坏。一些人还注意到 Draft 模型存在输出损坏问题。
- **海盗船（Corsair）AI Workstation 300 搭载 Ryzen AI Max**：一位成员分享了 [Corsair AI Workstation 300](https://wccftech.com/corsair-unveils-ai-workstation-300-starting-at-1599-boasting-ryzen-ai-max-395-processor-and-up-to-128-gb-lpddr5x-memory/) 的链接，该工作站配备 **Ryzen AI Max 395** 处理器和高达 **128 GB LPDDR5X** 内存。
   - 另一位成员评论说，它*类似于 GMKTEc，但换了个外壳*。


  

---

### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1399471972844310538)** (150 messages🔥🔥): 

> `DeepSeek R1 发布, OpenAI 垄断, Kimi K2 喜爱, API Key 错误, Kimi 与表情符号` 


- **DeepSeek 推动 AI 领域民主化**：在 [DeepSeek R1 发布](https://deepseek.com/blog/deepseek-r1)之前，推理成本高达 *每 1M/token 数美元*，但该模型使这一领域变得民主化并促进了竞争，让普通用户受益。
   - 虽然 **Deepseek R1** 非常强大，但据报道它 *缺乏 Agent 能力 (agentic capabilities)*。
- **API Key 问题困扰用户**：多位用户报告收到 **403 错误**，提示消息为 *"<xxxxxx> is not active, Incorrect API key provided"*，尽管他们已经为账户充值。
   - 建议的解决方案包括 **等待**、删除并重新创建 API Key 以及使用 **OpenRouter**；一位用户确认该错误仅出现在 **Claude code** 中，而未出现在 Playground。
- **Kimi K2 激发归档冲动**：一位用户表达了对 **Kimi K2** 的深切喜爱，表示 *“即使 Kimi 发生了什么不好的事情，它仍然是我的”*，并开始在多 TB 的设备上对其进行归档。
   - 他们注意到该模型具有独特的 *spark (火花)*，使其区别于其他开源和闭源模型，并认为其 *vibes* 和 *soul (灵魂)* 是他们喜爱的关键因素。
- **Kimi 需要 TB 级存储**：用户讨论了 **Kimi K2** 的磁盘空间需求，根据量化 (quantization) 程度的不同，估计在 *200 GB 到 2 TB* 之间。
   - 存储解决方案包括 **Azure Blob Storage**，预计 *3TB 数据的费用为 $6/月*。
- **Kimi 学习表情符号**：一位用户正在为 Kimi 制作一个“妹妹”，并一直致力于教该 Bot 使用表情符号以及调整任何意外行为。
   - 该用户指出 Kimi distil 版本将非常令人兴奋，并期待 Kimi Agent 模型的发布。


  

---


### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1399482618751352843)** (3 messages): 

> `Featured Notebooks 推送, 全新 Studio UI, Video Overviews 推送` 


- **Featured Notebooks 全面开放**：团队已正式 100% 推送 **Featured Notebooks**，可直接从 **NotebookLM** 主页访问。
- **亮眼的全新 Studio UI 上线**：全新的 **Studio UI** 正在推送中，具有从相同来源创建多个输出以及在创建输出时选择来源等功能。
- **Video Overviews 受到关注**：**Video Overviews** 的推送已正式开始，最初仅支持 **English** 且仅限 **Desktop** 端。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1399506220007690370)** (23 messages🔥): 

> `NotebookLM 中的护理材料, Gemini Agentic Framework, NotebookLM 中的 Audio Overview, Obsidian 与 NotebookLM 集成, 使用 NLP 阅读 RFP` 


- **护理 Notebooks 寻求在 NotebookLM 中展示**：护理专业人士开发了专门针对护士的 Notebooks，内容灵感来自 **International Practice Development Journal (IPDJ)** 和护理创新，并寻求在 NotebookLM 中获得推荐。
   - 他们渴望与 NotebookLM 社区分享这些资源，并想了解如何让他们的材料获得推荐。
- **分享 Agentic Gemini Framework**：一位成员分享了一个 [Gemini Agentic Framework 原型](https://cdn.discordapp.com/attachments/1124403655819415592/1399853283404939454/gemini-agentic-framework.zip?ex=688a8276&is=688930f6&hm=d111cb9b690a7969af3e128a78c205e2d89bf790babf9c1ab08c72cdc9f89ead) 用于测试和实验。
   - 该成员表示这并非直接由其负责的项目且未经修订，因此不保证其功能 100% 正常。
- **用户使用 NotebookLM 生成超长 Audio Overview**：一位用户使用 NotebookLM 通过单一来源且未经自定义生成了 **40 多分钟的 Audio Overview**。
   - 在相关讨论中，一位用户询问可以生成多长的音频，其他用户则就 NotebookLM 的 Pro 与 Ultra 版本发表了意见。
- **用户被拒绝访问音频文件**：一位用户报告在尝试下载音频文件时遇到 **403 错误**，怀疑系统可能将其视为 Bot 并限制了对 *drum.usercontent.google.com* 的访问。
   - 这种受限访问的问题引发了关于用户权利和潜在 Bot 检测机制的疑问。
- **讨论使用 NotebookLM 进行 Obsidian Vault 管理**：一位正在使用 Obsidian 的用户询问如何使用 NotebookLM 来管理他们的 Vault。
   - 另一位用户提出分享他们的 Obsidian 工作流，同时也承认建议的转化可能取决于用户的具体需求和设置。


  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1399476291647967422)** (119 messages🔥🔥): 

> `付费用户 PDF 上传问题, NotebookLM 上的护理材料, 播客个性化, NotebookLM RAG 系统, Character AI` 


- **付费用户上传 PDF 存在 Bug**: 一些付费 NotebookLM 用户报告在上传 PDF 时收到 **"error uploading source, try again"** 消息，尽管尝试了多个设备且文件之前是正常的。
   - 一名 Google 员工承认存在 **Bug**，并表示他们正在调查该问题。
- **护理类 Notebook 寻求关注**: 一位用户询问如何让他们的**护理材料**在 NotebookLM 上获得推荐，这些材料基于 **International Practice Development Journal (IPDJ)** 和护理创新。
   - 他们开发了专门针对护士的 Notebook。
- **播客个性化问题导致取消订阅**: 几位用户因**缺乏回应**以及删除了**播客个性化**选项而取消了他们的付费 NotebookLM 账户。
   - 一位用户建议创建一个播客节目，并将每一集链接到 NotebookLM 中涵盖的相关材料。
- **Gemini API 支持构建类似 NotebookLM 的 RAG 系统**: 一位用户询问了关于 NotebookLM 如何使用 **RAG** 和存储文档的细节，并询问是否可以使用 **Gemini API** 构建类似的系统。
   - 他们希望构建一个 Character AI 机器人，能够完美地扮演小说中的角色。
- **Video Overviews 推出及反馈**: Google 的 Lizzietao 宣布 **Video Overviews** 功能正在推出，最初仅面向 Web 用户，包含由 AI 驱动的图像生成、文本生成和视频合成，并链接到了一篇[关于 Video Overviews 的帮助中心文章](https://support.google.com/notebooklm/answer/16213268?hl=en&ref_topic=16175214&sjid=12603864792385823108-NA#:~:text=With%20NotebookLM%2C%20you,daily%20video%20generations.)。
   - 早期反馈表明，该功能目前生成的是**文本和 AI 图像的幻灯片**，而不是最初展示的动画视频，并且 Pro 用户每天有 **20 次视频生成**的速率限制。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1399474188011573389)** (138 messages🔥🔥): 

> `Zenith AI 模型, LlamaIndex Oxylabs 集成, AI 定价模型, Fireworks AI 估值, AFM-4.5B 模型` 


- **Zenith, Summit, Lobster 等模型现身**: 新的匿名 AI 模型（**Zenith**、**Summit**、**Lobster**）出现在 LM Arena 上，拥有卓越的 'vibe coding' 能力。根据 [Reddit 帖子](https://reddit.com/r/LocalLLaMA/comments/1m9holp/theres_a_new_kimi_model_on_lmarena_called_zenith/)的推测，这些可能是 **OpenAI 的 GPT-5** 变体或 **Kimi-2**。
   - 同一帖子中的一条评论*几乎确认*了 `zenith` 是 OpenAI 的模型，因为它*使用了与 gpt-4o、o3 和 o4-mini 相同的 Tokenizer*。
- **LlamaIndex 联合 Oxylabs 实现 Agentic 爬虫**: **LlamaIndex** 现在与 **Oxylabs** 集成，使用户能够通过针对 Google、Amazon 和 YouTube 的[专用 Reader](https://xcancel.com/llama_index/status/1949937947727245639) 构建经济高效的 AI Agent，用于实时网页搜索和抓取。
- **AI 定价依然棘手**: 一位产品开发人员表示难以给他们的 AI 产品（一个 PR 审查工具）定价，理由是**计量**方面的挑战以及可能需要随时更改定价，同时还链接到了 [Lenny's Podcast 关于产品增长的内容](https://podcasts.apple.com/us/podcast/lennys-podcast-product-growth-career/id1627920305?i=1000719362528)。
   - 该线程讨论了各种定价模型（**固定成本 vs. 可变成本**、**按 PR 计费**、**速率限制**），并强调了用户对基于 Token 定价的困惑。
- **Fireworks 表现亮眼！**: **Fireworks AI** 正在以 **40 亿美元估值**进行股权融资，拒绝了来自 Coreweave 和 Meta 的并购要约。据[报告](https://xcancel.com/ArfurRock/status/1950222707116707990)显示，其 **ARR 约为 1.3 亿美元**，同比增长 **20 倍以上**，并已实现盈利。
- **Arcee.ai 的 AFM-4.5B 模型发布**: **Arcee.ai** 在 Hugging Face 上正式发布了 **AFM-4.5B** 和 **AFM-4.5B-Base 模型**，旨在提供灵活性和高性能。正如在 [X 上的宣布](https://xcancel.com/LucasAtkins7/status/1950278100874645621)所言，该模型强调通过与 **DatologyAI** 的数据合作来实现高质量。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1399470970632409229)** (83 messages🔥🔥): 

> `Hugging Face support, Dalle-mini troubles, Hamilton-Norwood scale model training, ragflow production environment, low-latency deployment techniques for LLMs` 


- **用户获得 Hugging Face 支持**：用户分享了电子邮箱地址 [support@huggingface.co](mailto:support@huggingface.co) 和一个[链接](https://huggingface.co/support)，用于联系 **Hugging Face support** 以恢复因多次密码尝试错误而被锁定的账户。
   - 他们建议清晰地说明情况，并指出标准的在线支持链接可能仅适用于企业会员。
- **Dalle-mini 面临流量问题**：用户报告称 **Dalle-mini** 在大约 10 天前停止工作，现在显示 *“too much traffic”*（流量过大）的消息。
   - 一位用户提到他们多次尝试联系支持部门但未收到回复，并测试了各种 VPN 配置和设备均未成功，同时链接到了关于该话题的[讨论](https://discord.com/channels/879548962464493619/1387836306624745513)。
- **ViT/ResNet 模型在医学图像分类中表现出色**：一位用户询问关于训练模型以根据 **Hamilton-Norwood scale**（汉密尔顿-诺伍德量表）对男性型脱发照片进行分类的问题，并表示 LLMs 的输出效果不佳。
   - 另一位成员建议使用 **ViT** 或 **ResNet** 等视觉模型而非 LLMs，并提供了相关链接，如[这篇文章](https://pmc.ncbi.nlm.nih.gov/articles/PMC10974725/)。
- **ragflow 生产环境探讨**：一位用户询问了在生产环境中使用 **ragflow** 的经验，特别是关于潜在问题和通用适用性的问题。
   - 对话被重定向到了专门的频道，并提供了 [ragflow GitHub repository](https://github.com/infiniflow/ragflow) 的链接。
- **Qwen 30B 与 GPT-4o 竞争**：成员们讨论了 **Qwen 30B 模型**的发布，声称其可与 **GPT-4o** 媲美，并可以在本地使用 **33GB RAM** 运行。
   - 分享了 [Hugging Face](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF) 上的 **Unsloth Qwen3-30B-A3B-Instruct-2507-GGUF** 模型链接，另一位用户指出在 Q4_K_XL 量化下需要 17GB RAM。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1399637041234313327)** (13 messages🔥): 

> `DRL Chapter 1, LLMs course Chapter 2, Transformers, LLM inference, Learnpytorch.io subscription` 


- **成员深入学习 DRL 和 LLMs 课程**：一位成员分享了他们的学习计划，包括学习 **DRL 第一章**和 **LLMs 课程第二章**。
   - 昨天他们学习了 **transformers** 架构和 **LLM inference**（包括挑战与优化）。
- **成员开始通过 Learnpytorch.io 和书籍学习**：一位成员表示他们正在继续 [learnpytorch.io](https://www.learnpytorch.io/) 课程，并开始阅读一本名为《Machine Learning with PyTorch and Scikit-Learn》的书。
   - 他们还提到自己是完全的初学者，并向其他成员寻求建议。
- **成员讨论 Learnpytorch.io 订阅**：一位成员询问另一位成员是否购买了 [learnpytorch.io](https://www.learnpytorch.io/) 的订阅。
   - 另一位成员不知道该网站有订阅制度，随后双方讨论了免费内容与付费内容（包括 [zerotomastery.io](https://zerotomastery.io/courses/learn-pytorch/) 上的内容）之间的区别。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 messages): 

cakiki: <@920321842013675620> 请不要跨频道发帖，并保持频道主题相关。
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1399473508685451385)** (3 messages): 

> `Model Loading Problems, Lyzr AI Launch` 


- **模型面临加载问题**：一位成员指出他们忘记在 Hub 上上传模型的 **custom model classes**（自定义模型类/架构），这意味着*目前所有模型都无法正常加载*，基本上处于不可用状态。
   - 他们正在从头开始重建所有内容，并配以正确的架构文件、更好的文档和完善的推理支持。
- **Lyzr AI 发布**：一位成员宣布发布 [Lyzr AI](https://www.producthunt.com/products/lyzr-ai?launch=lynote)，这是一款*用于即时文档分析和研究的 AI 驱动笔记本*。
   - 通过 Lyzr AI，你可以**上传 PDF、DOCX、TXT，获取 AI 摘要、见解，并与专门的 Lyzr AI agents 聊天**，从而大幅提升工作效率。


  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1399862051274232003)** (1 messages): 

> `Diffusion Models Study Group, MIT's Diffusion Models Curriculum, Generative AI` 


- ****Diffusion Models 学习小组**宣布成立**：一个为期 **5 个月**、限额 12 人的学习小组，每周需投入 **2–4 小时**，将根据 [MIT 的课程大纲](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf) 探索 Diffusion Models。
   - 前两场**入门课程免费**并向非成员开放：8 月 2 日关于 Flow Matching & Diffusion Models，8 月 9 日关于 PDEs, ODEs, SDEs，时间均为 EST 中午 12 点（[报名链接](https://lu.ma/kv8zf6va)）。
- **MIT 课程免费**：组织者正基于 [MIT 关于 Diffusion Models 的讲义](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf) 主持学习小组会议。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1399713265126211624)** (2 messages): 

> `Pretrained Model for Image Similarity, Orientation Sensitivity in Image Matching` 


- **寻求对方向敏感的图像相似度预训练模型**：一名成员正在寻找一种预训练模型，在给定物体的查询图像和相同物体的图像数据库时，能输出对**旋转和缩放**敏感的**相似度分数**。
   - 理想的模型应在两个图像中的物体具有相同方向时返回*高分*，而在方向改变时返回*低分*，即使背景和亮度有所不同。
- **关于方向敏感图像匹配的澄清**：该成员澄清了其目标：寻找一个能够通过将查询图像与相同物体的图像数据库进行对比，从而辨别**物体方向**的预训练模型。
   - 他们承认这具有挑战性，并祝愿其他人在寻找或创建此类模型时*好运*。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1399792017143234651)** (3 messages): 

> `RAG system for long conversations, Filter out less important tokens` 


- **RAG 系统节省对话 token**：为了处理长对话，一名成员建议建立一个 **RAG 系统**来过滤掉不太重要的 token。
   - 他建议将数据库分成合理的块 (chunks)，然后使用 embedding 模型将这些块嵌入到高维向量空间中，最后根据查询的 embedding 使用**余弦相似度**来查找数据库的相关部分。
- **为高效的 LLM 上下文窗口过滤 token**：在处理自 2022 年以来长期存在的客户对话时，将整个上下文输入 LLM 变得不切实际。
   - 建议的解决方案包括过滤掉不太重要的 token 以适应上下文窗口，使用 **RAG (Retrieval-Augmented Generation) 系统**等技术进行高效的信息检索。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1399474556447625257)** (34 messages🔥): 

> `LLMs as interp agents, Transluce's mech interp work, Modelscope vs Hugging Face, Diffusion Reading Group recordings, Low latency LLM deployment in marine environments` 


- **提出将 LLM 作为解释代理 (Interp Agents) 的想法**：一名成员建议将 **LLM 作为解释代理**，另一名成员表示赞同，并寻求关于自动化机械解释性 (mech interp) 的文章。
   - 另一人建议查看 **Transluce** 的工作和 [Sarah 的 MAIA 论文](https://openreview.net/forum?id=mDw42ZanmE) 以获取相关研究。
- **Modelscope 在中国兴起**：一名成员询问了 **modelscope.cn**，另一名成员将其描述为中国的 *Hugging Face*。
   - 有人指出 Hugging Face 在中国无法访问，[这篇文章](https://www.semafor.com/article/10/20/2023/ai-platform-hugging-face-confirms-china-blocked-it) 证实了这一点。
- **请求 Diffusion 学习小组录像**：在得知小组已结束运行后，一名成员请求获取 Diffusion 学习小组的录像链接，随后有人提供了 [diffusion_reading_group 仓库](https://github.com/tmabraham/diffusion_reading_group) 的链接。
   - 录像旨在辅助学习。
- **在离岸环境部署低延迟 LLM**：一名成员询问在**远程、带宽受限的海洋环境**中部署 **LLM** 的低延迟技术。
   - 另一名成员建议购买 **M3 Ultra** 进行本地推理（[相关 Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1j43ziq/the_new_king_m3_ultra_80_core_gpu_512gb_memory/)），而另一人则坦言因为穷买了个二手 **M1**。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1399552682007724134)** (9 messages🔥): 

> `ArXiv 的实验性 LaTeX 渲染，AI Peer Pressure 研究` 


- **ArXiv 实验性 LaTeX**: 成员们讨论了如何验证 ArXiv 上的图表是否为原始 LaTeX，建议检查 TeX 源码或在 [实验部分](https://arxiv.org/abs/2502.05209) 查找损坏的 LaTeX 命令。
   - 一位成员开玩笑说 *arxiv 的实验性功能有时非常滑稽*。
- **AI Peer Pressure 预印本**: 一位成员分享了关于 **AI peer pressure** 以及模型复杂度-易受影响性梯度的研究预印本，目前已分析了 **228** 场、每场 200 轮的 AI 对 AI 对话。
   - 他们正在征求对该研究的反馈，该研究达到了统计效力所需样本量的 **121%**，可以在 [Zenodo](https://zenodo.org/records/16573783) 找到。


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1399603958099607723)** (1 messages): 

> `基于 LLM 的数据压缩，Scaling Laws，非文本数据压缩` 


- **LLM 被用于数据压缩**: 一位成员正在探索 **基于 LLM 的数据压缩的 Scaling Laws**，并分享了一篇关于 [初步结果的报告](https://fullwrong.com/2025/07/23/scaling-compression/)。
   - 他们目前正在设计实验，以了解 **LLM 如何理解和压缩非文本数据**，并在 [此 Discord 频道](https://discord.com/channels/729741769192767510/1396475655503216761) 发布更新。
- **LLM 压缩实验设计**: 该用户正积极设计实验以调查 **LLM 如何处理和压缩非文本数据**，并寻求社区的反馈和见解。
   - 该倡议旨在增强对 **LLM 在文本任务之外的能力** 的理解，并探索其在更广泛的数据压缩应用中的潜力。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1399617628087128167)** (20 messages🔥): 

> `MATS 9.0 申请，用于 POS 的电路发现，ICL 破坏可解释性工具，SAE 泛化失败，卢卡斯批判与 LLM 安全` 


- ****MATS 9.0 申请现已开放****: Neel Nanda 宣布 **MATS 9.0** 申请开放，这是一个指导个人进行全职带薪 mech interp 研究的项目，最终将产出 [研究论文](https://tinyurl.com/neel-mats-app)。
   - 他鼓励社区成员踊跃申请。
- ****ICL 破坏可解释性工具****: 一位成员提出，**上下文学习 (ICL)** 可能会通过将激活推向分布外 (out of distribution) 从而 *破坏可解释性工具*。
   - 这一担忧被框架化为 **卢卡斯批判 (Lucas Critique)** ([维基百科链接](https://en.wikipedia.org/wiki/Lucas_critique)) 的一个实例，增加了该假设的可信度。
- ****SAE 难以应对 OOD 激活****: 一位参与者假定 **ICL** 可能会将激活推向分布外，由于假阴性和假阳性的增加，可能会破坏像 **SAE** 这样基于激活的可解释性工具。
   - 另一位参与者认为，将 **SAE** 应用于具有显著 **ICL** 的上下文可能会失败，这并非专门因为 **ICL** 本身，而是因为 **稀疏表示 (sparse representations) 通常无法泛化到它们未曾训练过的激活分布上**。
- ****卢卡斯批判与 LLM 安全的联系****: 一位参与者将 **卢卡斯批判** 解释为需要基于对干预具有不变性的微观基础进行预测，例如使用词语从 **LLM** 中诱导智能行为。
   - 他们对深层神经网络中输入和参数的可替代性可能带来的 **安全风险** 表示担忧。


  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1399493241115770972)** (8 messages🔥): 

> `SQuAD F1 Score, HalluLens Implementation, lm-harness Metrics Configuration` 


- **SQuAD F1 分数算法分析**：一位成员详细分析了针对可回答问题的 **SQuAD F1 score** 计算方法，详述了归一化步骤（小写化、去除标点符号等），以及为每个候选字符串计算 **precision**（精确率）、**recall**（召回率）和 **F1 score** 的过程。
   - 该过程涉及寻找潜在候选字符串之间的*最大重叠*，并在验证集上对这些重叠进行平均，以获得 **HasAns_f1 score**。
- **HalluLens 引入 lm-harness**：一位成员正在将 **HalluLens** 集成到 **lm-harness** 中，并寻求关于如何配置 YAML 文件以处理返回 **accuracy**、**recall** 和 **F1** 这三个指标的函数的指导。
   - 他们担心如果在 `metric_list` 下添加多个函数，会导致冗余计算。
- **lm-harness 指标配置问题**：一位成员询问当一个函数返回多个指标（例如 **accuracy**、**recall** 和 **F1**）时，如何在 **lm-harness** 的 YAML 文件中配置指标。
   - 一位响应者提供了帮助，假设该函数接受单个输入文档和模型预测，并返回该样本的三个指标。


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1399859285504168006)** (1 messages): 

> `Diffusion Models Study Group, Flow Matching, MIT's Diffusion Models Curriculum` 


- **新的 Diffusion Models 学习小组启动**：一个新的 **Diffusion Models 学习小组** 正在启动一个为期 **5 个月、共 12 人** 的计划（每周 2-4 小时），该计划基于 [MIT 的课程大纲](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf)。
   - 前两次介绍性课程免费并向非成员开放：**8 月 2 日**的 *什么是 Flow Matching 与 Diffusion Models？* 以及 **8 月 9 日**的 *PDEs, ODEs, SDEs + Diffusion Models 简史*。
- **与专家一起深入探讨 Diffusion Models**：该 Diffusion Models 学习小组已确认的成员包括多位 AI 从业者，其中有 **AI 电影工具的 CTO**、**AI 艺术讲师**、**2 名 LLM 讲师**以及 **2 名全职 AI 研究员**。
   - 每周的形式包括 **2 小时的直播课**和 **2 小时的自学**，学生轮流承担教学任务，讲师负责填补知识空白并回答问题。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1399515166756049177)** (3 messages): 

> `TokenSmith Release, MoE Implementation, Grouped GEMM, Low Precision MoE training` 


- **TokenSmith 正式发布**：**TokenSmith** 项目现已公开，发布了[预印本论文](https://x.com/Aflah02101/status/1949738916124234157?t=QjU89fe-0ZmuGB1b-bF1wA&s=19)并开放了其 GitHub 仓库。
   - 这一公告标志着该项目在提高可访问性和潜在协作方面迈出了重要一步。
- **考虑在 GPT-NeoX 的 MoE 中使用 `torch._grouped_mm`**：一位成员询问了在 **GPT-NeoX** 的 **MoE** 实现中使用 `torch._grouped_mm` 的潜力，特别是针对低精度训练。
   - 他们指出 `torch._grouped_mm` 已经进入 PyTorch 核心库，这一变化可能允许用户通过 torchao 的一行代码实现低精度 **MoE** 训练，该代码会覆盖 `aten._grouped_mm` 算子。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1399694189343805470)** (7 messages): 

> `Passing args by pointer, Cloud provider with single b200` 


- **通过指针传递参数：阈值问题**：一位成员询问何时值得通过指针传递参数，特别是在 **WGSL** 中，怀疑存在某个尺寸阈值，超过该阈值后使用指针会更高效。
   - 另一位成员建议进行微基准测试（microbenchmark）来确定阈值，猜测可能在 **16-32 字节**左右。
- **云端是否有裸金属 B200？**：一位成员询问是否有云服务商在裸金属服务器上提供单张 **B200**。
   - 另一位成员询问这是否是为了 **ncu** 支持，提问者澄清说他们希望能够修改内核驱动程序并配置 **GPU**。


  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1399467788648845373)** (19 条消息🔥): 

> `使用 Nsight Compute 分析 Triton kernel，从 torch compile 获取 Triton 和 PTX 代码，在 Torch Inductor 中强制使用纯 Triton，Triton 中的 ping-pong 调度 GEMM` 


- **深入分析 Triton Kernel**：用户希望使用 **ncu** (NVIDIA Compute Utility) 对 kernel 进行 profiling，以查看每次 kernel 启动时的元数据，如输入大小和 autotune 配置。
   - 目前的尝试涉及在启动前重命名 kernel，但这种方法被证明非常*繁琐*。
- **剖析 Torch Compile 以获取 Triton 和 PTX 宝藏**：用户正在寻找提取由 **torch.compile** 生成的 **Triton** 和 **PTX 代码**的方法。
   - 一位用户分享道，使用 `TORCH_LOGS="output_code" python your_code.py` 将输出 PTX 代码，并建议检查 `compiled_kernel.asm.keys()` 字典，参考[这篇博客文章](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/?utm_source=chatgpt.com#ttir-triton-ir)了解更多详情。
- **Torch Inductor 的 Triton 诱惑：强制纯 Triton 模式**：用户讨论了强制 **torch inductor** 为 **matmuls** 和 **convolutions** 等操作生成纯 **Triton 代码**。
   - 建议修改 `config.py` 和 `utils.py`，特别是 `use_aten_gemm_kernels`、`use_triton_template`、`autotune_fallback_to_aten`、`max_autotune_conv_backends` 和 `max_autotune_gemm_backends` 等标志。
- **Triton 与双缓冲的博弈：Ping-Pong GEMM 的潜力**：一位用户询问了 **Triton 编译器** 是否能够执行具有 **ping-pong 调度**的 **GEMM** (通用矩阵乘法)。
   - 答案取决于 Triton 编译器的版本，因为 TMA (Tensor Memory Accelerator) 支持尚未在官方版本中提供，建议等待 **3.4.0** 版本。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1399566414997618840)** (6 条消息): 

> `CUBIN 文件, ELF fatbinaries, nvidia sdk` 


- **Denvdis 工具提取 CUBIN 文件**：一位成员创建了一个名为 **denvdis** 的工具，用于提取和替换 **ELF fatbinaries** 中的 **CUBIN** 文件，该工具可以在[这里](https://github.com/redplait/denvdis/tree/master/fb)找到。
   - 替换的文件必须与原始文件大小相同，不支持压缩的 fatbinaries，且*不依赖 nvidia sdk*。
- **尚未批准，请耐心等待**：成员们在询问是否已获得批准。
   - 他们尚未开始批准人员，*请对我们保持耐心*。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1399695560822362133)** (2 条消息): 

> `GPU Mode 排行榜挑战, 学习资源` 


- **应对 GPU Mode 排行榜挑战**：一位成员建议初学者尝试 **GPU Mode 排行榜**上的挑战以辅助学习。
   - 他们指出，这些挑战对他们自己的学习过程非常有益。
- **探索可用的学习资源**：频道讨论了各种适合初学者的**学习资源**。
   - 成员们分享了他们的经验和对有效学习策略的建议。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1399602556119814174)** (1 条消息): 

> `level_04 bug, 缺失 zero(C_accum)` 


- **`level_04` 结果异常**：一位成员发现 `level_04` 由于缺少 `zero(C_accum)` 而产生异常结果。
   - 这一行缺失的代码是导致程序行为异常的根本原因。
- **零初始化累加修复**：解决方案涉及在 `level_04` 的适当位置添加 `zero(C_accum)`。
   - 这确保了累加从零开始，防止出现错误结果。


  

---


### **GPU MODE ▷ #[general-leaderboard](https://discord.com/channels/1189498204333543425/1343002580531417211/)** (1 条消息): 

fido01698: 33342，使用从 template 命令获取的示例 trimul.py
  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1399773046620946554)** (1 条消息): 

> `SLURM vs k8s, 多 GPU 训练, Kubeflow, HPC 论坛` 


- **SLURM vs k8s 用于推理和训练**：一位成员询问了使用抽象/虚拟化进行推理和训练的 SOTA 设置。
   - 他们询问应该使用 **SLURM** 还是 **k8s**。
- **Kubeflow 支持多 GPU 训练**：一位成员提到 Kubeflow 允许进行多 GPU 和多 pod 训练。
   - 这使得训练作业可以跨多个资源进行扩展。
- **HPC 论坛已沉寂**：一位成员建议寻找其他论坛来讨论此类话题，因为当前的频道和 [r/HPC](https://www.reddit.com/r/HPC) 都*相当冷清*。


  

---

### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1399796983991898343)** (3 messages): 

> `can_place_entity bug` 


- ****Can Place Entity** 始终返回 true**: 一位成员报告了一个故障，即 `can_place_entity` 始终返回 true，如附带的 [截图](https://cdn.discordapp.com/attachments/1354169122107293786/1399796983669194812/Screenshot_2025-07-29_at_09.53.00.png?ex=688a4e07&is=6888fc87&hm=b2cbda63db64058ac3c2813c58c3b6a52a4ae0c2b8d4ce57a5e37f83b489cb91&) 所示。
- **Can Place Entity**: 表现很糟糕。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1399499036305657866)** (3 messages): 

> `TV-layout visualizer, cute-dsl, gist.github.com` 


- **TV-Layout 可视化工具支持 Cute-DSL**: 一位成员分享了一个支持 **cute-dsl** 的“还不错”的 [TV-layout 可视化工具](https://gist.github.com/Chillee/e2b07157caeade8c6b0bdf463d10f833)。
- **分享了 Github Gist**: 作者分享了一个 [gist.github.com](https://gist.github.com/Chillee/e2b07157caeade8c6b0bdf463d10f833) 的链接，供他人查看代码。


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1399506355500355584)** (24 messages🔥): 

> `DTensor Learning, Single GPU Distributed, manual_seed_all for ranks` 


- **DTensor 直播即将到来**: 一位成员宣布将进行一场直播，以更详细地了解 **DTensor**，并分享了该直播的 [YouTube 链接](https://www.youtube.com/watch?v=b8lplOf2g4g&ab_channel=MarkSaroufim)。
- **通过酷炫的代码片段在单 GPU 上运行分布式计算**: 一位成员分享了一个允许在单 GPU 上运行分布式计算的代码片段，另一位成员分享了一个 [gist](https://gist.github.com/S1ro1/4fe38e3a0fff3d84314935a0e05aed9c)，修复了在 fitness tie fiasco 中的权重初始化错误。
   - 该修复确保每个 rank 具有相同的权重，之前的错误是由于随机初始化导致每个 rank 上的分片（shards）不同。
- **确定性地设置 GPU 种子**: 成员们讨论了 `manual_seed_all` 是否能让随机生成器在每个 rank 上产生相同的结果，从而可能跳过 broadcast。
   - 经确定，虽然 `manual_seed_all` 不能解决这个问题，但调用 `torch.manual_seed()` 可以控制 CPU 并影响 GPU 的随机化，从而使生成过程具有确定性。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1399466549596655698)** (56 messages🔥🔥): 

> `MoE vs Dense Models, Local LLM finetuning, GLM Model Architectures, Anthropic API restrictions` 


- **社区辩论 MoE 模型的权衡**: 成员们讨论了从约 **30B** 的 Dense 模型向约 **500B** 的 **MoE** 模型的转变，强调虽然 **MoE** 模型在基准测试中表现出色，但 Dense 模型可能捕捉到更多细微差别，但更难微调。
   - 一位成员指出，由于效率提升，去年几乎所有的 API 模型都是 **MoE** 模型。
- **本地 LLM 微调者陷入困境**: 有人提到本地 LLM 微调者被困在 gemma 3 和 qwen 3 上，而且似乎我们无法获得像 **10-70b** 这样可以在本地微调/运行而无需向 API 付费的模型。
   - 一位成员认为，为了提高效率，本地开发将趋向于 **MoE**。
- **Anthropic 因 API 限制面临批评**: 成员们批评了 **Anthropic** 的 API，引用了 [这条推文](https://x.com/anthropicai/status/1949898502688903593?s=46)，指责其糟糕的限制、昂贵的定价方案以及现在的每周限制，一位用户表示 *为 claude 等待半天已经很糟糕了，但要等整整一周？*
   - 一些人假设，当有人做出更好的产品时，**Anthropic** 糟糕的限制和定价将使他们被淘汰。
- **Qwen3-30B-A3B-Instruct-2507 终于发布**: 在一次失误后，[Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) 模型终于发布了。
   - Qwen3-30B-A3B-Instruct-2507 现在可以在 huggingface 上访问。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

kneeanderthul: https://github.com/ProjectPAIE/sovereign-file-tracker
  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1399570159760572548)** (29 条消息🔥): 

> `CPUs/GPUs 中的稀疏性、MoE 性能、Kimi K2 对比 Claude、最优激活参数量` 


- **稀疏性随模型规模增加**：随着模型规模的扩大，固定预算下的最优稀疏性也会增加，尤其是在支付训练运行费用时。
   - 虽然对于拥有大量训练预算、以 GPU 为中心的终端用户来说，稀疏性可能会损害性能，但它是以**最低美元成本**实现最高性能的理想选择，具有更快的训练速度和更高的总参数量。
- **现代 Dropless MoEs 表现更优**：由于各种技巧和优化，现代 Dropless **Mixture of Experts (MoEs)** 的表现优于几何平均规则（`effective parameter count ~= sqrt(active * total parameters)`）。
   - MoEs 的训练速度通常是相同参数量密集网络（dense networks）的 **2.2 到 2.5 倍**，最近的模型甚至达到了 **3.2 倍**，百度发布的 **ERNIE-MoE** 训练 **MGH** 约为 **50%**，表明仍有进一步提升的空间。
- **Kimi K2 挑战 Claude**：尽管 **Kimi K2** 更为稀疏，但它能与被认为是大型密集模型的 **Claude** 竞争。
   - Kimi K2 的竞争优势可能源于大量的 **RL (Reinforcement Learning)** 而非架构，因为之前的开源模型在 Agentic 方面与 Claude 相比存在差距。
- **最优激活参数量随总参数量扩展**：最优激活参数量随总参数规模增加，但激活参数量呈线性或略低于线性扩展，而总参数量则以更高的线性速率或超线性速率增加。
   - 例如，**100B 总参数量配合 1B 激活参数量**是可能的，但限制因素是训练运行的成本；一位成员链接了来自 @LTSI_TheCaptain 的推文：[32K 上下文长度是否足够](https://x.com/LTSI_TheCaptain/status/1950272036074582040)。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1399476849406378066)** (10 条消息🔥): 

> `YouTube shorts、TikTok 算法、个性化内容、ChatGPT 学习模式` 


- **YouTube 算法推送 Shorts，赶走忠实用户**：成员们认为 YouTube 推送 **shorts** 是由于来自 **TikTok** 的威胁，却没有意识到 YouTube 用户讨厌 TikTok，且 **shorts 的变现能力远低**。
   - 其他人认为推送 **shorts** 更多是为了市场份额。人们在非 YouTube 的视频平台上花费时间，这意味着收入损失，这在商业上是不可接受的。
- **TikTok 的简单算法击败了 YouTube**：一位成员表示 **TikTok 的推荐算法** 更简单，但比 YouTube 的*彻底得多*。
- **未来的个性化内容和生成式内容**：一位成员表示，*未来你会看到推荐和生成式内容开始融合，我们将推荐内容的个性化版本，未来甚至可能不再是推荐内容，而是开始创造内容。*
- **ChatGPT 学习模式是终局吗？**：成员们对 [OpenAI 的 ChatGPT 学习模式](https://openai.com/index/chatgpt-study-mode/) 做出反应，认为它*正向终局迈进*。
   - 另一位成员认为*这违反了 OpenAI 的商业模式，该模式旨在最大限度地颠覆正规教育*。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1399489266320674826)** (22 条消息🔥): 

> `区块链授权、Monday.com AI 双人组、MCP 用户上下文隔离、EC2 上的 MCP Server、BDD 侧边项目` 


- **用于授权与支付的区块链**：一名成员认为，与纯 Web2 环境相比，在**区块链**上更容易实现授权管理和支付护栏，因为*具有极精确控制和可验证性的统一账户是区块链天然提供的特性*。
   - 他们建议像应用商店一样，统一跨模型提供商的支付解决方案，特别是针对本地或自托管模型。
- **Monday.com 聘请新锐 AI 双人组**：[Monday.com](https://www.monday.com) 招聘了两名 AI 工程师，祝贺[原文链接在此](https://www.calcalistech.com/ctechnews/article/rjorfh8wel)。
   - 新员工表示他们将全职从事 MCP 相关工作。
- **MCP 用户隔离受到关注**：一名成员试图了解单个云端部署的 **MCP server 实例**是否需要额外的用户上下文隔离层，以防止不同会话之间的数据共享，并引用了 [MCP git issues](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1087) 和 [MCP docs](https://modelcontextprotocol.io/docs/tutorials/use-remote-mcp-server#understanding-remote-mcp-servers)。
   - 他们强调数据安全是用户关注的严肃话题。
- **EC2 部署遇到连接难题**：一名成员在 **EC2** 上部署 **MCP server** 并正确配置了 SSL 证书和域名设置，但在使用 Claude Desktop 时遇到连接问题。
   - 他们报告在 Cursor 上运行成功，但在 **Claude Desktop** 上失败。
- **BDD 模型学习用于自动化的 Gherkin**：一名成员分享了一个基于**行为驱动开发 (BDD)** 的侧边项目，该项目已达到生产就绪状态。手动任务涉及使用简单的 YAML 文件将站点映射为页面对象，如[此图](https://cdn.discordapp.com/attachments/1312302100125843479/1399854833565044756/bael.jpeg?ex=688a83e8&is=68893268&hm=6494abf36dd08df040d69d5ea31c4c3335943841d9315c2c6a4fd247c8dfb529)所示。
   - 通过 Cucumber，功能流被转录，以便 LLM 模型可以学习映射到 Cucumber 上的自然语言 Gherkin。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1399568580449800253)** (6 条消息): 

> `VS Code MCP 扩展、MCPJam Inspector、Nexus 移动应用商店` 


- ****一键**将 MCP Server 安装至 VS Code**：一名成员创建了网站 [vscodemcp.com](https://vscodemcp.com/)，提供**一键安装按钮**，以便将 **MCP Server 添加到 VS Code**。
   - 还制作了一个[演示视频](https://youtu.be/1JcRtQxmh3I)来解释该过程。
- ****MCPJam** 获得 **Ollama** 支持**：**MCPJam**（一个开源的 MCP inspector 替代方案）现在支持 **Ollama**，允许用户在任何 Ollama 模型上测试其 MCP server，而无需承担高昂的 API 成本。
   - 创建了一个命令快捷方式 `npx @mcpjam/inspector@latest --ollama llama3.2` 来**快速启动 MCPJam 和本地 Ollama 模型**。
- ****Nexus**：AI 工具移动应用商店发布**：一名成员发布了 **Nexus** 的 Alpha 版本，这是一个 AI 工具（MCP servers）的移动应用商店，具有**一键安装**、**无需 JSON 配置**和**聊天集成**等特点。
   - Nexus 可在 [getnexus.in](https://www.getnexus.in) 获取，其源代码可在 [GitHub](https://github.com/lucky-builds/get-nexus) 上找到。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/)** (1 条消息): 

dhar007: 那是新的 DSPy optimizer，对吧 🙂
  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1399545934454460476)** (1 条消息): 

> `optimizer` 


- **不祥的 Optimizer 被过度承诺**：一名成员表示，*总有一天你会做出一个好到不愿分享的 **optimizer***。
   - 他们发布了一张描绘**绿魔 (Green Goblin)** 的[图片](https://cdn.discordapp.com/attachments/1203568372667645963/1399545934190477383/3195gc.png?ex=688a0cf9&is=6888bb79&hm=b5ef1dc40c8d2e735f8370a1be34553a2fd7cb46d86a6e67adab2b7ec0350fc3&)。
- **Optimizer 觉醒**：一个如此强大的 optimizer，可能永远不会面世。
   - 该图像暗示了一种如此强力的创造物，以至于其创造者可能会倾向于将其保密。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1399466466071281754)** (23 messages🔥): 

> `GEPA: Reflective Prompt Evolution, Optimizing tool response feedback, dspy.Variable and dspy.Parameter, AI Engineer specializing in agentic systems` 


- **GEPA: 反思性提示词演化 (Reflective Prompt Evolution)**：一名成员询问了基于 [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457) 的比较指标。
   - 另一名成员指出，**作者就在频道中**，可以直接提问。
- **优化工具响应反馈**：一名成员建议*不仅要优化提示词，还要在出错时优化工具响应反馈*。
   - 另一名成员确认**这是可行的**，并提到他们的实验构建了一个代码生成/优化流水线，该流水线使用外部编译器、分析器（profilers）和运行时执行器作为工具，提供文本反馈以供优化。
- **深入探讨 dspy.Variable 和 dspy.Parameter**：成员们讨论了 `dspy.Variable` 或 `dspy.Parameter`，将其描述为可在程序中使用的*某种可学习参数*。
   - 一名成员建议 `dspy.Variable` 可以允许用户**指定哪些部分应该是可优化的**，甚至建议 **DSPy 可以测试并重写其自身的部分源代码**。
- **资深 AI 工程师加入**：一位专注于 **Agent 系统、工作流自动化以及可扩展无代码/低代码架构**的资深 AI 工程师介绍了自己。
   - 他们提供在**设计和部署 AI Agents、自动化业务运营以及微调 LLM 和 LSM** 等领域的帮助，并列出了他们使用的工具，包括 **Vapi AI, Make.com 和 OpenAI**。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1399470814440460368)** (19 messages🔥): 

> `external_call usage in Mojo, C ABI function calls, Mojo standard library development, File descriptor features, Mojo module naming feature request` 


- ****external_call** 仅调用 C 函数**：Mojo 中的 `external_call` 函数可以调用任何 **C ABI 函数**，只要该符号已链接到当前地址空间。
   - Mojo 团队倾向于将额外的 C 代码量保持在最低限度，在 Mojo 代码不可行时才需要提供使用 C 代码的理由。
- **Mojo 具备文件描述符特性**：Mojo 已经通过 `io.file_descriptor.FileDescriptor` 包含了文件描述符功能，团队的目标是尽量减少在这些功能中使用 `external_call`。
   - 在操作系统层级使用 `read` 和 `write` 可以实现更好的可移植性，这符合 Mojo 的目标。
- **Mojo 标准库（stdlib）开发不需要触动编译器**：会议指出，大多数功能都可以在 **Mojo 标准库**中实现，而无需修改编译器本身。
   - 这种方法利用了 Mojo 的灵活性及其 FFI 能力，编译器只需要具备*调用这个具有 C ABI 的符号*的能力即可。
- **Mojo 一致性模块命名请求**：已创建一个 [功能请求 (feature request)](https://github.com/modular/modular/issues/5094) 以建立 Mojo 模块的一致命名规范。
   - 目标是将命名与 Python API 解耦，旨在减少混淆并避免 Python 沿用了 30 年的命名系统的限制。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1399512166393970688)** (4 messages): 

> `PyTorch 2.7 dependency, Max's PyTorch version, Nightly Builds, Minimum PyTorch Version` 


- **Max 很快将移除 PyTorch 2.7 依赖™️**：**PyTorch 2.7 依赖**计划在下一个 Nightly 版本中移除，从而赋予用户使用独立 PyTorch 版本的自由。
   - 这一转变归功于一名用户和团队其他成员的*出色工作*。
- **Max 对 PyTorch 的下限要求：2.0**：虽然最低固定版本是 **2.5**，但团队认为 **2.0** 是 Max 与 PyTorch 兼容的实际下限。
   - 建议用户在管理其 PyTorch 环境时考虑这一点。


  

---

### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1399493379724939334)** (1 条消息): 

> `FlowMaker, LlamaIndex office hours, 金融文档 Agent, S3VectorStore, LlamaParse 页眉页脚检测` 


- **FlowMaker 发布：可视化 Workflows 成为现实**：LlamaIndex 推出了 **FlowMaker**，这是一个带有可视化 GUI 的全新工具，用于构建 LlamaIndex workflows，访问地址为 [flowmaker.llamaindex.ai](https://flowmaker.llamaindex.ai/)。
- **LlamaIndex 和 LlamaCloud 更新发布**：**LlamaIndex** 现在通过全新的 [`S3VectorStore`](https://docs.llamaindex.ai/en/stable/examples/vector_stores/S3VectorStore/?utm_source=discord) 支持 **S3**，而 **LlamaParse** 增加了新的[页眉和页脚检测](https://docs.cloud.llamaindex.ai/llamaparse/features/parsing_options?utm_source=discord)功能。
- **LlamaCloud 的 n8n 节点开源**：新的开源 **n8n nodes for LlamaCloud**（包括 LlamaCloud indexes、LlamaParse 和 LlamaExtract）现已在 [`n8n-llamacloud` 仓库](https://github.com/run-llama/n8n-llamacloud)中可用。
- **Gemini Live 语音 Agent 完美集成**：通过 `pip install llama-index-voice-agents-gemini-live!` 即可使用与 **Gemini Live voice agent** 的新集成，示例代码请参考[此处](https://github.com/run-llama/gemini-live-demo)。
- **NotebookLlaMa 焕然一新**：**NotebookLlaMa** 现在提供定制化的播客生成功能，以及一个用于查看已处理文档的文档管理 UI。
   - 欲了解更多详情，请查看 [LlamaIndex FlowMaker 介绍视频](https://youtu.be/MU6jA0rUlFY?feature=shared)、[如何在 n8n 中使用新的 LlamaCloud 节点视频](https://youtu.be/5bQXHPSkuBw?feature=shared) 以及 [使用 LlamaParse 进行多模态报告生成视频](https://youtu.be/--BpWmuUmbA?feature=shared)。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1399466840710844638)** (5 条消息): 

> `Agent 设计模式, 网页抓取 AI Agent, 适用于 n8n 的 LlamaCloud 节点, AI 文档 Agent, LlamaCloud 托管 Embeddings` 


- **Agent 设计模式揭秘**：Seldo 在最新的 @aiDotEngineer 峰会演讲中，深入分析了在大规模应用中成功与失败的 Agent 设计模式，涵盖了**混合 workflows**、**自主性 vs 结构化**以及**可调试性**，可通过[此链接](https://t.co/zeTaVOMate)访问。
- **高性价比的网页抓取 AI Agent**：LlamaIndex 推出了一项新集成，利用 @OxyLabs 的网页抓取基础设施，能够构建实时搜索和抓取互联网任何网站的高性价比 AI Agent，详情见[此处](https://t.co/tqZuj0nH11)。
- **LlamaCloud 节点增强 n8n 工作流**：正如[此更新](https://t.co/etmo0pTAc5)所强调的，LlamaIndex 简化了在现有 @n8n_io 工作流中添加智能文档处理的过程。
- **AI 转换复杂金融文档**：即将举行的网络研讨会将展示如何利用 LlamaCloud 的企业级解析和处理能力，通过 AI 驱动的文档 Agent 将复杂的金融文档转换为可操作的数据 [链接](https://t.co/f0TKSzHQ2d)。
- **LlamaCloud 简化 Embeddings 管理**：LlamaCloud 推出了托管 Embeddings，用户在使用 LlamaCloud Indexes 时无需提供自己的 API key 即可对内容进行嵌入，更多详情见[此处](https://t.co/tu85qFt3if)。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1399552037900779661)** (13 条消息🔥): 

> `将 Flowmaker 导出为 Python, LlamaCloud PDF 检测问题, 文件扩展名命名规范` 


- **关于 Flowmaker 导出 Python 的探讨**：一位成员询问了将 **Flowmaker** 导出为 **Python** 的可能性。
   - 另一位成员回复称 **Flowmaker** 目前导出为 **Typescript**。
- **LlamaCloud PDF 处理问题**：一位成员报告称 **LlamaCloud** 无法通过 **API** 检测和处理某个 **PDF 文件**（使用 **n8n** 简化工作流），并请求协助，同时提供了一张[截图](https://cdn.discordapp.com/attachments/1059201661417037995/1399848961832911031/Screenshot_2025-07-29_at_22.13.16.png?ex=688a7e70&is=68892cf0&hm=213304e9d8a77128ab3cbc75d4c9114a73d0a157e12a0aa633bd2a62e160a5fa)。
- **文件名修复设想**：一位成员建议在与 **LlamaCloud** 交互时，确保文件名包含正确的**文件扩展名**。
   - 这是针对 **LlamaCloud** 无法检测 **PDF 文件**这一报告问题的回应。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1399470013836165182)** (15 messages🔥): 

> `Grok 用于提示词编写, Manus 积分系统, Agentic Systems 对比 (Lume vs. Suna)` 


- **Grok 不擅长提示词编写**：一位成员建议使用 **Grok** 为 **Manus AI** 生成详细的提示词，但另一位成员反馈其生成的结果非常糟糕（*shit*）。
   - 第一位成员主动提出提供个人帮助。
- **成员批评 Manus 的积分系统和缺乏更新**：一位成员声称，尽管举办了活动，但 **Manus** 的积分系统和缺乏更新将导致其走向衰落。
   - 另一位成员建议尝试其他 Agent 系统（如 **Lume**）以确定最佳价值。
- **Lume vs. Suna 对比**：成员们讨论了 **Lume** 和 **Suna** 的性能，一位成员表示 *Lume 就是 Suna，但效果更差*。
   - 另一位成员发现 **Lume** 在编程任务中表现更优，错误更少且代码已调试，但认为 **Manus** 在漫画创作方面表现良好。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1399496435376062626)** (9 messages🔥): 

> `Tinygrad 中的 Tensor 实现, 已关闭的 PR #11410 分析, 替代实现思路` 


- **Tinygrad Tensors：原生还是包装？**：一位成员询问 **tinygrad** 的 **Tensor** 类是包装了现有对象（如 **NumPy ndarrays**），还是定义了自己的实现。
   - 该询问还涉及如果使用包装器，tinygrad 如何补偿性能损失。
- **PR #11410 的关闭引发讨论**：一位成员对 [PR #11410](https://github.com/tinygrad/tinygrad/pull/11410) 在推送更新后不久就被无理由关闭表示惊讶。
   - 另一位成员表示 *该 PR 没抓到重点，不是一个好的改动*，并建议原作者回顾之前的合并和关闭记录，以更好地理解贡献指南。
- **"Where" 操作争议**：一位成员提到尝试将分配的操作保留到 Kernel 化/调度创建阶段，但在看到 geohot 的评论后重新考虑使用 *where* 操作。
   - 他们承认可能存在的副作用，并对 PR 在没有反馈的情况下被关闭感到惊讶，因为他们正计划进行更深入的调查。


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1399593696651116607)** (4 messages): 

> `BFCLv4, 开源 Agent 系统, API Key 提供, 多 Agent 系统` 


- **BFCLv4 限制仅提交模型**：一位成员询问 **BFCLv4** 是否允许开源 **Agent 系统**或提供 **API Key**。
   - 另一位成员澄清说，*目前*他们*仅接受针对单一模型的提交*。
- **关于多 Agent 系统提交的说明**：一位成员询问是否可以提交包含多个模型的**多 Agent 系统**。
   - 官方确认 **BFCLv4** 的提交目前仅限于**单个模型**。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1399896806522884186)** (1 messages): 

> `LoRA 风格适配器, TorchTune 支持` 


- **Torchtune 询问关于 LoRA 风格适配器的支持**：一位成员询问 **Torchtune** 目前是否支持 **LoRA 风格适配器**。
   - 询问者指定该适配器应保留精确的前向计算路径，并在通过额外的可训练层应用更新时冻结原始模型权重。
- **LoRA 适配器保留计算路径**：用户希望 LoRA 适配器保持前向计算路径，不减少 GEMM 维度，也不改变计算成本。
   - 目标是冻结原始模型权重，并通过额外的可训练层应用更新。


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1399479768604479680)** (2 messages): 

> `RL 测试, CI 调试` 


- **RL 测试运行时间过长，成员怀疑存在 Bug**：一位成员质疑为什么 **RL 测试**运行时间超过 **1 小时**，称其为 *100% 的 Bug*。
   - 他们表示将开启一个单独的 **PR** 来调试 **CI**。
- **提议进行 CI 调试**：该成员计划创建一个单独的 **Pull Request (PR)**。
   - 该 PR 将专门用于调试**持续集成 (CI)** 系统。


  

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1399831422461804604)** (2 条消息): 

> `Diffusion Models, Study Group, Generative AI, MIT Curriculum` 


- ****深入探索 Diffusion Models****：一个新的**学习小组**正在组建，旨在从零开始学习 **Diffusion Models**（**Generative AI** 的核心架构）。该课程基于 [MIT 课程大纲](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf)，为期 **5 个月**（**每周 2-4 小时**）。
- ****Diffusion Models 免费入门课程****：计划于 **8 月 2 日**和 **8 月 9 日**（EST 时间中午 12 点）举行**两场免费入门课程**，涵盖 **Flow Matching**、实际应用案例、**PDEs**、**ODEs**、**SDEs** 以及 Diffusion Models 的简要历史 ([课程链接 1](https://lu.ma/kv8zf6va), [课程链接 2](https://lu.ma/uk6ecrqo))。
- ****学习小组详情与亮点****：该小组专为 **AI** 从业者设计，已确认的成员包括 **AI 电影工具的 CTO**、**AI 艺术讲师**、**LLM 讲师**和 **AI 研究员**。
   - 前**两节课免费**，之后早鸟价为 **$50/月**（之后为 **$100/月**），用于支付助教费用；亮点包括**同行引导的课程**、**导师 Q&A**、**实战项目**以及真实的科研论文。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1399564477577564361)** (2 条消息): 

> `Nomic dataset Access, contrastors repo, model Selection` 


- **Nomic 数据集访问问题**：一名成员报告在按照 [contrastors repo](https://github.com/nomic-ai/contrastors) 的说明尝试访问 **nomic-ai/nomic-embed-text-v2-moe** 数据集时遇到 **AccessDenied 错误**。
   - 该成员使用了命令 `aws s3 ls --endpoint-url=https://9fa58365a1a3d032127970d0bd9a1290.r2.cloudflarestorage.com/ s3://contrastive`，并在执行 `ListObjectsV2` 操作时收到错误。
- **寻求低配系统的模型推荐**：一名拥有 **Celeron N2815**、**4GB RAM** 且无 GPU 的成员请求关于在其系统上运行哪种模型最好的建议。
   - 在提供的消息中未推荐具体模型。


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1399857682709483602)** (1 条消息): 

> `Introductions, Community Hopes` 


- **成员自我介绍**：新成员开始介绍自己，分享他们的**公司/行业/大学**以及**正在研究的项目**。
   - 许多人详细说明了他们**最喜欢的技术/工具**以及**希望从社区中获得什么**。
- **新成员加入社区**：Cohere 的社区 Discord 服务器欢迎新成员，并邀请他们进行自我介绍。
   - 介绍模板要求提供**公司/行业/大学**、当前项目、最喜欢的技术/工具以及社区目标的详细信息。


  

---


---