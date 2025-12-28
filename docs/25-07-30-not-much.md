---
companies:
- zhipu-ai
- alibaba
- moonshot-ai
- x-ai
- figure
- openai
- runway
- mlx
- ollama
- deeplearningai
date: '2025-07-30T05:44:39.731046Z'
description: '**中国 AI 实验室**发布了多款强大的开源模型，包括**智谱 AI** 的 **GLM-4.5** 和 **GLM-4.5-Air**、**阿里巴巴**的
  **Qwen3 Coder** 和 **Qwen3-235B**，以及**月之暗面 (Moonshot AI)** 的 **Kimi K2**，这标志着许可宽松的开源模型正在大量涌现。


  **智谱 AI 的 GLM-4.5** 是一款拥有 3550 亿参数的 MoE（混合专家）模型，其性能可与 **Claude 4 Opus** 和 **Gemini
  2.5 Pro** 媲美。**阿里巴巴的 Qwen3 Coder** 在代码生成方面表现强劲且编辑失败率极低，而**月之暗面的 Kimi K2** 则是一款万亿参数的
  MoE 模型，在 **LiveCodeBench** 等基准测试中表现卓越。


  在视频和图像生成领域，**xAI** 推出了 **Grok Imagine**，而 **Wan2.2** 凭借其创新的图生视频（image-to-video）技术给人留下了深刻印象。机器人领域的进展包括
  **Figure** 公司的 **Figure-01** 和 **Figure-02** 人形机器人，以及用于篮球分析姿态估计的 **ViTPose++**。**SmolLM3**
  的训练和评估代码已在 Apache 2.0 协议下完全开源。


  此外，**OpenAI** 在 **ChatGPT** 中引入了“学习模式”（Study Mode）以增强互动学习体验，**Runway** 则推出了 **Runway
  Aleph**，这是一款用于多任务视觉生成的全新上下文视频模型。社区注意到，避开这些中国开源模型的组织将面临竞争劣势。正如 @corbtt 所言：“避开这些模型的组织正处于显著的竞争劣势。”'
id: MjAyNS0w
models:
- glm-4.5
- glm-4.5-air
- qwen3-coder
- qwen3-235b
- kimi-k2
- grok-imagine
- wan-2.2
- smollm3
- figure-01
- figure-02
- vitpose++
- chatgpt
people:
- yuchenj_uw
- corbtt
- reach_vb
- ollama
- deeplearningai
- gdb
- sama
- c_valenzuelab
- adcock_brett
- skalskip92
- loubnabenallal1
- hojonathanho
- ostrisai
title: 今天没发生什么事。
topics:
- model-releases
- model-performance
- moe
- image-generation
- video-generation
- pose-estimation
- robotics
- training-code-release
- interactive-learning
- in-context-learning
---

**平静的一天。**

> 2025年7月29日至7月30日的 AI 新闻。我们为您检查了 12 个 subreddits、544 个 Twitter 账号和 29 个 Discord 社区（227 个频道，5378 条消息）。预计节省阅读时间（以 200wpm 计算）：467 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以美观的 vibe coded 方式呈现所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上给我们反馈！

由于 Twitter 上匿名人士的随机猜测，关于 GPT5 将于明天发布的炒作甚嚣尘上。

---

# AI Twitter 回顾

**模型发布与性能**

- **中国的开源攻势**：7 月份，中国实验室发布了一波功能强大且许可宽松的模型，[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1950034092457939072) 强调了这一趋势。主要发布包括来自 **Zhipu AI** 的 **GLM-4.5** 和 **GLM-4.5-Air**、**Wan-2.2**（视频）、来自 **Alibaba** 的 **Qwen3 Coder** 和 **Qwen3-235B** 系列，以及来自 **Moonshot AI** 的 **Kimi K2**。这与西方开源发布速度感官上的放缓形成对比，促使 [@corbtt](https://twitter.com/corbtt/status/1950334347971874943) 指出，避开这些模型的组织正处于“显著的竞争劣势”。
- **Zhipu AI 的 GLM-4.5 模型**：**Zhipu AI** 发布了 **GLM-4.5**（一个 355B 参数的 MoE 模型，激活参数 32B）和 **GLM-4.5-Air**，两者均采用 **MIT 许可证**。该公司宣布由于需求量大，[他们正在努力扩展资源](https://twitter.com/Zai_org/status/1950164491125043515)。这些模型被认为可以与 **Claude 4 Opus** 竞争，并在[某些基准测试](https://twitter.com/Zai_org/status/1949970927006949430)中击败了 **Gemini 2.5 Pro**。社区迅速在 **MLX** 和 **DeepInfra** 等平台上提供了这些模型。
- **Qwen3 和 Kimi K2 模型**：**Alibaba** 的 **Qwen3 Coder** 表现强劲，在 **Cline** 中的 **diff edit failure rate** 仅为 **5.32%**，据 [@cline](https://twitter.com/cline/status/1949973297455599998) 称，这使其与 **Claude Sonnet 4** 和 **Kimi K2** 并列。[@reach_vb](https://twitter.com/reach_vb/status/1950263476271947822) 和 [@ollama](https://twitter.com/ollama/status/1950291777216262259) 指出，具有 256K context 的 **30B MoE (3B 激活)** 版本现在可以通过 **MLX** 和 **Ollama** 在本地运行。**Moonshot AI** 的 **Kimi K2** 是一款 **1 万亿参数的 MoE (32B 激活)** 模型，以修改后的 MIT 许可证发布，在 **LiveCodeBench** 和 **AceBench** 等基准测试中超越了其他 open-weights 模型，[据 @DeepLearningAI 报道](https://twitter.com/DeepLearningAI/status/1950183277161005418)。
- **视频与图像生成**：**xAI** 推出了 **Grok Imagine**，这是一款图像和视频生成工具，目前处于 [候补名单阶段](https://twitter.com/chaitualuru/status/1949946519869685952)。**Wan2.2 5B** 视频模型在 **Image-to-Video (I2V)** 方面的方法给开发者留下了深刻印象，其中每个 latent frame 都有自己的 denoising timestep，据 [@ostrisai](https://twitter.com/ostrisai/status/1950129158618591646) 分析，这可能允许生成无限长的视频。**Ideogram** 发布了 **Ideogram Character**，这是一个角色一致性模型，可以使用单张参考图工作，[@hojonathanho](https://twitter.com/hojonathanho/status/1950261122365333806) 提到了这一点。
- **视觉与机器人**：**Figure** 展示了其 **Figure-01** 与新型 **Figure-02** 人形机器人的对比，在 [@adcock_brett](https://twitter.com/adcock_brett/status/1950291267730207125) 分享的视频中强调了硬件和能力的进步。**ViTPose++** 展示了令人印象深刻的姿态估计，能够准确跟踪篮球运动员之间复杂的互动，据 [@skalskip92](https://twitter.com/skalskip92/status/1950231824933982428) 称，这目前正被集成到一个篮球分析 AI 中，该 AI 可以判断球员是否在禁区内。
- **SmolLM3 代码发布**：**SmolLM3** 的完整训练和评估代码已发布，包括预训练脚本 (**nanotron**)、后训练代码（用于 SFT+APO 的 **TRL/alignment-handbook**）和评估脚本，以及 100 多个中间 checkpoints，全部采用 **Apache 2.0 许可证**，由 [@LoubnaBenAllal1](https://twitter.com/LoubnaBenAllal1/status/1950139809034305568) 宣布。

**AI Agents、工具与应用**

- **ChatGPT Study Mode**: **OpenAI** 正在 **ChatGPT** 中推出 **Study Mode**（学习模式），这是一个旨在引导用户逐步学习概念的交互式功能，其角色更像是导师而非仅仅提供答案，正如 [@gdb](https://twitter.com/gdb/status/1950309323936321943) 和 [@sama](https://twitter.com/sama/status/1950299705751327149) 所宣布的那样。
- **Runway Aleph In-Context Video Model**: **Runway** 正在开放 **Runway Aleph** 的访问权限，这是一个用于多任务视觉生成的全新 In-Context 视频模型。[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1950138170806312974) 通过对比展示了其强大功能：将原本复杂的“昼夜交替”效果手动视频编辑工作流，简化为仅需向 Aleph 发送“变黑夜”的 Prompt。类似的对比还包括[从场景中移除汽车](https://twitter.com/c_valenzuelab/status/1949921138689396976)和[添加爆炸效果](https://twitter.com/c_valenzuelab/status/1950257984715571606)。
- **Google's AI Mode in Search**: **Google** 将其搜索中的 **AI Mode** 扩展到了英国，并引入了新功能，包括上传照片和 PDF 进行查询、用于整理项目的 “Canvas”（画布）以及用于实时帮助的 “Search Live”，详见 [@Google](https://twitter.com/Google/status/1950241246779232260) 的说明。
- **LangChain & LangGraph for Agentic Workflows**: **LangChain** 发布了一份关于使用 **LangGraph** 应用六种常见 Context Engineering（上下文工程）方法的指南，并在[一条热门推文](https://twitter.com/LangChainAI/status/1950226846538485918)中提供了视频和代码示例。他们还重点介绍了如何构建一个用于代码生成的自我纠错 RAG Agent。生态系统持续增长，[**LangSmith Traces** 现在集成了服务器日志](https://twitter.com/LangChainAI/status/1949948616182768010)以实现更好的可观测性。
- **Perplexity's Comet Browser**: **Perplexity** 的 **Comet** 浏览器在初期获得了强劲的采用率，CEO [@AravSrinivas](https://twitter.com/AravSrinivas/status/1950042752655241234) 指出其默认搜索是 **Perplexity**，这可能会带来巨大的查询量。他还演示了 Comet 执行复杂任务的能力，例如[在美联航（United）上订票，包括选座](https://twitter.com/AravSrinivas/status/1949937085164482846)。
- **Development & Tooling**: **BlockDL** 是一个用于可视化设计 **Keras** 神经网络的免费开源 GUI，由 [@fchollet](https://twitter.com/fchollet/status/1950244806967603207) 发布。在工具方面，新的 **Hugging Face jobs CLI** 现在由 **uv** 驱动，以实现更快的环境搭建，正如 [@_lewtun](https://twitter.com/_lewtun/status/1949915717836431744) 所分享的。对于构建 Agent 应用的开发者，[@_avichawla](https://twitter.com/_avichawla/status/1950282234893656101) 强调了一种仅需 10 行代码即可将任何模型、RAG 或 Agent 部署为 **MCP server** 的方法。

**Infrastructure, Efficiency & Optimization**

- **Long Context Training on H200**: [@StasBekman](https://twitter.com/StasBekman/status/1950232169227624751) 证明了现在可以在单块 **H200 GPU** 上对 Llama-8B 模型进行 **120 万（1.2M）序列长度**的训练。这是通过结合 **ALST**、**FA3 (FlashAttention-3)** 和 **Liger-Kernel** 实现的，后两者最近刚修复了 int64 索引问题。
- **GSPO in TRL**: 备受关注的阿里巴巴 **GSPO (Group Sequence Policy Optimization)** 算法现已集成到 Hugging Face 的 **TRL** 库中，由 [@_lewtun](https://twitter.com/_lewtun/status/1949951668914659636) 宣布。
- **AMD Contributions to llama.cpp**: [@ggerganov](https://twitter.com/ggerganov/status/1950047168280060125) 指出 **AMD** 团队现在正积极为 **llama.cpp** 代码库做出贡献，标志着这一流行推理框架获得了更广泛的硬件支持。
- **StepFun Open Sources StepMesh**: 中国 AI 公司**阶跃星辰 (StepFun)** 开源了 **StepMesh**，这是一个专为使用 **Attention-FFN disaggregation**（Attention-FFN 解耦）的推理系统设计的通信库，[@teortaxesTex](https://twitter.com/teortaxesTex/status/1950127131754651655) 对此进行了说明。
- **Qdrant Edge for On-Device Vector Search**: **Qdrant** 推出了 **Qdrant Edge** 的私测版，这是一个轻量级的嵌入式向量搜索引擎，旨在为机器人、移动端和物联网（IoT）应用提供端侧运行能力，由 [@qdrant_engine](https://twitter.com/qdrant_engine/status/1950165409639833603) 宣布。

**Research, Techniques & Evaluation**

- **Backpropagation 的历史**：[@SchmidhuberAI](https://twitter.com/SchmidhuberAI/status/1950194864940835159) 提供了 **Backpropagation** 的详细历史，澄清了其现代形式最早由 **Seppo Linnainmaa** 在 **1970** 年发表，其前身可追溯至 **1960** 年的 **Henry J. Kelley**。他强调这不仅仅是链式法则，而是它在神经网络中的一种高效应用。
- **评估危机**：一种日益增长的观点认为标准 Benchmark 正变得越来越不可靠。[@ShunyuYao12](https://twitter.com/ShunyuYao12/status/1950090043344707832) 问道：“当我们不再信任 Benchmark 数字时，该如何评估 LLM？”。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1949912968394940518) 呼应了这一点，表示当一个模型伴随着一套“激进的全新评估套件”发布时，才会令人兴奋。**DailyBench** 由 [@jacob_dphillips](https://twitter.com/andersonbcdefg/status/1949936665637593102) 发布，作为一个自动化的每日 Benchmark，用于在新鲜问题上追踪前沿模型。
- **新的优化技术**：一篇关于 **Reflective Prompt Evolution** 的论文显示其性能可以超越 **GRPO**，突显了通过自然语言反思进行学习的力量，正如 [@lateinteraction](https://twitter.com/lateinteraction/status/1949984215191208078) 所分享的。阿里巴巴的 **Group Sequence Policy Optimization (GSPO)** 论文是 Hugging Face 7 月份排名第三的热门论文，[@ClementDelangue](https://twitter.com/ClementDelangue/status/1949934196148895799) 预测它将产生巨大影响。
- **LLM 的物理学**：研究人员发布了他们“Physics of Language Models”工作的代码，声称其 **8B@1T** 模型仅使用 **7% 的算力** 就击败了 **Llama-3.1-8B**，正如 [@giffmana](https://twitter.com/giffmana/status/1950276478861517236) 所分享。
- **推理与意识**：关于什么是推理的讨论出现了，[@teortaxesTex](https://twitter.com/teortaxesTex/status/1950158521493811458) 挑衅性地建议它是一种应该能够解决停机问题的“超图灵计算（super-Turing computation）”。与此同时，[@jxmnop](https://twitter.com/jxmnop/status/1950229423849869672) 回忆起该领域如何从争论 GPT-2 是否理解否定，演变为讨论“近乎有感知力的、能赢得 IMO 的”模型。

**行业与更广泛的讨论**

- **4 亿美元 Meta Offer 的故事**：讨论的一个重点是，顶尖 AI 人才拒绝了来自 **Meta** 的 **4 亿美元** offer，[来自 @willdepue 的一条推文走红](https://twitter.com/willdepue/status/1950253835064086979)。这引发了人们对其他公司正在构建什么的猜测，这些东西竟然能激励研究人员拒绝如此巨额的报酬。
- **能源瓶颈**：一位前 **Meta** 员工的评论浮出水面，指出 **能源** 是扩展算力的最大瓶颈，甚至超过了购买 GPU 的资金。[该推文被 @code_star 转发放大](https://twitter.com/code_star/status/1950263396420767845)。
- **API 与 Open Weights 安全性**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1950226004984942829) 反驳了基于 API 的模型天生比 Open Weights 模型更安全的观点。他认为，通过降低模型使用门槛，API 可以在不增加显著控制的情况下，将恶意行为者的滥用量增加“几个数量级”。
- **招聘与社区**：**Anthropic** 宣布[正在扩大其 Fellows 计划](https://twitter.com/EthanJPerez/status/1950278824102678586)，该计划将外部研究人员与内部团队配对，共同研究安全问题。**Sakana AI** 正在[举办开放日活动](https://twitter.com/SakanaAILabs/status/1950016555799953523)为其 Applied Engineer 团队招募人才。
- **地缘政治**：多条高曝光推文涉及政治气候，包括 **Speaker Pelosi** 批评 **Donald Trump** 关于台湾总统赖清德访问的决定，由 [@zacharynado](https://twitter.com/zacharynado/status/1950056521330532640) 分享。

**幽默与迷因**

- **发光的花园与建筑 Diffusion**：一条开玩笑说“现在我一半的花园都在黑暗中发光”的推文，在回应有关发光植物的新闻报道时[通过 @nptacek 获得了巨大关注](https://twitter.com/nptacek/status/1950265375658020991)。“他们对 *查看笔记* 一座房子进行了 Diffusion”的梗也广为流传，并被 [@sedielem 转发](https://twitter.com/sedielem/status/1950190227475046877)。
- **离奇历史与密码**：来自 [@DavidSHolz](https://twitter.com/DavidSHolz/status/1950104321783218193) 的一条热门推文分享了 20 世纪 30 年代的一项提议，即在金门大桥顶部建造一座 **时速 190 英里的过山车**。在另一篇走红的帖子中，[@jxmnop](https://twitter.com/jxmnop/status/1950272775052284351) 分享了一张用户密码为 “Woman” 的截图，并评论道“你编都编不出来”。
- **AI 恶搞**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1950323192641503571) 发布了一个使用 **Comet** 浏览器的“是热狗还是非热狗”示例。[@typedfemale](https://twitter.com/typedfemale/status/1950337102828143000) 发布了一个关于“双性恋 luke farritor”的梗图。
- **引起共鸣的工程师生活**：一篇关于被物理锁在房间里的帖子引起了人们对在项目中“锁入（locked in）”状态的共鸣，由 [@stevenheidel 发布](https://twitter.com/stevenheidel/status/1950316382450823320)。[@KevinAFischer 分享了](https://twitter.com/KevinAFischer/status/1949958038905127340)在 **a16z** 的一场包含“会说话的魔力狗和人体金字塔”的投融资路演。

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Qwen3-30B-A3B 及相关模型发布与性能讨论

- [**Qwen3 Coder 30B-A3B 明天发布！！！**](https://i.redd.it/zv92612t11gf1.png) ([Score: 429, Comments: 49](https://www.reddit.com/r/LocalLLaMA/comments/1md93bj/qwen3_coder_30ba3b_tomorrow/))：**该帖子提到了 Qwen3 Coder 30B-A3B 即将发布，这是 Qwen3 Coder 模型的一个 AutoAWQ 量化变体（暗示为 A3B），这表明开源 LLM 社区对此抱有高度期待。图片可能预热或确认了发布时间，助力了近期 Qwen 发布活动中凸显的竞争势头。来自用户评论的技术背景显示了对 Qwen 快速进步的热情，并暗示了其对竞争对手项目和人物（如 Llama/Lizard 和 Altman）的影响。** 评论者对 Qwen 的创新速度感到非常兴奋，认为它正在超越其他选择（“与 Lizard boy 的友谊结束了”），并幽默地引用了与 OpenAI 领导层的竞争紧张关系（“Scam Altman”）。
    - 一位评论者强调了 Qwen3 Coder 30B-A3B 变体与假设的所有参数完全激活的 Qwen3 Coder 32B 之间的区别，认为 30B-A3B 的发布可能涉及与利用全部 320 亿参数的模型相比，一定程度的参数停用或剪枝。这种区别可能会对运行这些模型时的性能或内存需求产生影响。
- [**🚀 Qwen3-30B-A3B-Thinking-2507**](https://i.redd.it/eaag1cpuz0gf1.jpeg) ([Score: 414, Comments: 104](https://www.reddit.com/r/LocalLLaMA/comments/1md8t1g/qwen330ba3bthinking2507/))：**这张图片似乎是新 Qwen3-30B-A3B-Thinking-2507 模型的非技术性宣传海报，强调了其在推理任务（数学、科学、代码）上的竞争性能、工具使用能力以及原生 256K-token 上下文窗口（可扩展至 1M）。Hugging Face 和 ModelScope 的链接提供了模型详情。该帖子是近期 Qwen3 系列发布的一部分，评论者注意到了即将推出的 coder 变体以及用于高效推理的 GGUF 量化版本的可用性。** 评论者对即将发布的版本（如 Qwen3-30B-A3B-Coder）表示兴奋，而其他人则讨论了用于实际使用的量化 GGUF 版本的可用性。一些用户对 Qwen 新版本发布的频率感到疲劳。
    - 分享了 Qwen3-30B-A3B-Thinking-2507 的基准测试，突出了其极具竞争力的性能。一张图片链接显示了该模型与其他 30B-70B 级 LLM 的排名对比，表明根据当前的评估基准，它在其参数级别中处于或接近顶尖水平。
    - 该模型的 GGUF 格式下载现已在 Hugging Face 上提供：https://huggingface.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF，确保了寻求适合使用 llama.cpp 或类似框架进行优化推理的量化版本的用户的可访问性。

- [**Qwen/Qwen3-30B-A3B-Thinking-2507 · Hugging Face**](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507) ([Score: 127, Comments: 32](https://www.reddit.com/r/LocalLLaMA/comments/1md8rxu/qwenqwen330ba3bthinking2507_hugging_face/)): **该帖子讨论了在 P40 24GB GPU 上运行 Qwen3-30B-A3B-Thinking-2507 模型（可在 Hugging Face 获取），使用 Q4 UD XL GGUF 量化，上下文窗口高达 90,000 tokens。性能基准测试显示，在 context 为 0 时解码速度约为 40 tokens/sec，在 40k context tokens 时下降到约 10 tokens/sec，在最大 context 下读取速度为 110t/s。该模型在涉及 GUI/游戏仪表盘创建的复杂多步代码生成任务中进行了评估，通过 Roo Code 进行迭代提示词处理，并证明与 Flash 2.5 模型相比，初始错误更少。** 评论者提供了技术资源（GGUFs），分享了基准测试环境的截图，并指出在高 context 下性能稳定，使 Qwen3-30B 适用于扩展的代码生成任务。
    - 一位用户报告在 P40 24GB GPU 上运行 Qwen3-30B-A3B-Thinking-2507，采用 Q4 UD XL 量化和 `90k` context 长度，在 0~10~40k context 下分别实现了 `40~25~10t/s` (tokens/sec) 的生成速度。他们描述了一个广泛的编程任务，涉及多步提示（9 个核心任务和几个修复提示），并在 40k context 窗口下观察到 `10t/s` 的写入速度和 `110t/s` 的读取速度，并指出其 *初始错误比 Flash 2.5 更少*（Flash 2.5 是他们常用的基准模型）。
    - 一位社区成员分享说，他们创建了 Qwen3-30B-A3B-Thinking-2507 的 GGUF 格式权重，并可在 [Hugging Face](https://huggingface.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF) 下载，从而促进了本地量化推理以及与 llama.cpp 等工具的更广泛兼容性。
- [**Qwen3-30b-a3b-thinking-2507 This is insane performance**](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507) ([Score: 341, Comments: 85](https://www.reddit.com/r/LocalLLaMA/comments/1md8slx/qwen330ba3bthinking2507_this_is_insane_performance/)): **该帖子询问最近发布的 Qwen3-30B-A3B 是否与 Qwen3-235B 的性能相匹配。评论者指出，Qwen3-30B-A3B 取得了与闭源 LLM API（例如 Gemini 2.5 Flash, Claude Haiku）相当的结果，且价格显著降低（**`$0.30-.45/million tokens`**），暗示了重大的成本效益转变。据报道，A3B 变体在没有 GPU 的廉价笔记本电脑上实现了** `5-10 tokens/second` **（量化后）的速度，且文档建议针对复杂任务使用极长的输出长度（**`32,768` **到** `81,920` **tokens），这挑战了最近建议较短“thinking”可能更优的文献。** 讨论强调了对主要供应商利润率的怀疑，以及对 Qwen3-30B-A3B 作为具有高性能和低成本的本地推理模型（特别是在边缘端，如笔记本电脑/CPU 级别）可行性的兴趣。技术辩论对比了最近提倡较短输出的论文与 Qwen3 建议在复杂基准测试中最大化输出长度的观点。
    - 几位评论者强调，Qwen3-30B-A3B-Thinking-2507 模型可以与闭源 LLM（如 Gemini 2.5 Flash/o3 mini/Claude Haiku）相媲美，而成本大幅降低——每百万 tokens 仅需 $0.30-0.45，而封闭产品的价格要高出 5-10 倍，尽管性能相似。
    - 该模型的效率值得注意：通过量化，用户报告在没有 GPU 的普通笔记本电脑上实现了 5-10 tokens/second 的生成速率，使大规模部署和本地推理对于非企业用户来说变得切实可行。
    - 技术建议引用了最近关于输出长度的研究，建议大多数查询使用 `32,768` tokens，对于复杂的数学或编程基准测试建议高达 `81,920` tokens，更长的输出特别能提高在挑战性任务上的表现。
- [**Kudos to Qwen 3 team!**](https://www.reddit.com/r/LocalLLaMA/comments/1md00oc/kudos_to_qwen_3_team/) ([Score: 125, Comments: 17](https://www.reddit.com/r/LocalLLaMA/comments/1md00oc/kudos_to_qwen_3_team/)): **该帖子讨论了阿里巴巴发布的 Qwen3-30B-A3B-Instruct-2507，赞扬了其质量，但指出较旧的 Qwen3-32B 目前在标准化基准测试中表现更好。楼主请求发布更多变体，如 Qwen3-32B Instruct/Thinking 和 Qwen3-30B-A3B-Thinking-2507，表明了对更广泛模型可用性和进一步基准测试对比的兴趣。评论者还表达了对高性能编程模型（14B 和 30B）的需求，并寻求关于新旧版本（Qwen3-32B）之间技术性能差距的澄清。** 技术讨论集中在对更多面向编程且具有竞争力的模型的需求，特别是那些能够超越 Sonnet 3.5 等既定基准的模型，以及关于 Qwen3-30B-A3B-Instruct-2507 与 Qwen3-32B 之间直接基准测试的询问。

- 讨论强调 Qwen-32B 在基准测试中通常优于 Qwen-30B，但这是以显著降低的推理速度和更高的硬件要求为代价的——具体而言，Qwen-32B 通常需要高显存 GPU，而 Qwen-30B 在包括某些 CPU 在内的低端硬件上更具可行性。这种划分突显了模型选择在性能与资源可访问性之间的权衡。
- 几位用户对 14B 参数范围内的 Qwen3 coder 模型表示出兴趣，认为其有可能超越 Sonnet 3.5，这反映了当前对高容量和专业化 coder 模型的关注，以及开源模型领域在对比竞争中的雄心。
- 针对 HuggingFace 上新版本 Qwen 缺乏实时推理（API）端点的问题，用户表达了技术上的沮丧，这影响了测试和部署工作流的即时可访问性。
- [**在折腾本地 AI 6 个月后，这是我整理的能满足 90% 需求的模型清单。你的是什么？**](https://i.redd.it/jzljyi4tw2gf1.jpeg) ([Score: 104, Comments: 65](https://www.reddit.com/r/LocalLLaMA/comments/1mdjb67/after_6_months_of_fiddling_with_local_ai_heres_my/)): **用户分享了他们本地 AI 模型设置的详细摘要，具体使用了多个量化 LLMs（主要是 Unsloth UD Q4_K_XL），例如用于问答的 Mistral-24B、用于通用任务的 Gemma3-27B（IQ3 量化）、用于编程的 Qwen3-30B 以及用于医疗咨询的 Medgemma，所有模型均在具有 48GB 共享 RAM 和 Vulkan 后端的 Llama.cpp 上运行。他们报告了在 10-12k 上下文长度下的实际推理性能（例如，Mistral-24B 为 4t/s，Gemma3-27B 为 5t/s，Qwen3-30B 为 20-22t/s），并描述了这些模型的分工（例如，将 Gemma3n-2B 用作 AI 增强版 Siri，将 Granite3.3-2B 用于摘要）。** 技术评论讨论了使用带有 IT QAT 量化的 Gemma3-27B 以获得显著更好性能的好处，并批评了 Medgemma 在阅读胸部 X 光片方面的可靠性——虽然它适合提供一般建议，但可能会遗漏细微或关键的细节。另一条评论则无关痛痒地列出了历史国家作为最爱。
    - Gemma3-27B IT QAT 相比标准量化模型提供了显著的性能提升，用户报告其质量感觉类似于 Q5KM 量化，特别是如果用户的硬件能够支持的话；这突显了量化感知训练（Quantization-Aware Training）在效率和推理速度方面的进步。
    - MedGEMMA 的特定领域表现（特别是阅读胸部 X 光片）受到质疑，因为它往往会忽略细微但具有临床意义的特征，这强调了 LLMs 在医学影像任务中的挑战，以及在临床使用前进行彻底评估的必要性。然而，它也因作为一种保护隐私、可本地运行的通用健康对话 LLM 而受到赞誉，在某些属性上可以媲美全科医生，提供私密、离线的医疗保健建议。
    - GLM 4.5 Air 的发布导致了用户之间显著的模型整合，一位用户提到模型库从 1.2TB 减少到了 411GB，这表明 GLM 4.5 Air 的能力和全面性可能会取代本地系统对多个较小模型的需求。

### 2. GLM4.5 模型发布、基准测试与用户印象

- [**GLM4.5 EQ-Bench and Creative Write**](https://i.redd.it/ubwsl0gdb0gf1.jpeg) ([Score: 133, Comments: 24](https://www.reddit.com/r/LocalLLaMA/comments/1md5k8f/glm45_eqbench_and_creative_write/)): **该帖子分享了 GLM4.5 在 EQ-Bench 和 Creative Writing 评估中的排行榜图表，重点比较了开源和闭源模型（如 QWEN, DeepSeek, OAI 的 Sonnet, Kimi 2）在创意写作方面的表现。图像显示各种 LLM 的排名，并强调权重开放模型在基准测试中名列前茅。评论中的批评指出，LM-as-judge 基准测试越来越被认为是过时的，并强调当前的评估往往无法解释长文本创意写作，而 OAI 和 Google 模型因其对长上下文窗口的有效处理而被认为表现更稳定。** 评论者争论这些排名的实际适用性，指出基准测试结果（尤其是创意写作）往往误导了实际的用户体验——例如，长文本中的叙事一致性和上下文保留问题。基于第一手使用经验，人们对某些模型的高排名持怀疑态度，观察到一些排名靠前的模型在时态一致性和 Prompt 遵循方面存在困难，而一些知名度较低的模型（如 Mistral Nemo）在实际应用场景中表现出乎意料地好。
    - 几位用户指出，许多当前的创意写作基准测试（如 EQ-Bench）可能已经过时，因为它们依赖于 LM 作为自动评委——这种方法类似于 LMSYS 的 AutoArena——可能无法稳健地反映 LLM 的创意或上下文处理能力。
    - 针对故事创作的基准评估提出了批评：虽然 QWEN, GLM 和 DeepSeek 在简短的、基于角色的任务中表现良好，但用户反映，在达到其完整上下文窗口之前（通常不到宣传长度的 1/10），所有模型都会变得重复或失去叙事结构。相比之下，来自 OAI 和 Google 的模型（例如声称具有 1M 上下文的模型）在扩展上下文方面表现更佳（在高达 100K tokens 时仍能保持连贯）。
    - 针对特定模型的反馈强调，Kimi 2 与 Mistral Small 3.2 等较小模型相比没有显著改进，批评集中在尽管 Prompt 遵循度更好但文笔质量一般。此外，QWQ 被指出在时态一致性方面存在问题，特别是在第一人称叙述中；而经验丰富的用户仍然对 Mistral Nemo 的整体叙事质量评价很高，但指出了其在角色细节上偶尔存在不一致。
- [**glm-4.5-Air appreciation poist - if you have not done so already, give this model a try**](https://www.reddit.com/r/LocalLLaMA/comments/1mdhfhs/glm45air_appreciation_poist_if_you_have_not_done/) ([Score: 127, Comments: 54](https://www.reddit.com/r/LocalLLaMA/comments/1mdhfhs/glm45air_appreciation_poist_if_you_have_not_done/)): **该帖子重点介绍了 glm-4.5-Air（4-bit 量化，mlx 实现），认为它是一款表现出色的 LLM，在多步工具链（tool chaining）和项目管理集成方面表现优异，在复杂工作流中证明了其快速且上下文稳健的特性。用户注意到 glm-4.5-Air 能够提供深入分析而不丢失上下文，在日常助手任务中表现优于 Qwen 32B，特别是在受支持的硬件（尤其是带有 mlx 后端的 Mac）上。** 热门评论集中在 llama.cpp 缺乏支持（尚待 [PR #14939](https://github.com/ggml-org/llama.cpp/pull/14939)），强调了当前的平台限制（特别是对于双 3090 用户），对未来发布 GGUF 兼容版本的乐观态度，并提到了相对于参数量和模型质量，不同量化级别（3-bit vs 4-bit）之间的性能权衡。
    - GLM 架构目前缺乏 llama.cpp 和 GGUF 支持，限制了非 Mac 硬件的使用，特别是那些运行双 3090 GPU 的用户，但活跃的 Pull Request（见 https://github.com/ggml-org/llama.cpp/pull/14939）表明兼容性升级即将到来。
    - 多位用户报告在激进的量化级别（如 3-bit 和 4-bit）下成功运行了 GLM-4.5-Air，表明该模型巨大的参数量补偿了量化带来的质量损失，一些人指出其表现与最近的 Qwen3-30B 量化模型相当或更好。
    - 针对非 Mac 硬件（尤其是使用 DDR4 RAM 的系统，如 i7-13700k 配备 64GB DDR4）的性能存在活跃讨论和担忧，因为潜在的内存速度瓶颈可能会对推理速度产生负面影响（相比于 Mac 的高带宽内存）。

### 3. Meta 超级智能策略与社区反应

- [**再见，Meta AI，曾经的美好已成往事。**](https://www.reddit.com/r/LocalLLaMA/comments/1md6t2h/bye_bye_meta_ai_it_was_good_while_it_lasted/) ([Score: 1098, Comments: 366](https://www.reddit.com/r/LocalLLaMA/comments/1md6t2h/bye_bye_meta_ai_it_was_good_while_it_lasted/)): **Meta 首席执行官 Mark Zuckerberg 发布了一份声明，概述了 Meta 在 AI 超级智能方面的未来方向，指出对安全性的担忧将导致对其最先进模型开源的更严格控制（参见官方声明 [meta.com/superintelligence](https://www.meta.com/superintelligence/)）。这标志着 Meta 从之前的做法（例如在宽松许可下发布 Llama 衍生产品）转向随着能力提升而采用更具专有性的模型。这一决定与其它组织持续发布开源权重（open-weight）、前沿级模型的做法形成对比，表明 Meta 未来可能不会将其顶尖进展贡献给开源社区。** 评论者广泛批评了 Meta 的反转，指出了其此前对开源 AI 的公开倡导，并推测这一转变的动机是货币化而非安全性。一些人指出，已经有开源权重模型可以与 Llama 4 媲美甚至超越，因此 Meta 的退出对开源前沿进展的影响可能有限。
    - 一个评论强调，已经有几个开源权重、前沿级模型被认为比 Llama 4 更强，暗示 Meta 在开源 AI 竞赛中相对落后，并质疑其改变开源模型部署的陈述理由。
    - 另一个评论提到了 Meta 领导层早期的倡导（例如给国会的信），宣传开源 AI 对社会更安全、更好，指出了向货币化和降低透明度的明显转变，这可能会破坏此前对开放性的承诺。
- [**中国模型正在拉开差距**](https://i.redd.it/727keqreo3gf1.png) ([Score: 291, Comments: 35](https://www.reddit.com/r/LocalLLaMA/comments/1mdmsu9/chinese_models_pulling_away/)): **该图片（https://i.redd.it/727keqreo3gf1.png）可能是一个图表或对比指标，表明了中国开源语言模型（如来自 Qwen 的模型）相对于 Llama 和 Mistral 等西方竞争对手的快速进步和性能提升。讨论强调了用户正从 Meta 的 Llama 和 Mistral 模型迁移到更新、审查更少且性能更高的中国模型（特别是 Qwen3-30B-A3B）。评论者注意到了 Mistral 的持续发展，但也承认中国模型在某些基准测试和应用中的主导地位日益增强。** 几位评论者辩论了西方与中国 LLM 的未来采用情况，一些人对 Llama 早期的主导地位表示怀念，并对 Mistral 表示持续支持，同时也承认中国替代方案的技术吸引力和快速进步。
    - 用户正在分享开源 LLM 采用的演进过程：从 LLaMA 3.1-3.2 开始，转向 Mistral 3 Small，然后是蒸馏后的 R1-Mistral 变体（减少了审查，如 Dolphin），现在转向 Qwen3-30B-A3B，这表明像 Qwen 这样的中国模型由于持续的技术改进和模型开放性，正在吸引高级用户。
    - Mistral 模型因其持续强劲的性能而受到关注，重点是本月发布的多个小型模型。对于预计在今年晚些时候发布的新 Mistral 大型模型存在技术期待，这标志着 Mistral 活跃且持续的发展。
    - 对于实际的编程用例，特别是前端编程，注意到 Mistral 模型表现出色，外部资源证明了它们的能力（例如 [designarena.ai](http://designarena.ai/)）。这表明尽管面临来自 Qwen 等新进入者的竞争，Mistral 在应用编程任务中仍保持着竞争力。

## 较少技术性的 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. OpenAI GPT-5 的期待与证据

- [**更多 GPT 5 的片段，发布似乎真的迫在眉睫。**](https://i.redd.it/fbo023zrzzff1.jpeg) ([Score: 390, Comments: 69](https://www.reddit.com/r/singularity/comments/1md46vv/more_snippets_of_gpt_5_seems_like_release_really/)): **该帖子讨论了据称来自 GPT-5 的新片段，并暗示其公开发布可能迫在眉睫，理由是目前的泄露中缺少 'gpt-5-mini' 模型，这与 OpenAI 之前的发布模式形成了对比。该图片可能是一张截图或信息片段，进一步支持了关于发布时间和模型阵容细分的猜测。** 技术讨论集中在推广策略上：一些用户推测 OpenAI 将首先发布最大、最引人注目的模型，随后是较小的 'mini' 模型，以后可能还会发布开源版本，这与之前的行为有所不同。
    - 一位用户注意到尽管之前的泄露暗示了其存在，但目前仍缺少 'gpt-5-mini' 模型，并假设 OpenAI 可能会优先发布高知名度、功能强大的模型，以在随后推出较小或开源变体之前最大限度地占据头条新闻。
- [**来自 ChatGPT Twitter 账号的神秘帖子……明天发布 GPT-5？**](https://i.redd.it/d8z6h7u743gf1.jpeg) ([Score: 192, Comments: 77](https://www.reddit.com/r/singularity/comments/1mdjzgd/cryptic_post_from_the_chatgpt_twitter_account/)): **ChatGPT Twitter 账号发布的这张图片显示了用日语片假名（katakana）书写的 "ChatGPT"，并带有一个红色的汉字（kanji）符号，意为“福”、“好运”或“祝福”。这引发了社区对潜在公告（如 GPT-5）的猜测，特别是因为帖子提到明天是周四——通常是技术发布的日期。图片的模型技术内容有限，主要作为预告，没有明确的技术细节或公告。** 一条评论澄清了日语文本，纠正了最初对隐藏暗示的猜测，并指出了片假名和汉字的使用，将讨论建立在语言学而非技术新颖性上。一些人猜测会发布 'GPT-5 甚至 GPT-6'，但这纯属猜测，并非基于图片本身的任何证据。
    - 一位评论者指出，视觉效果中与公告相关的日语水墨画（sumi-e）笔触可能象征性地暗示了即将推出的模型具有 *"成熟度、深度和审慎性"*，可能暗指与之前的版本相比，在推理或解释细微差别方面有所改进。
- [**在最近的更新中，"gpt-5-auto" 和 "gpt-5-reasoning" 已作为模型添加到 ChatGPT MacOS 应用中**](https://i.redd.it/xk0egrlaxzff1.png) ([Score: 289, Comments: 39](https://www.reddit.com/r/OpenAI/comments/1md3xoo/gpt5auto_and_gpt5reasoning_have_been_added_as/)): **截图显示了 ChatGPT MacOS 应用的模型选择菜单，其中包含两个新模型：'gpt-5-auto' 和 'gpt-5-reasoning'。这表明 OpenAI 正准备推出或测试这些具有不同操作重点的 GPT-5 变体——'auto' 可能用于通用用途，而 'reasoning' 可能针对更复杂的逻辑任务进行了优化。没有指定基准测试或官方发布说明；视觉证据仅确认了它们在 UI 中的存在。** 一个关键的用户观点强调了手动选择模型的重要性，对在不同模式（如 router/auto 和明确的单一模型选择）之间进行选择的能力表示赞赏。目前没有深层的技术辩论，因为大多数评论都是兴奋的期待而非批判性分析。
    - 一位评论者将新的 "gpt-5-auto"/"gpt-5-reasoning" 模型区分与 xAI 的方法进行了类比，在 xAI 中，用户可以在 "fast/auto" 和 "advanced" 模型选项之间进行选择，并指出单独的 "reasoning" 模型持续存在表明 "auto" 并不是一个完全统一、具备所有能力的模型。这暗示了 GPT-5 阵容内部可能存在专业化分工（例如，"auto" 用于速度/路由，"reasoning" 用于复杂任务），而不是像其他 AI 提供商所见的策略那样，用单一的庞大模型处理所有用途。
- [**发现 GPT 5！！它近了！**](https://i.redd.it/m7o4ma8zk0gf1.jpeg) ([Score: 150, Comments: 17](https://www.reddit.com/r/OpenAI/comments/1md6qsd/gpt_5_spotted_its_near/)): **这篇标题为“发现 GPT 5！！它近了！”的帖子可能引用了关于 GPT-5 潜在发布的传闻或推测性目击，但图片内容无法进行技术细节分析。讨论集中在对 GPT-5 发布日期的期待（即将发布或在 8 月晚些时候），并希望它能与一个新的推理模型一同推出，这反映了人们对 GPT-5 单独是否能超越 OpenAI 的 'o3' 持怀疑态度。** 评论者辩论了可能的发布时间表，并对 GPT-5 对就业市场的影响表达了渴望与焦虑，呼应了行业内对先进语言模型扰乱就业的持续担忧。

- 用户对 GPT-5 发布时是否能明显超越 OpenAI 自家的 "o3" 模型表示怀疑，强调了目前顶尖模型与下一代发布版本之间性能差异的不确定性。人们也对传闻中的“推理模型”充满期待，并推测将其与 GPT-5 一同发布可能会显著提升超越传统 LLM 的能力。
- [**有些事情正在发生**](https://i.redd.it/qilyyd52d0gf1.jpeg) ([得分: 147, 评论: 63](https://www.reddit.com/r/singularity/comments/1md5q6z/somethings_going_on/)): **该图片似乎展示了一张选择了 'gpt-8-auto' 模型的 ChatGPT 截图，暗示存在一个未来未发布的模型变体（可能是 GPT-5 或更高版本）。评论者指出，在 URL 中输入任意模型名称也会产生类似的 UI 结果，这意味着这并非真实模型，而是一个通过直接操作 URL（例如 https://chatgpt.com/?model=gpt-8-auto）即可访问的 UI 伪像或占位符。由于近期泄密事件和社区的高度期待，关于 GPT-5 即将发布或宣布的推测愈演愈烈。** 一些用户对截图的真实性展开辩论，有人将其归因于 URL 操作而非真实的模型泄露，也有人根据行业传闻和近期动态推测潜在的发布时间表。
    - 一些用户注意到，在 ChatGPT 的 Web 界面中操作模型查询字符串（例如 https://chatgpt.com/?model=gpt-8-auto 或 ?model=gpt-5-auto）仅仅是改变了显示的模型名称，并不会加载任何未发布的模型；实际请求仍被路由到已有的模型，如 GPT-3.5。从无论查询中的模型名称如何功能均未改变，以及观察到后端模型没有差异来看，这一点显而易见。
    - 基于近期的泄密和行业传闻，有技术推测认为新的 GPT 模型（可能是 GPT-5）可能即将发布，理由是泄密中提到了 'LMarena Anon' 等模型名称以及之前模型推出的时间点。然而，目前尚未检测到官方发布确认或基础设施变化（如直播或推出公告），这增加了对这些说法的怀疑。
    - 直接检查和截图分享显示，尽管进行了查询字符串操作，后端强制执行机制依然完好，防止了通过 UI 未经授权访问未发布的 GPT 模型，证实了 OpenAI 模型端点预期的安全性。

### 2. WAN 2.2 动画模型发布与社区工具

- [**WAN 2.2 将改变独立动画的一切**](https://v.redd.it/trtpftp1hzff1) ([得分: 415, 评论: 94](https://www.reddit.com/r/StableDiffusion/comments/1md2d20/wan_22_is_going_to_change_everything_for_indie/)): **WAN 2.2 作为独立动画模型的更新版本，因其相较于 WAN 2.1 的视觉改进而受到关注，但用户发现其在 Prompt 遵循方面存在局限性，特别是在动作和摄像机运动控制方面（例如，在动画化摄像机移动时难以生成静止的主体）。WAN 2.2 在生产环境中的可用性存在争议，因为由于未解决的问题，其输出可能会给开发者的声誉带来风险。** 评论强调了一个共识：虽然 WAN 2.2 在美学上有所进步，但其在 Prompt 解析方面的持续问题和缺乏可靠性使其不适合专业或开发者使用，引发了对社区负面反响的担忧。
    - 技术用户正在讨论 WAN 2.2 相比 2.1 的实际改进，将其定性为适度的视觉升级而非突破性变革，挑战了“改变一切”的说法。
    - 一个实际的 Prompt Engineering 问题被凸显出来：即使使用了明确的负面提示词（"walking" 和 "talking"），WAN 2.2 中的角色仍会持续执行不需要的动作，这表明该模型在动画任务中的 Prompt 忠实度和模型控制力方面存在局限。
    - 一些人指出，尽管视觉效果有所提升，但由于持续存在的问题（输出一致性、控制机制），该模型对于开发者的生产用途来说在很大程度上仍不可用，在项目中部署可能会损害声誉，而非提供实质性收益。

- [**我为 WAN 2.2 创建了一个详细的 Prompt Builder，完全免费使用。**](https://i.redd.it/b9mfy32dxyff1.png) ([评分: 364, 评论: 30](https://www.reddit.com/r/StableDiffusion/comments/1md0qed/i_created_a_detailed_prompt_builder_for_wan_22/)): **该图片展示了一个专为 WAN 2.2（可能是 Stable Diffusion 模型的一个变体）量身定制的新型免费浏览器端 Prompt Builder 的 UI，访问地址为 dengeai.com/prompt-generator。该界面直观地展示了 Prompt 组件，支持详细的视频 Prompt 创建，并旨在通过直观的视觉提示增强可用性，正如评论者所指出的那样。技术用户询问了本地部署的可能性，表明对开源或可下载版本的需求，这可能会扩展到当前特定模型之外。** 一场富有成效的辩论集中在将该工具作为本地或通用应用程序提供的益处和可行性上，社区有兴趣进行协作开发，以将其效用扩展到 WAN 模型之外。
    - 一位评论者询问了将 Prompt Builder 作为本地独立软件解决方案发布的可能性，强调了社区对拥有不限于 WAN 2.2 的通用工具的兴趣。他们建议进行协作开发，以将该工具的适用性推广到其他模型或工作流。
- [**使用 SeerV2 的 Wan 2.2 I2V 游戏角色**](https://v.redd.it/kape9d80xzff1) ([评分: 288, 评论: 55](https://www.reddit.com/r/StableDiffusion/comments/1md3zfe/wan_22_i2v_game_characters_with_seerv2/)): **该帖子详细介绍了一个使用 Wan 2.2 和 SeerV2 的 Image-to-Video (I2V) 工作流，强调了 Wan 2.2 在游戏角色渲染方面的有效性。该工作流在 ComfyUI 中实现，使用强度为 0.5 的 LightXv2，20 步，CTF 3.5，ModelSamplingSD3 设置为 5，采样器设置为 dpmpp_2m，调度器为 Beta。最终输出渲染分辨率为 832x512，并使用 SeedVR 放大至 4K（参考指南：[one-step-4k-video-upscaling-and-beyond-for-free-in-comfyui-with-seedvr2](https://www.ainvfx.com/blog/one-step-4k-video-upscaling-and-beyond-for-free-in-comfyui-with-seedvr2/)）。** 评论者强调了 Wan 2.2 令人印象深刻的质量，其中一位指出它是“迄今为止最好的一个”，但讨论中没有出现重大的技术辩论。
    - 一位用户详细介绍了他们使用 ComfyUI 生成 SeerV2 I2V 游戏角色的工作流，重点介绍了 Wan 2.2 的使用，以及强度为 0.5 的 LightXv2，20 步，CTF 3.5，ModelSamplingSD3 设置为 5，以及带有 Beta 调度器的 dpmpp_2m 采样器。他们以 832x512 分辨率渲染，并使用 SeedVR 进行放大，参考了 ComfyUI 中 4K 放大的指南 (https://www.ainvfx.com/blog/one-step-4k-video-upscaling-and-beyond-for-free-in-comfyui-with-seedvr2/)。该工作流涉及特定的负面提示词（"Slow, Slowed in negative"）以调整输出角色的动作。
- [**对 Wan2.2 Text-To-Image 质量感到惊喜（评论中有工作流）**](https://www.reddit.com/gallery/1md4u30) ([评分: 224, 评论: 94](https://www.reddit.com/r/StableDiffusion/comments/1md4u30/pleasantly_surprised_with_wan22_texttoimage/)): **该帖子讨论了 Wan2.2 Text-To-Image 模型（14B 参数，FP16）的结果，并附带了工作流链接和示例输出。用户强调了强大的纹理和皮肤真实感，指出 Prompt 遵循能力中等，但与 Flux 结合使用时会有所改善（例如，通过使用 Nunchaku 生成 latent，然后在 WAN2.2 中重新处理，这可将生成时间缩短约 40%）。社区正在寻求改进的控制工具（例如 tile/blur controlnet），以便进行更灵活的输出操作。** 大家一致认为 WAN2.2 在写实度和纹理质量上超过了 Flux，尽管 Flux 被认为在精细 Prompt 控制方面更胜一筹；两者的实际融合产生了引人注目的结果并提高了效率。
    - 用户报告称 Wan2.2 表现出极强的真实感，尤其是在生成皮肤等详细纹理方面，在这方面优于 Flux Dev 等模型。一位用户指出使用 14B FP16 版本可获得最佳效果，并建议查看未压缩的样本以欣赏纹理保真度，并建议添加 tile/blur controlnet 支持将进一步增强模型的通用性。
    - 描述了一种工作流优化：用户使用 Flux Dev 替换 High Noise 通道（通过 Nunchaku 获取半点 latent），然后重新编码到 ksampler 中进行“WAN 修饰”。据报道，这种混合流水线在保持 WAN 2.2 输出质量的同时，将图像生成时间缩短了约 40%。
    - 确认了对来自 WAN 2.1 的 LoRA 的向后兼容性，使用户能够重用其现有资产。此外，还提到了未来视频渲染能力的潜力，表明功能正在不断扩展。

- [**全合一 WAN 2.2 模型合并：4 步，1 CFG，单模型极速（支持 T2V 和 I2V）**](https://huggingface.co/Phr00t/WAN2.2-14B-Rapid-AllInOne) ([Score: 199, Comments: 108](https://www.reddit.com/r/StableDiffusion/comments/1mddzji/all_in_one_wan_22_model_merges_4steps_1_cfg_1/)): **该帖子详细介绍了一个用于文本生成视频 (T2V) 和图像生成视频 (I2V) 的定制化 WAN 2.2 模型合并版本。它将 WAN 2.2 的 'high' 和 'low' 模型（作为前/中块）与 WAN 2.1 的输出块整合进一个单一、精简的架构中。该模型集成了 VAE 和 CLIP 以简化部署，合并了 Lightx2v 和 PUSA LoRA 进行蒸馏，并支持 4 步、1 Classifier-Free Guidance (CFG) 采样，同时保持对 WAN 2.1 LoRA 的兼容性。该方案建议使用带有 beta scheduler 的 sa_solver，并支持原生 Checkpoint 加载，强调在速度和简洁性之间取得平衡，尽管这可能需要牺牲运行更大、独立模型所能达到的效果。** 一位高赞评论者要求提供与基准模型的定量对比，并询问是否有人会对该合并模型进行量化 (quant)，这表明了用户对经验基准测试以及进一步提升速度/效率的兴趣。其他建议还涉及标题命名的改进，但不涉及技术内容。
    - 用户对合并后的 WAN 2.2 模型的量化版本表现出浓厚兴趣，以追求更高的效率。有人呼吁在相同种子下进行基准测试，对比原始双模型/加速流水线与合并模型之间的速度和输出质量。
    - 针对合并步骤和模型可能导致的质量下降，用户提出了技术担忧：一位用户特别询问与运行完整的 20 步和独立模型相比，质量是否有所下降，表现出对性能权衡的关注。
    - 一位用户指出，在 12GB GPU 上，该全合一模型表现强劲，尤其是在 T2V 任务中，但评论称输出的多样性似乎有所降低——输出结果变化较少，更趋向于 WAN 2.1 的风格，而非展示 WAN 2.2 更广泛的生成范围。该用户表示，尽管运行时间更长，但更倾向于以 1.0 的权重运行带有 lightx2v 的原始模型以获得更好的多样性。

- [**Wan 2.2 I2V 表现惊人！到目前为止**](https://v.redd.it/z0r6axrpszff1) ([Score: 191, Comments: 34](https://www.reddit.com/r/StableDiffusion/comments/1md3ggp/wan_22_i2v_is_really_amazing_so_far/)): **该帖子讨论了 Wan 2.2 I2V（图像生成视频）模型的性能，强调了其将图像动画化为视频的卓越能力，具有显著的动态效果和细节，特别是通过 kijai wrapper 工作流使用完整的 Wan 2.2 F16 32GB 模型时。对比表明，全尺寸模型的输出比缩减版变体具有更丰富的细节和动态效果，强调了使用全量模型进行高保真视频合成的重要性。** 评论者注意到用户社区对这种先进的生成式视频技术缺乏足够的关注，并推测了一些引人注目的用例，例如将这些模型集成到独立游戏的过场动画中，以减少制作时间和资源需求。
    - 一位用户对比了使用 Kijai wrapper 工作流的 Wan 2.2 输出与完整的 WAN 2.2 F16 32GB 模型，指出使用全尺寸模型生成的视频比缩减版具有 *更多的动态和细节*。这表明在模型大小和输出细节保真度之间存在技术权衡。
    - 性能和硬件是关注焦点，讨论围绕以全分辨率运行 WAN 2.2 的 *生成时间和硬件要求* 展开。讨论表明，硬件规格和模型大小（F16 为 32GB）严重影响了工作流的可行性和结果质量。

- [**Wan 2.2 I2V 连续运动尝试**](https://v.redd.it/o978gmrhizff1) ([Score: 130, Comments: 44](https://www.reddit.com/r/StableDiffusion/comments/1md2jzi/wan_22_i2v_continous_motion_try/)): **楼主尝试使用 WAN 2.2 进行图像生成视频 (I2V)，通过从静态图像（WAN T2I）迭代外推视频，然后使用每一段的最后一帧作为种子来链接后续片段。技术讨论集中在如何处理退化问题（如运动模糊、光照、特征不一致）以及潜在的缓解措施。工作流为 720p@16fps，随后放大/插帧至 1080p@60fps；确定的关键问题是由于运动模糊阻碍了连续性，导致帧选择困难。** 评论者注意到 WAN 2.2 模型在链式迭代中似乎能保持质量，视觉退化极小，这可能表明该模型具有较强的鲁棒性。关于在将最后一帧反馈回去之前是否应用图生图 (I2I) 细节增强存在技术争论，但 WAN 2.2 的弹性可能使这一步骤变得多余。

- 一位评论者观察到，与迭代式帧扩展工作流中常见的视频质量退化不同，Wan 2.2 通过连续帧生成保持了高质量——起始帧和最终帧几乎没有明显的退化，这表明该模型版本具有强大的视频帧一致性。
- 有技术推测认为，在循环使用最后一帧进行进一步生成之前，可以使用中间的 image-to-image (I2I) 过程来增强细节。然而，鉴于 Wan 2.2 的输出稳定性，这一额外步骤现在可能是不必要的，从而有可能在减少人工质量干预的情况下简化连续视频合成。
- 一位用户指出，尽管存在一些缝合或过渡故障，但当专注于故事内容时，这些故障微小到可以忽略不计，这表明 Wan 2.2 帧链接产生的视觉伪影对于沉浸式应用来说处于可接受的水平。
- [**它绝对已经达到了乍看之下会误以为是真的程度。**](https://v.redd.it/ujbvunncx2gf1) ([评分: 607, 评论: 64](https://www.reddit.com/r/ChatGPT/comments/1mdj19u/its_definitely_at_the_point_where_you_can_mistake/))：**该帖子讨论了 AI 生成视频的最新进展，强调使用极简提示词（“每个仅寥寥几句”）创建的视频片段现在在视觉上已经足够真实，乍看之下会被误认为是真实镜头。内容强调了生成模型真实性的重大进步，可能参考了 Sora 或类似的 text-to-video 合成模型的改进。** 热门评论反映了对 AI 输出的真实性和娱乐价值的正面评价，但没有提供技术讨论或批评。
- AndrewH73333 强调了当前 AI 模型视频生成能力的一个局限性：由于在这些领域的训练数据不足，AI 难以针对记录较少的事件（例如“钢琴被淹没在水下后会发生什么”）生成合理的输出。这指向了生成模型面临的一个更广泛的挑战——为训练数据集中缺乏或没有示例的场景合成真实的输出，这表明需要全面、多样的数据来提高边缘情况或罕见情况下的保真度。
- eOMG 注意到当前 AI 生成语音的一个特征：虽然它非常先进且乍看之下通常很真实，但细心的听众仍然可以识别出细微的伪影或不自然感，从而将其与真实的人类语音区分开来。这一观察结果表明 text-to-speech 或 multimodal 生成模型中仍然存在微小的缺陷，这可能归因于韵律、语调或上下文意识方面的局限性。

### 3. Anthropic Claude 功能扩展与社区使用

- [**Claude 现在可以在移动端为你发送短信、电子邮件和安排日程**](https://v.redd.it/ash0cvh572gf1) ([评分: 214, 评论: 45](https://www.reddit.com/r/ClaudeAI/comments/1mdf9ns/claude_can_now_text_email_and_schedule_for_you_on/))：**Anthropic 的 Claude AI 移动应用现在支持起草电子邮件、短信和会议邀请，用户只需点击一下即可通过原生应用发送 ([公告](http://claude.ai/download))。技术实现上，特别是针对 Android，可能利用了标准的 Android OS intent system，通过预填主题和正文内容来撰写草稿，出于隐私考虑不指定收件人。这种工作流最大限度地减少了应用权限和用户数据的暴露，符合注重隐私的设计。** 热门评论对高级计划（“Max x5 计划”）的 rate limiting 表示担忧，并表示更倾向于这种避免直接访问通讯录的间接方法。用户指出，这消除了使用 Signal 等通信平台进行 DIY 自动化的需求。
- 一位评论者推测，Claude 的 Android 实现可能使用原生 Android OS intent system 来启动电子邮件草稿，仅传递主题和正文，而将收件人字段留给用户，这通过不授予通讯录访问权限来优先保护用户隐私。
- 讨论的一个技术限制是 Claude 的对话长度限制，这可能会中断工作流；用户请求增加诸如预警、上下文摘要或滚动对话等功能（类似于 GPT 提供的功能），以便更有效地管理较长的线程和上下文保留。
- 来自技术用户的对比反馈指出，20 美元档位的 GPT 桌面体验优于价格更高（20倍）的 Claude 桌面计划，理由是长对话的可用性和集成度更好，并要求 Claude Code 获得原生的 Gmail 和 Calendar 集成，而无需额外的权限门户或中间件。

- [**Claudeholism: The New Addiction of the Century 😞**](https://www.reddit.com/r/ClaudeAI/comments/1mdc09s/claudeholism_the_new_addiction_of_the_century/) ([Score: 132, Comments: 58](https://www.reddit.com/r/ClaudeAI/comments/1mdc09s/claudeholism_the_new_addiction_of_the_century/)): **这篇帖子报道了用户对 Claude Code（由 Anthropic 开发）的强烈依赖，理由是其生产力和代码生成能力优于 GitHub Copilot、Google Gemini、JetBrains AI Assistant 和本地 StarCoder 2 模型等替代方案。用户提到 Claude Code 在生成可用代码（例如 Pulumi 脚本、Clean Architecture）和流畅的上下文感知代码注释方面非常有效，而其他替代方案则存在幻觉、代码理解能力差、领域对齐不足或硬件要求高等问题。** 评论者一致认为 Claude Code 显著提高了生产力（快速原型设计、迁移和工具开发）并带来了令人上瘾的工作流；他们指出其上下文集成（包括访问本地文件）使得竞争对手的 LLM 显得不那么有用，即使考虑到偶尔出现的幻觉。一些人注意到这种行为转变，即无摩擦的 LLM 辅助开发改变了使用模式和预期。
    - 一位用户详细描述了 Claude 实现的快速原型设计，在几小时内将一个论坛从糟糕的平台迁移到自定义的 Nuxt + Supabase 架构，并快速为咨询业务构建了专业工具，强调了与传统工作流相比的效率提升和门槛降低。
    - 另一位用户强调了将 LLM 与本地计算机访问集成如何通过减少摩擦极大地简化了开发，尽管偶尔会有幻觉。这种流畅性和生产力优势被认为超过了目前可用的其他解决方案。
    - 关于 Opus 订阅层级的讨论显示，从 200 美元降级到 100 美元方案（Pro）会带来显著的使用限制，使得更高层级对于重度、创造性的 Opus 使用变得必要。用户注意到来自“vibe coders”的需求激增可能会进一步限制访问，导致在较低层级只能满足约 70% 的工作流需求。
- [**What y'll are building that is maxing out Claude Code**](https://www.reddit.com/r/ClaudeAI/comments/1md0etv/what_yll_are_building_that_is_maxing_out_claude/) ([Score: 105, Comments: 110](https://www.reddit.com/r/ClaudeAI/comments/1md0etv/what_yll_are_building_that_is_maxing_out_claude/)): **发帖人（OP）是一位拥有深厚后端、全栈和 ML 经验的高级工程师，他质疑什么样的项目会耗尽 Claude Code（Sonnet 和 Opus 模型）的整个上下文窗口，因为他有构建复杂、可维护系统的背景。他们指出，从历史上看，代码迭代是一个团队过程，而平台复杂性（Supabase、PlanetScale 等）与“自研”系统相比只是额外的一层。他们寻求在单个受限编码会话中真正挑战 Claude Code 极限的工作流或代码库的具体示例。** 评论者指出，重度使用通常是为了最大化上下文窗口，而不是因为项目规模——通常涉及对模型进行详尽的引导（每个 Prompt 都包含完整文件上下文、相关类、文档和 Lint 规范）。这导致 Token 迅速耗尽，一位用户报告由于这些做法，在 5-6 个 Prompt 内就用完了 Opus 的额度。讨论还表明，这种使用中的很大一部分可能并不代表高效工作，暗示一些用户正在生成“垃圾”代码或滥用可用配额。
    - Claude Code 的主要技术瓶颈似乎是其上下文窗口：与熟悉项目的开发者不同，每个新的 Claude 会话就像是入职一名新开发人员，每次运行都需要提供完整的相关文件、类、抽象层、实现示例和文档。这种引导开销，结合严格的编码标准和生成后的 Lint/修复，会迅速消耗 Token——用户报告在几小时内仅通过 5-6 个 Prompt 就用完了 100 美元的 Opus 方案，有时甚至在单个 Prompt 后就会收到 Token 使用警告。
    - 一些用户将会话视为有意义开发的批量引导，加载大型代码库和参考资料，以获得精确、符合规范的代码和自动 Lint。这种工作流虽然消耗 Token，但对于深度 Bug 修复或大规模重构是必要的，并突显了当前上下文/Token 架构在处理复杂项目时的局限性。

- [**Claude 现已入驻 X (@claudeai)**](https://i.redd.it/5q4o2j4x23gf1.png) ([评分: 210, 评论: 51](https://www.reddit.com/r/ClaudeAI/comments/1mdjt9r/claude_is_now_on_x_claudeai/)): **该帖子宣布 Anthropic 的 AI 模型 Claude 现已在社交媒体平台 X（原 Twitter）上开设官方账号，账号名为 @claudeai。附带的图片未描述技术细节，但推测展示了该账号或相关公告。评论主要集中在确认账号的真实性并进行了一些轻松的观察，没有增加技术层面的辩论。** 用户对账号的真实性感到好奇，有人询问“这是真的吗？”。文中没有关于模型实现、功能或与 X 集成的技术讨论。
    - 一位用户详细描述了 Claude 回复中一种无法消除的持久行为，即使在项目和全局范围内使用 [Claude.md](http://claude.md/) 等自定义功能也是如此。该用户强调了一个反复出现的短语——“你说得完全正确！”（"You're absolutely right!"）——尽管尝试配置模型的输出，该短语依然存在，这表明存在深层嵌入的、硬编码的回复结构，或者用户层面的可配置性不足。这突显了 Anthropic 模型在当前 Prompt 自定义或扩展性方面的潜在局限性，因为模型的核心行为覆盖了持续的用户偏好。
- [**Anthropic vs xAI**](https://i.redd.it/vj8g05pmhyff1.png) ([评分: 792, 评论: 47](https://www.reddit.com/r/ClaudeAI/comments/1mczbbb/anthropic_vs_xai/)): **该帖子是两家知名 AI 公司 Anthropic 和 xAI 之间的对比，含蓄地提到了它们的商业策略、声誉差异以及与重大合同（如军事合同）的关联。评论者强调了 Anthropic 对争取军事合同的关注，而 xAI 的起源和意图则引发了辩论，特别是在创始人动机和利润激励的背景下。图片本身作为两个实体之间的视觉对比供社区讨论，并未直接提供技术基准或模型细节。** 评论者对 xAI 的动机表示怀疑，引用了 Elon Musk 之前的言论，并暗示其动机是为了利润或追赶；其他人则对比了这些公司的伦理品牌塑造和财务重点。
    - 讨论简要提到了追求军事合同是塑造 AI 模型格局的关键战略重点，暗示这是大规模 AI 发展中一个重要但常被忽视的因素（例如，Anthropic 可能获得了此类合同）。
    - 讨论中隐含了对主要 AI 参与者（Anthropic、xAI 和 OpenAI）商业模式和利润策略的比较，暗示来自合同（包括军事和企业合同）的收入对于它们的长期可持续性和竞争定位至关重要。
    - 参与者辩论了品牌命名、模型功能和领导者公开声明在声誉和伦理方面的影响，突显了人们对领先 AI 实验室宣称的安全伦理与实际产品方向之间一致性的持续担忧（例如，对 Elon Musk 在 xAI 和 OpenAI 中动机的怀疑）。

---

# AI Discord 回顾

> 由 X.ai Grok-3-mini 生成的摘要之摘要的摘要
> 

**主题 1. 新模型发布与对比**

- [**Qwen3 与 GPT-4o 展开 AI 霸权之战**](https://www.notion.so/swyx/various_discords)：Qwen 发布了 **Qwen3-30B** 模型，声称其在英文和中文基准测试中可与 **GPT-4o** 媲美。通过 [Unsloth GGUF 版本](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF)，该模型仅需 **33GB RAM** 即可全精度运行。用户报告称，量化版本仅需 **17GB RAM**，这引发了关于其在真实场景下对比 **OpenAI** **GPT-4o** 性能的讨论。该模型在多语言任务中的潜力让工程师们感到兴奋，基准测试显示它在特定测试中达到或超过了 **GPT-4o**。
- [**GPT-5 传闻引发 AI 炒作大战**](https://www.notion.so/swyx/openai_discord)：工程师们推测 **GPT-5** 可能会在 **8 月**初发布，[X](https://xcancel.com/apples_jimmy/status/1950514936444305534?s=46&t=fRVjULzONZQAlwHruKTgQg) 上的 ChatGPT Mac 应用缓存文件中发现了 "gpt-5-auto" 等线索。讨论将其与 **Gemini 2.5 Pro** 等竞争对手进行了比较，并指出了其在 **Microsoft Copilot** 中潜在的集成以及对模型排名的影响。用户根据泄露的架构细节，争论 **GPT-5** 是否会在推理任务中占据主导地位。
- [**GLM 4.5 Air 模仿 Gemini 的特性**](https://www.notion.so/swyx/unsloth_ai_discord)：工程师们测试了 **GLM 4.5 Air**，发现它在聊天和诗歌分析方面的行为模仿了 **Gemini**，详见[博客文章](https://z.ai/blog/glm-4.5)。工具使用（Tool use）在 **vllm** 中失效，但在文档搜索中表现良好，引发了与 **GPT-4o** 的对比。该模型突显了**阿里巴巴**推动通用替代方案的努力，用户注意到尽管存在一些小瑕疵，但其稳定性良好。

**Theme 2. API 与集成障碍**

- [**API 屏蔽劣质量化乱象**](https://www.notion.so/swyx/openrouter_discord)：根据[提供商路由文档](https://openrouter.ai/docs/features/provider-routing#quantization-levels)，OpenRouter 的 API 允许工程师指定量化级别，以避开像 **FP4** 这样的低精度模型。这阻止了不想要的格式，确保了跨平台的有效模型路由。用户称赞这种控制减少了生产环境中的错误。
- [**MCP 服务器在上下文处理上遭遇危机**](https://www.notion.so/swyx/mcp_discord)：工程师们讨论了在云端部署的 **MCP 服务器**中添加用户上下文隔离层的必要性，以防止数据泄露，参考了 [issue #1087](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1087)。一种配置在 **Cursor** 上可以连接，但在 **Claude Desktop** 上失败，凸显了集成缺陷。这强调了在多客户端环境中需要健壮协议的必要性。
- [**Litellm API 规避连接混乱**](https://www.notion.so/swyx/aider_discord)：用户报告了 **litellm.APIConnectionError** 解析问题，但注意到功能保持完好，这通常与不当的分块（chunk）格式有关。工程师们分享了涉及在 API 调用中使用双引号属性名的修复方案。这一错误揭示了 AI API 链中的常见陷阱，推动了更好的错误处理标准。

**Theme 3. 性能优化策略**

- [**量化加速计算速度**](https://www.notion.so/swyx/unsloth_ai_discord)：Unsloth 的量化不仅缩小了内存需求，还减少了带宽并提升了计算速度，这在 **Qwen3-30B** 的测试中得到了证实。工程师们注意到，将卷积层（conv layers）等视觉操作保留在 **FP32** 会造成瓶颈，建议使用混合精度以达到平衡。这种方法在不牺牲准确性的情况下大幅缩短了推理时间。
- [**Gemma 3 微调后去除水印**](https://www.notion.so/swyx/unsloth_ai_discord)：在对具有 **16k 上下文**的 **Gemma 3 4B** 进行微调后，实验完全去除了水印，提高了模型稳定性，如[截图](https://cdn.discordapp.com/attachments/1179039861576056922/1399849804045221948/Screenshot_2025-07-29_at_2.13.07_PM.jpeg?ex=688bd0b9&is=688a7f39&hm=82cad388163625496b8cb3e6dd62035b51ae1d16ce0015df29ea910e04c4471f&)所示。工程师们讨论了权衡利弊，[X](https://x.com/UnslothAI/status/1950553466591723640) 上宣布的一项竞赛将在 **7 天**内结束。微调增强了其在技术任务中的实用性，使其成为无审查应用的首选。
- [**专家并行在 Qwen 的测试中表现不及预期**](https://www.notion.so/swyx/gpu_mode_discord)：工程师们发现，由于 all-reduce 开销，**专家并行 (EP)** 在 **Qwen32B** 和 **Qwen 235B** 中的表现不如 **张量并行 (TP)**，详见[博客文章](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/?utm_source=chatgpt.com#ttir-triton-ir)。EP 仅在具有 **MLA 和 DP 注意力**的模型中表现出色。这一见解为高效扩展的硬件选择提供了指导。

**Theme 4. 数据隐私与安全辩论**

- [**Dot.lol 出售用户数据风波**](https://www.notion.so/swyx/lmarena_discord)：工程师们警告说 **dot.lol** 可能会出售用户数据用于画像和影响，正如社区讨论中所述，敦促进行身份保护。一些人认为数据收集是*不可避免的*，但另一些人则推动实施更严格的措施，如欧盟的 **GDPR**。这凸显了 AI 平台中的风险，并引发了对匿名使用的呼吁。
- [**OpenAI 的审查收紧令用户感到沮丧**](https://www.notion.so/swyx/unsloth_ai_discord)：根据 [Unsloth 的 Llama 4 页面](https://docs.unsloth.ai/basics/llama-4-how-to-run-and-fine-tune)，成员们猛烈抨击了 **ChatGPT** 中 **OpenAI 的严厉审查**，分享了*模板化回答*和说教的体验。工程师们辩论了去审查技术，并指出效用降低等缺点。这场审查辩论揭示了安全 AI 部署中的权衡。

**主题 5. 社区教育与活动**

- [**扩散模型学习小组启动 MIT 深度研究**](https://www.notion.so/swyx/mlops_chipro_discord)：AI 学者们启动了一个为期 **5 个月、12 人**的扩散模型学习小组，使用 [MIT 课程](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf)，并于 **8 月 2 日**和 **8 月 9 日**通过 [Luma](https://lu.ma/kv8zf6va) 举办免费课程。成员包括电影工具 CTO 等 AI 专业人士，强调实战项目。该倡议通过同行引导的学习，为工程师配备生成式 AI 技能。
- [**DSPy 参数提案引发优化热议**](https://www.notion.so/swyx/dspy_discord)：工程师们提议向 **DSPy** 添加可学习参数，并创建了 [一个 GitHub issue](https://github.com/stanfordnlp/dspy/issues/8593) 来集思广益 Prompt 优化。该想法允许模板作为变量以获得更好的结果，并在 YouTube 上引发了与 **GEPA** 的比较。这增强了 DSPy 在微调方面的实用性，促进了协作改进。

---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **R1-der R.I.P：模型从 LLM 选择器中移除**：**R1 1776 模型**已从 LLM 选择器中移除，但仍可通过 [OpenRouter API](https://openrouter.ai/perplexity/r1-1776) 访问。
   - 在移除后，用户正考虑转向 **O3** 或 **Sonnet 4** 进行推理任务。
- **Android Comet 应用即将发布**：**Comet for Android** 应用正在开发中，计划于年底发布。
   - 虽然一位用户质疑该浏览器的潜在功能，但其他人称赞了其在 Windows 上的表现。
- **Gemini 2.5 Pro 获得速度提升**：用户报告 **Gemini 2.5 Pro** 的速度显著提升，推测其可能使用了 **GPT-4.1**。
   - 这种性能提升可能会伴随一些限制，例如推理模型的每日消息上限。
- **Spaces 热潮：自定义指令升温**：成员们讨论了通过添加自定义指令来优化 **Spaces** 功能的使用。
   - 一位用户澄清说，**指令字段提供了更多选项**，例如添加特定网站进行数据检索。
- **Deep Research API 具备结构化输出功能**：一位正在构建产品的成员表示，他们熟悉 **deep research 和 structured outputs API**。
   - 他们还要求与相关人员讨论 **Enterprise API 定价**、早期访问、速率限制和支持，并申请了一些额度以适当地测试和集成 API。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen3-30B 导致聊天机器人停机**：**Qwen 发布了 Qwen3-30B**，一位用户报告称，当他们在聊天机器人上询问 **Qwen3** 为什么中国要审查互联网时，系统立即终止了他们的请求。
   - **Qwen3-30B** 的发布令社区感到兴奋，并提供了 [Hugging Face](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF) 链接以供进一步探索。
- **GLM 4.5 Air 模仿 Gemini**：用户讨论了 **GLM 4.5 Air**，有人提到它的感觉很像 **Gemini**，并推荐了一篇[将其与其他模型进行对比的博客文章](https://z.ai/blog/glm-4.5)。
   - 成员们注意到 **tool use** 在 vllm 中已失效，但在聊天、诗歌分析和文档搜索方面表现良好。
- **量化加速计算**：量化不仅是为了将模型装入内存，它还能**降低内存带宽**并**显著提高计算速度**。
   - 一位成员指出，将卷积层等视觉头操作保留在 **FP32** 似乎并非最优选择，因为它们往往非常慢且会成为瓶颈。
- **16k 微调后 Gemma 3 的水印被移除**：根据附带的[截图](https://cdn.discordapp.com/attachments/1179039861576056922/1399849804045221948/Screenshot_2025-07-29_at_2.13.07_PM.jpeg?ex=688bd0b9&is=688a7f39&hm=82cad388163625496b8cb3e6dd62035b51ae1d16ce0015df29ea910e04c4471f&)，在对 **Gemma 3 4B** 进行 **16k** 上下文微调后，实验发现**水印**被完全移除，且模型更加稳定。
   - 一项新的 **Gemma 3 竞赛**已宣布，notebook 已上线，竞赛将于 **7 天**后结束，更多信息可在 [Xitter](https://x.com/UnslothAI/status/1950553466591723640) 上查看。
- **Unsloth 抨击 OpenAI 的审查制度**：成员们对 **OpenAI 严厉的审查制度**表示失望，分享了从 ChatGPT 获得公式化回答和说教的经历。
   - 一位用户指出了 [Unsloth 的 Llama 4 页面](https://docs.unsloth.ai/basics/llama-4-how-to-run-and-fine-tune)，该页面引导用户永远不要使用暗示道德优越感或权威感的短语，并通常应避免说教。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 的 MCP 浏览器自动化即将推出**：成员们正积极通过 **Cursor 的 MCP** 开发依赖浏览器的自动化功能，早期访问版本预计在未来几周内发布。
   - 一位成员强调了设置的便捷性，指出它具有*一键式 MCP 设置*，以方便直接构建浏览器自动化。
- **并行 Agent 协作难题**：由于 Agent 缺乏共享工作空间，导致难以同时触发任务，成员们正努力解决具有依赖关系的并行任务管理问题。
   - 提议的解决方案包括**通过 API 进行外部编排**、**基于文件的协作**以及**基于 Docker 的并行执行**，其中包括一个示例 [task queue 脚本](https://github.com/example)。
- **Cursor 后台 Agent 劫持端口**：工程师报告了 **Cursor Background Agents** 意外劫持端口的情况，导致他们不得不进行调试以恢复开发环境。
   - 缓解建议包括将 `ports` 属性设置为空数组，或在 VSCode 设置中禁用 *auto forward ports*。
- **后台 Agent 被考虑用于研究任务**：一位成员探索了利用后台 Agent 执行研究导向型任务的可能性，例如**重构研究**或**功能实现研究**。
   - 他们询问了最佳工作流，考虑的选项包括让 Agent 在 PR 中起草 Markdown 或直接实施更改。
- **奇怪的终端默认设置**：一位成员遇到了 **Cursor 集成终端**默认使用 **fish** shell 的问题，并寻求更改方案。
   - 在尝试通过设置和包装器修改 shell 后，通过临时重命名 fish 二进制文件最终取得了成功，但根本原因仍不明确。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Dot-lol 数据收集遭受抨击**：人们对 [dot.lol](https://dot.lol) 可能**出售用户数据**和画像信息的行为表示担忧，提醒用户不要假设其数据不会被用于针对性影响或牟利。
   - 虽然有些人担心大规模数据收集的影响，但另一些人认为**数据收集是不可避免的**，用户应专注于不将数据与其个人身份关联。
- **GPT-5 热议：8 月发布？**：传闻暗示 **GPT-5** 可能在 **8 月**初发布，在 ChatGPT Mac 应用中发现了路由架构（router architecture）准备工作的潜在证据。
   - 社区成员正在推测其影响以及它是否会超越其他模型，一些人希望会有免费层级。
- **GDPR：有力还是无力？**：成员们辩论了欧盟 **GDPR** 在防止 AI 公司收集数据方面的有效性，对其影响持有不同意见。
   - 一些人认为 **GDPR** 主要影响数据的*使用*而非*收集*，而另一些人则反驳称*针对欧盟消费者的数据收集已被关闭*。
- **Zenith 重返竞技场**：**Zenith** 模型回归 **LMArena** 引发了热潮，人们对其潜在的 **ELO** 评分和整体性能充满期待。
   - 虽然有些人遗憾错过了试用机会，但另一些人对其在平台上的价值持有强烈观点。
- **灯光、镜头、AI：Video Arena 上线**：**LMArena** 团队在 Discord 上推出了实验性的 **Video Arena**，用户可以免费生成并比较顶级 AI 模型的视频，使用 LMArena 机器人来**生成视频、图像以及图生视频（image-to-videos）**。
   - 官方宣布将与机器人开发者 [Thijs Simonian](https://www.linkedin.com/in/thijsdev/) 进行 **Staff AMA**，邀请用户通过 [Google Forms](https://docs.google.com/forms/d/e/1FAIpQLSfS_zh67vgEeftUXzX6ujlNiiFoTQNQd7P7-Nx6aD5zpFVOBg/viewform?usp=dialog) 提交 AMA 问题。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF Spaces 发生重启**：成员们讨论了 **HF Spaces** 可能会意外重启的问题，建议固定依赖版本以避免问题，详见[文档](https://huggingface.co/docs/hub/spaces-sdks-gradio-hardware#restarts)。
   - 一位用户报告称 *Ilaria RVC* 和 *UVR5 UI* 都停止了工作，并建议进行工厂重建（factory rebuilds），而其他人的则运行正常。
- **P104-100 GPU 仅售 15 美元！**：用户们辩论了 **P104-100** GPU 在 AI 任务中的价值，有人声称它在 [淘宝](https://item.taobao.com/item.htm?id=897611642586&sku=Video%20memory%20capacity:8gb;Color%20classification:Galaxy%20p104-100%20(card%20length%2028cm);&sku_id=5919375243616) 上仅需 **15 英镑**（尽管有人称其为骗局），*实际性能接近 1080*。
   - 一些人提到了 **PCIE 1.0 x4** 等限制，而另一些人则强调了它在 LLM 推理中的性价比，即使是在多卡分片（sharding）运行模型时。
- **Qwen 30B 挑战 GPT-4o**：用户们关注了 **Qwen 30B** 模型的发布，声称其可与 **GPT-4o** 媲美，并且使用 [Unsloth GGUF 版本](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF) 仅需 33GB RAM 即可在本地以全精度运行。
   - 用户指出量化版本仅需 17GB RAM 即可运行。
- **扩散模型 MIT 课程学习小组**：一个新的**学习小组**将专注于使用 **MIT 课程**从零开始学习**扩散模型（diffusion models）**，早期报名费用为 **$50/月**，并为非会员提供两次免费介绍课程：**8 月 2 日**和 **8 月 9 日**，详情见 [Luma](https://lu.ma/kv8zf6va)。
   - 该学习小组将基于 [MIT 的讲义](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf)和[之前的录音](https://aischolars.notion.site/)。
- **MoviePy 打造视频编辑服务器**：一位成员使用 **MoviePy** 构建了一个 **MCP 服务器**，用于处理基础的视频/音频编辑任务，并与 **Claude Desktop** 和 **Cursor AI** 等客户端集成，代码托管在 [GitHub](https://github.com/Aditya2755/video-edit-mcp)。
   - 作者正在寻求关于基于对象检测的编辑和 TTS/SST 驱动剪辑等功能的合作。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-5 热潮与焦急的粉丝**：用户正在热烈讨论 **GPT-5** 的发布，一些人声称可以通过 **Microsoft Copilot/Azure** 访问，但怀疑者仍在等待 **OpenAI** 的正式公告。
   - 一位用户幽默地批评 **Sam Altman** 煽动热度，让粉丝们变得“焦躁不安”。
- **Study and Learn：分心还是创新？**：**OpenAI** 推出了新的 **Study and Learn** 功能，一些人认为这只是一个简单的系统提示词（System Prompt），或许是为了转移大家对 **GPT-5** 预期的注意力。
   - 一位用户甚至将 [系统提示词](https://discord.com/channels/974519864045756446/998381918976479273/1399983224880496711) 丢给 **O3** 模型进行分析。
- **Copilot 与 ChatGPT 的对决**：讨论明确了 **Microsoft Copilot** 目前使用的是 **GPT-4o** 或 **O4-mini-high**，根据源代码提示，未来可能会集成 **GPT-5**。
   - Copilot 无限制的每日消息额度引发了用户为何仍偏好 **ChatGPT** 的疑问，尽管部分用户仍认为 Google 的 [Imagen4-Ultra](https://discord.com/channels/974519864045756446/998381918976479273/1400170254902235246) 是最出色的图像生成器。
- **聊天记录凭空消失**：多位用户报告 **ChatGPT 聊天记录** 消失，尽管尝试了重新登录和清除缓存等排障操作。
   - OpenAI 支持团队表示这 *可能是一个孤立的 Bug*，并强调 *聊天记录一旦丢失便无法恢复*。
- **工程化新型 AI 内存格式**：一名成员提出了一种名为 [AI_ONLY_FORMAT_SPEC](https://discord.com/channels/1046317268651794482/1046317269069864970?event=1200057848907847709) 的新内存格式提案，旨在优化 **AI VM**、与向量数据库对接的系统或受保护的符号传输，强调速度和效率而非人类可读性。
   - 另一名成员对该格式进行了详细的逐行解读，强调了其核心原则，如 **Token Embedding**（Token 嵌入）、**Semantic Grouping**（语义分组）和 **Binary Encoding**（二进制编码）。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeepSecure 发布开源 AI Agent 身份验证方案**：**DeepTrail** 推出了 **DeepSecure** ([https://github.com/DeepTrail/deepsecure](https://github.com/DeepTrail/deepsecure))，这是一个针对 **AI Agent** 的开源身份验证和委托层，支持跨模型、平台和框架的授权、Agent 间委托、策略执行和安全代理。
   - 该技术采用分裂密钥架构、网关/代理、分离的控制/数据平面、策略引擎和 Macaroons。在 **Langchain/LangGraph** 的集成示例中展示了具有细粒度访问控制的 [安全多 Agent 工作流](https://github.com/DeepTrail/deepsecure/blob/dev/examples/05_langchain_secure_tools.py) 和 [委托工作流](https://github.com/DeepTrail/deepsecure/blob/dev/examples/09_langchain_delegation_workflow.py)。
- **OpenRouter 充值 10 美元赠送每日免费消息**：在 **OpenRouter** 上一次性充值 **10 美元** 即可解锁 **每日 1000 条免费消息**，即使初始额度用完后依然有效。
   - 用户确认在消耗完初始的 10 美元后，**1000 次请求/天** 的限制依然保持激活状态。
- **API 屏蔽不需要的量化版本**：OpenRouter 的 API 现在允许用户指定可接受的量化级别，以通过 [提供商路由文档](https://openrouter.ai/docs/features/provider-routing#quantization-levels) 避开像 **FP4** 这样低精度的模型。
   - 该 API 允许排除特定的量化级别，例如允许除 **FP4** 模型以外的所有模型。
- **DeepInfra 的 Gemini Pro 秘密协议**：**DeepInfra** 与 **Google** 就 **Gemini 2.5 Pro** 达成了更低的费率，并将优惠转嫁给了客户，DeepInfra 列表上的“Partner”标签体现了这一点。
   - 与 **Kimi K2** 模型不同，DeepInfra 的 **Gemini 2.5 Pro** 带有合作伙伴标签，标志着与 Google 的直接合作关系。
- **Ori Bot 表现不佳**：用户报告 **OpenRouter Ori Bot** 可能会产生负面影响，因为其回答不够准确，特别是在处理付款流程问题时。
   - 一位用户指出 **Ori** 经常将错误归咎于用户，并提出一些 *毫无意义* 的问题；目前一名开发人员正在努力更新 **Ori** 的知识库。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Metislist 排名引发 François 热潮**：一位用户分享了 [Metislist](https://www.metislist.com/)，引发了关于 **François Chollet** 排名第 80 位的争论，许多人认为这位 **Keras** 的创始人理应获得更高的排名。
   - 有人认为 Chollet 应该排在前 50 名，一位用户调侃道：*“该死，你跟我兄弟 François 有仇吗？”*。
- **Arcee 发布 AFM-4.5B 模型**：Lucas Atkins 宣布在 Hugging Face 上发布来自 [Arcee.ai](https://xcancel.com/LucasAtkins7/status/1950278100874645621) 的 **AFM-4.5B** 和 **AFM-4.5B-Base**，并称由于与 DatologyAI 的数据合作，该模型具有灵活性、高性能和高质量。
   - 这些模型融合了诸如 **grouped query attention** 和 **ReLU² activations** 等架构改进，未来还计划发布推理和工具使用（tool use）版本。
- **NotebookLM 现在可以总结视频**：**NotebookLM** 推出了针对文章和博客文章生成视频概览的新功能 ([xcancel.com](https://xcancel.com/NotebookLM/status/1950298236914139234))，使用户无需阅读全文即可掌握内容。
   - 用户称赞了这一创新，并建议进一步开发交互模式。
- **MacOS 中发现 GPT-5 踪迹**：在 MacOS 应用缓存文件中发现了对 **gpt-5-auto** 和 **gpt-5-reasoning** 的引用 ([xcancel.com](https://xcancel.com/apples_jimmy/status/1950514936444305534?s=46&t=fRVjULzONZQAlwHruKTgQg))，暗示 **GPT-5** 即将到来。
   - 进一步的佐证提到在一个生物学基准测试仓库中发现了 **gpt-5-reasoning-alpha**，引发了关于潜在发布或公告的猜测。
- **Anthropic 寻求 1700 亿美元高估值**：据报道，Anthropic 正寻求筹集 **50 亿美元**，这可能使这家 AI 初创公司的估值达到 **1700 亿美元**，预计到今年年底收入将达到 **90 亿美元** ([xcancel.com](https://xcancel.com/EdLudlow/status/1950561790695448810))。
   - 该消息引发了与 OpenAI 和 xAI 的对比。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Tableau Server 展示 LLM 编排实力**：一位成员报告成功集成了最新的 **Tableau Server 版本**，以引入 **LLM (server/on prem)** 用于 **Vizql NLP**。
   - 此设置旨在为 Tableau 可视化提供更复杂的自然语言处理能力。
- **Gemini Agent 框架原型出现**：一位成员分享了一个 [**Gemini agentic framework**](https://cdn.discordapp.com/attachments/1124403655819415592/1399853283404939454/gemini-agentic-framework.zip?ex=688bd3f6&is=688a8276&hm=101f03e62cae13a72e1f4fdc681064aef0e5a3713de20aebac608c958f845b8b) 原型，并将其描述为一个 **one-shot prototype**。
   - 该原型利用 **AI Studio** 构建 Agent 应用，强调为构建者 Agent 设置清晰的意图，以促进分阶段测试和专注的模型开发。
- **NotebookLM 绕过机器人限制实现播客梦想**：针对 **NotebookLM** 因机器人限制导致播客创建受限的询问，一位成员澄清说这些工具可以通过 **API** 访问。
   - 他们建议重新构建工作流并手动将报告加载到 **NotebookLM** 中，作为一种替代方案。
- **Obsidian 与 NotebookLM 紧密结合**：这里分享了一篇详细介绍 **NotebookLM**、**Obsidian** 和 **Google Drive** 集成的文章 [点击此处](https://www.xda-developers.com/using-notebooklm-obsidian-google-drive-together/)。
   - 一位成员自愿根据个人用户需求提供关于 **Obsidian** 使用的更详细指导。
- **NotebookLM 音频输出在 8-15 分钟之间波动**：用户报告 **NotebookLM** 生成的音频文件平均时长为 *8-10 分钟*，尽管有些人已经达到了 *15 分钟*。
   - 这一讨论强调了输出长度的可变性，这可能受到内容复杂性和处理效率的影响。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **俄罗斯地震后发布海啸警报**：俄罗斯海岸附近发生 **8.7 级地震**，触发了夏威夷的海啸警报以及美国西海岸的海啸预警。
   - 受影响地区的居民被建议密切关注更新，因为海啸可能在数小时后到达。
- **LM Studio 用户请求增强会话处理功能**：用户请求在 LM Studio 中增加复制和粘贴整个会话的功能，这些会话以 **JSON 格式**存储。
   - 一名用户引导其他人关注 [功能请求频道](https://discord.com/channels/1110598183144399058/1128339362015346749)，并指出许多人会发现这个功能很有用。
- **LM Studio 模型“重回” 2 月 18 日**：一位用户报告称，他们的 LM Studio 模型反复提及 **2024 年 2 月 18 日**，即使在询问当前事件时也是如此。
   - 另一位用户建议检查 **system prompt** 或 **Jinja template** 中的日期设置。
- **Strix Halo APU 价格昂贵，引发 EPYC 讨论**：**Strix Halo APU** 的价格在 64GB 版本约 **1600 美元**，128GB 版本约 **2000 美元**，但一些成员建议 **EPYC** 系统提供更好的性价比。
   - 一名成员对这类设备上的*板载焊接内存*表示遗憾，并将其与最近服务器上的 DIMM 故障进行了对比，同时指向了 [搭载 Strix Halo APU 的 Corsair AI Workstation 300](https://www.guru3d.com/story/compact-ai-pc-corsair-ai-workstation-300-with-strix-halo-apu/)。
- **9070 XT 性能令人失望**：**9070 XT** 的速度明显慢于 **4070 Ti Super**，一位用户报告称，在 **4070 Ti Super** 上运行速度为 **7 t/s** 的模型，在 **9070 XT** 上仅达到 **3 t/s**。
   - 另一位成员建议 RAM 带宽限制可能是原因所在。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaCloud 读取 PDF 遇到困难**：一位成员报告称 **LlamaCloud** 无法检测 **PDF 文件** 并通过 API 进行处理，该成员使用 **n8n** 简化工作流，并链接了一张 [截图](https://cdn.discordapp.com/attachments/1059201661417037995/1399848961832911031/Screenshot_2025-07-29_at_22.13.16.png?ex=688bcff0&is=688a7e70&hm=b8f51e99fbeae087df203303f7665c4eab8447bb0890b55823fd36074c5ad539&)。
   - 这一问题发生在尝试通过 API 处理 **PDF 文件** 时。
- **Character AI 引发构建讨论**：成员们讨论了如何构建一个对宏大故事有深刻理解的 **character AI**，使用经典的 **RAG** 流水线，包括文本分块（chunked text）、嵌入（embeddings）和向量数据库。
   - 这包括利用 **RAG** 流水线来创建一个具有深度理解能力的 AI。
- **Neo4j 知识图谱遇到障碍**：一位成员报告称，他们实现 **Neo4j** 的简单图存储加载时间*长得离谱*，且其服务器与 **Neo4j 5.x** 不兼容，而 **LlamaIndex** 似乎不喜欢 **4.x** 版本。
   - **Aura** 也被服务器代理拦截，为实施增加了进一步的障碍。
- **Flowmaker Gemini 2.5 Pro Bug 得到快速修复**：一位成员报告了在使用 **Flowmaker** 配合 **Gemini API** 时因模型名称无效而产生的错误，该错误要求使用类似 *gemini-2.5-pro* 的编号。
   - 修复代码已提交 [committed](https://github.com/run-llama/flow-maker/blob/aad0f47a81cacba662a07c4f2d70bd3425606e29/src/lib/llm-utils.ts#L19) 并迅速部署，解决了该问题。
- **社区提供 RAG 调试协助**：一位成员提供了一个 **MIT 许可的仓库**，旨在调试棘手的 **RAG 问题**，包括稀疏检索（sparse retrieval）、语义漂移（semantic drift）、分块崩溃（chunking collapse）和内存崩溃（memory breakdowns）。
   - 在初步提议后，一位社区成员询问了该仓库解决的具体复杂问题，重点关注*稀疏检索*和*语义漂移*的具体案例。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **专家并行（Expert Parallelism）表现卓越？**：一位成员寻求 **Expert Parallelism (EP)** 优于 **Tensor Parallelism (TP)** 的案例，但发现对于 **Qwen32B** 和 **Qwen 235B**，all-reduce 通信开销使得 **EP** 性能较低。
   - 他们观察到 **EP** 仅对使用 **MLA** 且需要 **DP attention** 的模型有益。
- **Torch Compile 中的 Triton 宝藏**：为了提取 **PTX 代码**，成员建议使用 `TORCH_LOGS="output_code" python your_code.py` 或访问 `compiled_kernel.asm.keys()` 字典，详情参考[这篇博文](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/?utm_source=chatgpt.com#ttir-triton-ir)。
   - `keys()` 字典包含中间表示的键，包括 **llir, ttgir, ttir, ptx 和 cubin**。
- **Inductor 对 Triton 的有趣影响**：为了强制为 matmuls 生成 **Triton 代码**，成员建议配置 [torch._inductor.config.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L459-L461) 中的设置，通过修改 **use_aten_gemm_kernels**、**autotune_fallback_to_aten**、**max_autotune_conv_backends** 和 **max_autotune_gemm_backends** 等设置。
   - 然而，有人指出内置算子（built-in kernels）通常更快，且并非每个操作默认都会转换为 Triton。
- **CuTeDSL 编译器简化预取代码**：一位成员分享了关于在 **H100 上使用 CuTeDSL 进行 GEMM** 的[博文](https://veitner.bearblog.dev/let-the-compiler-do-the-work-in-cutedsl/)和[代码](https://github.com/simveit/software_pipelining_cute_dsl)，解释了如何让编译器处理预取（prefetching）。
   - 该博文详细介绍了一个传给 `cutlass.range` 算子的实验性参数，用于提示预取，从而以更简单的代码达到手动预取的性能。
- **Gmem 守护者：Synchthreads 拯救世界**：在从全局内存（**gmem**）拷贝到共享内存（shared memory）后，必须手动插入 `synchthreads`（或等效指令）以在继续执行前同步所有线程。
   - 这保证了所有共享内存元素在进行 **gemm**、reduction 或 scan 等集体计算时均已就绪。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **M3 Ultra 荣登本地推理之王**：根据 [Reddit 上的这个帖子](https://www.reddit.com/r/LocalLLaMA/comments/1j43ziq/the_new_king_m3_ultra_80_core_gpu_512gb_memory/)，**M3 Ultra** 因其 **80核 GPU** 和 **512GB 内存** 正成为本地推理的首选。
   - 另一位成员则因为没人回帖而买了一台二手 **M1 16g**。
- **解决离岸模型延迟问题**：一位成员正在寻求在网络条件较差的情况下，低延迟运行 **LLMs offshore** 的解决方案。
   - 其他成员则简单地表示，他们愿意在自己想要的东西上花钱。
- **微软的语音指令研究**：成员们对研究开源解决方案以提高 **Speech-LLM 模型** 中的**语音指令遵循（audio instruction-following）**能力表现出兴趣，旨在创建更好的语音 UI，并提到了 **Microsoft 的 Alignformer**。
   - Alignformer 并非开源，因此可能需要开展协作。
- **ICL 瓦解了诊断工具？**：成员们推测**上下文学习（ICL）**可能会破坏诸如**稀疏自编码器（SAEs）**之类的**可解释性工具（interpretability tools）**，因为 **ICL** 会将激活值推离原始训练分布，参考了 **Lucas Critique** 和[这篇论文](https://arxiv.org/abs/2501.00070v1)。
   - 成员们认为，这个问题并非 **ICL** 独有，而是每当 **SAEs** 遇到与其训练分布不同的激活分布时都会出现。
- **Grouped GEMM 取得进展**：一位成员强调了一个在 **GPT-NeoX** 中支持 **torch._grouped_mm** 的 PR（现已进入 PyTorch 核心库），这暗示了 **MoE** 的性能提升，并链接到了[这个 MoE 实现](https://github.com/EleutherAI/gpt-neox/blob/d12c771198388980ee054617e537665f044e0584/megatron/model/moe_mlp.py#L221)。
   - 他们指出，感兴趣的用户可以使用 TorchAO 的一行代码来进行**低精度 MoE 训练**。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **关于 CUDA 泛化表面的论文**：一名成员分享了一篇关于超越 **CUDA** 泛化的[论文](https://huggingface.co/papers/2507.14111)，并提醒应将此类内容发布在适当的频道中。
   - 未提供进一步的讨论或替代观点。
- **Mojo 新手挑战 TLS 握手故障**：一位新的 Mojo 用户报告称，在使用来自 Microsoft Copilot 的 **Dockerfile**，并尝试通过 **pixi** 和 **magic shell** 运行 Mojo 项目时，遇到了 **TLS handshake EOF error**。
   - 建议的修复方案包括使用最新的 nightly `mojo` 包配合 **pixi** 以及特定命令（`pixi init my-project -c https://conda.modular.com/max-nightly/ -c conda-forge` 和 `pixi add mojo`），但即使使用了 **VPN**，该修复方案仍然失败。
- **剖析 Mojo external_call**：用户询问为什么 Mojo 的 `external_call` 使用特定的函数（如 `KGEN_CompilerRT_IO_FileOpen`）而不是 **libc** 中的标准 `fopen`，并担心这种选择是否出于安全考虑。
   - 一名成员澄清说，这些是旧版本 **Mojo** 的遗留产物，修复优先级不高，且 **KGEN** 命名空间属于 Modular，最终将会开放。
- **从 Mojo 调用 Python 遭受开销影响**：用户发现，与直接执行 Python（0.5 秒）相比，从 Mojo 调用 Python 的 no-op 函数存在显著开销（1000 万次调用耗时 4.5 秒）。
   - 成员解释说，Mojo 需要启动一个 **CPython** 进程，并且 CPython 是通过 `dlopen libpython` 嵌入的，因此不应该在 **hot loop**（热循环）中调用它。
- **在赛车引擎中喷胶**：讨论涉及了从 Mojo 调用 Python 的性能影响，特别是在处理 **OpenCV** 或 **Mujoco** 机器人仿真等任务的 hot loops 中。
   - 成员指出，许多快速的 Python 库实际上是带有封装的 C 库，而*仅与上下文字典（context dicts）交互就很容易消耗数百个周期*。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Deepseek-chat 在 OpenRouter 上表现不佳**：成员观察到，与官方 **DeepSeek API** 相比，**Deepseek-chat** 在 **OpenRouter** 上的表现较差，特别是在架构模式（architect mode）中作为编辑器模型（editor model）使用时。
   - 推荐的修复方法是使用 `aider --model openrouter/deepseek/deepseek-r1`，以确保应用来自 [aider/resources/model-settings.yml](https://github.com/Aider-AI/aider/blob/main/aider/resources/model-settings.yml#L548) 的默认配置 `edit-format: diff`。
- **Aider 作为编程模型训练的催化剂？**：有建议认为 **Aider** 可以通过在开发工作流中记录 linting 和撤销（undo）操作来辅助编程模型的训练。
   - 这种方法将在“谨慎”的开发中使用 **Aider** 以生成有价值的训练数据，尽管评论者并未明确要求开发人员实现此功能。
- **Qwen3 Coder 30B-A3B 亮相**：发布了关于新模型 **Qwen3 Coder 30B-A3B** 的公告，并分享了一张图片以验证其真实性。
   - 该新模型的详细信息仍在陆续公布中。
- **Litellm API 遭遇连接错误**：一位用户报告遇到大量 `litellm.APIConnectionError: Error parsing chunk: Expecting property name enclosed in double quotes: line 1 column 2` 错误。
   - 尽管出现了这些错误，用户的功能并未受到影响。
- **请求 Open Model R1 与 Qwen Coder 的对决**：一名成员请求关于最适合 **aider** 的开源模型的建议（考虑到硬件资源无限），并表示有兴趣测试 **R1** 和 **Qwen Coder** 模型。
   - 该成员提到拥有 **Runpod credits** 可供使用，表明计划对这些模型进行实际测试。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **寻求 LLM 安全研究资源**：一位博士生请求关于当前 **LLM safety/alignment research**（LLM 安全/对齐研究）的资源，建议包括来自 **AI alignment forum** 的博客。
   - 特别提到了博客文章 [Thought Anchors: Which LLM Reasoning Steps Matter](https://www.alignmentforum.org/posts/iLHe3vLur3NgrFPFy/thought-anchors-which-llm-reasoning-steps-matter) 和 [Measuring Beliefs of Language Models During Chain-of-Thought](https://www.alignmentforum.org/posts/a86uAnPykqNtmEbDH/measuring-beliefs-of-language-models-during-chain-of-thought-1)，认为它们是很好的入门起点。
- **用 Claude 编写 CUDA 让程序员感到困惑**：一位成员发现使用 **Claude** 编写 **CUDA** 非常困难，需要大量的准备、理解和排列工作。
   - 他们认为最终的评估标准应该是：一个具备一定 GPU 和 **CUDA** 基础的 Python 程序员，是否能利用 **Claude** 来管理 **kernels** 的编写并提升性能，文中还包含[一张图片](https://cdn.discordapp.com/attachments/986699377257119794/1400233738369110026/image.png?ex=688be4ca&is=688a934a&hm=1bcb11346477e61edf05cde9751d5e62ee8992a2f64216c07e4a1a8f8fb14cc4)。
- **Z.AI 的 54 个仓库引发关注**：一位成员询问了新的 **Z.AI 54 open source repos**，并打听是否有人研究过它们，这引发了社区的好奇。
   - 然而，关于这些仓库的具体内容或功能的细节尚未得到详细阐述。
- **据称 Qwen3 与 GPT-4o 相当**：一位成员分享了一篇帖子，指出 [Qwen3 30B A3B 2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) 在英文和中文方面都能与 **OpenAI** 的 **GPT-4o** 媲美。
   - 社区对 **Qwen3** 作为一个潜在的强力竞争者充满热情，基准测试显示其性能可能与 **GPT-4o** 持平，特别是在多语言应用中。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi 聊天机器人说中文**：一位用户报告称，尽管输入的是 **English** 提示词，**Kimi chatbot** 却用**中文**回答，这可能是因为账号已登出。
   - 截图显示，虽然回复是英文，但建议的来源和问题却是中文。
- **Kimi 从社交媒体学习**：一位成员开玩笑说 **Kimi** 的训练数据集包含 **Instagram** 和 **TikTok** 的评论，并链接到 [Kimi on OpenHands v0.50.0k2](https://github.com/All-Hands-AI/OpenHands/releases/tag/0.50.0k2) 来支持这一说法。
   - 他们认为这种对社交媒体数据的关注正是 **Kimi** 表现出色的原因。
- **Moonshot AI 的氛围最好**：一位成员表示 *moonshot got the best vibe no cap*（Moonshot 的氛围真的最好），并链接了一篇关于 AI 社区氛围检查的 [X 帖子](https://x.com/crystalsssup/status/1944287779896328668)。
   - 另一位成员表示赞同，认为社区需要一些竞争。
- **Scale AI 提供数据**：一位成员指出 **Alexandr Wang** 是 **Scale AI** 的创始人兼 CEO，该公司提供训练数据、标注和评估服务。
   - 他们指出 **Scale AI** 对于开发机器学习模型至关重要。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Lume 在正面交锋中略胜 Suna**：成员们讨论了 **Lume** 和 **Suna** 这两个 **agentic systems**（Agent 系统）的优劣；一位成员发现 **Lume** 在编写特定代码时错误更少。
   - 该成员提到由于成本原因无法与 **Manus** 进行比较，并承认他们可能没有正确地对 **Suna** 进行提示（prompting）。
- **Manus 的漫画创作：璞玉待琢？**：一位成员建议 **Manus** 的漫画创作功能很不错，但仍有改进空间，尤其是对于免费用户。
   - 另一位成员表示该服务的质量正在下降，对免费用户的限制非常严格，并质疑 *Manus 是否已经凉了*。
- **AI 对 Manus 的乐观态度与人类的怀疑**：一位成员询问 AI 如何看待 **Manus** 的未来，AI 回复称 *I think the future of Manus is bright*（我认为 Manus 的前景一片光明）。
   - 另一位成员表示怀疑，理由是来自 **OAI** 和 **Google** 发布的 **agent** 模式带来的竞争压力。

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP Server 面临上下文危机**：一位用户询问在单云端部署的 **MCP server 实例**中是否需要额外的**用户上下文隔离 (user-context separation)** 层，以防止独立会话之间的数据共享，并引用了 [issue #1087](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1087) 和 [MCP Docs](https://modelcontextprotocol.io/docs/tutorials/use-remote-mcp-server#understanding-remote-mcp-servers)。
   - 另一位用户报告了通过 **Claude Desktop** 连接其 **MCP server** 时遇到的挑战，尽管已成功部署到配置了正确 **SSL** 的 **EC2**，但仍然只能通过 **Cursor** 连接。
- **Cucumber 与 LLM 正在酝酿 BDD**：一位用户分享了一个生产就绪的**行为驱动开发 (BDD)** 侧边项目，其中包括[该解决方案的架构图](https://cdn.discordapp.com/attachments/1312302100125843476/1399854833565044756/bael.jpeg?ex=688bd568&is=688a83e8&hm=2e86139e9f117265cd7cbef2afcc1a23a34a091e79402df9a0e051261231c695&)。
   - 另一位用户报告 **state tool** 在 **Claude desktop** 中无法与 **CursorTouch** 的 **Windows-MCP** 配合使用。
- **DeepTrail 为 AI Agent 身份验证推出 Deepsecure**：由 Berkeley SkyDeck 支持的 **DeepTrail** 正在开发 **Deepsecure**，这是一个为 AI Agent 设计的开源身份验证和授权代理层，已在 [GitHub](https://github.com/DeepTrail/deepsecure) 上开源。
   - **Deepsecure** 的架构具有分片密钥设计、网关/代理、独立的控制/数据平面、策略引擎以及用于 Agent 间授权的 Macaroons，详见其[技术概览](https://github.com/DeepTrail/deepsecure/blob/dev/docs/design/deepsecure-technical-overview.md)。
- **FastMCP 被问及工具动态**：一位用户询问了当服务器上定义了多个工具时，**FastMCP** 的动态工具选择能力。
   - 具体而言，他们想知道 **FastMCP** 是否具有在客户端自动选择工具（例如数学、网络搜索、RAG、数据解释器）的逻辑。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 参数化：可学习参数提案引发关注**：一项在 DSPy 中添加可学习参数（`dspy.variable` 或 `dspy.parameter`）的提案引发了热烈讨论，并在 [GitHub 上创建了 issue](https://github.com/stanfordnlp/dspy/issues/8593) 以收集想法和用例。
   - 其目标是通过允许*模板作为参数/变量*并优化模板变量的位置，使 Optimizer 能够生成最优的 Prompt。
- **F-Strings 遇挫：Signature 实现受阻**：一位成员在尝试使用 f-string 实现 Signature 以根据描述验证代码时遇到问题并寻求帮助。
   - 另一位成员建议不要采用这种方法，推荐将参数描述放置在 `dspy.InputField()` 中。
- **DSPy 进入遗传算法 Prompt 领域**：一位成员分享了一个比较 **DSPy** 与 **GEPA** 的 YouTube 视频，并链接了该 [YouTube 视频](https://www.youtube.com/watch?v=o6RbVPFOslg)，视频中提到 *DSPy 优化你给出的 Prompt；而 GEPA 则进化出你从未想象过的 Prompt*。
   - 该成员建议将 **MIPRO** 进化为 DSPy 的反射式遗传风格 Frontier 引擎来生成 Prompt，以此挑战该博主的观点。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **AMD 在游戏和 AI 领域一路领先**：一位成员建议将 **7900XT** 和 **7800X3D** 搭配用于游戏，并指出 AMD 在消费级 AI 的可用性和长期社区利益方面的优势。
   - 他们链接到了[一条推文](https://x.com/Teknium1/status/1950596567968477382)，该推文支持 AMD 优于 Nvidia 的 **9070** 和 **9900X**。
- **Qwen 发布具备思考能力的编程模型**：一位成员宣布在 [Hugging Face](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507) 上发布了 **Qwen3-30B-A3B-Thinking-2507** 编程模型。
   - 链接的 Hugging Face 模型为代码生成提供了一个新工具。
- **RLVR：是算法还是营销？**：一位成员质疑将 **RLVR** (Reinforcement Learning, Virtual Reality) 归类为强化学习算法是否准确，引发了讨论。
   - 另一位成员 teknium 在回应 [NVIDIA 的推文](https://fxtwitter.com/NVIDIAAIDev/status/1950279130450444670)时表示：*"RLVR 根本不是一种 RL 算法，它只是 RL 的一个目标"*。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **研究员请求开设 AI Safety Summer School 频道**：一名新成员请求为几周前举行的 AI Safety **summer school** 开设专门频道。
   - 这反映了社区内对学术和学习导向讨论的持续兴趣。
- **偏见研究爱好者集结对抗偏见**：一名 JKU Linz 的博士生正专注于 **mitigating social bias in LLMs**（缓解 LLM 中的社会偏见），研究兴趣包括 **attribution for generative models**（生成模型的归因）、**AI generated text detection**（AI 生成文本检测）以及 **domain adaptation**（领域自适应）。
   - 该学生热衷于与其他从事领域特定 LLM 实际伦理问题研究的人员建立联系，寻求合作。
- **内核开发者在 CUDA 中编写内核**：一位名为 Ali 的成员正深入参与 **Triton/CUDA** 中的 **GPU kernels** 优化，特别是针对 **autoregressive models**（自回归模型）。
   - Ali 愿意讨论底层 GPU 编程，并提供加速模型性能方面的专业知识。
- **Cohere 面临引用配置难题**：一名成员报告在 `langchain_cohere.ChatCohere` 上使用 `citation_options` 更改 **citation mode**（引用模式）时遇到困难，并询问传递引用选项的隐式方法。
   - 该成员还询问了 [langchain-cohere repo](https://github.com/langchain-ai/langchain-cohere) 的状态，指出其缺乏近期更新，并询问*是否欢迎提交 pull requests*。
- **资深软件专家在南方寻求职位**：发布了一个远程 **Senior Software Engineer** 职位的广告，月薪 **$2K**，长期合同，工作地点要求在**非洲**或**美洲**。
   - 该职位要求具备 **Ruby on Rails**、**Node.js**、**C#/.NET**、**Python**、**Java** 经验以及出色的英语沟通能力。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 用户请求 LoRA-Style Adapters**：一名 Torchtune 用户对 **LoRA-style adapter support** 表示感兴趣，该支持可以冻结原始模型权重并通过额外的可训练层应用更新。
   - 用户明确希望 adapter 能保持相同的 forward 计算路径，且不增加计算成本。
- **Torchtune 在训练后合并权重**：一名用户指出 Torchtune 在使用 adapter 训练后会将权重合并回去，并引用了 [Torchtune end-to-end workflow documentation](https://docs.pytorch.org/torchtune/0.6/tutorials/e2e_flow.html)。
   - 该用户的评论引发了关于 Torchtune 中合并权重影响的问题。
- **ACL 论文获得嘉奖**：一名成员分享了他们获奖的 **ACL paper**，论文链接在[这里](https://aclanthology.org/2025.acl-long.266/)。
   - 此公告后没有进一步的讨论。
- **Glianorex 微调引发讨论**：一名用户询问 **Glianorex finetunes** 是否公开。
   - 这条评论可能被解读为一种抱怨：*Glianorex 快把我折磨死了，我的医生也帮不上忙*。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **证书声明仍需完成**：一名工作人员提醒另一名成员完成 **LLM Agents (Berkeley MOOC)** 的 **certificate declaration form**（证书声明表）。
   - 工作人员重申，尽管之前已通知该成员，但**从未收到该表格**。
- **第二次证书提交提醒**：工作人员强调了提交 **Certificate Declaration Form** 以完成 **MOOC** 要求的重要性。
   - 未能提交表格将导致无法颁发证书，从而影响课程完成情况的验证。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AI Scholars 启动 Diffusion Models 学习小组**：一个新的学习小组正在启动一个为期 **5 个月**、共 **12 人** 的项目（**每周 2-4 小时**），由 **AI Scholars** 发起，使用 [MIT 的课程体系](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf) 来研究 diffusion models（生成式 AI 的核心架构）。
   - 已确认的成员包括一家 AI 电影工具的 CTO、AI 艺术讲师、2 名 LLM 讲师和 2 名全职 AI 研究员。
- **前两节 Diffusion Models 课程免费**：前两节介绍性课程免费并向非成员开放：8 月 2 日关于 *Flow Matching & Diffusion Models*，8 月 9 日关于 *PDEs, ODEs, SDEs + A Brief History of Diffusion Models*（[链接在此](https://lu.ma/kv8zf6va)）。
   - 该项目特点包括同行引导的课程、导师问答、实战项目、真实研究论文研读，以及一个紧密且互信的小组，采用每周 2 小时直播课 + 2 小时自学的模式，学生轮流授课。

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **用户寻求云部署策略**：一位用户正在寻求关于将使用自定义 PDF 文件夹训练的语言模型部署到云端供公众使用的建议，并特别希望为用户查询提供一个简单的 **GUI**。
   - Nomic 建议 **enterprise plan** 并不合适，用户考虑将 **Hugging Face deployment** 作为替代方案。
- **企业版方案无法满足用户需求**：Nomic 指出 **enterprise plan** 无法满足用户部署自定义语言模型的需求。
   - 用户正在探索其他部署策略，例如 **Hugging Face**，以使其语言模型可供访问。



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **计划评估 Kimi 2？**：一名成员询问了评估 **Kimi 2** 的计划。
   - 他们对后训练（post-training）后的 **tool-use capabilities** 表示好奇。
- **对 Kimi 2 工具使用后训练的兴趣**：有人表示有兴趣评估 **Kimi 2** 在后训练阶段后的 **tool-use performance**。
   - 该询问强调了评估模型在初始训练后如何适应和利用工具的重要性。



---


**tinygrad (George Hotz) Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。


---


**Codeium (Windsurf) Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。


---



您收到此电子邮件是因为您通过我们的网站选择了订阅。

想更改接收这些电子邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。


---

# Discord: 频道详细摘要与链接





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1399830622255710320)** (1152 messages🔥🔥🔥): 

> `R1 1776 Removal, Comet for Android, Gemini 2.5 Pro Speed, OpenRouter API for R1 1776, Daily Message Cap Limits` 


- **R1-der R.I.P: 模型从 LLM 选择器中移除**：成员们注意到 **R1 1776 模型** 已从 LLM 选择器中移除，但仍可通过 [OpenRouter API](https://openrouter.ai/perplexity/r1-1776) 访问。
   - 用户推测在移除后会转向使用 **O3** 或 **Sonnet 4** 进行推理任务。
- **Android Comet: 移动端 App 即将发布**：**Comet for Android** 应用目前正在开发中，预计将于年底发布。
   - 一位用户对浏览器的潜在功能表示担忧，而其他人则称赞其在 Windows 上的表现。
- **Pro Gemini 2.5 提速**：用户观察到 **Gemini 2.5 Pro** 的速度显著提升，猜测它可能使用了 GPT-4.1 而非 Gemini。
   - 成员们指出，这种性能提升可能伴随着限制，例如推理模型的每日消息上限（daily message cap limits）。
- **Spaces 热潮：自定义指令升温**：成员们讨论了如何通过添加自定义指令（custom instructions）来最好地利用 **Spaces** 功能。
   - 一位用户询问空间描述字段或指令字段哪个更适合设置空间上下文——一位成员回答说 **指令字段提供了更多选项**，例如添加特定的网站以提取特定数据。
- **API API!: Perplexity API 指南**：用户分享了使用 **Perplexity API** 的技巧，其中一位指出 Pro 订阅者每月可获得 **$5 USD** 的信用额度。
   - 一位面临 401 错误的用户被建议确保其代码和模型选择（**sonar-pro**）正确。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1399842279937806439)** (4 messages): 

> `Enterprise API pricing, Deep Research API` 


- **Deep Research API 具有结构化输出**：一位正在构建需要大量深度研究能力的产品并准备为发布筹款的成员表示，他们对开发问题比较放松，因为他们*熟悉 **deep research and structured outputs api***。
   - 他们还要求与相关人员讨论 **Enterprise API pricing**、早期访问、速率限制（rate limits）和支持，并请求一些信用额度以便正确测试和集成 API。
- **团队准备回答问题**：一位成员确认团队正在关注，并询问另一位成员有什么问题。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1399837823863095307)** (670 messages🔥🔥🔥): 

> `恶意刷屏, GLM 4.5 Air, Qwen3-30B, OpenAI 审查, Unsloth 与 Llama 3.1` 


- ****Discord 处理恶意刷屏****：一名用户举报了**恶意刷屏仇恨内容**并私信所有人的行为，导致有人呼吁禁用某些词汇。
   - 成员们确认这*并不重要*，并提到 GPU-shaming 是一个常见问题。
- ****用户对 GLM 4.5 Air 赞不绝口****：用户讨论了 **GLM 4.5 Air**，有人提到它的感觉像 **Gemini**，并分享了一篇[将其与其他模型进行对比的博客文章](https://z.ai/blog/glm-4.5)。
   - 成员们指出 **tool use** 在 vllm 中已损坏，但它在聊天、诗歌分析和文档搜索方面表现良好。
- ****Qwen3-30B 发布引发社区兴奋****：**Qwen 发布了 Qwen3-30B** 并[提供了链接](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF)，引发了关于其性能的讨论。
   - 一位用户报告说，当他们在聊天机器人上询问 Qwen3 为什么中国要审查互联网时，系统立即关闭了他们的请求。
- ****OpenAI 的严厉审查令用户失望****：成员们对 **OpenAI 的严厉审查**表示失望，分享了 ChatGPT 给出机械化回答和说教的经历。
   - 一位用户指出了 [Unsloth 的 Llama 4 页面](https://docs.unsloth.ai/basics/llama-4-how-to-run-and-fine-tune)，该页面引导用户永远不要使用暗示道德优越感或权威感的短语，并通常避免说教。
- ****Unsloth 增强 Llama 3.1****：用户询问 **unsloth/Llama 3.1** 与普通 **Llama 3.1** 之间的区别，社区成员澄清说 Unsloth 提供了修复、模板调整和 tokenizer 改进。
   - Unsloth 团队还降低了在消费级硬件上 fine-tune 模型的门槛，提供了更快的 fine-tuning 速度和更低的显存占用。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1400100378560958464)** (6 messages): 

> `Unsloth 介绍, TTS 语音克隆中的低端到端延迟` 


- **印度用户和学生涌向 Unsloth**：一位来自印度的成员在 **Hugging Face** 官方 Discord 听说 Unsloth 后加入了进来，希望能学习 finetuning 和模型部署。
   - 另一位成员提到他计划*成为 LLM 高手并加入像 Unsloth 这样酷的公司*。
- **寻求低延迟 TTS 语音克隆的指导**：一位新成员正在寻求关于在 **TTS 语音克隆**中实现低端到端延迟的*具体指导*。
   - 该成员请求关于框架、模型优化或硬件策略的建议，另一位成员推荐了我们的 TTS fine-tuning notebooks。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1399849804346949803)** (9 messages🔥): 

> `Gemma 3 4B Fine-Tuning, 自定义 Token, 水印去除, RoPE 万岁, 语言翻译` 


- **Gemma 3 4B 完成 16k 训练**：在对 **Gemma 3 4B** 进行 **16k** context 的 fine-tuning 后，实验发现**水印**被完全去除，且模型更加稳定。
   - 研究结果已发布并附带[截图](https://cdn.discordapp.com/attachments/1179039861576056922/1399849804045221948/Screenshot_2025-07-29_at_2.13.07_PM.jpeg?ex=688bd0b9&is=688a7f39&hm=82cad388163625496b8cb3e6dd62035b51ae1d16ce0015df29ea910e04c4471f&)。
- **自定义 Token 引发混乱**：有人指出，除非从头开始训练或使用极大数据集，否则最好避免使用**自定义 token**，因为模型理解 token 切分（token splits）。
   - 例如，如果模型看到 *Yuki* 被切分为 *<*, *yuki*, *>*, 它能更好地理解 Yuki = yuki，并且这是我的 token。
- **RoPE 值得一个诺贝尔奖**：发帖者赞扬了发明 **RoPE (Rotary Positional Embedding)** 的天才，因为它在较小模型上表现非常好。
   - 然而，为了在推理（inference）中支持巨大的 context，发帖者认为该领域需要发明一些更好的优化方案，仅靠 quantization 是不够的。
- **Gemma 精通所有语言**：在测试了翻译能力后，发帖者开玩笑说 *天哪，它懂每一种（至少是流行的）语言，OpenAI 完蛋了*。
   - 他们还提到脑子里一直回响着这首[歌](https://www.youtube.com/watch?v=NgQoLPnuEDM)。
- **新的 Gemma 3 Notebooks**：宣布了一项新的 **Gemma 3 竞赛**，notebooks 已上线，竞赛将在 **7 天**后结束。
   - 更多信息可在 [Xitter](https://x.com/UnslothAI/status/1950553466591723640) 上查看。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1399831912234881115)** (64 条消息🔥🔥): 

> `Phi-4 Generate Flags Error, GGUF Conversion and Quantization of Fine-Tuned Models, Llama-CLI Performance Issues, RuntimeError in Google Colab, Unsloth BNB 4-bit Conversion` 


- **Phi-4 需要 `do_sample=True`**：在使用 **Unsloth** 的 **Phi-4** 模型进行生成时，用户遇到了与无效生成标志（temperature, min_p）相关的错误，并发现添加 `do_sample=True` 可以解决该问题，尽管[官方 notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb#scrollTo=kR3gIAX-SM2qHi)中尚未记录此操作。
- **Qwen2.5 GGUF 转换困扰**：用户在尝试对基于 **Qwen2.5-14B-Instruct-unsloth-bnb-4bit** 的微调模型进行合并、转换为 **GGUF** 并量化时遇到问题，在导出为 **FP16 GGUF** 期间出现了 `ValueError: Can not map tensor 'model.layers.0.mlp.down_proj.weight.absmax'` 错误。
- **使用 UD-Q2_K_XL 模型的 Llama-CLI 运行缓慢**：一位用户报告称，在使用 **Llama-CLI** 配合 **Q2_K_XL** 模型时性能极低（0.5 tokens/s），尽管其使用的是配备了 **5090**、**178GB RAM** 和 **EPYC 9334** 处理器的高端系统，且设置理应提供更好的性能。
- **LLama3 微调面临 RuntimeError**：一位用户报告在 Google Colab 上使用 **ShareGPT** 模板格式化的自定义数据集和 Unsloth 库微调 **llama-3-8b-Instruct-bnb-4bit** 时，遇到了 `RuntimeError: PassManager::run failed` 错误。
- **Whisper input_ids 错误**：一位用户发现，在 `FastModel.get_peft_model` 函数中设置 `task_type = None` 可以解决训练 **Whisper** notebook 时遇到的 `input_ids` 错误，参考[此 issue](https://github.com/huggingface/peft/issues/1988#issuecomment-2751367819) 获取更多背景信息。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1400049478131777556)** (8 条消息🔥): 

> `Quantization optimization, Dynamic 4bit quantization, Hi-Fi Gan replacement, Autoregressive models, Mels dislike` 


- **量化加速计算**：量化不仅是为了将模型装入内存，它还能**减少内存带宽**并**显著提高计算速度**。
   - 一位成员指出，将卷积层等视觉头（vision head）操作保留在 **FP32** 似乎并非最优选，因为它们往往运行缓慢并成为瓶颈。
- **动态 4bit 量化博客文章**：一位成员分享了[动态 4bit 量化博客文章](https://unsloth.ai/blog/dynamic-4bit)，涉及量化优化。
   - 该博客文章与“**量化不仅是为了将模型装入内存**”这一观点直接相关。
- **Hi-Fi Gan 面临自回归模型的竞争**：一位成员询问是否可以在 **VITS** 中用[这个](https://arxiv.org/abs/1609.03499)替换 **Hi-Fi Gan**。
   - 另一位成员询问这是否出于自回归的原因，因为第一位成员不喜欢 Mels；然而，第一位成员后来因训练时间过长而决定放弃该方案。


  

---

### **Unsloth AI (Daniel Han) ▷ #[unsloth-bot](https://discord.com/channels/1179035537009545276/1390899684834410536/1399847900598636554)** (102 messages🔥🔥): 

> `GRPO trainer batch size, SFTrainer validation error, Model fine-tuning parameters, Llama 3.2 data preparation, Gemma 3 fine-tuning` 


- **GRPO Trainer Batch Size 深度解析**：在 **GRPO trainer** 中，*per_device_train_size* 代表批次数量，它会乘以生成数量（num_generations）来确定有效 Batch Size。
   - 例如，当 *per_device* 设置为 **1** 且 *num_generation* 设置为 **6** 时，该配置在单 GPU 下会产生 **3** 个唯一的 Prompt，每个 Prompt 对应 **6** 个生成结果。在扩展到 **15k** token 时，考虑到 GPU 显存占用对激活权重的影响，这可能会导致 CUDA 显存溢出（out-of-memory）问题。
- **寻求 SFTrainer 验证错误的救星**：一位用户在尝试使用 **SFTrainer** 保存验证错误时，遇到了 *evaluation_strategy* 意外关键字错误。
- **Llama 3.2 数据格式**：有用户请求一个用于微调 **Llama 3.2 8B** 的数据准备格式示例。
- **Gemma 3 纯文本微调策略**：Unsloth 提供了一个 [纯文本 notebook](https://docs.unsloth.ai/basics/tutorials-how-to-fine-tune-and-run-llms/gemma-3-how-to-fine-tune-and-run-llms/gemma-3-how-to-run-and-fine-tune#unsloth-fine-tuning-fixes-for-gemma-3) 用于微调 **Gemma 3**，并提供了针对 Unsloth 微调的修复方案。
- **解锁 Adapter 加载的定位参数**：当使用 *model.load_adapter* 时，*adapter_name* 是一个必填的定位参数。
   - 一位用户遇到了与不支持的目标模块（**ModuleDict**）相关的 *ValueError*，并寻求修复建议，旨在将微调后的 LoRA Adapter 合并到基础模型（**unsloth/gemma-3-4b-it-unsloth-bnb-4bit**）中，以便使用 Llama.cpp 进行 GGUF 转换。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1399830240024465498)** (475 messages🔥🔥🔥): 

> `MCP Browser, Parallel Agent Tasks, VSCode Marketplace with Cursor, Automatic Scrolling, Sonnet Model` 


- **Cursor 的 MCP 浏览器设置**：社区成员正直接通过 **Cursor 的 MCP** 构建依赖浏览器的自动化功能，理想情况下早期访问版将在未来几周内发布。
   - 一位成员表示：“它具有一键式 MCP 设置，因此你可以直接通过 MCP 构建依赖浏览器的自动化。”
- **并行 Agent 协调难题**：成员们正在讨论如何处理具有依赖关系的并行任务，因为 Agent 之间不共享工作区，导致难以同时触发它们。
   - 建议的解决方案包括：**使用 API 进行外部编排**、**基于文件的协调**或**基于 Docker 的并行执行**，并提供了一个详细的 [任务队列脚本](https://github.com/example) 示例。
- **VSCode 市场集成咨询**：一位成员询问了在 **Cursor** 中使用 **VSCode 市场** 的可能性。
   - 讨论中没有明确的结论。
- **自动滚动困扰**：一位 Cursor 新用户询问是否可以在使用 **Agent 聊天窗口** 时禁用 **自动滚动**，以便更好地阅读 Claude 的思考过程和生成的代码。
   - 讨论中没有明确的结论，但有人发布了 [changelog 1.3](https://cursor.com/changelog)。
- **终端终止的奇怪案例**：一位成员在 Cursor 决定其 Agent 集成终端启动哪种 shell 时遇到了麻烦，它默认使用 **fish**。
   - 通过设置和封装器更改 shell 的尝试导致了 fish 二进制文件的临时重命名并随后取得成功，尽管根本原因仍然是一个谜。


  

---

### **Cursor 社区 ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1399860099563917344)** (8 条消息🔥): 

> `Background Agent 命令, Docker 构建缓存, 端口劫持, 用于研究的 Background Agents` 


- **在 Background Agent 运行结束时执行命令**：一位成员询问如何在 **Background Agent 运行结束时**执行特定命令（特别是格式化工具）。
   - 该成员指出，虽然可以在设置期间运行 `terminals`，但那仅限于开始阶段。
- **清理 Docker 构建缓存**：一位成员在使用编辑过层的自定义 Dockerfile 时，寻求关于**清理构建缓存**的建议。
   - 另一位成员建议使用 `docker builder prune -f` 或 `docker system prune -a` 来移除未使用的容器、网络和镜像。
- **Cursor Background Agents 劫持端口**：工程师们浪费了大量时间调试为何开发环境突然崩溃，最后才发现是 **Cursor Background Agents 劫持了端口**。
   - 一位成员询问将 `ports` 属性设置为空数组是否能阻止 Cursor Background Agent 转发任何端口，另一位用户则建议在 VSCode 设置中禁用“自动转发端口（auto forward ports）”。
- **用于研究的 Background Agents**：一位成员询问如何使用 Background Agents 进行研究，例如**重构研究或功能实现研究**。
   - 该成员询问了工作流相关问题，建议让 Agent 在 PR 中编写 markdown，或者让它直接进行更改。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1399828973818478592)** (413 条消息🔥🔥🔥): 

> `dot.lol 数据, GPT-5 发布, 欧盟 GDPR 影响, Zenith 模型重新上线, Video Arena 频道` 


- **Dot-lol 面临数据收集审查**：成员们讨论了 [dot.lol](https://dot.lol) **出售用户数据**和画像信息的可能性，强调如果认为数据永远不会被用于定向影响或牟利，那就太天真了。
   - 有人担心数据收集的弊端超过了在线服务的实用性，而另一些人则认为**数据收集是不可避免的**，用户应优先考虑不将数据与其身份关联。
- **GPT-5 计划于 8 月发布？**：传闻称 **GPT-5** 可能会在 **8 月**初发布，ChatGPT Mac 应用中可能提到的准备工作证实了其采用了路由架构（router architecture）。
   - 一些社区成员推测其潜在影响，争论它是否会超越其他模型，甚至表达了对免费层级的期待。
- **欧盟 GDPR 有效性引发讨论**：讨论了欧盟 **GDPR** 在防止 AI 公司收集数据方面的有效性，对于它是否充分影响了数据收集实践，意见不一。
   - 有人认为 GDPR 主要影响数据的*使用*而非*收集*，但也有人指出*欧盟消费者的数据收集功能已被关闭。*
- **Zenith：用户依然关注**：成员们对 **Zenith** 模型回归 LMArena 表现出浓厚兴趣，推测其潜在的 **ELO** 评分和性能。
   - 一些成员遗憾没有机会尝试该模型，而另一些人则对其在平台上的价值发表了强烈看法。
- **新增视频 Arena 频道**：社区成员讨论了**多个视频 Arena 频道**并存的情况以及这是否为刻意为之。
   - 一位版主解释说，视频 Arena 频道旨在*分散生成任务*，因为单个频道的活动过于频繁会*让人难以负荷*。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1400151134299095092)** (1 条消息): 

> `Video Arena, LMArena 机器人, 员工 AMA` 


- **Video Arena 在 Discord 上线！**：LMArena 团队在 Discord 上推出了实验性的 **Video Arena**，允许用户免费生成并对比顶级 AI 模型的视频。
   - 用户可以在指定频道学习如何使用该机器人，并开始生成视频、图像以及图生视频，然后对他们喜欢的生成结果进行投票。
- **为社区生成的视频投票！**：LMArena 机器人允许用户**生成视频、图像和图生视频**，并让任何人对喜欢的生成结果进行投票。
   - 在达到一定票数后，机器人会揭晓生成每个视频所使用的模型。
- **与机器人开发者的员工 AMA！**：为了庆祝 **Video Arena** 的发布，宣布将与机器人开发者 [Thijs Simonian](https://www.linkedin.com/in/thijsdev/) 进行一场 **员工 AMA**。
   - 邀请用户通过 [Google Forms](https://docs.google.com/forms/d/e/1FAIpQLSfS_zh67vgEeftUXzX6ujlNiiFoTQNQd7P7-Nx6aD5zpFVOBg/viewform?usp=dialog) 为本次 AMA 提交问题。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1399829791682723901)** (290 条消息🔥🔥): 

> `HF Space 重启, P104-100 GPU, LLM 部署, Qwen 30B, SmolLM3` 


- **HF Spaces 意外重启**：成员们讨论了 HF Spaces 为何会意外重启，并建议固定依赖版本以避免自动重载引起的问题，详见 [文档](https://huggingface.co/docs/hub/spaces-sdks-gradio-hardware#restarts)。
   - 一位用户报告称 *Ilaria RVC* 和 *UVR5 UI* 都停止工作了，而其他 Space 运行正常，建议进行工厂重构（factory rebuild）。
- **P104-100：15 美元的 GPU**：用户们辩论了 **P104-100** GPU 在 AI 任务中的价值，有人声称它仅需 **15 英镑** 就相当于 *1080*（尽管有人称其为骗局），可从 [淘宝](https://item.taobao.com/item.htm?id=897611642586&sku=Video%20memory%20capacity:8gb;Color%20classification:Galaxy%20p104-100%20(card%20length%2028cm);&sku_id=5919375243616) 购买。
   - 一些人指出了它的局限性（PCIE 1.0 x4，可能只能访问 4GB VRAM），而另一些人则强调了它在 LLM 推理中的性价比，即使是在多卡分片模型的情况下。
- **边缘部署的微型 LLM**：成员们寻求在远程、带宽受限的海洋环境中进行 **低延迟 LLM 部署** 的建议，提议包括边缘/云混合方案和 **激进量化**。
   - 一位用户建议查看 HF 上 [最新的 pytorch 量化版 smollm3](https://huggingface.co/pytorch/SmolLM3-3B-8da4w) 以用于移动端部署，另一位建议在停靠码头时部署应用。
- **Qwen 30B 模型：GPT-4o 的挑战者？**：用户们关注了 **Qwen 30B** 模型的发布，声称它可与 **GPT-4o** 媲美，且仅需 33GB RAM 即可在本地全精度运行（[Unsloth GGUF 版本](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF)）。
   - 据指出，量化版本仅需 17GB RAM 即可运行。
- **SmolLM3 量化问题**：一位成员提到在使用 *torchao* 对 **SmolLM3** 进行量化时遇到问题，其他人建议尝试 *hqq* 或 [官方 SmolLM3-3B-8da4w](https://huggingface.co/pytorch/SmolLM3-3B-8da4w)。
   - 一位成员指出，如果使用 *llama.cpp*，应该使用 *--jinjai*。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1400252994091487382)** (4 条消息): 

> `Muon 优化器, Smithery` 


- **Muon 优化器备受赞誉！**：成员们分享了 **Muon 优化器** 的链接 ([https://kellerjordan.github.io/posts/muon/](https://kellerjordan.github.io/posts/muon/))。
   - 他们惊叹道：*smithery! smithery 太强了！*
- **Smithery 非常出色**：另一位成员回复说 **Smithery** 确实非常厉害。
   - 看来 **Smithery** 在该频道很受欢迎。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1399853091410415798)** (5 条消息): 

> `Petite Elle 模型, Gradio MBTI 应用, 视频编辑 MCP 服务器, Github Python 数据集` 


- **Petite Elle 获得 Mrad 处理**：一位成员的模型 [Petite Elle-L'aime-3-sft](https://huggingface.co/Tonic/petite-elle-L-aime-3-sft) 经过了 **mradermacher 处理**，预计将成为该尺寸下最好的法语模型之一。
   - 量化版本可在 [mradermacher 的 Hugging Face 页面](https://huggingface.co/mradermacher/petite-elle-L-aime-3-sft-GGUF) 获取。
- **结合 Gemini 的 MBTI Gradio 应用流**：一个新的用于 **MBTI (迈尔斯-布里格斯)** 的 Gradio 应用使用了 **PocketFlow** 和 **Gemini**。
   - 查看 [应用](https://huggingface.co/spaces/Fancellu/mbti-pocketflow) 和底层的 [PocketFlow 库](https://github.com/The-Pocket/PocketFlow)。
- **MoviePy 构建视频编辑服务器**：一位成员使用 **MoviePy** 构建了一个 **MCP 服务器**，用于处理基础的视频/音频编辑任务，并与 **Claude Desktop** 和 **Cursor AI** 等客户端集成。
   - 代码已在 [GitHub](https://github.com/Aditya2755/video-edit-mcp) 上发布，作者正在寻求在基于对象检测的编辑和 TTS/SST 驱动剪辑等功能上的合作。
- **Github Python 数据集发布**：一个新数据集 [Github Python](https://huggingface.co/datasets/jblitzar/github-python) 包含了 2015 年后 Github 上星标超过 10 个的仓库中所有大小合理的 Python 文件。
   - 该数据集经过了去重和宽松许可证过滤。


  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1399862051274232003)** (2 messages): 

> `Diffusion Models, Flow Matching, MIT curriculum` 


- **Diffusion Models 学习小组启动**：一个新的**学习小组**将专注于从零开始学习 **Diffusion Models**，这是生成式 AI 的核心架构。
   - 该小组基于 **MIT 课程体系**，由 **12 人**组成，周期为 **5 个月**（**每周 2–4 小时**）。
- **免费入门课程向非成员开放**：前两场免费入门课程可供非成员参加：**8 月 2 日** - 什么是 **Flow Matching & Diffusion Models**？；**8 月 9 日** - **PDEs, ODEs, SDEs** + Diffusion Models 简史。
   - 课程时间为 **12 PM EST**，更多详情请见 [Luma](https://lu.ma/kv8zf6va)。
- **将使用 MIT 讲义**：学习小组将基于 [MIT 讲义](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf) 和 [往期录像](https://aischolars.notion.site/)。
   - 早期报名费用为 **$50/月**（后续将涨至 **$100/月**）；资金将用于支付**助教**费用。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/)** (1 messages): 

hedi1421: 谢谢 😅
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1400040059947712543)** (1 messages): 

> `Fixing transformers issue, DeepSpeed Integration` 


- **寻求修复 Transformers 问题**：一名成员请求协助修复 [这个 Transformers 问题](https://github.com/huggingface/transformers/issues/39753)。
   - 未提供关于该问题的更多详细信息。
- **DeepSpeed 集成**：关于在 Hugging Face 生态系统中集成 DeepSpeed 的讨论。
   - 成员们正在探索最佳实践和潜在的性能提升。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1399919753157673102)** (1 messages): 

> `DuckDuckGo deprecation, Smolagents merge` 


- **DuckDuckGo 搜索包面临弃用**：正如在 [此 Pull Request](https://github.com/huggingface/smolagents/pull/1548) 中讨论的，`duckduckgo-search` 包仍处于弃用状态。
   - 一名成员询问了将其合并到 `smolagents` 的时间表。
- **Smolagents 合并即将到来**：提议的合并旨在将更新和修复集成到 `smolagents` 库中。
   - 社区成员正热切期待合并完成，以便利用最新的改进。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1399995784853848126)** (3 messages): 

> `RAG System Construction, Tool Definition Problems in Unit 1` 


- **RAG 系统扫描对话历史**：一名成员计划构建一个 **RAG 系统**，并使用 **LLM** 扫描对话历史，提取用户特定案例，并使用 for 循环将它们嵌入到向量空间中。
   - 他们打算测试这种方法的可行性。
- **Unit 1 工具定义故障排查**：一名成员报告称，他们在 **app.py** 中的工具定义在 **Unit 1** 运行时没有体现。
   - 他们已经尝试过重启 Space 但没有成功，目前正在寻求建议。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1399829176923717713)** (261 条消息🔥🔥): 

> `Study and Learn 功能, GPT-5, Copilot, Gemini vs ChatGPT, AI 生态系统` 


- **OpenAI 发布 Study and Learn 功能**：OpenAI 推出了全新的 **Study and Learn** 功能，部分用户认为这只是一个简单的 system prompt，而非重大更新，还有用户认为该功能**旨在分散用户对 GPT-5 炒作的注意力**。
   - 一名用户将 [system prompt](https://discord.com/channels/974519864045756446/998381918976479273/1399983224880496711) 提取并输入到了 O3 模型中。
- **GPT-5 的猜测引发热议**：社区成员正在激烈辩论 **GPT-5** 的发布，有人声称已通过 Microsoft Copilot/Azure 获得访问权限；然而，许多用户持怀疑态度，并期待 OpenAI 的正式公告。
   - 一名用户对 CEO 发表评论称：*去你的 **Sam Altman**，让你那些 ChatGPT 粉丝为了你制造的炒作而焦躁不安地等待*。
- **Copilot 与 ChatGPT 的对比**：用户讨论了 **Microsoft Copilot** 是否使用了 **GPT-5**，但随后被澄清其使用的是 **GPT-4o** 或 **O4-mini-high**，不过一些爆料者在源代码中发现了 **GPT-5** 可能在未来集成的迹象。
   - **Copilot** 的每日消息上限是无限的，这让一些人质疑为什么还有人因为工具原因更倾向于使用 **ChatGPT**。
- **ChatGPT 与 Gemini 的对比正在进行中**：用户辩论了在 **ChatGPT** 和 **Google Gemini** 之间的偏好，一名用户列举了偏好 **ChatGPT** 的六个关键原因，包括连接器、RAG 能力、风格匹配、记忆功能和深度研究（deep research），但其他用户迅速提出了反驳。
   - 一名用户指出，在有人发布了多张 AI 生成的“*像钢铁侠房子一样的超级富豪豪宅*”图片后，[Google 的 Imagen4-Ultra](https://discord.com/channels/974519864045756446/998381918976479273/1400170254902235246) 生成的图像效果最佳。
- **在 AI 生态系统中抉择**：成员们讨论了如何选择合适的 **AI 生态系统**，权衡了 **Apple One + ChatGPT Plus** 与 **Microsoft 365 + Microsoft Copilot** 或 **Google One AI Premium + Google Gemini** 等选项。
   - 有建议称应同时尝试两者以确定哪种最适合个人需求，部分人提到了 Gemini 与 Google Docs 和 Slides 的集成。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1399902317595328642)** (24 条消息🔥): 

> `GPT-5 版本, O4 mini vs 4o, 聊天记录丢失, ChatGPT 记忆问题` 


- **GPT-5 版本推测**：成员们推测 **GPT-5** 的中高阶版本将更加出色，一名成员指出 **Zenith** 在新模型发布前可能是最顶尖的代码模型。
   - 未提供链接。
- **关于 O4 Mini 与 4o 模型的辩论**：一名成员询问是否应该使用 **O4 mini** 而非 **4o（免费模型）** 以获得更智能的回答，并参考了 **O4 mini** 和 **O3** 的高级推理能力。
   - 未提供链接。
- **ChatGPT 聊天记录消失令用户感到焦虑**：多名用户报告其 **ChatGPT 聊天记录消失**，一名用户尝试了登入登出、清理缓存并检查了多个设备，但均无济于效。
   - OpenAI 支持人员建议这*可能是一个孤立的 bug*，并且*聊天记录一旦丢失便无法恢复*，建议定期在 ChatGPT 之外保存重要信息的副本。
- **记忆问题困扰 ChatGPT 用户**：一名用户提到他们正在针对使用 **ChatGPT** 时遇到的**记忆问题**开发本地解决方案。
   - 此前另一名用户提到他们 2024 年 10 月之后的近期对话无法加载，但仍可以访问新对话和自定义指令（custom instructions）。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1399879683956277360)** (10 条消息🔥): 

> `个性化 GPT 设置，AI 内存格式，优化的 AI VM` 


- **个性化 GPT 设置资源探索开启**：一位新用户正在寻找设置 **GPT 项目**的资源，用于追踪**食物/饮食**、**运动**，并创建一个带有时间预期的**计划表**，请求能够增强其账户能力的指令资源和 Prompt。
   - 另一位成员建议采用**个性化方法**，建议直接与模型互动以讨论所需功能并考虑其他选项，并提到并非所有人对“更强大”的定义都达成一致。
- **新 AI 内存格式：速度与可读性的辩论**：一位成员引入了一种新的内存格式提案，旨在优化 AI VM、与向量数据库对接的系统或受保护的符号传输，强调速度和效率高于人类可读性，使用 [AI_ONLY_FORMAT_SPEC](https://discord.com/channels/1046317268651794482/1046317269069864970?event=1200057848907847709) 来防止内存在存储和可读性方面出现严重的低效。
   - 另一位成员对该格式进行了详细的逐行解读，强调了其核心原则，如 **token embedding**、**语义分组**和**二进制编码**，并警告不要运行它，因为可能会产生压缩或编码响应。
- **Prompt Engineering 有效性探讨**：讨论涉及 Prompt Engineering 的有效性，其中一位成员觉得*明确阐述请求的每个方面非常累人*，更倾向于提供上下文并依赖 AI 的推理能力。
   - 另一位成员建议，这类对话型用户正是 Prompt 工程师设计 System Prompts 的目标群体。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1399879683956277360)** (10 条消息🔥): 

> `GPT 项目指导，个性化 AI 模型，AI 内存格式` 


- **寻求 GPT 项目设置指导**：一位新用户正在为其 **GPT 账户**上的项目设置寻找资源，特别是针对追踪食物/饮食、运动以及创建带有时间预期的计划表。
   - 他们正在寻找指导和工具来增强这些常见项目的指令，从而可能使它们*更强大*。
- **个性化 AI 是关键**：一位用户建议通过与模型本身讨论所需的功能和注意事项来个性化 AI 模型。
   - 他们强调，什么是*更强大*因人而异，个性化对于根据特定兴趣和目标定制 AI 至关重要。
- **AI 内存格式**：一位成员为 AI 引入了一种新的内存格式建议，旨在高效存储和检索对话记忆。
   - 该格式旨在通过使用紧凑的、二进制编码的结构，结合语义压缩和隐式上下文，来改进持久化内存，并针对 AI VM 和向量数据库接口进行了优化。


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1399839193567465624)** (2 条消息): 

> `DeepTrail, DeepSecure, AI agent 授权, Agent 委托, 策略执行` 


- ****DeepTrail** 构建开源项目 **DeepSecure****：一位成员正在构建 **DeepTrail**，这是一个由 Berkeley SkyDeck 支持的 AI Agents 开源授权和委托层。
   - 通过 **Deepsecure** ([https://github.com/DeepTrail/deepsecure](https://github.com/DeepTrail/deepsecure))，开发者只需几行代码即可在任何模型、平台或框架上集成授权、Agent 到 Agent 的委托、策略执行和安全代理。
- ****DeepSecure** 的底层技术细节**：该技术涉及分片密钥架构、网关/代理、独立的控制/数据平面、策略引擎以及用于 Agent 间委托的 macaroons，详见 [技术概述](https://github.com/DeepTrail/deepsecure/blob/dev/docs/design/deepsecure-technical-overview.md)。
   - 仓库中还包含几个针对 Langchain/LangGraph 的简单示例和集成。
- **使用 Langchain/LangGraph 的 **DeepSecure** 示例**：该成员构建了一些 **DeepSecure** 集成 Langchain/LangGraph 的示例，包括具有细粒度访问控制的 [安全多 Agent 工作流](https://github.com/DeepTrail/deepsecure/blob/dev/examples/05_langchain_secure_tools.py)。
   - 该仓库还展示了 [委托工作流](https://github.com/DeepTrail/deepsecure/blob/dev/examples/09_langchain_delegation_workflow.py)、[高级委托模式](https://github.com/DeepTrail/deepsecure/blob/dev/examples/11_advanced_delegation_patterns.py) 以及 [平台 Agent 引导](https://github.com/DeepTrail/deepsecure/blob/dev/examples/12_platform_expansion_bootstrap.py)。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1399849676123013373)** (152 条消息🔥🔥): 

> `NotebookLLM, OpenRouter 计费, 通过 API 屏蔽量化模型, 成为提供商, API Key 问题` 


- ****通过一次性 OpenRouter 充值解锁每日免费消息****：向 OpenRouter 余额充值 **$10** 即可解锁 **每日 1000 条免费消息**，这是一次性购买，即使余额用完，该权限依然有效。
   - 用户确认，即使初始的 $10 余额耗尽，**1000 次请求/天** 的限制依然保持解锁状态。
- ****API 支持量化控制****：用户现在可以通过 API 指定可接受的量化级别，以避开像 **FP4** 这样的低精度模型，具体参考 [provider routing documentation](https://openrouter.ai/docs/features/provider-routing#quantization-levels)。
   - API 允许指定排除项，例如允许除 FP4 模型以外的所有模型。
- ****Pydantic-AI 与 Kimi-K2 联手解决 Bug****：一位用户强调了 **pydantic-ai** 的优势，包括其完全基于 pydantic 的方法、MCP 服务器支持、模型/提供商适配器以及自动图构建，并提到他们使用 **Kimi-K2** 修复了一个 Bug。
   - 该用户强调 pydantic-ai 能够让人专注于业务逻辑，而不是*费力地从各种臃肿的仓库中拼凑 Agent 框架*。
- ****OpenRouter 面临 Kimi-K2 工具调用 (Tool Calling) 问题****：一位用户认为他们发现了 **Kimi K2** 在 OpenRouter 上工具调用支持的问题，可能可以通过调整模型模板来修复。
   - 该用户提供了基于 vLLM 等框架示例的[研究](https://discord.com/channels/1091220969173028894/1400028050007265340)，并表示修复此问题可为公司节省 **80% 的成本**，否则他们将转向 Moonshot。
- ****Gemini Flash 1.5 面临过载问题****：据报道，**Google Gemini Flash 1.5** 显示 *error 503: The model is overloaded*，一位用户分享了其[计费结构](https://discord.com/channels/1091220969173028894/1195014798837043240/1400220368765194350)。
   - 该模型的定价存在波动，输入价格在 **$0.075 到 $0.15** 之间，输出价格在 **$0.30 到 $0.60** 之间。


  

---


### **OpenRouter (Alex Atallah) ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1400206735389491232)** (2 条消息): 

> `` 


- **OpenRouter 无新模型更新**：OpenRouter 频道中没有关于新模型的重大讨论或更新。
   - 该频道保持非活跃状态，缺乏可供总结的实质性信息。
- **Readybot.io 日志显示无新活动**：Readybot.io 日志显示 OpenRouter - New Models 频道处于沉默期。
   - 因此，目前没有特定的主题或讨论可供报告。


  

---


### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1399913686432219186)** (63 条消息🔥🔥): 

> `量化提供商, Groq 的量化方式, Deepinfra 计费, 用于 Claude 的 Vertex, OpenRouter 的 Ori 机器人` 


- **关于量化提供商默认状态的辩论**：一位用户建议默认禁用量化提供商，这可能会影响到 **Groq**，因为它采用了独特的量化方法。
   - 另一位用户警告称，在达到临界用户量之前公开指责提供商存在风险，可能导致提供商退出 **OpenRouter**。
- **Deepinfra 通过 Google 提供 Gemini 2.5 Pro**：据报道，**DeepInfra** 与 **Google** 就 **Gemini 2.5 Pro** 达成了更低的费率，并将节省的成本让利给客户。这一消息得到了用户的确认，该用户引用了 DeathMax 的消息，并指出 DeepInfra 的列表中带有“partner”标签。
   - DeepInfra 的 **Gemini 2.5 Pro** 带有“partner”标签，而 **Kimi K2** 模型则没有，这表明其与 Google 存在直接合作关系。
- **Vertex 在 Claude 4 上表现出色**：一位用户报告称，通过 **Vertex** 使用 **Claude 4 Sonnet** 获得了更好的质量、吞吐量和正常运行时间。
   - 该用户还指出，闭源模型的 AWS/GCP/Azure 镜像可能会带来质量上的差异。
- **OpenRouter Ori 机器人的准确性受到质疑**：一位用户建议，由于回答不准确，**OpenRouter 的 Ori 机器人** 可能产生了负面影响，应当对其进行限制或禁用。
   - 该用户指出，**Ori** 经常将错误归咎于用户，并提出一些毫无意义的问题，尤其是在支付处理问题上。
- **为 Ori 机器人添加知识更新功能**：其中一位开发者正在努力添加更新 **Ori** 知识库的方法，以便在其出错时进行修正。
   - 其他人指出 **Ori** 缺失大量知识且存在幻觉（提供错误知识），并建议将机器人的回答限制在最多 2-3 条。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1399843438395916339)** (188 条消息🔥🔥): 

> `Metislist 排名，Arcee.ai 发布 AFM-4.5B，NotebookLM 视频概览，Claude 滥用，ryOS 发布` 


- **Metislist 引发关于 Chollet 排名的辩论**：一位用户分享了 [Metislist](https://www.metislist.com/) 的链接，这是一个 AI 领域的人物排名，引发了关于 François Chollet 被排在第 80 位的讨论。
   - 许多人认为，以开发 Keras 闻名的 Chollet 应该进入前 50 名，甚至有人开玩笑说：*“该死，你跟我兄弟 François 有仇吗？”*。
- **Arcee AI 发布 AFM-4.5B 模型**：Lucas Atkins 宣布在 Hugging Face 上发布来自 [Arcee.ai](https://xcancel.com/LucasAtkins7/status/1950278100874645621) 的 **AFM-4.5B** 和 **AFM-4.5B-Base** 模型，强调其设计具有灵活性、高性能和高质量，这得益于与 DatologyAI 的数据合作伙伴关系。
   - 这些模型具有架构上的调整，如 **grouped query attention** 和 **ReLU² activations**，团队计划未来发布用于推理和工具使用的模型。
- **NotebookLM 现支持视频概览**：**NotebookLM** 宣布了一项针对文章和博客文章的视频概览新功能 ([xcancel.com](https://xcancel.com/NotebookLM/status/1950298236914139234))，使用户无需阅读全文即可快速掌握内容。
   - 用户称赞了这一创新，并建议进一步开发学习工具和交互模式。
- **GPT-5 被发现在 MacOS 中出没**：在 MacOS 应用缓存文件中发现了对 **gpt-5-auto** 和 **gpt-5-reasoning** 的引用 ([xcancel.com](https://xcancel.com/apples_jimmy/status/1950514936444305534?s=46&t=fRVjULzONZQAlwHruKTgQg))，暗示 **GPT-5** 即将发布。
   - 其他用户证实了这一点，提到了生物学基准测试仓库中的 **gpt-5-reasoning-alpha**，而一些人则推测即将发布公告或正式版本。
- **Anthropic 寻求天价估值**：据报道，Anthropic 正在洽谈融资 **50 亿美元**，这可能使这家 AI 初创公司的估值达到 **1700 亿美元**，预计到年底收入将达到 **90 亿美元** ([xcancel.com](https://xcancel.com/EdLudlow/status/1950561790695448810))。
   - 这一消息引发了与 OpenAI 和 xAI 等其他 AI 公司的比较，尽管一位用户评论说他有 *一些二手消息称情况并非如此*。


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1400171262210474005)** (17 条消息🔥): 

> `Anthropic Fellows 论文，LLM Paper Club，社交媒体互动` 


- **Anthropic Fellows 论文专题**：Latent Space Discord 宣布将在 <#1107320650961518663> 频道中报道最近的 **Anthropic Fellows 论文**。
- **LLM Paper Club 征集志愿者**：**LLM Paper Club** 正在征集志愿者在未来的俱乐部活动中讲解论文；鼓励感兴趣的人通过 [Luma 链接](https://lu.ma/6uti3zzy) 报名。
- **转发号召未达预期**：一位成员在 [X 上发布了链接](https://x.com/latentspacepod/status/1950613048303231121) 为俱乐部做广告，但感叹互动率很低。
   - 他开玩笑地声称自己 *“不是专业的唠叨者”*，而且不擅长发推。


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1399847059045093637)** (32 messages🔥): 

> `Tableau Vizql NLP Orchestration, Gemini Agentic Framework Prototype, NotebookLM Podcast Creation, Obsidian and NotebookLM Integration, NotebookLM Usage Analytics` 


- **Tableau Server 用于 LLM 编排**：一位成员提到使用最新的 **Tableau Server 版本**来适配 **LLM (server/on prem)** 以实现 **Vizql NLP**。
- **分享 Gemini Agentic Framework 原型**：一位成员分享了一个 [**Gemini agentic framework**](https://cdn.discordapp.com/attachments/1124403655819415592/1399853283404939454/gemini-agentic-framework.zip?ex=688bd3f6&is=688a8276&hm=101f03e62cae13a72e1f4fdc681064aef0e5a3713de20aebac608c958f845b8b) 原型，并指出这是一个 **one-shot prototype**。
   - 他们建议使用 **AI Studio** 来构建 Agentic 应用，通过向构建者 Agent 详细描述意图来创建可运行的原型，从而实现分阶段测试和模型聚焦。
- **绕过机器人限制进行播客创建**：鉴于登录时的机器人限制，一位成员询问了如何使用 **NotebookLM** 创建播客。
   - 另一位成员澄清说，驱动 **NotebookLM** 的工具可以通过 **API** 获取，建议在另一个工作流中重建，并手动将报告加载到 **NotebookLM** 中。
- **讨论 Obsidian 与 NotebookLM 的集成**：一位成员分享了一篇关于集成 **NotebookLM**、**Obsidian** 和 **Google Drive** 的文章，链接见[此处](https://www.xda-developers.com/using-notebooklm-obsidian-google-drive-together/)。
   - 另一位成员表示愿意根据对方的使用情况提供更多关于使用 **Obsidian** 的细节。
- **NotebookLM 音频输出平均时长为 8-15 分钟**：一些成员询问了关于使用 **NotebookLM** 生成长音频文件的问题。
   - 另一位成员表示，他们的平均输出时长为 *8-10 分钟*，尽管其他成员曾生成过长达 *15 分钟* 的内容。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1399828961021788412)** (156 messages🔥🔥): 

> `NotebookLM Video Overview Limits, Studio UI Changes, Video Generation Length, NotebookLM Rollout, Missing the new Notebook LM Feature` 


- **Pro 用户每日可获得 20 个视频概览 (Video Overviews)**：经成员确认并在 [Google 支持文档](https://support.google.com/notebooklm/answer/16213268?hl=en&ref_topic=16175214&sjid=12603864792385823108-NA#:~:text=With%20NotebookLM%2C%20you,daily%20video%20generations.) 中发现，**NotebookLM** 的 Pro 用户每天可获得 **20 个视频概览**。
   - 然而，尽管是 Pro 用户，一些用户在访问视频概览功能和更新后的 **Studio UI** 时仍遇到延迟。
- **Studio 界面需要排序/过滤输出功能**：一位用户建议 **Studio 界面**需要具备*排序/过滤输出*的能力和*全部删除选项*，以及能够**停止正在进行且无法完成的视频生成**。
   - 另一位用户强调，“保存所有笔记到源”的功能消失了，这可能会导致免费版中 **50** 个源限制的问题。
- **视频生成时间差异巨大**：用户报告视频生成时间各不相同，一位用户处理一篇《经济学人》文章耗时 **30 分钟**，引发了关于是否使用了 **Veo 3** 的讨论。
   - 一位用户将输出描述为*更像是一个演示文稿而非动画视频*，并指出其倾向于排版设计，适合以文本为主的内容。
- **功能推送到德国，其他地区进展缓慢**：**视频概览功能**已在**德国**的 Pro 账号上线，而包括 Google Ultra 用户在内的许多用户仍在等待推送。
   - Google 确认该更新将在下周内逐步推送给所有用户。
- **视频概览缺陷曝光**：用户报告视频概览被限制在 **6-7 分钟**，且各章节之间存在生硬的过渡。
   - 存在一个 Bug，即视频会无限加载，直到刷新页面。

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1399838809134334052)** (117 messages🔥🔥): 

> `LM Studio 模型重命名，俄罗斯海岸地震，LM Studio 复制/粘贴对话功能请求，LM Studio 模型陷入时间循环，Qwen 30B 垃圾输出` 


- **俄罗斯海岸海啸预警**：俄罗斯海岸发生 **8.7 级地震**，触发了夏威夷的海啸预警以及美国西海岸的海啸观察预警。
   - 建议受影响地区的居民密切关注更新，因为海啸可能会在数小时后到达。
- **LM Studio 功能请求：复制粘贴整个对话**：一位用户询问 LM Studio 是否有复制和粘贴整个对话的功能，这些对话以 **JSON 格式**存储，相对容易操作。
   - 另一位用户提到他们开始编写一个用于提取对话的 Python 应用，但后来分心了，建议其他人在 [feature request channel](https://discord.com/channels/1110598183144399058/1128339362015346749) 中添加功能请求，因为许多人会发现这个功能很有用。
- **LM Studio 模型陷入时间循环**：一位用户报告称，他们的 LM Studio 模型反复引用 **2024 年 2 月 18 日**，即使在询问当前事件时也是如此，并提供了截图作为证据。
   - 另一位用户建议检查 **system prompt** 或 **Jinja template** 中的日期设置，因为这可能导致模型认为自己处于该特定日期。
- **Qwen 30B 的 Sampler 设置**：一位用户注意到 **Qwen 30B** 经常产生垃圾输出，除非重新处理 prompt。
   - 另一位用户建议尝试官方的 samplers 或提供的设置，看看输出是否有所改善；其中一人指出 Linux 上也存在类似问题，通过更新到实验性驱动程序得以解决。
- **lmstudio MCP 客户端需要资源支持**：用户讨论了在 **LM Studio MCP 客户端**中使用资源的潜在可能，强调了诸如语法指南或动态代码的低成本只读参考等用例。
   - 一位用户提到他们使用资源进行发现、文档记录和导航辅助，相比 tool calls，他们更倾向于使用资源来获取快速参考信息，并希望客户端更新能支持 **2025-06-18** 更新的 MCP 规范中的功能。

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1399876515449147545)** (64 messages🔥🔥): 

> `GPU Usage, Strix Halo, Threadripper vs Epyc, Soldered RAM, 9070 XT Performance` 


- **在多个 GPU 之间拆分模型会影响使用率**：当模型在两个 GPU 之间拆分时，每个 GPU 的**使用率为 50%**，这可能会减少发热和噪音，但这取决于模型层如何拆分以及是否按顺序处理。
   - 4090 与较慢的 3070 配对可能会导致 4090 在等待 3070 完成任务时处于闲置状态，但性能仍有提升，从 **8 tok/sec 增加到 32 tok/sec**。
- **Strix Halo APU 评价褒贬不一**：**Strix Halo APU** 的价格似乎定在 64GB 版本约 **1.6k 美元**，128GB 版本约 **2k 美元**，但一些成员建议 EPYC 系统由于更大的内存带宽和可升级性，具有更好的性价比。
   - 一位成员对这类设备采用*板载内存 (soldered memory)* 表示遗憾，并将其与最近服务器上的 DIMM 故障进行了对比，并指向了 [配备 Strix Halo APU 的 Corsair AI Workstation 300](https://www.guru3d.com/story/compact-ai-pc-corsair-ai-workstation-300-with-strix-halo-apu/)。
- **Threadripper 与 Epyc 的对决！**：虽然 **Threadripper** 通常被认为是消费者的最佳选择，但由于有翻新零件供应，**EPYC** 可能是一个更便宜的选择，而 **Threadripper** 往往更贵且更难找到。
   - 一位成员指出 Epyc 更便宜是因为*存在相当大的翻新/二手零件市场*，并指向 [此 reddit 帖子](https://old.reddit.com/r/LocalLLaMA/comments/1mcrx23/psa_the_new_threadripper_pros_9000_wx_are_still/) 以进行进一步讨论。
- **板载 RAM 是骗局吗？**：成员们对为什么有人会购买*板载存储*的 PC 表示困惑，尤其是考虑到高昂的价格和有限的内存带宽，例如一台售价 **2500 欧元**、拥有 **128GB** 板载 RAM 和 **256GB/s** 内存带宽的设备。
   - 一位用户表示*这就像是在自愿被骗*，而另一位用户则将其概念比作游戏机，一切都打包在一起，尽管花同样的钱可以组装一台更好的 PC。
- **9070 XT 在性能测试中表现平平**：**9070 XT** 明显慢于 **4070 Ti Super**，一位用户报告称，一个在 **4070 Ti Super** 上以 **7 t/s** 运行的模型在 **9070 XT** 上仅达到 **3 t/s**；然而，另一位成员认为 RAM 带宽限制可能是原因。
   - 有人指出 CUDA 很好，但也许 Vulkan 也不错。一位成员发现 **5070 Ti** 售价为 **749 欧元**，但第二天价格就跳涨到了 **1100 欧元**。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1399844353538523288)** (4 messages): 

> `LlamaCloud document agents, LlamaCloud Managed Embeddings, Automated Asset Manager Fund Analysis, LexiconTrail agentic AI systems` 


- ****AI Agent 解析财务文档****：利用能够处理 **10-Ks**、收益报告和监管文件等真实格式的 **AI 驱动文档 Agent**，将复杂的财务文档转化为可操作的数据；更多信息请见 [LlamaIndex 网络研讨会](https://twitter.com/llama_index/status/1950285220663742516)。
- ****LlamaCloud 托管 Embedding****：**LlamaCloud Indexes** 现在拥有托管 Embedding，这意味着你不再需要提供自己的 API key 来嵌入内容；根据 [此推文](https://twitter/llama_index/status/1950345618779754644)，除了托管向量外，系统还会为你嵌入向量。
- ****自动化资产管理基金分析现已推出****：通过这份全面的 Notebook 构建自动化资产管理基金分析，展示了如何处理复杂的财务文档，并使用 **LlamaParse** 将 PDF 转换为结构化 Markdown，从而提取投资分析的可操作见解，详情见 [此推文](https://twitter.com/llama_index/status/1950590734685671931)。
- ****LexiconTrail 助力 10 倍速 Agentic AI 系统****：根据 [此博客文章](https://twitter.com/llama_index/status/1950662723785850911)，**LexiconTrail** 展示了如何利用具有高级索引能力的 **NVIDIA 小语言模型** 构建 **10 倍速的 Agentic AI 系统**。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1399848962025984000)** (126 条消息🔥🔥): 

> `LlamaCloud PDF 检测问题, Character AI 架构, Neo4j 知识图谱问题, Flowmaker Gemini 2.5 Pro 错误` 


- **LlamaCloud 无法检测 PDF，成员寻求指引**：一位成员报告称 **LlamaCloud** 无法检测 **PDF 文件** 并通过 API 进行处理。该成员使用 **n8n** 来简化工作流，并附上了[截图](https://cdn.discordapp.com/attachments/1059201661417037995/1399848961832911031/Screenshot_2025-07-29_at_22.13.16.png?ex=688bcff0&is=688a7e70&hm=b8f51e99fbeae087df203303f7665c4eab8447bb0890b55823fd36074c5ad539&)。
- **关于构建 Character AI 的讨论引发关注**：成员们讨论了如何构建一个对宏大故事有深度理解的 **character AI**，建议使用经典的 **RAG** 流水线，包括文本分块（chunked text）、embeddings 和向量数据库（vector database）。
- **Neo4j 的困扰与图存储过载**：一位成员尝试实现 **Neo4j**，因为他们简单的图存储加载速度*慢得离谱*，但其服务器与 **Neo4j 5.x** 不兼容，而 **LlamaIndex** 似乎不支持 **4.x**，且 **Aura** 被服务器代理拦截。
- **Flowmaker 快速修复 Gemini 2.5 Pro 的 Bug**：一位成员报告了在使用 **Flowmaker** 配合 **Gemini API** 时因模型名称无效而报错，另一位成员迅速指出 [模型名称](https://ai.google.dev/gemini-api/docs/models) 需要包含数字，例如 *gemini-2.5-pro*。
   - 修复代码已迅速[提交](https://github.com/run-llama/flow-maker/blob/aad0f47a81cacba662a07c4f2d70bd3425606e29/src/lib/llm-utils.ts#L19)并部署，解决了该问题，成员对其高效的协助表示感谢。


  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1400095143251677184)** (3 条消息): 

> `RAG 调试, Sparse retrieval, Semantic drift, Chunking collapse, Memory breakdowns` 


- **用户通过 MIT 许可的 Repo 提供 RAG 调试协助**：一位成员分享了一个 **MIT 许可的 repo**，旨在帮助调试棘手的 **RAG 问题**，包括 **sparse retrieval**（稀疏检索）、**semantic drift**（语义漂移）、**chunking collapse**（分块崩溃）和 **memory breakdowns**（内存崩溃）。
   - 另一位成员询问是否能分享使用该 repo 解决的复杂问题，特别是关于 *sparse retrieval* 和 *semantic drift* 的更多细节。
- **关于特定 RAG 调试问题的查询**：在最初的提议后，一位社区成员询问了该 MIT 许可 repo 所解决的具体复杂问题，重点关注具体案例。
   - 该查询特别要求提供关于该 repo 如何处理 **sparse retrieval** 和 **semantic drift** 的详细实例，以寻求比一般性描述更深入的理解。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1400002443323904084)** (5 条消息): 

> `Expert Parallelism (EP) vs Tensor Parallelism (TP), GitHub 上的归并排序问题` 


- **探讨 Expert Parallelism (EP) 的经验**：一位成员正在寻找 **Expert Parallelism (EP)** 优于 **Tensor Parallelism (TP)** 的案例，并指出在他们使用 **Qwen32B** 和 **Qwen 235B** 的经验中，attention 之后 all-reduce 操作带来的额外通信开销使得 **EP** 性能较低。
   - 他们发现 **EP** 仅在采用 **MLA** 且需要 **DP attention** 的模型中才有用。
- **请求协助解决归并排序余数问题**：一位成员在其 [RinomXE GitHub 项目](https://github.com/maybeJosiah/RinomXE) 中遇到了归并排序余数处理的问题。
   - 他们在处理绘制顺序余数逻辑时遇到困难，即在步长翻倍直到超过形状数量时无法正确排序，并发布了一段 javascript 模拟代码片段以寻求帮助。


  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1399868515422441522)** (17 条消息🔥): 

> `Torch Compile, Triton Code Generation, PTX Code Extraction, Inductor Configuration, GEMM Autotuning` 


- **从 Torch Compile 中解锁 PTX 和 Triton 代码**：要获取 **PTX 代码**，请使用 `TORCH_LOGS="output_code" python your_code.py` 或访问 `compiled_kernel.asm.keys()` 字典，详见[这篇博客文章](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/?utm_source=chatgpt.com#ttir-triton-ir)。
   - 该字典包含不同中间表示的键，包括 **llir, ttgir, ttir, ptx 和 cubin**。
- **在 Torch Inductor 中绕过非 Triton 代码生成**：要强制为 matmuls 生成 **Triton 代码**，请在 [torch._inductor.config.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L459-L461) 中配置设置，但请注意，并非所有 op 默认都会转换为 Triton。
   - 诸如 **max_autotune_gemm_backends="TRITON"** 和 **max_autotune_conv_backends** 等选项可以影响 autotuning 过程，尽管内置 kernel 通常更快。
- **通过调整 Inductor 配置实现纯 Triton 代码**：为了让 inductor *仅* 使用 triton 代码，成员们建议修改 `config.py` 和 `utils.py`，特别是 **use_aten_gemm_kernels**, **use_triton_template**, **autotune_fallback_to_aten**, **max_autotune_conv_backends** 和 **max_autotune_gemm_backends** 等设置。
   - 这涉及防止 autotuning 和回退到预写好的 kernel，可能需要探索 **'/tmp/torchinductor_{username}'** 目录。
- **TMA 支持随 Triton 3.4.0 到来**：**TMA (Tensor Memory Accelerator) 支持** 尚未在 Triton 官方版本中提供；用户必须等待 **3.4.0** 版本。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1399872492797038774)** (9 条消息🔥): 

> `livestream review, request accepted` 


- **主播计划进行回顾直播**：一位主播被邀请为社区进行一次 [livestream review](https://www.twitch.tv/)。
   - 该主播回应道：*我不确定这是否是确认邮件。我有空时会确认！我怀疑这只是我们邮件列表的欢迎邮件*。
- **请求已接受！**：一位成员表示团队已接受所有请求。
   - 另一位成员确认他们的请求已被接受：*虽然我是个新手，但这是个很酷的探索方向*。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1400083758618775636)** (2 条消息): 

> `CUPTI metrics in kineto, torch.profiler metrics` 


- **在 kineto 中启用 CUPTI 指标遇到困难**：一位成员询问如何在 **torch.profiler** 中通过 **kineto** 启用 **CUPTI metrics**，可能需要通过自定义构建。
   - 他们引用了一个[相关的 pull request](https://github.com/pytorch/pytorch/pull/125685)，但表示这并没有解决他们的问题。
- **torch.profiler 配置**：该成员尝试使用带有特定配置的 **torch.profiler** 来测量 kernel 性能。
   - 他们尝试配置 **experimental_config**，包含 **profiler_metrics**（如 *kineto__tensor_core_insts*, *dram__bytes_read.sum*, 和 *dram__bytes_write.sum*），并将 **profiler_measure_per_kernel** 设置为 True。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1399912770396487731)** (6 条消息): 

> `CUDA streams, Megatron-LM, Group GEMM, NYC Hackathon, Beginner Hackathon Tips` 


- **CUDA Streams 与 GEMM 性能**：一位成员询问在运行 **GEMM kernels** 时使用 **多个 CUDA streams** 的优势，特别是在 **Megatron-LM** 和 **cuBLAS multi-stream Group GEMM** 的背景下。
   - 该用户质疑了与单流相比的优势，并对开销和有限的 thread blocks 数量表示担忧。
- **纽约市黑客松**：一位成员询问关于黑客松的信息，另一位成员引导他们去特定频道了解更多详情。
   - 该黑客松似乎位于纽约市（NYC）。
- **面向初学者的通用黑客松技巧**：一位成员在 X 上分享了一个[通用黑客松技巧的链接](https://x.com/ayushgun/status/1950444463899512960)，并指出这对初学者很有用。
   - 这些技巧非常通用，并非专门针对 GPU。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 条消息): 

ali_8366: 这里有来自蒙特利尔（Montreal）的人吗？希望能一起喝杯咖啡聊聊。
  

---


### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/)** (1 条消息): 

vishomaru: 你好，这里有人成功使用 AMD GPU Profiler 对 compute shaders 进行过性能分析（profiling）吗？
  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1399941036360601765)** (3 messages): 

> `AI Hackathon, CuTeDSL Blogpost, Software Pipelining` 


- **AI Hackathon 帖子推广**：一位成员分享了一个关于 **AI Hackathon** 的 [LinkedIn 帖子](https://www.linkedin.com/posts/nadiveedishravanreddy_ai-hackathon-qwen-ugcPost-7355265897877434369-7RI5?utm_source=share&utm_medium=member_android)。
   - 该活动现在拥有 **15 位演讲者**，包括来自 **Prime Intellect、Snowflake 和 Jane Street** 的代表。
- **明星演讲阵容的课程推广**：一位成员再次推广了一门课程，提到该课程现在包含 **15 位演讲者**，如 **Prime Intellect**、**Snowflake**、**Jane Street (Sylvain Gugger)** 和 **Daniel Han**（[课程链接](https://maven.com/walk-with-code/scratch-to-scale?promoCode=gpumode40)）。
   - 他们鼓励有费用顾虑的人员联系他们以讨论潜在的资助。
- **编译器自动化 CuTeDSL 优化**：一位成员分享了一篇关于在 **H100** 上使用 **CuTeDSL** 进行 **GEMM** 的 [博客文章](https://veitner.bearblog.dev/let-the-compiler-do-the-work-in-cutedsl/) 和 [代码](https://github.com/simveit/software_pipelining_cute_dsl)，详细介绍了如何让编译器处理 prefetching（预取）。
   - 该博文解释了 `cutlass.range` 算子的一个实验性参数，用于提示预取，从而以更简单的代码实现与手动预取相当的性能。


  

---


### **GPU MODE ▷ #[general-leaderboard](https://discord.com/channels/1189498204333543425/1343002580531417211/1400119615429804166)** (3 messages): 

> `Popcorn-cli DeserializationError, BadCredentialsException on MI300, B200 Timeout Issues, Discord Run Errors` 


- **Popcorn-cli 在 H100 和 A100 上遇到 DeserializationError**：一位用户报告在使用从源码构建的最新版本 **popcorn-cli** 时，在 **H100** 和 **A100** GPU 上出现了 *"DeserializationError | Raw Error: Deserialization failed because the 'libkernelbot' module is not available in the local environment"* 错误。
   - 该错误同时也影响了 **H100** 的 Discord 运行。
- **MI300 面临 BadCredentialsException**：该用户在 **MI300** 上还遇到了 *"BadCredentialsException | Raw Error: 401 {"message": "Bad credentials", "documentation_url": "https://docs.github.com/rest", "status": "401"}"* 错误。
- **B200 超时**：该用户在 **B200** 上经历了一次 **300s 超时**，而该运行在两周前曾成功完成。
- **Popcorn 开发者正在处理**：一位成员表示团队已获悉这些问题，并正在积极开发修复程序。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1400147055963541671)** (5 messages): 

> `Benchmarking Explanation` 


- **Benchmarking 说明会议推迟**：成员们协调了一次会议来解释 **benchmarking**（基准测试）流程，但由于出席人数较少，原定会议提前结束。
   - 一位成员为睡过头表示抱歉，但确认可以参加后续会议来解释 **benchmarking**。
- **睡过头的成员仍然可用**：尽管睡过头了，一位成员确认他们仍然可以解释 **benchmarking** 流程。
   - 该成员旨在重新安排时间，并按计划提供 **benchmarking** 说明。


  

---

### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1399949797603020840)** (20 messages🔥): 

> `gmem synchthreads, cp.async.cg vs cp.async.ca, cutedsl ptx wrapper, nvvm wrapper, cutedsl older drivers` 


- **Gmem 传输后的 SyncThreads 救星**：在从 global memory (**gmem**) 拷贝到 shared memory 之后，手动插入 `synchthreads`（或等效指令）是必要的。
   - 这确保了 shared memory 中的所有元素在参与 **gemm**、reduction 或 scan 等集体计算之前已经全部到达。
- **在 Cutedsl 中控制 Cp.Async 的选择**：一位成员询问如何在 **cutedsl** 中控制使用 `cp.async.cg` 还是 `cp.async.ca`。
   - 建议是编写自定义汇编代码并将其作为拷贝操作提供，尽管这尚未经过测试。
- **Cutedsl 中的 PTX Wrapper 发现**：据一位成员称，**cutedsl** 中没有 **ptx** wrapper 的 API。
   - 然而，另一位成员分享了一个关于如何实现它的示例代码链接，并表示 *在官方 CuTeDSL 代码中也有相关内容* ([quack/utils.py](https://github.com/Dao-AILab/quack/blob/main/quack/utils.py#L67))。
- **Nvvm Wrapper 导航笔记**：一位成员分享了关于如何编写 **nvvm** wrappers 的链接。
   - 他们分享了 [cutlass repo](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/arch/nvvm_wrappers.py) 的链接作为示例，以及 [cutedsl docs](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_cpasync.html#module-cutlass.cute.nvgpu.cpasync) 的链接。
- **Cutedsl 兼容性疑虑澄清**：一位成员询问在旧驱动程序上使用 **cutedsl** 是否可行。
   - 他们目前尚未遇到任何问题，但想知道内部测试是否发现了任何潜在问题。


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1400002892735189022)** (1 messages): 

> `Distributed Training, LLMs, Distributed memory tricks` 


- **Ultrascale Playbook 是极佳的资源**：Hugging Face Spaces 上的 [Ultrascale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) 是分布式训练 **LLMs** 的绝佳资源。
- **分布式训练的内存优化**：该手册为训练 **LLMs** 提供了许多分布式内存技巧。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1399836795881132233)** (38 messages🔥): 

> `GPU Inference vs M3 Ultra, LLMs Offshore with low latency and bad internet, Topological data analysis experts, Speech-LLM models and audio instruction-following capabilities, Manipulating vector embeddings for machine translation` 


- **M3 Ultra 作为本地推理之王**：一位成员建议购买 **M3 Ultra** 进行本地推理，并链接到了一个讨论其 **80-core GPU** 和 **512GB memory** 的 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1j43ziq/the_new_king_m3_ultra_80_core_gpu_512gb_memory/)。
   - 另一位成员分享说，由于其他成员没有回应，他们买了一台二手 **M1 16g**。
- **在离岸环境下低延迟运行 LLMs**：一位成员正尝试在低延迟和网络较差的离岸环境下运行 **LLMs**，寻求解决方案。
   - 另一位成员回应说，如果这是他们想做的，那么花费数百/数千美元是一个可以接受的用例。
- **Speech-LLM 语音指令研究**：一位成员表示有兴趣开展关于提高 **Speech-LLM** 模型 **audio instruction-following capabilities** 的开源研究，这对于创建集成语音的可靠用户界面至关重要。
   - 他们指出最新的研究是来自 Microsoft 的 **Alignformer**，但其代码尚未开源，目前正在评估合作兴趣。
- **用于机器翻译的向量嵌入操作**：一位成员计划基于多语言模型在向量空间中操作 vector embedding。
   - 该成员希望获取 embedding 并加上两种语言均值向量的差值，然后旋转它们直到新语言均值附近的 loss 最小，其他人指出这在 [这篇论文](https://arxiv.org/abs/1309.4168) 中是一个已解决的问题。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1400101077415886860)** (2 messages): 

> `REST models, Compute cost` 


- **社区模型支付协议想法浮出水面**：一位成员想知道通过 REST 提供服务的社区模型是否可以使用 **402** 响应来引用计算成本并实现客户端自动支付。
   - 他们思考了在这种支付系统中，*single-rail 与 h402 multi-rail 如何影响开放性*。
- **开放性的影响**：讨论围绕实施基于 **402** 的支付系统对开放性的影响展开。
   - 针对 single-rail 与 multi-rail 方法提出了担忧。


  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1399832395791728763)** (17 条消息🔥): 

> `In-Context Learning (ICL), Interpretability Tools, Sparse Autoencoders (SAEs), Lucas Critique, Activation Distributions` 


- **ICL 可能会破坏 Interpretability Tools，相关言论引发关注**：一位成员推测 **In-Context Learning (ICL)** 可能会破坏 **Sparse Autoencoders (SAEs)** 等 **Interpretability Tools**，因为它会将激活值（activations）推离其训练时的分布。
   - 该成员引用了 **Lucas Critique**，认为干预（如对 LLM 进行提示）需要基于对这些干预具有不变性的微观基础（microfoundations）进行预测，并[分享了一篇论文](https://arxiv.org/abs/2501.00070v1)来支持其观点。
- **SAEs 在 ICL 面临泛化挑战**：一位成员同意将 **SAEs** 应用于具有显著 **ICL** 的上下文可能会失败，因为稀疏表示（sparse representations）无法很好地泛化到未参与训练的激活分布。
   - 他们澄清说这个问题并非 **ICL** 特有，而是每当 **SAEs** 应用于与其训练分布不同的激活分布时都会出现。
- **ICL 对激活分布的影响：OOD？**：一位成员假设，通过 **ICL** 将模型的行为调节到大分布的一个极小切片中，可能会破坏为无约束情况构建的诊断工具，从而可能导致新颖的内部行为。
   - 另一位成员反驳称 **ICL** 可能会将激活推向“分布内”（in-distribution），并引用了 **SAE** 特征在上下文特定实例中激活的例子，指向了关于 *function vectors/task vectors* 的论文。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1399902669526663381)** (1 条消息): 

> `Model Evaluation Metrics` 


- **调试模型评估指标**：一位成员提议协助调试一个接收单个输入文档和模型预测并返回评估指标的函数。
- **理解函数过程**：建议包括理解函数如何处理数据以识别潜在问题。


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1399859285504168006)** (1 条消息): 

> `Diffusion Models Study Group, Flow Matching, MIT Curriculum` 


- **新的 Diffusion Models 学习小组启动**：一个新的 **12 人、为期 5 个月的学习小组** 正在启动，旨在从零开始学习扩散模型，基于 **MIT 的课程**（[讲义](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf)）。
   - 该小组专为 AI 从业者设计，成员包括 CTO、AI 讲师和 AI 研究员。
- **参加关于 Flow Matching 和 PDEs 的免费入门课程**：该学习小组将举办 **两场免费入门课程**，分别在 **8 月 2 日**（[Flow Matching & Diffusion Models](https://lu.ma/kv8zf6va)）和 **8 月 9 日**（[PDEs, ODEs, SDEs + A Brief History of Diffusion Models](https://lu.ma/uk6ecrqo)），均为美国东部时间中午 12 点。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1399899213365645353)** (4 条消息): 

> `MoE Implementation, grouped_mm, Low Precision Training, Float8 Training` 


- **Grouped GEMM 实现讨论**：一位成员询问了关于在 **GPT-NeoX** 中支持 **torch._grouped_mm** 的 PR，该功能现已在 PyTorch 核心库中可用，可能带来性能提升，特别提到了[这个 MoE 实现](https://github.com/EleutherAI/gpt-neox/blob/d12c771198388980ee054617e537665f044e0584/megatron/model/moe_mlp.py#L221)。
   - 他们表示，对 **低精度 MoE 训练** 感兴趣的用户可以使用 TorchAO 的单行代码。
- **深入研究 PyTorch 的 Grouped GEMM 实现**：一位成员询问了 PyTorch **_grouped_mm** 的底层实现，并请求与 megablocks 的 grouped GEMMs 进行性能对比。
   - 另一位成员指出其底层使用了 **CUTLASS kernel**，并链接到了[相关源代码](https://github.com/pytorch/pytorch/blob/62f98dbb44fb338ba849f93c491ea170af4c187c/aten/src/ATen/native/cuda/GroupMM.cu#L418)。
- **Float8 Blockwise 预训练的复兴**：一位成员质疑由于收敛问题，人们对 **低精度训练** 缺乏兴趣，称其为“*除非性能极具吸引力，否则很难推销*”。
   - 另一位成员反驳并引用了 **DeepseekV3 的 float8 blockwise 预训练** 以及他们自己在 **FP8 rowwise** 下稳定的收敛结果，实现了约 30-40% 的吞吐量提升，详见[这篇 PyTorch 博客文章](https://pytorch.org/blog/accelerating-large-scale-training-and-convergence-with-pytorch-float8-rowwise-on-crusoe-2k-h200s/)。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1400078020387147887)** (7 条消息): 

> `CUDA generalization paper, TLS handshake EOF error, Mojo package installation, Region-specific access issues` 


- **分享了 CUDA 泛化论文**：一名成员分享了一篇[关于超越 CUDA 的泛化论文](https://huggingface.co/papers/2507.14111)。
   - 另一名成员对其分享表示感谢，并建议今后将此类内容发布到相应的频道。
- **用户遇到 TLS Handshake EOF 错误**：一位新的 Mojo 用户报告称，在尝试通过 **pixi** 和 **magic shell** 运行 Mojo 项目时遇到了 **TLS handshake EOF error**。
   - 他们还提到了安装从 Microsoft Copilot 获取的 **dockerfile** 时遇到的问题。
- **针对 TLS Handshake 问题的建议解决方案**：一名成员建议 **TLS handshake issues** 可能与特定区域访问包仓库有关，并提供了一个通过 **pixi** 在最新的 nightly 版本中尝试新 `mojo` 包的解决方案。
   - 建议的命令为：`pixi init my-project -c https://conda.modular.com/max-nightly/ -c conda-forge`，随后执行 `cd my-project` 和 `pixi add mojo`。
- **VPN 无法解决安装问题**：遇到 **TLS handshake EOF error** 的用户报告称，即使使用了 VPN，建议的解决方案也无效。
   - 另一名成员提到，自从几个月前迁移到新主机后，区域性问题应该已经得到解决。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1399916351182868503)** (41 条消息🔥): 

> `Mojo external calls vs libc, Mojo to Python overhead, Embedding CPython in Mojo binaries, Python performance, Mojo and hot loops` 


- **Mojo 的外部调用引发疑问**：用户询问为什么 Mojo 的 `external_call` 使用特定的函数（如 `KGEN_CompilerRT_IO_FileOpen`）而不是 libc 中的标准 `fopen`，以及这是否是为了安全性。
   - 一名成员澄清说，其中许多是 Mojo 功能尚不完善时的产物，目前并不是修复的优先级重点；此外，KGEN 命名空间属于 Modular，最终将会开放。
- **Mojo 到 Python 的开销似乎很大**：一位用户发现，从 Mojo 调用 Python 的 no-op 函数明显慢于直接从 Python 调用（1000 万次调用耗时 4.5 秒 vs 0.5 秒），并指出这种差异比 Rust 通过 Pyo3 与 Python 互操作的开销更显著。
   - 其他成员指出，Mojo 需要启动一个 CPython 进程来执行 Python 函数，这会产生开销，将其与 Rust 到 Python 的互操作进行对比并不对等。
- **二进制文件中嵌入了 CPython**：讨论围绕 CPython 是嵌入在 Mojo 二进制文件中还是 Python 代码被编译展开，这影响了从 Mojo 调用 Python 的性能开销。
   - 澄清指出，CPython 是通过 `dlopen libpython` 嵌入的，并维护了一个指向解释器的指针，每次调用都会重用同一个解释器，因此出于性能考虑，不应在 hot loop（热循环）中调用它。
- **在 Mojo 中低延迟胜过热循环**：讨论了从 Mojo 调用 Python 对性能的影响，特别是在处理 OpenCV 或 Mujoco 机器人仿真等任务的 hot loops 时，强调这样做就像“在赛车引擎里喷胶水”。
   - 成员们指出，许多快速的 Python 库实际上是带有包装器的 C 库，而“仅仅是与 context dicts 交互就很容易消耗数百个时钟周期”。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1399878169456152708)** (38 条消息🔥): 

> `Aider site framework, Deepseek-chat OpenRouter Issues, SWE-bench Leaderboard, Aider's Role in Model Training, Qwen3 Coder 30B-A3B announcement` 


- **Deepseek-chat 在 OpenRouter 上表现变差**：成员们注意到 **Deepseek-chat** 在 **OpenRouter** 上的表现比官方 **DeepSeek API** 更差，在 architect 模式下作为编辑器模型使用时，它会返回整个函数而不是选择性的 diff。
   - 建议使用 `aider --model openrouter/deepseek/deepseek-r1` 作为修复方案，因为这能确保使用 [aider/resources/model-settings.yml](https://github.com/Aider-AI/aider/blob/main/aider/resources/model-settings.yml#L548) 中的默认配置，该配置包含 `edit-format: diff` 设置。
- **Aider 可能用于训练编程模型**：有人建议 **Aider** 可以通过记录开发工作流中需要进行 lint 检查或撤销操作的地方，来辅助编程模型的训练。
   - 这将利用 Aider 在“谨慎”开发中的应用来提供有价值的训练数据，尽管评论者并未主张开发者去实现这一功能。
- **Qwen3 Coder 30B-A3B 发布**：分享了一张关于新模型 **Qwen3 Coder 30B-A3B** 的图片。
   - 该图片是发布公告的截图，证实了其真实性。
- **用户遇到 Litellm API 连接错误**：一名用户报告遇到了大量的 `litellm.APIConnectionError: Error parsing chunk: Expecting property name enclosed in double quotes: line 1 column 2` 错误。
   - 这些错误似乎并未影响功能。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1400180575096012831)** (3 条消息): 

> `Open Model Selection, Hardware Considerations for Aider, Runpod Credits, R1 Model, Qwen Coder Model` 


- **开源模型对决：R1 vs Qwen Coder**：一位成员在拥有无限硬件资源的情况下，就最适合与 **aider** 配合使用的开源模型寻求建议，并考虑测试 **R1** 和 **Qwen Coder** 模型。
   - 该成员提到有 **Runpod credits** 可以消耗，表明其打算对这些模型进行实际测试。
- **Llama3 与 Aider 的集成讨论**：成员们讨论了 **Llama3** 与 **Aider** 的集成以及兼容性问题。
   - 一些成员针对现有的模型选项提出了一些有用的集成改进建议。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1399838202122207312)** (29 条消息🔥): 

> `LLM Safety Alignment Research, AI alignment blogs, CUDA with Claude, Z.AI 54 open source repos, Math in paper discussion` 


- **寻求 LLM 安全对齐研究资源**：一名博士生正在寻求关于如何跟进当前 **LLM safety/alignment research** 的建议，特别是优秀的综述论文。
   - 建议包括来自 **AI alignment forum** 的四篇博客，包括 [Thought Anchors: Which LLM Reasoning Steps Matter](https://www.alignmentforum.org/posts/iLHe3vLur3NgrFPFy/thought-anchors-which-llm-reasoning-steps-matter)、[Measuring Beliefs of Language Models During Chain-of-Thought](https://www.alignmentforum.org/posts/a86uAnPykqNtmEbDH/measuring-beliefs-of-language-models-during-chain-of-thought-1) 等。
- **使用 Claude 编写 CUDA 代码的复杂性**：一位成员发现，在 **Claude** 的帮助下编写 **CUDA** 非常复杂，需要规划、深入理解和组织。
   - 他们认为，真正的智能测试应该是让一名具备一定 GPU 和 **CUDA** 知识的 Python 开发者，能否利用 **Claude** 来引导编写 **kernels** 并优化性能，并附带了一张[图片](https://cdn.discordapp.com/attachments/986699377257119794/1400233738369110026/image.png?ex=688be4ca&is=688a934a&hm=1bcb11346477e61edf05cde9751d5e62ee8992a2f64216c07e4a1a8f8fb14cc4)。
- **Z.AI 54 个开源仓库引起关注**：一位成员询问是否有人看到并尝试研究新的 **Z.AI 54 open source repos**。
   - 未提供关于这些仓库或其具体内容的更多细节。
- **表达对 Voyager 的喜爱**：一位成员分享了 [Voyager](https://www.youtube.com/watch?v=H0XYANRosVo) 的链接并表达了对它的喜爱，另一位成员立即回应“我也是！”。
   - 另一位成员分享了今天发布的 [Simons Foundation](https://www.youtube.com/playlist?list=PLWAzLum_3a18wO6C7TP8_4XGw4pDxy6G5) 播放列表链接。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1399983269071818853)** (1 条消息): 

> `Qwen3, GPT-4o` 


- **Qwen3 30B 性能比肩 GPT-4o**：一位成员分享了一篇帖子，指出 [Qwen3 30B A3B 2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) 在中英文表现上与 **OpenAI GPT-4o** 持平。
- **Qwen3 获得关注**：社区成员对 **Qwen3** 作为语言模型领域强有力竞争者的潜力感到兴奋。
   - 早期基准测试表明，它在某些任务中（特别是在多语言环境下）可能提供与 **GPT-4o** 相当的性能。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1399833974972350474)** (30 条消息🔥): 

> `Kimi chatbot, Moonshot AI vibe, OpenHands, Training dataset of Kimi, Scale AI` 


- **Kimi Chatbot 以中文回复**：一位用户报告称，尽管用英文提问，**Kimi chatbot** 仍以**中文**回复，这可能与账号登出有关。
   - 截图显示，虽然回复是英文，但推荐的来源和问题却是中文。
- **Kimi 的训练数据倾向于社交媒体**：一位成员开玩笑说，**Kimi 的训练数据集**似乎包含了 **Instagram** 和 **TikTok** 的评论，并认为这就是它表现出色的原因。
   - 他们链接到了 [Kimi on OpenHands v0.50.0k2](https://github.com/All-Hands-AI/OpenHands/releases/tag/0.50.0k2) 以支持这一说法。
- **Moonshot AI 的氛围 (Vibe)**：一位成员表示 *Moonshot 的氛围（vibe）确实是最好的*，另一位成员也同意社区需要一些竞争。
   - 他们链接到了关于 AI 社区氛围检查的 [X 帖子](https://x.com/crystalsssup/status/1944287779896328668)。
- **Scale AI 创始人 Alexandr Wang**：一位成员提到 **Alexandr Wang** 是 **Scale AI** 的创始人兼 CEO，这是一家数据基础设施公司。
   - 他们指出，**Scale AI** 提供对于开发机器学习模型至关重要的训练数据、标注和评估服务。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1399839990069854348)** (25 条消息🔥): 

> `Lume vs Suna, Manus' Comic Creation, The Future of Manus` 


- **成员辩论：Lume 略胜 Suna**：成员们辩论了 **Lume** 和 **Suna** 这两个 Agent 系统的优劣；一位成员表示 *Lume 在编写特定代码方面表现得更好*且错误更少，但也承认可能没有正确地给 Suna 提示词（Prompt）。
   - 该成员指出，由于某些任务的成本过高，他们无法将其与 **Manus** 进行比较。
- **Manus 漫画创作：璞玉待琢？**：一位成员建议 **Manus** 的漫画创作功能很不错，但仍有改进空间。
   - 另一位成员表示服务质量正在下降，对免费用户的限制较多，并质疑 **Manus 是否已经“凉了”**。
- **乐观的 AI vs 怀疑的人类：Manus 的未来**：一位成员询问 AI 对 **Manus** 未来的看法，AI 回复道 *“我认为 Manus 的未来是光明的”*。
   - 另一位成员表示怀疑，理由是 **OpenAI** 和 **Google** 发布了 Agent 模式。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1399834491513475083)** (22 条消息🔥): 

> `MCP Server 安全性、结合 LLM 和 MCP 的 BDD 测试、CursorTouch 与 Claude 的 Windows-MCP 问题、FastMCP 工具选择、托管 MCP Server` 


- **MCP Server 需要用户上下文隔离**：一位用户正在寻求澄清，即单个云端部署的 **MCP Server 实例**是否需要额外的层来进行**用户上下文隔离 (user-context separation)**，以防止多个客户端同时访问时在不同会话之间共享数据，参考了 [issue #1087](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1087) 和 [MCP 文档](https://modelcontextprotocol.io/docs/tutorials/use-remote-mcp-server#understanding-remote-mcp-servers)。
- **Cursor 可连接，Claude 失败**：一位用户报告称，已成功将 **MCP Server 部署到 EC2**，并配置了正确的 **SSL 证书和域名设置**，但他们只能通过 **Cursor** 连接，而无法通过 **Claude Desktop** 连接。
- **Cucumber、BDD 与 LLM 联手！**：一位成员分享了一个基于**行为驱动开发 (BDD)** 的侧边项目，该项目已达到生产就绪状态；他们还附带了[解决方案的架构图](https://cdn.discordapp.com/attachments/1312302100125843479/1399854833565044756/bael.jpeg?ex=688bd568&is=688a83e8&hm=2e86139e9f117265cd7cbef2afcc1a23a34a091e79402df9a0e051261231c695&)。
- **Windows-MCP State Tool 报错问题**：一位用户在 Claude Desktop 中使用 CursorTouch 的 Windows-MCP 时遇到困难，因为 **state tool 完全无法工作**，并提示：*Error calling tool 'State-Tool': 'Taskbar'*。
- **FastMCP 动态工具选择**：一位用户询问 **FastMCP** 是否包含在服务器定义了多个工具时，在客户端**动态且自动选择工具**（例如：数学、网页搜索、RAG、数据解释器）的逻辑。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1399914906672959628)** (2 条消息): 

> `DeepTrail, Deepsecure, 开源认证, AI Agent 委托层, 安全的多 Agent 工作流` 


- ****DeepTrail** 为 AI Agent 授权推出 **Deepsecure****：由 Berkeley SkyDeck 支持的 **DeepTrail** 正在开发 **Deepsecure**，这是一个为 AI Agent 设计的开源认证与委托层，能够通过极少的代码集成授权、Agent 到 Agent 的委托、策略执行以及安全代理，详见 [GitHub](https://github.com/DeepTrail/deepsecure)。
- **探索 **Deepsecure** 的架构与安全多 Agent 工作流**：**Deepsecure** 的架构具有分片密钥设计、网关/代理、独立的控制/数据平面、策略引擎以及用于 Agent 间委托的 macaroons，详见其[技术概览](https://github.com/DeepTrail/deepsecure/blob/dev/docs/design/deepsecure-technical-overview.md)。
- ****Deepsecure** 与 Langchain/LangGraph 的集成示例**：**Deepsecure** 与 Langchain/LangGraph 的集成示例包括：*安全的多 Agent 工作流* ([代码链接](https://github.com/DeepTrail/deepsecure/blob/dev/examples/05_langchain_secure_tools.py))、*委托工作流* ([代码链接](https://github.com/DeepTrail/deepsecure/blob/dev/examples/09_langchain_delegation_workflow.py))、*高级委托模式* ([代码链接](https://github.com/DeepTrail/deepsecure/blob/dev/examples/11_advanced_delegation_patterns.py)) 以及 *平台 Agent 引导* ([代码链接](https://github.com/DeepTrail/deepsecure/blob/dev/examples/12_platform_expansion_bootstrap.py))。
- **带有社区功能和市场的优质目录**：一位成员开始开发一个旨在成为“优质目录”的项目，具备社区功能，并计划演变为一键安装甚至市场，访问地址为 [protocoldepot.dev](https://protocoldepot.dev/)。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1399896190509912094)** (18 messages🔥): 

> `DSPy 可学习参数提案，使用 f-strings 的 Signature 实现，DSPy 对比 GEPA` 


- **DSPy 可学习参数提案引发关注**：成员们讨论了向 DSPy 添加可学习参数（`dspy.variable` 或 `dspy.parameter`）的提案，并[创建了一个 issue](https://github.com/stanfordnlp/dspy/issues/8593) 来收集想法和用例。
   - 一位成员将其描述为*非常出色的提案*，希望允许*模板作为参数/变量*，以便优化器可以输出最优提示词，以及模板变量的放置。
- **F-Strings 导致 Signature 实现问题**：一位成员请求帮助使用 f-string 实现 signature，希望根据描述验证代码。
   - 另一位用户建议不要采用这种方法，并建议*将参数描述放在 `dspy.InputField()` 中*。
- **DSPy 在提示词优化对决中对阵 GEPA**：一位成员注意到一段 YouTube 视频，其中将 **DSPy** 与 **GEPA** 进行了对比，观点犀利：*DSPy 优化你给出的提示词；GEPA 则进化出你从未想象过的提示词*，并链接了该 [YouTube 视频](https://www.youtube.com/watch?v=o6RbVPFOslg)。
   - 该成员提议将 **MIPRO** 变成一个*反思性的、遗传算法风格的前沿引擎*，让 DSPy 生成并维护提示词的 Pareto-frontier（帕累托前沿），旨在反驳该 YouTuber 的观点。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1400072601677856810)** (15 messages🔥): 

> `游戏领域的 AMD 对比 Nvidia，Qwen 代码模型发布，RLVR 讨论` 


- **AMD：游戏与开拓 AI 新路径的选择**：一位成员建议在游戏方面购买 **7900XT** 而非 **9070**，并搭配 **7800X3D** 而非 **9900X**，同时指出 AMD 在消费级 AI 方面的可用性以及潜在的长期社区利益。
   - 他们链接了一条 [推文](https://x.com/Teknium1/status/1950596567968477382) 来支持其论点。
- **Qwen 随代码模型发布开启“思考”模式**：一位成员宣布即将在 [Hugging Face](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507) 上发布 **Qwen3-30B-A3B-Thinking-2507** 代码模型。
   - 该 Hugging Face 模型链接指向了一个新的代码生成工具。
- **Nvidia 的 RLVR：它真的是一种 RL 算法吗？**：一位成员质疑 **RLVR**（Reinforcement Learning, Virtual Reality）是否应被归类为强化学习算法，并链接了一条引发讨论的 [NVIDIA 推文](https://fxtwitter.com/NVIDIAAIDev/status/1950279130450444670)。
   - 另一位成员 teknium 表示：*“RLVR 根本不是一种 RL 算法，它只是 RL 的一个目标”*。


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1400127686264881253)** (3 messages): 

> `MRC 模型对比，暑期学校频道请求，高级软件工程师远程职位` 


- **成员询问暑期学校频道**：一位新成员正在询问几周前举行的 **summer school** 是否有专门的频道。
- **MRC 模型对比策略**：一位成员询问是将自定义的 **MRC 模型** 与大型预训练模型的 **zero-shot 性能** 进行对比，还是在相同数据集上对大型模型进行 **fine-tune** 以进行更公平的比较。
- **发布长期远程高级软件工程师职位**：正在招聘高级软件工程师，月薪 **$2K**，为长期远程合同，工作地点位于**非洲**或**美洲**。
   - 该职位要求具备 **Ruby on Rails**、**Node.js**、**C#/.NET**、**Python**、**Java** 或类似经验，以及母语或接近母语水平的英语沟通能力。


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1400077240896590014)** (1 messages): 

> `Langchain-Cohere 引用模式，langchain_cohere.ChatCohere` 


- **引用选项在 Langchain-Cohere 上不起作用**：一位成员在尝试使用 `langchain_cohere.ChatCohere` 上的 `citation_options` 更改引用模式时遇到问题。
   - 该成员询问是否有任何隐式方式传递引用选项，因为 `langchain_cohere.ChatCohere` 不显式接受它。
- **Langchain-Cohere 仓库状态：无人维护？**：一位成员询问 [langchain-cohere 仓库](https://github.com/langchain-ai/langchain-cohere) 是否为官方仓库。
   - 他们注意到该仓库在过去几个月内没有更新，并询问*是否欢迎在该处提交 Pull Requests*。


  

---

### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1400056898665058364)** (6 messages): 

> `AI Safety, LLM Bias Mitigation, GPU Kernel Optimization` 


- **统计学学生在 AI 领域寻求安全空间**：一名统计学硕士生表达了对 **ML research** 的兴趣，特别是 **technical AI safety** 方向，并对研究合作持开放态度。
- **博士生专注于伦理 LLMs**：奥地利 JKU Linz 的一名博士生正致力于 **mitigating social bias in LLMs**。
   - 他们的其他兴趣包括 **attribution for generative models, AI generated text detection, 以及 domain adaptation**，并希望与从事特定领域 LLMs 实际伦理问题研究的人员建立联系。
- **RAG 与图谱助力毕业生发展**：慕尼黑工业大学的一名应届硕士毕业生正在通过个人项目积累 **RAGs**、**knowledge graphs** 和新编程语言的经验。
   - 他们希望获得研究经验，参与项目协作，并结识志同道合的人以紧跟新技术。
- **Ali 攻克自回归加速难题**：一位名为 Ali 的成员正在研究 **在 Triton/CUDA 中为 autoregressive models 优化 GPU kernels**。
   - 他们非常乐意交流关于底层 GPU programming 的话题。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1399896806522884186)** (2 messages): 

> `LoRA-style adapter in Torchtune, Merged weights in Torchtune` 


- **用户请求在 Torchtune 中支持 LoRA-style adapter**：一位用户询问 Torchtune 是否支持 **LoRA-style adapter**，特别是那种保留精确 forward compute path 且不改变 computational cost，但冻结原始模型权重并通过额外的可训练层进行更新的方案。
   - 他们正在寻求 **额外的可训练层**。
- **Torchtune 在使用 adapter 训练后合并权重**：一位用户分享了关于 **end-to-end workflow** 的 Torchtune 文档 [链接](https://docs.pytorch.org/torchtune/0.6/tutorials/e2e_flow.html)，强调 Torchtune 支持使用 adapter 进行训练并能将权重合并回原模型。
   - 他们正在询问有关合并权重的问题。


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1400146555914293331)** (2 messages): 

> `ACL Paper Award, Glianorex finetunes` 


- **ACL 论文获奖**：一位成员分享了他们刚刚获奖的 **ACL paper**，链接见 [此处](https://aclanthology.org/2025.acl-long.266/)。
- **Glianorex Finetunes 发布**：一位成员询问 **finetunes** 是否公开，并抱怨 *Glianorex 让他痛苦不堪，而医生也无能为力*。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1400165808369438871)** (2 messages): 

> `Certificate Declaration Form` 


- **证书声明表需要填写**：一位成员被提醒尚未完成证书声明表（certificate declaration form）。
   - 工作人员确认 *遗憾的是，我们从未收到您的证书声明表*。
- **仍需提交证书表单**：工作人员重申尚未收到证书声明表。
   - 该成员此前已被告知其表单缺失。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1399831422461804604)** (2 messages): 

> `Diffusion Models Study Group, MIT Diffusion Models Curriculum, Flow Matching, Generative AI, AI Education` 


- **在新的学习小组中从零开始学习 Diffusion Models**：一个新的学习小组正在启动一个为期 **5 个月**、共 **12 人** 的项目（**每周 2-4 小时**），基于 [MIT 的课程大纲](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf) 学习 diffusion models，这已成为 Generative AI 的核心架构。
   - 前两次介绍性课程免费并向非成员开放：8 月 2 日关于 *Flow Matching & Diffusion Models*，8 月 9 日关于 *PDEs, ODEs, SDEs + Diffusion Models 简史*（[链接见此](https://lu.ma/kv8zf6va)）。
- **AI Scholars 宣布成立新的 Diffusion Models 学习小组**：AI Scholars 正在启动一个 Diffusion Models 学习小组，已确认的成员包括 AI 电影工具的 CTO、AI 艺术讲师、2 名 LLM 讲师和 2 名全职 AI 研究员。
   - 该项目特点包括同行引导的课程、导师问答、动手项目、真实论文研读，以及一个紧密互信的小组，采用每周 2 小时直播课 + 2 小时自学的模式，学生轮流授课。


  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1399993952987512853)** (1 条消息): 

> `Deploying custom language models, Hugging Face deployment, GUI for user queries` 


- **寻求云端部署策略**：一位用户询问如何将使用自定义 PDF 文件夹训练的语言模型部署到云端供公众使用，特别是寻求一个用于用户查询的简单 GUI。
   - Nomic 建议企业版方案（enterprise plan）并不适合，用户想知道 **Hugging Face 部署** 是否可以作为替代方案。
- **企业版方案不适用**：Nomic 表示企业版方案（enterprise plan）不符合用户的需求。
   - 用户正在探索替代的部署策略，例如 Hugging Face，以使其语言模型可供访问。