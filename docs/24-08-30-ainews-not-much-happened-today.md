---
companies:
- meta-ai-fair
- hugging-face
- magic-ai-labs
- lmsys
- alibaba
- openai
date: '2024-08-31T00:41:42.203560Z'
description: '以下是为您翻译的中文内容：


  **Meta** 宣布 **LLaMA 3.1** 获得了广泛采用，在 Hugging Face 上的下载量已接近 **3.5 亿次**。**Magic AI
  Labs** 推出了 **LTM-2-Mini**，这是一款拥有 **1 亿 token 上下文窗口**的长文本模型，并引入了名为 HashHop 的新评估方法。**LMSys**
  在其 Chatbot Arena 排行榜中加入了风格控制（style control），提升了 **Claude 3.5 Sonnet** 和 **LLaMA
  3.1 405B** 等模型的排名。**阿里巴巴**发布了 **Qwen2-VL**，这是一款采用 Apache 2.0 协议的多模态大语言模型，性能可与 **GPT-4o
  mini** 媲美。**OpenAI** 首席执行官 **Sam Altman** 宣布与美国 AI 安全研究所（US AI Safety Institute）合作，开展模型发布前的测试。**Ajeya
  Cotra** 重点讨论了 AI 安全及潜在的 AI 接管风险。文中还提到了用于网页爬取的 **firecrawl** 等工具以及 PDF 处理中的挑战。**François
  Chollet** 探讨了 AI 炒作周期和市场趋势，**Rohan Paul** 则分享了 AI 对呼叫中心可能带来的颠覆。'
id: 7de5091f-80d2-45af-b049-8df262bd2e14
models:
- llama-3-1
- claude-3-5-sonnet
- llama-3-1-405b
- ltm-2-mini
- qwen2-vl
- gpt-4o-mini
original_slug: ainews-not-much-happened-today-5498
people:
- sam-altman
- ajeya-cotra
- fchollet
- rohanpaul_ai
- philschmid
title: 今天没发生什么特别的事。
topics:
- long-context
- style-control
- multimodality
- ai-safety
- model-evaluation
- web-crawling
- pdf-processing
- ai-hype-cycles
- call-center-automation
---

<!-- buttondown-editor-mode: plaintext -->**3天周末就是你所需要的一切。**

> 2024年8月29日至8月30日的 AI 新闻。我们为你检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 社区（包含 **213** 个频道和 **3131** 条消息）。预计节省阅读时间（以 200wpm 计算）：**340 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 来进行 AINews 讨论！

我们考虑的一些零散内容：

- 重点关注本周的 [Google Gemini](https://x.com/OfficialLoganK/status/1828508078955696337) 和 [Cohere Command R](https://x.com/itsSandraKublik/status/1829519989969133757)（[博客文章](https://docs.cohere.com/changelog/command-gets-refreshed)，但排行榜尚未更新）模型更新。
- Lmsys 通过[引入风格控制（style control）](https://x.com/lmsysorg/status/1829216988021043645)排行榜来回应批评，尽管 ChatGPT-4o-latest 依然碾压其他所有人。
- Meta 的 AI 助手[宣布](https://x.com/Ahmad_Al_Dahle/status/1829541138736509102)其 MAU（月活跃用户）达 4 亿，WAU（周活跃用户）为 1.85 亿，DAU（日活跃用户）为 4000 万。

但似乎没有什么是“必须了解”的。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道总结**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

> 所有总结均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型开发与基准测试**

- **LLaMA 3.1 的采用**：Meta 宣布 LLaMA 模型被大量采用，在 Hugging Face 上的下载量接近 3.5 亿次，并在各行各业得到广泛应用。[@AIatMeta](https://twitter.com/AIatMeta/status/1829157383052111946) 强调了开源 AI 在将利益扩展到每个人方面的重要性。

- **长上下文模型**：Magic AI Labs 推出了 LTM-2-Mini，该模型具有 1 亿 token 的上下文窗口。[@magicailabs](https://twitter.com/magicailabs/status/1829206893765767282) 声称这相当于 1000 万行代码或 750 本小说。他们还引入了 HashHop，一种用于长上下文模型的新评估方法。

- **AI 评估中的风格控制**：LMSys 在其 Chatbot Arena 的回归模型中引入了风格控制，旨在将排名中的风格影响与实质内容分开。[@lmsysorg](https://twitter.com/lmsysorg/status/1829216988021043645) 报告称，当风格受到控制时，Claude 3.5 Sonnet 和 Llama-3.1-405B 等模型的排名显著上升。

- **Qwen2-VL 发布**：阿里巴巴发布了 Qwen2-VL，这是一款新的多模态 LLM，提供 2B 和 7B 尺寸，采用 Apache 2.0 许可证。[@_philschmid](https://twitter.com/_philschmid/status/1829190887399673908) 指出其在各种基准测试中与 GPT-4o mini 相比具有竞争力。

**AI 安全与监管**

- **美国 AI 安全研究所测试**：OpenAI CEO [@sama](https://twitter.com/sama/status/1829205847731515676) 宣布与美国 AI 安全研究所达成协议，对未来模型进行发布前测试，强调了国家级测试的重要性。

- **关于 AI 接管的担忧**：[@ajeya_cotra](https://twitter.com/ajeya_cotra/status/1829214030629876106) 讨论了针对潜在 AI 接管的预防措施，质疑如何在灾难性伤害发生之前建立共识和行动意愿。

**AI 应用与工具**

- **网页爬取工具**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1829158662964691159) 分享了关于 firecrawl 的信息，这是一个开源工具，用于爬取整个网站并将其转换为适用于 LLM 的 Markdown 或结构化数据。

- **PDF 处理挑战**：[@svpino](https://twitter.com/svpino/status/1829137471717658884) 强调了使用当前 AI 模型处理 PDF 文档的困难，并建议将文档预处理为文本格式以获得更好的效果。

**AI 行业与市场趋势**

- **AI 炒作周期**：[@fchollet](https://twitter.com/fchollet/status/1829258691100737701) 观察到，科技界的 AI 炒作高峰在 2023 年 Q1-Q2，而公开市场的 AI 贪婪高峰在 2024 年 Q1-Q2，并指出无论如何，AI 研究和应用都在继续取得进展。

- **呼叫中心行业颠覆**：一篇热门的 Reddit 帖子讨论了 AI 对呼叫中心行业的潜在影响，认为 AI Agent 可能会在两年内取代人类员工。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1829204901957706037) 分享了这一点，并指出了其对客户服务和就业的影响。


---

# AI Reddit 回顾

## /r/LocalLlama 摘要

**主题 1：长上下文 AI 推理的进展**

- **本地 1M 上下文推理，速度达 15 tokens/s 且“大海捞针”准确率约 100%：InternLM2.5-1M 在 KTransformers 上运行，仅需 24GB VRAM 和 130GB DRAM。支持 Windows/Pip/多 GPU 等。** ([Score: 114, Comments: 28](https://reddit.com//r/LocalLLaMA/comments/1f3xfnk/local_1m_context_inference_at_15_tokenss_and_100/))：KTransformers 项目为 **InternLM2-1M 模型**引入了**本地 1M 上下文推理**，在仅使用 **24GB VRAM** 和 **130GB DRAM** 的情况下，实现了 **15 tokens/s** 的推理速度，并在“大海捞针”（Needle In a Haystack）挑战中达到约 100% 的准确率。该项目基于 H2O、InfLLM、Quest 和 SnapKV 等研究，实现了一个**高效的 CPU 稀疏注意力算子（sparse attention operator）**，使 1M 挑战的推理速度提升了 **6 倍**，成功率达到 **92.88%**，同时在 128K 测试中保持 **100% 准确率**。
  - **RULER 基准测试**显示 **InternLM2.5** 的“有效”上下文长度仅为 **4K tokens**，超过此长度后表现逊于 **Llama2-7b**。项目开发者表示稍后将测试 RULER，并强调他们的演示展示了**稀疏注意力算子**的有效性。
  - 用户表示有兴趣将 **Mistral Large 2** 添加到项目的模型列表中，目前该列表已包含 **Mixtral-8x22B**。一些评论者认为该项目的进展“令人兴奋”。
  - 部分用户报告了安装问题，有人在 cmake 过程中遇到了 pip 的 **404 错误**。这表明某些用户在设置该项目时可能面临技术挑战。

**主题 2：加州 SB 1047 法案：对 AI 开发的影响**

- **[SB 1047 法案通过。你认为这会影响 LLAMA 吗？]** ([Score: 52, Comments: 68](https://reddit.com//r/LocalLLaMA/comments/1f4lbfy/sb_1047_got_passed_do_you_think_this_will_affect/))：**SB 1047** 是一项针对 **AI 生成内容**的法案，已在加利福尼亚州通过。该立法要求在某些情况下**披露 AI 生成的内容**，这可能会对 **LLAMA** 和其他 AI 语言模型产生影响。虽然对 LLAMA 的具体影响尚不确定，但该法案的通过可能需要改变 AI 生成内容的呈现和使用方式，特别是在商业和政治应用中。
  - 该法案 **1 亿美元的训练成本阈值**引发了关于其对**开源 AI** 影响的辩论。一些人认为它不会影响本地模型，而另一些人则认为它可能会影响像 **LLAMA 405B** 及其蒸馏版本这样的大型模型。
  - 批评者担心该法案可能会**扼杀创新**并有利于大公司。一些用户致电 **Newsom 州长办公室**反对 **SB 1047**，理由是担心不必要的监管和 AI 公司成本的增加。
  - 该立法要求对大型 AI 模型采取**安全措施**，包括**关停能力（shutdown capabilities）**、**第三方审计**和**举报人保护**。一些人认为这些是合理的预防措施，而另一些人则认为它们是对开源开发和言论自由的潜在威胁。

- **加州议会通过 SB 1047** ([Score: 165, Comments: 73](https://reddit.com//r/LocalLLaMA/comments/1f4jftq/california_assembly_passed_sb_1047/))：加州议会通过了 **SB 1047** 法案，该法案可能对**开源 AI 模型**产生重大影响。据报道，该立法包含要求模型作者具备**关停其模型**能力的条款，这可能导致最先进的 AI 模型难以开源，并可能使 AI 开发集中在少数几家公司手中。
  - 由于 **Meta** 总部位于加利福尼亚州，该公司可能面临重大挑战。用户推测该公司可能会**搬迁到西雅图**或**剥离子公司**以规避法律，而另一些人则认为他们可能会直接停止发布**开源模型**。
  - 据一段 [YouTube 视频](https://youtu.be/7PMUVqtXS0A)（20:15 处）称，受监管模型的 **1 亿美元训练成本**阈值是由 **Eric Schmidt 及其同事**任意确定的。一些用户认为，这项立法可能会将创新赶出加州，并有利于中国的 AI 发展。
  - 法律学者建议，由于加州经济地位重要，**在加州开展业务**的公司无论所在地在哪里，都需要遵守该法案。一些用户认为这是加州在**阻碍整个行业的发展**，而另一些人则认为这是**大型科技公司希望通过监管**来限制竞争。

## 其他 AI Subreddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 视频生成与视觉效果**

- **AI 生成的怪兽电影片段**：一段[展示 AI 生成的海怪场景的视频](https://www.reddit.com/r/singularity/comments/1f4iebb/fishing_for_megalodons_cousins_the_best_ai_video/)引发了关于 AI 视频生成现状的讨论。虽然令人印象深刻，但许多评论者指出它仍未达到好莱坞水准，并提到了物理规律、几何形状和人类反应方面的问题。

- **AI 电影即将来临**：一篇关于[即将上映的 AI 生成电影](https://www.reddit.com/r/singularity/comments/1f4fv05/ai_movies_are_coming/)的帖子受到了广泛关注，表明人们对 AI 对电影行业潜在影响的兴趣日益浓厚。

**AI 模型进展**

- **Magic 的 1 亿 token 上下文窗口**：[Magic 训练了一个拥有 1 亿 token 上下文窗口的模型](https://www.reddit.com/r/singularity/comments/1f4917u/magic_has_trained_their_first_model_with_a_100/)，相当于 1000 万行代码或 750 本小说，代表了模型上下文容量的重大进步。

**AI 安全与监管**

- **Anthropic 与美国 AI 安全研究所达成协议**：[Anthropic 已与美国 AI 安全研究所达成协议](https://www.reddit.com/r/singularity/comments/1f47y4n/sama_we_are_happy_to_have_reached_an_agreement/)，对其未来模型进行发布前测试，这标志着向更受监管的 AI 开发迈出了一步。

**AI 在游戏和交互式环境中的应用**

- **AI 玩 Minecraft**：一段[演示 AI 像人类一样玩 Minecraft 的视频](https://www.reddit.com/r/singularity/comments/1f4ap60/ai_playing_minecraft_with_me_like_a_human/)展示了 AI 在复杂的开放世界游戏环境中交互能力的提升。


---

# AI Discord 回顾

> 由 Claude 3.5 Sonnet 提供的总结之总结


**1. LLM 进展与基准测试**

- **Llama 3 登顶排行榜**：来自 Meta 的 **[Llama 3](https://lmsys.org/blog/2024-05-08-llama3/)** 在 **ChatbotArena** 等排行榜上迅速攀升至榜首，在超过 50,000 场对决中表现优于 **GPT-4-Turbo** 和 **Claude 3 Opus** 等模型。
   - 社区对 Llama 3 的表现感到兴奋，讨论了它对 AI 格局的潜在影响，以及它与专有模型的对比。
- **Grok 2 在代码生成方面表现出色**：讨论强调了 **Grok 2**、**Gemini** 和 **ChatGPT** 之间的性能对比，其中 Grok 2 在**代码生成**任务中被认为特别强大。
   - 用户推测了即将推出的模型（如 Grok 3），并对在强大硬件支持下可能实现的性能优势提出了疑问。
- **Word Game Bench 挑战 LLM**：新开发的 **[Word Game Bench](https://wordgamebench.github.io)** 作为一个基准测试，用于评估语言模型在 **Wordle** 等文字拼图游戏中的表现，目前没有模型能达到超过 **50% 的胜率**。
   - 该基准测试侧重于模型交互和推理，强调了 LLM 在动态、游戏化环境中所面临的挑战。
  


**2. 开源 AI 发展**

- **Re-LAION-5B 数据集发布**：**[Re-LAION-5B](https://laion.ai/blog/relaion-5b/)**（LAION-5B 数据集的清理版本）的发布受到了社区的欢迎，因为它解决了之前的安全顾虑。
   - 这个与关键机构合作创建的更新版数据集，标志着在确保大规模 AI 训练数据的安全性和合规性方面迈出了重要一步。
- **RunwayML 删除 Stable Diffusion 仓库**：**RunwayML** 删除了他们在 HuggingFace 和 GitHub 上的所有 **Stable Diffusion 1.5** 仓库，引起了用户的不满，并导致 Diffusers 1.5 中的功能失效。
   - 社区推测删除行为背后可能存在法律问题，强调了此类行为对开源 AI 生态系统的影响。
- **GameNGen：神经游戏引擎突破**：**[GameNGen](https://gamengen.github.io/)** 是第一个完全由神经模型驱动的游戏引擎，可以在单个 TPU 上以每秒超过 20 帧的速度模拟 **DOOM**，PSNR 达到 29.4。
   - 这一突破展示了神经模型在实时游戏模拟中的潜力，人类评分者很难区分真实游戏画面和模拟画面。
  


**3. 模型优化技术**

- **动态专家路由增强适应性**：讨论了允许模型在训练期间定义自己的专家，而不是使用固定配置的概念，作为提高适应性的一种方式。
   - 这一想法与正在进行的研究相关，例如 **[LayerSkip 论文](https://medium.com/@techsachin/layerskip-faster-llm-inference-with-early-exit-and-self-speculative-decoding-3110cb93c94e)** 中提出的方法，旨在提高模型性能和效率。
- **大型模型的量化技术**：讨论重点介绍了 **AQLM** 和 **QuaRot** 等量化技术，旨在在保持性能的同时，在单个 GPU 上运行大型语言模型 (**LLMs**)。
   - 成员们分享了实现细节和基准测试，例如在 RTX3090 上运行 **Llama-3-70b**，展示了这些优化方法的潜力。
- **有限标量量化 (FSQ) 作为 VQ-VAE 的替代方案**：讨论了引入 **有限标量量化 (FSQ)** 作为 VQ-VAEs 中传统矢量量化技术的一种潜在有效且更简单的替代方案。
   - 正如 [相关论文](https://arxiv.org/abs/2309.15505) 中所述，FSQ 方法有望在各种任务中提高性能，并对语言模型中的 token 利用产生影响。
  


**4. AI 部署与基础设施**

- **Tinygrad 推出实惠的云服务**：**Tinygrad** 宣布了一项新的云服务，仅需 **$60/月** 即可提供 **4090 GPU** 和 **500 GB 存储空间**，比 Vast AI 等竞争对手便宜 3 倍。
   - 该服务引入了 'CLOUD=1' 功能，允许用户在本地运行 Tinygrad，同时利用云端速度通过 10 步处理来增强性能。
- **OpenRouter 秘密发布并上线**：**OpenRouter** 成功上线，以 **$2.5/百万 tokens** 的竞争性价格提供支持 **128k 上下文**和 function calling 的 **Llama 3.1-405B-instruct**。
   - 团队强调建立可靠的基础设施而非基于推荐的补偿，突显了他们对服务质量和可访问性的关注。
- **Cohere 的 Command R 系列更新**：Cohere 宣布了更新后的 **Command R** 和 **R+** 模型，在推理、编程和多语言 RAG 方面的性能有所提升，现在以新的别名提供。
   - 更新后的模型具有更低的每 token 价格，其中 R 的输入 token 价格显著降低至 **$0.15**，展示了在性能和成本效益方面的进步。

---

# 第一部分：Discord 高层摘要




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **微调与 RAG 的辩论**：讨论表明，虽然 **RAG** 可能会减少幻觉，但在微调过程中，受控的过拟合至关重要。其有效性在很大程度上取决于数据集大小以及 rank 和 alpha 等超参数。
   - 参与者强调，这两种方法都没有明显的优劣之分，必须根据具体的项目需求定制这两种策略。
- **LLMs 的多样化用例**：LLMs 目前被应用于各个行业，如 **AT&T** 等公司将其用于客户支持，其他公司则用于专有研究应用。类似于 **GPT** 的指令型模型在部署领域占据主导地位。
   - 这些应用中展示的多功能性表明，将 LLMs 整合到实际日常运营中已成为一种强劲趋势。
- **OpenRouter 发布并立即投入运行**：**OpenRouter** 成功上线了 **Llama 3.1-405B-instruct**，具有 **128k 上下文**和 function calling 能力，价格极具吸引力，为 **$2.5/百万 tokens**。
   - 澄清说明开发者的报酬不受推荐链接使用的影响，而是专注于构建可靠的基础设施。
- **即将推出的模型和新的定价趋势**：围绕 **Meta** 即将发布的 **Llama 模型**的猜测引起了热议，尽管关于 **Llama 4** 的细节尚不清楚。与此同时， **OpenAI** 披露了其 **GPT-4o 模型**的降价信息，现在每 100 万 tokens 的成本为 **$4**。
   - 这些调整为开发者提供了一条优化成本的途径，同时可以访问更新的模型和功能，例如严格符合 JSON Schemas 的结构化输出。
- **关于微调目标的社区协作**：一位社区成员表达了在没有特定目标的情况下微调 LLM 的渴望，纯粹是为了乐趣。这种开放性突显了社区内的探索精神。
   - 这种心态可能会激励其他开发者在固定项目框架之外进行实验和创新。

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 模型引发褒贬不一的反应**：新的 **Gemini 模型** 因其声称的性能提升而引起轰动，但用户对其与 **Sonnet** 等现有模型相比的有效性仍持谨慎态度。
   - 怀疑主要集中在该模型在 Aider 场景中的实际效用，导致用户纷纷分享使用体验以进行验证。
- **Sonnet 表现持续稳定**：最近的基准测试确认 **Sonnet** 的性能保持一致，反驳了此前关于性能下降的猜测。
   - 基于其稳定的基准测试分数，用户对该模型的能力和可靠性表现出持续的兴趣。
- **Aider 的投资讨论升温**：社区内围绕 **Aider** 的潜在投资展开了热烈讨论，特别是需要一个更精细的 GUI 来扩大其可用性。
   - 建议包括使用用户生成的数据来增强排行榜功能，以更好地反映性能指标。
- **长上下文模型受到关注**：围绕能够处理 **1 亿 token** 的模型的讨论可能会显著影响编码工作流，**Magic dev** 等工具被提及为行业颠覆者。
   - 用户对这些模型在 AI 辅助开发中实际应用的兴趣持续增长。
- **Aider 缺乏 Swift 支持**：由于 **tree-sitter** 包的限制，目前 Aider 缺乏对 **Swift** 的支持，这让开发者感到沮丧。
   - 用户承认，为 Swift 添加后端支持可能需要额外的自定义开发工作。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **LLM 个性化受到关注**：成员们对语言模型的**个性化**表现出浓厚兴趣，提倡可定制的个性和**长期记忆**，以增强用户交互。
   - 出现了对高昂实现成本和维护复杂性的担忧，**RAG** (Retrieval-Augmented Generation) 等想法被视为潜在的解决方案。
- **利用 OpenAI API 构建聊天机器人**：社区讨论了利用 OpenAI API 进行自定义**聊天机器人开发**，涉及对编程技能的要求和适用场景。
   - 虽然出现了像 **Zendesk** 这样的无代码解决方案建议，但人们也承认了在自动化以及与 **Jira** 等系统集成方面的局限性。
- **Grok 2 在性能测试中脱颖而出**：讨论强调了 **Grok 2**、**Gemini** 和 **ChatGPT** 之间的性能比较，指出 Grok 2 在**代码生成**任务中表现尤为强劲。
   - 对即将推出的 Grok 3 等模型的猜测引发了兴奋，人们对其在强大硬件支持下可能具备的性能优势提出了疑问。
- **AGI 发展引发全球担忧**：参与者表达了对哪个国家可能率先实现 **AGI** 及其随之而来的权力转移影响的担忧。
   - 强调了美国保持技术领先地位以减轻全球稳定风险的必要性。
- **简历匹配评分的挑战**：一位用户报告了通过 API 提示词根据职位描述对简历进行评分的困难，并指出一个不相关的商务总监职位竟然得到了令人费解的 **65** 分。
   - 调整评分参数没有带来改善，不同工程角色之间依然存在严重的错位问题。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Inference Endpoints 宕机**：成员们报告了 **Inference Endpoints** 的问题，可能由于一个与支付方式相关的 bug 引起，由于生产环境网站依赖这些端点，修复工作迫在眉睫。
   - 已提交一个 Pull Request，团队表示正在处理该问题。
- **关于模型训练与性能的讨论**：用户探讨了使用各种模型训练对话数据的细微差别，讨论了结合 **system prompts** 与从上下文学习的效果。
   - 针对本地模型的 **VRAM** 限制问题，有人建议使用 **Colab** 以获得更强大的资源。
- **人类反馈对模型评估至关重要**：一篇论文强调 **human feedback** 对于训练 **Large Language Models** 至关重要，尽管会受到偏见的影响。
   - 研究人员指出，虽然偏好评分有助于评估，但它们往往无法代表 **factuality** 等关键方面 ([查看 PDF](https://arxiv.org/abs/2309.16349))。
- **LLM 中的高效层剪枝**：一项研究审查了 LLM 的层剪枝策略，发现直到移除 **多达一半** 的层时，性能退化才非常微小。
   - 该技术涉及 **parameter-efficient finetuning (PEFT)** 和 [quantization](https://arxiv.org/abs/2403.17887) 以在剪枝后恢复模型性能。
- **FLUX LoRA 训练简化**：一篇名为 [FLUX LoRA Training Simplified](https://youtu.be/nySGu12Y05k) 的指南指导用户如何使用 Kohya SS GUI 在 8GB GPU 上进行训练。
   - 该教程使初学者能够顺利开启他们的训练之旅。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Flash Attention 面临内存挑战**：用户正面临 **flash attention kernel** 中共享内存大小的挑战，特别是 Q 的大小需求达到了 **131,072 bytes**，这引发了对非 Hopper GPU 效率的担忧。
   - 在使用 NVIDIA GeForce RTX **3090** 进行测试时，用户在使用 Hugging Face 示例时遇到了 `OutOfMemoryError`，这表明当前软件包版本的内存管理存在挑战。
- **LayerNorm Kernel 更新提升性能**：随着 Liger Kernel 仓库中 [PR #169](https://github.com/linkedin/Liger-Kernel/pull/169) 的合并，LayerNorm 自定义 kernel 的集成已得到确认，并在 RTX 3090 上通过了正确性测试。
   - 进一步的讨论集中在原子操作的动态分派上，以优化多 GPU 设置中的性能。
- **回归 FP8 进行开发**：一位成员正回归到 **FP8** 代码开发，以巩固他们的理解并推进正在进行的项目，对早期的进展感到满意。
   - 这表明在预期进一步优化的当前环境下，重点是增强性能和兼容性。
- **L2 Side Aware 优化实现速度提升**：L2 Side Aware 代码在 GELU forward 中实现了 **1823GB/s** 的稳定速度，比早期 **x128** 配置的性能提升了 **2%**。
   - 尽管有所改进，但讨论指出需要进一步简化以维持优化并降低功耗。
- **社区质疑量化技术**：在讨论量化注意力层时，成员们对 QKV projections 的准确性表示担忧，建议需要改进策略以维持系统性能的延迟。
   - 值得注意的是，在使用浮点整数时发现 AWQ 性能下降的问题，引发了对高性能最佳实现的询问。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Flux 的 IP Adapter 引发褒贬不一的反应**：成员们讨论了近期推出的 **Flux IP Adapter**，并指出用户之间的性能反馈结果不一。
   - *尽管对其效果意见不一*，许多人仍对工具箱中增加这一新功能感到兴奋。
- **在有限 VRAM 下训练模型面临挑战**：分享了在 **RTX 3060** 上利用有限 VRAM 进行训练的经验，揭示了更高分辨率（如 **1024**）会消耗巨大的内存。
   - 有建议称降低分辨率会有所帮助，尤其是考虑到 **12GB RAM** 可能不足以处理复杂任务。
- **图像处理中的分割（Segmentation）引发疑问**：讨论强调了图像处理工作流中 **SEG (Segmentation)** 的概念，特别是它在 **ComfyUI** 等系统中的作用。
   - 成员们对其实现方式表示困惑，并质疑其相对于更简单替代方案的必要性。
- **RunwayML SD 1.5 仓库从平台消失**：**RunwayML** 已删除了 HuggingFace 和 GitHub 上所有的 **Stable Diffusion 1.5** 仓库，引发了关于此举影响的讨论。
   - 用户推测这是否标志着 **1.5 模型** 的终结，因为这些模型的使用率似乎已经下降。
- **SDXL 与 SD 1.5 引发辩论**：一位用户考虑从 **SD 1.5** 迁移到 **SDXL**，权衡其 GPU 的生成时间和存储需求。
   - 建议集中在利用命令行参数**优化性能**，以适配性能较弱的 GPU。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes 3 的失忆模式（Amnesia Mode）展现了专业性**：用户报告称 **Hermes 3** 中的“失忆模式”更倾向于专业性而非日常语言，限制了其对话的灵活性。
   - 一位用户表示沮丧，称该模型保持着“家庭友好”的风度，引发了对其预定义行为的猜测。
- **训练技术产生更好的 AI 输出**：讨论强调，与在指令微调期间加入用户输入相比，仅针对输出进行模型训练能获得更好的 Benchmark 结果。
   - 成员们一致认为，这种特定的训练方法增强了连贯性，并减少了不必要的“AI 味”回复。
- **梯度策略可降低通信成本**：一位用户提议在分布式训练中利用低秩近似（low-rank approximations）进行梯度同步，以最小化通信开销。
   - 这引发了关于有效结合各种优化技术以增强模型训练性能的讨论。
- **引入 Word Game Bench 用于 AI 评估**：新的“Word Game Bench”基准测试通过 Wordle 等文字拼图游戏捕捉语言模型性能，允许基于先前动作进行独特的交互。
   - 社区成员对其引人入胜的方法论以及评估模型行为的潜力表现出好奇。
- **GameNGen 变革游戏开发格局**：_GameNGen_ 作为首个神经模型游戏引擎，能够在不使用传统工具的情况下实现实时 **DOOM** 模拟，帧率超过 **20 fps**。
   - 人类评分者难以区分模拟画面与真实画面，展示了其先进的现实主义潜力。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **API 推理速度限制讨论**：一位用户提出了关于限制 API 推理速度的问题；另一位成员指出，使用不同模型处理多个请求是可行的。
   - 该用户更倾向于使用同一模型以节省 VRAM，但也意识到了其中的局限性。
- **用户对 LM Studio 0.3 版本的反馈**：针对最新的 LM Studio 更新出现了一些担忧，认为其导致 AI 响应能力下降以及出现异常的重复输出。
   - 成员们建议这可能与 Prompt 设置或模板解析有关，并建议进行微调以改进。
- **M2 Ultra Mac 已准备好进行开发**：一位成员配置了拥有 **192 GB** 统一内存的 **M2 Ultra Mac** 用于探索 LLM，并配备了 **2 TB** 硬盘进行存储。
   - 他们还使用一台单独的 PC 作为服务器来增强其开发环境。
- **在 RTX 4090 上探索 LLM 性能**：讨论重点是在 **6 张 RTX 4090** 上运行 **405b 模型**，受 offload 设置影响，产出速度约为 **每秒 1 个 token**。
   - 一位成员尝试了各种 GPU 配置，发现当模型分布良好时，内存链路可以提升速度。
- **PCIe 通道设置对性能的影响**：成员们讨论了在 gen4 x8 与 x16 设置下运行 **RTX 4090** 的情况，研究其对多 GPU 环境速度的潜在影响。
   - 虽然 gen4 x8 对于单 GPU 可能无关紧要，但在模型更密集的设置中可能会阻碍性能。

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini Flash 模型现在免费！**：**Gemini Flash 8B (EXP)** 模型现在可以通过[此链接](https://openrouter.ai/models/google/gemini-flash-8b-1.5-exp)使用，同时 **Gemini Flash Experiment** 也已确认免费，直到 **AI Studio** 的定价最终确定。
   - 用户庆祝 **Gemini Experimental 模型**的上线，这标志着迈向更广泛访问的重要一步。
- **为 Daun.ai 的发布欢呼！**：社区成员对 **Daun.ai** 的发布表示兴奋，认为它是 AI 工具领域一个值得关注的新成员。
   - 这种热情反映了开发者社区对创新 AI 解决方案日益增长的需求。
- **Cohere 模型更新引发关注**：**Cohere 的 Command R 模型**最近的更新引入了新功能和定价变化，在渴望探索这些增强功能的用户中引起了热议。
   - 提出了关于 **OpenRouter** 处理安全模式方式的担忧，突显了社区对安全实现的关注。
- **实验性模型遇到速率限制**：用户在尝试实验性模型时报告了 **rate limit 错误**，表明在高峰使用期间访问新功能存在挑战。
   - 随后引发了关于通过 **API** 管理安全设置的讨论，指出需要更清晰的文档。
- **对基础设施稳定性的担忧**：最近一系列归因于数据库容量的**停机问题**引起了社区的担忧，正在进行的升级被提议作为解决方案。
   - 开发者承认了这些停机的持续影响，并确保已制定计划以增强未来的稳定性。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **嵌入权重过早出现 NaN**：一位用户报告称，在训练开始几步后，**Embedding Weights** 就变成了 **NaN**，这可能是由于损失函数分母四舍五入为零造成的，并受到数据依赖衰减项的加剧。
   - 成员们追踪了梯度以更好地理解这种情况的复杂性，提供了关于损失函数优化的见解。
- **寻求关于压缩技术的见解**：Jeremy Vonderfecht 正在征求关于他使用 **Stable Diffusion** 等扩散模型压缩图像研究的反馈，并认识到协作的必要性。
   - 成员建议使用特定频道进行持续讨论，以促进建设性对话。
- **动态专家路由提升适应性**：讨论强调了动态专家路由 (**Dynamic Expert Routing**) 的潜力，允许模型在训练期间定义自己的专家，以增强适应性。
   - 这与正在进行的研究相关，例如 [LayerSkip 论文](https://medium.com/@techsachin/layerskip-faster-llm-inference-with-early-exit-and-self-speculative-decoding-3110cb93c94e)中的方法。
- **推出 Word Game Bench 以挑战模型**：**Word Game Bench** 是一个新的基准测试，用于评估语言模型在 **Wordle** 等文字游戏上的表现，目前没有模型的胜率超过 **50%**；它专注于动态交互。
   - 更多信息可以在 [Word Game Bench](https://wordgamebench.github.io) 和[推文公告](https://x.com/zafstojano/status/1829398835585520076)中找到。
- **应对 Tokenization 挑战**：参与者讨论了 **Tokenization** 的重大局限性，特别是在非拉丁语言方面，以及它对模型训练效率的影响。
   - 提出了关于 **Tokenization** 如何掩盖关键数据特征，从而导致优化变慢的担忧。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Discord 服务器庆祝成员突破 10 万！**：Discord 服务器正式达到 **100K members**，标志着社区的一个重要里程碑，并衷心感谢所有成员的支持。
   - 团队对持续增长表示兴奋，强调了每位成员的贡献丰富了小组的氛围。
- **用户反映 Pro API 额度缺失**：用户报告在购买 Pro 后未收到 **$5 PPLX API credits**，导致呼吁紧急支持以解决这些问题。
   - 成员们正在分享账户详情以便更快解决，强调了对 API 额度使用和可访问性的关注。
- **对 Pro Searches 功能的担忧**：关于通过 API 进行 **Pro Searches** 的功能存在不确定性，特别是对于运行 **llama-3.1-sonar-huge-128k-online** 的用户。
   - API 中缺少 **Pro** 选项让用户质疑该功能何时可用。
- **用户遇到 API Rate Limit 错误**：几位用户报告在访问 API 时遇到 **429 Client Error: Too Many Requests**，引起了对潜在使用上限的关注。
   - 这种情况预示着可能影响依赖稳定性能的工程师整体 API 功能的潜在问题。
- **关于 AI 模型行为和性能的反馈**：用户仔细检查了他们的 AI 模型，注意到即使在切换模型后输出仍不一致，这表明可能存在影响用户体验的 bugs。
   - 关于模型行为的疑问引发了围绕近期更新的讨论，表明需要明确输出和模型标识。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **MMLU 缺乏实际相关性**：成员指出 **MMLU** 与构建 **LLMs** 的实际效用相关性不强，强调了像弗洛伊德理论这样过时的例子，并评论了最近的模型刷新提高了来自互联网的数据相关性。
   - 这引发了关于评估 **LLM** 在现实场景中适用性的基准指标（benchmark metrics）未来的讨论。
- **Command R+ 的更新令人印象深刻**：Cohere 宣布了刷新的 **Command R** 和 **R+** 模型的显著性能提升，具有更好的多语言 **RAG** 和极具成本效益的每 input token **$0.15**。
   - 成员确认更新已在 [Hugging Face](https://huggingface.co/) 上可用，并指出在其他平台部署前需要进行 **quantization**。
- **Cohere 聊天界面保持不变**：用户对 **Cohere chat interface** 提出担忧，质疑更新是否与新模型功能同步，特别是缺少暗黑模式选项。
   - 对用户界面选项增强的呼吁表明，用户对改进模型交互体验的愿望日益增长。
- **API 试用密钥限制引发挫败感**：一名用户在使用试用 API key 时遇到 **rate limit error (429)**，抱怨 **1,000 API calls/month** 的限制，同行确认了生产 key 的必要性。
   - 讨论强调了优化 API 使用以增强性能和进行更广泛实验的重要性。
- **Maya LLaVA-Pretrain 数据集发布**：新发布的 **Maya LLaVA-Pretrain** 数据集包含跨 **8 languages** 的 **4,404,776** 条条目，专为预训练大模型开发，并通过机器翻译进行了扩展。
   - 成员们对解决与该数据集相关的 **batch processing** 和 API 能力的疑问表示感谢。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Codeium C 轮融资 1.5 亿美元**：Codeium 成功完成了由 General Catalyst 领投的 **1.5 亿美元**融资，目前估值达到 **12.5 亿美元**，自成立以来总融资额已达 **2.43 亿美元**。联合创始人 Varun Mohan 提到，他们尚未动用 **6500 万美元**的 B 轮资金。
   - 这一战略储备可能表明他们在应对市场需求时采取了谨慎的态度。
- **Meta AI Assistant 月活用户 (MAU) 突破 4 亿**：Meta 的 AI Assistant **月活跃用户数 (MAU)** 飙升至 **4 亿**，**日活跃用户数 (DAU)** 达到 **4000 万**，展示了其不断扩大的用户群和参与度。讨论强调，随着用户数量持续增长，许可授权可能变得必要。
   - 这些指标反映了极高的采用率，引发了关于未来扩展需求的讨论。
- **Google DeepMind 推出可定制的 Gems**：Google DeepMind 推出了 **可定制的 Gems**，这是针对特定领域（如 **Learning Coach** 和 **Coding Partner**）量身定制的 Gemini 模型专用版本。该计划旨在通过针对性的功能增强用户体验。
   - 反馈集中在这些 Gems 的有效性及其在现实场景中的可用性上。
- **Tome 转型专注于企业级 AI**：Tome 宣布转型为一款旨在帮助用户渗透新企业客户的 AI 助手，标志着其业务重心的重大转变。公司代表确认了这一消息，并概述了这一战略历程。
   - 成员们对这次转型将如何重新定义 Tome 的市场定位和目标表示了兴趣。
- **Nicholas Carlini 的新播客**：最新一期的 [Latent Space 播客](https://x.com/latentspacepod/status/1829173832877519152) 展示了来自 **Google DeepMind** 的 Nicholas Carlini 对 LLM 基准测试和训练数据提取方法论的见解。关键亮点包括对停止提供 *OpenAI logprobs* 的批判性观点。
   - Carlini 的思考引发了社区关于 AI 基准测试实践的对话。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 在区块链协议中的潜力**：关于将 **Mojo** 用于区块链协议的讨论正在进行中，开发者指出与 **Go, Rust 和 C++** 相比，它目前尚不成熟。
   - *一位开发者评论说，Mojo 和 Go 是最胜任的语言，但 Go 的 **20% 性能损失** 对某些项目来说可能至关重要。*
- **关于 Mojo 开源前景的疑问**：有人询问 **Mojo 编译器源码** 的可用性，目前该源码仍为闭源。
   - *Modular 团队表示，在平衡开发速度与社区参与的过程中，他们可能还不知道何时或是否会将其开源。*
- **性能对比见解**：成员们辩论了 **Go** 与 **C** 的性能，强调了 Go 在各种任务中的局限性。
   - *Darkmatter 指出 Go 的性能可能会显著下降，引用其每秒 **30 个请求** 的处理能力，而 **C** 为 **100 个**。*
- **架构师在内存管理中的角色**：一位成员认为，如果程序员不确定内存管理，这标志着系统设计存在缺陷。
   - *他们强调需要坚实的架构设计，以尽量减少应用程序员的顾虑。*
- **Fastai 令人兴奋的导出想法**：一项提议的增强功能涉及在 fastai 中重写 **Learner.export**，以便将 **Mojo** 代码与 **PyTorch 模型** 一起导出。
   - *这种策略可以改善输入流水线与模型之间的集成，从而简化生产环境的使用。*

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain 拥抱 Function Calling 与 Streaming**：一位成员在使用 **LangChain v2.0** 进行 Function Calling 和 Streaming 时遇到困难，并指出文档存在空白。另一位成员澄清说 Function Calling 是受支持的，但在 JavaScript 中需要仔细配置 Streaming 输出。
   - 探索 [AgentExecutor 文档](https://api.python.langchain.com/en/latest/agents/langchain.agents.agent.AgentExecutor.html) 等资源可能有助于理清配置。
- **Docker 轶事：Ollama 连接困扰**：一位用户在 Docker 中运行 LangChain 应用并尝试使用 **Ollama API** 时遇到了连接拒绝错误。随后他们通过将基础 URL 修正为直接的 **Ollama host URL** 解决了该问题。
   - 这一问题凸显了在容器化环境中正确设置 URL 的重要性，尤其是在利用 Docker 等工具时。
- **为 HR 打造自定义 GPT 激发创意**：一位用户表示希望为他们的 HR 团队创建一个专门的 **GPT**，目标是减少幻觉（hallucination）并建立反馈机制。讨论转向通过 Fine-tuning 和 **RAG** 技术增强 **LLM** 交互。
   - 实施反馈循环可以显著提高性能，尤其是在适配现有的手册内容时。
- **LangChain Streaming 输出的挑战**：一位用户报告了 LangChain Agent executors 的问题，即它们在交付最终响应之前会收集所有输出，而不是实时进行 Streaming。有建议提出利用 `streamRunnable` 选项来实现实时输出交付。
   - 利用此功能可以缩短响应时间，提升实时应用中的用户体验。
- **GraphRAG 对比传统 RAG：一场偏好之争**：围绕混合 **RAG** 方法的有效性展开了讨论，一位成员在流程中更倾向于**传统 RAG** 技术。他们指出，探索 Self-query 和 Large Context RAG 等新方法可能值得一试。
   - 这场对话可能为在 **RAG** 方法论中进行更高级的探索以增强响应能力打开了大门。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **GymNation 与 LlamaIndex 合作取得成功**：GymNation 与 LlamaIndex 合作，实现了数字化线索到销售转化率 **20% 的增长**，以及数字化线索 **87% 的对话率**。欲了解更多详情，请查看其[完整成功案例](https://t.co/CXsiySj4zq)。
   - *显著的成果*展示了 LlamaIndex 如何有效增强用户参与度。
- **LLMs in Production 见解分享**：即将在 **9 月 9 日**举行的讨论将分享关于有效部署 **LLM** 的见解。详情请见[此处](https://t.co/Ozb1xTF2Lh)。
   - 与会者可以期待关于现实世界 **LLM** 应用的*实用技巧*。
- **MLFlow 播客介绍 LlamaIndex**：联合创始人在播客中讨论了 **MLFlow** 与 LlamaIndex 的集成，重点关注简化的日志记录和应用评估。在此处观看演示和见解：[此处](https://t.co/2wwvn7HRBm)。
   - 会议展示了在管理 AI 应用方面的*强大增强功能*。
- **LLM x Law 黑客松宣布举办**：将于 **9 月 8 日**举行的 **LLM x Law Hackathon** 邀请参与者探索 AI 在法律实践中的应用。更多信息请见[此处](https://t.co/AksB9V6akr)。
   - 本次活动将设有*多个赛道*，强调 AI 与法律集成中的创新。
- **利用 MoW 进行财务数据分析**：讨论了采用 **Mixture of Workflows (MoW)** 和 Corrective **RAG** 的创新财务数据分析，利用了 **Phi-3**、**Qwen-2** 等模型。更多详情请见[此处](https://t.co/CIaEwmWB0S)。
   - 该方法提供了对财务报表的**上下文感知分析**，有望带来更好的洞察。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **下周 House Party**：欢迎参加下周提前举行的 **House Party**，以提升社区参与度！[加入 Discord 活动](https://discord.gg/open-interpreter-1146610656779440188?event=1278796923892924498)。
   - 该活动旨在营造有趣的氛围，并鼓励关于 **潜在应用** 的讨论。
- **寻求终端应用建议**：由于屏幕溢出/显示异常问题，一名成员正在寻找 KDE 上 **Konsole** 终端应用的替代方案。用户报告在标准终端设置中使用 **GPT-4o-mini** 时也遇到了类似问题。
   - 这凸显了在高需求环境下对终端性能的持续关注。
- **需要 Obsidian OI 插件安装帮助**：一位用户称赞了 **Obsidian OI 插件** 的资源，但正面临全局安装问题。他们被建议在指定频道分享安装细节以获取进一步支持。
   - 这反映了社区在解决技术挑战方面的协作努力。
- **GameNGen：游戏模拟的飞跃**：_GameNGen_ 现在使用神经模型以超过 **每秒 20 帧** 的速度模拟 **DOOM**，在单块 TPU 上展示了卓越的性能，PSNR 达到 **29.4**。
   - 这种体验让真人评分者难以区分真实游戏画面与其模拟画面，标志着游戏技术的重大进步。
- **对 AgentOps 进展的期待**：成员们对 **Adam 和 AgentOps** 团队即将推出的计划充满热情。这种兴奋体现了社区对下一代 Agent 技术突破的兴趣。
   - 这种期待预示着关于智能 Agent 系统未来前景的良性对话。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Google 采购 GPU 引发好奇**：成员们质疑为什么 **Google** 在拥有自家 **TPU** 的情况下仍从 **NVIDIA** 购买 **GPU**，这暗示了对 NVIDIA 技术的潜在缺口或兴趣。
   - *TPU 还不够吗？* 一位成员对 Google 在硬件方面的战略选择表示思考。
- **RunwayML 删除所有 Stable Diffusion 仓库**：关于 **RunwayML** 删除了他们在 **HuggingFace** 和 **GitHub** 上所有 **Stable Diffusion 1.5** 仓库的讨论爆发，令许多用户感到沮丧。
   - 一位成员指出，此举破坏了 **Diffusers 1.5** 中的许多功能，特别是影响了单文件加载。
- **仓库删除带来的混乱**：成员们对 RunwayML 这种看似草率的删除行为表示恼火，有人称这感觉就像他们想要制造 **混乱**。
   - 虽然出现了关于潜在法律问题的猜测，但尚未确认删除的具体原因。
- **为书封生成写实图像**：一位成员寻求关于为其小说封面生成 **漫画风格** 或卡通图像的建议，因为他们正苦于 **DALL·E** 输出的图像过于写实。
   - 尽管进行了尝试，他们发现 DALL·E 无法满足他们想要的特定风格。
- **Re-LAION-5B 发布**：成员们庆祝 **Re-LAION-5B** 的发布，这是 **LAION-5B** 数据集的清理版本，解决了之前 [安全修订程序](https://laion.ai/blog/relaion-5b/) 后的担忧。
   - 该数据集是与关键组织合作更新的，以确保安全性和合规性，标志着一个重要的里程碑。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **科技巨头看好 OpenAI**：Nvidia、Apple 和 Microsoft 正在讨论投资 **OpenAI**，作为新一轮 **$100 billion 融资**的一部分 [来源](https://www.bloomberg.com/news/articles/2024-08-29/nvidia-has-held-discussions-about-joining-openai-s-funding-round)。此举表明各大巨头对推动 AI 资金投入和创新的浓厚兴趣。
   - *Chatbot 战争正在升温*，这些公司正竞相在 AI 发展的关键领域占据一席之地。
- **Chatbot 战争白热化**：**ChatGPT** 的周活跃用户已突破 **2 亿**，对 **Meta AI** 等竞争对手构成了挑战，而后者也在不断提升市场吸引力 [来源](https://www.theinformation.com/articles/metas-ai-assistant-wins-millions-of-users-in-challenge-to-chatgpt?utm_source=ti_app&rc=c48ukx)。这种竞争格局引发了关于不同平台的用户参与度和有效性的讨论。
   - 针对 **Meta AI** 的真实利用率存在疑虑，因为仅有 **40 million DAU** 可能暗示用户是无意中接触到了其产品。
- **Tinygrad 推出高性价比云解决方案**：Tinygrad 推出了一项新的云服务，配备 **4090 GPU** 和 **500 GB 存储**，每月仅需 **$60**，价格远低于 Vast AI 等竞争对手 [来源](https://x.com/__tinygrad__/status/1829379908017238210?s=46)。这种新模式为寻求利用先进硬件的开发者提供了一个极具成本效益的解决方案。
   - *即将推出：CLOUD=1* 允许用户在本地运行 Tinygrad，同时利用云端处理速度进行高效处理。
- **关于 System Prompts 影响的探究**：成员们正在深入研究 **System Prompts** 对评估分数的影响，引发了关于不同 Prompting 技术是否能显著调整结果的兴趣。目前正征集相关研究论文以支持这一探索。
   - 这一探究突显了通过精心设计的 Prompt 来优化 AI 性能指标的持续需求。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **QLoRA 面临显存难题**：一名成员在拥有 **4 张 48GB GPU 卡**的情况下遇到了提示非法内存访问的 **CUDA error**，随后对 **QLoRA** 的内存充足性提出了质疑。
   - 这突显了在配置内存资源时，硬件设置中需要仔细考虑的潜在陷阱。
- **A6000 GPU 引起困惑**：澄清确认 **A6000 GPU** 已升级至 **48GB**，因此四张此类显卡应能满足所需容量。
   - 成员建议 CPU offloading 和序列长度调整可能会额外影响训练期间的内存分配。
- **训练序列长度受到关注**：一名成员尝试了不同的训练序列长度（**8K** 和 **4K**），展示了这些变化如何影响 **vRAM** 的使用。
   - 对这些细节的探究展示了在序列配置与内存需求之间进行平衡的重要性。
- **对多 GPU 评估的兴趣**：关于 **TorchTune** 是否支持 **multi-GPU evaluation** 的咨询表明了用户对优化性能的浓厚兴趣。
   - 这反映了一个更广泛的趋势，即 AI 工程师在处理高要求的训练设置时，不断追求可扩展性和效率。
- **调试 CUDA 错误以确保数据完整性**：一名成员收到了调试建议，例如设置 **CUDA_LAUNCH_BLOCKING=1**，以解决训练过程中出现的非法内存访问错误。
   - 这指向了在使用 **PyTorch** 执行分布式训练并有效管理内存限制时所面临的持续复杂性。

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **对仓库关联的困惑**：一位成员对他们的陈述与 [GitHub 仓库](https://github.com/feder-cr/linkedIn_auto_jobs_applier_with_AI) 之间的联系表示困惑，并澄清该仓库是独立的，展示它是为了激发社区参与。
   - *它每天获得超过 2000 个点赞*，表明人们对 **LinkedIn Auto Jobs Applier** 工具表现出浓厚兴趣。
- **对 LinkedIn 工具性能的担忧**：另一位成员对 **LinkedIn Auto Jobs Applier** 的性能表示担忧，并指出 GitHub Issues 显示该工具仍有改进空间。
   - 这突显了持续的反馈，表明该工具的能力仍有待加强。
- **可靠 AI Agent 研讨会**：一位成员分享了关于 **有用且可靠的 AI Agents** 研讨会的 [YouTube 视频](https://www.youtube.com/live/-aKRsvgDEz0) 链接，该研讨会重点讨论了准确性、可靠性和成本效益。
   - 该研讨会探讨了关于 AI Agent 的活跃研究及其在现实应用中的有效利用。
- **用于 AI 开发的 AgentOps 工具**：[AgentOps](https://agents.staf.ai/AgentOps) 提供构建 Agent 的资源，其工具通过消除 Prompt 过程中的猜测来简化开发流程。
   - 这种透明度旨在改进开发者构建 AI 解决方案的方式。
- **湾区 AI 见面会上的 DSPy 研讨会**：即将举行的湾区 AI 见面会将由 Michael Ryan 讨论 **DSPy: Prompt Optimization for LM Programs**，展示他在 MIPROv2 算法上的工作。
   - 该见面会由 [Neo4j](https://x.com/ChiefScientist/status/1829231009344434400?t=wow3U2BluHEv16-MI2YcaQ&s=19) 赞助，有望提供宝贵的见解。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Axolotl GitHub 文档需要深色模式**：一位成员请求 [Axolotl GitHub 文档](https://github.com/axolotl) 提供 **深色模式**，理由是频繁访问时当前的浅色模式令人不适。
   - 他们强调了在当前主题下检查配置参数的困难。
- **训练 LLaMA 70B 的硬件**：讨论围绕训练 **LLaMA 70B** 模型的 **硬件需求** 展开，推测可能只需要几个 **NVIDIA A6000 GPU**。
   - 一位成员确认 **3x A6000 GPU** 应该足以训练完整模型，突显了 GPU 能力的潜在进步。
- **Llama 3.1 仍受特殊 Token 困扰**：有人担心 **Llama 3.1 base** 仍存在未初始化的特殊 Token 和分布外 Embedding 的问题。
   - 成员们表示管理特殊 Token 仍面临挑战，这可能会影响模型性能。
- **未训练 Token 的潜在修复方案**：引入了一个新选项 `fix_untrained_tokens: true` 来解决 Llama 3.1 中未初始化的特殊 Token 问题，标志着改进迈出了一步。
   - 这一修复反映了在优化模型交互和性能方面的持续努力。
- **新的 Assistant Prefill 功能发布**：**Hugging Face** 最近的 [Pull Request #33198](https://github.com/huggingface/transformers/pull/33198) 添加了长期以来被要求的 **assistant prefill** 功能，该功能可自动启动模型响应。
   - 此次更新旨在提升 **TextGenerationPipeline** 的用户体验，采用了一种创造性的方式来生成响应。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Groq 等待排行榜 PR**：**Groq** 尚未被添加到排行榜中，因为团队仍在等待其 PR，预计下周左右完成。
   - 这一延迟引发了关于其集成和预期性能影响的讨论。
- **模型步骤文档至关重要**：一位成员确认，记录模型步骤对于可复现性至关重要，能增强模型的可理解性。
   - 完善的文档确保了可用性，并最大限度地减少了模型实现过程中的困惑。
- **Java 测试用例揭示 GIS 问题**：一位用户报告了与 GIS 几何初始化相关的 **Java** 测试用例性能问题。
   - 他们得出结论，鉴于用户查询的情况，简单的直接示例可能比复杂的函数调用效果更好。
- **关于评估温度设置的查询**：成员们询问评估是否使用 greedy decode 且温度为 0，以确保指标公平。
   - 讨论引用了最近关于排行榜评估标准的 GitHub 链接，并思考了输出中的随机性。
- **讨论 OSSHandler 默认参数**：**OSSHandler** 的默认温度设置为 0.001，曾简要考虑过调整但最终被否决。
   - 这一选择旨在保持一致的函数输出和整体模型性能优化。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **探索 tinygrad 的局限性**：*codeman3786* 询问 **tinygrad** 是否对 **statically scheduled operations** 有效，但在 **semi-structured sparsity** 选项上表现不佳。George Hotz 邀请提供 tinygrad 缺点的具体示例，这突显了社区对其运行极限的好奇。
   - 随后的讨论表明，大家共同关注剖析 tinygrad 在现实世界中的适用性，特别是在复杂数据处理的背景下。
- **Tensor.cat 在处理 sharded tensors 时的困扰**：一位用户在使用 **Tensor.cat** 处理 sharded tensors 时遇到问题，收到关于 *padding not supported* 的错误。他们设计了一个利用 `unsqueeze` 的变通方法，但额外的 reshape 错误不断出现。
   - 这表明需要明确该限制是源于核心功能还是仅仅是不支持的行为，因为用户正在考虑调整代码以支持 batch 维度。

---

**Alignment Lab AI Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

# 第 2 部分：频道详细摘要与链接

{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1278796860781232260)** (459 条消息 🔥🔥🔥): 

> - `Fine-Tuning vs RAG`
> - `LLM 使用案例`
> - `量化技术 (Quantization Techniques)`
> - `模型训练挑战`
> - `Hopfield Networks`

- **Fine-Tuning 与幻觉**：关于 RAG 在减少幻觉方面是否优于 Fine-Tuning 经常存在争论；然而，一些参与者认为两者都没有绝对的优势，受控的过拟合（controlled overfitting）是训练中必须考虑的关键因素。
   - Fine-Tuning 的效果受数据集大小和模型超参数（如 rank 和 alpha）的影响，这些参数定义了权重的训练方式及其对学习的影响。
- **LLM 的使用案例**：参与者讨论了 LLM 的各种应用，强调了像 AT&T 这样的公司利用模型进行客户服务，而其他公司则将其用于专有研究和搜索功能。
   - 有人指出，许多企业使用类似于 GPT 的指令型模型（instruction-based models），以便在实际任务中进行有效部署。
- **量化技术**：讨论了模型推理的量化类型，特别是目前对 4-bit 加载的支持，而 8-bit 支持仍然缺失。
   - 对话深入探讨了量化中不同 rank 大小的影响，其中较高的 rank 可能会在模型训练中提供更好的结果，特别是在稳定性和准确性方面。
- **模型训练中的挑战**：许多参与者表达了理解模型训练动态的重要性，强调了尝试不同技术以找到最佳配置的必要性。
   - 训练模型涉及大量的试错，分享成功方法的知识对于新手应对 Fine-Tuning 的复杂性至关重要。
- **Hopfield 网络与记忆**：Hopfield 网络被提及作为联想记忆（associative memory）的基础模型，一位参与者分享了一个讨论其原理和应用的 YouTube 视频。
   - 关于记忆衰减的幽默以及此类网络与新模型相比的效用，展示了神经网络讨论中怀旧与现代相关性的融合。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://mistral.ai/news/mathstral/">MathΣtral</a>：为了向阿基米德致敬（今年是我们庆祝他 2311 周年诞辰），我们自豪地发布了首个 Mathstral 模型，这是一个专门为数学推理和科学讨论设计的 7B 模型...</li><li><a href="https://arxiv.org/abs/2405.05904">Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?</a>：当大语言模型（LLM）通过监督微调进行对齐时，它们可能会遇到预训练期间未获取的新事实信息。人们通常推测这可能会教会模型...</li><li><a href="https://www.mercity.ai/blog-post/guide-to-fine-tuning-llms-with-lora-and-qlora">In-depth guide to fine-tuning LLMs with LoRA and QLoRA</a>：在这篇博客中，我们详细解释了 QLoRA 的工作原理，以及如何在 Hugging Face 中使用它来微调你的模型。</li><li><a href="https://tenor.com/view/fumo-touhou-fumo-touhou-gif-23545090">Fumo Touhou GIF - Fumo Touhou Fumo Touhou - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>：请参阅下面的列表以获取我们所有的 Notebooks：</li><li><a href="https://www.kaggle.com/code/mohsenghafari/kaggle-mistral-7b-unsloth">Kaggle Mistral 7b Unsloth &#x645;&#x62D;&#x633;&#x646; &#x6A9;&#x631;&#x647;</a>：使用 Kaggle Notebooks 探索并运行机器学习代码 | 使用来自“无附加数据源”的数据</li><li><a href="https://huggingface.co/posts/dylanebert/255000504996462">@dylanebert on Hugging Face: &quot;Here&#39;s a 1-minute video tutorial on how to fine-tune…&quot;</a>：未找到描述</li><li><a href="https://magic.dev/blog/100m-token-context-windows">100M Token Context Windows</a>：关于超长上下文模型的研究更新、我们与 Google Cloud 的合作伙伴关系以及新融资。</li><li><a href="https://tenor.com/view/orange-cat-smile-cat-smile-orenge-cat-smiling-gif-23133369">Orange Cat Smile Orenge Cat Smiling GIF - Orange Cat Smile Cat Smile Orenge Cat Smiling - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.anyscale.com/blog/fine-tuning-is-for-form-not-facts">Fine Tuning Is For Form, Not Facts | Anyscale</a>：微调是领域特定模型精炼（DSMR）的一种方法，但它并不是提高领域特定性能的万灵药。</li><li><a href="https://www.youtube.com/watch?v=1WPJdAW-sFo">A Brain-Inspired Algorithm For Memory</a>：在 https://shortform.com/artem 获取 20% 折扣。在此视频中，我们将探索 Hopfield 网络的概念——一种联想记忆的基础模型...</li><li><a href="https://github.com/facebookresearch/xformers#installing-xformers)">GitHub - facebookresearch/xformers: Hackable and optimized Transformers building blocks, supporting a composable construction.</a>：可黑客攻击且优化的 Transformers 构建块，支持组合式构建。 - facebookresearch/xformers</li><li><a href="https://github.com/mlabonne/llm-autoeval?">GitHub - mlabonne/llm-autoeval: Automatically evaluate your LLMs in Google Colab</a>：在 Google Colab 中自动评估你的 LLM。通过在 GitHub 上创建账号来为 mlabonne/llm-autoeval 做出贡献。</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/57">Benchmark against unsloth · Issue #57 · linkedin/Liger-Kernel</a>：🚀 功能、动机和推介。嘿，你有没有针对使用类似 Kernel 的 Unsloth 进行过基准测试？我猜你的项目可以作为具有多 GPU 支持的掉落式替代品。Alt.....</li><li><a href="https://x.com/BramVanroy/status/1827090122363564251">Tweet from Bram (@BramVanroy)</a>：@hsu_byron 这个稳定吗？如果是的话，与 @huggingface trainer 的下游集成将非常有价值 :o 我可能需要通过 accelerate，抄送 @TheZachMueller。</li><li><a href="https://ollama.com/unclemusclez/smollm-135m-instruct-devinator">unclemusclez/smollm-135m-instruct-devinator</a>：在 DEVINator 数据上训练的 SmolLM 135M Instruct，用于 Open Hands</li><li><a href="https://github.com/unslothai/unsloth/issues/636">Storing models to huggingface is not working · Issue #636 · unslothai/unsloth</a>：你好，我认为将模型存储到 Hugging Face 的说明不太清楚。Notebook 中的以下行尝试将模型推送到 HF 模型仓库 (&quot;hf/model&quot;, tokenizer, quantization_m...</li><li><a href="https://github.com/linkedin/Liger-Kernel?trk=public_pos">GitHub - linkedin/Liger-Kernel: Efficient Triton Kernels for LLM Training</a>：用于 LLM 训练的高效 Triton Kernels。通过在 GitHub 上创建账号来为 linkedin/Liger-Kernel 做出贡献。</li><li><a href="https://github.com/e-p-armstrong/augmentoolkit">GitHub - e-p-armstrong/augmentoolkit: Convert Compute And Books Into Instruct-Tuning Datasets (or classifiers)!</a>：将算力和书籍转换为指令微调（Instruct-Tuning）数据集（或分类器）！ - e-p-armstrong/augmentoolkit</li>

oolkit</li><li><a href="https://github.com/linkedin/Liger-Kernel?trk=public_post_comment-text">GitHub - linkedin/Liger-Kernel: Efficient Triton Kernels for LLM Training</a>: 用于 LLM 训练的高效 Triton 内核。通过在 GitHub 上创建账号来为 linkedin/Liger-Kernel 的开发做出贡献。</li><li><a href="https://github.com/huggingface/lighteval">GitHub - huggingface/lighteval: LightEval is a lightweight LLM evaluation suite that Hugging Face has been using internally with the recently released LLM data processing library datatrove and LLM training library nanotron.</a>: LightEval 是一个轻量级的 LLM 评估套件，Hugging Face 内部一直在将其与最近发布的 LLM 数据处理库 datatrove 和 LLM 训练库 nanotron 配合使用。</li><li><a href="https://github.com/unslothai/unsloth/pull/974">Fix for multi gpu setup training with a single GPU. by Sehyo · Pull Request #974 · unslothai/unsloth</a>: check_nvidia() 最初会为 nvidia-smi 派生一个新进程，从而绕过了 GPU 数量可能受 OS 环境变量限制的情况，因为这不会反映在新进程中。添加...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1278854896661168159)** (12 条消息🔥): 

> - `简易 AI 训练脚本`
> - `Meta 即将推出的模型`
> - `OpenAI 对 GPT-4o 的新定价`
> - `Gemini 2.0 更新`
> - `LLM 提供商作为云服务` 


- **使用单个脚本简化 AI 训练**：一名成员正在创建 **2 个脚本**，允许任何人在本地或云端设置上轻松训练 AI，而无需使用 Unsloth 或 Deepspeed 等复杂库。
   - 这些脚本只需要极少的依赖项，运行它们的具体指令已与 [text generation web UI](https://github.com/oobabooga/text-generation-webui) 的链接一起分享。
- **Meta 即将发布的模型揭晓**：讨论关于 **Meta** 可能很快宣布更新和下一代 **Llama 模型**，尽管目前尚不清楚是否包含 **Llama 4**。
   - 推测认为此次发布可能包含 **多模态 Chameleon-type 模型**。
- **OpenAI 的新 GPT-4o 定价**：新的 **GPT-4o 模型** 已经发布，成本显著降低，输入为 **每 1M tokens 4$**，输出 tokens 便宜了 **33%**。
   - 该模型还支持 **Structured Outputs**，允许输出严格遵守 JSON Schemas。
- **Gemini 2.0 引起关注**：**Gemini 2.0** 被提及并引发兴奋，暗示它可能与 **AI Studio** 内的实验性模型有关。
   - 一位用户指向了一篇讨论 Gemini 2.0 新功能的 [Reddit 帖子](https://www.reddit.com/r/Bard/comments/1f4xamv/wow_gemini_20/)。
- **LLM 提供商作为 App Store 模式**：一位用户将 Anthropic 和 OpenAI 等 **LLM 提供商** 比作 **App Store** 模式，暗示他们更希望开发者创建应用程序，而不是从销售中抽成。
   - 这引发了关于与 **Firebase** 等 **云服务** 相似性的讨论，表明了模型访问货币化的更广泛趋势。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/OpenAIDevs/status/1820987573793386527?utm_campaign=The+Batch&utm_source=hs_email&utm_medium=email">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>: 我们最新的 GPT-4o 模型输入 token 便宜了 50%，输出 token 便宜了 33%。它还支持 Structured Outputs，确保模型输出完全符合您的 JSON Schemas。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1f43ep8/meta_to_announce_updates_and_the_next_set_of/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/Bard/comments/1f4xamv/wow_gemini_20/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://github.com/oobabooga/text-generation-webui">GitHub - oobabooga/text-generation-webui: A Gradio web UI for Large Language Models.</a>: 一个用于 Large Language Models 的 Gradio Web UI。通过在 GitHub 上创建账号来为 oobabooga/text-generation-webui 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1278799638295744522)** (67 条消息🔥🔥): 

> - `Learning Rate Scheduler`
> - `GPU Rental vs. Ownership`
> - `DPO Model RAM Optimization`
> - `Fine-tuning Parameters`
> - `Tokenizer Management` 


- **理解 Learning Rate Scheduler 的效果**：一位成员询问带有 Warmup 步数的 Cosine Learning Rate Scheduler 如何影响训练期间的 LR 图表。
   - 讨论强调了观察 Learning Rate 优雅衰减对于获得更好模型性能的重要性。
- **关于租赁与购买 GPU 的辩论**：成员们深入探讨了租赁 GPU 相对于购买 GPU 的优势，认为租赁在运营成本上显著更低。
   - 一位用户强调，租赁选项提供了灵活性和成本效益，特别是对于偶尔使用的情况。
- **有限 RAM 下 DPO 模型的优化技巧**：几位成员讨论了在 RAM 有限的系统（如 16GB 的 Colab T4）上尝试运行 DPO 模型时遇到的 Out-of-Memory (OOM) 错误。
   - 通用建议包括减小 Batch size 和 Sequence length，但一些人指出 DPO 模型比常规 Fine-tuning 需要更多的 VRAM。
- **Fine-tuning 的参数调优**：一位用户寻求关于如何选择模型训练参数的澄清，特别是基于可用显存的 Batch size 选择。
   - 见解指出，在严格的内存限制下工作时，可能需要较低的 Batch size，特别是在处理需要更长 Context lengths 的模型时。
- **训练后的 Tokenizer 管理**：关于何时在训练模型后推送 Tokenizer 的问题引起了讨论，共识是仅在添加新 Tokens 时才推送更改。
   - 成员们讨论认为，如果 Tokenizer 在训练期间保持不变，则没有必要推送更新。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/llama3-1">使用 Unsloth 微调 Llama 3.1</a>：通过 Unsloth 微调并运行 Meta 更新的 Llama 3.1 模型，支持 6 倍长的 Context lengths！</li><li><a href="https://hastebin.com/share/ilelinosan.python">Hastebin</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1278885321509179485)** (4 条消息): 

> - `OpenRouter Launch`
> - `Llama 3.1 Model` 


- **OpenRouter 隐身发布正式上线！**：经过数周的努力，该产品现已在 [OpenRouter](https://openrouter.ai/models/meta-llama/llama-3.1-405b-instruct/providers) 上线并迎来真实用户，提供支持 **128k context** 和 Function calling 的 **Llama 3.1-405B-instruct** 服务。
   - 价格为 **$2.5/mil tokens**，是目前最便宜的选择。
- **澄清付费情况**：该成员澄清说，无论用户是否通过其链接访问服务，他们都会收到报酬，并强调了对构建基础设施的自豪感。
   - 提到 *“我没有赚取任何额外的钱、佣金或推荐费”*，以强调重点在于付出的努力而非佣金。



**提到的链接**：<a href="https://openrouter.ai/models/meta-llama/llama-3.1-405b-instruct/providers">Meta: Llama 3.1 405B Instruct – 供应商状态</a>：查看供应商状态并向 Meta: Llama 3.1 405B Instruct 发起负载均衡请求 —— 备受期待的 400B 级 Llama3 来了！拥有 128k context 和令人印象深刻的评估分数...

  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/)** (1 条消息): 

hamchezz: 我想纯粹因为好玩，在一些未定义的目标上 Fine-tuning 一个 LLM 😄
  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1278793009789534239)** (285 messages🔥🔥): 

> - `Gemini Model Performance` (Gemini 模型性能)
> - `Sonnet Benchmark Updates` (Sonnet 基准测试更新)
> - `Investment in Aider` (对 Aider 的投资)
> - `Long Context Models` (长上下文模型)
> - `Coding with AI Tools` (使用 AI 工具编程)


- **关于 Gemini 模型的讨论**：新的 Gemini 模型引发了热议，尽管一些用户对其在 Aider 中的实际效果（与其他模型相比）持怀疑态度。
   - 用户们渴望验证其性能提升的说法，同时分享了关于其实际应用的经验和怀疑。
- **Sonnet 性能更新**：最近的基准测试表明，尽管有传言称其性能下降，但 Sonnet 依然表现良好，没有明显的退化。
   - 用户仍然对 Sonnet 的能力感兴趣，特别是其当前的性能指标。
- **对 Aider 的潜在投资**：社区成员推测 Aider 的未来和潜在的投资兴趣，并思考开发一个精美的 GUI 版本以吸引更广泛受众的好处。
   - 有人建议可以通过整合用户生成的数据来改进 Aider 的排行榜功能，从而提供更准确的性能评估。
- **长上下文模型的探索**：关于能够处理 1 亿 token 推理的模型的讨论正在进行中，这可能会改变编程任务和 AI 集成。
   - 用户对 Magic dev 等新兴工具及其对未来 AI 辅助软件开发的影响表示好奇。
- **AI 工具对编程职业的影响**：微软 CEO Satya Nadella 强调了 GitHub Copilot 的成功，指出它在总用户贡献方面已经超过了之前的 GitHub 收入基准。
   - 讨论强调了开发者对 AI 工具日益增长的依赖，突出了它们对生产力和编程效率的影响。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/openaidevs/status/1823510395619000525?s=46">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>：此模型现在也已在 API 中作为 `chatgpt-4o-latest` 提供。我们建议大多数 API 使用场景采用 `gpt-4o-2024-08-06`，但很高兴能让开发者测试我们最新的改进...</li><li><a href="https://aider.chat/2024/08/26/sonnet-seems-fine.html">Sonnet 似乎一如既往地出色</a>：Sonnet 在 aider 代码编辑基准测试中的得分自发布以来一直保持稳定。</li><li><a href="https://tenor.com/view/dancing-cat-dance-cat-cat-meme-chinese-cat-gif-12629347036627000898">跳舞猫 GIF - Dancing cat Dance Cat - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/homer-brain-monkey-gif-11098413">荷马大脑 GIF - Homer Brain Monkey - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://docs.continue.dev/features/codebase-embeddings">代码库检索 | Continue</a>：与你的代码库对话</li><li><a href="https://magic.dev/blog/100m-token-context-windows">1 亿 Token 上下文窗口</a>：关于超长上下文模型的研究更新、我们与 Google Cloud 的合作伙伴关系以及新融资。</li><li><a href="https://github.com/nu">Nu Deployment</a>：Nu Deployment 有 3 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/nus-apr/auto-code-rover">GitHub - nus-apr/auto-code-rover：一个具备项目结构感知能力的自主软件工程师，旨在实现自主程序改进。在 SWE-bench lite 中解决了 30.67% 的任务 (pass@1)，在 SWE-bench verified 中解决了 38.40% 的任务 (pass@1)，每项任务成本低于 $0.7。</a>：一个具备项目结构感知能力的自主软件工程师，旨在实现自主程序改进。在 SWE-bench lite 中解决了 30.67% 的任务 (pass@1)，在 SWE-bench verified 中解决了 38.40% 的任务 (pass@1)...
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1278853802145091595)** (69 条消息🔥🔥): 

> - `Aider 与 Swift 语言支持`
> - `在 Aider 中自动化命令输入`
> - `Aider 中的文件检测`
> - `仓库大小对 Aider 性能的影响`
> - `在 Aider 中使用 GitHub Copilot` 


- **Aider 在 Swift 语言支持方面遇到困难**：一位用户询问如何为 Aider 添加 **Swift** 支持，但另一位成员指出 **tree-sitter** 软件包无法解析 Swift 文件。他们引用了相关文档，表明 Aider 在某些语言上存在局限性。
   - 进一步的讨论使人们意识到，为新语言增强 repo-map 可能需要额外的努力或自定义实现。
- **在 Aider 中自动化命令**：一位成员对 Aider 仅提供命令列表而不直接执行感到沮丧，并将其与 Cursor Compose 功能进行了对比。建议他们使用 *Sonnet* 或 *gpt-4o* 等不同的 LLM 模型以获得更好的效果。
   - 有人指出，使用 `aider --deepseek` 有助于简化某些流程，但用户仍然希望获得更集成的体验。
- **在 Aider 中自动检测文件**：一位用户询问如何刷新 Aider 以自动检测新创建的文件，而不是使用 `/add` 命令。尽管讨论了 `/drop` 和 `/clean` 等命令，但结论是仍有必要通过 `/add` 手动添加。
   - 几位用户确认，一旦文件是最近创建的，自动补全功能可以建议这些文件，但指出可能存在一些与 git 相关的限制。
- **仓库大小与 Aider 性能**：一位用户提出了关于 Aider 在何种规模下会难以应对仓库复杂性的问题，引发了关于 *Wine* 和区块链代码库等大型仓库使用经验的讨论。成员们强调，在大型仓库中进行更改时，管理关注点（focus）至关重要。
   - Aider 在处理与任务相关的特定文件时表现更好，鼓励用户避免向模型提供不必要的文件，以保持效率。
- **在 Aider 中使用 GitHub Copilot API 的可能性**：一位用户询问 Aider 理论上是否可以使用 **GitHub Copilot API**，因为他们的组织已批准使用 Copilot，但尚未批准其他 LLM。这突显了组织对各种 AI 工具审批流程的复杂性。
   - 在 Aider 中结合使用像 Copilot 这样被广泛接受的工具，可能为在企业环境中实现更灵活的集成铺平道路。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/llms/warnings.html">模型警告</a>：aider 是你终端里的 AI 配对编程助手</li><li><a href="https://openrouter.ai/settings/keys">Keys | OpenRouter</a>：管理你的密钥或创建新密钥</li><li><a href="https://aider.chat/docs/usage/tips.html">技巧</a>：使用 aider 进行 AI 配对编程的技巧。</li><li><a href="https://aider.chat/docs/faq.html#can-i-use-aider-in-a-large-mono-repo">常见问题</a>：关于 aider 的常见问题。</li><li><a href="https://aider.chat/docs/config/dotenv.html">使用 .env 配置</a>：使用 .env 文件为 aider 存储 LLM API 密钥。</li><li><a href="https://docs.litellm.ai/docs/providers">提供商 | liteLLM</a>：了解如何在 LiteLLM 上部署和调用来自不同提供商的模型</li><li><a href="https://aider.chat/docs/languages.html#how-to-add-support-for-another-language">支持的语言</a>：Aider 几乎支持所有流行的编程语言。</li><li><a href="https://github.com/paul-gauthier/grep-ast/issues/7">`py-tree-sitter-languages` 已停止维护 · Issue #7 · paul-gauthier/grep-ast</a>：你好 @paul-gauthier，感谢你在 aider 上的工作。我一直用得很开心。这个项目使用了 https://github.com/grantjenks/py-tree-sitter-languages，但该项目已停止维护且……</li><li><a href="https://github.com/ChimeHQ/SwiftTreeSitter">GitHub - ChimeHQ/SwiftTreeSitter: tree-sitter 增量解析系统的 Swift API</a>：tree-sitter 增量解析系统的 Swift API - ChimeHQ/SwiftTreeSitter
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1278937913689899091)** (1 条消息): 

> - `Anthropic Prompt Engineering`
> - `Jupyter Notebooks`
> - `uvx tool`
> - `Anthropic API`
> - `Documentation quality` 


- **探索 Anthropic 的 Prompt Engineering 教程**：查看 [Anthropic 的 Prompt Engineering 交互式教程](https://simonwillison.net/2024/Aug/30/anthropic-prompt-engineering-interactive-tutorial/)，该教程通过 Jupyter notebooks 展示了其卓越的文档编写能力。
   - 据指出，Anthropic 在 LLM 供应商中的**文档质量**（documentation quality）持续保持领先。
- **使用 uvx 轻松设置 Jupyter**：描述了使用 **uvx** 实现 Jupyter notebooks 的过程，展示了如何通过运行几条命令快速设置服务器。
   - 使用 `git clone` 随后运行 `uvx --from jupyter-core jupyter notebook courses` 启动了 Jupyter 服务器，并几乎瞬间打开了浏览器。
- **通过 Anthropic API 进行基础 Prompt 演示**：教程从基础章节开始，展示了通过 **Anthropic API** 执行的基础 Prompt，并使用 `%pip install anthropic` 进行包管理。
   - 这强调了在正确的虚拟环境中保持安装整洁的重要性。
- **参与 Anthropic 社区**：一位用户通过在 GitHub 课程仓库提交 issue 和创建 pull request，积极为 Anthropic 社区做出贡献。
   - 这展示了社区参与和协作在软件开发中的重要性。



**提到的链接**：<a href="https://simonwillison.net/2024/Aug/30/anthropic-prompt-engineering-interactive-tutorial/">Anthropic’s Prompt Engineering Interactive Tutorial</a>：Anthropic 继续保持其在领先 LLM 供应商中提供最佳文档的趋势。本教程以一组 Jupyter notebooks 的形式呈现 —— 我使用了它……

  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1278796810671882260)** (318 条消息🔥🔥): 

> - `Personalization of LLMs`
> - `OpenAI API for chatbots`
> - `Grok 2 performance`
> - `AGI development concerns`
> - `Creating custom AIs` 


- **关于 LLM 个性化的讨论**：成员们强调了对 AI **个性化**（personalization）的需求，例如可定制的性格和用于有意义互动的长期记忆。他们讨论了以用户友好方式实现这些功能的可行性和挑战。
   - 有人对维护个性化 AI 的潜在高成本和复杂性表示担忧，并考虑了 **RAG** (Retrieval-Augmented Generation) 等方案。
- **使用 OpenAI API 开发聊天机器人**：随后展开了关于使用 OpenAI API 构建自定义聊天机器人的对话，强调了编程技能和对特定用例理解的需求。成员们指出了像 **Zendesk** 这样的现有无代码解决方案，但也承认其在自动化和支持能力方面的局限性。
   - 概述了高效聊天机器人的关键功能，包括本地 vector databases 以及与 Jira 和 Sharepoint 等现有系统的集成。
- **AI 模型性能对比**：用户对比了包括 **Grok 2**、**Gemini** 和 **ChatGPT** 在内的各种模型的性能，并注意到了代码生成能力方面的差异。有人认为 Grok 2 的效果出奇地好，而一些成员则对模型在特定编码任务上的输出表示失望。
   - 社区推测了新模型（如 Grok 3 等）的即将发布，考虑了它们的潜在性能以及大规模硬件设置的优势。
- **对 AGI 发展的担忧**：参与者对哪个国家率先实现 **AGI** 的影响表示担忧，特别是关于全球权力动态。大家一致认为，应仔细监控 AGI 的发展，以防止任何实体形成垄断。
   - 讨论强调了美国等国家在 AI 技术方面保持领先地位的必要性，以防止对全球稳定产生任何不利影响。
- **创建自定义 AI**：成员们分享了关于如何创建自定义 AI 的见解，建议在挑战 LLM 之前先从简单的项目开始。推荐的资源包括 **TensorFlow**、**Colab** 以及像图像放大器（image upscalers）这样对初学者友好的模型。
   - 鼓励个人专注于编程技能和 AI 开发的基础知识。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/)** (1 条消息): 

smilebeda: 👍
  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1279017909661863967)** (16 条消息🔥): 

> - `Prompt Engineering 讨论`
> - `职位描述匹配`
> - `用于文档分析的 API 实用程序`
> - `深度文档分析`
> - `Batch Processing` 


- **职位描述匹配评分**：一位用户描述了通过 Prompt 对简历与职位描述进行评分时面临的挑战，并指出了 API 返回非预期相似度评分的具体案例。
   - 其中一个例子是，一名物联网（IoT）专业的工程系学生在申请商务总监职位时，竟然获得了 **65** 分。
- **文档分析的 API 设计**：另一位用户询问在从大型文档中提取摘要和申请信息等各种细节时，应该使用多次 API 调用还是单个 Prompt。
   - 有建议指出，分开请求有助于减少 Hallucinations（幻觉）并增强连贯性。
- **Batch Processing 讨论**：一位社区成员建议探索 [Batch Processing](https://platform.openai.com/docs/guides/batch/getting-started) 以提高效率。
   - 背景包括通过分别处理问题来降低响应复杂度的讨论。
- **寻求深度文档分析讨论**：一位用户表示有兴趣讨论深度文档分析技术，并计划在收集足够的 ChatGPT 数据后进行 Fine-tuning。
   - 他们询问了社区内关于该话题的可用讨论空间。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1279017909661863967)** (16 条消息🔥): 

> - `用于简历匹配的 Prompt Engineering`
> - `使用 ChatGPT 进行文档分析` 


- **Prompt 调整导致评分错误**：一位用户调整了用于根据职位描述评估简历的 Prompt，但仍然收到不准确的相似度评分，例如一名工程系学生在商务总监职位上获得了 **65** 分。
   - 增加严格的评分规则也没有帮助，一名 Cloud Engineer 尽管有相关经验，但由于职位重点不匹配，仅获得了 **5** 分。
- **通过分开 API 调用减少幻觉**：一位用户询问是否应使用多个查询从大型文档中提取信息，得到的建议是分开请求可以最大限度地减少 Hallucinations 的几率。
   - 有人指出，更大、更复杂的 Prompt 可能会阻碍连贯的响应，这支持了将查询拆分为更小、更清晰片段的想法。
- **探索 Batch Processing 以提高效率**：一位用户提到了在 API 调用中使用 Batch Processing 以简化操作的潜在好处，并提供了一个有用的[链接](https://platform.openai.com/docs/guides/batch/getting-started)作为指导。
   - 另一位用户表示有兴趣将 ChatGPT 的响应作为 Fine-tuning 的起点，表明了改进文档分析的长期目标。
- **参与深度文档分析讨论**：一位用户询问了讨论深度文档分析的平台，特别是关于为模型 Fine-tuning 收集数据方面。
   - 他们被引导至专门讨论该话题的频道，显示了社区对其探索的支持。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1278799856135311360)** (223 条消息🔥🔥): 

> - `Inference Endpoints 问题`
> - `模型训练`
> - `视频处理`
> - `LLMs 与 AI 项目`
> - `AI 驱动的应用` 


- **Inference Endpoints 宕机**: 成员们报告了 Inference Endpoints 的问题，可能与支付方式相关的 bug 有关，由于生产环境网站依赖这些服务，修复工作迫在眉睫。
   - 已提交一个 Pull Request 来解决此问题，并收到了正在调查该问题的回复。
- **关于模型训练与性能的讨论**: 用户探讨了使用各种模型训练对话数据的细微差别，讨论了加入 system prompts 与从对话上下文中学习的有效性。
   - 针对由于 VRAM 限制而在本地运行模型的局限性提出了担忧，并建议使用 Colab 以获得更强大的资源。
- **视频处理与上传的挑战**: 一位成员分享了将视频文件切分为较小尺寸以进行 Hugging Face 上传的策略，并承认了使用 Git LFS 时文件大小的限制。
   - 小组讨论了视频处理速度和资源使用的经验，指出了运行某些模型时遇到的挑战。
- **AI 驱动应用的探索**: 成员们对 AI 的实际应用表现出兴趣，例如通过模型训练实现身份证创建的自动化。
   - 分享了关于将 AI 与其他技术集成的见解，展示了 AI 在现实项目中的潜在想象力用途。
- **氛围与社区参与**: 成员们庆祝了他们的成就，并对 AI 开发项目表现出极大热情，促进了同僚情谊与协作。
   - 对话强调了实验 AI 的乐趣和兴奋感，并提到了钢铁侠中的 J.A.R.V.I.S 等流行文化标志，激发了创造力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/NM_Reid/status/1825997577151525338">来自 Noah Reid (@NM_Reid) 的推文</a>: 呃，Anaconda 刚刚给我们的 HPC 管理员发了消息，说我们违反了他们的服务条款（ToS），现在需要支付许可费用或从我们的系统中移除他们所有的软件？</li><li><a href="https://pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch.Tensor.to">torch.Tensor.to &mdash; PyTorch 2.4 文档</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/hub/repositories-licenses">Licenses</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=IW7jFq3vQbw">AI 处理数千个视频？！- SAM2 深度解析 101</a>: 构建你自己的 SAM2 AI 来分析/编辑视频剪辑。下载免费 Python 入门电子书: https://clickhubspot.com/1sf7🔗 链接 - 获取完整代码解析 &amp; J...</li><li><a href="https://ollama.com/unclemusclez/smollm-135m-instruct-devinator">unclemusclez/smollm-135m-instruct-devinator</a>: 为 Open Hands 在 DEVINator 数据上训练的 SmolLM 135M Instruct</li><li><a href="https://youtu.be/0Ef7K18Eyxc">Pandas : 分组与排序</a>: 在本视频中，我将通过一些示例和代码讨论如何在 pandas 中进行分组或排序。如果你想查看资源或代码，请检查仓库...</li><li><a href="https://youtube.com/shorts/c1QI7r9AP_g?si=GWgdeHiWcPm9DfvE">TCP TIME_WAIT 导致 "address already in use" 错误</a>: 针对 SDE-2 及以上的系统设计: https://arpitbhayani.me/masterclass 针对初学者的系统设计: https://arpitbhayani.me/sys-design Redis 内部原理: https:/...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1278900366250872914)** (4 messages): 

> - `模型训练中的 Human Feedback`
> - `Low Bit Quantisation`
> - `AI 模型的训练要求` 


- **Human Feedback 对模型评估至关重要**：最近的一篇论文讨论了 **human feedback** 在评估和训练 **Large Language Models** 中如何变得不可或缺，但也可能受到主观偏见的影响。
   - 论文强调，虽然偏好评分涵盖了许多方面，但它们对事实性（factuality）等**重要标准代表性不足** ([查看 PDF](https://arxiv.org/abs/2309.16349))。
- **Low Bit Quantisation 的探索**：一位成员提到他们关注 **low bit quantisation**，并引用了该主题的一篇基础论文。
   - 这种技术对于在保持效率的同时优化模型至关重要 ([阅读论文](https://arxiv.org/pdf/1609.07061))。
- **训练 AI 模型需要 GPU**：有人提出建议，强调 **training AI models** 不应在没有 **GPU** 的情况下进行，并推荐了 **Colab** 和 **Kaggle** 等平台。
   - 明确坚持认为 GPU 访问对于有效训练是**必不可少**的。



**提到的链接**: <a href="https://arxiv.org/abs/2309.16349">Human Feedback is not Gold Standard</a>：Human feedback 已成为评估 Large Language Models 性能的事实标准，并越来越多地被用作训练目标。然而，目前尚不清楚哪些属性...

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1278820480358420603)** (4 messages): 

> - `LLM Pruning`
> - `Text-to-Speech ML`
> - `Multi-Party Chat Agents`
> - `Qwen2-VL Vision Language Models` 


- **LLM 中的高效层剪枝（Layer Pruning）**：一项研究探索了针对开源权重预训练 **LLMs** 的层剪枝策略，发现直到移除**多达一半**的层时，性能下降才非常轻微。团队采用了 **parameter-efficient finetuning (PEFT)** 和 [quantization](https://arxiv.org/abs/2403.17887) 技术来恢复剪枝后的模型性能。
   - 这表明剪枝有助于降低计算成本，同时提高内存和推理速度。
- **Text-to-Speech ML 的 GitHub 仓库**：一个新的名为 [Text-to-Speech-ML](https://github.com/Azymack/Text-to-Speech-ML-) 的仓库已经发布，旨在为 text-to-speech 模型领域做出贡献和开发。该项目是一个协作成果，邀请用户参与。
   - 该仓库展示了最新进展，并为 text-to-speech 领域的进一步开发提供了工具。
- **探索 AI 的多方对话**：对多方对话的研究表明，现有的基于成对（pairwise）对话训练的模型在处理群体动态方面表现不佳，并识别出这些模型缺乏的关键技能。该研究发布了一个新数据集 **MultiLIGHT**，以提高 AI 在多参与者对话中的表现，适用于 [AI chatbots](https://arxiv.org/abs/2304.13835)。
   - 这项工作强调了对话上下文以及多个角色之间连贯交互的重要性。
- **Qwen2-VL 的 SOTA 视觉语言模型**：**Qwen2-VL** 系列已发布，在 **MathVista** 和 **DocVQA** 等视觉理解基准测试中达到了 state-of-the-art 性能。该先进模型可以理解超过 **20 分钟长**的视频，增强了视觉语言集成的通用性。
   - Qwen2-VL 的发布强调了其理解不同分辨率图像的能力，展示了 Qwen 模型家族的重大演进。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2403.17887">The Unreasonable Ineffectiveness of the Deeper Layers</a>：我们对流行的开源权重预训练 LLM 家族进行了简单的层剪枝策略实证研究，发现在不同的问答基准测试中，直到剪掉...</li><li><a href="https://arxiv.org/abs/2304.13835">Multi-Party Chat: Conversational Agents in Group Settings with Humans and Models</a>：目前的对话研究主要研究成对（两方）对话，没有解决两个以上发言者共同交谈的日常场景。在这项工作中，我们收集了...</li><li><a href="https://qwenlm.github.io/blog/qwen2-vl/">Qwen2-VL: To See the World More Clearly</a>：DEMO GITHUB HUGGING FACE MODELSCOPE API DISCORD 经过一年的不懈努力，今天我们非常激动地发布 Qwen2-VL！Qwen2-VL 是基于...的视觉语言模型的最新版本。</li><li><a href="https://github.com/Azymack/Text-to-Speech-ML-">GitHub - Azymack/Text-to-Speech-ML-</a>：通过在 GitHub 上创建账户来为 Azymack/Text-to-Speech-ML- 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1278811373656215704)** (12 messages🔥): 

> - `FLUX LoRA Training`
> - `ToonGPT Launch on Product Hunt`
> - `Word Game Bench for Language Models`
> - `VividNode Chatbot Release`
> - `Thoth Bot CLI Tool` 


- **FLUX LoRA 训练简化版**：一篇名为 [FLUX LoRA Training Simplified](https://youtu.be/nySGu12Y05k) 的教程指南，引导用户在 Windows 上使用 8GB GPU 通过 Kohya SS GUI 进行训练。
   - 该指南旨在让从零开始的用户也能轻松掌握训练流程。
- **ToonGPT 现已上线！**：ToonGPT 已正式在 [Product Hunt](https://www.producthunt.com/products/toontales-kiddiegpt) 发布，这是一款受个人经历启发、为儿童打造的互动式 AI 驱动伴侣。
   - 创作者希望通过技术为儿童参与提供独特的方法，并期待获得反馈和支持。
- **使用 Word Game Bench 评估语言模型**：新开发的 **Word Game Bench** 作为一个基准测试，用于评估语言模型在各种文字拼图游戏中的表现，目前尚无模型胜率超过 50%。
   - 它专注于两项任务：用于单词猜谜的 **Wordle** 和用于单词关联的 **Connections**，强调模型的交互与推理能力。
- **VividNode 聊天机器人发布**：开源聊天机器人 **VividNode** 已发布，具备 GPT 和图像生成功能，展示了开发者技能的成长。
   - 一篇[教程文章](https://medium.com/@yjg30737/what-is-vividnode-how-to-use-it-4d8a9269a3c0)详细介绍了其用法以及未来增加功能的计划。
- **Thoth Bot CLI 工具介绍**：[Thoth Bot](https://github.com/U-C4N/Thoth-Bot) 是一款 AI 驱动的 CLI 工具，旨在通过 Groq API 调用多个 LLM 进行对话、Python 代码生成及改进，从而简化编码工作流。
   - 它提供代码生成、执行和错误修复的自动化功能，提高了开发者的生产力。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://wordgamebench.github.io">Word Game Bench</a>: 未找到描述</li><li><a href="https://dev.to/p3ngu1nzz/scaling-up-parallel-training-with-tau-llm-and-unity-ml-agents-53bh">未找到标题</a>: 未找到描述</li><li><a href="https://airesearch.wiki/index.html">ai-research-agent</a>: 未找到描述</li><li><a href="https://medium.com/@yjg30737/what-is-vividnode-how-to-use-it-4d8a9269a3c0">What is VividNode &amp; How to Use It</a>: VividNode 是一款允许你在桌面上直接体验 GPT 聊天机器人 (ChatGPT) 和图像生成功能的软件，无需……</li><li><a href="https://www.producthunt.com/products/toontales-kiddiegpt"> ToonTales - KiddieGPT - Product Information, Latest Updates, and Reviews 2024 | Product Hunt</a>: 介绍 ToonGPT：一款为孩子们精心打造的愉快 AI 伴侣！受我女儿 Becky 的启发，ToonGPT 将卡通的魔力与互动乐趣相结合，激发创造力并带来快乐……</li><li><a href="https://github.com/U-C4N/Thoth-Bot">GitHub - U-C4N/Thoth-Bot: AI-powered CLI tool for chat, Python code generation, and improvement using multiple LLMs via Groq API. Streamlines coding workflow with automated code generation, execution, and error fixing.</a>: AI 驱动的 CLI 工具，用于对话、Python 代码生成，以及通过 Groq API 使用多个 LLM 进行改进。通过自动化的代码生成、执行和错误修复来简化编码工作流。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1278904332636389430)** (5 messages): 

> - `Meta FAIR's Transfusion`
> - `Multimodal modeling advancements`
> - `GitHub updates` 


- **Meta FAIR 揭晓 Transfusion 突破**：Meta FAIR 关于 **Transfusion** 的研究代表了 **multimodal modeling** 的重大飞跃，允许在统一框架内并发预测 token 和图像扩散（image diffusion）。
   - 该模型展示了*令人印象深刻的可扩展性*，并证明了其**优于传统方法**的性能，这可能会**彻底改变多模态应用**。
- **社区对 Transfusion 的兴奋**：成员们对 **Transfusion** 表示兴奋，认可其在处理多模态任务海量数据集方面的变革性能力。
   - 有成员通过提到论文中出现的大量 **gen AI 关键词**，指出了其*性能的重要性*。
- **GitHub 更新以供记录**：一位成员更新了社区的 **GitHub** 仓库以便更好地记录，并征求关于所遇问题的反馈。
   - 另一位成员对 **Transfusion 的质量**表示好奇，表示会去查看。


  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1278908354969997403)** (13 messages🔥): 

> - `Document Quality Assessment` (文档质量评估)
> - `Transfer Learning Challenges` (Transfer Learning 挑战)
> - `OpenCV Techniques` (OpenCV 技术)
> - `GitHub Repo for Document Classifier` (文档分类器的 GitHub 仓库)
> - `Networking and Friend Requests` (社交与好友请求) 


- **使用图像处理进行文档质量评估**：一位成员建议利用 **image processing techniques** 和预训练模型（如 **OpenCV**），通过模糊检测和直方图分析等方法来评估文档质量。
   - 他们还提议探索 **CNNs**（如 **VGG** 和 **ResNet**），针对特定的文档质量要求进行微调。
- **文档数据在 Transfer Learning 中的困境**：另一位成员尝试在通过增加亮度和模糊处理的数据集上应用 **transfer learning**，但指出其在真实世界文档中的表现不佳，因此正在寻求策略。
   - 他们表达了对 kernel 应用资源的渴求，并强调了这一问题在 **organizations** 中的重要性。
- **分享文档分类器的 GitHub 仓库**：一位用户分享了他们的 GitHub 仓库，其中包含一个 Notebook，详细记录了他们在 **FUNSD** 数据集上进行 **transfer learning** 的尝试，并强调了所使用的数据增强技术。
   - 项目链接在[这里](https://github.com/ajkdrag/noisy_doc_clf/blob/main/notebooks/train.ipynb)，展示了应用的各种图像和方法。
- **深夜讨论计划**：成员们讨论了时间已晚，建议第二天早上继续交流，体现了协作的态度。
   - 一位成员表示他们已发送好友请求，以便进一步讨论。
- **接受好友请求**：一位成员确认收到了好友请求并表示感谢，营造了友好的协作氛围。
   - 这一举动凸显了他们持续讨论中的人际交往层面。



**提到的链接**：<a href="https://github.com/ajkdrag/noisy_doc_clf/blob/main/notebooks/train.ipynb">noisy_doc_clf/notebooks/train.ipynb at main · ajkdrag/noisy_doc_clf</a>：通过在 GitHub 上创建账号来为 ajkdrag/noisy_doc_clf 的开发做出贡献。

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1278803082410721300)** (9 messages🔥): 

> - `LLaMA 3 models`
> - `Inference GPUs`
> - `GPU RAM configurations` 


- **寻求 LLaMA 3 模型的指导**：一位用户在计划构建 **RAG applications** 时请求关于 **LLaMA 3** 模型的帮助，并需要关于合适的本地 GPU 的建议。
   - 他们专门询问了与不同模型尺寸（**8B**、**70B** 和 **405B**）相关的 GPU 和 RAM 配置。
- **GPU 推荐**：一位成员建议 **Nvidia A100** 是运行这些模型的最佳选择，尽管他们没有具体说明 RAM 需求。
   - 随后提出了关于 **A100** 应该搭配哪种 RAM 以及使用哪种模型的问题，表明需要更详细的建议。
- **澄清 LLaMA 405B 的需求**：另一位成员指出，运行 **LLaMA 405B** 模型至少需要 **300Gb 的 GPU RAM**，具体取决于精度。
   - 他们警告说使用如此庞大的模型极其昂贵，建议探索基于云端的方法。
- **对所提供建议的怀疑**：一位成员对之前回复的准确性表示怀疑，认为其中一个回答是由模型生成的，且事实错误。
   - 这引发了进一步的猜测，即该答案可能源自 **LLaMA 3** 本身。


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1278919068015005746)** (2 messages): 

> - `Animating Fireball in Photos` (动画化照片中的火球)
> - `Using AnimateDiff with IP Adapter Plus or SVD` (使用 AnimateDiff 配合 IP Adapter Plus 或 SVD) 


- **询问如何动画化照片中的火球**：一位用户询问是否可以仅对他们上传的照片中的 **fireball** 进行动画处理。
   - 这凸显了对图像中选择性动画技术的兴趣。
- **推荐使用 AnimateDiff**：另一位成员建议使用 **AnimateDiff** 配合 **IP Adapter Plus** 或 **SVD** 作为动画化火球的解决方案。
   - 他们的建议表明了对用于动画任务的 AI 工具的潜在兴趣。


  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/)** (1 messages): 

iron_bound: 听起来他们的 LTM 架构有一个用于 **attention** 的 **RNN**

### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1278902836868022282)** (1 条消息): 

> - `Triton atomic_add 功能`
> - `多 GPU 配置`
> - `Triton 中的 Scope 定义` 


- **关于 Triton 中 scope=GPU 的澄清**：一位成员询问了在多 GPU 环境下，为 `atomic_add` 函数使用 **scope=GPU** 的影响。
   - 他们质疑默认的 **scope=GPU** 在多 GPU 设置中是否能有效运行。
- **理解 Triton 中的 scope=system**：讨论还涉及了 **scope=system** 的含义，特别是它指的是多 GPU 还是包括与 **host** 的交互。
   - 一位成员对 **scope=system** 是否包含 GPU 以及 **host** 操作表示困惑。


  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1279199308410781848)** (3 条消息): 

> - `带有 Triton kernel 的 FX pass`
> - `从 PyTorch 调用 Triton`
> - `FX pass 示例`
> - `Triton 代码参考` 


- **询问关于 Triton 的 FX pass**：一位成员询问是否可以实现一个 **FX pass**，将 **aten ops** 映射到自定义的 **Triton kernel** 上。
   - 这一询问表明人们对利用 Triton 的能力优化 PyTorch 性能持续关注。
- **原生调用 Triton 代码**：澄清了可以直接从 **PyTorch program** 原生调用 **Triton code**，使其能够与 **torch.compile** 协同工作。
   - 这强调了 **Triton** 在 PyTorch 生态系统中的集成，以增强功能。
- **FX pass 示例资源**：成员们提到，对于 **FX passes** 的示例，查看 **Triton code** 会很有帮助。
   - 分享了一个指向 [pre_grad.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/pre_grad.py) 的具体链接作为参考。



**提及的链接**：<a href="https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/pre_grad.py">pytorch/torch/_inductor/fx_passes/pre_grad.py at main · pytorch/pytorch</a>：Python 中具有强大 GPU 加速的张量和动态神经网络 - pytorch/pytorch

  

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1278803556308353197)** (25 条消息🔥): 

> - `Quantization Techniques` (量化技术)
> - `AWQ Implementation Issues` (AWQ 实现问题)
> - `Low-Bit Optimizer Code` (低比特优化器代码)
> - `VLLM Integration` (VLLM 集成)
> - `Layer Quantization Strategies` (层量化策略)


- **注意力层的量化 (Quantization of Attention Layers)**：在注意力层中对 QKV projections 进行量化似乎很常见，默认的 filter 函数通过检查形状来处理 **2D Linear layers**。
   - 成员们对维持这些层的准确性表示担忧，并引发了关于此类量化是否为有意为之的辩论。
- **带有 Zero Points 的 AWQ 性能**：成员们讨论了与整数相比，使用 **floating point integers** 进行量化时 AWQ 的性能会显著下降，导致 perplexity 增加。
   - *量化过程中的 Rounding 似乎会影响兼容性*，成员们分享了来自早期调查的实现细节。
- **调查低比特优化器代码**：针对低比特优化器代码中关于 **non-sign bits** 的一行可疑代码提出了担忧，认为这可能是从另一个项目复制而来的。
   - 建议简化部分代码，尽管某些函数的 kernel fusions 存在限制。
- **VLLM 与 AWQ 的集成**：大家有兴趣探索 **较新版本的 VLLM 如何利用 AWQ**，因为过去的实现在操作 quant/dequant 函数时遇到了挑战。
   - 成员们强调了在不同量化技术之间进行准确比较的必要性，特别是与 embeddings 相关的部分。
- **测试低比特量化策略**：关于混合精度量化（mixed precision quantization）的讨论揭示了一个 GitHub 原型，可能为不同模型规模提供有用的见解。
   - 鼓励成员们查看该仓库，因为它为更好地理解量化结果提供了潜在途径。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/ao/tree/main/torchao/quantization/prototype/mixed_precision">ao/torchao/quantization/prototype/mixed_precision at main · pytorch/ao</a>: PyTorch 原生量化与稀疏化，用于训练和推理 - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/blob/main/torchao/prototype/low_bit_optim/quant_utils.py#L28C5-L28C54).">ao/torchao/prototype/low_bit_optim/quant_utils.py at main · pytorch/ao</a>: PyTorch 原生量化与稀疏化，用于训练和推理 - pytorch/ao</li><li><a href="https://gist.github.com/mobicham/8b3147742beb3b302064453a15ced428#file-awq_hqq_test-py-L52">awq_hqq_test.py</a>: awq_hqq_test.py。GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://github.com/pytorch/ao/blob/main/torchao/prototype">ao/torchao/prototype at main · pytorch/ao</a>: PyTorch 原生量化与稀疏化，用于训练和推理 - pytorch/ao</li><li><a href="https://github.com/pytorc">pytorc - Overview</a>: pytorc 有 2 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/pytorch/ao/blob/main/torchao/prototype/low_bit_optim/quant_utils.py#L69-L106">ao/torchao/prototype/low_bit_optim/quant_utils.py at main · pytorch/ao</a>: PyTorch 原生量化与稀疏化，用于训练和推理 - pytorch/ao</li><li><a href="https://pytorch.org/docs/stable/generated/torch.searchsorted.html">torch.searchsorted &mdash; PyTorch 2.4 documentation</a>: 无描述</li><li><a href="https://github.com/bitsandbytes-foundation/bitsandbytes/blob/e4674531dd54874c0abbc786ad5635c92c34dc3e/bitsandbytes/functional.py#L360">bitsandbytes/bitsandbytes/functional.py at e4674531dd54874c0abbc786ad5635c92c34dc3e · bitsandbytes-foundation/bitsandbytes</a>: 通过针对 PyTorch 的 k-bit 量化实现可访问的大型语言模型。 - bitsandbytes-foundation/bitsandbytes</li><li><a href="https://github.com/pytorch/ao/pull/769">Fixed the llama model by yiliu30 · Pull Request #769 · pytorch/ao</a>: 如果我们不将 input_pos 传递给模型，freqs_cis = self.freqs_cis[input_pos] 将选择整个 freqs_cis。测试命令 pytest -sv ./test/test_ao_models.py cc @HDCharles
</li>
</ul>

</div>

### **CUDA MODE ▷ #[sequence-parallel](https://discord.com/channels/1189498204333543425/1208496482005549086/1279136959381639301)** (2 条消息): 

> - `Flash Attention Kernel`
> - `Shared Memory Sizes in FA`
> - `NVIDIA GeForce RTX 3090 Support`
> - `Attention Heads and Model Dimensions` 


- **在 Flash Attention 中挣扎于 Shared Memory 大小**：一位用户提到在编写 Flash Attention Kernel 时遇到困难，特别是关于 Shared Memory 的大小；他们指出当 Q 的 Block Size 达到 **131,072 bytes** 时，内存需求巨大。
   - 这引发了一个问题：Flash Attention (FA) 如何在 SRAM 容量较小的非 Hopper GPU 上高效运行。
- **NVIDIA GeForce RTX 3090 问题**：另一位用户报告了在 NVIDIA GeForce RTX 3090 GPU（Compute Capability 为 **8.6**）上运行 flash_attn 包时遇到的问题。
   - 他们链接了一个 [GitHub issue](https://github.com/Dao-AILab/flash-attention/issues/190)，讨论了在该特定硬件上运行该包时遇到的问题。
- **关于跨 Attention Heads 维度拆分的疑问**：有一个关于大型模型维度是否被拆分到各个 Attention Heads 的提问，建议每个 FA Head 仅处理 **64 或 128** 左右的较小内部维度。
   - 这一推测突出了 Flash Attention 的机制及其对不同底层架构的潜在适应性。



**提到的链接**：<a href="https://github.com/Dao-AILab/flash-attention/issues/190">Support for NVIDIA GeForce RTX 3090 with Compute Capability 8.6 · Issue #190 · Dao-AILab/flash-attention</a>：Issue 描述：你好，我正在一个拥有两块 NVIDIA GeForce RTX 3090 GPU 的系统上使用 flash_attn 包，这两块 GPU 的 Compute Capability 均为 8.6。在尝试运行该包时，我遇到了...

  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1278869677178884159)** (15 条消息🔥): 

> - `Twitter Profile Recommendations`
> - `Pros and Cons of Twitter`
> - `Twitter for Research and Networking`
> - `Logistics for CUDA Mode Event` 


- **Twitter 账号推荐**：一位用户征求值得关注的 Twitter 账号推荐，其中一人推荐了 marksaroufim 的[特定列表](https://x.com/marksaroufim/following)。
   - 另一位用户持怀疑态度，建议或许根本不注册账号更好。
- **关于 Twitter 价值的辩论**：发起了一项投票，询问用户 24 年夏天在 Twitter 上花费的时间是**净正面（net positive）**还是**净负面（net negative）**，用户分享了不同的观点。
   - 一些用户同意 Twitter 有利于接触前沿研究和分享个人工作。
- **关于 Twitter 使用的担忧**：参与者讨论了如何通过仔细筛选 Twitter 关注列表来提升体验，其中一人表示主要将其用于阅读选定内容。
   - 另一位用户幽默地指出，需要定期将帖子标记为“不感兴趣”来清理信息流。
- **关于 CUDA Mode 活动后勤的咨询**：一位新成员询问了即将举行的 CUDA Mode 活动的后勤安排，特别是关于住宿和餐饮供应。
   - 他们询问是否需要预订酒店，并征求关于活动结构的更多细节。


  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1278873486789447681)** (6 条消息): 

> - `L2 Side Aware optimization`
> - `FP8 switching`
> - `Loss landscape stationary points`
> - `Training sample dropping` 


- **L2 Side Aware 代码实现速度提升**：“L2 Side Aware”代码已修复并简化，在 GELU forward 中稳定达到 **1823GB/s**，超过了之前使用 x128 Kernel 的 **1791GB/s**。
   - 改进包括 **2% 的速度提升**和显著降低的功耗，尽管仍需要进一步的简化和优化。
- **回归 FP8 进行开发**：一位成员计划明天切换回 **FP8** 代码开发，以便在当前项目进一步推进前刷新理解。
   - 他们对 L2 Side Aware 代码取得的进展表示满意，但认识到还需要额外的优化。
- **关于 Loss landscape 和训练约束的讨论**：一位用户讨论了在全权重空间优化时，Loss landscape 中驻点（stationary point）的影响，质疑其与传统方法相比实际施加的约束。
   - 他们强调需要实现 vanilla fine-tuning 以验证所达到的极小值（minima）的质量。


  

---


### **CUDA MODE ▷ #[sparsity-pruning](https://discord.com/channels/1189498204333543425/1247663759434977453/)** (1 条消息): 

mobicham: https://x.com/JamesLiuID/status/1829554782287413513
  

---

### **CUDA MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1278791518769119324)** (140 条消息🔥🔥): 

> - `Release v0.2.0 讨论`
> - `LayerNorm Kernel 更新`
> - `Hugging Face 示例中的内存问题`
> - `调试 RMS Norm Kernel`
> - `文档增强` 


- **Release v0.2.0 讨论**：社区讨论了 [v0.2.0](https://github.com/linkedin/Liger-Kernel/releases/tag/v0.2.0) 的发布，重点介绍了 API 和模型支持方面的改进，以及新功能的引入和 Bug 修复。
   - 然而，一些用户报告了内存问题，一位用户在使用此版本运行 Hugging Face 示例时遇到了 `OutOfMemoryError`。
- **LayerNorm Kernel 更新**：[PR #169](https://github.com/linkedin/Liger-Kernel/pull/169) 已合并，集成了 LayerNorm 自定义 Kernel 和 LigerLayerNorm 模块，并在 RTX 3090 上进行了正确性测试。
   - 讨论的更新包括分析结果和针对原子操作（atomic operations）的潜在动态调度，旨在提高多 GPU 场景下的性能。
- **Hugging Face 示例中的内存问题**：在使用 v0.2.0 进行测试后，用户注意到该示例的内存效率低于 v0.1.1，从而引发了对其默认设置的担忧。
   - 一位用户确认，在不使用 Liger 的情况下运行该示例会导致立即出现 OOM 错误，这表明 Liger 集成对于运行大 Batch Size 至关重要。
- **调试 RMS Norm Kernel**：一位贡献者报告称，在重写 rms_norm Kernel 以使用部分聚合（partial aggregation）时，特定测试反复失败，通过手动设置种子（seed）后行为变得具有确定性。
   - 进一步调查发现了更多的内容不匹配，以及 `assert_verbose_allclose` 函数中可能存在的 Bug，建议该条件应检查大于 0 的不匹配值。
- **文档增强**：README 中添加了关于 LayerNorm 的新章节，明确了其在库中的功能和实现。
   - 社区表示有兴趣创建文档网站和教程，以帮助用户集成自定义操作并更好地利用该工具。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://hidet.org/docs/stable/gallery/developer-guides/add-operator-resolve-rule.html">添加算子解析规则 — Hidet 文档</a>: 未找到描述</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/174">使用 LigerKernel 时 torch.compile() 抛出异常 · Issue #174 · linkedin/Liger-Kernel</a>: 🐛 描述 Bug ... 文件 "/home/tromero/workspace/seahorse/.venv/lib/python3.11/site-packages/torch/_inductor/async_compile.py", 第 173 行, 在 triton kernel = TritonCodeCache.load(kernel_...</li><li><a href="https://github.com/linkedin/Liger-Kernel?tab=readme-ov-file#kernels">GitHub - linkedin/Liger-Kernel: 用于 LLM 训练的高效 Triton 内核</a>: 用于 LLM 训练的高效 Triton 内核。通过在 GitHub 上创建账号来为 linkedin/Liger-Kernel 的开发做出贡献。</li><li><a href="https://github.com/linkedin/Liger-Kernel/releases/tag/v0.2.0">Release v0.2.0 发布说明 · linkedin/Liger-Kernel</a>: 开篇感言 🫶 谢谢大家！我们想借此机会向社区表达诚挚的感谢！2500+ ⭐，10+ 新贡献者，50+ PR，以及与 Hugging Face 🤗 的集成，还有...</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/179)">Issues · linkedin/Liger-Kernel</a>: 用于 LLM 训练的高效 Triton 内核。通过在 GitHub 上创建账号来为 linkedin/Liger-Kernel 的开发做出贡献。</li><li><a href="https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)`">CUDA 语义 — PyTorch 2.4 文档</a>: 未找到描述</li><li><a href="https://github.com/linkedin/Liger-Kernel.git">GitHub - linkedin/Liger-Kernel: 用于 LLM 训练的高效 Triton 内核</a>: 用于 LLM 训练的高效 Triton 内核。通过在 GitHub 上创建账号来为 linkedin/Liger-Kernel 的开发做出贡献。</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/169">[算子] LayerNorm 内核 + LigerLayerNorm 由 AndreSlavescu 提交 · Pull Request #169 · linkedin/Liger-Kernel</a>: 摘要：集成了 LayerNorm 自定义内核 + LigerLayerNorm 模块。测试完成：测试了 LayerNorm 内核的正确性。硬件类型：RTX 3090。运行 make test 以确保正确性，运行 make...</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/135/files#diff-0b31d056b2cdb59db1baaba4c4e7e0a79ed70b445ca67ff928ec57ffa89c6d0fR71">自定义 Embedding 内核 由 AndreSlavescu 提交 · Pull Request #135 · linkedin/Liger-Kernel</a>: 摘要：添加了 Embedding 前向/反向内核 + 映射到 nn.Embedding 的 LigerEmbedding 类。nn.Embedding 对于像 BERT 这样的纯编码器模型非常有用，参考：#131。测试完成：测试了...</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/180">[文档] 将 LayerNorm 添加到 README 由 AndreSlavescu 提交 · Pull Request #180 · linkedin/Liger-Kernel</a>: 摘要：在 README 中添加了 LayerNorm 描述。测试完成：不适用。硬件类型：RTX 3090。运行 make test 以确保正确性，运行 make checkstyle 以确保代码风格，运行 make test-convergenc...
</li>
</ul>

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1278792548558835713)** (187 条消息🔥🔥): 

> - `Flux 的 IP Adapter`
> - `在有限显存 (VRAM) 下训练模型`
> - `图像处理中的分割 (Segmentation)`
> - `RunwayML 删除 SD 1.5 仓库`
> - `SDXL vs SD 1.5` 


- **Flux 的 IP Adapter 受到关注**：成员们讨论了最近推出的 Flux IP Adapter，其性能表现褒贬不一，部分用户认为其效果欠佳。
   - 一位成员指出，尽管评价不一，但这仍是社区中一个令人兴奋的进展。
- **在有限显存 (VRAM) 下训练模型**：用户分享了在有限显存（特别是 RTX 3060）下进行训练的经验，指出高分辨率（如 1024）会消耗大量显存。
   - 建议采用较低的分辨率以减少显存占用，并确认 12GB 显存可能不足以处理复杂任务。
- **图像处理中的分割 (Segmentation)**：讨论强调了图像处理工作流中 SEG（分割）的概念，特别是它如何连接到 ComfyUI 等系统中的现有节点。
   - 参与者对其实现方式以及与更简单的替代方案相比的必要性表示困惑。
- **RunwayML 删除 SD 1.5 仓库**：社区注意到 RunwayML 已删除了其在 HuggingFace 和 GitHub 上的所有 Stable Diffusion 1.5 仓库，引发了关于此举影响的各种反应。
   - 用户猜测这次删除是否意味着重心正从据称使用率较低的 1.5 模型转移。
- **比较 SDXL 与 SD 1.5**：一位用户考虑从 SD 1.5 切换到 SDXL，权衡了现有 GPU 下生成时间和模型存储需求的问题。
   - 建议通过命令行参数优化性能，以适应性能较弱的 GPU。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://imgur.com/ygD5YMm">imgur.com</a>: 在 Imgur 探索互联网的魔力，这是一个由社区驱动的娱乐目的地。通过幽默的笑话、热门的梗图、有趣的 GIF、励志的故事、病毒式传播的视频等来放松心情...</li><li><a href="https://imgur.com/Xr44AHl">imgur.com</a>: 在 Imgur 探索互联网的魔力，这是一个由社区驱动的娱乐目的地。通过幽默的笑话、热门的梗图、有趣的 GIF、励志的故事、病毒式传播的视频等来放松心情...</li><li><a href="https://www.amazon.de/Fantastische-Fabelwesen-Stressabbau-Entspannung-Fantasie-Kreaturen/dp/B0CN5B8WTG/ref=sr_1_1?crid=3IBODT2J8X6H6&dib=eyJ2IjoiMSJ9.-3XggVW3uObjvvXQqObf-g-EWf_V6QDcBkrHerEySuY2P3W0J8JG92mAOXoFt2DWOwZHT1w0m6M4IrDxhUwXVi523Affpx6n5y5TI3Pal5iMGXUuSJEje7x1BSRxDuAhRJqcESyU0awWBpc07xA90cucn7Z_uETG34wev0if1-ON4ICntYnPnlLPGVH6WUk532dqEr89fXftuzS4TrhIrYMCKNik-WVzuMj3aU2Vvr8.d_Vd1P3m4memC-Dd8Agtfsyxu8CgD6J3vjQdJ--SaDo&dib_tag=se&keywords=fabelwesen+malbuch&qid=1724956770&sprefix=Fabelwesen+%2Caps%2C126&sr=8-1">未找到标题</a>: 未找到描述</li><li><a href="https://youtu.be/3LnbI5pcQko">我用过的最奇怪的 AI 应用</a>: 这个 AI 颠覆了社交媒体。它就像 Instagram，但所有人都是 AI。#ainews #ai #agi #socialmedia #npcTurboType 帮助你通过键盘快捷键更快地打字...</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Command-Line-Arguments-and-Settings">命令行参数与设置</a>: Stable Diffusion web UI。通过在 GitHub 上创建账户为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://youtu.be/v9KGQoaqhkw">ComfyUI 中的 CogVideoX 5B 质量更佳 - 真正可用的本地 AI 视频模型！</a>: CogVideoX 5B 更好的质量 - 一个真正可用的本地 AI 视频模型！在这段引人入胜的视频中，我们深入探讨了 AI 技术的最新进展...</li><li><a href="https://arxiv.org/abs/2408.16232">通过可解释的潜空间操作增强条件图像生成</a>: 在图像合成领域，在遵循条件提示的同时实现对参考图像的忠实度仍然是一个重大挑战。本文提出了一种集成...</li><li><a href="https://github.com/kshitij79/CS-7476-Improvements-in-Diffusion-Model">GitHub - kshitij79/CS-7476-Improvements-in-Diffusion-Model</a>: 通过在 GitHub 上创建账户为 kshitij79/CS-7476-Improvements-in-Diffusion-Model 的开发做出贡献。</li><li><a href="https://mp.weixin.qq.com/s/ZKJieSzqISyzCB8Iz9tY8A">【AI行业报告】Top 100 AI 产品 (第3期)</a>: AI行业报告第三期，来看看哪些AI产品上榜了？
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1278805467065352263)** (118 条消息🔥🔥): 

> - `AI 模型中的失忆模式 (Amnesia Mode)`
> - `LLM 训练技术`
> - `梯度通信策略`
> - `Hermes 3 模型行为`
> - `新的 AI 评估框架` 


- **失忆模式体验 (Amnesia Mode Experiences)**：用户讨论了 Hermes 3 的“失忆模式”，强调其更倾向于专业性而非日常俚语。一位用户对模型在面对日常问候时坚持保持“家庭友好（family-friendly）”的态度表示沮丧。
   - 即使在用户尝试进行日常互动时，模型也表现出奇特的反应，引发了关于这是否是预定义行为的讨论。
- **LLM 训练技术**：一位成员分享了他们正在使用来自 Reddit 等平台的合成及真实指令数据训练 Llama 3。他们的目标是研究这一过程是否能通过使数据更具指令导向性来减少“AI 味（AI-y）”的回答。
   - 社区参与了关于处理训练损失、异常训练行为的经验以及管理梯度问题重要性的讨论。
- **探索梯度通信策略**：一位用户提出了模型同步期间梯度的低秩近似 (low-rank approximations)，旨在减少通信开销。他们强调了通过分析来自数据并行节点的梯度影响可能带来的增强。
   - 讨论围绕结合各种优化技术以促进更有效的分布式训练策略展开。
- **Hermes 3 模型行为见解**：用户注意到 Hermes 3 表现出某些行为模式，包括对沟通风格的潜在偏好。有人对这些行为背后的原因以及它们如何受到系统提示词 (system prompts) 影响提出了疑问。
   - 互动显示某些短语会触发意想不到的反应，暗示了失忆模式的混合，促使成员们分享相关经验。
- **新的 AI 评估框架：Word Game Bench**：介绍了一个名为 “Word Game Bench” 的新基准测试，旨在通过 Wordle 和 Connections 等文字拼图游戏评估语言模型。创建者允许独特的交互方式，模型可以根据之前的游戏动作生成输出。
   - 成员们对该基准测试的方法及其在以参与和互动方式评估模型性能方面的意义表示了兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/sama/status/1829205847731515676?s=19">Sam Altman (@sama) 的推文</a>: 我们很高兴与美国 AI 安全研究所达成协议，对我们未来的模型进行发布前测试。出于多种原因，我们认为这在国家层面发生非常重要...</li><li><a href="https://x.com/wingsoverheaven/status/1829024789693968628">wings (@wingsoverheaven) 的推文</a>: 未找到描述</li><li><a href="https://wordgamebench.github.io">Word Game Bench</a>: 未找到描述</li><li><a href="https://x.com/zafstojano/status/1829398835585520076">zafir (@zafstojano) 的推文</a>: 很高兴分享 “Word Game Bench” —— 一个用于在文字拼图游戏上评估语言模型的有趣基准测试！这是一个相对困难的基准测试，目前没有模型的平均得分超过 50%...</li><li><a href="https://arxiv.org/abs/2311.08105">DiLoCo: Distributed Low-Communication Training of Language Models</a>: 大语言模型 (LLM) 已成为机器学习许多应用中的关键组件。然而，训练 LLM 的标准方法需要大量紧密互连的加速器...</li><li><a href="https://github.com/oobabooga/text-generation-webui">GitHub - oobabooga/text-generation-webui: 用于大语言模型的 Gradio Web UI。</a>: 用于大语言模型的 Gradio Web UI。通过在 GitHub 上创建账号为 oobabooga/text-generation-webui 的开发做出贡献。</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-70b">Hermes 3 70B Instruct - API, Providers, Stats</a>: Hermes 3 是一款通用语言模型，相比 [Hermes 2](/models/nousresearch/nous-hermes-2-mistral-7b-dpo) 有许多改进，包括先进的 Agent 能力、更好的角色扮演、推理...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1278815792594686032)** (43 条消息🔥): 

> - `Instruction Tuning`
> - `Hermes 3 Performance`
> - `Full Precision vs 8 Bit Models`
> - `Hardware Requirements for Large Models`
> - `100 Million Token Context Window` 


- **Instruction Tuning 见解**：一位成员询问 Instruction Tuning 是否通常涉及对对话的用户端进行训练，另一位成员确认仅在输出端（outputs）进行训练效果更好。
   - **与包含用户输入相比，仅在输出端进行训练能产生显著更好的 Benchmark 结果**。
- **寻求全精度 Hermes 3 模型**：一位用户对寻找全精度 **Hermes 3 模型** (bf16) 的托管方感到沮丧，据报道目前没有提供商。
   - 讨论显示目前还没有提供商提供该模型，对**效率和硬件需求**的担忧是主要障碍。
- **量化对模型性能的影响**：讨论指出，较大的模型往往更具**量化抗性**（quantization resistant），这会影响低比特量化水平下的性能。
   - 例如，一个 **70B 模型**在 2-bit 下仍能生成连贯的文本，而不像较小的模型那样会出现退化。
- **托管大语言模型的担忧**：讨论强调，运行像 **Hermes 3** (405B) 这样的模型需要大量的硬件配置，通常需要多节点（multinode）配置。
   - 成员们指出了平衡需求与硬件能力之间的挑战，这导致许多提供商坚持使用低比特量化模型。
- **1 亿上下文窗口的魔力**：一位用户强调了关于 **1 亿 Token 上下文窗口**的引人注目的新闻，这可能代表了与 Q* 相当的突破。
   - 其他人则幽默地评论了此类进展中被感知的“魔法”方面。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.lambdalabs.com/on-demand-cloud/using-the-lam">Lambda Docs</a>：未找到描述</li><li><a href="https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-405B">NousResearch/Hermes-3-Llama-3.1-405B · Hugging Face</a>：未找到描述</li><li><a href="https://docs.lambdalabs.com/on-demand-cloud/using-the-lambda-chat-completions-api">Using the Lambda Chat Completions API | Lambda Docs</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1278890189103566868)** (8 条消息🔥): 

> - `GameNGen`
> - `Real-time Game Simulation`
> - `Neural Network Integration in Gaming`
> - `Unique Hallucinations in Gaming`
> - `Potential for Horror Games` 


- **GameNGen：神经模型成为焦点**：围绕 _GameNGen_ 展开了讨论，这是第一个完全由神经模型驱动的游戏引擎，可以在没有传统游戏引擎工具的情况下，以超过 **每秒 20 帧** 的速度模拟经典游戏 **DOOM**。
   - 参与者对这一**概念验证**（proof of concept）感到兴奋，并关注像 **Unreal Engine** 这样的主流引擎如何集成类似技术。
- **迷幻的游戏体验吸引玩家**：游戏画面显示 _GameNGen_ 的模拟看起来很**迷幻**（trippy）甚至像**梦境**一样，激发了对复制现有游戏之外的未来应用的兴趣。
   - 一位成员指出，这些独特的幻觉（hallucinations）有潜力激发一个**完全原创的恐怖 IP**，为该类型游戏带来新鲜感。
- **AI 驱动游戏创作的挑战**：讨论强调了在使用神经模型时需要**引导**（guidance），暗示了构建连贯游戏体验所涉及的复杂性。
   - 随着技术的发展，关于 AI 创造力与玩家交互之间平衡的问题也随之而来，以实现引人入胜且连贯的游戏玩法。



**提到的链接**：<a href="https://gamengen.github.io/">GameNGen</a>: Diffusion Models Are Real-Time Game Engines

  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1278890189103566868)** (8 messages🔥): 

> - `GameNGen 神经模型`
> - `DOOM 模拟`
> - `与游戏引擎的集成`
> - `游戏中的独特幻觉`
> - `原创恐怖 IP 的潜力` 


- **GameNGen 在没有游戏引擎的情况下模拟 DOOM**：_GameNGen_ 神经模型实现了经典游戏 [DOOM](https://en.wikipedia.org/wiki/Doom_(1993_video_game)) 的实时模拟，且不涉及传统的游戏引擎。
   - 它使用单个 TPU 即可达到每秒 **20 帧**以上的速度，并在真实感方面表现出前途，人类评分者很难区分模拟片段和真实片段。
- **对将该技术与 Unreal Engine 集成的兴奋感**：一位成员对未来看到像 **Unreal Engine** 这样的主流游戏引擎如何集成这种神经模拟技术表示了极大的热情。
   - 他们还表现出自行复制该技术的兴趣，强调了其在创新游戏开发方面的潜力。
- **迷幻的游戏画面引发讨论**：_GameNGen_ 的游戏画面被描述为**迷幻的 (trippy)** 和**梦幻般的 (dreamlike)**，引发了关于其独特视觉体验的讨论。
   - 成员们分享了利用这些特性进行原创游戏设计的想法，特别是在恐怖类型中，这可能会提供一个令人耳目一新的视角。
- **对游戏中独特幻觉的兴趣**：大家共同关注如何利用神经模型产生的独特幻觉来创建原创恐怖 IP。
   - 这种方法可以为玩家提供一种不同于传统游戏机制的独特游戏体验。
- **模型需要引导才能有效使用**：针对利用神经模型进行复杂游戏创作时需要大量的监督和引导，人们提出了担忧。
   - 对“手把手引导 (hand-holding)”的需求表明，在使该技术易于使用并有效应用于原创内容方面存在挑战。



**提到的链接**：<a href="https://gamengen.github.io/">GameNGen</a>：扩散模型是实时游戏引擎

  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1278791887024689195)** (93 messages🔥🔥): 

> - `API 推理速度`
> - `LM Studio 更新稳定性`
> - `模型性能与兼容性`
> - `文本转图像/语音集成` 


- **API 推理速度限制讨论**：一位用户询问了关于限制 API 推理速度的问题，另一位成员澄清说，在加载多个模型的情况下处理多个请求是可行的。
   - 该用户表示倾向于使用同一个模型以节省 VRAM，但也承认这可能无法实现。
- **用户对 LM Studio 0.3 版本的反馈**：一位成员对最新的 LM Studio 更新降低了 AI 的响应速度表示担忧，并提到了异常的重复输出。
   - 其他用户建议该问题可能与 Prompt 设置或模板解析有关，建议进行调整以解决。
- **评估模型性能**：讨论围绕 Gemma 2 和 Yi 1.5 的性能对比展开，一些人认为 Gemma 2 被过度审查 (censored) 了。
   - 此外，用户评估了潜在的替代方案，强调需要一个通用的、无审查 (uncensored) 的模型。
- **关于文本转图像/语音能力的查询**：一位用户询问了在 LM Studio 中集成文本转图像或文本转语音功能的可能性。
   - 目前的讨论表明，现有的 LM Studio 设置中缺乏此类功能或支持。
- **在 CPU 上设置**：一位参与者询问了使用 CPU 时初始 Prompt 处理速度较慢的问题，引发了关于预期性能结果的讨论。
   - 建议认为，考虑到模型的架构，使用 CPU 进行处理的局限性可能是不可避免的。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/sophosympatheia/Midnight-Miqu-70B-v1.5/blob/main/tokenizer_config.json#L31">tokenizer_config.json · sophosympatheia/Midnight-Miqu-70B-v1.5 at main</a>：未找到描述</li><li><a href="https://huggingface.co/lmstudio-community/xLAM-7b-r-GGUF/tree/main">lmstudio-community/xLAM-7b-r-GGUF at main</a>：未找到描述</li><li><a href="https://github.com/YorkieDev/LMStudioWebUI">GitHub - YorkieDev/LMStudioWebUI: 用于 LM Studio 的简单 Web UI 开发版</a>：一个用于 LM Studio 的简单 Web UI 开发版 - YorkieDev/LMStudioWebUI</li><li><a href="https://github.com/THUDM/CogVideo">GitHub - THUDM/CogVideo: 文本转视频生成：CogVideoX (2024) 和 CogVideo (ICLR 2023)</a>：文本转视频生成：CogVideoX (2024) 和 CogVideo (ICLR 2023) - THUDM/CogVideo
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1278858932751695872)** (82 messages🔥🔥): 

> - `M2 Ultra Mac setup`
> - `LLM performance on GPUs`
> - `Parallel processing with multiple GPUs`
> - `Power consumption management`
> - `Model loading and inference speeds` 


- **M2 Ultra Mac 开发就绪**：一位成员提到他们配置了拥有 **192 GB** Unified Memory 的新 **M2 Ultra Mac**，以便在开始实验 LLM 之前建立开发环境。
   - 他们指出为此分配了一个 **2 TB** 的驱动器，并使用一台独立的 PC 作为服务器。
- **探索 RTX 4090 上的 LLM 性能**：讨论强调了在 **6 个 RTX 4090** 上运行 **405b model** 的速度约为 **每秒 1 个 token**，其中 offload 设置会影响性能。
   - 一位成员测试了多种 GPU 配置，观察到当模型分布良好时，跨 GPU 的内存链接如何潜在地提高速度。
- **测试并行处理能力**：多位用户争论 **LM Studio** 是否支持跨多个 GPU 的真正 parallel processing，并讨论了其对推理速度的影响。
   - 一位成员指出，在 Python 中拆分模型层并利用 memory offload 对于在更高的 token 速度下获得更好的性能可能非常有效。
- **管理 GPU 配置中的功耗**：用户对功耗表示担忧，特别是在运行多个 **RTX 4090** 时，这种配置通常需要共享相位以避免断路器跳闸。
   - 一位成员解释了他们如何配置电源供应单元 (PSUs) 以满足高需求，同时将负载分配到不同的电路上。
- **PCIe lane 设置对性能的影响**：随后讨论了在 gen4 x8 而非 x16 设置下运行 **RTX 4090** 的影响，特别是在使用多个 GPU 处理密集模型时。
   - 成员们一致认为，虽然 gen4 x8 配置可能不会显著影响单 GPU 设置的性能，但它可能会阻碍多 GPU 环境下的速度。



**Link mentioned**: <a href="https://tenor.com/view/power-usage-auxiliary-nuclear-gif-22138997">Power Usage Auxiliary Nuclear GIF - Power Usage Auxiliary Nuclear - Discover &amp; Share GIFs</a>: 点击查看 GIF

  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1278938553975308310)** (2 messages): 

> - `Gemini Flash models`
> - `Database downtime` 


- **Gemini Flash 模型现已可用且免费**：**Gemini Flash 8B (EXP)** 模型现在可以通过[此链接](https://openrouter.ai/models/google/gemini-flash-8b-1.5-exp)获取，**Gemini Flash Experiment** 可以在[这里](https://openrouter.ai/models/google/gemini-flash-1.5-exp)找到。
   - 所有 **Gemini Experimental models** 现已确认免费，直到 **AI Studio** 确定进一步的定价。
- **数据库错误导致的停机**：记录了由于数据库失误导致的 **15 分钟停机**，但该问题随后已被修复。
   - 未提供关于此次停机影响的更多细节。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/google/gemini-flash-8b-1.5-exp">Gemini Flash 8B 1.5 Experimental - API, Providers, Stats</a>: Gemini 1.5 Flash 8B Experimental 是 [Gemini 1. 的一个实验性的 8B 参数版本。通过 API 运行 Gemini Flash 8B 1.5 Experimental</li><li><a href="https://openrouter.ai/models/google/gemini-flash-1.5-exp>">Gemini Flash 1.5 - API, Providers, Stats</a>: Gemini 1.5 Flash 是一个基础模型，在各种多模态任务中表现良好，如视觉理解、分类、摘要以及从图像、音频和视频创建内容...
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1278792284514549924)** (2 条消息): 

> - `Daun.ai 发布`
> - `AI Chat CLI 工具` 


- **祝贺 Daun.ai 发布！**：社区表达了兴奋之情，成员们纷纷祝贺 **Daun.ai** 团队最近的发布。
   - 这种情绪反映了人们对新 AI 工具日益增长的兴趣和积极反响。
- **GitHub 上的全能 AI CLI 工具**：一位成员分享了 [AI Chat CLI Tool](https://github.com/sigoden/aichat) 的链接，该工具具有 Chat-REPL、Shell Assistant、RAG、AI tools & agents 等功能，并支持访问包括 OpenAI 和 Claude 在内的各种平台。
   - 该项目被誉为 AI 交互的综合解决方案，集成了多种功能以提升用户体验。



**提到的链接**：<a href="https://github.com/sigoden/aichat">GitHub - sigoden/aichat: All-in-one AI CLI tool featuring Chat-REPL, Shell Assistant, RAG, AI tools &amp; agents, with access to OpenAI, Claude, Gemini, Ollama, Groq, and more.</a>：集 Chat-REPL、Shell Assistant、RAG、AI tools &amp; agents 于一体的全能 AI CLI 工具，支持访问 OpenAI、Claude、Gemini、Ollama、Groq 等。 - sigoden/aichat

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1278791605289095269)** (146 messages🔥🔥): 

> - `OpenRouter 反馈`
> - `Cohere 模型更新`
> - `实验性模型的速率限制 (Rate Limiting)`
> - `Perplexity 模型问题`
> - `基础设施停机时间` 


- **OpenRouter 用户报告问题并提出建议**：用户对聊天中的默认模型和前端问题表示担忧，要求进行改进并与开发人员直接沟通。
   - 一位用户提到可以提供屏幕录像，以方便对这些前端问题进行故障排除。
- **Cohere 更新引发关注**：讨论集中在 Cohere 的 Command R 模型的最新更新上，重点介绍了 API 访问的新功能和定价结构。
   - 用户渴望尝试新功能，但对 OpenRouter 如何处理安全模式表示疑问。
- **实验性模型遇到速率限制**：用户报告在使用实验性模型时遇到速率限制错误，突显了测试这些新功能的挑战和局限性。
   - 讨论了通过 API 处理安全设置的影响，以及对端点 (endpoint) 设置的默认值的困惑。
- **Perplexity 模型错误报告**：一位用户报告收到了关于不再有效的模型的错误，表明模型 ID 和可用性存在问题。
   - 另一位用户确认该问题正在积极解决中，并建议使用特定频道进行进一步讨论。
- **停机担忧中的基础设施升级**：用户对停机时间增加表示担忧，促使官方回应称正在进行基础设施升级，旨在减轻系统压力。
   - 开发人员承认了最近的停机事件，将其归因于数据库容量问题，并概述了在不久的将来提高系统整体稳定性的计划。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/alibaba_qwen/status/1829187292038115413?s=46&t=stOPrwZiN_fxSK0RuC8Flg">来自 Qwen (@Alibaba_Qwen) 的推文</a>: 要访问 Qwen2-VL-72B，暂时应按以下方式使用我们的官方 API：</li><li><a href="https://openrouter.ai/chat">聊天室 | OpenRouter</a>: LLM 聊天室是一个多模型聊天界面。添加模型并开始聊天！聊天室将数据本地存储在您的浏览器中。</li><li><a href="https://api.together.ai/models/Qwen/Qwen1.5-4B-Chat">未找到标题</a>: 未找到描述</li><li><a href="https://docs.perplexity.ai/docs/model-cards.","type":"invalid_model","code":400}}">Perplexity API 入门 - Perplexity</a>: 未找到描述</li><li><a href="https://openrouter.ai/models/qwen/qwen-2-7b-instruct)">Qwen 2 7B Instruct - API, 提供商, 统计数据</a>: Qwen2 7B 是一款基于 Transformer 的模型，在语言理解、多语言能力、编程、数学和推理方面表现出色。它具有 SwiGLU 激活、Attention QKV 偏置和分组...</li><li><a href="https://cohereforai-c4ai-command.hf.space/">Cohere Command 模型</a>: Command R 模型针对各种用例进行了优化，包括推理、摘要和问答。由 Cohere 和 Cohere For AI 开发。</li><li><a href="https://x.com/OfficialLoganK/status/1828922199425548486)">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>: @DaveManouchehri 在 AI Studio 中免费。我不确定 Vertex 的实验性端点是否免费</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/gemini-experimental#pricing)">未找到标题</a>: 未找到描述</li><li><a href="https://x.com/OfficialLoganK/">来自 GitHub - FixTweet/FxTwitter 的推文</a>: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等 - FixTweet/FxTwitter</li><li><a href="https://github.com/Pythagora-io/gpt-pilot/issues">问题 · Pythagora-io/gpt-pilot</a>: 第一个真正的 AI 开发者。通过在 GitHub 上创建账户，为 Pythagora-io/gpt-pilot 的开发做出贡献。</li><li><a href="https://openrouter.ai/rankings">LLM 排名 | OpenRouter</a>: 根据应用使用情况对语言模型进行排名和分析</li><li><a href="https://marketplace.visualstudio.com/items?itemName=PythagoraTechnologies.gpt-pilot-vs-code&ssr=false#review-details">Pythagora (GPT Pilot) Beta - Visual Studio Marketplace</a>: Visual Studio Code 扩展 - 第一个真正的 AI 开发者。</li><li><a href="https://huggingface.co/CohereForAI">CohereForAI (Cohere For AI)</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1278825342605201611)** (56 messages🔥🔥): 

> - `Embedding 权重 NaN 问题`
> - `压缩项目研究反馈`
> - `SAE 讨论`
> - `正则化技术`
> - `Vision Embedding 与 Vision Token` 


- **训练期间 Embedding 权重出现 NaN**：一位用户报告称，在训练开始几步后，Embedding 权重就变成了 **NaN**，可能是由于损失函数中的分母被四舍五入为零导致的。
   - 进一步调查表明，他们的数据依赖衰减项是问题的根源，通过追踪梯度帮助定位了该问题。
- **寻求压缩研究的反馈**：博士生 Jeremy Vonderfecht 正在寻求关于使用主流扩散模型（如 **Stable Diffusion**）进行图像压缩的研究想法的反馈。
   - 成员们建议使用当前频道和另一个指定频道来分享想法，显示出一种欢迎讨论的氛围。
- **关于 SAE 和输入的澄清**：有一场关于 SAE 背景下术语 **x** 的澄清讨论，涉及对其在网络中作用的误解。
   - 成员们强调了在讨论中明确前提的重要性，特别是在处理激活向量输入的功能时。
- **正则化技术研究**：一位用户讨论了潜在的正则化策略，例如强制输入的均值为零，或使用 Batch Normalization 来稳定训练。
   - 讨论澄清了任何可能减慢优化过程的因素都可能是有害的，并强调了损失函数设计的谨慎性。
- **Vision Embedding 与 Vision Token 的优势对比**：有人提出了关于 **Vision Token** 视觉嵌入相对于传统方法的优势问题，强调了目前对其优缺点的认识尚不清晰。
   - 讨论承认 Vision Token 可能具有更原生的应用场景，促使人们在视觉任务背景下进一步探索其优势。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1278803733127495800)** (88 条消息🔥🔥): 

> - `模型中的动态专家路由 (Dynamic Expert Routing)`
> - `AI Safety 中的对抗性方法`
> - `语言模型中的 Tokenization 挑战`
> - `多 Token 预测 (Multi-Token Prediction) 效率`
> - `模型量化技术` 


- **动态专家路由增强模型训练**：允许模型在训练期间定义自己的专家，而不是使用固定配置，这一概念被讨论为提高适应性的一种方式。
   - 成员们指出，这一想法与 [LayerSkip 论文](https://medium.com/@techsachin/layerskip-faster-llm-inference-with-early-exit-and-self-speculative-decoding-3110cb93c94e) 中提出的方法等正在进行的研究相关。
- **探索 AI Safety 中的对抗性方法**：有人建议将对抗性策略作为 AI Safety 讨论中的一个关键关注领域。
   - 这种观点强调了探索 AI 系统底层漏洞的重要性。
- **Tokenization 给语言模型带来挑战**：参与者讨论了 Tokenization 的局限性，特别是针对非拉丁语言以及它为模型训练增加的复杂性。
   - 人们担心 Tokenization 会模糊重要的数据特征并降低训练效率。
- **多 Token 预测 (Multi-token prediction) 的有效性引发争论**：讨论强调，多 Token 预测 (MTP) 的效率可能不会显著惠及较小的语言模型，甚至在较大的模型中也无法提高训练速度。
   - 关于 MTP 的计算成本是否与其带来的模型性能提升相匹配，目前仍存在争议。
- **探索模型量化方法**：讨论了引入有限标量量化 (Finite Scalar Quantization, FSQ) 作为传统矢量量化 (Vector Quantization) 技术的一种潜在有效且更简单的替代方案。
   - 如相关论文所述，FSQ 方法有望在各种任务中提高性能，其对 Token 利用率的影响被认为非常重要。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2408.15495">Remove Symmetries to Control Model Expressivity</a>: 当损失函数中存在对称性时，模型很可能会陷入被称为“崩溃”的低容量状态。陷入这些低容量状态会导致...</li><li><a href="https://arxiv.org/abs/2403.00417">Rethinking Tokenization: Crafting Better Tokenizers for Large Language Models</a>: Tokenization 显著影响语言模型 (LMs) 的性能。本文追溯了 Tokenizer 从词级到子词级的演变，分析了它们如何平衡 Token 和类型...</li><li><a href="https://arxiv.org/abs/2408.16532">WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling</a>: 语言模型已有效地应用于建模自然信号，如图像、视频、语音和音频。这些模型的一个关键组件是编解码 Tokenizer，它压缩高维...</li><li><a href="https://arxiv.org/abs/2406.07548">Image and Video Tokenization with Binary Spherical Quantization</a>: 我们提出了一种基于 Transformer 的新型图像和视频 Tokenizer，采用二进制球面量化 (BSQ)。BSQ 将高维视觉嵌入投影到低维超球面上，然后应用...</li><li><a href="https://arxiv.org/abs/2309.15505">Finite Scalar Quantization: VQ-VAE Made Simple</a>: 我们建议在 VQ-VAEs 的潜在表示中，用一种称为有限标量量化 (FSQ) 的简单方案取代矢量量化 (VQ)，我们将 VAE 表示投影到有限的...</li><li><a href="https://arxiv.org/abs/2310.05737">Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation</a>: 虽然大语言模型 (LLMs) 是语言生成任务中的主导模型，但它们在图像和视频生成方面的表现不如扩散模型。为了有效地将 LLMs 用于...</li><li><a href="https://medium.com/@techsachin/layerskip-faster-llm-inference-with-early-exit-and-self-speculative-decoding-3110cb93c94e">LayerSkip: faster LLM Inference with Early Exit and Self-speculative decoding</a>: 简介
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1279061226571169802)** (5 messages): 

> - `Word Game Bench`
> - `Consistency Measurement` (一致性测量)
> - `Dataset Construction` (数据集构建)


- **Introducing Word Game Bench for Language Models**: **Word Game Bench** 是一个旨在评估语言模型在 **Wordle** 和 **Connections** 等文字拼图游戏中表现的基准测试，目前没有任何模型的平均胜率超过 **50%**。它强调交互和反馈的整合，并采用独特的测试集管理方法，避免固定评估以防止泄漏 (leakage)。
   - 更多详情请访问 [Word Game Bench](https://wordgamebench.github.io)，并查看 [@zafstojano](https://x.com/zafstojano/status/1829398835585520076) 在 Twitter 上的发布公告。
- **Measuring Consistency in Responses**: 一位成员正在探索比较多项选择题回复的方法，以评估当 prompts 略有变化时的一致性，建议使用 `process_results` 和聚合函数。他们转换了数据集，为相同的问题包含重复条目以及不同的 prompts 以供比较。
   - 另一位成员建议，直接使用该库可能并不简单，并建议构建代表所需内容的特定数据集，尽管这需要为每个模型进行单独设置。
- **Adjusting Prompts for Consistency Analysis**: 有建议提出在同一数据集上多次运行模型，每次运行时更改 prompts，以方便对回复进行比较。该策略涉及使用 `doc_to_text` 来集成其他 prompts，从而测量回复的偏差。
   - 这种方法强调需要仔细处理数据集，以确保准确的比较并避免数据处理过程中的错误。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://wordgamebench.github.io">Word Game Bench</a>: 未找到描述</li><li><a href="https://x.com/zafstojano/status/1829398835585520076">zafir (@zafstojano) 的推文</a>: 很高兴分享 &#34;Word Game Bench&#34; —— 一个用于评估语言模型在文字拼图游戏上表现的有趣基准测试！这是一个相对困难的基准测试，目前没有任何模型的平均得分超过 50%...
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1278814192404664381)** (1 messages): 

> - `Discord server growth` (Discord 服务器增长)
> - `Community appreciation` (社区感谢)


- **Discord server hits 100K members!**: Discord 服务器成员正式突破 **10万**，标志着社区的一个重要里程碑。
   - 团队向所有成员的支持和反馈表示了*由衷的感谢*，并对持续的增长感到兴奋。
- **Community's incredible support recognized**: 团队对社区在增长阶段提供的所有*支持和反馈*表示感谢。
   - 他们很高兴能与每一位为活跃氛围做出贡献的成员一起进化并继续这段*旅程*。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1278814360260710411)** (120 条消息🔥🔥): 

> - `Subscription Issues` (订阅问题)
> - `AI Model Performance` (AI 模型性能)
> - `Event Announcements` (活动公告)
> - `AI Exhibition in France` (法国 AI 展览)
> - `User Experience Issues` (用户体验问题)


- **周期性订阅问题**：多位用户报告其 Pro 订阅消失或无法正常工作，并建议联系支持部门以澄清代金券（voucher）相关问题。
   - 一位用户对未收到申请确认表示担忧，凸显了用户支持方面可能存在的问题。
- **关于 AI 模型行为的疑问**：一位用户质疑其选定的 AI 模型是否正常工作，指出即使切换了模型，答案依然雷同，引发了关于 Bug 的猜测。
   - 讨论中提到了模型识别响应中存在的不一致性，表明可能的更新影响了用户体验。
- **活动更新与会议**：一位组织者宣布将在法国举办 AI 展览，并请求提供宣传材料和资源，以便有效地展示 Perplexity AI。
   - 用户对标准 YouTube 资源以外的宣传内容表现出兴趣。
- **用户界面关注点**：多位用户报告经历了 Thread 被删除或查询提交无法通过的问题，对内容丢失表示沮丧。
   - 一些用户分享了解决这些问题的排查策略，表明需要提高系统的可靠性。
- **关于模型使用限制的讨论**：用户讨论了模型使用限制随时间的变化，注意到容量从历史上的 600 次变动到当前的限制，这反映了定价策略的调整。
   - 对话强调了理解模型限制如何影响用户体验和预期的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://rohansai22.github.io/resume/">Maragoni Rohan Sai - Portfolio</a>: 未找到描述</li><li><a href="https://tenor.com/view/griffith-berserk-eclipse-guts-berserk-anime-meme-gif-10622855093064880455">Griffith Berserk GIF - Griffith Berserk Eclipse - Discover &amp; Share GIFs</a>: 点击查看 GIF
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1278946013813542966)** (10 条消息🔥): 

> - `MrBeast News` (MrBeast 新闻)
> - `C++ Programming` (C++ 编程)
> - `Vikings Influence` (维京人的影响)
> - `OpenAI's DALL-E` (OpenAI 的 DALL-E)
> - `Muscle Knots` (肌肉结节)


- **MrBeast 发生了什么？**：一位成员分享了一篇文章链接，讨论了关于 **MrBeast** 活动和事业的最新动态，可以在[这里](https://www.perplexity.ai/search/what-happened-to-mrbeast-S0hJBJ01TSKV6CqiLDXnvw)查看。
   - 这可以为他的内容方向或商业投资的变化提供见解。
- **C++ 编程基础**：分享了一个链接，概述了如何在社区帮助下编写 C++ 程序，可以通过[这里](https://www.perplexity.ai/search/write-a-c-plus-plus-program-fo-aJscZujqQZGLq2_8THGP5A)访问。
   - 该文章可能涵盖了初学者的基本概念和示例。
- **深入探讨维京人的贡献**：一位用户提到探索 **Vikings**（维京人）对现代文化的影响，并分享了资源链接，见[这里](https://www.perplexity.ai/search/what-have-vikings-done-for-mod-Cb_PHCx7Ty2cDQZVa14iJA)。
   - 这可以提供关于他们的遗产和影响的全面视角。
- **了解 DALL-E**：分享了一个讨论 **OpenAI's DALL-E** 的链接，可以在[这里](https://www.perplexity.ai/search/openai-s-dall-e-0eZkD0GfRliPUTnsBpKBIQ)找到。
   - 它可能涵盖了其功能、能力和应用。
- **什么是肌肉结节？**：一位成员询问了关于 **muscle knots**（肌肉结节）的问题，并链接到一篇科普文章 *[这里](https://www.perplexity.ai/search/what-are-muscle-knots-also-kno-.GsfiArjRTW.5wmBcUIYtA)*，讨论了其成因和治疗方法。
   - 这可以帮助许多人理解并缓解这一常见问题。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1278811459739979888)** (9 条消息🔥): 

> - `Pro API 积分问题`
> - `Pro Searches 可用性`
> - `API 速率限制`
> - `API 账户支持` 


- **用户报告 Pro API 积分缺失**：包括 @mihir2033 在内的多位用户报告称，在购买 Pro 会员后未收到 **$5 PPLX API 积分**。
   - 他们正在积极寻求支持，并分享账户详情以寻求解决。
- **Pro Searches 在 API 上无法使用**：@balapete 对 **Pro Searches** 是否能在 API 中运行表示不确定，并提到正在使用 **llama-3.1-sonar-huge-128k-online**。
   - 用户 @ok.alex 确认 **Pro** 目前无法通过 API 使用，这让用户们好奇何时会开放此功能。
- **遇到速率限制错误**：@nicconike 分享了在调用 API 时遇到 **429 Client Error: Too Many Requests** 的经历，并询问原因。
   - 这一问题凸显了 API 可能存在的限制或使用上限对功能造成了影响。


  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1278792607710969916)** (70 条消息🔥🔥): 

> - `MMLU 与模型性能`
> - `Command R+ 更新`
> - `Cohere Chat 界面`
> - `GQA 与吞吐量提升`
> - `Cohere Scholars Discord` 


- **MMLU 与实际用途不相关**：一位成员提到 **MMLU** 与构建实用的 LLM 之间并没有强相关性，并举例说明了关于弗洛伊德理论等过时话题的题目。
   - 他们指出，由于 MMLU 数据在互联网上的存在感更强，模型刷新正在提升其在该基准测试上的表现。
- **Command R+ 展示了令人印象深刻的更新**：[Command R+ 08-2024](https://cohere.com/blog/command-series-0824) 相比前代产品，在多语言检索增强生成（RAG）和性能指标上都有所提升，包括吞吐量提高了 50%。
   - 成员们讨论了 Command R 现在如何与更大的 Command R+ 模型并驾齐驱，展示了稳健的性能提升。
- **对 Cohere Chat 界面的关注**：用户询问 **Cohere chat 界面** 是否针对新模型进行了更新，一些人提到界面保持不变。
   - 讨论中还涉及了聊天界面缺乏夜间/深色模式选项的问题。
- **GQA 在吞吐量提升中的作用**：**GQA**（Grouped Query Attention）的引入被视为 Command R 模型更新中吞吐量提升的关键因素。
   - 关于吞吐量的增加是否也归功于新的量化（Quantization）方法，各方意见不一。
- **加入 Cohere Scholars Discord**：有人询问如何加入 **Cohere Scholars Discord**，得到的指导是在 Cohere 网站上查找“Join Us”按钮。
   - 几位成员对社区氛围和正在开展的工作表示了赞赏。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.cohere.com/docs/safety-modes">Safety Modes — Cohere</a>：安全模式文档介绍了如何使用默认模式和严格模式，以便对模型输出进行额外控制。</li><li><a href="https://docs.cohere.com/changelog/command-gets-refreshed">Command models get an August refresh — Cohere</a>：未找到描述</li><li><a href="https://cohere.com/blog/">The Cohere Blog</a>：探索我们收集的富有洞察力的博客文章，涵盖各种生成式 AI 主题。我们的文章提供深入分析、专家意见和实用建议，以提供信息和启发。</li><li><a href="https://docs.cohere.com/">Cohere Documentation — Cohere</a>：未找到描述</li><li><a href="https://huggingface.co/CohereForAI/c4ai-command-r-plus-08-2024">CohereForAI/c4ai-command-r-plus-08-2024 · Hugging Face</a>：未找到描述</li><li><a href="https://cohere.com/blog/command-series-0824">Updates to the Command R Series</a>：最新版本的 Command R 模型系列在代码、数学、推理和延迟方面均有改进。</li><li><a href="https://huggingface.co/datasets/joey234/mmlu-human_sexuality-original-neg">joey234/mmlu-human_sexuality-original-neg · Datasets at Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1279078308243439656)** (6 messages): 

> - `Command R 和 R+ 模型更新`
> - `模型在不同平台上的可用性`
> - `微调 (Fine-tuning) 默认设置`
> - `新模型的基准测试 (Benchmarks)` 


- **Command R 和 R+ 模型迎来重大更新**：Cohere 宣布更新了 **Command R** 和 **R+** 模型，提升了在**推理 (Reasoning)**、**编程 (Coding)** 和**多语言 RAG** 方面的**性能 (Performance)**，现在可以通过别名 `command-r-08-2024` 和 `command-r-plus-08-2024` 使用。
   - 更新后的模型还具有**更低的每 Token 定价**，其中 R 模型非常便宜，输入 Token 仅需 **$0.15**。
- **新模型在各平台上的可用性**：社区成员确认更新后的模型已在 **Hugging Face** 上可用，并将在转换后最终进入 **Ollama**。
   - 他们强调需要时间来对模型进行适当的**量化 (Quantized)** 并上传到其他平台。
- **关于新模型微调默认设置的咨询**：一位用户询问新的 **Command 模型** 是否将作为微调 (Fine-tuning) 的默认模型。
   - 目前没有直接回应，但该问题表明了在微调场景中应用更新模型的兴趣。
- **呼吁发布新模型的基准测试**：一位用户请求发布新模型的**基准测试 (Benchmarks)** 以评估其性能。
   - 这显示了社区对定量评估更新模型的渴望。



**Link mentioned**: <a href="https://docs.cohere.com/changelog/command-gets-refreshed">Command models get an August refresh — Cohere</a>: no description found

  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1278833174037925909)** (10 messages🔥): 

> - `C4AI Scholars Program`
> - `Command R+ 发布`
> - `GDPR 合规性` 


- **关于 C4AI Scholars Program 资格的咨询**：一位成员询问 **C4AI Scholars Program** 是否接受在读研究生，可能采用类似于暑期实习但在 1 月份开始的形式。
   - 另一位成员建议直接联系 **C4AI** 以获取明确答复。
- **关于 Command R+ 发布的讨论**：一位成员询问了最新版本 **Command R+** 的潜在发布情况。
   - 该问题没有得到明确回应，发布状态仍不确定。
- **提出 GDPR 合规性问题**：一位成员询问了 **Cohere** 在使用 API 方面对 **GDPR** 法规的合规性，特别是关于 **Command R+** 相关训练的数据使用情况。
   - 另一位成员分享了 **Cohere Trust Center** 的链接，表示该中心应能提供关于合规性查询的全面解答。



**Link mentioned**: <a href="https://cohere-inc.secureframetrust.com/">  Cohere Inc | Trust Center
</a>: no description found

  

---

### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1279021162814505012)** (46 messages🔥): 

> - `API Rate Limiting`
> - `Citations Management`
> - `Safety Mode Interaction`
> - `Trial Key Limitations`
> - `Financial Data Analysis App` 


- **API 测试版密钥限制导致错误**：一位用户在使用测试版 API Key 时遇到了 **速率限制错误 (429)**，表明他们超过了 **每月 1,000 次 API 调用** 的限制。
   - 几位成员确认需要 **Production Key** 才能避免这些限制，并建议添加信用卡以获取更高级别的访问权限。
- **处理输出中的引用过载**：一位成员反映一段 **180 字的文本** 出现了过多的引用，希望对其进行限制，并询问如何优先保留最重要的引用。
   - 建议对引用进行 **Rerank** 并仅分享参考资料，这一方案被认为是一个可行的解决办法。
- **安全模式与 Preamble 之间的交互**：官方澄清了新的 `safety_mode` 不会覆盖自定义的 `preamble`，它们在生成响应时独立运行。
   - 测试显示，当 **Safety Modes** 激活时，它们会通过将安全指令与用户 `preamble` 相结合来相应地修改 Prompt。
- **无需信用卡即可使用测试版密钥**：参与者讨论了在不输入信用卡信息的情况下使用测试版 API Key 的可行性，确认测试访问是允许的。
   - 值得注意的是，虽然测试版密钥有限制，但如果坚持使用测试选项，则不需要填写卡片信息。
- **构建金融数据分析应用**：一位用户分享称他们正在开发一个专注于 **金融数据分析** 的应用程序，并利用引用功能来确保数据的准确性。
   - 成员们表现出极大的热情并提供了支持，认可了此类工具在金融领域的潜在影响力。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.cohere.com/docs/rate-limits">API Keys and Rate Limits — Cohere</a>: 此页面描述了 Cohere API 的相关限制。</li><li><a href="https://docs.cohere.com/reference/rerank">Rerank — Cohere</a>: 该端点接收一个查询和一组文本列表，并生成一个按相关性评分排序的数组。
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1279088423038222451)** (1 messages): 

> - `Maya LLaVA-Pretrain Dataset`
> - `Large-scale multilingual datasets`
> - `Image Captioning and VQA`
> - `Translation quality results`
> - `API support and queries` 


- **Maya LLaVA-Pretrain 数据集发布**：**Maya LLaVA-Pretrain** 数据集现已发布，包含跨 **8 种语言** 的 **4,404,776** 条条目，专为预训练大型语言与视觉模型（VLM）而设计。
   - 该数据集是在原始 [llava-pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) 英语数据集的基础上，通过机器翻译和毒性过滤扩展而来的。
- **使用强大的 API 准备数据集**：该数据集是使用 **c4ai-aya-35B** 模型 API 准备的，并使用 **command-r-plus** API 进行了精细化处理，以增强毒性控制。
   - 成员们向另一位回答了有关批处理（Batch Processing）和 API 支持问题的用户表示了感谢。
- **即将发布翻译质量结果**：团队计划近期在数据集卡片上展示 **翻译质量结果**。
   - 这与其提高数据集在图像字幕（Image Captioning）和视觉问答（VQA）任务中可用性的目标一致。



**提及的链接**: <a href="https://huggingface.co/datasets/kkr5155/Maya-llava-pretrain">kkr5155/Maya-llava-pretrain · Datasets at Hugging Face</a>: 未找到描述内容。

  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1278795696421732435)** (31 messages🔥): 

> - `Codeium funding`
> - `Meta AI Assistant growth`
> - `Google DeepMind's Gems`
> - `State of Code Generation`
> - `Tome pivot` 


- **Codeium 筹集 1.5 亿美元，评估融资策略**：Codeium 完成了由 General Catalyst 领投的 1.5 亿美元 Series C 融资，投后估值为 **12.5 亿美元**，自发布以来总融资额接近 **2.43 亿美元**。
   - 联合创始人 Varun Mohan 表示，他们尚未动用 **6500 万美元** 的 Series B 融资，展示了其战略性的融资方式。
- **Meta 的 AI 助手数据表现亮眼**：Aravind Srinivas 报告称，Meta 的 AI 助手实现了 **4 亿 MAU** 和 **4000 万 DAU**，表明用户参与度极高。
   - 随着服务规模的扩大，讨论中涉及了潜在的许可需求，该助手近期的表现突显了其日益增长的采用率。
- **Google DeepMind 推出可定制的 Gems**：Google DeepMind 宣布推出可定制的 **Gems**，这是其 Gemini 模型的专业版本，可作为各种场景的主题专家。
   - **Learning Coach** 和 **Coding Partner** 等功能旨在增强用户交互，这取决于无缝的集成和执行。
- **代码生成工具的进展**：最近的报告强调了 **Townie** 和 **Claude 3.5 Sonnet** 等工具在代码生成方面的重大进展，通过对话式界面增强了软件开发。
   - 用户表示希望工具能够允许修改现有应用程序，而不仅仅是从头开始创建新程序，强调了对灵活性的需求。
- **Tome 转型专注于企业级 AI**：Tome 宣布转型为一款 AI 助手，旨在帮助用户开拓新的企业客户，标志着战略重心的转变。
   - 公司代表分享了这一新方向，概述了影响这一决定的历程和变化。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/AravSrinivas/status/1829261003164696703">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>: 令人印象深刻的数据</li><li><a href="https://x.com/1x_tech/status/1829567690681307284?s=46">来自 1X (@1x_tech) 的推文</a>: 介绍 NEO Beta。为人而设计。为家庭而造。</li><li><a href="https://x.com/hliriani/status/1829284172470620613?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Henri Liriani (@hliriani) 的推文</a>: 我们正在重启 Tome，使其成为一家不同的公司。@magicaltome 现在是一款用于开拓新企业客户的 AI 助手。这里有关于我们所经历的历程的一些介绍……</li><li><a href="https://x.com/GoogleDeepMind/status/1828855383131074997">来自 Google DeepMind (@GoogleDeepMind) 的推文</a>: 在接下来的几天里，开始创建 Gems 并与之聊天：这是 Gemini 的可定制版本，可作为主题专家。🤝 我们还将针对不同场景推出预设的 Gems - 包括 Learni...</li><li><a href="https://x.com/Ahmad_Al_Dahle/status/1829541138736509102">来自 Ahmad Al-Dahle (@Ahmad_Al_Dahle) 的推文</a>: 紧随我昨天分享的 Llama 更新之后，我们也看到 Meta AI 的使用量正在飞速增长，周活跃用户达到 1.85 亿！🚀</li><li><a href="https://blog.val.town/blog/codegen/">我们如何构建 Townie —— 一个生成全栈应用的 App</a>: 类似于 Claude Artifacts，但带有后端和数据库</li><li><a href="https://docs.cohere.com/changelog/command-gets-refreshed">Command 模型获得 8 月更新 — Cohere</a>: 未找到描述</li><li><a href="https://www.1x.tech/androids">我们的机器人 | 1X Technologies</a>: 受人类天性启发。认识 EVE 和 NEO，了解更多关于它们如何利用具身学习解决问题的信息，从满足劳动力需求到日常协助。</li><li><a href="https://techcrunch.com/2024/08/29/github-copilot-competitor-codeium-raises-150m-at-a-1-25b-valuation/">GitHub Copilot 竞争对手 Codeium 以 12.5 亿美元估值融资 1.5 亿美元 | TechCrunch</a>: Codeium 是一家开发 AI 驱动工具以对抗 GitHub Copilot 的初创公司，已以 12.5 亿美元的估值筹集了 1.5 亿美元。</li><li><a href="https://techcrunch.com/2024/0">2024 | TechCrunch</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1278805134695989333)** (1 条消息): 

> - `LLM benchmarks`
> - `Nicholas Carlini`
> - `Latent Space podcast`
> - `Community meetup` 


- **与 Nicholas Carlini 合作的新播客集**：最新一期的 [Latent Space podcast](https://x.com/latentspacepod/status/1829173832877519152) 邀请了来自 **Google DeepMind** 的 Nicholas Carlini，讨论了关于 LLM 的个人见解和基准测试。
   - 关键话题包括 *他如何使用 AI*、他的基准测试方法，以及对 *从 LLM 中提取训练数据* 的批判性观点，特别是提到了 *OpenAI logprobs* 的停用。
- **社区聚会预告**：对一位成员组织的即将于下个月举行的社区聚会进行了预告。
   - 聚会活动的详情预计将汇集 AI 爱好者和从业者进行社交和讨论。



**提到的链接**：<a href="https://x.com/latentspacepod/status/1829173832877519152">来自 Latent.Space (@latentspacepod) 的推文</a>：🆕 为什么你应该编写自己的 LLM 基准测试，对话 @GoogleDeepMind 的 Nicholas Carlini。涵盖了他的热门内容：- 我如何使用 AI - 我的 LLM 基准测试 - 提取训练数据...

  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1279169268168265750)** (57 条消息🔥🔥): 

> - `Research Paper Generation Techniques`
> - `Ambassador Program Assistance`
> - `AI Scientist Limitations`
> - `CogVLM Introduction`
> - `UI/UX Patterns for GenAI` 


- **研究论文生成技术引发辩论**：成员们讨论了对研究论文生成方法的偏好；一些人建议迭代反馈可能比 one-shot 产生更好的结果。
   - 一位成员指出，仅仅依赖 “one-shot” 方法可能会导致**繁琐的人工验证**。
- **对大使计划（Ambassador Program）帮助的兴趣**：一位成员分享了他们过去的经验，并表示愿意在建立 **Ambassador program** 方面提供帮助。
   - 他们澄清道：*“不过我不是一个 AI research agent，”* 为他们的乐于助人增添了一丝幽默。
- **CogVLM 模型引发疑问**：**CogVLM** 的引入引发了讨论，有人质疑其在生成的论文中的相关性，促使一位成员评价其看起来像是 **LLM 堆砌的内容（LLM barf）**。
   - *“除非我理解错了，”* 一位成员反思道，暗示需要对该话题进行进一步澄清。
- **探索 AI Scientist 的局限性**：成员们评论了 **AI Scientist 的局限性**，引发了关于使 AI 更加有效的持续挑战的见解。
   - 有人分享了一个帖子，质疑什么才是真正对用户有益的透明度，并补充道：*“我不认为那里有什么实质内容。”*
- **呼吁 GenAI 中的 UI/UX 模式**：讨论包括即将举行的关于 **GenAI 的 UI/UX 模式**的会议，并分享了各种资源的链接。
   - 提到的关键资源之一是 [Maggie Appleton 的作品](https://maggieappleton.com/squish-structure)，强调了创新的界面方法。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/jimmykoppel/status/1828077206204981423">来自 Jimmy Koppel (@jimmykoppel) 的推文</a>：但所有这些都是为了阻止你过于仔细地观察他们实际在做什么。因为我不认为那里有什么实质内容。</li><li><a href="https://storm.genie.stanford.edu/">未找到标题</a>：未找到描述</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=0#gid=0">AI In Action: Weekly Jam Sessions</a>：未找到描述</li><li><a href="https://github.com/THUDM/CogVLM">GitHub - THUDM/CogVLM: a state-of-the-art-level open visual language model | 多模态预训练模型</a>：一个达到 state-of-the-art 级别的开源视觉语言模型 | 多模态预训练模型 - THUDM/CogVLM
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1278890172225683557)** (34 messages🔥): 

> - `Mojo 在区块链中的应用`
> - `Mojo 的开源进程`
> - `性能对比：Mojo vs Go vs C`
> - `Mojo 开发中的社区参与`
> - `与 OPENSEA 的合作` 


- **Mojo 在区块链协议中的潜力**：关于将 **Mojo** 用于区块链协议的讨论正在进行中，一位开发者指出，与 **Go、Rust 和 C++** 相比，它目前还不够成熟。
   - 有评论提到 **Mojo** 和 **Go** 是最有竞争力的语言，但 **Go 20% 的性能损失**对某些项目来说可能是至关重要的。
- **关于 Mojo 开源未来的疑问**：有人询问了 **Mojo 编译器源代码**的可用性，目前该代码仍为闭源状态。
   - **Modular 团队**旨在开发速度与社区参与之间取得平衡，并表示他们可能还不确定何时或是否会将其开源。
- **性能对比见解**：成员们辩论了 **Go** 相对于 **C** 的性能，声称在各种任务中速度较慢，引发了关于 Go 优化策略的细致讨论。
   - Darkmatter 强调，在更复杂的场景中，**Go 的性能可能会大幅下降**，并引用了每秒 **30 个请求**的容量与 **C 的 100 个请求**的对比。
- **社区参与和开发者角色**：有关于扩大 **Modular 团队**兴趣的对话，特别是寻找在 **MLIR 和编译器**方面有经验的人才。
   - 挑战在于平衡开发者资源与社区参与，同时保持项目高效推进。
- **与 OPENSEA 的合作**：宣布了与 **OPENSEA** 合作进行新的免费铸造（free mint），鼓励服务器用户参与。
   - 参与者被引导至一个领取链接，并注明某些领取可能会产生 Gas 费用。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.modular.com/legal/max-mojo-license">Modular: MAX &amp; Mojo 社区许可证</a>：MAX SDK ("MAX") &amp; Mojo 社区许可证规定了我们允许如何使用我们的软件，以及你如何通过它改变世界。</li><li><a href="https://docs.modular.com/max/faq#will-it-be-open-sourced">MAX 常见问题解答 | Modular 文档</a>：关于 MAX Engine 预期问题的解答。</li><li><a href="https://www.modular.com/company/career-post?4419827005&gh_jid=4419827005)">Modular: 职业职位</a>：在 Modular，我们相信优秀的文化是创建伟大公司的关键。我们工作的三个支柱是：打造用户喜爱的产品、赋能他人、以及成为一支不可思议的团队。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1278792965090967552)** (15 messages🔥): 

> - `内存管理见解`
> - `间接层 (Layers of Indirection)`
> - `设计灵活性`
> - `Mojo 文件输出`
> - `编辑器中的错误处理` 


- **架构师在内存管理中的角色**：一位成员表示，如果程序员不确定指针引用的内存是否应该被释放，这意味着系统架构师在设计上是失败的。
   - 他们强调内存管理不应该是应用程序员关心的问题，这表明需要坚实的架构设计。
- **对间接层的赞美**：一位成员分享了对他们正在处理的“优美的间接层”的兴奋之情，表示对进展反应积极。
   - 他们指出，这种架构几乎适用于所有情况，这增加了他们的满意度。
- **将查找表输出到 Mojo 文件**：另一位成员宣布计划创建一个简单的脚本，用于生成包含可定制查找表的 `.mojopkg` 文件。
   - 这反映了在软件开发过程中改进功能的持续努力。
- **Tuple 中的错误处理**：一位成员指出，编辑器中仍然会报告 **Tuple** 的越界错误，影响了他们的开发体验。
   - 他们提到这可能与编辑器中的类型感知（type awareness）有关，建议改进方向可以包括更好地管理无效类型。
- **错误消息中对 InvalidType 的需求**：一位成员提议引入 `InvalidType` 消息可以增强错误报告的清晰度，特别是在类型不匹配的场景下。
   - 他们幽默地指出，这类消息将是 `Type != Type` 错误唯一能派上用场的时候。


  

---

### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1279140100516741121)** (2 messages): 

> - `fastai 模型导出`
> - `Modular 框架愿景` 


- **fastai 令人兴奋的导出想法**：一位成员建议在 fastai 中重写 **Learner.export**，以便在导出 **PyTorch 模型**的同时，导出用于输入流水线（input pipeline）的 **Mojo** 代码。
   - 这种方法可以增强生产环境中输入流水线与模型的集成。
- **Modular 的跨平台愿景**：有迹象表明 **Modular** 旨在解决 **pickle 问题**，并创建一种**跨平台且与框架无关的模型格式**。
   - 该计划预计将促进不同框架之间的兼容性和易用性。


  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1278800835765338175)** (46 messages🔥): 

> - `LangChain Function Calling 与流式传输`
> - `Docker 与 Ollama 的连接问题`
> - `为 HR 团队构建胜任的 GPT`
> - `LangChain 中的实时流式输出`
> - `GraphRAG 与传统 RAG 技术的对比` 


- **LangChain 的 Function Calling 与流式传输**：一位成员询问如何在使用 LangChain v2.0 时结合 function calling 和流式传输功能，并提到难以找到相关文档。
   - 另一位成员澄清说，虽然支持 function calling，但流式输出可能需要在 JavaScript 中进行特定配置或异步处理。
- **Docker 与 Ollama 的连接问题**：一位用户报告称，在将其调用 Ollama API 的 LangChain 应用容器化时出现连接拒绝错误，尽管在非容器化环境下运行正常。
   - 他们后来发现问题与基础 URL 配置有关，通过使用直接的 Ollama 主机 URL 解决了该问题。
- **为 HR 团队构建胜任的 GPT**：一位用户希望根据一份冗长的手册为 HR 团队创建一个专门的 GPT，强调需要减少幻觉并建立反馈机制。
   - 随后讨论了通过反馈、fine-tuning 以及实施替代 RAG 技术来改进 LLM 交互，以构建更高效的系统。
- **LangChain 中的实时流式输出**：一位用户在使用 LangChain 中的 agent executors 时遇到挑战，这些执行器在交付最终响应之前会收集所有输出，而不是实时流式传输。
   - 讨论建议探索 `streamRunnable` 选项，以实现潜在的实时输出流。
- **GraphRAG vs 传统 RAG 技术**：.removandesande 建议，虽然混合 RAG 方法可能有效，但针对他们的用例，他们更倾向于传统 RAG 技术而非 GraphRAG。
   - 对话暗示了探索 self-query 和大上下文 RAG 等新 RAG 方法作为极具前景的替代方案。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://api.python.langchain.com/en/latest/agents/langchain.agents.agent.AgentExecutor.html">langchain.agents.agent.AgentExecutor &mdash; 🦜🔗 LangChain 0.2.15</a>：未找到描述</li><li><a href="https://v02.api.js.langchain.com/classes/langchain.agents.AgentExecutor.html">AgentExecutor | LangChain.js</a>：未找到描述</li><li><a href="https://github.com/langchain-ai/langchain/issues/25022>).">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账户为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="http://ollama:11434">)">无标题</a>：未找到描述</li><li><a href="https://github.com/ollama/ollama/issues/6398">通过 Docker 运行 Ollama 时，它不会响应任何 API 调用或 Python 客户端库的请求 · Issue #6398 · ollama/ollama</a>：问题是什么？我在装有 RTX-4000 的 Ubuntu 22 机器上成功安装了 nvidia docker toolkit，并将 ollama 作为 docker 容器启动，暴露端口 11434：docker run -d --gpus=all --en...
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/)** (1 messages): 

资源发现：https://www.getaiphone.app/
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1278831898885357568)** (5 条消息): 

> - `GymNation 成功案例`
> - `生产环境中的 LLMs`
> - `LlamaIndex MLFlow 集成`
> - `LLM x Law 黑客松`
> - `增强型财务数据分析` 


- **GymNation 与 LlamaIndex 合作取得成功**：GymNation 与 LlamaIndex 合作提升成员体验，实现了**数字线索到销售转化率提高 20%**，以及数字线索 **87% 的对话率**。
   - 欲了解更多详情，请查看他们的[完整成功案例](https://t.co/CXsiySj4zq)。
- **关注 @seldo 讨论 LLMs**：不要错过 @[seldo] 在 **9 月 9 日**分享关于生产环境中 LLMs 的见解！你可以在这篇帖子中找到详情 [此处](https://t.co/Ozb1xTF2Lh)。
   - 这次讨论将为有效部署 LLM 技术提供宝贵的见解。
- **LlamaIndex 亮相 MLFlow 播客**：联合创始人 @jerryjliu0 参加了 **MLFlow 播客**，讨论了与 MLFlow 的新集成，该集成简化了 LlamaIndex 应用程序的日志记录和评估。
   - 点击[此处](https://t.co/2wwvn7HRBm)查看播客中的完整演示和见解。
- **加入 LLM x Law 黑客松！**：**9 月 8 日**将举行一场激动人心的 **LLM x Law 黑客松**，由 @hexapode 组织，重点关注 AI 与法律实践的融合。
   - 参与者可以探索包括 First-Build Track 在内的三个赛道，在[此处](https://t.co/AksB9V6akr)展示他们的 AI 开发技能。
- **利用 MoW 增强财务数据分析**：讨论了一种使用 **Mixture of Workflows (MoW)** 和 Corrective RAG 进行财务数据分析的创新方法，涵盖了 **Phi-3**、**Qwen-2**、**Gemma-2** 和 **Stablelm2** 等模型。
   - 该方法提供了对财务报表的**上下文感知分析**，更多详情请见[此处](https://t.co/CIaEwmWB0S)。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1278804321340756092)** (28 条消息🔥): 

> - `LlamaIndex API 中的警告`
> - `QueryEngine 弃用讨论`
> - `将 LlamA3 与 OpenAI 结合使用`
> - `在 LLM 中处理 JSON 数据`
> - `结合工具和 Workflow 步骤` 


- **LlamaIndex API 配置中的警告**：一位成员报告收到关于 V2 中配置键更改的 UserWarning，特别是提到 'allow_population_by_field_name' 被重命名为 'populate_by_name'。
   - 另一位成员建议这可能与所使用的 SQLAlchemy 版本有关。
- **关于 QueryEngine 弃用的澄清**：一位成员询问 QueryEngines 是否正在被弃用，因为在文档中发现了弃用方法的引用。
   - 社区澄清说，只是提取结构化输出的方法被弃用，而不是所有的 QueryEngines。
- **将 LlamA3 与 OpenAI 结合使用**：一位成员询问如何将 Llama3 与 OpenAI 结合使用来生成 QA embedding 对，寻求配置方面的澄清。
   - 另一位成员建议通过 Settings 全局设置 LLM 对象，或者将 LLM 作为 kwarg 传递给 'generate_qa_embedding_pairs'。
- **在 LLM Workflow 中处理 JSON 数据**：一位用户创建了一个 Agent 来进行返回 JSON 数据的外部 API 调用，并就如何为 LLM 格式化这些数据寻求建议。
   - 得到的指导是在将响应发送回 LLM 之前对其进行良好的格式化，以避免复杂化。
- **结合工具和 Workflow 步骤**：一位新用户询问有关在 LlamaIndex 中集成工具和 Workflow 步骤的示例，对两者之间的连接感到困惑。
   - 一位成员分享了一个具体示例，演示了如何构建一个集成 Workflow 和工具调用的 Agent。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/module_guides/querying/structured_outputs/query_engine/">(已弃用) Query Engines + Pydantic 输出 - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/workflow/function_calling_agent/">Function Calling Agent 的 Workflow - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1278950929756065876)** (1 messages): 

> - `LitServe`
> - `LlamaIndex`
> - `AI model serving` 


- **LitServe 增强 AI 模型部署**：LitServe 是一个高性能的推理服务引擎，允许开发者高效地部署和管理各种 **AI models**。
   - 与 **LlamaIndex** 搭配使用时，它会转化为构建智能应用程序的多功能工具。
- **结合 LitServe 和 LlamaIndex**：**LitServe** 与 **LlamaIndex** 的结合为开发者提供了一个强大的 AI 应用数据框架。
   - 这种协同效应为在真实场景中提供 AI 模型服务带来了更高的便利性和灵活性。



**提及链接**：<a href="https://medium.com/ai-artistry/serving-ai-models-at-lightning-speed-with-litserve-and-llamaindex-4e7decdb5ae1">Serving AI Models at Lightning Speed with LitServe and LlamaIndex</a>：Ankush k Singal

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1278797119464935457)** (11 messages🔥): 

> - `House Party`
> - `终端应用推荐`
> - `Obsidian OI 插件问题`
> - `GPT-4o 交互记忆` 


- **House Party 时间**：欢迎参加下周提前举行的 **House Party**，让大家更多地聚集在一起！[加入 Discord 活动](https://discord.gg/open-interpreter-1146610656779440188?event=1278796923892924498)。
   - 该活动旨在增强社区参与度并营造有趣的氛围 ❤️。
- **寻求终端应用替代方案**：一位成员正在寻求 KDE 上的终端应用推荐，并表达了在使用 **Konsole** 时对屏幕漏光/显示异常问题的担忧。
   - 另一位用户报告称，在使用 **GPT-4o-mini** 运行标准 conga 终端时也遇到了类似问题。
- **Obsidian OI 插件故障**：一位用户称赞了关于 **Obsidian OI plugin** 的视频，但遇到了问题，正在寻求有关全局安装问题的建议。
   - 建议他们在指定频道中提供有关安装过程和所用界面的详细信息。
- **对 GPT-4o 记忆能力的担忧**：一位成员对 **GPT-4o** 无法记住过去的交互感到沮丧，并询问如何在 Web 开发中有效地利用它。
   - 他们考虑向 GPT-4o 询问创建记忆系统的技巧，并寻求频道内其他人的建议。


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1279126210349236278)** (2 messages): 

> - `潜在应用`
> - `House Party 讨论` 


- **对参与开发的兴奋感**：一位成员表达了热情，表示希望看到一些 **developments**，并渴望以任何可能的方式参与其中。
   - 他们还提到对 **potential applications** 有一些想法，并有兴趣进一步讨论。
- **通过 House Party 进行讨论**：另一位成员提议，**下周四的 House Party** 将是讨论潜在应用的好机会。
   - 这表明了一个在社区内分享见解和想法的轻松环境。


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1278795160880156793)** (3 messages): 

> - `GameNGen 实时模拟`
> - `AgentOps 动态`
> - `YouTube 提及` 


- **GameNGen：革新游戏模拟**：介绍 _GameNGen_，这是第一个完全由神经模型驱动的游戏引擎，能够在单个 TPU 上以超过 **每秒 20 帧** 的速度模拟 **DOOM**，PSNR 达到 **29.4**。
   - 人类评分者难以区分游戏片段和模拟画面，突显了该模型在游戏领域的有效性和潜力。
- **AgentOps 团队引发关注**：成员们对 **Adam 和 AgentOps** 团队的潜在进展感到兴奋，表明对他们即将开展的项目寄予厚望。
   - 这种热情反映了人们对 Agent 技术领域进步的广泛兴趣。
- **YouTube 提及引发关注**：一位成员分享了一个 [YouTube 视频](https://youtu.be/z4QsBsO3SS0?t=371&si=lzexLc5j0gjdjRht)，其中提到了另一位成员，在社区内引起了轰动。
   - 这种点名提升了社区参与度，并展示了同行之间的认可。



**提及链接**：<a href="https://gamengen.github.io/">GameNGen</a>：Diffusion Models Are Real-Time Game Engines

  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1278886572242239559)** (14 条消息🔥): 

> - `Google 购买 GPU`
> - `RunwayML 移除 Stable Diffusion 仓库`
> - `仓库删除引起的问题`
> - `生成写实图像`
> - `Re-LAION-5B 发布` 


- **Google 采购 GPU 引发好奇**：成员们质疑为什么 **Google** 在拥有自家 **TPU** 的情况下仍从 **NVIDIA** 购买 **GPU**，这暗示了其在硬件选择上的潜在缺口或对 NVIDIA 技术的兴趣。
   - *TPU 不够用吗？* 一位成员对 Google 在硬件上的战略选择陷入沉思。
- **RunwayML 删除所有 Stable Diffusion 仓库**：关于 **RunwayML** 删除了其在 **HuggingFace** 和 **GitHub** 上所有 **Stable Diffusion 1.5** 仓库的讨论爆发，令许多用户感到沮丧。
   - 一位成员指出，此举破坏了 **Diffusers 1.5** 中的许多功能，特别是影响了单文件加载。
- **仓库删除带来的混乱**：成员们对 RunwayML 这种看似草率的删除行为表示恼火，有人表示这感觉就像他们想要制造**混乱**。
   - 围绕潜在的法律问题出现了各种猜测，但尚未确认删除的具体原因。
- **为书封创建写实图像**：一位成员寻求关于为其小说封面生成**漫画风格**或卡通图像的建议，因为他们正苦于 **DALL·E** 输出的图像过于写实。
   - 尽管进行了尝试，他们发现 DALL·E 无法满足他们想要的特定风格。
- **Re-LAION-5B 发布**：成员们庆祝 **Re-LAION-5B** 的发布，这是 **LAION-5B** 数据集的清洗版本，解决了之前的担忧。
   - 该数据集是与关键组织合作更新的，以确保安全性和合规性，标志着一个重要的里程碑。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://laion.ai/blog/relaion-5b/">发布 Re-LAION 5B：在 LAION-5B 基础上进行透明迭代并增加安全修复 | LAION</a>: &lt;p&gt;今天，在经过&lt;a href=&quot;https://laion.ai/notes/laion-maintenance/&quot;&gt;安全修订程序&lt;/a&gt;后，我们宣布推出 Re-LAION-5B，这是 LAION 的更新版本...</li><li><a href="https://huggingface.co/runwayml">runwayml (Runway)</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LAION ▷ #[announcements](https://discord.com/channels/823813159592001537/826154622644649985/)** (1 条消息): 

mega_b: https://laion.ai/blog/relaion-5b/
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1278793158058180700)** (10 条消息🔥): 

> - `OpenAI 融资轮`
> - `聊天机器人之战`
> - `Meta AI 用户增长` 


- **科技巨头看好 OpenAI**：Nvidia、Apple 和 Microsoft 这三家全球市值最高的科技公司正在讨论投资 **OpenAI**，作为其新一轮 **1000 亿美元融资**的一部分 [来源](https://www.bloomberg.com/news/articles/2024-08-29/nvidia-has-held-discussions-about-joining-openai-s-funding-round)。
   - 此举凸显了主要参与者对 AI 融资和创新的兴趣。
- **聊天机器人之战升温**：随着 **ChatGPT** 宣称拥有超过 **2 亿周活用户**，竞争日益激烈，同时 **Meta AI** 也在市场上获得关注 [来源](https://www.theinformation.com/articles/metas-ai-assistant-wins-millions-of-users-in-challenge-to-chatgpt?utm_source=ti_app&rc=c48ukx)。
   - 然而，对于 Meta AI 是被有效使用还是存在误触参与，人们仍存有疑问。
- **Meta AI 的可用性受限**：有人担心 **Meta AI** 并非随处可用，特别是在 **EU**（欧盟），这可能会影响其增长 [来源](https://x.com/amir/status/1829248019910537470?s=46)。
   - 凭借仅 **4000 万 DAU**，其用户基数显著落后于 ChatGPT。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/amir/status/1829248019910537470?s=46">Amir Efrati (@amir) 的推文</a>: 聊天机器人之战已经开始。ChatGPT：2 亿+ 周活。Meta AI 可能紧随其后（尽管尚不清楚人们是以同样的方式使用它还是误触！）https://www.theinformation.com/articles/m...</li><li><a href="https://x.com/markgurman/status/1829233740704559182">Mark Gurman (@markgurman) 的推文</a>: Nvidia、Apple 和 Microsoft —— 这三家全球市值最高的科技公司 —— 正在洽谈投资 OpenAI，作为该公司新一轮 1000 亿美元融资的一部分。https://www.bloomberg.com/news/articles...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1278967107492778015)** (3 messages): 

> - `Tinygrad Cloud Service`
> - `Impact of System Prompts` 


- **Tinygrad 发布高性价比云解决方案**：Tinygrad 宣布了一项新的云服务，以每月仅 **$60** 的价格提供 **4090 GPU** 和 **500 GB 存储**，比 Vast AI 等竞争对手便宜 **3 倍**。
   - *即将推出：CLOUD=1* 允许用户在本地运行 Tinygrad，同时利用云端速度通过 **10 步处理** 实现性能增强。
- **关于 System Prompts 影响的咨询**：一位成员询问是否有论文研究 **System Prompts** 对评估分数的影响。
   - 他们质疑是否可能通过不同的 Prompting 技术来**显著改变分数**。



**提及的链接**：<a href="https://x.com/__tinygrad__/status/1829379908017238210?s=46">来自 tiny corp (@__tinygrad__) 的推文</a>：即将推出：CLOUD=1。只需 $60/月（比 vast ai 便宜 3 倍），我们将为你租用一台 4090 和 500 GB 的云存储。在你的开发机上照常使用 tinygrad，但它在云端运行速度飞快……

  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1278800875212636261)** (11 messages🔥): 

> - `QLoRA memory issues`
> - `Multi-GPU evaluation in TorchTune`
> - `CUDA errors during training`
> - `Memory requirements for A6000 GPUs`
> - `Training sequence lengths` 


- **提出的 QLoRA 显存问题**：一位成员表示怀疑他们的设置应该有足够的显存来运行 **QLoRA**，并质疑是否出了问题。
   - 他们提到在运行带有 **4 张 48GB GPU 卡** 的配置时，出现了指示非法内存访问（illegal memory access）的 **CUDA 错误**。
- **关于 GPU 显存需求的澄清**：一位成员指出 **A6000 GPU** 现在是 **48GB** 而非 **24GB**，表明四张此类显卡应该足以完成任务。
   - 他们还指出，在没有 CPU offloading 的情况下可能会造成资源紧张，并建议序列长度（sequence length）可能是一个因素。
- **对序列长度的关注**：另一位成员尝试了不同的训练序列长度（**8K** 和 **4K**），暗示显存问题可能取决于所使用的长度。
   - 他们提到了训练设置中的一些细节，这些细节可能会在过程中影响 **vRAM**。
- **TorchTune 中的多 GPU 评估**：一位成员询问 **TorchTune** 是否支持**多 GPU 评估**，表明了对优化性能的潜在兴趣。
   - 他们的问题突出了在使用多 GPU 的训练设置中对可扩展性的普遍需求。
- **理解非法内存访问错误**：在发生操作错误后，一位成员收到建议，设置 **CUDA_LAUNCH_BLOCKING=1** 以调试训练期间的非法内存访问问题。
   - 这指向了在有效管理显存的同时，使用 **PyTorch** 进行分布式训练（distributed training）的复杂性。


  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1278791464586969140)** (5 messages): 

> - `LinkedIn Auto Jobs Applier`
> - `DSPy Community Engagement` 


- **对仓库关联的困惑**：一位成员对某项陈述与 [链接的 GitHub 仓库](https://github.com/feder-cr/linkedIn_auto_jobs_applier_with_AI) 之间的联系表示困惑。另一位成员澄清说该仓库是完全独立的，但希望将其展示给 DSPy 社区以激发参与。
   - *它每天获得超过 2k 个点赞*，表明该工具引起了极大关注。
- **对 GitHub Issues 的担忧**：一位成员对 LinkedIn Auto Jobs Applier 的性能表示担忧，询问其是否经过测试，并指出 GitHub issues 显示仍有改进空间。讨论暗示该仓库的反馈表明仍有很多不尽如人意之处。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1278853571194130434)** (5 条消息): 

> - `Workshop on Useful and Reliable AI Agents`
> - `DSPy: Prompt Optimization for LM Programs`
> - `AgentOps`
> - `Nelima`
> - `Bay Area AI Meetup` 


- **Workshop on Useful and Reliable AI Agents**：一位成员分享了标题为 [Workshop on Useful and Reliable AI Agents](https://www.youtube.com/live/-aKRsvgDEz0) 的 **YouTube 视频**链接，讨论了 AI Agent 在准确性、可靠性和成本效益方面的重要性。
   - 该研讨会旨在探讨围绕 AI Agent 的活跃研究，以及如何在现实场景中有效地利用它们。
- **用于构建 AI Agent 的 AgentOps 工具**：分享了关于 [AgentOps](https://agents.staf.ai/AgentOps) 的信息，它为构建 Agent 提供如图表（graphs）和监控等功能。
   - 他们的目标是消除 Agent Prompt 中的猜测，强调以透明的方法开发 AI 解决方案。
- **与 Michael Ryan 合作的 DSPy 研讨会**：由 @ChiefScientist 主持的即将举行的 Bay Area AI Meetup 将邀请 Michael Ryan 讨论 “DSPy: Prompt Optimization for LM Programs” 以及 LM Programs 的概念。
   - 斯坦福大学学生 Michael 将在由 @Neo4j 赞助的活动中展示他最新的优化工作，包括 MIPROv2 算法。
- **对活动录像的关注**：一位成员对上述活动表示兴奋，并获知该活动将被录制以供发布。
   - 这反映了社区渴望获取 Meetup 中分享的有价值见解。
- **DSPy 使用问题**：一位用户询问了发布关于 DSPy 使用疑问的合适频道。
   - 这表明社区内正在积极寻求关于 DSPy 库的支持和指导。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/ChiefScientist/status/1829231009344434400?t=wow3U2BluHEv16-MI2YcaQ&s=19">来自 Alexy 🤍💙🤍 (@ChiefScientist) 的推文</a>: 非常激动能在 @github 旧金山 SOMA 总部举办的 @AIconference 后的 http://Bay.Area.AI meetup 中接待 Michael Ryan！DSPy: Prompt Optimization for LM Programs Michael Ryan, @Stanford...</li><li><a href="https://www.youtube.com/live/-aKRsvgDEz0">Workshop on Useful and Reliable AI Agents</a>: AI Agent 已成为一个活跃的研究领域。但要在现实世界和大规模应用中发挥作用，Agent 需要准确、可靠且廉价。了解如何...</li><li><a href="https://docs.google.com/spreadsheets/d/1VnOv_C0v_FgDeKuQBaGuMNsWgoWOpLkGbE_XS_2Vb3Q/edit?gid=0#gid=0">由 AgentOps.ai 提供的 Agent 数据库</a>: 未找到描述。
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1279021509838639264)** (5 条消息): 

> - `Axolotl GitHub Documentation`
> - `Training LLaMA 70B`
> - `NVIDIA A6000 GPUs` 


- **Axolotl GitHub 文档深色模式请求**：一位成员表示希望 [Axolotl GitHub 文档](https://github.com/axolotl) 能够提供**深色模式**，理由是目前的浅色模式令人不适。
   - 他们提到经常访问文档以检查配置参数，强调目前的浅色模式存在问题。
- **LLaMA 70B 训练的硬件考量**：关于全量训练 **LLaMA 70B** 的硬件要求展开了讨论，一位成员询问了目前的推荐配置。
   - 他们推测，鉴于近期训练效率的提升，仅需几块 **NVIDIA A6000** GPU 可能就足够了。
- **3x A6000 GPU 应足以应对 70B**：一位成员对 GPU 问题给出了肯定回答，建议 **3x A6000 GPU** 应该足以训练全量 **70B 模型**。
   - 这让一些人对该硬件的能力感到惊讶，表明 GPU 性能可能有所突破。


  

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1278894713742561300)** (1 messages): 

> - `Axolotl`
> - `Hugging Face transformers` 


- **Axolotl 在更新后无需更改**：一位成员强调，**Axolotl** 的结果现在甚至更好了，在最近的更新之后不需要进行任何更改。
   - 这是针对 **Hugging Face** 由 [Rocketknight1 提交的 Pull Request #33198](https://github.com/huggingface/transformers/pull/33198) 而发布的，该 PR 改进了聊天模板 (chat templates)。
- **新增 assistant prefill 功能**：最近的 Pull Request 解决了长期以来对 **assistant prefill** 功能的需求，允许模型自动开始其响应。
   - 这一增强旨在通过一种稍微“hacky”的方法来启动响应，从而在 **TextGenerationPipeline** 中提供更流畅的体验。



**提到的链接**：<a href="https://github.com/huggingface/transformers/pull/33198">Add assistant prefill for chat templates and TextGenerationPipeline by Rocketknight1 · Pull Request #33198 · huggingface/transformers</a>：内部和 GitHub 上多次请求的功能是 assistant prefill：即为模型预先设定响应的开头并让其继续。我们使用了一种稍微...

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1278976543678533692)** (3 messages): 

> - `Llama 3.1`
> - `Uninitialized Special Tokens`
> - `Fixing Untrained Tokens` 


- **Llama 3.1 仍然存在特殊 Token 问题？**：一位成员询问 **Llama 3.1 base** 是否仍然受到未初始化特殊 Token 的困扰，特别是关于嵌入 (embeddings) 处于分布外 (out of distribution) 的问题。
   - 这一担忧表明在处理模型中的特殊 Token 方面仍存在挑战。
- **引入了针对未训练 Token 的新修复方案**：另一位成员透露，已添加了一个选项 `fix_untrained_tokens: true`，以潜在地解决未初始化特殊 Token 的问题。
   - 这一改进表明了在优化模型性能方面采取了积极的方法。


  

---



### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1278839619231682632)** (6 messages): 

> - `Groq Leaderboard Update`
> - `Documenting Model Steps`
> - `Java GIS Geometry Initialization`
> - `Temperature Settings in Evaluations`
> - `OSSHandler Parameter Adjustments` 


- **Groq 等待 PR 以进入排行榜**：据指出，**Groq** 尚未被添加到排行榜中，因为团队仍在等待他们的 PR，预计下周左右完成。
   - 这引发了一些关于其集成和预期性能的持续讨论。
- **确认步骤文档化**：一位成员确认，确保模型步骤 (model steps) 被正确记录对于可复现性 (reproducibility) 至关重要。
   - 该声明强调了适当的文档记录能增强模型的可理解性和可用性。
- **Java 测试用例揭示性能问题**：一位用户分享了一个 **Java** 测试用例，其模型表现不佳，特别是在 GIS 几何表示的初始化方面。
   - 得出的结论是，鉴于用户的查询，提供一个直接的示例可能比复杂的函数调用更有益。
- **关于评估 Temperature 设置的疑问**：有人提问模型评估是否严格使用贪婪解码 (greedy decode) 和 0 的 Temperature 以确保指标公平。
   - 成员们参考了最近关于排行榜评估标准的 GitHub 链接，讨论了输出随机性的影响。
- **OSSHandler 默认参数讨论**：注意到 **OSSHandler** 的默认 Temperature 设置为 0.001，虽然考虑过调整，但最终决定不进行更改。
   - 这一决定旨在保持一致的函数输出并优化模型性能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#model-specific-optimization">gorilla/berkeley-function-call-leaderboard at main · ShishirPatil/gorilla</a>：Gorilla：训练和评估用于函数调用（工具调用）的 LLM - ShishirPatil/gorilla</li><li><a href="https://github.com/ShishirPatil/gorilla/discussions/562">Set Model Temperature to 0 for Consistent Leaderboard Results · ShishirPatil/gorilla · Discussion #562</a>：当前的模型生成脚本 (model_handlers) 在推理时使用默认的 0.7 Temperature。这给模型输出生成引入了一定程度的随机性，导致潜在的...
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1278838600510996593)** (2 条消息): 

> - `tinygrad capabilities`
> - `sparsity techniques` 


- **质疑 tinygrad 的优势**：*codeman3786* 询问 **tinygrad** 是否主要对 **statically scheduled operations**（静态调度操作）有效，而不适用于涉及 **semi-structured sparsity**（半结构化稀疏）或权重选择的方法。
   - 这促使 *georgehotz* 询问是否有 *codeman3786* 无法使用 tinygrad 实现的具体示例。
- **tinygrad 局限性的实例**：Georgehotz 的回应表明他愿意讨论潜在的局限性，并要求提供 tinygrad 可能表现不足的例子。
   - 这种互动表明社区有兴趣探索 tinygrad 性能和通用性的实际限制。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1279210301232906393)** (2 条消息): 

> - `Tensor.cat with sharded tensors`
> - `Padding and reshaping issues`
> - `Batch dimension manipulation` 


- **Tensor.cat 在处理 sharded tensors 时遇到困难**：一位用户在尝试沿 batch 轴对两个 **sharded tensors** 进行 **Tensor.cat** 时遇到错误，具体提示为 *padding not supported for arg=((0, 9), (0, 0), (0, 0))*。
   - 他们提供了一个使用 `unsqueeze` 的变通方法，但遇到了另一个与重塑维度（reshaping dimensions）相关的错误。
- **用户询问操作的基础支持**：用户正在质疑无法连接 **sharded tensors** 是一个根本性问题还是仅仅是尚未支持的功能，并寻求对该问题的澄清。
   - 他们正在探索各种选项，包括修改代码以支持额外的 batch 维度，或执行多个操作以避免使用 **Tensor.cat**。


  

---



---



---



---



---



---



---



{% else %}


> 完整的频道详情已为邮件格式进行截断。
> 
> 如果您想查看完整的详情，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}