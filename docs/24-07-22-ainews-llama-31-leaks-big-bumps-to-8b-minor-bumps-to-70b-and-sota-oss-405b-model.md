---
companies:
- meta-ai-fair
- openai
- alibaba
date: '2024-07-23T01:12:50.598107Z'
description: '**Llama 3.1** 的泄露信息揭示了一个拥有 **405B（4050亿）参数的稠密模型**，支持 **128k 上下文长度**。该模型在
  H100-80GB GPU 上训练了 **3930 万 GPU 小时**，并使用了**超过 2500 万个合成示例**进行微调。该模型在基准测试中表现出显著提升，尤其是
  8B 和 70B 变体，部分评估甚至表明 70B 版本的表现优于 **GPT-4o**。同时，**GPT-4o Mini** 作为一款高性价比版本发布，虽然性能强劲，但在推理能力上存在一些短板。此外，像
  **NuminaMath** 这样的合成数据集助力阿里巴巴的 **Qwen 2** 等模型在数学竞赛中超越了 GPT-4o 和 Claude 3.5。相关讨论还涉及了推理任务基准测试以及为提升推理能力而进行的数据集构建。'
id: e46d711c-9bcd-4063-a752-d4e2a7ee14a8
models:
- llama-3-1-405b
- llama-3-8b
- llama-3-70b
- llama-3-1-8b
- gpt-4o
- gpt-4o-mini
- claude-3-5
- qwen-2
original_slug: ainews-llama-31-leaks
people:
- swyx
- philschmid
- jjitsev
- lewtun
- teknium1
- adcock_brett
title: Llama 3.1 爆料：8B 版本大幅提升，70B 版本小幅改进，以及 SOTA 级别的开源 405B 模型。
topics:
- multilinguality
- code-generation
- context-windows
- model-training
- synthetic-data
- benchmarking
- reasoning
- fine-tuning
- model-performance
- dataset-release
---

<!-- buttondown-editor-mode: plaintext -->**TODO: 单行副标题**

> 2024年7月19日至7月22日的 AI 新闻。我们为您检查了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 社区（包含 **474** 个频道和 **7039** 条消息）。为您节省了预计阅读时间（以 200wpm 计算）：**765 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

我们知道它明天就要发布了（伴随 [Soumith 的 ICML 主旨演讲](https://x.com/soumithchintala/status/1814704963332767833)），所以我们本想尽量避免讨论泄露内容，因为明天我们会进行全面报道。但 Llama 3.1 的泄露简直像筛子一样（[权重](https://x.com/rohanpaul_ai/status/1815371623815356751)、[评估结果](https://x.com/rohanpaul_ai/status/1815459760227168764)、[模型卡 (model card)](https://pastebin.com/clone/9jGkYbXY)），到处都在漏。不幸的是，尽管其中很多内容只是 [4 月份第一次发布 Llama 3](https://buttondown.email/ainews/archive/ainews-llama-3/) 时的重复，但今天整个社区都在讨论它。

除了早已预告的 405B 稠密模型发布外，据我们所知，Llama 3.1 的主要差异如下，大部分来自[模型卡](https://pastebin.com/clone/9jGkYbXY)中列出的各项优先级：

- “Llama 3.1 指令微调的纯文本模型（8B, 70B, 405B）针对**多语言**对话场景进行了优化，在常见的行业基准测试中表现优于许多现有的开源和闭源聊天模型。”
- 明确宣传“多语言文本**和代码**”作为输出模态。
- 每个模型的**上下文长度 (context length)** 都提升到了 **128k**（之前为 8k）。
- 训练在 H100-80GB（TDP 为 700W）上累计使用了 **3930 万 GPU 小时**的计算量：8B 模型 150 万，70B 模型 700 万，405B 模型 3100 万。
- Llama 3.1 在来自公开来源的约 15 万亿 token 数据上进行了预训练。微调数据包括公开可用的指令数据集，以及**超过 2500 万个合成生成的示例**。
- 8B 和 70B 的基准测试分数有显著提升（8B 的 MMLU 从 65 提升到 73（+8 分），70B 从 81 提升到 86（+5 分）；8B 的 MATH 从 29 提升到 52（+23 分））。

我们[制作了一个差异对比表格](https://x.com/swyx/status/1815553411808653513)来进行可视化 —— TLDR：8B 模型在各方面都有巨大提升，指令微调版的 70B 略有进步。405B 仍落后于旗舰模型。

 
![image.png](https://assets.buttondown.email/images/1cd032fd-5219-402d-80a5-bcd95eac43dd.png?w=960&fit=max)
 

然而，一些[独立运行的评估显示 Llama 3.1 70b 的表现优于 GPT 4o](https://x.com/mattshumer_/status/1815444612414087294) —— 结论尚无定论。


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，从 4 次运行中选取最佳结果。

**GPT-4o Mini 的发布与性能**

- **GPT-4o Mini 发布**：[@adcock_brett](https://twitter.com/adcock_brett/status/1815054909378580728) 宣布发布 GPT-4o mini，这是 GPT-4o 模型的一个紧凑且具有成本效益的版本，**价格为每百万输入 Token 15 美分，每百万输出 Token 60 美分，比 GPT-3.5 Turbo 便宜 60% 以上**。
- **强劲性能**：[@swyx](https://twitter.com/swyx/status/1815037679014388172) 强调了 GPT-4o mini 令人印象深刻的性能，在 **$0.15/mtok 的价格下达到了 82 MMLU**，超越了仅在 3 个月前还是 SOTA（州级/顶尖）水平的模型。
- **推理缺陷**：[@JJitsev](https://twitter.com/JJitsev/status/1815011239912755392) 在 AIW 问题上测试了 GPT-4o mini，发现其在简单问题的变体上存在**基础推理缺陷且缺乏鲁棒性**，尽管计算规模相似，但表现逊于 Llama-3-8B。

**合成数据与模型性能**

- **超越导师**：[@_philschmid](https://twitter.com/_philschmid/status/1814982420602421414) 分享了 AI-MO 团队的获胜数据集，配合微调后的 @Alibaba_Qwen 2 模型，在数学竞赛中**接近或超越了 GPT-4o 和 Claude 3.5**，展示了合成数据集使模型超越其导师模型的潜力。
- **NuminaMath 数据集**：[@_lewtun](https://twitter.com/_lewtun/status/1814958635732140336) 介绍了 **NuminaMath 数据集，这是最大的数学竞赛问题-解答对集合（约 100 万条）**，该数据集被用于赢得 AI 数学奥林匹克竞赛（AI Math Olympiad）的首届进步奖。在 NuminaMath 上训练的模型在开源权重模型（open weight models）中实现了**同类最佳性能**。

**推理与鲁棒性基准测试**

- **全面推理任务清单**：[@Teknium1](https://twitter.com/Teknium1/status/1815105755613376792) 建议创建一个推理任务的主列表供大家贡献，以帮助数据集构建者针对性地开发能够提高推理能力的任务。
- **强劲性能的幻象**：[@JJitsev](https://twitter.com/JJitsev/status/1815011276684173312) 认为当前的 Benchmark 忽视了 SOTA LLM 的明显缺陷，为那些虽然得分高但无法稳健进行基础推理的模型营造了性能强劲的幻象。

**AI 社区中的 Meme 与幽默**

- **Meme 潜力**：[@kylebrussell](https://twitter.com/kylebrussell/status/1815096890595369165) 分享了一个 Meme，暗示其在 AI 社区中的潜在影响力。
- **AI 生成的幽默**：[@bindureddy](https://twitter.com/bindureddy/status/1815162164115808691) 分享了一张 AI 生成的图片，强调了 AI 在创作幽默内容以及从严肃话题中提供放松方面的作用。

---

# AI Reddit 摘要

## /r/LocalLlama 回顾

**主题 1. AI 驱动的数学训练**

- **NuminaMath 数据集：约 100 万个数学竞赛问题-解答对的最大集合** ([Score: 53, Comments: 1](https://reddit.com//r/LocalLLaMA/comments/1e8kme3/numinamath_datasets_the_largest_collection_of_1m/)): **NuminaMath 发布海量数学数据集**：**NuminaMath** 集合包含约 **100 万个数学竞赛问题-解答对**，已在 **Hugging Face Hub** 上发布。该数据集随附模型和技术报告，是此类数据中规模最大的集合，有望提升 AI 在数学解题方面的能力。

**主题 2. 本地 LLM 资源优化**

- **[large-model-proxy 允许在同一台机器的不同端口上运行多个 LLM，同时通过在需要时停止/启动它们来自动管理 VRAM 使用。](https://github.com/perk11/large-model-proxy)** ([Score: 68, Comments: 10](https://reddit.com//r/LocalLLaMA/comments/1e8hges/largemodelproxy_allows_to_run_multiple_llms_on/)): **large-model-proxy** 是一个工具，允许在同一台机器的不同端口上运行 **多个大型语言模型 (LLMs)**，同时 **自动管理 VRAM 使用**。该代理根据需要动态停止和启动模型，使用户无需人工干预即可高效利用其 **GPU** 资源。该解决方案解决了在 VRAM 有限的单台机器上运行多个内存密集型 **LLM** 的挑战。
  - 作者开发 **large-model-proxy** 是为了在工作流中高效管理多个 **LLM**。它实现了 **VRAM 管理**和模型启动/停止的自动化，使得编写脚本和利用各种模型变得更加容易，无需人工干预。
  - 一位用户指出 **Ollama** 提供了类似的功能，允许根据 **VRAM** 可用性自动卸载/加载，并发运行多个模型，而无需多个端口或编辑配置文件。
  - 另一位开发者提到使用 **Python** 和 **OpenResty Lua 脚本**来代理 **OpenAI API** 请求并按需管理 **LLaMa.cpp** 实例，并对 **large-model-proxy** 的 **VRAM** 管理方面表示感兴趣。

## All AI Reddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**主题 1. LLaMA 3 405B 模型发布及其影响**


- [/r/singularity] **[看起来我们本周就能迎来 LLama3 405B](https://i.redd.it/4kvx4omgtxdd1.jpeg)** ([Score: 443, Comments: 113](https://reddit.com//r/singularity/comments/1e8wptx/looks_like_were_getting_llama3_405b_this_week/)): **LLaMA 3 405B 模型发布在即**：根据内部消息，Meta 的 AI 研究团队预计将于本周发布 **LLaMA 3 405B** 模型。这款新模型被预期为对其前代产品的重大升级，潜力足以媲美或超越 **GPT-4** 的能力。


**主题 2. 医疗保健中的 AI：提高癌症检测率**


- [/r/singularity] **[全科医生使用 AI 将英格兰的癌症检测率提高了 8%](https://www.theguardian.com/society/article/2024/jul/21/gps-use-ai-to-boost-cancer-detection-rates-in-england-by-8)** ([Score: 203, Comments: 25](https://reddit.com//r/singularity/comments/1e8htmy/gps_use_ai_to_boost_cancer_detection_rates_in/)): 英格兰的 **AI 辅助癌症检测** 已导致疑似癌症的转诊量**增加了 8%**。使用 **C the Signs** 工具的**全科医生**在**两年期间**多转诊了 **92,000 名患者**进行紧急癌症检查。该 AI 系统可帮助医生识别潜在的癌症症状并确定适当的后续步骤，展示了 AI 在增强初级保健机构早期癌症检测方面的潜力。

**主题 3. LocalLLaMA 的进展与应用**



- [/r/StableDiffusion] **[在同一天，有两个人偷走了我的设计和应用，并当作他们自己的作品发布。](https://i.redd.it/xrwxvzxjqtdd1.png)** ([Score: 259, Comments: 195](https://reddit.com//r/StableDiffusion/comments/1e8gter/on_the_same_day_two_people_stole_my_design_and/)): **抄袭行为冲击 AI 开发者**：一款使用 **LocalLLaMA** 的**虚拟试衣 Chrome 扩展程序**的创作者报告称，有两人复制了他们的设计和应用程序，并在同一天将其作为原创作品展示。这一事件凸显了在快速发展的 AI 开发和应用领域中，**知识产权保护**所面临的持续挑战。
  - **两名个人**涉嫌复制了原帖作者（OP）的**虚拟试衣 Chrome 扩展程序**，引发了关于 AI 开发中**知识产权**的辩论。许多用户指出类似的产品已经存在，对 OP 的原创性声明表示质疑。
  - 用户强调，该项目使用了 **fal.ai API**，只需基本的输入和一个按钮，在 **15 分钟**内即可重建。这种复制的简易性引发了人们对简单 AI 实现的价值以及是否需要更强大的准入门槛的质疑。
  - 讨论集中在**开源项目**和正确**注明创意来源**的重要性上。一些人认为创意不受版权保护，而另一些人则强调即使是简单的实现也需要承认灵感来源。


---

# AI Discord Recap

> 摘要之摘要的摘要


**1. LLM 模型发布与基准测试**

- **DeepSeek-V2 登顶基准测试**：**DeepSeek-V2** 是一个拥有 236B 参数的 MoE 模型（每个 token 激活 21B），因其出色的性能和每 100 万输入 token 仅 0.14 美元的成本效益而受到赞誉，在 **AlignBench** 和 **MT-Bench** 等某些领域表现优于 GPT-4。
   - 该模型针对 **DeepSeek-V2-Chat-0628** 的 1-bit 量化结果显示了优化的 CPU 性能，在 LMSYS Arena Hard 全球排名第 7。用户注意到其在多语言任务中的强劲表现。
- **Llama 3.1 泄露引发热议**：泄露的 **Llama 3.1** 评估结果表明，其 8B、70B 和 405B 模型在进行指令微调（instruct tuning）之前，性能就可能超过目前的顶级模型，其中 70B 模型被认为非常接近领先模型。
   - 泄露信息显示，405B 模型被蒸馏成了具有 128k 上下文的 8B 和 70B 版本。社区成员对潜在的能力感到兴奋，尤其是应用指令微调之后。
  
**2. AI 基础设施与优化**

- **Elon Musk 的 Memphis Supercluster 发布**：Elon Musk 宣布启动 **Memphis Supercluster**，声称它是全球最强大的 AI 训练集群，在单个 RDMA fabric 上拥有 10 万块液冷 H100。
   - 然而，[事实核查](https://x.com/dylan522p/status/1815494840152662170?s=46)揭示了在电力消耗和 GPU 可用性方面的差异，表明该设施尚未像声称的那样完全投入运行。
- **模型量化技术的进展**：讨论强调了模型量化技术的进步，**AQLM** 和 **QuaRot** 旨在在保持性能的同时，在单个 GPU 上运行大语言模型 (**LLMs**)。
   - 分享的一个例子是 [AQLM 项目](https://github.com/Vahe1994/AQLM) 成功在 RTX3090 上运行 **Llama-3-70b**，展示了在使大型模型在消费级硬件上更易于获取方面取得的重大进展。
  
**3. AI 模型性能与效率**

- **隐式 CoT 提升 GPT-2 性能**：**[Implicit Chain-of-Thought (CoT)](https://arxiv.org/pdf/2405.14838)** 通过移除中间阶段并进行微调来使步骤内部化，使 **GPT-2 Small** 能够以 99% 的准确率解决 9x9 乘法。
   - 该方法还增强了 **Mistral 7B**，在没有中间步骤的情况下在 GSM8K 上实现了超过 50% 的准确率。
- **ReFT 以参数效率令人震惊**：**ReFT** 的参数效率比 LoRA 高出 15 到 60 倍，并且可以在 A10 上使用约 100 个示例在不到一分钟的时间内微调 **Llama 2 7B** 等模型。
   - Greg Schoeninger [讨论](https://www.reddit.com/r/MachineLearning/comments/1e8qwnl/r_discussion_of_reft_paper_with_lead_author/) 了其具体应用和挑战，并在 [YouTube 视频](https://www.youtube.com/watch?v=to2oKwnknUk) 中进行了深入探讨。
- **DeepSeek 1-bit 量化结果令人印象深刻**：**DeepSeek-V2-Chat-0628** 的 1-bit 量化显示出令人印象深刻的 CPU 优化，在 LMSYS Arena Hard 全球排名第 7 ([链接](https://huggingface.co/nisten/deepseek-0628-gguf))。
   - *kotykd* 询问了该模型的连贯性以及与之前版本相比的性能变化。
    

**4. 知识图谱与检索增强生成 (RAG)**

- **Triplex 将 KG 成本降低 98%**：来自 [SciPhi.AI](https://www.sciphi.ai) 的 **Triplex** 将知识图谱提取成本降低了 98%，通过使用 SciPhi 的 R2R 进行本地图谱构建，以 1/60 的价格超越了 GPT-4。
   - [R2R](https://r2r-docs.sciphi.ai/cookbooks/knowledge-graph) 支持多模态数据和混合搜索，优化了知识图谱，而 [微软的方法](https://neo4j.com/developer-blog/global-graphrag-neo4j-langchain/) 则使用更深的邻接矩阵来实现更高效的 RAG。
- **将 RAG 应用部署到生产环境**：一位成员分享了关于使用 MongoDB Atlas 和 LangChain 构建 RAG 实现的 [教程](https://www.mongodb.com/docs/atlas/atlas-vector-search/ai-integrations/langchain/)。
   - 该教程涵盖了环境搭建、数据存储、创建搜索索引以及运行向量搜索查询。
- **通过 Deasie 工作坊改进 RAG**：与 Deasie 联合创始人进行的 [YouTube 会话](https://t.co/cJPsNaWgoc) 涵盖了用于改进 RAG 的高级解析和元数据。
   - 解析和元数据增强被强调为提升 RAG 性能的关键技术。
    

**5. 社区贡献与开源项目**

- **Nemotron-340B 悬赏引发关注**：[Nathan](https://x.com/natolambert/status/1814735390877884823) 悬赏 75 美元起，征集将 **Nemotron-340B** 转换为 **HuggingFace** 格式，并实现 **FP8 量化** 和多节点部署。
   - 悬赏金额已飙升至 **2,000 美元以上**，引起了合成数据社区的极大兴趣。
- **GPTScript 的 OpenRouter 提供商现已可用**：宣布了一个新的 [GPTScript OpenRouter 提供商](https://github.com/RobinVivant/gptscript-openrouter-provider)，并在 GitHub 上提供了图片和详细说明。
   - 该工具为 GPTScript 应用程序的开发做出了重大贡献。
- **Bud-E 展示具有开源目标的新 Demo**：分享了 **Bud-E 语音助手** 的演示，展示了未来每个人都能以电力成本获得高性能开源系统的愿景。
   - 目前针对 Ubuntu 优化的代码库将进行重构，以实现客户端、服务器以及可互换的 ASR、TTS、LLM 组件之间的清晰分离。


---

# 第 1 部分：Discord 高层级摘要

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **隐式思维链 (Implicit CoT) 提升 GPT-2 性能**：[隐式思维链 (Implicit CoT)](https://arxiv.org/pdf/2405.14838) 通过移除中间阶段并进行微调来使步骤内化，使 **GPT-2 Small** 能够以 99% 的准确率解决 9x9 乘法问题。
   - 该方法还增强了 **Mistral 7B**，在没有中间步骤的情况下，在 GSM8K 上实现了超过 50% 的准确率。
- **ReFT 的参数效率令人震惊**：**ReFT** 的参数效率比 LoRA 高出 15x-60x，在 A10 上使用约 100 个示例，可在不到一分钟内完成对 **Llama 2 7B** 等模型的微调。
   - Greg Schoeninger [讨论了](https://www.reddit.com/r/MachineLearning/comments/1e8qwnl/r_discussion_of_reft_paper_with_lead_author/) 它的实际应用和挑战，并在 [YouTube 视频](https://www.youtube.com/watch?v=to2oKwnknUk)中进行了深入探讨。
- **DeepSeek 的 1-bit 量化结果令人印象深刻**：**DeepSeek-V2-Chat-0628** 的 1-bit 量化显示出令人印象深刻的 CPU 优化，在 LMSYS Arena Hard 全球排名第 7 ([链接](https://huggingface.co/nisten/deepseek-0628-gguf))。
   - *kotykd* 询问了该模型与之前版本相比在连贯性和性能上的变化。
- **图谱提升 RAG 性能**：来自 [SciPhi.AI](https://www.sciphi.ai) 的 **Triplex** 将知识图谱提取成本降低了 98%，通过使用 SciPhi 的 R2R 进行本地图谱构建，以 1/60 的价格超越了 GPT-4。
   - [R2R](https://r2r-docs.sciphi.ai/cookbooks/knowledge-graph) 支持多模态数据和混合搜索，优化了知识图谱，而 [微软的方法](https://neo4j.com/developer-blog/global-graphrag-neo4j-langchain/) 则使用更深的邻接矩阵来实现更高效的 RAG。
- **QuietStar 激发了自动生成提示词的讨论**：*QuietStar* 引发了关于 LLM 并行生成后续提示词的讨论，旨在动态增强其推理能力。
   - 参与者辩论了如何通过中间表示和类型系统调整 LLM 架构，以实现更好的 Token 级推理。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hermes 2.5 表现优于 Hermes 2**：在添加了[代码指令示例](https://link.to.examples)后，**Hermes 2.5** 在各种基准测试中的表现似乎优于 **Hermes 2**。
   - Hermes 2 在 MMLU 基准测试中得分为 **34.5**，而 Hermes 2.5 得分为 **52.3**。
- **Mistral 难以扩展到 8k 以上**：成员表示，如果不进行持续预训练，**Mistral** 无法扩展到 8k 以上，[这是一个已知问题](https://link.to.issue)。
   - 他们指出，针对性能的下一个前沿领域是 *mergekit* 和 *frankenMoE finetuning* 的进一步工作。
- **关于模型合并策略的讨论**：一位成员建议将 **UltraChat** 和基础版 **Mistral** 之间的差异应用于 **Mistral-Yarn**，作为一种潜在的合并策略。
   - 其他人表示怀疑，但该成员保持乐观，并引用了过去在他们所谓的“诅咒模型合并 (cursed model merging)”方面的成功尝试。
- **Open Empathic 项目寻求协助**：一位成员呼吁帮助扩大 **Open Empathic** 项目的类别，特别是在低端部分。
   - 他们分享了一个关于 [Open Empathic 发布与教程的 YouTube 视频](https://youtu.be/GZqYr8_Q7DE)，指导用户贡献他们喜欢的 YouTube 视频电影场景，以及 [OpenEmpathic 项目本身](https://dct.openempathic.ai/)的链接。
- **SmolLM Arena 发布**：一个名为 SmolLM Arena 的[新项目](https://huggingface.co/spaces/as-cle-bert/smolLM-arena)已经发布，允许用户比较各种小型语言模型 (Small Language Models, <1.7B 参数)。
   - 该竞技场具有聊天机器人界面，运行速度更快，并包含使用说明以提供更流畅的用户体验。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 2.0: Socket 实现**：讨论集中在 Mojo 的[各种 Socket 实现](https://tigerbeetle.com/blog/a-friendly-abstraction-over-iouring-and-kqueue)，重点关注 Windows Socket 等特定平台的挑战。
   - 成员们还强调了使用 Rust 的 Socket 实现的潜力，并讨论了为未来协议适配 SCTP 的重要性。
- **关于双栈 Socket 的辩论**：新的服务器 Socket 倾向于使用**双栈 Socket (Dual stack sockets)**，允许同时进行 IPv4 和 IPv6 连接，这与 Python 的实现类似。
   - 达成共识在 Linux 上使用 `io_uring` 来处理高性能工作负载。
- **Flat Buffers 在社区会议中表现出色**：[Mojo 🔥 社区会议 #4](https://www.youtube.com/watch?v=_QVs626Vn2k) 涵盖了用于内存高效序列化的 **Flat Buffers** 以及 **Forge Tools** 的更新。
   - 讨论重点在于优化数据处理和扩展 **Mojo 标准库**。
- **浮点字面量的牛顿迭代法**：一位成员分享了 Mojo 中浮点字面量的牛顿迭代法 (Newton's Method) 实现，引发了关于捕获数值方程关键字的详细讨论。
   - 这引发了关于闭包以及在 Mojo 中解决复杂数值问题的对话。
- **Mojo GPU：展望夏季**：来自 Google 的前 XLA 团队已加入 Mojo，为 AI 基础设施开发带来了新见解。
   - Mojo GPU 支持预计将于今年夏天推出，增强计算能力。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **推荐使用 ComfyUI 而非 Forge**：多位用户建议从 **Forge** 切换到 **ComfyUI** 以获得更好的体验，理由是 Forge 的功能和兼容性存在问题。
   - 用户称赞 **ComfyUI** 拥有明显更多的工具和功能，尽管指出其基于节点的界面更为复杂。
- **Forge 与 Easy Diffusion 的对比**：一位用户指出，虽然 **Forge** 比 **Easy Diffusion** 快，但它缺少一些功能，并且在放大 (upscaling) 时会报错。
   - 其他人评论了 **Forge** 中放大问题和分辨率处理不当的问题，并提出了替代方案。
- **在 Regional Prompter 中使用 Latent 模式**：提供了关于使用 **Latent 模式**而非 **Attention 模式**进行 **Regional Prompter** 的指导，以防止角色融合。
   - 分享了详细的 Prompt 和说明，以改进在多角色 LoRA 中使用 **Latent 模式**的效果。
- **VRAM 和 GPU 兼容性问题**：讨论涵盖了 Stable Diffusion 的 VRAM 需求以及 GPU（尤其是 **AMD** 显卡）的 VRAM 问题。
   - 解决方案包括为家庭 GPU 能力有限的用户提供**本地安装**和云端 GPU。
- **Forge 中的放大错误**：用户在使用 **Forge** 放大图像时遇到了 'NoneType' 错误。
   - 建议包括切换到 **hi-res fix** 以及 **real-ESRGAN** 等替代放大工具。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **SVD CUDA 实现的困扰**：一位用户询问 **cusolver/hipSolver** 如何执行 SVD，因为发现的大多数实现仅仅是闭源解决方案的包装器（wrappers）。
   - 引用了一个 [GitHub 仓库](https://github.com/Michalos88/Randomized_SVD_in_CUDA) 以获取见解，其中提到了 **Gram-Schmidt**、**Householder reflections** 和 **Givens rotations** 等方法。
- **使用 LLM 构建：年度回顾**：一段[视频和博客文章](https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/)总结了从业者使用 LLM 工作一年的经验教训，强调了战术、运营和战略方面的见解。
   - 作者创建了一个视觉化的 TLDR 以使该系列更易于理解，并强调观看视频对深入理解很有价值。
- **Triton 性能分析工具**：成员们讨论了适用于分析 Triton kernel 的工具，特别是针对峰值内存占用等任务。
   - 推荐使用 *nsight-compute* 和 *nsight-systems* 进行详细的性能分析（profiling），并指出应避免使用 *nvprof*，因为它已被这些新工具取代。
- **A100 上 FP16 与 FP32 的性能对比**：讨论集中在为什么 A100 的 **FP16 性能**是 **FP32 的 2 倍**，尽管从计算复杂度来看，预期的比例应该在 **2.8 倍**左右（参考 [Ampere Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)）。
   - 成员们调查了性能瓶颈是否可能是 **I/O-bound**（I/O 密集型）而非计算密集型，并讨论了硬件架构和开销。
- **9 月 21 日 CUDA MODE 线下活动邀请**：**CUDA MODE** 团队成员受邀参加 **9 月 21 日**在旧金山举行的线下（IRL）活动，该活动恰逢 PyTorch devcon，届时可能会有关于 **llm.c** 的 20 分钟演讲。
   - 活动的物流和详情通过 [Google Document](https://docs.google.com/document/d/10LkM5_xLh9r_ycul2ywOfgrOGmNP9YDTa9c4V755QgY/edit) 共享。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **在 C# 中解析 GGUF 文件元数据**：一位用户宣布在 C# 中实现了 GGUF 文件元数据的解析，并介绍了一个旨在报告元数据和提供统计信息的控制台工具。
   - 最初打算作为用于文件属性的 ShellExtension，但开发者遇到了注册问题，因此选择专注于控制台工具。
- **量化揭秘**：关于量化过程（q5, q6, f16）的详细解释，旨在帮助在旧硬件上运行大型模型。
   - 讨论包括 q5 如何比 q8 具有更高的量化程度，以及关于在配备 **RTX 3050 的 Dell 笔记本电脑**等显存（VRAM）有限的设备上运行大型模型的见解。
- **Hugging Face API 中断 LM Studio 搜索**：由于 Hugging Face API 的问题，LM Studio 的搜索功能失效，导致用户进行了多次故障排除尝试。
   - 该问题最终得到解决，恢复了应用内的搜索功能。
- **Nexusflow 发布 Athese 模型**：**Nexusflow** 推出了 **Athese 模型**，展示了令人印象深刻的结果，并可能成为其同尺寸模型中当前的 SOTA。
   - 该模型展示了卓越的**多语言性能**，使其适用于英语社区以外的用户。
- **使用 LM Studio 创建 Discord 机器人**：一位开发者分享了一篇关于使用 LM Studio.js 制作 Discord 机器人的[博客文章](https://dev.to/mrdjohnson/i-made-a-discord-bot-with-lmstudiojs-4fd6)。
   - 该文章包含教程和在 [GitHub](https://github.com/mrdjohnson/lmstudio-discord-bot) 上提供的源代码，详细说明了私密响应所需的修改。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Chrome 重启修复 Pro 图像生成问题**：一位用户通过[重启 Chrome 浏览器](https://www.perplexity.ai/page/image-generation-issue)解决了 Pro 订阅用户只能生成一张图片的问题。**在此修复后，Pro 用户可以期待更流畅的图像生成体验**。
   - 社区成员指出，图像生成功能需要更好的错误处理机制，以避免依赖浏览器重启。
- **GPTs Agents 在训练后难以学习新信息**：讨论指出 GPTs Agents 在初始训练阶段后无法从新信息中学习。
   - 克服这一问题的建议包括增量更新和社区驱动的补丁，以增强适应性。
- **Perplexity API 关于 Token 计费的说明**：**Perplexity API** 现在对入站和出站 Token 均进行计费，正如最近的一个讨论串中所述。
   - 用户对计费透明度表示担忧，并要求提供详细文档以更好地理解这些费用。
- **YouTube 测试其对话式 AI 功能**：Perplexity AI 报道了 [YouTube 测试](https://www.perplexity.ai/page/youtube-tests-ai-conversationa-WMQ_b8XNQZuIhMpPPMyfGg)新的 AI 对话功能，以评估其在增强用户参与度方面的效果。
   - 社区初步反应不一，一些人对更好交互的潜力感到兴奋，而另一些人则对 AI 的对话深度持怀疑态度。
- **OpenAI 推出 GPT-4.0 Mini**：[OpenAI 的 GPT-4.0 Mini](https://www.perplexity.ai/page/openai-drops-gpt-4o-mini-viKDYptISzufyJDPoL3Etg) 首次亮相，提供了一个专注于易用性且不牺牲复杂功能的精简版本。
   - 早期反馈强调了其在计算效率和性能之间的出色平衡，使其适用于更广泛的应用场景。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **4O Mini 表现优于 Sonnet 3.5**：**4O Mini** 可以解决连整个 Claude 系列都难以处理的复杂问题，展示了其卓越的能力。
   - 这激发了用户对 GPT-4o mini 取代 GPT-3.5 并主导高级用例潜力的期待。
- **为珊瑚礁研究微调多模态模型**：一位用户询问关于微调多模态模型以研究珊瑚礁的问题，但另一位用户建议使用 Azure AI 的 Custom Vision Service 以获得更好的准确性。
   - 未引用具体的模型或数据集，凸显了该领域对更具针对性建议的需求。
- **API 缺乏实时语音集成**：讨论强调 OpenAI 的新语音功能可在 ChatGPT 中使用，但尚未在 API 中提供，引发了对功能限制的担忧。
   - 成员们注意到了显著的延迟和质量差异，观点倾向于认为 ChatGPT 更适合终端用户的实时交互。
- **改进 ChatGPT 的响应修改**：一位用户在指示 ChatGPT 仅修改特定文本部分而不重写整个响应时遇到挑战，这是用户中的常见问题。
   - 建议包括使用设置中的“Customize ChatGPT”部分，并分享详细的自定义指令以提高准确性。
- **ChatGPT Voice 与 API Text-to-Speech 的对比**：用户对 ChatGPT 的新语音功能与 API 的 Text-to-Speech 端点之间的延迟和质量差异提出了担忧。
   - 成员们提出了潜在的改进和替代方案，但承认了当前 API 在实时应用方面的局限性。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **排行榜页面因迁移变慢**：由于向新基础设施迁移，排行榜页面在本周末更新缓慢，且经常显示**过时数据**。
   - 用户应预期在此期间排行榜会出现延迟和不准确。
- **GPTScript 的 OpenRouter 提供程序现已发布**：宣布了新的 [GPTScript OpenRouter 提供程序](https://github.com/RobinVivant/gptscript-openrouter-provider)，并在 GitHub 上提供了图片和详细说明。
   - 该工具对 GPTScript 应用程序的开发做出了重大贡献。
- **Dolphin Llama 70B 的性能问题**：在 0.8-1 的温度（temperature）下使用 **Dolphin Llama 70B**，在 7k token 的上下文对话中导致了异常行为，产生了由代码和无关输出组成的混乱内容。
   - 另一位成员指出 **Euryale 70B** 的 fp8 量化模型也存在类似问题，认为问题可能源于量化过程。
- **DeepSeek 的低成本与高效率**：**DeepSeek v2** 是一款拥有 236B 参数的 MoE 模型（每个 token 激活 21B），因其出色的性能和每百万输入 token 0.14 美元的成本效益而受到赞誉。
   - *“DeepSeek 的定价非常有竞争力，而且似乎利润丰厚，”* 解释了他们使用高 Batch Sizes 和压缩技术的策略。
- **Llama 3.1 405B 的泄露信息**：**Llama 3.1 405B Base** 显然由于 HuggingFace 的失误提前泄露，引发了关于其通过 RoPE 缩放扩展上下文能力的讨论。
   - 成员们感到兴奋，期待能有效利用该模型的软件更新，并渴望官方指令（instruct）模型的发布。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **利用 LLM 增强 RPG 游戏**：社区对在 RPG 项目中使用 **LLM** 进行分类、JSON 和对话生成表现出兴趣，**CMD-R** 因其严格的指令遵循能力成为首选。
   - 在成功测试后，成员们讨论了进一步的增强和集成可能性，扩展了 AI 在 RPG 游戏玩法中的作用。
- **Cohere API 中的结构化输出**：Cohere 宣布 **Command R** 和 **Command R+** 现在可以生成 **JSON** 格式的结构化输出，增强了下游应用程序的集成和数据分析。
   - 这一新功能在[此处](https://docs.cohere.com/docs/structured-outputs-json)有详细文档，旨在简化开发人员的数据工作流。
- **Cohere 和 Fujitsu 推出新的企业级 AI 服务**：Cohere 和 **Fujitsu** 建立了战略合作伙伴关系，在日本提供新的企业级 AI 服务，详见其[博客](https://cohere.com/blog/toolkit-features-july-2024)。
   - 此次合作旨在提高各种应用程序的 AI 服务可访问性和性能，重点展示了 Cohere 工具包的进步。
- **使用 Command R+ 的互动多人文字游戏**：一位成员介绍了 **Command R+**，这是一款用于创建和玩多人文字游戏的 Discord 应用，增强了游戏社区的社交和互动方面。
   - 该应用在 [Product Hunt](https://www.producthunt.com/posts/create-n-play) 上展示，为参与社区体验提供了无限可能性。
- **开发者办公时间 (Developer Office Hours) 2.0 启动**：Cohere 举办了另一场开发者办公时间会议，讨论了新的 API 功能、工具包更新以及最近的 **Cohere For AI** 研究论文。
   - 邀请社区成员参加这些会议，讨论更新、分享见解并就各种倡议进行交流。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Nemotron-340B 悬赏引发关注**：[Nathan](https://x.com/natolambert/status/1814735390877884823) 提供了起步价为 75 美元的悬赏，用于将 **Nemotron-340B** 转换为 **HuggingFace** 格式，并实现 **FP8 量化**和多节点部署。
   - 该悬赏金额已飙升至 **2,000 美元以上**，合成数据社区对此表现出浓厚兴趣。
- **Hypernetworks 与 Scaling Law 之争**：Hypernetworks 面临 Scaling Law 的约束，需要达到 **小于 O(scaling_law(output_model_compute)(target_error))** 才能实现目标误差。
   - 讨论集中在预测神经网络的任务是否需要更简单，或者是否需要一个“良好”的 Scaling Law 才能奏效。
- **特征污染与 OOD 泛化**：一篇关于 [OOD 泛化](https://arxiv.org/abs/2406.03345) 的论文详细说明了神经网络深受特征污染（Feature Contamination）之苦，从而影响泛化性能。
   - 相关讨论强调了归纳偏置（Inductive Biases）和 SGD 动力学在构建解释这些模型失效的潜在统一理论中的重要作用。
- **跨参数化方案与优化器的缩放指数**：一条关于缩放指数的 [推文](https://x.com/main_horse/status/1810647037718999342) 讨论了跨优化器和模型的发现，涉及超过 **10,000 个模型**。
   - 核心见解：**O(1/n) LR 调度**优于 mUP，成功的 **hparam transfer** 跨越了各种配置，并提出了一种新的 **Adam-atan2 优化器**以避免梯度下溢问题。
- **MATS 7.0 申请开放**：**Neel Nanda** 和 **Arthur Conmy** 已开放其冬季 MATS 7.0 课程的申请，截止日期为 *8 月 30 日*。[公告](https://x.com/NeelNanda5/status/1813921161052635209) 和 [申请文档](https://tinyurl.com/neel-mats-app) 已发布。
   - MATS 项目强调其在促进机械可解释性（Mechanistic Interpretability）研究方面的独特贡献。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Nemotron-4-340B 转换为 HuggingFace**：[Nathan Lambert](https://x.com/natolambert/status/1814735390877884823) 提供 75 美元的付费悬赏，将 **nvidia/Nemotron-4-340B-Instruct** 转换为 HuggingFace 格式。
   - 此举旨在为蒸馏（Distillation）项目解锁合成许可数据，需要同时实现 FP8 量化和多节点部署。
- **Llama-3 和 3.1 泄露引发热议**：关于 **Llama-3 405b** 和 **Llama 3.1** 模型及其基准测试的传闻和泄露被广泛讨论，参考了 [Azure 的 GitHub](https://github.com/Azure/azureml-assets/pull/3180/files) 和 [Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1e9hg7g/azure_llama_31_benchmarks/)。
   - 泄露的基准测试显示 **Llama 3.1** 在多个领域优于 GPT-4（HumanEval 除外），引发了关于其潜在优越性的讨论。
- **ICML 2024 关注可测量忠实度模型**：**Andreas Madsen** 宣布了他们在 ICML 2024 上的 Spotlight 论文，介绍了一种[新的可解释性方法](https://x.com/peterbhase/status/1814692347407429706?s=46)：Faithfulness Measurable Models（可测量忠实度模型），声称解释效果提升 2-5 倍，并提供准确的忠实度指标。
   - 一位用户指出其与 2021 年 NeurIPS 的一篇论文相似，强调了在投稿中改进**文献综述**的必要性。
- **Meta AI 潜在的付费服务**：有推测称 **Llama 405B** 可能是 Meta AI 付费服务的一部分，代码片段和 [Testing Catalog 的推文](https://x.com/testingcatalog/status/1815439546722451493?s=46) 暗示了这一点。
   - 热议内容包括可能的 Meta AI API 平台 AI Studio，预计将于 7 月 23 日发布公告。
- **UltraChat 出人意料的有效性**：讨论指出 **Zephyr 论文**将 UltraChat 数据从 **150 万**大幅过滤至 **20 万**，对数据质量提出了质疑。
   - 尽管经过严格过滤，**UltraChat** 的效果依然出奇地好，引发了对其生成过程的进一步探究。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Langfuse 优于 Langsmith**：来自用户的反馈表明 [Langfuse](https://github.com/langfuse/langfuse) 的表现优于 Langsmith，用户分享了其在自托管和集成方面的良好体验。
   - 创始人 *Clemo_._* 鼓励更多的社区互动，强调他们致力于维护一个优秀的 OSS 解决方案。
- **GPT-4o Mini 赋能 AI 生成内容**：OpenAI 的新模型 [GPT-4o mini](https://batchmon.com/blog/ai-cheaper-than-ads/) 每 1M 输入 token 的成本仅为 $0.15，这使得完全由广告支持的动态 AI 生成内容成为可能。
   - 讨论内容包括对 Web 内容的潜在影响，假设内容产出将向更多 AI 生成的方向转变。
- **Harvey AI 的传闻与预测**：关于 [Harvey AI](https://x.com/emilyinvc/status/1814741780010844289?s=46) 可行性的传闻和质疑浮出水面，有人称其为一家“虚有其表（smoke and mirrors）”的公司。
   - 随后引发了关于垂直领域 AI 初创公司面临挑战的辩论，包括对大型 AI 实验室的依赖以及当前的行业周期。
- **Elon Musk 的孟菲斯超级集群**：Elon Musk 宣布启动孟菲斯超级集群（Memphis Supercluster），声称这是世界上最强大的 AI 训练集群，在单个 RDMA fabric 上拥有 10 万块液冷 H100。
   - 然而，[事实核查](https://x.com/dylan522p/status/1815494840152662170?s=46)揭示了在功耗和 GPU 可用性方面的差异，表明该设施尚未完全投入运营。
- **LLaMA 3.1 泄露引发关注**：泄露的 [LLaMA 3.1](https://x.com/mattshumer_/status/1815444612414087294?s=46) 评估结果表明，其 8B、70B 和 405B 模型甚至在进行 instruct tuning 之前，就可能超越当前最先进的模型。
   - 这些泄露引发了广泛的期待，以及对开源 AI 模型未来能力的推测。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Triplex 将 KG 成本降低 98%**：SciPhi 的新模型 [Triplex](https://huggingface.co/SciPhi/Triplex) 将知识图谱（Knowledge Graph）的构建成本降低了 98%，以极低的成本超越了 **GPT-4**。
   - 该模型从非结构化数据中提取三元组（triplets）并支持本地运行，使经济实惠、易于获取的知识图谱成为现实。
- **Mistral 12b 分词器问题**：多位成员提出了 **Mistral 12b** 分词器（tokenizer）的问题，尽管其指标看好，但输出的文本没有空格。
   - 输出内容被批评为“垃圾”，可能与特殊 token 处理问题有关。
- **LLaMA 3.1 基准测试表现亮眼**：成员们赞扬了 **LLaMA 3.1** 的基准测试结果，强调了 8B 和 70B 模型的卓越表现。
   - 70B 模型被特别指出**非常接近领先模型**，甚至超出了一些预期。
- **DeepSpeed Zero-3 兼容性修复**：一位用户解决了 **DeepSpeed Zero-3** 的兼容性问题，该问题涉及与 `low_cpu_mem_usage=True` 和自定义 `device_map` 设置相关的 **ValueError**。
   - 通过删除 accelerate 配置解决了该问题，恢复了无错误的设置。
- **Axolotl 训练遇到 GPU 瓶颈**：正如 **Phorm** 所指出的，Axolotl 中的训练错误追溯到了 GPU 显存瓶颈。
   - 排查步骤包括减小 batch size、调整梯度累积（gradient accumulation）以及切换到混合精度训练（mixed precision training）。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Hermes 2.5 性能超越 Hermes 2**：在添加了[代码指令示例](https://link.to.examples)后，**Hermes 2.5** 在各项基准测试中的表现似乎优于 **Hermes 2**。
   - Hermes 2 在 MMLU 基准测试中得分为 **34.5**，而 Hermes 2.5 得分为 **52.3**。
- **Triplex 大幅降低知识图谱构建成本**：SciPhi.AI 最新开源的 [Triplex](https://huggingface.co/SciPhi/Triplex) 将知识图谱构建成本降低了 98%，以 1/60 的成本实现了超越 GPT-4 的性能。
   - Triplex 是 Phi3-3.8B 的微调版本，能够从非结构化数据中提取三元组（triplets），增强了如微软 Graph RAG 等 RAG 方法。
- **将 RAG 应用部署到生产环境**：一名成员分享了关于使用 MongoDB Atlas 与 LangChain 构建 RAG 实现的[教程](https://www.mongodb.com/docs/atlas/atlas-vector-search/ai-integrations/langchain/)。
   - 该教程涵盖了环境搭建、数据存储、创建搜索索引以及运行向量搜索查询。
- **LangChain 初学者友好文章发布**：一位用户分享了一篇关于 LangChain 及其组件的 [Medium 文章](https://medium.com/@ambaliaharshit25/ea17820a5c01)，旨在帮助对了解其应用感兴趣的初学者。
   - *想象一下，拥有一个可以通过简单的自然语言命令处理复杂任务的虚拟助手*，文章深入探讨了这些组件为何如此重要。
- **面向 TypeScript 的 AI 驱动函数构建器**：在一次黑客松活动中开发了一个名为 [AI Fun](https://github.com/mishushakov/ai-fun) 的新项目，用于构建由 LLM 驱动的 TypeScript 函数。
   - 该项目利用 AI 来自动化并简化 TypeScript 函数的构建过程。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Bud-E 展示具有开源目标的新 Demo**：**Bud-E 语音助手**发布了一个 Demo，展示了未来每个人都能以电力成本获得高性能开源系统的愿景。
   - 目前针对 Ubuntu 优化的代码库将进行重构，以实现客户端、服务器以及可互换的 ASR、TTS、LLM 组件之间的清晰分离。
- **加入 BUD-E Discord 服务器进行协作**：邀请志愿者加入新的 **BUD-E Discord 服务器**，帮助进一步开发语音助手并贡献类似于 Minecraft Mods 的新技能。
   - 每天欧洲中部夏令时间（CEST）晚上 9 点将举行在线黑客松会议，以引导新志愿者并协调项目工作。
- **切换回 Epochs 进行损失曲线绘制**：一名成员最初使用墙上时钟时间（wall-clock time）绘制损失曲线，但发现切换回 Epochs 来衡量模型学习效率更有意义。
   - 该成员发现 **WandB** 在这方面非常有用，但承认最初的改变是错误的，是一个“愚蠢”的决定。
- **Mem0 为 LLM 引入智能记忆层**：**[Mem0](https://docs.mem0.ai/overview)** 发布了针对大语言模型的记忆层，通过用户、会话和 AI Agent 记忆以及自适应个性化等功能，实现个性化的 AI 体验。
   - 有关集成和功能的更多信息，请查看 Mem0 的 **[GitHub 页面](https://github.com/mem0ai/mem0)**。
- **Datadog 发布时间序列建模的 SOTA 结果**：Datadog 发布了关于时间序列建模的 **[最新 SOTA 结果](https://www.datadoghq.com/blog/datadog-time-series-foundation-model/)**，并正在积极招聘研究职位。
   - Datadog 的基础模型旨在通过识别趋势、解析高频数据和管理高基数数据来有效处理时间序列数据。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **PostgresML 提升 Reranking**：[PostgresML](https://t.co/HWfitT0CJt) 用于 Reranking，通过额外的参数实现精确控制，从而增强搜索结果的相关性。
   - 一篇[客座博客](https://t.co/HWfitT0CJt)解释了托管索引方法如何在实际应用中优化 Reranking。
- **LLM 作为生产环境评判者**：与 Yixin Hu 和 Thomas Hulard 的[会议](https://t.co/i84Cg5pqsy)讨论了在生产系统中部署 LLM 作为评判者。
   - *本次会议涵盖了开发过程中 RAG 评估背后的核心概念和实践。*
- **Merlinn：开源 On-call Copilot**：[Merlinn](https://t.co/rAM5OOxQ34) 推出了一款用于事件管理的 AI 驱动 Slack 助手。
   - *它与 Datadog 等可观测性和事件管理工具集成。*
- **使用 Ollama 和 Qdrant 简化多模态 RAG**：[Pavan Mantha](https://t.co/0gcz4GfCh5) 发表了一篇关于使用 Ollama 和 Qdrant 构建多模态 RAG 的文章。
   - 该指南包括摄取音频/视频源以及通过文本转录索引数据的步骤。
- **通过 Deasie 工作坊改进 RAG**：与 Deasie 联合创始人的 [YouTube 会议](https://t.co/cJPsNaWgoc)涵盖了用于改进 RAG 的高级解析和元数据。
   - 解析和元数据增强被强调为提升 RAG 性能的关键技术。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **GPT4o-mini 模型在冗余度方面表现不佳**：据报告，**GPT4o-mini** 存在冗长和重复的问题，与 **GPT3.5-turbo** 相比，影响了数据提取。
   - 此问题导致数据流水线效率显著降低，需要更好的模型调优或替代方案。
- **DSPy Tracing 发布，增强工作流**：新的 **DSPy tracing** 功能现已上线，提供对模块、预测、LM 和检索器的有效跟踪（[文档点击此处](https://docs.langwatch.ai/integration/python/guide#capturing-llm-spans)）。
   - 此次更新预计将显著简化调试和性能跟踪。
- **TypedPredictors 兼容性受限**：**GPT-4o** 和 **Sonnet-3.5** 在处理复杂的 Pydantic 类生成方面表现独特，而其他模型则表现不足。
   - 这种局限性要求根据项目需求仔细选择模型，特别是在处理复杂数据结构时。
- **DSPy 中的联合优化带来巨大收益**：一篇新的 [DSPy 论文](https://x.com/lateinteraction/status/1815423177272824022) 揭示，在提示词优化和微调之间交替进行，可带来高达 **26% 的性能提升**。
   - 该研究验证了双重优化策略优于单一方法策略的效率（[论文链接](https://arxiv.org/abs/2407.10930)）。
- **DSPy 优化器的可靠性讨论**：**BootstrapFewShotWithRandomSearch** 优化器被强调为一个可靠且简单的起点。
   - 成员们讨论了各种优化器的可靠性，指出 **BootstrapFewShotWithRandomSearch** 因其简单性和鲁棒性而脱颖而出。



---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **George Hotz 推动 OpenPilot 洞察**：George Hotz 分享了 [OpenPilot 模型运行分析](https://gist.github.com/geohot/8d7edc7ac2fd9a31ea563c134b66cddb)，重点记录了 kernel 更改和潜在的性能下降。
   - 他提到，对于任何有技术倾向的人来说，这项任务都应该是可以上手的，但指出一些初学者可能会忽略最初的问题解决过程。
- **关于 Tinygrad 中 Bitcast 形状的辩论**：Tyoc213 询问 Tinygrad 中的 `bitcast` 函数是否应与 TensorFlow 的 `bitcast` 保持一致，特别是在形状差异方面。
   - George Hotz 和成员们一致认为将 Tinygrad 与 TensorFlow/Torch/Numpy 同步是合理的，Tyoc213 承诺进行必要的更新。
- **Tinygrad 中极具前景的 PR**：George Hotz 认可了 [Tyoc213 提交的一个 pull request](https://github.com/tinygrad/tinygrad/compare/master...tyoc213-contrib:tinygrad:tyoc213/bitcast-all)，认为其测试详尽，非常值得关注。
   - Tyoc213 对此表示感谢，并透露了进一步与其他框架标准对齐的计划。
- **Tinygrad 每周会议亮点**：Chenyuy 分享了周一会议的议程，详细介绍了 tinybox、hcopt 速度恢复以及 [MCTS 搜索增强](https://github.com/tinygrad/tinygrad/blob/master/extra/mcts_search.py) 的更新。
   - 讨论还包括更好的搜索功能、conv backward fusing、fast Llama 改进，以及针对 kernel 和 driver 改进的各种悬赏任务（bounties）。
- **关于 Tinygrad 可行性的辩论**：成员们辩论了 **Tinygrad 的可行性** 与 PyTorch 的对比，讨论是现在切换还是等待 **1.0 版本**。
   - 讨论反映了对生产力的担忧，并且明显受到了一段关于 **Shapetrackers** 的详细 [YouTube 实现教程](https://www.youtube.com/watch?v=g1rCrv1fx1A) 的推动。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Crowdstrike 入职首日更新**：[Vinceflibustier](https://fixupx.com/vinceflibustier/status/1814233715641389456) 分享了一个关于他在 **Crowdstrike** 第一天的轻松更新，提到他们推送了一个小更新后下午就下班了。
   - 消息以和平手势表情符号结束，营造出一种随性友好的氛围。
- **Python 3.12 中的 Python 子解释器**：一位成员分享了 [Python 子解释器教程](https://realpython.com/python312-subinterpreters/?utm_source=perplexity)，详细介绍了 Python 3.12 中 **GIL 控制**和并行性的增强，以及 3.13 变化的预览。
   - 该教程讨论了 **CPython 全局状态**的更改，旨在提高并行执行效率，并建议读者熟悉 Python 基础知识。
- **Meta Llama 3.1 仓库泄露**：[AlpinDale 确认](https://x.com/alpindale/status/1814814551449244058?s=12) Meta Llama 3.1 包含一个由 405B 模型蒸馏而成的 8B 和 70B 模型，具有 128k context，并指出 405B 模型无法画出独角兽。
   - 该仓库被[提前意外公开](https://x.com/alpindale/status/1814717595754377562?s=46)，保留了与 **Llama 3** 相同的架构，其 instruct tuning 可能进行了安全对齐。
- **Deepseek Chat v2 6.28 表现优于 Deepseek Coder**：一位成员提到 *Deepseek chat v2 6.28 更新* 表现极其出色，甚至超过了 *Deepseek coder*，且比 **4o mini** 更具成本效益。
   - 此次更新强调了 Deepseek chat v2 6.28 改进的性能指标和成本优势。
- **Pinokio 的 Augmentoolkit 在 GitHub 上发布**：**Pinokio** 的新项目 [Augmentoolkit](https://github.com/pinokiofactory/augmentoolkit) 已在 GitHub 上发布供公众使用，该工具集旨在增强 AI 应用。
   - 该项目在 [Discord](https://discord.gg/TQdNwadtE4)、[GitHub](https://github.com/pinokiocomputer/pinokio) 和 [Twitter](https://twitter.com/cocktailpeanut) 上都获得了关注，引发了广泛兴趣。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **使用 GPT 模型进行微调成本高昂**：由于**高昂的成本**和**供应商锁定 (vendor lock-in)**，微调 GPT 模型的情况较少。这涉及昂贵的 API 调用以及对特定公司基础设施的依赖。
   - #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1264159625255587840) 频道中的讨论强调了这些因素如何阻碍了许多人采用微调实践。
- **OpenAI credits 依然难以获取**：有报告称在接收 [OpenAI credits](https://link.to/openai-credits) 方面存在问题，成员们提供了组织 ID **org-EX3LDPMB5MSmidg3TrlPfirU** 以及多次提交表单的细节。
   - 尽管遵循了流程，但额度仍未分配，详见 #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/1264575901673193625) 频道。
- **探索在其他提供商上使用 Openpipe**：除了 **OpenAI** 或 **Anthropic**，有人咨询了如何在 **Replicate** 或 **Modal** 等提供商上使用 **Openpipe**。
   - 讨论集中在集成来自 **Replicate** 的模型，同时确保与现有系统的兼容性，如 #[openpipe](https://discord.com/channels/1238365980128706560/1245927847437008896/1264115136164270090) 频道所示。
- **东海岸聚会定于 8 月下旬**：#[east-coast-usa](https://discord.com/channels/1238365980128706560/1245411101718610001/) 频道提出了 8 月下旬在纽约举行聚会的建议。
   - 成员们正在考虑这次非正式聚会的物流安排。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **OpenAI Scale Tier 困惑**：关于新推出的 [OpenAI Scale Tier](https://openai.com/api-scale-tier) 的讨论让许多人感到困惑，特别是关于不同模型的每秒吞吐量 (TPS) 计算。
   - 疑问集中在现收现付 (pay-as-you-go) 层级的 19 TPS 计算，以及与 GPT-4-o 约 80 TPS 吞吐量的对比。
- **Websim 寻找创始 AI 工程师**：[Websim](https://websim.ai/) 的使命是创建世界上适应性最强的软件创作平台，赋能个人解决自己的挑战。
   - 该公司正在招聘一名创始 AI 工程师，负责建立系统，以便针对自动化产品开发快速迭代非确定性程序。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **构建 LLM 一年的深刻见解**：一位用户分享了 [视频和博客文章](https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/)，总结了从业者使用 LLM 构建应用一年的经验教训三部曲。
   - 总结强调了战术、运营和战略方面的见解，并建议通过视频观看内容以便更好地理解。
- **BUD-E 语音助手邀请合作**：一位用户分享了 [YouTube 视频](https://youtu.be/O4IXfa8CROs)，展示了开源 BUD-E 语音助手的演示，并邀请其他人加入他们新的 Discord 服务器进行合作。
   - 每日在线黑客松 (hackathons) 将于 **9 PM CEST** 开始，以引导新志愿者并协调项目工作。

---

## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **艺术家 Aria 寻求创意合作**：Aria 介绍自己是一名 **2D/3D 艺术家**，正在 AI 社区寻找合作机会。
   - 他们邀请感兴趣的成员通过 **私信 (direct message)** 联系，探讨潜在的伙伴关系项目。
- **无其他可用话题**：提供的消息历史中没有讨论或分享其他重要话题。
   - 此摘要反映了缺乏进一步的技术讨论、公告或值得注意的事件。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **明确目标受众需求**：一位成员就目标受众和沟通策略背后的主要目标提出了疑问。
   - 讨论提到了在讨论产品时，需要为工程师、准工程师、DevRel 和解决方案架构师制定不同的方法。
- **针对不同角色的战略沟通**：探索了各种沟通策略，以有效地吸引工程师、DevRel、解决方案架构师和准工程师。
   - 参与者一致认为，每个角色都需要量身定制的信息，以清晰地传达产品功能和优势。

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **使用 LLM 构建一年的经验教训**：[Lessons from 1 Year of Building with LLMs](https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/) 详细介绍了来自六位从业者的战术、运营和战略见解。
   - 该系列配有一个视觉化的 [TLDR 视频](https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/)，使这些经验教训更易于理解。
- **TLDR 系列重磅条目**：TLDR 系列为深入参与 LLM 的人员提供了由 [六位作者](https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/) 分享的深入且具可操作性的建议。
   - 作者们推荐该系列作为 LLM 从业者的重要资源。

---

**Mozilla AI Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。

---

# 第二部分：按频道分类的详细摘要和链接

{% if medium == 'web' %}

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1264137046340009984)** (3 条消息): 

> - `Implicit Chain-of-Thought (CoT)`
> - `UltraChat and Multi-turn Interactions`
> - `Multiagent Debate Models` 

- **在 GPT-2 中通过隐式 CoT 实现高准确率**：提出了一种新方法，通过从为显式 **Chain-of-Thought (CoT)** 训练的模型开始，逐渐移除中间步骤并进行微调，从而使 **GPT-2 Small** 模型能够以 99% 的准确率解决 9x9 乘法问题 ([arxiv](https://arxiv.org/pdf/2405.14838))。
   - 同样的方法也提升了如 **Mistral 7B** 等更大模型的性能，在不产生任何中间步骤的情况下，在 GSM8K 上实现了超过 50% 的准确率。
- **寻求关于多轮用户-Agent 交互的论文**：一位成员正在寻求类似于 **UltraChat** 的论文推荐，讨论如何构建多轮用户-Agent 交互，并提到 **SODA** 论文是一个潜在的读物 ([arxiv](https://arxiv.org/abs/2212.10465))。
   - “UltraChat 引用了 SODA 论文，因为在给定的图表中，两者的平均轮数相似。”
- **通过多智能体辩论提升事实性与推理能力**：一种新方法通过多智能体辩论增强了语言模型的**事实性与推理能力**，其中多个语言模型实例在多轮中提出并辩论响应 ([composable-models](https://composable-models.github.io/llm_debate/))。
   - 该方法显著提升了跨各种任务的数学和战略推理能力。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://composable-models.github.io/llm_debate/"> Improving Factuality and Reasoning in Language Models with Multiagent Debate</a>：通过多智能体辩论提升语言模型的事实性与推理能力</li><li><a href="https://arxiv.org/abs/2212.10465">SODA: Million-scale Dialogue Distillation with Social Commonsense Contextualization</a>：数据稀缺一直是开放域社交对话领域长期存在的问题。为了解决这一需求，我们推出了 SODA：第一个公开可用的、百万级高质量社交对话数据集...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[datasets](https://discord.com/channels/1053877538025386074/1105324249721356298/)** (1 条消息): 

fedorovist: https://huggingface.co/datasets/jdpressman/retroinstruct-mix-v0.2
  

---

### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1264698222387925073)** (5 messages): 

> - `PPLX Pro Search AI`
> - `ReFT 论文讨论`
> - `Greg Schoeninger 关于 ReFT 的 Reddit 帖子`
> - `关于 ReFT 的 YouTube 视频`
> - `Oxen.ai 社区和 Paper Club` 


- **PPLX Pro Search 发现自己是 AI**：一位用户幽默地评论说，**PPLX Pro Search** 在发现自己是 AI 而非人类时，反之亦然的想法也很有趣。
- **ReFT 论文在参数效率方面令人印象深刻**：ReFT 的参数效率比 LoRA 高出 15x-60x，微调模型速度极快，例如在 A10 上使用约 100 个示例，不到一分钟即可完成 Llama 2 7B 的微调。
   - 该技术作用于 **residual stream**（残差流）而非 K-V 矩阵，使其既**高效**又**可组合**。
- **Greg Schoeninger 关于 ReFT 的见解**：Greg Schoeninger 分享了他的 [Reddit 帖子](https://www.reddit.com/r/MachineLearning/comments/1e8qwnl/r_discussion_of_reft_paper_with_lead_author/)，讨论了 ReFT 论文，并附带了他的 Notion 笔记链接。
- **与作者 Zhengxuan Wu 合作的 ReFT YouTube 视频**：一段名为 *"How ReFT Works w/ Author Zhengxuan Wu"* 的 YouTube 视频深入探讨了来自 Stanford 的 ReFT 论文，Greg 提供了深度知识和易于理解的解释。
   - 在 [YouTube](https://www.youtube.com/watch?v=to2oKwnknUk&t=2770s) 上观看视频，详细了解 ReFT 技术的工作原理。
- **Oxen.ai 培养 AI 爱好者社区**：Oxen.ai 通过每周五举办 **Paper Clubs** 来讨论和应用研究论文，从而促进学术研究人员和开发者的社区建设。
   - 加入社区并在 [Oxen.ai](https://oxen.ai/community) 订阅未来的 Paper Club 邀请。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=to2oKwnknUk&t=2770s">How ReFT Works w/ Author Zhengxuan Wu</a>: 我们与作者之一 Zhengxuan Wu 一起深入探讨来自 Stanford 的 ReFT 论文。--使用 Oxen AI 🐂           https://oxen.ai/Oxen AI 让你的数据版本化...</li><li><a href="https://oxen.ai/community">Community Resources | Oxen.ai</a>: 使用 Oxen AI 管理你的机器学习数据集。</li><li><a href="https://www.reddit.com/r/MachineLearning/comments/1e8qwnl/r_discussion_of_reft_paper_with_lead_author/">Reddit - Dive into anything</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

alexanderlong_84476: https://pluralisresearch.substack.com/p/decentralized-ai-looms
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1263934853448077343)** (434 messages🔥🔥🔥): 

> - `DeepSeek-V2-Chat-0628 的 1-bit 量化结果`
> - `新 AI 项目和模型更新`
> - `Llama 3.1 基准测试`
> - `AI 模型法律合规担忧`
> - `技术工具和部署经验` 


- **DeepSeek 惊人的 1-bit 量化结果**：一位成员分享了 DeepSeek-V2-Chat-0628 1-bit 量化的[疯狂结果](https://huggingface.co/nisten/deepseek-0628-gguf)，该结果针对 CPU 进行了优化，目前在 LMSYS Arena Hard 全球排名第 7。
   - *kotykd* 质疑了模型的连贯性以及与之前版本的差异，强调了具体的性能和内存使用数据。
- **Hermes 实现递归函数调用**：一位成员分享了 Ollama 上的 [Hermes-2-Pro-Llama-3-8b 模型](https://ollama.com/interstellarninja/hermes-2-pro-llama-3-8b-tools) 已实现递归函数调用，并提供了一个来自 Jupyter notebook 的示例。
   - 讨论包括对类似模型中 tool calling 的潜在改进和配置。
- **Llama 3.1 即将发布**：据报道，Llama 3.1 模型即将发布，较 3.0 有显著改进，包括用于提升性能的 405B distillation。
   - 成员们讨论了预期的基准测试和功能，例如即将发布的 3.1 instruct 版本中的原生 tool calling。
- **AI 工具部署和法律灰色地带**：作者分享了部署 AI 工具的经验，包括一位成员在利用聊天数据微调 Mistral instruct 并应用正确模板时遇到的困难。
   - 另一位用户对托管泄露模型及潜在的法律后果表示担忧。
- **新 AI 研究和教育内容**：分享了一篇题为 [“构建 LLM 一年的教训”](https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/) 的新博客文章，提供了从业者一年经验的简要总结。
   - 成员们讨论了易于获取的教育内容（尤其是视频格式）在使复杂概念变得易于理解方面的实用性。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/_akhaliq/status/1815225121637834802">来自 AK (@_akhaliq) 的推文</a>: Nvidia 发布 ChatQA 2，缩小与闭源 LLM 在长文本（Long Context）和 RAG 能力上的差距。在这项工作中，我们介绍了 ChatQA 2，这是一个基于 Llama3 的模型，旨在缩小开源与...</li><li><a href="https://x.com/nightgrey_/status/1815443846265774571">来自 Nico (@nightgrey_) 的推文</a>: @Teknium1 此外，这些显然是 Base 模型的评分，而不是 Instruct 微调后的模型！LFG!!</li><li><a href="https://discord.gift/ud3CQyFM2f6M6CdQ">Discord - 充满乐趣与游戏的群聊</a>: Discord 是玩游戏、与朋友聚会或建立全球社区的绝佳场所。自定义你的专属空间来聊天、玩耍和聚会。</li><li><a href="https://x.com/elonmusk/status/1815187468691316946">来自 Elon Musk (@elonmusk) 的推文</a>: 是时候举办一场 AI 时装秀了</li><li><a href="https://huggingface.co/nisten/deepseek-0628-gguf">nisten/deepseek-0628-gguf · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3">UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/mlx-community/Meta-Llama-3-70B-Instruct-4bit">mlx-community/Meta-Llama-3-70B-Instruct-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/twuK.gif">猫咪键盘 GIF - Cat Keyboard Cats - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/NousResearch/Hermes-Function-Calling/blob/main/examples/ollama_openai_tools_recursive.ipynb">Hermes-Function-Calling/examples/ollama_openai_tools_recursive.ipynb at main · NousResearch/Hermes-Function-Calling</a>: 通过在 GitHub 上创建账号，为 NousResearch/Hermes-Function-Calling 的开发做出贡献。</li><li><a href="https://www.deepspeed.ai/docs/config-json/">DeepSpeed 配置 JSON</a>: DeepSpeed 是一个深度学习优化库，使分布式训练变得简单、高效且有效。</li><li><a href="https://ollama.com/interstellarninja/hermes-2-pro-llama-3-8b-tools">interstellarninja/hermes-2-pro-llama-3-8b-tools</a>: [HERMES 工具模板] Hermes 2 Pro 是 Nous Hermes 2 的升级重新训练版本，包含更新且清洗过的 OpenHermes 2.5 数据集，以及新引入的 Function...</li><li><a href="https://github.com/exo-explore/exo">GitHub - exo-explore/exo: 在家中使用日常设备运行你自己的 AI 集群 📱💻 🖥️⌚</a>: 在家中使用日常设备运行你自己的 AI 集群 📱💻 🖥️⌚ - exo-explore/exo</li><li><a href="https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/">TLDR: 使用 LLM 构建的一年 – D-Squared</a>: 未找到描述</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1e9hg7g/azure_llama_31_benchmarks/">Azure Llama 3.1 基准测试</a>: 由 u/one1note 发布于 r/LocalLLaMA • 263 点赞和 245 条评论
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1264297511066144768)** (20 条消息🔥): 

> - `使用 LLM 破解验证码`
> - `LLama-3 405B 的 VRAM 估算`
> - `用于 AI 模型的消费级硬件` 


- **使用 LLM 破解文本验证码（Captcha）的挑战**: 一位成员分享了使用 **Florence**、**Moondream** 和 **InternVL2** 等模型破解文本验证码的经验，并指出准确率参差不齐。
   - 虽然 InternVL2 表现最**出色**，但该成员无法在本地运行，必须依赖在线 Demo。
- **LLama-3 405B 量化的 VRAM 需求**: 有人询问了在不同量化（Quantization）水平下运行 LLama-3 405B 的 **VRAM 估算**，结果显示其需要大约 **8-bit 410 GB、4-bit 205 GB 和 2-bit 102 GB**。
   - 一位参与者指出，这意味着 4-bit 版本至少需要 **9 块 24GB GPU**，尽管拥有多 GPU 插槽的服务器，这对大多数消费级配置来说仍不切实际。
- **对高 VRAM 需求的沮丧**: 用户希望能有更可行的消费级硬件，以便在本地运行像 LLama-3 405B 这样的大型模型。
   - 用户对硬件限制表示沮丧，并指出可能需要探索云端托管方案。


  

---

### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1264006054543626293)** (40 messages🔥): 

> - `Triplex 用于知识图谱构建`
> - `用于本地 LLM 实验的 R2R 平台`
> - `知识图谱在 RAG/LLM 中的应用`
> - `Microsoft's Graph RAG`
> - `更深层的邻接矩阵与符号推理` 


- **Triplex 彻底改变了知识图谱的创建**：Triplex 是 Phi3-3.8B 的微调版本，可将知识图谱的创建成本降低 98%，且成本仅为 GPT-4 的 1/60，同时性能更优，支持通过 SciPhi 的 R2R 进行本地图谱构建。
   - 由 [SciPhi.AI](https://www.sciphi.ai) 开发，Triplex 从非结构化数据中提取三元组（triplets），以创建具有成本效益的知识图谱。
- **R2R 弥合了本地 LLM 与可扩展 RAG 之间的差距**：[R2R](https://r2r-docs.sciphi.ai/cookbooks/knowledge-graph) 由 SciPhi.AI 开发，支持多模态数据摄取和混合搜索，允许通过 RESTful API 高效创建和管理知识图谱及文档。
   - 功能包括 **graph RAG**、可观测性、应用管理和可扩展性，助力构建可扩展且生产就绪的应用。
- **图谱增强了 RAG，提升对话式 AI 体验**：知识图谱通过促进符号推理和提高推荐系统的可用性，为检索增强生成（RAG）带来了新的分析形式。
   - **混合模式 RAG** 可以利用图谱进行中间符号推理步骤，优化从知识库中检索支持性事实的过程。
- **Microsoft's Graph RAG 方法革新了主观数据集**：**Microsoft's Graph RAG** 方法扩展了知识图谱，为更通用的问答任务创建增强的 RAG 数据集，在处理主观数据方面展现出潜力。
   - 该技术将知识图谱集成到 LLM 中，以获得更深层的上下文和稳健的响应。
- **利用深层邻接矩阵优化知识图谱**：三元组提取只是起点，需要更深层的**邻接矩阵（adjacency matrices）**来充分利用 LLM 的上下文长度。
   - 实体去重和消解将进一步增强知识图谱在符号推理任务中的准确性和实用性。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://r2r-docs.sciphi.ai/cookbooks/knowledge-graph.">Introduction - 优秀的开源 AI 驱动问答引擎</a>: 未找到描述</li><li><a href="https://kg.sciphi.ai/">SOTA 三元组提取</a>: 未找到描述</li><li><a href="https://huggingface.co/SciPhi/Triplex">SciPhi/Triplex · Hugging Face</a>: 未找到描述</li><li><a href="https://neo4j.com/developer-blog/global-graphrag-neo4j-langchain/?utm_source=Twitter&utm_medium=OrganicSocial&utm_campaign=GenAI-RAG--&utm_ID=&utm_term=&utm_content=-DevBlog--&utm_creative_format=&utm_marketing_tactic=&utm_parent_camp=&utm_partner=&utm_persona=">使用 Neo4j 和 LangChain 实现“从局部到全局”的 GraphRAG：构建图谱</a>: 了解如何结合文本提取、网络分析以及 LLM 提示词和摘要来提高 RAG 的准确性。</li><li><a href="https://huggingface.co/datasets/xlangai/BRIGHT?row=1">xlangai/BRIGHT · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1264807709685973024)** (3 messages): 

> - `World Sim`
> - `World Client`
> - `Nous Research` 


- **World Sim 简介**：一位成员询问关于 **World Sim** 的解释，随后有人提供了 [World Sim by Nous Research](https://worldsim.nousresearch.com) 的直接链接。
- **对 World Sim/World Client 的贡献**：另一位成员询问是否有人是 **World Sim/World Client** 项目的**贡献者**。
   - *此询问后没有详细的社区讨论或意见。*



**提及的链接**: <a href="https://worldsim.nousresearch.com">worldsim</a>: 未找到描述

  

---


### **Nous Research AI ▷ #[reasoning-tasks-master-list](https://discord.com/channels/1053877538025386074/1264666760972472481/1264666774624931922)** (691 messages🔥🔥🔥): 

> - `QuietStar`
> - `自动生成提示词`
> - `LLM 的类型系统`
> - `中间表示`
> - `Open-Reasoning-Tasks 中的任务结构`

- **QuietStar 概念启发了自动生成 Prompt**：在关于 *QuietStar* 的讨论中，成员们辩论了 LLM 如何从并行处理上下文的架构中受益，并提出了模型自动构建其后续 Prompt 的想法。
   - 参与者探讨了调整 LLM 架构如何实现动态 Prompt 生成，从而增强 Token 级别的推理能力。
- **LLM 类型系统提案引发辩论**：一位成员提议实现一种允许 LLM 构建和验证代码的类型系统，引发了关于其可行性和必要性的复杂讨论。
   - 尽管存在反对意见和困惑，这场辩论突显了关于将 LLM 输出形式化为机器可检查且具有表现力的语言之重要性的不同观点。
- **中间表示桥接代码与自然语言**：社区深入研究了使用中间表示（Intermediate representations）来管理 LLM 输出，在代码和自然语言之间取得平衡，特别是针对推理和重构等复杂任务。
   - 讨论强调了将自然语言任务转换为结构化中间体的框架潜力，以促进更好的程序化控制和验证。
- **Open-Reasoning-Tasks 仓库中的任务结构优化**：参与者致力于优化 *Open-Reasoning-Tasks* 仓库中推理任务的结构，强调需要更清晰的示例，并可能为每个任务建立独立文件。
   - 会议考虑了如何使任务示例更加严谨，并使任务描述更具可读性和机器可解析性。
- **为增强 LLM 推理能力提出的各种框架和工具**：在关于提升 LLM 推理能力的辩论中，Prolog、ProbLog 和其他逻辑编程语言与 Python 一起成为将形式逻辑引入 LLM 任务的候选方案。
   - 受 *Logic-LLM* 等工具的启发，对话强调了概率推理和 Multi-agent 系统的必要性，以实现基于经验的多框架推理。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://operation-athena.repleteai.com/">未找到标题</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2401.06751">简单训练数据对困难任务的非凡有效性</a>：当困难的训练数据在定义上就难以正确标注时，我们如何训练模型在困难的测试数据上表现良好？这个问题被称为可扩展监督（scalable oversight）问题，并且已经...</li><li><a href="https://arxiv.org/abs/2402.01825">分形模式可能揭示 Next-Token Prediction 的成功</a>：我们研究语言的分形结构，旨在为量化那些之前可能被怀疑但未被正式证明的属性提供精确的形式化方法。我们确定语言...</li><li><a href="https://x.com/HusseinHElafifi/status/1815107404046233979">来自 H (@HusseinHElafifi) 的推文</a>：这是我们通常对孩子要求的任务的改编版本：1. 逻辑推理 - 三段论求解 - 真值表完成 - 识别逻辑谬误 2. 数学推理 - 应用题...</li><li><a href="https://baai-agents.github.io/Cradle/">Cradle: 赋能 Foundation Agents 实现通用计算机控制</a>：未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow">思考，快与慢 - 维基百科</a>：未找到描述</li><li><a href="https://en.wikipedia.org/">维基百科，自由的百科全书</a>：未找到描述</li><li><a href="https://docs.google.com/document/d/1_AaViP_OrfQ8K256cHAYjS9pbKidCSgEXGaXMAyyWw4/edit">复杂推理分类法</a>：分类法改编自：学习的维度 (Marzano &amp; Pickering)；教育目标新分类法 [Marzano &amp; Kendall] ...处理情境 81 问题 ...阐明现象 &amp; E...</li><li><a href="https://tenor.com/view/eliezer-yudkowsky-george-hotz-ai-alignment-ai-safety-gif-2309383719188990402">Eliezer Yudkowsky George Hotz GIF - Eliezer yudkowsky George hotz AI alignment - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/_saurabh/status/1763626711407816930">来自 Saurabh Srivastava (@_saurabh) 的推文</a>：超过 50% 被报道的 LLMs 推理能力可能不是真正的推理。我们如何评估在整个互联网上训练的模型？也就是说，对于一个已经...的东西，我们可以提出哪些新颖的问题？</li><li><a href="https://pastebin.com/fU2fHbBr">主任务列表 - Pastebin.com</a>：Pastebin.com 自 2002 年以来一直是排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://x.com/jd_pressman/status/1815109651220107309">来自 John David Pressman (@jd_pressman) 的推文</a>：@Teknium1 一些确保如果必要步骤之一发生变化，论证结论也会随之改变的过程。Chain of thought 提示词经常存在这样一个问题：如果你改变其中一个中间...</li><li><a href="https://x.com/max_paperclips/status/1783412470025187398?t=ymrIzKbiZ-6OfqcpJ-IEsQ&s=19">来自 Shannon Sands (@max_paperclips) 的推文</a>：我目前正在开发一种旨在供 LLMs 而非人类使用的脚本语言。通过做类似的事情，我能够让 Claude 摆脱“重写 Python”...</li><li><a href="https://gist.github.com/pipinstallyp/ba91773cba35a0e30c9dc26101e74dde">latex_draft.md</a>：GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://www.overleaf.com/latex/templates">模板 - 期刊、简历、演示文稿、报告等 - Overleaf，在线 LaTeX 编辑器</a>：一个易于使用的在线 LaTeX 编辑器。无需安装，实时协作，版本控制，数百个 LaTeX 模板等等。</li><li><a href="https://huggingface.co/datasets/jdpressman/retro-easy-prose-repair-diffs-v0.1">jdpressman/retro-easy-prose-repair-diffs-v0.1 · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://github.com/westoncb/latent-langs/blob/main/ConceptScript/samples/AlzheimersDisease.txt">latent-langs/ConceptScript/samples/AlzheimersDisease.txt 在 main 分支 · westoncb/latent-langs</a>：通过在 GitHub 上创建账号来为 westoncb/latent-langs 的开发做出贡献。</li><li><a href="https://github.com/NousResearch/Open-Reasoning-Tasks">GitHub - NousResearch/Open-Reasoning-Tasks: 一个面向 LLMs（及更多）的推理任务综合仓库</a>：一个面向 LLMs（及更多）的推理任务综合仓库 - NousResearch/Open-Reasoning-Tasks</li><li><a href="https://github.com/probcomp/hfppl">GitHub - probcomp/hfppl: 使用 HuggingFace 语言模型进行概率编程</a>：使用 HuggingFace 语言模型进行概率编程 - probcomp/hfppl</li><li><a href="https://x.com/VivekGRamaswamy/status/1815093862748303795">来自 Vivek Ramaswamy (@VivekGRamaswamy) 的推文</a>：预测未来的最佳方式：只需关注激励机制。令人震惊的是，你可以预测得多么精准，甚至精确到具体的时间点。</li><li><a href="h">

<li><a href="https://huggingface.co/datasets/jdpressman/retroinstruct-mix-v0.2">jdpressman/retroinstruct-mix-v0.2 · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks">lm-evaluation-harness/lm_eval/tasks at main · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/teacherpeterpan/Logic-LLM">GitHub - teacherpeterpan/Logic-LLM: The project page for &quot;LOGIC-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning&quot;</a>: “LOGIC-LM：通过符号求解器增强大语言模型以实现可靠的逻辑推理”的项目页面 - teacherpeterpan/Logic-LLM</li><li><a href="https://en.wikipedia.org/wiki/Automated_theorem_proving">Automated theorem proving - Wikipedia</a>: 未找到描述</li><li><a href="https://github.com/MineDojo/Voyager">GitHub - MineDojo/Voyager: An Open-Ended Embodied Agent with Large Language Models</a>: 一个基于大语言模型的开放式具身 Agent - MineDojo/Voyager</li><li><a href="https://github.com/METR/public-tasks">GitHub - METR/public-tasks</a>: 通过在 GitHub 上创建账户来为 METR/public-tasks 的开发做出贡献。</li><li><a href="https://pastebin.com/Ki4kq4GX">• Friedrich August Kekulé: The renowned German chemist, Kekulé was responsible f - Pastebin.com</a>: Pastebin.com 是自 2002 年以来排名第一的文本粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://huggingface.co/datasets/jdpressman/retro-weave-eval-rubrics-v0.1">jdpressman/retro-weave-eval-rubrics-v0.1 · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/openai/evals/tree/main/evals/registry/evals">evals/evals/registry/evals at main · openai/evals</a>: Evals 是一个用于评估 LLM 和 LLM 系统的框架，也是一个开源的基准测试注册表。 - openai/evals</li><li><a href="https://thangs.com/explore">Explore 3D models on Thangs</a>: 探索来自顶级设计师的 3D 模型</li><li><a href="https://en.wikipedia.org/wiki/First-order_logic">First-order logic - Wikipedia</a>: 未找到描述</li><li><a href="https://plato.stanford.edu/entries/ryle/#Car">
Gilbert Ryle (Stanford Encyclopedia of Philosophy)
</a>: 未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Martin_Heidegger">Martin Heidegger - Wikipedia</a>: 未找到描述</li><li><a href="https://doi.org/10.3758/s13423-013-0467-3">A taxonomy of inductive problems - Psychonomic Bulletin &amp; Review</a>: 关于物体、特征、类别和关系的归纳推理已经研究多年，但很少有人尝试绘制人类能够处理的归纳问题范围...
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1263935133032124579)** (959 条消息🔥🔥🔥): 

> - `LLM fine-tuning`
> - `GPU and hardware capabilities` (GPU 与硬件能力)
> - `Model deployment issues` (模型部署问题)
> - `Whisper Model for transcriptions` (用于转录的 Whisper 模型)
> - `LLM code and architecture troubleshooting` (LLM 代码与架构故障排除)


- **针对不同任务微调 LLM 的挑战**：用户讨论了在不同数据集和架构上微调大语言模型 (LLM) 的问题，包括对有限训练时间和 GPU 能力的沮丧。
- **关于 GPU 能力和配置的深入讨论**：社区成员分享了他们在训练和部署 AI 模型时使用各种 GPU（包括 RTX 3060, RTX 4070 和 H100）的经验，并强调了性能差异。
- **探索模型部署的可用资源**：一位用户询问了如何自动执行带有时间戳的音频文件说话人日志 (speaker diarization) 和 Whisper 转录，并提到之前在 RTX 4070 上使用过 Whisper Large v3。
- **排除 LLM 代码和架构故障**：讨论了语言模型的详细故障排除，重点关注模型大小、训练时长以及优化过程中针对的特定层等问题。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/HochreiterSepp/status/1813189814373462295">来自 Sepp Hochreiter (@HochreiterSepp) 的推文</a>: xLSTM 在时间序列预测方面表现出色。“我们的 xLSTMTime 模型在与最先进的基于 Transformer 的模型以及其他最近提出的时间序列模型对比中展现了卓越的性能...”</li><li><a href="https://x.com/alpindale/status/1814814551449244058?s=12">来自 Alpin (@AlpinDale) 的推文</a>: 已确认有 8B、70B 和 405B。前两个是从 405B 蒸馏（distilled）出来的。128k（十进制为 131k）上下文。405B 画不出独角兽。指令微调（Instruct tune）可能进行了安全对齐。架构...</li><li><a href="https://x.com/alpindale/status/1814717595754377562?s=46">来自 Alpin (@AlpinDale) 的推文</a>: 似乎 HF 的某人忘记及时将此仓库设为私有，Google 索引了它：sllhf/Meta-Llama-3.1-405B-Instruct-FP8。405B 是 Llama 3.1？非常有趣。我想知道他们是否只会发布...</li><li><a href="https://doc.rust-lang.org/book/#the-rust-programming-language">Rust 程序设计语言 - The Rust Programming Language</a>: 未找到描述</li><li><a href="https://x.com/teknium1/status/1815103461404606561?s=46">来自 Teknium (e/λ) (@Teknium1) 的推文</a>: 有人想合作制定一份“推理（reasoning）”任务类型的总表，以便大家参考并专注于提高推理能力的目标任务吗？请告诉我！</li><li><a href="https://x.com/teknium1/status/1815114033399558237?s=46">来自 Teknium (e/λ) (@Teknium1) 的推文</a>: 更新：我们在 @NousResearch Discord 为这个项目开设了一个频道，请加入 @ https://discord.gg/NousResearch。今天晚些时候将创建一个 GitHub 仓库来汇总想法。</li><li><a href="https://www.meetup.com/data-scientist-meetup-in-seoul/events/302347555/">使用 Upstage 预训练 LLM - [在线学习小组]📚🍰🤖🌏🤼‍♂️, 2024年7月28日，周日，晚上 9:00 | Meetup</a>: 这是一个面向 AI/ML 从业者/学习者的学习小组。🤖📚 我们将回顾并讨论从 Upstage 的“预训练 LLM”短课程中学到的内容。</li><li><a href="https://arxiv.org/abs/2407.10240">xLSTMTime : 使用 xLSTM 进行长期时间序列预测</a>: 近年来，基于 Transformer 的模型在多变量长期时间序列预测（LTSF）中占据了重要地位，尽管面临高昂的...</li><li><a href="https://huggingface.co/starsnatched/MemeGPT">starsnatched/MemeGPT · Hugging Face</a>: 未找到描述</li><li><a href="https://preview.devin.ai/">Devin (开发者)</a>: 你可靠的 AI 软件工程师</li><li><a href="https://huggingface.co/spaces/Xenova/whisper-speaker-diarization">Whisper 说话人日志 (Speaker Diarization) - Xenova 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/ylacombe/parler-tts-mini-v1">ylacombe/parler-tts-mini-v1 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/meta-releases/Meta-Llama-3.1-405B-Instruct">meta-releases/Meta-Llama-3.1-405B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://youtu.be/PmhsIHl27ZM">45 秒内了解前 5 名最快的编程语言</a>: 基于来自 Google、Wikipedia 和其他平台的研究，我选择了与汇编语言相比最快的前五种编程语言。我希望...</li><li><a href="https://huggingface.co/v2ray/Llama-3.1-405B">v2ray/Llama-3.1-405B · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/bugs-bunny-no-no-bunny-bugs-gif-7909500831201365932">Bugs Bunny No GIF - Bugs bunny no No Bunny - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/sad-upset-violin-sponge-bob-mr-crab-gif-3466351">Sad Violin GIF - Sad Upset Violin - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/blog/nroggendorff/train-with-llama-architecture">从零开始训练 Llama 模型</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/nroggendorff/colors/commits/main">提交记录 · nroggendorff/colors</a>: 未找到描述</li><li><a href="https://tenor.com/view/batman-mad-angry-tell-me-interogating-gif-17869813">Batman Mad GIF - Batman Mad Angry - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/biggest-boy-family-guy-chris-griffin-dancing-gif-17316116">Biggest Boy Family Guy GIF - Biggest Boy Family Guy Chris Griffin - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/stfu-kanye-kanye-west-shut-up-dance-gif-23839788">Stfu Kanye GIF - Stfu Kanye Kanye West - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/what-gif-21384529">What GIF - What - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/good-morning-gif-11437316614611695342">早安 GIF - Good morning - 发现并分享 GIF</a>

</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/subida-gif-18379274">Subida GIF - Subida - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/spaces?search=Whi">Spaces - Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/transformers/main/en/chat_templating">Templates for Chat Models</a>: 未找到描述</li><li><a href="https://tenor.com/view/patrick-stupid-drooling-patrick-star-spongebob-gif-12221001666588210206">Patrick Stupid GIF - Patrick Stupid Drooling - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/scared-dog-shivering-dog-dog-shaking-meme-gif-26566244">Scared Dog Shivering Dog GIF - Scared Dog Shivering Dog Dog Shaking Meme - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/muslehal/xLSTMTime">GitHub - muslehal/xLSTMTime: 用于时间序列预测的 xLSTMTime</a>: 用于时间序列预测的 xLSTMTime。通过在 GitHub 上创建账号来为 muslehal/xLSTMTime 的开发做出贡献。</li><li><a href="https://huggingface.co/spaces/nroggendorff/zelda-lora">Zelda Diffusion XL - nroggendorff 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/nroggendorff/animexl">Anime Diffusion XL - nroggendorff 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://github.com/lucidrains/e2-tts-pytorch/blob/04b9c1cdacff4b459187e101344e718ed5b7142c/e2_tts_pytorch/e2_tts.py#L425">e2-tts-pytorch/e2_tts_pytorch/e2_tts.py at 04b9c1cdacff4b459187e101344e718ed5b7142c · lucidrains/e2-tts-pytorch</a>: E2-TTS 的 Pytorch 实现，“极其简单的完全非自回归零样本 TTS” - lucidrains/e2-tts-pytorch</li><li><a href="https://tenor.com/view/mark-zuckerberg-gif-14169217">Mark Zuckerberg GIF - Mark Zuckerberg - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.techpowerup.com/gpu-specs/voodoo3-3000-agp.c3555#:~:text=it%20might%20not%20be%20able%20to%20run%20all%20the%20latest%20games)">3dfx Voodoo3 3000 AGP 规格</a>: 3dfx Avenger, 166 MHz, 1 Pixel Shaders, 0 Vertex Shaders, 2 TMUs, 1 ROPs, 16 MB SDR, 166 MHz, 128 bit</li><li><a href="https://huggingface.co/docs/api-inference/quicktour">Overview</a>: 未找到描述</li><li><a href="https://tenor.com/view/wizard-dance-ena-gif-27696814">Wizard Dance GIF - Wizard Dance Ena - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/lindsey-stirling-lindsey-stirling-cute-adorable-gif-19359953">Lindsey Stirling Cute GIF - Lindsey Stirling Lindsey Stirling - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://discuss.huggingface.co/t/metrics-for-training-set-in-trainer/2461/5">Trainer 中训练集的指标</a>: 我通过添加一个自定义回调来实现这一点，该回调在每个回调结束时使用 train_dataset 调用 evaluate() 方法。 class CustomCallback(TrainerCallback):          def __init__(self, trainer) -...</li><li><a href="https://github.com/OpenDevin/OpenDevin">GitHub - OpenDevin/OpenDevin: 🐚 OpenDevin: 少写代码，多做产出</a>: 🐚 OpenDevin: 少写代码，多做产出。通过在 GitHub 上创建账号来为 OpenDevin/OpenDevin 的开发做出贡献。</li><li><a href="https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/">TLDR: 使用 LLM 构建的一年 &#8211; D-Squared</a>: 未找到描述</li><li><a href="https://github.com/ToonCrafter/ToonCrafter">GitHub - ToonCrafter/ToonCrafter: 关于生成式卡通插值的研究论文</a>: 关于生成式卡通插值的研究论文 - ToonCrafter/ToonCrafter</li><li><a href="https://huggingface.co/spac">Spac (Stéphan Pacchiano)</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/MachineLearning/s/muGnbfl6yf">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://www.techpowerup.com/gpu-specs/geforce-rtx-5090.c4216">NVIDIA GeForce RTX 5090 规格</a>: NVIDIA GB202, 2520 MHz, 20480 Cores, 640 TMUs, 192 ROPs, 28672 MB GDDR7, 2500 MHz, 448 bit</li><li><a href="https://www.techpowerup.com/gpu-specs/geforce-rtx-5060.c4219">NVIDIA GeForce RTX 5060 规格</a>: NVIDIA GB206, 2520 MHz, 4608 Cores, 144 TMUs, 48 ROPs, 8192 MB GDDR7, 2500 MHz, 128 bit
</li>
</ul>

</div>

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1263987039850729492)** (4 条消息): 

> - `Knowledge Graphs`
> - `News Reading Experience`
> - `Hugging Face Model Kwargs`
> - `Speaker Diarization & Whisper Transcription` 


- **探索 Knowledge Graphs**：一位用户提供了关于 **Knowledge Graphs** 的帮助，并提到了他们在构建和使用这些图谱方面的过往经验。
   - *“它们非常有趣且有用！”* 是关于使用 Knowledge Graphs 的心得分享。
- **改善 News Reading Experience**：另一位用户表达了对 **改善新闻阅读体验和 Sentiment Analysis** 的兴趣，并称赞了 Hugging Face 的工具：*“我不是程序员，但我真的很喜欢 Hugging Face 到目前为止所构建的一切！”*
- **在 Hugging Face 上查找 Model Kwargs**：一位用户询问在 Hugging Face 上使用模型时，应该去哪里查找 **model_kwargs**。
   - 他们分享了一个示例代码片段，其中将 `{'temperature': 1.0, 'top_k': 20, 'max_length': 128}` 作为 model_kwargs 使用。
- **自动化 Speaker Diarization 和 Transcription**：一位成员征求关于自动化 **Speaker Diarization、Whisper Transcriptions 和 Timestamps** 流水线的建议。
   - 他们愿意学习如何对输出进行数据库管理，并正在寻找推荐的模型或开源仓库。


  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1264459140256763965)** (8 条消息🔥): 

> - `Apple Intelligence`
> - `LoRA fine-tuning`
> - `arXiv 上的 AI 论文`
> - `AI 对世界的变革`
> - `免费在线课程` 


- **Apple 的单模型能力令人印象深刻**：一位热心成员分享了关于 Apple Intelligence 如何使用单个模型处理多项任务的见解，强调了其天才的设计和多功能性。
   - 他们建议读者查看 [LoRA fine-tuning 的 YouTube 视频教程](https://www.youtube.com/watch?v=8N9L-XK1eEU) 以及一篇关于 [arXiv 上 LoRA 的详细论文](https://arxiv.org/abs/2106.09685)。
- **arXiv 上有趣的 AI 论文**：一位成员在 arXiv 上发现了一篇富有见地的 [AI 论文](https://arxiv.org/pdf/2407.08683v1)，并分享给社区供进一步阅读。
   - *未提供额外评论。*
- **AI 如何改变世界**：一位成员分享了来自 [Brookings](https://www.brookings.edu/articles/how-artificial-intelligence-is-transforming-the-world/) 的一篇综合文章，讨论了 AI 对各个领域的影响并提出了政策建议。
   - 文章强调了 AI 在改善决策和改变日常生活方面的作用。
- **带证书的免费在线课程**：推荐了几门免费在线课程，包括 Udacity 上的 [Python 编程入门](https://www.udacity.com/course/introduction-to-python--ud1110)、[产品设计](https://www.udacity.com/course/product-design--ud509) 和 [数据分析](https://www.udacity.com/course/intro-to-data-analysis--ud170)。
   - 这些课程涵盖了 Python、产品验证以及使用流行的 Python 库进行数据分析等基本技能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.brookings.edu/articles/how-artificial-intelligence-is-transforming-the-world/">人工智能如何改变世界 | Brookings</a>: Darrell West 和 John Allen 探讨了开发人工智能技术的社会和政治层面。</li><li><a href="https://www.udacity.com/course/introduction-to-python--ud1110">免费 Python 入门课程 | Udacity</a>: 在线学习并通过编程、数据科学、人工智能、数字营销等课程提升您的职业生涯。获得热门技术技能。立即加入！</li><li><a href="https://www.udacity.com/course/product-design--ud509">产品设计 | Udacity</a>: 在线学习并通过编程、数据科学、人工智能、数字营销等课程提升您的职业生涯。获得热门技术技能。立即加入！</li><li><a href="https://www.udacity.com/course/intro-to-data-analysis--ud170">数据分析入门 | 数据分析 | Udacity</a>: 在线学习并通过编程、数据科学、人工智能、数字营销等课程提升您的职业生涯。获得热门技术技能。立即加入！</li><li><a href="https://www.ft.com/content/0b210299-4659-4055-8d81-5a493e85432f?utm_source=superhuman&utm_medium=newsletter&utm_campaign=the-godmother-of-ai-is-building-a-new-startup">“AI 教母”李飞飞在 4 个月内创办了价值 10 亿美元的初创公司</a>: 未找到描述</li><li><a href="https://medium.com/@teendifferent/the-secret-behind-apple-intelligence-one-model-endless-possibilities-833ad1b989af)">未找到标题</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=8N9L-XK1eEU)">如何使用 LoRA 微调模型（分步指南）</a>: LoRA 是一个天才的想法。到本视频结束时，你将了解关于 LoRA 工作原理的所有重要信息。我将向你展示一个微调模型的示例...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1263947778686718002)** (49 条消息🔥): 

> - `Hermes 2.5`
> - `Model Merging`
> - `Open Empathic`
> - `Gary4live`
> - `SmolLM`

- **Gary4live：面向 MusicGen 的 Ableton 插件**：在一段 [speedrun 视频](https://x.com/thepatch_kev/status/1814386138972598446?s=46)中展示了名为 **Gary4live** 的新 Ableton 插件，演示了其在音乐生成方面的能力。
   - 该插件已被证明可以协同工作，在实时音乐即兴演奏（jam sessions）中与他人共同创作歌曲。
- **SmolLM Arena 发布**：一个名为 SmolLM Arena 的 [新项目](https://huggingface.co/spaces/as-cle-bert/smolLM-arena) 已发布，允许用户比较各种小型语言模型（Small Language Models，参数量 <1.7B）。
   - 该竞技场（arena）具有聊天机器人界面，运行速度更快，并包含使用说明以提供更流畅的用户体验。
- **Manifold Research 研讨会**：Manifold Research Group 正在举办一场 [社区研究研讨会 (Community Research Call)](https://www.manifoldrg.com/community-research-call/)，涵盖通用多模态（Generalist Multimodality）、LLM Agents 和机器人技术（Robotics）等主题。
   - 此次会议旨在提供项目更新并促进问答环节，以吸引更多社区成员参与他们的工作。
- **端侧 LLMs 工作坊**：由 Enrico Rampazzo 主持的 [YouTube 工作坊](https://www.youtube.com/watch?v=zuNMvL4dtvM) 讨论了端侧（on-device）大语言模型（LLMs）的未来及其能力。
   - 该环节探讨了端侧 LLMs 如何为移动应用和 AI 部署带来革命性变化。
- **Gradio 的 Rust 客户端库**：发布了一个新的 [Gradio](https://github.com/JacobLinCool/gradio-rs) Rust 客户端库，旨在简化与各种 Gradio spaces 的集成。
   - 该库支持 `hf-audio/whisper-large-v3` 和 `stabilityai/stable-diffusion-3-medium` 等模型，开发者正在寻求社区的反馈和贡献。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/thepatch_kev/status/1814386138972598446?s=46">来自 thecollabagepatch (@thepatch_kev) 的推文</a>：又一个关于开源 Ableton 插件 gary4live 的 speedrun，它可以和你一起即兴演奏。这次是和好友 tom's beat 的合作。跳到 2:56 听听看 @_buildspace @_nightsweekends @ma...</li><li><a href="https://huggingface.co/spaces/ptx0/PixArt-900M-EDiffi">PixArt 900M 1024px E-Diffi (Mixture-of-Experts) - ptx0 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/KingNish/OpenCHAT-mini">OpenCHAT Mini - KingNish 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://www.manifoldrg.com/community-research-call/">社区研究电话会议 (Community Research Calls)</a>：社区研究电话会议是我们专门为广大 Manifold 社区提供组织级和项目级更新的会议。如果您有兴趣参加这些更新会议...</li><li><a href="https://huggingface.co/spaces/as-cle-bert/smolLM-arena">SmolLM Arena - as-cle-bert 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/gokaygokay/UltraPixel">UltraPixel - gokaygokay 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://x.com/AnindyadeepS/status/1815284584332099840">来自 Anindya (@AnindyadeepS) 的推文</a>：周一快乐。我知道我加入这个游戏有点晚了，但今天，我发布了关于 MakeMore 系列博客的第一篇。一段时间以来，我一直在学习 Andrej Karpathy 的...</li><li><a href="https://huggingface.co/spaces/KingNish/Image-Gen-Pro">Image Gen Pro - KingNish 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=zuNMvL4dtvM.">设备端 LLMs：口袋里的 AI 未来 | 超越 ChatGPT</a>：🚀 深入探索 AI 的未来：设备端大语言模型 (LLMs) 🧠📱 在这场前沿会议中，我们探索了设备端大模型的革命性世界...</li><li><a href="https://huggingface.co/blog/nroggendorff/create-diffusers-dataset">为 Stable Diffusion 微调创建 Diffusers 兼容的数据集</a>：未找到描述</li><li><a href="https://youtu.be/Gscelu22FWI">设计 TikTok 的推荐系统 | ML 系统设计 | #systemdesign</a>：你知道为什么 TikTok 的推荐算法这么好吗？在视频中，我们设计了 TikTok 的推荐系统。视频涵盖了机器学习方面...</li><li><a href="https://www.udacity.com/course/introduction-to-python--ud1110">免费 Python 入门课程 | Udacity</a>：通过编程、数据科学、人工智能、数字营销等课程在线学习并提升您的职业生涯。获得热门技术技能。立即加入！</li><li><a href="https://www.udacity.com/course/product-design--ud509">产品设计 | Udacity</a>：通过编程、数据科学、人工智能、数字营销等课程在线学习并提升您的职业生涯。获得热门技术技能。立即加入！</li><li><a href="https://www.udacity.com/course/intro-to-data-analysis--ud170">数据分析入门 | 数据分析 | Udacity</a>：通过编程、数据科学、人工智能、数字营销等课程在线学习并提升您的职业生涯。获得热门技术技能。立即加入！</li><li><a href="https://github.com/cappuch/mfetch">GitHub - cappuch/mfetch：纯 Python 获取实用工具（类似 neofetch）</a>：纯 Python 获取实用工具（类似 neofetch）。通过在 GitHub 上创建账号为 cappuch/mfetch 的开发做出贡献。</li><li><a href="https://medium.com/@teendifferent/the-secret-behind-apple-intelligence-one-model-endless-possibilities-833ad1b989af)">未找到标题</a>：未找到描述</li><li><a href="https://github.com/aaryadevchandra/seq2seq-machine-translation">GitHub - aaryadevchandra/seq2seq-machine-translation：使用原生 sequence-to-sequence 模型的英德语言翻译器</a>：使用原生 sequence-to-sequence 模型的英德语言翻译器 - aaryadevchandra/seq2seq-machine-translation</li><li><a href="https://github.com/ParagEkbote/Hugging_Face_Docs">GitHub - ParagEkbote/Hugging_Face_Docs：Hugging Face 库的文档支持。</a>：Hugging Face 库的文档支持。通过在 GitHub 上创建账号为 ParagEkbote/Hugging_Face_Docs 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e9bhs8/rd2md_the_missing_link_between_reddit_and_your/?utm_source=share&utm_medium=mweb3x&utm_name=mweb3xcss&utm_term=1&utm_content=share_button">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://github.com/JacobLinCool/gradio-rs">GitHub - JacobLinCool/gradio-rs：Rust 版本的 Gradio 客户端。</a>：Rust 版本的 Gradio 客户端。通过在 GitHub 上创建账号为 JacobLinCool/gradio-rs 的开发做出贡献。</li><li><a href="https://isari.ai">Isari - AI 增强的 Wo</a>

rkforce</a>: 未找到描述</li><li><a href="https://github.com/Cycls/examples/blob/main/openai.py">examples/openai.py at main · Cycls/examples</a>: Cycls SDK 示例应用。通过在 GitHub 上创建账号来为 Cycls/examples 的开发做出贡献。</li><li><a href="https://huggingface.co/spaces/gokaygokay/Inspyrenet-Rembg">Inspyrenet Remove Background - a Hugging Face Space by gokaygokay</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/gokaygokay/360PanoImage">360PanoImage - a Hugging Face Space by gokaygokay</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/gokaygokay/TileUpscalerV2">Tile Upscaler V2 - a Hugging Face Space by gokaygokay</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/gokaygokay/AuraFlow-with-Captioner">AuraFlow with Captioner - a Hugging Face Space by gokaygokay</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/gokaygokay/PonyRealism">Pony Realism ++ - a Hugging Face Space by gokaygokay</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1263940291623387237)** (2 条消息): 

> - `活动创建`
> - `图表反馈` 


- **活动创建通知**：<@607608455976583206> 通知了关于创建一个 [活动](https://discord.com/events/879548962464493619/1263939506281779342) 的消息，并寻求反馈或修改建议。
- **对图表的正面反馈**：成员 *lunarflu* 称赞了该活动，特别提到了图表：*'太棒了，我也很喜欢这个图表！'*。


  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1264778383703674912)** (1 条消息): 

> - `SD3 训练 Bug`
> - `Diffusers 仓库` 


- **修复 SD3 训练 Bug**：社区的共同努力促成了对 [Diffusers 仓库](https://github.com/huggingface/diffusers/pull/8917/) 中 **SD3 训练脚本** Bug 的发现和修复。
   - 该修复解决了 issue #8887 和 #8708，并增加了一个选项来控制模型输出的预调节（pre-conditioning）行为。
- **Diffusers 仓库更新**：[Diffusers 仓库](https://github.com/huggingface/diffusers/pull/8917/) 已更新以修复训练脚本 Bug。
   - 社区贡献极大地帮助了识别和解决 SD3 训练过程中的问题。



**提到的链接**：<a href="https://github.com/huggingface/diffusers/pull/8917/.">[Training] SD3 training fixes by sayakpaul · Pull Request #8917 · huggingface/diffusers</a>：此 PR 做了什么？修复了 #8887 和 #8708。此外，它增加了一个选项来控制模型输出的预调节行为。许多人报告说，对于 rectified-flows，我们...

  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1264494540166004786)** (19 条消息🔥): 

> - `Inception 与 ViT 的混合模型`
> - `拼字游戏 (Scrabble) 棋盘棋子检测`
> - `二值分割项目` 


- **构建 Inception 与 ViT 的混合模型**：一位成员询问如何将 **Inception 网络**与 **ViT** 集成用于图像分类，另一位成员建议使用 [timm 中的混合 ViT 实现](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer_hybrid.py)，并详细说明了将特征图展平（flattening）以输入 ViT 的过程。
- **拼字游戏棋盘棋子检测的挑战**：一位成员描述了尝试使用 CV 和 ML 检测拼字游戏棋子的过程，但面临准确性问题，其他人建议参考包括 [scrabble-opencv GitHub 项目](https://github.com/jheidel/scrabble-opencv) 在内的方法来获取灵感。
   - 建议包括使用 CVAT 等工具标注棋盘，并应用考虑固定棋盘尺寸和摄像机角度的硬编码方法，尽管为了获得更好的准确性，ML 可能不可避免。
- **二值分割项目讨论**：发生了一次简短的互动，一位成员询问是否有人有二值分割项目的经验，并得到了另一位使用 **UNet** 开展过相关工作的成员的回应。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://app.cvat.ai/.">Computer Vision Annotation Tool</a>: 无描述</li><li><a href="https://github.com/jheidel/scrabble-opencv/tree/master">GitHub - jheidel/scrabble-opencv: 一个使用 OpenCV 的有趣拼字游戏计分器</a>: 一个使用 OpenCV 的有趣拼字游戏计分器。通过创建账号为 jheidel/scrabble-opencv 的开发做出贡献。</li><li><a href="https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer_hybrid.py">pytorch-image-models/timm/models/vision_transformer_hybrid.py at main · huggingface/pytorch-image-models</a>: 最大的 PyTorch 图像编码器/骨干网络集合。包括训练、评估、推理、导出脚本和预训练权重 —— ResNet, ResNeXT, EfficientNet, NFNet, Vision Transformer (V...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1264088496507846756)** (10 条消息🔥): 

> - `SQL RAG`
> - `日期提取`
> - `开源文本转 HTML/CSS 模型`
> - `微调语言模型`
> - `Transformer 微调中的指标` 


- **SQL RAG 的正面用例**：一位成员指出 **Perplexity AI** 成功实现了 SQL Retrieval-Augmented Generation (RAG) 以获得正确结果。
- **NLP 中日期提取的挑战**：一位用户在使用 dateparser 和名为 **Qwen2** 的 LLM 从“过去 6 个月内”等语句中提取正确的**开始和结束日期**时遇到困难。
- **寻求开源文本转 HTML/CSS 模型**：一位用户正在寻找**开源文本转 HTML/CSS 生成模型**，并请求社区推荐。
- **Transformer 微调期间指标相同**：一位用户在微调 **LILT 模型**时，召回率、F1、准确率和精确度指标出现了**相同的值**，并询问其他人是否遇到过同样的情况。
- **使用 LLM 进行高精度角色扮演**：一位初学者用户想要使用未标记的文本数据微调 **Llama3** 模型，以模仿一位哲学家。
   - 另一位用户建议尝试 **Retrieval-Augmented Generation**，但这位初学者对其在角色扮演中的准确性和连贯性持怀疑态度。


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1264286329118789682)** (4 条消息): 

> - `开源文本转 HTML/CSS 生成`
> - `SDv1.4 中的艺术风格`
> - `评估扩散模型`
> - `Stable Diffusion 论文编码器方法` 


- **寻找开源文本转 HTML/CSS 模型**：一位成员正在寻找**开源文本转 HTML/CSS 生成模型**，并向社区寻求指导。
- **对 SDv1.4 艺术风格的好奇**：一位成员询问了 **SDv1.4** 训练中使用的**艺术风格**列表。
- **评估扩散模型的技巧**：一位**Diffusion Models**和**图像生成**领域的新人正在寻求关于如何**评估其模型**的建议。
- **Stable Diffusion 论文编码器技术**：一位成员对原始 **Stable Diffusion 论文**中**潜空间 (latent space) 的构建**提出疑问，特别是该模型是在量化权重上训练的还是使用了其他方法。


  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1263962787798061117)** (213 条消息🔥🔥):

> - `Mojo Socket Implementation` (Mojo 套接字实现)
> - `Dual Stack Sockets Issue` (双栈套接字问题)
> - `Mojo Integration with Existing Libraries` (Mojo 与现有库的集成)
> - `Interoperability between Mojo and Python` (Mojo 与 Python 之间的互操作性)
> - `Production Readiness of Mojo` (Mojo 的生产就绪性)


- **Mojo 套接字实现讨论**：用户探讨了 Mojo 的各种套接字实现，特别关注平台特定差异带来的挑战，例如 Windows 套接字中的差异。
   - *darkmatter__* 指出 Rust 的套接字实现可能提供更清晰的参考，小组还强调了为未来协议适配 SCTP 的重要性。
- **关于 Mojo 中双栈套接字的辩论**：大家一致倾向于在新服务器套接字中使用双栈方法，即单个套接字同时接受 IPv4 和 IPv6 连接，这与 Python 的实现一致。
   - *darkmatter__* 建议在 Linux 上使用 `io_uring`，以便在高并发工作负载下统一处理传入连接。
- **Mojo 与外部库的集成**：社区讨论了在 Mojo 中使用各种外部库的情况，包括 *darkmatter__* 正在开发的类似 Zig 的 `translate-c` 解决方案，以在完善的 Interop 可用之前填补空白。
   - 还提到了调用 DPDK 进行网络操作，以及大型代码库中依赖关系的潜在复杂性。
- **在 Mojo 中处理 Python 类**：用户尝试将 Python 类集成到 Mojo 中以实现自定义数据类型，但在错误处理和 Python 动态类型的别名（aliasing）方面面临挑战。
   - 讨论强调了 Mojo 在完全支持 Python 类方面的局限性，并探索了可能的变通方案，包括采用 Mojo 内置结构。
- **社区对 Mojo 生产就绪性的看法**：关于 Mojo 是否已具备生产就绪性，意见不一。*darkmatter__* 建议在语言发布稳定的 1.0 版本之前保持谨慎。
   - *darinsimmons* 提供了更长远的观点，认为在几年内实现稳定的生产使用可能是现实的，具体取决于特定的用例和所需功能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tigerbeetle.com/blog/a-friendly-abstraction-over-iouring-and-kqueue">A Programmer-Friendly I/O Abstraction Over io_uring and kqueue</a>：为未来 30 年联机事务处理（OLTP）提供动力的金融交易数据库。</li><li><a href="https://docs.kernel.org/networking/tls.html">Kernel TLS &#8212; The Linux Kernel  documentation</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v">YouTube</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=5znybwzUZog)">Asynchronous and Direct IO for PostgreSQL on FreeBSD Thomas Munro</a>：完整描述见 https://www.bsdcan.org/events/bsdcan_2022/schedule/session/90-asynchronous-and-direct-io-for-postgresql-on-freebsd/</li><li><a href="https://github.com/Legrandin/pycryptodome">GitHub - Legrandin/pycryptodome: A self-contained cryptographic library for Python</a>：一个独立的 Python 加密库。可以通过在 GitHub 上创建账户为 Legrandin/pycryptodome 的开发做出贡献。</li><li><a href="https://man7.org/linux/man-pages/man7/sctp.7.html">sctp(7) - Linux manual page</a>：未找到描述</li><li><a href="https://github.com/modularml/mojo/issues/3262">Ability to Link to C Libraries · Issue #3262 · modularml/mojo</a>：审查 Mojo 的优先级。我已阅读路线图和优先级，并认为此请求符合优先级。你的请求是什么？理想情况下应该有类似 @link(…) 装饰器的东西...</li><li><a href="https://github.com/dmitry-salin/io_uring">GitHub - dmitry-salin/io_uring: The io_uring library for Mojo</a>：适用于 Mojo 的 io_uring 库。可以通过在 GitHub 上创建账户为 dmitry-salin/io_uring 的开发做出贡献。</li><li><a href="https://github.com/mzaks/compact-dict">GitHub - mzaks/compact-dict: A fast and compact Dict implementation in Mojo 🔥</a>：Mojo 中一个快速且紧凑的 Dict 实现 🔥。可以通过在 GitHub 上创建账户为 mzaks/compact-dict 的开发做出贡献。</li><li><a href="https://modul.ar/community-meeting-doc.">[Public] Mojo Community Meeting</a>：Mojo 社区会议。文档链接：https://modul.ar/community-meeting-doc。这是一个公开文档；欢迎所有人查看并发表评论/建议。所有会议参与者必须遵守...</li><li><a href="https://github.com/bytecodealliance/rustix/tree/main/src/net">rustix/src/net at main · bytecodealliance/rustix</a>：POSIX 风格 API 的安全 Rust 绑定。可以通过在 GitHub 上创建账户为 bytecodealliance/rustix 的开发做出贡献。</li><li><a href="https://github.com/rust-lang/rfcs/blob/master/text/3128-io-safety.md">rfcs/text/3128-io-safety.md at master · rust-lang/rfcs</a>：Rust 变更的 RFC。可以通过在 GitHub 上创建账户为 rust-lang/rfcs 的开发做出贡献。
</li>
</ul>

</div>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: 来自 *Modular*:
<https://twitter.com/Modular/status/1815463417391837596>
  

---


### **Modular (Mojo 🔥) ▷ #[📺︱youtube](https://discord.com/channels/1087530497313357884/1098713700719919234/1264995931292700685)** (1 messages): 

> - `Mojo 🔥 Community Meeting #4`
> - `Flat Buffers`
> - `Forge Tools`
> - `Mojo 🔥 Standard Library`
> - `Mojo 🔥 Gen` 


- **Mojo 🔥 Community Meeting #4 亮点**: Modular 刚刚发布了一个新的 [YouTube 视频](https://www.youtube.com/watch?v=_QVs626Vn2k)，题为“**Mojo 🔥 Community Meeting #4**”，内容涉及 **Flat Buffers**、**Forge Tools** 以及扩展 **Mojo 🔥 标准库**的讨论。
   - 会议涵盖了内存高效的序列化、Mojo 🔥 生态系统的新工具以及 Mojo 🔥 Gen 的更新等主题。
- **Flat Buffers：高效序列化**: Mojo 🔥 Community Meeting #4 的一个核心亮点是关于用于内存高效序列化的 */Flat Buffers/* 的讨论。
   - 与会者探讨了它们在优化 Mojo 🔥 框架内数据处理方面的应用。



**提到的链接**: <a href="https://www.youtube.com/watch?v=_QVs626Vn2k">Mojo 🔥 Community Meeting #4</a>: Mojo 社区会议 #4 录像 🫓 Flat Buffers: 内存高效序列化 ⚒️ Forge Tools: 扩展 Mojo 🔥 标准库 🔄 Mojo 🔥 Gen...

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1263990875340673158)** (185 messages🔥🔥): 

> - `Anti-Pattern Discussions`
> - `OpenSSL and Mojo Projects`
> - `Newton's Method for Float Literals`
> - `Future of Mojo's Async Scheduler`
> - `CPU Performance Comparisons` 


- **反模式引发热烈辩论**: 成员们幽默地讨论了一个针对条件一致性（conditional conformance）失效的“反模式”权宜之计，引发了从赞同到调侃的各种反应。
- **OpenSSL 集成面临挑战**: 讨论揭示了 OpenSSL 的庞大体积以及将其与 Mojo 项目集成的障碍，强调了密码学实现的复杂性。
- **牛顿迭代法（Newton's Method）辅助工具提案**: 一位成员分享了在 Mojo 中针对浮点字面量实现牛顿迭代法的方案，引发了关于闭包和用于数值方程求解的捕获关键字的详细讨论。
- **Mojo 异步调度器的未来引发争论**: 成员们就 Mojo 是否应该在其 stdlib 中包含默认的异步调度器展开辩论，支持者认为这能带来更好的生态系统互操作性，反对者则担心这会限制替代方案的发展。
- **AMD vs Intel CPU 性能对比**: 社区对比了 AMD 和 Intel CPU 在各种任务中的表现，重点关注了 Intel 最近的稳定性问题以及 AMD L3 缓存针对特定用例的性能优势。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://learn.microsoft.com/en-us/windows/win32/api/ioringapi/">ioringapi - Win32 apps</a>: 提供用于创建和管理 I/O rings 的 API。</li><li><a href="https://www.youtube.com/watch?v=QzHcrbT5D_Y">Intel has a Pretty Big Problem</a>: Intel 的 13900k 和 14900k 正在以惊人的速度崩溃？为什么没人讨论这件事，Intel 的解决方案是什么？论坛帖子在此：https://for...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1264897624964599839)** (1 messages): 

> - `Matrix Multiplication in Mojo`
> - `Comparing Mojo to Numpy Performance` 


- **Mojo 中的矩阵乘法示例**: 一位用户报告称尝试了文档中的 [矩阵乘法示例](https://docs.modular.com/mojo/notebooks/Matmul)，并发现它们非常出色，实现了从纯 Python 到优化实现的过渡。
   - 该 Notebook 从类似 Python 的实现开始，通过添加类型、向量化、分块（tiling）和并行化进行优化。
- **Mojo 实现比 Numpy 慢 4 倍**: 一位用户分享了一项实际评估，将 Mojo 的最终版本与 `numpy` 进行对比，发现其在自己的机器上仍然 **慢 4 倍**。
   - 为了寻求见解，他们询问是否自己的预期不切实际。



**提到的链接**: <a href="https://docs.modular.com/mojo/notebooks/Matmul">Matrix multiplication in Mojo | Modular Docs</a>: 了解如何利用 Mojo 的各种功能编写高性能的 matmul。

  

---

### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1264154200078549094)** (5 messages): 

> - `nightly/max feed 可靠性`
> - `MAX 入门指南`
> - `对 MAX 的开源贡献`
> - `新贡献者指南` 


- **nightly/max feed 可靠性问题**：一名成员报告了 `nightly/max` feed 无法与 `nightly/mojo` 保持同步的问题，导致他们切换回了 `nightly/mojo`。
   - 另一名成员承认了这些问题，并提到正在努力尽快解决。
- **MAX 入门指南**：渴望学习的新成员被引导至 [MAX 文档](https://docs.modular.com/max/get-started) 和 [快速入门指南](https://docs.modular.com/max/install) 以开始使用。
   - 资源包括一篇关于 [如何贡献 Mojo 标准库的博客文章](https://www.modular.com/blog/how-to-contribute-to-mojo-standard-library-a-step-by-step-guide) 以及一份 [贡献指南](https://github.com/modularml/mojo/blob/nightly/CONTRIBUTING.md)。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.modular.com/max/get-started">MAX 入门指南 | Modular 文档</a>: 欢迎阅读 MAX 快速入门指南！</li><li><a href="https://www.modular.com/blog/how-to-contribute-to-mojo-standard-library-a-step-by-step-guide">Modular: 如何贡献 Mojo 标准库：分步指南</a>: 我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：如何贡献 Mojo 标准库：分步指南
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max-gpu](https://discord.com/channels/1087530497313357884/1212827673257316453/1264293038574801047)** (3 messages): 

> - `XLA 参与情况`
> - `Mojo 在 GPU 上的可用性` 


- **前 XLA 团队加入 Mojo**：一名成员分享道，他们团队中的许多人曾直接参与 Google 的 XLA 项目，他们正将所学经验和不同的哲学带到 Mojo 的 AI 基础设施开发中。
- **Mojo GPU 支持将于今夏推出**：当被问及 Mojo 的 GPU 可用性时，一名成员确认其应该会在今年夏天推出。


  

---

### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1264102837827141703)** (83 messages🔥🔥): 

> - `Nightly Mojo 编译器更新`
> - `LegacyPointer 和 DTypePointer 讨论`
> - `memcpy 函数的问题`
> - `Mojo API 的变更`
> - `社区互动与文档` 


- **Nightly Mojo 编译器 2024.7 版本发布**：新的 Nightly Mojo 编译器已发布，现已通过命令 `modular update nightly/mojo` 更新至 `2024.7.2205`。查看 [当前更新日志](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 和 [原始差异 (raw diff)](https://github.com/modularml/mojo/compare/b007d77a5832026553a9fccd2ea99ef21031336d...0d805c178526fc755edc6cd226af22b8da7341b6) 以了解详细变更。
- **LegacyPointer 和 DTypePointer 已弃用**：一场深入的讨论透露 `LegacyPointer` 和 `DTypePointer` 已被弃用，当前及未来的支持将集中在 `UnsafePointer` 上。
   - *用户表示担忧*：频繁的变更使得代码维护变得困难，并指出了旧函数不断被弃用的问题。
- **memcpy 重载的迁移问题**：`memcpy` 函数新增了三个重载，这引发了用户的困惑。**Carl Caulkett** 分享了一个涉及从 `DTypePointer` 过渡到 `UnsafePointer` 的*变通方法 (workaround)*。
- **旨在提高指针安全性的 Mojo API 变更**：社区讨论了旨在增强指针安全性的 API 变更，特别是从 `DTypePointer` 向 `UnsafePointer` 的转变。Daniel 确认使用 `UnsafePointer[Float64]` 将取代 `DTypePointer[DType.float64]`。
- **对更好文档的需求**：用户强调了需要更好的文档来管理快速演进的 API。社区承认，大量需要频繁更新的文档涉及技术债。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://sourcegraph.com/search">Sourcegraph</a>：未找到描述</li><li><a href="https://youtu.be/_QVs626Vn2k?t=3617">Mojo 🔥 社区会议 #4</a>：Mojo 社区会议 #4 的录音 🫓 Flat Buffers：内存高效的序列化 ⚒️ Forge Tools：扩展 Mojo 🔥 标准库 🔄 Mojo 🔥 Gen...</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/src/memory/__init__.mojo">modularml/mojo 在 nightly 分支的 mojo/stdlib/src/memory/__init__.mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/3126)">modularml/mojo 的 Issues</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/blob/16cc60dc3fbed1eff01a8f5fee94f97cf97cca33/stdlib/src/memory/__init__.mojo">modularml/mojo 在特定提交的 mojo/stdlib/src/memory/__init__.mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo-marathons](https://discord.com/channels/1087530497313357884/1255303604894437388/1263990373001330699)** (43 messages🔥): 

> - `NumPy Performance Testing`
> - `Understanding CPU FLops`
> - `Matrix Multiplication Benchmarks`
> - `Architecture-Specific Optimizations`
> - `Mojo's Generics Limitations` 


- **NumPy 性能数据引起关注**：一位成员分享了 `numpy` 的基准测试结果，显示出显著的性能差异，NumPy 的数值约为 **271.864 GFLOPS**，而纯 Python 实现约为 **249.371 GFLOPS**。
   - 他们对循环展开（unrolling）几乎没有带来性能差异感到惊讶，质疑是否是由于 BLAS 中旧的特定架构优化所致。
- **计算与比较理论峰值 GFLOPS**：成员讨论了如何利用核心数、频率和 SIMD 宽度计算峰值 GFLOPS，并提供了一个 i7 7th Gen CPU 的计算示例。
   - *有人指出，对于 NumPy 性能来说，达到理论峰值 **358 GFLOPS** 的 80% 左右是合理的。*
- **矩阵乘法中的流水线效率**：讨论深入探讨了流水线效率，重点是通过避免内存相关的停顿（stalls）来保持 ALU 满载。
   - *特别提到在 Kaby Lake 架构上，FMA 指令是 2 个周期的操作，并引用了 [uops.info](https://uops.info/table.html)。*
- **在 Mojo 中调用 CPUID 获取目标信息**：对话转向在 Mojo 中使用 cpuid 指令来收集特定目标的架构信息，一些成员表示需要更好地暴露这些细节。
   - 讨论了现有的工具如 [CpuId.jl](https://github.com/m-j-w/CpuId.jl) 和 Intel 的 C 库，但指出它们是特定于架构的。
- **Mojo 在高性能计算泛型（Generics）方面的挑战**：成员一致认为 Mojo 需要暴露更多关于目标架构的信息，以支持真正的通用高性能算法。
   - 还提到了超级计算机拥有特定架构的 BLAS 库，强调了所需的复杂性和针对性。



**Link mentioned**: <a href="https://github.com/m-j-w/CpuId.jl">GitHub - m-j-w/CpuId.jl: Ask the CPU for cache sizes, SIMD feature support, a running hypervisor, and more.</a>: Ask the CPU for cache sizes, SIMD feature support, a running hypervisor, and more. - m-j-w/CpuId.jl

  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1263965843717623880)** (424 messages🔥🔥🔥): 

> - `ComfyUI recommendations`
> - `Forge vs. Easy Diffusion experience`
> - `Using Latent mode for Regional Prompter`
> - `Issues with VRAM and GPU compatibility`
> - `Upscaling errors in Forge` 


- **推荐 ComfyUI 而非 Forge**：多位用户建议从 Forge 切换到 **ComfyUI** 以获得更好的体验，理由是 **Forge** 的功能和兼容性存在问题。
   - 用户称赞 **ComfyUI** 拥有显著更多的工具和功能，尽管注意到它具有更复杂的基于节点的界面。
- **Forge 与 Easy Diffusion 的对比**：一位用户指出，虽然 **Forge** 比 **Easy Diffusion** 快，但它缺少一些功能，并且在放大（upscaling）时会输出错误。
   - 其他人评论了 **Forge** 中放大问题和分辨率处理不当的问题，并提出了替代方案。
- **在 Regional Prompter 中使用 Latent 模式**：提供了关于在 **Regional Prompter** 中使用 **Latent mode** 而非 **Attention mode** 的指导，以防止角色融合（character blending）。
   - 分享了详细的提示词和说明，以改进在多个角色 LoRA 中使用 **Latent mode** 的效果。
- **VRAM 和 GPU 兼容性问题**：讨论涵盖了 Stable Diffusion 的 VRAM 需求以及 GPU 的 VRAM 问题，特别是 **AMD** 显卡。
   - 解决方案包括使用**本地安装**，以及为家庭 GPU 能力有限的用户提供云端 GPU。
- **Forge 中的放大错误**：用户在利用 **Forge** 放大图像时遇到了 'NoneType' 错误。
   - 建议包括切换到 **hi-res fix** 和其他放大工具，如 **real-ESRGAN**。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://comfyanonymous.github.io/ComfyUI_examples/controlnet/">ControlNet 和 T2I-Adapter 示例</a>：ComfyUI 工作流示例</li><li><a href="https://www.youtube.com/watch?v=O8-0ZidswTw">Invoke AI (Stable Diffusion) 外延绘制 (Outpainting) 延时摄影 #01</a>：刚试用了适用于 Windows 的最新 InvokeAI 版本。总耗时约 70 分钟，但我进行了加速处理……我尝试了很多操作……</li><li><a href="https://www.shakker.ai/activitys/shake-the-world">Shakker AI - 优质 Stable Diffusion 模型枢纽</a>：未找到描述</li><li><a href="https://www.nasa.gov/missions/mars-2020-perseverance/perseverance-rover/heres-how-ai-is-changing-nasas-mars-rover-science/">AI 如何改变 NASA 火星车科学研究 - NASA</a>：人工智能正在帮助科学家识别毅力号 (Perseverance) 火星车研究的岩石中的矿物质。</li><li><a href="https://github.com/2kpr/ComfyUI-UltraPixel">GitHub - 2kpr/ComfyUI-UltraPixel</a>：通过在 GitHub 上创建账号，为 2kpr/ComfyUI-UltraPixel 的开发做出贡献。</li><li><a href="https://huggingface.co/stab">Stab (Fua)</a>：未找到描述</li><li><a href="https://youtu.be/kg9qpyupXbI">Follow-Your-Emoji: 精细可控且富有表现力的自由风格肖像动画</a>：Follow-Your-Emoji: 精细可控且富有表现力的自由风格肖像动画</li><li><a href="https://liveportrait.github.io/">具有拼接和重定向控制的高效肖像动画</a>：未找到描述</li><li><a href="https://github.com/comfyanonymous/ComfyUI/blob/master/extra_model_paths.yaml.example">comfyanonymous/ComfyUI master 分支下的 extra_model_paths.yaml.example</a>：最强大且模块化的 Stable Diffusion GUI、API 和后端，采用图形/节点界面。 - comfyanonymous/ComfyUI</li><li><a href="https://github.com/wootwootwootwoot/ComfyUI-RK-Sampler">GitHub - wootwootwootwoot/ComfyUI-RK-Sampler: 适用于 ComfyUI 的批量 Runge-Kutta 采样器</a>：适用于 ComfyUI 的批量 Runge-Kutta 采样器。通过在 GitHub 上创建账号，为 wootwootwootwoot/ComfyUI-RK-Sampler 的开发做出贡献。</li><li><a href="https://www.gmktec.com/products/amd-ryzen-7-7840hs-mini-pc-nucbox-k6?variant=7530f28e-cce6-4e22-ac92-e7999375a6be)">AMD Ryzen 7 7840HS 迷你电脑--NucBox K6</a>：一款紧凑型台式电脑，搭载 AMD Ryzen 7 7840HS 处理器，GPU 性能相当于 GTX1060Ti。双 2.5G 网口。双 M.2 插槽。双风扇。</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui.git">GitHub - AUTOMATIC1111/stable-diffusion-webui: Stable Diffusion Web UI</a>：Stable Diffusion Web UI。通过在 GitHub 上创建账号，为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://github.com/ehristoforu/DeFooocus">GitHub - ehristoforu/DeFooocus: 始终专注于提示词和生成</a>：始终专注于提示词和生成。通过在 GitHub 上创建账号，为 ehristoforu/DeFooocus 的开发做出贡献。</li><li><a href="https://github.com/comfyanonymous/ComfyUI">GitHub - comfyanonymous/ComfyUI: 最强大且模块化的 Stable Diffusion GUI、API 和后端，采用图形/节点界面。</a>：最强大且模块化的 Stable Diffusion GUI、API 和后端，采用图形/节点界面。 - comfyanonymous/ComfyUI</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers">stabilityai/stable-diffusion-3-medium-diffusers · Hugging Face</a>：未找到描述</li><li><a href="https://www.stablediffusiontutorials.com/2024/03/stable-diffusion-error-amd.html">修复在 AMD 上运行 Stable Diffusion 时的错误</a>：未找到描述</li><li><a href="https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/">TLDR: 使用 LLM 构建的一年总结 – D-Squared</a>：未找到描述
</li>
</ul>

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1264220301546361033)** (28 条消息🔥): 

> - `Google Meet 故障排除`
> - `CUDA 与 SVD`
> - `使用 LLM 构建应用`
> - `工作站中的 ECC 内存`
> - `Flash Attention 中的寄存器分配` 


- **活动 Google Meet 链接已确认**：确认后指出，该活动的正确 Google Meet 链接是 [这一个](https://meet.google.com/pha-emvg-pem)。
   - 一位用户提到在使用 Firefox 时遇到问题，但切换到 Chrome 后问题得到了解决。
- **了解 GPU 上的 SVD 性能**：一位用户询问 **cusolver/hipSolver** 如何执行 SVD，因为发现的大多数实现只是闭源解决方案的封装。
   - 引用了一个 [GitHub 仓库](https://github.com/Michalos88/Randomized_SVD_in_CUDA) 以获取见解，其中提到了 **Gram-Schmidt**、**Householder reflections** 和 **Givens rotations** 等方法。
- **LLM：一年构建经验的关键教训**：一段[新视频和博客文章](https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/)总结了从业者使用 LLM 工作一年的经验教训，强调了战术、运营和战略方面的见解。
   - 作者创建了一个视觉化的 TLDR 版本，使该系列更易于理解，并强调观看视频对深入理解很有价值。
- **辩论工作站中 ECC 内存的必要性**：用户们辩论了工作站中 **ECC 内存** 的必要性，共识认为对于大多数桌面应用来说它并非至关重要。
   - 讨论内容包括对特殊计算环境中**宇宙射线**的考虑，以及对 **CPU 选择和成本** 的影响。
- **关于 Flash Attention 中寄存器分配的澄清**：一位用户询问如何在 Flash Attention 中显式分配寄存器，以及输入矩阵的初始投影是否被融合（fused）进一个 kernel 中。
   - 疑问在于矩阵尺寸是否过大，以至于无法有效支持此类 kernel 融合。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/Michalos88/Randomized_SVD_in_CUDA">GitHub - Michalos88/Randomized_SVD_in_CUDA: FAST Randomized SVD on a GPU with CUDA 🏎️</a>：在带有 CUDA 的 GPU 上实现快速随机化 SVD 🏎️。可以通过创建账号为 Michalos88/Randomized_SVD_in_CUDA 的开发做出贡献。</li><li><a href="https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/">TLDR: 1 year of building with LLMs &#8211; D-Squared</a>：未找到描述
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1264729033674068031)** (4 条消息): 

> - `分析 Triton Kernel`
> - `内存使用情况`
> - `用于分析的 CUDA 工具`
> - `Nsight Compute`
> - `Nsight Systems` 


- **讨论用于分析 Triton kernel 的工具**：一位成员询问哪些工具适合分析 Triton kernel，特别是针对峰值内存使用量等任务。
   - 推荐使用 *nsight-compute* 和 *nsight-systems* 进行详细分析，并指出应避免使用 *nvprof*，因为它已被这些工具取代。
- **标准 CUDA 工具与 Triton kernel 的兼容性**：有一段关于 *nvprof* 等标准 CUDA 工具和内置 torch profiler 是否适用于 Triton kernel 的对话。
   - 一位成员提到，虽然传统工具可能不起作用，但像 *nsight-compute* 和 *nsight-systems* 这样的新工具提供了所需的功能。


  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1263941656420421763)** (7 条消息): 

> - `at::Tensor.mutable_data_ptr`
> - `torch.cond`
> - `torch 控制流算子` 


- **对 at::Tensor 中 const 方法的困惑**：一位成员对为什么 `at::Tensor.mutable_data_ptr` 是一个 `const` 方法表示困惑，并对其背后的设计选择提出质疑。
- **在 Torch 中模拟 jax.lax.while_loop()**：一位成员询问 Torch 是否支持类似于 `jax.lax.while_loop()` 的功能，以及是否可以通过 `torch.cond()` 模拟这种行为。
   - 另一位用户指出，`torch.cond()` 目前仅支持推理（inference），不支持训练（training），并引用了 [官方文档](https://pytorch.org/docs/stable/generated/torch.cond.html#torch.cond)。



**提到的链接**：<a href="https://pytorch.org/docs/stable/generated/torch.cond.html#torch.cond">torch.cond &mdash; PyTorch 2.3 documentation</a>：未找到描述

  

---

### **CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1264219545598562324)** (1 条消息): 

> - `AMD ROCm`
> - `Composable Kernel library`
> - `AMD tech stack` 


- **AMD ROCm 的 Composable Kernel 库演示**：一位成员宣布了一个学习 AMD ROCm **Composable Kernel 库**的机会，由 <@813802565426479145> 在约 4 分钟内进行演示。
   - 此次演讲被强调为深入了解 **AMD 技术栈 (AMD tech stack)** 的绝佳机会。
- **AMD 技术栈概览**：<@813802565426479145> 将提供 AMD 技术栈的概览，重点关注 ROCm 生态系统。
   - 这包括关于 **Composable Kernel** 库如何集成到 ROCm 中的见解。


  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1264249726530879559)** (3 条消息): 

> - `Similarity Search Algorithm`
> - `Schotastic Rounding in Quantization` 


- **为 SQLite 选择高效的相似性搜索算法**：一位成员询问了适用于像 SQLite 这样**轻量级且高效的向量数据库**的最佳相似性搜索算法建议。
   - *任何关于特定算法的有益建议或个人经验都将有助于决策。*
- **量化中随机舍入 (Stochastic Rounding) 的有效性**：一位成员询问在量化背景下引入随机元素是否符合 **stochastic rounding** 的定义。
   - *目前没有回复提供进一步的澄清或替代方案。*


  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1263989830673502249)** (27 条消息🔥): 

> - `CubeCL`
> - `FlashAttention2 Custom Mask`
> - `FLUTE Kernel` 


- **CubeCL 在 Rust 中启用 CUDA kernel**：[CubeCL 项目](https://github.com/tracel-ai/cubecl)为 Rust 引入了多平台高性能计算语言扩展，使用 nvrtc 处理 CUDA，并使用 comptime 系统进行 kernel 特化 (specialization)。
   - 讨论点包括借用检查 (borrow checking) 和指针的限制，虽然 CubeCL 的系统目前仍远未达到 Zig 的能力，但在避免边界检查 (bounds checks) 和循环展开 (loop unrolling) 方面非常有用。
- **FlashAttention2 中的自定义掩码 (Custom Masking)**：az_zz_za 发布的一个新 [GitHub 仓库](https://github.com/alexzhang13/flashattention2-custom-mask) 为 FlashAttention2 的 Triton 实现引入了自定义掩码，解决了常见的局限性。
   - 讨论涉及正确的掩码维度以及实现任意注意力偏置 (attention biases) 的潜在简易修改方法。
- **FLUTE 加速 LLM 推理**：[FLUTE](https://github.com/HanGuo97/flute) 提供了一个用于非均匀量化（通过查找表）LLM 推理的 CUDA kernel，相比传统方法有显著的速度提升。
   - 该 kernel 已集成到 vLLM 中，并利用 CUTLASS 3、TensorCore 和 Async Copy 实现了高达 2-3 倍的性能提升，同时还讨论了在 Triton 和 Torch 中实现的潜力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="http://ziglang.org>">no title found</a>: no description found</li><li><a href="https://x.com/hanguo97/status/1815104671289413664?s=46">Han Guo (@HanGuo97) 的推文</a>: 介绍 FLUTE，一个用于非均匀量化（通过查找表）LLM 推理的 CUDA kernel。它开箱即用地加速了 QLoRA 的 NormalFloat (NF) 等。作为应用，我们扩展了 NF...</li><li><a href="https://github.com/alexzhang13/flashattention2-custom-mask">GitHub - alexzhang13/flashattention2-custom-mask: 添加了自定义掩码的 FlashAttention2 Triton 实现。</a>: 添加了自定义掩码的 FlashAttention2 Triton 实现。 - alexzhang13/flashattention2-custom-mask</li><li><a href="https://github.com/tracel-ai/cubecl">GitHub - tracel-ai/cubecl: Rust 的多平台高性能计算语言扩展。</a>: Rust 的多平台高性能计算语言扩展。 - tracel-ai/cubecl
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1264131761311580221)** (17 条消息🔥): 

> - `Triton Block Size`
> - `NVCC 编译器的非 GPU 部分与 VLA`
> - `FP16 与 FP32 性能`
> - `Triton 多阶段流水线 (Multi-Stage Pipelining)` 


- **Triton Block Size 详解**：一位成员澄清说，**Triton 中的 BLOCK_SIZE** 更像 **CUDA 中的 TILE_SIZE**，而 **num_warps** 则控制线程块（thread block）的大小 ([参考资料](https://triton-lang.org/main/python-api/generated/triton.Config.html))。
   - 另一位成员在讨论相似点和差异时承认：“*起初我也感到困惑*”。
- **NVCC 编译器与变长数组 (Variable-Length Arrays)**：一位成员询问在 NVCC 编译器的非 GPU 部分使用**运行时定义长度的数组** (VLA) 是否可行，即使不考虑内存或速度问题 ([参考资料](https://en.wikipedia.org/wiki/Variable-length_array))。
   - 另一位成员解释说，VLA 从技术上讲是 **C 语言特性**，在 **C++** 标准中可能未获得完全支持，尽管大多数编译器都实现了它。
- **FP16 性能考量**：讨论集中在为什么 A100 的 **FP16 性能** 是 **FP32 的 2 倍**，尽管从计算复杂度来看，预期的比例似乎应该在 2.8 倍左右 ([Ampere 架构白皮书](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf))。
   - 成员们调查了性能瓶颈是否可能是 **I/O 密集型 (I/O-bound)** 而非计算密集型，并讨论了硬件架构和开销。
- **理解 Triton 多阶段流水线**：一位成员询问了 **Triton 中流水线阶段 (pipelining stages)** 的目的和实现细节，特别是 kernel 阶段中使用的那些异常具体的固定值 ([入门教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html))。
   - 他们难以想象多阶段流水线除了基础的 CPU 架构流水线之外还能完成什么，这表明需要进一步的澄清。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Variable-length_array">Variable-length array - Wikipedia</a>：未找到描述</li><li><a href="https://triton-lang.org/main/python-api/generated/triton.Config.html">triton.Config &mdash; Triton 文档</a>：未找到描述</li><li><a href="https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html#sphx-glr-getting-started-tutorials-06-fused-attention-py">Fused Attention &mdash; Triton 文档</a>：未找到描述
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1264673584341581844)** (3 条消息): 

> - `Shared Memory 中的模型权重`
> - `CUDA 寄存器容量` 


- **讨论 Shared Memory 中的模型权重**：一位用户询问，如果寄存器文件 (register files) 不是问题，**模型权重**是否是使用 Shared Memory 的一个好例子。
   - 另一位成员给出了肯定的回答，但反问为什么不把所有模型权重都放入**寄存器**中。
- **关于寄存器容量的澄清**：引发了关于寄存器容量问题是否会影响将模型权重放入 Shared Memory 决策的讨论。
   - 澄清指出，如果寄存器容量不是限制因素，那确实是一个可行的选择。


  

---


### **CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/)** (1 条消息): 

andreaskoepf: https://youtu.be/-732zELVbpU?si=HBXEE8t2fxCKhC5v
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1264620506754060371)** (8 条消息🔥): 

> - `请求开设 Cuda/C++ 频道`
> - `关于现有频道使用的讨论`
> - `LLM 部署成本咨询` 


- **请求开设专门的 Cuda/C++ 频道**：一位成员询问是否可以创建一个 **Cuda/C++** 频道，理由是编写 **Triton** 和 **Cuda C++** 的人对此很感兴趣。
   - 另一位用户将他们引导至一个现有频道，但请求者不确定那是否合适，因为相关的讨论非常有限。
- **关于现有频道使用的讨论**：成员们辩论了现有频道是否充分涵盖了 **Cuda** 和 **Triton 的讨论**。
   - 一位用户注意到上一次提到 **Triton** 还是在 6 月 5 日，这引发了对该频道相关性的怀疑。
- **LLM 部署成本咨询**：一位成员询问了大规模 **LLM 部署成本** 的准确数字，并对比了 **Mistral, OpenAI 和 Fireworks AI** 等各家供应商。
   - 他们推测，除非硬件利用率接近 100% 或能获得非常廉价的硬件，否则这些公司可能会亏损。


  

---

### **CUDA MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1264560997079842816)** (6 条消息): 

> - `ICML 会议线下聚会`
> - `伯克利的 CUDA 学习者` 


- **ICML 聚会协调**: 一位成员提到正在参加 **ICML**，促使另一位成员建议在会议期间见面。
   - 该提议得到了积极回应（“当然”），标志着双方对见面有共同兴趣。
- **寻找伯克利的 CUDA 学习者**: 一位成员询问伯克利地区的 **CUDA 学习者**，表示有兴趣建立一个社区进行线下聚会，并共同解决书中的章节练习。
   - 另一位用户回忆起[过去对类似聚会的兴趣](https://link.to/nolink)，表明之前对此类活动已有热情。


  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1263988699260129311)** (297 条消息🔥🔥): 

> - `CUDA MODE IRL 线下活动`
> - `从训练 GPT-2 到 GPT-3 的进展`
> - `多 GPU 与 ZeRO 实现`
> - `MuP 分支进展`
> - `FP8 训练优化` 


- **9 月 21 日 CUDA MODE IRL 邀请**: **CUDA MODE** 团队成员受邀参加 **9 月 21 日**在旧金山举行的 IRL 线下活动，该活动恰逢 PyTorch devcon，届时可能包含一个关于 **llm.c** 的 20 分钟演讲。
   - 活动的物流和细节已通过 [Google Document](https://docs.google.com/document/d/10LkM5_xLh9r_ycul2ywOfgrOGmNP9YDTa9c4V755QgY/edit) 共享。
- **从 train_gpt2.c 转向 train_llama3.cu**: CUDA MODE 的开发任务包括将代码库从 **train_gpt2.c** 迁移到 **train_llama3.cu**。
   - 由于与 Meta 发布的 LLaMA 存在差异，这涉及大量的重构工作，包括 torchrun 的问题以及复杂的未记录代码。
- **多 GPU 与 ZeRO 的挑战**: 关于**多 GPU 训练**的讨论包括应对集成 ZeRO-2 和 ZeRO-offload 以有效管理大型模型权重的挑战。
   - 核心问题在于平衡 GPU 和 CPU 之间的内存、保持确定性以及高效集成 master weights。
- **MuP 分支稳定性问题**: 尝试将 **MuP 分支**与 master 合并以稳定训练运行，但遇到了冲突并需要进行大量的 rebasing。
   - 目前的重点是解决这些冲突，并测试 MuP 在长期运行中增强稳定性的潜力。
- **FP8 训练优化的进展**: **FP8 训练优化**的工作正在进行中，计划实现 cuDNN FP8 前向和后向 attention 等技术。
   - 进一步的计划包括添加张量值的可视化功能，以便在训练优化期间跟踪和调试模型性能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://lambdalabs.com/nvidia-gh200">Lambda Reserved Cloud Powered by NVIDIA GH200</a>：由 NVIDIA GH200 提供支持的 Lambda Reserved Cloud。GH200 Superchip 的设计在 NVIDIA Grace™ CPU 和 Hopper™ GPU 之间建立了高带宽连接。</li><li><a href="https://arxiv.org/abs/2407.05872">Scaling Exponents Across Parameterizations and Optimizers</a>：跨参数化和优化器的缩放指数：将模型从窄宽度稳健且有效地缩放到宽宽度，通常需要精确调整许多算法和架构细节，例如参数化和优化器的选择...</li><li><a href="https://arxiv.org/abs/2309.14322">Small-scale proxies for large-scale Transformer training instabilities</a>：大规模 Transformer 训练不稳定性的小规模代理：训练大型 Transformer 模型的团队报告称，在大规模训练时会出现小规模使用相同超参数训练时未出现的不稳定性。虽然...</li><li><a href="https://x.com/Yuchenj_UW/status/1814703583453192272">Tweet from Yuchen Jin (@Yuchenj_UW)</a>：Yuchen Jin (@Yuchenj_UW) 的推文：使用 @karpathy 的 llm.c + FineWeb-Edu 训练的不同尺寸 GPT-2（从 124M 到 2.7B）的对比🤖 GPT-2 最初有 4 种尺寸，最大为 1.5B。使用 llm.c 代码库，...</li><li><a href="https://github.com/karpathy/llm.c/pull/700/files">Fix integer overflow by using `size_t` for parameter sizes. by YuchenJin · Pull Request #700 · karpathy/llm.c</a>：通过为参数大小使用 `size_t` 来修复整数溢出。由 YuchenJin 提交 · Pull Request #700 · karpathy/llm.c：由于 GPT-2 7.3B 模型中的某些参数非常大，当前的 llm.c 代码存在整数溢出问题。这是因为它使用 32 位 int 来存储权重字节数...</li><li><a href="https://arxiv.org/abs/2406.16793">Adam-mini: Use Fewer Learning Rates To Gain More</a>：Adam-mini：用更少的学习率获得更多收益：我们提出了 Adam-mini，这是一种优化器，其性能与 AdamW 相当或更好，但内存占用减少了 45% 到 50%。Adam-mini 通过削减学习率资源来减少内存...</li><li><a href="https://developer.download.nvidia.com/cg/atan2.html">atan2</a>：未找到描述</li><li><a href="https://github.com/karpathy/llm.c/blob/master/llmc/matmul.cuh#L134">llm.c/llmc/matmul.cuh at master · karpathy/llm.c</a>：llm.c/llmc/matmul.cuh (master 分支) · karpathy/llm.c：使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账户来为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://docs.google.com/document/d/10LkM5_xLh9r_ycul2ywOfgrOGmNP9YDTa9c4V755QgY/edit?usp=sharing">CUDA MODE IRL invitation</a>：CUDA MODE IRL 邀请函：这是对首届 CUDA MODE IRL 黑客松的正式邀请，我们想邀请您进行主题演讲。该活动由 Accel 赞助，并将在...举办。</li><li><a href="https://github.com/karpathy/llm.c/pull/705">Refactor C code by gordicaleksa · Pull Request #705 · karpathy/llm.c</a>：由 gordicaleksa 重构 C 代码 · Pull Request #705 · karpathy/llm.c：意识到我们是在内联进行激活大小计算，因此将其更改为与我们计算参数大小的方式一致。</li><li><a href="https://github.com/karpathy/llm.c/pull/702">Restore from master weights (&amp; allow restoring from a checkpoint of different precision) by ademeure · Pull Request #702 · karpathy/llm.c</a>：从主权重恢复（并允许从不同精度的检查点恢复）由 ademeure 提交 · Pull Request #702 · karpathy/llm.c：对于保存了新 `rng_state_last_update` 的新检查点，这是完全确定性的，因此来自主权重的随机舍入（stochastic rounding）是使用完全相同的种子完成的（在恢复...时）</li><li><a href="https://github.com/Dao-AILab/flash-attention/blob/74b0761ff7efc7b90d4e5aeb529c1b2a09a7458c/README.md?plain=1#L201">flash-attention/README.md at 74b0761ff7efc7b90d4e5aeb529c1b2a09a7458c · Dao-AILab/flash-attention</a>：flash-attention/README.md · Dao-AILab/flash-attention：快速且内存高效的精确注意力机制（exact attention）。通过在 GitHub 上创建账户来为 Dao-AILab/flash-attention 的开发做出贡献。</li><li><a href="https://github.com/jorahn/llm.c">GitHub - jorahn/llm.c: LLM training in simple, raw C/CUDA</a>：GitHub - jorahn/llm.c：使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账户来为 jorahn/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/pull/694">Model init cleanup by ngc92 · Pull Request #694 · karpathy/llm.c</a>：由 ngc92 提交的模型初始化清理 · Pull Request #694 · karpathy/llm.c：将模型参数分配整合到单一源位置，使梯度缓冲区累加变为 eager 模式，移动了编码器确定性辅助缓冲区，以便它们由 forward 提前分配 -> ...</li><li><a href="https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g3f0bdfc73288f9dda45e5c9be7811c9d">CUDA Math API :: CUDA Toolkit Documentation</a>：CUDA Math API :: CUDA Toolkit 文档：未找到描述</li><li><a href="https://x.com/rosieyzh/status/1811790177246888075">Tweet from Rosie Zhao @ ICML (@rosieyzh)</a>：Rosie Zhao @ ICML (@rosieyzh) 的推文：在我们关于评估 LLM 训练优化器的新工作中，我们进行了一系列实验，以研究 Adam 等优化器中的自适应性在实现良好性能和稳定性方面的作用....</li><li><a href="https://arxiv.org/abs/2407.07972">Deconstructing What Makes a Good Optimizer for Language Models</a>：解构什么才是语言模型的优秀优化器：随着规模的扩大，训练语言模型的成本变得越来越高，这促使人们尝试提高优化效率。尽管这些

尽管付出了努力，Adam 优化器仍然是使用最广泛的...</li><li><a href="https://github.com/EurekaLabsAI/mlp/pull/11">由 gordicaleksa 添加 C 语言实现 · Pull Request #11 · EurekaLabsAI/mlp</a>：添加了一个遵循 llm.c 的 train_gpt2.c 风格的 C 语言实现。在我觉得可以做得更好的地方，我做了一些微调。经过等效性测试 -> logits、loss、grads 均...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1264244442253234196)** (10 条消息🔥): 

> - `ROCm 硬件入门选项`
> - `RDNA 与 CDNA 的区别`
> - `MI300 的能力`
> - `在 ROCm 中使用 AMD GPU 的挑战`
> - `MI300 上的 FP8 MMA 加速` 


- **讨论了 ROCm 硬件入门选项**：成员们质疑 **Radeon RX 7900 XTX** 和 **Radeon PRO W7900** 是否是 ROCm 开发的最佳选择，并将其与 **MI300X** 进行了对比。
   - *另一位成员询问了在本地拥有性能较弱的 AMD GPU 与使用云端方案进行 ROCm kernel 开发的价值对比。*
- **RDNA 与 CDNA 的分化**：自 GCN 以来，RDNA 和 CDNA GPU 已经分道扬镳，RDNA GPU 每个 CU 拥有更多寄存器、更多 shared memory，并同时支持 w32 和 w64。
   - 有人指出 RDNA dGPU 缺乏 XNACK（一种允许详细页面错误处理的功能），这一缺陷限制了如启用 ASAN 的 kernel 编译等功能。
- **MI300 系列能力亮点**：**MI300A** 与 CPU 共享内存，而 **MI300X** 则不共享且缺乏纹理单元，且 MI300 没有可用的 Vulkan 驱动。
   - *MI300X 主要通过云服务商提供，这些差异会影响某些纹理缓存操作，但通常与 HIP 无关。*
- **在 ROCm 开发中使用 AMD GPU 的挑战**：近期的 AMD GPU 可用于 ROCm kernel 开发，但像 RX6500 这样的一些型号缺乏某些操作，且 APU 存在支持问题。
   - 为不支持的 GPU 重新编译 ROCm 栈以及缺乏性能调优也被列为挑战。
- **独有特性：MI300 上的 FP8 MMA 加速**：MI300 的一个独特之处在于它支持 **FP8 MMA 加速**，这使其与其他 GPU 区别开来。


  

---


### **CUDA MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1263933386515419208)** (1 条消息): 

> - `算术强度 (Arithmetic Intensity)`
> - `GPU 性能指标` 


- **关于算术强度在 GPU 性能中使用的疑问**：一位成员询问为什么使用 **算术强度 1** 来检查工作负载是内存受限还是计算受限，并质疑这是否应该取决于特定 GPU 的 FLOPS/GB/s 带宽比。
   - 该成员指出，根据 GPU 型号的不同，这个比例可能高达 **20**。
- **GPU FLOPS/GB/s 比例对性能评估的影响**：讨论强调了 GPU 的 FLOPS/GB/s 比例（该比例会变化）如何影响对内存受限或计算受限工作负载的判定。
   - *这种差异使得在评估性能时必须考虑具体的 GPU 型号。*


  

---

### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1263994137670320259)** (168 条消息🔥🔥): 

> - `C# 中的 GGUF 文件元数据`
> - `量化过程 (q5, q6, f16)`
> - `LM Studio 与 Hugging Face 的问题`
> - `在本地运行大型模型`
> - `使用本地 LLMs 获取收益` 


- **在 C# 中解析 GGUF 文件元数据**：一位用户宣布在 C# 中实现了 GGUF 文件元数据的解析，并介绍了一个旨在报告元数据并可能提供一些统计数据的控制台工具。
   - 最初打算作为文件属性的 ShellExtension，但该用户遇到了注册问题，因此决定专注于控制台工具。
- **量化过程的困惑**：针对用于使大型模型适应旧硬件的量化过程（例如 **q5**、**q6**、**f16**）进行了详细解释，其中 **q5** 的量化程度比 **q8** 更高。
   - 一位用户澄清说 **f16 未经量化**且体积大得多，而另一位用户分享了在配备 **RTX 3050 的 Dell 笔记本电脑**上，利用有限的 VRAM 运行大型模型的经验。
- **LM Studio 和 Hugging Face 搜索问题**：多位用户报告 LM Studio 的搜索功能损坏，可能是由于 Hugging Face API 的问题。
   - 尽管采取了使用 VPN 和重新安装等故障排除步骤，搜索功能仍间歇性失败，直到确认问题已解决。
- **在本地高效运行大型模型**：用户分享了他们在本地运行大型模型的配置，使用 LM Studio 和 llama.cpp 等工具高效平衡 VRAM 和 RAM 的使用。
   - 一位用户建议更新 NVIDIA 和 CUDA 驱动程序以解决大型模型的加载问题，使本地推理更加顺畅。
- **本地 LLMs 的收益与实用性**：讨论探讨了使用本地 LLM 的原因：维护企业使用的数据隐私以及个人项目的离线能力。
   - 一位用户强调，与 **ChatGPT** 等云模型不同，本地 LLM 将数据保留在用户的机器上，并强调收益通常是通过实用性获得的，而不是像加密货币挖矿那样的直接金钱收益。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.ibm.com/topics/ai-hallucinations">什么是 AI 幻觉？ | IBM</a>：AI 幻觉是指大型语言模型 (LLM) 感知到不存在的模式或对象，从而产生荒谬或不准确的输出。</li><li><a href="https://github.com/ggerganov/llama.cpp/commit/50e05353e88d50b644688caa91f5955e8bdb9eb9">llama : 添加 Mistral Nemo 推理支持 (#8604) · ggerganov/llama.cpp@50e0535</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e73zxd/is_there_any_working_gemma_27b_in_gguf_format/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://huggingface.co/lmstudio-community/gemma-2-27b-it-GGUF">lmstudio-community/gemma-2-27b-it-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/8608">支持 SmolLM · Issue #8608 · ggerganov/llama.cpp</a>：前提条件：我正在运行最新代码。如果可能，请提及版本。我仔细阅读了 README.md。我使用与我的问题相关的关键词进行了搜索，以确保我正在创建...</li><li><a href="https://x.com/alpindale/status/1814814551449244058?s=12">Alpin (@AlpinDale) 的推文</a>：已确认有 8B、70B 和 405B。前两个是从 405B 蒸馏出来的。128k（十进制为 131k）上下文。405b 无法画出独角兽。Instruct 微调可能进行了安全对齐。架构...</li><li><a href="https://x.com/alpindale/status/1814717595754377562?s=46">Alpin (@AlpinDale) 的推文</a>：似乎 HF 的某人忘记及时将此仓库设为私有，Google 索引了它：sllhf/Meta-Llama-3.1-405B-Instruct-FP8。405B 是 Llama 3.1？非常有趣。我想知道他们是否只会发布...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/8579">llama : 添加了对 Tekken 预分词器的支持 (#8577) 由 m18coppola 提交 · Pull Request #8579 · ggerganov/llama.cpp</a>：为 Mistral Nemo 模型添加 Tekken 预分词器支持。在 convert-hf-to-gguf-update.py 中为 Mistral-Nemo-Base-2407 添加了分词器类型。在 convert-... 中为 Mistral-Nemo-Base-2407 添加了 chkhsh...
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1263936712871252018)** (126 条消息🔥🔥): 

> - `Open Model Test Results` (开源模型测试结果)
> - `Converting Models to GGUF` (将模型转换为 GGUF)
> - `New Jail-Breaking Technique` (新的越狱技术)
> - `DeepSeek-Coder Issues` (DeepSeek-Coder 问题)
> - `Memory Usage with Qwen2 72B` (Qwen2 72B 的内存占用)


- **分享开源模型测试结果**：一名成员分享了多种**开源模型测试**的新结果，并询问除了 **Gemma 2 27B** 之外，其他人是否发现了重大偏差。
   - 他们提到无法修复**损坏的 Gemma 2 27B** 问题。
- **为 LM Studio 将模型转换为 GGUF**：一位用户询问如何将 `microsoft/llava-med-v1.5-mistral-7b` 转换为 **LM Studio 使用的 GGUF**，并提到这是一个业余项目所需。
   - 另一名成员确认了他们是否成功获得了 **GGUF 转换**。
- **公布新的越狱技术**：一名成员宣布了一种针对前沿模型的新 **jail-breaking**（越狱）技术，并分享了[论文链接](https://arxiv.org/pdf/2407.11969)。
   - 他们建议在补丁发布前尝试使用。
- **DeepSeek-Coder V2 Lite Instruct 模型故障**：用户遇到 **DeepSeek-Coder V2 Lite Instruct** 在初始正常输出后生成不连贯文本的问题。
   - 故障排除尝试包括更改 context windows（上下文窗口）、禁用 flash、以及降级 **LM Studio 版本**，但均未获得持久的成功。
- **llama.cpp 中 Qwen2 72B 的高内存占用**：一名成员报告在使用 `llama.cpp` 加载 **Qwen2 72B** 时内存占用过高。
   - 另一名成员建议降低 **context length**（上下文长度）以更有效地管理内存使用。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/win10/DeepSeek-Coder-V2-Lite-Instruct-Q8_0-GGUF">win10/DeepSeek-Coder-V2-Lite-Instruct-Q8_0-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF#model-settings">lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF/discussions/2">lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF · Problem with LLM Studio</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1264981042142515202)** (2 条消息): 

> - `Hugging Face API Networking Errors` (Hugging Face API 网络错误)
> - `Issue Resolution` (问题解决)


- **Hugging Face API 遭遇网络错误**：发布了一则关于 **Hugging Face API** 出现**间歇性网络错误**的公告，该错误导致应用内的搜索功能出现问题。
   - *我们将在了解更多信息后在此更新。*
- **Hugging Face API 问题已解决**：**Hugging Face API** 的网络错误已解决，用户现在应该可以再次通过应用进行搜索。
   - *对给您带来的不便深表歉意。*


  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1264054340847669390)** (11 条消息🔥): 

> - `Failed model load` (模型加载失败)
> - `Flash Attention troubleshooting` (Flash Attention 故障排除)
> - `HuggingFace API issues` (HuggingFace API 问题)
> - `Alternative model repositories` (备选模型仓库)


- **由于 llama.cpp 错误导致模型加载失败**：一名成员分享了一个关于在 LM Studio 中加载模型失败的错误，原因是 llama.cpp 中存在未知的 pre-tokenizer 类型：'mistral-bpe'。
   - 另一名成员提到，新发布的模型在当前的 LM Studio 版本中无法运行，需要 llama.cpp 架构支持的更新。
- **Flash Attention 可能导致模型加载问题**：一位用户建议，启用 Flash Attention 可能会导致模型无法加载，并建议关闭 Flash Attention 以解决此问题。
   - *尝试将其关闭并重新加载模型*，看看是否能解决问题。
- **HuggingFace API 稳定性问题影响 LM Studio**：HuggingFace API 的稳定性问题导致 LM Studio 中的模型浏览器（Model Explorer）目前无法正常工作。
- **请求备选模型仓库镜像**：由于 HuggingFace API 持续出现问题，一位用户建议在 LM Studio 中增加切换到第三方仓库镜像的功能。
   - 给出的示例是 [hf-mirror.com](https://github.com/padeoe/hf-mirror-site)，并提供了关于使用镜像脚本（如 [hfd.sh](https://gist.github.com/padeoe/697678ab8e528b85a2a7bddafea1fa4f)）进行下载的更多细节。



**提到的链接**：<a href="https://huggingface.co">Hugging Face – 构建未来的 AI 社区。</a>：未找到描述

  

---

### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1264345538682294374)** (2 messages): 

> - `Renaming Presets`
> - `Finding Models` 


- **Renaming Presets**: 一位成员询问是否可以重命名他们为不同用例创建的 Presets。
   - 他们提到在提问后不久就找到了解决方法。
- **Finding Models**: 同一位成员提到他们终于找到了一个适合自己需求的理想 Model。
   - 他们计划针对特定的用例创建 4 个 Prompts。


  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1263938288167686263)** (40 messages🔥): 

> - `NVidia Tesla P40 usage`
> - `AMD GPU compatibility`
> - `Home NAS recommendations`
> - `Choosing GPUs for AI/ML workloads`
> - `Finetuning Large Language Models` 


- **Tesla P40 与混合 GPU 运行**: 一些用户通过安装特定驱动程序（例如默认的 P40 数据中心驱动程序，然后是 Studio RTX 驱动程序），成功在 Windows 10 x64 上将 NVidia Tesla P40 与 RTX 3070 等其他 GPU 共同运行。
- **可以使用不同世代的 AMD GPU**: 一位用户询问是否可以在 RX 6800 和可能的 RX 7900XTX 上运行 LM Studio，另一位用户回答说，如果两者都支持 ROCm，就应该可以运行。
- **为 iPhone/iPad 设置家用 NAS**: 一位用户讨论了设置家用 NAS 以存储来自 iPhone 和 iPad 的内容，而不是购买内存容量更大的设备。
- **为 AI 工作负载升级 GPU 配置**: 关于各种 GPU 组合对 AI/ML 工作负载的适用性展开了辩论，一些用户考虑使用双 RTX 3090 以增强 VRAM 和性能。
   - 一位用户讨论了 RTX 3090 等某些 GPU 与 4060 相比功耗较高的问题，但得到的建议是 3090 将显著提高他们处理大型 Model 的能力。
- **量化模型 Finetuning 的挑战**: 成员们讨论了对 GGUF 量化模型进行 Finetuning 的困难，建议使用基础的 Safetensors 模型进行 Finetuning 会更有效。
   - 一位用户分享了一篇有用的 [Microsoft 文章](https://learn.microsoft.com/en-us/ai/playbook/technology-guidance/generative-ai/working-with-llms/fine-tuning)，介绍如何开始 Large Language Model 的 Finetuning。



**Link mentioned**: <a href="https://learn.microsoft.com/en-us/ai/playbook/technology-guidance/generative-ai/working-with-llms/fine-tuning">Getting started with LLM fine-tuning</a>: Large Language Model (LLM) Fine-tuning 是将预训练模型适配到特定任务的过程。该过程通过在新的数据集上更新其参数来完成。具体而言，LLM 会被传递...

  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1264315511152443515)** (18 messages🔥): 

> - `Nemo issues`
> - `Search functionality problems`
> - `Huggingface API` 


- **用户报告 Mistral Nemo 的问题**: *madan.pandit* 询问了他们在运行来自 **Mistral** 的 Nemo 模块时遇到的问题。
- **LM Studio 0.2.27 搜索功能失效**: 多位用户报告 **LM Studio 0.2.27** 的搜索功能损坏，在 Linux 和 Mac M2 上返回 0 个结果。
   - 报告指出 *直接在 Huggingface 上搜索正常*，但在应用内无效。
- **Huggingface API 宕机影响 LM Studio**: *heyitsyorkie* 确认了问题，并指出 **Huggingface API** 处于宕机状态，影响了 LM Studio 中的 Model Explorer。
   - *a_dev_called_dj_65326* 承认了 API 问题，并表示已通知团队。


  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/)** (1 messages): 

captainpumpkinhead: "natively" (原生)
  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1264584877794267156)** (1 messages): 

> - `Athese by Nexusflow`
> - `Model benchmarks`
> - `Multilingual performance` 


- **Nexusflow 推出 Athene 模型**: Nexusflow 的 Athene 在广泛的类别中表现出令人印象深刻的改进，可能是目前同尺寸中通用场景的 SOTA。
   - Nexusflow 使用内部开发的 Benchmark 创建了高质量的 RLHF 数据集，由此产生的 Benchmark 数据极具说服力。[在此查看模型](https://huggingface.co/lmstudio-community/Athene-70B-GGUF)。
- **Athene 在多语言性能方面表现出色**: Nexusflow 的 Athene 拥有大幅提升的 **Multilingual performance**（多语言性能），使其成为英语以外用途的绝佳选择。


  

---

### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1264760981859336223)** (2 messages): 

> - `LM Studio Discord bot`
> - `使用 LM Studio 的私密响应`
> - `GitHub 教程链接` 


- **使用 LM Studio 创建 Discord 机器人**：一位开发者分享了一篇题为《[我用 LM Studio.js 制作了一个 Discord 机器人](https://dev.to/mrdjohnson/i-made-a-discord-bot-with-lmstudiojs-4fd6)》的博文，详细介绍了如何创建一个使用 LM Studio 进行响应的 Discord 机器人。
   - 该博文包含教程以及发布在 [GitHub](https://github.com/mrdjohnson/lmstudio-discord-bot) 上的源码，并提供了关于机器人实现私密响应所需修改的见解。
- **感谢分享**：另一位成员感谢开发者分享了关于使用 LM Studio 创建 Discord 机器人的博文。
   - 他们对该教程和提供的 GitHub 链接表示赞赏。



**提及的链接**：<a href="https://github.com/mrdjohnson/lmstudio-discord-bot">GitHub - mrdjohnson/lmstudio-discord-bot: 一个创建使用 LM Studio 响应的 Discord 机器人的教程！此代码基于此处发现的博文：https://dev.to/mrdjohnson/i-made-a-discord-bot-with-lmstudiojs-4fd6</a>：一个创建使用 LM Studio 响应的 Discord 机器人的教程！此代码基于此处发现的博文：https://dev.to/mrdjohnson/i-made-a-discord-bot-with-lmstudiojs-4fd6 - mrdjohnson/lm...

  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1263935249000435762)** (338 messages🔥🔥): 

> - `Pro 订阅的图像生成`
> - `GPTs Agents`
> - `Pro 搜索的问题`
> - `Profile 和 Collection 提示词`
> - `Perplexity 的上下文窗口` 


- **浏览器重启解决了图像生成问题**：一位用户报告称在使用 Pro 订阅时无法生成多张图像，但重启 Chrome 浏览器解决了该问题。
- **GPTs Agents 在初始训练后无法学习**：一位成员表达了对 GPTs Agents 在初始训练后无法从提供的额外信息中学习的担忧。
- **Pro 搜索的限制和偏好**：几位用户讨论了 Pro 搜索有时会分散注意力，且不遵循用户提示词，使其在某些任务中不那么受欢迎。
- **Profile 和 Collection 提示词并非总是有效**：用户讨论了 Profile 和 Collection 提示词通常不会影响 Pro 搜索期间 AI 的搜索过程，从而限制了它们的实用性。
- **Perplexity 的上下文窗口和 Token 使用情况**：Perplexity 的上下文窗口限制为每轮 32k Tokens，但累积的多轮对话最高可处理 128k Tokens。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://paul.kinlan.me/use-bookmarklets-on-chrome-on-android/">在 Android 版 Chrome 上使用书签脚本 (Bookmarklets)</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=ts3DqM_t3fA&t=38s">这可能让 AI 真正变得有用</a>：使用我们的链接 https://ground.news/TechLinked 获取 Vantage 计划 40% 的折扣。获取本地视角以更好地了解世界政治和时事...</li><li><a href="https://prateekkeshari.gumroad.com/l/peek">Peek - 在 MacOS 任何地方访问 AI (ChatGPT, Gemini, Perplexity, Poe, Claude) 和 Threads 的免费应用</a>：截至 2024 年 4 月 30 日已更新新版本 🚀 Peek 是一款 MacOS 菜单栏应用程序，允许您在一个地方与多个 AI 聊天机器人互动。您可以访问的 AI 列表：ChatGPT、Gemini、Perple...</li><li><a href="https://x.com/perplexity_ai/status/1815431484767142272?s=61">来自 Perplexity (@perplexity_ai) 的推文</a>：懂的都懂。</li><li><a href="https://llmtokencounter.com/">LLM Token 计数器</a>：未找到描述</li><li><a href="https://zapier.com/blog/add-search-engine-to-chrome/">如何向 Chrome 添加自定义搜索引擎 | Zapier</a>：当您向 Chrome 添加自定义搜索引擎时，您可以直接从 Chrome 轻松搜索您常用的网站或应用。操作方法如下。</li><li><a href="https://arxiv.org/html/2407.10887">嘿，那是我的模型！介绍 Chain &amp; Hash，一种 LLM 指纹识别技术</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1263941449926312067)** (23 条消息🔥): 

> - `YouTube 测试 AI 对话功能`
> - `OpenAI 发布 GPT-4.0 mini`
> - `Unraveling Chaos 项目`
> - `CrowdStrike 全球 IT 故障`
> - `软件开发的各种可能性` 


- **YouTube AI 对话功能开启测试**：Perplexity AI 探索了 [YouTube 正在测试的新型 AI 对话功能](https://www.perplexity.ai/page/youtube-tests-ai-conversationa-WMQ_b8XNQZuIhMpPPMyfGg)，以了解其功能逻辑和用户交互。
- **OpenAI 发布 GPT-4.0 Mini**：OpenAI 最近发布了 [GPT-4.0 Mini](https://www.perplexity.ai/page/openai-drops-gpt-4o-mini-viKDYptISzufyJDPoL3Etg)，这是一个旨在提高可访问性同时保持先进 AI 能力的精简版本。
- **CrowdStrike 面临重大全球 IT 故障**：CrowdStrike 遭遇了严重的 [全球 IT 故障](https://www.perplexity.ai/page/crowdstrike-global-it-outage-qKRKi2QWRuaWxf44d1G5nQ)，影响了众多客户，并凸显了其服务基础设施的脆弱性。
   - 社区讨论强调了云端安全提供商需要更好的韧性和应急预案。
- **剖析 Mistral NeMo 的 AI 飞跃**：Perplexity AI 的 [最新 YouTube 视频](https://www.youtube.com/embed/TA4E69jtF_U) 涵盖了 Mistral NeMo 在 AI 技术方面的重大进展、CrowdStrike 的全球故障，以及在火星上发现硫磺的惊人消息。
   - 一位爱好者表示：*“这展示了 AI 发展和太空探索未来令人兴奋的前景。”*
- **剑龙化石以 4460 万美元拍卖**：一件剑龙化石以惊人的 [4460 万美元](https://www.perplexity.ai/search/stegosaurus-sells-for-44-6m-_E76GAUBQ4mAHurns1O.ZA) 成交，吸引了古生物学爱好者和艺术收藏家的关注。
   - 这一创纪录的拍卖价格凸显了稀有历史文物的极高价值。



**提到的链接**: <a href="https://www.youtube.com/embed/TA4E69jtF_U">YouTube</a>: 未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1264253137041227807)** (9 条消息🔥): 

> - `功能路线图`
> - `Perplexity API Token 计费`
> - `在线模型使用` 


- **功能路线图页面消失**：**Feature Roadmap** 页面从 [文档网站](https://docs.perplexity.ai/docs/feature-roadmap) 中消失了。
- **Perplexity API 对输入和输出 Token 计费**：在一场讨论中明确了 **Perplexity API** 会同时对输入（In）和输出（Out）Token 进行计费。
- **使用在线模型解释 PDF**：一名成员使用 Perplexity 的在线模型来解释 PDF 文档。


  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1263933578358554716)** (170 条消息🔥🔥): 

> - `Sonnet 3.5 对比 4o mini`
> - `微调多模态模型`
> - `高性价比 TTS 方案`
> - `语音助手应用`
> - `GPT-4o mini 对比 GPT-5` 


- **4o mini 表现优于 Sonnet 3.5**：**4o mini** 能够解决甚至连整个 Claude 系列都难以应对的复杂问题，凸显了其卓越的能力。
- **为珊瑚礁研究微调多模态模型**：一位用户询问关于微调多模态模型以研究珊瑚礁的问题，但另一位用户建议使用 Azure AI 的 Custom Vision Service 以获得更好的准确度。
- **选择高性价比 TTS 方案**：一名成员建议在聊天中使用 OpenAI 的 [Text-to-Speech](https://platform.openai.com/docs/guides/text-to-speech) 服务进行语音处理。
   - 为了提高成本效益和性能，还讨论了 **Coqui.ai** 和本地运行模型等其他选项。
- **集成 GPT 的语音助手应用**：成员们讨论了语音助手应用，提到 **VoiceGPT** 作为一个开源选项，可以在 Android 屏幕上叠加 ChatGPT 的监听动画。
   - 由于该助手无法直接打开应用，因此建议使用 **macrodroid/tasker** 或与 **Home Assistant** 集成作为替代方案。
- **社区对 GPT-4o mini 和 GPT-5 充满期待**：社区对 **GPT-4o mini** 表现出极大的热情，讨论了其潜在影响，并对 **GPT-5** 等未来进展进行了推测。



**提到的链接**: <a href="https://search.arc.net/0p86iGsUHFk34vkGOclc">Markdown 代码块中的系统提示词 | Arc Search</a>: Arc Search 阅读了互联网上的网站，为您生成了这个完美的标签页。

  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1264073037695811585)** (32 messages🔥): 

> - `GPT-4o mini replacing GPT-3.5`
> - `Differences between GPT-4o and GPT-4o mini`
> - `New features for GPT-4o`
> - `API vs. ChatGPT features`
> - `GPT-4o mini for longform content` 


- **GPT-4o mini 将替代 GPT-3.5**：一名成员询问 **GPT-4o mini** 是否会成为新的免费模型并逐步淘汰 **GPT-3.5**，得到了肯定的答复。
   - *lumirix*: “是的。”
- **API 缺乏实时语音集成**：讨论强调了 OpenAI 的新语音功能可在 ChatGPT 中使用，但尚未在 API 中提供。
   - 一名成员提到，由于平台之间的功能差异，实时聊天语音功能不太可能很快进入 API。
- **ChatGPT 语音与 API Text-to-Speech 的对比**：用户对 ChatGPT 的新语音功能与 API 的 Text-to-Speech 终端之间的延迟和质量差异表示担忧。
   - *the_big_block_pc*: “我知道 TTS 终端，但那会增加很多延迟，而且听起来不如新的 ChatGPT 语音功能好。”
- **功能发布：API vs ChatGPT**：一名成员质疑为什么语音等新功能先添加到 ChatGPT 而不是 API。
   - 有人建议，用户对 ChatGPT 功能的需求可能更高，而且实时聊天更多是一种最终用户体验，而非开发者工具。
- **GPT-4o mini 用于长内容创作**：讨论了 **GPT-4o mini** 是否能生成 YouTube 脚本或故事等长内容。
   - 一名成员表示有兴趣将 API 用于此类任务，以整合 Text-to-Speech 甚至视频模型。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1264196613518131321)** (9 messages🔥): 

> - `Solving Mathematical or Logical Problems Accurately`
> - `Custom Instructions in ChatGPT`
> - `Using Custom Instructions Effectively`
> - `Guidance AI for Prompt Engineering` 


- **数学问题准确性问题**：用户讨论了一种逐步解决数学或逻辑问题的方法，但担心如果只给出最终答案会损失准确性。
- **修改先前回复时的 Custom Instructions 问题**：用户分享了 ChatGPT 重写整个文本而不是修改特定部分的问题，并寻求解决方案。
   - 另一名成员建议使用设置中的“Customize ChatGPT”部分，但该问题对该用户依然存在。
- **为 Custom Instructions 问题提供详细帮助**：建议分享 Custom Instructions、对话的共享链接以及所需更改的具体细节，以便进一步提供帮助。
- **用于 Prompt Engineering 的 Guidance AI**：用户询问了关于使用 Guidance AI 或其他工具进行 Prompt Engineering 的情况。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1264196613518131321)** (9 messages🔥): 

> - `Improving ChatGPT Accuracy`
> - `Modification of ChatGPT Responses`
> - `Custom Instructions for ChatGPT`
> - `Prompt Engineering Tools` 


- **处理带有精确答案的数学问题**：用户提出了关于为数学问题生成独立最终答案的担忧，但注意到该方法导致了准确率下降。
- **改进 ChatGPT 回复修改**：用户 Sparrow.hwk 寻求帮助，希望指示 ChatGPT 仅修改文本的特定部分，而不重复整个回复。
- **分享 Custom Instructions 以获得更好帮助**：Sparrow.hwk 尝试了 Custom Instructions，但反复遇到问题。
- **使用的 Prompt Engineering 工具**：用户询问其他人是否使用 Guidance AI 等工具进行 Prompt Engineering，或者是否使用替代方法。


  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1264026781984493653)** (1 messages): 

> - `Rankings page update`
> - `Infrastructure migration` 


- **排行榜页面因迁移变慢**：在周末迁移到新基础设施期间，排行榜页面的更新速度将变慢，并经常显示**陈旧数据**。
   - 用户应预料到此时间段内排行榜会出现延迟和不准确的情况。
- **基础设施迁移导致延迟**：通知指出，基础设施迁移将在周末特别影响**排行榜页面**。


  

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1264226677194620938)** (2 条消息): 

> - `GPTScript 的 OpenRouter 提供商`
> - `命令行上的 gptscript`
> - `gptscript 演示视频` 


- **GPTScript 的 OpenRouter 提供商现已发布**：一个新的 [GPTScript 的 OpenRouter 提供商](https://github.com/RobinVivant/gptscript-openrouter-provider) 已发布，并在 GitHub 上附有图片和详细说明。
   - 该工具对 GPTScript 应用程序的开发做出了重大贡献。
- **GPTScript 在命令行上的表现令人印象深刻**：讨论重点介绍了 [gptscript GitHub 仓库](https://github.com/gptscript-ai/gptscript)，该仓库以构建能直接与系统交互的 AI assistants 的能力而闻名。
   - 一位成员提到在仓库页面上有一个 *令人印象深刻的演示视频*。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/RobinVivant/gptscript-openrouter-provider">GitHub - RobinVivant/gptscript-openrouter-provider</a>: 通过在 GitHub 上创建账户，为 RobinVivant/gptscript-openrouter-provider 的开发做出贡献。</li><li><a href="https://github.com/gptscript-ai/gptscript">GitHub - gptscript-ai/gptscript: 构建与你的系统交互的 AI assistants</a>: 构建与你的系统交互的 AI assistants - gptscript-ai/gptscript
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1263970764978257933)** (202 条消息🔥🔥): 

> - `Hermes 2.5`
> - `GPTs Agents`
> - `OpenRouter Feature Requests` (功能请求)
> - `Model Merging` (模型合并)
> - `Dolphin Llama 70B` 


- **Dolphin Llama 70B 的性能问题**：在 0.8-1 的 Temperature 下使用 Dolphin Llama 70B，在 7k token 上下文的对话中会导致异常行为，产生混合了代码和无关内容的语无伦次的结果。
   - 另一位成员指出 Euryale 70B 的 fp8 量化模型也存在类似问题，暗示这些问题可能源于量化过程。
- **DeepSeek 的低成本与高效率**：DeepSeek v2 是一款拥有 236B 参数的 MoE 模型（每个 token 激活 21B 参数），因其出色的性能和每百万输入 token 0.14 美元的高性价比而受到赞誉。
   - *“DeepSeek 的定价非常有竞争力，而且似乎利润丰厚，”* 解释了他们采用高 Batch Size 和压缩技术的策略。
- **GPT-4o mini 的多语言能力与代码表现**：成员们讨论认为，**GPT-4o mini** 在编程任务中的表现不如 **DeepSeek**，但在多语言能力方面优于 **gemma2-27b**。
   - 一位成员指出：*“与 gemma2-27b 相比，4o mini 在多语言能力上似乎更好，但在推理能力上较弱。”*
- **Llama 3.1 405B 的泄露信息**：Llama 3.1 405B Base 版显然由于 HuggingFace 上的失误而提前泄露，引发了关于其通过 RoPE 缩放实现扩展上下文能力的讨论。
   - 成员们感到兴奋，期待能实现高效利用的软件更新，并渴望官方 Instruct 模型的发布。
- **免费版与付费版模型限制的问题**：一位用户发现，像 **google/gemma-2-9b-it:free** 这样的免费模型变体与对应的付费版本（8192 tokens）相比，具有更严格的 token 限制（4096 tokens）。
   - 这种差异导致了困惑和错误消息，引发了关于 token 限制执行方式中可能存在的误解或配置错误的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/models/google/gemma-2-9b-it:free">Google: Gemma 2 9B (free) by google</a>：Google 推出的 Gemma 2 9B 是一款先进的开源语言模型，为其尺寸级别的效率和性能设定了新标准。它专为各种任务而设计，为开发者赋能……</li><li><a href="https://openrouter.ai/docs/requests#images-_-multimodal-requests">Requests | OpenRouter</a>：处理输入和输出请求</li><li><a href="https://openrouter.ai/docs/limits">Limits | OpenRouter</a>：设置模型使用限制</li><li><a href="https://openrouter.ai/models?order=newest&supported_parameters=tools),">Models | OpenRouter</a>：在 OpenRouter 上浏览模型</li><li><a href="https://openrouter-3d-generator.vercel.app/browse">未找到标题</a>：未找到描述</li><li><a href="https://www.ben-evans.com/benedictevans/2024/7/9/the-ai-summer">The AI summer &mdash; Benedict Evans</a>：数亿人尝试过 ChatGPT，但大多数人没有再次使用。每家大公司都做了试点，但投入部署的却少得多。这其中有些只是时间问题。但……</li><li><a href="https://github.com/NolanGC/window-3d-demo">GitHub - NolanGC/window-3d-demo: Generative 3D in the web via window.ai, Next, Three, Neon, Drizzle, and GCS.</a>：通过 window.ai、Next、Three、Neon、Drizzle 和 GCS 在 Web 中生成 3D。- NolanGC/window-3d-demo</li><li><a href="https://openrouter.ai/docs/parameters">Parameters | OpenRouter</a>：配置请求参数</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e9hg7g/azure_llama_31_benchmarks/">Reddit - 深入探索</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1263940542023209132)** (156 条消息🔥🔥): 

> - `开源贡献`
> - `LLM 在 RPG 游戏中的应用`
> - `Nextjs app router 更新`
> - `开发者答疑时间 (Office Hours) 形式`
> - `集成 Cohere 模型` 


- **Toolkit 欢迎开源贡献**：社区成员强调了该工具包的开放性，鼓励他人构建、修改并为项目做出贡献。
   - “如果可以的话，我很乐意贡献力量，”一位成员表达了对在工具包上进行协作的热情。
- **在 RPG 游戏中使用 LLM**：Lyrcaxis 讨论了在其 AI 驱动的 RPG 项目中使用 LLM 进行分类、JSON 生成和对话生成。
   - 在将分类任务卸载到本地模型后，由于 CMD-R 具有严格的指令遵循能力，它成为了他们生成对话的首选。
- **Nextjs App Router 更新**：Hamedmp 对工具包及时更新到 Nextjs app router 表示赞赏，并对将组件整合到项目中感到兴奋。
   - 他们提到：“我仍然希望看到 AI SDK 对 Cohere 的更多支持，”并指出需要扩展工具调用（tool call）支持。
- **调整开发者答疑时间形式**：开发者答疑时间从 Discord 切换到 Twitter Spaces 的做法引发了讨论，主要担忧在于可访问性和平台依赖性。
   - 一位成员建议：“也许可以进行镜像同步，”主张结合 Discord 和 Twitter 形式以覆盖更广泛的受众。
- **将 Cohere 模型集成到写作工作中**：Petersmall 讨论了使用 Gemini 和 ChatGPT 进行叙事创作，并考虑添加基于 Cohere 的产品以增强输出效果。
   - 经过测试，他们发现来自 CMD-R 等 Cohere 产品的响应结果令人印象深刻，并能与其他 AI 工具形成互补。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.bloomberg.com/news/articles/2024-07-22/ai-startup-cohere-valued-at-5-5-billion-in-new-funding-round">Bloomberg - 你是机器人吗？</a>：未找到描述</li><li><a href="https://youtu.be/KFXeTHzIgf0">西部世界 S02E04 最后一个记得你的人</a>：只要还有一个记得你的人，你就还活着。——《西部世界》第二季第四集</li><li><a href="https://jsfiddle.net/razodactyl/2s60uyw5/">88 键钢琴 - JSFiddle - 代码沙箱</a>：未找到描述</li><li><a href="https://x.com/i/spaces/1eaJbaEkgdVGX]">来自 GitHub 的推文 - FixTweet/FxTwitter: 修复损坏的 Twitter/X 嵌入！</a>：在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能 - FixTweet/FxTwitter</li><li><a href="https://x.com/i/spaces/1eaJbaEkgdVGX">来自 GitHub 的推文 - FixTweet/FxTwitter: 修复损坏的 Twitter/X 嵌入！</a>：在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能 - FixTweet/FxTwitter
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1264014615055171596)** (4 条消息): 

> - `基于本地 LLM 的聊天 GUI`
> - `多人文字游戏 Discord 应用` 


- **令人兴奋的本地 LLM 聊天 GUI 项目**：一位成员分享了一个正在进行的由本地 **LLM** 驱动的聊天 GUI 项目，具有 Web 搜索、Python 解释器和图像识别等功能。他们提供了 [GitHub 仓库](https://github.com/yamikumo-DSD/chat_cmr) 以获取更多细节。
- **在 Discord 上创建并玩多人文字游戏**：一位成员介绍了一款 Discord 应用 **Command R+**，允许用户创建和玩多人文字游戏。



**提到的链接**：<a href="https://www.producthunt.com/posts/create-n-play"> Create &#x27;n&#x27; Play - 为你的 Discord 服务器创建 AI 多人文字游戏！ | Product Hunt</a>：AI 驱动的 Discord 机器人将你的服务器变成文字游戏天堂！使用我们的 /search-games 命令制作任何游戏。无限可能——它是终极的社区互动工具。让 AI 为你的游戏注入动力...

  

---

### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1264927720672858125)** (1 messages): 

> - `Developer Office Hours`
> - `Structured Generations in the API`
> - `Cohere Toolkit features`
> - `Cohere For AI research papers`
> - `Community events` 


- **Cohere Developer Office Hours 2.0 宣布**：Cohere 与 <@1199000650982379524> 主持了另一场 Developer Office Hours 会议，讨论了最近的产品更新。
   - 他们介绍了 API 中的新功能、Toolkit 更新以及最近的 **Cohere For AI** 研究论文，并邀请社区加入。
- **Cohere API 中的结构化生成 (Structured Generations)**：Cohere 宣布其模型 **Command R** 和 **Command R+** 现在可以生成 **JSON** 格式的结构化输出。
   - 此功能（文档见[此处](https://docs.cohere.com/docs/structured-outputs-json)）允许为下游应用提供更好的集成和数据分析。
- **2024 年 7 月新增 Cohere Toolkit 功能**：Cohere 和 **Fujitsu** 宣布建立战略合作伙伴关系，为日本市场提供新的企业级 AI 服务，该消息通过[博客](https://cohere.com/blog/toolkit-features-july-2024)发布。
   - 此次合作旨在增强 AI 服务在各种应用中的可访问性和性能。
- **即将举行的社区活动**：Cohere 提到了即将举行的社区活动，鼓励成员积极参与和互动。
   - 他们敦促大家带着问题参加会议，并一起喝杯咖啡交流。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.cohere.com/docs/structured-outputs-json">Structured Generations (JSON)</a>: 未找到描述</li><li><a href="https://cohere.com/blog/toolkit-features-july-2024">New Cohere Toolkit Features: Authentication, HTML and More</a>: 在 Cohere，我们继续助力开发者加速生成式 AI 应用的开发。我们正在扩展开源 Cohere Toolkit 的功能，引入 HTML 渲染、可配置项...</li><li><a href="https://x.com/cohere">来自 undefined 的推文</a>: 未找到描述
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1263981952600506389)** (13 messages🔥): 

> - `Softmax invariance and z loss`
> - `Understanding large language models (LLM)`
> - `GPT model example code`
> - `Scaling diarization pipelines`
> - `Vector quantization models` 


- **Softmax 具有平移不变性，需要 z loss**：一位成员指出 **Softmax** 是平移不变的，因此有必要使用 **z loss**。
   - *官方原因包括防止 Logits 偏离零点过远，并鼓励归一化的对数概率 (log-probabilities)*。
- **简易 GPT 模型解析**：一位成员分享了一个极简的 **GPT 模型代码** 以帮助理解其工作原理，其中包含 **LayerNorm**、**Linear layers** 和 **scaled dot product attention** 等组件。
   - [示例代码](https://github.com/alibaba/easydist/blob/3dbb146812fddf6259590a0b4611a251f3e7cbe5/benchmark/torch/model/gpt.py) 类似于 Alibaba 的 **easydist/benchmark/GPT model**。
- **扩展 Diarization 和矢量量化模型**：一位成员寻求关于扩展 **Diarization 流水线**和**矢量量化模型**的见解，考虑运行一个结合 **Whisper** 和 **Pyannote** 的慢速流水线来预训练一个 **LSTM**。
   - 他们还探索了训练一个模型，利用 **Perceiver IO** 作为编码器，从复合音频、关键帧和文本中生成统一的代码本 (codebooks)。



**提到的链接**：<a href="https://github.com/alibaba/easydist/blob/3dbb146812fddf6259590a0b4611a251f3e7cbe5/benchmark/torch/model/gpt.py#L112">easydist/benchmark/torch/model/gpt.py at 3dbb146812fddf6259590a0b4611a251f3e7cbe5 · alibaba/easydist</a>: 适用于多种生态系统的自动并行化系统和基础设施 - alibaba/easydist

  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1263949707126771752)** (39 messages🔥): 

> - `AI 中的自我建模 (Self-Modeling)`
> - `混合后量子加密 (Hybrid Post-Quantum Encryption)`
> - `神经网络中的特征污染 (Feature Contamination)`
> - `CNN 中的类人响应时间`
> - `使用 Switch SAE 进行高效字典学习` 


- **自我模型简化神经网络**：关于[自我模型](https://arxiv.org/abs/2407.10188)的论文表明，将预测其内部状态作为辅助任务的网络会变得更简单且更具正则化，从而提高参数效率。
   - 然而，在 MNIST、CIFAR 和 IMDB 等数据集上的结果符合预期，而非提供令人惊讶的见解，一些成员对这些发现的新颖性表示怀疑。
- **实现混合后量子加密**：[liboqs-python GitHub 仓库](https://github.com/qompassai/liboqs-python)描述了使用 Kyber/Dilithium 的 Python 绑定以及用于无根容器管理的 Podman 来实现混合后量子加密。
   - 这种方法旨在这一脆弱时期最小化攻击面，展示了安全加密技术的进步。
- **特征污染限制 OOD 泛化**：一篇关于[分布外泛化 (OOD Generalization)](https://arxiv.org/abs/2406.03345)的论文发现，神经网络受到特征污染的影响，即无关特征会阻碍其性能。
   - 讨论表明归纳偏置 (Inductive Biases) 和 SGD 动力学起着至关重要作用，暗示可能存在一个统一理论来解释模型失效。
- **RTNet 模拟人类响应时间**：[RTNet 模型](https://www.biorxiv.org/content/10.1101/2022.08.23.505015v2.full)通过利用随机决策，重现了人类在决策中的响应时间和信心。
   - 尽管该模型的实际应用尚存争议，但它通过使用采样的 Gaussian 权重，准确预测了人类在图像分类任务中的行为。
- **Switch SAE 用于高效字典学习**：提出了一种新架构 [Switch SAE](https://www.lesswrong.com/posts/47CYFbrSyiJE2X5ot/efficient-dictionary-learning-with-switch-sparse)，用于高效扩展稀疏自编码器 (Sparse Autoencoders)，旨在从超智能语言模型中恢复特征。
   - 利用条件计算 (Conditional Computation)，Switch SAE 为将 SAE 扩展到数十亿特征提供了一个实用的解决方案，克服了当前的计算限制。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2407.13623">Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies</a>：关于缩放大型语言模型 (LLM) 的研究主要集中在模型参数和训练数据大小上，忽略了词汇量大小的作用。直观地说，更大的词汇量能够实现更...</li><li><a href="https://arxiv.org/abs/2407.10188">Unexpected Benefits of Self-Modeling in Neural Systems</a>：几十年来，自我模型一直是人类认知研究以及最近机器学习研究中备受关注的话题。然而，自我模型究竟能带来什么好处？在这里我们展示了当人工...</li><li><a href="https://arxiv.org/abs/2406.03345">Feature Contamination: Neural Networks Learn Uncorrelated Features and Fail to Generalize</a>：学习在分布偏移下泛化的表示对于构建鲁棒的机器学习模型至关重要。然而，尽管近年来取得了重大进展，算法上的进步...</li><li><a href="https://arxiv.org/abs/2208.10291">Efficient Planning in a Compact Latent Action Space</a>：基于规划的强化学习在离散和低维连续动作空间的任务中表现出强大的性能。然而，规划通常会带来显著的计算开销...</li><li><a href="https://github.com/qompassai/liboqs-python">GitHub - qompassai/liboqs-python: A Qompass fork of open-quantum-safe/liboqs-python</a>：open-quantum-safe/liboqs-python 的 Qompass 分支。通过在 GitHub 上创建账户为 qompassai/liboqs-python 的开发做出贡献。</li><li><a href="https://www.biorxiv.org/content/10.1101/2022.08.23.505015v2.full">RTNet: A neural network that exhibits the signatures of human perceptual decision making</a>：卷积神经网络目前提供了生物视觉的最佳模型。然而，它们的决策行为，包括它们是确定性的且使用相同数量的计算...</li><li><a href="https://www.lesswrong.com/posts/47CYFbrSyiJE2X5ot/efficient-dictionary-learning-with-switch-sparse">Efficient Dictionary Learning with Switch Sparse Autoencoders — LessWrong</a>：作为 ML Alignment &amp; Theory Scholars 项目 - 2024 年夏季批次的一部分产出 …
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1263950155401662464)** (3 条消息): 

> - `Scaling laws 与 Hypernetworks`
> - `大脑与 Backprop`
> - `跨 Parameterizations 与 Optimizers 的 Scaling Exponents` 


- **受 Scaling laws 约束的 Hypernetworks**：一位用户讨论了 Hypernetworks 因 Scaling laws 面临的约束，认为为了让 Hypernetwork 达到目标误差，其规模必须 **小于 O(scaling_law(output_model_compute)(target_error))**。
   - 该用户提到，为了让 Hypernetworks 有效，预测神经网络的任务必须更简单，或者输出模型的 Scaling law 必须 **“非常理想（really nice）”**。
- **大脑仅是 Backprop 的近似**：有观点认为 **大脑仅是 Backprop 的近似**，这暗示了在传统 Backpropagation 之外存在替代学习机制的可能性。
- **跨 Parameterizations 与 Optimizers 的 Scaling Exponents**：一条 [推文](https://x.com/main_horse/status/1810647037718999342) 讨论了一篇关于跨不同 Parameterizations 和 Optimizers 的 Scaling Exponents 的论文，涉及 **10,000 多个模型**，涵盖了不同的 Optimizers、模型大小和 Parameterizations。
   - 主要发现包括 **O(1/n) LR schedule 表现优于 mUP**，在所有测试配置中成功的 hparam 迁移，以及提出了一种 Adam-atan2 Optimizer，以避免 Adam 中出现的梯度欠载（gradient underflow）问题。



**提到的链接**：<a href="https://x.com/main_horse/status/1810647037718999342">来自 main (@main_horse) 的推文</a>：Scaling Exponents Across Parameterizations and Optimizers [GDM] [nocode/weights] https://arxiv.org/abs/2407.05872 训练了 10,000+ (!) 个模型，涵盖了 * Optim (SGD/Adam/Adafactor) * 模型大小 (1.1B ~...

  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1263938275144499312)** (9 messages🔥): 

> - `MATS 7.0 Applications`
> - `**nnsight** Paper Release`
> - `Apollo's Mech Interp Projects List`
> - `Tokengrams Project`
> - `Suffix Arrays for Pile` 


- **MATS 7.0 Applications Open Now**: **Neel Nanda** 和 **Arthur Conmy** 已经开启了他们冬季 MATS 7.0 streams 的申请，截止日期为 8 月 30 日。
   - Nanda 强调了该计划在培养 Mechanistic Interpretability 研究方面的独特价值，并提供了[公告链接](https://x.com/NeelNanda5/status/1813921161052635209)和[录取文档](https://tinyurl.com/neel-mats-app)。
- **New Mech Interp Paper: **nnsight****：一篇名为 **nnsight** 的新论文即将发布在 arXiv 上。
   - 预计它将涵盖 Mechanistic Interpretability 方面的重大进展。
- **Apollo Shares 45 New Mech Interp Project Ideas**：Apollo Research 在最近的一篇 [Alignment Forum 帖子](https://www.alignmentforum.org/posts/KfkpgXdgRheSRWDy8/a-list-of-45-mech-interp-project-ideas-from-apollo-research)中分享了 45 个 Mechanistic Interpretability 项目想法。
   - 该帖子引发了关于如何使小型语言模型（Small Language Models）对可解释性研究更有用的[讨论](https://docs.google.com/document/d/1XRb-EDDw-h6c-L6pKltI9A8zJmufE71I2_4wx8d5rMg/edit?usp=sharing)。
- **Tokengrams Project Inquiry**：用户询问了如何为 Tokengrams 项目获取 Pile 和 Dolma 等数据集的 `document.bin` 文件。
   - 一名团队成员确认即将发布 Pile 数据集分片的 Suffix Arrays，未来还会考虑其他数据集。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/NeelNanda5/status/1813921161052635209">来自 Neel Nanda @ ICML (@NeelNanda5) 的推文</a>：你对 @ch402 风格的 Mechanistic Interpretability 研究感兴趣吗？我正寻求通过 MATS 指导学者 - 请在 8 月 30 日前申请！我对比往届学者的工作印象深刻，并且热爱...</li><li><a href="https://tinyurl.com/neel-mats-app">Neel Nanda / Arthur Conmy MATS 7.0 Stream - 录取流程 + FAQ</a>：Neel Nanda / Arthur Conmy MATS Stream - 录取流程 + FAQ。如何申请：填写 MATS 通用申请表（少于 10 分钟）。截止日期为太平洋时间 8 月 30 日星期五晚上 11:59。请注意，这是一个特殊的...</li><li><a href="https://www.alignmentforum.org/posts/KfkpgXdgRheSRWDy8/a-list-of-45-mech-interp-project-ideas-from-apollo-research),">来自 Apollo Research 可解释性团队的 45+ 个 Mech Interp 项目想法列表 — AI Alignment Forum</a>：我们制作此列表的原因：• Apollo Research 的可解释性团队最近完成了几个项目 [1]。为了决定我们将开展什么工作……</li><li><a href="https://docs.google.com/document/d/1XRb-EDDw-h6c-L6pKltI9A8zJmufE71I2_4wx8d5rMg/edit?usp=sharing)**!">[社区草案] 扩展 Tinystories</a>：用于训练可解释语言模型的改进版短篇简单故事数据集。我们正在解决什么问题？TinyStories 是一个广受欢迎的数据集，包含约 200 万个由模型生成的简单短篇故事...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1263933785179820093)** (56 messages🔥🔥): 

> - `Zeno Upload Feature` (Zeno 上传功能)
> - `Commit Branch Queries` (提交分支查询)
> - `Logging Issues` (日志记录问题)
> - `Multinode Inference Support` (多节点推理支持)
> - `PSA: ICML Conference` (公告：ICML 会议)


- **Zeno 上传功能困惑**：一位成员在 `visualize_zeno.py` 中使用 Zeno 上传功能时遇到问题，特别是 `get_latest_filename` 函数。
   - 另一位成员建议使用 `main` 分支而非 `big-refactor` 分支，并使用 `pip install -e .` 进行正确安装。
- **提交分支版本澄清**：关于 `pyproject.toml` 中的版本号显示为 0.4.3，而 README 宣称为 0.4.4 的情况存在困惑。
   - 成员们一致认为需要检查这一差异，有人建议更新 README 并添加新的 FAQ 文档。
- **日志信息未打印**：一位成员报告称 `lm-eval-harness` 中的 `eval_logger.info` 语句没有打印，而 `print` 语句却正常工作。
   - 确认了安装方式为可编辑模式（editable），并建议检查 logger 配置。
- **大模型的多节点推理**：有人提问 eval harness 是否支持大模型在 2 个节点上进行分片推理。
   - 讨论提到 Open LLM Leaderboard 团队的一个 PR 可能会实现此功能，且使用支持节点间 PP 的 `vllm` 可能是一个有效的解决方案。
- **公告：参加 ICML 会议**：发布了一项公告，称部分团队成员将参加 ICML，可能会延迟 PR 审查。
   - 鼓励成员在频道中联系或在 ICML 现场会面，以便获得更快的响应和讨论。



**提及的链接**：<a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/2008">Refactor API models by baberabb · Pull Request #2008 · EleutherAI/lm-evaluation-harness</a>：此 PR 为 API 请求模型引入了一个新的超类，提供了：下游类的模块化、用于请求转换的可重载方法、API 请求和响应解析、Tokenization...

  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1264480868668280884)** (4 messages): 

> - `Latent downscaling` (Latent 下采样)
> - `Image classification performance` (图像分类性能)
> - `Generated latents` (生成的 Latents)


- **Latent 下采样的挑战**：一位用户讨论了直接对 latents 应用操作的挑战，并建议在编码前对图像进行下采样。
- **基于 ImageNet 的生成 Latents**：一位用户提到创建了一个生成版本的 **ImageNet**，生成器的原生分辨率为 **128x128x4**，分类器为 **64x64x4**，通过图像缩放（对比朴素的 latent 缩放）实现了 **20%** 的性能提升。
   - 他们正在探索针对 latents 的高性价比方法，以获得类似的分类性能收益。


  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1264299956324405330)** (19 messages🔥): 

> - `Nemotron-340B 细节`
> - `Nathan 为 Nemotron-340B 转换提供的悬赏`
> - `vLLM 多节点推理`
> - `Nemotron-340B 的多节点性能`
> - `评估框架 (Evaluation harness) 讨论` 


- **Nathan 为 Nemotron-340B 的 HF 转换提供悬赏**：[Nathan](https://x.com/natolambert/status/1814735390877884823) 提供初始 75 美元的悬赏，旨在将 **Nemotron-340B-Instruct 转换为 HuggingFace** 格式，并支持 **FP8 量化** 和多节点 HF 实现。
   - 随着合成数据社区捐赠者的加入，悬赏金额现已增长至 **2,000 美元以上**。
- **讨论 Nemotron-340B 的架构独特性**：成员们讨论认为 **Nemotron-340B** 基本上是标准架构，只有少数独特组件，如 **sqrelu** 和 **自定义 rope pct**。
   - Hailey 指出，“如果我们有一个可以轻松加载模型的设置，我可以毫不费力地将其添加到 vLLM 中”。
- **vLLM 多节点推理的可行性**：Hailey 讨论了 **vLLM 多节点推理** 的可行性，但不确定其性能，表示“我实际上不知道 vLLM 多节点性能如何。我觉得可能很差？”。
   - Stella 指出，目前不存在一个既好又易于运行的多节点推理设置，且对于大多数现有硬件来说并不特别合理。
- **多节点性能和测试挑战**：小组承认支持 Nemotron 架构和高效运行多节点是交织在一起的问题，这表明缺乏用于测试的多节点环境。
   - Tastybucketofrice 认为该设置是可行的，并在看到悬赏增加后评论道：“我今晚就搞定它”。
- **智力诱导 (Nerdsniping) 与评估框架**：Catboy_slim_ 对“琐碎工作 (scutwork)”表现出兴趣，并讨论将“防作弊评估 (uncheatable eval)”集成到评估框架中。
   - Baber_ 幽默地指出：“一旦你把它加入框架，它就不再是防作弊的了”。



**提及的链接**：<a href="https://x.com/natolambert/status/1814735390877884823?s=46">Nathan Lambert (@natolambert) 的推文</a>：我正在提供有偿悬赏，以成功将 nvidia/Nemotron-4-340B-Instruct 转换为 HuggingFace / 相关库。初始奖励 75 美元。我们非常需要这个来解锁许可的合成数据……

  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1264294243862581278)** (61 messages🔥🔥): 

> - `Nemotron-4-340B 转换为 HuggingFace`
> - `Llama-3 和 3.1 泄露`
> - `Meta AI 潜在的付费服务`
> - `大模型蒸馏技术`
> - `HuggingFace 的 SOC2 合规性` 


- **Nemotron-4-340B 转换为 HuggingFace**：[Nathan Lambert](https://x.com/natolambert/status/1814735390877884823) 正在为将 **nvidia/Nemotron-4-340B-Instruct** 转换为 HuggingFace 提供有偿悬赏，初始捐赠总额为 75 美元。
   - 目标是解锁许可的合成数据并启用蒸馏项目，这需要 FP8 量化和多节点实现。
- **Llama-3 和 3.1 泄露引发关注**：关于 **Llama-3 405b** 和 **Llama 3.1** 模型的传闻和泄露（包括基准测试和潜在功能）被广泛讨论，链接指向特定的 [Azure GitHub 基准测试](https://github.com/Azure/azureml-assets/pull/3180/files) 和社区 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1e9hg7g/azure_llama_31_benchmarks/)。
   - 泄露的基准测试显示 **Llama 3.1** 在多个领域优于 GPT-4（HumanEval 除外），引发了关于其潜在优越性的讨论。
- **Meta AI 潜在的付费服务**：有推测称 **Llama 405B** 可能是 Meta AI 付费服务的一部分，代码片段和 [Testing Catalog 的推文](https://x.com/testingcatalog/status/1815439546722451493?s=46) 暗示了这一点。
   - 一个可能的 Meta AI API 平台 AI Studio 也被提及，围绕即将到来的 7 月 23 日发布会引发了热议。
- **HuggingFace 的 SOC2 合规性担忧**：讨论强调 HuggingFace 的 **SOC2 合规性** 可能会导致一些问题，但未提供具体细节。
   - Nathan Lambert 对 HuggingFace 拥有 SOC2 合规性表示惊讶，认为这可能是导致延迟或复杂性的原因。
- **关于 Llama 3.1 蒸馏的讨论**：受 **Llama 3.1** 潜在影响的启发，Nathan Lambert 考虑撰写一篇关于蒸馏技术的文章。
   - 有推测认为 Llama 3.1 的大部分性能提升可能归功于蒸馏方法，类似于 **Gemma 2**。


<div class="linksMentioned">

<strong>提及的链接</strong>：

</div>

<ul>
<li>
<a href="https://www.bloomberg.com/news/articles/2024-07-22/ai-startup-cohere-valued-at-5-5-billion-in-new-funding-round">Bloomberg - Are you a robot?</a>: 未找到描述</li><li><a href="https://x.com/TheXeophon/status/1815317803164971469">来自 Xeophon (@TheXeophon) 的推文</a>: o7</li><li><a href="https://x.com/testingcatalog/status/1815439546722451493?s=46">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>: 除此之外，Llama 405B 似乎可能成为 Premium 服务的一部分，在这种情况下，Meta AI Premium 也可能在 7 月 23 日发布（在代码中发现）。此外，还提到了 AI St...</li><li><a href="https://x.com/danielhanchen/status/1814752981725946227?s=46">来自 Daniel Han (@danielhanchen) 的推文</a>: @AlpinDale 未经证实的传言，但 fp8 是从 fp16 转换而来的，所以希望 fp16 会发布 & Llama-3 405b 只是 LLM。文本 + 图像功能延迟到秋季 & 针对 8B 和 70B 的 Llama 3.1 更新（基础版 + ...</li><li><a href="https://x.com/gblazex/status/1815441807385252024?s=46">来自 Blaze (Balázs Galambosi) (@gblazex) 的推文</a>: 这是 405B 对比 GPT-4 Omni => Llama 几乎在所有方面都表现更好，除了 HumanEval（这不是最好的编码基准测试）。最显著的是 MMLU STEM，但请注意 4-Turbo 在这方面会更接近。</li><li><a href="https://x.com/natolambert/status/1814735390877884823">来自 Nathan Lambert (@natolambert) 的推文</a>: 我正悬赏寻求将 nvidia/Nemotron-4-340B-Instruct 成功转换到 HuggingFace / 相关库的方法。初始奖励 $75。我们确实需要这个来解锁合成许可数据...</li><li><a href="https://x.com/gblazex/status/1815426702425928118?s=46">来自 Blaze (Balázs Galambosi) (@gblazex) 的推文</a>: LLama 3.1 基准测试结果在 Azure 的 GitHub 账号上泄露，包括 405B、70B、8B。来源：https://github.com/Azure/azureml-assets/pull/3180/files。发现者：https://www.reddit.com/r/LocalLL...</li><li><a href="https://x.com/kalomaze/status/1815305220118769952?s=46">来自 kalomaze (@kalomaze) 的推文</a>: 噢，那是 llama3.1-405b 前一天凌晨 3 点在 4chan 上泄露的。</li><li><a href="https://www.joelonsoftware.com/2002/06/12/strategy-letter-v/">战略信函 V</a>: 上大学时我修了两门经济学入门课程：宏观经济学和微观经济学。宏观经济学充满了诸如“低失业率导致通货膨胀”之类的理论，但从未真正站得住脚……</li><li><a href="https://x.com/morqon/status/1815118198985101444?s=46">来自 morgan — (@morqon) 的推文</a>: 扎克伯格的大日子</li><li><a href="https://x.com/JoeBiden/status/1815080881981190320">来自 Joe Biden (@JoeBiden) 的推文</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e9hg7g/comment/leedpl3">Reddit - 潜入任何领域</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1264381727124488324)** (4 条消息): 

> - `WizardMath paper`
> - `Instruction Reward Model (IRM)`
> - `PRM by Uesato et al 2022 paper`
> - `Step-by-step reward labeling`
> - `UltraChat vs Zephyr paper` 


- **WizardMath 中的指令奖励模型引发好奇**：[WizardMath 论文](https://arxiv.org/abs/2308.09583) 介绍了 **Instruction Reward Model (IRM)**，它对指令质量进行评分，并利用该评分来影响 PPO 奖励。
   - 一位用户质疑基于指令质量进行条件化是否有价值，并想知道其他地方是否也使用了类似的想法。
- **推理步骤的二元 vs 分类 vs 标量奖励**：一位用户对比了 Uesato 等人 2022 年论文中采用二元奖励的 PRM 系统，以及《Let's Verify Step by Step》论文中采用分类标签（正面、负面、中性）的系统。
   - 他们质疑为什么研究人员在模型训练中会为推理步骤选择不同的奖励系统。
- **关于 UltraChat 数据质量的辩论**：一位用户注意到 Zephyr 论文将 UltraChat 数据从 **150万** 大幅过滤至 **20万**，并征求关于 UltraChat 生成过程的意见。
   - 他们将 **UltraChat** 的自顶向下方法与来源多样的种子文档方法（例如 WRAP/Rephrasing the Web）进行了比较，质疑哪种方法更有效。
- **UltraChat 令人惊讶的有效性**：一位用户对 **UltraChat** 在经过大幅过滤和生成过程审查后依然表现出的有效性表示惊讶。
   - 另一位用户承认了这一评论，但没有进一步阐述。


  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1264299609992335500)** (3 条消息): 

> - `ICML 2024 Spotlight`
> - `文献综述相关担忧`
> - `对 Harvey Legal AI 的批评` 


- **ICML 2024 庆祝 Faithfulness Measurable Models**: **Andreas Madsen** 宣布了他们在 ICML 2024 上的 Spotlight 论文，介绍了一种[新的可解释性方法](https://x.com/peterbhase/status/1814692347407429706?s=46)：Faithfulness Measurable Models，号称在不增加成本的情况下，解释效果提升了 2-5 倍，且具有准确的可信度指标。
   - 一位用户指出，这与他们 2021 年的 NeurIPS 论文非常相似，强调了在提交和评审过程中改进 **文献综述 (literature reviews)** 的必要性。
- **法律 AI 公司 Harvey 被抨击为“故弄玄虚”**: 针对 [法律 AI 公司 Harvey](https://x.com/emilyinvc/status/1814724166593225050?s=46) 的一条评论预测其将会失败，将其斥为“故弄玄虚 (smoke and mirrors)”。
   - *Emilyinvc* 直言不讳地预测，Harvey 最终将成为“高速公路边被撞死的动物”。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/peterbhase/status/1814692347407429706?s=46">来自 Peter Hase (@peterbhase) 的推文</a>: 这篇 ICML 2024 Spotlight 论文的 GIF 几乎完全描述了我们 2021 年在 NeurIPS 发表的论文（以及 Vafa 等人 2021 年在 EMNLP 发表的论文）。祝贺 Andreas 和其他人完成这项工作（我认为这是一个好主意）...</li><li><a href="https://x.com/emilyinvc/status/1814724166593225050?s=46">来自 emily is in sf (@emilyinvc) 的推文</a>: 现在就放话：harvey（那家法律 AI 公司）最终会成为高速公路边被撞死的动物。完全是一家故弄玄虚的公司。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1263943111210700934)** (52 条消息🔥): 

> - `采访邀请`
> - `MosaicML 的赠剑传统`
> - `Claude AI 文本限制`
> - `奖励模型创新`
> - `Stripe 设置问题` 


- **Lambert 反思采访策略**: Lambert 考虑联系 **Karpathy**、**Sebastian Raschka** 和 **Gary Marcus** 等知名人士进行采访，并提到由于政治因素，他不愿联系 **Andreessen**。
   - 他还提到了涉及 **Ross Tayler** 和 **Andrew Trask** 的后续计划，并对与 **HuggingFace** 的潜在合作表示兴奋。
- **MosaicML 的赠剑传统逐渐取消**: 讨论透露，随着人力资源部门的成立，**MosaicML** 赠送宝剑的传统已逐渐取消，转向更职业化的规范。
   - 一条幽默的笔记提到，据称 **Databricks** 法务团队的成员收到了宝剑，Lambert 还开玩笑说要引入 **Interconnects 宝剑**。
- **Claude AI 面临文本限制**: 成员们讨论了 **Claude AI** 拒绝处理某些文本的情况，特别是像 **“我有一个梦想” ('I Have a Dream')** 演讲这样的神圣文本。
   - 一种解决方法是利用 API 中的 pre-filling responses（预填充回复）来克服使用受限文本的问题。
- **讨论奖励模型 (Reward Model) 的创新**: 成员们分享了链接并讨论了 [Absolute-Rating Multi-Objective Reward Model](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1)，该模型号称采用了 Mixture-of-Experts (MoE) 聚合技术。
   - 讨论中还提到了 **RewardBench** 的挑战以及奖励模型 (RMs) 尚处于起步阶段。
- **Stripe 设置中的电话号码问题**: Lambert 正在处理一个由于 Stripe 设置导致其电话号码出现在收据上的问题。
   - 他开玩笑说要换成 **Google Voice 号码**，并讨论了虚拟信箱等替代方案。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1">RLHFlow/ArmoRM-Llama3-8B-v0.1 · Hugging Face</a>: 无描述</li><li><a href="https://x.com/soldni/status/1695087021520457939">来自 Luca Soldaini 🎀 (@soldni) 的推文</a>: 感谢 @DippedRusk 拍到了我在自然栖息地（办公室座位）拿着 @MosaicML 宝剑的样子</li><li><a href="https://github.com/project-numina/aimo-progress-prize">GitHub - project-numina/aimo-progress-prize</a>: 为 project-numina/aimo-progress-prize 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[nlp](https://discord.com/channels/1179127597926469703/1208183200099344445/1265052470636187790)** (2 条消息): 

> - `关于蒸馏的博客文章`
> - `Lilian Wang`
> - `对资源匮乏感到惊讶` 


- **寻求关于蒸馏 (distillation) 的博客文章**: 一位成员询问是否有人推荐关于 **蒸馏 (distillation)** 的博客文章。
- **对缺少 Lilian Wang 的作品感到惊讶**: 另一位成员对该主题缺少 **Lilian Wang** 撰写的全面博客文章表示惊讶，并暗示目前还没有 2 万字级别的深度文章。

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1264116021552218192)** (2 条消息): 

> - `Yitay 关于模型架构的博客文章`
> - `Encoder vs. Encoder-Decoder 模型`
> - `LLMs 的演进`
> - `@srush_nlp 的推文` 


- **Yitay 推出模型架构系列文章**：[Yitay 的博客文章](https://www.yitay.net/blog/model-architecture-blogpost-encoders-prefixlm-denoising) 讨论了从 **BERT** 等 Encoder 模型向 LLMs 新趋势的转变。
   - 他旨在向对语言和 NLP 感兴趣的人更新模型架构的演进，并引用了 [@srush_nlp 一条已删除的推文](https://x.com/srush_nlp/status/1779938508578165198)。
- **Encoder 模型都去哪了？**：**Yitay** 探讨了为什么尽管 BERT 及其类似的 Encoder 模型取得了成功，但扩展这些模型却不再受到青睐。
   - 他计划在一系列博客文章中深入探讨这一话题，从链接中的入门文章开始。



**提到的链接**：<a href="https://www.yitay.net/blog/model-architecture-blogpost-encoders-prefixlm-denoising">What happened to BERT &amp; T5? On Transformer Encoders, PrefixLM and Denoising Objectives &mdash; Yi Tay</a>：模型架构系列博客第 1 部分：BERT 和 T5 怎么了？关于 Transformer Encoders、PrefixLM 和 Denoising 目标的思考

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1263971600932540416)** (74 条消息🔥🔥): 

> - `Langfuse vs Langsmith`
> - `GPT-4o mini 与 AI 生成内容`
> - `关于 Harvey AI 的传闻`
> - `Elon Musk 的孟菲斯超级集群 (Memphis Supercluster)`
> - `LLaMA 3.1 泄露与评估` 


- **Langfuse 表现优于 Langsmith**：来自用户的轶事反馈表明，[Langfuse](https://github.com/langfuse/langfuse) 的表现优于 Langsmith，用户分享了关于其易于自托管和集成的积极体验。
   - 创始人 *Clemo_._* 鼓励更多的社区互动，强调他们致力于维护一个优秀的 OSS 解决方案。
- **GPT-4o Mini 助力 AI 生成内容**：OpenAI 的新 [GPT-4o mini 模型](https://batchmon.com/blog/ai-cheaper-than-ads/) 每 100 万输入 token 的成本为 0.15 美元，这使得创建完全由广告支持的动态 AI 生成内容成为可能。
   - 讨论包括对网络内容的潜在影响，假设将向更多 AI 生成的输出转变。
- **Harvey AI 的传闻与预测**：关于 [Harvey AI](https://x.com/emilyinvc/status/1814741780010844289?s=46) 可行性的传闻和怀疑浮出水面，有人称其为一家“故弄玄虚的公司”。
   - 随后引发了关于垂直领域 AI 初创公司面临挑战的辩论，包括对大型 AI 实验室的依赖以及行业的当前周期。
- **Elon Musk 的孟菲斯超级集群 (Memphis Supercluster)**：Elon Musk 宣布启动孟菲斯超级集群，声称它是世界上最强大的 AI 训练集群，在单个 RDMA fabric 上拥有 10 万个液冷 H100。
   - 然而，[事实核查](https://x.com/dylan522p/status/1815494840152662170?s=46) 揭示了在功耗和 GPU 可用性方面的差异，表明该设施尚未完全投入运营。
- **LLaMA 3.1 泄露引发关注**：泄露的 [LLaMA 3.1](https://x.com/mattshumer_/status/1815444612414087294?s=46) 评估结果表明，其 8B、70B 和 405B 模型甚至在 instruct tuning 之前就可能优于当前的 state-of-the-art 模型。
   - 这些泄露引发了对开源 AI 模型未来能力的广泛期待和推测。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/emilyinvc/status/1814724166593225050?s=46">来自 emily is in sf (@emilyinvc) 的推文</a>：现在就放话：Harvey（这家法律 AI 公司）最终会成为公路边的路毙动物。完全是一家虚张声势的公司。</li><li><a href="https://share.snipd.com/snip/686d53b1-5696-427f-bb8d">Snipd — 突出显示并分享播客中的精彩瞬间</a>：未找到描述</li><li><a href="https://x.com/brunokoba_/status/1814893302698926326?s=46">来自 Bruno Koba (@brunokoba_) 的推文</a>：过去几周我一直在深入调查垂直领域 AI 初创公司。我认为我们正处于 AI 初创公司融资/发展的周期中一个非常奇怪的阶段。一些想法（犀利观点？）...</li><li><a href="https://www.bloomberg.com/news/articles/2024-07-22/ai-startup-cohere-valued-at-5-5-billion-in-new-fu">彭博社 - 你是机器人吗？</a>：未找到描述</li><li><a href="https://x.com/dylan522p/status/1815494840152662170?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Dylan Patel @ ICML (@dylan522p) 的推文</a>：Elon 在撒谎。目前从电网获取的功率为 7MW，约 4k 个 GPU。如果 http://X.ai 最终与 Tennessee Valley Authority 签署协议，8 月 1 日将获得 50MW。150MW 的变电站...</li><li><a href="https://www.bloomberg.com/news/articles/2024-07-22/ai-startup-cohere-valued-at-5-5-billion-in-new-funding-round">彭博社 - 你是机器人吗？</a>：未找到描述</li><li><a href="https://x.com/nisten/status/1814558806770172181?s=46">来自 nisten (@nisten) 的推文</a>：既然你们真的非常想知道实际的完整 Prompt... 唉.. 给你们，拿去研究吧。</li><li><a href="https://x.com/maccaw/status/1815435539669283204?s=46">来自 Alex MacCaw (@maccaw) 的推文</a>：如果这是真的，世界即将改变。</li><li><a href="https://x.com/sarahookr/status/1815360812787380701?s=46">来自 Sara Hooker (@sarahookr) 的推文</a>：越大总是越好吗？🐘 认为 Scaling 比任何其他因素更能推动进步的想法已被正式定名为“惨痛教训”（Bitter Lesson）。Sutton 是对的吗？📜https://arxiv.org/abs/2407.05694v...</li><li><a href="https://x.com/emilyinvc/status/1814741780010844289?s=46">来自 emily is in sf (@emilyinvc) 的推文</a>：来自一位“使用”Harvey 的大型律所合伙人的私信。引用 emily is in sf (@emilyinvc) 的话：现在就放话：Harvey（这家法律 AI 公司）最终会成为公路边的路毙动物。完...</li><li><a href="https://x.com/latentspacepod/status/1815411709085143197">来自 Latent.Space (@latentspacepod) 的推文</a>：🆕 AI 寒冬之风。氛围已经改变。关于 @leopoldasch 对阵 Sequoia、Goldman Sachs、@benedictevans、@cpaik 的《软件的终结》，以及为什么 AI Engineer 能超越这一切。未来已...</li><li><a href="https://share.snipd.com/snip/e0e81b28-78d1-430b-831b-1ff1b76b58a8">AI 带来的效率提升将被竞争抵消 | 来自 No Priors 的 1 分钟剪辑：人工智能 | 技术 | 初创公司</a>：来自 Sarah Guo 和 Elad Gil 讨论 AI 如何开启新市场并影响初创公司现状的 1 分钟剪辑 | No Priors: Artificial Intelligence | Technolo…</li><li><a href="https://x.com/mattshumer_/status/1815444612414087294?s=46">来自 Matt Shumer (@mattshumer_) 的推文</a>：泄露的（可能是真的？）Llama 3.1 评测。是 Base 模型，不是 Instruct。开源即将达到 SOTA —— 甚至 70B 都优于 GPT-4o，而且这还是在 Instruct Tuning 之前，之后应该会更...</li><li><a href="https://share.snipd.com/snip/686d53b1-5696-427f-bb8d-7e22ef157558">前沿 AI 的专有模型：成本视角 | 来自 Grit 的 2 分钟剪辑</a>：来自 Together AI 首席执行官兼联合创始人 Vipul Ved Prakash 与 Bucky Moore 的第 200 期：超级周期 | Grit</li><li><a href="https://x.com/tszzl/status/1814787334166224962?s=46">来自 roon (@tszzl) 的推文</a>：@Teknium1 @dionysianyawp @sama 我可以肯定地说，并以我的名誉担保这不是真的。AI 的进展目前快得惊人。</li><li><a href="https://share.snipd.com/snip/53b18e90-8408-42ea-8824-e2ea17faf693">生成式 AI 核心运营的昂贵本质 | 来自 Grit 的 1 分钟剪辑</a>：来自 Together AI 首席执行官兼联合创始人 Vipul Ved Prakash 与 Bucky Moore 的第 200 期：超级周期 | Grit</li><li><a href="https://x.com/cohere/status/1815377543182303410?s=46">来自 cohere (@cohere) 的推文</a>：我们很高兴宣布 D 轮融资，以加速增长、扩大团队，并开发下一代前沿、企业级、专注于数据隐私的 AI 技术。我们正在带来高...</li><li><a href="https://x.com/alpindale/status/1814814551449244058?s=46">来自 Alpin (@AlpinDale) 的推文</a>：已确认有 8B、70B 和 405B 版本。前两个是从 405B 蒸馏出来的。128k（十进制为 131k）上下文。405B 画不出独角兽。Instruct 版本可能进行了安全对齐。架构...</li><li><a href="https://www.economist.com/business/2024/06/13/a-price-war-breaks-out-among-chinas-ai-model-builders">中国 AI 模型构建者之间爆发价格战</a>

在中国 AI 模型构建者中：它可能会阻碍创新</a></li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e98zrb/llama_31_405b_base_model_available_for_download/">Reddit - 深入探讨任何事物</a>：未找到描述</li><li><a href="https://x.com/elonmusk/status/1815325410667749760?s=46">来自 Elon Musk (@elonmusk) 的推文</a>：@xAI 团队、@X 团队、@Nvidia 及支持公司做得好，孟菲斯超级集群（Memphis Supercluster）在当地时间凌晨 4:20 左右开始训练。通过在单个 RDMA fabric 上部署 10 万块液冷 H100，它是最...</li><li><a href="https://batchmon.com/blog/ai-cheaper-than-ads/">由广告支付的 AI —— gpt-4o mini 转折点</a>：随着 OpenAI 最新发布的 gpt-4o mini，AI 比以往任何时候都更便宜。现在的 AI 如此廉价，甚至比平均广告展示成本还要低。</li><li><a href="https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/">TLDR：使用 LLM 构建的一年 —— D-Squared</a>：未找到描述</li><li><a href="https://news.ycombinator.com/item?id=41010188">由广告支付的 AI —— GPT-4o mini 转折点 | Hacker News</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 条消息): 

swyxio: 大型的每月回顾已上线：https://x.com/latentspacepod/status/1815411709085143197
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1263948175635513405)** (29 条消息🔥): 

> - `音频问题`
> - `布局检测`
> - `Texify vs Mathpix`
> - `演示反馈`
> - `模型训练数据集` 


- **音频问题困扰会议**：几位成员报告了音频问题，**nuvic_** 的声音没人能听到，而 **vikas.p** 对其他成员来说声音很清晰。
   - 这个问题并非个例，**slono** 观察到 Discord 上不同的人都出现了类似问题。
- **揭秘布局检测**：讨论集中在布局检测的机制上，推测其是否基于具有大量训练数据的经典目标检测。
   - 成员们赞赏了在布局检测和阅读顺序模型背景下对任务分解的解释。
- **Texify 对比 Mathpix**：有人提出了关于 **Texify** 在性能和使用方面与 **Mathpix** 相比如何的问题。
   - 该查询没有得到直接的对比，但引起了人们对这两种工具所使用的独特方法的兴趣。
- **演示获得高度赞扬**：一位成员表示“整个演示太棒了 🤯 👏”，表明对该环节的高度认可。
   - 会议在与会者的热烈好评和感谢中结束。
- **关于训练数据集的疑问**：成员们对训练数据集的创建感到好奇，询问阅读顺序模型的标签是手动的还是启发式的（heuristic）。
   - 提供的解释受到了好评，并得到了成员们的积极认可。



**提到的链接**：<a href="https://github.com/VikParuchuri">VikParuchuri - 概览</a>：VikParuchuri 有 90 个可用的仓库。在 GitHub 上关注他们的代码。

  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1263973960295645214)** (64 条消息🔥🔥): 

> - `训练 Qwen2-7b`
> - `用于知识图谱的 Triplex`
> - `Mistral 12b 问题`
> - `LLaMA 3 不一致性`
> - `LLaMA 3.1 基准测试` 


- **Qwen2-7b 训练设置问题**：一位成员询问了关于训练 **Qwen2-7b** 的配置，寻求关于使用合适设置的指导。
   - *具体来说，用户询问是否有人拥有运行 `dolphin-2.9.2-qwen2-7b` 的配置。*
- **Triplex 将知识图谱成本降低 98%**：SciPhi 的新 [Triplex 模型](https://huggingface.co/SciPhi/Triplex) 将知识图谱的构建成本降低了 98%，以极低的成本超越了 **GPT-4**。
   - 该模型可以从非结构化数据中提取三元组（triplets），并支持本地运行，使知识图谱更加易于获取且成本更低。
- **Mistral 12b 面临 Tokenizer 问题**：多位成员报告了 **Mistral 12b** 的严重问题，特别是其 Tokenizer 输出的文本没有空格。
   - 尽管 Loss 指标看起来不错，但 **输出被认为是“垃圾”**，这表明存在可能与特殊 Token 相关的未解决问题。
- **LLaMA 3.1 EOS Token 问题**：发现 Hugging Face 上 LLaMA 3 仓库的 **EOS token** 配置存在差异，导致了严重问题。
   - EOS token 设置错误，一位成员提供了更新后的 Token 配置以解决此问题。
- **LLaMA 3.1 基准测试令社区印象深刻**：成员们对 **LLaMA 3.1** 的基准测试印象深刻，特别注意到 8B 和 70B 模型的强劲表现。
   - 405B 模型表现也很好，但 70B 模型被认为**紧随领先模型之后**。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct/discussions/8">NousResearch/Meta-Llama-3-8B-Instruct · 匹配官方 tokenizer_config.json。更改是在此仓库创建后进行的。</a>：未找到描述</li><li><a href="https://huggingface.co/SciPhi/Triplex">SciPhi/Triplex · Hugging Face</a>：未找到描述</li><li><a href="https://r2r-docs.sciphi.ai/cookbooks/knowledge-graph">知识图谱 - 最好的开源 AI 驱动问答引擎。</a>：未找到描述
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1264054648663183470)** (22 条消息🔥): 

> - `知识蒸馏支持`
> - `DPO 改进`
> - `DeepSpeed Zero-3 兼容性` 


- **成员讨论添加知识蒸馏支持**：成员们集思广益，讨论为预分词（pretokenized）数据集添加 **知识蒸馏（knowledge distillation）** 支持，特别是专注于为每个 input id 提供 logits 而不是 labels。
   - *他们考虑了在 HF Trainer 限制下的可行性，并指出 TRL 可能不会开箱即用地提供兼容的 Trainer。*
- **关于合并 DPO 改进的对话**：成员们开始合并 **DPO 改进**，并讨论了与更新后的 chat_template 功能保持一致的必要性。
   - [DPO 增强功能](https://github.com/axolotl-ai-cloud/axolotl/pull/1725) 可能需要深入研究 TRL 的 DPO Trainer，以实现更精细的控制，例如在 Loss 计算期间掩盖（masking out）查询部分。
- **成员解决了 DeepSpeed Zero-3 的兼容性问题**：一位用户在使用 DeepSpeed Zero-3 时遇到了 **ValueError**，提示与 `low_cpu_mem_usage=True` 和自定义 `device_map` 不兼容。
   - 通过删除 **accelerate config** 解决了该问题，从而使设置能够无错运行。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/axolo">axolo - 概览</a>：逆境通向繁荣。axolo 拥有 32 个公开仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/axolotl-ai-cloud/axolotl/pull/1756">由 Tostino 提交的“为 `chat_template` 数据集训练添加灵活配置选项” · Pull Request #1756 · axolotl-ai-cloud/axolotl</a>：引入 roles_to_train 参数以按角色设置训练标签；添加 train_on_eos 选项以配置在序列结束 Token 上进行训练；在数据集中实现每条消息的训练配置...
</li>
</ul>

</div>
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1264005864730267728)** (5 messages): 

> - `Axolotl training error`
> - `GPU memory issues`
> - `Batch size adjustment`
> - `Mixed precision training`
> - `NCCL timeouts` 


- **Axolotl 训练遇到 GPU 显存瓶颈**：成员询问了 Axolotl 训练期间出现错误的原因，**Phorm** 回复称这很可能是由于 GPU 显存不足（OOM）引起的，这在大模型或大 Batch size 的情况下很常见。
   - *Phorm* 建议了故障排除步骤，例如减小 Batch size、调整梯度累积步数（gradient accumulation steps）以及使用混合精度训练（Mixed precision training）来缓解该问题。
- **解决 NCCL 超时以实现更顺畅的训练**：针对分布式训练中的 NCCL 超时问题，**Phorm** 引导用户参考 [NCCL 指南](https://github.com/openaccess-ai-collective/axolotl/tree/main/docs/nccl.qmd) 以获取详细的故障排除建议。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/openaccess-ai-collective/axolotl/tree/main/README.md#L562L614)">axolotl/README.md at main · axolotl-ai-cloud/axolotl</a>: 尽管提出关于 axolotl 的问题。通过在 GitHub 上创建账号来为 axolotl-ai-cloud/axolotl 的开发做出贡献。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=15dfc26f-b460-49e5-ae58-0ffd7039cc47)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1264269426392432732)** (57 条消息🔥🔥): 

> - `LangChain.js Token 问题`
> - `关于 LLM 经验教训的视频和博客文章`
> - `LangChain 初学者指南`
> - `Vector store 过滤`
> - `关于将 RAG 应用部署到生产环境的讨论` 


- **LangChain.js Token 问题仍然存在**：一位用户询问 4omini 的 LangChain.js Token 是否仍然无法正常工作。
   - 频道中未提及具体的回复或解决方案。
- **关于 LLM 经验教训的新视频和博客文章**：一位成员分享了一个 [视频和博客文章](https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/)，总结了使用 LLM 构建应用一年的经验教训，并建议观众观看视频以获得更好的理解。
   - 该文章总结了由六位从业者编写的三部分系列文章，重点关注战术、运营和战略见解。
- **LangChain 初学者指南**：一位用户分享了一篇 [Medium 文章](https://medium.com/@ambaliaharshit25/ea17820a5c01)，提供了对 LangChain 及其组件的入门级介绍。
   - 该文章旨在解释为什么要使用这些组件及其重要性，并使用了像《钢铁侠》中的 JARVIS 这样贴近生活的例子。
- **Vector store 中的过滤应用**：多位用户讨论了在 LangChain 中使用 MongoDB Atlas VectorStore 时如何应用过滤器，并提供了详细的代码片段。
   - 还解释了自定义 retriever 的方法以及如何将它们与 EnsembleRetriever 集成。
- **将 RAG 应用部署到生产环境**：一位成员分享了一篇关于使用 MongoDB Atlas 和 LangChain 构建 RAG 实现的 [教程](https://www.mongodb.com/docs/atlas/atlas-vector-search/ai-integrations/langchain/)。
   - 该教程涵盖了环境搭建、数据存储、创建搜索索引以及运行向量搜索查询。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://medium.com/@ambaliaharshit25/ea17820a5c01">LangChain components. Why and How?</a>: 作为完全的初学者和我一起学习 LangChain。这篇博客不仅关于“是什么”，还关于“为什么”以及“如何”利用 LangChain 的组件。</li><li><a href="https://js.langchain.com/v0.2/docs/integrations/vectorstores/mongodb_atlas/#metadata-filtering>).">MongoDB Atlas | 🦜️🔗 Langchain</a>: 仅在 Node.js 上可用。</li><li><a href="https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/">TLDR: 1 year of building with LLMs &#8211; D-Squared</a>: 未找到描述</li><li><a href="https://python.langchain.com/v0.2/docs/integrations/vectorstores/google_vertex_ai_vector_search/#use-vector-store-as-retriever>)">Google Vertex AI Vector Search | 🦜️🔗 LangChain</a>: 此 Notebook 展示了如何使用与 Google Cloud Vertex AI Vector Search 向量数据库相关的功能。</li><li><a href="https://github.com/langchain-ai/langchain/issues/14227>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://www.mongodb.com/docs/atlas/atlas-vector-search/ai-integrations/langchain/">arrow-right</a>: 未找到描述</li><li><a href="https://github.com/langchain-ai/langchain/issues/17464>),">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/19885>),">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/2095>),">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/14227>).">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1264009384497778749)** (6 条消息): 

> - `Triplex 模型`
> - `Embedding 的向量可视化`
> - `使用 LangChain 进行语义搜索`
> - `面向 TypeScript 的 AI 函数构建器`
> - `LangChain 教程` 


- **Triplex 大幅降低知识图谱构建成本**：SciPhi.AI 最新开源的 [Triplex](https://huggingface.co/SciPhi/Triplex) 将知识图谱的构建成本降低了 98%，性能超越 GPT-4，而成本仅为后者的 1/60。
   - Triplex 是 Phi3-3.8B 的微调版本，能够从非结构化数据中提取三元组（triplets），增强了如微软 Graph RAG 等 RAG 方法。
- **在 NLP 中使用图表可视化向量**：一个新的 [GitHub 项目](https://github.com/rajatasusual/realtime-vector-embeddings) 旨在帮助在图表上可视化向量，使理解 NLP 任务中的 Embedding 变得更加容易。
   - *图表比方程式更容易理解*，并有助于文本分类、聚类和推荐系统。
- **使用 LangChain 的语义搜索教程**：[Substack 上的一篇新博客文章](https://sonamcoffeenlp.substack.com/p/semantic-search-to-glean-valuable-deb) 深入探讨了如何使用 LangChain、Cohere LLM 和 ApertureDB 实现语义搜索。
   - 作者描述了使用 Cohere 的 Command R+ 实现聊天模块的过程，并鼓励读者提供反馈。
- **面向 TypeScript 的 AI 驱动函数构建器**：在一次黑客松中开发了一个名为 [AI Fun](https://github.com/mishushakov/ai-fun) 的新项目，用于为 TypeScript 构建 LLM 驱动的函数。
   - 该项目利用 AI 自动化并简化了 TypeScript 函数的构建过程。
- **使用 Composio 和 LangChain 创建调度 Agent**：分享了一份[详细指南](https://git.new/scheduler)，介绍如何创建一个调度 Agent (Scheduler Agent)，利用 Composio、LangChain 和 ChatGPT 根据电子邮件安排事件。
   - 该指南旨在赋能用户利用这些工具进行更高效的任务管理。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/SciPhi/Triplex">SciPhi/Triplex · Hugging Face</a>：未找到描述</li><li><a href="https://git.new/scheduler">composio/python/examples/scheduler_agent at master · ComposioHQ/composio</a>：Composio 为 Agent 配备了精心设计的工具，使其能够处理复杂任务 - composio/python/examples/scheduler_agent at master · ComposioHQ/composio</li><li><a href="https://github.com/mishushakov/ai-fun">GitHub - mishushakov/ai-fun: 面向 TypeScript 的 LLM 驱动函数构建器</a>：面向 TypeScript 的 LLM 驱动函数构建器。通过在 GitHub 上创建账户为 mishushakov/ai-fun 做出贡献。</li><li><a href="https://github.com/rajatasusual/realtime-vector-embeddings">GitHub - rajatasusual/realtime-vector-embeddings: 该项目旨在理解不同文本片段在多维空间中的相似程度。这是自然语言处理 (NLP) 任务（如文本分类、聚类和推荐系统）中的核心概念。</a>：该项目旨在理解不同文本片段在多维空间中的相似程度。这是自然语言处理 (NLP) 任务中的核心概念...</li><li><a href="https://medium.com/@ambaliaharshit25/ea17820a5c01">LangChain 组件：为什么以及如何使用？</a>：作为完全的初学者和我一起学习 LangChain，这篇博客不仅关于“是什么”，还关于我们“为什么”以及“如何”利用 LangChain 的组件。
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1264844424622637178)** (3 条消息): 

> - `Harshit Ambalia 在 Medium 上发表的 LangChain 文章`
> - `部署 RAG 应用的指南`
> - `使用 Composio、LangChain 和 ChatGPT 的调度 Agent 指南` 


- **LangChain 初学者友好文章发布**：一位用户分享了一篇关于 LangChain 及其组件的 [Medium 文章](https://medium.com/@ambaliaharshit25/ea17820a5c01)，旨在帮助有兴趣了解其应用的初学者。
   - *想象一下，拥有一个可以通过简单的自然语言指令处理复杂任务的虚拟助手*，文章深入探讨了为什么这些组件如此重要。
- **使用 Composio 的调度 Agent 指南**：一位用户发布了一个 [GitHub 指南](https://git.new/scheduler)，介绍如何利用 Composio、LangChain 和 ChatGPT 创建一个调度 Agent，根据收到的电子邮件安排事件。
   - 该指南提供了详细的步骤，用户鼓励其他人尝试并为该仓库点赞（Star）。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://git.new/scheduler">composio/python/examples/scheduler_agent at master · ComposioHQ/composio</a>：Composio 为 Agent 配备了精心设计的工具，使其能够处理复杂任务 - composio/python/examples/scheduler_agent at master · ComposioHQ/composio</li><li><a href="https://medium.com/@ambaliaharshit25/ea17820a5c01">LangChain components. Why and How?</a>：作为完全的初学者和我一起学习 LangChain，这篇博客不仅关于“是什么”，还关于我们“为什么”以及“如何”利用 LangChain 的组件。
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1263980830934700042)** (34 条消息🔥): 

> - `sdxl vae latents`
> - `HF 托管优势`
> - `新的 BUD-E 演示`
> - `Linux 终端上的本地 LLM`
> - `Kolors 扩散模型` 


- **使用 sdxl vae latents 节省存储空间**：**Ramimmo** 讨论了使用 **sdxl vae latents** 来降低大型图像数据集存储成本的潜在好处。
   - *Puffy310* 提出了对版权影响的担忧，但 *Nodja* 澄清说，潜空间数据压缩并不能免除版权法的约束。
- **HF 支付存储费用，尽管使用**：**Thejonasbrothers** 建议将数据集上传到 **Hugging Face**，因为他们承担了 **S3 存储成本**，这使其成为一种具有成本效益的解决方案。
   - *Unknown* 指出，尽管如此，HF 仍然保持盈利，暗示其财务管理非常高效。
- **在 YouTube 上观看新的 BUD-E 演示**：**Spirit_from_germany** 在 YouTube 上分享了 **BUD-E** 语音助手的[新演示](https://youtu.be/O4IXfa8CROs)。
   - 该演示邀请社区加入他们的 Discord，共同构建这个助手。
- **Kolors 模型在 3090 上运行**：**Spirit_from_germany** 询问了在 NVIDIA 3090 上运行 **Kolors 扩散模型**的情况。
   - *Segmentationfault8268* 确认了其兼容性，并推荐使用 **ComfyUI** 工作流以及 **Int8 精度**以获得最佳性能。
- **寻找适用于 Linux 终端的本地 LLM**：由于 AMD 显卡的限制，**Alexiosthesixth** 正在寻找一种可以通过 CPU 在 Linux 终端中运行的**本地 LLM**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/Kwai-Kolors/Kolors-diffusers">Kwai-Kolors/Kolors-diffusers · Hugging Face</a>：未找到描述</li><li><a href="https://youtu.be/O4IXfa8CROs">BUD-E - Demo</a>：加入我们的 Discord 社区，亲自尝试 BUD-E，并帮助我们构建我和 BUD-E 在视频中讨论的语音助手：https://discord.gg/sTKSB2AwBvhttps...
</li>
</ul>

</div>
  

---

### **LAION ▷ #[announcements](https://discord.com/channels/823813159592001537/826154622644649985/1264865683662045226)** (1 条消息): 

> - `Bud-E voice assistant`
> - `Daily Online-Hackathons`
> - `BUD-E Discord Server` 


- **Bud-E 展示了具有开源目标的新 Demo**：分享了 **Bud-E 语音助手** 的 Demo，展示了一个愿景：在未来，每个人都能以仅相当于电费的成本，获得高性能的开源系统。
   - 目前针对 Ubuntu 优化的代码库将进行重构，以实现客户端、服务器以及可互换的 ASR、TTS、LLM 组件之间的清晰分离。
- **加入 BUD-E Discord 服务器进行协作**：邀请志愿者加入新的 **BUD-E Discord 服务器**，帮助进一步开发语音助手，并贡献类似于 Minecraft Mods 的新技能。
   - 每日在线黑客松（Daily Online-Hackathon）会议将在每天 CEST 时间晚上 9 点举行，以引导新志愿者并协调项目工作。
- **BUD-E 开发的每日在线黑客松正式启动**：**从今天（7 月 22 日星期一）开始**，将在 CEST 时间晚上 9 点举行每日在线黑客松会议，提供项目概览、引导志愿者并协调项目进度。
   - 这些会议将在 Discord 的专用语音频道中进行：[https://discord.gg/nMexRzbJ3W](https://discord.gg/nMexRzbJ3W)。


  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1264387914712875089)** (14 条消息🔥): 

> - `Plotting Loss Curves`
> - `Mem0 AI Memory`
> - `Datadog Time Series Modeling`
> - `Research Recruitment` 


- **切回使用 Epochs 绘制 Loss 曲线**：一位成员最初使用墙上时钟时间（wall-clock time）绘制 Loss 曲线，但发现衡量模型学习效率更有意义，随后后悔并切回了 Epochs。
   - 该成员提到使用 **WandB** 可以轻松实现此目的，但承认之前的改变是错误的，是一个“愚蠢”的决定。
- **Mem0 为 LLM 引入智能记忆层**：[Mem0](https://docs.mem0.ai/overview) 发布了一个针对 LLM 的记忆层，通过用户、会话和 AI Agent 记忆以及自适应个性化等功能，实现个性化的 AI 体验。
   - 有关集成和功能的更多信息，请查看 Mem0 的 [GitHub 页面](https://github.com/mem0ai/mem0)。
- **Datadog 发布时间序列建模的 SOTA 结果**：一位成员分享了 Datadog 已发布关于时间序列建模的 [SOTA 结果](https://www.datadoghq.com/blog/datadog-time-series-foundation-model/)，并正在积极招聘研究职位。
   - Datadog 的基础模型旨在通过识别趋势、解析高频数据和管理高基数（high-cardinality）数据来有效处理时间序列数据。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.mem0.ai/overview">📚 Overview - Mem0.ai</a>: 未找到描述</li><li><a href="https://www.datadoghq.com/blog/datadog-time-series-foundation-model/">Introducing Toto: A state of the art time series foundation model by Datadog</a>: 介绍 Toto（即 Time Series Optimized Transformer for Observability），这是 Datadog 开发的 SOTA 时间序列预测基础模型，我们在 1 万亿个数据点上对其进行了训练。</li><li><a href="https://github.com/mem0ai/mem0">GitHub - mem0ai/mem0: The memory layer for Personalized AI</a>: 个性化 AI 的记忆层。通过在 GitHub 上创建账号来为 mem0ai/mem0 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1263946279734480986)** (8 条消息🔥): 

> - `使用 PostgresML 进行 Reranking`
> - `LLMs as Judges`
> - `Merlinn: 开源 On-call Copilot`
> - `使用 Ollama 和 Qdrant 构建 Multimodal RAG`
> - `Deasie RAG 工作坊` 


- **使用 PostgresML Reranking 增强结果**：使用 [PostgresML](https://t.co/HWfitT0CJt) 进行 reranking，只需增加一两个参数，即可显著提升搜索结果的相关性。
   - [博客](https://t.co/HWfitT0CJt)上的一篇客座文章详细介绍了这种托管索引（managed index）方法的工作原理。
- **生产环境中的 LLMs as Judges**：由 [Yixin Hu 和 Thomas Hulard](https://t.co/i84Cg5pqsy) 主讲的视频展示了如何利用 LLMs as judges 将应用程序投入生产。
   - *本次会议涵盖了开发过程中 RAG 评估背后的关键概念和实践。*
- **Merlinn: AI 驱动的 On-call Copilot**：[Merlinn](https://t.co/rAM5OOxQ34) 是一款开源的、由 LLM 驱动的 Slack 助手，用于监听并解决生产事故。
   - *它与 Datadog 等可观测性和事故管理工具集成。*
- **使用 Ollama 和 Qdrant 构建 Multimodal RAG**：[Pavan Mantha](https://t.co/0gcz4GfCh5) 撰写的一篇入门文章解释了如何使用 Ollama 和 Qdrant 搭建 Multimodal RAG 应用。
   - 文章详细介绍了如何通过文本转录摄取音频/视频源，并对多模态数据进行索引。
- **通过高级解析和元数据改进 RAG**：与 Deasie 联合创始人举办的工作坊讨论了如何通过高级解析和元数据改进 RAG，[YouTube 上已提供录像](https://t.co/cJPsNaWgoc)。
   - 关键要点包括：解析和元数据都能显著增强 RAG 性能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://t.co/HWfitT0CJt">Improving Vector Search - Reranking with PostgresML and LlamaIndex — LlamaIndex, Data Framework for LLM Applications</a>：LlamaIndex 是一个简单、灵活的数据框架，用于将自定义数据源连接到大语言模型 (LLMs)。</li><li><a href="https://t.co/vTV3t8daqT">TiDB Future App Hackathon 2024</a>：创新并创建令人惊叹的 AI 应用程序
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1263948982380531723)** (33 messages🔥): 

> - `llama-parse API 问题`
> - `ReActAgent 最大迭代次数`
> - `VectorStoreIndex 嵌入模型`
> - `LlamaIndex 网络研讨会`
> - `从 PDF 中提取图片` 


- **llama-parse API 显示乱码输出**：用户报告了 llama-parse API 产生乱码输出的问题，其中包含类似 `:->|>11_NaN<|<-:` 的符号。
   - 一位成员建议重新运行任务并分享 JobID 以便检查日志。
- **ReActAgent 达到最大迭代次数**：一位成员在使用带有 retriever 的 ReActAgent 时遇到了 `ValueError: Reached max iterations` 错误。
   - 建议增加 `max_iterations` 的值，但也引发了关于 Agent 可能陷入死循环的担忧。
- **为 VectorStoreIndex 指定自定义嵌入模型**：用户希望在 VectorStoreIndex 中使用自定义嵌入模型，而不是默认调用 OpenAI。
   - 解决方案包括全局设置 `Settings.embed_model` 为自定义模型，或者在初始化期间直接传递模型。
- **最近的 LlamaIndex 网络研讨会已在 YouTube 上线**：一位成员询问在哪里可以找到最新的 LlamaIndex 网络研讨会录像。
   - 该研讨会已在 [LlamaIndex YouTube 频道](https://youtu.be/V_-WNJgTvgg?si=_b5qNd3gM6NXRfWy)发布，题为“通过高级解析 + 元数据提取改进 RAG”。
- **使用 llama-parse 从 PDF 中提取图片**：用户询问如何使用 LlamaIndex 从 PDF 中提取图片及相应的标题。
   - 建议使用 llama-parse 的 JSON 模式来返回图像，并利用多模态 LLMs 进行进一步处理。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/V_-WNJgTvgg?si=_b5qNd3gM6NXRfWy">LlamaIndex Webinar: Improving RAG with Advanced Parsing + Metadata Extraction</a>: 在此视频中，我们与 Deasie 的联合创始人（Reece, Leonard, Mikko）共同举办了一场关于通过高级解析和元数据改进 RAG 的研讨会。数据处理...</li><li><a href="https://github.com/carolinedlu/llamaindex-chat-with-streamlit-docs/blob/main/streamlit_app.py?ref=blog.streamlit.io">llamaindex-chat-with-streamlit-docs/streamlit_app.py at main · carolinedlu/llamaindex-chat-with-streamlit-docs</a>: 构建一个由 LlamaIndex 驱动的聊天机器人，使用 Streamlit 文档（或您自己的数据）增强 GPT 3.5。 - carolinedlu/llamaindex-chat-with-streamlit-docs</li><li><a href="https://github.com/run-llama/llama_parse">GitHub - run-llama/llama_parse: Parse files for optimal RAG</a>: 为优化 RAG 解析文件。通过在 GitHub 上创建账号为 run-llama/llama_parse 的开发做出贡献。</li><li><a href="https://docs.unstructured.io/open-source/concepts/document-elements#elements-coordinates)">Document elements and metadata - Unstructured</a>: 未找到描述内容。
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1263982167613116498)** (4 条消息): 

> - `视频和音乐数据的 ETL`
> - `LLM 中的 SycoPhancy（阿谀奉承效应）`
> - `用于 RAG Pipeline 的 Korvus` 


- **探索非结构化数据的新 ETL 方法**：一位用户提到了 Jerry Liu 关于[新型 ETL](https://www.youtube.com/watch?v=imlQ1icxpBU) 的讨论，旨在使视频和音乐数据能被 LLM 理解。
   - 他们认为文本和 PDF 已经可以得到很好的处理，但好奇社区是否在其他类型的非结构化数据上取得了类似成果。
- **分析 LLM 中的 SycoPhancy**：一位用户分享了一篇 [LinkedIn 文章](https://www.linkedin.com/posts/subham-kundu-2746b515b_llm-knowledgesharing-evaluation-activity-7220695691021455361-72PE)，详细分析了 LLM 中的 **SycoPhancy** 概念，希望能为社区提供见解。
- **Korvus 用于简化 RAG Pipeline**：一位成员好奇 **Korvus** 是否真的能在不牺牲质量的情况下简化 RAG Pipeline。
   - 他们提供了一个 **Korvus** 的 [GitHub 链接](https://github.com/postgresml/korvus)，这是一个基于 PostgreSQL 构建的搜索 SDK，它将整个 RAG Pipeline 统一在单个数据库查询中。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=imlQ1icxpBU">Jerry Liu - 什么是 LlamaIndex、Agents 以及给 AI 工程师的建议</a>：在本期节目中，我们与 LlamaIndex 的创始人 Jerry Liu 进行了交谈。LlamaIndex 是一个专为 LLM 开发设计的尖端 Python 框架...</li><li><a href="https://github.com/postgresml/korvus">GitHub - postgresml/korvus: Korvus 是一个搜索 SDK，它将整个 RAG Pipeline 统一在单个数据库查询中。基于 Postgres 构建，支持 Python、JavaScript、Rust 和 C 的绑定。</a>：Korvus 是一个搜索 SDK，它将整个 RAG Pipeline 统一在单个数据库查询中。基于 Postgres 构建，支持 Python、JavaScript、Rust 和 C 的绑定。 - postgresml/korvus
</li>
</ul>

</div>
  

---



### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1263946422554722479)** (44 条消息🔥): 

> - `GPT4o-mini 模型的问题`
> - `DSPy Tracing 发布`
> - `TypedPredictors 兼容性`
> - `DSPy 论文发布`
> - `DSPy 中 Optimizers 的可靠性` 


- **GPT4o-mini 模型的问题**：一位成员报告称，与 **GPT3.5-turbo** 相比，**GPT4o-mini** 过于冗长，并重复了示例的大部分结构，这影响了数据提取 Pipeline。
- **DSPy Tracing 发布增强工作流**：新的 **DSPy tracing** 功能现已推出，可高效跟踪所有模块、预测、LM 和检索器（[文档链接](https://docs.langwatch.ai/integration/python/guide#capturing-llm-spans)）。
- **TypedPredictors 与复杂 Pydantic 类的挑战**：一位成员指出，只有 **GPT-4o** 和 **Sonnet-3.5** 能成功处理复杂的 Pydantic 类生成，而其他模型则失败了。
- **DSPy 论文验证联合优化方法**：新发表的[论文](https://x.com/lateinteraction/status/1815423177272824022)显示，在 Prompt 优化和微调（finetuning）之间交替进行，比单一方法可带来高达 **26% 的提升**。
- **讨论 DSPy Optimizers 的可靠性**：成员们讨论了 DSPy optimizers 的可靠性，指出 **BootstrapFewShotWithRandomSearch** 是一个简单且可靠的起点。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/stanford-futuredata/lotus">GitHub - stanford-futuredata/lotus</a>：为 stanford-futuredata/lotus 的开发做出贡献。</li><li><a href="https://x.com/lateinteraction/status/1815423177272824022">Omar Khattab (@lateinteraction) 的推文</a>：🚨在为任务构建 LM 系统时，你应该探索微调还是 Prompt 优化？与 @dilarafsoylu @ChrisGPotts 合作的论文发现你应该两者兼顾！新的 DSPy optimizers 交替进行优化...</li><li><a href="https://arxiv.org/abs/2407.10930">微调与 Prompt 优化：两个强强联手的步骤</a>：自然语言处理 (NLP) 系统越来越多地采用多阶段 Pipeline 的形式，涉及多个不同的语言模型 (LM) 和 Prompt 策略。在这里，我们探讨了...</li><li><a href="https://docs.langwatch.ai/integration/python/guide#capturing-llm-spans">Python 集成指南 - LangWatch</a>：未找到描述
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1263973288133132348)** (8 messages🔥): 

> - `OpenPilot 模型运行分析`
> - `Tinygrad 中的 Bitcast 功能`
> - `有前景的 Pull Requests`
> - `Tinygrad 每周会议` 


- **分析 OpenPilot 模型运行性能**：George Hotz 分享了一个 [14.64 ms OpenPilot 模型运行的 trace](https://gist.github.com/geohot/8d7edc7ac2fd9a31ea563c134b66cddb)，并概述了记录 kernel 更改及其潜在减速的步骤。
   - 他强调这项任务对于任何有技术背景的人来说都是可以上手的，但指出初学者经常在没有经过彻底思考的情况下提问。
- **辩论 bitcast 形状一致性**：Tyoc213 提出了一个问题，即 Tinygrad 中的 `bitcast` 函数是否应该与 TensorFlow 的 `bitcast` 保持一致，特别是考虑到形状差异。
   - George Hotz 和另一位成员一致认为匹配 TensorFlow/Torch/Numpy 是有意义的，Tyoc213 承诺将跟进所需的更改。
- **最有前景的 PR 获得认可**：George Hotz 称赞了 [tyoc213 的一个 PR](https://github.com/tinygrad/tinygrad/compare/master...tyoc213-contrib:tinygrad:tyoc213/bitcast-all)，称其为他见过的最有前景的 PR，并指出其中包含了预期的测试。
   - Tyoc213 对此表示感谢，并提到计划检查其他框架以进一步保持一致。
- **Tinygrad 每周会议要点**：Chenyuy 分享了周一会议的议程，包括 tinybox 的更新、hcopt 速度恢复以及 [MCTS 搜索改进](https://github.com/tinygrad/tinygrad/blob/master/extra/mcts_search.py)。
   - 其他亮点包括更好的搜索功能、conv backward fusing、快速 Llama 改进，以及针对 kernel 和 driver 增强的各种 bounty。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://stats.tinygrad.org>,">未找到标题</a>：未找到描述</li><li><a href="https://github.com/tinygrad/tinygrad/compare/master...tyoc213-contrib:tinygrad:tyoc213/bitcast-all">比较 tinygrad:master...tyoc213-contrib:tyoc213/bitcast-all · tinygrad/tinygrad</a>：你喜欢 pytorch？你喜欢 micrograd？你一定会爱上 tinygrad！❤️ - 比较 tinygrad:master...tyoc213-contrib:tyoc213/bitcast-all · tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/pull/3740/files">由 obadakhalili 提交的支持形状更改 bitcast 的 Pull Request #3740 · tinygrad/tinygrad</a>：解决了 #3422 的第一部分：目前无法在具有不同 itemsize 的 dtype 之间进行 bitcast
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1263940950254686269)** (11 messages🔥): 

> - `组合 LazyBuffers`
> - `Shapetrackers 教程`
> - `Tinygrad 相对于 PyTorch 的可行性` 


- **关于组合 LazyBuffers 的辩论**：讨论了 **组合 lazybuffers** 以及 **srcs** 和 **base** 如何形成一棵树，从而引发了关于序列的想法。
   - 成员们将其与 **PyTorch 的 layout/view 系统**进行了比较，但指出 **Tinygrad** 的系统似乎更强大且依赖于 **shapetracker**。
- **Shapetrackers 教程实现直播**：一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=g1rCrv1fx1A)，关于实现 **Shapetrackers** 教程并重点关注 view merging（视图合并）。
   - 该视频在 **tinygrad fork** 上提供了详细的代码演练，引导观众了解优化技术。
- **Tinygrad 对 PyTorch 用户可行性的质疑**：成员们辩论了 **tinygrad** 作为 PyTorch 替代方案的可行性，其中一位成员正在考虑切换。
   - 提出了是等待 **1.0** 版本还是继续使用 **0.9** 版本的问题，反映了对生产力的担忧。



**提到的链接**：<a href="https://www.youtube.com/watch?v=g1rCrv1fx1A">Tinygrad 教程 - shapetrackers 和视图合并 - 机器学习优化</a>：带有代码教程的 tinygrad fork：https://github.com/Zaffer/tinygrad/tree/tuturial-notebook tinygrad 文档：https://docs.tinygrad.org/ tinygrad 笔记：https://m...

  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1264054041751851071)** (4 条消息): 

> - `Crowdstrike update`
> - `Python subinterpreters`
> - `Meta Llama 3.1` 


- **Crowdstrike 入职首日更新**：[Vinceflibustier](https://fixupx.com/vinceflibustier/status/1814233715641389456) 分享了一个关于他在 Crowdstrike 第一天工作的轻松更新，提到他推送了一个小更新并休了下午假。他以一个和平手势表情符号 ✌️ 结束了消息。
- **探索 Python subinterpreters**：一位成员分享了关于 [Python subinterpreters 的教程](https://realpython.com/python312-subinterpreters/?utm_source=perplexity)，涵盖了即将发布的 Python 3.12 以及 Python 3.13 的变更预览，重点强调了在 GIL 控制和并行性方面的增强。
   - 该教程提供了关于 Python subinterpreters、CPython 全局状态变更以及后续版本中潜在增强功能的见解，[建议熟悉 Python 基础知识和 GIL](https://realpython.com/learning-paths/python-basics/)。
- **Meta Llama 3.1 仓库泄露**：AlpinDale 确认 Meta Llama 3.1 包含一个由 405B 模型蒸馏而成的 8B 和 70B 模型，具有 128k context，并且奇怪地无法画出独角兽。
   - [AlpinDale 的帖子](https://x.com/alpindale/status/1814814551449244058?s=12)指出 405B 的 instruct tuning 可能进行了 safety aligned（安全对齐），且该仓库被意外提前公开，保留了与 Llama 3 相同的 architecture。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fixupx.com/vinceflibustier/status/1814233715641389456">来自 Vincent Flibustier 👽 (@vinceflibustier) 的推文</a>：在 Crowdstrike 的第一天，推送了一个小更新，下午放假 ✌️</li><li><a href="https://x.com/alpindale/status/1814814551449244058?s=12">来自 Alpin (@AlpinDale) 的推文</a>：已确认有 8B、70B 和 405B。前两个是从 405B 蒸馏出来的。128k（十进制为 131k）context。405b 画不出独角兽。Instruct tune 可能经过了 safety aligned。架构...</li><li><a href="https://x.com/alpindale/status/1814717595754377562?s=46">来自 Alpin (@AlpinDale) 的推文</a>：似乎 HF 的某人忘记及时将此仓库设为私有，Google 索引了它：sllhf/Meta-Llama-3.1-405B-Instruct-FP8。405B 是 llama 3.1？非常有趣。我想知道他们是否只会发布...</li><li><a href="https://realpython.com/python312-subinterpreters/?utm_source=perplexity">Python 3.12 预览：Subinterpreters – Real Python</a>：在本教程中，你将预览 Python 3.12 即将推出的功能之一以及 Python 3.13 的拟议变更，涉及 subinterpreters 在 CPython 程序中的工作方式。这些变更描述为...
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1264352779724656710)** (8 条消息🔥): 

> - `Deepseek chat v2 6.28`
> - `4o mini performance`
> - `Apple Watch support`
> - `Device shipping updates`
> - `Coqui model on MacOS` 


- **Deepseek chat v2 6.28 表现优于 Deepseek coder**：一位成员提到 *Deepseek chat v2 6.28 更新* 非常出色，甚至优于 *Deepseek coder*，且比 *4o mini* 更便宜。
- **4o mini 擅长逻辑，但在代码方面表现不佳**：虽然 **4o mini** 在逻辑和推理方面表现出色，但在 coding 任务中表现较差。
- **关于 iOS 应用支持 Apple Watch 的咨询**：一位成员询问 **iOS app** 是否支持 **Apple Watch**，并表示如果支持，**01** 在其研究用例中将大放异彩。
- **请求设备发货时间线的更新**：用户正在询问 **devices** 何时发货。
- **作为开发者为项目做贡献**：一位成员询问是否有机会让有能力的开发者为项目做贡献。回复指出欢迎提供帮助，[GitHub 上有许多 open issues](https://github.com)。


  

---

### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1264859221611118634)** (1 条消息): 

> - `GitHub 上的 Augmentoolkit`
> - `Pinokio 项目启动` 


- **Pinokio 的 Augmentoolkit 在 GitHub 上发布**: Pinokio 的新项目 [Augmentoolkit](https://github.com/pinokiofactory/augmentoolkit) 已在 GitHub 上公开发布，其特色是用于增强 AI 应用程序的工具。
   - 项目启动消息已在包括 [Discord](https://discord.gg/TQdNwadtE4)、[GitHub](https://github.com/pinokiocomputer/pinokio) 和 [Twitter](https://twitter.com/cocktailpeanut) 在内的多个平台宣布。
- **Pinokio 项目势头强劲**: **Pinokio** 项目正在社交媒体和开发者论坛上获得关注。
   - [点击此处查看 Twitter 上的更多详情](https://twitter.com/cocktailpeanut) 并加入 [Discord](https://discord.gg/TQdNwadtE4) 上的讨论。



**提到的链接**: <a href="https://pinokio.computer/item?uri=https://github.com/pinokiofactory/augmentoolkit">Pinokio</a>: AI 浏览器

  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1264159625255587840)** (4 条消息): 

> - `使用 GPT 模型进行微调 (Finetuning)`
> - `OpenAI 额度 (credits) 问题` 


- **为什么使用 GPT 模型进行微调并不常见**: **成本和供应商锁定 (vendor lock-in)** 是 GPT 模型很少被微调的主要原因。这涉及昂贵的 API 调用以及对特定公司基础设施的依赖。
- **接收 OpenAI 额度的问题**: 成员们报告了未收到承诺的 [OpenAI credits](https://link.to/openai-credits) 的问题。一位成员分享了组织 ID **org-EX3LDPMB5MSmidg3TrlPfirU**，并表示他们已经多次填写了表单。


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jarvis-labs](https://discord.com/channels/1238365980128706560/1241117895740625099/)** (1 条消息): 

vishnu9158: 不 (Nope)
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[east-coast-usa](https://discord.com/channels/1238365980128706560/1245411101718610001/)** (1 条消息): 

karmapa: 是的，8 月下旬在纽约举办见面会怎么样？
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openpipe](https://discord.com/channels/1238365980128706560/1245927847437008896/1264115136164270090)** (1 条消息): 

> - `Openpipe 与其他供应商`
> - `Replicate 模型的集成`
> - `Modal 模型兼容性` 


- **在 Replicate 或 Modal 模型上使用 Openpipe**: 一位成员询问是否可以将 Openpipe 与 **OpenAI** 或 **Anthropic** 以外的供应商一起使用，例如托管在 **Replicate** 或 **Modal** 上并具有 OpenAI 兼容 API 的模型。
   - *有人有见解吗？*
- **将 Replicate 模型集成到 Openpipe 中**: 讨论集中在将托管在 **Replicate** 上的模型集成到 **Openpipe** 中，并确保 API 兼容性。
   - 主要关注点是在保持与现有系统兼容性的同时，添加这些模型的便捷性。


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/1264575901673193625)** (1 条消息): 

> - `额度分配问题`
> - `课程表单` 


- **课程参与者的额度分配问题**: 一位成员报告说，尽管为他们的组织 **org-EX3LDPMB5MSmidg3TrlPfirU** 填写了必要的表单，但仍未收到额度。
   - 他们提到在课程开始时以及在报告日再次填写了表单。
- **重复提交课程额度表单**: 同一位成员重申多次填写表单以确保收到额度。
   - 尽管遵守了表单提交流程，但额度分配似乎仍存在问题。


  

---



### **LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1263995128284971128)** (3 条消息): 

> - `OpenAI Scale Tier`
> - `TPS 计算`
> - `GPT-4-o 吞吐量` 


- **对新的 OpenAI Scale Tier 感到困惑**: 一位用户对新的 [OpenAI Scale Tier](https://openai.com/api-scale-tier) 表示困惑，询问是否有人理解它。
   - 他们似乎对不同模型每秒吞吐量 (TPS) 涉及的具体计算感到特别困惑。
- **Pay-As-You-Go 层级的 TPS 计算不明确**: 一位用户质疑 OpenAI 在 pay-as-you-go 层级上 19 tokens 每秒 (TPS) 的计算方式，并将其与 GPT-4-o 约 80 tokens 每秒的吞吐量进行了比较。


  

---

### **LLM Perf Enthusiasts AI ▷ #[jobs](https://discord.com/channels/1168579740391710851/1169107992587812864/1264720227552329778)** (1 messages): 

> - `Websim platform`
> - `Founding AI Engineer role`
> - `AI-assisted software creation`
> - `Non-deterministic programs`
> - `Human-AI system` 


- **Websim 让软件创作具有可塑性**：Websim 旨在构建世界上最具可塑性的软件创作平台，让每个人都能解决自己的问题并实现梦想。
- **加入 Websim 担任创始 AI Engineer**：Websim 正在寻找一名创始 AI Engineer，为旨在实现自动化产品开发的非确定性程序（non-deterministic programs）建立快速迭代的基础。



**提及的链接**：<a href="https://websim.ai/">websim.ai</a>：未找到描述

  

---



### **Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1264662138966904944)** (2 messages): 

> - `Building with LLMs`
> - `BUD-E Voice Assistant` 


- **使用 LLM 构建一年的经验教训**：一位用户分享了[视频和博客文章](https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/)，总结了一个由三部分组成的系列，内容关于从业者使用 LLM 构建一年所获得的经验教训。
   - 该总结强调了战术、运营和战略层面的见解，并建议通过视频观看内容以便更好地理解。
- **BUD-E 语音助手演示及协作邀请**：一位用户分享了一个 [YouTube 视频](https://youtu.be/O4IXfa8CROs)，展示了开源 BUD-E 语音助手的演示，并邀请其他人加入他们新的 Discord 服务器进行协作。
   - 每日在线黑客松将于 **晚上 9 点 CEST** 开始，以引导新志愿者并协调项目工作。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/O4IXfa8CROs">BUD-E - Demo</a>：加入我们的 Discord 社区，亲自尝试 BUD-E，并帮助我们构建我和 BUD-E 在视频中讨论的语音助手：https://discord.gg/sTKSB2AwBvhttps...</li><li><a href="https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/">TLDR: 1 year of building with LLMs &#8211; D-Squared</a>：未找到描述
</li>
</ul>

</div>
  

---



### **AI Stack Devs (Yoko Li) ▷ #[team-up](https://discord.com/channels/1122748573000409160/1128471951963328512/)** (1 messages): 

ari991963: 大家好，我是 Aria，一名 2D/3D 艺术家，如果你有兴趣合作请私信（dm）。
  

---



### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1263992625602494536)** (1 messages): 

> - `Target Audience Clarification`
> - `Communication Strategy` 


- **明确目标受众**：一位成员询问了目标受众以及沟通策略背后的主要意图。
   - 讨论强调了在讨论产品时，针对工程师、准工程师、DevRel 和解决方案架构师的不同方法。
- **针对不同角色的沟通策略**：讨论了与工程师、DevRel、解决方案架构师和准工程师沟通的不同策略。
   - 每个角色可能需要量身定制的信息，以有效地传达产品功能和优势。


  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1264661996553502790)** (1 messages): 

> - `1 year of building with LLMs`
> - `TLDR series on LLMs`
> - `Lessons from LLM practitioners` 


- **关于使用 LLM 构建经验的 TLDR 系列**：[使用 LLM 构建 1 年的经验教训](https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/) 总结了一个由三部分组成的系列，详细介绍了所学到的战术、运营和战略教训。
   - *六位从业者*发布了该系列，推荐给那些认真对待 LLM 的人。
- **关于 LLM 学习心得的视觉化 TLDR 视频**：博客文章附带了一个视觉化的 [TLDR 视频](https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/)，使这些经验教训更易于理解。
   - *作者建议观看视频*，以便更好地理解所讨论的视觉内容。



**提及的链接**：<a href="https://www.dylandavis.net/2024/07/tldr-1-year-of-building-with-llms/">TLDR: 1 year of building with LLMs &#8211; D-Squared</a>：未找到描述

  

---



---



{% else %}


> 完整的频道细分内容已在邮件中截断。
> 
> 如果你想查看完整内容，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}