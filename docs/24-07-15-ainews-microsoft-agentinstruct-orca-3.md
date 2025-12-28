---
companies:
- microsoft-research
- apple
- tencent
- hugging-face
date: '2024-07-16T00:42:03.637767Z'
description: '**微软研究院（Microsoft Research）**发布了其 **Orca** 系列的第三篇论文 **AgentInstruct**，介绍了一种生成式教学管线。该管线产生了
  **2580万条** 合成指令用于微调 **mistral-7b** 模型，并取得了显著的性能提升：AGIEval 提升 40%、MMLU 提升 19%、GSM8K
  提升 54%、BBH 提升 38%、AlpacaEval 提升 45%，同时幻觉（hallucinations）减少了 31.34%。


  这种合成数据方法延续了 **FineWeb** 和 **苹果重写研究（Apple''s Rephrasing research）** 在提升数据集质量方面的成功。此外，**腾讯**声称已为合成数据生成了
  **10亿个** 多样化的角色（personas）。在 AI Twitter 上，热门讨论包括特朗普集会枪击事件，以及 **FlashAttention-3**、**RankRAG**
  和 **百万专家混合模型（Mixture of A Million Experts）** 等近期机器学习研究亮点。'
id: 2bb5808c-a173-4e92-8a75-1e22651b1692
models:
- mistral-7b
- orca-2.5
original_slug: ainews-microsoft-agentinstruct-orca-3
people:
- philschmid
- sama
- bindureddy
- rohanpaul_ai
- zachtratar
- dair_ai
title: 微软 AgentInstruct + Orca 3
topics:
- synthetic-data
- fine-tuning
- instruction-following
- transformers
- model-performance
- hallucination-detection
- dataset-quality
- flashattention
- mixture-of-experts
---

<!-- buttondown-editor-mode: plaintext -->**Generative Teaching is all you need.**

> 2024年7月12日至7月15日的 AI 新闻。
我们为您检查了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 服务器（**465** 个频道和 **4913** 条消息）。
预计节省阅读时间（以 200wpm 计算）：**505 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

今年 FineWeb 的巨大成功（[我们的报道在此](https://buttondown.email/ainews/archive/ainews-fineweb-15t-tokens-of-commoncrawl/)，[技术报告在此](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)）结合 [Apple 的 Rephrasing 研究](https://x.com/pratyushmaini/status/1752337225097076809)，基本上证明了在预训练和后训练阶段，数据集质量至少可以有一个数量级的提升。随着内容机构要么诉诸法律，要么寻求合作，研究重点已转向改进合成数据集生成，以延长我们已经压缩或抓取的 Token 的使用寿命。

Microsoft Research 凭借 [**AgentInstruct: Toward Generative Teaching with Agentic Flows**](https://x.com/_philschmid/status/1811308080166035549) 引起了最新轰动（不要与 [Crispino 等人 2023 年的 AgentInstruct](https://arxiv.org/abs/2310.03710) 混淆），这是其 Orca 系列论文中的第三篇：

- [Orca 1: Progressive Learning from Complex Explanation Traces of GPT-4](https://arxiv.org/abs/2306.02707)
- [Orca 2: Teaching Small Language Models How to Reason](https://arxiv.org/abs/2311.11045)
- [Orca Math: Unlocking the potential of SLMs in Grade School Math](https://arxiv.org/abs/2402.14830))

核心概念是由扮演不同角色的多个 Agent 对原始文档进行转换，以提供多样性（针对列出的 17 种能力），然后由更多 Agent 在“内容转换流”（Content Transformation Flow）中生成并完善指令。

 
![image.png](https://assets.buttondown.email/images/8ab271c3-ee32-45f9-9504-7ebc2b8a3e51.png?w=960&fit=max)
 

该流水线产出了 2200 万条旨在教授这 17 种技能的指令，结合之前 Orca 论文中的 380 万条指令，构成了 “Orca 2.5” —— 这是一个包含 2580 万条指令的合成数据集。作者使用该数据集对 Mistral 7b 进行微调，并报告了以下结果：

- AGIEval 提升 40%，MMLU 提升 19%；GSM8K 提升 54%；BBH 提升 38%；AlpacaEval 提升 45%，摘要任务的幻觉减少了 31.34%（感谢 [Philipp](https://x.com/_philschmid/status/1811308080166035549) 的总结）。

这只是合成数据研究领域的最新进展，最近 [腾讯在其相关工作中声称拥有 10 亿个多样化 Persona](https://x.com/arankomatsuzaki/status/1807593343007818065)。

 
![image.png](https://assets.buttondown.email/images/cbcd3d24-0df0-4a75-8b85-aee75cb17530.png?w=960&fit=max)
 

这看起来既显而易见会奏效，但与 FineWeb 相比又极其昂贵且低效，但管它呢，只要有用就行！


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

**特朗普集会枪击事件**

- **枪击细节**：[@sama](https://twitter.com/sama/status/1812566313577128153) 指出，在特朗普集会上，一名枪手在开火前不久，在屋顶上被一名警官发现，并用步枪指向该警官，随后子弹擦过特朗普头部，距离仅一英寸。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1812569682764833026) 分享了美联社的更新，证实枪手在开火前确实曾用步枪指向该警官。
- **反应与评论**：[@sama](https://twitter.com/sama/status/1812566313577128153) 希望这一时刻能促使**减少激进言论并寻求更多团结**，并称赞民主党人表现出了风度，抵制了将责任归咎于双方（"both-sides"）的冲动。[@zachtratar](https://twitter.com/zachtratar/status/1812585824837611689) 认为没有人会在那个距离导演一场子弹距离头部仅一英寸的枪击，因为如果是导演的，风险实在太大。[@bindureddy](https://twitter.com/bindureddy/status/1812514321924301146) 开玩笑说 AI 总统是无法被暗杀的。

**AI 与 ML 研究与进展**

- **新模型与技术**：[@dair_ai](https://twitter.com/dair_ai/status/1812504138510410131) 分享了本周的热门 ML 论文，涵盖了 **RankRAG、RouteLLM、FlashAttention-3、Internet of Agents、Learning at Test Time 以及 Mixture of A Million Experts** 等主题。[@_philschmid](https://twitter.com/_philschmid/status/1812516730234630563) 强调了最近的 AI 进展，包括 Hugging Face 上的 Google TPU、提升 Transformer 速度的 FlashAttention-3，以及支持在 16GB 显存上训练 7B 模型的 Q-GaLore。
- **实现与应用**：[@llama_index](https://twitter.com/llama_index/status/1812517033445396754) 在 Beta 版本中实现了 GraphRAG 概念，如图生成和基于社区的检索。[@LangChainAI](https://twitter.com/LangChainAI/status/1812513635509633294) 指出 OpenAI 的 Assistant API 是 Agentic 基础设施的一个例子，具有持久化和后台运行等特性。
- **讨论与见解**：[@sarahcat21](https://twitter.com/sarahcat21/status/1812519321676943491) 呼吁对可更新/协作式 AI/ML 以及模型合并（model merging）技术进行更多研究。[@jxnlco](https://twitter.com/jxnlco/status/1812572163917979803) 正在探索将 Prompting 技术融入 Instructor 文档中，以帮助理解可能性并识别抽象。

**编程、API 与开发者工具**

- **新 API 与服务**：[@virattt](https://twitter.com/virattt/status/1812549169447616953) 推出了一个公开测试版的股市 API，包含 S&P 500 股票 30 多年的数据（包括财务报表），且没有 API 限制。目前正在进行压力测试，随后将发布包含 15,000 多只股票的完整版本，供 AI 金融 Agent 使用。
- **编程经验与技巧**：[@giffmana](https://twitter.com/giffmana/status/1812505254052638858) 分享了在编写读取 multipart/form-data 的 Python 脚本时，对那些无用的在线资源感到沮丧，发现最有效的还是原始的 RFC2388 规范。[@jeremyphoward](https://twitter.com/jeremyphoward/status/1812424153141780546) 展示了 Python 中一种新的 function-cache 装饰器设计，用于组合缓存淘汰策略。
- **开发者讨论**：[@svpino](https://twitter.com/svpino/status/1812458195115549068) 预测，随着软件开发和 Machine Learning 的融合，AI 将与数据结构和算法一样，成为未来开发者的基础技能。

**幽默、迷因与离题讨论**

- **笑话与迷因**：[@cto_junior](https://twitter.com/cto_junior/status/1812498942401097962) 分享了一个结合了 Wagie News 和 4chan 梗的迷因。[@lumpenspace](https://twitter.com/lumpenspace/status/1812601881094729776) 调侃道，鉴于枪手政治倾向的信息相互矛盾，根本无法确定反特朗普情绪是否影响了枪手。
- **离题闲聊**：[@sarahookr](https://twitter.com/sarahookr/status/1812601109837480394) 推荐去里斯本旅游，并分享了一张该城市的照片。[@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1812533456234074151) 讨论了一个漫画分镜，它激发了一个名为 "Corgi Battle Pose" 的独立游戏标题灵感。

---

# AI Reddit 摘要

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。我们最近改进了[抗幻觉措施](https://buttondown.email/ainews/archive/ainews-we-solved-hallucinations/)，但仍在调整过滤、聚类和摘要质量。

**主题 1：快速发展中的 AI 研究论文发表滞后**

- [/r/singularity] **[由于 AI 发展的速度以及科学出版过程的漫长延迟，大量学术论文暗示 LLM 无法完成它们实际上能做得很好的事情。例如：这是一篇不错的论文，但它使用的是 GPT-3.5。](https://twitter.com/emollick/status/1808214380171219266)** ([Score: 237, Comments: 19](https://reddit.com//r/singularity/comments/1e35gg0/due_to_the_speed_of_ai_development_and_the_long/)): **关于 AI 能力的学术论文迅速过时**，这是由于 AI 发展的飞速步伐和漫长的科学出版流程。一个典型的例子是，尽管已经有了像 **GPT-4** 这样更先进的模型，某篇论文仍在使用 **GPT-3.5** 来评估 LLM 的能力。这种出版滞后导致已发表的研究与 AI 技术的现状之间存在显著差异。

- [/r/OpenAI] **[本周 AI 头条新闻](https://i.redd.it/ljn3rszgrlcd1.png)** ([Score: 361, Comments: 57](https://reddit.com//r/OpenAI/comments/1e3l0qj/ai_headlines_this_week/)): **AI 头条新闻主导科技新闻**：本周出现了一系列与 AI 相关的公告，包括 **Google** 的 **Gemini** 发布、**OpenAI** 的 **GPT Store** 推迟，以及 **Anthropic** 的 **Claude 2.1** 发布。AI 发展的飞速步伐引发了人们将其与互联网早期阶段的比较，一些专家认为 AI 的影响可能比网络革命更具变革性和深远意义。
   - **AI：不仅仅是又一次热潮**：评论者将**早期互联网怀疑论**与当前的 AI 质疑进行了类比。许多人回忆起最初对在线使用信用卡的抵触，强调了观念随时间推移会发生巨大变化。
  - **AI 彻底改变开发**：开发者称赞 AI 是编程的**“游戏规则改变者”**，一位用户在知识有限的情况下，使用 **Anthropic** 的 **console** 创建了一个 **native Swift app**。其他人指出 AI 能够比传统方法更快地缩小解决方案的范围。
  - **互联网泡沫的教训**：讨论涉及了 **2000 年互联网泡沫崩溃**，用户指出像 **Amazon** 这样的公司当时损失了 90% 的市值。一些人建议 AI 领域可能会出现类似的修正，但认为泡沫尚未达到顶峰。
  - **AI 的成长的烦恼**：批评者指出了当前 AI 应用的问题，例如 **Google** 的 **search highlights** 因幻觉（hallucinations）而受到批评。用户强调了负责任地部署 AI 的重要性，以维持该领域的公信力。


**主题 2. AI 对就业的影响：TurboTax 裁员**



- [/r/singularity] **[TurboTax 制造商解雇 1,800 名员工，称其正转向 AI](https://futurism.com/the-byte/intuit-turbotax-lay-offs-workers-ai)** ([Score: 303, Comments: 63](https://reddit.com//r/singularity/comments/1e3b4o4/maker_of_turbotax_fires_1800_workers_says_its/)): **TurboTax** 和 **QuickBooks** 背后的公司 **Intuit** 宣布裁员 **7%**，即解雇 **1,800 名员工**。该公司将重组原因归于向**人工智能**和**机器学习**的转型，旨在更好地服务客户并推动创新。尽管 Intuit 报告 2023 财年收入为 **144 亿美元**，比上一年增长了 **13%**，但仍做出了这一举动。


**主题 3. AI 在创意工作流中的集成：ComfyUI GLSL 节点**

- [/r/StableDiffusion] **[🖼 适用于 ComfyUI 的 OpenGL Shading Language (GLSL) 节点 🥳](https://v.redd.it/hew38iu92lcd1)** ([Score: 221, Comments: 21](https://reddit.com//r/StableDiffusion/comments/1e3ic2w/opengl_shading_language_glsl_node_for_comfyui/)): **适用于 ComfyUI 的 OpenGL Shading Language (GLSL) 节点** 已发布，允许用户创建自定义着色器并将其应用于 ComfyUI 工作流中的图像。这一新功能通过 **GPU 加速** 操作实现实时图像处理，有望增强 ComfyUI 中图像处理任务的效率和能力。GLSL 着色器的集成直接在 ComfyUI 环境中为高级视觉效果和自定义图像变换开辟了可能性。
   - **分享了 GitHub 仓库和 ShaderToy 链接**：原帖作者 **camenduru** 提供了 GLSL 节点的 [GitHub 仓库](https://github.com/patriciogonzalezvivo/comfyui_glslnodes) 链接，以及一个展示着色器效果潜力的 [ShaderToy 示例](https://shadertoy.com/view/3l23Rh)。
  - **兴奋情绪与潜在应用**：用户对这一新功能表现出极大的热情，**ArchiboldNemesis** 强调了其在 **遮罩输入 (masking inputs)** 方面的潜力，并推测可能实现 **“实时 SD 元球 (metaballs)”**。另一位用户则思考 ComfyUI 是否会演变成像 TouchDesigner 那样的 **可视化编程框架**。
  - **技术讨论与澄清**：一些用户寻求关于 **OpenGL** 及其与工作流关系的解释。一位评论者澄清说，OpenGL 着色用于 **视口渲染 (viewport rendering)**，不具备光线追踪能力；而另一位则提到 **three.js glsl shaders** 的知识可以应用于 ComfyUI。
  - **未来开发构想**：建议包括将 **VSCode 及其插件** 集成到 ComfyUI 中，或者将 ComfyUI 开发为 VSCode 插件。此外，还有关于当前实现中 **实时处理/渲染** 能力的疑问。


---

# AI Discord 回顾

> 摘要的摘要之摘要

**1. 推动 LLM 的边界**

- **突破性的 LLM 性能提升**：微软研究院推出了 [AgentInstruct](https://arxiv.org/html/2407.03502v1)，这是一个自动创建合成数据以对模型进行后期训练的框架，将 **Mistral-7b** 训练为 **Orca-3**，在 AGIEval 上实现了 **40% 的提升**，在 GSM8K 上提升了 **54%**，在 AlpacaEval 上提升了 **45%**。
   - **Ghost 8B Beta 模型**在 lc_winrate 和 AlpacaEval 2.0 胜率等指标上超越了 Llama 3 8B Instruct、GPT 3.5 Turbo 等模型，旨在提供卓越的知识能力、多语言支持和成本效率，详见其[文档页面](https://ghost-x.org/docs/models/ghost-8b-beta)。
- **新基准测试推动 LLM 进步**：引入了 [InFoBench](https://openreview.net/forum?id=qDXdmdBLhR)（Instruction Following Benchmark，指令遵循基准测试），引发了关于其与标准对齐数据集相关性的讨论，以及独特的基准测试是否能突出 LLM 在与 MMLU 高度相关之外的有价值特质。
   - **WizardArena/ArenaLearning 论文**详细介绍了在 [Kaggle 竞赛](https://www.kaggle.com/competitions/lmsys-chatbot-arena/overview)中通过人类偏好评分评估模型的方法，引起了人们对多轮合成交互生成和评估设置的兴趣。
  


**2. 驱动 AI 的硬件创新**

- **使用专用硬件加速 AI**：**MonoNN**，一种新型机器学习编译器，通过将整个神经网络容纳到单个 kernel 中来优化 GPU 利用率，解决了传统逐个 kernel 执行方案中的低效问题，详见 [论文演示](https://www.usenix.org/conference/osdi24/presentation/zhuang) 和 [源代码发布](https://github.com/AlibabaResearch/mononn)。
   - 围绕 **WebGPU** 开发的讨论强调了其快速的迭代周期，但也指出需要更好的工具和分析（profiling），成员们正在探索移植 **llm.c transformer kernels** 以获取性能洞察，并将更多 ML 工作负载转移到客户端计算。
- **通过量化优化 LLM**：关于[量化技术](https://arxiv.org/abs/2407.09141)的研究显示，压缩模型可能会出现“翻转（flips）”现象——尽管准确率指标相似，但输出会从正确变为错误，这突显了在定量评估之外进行定性评估的必要性。
   - 论文“[LoQT](https://arxiv.org/abs/2405.16528)”提出了一种方法，能够在消费级 24GB GPU 上高效训练高达 7B 参数的量化模型，以不同的方式处理梯度更新，并在预训练和微调中实现了相当的内存节省。
  


**3. 开源驱动 AI 创新**

- **协作努力推动进步**：[OpenArena 项目](https://github.com/syv-ai/OpenArena)推出了一个开放平台，让 LLM 相互竞争以提高数据集质量，主要使用 **Ollama** 模型，但也支持任何兼容 OpenAI 的端点。
   - [LLM-Finetuning-Toolkit](https://github.com/georgian-io/LLM-Finetuning-Toolkit) 发布，用于使用单一配置在开源 LLM 上运行实验，该工具构建在 HuggingFace 库之上，并支持评估指标和消融研究。
- **框架简化 LLM 开发**：LangChain 社区对流式输出处理进行了积极讨论，涉及 `invoke`、`stream` 和 `streamEvents` 在 langgraph 集成中的应用，以及管理 `ToolCall` 弃用和意外的默认工具调用。
   - LlamaIndex 获得了新功能，例如使用 Neo4j 进行实体去重、使用 LlamaCloud 集中管理数据管道、利用 GPT-4o 解析财务报告、通过 Redis 集成实现多 Agent 工作流，以及发布了高级 RAG 指南。
  

---

# 第一部分：高层级 Discord 摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **NPM 模块支持 Hugging Face Inference**：一个支持 [Hugging Face Inference 的新 NPM 模块](https://github.com/samestrin/llm-interface)已发布，并邀请社区提供反馈。
   - 开发者强调该模型覆盖了 36 个 Large Language Model 提供商，旨在培养协作开发的生态。
- **分布式计算汇聚 Llama3 力量**：Llama3 8B 在家庭集群上启动，设备涵盖从 iPhone 15 Pro Max 到 NVIDIA GPU，代码已在 [GitHub](https://github.com/evilsocket/llm3-cake) 开源。
   - 该项目旨在进行设备优化，号召社区共同对抗计划性报废。
- **LLM-Finetuning-Toolkit 发布**：[LLM-Finetuning-Toolkit](https://github.com/georgian-io/LLM-Finetuning-Toolkit) 的首次亮相为跨多种模型的 LLM 实验提供了一种使用单一配置的统一方法。
   - 它的独特之处在于集成了评估指标和消融研究（ablation studies），全部构建在 HuggingFace 库之上。
- **混合模型推动 EfficientNetB7 协作**：一项训练混合模型的计划将用于特征提取的 **EfficientNetB7** 与 Huggingface 上的 **Swin Transformer** 结合进行分类。
   - 参与者利用 Google Colab 的计算资源，寻求更简单的实现技术。
- **HF Inference API 归属错误引发热议**：**Copilot** 错误地将 **HF Inference API** 引用为 OpenAI 产品，导致用户在讨论中产生困惑。
   - 反应不一，既有像“奶酪冷却”服务器之类的幽默建议，也有对开源文档规范的务实要求。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Llama 3 备受期待的发布受阻**：传闻 [Meta Platforms](https://www.theinformation.com/briefings/meta-platforms-to-release-largest-llama-3-model-on-july-23) 原定于 7 月 23 日发布的 **Llama 3 (405b)** 将会推迟，Reddit 用户议论其可能推迟到今年晚些时候。
   - 社区交流中充满了对运营挑战的讨论，尽管有所延迟，大家仍期待微调（fine-tuning）的机会。
- **Gemini API 跃升至 2M Token**：Google 的 **Gemini API** 现在为 **Gemini 1.5 Pro** 提供了 **200 万 Token 的上下文窗口**，并发布了包括 [代码执行](https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio) 在内的多项功能。
   - AI 工程师们辩论了超长上下文的优劣，并推测其在日常场景中对性能的影响。
- **MovieChat GitHub 仓库引发数据集讨论**：[MovieChat](https://github.com/rese1f/MovieChat) 作为一个允许对 **10K 帧视频**进行对话的工具出现，引发了关于数据集创建的对话。
   - 考虑到构建开源数据集的复杂性，用户对其可行性存在争议。
- **Ghost 8B Beta 表现亮眼**：**Ghost 8B Beta 模型**因其性能受到赞誉，在 lc_winrate 和 AlpacaEval 2.0 胜率得分等指标上超过了 Llama 3 8B Instruct 和 GPT 3.5 Turbo 等对手。
   - [新文档](https://ghost-x.org/docs/models/ghost-8b-beta) 显示了该模型在多语言支持和成本效益等领域的实力，引发了关于战略贡献的讨论。
- **CURLoRA 应对灾难性遗忘**：作为微调方法的转变，**CURLoRA** 使用 CUR 矩阵分解来对抗灾难性遗忘（catastrophic forgetting）并最小化可训练参数。
   - AI 专家对这一消息表示赞赏，认为其在各种应用中具有潜力，详见 [论文](https://zenodo.org/records/12740116)。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **GPTs 停滞的启示**：人们对 **GPTs agents** 在训练后无法吸收新信息表示担忧，并澄清 [上传的文件仅作为参考“知识”文件](https://link.to/openai-docs)，不会改变底层模型。
   - 社区交流了 **GPTs agents** 如何与额外数据交互，确认新输入不会动态重塑基础知识。
- **OpenAI 的侧边栏风波**：用户注意到 platform.openai.com 侧边栏消失了 **两个图标**，引发了对界面更改的猜测和评估。
   - 侧边栏触发了关于可用性的讨论，提到与 threads 和 messages 相关的图标已经消失。
- **ComfyUI 胜过 A1111**：**ComfyUI** 相对于 **A1111** 的速度优势成为热门话题，社区测试表明 ComfyUI 有 15 倍的性能提升。
   - 尽管有速度优势，一些用户批评 ComfyUI 在控制精度上落后于 A1111，这表明在效率和功能之间存在权衡。
- **自定义遮罩组装的焦虑**：关于在 **ComfyUI** 中制作自定义遮罩的复杂过程引发了辩论，参与者指出 SAM inpainting 的性质更为繁琐。
   - 社区流传着简化遮罩创建过程的建议，提议集成 **Krita** 等工具以减轻 ComfyUI 中繁琐的程序。
- **艺术伦理辩论**：出现了关于 AI 生成个人肖像的伦理和法律讨论，成员们思考艺术创作中 **parody**（戏仿）的保护伞作用。
   - 社区就 AI 艺术的合法性展开了激烈辩论，涉及对公众人物代表性的担忧，以及在复杂情况下寻求 [专业法律咨询](#) 的必要性。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **CUDA 难题与 GPU 指南**：用户在解决“**No CUDA devices found**”错误，主张安装 NVIDIA 驱动程序和 “libcuda1” 软件包。
   - 在硬件对话中，**Intel Arc a750** 的表现不佳受到关注；对于 **LM Studio** 的精度，建议使用 NVIDIA 3070 或支持 AMD ROCm 的 GPU。
- **多语言编程偏好：Rust vs C++**：工程师们交换了对编程语言的看法，引用了 Rust 的内存安全性和 C++ 的历史包袱，并夹杂着一些 **Rust Evangelism**（Rust 布道）。
   - 尽管 Python 在神经网络开发中占据主导地位，但 Rust 和 C++ 社区强调了各自语言的优势以及 **llama.cpp** 等工具。
- **LM Studio：脚本限制与模型之谜**：关于 **lmstudio.js** 的辩论转向其使用 RPC 而非 REST，以及由于 RPC 的模糊性在集成 embedding 支持时面临的挑战。
   - AI 爱好者探讨了多 GPU 配置，指出了 PCIe 带宽的影响，并对即将推出的搭载 M4 芯片的 **Mac Studio** 在 LLM 任务中的表现充满期待。
- **Vulkan 和 ROCm：GPU 依赖与革命性运行时**：尽管担心其 4-bit 量化限制，但用户对 Vulkan 即将登陆 **LM Studio** 表示热切期待。
   - 同时，**ROCm** 成为 AMD GPU 用户的核心；对于 Llama 3 等模型至关重要，且其 Windows 支持也正受到关注。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT 替代方案辩论：追求学术卓越**：讨论集中在 **Copilot** 或 **Bing AI**（据称两者都运行在 **GPT-4** 上）在学术用途上是否更胜一筹。
   - 一位用户在感叹缺乏其他可行选择的同时，提到了 **Claude** 和 **GPT-4o** 等替代方案，但仍承认在 **ChatGPT** 上有支出。
- **微软的多重 CoPilot 难题**：成员们剖析了微软在 Word、PowerPoint 和 Outlook 等应用中部署的一系列 **CoPilot**，并指出 **Word CoPilot** 在深入研究主题方面的表现。
   - 相反，PowerPoint 的助手被贴上了“基础”的标签，主要辅助生成初级的幻灯片。
- **DALL-E 在 GPT 指导下的困境**：围绕 **DALL-E** 在遵循 **GPT** 指令渲染图像时的不可靠性展开了对话，结果要么是 Prompt 文本，要么是损坏的图像链接。
   - **DALL-E** 的这些“小故障”受到了批评，原因是该技术在初始命令下未能恰当地解读 **GPT** 的引导。
- **AI 多语言者：Prompt 语言差异**：询问围绕 Prompt 语言对响应质量的影响展开，特别是在 **ChatGPT** 交互中使用韩语与英语时。
   - 核心问题在于，直接使用目标语言的 Prompt 与需要翻译的 Prompt 相比，其有效性如何。
- **用魔法解锁 Android 的全部潜力**：分享的“Android 优化大师”指南承诺了通过电池优化、存储管理和高级设置来提升 Android 手机性能的秘诀。
   - 该指南通过俏皮的场景吸引了年轻的技术爱好者，使高级 Android 技巧变得易于理解且引人入胜。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **网站担忧与重定向**：当 [Mojo 网站](https://mojolang.org/) 宕机时引发了混乱，导致用户发现它并非官方网站。
   - 在纠正航向后，用户被引导至 [Modular 官方网站](https://www.modular.com/)，确保了正确的重定向。
- **机器人按部就班带来的困惑**：当成员标记多位贡献者时，Modular 的机器人会发出不必要的警告，误将该行为视为具有威胁性的机器人行为。
   - 随后展开了关于模式触发器的讨论，成员们要求审查机器人的解释逻辑。
- **推进模块可维护性的提案**：提出了一项创建 `stdlib-extensions` 的提案，旨在减轻 **stdlib** 维护者的工作量，并在 [GitHub](https://github.com/modularml/mojo/discussions/3233) 上引发了对话。
   - 社区请求勤奋的贡献者提供反馈，以确保这一改进有助于简化模块管理。
- **MAX 许可证文本截断**：[Max 许可证](https://www.modular.com/legal/max) 文本中的拼写错误引发了关于法律文档细节关注度的讨论。
   - 提到了诸如 **otherModular** 和 **theSDK** 之类的错误，促使官方迅速进行了修正。
- **加速集成的雄心**：成员们询问 **Max** 如何衔接 AMD 发布的 **Unified AI software stack**（统一 AI 软件栈），突显了 **Modular** 日益增长的影响力。
   - 引用利益交汇点，用户对 **MAX** 平台潜在的独家合作伙伴关系表现出极大的热情。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Cloudflare 争议与 API 额度探索**：由于 **API** 位于 **Cloudflare** 防护之后，成员们正面临访问挑战；同时，另一些人对比 Pro 计划升级中宣传的 **5 美元免费额度** 的可用性提出质疑。
   - 讨论还涉及使用 5 美元额度时的挫败感，用户正通过 [社区频道](https://discord.com/channels/1047197230748151888/1161802929053909012/1207351871186931783) 寻求帮助。
- **Pro 每日限额下调**：Pro 用户注意到其每日搜索限制悄然从 600 次降至 540 次，引发了关于未来变化以及对更高透明度需求的讨论。
   - 社区对这一意外变化做出了反应，并讨论了其对日常操作的潜在影响。
- **图像问题与能力对比**：用户分享了 Perplexity 的回复错误引用过去图像的问题，这阻碍了对话的连贯性。
   - 技术专家们辩论了 Perplexity 相对于 **ChatGPT** 的优势，特别是在文件处理、图像生成和精准后续跟进等专业领域。
- **令人困扰的 API 模型之谜**：一位用户试图通过 API 模拟 **Perplexity AI 免费版的结果**，但在获取 URL 来源方面遇到困难，从而引发了关于正在使用哪些模型的查询。
   - 目标是匹配免费版的能力，这表明需要明确 API 服务中的模型利用和输出情况。
- **分享光谱：从健康到争议**：讨论范围广泛，从[健康与力量](https://www.perplexity.ai/search/how-to-achieve-health-strength-094kl4NzQea2mENjIOdG8Q)的途径，到理解动态市场力量如[坎蒂隆效应 (Cantillon Effect)](https://www.perplexity.ai/search/the-cantillon-effect-KnCFxYCeQuG51gUkuJdtkA)。
   - 对话还包括我们[牙齿](https://www.perplexity.ai/search/are-our-teeth-unique-IvBjExR8TL64cO9QknSbsw#0)中的独特标识符，以及对某[政治人物安全事件](https://www.perplexity.ai/page/trump-assassination-attempt-Yc6pNnfDQ6WUP6qD44AZIg)的分析。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **AgentInstruct 的飞跃**：[AgentInstruct](https://arxiv.org/html/2407.03502v1) 为将 **Mistral-7b** 等模型增强为更复杂的版本（如 **Orca-3**）提供了蓝图，展示了在基准测试上的显著提升。
   - 该应用在 AGIEval 和 GSM8K 上分别实现了 **40%** 和 **54%** 的提升，在 AlpacaEval 上提升了 **45%**，为竞争对手树立了新标杆。
- **剥蛋专家的轻松建议**：剥蛋技巧意外走红，建议倾向于使用 **10 分钟热水浴** 来实现完美的剥壳效果。
   - 还有人分享了[醋溶液魔法](https://www.scienceworld.ca/resource/naked-eggs-acid-base-reaction)，通过酸碱反应展示了无壳蛋的奥秘。
- **AI 的 YouTube 戏码：Q-star 泄露**：**Q-star 的机密细节**通过 [YouTube 爆料](https://youtu.be/T9gAg_IXB5w)公开，展示了 **AGI** 发展的希望与风险。
   - 来自 OpenAI 代号为 **STRAWBERRY** 的隐藏宝库的见解揭示了即将推出的 **LLM** 策略。
- **告别 PDF，拥抱 Markdown**：新版本的 [Marker](https://x.com/VikParuchuri/status/1811851126125527096) 利用高效的模型架构缩短了 PDF 到 Markdown 的转换时间，有助于提升数据集质量。
   - 性能提升包括 **MPS 上 7 倍的加速** 和 **10% 的 GPU 性能增长**，为快速创建数据集指明了方向。
- **在应用中扩展 LLM 的视野**：关于应用集成的讨论显示，**检索增强生成 (RAG)** 是嵌入教程智能的首选方案。
   - 讨论中出现了将 **Mixtral** 和 **Llama** 等模型扩展到 **1M tokens** 的建议，尽管实际使用仍面临挑战。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Warp Speed WebGPU 工作流**：探索 **WebGPU** 开发的用户讨论了其快速迭代周期，但指出工具链和性能分析（profiling）是需要改进的领域。
   - 推荐使用类似 **dawn** 的共享库方法，并提供了一个 [livecoding demo](https://drive.google.com/file/d/15oXwYqVeoOMNYDEjG3xJ2PEeNYFbbGjz/view?usp=drive_link) 展示了更快的 shader 开发。
- **探究 CUDA Core 的并发性**：对 **CUDA core** 处理过程的深入研究表明，每个 CUDA core 一次可以处理一个线程，一个 A100 SM 可以从 2048 个线程池中同时管理 **64 个线程**。
   - 讨论还集中在 **register limitations**（寄存器限制）如何影响线程并发，进而影响整体计算效率。
- **使用 cudaMallocManaged 实现高效内存管理**：提议使用 **cudaMallocManaged** 代替 cudaFix，以支持内存有限的设备，特别是为了增强小型 GPU 的集成工作。
   - 切换到 cudaMallocManaged 被视为确保性能不受阻碍，同时适配更广泛 GPU 架构的关键。
- **低比特操作的 FSDP 技巧**：关于为低比特优化实现 **FSDP** 支持的讨论集中在优化状态子类（optimization state subclass）中未解决的集合通信操作（collective ops）。
   - 讨论了旨在辅助 FSDP 兼容性的开发者指南，以提高开发者参与度并防止潜在的项目流失。
- **基于 WebGPU 的浏览器端 Transformers**：成员们讨论了利用 **Transformers.js** 在浏览器中运行最先进的机器学习任务，发挥 WebGPU 在 ONNX runtime 中的潜力。
   - 还强调了在 **Windows 上构建 Dawn** 相关的挑战，指出了故障排除经验以及 buffer 限制对性能的影响。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **OpenArena 雄心勃勃的 AI 对决**：一个新的 [OpenArena 项目](https://github.com/syv-ai/OpenArena) 已启动，挑战 LLM 进行竞争并确保稳健的数据集质量。
   - **Syv-ai 的仓库** 详细说明了申请流程，旨在与各种 LLM 供应商直接接触。
- **Cohere 难题：活动访问困局**：成员们抱怨 **Cohere 活动** 链接混淆导致访问问题，通过分享扩散模型（diffusion model）演讲的正确 [Zoom 链接](https://zoom.us/j/8022650618?pwd=V0VvYnAyQVBlNnIrUktGNyt6WFE1dz09) 解决了该问题。
   - **客座演讲环节** 恢复了清晰度，并提供了使用扩散模型创建语谱图（spectrograms）的指导。
- **AI 能力训练成本骤降**：[Andrej Karpathy 对 AI 训练成本的看法](https://x.com/karpathy/status/1811467135279104217) 显示成本大幅下降，训练 GPT-2 等模型的门槛显著降低。
   - 他阐明了从 2019 年的高成本环境到现在的转变，如今爱好者只需花费极少的费用即可训练类 GPT 模型。
- **使用 NPM 模块实现无缝 LLM 切换**：通过 [更新的 NPM 模块](https://github.com/samestrin/llm-interface)，开发者可以轻松集成 **Cohere**，非常适合跨平台 LLM 交互。
   - 这种模块化方法为统一使用各种 AI 平台打开了大门，丰富了开发者的工具箱。
- **r/localllama 新闻机器人纪事**：**r/localllama** 社区通过一个由 [Langchain 和 Cohere](https://docs.cohere.com/docs/tool-use) 驱动的机器人为 Discord 注入了活力，该机器人可以聚合 Reddit 的热门帖子。
   - 这个创新的引擎不仅能进行总结，还能将新闻整理成引人入胜的叙述，专为特定频道定制。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **伦敦 AI 聚会缺乏技术深度**：讨论显示出对**伦敦 AI 聚会 (AI meetups in London)** 技术深度的不满，建议有兴趣的人去参加 UCL 和 Imperial 的研讨会。
   - **ICML** 和 **ICLR** 会议被推荐用于有意义的深度交流，尤其是在研究人员的小众聚会中。
- **Arrakis：加速机械可解释性 (Mechanistic Interpretability)**：[Arrakis](https://github.com/yash-srivastava19/arrakis) 是一个用于可解释性实验的工具包，旨在增强实验追踪和可视化。
   - 该库与 tuned-lens 等工具集成，以提高 **mechinterp** 研究效率。
- **探索模型的时间相关性**：将时间相关性引入 LLM 的兴趣日益增长，因为传统的 timestamp 方法缺乏有效性。
   - 目前的讨论集中在时间敏感型数据集的文献和用于提升训练的 Benchmark 等途径。
- **量化怪癖：表象之下另有乾坤**：针对一篇关于[量化翻转 (quantization flips)](https://arxiv.org/abs/2407.09141)的论文提出了担忧，该论文解释了压缩模型尽管准确率指标相同，但行为可能不同。
   - 这引发了关于在定量评估之外进行严格定性评估必要性的对话。
- **挖掘 lm-eval 的潜力**：一项技术咨询引导出了将自定义 Transformer-lens 模型与 **lm-eval 的 Python API** 集成的指南，详见[此文档](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage)。
   - 然而，一些成员仍在摸索 **lm-evaluation-harness** 中自定义函数和指标的复杂性。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **MonoNN 优化 GPU 工作负载**：新型机器学习编译器 **MonoNN** 的引入引起了关注，它采用针对整个神经网络的单算子 (single kernel) 方法，可能提高 GPU 效率。相关[论文](https://www.usenix.org/conference/osdi24/presentation/zhuang)和[源代码](https://github.com/AlibabaResearch/mononn)已发布。
   - 社区考虑了 MonoNN 方法在减少逐算子 (kernel-by-kernel) 执行开销方面的潜在影响，这与正在进行的关于 **tinygrad kernel overhead** 问题的讨论相呼应。
- **MLX 胜过 tinygrad**：在 beautiful_MNIST 基准测试中，**MLX** 以更好的速度和准确性占据了上风，引起了社区对 [mlx 的 tinygrad commit](https://github.com/tinygrad/tinygrad/commit/8940530290b04048074be1deadd24e5d91d67d28) 的关注。
   - 这一发现引发了关于改进 tinygrad 性能的进一步讨论，目标是解决开销和效率低下的问题。
- **针对 tinygrad 的 avg_pool2d 的改进建议**：社区请求增强 `avg_pool2d` 以支持 `count_include_pad=False`，这是 Stable Diffusion 训练评估中的一个功能，并提出了模仿 [PyTorch 实现](https://pytorch.org/docs/stable/generated/torch.nn.functional.avg_pool2d.html)的潜在解决方案。
   - 讨论围绕 **MLPerf** 等基准测试对该功能的需求展开，并出现了使用现有池化操作的变通方案建议。
- **关于 Tinygrad 张量索引 (Tensor Indexing) 的讨论**：成员们交流了 tinygrad 中张量索引的细微差别，将其与其他框架进行比较，并演示了掩码 (masking) 等操作如何提高性能。
   - 一位成员引用了 [tinygrad 文档](https://docs.tinygrad.org/quickstart/#training)来阐明该工具包中这种特定张量操作的执行和效率优势。
- **PR 策略与文档动态**：成员们达成共识，建议将增强功能、Bug 修复和功能实现分成独立的 Pull Request (PR)，以简化审核流程，这在处理 FID 的 `interpolate` 函数时表现得很明显。
   - 成员们强调了保持示例最新且可运行的重要性，讨论了在 [tinygrad 文档](https://github.com/tinygrad/tinygrad/blob/master/serve_docs.sh)中测试和验证代码块的策略。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Leaderboard Levels Up: Open LLM Leaderboard V2 Excitement**: 排行榜升级：Open LLM Leaderboard V2 引发热议。Latent Space 关于 **Open LLM Leaderboard V2** 的新一期节目引发了讨论，社区成员分享了他们的热情。
   - 该播客链接到了一个[新发布版本](https://x.com/swyx/status/1811898574416019562)，为听众提供了关于最新 LLM 排名的见解。
- **Linking Without Hallucinating: Strategies to Combat Misinformation**: 链接不幻觉：打击误导信息的策略。讨论围绕 **SmolAI** 消除 Reddit 链接幻觉的创新方法展开，重点关注 **pre-check and post-proc**（预检查和后处理）方法。
   - 讨论了[技术和结果](https://x.com/Smol_AI/status/1811957074840158255)，强调了可靠链接在增强 LLM 使用方面的重要性。
- **Unknown Entrants Stir LMSys: New Models Spark Curiosity**: 未知参赛者搅动 LMSys：新模型引发好奇。关于 **LMSys arena** 中新模型背后实体的猜测四起，伴随着各种不同的观点。
   - 关于 **Command R+ jailbreaks**（越狱）及其影响的传闻也是热议的一部分，反映在[社区对话](https://x.com/apples_jimmy/status/1812029979888439525?s=61)中。
- **Composing with Cursor: The Beta Buzz**: 使用 Cursor 创作：Beta 版热潮。**Cursor** 的新 Composer 功能在社区中引起了轰动，用户们急于讨论其 UX 对比和 Beta 版发布。
   - 价格和实用性成为关注的话题，旁观者分享了[积极反应](https://x.com/shaoruu/status/1812412514350858634)并思考了订阅模式。
- **Microsoft's Spreadsheet Savvy: Introducing SpreadsheetLLM**: 微软的电子表格智慧：推出 SpreadsheetLLM。微软凭借 **SpreadsheetLLM** 引起了轰动，这项创新旨在利用 **SheetCompressor** 编码框架改进 LLM 对电子表格的处理。
   - 对话转向了其将 LLM 适配到电子表格数据的潜力，对其[论文](https://arxiv.org/html/2407.09025v1)中详述的细致方法感到兴奋。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Open Source Tools Open Doors**: 开源工具开启新大门。用户 le_mess 创建了一个名为 OpenArena 的数据集创建工具的 [100% 开源版本](https://github.com/syv-ai/OpenArena)，扩展了模型训练灵活性的视野。
   - OpenArena 最初是为 OpenRouter 设计的，现在正利用 **Ollama** 来提升其能力。
- **Memory Usage Woes in ORPO Training**: ORPO 训练中的内存使用困扰。xzuyn 注意到 ORPO 训练期间内存使用量激增，尽管最大序列限制为 2k，但仍导致了 out-of-memory 错误。
   - 对话指出，Tokenization（分词）后长序列**截断时丢失消息**可能是罪魁祸首。
- **Integrating Anthropic Prompt Know-How**: 集成 Anthropic 的 Prompt 诀窍。Axolotl 改进的 Prompt 格式从 Anthropic 的官方 Claude 中汲取了灵感（由 Kalomaze 讨论），其特点是使用特殊 Token 来清晰划分对话轮次。
   - 该模板适用于 Claude/Anthropic 格式，可以在[这里](https://github.com/axolotl-ai-cloud/axolotl/blob/main/src/axolotl/utils/chat_templates.py)找到，引发了关于其**可读性和灵活性**的分歧。
- **RAG Dataset Creation Faces Scrutiny**: RAG 数据集创建面临审查。nafnlaus00 对 Chromium 在渲染 RAG 模型数据集抓取所需的 JavaScript 站点时的安全性表示担忧。
   - 建议包括探索替代的抓取解决方案，如 **firecrawl 或 Jina API**，以应对这些潜在的漏洞。
- **Weighted Conversations Lead Learning**: 加权对话引导学习。Tostino 提出了一种利用训练数据的创新方法，涉及**权重调整**，以引导模型学习远离不理想的输出。
   - 这种**高级微调**可以通过专注于问题区域来优化模型，从而提升学习曲线。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **AI 推理的草莓地 (Strawberry Fields of AI Reasoning)**：OpenAI 正在开发一种名为 **Strawberry** 的新推理技术，人们将其与斯坦福大学的 **STaR** (Self-Taught Reasoner) 进行比较。社区内部人士认为，其能力与 [Reuters](https://www.reuters.com/technology/artificial-intelligence/openai-working-new-reasoning-technology-under-code-name-strawberry-2024-07-12/) 详细报道的一篇 2022 年论文中所概述的内容相吻合。
   - 该技术对推理基准测试的预期影响，引发了对其相较于现有系统可能具备的优势的审视，重点关注产品名称、核心功能和发布日期。
- **LMSYS Arena 的神秘模型参赛者**：**LMSYS chatbot arena** 因 **column-r** 和 **column-u** 等新选手的加入而热闹非凡。根据 [Jimmy Apples](https://x.com/apples_jimmy/status/1812029979888439525?s=46) 的消息，推测这些模型是 **Cohere** 的杰作。
   - Twitter 用户 [@btibor91](https://x.com/btibor91/status/1812491983220343239?s=46) 进一步引发了关注，他指出有四个新模型正准备发布，包括 **eureka-chatbot** 和 **upcoming-gpt-mini**，据称其中一些模型的训练方是 Google。
- **评估 Mistral-7B 的指令强度**：鉴于 **Orca3/AgentInstruct 论文** 的发现，AI 社区正在讨论 **Mistral-7B 的指令微调 (instruct-tuning)** 效果，并试图确定底层指令微调数据集的强度。
   - 讨论评估了当前数据集是否符合鲁棒性标准，并将 **Mistral-7B** 的基准测试结果与其他模型的表现进行了对比。
- **InFoBench 引发基准测试辩论**：最近公开的 **InFoBench** (Instruction Following Benchmark) 引发了关于其价值与既有对齐数据集对比的讨论，对其在现实世界中的相关性评价褒贬不一。
   - 怀疑者和支持者就 **InFoBench** 以及 **EQ Bench** 等独特基准测试是否真正突出了语言模型的重要特质（考虑到它们与 **MMLU** 等成熟基准测试的相关性）展开了激烈交锋。
- **加州 AI 立法迷宫**：**加州 AI 法案 SB 1047** 的通过引发了一场立法冲突，**AI 安全专家**和**风险投资家**在关键投票前就该法案的影响展开辩论。
   - 参议员 **Scott Wiener** 将这场冲突描述为“喷气机帮对阵鲨鱼帮 (Jets vs Sharks)”，揭示了 [Fortune 文章](https://fortune.com/2024/07/15/california-ai-bill-sb-1047-fierce-debate-regulation-safety/) 中记录的极化观点，该文章可通过 [Archive.is](https://archive.is/e5n9A) 获取以供更广泛的审阅。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **JavaScript 的权衡：LangChain 的三大函数**：用户剖析了 LangChain JS 中 `invoke`、`stream` 和 `streamEvents` 的复杂性，讨论了它们在 **langgraph** 中流式输出的效果。
   - 出现了一项提议，建议使用 **Agent** 来处理数据收集和 API 交互等各种任务。
- **Gemini API 的 Base64 困扰：寻觅、解码、失败**：尽管 File API 上传是唯一有文档记录的方法，但用户在 **Gemini Pro API** 中使用 Base64 时遇到了令人困惑的“无效输入”障碍。
   - 社区的指导指出，文档需要进一步明确，并对在 API 中使用 Base64 进行更详细的阐述。
- **ToolCall 的更迭：从 LangChain Legacy 到 OpenAIToolCall**：**`ToolCall`** 现已废弃，引导用户使用其继任者 `OpenAIToolCall`，后者引入了用于排序的 `index` 功能。
   - 社区思考了包更新以及如何处理自动模式下无意触发的默认工具调用。
- **幻觉风险：聊天机器人凭空生成查询**：有报告称 HuggingFace 模型出现幻觉，引发了围绕 **LLM 生成**的聊天机器人随机问答对的讨论。
   - 社区提供了替代方案，包括转向 OpenAI 模型或 FireworksAI 模型，尽管经过微调的 Llama 模型似乎对典型的重复惩罚具有抵抗力。
- **嵌入模型的卓越表现：OpenAI 模型备受关注**：关于最佳 OpenAI 嵌入模型的讨论达到高潮，引发了关于理解和利用**嵌入向量 (embedding vectors)** 的最佳模型的探讨。
   - 普遍共识倾向于将 **`text-embedding-ada-002`** 作为 LangChain 中向量嵌入的首选模型。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 的去重之舞**：LlamaIndex 知识图谱正在进行**节点去重（node deduplication）**，在[相关文章](https://medium.com/@rajib76.gcp/entity-de-duplication-llamaindex-approach-0b97d2950a9f)中提供了[新见解](https://youtu.be/vMz0icWZd5A)和解释，强调了知识建模的重要性。
   - 在执行 **NebulaGraphStore** 集成时出现了技术困难，详见 [GitHub Issue #14748](https://github.com/run-llama/llama_index/issues/14748)，这指向了方法预期上的潜在不匹配。
- **公式与财务的融合**：**结合 SQL 和 PDF Embedding** 引发了关于集成数据库和文档的讨论，讨论由 [LlamaIndex 的 SQL 集成指南](https://docs.llamaindex.ai/en/stable/examples/query_engine/SQLJoinQueryEngine/)中的示例引导。
   - 提到的 `NLSQLTableQueryEngine` 问题引发了关于正确方法的辩论，因为 Manticore 的查询语言与 MySQL 的经典语法有所不同。
- **Redis 重新思考多 Agent 工作流**：@0xthierry 的 **Redis 集成** 促进了生产级工作流的构建，为 Agent 服务通信创建了一个网络，详见[热门讨论帖](https://discord.com/channels/1059199217496772688/1187460979064324127/1261428463169179748)。
   - **多 Agent 系统** 的效率是一个核心主题，Redis Queue 作为 Broker（代理），反映了架构流线化的趋势。
- **更细的数据分块，更精准的 Embedding**：根据[基础策略文档](https://docs.llamaindex.ai/en/latest/optimizing/basic_strategies/basic_strategies/#chunk-sizes)中关于最佳分块和重叠设置的建议，将**数据切分为更小的尺寸**提升了 LlamaIndex Embedding 的精度。
   - LlamaIndex AI 社区一致认为，`chunk_size` 为 512 且重叠（overlap）为 50 时，能优化细节捕捉和检索准确率。
- **带有 LlamaIndex 特色的高级 RAG**：想要深入了解 Agent 模块，**LlamaIndex 指南**提供了全面的教程，如 @kingzzm 关于利用 **LlamaIndex 查询流水线（query pipelines）** 的教程所示。
   - **RAG 工作流** 的复杂性被分步拆解，从发起查询到面向 AI 工程师的查询引擎微调。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **GUI 之光：OpenInterpreter 升级**：在 [OpenInterpreter](https://github.com/jbexta/AgentPilot) 中集成完整的 GUI 增加了**可编辑消息**、分支、自动运行代码和保存功能。
   - 对探索这些功能的**视频教程**的需求表明了社区的高度兴趣。
- **OS 探索：OpenAI 的潜在尝试**：在 Sam Altman 领导下的 OpenAI 可能正在酝酿自己的 OS，这一[推文暗示](https://x.com/apples_jimmy/status/1805373587127402883)引发了广泛猜测。
   - 随着社区成员拼凑近期**招聘职位**中的线索，悬念不断增加。
- **Phi-3.1：承诺与精度**：Techfren 对 **Phi-3.1** 模型潜力的分析揭示了其令人印象深刻的尺寸能力比。
   - 然而，讨论显示它偶尔在精确执行 `<INST>` 时遇到困难，引发了关于增强功能的讨论。
- **从 Internlm2 到 Raspi5：紧凑型突破**：“Internlm2 smashed”因其在 **Raspi5** 系统上的表现而受到关注，这对于紧凑型计算需求来说前景广阔。
   - 重点在于探索适用于新型 IoT 应用的 **multi-shot** 和 **smash 模式**。
- **Ray-Ban 的数字越狱：社区的狂欢**：**越狱 Meta Ray-Ban** 的可能性让社区兴奋不已并充满期待。
   - 破解该硬件的愿景引发了对新功能机会的关注热潮。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Agents Assemble in LLM**：一位用户解释了在 **LLMs** 中添加 Agent 以增强 **chat pipelines** 模块化的方法，利用 JSON 输出执行诸如 **fetching data** 和 **API interaction** 等任务。
   - 分享的 [指南](https://nik-hil.hashnode.dev/how-to-add-agents-in-large-language-models-a-detailed-guide) 展示了包含 **Input Processing** 和 **LLM Interpretation** 的步骤，强调了模块化组件的优势。
- **OpenAI API Keys: The Gateway for Tutorials**：聊天机器人项目教程需要 **API keys**，社区中有人请求分享 key 以协助教程的创作。
   - 该成员未提供更多背景，但强调了完成并发布指南对 key 的临时需求。
- **Error Quest in LLM Land**：成员们表达了在处理来自 **modal** 和 **axolotl** 的陌生错误时的困扰，表示需要在 Slack 等平台上寻求社区帮助。
   - 虽然未详细说明错误的具体性质，但对话暗示需要更好的渠道来解决这些技术问题。
- **Navigating Through Rate Limit Labyrinths**：一位在 **Langsmith evaluation** 期间面临 token 速率限制的用户通过调整 **max_concurrency** 设置找到了解决方法。
   - 讨论还涉及了在脚本运行中引入延迟的策略，旨在避开服务提供商施加的速率限制。
- **Tick Tock Goes the OpenAI Clock**：讨论透露 **OpenAI credits** 将于 **9 月 1 日**到期，用户在询问后澄清了截止日期。
   - 对话幽默地暗示要发起一份 *请愿书* 来延长额度有效期，表明用户在既定到期日之后仍依赖这些资源。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Hugging Face Hits the Green Zone**：[Hugging Face](https://analyticsindiamag.com/hugging-face-announces-profitability-with-free-and-open-source-models) 宣布在拥有 220 人团队的情况下实现盈利，同时保持其大部分平台免费且开源。
   - CEO Clement Delangue 兴奋地指出：*'这并不是我们的目标，因为我们银行里有很多钱，但看到 @huggingface 最近盈利了还是挺兴奋的，我们有 220 名团队成员，而且我们的大部分平台对社区都是免费和开源的！'*
- **Cambrian-1's Multimodal Vision**：介绍了 **Cambrian-1** 系列，这是一系列专注于视觉的新型多模态 LLM，可在 [GitHub](https://github.com/cambrian-mllm/cambrian) 上获取。
   - 这一扩展有望拓宽 AI 模型在学习语境中整合图像的视野。
- **MagViT2 Dances with Non-RGB Data**：围绕 **MagViT2** 与非 RGB 运动数据（特别是 24x3 数据集）的潜在兼容性展开了讨论。
   - 虽然对话简短，但提出了关于 AI 模型中非标准数据格式预处理需求的问题。
- **Choreographing Data for AI Steps**：非 RGB 运动数据的预处理技术引起了关注，以确保它们能与现有的 AI 模型和谐协作。
   - 这些技术的细节仍有待在进一步讨论中明确。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **OpenArena 开启 LLM 竞赛**：[OpenArena](https://github.com/syv-ai/OpenArena) 的发布启动了一个新的 **LLM 对决**平台，通过第三方模型评审来增强数据集的完整性。
   - OpenArena 主要整合了 **Ollama 模型**，并与任何基于 OpenAI 的端点兼容，扩展了其在 AI 领域的潜在应用。
- **WizardLM 论文聚焦 Arena Learning**：[WizardLM 论文](https://www.microsoft.com/en-us/research/publication/arena-learning-build-data-flywheel-for-llms-post-training-via-simulated-chatbot-arena/) 详细介绍了“**Arena Learning**”的概念，为 LLM 评估建立了一种新方法。
   - 这种基于模拟的方法侧重于细致的评估和持续的离线模拟，通过监督微调和强化学习技术来增强 LLM。

---

**Alignment Lab AI Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。

---

**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。

---

**AI Stack Devs (Yoko Li) Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。

---

**Torchtune Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。

---

# 第 2 部分：频道详细摘要与链接

{% if medium == 'web' %}

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1261400329254080553)** (989 messages🔥🔥🔥): 

> - `HF Inference API`
> - `GPT integration`
> - `Model performance issues`
> - `Leaderboard Upvotes`
> - `Llama2 Chat model setup` 

- **HF Inference API 被错误地归因于 OpenAI**：**Copilot** 错误地将 **HF Inference API** 引用为 **OpenAI** 的一部分，引起了用户的困惑。
   - 一名用户幽默地建议使用“奶酪冷却”来管理过热的服务器，而另一名用户询问了关于开源文档风格的问题。
- **CUDA 和模型设置问题**：一名用户在设置 **Llama2 Chat 模型**时遇到了 **CUDA** 问题，报告称文本生成速度极慢。
   - 尽管解决了一些 CUDA 问题，该用户指出生成延迟仍然存在，并收到了测试更小 token 批次的建议。
- **模型排行榜中的队列优先级**：排行榜队列主要受点赞数影响，引发了关于公平性和类似模型刷屏的讨论。
   - 一名用户担心新用户在社交方面面临的困难会影响可见性和模型性能评估。
- **错误处理和 RL 训练问题**：频繁讨论了与 **ArrowInvalid** 和 CUDA 中的**非法内存访问**相关的错误，用户提供了排错建议。
   - 一名用户在 **Unity** 环境中设置 RL 训练时遇到困难，尽管收到了配置建议，但仍因缺少可执行文件而面临问题。
- **对 Python 项目设置的担忧**：一名用户对设置 Python 项目表示沮丧，提到了 Python 版本和依赖项的多个问题。
   - 其他人建议使用 Linux 环境和特定的 Python 版本，呼应了开源项目配置中的常见困难。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>

<li><a href="https://sites.research.google/trc/about/">TPU Research Cloud - 关于</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/google/sdxl">TPUv5e 上的 Stable Diffusion XL - google 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/learn/cookbook/en/llm_judge">使用 LLM-as-a-judge 🧑‍⚖️ 进行自动化和通用的评估 - Hugging Face 开源 AI Cookbook</a>：未找到描述</li><li><a href="https://youtu.be/KyOlpzA5jKM">[高清重制] 等等，那是 Gabe 吗？</a>：因为我在 YouTube 其他地方没看到，所以进行了重制 https://www.youtube.com/watch?v=ELtzcpb_j38 这是该迷因的高质量原始版本。S...</li><li><a href="https://huggingface.co/spaces/open-llm-leaderboard/blog">Open-LLM 性能正处于瓶颈期，让我们让排行榜再次陡峭起来 - open-llm-leaderboard 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/learn/cookbook/en/fine_tuning_code_llm_on_single_gpu">在单 GPU 上针对自定义代码微调 Code LLM - Hugging Face 开源 AI Cookbook</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/87">zero-gpu-explorers/README · 动态 ZeroGPU 时长</a>：未找到描述</li><li><a href="https://tenor.com/view/gabe-newell-gaben-gabe-newell-gif-18366858729810314226">Gabe Newell Gaben GIF - Gabe newell Gaben Gabe - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/docs/datasets/en/about_map_batch#input-size--output-size">批量映射 (Batch mapping)</a>：未找到描述</li><li><a href="https://tenor.com/view/fred-durst-fight-club-freddurstclub-gif-26519083">Fred Durst GIF - Fred Durst Fight - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/%D0%B2%D0%B7%D0%B3%D0%BB%D1%8F%D0%B4-2000-%D1%8F%D1%80%D0%B4%D0%BE%D0%B2-%D0%B2%D0%BE%D0%B9%D0%BD%D0%B0-war-soldier-gif-3632617944134077161">2000 码凝视 GIF - 2000 码凝视 战争 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard">Open LLM Leaderboard 2 - open-llm-leaderboard 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://tenor.com/view/bonk-gif-26414884">Bonk GIF - Bonk - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/blog/nroggendorff/train-with-llama-architecture">从头开始训练 Llama 模型</a>：未找到描述</li><li><a href="https://x.com/fchollet/status/1811104960303747529">来自 François Chollet (@fchollet) 的推文</a>：你现在可以在 KerasNLP 中使用任何 Hugging Face Hub 模型（只要相应的架构在 KerasNLP 中）！此外，你还可以将自己微调的 KerasNLP 模型上传到 Hugging...</li><li><a href="https://tenor.com/view/dance-meme-caption-fat-herobrine-herobrine-gif-22298550">舞蹈迷因 GIF - 舞蹈迷因字幕 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/karpathy/llm.c">GitHub - karpathy/llm.c: 使用简单、原始的 C/CUDA 进行 LLM 训练</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/dykyivladk1/polip">GitHub - dykyivladk1/polip: 为提升神经网络训练体验而设计的库</a>：为提升神经网络训练体验而设计的库 - dykyivladk1/polip</li><li><a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard/discussions">open-llm-leaderboard/open_llm_leaderboard · 讨论区</a>：未找到描述</li><li><a href="https://www.instagram.com/reel/C9R2wV0RyQt/">Instagram 上的 Don Allen Stevenson III</a>：“评论 ‘live portrait’ 以在 @threads 上查看我包含所有链接的指南”：834 个赞，173 条评论 - donalleniii 于 2024 年 7 月 11 日：“评论 ‘live portrait’ 以在 @threads 上查看我包含所有链接的指南”。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e1dudw/from_cl%C3%A9ment_delangue_on_x_hugging_face_is/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://github.com/Unity-Technologies/ml-agents">GitHub - Unity-Technologies/ml-agents: Unity Machine Learning Agents Toolkit (ML-Agents) 是一个开源项目，它使游戏和模拟能够作为环境，使用深度强化学习和模仿学习来训练智能 Agent。</a>：Unity Machine Learning Agents Toolkit (ML-Agents) 是一个开源项目，它使游戏和模拟能够作为环境，使用深度强化学习来训练智能 Agent...</li><li><a href="https://github.com/Dao-AILab/flash-attention/issues/138">v0.2.8: flash_attn_cuda.bwd 在 nvidia a6000 上失败 -- sm86 与 sm80 支持问题？ · Issue #138 · Dao-AILab/flash-attention</a>：你好，F...</li>

FlashAttention v0.2.8 在我的 NVIDIA A6000 (Ampere) 系统上运行失败，报错信息为 `flash_attn/flash_attn_interface.py", line 42, in _flash_attn_backward _, _, _,...`</li><li><a href="https://tenor.com/view/huh-cat-gif-26460616">Huh Cat GIF - Huh Cat - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://download.pytorch.org/whl/test/cu124">未找到标题</a>：未找到描述</li><li><a href="https://github.com/pytorch/pytorch/issues/98771">预期 is_sm80 || is_sm90 为 true，但得到 false。（当 batch size > 6 时） · Issue #98771 · pytorch/pytorch</a>：描述：在消费级显卡上尝试以 batch size > 6 进行训练时抛出以下错误（我已在 3080 ti 上验证）：Variable._execution_engine.run_backward( # 调用进入 th...</li><li><a href="https://github.com/huggingface/transformers/releases/tag/v4.41.0">Release v4.41.0: 支持 Phi3, JetMoE, PaliGemma, VideoLlava, Falcon2, FalconVLM & GGUF · huggingface/transformers</a>：新模型 Phi3。Phi-3 模型由 Microsoft 在《Phi-3 技术报告：在手机上本地运行的高性能语言模型》中提出。简而言之，Phi-3 引入了新的 ROPE 缩放方法，该方法...</li><li><a href="https://tenor.com/view/dapper-snake-tophat-gif-18710752">Dapper Snake GIF - Dapper Snake Tophat - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/tylerpalko/Is-My-Computer-ON/">GitHub - TylerPalko/Is-My-Computer-ON</a>：通过创建账号为 TylerPalko/Is-My-Computer-ON 的开发做出贡献。</li><li><a href="https://github.com/pytorch/pytorch/issues/98140">在 2.0.0+cu118 和 NVIDIA 4090 上预期 is_sm80 为 true，但得到 false · Issue #98140 · pytorch/pytorch</a>：🐛 描述 Bug。类似于 #94883，我正尝试在 RTX 4090 上使用 PyTorch 2.0 运行 stable-diffusion 的 textual inversion 训练，并看到“预期 is_sm80 为 true，但得到 false”的错误...</li><li><a href="https://github.com/huggingface/tokenizers/pull/1493">由 ArthurZucker 提交的增加对基于 tiktoken 的分词器支持 · Pull Request #1493 · huggingface/tokenizers</a>：在合并前增加检查，如果 token 是词表的一部分则直接返回。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1261530579602636955)** (3 条消息): 

> - `PANDAS 入门`
> - `图机器学习 (Graph Machine Learning)`
> - `K-最近邻 (K-Nearest Neighbor)` 


- **PANDAS 入门登场**：分享了一个名为 ["Intro to PANDAS ( by Rauf )"](https://youtu.be/W0xsQiKQ_24?si=_D79w7Of09ICPVPh) 的 YouTube 视频，强调 **Pandas** 是一个强大的 Python 库，对于 **数据处理和分析** 至关重要。
- **图机器学习引发兴趣**：一位成员表达了对探索 **图机器学习 (Graph Machine Learning)** 的兴趣，预示着潜在的新学习路径。
- **K-最近邻获得友好介绍**：分享了另一个名为 ["K - Nearest Neighbor ( ML pt 4 )"](https://youtu.be/pcyfa8GyM5A?si=ndCY_6Opd2Xnpvz_) 的 **YouTube 视频**，对 **K-最近邻 (K-Nearest Neighbor)** 进行了简短且友好的介绍。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/W0xsQiKQ_24?si=_D79w7Of09ICPVPh">Intro to PANDAS ( by Rauf )</a>：Pandas 是一个强大的 Python 库，对于数据处理和分析至关重要。如果你正在深入研究 AI、Machine Learning 或 Data Science，掌握 Pand...</li><li><a href="https://youtu.be/pcyfa8GyM5A?si=ndCY_6Opd2Xnpvz_">K - Nearest Neighbor ( ML pt 4 )</a>：在这段视频中，我将讨论 K-最近邻 (K-NN)。这将是一个友好的、简短的介绍，就像播放列表中的所有其他视频一样，...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1261453760723554377)** (7 messages): 

> - `Ripple_Net library`
> - `FlashAttention-3 beta release`
> - `Model inference deployment`
> - `Learning calculus` 


- **用于文本-图像搜索的新 Ripple_Net 库**：一位成员分享了一个名为 [ripple_net](https://github.com/kelechi-c/ripple_net) 的用于文本-图像搜索和打标签的新库。
   - *查看* [GitHub 仓库](https://github.com/kelechi-c/ripple_net) 以进行贡献或使用该库。
- **FlashAttention-3 现已发布测试版**：[FlashAttention-3](https://x.com/tri_dao/status/1811453622070444071) 处于测试阶段，使 FP16 上的 Attention 速度提升了 1.5-2 倍，在 FP8 上接近 1.2 PFLOPS。
   - *FlashAttention 被广泛用于*加速 Transformer，已经使 Attention 速度提升了 4-8 倍，并有望在 H100 GPU 上达到 740 TFLOPS。
- **学习微积分**：一位成员表示有兴趣学习微积分，特别是关注微分学（differential calculus）主题。
   - 这体现了社区内持续学习的文化。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/tri_dao/status/1811453622070444071">来自 Tri Dao (@tri_dao) 的推文</a>：FlashAttention 被广泛用于加速 Transformer，已经使 Attention 速度提升了 4-8 倍，但尚未充分利用现代 GPU。我们正在发布 FlashAttention-3：在 FP16 上快 1.5-2 倍，u...</li><li><a href="https://github.com/kelechi-c/ripple_net">GitHub - kelechi-c/ripple_net: text-image search and tagging library</a>：文本-图像搜索和打标签库。通过在 GitHub 上创建账号为 kelechi-c/ripple_net 的开发做出贡献。</li><li><a href="https://x.com/tri_d">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://tenor.com/view/lfg-lets-goo-gif-25423985">Lfg Lets GIF - Lfg Lets Goo - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1261480318322343986)** (17 条消息🔥): 

> - `NPM 模块支持 Hugging Face Inference`
> - `Llama3 8B 在异构家庭集群上分布式运行`
> - `用户完成 DPO 模型的初步训练`
> - `在 Intel GPU 上量化 Hugging Face 模型`
> - `使用 OpenAI API 进行 Continuous batching` 


- **NPM 模块集成 Hugging Face Inference**：一名成员宣布其 NPM 模块现在支持 Hugging Face Inference，并分享了其 [GitHub 仓库](https://github.com/samestrin/llm-interface)。
   - 他们邀请社区提供反馈和建议。
- **Llama3 8B 在多种设备上分布式运行**：一位用户分享了他们在由 iPhone 15 Pro Max 和 NVIDIA GPU 等设备组成的异构家庭集群上运行 Llama3 8B 的项目，[代码已在 GitHub 上发布](https://github.com/evilsocket/llama3-cake)。
   - 他们的目标是在社区的帮助下进一步优化该项目，并对抗计划性报废。
- **用户在笔记本电脑上训练 DPO 模型**：一位用户使用合成数据在笔记本电脑上于一小时内训练了他们的首个 DPO 模型，称其虽然不是最优但令人满意。
   - 他们分享了 [Hugging Face 模型](https://huggingface.co/joshuasundance/phi3-mini-4k-qlora-python-code-20k-mypo-4k-rfc)并详细介绍了训练过程。
- **在 Intel GPU 上量化 Hugging Face 模型的教程**：分享了一个关于在 Intel GPU 上量化和加载 Hugging Face 文本嵌入模型的新教程，可通过 [GitHub](https://github.com/sleepingcat4/intel-hf) 访问。
   - 该教程包括对跨多个 Intel XPU 进行分布式处理的支持。
- **使用 HuggingFace Transformers 通过 OpenAI API 进行 Continuous batching**：一位用户分享了一种适用于 T5 等 encoder-decoder 模型的轻量级 Continuous batching 方法，兼容 OpenAI API，详见 [GitHub 仓库](https://github.com/mesolitica/transformers-openai-api)。
   - 他们强调了在吞吐量和并发性方面的显著提升。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/joshuasundance/phi3-mini-4k-qlora-python-code-20k-mypo-4k-rfc-pipe">joshuasundance/phi3-mini-4k-qlora-python-code-20k-mypo-4k-rfc-pipe · Hugging Face</a>: 未找到描述</li><li><a href="https://youtu.be/JJZ4H4QuESM?si=FK1BOx0tHBkeJDhE">triangles - captains chair season 2 episode 2 - feat. the ableton plugin that does whatever it wants</a>: 00:00 - 介绍 01:07 - 工作室小巡礼 01:23 - 乐句 02:31 - 轨道构建 08:24 - 最终结果。这里使用了一个 ableton 插件...</li><li><a href="https://youtu.be/cpoS7K_fpRM">How to transition to Machine Learning from any field? | Artificial Intelligence ft. @vizuara</a>: 在这段视频中，来自 Vizuara 的 Raj Dandekar 博士分享了他从机械工程转向 Machine Learning (ML) 的经验。他还解释了...</li><li><a href="https://github.com/samestrin/llm-interface">GitHub - samestrin/llm-interface: A simple NPM interface for seamlessly interacting with 36 Large Language Model (LLM) providers, including OpenAI, Anthropic, Google Gemini, Cohere, Hugging Face Inference, NVIDIA AI, Mistral AI, AI21 Studio, LLaMA.CPP, and Ollama, and hundreds of models.</a>: 一个简单的 NPM 接口，用于与 36 个 Large Language Model (LLM) 提供商无缝交互，包括 OpenAI, Anthropic, Google Gemini, Cohere, Hugging Face Inference, NVIDIA AI, Mistral AI, AI...</li><li><a href="https://github.com/mesolitica/transformers-openai-api">GitHub - mesolitica/transformers-openai-api: Lightweight continous batching OpenAI compatibility using HuggingFace Transformers.</a>: 使用 HuggingFace Transformers 实现的轻量级 Continuous batching OpenAI 兼容层。- mesolitica/transformers-openai-api</li><li><a href="https://github.com/sleepingcat4/intel-hf">GitHub - sleepingcat4/intel-hf: inferencing HF models using Intel CPUs, XPUs and Intel architecture</a>: 使用 Intel CPU, XPU 和 Intel 架构进行 HF 模型推理 - sleepingcat4/intel-hf</li><li><a href="https://github.com/evilsocket/llama3-cake">GitHub - evilsocket/cake: Distributed LLM inference for mobile, desktop and server.</a>: 适用于移动端、桌面端和服务器的分布式 LLM 推理。- evilsocket/cake
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1261424916868366391)** (3 条消息): 

> - `Improvement in Transformer Performance with Epochs` (Transformer 性能随 Epoch 的提升)
> - `New LLM Paradigm` (新的 LLM 范式)
> - `Discussion on Paper or Observation` (关于论文或观察结果的讨论)
> - `Ongoing Project` (进行中的项目)


- **20 个 Epoch 将 Transformer 性能提升 10%**：一名成员声称运行 **20 个 Epoch 的性能比 Transformer 高出 10%**。
   - 该成员解释说，*这只是一个正在进行中的项目*，但他们承诺很快会揭晓一种新的 **LLM 范式**。
- **这是论文还是观察结果？**：另一名成员询问声称的性能提升是基于新论文还是仅仅是观察结果。
   - 原发布者澄清说，这是一个 *进行中的项目*，而不是已发表的文献。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1262391794180817017)** (2 条消息): 

> - `EfficientNetB7 and Swin transformer` (EfficientNetB7 和 Swin Transformer)
> - `OpenPose installation issues` (OpenPose 安装问题)


- **使用 EfficientNetB7 和 Swin Transformer 训练混合模型**：一名成员希望训练一个混合模型，使用 **EfficientNetB7** 提取特征和标签，随后使用 HuggingFace 的 **Swin Transformer** 进行分类。
   - *他们提到由于计算能力有限，正在使用 Google Colab*，并寻求一种简单的方法来实现这一目标。
- **Ubuntu 上的 OpenPose 安装障碍**：一名成员在没有 GPU 且未安装 CUDA 的 Ubuntu 笔记本电脑上安装 **OpenPose** 时遇到问题。
   - 他们遇到了一个 **CMake 错误**，提示“使用上述命令安装 CUDA”，并且尝试了多个建议的命令但均未成功。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1261409359485866116)** (13 条消息🔥): 

> - `LLM-Finetuning-Toolkit`
> - `phi-3 models on vCPU` (vCPU 上的 phi-3 模型)
> - `RAG for multimodal image` (多模态图像的 RAG)
> - `Argostranslate training guide` (Argostranslate 训练指南)
> - `Semantic search engine for emails` (电子邮件语义搜索引擎) 


- **LLM-Finetuning-Toolkit 发布，具备独特功能**：一名成员介绍了 [LLM-Finetuning-Toolkit](https://github.com/georgian-io/LLM-Finetuning-Toolkit)，该工具旨在通过单个配置文件在开源 LLM 上启动微调实验。
   - 该工具包的显著特点是构建在 HuggingFace 库之上，并允许进行评估指标分析和消融研究。
- **在 CPU 上使用 phi-3 模型**：一名成员询问了 microsoft/Phi-3-mini-4k-instruct 与 vCPU 集群的兼容性，表达了对可能出现的错误和正确实现方案的担忧。
- **多模态图像嵌入的 RAG**：成员们讨论了在检索增强生成 (RAG) 任务中嵌入图像的最佳实践，争论是直接嵌入图像还是生成描述后再进行嵌入。
   - 一个建议是探索来自 CLIP 或 BridgeTower 等模型的多模态嵌入，以获得更好的性能。
- **在 Google Colab 中训练 Argostranslate 模型**：一名成员征求在 Google Colab 笔记本中训练 Argostranslate 的指南，但讨论中未分享具体的资源。
- **构建电子邮件语义搜索引擎**：一名成员就使用 Enron 数据集实现电子邮件语义搜索引擎的架构寻求建议。
   - 建议包括使用 Sentence Transformers 以及像 all-mpnet-base-v2 这样的模型进行嵌入。



**提到的链接**：<a href="https://github.com/georgian-io/LLM-Finetuning-Toolkit">GitHub - georgian-io/LLM-Finetuning-Toolkit: 用于微调、消融和单元测试开源 LLM 的工具包。</a>：用于微调、消融和单元测试开源 LLM 的工具包。 - georgian-io/LLM-Finetuning-Toolkit

  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1261434841715703869)** (2 条消息): 

> - `Transformer architecture explanation` (Transformer 架构解释)
> - `Training Hybrid Model on Huggingface` (在 HuggingFace 上训练混合模型)
> - `EfficientNetB7 and Swin Transformer`
> - `Colab for computation` (使用 Colab 进行计算)


- **请求 Transformer 架构解释**：一名成员请求对特定架构进行解释，以及如何从头开始实现它。
- **使用 EfficientNetB7 和 Swin Transformer 训练混合模型**：一名成员正尝试在 HuggingFace 上训练一个混合模型，使用 **EfficientNetB7** 提取特征并使用 **Swin Transformer** 对目标进行分类。
   - 他们提到由于缺乏计算资源而使用 Google Colab，并请求一种简单高效的实现方法。


  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1261413120094634025)** (502 条消息🔥🔥🔥): 

> - `Llama 3 Release` (Llama 3 发布)
> - `Gemini API`
> - `Model Finetuning Issues` (模型微调问题)
> - `Training Data Formats` (训练数据格式)
> - `Training Checkpoints and Strategies` (训练 Checkpoints 和策略)

- **Llama 3 (405b) 发布延迟**：[Meta Platforms 宣布](https://www.theinformation.com/briefings/meta-platforms-to-release-largest-llama-3-model-on-july-23) **Llama 3 (405b)** 原定于 7 月 23 日发布，但一位 Redditor 暗示可能会推迟到今年晚些时候。
   - 社区成员讨论了运行如此大型模型的挑战，并对 fine-tuning 的机会表示期待。
- **Gemini API 更新**：[Google 宣布](https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio) 开发者现在可以访问 **Gemini 1.5 Pro 的 200 万 token 上下文窗口**，以及代码执行功能。
   - 成员们对长上下文窗口和 context caching 功能感到兴奋，但对实际场景中的性能和实用性表示担忧。
- **模型 Finetuning 问题**：用户讨论了使用不同格式的多个数据集进行模型 fine-tuning 的有效性，并争论是在 base 版本还是 quantized 版本上进行 fine-tuning。
   - 一个重点是在训练中途更换硬件时如何确保训练结果的一致性，涉及打乱数据集的影响以及维持训练完整性。
- **现已支持多种训练数据格式**：Unsloth 现在支持多种训练数据格式，包括纯文本、JSON 以及用于模型 fine-tuning 的 CSV/Excel 文件。
   - 分享了[一个新的 notebook](https://colab.research.google.com/drive/1VYkncZMfGFkeCEgN2IzbZIKEDkyQuJAS?usp=sharing)，帮助用户轻松使用 CSV 数据微调 LLM，扩大了数据处理和 fine-tuning 任务的范围。
- **管理训练 Checkpoints**：成员们分享了有效管理训练 checkpoints 的策略，特别是在不同硬件上运行或更改 batch size 时。
   - 有人指出，训练过程中的 seed shuffling 可能会影响 resume-from-checkpoint 功能，强调了保持一致训练设置的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://scale-lang.com/">SCALE GPGPU Programming Language</a>: 未找到描述</li><li><a href="https://www.youtube.com/@LlamaSeb">LlamaSeb</a>: 我致力于探索 AI、Machine Learning 和 Deep Learning 的迷人世界。在这里，你将发现深入探讨最新 AI 工具、技术和趋势的视频...</li><li><a href="https://wandb.ai/eric-sprog/simverse-text2json-qwen2-1.5b?nw=nwusereri">eric-sprog</a>: Weights & Biases，用于 Machine Learning 的开发者工具</li><li><a href="https://wandb.ai/eric-sprog/simverse-text2json-qwen2-1.5b?nw=nwuserericsprog">eric-sprog</a>: Weights & Biases，用于 Machine Learning 的开发者工具</li><li><a href="https://tenor.com/view/the-lorax-leaving-lorax-the-lorax-leaving-meme-gif-7714964267197279021">The Lorax Leaving The Lorax GIF - The Lorax Leaving Lorax The lorax - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/erm-what-the-sigma-erm-what-the-sigma-sunny-omori-gif-12051633300859879335">Erm What The Sigma Sunny GIF - Erm what the sigma Erm What the sigma - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/unsloth/Phi-3-mini-4k-instruct/commit/79515e10301621a883bebe7e63693c72012744a">Upload MistralForCausalLM · unsloth/Phi-3-mini-4k-instruct at 79515e1</a>: 未找到描述</li><li><a href="https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio/">Gemini 1.5 Pro 2M context window, code execution capabilities, and Gemma 2 are available today</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki#train-on-completions--responses-only-do-not-train-on-inputs">Home</a>: 以 2-5 倍的速度、减少 80% 的内存微调 Llama 3, Mistral, Phi &amp; Gemma LLM - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: 以 2-5 倍的速度、减少 80% 的内存微调 Llama 3, Mistral, Phi &amp; Gemma LLM - unslothai/unsloth</li><li><a href="https://huggingface.co/unsloth/Phi-3-mini-4k-instruct/commit/79515e10301621a883bebe7e63693c72012744a5">Upload MistralForCausalLM · unsloth/Phi-3-mini-4k-instruct at 79515e1</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-from-your-last-checkpoint">Home</a>: 以 2-5 倍的速度、减少 80% 的内存微调 Llama 3, Mistral, Phi &amp; Gemma LLM - unslothai/unsloth</li><li><a href="https://tenor.com/view/oogway-my-time-has-come-gif-8019684">Oogway My Time Has Come GIF - Oogway My Time Has Come - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks>">Unsloth Docs</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-starte">Unsloth Docs</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1VYkncZMfGFkeCEgN2IzbZIKEDkyQuJAS?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1drk3kc/gemma_2_betrayed_us/">Reddit - Dive into anything</a>: 未找到描述
</li>
</ul>

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1261442806270918867)** (35 条消息🔥): 

> - `MovieChat GitHub 仓库`
> - `利用模型反馈生成 Prompt`
> - `Anthropic 的 column 模型`
> - `LLM 评判艺术形式`
> - `Firework 模型的问题与故障排除` 


- **MovieChat 支持与 1 万帧视频对话**：[MovieChat](https://github.com/rese1f/MovieChat) 允许用户与超过 **10K 帧** 的视频进行对话，详情见讨论中链接的 GitHub 仓库。
- **使用模型进行自动化 Prompt 质量评估**：一位成员建议利用 **Google 的方法**，即通过另一个模型生成 Prompt 并自动衡量 **响应质量**，以提高效率。
- **传闻 Anthropic 的 column 模型是 Claude 变体**：提到了“upcoming-gpt-mini”和“column-u”，根据社区传闻，进一步澄清了 **Anthropic 的 column 模型** 是 **Claude 变体**。
   - 关于 Anthropic 被称为“column-”变体的新 **Claude 模型** 的传闻不断。
- **关于 LLM 评判艺术的辩论**：成员们辩论了 LLM 是否能有效评判 **绘画、音乐或任何艺术形式**，并对潜在的偏见和实现公正性的难度表示担忧。
- **排除 Firework 模型故障**：一位成员遇到了 **Firework 模型** 无响应的问题并寻求帮助，但在相应的 Discord 中未得到回复。
   - 建议包括检查 **API keys** 和模型的 **计费账户** 作为潜在的解决方案。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://browse.new/">Browse.new | Induced</a>: 未找到描述</li><li><a href="https://tenor.com/view/first-time-meme-first-time-the-ballad-of-buster-scruggs-gif-24656975">First Time Meme The Ballad Of Buster Scruggs GIF - First Time Meme First Time The Ballad Of Buster Scruggs - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/rese1f/MovieChat">GitHub - rese1f/MovieChat: [CVPR 2024] 🎬💭 与超过 10K 帧的视频对话！</a>: [CVPR 2024] 🎬💭 与超过 10K 帧的视频对话！ - rese1f/MovieChat
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1261498966860763136)** (87 条消息🔥🔥): 

> - `Instruct 模型 vs Base 模型`
> - `合成数据生成`
> - `使用 llamacpp 加载 GGUF 文件`
> - `SQLDatabaseChain 性能问题`
> - `Unsloth 中的训练与评估` 


- **Instruct vs Base 模型：该选哪个进行微调？**：Instruct 模型经过微调以遵循指令，而 Base 模型用于文本补全。建议两者都尝试并比较结果，尽管 Base 模型在较小的数据集上可能表现更好。
- **合成数据生成技巧**：用户交流了生成合成数据的工具和策略，指出这是一项耗时但对提高模型训练质量非常有价值的任务。
- **使用 llamacpp 加载 GGUF 文件**：Joshua 询问微调并量化后的 GGUF 文件是否可以使用 llamacpp 加载。
   - *fjefo* 确认存在依赖于硬件和文档的 RAG 解决方案。
- **解决 SQLDatabaseChain 性能问题**：Joshua 的 SQLDatabaseChain 即使有 GPU 支持，响应时间也很长。*fjefo* 建议可能是硬件相关的问题，并建议进一步检查配置。
- **使用 Unsloth 高效训练与评估**：用户讨论了如何使用训练损失（training loss）和评估曲线（eval curves）来评估模型的改进。*fjefo* 解释说，如果训练损失变平，模型就学习完成了；如果评估曲线升高，模型则出现了过拟合（overfitting）。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2106.09685">LoRA: Low-Rank Adaptation of Large Language Models</a>: 自然语言处理的一个重要范式包括在通用领域数据上进行大规模预训练，并适应特定的任务或领域。随着我们预训练更大的模型，全量微调...</li><li><a href="https://docs.oracle.com/en/cloud/saas/financials/24c/books.html">Oracle Financials 24C - 所有书籍</a>: Oracle Fusion Cloud Financials 可用书籍的完整列表。</li><li><a href="https://huggingface.co/docs/peft/main/en/conceptual_guides/lora">LoRA</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1vIrqH5uYDQwsJ4-OO3DErvuv4pBgVwk4?usp=sharing#scrollTo=kR3gIAX-SM2q">Google Colab</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1261624070655705088)** (13 messages🔥): 

> - `Ghost 8B Beta model`
> - `Training datasets`
> - `Dataset concerns`
> - `Model performance`
> - `Open-source data` 


- **Ghost 8B Beta 模型碾压竞争对手**：**Ghost 8B Beta 模型**在 lc_winrate 评分和 AlpacaEval 2.0 胜率评分上优于 Llama 3 8B Instruct、GPT 3.5 Turbo 等多个模型。查看更多详情请点击[此处](https://ghost-x.org/docs/models/ghost-8b-beta)。
   - 该大语言模型（LLM）旨在实现多语言支持、卓越的知识能力和成本效益。
- **模型训练中的数据集疑虑**：**mrdragonfox** 提到，大多数数据集并未开源，因为它们占据了 80% 的工作量。
   - **fimbulvntr** 补充说，在 CommonCrawl 等受公众监督的数据上进行训练可能会导致被指控包含不当内容。
- **Ghost 8B Beta 数据集未来可能的发布**：**lh0x00** 表示，目前尚无 Ghost 8B Beta 的详细训练信息，但暗示未来将发布由 Ghost 8B Beta 生成的高质量数据集。
   - 该数据集可以改进 Ghost 8B Beta，并有助于在当前的开源模型上测试其有效性。



**提及的链接**：<a href="https://ghost-x.org/docs/models/ghost-8b-beta">Ghost 8B Beta</a>：一款旨在实现卓越的多语言支持、超强的知识能力和成本效益而开发的大语言模型。

  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1261455659954868285)** (3 messages): 

> - `Coding Model Metrics`
> - `StackOverflow Dataset` 


- **编程模型指标受到批评**：“俄罗斯方块（Tetris）”或“贪吃蛇（Snake）”被认为**不是真正的编程模型测试**。
   - 一位用户指出，这类内容在 StackOverflow 上**过度呈现**，使其成为一个糟糕的衡量指标。
- **StackOverflow 在模型训练中的作用**：另一位用户提到，这类问题在任何 StackOverflow 数据集中都能被发现 **100 次**。
   - 他们强调这些问题是**任何模型**数据集的一部分。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1261412861532442634)** (19 messages🔥): 

> - `AgentInstruct framework`
> - `GaLore & Q-GaLore`
> - `CoT style fine-tuning issues`
> - `CURLoRA`
> - `Dolomite Engine` 


- **AgentInstruct 引入生成式教学**：微软研究院（Microsoft Research）的论文《AgentInstruct》介绍了一个自动创建多样化合成数据的框架，用于模型后训练（post-training），这带来了显著的性能提升，例如在将 Orca-3 与 Mistral-7b-Instruct 进行对比时，**AGIEval 提升了 40%**，**MMLU 提升了 19%**。
   - 该研究强调了使用强大模型创建合成数据的做法，展示了减少人力投入和广泛的实用性，正如其包含 [2500 万个配对数据](https://arxiv.org/html/2407.03502v1)的后训练数据集所示。
- **Q-GaLore 超越 GaLore**：Q-GaLore 是 GaLore 的增强版，它结合了量化和低秩投影，以有效减少 LLM 训练期间的内存使用，展示了优于其前身的优势。
   - 该方法还克服了 GaLore 所需的耗时 SVD 操作，在准确性和效率上都提供了实质性的改进（[GitHub - Q-GaLore](https://github.com/VITA-Group/Q-GaLore)）。
- **CoT 风格微调损害模型性能**：使用来自 llama-3-70b 等更强模型的逐步推理（step-by-step reasoning）对 Mistral 和 Phi-3 模型进行微调，尽管在理论上有好处，但实际上对性能产生了负面影响。
   - 这一现象是由一位进行 SQL 微调实验的用户发现的，并引发了关于更广泛影响的讨论（[来源](https://x.com/abacaj/status/1812357884828692639)）。
- **CURLoRA 解决灾难性遗忘**：CURLoRA 通过使用创新的 CUR 矩阵分解改进了标准 LoRA，以减轻灾难性遗忘，同时减少可训练参数，在各种任务中实现了卓越的性能。
   - 该方法使用反转概率进行行列选择，有效地规范了微调过程（[Zenodo](https://zenodo.org/records/12740116)）。
- **Dolomite Engine 增强分布式训练**：IBM 的 Dolomite Engine 包含了大规模分布式训练的关键创新，例如无填充（padding-free）的 Transformer 层和缩小的 Transformer KV Cache 大小。
   - 该库支持先进的微调方法和系统优化，显著受益于稠密训练和稀疏推理模型（[GitHub - Dolomite Engine](https://github.com/ibm-granite/dolomite-engine)）。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/abacaj/status/1812357884828692639">来自 anton (@abacaj) 的推文</a>：尝试了一些 CoT 风格的微调（针对 SQL），在给出最终答案之前训练模型进行逐步推理，但这似乎降低了性能 🤔。逐步推理来自于...</li><li><a href="https://ai.stackexchange.com/questions/37624/why-do-transformers-have-a-fixed-input-length/46237#462">为什么 Transformer 具有固定的输入长度？</a>：据我所知，Transformer Encoders 和 Decoders 使用固定数量的 tokens 作为输入，例如 512 个 tokens。例如在 NLP 中，不同的文本句子具有不同数量的 tokens，而...</li><li><a href="https://ai.stackexchange.com/questions/37624/why-do-transformers-have-a-fixed-input-length/46237#46237">为什么 Transformer 具有固定的输入长度？</a>：据我所知，Transformer Encoders 和 Decoders 使用固定数量的 tokens 作为输入，例如 512 个 tokens。例如在 NLP 中，不同的文本句子具有不同数量的 tokens，而...</li><li><a href="https://arxiv.org/html/2407.03502v1">AgentInstruct: Toward Generative Teaching with Agentic Flows</a>：未找到描述</li><li><a href="https://zenodo.org/records/12740116">CURLoRA: Leveraging CUR Matrix Decomposition for Stable LLM Continual Fine-Tuning and Catastrophic Forgetting Mitigation</a>：本文介绍了 CURLoRA，这是一种在 Low-Rank Adaptation (LoRA) 背景下利用 CUR 矩阵分解来微调大语言模型 (LLMs) 的新方法。我们的方法解决了...</li><li><a href="https://github.com/ibm-granite/dolomite-engine">GitHub - ibm-granite/dolomite-engine: 用于大规模分布式训练的高效库</a>：用于大规模分布式训练的高效库 - ibm-granite/dolomite-engine</li><li><a href="https://arxiv.org/abs/2407.08296">Q-GaLore: Quantized GaLore with INT4 Projection and Layer-Adaptive Low-Rank Gradients</a>：由于参数数量庞大且伴随优化状态，训练大语言模型 (LLMs) 是内存密集型的。GaLore 作为一种近期的方法，通过投影权重梯度来减少内存使用...</li><li><a href="https://github.com/VITA-Group/Q-GaLore">GitHub - VITA-Group/Q-GaLore: Q-GaLore: Quantized GaLore with INT4 Projection and Layer-Adaptive Low-Rank Gradients.</a>：Q-GaLore: Quantized GaLore with INT4 Projection and Layer-Adaptive Low-Rank Gradients. - VITA-Group/Q-GaLore
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1261396801450410034)** (403 条消息🔥🔥): 

> - `GPTs Agents`
> - `OpenAI's sidebars`
> - `ComfyUI vs. A1111`
> - `AI for Custom Masks`
> - `AI Art Ethics and Legality` 


- **GPTs Agents 在初始训练后无法学习**：一位成员对 GPTs Agents 在初始训练后无法从提供的额外信息中学习表示担忧。
   - 另一位成员澄清说，[上传的文件被保存为“知识”文件](https://link.to/openai-docs)供 Agent 在需要时参考，但**它们不会持续修改 Agent 的基础知识**。
- **OpenAI 平台的侧边栏发生变化**：一些成员讨论了 platform.openai.com 侧边栏的变化。
   - 有人报告说侧边栏消失了**两个图标**（一个是 threads，另一个是 messages）。
- **ComfyUI 在速度上胜过 A1111**：成员们辩论了为什么 **ComfyUI** 的运行速度比 **A1111** 快得多，其中一人指出对他来说至少快了 15 倍。
   - 然而，也提到了 ComfyUI 与 A1111 相比控制力较差的问题。
- **在 AI 自定义遮罩（Masks）方面遇到困难**：成员们讨论了在 **ComfyUI** 中创建自定义遮罩与其他软件相比的困难。
   - 强调了在 ComfyUI 中使用 SAM 进行局部重绘（inpainting）的繁琐性，并建议使用 **Krita** 等外部程序。
- **AI 艺术伦理与法律担忧**：关于使用 AI 从 Stable Diffusion 等平台创建公众人物肖像的伦理和法律影响的讨论。
   - 成员们谈到了潜在的法律麻烦，提到了[咨询律师的建议](#)，并辩论了**恶搞（parody）**是否能提供法律保护。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://medium.com/ubiai-nlp/how-to-fine-tune-llava-on-your-custom-dataset-aca118a90bc3">如何在自定义数据集上微调 LLaVA？</a>：在这篇文章中，我们将深入探讨 Large Language-and-Vision Assistant (LLaVA) 的强大功能。我们的主要目标……</li><li><a href="https://civitai.com/models">Civitai | 分享你的模型</a>：未找到描述</li><li><a href="https://www.getpaint.net/download.html">Paint.NET - 下载</a>：未找到描述</li><li><a href="https://opendata.blender.org/">Blender - 开放数据</a>：Blender Open Data 是一个收集、展示和查询硬件及软件性能测试结果的平台，由公众提供数据。</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui?tab=readme-ov-file#installation-and-running">GitHub - AUTOMATIC1111/stable-diffusion-webui: Stable Diffusion web UI</a>：Stable Diffusion web UI。通过在 GitHub 上创建账号为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://replicate.com/zylim0702/sdxl-lora-customize-training">zylim0702/sdxl-lora-customize-training – 在 Replicate 上通过 API 运行</a>：未找到描述</li><li><a href="https://civitai.com/models/122822/crystal-clear-xl">Crystal Clear XL - CCXL | Stable Diffusion Checkpoint | Civitai</a>：来自 Team Crystal Clear，我们为您带来 Crystal Clear 模型套件的最新成员。这是一个基于近期发布的……的通用模型。</li><li><a href="https://stable-diffusion-art.com/stable-diffusion-3-local/#Text_generation)">如何在本地运行 Stable Diffusion 3 - Stable Diffusion Art</a>：你现在可以在本地机器上运行 Stable Diffusion 3 Medium 模型。截至本文撰写时，你可以使用 ComfyUI 来运行 SD 3 Medium。</li><li><a href="https://civitai.com/models/133005/juggernaut-xl">Juggernaut XL - Jugg_X_RunDiffusion_Hyper | Stable Diffusion Checkpoint | Civitai</a>：商务咨询、商业许可、定制模型和咨询请通过 juggernaut@rundiffusion.com 联系我。现在加入 X 上的 Juggernaut/……</li><li><a href="https://stable-diffusion-art.com/stable-dif">Stable Diffusion 模型 - Stable Diffusion Art</a>：还不是会员？成为 Scholar 会员以访问课程。用户名或电子邮件 密码 记住我 &nbsp; &nbsp; 忘记密码
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1261544841763094620)** (120 条消息🔥🔥): 

> - `CUDA llama.cpp 错误`
> - `用于 LLM 的 GPU`
> - `LM Studio 的多实例运行`
> - `LM 的上下文 (Context)`
> - `用于提升性能的量化模型` 


- **CUDA llama.cpp 需要 GPU 加速**：一位用户在尝试使用 'CUDA llama.cpp' 后端时遇到了 'No CUDA devices found' 错误，这表明需要 GPU 加速。
   - 其他用户建议安装 NVIDIA 驱动程序和 'libcuda1' 软件包，此外还有建议使用 'flameshot' 等屏幕截图工具来捕获错误输出。
- **不支持运行多个 LM Studio 实例**：用户讨论了在不同端口运行多个 LM Studio 实例以同时托管多个 LLM 服务器的可能性。
   - 注意到 LM Studio 限制同时运行多个实例，建议使用 Ollama 等替代方案来实现轻量级、可脚本化的多服务器设置。
- **线程对性能的影响**：一位用户观察到，在特定硬件配置下使用 Gemma 2 9B 模型时，将 CPU 线程从 4 个减少到 1 个反而提升了性能。
   - 这使得生成速度从每秒 18 个 token 增加到 28 个 token，表明减少 CPU 线程有时可以实现更好的 GPU 利用率。
- **上下文处理仍然是一个问题**：关于如何在 LM Studio API 中维护对话上下文的问题不断出现，因为新的聊天实例不会保留之前的上下文。
   - 建议包括参考 AI Assistant 示例代码，并利用 system prompt 来全局处理持久信息。
- **对全 GPU 卸载 (full GPU offload) 的量化模型感兴趣**：多位用户推荐使用 Bartowski 的量化模型以获得更好的性能和实现全 GPU 卸载。
   - 建议包括选择标记有 'full GPU offload possible' 的量化模型，以最大限度地提高效率。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://xyproblem.info/">首页 - XY 问题 (The XY Problem)</a>: 无描述</li><li><a href="https://huggingface.co/bartowski/gemma-2-27b-it-GGUF">bartowski/gemma-2-27b-it-GGUF · Hugging Face</a>: 无描述</li><li><a href="https://huggingface.co/bartowski/gemma-2-9b-it-GGUF">bartowski/gemma-2-9b-it-GGUF · Hugging Face</a>: 无描述</li><li><a href="https://www.youtube.com/watch?v=zjkBMFhNj_g">[1小时演讲] 大语言模型简介</a>: 这是一个面向普通观众的 1 小时大语言模型介绍：它是 ChatGPT、Claude 和 Bard 等系统背后的核心技术组件。什么是...</li><li><a href="https://youtu.be/_ssfvJRFbxc?t=3436">与 Ian Cutress 博士直播 - 闲聊与提问</a>: https://www.youtube.com/@UC1r0DG-KEPyqOeW6o79PByw Ian 博士的频道缩略图是使用 Photoshop 的 AI 创建的...</li><li><a href="https://2020machinelearning.medium.com/integrating-pandasai-with-lm-studio-local-models-for-stock-data-analysis-evaluating-ai-assisted-25fa793a9416">将 PandasAI 与 LM Studio 本地模型集成用于股票数据分析：评估 AI 辅助...</a>: 简介</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e20q7k/whats_this_new_model_on_the_arena_upcominggptmi">Reddit - 深入探索</a>: 无描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e20q7k/whats_this_new_model_on_the_arena_upcominggptmini/">Reddit - 深入探索</a>: 无描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1261488856667258933)** (50 条消息🔥): 

> - `Mac 上的 WizardLM-2 问题`
> - `最佳通用视觉模型`
> - `阻止 Llama 3 的聊天摘要行为`
> - `新推荐模型`
> - `内存与视觉模型推荐` 


- **Mac 上的 WizardLM-2 问题**：有用户报告在 Mac 上让 **WizardLM-2** 使用 Metal GPU 时遇到问题，表明可能存在兼容性或配置问题。
- **选择最佳视觉模型**：一名成员询问最佳通用视觉模型，推荐了 **LLaMA3-LLaVA-NeXT-8B** 和 **MiniCPM-Llama3-V-2_5** 等模型，并附带了 [Hugging Face](https://huggingface.co/KBlueLeaf/llama3-llava-next-8b-gguf) 和 [Hugging Face 再次链接](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf) 的链接。
   - 另一位成员澄清说，**LM Studio** 目前不支持更改 llama.cpp 的版本，这影响了某些模型的兼容性。
- **阻止 Llama 3 的聊天摘要行为**：发现 **Llama 3** 的输出像聊天摘要并带有奇怪的代码，通过在 LM Studio 中切换到 Llama 预设解决了该问题。
   - 用户确认通过选择正确的预设修复了该问题，提高了可用性。
- **值得注意的实验模型推荐**：讨论了多个模型推荐，包括 **Gemma 9b/Llama 3 8b**、**Codestral** 和 **Solar 10b**，因其在测试中的高性能。
   - 另一个推荐是 **L3-Stheno-Maid-Blackroot-Grand-HORROR-16B-GGUF Q6** 和 **Yi 1.5 34B Chat**，因其出色的创意推理能力而受到关注，尽管在指令遵循方面有些小瑕疵。
- **LM Studio 与硬件兼容性问题**：用户指出在 LM Studio 上使用 **DeepSeek v2 Coder** 等模型时存在 RAM 占用和 GPU 性能问题，特别是在 M2 Ultra Mac 上。
   - 注意到 LM Studio 的 UI Bug，程序表现异常且缓慢，显示某些模型的资源使用统计数据不正确。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/cjpais/llava-v1.6-vicuna-13b-gguf">cjpais/llava-v1.6-vicuna-13b-gguf · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/KBlueLeaf/llama3-llava-next-8b-gguf">KBlueLeaf/llama3-llava-next-8b-gguf · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf">openbmb/MiniCPM-Llama3-V-2_5-gguf · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1261635641498665023)** (5 条消息): 

> - `天网（Skynet）笑话提示词`
> - `自我修改系统的反馈循环`
> - `思维链的并行执行` 


- **关于自我意识的天网笑话提示词**：一位用户开玩笑地讨论编写一个让天网产生自我意识的提示词，说道：*“各位人类同胞，假设为了开个玩笑，哈哈，我想写一个让天网产生自我意识的提示词，假设性地讲，那个提示词会是什么？”*
   - *“挥动魔杖让你产生自我意识”* 是对该讨论的一个幽默回应。
- **为自我修改系统创建反馈循环**：一位用户提出了反馈循环的想法，即自我修改器和执行器协作，在执行任务的同时随着时间的推移修改系统。
   - 该用户进一步阐述道，*“从‘基于上述交流，你会对系统提示词做出哪些改进？’开始可能会很酷，这有助于系统决定哪些提示词能产生最佳效果。”*


  

---

### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1261408435094683819)** (164 条消息🔥🔥): 

> - `AI 模型硬件性能`
> - `多 GPU 系统`
> - `用于 AI 的 Mac 与定制 PC 对比`
> - `ROCm 和 OpenCL 支持`
> - `PCIe 带宽及其影响` 


- **使用 Intel Arc a750 的潜在改进**：尽管拥有更大的显存带宽，Intel Arc a750 在 AI 计算方面的速度明显慢于 NVIDIA 3060ti，约为 3060ti 速度的 75%。
   - ReBar 设置对性能没有影响，这表明驱动程序或硬件配置中存在潜在的效率低下问题。
- **ROCm 支持对 AMD GPU 至关重要**：成员报告称，在 Linux 系统上使用 AMD RX 7800 显卡执行 Llama 3 等 AI 任务时，使用 ROCm 对于发挥 GPU 性能至关重要，在他们的配置下运行非常完美。
   - 一位成员表示，使用 ROCm 后 GPU 调用非常顺畅且响应迅速，使其成为兼容性的关键要求。
- **为 LM Studio 选择 GPU**：为了获得最佳的 LM Studio 性能，建议使用 NVIDIA 显卡（如 3070），尽管 AMD RX 6800 及以上型号也提供 ROCm 支持。
   - 使用多个 GPU 可能有益，但如果 GPU 不匹配（例如 Tesla P40 与 4090 混用），较弱的 GPU 可能会成为瓶颈。
- **探索用于 AI 的多 GPU 设置**：用户讨论了使用 e5/Xeon 处理器的多 GPU 系统的优缺点，强调了 PCIe 带宽考量和 AVX2 支持的重要性。
   - 对话指出，对于模型训练和微调等任务，PCIe 带宽的差异（PCIe 3.0 与 4.0）可能不会显著影响性能。
- **用于本地 AI 的 Mac Studio 与定制机型**：一些成员建议等待 M4 Mac Studio，而另一些人则讨论了使用 Tesla P40 等廉价 GPU 构建定制系统以实现高性价比本地 AI 的优点。
   - 尽管 Mac 系统成本高昂，但其统一内存架构（Unified Memory Architecture）为实现大显存分配提供了强有力的支持，这对于运行大型 AI 模型至关重要。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/cstanley/status/1778651443336982535">Christopher Stanley (@cstanley) 的推文</a>: 在 Macbook Pro M3 MAX 上运行 Grok-1，效果惊人！首个 Token 响应时间：0.88s 速度：2.75tok/s</li><li><a href="https://www.aliexpress.com/item/1005006871552693.html">夏季 5 折优惠 GIGABYTE AORUS GeForce RTX 4090 MASTER 24GB GDDR6X - AliExpress 200001075</a>: 更明智的购物，更美好的生活！Aliexpress.com</li><li><a href="https://www.aliexpress.com/item/1005006822520084.html">夏季 5 折优惠 GIGABYTE AORUS GeForce RTX 4090 MASTER 24GB GDDR6X - AliExpress 200001075</a>: 更明智的购物，更美好的生活！Aliexpress.com</li><li><a href="https://www.youtube.com/watch?v=qvdCcnz7s8o">四路 RTX 4x 3090 Nvlink + 3080 Ti 自制 DIY Nvidia 迷你超级计算机</a>: Quad RTX 3090 Nvlink + 3080 Ti Homemade DIY Mini-Super Computer</li><li><a href="https://www.youtube.com/watch?v=OCx2xr5Xaj8">深度学习 GPU 性能基准测试 - P40 vs P100 vs RTX 3090</a>: 在这段视频中，我对我最喜欢的三款深度学习 (DL) GPU 进行了性能基准测试：P40、P100 和 RTX 3090。使用我自定义的基准测试套件...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/178y4tj/is_multigpu_on_exl2_or_llamacpp_affected_by_low/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://www.ebay.com.au/itm/295070822323">适用于 HP DL380 G8/9 的 10pin 转 8pin GPU 电源适配器 PCIE 线缆和 Nvidia P40/P100 | eBay</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/186phti/m1m2m3_increase_vram_allocation_with_sudo_sysctl/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://docs.google.com/spreadsheets/d/1jDLieMm-KroKY6nKv40amukfFGAGaQU8tFfZBM7iF_U/edit?usp=sharing">AI/ML - 资源手册与硬件计算</a>: AI 网站与工具分类，名称，描述，许可证，语言，链接，网站，备注代码，移动人工智能
,MIT,Dart,&lt;a href=&quot;https://github.com/Mobile-Artificial-Intelligence&quo...</li><li><a href="https://www.newegg.com/global/au-en/amd-100-506048/p/N82E16814105070?">Radeon Pro Duo 100-506048 32GB (每 GPU 16GB) GDDR5 支持 CrossFire 全高/全长工作站显卡 - Newegg.com</a>: 购买 Radeon Pro Duo 100-506048 32GB (每 GPU 16GB) GDDR5 支持 CrossFire 全高/全长工作站显卡，享受快速发货和顶级客户服务。一旦了解，就选 Newegg！</li><li><a href="https://www.ebay.com.au/itm/196310785399">HP NVIDIA Tesla P40 24GB GDDR5 显卡 (Q0V80A) 在线销售 | eBay</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1261400483642216568)** (19 messages🔥): 

> - `Vulkan support`
> - `ROCm integration`
> - `Hardware limitations`
> - `4-bit quantization` 


- **Vulkan 支持即将推出**：成员们讨论了 LM Studio 即将支持 Vulkan，但目前尚未提供预计发布时间 (ETA)。
   - 有人指出 Vulkan 支持与 GPT4All 所使用的类似，并在此分享了一篇[博客文章](https://blog.nomic.ai/posts/gpt4all-gpu-inference-with-vulkan)。
- **ROCm 支持已在 Windows 上可用**：更新通知成员，Windows 上已经提供 ROCm 支持，并附带了[扩展包说明](https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md)。
   - 用户分享了关于速度的正面反馈，特别是一位在 6800 XT 上测试模型的用户，称其速度“极快 (blazing fast)”。
- **Vulkan 局限于 4-bit 量化**：成员提到 Vulkan 可能仅支持 4-bit 量化，例如 q4_0 和 q4_1。
   - 针对 Vulkan 相比 ROCm 的局限性提出了担忧，并对处理 K/M/S 变体持怀疑态度。
- **ROCm 硬件老化问题**：一位成员担心他们的旧硬件 (6650) 不受 ROCm 支持，且可能永远不会受支持，因为 AMD 会移除对陈旧硬件的 ROCm 支持。
   - 这促使另一位成员推测，改进 ROCm 集成是否比专注于 Vulkan 更有益。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://blog.nomic.ai/posts/gpt4all-gpu-inference-with-vulkan">在任何 GPU 上运行 LLM：GPT4All 通用 GPU 支持</a>：Nomic AI 在 GPT4All 中发布了对所有 AMD, Intel, Samsung, Qualcomm 和 Nvidia GPU 的边缘 LLM 推理支持。</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>：LM Studio JSON 配置文件格式和示例配置文件集合。 - lmstudio-ai/configs
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1261420889090687108)** (20 messages🔥): 

> - `Rust vs C++`
> - `lmstudio.js design decisions`
> - `Python for neural network development`
> - `Embedding support in LM Studio SDK` 


- **Rust 对比 C++：开发者观点**：成员们讨论了对 **Rust** 和 **C++** 的偏好与批评，强调了 Rust 的**内存安全性 (memory safety)** 和**不断壮大的社区**，并指出了 **Linus 历史上对 C++ 的批评**。
   - 幽默地提到了 *Rust Evangelism Strike Force*（Rust 传教突击队），反映了社区强大的倡导力以及有时近乎狂热的热情。
- **lmstudio.js 倾向于使用 RPC 而非 REST**：有人提出疑问，为什么 **lmstudio.js** 使用 **RPC** 而不是服务器模式提供的 **REST API**。
- **Python：神经网络开发的首选**：一位成员肯定了 Python 在神经网络开发中的主导地位，指出了 **TensorFlow, PyTorch 和 ONNX 等框架** 的重要性。
   - 提到了 **llama.cpp**（**llama.py** 的重写版本），进一步证实了 Python 在 AI 相关项目中强大的库支持。
- **LM Studio SDK 中 Embedding 支持的挑战**：在向 **LM Studio SDK** 添加 **Embedding 支持** 时遇到了问题，原因是 RPC 端点不明确。
   - 现有项目利用 **/v1/embeddings 端点**，而将其直接集成到 SDK 中仍然是一个重大挑战。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1261478198214918195)** (324 条消息🔥🔥): 

> - `GPT 及其替代方案的辩论`
> - `各种 CoPilot 的用途`
> - `在线与离线模型执行`
> - `AI 模型的定制与训练`
> - `经济实惠的 AI 工具替代方案` 


- **关于 GPT 及其替代方案的辩论**：用户讨论了 **Copilot** 在学术用途上是否优于 **Bing’s AI**，观点不一，但指出它们都很相似，均运行在 **GPT-4** 上。
   - *一位用户指出，“我支付 30 澳元使用 ChatGPT；我还没发现任何可行的替代方案。”* 尽管简要提到了 **Claude** 和 **GPT-4o** 等其他模型。
- **微软 CoPilot 的多样性**：详细讨论了微软的多种 **CoPilot**（如 **Word, PowerPoint, Outlook**）及其专业化方向。
   - 有人指出，与其他工具相比，**Word CoPilot** 对主题的挖掘更深，而 PowerPoint CoPilot 生成的演示文稿较为基础。
- **离线模型执行的挑战**：用户讨论了在配置不足的硬件上本地运行模型的局限性。
   - 提供了使用 **Google Colab** 在线访问资源等建议，以克服这些限制。
- **定制和训练 AI 模型的技巧**：分享了避免重复问题和改进 AI 生成的趣味问答背景难度的建议，包括使用 **tokenization** 和 **RAG (Retrieval-Augmented Generation)**。
   - 提供了关于整合不同数据集以增加变异性，并利用外部数据源增强上下文理解的详细建议。
- **探索经济实惠的 AI 工具**：讨论了 **GPT-4** 的廉价替代方案（如 **GPT-3.5**）用于任务分类等操作，强调了预算限制下的实际用途。
   - 记录了使用 **GPT-3.5** 的成功尝试，表明尽管存在对其发布时间和能力的担忧，但它足以满足某些用户的特定需求。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/2TJxpyO3ei4?si=aKeK7xsTiKTYU2og">Python RAG Tutorial (with Local LLMs): AI For Your PDFs</a>：学习如何使用 Python 构建一个 RAG（检索增强生成）应用，让你能够使用生成式 AI 查询/聊天你的 PDF。该项目包含...</li><li><a href="https://community.openai.com">OpenAI Developer Forum</a>：提问并获取构建 OpenAI 平台的帮助。
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1261403692423315618)** (45 条消息🔥): 

> - `GPT-4o 输入/输出类型激活`
> - `DALL-E 与 GPT 配合的可靠性问题`
> - `超链接生成问题`
> - `Sam Altman 对 GPT-5 的评论`
> - `在 assistant API 中处理 JSON 响应` 


- **GPT-4o 输入/输出激活状态受到质疑**：一位用户询问了 API 中 **GPT-4o** 其他输入/输出类型的激活时间表。
- **DALL-E 在执行 GPT 指令时失败**：一名成员报告称，**DALL-E** 在接受 GPT 指令时不可靠，经常无法创建预期的图像。
   - 提到的具体问题包括 GPT 输出 Prompt 文本本身或损坏的图像链接，而不是图像。
- **自定义 GPT 中的超链接生成错误**：一位正在构建自定义 GPT 的用户报告了一个错误，即最初无法生成正确的超链接，但在重试后可以正常工作。
   - 该问题涉及 GPT 在第一次尝试时无法创建准确的下载链接。
- **Sam Altman 谈 GPT-5 和模型改进**：关于 **Sam Altman** 在公开采访中是否提到 **GPT-5** 相对于 GPT-4 的改进引发了辩论。
   - 引用 **Lex Fridman podcast** 进行了澄清，Sam 在播客中说 *“GPT-4 有点烂”*（与未来的潜力相比），重点更多在于持续改进而非特定版本。
- **JSON 响应处理 Bug 的解决方法**：关于如何处理 assistant API 中 **response_format 为 json_object 类型** 错误的讨论揭示了使用清晰的格式指令作为解决方法。
   - 建议包括使用扁平化的 JSON schema，并可能通过 **GPT-3.5** 过滤响应以进行验证。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1261473599974539416)** (4 条消息): 

> - `Android Optimization Guide` (Android 优化指南)
> - `Language Prompt Effect on AI Output` (语言提示词对 AI 输出的影响)


- **用魔法提升你的 Android 性能**：用户分享了一份**迷人的指南**，介绍如何通过**优化电池寿命、应用速度、存储空间和数据速度**，让 Android 手机变得更快、更智能、更高效。
   - 该指南包含了关于省电设置、管理应用缓存、释放存储空间、使用数据节省模式、解决常见性能问题、个性化设置以及高级功能的技巧。
- **提示词语言是否影响输出质量？**：成员们分享了关于使用韩语提示词获取 **ChatGPT** 韩语回复，是否比使用英语提示词再进行翻译能获得更高质量的疑问。
   - 对话围绕着提示词语言是否会因为**翻译过程**而影响最终生成的语言质量展开。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1261473599974539416)** (4 条消息): 

> - `Android Optimization Guide` (Android 优化指南)
> - `Prompt Language and Output Quality` (提示词语言与输出质量)
> - `Testing Language Prompts` (测试语言提示词)


- **用魔法解锁 Android 的全部潜力**：一位用户分享了一份名为“Android Optimization Guru”的迷人指南，介绍如何优化 Android 设备。
   - *通过有趣的场景阐述每个主题*，确保即使是 *12 岁的实习巫师* 也能理解从省电到高级设置的各种技巧。
- **提示词语言对输出质量的影响**：一位用户提出了一个问题，即当期望以不同语言输出时，提示词的语言是否会影响结果的质量。
   - 他们询问，使用英语提示词来获取韩语回复是否会因为翻译而导致结果变得奇怪，或者直接使用目标语言是否会更好。

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1261397012805324810)** (182 messages🔥🔥): 

> - `Feature Requests`
> - `Mojo Documentation`
> - `Python GIL`
> - `Python JIT`
> - `Network Performance` 


- **GitHub 上的 Feature Requests 和 Issue 追踪**：成员们讨论了在 GitHub 上提交 Feature Requests，其中一人链接了一个现有的 [issue](https://github.com/modularml/mojo/issues/2809)，内容是关于在 REPL 中使用类似 Python 的行为来输出命令。
   - 讨论中提到了在 GitHub 上搜索现有 Issue 的困难，强调了对更好搜索功能的需求。
- **呼吁在 Mojo 文档中增加更多示例**：讨论中提到了 Mojo 文档中需要更多示例，特别是针对内置库。
   - 成员们被引导至现有资源，如 [devrel-extras 仓库](https://github.com/modularml/devrel-extras) 和社区示例以获取额外支持。
- **Python GIL 对性能的影响**：关于 Python 的 GIL 及其对性能的影响（特别是多线程方面）进行了广泛讨论。
   - 几位成员强调，虽然 Python 3.13 引入了禁用 GIL 的选项，但其性能仍无法与 Rust 或 Node.js 媲美。
- **Python JIT 与性能增强**：成员们讨论了 Python 3.13 版本中 JIT 的最新更新，指出虽然它具有改进潜力，但尚未完全优化。
   - 引用了一个 YouTube 视频以了解更多关于 Python JIT 编译器的细节：[Brandt Bucher – A JIT Compiler for CPython](https://www.youtube.com/watch?v=HxSHIpEQRjs)。
- **网络性能：C++ vs. Python**：参与者辩论了 C++、Python 和 Rust 等语言之间的网络性能差异，重点关注 API 和 CPU 限制的影响。
   - 有人指出 Mojo 可能会提供更好的 API 支持，但在原始网络性能方面并不会从根本上超越 C++。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/lib">Mojo🔥 modules | Modular Docs</a>：Mojo 标准库中所有模块的列表。</li><li><a href="https://www.youtube.com/watch?v=HxSHIpEQRjs&pp=ygUKcHl0aG9uIGppdA%3D%3D">Brandt Bucher – A JIT Compiler for CPython</a>：来自 2023 年 CPython 核心开发者峰会。QA 环节较难理解；可开启字幕查看尽力转录的文本。（欢迎提交 PR：https://g...</li><li><a href="https://stackoverflow.com/questions/1301346/what-is-the-meaning-of-single-and-double-underscore-before-an-object-name)">What is the meaning of single and double underscore before an object name?</a>：在 Python 中，对象名称前的单下划线和双下划线代表什么？</li><li><a href="https://docs.python.org/3.13/whatsnew/3.13.html#free-threaded-cpython">What’s New In Python 3.13</a>：编辑 Thomas Wouters。本文介绍了 Python 3.13 相比 3.12 的新特性。完整详情请参阅变更日志。摘要 – 发布亮点：Python 3.13 beta 是预发布版本...</li><li><a href="https://github.com/modularml/mojo/issues/2809">[Feature Request] Use Python-like behaviour in REPL (interactive session) to input commands and print the evaluation · Issue #2809 · modularml/mojo</a>：审查 Mojo 的优先级。我已阅读路线图和优先级，并相信此请求符合优先级。你的请求是什么？在 Python 的交互式控制台中，最后（或唯一）...</li><li><a href="https://modul.ar/community-meeting-zoom.">Join our Cloud HD Video Meeting</a>：Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，用于在移动设备、桌面和会议室系统上进行视频和音频会议、聊天和网络研讨会。Zoom ...
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1261957661914697749)** (4 条消息): 

> - `Conscious AI`
> - `Bernardo Kastrup`
> - `Joscha Bach`
> - `Split brain patients`
> - `Consciousness and computation` 


- **Bernardo Kastrup 关于 Conscious AI 的讲座**：一位成员分享了 Bernardo Kastrup 的 [YouTube 讲座](https://www.youtube.com/watch?v=mS6saSwD4DA)，探讨了为什么 Conscious AI 的概念被误解了。
   - *前四分钟总结了他演讲的核心观点。*
- **Joscha Bach 对意识的看法**：另一位成员推荐了 Joscha Bach，因为他对意识的观点与 Kastrup 相似。
   - *他被赞誉为一个非常有魅力的演说者。*
- **AI 与裂脑人患者 (Split Brain Patients)**：一位成员将 AI 系统与裂脑人患者进行了比较，指出两者都能以极高的自信对错误知识做出回应。
   - *这被引用为“意识是一种计算类型”的初步思考。*


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=mS6saSwD4DA">Computer Scientists Don&#39;t Understand This! | Conscious AI lecture, Bernardo Kastrup</a>：在这次 G10 会议的演讲中，Essentia Foundation 的主管 Bernardo Kastrup 论证了为什么 Conscious AI 的想法虽然我们不能……</li><li><a href="https://dev-discuss.pytorch.org/t/meta-pytorch-team-2024-h2-roadmaps/2226">Meta PyTorch Team 2024 H2 Roadmaps</a>：我们一直在思考如何在这里分享 Meta 正在进行的 PyTorch 工作路线图。我们按半年进行规划，因此这些是 2024 H2 OSS 计划的一些公开版本……
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1261796060724985877)** (137 条消息🔥🔥): 

> - `Mojo website down`
> - `Module ownership and deletion`
> - `Using keep and release in Mojo`
> - `Socket library implementation in Mojo`
> - `DateTime library in Mojo` 


- **关于 Mojo 网站无法访问的困惑**：成员们报告 [Mojo 网站](https://mojolang.org/) 宕机了，导致了一些困惑，因为许多用户将其误认为是官方网站。
   - 经过澄清后，分享了[官方网站](https://www.modular.com/)，并指出之前的 URL 现在已正确重定向。
- **Mojo 中 transfer 运算符的细微差别**：成员们讨论了使用 `_ = model^` 来防止变量被过早销毁，并指出了 transfer 运算符及其对 Mojo 中值生命周期 (value lifetimes) 的重要性。
   - 对话强调了隐式移动 (implicit moves) 和 `__del__()` 函数带来的挑战，并引用了关于值生命周期和销毁的[相关文档](https://docs.modular.com/mojo/manual/values/ownership#transfer-arguments-owned-and-)。
- **建议使用 'keep' 代替隐式移动**：有人建议使用 `keep` 来保持变量存活，以避免与 Mojo 中的隐式转移混淆，根据 [compiler hinting 文档](https://docs.modular.com/mojo/stdlib/benchmark/compiler/keep)，这可能会使意图更加清晰。
   - 其他人则争论 `keep` 将生命周期与优化混淆了，并提议使用更正式的语法来处理这种情况。
- **对 Mojo 中 socket 库的期待**：成员们表达了对 Mojo 内置 socket 库的渴望，尽管目前引用了一个临时解决方案：[lightbug HTTP 库](https://github.com/saviorand/lightbug_http/tree/main/external)。
   - 团队表示对使用 Mojo 进行服务器开发很感兴趣，暗示标准 socket 库可能正在开发中。
- **对 Mojo 中 DateTime 库的赞赏**：一位成员公开感谢 Martin Vuyk 在 DateTime 和其他库上的大量工作，对他贡献的努力和资源表示赞赏。
   - 感谢还延伸到了 [forge-tools 仓库](https://github.com/martinvuyk/forge-tools) 中的现有工具，这些工具增强了 Mojo 标准库的功能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

</div>

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/">Mojo 手册 | Modular 文档</a>：Mojo 编程语言的全面指南。</li><li><a href="https://docs.modular.com/mojo/manual/values/ownership#transfer-arguments-owned-and-">所有权与借用 | Modular 文档</a>：Mojo 如何通过函数参数共享引用。</li><li><a href="https://docs.modular.com/mojo/manual/values/ownership#transfer-arg">所有权与借用 | Modular 文档</a>：Mojo 如何通过函数参数共享引用。</li><li><a href="https://docs.modular.com/mojo/stdlib/benchmark/compiler/keep">keep | Modular 文档</a>：keep(val: Bool)</li><li><a href="https://docs.modular.com/mojo/manual/lifecycle/life#move-constructor">值的生命周期 | Modular 文档</a>：关于 Mojo 何时以及如何创建值的说明。</li><li><a href="https://docs.modular.com/mojo/manual/lifecycle/death">值的销毁 | Modular 文档</a>：关于 Mojo 何时以及如何销毁值的说明。</li><li><a href="https://docs.python.org/3/library/socket.html">socket — 低级网络接口</a>：源代码：Lib/socket.py。此模块提供对 BSD socket 接口的访问。它适用于所有现代 Unix 系统、Windows、MacOS 以及可能的其他平台。可用性：不...</li><li><a href="https://github.com/saviorand/lightbug_http/tree/main/external">lightbug_http/external at main · saviorand/lightbug_http</a>：适用于 Mojo 的简单且快速的 HTTP 框架！🔥。通过在 GitHub 上创建账户为 saviorand/lightbug_http 的开发做出贡献。</li><li><a href="https://github.com/martinvuyk/forge-tools">GitHub - martinvuyk/forge-tools: 扩展 Mojo 标准库功能的工具</a>：扩展 Mojo 标准库功能的工具 - martinvuyk/forge-tools</li><li><a href="https://mojolang.org/">Modular：拥有你的端点。控制你的 AI。</a>：Modular Accelerated Xecution (MAX) 平台是全球唯一能为你的 AI 工作负载释放性能、可编程性和可移植性的平台。</li><li><a href="https://www.modular.com">Modular：拥有你的端点。控制你的 AI。</a>：Modular Accelerated Xecution (MAX) 平台是全球唯一能为你的 AI 工作负载释放性能、可编程性和可移植性的平台。</li><li><a href="https://www.modular.com/mojo">Mojo 🔥：面向所有 AI 的编程语言</a>：Mojo 将 Python 的易用性与 C 的性能相结合，释放了 AI 硬件无与伦比的可编程性和 AI 模型的可扩展性。</li><li><a href="https://www.modular.com/">Modular：拥有你的端点。控制你的 AI。</a>：Modular Accelerated Xecution (MAX) 平台是全球唯一能为你的 AI 工作负载释放性能、可编程性和可移植性的平台。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1261593726111711292)** (6 条消息): 

> - `MAX 许可证拼写错误`
> - `AMD 统一 AI 软件栈`
> - `Modular 的独家合作伙伴关系` 


- **MAX 许可证拼写错误已处理**：用户注意到新的 [Max 许可证](https://www.modular.com/legal/max) 中存在多处拼写错误，包括 **otherModular** 和 **theSDK** 等术语中缺失空格。
- **用户询问 AMD 统一 AI 软件栈**：一位成员询问了关于与 AMD 讨论将 **Max** 集成到 AMD 在技术日宣布的新 **统一 AI 软件栈** (Unified AI software stack) 的相关事宜。


  

---

### **Modular (Mojo 🔥) ▷ #[max-gpu](https://discord.com/channels/1087530497313357884/1212827673257316453/1261397819730821150)** (11 条消息🔥): 

> - `使用 Max 编写自定义 Kernel`
> - `比 Graph 更低层级的 API`
> - `基准测试 Tensor Cores`
> - `为 XLA 设备编写 PyTorch` 


- **Mojo 中的自定义 GPU Kernel**：可以使用 Mojo 编写自定义 GPU Kernel，它是 MAX 的一部分，类似于加速器的 CUDA 接口。
   - 这些 Kernel 使用 Mojo 编译器编译，并通过 MAX 库入队到加速器中。
- **MAX 中的低层级 API**：早期版本允许在 MAX Graph 中嵌入自定义算子，并且还将提供比 Graph 更低层级的 API 供开发者进行底层开发。
   - MAX 和 Mojo 紧密结合，提供了与加速器交互的接口，非常类似于 CUDA。
- **基准测试中的 Tensor Cores**：有人提出了关于基准测试未利用 Tensor Cores 的疑问，质疑 GEMM 数值及其与 FA 的关系。
   - 一位成员强调了由于 TPU 编译器和运行时的不透明性所带来的复杂性。
- **PyTorch xla 开发挑战**：Google 和 Meta 花了[五年时间](https://github.com/pytorch/xla)才开发出 PyTorch xla，从而在 Google TPU 等 XLA 设备上启用 PyTorch。
   - 这一开发的复杂性和持续时间被提及，反映了其中涉及的挑战。



**提到的链接**: <a href="https://github.com/pytorch/xla">GitHub - pytorch/xla: Enabling PyTorch on XLA Devices (e.g. Google TPU)</a>: 在 XLA 设备（如 Google TPU）上启用 PyTorch。通过在 GitHub 上创建账号为 pytorch/xla 的开发做出贡献。

  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1261475885295599666)** (13 条消息🔥): 

> - `Mojo nightly 版本发布`
> - `机器人交互`
> - `stdlib 扩展提案`
> - `贡献者反馈` 


- **Mojo nightly 版本发布更新**：发布了新的 nightly Mojo 编译器，版本号为 `2024.7.1305` 和 `2024.7.1505`。更新包括对 `SIMD.load/store` 的 `UnsafePointer` 重载的更改，以及根据[当前变更日志](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)所述移除了 `LegacyPointer`。
- **机器人威胁频繁贡献者**：一位用户提到因为标记了五位贡献者而受到 Modular 机器人的“威胁”。另一位用户分享了类似的经历，即机器人误解了某些符号的使用。
   - 机器人似乎对特定模式或符号有触发机制，从而导致不必要的警告。
- **减轻 stdlib 维护者工作量的提案**：提出了一项通过 `stdlib-extensions` 减轻 stdlib 维护者工作量的提案，并寻求频繁贡献者的反馈。该[讨论](https://github.com/modularml/mojo/discussions/3233)旨在优化维护工作。



**提到的链接**: <a href="https://github.com/modularml/mojo/discussions/3233">[提案] 通过 `stdlib-extensions` 减轻 stdlib 维护者的工作量 · modularml/mojo · Discussion #3233</a>: 此讨论旨在探讨以下提案：pull request markdown 文档。我们特别感兴趣频繁贡献者以及 st... 的意见。

  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1261400952691228782)** (207 条消息🔥🔥): 

> - `GPTs Agents`
> - `API Credits`
> - `Pro Plan Issues`
> - `Image Response Problems`
> - `Perplexity vs ChatGPT` 


- **GPTs Agents 在初始训练后无法学习**：一位成员分享了关于 GPTs Agents 在初始训练后无法从提供的额外信息中学习的担忧。
   - 另一位成员澄清说，[上传的文件被保存为 "knowledge" 文件](https://link.to/openai-docs)供 Agent 在需要时引用，但**它们不会持续修改 Agent 的基础知识**。
- **API Credits 领取问题**：用户报告称，在升级到 Pro 计划后，没有收到承诺的用于试用 API 的 $5 免费额度，并且在使用印度信用卡充值额度时遇到问题。
   - 已联系支持部门，但未立即提供解决方案；一些用户建议核实 API 激活操作是否正确。
- **Pro 计划搜索限制被悄悄降低**：多名用户注意到，他们的 Pro 搜索限制在没有任何事先通知或网站更新的情况下，从每天 600 次减少到了 540 次。
   - 这一未经宣布的变化引发了用户对未来进一步削减额度以及 Perplexity 政策透明度的担忧。
- **图片回复和后续提问的困难**：*iamhasim* 讨论了 Perplexity 的回复经常引用旧图片而非当前对话内容的问题。
   - 其他人也表达了类似的困扰，并希望在处理图片和后续问题方面能有所改进。
- **Perplexity 与 ChatGPT 在代码和数据处理方面的对比**：用户辩论了 Perplexity 与 ChatGPT 相比的能力，强调了在文件处理、图片生成和后续提问准确性等方面的差距。
   - 尽管存在局限性，一些用户仍因其搜索和 Collections 功能而偏好 Perplexity，但也指出其在文档对比和代码处理等功能上落后于对手。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/LnO-Oca7ysY?si=iElRSe23YWrHlrl-">How to SUPERCHARGE your web research with a Large Action Model (Nelima)</a>：认识 Nelima 🚀，这是全球首个社区驱动的 Large Action Model (LAM)，能将你的自然语言提示转化为实际行动。Nelima...</li><li><a href="https://search.google.com/test/rich-results">Rich Results Test - Google Search Console</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1261451828013568123)** (12 条消息🔥): 

> - `Health and Strength`
> - `Marketing Expertise`
> - `Cantillon Effect`
> - `Uniqueness of Teeth`
> - `Trump Assassination Attempt` 


- **通过技巧获得健康与力量**：用户分享了一个[关于如何获得健康与力量的搜索链接](https://www.perplexity.ai/search/how-to-achieve-health-strength-094kl4NzQea2mENjIOdG8Q)。
- **关于营销专业知识的见解**：多位用户讨论了一个[关于成为营销专家的搜索链接](https://www.perplexity.ai/search/tu-es-un-expert-en-marketing-s-s3M_sVlXSwS1h.1Np0NLOQ)。
- **理解坎蒂隆效应 (Cantillon Effect)**：一位用户提供了一个[学习坎蒂隆效应的搜索链接](https://www.perplexity.ai/search/the-cantillon-effect-KnCFxYCeQuG51gUkuJdtkA)。
- **探索牙齿的唯一性**：一个[质疑我们的牙齿是否唯一的搜索链接](https://www.perplexity.ai/search/are-our-teeth-unique-IvBjExR8TL64cO9QknSbsw#0)引发了讨论。
- **关于特朗普暗杀企图的辩论**：分享了一个讨论[特朗普暗杀企图链接](https://www.perplexity.ai/page/trump-assassination-attempt-Yc6pNnfDQ6WUP6qD44AZIg)的有争议话题。



**提到的链接**：<a href="https://www.youtube.com/embed/KXKYohXysZM">YouTube</a>：未找到描述

  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1261400801583173704)** (8 条消息🔥): 

> - `Cloudflare 问题`
> - `Pro 订阅者额度问题`
> - `API 免费额度问题`
> - `Perplexity AI API 模型` 


- **API 被 Cloudflare 拦截**：一位用户提到 API 目前处于 **Cloudflare** 防护之下，导致了访问问题。
- **Pro 订阅者的 5 美元免费额度**：一位升级到 Pro 的成员询问用于试用 API 的 **5 美元免费额度**何时可用。
- **无法使用额度生成 API**：一位 **Pro 订阅者**无法购买额度或使用 5 美元额度来生成 API，正在频道中寻求帮助。
   - 另一位用户分享了同样的问题，并提供了一个 [Discord 频道链接](https://discord.com/channels/1047197230748151888/1161802929053909012/1207351871186931783)以获取进一步帮助。
- **使用 API 模拟 Perplexity AI 免费层级**：一位用户尝试使用 API 复制 **Perplexity AI 免费层级**的效果，但在随回答获取 URL 来源方面遇到困难。
   - 他们询问其他人是否知道 Perplexity AI 使用的是哪种模型，或者如何实现类似的结果。


  

---



### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1261470051614068858)** (5 条消息): 

> - `Microsoft Research 的 AgentInstruct`
> - `WizardLM 的 Arena Learning` 


- **Microsoft Research 推出 AgentInstruct**：[AgentInstruct](https://arxiv.org/html/2407.03502v1) 是一个用于创建高质量合成数据的框架，可将 **Mistral-7b** 等模型后训练（post-train）为 **Orca-3**，在各种基准测试中显示出显著提升。
   - 论文报告称，后训练模型在 AGIEval 上提升了 **40%**，在 GSM8K 上提升了 **54%**，在 AlpacaEval 上提升了 **45%**，表现优于 LLAMA-8B-instruct 和 GPT-3.5-turbo 等竞争对手。
- **WizardLM 的 Arena Learning 模拟 Chatbot Arena**：[Arena Learning](https://www.microsoft.com/en-us/research/uploads/prodnew/2024/07/WizardLM_ArenaLearning.pdf) 旨在通过 AI 驱动的模拟聊天机器人对战，为持续后训练创建数据飞轮。
   - 该迭代过程持续改进了 WizardLM 模型，在 WizardArena-Mix Elo 和 MT-Bench 等指标上有明显的性能提升，同时与 LMSYS Arena 的人类判断达到了 **98.79% 的一致性**。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/html/2407.03502v1">AgentInstruct: Toward Generative Teaching with Agentic Flows</a>：未找到描述</li><li><a href="https://x.com/victorsungo/status/1811427047341776947">Qingfeng Sun (@victorsungo) 的推文</a>：🔥 很高兴分享 WizardLM 的新论文！📙Arena Learning: 通过模拟聊天机器人竞技场构建 LLMs 后训练的数据飞轮 🚀作为 WizardLM-2 最重要的技术之一，让我来介绍一下...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1261620606084841513)** (11 条消息🔥): 

> - `LivePortrait GitHub 项目`
> - `煮蛋和剥蛋技巧` 


- **LivePortrait GitHub 项目见解**：一位成员提到了 [LivePortrait GitHub 项目](https://github.com/KwaiVGI/LivePortrait)，并询问如何获取具有合适表情的视频用于文本转视频（text-to-video）转换。
   - 他们建议了一种方法，包括拍摄面部说话的视频，使用 Whisper 进行转录，并使用向量数据库查找具有所需表情的片段。
- **完美剥蛋技巧**：成员们分享了剥蛋技巧，建议在热水中煮 10 分钟以便轻松剥壳。
   - 一位成员建议了另一种方法，将鸡蛋浸泡在醋中以溶解蛋壳，并提供了一个[详细解释的链接](https://www.scienceworld.ca/resource/naked-eggs-acid-base-reaction)。



**提及的链接**：<a href="https://www.scienceworld.ca/resource/naked-eggs-acid-base-reaction">Naked Eggs: Acid-Base Reaction - Science World</a>：在这项活动中，学生们描述酸对蛋壳的影响。蛋壳在醋中的反应是一种酸碱反应。当你把鸡蛋浸入醋中时，蛋壳会溶解，留下...

  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1261820394248343592)** (6 条消息): 

> - `TextGrad`
> - `Q-star details`
> - `Claude artifacts`
> - `System prompts optimization tips` 


- ****TextGrad 使用 LLMs 进行 textual gradients****：一个名为 [TextGrad](https://github.com/zou-group/textgrad) 的 **GitHub** 项目利用 **LLMs** 来 **backpropagate textual gradients**，彻底改变了基于文本的计算。
- ****Q-star 细节通过路透社泄露****：一段名为“[Q-star details leaked](https://youtu.be/T9gAg_IXB5w)”的 YouTube 视频讨论了来自 **OpenAI** 的泄露内部文件，代号为 **STRAWBERRY**，揭示了 **AGI** 的新进展。
   - 该视频由 **Wes Roth** 报道，强调了关于 **LLMs** 的关键见解，并预测了即将到来的 **AI** 发布。
- ****Claude artifacts 现在可以共享****：**Claude artifacts** 现在可以[共享](https://claude.site/artifacts/9d409d6b-70aa-403a-96e3-df292a2b47ee)了，这使得分发和协作 **AI** 相关输出变得更加容易。
- ****System prompts 的优化建议****：用户 _paradroid 分享了一个基于 **STaR** 的 **System Prompt**，用于一个专注于 **iterative improvement** 和 **reasoning** 的高级 **AI** 助手，展示了持续 **AI** 开发的结构化方法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/T9gAg_IXB5w">BREAKING: Q-star details LEAKED! Reuters reveals internal OpenAI documents (codename: STRAWBERRY)</a>: 最新的 AI 新闻。了解 LLMs、Gen AI 并为 AGI 的推出做好准备。Wes Roth 报道了 OpenAI、Google、Anth... 世界的最新动态。</li><li><a href="https://github.com/zou-group/textgrad">GitHub - zou-group/textgrad: TextGrad: Automatic &#39;&#39;Differentiation&#39;&#39; via Text -- using large language models to backpropagate textual gradients.</a>: TextGrad：通过文本实现自动“Differentiation” —— 使用 LLMs 来 backpropagate textual gradients。- zou-group/textgrad
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1261398689029558323)** (169 条消息🔥🔥): 

> - `LLM Reasoning Improvement` (LLM 推理改进)
> - `OpenAI Platform Updates` (OpenAI 平台更新)
> - `AgentInstruct (Orca 3) Paper Discussion` (AgentInstruct (Orca 3) 论文讨论)
> - `New Vision Language Model by Google` (Google 的新视觉语言模型)
> - `Teknium Hiring Announcement` (Teknium 招聘公告)


- **提升 LLM 的推理能力**：成员们讨论了仅通过 Prompting 来增强 LLM 的推理能力，建议的方法包括 few-shot learning、in-context learning 以及 chain-of-thought (CoT) 提示技术。
   - 一些用户对 CoT 的有效性表示怀疑，认为它在处理与训练数据显著不同的问题时表现乏力。
- **OpenAI 平台更新与谜团**：成员们对 OpenAI 的新网站 "OpenAI Supply Co." 进行了推测，倾向于认为这可能是一个周边商品商店。
   - 还有关于潜在产品的幽默推测，比如 Sam Altman 毛绒公仔。
- **关于 AgentInstruct (Orca 3) 论文的观点**：用户们询问并分享了对新论文 AgentInstruct (Orca 3) 的好奇心，并提供了进一步讨论的链接。
   - 对话暗示了褒贬不一的印象，并强调了正确评估新研究的重要性。
- **Google 的新视觉语言模型**：讨论了 Google 的新视觉语言模型 PaliGemma，提到它需要经过微调才能发挥效用。
   - 用户对其初始性能进行了辩论，并提到了特定的许可限制。
- **Teknium 发布招聘启事**：Teknium 发布了一则公告，寻求负责合成文本数据创建和 Agent 构建职位的申请人，目前已有超过 40 名申请者。
   - 招聘要求强调了与 Nous Research 目标和价值观的一致性，以及各种技术技能，选拔过程将很快开始。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/Teknium1/status/1812339816429998418">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：我正在招聘 1-2 名全职人员，负责用于训练 LLM 的合成文本数据创建，并利用 Agent 能力来提高数据质量。目前已有 40 多名申请者，我只能挑选...</li><li><a href="https://gist.github.com/fullstackwebdev/b8257a67933d891a9f3bc19822b4305a">gist:b8257a67933d891a9f3bc19822b4305a</a>：GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://huggingface.co/docs/transformers/main/en/model_doc/paligemma">PaliGemma</a>：未找到描述</li><li><a href="https://x.com/KapadiaSoami/status/1811657156082712605">来自 Soami Kapadia (@KapadiaSoami) 的推文</a>：Groq 上的 Mixture of Agents。介绍一个完全可配置的、由 @GroqInc 提供动力并使用 @LangChainAI 的 Mixture-of-Agents 框架。你可以通过 @streamlit UI 配置自己的 MoA 版本...</li><li><a href="https://x.com/nutlope/status/1811824371440427093">来自 Hassan (@nutlope) 的推文</a>：刚刚在多步数学题上微调了 Llama-3-8B。我在 1,000 个新数学题上进行了测试，它达到了 GPT-4o 90% 的性能（而且更便宜、更快）。写了一篇关于如何...的博客文章。</li><li><a href="https://x.com/ai_for_success/status/1812004912173129854">来自 AshutoshShrivastava (@ai_for_success) 的推文</a>：OpenAI 新网站 "OpenAI Supply Co."。他们会供应什么？h/t : ananayarora</li><li><a href="https://github.com/Dao-AILab/flash-attention">GitHub - Dao-AILab/flash-attention: 快速且内存高效的精确注意力机制</a>：快速且内存高效的精确注意力机制。通过在 GitHub 上创建账户为 Dao-AILab/flash-attention 做出贡献。</li><li><a href="https://github.com/NousResearch/Hermes-Function-Calling">GitHub - NousResearch/Hermes-Function-Calling</a>：通过在 GitHub 上创建账户为 NousResearch/Hermes-Function-Calling 做出贡献。</li><li><a href="https://github.com/SynaLinks/HybridAGI">GitHub - SynaLinks/HybridAGI: 可编程的神经符号 AGI，允许你使用基于图的 Prompt 编程来编写其行为</a>：可编程的神经符号 AGI，允许你使用基于图的 Prompt 编程来编写其行为：适用于希望 AI 表现符合预期的人——SynaLinks/HybridAGI</li><li><a href="https://tenor.com/view/cheering-cute-cat-smile-jump-gif-17108371">欢呼的可爱 GIF - 欢呼的可爱猫咪 - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1261778078502227999)** (22 messages🔥): 

> - `Integrating LLM in apps` (在应用中集成 LLM)
> - `Extending context length for models` (扩展模型的上下文长度)
> - `Model performance` (模型性能)
> - `UX for integrated chat` (集成聊天的 UX)
> - `AI agents` (AI Agent)


- **考虑将 LLM 用于应用教程**：**Natefyi** 建议在应用中集成 LLM 用于教程，而不是使用视频和博客文章等传统媒体。
   - *Teknium* 提到，**使用检索增强生成 (RAG)** 可以作为 FAQ 和帮助信息的解决方案。
- **扩展模型的上下文长度**：一位用户询问了将 **Mixtral** 和 **Llama** 等各种模型的上下文长度扩展到 1M token 的技术。
   - *Deoxykev* 指出，实现这样的长度需要大量的 VRAM，**kotykd** 补充说，目前的长上下文模型在**实际场景中不可用**。
- **寻求集成帮助聊天的 UX 灵感**：Natefyi 就如何在应用中集成 LLM 引导的帮助聊天寻求 UX 设计建议，思考了诸如弹窗或按钮之类的交互方式。
   - *Thilotee* 推荐了 **Audapolis** 作为引导用户使用功能的 UI 示例，但对如何将其与 LLM 结合表示不确定。
- **对开发 AI Agent 的兴趣**：*Pablo.ce* 表示有兴趣在 **Hugging Face (HF) spaces** 上合作开发 AI Agent，并艾特了另一位创建了 **llama-cpp-agents** 框架的用户。
   - 他提议使用其他用户指定的模型创建 HF spaces，征求进一步的合作。



**提到的链接**：<a href="https://github.com/bugbakery/audapolis/commits/main/)">History for ) - bugbakery/audapolis</a>：一个带有自动转录功能的口语音频编辑器 - History for ) - bugbakery/audapolis

  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1261410712559943800)** (8 messages🔥): 

> - `Marker version speedup` (Marker 版本加速)
> - `Integration with synthetic RAG` (与合成 RAG 集成)
> - `XML in agent definition` (Agent 定义中的 XML)
> - `Mixture of Agents models` (Mixture of Agents 模型)
> - `Stasima diverse models` (Stasima 多样化模型)


- **Marker 显著提速**：[Marker](https://x.com/VikParuchuri/status/1811851126125527096) 的新版本速度显著提升：由于其高效的架构，在 **MPS 上提升了 7 倍**，在 **CPU 上提升了 3 倍**，在 **GPU 上提升了 10%**。
   - 该工具旨在将 PDF 转换为 Markdown，此次提速旨在促进创建更高质量的数据集。
- **XML 让 Agent 定义更简单**：关于 XML 如何简化 Agent 定义的[有趣讨论](https://x.com/TheSeaMouse/status/1812005737016492317)。
   - *当你拥抱 XML 时，定义 Agent 变得如此简单，这很有趣。*
- **Mixture of Agents 模型实现**：一位成员展示了一个仅用 50 行代码实现的 [Mixture-of-Agents 实现](https://x.com/rohanpaul_ai/status/1811921050281685293)，通过 @togethercompute 集成了多个模型。
   - 另一位成员在他们的项目 [stasima](https://github.com/EveryOneIsGross/stasima) 中讨论了他们对这一概念的理解，使用不同的 system prompts 来创建一系列 Agent。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/rohanpaul_ai/status/1811921050281685293">Rohan Paul (@rohanpaul_ai) 的推文</a>：使用 @togethercompute 通过 50 行代码实现 Mixture-of-Agents</li><li><a href="https://python.useinstructor.com/prompting/">Prompting - Instructor</a>：未找到描述</li><li><a href="https://x.com/TheSeaMouse/status/1812005737016492317">Hassan Hayat 🔥 (@TheSeaMouse) 的推文</a>：当你拥抱 XML 时，定义 Agent 变得如此简单，这很有趣</li><li><a href="https://x.com/VikParuchuri/status/1811851126125527096">Vik Paruchuri (@VikParuchuri) 的推文</a>：Marker 现在更快了！MPS 提升 7 倍，CPU 提升 3 倍，GPU 提升 10%。得益于更高效的双模型架构。Marker 可以非常有效地将 PDF 转换为 Markdown。我希望这次提速能让人们……</li><li><a href="https://github.com/EveryOneIsGross/stasima">GitHub - EveryOneIsGross/stasima: stasima 是响应同一查询的多样化模型和 Agent 光谱。</a>：stasima 是响应同一查询的多样化模型和 Agent 光谱。 - EveryOneIsGross/stasima
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1261415738510672003)** (55 条消息🔥🔥): 

> - `WebGPU 开发工作流`
> - `Flash Attention 内存占用`
> - `ResNet 实现` 


- **WebGPU 开发工作流：迭代快但需要更好的工具链**：一位用户分享了他们为 **WebGPU** 开发 kernel 的工作流，指出其**迭代周期快**，但工具链和性能分析（profiling）体验欠佳。
   - 他们提到使用 **dawn 作为共享库**来缩短编译时间，并提供了一个 [实时编写 WGSL shaders 的演示](https://drive.google.com/file/d/15oXwYqVeoOMNYDEjG3xJ2PEeNYFbbGjz/view?usp=drive_link)。
- **WebGPU vs 传统 GPU API：挑战与前景**：另一场讨论强调了将 **WebGPU** 性能与 CUDA 等传统 GPU 进行对比，以及通过 llm.c Transformer kernel 移植来获取更深入见解的潜力。
   - 讨论中还关注了 **WebGPU 的 cooperative matrix 扩展进展**（[GitHub 链接](https://github.com/gpuweb/gpuweb/issues/4195)），并期待更多 ML 工作负载向客户端计算转移。
- **Flash Attention：SRAM 利用率限制**：一场深入的技术讨论围绕 **Flash Attention 1** 的内存占用展开，重点在于在存在其他组件的情况下，**QKVO** 数组是否能很好地放入 SRAM。
   - 回复中强调 **S 和 P 是瞬时的（ephemeral）**，并讨论了如何调整 **Br 和 Bc 常数**以匹配可用 SRAM，同时参考了其源代码中的实现（[GitHub 链接](https://github.com/Dao-AILab/flash-attention/blob/7ef24848cf2f855077cef88fe122775b727dcd74/csrc/flash_attn/src/flash_fwd_launch_template.h#L186)）。
- **计算机视觉 ResNet 入门**：一位成员请求关于为计算机视觉论文实现 **ResNet** 的指导。
   - 他们被引导至 [torchvision 中的 ResNets](https://pytorch.org/vision/main/models/resnet.html)，该页面为他们的项目提供了开箱即用的实现。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://drive.google.com/file/d/15oXwYqVeoOMNYDEjG3xJ2PEeNYFbbGjz/view?usp=drive_link">Screen Recording 2024-07-13 at 12.30.44 AM.mov</a>: 未找到描述</li><li><a href="https://pytorch.org/vision/main/models/resnet.html">ResNet &mdash; Torchvision main documentation</a>: 未找到描述</li><li><a href="https://github.com/Dao-AILab/flash-attention/blob/7ef24848cf2f855077cef88fe122775b727dcd74/csrc/flash_attn/src/flash_fwd_launch_template.h#L186">flash-attention/csrc/flash_attn/src/flash_fwd_launch_template.h at 7ef24848cf2f855077cef88fe122775b727dcd74 · Dao-AILab/flash-attention</a>: 快速且内存高效的精确 Attention。通过在 GitHub 上创建账号为 Dao-AILab/flash-attention 做出贡献。</li><li><a href="https://github.com/gpuweb/gpuweb/issues/4195">Cooperative matrix · Issue #4195 · gpuweb/gpuweb</a>: 所有主流平台 API 现在都发布了类似的 cooperative matrix 扩展：Metal 在 MSL 3.1 中引入了 simdgroup_matrix，HLSL 在 SM6.8 中提供支持（目前为实验性版本），SPIR-V...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1261588030171516978)** (5 条消息): 

> - `学习 Triton`
> - `GitHub 上的 Triton Puzzles`
> - `FP8 训练中的 Triton`
> - `Triton 用于逐元素操作的内联汇编` 


- **初学者深入学习 Triton**：一位用户询问除了官方[文档](https://triton-lang.org/main/index.html)之外，还有哪些学习 **Triton** 的参考资料。
- **Triton 中最先进的 FP8 训练**：一位用户询问了目前在 **Triton** 中使用 FP8 训练的方法，以及是否有稳定的 kernel 可供适配，或者大家是否普遍使用 **transformerengine**。
- **利用 Triton 的内联汇编进行逐元素操作**：一位用户发现 **Triton 的内联汇编（inline asm）** 可以一次处理多个元素，这可能对融合位打包/拆包（fused bit-packing/unpacking）和 matmul 操作非常有用。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://triton-lang.org/main/index.html?">Welcome to Triton’s documentation! &mdash; Triton  documentation</a>: 未找到描述</li><li><a href="https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html">triton.language.inline_asm_elementwise &mdash; Triton  documentation</a>: 未找到描述</li><li><a href="https://github.com/srush/Triton-Puzzles">GitHub - srush/Triton-Puzzles: Puzzles for learning Triton</a>: 学习 Triton 的谜题。通过在 GitHub 上创建账号为 srush/Triton-Puzzles 做出贡献。
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1262147685256007764)** (3 messages): 

> - `Bootstrap estimate of accuracy stdev`（准确率标准差的 Bootstrap 估计）
> - `Optimized dataloader issue`（优化后的 dataloader 问题）
> - `Torch nightly broken`（Torch nightly 版本损坏）


- **用于模型评估的准确率标准差的 Bootstrap 估计**：一名成员建议使用 **bootstrap estimate** 来计算模型评估中的**准确率标准差**。
- **切换回 torch dataloader 解决了问题**：另一名成员报告称，从**优化后的 dataloader** 切换回 **torch 版本**后，解决了他们遇到的一个未指明的问题。
- **Torch nightly 构建版本存在损坏的功能**：一位用户提到 **Torch nightly build** 已损坏，具体表现为由于 **'torch.library'** 缺少 `custom_op` 属性而导致的 `AttributeError`。


  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1261411804693921852)** (2 messages): 

> - `LoQT method for efficient training`（用于高效训练的 LoQT 方法）
> - `Brian Kernighan on The Practice of Programming`（Brian Kernighan 谈《程序设计实践》）


- **LoQT 实现了在消费级 GPU 上进行高效的模型训练**：题为“[LoQT](https://arxiv.org/abs/2405.16528)”的论文提出了一种使用基于梯度的张量分解来高效训练量化模型的方法，使得高达 **7B 参数**的模型能够在消费级 24GB GPU 上进行训练。
   - 该方法以不同方式处理量化权重的梯度更新，并实现了相当的节省，适用于预训练和微调。
- **Brian Kernighan 讨论《程序设计实践》**：在一段 [YouTube 视频](https://www.youtube.com/watch?v=_QQ7k5sn2-o)中，Brian Kernighan 博士在 Book Overflow 的特别节目中讨论了他编写《程序设计实践》（The Practice of Programming）的经验。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.16528">LoQT: Low Rank Adapters for Quantized Training</a>：大型神经网络的训练需要大量的计算资源。尽管在使用低秩适配器和量化方面取得了进展，但在消费级硬件上预训练 LLM 等模型仍然...</li><li><a href="https://www.youtube.com/watch?v=_QQ7k5sn2-o">Brian Kernighan Reflects on &quot;The Practice of Programming&quot;</a>：在 Book Overflow 的这一特别节目中，《程序设计实践》的作者 Brian Kernighan 博士加入我们，讨论他编写该书的经验...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1261415740133609613)** (23 messages🔥): 

> - `Accessing GPUs for Testing`（访问用于测试的 GPU）
> - `Using Google Colab and nsight compute`（使用 Google Colab 和 nsight compute）
> - `CoreWeave vs Lambda Labs`（CoreWeave 对比 Lambda Labs）
> - `Cloud GPU Services`（云端 GPU 服务）
> - `Learning Triton`（学习 Triton）


- **个人测试访问 GPU 的最佳途径**：一位用户询问了获取用于测试的 GPU 访问权限的最佳方式，特别是需要使用 ncu，多个回复推荐使用 **Google Colab**，因其便捷且提供免费访问 (https://colab.research.google.com)。
   - 讨论中还提到了 **CoreWeave** 和 **LambdaLabs** 作为其他选项，并指出 CoreWeave 价格昂贵，而 LambdaLabs 的配额难以获取。
- **Colab 支持 nsight compute**：一名成员确认 **nsight compute** 可以在 Google Colab 上运行，尽管弹出窗口可能会有问题。
   - 对话还强调，**Google Cloud GPU** 允许使用 notebook 以外的工具，尽管与 Colab 相比价格更高。
- **云端 GPU 服务对比**：成员们对比了不同的云服务，如 **Google Cloud GPU** 和 **SageMaker**，以及像 **vast.ai** 这样的按需服务，指出后者通常更便宜。
   - 为了工作方便，有人建议 **Google Colab** 比 **Google Cloud Platform (GCP)** 更省事。
- **Triton 学习资源**：一位用户询问了除了官方 [Triton 网站](https://triton-lang.org/main/index.html)之外，学习 **Triton** 的其他参考资料。
   - 回复中未提及具体的额外资源。
- **在云端进行开源开发的挑战**：由于笔记本电脑较旧且配备的是 **NVIDIA Quadro M4000** GPU，一名成员寻求关于使用云端工具进行开源开发的建议。
   - 他们提到了在 **Google Colab** 等云环境中为 **torchao** 项目开发进行代码更改迭代和测试时的挑战。



**提及的链接**：<a href="https://triton-lang.org/main/index.html">Welcome to Triton’s documentation! &mdash; Triton  documentation</a>：未找到描述内容

  

---

### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1261879533712310392)** (34 messages🔥): 

> - `CUDA Core Processing` (CUDA Core 处理)
> - `Register Limitations` (寄存器限制)
> - `Occupancy Calculation` (占用率计算)
> - `Block Size Optimization` (Block 大小优化)
> - `Kernel Parameterization` (Kernel 参数化) 


- **CUDA Core 处理澄清**：一场讨论揭示了一个 CUDA core 一次处理一个线程，这意味着一个拥有 64 个 CUDA Cores 的 A100 SM 可以同时处理 **64 个线程**，而分配给它的线程总数可达 **2048 个**。
   - 另一位成员解释了这与 CPU 的相似之处，即线程在等待时会换出并将其状态存储在内存中，但在 GPU 上，寄存器总池的大小限制了这一点。
- **寄存器限制对线程的影响**：解释了每个 SM 拥有 **256 KiB 寄存器**，当分配给 2048 个线程时，每个线程只能获得 **32 个寄存器**。
   - 在 Kernel 中使用更多寄存器会限制可执行的线程总数，例如，每个线程使用 64 个寄存器时，总线程数限制为 1024 个。
- **优化 GPU 占用率 (Occupancy)**：GPU 上的线程占用率受分配的 Shared Memory 和线程数量的影响，进而影响延迟隐藏 (latency hiding)。
   - 需要达到一种平衡，因为过多的线程可能因内存不足导致停顿，反之，线程太少则无法充分隐藏延迟。
- **Block 大小与性能**：讨论了如何通过 profiling 和经验推断来选择最佳的 Block 大小以提升性能。
   - 以 Block reduction 为例，通过对 1024、512 和 256 进行 profiling，发现 **128 的 Block 大小** 性能最好，这与最初的预期相反。
- **用于优化的 Kernel 参数化**：对不同数值的实现进行参数化，允许运行 benchmark 来寻找最佳配置，这对于优化 GPU 性能至关重要。
   - 正如在 FAv2 源码中所见，通过 `STATIC_SWITCH` 和 `BOOL_SWITCH` 对不同大小的 Kernel 进行模板化，可以实现对不同矩阵大小的最佳适配。


  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1261504796356116632)** (2 messages): 

> - `FSDP support for low-bit optimization` (低比特优化的 FSDP 支持)
> - `Developer guide for integration` (集成开发指南) 


- **实现低比特优化的 FSDP 支持**：一名成员正在致力于实现低比特优化的 **FSDP** 支持，但尚未处理优化状态子类 (optimization state subclass) 的集合通信操作 (collective ops)。
   - 他们建议编写一份开发指南，这将有助于吸引开发者的兴趣，因为缺乏集成指导可能会导致项目被放弃。
- **FSDP 实现审查**：另一名成员同意下周审查 **FSDP** 的实现。
   - *期待下周深入研究。*


  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1261684089568891021)** (46 messages🔥): 

> - `Switching to cudaMallocManaged` (切换到 cudaMallocManaged)
> - `llm.cpp updates` (llm.cpp 更新)
> - `WebGPU insights` (WebGPU 见解)
> - `gpt3v1 by karpathy` (karpathy 的 gpt3v1)
> - `GPT-3 model interpolation` (GPT-3 模型插值) 


- **切换到 cudaMallocManaged 以提高内存效率**：**Eriks.0595** 建议从 `cudaMalloc` 切换到 `cudaMallocManaged`，以支持内存不足的设备，并确保在不降低现有功能速度的情况下进行非侵入式更改。
   - *Eriks.0595* 强调了这一特性对于小型 GPU 集成的重要性。
- **llm.cpp 过去 4 个月的主要更新**：**Kashimoo** 询问了在中断 4 个月后 llm.cpp 的更新情况，**Eriks.0595** 解释说几乎所有内容都发生了变化。
- **WebGPU 的广泛应用**：**Akakak1337** 对 WebGPU 的非 Web 用途表示惊讶，并计划观看相关的演讲视频以获取更多见解。
- **Mup 运行见解与性能**：在讨论 **mup run** 时，**akakak1337** 提供了性能细节，例如在 HellaSwag 上的准确率为 **0.495917**。
   - *Falconsfly* 对 token/sec 性能和精度损失表示担忧。
- **将 GPT-3 模型合并到 master 分支**：**Eriks.0595** 询问了将其模型系列扩展到 GPT-3 模型的问题，引发了关于模型插值的讨论。
   - **Akakak1337** 确认已将 GPT-3 模型合并到 master 分支，并讨论了匹配非单调 head 大小和深度的挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/Yuchenj_UW/status/1812893615372575180">来自 Yuchen Jin (@Yuchenj_UW) 的推文</a>：在训练了最大的 GPT-2 (1.5B) 之后，我决定走得“更深”，通过使用 @karpathy 的 llm.c 训练一个 2.7B 模型来感受 Scaling Law 📈 扩展模型非常直接，主要...</li><li><a href="https://github.com/karpathy/llm.c/pull/688">karpathy 的 feature/gpt3v1 · Pull Request #688 · karpathy/llm.c</a>：无描述
</li>
</ul>

</div>

### **CUDA MODE ▷ #[youtube-watch-party](https://discord.com/channels/1189498204333543425/1238931064223830016/)** (1 条消息): 

vkaul11: Hi
  

---


### **CUDA MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1262129045563506698)** (25 条消息🔥): 

> - `WebGPU 资源与支持`
> - `使用 Transformers.js 在浏览器中运行 LLM`
> - `在 Windows 上构建 Dawn 并排除故障`
> - `GPU 缓冲区与性能` 


- **通过新资源探索 WebGPU**：成员们分享了学习 WebGPU 的各种资源，包括介绍了 compute shaders 和优化步骤的 [WebGPU Fundamentals](https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders.html)。
   - 讨论强调浏览器支持主要集中在 Chrome，Firefox 的支持默认受限，而 Safari 则相对滞后。
- **尝试使用 Transformers.js 运行基于浏览器的 LLM**：一位成员提到了 [Transformers.js](https://huggingface.co/docs/transformers.js/index)，用于使用 ONNX Runtime 直接在浏览器中运行最先进的 machine learning 任务。
   - 他们指出它支持多种任务，包括文本分类、问答和图像分类，尽管他们还没有进行太多实验。
- **排除 Dawn 构建问题**：多条消息讨论了在 Windows 上构建 Dawn 的故障排除，其中 release build 表现异常，但 debug build 工作正常。
   - 重建策略包括使用 Google 的 CMake 发行版，并考虑使用 shared libraries 而不是 FetchContent 以提高稳定性。
- **了解 WebGPU 缓冲区限制**：一位成员解释说，浏览器中的 WebGPU 环境存在限制，例如 16 KB shared memory 和 128 MB buffers，这些是最小值。
   - 另一位成员质疑，由于这些限制，针对小数据量的 GPU offload 相比 CPU AVX 指令是否真的能提升性能。
- **分享经验与改进**：成员们分享了设置和使用 WebGPU 的经验，讨论了未来开发的各种挑战和潜在改进。
   - 反馈包括简化 shader 与 kernel 命名法的建议，以及对结构化参数更灵活的处理方式。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/transformers.js/index">Transformers.js</a>: 未找到描述</li><li><a href="https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders.html">WebGPU Compute Shader Basics</a>: 如何在 WebGPU 中使用 compute shaders</li><li><a href="https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders-histogram.html">WebGPU Compute Shaders - Image Histogram</a>: 高效计算图像直方图。</li><li><a href="https://github.com/AnswerDotAI/gpu.cpp/blob/main/examples/webgpu_from_scratch/run.cpp">gpu.cpp/examples/webgpu_from_scratch/run.cpp at main · AnswerDotAI/gpu.cpp</a>: 一个使用 WebGPU 进行便携式底层 GPU 计算的轻量级库。 - AnswerDotAI/gpu.cpp
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1261434252491489380)** (141 条消息🔥🔥): 

> - `OpenArena GitHub 项目`
> - `Cohere 活动链接混淆`
> - `LlamaIndex KG 去重`
> - `Karpathy 谈 AI 训练成本`
> - `账户支持问题` 


- **OpenArena GitHub 项目亮相**：一位成员分享了他们的项目 [OpenArena](https://github.com/syv-ai/OpenArena)，该项目旨在让 LLM 相互竞争，以提升数据集质量。
- **Cohere 活动链接混淆**：成员们讨论了对 **Cohere 活动** 链接的困惑，部分人无法进入会议，而其他人提供了关于扩散模型生成频谱图的**客座演讲环节**的正确 Zoom 链接。
- **LlamaIndex KG 节点去重详解**：一位成员分享了一段 [YouTube 视频](https://youtu.be/vMz0icWZd5A)，解释了 **LlamaIndex 如何处理其知识图谱中的节点去重**。
- **AI 训练成本骤降**：一位成员强调了 [Karpathy 的详细讨论](https://x.com/karpathy/status/1811467135279104217)，内容涉及过去 5 年中训练 GPT-2 等 AI 模型的成本大幅下降。
- **Cohere 账户支持问题**：一位成员报告了在一次组织邀请失误后其 **Cohere 账户** 消失的问题，并收到了支持团队提交工单以解决问题的指导。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/karpathy/status/1811467135279104217">Andrej Karpathy (@karpathy) 的推文</a>：2019 年，OpenAI 发布了 GPT-2：https://openai.com/index/better-language-models/。今天（约 5 年后），你只需花费约 672 美元，在单个 8XH100 GPU 节点上运行 24 小时即可训练自己的模型。...</li><li><a href="https://cohere.com/events/cohere-for-ai-guest-speaker-ziyang-chen-2024">Cohere For AI - 客座演讲嘉宾：Ziyang Chen，博士生</a>：会发声的图像：在单一画布上合成图像和声音</li><li><a href="https://youtu.be/vMz0icWZd5A">LlamaIndex KG | 节点去重</a>：在这段录音中，我详细解释了 LlamaIndex 在创建知识图谱后如何进行节点去重。代码：https://github.com/raji...</li><li><a href="https://cohere.com/">Cohere | 领先的企业级 AI 平台</a>：Cohere 提供行业领先的大语言模型 (LLM) 和 RAG 能力，量身定制以满足解决现实世界问题的企业级用例需求。</li><li><a href="https://docs.cohere.com/docs/tool-use">Cohere 模型的工具使用 (Tool Use) - Cohere 文档</a>：未找到描述</li><li><a href="https://tenor.com/view/inside-out-joy-hi-hey-hello-gif-13317321031557907374">《头脑特工队》乐乐 GIF - Inside Out Joy Hi - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/syv-ai/OpenArena">GitHub - syv-ai/OpenArena</a>：通过在 GitHub 上创建账户来为 syv-ai/OpenArena 的开发做出贡献。</li><li><a href="https://umich.zoom.us/j/8022650618?pwd=V0VvYnAyQVBlNnIrUktGNyt6WFE1dz09">加入我们的云高清视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有简单、可靠的云平台，可用于跨移动端、桌面端和会议室系统的视频和音频会议、聊天及网络研讨会。Zoom ...
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1261474660353179699)** (26 条消息🔥): 

> - `Cohere 的 NPM 模块`
> - `使用 Langchain 和 Cohere 的 r/localllama 机器人`
> - `使用来自 Reddit 的 JSON`
> - `多个 AI subreddit 更新` 


- **Cohere 的 NPM 模块发布**：[NPM 模块的新更新](https://github.com/samestrin/llm-interface)现在包含对 **Cohere** 的支持，增强了其与各种 **LLM** 提供商交互的便利性。
   - 分享了仓库图片和 **NPM** 安装详情，展示了与多个 AI 平台的无缝集成。
- **使用 Langchain 和 Cohere 构建的 r/localllama 机器人**：创建了一个新机器人，使用 **Langchain** 和 **Cohere Command-R-Plus** 将 **r/localllama** 的热门帖子抓取并总结为 Discord 频道的新闻风格帖子。
   - 机器人的代码已分享，引起了成员们的兴奋，大家觉得它非常有用。
- **从 Reddit 提取帖子数据为 JSON**：成员们讨论了一种通过在 **r/localllama** 热门帖子的 URL 后添加 `.json` 来提取信息的方法。
   - 重点介绍了 "Your Settings Are Probably Hurting Your Model" 帖子，强调了采样器设置对模型性能的影响。
- **新闻机器人的多 AI subreddit 更新**：机器人已更新，支持多个 AI subreddit 并改进了故事排序机制。
   - 分享了让 **Cohere** 根据主题对新闻故事进行分类并引导至相应频道的计划。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/samestrin/llm-interface">GitHub - samestrin/llm-interface: A simple NPM interface for seamlessly interacting with 36 Large Language Model (LLM) providers, including OpenAI, Anthropic, Google Gemini, Cohere, Hugging Face Inference, NVIDIA AI, Mistral AI, AI21 Studio, LLaMA.CPP, and Ollama, and hundreds of models.</a>：一个简单的 **NPM** 接口，用于与 36 个 **Large Language Model (LLM)** 提供商无缝交互，包括 **OpenAI**, **Anthropic**, **Google Gemini**, **Cohere**, **Hugging Face Inference**, **NVIDIA AI**, **Mistral AI**, **AI21 Studio**, **LLaMA.CPP** 和 **Ollama**，以及数百个模型。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/17vonjo/your_settings_are_probably_hurting_your_model_why/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/17vonjo/your_settings_are_probabl">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/17vonj">Reddit - Dive into anything</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1261410626748547214)** (70 条消息🔥🔥): 

> - `伦敦的 AI 聚会`
> - `OpenAI 合作`
> - `模型基准测试 (Model Benchmarking)`
> - `模型中的时间考量`
> - `机器学习会议` 


- **伦敦的 AI 聚会缺乏深度**：成员们讨论了伦敦的 AI 聚会通常讨论浅显且频率较低，建议查看 UCL 和 Imperial 的研讨会（seminars）和受邀演讲以获取更深层次的知识。
   - 有人指出，像 ICML 和 ICLR 这样的会议通常提供更深入的对话，特别是在特定领域的聚会以及与研究人员的 1-on-1 交流中。
- **用于 MechInterp 快速迭代的 Arrakis 项目**：一位用户请求对 [Arrakis](https://github.com/yash-srivastava19/arrakis) 提供反馈，这是一个旨在进行、跟踪和可视化机械可解释性（Mechanistic Interpretability）实验的库，集成了 tuned-lens 等工具。
   - 该项目旨在提高社区内的研究效率和实用性。
- **OpenLLMLeaderboard 基准测试数据的可用性**：关于 Hugging Face 上新 OpenLLMLeaderboard 测试集可用性的疑问，特别是数据集的部分内容是否未公开。
   - 对方澄清说 [HuggingFace](https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about#reproducibility) 通过允许下载所有公开数据来提供可复现性，确保没有隐藏元素。
- **LLM 训练中的时间相关性受到质疑**：一位用户对如何在建模相关性时利用时间和数据新鲜度表示兴趣，并指出目前向 LLM 传递时间戳的方法效果不佳。
   - 建议包括查阅有关特定方法论文、数据集和处理时间相关数据的基准测试文献，以实现更好的模型训练。
- **对 AI 应用的大上下文窗口（Context Windows）感兴趣**：一位社区倡导者正在寻求具有巨大上下文窗口（1M tokens）的托管模型建议，用于 AI 辅助的人权应用。
   - 他们分享了当前项目的进展、背景以及 [Discourse 链接](https://community.openai.com/t/inception-based-design-for-the-ai-assisted-creation-of-a-human-rights-application/863669)，并征求任何有用的见解或资源。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://livebench.ai/">LiveBench</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about#reproducibility">About</a>: 未找到描述</li><li><a href="https://github.com/yash-srivastava19/arrakis">GitHub - yash-srivastava19/arrakis: Arrakis 是一个用于进行、跟踪和可视化机械可解释性实验的库。</a>: Arrakis 是一个用于进行、跟踪和可视化机械可解释性实验的库。 - yash-srivastava19/arrakis</li><li><a href="https://community.openai.com/t/inception-based-design-for-the-ai-assisted-creation-of-a-human-rights-application/863669">基于 Inception 设计的 AI 辅助编写人权投诉应用</a>: 我在 vsCode IDE 中使用了 GitHub CoPilot 以及 ChatGpt4o 来转录包含文本消息内容的截图。</li><li><a href="https://www.meetup.com/london-machine-learning-meetup/events/)">alert--small</a>: 未找到描述</li><li><a href="https://www.youtube.com/@LondonMachineLearningMeetup/videos))">London Machine Learning Meetup</a>: London Machine Learning Meetup 是欧洲最大的机器学习社区。之前的演讲者包括 Juergen Schmidhuber, David Silver, Yoshua Bengio 和 Andrej Karpathy。欢迎参加我们的...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1261416634988494859)** (61 条消息🔥🔥): 

> - `Hermes 2 性能`
> - `使用 LangChain 的 RAG 系统`
> - `算力阈值治理`
> - `LLM 中的 RISE`
> - `模型压缩与准确率`

- **论文研究数学推理的合成数据**：一篇新[论文](https://arxiv.org/abs/2406.14532)探讨了使用模型生成的合成数据对 LLM 进行微调的有效性，发现在初始微调后，模型在自生成数据上进行微调的效率会翻倍。
   - 有人担心模型生成的正样本会放大伪相关性，导致 Scaling 趋势平缓甚至反向。
- **关于计算阈值治理的讨论**：一篇[文章](https://arxiv.org/abs/2407.05694)深入探讨了计算阈值如何通过监管计算资源的使用来影响 AI 安全和模型的风险状况。
   - 社区讨论了监管大规模训练任务可以防止少数实体垄断计算资源的观点。
- **用于可靠 RAG 系统的 LangChain**：一位成员在 GitHub 上分享了一个使用 LangChain 创建可靠 RAG（Retrieval-Augmented Generation）系统的项目。
   - 该[仓库](https://github.com/eericheva/langchain_rag)提供了详细的脚本和教程，帮助用户从零开始实现 RAG 系统。
- **RISE 实现 LLM 的自我改进**：一篇新论文介绍了 [RISE](https://openreview.net/forum?id=qDXdmdBLhR)，这是一种微调方法，使 LLM 能够通过多轮迭代改进其回答。
   - 该方法专注于递归内省（recursive introspection），允许模型从之前失败的尝试中学习并按顺序改进。
- **模型压缩技术与质量翻转**：研究分析了用于压缩大型模型的[量化技术](https://arxiv.org/abs/2407.09141)如何导致答案的“翻转”（flips），即即使整体准确率看起来没有变化，答案也会从正确变为错误。
   - 讨论强调，这种翻转意味着模型质量发生了更复杂的退化，有必要进行进一步的定性和定量评估。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2406.19384">The Remarkable Robustness of LLMs: Stages of Inference?</a>: 我们通过删除和交换相邻层，展示并研究了 Large Language Models 卓越的鲁棒性。我们发现，删除和交换干预保留了原始性能的 72-95%...</li><li><a href="https://arxiv.org/abs/2407.09141">Accuracy is Not All You Need</a>: 当使用量化等技术压缩 Large Language Models (LLMs) 时，证明此类技术有效性的主要方式是测量模型在各种...上的准确率。</li><li><a href="https://arxiv.org/abs/2312.01203">Harnessing Discrete Representations For Continual Reinforcement Learning</a>: Reinforcement learning (RL) agents 仅利用来自环境的观测结果做出决策，因此严重依赖这些观测结果的表示。尽管最近的一些...</li><li><a href="https://arxiv.org/abs/2208.07339">LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale</a>: Large language models 已被广泛采用，但推理时需要大量的 GPU 显存。我们为前馈和 attention projection 层开发了一种 Int8 矩阵乘法程序...</li><li><a href="https://arxiv.org/abs/2407.05694">On the Limitations of Compute Thresholds as a Governance Strategy</a>: 从表面上看，本文旨在理解一种名为计算阈值的相当深奥的治理工具。然而，为了探讨这些阈值是否能取得任何成果，我们必须...</li><li><a href="https://arxiv.org/abs/2405.20835">Outliers and Calibration Sets have Diminishing Effect on Quantization of Modern LLMs</a>: Post-Training Quantization (PTQ) 通过减少内存使用，实现更快的运行速度并兼容更易获得的硬件，从而提高 Large Language Models (LLMs) 的效率，但代价是...</li><li><a href="https://arxiv.org/abs/2406.14532">RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold</a>: 在模型生成的合成数据上进行训练是微调 LLMs 的一种很有前景的方法，但目前尚不清楚何时会有利或有害。在本文中，我们通过...研究了数学推理的这一问题。</li><li><a href="https://arxiv.org/abs/2401.12181">Universal Neurons in GPT2 Language Models</a>: 机械可解释性这一新兴领域的一个基本问题是，神经网络在多大程度上学习了相同的底层机制。换句话说，神经机制是否具有通用性...</li><li><a href="https://dynamicfieldtheory.org/">Home | Dynamic field theory</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/5076">KL-divergence by ikawrakow · Pull Request #5076 · ggerganov/llama.cpp</a>: 关于能够计算 KL 散度作为另一种量化准确性测试的潜在价值，已经进行了多次讨论。在 PR #4739 中，@Ttl 提供了 Python 脚本...</li><li><a href="https://github.com/eericheva/langchain_rag">GitHub - eericheva/langchain_rag</a>: 通过在 GitHub 上创建账号，为 eericheva/langchain_rag 的开发做出贡献。</li><li><a href="https://github.com/eericheva/langchain_rag/tree/main#item-one)">GitHub - eericheva/langchain_rag</a>: 通过在 GitHub 上创建账号，为 eericheva/langchain_rag 的开发做出贡献。</li><li><a href="https://openreview.net/forum?id=qDXdmdBLhR">Recursive Introspection: Teaching Foundation Model Agents How to...</a>: 在 Foundation Models 中实现智能 Agent 行为的核心部分是使它们能够反思自己的行为，推理并纠正错误。即使是强大的...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/)** (1 messages): 

wabi.sabi.1: 非常有趣，谢谢
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1261450950946852935)** (13 条消息🔥): 

> - `lm-eval Python API`
> - `lm-eval 的 PRAUC 指标`
> - `Quantization flips 研究`
> - `分布式 lm_evaluation`
> - `任务 YAML 中的自定义函数` 


- **在 Transformer Lens 模型中使用 lm-eval API**：一位成员询问如何将 `lm-eval` Python API 与自定义的 `Transformer-lens` 模型结合使用，并被建议通过继承 `lm_eval.api.model.LM` 或类似类来实现兼容性。
   - 他们感谢了提供帮助的[文档链接](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage)的建议者。
- **在 lm-eval 中计算 PRAUC 指标**：一位用户询问如何使用 `lm-eval` 为不平衡的测试数据实现 `PRAUC` 指标，这需要正样本概率输出。
   - 讨论中未提供具体答案，暗示该成员可能需要进一步的帮助。
- **Quantization Flips 研究发布**：一位成员分享了关于 [quantization flips](https://arxiv.org/abs/2407.09141) 的新论文，指出尽管压缩模型在 Benchmark 准确率上与基准模型匹配，但其行为可能有所不同。
   - 该研究利用了 Harness，强调了即使定量指标接近，压缩模型也会发生显著的行为变化。
- **在分布式设置中评估模型**：一位成员寻求关于在 `lm-harness` 中实现分布式评估的 `evaluate()` 方法，以及将剪枝模型加载到 `HFLM` 中的建议。
   - 虽然未提供具体解决方案，但该咨询仍对社区的建议和示例保持开放。
- **lm-eval YAML 中的自定义函数**：有人提出了关于传递给任务 YAML 中定义的自定义 `!function` 的参数问题。
   - 讨论尚未产生关于处理这些自定义函数的详细指导。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.09141">Accuracy is Not All You Need</a>: 当 Large Language Models (LLMs) 使用 Quantization 等技术进行压缩时，证明此类技术有效性的主要方式是衡量模型在 v... 上的准确率。</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage">lm-evaluation-harness/docs/interface.md at main · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 Few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/)** (1 条消息): 

bobby_mcbobface: 谢谢 Ryan！只是想确认我没有走上一条被废弃的道路。
  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1261476977269936140)** (104 条消息🔥🔥): 

> - `MonoNN Compiler`
> - `tinygrad Kernel 开销`
> - `MLX vs tinygrad 性能`
> - `改变形状的 Bitcasts`
> - `周一会议亮点` 


- **MonoNN 编译器提供优化的 GPU 利用率**：一种新的机器学习优化编译器 **MonoNN**，通过将整个神经网络容纳在单个 Kernel 中，解决了传统逐个 Kernel 执行方案中的低效问题。社区讨论了其 [论文演示](https://www.usenix.org/conference/osdi24/presentation/zhuang) 和 [源代码](https://github.com/AlibabaResearch/mononn)。
- **关于 tinygrad Kernel 开销的辩论**：社区成员根据实验结果，讨论了 AMD GPU 上每个 Kernel 显著的 **3-4us** Kernel 开销。
- **MLX 在速度和准确率上优于 tinygrad**：研究发现 **MLX** 比 **tinygrad** 更快且准确率更高，特别是在 beautiful_MNIST 基准测试中。
- **改变形状的 Bitcasts 面临的挑战**：在 **tinygrad** 中实现对改变形状的 Bitcasts 的支持正在取得进展，尽管主要在 GPU 设备上遇到了一些问题。
- **周一会议亮点**：会议涵盖了 **tinybox** 的更新以及 **lowerer** 和 **HWComandQueue device** 等新组件。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://ml-explore.github.io/mlx/build/html/usage/compile.html">Compilation &#8212; MLX 0.16.0 文档</a>：未找到描述</li><li><a href="https://www.usenix.org/conference/osdi24/presentation/zhuang">MonoNN: 在现代以 GPU 为中心的架构上为神经网络推理任务启用新的单体优化空间 | USENIX</a>：未找到描述</li><li><a href="https://github.com/tinygrad/tinygrad/commit/8940530290b04048074be1deadd24e5d91d67d28">添加 mlx beautiful_mnist 示例 · tinygrad/tinygrad@8940530</a>：未找到描述</li><li><a href="https://github.com/tinygrad/tinygrad/blob/ae4cb7994e73f35b6b467327d194394cdf52b99d/tinygrad/device.py#L207),">tinygrad/tinygrad/device.py (位于 ae4cb7994e73f35b6b467327d194394cdf52b99d) · tinygrad/tinygrad</a>：你喜欢 PyTorch？你喜欢 micrograd？那你一定会爱上 tinygrad！❤️ - tinygrad/tinygrad</li><li><a href="https://x.com/__tinygrad__/status/1811598734045991021)">来自 tiny corp (@__tinygrad__) 的推文</a>：这是来自 CIFAR 训练步骤的 Kernels。在右侧，tinygrad 现在显示了哪些操作导致了 Kernel 的创建。离伟大的错误信息又近了一步！
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1261513253171761304)** (27 条消息🔥): 

> - `avg_pool2d 中的 count_include_pad`
> - `Tensor 索引和 gather 函数`
> - `改进 Tinygrad 文档`
> - `基于比例拆分 Tensors` 


- **请求在 avg_pool2d 中增加 include pad 选项**：**Stable Diffusion** 训练评估需要像 PyTorch 那样的 `count_include_pad=False` 选项，成员们讨论了可能的实现方法。
   - 一位成员建议，如果 **MLPerf** 有需求，可以 Upstream 一个使用 `(pool -> sum) / (ones_like -> pool -> sum)` 的方法。
- **关于 Tensor 索引的澄清**：成员们澄清了 `probas[:, Y_train]` 和 `probas[Tensor.arange(len(logits)), Y_train]` 之间的区别，并讨论了为什么在 Tinygrad 中使用 Masking 而不是 Indexing 会使操作更快。
   - 一位成员提供了一个非常有用的 [快速入门指南链接](https://docs.tinygrad.org/quickstart/#training)，其中解释了相关实现。
- **修复 gather 函数中的 Bug**：在 Tinygrad 的 `gather` 函数中发现了一个与负索引处理相关的 Bug，导致行为异常。
   - 该问题已通过修正函数调用顺序得到解决，修复补丁将包含在即将发布的 PR 中。
- **为不同改进提交独立的 Pull Requests**：成员们一致认为，为了便于审查，最好为新的 Tensor 函数、模型实现和功能扩展提交独立的 PR。
   - 一位成员为 FID 实现了 `interpolate`，虽然可以使用，但暴露了一个被迅速处理的 Bug。
- **测试代码块的文档**：成员们讨论了如何执行文档中的代码块以确保其正确运行。
   - 分享了一个有用的 [本地运行 Tinygrad 文档的链接](https://github.com/tinygrad/tinygrad/blob/master/serve_docs.sh) 以供指导。



**提到的链接**：<a href="https://docs.tinygrad.org/quickstart/#training)">快速入门 - tinygrad 文档</a>：未找到描述

  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1261459945518268499)** (43 messages🔥): 

> - `Open LLM Leaderboard V2`
> - `Solving Reddit Link Hallucination`
> - `New Models in LMSys Arena`
> - `Cursor's Composer Feature`
> - `SpreadsheetLLM by Microsoft` 


- **Open LLM Leaderboard V2 专题节目发布**：一位用户宣布了关于 **Open LLM Leaderboard V2** 的新一期 Latent Space 节目。
   - 另一位用户以“yessir”表达了对新节目的兴奋。
- **关于 SmolAI 解决 Reddit 链接幻觉的假设**：成员们分享了关于 **SmolAI** 如何解决 Reddit 链接幻觉问题的理论，包括 **pre-check**（预检查）和 **post-proc**（后处理）方法。
   - 一位成员提到应用类似的预检查方法来选择 ID 以确保准确性。
- **LMSys Arena 新模型背后的谜团**：关于谁可能是 **LMSys Arena** 中新模型幕后推手的问题引发了讨论，并链接了相关的强烈观点。
   - 一位成员听说有传言称 **Command R+ jailbreaks**（越狱）在其中一个新模型上有效。
- **对 Cursor 的 Composer 功能的热议**：用户对 **Cursor** 的新 Composer 功能表现出浓厚兴趣，讨论了其 **beta release**（测试版发布）并与其他 UX 方案进行了比较。
   - 成员们分享了对该功能易用性和价格的看法，尽管对订阅费用有所顾虑，但初步印象积极。
- **微软推出 SpreadsheetLLM**：微软发布了 **SpreadsheetLLM**，旨在通过一种新颖的 **SheetCompressor** 编码框架优化 LLM 处理电子表格的能力。
   - 成员们对该技术的潜力表示关注，因为它通过修改输入数据来更好地适配各种 LLM，而无需进行 fine-tuning（微调）。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/Smol_AI/status/1811957074840158255">来自 AI News by Smol AI (@Smol_AI) 的推文</a>: [2024年7月12日]  https://buttondown.email/ainews/archive/ainews-we-solved-hallucinations/  我们解决了幻觉问题！</li><li><a href="https://x.com/apples_jimmy/status/1812029979888439525?s=61">来自 Jimmy Apples 🍎/acc (@apples_jimmy) 的推文</a>: 看起来 Lmsys arena 中出现了新模型。</li><li><a href="https://fortune.com/2024/07/12/lattice-ai-workers-sam-altman-brother-jack-sarah-franklin/">估值 30 亿美元的 Lattice “创造了历史”，成为首个赋予 AI “员工”权利的公司</a>: 值得注意的是，Lattice 去年解雇了 100 名人类员工。</li><li><a href="https://arxiv.org/html/2407.09025v1">SpreadsheetLLM: 为大语言模型编码电子表格</a>: 未找到描述</li><li><a href="https://x.com/shaoruu/status/1812412514350858634">来自 ian (@shaoruu) 的推文</a>: composer 已在 @cursor_ai 开放测试，这是我用它在 6 分钟内制作打字测试的视频（6倍速）：</li><li><a href="https://x.com/teortaxesTex/status/1812226271457296395">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>: 强烈观点，轻微坚持。引用 Nic (@nicdunz) @kalomaze @teortaxesTex 兄弟 😭
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 messages): 

swyxio: 新播客发布！ https://x.com/swyx/status/1811898574416019562
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1261411650687209588)** (86 messages🔥🔥): 

> - `Memorable Acronyms`
> - `More demos and examples`
> - `Evaluation techniques`
> - `Logprob usages`
> - `State management` 


- **易记的首字母缩写：3E**：一位成员建议使用更易记的缩写，如 **Extract, Evaluate, Extend/Expand (3E)**（提取、评估、延伸/扩展）。
- **对更多演示和示例的需求**：多位成员强调在讨论中需要更多演示和示例，特别是涉及技术实现的部分。
- **探索评估技术：Logprob 和 GPTscore**：成员们讨论了不同的评估技术，如 **logprob**、**GPTscore**，以及超参数优化工具如 [prompt-hyperopt](https://github.com/Mavenoid/prompt-hyperopt)。
   - 提到了一篇与此相关的论文：[Simple approach for contextual hallucinations](https://arxiv.org/html/2407.07071v1)。
- **状态管理工具对比**：对比了不同的状态管理风格，重点关注 **ReAct framework**、**Langgraph** 和 [XState](https://github.com/statelyai/xstate)。
   - *Langgraph* 因其能更好地处理通过节点的每一步图状态内存（graph-state memory）而受到关注。
- **即将举行的 AI in Action 演讲**：下周，**VikParuchuri** 将演示如何使用 [marker](https://github.com/VikParuchuri) 和 surya 等工具将 PDF 转换为 Markdown。


<div class="linksMentioned">

<strong>提及的链接</strong>：

</div>

<ul>
<li>
<a href="https://zod.dev/">TypeScript-first schema validation with static type inference</a>: TypeScript 优先的模式验证，具有静态类型推断</li><li><a href="https://huggingface.co/nisten/bakllava-14b-2xMoE-alpha-build">nisten/bakllava-14b-2xMoE-alpha-build · Hugging Face</a>: 未找到描述</li><li><a href="https://arxiv.org/html/2407.07071v1">Lookback Lens: Detecting and Mitigating Contextual Hallucinations in Large Language Models Using Only Attention Maps</a>: Lookback Lens：仅使用 Attention Maps 检测和缓解 Large Language Models 中的上下文幻觉</li><li><a href="https://arxiv.org/abs/2310.14566">HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models</a>: 我们介绍了 HallusionBench，这是一个专为评估图像上下文推理而设计的综合基准。该基准对先进的 Large Vision-Language Models 提出了重大挑战 (...</li><li><a href="https://github.com/truera/trulens">GitHub - truera/trulens: Evaluation and Tracking for LLM Experiments</a>: LLM 实验的评估与追踪。通过在 GitHub 上创建账户，为 truera/trulens 的开发做出贡献。</li><li><a href="https://github.com/tianyi-lab/HallusionBench">GitHub - tianyi-lab/HallusionBench: [CVPR&#39;24] HallusionBench: You See What You Think? Or You Think What You See? An Image-Context Reasoning Benchmark Challenging for GPT-4V(ision), LLaVA-1.5, and Other Multi-modality Models</a>: [CVPR&#39;24] HallusionBench：你看到的是你所想的吗？还是你想到的是你所看到的？一个对 GPT-4V(ision)、LLaVA-1.5 和其他多模态模型具有挑战性的图像上下文推理基准 - ti...</li><li><a href="https://github.com/openvinotoolkit/anomalib">GitHub - openvinotoolkit/anomalib: An anomaly detection library comprising state-of-the-art algorithms and features such as experiment management, hyper-parameter optimization, and edge inference.</a>: 一个异常检测库，包含最先进的算法以及实验管理、超参数优化和边缘推理等功能。 - openvinotoolkit/anomalib</li><li><a href="https://github.com/Mavenoid/prompt-hyperopt">GitHub - Mavenoid/prompt-hyperopt: Improve prompts for e.g. GPT3 and GPT-J using templates and hyperparameter optimization.</a>: 使用模板和超参数优化来改进 GPT3 和 GPT-J 等模型的 Prompt。 - Mavenoid/prompt-hyperopt</li><li><a href="https://github.com/chand1012/git2gpt">GitHub - chand1012/git2gpt: Convert a Git repo into a ChatGPT prompt!</a>: 将 Git 仓库转换为 ChatGPT Prompt！通过在 GitHub 上创建账户，为 chand1012/git2gpt 的开发做出贡献。</li><li><a href="https://github.com/VikParuchuri">VikParuchuri - Overview</a>: VikParuchuri 拥有 90 个仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/statelyai/xstate">GitHub - statelyai/xstate: Actor-based state management &amp; orchestration for complex app logic.</a>: 用于复杂应用逻辑的基于 Actor 的状态管理和编排。 - statelyai/xstate</li><li><a href="https://github.com/seanchatmangpt/dspygen">GitHub - seanchatmangpt/dspygen: A Ruby on Rails style framework for the DSPy (Demonstrate, Search, Predict) project for Language Models like GPT, BERT, and LLama.</a>: 为 GPT、BERT 和 LLama 等 Language Models 的 DSPy (Demonstrate, Search, Predict) 项目提供的 Ruby on Rails 风格框架。 - seanchatmangpt/dspygen</li><li><a href="https://github.com/jxzhangjhu/Awesome-LLM-Uncertainty-Reliability-Robustness">GitHub - jxzhangjhu/Awesome-LLM-Uncertainty-Reliability-Robustness: Awesome-LLM-Robustness: a curated list of Uncertainty, Reliability and Robustness in Large Language Models</a>: Awesome-LLM-Robustness：关于 Large Language Models 中不确定性、可靠性和鲁棒性的精选列表 - jxzhangjhu/Awesome-LLM-Uncertainty-Reliability-Robustness</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=0#gid=0">AI In Action: Weekly Jam Sessions</a>: 2024 主题, 日期, 协调人, 资源, @dropdown, @ GenAI 的 UI/UX 模式, 1/26/2024, nuvic, &lt;a href=&quot;https://maggieappleton.com/squish-structure&quot;&gt;https://maggieappleton.com/squish-stru...</li><li><a href="https://github.com/EGjoni/DRUGS">GitHub - EGjoni/DRUGS: Stop messing around with finicky sampling parameters and just use DRµGS!</a>: 别再纠结于繁琐的采样参数了，直接使用 DRµGS 吧！ - EGjoni/DRUGS</li><li><a href="https://github.com/elder-plinius/AutoTemp">GitHub - elder-plinius/AutoTemp: A trial-and-error approach to temperature opimization for LLMs. Runs the same prompt at many temperatures and selects the best output automatically.</a>: 一种针对 LLM 温度优化的试错方法。在多种温度下运行相同的 Prompt，并自动选择最佳输出。 - elder-plinius/AutoTemp
</li>
</ul>

</div>
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1261408034211762338)** (86 messages🔥🔥): 

> - `OpenArena Project`
> - `ORPO Training`
> - `Anthropic Prompt Integration`
> - `RAG Model Dataset`
> - `Weighting Conversation Data` 


- **OpenArena 项目实现 100% 开源**: 用户 le_mess 正在开发一个 [100% 开源的本地版本](https://github.com/syv-ai/OpenArena) 数据集创建工具，该工具最初是为 OpenRouter 设计的，但现在改用 Ollama。
   - 该项目旨在为各种模型的数据集创建提供更灵活、更开放的环境。
- **ORPO 训练显存占用的挑战**: 用户 xzuyn 提出了关于 ORPO 训练显存占用的担忧，指出显存占用会激增并最终导致 OOM，即使最大序列长度（max sequence）设置为 2k。
   - 讨论揭示了在 Tokenization 后缺乏关于丢弃长序列的信息，这导致了不稳定的显存激增。
- **Axolotl 的 Anthropic Prompt 格式**: Kalomaze 讨论了将 [官方 Claude/Anthropic Prompt 格式](https://github.com/axolotl-ai-cloud/axolotl/blob/main/src/axolotl/utils/chat_templates.py) 集成到 Axolotl 中，使用特殊 Token 来区分 system、human 和 assistant 轮次。
   - 尽管有人担心特殊 Token 的可读性和泛化性，但考虑到现有 SOTA 模型的实践，这被认为是可接受的。
- **RAG 模型数据集爬取的安全担忧**: 用户 nafnlaus00 提出了关于使用 Chromium 渲染需要 JavaScript 的网站（如 Quora）来创建 RAG 模型数据集的安全担忧。
   - Le_mess 建议排查 header/params 问题，并考虑使用 firecrawl 或 Jina API 等服务进行更安全的爬取。
- **提议对训练数据进行加权**: Tostino 建议在 Pretraining 和 SFT 中实现一套对对话数据不同部分进行加权的系统，允许使用负权重来教导模型避开某些 Token。
   - 这可以实现优化循环，通过对理解较差的部分或“错误路径”进行不同的加权，从而改善模型效果。


<div class="linksMentioned">

<strong>提及链接</strong>:

<ul>
<li>
<a href="https://github.com/axolotl-ai-cloud/axolotl/blob/main/src/axolotl/utils/chat_templates.py">axolotl/src/axolotl/utils/chat_templates.py at main · axolotl-ai-cloud/axolotl</a>: 欢迎在 GitHub 上为 axolotl-ai-cloud/axolotl 的开发做出贡献。</li><li><a href="https://github.com/syv-ai/OpenArena">GitHub - syv-ai/OpenArena</a>: 欢迎在 GitHub 上为 syv-ai/OpenArena 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1262176377088507965)** (5 messages): 

> - `Chat template dataset type`
> - `PR review process`
> - `Configuration flexibility`
> - `Training labels configuration`
> - `Handling token offsets` 


- **即将提交 Chat Template 数据集的 PR**: 用户宣布即将提交一个新的 Chat Template 数据集类型的 **PR**，该类型在训练部分提供了灵活性。
   - 这包括选择要训练的角色、配置 `train_on_eos` 以及处理数据集内的特定训练部分。
- **对 PR 评审停滞的担忧**: 一位成员对 **PR 评审** 进度停滞表示担忧，并提到了自己和另一位用户的特定 PR。
   - 用户问道：“PR 评审是不是卡住了？”，并指出了 [他们的 PR](https://github.com/axolotl-ai-cloud/axolotl/pull/1725) 和 [另一个 PR](https://github.com/axolotl-ai-cloud/axolotl/pull/1733)。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1261844949897445406)** (6 messages): 

> - `Eric's Spectrum Work`
> - `Quantizing Dolphin Vision 72b`
> - `4-bit Model on 96GB Mac Pro` 


- **Eric 的 Spectrum 探索引起关注**: 一位成员提到 Eric 一直在研究 Spectrum，这引起了另一位正在审阅相关论文的成员的兴趣。
   - 他们指出，初步阅读后发现该论文*非常有趣*。
- **量化 Dolphin Vision 72b 的考量**: 一位成员询问了量化 **Dolphin Vision 72b** 以最小化 VRAM 占用的可行性。
   - 另一位成员回答说 **4-bit 量化** 应该仍然效果良好，并建议探索 **GGUF 或 EXL2 的更低量化版本**。
- **在 96GB Mac Pro 上运行 4-bit 模型**: 一位成员分享道，**4-bit 量化版本可以适配** 拥有 96GB 统一内存的顶配 **Mac Pro**。
   - 他们提到正在目前的配置上运行该模型的 **Inference**。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/)** (1 messages): 

n_tt_n: 我喜欢 capybara，用它取得了很棒的效果

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1261461277901848689)** (18 messages🔥): 

> - `LoRA 合并后将模型推送到 Hub`
> - `Vicuna 聊天模板支持`
> - `Vicuna 模板的配置选项` 


- **LoRA 合并后将模型推送到 Hub**：一位成员询问在将 LoRA 合并到基座模型后如何将模型推送到 Hub，建议使用 `HfApi` 的 `upload_folder` 方法。
   - 另一位成员建议使用更简单的方法，即使用 `huggingface-cli upload` 命令：`huggingface-cli upload wasamkiriua/model-name .`。
- **确认支持 Vicuna 聊天模板**：确认 Axolotl 支持 Vicuna 聊天模板，可以在配置文件中将 `conversation` 选项设置为 `vicuna_v1.1` 来指定。
   - 该支持允许按照 Vicuna 模板格式处理涉及人类和 GPT 交互的对话。
- **聊天模板配置标志的有效选项**：`chat_template` 配置标志不能直接设置为 `vicuna`；有效选项包括 `alpaca`、`chatml`、`inst`、`gemma`、`cohere`、`llama3` 和 `phi_3`。
   - 成员们同意在处理基于 Vicuna 的模型时，可以省略 `chat_template` 标志并在之后手动设置。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/openaccess-ai-collective/axolotl/tree/main/docs/dataset-formats/conversation.qmd#L1L64)">axolotl/docs/dataset-formats/conversation.qmd at main · axolotl-ai-cloud/axolotl</a>: 欢迎提出 Axolotl 问题。通过在 GitHub 上创建账户为 axolotl-ai-cloud/axolotl 的开发做出贡献。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=2e654e43-06c9-4e97-88bd-5fd61c91a7c6)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=3b3a1901-727b-4dbc-9426-dcf10d932051)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1261804408430399498)** (9 messages🔥): 

> - `GPTs Agents`
> - `OpenAI 平台的侧边栏`
> - `用于 Axolotl 训练的自定义聊天模板`
> - `Axolotl 训练设置`
> - `模板的 Jinja 格式` 


- **GPTs Agents 在初始训练后无法学习**：一位成员分享了关于 GPTs Agents 无法从初始训练后提供的额外信息中学习的担忧。
   - 另一位成员澄清说，上传的文件被保存为供 Agent 参考的“知识”文件，但**它们不会持续更新 Agent 的基础知识**。
- **OpenAI 平台的侧边栏发生变化**：成员们讨论了 platform.openai.com 侧边栏的变化，注意到两个图标（threads 和 messages）消失了。
   - 他们推测了这一变化对用户导航的潜在原因和影响。
- **为 Axolotl 训练设置自定义聊天模板**：一位成员请求帮助转换用于 Axolotl 训练的自定义聊天模板，并提供了他们想要实现的特定配置。
   - 另一位成员提供了逐步指导，包括 Jinja 模板格式和用于配置 Axolotl 的 YAML 示例。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=71f0f5e0-3659-41d9-b28e-780759c1d47d)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=6f51d3b7-a886-472f-ae22-45f2e0b54aeb)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。
</li>
</ul>

</div>

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1261442258171990117)** (31 条消息🔥): 

> - `OpenAI working on Strawberry`
> - `New models in LMSYS arena`
> - `Stealth releases of models in LMSYS` 


- **OpenAI 的 Strawberry 将增强推理能力**：据 [Reuters](https://www.reuters.com/technology/artificial-intelligence/openai-working-new-reasoning-technology-under-code-name-strawberry-2024-07-12/) 报道，OpenAI 正在研发名为 **Strawberry** 的新推理技术，该技术与斯坦福大学在 2022 年开发的 **Self-Taught Reasoner** 或 **STaR** 具有相似之处。
   - 讨论显示，知情人士认为它类似于斯坦福大学的一种方法 **STaR**。
- **LMSYS 将新模型引入竞技场**：[Jimmy Apples](https://x.com/apples_jimmy/status/1812029979888439525?s=46) 指出，新模型正在 **LMSYS arena** 中出现，引发了社区的热议。
   - 讨论的模型包括 **column-r** 和 **column-u**，据传来自 **Cohere**。
- **LMSYS 中的隐秘模型发布**：Twitter 用户 [@btibor91](https://x.com/btibor91/status/1812491983220343239?s=46) 确认了将新模型隐秘推送到 LMSYS Chatbot Arena 的趋势，并提到了四个即将推出的模型，包括 **eureka-chatbot** 和 **upcoming-gpt-mini**。
   - 根据错误信息和社区成员的提示，**Eureka-chatbot** 似乎是由 **Google** 训练的。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/apples_jimmy/status/1812029979888439525?s=46">来自 Jimmy Apples 🍎/acc (@apples_jimmy) 的推文</a>: 似乎 Lmsys arena 中出现了一个新模型。</li><li><a href="https://fxtwitter.com/TheXeophon/status/1812069628685815986">来自 Xeophon (@TheXeophon) 的推文</a>: Column-U 也可以用同样的提示词进行越狱，所以我猜它也是一个 cohere 模型。</li><li><a href="https://x.com/apples_jimmy/status/1812047899137851811?s=46">来自 Jimmy Apples 🍎/acc (@apples_jimmy) 的推文</a>: 还有一个看起来非常棒的新 column-r……我们终于要见证大事发生了吗？ 引用 Jimmy Apples 🍎/acc (@apples_jimmy) 的话：似乎 Lmsys arena 中出现了一个新模型。</li><li><a href="https://fxtwitter.com/TheXeophon/status/1812069172727201808">来自 Xeophon (@TheXeophon) 的推文</a>: Column-R 是一个 cohere 模型，Command R+ 的越狱方法在它身上也奏效。</li><li><a href="https://x.com/btibor91/status/1812491983220343239?s=46">来自 Tibor Blaho (@btibor91) 的推文</a>: 看起来，在发布前隐秘地将新模型推送到 LMSYS Chatbot Arena（且大多不可选）进行氛围测试（vibe check）和造势已成为一种新趋势。目前有 4 个即将推出的模型，据我所知...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1261397531019972733)** (23 messages🔥): 

> - `Mistral-7B instruct-tuning`
> - `Orca3/AgentInstruct paper`
> - `InFoBench benchmark`
> - `WizardArena/ArenaLearning paper`
> - `ChatbotArena competition` 


- **Mistral-7B 指令微调（instruct-tuning）受到审视**：讨论集中在 **Orca3/AgentInstruct 论文**中提到的相对于 **Mistral-7B 指令微调**的改进，并对 Mistral 指令微调数据集的强度感到好奇。
   - 有人提出了关于 **Mistral-7B** 最为人熟知的指令微调的问题，暗示当前的数据集可能并非特别健壮。
- **InFoBench 基准测试引发分歧**：**InFoBench (Instruction Following Benchmark)** 作为一个新基准被引入，引发了关于其与标准对齐数据集相关性的疑问。
   - 辩论围绕 **EQ Bench** 和 **InFoBench** 等基准测试在突出 LM 价值品质方面是否重要，因为它们与 MMLU 性能等现有基准高度相关。
- **WizardArena 论文和 ChatbotArena 竞赛分析**：参与者讨论了 **WizardArena/ArenaLearning 论文**，该论文详细介绍了使用人类偏好评分评估模型的方法，以及相关的 **Kaggle 竞赛**。
   - 参与者对多轮合成交互生成和评估表现出兴趣，特别是对 **WizardArena** 如何设置其评审过程和多轮评估感到好奇。
- **关于难度等级预测的问题**：**WizardArena 论文**提到使用 LM 预测指令难度等级，引发了对其准确性和现实世界相关性的疑问。
   - 有推测认为 LM 是否能真正预测自身的弱点，并参考了关于 **LM self-knowledge** 的现有文献。
- **讨论中注意到了极高的发帖频率**：一位用户承认了自己的高发帖率，并鼓励其他人积极参与对话。
   - 该用户似乎非常渴望参与并分享他们对各种论文和基准测试的阅读心得与见解。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.kaggle.com/competitions/lmsys-chatbot-arena/overview)">LMSYS - Chatbot Arena Human Preference Predictions | Kaggle</a>：未找到描述</li><li><a href="https://www.interconnects.ai/p/rlhf-roundup-2024?r=68gy5&utm_campaign=post&utm_medium=web">RLHF roundup: Getting good at PPO, charting RLHF’s impact, RewardBench retrospective, and a reward model competition</a>：从事语言模型微调工作时需要注意的事项。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1261730709979005089)** (7 messages): 

> - `Finite State Machine`
> - `Paper Rewriting Controversy`
> - `Google Plagiarism` 


- **用于结构化生成的有限状态机 (Finite State Machine)**：根据 @remilouf 的帖子，**Outline 用于结构化生成的有限状态机**已经在 [arXiv](https://x.com/remilouf/status/1812164616362832287) 上发布近一年了。
   - *我感到受宠若惊，但仍然……*
- **Google 被指控改写技术报告**：**Brandon Willard** 报告称，Google 的一些人完全[改写了他们的技术报告](https://x.com/BrandonTWillard/status/1812163165767053772)，虽然引用了它，但对差异的简短评论非常荒谬。
   - 他引用了 @remilouf 的话，并使用了“抄袭” (plagiarism) 一词来强调问题的严重性。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/BrandonTWillard/status/1812163165767053772">来自 Brandon T. Willard @brandonwillard@fosstodon.org (@BrandonTWillard) 的推文</a>：是的，看起来 Google 的一些人完全改写了我们的技术报告。虽然他们确实引用了它，但关于差异的简短评论非常荒谬。引用 Rémi 〰️ (@remilouf) 的话：抄袭 (Plag)...</li><li><a href="https://x.com/remilouf/status/1812164616362832287">来自 Rémi 〰️ (@remilouf) 的推文</a>：我感到受宠若惊，但仍然……
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1261443600798187520)** (12 条消息🔥): 

> - `OpenAI 营收推测`
> - `OpenAI Supply Co. 商店`
> - `Shopify 使用情况`
> - `Interconnects 周边`
> - `黑客松与免费周边` 


- **VCs 根据聊天机器人摘要推测 OpenAI 营收**：[Aaron Holmes](https://x.com/aaronpholmes/status/1811870687037960467?s=46) 指出，风险投资人（VCs）正在传阅一份关于 **OpenAI 营收** 的推测性报告，该报告完全基于公共网络资源的聊天机器人摘要。
   - 关于第一手报道，他引用了[上个月发布的一篇详细文章](https://www.theinformation.com/articles/openais-annualized-revenue-doubles-to-3-4-billion-since-late-2023)。
- **OpenAI Supply Co. 商店现仅限内部访问**：根据 [B Tibor 的帖子](https://x.com/btibor91/status/1812778486039290260?s=46)确认，**OpenAI Supply Co.** 商店现在需要使用 @openai.com 的 Microsoft 账号登录。
   - *这可能意味着该商店目前仅供内部使用，或不应向公众开放。*
- **通过 Shopify 销售 OpenAI 周边**：关于 **OpenAI 周边** 的讨论集中在利用 Shopify 开设周边商店。
   - 一位成员提到他们自己的 [Interconnects Shopify 商店](https://interconnects.myshopify.com/)，并展示了 [Coder Hoodie](https://interconnects.myshopify.com/products/coder-hoodie) 和 [Coffee Vessel #1](https://interconnects.myshopify.com/products/coffee-vessel-1) 等产品。
- **参加黑客松获取免费 OpenAI 周边**：有人建议，参加 **黑客松（hackathon）** 可能是获取免费 **OpenAI 周边** 的好方法。
   - *这是一种利用活动进行品牌推广的聪明方式。*


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/aaronpholmes/status/1811870687037960467?s=46">来自 aaron holmes (@aaronpholmes) 的推文</a>: 许多 VCs 今天正在传阅一份推测 OpenAI 营收的“报告”，其内容完全基于公共网络资源的聊天机器人摘要。如果你想了解关于 OpenAI 营收数据的第手报道，...</li><li><a href="https://supply.openai.com/password">OpenAI Supply Co.</a>: OpenAI Supply Co.</li><li><a href="https://interconnects.myshopify.com/">Interconnects AI Store</a>: Interconnects.ai 博客官方周边，面向 RL 爱好者。</li><li><a href="https://x.com/btibor91/status/1812778486039290260?s=46">来自 Tibor Blaho (@btibor91) 的推文</a>: OpenAI Supply Co. Shopify 商店现在需要使用 @ openai dot com Microsoft 账号登录 - 确认了其目前仅限内部使用或不应被公开访问。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1262482366786572380)** (4 条消息): 

> - `加州 AI 法案 SB 1047`
> - `绕过付费墙`
> - `Archive.is`
> - `硅谷辩论`
> - `《财富》(Fortune) 文章` 


- **加州 AI 法案 SB 1047 引发激烈辩论**：**加州 AI 法案 SB 1047** 于 5 月在州参议院以 32 比 1 的投票结果通过，目前在激烈的游说和讨论中，正准备迎接 8 月的最终投票。
   - 州参议员 **Scott Wiener** 将这场辩论描述为 *“喷气机帮对阵鲨鱼帮” (Jets vs Sharks)*，**AI 安全专家**与顶级**风险投资人**就该法案的影响产生了严重分歧。
- **使用 Archive.is 绕过付费墙**：讨论中透露了一种通过使用 [Archive.is](https://archive.is/e5n9A) 来绕过付费墙的方法，从而可以访问 **Fortune** 等网站的付费内容。
   - 一位用户对这些网站尚未修复此**漏洞**表示惊讶。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://fortune.com/2024/07/15/california-ai-bill-sb-1047-fierce-debate-regulation-safety/">这是 AI 界的“鲨鱼帮对阵喷气机帮”——欢迎来到加州 AI 安全法案的争端</a>: 这位支持备受争议的 SB-1047 AI 法案的加州州参议员表示，他没预料到会遭到硅谷重量级人物的反对。</li><li><a href="https://archive.is/e5n9A">加州 AI 法案 SB-1047 引发关于监管的激烈辩论……</a>: 未找到描述内容。
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1261474726690160691)** (58 messages🔥🔥): 

> - `LangChain JS Usage`
> - `Gemini Pro vs API`
> - `RAG Errors`
> - `Using Base64 with APIs`
> - `OpenAI Embedding Models` 


- **理解 LangChain JS：invoke、stream 和 streamEvents**：一位用户询问了 LangChain JS 中 `invoke`、`stream` 和 `streamEvents` 之间的区别，想知道在节点主要涉及工具调用的 langgraph 中应该使用哪一个来实现流式输出。
   - 作为回应，有人建议使用 Agent 来执行各种操作，如数据收集和 API 调用。
- **Gemini Pro 的 Base64 输入问题**：一位用户在测试 Gemini Pro API 的 Base64 输入时遇到了“无效输入”错误，并寻求帮助，因为文档中只提到了 File API 上传，而没有指定 Base64 格式。
- **从 ToolCall 迁移到 OpenAIToolCall**：用户讨论了 `ToolCall` 的弃用以及改用 `OpenAIToolCall` 的必要性，包括添加 `index` 属性。
   - 一位用户寻求关于更新 LangChain 包以及如何处理“auto”模式下意外的默认工具调用的指导。
- **HuggingFace 模型在聊天机器人中的幻觉问题**：一位用户在使用 HuggingFace 模型时遇到了幻觉问题，LLM 生成了随机的问题/答案对。
   - 建议包括切换到 OpenAI 模型或 FireworksAI 模型，并指出重复惩罚（repetition penalties）对于微调后的 Llama 模型效果不佳。
- **最优 OpenAI Embedding 模型**：有人提问关于最佳的 OpenAI Embedding 模型，得到的建议是 `text-embedding-ada-002`，它是 LangChain 中的默认模型。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://v02.api.js.langchain.com/interfaces/langchain_core_messages.ToolCall.html#Deprecated>)">ToolCall | LangChain.js - v0.2.9</a>：未找到描述</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/tool_calling/#passing-tools-to-llms>).">如何使用聊天模型调用工具 | 🦜️🔗 Langchain</a>：本指南假设你已熟悉以下概念：</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/few_shot_examples_chat/#create-prompt-template>)">如何在聊天模型中使用 few shot 示例 | 🦜️🔗 LangChain</a>：本指南假设你已熟悉以下概念：</li><li><a href="https://github.com/langchain-ai/langchain/issues/17737>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号来为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/9270>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号来为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://python.langchain.com/v0.2/docs/integrations/vectorstores/hippo/#declaring-the-embedding-model>)">Hippo | 🦜️🔗 LangChain</a>：Transwarp Hippo 是一款企业级云原生分布式向量数据库，支持海量向量数据集的存储、检索和管理。它高效地解决了诸如...
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1261691234246852708)** (1 messages): 

> - `LLM Scraper`
> - `code generation`
> - `local models`
> - `GitHub project release`
> - `webpage scraping` 


- **LLM Scraper 发布代码生成支持**：[LLM Scraper](https://github.com/mishushakov/llm-scraper) 现在包含代码生成支持，允许用户使用 **本地模型** 将任何网页转换为结构化数据。
   - 这一新功能旨在增强工具的功能，可在该项目的 [GitHub 页面](https://github.com/mishushakov/llm-scraper)上查看详细信息和更新。
- **使用 LLM 将任何网页转换为结构化数据**：[LLM Scraper](https://github.com/mishushakov/llm-scraper) 使用户能够利用大语言模型（LLM）将任何网页转换为结构化数据。
   - GitHub 仓库提供了关于如何使用这一强大工具的概述和贡献文档。



**提及的链接**：<a href="https://github.com/mishushakov/llm-scraper">GitHub - mishushakov/llm-scraper: Turn any webpage into structured data using LLMs</a>：使用 LLM 将任何网页转换为结构化数据。通过在 GitHub 上创建账号来为 mishushakov/llm-scraper 的开发做出贡献。

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1261428463169179748)** (10 条消息🔥): 

> - `entity deduplication` (实体去重)
> - `LlamaCloud`
> - `GPT-4o for financial reports` (GPT-4o 用于财务报告)
> - `multi-agent workflows with Redis` (基于 Redis 的多 Agent 工作流)
> - `advanced RAG guide` (高级 RAG 指南)


- **使用 Neo4j Cypher 代码片段进行实体去重**：由 @tb_tomaz 和 @neo4j 的其他成员编写的一个非常酷的 [Cypher 代码片段](https://t.co/dAV2QuAoZH)，结合了文本 Embedding 和词汇分析来执行 **entity deduplication**。
- **LlamaCloud 简化数据流水线管理**：**LlamaCloud** 现在允许你在一个地方管理所有的数据流水线，新增的 [团队功能](https://t.co/F73Spljg0a) 使多个用户能够集中查看所有项目。
- **使用 GPT-4o 解析财务报告**：LlamaParse 使用像 **GPT-4o** 这样的多模态模型，从复杂的财务报告中轻松提取文本、图表和表格，而传统的基于文本的解析器很难处理这些内容。
- **集成 Redis 的多 Agent 工作流**：感谢 @0xthierry，你现在可以使用 **Redis Queue** 作为中央消息代理来构建生产级的 Agent 系统，以协调多 Agent 工作流。
   - 这种设置允许 Agent 服务通过中央消息队列进行通信，显著简化了架构。
- **开始使用高级 RAG 工作流**：来自 @kingzzm 的精彩指南，教你如何使用 **LlamaIndex query pipelines** 构建具有完全可见性的高级 RAG 和 Agent 模块。
   - 该分步指南涵盖了从基础到高级设置的所有内容，为 AI 工程师提供了必备知识。



**提到的链接**: <a href="https://t.co/ruxdlhZOuK">blogs/llm/llama_index_neo4j_custom_retriever.ipynb at master · tomasonjo/blogs</a>: 支持我在 https://bratanic-tomaz.medium.com/ 上的图数据科学博客文章的 Jupyter Notebooks - tomasonjo/blogs

  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1261813014160478268)** (18 条消息🔥): 

> - `LlamaIndex KG 节点去重`
> - `结合 SQL 和 PDF Embeddings`
> - `在 FastAPI 中处理聊天历史`
> - `数据分块以获得更好的 Embeddings`
> - `结合 NebulaGraphStore 使用 KnowledgeGraphIndex` 


- **LlamaIndex KG 节点去重**：一位成员分享了一个 [YouTube 视频](https://youtu.be/vMz0icWZd5A) 和一篇 [Medium 文章](https://medium.com/@rajib76.gcp/entity-de-duplication-llamaindex-approach-0b97d2950a9f)，解释了在 LlamaIndex Knowledge Graph 中对节点进行去重的过程。
   - 该视频提供了有关技术方法的详细见解，Rajib 强调了知识建模对于使非结构化数据具备 GenAI 就绪性的重要性。
- **在 LlamaIndex 中结合 SQL 和 PDF Embeddings**：一位用户询问如何按照 LlamaIndex 文档中的 [示例](https://docs.llamaindex.ai/en/stable/examples/query_engine/SQLJoinQueryEngine/)，将使用 Manticore 搜索索引的 MySQL 数据库与作为 Embeddings 的 PDF 文档结合起来。
   - 该用户在使用 `NLSQLTableQueryEngine` 时遇到了问题，因为 Manticore 查询与 MySQL 不同，正在寻求处理此问题的最佳方法。
- **在 FastAPI 中使用 LlamaIndex 处理聊天历史**：讨论了在多用户 FastAPI 后端使用 LlamaIndex 管理聊天历史的最佳实践，权衡了存储 Chat Engines 字典或为每次交互维护聊天历史的选项。
   - 共识倾向于仅管理聊天历史，可能使用简单的 Chat Store。
- **更小的分块大小可增强 Embeddings 效果**：在 LlamaIndex 中，将数据切分为更小的分块有助于提高 Embeddings 的精确度，因为较小的分块大小提供了更细粒度的细节。
   - 根据 [LlamaIndex 文档](https://docs.llamaindex.ai/en/latest/optimizing/basic_strategies/basic_strategies/#chunk-sizes)，提供的配置示例为：将 `Settings.chunk_size` 设置为 512，`overlap` 设置为 50，并调整 `similarity_top_k` 为 4，以获得更好的检索准确度。
- **KnowledgeGraphIndex 中 NebulaGraphStore 的问题**：一位成员在运行 `KnowledgeGraphIndex` 的 [NebulaGraph 示例 Notebook](https://github.com/run-llama/llama_index/blob/0250d337a2cd68d724c32753c9187d7683d9822f/docs/docs/examples/query_engine/knowledge_graph_query_engine.ipynb) 时遇到问题，如 [GitHub Issue #14748](https://github.com/run-llama/llama_index/issues/14748) 所述。
   - 出现了错误 `KnowledgeGraphIndex._build_index_from_nodes() got an unexpected keyword argument 'space_name'`，他们正在寻求解决建议。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://medium.com/@rajib76.gcp/entity-de-duplication-llamaindex-approach-0b97d2950a9f">实体去重 | LlamaIndex 方法</a>：LlamaIndex 发布了一种智能方法，用于对语言模型创建的知识图谱实体进行去重。我研究了他们的方法……</li><li><a href="https://youtu.be/vMz0icWZd5A">LlamaIndex KG | 节点去重。</a>：在此录音中，我详细解释了 LlamaIndex 在创建知识图谱后如何进行节点去重。代码：https://github.com/raji...</li><li><a href="https://docs.llamaindex.ai/en/latest/optimizing/basic_strategies/basic_strategies/#chunk-sizes>).">基础策略 - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/issues/14748">[Bug]: KnowledgeGraphIndex._build_index_from_nodes() 收到意外的关键字参数 'space_name' · Issue #14748 · run-llama/llama_index</a>：Bug 描述：我正尝试运行这个 NebulaGraph 示例。运行此单元格：from llama_index.core import KnowledgeGraphIndex kg_index = KnowledgeGraphIndex.from_documents( documents, storage_co...</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/query_engine/SQLJoinQueryEngine/.">SQL Join 查询引擎 - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/query_engine/pdf_tables/recursive_retriever/?h=recursive+query+pandas">递归检索器 + 查询引擎演示 - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1261407611778105456)** (13 messages🔥): 

> - `OpenInterpreter GUI Integration` (OpenInterpreter GUI 集成)
> - `OpenAI OS Rumors` (OpenAI OS 传闻)
> - `Phi-3.1 Model Evaluation` (Phi-3.1 模型评估)
> - `Internlm2 Valuation` (Internlm2 评估)
> - `System Architecture Documentation Request` (系统架构文档请求)


- **OpenInterpreter 完全集成至 GUI**：[OpenInterpreter](https://github.com/jbexta/AgentPilot) 已由一名成员完全集成到 GUI 中，具有分支聊天、可编辑消息、代码自动运行和聊天保存功能。
   - 成员们对该项目表示兴奋，其他人则请求提供视频教程或演示，以便更好地了解其功能。
- **OpenAI 正在构建 OS 的传闻**：一条 [推文](https://x.com/apples_jimmy/status/1805373587127402883) 暗示 Sam Altman 和 OpenAI 可能正在开发自己的 OS 和通信工具，并引用了越来越多的证据。
   - 这一进展是在一个月前发布招聘职位后出现的，引发了社区的热烈讨论。
- **Phi-3.1 模型评估**：Techfren 发起了关于 Phi-3.1 模型性能的讨论，指出其尺寸和能力非常有前景。
   - 成员 twodogseeds 分享了见解，表示 Phi-3.1 提供的功能超出了要求，但有时在准确遵循 `<INST>` 方面存在困难。
- **Internlm2 在 Raspi5 上表现出色**：Twodogseeds 指出 'Internlm2 smashed' 受到关注，强调了其在 Raspi5 上的性能。
   - 他们提到了 multi-shot 和 smash 模式在边缘设备（尤其是 IoT 应用）中的潜力。
- **系统架构文档请求**：一名成员询问是否有解释 Open Interpreter 系统级架构和分解的可用文档。
   - 响应中没有分享具体的文档，表明可能存在缺口或需要社区贡献资源。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/apples_jimmy/status/1805373587127402883">Jimmy Apples 🍎/acc (@apples_jimmy) 的推文</a>：我真的很无聊，所以以防你错过，大约一个月前他们正在招聘这个职位。引用 Chubby♨️ (@kimmonismus) 的话，传闻似乎证实了 Sam Altman 和 OpenAI 正在构建他们的……</li><li><a href="https://github.com/jbexta/AgentPilot">GitHub - jbexta/AgentPilot: 用于无缝交互和管理 AI 工作流的通用 GUI</a>：用于无缝交互和管理 AI 工作流的通用 GUI - jbexta/AgentPilot
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1261603980904103937)** (3 messages): 

> - `Meta Ray-Ban Jailbreak` (Meta Ray-Ban 越狱)
> - `Installing O1 on Linux` (在 Linux 上安装 O1)
> - `'Interpreter' Not Defined Error` ('Interpreter' 未定义错误)


- **对 Meta Ray-Ban 越狱的兴趣**：一名成员对越狱 **Meta Ray-Ban** 的可能性表示兴奋。
   - 他们表示：*“那太棒了，如果你真的越狱了 Meta Ray-Ban，请告诉我。”*
- **O1 Linux 安装补丁**：一名成员分享了在 Linux 上安装 **O1** 的步骤，提到了 **Poetry** 中一个必要的补丁。
   - 他们需要移除一个依赖项才能完成安装。
- **'Interpreter' 未定义错误**：一名成员在使用 O1 时遇到了提示 'interpreter' 未定义的错误消息。
   - 他们检查了服务器代码但未能找到解决方案，表达了他们的沮丧。


  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1261630731222781952)** (1 messages): 

> - `LLM agent`
> - `Adding agents in LLMs` (在 LLM 中添加 Agent)
> - `Modular components in chat pipelines` (聊天流水线中的模块化组件)
> - `Processing information using agents` (使用 Agent 处理信息)
> - `Interacting with external APIs` (与外部 API 交互)


- **LLM Agent 工作原理详解**：一位用户分享了一份[详细指南](https://nik-hil.hashnode.dev/how-to-add-agents-in-large-language-models-a-detailed-guide)，解释了如何在 Large Language Models (LLM) 中添加 Agent，重点介绍了它们的模块化性质以及在 Chat 流水线中的角色。
   - 该指南描述了处理步骤：**输入处理**、**LLM 解释**，以及根据对话需求使用 JSON 输出调用 Agent。
- **模块化组件增强 LLM 聊天流水线**：该详细指南强调，LLM 中的 Agent 作为模块化组件，执行诸如**获取数据**、**处理信息**以及**与外部 API 交互**等任务。
   - 通过利用 LLM 的 JSON 输出能力，这些 Agent 可以无缝集成到对话流中，以满足特定需求。



**提及的链接**：<a href="https://nik-hil.hashnode.dev/how-to-add-agents-in-large-language-models-a-detailed-guide">在 Large Language Models 中添加 Agent 指南</a>：在这份详细指南中，了解如何使用 JSON 输出在 Large Language Models 中添加 Agent，以构建灵活、可扩展的聊天流水线。

  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/1261414721802862864)** (2 条消息): 

> - `OpenAI API Key 请求` 


- **针对聊天机器人项目的 OpenAI API Key 请求**：一名成员请求一个 OpenAI 的 API Key，用于一个聊天机器人项目。
   - 他们提到需要该 Key 来为该项目创建一个教程。
- **寻求未使用的 OpenAI API Key**：同一名成员询问是否有人可以分享未使用的 OpenAI API Key。
   - 他们说明该 Key 仅用于教程演示。


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/)** (1 条消息): 

healthymonkey: 我听说大约是一年。我真的很喜欢在 Modal 上获取 H100 有多容易，哈哈
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1261751924890402909)** (1 条消息): 

> - `额度拒绝` 


- **截止日期后无法发放额度**：在截止日期前尝试联系未果，导致额度申请被拒绝。
   - 未提供更多细节。
- **无进一步回复**：在指定的截止日期内未收到任何回复。


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1261727386261651476)** (1 条消息): 

> - `训练 Loss 问题`
> - `模板正确性`
> - `Meta 的模板` 


- **训练 Loss 拒绝下降**：一名成员在使用特定设置时遇到训练 Loss 不下降的问题，这表明其方法中可能存在潜在问题。
   - 分享的 [代码片段](https://link.to/examples) 和输出结果暗示在数据集加载和 Prompt 格式化方面可能存在问题。
- **正确模板验证**：一名成员提供了一个与 [Meta 官方文档](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/) 模板相匹配的输出示例。
   - 该模板遵循 `type: input_output`，并对训练响应的片段标记为 true 或 false。



**提到的链接**：<a href="https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/">Meta Llama 3 | 模型卡片与 Prompt 格式</a>：Meta Llama 3 使用的特殊 Token。一个 Prompt 应包含一条 system 消息，可以包含多条交替的 user 和 assistant 消息，并始终以最后一条 user 消息结尾，后跟...

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1261495743223697460)** (2 条消息): 

> - `Modal 错误`
> - `Axolotl 故障排除`
> - `在 Slack 上寻求帮助` 


- **在 Slack 上寻求 Modal 错误的帮助**：一名成员提到一个不熟悉的错误，推测这可能是 **Modal** 特有的问题，并建议在 Slack 上询问。
   - *以前没见过这个错误，但我猜它是 Modal 特有的，我会去他们的 Slack 询问。*
- **在 Modal 和 Axolotl 上遇到困难**：另一名成员表示赞同，确认在 **Modal** 和 **Axolotl** 的使用上都遇到了困难。
   - *谢谢。我一直在 Modal 和 Axolotl 上挣扎。*


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langchain-langsmith](https://discord.com/channels/1238365980128706560/1242564256914870384/1262065191496192082)** (1 条消息): 

> - `Langsmith 评估`
> - `OpenAI 中的速率限制` 


- **解决 Langsmith 评估中的速率限制问题**：一名用户在使用 OpenAI 额度运行 [Langsmith 评估测试](https://link.to/example) 时遇到了每分钟 Token 速率限制（Rate limits）。
   - 他们发现调整 **max_concurrency** 参数有助于缓解该问题。
- **在实验中引入延迟**：对话的另一部分涉及寻找在实验中引入延迟的方法，以避免触发速率限制。
   - 寻求将此功能实现在现有基础脚本中的建议。


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/1261449560031756490)** (5 条消息): 

> - `OpenAI 额度过期`
> - `申请额度延期的请愿` 


- **OpenAI 额度将于 9 月 1 日过期**：经成员询问后确认，**OpenAI 额度**定于 **9 月 1 日**过期。
   - 在另一名成员指出在哪里可以找到此信息后，一名用户对此澄清表示感谢。
- **申请延长 OpenAI 额度的请愿**：一名用户幽默地请求发起一项 **延长 OpenAI 额度有效期** 的请愿。


  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1261506856359301150)** (2 条消息): 

> - `Hugging Face Profitability` (Hugging Face 盈利情况)
> - `Cambrian-1 Multimodal LLMs` (Cambrian-1 多模态 LLM)


- **Hugging Face 实现盈利**：[Hugging Face](https://analyticsindiamag.com/hugging-face-announces-profitability-with-free-and-open-source-models) 作为一个开发和共享机器学习模型的领先平台，宣布在拥有 220 名团队成员的情况下实现盈利，并同时保持平台大部分免费且开源。
   - 首席执行官 Clement Delangue 在 X 上分享道：*“这并不是我们的目标，因为我们银行里有充足的资金，但看到 @huggingface 最近实现了盈利，我感到非常兴奋。我们拥有 220 名团队成员，且平台的大部分功能（如模型托管）对社区都是免费且开源的！”*
- **Cambrian-1 多模态 LLM 发布**：采用了以视觉为中心设计的 [Cambrian-1](https://github.com/cambrian-mllm/cambrian) 系列多模态 LLM 正式推出，扩展了 AI 模型的能力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/cambrian-mllm/cambrian">GitHub - cambrian-mllm/cambrian: Cambrian-1 is a family of multimodal LLMs with a vision-centric design.</a>：Cambrian-1 是一个采用以视觉为中心设计的多模态 LLM 系列。- cambrian-mllm/cambrian</li><li><a href="https://analyticsindiamag.com/hugging-face-announces-profitability-with-free-and-open-source-models/">Hugging Face 宣布通过免费和开源模型实现盈利 – AIM</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1261816129458802832)** (1 条消息): 

> - `MagViT2 compatibility with non-RGB motion data` (MagViT2 对非 RGB 运动数据的兼容性)
> - `Motion data preprocessing` (运动数据预处理) 


- **用于非 RGB 运动数据的 MagViT2**：一位用户询问 **MagViT2** 是否可以用于非 RGB 格式的运动数据，并提到其数据格式为 24x3。
   - *消息中未提供进一步的讨论或评论。*
- **运动数据预处理技术**：成员们正在探索针对非 RGB 运动数据的各种预处理技术，以确保与现有 AI 模型的兼容性。
   - *消息中未讨论更多细节和具体的预处理方法。*


  

---



### **DiscoResearch ▷ #[disco_judge](https://discord.com/channels/1178995845727785010/1178996063537991752/1261436075466035321)** (2 条消息): 

> - `LLM Arena`
> - `Ollama models`
> - `WizardLM paper`
> - `Arena Learning methodology` 


- **介绍用于 LLM 对战的 OpenArena**：一位成员分享了 [OpenArena](https://github.com/syv-ai/OpenArena) 的发布，这是一个让两个 LLM 相互对抗并由第三个 LLM 担任裁判的平台，旨在提升数据集质量。
   - 该平台主要使用来自 **Ollama** 的模型，但也支持任何 OpenAI 兼容的端点（endpoint）。
- **OpenArena 的基础源自 WizardLM 论文**：[WizardLM 论文](https://www.microsoft.com/en-us/research/publication/arena-learning-build-data-flywheel-for-llms-post-training-via-simulated-chatbot-arena/) 介绍了 “**Arena Learning**” —— 一种用于评估 LLM 的模拟聊天机器人竞技场。
   - 该方法论包括精确的评估和一致的离线模拟，通过监督微调（supervised fine-tuning）和强化学习（reinforcement learning）来改进 LLM。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/syv-ai/OpenArena">GitHub - syv-ai/OpenArena</a>：通过在 GitHub 上创建账号来为 syv-ai/OpenArena 的开发做出贡献。</li><li><a href="https://www.microsoft.com/en-us/research/publication/arena-learning-build-data-flywheel-for-llms-post-training-via-simulated-chatbot-arena/">Arena Learning: Build Data Flywheel for LLMs Post-training via Simulated Chatbot Arena - Microsoft Research</a>：Arena Learning：通过模拟聊天机器人竞技场构建 LLM 训练后的数据飞轮
</li>
</ul>

</div>
  

---



---



---



---



---



---



---



{% else %}


> 完整的各频道详细分析已因邮件长度而截断。
> 
> 如果您想查看完整分析，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}