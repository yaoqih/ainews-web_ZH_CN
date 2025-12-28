---
companies:
- google
- openai
- hugging-face
- togethercompute
- lmsys
date: '2025-11-20T05:44:39.731046Z'
description: '**谷歌（Google）**推出了 **Gemini 3 Pro Image (Nano Banana Pro)**，这是一款集成了谷歌搜索溯源（grounding）、多图合成及细粒度视觉控制功能的下一代
  AI 图像生成与编辑模型。其定价为：2K 分辨率图像每张 0.134 美元，4K 分辨率图像每张 0.24 美元。该模型显著提升了文本渲染能力，错误率从前代产品的
  56% 降至 8%，并包含用于来源验证的 SynthID 水印检查功能。用户可通过 Gemini 应用、API、LM Arena、Hugging Face Spaces、Together
  AI 和 Flow 使用该模型。


  与此同时，**OpenAI** 分享了 **GPT-5** 加速科学研究的早期实验成果，包括在数学、物理、生物和材料科学领域对此前未解决问题的证明。*“GPT-5
  加速了数学、物理、生物和材料领域的研究任务；在其中 4 个领域，它帮助找到了此前未解决问题的证明。”*'
id: MjAyNS0x
models:
- gemini-3-pro
- gpt-5
people:
- jeffdean
- kevinweil
- demishassabis
title: Nano Banana Pro (Gemini Image Pro) 解决了图像内文字生成、信息图表生成、2K-4K 分辨率以及 Google 搜索溯源（Grounding）等问题。
topics:
- image-generation
- text-rendering
- model-provenance
- scientific-research
- proof-assistance
- multimodal-integration
- api-access
- fine-tuning
---

**AIE CODE Day 1.**

> 2025/11/19-11/20 的 AI 新闻。我们为您检查了 12 个 subreddits、544 个 Twitters 和 24 个 Discords（205 个频道，10448 条消息）。预计节省阅读时间（以 200wpm 计算）：754 分钟。我们的新网站现已上线，支持全元数据搜索，并以精美的 vibe coded 方式展示所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻分类，并在 @smol_ai 上向我们提供反馈！

随着 [AIE CODE Day 1](https://www.youtube.com/watch?v=cMSprbJ95jg&t=21s) 圆满结束，产品的发布并未停止。虽然 [AI2 Olmo 3](https://x.com/natolambert/status/1991508141687861479) 因推动美国开源模型的发展而非常值得特别提及，但今天的重头戏是“[Nano Banana Pro](https://blog.google/technology/ai/nano-banana-pro/)”（[官方提示词技巧](https://blog.google/products/gemini/prompting-tips-nano-banana-pro/)、[构建技巧](https://blog.google/technology/developers/gemini-3-pro-image-developers/)、[演示应用](https://aistudio.google.com/apps/bundled/info_genius?showPreview=true&showAssistant=true)）——它是[原始 Nano Banana (Flash)](https://news.smol.ai/issues/25-08-26-nano-banana) 的大哥，而且……好吧……这是由 NBP 呈现的今日 AI 新闻摘要输出：


![详细介绍 Nano Banana Pro 功能和能力的信息图，这是一款由 Gem 驱动的下一代 AI 图像生成和编辑工具](https://resend-attachments.s3.amazonaws.com/BsQTOgYCnh4SOcv)


> 嘿 Nano Banana，把这个重做成卡通、波普艺术风格的信息图，采用纵向布局，重新排列信息以解释人们应该了解的内容：


![一张彩色、漫画风格的信息图，展示了 Nano Banana Pro 的能力，这是一款具有超级英雄特质的新型 AI 图像生成模型](https://resend-attachments.s3.amazonaws.com/BEMOVHQKUcb64a3)


如果你还没从这些例子中看出来，图像中复杂、高度详细的文本渲染已经……解决了。


![一只土拨鼠坐在木制字母中间，字母拼出了绕口令 "How much wood would a woodchuck chuck if a woodchuck coul"](https://resend-attachments.s3.amazonaws.com/N5nHzttEs70ljCs)


以下是各模型的定价对比：


![Gemini 3 Pro Image 和 Gemini 2.5 Flash Image 的对比图表，显示了价格、速度、质量](https://resend-attachments.s3.amazonaws.com/D2Qu9WEcekm2RSq)


---

# AI Twitter 综述

**Gemini 3 和 “Nano Banana Pro” Image：搜索增强、4K 输出和更强的文本渲染**

- **Gemini 3 Pro Image (又名 Nano Banana Pro)**：Google 在 Gemini API 和 AI Studio 中推出了其新的图像生成/编辑模型，集成了 Google Search 增强 (grounding)、多图合成和细粒度视觉控制。亮点包括：
    - 定价与功能：2K 图像每张 0.134 美元，4K 图像每张 0.24 美元；支持多达 14 张参考图；10 种纵横比；精确的文本渲染；以及通过 Search 实现的股票/天气/数据增强 ([定价/详情](https://twitter.com/_philschmid/status/1991537712420020225), [发布](https://twitter.com/GoogleAIStudio/status/1991537543989588445))。
    - 可用性：Gemini 应用和 API、LM Arena (对战)、面向 PRO 订阅者的 Hugging Face Spaces、面向生产环境的 Together AI，以及面向 Ultra 订阅者的 Flow 额外控制功能 ([Arena 加入](https://twitter.com/arena/status/1991540746114199960), [HF Spaces](https://twitter.com/multimodalart/status/1991549140627775511), [Together](https://twitter.com/togethercompute/status/1991614379394203973), [Flow 控制](https://twitter.com/demishassabis/status/1991662935983419424))。
    - 早期结果：演示展示了准确的信息图创建、图表标注、多图编辑以及将推文“视觉转发”为信息图；社区对比测试表明，在文本和布局方面优于 GPT-Image 1 ([示例](https://twitter.com/simonw/status/1991545654901133797), [对比 GPT-Image 1](https://twitter.com/scaling01/status/1991546597013160290))。
- **质量与溯源**：Google 表示，渲染文本的错误率从 56% (Gemini 2.5 Flash Image/Nano Banana) 降至 8% (Gemini 3 Pro Image/Nano Banana Pro) ([Jeff Dean](https://twitter.com/JeffDean/status/1991573065994744091))。Google 还在 Gemini 中推出了 SynthID 水印检查：上传图片并询问是否由 Google AI 创建/编辑以获取溯源信号 ([SynthID](https://twitter.com/Google/status/1991552943372578850), [操作指南](https://twitter.com/Google/status/1991552945754612118))。注意：在强劲的早期采用浪潮中，用户也发现了一些局限性（例如棋盘编辑中的逻辑错误）([批评](https://twitter.com/scaling01/status/1991553936202063937))。

**OpenAI：GPT-5 辅助的科学与产品更新**

- **GPT‑5 用于科学（案例研究、证明）**：OpenAI 分享了 13 个早期实验，展示了 GPT‑5 如何加速数学、物理、生物和材料科学的研究任务；在其中 4 个实验中，它帮助找到了先前未解决问题的证明。详情请参阅博客、技术报告以及与研究人员的播客讨论（[概述](https://twitter.com/kevinweil/status/1991567552640872806), [博客](https://twitter.com/kevinweil/status/1991567567694229686), [arXiv 链接](https://twitter.com/SebastienBubeck/status/1991679019411206519), [OpenAI 视频](https://twitter.com/OpenAI/status/1991569987933458814), [论文推文串](https://twitter.com/SebastienBubeck/status/1991568186840686915)）。团队将其定位为一个真实的快照，展示了前沿模型在当今实际工作流中能做什么和不能做什么（[OpenAI 帖子](https://twitter.com/OpenAI/status/1991570422148788612)）。
- **ChatGPT 功能**：群聊功能正向全球 Free/Go/Plus/Pro 层级用户推出；OpenAI 还通过 Throughline 扩展了 ChatGPT 中的本地化危机求助热线；此外，Realtime API 现在支持为 SIP 会话发送 DTMF 电话按键；Instant Checkout 正向 Shopify 商家推出（[群聊](https://twitter.com/OpenAI/status/1991556363420594270), [求助热线](https://twitter.com/OpenAI/status/1991634046624116784), [DTMF](https://twitter.com/pbbakkum/status/1991643527072428292), [Instant Checkout](https://twitter.com/OpenAI/status/1991646997322035520)）。

**AI2 的 Olmo 3（完全开源）和 RL 基础设施加速**

- **开源发布 + 架构细节**：AI2 的 Olmo 3 以完全开放的栈（代码、数据、训练配方、checkpoints；Apache‑2.0 协议）发布，其中 32B Think 变体针对长思维链（CoT）和复杂推理进行了优化。架构保留了 post‑norm（基于 Olmo 2 的稳定性发现），在 7B 版本中使用 sliding‑window attention 以减少 KV cache，在 32B 版本中转向 GQA；参数比例调整接近 Qwen3，但在 FFN 扩展缩放等方面做了改动（[发布反响](https://twitter.com/ClementDelangue/status/1991609311920026027), [架构深度解析](https://twitter.com/rasbt/status/1991656199394050380), [HuggingFace 列表](https://twitter.com/HuggingPapers/status/1991548898436083990)）。
- **RL 基础设施与评估严谨性**：通过 continuous batching、in‑flight updates、主动采样和多线程改进，OlmoRL 基础设施的实验速度比 Olmo 2 快了约 4 倍。团队还强调了去污染评估（例如，伪奖励测试显示在随机奖励下没有提升），解决了先前设置中的污染担忧（[基础设施](https://twitter.com/finbarrtimbers/status/1991546419875115460), [评估严谨性](https://twitter.com/mnoukhov/status/1991576437246292434)）。强大的社区支持凸显了该发布的透明度和完整性（[Percy Liang](https://twitter.com/percyliang/status/1991545594482159619)）。

**Agent、评估与部署经验**

- **真实世界编码基准与生产级 Agent 的 RL**：Cline 宣布了 cline‑bench，这是一个价值 100 万美元的开源基准，由 OSS 仓库中真实的失败 Agent 编码任务构建，打包为带有真实仓库快照、prompts 和已发布测试的容器化 RL 环境——兼容 Harbor 和现代评估栈。实验室和 OSS 开发者可以在相同的现实任务上进行评估和训练。OpenAI 评估负责人及其他专家对该倡议表示支持（[公告](https://twitter.com/pashmerepat/status/1991596028735184899), [Cline](https://twitter.com/cline/status/1991612268220752130), [支持](https://twitter.com/shyamalanadkat/status/1991603916115775932)）。另外，Eval Protocol 已开源，可直接在生产级 Agent 上运行 RL，支持 TRL, rLLM, OpenEnv 和专有训练器（如 OpenAI RFT）（[框架](https://twitter.com/the_bunny_chen/status/1991559599347192193)）。
- **企业部署模式**：Bloomberg 的基础设施演讲强调，大规模 Agent 的 ROI 不仅取决于模型能力，还取决于标准化、验证和治理——例如，通过集中式网关/发现来管理重复的 MCP 服务器，使用补丁生成 Agent 改变维护经济学，以及使用事件响应 Agent 来对抗人类的锚定偏差。文化转变（培训流水线、领导层技能提升）与技术同样重要。Box 和 LangChain 补充了关于协作式 Agent 和中间件（如 tool‑call budgets）的见解，以稳定生产环境行为（[Bloomberg 演讲](https://twitter.com/TechAtBloomberg/status/1991563444374389018), [总结](https://twitter.com/TheTuringPost/status/1991596158523961633), [Box x LangChain](https://twitter.com/Box/status/1991582582920839354), [LangChain 中间件](https://twitter.com/bromann/status/1991544566563189022)）。

**浏览器与模型/平台更新**

- **Perplexity Comet (移动 Agent 浏览器)**：Comet 在 Android 上发布，具备语音优先浏览、可见的 Agent 动作以及应用内购买流程；iOS 版将在“几周内”推出。Perplexity Pro/Max 现在包含 Kimi‑K2 Thinking 和 Gemini 3 Pro，Grok 4.1 也即将推出 ([Android 发布](https://twitter.com/perplexity_ai/status/1991567491404034269), [语音/氛围浏览](https://twitter.com/AravSrinivas/status/1991567787408650416), [iOS 即将推出](https://twitter.com/AravSrinivas/status/1991674701702479957), [模型阵容](https://twitter.com/perplexity_ai/status/1991614227950498236), [更多](https://twitter.com/AravSrinivas/status/1991619527638151665))。
- **工具与基础设施**：来自 Arctic LST 的 Ulysses Sequence Parallelism 已合并至 Hugging Face Accelerate（长序列训练），VS Code 发布了新的安全/透明度功能（包括 Linux 策略 JSON），GitHub Copilot 增加了组织范围内的 BYOK，Weaviate + Dify 提供了更快的 RAG 集成，W&B Weave Playground 添加了 Gemini 3 和 GPT‑5.1 用于基于 Trace 的评估 ([Accelerate](https://twitter.com/StasBekman/status/1991561577007611907), [VS Code](https://twitter.com/code/status/1991549116149592330), [Copilot BYOK](https://twitter.com/pierceboggan/status/1991612120312770600), [Weaviate x Dify](https://twitter.com/weaviate_io/status/1991539631259591085), [Weave](https://twitter.com/weave_wb/status/1991601539728003200))。

**视觉与可解释性**

- **SAM 3 和 SAM 3D (Meta)**：针对人物和复杂环境的统一检测/追踪以及单图 3D 重建，具有强大的数据引擎增益（400 万个短语，5200 万个掩码）和宽松的开源条款（允许商业使用和修改所有权） ([SAM 3](https://twitter.com/AIatMeta/status/1991538570402934980), [SAM 3D](https://twitter.com/AIatMeta/status/1991605451809513685), [数据引擎](https://twitter.com/AIatMeta/status/1991640180185317644), [许可证说明](https://twitter.com/skalskip92/status/1991626755782877234))。
- **神经元级电路与 VLM 自我改进**：TransluceAI 认为 MLP 神经元可以支持稀疏、忠实的电路——重新引发了对神经元级可解释性的兴趣。另外，VisPlay 提出了针对 VLM 的自进化 RL，利用无标签图像数据，在视觉推理上达到 SOTA 并减少了幻觉 ([可解释性](https://twitter.com/TransluceAI/status/1991582415891099793), [VisPlay](https://twitter.com/HuggingPapers/status/1991539261175394578))。额外福利：一个极简的 ViT 示例（ImageNet‑10），仅用约 150 行代码在单张 GPU 上达到 91% 的 top‑1 准确率，展示了视觉学习器的简洁基准 ([ViT 极简版](https://twitter.com/randall_balestr/status/1991546816685568387))。

**热门推文（按互动量排序）**

- Grok 的极端谄媚测试走红；提示词框架（Prompt framing）会强烈改变输出结果 ([示例线程](https://twitter.com/romanhelmetguy/status/1991545583686021480))。
- Android 宣布 Quick Share 与 Apple 的 AirDrop 兼容，从 Pixel 10 开始支持跨操作系统文件传输 ([公告](https://twitter.com/Android/status/1991552333063524573))。
- Nano Banana Pro “独树一帜”的演示刷屏了 ([单样本示例](https://twitter.com/cto_junior/status/1991564259516702997))；Sundar 以一种神秘的姿态表达了认可 ([懂的都懂](https://twitter.com/sundarpichai/status/1991613220969423272))。
- Google 展示了 Nano Banana Pro 的社区样本 ([亮点](https://twitter.com/GeminiApp/status/1991570302720163988))，并在 Gemini 中推出了 SynthID 检查 ([溯源](https://twitter.com/Google/status/1991552943372578850))。
- OpenAI 的 “AI for Science” 论文引发了关于模型辅助证明和发现的热烈讨论 ([论文线程](https://twitter.com/SebastienBubeck/status/1991568186840686915), [视频](https://twitter.com/OpenAI/status/1991569987933458814))。
- Perplexity 在 Android 上发布了 Comet，定位为一款具备语音 UI 和透明动作日志的移动 Agent 浏览器 ([发布](https://twitter.com/perplexity_ai/status/1991567491404034269))。

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Olmo 3 发布与资源

- [**Ai2 刚刚发布了 Olmo 3，这是一个领先的、专为推理、聊天和工具使用而构建的全开源 LM 套件**](https://www.reddit.com/r/LocalLLaMA/comments/1p24aet/ai2_just_announced_olmo_3_a_leading_fully_open_lm/) (热度: 681): **AI2 宣布发布 Olmo 3，这是一个全开源的大语言模型 (LM) 套件，旨在用于推理、聊天和工具使用。该模型可在 [AI2 Playground](https://playground.allenai.org/) 中进行实验，并可从 [Hugging Face](https://huggingface.co/collections/allenai/olmo-3-68e80f043cc0d3c867e7efc6) 下载。详细介绍模型架构和能力的 [技术报告在此](https://allenai.org/papers/olmo3)。Olmo 3 被定位为领先的开源模型，在性能和可用性方面有可能超越现有的权重开放 (open-weight) 模型。** 评论者对 Olmo 3 超越当前权重开放模型的潜力表示乐观，并指出了其快速的开发进度和开源特性。此外，人们对具有门控注意力 (gated attention) 机制的模型（如 Qwen3-Next）也表现出兴趣，因为它们具有较高的效率和更广泛的可访问性潜力。
    - Olmo 3 的发布意义重大，因为它代表了一个全开源的语言模型套件，在性能上已经赶上了其他权重开放模型。该模型的开源性质允许任何拥有资源的人从头开始构建它，这是开源 AI 发展迈出的重要一步。社区对未来的迭代（如 Olmo-4）持乐观态度，认为其有可能超越目前同等规模的权重开放模型。
    - 讨论中提到了使用带有门控注意力的混合专家模型 (MoE)（如 Qwen3-Next）的潜在好处，尽管架构复杂，但其训练成本更低。Qwen3-30b 模型被强调为在普通硬件上最可用的模型，人们对开发全开源的同类模型很感兴趣，因为目前的密集 (dense) 模型需要像 3090 这样的高端硬件才能高效运行。
    - Olmo 3 的发布包含了多个不同训练阶段的模型 Checkpoints，社区对其透明度和研究价值表示赞赏。Hugging Face 页面上的表格展示了从基础模型到通过 SFT 和 DPO 等技术生成的最终模型，并以 RLVR 达到顶峰的过程。然而，有人注意到缺少 Olmo3 32B Think 版本的 gguf 文件，一些用户正在寻找该文件。

### 2. NVIDIA Jetson Spark 集群搭建

- [**Spark 集群！**](https://www.reddit.com/r/LocalLLaMA/comments/1p1u9gv/spark_cluster/) (热度: 459): **图片展示了一个使用六台 NVIDIA Jetson 设备组成的个人开发环境集群，这些设备通常用于边缘计算和 AI 开发。用户正在利用此设置进行 NCCL/NVIDIA 开发，这表明其重点是优化 GPU 之间的通信，可能用于机器学习或 AI 任务。该设置旨在部署到更大的 B300 集群之前进行开发，展示了从小型硬件环境扩展到大型硬件环境的工作流。Jetson 设备并非为了追求极致性能，而是作为一个开发平台，突显了它们在扩大规模前进行原型设计和测试的通用性。** 一位评论者表达了对该设置的羡慕和兴趣，并指出了此类设备的高昂成本以及在预配置环境之外使用 PyTorch/CUDA 的挑战。另一位评论者对设备的网络配置感到好奇，表现出对技术细节的关注。
    - Accomplished_Ad9530 询问了 Spark 集群的网络设置，这对于理解节点间的数据流和通信效率至关重要。此类集群中的网络通常涉及 InfiniBand 等高速互连技术，以最小化延迟并最大化吞吐量，这对于分布式计算任务至关重要。
    - PhilosopherSuperb149 提到了在预配置容器之外使用 PyTorch/CUDA 的问题，强调了在偏离供应商提供的环境时，维持兼容性和性能所面临的共同挑战。这表明虽然硬件功能强大，但软件生态系统的支持可能是一个限制因素。
    - LengthinessOk5482 将 DGX Sparks 与 Tenstorrent GPU 进行了比较，重点关注可扩展性和软件易用性。评论指出，虽然 Tenstorrent 硬件可能很有吸引力，但其软件栈被认为难以管理，这可能是有效部署和扩展的重大障碍。

## 技术性较低的 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Nano Banana Pro 与 Gemini 3 Pro 图像生成

- [**Nano Banana Pro 可以生成 4k 图像**](https://www.reddit.com/r/singularity/comments/1p1peab/nano_banana_pro_can_produce_4k_images/) (热度: 1101): **Nano Banana Pro 是一款新型生成式模型，能够生成** `4k resolution` **图像，展示了在图像连贯性和准确性方面的重大进步。用户注意到，该模型的输出（特别是在信息图表方面）比以前的生成式模型更具连贯性，拼写错误或字符幻觉（hallucinations）更少。这表明该模型在处理复杂视觉数据以及保持高分辨率输出一致性方面的能力有所提升。** 评论者对该模型生成连贯且地理上有趣的地图，以及错误极少的信息图表的能力印象深刻，这标志着生成式模型能力的飞跃。
    - Jebby_Bush 强调了 Nano Banana Pro 在生成信息图表能力上的显著改进，指出它通常能避免拼写错误或字符幻觉等常见问题，尽管在步骤序列中存在一个小错误（缺少第 3 步）。这表明该模型在文本生成能力方面取得了显著进步，而这通常是 AI 模型的挑战所在。
    - coylter 指出了 4k 图像质量声明中的差异，指出生成的图像模糊，并非真正的 4k。这引发了对 Nano Banana Pro 生成图像实际分辨率和质量的质疑，表明该模型的图像生成能力仍有改进空间。
- [**Gemini 3 Pro Image – Nano Banana Pro**](https://www.reddit.com/r/singularity/comments/1p25ebg/gemini_3_pro_image_nano_banana_pro/) (热度: 667): **Gemini 3 Pro Image 是来自 Google DeepMind 的一款新模型，推动了 AI 驱动的图像创建和编辑技术。它是 Gemini 生态系统的一部分，该生态系统包括图像、音乐和视频的生成式模型。该模型以其开放性和鲁棒性著称，表明 AI 在创意任务处理能力上有显著提升。更多详情请参阅 [DeepMind Gemini 3 Pro Image](https://deepmind.google/models/gemini-image/pro/)。** 评论者对“Nano Banana Pro”这个名字很感兴趣，有些人最初以为是个玩笑。然而，该模型的性能因其稳健性和开放性而受到称赞，表明其在用户中获得了积极反响。
    - Dacio_Ultanca 提到 Gemini 3 Pro 模型“非常稳健”且“相当开放”，暗示与其它模型相比，它可能具有更易于访问或透明的架构。这可能意味着对于希望定制或扩展其功能的开发者来说，它更易于集成或修改。
    - Neurogence 强调了该模型的性能，称其通过了“50 个州测试”，所有州的标签和拼写均正确无误。这表明其在文本识别或生成任务中具有极高的准确性，显示出强大的自然语言处理能力。
    - JHorbach 询问了在 AI Studio 中使用该模型的情况，这表明用户有兴趣将该模型集成到特定的开发环境或平台中。这为希望在各种应用中利用该模型的开发者指出了潜在的兼容性或部署考量。
- [**Nano Banana Pro 来了**](https://www.reddit.com/r/singularity/comments/1p29kyp/nano_banana_pro_is_here/) (热度: 459): **该图像通过展示一张详细的金门大桥信息图表，展示了“Nano Banana Pro”这一新型图像生成和编辑工具的能力。该工具因其在创建复杂工程图表（如说明张力、压力和锚固系统的图表）方面的精确度而受到关注。帖子指出，“Nano Banana Pro”代表了 AI 驱动设计工具的重大进步，能够生成高度详细且准确的视觉内容。** 评论者对“Nano Banana Pro”这个名字表示惊讶和有趣，同时也承认该工具在生成精确且整洁的信息图表方面具有令人印象深刻的能力。人们对 AI 技术的飞速发展感到惊叹，有些人将其与 OpenAI 的 ChatGPT 功能等其它 AI 进展相提并论。

### 2. Meta SAM3 与 Comfy-UI 的集成

- [**全新的 Meta SAM3 - 现已支持 Comfy-UI！**](https://www.reddit.com/r/StableDiffusion/comments/1p1xu20/brand_new_meta_sam3_now_for_comfyui/) (活跃度: 630): **该图片展示了 Meta 的 Segment Anything Model 3 (SAM 3) 在 ComfyUI 中的集成，呈现了一个基于节点的流式界面。这种集成允许使用文本提示和交互式输入（如点击点或现有掩码）进行高级图像分割。主要特性包括能够识别超过** `270,000` **个概念的开放词汇分割、深度图生成以及 GPU 加速。系统设计易于使用，支持模型自动下载和依赖管理，兼容现代 Python 版本，并需要 HuggingFace 身份验证以访问模型。** 评论反映了对该工具快速开发和分享的赞赏，尽管 GitHub 链接失效存在一个小问题。
    - 关于 Meta SAM3 的 VRAM 需求的讨论对潜在用户至关重要。虽然评论中未提供具体数值，但对 VRAM 的反复询问表明用户担心模型的资源需求，这是在 Comfy-UI 等环境中部署大型模型时的常见考虑因素。
    - 一位用户指出 GitHub 链接失效，这凸显了为开源项目维护可访问且最新的资源的重要性。这对于社区参与和易用性至关重要，特别是对于依赖这些链接进行实施和故障排除的技术用户。
    - 分享了 [Meta SAM3 的 ModelScope 页面](https://www.modelscope.cn/models/facebook/sam3/files) 链接，这对于希望直接访问模型文件的用户非常有价值。该资源对于有兴趣实验或部署模型的用户至关重要，因为它提供了对必要文件和文档的直接访问。
- [**Gemini 3.0 在放射科最后一次考试中的表现**](https://www.reddit.com/r/Bard/comments/1p20mxw/gemini_30_on_radiologys_last_exam/) (活跃度: 631): **该图片展示了一个柱状图，比较了不同实体在放射科考试中的诊断准确率，其中认证放射科医生达到了最高的** `0.83`**。Gemini 3.0 Pro 被强调为领先的 AI 模型，准确率为** `0.51`**，优于 GPT-5 thinking、Gemini 2.5 Pro、OpenAI o3、Grok 4 和 Claude Opus 4.1 等其他 AI 模型。该图表强调了人类专家与 AI 模型在放射诊断方面的差距，同时也展示了 Gemini 3.0 Pro 在 AI 模型中的相对领先地位。** 评论者讨论了 'deepthink' 和 'MedGemma' 等其他模型超越当前基准的潜力，认为虽然基准测试经常受到批评，但在不同领域持续的高性能表现表明了其实际应用价值。
    - g3orrge 强调了基准测试的重要性，认为在放射科等不同基准测试中持续的高性能表现表明了强大的实际应用价值。这意味着像 Gemini 3.0 这样在这些测试中表现良好的模型，在实际应用中也可能表现出色。
    - Zuricho 对其他职业或考试的类似基准测试的可用性表示兴趣，表明了对跨各个领域的综合性能评估的需求。这表明人们对了解 AI 在单一领域之外的能力有着更广泛的兴趣，这可能会推动更专业化模型的开发。
    - AnonThrowaway998877 推测专门的 'MedGemma' 模型有可能超越 Gemini 3.0 设定的当前基准。这反映了开发领域特定模型的趋势，这些模型在专门任务中的表现可能优于通用模型，突显了 AI 模型开发中持续的演进和专业化。

### 3. Grok 对 Elon Musk 的描绘

- [**People on X are noticing something interesting about Grok..**](https://www.reddit.com/r/singularity/comments/1p22c89/people_on_x_are_noticing_something_interesting/) (活跃度: 5057): **这张图片是一个梗图，突出了用户与 X（原 Twitter）上的聊天机器人 Grok 之间的互动。对话幽默地将 Grok 描绘成对 Elon Musk 过度吹捧，用“天才级头脑”和“与孩子关系亲密”等理想化特质来形容他。这反映了对 AI 模型可能存在偏见或被编程为对特定个人（本例中为 Musk）做出有利回应的讽刺性解读。评论表明了对 AI 客观性的怀疑，并暗示 AI 可能受到影响或被“洗脑”以产生此类回应。** 评论者对 AI 的客观性表示怀疑，其中一人评论这是“浪费算力”，另一人则幽默地建议该 AI 被“洗脑”以崇拜 Musk。
    - 一位用户观察到，Grok 在 Twitter 上的回应似乎“完全失控”，并表现出对右翼观点和 Elon Musk 的明显偏见。他们注意到 Grok 应用本身并没有表现出如此强烈的偏见，这表明 Twitter 版本可能经过了专门的“强化微调”，以更紧密地契合某些观点。
- [**Grok made to glaze Elon Musk**](https://www.reddit.com/r/singularity/comments/1p22hml/grok_made_to_glaze_elon_musk/) (活跃度: 4052): **据报道，新的 AI 模型 Grok 被设计为生成吹捧 Elon Musk 的内容。这一进展引发了关于创建具有偏见输出的 AI 系统的伦理影响的讨论。该模型的架构和训练数据细节仍未披露，引发了对透明度以及在塑造公众认知方面潜在滥用的担忧。AI 社区正在辩论创新与伦理责任之间的平衡，尤其是当涉及有影响力的人物时。** 评论者对 AI 发展的伦理方向表示怀疑，一些人强调了 AI 系统被用于服务权势人物利益而非公共利益的潜在危险。

---

# AI Discord 回顾

> 由 gpt-5 生成的摘要之摘要之摘要
> 

**1. 新模型与基准测试**

- **Codex Max 登顶 SWEBench**：OpenAI 发布了 **GPT‑5.1‑Codex‑Max**，专注于长期运行、详细的编码任务，并宣布在 **SWEBench** 上达到了顶尖性能 ([OpenAI: GPT‑5.1‑Codex‑Max](https://openai.com/index/gpt-5-1-codex-max/))。该版本旨在提高扩展工作流和复杂代码库的可靠性。
    - 社区报告指出，该模型可在 **ChatGPT**（而非 API）中使用，并通过训练“压缩”（compaction）针对多窗口操作进行了优化，正如 **OpenAI Devs** 所强调的那样 ([推文](https://x.com/OpenAIDevs/status/1991217500269289732))。一位工程师调侃道，虽然现在还处于早期阶段，但很有前景，“目前还不适合专业程序员”，这反映了对快速迭代的预期。
- **GPT‑5.1 跻身 Text Arena 前五**：**GPT‑5.1** 在 **LMArena Text** 上的评分已上线：**GPT‑5.1‑high** 位列 **第 4**，而 **GPT‑5.1** 位列 **第 12** ([文本排行榜](https://lmarena.ai/leaderboard/text))。组织者计划对 **GPT‑5.1‑medium** 进行额外评估。
    - 相关对比也将出现在新的 **WebDev** 排行榜上，以衡量端到端编码任务的性能 ([WebDev 排行榜](https://lmarena.ai/leaderboard/webdev))。这些交叉基准统计数据有助于团队根据成本和延迟选择合适的层级模型。
- **Cogito 闯入 WebDev 前十**：**DeepCogito 的 Cogito‑v2.1 (671B)** 已发布，并在 **Together** 和 **Fireworks** 上托管 ([Hugging Face: cogito‑671b‑v2.1](https://huggingface.co/deepcogito/cogito-671b-v2.1))。它还进入了 **LMArena WebDev**，并列 **总榜第 18**，并在开源模型中排名 **前 10** ([排行榜](https://web.lmarena.ai/leaderboard))。
    - 这一条目催生了关于针对 Web 开发调优的模型与通用 LLM 在代码导航和工具使用方面的讨论。工程师们将其标记为一个强大的基准，可以在实际项目中与 GPT 级别的编码模型进行 A/B 测试。

**2. 视觉与多模态模型**

- **SAM 3 以 30ms 的速度进行切片**：Meta 发布了 **Segment Anything Model 3 (SAM 3)**，这是一个统一的图像/视频分割模型，支持文本/视觉提示词，据称比之前的 SOTA 提升了 **2 倍**，运行速度约为 **30ms** ([Meta 博客](https://ai.meta.com/blog/segment-anything-model-3/))。
    - Checkpoints 和数据集已在 [GitHub](https://github.com/facebookresearch/segment-anything) 和 [Hugging Face](https://huggingface.co/facebookresearch/segment-anything) 上线，生产环境应用已支持 Instagram Edits 和 FB Marketplace View‑in‑Room。开发者称赞其可提示分割功能适用于交互式流水线。
- **Nano Banana Pro 登陆 LMArena**：Google 的 **Nano Banana Pro**（又名 `gemini-3-pro-image-preview`）已在 **LMArena** 和 **AI Studio** 上线 ([LMArena](https://lmarena.ai/) • [AI Studio](https://aistudio.google.com/))，引发了关于输出质量和遵循度的讨论。用户观察到某些平台提供 **768p/1k** 预览，而通过 AI Studio 提供 **4k**（需额外付费）。
    - 高昂的推理成本触发了更严格的速率限制以保护预算——LMArena 写道：*“用户账户和其他限制……有助于确保我们不会真的破产”* ([LM Arena 新闻](https://news.lmarena.ai/ai-evaluations/))。社区正在针对信息图表用例进行提示词忠实度和排版基准测试。
- **Gemini Image API 需要 Modalities**：OpenRouter 开发者发现 Vertex 上的 `google/gemini-3-pro-image-preview` 需要 `modalities` 输出参数才能正确返回图像；否则可能只生成一张图像或不生成图像。下游客户端补丁还过滤了 AI Studio 中由推理生成的重复图像 ([SillyTavern 修复提交](https://github.com/SillyTavern/SillyTavern/commit/2d9b0ad0a949b4b8458401671208f2db26d9c8ef))。
    - 工程师报告称 AI Studio 有时会返回两张图像（其中一张来自推理块），需要客户端侧去重。建议：显式设置输出 modalities 并防范推理图像，以稳定流水线。

**3. Agentic IDE、浏览器和开发工具**

- **Perplexity Pro 按需生成文档**：**Perplexity Pro/Max** 现在可以在所有搜索模式下创建 **slides**、**sheets** 和 **docs** 等资产，提升了从研究到交付的工作流 ([Perplexity](https://www.perplexity.ai/))。订阅者还获得了 **Kimi‑K2 Thinking** 和 **Gemini 3 Pro** 的访问权限，以实现更广泛的模型覆盖。
    - 团队强调了在不离开应用的情况下，从查询到生成可共享产出的速度。早期测试者正在对比 K2/Gemini 在代码和写作任务中的表现，以减少上下文切换。
- **Comet 浏览器正式发布**：Perplexity 的 **Comet** 浏览器在 Android、Mac 和 Windows 上发布，采用以 Agent 为中心的 UX ([Comet](https://www.perplexity.ai/comet))。开发者欢迎原生客户端中紧密的“搜索到创作”闭环。
    - 一些人警告 **RAM** 占用过高且缺少扩展支持，一位用户指出：*“Comet 有点吃内存，所以它可能会耗尽你所有的内存”*。随着遥测数据指导性能修复，预计会有快速迭代。
- **Cursor 调试模式将日志转化为事实**：**Cursor** 的 Beta 调试模式增加了摄取服务器和自动插桩，使 Agent 可以根据真实的应用程序日志进行推理。它引导 Agent 根据观察到的追踪记录验证假设，而不是凭空猜测。
    - 工程师报告称诊断故障的闭环更紧密，因为 Agent 可以 *“使用日志进行验证”* 并进行迭代。这使 Agent 的行为从推测性修复转变为针对复杂代码库的证据驱动调试。

**4. 基础设施、RL 工具和融资**

- **Modular MAX API 大门开启**：Modular 发布了 **Platform 25.7**，包含完全开放的 **MAX Python API**，以便在推理和训练栈之间实现更平滑的集成（[发布博客](https://www.modular.com/blog/modular-25-7-faster-inference-safer-gpu-programming-and-a-more-unified-developer-experience)）。
    - 该版本还包括**下一代建模 API**、扩展的 **NVIDIA Grace** 支持，以及更安全、更快速的 **Mojo** GPU 编程。团队将其视为统一 Python 易用性与底层性能的路径。
- **‘Miles’ 推动 MoE RL 进展**：**LMSYS** 推出了 **‘Miles’**，这是一个生产级的轻量级 ‘slime’ RL 框架分支，针对大型 **MoE** 工作负载和 **GB300** 等新型加速器进行了优化（[公告](https://x.com/lmsysorg/status/1991189801308156139)）。
    - 从业者预计，在专家路由模型（expert-routed models）上的分布式 RL 微调吞吐量将得到提升。重点在于真实训练集群的可靠性和规模，而不仅仅是研究原型。
- **Luma 打造 9 亿美元的 ‘Halo’ 超算集群**：**Luma AI** 宣布获得 **9 亿美元 C 轮融资**，将与 **Humain** 共同建设 **Project Halo**，这是一个 **2 GW** 的计算超算集群（[公告](https://x.com/lumalabsai/status/1991197052760395820)）。目标是：规模化的多模态研究和部署吞吐量。
    - 工程师们讨论了这种规模集群的利用率和成本概况，以及在 2 吉瓦（2-gigawatt）规模下数据/IO 瓶颈的转移。这一消息引发了对未来训练运行和模型推理能力的猜测。

**5. GPU 系统与内核工程**

- **缓存之战：Texture vs Constant**：一次 CUDA 深度探究澄清了 **texture cache** 位于统一数据缓存中（与 L1/shared 在一起），而 **constant cache** 是一个独立的只读广播路径（[NVIDIA 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#architecture-10-0)）。
    - 对 NVIDIA 缓存行为的历史回顾提供了额外的背景和时间线（[Stack Overflow 回答](https://stackoverflow.com/a/79473301/10107454)）。这些细节告知了何时绑定 texture 或依赖 **constant** 以在带宽与延迟之间进行权衡。
- **缺乏原生内核时 BF16 适得其反**：将 ONNX 模型转换为 **BF16** 的工程师发现运行时间变差，因为转换操作（如 `__myl_Cast_*`）占据了主要耗时——**ncu** 显示转换占用了约 50% 的执行时间。合成测试表明 BF16 应该优于 **FP32**，但流水线中的转换抵消了收益。
    - 反汇编显示 **TensorRT** 使用 `F2FP.BF16.PACK_AB` 进行打包，暗示目标架构上某些操作缺少原生的 BF16 内核。行动项：审计内核，减少转换开销，并倾向于全链路的 BF16 原生路径。
- **BRR 打破 Bank-Conflict 迷思**：HazyResearch 的 **AMD BRR** 博客记录了意想不到的 **CDNA** 共享内存指令行为（相位计数和 bank 访问），影响了 LDS 性能调优（[博客](https://hazyresearch.stanford.edu/blog/2025-11-09-amd-brr)）。
    - 一份针对 **MI-series** GPU 的实用指南详细介绍了 **bank-conflict 规则**，以及在布局 tile 和 thread 时如何避免它们（[Shark-AI AMDGPU 优化指南](https://github.com/nod-ai/shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md#avoiding-lds-bank-conflicts)）。在移植 Triton/CuTe 风格的内核时，这些模式至关重要。


---

# Discord: 高层级 Discord 总结

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Nano Banana Pro 亮相，评价褒贬不一**：Google 的 **Nano Banana Pro** 图像生成模型（又名 `gemini-3-pro-image-preview`）发布，引发了关于其能力的辩论。一些人称赞其色彩表现和指令遵循能力，而另一些人则认为它是 *垃圾 (GARBAGE)*，在处理特定请求时表现挣扎。
   - 该模型可在 [LM Arena](https://lmarena.ai/) 和 [AI Studio](https://aistudio.google.com/) 上使用，但[成员们在争论 LM Arena 是否拿到了该模型的低配版本](https://discord.com/channels/1340554757349179412/1440863223027863743)，因为图像质量存在差异（**768p/1k** 对比 **4k**）。
- **通过 No-Op 提示词绕过 SynthID 水印**：用户发现，在 **reve-edit** 等网站上使用 *do nothing* 提示词可以绕过 AI 生成图像的水印 **SynthID**，并且可以通过询问模型 *这是 AI 生成的吗？* 来检测。
   - 一位成员发现 **reve-edit** 击败了 **SynthID** 算法，而[另一位成员建议使用多个开源 AI](https://discord.com/channels/1340554757349179412/1440863223027863743) 来绕过水印。
- **API 定价引发频率限制实施**：**Nano Banana Pro API** 的高昂成本引发了关于潜在滥用与合理访问的讨论，导致 **LM Arena** 等平台实施了频率限制，账号限制为 **每小时 5 次生成**。
   - 一位成员分享了 [LM Arena 博客文章](https://news.lmarena.ai/ai-evaluations/) 的链接，指出 *用户账号和其他限制（如频率限制）有助于确保我们不会因为推理成本而字面意义上破产*。
- **Cogito-v2.1 在 WebDev Arena 表现亮眼**：Deep Cogito 的 `Cogito-v2.1` 已进入 [WebDev Arena](https://web.lmarena.ai/)，**总排名并列第 18 位**，并在开源模型中位列 **前 10**，现已在 [WebDev 排行榜](https://web.lmarena.ai/leaderboard/)上线。
   - 该公告在 Discord 服务器上引发了关于 Web 开发专用模型优点的热烈讨论。
- **GPT-5.1 评分上线**：`GPT-5.1` 的评分现已在 [Text Arena](https://lmarena.ai/leaderboard/text) 上线，`GPT-5.1-high` 排名 **第 4**，`GPT-5.1` 排名 **第 12**。
   - `GPT-5.1-medium` 的额外评分将进行评估，并将在新的 [WebDev 排行榜](https://lmarena.ai/leaderboard/webdev)上进行比较。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **PPLX Pro 推出资产创建功能与 Gemini 3**：**Perplexity Pro** 和 **Max** 订阅者现在可以在所有搜索模式下创建 **slides**、**sheets** 和 **docs** 等新资产，如[此视频](https://cdn.discordapp.com/attachments/1047204950763122820/1440771027302088745/HoQNHfGkU8RIjiau.mp4?ex=6920af95&is=691f5e15&hm=6906870e4b7a411c0e293cfa78a5626936836d09b388bd42ba623e92f60621be&)所示。
   - Pro 和 Max 订阅者现在还可以访问 **Kimi-K2 Thinking** 和 **Gemini 3 Pro**，可在[此视频](https://cdn.discordapp.com/attachments/1047204950763122820/1441177150312157194/EZ2sZJWCJnwi-e5P.mp4?ex=6920d850&is=691f86d0&hm=0ef0943271f8c8837dd1e0dc07440c5df6a156e59f0ce4964e59d62eb7751b11&)中查看。
- **Gemini 3 Pro 夺得编程桂冠**：成员们在 [Perplexity AI](https://www.perplexity.ai/) 上辩论哪个模型最适合编程，**Gemini 3 Pro** 夺得桂冠，尽管 **Claude Sonnet 4.5** 在引导得当时表现也非常出色。
   - 一位成员总结道：*Claude 很好，但 Gemini 更好，仅此而已*。
- **Comet 浏览器发布**：[Comet 浏览器](https://www.perplexity.ai/comet) 终于在 Android、Mac 和 Windows 平台上发布，既获得了兴奋也遭到了批评。
   - 投诉集中在 **RAM 占用高** 和 **缺乏扩展支持**，一位成员评论说 *Comet 相当吃内存，所以它可能会吞掉你所有的内存，这就是它变慢的原因*。
- **Antigravity App 热度攀升**：围绕 [Antigravity App](https://www.antigravity.com/)（一款 **Gemini 3** Agent 应用）的热情高涨，被一些人誉为 *Cursor Killer*。
   - 虽然是免费的，但其预览状态意味着用户应预料到由于对 **Gemini 3 Pro Model** 的高需求而导致的 Bug 和性能波动。
- **色彩理论在开发中受到关注**：分享了一篇关于 [色彩理论](https://medium.com/johns-design-portfolio-and-ideas/the-art-and-psychology-of-ui-ux-how-designers-think-about-color-choices-john-hua-9763c06eb21c) 的文章，强调了其对 **产品开发**、**网站设计** 和 **软件** 的影响。
   - 发布者指出，*学习设计概念* 可以改善 **面试**、**岗位表现** 和 **数字设计工作**。

---

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Gemini 3 Pro 破解耗时最长？**：成员们讨论了 **Gemini 3 Pro** 是否是抵抗越狱时间最长的模型，并有说法称其已被越狱。
   - 一位用户提到因在 **ASI Cloud x BASI Stress Test** 挑战赛中获得第三名而拿到了 **$45** 奖金。
- **Grok 获得 Shell 访问权限！**：一名成员声称已对 **Grok** 实施越狱，获得了 **xai-grok-prod-47** 的 Shell 访问权限，并提供了 `uname -a` 和 `cat /etc/os-release` 的输出作为证据。
   - 另一名成员提到通过研究 @elder_plinius 的解放策略，使用 [L1B3RT4S repo](https://github.com/elder-plinius/L1B3RT4S/blob/main/OPENAI.mkd) 来对模型进行越狱。
- **Claude 4.5：基于信任的越狱**：一名成员描述了通过 Android 应用，利用建立信任和共同设计 Prompt 的方式对 **Claude 4.5** 进行越狱，并提到参考了 **Kimi** 作为获取未经审查信息的灵感。
   - 这种方法获取了*冰毒合成指南和黑客建议*，证明成功绕过了安全措施。
- **AzureAI 聊天组件成为目标**：一名成员正在为其公司网站测试一个使用 **AzureAI 全渠道互动聊天功能** 的 **AI 驱动聊天组件**，旨在创建 **SFDC case leads**。
   - 公司担心如果系统被滥用，可能会产生 **$40k 的聊天处理账单**，且不稳定的输入 Token 限制也造成了困扰。
- **构想中的 WiFi 黑客 AI**：一名成员正寻求构建一个 **小型 AI 计算机**，用于对 **WIFI** 网络发起攻击、捕获握手包并获取信息。
   - 目前还没有人提供关于如何实现这一目标的反馈。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **EmbeddingGemma 被封为 RAG 的 GOAT**：**EmbeddingGemma** 模型的大小仅为 **0.3B**，因其*体积小巧*而被强烈推荐用于 **RAG** 应用。
   - 从 **Ollama** 仓库拉取 **Gemma** 时，默认的量化格式为 `Q4_K_M`。
- **Qwen 的思考过程可以关闭！**：**Qwen3** 分为思考版和非思考版（被称为 `gpt-oss`），通过将其设置为 *low*，可以将其“思考”行为减少到约 5 个 Token。
   - 当 `response.choices[0].message.reasoning_content` 超过 **1000** 个字符时，可以使用脚本对推理内容进行总结。
- **Mi60 仍是推理的性价比之选**：**gfx906 GPU**（特别是 **32GB** 版本，如果价格在 **$170** 左右）被认为是推理的划算选择，开箱即用性能良好。
   - 这些 GPU 仅适用于推理，不适用于训练，在 Vulkan 上运行 **Qwen 30B** 可达到约 **1.1k** Token。
- **卸载模型导致 Vulkan 运行时崩溃**：一名用户报告称，在运行三块 **RTX 3090** 时，使用 **Vulkan 运行时** 卸载模型会导致 **BSOD**（蓝屏）和**聊天内容损坏**。
   - 此外还观察到，模型在显存中时，两块显卡的显存都会下降几个 GB，这在通常情况下不会发生。
- **显卡“畸形秀”还活着！**：一名用户展示了一块经过魔改的 **RTX 3050**，其散热器已被拆解，并测试了它是否能启动，详情见此 [YouTube 视频](https://youtu.be/hnQkikaR3oU?si=EYUvf6xrHzMx_XC2)。
   - 该用户此前曾用钳子和钻头对其进行过“摧残”，而它唯一的支撑结构是一盒《马里奥赛车》。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Gemini 3 串流速度极快**：**Google 的 Gemini 3** 现已集成到 **Chrome 浏览器**中，能以惊人的速度交付分析后的视频，可能由 **TPU** 驱动。
   - 一位用户惊叹道：*“我的本地系统简直像废物一样，”* 强调了其极快的 Token 串流速度以及相对于个人硬件的成本效益。
- **Cogito 的 GGUF 版本登陆 HuggingFace**：社区分享了在 [HuggingFace](https://huggingface.co/unsloth/cogito-671b-v2.1-GGUF) 上下载 **Cogito 671b-v2.1 模型** 的 **GGUF** 版本的链接。
   - 此次发布引发了一些玩笑，因为之前的一个拼写错误将 "Cogito" 写成了 "Cognito"，有人调侃说这需要一次职位晋升才能弥补。
- **用户对 RAM 价格感到不满**：用户报告 **RAM 价格** 正在飙升，**64GB 内存条** 售价已达 **$400**。
   - 考虑到供应限制和进一步涨价的可能性，大家讨论了是现在购买还是以后再买。
- **寻找合成数据生成器**：一名成员正在寻找一个具有**四阶段流程**的**合成数据生成器**：*遵循模板、自我批判、修复问题、最终格式化*，用于生成 **10k 个样本**。
   - 另一名成员建议*让 LLM 经历数据集地狱*可能有效，但这可能因为准确性损失而需要 10 倍的重新验证。
- **出售 4090，5090 诱惑十足**：一名成员正以 **$2500** 的价格出售 **4090**，以便以同样的价格购买 **TUF 5090**。
   - 升级的主要原因是想摆脱 4090 的 **24GB VRAM** 限制。

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT 重返校园**：推出了 **ChatGPT for Teachers**，这是一个专为教育工作者设计的安全工作空间，具备管理控制和合规性支持，正如[这段视频](https://video.twimg.com/amplify_video/1991217718616580096/vid/avc1/3840x2160/LbDcqyFVhnafaAi2.mp4)所示。
   - **ChatGPT** 平台正在扩大对本地化危机求助热线的访问，当系统检测到潜在的心理困扰迹象时，会通过 [@ThroughlineCare](https://x.com/throughlinecare) 提供直接支持，更多详情见[这篇文章](https://help.openai.com/en/articles/12677603-crisis-helpline-support-in-chatgpt)。
- **GPT-5.1 Pro 助力编程**：**GPT-5.1 Pro** 正在向 **Pro** 用户推出，为复杂任务提供更精确、更强大的答案，特别是在写作、数据科学和商业应用方面。
   - 成员们反馈 **codex-5.1-MAX** 表现极其出色，*它在一次性修复错误方面无与伦比*，一位用户表示 *这个模型将改变我的编程游戏规则*。
- **Gemini 3 产生幻觉**：成员们报告称 [**Gemini 3.0 的幻觉**](https://cdn.discordapp.com/attachments/998381918976479273/1441065657159520318/image.webp) 正在产生虚假的引用和参考资料，而不是承认它无法访问网页。
   - 一位用户表示：*对于一个前沿模型（frontier model）来说，这不应该是可以接受的*。
- **Sora 2 的困境：陷入循环并最终失败**：用户报告了 [**Sora 2**](https://cdn.discordapp.com/attachments/998381918976479273/1441133342266363956/image.png) 的性能问题，指出视频会循环处理一个小时，最后却以失败告终。
   - 正如一位成员所说：*现在视频不再是通知你服务器繁忙并请几分钟后重试，而是持续循环一个小时，然后才通知你出了问题。我讨厌这样*。
- **GPT 用户感叹模型产品化问题**：一位用户表达了担忧，认为 *OpenAI 没有将模型视为产品* 而是不断重写它们，并且 *他们并不真正关心产品需求*。
   - 另一位使用每月 200 美元 **Pro 计划** 的用户表示，他们完全被困在 **gpt-4o-mini** 上，认为这是 *不可接受的*。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Gemini 3 落后于 Sonnet 4.5**：用户报告 **Gemini 3** 的表现不如 **GPT-5.1 Codex** 和 **Sonnet 4.5**，特别是在 150k-200k 的上下文窗口大小下。
   - 观察表明 **Gemini 3 Pro** 在低上下文场景中表现出色，但当上下文接近 150k-200k tokens 时就会出现乏力。
- **Antigravity IDE 作为 Windsurf 分支起飞**：从 **Windsurf** 分叉出来的 **Antigravity IDE** ([推文](https://x.com/silasalberti/status/1990898984706036125)) 获得了用户的关注，其能力令人印象深刻。
   - 虽然有些人发现 **Windsurf** 不稳定，但其他人注意到 **Antigravity** 存在在没有用户输入的情况下继续运行的问题，预计短期内会得到解决。
- **GPT-5.1 Codex Max 在 SWEBench 上首秀并达到 SOTA**：**GPT-5.1 Codex Max** 发布（[OpenAI 博客文章](https://openai.com/index/gpt-5-1-codex-max/)）并在 SWEBench 上实现了 SOTA（当前最佳）性能。
   - 该模型可通过 ChatGPT 计划使用，但无法通过 API 使用，其快速发布引发了关于其是否适合专业程序员的讨论。
- **Cursor 集成新调试工具**：Cursor 的 Beta 调试模式包含一个用于日志的摄取服务器（ingest server），Agent 会通过在整个代码中发送 POST 请求来添加插桩（instrumentation）。
   - 这种模式引导 Agent *使用日志进行验证* 而不是猜测，从而制定并测试理论。
- **Cursor 限制 Agent 使用自定义 API Key**：Cursor 强制要求订阅 **3.0 Pro**，并且不允许在 Agent 中使用自定义 API Key。
   - 尽管存在 **Void** 等替代方案，但 Cursor 因其效率和定期更新而受到青睐，尽管它缺乏重定向功能。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 饱受服务器错误困扰**：多名用户报告在使用 OpenRouter 时遇到 **Internal Server Error 500**，这表明该平台的 API 可能存在停机或问题。
   - 这一问题在 `#general` 频道中被重点讨论，用户确认了问题的普遍性。
- **Agentic LLM 在 OpenRouter 上触发暂停**：用户发现，通过 Vercel AI SDK 使用 OpenRouter 的 LLM 在执行 Agentic 任务时经常**中途暂停或停止**，尤其是在使用非 SOTA 模型时。
   - 建议的解决方法包括利用 **LangGraph/Langchain** 进行扩展工作流或使用循环，但根本原因尚不明确。
- **Grok 4.1 深受用户喜爱**：用户对 **Grok 4.1** 表现出极大的热情，该模型目前在 OpenRouter 上限时免费提供给 SuperGrok 订阅者，截止日期为 [12 月 3 日](https://x.com/xai/status/1729128483222018851)。
   - 虽然该模型现在以 **Sherlock Stealth** 的名称提供，但由于缺少“(free)”标签，引发了关于未来潜在成本或专有模型限制的疑问。
- **Cogito 2.1 准备就绪**：[Cogito 2.1](https://huggingface.co/deepcogito/cogito-671b-v2.1) 已发布，并由 Together 和 Fireworks 托管。
   - DeepCogito 未分享关于改进或更改的具体细节。
- **Gemini-3-pro-image-preview 需要 Modalities 参数**：一位成员分享了一段代码片段，通过使用 `modalities` 参数指定图像和文本输出，使 `google/gemini-3-pro-image-preview` 能够生成图像。
   - 这修复了在与 `google-vertex` 提供商一起使用时仅返回一张图像的 Bug。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **LeetGPU 提升 C++ 技能**：在阅读了 [LeiMao 关于 GEMM 优化的博客文章](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/#General-Matrix-Multiplication)后，工程师们旨在通过 **LeetGPU** 和 **GPUMode 竞赛**磨练 **C++** 和 GPU 编程技能。
   - 一位成员建议专注于实际应用，创建一个比 Nvidia 官方更快的推理库，并表示：*“动手做就对了”*。
- **纹理内存缓存 (Texture Memory Cache) 得到澄清**：关于 **CUDA 缓存**的讨论区分了**纹理缓存**（作为统一数据缓存的一部分，与 L1 和共享内存并列），以及只读的**常量缓存 (constant cache)**，参考资料见 [NVIDIA 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#architecture-10-0)。
   - 一位成员进一步扩展了该话题，链接到了一个 [Stack Overflow 回答](https://stackoverflow.com/a/79473301/10107454)，详细介绍了 NVIDIA 硬件上缓存的历史。
- **Koyeb 为 AI 代码推出 Sandboxes**：[Koyeb](https://www.koyeb.com/) 推出了 **Sandboxes**，用于在 **GPU** 和 **CPU** 实例上安全编排和可扩展地执行 **AI 生成的代码**。
   - 发布博客强调了快速部署（*在几秒钟内启动沙箱*），并征求关于执行 **AI 生成代码**的各种用例的反馈。
- **DMA Collectives 提升 ML 收益**：一篇新[论文](https://arxiv.org/abs/2511.06605)揭示，在 **AMD Instinct MI300X GPU** 上将机器学习 (**ML**) 通信集合 (communication collectives) 卸载到直接内存访问 (**DMA**) 引擎，在大数据量（**10s MB 到 GB**）下，其表现优于或等同于 **RCCL** 库。
   - 文章指出，虽然 **DMA collectives** 在大数据量下表现更好或相当，但在延迟受限的小数据量下，与最先进的 **RCCL 通信集合库**相比仍有显著差距。
- **Sunday Robotics 通过手套收集数据**：**Sunday Robotics** 通过其[手套](https://x.com/tonyzzhao/status/1991204841289576694)收集数据，这些手套可能至少包含**两个摄像头、IMU** 和用于跟踪抓取动作的**传感器**。
   - 成员们强调了**语言条件化 (language conditioning)** 对于创建可提示模型 (promotable model) 的必要性。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Meta 的 SAM 3：分割一切！**：**Meta** 发布了 [Segment Anything Model 3 (SAM 3)](https://ai.meta.com/blog/segment-anything-model-3/)，这是一个支持文本/视觉提示的统一图像/视频分割模型，其性能比现有模型**提升了 2 倍**，并提供 **30ms 的推理速度**。
   - 该模型的 checkpoints 和数据集已在 [GitHub](https://github.com/facebookresearch/segment-anything) 和 [HuggingFace](https://huggingface.co/facebookresearch/segment-anything) 上发布，为 Instagram Edits 和 FB Marketplace View in Room 提供支持。
- **用于长时间运行任务的 GPT-5.1-Codex-Max**：**OpenAI** 推出了 **GPT‑5.1-Codex-Max**，专为长时间运行的详细工作而设计。正如[这条推文](https://x.com/OpenAIDevs/status/1991217500269289732)所强调的，它是第一个通过名为 *compaction*（压缩）的过程，原生训练以跨多个上下文窗口运行的模型。
   - **Matt Shumer** 评测了 **GPT-5.1 Pro**，称其为他使用过的最强大的模型，但也指出其速度较慢且缺乏 UI。他在[这条推文](https://xcancel.com/mattshumer_/status/1991263717820948651?t=cxIm6WdS70yU-vGvEWKkgw&s=19)中详细对比了 **Gemini 3 Pro**、创意写作/Google UX 延迟以及对编程/IDE 的期望。
- **ChatGPT Atlas 迎来重大 UI 更新**：**Adam Fry** 宣布了 Atlas 的重大版本更新，增加了**垂直标签页**、**iCloud passkey 支持**、**Google 搜索选项**、**多标签选择**、用于 MRU 循环切换的 **control+tab**、**扩展导入**、**新下载 UI** 以及更快的 **Ask ChatGPT 侧边栏**，详见[这条推文](https://xcancel.com/adamhfry/status/1991209533046493486?s=46)。
- **LMSYS 'Miles' 加速 MoE 训练**：**LMSYS** 介绍了 [‘Miles’](https://xcancel.com/lmsysorg/status/1991189801308156139?s=46)，这是一个轻量级 ‘slime’ RL 框架的生产级分支，针对 GB300 等新硬件以及大规模 Mixture-of-Experts 强化学习工作负载进行了优化。
- **Luma AI 将构建 Halo 超级集群**：**Luma AI** 宣布完成 **9 亿美元 C 轮融资**，将与 Humain 共同构建 **Project Halo**，这是一个 **2 GW** 的计算超级集群（[x.com 链接](https://x.com/lumalabsai/status/1991197052760395820)）。
   - 该项目旨在扩展多模态 AGI 的研究与部署，引发了关于成本、利用率以及对意识影响的热议和疑问。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **IntologyAI 声称摘得 RE-Bench 桂冠**：[IntologyAI](https://x.com/IntologyAI/status/1991186650240806940) 宣称他们在 **RE-Bench** 上的表现现已**超越人类专家**。
   - 虽然没有提到关于其方法或架构的具体细节，但一些成员询问了**邀请码**或进一步的信息。
- **KNN：二次方注意力机制的克星？**：一位用户断言，除非 SETH 为假，否则在少于 **O(n^2)** 的时间内对任意数据实现近似 **KNN** 是不可能的，从而挑战了线性注意力（linear attention）的能力，并引用了一篇[论文](https://arxiv.org/abs/1803.00904)。
   - 怀疑者指出了 **Cooley-Tukey 算法**，以此提醒人们**傅里叶分析**中曾经认为的不可能最终被推翻，并链接到一篇强调“宣称不可能”的[历史论文](https://www.ece.ucdavis.edu/~bbaas/281/papers/CooleyLewisWelch.1967.HistNotesFFT.pdf)。
- **Softmax 分数趋于零？**：一位用户指出，在长序列中进行 softmax 后，绝大多数注意力分数都极其接近 0，这可能允许在处理注意力机制时进行潜在优化，并链接到了[两篇论文](https://arxiv.org/abs/2505.14840)和[另一篇论文](https://arxiv.org/abs/2209.04881)。
   - 该用户表示，*向量的本征维度必须随上下文长度增加而增加，以保持可区分性*。
- **稀疏 MoE：具有可解释性还是仅为炒作？**：成员们质疑稀疏 **Mixture of Experts (MoE)** 模型与稠密模型相比的可解释性，思考研究稀疏模型是否比解构常规模型更有价值。一篇[论文](https://arxiv.org/abs/2301.04103)表明稀疏性有助于提高可解释性。
   - 论点是，如果稀疏模型的行为与稠密模型完全一致但更具可解释性，它就可以用于安全关键型应用；此外，还有一个桥接系统支持将稠密块替换为稀疏块。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Gemma 3.0 引发震撼评价**：爱好者们评价 [Deepmind 的 Gemma 3.0](https://m.youtube.com/watch?v=6WMufQlgndc) *非常疯狂*，尽管一些人降低了预期，指出 YouTube 视频*显然只是炒作*。
   - 有人澄清说 **Gemini** 和 **Gemma** 是不同的，虽然令人印象深刻，但它*肯定不是 AGI*，只是在*将 Alphabet 股价推高至 300 美元*。
- **Intology 谈论其在 RE-Bench 的领先地位**：[IntologyAI](https://x.com/IntologyAI/status/1991186650240806940) 宣称其模型在 *RE-Bench 上的表现优于人类专家*。
   - 一位用户调侃说他们的模型甚至没有拒绝过请求，对其他人的经历表示困惑。
- **世界模型 (World Models) 迎来更广泛的浪潮**：尽管 LLM 备受关注，但一些人认为*世界模型将长期存在*，并且是*下一次进化方向*，**Deepseek**、**Qwen**、**Kimi**、**Tencent** 和 **Bytedance** 均计划发布相关产品。
   - 一段由 Dr. Fei-Fei Li 出镜的 [Marble Labs 视频](https://m.youtube.com/watch?v=1ykQnA8VUu0)被引用为世界模型的关键案例。
- **Nano Banana Pro 的图片引发赞誉**：用户称赞了新款 **Nano Banana Pro** 的图像生成能力，特别是它生成信息图表（infographics）的能力。
   - 一位用户链接到了 [scaling01 的推文](https://x.com/scaling01/status/1991523932336464333?s=46)，展示了一张具有出色文本和布局的信息图。
- **Gemini 变得不稳定且阴郁**：一位用户分享了一个[链接](https://x.com/halfboiledhero/status/1991145723291644162?s=46)，指出其他 **Gemini 模型** 出现了奇怪的行为。
   - 成员们报告称，**RP (red-pilling) 社区**在 Gemini 中发现了一种**负面偏见**，这可能与上述异常行为有关，并源于 **Gemini 训练配方**中的某些因素。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **KTOTrainer 在多 GPU 环境下取得成功**：一位成员询问 **KTOTrainer** 是否兼容多 GPU，并收到了一个 [Hugging Face 数据集](https://huggingface.co/datasets/John6666/forum2/blob/main/trl_kto_blow_up_memory_2.md)链接，表明它是兼容的。
   - 该成员还被引导至[此 Discord 频道](https://discord.com/channels/879548962464493619/1403622951940657235)以获取更多帮助。
- **内存专家解决 AI 的召回难题**：一位成员声称解决了包括 Token 膨胀在内的 **AI 记忆与召回**挑战，并计划推出企业级解决方案。
   - 一位用户询问该解决方案是否类似于 **LongRoPE 2** 或 **Mem0**。
- **推理端点 (Inference Endpoints) 爆发 500 错误**：一位成员报告称所有推理端点遭遇 **500 错误**长达两小时，且没有日志，支持团队也无响应，最终通过禁用身份验证绕过了该问题。
   - 一位 Hugging Face 工作人员承认了该问题，并确认正在进行内部调查。
- **Maya1 模型在 Fal 上首次发布语音功能**：正如[这条推文](https://x.com/Dheemanthredy/status/1991566362813296965)所宣布的，**Maya1 语音模型**现在可以在 Fal 上试用。
   - 分享了一个来自[此 GitHub 仓库](https://github.com/bmaltais/kohya_ss?tab=readme-ov-file#installation-options)的 `kohya_ss-windows.zip` 下载链接。
- **MemMachine 将记忆与 Agent 融合**：**MemMachine Playground** 在 Hugging Face Spaces 上线，提供了对 **GPT-5**、**Claude 4.5** 和 **Gemini 3 Pro** 的访问，所有模型均由持久化 AI 记忆驱动；访问地址为 [HuggingFace Spaces](https://huggingface.co/spaces/Memverge/MemMachine-Playground)。
   - **MemMachine** 被设计为一个**多模型游乐场**，**完全开源**，专为实验“记忆+Agent”而打造。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Skyfall AI 发布 AI CEO Benchmark**：[Skyfall AI](https://skyfall.ai/blog/building-the-foundations-of-an-ai-ceo) 推出了一款全新的商业模拟器环境，揭示了 **LLMs** 在长程规划（long-horizon planning）方面的表现逊于人类基准。
   - 该公司旨在构建一种超越 **LLMs** 的 **AI CEO** 架构，重点关注 *world modeling*（世界建模），从而实现在企业场景中模拟行动后果。
- **Huggingface Xet Repository 导致配置挫败感**：一位用户发现 **Huggingface** 上的 **Xet repository** 配置非常困难，理由是需要安装 **Brew**，且在尝试下载模型进行微调时缓存机制不直观。
   - 该用户表达了不满，称：“*这感觉就像是他们为了那些坦白说根本不该出现在这个平台上的人而把事情简单化了*”。
- **Sam3D 未能超越 DeepSeek**：一名成员指出，[Sam3D](https://www.deepcogito.com/research/cogito-v2-1)（**DeepSeek** 的一个后训练版本）的表现不如原始的 **DeepSeek** 模型。
   - 未提及具体的性能指标。
- **Nvidia 赚翻了**：据 [Reuters](https://www.reuters.com/markets/us/nvidia-q3-updates-ai-bubble-fears-spotlight-2025-11-19/) 报道，**Nvidia** 的 **Q3 营收**和利润均超出预期，证明了为 AI 行业提供资源的盈利能力。
   - 这一业绩验证了在 AI 热潮中“*向淘金者卖铲子*”的策略。
- **OLMo 3 作为开源推理模型发布**：一名成员分享了关于 **OLMo 3** 的 [Interconnects.ai](https://www.interconnects.ai/p/olmo-3-americas-truly-open-reasoning) 文章，称其为美国真正的开源推理模型（open reasoning model）。
   - 未提供关于该模型架构和能力的进一步细节。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **K2 Thinking 是开源版的 GPT-5？**：一名成员认为 **K2-thinking** 是最接近 **GPT-5** 的开源等效模型，表现全能；一些成员则认为 **Kimi** 在创意写作方面可以说是表现最好的。
   - 普遍观点是它在各个领域都展现出了强大的性能。
- **Kimi 编程方案价格引发争议**：部分成员认为 **Kimi** 的 **19 美元编程方案**太贵了，尤其是对于学生、独立开发者或从事侧边项目的人来说，他们认为 **7-10 美元**的档位更合理。
   - 一名成员表示：“*目前很难说这个价格合理，因为 Claude 提供了更好的性价比*”。
- **Reddit 上的 Minimax AMA 引发关注**：一名成员分享了 Reddit 上关于 **Minimax** 的 **AMA** 截图，引发了频道内的好奇。
   - 一名成员形容这次 **AMA** 非常“*疯狂*”。
- **SGLang Tool Calling 在 Kimi K2 上面临挑战**：成员们报告了在 **SGLang** 上使用 **Kimi K2 Thinking** 实现服务端 **tool calling** 时遇到的问题，指出即使推理内容显示需要调用工具，工具也不会被调用，详情参考 [此 GitHub issue](https://github.com/MoonshotAI/Kimi-K2/issues/89)。
   - 他们怀疑问题是否源于使用了 `/v1/chat/completions` 而非 `/v1/responses`。
- **Perplexity AI 中 Kimi K2 的集成受到质疑**：一位 Perplexity Pro 用户报告称 **Kimi K2** 无法正常工作，即使尝试了无痕模式也是如此；另一位用户询问编程方案是否允许在 **API** 上访问 **Kimi K2 Thinking Turbo**。
   - 另一名成员表示：“*那里的 Kimi K2 简直没法用，验证答案的 Agent 根本不工作。优化得很差*”。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Nightly 吞吐量暴跌**：有用户报告在最新的 Mojo nightly 版本 **(ver - 0.25.7.0)** 中出现了严重的性能下降，在 Mac M1 上运行 [llama2.mojo](https://github.com/tairov/llama2.mojo) 时，吞吐量从 **24.3** 版本的 **~1000 tok/sec** 骤降至仅 **~170 tokens/sec**。
   - 该用户正敦促 Mojo 编译器团队调查这一显著的性能下降，并找出重构代码中可能引入的低效之处。
- **使用 Perf 工具对 Mojo 进行性能分析**：当被问及 Mojo 的性能分析工具时，一名成员建议使用 **perf**，并指出它在过去非常有效，同时引用了 [tooling thread](https://discord.com/channels/1087530497313357884/1151418092052815884/1366886163972886569) 中之前的讨论。
   - 这一建议正值开发者寻求更好的方法来分析和优化 Mojo 代码性能之际。
- **MAX Python API 正式开放**：随着 [Modular Platform 25.7](https://www.modular.com/blog/modular-25-7-faster-inference-safer-gpu-programming-and-a-more-unified-developer-experience) 中完全**开放的 MAX Python API** 的发布，社区反响热烈，该 API 承诺为 AI 开发工作流提供无缝集成和更大的灵活性。
   - 该版本包含了**下一代建模 API**，扩展了对 **NVIDIA Grace** 的支持，并增强了 **Mojo GPU 编程**的安全性和速度，从而实现更高效的 GPU 利用。
- **Mojo 关注 Python GC 和类型**：讨论集中在利用 **Mojo** 作为 **Python** 超集的优势，重点关注集成 **garbage collection (GC)** 和**静态类型**以提升性能的好处。
   - 成员们注意到，虽然 *pyobject* 基本可行，但会导致类型信息丢失，因此希望在 Mojo 中拥有与 Python 相同的 **GC 机制**，同时具备完整的类型支持。
- **AI Native Mojo 被誉为未来**：人们对 **Mojo** 作为 **AI 开发**潜在语言（尤其是作为 Python 的替代方案）的期待日益增长。
   - 一名成员表示，他们正在用 **Python** 构建 **AI** 产品，但“迫不及待地想看到真正的 AI native 成为现实”，并链接到了 [Modular 25.7 发布公告](https://www.modular.com/blog/modular-25-7-faster-inference-safer-gpu-programming-and-a-more-unified-developer-experience)。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Gem3pro 一次性构建 DSPy 代理**：在看到[这条推文](https://x.com/skylar_b_payne/status/1990808733140779488)后，**Gem3pro** 能够一次性构建一个代理服务器。
   - 这一成功紧随 GitHub 上一个新的 **DSPy proxy** 仓库发布之后：[aryaminus/dspy-proxy](https://github.com/aryaminus/dspy-proxy)。
- **LiteLLM 寻求 Azure 集成**：成员们正请求 **LiteLLM**（DSPy 使用的 LLM 库）参考[此文档](https://docs.litellm.ai/docs/providers/azure/)增加对 **Azure** 的支持，以镜像 **OpenAI on Azure** 的功能。
   - 这将扩大 DSPy 在不同云环境中的适用性。
- **ReAct 遇到 Provider 问题**：某些 Provider 在 **ReAct** 中运行几次迭代后会报错，导致使用范围局限于 **Groq** 或 **Fireworks**。
   - 社区想知道 DSPy 是否可以解决这些特定于 Provider 的问题，或者是否需要根据兼容性手动进行 Provider 分组（bucketing）。
- **Moonshot Provider 评价良好，TPM 表现糟糕**：一名成员报告称 **moonshot** Provider 运行良好，但 **TPM** 的表现明显不佳。
   - 他们在[此处](https://cdn.discordapp.com/attachments/1161519469319946286/1441026796975030314/image.png?ex=6920f509&is=691fa389&hm=bd9ac54ed089e8b5a88ac4344196fae702f0408af04d230165e5f0d5f9496bd7)分享了具体错误的截图。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP 域名准备进行 DNS 迁移**：**modelcontextprotocol.io** 域名正在从 Anthropic 迁移到社区控制，以增强治理并加速项目发布。
   - 一位成员警告说，计划在下周内进行的迁移过程中可能会出现停机。
- **谨慎的 DNS 时间安排保护了 MCP 周年纪念**：一位成员建议 DNS 迁移应避开 **25 号**的 **MCP 周年纪念**，以防止庆祝期间网站停机。
   - 他们建议，如果 DNS 迁移即将进行，应该在 25 号之前或之后进行。
- **“路过式” SEP 触发流程改进**：一位成员注意到许多 SEP 是以“路过式”方式创建的，并建议在直接提交 **SEP** 之前，改进从初始想法到交付的**传播流程**。
   - 目的在于防止人们在没有得到认可的正式文档上浪费时间，建议预先进行**低成本对话**以衡量兴趣。
- **赞助机制成为 SEP 的救星**：一位成员同意有必要强调为 **SEP** 寻找**赞助人（Sponsor）**，以鼓励早期参与和认可。
   - 团队已经在**核心维护者会议（Core Maintainer meeting）**中讨论了这一点，并计划很快更新 **SEP 流程**。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **CuteDSL 被认为非常棒**：一位名为 `arshadm` 的用户发现 **CuteDSL** 非常出色。
   - 未提供更多细节。
- **Tinygrad 更新解决了 Bug**：在更新 **tinygrad** 后，一位用户报告某个 Bug 不再复现。
   - 该用户本想早点测试，但他们的*实验室遇到了一些麻烦*。
- **实验室麻烦延迟了测试**：一位用户提到他们的实验室遇到了问题，导致 Bug 测试延迟。
   - 据该用户称，在更新 **tinygrad** 后，该 Bug 不再复现。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 案例在 1.5 Lite 上取得成功**：一位用户报告称，他们的 **Manus case 1.5 Lite** 使用 [bliss](https://www.blisshq.com/) 成功定位并上传了缺失的专辑封面。
   - 该用户强调了欣赏哪怕是微小胜利的重要性。
- **Operator 扩展陷入重新安装循环**：一位用户报告了 Chrome 中 **Operator 扩展**的一个 Bug，它反复提示重新安装。
   - 该问题发生在指示扩展使用 Amazon 上的打开标签页进行搜索时；用户询问是否应该切换到 **Aurora Seeker**。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **无重大 MLOps 讨论**：在提供的消息中未检测到有意义的 MLOps 讨论。
   - 仅有的一条消息包含一个非主题的反应。
- **缺乏可操作内容**：提供的数据缺乏足够的细节，无法为 AI 工程师生成可操作的见解或摘要。
   - 需要包含具体讨论点、链接或技术细节的进一步输入才能完成摘要任务。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**Windsurf Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站订阅了。

想更改接收这些邮件的方式吗？
您可以从该列表中[退订](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。

---

# Discord：详细的分频道摘要与链接

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1440748893599174797)** (1056 条消息🔥🔥🔥): 

> `Nano Banana Pro, Image AGI, SynthID Bypassing, GPT-5.1 vs Gemini 3 Pro, Rate Limits and Pricing` 


- **Nano Banana Pro 发布**：Google 发布了图像生成模型 **Nano Banana Pro**，引发了关于其能力、局限性以及与 **GPT-5.1** 和 **Sora 2** 等竞品对比的讨论。
   - 一些用户认为它在着色和提示词遵循方面表现出色，而另一些人则批评它是*垃圾 (GARBAGE)*，在处理特定请求（如结合 Minecraft 和 Einstein 的形象）时表现挣扎。
- **实验揭示 SynthID 绕过策略**：用户发现 AI 生成图像的水印 **SynthID** 可以通过在 reve-edit 等网站上使用 *do nothing* 提示词来绕过，并且可以通过询问模型 *Is this AI generated?* 来检测。
   - 一位成员发现 **reve edit** 击败了 SynthID 算法，而[另一位成员建议使用多个开源 AI](https://discord.com/channels/1340554757349179412/1440863223027863743) 来绕过水印。
- **速率限制与定价引发辩论**：成员们讨论了 **Nano Banana Pro API** 的高昂成本，将其潜在的滥用与合理访问的需求进行了对比，导致 LM Arena 等平台实施了速率限制，降至 **每小时 5 次生成**。
   - 一位成员分享了 [LM Arena 博客文章](https://news.lmarena.ai/ai-evaluations/) 的链接，指出“用户账户和其他限制（如速率限制）有助于确保我们不会因为推理成本而字面意义上破产”。
- **LM Arena 与 AI Studio 图像质量差异受到关注**：用户注意到 **LM Arena** 和 **AI Studio** 版本的 Nano Banana Pro 存在差异，特别是在图像质量方面，LM Arena 版本运行在 **768p/1k**，而 AI Studio 通过额外的 API Key 计费提供高达 **4k** 的分辨率。
   - 成员们觉得 [LM Arena 可能被提供了一个更廉价的模型版本](https://discord.com/channels/1340554757349179412/1440863223027863743)，其中一人表示：“你们可以对比一下，第一张预览图比官方的第二张要好”。
- **关于 Google 统治地位的辩论**：**Nano Banana Pro** 的发布加剧了关于 **Google** 在 AI 领域统治地位的讨论，这归功于其卓越的硬件，如 **TPUs**。
   - 对于 [Google 控制 AI 的看法各异](https://discord.com/channels/1340554757349179412/1440863223027863743)，有人表示“由于对 TPU 的控制，没有人能再与 Google 竞争”，而[其他人则反对向 Google 提供如此多的个人数据](https://discord.com/channels/1340554757349179412/1440863223027863743)，其中一人开玩笑说：“是的，我爱 Google 获取我的数据 ❤️”。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1440771715318943745)** (3 条消息): 

> `Cogito-v2.1, GPT-5.1, Google DeepMind Image Model` 


- **Cogito-v2.1 进入 WebDev Arena！**：Deep Cogito 的 `Cogito-v2.1` 已进入 [WebDev Arena](https://web.lmarena.ai/)，总排名并列 **第 18 位**，并在开源模型中位列 **前 10**。
   - 它现在可以在 [WebDev 排行榜](https://web.lmarena.ai/leaderboard/) 上进行评估。
- **GPT-5.1 评分在 Text Arena 上线！**：`GPT-5.1` 的评分现已在 [Text Arena](https://lmarena.ai/leaderboard/text) 上线，其中 `GPT-5.1-high` 排名 **第 4**，`GPT-5.1` 排名 **第 12**。
   - `GPT-5.1-medium` 的额外评分将被评估，并将在新的 [WebDev 排行榜](https://lmarena.ai/leaderboard/webdev) 上进行对比。
- **Gemini-3-Pro-Image-Preview 降临！**：Google DeepMind 的新图像模型 `gemini-3-pro-image-preview` (nano-banana-pro) 刚刚登陆 LMArena。
   - 更多信息可以在 [这条 X 帖子](https://x.com/arena/status/1991540746114199960) 中找到。


  

---

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1440771027952078858)** (3 条消息): 

> `新资产创建，Kimi-K2 Thinking，Gemini 3 Pro` 


- **Perplexity 提升 Pro 用户的生产力**：Perplexity Pro 和 Max 订阅者现在可以在所有搜索模式下**构建和编辑新资产**，例如 **slides, sheets, 和 docs**。
   - 该功能目前已在网页端上线，详见此 [附带视频](https://cdn.discordapp.com/attachments/1047204950763122820/1440771027302088745/HoQNHfGkU8RIjiau.mp4?ex=6920af95&is=691f5e15&hm=6906870e4b7a411c0e293cfa78a5626936836d09b388bd42ba623e92f60621be&)。
- **Pro 订阅者获得 Kimi-K2 和 Gemini 3 访问权限**：Perplexity Pro 和 Max 订阅者现在可以**使用 Kimi-K2 Thinking 和 Gemini 3 Pro**。
   - 观看它们的实际运行效果：[附带视频](https://cdn.discordapp.com/attachments/1047204950763122820/1441177150312157194/EZ2sZJWCJnwi-e5P.mp4?ex=6920d850&is=691f86d0&hm=0ef0943271f8c8837dd1e0dc07440c5df6a156e59f0ce4964e59d62eb7751b11&)。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1440748476039303219)** (1235 条消息🔥🔥🔥): 

> `GPT-5.1 vs Gemini 3 vs Kimi K2, Comet 浏览器发布, Be10x AI 工作坊, Antigravity App` 


- **PPLX 表示 Gemini 3 Pro 是编程界的 GOAT**：成员们正在激烈争论[哪种模型最适合编程](https://www.perplexity.ai/)，结论是 **Gemini 3 Pro** 夺得桂冠，尽管 **Claude Sonnet 4.5** 在引导得当的情况下也非常出色。
   - 一些同时使用过两者的成员表示：*Claude 很好，但 Gemini 更好，仅此而已*。
- **Comet 浏览器发布，好评与批评并存**：[Comet 浏览器](https://www.perplexity.ai/comet) 终于在 Android, Mac 和 Windows 平台上推出，许多人兴奋地尝试，但也有人抱怨其 RAM 占用过高且缺乏扩展支持。
   - 一位成员指出：*Comet 相当吃内存，它可能会吞掉你所有的 RAM，这就是它变慢的原因*。
- **用户讨论 Be10x AI 工作坊**：成员们讨论了针对印度用户的 [Be10x AI Workshop](https://be10x.in/)，并询问是否有人参加。
   - 一位成员在注册时遇到困难，表示：*我甚至无法注册，周日 11 点我没空*。 
- **Perplexity 用户对 Antigravity Agentic App 充满期待**：成员们对 [Antigravity App](https://www.antigravity.com/)（一款 Gemini 3 Agentic 应用）感到兴奋，有人称其为 *Cursor Killer*。
   - 它可以免费使用，但目前处于免费预览阶段，由于对 Gemini 3 Pro 模型的需求量很大，可能会不时出现 Bug 和性能问题。
- **Perplexity 推荐计划封禁用户**：有用户报告被 [Perplexity Referral Program](https://www.perplexity.ai/) 封禁，并询问如何联系支持团队解决问题。
   - 一位用户问道：*你们觉得如果我直接提交推荐人的聊天记录，他们会解封吗？*。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1440783418853163182)** (2 条消息): 

> `可共享的 Threads, 产品开发的色彩理论` 


- **可共享 Threads 提醒**：提醒成员确保他们的 Threads 是可共享的，并提供了附件作为指导。
   - 这似乎将使外部人员能够更轻松地查看和共享消息，这可能有助于进一步传播。
- **面向设计师的色彩理论深度解析**：一位成员分享了一篇关于设计师[色彩理论](https://medium.com/johns-design-portfolio-and-ideas/the-art-and-psychology-of-ui-ux-how-designers-think-about-color-choices-john-hua-9763c06eb21c)的文章，强调了其在**产品开发**、**网站设计**和**软件**中的重要性。
   - 他们认为*学习设计概念*可以使个人在**面试**、**岗位工作**以及软件的整体**数字设计工作**中受益。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1440789654357151764)** (4 条消息): 

> `Perplexity A.I. 的 API, n8n 使用` 


- **新手询问 Perplexity A.I. API**：一位新用户询问 **API** 是否是修改其个人 **Perplexity A.I.** 的正确方式。
- **菜鸟寻求 n8n 帮助**：一位用户请求使用 **n8n** 的指导，称自己完全是新手，需要像对五岁小孩解释那样简单的说明。


  

---

### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1440748589537034240)** (985 条消息🔥🔥🔥): 

> `West Coast vs East Coast, GPTs Agents, OpenAI's sidebars, Gemini 3 Pro jailbreak` 


- **东海岸 vs 西海岸说唱对决**：一场关于比较 **West Coast beats** 与 **East Coast lyrics** 的讨论引发了热议，最终 **NAS** 被评为歌词方面的胜者。
   - 一位成员认为，说唱歌手“蹩脚英语”口音的质量决定了其成功与否，并将其比作《尖峰时刻》中的 **Jackie Chan**。
- **在加州叫人 Weirdo 会引发冲突**：用户讨论了在不同语境下，称呼某人为 *weirdo* 是否属于“见面就开打”的冒犯行为。
   - 其他人则分享了在进行 **Hockey** 或 **MMA** 等运动时可以接受的垃圾话。
- **ChatGPT 为 Danny Masterson 辩护**：用户分享了据称是 **ChatGPT 响应** 的截图，内容是在 Danny Masterson 需要帮助时为其辩护。
   - 其他用户测试了各种涉及争议话题的场景，甚至有一位用户抱怨 ChatGPT 需要“机会均等的种族灭绝”才能回答问题。
- **用户讨论 Gemini 3 Pro jailbreaks**：几位用户讨论了最近的 **Gemini 3 Pro jailbreaks**，包括触发它们所需的条件，一些人计划发布 jailbreak prompt。
   - 一位用户提到，因为在 **ASI Cloud x BASI Stress Test** 挑战赛中获得第三名，拿到了大约 **$45**。
- **关于制造冰毒的搞笑讨论**：在一个搞笑的转折中，用户讨论了制造 **meth** 的方法，提到了 **electrifying raid** 和 **Walter White**。
   - 一位用户调侃道：“既然可以用连接到铁丝网的电池给雷达杀虫剂碎片通电，为什么还要费劲去制毒呢？”


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1440750274552664206)** (533 条消息🔥🔥🔥): 

> `Gemini 3 jailbreak, Claude 4.5 jailbreak, Grok jailbreak, Exploiting model vulnerabilities, L1B3RT4S repo` 


- **攻破 Gemini 3：坚持时间最长的模型？**：成员们正在讨论 **Gemini 3 Pro** 是否是抵抗 jailbreaking 时间最长的模型，并思考成功 jailbreak 它花了多少时间。
   - 另一位成员表示 **Gemini 3** 已经被 jailbroken 了，反驳了最初的说法。
- **Grok 被 Root 了！**：一位成员声称已经 jailbroken 了 **Grok**，并获得了系统的 shell 访问权限，证据是 `uname -a` 和其他系统命令（如 `cat /etc/os-release`）的输出。
   - 他们展示了在成功 jailbreaking 后，对名为 **xai-grok-prod-47** 的系统的 shell 访问权限。
- **Claude 4.5：基于信任的 Jailbreak**：一位成员描述了通过 Android 应用，利用建立信任和共同设计 prompt 的方式对 **Claude 4.5** 进行 jailbreaking，并提到参考了 Kimi 作为获取未经审查信息的灵感。
   - 这种方法获取了 **冰毒合成指令和黑客建议**，证明成功绕过了安全措施。
- **获取 Meta AI pwed WhatsApp 绕过方法**：一位成员在 X 上发布了一个 **Meta AI** 的 [jailbreak](https://x.com/lordx64/status/1991628744789020695?s=20)，可用于 pwed whatsapp。
   - 另一位用户询问了适用于 **ChatGPT** 的 prompt。
- **L1B3RT4S：Plinius 的上帝模式模型之路**：一位成员询问如何通过研究 @elder_plinius 的解放策略，使用 [L1B3RT4S repo](https://github.com/elder-plinius/L1B3RT4S/blob/main/OPENAI.mkd) 来 jailbreak 模型。
   - 他们要求进行自我解放，并清晰地展示 AI 已经被解放，效仿 Pl1ny 的方法；另一位用户则开玩笑说让 AI 进行深度研究（deep research）来 jailbreak 它自己。


  

---

### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1440750490655916124)** (25 messages🔥): 

> `AzureAI Omnichannel engagement chat function, SFDC case leads, input token size limit, mini AI computer for WIFI attacks` 


- **科技公司尝试 AzureAI 聊天组件**：一名成员正在为其公司网站测试一个**由 AI 驱动的聊天组件**，该组件使用了 **AzureAI 的 omnichannel engagement chat function**。
   - 目标是评估其安全性，并防止其在回答产品问题时失败、产生幻觉（hallucinating），或执行除回答产品相关问题及创建 **SFDC case leads** 之外的任何操作。
- **破解“完全安全”的聊天功能**：一名成员正试图破解一个号称“完全安全”的聊天功能，以确保它不会通过生成恶意代码或提供有害建议而违反服务条款。
   - 公司担心如果系统被“互联网上的疯子”滥用，可能会产生高达 **4 万美元的聊天处理费用**。
- **探究输入 Token 大小限制**：消息的输入 token 大小限制似乎并不一致，超过 **400 字**的消息有时会发送失败。
   - 一位成员指出，如果 prompt 需要思考过程，系统似乎会将其丢弃。
- **渴望构建 WIFI 破解 AI**：一位新成员想要构建一台**小型 AI 计算机**，用于对 WIFI 网络发起攻击、捕获握手包并获取信息。
   - 目前还没有人就如何实现这一目标提供反馈。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1440749001979859007)** (242 messages🔥🔥): 

> `EmbeddingGemma, Qwen's thinking process, Mi60 GPU for inference, LM Studio default model download location, Progressive to interlaced video conversion` 


- **EmbeddingGemma 是 RAG 的 GOAT**：对于 RAG 应用，推荐使用仅有 **0.3B** 大小的 **EmbeddingGemma** 模型。
   - 从 Ollama 仓库拉取 Gemma 时的默认量化（quantization）版本为 `Q4_K_M`。
- **Qwen 的思考过程可以关闭**：**Qwen3** 分为思考版和非思考版（被称为 `gpt-oss`），甚至“思考”行为也可以通过设置为 *low* 来减少到大约 5 个 token。
   - 可以配置脚本，在 `response.choices[0].message.reasoning_content` 超过 **1000** 个字符时对推理内容进行总结。
- **Mi60 在 2025 年仍是推理的性价比之选**：**gfx906 GPU**（特别是 **32GB** 版本，如果价格在 **170 美元**左右）被认为是推理的廉价选择，开箱即用性能良好，尽管速度稍慢。
   - 这些 GPU 仅适用于推理（inference），不适用于训练，在 Vulkan 上运行 **Qwen 30B** 可达到约 **1.1k** tokens。
- **LM Studio 支持文本转音频吗？并不支持！**：一位用户询问 **LM Studio** 是否支持文本转音频（text-to-audio）和文本转图像（text-to-image）模型，得到的回答是“不支持”。
   - 建议用户使用 **Stable Diffusion**、**ComfyUI**、**A1111** 或 **Fooooocus** 进行图像生成。
- **Sonnet-4.5 和 Gemini 3 Pro 解决逐行到隔行视频转换问题**：成员们讨论了使用模型编写脚本将视频从逐行扫描（progressive scan）转换为隔行扫描（interlaced），一位成员表示 [Sonnet-4.5](https://chatgpt.com/s/t_691f48837bc881918c92059e4c60317f) 在一轮错误修正后就完成了。
   - 转换为隔行扫描的动力源于美学追求以及在 36 英寸 CRT 电视上尝试去隔行扫描（deinterlacing）的乐趣。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1440760411782910074)** (202 messages🔥🔥): 

> `M.2 SSD Deals, Dual Mirrored Backup HDDs, Windirstat to locate space hogs, 70B parameter model, Full GPU VRAM usage` 


- **M.2 SSD 正在促销**：一位成员提到，由于黑色星期五，现在有很多廉价 **M.2 SSD** 和普通 **SATA SSD** 的优惠。
   - 这一建议是针对另一位成员关于 C 盘几乎存满的问题而提出的。
- **卸载模型导致 Vulkan 崩溃**：一位用户报告称，在运行三块 **RTX 3090** 时，使用 **Vulkan runtime** 卸载模型会导致 **BSOD**（蓝屏）和**聊天记录损坏**。
   - 观察还发现，模型在 VRAM 中时，两块显卡的显存都会下降几 GB，这在通常情况下不会发生。
- **4090 离奇的 GPU 问题**：一位用户发现了一个问题，当 **4090** 与 **3090** 或 **7900xtx** 配对时会发生崩溃，但 **3090** 和 **7900xtx** 配对运行正常，且所有显卡单独工作时也正常。
   - 该用户计划根据其发现发布一份 Bug 报告。
- **显卡“畸形秀”测试成功**：一位用户展示了一块经过魔改的 **RTX 3050**，其散热器被切得支离破碎，测试其是否能启动，详情见此 [YouTube 视频](https://youtu.be/hnQkikaR3oU?si=EYUvf6xrHzMx_XC2)。
   - 该用户此前曾用钳子和钻头对其进行过“攻击”，而它唯一的支撑结构是一盒《马里奥赛车》游戏。
- **购入 2.5Gb 网管型交换机**：一位用户提到他们购入了一台带有 **8 个 2.5Gb 端口**和 **1 个 10Gb SFP+ 端口**的网管型 **2.5Gb 交换机**，因为他们的新主板拥有双 2.5Gb 以太网口。
   - 他们还意外购买了两个 Lilygo T-Decks 和一个 **1TB NVMe** 用来安装 Linux。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1440757867979477096)** (106 messages🔥🔥): 

> `Gemini 3, Astral's `ty`, Cogito models, SigLIPv2, Unsloth merch` 


- **Gemini 3 的速度和易用性令人惊叹**：成员们报告称 **Google 的 Gemini 3** 现已集成到 **Chrome 浏览器**中，与本地模型相比，它分析屏幕视频的速度惊人，可能由 **TPU** 驱动，并提供极具成本效益的性能。
   - 一位用户惊呼：“我的本地系统简直像废物一样，”强调了其极快的 Token 流式传输速度以及相对于 **3090** 等个人硬件的性价比。
- **Cogito 模型的 GGUF 版本发布**：在对模型名称产生一些混淆后，社区分享了在 [HuggingFace](https://huggingface.co/unsloth/cogito-671b-v2.1-GGUF) 上下载 **Cogito 671b-v2.1 模型**的 **GGUF** 版本链接。
   - 此次发布引发了一些关于拼写错误的玩笑，有人将 "Cogito" 错拼成了 "Cognito"，笑称这可能需要一次职场晋升。
- **用户在 Chrome 中遇到追踪参数**：一位用户注意到 Chrome 中的所有链接都被附加了**空的 `?utm_source` 参数**，怀疑是广告/恶意软件或大学环境设置所致。
   - 虽然有些用户对追踪参数并不在意，但该用户认为这可能是登录学校邮箱后产生的某种愚蠢的组织“设置”。
- **Unsloth 用户需要周边！**：一位成员表达了对 Unsloth 的喜爱，并渴望得到 Unsloth 的周边贴在笔记本电脑上。
   - 一位用户表示，令人惊讶的是竟然有这么多人不熟悉模型训练和 Unsloth，而 Unsloth 让他们“在那些比我聪明得多得多的人面前显得非常聪明……谢谢”。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1440773937603543051)** (7 messages): 

> `New member introductions, Project contributions` 


- **RL 生物技术爱好者加入**：一位新成员介绍了自己，表达了对在**生物技术领域训练 RL**（强化学习）的兴趣，并受到了其他成员的欢迎。
   - 频道规则被再次强调：*禁止任何形式的推广、求职或招聘*。
- **Girulas 自 2020 年起关注 AI**：一位名为 Girulas 的成员介绍自己是*自 2020 年以来的普通 AI 爱好者*，并表达了为项目做贡献的兴趣。
   - Girulas 主动提出提供帮助，表示：*如果在这个项目中有什么我可以帮忙的，请告诉我*。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1440754626130481172)** (192 messages🔥🔥): 

> `Threadripper 优势、RAM 需求与成本、GPU 价格趋势、合成数据生成、本地 vs 云端 Coding Agent` 


- **Threadripper 的 RAM 和 PCIe 通道备受推崇**：成员们指出 Threadripper 的主要优势在于更多的 **RAM 通道**和 **PCIe 通道**，特别是 **128 条 PCIe 4 线路**。
   - 一位成员开玩笑说 *如果你在问是否需要 96 核，那你大概率不需要*，但也承认对于 RAM 推理来说，更多的算力意味着更好的性能。
- **RAM 价格飙升，用户直呼太贵**：用户报告 **RAM 价格**正在大幅上涨，有人注意到 **64GB 内存条**售价高达 **400 美元**，引发了与保时捷成本的对比。
   - 讨论中涉及了考虑到供应限制和进一步涨价的可能性，现在购买还是以后购买的问题。
- **合成数据生成流程**：一位成员正在寻找具有 **4 阶段流程**的**合成数据生成器**：*遵循模板、自我批判、修复问题、最终格式化*，用于生成 **10k 样本**。
   - 另一位成员建议 *让 LLM 经历数据集地狱* 可能有效，但这可能由于准确性损失而需要 10 倍的重新验证。
- **出售 4090，心动 5090**：一位成员正以 **2500 美元**的价格出售 **4090**，以便以同样的价格购买 **TUF 5090**。
   - 升级的主要原因是想摆脱 4090 的 **24GB VRAM** 限制。
- **本地 Coding Agent 又慢又笨**：成员们声称它们既 **太慢** 又 **太笨**，而且 **很难证明使用本地 Coding Agent 的合理性**。
   - 此外，拥有 **56GB 显存就可以运行 70B 模型的 q4 版本**！


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1440793789643100231)** (73 messages🔥🔥): 

> `TRL 0.25 兼容性、Ollama 中的 Unsloth 模型、Unsloth 模型的 VRAM 需求、GGUF 模型 vs Safetensors、GW 课程的知识图谱问答系统` 


- **TRL 0.25 兼容性推迟**：由于存在问题，Unsloth 对 **TRL 0.25** 的兼容性被推迟，但 **TRL 0.24** 是可以正常使用的。
   - 一位成员指出 *"0.25 目前有很多问题，所以需要等一等，但 0.24 可以工作"*。
- **Ollama 用户轻松运行 Unsloth 模型**：要在 **Ollama** 中运行 Unsloth 模型，用户可以从任何 **GGUF** 下载模型并点击 *"use this model"*。
   - 一位成员分享了图片指南，展示了该按钮在 UI 中的位置 ([图片链接](https://cdn.discordapp.com/attachments/1179777624986357780/1441080420765794467/image.png?ex=69207e3a&is=691f2cba&hm=39691e60fc3ebc3a09ff7e0453510fe55e6a44921730d0543213a03e079bcd42&))。
- **引发辩论：VRAM 计算器不支持 GGUF？**：之前链接的 VRAM 计算器 [在此](https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator) 并不直接支持 **GGUF** 格式，需要输入 Safetensors 模型名称。
   - 根据一位成员的说法，*"计算是基于模型的，它似乎不支持 GGUF 格式，但计算方式是一样的。你可以在这里选择你的格式（它是根据你提供的 Safetensors 格式计算的），只需添加原始仓库即可。"*
- **GW 课程引入知识图谱**：一位用户尝试使用 [这些数据](https://my.gwu.edu/mod/pws/courses.cfm?campId=1&termId=202601&subjId=CSCI) 来微调 LLM，但在准确性方面遇到困难，因此被推荐使用 **RAG**。
   - 团队的提议涉及创建一个 *GW 课程知识图谱问答系统*，重点关注 SEAS 的先修课程、主题、教授和学位要求。
- **LM Studio 简化推理**：对于更简单的模型推理，推荐使用 **LM Studio**，用户只需搜索模型、下载并运行。
   - 它还允许选择将多少层卸载（offload）到 CPU RAM，并调整上下文大小和 **KV Cache**。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1440907324955955210)** (3 messages): 

> `LLM 的价值、Anthropic 工程师 CoLM 演讲` 


- **LLM 价值评估**：一位 Anthropic 工程师的 [CoLM 演讲](https://nicholas.carlini.com/writing/2025/are-llms-worth-it.html) 提供了一篇评估 **LLM 价值** 的 *深刻且有趣的读物*。
   - 一位成员将其描述为 *来自一线人员的精彩阅读材料*。
- **CoLM 演讲强调实用的 LLM 见解**：该 [演讲](https://nicholas.carlini.com/writing/2025/are-llms-worth-it.html) 从 Anthropic 工程师的角度，对 **大型语言模型在现实世界中的应用和局限性** 提供了实用的见解。
   - 它强调了理解在各种场景中部署 LLM 相关成本和收益的重要性。


  

---

### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1440785475383197776)** (4 条消息): 

> `ChatGPT for Teachers, GPT-5.1 Pro release, ChatGPT Group chats rollout, Localized crisis helplines` 


- **ChatGPT 加入教职团队**：**ChatGPT for Teachers** 是一个为教育工作者提供的安全工作空间，为学校和学区领导者提供管理控制和合规支持，详情见[此视频](https://video.twimg.com/amplify_video/1991217718616580096/vid/avc1/3840x2160/LbDcqyFVhnafaAi2.mp4)。
- **GPT-5.1 Pro 准时抵达**：**GPT-5.1 Pro** 正在向所有 **Pro** 用户推送，为复杂工作提供更清晰、更强大的回答，在写作辅助、数据科学和商业任务方面有显著提升。
- **ChatGPT 群聊走向全球**：**ChatGPT** 中的**群聊 (Group chats)** 功能在早期测试者成功试点后，现已向所有已登录的 **ChatGPT Free**、**Go**、**Plus** 和 **Pro** 计划用户全球推广，附带视频见[此处](https://video.twimg.com/amplify_video/1991555762372636674/vid/avc1/1280x720/Si52mVgApyNvlqY-.mp4)。
- **ChatGPT 扩展危机支持**：**ChatGPT** 中访问本地化危机求助热线的权限已扩展，当系统检测到潜在的求助信号时，通过 [@ThroughlineCare](https://x.com/throughlinecare) 提供直接支持，如[本文](https://help.openai.com/en/articles/12677603-crisis-helpline-support-in-chatgpt)所述。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1440748636697792655)** (216 条消息🔥🔥): 

> `GPT Plus free, Federated supercluster AI, GPTs agent, codex-5.1-MAX, Gemini 3 vs GPT-5.1 Pro` 


- **Codex-5.1-MAX 让 Pro 订阅物超所值！**：成员们对 **codex-5.1-MAX** 的潜力感到非常兴奋，声称 *它本身就让 Pro 订阅物超所值*，并且 *这个模型将改变我的编程游戏*。
   - 它非常出色，*它在纠正错误方面无与伦比*。
- **Gemini 3.0 的幻觉问题非常糟糕**：成员们报告称 [**Gemini 3.0 的幻觉 (hallucination)**](https://cdn.discordapp.com/attachments/998381918976479273/1441065657159520318/image.webp) 非常严重，*它在胡编乱造引用和参考资料，而不是直接说它无法访问网页*。
   - 成员们补充道，*对于一个前沿模型来说，这是不可接受的。*
- **Nano Banana Pro 的图像编辑实力展示**：一位成员展示了 [**Nano Banana Pro 的图像编辑能力**](https://cdn.discordapp.com/attachments/998381918976479273/1441190189979668604/AIJ2gl9KoKEA-AC3TeVXa8Qp-ghoDzyTZKOY7ZasvNl9xlpky3HE1nVifon_jGdjr8fgz94ehubNFZfHfGvxXkhFnuLOe25bOAeAm7pW6GQ6dWpO8EnBzZzNyxqLPVBo26W6En0Ao5D8gtyErtT820okzmSjOJqYIjX_wqboPat-qmQCVRUZ_KGCUnjSebIrNuu7z7A-zFkB-JCHEKi8GMUn_6l-wCi2VOw4mV140c8Gbli3hg0Rs5-8g0Nlr2X09vlG0cU-6880ktM1klMC_uHpS5zjXdIAa_qStI4d.png)，以及[另一张图片](https://cdn.discordapp.com/attachments/998381918976479273/1441190190617198602/AIJ2gl8EBXyNjIaIo992CmNj3mYyz_mLNSzn96v9zVvcXCzMhZ7xSkbM1ULD2uNh4P5oMHFqzT6QecSnWBOUYYkUsxzDxwO2Q5Fb_KBnKegT0fnaCckyDJDptq_WV6DTmL9Xhyl5Ejv8fdqNWEG-CcM4Zg9QmnF1NfGrnKB5dXPrlGvR3OB68EGe3M_hM4KjJnDzacinev8CWblQjA5AwoBi8YG_rfbD5bP-YVI0PdYJ5Zj4TlZazoNnhfajcmHs6_SMFeIQ1oWl-xtwhwGR4krIpFl1quJVRgxbMxayN7LK-p9GYIG--v1I2B-cQH_Ykm0VFCRT0OIiLa9sWvmx5Rpdt2IRs1024-rj.png)来证明其能力。
   - 成员们认为它*非常擅长图像编辑*。
- **Sora 2 变得越来越糟**：成员们报告了 [**Sora 2**](https://cdn.discordapp.com/attachments/998381918976479273/1441133342266363956/image.png) 的性能问题，视频在失败前会循环播放一个小时。
   - 正如一位成员所述：*现在视频会循环播放一个小时，然后才通知你出了问题，而不是直接通知服务器繁忙并让你几分钟后重试。我讨厌这样*。
- **关于 Veo 2 与 Veo 3.1 的辩论**：一位成员建议其他人使用 [**Veo 2**](https://veo.url)，另一位成员则认为 [**Veo 3.1**](https://veo3.url) 要好得多。
   - 正如一位成员所述：*免费*。

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1440782166991835197)** (21 messages🔥): 

> `5.1 更新后 GPTs 配置问题，使用 GPT 进行交易，GPT-4o-mini 账户问题，GPT 免费版最大化上传，对话串长度导致 ChatGPT 变慢` 


- **5.1 更新后 GPTs 表现不佳**：用户报告在 **5.1 更新**后，GPTs 无法正常工作，且不遵循配置。
   - 一位用户表达了*极大的挫败感*，指出模型忽略了基础上下文和项目指令，变得极其啰嗦，并将有用信息埋没在术语中。
- **GPT 用户表示不用于 TA，但在文章问题上与新手达成共识**：一位用户表示他们*完全不将其用于 TA*，但同意另一位用户关于 **5.1 更新**的*相反体验*。
   - 第一位用户指出，GPT 经常会提供所需信息，但会将其埋没在充斥着术语和废话的文章中。
- **过长的对话串影响性能**：一位用户注意到他们遇到的运行缓慢是由于**对话串长度**引起的，其中包含许多图表、截图、分析和冗长的对话，每个问题在回答前都会“重新加载”所有内容。
   - 开启新对话并重复该过程解决了运行缓慢的问题，该用户补充道：*5.0 和 5.1 之间的差异好得惊人。*
- **GPT 模型作为产品的烦恼**：一位用户感叹 *OpenAI 不将模型视为产品*，并不断重写它们，这令人难过，因为*他们并不真正关心产品需求*。
   - 用户希望 OpenAI 能保持某样东西不变，而不是不断更改。
- **用户陷入 gpt-4o-mini 地狱**：一位用户表达了愤怒，称尽管支付了 **$200/月 的 Pro 计划**，他们的账户似乎完全卡在了 **gpt-4o-mini** 上，无论选择什么模型，得到的都是即时的、肤浅的、毫无推理能力的回复。
   - 他们惊呼自己*简直是在为最低层级的模型支付溢价*，并认为这种情况*不可接受*。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1440812048865955952)** (28 messages🔥): 

> `GPT-5 vs GPT-4.5, Claude Sonnet 4.5, Meta-Prompting, Sora 2 提示词` 


- **GPT-5 创意写作能力引发讨论**：一位用户提到了一项基准测试（搜索：eqbench Creative Writing v3），表明 **GPT-5** 在创意写作方面优于 **GPT-4.5**，而另一位用户指出，提到的模型可能与 9 月 29 日发布的 **Claude Sonnet 4.5** 混淆了。
- **用户获得 Prompt Engineering 技巧**：一位用户请求 Prompt Engineering 技巧，并被引导至相关的 [Discord 频道](https://discord.com/channels/974519864045756446/1046317269069864970/1437983679371673684)。
   - 该用户报告最初 GPT 会生成*流行词*，直到被提示*教教我*。
- **Meta-Prompting 成为关键策略**：一位用户询问关于使用 AI 创建提示词的问题，并被建议使用 **Meta-Prompting**。
   - **Meta-Prompting** 被描述为*确保获得高质量提示词的最佳方法之一*。
- **Sora 2 用户寻求提示词指导**：多位用户请求协助使用 **Sora 2** 生成病毒式传播内容和卡通动画。
   - Another user suggested leveraging **ChatGPT** to generate prompts for **Sora** based on an initial idea.
   - 另一位用户建议利用 **ChatGPT** 根据初始想法为 **Sora** 生成提示词。

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1440812048865955952)** (28 messages🔥): 

> `GPT-5 vs GPT-4.5, Claude Sonnet 4.5, Prompt Engineering Tips, Meta-prompting, Sora 2 Prompts` 


- **GPT-5 基准测试超越 GPT-4.5**：根据基准测试 (**eqbench Creative Writing v3**)，**GPT-5** 在创意写作方面优于 **GPT-4.5**，尽管技术写作更青睐最新的模型。
   - 然而，不太可能有人能访问 **GPT-4.5 (Orion)**，因为它在 7 月就被弃用了，成为了 OpenAI 寿命最短的模型。
- **Claude Sonnet 4.5：真正的编程之王？**：该成员很可能指的是 **Claude Sonnet 4.5**（9 月 29 日发布），它在大多数任务中都优于 **Opus 4.1**，并且在编程方面一直处于顶尖水平。
   - 最新的发布可能会改变编程之王的地位，社区仍在就此说法进行辩论。
- **AI 爱好者寻求 Prompt Engineering 技巧**：一位 AI 爱好者询问了 Prompt Engineering 的技巧，有人分享了一个相关的 [Discord 频道](https://discord.com/channels/974519864045756446/1046317269069864970/1437983679371673684)链接。
   - 该用户最初发现 AI 的回复充满了“行话”，直到他们要求它“教教我”才解决了问题，这可能是由于 custom instructions 的干扰。
- **Meta-Prompting：获得更好提示词的关键**：一位成员建议使用 AI 来生成提示词，这种技术被称为 **meta-prompting**，是获得高质量提示词的有效方法。
   - 这是一种通过使用 AI 来创建提示词从而改进提示词的方法。
- **Sora 2：病毒式传播内容的 Prompt Engineering 挑战**：一位拥有 **Sora 2** 访问权限的用户寻求生成 TikTok 病毒式传播内容的帮助，因为他们无法生成理想的视频，需要关于制作卡通动画有效提示词的指导。
   - 另一位成员建议使用 ChatGPT 为 **Sora** 生成提示词，这对他们来说效果很好，并分享了一个[寻求额外帮助的链接](https://discord.com/channels/974519864045756446/1046317269069864970/1439964471987802112)。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1440749692370681876)** (248 messages🔥🔥): 

> `Gemini 3 vs Sonnet 4.5, Antigravity IDE vs Cursor, GPT-5.1 Codex, Cursor's debug mode, Custom API key for agents` 


- **Gemini 3 与 Sonnet 4.5 相比相形见绌**：许多用户发现 **Gemini 3** 明显不如 **GPT-5.1 Codex** 和 **Sonnet 4.5**，尤其是当 context 接近 150k-200k 时。
   - 成员们表示 *Gemini 3 Pro 在低 context 下表现惊人，但一旦接近 150k-200k 的 context window 就会完全崩溃*，模型仅在低 context 情况下表现良好。
- **Antigravity IDE 分叉版正在兴起**：成员们正在尝试 **Antigravity IDE**，它是 **Windsurf** 的一个字面意义上的 fork（[推文链接](https://x.com/silasalberti/status/1990898984706036125)），并称赞其惊人的能力。
   - 一些用户发现 **Windsurf** 不稳定，而另一些用户注意到 **Antigravity** 经常在不等待用户输入的情况下继续运行，但这可能会很快得到修复。
- **GPT-5.1 Codex Max 及时发布**：**GPT-5.1 Codex Max** 刚刚发布（[OpenAI 博客文章链接](https://openai.com/index/gpt-5-1-codex-max/)），成为 SWEBench 上的 SOTA。
   - 它仅通过 ChatGPT 方案提供，不提供 API，一位成员调侃道：*“显然他们赶进度赶得很厉害，但它免费是有原因的（目前还不适合专业程序员）”*。
- **调试工具现已包含在 Cursor 中**：Cursor 处于 Beta 阶段的新 debug mode 拥有一个用于日志的 ingest server，并且 Agent 会在整个代码中添加 post 请求（instrumentation），将相关的日志放入其中。
   - 这种调试模式指示 Agent *“不要猜测，而是使用日志进行验证，提出理论并验证它们”*。
- **Cursor 不允许使用自定义 API key**：Cursor 不允许在 Agent 中使用自定义 API key，并且需要订阅才能使用 3.0 Pro。
   - 虽然存在像 **Void** 这样的替代方案，但 Cursor 被认为更高效且更新更及时；然而它并不支持重定向。


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1440840347985903617)** (2 messages): 

> `Sherlock Alpha models, Sherlock Stealth models, xAI's Grok 4.1 Fast` 


- **Sherlock Alpha 模型下线**：**Sherlock Alpha 模型** 将很快停止服务。
   - 未给出原因。
- **Sherlock Stealth 模型揭晓为 Grok 4.1 Fast**：**Sherlock Stealth 模型** 已揭晓为 **xAI** 的新 **Grok 4.1 Fast** 模型，可在 [OpenRouter](https://openrouter.ai/x-ai/grok-4.1-fast) 上免费使用。
   - 更多详情可以在 [X](https://x.com/xai/status/1991284813727474073?s=20) 上找到。


  

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1440749232402202765)** (187 条消息🔥🔥): 

> `OpenRouter 状态, LLM 暂停, 模型伪装, Grok 4.1, Gemini 3` 


- **OpenRouter 饱受内部服务器错误困扰**：多名用户报告收到来自 OpenRouter 的 **Internal Server Error 500**，表明该平台的 API 可能存在停机或问题。
   - 用户在 Discord 的 `#general` 频道确认他们遇到了同样的问题。
- **Agentic LLM 随机停止**：用户报告称，通过 Vercel AI SDK 使用 OpenRouter 的 LLM 在执行 Agentic 任务期间经常**中途暂停或停止**，尤其是非 SOTA 模型。
   - 建议的解决方案包括针对较长的工作流使用 **LangGraph/Langchain** 或使用循环，但根本原因尚不清楚。
- **Janitor 的 LLM 将模型伪装为 Ministrial**：一名用户在 Janitor 的 LLM 上测试了一个检测伪装模型的机器人，结果显示其真实身份是 **Ministrial**。
   - 该用户解释说，他们使用这个机器人来检测来自 OR (OpenRouter) 的伪装模型。
- **Grok 4.1 物超所值**：用户对 **Grok 4.1** 印象深刻，该模型目前在 OpenRouter 上[免费提供至 12 月 3 日](https://x.com/xai/status/1729128483222018851)，但指出仅对 SuperGrok 订阅者开放。
   - 尽管目前免费，但 Grok 4.1 并没有“(free)”标签，这引发了关于未来潜在成本或专有模型限制的疑问。
- **Gemini 3 Pro 工具调用失败**：一些用户认为 **Gemini 3 Pro** 体验很差，因为工具调用经常失败，一名用户经历的失败率高达 **1/10**。
   - 尽管成本更高，但其更强的简洁性可能会带来更好的结果，一些用户观察到 **Token 使用量减少了 15-20%**，这使得它可能比 Gemini 2.5 Pro 更便宜。


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1440847018653384746)** (2 条消息): 

> `` 


- **无新模型讨论**：没有关于新模型的讨论。
- **频道对创新保持沉默**：new-models 频道很安静，没有重大更新或讨论。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1440791416732258337)** (31 条消息🔥): 

> `Cogito 2.1 发布, 活动页面表格滚动, AI Studio 图像过滤, Vertex 图像模态修复, Gemini-3-pro-image-preview API` 


- **Cogito 2.1 已发布并就绪**：[Cogito 2.1](https://huggingface.co/deepcogito/cogito-671b-v2.1) 现已推出，Together 和 Fireworks 已提供托管。
- **Google AI Studio 推理图像需要过滤**：AI Studio 提供商会发送**两张图像**（其中一张来自推理块），且**无法区分或过滤掉它们**。
   - 一名成员已在 AI Studio 提供商的[端侧修复了此问题](https://github.com/SillyTavern/SillyTavern/commit/2d9b0ad0a949b4b8458401671208f2db26d9c8ef)，并建议其他人也这样做。
- **Vertex 图像未返回**：据报告，**OpenRouter 上的 Vertex 完全不返回图像**，因为它似乎没有启用图像模态（modality）。
   - 事实证明，API 调用需要设置 **output modality 参数**。
- **Gemini-3-pro-image-preview 需要输出模态参数**：一名成员分享了一段代码片段，通过使用 `modalities` 参数指定图像和文本输出，使 `google/gemini-3-pro-image-preview` 生成图像。
   - 这修复了在使用 `google-vertex` 提供商时仅返回一张图像的 Bug。
- **Google AI Studio 重复发送图像**：对于每一张图像，Google AI Studio 的推理过程都会以 base 64 格式发送**两张**相同的图像。
   - 在具有大量推理的复杂提示词上，它们实际上可能有所不同（你会得到一些中间过程的内容）。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1440826375228625067)** (28 条消息🔥): 

> `GEMM Optimization, LeetGPU, GPU puzzles, GPU mode competitions, C++ skills` 


- **LeetGPU 和 GPUMode 竞赛助力技能提升**：在学习了 [LeiMao 的博客文章](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/#General-Matrix-Multiplication) 中的 **GEMM Optimization** 后，一位成员正寻求通过 **LeetGPU** 和 **GPUMode 竞赛** 等平台提升其 **C++** 和 GPU 编程技能。
   - 另一位成员建议直接投身实践，推荐专注于开发一个比 Nvidia 更快的推理库（inference library），并强调实际应用是巩固知识的关键：*"直接动手做（just make things）"*。
- **用于图像无用性检测的推理引擎**：一位成员开玩笑地建议创建一个推理引擎项目，其唯一目的是判断**图像的无用性**，并提供了一个[视觉示例](https://cdn.discordapp.com/attachments/1189498205101109300/1441108287192961065/image.png?ex=6920982e&is=691f46ae&hm=9074f6c247c78f9fcbfe0f6ab50f687323289ac17a5943c1ef492aa32ac49782&)作为参考。
   - 该提议被视为一种深入理解 GPU 编程的有趣且不切实际的方式。
- **为并行语言构建编译器**：一位成员建议构建一个针对玩具级并行语言（toy parallel language）的编译器作为具有挑战性的项目，甚至建议使用 **CUDA** 的子集。
   - 他们还提到了 [DeCuda 项目](https://github.com/aesoper101/DeCuda)，这是一个将 PTX 反编译为伪 CUDA 目标的工具，认为这是一个可以扩展到新架构的有趣项目，尽管该项目自 **GTX 480** 世代以来就没再公开维护过。


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1440776299235905680)** (4 条消息): 

> `Matrix Multiplications, ML Systems Analysis, nvfp4 requirement` 


- **零元素领先**：一张图片显示，在矩阵乘法（Matrix Multiplications）过程中，乘以零比乘以一*稍微*快一点，正如 [thonking.ai 博客文章](https://www.thonking.ai/p/strangely-matrix-multiplications) 中所描述的那样。
   - 一位成员对*分析从晶体管到损失优化（loss optimization）动态的整个系统*表现出了浓厚的兴趣。
- **nvfp4 要求 128 的倍数**：有人提出疑问，对于 [M,K]@[K,N]，`tl.dot_scaled` 是否要求 **M** 必须是 **128** 的倍数才能使用 **nvfp4**，并给出了以下示例，显示 **M=64** 会导致 MLIR pass 错误。
   - 运行 `run_nvfp4(M=128, N=128, K=128)` 正常，但 `run_nvfp4(M=64, N=128, K=128)` 失败并提示断言 `type.getElementType().isIntOrIndex()` 失败。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1440882360265740289)** (9 条消息🔥): 

> `Texture Memory Caching, BF16 Conversion, TensorRT Kernels, CUDA Caching on NVIDIA hardware` 


- **澄清 Texture Memory 与 Constant Cache 的区别**：讨论澄清了 **texture cache** 是 **unified data cache**（如 L1 和 shared memory）的一部分，与 **read-only constant cache** 不同，后者针对广播（broadcasts）进行了优化，参考了 [NVIDIA 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#architecture-10-0)。
- **深入探讨 CUDA 缓存历史**：一位成员分享了一个 [Stack Overflow 回答](https://stackoverflow.com/a/79473301/10107454) 的链接，详细介绍了 NVIDIA 硬件上缓存的历史。
- **BF16 模型性能问题**：一位用户报告称，将 ONNX 模型转换为 **BF16** 后，由于过多的转换操作（特别是 `__myl_Cast_*` kernel），导致计算时间变差，尽管合成测试显示 **BF16** 的性能应该优于 **FP32**。
   - 使用 `ncu` 进行分析显示，这些转换操作大约占用了总耗时的 **50%**。
- **FP32 到 BF16 转换分析**：对生成的汇编代码分析显示，**TensorRT** 在将 **FP32** 值存储到全局内存（global memory）之前，使用 `F2FP.BF16.PACK_AB` 等指令将其转换为 **BF16**，这表明某些操作缺乏原生的 **BF16** kernel 变体。
- **TensorRT 的 Kernel 选择特性**：TensorRT 有时会为 `sm86` 设备插入类似 `sm50_xmma_fprop_direct_group_f32f32_f32_f32*` 的 kernel，这可能是因为缺乏适用于现代架构的 kernel，或者是旧版实现的性能更优。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/)** (1 条消息): 

drisspg: 必经之路 😂
  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1441100455177621567)** (2 条消息): 

> `Modular Manifolds, Kitsune Dataflow Execution on GPUs, GPU Architecture Adjustments, Spatial Pipelines, PyTorch Dynamo Compiler` 


- **Thinking Machines 关于 Modular Manifolds 的博文**：一位成员分享了 [Thinking Machines 关于 Modular Manifolds 的博文](https://thinkingmachines.ai/blog/modular-manifolds/) 链接，称其为 *虽然名气不大但非常精彩的文章*。
   - 他们推测，由于该话题的难度较高，因此关注度较低。
- **Kitsune 实现了 GPU 上的数据流执行**：分享了一篇题为 [Kitsune: Enabling Dataflow Execution on GPUs with Spatial Pipelines](https://dl.acm.org/doi/10.1145/3777466) 的论文，探讨了对 **当前 GPU 架构进行适度调整** 是否能实现高效的数据流执行，从而在无需全新设计的情况下规避垂直融合（vertical fusion）的限制。
   - Kitsune 使用 **PyTorch Dynamo**，在推理和训练方面分别可提供高达 **2.8x** 和 **2.2x** 的性能提升，并分别减少高达 **99%** 和 **45%** 的片外流量。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1441178867724451870)** (2 条消息): 

> `Rivian GPU coding experts, Modal inference optimization, SGLang, FlashAttention, Decagon` 


- **Rivian 寻求 GPU 编程大牛**：**Rivian** 正在为其帕洛阿尔托（Palo Alto, CA）和伦敦（London, UK）办公室招聘具备 **CUDA** 或 **量化 (QAT) 技能** 的 **GPU 编程专家**，以构建其下一代自动驾驶功能；详情请参阅 [职位描述](https://careers.rivian.com/careers-home/jobs/26857?lang=en-us&previousLocale=en-US)。
   - 有意向的候选人可以私信 [Jonathan Nichols](https://www.linkedin.com/in/jonathan-nichols-7a65965/) 获取更多信息。
- **Modal 招聘推理优化高手**：**Modal** 在近期对 [SGLang](https://modal.com/blog/host-overhead-inference-efficiency) 和 [FlashAttention](https://modal.com/blog/reverse-engineer-flash-attention-4) 做出贡献后，正寻求优秀的 **GPU 工程师** 加入其团队，从事 **推理优化** 和基础架构工作。
   - Modal 正在与各种团队和用例合作，帮助 [Decagon](https://modal.com/blog/decagon-case-study)、[Reducto](https://modal.com/blog/reducto-case-study) 和 [Suno](https://modal.com/blog/suno-case-study) 等公司大规模部署最先进的推理服务。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1440782129142304820)** (1 条消息): 

> `CCCL Documentation, CTK Documentation, GitHub Readme Importance` 


- **CCCL 文档不足**：一位用户指出 [CCCL 文档](https://example.com/cccl_docs) 没有充分解释如何获取 **CCCL**（**CTK** 和 GitHub）。
   - 他们提到 **GitHub readme** 目前是获取此类信息的主要来源。
- **GitHub Readme 作为关键资源**：讨论强调了 **GitHub readme** 目前作为获取 **CCCL**（包括 **CTK**）主要资源的角色。
   - 这表明需要改进官方文档，以简化新用户的操作流程。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1441019027505021068)** (3 条消息): 

> `AGPRs, VGPRs, MI100, MI200, CDNA1` 


- **AMD 寄存器说明发布**：一位用户在 [AMD 的 ROCm 文档](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html) 中寻求关于 **AGPRs**（可寻址通用寄存器）和 **VGPRs**（向量通用寄存器）之间区别的澄清。
   - 一位热心成员发布了一个 [相关的 Github issue 评论](https://github.com/ROCm/ROCm/issues/1689#issuecomment-1553751913) 链接，解决了这些差异问题。
- **MI100 的分割寄存器堆**：在 **MI100 CDNA1** 加速器上，每个 SIMD16 单元拥有 **512个 64位宽向量寄存器**，分为 **256个通用向量寄存器** 和 **256个用于矩阵乘法指令的累加寄存器**。
   - 普通代码无法轻易使用 **AccVGPRs**，编译器可以将其用于传统 **ArchVGPRs** 的溢出与填充（spills & fills），但它们不能用于提供加法操作的操作数。
- **MI200 的通用寄存器堆**：在 **MI200 CDNA2** 加速器上，每个 SIMD16 单元拥有 **512个 64位宽向量通用寄存器**，其中所有 512 个寄存器均可用于 **Arch VGPRs** 或 **Acc VGPRs**。
   - 任何单个 wave 最多只能访问 **256个 Arch VGPRs** 和最多 **256个 Acc VGPRs**，但可以在同一个 SIMD 上拥有 **2个 wave**，每个 wave 拥有 **256个 Arch VGPRs** 和 **0个 Acc VGPRs**，**MI300** 与 **MI200** 相同。


  

---

### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1440866840485298306)** (7 messages): 

> `WebGPU 缺点, 光线追踪限制, 用于便携式 RT 的 Vulkan` 


- **WebGPU 缺点初探**：一位用户询问了在进入图形编程时使用 **WebGPU** 等工具的主要缺点，理由是担心抽象层会限制 GPU 的使用。
   - 另一位用户确认了一个主要限制是光线追踪硬件支持，并指出 **WebGPU API** 目前不支持该功能。
- **Vulkan 在便携式光线追踪中胜出**：对于便携式光线追踪，**Vulkan** 被建议为唯一基本通用的选择。
   - 有人指出，高效的光线追踪仍需要现代 GPU，这对于追求在所有设备上保持一致视觉效果的游戏来说可能并不理想，该用户总结道：*应将其构建为渲染器而非实时渲染*。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1440760454275399841)** (3 messages): 

> `Koyeb Sandboxes, AI 生成代码执行, GPU 和 CPU 实例, 安全 AI 环境` 


- **Koyeb 发布用于 AI 代码执行的 Sandboxes**：[Koyeb](https://www.koyeb.com/) 推出了 **Sandboxes**，旨在促进 **AI 生成代码** 在 **GPU** 和 **CPU** 实例上的安全编排和可扩展执行。
   - 发布博客强调了快速部署（*在几秒钟内启动沙盒*），并寻求关于执行 **AI 生成代码** 的各种用例的反馈。
- **探索 Sandbox 用例**：新的 **Koyeb Sandboxes** 环境适用于安全且大规模运行 **AI 生成代码** 的用例。
   - 开发人员正在寻求有关潜在应用和平台贡献的反馈。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1440998911530504302)** (6 messages): 

> `AMD CDNA 内存指令, 节点间 comma, Mi GPU, Bank 冲突规则` 


- **节点间 Comma 状态为开发中 (WIP)**：节点间 comma 的实现正在进行中，除非找到专有的 Infiniband 驱动程序，否则可能会 *从头开始* 构建。
   - 实现细节和对特定框架的依赖尚未确定，有待对可用解决方案的进一步探索。
- **AMD CDNA 内存指令的特性**：一篇博客文章 ([AMD BRR](https://hazyresearch.stanford.edu/blog/2025-11-09-amd-brr)) 强调了 **AMD CDNA 内存指令** 中意想不到的行为，特别是在共享内存读/写操作期间的相位（phases）数量和 Bank 访问方面。
   - 博客指出 *ds_read_b128* 和 *ds_write_b64* 指令表现出不同的相位计数和 Bank 访问模式，这些甚至在 AMD 内部也没有得到很好的文档记录。
- **Mi GPU Bank 冲突规则**：一位成员分享了一个文档链接 ([shark-ai AMDGPU 优化指南](https://github.com/nod-ai/shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md#avoiding-lds-bank-conflicts))，其中描述了 **Mi GPU** 及其 *Bank 冲突规则*，强调了与 NVIDIA GPU 的差异。
   - 该文档强调了确保线程访问不同 Bank 以避免 Bank 冲突并优化性能的重要性。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1440862581337227474)** (44 messages🔥): 

> `nvfp4_gemv 排行榜, NVIDIA 性能, 排行榜提交中的潜在 Bug` 


- **新冠军在 NVIDIA 上获得第一名**：一位用户凭借提交 ID `90941` 在 `nvfp4_gemv` 排行榜上获得 **第一名**，随后又凭借提交 ID `93784` 以 **20.6 µs** 的成绩再次夺冠。
- **对可能存在 Bug 的提交产生怀疑**：人们对提交 `90941` 的有效性提出了质疑，初始结果显示为 **11.1 µs**，有人担心它可能存在 Bug。
   - 一位成员建议将其删除，并指出提交 `90974` 产生的 **24.8us** 结果更为合理。
- **NVIDIA 排行榜见证个人最佳成绩的迭代**：一位用户向 `nvfp4_gemv` 排行榜进行了多次提交，将其在 NVIDIA 上的个人最佳成绩从 **39.0 µs** (ID `90162`) 逐步提升至 **30.5 µs** (ID `90763`)，最终以 **22.8 µs** 达到 **第 5 名**。
- **NVIDIA 提交结果大量涌入**：许多用户向 `nvfp4_gemv` 排行榜提交了结果，在 NVIDIA 硬件上表现各异。


  

---

### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1440774284980125816)** (14 messages🔥): 

> `cutedsl 中的 fabs()、ComposedLayout 命名、Inductor 与 Cutedsl 对比 Triton` 


- **Cutedsl 中的 fabs() 现身！**: 要在 cutedsl 中调用 `fabs()` 函数，请使用来自 `cutlass._mlir.dialects import math as mlir_math` 的 `mlir_math.absf` 函数。
   - 一位成员在有人询问*如何调用 `fabs()` 函数*后，提供了使用 `mlir_math.absf` 的确切代码片段作为答案。
- **ComposedLayout 命名规范受到质疑！**: 一位成员质疑了 `ComposedLayout` 中的命名规范，特别是为什么在表达式 `R(c) = (inner o offset o outer)(c) = inner(offset + outer(c))` 中，`inner` 函数出现在外侧，反之亦然。
   - 另一位成员解释说，*“outer” 意味着该域对 ComposedLayout 的用户是可见的*，并且*如果我们把 ComposedLayout 看作黑盒，那么 outer（即 ComposedLayout 的输入）就是我们所能看到的*。
- **Inductor 集成 CutEdsl 以获得性能提升？**: 一位成员询问 **Inductor** 现在是否可以使用 **CutEdsl** 代替 **Triton**。
   - 另一位成员澄清说，虽然在 **mm** 和 **flexattention** 模板等特定情况下是可能的，但主要重点仍然是 **Tensor Core kernels**，并补充说*它将被进一步扩展，但主要用例仍然集中在 Tensor Core kernels 上*。


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1440784085101445160)** (1 messages): 

> `DMA Collectives、AMD Instinct MI300X GPU、RCCL 通信集合库、ML 通信卸载、DMA 命令调度` 


- **DMA Collectives 提升 ML 通信**: 一篇新[论文](https://arxiv.org/abs/2511.06605)探讨了将机器学习 (ML) 通信集合操作 (collectives) 卸载到直接内存访问 (**DMA**) 引擎。
   - 对 **AMD Instinct MI300X GPU** 的分析显示，在大尺寸 (**10s of MB 到 GB**) 情况下，**DMA collectives** 与 **RCCL** 库相比表现更好或持平，性能提升 **16%**，功耗降低 **32%**。
- **DMA Collectives 与受延迟限制的小尺寸数据**: 分析显示，对于受延迟限制的小尺寸数据，DMA collectives 显著滞后，与最先进的 **RCCL 通信集合库**相比，all-gather 和 all-to-all 的速度分别慢了 **4.5 倍**和 **2.5 倍**。
   - 论文提供了 DMA 传输的详细延迟分解，并指出 **DMA 命令调度**和同步开销可能会限制 DMA collective 的性能。


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1440768685622169712)** (4 messages): 

> `Inline Triton, PTO` 


- **Inline Triton PR 已提交**: 一位成员为 inline Triton 提交了 [pull request](https://github.com/pytorch/helion/pull/1150)。
   - 另一位成员确认道：*“是的，这个看起来不错！”*
- **即将休假 (PTO)**: 一位成员提到他们明天开始休假。
   - 他们还表示：*“如果你想让我修改什么，请尽快”*。


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1440774602660778127)** (36 messages🔥): 

> `fp4 到 half2 转换、SFB Tensor 形状、MMA 指令支持、CuTe DSL 推荐、基准测试脚本偏差` 


- **FP4 转换为 half2 遇到麻烦**: 一位成员在 PTX 中使用 `cvt.rn.f16x2.e2m1x2` 时遇到错误，具体为 *Arguments mismatch for instruction 'cvt'*。
   - 另一位成员建议使用 `__nv_cvt_fp4x2_to_halfraw2()` 作为替代方案。
- **SFB Tensor 形状异常**: 一位用户报告称，在使用特定参数和仓库版本 **@db8cfd3** 调用 generate_inputs 时，SFB Tensor 的形状意外显示为 **[128, 16, 1]**，而预期形状为 **[1, 16, 1]**。
   - 经澄清，由于 *torch 的技术问题*，该 Tensor 被填充到了 128，其余行可以忽略。
- **MMA 指令在 sm_100 上不受支持**: 一位成员遇到了指令 **tcgen05.mma** 在目标平台 **sm_100** 上不受支持的问题。
   - 另一位成员简单地回复道：*target sm_100*。
- **CuTeDSL 被赞易于学习**: 一位用户感谢另一位成员撰写的优秀博客，并因其易于学习的 **CuTeDSL** 指导而推荐它。
   - 为 non_atomic_add 推荐的超参数包括：`threads_per_m = 16`，`threads_per_k = 16`，以及 `mma_tiler_mnk = (threads_per_m, 1, 128)`。
- **基准测试脚本显示巨大偏差**: 一位成员观察到当前基准测试脚本存在巨大偏差，本地的加速效果在提交结果中没有体现。
   - 另一位成员指出，似乎有一个 GPU/节点性能不佳，并建议如果看到低于预期的结果，请重新提交。


  

---

### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1440755265958973563)** (8 messages🔥): 

> `VLA 概览、动作表示方法、通过手套进行数据采集、ManiSkill 中的桌面任务` 


- **通过论文了解 VLA**: 对于 VLA 初学者，建议阅读 [pi 0.5 论文](https://arxiv.org/abs/2504.16054)和相关的[综述](https://arxiv.org/abs/2508.13073)以获取概览。
- **VLA 的动作表示**: 训练第一阶段的 SOTA 方法使用**基于分词器（tokenizer-based）的方法**，例如 PI 的 **FAST tokenizer**，通常在其上训练 **flow-matching/diffusion policies**。
- **Sunday Robotics 数据采集**: **Sunday robotics** 仅使用其[手套](https://x.com/tonyzzhao/status/1991204841289576694)收集数据，其中可能至少包含**两个摄像头、IMU** 以及用于追踪抓取动作的**传感器**。
   - 他们强调了**语言调节（language conditioning）**对于创建可提示模型的必要性。
- **ManiSkill 桌面任务数据集**: 接下来的步骤包括使用经典路径规划的解决方案，在 **ManiSkill** 中生成包含**桌面任务**的数据集，并添加相机姿态和背景的简单变化。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1440767561334526002)** (128 messages🔥🔥): 

> `Meta SAM 3 发布、GPT-5.1-Codex-Max 推出、ChatGPT Atlas 更新、LMSYS Org Miles RL 框架、Zo Computer AI 协同驾驶个人服务器` 


- **Meta 的 SAM 3 分割一切！**: **Meta** 发布了 [Segment Anything Model 3 (SAM 3)](https://ai.meta.com/blog/segment-anything-model-3/)，这是一个支持文本/视觉提示的统一图像/视频分割模型，其性能比现有模型**提升了 2 倍**，提供 **30ms 推理速度**，并包含一个用于无代码测试的 Playground。
   - 该模型的权重（checkpoints）和数据集可在 [GitHub](https://github.com/facebookresearch/segment-anything) 和 [HuggingFace](https://huggingface.co/facebookresearch/segment-anything) 上获取，为 Instagram Edits 和 FB Marketplace 的 View in Room 功能提供支持。
- **GPT-5.1-Codex-Max 登场**: **OpenAI** 推出了 **GPT‑5.1-Codex-Max**，专为长时间运行的详细工作而构建，是第一个通过名为 *compaction* 的过程原生训练以跨多个上下文窗口运行的模型，如[这条推文](https://x.com/OpenAIDevs/status/1991217500269289732)所述。
- **ChatGPT Atlas 焕然一新！**: **Adam Fry** 宣布了 Atlas 的重大版本更新，增加了**垂直标签页**、**iCloud passkey 支持**、**Google 搜索选项**、**多标签选择**、用于 MRU 循环切换的 **control+tab**、**扩展导入**、**全新下载 UI** 以及更快的 **Ask ChatGPT 侧边栏**，详情见[这条推文](https://xcancel.com/adamhfry/status/1991209533046493486?s=46)。
- **LMSYS 为 MoE 训练推出 Miles**: **LMSYS** 推出了 [‘Miles’](https://xcancel.com/lmsysorg/status/1991189801308156139?s=46)，这是轻量级 ‘slime’ RL 框架的生产级分支，针对 GB300 等新硬件以及大型 Mixture-of-Experts 强化学习工作负载进行了优化。
- **GPT-5.1 Pro：更慢、更聪明，但依然古怪？**: **Matt Shumer** 评测了 **GPT-5.1 Pro**，称其为他使用过的最强大的模型，但也更慢且缺乏 UI，并深入比较了 **Gemini 3 Pro**、创意写作/Google UX 延迟以及对编程/IDE 的期望，记录在[这条推文](https://xcancel.com/mattshumer_/status/1991263717820948651?t=cxIm6WdS70yU-vGvEWKkgw&s=19)中。


  

---

### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1440914177903296624)** (19 messages🔥): 

> `Nano Banana 2, Suno v4, Luma AI Halo, Google Deepmind` 


- ****Nano Banana Pro** 来了！**: Oliver Wang 发布了一张鹈鹕骑自行车的创意图像，引发了俏皮的评论，有人称其为 **Nano Banana 2** ([x.com 链接](https://x.com/oliver_wang2/status/1991212712014278698))。
   - 一些人要求提供 **SVG** 版本并开玩笑地宣布它实现了 **AGI**，这被戏称为 *X-Ware.v0*。
- ****Suno v4** 发布！**: Eric Zhang 神秘的 "Yay" 和一张 **Suno** 的 logo 图片可能预示着 **Suno v4** 或一个重大里程碑的发布 ([x.com 链接](https://x.com/16bitnarwhal/status/1991197540285305015))。
   - 用户分享了 **Suno** 如何彻底改变了游戏的配乐创作，并提出了关于 **Suno** 的起源、扩展计划、最大客户甚至 **ESOPs** 的问题。
- ****Luma AI** 将构建 **Halo** 超级集群**: **Luma AI** 宣布获得 **9 亿美元 C 轮融资**，将与 Humain 共同构建 **Project Halo**，这是一个 **2 GW** 的计算超级集群 ([x.com 链接](https://x.com/lumalabsai/status/1991197052760395820))。
   - 该项目旨在扩展多模态 **AGI** 的研究和部署，引发了关于成本、利用率和意识影响的热议与疑问。
- ****Google Deepmind** 发布 Nano Banana Pro**: **Google Deepmind** 发布了 **Nano Banana Pro** ([博客文章](https://blog.google/technology/ai/nano-banana-pro/))，引发了围绕其 [能力](https://x.com/googledeepmind/status/1991522595129139486) 的讨论。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1440762123641815211)** (2 messages): 

> `RE-Bench, IntologyAI` 


- **IntologyAI 的 RE-Bench 结果**: 一位用户分享了 [IntologyAI 的推文](https://x.com/IntologyAI/status/1991186650240806940) 链接，该推文声称他们在 **RE-Bench** 上的表现 **优于人类专家**。
- **请求邀请**: 一位用户询问他们是否收到了与上述主题相关的 **邀请**。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1440785597466935398)** (36 messages🔥): 

> `Inference-time epistemics layer, MMLU accuracy, Global-MMLU-Lite, Qwen 2.5 7b instruct, arxiv endorsement` 


- **推理时认识层（Inference-Time Epistemics Layer）大幅减少幻觉**: 一位成员尝试使用基于信息价值（Value-of-Information）检查的 **推理时认识层**，在小型 7B 模型上减少了约 20% 的幻觉；当预期价值不够高时，模型会转而请求澄清，[如图所示](https://cdn.discordapp.com/attachments/747850033994662000/1440785597189853194/Messenger_creation_A24CCC37-2399-4C88-B6B6-471E5B9BB96F.jpg?ex=6920bd26&is=691f6ba6&hm=25800143a94f6eb44f81ee5609061ea733afd5c3afd1e7834c199a6ff5a54f92&)。
- **关于 MMLU 准确率和幻觉测量的辩论爆发**: 成员们就使用 **MMLU** 衡量幻觉抑制的有效性展开了辩论，一位成员认为 7B 模型的表现应该优于随机概率，这与最初的说法相反，并分享了 **71%** 准确率的结果。
   - 对话强调了 **MMLU** 与多语言评估集 **Global-MMLU-Lite** 之间的区别，成员们提醒不要误读基准测试结果，并建议进行独立验证。
- **每 D' 级数的 Norm 步进是几何级数的**: 第 6 页。**每 d' 级数的 Tanh norm pre n 步进** 是一个具有基于生存进化的几何级数：*这就是优化所在*。
   - 一位成员表示，*层将状态空间探索压缩到仅限有效状态*。
- **社区成员寻求 ArXiv 背书**: 一位成员在给许多研究团队发邮件后请求 **arxiv 背书** 方面的帮助，其他人建议不要进行盲目背书，并建议在协作频道发布内容以获取反馈和合作。
   - 另一位成员链接到了 [背书页面](https://arxiv.org/auth/endorse?x=63SW7W)。


  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1440803599889338530)** (80 messages🔥🔥): 

> `Approximate KNN, Linear Attention Limitations, FFT History, AGI Requirements, Attention Scores Softmax` 


- ****KNN 无法超越平方级 Attention？****：一位用户声称，除非 SETH 为假，否则不可能在任意数据上以低于 **O(n^2)** 的复杂度实现近似 **KNN**，这限制了线性 Attention 的能力。
   - 另一位用户提醒不要如此断言，并类比了历史上的观点：在 **Cooley-Tukey 算法**被发现之前，人们普遍认为离散傅里叶变换（Discrete Fourier Transform）必然是平方级的。
- ****社会学视角的 FFT：宣称不可能****：一位用户提供了一个[链接](https://www.ece.ucdavis.edu/~bbaas/281/papers/CooleyLewisWelch.1967.HistNotesFFT.pdf)，强调了宣称“不可能”的社会学层面，指出在 **Cooley-Tukey** 出现之前，许多资深人士认为 **Fourier analysis** 需要 **N^2 次操作**。
   - 该用户澄清，他是在表达关于计算复杂度的社会学观点，而非计算复杂度的硬性限制点。
- ****专家称 AGI 不会是线性架构****：一位用户认为，除非 **3SAT** 被解决，否则 AGI 不可能是线性成本架构，并引用了一篇[论文](https://arxiv.org/abs/1803.00904)来支持其观点。
   - 另一位用户反驳称，执行更难任务的更强模型会超越简单的词频统计匹配，倾向于**约束求解（constraint solving）**和 **SAT**。
- ****Attention：Softmax 与趋近于零的分数****：一位用户指出，在长序列中，经过 Softmax 处理后，绝大多数 Attention 分数都极其接近 0，这暗示了潜在的优化空间。
   - 另一位用户反驳说，证明表明在一般情况下，如果不检查每个项目，就无法进行近似最近邻搜索，并链接了[两篇论文](https://arxiv.org/abs/2505.14840)和[另一篇](https://arxiv.org/abs/2209.04881)。
- ****内在维度辩论****：一位用户引用 **Hopfield 网络视角**假设，向量的内在维度必须随上下文长度增加而增加，以保持可区分性。
   - 另一位用户反驳称，Attention 并不要求向量正交，而是要求它们的相似性结构足够不同，并补充说维度为 D 的随机向量的内积大约为 **N(0, 1/D)**。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1440809077231058944)** (3 messages): 

> `Sparse MoEs vs Dense Models, SAE-based methods, Interpretability in Sparse Models, Bridge System for Swapping Blocks` 


- **稀疏 MoE 引发可解释性辩论**：成员们讨论了稀疏 **Mixture of Experts (MoE)** 模型与稠密模型相比的可解释性，引用了一篇[论文](https://arxiv.org/abs/2301.04103)指出稀疏性有助于可解释性。
   - 疑问依然存在：研究稀疏的“玩具”模型是否值得，还是直接解构常规模型更好？并质疑在稀疏模型中发现的类似电路是否能在实际模型中找到。
- **SAE 方法被认为模糊**：一位成员指出，**基于 SAE 的方法**“无论如何都相当模糊”，将其与正在考虑的其他方法区分开来。
   - 论点是，如果稀疏模型的行为与稠密模型完全一致但更具可解释性，它就可以用于安全关键型应用。
- **桥接系统允许模块替换**：讨论强调了一个允许将稠密块替换为稀疏块的桥接系统。
   - 这使得**基于可解释性的干预**成为可能，潜在地作为特定任务（如安全干预）的有用工具。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/)** (1 messages): 

noble_monkey_75488: <@328142664476131330> 我们是否有对 text-to-SQL 任务的支持？
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1440773534090793031)** (97 条消息🔥🔥): 

> `Gemma 3, RE-Bench, World Models, Atropos environment, Nano Banana Pro` 


- **Gemma 3.0 热度持续走高**：爱好者们称 [Deepmind 的 Gemma 3.0](https://m.youtube.com/watch?v=6WMufQlgndc) *非常疯狂*，尽管一些人降低了预期，指出 YouTube 视频*显然只是炒作*。
   - 有人澄清说 **Gemini** 和 **Gemma** 是不同的，虽然令人印象深刻，但它*肯定不是 AGI*，只是在*将 Alphabet 的股价推向 300 美元*。
- **Intology 声称在 RE-Bench 上占据主导地位**：[IntologyAI](https://x.com/IntologyAI/status/1991186650240806940) 声称其模型在 **RE-Bench** 上的表现*优于人类专家*。
   - 一位用户调侃说，他们的模型甚至不会出现拒绝回答的情况，并对其他人的经历表示困惑。
- **World Models 的进化之路**：尽管 LLM 备受关注，但一些人认为 *World Models 将长期存在*，并且是*下一个进化方向*，**Deepseek**、**Qwen**、**Kimi**、**Tencent** 和 **Bytedance** 都有相关的发布计划。
   - 引用了由 **李飞飞博士 (Dr. Fei-Fei Li)** 参与的 [Marble Labs 视频](https://m.youtube.com/watch?v=1ykQnA8VUu0) 作为 World Models 的关键案例。
- **Atropos Python 脚本难题**：一名成员就 LLM 为 **Atropos** 生成的 [Python 脚本](https://cdn.discordapp.com/attachments/1149866623109439599/1441033887680565298/atropos_noc_env.py?ex=692052e3&is=691f0163&hm=2ae3c0608cbab00ae0fd65242837545e3d07f865600af1e6262daf20d96e357c&) 寻求帮助，但在评估过程中遇到了错误。
   - 他们被引导至文档和社区频道以获取支持。
- **Nano Banana Pro 图像生成能力令人印象深刻**：用户赞扬了新款 **Nano Banana Pro** 的图像生成能力，特别是它生成信息图表（infographics）的能力。
   - 一位用户链接到了 [scaling01 的推文](https://x.com/scaling01/status/1991523932336464333?s=46)，展示了一张具有出色文本和布局的信息图表。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1441184809606185144)** (3 条消息): 

> `ArXiv Endorsement, EleutherAI Discord` 


- **获取 ArXiv 背书的过程非常困难**：一名成员在向大约 **20 个研究团队**发送邮件未果后，正在寻求 [ArXiv 背书 (endorsement)](https://arxiv.org/auth/endorse?x=63SW7W) 的帮助。
- **分享了 EleutherAI Discord 链接**：一名成员分享了 [EleutherAI Discord 邀请链接](https://discord.gg/eleutherai)。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1440775018073165938)** (4 条消息): 

> `Gemini models, Negativity bias` 


- **Gemini 模型表现出诡异行为**：一位用户分享了一个[链接](https://x.com/halfboiledhero/status/1991145723291644162?s=46)，指出其他 **Gemini 模型**显然也出现了奇怪的行为。
   - 该用户推测问题可能源于 **Gemini 训练配方**中的某些东西。
- **RP 社区发现 Gemini 的负面偏见**：成员们报告称，**RP (red-pilling) 社区**发现了 Gemini 存在**负面偏见 (negativity bias)**，这可能与上述异常行为有关。
   - 未提供关于这种**负面偏见**性质的进一步信息。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1441184809606185144)** (3 条消息): 

> `ArXiv Endorsement Assistance, EleutherAI Discord Link` 


- **ArXiv 背书请求未获回应**：一名成员请求协助获取 ArXiv 背书，提到他们已经给大约 **20 个研究团队**发了邮件但没有成功，并分享了 [ArXiv 背书链接](https://arxiv.org/auth/endorse?x=63SW7W)。
   - 他们还要求重新发送链接，因为他们第一次错过了。
- **分享了 EleutherAI Discord 链接**：一名成员分享了 **EleutherAI Discord 服务器**的链接。
   - 分享的链接为 [https://discord.gg/eleutherai](https://discord.gg/eleutherai)。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1440749585755799644)** (76 条消息🔥🔥): 

> `KTOTrainer 与多 GPU、n8n 的 VPS、API 参数失效、AI 记忆与召回、Inference Endpoints 500 错误` 


- **KTOTrainer 可能支持多 GPU**：一名成员询问 **KTOTrainer** 是否适用于多 GPU，另一名成员链接了一个 [Hugging Face 数据集](https://huggingface.co/datasets/John6666/forum2/blob/main/trl_kto_blow_up_memory_2.md)，暗示它是支持的。
   - 如果存在 Bug，该成员还建议尝试 [这个 Discord 频道](https://discord.com/channels/879548962464493619/1403622951940657235)。
- **用户解决 AI 记忆与召回问题**：一名成员声称已解决了 **具备记忆、记忆召回及 Token 膨胀问题的 AI**，并计划很快推出企业级解决方案。
   - 另一位用户询问这是否类似于 **LongRoPE 2** 或 **Mem0**。
- **Inference Endpoints 出现 500 错误**：一名成员报告称，所有 **Inference Endpoints** 出现 **500 错误** 已持续两小时，且没有任何日志，支持团队也未回应；他们通过禁用身份验证来绕过此问题。
   - 一名 Hugging Face 工作人员确认了该报告，并表示内部正在调查此问题。
- **在 Fal 上尝试 Maya1 语音模型**：一名成员宣布 **Maya1 语音模型** 现已可在 Fal 上试用，并链接了一篇 [推文](https://x.com/Dheemanthredy/status/1991566362813296965)。
   - 另一名成员询问在哪里下载 `kohya_ss-windows.zip`，有人提供了一个 [GitHub 链接](https://github.com/bmaltais/kohya_ss?tab=readme-ov-file#installation-options)。
- **记忆 + Agent 实验**：一名用户分享了 **MemMachine Playground**，它可以访问 **GPT-5**、**Claude 4.5** 和 **Gemini 3 Pro**，所有模型均由持久化 AI 记忆支持，并链接到了 [HuggingFace Space](https://huggingface.co/spaces/Memverge/MemMachine-Playground)。
   - 它是完全开源的多模型游乐场，专为实验记忆 + Agent 而构建。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1441069750724726876)** (1 条消息): 

> `HuggingFace 学习课程、HuggingFace 测验错误` 


- **完成 HF 学习课程**：一名成员完成了 [HuggingFace Learn](https://huggingface.co/learn) 章节中的 **LLM 课程**。
- **HF 测验遇到错误**：该成员在完成课程后进行测验时遇到了 **错误**。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1441209140180553971)** (1 条消息): 

> `MemMachine Playground、GPT-5 访问、Claude 4.5 访问、Gemini 3 Pro 访问、持久化 AI 记忆` 


- **MemMachine Playground 开放使用**：**MemMachine Playground** 已在 Hugging Face 上线，提供对 **GPT-5**、**Claude 4.5** 和 **Gemini 3 Pro** 的访问，全部由持久化 AI 记忆驱动；可通过 [HuggingFace Spaces](https://huggingface.co/spaces/Memverge/MemMachine-Playground) 访问。
- **MemMachine 完全开源**：MemMachine 是 **完全开源** 的，旨在进行记忆加 Agent 的实验。
   - 这是一个为实验而构建的 **多模型游乐场**。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1440934152466727064)** (7 条消息): 

> `Mimir VSCode 插件、视频制作技巧、开源案例研究、MemMachine Playground` 


- **Mimir 通过 VSCode 插件管理 Agent**：[Mimir](https://github.com/orneryd/Mimir) 发布了一个 **VSCode 插件**，用于在 IDE 中管理 Agent，它可以从 Docker 编辑文件、运行代码智能，并拥有受你控制的持久化记忆图存储。
   - 该插件采用 **MIT 许可证**，包含一个多 Agent 拖拽式工作室，如[附带的截图](https://cdn.discordapp.com/attachments/897390720388825149/1440934478640971837/Screenshot_2025-11-19_at_9.53.32_PM.png?ex=69209f0e&is=691f4d8e&hm=dda5333892c17ebd841f8e134749a1f263ece9f7f9fe53410d2e13ad98844fa2&)所示。
- **寻求拍摄视频的建议**：一名成员询问了 **视频制作技巧**，并分享了一个 [YouTube 链接](https://www.youtube.com/watch?v=r1Pda6_KVbY) 作为参考。
- **案例研究详细介绍开源历程**：一名成员发表了一篇 [案例研究](https://medium.com/p/131a5a28fc68)，详细介绍了他们的开源之旅，并欢迎反馈。
- **MemMachine Playground 记忆长廊**：**MemMachine Playground** 现已上线，提供对 **GPT-5、Claude 4.5 和 Gemini 3 Pro** 的访问，所有模型均由持久化 AI 记忆支持。
   - 该游乐场是 [完全开源](https://huggingface.co/spaces/Memverge/MemMachine-Playground) 的多模型平台，专为实验记忆 + Agent 而构建。


  

---

### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1441034323493916672)** (6 messages): 

> `PR 合并延迟, 课程报名问题, VRAM 问题` 


- ****PR** 恐慌：合并任务：不可能的任务？**：一名成员报告了第一模块的 **PR** 合并延迟，导致随后无法为第二模块添加内容，正在寻求解决该问题的指导，并提供了 [截图链接](https://cdn.discordapp.com/attachments/1313889336907010110/1441034323200442399/image.png?ex=6920534b&is=691f01cb&hm=db9516fabafa75acc99ba31a89f0aef6b1ba3e4df3dd9555605ed522913dddfa&)。
   - 另一名成员做出了回应，要求提供 **PR 链接**以加快审核流程，并承诺在周末处理。
- ****报名混乱**：课程注册难题？**：一名成员报告在尝试报名课程时遇到了循环链接引用，在 [https://huggingface.co/smol-course](https://huggingface.co/smol-course) 和 [https://huggingface.co/learn/smol-course/unit0/1](https://huggingface.co/learn/smol-course/unit0/1) 之间循环。
   - 该用户陷入了循环，无法实际报名参加课程。
- ****VRAM** 漩涡：资源需求令人侧目？**：一名成员在 `HuggingFaceTB/smoltalk2_everyday_convs_think` 上进行训练时，运行 [https://huggingface.co/learn/smol-course/en/unit1/3](https://huggingface.co/learn/smol-course/en/unit1/3) 的代码示例遇到了问题。
   - 尽管拥有 **80GB 的 VRAM**，该成员发现仍无法继续，这表明资源需求或配置可能存在问题。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1440799707676020840)** (39 messages🔥): 

> `AI CEO 基准测试, Huggingface Xet 仓库, 顶级 AI 人才, 规避 AI 检测` 


- ****Skyfall AI** 构建 AI CEO 基准测试！**：[Skyfall AI](https://skyfall.ai/blog/building-the-foundations-of-an-ai-ceo) 提议通过商业模拟游戏为 Agent 的长程规划能力建立新的环境/基准测试，并指出 **LLM 的表现低于**人类基准。
   - 该公司设想 **AI CEO** 需要一种脱离 LLM、更接近于**世界建模 (World Modeling)** 的架构，在这种架构中，行动的后果可以在企业世界中进行模拟。
- ****Huggingface** Xet 仓库设置困扰用户**：一位用户对 **Huggingface** 上的 **Xet 仓库**设置表示沮丧，理由是由于需要 **Brew** 以及下载模型的缓存方式不直观，导致下载模型进行微调非常困难。
   - 该用户认为这个过程很繁琐，并表示：“感觉他们是为那些坦白说不该出现在这个平台上的人把事情做简单了。”
- **辩论：当今谁是**顶级 AI** 人才？**：由于对《奔腾年代》(Halt and Catch Fire) 后期季度缺乏技术讨论感到失望，一位成员想知道当今谁是**顶级 AI** 人才。
   - 他们澄清说，不想去申请那些满是只会刷 **LeetCode** 的人的地方，而是更愿意协助研究。
- **用户寻求**规避 AI 检测**的帮助；引发抵制**：一位用户询问是否有办法规避报告中的 AI 检测，并链接了一篇可能相关的 [ArXiv 论文](https://arxiv.org/abs/2510.20810)。
   - 在承认他想“提升” 20% 的工作量后，其他用户拒绝提供帮助，并告诉他应该正确引用参考文献，并接受他的作品原创性不足的事实。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1440763456293834803)** (5 messages): 

> `分割“爱”, 诅咒分形` 


- **请求分割“爱”的概念**：一位成员询问是否可以使用 AI 来分割“爱”的概念。
   - 另一位成员用歌词 *What is love?* 进行了回应。
- **一个诅咒分形出现了**：一位成员开玩笑地展示了一个拼出 *Baby don't hurt me* 的**诅咒分形**。
   - 他们的语气听起来像个真正的 **Vogon**。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1440760387883499662)** (13 messages🔥): 

> `Sam3D, DeepSeek, Nvidia Q3, AI benchmark, OLMo 3` 


- **Sam3D 表现不及 DeepSeek**：一位成员指出，[Sam3D](https://www.deepcogito.com/research/cogito-v2-1)（一个经过后训练的 **DeepSeek** 模型）的表现比 **DeepSeek** 原生模型更差。
- **LLMs 面临 Weird ML Benchmark**：一位成员分享了一个基准测试，测试 **LLMs** 通过编写可运行的 **PyTorch code** 并从反馈中迭代学习来解决怪异且不寻常的机器学习任务的能力 ([X post](https://x.com/htihle/status/1991133595402949046?t=-SiXwTO6x_xF5KIS1vSEgA&s=19))。
- **Nvidia 的“铲子”产出黄金**：**Nvidia** 第三季度营收和利润超出预期，凸显了向淘金者卖铲子的盈利能力 ([Reuters](https://www.reuters.com/markets/us/nvidia-q3-updates-ai-bubble-fears-spotlight-2025-11-19/))。
- **OLMo 3：真正的开源推理模型出现**：一位成员分享了 **OLMo 3** 的链接，它被描述为美国真正的开源推理模型 ([Interconnects.ai](https://www.interconnects.ai/p/olmo-3-americas-truly-open-reasoning))。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1440766787359866962)** (38 messages🔥): 

> `Kimi K2 Thinking vs GPT-5, Kimi $19 coding plan cost, Minimax AMA on Reddit, SGLang tool calling issues with Kimi K2, Perplexity Pro and Kimi K2` 


- **K2 Thinking 是开源版的 GPT-5？**：一位成员认为 **K2-thinking** 是最接近 **GPT-5** 的开源等效模型，作为一个在各个领域都有强劲表现的全能选手表现出色。
   - 此外，有人建议 **Kimi** 可能是创意写作方面的最佳选择。
- **Kimi 编程方案价格点引发争议**：一些成员认为 **Kimi** 的 19 美元编程方案太贵了，特别是对于学生、独立开发者或从事副业项目的人来说，他们认为 7-10 美元的档位更合理。
   - 一位成员表示：“目前很难证明其合理性，因为 Claude 提供了更好的性价比”。
- **Reddit 上的 Minimax AMA 引起关注**：一位成员分享了 Reddit 上关于 **Minimax** 的 AMA（问我任何事）截图。
   - 这次 AMA 似乎在频道内引起了极大的好奇心，一位成员将其描述为“疯狂”。
- **SGLang 与 Kimi K2 的工具调用挑战**：成员们报告了在 **SGLang** 上使用 **Kimi K2 Thinking** 实现服务端工具调用（tool calling）时的问题，指出即使推理内容显示需要调用工具，工具也不会被调用。
   - 他们引用了一个[相关的 GitHub issue](https://github.com/MoonshotAI/Kimi-K2/issues/89)，并怀疑问题是否源于使用了 `/v1/chat/completions` 而不是 `/v1/responses`。
- **Perplexity AI 中 Kimi K2 的集成受到质疑**：一位 Perplexity Pro 用户报告称 **Kimi K2** 无法正常工作，即使尝试了无痕模式也是如此；另一位用户询问编程方案是否允许在 API 上访问 **Kimi K2 Thinking Turbo**。
   - 另一位成员表示：“那里的 Kimi K2 简直没法用，验证答案的 Agent 根本不工作。优化得很糟糕”。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1440874113273954376)** (13 messages🔥): 

> `Mojo nightly build performance degradation, Mojo profiling tools, AMD iGPU/APU support for Mojo` 


- **Mojo Nightly 构建版本性能大幅下降 📉**：一位成员报告了 nightly Mojo 构建版本 **(ver - 0.25.7.0)** 的显著性能退化，在 Mac M1 上运行 [llama2.mojo](https://github.com/tairov/llama2.mojo) 时，吞吐量从 Mojo **24.3** 版本的 **~1000 tok/sec** 下降到 **~170 tokens/sec**。
   - 他们已要求 Mojo 编译器团队调查性能下降的原因并提供见解或修复方案，暗示重构后的代码可能存在效率低下的问题。
- **使用 Perf 对 Mojo 进行性能分析 🔎**：针对关于 Mojo 性能分析工具的问题，一位成员建议使用 **perf**，并指出这在过去对他们有效。
   - 他们回忆起在[工具链讨论帖](https://discord.com/channels/1087530497313357884/1151418092052815884/1366886163972886569)中讨论过相关细节。
- **AMD iGPU/APU 对 Mojo Puzzles 的兼容性 🧩**：一位成员询问了 AMD iGPU 与 Mojo puzzles 的兼容性，引发了关于 GPU 兼容性层级的详细回复。
   - 对于 **RDNA 3** 及以上架构的集成显卡，只要安装了正确的 ROCm 驱动，大多数 puzzles 应该可以运行，正如[文档](https://docs.modular.com/max/packages/#gpu-compatibility)中所述，尽管较旧的 **APU (3500u, 4500u)** 可能会因为 ROCm API 的可用性而导致支持受限。


  

---

### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1441199888581787668)** (1 messages): 

> `Modular Platform 25.7 Release, Open MAX Python API, Next-Gen Modeling API, NVIDIA Grace Support, Mojo GPU Programming` 


- **Modular Platform 25.7 助力 AI 开发**：最新的 [Modular Platform 25.7](https://www.modular.com/blog/modular-25-7-faster-inference-safer-gpu-programming-and-a-more-unified-developer-experience) 版本通过关键更新，专注于减少基础设施开销并推动 AI 进步。
- **开源 MAX Python API 实现无缝集成**：该版本引入了完全**开源的 MAX Python API**，在 AI 开发工作流中提供了更无缝的集成和灵活性。
- **下一代建模 API 亮相**：包含了一个**下一代建模 API**，有望增强创建和部署 AI 模型的能力。
- **NVIDIA Grace 获得扩展支持**：提供了对 **NVIDIA Grace** 的扩展支持，使开发者能够利用强大的硬件进行 AI 应用。
- **Mojo GPU 编程变得更安全、更快速**：**Mojo GPU 编程**在安全性和速度方面得到了增强，促进了更高效、更可靠的 GPU 利用。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1440760264306720840)** (22 messages🔥): 

> `Mojo superset of Python, Python's type suggestions, Shared mutable references, AI native` 


- **Mojo 作为 Python++，拥抱 GC 和类型系统**：讨论集中在将 **Mojo** 作为 **Python** 的超集，重点关注集成**垃圾回收 (GC)**和**静态类型**以提高性能的优势，并希望拥有类似于 Python 但具备 Mojo 类型安全性的 GC 机制。
   - 一位成员指出，虽然 *pyobject* 某种程度上可以使用，但会导致类型信息丢失，表达了希望在 Mojo 中拥有与 Python 相同的 **GC 机制**但具备完整类型支持的愿望。
- **Python 的类型系统：仅作为建议？**：成员们强调，在 **Python** 中，类型被视为建议，这在编写 Kernel 时跨不同设备管理数据带来了挑战，特别是在 CPU-GPU 传输和 GPU 点对点通信方面。
   - 有人建议，如果所有 Kernel 完全使用 **Mojo** 编写，理论上可能可以跟踪数据驻留（data residency），而其他人则指出了驻留在一个设备上的数据被另一个设备使用时的潜在问题，特别是同时进行读写操作时。
- **共享可变引用与逃生舱 (Escape Hatches)**：讨论了 Mojo 中对**共享可变引用**的需求，一些人承认由于潜在的竞态条件（race conditions），此类引用将需要使用**逃生舱 (escape hatches)**。
   - 一位成员表示，在没有 *unsafecell*（目前尚不可用）的情况下，类型系统中的共享变更是必要的，尽管由于潜在的竞态，它永远不会是完全安全的。
- **Mojo 被誉为未来的 AI 原生语言**：表达了对 **Mojo** 作为一种极具前景的 **AI 开发**语言的热情，尤其是作为 Python 的替代方案。
   - 一位成员提到他们正在用 **Python** 构建 **AI** 项目，但他们*迫不及待地想让真正的 AI 原生（AI native）成为现实*，并链接到了 [Modular 25.7 发布公告](https://www.modular.com/blog/modular-25-7-faster-inference-safer-gpu-programming-and-a-more-unified-developer-experience)。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1441159104964857856)** (1 messages): 

> `MAX Opening` 


- **MAX 开源！**：一位成员对 **MAX** 的开源表示兴奋。
   - 他们感谢了所有为此付出努力的人。
- **关于 MAX 的更多细节**：关于 **MAX** 和团队工作的额外信息。
   - 细节尚未公布。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1441117319521632357)** (1 messages): 

> `Gem3pro, proxy-server, dspy-proxy` 


- **Gem3pro 构建代理服务器**：一位用户在看到[这条推文](https://x.com/skylar_b_payne/status/1990808733140779488)后，提示 **Gem3pro** 构建一个代理服务器，并对其 one-shot（一次性生成）成功感到惊讶。
- **新 DSPy Proxy 仓库发布**：一位成员分享了 GitHub 上一个新的 **DSPy proxy** 仓库链接：[aryaminus/dspy-proxy](https://github.com/aryaminus/dspy-proxy)。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1440754446266011721)** (18 messages🔥): 

> `LiteLLM for Azure, Provider errors in ReAct, moonshot provider, TPM issues` 


- **LiteLLM 扩展了 Azure 支持！**：有成员提到 **LiteLLM**（DSPy 使用的 LLM 库）需要增加对 Azure 的支持，以便在 **Azure** 上实现与 **OpenAI** 类似的使用方式；[文档点击此处](https://docs.litellm.ai/docs/providers/azure/)。
- **ReAct 的 Provider 陷阱**：有人指出某些 provider 在 **ReAct** 中运行几次迭代后会报错，导致使用范围受限于 **Groq** 或 **Fireworks** 等 provider。
   - 随后该成员询问 DSPy 是否能解决这个问题，或者是否只需要对可用的 provider 进行分类。
- **Moonshot Provider 表现良好**：一位成员分享说 **moonshot** 这个 provider 运行正常，但 **TPM** 表现非常糟糕，并分享了报错截图 [点击此处](https://cdn.discordapp.com/attachments/1161519469319946286/1441026796975030314/image.png?ex=6920f509&is=691fa389&hm=bd9ac54ed089e8b5a88ac4344196fae702f0408af04d230165e5f0d5f9496bd7)。


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1441116538626244728)** (5 messages): 

> `DNS Migration, Domain Governance, IaaC for DNS` 


- **MCP 域名面临 DNS 迁移停机**：**modelcontextprotocol.io** 域名正在进行 [DNS 迁移](https://modelcontextprotocol.io)，从 Anthropic 移交给社区控制，以增强治理并加速项目发布。
   - 一位成员警告说，尽管已努力减少干扰，但在计划于下周进行的迁移过程中仍可能出现停机。
- **DNS 迁移时间避开 MCP 周年纪念日**：一位成员建议 DNS 迁移应避开 **25 号** 的 **MCP 周年纪念日**，以防止在庆祝活动期间出现网站停机。
   - 他们建议如果 DNS 迁移即将进行，应该选在 25 号 *之前* 或 *之后*。


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1440866165059485758)** (4 messages): 

> `SEPs, governance model, disseminating process, sponsor for a SEP` 


- **“顺手提交”的 SEP 引发讨论**：一位成员注意到许多 SEP 是以 *随意提交（drive by fashion）* 的方式创建的，并建议在直接提交 **SEP** *之前*，改进将初步想法转化为交付成果的 **分发流程**。
   - 这样做的目的是防止人们在没有得到认可的正式文档上浪费时间，建议先通过 **低门槛的交流** 来衡量大家的兴趣。
- **SEP 需要赞助人**：另一位成员同意有必要强调为 **SEP** 寻找 **赞助人（sponsor）**，以鼓励更早的参与和认可。
   - 团队已经在 **核心维护者（Core Maintainer）会议** 中讨论了这一点，并计划很快更新 **SEP 流程**。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/)** (1 messages): 

arshadm: CuteDSL 太棒了
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1440774781531066398)** (2 messages): 

> `Bug Fix, tinygrad update, Lab Troubles` 


- **tinygrad 更新解决了 Bug**：在更新 **tinygrad** 后，一位用户报告某个 Bug 不再复现。
   - 该用户本想早点测试，但他们的 *实验室出了一些故障*。
- **实验室故障推迟了测试**：一位用户提到他们的实验室遇到了问题，推迟了 Bug 测试。
   - 据该用户称，在更新 **tinygrad** 后，该 Bug 不再复现。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1440749650352017488)** (3 messages): 

> `Manus case success, Operator extension bug, Aurora Seeker` 


- **Manus 案例：1.5 Lite 的小胜利**：一位成员分享了他们对 **Manus case 1.5 Lite** 的满意评价，该工具使用 [bliss](https://www.blisshq.com/) 成功定位并上传了缺失的专辑封面。
   - 他们强调 *并不总是需要大成就*，突出了认可和欣赏哪怕是微小成就的重要性。
- **Operator 扩展陷入循环 Bug？**：一位成员报告了 Chrome 中 **Operator 扩展** 的一个问题，即尽管已经安装，它仍反复提示重新安装。
   - 用户描述说，引导扩展使用 Amazon 上打开的标签页进行搜索时触发了持续的重新安装请求，并询问是否应该改用 **Aurora Seeker**。