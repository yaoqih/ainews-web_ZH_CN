---
companies:
- facebook-research
- meta-ai-fair
- tsinghua-university
date: '2024-07-10T05:57:13.049109Z'
description: '**Lilian Weng** 发布了一篇关于**幻觉检测**和**抗幻觉方法**的全面文献综述，涵盖了 FactualityPrompt、SelfCheckGPT
  和 WebGPT 等技术。**Facebook AI 研究院 (FAIR)** 发布了 **MobileLLM**，这是一种参数量在十亿以下的端侧语言模型架构，通过“窄而深”的模型设计和权重共享等创新，实现了与
  **llama-2-7b** 相当的性能。此外，一种具有强表达能力隐藏状态的新型**基于 RNN 的大模型架构**问世，它取代了注意力机制，在长文本建模方面的扩展性优于
  Mamba 和 Transformer 模型。最后，**清华大学**开源了 **CodeGeeX4-ALL-9B**，这是一款在代码辅助方面表现出色的多语言代码生成模型。'
id: 05cb6ee9-cadf-4e4e-8e05-5ff72d90179d
models:
- llama-2-7b
- codegeex4-all-9b
- mamba
original_slug: ainews-to-be-named-3686
people:
- lilian-weng
- yann-lecun
title: 测试时训练 (Test-Time Training)、MobileLLM、Lilian Weng 谈幻觉（外加：Turbopuffer）
topics:
- hallucination-detection
- anti-hallucination-methods
- on-device-ai
- model-architecture
- rnn
- long-context-modeling
- model-scaling
- expressive-hidden-states
- code-generation
---

<!-- buttondown-editor-mode: plaintext -->**Depth is all you need.** 我们无法决定该重点推荐什么，所以这里有 3 个头条故事。

> 2024年7月8日至7月9日的 AI 新闻。
我们为你检查了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord（**463** 个频道和 **2038** 条消息）。
预计节省阅读时间（以 200wpm 计算）：**250 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论了！

两个我们错过的重大故事，以及一个我们喜欢但不想占用全部篇幅的新故事：

1. [**Lilian Weng 关于 Extrinsic Hallucination 的文章**](https://lilianweng.github.io/posts/2024-07-07-hallucination/)：每当 Lil'Log 更新时，我们[通常](https://buttondown.email/ainews/archive/ainews-lilian-weng-on-video-diffusion/)会放下手头的一切去阅读，但她似乎悄无声息地发布了这篇极其详尽的文献综述，甚至没有在 Twitter 上宣布。Lilian 定义了 **Hallucination Detection**（FactualityPrompt, FActScore, SAFE, FacTool, SelfCheckGPT, TruthfulQA）和 **Anti-Hallucination Methods**（RARR, FAVA, Rethinking with Retrieval, Self-RAG, CoVE, RECITE, ITI, FLAME, WebGPT）的 SOTA，并以一份关于其他 **Hallucination eval** 基准测试的简短阅读清单结束。我们肯定需要针对 Reddit 摘要在这方面做大量工作。
2. [**MobileLLM：为端侧使用优化十亿参数以下的语言模型**](https://github.com/facebookresearch/MobileLLM)：这是即将在 ICML 上发表的 [最受关注](https://x.com/_akhaliq/status/1761951318711640355/quotes) 的 FAIR 论文之一（尽管甚至没有获得 Spotlight，嗯），专注于十亿以下规模的端侧模型架构研究，使一个 350M 模型达到了与 Llama 2 7B 相同的性能，[令人惊讶的是在对话语境下](https://x.com/reach_vb/status/1809866925637345750?ref_src=twsrc%5Etfw%7Ctwcamp%5Etweetembed%7Ctwterm%5E1809866925637345750%7Ctwgr%5E984c999745e3e6e2d8c7fddc68a5da7d52f1352f%7Ctwcon%5Es1_&ref_url=https%3A%2F%2Fwww.emergentmind.com%2Fpapers%2F2402.14905)。[Yann LeCun 的要点总结](https://x.com/ylecun/status/1810035281472491665?ref_src=twsrc%5Etfw%7Ctwcamp%5Etweetembed%7Ctwterm%5E1810035281472491665%7Ctwgr%5E984c999745e3e6e2d8c7fddc68a5da7d52f1352f%7Ctwcon%5Es1_&ref_url=https%3A%2F%2Fwww.emergentmind.com%2Fpapers%2F2402.14905)：1) 窄而**深**，而非宽；2) token->embedding 和 embedding->token 使用**共享矩阵**；在多个 Transformer 块之间使用**共享权重**。 
![image.png](https://assets.buttondown.email/images/d10e727f-a5cd-4296-82d6-0aab39bf2fb0.png?w=960&fit=max)
 
3. [**Learning to (Learn at Test Time): 具有表达性隐藏状态的 RNN**](https://github.com/test-time-training/ttt-lm-pytorch)（[导师](https://x.com/xiaolonw/status/1810387662060269668) 和 [作者](https://x.com/karansdalal/status/1810338845659131940) 的推文）：继 [ICML 2020](https://x.com/xiaolonw/status/1283447035673210880) 关于 Test-Time Training 的工作之后，Sun 等人发布了一种“**新型 LLM 架构**，具有线性复杂度和表达性隐藏状态，用于长上下文建模”，它**直接取代了 Attention**，“**比 Mamba 和 Transformer 具有更好的扩展性（从 125M 到 1.3B）**”且“**在长上下文下表现更好**”。
 
![image.png](https://assets.buttondown.email/images/e7a54ad3-16ba-41cb-af6d-6a963705490c.png?w=960&fit=max)
 主要见解是将 RNN 的隐藏状态替换为一个小型神经网络（而不是用于记忆的特征向量）。 
![image.png](https://assets.buttondown.email/images/83f3f5fd-1460-4e33-8881-dd75a3a9b6a3.png?w=960&fit=max)
 [基本直觉](https://x.com/xiaolonw/status/1810387664929173520) 是合理的：“如果你相信训练神经网络通常是压缩信息的好方法，那么训练一个神经网络来压缩所有这些 token 就是有意义的。”如果我们能一直嵌套网络，这个兔子洞到底有多深？

[Turbopuffer 也结束了隐身模式](https://turbopuffer.com/blog/turbopuffer)，发布了一篇广受好评的小文章。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

> 我们的 Twitter 流水线遇到了问题，请明天再来查看。

---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**AI 模型与架构**

- **CodeGeeX4-ALL-9B 开源**：在 /r/artificial 中，清华大学开源了 CodeGeeX4-ALL-9B，这是一个突破性的多语言代码生成模型，[**性能超越了主要竞争对手并提升了代码辅助能力**](https://www.marktechpost.com/2024/07/07/tsinghua-university-open-sources-codegeex4-all-9b-a-groundbreaking-multilingual-code-generation-model-outperforming-major-competitors-and-elevating-code-assistance/)。
- **Mamba-Transformer 混合模型展现潜力**：在 /r/MachineLearning 中，Mamba-Transformer 混合模型提供了[**巨大的推理加速，对于 120K 输入 token，速度提升高达 7 倍**](https://www.reddit.com/gallery/1dy5w23)，同时在能力上略微优于纯 Transformer。输入上下文越长，优势越明显。
- **Phi-3 框架在 Mac 上发布**：/r/LocalLLaMA 分享了 Phi-3 for Mac 的消息，这是一个多功能的 AI 框架，[**利用了 Phi-3-Vision 多模态模型和最近更新的 Phi-3-Mini-128K 语言模型**](https://www.reddit.com/r/LocalLLaMA/comments/1dy9ap9/phi3_for_mac_locallyrun_vision_and_language/)。它旨在利用 MLX 框架在 Apple Silicon 上高效运行。

**AI 安全与伦理**

- **前 OpenAI 研究员警告安全被忽视**：在 /r/singularity 中，前 OpenAI 研究员 William Saunders 表示，当他意识到 [**OpenAI 就像泰坦尼克号——一场激励机制驱动公司忽视安全**](https://v.redd.it/445pb0eg5bbd1)并建造更大的船只最终导致灾难的竞赛时，他选择了辞职。
- **AI 模型合规性测试显示审查差异**：/r/singularity 中的一项 AI 模型合规性测试显示了[**哪些模型的审查最少**](https://i.redd.it/54tntzz8jfbd1.png)。Claude 模型除一个外均处于后半部分，而 GPT-4 则进入了前半部分。
- **超个性化可能导致共享现实的破碎**：在 /r/singularity 中，Anastasia Bendebury 警告说，[**由于 AI 导致的媒体内容超个性化可能导致我们生活在本质上不同的宇宙中**](https://v.redd.it/e1dcvd2mrfbd1)。这可能会加速社交媒体算法中已经出现的过滤泡（filter bubble）效应。

**AI 应用**

- **Pathchat 实现 AI 医疗诊断**：/r/singularity 介绍了 Modella 的 Pathchat，这是一个[**专为医学和病理学目的设计的多模态 AI 模型，能够识别肿瘤并诊断癌症患者**](https://v.redd.it/syajrw6t2cbd1)。
- **Thrive AI Health 提供个性化教练服务**：/r/artificial 讨论了 Thrive AI Health，[**这是一个由 OpenAI Startup Fund 资助的超个性化 AI 健康教练**](https://time.com/6994739/ai-behavior-change-health-care/)。
- **Odyssey AI 旨在彻底改变视觉特效**：在 /r/OpenAI 中，Odyssey AI 正在致力于[“好莱坞级”的视觉特效，基于真实世界的 3D 数据进行训练](https://www.reddit.com/r/OpenAI/comments/1dyv1ve/odyssey_ai_working_on_hollywoodgrade_visual_fx/)。其目标是大幅缩短电影制作时间和成本。

**AI 能力与担忧**

- **AI 通过面部预测政治信仰**：/r/artificial 分享了一项研究，显示了 [**AI 仅凭面部特征推断政治倾向的能力**](https://www.psypost.org/artificial-intelligence-can-predict-political-beliefs-from-expressionless-faces/)。
- **红杉资本警告潜在的 AI 泡沫**：在 /r/OpenAI 中，红杉资本警告称，[**AI 每年需要产生 6000 亿美元的收入才能证明当前的硬件支出是合理的**](https://www.reddit.com/r/OpenAI/comments/1dynim4/ai_bubble_ahead_sequoia_capital_warns_that_ai/)。即使是乐观的收入预测也达不到这一水平，这表明潜在的过度投资可能导致泡沫。
- **中国面临 AI 模型过剩且利用不足的问题**：/r/artificial 报道了中国 AI 模型过剩的情况，百度 CEO 称由于 100 多个 LLM 缺乏实际应用，这是[“资源的重大浪费”](https://www.yahoo.com/tech/chinas-ai-model-glut-significant-171150163.html)。

**迷因与幽默**

- **推特用户误解 AI 技术**：/r/singularity 分享了一个关于[**推特上普通大众对 AI 技术缺乏了解**](https://i.redd.it/tfjwllnj4abd1.jpeg)的幽默看法。
- **AI 构思老龄化马里奥游戏**：/r/singularity 展示了一个幽默的 AI 生成的游戏封面，[**描绘了一个患有背痛的老年马里奥**](https://i.redd.it/d6jvf86ikfbd1.jpeg)。

---

# AI Discord 回顾

> 摘要的摘要的摘要


**1. 大语言模型进展**

- **细腻的语音模型涌现**：[JulianSlzr](https://x.com/julianslzr/status/1810303916686577858) 强调了 **GPT-4o 精致的轮询式（turn-based）语音模型**与 **Moshi 未经雕琢的全双工（full-duplex）模型**之间的细微差别。
   - **Andrej Karpathy** 等人对这些差异发表了看法，展示了 **AI 模型中涌现出的多样化语音能力**。
- **Gemma 2 更新后表现出色**：**Gemma2:27B 模型**收到了来自 **Ollama** 的重大更新，修复了之前的问题，其**令人印象深刻的性能**赢得了广泛好评，正如[这段 YouTube 视频](https://youtu.be/38ae7hqzX5s)所示。
   - 社区成员称赞了该模型的**转变**，在经历了之前的输出不连贯问题后，称其表现“令人难以置信”。
- **Supermaven 发布 Babble**：**Supermaven** 宣布推出其最新的语言模型 **Babble**，该模型拥有 **100 万 token 的海量上下文窗口**，比其之前提供的产品[大 2.5 倍](https://x.com/SupermavenAI/status/1808256013788676438)。
   - 此次升级有望凭借其广阔的上下文处理能力**丰富对话场景**。

**2. 创新 AI 研究前沿**

- **测试时训练（Test-Time Training）提升 Transformer 性能**：一篇新论文提出使用**测试时训练（TTT）**，通过在**未标记的测试实例上进行自监督学习**来改进模型预测，在 ImageNet 等基准测试中表现出显著提升。
   - **TTT 可以集成到线性 Transformer 中**，实验设置中用神经网络替代线性模型显示出**增强的性能**。
- **无矩阵乘法（MatMul-Free）模型革新 LLM**：研究人员为大语言模型开发了**矩阵乘法消除技术**，在**保持十亿参数规模强劲性能**的同时，[显著降低了内存使用量](https://arxiv.org/abs/2406.02528)，实验显示比未优化的基准线**降低了高达 61%**。
   - 一种名为 **Test-Time-Training 层**的新架构用机器学习模型取代了 RNN 隐藏状态，实现了**线性复杂度，并达到或超越了顶尖的 Transformer**，正如[最近的一条推文](https://x.com/karansdalal/status/1810338845659131940)所宣布的那样。
- **生成式变色龙（Generative Chameleon）出现**：首个**生成式变色龙模型**已发布，其详细研究内容记录在 [arXiv 论文](https://arxiv.org/pdf/2407.06135)中。
   - 研究界渴望调查该模型**适应各种绘画风格**的能力，这有可能彻底改变数字艺术创作。

**3. AI 工具化与部署进展**

- **Unsloth 加速模型微调**：[Unsloth AI 的新文档网站](https://docs.unsloth.ai/)详细介绍了在微调 **Llama-3** 和 **Gemma** 等大语言模型时，如何在不牺牲准确性的情况下**将速度提高一倍并将内存使用量减少 70%**。
   - 该网站指导用户**创建数据集、部署模型**，甚至通过建议从 **llama.cpp 仓库**构建来解决 **gguf 库**的问题。
- **LlamaCloud 简化数据集成**：**LlamaCloud** 的测试版发布，承诺提供一个**用于非结构化数据解析、索引和检索的托管平台**，目前已为热切的测试者开放[候补名单](https://docs.google.com/forms/d/e/1FAIpQLSdehUJJB4NIYfrPIKoFdF4j8kyfnLhMSH_qYJI_WGQbDWD25A/viewform)。
   - 通过集成 **LlamaParse** 进行高级文档处理，LlamaCloud 旨在**简化跨不同后端的同步**，实现无缝的 LLM 集成。
- **Crawlee 简化网页抓取**：**Crawlee for Python** 发布，具有 **HTTP 和 Playwright 的统一接口**以及**自动扩展和会话管理**等功能，详见 [GitHub](https://github.com/apify/crawlee-python) 和 [Product Hunt](https://www.producthunt.com/posts/crawlee-for-python)。
   - 凭借对**网页抓取和浏览器自动化**的支持，Crawlee 被定位为 Python 开发人员从事 **AI、LLM、RAG 或 GPT 数据提取**的强大工具。

**4. AI 伦理辩论与法律影响**

- **Copilot 诉讼范围缩小**：针对 [GitHub Copilot](https://www.theregister.com/2024/07/08/github_copilot_dmca/) 涉嫌在未注明出处的情况下复制代码的大部分指控已被驳回，在这场涉及 GitHub、Microsoft 和 OpenAI 的法律诉讼中仅剩下两项指控。
   - 最初的集体诉讼认为 Copilot 在开源软件上进行训练构成了知识产权侵权，这引起了开发者社区的关注。
- **AI 的社会影响受到审视**：讨论揭示了**对 AI 社会影响的担忧**，特别是关于潜在的**成瘾问题以及未来监管的必要性**。
   - 成员们强调，鉴于 AI 在**各个领域的变革性特质**，采取前瞻性措施迫在眉睫。
- **面向开发者的 Anthropic 额度**：社区成员询问 Anthropic 是否存在**类似于 OpenAI 的额度系统**，寻求在其平台上进行**实验和开发**的机会。
   - 对话强调了人们对**获取 Anthropic 产品**（类似于 OpenAI 的举措）以促进 AI 研究和探索的兴趣日益增长。

**5. Model Performance Optimization**

- **Deepspeed 提升训练效率**：**[Deepspeed](https://huggingface.co/AI-Sweden-Models/DeepSpeed)** 能够实现在单张 **RTX 3090** 上训练 **25 亿参数模型**，并获得更高的 Batch Size 和效率。
   - 一位成员分享了他们使用 Deepspeed 的成功经验，引发了人们对其在更易获得的训练方案中潜力的兴趣。
- **FlashInfer 的速度秘诀**：**[FlashInfer Kernel Library](https://github.com/flashinfer-ai/flashinfer)** 支持 **INT8** 和 **FP8** Attention Kernel，有望提升 LLM Serving 的性能。
   - AI 社区热衷于测试和讨论 FlashInfer 对模型效率的影响，反映出极高的期待。
    


**6. Generative AI in Storytelling**

- **生成式 AI 影响叙事**：**[Medium 文章](https://medium.com/@shikharnautiyal29/the-impact-of-generative-ai-on-storytelling-and-narrative-creation-d37898cc0126)** 探讨了生成式 AI 给叙事和叙事创作带来的深刻变革，开启了丰富的叙事机会。
   - **KMWorld** 重点介绍了塑造知识管理领域的 AI 领导者，强调了生成式 AI 的变革潜力。
- **AI 在文化影响中的角色**：关于 AI 社会影响的讨论强调了对成瘾和**未来监管必要性**的担忧，反映了 AI 技术的变革性。
   - 社区强调了采取前瞻性措施以应对 AI 的文化效应和社会影响的紧迫性。
    


**7. AI in Education**

- **教师探索使用 CommandR 构建学习平台**：一位公立学校教师正在利用 **[CommandR 的 RAG 优化特性](https://link.to.commandr)** 开发一个**教学与学习平台**。
   - 该倡议得到了社区的积极回应和协助提议，展示了集体的热情。
- **Claude 竞赛提醒**：**[Build with Claude 竞赛](https://x.com/alexalbert__/status/1810376544734556540)** 提供价值 **$30k** 的 Anthropic API 额度，即将结束。
   - 提醒社区成员参与，强调了该竞赛对开发者和创作者的重要性。
    

---

# PART 1: High level Discord summaries

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Intel 为 HF 模型带来的推理加速**：一个新的 **[GitHub 仓库](https://github.com/sleepingcat4/intel-hf)** 展示了如何在 **Intel CPU** 上更高效地运行 HF 模型，这对拥有 Intel 硬件的开发者来说是一个福音。
   - 该资源的出现填补了 **Intel 特定** 指导的空白，可能是提升模型运行性能的宝库。
- **Gemma 更新后的巨大进步**：**Gemma2:27B** 模型得到了大幅增强，一段富有启发性的 **[YouTube 视频](https://youtu.be/38ae7hqzX5s)** 展示了这一点，并获得了社区的高度认可。
   - Ollama 的及时更新修正了之前的问题，现在 Gemma 因其出色的性能而广受好评。
- **关于上下文窗口的巧妙考量**：LLM 训练期间的 VRAM 占用情况各不相同，而 **context window**（上下文窗口）大小是这一计算难题的核心。
   - 社区成员交流了经验，分享了 padding（填充）和 max token 调整是保持 VRAM 负载稳定的关键。
- **深入渗透测试主题**：焦点聚集在 **PentestGPT** 上，一场回顾会议将深入探讨 **AI pentesting**（AI 渗透测试）的细微差别。
   - 随着一篇 **[专题论文](https://arxiv.org/abs/2308.06782)** 的深入剖析，该小组正准备推进关于 AI 稳健渗透测试实践的对话。
- **叙事细微差别与 Generative AI**：Generative AI 对故事创作的影响成为核心话题，一篇 **[Medium 文章](https://medium.com/@shikharnautiyal29/the-impact-of-generative-ai-on-storytelling-and-narrative-creation-d37898cc0126)** 揭示了其在叙事方面的潜力。
   - 与此同时，**KMWorld** 报道了正在塑造知识管理领域的 AI 领导者，并重点关注了 Generative AI。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 文档释放效率**：Unsloth AI 新的 [文档网站](https://docs.unsloth.ai/) 提升了 **Llama-3** 和 **Gemma** 等模型的训练效率，在不牺牲准确性的情况下，将速度提高了一倍，并将内存占用减少了 70%。
   - 该网站的教程简化了数据集创建和模型部署的过程，甚至通过建议从 [llama.cpp](https://github.com/unslothai/unsloth) 仓库进行构建来解决 **gguf 库** 的问题。
- **LlaMA 3 的微调技巧**：[Modular Model Spec](https://modular-model-spec.vercel.app) 的开发旨在优化 **LLaMA 3** 等 AI 模型的训练流程。
   - SiteForge 将 **LLaMA 3** 整合到其 [网页设计生成](https://siteforge.io) 中，承诺提供 AI 驱动的革命性设计体验。
- **利用 LlaMA 3 精通医疗翻译**：关于使用 **Llama 3** 将 5000 条医疗记录翻译成瑞典语的讨论，揭示了该模型在瑞典语方面的熟练程度和应用潜力。
   - 用户验证了针对瑞典语特定应用微调 **Llama 3** 的实用性，正如 [AI-Sweden-Models/Llama-3-8B-instruct](https://huggingface.co/AI-Sweden-Models/Llama-3-8B-instruct) 所展示的那样。
- **数据集获得合成增强**：一种 AI 方法正在创建合成聊天数据集，为超过 **100 万条语句** 提供逻辑依据和上下文，从而丰富了 **PIPPA 数据集**。
   - 在医疗对话领域，使用现有的微调模型暗示可以跳过预训练，这与 [研究](https://arxiv.org/abs/2308.05884) 中提到的益处相呼应。
- **利用无矩阵乘法模型重构 LLM**：一项 [研究](https://arxiv.org/abs/2406.02528) 揭示，LLM 可以舍弃矩阵乘法，在十亿参数规模下可节省 61% 的内存。
   - **Test-Time-Training layers**（测试时训练层）作为 RNN 隐藏状态的替代方案出现，并在具有线性复杂度的模型中得到展示，这一点在 [社交媒体](https://x.com/karansdalal/status/1810338845659131940) 上引起了关注。

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **集成之谜：Triton 邂逅 PyTorch**：好奇的开发者们正在探索如何将 **Triton kernels** 集成到 **PyTorch** 中，特别是旨在为 `torch.compile` 自动优化注册自定义函数。
   - 讨论仍在进行中，技术社区正热切期待关于这一挑战的权威指南或解决方案。
- **纹理对话：Vulkan 后端算子**：为什么 executorch 的 **Vulkan backend** 在其算子中使用纹理（textures）？这个问题引发了成员们的一系列探究。
   - 目前尚未达成具体结论，该话题仍保持开放，以待进一步的见解和探索。
- **INT8 与 FP8：FlashInfer 的速度秘诀**：**FlashInfer Kernel Library** 的发布引起了广泛关注，它支持 **INT8** 和 **FP8** 注意力算子，有望提升 LLM 推理服务的性能。
   - 附带 [库链接](https://github.com/flashinfer-ai/flashinfer)，AI 社区正热衷于测试并讨论其对模型效率的潜在影响。
- **量化清晰化：校准是关键**：量化讨论转向了技术深度，揭示了在使用静态量化时，配合数据进行适当的校准（calibration）是必不可少的。
   - 一个 [GitHub pull request](https://github.com/pytorch/ao/pull/487) 重点强调了这一必要性，引发了对该实践的技术深挖。
- **分而治之：利用 Ring Attention 进行 GPU 切分**：跨 GPU 切分 **KV cache**：这是一个正通过 ring attention 解决的挑战，特别是在 **AWS g5.12xlarge** 实例环境下。
   - 针对此实现的最佳拓扑结构的追求非常热烈，成员们分享了如 [此 gist](https://gist.github.com/msaroufim/abbb01c5bb037c2b1009df4a0baeb74b) 之类的资源以辅助研究。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **GIF 引发笑声：Marcus 对阵 LeCun**：一位用户分享了一个幽默的 GIF，捕捉了 **Gary Marcus** 和 **Yann LeCun** 辩论的瞬间，在不涉及技术细节的情况下突显了不同的 AI 观点。
   - 它以轻松的方式展现了 AI 专家之间有时会发生的*紧张交流*，吸引了社区的关注 [GIF 来源](https://tenor.com/view/gary-marcus-yann-lecun-lecun-ai-machine-learning-gif-9041590723446061255)。
- **Hermes 2 Pro：性能新高度**：**Hermes 2 Pro** 因其增强的 Function Calling 和 JSON Structured Outputs 而受到赞誉，在基准测试中展现了强劲的提升 [Hermes 2 Pro](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B)。
   - 该平台的进步令社区着迷，反映了 LLM 能力和用例的进展。
- **合成数据落地：Distilabel 占据主导**：**Distilabel** 成为合成数据生成的卓越工具，因其高效和高质量的输出而受到称赞 [Distilabel 框架](https://argilla-io.github.io/distilabel/1.2.1/)。
   - 成员们建议将 LLM 输出与合成数据协调一致，从而增强 AI 工程师的开发和调试工作流。
- **Sonnet 3.5 的 PDF 难题与解决方案**：由于缺乏直接使用 **Sonnet 3.5 API** 处理 PDF 的方案，社区开始探索替代路径，如 [Marker Library](https://github.com/VikParuchuri/marker)。
   - **Everyoneisgross** 强调了 Marker 将 PDF 转换为 Markdown 的能力，提议将其作为需要更好模型兼容性场景下的变通方案。
- **RAG 的新前沿：RankRAG 革命**：**RankRAG** 的方法论取得了飞跃，通过训练一个 LLM 同时进行排序（ranking）和生成（generation）获得了显著收益 [RankRAG 方法论](https://x.com/rohanpaul_ai/status/1810329112558371089)。
   - 这种被称为“Llama3-RankRAG”的方法展示了引人注目的性能，在多项基准测试中优于同类产品。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Chris Lattner 备受期待的访谈**：The Primeagen 将在 [Twitch](https://www.twitch.tv/theprimeagen) 上对 **Chris Lattner** 进行一场备受期待的访谈，引发了社区内的兴奋和期待。
   - 随后引发了热烈讨论，helehex 预告了明天将有一个涉及 **Lattner** 的特别活动，进一步提升了 Modular 社区的热度。
- **前沿 Nightly 版 Mojo 发布**：**Mojo 编译器最新 Nightly 版本** [2024.7.905](https://github.com/modularml/mojo/compare/bc18cb454cd1bf7384da6eb86f79907b589c2419...d836be2d478bad12588843ce2b781e3c611df390) 引入了多项改进，例如增强的 `memcmp` 使用以及针对条件一致性（conditional conformances）的精细化参数推断。
   - 开发者们敏锐地研究了 [更新日志](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)，并讨论了条件一致性对类型组合的影响，特别关注了一个 [重要的 commit](https://github.com/modularml/mojo/commit/97d70d3ecdfa289e61c33c323c3e04a71c19038a)。
- **Mojo 的 Python 超能力释放**：**Mojo 与 Python 的集成**已成为讨论的核心，正如 [Mojo 官方文档](https://docs.modular.com/mojo/manual/python/) 所述，旨在评估利用 Python 庞大包生态系统的潜力。
   - 讨论转向了 Mojo 潜力成为 Python 超集的话题，强调了通过 Python 的多功能性为 Mojo 赋能的战略举措。
- **时钟精度困境**：对时钟校准的详细检查显示，在连续使用 `_clock_gettime` 调用时存在 1 ns 的微小但关键的差异，揭示了对高精度测量的需求。
   - 这一发现促使了对时钟不准确性影响的进一步分析，强调了其在时间敏感型应用中的重要性。
- **向量化 Mojo 马拉松开辟性能之路**：**Mojo 马拉松对向量化进行了测试**，发现在不同宽度的向量化下性能存在变量，有时 **width 1** 的表现甚至优于 **width 2**。
   - 社区成员强调了调整基准测试以包含对称和非对称矩阵的重要性，使测试与真实的 **地理和图像处理（geo and image processing）** 场景保持一致。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **让 LLM 发声：自定义语音引发关注**：社区正在探索将 Eleven Lab 自定义语音与 LM Studio 集成，提议通过 **服务器功能的自定义编程** 来实现文本转语音。
   - 讨论建议保持谨慎，因为尽管有 **Claude** 等工具可以辅助开发，但仍需要额外的编程工作。
- **InternLM：滑动上下文窗口惊艳全场**：成员们称赞 **InternLM** 的滑动上下文窗口，即使在内存过载时也能保持连贯性。
   - 带有截图的对话显示了 InternLM 如何通过遗忘早期消息来进行调整，但令人赞赏的是它依然能保持在正轨上。
- **网页工匠：AI 助力自定义爬虫腾飞**：一位成员展示了使用 Claude 以极快速度编写 **Node.js 网页爬虫** 的成果，引发了关于 AI 在工具创建中作用的讨论。
   - “我得到了 78 行完全符合我要求的代码，”他们分享道，强调了 AI 对开发效率的影响。
- **AI 代码：谨慎处理**：AI 代码生成引发了社区辩论；它是一个有价值的工具，但应谨慎使用，并理解底层代码逻辑。
   - 共识是：使用 AI 进行快速开发，但要验证其输出以确保代码质量和可靠性。
- **AMD 的关键显卡：GPU 成为焦点**：成员们讨论了使用 RX 6800XT 进行 LLM 多 GPU 设置，深入了解了 **LM Studio** 对资源和配置的处理。
   - 随着关于 AMD ROCm 支持寿命的辩论展开，人们以面向未来的视角权衡了 RX 6800XT 和 7900XTX 之间的选择。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- ****TTT 与 Delta Rule 的探戈****：讨论揭示了当使用 mini batch size 为 1 时，**TTT-Linear** 与 **delta rule** 保持一致，从而优化了模型预测的性能。
   - 进一步的讨论包括 **rwkv7** 架构计划利用改进的 delta rule，以及 **ruoxijia** 阐明了 TTT-linear 并行化的可能性。
- ****Shapley 搅动数据归因****：**In-Run Data Shapley** 作为一个创新项目脱颖而出，有望为 pre-training 期间的实时数据贡献评估提供可扩展的框架。
   - 其目标是从训练中排除有害数据，从根本上影响模型能力，并根据 AI 社区的观点澄清“涌现”（emergence）的概念。
- ****规范化梯度风暴****：一种新兴的**梯度归一化（gradient normalization）**技术旨在解决深度网络挑战，如臭名昭著的梯度消失或梯度爆炸。
   - 然而，它并非没有缺点，AI 社区强调了诸如 batch-size 依赖性以及相关的跨设备通信故障等问题。
- ****RNN 挑战 Transformer 巨头****：新兴的 **Mamba 和 RWKV** RNN 架构引发了关注，因为它们提供恒定的内存占用，并在 perplexity 任务中被证明是 Transformer 强大的对手。
   - 内存管理效率及其对 long-context recall 的影响是当前讨论中理论和实证研究的焦点。
- ****弥合大脑容量难题****：最近的一项研究挑战了人们对大脑容量进化的直观认知，特别是人类与体型最大的动物相比，后者并没有按比例拥有更大的大脑。
   - 对话还涉及了神经元密度在跨物种智力映射中的作用，进一步复杂化了对智力及其进化优势的理解。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **从像素到完美：提升分辨率**：辩论集中在 fine-tuning 模型时，从 **512x512 分辨率**开始是否比直接跳转到 **1024x1024** 更有优势。
   - 共识倾向于渐进式缩放，以便在控制计算成本的同时获得更好的 gradient propagation。
- ****Booru 之战：标签张力****：围绕使用 **booru tags** 训练 AI 的讨论变得激烈，支持既定词汇的人群与支持更自然语言标签的人群之间存在分歧。
   - 争论强调了在模型标签的精确性和泛化性之间取得平衡的必要性。
- **AI 与社会：计算文化成本**：成员们就 AI 在社会中的角色展开对话，思考其对成瘾的影响，并思索**未来监管**的必要性。
   - 该小组强调了鉴于 AI 技术的变革性质，采取主动措施的紧迫性。
- **Roop-Unleashed：革命性的替代方案**：**Roop-Unleashed** 被推荐为视频换脸的卓越解决方案，取代了已过时的 mov2mov 扩展。
   - 该工具因其一致性和易用性而受到赞誉，标志着社区偏好的转变。
- **模型组合：SD 工具与技巧**：针对 Stable Diffusion 模型和扩展进行了热烈的**推荐**交流，重点关注 pixel-art 转换和 inpainting 等任务。
   - 成员们提到了 **Zavy Lora** 和带有 **IP adapters** 的 **comfyUI** 等工具，分享经验并提升了同伴的知识储备。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **DALL-E 的竞争对手崭露头角**：讨论重点关注了 **StableDiffusion** 以及工具 **DiffusionBee** 和 **automatic1111**，认为它们是 DALL-E 的主要对手，因其能为用户提供更高的质量和控制力而受到青睐。
   - 这些模型还因其对不同操作系统的兼容性而受到认可，重点是在 Windows 和 Mac 上的本地使用。
- **文本检测器未能通过宪法测试**：社区情绪对 **AI 文本检测器** 的可靠性表示怀疑，并举例说明了误报内容的情况，滑稽的是其中甚至包括了 **美国宪法**。
   - 辩论仍在继续，没有明确的结论，反映了区分 AI 生成文本与人类创作文本的复杂性。
- **GPT 的变现前景尚不明朗**：用户询问了 **GPTs 变现** 的潜力，但讨论陷入停滞，没有出现具体的细节。
   - 这个话题在社区内似乎缺乏实质性的参与或解决方案。
- **VPN 导致 GPT-4 连接中断**：用户报告称，当 **开启 VPN** 时，GPT-4 的交互会受到干扰，建议禁用 VPN 可以缓解问题。
   - 还提到影响 GPT-4 服务的 **服务器问题** 已得到解决，但未提供具体细节。
- **内容创作者渴望前沿建议**：内容创作者寻求 **5-10 个以趋势为中心的内容创意** 以促进受众增长，并询问了追踪内容表现的关键指标。
   - 他们还探索了结构化内容日历的策略，强调了有效的 **受众参与** 和平台优化的必要性。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaCloud 开启 Beta 测试**：**LlamaCloud** 的 Beta 版发布，承诺提供一个用于非结构化数据解析、索引和检索的高级平台，目前已面向热切的测试者 [开放候补名单](https://docs.google.com/forms/d/e/1FAIpQLSdehUJJB4NIYfrPIKoFdF4j8kyfnLhMSH_qYJI_WGQbDWD25A/viewform)。
   - 该服务旨在优化 LLMs 的数据质量，集成了用于高级文档处理的 **LlamaParse**，并力求简化跨各种数据后端的同步。
- **图技术迎来 Llama 式创新**：LlamaIndex 通过一系列展示 **Property Graphs** 的 [新视频系列](https://twitter.com/llama_index/status/1810410943215710510) 推动参与，这是一项强调节点和边中模型复杂性的协作成果。
   - 这一教育推广由 **mistralai, neo4j, 和 ollama** 团队合作提供支持，在复杂的文档关系与 AI 易用性之间架起了一座桥梁。
- **聊天机器人向电子商务的复杂化迈进**：为了增强客户互动，该公会的一项工程推进重点是利用关键词搜索和元数据过滤器改进 **RAG** 聊天机器人，以解决复杂的建筑项目查询。
   - 这种方法涉及一种混合搜索机制，从而实现更细致的后续问题交流，旨在实现对话精准度的飞跃。
- **FlagEmbeddingReranker 的导入困局**：社区的排错工作建议独立安装 `peft` 以克服 `FlagEmbeddingReranker` 遇到的导入错误，帮助用户最终解决了技术难题。
   - 这一小插曲凸显了设置机器学习环境时经常隐藏的复杂性，包依赖关系可能会成为一个隐蔽的障碍。
- **Groq API 的速率限制难题**：AI 工程师在利用 **Groq API** 进行索引时遇到了 LlamaIndex 中的 429 速率限制错误，凸显了与 OpenAI 嵌入模型同步时的挑战。
   - 讨论转向了 API 交互的复杂性以及采取战略性方法规避此类限制的必要性，以保持无缝的索引体验。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 的 API 与 UI 表现存在差异**：关于 Perplexity 中 **API 和 UI 结果之间明显差异** 的辩论被触发，特别是在未应用 Pro 版本或溯源功能的情况下。
   - 考虑了一个积极的解决方案，即涉及 *labs* 环境，以在没有额外 Pro 功能的情况下测试 API 和 UI 输出之间的一致性。
- **PPLX 集成中的 Nodemon 问题**：在为使用 **PPLX** 库的项目配置 **Nodemon** 时出现了问题，尽管本地执行正确且对 **tsconfig.json** 进行了调整，但一名成员仍未成功。
   - 该用户寻求其他 AI 工程师的见解，分享了指示可能与 **PPLX** 设置相关的模块缺失问题的错误日志。

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **DeepSpeed 在低成本下表现出色**：一位创新工程师报告称，使用 **DeepSpeed** 在单块 **RTX 3090** 上成功训练了一个 **25 亿参数模型**，且有潜力实现更高的 **batch sizes**。
   - 这场对话激发了人们探索在资源受限情况下进行高效训练的边界，暗示了更易于获取的训练方案。
- **OpenAI 的编程助手在法庭获胜**：加州法院部分驳回了针对 **Microsoft** 的 **GitHub Copilot** 和 **OpenAI** 的 **Codex** 的诉讼，这是一项关键的法律裁决，展示了 AI 系统在应对版权指控时的韧性。
   - 社区正在剖析这一法律进展对 AI 生成内容的影响。[阅读更多](https://the-decoder.com/court-ruling-suggests-ai-systems-may-be-in-the-clear-as-long-as-they-dont-make-exact-copies/)。
- **Chameleon：模仿大师的生成模型**：首个 **generative chameleon model** 已发布，其详细研究内容记录在 [arXiv](https://arxiv.org/pdf/2407.06135) 的论文中。
   - 研究社区渴望调查该模型适应各种绘画风格的能力，这可能会彻底改变数字艺术创作。
- **扩展复数值前沿**：一位先锋成员在扩展用于视觉任务的复数值神经网络（**complex-valued neural networks**）深度时遇到了挑战。
   - 尽管在扩展方面存在障碍，一个仅有 65k 参数的复数值模型在 **CIFAR-100** 上的准确率表现优于其 400k 参数的实数值对应模型，展现了良好的前景。
- **揭秘 Diffusion：资源库**：一个新的 GitHub 仓库提供了一个直观的基于代码的课程，用于掌握非常适合在普通硬件上训练的图像 **diffusion** 模型。[探索仓库](https://github.com/swookey-thinky/mindiffusion)。
   - 该资源旨在通过简明的课程和教程培养实践理解，并邀请各界贡献以完善其教育内容。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **突破配额上限：OpenRouter 上的超限困扰**：用户在使用 **gemini-1.5-flash** 模型时遇到了来自 `aiplatform.googleapis.com` 的 “**Quota exceeded**” 错误，这表明存在 Google 施加的限制。
   - 有关使用情况的见解，请查看 [Activity | OpenRouter](https://openrouter.ai/activity)；有关自定义路由解决方案，请参阅 [Provider Routing | OpenRouter](https://openrouter.ai/docs/provider-routing#custom-routing)。
- **看到 None：OpenRouter 上的图像问题挑战**：有报告称在 **gpt-4o**、**claude-3.5** 和 **firellava13b** 等模型上进行**图像查看**时出现 **None** 响应，用户对功能的确认情况不一。
   - 这表明是一个选择性问题，并非影响所有用户，需要详细检查个人用户的配置。
- **Dolphin 潜水：排查 LangChain 最新成员的问题**：用户在将 OpenRouter 上的 **Dolphin 2.9 Mixstral** 作为语言工具集成到 **LangChain** 时面临挑战。
   - 未提供问题的技术细节，表明可能存在兼容性问题或配置错误。
- **JSON 震荡：Mistralai Mixtral 偶尔的支持失效**：错误 “not supported for **JSON mode/function calling**” 随机困扰着 **mistralai/Mixtral-8x22B-Instruct-v0.1** 的用户。
   - 故障排除确定 **Together** 是与该重复错误相关的供应商，凸显了进一步调查的必要性。
- **翻译的跷跷板：评估 LLM 作为语言学家的表现**：讨论集中在 **LLM** 在语言翻译任务中相对于专业模型的有效性，强调了偏好和性能指标。
   - 讨论考虑了现代 **decoder-only** 模型与真正的 **encoder/decoder transformers** 在准确翻译能力方面的可靠性对比。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Claude 竞赛进入尾声，代码换现金**：Build with Claude 竞赛即将结束，正如社区提醒的那样，该竞赛为开发者提供 **3 万美元的 Anthropic API 额度**。
   - **Alex Albert** 在这篇[揭秘帖子](https://x.com/alexalbert__/status/1810376544734556540)中提供了更多关于参与方式和背景的信息。
- **语音模型各显神通**：**GPT-4o** 和 **Moshi** 的语音模型因其截然不同的风格而备受关注，GPT-4o 拥有精致的轮询式（turn-based）方法，而 Moshi 则是原始的全双工（full-duplex）模式。
   - 感谢 [JulianSlzr](https://x.com/julianslzr/status/1810303916686577858?s=46&t=PW8PiFwluc0tdmv2tOMdEg) 和 **Andrej Karpathy** 的见解，让这场对话得以展开。
- **AI 在数学奥林匹克中大放异彩**：**Thom Wolf** 赞扬了 AI 数学奥林匹克（AIMO），**Numina** 与 Hugging Face 的联手展示了 AI 解决问题的强大实力。
   - 欲了解深入内容，请查看 [Thom Wolf 的推文串](https://x.com/Thom_Wolf/status/1809895886899585164)，其中详细介绍了比赛亮点和 AI 取得的成就。
- **Babble 拥有更强大的“大脑”**：Supermaven 推出了他们最新的语言模型 **Babble**，其上下文窗口进行了巨大升级，可容纳 100 万个 token。
   - [SupermavenAI 的公告](https://x.com/SupermavenAI/status/1808256013788676438)预示着它比前代产品实现了 2.5 倍的飞跃，并承诺提供更丰富的对话体验。
- **Lillian Weng 视角下的 LLM 缺陷**：**Lillian Weng** 在博客中探讨了 LLM 中的“幻觉”现象，研究了该现象的起源和分类。
   - [探索详情](https://lilianweng.github.io/posts/2024-07-07-hallucination)，了解为什么大语言模型有时会偏离事实。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LLMWhisperer 解码密集文档**：**LLMWhisperer** 在解析复杂 PDF 方面表现出色，建议在 **LangChain** 中集成来自 [Pydantic 或 zod](https://www.youtube.com/watch?v=dC7EhnEIdDA) 的 schema，以增强数据提取能力。
   - 通过结合逐页 LLM 解析和 JSON 合并，用户发现 LLMWhisperer 对于从冗长文档中提取精炼数据非常有用。
- **Crawlee for Python 首次亮相引发关注**：Apify 宣布推出 **Crawlee for Python**，并在 [GitHub](https://github.com/apify/crawlee-python) 和 [Product Hunt](https://www.producthunt.com/posts/crawlee-for-python) 上展示了统一接口和自动扩展等功能。
   - 凭借对 HTTP、Playwright 和会话管理的支持，Crawlee 被定位为 Python 开发者进行网页爬取的强大工具。
- **Llamapp 锁定本地化 RAG 响应**：[Llamapp](https://github.com/rajatasusual/llamapp) 作为一个本地检索增强生成器（RAG）出现，融合了检索器和语言模型以实现精准的回答准确度。
   - 通过启用倒数排名融合（Reciprocal Ranking Fusion），Llamapp 在提供定制化响应的同时，始终保持基于原始事实。
- **Slack Bot Agent 革命进行中**：一份[指南](https://git.new/slack-bot-agent)展示了如何利用 LangChain 和 ChatGPT 构建 Slack Bot Agent，用于 PR 评审自动化。
   - 文档指出了一个分步过程，整合了多个框架来优化 PR 评审工作流。
- **Rubik’s AI Pro 向 Beta 测试人员开放**：[Rubik's AI Pro](https://rubiks.ai/) 邀请 AI 爱好者进行 Beta 测试，展示了其使用“RUBIX”代码的科研辅助和搜索能力。
   - 他们强调了对高级模型的访问和高级版试用，支撑其对全面搜索解决方案的追求。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **OI 以示例精度执行**：通过引入 [代码指令示例](https://link.to/examples)，**OI 的执行** 与 **assistant.py** 保持一致，展示了其多功能的技能指令处理能力。
   - 这一功能增强表明其功能容量有所提升，与复杂语言模型的发展趋势相吻合。
- **Qwen 2.7B 的随机伪影**：**Qwen 2 7B 模型** 在 128k 上下文处理方面表现出色，但偶尔会生成随机的 '@' 符号，导致输出中出现意外的故障。
   - 虽然该模型的鲁棒性显而易见，但这些异常现象凸显了对其生成模式进行精细化调整的必要性。
- **本地视觉模式的兼容性疑问**：关于使用参数 '**--model i**' 启用 **Local vision mode** 的讨论引发了对其兼容性以及是否能开启多模态用例的探讨。
   - 凭借此类功能，工程师们正在探索整合多种输入模态，以构建更全面的 AI 系统。
- **GROQ 与 OS 模式的同步协作**：出现了一系列关于在 **OS mode** 下实现 **GROQ** 的咨询，并探讨了在这种情况下使用多模态模型的必要性。
   - 这些对话强调了 AI 工程领域对更无缝、更具凝聚力的工作流的积极追求。
- **使用 Open Interpreter 解析坐标**：用户询问了 **Open Interpreter** 解析屏幕坐标的方法，这表明人们正在深入研究该模型的交互能力。
   - 理解这一机制对于旨在利用 AI 进行更动态、更精确的用户界面交互的工程师来说至关重要。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Ampere 和 Ada 架构支持 NV=1**：讨论显示 **NV=1** 的支持主要针对 **Ampere** 和 **Ada** 架构，而早期模型则有待社区驱动的解决方案。
   - **George Hotz** 介入澄清，**Turing 架构显卡** 确实是兼容的，正如 [GSP 固件仓库](https://github.com/NVIDIA/open-gpu-kernel-modules) 中所述。
- **Karpathy 的课程助力理解 tinygrad 概念**：对于那些想要深入研究 **tinygrad** 的人，推荐了 **Karpathy 的转型教程**，承诺提供对该框架的深刻见解。
   - 这个基于 PyTorch 的视频起到了探索催化剂的作用，促使人们以交互式方式阅读 **tinygrad 文档**。
- **WSL2 在 NV=1 部署中遇到困难**：成员们在 **WSL2** 上部署 **NV=1** 时遇到了困境，面临设备文件缺失以及对 **CUDA** 兼容性不确定的问题。
   - 虽然路径尚不明确，但 NVIDIA 的 [开源 GPU 内核模块](https://github.com/NVIDIA/open-gpu-kernel-modules) 成为热心工程师解决问题的潜在关键。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Copilot 诉讼难题持续**：针对 [GitHub Copilot](https://www.theregister.com/2024/07/08/github_copilot_dmca/) 涉嫌未经授权复制大部分代码的指控大多已被驳回，在涉及 GitHub、Microsoft 和 OpenAI 的法律诉讼中仅剩两项指控。
   - 去年 11 月提出的担忧认为，Copilot 在开源软件上进行训练构成了知识产权侵权；开发者们正等待法院对剩余指控的最终裁定。
- **向量词汇统一**：**Control Vector**、**Steering Vector** 和 **Concept Vectors** 引发了辩论，达成的共识是 **Steering Vectors** 是在语言模型中应用 **Control Vectors** 的一种形式。
   - 此外，**Feature Clamping** 和 **Feature Steering** 之间的区别得到了澄清，并被视为 **RepEng** 领域中的互补策略。
- **Google Flame 的数据撤回**：在发现一个未指明的问题后，'Google Flame' 论文的分数被撤回，社区戏称该失误是否涉及“在测试数据上训练”。
   - **Scott Wiener** 在 [Twitter](https://x.com/hlntnr/status/1810713658860912914) 上抨击了 **a16z** 和 **Y Combinator** 对加州 **SB 1047 AI 法案** 的谴责，在 AI 立法讨论中掀起了风暴。



---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **多 GPU 的痛苦与 Accelerate 的焦虑**：一个 **6 个 H100 GPU** 的配置意外地出现了比预期慢 **10 倍**的训练速度，引发了困扰。
   - 建议围绕根据 [Hugging Face 的故障排除指南](https://huggingface.co/docs/accelerate/basic_tutorials/troubleshooting)调整 **batch size**，并分享代码进行社区驱动的调试。
- **多 GPU 魔法的现实考量**：成员们思考了多 GPU 设置下现实的速度提升，**打破了 10 倍速度增长的神话**，认为 **6-7 倍**是更可行的目标。
   - 关于速度提升的辩论主要基于对通信开销（communication overhead）的担忧以及对吞吐量优化（throughput optimization）的追求。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **教育工作者关注 CommandR 在教室中的应用**：一位公立学校教师正在探索整合一个利用 [CommandR 的 RAG 优化特性](https://link.to.commandr)的**教学平台**。
   - 该教师的倡议得到了社区的积极反应，成员们提供了协助并表达了共同的热情。
- **夜猫子们为深色模式的开发感到欢欣**：许多人期待已久的**深色模式（Dark Mode）**已确认正在开发中，并计划在即将发布的企业级版本中推出。
   - 讨论表明 **Darkordial Mode** 可能会适配给更广泛的受众，暗示免费的 **Coral** 平台用户可能会迎来更新。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Llama.cpp 性能大幅下滑**：在 **NVIDIA GPU** 上从 **0.8.8 版本迁移到 0.8.9 版本**时，发现 **llama.cpp** 出现了意外的 **~25% 性能损失**。
   - 这个问题非常明显，**NVIDIA 3090** GPU 的性能下降到了与上一代 **NVIDIA 3060** 相当的水平。
- **基准测试套件的升级烦恼**：在编写新的基准测试套件时，发现升级 **llamafile** 版本后产生了性能影响。
   - 社区反馈断言最近的更改不应导致性能下降，这给开发者留下了难题。

---

## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Rosebud AI 的文学游戏创作**：在 **Rosebud AI Book to Game Jam** 期间，开发者们受命将书籍转化为互动的**益智游戏、节奏游戏**和**文字冒险游戏**。
   - 此次创作大赛涵盖了对 **Lewis Carroll、China Miéville** 和 **R.L. Stine** 作品的改编，获胜者将于 **PST 时间 7 月 10 日星期三上午 11:30** 公布。
- **游戏开发者利用 AI 挑战文学作品**：参与者在 Rosebud AI Jam 中展示了他们的创意，整合了 **Phaser** 和 **AI 技术**来创作基于文学经典的游戏。
   - 人们对 Rosebud AI Discord 社区官方公告中揭晓的获胜者寄予厚望。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **KAN 论文在 alphaXiv 上的讨论升温**：**KAN** 论文的作者正在 [alphaXiv 论坛](https://alphaxiv.org/abs/2404.19756v4)上积极回答有关他们近期 **arXiv** 论文的问题。
   - 社区成员正与作者互动，讨论 **KAN** 的技术细节和方法论。
- **寻觅信息检索专家**：一位播客主持人正在协调采访来自 **Cohere**、**Zilliz** 以及 **Doug Turnbull** 的专家，主题涉及信息检索（information retrieval）和推荐系统。
   - 他们还寻求在信息检索领域推荐更多的嘉宾，以加入他们的系列节目。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **查询 Anthropic：是否有类似于 OpenAI 的额度计划？**：一位成员询问 Anthropic 是否存在类似于 OpenAI 的额度系统，以寻求实验机会。
   - 该询问反映了开发者对访问 **Anthropic** 平台进行开发和测试的兴趣日益增长，类似于 [OpenAI 10K credit program](https://openai.com/blog/openai-startup-fund/)。
- **了解 Anthropic 的可访问性：额度难题**：社区成员对 **Anthropic** 为开发者提供的支持措施感到好奇，并询问其是否与 OpenAI 的额度计划类似。
   - 这次对话强调了需要关于 **Anthropic** 服务的更清晰信息，以促进 AI 研究和探索。



---


**Alignment Lab AI Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---


**OpenAccess AI Collective (axolotl) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---


**Torchtune Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---


**DiscoResearch Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---

# 第 2 部分：按频道详细摘要和链接


{% if medium == 'web' %}




### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1259947478954938420)** (291 条消息🔥🔥): 

> - `GPTs agents 和知识文件`
> - `OpenAI Platform 变更`
> - `处理代码中的违禁词`
> - `Context window 对 AI 模型的影响`
> - `Gemma 模型问题` 


- **对 GPTs agents 学习能力的误解**：一位成员对 **GPTs agents** 无法从训练后提供的额外信息中学习表示担忧。另一位成员澄清说，[上传的文件被保存为“知识”文件](https://link.to/docs)，agent 会引用这些文件，但它们**不会修改 agent 的基础知识**。
- **OpenAI Platform 上消失的图标**：讨论了 **OpenAI Platform** 侧边栏的变化，据报道有两个图标（一个代表 threads，另一个代表 messages）消失了。
- **在代码中将违禁词存储为 secret**：成员们讨论了存储违禁词的安全性，倾向于使用 **Hugging Face Spaces secrets** 功能或对加密列表使用 **eval**。
- **理解 context window 的影响**：成员们讨论了模型的 **context window**，解释说它决定了模型在不损失性能的情况下可以处理多少个 token。
- **Google Gemma-2B 模型的问题**：成员们在处理 **Gemma-2B** 生成的不连贯文本时遇到了困难，并讨论了潜在的修复方案，例如使用正确的 chat template 提示词配置。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.continue.dev/how-to-use-continue#ask-questions-about-your-codebas">🧑‍🎓 如何使用 Continue | Continue</a>: 在使用 Continue 编写代码时使用 LLM</li><li><a href="https://www.youtube.com/@CodeBullet">Code Bullet</a>: 一个拥有计算机科学学位的笨蛋正在竭尽全力。</li><li><a href="https://huggingface.co/spaces/TencentARC/InstantMesh">InstantMesh - TencentARC 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://docs.coqui.ai/en/latest/docker_images.html">Docker 镜像 - TTS 0.22.0 文档</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2405.05904">在大型语言模型上进行新知识的微调是否会诱发幻觉？</a>: 当大型语言模型通过监督式微调进行对齐时，它们可能会遇到预训练阶段未获取的新事实信息。人们通常推测这可能会教导模型...</li><li><a href="https://huggingface.co/spaces/discord-community/HuggingMod/discussions/1">discord-community/HuggingMod · 请合并</a>: 未找到描述</li><li><a href="https://docs.continue.dev/how-to-use-continue#ask-questions-about-your-codebase">🧑‍🎓 如何使用 Continue | Continue</a>: 在使用 Continue 编写代码时使用 LLM</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k">gradientai/Llama-3-8B-Instruct-262k · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/google/gemma-2b">google/gemma-2b · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/ageron">ageron - 概览</a>: 《Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow》一书的作者。前 YouTube 视频分类项目经理，Wifirst 创始人兼 CTO。 - ageron</li><li><a href="https://huggingface.co/chat/">HuggingChat</a>: 让每个人都能使用社区最优秀的 AI 聊天模型。</li><li><a href="https://lu.ma/4t48gcy0">Shibuya Startup Support xTechstars Startup Weekend Tokyo Weekly Snack &amp; Connect · Luma</a>: 在与 Techstars Startup Weekend Tokyo 度过了一个令人兴奋的周末后 😎🚀🎯，组织团队希望提供一份关于该初创企业的活动后简报……</li><li><a href="https://huggingface.co/blog/nroggendorff/finetune-tinyllama">使用 TRL 对 TinyLlama 进行文本生成微调</a>: 未找到描述</li><li><a href="https://mlflow.org/docs/latest/python_api/mlflow.metrics.html">mlflow.metrics — MLflow 2.14.2 文档</a>: 未找到描述</li><li><a href="https://github.com/buaacyw/MeshAnything">GitHub - buaacyw/MeshAnything: 像人类艺术家一样将万物转化为网格。 "MeshAnything: Artist-Created Mesh Generation with Autoregressive Transformers" 的官方实现</a>: 像人类艺术家一样将万物转化为网格。 "MeshAnything: Artist-Created Mesh Generation with Autoregressive Transformers" 的官方实现 - buaacyw/MeshAnything</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/commit/3c8db1bb7be5662e4fd5b48a26b6214f758e483f">添加 Open LLM Leaderboard 任务 (#2047) · EleutherAI/lm-evaluation-harness@3c8db1b</a>: * 添加排行榜任务
 
 * 删除 lm_eval/tasks/leaderboard/leaderboard_chat_template.yaml
 
 * 添加 readme
 
 * 删除 lm_eval/tasks/leaderboard/mmlu_pro/mmlu_pro_chat_template.yaml
 
 * 修改 ...</li><li><a href="https://github.com/huggingface/lighteval?tab=readme-ov-file">GitHub - huggingface/lighteval: LightEval 是一个轻量级的 LLM 评估套件，Hugging Face 内部一直在将其与最近发布的 LLM 数据处理库 datatrove 和 LLM 训练库 nanotron 配合使用。</a>: LightEval 是一个轻量级的 LLM 评估套件，Hugging Face 内部一直在将其与最近发布的 LLM 数据处理库 datatrove 和 LLM 训练库 nanotron 配合使用。 - hug...</li><li><a href="https://tenor.com/view/red-kit-gif-11737462">Red Kit GIF - Red Kit - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://discuss.huggingface.co/t/how-to-convert-ckpt-to-diffusers-format/35635">如何将 ckpt 转换为 diffusers 格式</a>: 社区正在大量使用 .ckpt 和 diffusers 格式。我们正在努力为这些格式之间的互操作性提供更好的支持，但推荐的方法始终是……</li><li><a href="https://tenor.com/view/ishowspeed-speed-shocked-shock-shocked-meme-gif-8910406893424234862">Ishowspeed Shocked GIF - Ishowspeed Speed Shocked - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cd4yim/llama38binstruct_with_a_262k_context_length/">Reddit - 深入探索万物</a>: 未找到描述
</li>
</ul>

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1259983252584796240)** (11 messages🔥): 

> - `具有历史人物角色的 Discord bot`
> - `Huggingface NLP 课程`
> - `VRAM 使用量波动`
> - `从 Checkpoint 恢复训练`
> - `Padding 与 VRAM 稳定化` 


- **构建具有历史人物角色的 Discord Bot**：一位成员正在学习如何制作一个包含多个著名历史人物角色/LLM 的 Discord bot。
- **推荐 Huggingface NLP 课程**：一位成员建议从 [Huggingface NLP 课程](https://huggingface.co/learn/nlp-course/chapter1/1) 开始学习。
- **LLM 训练期间的 VRAM 使用峰值**：训练 LLM 有时需要不同数量的 VRAM，且使用量可能会突然飙升。
   - 一位成员指出，变长 Batch、特定的 Trainer 优化或 Bug 可能会导致波动。
- **关于从 Checkpoint 恢复训练的疑虑**：一位用户询问从 Checkpoint 恢复训练是否会对最终结果产生不利影响。
- **通过 Padding 稳定 VRAM 使用**：成员们讨论了将所有样本 Padding 到预定义的 `max_seq_len` 以稳定 VRAM 使用，特别是当数据集的 99% 低于 100 tokens 但某些元素超过 512 时。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1260327903191891989)** (2 messages): 

> - `生成式 AI 对叙事的影响`
> - `AI 知识管理` 


- **生成式 AI 影响叙事**：一篇 Medium 文章的链接探讨了 [生成式 AI 对故事讲述和叙事创作的影响](https://medium.com/@shikharnautiyal29/the-impact-of-generative-ai-on-storytelling-and-narrative-creation-d37898cc0126)。
   - *生成式 AI 为故事讲述方式带来了深刻变革，* 新方法开启了丰富的叙事机会。
- **KMWorld 聚焦知识管理领域的 AI 领导者**：[Marydee Ojala](https://www.kmworld.com/Authors/7211-Marydee-Ojala.htm) 在 KMWorld 上讨论了 2024 年 AI 100 强企业，展示了那些处于 **知识管理** 进步前沿的公司。
   - 她的文章指出了 AI 技术进步的 **飞速步伐**，以及各领域对 **生成式 AI** 日益增长的兴趣。



**提到的链接**：<a href="https://www.kmworld.com/Articles/Editorial/Features/The-KMWorld-AI-100-The-Companies-Empowering-Intelligent-Knowledge-Management-164117.aspx">The KMWorld AI 100: The Companies Empowering Intelligent Knowledge Management</a>：面对每天向我们涌来的大量关于 AI（尤其是 GenAI）的信息，人们很容易感到不知所措，甚至心生敬畏。AI 技术处理海量信息的能力...

  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1260144915967381524)** (4 messages): 

> - `Intel HF models`
> - `Gemma2:27B update`
> - `New Qdurllm demo`
> - `Early Exit in LLM research` 


- **新仓库展示了 Intel HF 模型**：一名成员创建了一个 [GitHub repo](https://github.com/sleepingcat4/intel-hf)，演示了如何使用 **Intel CPUs** 高效运行 HF 模型。
   - 这解决了在 Intel 硬件上运行 HF 模型缺乏 **Intel 专用教程和文档** 的问题。
- **Gemma2:27B 获得重大更新**：**Gemma2:27B** 模型获得了更新，目前表现异常出色，正如 [YouTube 视频](https://youtu.be/38ae7hqzX5s) 中所强调的那样。
   - 该更新由 **Ollama** 推送以修正之前的问题，根据社区反馈，该模型现在的表现“令人难以置信”。
- **新的 Qdurllm 演示空间上线**：基于 Qdrant、Sentence Transformers、llama-cpp 和 Langchain 的本地搜索引擎 **Qdurllm** 的新演示空间现已在 [HuggingFace Spaces](https://huggingface.co/spaces/as-cle-bert/qdurllm-demo) 上线。
   - 鼓励社区尝试这个功能齐全的版本，并在 [GitHub](https://github.com/AstraBert/qdurllm) 上通过点亮 star 来支持它。
- **展示了 LLM 研究中的 Early Exit**：社区成员在一个新的 [HuggingFace Space](https://huggingface.co/spaces/valcore/Branchy-phi-2) 中展示了 **LLM 中的 Early Exit** 研究。
   - 该空间由于在 CPU 上运行而速度较慢，但它演示了如何使用 Early Exit 在某些 token 上实现更快的推理，并具有可配置的 Epsilon 设置以平衡速度和准确性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/valcore/Branchy-phi-2">Branchy Phi 2 - valcore 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://youtu.be/38ae7hqzX5s">Gemma2:27 Ollama Correction ! Now Incredible !</a>：今天，我们将再次使用 ollama 测试 gemma 2 27b，因为 ollama 推送了一个更新来修正与 gemma 2 相关的问题，现在它可以正常工作了...</li><li><a href="https://huggingface.co/spaces/as-cle-bert/qdurllm-demo">Qdurllm Demo - as-cle-bert 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://github.com/sleepingcat4/intel-hf">GitHub - sleepingcat4/intel-hf: 使用 Intel CPUs 和 Intel 架构推理 HF 模型</a>：使用 Intel CPUs 和 Intel 架构推理 HF 模型 - sleepingcat4/intel-hf
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1260043136181469245)** (1 messages): 

> - `Pentesting in AI`
> - `PentestGPT` 


- **即将进行的 AI 渗透测试文献综述**：计划在下周六讨论关于 **AI 渗透测试 (pentesting in AI)** 的文献综述，主要基于 **PentestGPT**。
   - 该综述还将涵盖目前改进 AI 渗透测试方法的努力，并引用了 [PentestGPT 论文](https://arxiv.org/abs/2308.06782)。
- **PentestGPT 成为未来讨论的核心**：PentestGPT 被强调为正在进行的关于改进 AI 渗透测试讨论中的重要资源。
   - *目前正努力通过借鉴 PentestGPT 的见解来增强渗透测试方法。*


  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1259987788724436992)** (14 条消息🔥): 

> - `YoloV1 的局限性`
> - `YoloV8 的重新实现`
> - `情绪与肢体语言相关的研究论文`
> - `使用微调模型进行推理`
> - `文档图像质量预测` 


- **YoloV1 遇到网格单元限制**：在成功训练 **YoloV1** 后，面临着每个网格单元只能生成一个边界框（bounding box）的重大限制。
   - 针对这一问题，团队开始调试 **YoloV8** 的代码库，试图重新实现一个解决方案。
- **确认使用微调模型进行推理**：一名成员确认他们正在使用微调后的模型进行推理（inference）。
- **视觉模型微调中的幽默疏忽**：一位用户幽默地承认，他们在微调过程中没有向模型输入图像，而是输入了文件路径。
   - 他们意识到错误后补充道：*“哈哈，这有点蠢”*。
- **寻找关于情绪和肢体语言的研究**：一名成员询问是否有将言语情绪与肢体语言或手势联系起来的研究论文。
- **文档图像质量预测的建议**：一名成员请求关于预测文档图像质量的建议，无论是通过回归值，还是针对清洁、空白、模糊和脏污文档进行分类。



**提到的链接**：<a href="https://github.com/ultralytics/ultralytics/issues/10392#issuecomment-2215366567">Serializing Classifier and Regressor heads in Yolo models · Issue #10392 · ultralytics/ultralytics</a>：提问前先搜索。我搜索了 YOLOv8 的 issue 和讨论，没有发现类似问题。问题：Hi team，希望你们一切都好。如果能分享一下你们的想法，那就太棒了...

  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1259958137360879778)** (1 条消息): 

> - `sd-vae 伪影问题`
> - `蓝色和白色像素` 


- **sd-vae 伪影问题咨询**：一位用户询问在使用 **sd-vae** 进行重建（reconstruction）时，观察到的伪影（特别是蓝色和白色像素）是否正常。
   - *在使用 sd-vae 进行重建时，这种类型的伪影正常吗？*
- **VAE 重建中的像素化疑问**：关于在使用 **sd-vae** 机制时出现**蓝色和白色像素化**现象是否正常的疑问。
   - 用户似乎对重建阶段像素伪影的具体细节感到担忧。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1259963994127597579)** (136 messages🔥🔥): 

> - `新文档网站`
> - `Kaggle 上的微调挑战`
> - `训练问题`
> - `模型使用请求`
> - `社区贡献` 


- **Unsloth AI 发布新文档网站**：[Unsloth](https://github.com/unslothai/unsloth) 发布了新的文档网站，该工具可使 **Llama-3**、**Mistral**、**Phi-3** 和 **Gemma** 等大语言模型的微调速度提升 2 倍，同时减少 70% 的内存占用，且精度无损。
   - 该网站帮助用户引导完成自定义模型的训练，涵盖了[创建数据集](/tutorials/how-to-finetune-llama-3-and-export-to-ollama#id-6.-alpaca-dataset)和[部署模型](/tutorials/how-to-finetune-llama-3-and-export-to-ollama#id-13.-exporting-to-ollama)等核心内容。
- **用户指出 PyPI 上的 gguf 库已过时**：一位用户注意到 PyPI 上托管的 gguf 库在手动保存为 GGUF 格式时版本过旧，建议改为从 **llama.cpp** 仓库构建最新的 Python gguf。
   - 建议的安装命令为 `cd llama.cpp/gguf-py && pip install .`，以确保使用最新版本。
- **在有限硬件上进行微调的挑战**：一位用户强调了在 **Magicoder-Evol-Instruct-110K** 数据集上使用 rsLoRA 微调 **unsloth/Qwen2-0.5B** 模型时遇到的问题，指出训练损失（loss）没有下降。
   - 通过更改训练参数（如增加学习率和 rank）以及选择更大的模型（如 1.5b），获得了更好的性能结果。
- **关于模型微调的混合反馈**：针对小模型在特定任务中是否有效展开了讨论，例如针对瑞典语数据集微调 **GPT-Sw3 1.3B** 模型，部分用户对其性能表示怀疑。
   - “使用 **Llama 3**，”一位成员强调，并指出除非受资源限制必须使用小模型，否则它是更优的选择。
- **社区积极寻求贡献与合作**：用户表现出对贡献 Unsloth 文档和改进模型的浓厚兴趣。
   - 首席开发者鼓励社区贡献，承诺会发布更新，并表示正在努力在月底前支持更多模型。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/AI-Sweden-Models/gpt-sw3-1.3b">AI-Sweden-Models/gpt-sw3-1.3b · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/bartowski/Lexi-Llama-3-8B-Uncensored-GGUF/blob/main/Lexi-Llama-3-8B-Uncensored-IQ4_XS.gguf">Lexi-Llama-3-8B-Uncensored-IQ4_XS.gguf · bartowski/Lexi-Llama-3-8B-Uncensored-GGUF at main</a>：未找到描述</li><li><a href="https://x.com/kaggle/status/1810776803449131024">Kaggle (@kaggle) 的推文</a>：📚 看看由 @UnslothAI 联合创始人 @danielhanchen 编写的精彩 Notebook！了解如何使用 Kaggle Notebooks 微调 Gemma-2-9b。了解更多：👇https://www.kaggle.com/code/danielha...</li><li><a href="https://docs.unsloth.ai/">欢迎 | Unsloth 文档</a>：Unsloth 新手？从这里开始！</li><li><a href="https://docs.unsloth.ai/basics/finetuning-fro">Unsloth 文档</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/basics/finetuning-from-last-checkpoint),">Unsloth 文档</a>：未找到描述</li><li><a href="https://tenor.com/view/american-psycho-patrick-bateman-american-psycho-gif-7212093">American Psycho Patrick Bateman GIF - American Psycho Patrick Bateman American - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama">如何微调 Llama-3 并导出至 Ollama | Unsloth 文档</a>：为在 Ollama 上本地运行自定义 ChatGPT 的初学者指南
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1259990088843268097)** (24 条消息🔥): 

> - `Unscoming Unsloth Vision Model Support`
> - `Medical Data Translation with Llama 3`
> - `Llama 3 and Swedish`
> - `Training Llama 3 on Medical Data`
> - `Using Pre-trained Llama 3 Models on Unsloth` 


- **使用 Llama 3 进行医学数据翻译**：一位用户分享了一个项目想法，即使用 **Llama 3** 将 5000 行医学数据翻译成瑞典语，然后使用这些翻译后的数据对模型进行 Fine-tune。
   - 另一位用户建议，这种方法比在使用过程中依赖 LLM 自动翻译信息更有利。
- **Llama 3 与瑞典语兼容性**：一位用户确认 **Llama 3** 精通瑞典语，非常适合他们的翻译项目。
   - 他们还获知了通过 Unsloth 针对瑞典语特定需求对 **Llama 3** 进行 Fine-tune 的可用资源。
- **在医学数据上训练 Llama 3**：讨论了通过使用已经经过 Fine-tune 的模型（如 [AI-Sweden-Models/Llama-3-8B-instruct](https://huggingface.co/AI-Sweden-Models/Llama-3-8B-instruct)）来跳过 Continued Pre-training 步骤。
   - 一位成员建议从 Base Model 开始进行 Fine-tune，并使用 Instruction-based 模型进行翻译任务。
- **在 Unsloth 上使用预训练的 Llama 3 模型**：用户讨论了在 Unsloth 上使用预训练 Llama 模型的可行性，确认通过将 `AI-Sweden-Models/Llama-3-8B-instruct` 设置为模型名称是可行的。
   - 注意到使用 Base Model 进行训练通常会产生更好的结果，而 Instruction 模型更适合翻译等特定任务。



**提到的链接**：<a href="https://huggingface.co/AI-Sweden-Models/Llama-3-8B-instruct">AI-Sweden-Models/Llama-3-8B-instruct · Hugging Face</a>：未找到描述

  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1259957459515342939)** (43 条消息🔥): 

> - `结合微调模型的 RAG`
> - `使用 RAFT 获得更好的回复`
> - `从 PDF 创建合成数据集`
> - `加速推理`
> - `训练方法和 completion-only 微调` 


- **用户讨论将微调模型与 RAG 集成**：成员们讨论了将微调模型与 RAG 方法结合使用，并分享了关于 Alpaca 中所见的上下文感知微调的见解。
   - 他们建议研究 [RAFT](https://arxiv.org/abs/2403.10131) 以集成新知识并有效处理干扰文档。
- **使用工具从 PDF 生成数据集**：一位用户询问关于从 PDF 生成合成数据集的建议，推荐使用 [nougat](https://github.com/facebookresearch/nougat) 或 [marker](https://github.com/VikParuchuri/marker) 等工具进行转换。
   - 这些工具可以高精度地简化 PDF 到 Markdown 的转换流程，显著减少人工投入。
- **加速微调和推理**：成员们分享了加速 phi-3 mini 等模型微调的技术，包括使用 VLLM 进行推理。
   - 据建议，至少需要 300 个样本来微调基础模型，才能在新领域获得合理的结果。
- **微调中的训练损失不一致**：一位用户报告在使用 rsLoRA 微调 `unsloth/Qwen2-0.5B-Instruct-bnb-4bit` 时，训练损失（loss）无法下降。
   - 在另一个数据集上使用相同模型获得了成功，这表明可能存在特定于数据集的问题。
- **微调方法和 completion-only 训练**：讨论探讨了微调是否应同时训练指令和回复，建议使用 `DataCollatorForCompletionOnlyLM` 进行仅针对回复的预测训练。
   - 这种方法通过专注于预测答案 token 而非指令，有可能提高训练效率。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2403.10131">RAFT: Adapting Language Model to Domain Specific RAG</a>：在大规模文本语料库上预训练大型语言模型 (LLMs) 已成为标准范式。在将这些 LLMs 用于许多下游应用时，通常需要额外加入新知识...</li><li><a href="https://huggingface.co/bartowski/Lexi-Llama-3-8B-Uncensored-GGUF/blob/main/Lexi-Llama-3-8B-Uncensored-IQ4_XS.gguf">Lexi-Llama-3-8B-Uncensored-IQ4_XS.gguf · bartowski/Lexi-Llama-3-8B-Uncensored-GGUF at main</a>：未找到描述</li><li><a href="https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored">Orenguteng/Llama-3-8B-Lexi-Uncensored · Hugging Face</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2310.01208">Label Supervised LLaMA Finetuning</a>：大型语言模型 (LLMs) 最近的成功引起了学术界和工业界的广泛关注。人们在增强零样本和少样本泛化能力方面做出了巨大努力...</li><li><a href="https://huggingface.co/docs/trl/en/sft_trainer#train-on-completions-only">Supervised Fine-tuning Trainer</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Docs</a>：查看下方列表获取我们所有的 notebook：</li><li><a href="https://github.com/VikParuchuri/marker">GitHub - VikParuchuri/marker: Convert PDF to markdown quickly with high accuracy</a>：高精度快速将 PDF 转换为 Markdown - VikParuchuri/marker
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1260112523999314010)** (40 条消息🔥): 

> - `训练自定义 Embeddings`
> - `LLaMA3 的显存问题`
> - `EfficientPartialEmbedding 实现`
> - `Modular Model Spec`
> - `SiteForge 网页设计生成` 


- **训练自定义 Embeddings 的困扰**：Albert_lum 尝试在 LLaMA 3 7B 上为新的特殊 Token 训练 Embeddings，但在 Colab T4 上面临显存挑战，且在仅微调特定 Embeddings 时遇到困难。
- **Embedding 矩阵显存挑战**：Timotheeee1 指出 LLaMA 3 的 Head 和 Embedding 矩阵消耗大量 VRAM，这在尝试训练特定片段时造成了障碍。
- **EfficientPartialEmbedding 实现问题**：Albert_lum 讨论了各种尝试和解决方案（例如包装原始 Embedding），但在效率和确保 Embedding 正确训练方面仍有困难。
- **Modular Model Spec 开发**：Albert_lum 提到正在为 AI 模型开发一种新的行为规范，旨在提高灵活性、可靠性和开发者便利性。
   - 该规范详见 [Modular Model Spec](https://modular-model-spec.vercel.app)，旨在帮助开发者和策展人创建高级的 LLM 增强型应用。
- **用于网页设计的 SiteForge AI**：Albert_lum 正在为 SiteForge 微调 LLaMA 3，该公司专注于 AI 生成的网页设计。
   - SiteForge 提供 AI 站点地图生成器和拖放式网站重构等功能，详见其[官网](https://siteforge.io)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pastebin.com/GgeVQLZK"># %%filename = model_name.split(&quot;/&quot;)[1] + &quot;_tokens.pt&quot;if os.path.exists(file - Pastebin.com</a>: Pastebin.com 是自 2002 年以来排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://modular-model-spec.vercel.app">Modular Model Spec</a>: 未找到描述</li><li><a href="https://siteforge.io">AI Wireframe Generator » SiteForge</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1259954640917893342)** (9 messages🔥): 

> - `LLM 中的无矩阵乘法（MatMul-free）模型`
> - `测试时训练（Test-Time-Training）层`
> - `聊天机器人的合成数据集`
> - `通过 Orca 增强的模仿学习`
> - `Flash Attention 中的 Soft Capping` 


- **无 MatMul 模型革新 LLM 性能**：从大语言模型中[消除矩阵乘法](https://arxiv.org/abs/2406.02528)在十亿参数规模下仍能保持强劲性能，显著降低内存使用，实验显示相比未优化的基准线，内存占用**减少了高达 61%**。
- **测试时训练（Test-Time-Training）层提供新方法**：一种名为 **Test-Time-Training layers** 的新架构用机器学习模型取代了 RNN 隐藏状态，实现了线性复杂度，并达到或超越了顶尖的 Transformer，正如[最近的一条推文](https://x.com/karansdalal/status/1810338845659131940)所宣布的那样。
- **聊天机器人的高质量合成数据集**：关于为聊天机器人生成合成数据集的[研究](https://arxiv.org/abs/2308.05884)表明，设置原理（rationale）、上下文和角色（persona）可以产生高质量的对话，为 PIPPA 数据集贡献了来自 **26,000 个会话**的超过 **100 万条话语**。
- **Orca 增强小模型模仿学习**：**Orca 模型**通过从 GPT-4 等大型基础模型中学习复杂的推理过程，解决了模仿学习中的挑战，详见[论文](https://arxiv.org/abs/2306.02707)。
   - Orca 利用丰富的信号（如解释轨迹和逐步思考过程）显著提升了较小模型的能力。
- **Flash Attention 采用 Soft Capping 以获得卓越性能**：根据[最近的公告](https://x.com/_philschmid/status/1810733822100779487)，**FlashAttention** 现在支持 Soft Capping，从而增强了快速且准确的 **Google DeepMind Gemma2** 生成。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/_philschmid/status/1810733822100779487">Philipp Schmid (@_philschmid) 的推文</a>: Flash Attention 现在支持 Soft capping 了！🚀 为快速且准确的 @GoogleDeepMind Gemma2 生成做好准备。🏎️💥💨 感谢 @narsilou 和 @tri_dao ❤️</li><li><a href="https://x.com/karansdalal/status/1810338845659131940">Karan Dalal (@karansdalal) 的推文</a>: 我很高兴能分享一个我已经研究了一年多的项目，我相信它将从根本上改变我们处理语言模型的方式。我们设计了一种新架构，它取代了 h...</li><li><a href="https://arxiv.org/abs/2308.05884">PIPPA: 一个部分合成的对话数据集</a>: 随着日益强大的大语言模型的出现，利用这些模型进行闲聊和角色扮演应用的研究兴趣日益浓厚。然而，现有的对话...</li><li><a href="https://arxiv.org/abs/2306.02707">Orca: 从 GPT-4 的复杂解释轨迹中进行渐进式学习</a>: 最近的研究重点是通过模仿学习，利用大型基础模型 (LFMs) 生成的输出来增强较小模型的能力。许多问题影响了...</li><li><a href="https://arxiv.org/abs/2406.02528">可扩展的无 MatMul 语言建模</a>: 矩阵乘法 (MatMul) 通常在大语言模型 (LLMs) 的整体计算成本中占据主导地位。随着 LLMs 扩展到更大的嵌入维度和上下文长度，这一成本只会增加...
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1260213340865560688)** (2 messages): 

> - `将 Triton Kernel 与 PyTorch 集成`
> - `在 PyTorch 中注册自定义函数`
> - `torch.compile 与自定义函数`
> - `CUDA Kernel 集成` 


- **将 Triton Kernel 集成到 PyTorch 模型中**：一位用户询问了集成 **Triton kernel** 以替换 **PyTorch 模型**中某个函数的最佳方法。
   - 他们想知道是否可以**在 PyTorch 中注册此函数**，以便在运行 **torch.compile** 时，只要检测到该模式就会自动使用此函数。（目前尚无直接回答或进一步讨论。）
- **在 PyTorch 中注册自定义函数**：该用户正在寻找一种在 PyTorch 中**注册自定义 Triton 函数**的方法，以便通过 **torch.compile** 实现自动调用。


  

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1260279615029772411)** (1 messages): 

> - `executorch`
> - `vulkan backend` 


- **关于 executorch 在 Vulkan 中使用 textures 的查询**：一位成员询问为什么 executorch 的 **Vulkan backend** 中的算子（operators）使用 textures。
- **关于 executorch 和 Vulkan 算子的讨论**：成员们讨论了 executorch 中 **Vulkan backend** 对 textures 的使用，寻求该实现背后的背景和原因。


  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1260070461899935754)** (1 messages): 

> - `FlashInfer`
> - `Kernel Library for LLM Serving`
> - `INT8 and FP8 flash attention kernels` 


- **FlashInfer：用于 LLM Serving 的新算子库**：FlashInfer: Kernel Library for LLM Serving 通过 [GitHub 链接](https://github.com/flashinfer-ai/flashinfer) 分享，供社区评审。
   - 该库支持 **INT8** 和 **FP8** flash attention 算子，承诺提升性能。
- **FlashInfer 支持 INT8 和 FP8 算子**：最近发布的 FlashInfer 库包含 **INT8** 和 **FP8** flash attention 算子。
   - 此特性可能会大大增强大语言模型（LLM）的服务效率。



**提到的链接**：<a href="https://github.com/flashinfer-ai/flashinfer">GitHub - flashinfer-ai/flashinfer: FlashInfer: Kernel Library for LLM Serving</a>：FlashInfer：用于 LLM Serving 的算子库。通过创建账号为 flashinfer-ai/flashinfer 的开发做出贡献。

  

---


### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1260023773005090886)** (3 messages): 

> - `Job application enthusiasm`
> - `Team commendation`
> - `Positive reactions` 


- **职位申请热情飙升**：一位成员表达了他们的兴奋：*"我从未这么快点击过申请"*。
- **团队获得高度赞扬**：另一位成员为该团队担保，称其为一个 *"伟大的团队"*。
- **压倒性的积极情绪**：第三位成员分享了一个温暖的反应 *🥰*。


  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1259964129394036807)** (24 messages🔥): 

> - `Beginner CUDA Projects`
> - `Flash Attention`
> - `Benchmarking Techniques`
> - `Triton for Softmax`
> - `Tensor Offloading` 


- **初学者 CUDA 项目：Flash Attention 难度过高**：一位用户考虑实现 flash attention，但社区成员建议从更简单的项目开始，如普通的 attention 或简单的 MLP，并指出 *flash attention 和初学者是矛盾修辞法（oxymorons）*。
- **Flash Attention 与 PyTorch 的基准测试**：社区讨论了将自定义实现的 attention 机制与 PyTorch 的 **flash attention** 进行基准测试对比的可行性。
   - 有建议指出，虽然 PyTorch 的 flash attention 并非极快，但可以先从常规 attention 开始，然后再转向 flash attention 进行基准测试。
- **Attention 机制中的 Softmax 挑战**：Softmax 被认为是实现 attention 机制中最具挑战性的部分，建议先编写一个不带 softmax 的 attention 版本以确保乘法正确，然后再处理它。
   - 提到的另一种方法是在 Triton 中编写一个带有三个循环的简单 softmax，以生成并理解 PTX 代码。
- **NVIDIA 的 Tensor Offloading 概念**：一位用户询问了 [NVIDIA 白皮书](https://www.amax.com/content/files/2023/12/NVIDIA_Grace_Hopper_Superchip_Architecture_Overview_Whitepaper.pdf) 中提到的 “tensor offloading” 概念。
   - 另一位成员将其解释为使用来自 host 或另一个 GPU 的交换内存（swap memory）进行张量操作。


  

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1260024710129778769)** (1 messages): 

> - `使用静态量化的量化流程示例`
> - `数据校准的重要性` 


- **使用静态量化的量化流程示例**：一位用户分享了一个 [GitHub Pull Request](https://github.com/pytorch/ao/pull/487)，展示了一个需要数据校准的静态量化示例。
   - 此 PR 通过添加静态量化的实现，解决了当前 API 不需要使用样本数据进行模型校准的问题。
- **数据校准的重要性**：讨论的量化流程示例强调了在实现静态量化时进行校准的必要性。
   - 校准通过使用样本数据来优化模型性能，确保模型表现准确。



**提到的链接**：<a href="https://github.com/pytorch/ao/pull/487">由 jerryzh168 提交的“添加静态量化作为校准流程示例” · Pull Request #487 · pytorch/ao</a>：摘要：到目前为止，我们提供的量化流程 API (`quantize_`) 不需要校准（使用样本数据校准模型），此 PR 添加了一个静态量化示例，作为...

  

---


### **CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1260069994302013512)** (8 messages🔥): 

> - `Ring Attention`
> - `跨 GPU 切分 KV Cache`
> - `AWS g5.12xlarge 实例` 


- **探索使用 Ring Attention 切分 KV Cache**：一位成员表示有兴趣使用 **Ring Attention** 将其模型的 **KV Cache** 跨 GPU 切分，特别是在包含四个 A10G 的 **AWS g5.12xlarge** 实例中。
   - 另一位成员建议使用 `[nvidia-smi topo -m](https://gist.github.com/msaroufim/abbb01c5bb037c2b1009df4a0baeb74b)` 打印 GPU 拓扑，并分享了一个用于估算 **KV Cache** 和模型大小的 [脚本](https://gist.github.com/msaroufim/abbb01c5bb037c2b1009df4a0baeb74b)。
- **确认 GPU 环形拓扑**：询问者提到 **论文** 建议对 GPU 使用环形拓扑，但找不到关于所讨论的 AWS 实例的具体信息。



**提到的链接**：<a href="https://gist.github.com/msaroufim/abbb01c5bb037c2b1009df4a0baeb74b">kv-calc.py</a>：GitHub Gist：即时分享代码、笔记和代码片段。

  

---


### **CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1259988140320489653)** (1 messages): 

> - `Puzzle 9 解释`
> - `题目陈述困惑` 


- **Puzzle 9 指令不清晰**：一位成员正在寻求对 **Puzzle 9** 更清晰的解释，参考了 [此 Discord 链接](https://discord.com/channels/1189498204333543425/1222666751272161430) 中的讨论。
   - 他们指出讨论中的 **共识** 与 **题目陈述相矛盾**，并质疑选择任意的 **B1** 是否对问题考虑过度。
- **Puzzle 9 题目陈述困惑**：一位成员对 **Puzzle 9** 的题目陈述提出了担忧，表示这给参与者带来了困惑。
   - 讨论围绕着选择 **任意 B1** 是否符合题目条件，还是属于过度复杂化。


  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1259949042998181890)** (176 条消息🔥🔥): 

> - `Llama 模型框架改进`
> - `llm.cpp 更新`
> - `零初始化（Zero initialization）对 NLP 模型的影响`
> - `MuP 库集成`
> - `快速推理策略` 


- **Llama 模型支持被视为至关重要**：一名成员询问为何应优先添加 **Llama 支持**，引发了关于其有效性以及相比其他模型优势的讨论。
   - 一位成员提到了一个支持 Vulkan 和 AMD 等多种后端的 [Llama 相关仓库](https://github.com/ggerganov/llama.cpp)，另一位成员则推荐了一个 [CUDA 特定实现](https://github.com/karpathy/llama2.cu)。
- **llm.c 中用于加速推理的 PR**：一名成员创建了一个 [Pull Request](https://github.com/karpathy/llm.c/pull/671)，通过将内存处理方式从 (B,T) 更改为 (1,t) 来优化 **推理速度**，旨在获得位对位（bit-for-bit）完全一致的结果。
   - 初步测试显示训练 Loss 一致，但采样输出有所不同，这引发了关于 **cuDNN heuristics**（启发式算法）及其变异性的疑问。
- **零初始化可能损害 Embedding 层**：讨论了在不同设置下 **零初始化** 对 Embedding 层的影响，结论是使用零初始化可能会损害性能。
   - 成员们计划进行多次不使用零初始化的实验，以观察 Loss 值是否有所改善。
- **MuP 库的集成**：成员们讨论了集成 **MuP 库** 以确保训练结果的一致性，并注意到 MuP 自身仓库在 Embedding 层和输出层实现上的差异。
   - 他们决定进行一系列受控实验，以更好地理解初始化及其他因素对模型性能的影响。
- **400B Token 模型评估**：一个在 400B Token 上训练的模型达到了 **59.0 的 HellaSwag 分数**，表明其相比早期版本的 GPT 模型有显著改进。
   - 尽管结果令人印象深刻，但关于潜在优化的讨论仍在继续，包括使用 **truncated normal initialization**（截断正态初始化）和 **linear biases**（线性偏置）。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/microsoft/mutransformers/issues">Issues · microsoft/mutransformers</a>：一些使用最大更新参数化 (µP) 的常见 Huggingface Transformers - Issues · microsoft/mutransformers</li><li><a href="https://github.com/karpathy/llm.c/pull/671">通过将 (B,T) 更改为 (1,t) 实现更快的推理 by ademeure · Pull Request #671 · karpathy/llm.c</a>：目前的推理完整性检查会处理所有的 (B,T)，尽管默认只需要 (1,64)。此 PR 与之前版本位对位一致，同时将其减少到 (1,t)，其中 t 是取整后的...</li><li><a href="https://github.com/karpathy/llama2.c">GitHub - karpathy/llama2.c: 在单个纯 C 文件中推理 Llama 2</a>：在单个纯 C 文件中推理 Llama 2。通过创建账号为 karpathy/llama2.c 的开发做出贡献。</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: C/C++ 中的 LLM 推理</a>：C/C++ 中的 LLM 推理。通过创建账号为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://x.com/main_horse/status/1810647037718999342">来自 main (@main_horse) 的推文</a>：跨参数化和优化器的缩放指数 [GDM] [nocode/weights] https://arxiv.org/abs/2407.05872 训练了 10,000+ (!) 个模型，涵盖不同的 * 优化器 (SGD/Adam/Adafactor) * 模型大小 (1.1B ~...</li><li><a href="https://github.com/ankan-ban/llama2.cu/blob/master/llama2.cu">llama2.cu/llama2.cu at master · ankan-ban/llama2.cu</a>：在单个纯 CUDA 文件中推理 Llama 2。通过创建账号为 ankan-ban/llama2.cu 的开发做出贡献。</li><li><a href="https://github.com/microsoft/mup/issues/7#issuecomment-1082141121">MuAdam 未调整输出权重的学习率 · Issue #7 · microsoft/mup</a>：你好，感谢你们出色的超参数调优项目！当我们团队将 MuP 迁移到其他训练框架时，我们发现 MuAdam 没有缩放输出层的学习率...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1260317716737298462)** (1 messages): 

> - `Error PDF 讨论`
> - `Gary Marcus 和 Yann LeCun 的 GIF` 


- **由 Gary Marcus 和 Yann LeCun 的 GIF 引发的讨论**：一位用户分享了来自 [Tenor](https://tenor.com/view/gary-marcus-yann-lecun-lecun-ai-machine-learning-gif-9041590723446061255) 的 Gary Marcus 和 Yann LeCun 的 GIF。
   - 该 GIF 幽默地捕捉了这两位 AI 领域知名人物辩论中的一个瞬间。
- **Error PDF 链接**：分享了一个与 PDF 相关的错误消息，引发了参与者的一些困惑。
   - 简要讨论了对该错误进行解释的简洁解决方案的需求。



**提到的链接**：<a href="https://tenor.com/view/gary-marcus-yann-lecun-lecun-ai-machine-learning-gif-9041590723446061255">Gary Marcus Yann Lecun GIF - Gary Marcus Yann LeCun Lecun - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

metaldragon01: https://x.com/stefan_fee/status/1810695036432232576
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1259959183214907412)** (117 messages🔥🔥): 

> - `AI 对就业的影响`
> - `Hermes 2 Pro`
> - `LLMs 越狱`
> - `Worldsim 控制台`
> - `Sonnet 模型能力` 


- **AI 准备改变就业格局**：成员们讨论了 AI，特别是大语言模型 (LLMs)，如何改变并可能消除工作岗位，并举例说明非专业人士现在更容易完成创意任务。
   - 一些人表达了 AI 工具如何大幅加速流程，并使原本无法实现的项目成为可能。
- **Hermes 2 Pro 以新特性令人惊叹**：[Hermes 2 Pro](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B) 因其改进的 Function Calling、JSON Structured Outputs 以及在各项评估中的高分而受到称赞。
   - 该模型可在 Hugging Face 上获取，展示了相较于前代产品的显著改进，因其鲁棒性获得了社区的认可。
- **模型越狱仍是一项挑战**：用户分享了在 Claude 3.5 和 WizardLM2 等模型上进行越狱的经验和技巧，发现尽管做出了努力，限制依然存在。
   - 运行自己的 LLM 被认为是一种昂贵但有效的绕过限制的方法，尽管仍可能面临审核障碍。
- **用于娱乐和创意的 Worldsim 控制台**：Worldsim 控制台用于娱乐和创意项目，模拟一个由 LLM 执行命令的终端。
   - 虽然它提供了引人入胜的体验，但由于 Opus 的高计算成本，用户需要留意其有限的免费额度。
- **Sonnet 模型展示了令人印象深刻的能力**：用户对 Sonnet 处理复杂任务的能力感到惊叹，例如生成嵌入在 JavaScript 中的 base64 图像。
   - 在 Gemini 1.5 Flash 等模型上进行的简单越狱实验也产生了令人惊讶的结果，展示了无审查输出的潜力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard">UGI Leaderboard - a Hugging Face Space by DontPlanToEnd</a>：未找到描述</li><li><a href="https://huggingface.co/MaziyarPanahi/WizardLM-2-8x22B-GGUF">MaziyarPanahi/WizardLM-2-8x22B-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B">NousResearch/Hermes-2-Pro-Llama-3-8B · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1259990520889868428)** (8 messages🔥): 

> - `LLMs 与分类`
> - `BAML 使用`
> - `合成数据生成工具`
> - `使用 Sonnet 3.5 API 处理 PDF`
> - `使用 weight keys 进行微调` 


- **大型 LLM 在分类和反馈方面表现出色**：讨论强调了**大型 LLM** 在**分类和反馈**任务中正变得异常卓越，为这类用例提供了**强大的解决方案**。
   - *L* 指出，这些能力使得大多数其他处理过程可以依赖标准代码，从而有效提高了生产力。
- **BAML 提供改进的开发体验 (UX)**：**BAML** 因其**开发用户体验**而受到赞誉，特别是它能直接在 IDE 中为 LLM 函数提供**类型提示 (type hints)**。
   - *Deoxykev* 提到，这一特性使得使用 LLM 进行开发更加高效且直观。
- **合成数据生成工具**：**Distilabel** 被推荐作为**合成数据生成**工具，强调了其高质量的输出和高效的 AI 反馈机制。
   - *Remek1972* 分享了该框架的 [链接](https://argilla-io.github.io/distilabel/1.2.1/)，强调了它对 AI 工程师的实用性。
- **使用 Sonnet 3.5 API 处理 PDF**：社区成员讨论了目前缺乏使用 **Sonnet 3.5 API** 处理 PDF 的**开箱即用解决方案**。
   - *Everyoneisgross* 建议使用 [Marker 库](https://github.com/VikParuchuri/marker) 将 PDF 转换为 Markdown，以获得更好的兼容性。
- **OpenAI 中的 weight keys 微调**：OpenAI 文档描述了使用 **weight key** 来确定在微调期间哪些消息具有优先级。
   - *Everyoneisgross* 解释说，这个特性告诉训练器忽略某些字段，从而优化训练过程。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://argilla-io.github.io/distilabel/1.2.1/">Distilabel</a>：Distilabel 是一个用于为 LLM 构建数据集的 AI Feedback (AIF) 框架。</li><li><a href="https://github.com/VikParuchuri/marker">GitHub - VikParuchuri/marker: 快速且高精度地将 PDF 转换为 markdown</a>：快速且高精度地将 PDF 转换为 markdown - VikParuchuri/marker
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1260130325086863485)** (88 messages🔥🔥): 

> - `RankRAG`
> - `Zero-Shot Prompting`
> - `RAG 中的 Function Calling`
> - `结构化 Scratch Pad`
> - `Llama3-RankRAG` 


- **RankRAG 优于现有的 RAG 方法**：讨论围绕 [RankRAG 方法](https://x.com/rohanpaul_ai/status/1810329112558371089) 展开，该方法通过对 LLM 进行指令微调，使其同时胜任 RAG 中的排序和生成任务，性能显著优于现有方法。
   - "Llama3-RankRAG-8B 和 Llama3-RankRAG-70B 以较大优势超越了同类模型，" 突显了其在新领域泛化方面的卓越能力。
- **结合检索逻辑的 Zero-Shot Prompting**："everyoneisgross" 分享了一个专注于 Zero-Shot Prompting 的小型实现，指出减少额外的 LLM 生成步骤对于知识摄取和 RAG 的效率至关重要。
   - "interstellarninja" 建议在数据合成过程中使用多个 Agent 进行重排序 (reranking) 和生成，以提高效率。
- **RAG 中 Function Calling 的提案**："interstellarninja" 详细介绍了一个提议的 `<scratch_pad>` 模板，用于 RAG 架构，该模板将结构化 AI 的推理过程、引用来源以及对检索文档相关性的反思。
   - 这种结构化模板旨在通过将行动组织为目标 (Goals)、行动 (Actions)、观察 (Observations) 和反思 (Reflections)，来提高模型的确定性输出。
- **Llama3-RankRAG 的实际应用**：讨论集中在 Llama3-RankRAG 的实际使用上，"interstellarninja" 建议使用 scratch_pad 结构来促进引用和相关性评分，从而获得更有依据的回答。
   - 参与者强调了标准化模板对于增强 RAG 输出的功能性和可靠性的必要性。
- **全球经济趋势为财务策略提供参考**：一个演示展示了经济趋势和投资机会（尽管不直接涉及个人财务）如何为财务策略提供信息。
   - 将有限的文档信息与既定的财务原则相结合，可以提供一个综合考虑个人行动和更广泛经济背景的全面答案。



**提及的链接**：<a href="https://x.com/rohanpaul_ai/status/1810329112558371089">来自 Rohan Paul (@rohanpaul_ai) 的推文</a>：来自 @nvidia 模型的 RAG 领域惊人成果 👏。来自 @nvidia 的 Llama3-RankRAG 在 9 个知识密集型基准测试中显著优于 GPT-4 模型。🤯 表现与 GPT-4 相当...

  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1260040929998405703)** (5 条消息): 

> - `Primeagen 采访 Chris Lattner`
> - `Mojo 书籍`
> - `使用 Mojo 进行 AI 开发的社区资源`
> - `Qualcomm SNPE 与 Mojo` 


- **Primeagen 直播采访 Chris Lattner**：Primeagen 宣布他将在未来的某个日期在 [Twitch](https://www.twitch.tv/theprimeagen) 上直播采访 **Chris Lattner**。
- **类似于 Rust book 的潜在 Mojo 书籍**：一位成员询问是否会有类似于 Rust book 的 Mojo 书籍，jack.clayton 给予了肯定的回答。
- **询问使用 Mojo 进行 AI 开发的社区资源**：一位成员询问了关于使用 Mojo 编写 AI 的社区资源，但回复中未提到具体资源。
- **将 Snapdragon 的 SNPE 与 Mojo 功能进行比较**：一位成员询问 Mojo 是否具有类似于 Qualcomm **SNPE** 的功能，用于确定在 Snapdragon 的哪个位置运行 PyTorch 模型。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.twitch.tv/theprimeagen">ThePrimeagen - Twitch</a>：CEO @ TheStartup™ (数十亿规模)，困在 Vim 中并希望它是 Emacs</li><li><a href="https://ruhati.net/mojo/">Mojo By Example: A Comprehensive Introduction to the Mojo Programming Language</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 条消息): 

ModularBot: 来自 *Modular*:
<https://twitter.com/Modular/status/1810782477079957831>
  

---


### **Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1260321991131660450)** (3 条消息): 

> - `通过 Modular 引入你自己的 PyTorch 模型`
> - `本地开发，全球部署`
> - `掌控 AI` 


- **通过 Modular 引入你自己的 PyTorch 模型**：Modular 讨论了企业中 AI 的兴起以及管理和部署 PyTorch 模型的需求，强调了在全规模生产期间对 AI 基础设施进行控制的必要性。[阅读更多](https://www.modular.com/blog/bring-your-own-pytorch-model)。
   - *PyTorch 的灵活性*在研究环境中表现出色，但在大规模生产部署中由于资源管理和延迟问题带来了*挑战*。
- **通过 Modular 实现本地开发，全球部署**：Modular 强调了创建可扩展 AI 开发工作流的挑战，并指出了 AI 工具链碎片化的现状。[探索更多](https://www.modular.com/blog/develop-locally-deploy-globally)。
   - AI 开发者通常需要在其工作流中使用多种工具，这使得从本地开发到云端部署的流程简化变得复杂。
- **掌控 AI**：Modular 概述了企业采用 AI 以提高生产力和客户体验的重要性，并引用了 Bain & Company 的研究，该研究表明 AI 的开发和部署率很高。[了解更多](https://www.modular.com/blog/take-control-of-your-ai)。
   - 显著比例的 **87% 的公司** 正在开发或部署生成式 AI，常见应用包括软件开发、客户服务、营销和产品差异化。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.modular.com/blog/take-control-of-your-ai">Modular: Take control of your AI</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：Take control of your AI</li><li><a href="https://www.modular.com/blog/develop-locally-deploy-globally">Modular: Develop locally, deploy globally</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：Develop locally, deploy globally</li><li><a href="https://www.modular.com/blog/bring-your-own-pytorch-model">Modular: Bring your own PyTorch model</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：Bring your own PyTorch model
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[tech-news](https://discord.com/channels/1087530497313357884/1151359062789857280/)** (1 条消息): 

helehex: 我听说 Lattner 先生明天和 Primeagen 有一些特别的安排
  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1259972422585684130)** (129 条消息🔥🔥): 

> - `代码优化讨论`
> - `Mojo 与 Python 集成`
> - `Mojo 语言特性`
> - `Mojo 文档与资源`
> - `引用语义与值语义的辩论`

- **Mojo 日志项目 Stump 引起关注**：成员们讨论了优化 Mojo 项目的话题，特别提到了 [GitHub 上的 stump 日志项目](https://github.com/thatstoasty/stump)，激发了进一步贡献的兴趣。
   - 成员们分享了优化和调试技巧，其中一位成员提到：*“花几个小时改一行代码，只为提升 1% 的速度并牺牲可读性，这就是我的热情所在。”*
- **Mojo 集成 Python 模块**：一位成员强调了将 Python 模块导入 Mojo 的能力，并引用了 [Mojo 文档](https://docs.modular.com/mojo/manual/python/) 中的支持说明。
   - 讨论涉及了使 Mojo 成为 Python 超集的长期目标，以利用 Python 庞大的软件包生态系统。
- **Mojo 语言语法与易用性**：关于在 Mojo 中调用可能出错（fallible）的函数和捕获错误的语法出现了疑问，并展示了处理引用和值的示例。
   - 成员们希望在 Mojo 中能有更直观的列表（lists）和切片（slices）处理方式，并参考了有用的 [span 文档](https://github.com/modularml/mojo/blob/nightly/stdlib/src/utils/span.mojo)。
- **Mojo 标准库开源**：讨论了 Mojo 的开源状态，并引用了[宣布核心模块开源的博客文章](https://www.modular.com/blog/the-next-big-step-in-mojo-open-source)。
   - *“Mojo 库现在开源了，”* 一位成员兴奋地表示，并强调了协作开发的重要性。
- **Mojo 中的值语义与引用语义**：针对 Mojo 中的值语义和引用语义展开了辩论，并解释了默认行为及其旨在提高易用性的设计。
   - Mojo 语义的灵活性，结合 [Mojo 值语义文档](https://docs.modular.com/mojo/manual/values/value-semantics) 等资源，被强调为对新老用户都至关重要。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/values/ownership#argument-conventions">Ownership and borrowing | Modular Docs</a>：Mojo 如何通过函数参数共享引用。</li><li><a href="https://docs.modular.com/mojo/manual/values/value-semantics">Value semantics | Modular Docs</a>：关于 Mojo 默认值语义的解释。</li><li><a href="https://stackoverflow.com/questions/70368651/why-cant-linux-write-more-than-2147479552-bytes.">Why can't linux write more than 2147479552 bytes?</a>：在 man 2 write 的 NOTES 章节包含以下注释：在 Linux 上，write()（及类似的系统调用）最多传输 0x7ffff000 (2,147,479,552) 字节，返回...</li><li><a href="https://www.youtube.com/watch?v=QthAU-t3PQ4">Value Semantics: Safety, Independence, Projection, &amp; Future of Programming - Dave Abrahams CppCon 22</a>：https://cppcon.org/---C++ 值语义：安全性、独立性、投影与编程的未来 - Dave Abrahams - CppCon 2022</li><li><a href="https://docs.modular.com/mojo/manual/python/#import-a-python-module">Python integration | Modular Docs</a>：同时使用 Python 和 Mojo。</li><li><a href="https://www.modular.com/blog/the-next-big-step-in-mojo-open-source">Modular: The Next Big Step in Mojo🔥 Open Source</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：Mojo🔥 开源的下一个重大步骤。</li><li><a href="https://github.com/modularml/mojo/blob/main/proposals/ref-convention.md">mojo/proposals/ref-convention.md at main · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 开发做贡献。</li><li><a href="https://github.com/thatstoasty/stump/">GitHub - thatstoasty/stump: WIP Logger for Mojo</a>：开发中的 Mojo 日志工具。</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/src/builtin/builtin_slice.mojo">mojo/stdlib/src/builtin/builtin_slice.mojo at nightly · modularml/mojo</a>：Mojo 编程语言标准库。</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/src/utils/span.mojo">mojo/stdlib/src/utils/span.mojo at nightly · modularml/mojo</a>：Mojo 编程语言标准库。</li><li><a href="https://github.com/modularml/mojo/issues/2610">[Feature Request] Add more `List` methods to the `InlineList` struct · Issue #2610 · modularml/mojo</a>：审查 Mojo 的优先级。我已阅读路线图和优先级，并认为此请求符合优先级要求。InlineList struct 已添加到 stdlib...
</li>
</ul>

</div>

### **Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1260025466849333390)** (3 messages): 

> - `Clock Calibration Issue` (时钟校准问题)
> - `Timer Cycle Functions` (定时器周期函数)


- **时钟校准导致 1 ns 差异**：一名成员指出，连续运行两个 `_clock_gettime` 调用会导致 1 ns 的差异，这在尝试使用一个时钟校准另一个时钟时会产生问题。
   - *由于这是在尝试进行校准，微小的差异会影响准确性*，从而导致潜在的时间偏差。
- **适用于多种架构的定时器周期函数**：分享了在 x86 和 ARM 架构上获取定时器周期的代码，分别使用了特定的 LLVM intrinsics 和内联汇编。
   - 如果未检测到这两种架构，该函数默认进行错误处理，打印错误消息并退出。


  

---


### **Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/)** (1 messages): 

Zapier: Modverse Weekly - 第 39 期
https://www.modular.com/modverse/modverse-weekly-issue-39
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1260063335819378840)** (19 messages🔥): 

> - `Mojo Compiler Nightly Update` (Mojo 编译器 Nightly 更新)
> - `Conditional Conformance in Mojo` (Mojo 中的条件一致性)
> - `Handling Unix FIFO in Mojo` (Mojo 中的 Unix FIFO 处理)
> - `Load Iris Dataset in Mojo` (在 Mojo 中加载 Iris 数据集)
> - `Mojo Language Improvements` (Mojo 语言改进)


- **Mojo 编译器发布新的 Nightly 更新**：发布了新的 Nightly 版本 Mojo 编译器 [版本 2024.7.905](https://github.com/modularml/mojo/compare/bc18cb454cd1bf7384da6eb86f79907b589c2419...d836be2d478bad12588843ce2b781e3c611df390)，更新内容包括清理 `memcmp` 的使用、重构 `setitem/setattr` 的发射 (emission)，以及改进条件一致性的参数推导。
   - 详细更新请查看 [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)。
- **Mojo 中关于条件一致性支持的辩论**：成员们讨论了 Mojo 虽然支持方法约束，但缺乏对类型条件一致性 (conditional conformance) 的完整支持，这阻碍了组合 (composition)。
   - Chris 表示希望弃用当前的条件一致性表示方式，并参考了一个 [commit](https://github.com/modularml/mojo/commit/97d70d3ecdfa289e61c33c323c3e04a71c19038a)。
- **提出了 Mojo 处理 Unix FIFO 的问题**：一名成员在尝试使用 Mojo 以写入模式打开 Unix FIFO 文件时遇到错误，导致异常。
   - 已在 [GitHub 上提交了 Issue](https://github.com/modularml/mojo/issues/3208)，请求提供关于权限 (777) 和完整脚本的更多细节。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/modularml/mojo/commit/97d70d3ecdfa289e61c33c323c3e04a71c19038a">[mojo-lang] 改进条件一致性的参数推导 · modularml/mojo@97d70d3</a>：此提交修复了参数推导，以处理比仅 &amp;quot;x.method(&amp;quot; 更复杂的条件一致性案例，特别是包括使用二元运算符 `x == y` 的案例等...</li><li><a href="https://github.com/modularml/mojo/issues/3208">[BUG] 以 &quot;write&quot; 模式打开 unix fifo 会引发异常 · Issue #3208 · modularml/mojo</a>：Bug 描述：我不确定为什么会失败，在 Discord 上提到过并被要求提交 Issue：$ mojo run src/main.mojo 执行期间捕获到未处理的异常：无法移除已存在的...
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo-marathons](https://discord.com/channels/1087530497313357884/1255303604894437388/1259949792050544800)** (17 messages🔥): 

> - `Vectorization Performance` (向量化性能)
> - `Algorithm Benchmarking` (算法基准测试)
> - `Load/Store Issues` (加载/存储问题)
> - `Benchmark Stabilization Tips` (基准测试稳定技巧)


- **向量化性能变量**：讨论显示，使用不同的宽度进行向量化会导致性能波动；在基准测试中，**宽度 1** 有时优于 **宽度 2**。
   - 进一步调查显示，**Mojo** 的 vectorize/load/store 实现可能在 **宽度 2** 时较慢，但在 **宽度 4 或 8** 时更快，具体取决于 M、N 和 K 的值。
- **关于基准测试中更公平比较的建议**：成员们讨论了包含对称和非对称矩阵的重要性，以确保算法基准测试中更公平的比较。
   - 一位成员建议为中大型矩阵设置 **m=n=k**，而另一位成员指出有必要测试 **geo** 和 **image** 用例中常见的维度。
- **分享基准测试稳定技术**：分享了稳定基准测试结果的技巧，包括指向[一篇详细博客文章](https://easyperf.net/blog/2019/08/02/Perf-measurement-environment-on-Linux)的链接。
   - 该指南涵盖了禁用 **turboboost** 和 **hyper threading**、设置 **cpu affinity** 以及使用**统计方法**等步骤。



**提及的链接**：<a href="https://easyperf.net/blog/2019/08/02/Perf-measurement-environment-on-Linux">How to get consistent results when benchmarking on Linux? | Easyperf </a>：未找到描述

  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1259953729445298176)** (54 messages🔥): 

> - `Custom Voices with LLM` (LLM 自定义语音)
> - `Image Generation and Tools` (图像生成与工具)
> - `Local Perplexica with LM Studio` (在 LM Studio 中使用本地 Perplexica)
> - `Running LLMs on Android` (在 Android 上运行 LLM)
> - `Text-to-Speech Front Ends` (文本转语音前端)


- **LM Studio 的自定义语音**：一位成员询问如何将 Eleven Lab 自定义语音与 LM Studio 集成，另一位成员建议使用 LM Studio 的 server 功能为此目的构建自定义程序。
   - 社区成员强调，集成文本转语音或自定义语音通常需要额外的编程，可以使用 Claude 等工具辅助开发。
- **使用 Stability Matrix 和 Fooocus 进行 AI 图像生成**：成员们讨论了 LM Studio 无法像 DALL-E 那样生成图像的问题，并推荐使用 Stability Matrix、Fooocus 和 Stable Diffusion 等工具来获得全面的 AI 图像生成能力。
   - 为了易于使用，建议初学者使用 Fooocus，而对于高级用户，则推荐使用 Stable Diffusion 以及 StableSwarmUI 和 Automatic1111 等界面。
- **通过 LM Studio Server 运行本地 Perplexica**：一位用户询问如何将 Perplexica（Perplexity AI 的开源替代品）与 LM Studio server 配合使用。
   - 另一位用户引用了 GitHub issue 的讨论以寻求潜在解决方案，但承认在将 Perplexica 与 LM Studio 连接时仍存在问题。
- **通过 Termux 在 Android 上运行 LLM**：分享了一个在 S21 Ultra 上使用 llama.cpp 和 Termux 运行 Mistral 7b 的用例，速度接近每秒 10 个 tokens。
- **为 LM Studio 引入文本转语音**：成员们强调，通过构建自定义前端并利用 Claude 等工具进行实现，为 LM Studio 创建文本转语音集成是可行的。
   - 有人指出，虽然这不是 LM Studio 开发的优先级，但使用 server 模式可以有效地促进这些集成。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/blog/abhishek/phi3-finetune-macbook">How to Finetune phi-3 on MacBook Pro</a>：未找到描述</li><li><a href="https://github.com/ItzCrazyKns/Perplexica">GitHub - ItzCrazyKns/Perplexica: Perplexica is an AI-powered search engine. It is an Open source alternative to Perplexity AI</a>：Perplexica 是一个 AI 驱动的搜索引擎。它是 Perplexity AI 的开源替代品 - ItzCrazyKns/Perplexica</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui">GitHub - AUTOMATIC1111/stable-diffusion-webui: Stable Diffusion web UI</a>：Stable Diffusion web UI。在 GitHub 上为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=g2BMJVM5ZZE).">Mac users: Stable diffusion 3 on ComfyUI</a>：在 Macbook (M 系列处理器) 上使用 ComfyUI 运行 SD3 的分步指南。#stablediffusion #applesilicon 👉ⓢⓤⓑⓢⓒⓡⓘⓑⓔ https://medium.com/@ttio2tech_28094/stab...</li><li><a href="https://github.com/ItzCrazyKns/Perplexica/issues/128#issuecomment-2123993463">LM studio support (Ollama alternative) · Issue #128 · ItzCrazyKns/Perplexica</a>：提前感谢
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1259973664346996846)** (75 条消息🔥🔥): 

> - `InternLM 上下文处理`
> - `使用 LLM 进行网页抓取`
> - `AI 编程局限性`
> - `Gemma 2`
> - `QLo 性能` 


- **InternLM 上下文窗口表现惊人**：成员们观察到 **InternLM** 即使在内存满载时也能保持连贯性，展示了滑动上下文窗口（sliding context window）的能力。
   - 讨论中包含的截图显示了 **InternLM** 在超过上下文长度后虽然会遗忘早期的消息，但不会变得语无伦次。
- **使用 AI 创建自定义网页抓取工具**：一位成员分享了他们使用 Claude 在 20 分钟内创建一个 **Node.js 网页抓取工具**的成功经验，该工具完全根据其需求定制。
   - “我得到了 78 行完全符合我要求的代码，” 强调了 AI 在快速创建自定义工具方面的实用性。
- **AI 生成的代码：建议谨慎**：成员们讨论了 AI 生成代码的可靠性，警告不要盲目信任 AI，因为它可能会产生次优代码。
   - “务必将 AI 用于代码，但要确保你知道它在做什么，” 一位成员强调，主张进行人工审查和理解。
- **对 Gemma 2 模型尺寸的困惑**：一位用户幽默地质疑为什么会有 **Gemma 2 的 9B 版本**，而没有像 7B 或 8B 这样更小的版本。
   - 这一讨论因一个表达困惑和对模型尺寸选择好奇的 **Liam Neeson GIF** 而变得更加引人注目。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/ZweUbY0KIqk?si=XI94I9V-TC3Vscb4)">这条蠕虫是否证明了我们生活在计算机模拟中？ 🤯</a>：让我们探索这种微小的圆线虫，它已被完整地映射到神经元，并使用消费级计算机进行了模拟。这是模拟理论的有力证据...</li><li><a href="https://www.liquid.ai)">未找到标题</a>：未找到描述</li><li><a href="http://'">未找到标题</a>：未找到描述</li><li><a href="https://tenor.com/view/liam-neeson-why-darkman-gif-10580616">Liam Neeson Why GIF - Liam Neeson Why Darkman - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1260212890057707521)** (8 条消息🔥): 

> - `对即将推出的功能的兴奋`
> - `整理聊天的功能请求`
> - `下载速度问题`
> - `上下文窗口指示器的功能请求` 


- **对即将推出的功能的兴奋**：一位用户对即将推出的一些功能感到兴奋，并建议注册 [beta](https://discord.com/channels/1110598183144399058/1111797717639901324/1256323247704641609) 版本以获取最新更新。
- **整理聊天的请求**：有关于整理聊天的功能请求，例如重新排序聊天记录和设置文件夹的能力。
- **下载速度困扰**：一位用户问道：*“你好，有人知道为什么我的下载速度受限吗？”*
- **上下文窗口指示器**：一位用户建议为长对话中已超出上下文窗口的部分提供视觉指示器。
   - 另一位成员回应说，底部的上下文计数在此时会变橙色，但原用户认为他们不应该通过计算来弄清楚 AI 遗忘了什么。


  

---

### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1259967727452557383)** (18 条消息🔥): 

> - `RX 6800XT 多 GPU 设置`
> - `LLama 3 70B Q7 Instruct 的性能`
> - `RX 6800XT 对比 7900XTX`
> - `为 RTX 3090 组装台式机/服务器`
> - `对 AMD ROCm 支持的担忧` 


- **RX 6800XT 多 GPU 设置的可行性**：一位用户询问是否可以使用 4 张 RX 6800XT 配合 llama.cpp 进行多 GPU 设置，并质疑 LM Studio 是否支持自动拆分/配置功能。
   - 另一位用户确认多 GPU 可以工作，但指出由于模型被拆分到多张显卡上，性能不会有显著提升。
- **LLama 3 70B Q7 Instruct 在多 GPU 上的性能**：有人询问了 4 张 GPU 运行 LLama 3 70B Q7 Instruct 的性能，一位用户建议其性能将与使用单张 6800XT 相似。
   - 建议选择双 7900XTX 以获得更好的性能并降低复杂性。
- **为 RTX 3090 组装台式机/服务器**：一位用户概述了使用 X299 主板组装台式机/服务器的计划，以获得更好的 RAM 带宽，并有潜力增加第二张 RTX 3090。
   - 他们征求了能匹配其 Ryzen 7640u 性能的 CPU 建议，得到的建议是任何体面的现代游戏 CPU 都可以胜任。
- **LLM 任务中 RX 6800XT 与 7900XTX 的对比**：关于 4 张 RX 6800XT 还是 2 张 7900XTX 更适合 LLM 任务展开了辩论，后者因麻烦更少而被推荐。
   - 一位用户决定卖掉他们的 RX 6800XT 并购入 2 张 7900XTX 以提升性能。
- **对 AMD ROCm 支持寿命的担忧**：有人提出了关于 AMD 可能缺乏长期支持的警告，并引用了 Radeon VII 终止 ROCm 支持的例子。
   - 这一警告对于考虑在长期项目中投资 AMD GPU 的用户尤为重要。


  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1260284773255876658)** (2 条消息): 

> - `Yorkie 的可信度`
> - `可疑行为` 


- **用户争论 Yorkie 的真实性**：一位成员评论道 *“看着 Yorkie——不知道，我觉得他看起来挺靠谱的”*。
   - 另一位成员表示反对，坚持认为：*“非常可疑。相信我。”*
- **用户分歧凸显社区互动**：关于 Yorkie 的讨论演变成了社区内关于可信度和怀疑的辩论。
   - 成员们表达了不同的意见，反映了社区多元的观点。


  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1260323173669404844)** (1 条消息): 

> - `AMD 7700XT 显卡`
> - `LM Studio 更新问题`
> - `Fimbulvetr Q4_K_M 模型性能` 


- **LM Studio 更新后变慢**：一位成员注意到，在将 **LM Studio 从 0.2.24 更新到 0.2.27** 后，性能显著下降，系统变得超级慢。
- **怀疑是 AMD 7700XT 显卡问题**：该成员推测问题是否与其 **AMD 7700XT 显卡**有关，因为之前使用 **Fimbulvetr Q4_K_M** 模型时性能更好。


  

---


### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1259953063813185639)** (7 条消息): 

> - `LM Studio GPU offload`
> - `Linux 上的长上下文问题`
> - `Bug 报告流程`
> - `上下文需要 RAM 的建议` 


- **禁用 GPU Offload 以解决问题**：一位用户建议，在侧边配置菜单中禁用 GPU offload 可能会解决问题。
- **Linux 上的长上下文处理问题**：一位用户报告称，Linux 上的 LM Studio 在处理长上下文时很吃力，即使在 RAM 充足的情况下也会显示错误，而 Windows 则不会。
   - 他们提到 **llamafile** 在相同的 Linux 环境下可以毫无问题地处理高达 **65,535 tokens**。
- **提示在正确的频道报告 Bug**：一位新用户被引导至 bug 报告频道发布其问题，并提供更多细节和截图。
- **全上下文长度需要额外的 RAM**：一位用户提到，处理全上下文长度需要在文件大小的基础上增加额外的 GB 级 RAM。
   - 该用户澄清他们拥有 **32 GB RAM**，且仅在 Linux 上的 LM Studio 中遇到问题。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1259979963126251569)** (32 messages🔥): 

> - `模型中理想的输出分布`
> - `Chinchilla 与 Gopher 训练计算量对比`
> - `测试时训练 (Test time training)`
> - `合成数据生成工具` 


- **诱导理想输出分布的挑战**：一位成员讨论了在模型中诱导理想输出 *分布 (distribution)* 的挑战，指出虽然可以通过在具有目标分布的足够数据上进行 SFT 来部分实现，但目前还没有类似 RLHF 的设置能让系统自行找到最佳分布。
   - 另一位参与者建议通过向“好”集合优化并远离“坏”集合，来潜在地引导输出分布。
- **Chinchilla 和 Gopher 的对比揭示了计算效率差距**：围绕 **Gopher** 和 **Chinchilla** 模型的讨论强调了训练设置的差异，尽管整体训练目标相似，但 Gopher 的效率较低。
   - 参与者辩论了关于 TFLOP 效率的假设以及模型大小的影响，一致认为较大的模型往往计算效率较低。
- **测试时训练增强 Transformer 性能**：一篇新论文提出使用测试时训练 (TTT) 通过对未标记的测试实例进行自监督学习 (self-supervised learning) 来改进模型预测，在 ImageNet 等基准测试上显示出显著改进。
   - TTT 可以集成到线性 Transformer 中，实验设置中用神经网络替代线性模型显示出性能增强。
- **对合成数据生成工具的兴趣增加**：一位参与者询问了专门辅助合成数据生成的工具，寻求增强其项目的解决方案。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.10859">Forcing Diffuse Distributions out of Language Models</a>：尽管经过专门训练以遵循用户指令，但当今的语言模型在被要求产生随机输出时表现不佳。例如，当提示均匀地（uniformly）选择一个数字时...</li><li><a href="https://openreview.net/forum?id=l7n59aufeT">Learning to (Learn at Test Time)</a>：对于每个未标记的测试实例，测试时训练 (TTT) 在做出预测之前对该单个实例执行自监督学习。我们将自监督任务参数化，并且...</li><li><a href="https://arxiv.org/abs/2407.04620">Learning to (Learn at Test Time): RNNs with Expressive Hidden States</a>：Self-attention 在长上下文中表现良好，但具有平方复杂度。现有的 RNN 层具有线性复杂度，但它们在长上下文中的性能受到其隐藏状态表达能力的限制...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1259954289108058163)** (77 条消息🔥🔥): 

> - `TTT-Linear 与 Delta Rule`
> - `TTT-MLP 优化`
> - `使用 In-Run Data Shapley 进行数据归因`
> - `梯度归一化技术`
> - `新兴 RNN 架构 vs. Transformers` 


- **TTT-Linear 在 mini batch size 为 1 时与 Delta Rule 匹配**：成员们讨论了 **TTT-Linear** 在 mini batch size 为 1 时等同于 **delta rule**，并且在这种情况下表现最佳。
   - 有人指出 **TTT-MLP** 性能更强，但更难优化；另一位成员补充说，像 **rwkv7** 这样即将发布的作品计划加入改进后的 delta rule。
- **基于 In-Run Data Shapley 的规范化数据归因**：一个名为 **In-Run Data Shapley** 的项目发布了，它提供了一个可扩展且正式的框架，用于在预训练期间实时评估数据贡献，从而高效地识别出大量负价值数据。
   - 社区成员认为，这可以通过数据集筛选帮助**防止模型**产生**不良能力**，并更好地理解“涌现（emergence）”。
- **新的梯度归一化技术**：介绍了一种新的**梯度归一化（gradient normalization）**技术，该技术在反向传播中使用归一化层来控制梯度流，解决了极深网络中的梯度消失或爆炸问题。
   - 然而，成员们认为其对 batch-size 的依赖以及在 batch 维度归一化时需要跨设备通信是其缺点。
- **新兴 RNN 架构与 Transformers 竞争**：像 **Mamba** 和 **RWKV** 这样新的**循环大语言模型（recurrent LLMs）**在推理过程中提供恒定的内存占用，并在语言建模困惑度（perplexity）方面成为 Transformers 的有力竞争者。
   - 挑战仍然在于如何有效地管理内存以在长上下文中召回信息，最近的研究在理论和实证方面对此进行了探讨。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/karansdalal/status/1810338845659131940">Karan Dalal (@karansdalal) 的推文</a>: 我很高兴分享一个我工作了一年多的项目，我相信它将从根本上改变我们处理语言模型的方式。我们设计了一个新的架构，它取代了 h...</li><li><a href="https://arxiv.org/abs/2406.06484">Parallelizing Linear Transformers with the Delta Rule over Sequence Length</a>: 具有线性注意力（即 linear transformers）和状态空间模型（state-space models）的 Transformers 最近被提议作为具有 softmax 注意力的 transformers 的可行线性时间替代方案。然而，...</li><li><a href="https://x.com/ruoxijia/status/1810444361622245614">Ruoxi Jia (@ruoxijia) 的推文</a>: 1/n 对可扩展、规范的数据归因方法感兴趣吗？介绍 In-Run Data Shapley，一种足以用于预训练数据归因的高效方法！ (https://jiachen-t-wang.github.io/data-sha...</li><li><a href="https://arxiv.org/abs/2106.09475">Backward Gradient Normalization in Deep Neural Networks</a>: 我们介绍了一种在神经网络训练期间进行梯度归一化的新技术。在反向传播过程中，使用在某些点引入的归一化层对梯度进行重新缩放...</li><li><a href="https://arxiv.org/abs/2407.04358v1">An Adaptive Stochastic Gradient Method with Non-negative Gauss-Newton Stepsizes</a>: 我们考虑最小化大量平滑但可能非凸函数平均值的问题。在大多数机器学习应用的背景下，每个损失函数都是非负的...</li><li><a href="https://x.com/SonglinYang4/status/1810589870487908521">Songlin Yang (@SonglinYang4) 的推文</a>: TTT-linear 的在线梯度下降版本是 DeltaNet 的一个变体，可以被高效地并行化：https://arxiv.org/abs/2406.06484 引用 Aran Komatsuzaki (@arankomatsuzaki) Learning...</li><li><a href="https://arxiv.org/abs/2407.05483">Just read twice: closing the recall gap for recurrent language models</a>: 在语言建模困惑度方面与 Transformers 竞争的循环大语言模型正在迅速涌现（例如 Mamba, RWKV）。令人兴奋的是，这些架构使用恒定数量的...</li><li><a href="https://github.com/HazyResearch/prefix-linear-attention">GitHub - HazyResearch/prefix-linear-attention</a>: 通过创建一个账户为 HazyResearch/prefix-linear-attention 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1260282205431922708)** (9 messages🔥): 

> - `Brain size evolution` (大脑尺寸演化)
> - `Intelligence and evolutionary benefits` (智能与演化收益)
> - `Linearity of brain size` (大脑尺寸的线性关系)
> - `Neuronal density and intelligence` (神经元密度与智能)


- **大脑尺寸之谜已解**：发表在 [Nature Ecology & Evolution](https://www.nature.com/articles/s41559-024-02451-3) 上的一项研究揭示，体型最大的动物并不拥有成比例更大的大脑，而人类打破了这一趋势。
   - 雷丁大学和杜伦大学的研究人员收集了约 **1,500 个物种**的数据，以澄清围绕大脑尺寸演化的争议。
- **大脑尺寸的线性关系受到质疑**：有人指出，大脑尺寸图表中的黑线并非直线，而是在末端**略微弯曲**。
   - 这引发了关于“大型动物拥有更大大脑”这一关系的依赖性是否取决于其线性特性的疑问。
- **智能的生殖收益**：讨论了如果 Scaling 假设成立，在祖先环境中增加智能所带来的生殖收益是否有限。
   - 一位参与者指出，*大脑尺寸只是冰山一角*，结构和神经元密度同样重要。
- **不同物种的神经元密度与智能**：*在哺乳动物中*，由于结构基本一致，皮层神经元总数可以很好地映射智能分布。
   - *在鸟类和蜥蜴中*，所有神经元类型的密度更为重要，尽管除非按结构区分，否则数据非常稀缺。



**提及的链接**：<a href="https://phys.org/news/2024-07-brain-size-riddle-humans-exceed.html">Brain size riddle solved as humans exceed evolutionary trend</a>：发表在 Nature Ecology &amp; Evolution 上的一项研究揭示，体型最大的动物并不拥有成比例更大的大脑，而人类打破了这一趋势。

  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1260342891868590210)** (2 messages): 

> - `EleutherAI at ICML`
> - `ICML papers announcement` 


- **EleutherAI 确认参加 ICML**：**EleutherAI** 将携其论文参加 ICML。具体细节可在 [官方公告](https://discord.com/channels/729741769192767510/794042109048651818/1255332843534422038) 中找到。
- **ICML 参会者社交线程**：为参加 ICML 的人员开设了专门的社交线程。参与者可以在 <#1255332070369263707> 加入讨论。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1259947976701509693)** (6 messages): 

> - `Chain-of-Thought reasoning in models`
> - `Model's access to answer choices`
> - `RegexFilter for MedQA`
> - `Sampler initialization error`
> - `Error troubleshooting` 


- **CoT 推理不适用于多选题任务**：建议不要将 **Chain-of-Thought (CoT)** 推理用于 `multiple_choice` 任务。
- **模型在许多任务中能感知选项**：对于像 **MMLU** 这样的任务，答案是在上下文中提供的，因此模型知道它需要从中选择什么。
- **为 MedQA 适配 RegexFilter 导致错误**：尝试将 **RegexFilter** 从 MMLU 适配到 MedQA 时，由于初始化中意外的参数计数导致 **TypeError**。
- **Sampler 意外参数问题**：错误信息 “**TypeError: __init__() takes from 1 to 3 positional arguments but 4 were given**” 表明可能有意外参数传递给了 **Sampler** 的初始化过程。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1260244392074149991)** (2 messages): 

> - `Containers on Kubernetes`
> - `Pods with Neox Image` 


- **在 Kubernetes 上使用容器**：成员确认他们在 **Kubernetes** 上使用容器进行部署。
- **使用 Neox 镜像部署 Pods**：他们提到这些容器专门运行 **Neox 镜像**来管理 Pods。


  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1259947297148764283)** (119 messages🔥🔥): 

> - `模型训练技术`
> - `AI 中的 Booru 标签`
> - `AI 在社会中的角色`
> - `SD 扩展与工具` 


- **不同分辨率下的模型训练**：成员们讨论了在微调阶段，先在 **512x512 分辨率**上进行训练，然后再进行 **1024x1024** 训练是否更有利。
- **模型训练中的 Booru 标签争议**：讨论集中在将 **booru tags** 用于训练 AI 的问题上，一些成员为其已建立的词汇体系辩护，而另一些成员则质疑其相对于更自然语言模型的有效性。
- **AI 的文化影响与监管**：对话揭示了对 AI 社会影响的担忧，特别是在成瘾和潜在的**未来监管**方面。
- **用于换脸的 Roop-Unleashed**：一位成员推荐使用 **Roop-unleashed** 作为视频中保持一致换脸的工具，强调了它比已经失效的 mov2mov 扩展更有效。
- **SD 模型推荐与使用技巧**：成员们交流了针对各种特定任务（如像素艺术转换和局部重绘）的**模型和扩展推荐**，建议使用 **Zavy Lora** 以及配合 **IP adapters** 的 **ComfyUI** 等工具。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/bcZXlhy7KDE">MP Productions (Mark Pritchard) - One Way Mirror (Official Audio)</a>：One Way Mirror (Official Audio) Stream: https://markpritchard.ffm.to/one-way-mirror Visual by Jonathan Zawada。该艺术作品是使用 GAN (生成对抗网络) 创作的...</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Installation-Guides#nvidia-automatic1111-webui-stable-diffusion-webui">安装指南</a>：Stable Diffusion 知识库（设置、基础、指南等）- CS1o/Stable-Diffusion-Info
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1260046881275252806)** (67 messages🔥🔥): 

> - `DALL-E 替代方案`
> - `AI 文本检测器`
> - `StableDiffusion`
> - `Diffusion 工具`
> - `AI 模型推荐` 


- **成员讨论 DALL-E 替代方案**：成员们探索了 **DALL-E** 的替代方案，例如 **StableDiffusion** 以及 **DiffusionBee** 和 **automatic1111** 等平台，以获得更多控制权和更好的质量。
   - 他们还讨论了在 Windows 或 Mac 上本地运行这些模型。
- **辩论 AI 文本检测器的可靠性**：许多人反映 **AI 文本检测器**不可靠，会错误地标记 AI 和人类创作的内容。
   - 一位成员幽默地提到，**《美国宪法》**被此类工具错误地标记。
- **StableDiffusion 及其工具**：社区提到了用于 **StableDiffusion** 的各种工具，包括适用于 Mac 的 **DiffusionBee** 和适用于 Windows 的 **automatic1111**。
   - **ComfyUI** 被强调为 Windows 用户的另一个不错选择。
- **在 Web 端寻找高质量图像生成器**：一位成员对缺乏高质量的 Web 端图像生成器以及电脑速度慢表示沮丧。
   - 尽管有局限性，**DALL-E** 和 **MidJourney** 仍被推荐为基于 Web 的选项。
- **使用 DALL-E API 生成图标的 Python 代码**：成员们分享了使用 **DALL-E API** 生成图标的 Python 代码，并讨论了各种图像属性的参数。
   - 建议的参数包括尺寸、质量、风格、调色板和其他艺术细节。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://deepmedia.ai/">Deep Media AI</a>：为企业、政府和记者提供 AI 驱动的 Deepfake 检测和媒体情报。</li><li><a href="https://deepmedia.ai/Blog">博客 | Deep Media AI</a>：我们的专家团队分享他们对 AI、Deepfakes、检测最佳实践、现代信任与安全等方面的见解。
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1259947966777659568)** (9 messages🔥): 

> - `GPTs 的货币化`
> - `VPN 导致 GPTs 出现问题`
> - `服务器问题已解决`
> - `GPT 回复的一致性`
> - `用户不满` 


- **GPTs 的货币化：何时开始？**: 一位用户询问了 **GPTs 货币化** 的时间表，但未得到确切答复。
   - *未讨论进一步的细节或回复。*
- **VPN 问题影响 GPTs**: 一位用户指出开启 **VPN 会导致 GPTs** 回复出现问题，并建议禁用 VPN 以解决该问题。
   - *未提供关于此话题的额外评论或链接。*
- **服务器问题已解决**: 一位用户提到最近的 **服务器问题** 已得到解决，但未提供具体细节。
   - *对此话题没有进一步讨论。*
- **保持 GPT 回复的一致性**: 一位成员询问如何在不同语境下保持 **GPT 回复的一致性**，特别是关于语言偏好的问题。
   - 另一位用户回答说，在没有看到具体对话的情况下，很难回答此类问题。
- **用户对 GPT 服务的不满**: 一位用户对 **ChatGPT 的表现** 表示沮丧，并提到转向竞争对手，分享了[相关对话](https://chatgpt.com/share/fceb3b81-e719-45eb-9f7a-e58da17f20a0)的链接。
   - 另一位用户澄清说，**回复中的幻觉 (hallucinations)** 是由于 LLM 的训练数据造成的，并引导用户查看 [OpenAI 的价格页面](https://openai.com/chatgpt/pricing/)以获取 context window 详情。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1259963005303525476)** (2 messages): 

> - `内容创作策略`
> - `受众参与度`
> - `平台优化`
> - `内容日历结构`
> - `内容成功的关键指标` 


- **寻求新鲜内容创意以增加受众**: 一位内容创作者根据其领域的趋势话题请求 **5-10 个新鲜内容创意**，以及**提升参与度**和针对各种平台优化内容的有效策略。
   - 他们还寻求了关于创建 **内容日历结构** 以及衡量内容成功和粉丝增长所需追踪的关键指标的建议。
- **内容创作和参与技巧**: 内容创作者询问了**增强参与度的策略**，例如有效使用标签、与粉丝互动以及强有力的 call-to-actions。
   - 此外，他们还请求了针对 Instagram, YouTube, TikTok 和其他社交媒体平台优化内容的**特定平台建议**。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1259963005303525476)** (2 messages): 

> - `内容创作`
> - `受众参与度`
> - `社交媒体策略`
> - `内容日历`
> - `指标追踪` 


- **寻求增加受众的内容创意**: 用户正在寻找基于其领域趋势话题的 **5-10 个新鲜内容创意**，以增加受众并提高参与度。
   - 用户问道：*“你能为我提供内容创意、参与技巧、特定平台建议、内容日历结构和关键指标吗？”*
- **关于提高参与度的问题**: 用户正在询问**提升帖子参与度的策略**，例如有效的标签、与粉丝互动以及 call-to-actions。
   - 他们正在寻求关于为 Instagram, YouTube 和 TikTok 等不同平台**优化内容**以吸引新粉丝的建议。


  

---

### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1260280533817102467)** (1 messages): 

> - `LlamaCloud Beta 版本发布`
> - `数据质量`
> - `可扩展性障碍`
> - `LlamaParse 集成` 


- **LlamaCloud 预发布公告**：宣布了 **LlamaCloud** 的 Beta 版本，这是一个为非结构化数据解析、索引和检索提供的托管平台。用户可以加入 [候补名单](https://docs.google.com/forms/d/e/1FAIpQLSdehUJJB4NIYfrPIKoFdF4j8kyfnLhMSH_qYJI_WGQbDWD25A/viewform) 进行体验。
- **确保高质量数据**：**LlamaCloud** 通过提供高质量的数据输入和供 LLM 交互的复杂接口，解决了“垃圾进，垃圾出”的问题。
   - *数据质量问题* 是一个常见难题，LlamaCloud 旨在通过高级解析和索引等功能来解决这一问题。
- **通过解析解决可扩展性障碍**：**LlamaCloud** 旨在减少对新数据源进行自定义解析和调优所需的工程时间。它承诺通过连接到 Sharepoint、S3 和向量数据库的高级连接器实现多种数据源的同步。
- **LlamaParse：内置高级解析功能**：**LlamaParse** 已集成在 LlamaCloud 中，提供高级文档解析能力。
   - 它专为处理复杂文档而设计，并确保与其高级检索接口层同步。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.google.com/forms/d/e/1FAIpQLSdehUJJB4NIYfrPIKoFdF4j8kyfnLhMSH_qYJI_WGQbDWD25A/viewform">LlamaCloud 候补名单</a>：感谢您对 LlamaCloud 的关注！请在下方注册并告知我们您使用的邮箱地址，我们将以适度的节奏允许用户加入。</li><li><a href="https://x.com/llama_index/status/1810716602247348242">来自 LlamaIndex 🦙 (@llama_index) 的推文</a>：今天我们很高兴发布 LlamaCloud 的 Beta 版本——这是为您 LLM 应用程序准备的数据处理层。任何 RAG 流水线/Agent 的质量都取决于您的数据。LlamaCloud 提供了一个托管平台...</li><li><a href="https://www.llamaindex.ai/blog/llamacloud-built-for-enterprise-llm-app-builders">LlamaCloud - 为企业级 LLM 应用构建者打造 — LlamaIndex，LLM 应用的数据框架</a>：LlamaIndex 是一个简单、灵活的数据框架，用于将自定义数据源连接到大语言模型 (LLMs)。</li><li><a href="https://cloud.llamaindex.ai/">LlamaCloud</a>：未找到描述</li><li><a href="https://docs.cloud.llamaindex.ai/">欢迎 | LlamaCloud 文档</a>：这是 LlamaCloud 的文档，它是 LlamaIndex 的托管摄取和索引服务。</li><li><a href="https://youtu.be/3hc98dtMfFc">LlamaCloud 简介</a>：LlamaCloud 为您的 LLM 应用程序提供数据处理层。它让您能够构建企业级的上下文增强 RAG 流水线、Agent 等...
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1259970239991517265)** (3 messages): 

> - `LlamaIndex 中的 Property Graphs`
> - `LlamaCloud Beta 版本发布`
> - `AGI House 黑客松` 


- **令人兴奋的 LlamaIndex Property Graphs 六部分视频系列**：宣布与 **mistralai、neo4j 和 ollama** 合作推出关于 [Property Graphs 的六部分视频系列](https://twitter.com/llama_index/status/1810410943215710510)。
   - *从文档中建模复杂关系*，其特征是节点和边上均带有属性。
- **LlamaCloud Beta 版本发布**：宣布了 [LlamaCloud](https://twitter.com/llama_index/status/1810716602247348242) 的 Beta 版本，这是 LLM 应用程序的数据处理层。
   - LlamaCloud 为 RAG 流水线和 Agent 提供了一个**用于非结构化数据解析、索引和检索的托管平台**。
- **AGI House 黑客松邀请**：邀请参与者参加 **7/13 周六** 在 AGI House 举行的黑客松，合作伙伴包括 **togethercompute 和 SambaNovaAI**。
   - 申请详情见 [此处](https://twitter.com/llama_index/status/1810820193104580941)。



**提到的链接**: <a href="https://t.co/LOEgpc1BOs">AGI House</a>：未找到描述

  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1260148444157186098)** (65 条消息🔥🔥): 

> - `电商 RAG 聊天机器人增强`
> - `FlagEmbeddingReranker 导入错误`
> - `Groq API 的速率限制问题`
> - `处理聊天机器人的大型数据集`
> - `astream_chat 实现问题` 


- **增强电商 RAG 聊天机器人**：一位用户分享了他们使用关键词搜索、向量搜索和元数据过滤成功的原型，但他们正寻求添加后续提问功能，以处理有关构建项目（如桌子）的查询。
   - 他们讨论了潜在的方法，例如先进行混合搜索，然后细化发送给 LLM 的属性以进行后续提问。
- **受困于 FlagEmbeddingReranker 导入**：尽管在全球范围内安装了所需的包并调试了环境，一位用户仍遇到持续的 `FlagEmbeddingReranker` 导入错误。
   - 该问题最终通过单独安装 `peft` 得到解决，这在最初并未预料到。
- **Groq API 速率限制困境**：一位用户报告在使用 LlamaIndex 配合 Groq API 进行向量存储索引时出现错误代码 429（速率限制）。
   - 进一步讨论表明，该问题与 Groq 使用 OpenAI 客户端有关，可能与 OpenAI 默认的 embedding 模型相关。
- **聊天机器人大型数据集的有效策略**：一位用户寻求关于管理源自大量 PDF 的大型 Markdown 数据集的建议，以构建有效的 RAG 聊天机器人。
   - 他们询问了关于 loader、分块策略和向量数据库（最好是开源选项）的建议。
- **astream_chat 实现困难**：一位用户在 LlamaIndex 中实现 `astream_chat` 时遇到问题，收到关于 asyncio 方法的错误。
   - 经过多次尝试和调试，他们成功让异步生成器工作，但指出在 Server Side Event (SSE) 设置中使用时，它并没有像 `stream_chat` 那样按预期进行流式传输。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/asimdotshrestha/status/1810720478111371581">来自 Asim Shrestha (@asimdotshrestha) 的推文</a>：很高兴能更广泛地分享我们在 @ReworkdAI 正在做的工作 ⚡️ 过去一年我们全身心投入于构建下一代 web agents。它们已经在生产环境中上线...</li><li><a href="https://github.com/vsakkas/sydney.py">GitHub - vsakkas/sydney.py: Copilot（原名 Bing Chat，也称为 Sydney）的 Python 客户端。</a>：Copilot（原名 Bing Chat，也称为 Sydney）的 Python 客户端。 - vsakkas/sydney.py</li><li><a href="https://github.com/run-llama/llama_index/blob/510213d07b01ba4e80762f2c1ca3af61ed935074/llama-index-integrations/postprocessor/llama-index-postprocessor-flag-embedding-reranker/llama_index/postprocessor/flag_embedding_reranker/base.py#L31">llama_index/llama-index-integrations/postprocessor/llama-index-postprocessor-flag-embedding-reranker/llama_index/postprocessor/flag_embedding_reranker/base.py at 510213d07b01ba4e80762f2c1ca3af61ed935074 · run-llama/llama_index</a>：LlamaIndex 是一个用于你的 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/evaluation/semantic_similarity/#llama_index.core.evaluation.SemanticSimilarityEvaluator>):">语义相似度 - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1259949631580803072)** (43 messages🔥): 

> - `Arc Search 推荐`
> - `Perplexity 中的上下文问题`
> - `Notion 与 Perplexity 的集成`
> - `Claude 3.5 与 Gemini 1.5 的对比`
> - `API 额度说明` 


- **相比 Perplexity 更推荐 Arc Search**：一位用户建议尝试将 [Arc Search](https://arch.is) 作为 Perplexity AI 的替代方案。
   - 另一位用户将其标记为“精明（shrewd）”，表示赞同或认可。
- **Perplexity 的上下文处理问题**：有用户反映 Perplexity 在后续问题中经常丢失上下文，需要非常具体的查询才能维持。
   - 另一位用户表示同意，并补充说 **GPT-4o 比 Claude 3.5 能更好地维持上下文**，但 **Perplexity 有时会在后续提问中进行无关的搜索**。
- **可通过 Make 和 BuildShip 实现 Notion 集成**：针对寻求将 Notion 与 Perplexity 连接的用户，有人提供了 [Make](https://www.make.com/en/integrations/perplexity-ai/notion) 和 [BuildShip](https://buildship.com/integrations/apps/notion-and-perplexity) 等集成资源。
   - 该用户对提供的信息表示感谢，说明这些资源很有帮助。
- **Claude 3.5 与 Gemini 1.5 的对比混淆**：一位用户链接了 Gemini 1.5 Flash 与 Claude 3 Haiku 的[对比](https://www.perplexity.ai/search/gemini-1-5-flash-vs-claude-3-h-061hEtXqQ_ORe7BrsDfK1Q)，并指出在上下文窗口和价格方面存在差异。
   - 其他人指出该对比不准确，并强调了模型之间的区别，特别是 **Gemini 1.5 Flash 和 Pro** 之间的差异。
- **API 额度使用说明**：针对使用 Perplexity 服务器的开发者，对其 API 额度使用情况进行了说明，指出了用于上下文和 Token 使用的用途及可用模型规格。
   - 分享了模型和参数的详细信息，包括 [Meta 的建议](https://github.com/facebookresearch/llama/blob/008385a/UPDATES.md#token-sanitization-update) 关于 Token 清洗（sanitization）的更新。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://exa.ai/">Exa</a>：Exa API 从网络检索最佳的实时数据，以补充您的 AI。</li><li><a href="https://docs.perplexity.ai/docs/model-cards">支持的模型</a>：未找到描述。</li><li><a href="https://youtu.be/GzEhgl7uy0Y?si=DzwrTRkQEPF2DMjK">Feather 1.5 - Paint, Shift and Shoot</a>：Feather 现已发布大量新功能。绘画、编辑，甚至动画化你以前无法想象的创意。1.5 版本的核心功能 - 新笔刷...</li><li><a href="https://www.perplexity.ai/search/gemini-1-5-flash-vs-claude-3-h-061hEtXqQ_ORe7BrsDfK1Q">编程中的 Gemini 1.5 Flash vs Claude 3 Haiku</a>：根据现有信息，这里是 Gemini 1.5 Flash 和 Claude 3 Haiku 的对比，重点关注它们的编程能力及相关...</li><li><a href="https://www.sequoiacap.com/article/follow-the-gpus-perspective/">AI 的 2000 亿美元问题</a>：GPU 产能正处于过度建设状态。长期来看，这对初创公司有利；短期来看，情况可能会变得混乱。关注 GPU 以了解原因。</li><li><a href="https://x.com/appenz/status/1704915400096649696">Guido Appenzeller (@appenz) 的推文</a>：🔥 在最近的一篇文章中，红杉资本的 @DavidCahn6 认为 AI 基础设施过度建设：- NVIDIA GPU 年收入为 500 亿美元 - 这需要 2000 亿美元的“AI 收入” - 而目前只有 750 亿美元的“AI 收入”...</li><li><a href="https://buildship.com/integrations/apps/notion-and-perplexity">集成 Notion 和 Perplexity AI 以创建自动化</a>：连接 Notion 和 Perplexity AI 以自动化工作流。使用 BuildShip 进行无代码集成。构建后端、API 和 AI 工作流。低代码且可扩展。
</li>
</ul>

</div>

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1260078129716658208)** (10 条消息🔥): 

> - `Antikythera Mechanism`
> - `Nothing's New Phone`
> - `Hydrogen Cars`
> - `Boeing Guilty Plea`
> - `Digital Advertising in South Korea` 


- **波音公司就欺诈阴谋罪认罪 (Boeing Pleads Guilty to Fraud Conspiracy)**：波音公司已同意就其 737 MAX 飞机在 2018 年和 2019 年发生的致命坠机事故相关的刑事欺诈阴谋指控认罪，其中包括 2.436 亿美元的罚款、4.55 亿美元的安全投资以及为期三年的法院监督缓刑（[待批准](https://www.reuters.com/business/aerospace-defense/boeing-plead-guilty-us-probe-fatal-737-max-crashes-says-doj-official-2024-07-08/)）。
   - *该认罪协议还包括承认因违反先前的延期起诉协议而产生的阴谋罪。*
- **韩国顶尖数字广告渠道**：[Naver](https://saedu.naver.com/adguide/eng.naver) 和 [KakaoTalk](https://saedu.naver.com/adguide/eng.naver) 主导了韩国的数字广告领域，Naver 提供搜索和展示广告，而 KakaoTalk 提供基于消息的广告。
- **在 Fedora 上配置 WireGuard**：用户可以通过在 `/etc/wireguard/wg0.conf` 中创建配置文件并使用 `wg-quick` 命令启动接口，从而在 Fedora 上配置 WireGuard。
- **Perplexity AI Discord 社区的参与活动**：[Perplexity AI Discord 社区的成员](https://www.perplexity.ai/hub/faq/perplexity-discord-community)可以讨论该平台、测试新功能的 Beta 版本、与开发人员互动并分享个人使用案例。
- **第一版 D&D 攻击加成对比第二版 THAC0**：第一版 D&D 攻击加成与第二版 THAC0 之间的主要区别在于[第二版中引入的](http://beyondfomalhaut.blogspot.com/2019/07/blog-sinister-secret-of-thac0.html)攻击掷骰系统的简化。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/embed/Y1xeiqncRig">YouTube</a>：未找到描述</li><li><a href="https://www.perplexity.ai/search/wireguard-for-fedora-h9lIfmi9Q0iiJz8W1lSvtA">Wireguard for fedora</a>：以下是关于如何在 Fedora 上安装和配置 Wireguard VPN 查询的简要回答：要在 Fedora 上安装 Wireguard，请按照以下步骤操作：1. 对于...</li><li><a href="https://www.perplexity.ai/search/what-s-the-difference-between-.Rqak1XaSkyfy4pFv2MJlg#0">What&#x27;s the difference between 1st edition D&amp;D attack bonuses and 2nd edition...</a>：第一版 D&amp;D 攻击加成与第二版 D&amp;D THAC0 之间的主要区别在于它们如何简化和标准化攻击掷骰系统：1....</li><li><a href="https://www.perplexity.ai/search/upcoming-music-festivals-JzERaBnbTYK7vuY0dERhsA">Upcoming music festivals</a>：以下是 2024 年全美各地举办的一些著名音乐节：Outlaw Music Festival 2024 - 艺人：Bob Dylan, Willie Nelson, John...</li><li><a href="https://www.perplexity.ai/search/advertising-market-size-of-sou-LMUoI3pMRTec5ZkgahyUPg#2">advertising market size of South Korea</a>：在传统媒体和数字媒体渠道的推动下，韩国广告市场规模庞大且持续增长。广告总额...</li><li><a href="https://www.perplexity.ai/search/can-you-generate-an-ai-image-Cw9VQ9cpRO63iRiNifGRlg">Perplexity</a>：Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可靠且实时的答案。</li><li><a href="https://www.perplexity.ai/page/boeing-to-plead-guilty-LJzVVkFXReOcNpqfQ8EGLg">Boeing to Plead Guilty</a>：航空航天巨头波音公司已同意就其 737 MAX 飞机在 2018 年和...发生的致命坠机事故相关的刑事欺诈阴谋指控认罪。</li><li><a href="https://www.perplexity.ai/search/how-to-connect-with-perplexity-NNCxFfDqTo61sr0F0D9Ijg">how to connect with Perplexity.ai community on discord?</a>：要加入 Discord 上的 Perplexity AI 社区，请按照以下步骤操作：1. 访问 Perplexity AI 官方网站 (perplexity.ai)。2. 寻找...</li><li><a href="https://www.perplexity.ai/search/principality-of-sealand-60XhJQWxSVuVf37ZLfrQAQ">Principality of Sealand</a>：西兰公国 (Principality of Sealand) 是一个拥有丰富历史的迷人微型国家。以下是关于这个自封的主权国家的关键细节...</li><li><a href="https://www.perplexity.ai/page/blender-rendering-tips-aGlccrJTT_eSgzcofdFy7Q">Blender Rendering Tips</a>：Blender 是一款流行的开源 3D 创作套件，提供强大的渲染功能，可以产生惊人的效果。然而，对于初学者来说，...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1260204985606602929)** (10 messages🔥): 

> - `API vs UI results` (API 与 UI 结果对比)
> - `Nodemon setup issues with PPLX library` (PPLX 库的 Nodemon 设置问题)
> - `Rate limits and citation feature increases` (速率限制与引用功能的提升)


- **API 与 UI 结果之间的巨大差异**：一位成员对在不使用 Pro 版本或预期来源时，API 与 UI 结果之间存在的**巨大差异**表示担忧。
   - 另一位成员建议在 *labs* 中进行尝试，认为在未开启 Pro 或不需要来源的情况下，结果应该非常相似。
- **PPLX 库的 Nodemon 设置问题**：一位用户在运行使用 **PPLX** 库编译的项目时遇到问题，尽管在本地使用 **nodemon** 并已在 **tsconfig.json** 中指定了正确文件夹的情况下运行成功。
   - 错误详情提示为模块缺失问题；该用户寻求其他涉及 **PPLX** 库的配置方案反馈。
- **速率限制和引用功能的提升**：有人询问关于 **rate limits**（速率限制）和 **citation feature**（引用功能）可能提升的问题，但数周未收到回复。
   - *有没有人听说过关于速率限制或引用功能的提升？*


  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1259954724237475860)** (37 messages🔥): 

> - `Deepspeed Efficiency` (Deepspeed 效率)
> - `Open Source Video Upscalers` (开源视频放大器)
> - `PaintsUndo Project` (PaintsUndo 项目)
> - `AI System Copyright Lawsuit` (AI 系统版权诉讼)
> - `Copyright Term Opinions` (关于版权期限的观点)


- **Deepspeed 高效训练令人惊叹**：一位成员分享称，使用 **Deepspeed**，他们可以在 **RTX 3090** 上以 1 的 batch size 训练一个 **25 亿参数模型**，并指出可能还可以进一步提升。
- **探索开源视频放大器**：成员们讨论了各种**开源视频放大器 (video upscalers)**，其中一人推荐使用 **aurasr** 进行逐帧放大，但警告称其速度可能较慢。
- **PaintsUndo 增强艺术创作过程**：**PaintsUndo 项目**旨在提供人类绘画行为的基础模型，通过预测中间草图步骤，帮助 AI 更好地对齐人类艺术家的需求。
- **法院裁决在版权案中倾向于 AI 系统**：加州地方法院部分驳回了针对 **Microsoft** 的 **GitHub Copilot** 及其底层模型 **OpenAI** 的 **Codex** 的版权诉讼。
- **关于版权期限的辩论**：一位成员发表观点认为，版权应在**出版后 20 年**内有效，并强调需要彻底改革**美国版权立法**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lllyasviel.github.io/pages/paints_undo/">PaintsUndo: A Base Model of Drawing Behaviors in Digital Paintings</a>：未找到描述</li><li><a href="https://the-decoder.com/court-ruling-suggests-ai-systems-may-be-in-the-clear-as-long-as-they-dont-make-exact-copies/">Court ruling suggests AI systems may be in the clear as long as they don't make exact copies</a>：加州地方法院部分驳回了针对 Microsoft 的 GitHub Copilot 编程工具及其前底层语言模型 OpenAI 的 Codex 的版权诉讼。裁决表明...</li><li><a href="https://the-decoder.com/court-ruling-suggests-ai-systems-may-be-in-the-clear-as-long-as-they-dont-ma">THE DECODER</a>：人工智能正在改变世界。THE DECODER 为您带来关于 AI 的所有新闻。
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1260268276982747238)** (13 messages🔥): 

> - `Generative Chameleon` (生成式 Chameleon)
> - `Complex-Valued Architectures` (复数值架构)
> - `Vision Architecture with 2D DFT` (基于 2D DFT 的视觉架构)
> - `Training Challenges` (训练挑战)
> - `Model Scaling Issues` (模型缩放问题)


- **生成式 Chameleon 论文发布**：首个生成式 Chameleon 模型已发布，论文可在 [arXiv](https://arxiv.org/pdf/2407.06135) 获取。
- **探索复数值架构**：一位成员一直在实验复数值架构，试图构建一个每个像素都是 token 的视觉模型，并使用 **2D DFT** 代替 **attention** 进行 token 混合，类似于 **FNet**。
   - 尽管取得了一些成功，但在**更深的网络中遇到了逐渐显现的问题**，不过较浅的网络似乎训练得还可以。
- **复数值模型中的缩放问题**：该复数值模型无论参数量多少（从 11k 到 400k），在 **CIFAR-100** 上的准确率始终保持在 **30%** 左右。
   - 对复数值的*正确*处理提高了性能，一个 65k 的复数模型表现略优于之前 session 中 400k 的实数模型。


  

---

### **LAION ▷ #[resources](https://discord.com/channels/823813159592001537/991938328763056168/1260263446297968750)** (1 条消息): 

> - `Image Diffusion Models Repository`
> - `GitHub Repo for Image Diffusion`
> - `Educational Codes for Image Diffusion` 


- **Image Diffusion 模型大师级入门**：一名成员宣布了一个 [GitHub 仓库](https://github.com/swookey-thinky/mindiffusion)，其中包含关于 Image Diffusion 模型的课程，这些模型可以在最小化的 GPU 上使用小数据集进行训练。
   - *重点在于通过配套的教程视频和 Colab 链接，结合清晰的演示代码，学习每篇论文的内部运行机制*。
- **实用的代码引导式 Image Diffusion 学习包**：该仓库为 Image Diffusion 模型提供了一个实用的、以代码为导向的教育资源包。
   - 鼓励贡献者提供反馈以改进资源。



**提到的链接**：<a href="https://github.com/swookey-thinky/mindiffusion">GitHub - swookey-thinky/mindiffusion: Repository of lessons exploring image diffusion models, focused on understanding and education.</a>：探索 Image Diffusion 模型的课程仓库，专注于理解与教育。- swookey-thinky/mindiffusion

  

---



### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1260052940362743891)** (47 条消息🔥): 

> - `Quota Exceeded Issue`
> - `Image Viewing Issues`
> - `Dolphin 2.9 Mixstral on OpenRouter in LangChain`
> - `Mistralai Mixtral v0.1 Error`
> - `LLM Applications for Language Translation` 


- **OpenRouter API 配额超限**：一名用户遇到了配额超限错误：'Quota exceeded for aiplatform.googleapis.com/generate_content_requests_per_minute_per_project_per_base_model with base model: gemini-1.5-flash'。
   - *这可能是 Google 对 OpenRouter 施加的限制。*
- **OpenRouter 上的图片查看问题**：一名用户报告了在 gpt-4o、claude-3.5 和 firellava13b 等多种模型中，模型返回 None 导致无法查看图片的问题。
   - 另一名用户确认这些图片在他们那里运行良好，表明该问题可能并不普遍。
- **在 LangChain 中集成 Dolphin 2.9 Mixstral 的挑战**：一名用户尝试在 LangChain 中将 OpenRouter 上的 Dolphin 2.9 Mixstral 作为 tool calling Agent 使用，但面临一些问题。
- **Mistralai Mixtral v0.1 不支持 JSON 模式**：一名用户遇到了错误 'mistralai/Mixtral-8x22B-Instruct-v0.1 is not supported for JSON mode/function calling'，并指出这种情况偶尔发生。
   - 经过测试，用户确定 Together 是导致该问题的提供商。
- **用于语言翻译的 LLM 应用偏好**：用户讨论了 LLM 在翻译方面的有效性，并将其与专门的翻译模型进行了比较。
   - 一名用户强调，现代 LLM 使用 decoder-only 模型，在翻译任务中可能不如真正的 encoder/decoder Transformer 可靠。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/activity">Activity | OpenRouter</a>：查看你在 OpenRouter 上使用模型的情况。</li><li><a href="https://openrouter.ai/docs/provider-routing#custom-routing">Provider Routing | OpenRouter</a>：跨多个提供商进行请求路由
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1259956345801805946)** (27 条消息🔥): 

> - `Claude Contest Reminder`
> - `Nuance in Speech Models`
> - `AI Math Competition Success`
> - `Supermaven's Babble Upgrade`
> - `Lillian Weng's Blog on Hallucinations`

- **Claude 竞赛提醒**：一位成员提醒社区关于 Build with Claude 竞赛的消息，奖金包含 3 万美元的 Anthropic API 额度，竞赛将于两天后结束。[Alex Albert 的帖子](https://x.com/alexalbert__/status/1810376544734556540) 提供了更多细节。
- **语音模型细微差别讨论**：一个帖子强调了 **GPT-4o** 精雕细琢的轮询式（turn-based）模型与 **Moshi** 未经雕琢的全双工（full-duplex）模型之间的差异。讨论源于 [JulianSlzr](https://x.com/julianslzr/status/1810303916686577858?s=46&t=PW8PiFwluc0tdmv2tOMdEg) 和 Andrej Karpathy 分享的经验。
- **AI 在数学奥林匹克竞赛中取得成功**：**Thom Wolf** 赞扬了 AI 数学奥林匹克竞赛（AI Math Olympiad），其中 **Numina** 与 Hugging Face 的合作取得了令人印象深刻的成绩。有关该活动及其意义的详细信息请参见 [Thom Wolf 的推文串](https://x.com/Thom_Wolf/status/1809895886899585164)。
- **Supermaven 发布 Babble**：Supermaven 宣布部署 **Babble**，这是一个拥有 100 万 token 上下文窗口的新模型，比其之前的模型大 2.5 倍。在 [SupermavenAI 的推文](https://x.com/SupermavenAI/status/1808256013788676438) 中了解更多关于此次升级的信息。
- **Lillian Weng 讨论 LLM 中的幻觉**：**Lillian Weng** 的博文深入探讨了大语言模型中幻觉（hallucinations）的类型和原因。在此阅读完整讨论 [链接](https://lilianweng.github.io/posts/2024-07-07-hallucination)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://turbopuffer.com/blog/turbopuffer">turbopuffer: 基于对象存储的快速搜索</a>: turbopuffer 是一个构建在对象存储之上的向量数据库，这意味着成本降低 10-100 倍、按需付费以及极高的可扩展性。</li><li><a href="https://lilianweng.github.io/posts/2024-07-07-hallucination/">LLM 中的外源性幻觉 (Extrinsic Hallucinations)</a>: 大型语言模型中的幻觉通常指模型生成不忠实、虚假、不一致或无意义的内容。作为一个术语，幻觉在某种程度上被泛化为...</li><li><a href="https://x.com/SupermavenAI/status/1808256013788676438">来自 Supermaven (@SupermavenAI) 的推文</a>: 我们训练了 Babble，这是一个具有 100 万 token 上下文窗口的新模型。Babble 比之前的 Supermaven 模型大 2.5 倍，并将我们的上下文长度从 30 万升级到 100 万 token...</li><li><a href="https://x.com/alexalbert__/status/1810748433273344469?s=46">来自 Alex Albert (@alexalbert__) 的推文</a>: 1) Prompt 生成器。输入任务描述，Claude 3.5 Sonnet 将为你把任务描述转换为高质量的 Prompt。彻底解决了 Prompt 编写时的“空白页”难题。</li><li><a href="https://x.com/atroyn/status/1810717585442492686?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 anton (𝔴𝔞𝔯𝔱𝔦𝔪𝔢) (@atroyn) 的推文</a>: 今天很高兴向大家展示 Chroma 的下一份技术报告，即我们在 AI 应用背景下对分块策略 (chunking strategies) 对检索性能影响的评估。@brandonstarxel @tr...</li><li><a href="https://x.com/pathak2206/status/1810769359591330201?s=46">来自 Deepak Pathak (@pathak2206) 的推文</a>: 激动地宣布 @SkildAI！在过去的一年里，@gupta_abhinav_ 和 I 一直在与我们的顶尖团队合作，构建一个立足于物理世界的 AI 基础模型。今天，我们正迈出...</li><li><a href="https://x.com/Thom_Wolf/status/1809895886899585164">来自 Thomas Wolf (@Thom_Wolf) 的推文</a>: 上周发生了一场令人印象深刻的 AI 竞赛，许多人在 AI 世界的喧嚣中错过了它。我恰好认识几位参赛者，所以让我作为...来给你们讲讲这个故事。</li><li><a href="https://x.com/elonmusk/status/1810727394631950752?s=46&t=PW8PiFwluc0tdmv2tOMdEg">来自 Elon Musk (@elonmusk) 的推文</a>: xAI 从 Oracle 租用了 2.4 万张 H100，Grok 2 正是在这些卡上训练的。Grok 2 正在进行微调和 Bug 修复。可能下个月准备发布。xAI 正在自行构建 10 万张 H100 的系统...</li><li><a href="https://x.com/alexalbert__/status/1810376544734556540">来自 Alex Albert (@alexalbert__) 的推文</a>: 距离参加比赛还有两天！引用 Alex Albert (@alexalbert__) 的话：宣布 2024 年 6 月的 Build with Claude 竞赛。我们将发放价值 3 万美元的 Anthropic API 额度。你只需要...</li><li><a href="https://x.com/julianslzr/status/1810303916686577858?s=46&t=PW8PiFwluc0tdmv2tOMdEg">来自 Julian Salazar (@JulianSlzr) 的推文</a>: 像 @karpathy 这样的观点会让细微差别丢失（也被揭示！）。我主张：- GPT-4o 语音 (@openai) 是一个精致的端到端 *轮询式 (turn-based)* 模型 - Moshi (@kyutai_labs) 是一个粗糙的端到端 *全双工 (full-duplex)* 模型...</li><li><a href="https://the-decoder.com/sensetime-unveils-sensenova-5o-chinas-first-real-time-multimodal-ai-model-to-rival-gpt-4o/?utm_source=substack&utm_medium=email">商汤科技发布日日新 (SenseNova) 5o，中国首个可与 GPT-4o 媲美的实时多模态 AI 模型</a>: 中国 AI 公司商汤科技在世界人工智能大会上介绍了其新的多模态 AI 模型日日新 5o 以及改进后的语言模型日日新 5.5。</li><li><a href="https://x.com/xiaolonw/status/1810387662060269668">来自 Xiaolong Wang (@xiaolonw) 的推文</a>: 不敢相信这终于发生了！在过去的 1.5 年里，我们一直在开发一种新的 LLM 架构，具有线性复杂度和极具表现力的隐藏状态，用于长上下文建模。以下...
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1259994030251577455)** (16 条消息🔥): 

> - `LLMWhisperer PDF 提取`
> - `LangChain 中的多 Agent 聊天机器人问题`
> - `Crawlee for Python 发布`
> - `使用 RAG 对 PDF 文档进行问答`
> - `LangChain 中的 ConversationSummaryMemory` 


- **LLMWhisperer 像专家一样处理复杂的 PDF**：一位用户分享说 **LLMWhisperer** 能有效地解析复杂的 PDF，允许使用 LLM 解析每一页并最终合并 JSON 以实现全面的文档解析。他们建议利用 LangChain 中的 [Pydantic 或 zod schema](https://www.youtube.com/watch?v=dC7EhnEIdDA) 来实现这一点。
- **解决 LangChain 中的多 Agent 聊天机器人问题**：有人询问如何解决 **LangChain** 中的多 Agent 聊天机器人问题，解决方案包括理解 Tools、Agents 和 LLMs，使用 LangSmith，选择合适的聊天模型，并参考社区支持。详细说明和来源可在 LangChain 官方 [JavaScript 文档](https://js.langchain.com/v0.2/docs/how_to/agent_executor) 中找到。
- **Crawlee for Python 发布公告**：Apify 的开发者社区经理宣布了 **Crawlee for Python**，强调了其功能，如使用 Playwright 的 HTTP 和无头浏览器的统一接口，以及自动扩展和会话管理。他们邀请用户在 [GitHub](https://github.com/apify/crawlee-python) 上查看并支持 [Product Hunt](https://www.producthunt.com/posts/crawlee-for-python) 上的发布。
- **问答链中 RAG 的最佳实践**：一位用户询问如何将 RAG 组件集成到 LangChain 现有的问答链中，并保持历史记录和处理上下文长度。讨论指向为 RAG 创建一个新的链，以确保原始链不会修剪加载的 PDF 文档。
- **适用于多人的 ConversationSummaryMemory**：一位用户询问 **LangChain 的 ConversationSummaryMemory** 是否支持多人，并寻求关于总结大段对话的建议。该话题仍对社区进一步的输入和解决方案开放。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/asimdotshrestha/status/1810720478111371581">来自 Asim Shrestha (@asimdotshrestha) 的推文</a>: 很高兴能更广泛地分享我们在 @ReworkdAI ⚡️ 所做的工作。在过去的一年里，我们全力投入到构建下一代 Web Agent 中。它们已经在生产环境中上线...</li><li><a href="https://chat.whatsapp.com/F9naq8o3Cv14Hi1uZcxpYV">国际聊天群组 &#x1f495;</a>: WhatsApp 群组邀请</li><li><a href="https://www.youtube.com/watch?v=dC7EhnEIdDA">使用 LLMWhisperer 提取 PDF 复选框</a>: 这是一个演示，展示了如何使用 LLMWhisperer 处理 PDF 表单元素（如复选框和单选按钮），LLMWhisperer 是一种文本提取服务...</li><li><a href="https://github.com/apify/crawlee-python">GitHub - apify/crawlee-python: Crawlee——一个用于 Python 的网页抓取和浏览器自动化库，用于构建可靠的爬虫。为 AI、LLMs、RAG 或 GPTs 提取数据。从网站下载 HTML、PDF、JPG、PNG 和其他文件。支持 BeautifulSoup、Playwright 和原始 HTTP。支持有头和无头模式。支持代理轮换。</a>: Crawlee——一个用于 Python 的网页抓取和浏览器自动化库，用于构建可靠的爬虫。为 AI、LLMs、RAG 或 GPTs 提取数据。从网站下载 HTML、PDF、JPG、PNG 和其他文件...</li><li><a href="https://www.producthunt.com/posts/crawlee-for-python"> Crawlee for Python - 在 Python 中构建可靠的抓取工具 | Product Hunt</a>: 我们正在发布 Crawlee for Python，这是一个用于网页抓取和浏览器自动化的开源库。快速抓取数据、存储数据并避免被封锁、无头浏览器和智能代理轮换...</li><li><a href="https://js.langchain.com/v0.2/docs/tutorials/chatbot/#langsmith>)">构建聊天机器人 | 🦜️🔗 Langchain</a>: 概览</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/agent_executor/#using-language-models>)">如何使用旧版 LangChain Agents (AgentExecutor) | 🦜️🔗 Langchain</a>: 本指南假设你熟悉以下概念：</li><li><a href="https://github.com/langchain-ai/langchain/issues/7597>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建一个账号来为 langchain-ai/langchain 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1259965012626378772)** (5 messages): 

> - `Llamapp`
> - `Slack Bot Agent 指南`
> - `Rubik's AI Pro Beta 测试`
> - `RAG 文章`
> - `用于网页数据提取的 LLM` 


- **认识 Llamapp：本地检索增强生成器 (RAG)**：[Llamapp](https://github.com/rajatasusual/llamapp) 是一个本地运行的检索增强生成器 (RAG)，它结合了文档检索和语言模型生成，以提供准确且具有上下文相关性的响应。
   - 它使用自定义检索器和 Reciprocal Ranking Fusion 来提供具有说服力的文档集，并确保 LLM 遵循事实来源 (source of truth)。
- **使用 LangChain 和 ChatGPT 创建 Slack Bot Agent 指南**：一份 [指南](https://git.new/slack-bot-agent) 提供了详细步骤，用于创建一个利用 Composio、LangChain、OpenAI 和 ChatGPT 的 Slack Bot Agent，以便在每次创建 PR 时对其进行评审。
   - 该指南演示了如何使用各种框架来自动化 PR 评审流程。
- **Rubik's AI Pro：成为 Beta 测试员**：邀请 Beta 测试一款高级研究助手和搜索引擎，支持 Claude 3 Opus 和 GPT-4o 等模型，使用代码 `RUBIX` 可获得 2 个月的免费高级访问权限。
   - [Rubik's AI Pro](https://rubiks.ai/) 提供对尖端模型和在线引用的访问，并提供为期两个月的高级试用以换取反馈。
- **关于在本地运行 RAG 的新文章**：一名成员分享了一篇关于在本地运行 RAG 的 [文章](https://www.linkedin.com/pulse/tame-artificial-intelligence-from-your-laptop-rajat-kumar-pfnae?utm_source=share&utm_medium=member_ios&utm_campaign=share_via)，内容包括自定义 FRR、混合检索器和自定义加载器。
   - 本文旨在提供有关在笔记本电脑上运行检索增强生成器的见解，并鼓励社区反馈。
- **发布用于网页数据提取的 LLM**：分享了一个专注于自动网页数据提取的 LLM 新发布，并寻求社区支持。
   - 查看在 [X](https://x.com/asimdotshrestha/status/1810720478111371581) 和 [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7216488222560768001) 上的发布公告。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://git.new/slack-bot-agent">用于评审 PR 的 Slack Bot Agent</a>: 该指南提供了详细步骤，用于创建一个利用 Agent 框架、OpenAI 和 ChatGPT 的 Slack Bot Agent，以便在每次创建 PR 时对其进行评审。</li><li><a href="https://x.com/asimdotshrestha/status/1810720478111371581">来自 Asim Shrestha (@asimdotshrestha) 的推文</a>: 很高兴能更广泛地分享我们在 @ReworkdAI 正在做的工作 ⚡️ 过去的一年里，我们全力投入于构建下一代网页 Agent。它们已经在生产环境中上线...</li><li><a href="https://github.com/rajatasusual/llamapp">GitHub - rajatasusual/llamapp: 一个完全在本地运行的检索增强生成器 (RAG)，结合了文档检索和语言模型生成，以提供准确且具有上下文相关性的响应。基于 @Langchain-ai 构建</a>: 一个完全在本地运行的检索增强生成器 (RAG)，结合了文档检索和语言模型生成，以提供准确且具有上下文相关性的响应。基于...</li><li><a href="https://rubiks.ai/">Rubik's AI - AI 研究助手 & 搜索引擎</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1259976413054767145)** (1 messages): 

> - `Slack Bot Agent`
> - `Composio 和 LangChain`
> - `使用 OpenAI 和 ChatGPT 自动化 PR 评审` 


- **创建 Slack Bot Agent 的指南**：一名成员分享了一份关于创建 Slack Bot Agent 的 [指南](https://git.new/slack-bot-agent)，该 Agent 利用 Composio、LangChain、OpenAI 和 ChatGPT 来自动化 PR 评审。
   - 该指南包括设置 Agent 的详细步骤，并提到了多种框架和工具的使用。
- **使用 ChatGPT 自动化 PR 评审**：该指南提供了详细步骤，用于创建一个利用 **Agent 框架**、**OpenAI** 和 **ChatGPT** 的 Slack Bot Agent，以便在每次创建 PR 时对其进行评审。



**提到的链接**: <a href="https://git.new/slack-bot-agent">用于评审 PR 的 Slack Bot Agent</a>: 该指南提供了详细步骤，用于创建一个利用 Agent 框架、OpenAI 和 ChatGPT 的 Slack Bot Agent，以便在每次创建 PR 时对其进行评审。

  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1259997793888501830)** (20 messages🔥): 

> - `OI 配合代码示例执行`
> - `位置不当的自我广告`
> - `在本地视觉模式下使用 '--model i'`
> - `'i model' 功能`
> - `Qwen 2 7b 问题` 


- **OI 配合代码示例无缝执行**：一位成员提到，通过添加 [代码指令示例](https://link.to/examples)，**OI 执行得非常完美**，类似于 **assistant.py** 处理不同技能指令的方式。
- **Qwen 2 7B 模型打印随机的 '@'**：一位成员指出，**Qwen 2 7B 模型**处理 128k 上下文的能力令人印象深刻，但会在行内随意打印随机的 '@'，导致代码损坏。
- **在本地视觉模式下使用 '--model i'**：讨论集中在 **'--model i'** 是否可以在 **本地视觉模式** 下运行，以及它是否是多模态的。
- **关于在 OS 模式下使用 GROQ 的说明**：有人询问关于在 **OS 模式下使用 GROQ** 的问题，以及是否需要多模态模型。
- **使用 Open Interpreter 解释屏幕坐标**：一位成员寻求关于 **Open Interpreter** 如何获取屏幕坐标的澄清。



**提及的链接**：<a href="https://github.com/OpenInterpreter/open-interpreter/blob/main/interpreter/terminal_interface/profiles/defaults/os.py">open-interpreter/interpreter/terminal_interface/profiles/defaults/os.py at main · OpenInterpreter/open-interpreter</a>：计算机的自然语言界面。通过在 GitHub 上创建账户，为 OpenInterpreter/open-interpreter 的开发做出贡献。

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1260198681202921545)** (9 messages🔥): 

> - `NV=1 支持`
> - `与 Ampere 之前架构的兼容性`
> - `George Hotz 关于兼容性的评论`
> - `社区对旧架构的潜在贡献`
> - `基于 GSP 固件的代际` 


- **NV=1 仅在 Ampere 及更新架构上支持**：一位成员询问 **NV=1** 是否仅支持 **Ampere** 或更新架构，另一位成员确认它确实支持 Ampere 和 Ada 架构。
- **Turing 卡对 NV=1 的兼容性**：一位用户讨论了可能设置 **Linux** 来尝试 NV=1，但担心与 **Turing 卡** 的兼容性。
   - 另一位成员提到，使其兼容旧架构的优先级较低，可能需要社区贡献。
- **George Hotz 确认 Turing 代架构的兼容性**：**George Hotz** 澄清说，**Turing 代**（例如 2070, 2080）也支持 NV=1，因为它们包含在 [基于 GSP 固件的代际列表](https://github.com/NVIDIA/open-gpu-kernel-modules) 中。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1260096678120128543)** (9 messages🔥): 

> - `学习 tinygrad 的推荐视频课程`
> - `WSL2 上的 NV=1 问题`
> - `WSL2 上的 CUDA 兼容性`
> - `WSL2 上的 NVIDIA 开源 GPU 内核模块` 


- **通过视频课程学习 tinygrad**：一位成员征求学习 tinygrad 的视频课程推荐，并被建议观看 [Karpathy 的 Transformer 视频](https://www.youtube.com/watch?v=2-BK_E6r4P8)，尽管它是用 PyTorch 编写的，但它能更好地吸引观众。
   - "非常感谢 Tobi.. 我会尝试的.. 这可能是实现时探索文档的好方法。" - *ghost22111*
- **WSL2 上的 NV=1 问题**：一位成员在 **WSL2** 上运行 **NV=1** 时遇到问题，发现缺少 `dev/nvidiactl`，一些建议指向了 `dxg`。
   - 另一位成员指出可能需要 [NVIDIA 的开源 GPU 内核模块](https://github.com/NVIDIA/open-gpu-kernel-modules)，但不确定 **Microsoft** 在 WSL2 中捆绑了什么。



**提及的链接**：<a href="https://github.com/NVIDIA/open-gpu-kernel-modules">GitHub - NVIDIA/open-gpu-kernel-modules: NVIDIA Linux open GPU kernel module source</a>：NVIDIA Linux 开源 GPU 内核模块源码。通过在 GitHub 上创建账户，为 NVIDIA/open-gpu-kernel-modules 的开发做出贡献。

  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1260114633071525970)** (7 条消息): 

> - `GitHub Copilot lawsuit`
> - `Developer concerns on Copilot`
> - `Legal implications for Microsoft and OpenAI` 


- **GitHub Copilot 诉讼范围缩小**：开发者指控 [GitHub Copilot](https://www.theregister.com/2024/07/08/github_copilot_dmca/) 非法复制其代码的说法大部分已被驳回，使得工程师们在针对 GitHub、Microsoft 和 OpenAI 的诉讼中仅剩下两项指控。
   - 这项于 2022 年 11 月提起的[集体诉讼](https://www.theregister.com/2024/01/12/github_copilot_copyright_case_narrowed/)认为，Copilot 在未给予适当署名的情况下使用开源软件进行训练，违反了知识产权。
- **开发者对 Copilot 问题的现场反应**：当被当面问及对 Copilot 持续存在的担忧时，工程师幽默地肯定了这一回应：*“对律师来说已经足够好了”*。



**提到的链接**：<a href="https://www.theregister.com/2024/07/08/github_copilot_dmca/">法官驳回 GitHub Copilot 诉讼中的 DMCA 版权指控</a>：少数开发者对抗雷德蒙德（Redmond）的强大势力——你觉得谁会赢？

  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1260339344435052674)** (4 条消息): 

> - `Control Vector`
> - `Steering Vector`
> - `Concept Vectors`
> - `Feature Clamping`
> - `Feature Steering` 


- **Concept Vectors 被讨论为 Steering Vectors 的同义词**：一位成员询问 **Control Vector**、**Steering Vector** 和 **Concept Vectors** 是否基本上是同义词，而 **Steering Vectors** 只是 Control Vectors 在语言模型中的应用。
   - 另一位成员确认前两个术语被用于控制最后一个术语。
- **Feature Clamping 与 Feature Steering 的区别**：讨论还涉及了在 **RepEng** 的 **Feature Steering** 工具箱中，**Feature Clamping** 是如何既有区别又相互关联的。


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1260271285125251222)** (5 条消息): 

> - `Google Flame paper issue`
> - `AI bill controversy`
> - `Training on test data` 


- **Google Flame 分数因问题被移除**：Google 负责“Google Flame”分数和论文的团队由于“某些问题”移除了相关分数。
   - 存在一些幽默的怀疑，猜测他们是否“在测试数据上进行了训练”。
- **AI 法案引发争议**：[Twitter 链接](https://x.com/hlntnr/status/1810713658860912914)显示 **Scott Wiener** 指责 **a16z** 和 **Y Combinator** 对 **加州 SB 1047 AI 法案** 发表了“不准确且具有煽动性的言论”。
   - 他们一直在网上大声反对这项法案，引起了 AI 社区的极大关注。



**提到的链接**：<a href="https://x.com/hlntnr/status/1810713658860912914">Helen Toner (@hlntnr) 的推文</a>：@Scott_Wiener 开火了 👀 图片是上周的一封信，Wiener（加州 SB 1047 AI 法案背后的州参议员）直接点名 a16z 和 Y Combinator 的“不准确、煽动性...”

  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1260203695459667978)** (2 条消息): 

> - `Credit Issues`
> - `Member Response Time` 


- **成员填写了表格但未收到积分**：*一位成员报告说他们填写了必要的表格，但尚未收到积分。*
- **成员请求响应时间**：*一位成员请求其他人在未来 72 小时内回复他们的私信，以便核对某些细节。*


  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1260196762627346463)** (7 messages): 

> - `Multi GPU Training Issues`
> - `Accelerate Configuration`
> - `Batch Size Impact`
> - `Performance Expectations`
> - `Debugging Techniques` 


- **多 GPU 设置性能缓慢令人失望**：一位成员表示，他们在配置了 **H100 GPU x 6** 的多 GPU 环境下，训练速度比预期慢了 **10 倍**。
   - 另一位成员建议调整 **batch size**，并参考 [Hugging Face 的故障排除指南](https://huggingface.co/docs/accelerate/basic_tutorials/troubleshooting)。
- **为多 GPU 调整 Batch Size**：讨论强调了在测试多 GPU 的 1:1 加速比时，调整 **batch size** 的重要性。
   - 成员们建议提供吞吐量数据和源代码，以便根据文档中的最佳实践进行进一步的诊断和优化。
- **多 GPU 的现实性能预期**：成员们对速度提升的预期提出了质疑，指出由于多 GPU 设置中的通信开销，**10 倍 / 1:1 的加速比**是不现实的。
   - 如果吞吐量达到最优，从 1 个 GPU 增加到 8 个 GPU 的现实速度提升估计在 **6-7 倍** 左右。



**提及的链接**：<a href="https://huggingface.co/docs/accelerate/basic_tutorials/troubleshooting">Troubleshoot</a>：未找到描述

  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1260018653538287656)** (7 messages): 

> - `Teaching and Learning Platform`
> - `CommandR RAG-Optimized Features`
> - `Dark Mode Release`
> - `Enterprise Features Adaptation` 


- **教师为学习平台探索 CommandR**：一位公立学校教师正在开发一个**教学与学习平台**，并因其 **RAG 优化特性**而考虑使用 [CommandR](https://link.to.commandr)。
   - 社区成员对这一想法表示欢迎和兴奋，并表示愿意在需要时提供帮助。
- **深色模式即将发布**：新功能**深色模式**正在开发中，将作为针对企业的大版本更新的一部分发布。
   - 该功能也可能适配到 **Coral** 等免费平台，以惠及更多用户。


  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/)** (1 messages): 

competent: 同意 👍
  

---



### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1259995314388205568)** (4 messages): 

> - `Performance penalty in llama.cpp`
> - `Benchmark suite upgrade issues` 


- **NVIDIA GPU 上 0.8.8 到 0.8.9 版本的性能下降**：一位成员观察到在 **NVIDIA GPU** 上，llama.cpp 的 0.8.8 和 0.8.9 版本之间存在 **~25% 的性能损失**，并询问这是否为已知问题。
   - 性能下降非常明显，**3090** 在旧版本上的表现与 **3060** 相似。
- **基准测试套件升级影响性能**：一位正在编写基准测试套件的成员在升级 llamafile 版本后注意到了这个问题。
   - 另一位成员回应称，他们最近没有进行任何可能影响性能的更改。


  

---



### **AI Stack Devs (Yoko Li) ▷ #[app-showcase](https://discord.com/channels/1122748573000409160/1122748840819306598/)** (1 messages): 

__n2k: ^ 那些西瓜是我做的 🍉 😄
  

---


### **AI Stack Devs (Yoko Li) ▷ #[events](https://discord.com/channels/1122748573000409160/1131651713204498583/1260037561338957908)** (1 messages): 

> - `Book to Game Jam`
> - `Rosebud AI`
> - `Puzzle games`
> - `Rhythm games`
> - `Text-based adventures` 


- **Rosebud AI Jam 将书籍转化为游戏**：在 Rosebud AI Book to Game Jam 期间，参与者受 **Lewis Carroll, China Miéville** 和 **R.L. Stine** 等作家的启发，创作了益智游戏、节奏游戏和文字冒险游戏。
   - 获胜者将于 **太平洋标准时间 (PST) 7 月 10 日星期三上午 11:30** 在 Rosebud AI 服务器中公布。点击[此处](https://x.com/Rosebud_AI/status/1810464373363585186)查看参赛作品。
- **Rosebud AI 展示创意游戏作品**：**Rosebud AI** 重点展示了在 Book to Game Jam 期间提交的作品，强调了作品的创意和多样性。
   - 参赛作品将文学作品转化为各种游戏类型，展示了 **Phaser** 和 **AI 技术**的能力。



**提及的链接**：<a href="https://x.com/Rosebud_AI/status/1810464373363585186">来自 Rosie @ Rosebud AI 🌹 (@Rosebud_AI) 的推文</a>：利用 AI 将书籍变为游戏 🌹 我们最近的 Jam 活动让开发者使用 Rosebud AI 从文学作品中创作游戏，这就是结果！获胜者将于本周三 7 月 10 日 11:30 公布...

  

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1260325557112016996)** (1 messages): 

> - `KAN authors`
> - `alphaXiv forum`
> - `arXiv paper discussion` 


- **KAN 作者在 alphaXiv 论坛互动**：**KAN** 论文的作者本周正在 [alphaXiv 论坛](https://alphaxiv.org/abs/2404.19756v4) 积极回答问题。
   - 讨论集中在他们最近发表的 **arXiv** 论文的核心观点上。
- **alphaXiv 上关于 KAN 论文的讨论**：alphaXiv 论坛目前正在就 **KAN** 论文进行热烈讨论，作者正在回答社区提问。
   - 参与者正在深入探讨论文中概述的 **KAN** 方法论的技术细节。



**Link mentioned**: <a href="https://alphaxiv.org/abs/2404.19756v4">alphaXiv</a>: 未找到描述

  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1260145820963966997)** (1 messages): 

> - `Information Retrieval`
> - `Recommendations`
> - `Podcast Guests`
> - `Outreach` 


- **播客制作人寻求信息检索专家**：一位成员提到，他们正计划为关于信息检索（**Information Retrieval**）和推荐系统（**Recommendations**）的播客系列邀请 **Cohere**、**Zilliz** 和 **Doug Turnbull** 录制节目。
- **征求更多专家名单**：该成员还征求了其他信息检索和推荐系统领域专家的建议，以便为播客进行采访。


  

---



### **LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/)** (1 messages): 

frandecam: 有人知道 **Anthropic** 是否有类似 **OpenAI** 10K 额度或类似的计划吗？
  

---



---



---



---



---



{% else %}


> 完整的逐频道细分内容已针对电子邮件进行了截断。
> 
> 如果您想查看完整细分，请访问此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}