---
companies:
- qdrant
- cohere
- stripe
- anthropic
- hugging-face
- stablequan_ai
date: '2024-07-06T02:25:00.011918Z'
description: '**Qdrant** 试图通过一种名为“BM42”的新方法来取代 BM25 和 SPLADE。该方法结合了 Transformer 注意力和全集合统计信息，旨在用于语义和关键词搜索，但其在
  Quora 数据集上的评估被发现存在缺陷。来自 **Cohere** 的 **Nils Reimers** 在更优的数据集上重新测试了 BM42，发现其表现不佳。Qdrant
  承认了这些错误，但在对比中仍使用了次优的 BM25 实现。这突显了在搜索模型声明中，数据集选择和评估合理性检查（sanity checks）的重要性。


  此外，**Stripe** 因 AI/ML 模型故障导致账户和支付问题而面临批评，引发了寻求替代方案的呼声。**Anthropic** 透露 **Claude
  3.5 Sonnet** 会使用后端标签抑制部分回答内容，这引发了广泛争论。**Gemma 2** 模型的优化使得微调速度提升了 2 倍，内存占用减少了 63%，并支持更长的上下文窗口，能够在消费级
  GPU 上运行高达 34B 参数的模型。**nanoLLaVA-1.5** 作为一款紧凑的 1B 参数视觉模型正式发布，并带来了显著改进。'
id: 47aa9d67-913f-4412-ae5d-caa6cd205923
models:
- claude-3.5-sonnet
- gemma-2
- nano-llava-1.5
original_slug: ainews-qdrants-bm42
people:
- nils-reimers
- jeremyphoward
- hamelhusain
- rohanpaul_ai
title: Qdrant 的 BM42：“请不要相信我们”
topics:
- semantic-search
- benchmarking
- dataset-quality
- model-evaluation
- model-optimization
- vision
- fine-tuning
- context-windows
---

<!-- buttondown-editor-mode: plaintext -->**同行评审就是你所需要的一切。**

> 2024年7月4日至7月5日的 AI 新闻。我们为你检查了 7 个 subreddit、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord（**418** 个频道，**3772** 条消息）。预计节省阅读时间（按每分钟 200 字计算）：**429 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论，并尝试 [Smol Talk](https://smol.fly.dev)！

Qdrant 被广泛认为是 [OpenAI 首选的向量数据库](https://news.ycombinator.com/item?id=38280859)，在 7 月 4 日假期期间，他们发布了一些大胆的声明，声称要取代久负盛名的 [BM25](https://en.wikipedia.org/wiki/Okapi_BM25)（甚至是更现代的 [SPLADE](https://arxiv.org/abs/2109.10086)），并试图创造“BM42”这一术语：

 
![image.png](https://assets.buttondown.email/images/9c643b9a-85fb-4279-b396-abcc4869e931.png?w=960&fit=max)
 

该方案旨在通过结合用于词重要性评分的 Transformer 注意力机制与诸如 IDF 之类的全集合统计数据，来解决语义 + 关键词搜索的问题，并声称在所有用例中都具有优势：

 
![image.png](https://assets.buttondown.email/images/fd7dd267-fa6c-46a5-abe3-1a0d757088c5.png?w=960&fit=max)
 

只有一个问题……结果。来自 Vespa（竞争对手）的 Jo Bergum [指出](https://x.com/jobergum/status/1809157587612336402)，选择 Quora（一个“寻找相似重复”问题的数据集，而非 Q&A 检索数据集）作为数据集非常奇怪，而且如果你了解该数据集，就会发现其评估结果显然是错误的：

 
![image.png](https://assets.buttondown.email/images/bebb5c7c-192a-41a3-bb91-d54c7118238a.png?w=960&fit=max)
 

具体来说，Quora 数据集[每个查询只有约 1.6 个数据点](https://github.com/beir-cellar/beir)，因此他们声称每 10 个结果中有超过 4 个（precision@10）的数值显然是错误的。

[Cohere 的 Nils Reimers](https://x.com/Nils_Reimers/status/1809334134088622217) 采用了 BM42 并在金融、生物医学和 Wikipedia 领域更好的数据集上进行了重新运行，遗憾的是，BM42 在所有方面都表现不佳：

 
![image.png](https://assets.buttondown.email/images/dc98cac6-200f-427f-adfc-19a6c8377c33.png?w=960&fit=max)
  

就 Qdrant 而言，他们已经[回应并承认](https://x.com/qdrant_engine/status/1809291686625046816)了这些修正，并[发布了修正方案](https://x.com/Nils_Reimers/status/1809299249017856379)……除了仍然奇怪地运行着一个比其他人预期得分更低、且恰好比 BM42 表现更差的 BM25 实现。

对 Qdrant 来说这很不幸，但对我们其他人来说，这只是一次关于了解数据和对评估进行常识检查的速成课。最后，正如在公关（PR）中尤其是 AI 领域一贯如此：**非凡的主张需要非凡的证据。**


> **元注释**：如果你一直想定制自己版本的 AI News，我们现在已经[预览](https://x.com/Smol_AI/status/1809412102693818579)了一个简陋的早期版本 Smol Talk，你可以通过这里访问：https://smol.fly.dev

---

{% if medium == 'web' %}

**目录**

[TOC]

{% else %}

**目录**和**频道摘要**已移至此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成。

**Stripe 问题与替代方案**

- **Stripe 账户问题**：[@HamelHusain](https://twitter.com/HamelHusain/status/1808850347169100261) 指出，尽管没有退款请求，Stripe 仍以“无尽的官僚程序”为由“扣押了我所有的钱”。[@jeremyphoward](https://twitter.com/jeremyphoward/status/1808948204358283438) 称 Stripe 因“AI/ML 模型故障”而注销账户的行为是“可耻的”。
- **申诉 Stripe 的决定**：[@HamelHusain](https://twitter.com/HamelHusain/status/1808861338917351893) 对 Stripe 的拒绝提出了申诉，但在 5 分钟内就被驳回，Stripe 继续“扣押着数千美元”。
- **Stripe 的替代方案**：[@HamelHusain](https://twitter.com/HamelHusain/status/1808906891957055713) 指出需要一个“备份计划”，因为“被 AI/ML 误报抓到真的很糟糕”。[@virattt](https://twitter.com/virattt/status/1808859416491344040) 在看到许多关于此类问题的帖子后，对使用 Stripe 表示谨慎。

**AI 与 LLM 进展**

- **Anthropic Constitutional AI**：[@Anthropic](https://twitter.com/Anthropic/status/1808755146190446667) 指出 Claude 3.5 Sonnet 会使用 "antThinking" 标签隐藏部分回答内容，这些标签在后端会被移除，但一些人不同意这种隐藏做法。
- **Gemma 2 模型优化**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1808858018253074591) 分享了使用 UnslothAI 库可以使 Gemma 2 的微调速度**提升 2 倍且显存占用减少 63%**，允许比 HF+FA2 **长 3-5 倍的上下文**。它可以在单个消费级 GPU 上运行高达 34B 的模型。
- **nanoLLaVA-1.5 视觉模型**：[@stablequan_ai](https://twitter.com/stablequan_ai/status/1809009769195384851) 发布了 nanoLLaVA-1.5，这是一个紧凑的 **1B 参数视觉模型**，性能较 v1.0 有显著提升。推文中附带了模型和 Spaces 链接。
- **LLM 的 Reflection as a Service**：[@llama_index](https://twitter.com/llama_index/status/1808898730638389262) 介绍了将 Reflection（反射）作为 Agentic LLM 应用的独立服务，用于**验证输出并自我修正**以提高可靠性。文中引用了相关论文。

**AI 艺术与感知**

- **AI 与人类艺术感知投票**：[@bindureddy](https://twitter.com/bindureddy/status/1808946596845097406) 发布了一个包含 3 张 AI 生成图像和 1 张人类艺术作品的投票，挑战人们识别出人类的作品，作为一项关于艺术感知的“快速实验”。
- **AI 艺术并非剽窃**：[@bindureddy](https://twitter.com/bindureddy/status/1808804802991903222) 认为 AI 艺术不是剽窃，因为它在研究作品、获取灵感并创造新事物方面与“人类所做的完全一样”。完全的复制是剽窃，但全新的创作不是。

**迷因与幽默**

- **扎克伯格迷因视频**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1808888487975776520) 分享了一段 Mark Zuckerberg 反应的迷因视频。[@BrivaelLp](https://twitter.com/BrivaelLp/status/1808969132097839362) 调侃扎克伯格在转型为“硬核科技男”方面的“大师级表现”。
- **Caninecyte 定义**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1808871497634549961) 在一个模拟词典条目中开玩笑地将 "caninecyte" 定义为“一种以长得像狗为特征的细胞”。
- **有趣的家庭照片**：[@NerdyRodent](https://twitter.com/NerdyRodent/status/1808809146218918282) 幽默地问道：“为什么当我翻看旧家谱照片时，总有人要吐舌头？”并配上了一张像素艺术作品。

---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

AI 进展与影响

- **AI 突破的飞速步伐**：在 /r/singularity 中，一篇文章强调了[**最近的 AI 进展在人类历史长河中是多么压缩**](https://www.reddit.com/r/singularity/comments/1dv44b3/how_lucky_are_we_to_witness_this_rare_moment_in/)，如果将人类存在比作 24 小时，现代 Deep Learning 仅在最后一“秒”出现。然而，[一篇文章质疑了 AI 迄今为止的经济影响](https://archive.ph/jej1s)，尽管炒作不断。
- **AI 幽默能力**：[研究显示 AI 生成的幽默被评为比人类更幽默](https://www.psypost.org/ai-outshines-humans-in-humor-study-finds-chatgpt-is-as-funny-as-the-onion/)，且与 The Onion（洋葱新闻）不相上下，尽管 /r/singularity 的一些评论者对 AI 笑话的原创性持怀疑态度。
- **OpenAI 安全漏洞**：《纽约时报》[报道称，2023 年初，一名黑客入侵了 OpenAI 的通信系统](https://www.nytimes.com/2024/07/04/technology/openai-hack.html)并窃取了关于 AI 开发的信息，引发了人们对其在防止外国实体窃取 IP 方面做得不够的担忧。
- **抗衰老进展**：在一次 YouTube 采访中，[Altos Labs 的 CSO 讨论了通过细胞重编程在小鼠身上看到的重大抗衰老效果](https://www.youtube.com/live/Elt4xGalQu4?si=o-fBLMCzT3EEkbCl&t=1860)，老龄小鼠重焕青春。接下来将进行人体试验。

AI 模型与能力

- 讨论的新开源模型包括 [Kyutai 的 Moshi 语音模型](https://www.youtube.com/watch?v=bu7-YODAcfs)、[internlm 2.5 xcomposer 视觉模型](https://huggingface.co/internlm/internlm-xcomposer2d5-7b)，以及 [T5/FLAN-T5 被合并到 llama.cpp 中](https://github.com/ggerganov/llama.cpp/pull/8141)。
- 一项[针对 180 多个 LLM 代码生成的评估](https://symflower.com/en/company/blog/2024/dev-quality-eval-v0.5.0-deepseek-v2-coder-and-claude-3.5-sonnet-beat-gpt-4o-for-cost-effectiveness-in-code-generation/)发现，DeepSeek Coder 2 在性价比上击败了 Llama 3，而 Claude 3.5 Sonnet 的能力也旗鼓相当。只有 57% 的响应可以原样编译。

AI 安全与保障

- /r/LocalLLaMA [讨论了保护 LLM 应用安全的方法](https://www.reddit.com/r/LocalLLaMA/comments/1dvoydf/how_do_you_make_your_llm_apps_secure/)，包括通过 Fine-tuning 拒绝不安全请求、Prompt Engineering、安全模型、Regex 过滤以及不重写用户 Prompt。
- 分享了一个 [Google Gemini AI 重复已被拆穿的虚假信息](https://i.redd.it/9r1sj209yiad1.jpeg)的例子，表明目前的 AI 不能被盲目信任为事实。

AI 艺术与媒体

- 分享了[使用 Stable Diffusion、MimicMotion 和 Suno AI 生成 AI 歌手](https://www.reddit.com/r/singularity/comments/1dv0w47/ai_singers_made_by_lonely_men_stable_diffusion/)的工作流，以及[使用 ComfyUI 从单一参考图生成图像](https://i.redd.it/n544mbmdpmad1.png)的方法。
- /r/StableDiffusion 讨论了一种[在图像/视频之间迁移面部表情的新开源方法](https://www.reddit.com/r/StableDiffusion/comments/1dvmoue/wow_new_opensource_method_of_expression_transfer/)，以及 [AI Technical Artists（AI 技术美术师）新兴的角色](https://www.reddit.com/r/StableDiffusion/comments/1dv2ygq/technical_ai_artists/)，旨在为游戏工作室构建 AI 艺术流水线。
- /r/singularity [预测随着 AI 取代在线媒体，现场娱乐的需求将复苏](https://www.reddit.com/r/singularity/comments/1dvooca/with_the_rise_of_ai_in_entertainment_could_we_see/)。

机器人与具身智能

- 分享了 [Menteebot 在环境中导航](https://v.redd.it/eredb01h2lad1)、[机器人粗略采摘西红柿](https://v.redd.it/gxkgv85kggad1)以及[日本开发巨型人形机器人维护铁路](https://www.theguardian.com/world/article/2024/jul/04/japan-train-robot-maintain-railway-lines)的视频。
- 一条推文[呼吁开发开源机甲（Open Source Mechs）](https://twitter.com/xuxin_cheng/status/1808144850002628658?t=vAbXjEzAfUoa1dZu_aNjSw&s=19)。

其他

- /r/StableDiffusion 对 Auto-Photoshop-StableDiffusion 插件开发者的[突然失踪表示担忧](https://www.reddit.com/r/StableDiffusion/comments/1dvrti3/what_the_hell_happened_to_uabdullahalfaraj/)。
- Hugging Face 上分享了一个[极度恐怖主题的 16.5B 参数 LLaMA 模型](https://huggingface.co/DavidAU/L3-Stheno-Maid-Blackroot-Grand-HORROR-16B-GGUF)。
- /r/singularity 讨论了一个关于如果进度每天翻倍何时购买电脑的[“奇点悖论（Singularity Paradox）”思想实验](https://www.reddit.com/r/singularity/comments/1dv2vxh/the_singularity_paradox/)，评论指出该前提存在缺陷。


---

# AI Discord 回顾

> 总结的总结之总结


**1. LLM 性能与优化**

- 像 Llama 3、DeepSeek-V2 和 Granite-8B-Code-Instruct 这样的新模型在各种基准测试中表现强劲。例如，[Llama 3 已跃升至 ChatbotArena 等排行榜的首位](https://lmsys.org/blog/2024-05-08-llama3/)，超越了 GPT-4-Turbo 和 Claude 3 Opus 等模型。
- 优化技术正在迅速发展：
  
  - [ZeRO++](https://www.deepspeed.ai/tutorials/zeropp/) 承诺将大模型训练的通信开销降低 4 倍。
  - [vAttention 系统](https://arxiv.org/abs/2405.04437) 旨在动态管理 KV-cache 内存，以实现高效的 LLM 推理。
  - [QServe](https://arxiv.org/abs/2405.04532) 引入了 W4A8KV4 量化，以提升基于云端的 LLM 服务性能。

**2. 开源 AI 生态系统**

- 像 [Axolotl](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/) 这样的工具正支持多种数据集格式用于 LLM 训练。
- [LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex) 推出了关于构建 Agentic RAG 系统的课程。
- 像 [RefuelLLM-2](https://huggingface.co/refuelai/Llama-3-Refueled) 这样的开源模型正在发布，专注于特定的用例。

**3. 多模态 AI 与生成模型**

- 新的多模态模型正在增强各种能力：
  
  - [Idefics2 8B Chatty](https://twitter.com/sanhestpasmoi/status/1787503160757485609) 专注于改进对话交互。
  - [CodeGemma 1.1 7B](https://twitter.com/reach_vb/status/1786469104678760677) 优化了编程能力。
  - [Phi 3](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/) 通过 WebGPU 将 AI 聊天机器人带入浏览器。
- 模型组合（例如 Pixart Sigma + SDXL + PAG）旨在达到 DALLE-3 级别的输出效果。

**4. Stability AI 许可协议**

- 在社区反馈后，[Stability AI 修订了 SD3 Medium 的许可协议](https://stability.ai/news/license-update)，旨在为个人创作者和小企业提供更多清晰度。
- 关于 AI 模型许可条款及其对开源开发影响的讨论在多个社区中持续进行。
- Stability AI 推出的 **Stable Artisan**（一个集成多种 Stable Diffusion 模型用于媒体生成和编辑的 Discord 机器人）成为了热门话题（[Stable Artisan 公告](https://bit.ly/4aiVy6C)）。用户讨论了该机器人的影响，包括关于 **SD3 开源状态**的问题，以及将 **Artisan 作为付费 API 服务**推出的讨论。

**5. 社区工具与平台**

- Stability AI 推出了 [Stable Artisan](https://bit.ly/4aiVy6C)，这是一个集成 Stable Diffusion 3 和 Stable Video Diffusion 等模型的 Discord 机器人，用于在 Discord 内进行媒体生成。
- [Nomic AI 发布了 GPT4All 3.0](https://home.nomic.ai/gpt4all)，这是一款开源的本地 LLM 桌面应用，强调隐私保护并支持多种模型和操作系统。

**6\. 新 LLM 发布与基准测试讨论**：

- 多个 AI 社区讨论了新语言模型的发布，例如 **Meta 的 Llama 3**、**IBM 的 Granite-8B-Code-Instruct** 和 **DeepSeek-V2**，重点关注它们在各种基准测试和排行榜上的表现。([Llama 3 博客文章](https://lmsys.org/blog/2024-05-08-llama3/)、[Hugging Face 上的 Granite-8B-Code-Instruct](https://huggingface.co/ibm-granite/granite-8b-code-instruct)、[Hugging Face 上的 DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2))
- 一些用户对某些基准测试的有效性表示怀疑，呼吁使用更可靠的来源来设定现实的 LLM 评估标准。

**7\. 优化 LLM 训练与推理**：

- 在多个 Discord 频道中，用户分享了优化 LLM 训练和推理的技术和框架，例如用于减少通信开销的 **Microsoft ZeRO++**（[ZeRO++ 教程](https://www.deepspeed.ai/tutorials/zeropp/)）、用于动态 KV-cache 内存管理的 **vAttention**（[vAttention 论文](https://arxiv.org/abs/2405.04437)），以及用于基于量化提升性能的 **QServe**（[QServe 论文](https://arxiv.org/abs/2405.04532)）。
- 其他优化方法，如用于并行 Token 解码的 **Consistency LLMs**，也得到了讨论（[Consistency LLMs 博客文章](https://hao-ai-lab.github.io/blogs/cllm/)）。

**8\. 开源 AI 框架和数据集的进展**：

- 开源 AI 框架和数据集是各个 Discord 频道中的共同话题。**Axolotl** ([Axolotl Dataset Formats](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/))、**LlamaIndex** ([Building Agentic RAG with LlamaIndex Course](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex)) 和 **RefuelLLM-2** ([RefuelLLM-2 on Hugging Face](https://huggingface.co/refuelai/Llama-3-Refueled)) 等项目因其对 AI 社区的贡献而受到关注。
- **Modular** 框架也因其在 Python 集成和 AI 扩展方面的潜力而受到讨论 ([Modular Blog Post](https://www.modular.com/blog/developer-voices-deep-dive-with-chris-lattner-on-mojo))。

**9\. 多模态 AI 与生成模型**：

- 关于多模态 AI 和生成模型的对话非常普遍，提到了 **Idefics2 8B Chatty** ([Idefics2 8B Chatty Tweet](https://twitter.com/sanhestpasmoi/status/1787503160757485609))、**CodeGemma 1.1 7B** ([CodeGemma 1.1 7B Tweet](https://twitter.com/reach_vb/status/1786469104678760677)) 和 **Phi 3** ([Phi 3 Reddit Post](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/)) 等模型，用于聊天交互、编程和基于浏览器的 AI 等各种应用。
- 讨论中还涉及了结合 **Pixart Sigma, SDXL, and PAG** 以获得高质量输出的生成建模技术，以及用于图像重照明的开源 **IC-Light** 项目 ([IC-Light GitHub Repo](https://github.com/lllyasviel/IC-Light))。


**10\. Unsloth AI 社区的新模型发布与训练技巧**：

- Unsloth AI 社区对新发布的模型讨论热烈，例如 **IBM 的 Granite-8B-Code-Instruct** ([Granite-8B-Code-Instruct on Hugging Face](https://huggingface.co/ibm-granite/granite-8b-code-instruct)) 和 **RefuelAI 的 RefuelLLM-2** ([RefuelLLM-2 on Hugging Face](https://huggingface.co/refuelai/Llama-3-Refueled))。用户分享了使用这些模型的经验，包括 **Windows 兼容性** 方面的挑战以及对某些 **performance benchmarks** 的质疑。社区还交流了关于模型训练和 fine-tuning 的宝贵技巧和见解。

---

# 第 1 部分：Discord 高层级摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **越南语语言学之声：Vi-VLM 的愿景**：**Vi-VLM 团队**宣布推出一款专为越南语定制的 **Vision-Language model**，整合了 [Vistral 和 LLaVA](https://huggingface.co/Vi-VLM/Vistral-V-7B) 框架，专注于图像描述。用户可以在链接的仓库中找到 Demo 和支持代码。
  - **数据集可用性**：Vi-VLM 发布了一个专门用于越南语 VLM 训练的数据集，可用于增强本地语言模型应用。该数据集丰富了东南亚语言的语言资源。
- **图像处理难题：寻找 WHAM 的替代方案**：一位爱好者正在寻找 **WHAM** 的替代方案，用于复杂视频中的 **human pose estimation**（人体姿态估计），并指出其 Python 和 CV 依赖项非常笨重。社区交流暗示需要更适合非技术用户使用的高级 AI 应用工具。
  - 分享了 **ViT 和 U-Net** 实现的学习资源，包括来自 [Zero to Mastery](https://www.learnpytorch.io/08_pytorch_paper_replicating/) 的指南和 Andrew Ng 的课程，显示出社区对掌握这些 vision transformer 模型的兴趣。
- **调优：音频语言模型讨论**：**Moshi 的语言流畅性**：Yann LeCun 分享了一条 [推文](https://x.com/ylecun/status/1808573888642617406)，重点介绍了 **Kyutai.org 的数字海盗 (digital pirate)**，它可以理解带有法国口音的英语，展示了该模型多样化的听觉处理能力。
  - 对 **Flora** 论文和音频语言模型的兴趣依然浓厚，反映了 AI 社区对跨模态能力的关注。人们热切期待即将举行的关于这些主题的论文研读会。
- **陷入停滞：Mistral 模型僵局**：用户报告 **Mistral 模型** 的推理过程陷入停滞，特别是在 3000 次迭代中的第 1800 次，这可能暗示了缓存复杂性问题。这反映了在执行大规模计算任务时管理资源的实际挑战。
  - 围绕如何在不本地下载模型的情况下进行有效的 API 调用展开了讨论，强调了对精简远程推理协议的需求。以 API 为中心的对话凸显了向更灵活、基于云的 ML 操作发展的趋势。
- **Diffusion 讨论：RealVisXL 和 ISR**：专为渲染写实视觉效果而优化的 **RealVisXL V4.0** 正在训练中，拥有 [官方页面](https://huggingface.co/SG161222/RealVisXL_V4.0_Lightning) 并在 Boosty 上获得赞助，突显了社区对模型开发的支持。
  - 现有的 IDM-VTON 在 Google Colab 中出现的 “no file named diffusion_pytorch_model.bin” 错误是 Diffusion 模型领域中典型的排障对话，强调了 AI 部署的实际层面。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **消除困惑：Stability AI 的社区许可**：Stability AI 在收到反馈后修订了其 **SD3 Medium** 的发布许可，推出了全新的 **Stability AI Community License**，明确了个人创作者和小微企业的使用权限。详情见其[近期公告](http://stability.ai/news/license-update)，该公司在商业权益与社区支持之间**寻求平衡**。
  - 用户现在可以根据新许可免费将 **Stability AI 的模型用于非商业用途**，这对社区来说是**开源福音**，并引发了关于这些变化如何影响模型开发和可访问性的讨论。
- **动漫 AI 模型的蜕变：Animagine XL 3.1**：由 [Cagliostro Research Lab](https://huggingface.co/cagliostrolab) 和 [SeaArt.ai](https://www.seaart.ai/) 开发的 **Animagine XL 3.1** 模型因其较前代模型的增强而引发热议，将更高质量和更广泛的动漫图像推向了前沿。
  - **AAM Anime Mix XL** 也引起了关注，引发了与 **Animagine XL 3.1** 的大量对比，爱好者们讨论了他们在不同动漫专注生成模型之间的体验和偏好。
- **辩论 GPU 军备竞赛：多 GPU 配置**：技术社区正在积极讨论**多 GPU 配置**的优化，以提升 **Stable Diffusion 的性能**，重点关注像 **SwarmUI** 这样针对复杂配置的工具。
  - 辩论集中在高效管理资源和实现高质量输出的挑战上，强调了在不断发展的 AI 模型训练领域中，导航所需的**技术实力与创造力**的结合。
- **CivitAI 对 SD3 的立场引发争议**：CivitAI 禁止 **SD3 模型**的举动在社区内引发了分歧，一些人认为这可能是 **Stable Diffusion 3** 框架发展的潜在障碍。
  - 随着对许可复杂性、商业影响以及这一决定如何塑造未来协作和模型演进的深入见解，讨论进一步加深。
- **许可与限制：备受审视的 Stable Diffusion**：最新的对话审视了 **Stable Diffusion 3 的许可**及其与个人和企业使用的兼容性，考虑到社区对 AI 模型实验的清晰度和自由度的需求。
  - 社区情绪出现分歧，讨论围绕着感知到的许可限制是否不公平地惩罚了小型项目，或者它们是否是 AI 领域技术成熟过程中固有的一部分。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Gemma 的量子飞跃**：全新的 **Gemma 2** 已震撼技术圈，其 **finetuning 速度提升了 2 倍**，且 **VRAM 占用减少了 63%** ([Gemma 2 博客](https://unsloth.ai/blog/gemma2))。在 Unsloth 用户中，27B 模型支持高达 **9.7K token 上下文**成为了一个特别的亮点。
  - 发布初期伴随的 **notebook 问题**（如标签错误）被社区成员指出博客文章过于仓促，但这些问题已被开发者迅速解决 ([Notebook 修复](https://github.com/unslothai/unsloth/pull/67))。
- **Replete-AI 的海量数据集**：Replete-AI 推出了两个大型数据集：**Everything_Instruct** 及其多语言版本，每个都包含 11-12GB 的 instruct 数据 ([Replete AI 数据集](https://huggingface.co/datasets/Replete-AI/Everything_Instruct))。超过 600 万行数据可供 AI 开发者使用，有望推动下一波语言模型训练。
  - 社区的热情伴随着质量检查，开发者们对数据集的 **deduplication（去重）** 和 **内容平衡** 进行了探究，体现了对数据集精细构建的专业眼光。
- **固定 Notebook 资源**：**collaboration** 频道的请求促成了 **置顶多功能 notebook** 的承诺，旨在帮助成员快速定位核心资源。
  - 社区持续努力 **修正 notebook 链接**，并承诺将其整合到 Unsloth 的 **GitHub 页面**中，展示了一个动态的社区驱动型文档流程 ([GitHub Unsloth](https://github.com/unslothai/unsloth))。
- **Unsloth 2024.7 的补丁与进展**：Unsloth 的 2024.7 补丁因 **checkpoint 相关错误** 评价褒贬不一，但它通过将 **Gemma 2 支持** 集成到 Unsloth 不断扩展的工具包中，迈出了重要一步 ([2024.7 更新](https://github.com/unslothai/unsloth))。
  - 忠实用户和 Unsloth 响应迅速的开发者正致力于解决 **fine-tuning 中的小瑕疵和错误修复**，证明了对于精细化模型优化至关重要的强大反馈闭环。
- **Facebook 备受争议的 Token 策略**：Facebook 的 **multi-token prediction** 模型引发了关于访问壁垒的辩论，在 Unsloth 紧密的社区中激起了各种观点。
  - 对数据隐私的批评观点屡见不鲜，特别是涉及使用 Facebook 模型需要共享联系人数据的问题，这引发了关于 AI 伦理使用的持续讨论 ([Facebook 的 Multi-Token 模型](https://huggingface.co/facebook/multi-token-prediction))。

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **冲刺融资跑道**：紧随 [rakis](https://github.com/hrishioa/rakis?tab=readme-ov-file) 的链接，社区成员讨论了 AI 与 blockchain 交叉领域高达 8500 万美元的种子轮投资，引发了关于当前技术领域风险投资趋势的对话。
  - BM42 的开发者因 benchmark 可能存在偏差而面临压力，促使社区警惕并倡导严谨的评估实践；这促使其[修改了指标和数据集的方法](https://x.com/qdrant_engine/status/1809291686625046816)。
- **碰撞：编程工具**：用户对比了 git merge 工具的使用体验，特别点名了 [lazygit](https://github.com/jesseduffield/lazygit) 和 [Sublime Merge](https://www.sublimemerge.com/)，讨论转向了对更细致的代码冲突解决工具的需求。
  - Claude 3.5 和其他基于 AI 的工具因其在编程辅助方面的出色表现而备受关注，强调了代码补全的效率以及处理复杂多文件重构的能力。
- **调频技术对话**：在 **Latent Space Podcast** 中，来自 Reka 的 Yi Tay 阐述了为 frontier models 开发训练栈的过程，并将其规模和策略与 OpenAI 及 Google Gemini 团队进行了类比。
  - 听众受邀在 [Hacker News](https://news.ycombinator.com/item?id=40886218) 上参与实时讨论，架起了播客与更广泛的 AI 研究社区对话之间的桥梁。
- **应对音视频故障**：OpenAI 的 AV 在 AIEWF 演示期间出现中断，随后出现了切换到 Zoom 的呼声，紧接着迅速采取行动分享了 Zoom [会议链接](https://us06web.zoom.us/j/8807908941?pwd=eHBBdk9sWWluSzB2TFdLOVdEN3BFdz09) 以获得更好的 AV 稳定性。
  - Discord 与 Linux 之间的兼容性问题仍然是一个反复出现的技术难题，促使用户探索更适合 Linux 的通信替代方案。
- **解构模型合并热潮**：关于 **model merging tactics** 的辩论成为焦点，参与者思考了 **LlamaFile 和 Ollama** 等工具的不同目标和潜在的集成策略。
  - 对话深入探讨了可穿戴技术与 AI 集成以增强活动体验的可能性，并对隐私和知情同意进行了深度思考。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Snapdragon 令人惊讶的速度**：搭载 **Snapdragon Elite 的 Surface Laptop** 展示了强大的实力，在 **8bit 精度的 LLaMA3 8b** 上实现了 **1.5 秒的首个 token 响应时间**和**每秒 10 个 token** 的速度，而 **GPU** 占用率仅为 10%。目前尚未监测到 **NPU** 活动，但该笔记本的速度引发了人们对 **NPU** 最终提升 **LLaMA** 模型性能的猜测。
  - 技术爱好者将 **Snapdragon 的 CPU 实力**与旧款 **Intel** 产品进行了对比，发现前者的速度非常惊人。在欢声笑语中，技术社区调侃了一个临时搭建的“纸板 NPU”，并预测在进行适当的 **NPU** 编程后，性能可能会达到巅峰。
- **量化奇点与代码探索**：**Gemma-2-27b** 出现了量化难题，模型基准测试在不同量化版本中表现异常。与此同时，定制的系统提示词优化了 **Gemma 2 27B** 的性能，使其能够输出符合 **PEP 8** 标准且高效的算法。
  - 有建议指出，**Qwen2 模型**在使用 **ChatML** 和 **flash attention** 设置时表现最佳，而使用非 **CUDA** 设备的用户则对 **IQ 量化**带来的混乱提出了警告，指出在替代架构上表现明显更好。
- **LM Studio 的 ARM 僵局**：一位恼火的用户对 **LM Studio** 的 **AppImage** 无法在 **aarch64 CPU** 上运行表示沮丧。错误提示显示存在语法冲突，一行哀叹确认道：“*Linux 上不支持 ARM CPU。*”
  - 对话打破了立即包含 **ARM CPU** 支持的希望，让 **Linux** 忠实拥趸们感到渴望。一种共同的观点认为，**LM Studio** 的**架构调整**已在计划中，但尚未正式落地。
- **RTX 的坎坷之路**：**RTX 4060 8GB VRAM** 的所有者表达了他们在处理 **20B 量化模型**时的困境；与 token 的顽强搏斗最终导致系统完全冻结。论坛其他成员对他们表示同情，回想起自己使用 **RTX 4060** 时支离破碎的经历。
  - 社区指南为 **GPU** 困扰者带来了一线希望，建议中端机器用户使用 **Mistral 7B** 和 **Open Hermes 2.5** 等负载较小的模型。大家纷纷赞扬这些较小的模型，建议避开那些巨大的 token 吞噬者。
- **ROCm 的救援角色**：拥有 **7800XT** 的用户在模型运行混乱、无法实现 **GPU offload** 时表达了他们的痛苦。一个脚本标志着成功，安抚了寻求 **ROCm** 慰藉的超负荷系统。
  - 技术集体汇聚解决方案，证实了 **ROCm 安装脚本**的有效性。论坛中响起了欢快的旋律，因为 **GPGPU** 大师们已经找到了一个值得技术圈使用的变通方法。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA 难题与混合精度 MatMul**：CUDA MODE 社区的讨论转向了使用 CUDA 优化矩阵乘法，重点关注了一篇关于 GPU 矩阵乘法中列加载技术的[博客文章](https://siboehm.com/articles/22/CUDA-MMM)；另一个话题探讨了用于 int2\*int8 的定制 **gemv kernels** 的发布，以及用于混合精度操作的 BitBLAS 库。
  - 用户探讨了 **TorchDynamo** 在 PyTorch 性能中的作用，并对比了使用 Python 与 C++ 进行 **CUDA kernel** 开发的易用性，Python 因其在初始阶段的灵活性而更受青睐。一些用户在将 `torch.compile` 适配 Python 3.12 字节码变化时遇到挑战，这在最近的[讨论](https://dev-discuss.pytorch.org/t/torch-compile-support-for-python-3-12-completed/2054)中得到了解决。
- **GPT 撰写执行摘要与模型训练试验**：一篇详细介绍使用 GPT 起草执行摘要的[博客文章](https://www.dylandavis.net/2024/07/three-gpts-walk-into-a-bar-and-write-an-exec-summary/)引发了关注，而使用 FP8 梯度的 LLM 训练试验因导致 loss 增加而被标记，促使某些操作切换到 BF16。
  - Schedule-Free Optimizers 拥有更平滑的 loss 曲线，[用户分享](https://x.com/_clashluke/status/1808590060654108910)了其收敛优势的经验证据。同时，关于后端 SDE 转型 CUDA 推理优化的讨论也在进行，建议涵盖了在线资源、课程推荐和社区参与。
- **AI 播客与主题演讲引发热烈讨论**：**Lightning AI** 与 Luca Antiga 和 Thomas Viehmann 合作的 [Thunder Sessions 播客](https://x.com/LightningAI/status/1808610408481370205)引起了社区成员的注意，而 **Andrej Karpathy** 在 [UC Berkeley 的主题演讲](https://www.youtube.com/watch?v=tsTeEkzO9xc)则是创新和学生才华的亮点。
  - 随意的交谈和频道互动描绘了一个互动活跃的论坛图景，成员们分享了简短的兴奋或赞赏之情，但在标记为关注度较低的频道中则较少进行深入的技术交流。
- **深度学习框架与 Triton Kernel 修复**：尝试用 C++ 从零构建一个类似于 **tinygrad** 的深度学习框架揭示了复杂性障碍，引发了关于 C++ 与 Python 在此背景下提供的功能支持的辩论。同时，在并行 CUDA graph 实例中 Triton kernel 的 `tl.load` 问题需要通过创意手段来规避延迟问题。
  - 在讨论 **torch.ao** 中 `.to` 方法的功能时，发现了更多的复杂性，目前的限制约束了 dtype 和内存格式的更改，促使开发者对函数进行了临时修正，正如问题追踪器和 [commit 日志](https://github.com/pytorch/ao/blob/a8956992191853b13f82ceb3e6929bed7691a3fa/torchao/dtypes/affine_quantized_tensor.py#L262)中所讨论的那样。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Llamas Looping Lines：AI 中的重复故障**：用户反映 **Perplexity AI** 在 **Llama 3** 和 **Claude** 等模型中出现了重复输出的现象，并得到保证称该问题正在处理中，即将发布修复补丁。
  - **Alex** 确认了该问题已被记录并正在努力修复，这标志着 **Perplexity AI** 性能基准测试中一个亟待解决的问题。
- **实时现实检查失败：实时访问故障**：由于 **Perplexity AI** 用户面临实时互联网数据检索问题，接收到的是过时而非最新的信息，导致预期与实际出现差距。
  - 尽管用户尝试通过重启应用程序来解决准确性问题，但反馈频道显示该问题依然存在。
- **数学模型失误：Perplexity Pro 的计算挑战**：**Perplexity Pro** 的计算（如 CAPM beta）被指出存在不准确之处，尽管其源自 **GPT-4o**，这为其在可靠学术应用上的前景蒙上了阴影。
  - 社区对该模型在需要精确数学解题领域的实用性表达了不满和担忧。
- **股市成功案例：Perplexity 的盈利预测**：一些利用 **Perplexity AI** 进行股市决策并获利（如赚取 8,000 美元）的财务胜利案例浮出水面，引发了关于其多样化益处的讨论。
  - 这些用户故事证明了 **Perplexity AI** 的 **Pro** 版本在现实世界用例中的多样化能力。
- **订阅审查：解码 Perplexity AI 方案**：随着用户深入探讨 **Pro** 和 **Enterprise Pro** 方案之间的差异，特别是关于 **Sonnet** 和 **Opus** 等模型的分配，相关问题和对比也随之增多。
  - 咨询不仅集中在可用性上，还包括 Perplexity 不同订阅服务中所包含模型的具体细节。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **BUD-E 面板扩展**：[BUD-E 现在可以读取剪贴板文本](https://youtu.be/WMcEzVfEXpM)，这一新功能在一段 YouTube 视频中展示，详情见 [GitHub](https://github.com/christophschuhmann/Desktop_BUD-E/tree/main)。该功能演示虽然画质较低，但引发了轻松的讨论。
  - 社区讨论了由于反复使用重叠数据集而导致的 **AI model training** 挑战，**FAL.AI** 的数据集访问障碍凸显了这一问题。相比之下，像 **Chameleon** 这样的突破则与多种集成数据相关联。
- **Clipdrop 审查困惑**：Clipdrop 的 NSFW 检测出现误判，将一张正常的图片错误标记为不当内容，引起了社区的哄笑。
  - [Stability AI 修改了许可证](https://stability.ai/news/license-update)，**SD3 Medium** 现在采用 Stability AI Community License，在收到社区反馈后，允许个人创作者和小企业获得更多访问权限。
- **T-FREE 趋势引领者**：在[最近发表的论文](https://arxiv.org/abs/2406.19223)中详细介绍的新型 **T-FREE tokenizer**，承诺在字符三元组（character triplets）上实现稀疏激活，无需大型参考语料库，并可能减少超过 **85%** 的 embedding 层参数。
  - 该方法因增强了在冷门语言上的性能并精简了 embedding 计时器而受到赞誉，为 LLM 增添了紧凑的优势。
- **警报：公会中的诈骗者**：#[research] 频道发现了一名诈骗者，提醒社区保持高度警惕。
  - 一名用户在多个频道发布了一串完全相同的钓鱼链接，声称提供“50 美元礼品卡”，引起了关注。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **虚空中的声音**：新发布的 **Moshi AI demo** 引起了热议，人们对其**实时语音交互**感到兴奋，但同时也对其响应中断和循环回复的问题感到失望。
  - **Hume AI 的 playground** 因缺乏长期记忆而受到审视，这让追求**持久 AI 对话**的用户感到沮丧。
- **内存库受质疑**：GPT 的记忆能力受到抨击，因为它虽然保存了用户偏好但仍会编造响应；成员们建议通过**增强自定义功能**来缓解这一问题。
  - 出现了一场激烈的 **GPT-2 与现代模型**的辩论，对比了旧模型的**成本效益**与 GPT-3.5 Turbo 等当前迭代版本的性能飞跃。
- **ChatGPT：免费版 vs. Plus 计划**：明确了**付费版 ChatGPT Plus** 计划的优势，详细说明了更高使用限额、**DALL·E** 访问权限以及扩展的上下文窗口等福利。
  - 针对 **GPT-4 使用量**的担忧得到了回应，明确了达到限制后的冷却期，特别是 Plus 会员每 3 小时最多可发送 40 条消息。
- **AI 工具箱扩展**：社区成员探索了用于**测试多个 AI 响应**提示词的工具，并推荐了一个**定制工具**以及用于高效评估的现有选项。
  - 话题转向了 **API 集成**，研究了用于将 AI 模型链接到多样化数据集的严格聚合生成器（RAG），并利用现有的 **Assistant API endpoints**。
- **结合上下文的申诉**：在 **#prompt-engineering** 频道中，勾勒了申诉**交通罚单**的策略，建议采用结构化方法和法律辩论技巧。
  - 关于创建**员工表彰计划**以提高职场士气的讨论十分热烈，重点关注了重大贡献的目标和表彰标准。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Replete-AI 发布海量数据集**：**Replete-AI** 发布了两个巨大的数据集，名为 **Everything_Instruct** 和 **Everything_Instruct_Multilingual**，容量达 11-12GB，包含超过 600 万条数据。其意图是整合各种 instruct 数据以推进 AI 训练。
  - **Everything_Instruct** 针对英语，而 **Everything_Instruct_Multilingual** 则引入了多语言混合，以扩展 AI 的语言处理能力。这两套数据集呼应了以往 bagel 数据集的成功，并借鉴了 EveryoneLLM AI 模型。[在 Hugging Face 深入了解](https://huggingface.co/datasets/Replete-AI/Everything_Instruct_Multilingual)。
- **Nomic AI 发布 GPT4All 3.0**：**Nomic AI** 的最新作品 **GPT4All 3.0** 问世，这是一个开源的本地 LLM 桌面应用，支持多种模型并优先考虑隐私。该应用以重新设计的用户界面著称，并采用 MIT 许可证。[探索其功能](https://home.nomic.ai/gpt4all)。
  - **GPT4All 3.0** 拥有超过 25 万月活跃用户，促进了与 LLM 的私密本地交互，减少了对互联网的依赖。其采用率非常强劲，标志着向本地化和私密 AI 工具使用的转变。
- **InternLM-XComposer-2.5 提升标准**：**InternLM** 推出了 **InternLM-XComposer-2.5**，这是大型视觉语言模型中的佼佼者，能够出色地处理 24K 交错图文上下文，并通过 RoPE 外推扩展至 96K。
  - 该模型在 16 项基准测试中取得了顶尖结果，正在逼近 GPT-4V 和 Gemini Pro 等巨头。这款融合了创新与竞争力的 [InternLM 之作正等待探索](https://x.com/_akhaliq/status/1808747694317261114?s=46)。
- **Claude 3.5 的难题与封锁**：尝试绕过 **Claude 3.5 Sonnet** 伦理限制的行为让用户感到沮丧，围绕特定预设提示词（pre-prompts）的策略几乎没有效果。
  - 尽管 Claude 的限制非常坚固，但社区分享了尝试 **Anthropic's workbench** 的建议。然而，用户被警告此类尝试后存在账号受限的风险。[查看对话详情](https://console.anthropic.com/workbench)。
- **Apollo 的艺术 AI 之路**：**Achyut Benz** 向世界展示了 Apollo 项目，这是一个能够创作类似于备受推崇的 **3Blue1Brown** 动画视觉效果的 AI。它构建于 **Next.js** 之上，接入了 **GroqInc**，并交织使用了 **AnthropicAI 3.5 Sonnet** 和 **GPT-4**。
  - Apollo 致力于通过 AI 生成的内容来增强学习体验，这让技术型教育者非常受用。[观看 Apollo 的发布演示](https://x.com/achyut_benz/status/1808969030969274507?s=46)。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **LLM 部署的重大飞跃**：OpenRouter 的 LLM 模型部署策略指定 **FP16/BF16** 为默认量化标准，例外情况会通过相关的**量化图标**注明。
  - 这种量化方法的采用引发了关于技术影响和效率提升的详细讨论。
- **OpenRouter 避开 API 灾难**：**Microsoft API** 的突然变化本可能给 OpenRouter 用户带来灾难，但一个快速补丁使一切恢复正常，赢得了社区的掌声。
  - 该修复恢复了和谐，反映了 OpenRouter 在面对技术中断时快速周转的准备能力。
- **Infermatic 注入隐私信心**：在一份肯定的更新中，**Infermatic 宣布了其对实时数据处理的承诺**，并发布了新的[隐私政策](https://infermatic.ai/privacy-policy/)，明确表示不会保留输入提示词或模型输出。
  - 这一更新为用户带来了清晰度和安全感，使该平台摆脱了此前的数据保留疑虑。
- **DeepSeek 解码等式之谜**：用户在排查 **DeepSeek Coder** 问题时，发现了一个解决等式无法渲染的变通方法，即巧妙地使用 regex 来调整输出字符串。
  - TypingMind 前端无法正确处理提示词的持久性问题已被标记待修复，展示了积极的社区参与。
- **昂贵的 API 引起同行关注**：围绕 **Mistral** 的 **Codestral API** 定价策略展开了激烈辩论，一些社区成员认为 22B 模型定价过高。
  - 用户互相推荐更经济实惠的替代方案，如 **DeepSeek Coder**，它在不耗费巨资的情况下提供了极具竞争力的编程能力。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **数字思维的指纹**：社区探索了使用**拓扑数据分析 (TDA)** 进行独特的模型指纹识别，并辩论了用于模型验证的等效校验和指标的效用，例如使用 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 对 `LlamaForCausalLM` 进行验证。
  - 讨论还涉及通过**拓扑数据分析**利用其**不变量**来剖析模型权重，参考了 [TorchTDA](https://giotto-ai.github.io/gtda-docs/0.5.1/library.html) 等资源，并考虑了来自 [1.58-bit LLMs](https://arxiv.org/abs/2402.17764) 等论文的位级创新以提高效率。
- **扩展与优化之谈**：关注了用于扩展定律的 [efficientcube.ipynb](https://github.com/kyo-takano/chinchilla/blob/master/examples/efficientcube.ipynb) 笔记本，同时强调了 JAX 中的 **AOT 编译能力**是[预执行代码](https://jax.readthedocs.io/en/latest/aot.html#debug-information-and-analyses-when-available)优化的进步。
  - 分享了 Flax 中 JIT 函数的 **FLOPs 估算方法**，并重新调查了临界批量大小，挑战了性能在低于特定阈值时不受影响的假设。
- **稀疏编码器与残差启示**：讨论了在 Llama 3 8B 残差流上训练的**稀疏自编码器 (SAEs)** 的部署，介绍了与 LLM 集成以实现更好处理的实用工具，并提供了[模型实现](https://huggingface.co/EleutherAI/sae-llama-3-8b-32x)的细节。
  - 深入研究残差流处理，该策略按层组织 SAEs，以优化它们与 Llama 3 8B 的协同作用，详见相关的[模型卡片](https://huggingface.co/EleutherAI/sae-llama-3-8b-32x)。
- **利用并行评估的动力**：爱好者提出了关于缓存预处理输入的可行性以及解决 **Proof-Pile Config Errors** 的问题，指出切换到 `lambada_openai` 规避了该问题。
  - 值得注意的包括模型名称长度问题，导致 **OSError(36, 'File name too long')**，并就设置并行模型评估寻求指导，同时收到了关于单进程评估假设的警告。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain 的哀鸣**：LangChain 用户报告了在 CPU 上运行时的**性能问题**，响应时间长和处理步骤复杂是主要的痛点。
  - 关于这种迟钝是由于模型推理效率低下还是缺乏 GPU 加速的争论仍在继续，而一些人认为它被不必要的复杂性所拖累，正如[这里](https://www.octomind.dev/blog/why-we-no-longer-use-langchain-for-building-our-ai-agents)所讨论的。
- **AI 模型大决战：OpenAI vs ChatOpenAI**：由于前者可能会被逐步淘汰，关于使用 **OpenAI** 优于 **ChatOpenAI** 的优势展开了讨论，引发了对它们实现效率的比较。
  - 成员们分享了围绕特定任务需求的各种经验，而一些人则因其熟悉的界面和工具链而更倾向于使用 OpenAI。
- **Juicebox.ai：人才搜索奇才**：**Juicebox.ai** 的 **PeopleGPT** 因其无需 Boolean 的自然语言搜索能力而受到称赞，能够迅速识别合格人才，并通过易于使用的功能增强了人才搜索体验。
  - 技术社区赞扬了其过滤与自然语言搜索的结合，提升了用户的整体体验；详情请见[这里](https://juicebox.ai/)。
- **RAG 聊天机器人日历难题**：一位 **基于 LangChain RAG 的聊天机器人** 开发者寻求集成 **演示预约功能** 的指导，强调了在实现过程中发现的复杂性。
  - 社区的反应倾向于协助完成这一集成，表明了尽管缺乏明确的资源链接，大家仍在共同努力增强聊天机器人的功能。
- **视觉向量化精湛技艺**：[一篇博客文章](https://www.lightly.ai/post/vector-indexes-and-image-retrieval-using-lightly)概述了使用 **Lightly SSL** 和 **FAISS** 创建 **E2E 图像检索应用** 的过程，并配备了 vision transformer 模型。
  - 该文章附带了 [Colab Notebook](https://colab.research.google.com/drive/1n4CwX5T6Ch2v7OYTRe6g1j_QJHxxOvcM) 和 [Gradio app](https://huggingface.co/spaces/lightly-ai/food101-image-retrieval)，旨在鼓励同行学习和应用。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex RAG 专题网络研讨会热潮**：**LlamaIndex** 与 **Weights & Biases** 合作举办了一场网络研讨会，旨在揭开 **RAG 实验和评估** 中涉及的复杂性。该会议承诺提供关于准确的 LLM Judge 对齐的见解，并重点关注与 Weights & Biases 的协作。
  - 随着 **RAG pipeline** 成为即将举行的网络研讨会的焦点，人们的期待与日俱增，突显了该领域的挑战。对 RAG 细微评估的一丝怀疑也引发了社区对该活动的关注。
- **AI 明星崭露头角**：新星 **@ravithejads** 分享了他成为 **明星级 AI 工程师和教育者** 的晋升之路，激发了 **LlamaIndex** 社区内的抱负。
  - **LlamaIndex** 阐述了 @ravithejads 对 OSS 的贡献以及对 AI 趋势的持续关注，引发了关于 AI 职业发展路径的讨论。
- **反思“Reflection as a Service”**：'Reflection as a Service' 在 **LlamaIndex** 备受关注，它提出了一种内省机制，通过添加自我修正层来提高 LLM 的可靠性。
  - 这种创新方法吸引了社区，引发了关于其通过智能**自我修正**增强 Agent 应用潜力的对话。
- **Cloud Function 挑战与协作修复**：关于 **Google Cloud Function** 在 **多模型加载** 方面的困难出现了讨论，引发了 AI 爱好者集体寻找更高效的方法。
  - 随着成员们分享减少加载时间和优化模型使用的策略，社区智慧得以传播，展示了解决问题中的协作精神。
- **CRAG – 纠错机制登场**：**Yan 等人** 介绍了 **Corrective RAG (CRAG)**，这是一种创新的 LlamaIndex 服务，旨在检索过程中动态验证和纠正不相关的上下文，引起了 AI 从业者的兴趣。
  - 人们将 **CRAG** 与推进检索增强生成系统的可能性联系起来，激发了关于改进和准确性的前瞻性对话。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **AI 聚会的开放邀请**：社区成员澄清，参加**伦敦 AI 活动**不需要**特殊资格**；只需填写表格即可。这种包容性政策确保了活动对所有人开放，促进了多元化的思想交流。
  - 关于活动出席情况的讨论强调了 AI 聚会中**社区参与**和**开放访问**的重要性，因为这些政策促进了跨领域和专业水平的更广泛参与及知识共享。
- **生产模式下的 API 烦恼**：一名成员在生产环境中部署使用 Cohere 的 **rerank API** 的应用时遇到了 TypeError 问题，这引发了一个故障排除讨论帖，与其在本地的顺利运行形成对比。
  - 社区在解决 rerank API 问题上的协作努力展示了共享知识和即时同行支持在克服生产环境技术挑战中的价值。
- **AI 开发领域的新面孔**：具有不同背景的新成员（包括一名**计算机科学毕业生**和一名专注于教学的 **AI 开发者**）介绍了自己，表达了为公会的集体专业知识做出贡献的渴望。
  - 对新人的热烈欢迎凸显了公会致力于培育一个充满活力的 AI 爱好者社区，并为协作增长和学习做好准备。
- **Command R+ 抢占风头**：Cohere 宣布了 Command R 系列中最强大的模型 **Command R+** 现已可以使用，在技术型受众中引起了不小的轰动。
  - Command R+ 的发布被视为提升 AI 模型能力和应用的重要一步，表明了该领域持续创新的动力。
- **使用 Rhea.run 保存脚本**：**Rhea.run** 中引入的“保存到项目” (Save to Project) 功能受到了热烈欢迎，它允许用户通过对话式 HTML 脚本创建并保存交互式应用程序。
  - 这一新功能强调了 Rhea.run 致力于简化应用创建过程，从而使开发者能够轻松地进行构建和实验。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **MacOS Copilot 成为关注焦点**：搭载了 GPT-4、Gemini 1.5 Pro 和 Claude-3 Opus 的 [Invisibility MacOS Copilot](https://x.com/sulaimanghori/status/1791113392482377833) 因其上下文吸收能力而受到关注，目前可免费使用。
  - 社区成员对潜在的开源 grav.ai 以将类似功能整合到 **Open Interpreter (OI) 生态系统**中表现出兴趣。
- **'wtf' 命令为 OI 增添调试魅力**：'wtf' 命令允许 **Open Interpreter** 智能地切换 [VSC 主题](https://discord.com/invite/YQhmv5pd?event=1258399216078684242)并提供终端调试建议，引发了社区的兴奋。
  - 成员们对该命令直观执行动作的能力表示惊讶，并计划分享关于安全圆桌会议和即将举行的 **OI House Party 活动**的进一步更新。
- **O1 Light 爱好者的发货烦恼**：社区中充满了对 **01 Light 发货**的期待和沮丧，讨论围绕着延迟展开。
  - 等待的情绪引发了共鸣，强化了大家对发货时间表透明沟通的共同愿望。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 对象出现异常！**：成员们讨论了一个影响 **Mojo 对象**（与 **Python 对象** 相比）的类型转换（casting）Bug，可能与 [GitHub Issue #328](https://github.com/modularml/mojo/issues/328) 有关。
  - 随后引发了关于 **casting bug** 是否与对象处理方式的差异相关的辩论，正如 Issue [#3065](https://github.com/modularml/mojo/issues/3065) 和 [#3167](https://github.com/modularml/mojo/issues/3167) 中所述。
- **MLIR 的无符号整数风波**：社区发现 **MLIR** 将无符号整数（unsigned integers）解释为有符号整数，引发了讨论并导致了 [GitHub Issue #3065](https://github.com/modularml/mojo/issues/3065) 的创建。
  - 针对这一无符号整数转换问题对各类用户的影响，担忧情绪激增，对话焦点转向了这一新出现的 Bug。
- **编译器 Nightly 新闻：段错误（Segfaults）与解决方案**：最近 **nightly** 构建中的 **segfaults** 导致了 Bug 报告的提交和问题文件的分享，详见[此处](https://github.com/Mojo-Numerics-and-Algorithms-group/MojoSci/blob/dynobs/src/diffeq/runga_kutta.mojo)。
  - 此外，官方发布了新的编译器版本，在 `2024.7.505` 版本中包含了 `exclusive` 参数和新方法等改进，链接见 [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)。
- **马拉松进展：Mojo 的矩阵乘法**：Benny 分享了一种矩阵乘法技术，令人印象深刻，并建议定制分块大小（block sizes），建议同行参考德克萨斯大学奥斯汀分校（UT Austin）的论文以获取见解。
  - 在另一个讨论线程中，由于编译时间增加和最新测试套件中的 **segfaults**，进度有所减缓，参与者互相推荐了诸如 [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1TBz9Lp0JT1Ph7ndfbWqp-B30FQcRYl1959hP2lZ6yH4/edit) 之类的论文资源。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **无需 Chain 的独立 Smithing**：讨论确认 **LangSmith** 可以脱离 **LangChain** 独立运行，正如 [Colab](https://colab.research.google.com/github/langchain-ai/langsmith-cookbook/blob/main/tracing-examples/traceable/tracing_without_langchain.ipynb) 和 [GitHub](https://github.com/langchain-ai/langsmith-cookbook/blob/main/tracing-examples/traceable/tracing_without_langchain.ipynb) 上的示例所示。**LangSmith** 允许对 **LLMs** 进行**插桩（instrumentation）**，从而深入了解应用行为。
  - 社区成员缓解了关于 AI 课程期间 GPU 额度的担忧，强调了条款的正确传达，并引导至[课程平台](https://example.com/course-terms)上的清晰信息。
- **额度澄清与每月挑战**：一个热门话题围绕着 **1000 美元的月度额度**及其有效期展开，共识是额度不会结转（no rollover），但依然对这一优惠表示赞赏。
  - 一位用户对 Mistral 微调后余额莫名增加到 **1030 美元** 表示疑惑，引发了关于每月可能存在 **30 美元默认额度** 的猜测。
- **训练微调：折腾 Token**：关于使用 `type: input_output` 设置 `Meta-Llama-3-8B-Instruct` 的讨论引发了一些困惑，用户们正在检查特殊 Token 和模型配置，参考 [GitHub](https://github.com/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/special_tokens_map.json)。
  - 训练者发现 **L3 70B Instruct** 的效果优于 **L3 8B**，这是在配置默认指向 Instruct 模型时偶然发现的，凸显了模型选择的影响。
- **额度困惑与课程进度跟进**：关于服务额度资格的确定性尚不明确，一名成员寻求关于 **6 月 14 日**入群后条款的澄清。
  - 另一位用户也表达了对算力额度过期的担忧，请求为因日程疏忽而错过的剩余额度申请延期。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **揭秘演示困境 (Debunking the Demo Dilemma)**：社区成员挑战了一个 AI 演示的真实性，质疑其响应的现实感，并强调了显著的 **response time** 问题。该讨论包含了一个指向有争议的 [demonstration](https://x.com/benhylak/status/1808611023123067357) 的链接。
  - 作为一种道歉式的转向，**Stability AI** 针对社区反馈对 **Stable Diffusion 3 Medium** 进行了修订，并对其 [license](https://x.com/stabilityai/status/1809274908641489160?s=46) 进行了澄清，为未来的 **high-quality Generative AI** 努力划定了路径。
- **搜索大对决：BM42 vs. BM25**：**Qdrant Engine** 宣称其 BM42 是搜索技术的突破，承诺在 RAG 集成方面优于历史悠久的 **BM25**，详见其 [announcement](https://x.com/qdrant_engine/status/1808498752107241949)。
  - 包括 **Jo Bergum** 在内的批评者质疑了 BM42 报告成功的完整性，认为这些主张不太可能，并引发了关于 Quora 数据集结果有效性的辩论。
- **VAEs 的烦恼与 AI 投资眼光**：关于理解 **Variational Autoencoders** 难度的幽默描述浮出水面，与之并列的是社区内关于卓越 AI **investment strategy** 的主张。
  - 一项严肃的预测推断，为了有效支持 **GDP growth**，AI 的贡献必须在 **11-15%** 之间，而社区仍在努力应对 **Anthropic Claude 3.5 Sonnet** 不透明的运作。
- **谷歌在全球 AI 挑战中的磨砺**：用户讨论了 **Google** 在生成式 AI 领域的缓慢起步，对其 Gemini web app 等产品的消息传递清晰度和方向表示担忧。
  - 围绕 **Google’s Gemini 1.5** 的定价模型和有效性展开了讨论，并将其与其他 AI 产品及 **Vertex AI** 等软件进行了比较，同时反思了 **First Amendment** 在 AI 领域的应用。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **API 队列系统的怪癖**：关于 **build.nvidia API** 问题的报告导致发现了一个新的 **queue system** 来管理请求，这标志着服务可能已过载。
  - 一位成员遇到了 **build.nvidia API 的脚本问题**，在临时停机后观察到功能恢复，暗示服务存在间歇性。
- **YAML 助力 Pipeline 进度**：一位成员分享了其 **pipeline** 集成 **YAML examples** 用于 few-shot learning 对话模型的情况，引发了对其在教科书数据上应用的兴趣。
  - 进一步澄清了基于 YAML 的结构如何促进 pipeline 中高效的 **few-shot learning** 过程。
- **Gemma2 获得稳定性**：**Gemma2** 的更新解决了过去的 bug。通过 **pinned version of transformers** 强化版本控制，确保了未来更平滑的更新。
  - **Continuous Integration (CI)** 工具因其在预先捕捉问题方面的作用而受到赞誉，为开发环境提供了抵御困扰的鲁棒性。
- **呼吁更多 VRAM**：来自 'le_mess' 的一条简洁但有力的消息强调了小组内长期的需求：请求更多的 **VRAM**。
  - 这一行简单的恳求反映了从业者对高性能计算资源持续增长的需求，对话中未作进一步阐述。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 中的 Tensor 问题**：讨论指出 **Tensor.randn** 和 **Tensor.randint** 会创建连续的 **Tensors**，而 `Tensor.full` 则会导致非连续结构，这引发了对与 [PyTorch 预期](https://pytorch.org/docs/stable/tensors.html) 不同的方法的审查。
  - 一位社区成员询问了 **tinygrad** 中 Bug 测试的放置位置，在 **test_nn** 或 **test_ops** 模块之间进行了讨论，最终决定在 **test_ops** 中编写一个高效且命名规范的测试。
- **训练的挑战与收获**：Tinygrad 用户对该框架的**大规模训练效率**表示担忧，称其运行缓慢且在经济上不切实际，同时考虑尽管 BEAM search 具有复杂性和时间需求，仍尝试使用它。
  - 围绕在 Tinygrad 中使用预训练 **PyTorch 模型**展开了对话，引导用户使用 `tinygrad.nn.state.torch_load` 进行有效的模型推理操作。
- **Matmul 大师课**：社区分享了一篇展示高性能矩阵乘法指南的博客文章，该指南在 CPU 上实现了超过 1 TFLOPS 的性能，详细介绍了实际实现方法和 [源代码](https://github.com/salykova/matmul.c)。
  - 分享内容包括指向该[博客文章的链接](https://salykova.github.io/matmul-cpu)，该文章将矩阵乘法分解为一个易于理解的 150 行 C 程序，引发了关于 **Tinygrad** 性能优化的讨论。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 的微调讨论**：社区成员就设置 **Torchtune** 的评估参数交换了意见，提到了一个潜在的“验证数据集”参数来调整性能。
  - 其他人对缺失 **wandb logging** 指标表示担忧，特别是 **evaluation loss** 和 **grad norm** 统计数据，强调了对更强大的指标追踪的需求。
- **Wandb 的烦恼与胜利**：讨论的一个话题是 **wandb** 的可视化能力，其中 **grad norm** 图表的缺失引发了对其与 aoxotl 等工具相比可用性的质疑。
  - 建议包括调整初始 **learning rate** 以影响 **loss curve**，但尽管进行了优化，一位成员指出 loss 没有显著改善，强调了参数微调的挑战。

---

## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **代码冲突：Python 遇见 TypeScript**：分享了在设置 **Convex** 平台时将 **Python** 与 **TypeScript** 集成的挑战经历。当 **Convex** 因缺少预安装的 Python 而出现启动 Bug 时，问题浮出水面。
  - 此外，讨论还围绕在 **Docker** 环境中自动安装 **Convex** 本地后端的困难展开，强调复杂性源于将特定容器文件夹配置为 volumes（卷）。
- **像素猎寻：寻找完美的精灵图**：一位成员探索了 **sprite sheets**（精灵图表）领域，表达了寻找与 **Cloudpunk** 游戏风格共鸣的视觉效果的目标，但发现从 **itch.io** 获取的资源缺乏理想的赛博朋克细微差别。
  - 他们正在寻找更符合 **Cloudpunk** 独特美学的精灵图资源，因为之前获取的资源未能反映出该游戏的标志性氛围。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **使用 GPT 三人组进行总结**：[三个 GPT 走进酒吧并撰写执行摘要](https://www.dylandavis.net/2024/07/three-gpts-walk-into-a-bar-and-write-an-exec-summary) 博客文章展示了一个由三个 **Custom GPTs** 组成的动态组合，旨在快速提取见解、起草和修改执行摘要。
  - 该工具包能够在紧迫的截止日期前生成简洁且相关的执行摘要，简化了交付简练且**有影响力的简报**的过程。
- **Magpie 在 HuggingFace 上的首飞**：Magpie 模型在 [HuggingFace Spaces](https://huggingface.co/spaces/sroecker/Elster) 上首次亮相，提供了一个生成偏好数据的工具，尽管它与 [davanstrien/magpie](https://huggingface.co/spaces/davanstrien/magpie) 存在重复。
  - **用户体验**显示仍有改进空间，反馈表明该模型的表现并不完全令人满意，但社区对其潜在应用保持**乐观**。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Build With Claude 活动即将到来**：工程爱好者们被号召参加 [Claude hackathon](https://docs.anthropic.com/en/build-with-claude-contest/overview)，这是一场将于下周结束的创意编程冲刺。
  - 参与者旨在利用 **Claude 的能力**构建创新解决方案，争取在闭幕竞赛中脱颖而出。
- **Kafka 成本削减会议**：一场定于 **IST 时间 7 月 18 日下午 4 点**举行的网络研讨会，承诺将分享关于[优化 Kafka](https://www.meetup.com/futureofdata-bangalore/events/301849238/?notificationId=1389017441959817216) 以提升性能并降低开支的见解。
  - **Yaniv Ben Hemo** 和 **Viktor Somogyi-Vass** 将主持讨论，重点关注 Kafka 设置中的扩展策略和效率。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **对职位术语的幽默调侃**：围绕该领域 **Embeddings 潜在用途**的讨论不断增加，引发了一些关于职位名称的幽默调侃。
  - 一位参与者开玩笑说要将自己改名为 *Embeddings AyEngineer*，为 AI 领域不断演变的命名法增添了幽默色彩。
- **职位闲谈变得流行**：随着 Embeddings 特定角色的兴起，出现了一个轻松的建议，即使用 **Embeddings Engineer** 这一头衔。
  - 这个幽默的提议强调了 **Embeddings** 在当前工程工作中的重要性以及社区的创造精神。

---

**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

**YAIG (a16z Infra) Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

# 第 2 部分：频道详细摘要与链接

{% if medium == 'web' %}

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1258521435433861150)** (1 条消息):

> - `VLM training dataset in Vietnamese`
> - `Highlights parser`
> - `See 2 sound demo`
> - `text2cypher model`
> - `Guide to Designing New Functional Proteins`

- **越南语 VLM 数据集发布**：由 [user](https://huggingface.co/datasets/Vi-VLM/Vista) 发布的越南语 **VLM 训练数据集**。该数据集现已供社区使用。
- **Highlights 解析器工具**：由 [user](https://huggingface.co/spaces/rrg92/hf-community-highlights-parser) 创建的 **Highlights 解析器工具**现已可用。它能帮助用户有效地解析社区亮点。
- **See 2 Sound 演示**：查看基于新发布论文的 **See 2 Sound 演示**，可在该 [Space](https://huggingface.co/spaces/rishitdagli/see-2-sound) 中找到。它提供了一种体验声音的创新方式。
- **Text2Cypher 模型表现优于 GPT-4**：用户发布的新 [text2cypher 模型](https://huggingface.co/lakkeo/stable-cypher-instruct-3b) 表现优于 **GPT-4**。该模型代表了 Text-to-Cypher 翻译领域的重大进步。
- **设计功能性蛋白质指南**：**设计新功能性蛋白质**并利用 Generative AI 对其进行改进的指南现已发布在[此处](https://huggingface.co/blog/AmelieSchreiber/protein-optimization-and-design)。该指南涵盖了蛋白质的功能、稳定性和多样性。

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1258139701903097867)** (495 条消息 🔥🔥🔥):

> - `Use of Deepeval with HuggingFace Transformers`
> - `Proficiency certifications in ML`
> - `Uploading image on HuggingFace projects using Gradio API`
> - `GPU recommendations for ML beginners`
> - `Issues with renting A100 vs. 4090 GPUs for inference`

- **ML 能力认证**：成员们讨论了用于验证 ML 技能的各种认证，更倾向于来自 Harvard 和 Coursera 等平台的免费选项。
- **面向 ML 初学者的 GPU 推荐**：用户在推荐 RTX 3060 还是 4060 之间进行了辩论，综合考虑了 VRAM 和性能，建议倾向于选择具有 12GB VRAM 的 3060。
- **租用 A100 与 4090 GPU 进行推理的问题**：讨论围绕租用 GPU 配置以实现高效的 ML 模型推理展开，建议指出 H100 比多个 4090 具有更好的性能。
- **使用 AI 模型创建视频**：聊天探讨了 text-to-video 生成 AI 模型，如 ipivs-morph-img2vid-animatediff-lcm-hyper-sd，并指出在标准设备上处理速度较慢但可行。
- **Stable Diffusion 模型许可更新**：**Stability AI 修订了 SD3 Medium 的许可**，以更好地支持开源社区，解决了之前关于商业使用限制的问题。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://swtokyo.com/">Startup Weekend Tokyo</a>：未找到描述</li><li><a href="https://www.runpod.io/serverless-gpu">用于 AI 推理的 Serverless GPU 端点</a>：使用 RunPod Serverless GPU 端点进行大规模机器学习推理。</li><li><a href="https://huggingface.co/blog/alvdansen/training-lora-m3lt">我如何训练 LoRA：m3lt 风格训练概览</a>：未找到描述</li><li><a href="https://lumalabs.ai/dream-machine">Luma Dream Machine</a>：Dream Machine 是来自 Luma AI 的一款 AI 模型，可以根据文本和图像快速生成高质量、逼真的视频。</li><li><a href="https://tenor.com/view/happyfourthofjuly-july4th-gif-22215151">Happyfourthofjuly July4th GIF - Happyfourthofjuly July4th - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.instagram.com/p/C8luO4VM3l1/">ERLAX 在 Instagram 上："…</a><p><a href="https://www.instagram.com/p/C8luO4VM3l1/">#techno #dreamcore #rave #digitalart #aiart #stablediffusion"</a>：2,738 个赞，151 条评论 - erlax.case 于 2024 年 6 月 24 日："… #techno #dreamcore #rave #digitalart #aiart #stablediffusion"。</p></li><li><a href="https://huggingface.co/artificialguybr/doodle-redmond-doodle-hand-drawing-style-lora-for-sd-xl">artificialguybr/doodle-redmond-doodle-hand-drawing-style-lora-for-sd-xl · Hugging Face</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/13v3b6q/multiple_cheap_gpus_or_a_single_expensive_one/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb#scrollTo=r5PM6vOQPISl">Google Colab</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/InstantX/InstantStyle">InstantStyle - 由 InstantX 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://x.com/p4ino/status/1808560882189803931">来自 lilbotomy☆ (@p4ino) 的推文</a>：这就是我讲故事的方式</li><li><a href="https://github.com/nroggendorff/diffusion/blob/main/zelda.ipynb">diffusion/zelda.ipynb 分支 main · nroggendorff/diffusion</a>：通过在 GitHub 上创建账号，为 nroggendorff/diffusion 的开发做出贡献。</li><li><a href="https://huggingface.co/internlm/internlm2_5-7b-chat-1m">internlm/internlm2_5-7b-chat-1m · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/p4">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://tenor.com/view/i-just-work-here-idk-idk-about-that-i-don%27t-know-gif-3168423486813006711">我只是在这里工作 Idk GIF - 我只是在这里工作 Idk Idk about that - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://stability.ai/news/license-update">社区许可证 — Stability AI</a>：我们全新的社区许可证现在对研究、非商业和商业用途免费。只有当您的年收入超过 100 万美元且在...中使用 Stability AI 模型时，才需要付费的企业许可证。</li><li><a href="https://huggingface.co/spaces/aheedsajid/Edge-TTS/discussions/1#6685a353d8e85b570562e2c6">aheedsajid/Edge-TTS · 🚩 举报：垃圾信息</a>：未找到描述</li><li><a href="https://tenor.com/view/happy-tree-friends-htf-cuddles-giggles-flaky-gif-5696779679679953568">Happy Tree Friends Htf GIF - Happy tree friends Htf Cuddles - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/spaces/Nick088/Stable_Diffusion_Finetuned_Minecraft_Skin_Generator/discussions/4#667fd0173a46eeac17c80179">Nick088/Stable_Diffusion_Finetuned_Minecraft_Skin_Generator · 🚩 举报：垃圾信息</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/Nick088/SDXL-Flash/discussions/1#667fd1079f4f7654f000f465">Nick088/SDXL-Flash · 需要更好的版本</a>：未找到描述</li><li><a href="https://www.reddit.com/r/Stable">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1dq4y7r/how_are_videos_like_these_created/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://www.dylandavis.net/2024/07/three-gpts-walk-into-a-bar-and-write-an-exec-summary/">三个 GPT 走进酒吧并写了一份执行摘要 – D-Squared</a>：未找到描述<p></p></li></ul></div>

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1258349133480071208)** (2 messages):

> - `构建用于有害内容分类的 TikTok 视频数据集`
> - `排查 RGB 图像的 LDM 实现问题`

- **用于分类有害内容的 TikTok 数据集**：一位用户分享了一个 TikTok 视频数据集，大小为 **30 GB**，包含约 **3,000 个视频**，旨在构建一个视频分类模型来识别针对儿童的有害内容。他们还提供了一个 [notebook](https://www.kaggle.com/code/anhoangvo/how-to-use-hugging-face-for-fine-tuning-on-the-tik) 用于在该数据集上微调 Hugging Face 模型。
- **LDM 模型排查**：一位用户正在学习使用 **Flax** 库从头开始创建 LDM。在 MNIST 数据集上取得了成功，但在处理来自 **imagenette/160px-v2** 的 RGB 图像时遇到问题。他们请求排查建议，因为他们的模型在处理 RGB 图像时仅生成色块。

<div class="linksMentioned"><p><strong>提及的链接</strong>：</p><ul><li><a href="https://www.kaggle.com/datasets/anhoangvo/tikharm-dataset/">TikHarm Dataset</a>：一个用于训练模型分类有害内容的 TikTok 视频数据集。</li><li><a href="https://www.kaggle.com/code/anhoangvo/how-to-use-hugging-face-for-fine-tuning-on-the-tik">如何在 TikTok 上使用 Hugging Face 进行微调</a>：通过 Kaggle Notebooks 探索并运行机器学习代码 | 使用来自 TikHarm Dataset 的数据</li></ul></div>

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1258225348802248796)** (6 messages):

> - `Kyutai.org 的数字海盗能听懂带法国口音的英语`
> - `音频语言模型 Moshi 的小型演示`
> - `使用 GraphEdit 和大语言模型进行图结构学习 (GSL)`
> - `Claude 在构建深度学习可视化仪表板方面的便捷性`
> - `nanoLLaVA - 1B 以下的酷炫 VLM`

- **Kyutai 的数字海盗精通语言**：来自 [Yann LeCun](https://x.com/ylecun/status/1808573888642617406) 的一条推文透露，**Kyutai.org 的数字海盗**可以理解带有**法国口音**的英语。这是由 **Moshi** 项目的 Neil Zegh 在一个**小型演示**中展示的。
- **GraphEdit 突破 GSL 边界**：论文 [GraphEdit](https://arxiv.org/abs/2402.15183) 提出了一种新的**图结构学习 (Graph Structure Learning)** 方法，利用**大语言模型 (LLMs)** 通过对图结构进行指令微调来**增强可靠性**。
- **nanoLLaVA 备受关注**：Hugging Face Space [nanoLLaVA](https://huggingface.co/spaces/qnguyen3/nanoLLaVA) 被强调为一个参数量在 **10 亿 (1B)** 以下的**酷炫视觉语言模型 (VLM)**。它因其**出色的可视化能力**而受到关注。

<div class="linksMentioned"><p><strong>提及的链接</strong>：</p><ul><li><a href="https://x.com/ylecun/status/1808573888642617406">Yann LeCun (@ylecun) 的推文</a>：我们了解到 http://Kyutai.org 的数字海盗能听懂带法国口音的英语。引用 Guillaume Grallet (@guillaumgrallet) 的话：来自 #moshi 的 @neilzegh 的一个小演示，这是一个音频语言模型...</li><li><a href="https://huggingface.co/spaces/qnguyen3/nanoLLaVA">nanoLLaVA-1.5 - qnguyen3 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2402.15183">GraphEdit: 用于图结构学习的大语言模型</a>：图结构学习 (GSL) 专注于通过生成新的图结构来捕获图结构数据中节点之间的内在依赖和交互。图神经网络 (GNNs) 已经...</li></ul></div>

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1258155599452049538)** (32 条消息🔥):

> - `Vi-VLM 团队推出越南语视觉语言模型`
> - `Vi-VLM 发布用于越南语 VLM 训练的数据集`
> - `将消息转换为 pt-br 的简单翻译工具`
> - `针对 Transformer 的 CyclicFormer 架构增强`
> - `用于音频分离的 UVR5 UI 完成`

- **Vi-VLM 推出越南语视觉语言模型**：Vi-VLM 团队推出了一个针对越南语的视觉语言模型，该模型基于 LLaVA 和 Vistral 构建，专注于图像描述；演示和代码[在此查看](https://huggingface.co/Vi-VLM/Vistral-V-7B)。
- **CyclicFormer 增强 Transformer 性能**：CyclicFormer 架构在解码器层之间引入了循环回路，以增强 Transformer 的性能，[GitHub 链接在此](https://github.com/LegallyCoder/CyclicFormer)。
- **使用 Lightly SSL 的端到端图像检索应用**：使用来自 Hub 的任意图像数据集构建了一个图像检索应用，利用 FAISS 进行向量索引，并使用 Lightly SSL 进行自监督学习，详情见[博客文章](https://www.lightly.ai/post/vector-indexes-and-image-retrieval-using-lightly)。
  - *查看 [Gradio 应用](https://huggingface.co/spaces/lightly-ai/food101-image-retrieval)进行实际演示。*
- **用于音频分离的 UVR5 UI 已完成**：UVR5 的 UI 现已完成，可以轻松分离人声和乐器轨道；它使用了通过 [Gradio](https://huggingface.co/spaces/TheStinger/UVR5_UI) 提供的先进音频分离模型。
  - *在各种测试中实现了人声和旋律的完美分离，包括 1987 年的流行歌曲《Faroeste Caboclo》。*
- **针对 pt-br 的简单翻译工具**：创建了一个将社区亮点翻译成 pt-br 的工具，有助于更快地导入消息；[在此查看工具](https://huggingface.co/spaces/rrg92/hf-community-highlights-parser)。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://huggingface.co/rishitdagli/see-2-sound">rishitdagli/see-2-sound · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/rrg92/hf-community-highlights-parser">Highs Parser - rrg92 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/TheStinger/UVR5_UI">UVR5 UI - TheStinger 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://github.com/LegallyCoder/CyclicFormer">GitHub - LegallyCoder/CyclicFormer: CyclicFormer 是一种旨在增强 Transformer 架构性能的新架构。它为解码器层引入了新的视角，在所有层之间形成了一个循环回路。</a></li><li><a href="https://huggingface.co/datasets/Vi-VLM/Vista">Vi-VLM/Vista · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/Vi-VLM/Vistral-V-7B">Vi-VLM/Vistral-V-7B · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/hllj/Vistral-V">GitHub - hllj/Vistral-V: Vistral-V：针对 Vistral 的视觉指令微调 - 越南语大型视觉语言模型。</a></li><li><a href="https://www.lightly.ai/post/vector-indexes-and-image-retrieval-using-lightly">使用 lightly 进行向量索引和图像检索</a>：使用 Lightly 提供的预训练 Vision Transformer，在任意数据集上创建向量索引，以便使用 FAISS 进行图像检索。</li><li><a href="https://huggingface.co/spaces/lightly-ai/food101-image-retrieval">Food101 Image Retrieval - lightly-ai 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://x.com/MaheshkarSaurav/status/1808881869829853305">Saurav Maheshkar ☕️ (@MaheshkarSaurav) 的推文</a>：🚀 @LightlyAI 的最新工作。了解如何使用 FAISS (@AIatMeta) 作为向量索引 🗃️、来自 Lightly SSL 软件包的模型实现以及 @weights_biases 来创建图像检索应用...</li><li><a href="https://colab.research.google.com/drive/1n4CwX5T6Ch2v7OYTRe6g1j_QJHxxOvcM?usp=sharing">Google Colab</a>：未找到描述</li></ul></div>

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1258614538463477921)** (7 条消息):

> - `triton paper reading` (Triton 论文阅读)
> - `upcoming paper reading schedule` (即将到来的论文阅读日程)
> - `interest in audio-language models` (对音频语言模型的兴趣)
> - `flora paper discussion` (Flora 论文讨论)

- **即将进行的 Triton 论文阅读**：一位成员因忙碌为推迟原定的 **Triton** 论文阅读计划表示抱歉，并邀请有兴趣的其他成员进行分享。鼓励参与者联系另一位成员以获取更多信息。
- **Flora 论文引起关注**：一位成员对 **Flora** 论文表达了兴趣，称其非常酷。该论文似乎在即将到来的讨论中获得了关注。

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1258329870401011745)** (4 条消息):

> - `WHAM alternatives for human pose estimation in monocular, in-the-wild videos` (单目、野外视频中人体姿态估计的 WHAM 替代方案)
> - `Learning ViT and U-Net implementations` (学习 ViT 和 U-Net 的实现)
> - `Using visual-semantic information to boost fine-grained image classification performance` (利用视觉语义信息提升细粒度图像分类性能)
> - `Discussing zero/few shot multi-modal models at CVPR` (在 CVPR 讨论零样本/少样本多模态模型)

- **寻找用于摔跤动画的 WHAM 替代方案**：一位非编程人员正在寻找一种机器学习方法，用于处理巴西柔术等**复杂人体交互**的单目、野外视频中的**人体姿态估计 (human pose estimation)**。由于 WHAM 具有**复杂的 Python 和 CV 依赖关系**，他们在运行上遇到了困难，正在寻求更用户友好的替代方案。
- **通过在线资源学习 ViT 和 U-Net**：一位成员分享了一个[链接](https://www.learnpytorch.io/08_pytorch_paper_replicating/)，用于通过 **Andrew Ng 的深度学习专项课程 (DL Specialization)** 和 **CNN 课程第三周**的内容来学习 **ViT** 和 **U-Net** 的实现。
- **利用视觉语义信息提升图像分类**：另一位用户询问如何利用来自标题/元数据的**视觉语义信息**来增强**细粒度图像分类性能**，而不仅仅局限于零样本/少样本学习。Florence 2 被建议作为此类特定监督微调 (supervised fine-tuning) 的潜在模型。

**提到的链接**：[08\. PyTorch Paper Replicating - Zero to Mastery Learn PyTorch for Deep Learning](https://www.learnpytorch.io/08_pytorch_paper_replicating/)：通过编写 PyTorch 代码动手学习重要的机器学习概念。

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1258136650824028261)** (17 条消息🔥):

> - `Meta-LLaMA download issues` (Meta-LLaMA 下载问题)
> - `API calls to models without local download` (无需本地下载的模型 API 调用)
> - `Inference freeze in Mistral model` (Mistral 模型推理冻结)
> - `Static KV cache documentation` (静态 KV cache 文档)
> - `Troubleshooting errors related to memory` (内存相关错误排查)

- **Meta-LLaMA 下载困扰**：一位用户对 **Meta-LLaMA** 下载耗时过长表示沮丧，并担心潜在的临时文件会填满硬盘。
- **API 调用困惑**：关于是否可以在不进行本地下载的情况下构建对模型的 API 调用存在困惑，并对这种方法的可行性提出了质疑。
- **Mistral 模型在第 1800 次迭代时冻结**：**Mistral** 在 3000 次推理运行中的第 1800 次迭代时发生冻结，而在 100 次推理时运行正常，这引发了对某种缓存问题的怀疑。
- **静态 KV cache 引起混淆**：一位用户指出，自 4.41 版本以来，静态 **KV cache** 默认开启，建议查看[相关发布说明](https://github.com/huggingface/transformers/releases/tag/v4.38.0)以获取更多细节。
- **TypedStorage 弃用担忧**：用户对 **TypedStorage** 被弃用表示担忧，建议在进行任何代码更改之前等待稳定的解决方案。

**提到的链接**：[Release v4.38: Gemma, Depth Anything, Stable LM; Static Cache, HF Quantizer, AQLM · huggingface/transformers](https://github.com/huggingface/transformers/releases/tag/v4.38.0)：新增模型 💎 Gemma 💎 Gemma 是 Google AI 推出的一系列新型开源语言模型，包含 2B 和 7B 版本。该版本包含预训练和指令微调版本...

---

### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1258165680478621706)** (3 messages):

> - `使用 diffusers 运行 RealVisXL_V4.0_Lightning`
> - `在 Google Colab 中使用 yisol/IDM-VTON 时报错`
> - `改进简历分析器以评估项目强度`

- ****RealVisXL V4.0 Lightning 模型发布****: [RealVisXL V4.0 Lightning](https://huggingface.co/SG161222/RealVisXL_V4.0_Lightning) 正在训练中，支持 sfw 和 nsfw 类别的写实图像。用户可以在 [Boosty](https://boosty.to/sg_161222) 上支持创作者，并在[此处](https://civitai.com/models/139562/realvisxl-v40)找到 CivitAI 页面。
- ****Diffusers 质量不及 A1111****: 有用户报告称，RealVisXL V4.0 模型在 A1111 中表现良好，但在使用相同参数的情况下，使用 diffusers 生成的图像质量较差。
- ****Google Colab 中 IDM-VTON 报错****: 用户在 Google Colab 上使用 yisol/IDM-VTON 时遇到 'no file named diffusion_pytorch_model.bin' 错误。
- ****超越关键词增强简历分析器****: 用户正在寻求关于创建简历分析器的建议，该分析器应评估项目强度而不仅仅是匹配关键词。他们的目标是区分低复杂度的任务和更重大的项目。

**提及的链接**: [SG161222/RealVisXL_V4.0_Lightning · Hugging Face](https://huggingface.co/SG161222/RealVisXL_V4.0_Lightning): 未找到描述

---

### **Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1258830278483382343)** (1 messages):

> - `SD3 Medium 发布相关的许可证问题`
> - `Stability AI Community License 更新`
> - `先前版本中商业许可的问题`
> - `对开源社区的改进和支持`

- **Stability AI 更新许可证以扩大使用范围**: Stability AI 承认其 **SD3 Medium** 的发布未达到社区预期，且相关的商业许可证引起了困惑。他们已经为个人创作者和小微企业修订了许可证，涵盖在新的 **Stability AI Community License** 之下，[在此阅读完整更新](http://stability.ai/news/license-update)。
- **新 Stability AI 许可证下非商业用途免费**: 在新的 Stability AI Community License 下，**非商业用途保持免费**。这一变化通过提供对包括 *SD3 Medium* 在内的近期发布版本的更广泛访问，支持了开源社区。

**提及的链接**: [Community License — Stability AI](http://stability.ai/news/license-update): 我们的新 Community License 现在对研究、非商业和商业用途免费。只有当您的年收入超过 100 万美元且在...中使用 Stability AI 模型时，才需要付费的 Enterprise 许可证。

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1258163876957126717)** (528 条消息🔥🔥🔥):

> - `Hyper vs turbo`
> - `AAM Anime Mix XL`
> - `Animagine XL 3.1`
> - `Stable Diffusion GPU usage`
> - `CivitAI and SD3 discussions`

- **Hyper 是新的 Turbo：Animagine XL 3.1 更新**：用户讨论了动漫主题模型 **Animagine XL 3.1** 的优点。该模型在 **Animagine XL 3.0** 的基础上进行了改进，提供了更高质量的图像，并扩大了来自知名动漫系列的角色范围，由 [Cagliostro Research Lab](https://huggingface.co/cagliostrolab) 和 [SeaArt.ai](https://www.seaart.ai/) 开发。
- **AAM Anime Mix XL 受到关注**：一位用户分享了对 **AAM Anime Mix XL**（另一个流行的动漫图像生成模型）的热情。这引发了对 Animagine XL 3.1 等相关模型的比较和推荐。
- **多 GPU 配置的难题**：用户讨论了使用多 GPU 设置来提高 Stable Diffusion 速度和输出质量的挑战及潜在解决方案。**SwarmUI** 等特定工具因其处理多 GPU 操作的能力而受到关注。
- **CivitAI 对 SD3 的禁令引发辩论**：社区对 **CivitAI** 禁止 SD3 模型的举动反应不一。许多人表示此举可能会阻碍 **SD3** 的发展，而其他人则讨论了围绕该模型的各种技术和许可问题。
- **Stable Diffusion 许可与模型更新**：对话中包含了对 **Stable Diffusion 3** 及其新模型 **license**（许可）的担忧。关于许可条款是否过于严格、从而影响中小型及大型企业用户的争论也随之展开。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://civitai.com/articles/6024/using-a1111-why-not-swarmui-a-transition-guide">Using A1111? Why not SwarmUI? - A transition guide | Civitai</a>：我最近从 Forge 迁移到了 SwarmUI（以前称为 StableSwarmUI），我很高兴我这么做了！我之前尝试过它...</li><li><a href="https://youtu.be/6Q4BJOcvwGE?si=LdajWOtf4iTKGVWJ&amp;t=844">SegMoE - The Stable Diffusion Mixture of Experts for Image Generation!</a>：混合专家模型。这在 AI 文本生成领域很火... 但如果你在图像生成中也使用混合专家模型呢？哦，Segmind 刚刚做到了。欢迎来到...</li><li><a href="https://www.youtube.com/watch?v=XtMvk0dpnO4&amp;list=PLNlRhPQovztRqp_zyp-lY79fWZIzjnNTf">How to Make Concept Art with AI (Free and Easy) - Stable Diffusion Tutorial 2022</a>：注意！自从我制作这个视频以来，很多事情都变得更好了！这是我在 2023 年 6 月安装和使用 Stable Diffusion 的指南：https://youtu.be/nB...</li><li><a href="https://huggingface.co/cagliostrolab/animagine-xl-3.1">cagliostrolab/animagine-xl-3.1 · Hugging Face</a>：未找到描述</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1do5gvz/the_open_model_initiative_invoke_comfy_org/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://github.com/vladmandic/automatic/">GitHub - vladmandic/automatic: SD.Next: Advanced Implementation of Stable Diffusion and other Diffusion-based generative image models</a>：SD.Next：Stable Diffusion 及其他基于扩散的生成式图像模型的高级实现 - vladmandic/automatic</li><li><a href="https://github.com/ltdrdata/ComfyUI-Manager">GitHub - ltdrdata/ComfyUI-Manager: ComfyUI-Manager is an extension designed to enhance the usability of ComfyUI. It offers management functions to install, remove, disable, and enable various custom nodes of ComfyUI. Furthermore, this extension provides a hub feature and convenience functions to access a wide range of information within ComfyUI.</a>：ComfyUI-Manager 是一个旨在增强 ComfyUI 易用性的扩展。它提供了安装、删除、禁用和启用 ComfyUI 各种自定义节点的管理功能。此外，该扩展还提供了一个 Hub 功能和便捷功能，用于访问 ComfyUI 内部的广泛信息。</li><li><a href="https://huggingface.co/models?search=sdxl%20controlnet%20tile">Models - Hugging Face</a>：未找到描述</li><li><a href="https://poe.com/PhdExpert-CDvr4">PhdExpert-CDvr4 - Poe</a>：输入你想要的语言。[期待顶级的回答]</li><li><a href="https://github.com/kijai/ComfyUI-LivePortrait?tab=readme-ov-file">GitHub - kijai/ComfyUI-LivePortraitKJ: ComfyUI nodes for LivePortrait</a>：用于 LivePortrait 的 ComfyUI 节点。通过在 GitHub 上创建一个账户来为 kijai/ComfyUI-LivePortraitKJ 的开发做出贡献。</li><li><a href="https://huggingface.co/ptx0">ptx0 (PseudoTerminal X)</a>：未找到描述</li><li><a href="https://www.dylandavis.net/2024/07/three-gpts-walk-into-a-bar-and-write-an-exec-summary/">Three GPTs Walk into a Bar and Write an Exec Summary – D-Squared</a>：未找到描述</li></ul></div>

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1258137476757852291)** (267 条消息🔥🔥):

> - `Gemma 2 发布及其特性`
> - `Gemma 2 notebook 的问题及用户反馈`
> - `数据集准备及处理长上下文示例的方法`
> - `各种 LLM 的性能和优化技术`
> - `AI 模型和工具的最新进展及公告`

- ****Gemma 2 发布带来速度和 VRAM 提升****：**Gemma 2 发布**现已推出，声称与 Flash Attention 2 相比，**微调速度快 2 倍**，且 **VRAM 占用减少 63%** ([Gemma 2 博客](https://unsloth.ai/blog/gemma2))。关键细节包括 Unsloth 支持 **高达 9.7K 的上下文长度**。
  - “老实说，博文写得非常仓促 <slothlaughcry> 我已经发现了一些错误，”一位社区成员指出，强调了发布的节奏非常快。</slothlaughcry>
- ****Unsloth notebook 和模型目录问题****：**用户报告了** **Gemma 2 notebook** 的问题，特别是与模型目录命名和配置缺失相关的错误（例如，应该是 `unsloth_gemma` 却写成了 `unsloth/gemma`）。开发者进行了**协作并快速修复**以解决这些问题。
- ****长上下文示例训练和数据集准备技术****：成员们讨论了处理**长上下文数据集**的技术，部分示例达到了 **78,451 个 token**。建议包括设置合适的上下文长度，以及使用**特定函数来查找数据集中的最大 token 数**。
  - 分享函数和讨论 **Prompt Engineering** 方法是常见的主题。分享了一些实用建议，如“你可以在指令部分选择语气”，以帮助用户更好地格式化数据进行模型训练。
- ****缺乏 Flash Attention 支持时的 Gemma 2 性能和局限性****：在没有 Flash Attention 支持的情况下，据报告 **Gemma 2** 模型明显变慢，几乎**无法用于密集型任务**。这突显了优化的注意力机制对模型性能的重大影响。
  - 社区成员建议 **gradacc (gradient accumulation)** 可能比传统的 batching 更有效，有人指出：“如果非要说的话，gradacc 更快。”
- ****新 AI 模型和工具公告****：Nomic AI 发布了 **GPT4ALL 3.0**，这是一款全新的开源本地 LLM 桌面应用，强调隐私和本地数据处理 ([GPT4ALL 3.0 公告](https://home.nomic.ai/gpt4all))。它因支持数千个模型和主流操作系统而受到赞誉。
  - 还提到了 InternLM-XComposer-2.5，强调了其支持**长上下文输入和输出**的能力，仅凭 7B LLM 后端就达到了 GPT-4V 级别的性能 ([InternLM-XComposer-2.5](https://huggingface.co/internlm/internlm-xcomposer2d5-7b))。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://unsloth.ai/blog/gemma2">使用 Unsloth 微调 Gemma 2</a>：通过 Unsloth 以 2 倍的速度和减少 63% 的 VRAM 显存微调 Google 的新 Gemma 2 模型！支持 9B 和 27B 参数。</li><li><a href="https://youtu.be/ZJKglSWgD0w?si=20kiqxXIPvelywyJ">AI 中的情感：微调、分类和强化学习</a>：在本视频中，我们将探索如何使用 Unsloth 和 Ollama 创建 LLM 的微调数据集，以训练一个专门用于情感检测的模型。你...</li><li><a href="https://huggingface.co/mlx-community/Phi-3-mini-4k-instruct-8bit">mlx-community/Phi-3-mini-4k-instruct-8bit · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/danielhanchen/status/1808622550467162219">Daniel Han (@danielhanchen) 的推文</a>：使用 @UnslothAI，Gemma 2 的微调现在快了 2 倍，且 VRAM 占用减少了 63%！1. 我们修复了官方 Gemma 仓库中的 2 个问题 2. 27B 的 Softcapping 必须在 attn 和 logits 上进行，否则 losses 会发散。9...</li><li><a href="https://github.com/unslothai/unsloth/blob/9b4cc934efec66abd0a77df011779b393a99c026/unsloth/models/llama.py#L1175-L1179">unsloth/unsloth/models/llama.py (位于 9b4cc934efec66abd0a77df011779b393a99c026) · unslothai/unsloth</a>：以 2-5 倍的速度和减少 80% 的内存微调 Llama 3, Mistral, Phi 和 Gemma LLMs - unslothai/unsloth</li><li><a href="https://github.com/b4rtaz/distributed-llama">GitHub - b4rtaz/distributed-llama: Tensor parallelism is all you need. Run LLMs on weak devices or make powerful devices even more powerful by distributing the workload and dividing the RAM usage.</a>：Tensor parallelism is all you need。通过分布式工作负载和划分 RAM 使用，在弱设备上运行 LLMs，或让高性能设备更强大。- b4rtaz/distributed-llama</li><li><a href="https://x.com/nomic_ai/status/1808162955806097767">Nomic AI (@nomic_ai) 的推文</a>：发布 GPT4All 3.0：开源本地 LLM 桌面应用 - 完全私密的体验 - 支持数千种模型和所有主流操作系统 - 重大 UI/UX 改进 - 本地文件聊天 -...</li><li><a href="https://home.nomic.ai/gpt4all">GPT4All</a>：在本地运行大型语言模型：隐私优先且无需联网</li><li><a href="https://github.com/google/gemma_pytorch/pull/67">由 danielhanchen 修复的 downcasting 和 upcasting · Pull Request #67 · google/gemma_pytorch</a>：修复了 RMS Layernorm 过早 downcasting 的问题。我们将其移至最后。修复了 embedding matrix scaling / normalizer upcasting 至 float32 的问题。相反，我们必须对 normalizer 使用 float16 或 bfloat16...</li><li><a href="https://tenor.com/view/baby-face-palm-really-sigh-stupid-gif-12738431">婴儿捂脸 GIF - Baby Face Palm Really - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>：以 2-5 倍的速度和减少 80% 的内存微调 Llama 3, Mistral, Phi 和 Gemma LLMs - unslothai/unsloth</li><li><a href="https://x.com/_akhaliq/status/1808747694317261114">AK (@_akhaliq) 的推文</a>：InternLM-XComposer-2.5 一个支持长上下文输入和输出的多功能大型视觉语言模型。我们推出了 InternLM-XComposer-2.5 (IXC-2.5)，这是一款多功能大型视觉语言模型，支持...</li><li><a href="https://huggingface.co/internlm/internlm-xcomposer2d5-7b">internlm/internlm-xcomposer2d5-7b · Hugging Face</a>：未找到描述</li><li><a href="https://hqjiang.com/minference.html">MInference：针对 LLMs 的百万级 Token 提示词推理</a>：未找到描述</li><li><a href="https://research.nvidia.com/labs/toronto-ai/FMS/">Forecasting Model Search</a>：未找到描述</li></ul></div>

---

### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1258151227787706450)** (1 条消息):

> - `Gemma 2 Release`
> - `Training speed and VRAM reduction`
> - `Context length improvements`
> - `4-bit model support updates`
> - `Experimentation with models`

- ****Gemma 2 加速微调****：Unsloth 现在支持 **Gemma 2**，实现 **2倍训练速度** 并减少 **63% 的显存占用**。查看 [Gemma 2 Blog](https://unsloth.ai/blog/gemma2) 了解更多详情。
- ****上下文长度显著提升****：使用 Unsloth，你现在可以在 40GB GPU 上微调 **Gemma 2 (27B)**，支持 **9.7K 上下文长度**，而使用 HF+FA2 仅为 3K。**9B 模型** 在 24GB 显卡上可达到 **11K 上下文长度**，而 HF+FA2 仅为 2.6K。
- ****提供新的免费 Notebook****：访问 [Gemma 2 (9B) Colab notebook](https://colab.research.google.com/drive/1vIrqH5uYDQwsJ4-OO3DErvuv4pBgVwk4) 即可开始体验最新模型。同时已添加对 **Gemma 2 (27B)** notebook 的支持。
- ****现已支持 4-bit 模型****：探索新的 4-bit 模型：[Gemma 2 (9B) Base](https://huggingface.co/unsloth/gemma-2-9b-bnb-4bit)、[Gemma 2 (9B) Instruct](https://huggingface.co/unsloth/gemma-2-9b-it-bnb-4bit)、[Gemma 2 (27B) Base](https://huggingface.co/unsloth/gemma-2-27b-bnb-4bit) 以及 [Gemma 2 (27B) Instruct](https://huggingface.co/unsloth/gemma-2-27b-it-bnb-4bit)。**Phi 3 mini** 的更新也已在 HF 上线。
- ****呼吁社区实验****：Unsloth 鼓励用户在社区频道中分享、测试和讨论他们的 **模型与结果**。加入讨论并尝试最新的更新。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://unsloth.ai/blog/gemma2">使用 Unsloth 微调 Gemma 2</a>：通过 Unsloth 以 2 倍速、减少 63% VRAM 显存占用的方式微调 Google 的新 Gemma 2 模型！包含 9B 和 27B 参数版本。</li><li><a href="https://colab.research.google.com/drive/1vIrqH5uYDQwsJ4-OO3DErvuv4pBgVwk4?usp=sharing)">Google Colab</a>：未找到描述</li></ul></div>

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1258172763252850719)** (7 条消息):

> - `Release of Replete-AI datasets`
> - `Discussion on Facebook multi-token prediction`
> - `Fireworks.ai yi-large issues`

- ****Replete-AI 发布海量数据集****：Replete-AI 宣布发布 [两个新数据集](https://huggingface.co/datasets/Replete-AI/Everything_Instruct)，每个大小约为 11-12GB，包含超过 600 万行数据。数据集包括纯英文版本和多语言版本，旨在训练通用的 AI 模型。
- ****Facebook 的 Multi-Token Prediction 值得吗？****：关于 [Facebook 的 multi-token prediction 模型](https://huggingface.co/facebook/multi-token-prediction) 是否值得尝试引发了讨论，该模型需要分享联系信息才能访问。一位成员表示怀疑，而另一位成员认为尽管有 Facebook 的参与，它仍然值得一试。
- ****Fireworks.ai yi-large 让用户失望****：用户反馈了在 Fireworks.ai 上使用 yi-large 模型的挫败感。一位用户承认被该模型“忽悠”了（jebaited），表示其表现未达预期。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://huggingface.co/facebook/multi-token-prediction">facebook/multi-token-prediction · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/Replete-AI/Everything_Instruct">Replete-AI/Everything_Instruct · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/Replete-AI/Everything_Instruct_Multilingual">Replete-AI/Everything_Instruct_Multilingual · Hugging Face 数据集</a>：未找到描述</li></ul></div>

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1258142841218011297)** (121 条消息🔥🔥):

> - `Unsloth 2024.7 补丁与 checkpoints 的问题`
> - `Unsloth 对 Gemma 2 的支持`
> - `使用 Unsloth 微调模型`
> - `微调和评估过程中的错误`
> - `更新 Unsloth 及 GGUF 相关问题`

- ****Unsloth 宣布支持 Gemma 2！****：Unsloth 已添加对 **Gemma 2** 的支持；你现在可以更新并尝试最新补丁 [2024.7](https://github.com/unslothai/unsloth) 中的新功能。
- ****Unsloth 2024.7 补丁中的 Checkpoint 训练错误****：用户报告在 **Unsloth 2024.7** 补丁中从 checkpoint 恢复训练时出现 `RuntimeError: Expected all tensors to be on the same device` 等错误。一些人建议回退到旧版本，但问题仍然存在，需要进一步调查。
- ****Unsloth 微调陷阱****：一些用户在不使用 LoRA 的情况下微调 **Gemma 1.1** 和 **Phi-3 mini** 模型时遇到问题；这对 Phi-3 有效，但在对 Gemma 1.1 进行全量微调（full fine-tuning）时会报错。
- ****特定模型和配置的错误****：在处理 **Gemma-2-27B-bnb-4bit** 等大模型时遇到了各种错误，例如 `RuntimeError: The size of tensor a (4096) must match the size of tensor b (4608)`，以及在特定指标评估期间注意到的潜在 VRAM 问题。
- ****更新 Unsloth 并处理 GGUF 问题****：指导用户通过 wiki 更新 Unsloth；一些用户由于 GGUF 量化问题在将微调后的模型推送到 Hugging Face 时面临错误，根据开发者的更新，这些问题现已修复。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://huggingface.co/mlx-community/Phi-3-mini-4k-instruct-8bit">mlx-community/Phi-3-mini-4k-instruct-8bit · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/main/tokenizer_config.json">tokenizer_config.json · microsoft/Phi-3-mini-128k-instruct at main</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki">主页</a>：微调 Llama 3, Mistral, Phi &amp; Gemma LLM 快 2-5 倍，节省 80% 显存 - unslothai/unsloth</li><li><a href="https://discuss.huggingface.co/t/adding-accuracy-precision-recall-and-f1-score-metrics-during-training/16419/2">在训练期间添加准确率、精确率、召回率和 F1 分数指标</a>：你好，你可以定义你的计算指标函数并将其传递给 trainer。这是一个计算指标的示例。从 sklearn.metrics 导入 accuracy_score 定义准确率指标函数...</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: C/C++ 实现的 LLM 推理</a>：C/C++ 实现的 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 做出贡献。</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 微调 Llama 3, Mistral, Phi &amp; Gemma LLM 快 2-5 倍，节省 80% 显存</a>：微调 Llama 3, Mistral, Phi &amp; Gemma LLM 快 2-5 倍，节省 80% 显存 - unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki#evaluation-loop---also-oom-or-crashing">主页</a>：微调 Llama 3, Mistral, Phi &amp; Gemma LLM 快 2-5 倍，节省 80% 显存 - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/pull/671">Ollama 由 danielhanchen 提交 · Pull Request #671 · unslothai/unsloth</a>：未找到描述</li></ul></div>

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1258172565105541152)** (3 messages):

> - `Replete-AI 发布了两个新的海量数据集`
> - `Everything_Instruct_Multilingual 的细节与翻译`
> - `关于数据集去重和内容平衡的疑问`

- **Replete-AI 发布海量 instruct 数据集**：Replete-AI 发布了两个新数据集：**Everything_Instruct** 和 **Everything_Instruct_Multilingual**，每个数据集大小为 11-12GB，包含超过 600 万行数据。这些数据集结合了多种类型的 instruct 数据，旨在训练英文和多语言版本的高级 AI 模型。
- **Everything_Instruct_Multilingual 演示翻译**：一条消息展示了 **Everything_Instruct_Multilingual** 数据集的示例，为简单的英文指令提供了包括阿拉伯语、德语、西班牙语和法语在内的 **10 种不同语言** 的翻译。
  - 诸如 'wake me up at nine am on friday' 的翻译在每种语言中都有展示，例如德语：'weck mich am freitag um neun uhr auf'。
- **社区询问数据集质量**：社区成员对新数据集的质量提出了疑问，询问是否进行了 **deduped**（去重）和 **decontaminated**（去污染）。另一位成员对数据集的平衡性表示担忧，指出近 **50%** 的内容与代码相关。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://huggingface.co/datasets/Replete-AI/Everything_Instruct_Multilingual">Replete-AI/Everything_Instruct_Multilingual · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/Replete-AI/Everything_Instruct">Replete-AI/Everything_Instruct · Hugging Face 数据集</a>：未找到描述</li></ul></div>

---

### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1258148010303688834)** (10 messages🔥):

> - `置顶 notebook`
> - `将 notebook 添加到 GitHub 页面`
> - `修正频道中的 notebook 链接`

- ****置顶 notebook 的请求已确认****：一位成员请求置顶某些 notebook，另一位成员确认会执行此操作，并表示需要一点时间。
- ****频道中的 notebook 链接已修正****：对频道中链接的 notebook 进行了修正，澄清了目前有两个 notebook：一个关于使用多个数据集，另一个关于文本分类。
- ****Notebook 将添加到 GitHub 页面****：提到这些 notebook 将被添加到 GitHub 页面，但需要更多时间进行检查和编辑。

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1258138493779972189)** (94 messages🔥🔥):

> - `AI + Blockchain 融资讨论`
> - `Git 合并工具替代方案与冲突解决`
> - `AI 学习课程与建议`
> - `Claude 及其他 AI 编程辅助工具`
> - `对 BM42 等新搜索算法的评估与批评`

- ****AI + Blockchain 斩获 8500 万美元种子轮融资****：“AI + Blockchain = 8500 万美元种子轮 ☠️ VCs 没救了，”一位成员在分享一个免费项目链接 [rakis](https://github.com/hrishioa/rakis?tab=readme-ov-file) 时调侃道，以此吐槽这笔巨额融资。
- ****Git 合并工具大比拼****：成员们讨论了各种解决 Git 合并冲突的工具，包括 [lazygit](https://github.com/jesseduffield/lazygit) 和 [Sublime Merge](https://www.sublimemerge.com/) 等交互式 Rebase 工具，并强调了手动解决冲突的繁琐。
- ****面向初学者的 AI 学习课程****：一位寻找 AI 学习资源的用户收到了诸如 Replit 的 100 Days of Code 和吴恩达（Andrew Ng）的 Deep Learning Specialization 等建议，并且相比于 [Machine Learning Specialization](https://www.deeplearning.ai/courses/machine-learning-specialization/) 等书籍，该用户更倾向于交互式课程。
- ****Claude 3.5 及其他 AI 编程工具****：用户分享了使用 Claude 3.5 和 aider 等编程工具的经验，其中 Cursor 在代码补全和处理复杂多文件重构（multi-file refactors）方面的能力获得了好评。
- ****关于 BM42 搜索算法的争议****：Qdrant 推出的 BM42 遭到了批评，被指其展示的 Benchmarks 可能具有误导性，这促使开发者修改了他们的评估指标和数据集，详见他们的[后续帖子](https://x.com/qdrant_engine/status/1809291686625046816)。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://x.com/qdrant_engine/status/1808498752107241949?s=46">来自 Qdrant (@qdrant_engine) 的推文</a>：40 年来，BM25 一直是搜索引擎的标准。然而，它在现代 RAG 应用中表现不足。向 BM42 问好：语义搜索与关键词搜索的结合</li><li><a href="https://github.com/wesen/glazed/blob/e180e5d59031f20009c461466a2995ff28ee25a7/pkg/doc/topics/13-layers-and-parsed-layers.md">glazed/pkg/doc/topics/13-layers-and-parsed-layers.md at e180e5d59031f20009c461466a2995ff28ee25a7 · wesen/glazed</a>：一个让命令行工具轻松输出结构化数据的库。为你的数据锦上添花 - wesen/glazed</li><li><a href="https://x.com/jobergum/status/1809157587612336402">来自 Jo Kristian Bergum (@jobergum) 的推文</a>：好吧，摊牌了。@qdrant_engine 在 BM42 帖子中的做法是不可接受的。他们在很大程度上误导了 RAG 社区。1) 将 Quora 作为相关的 RAG 问答数据集呈现。我...</li><li><a href="https://x.com/qdrant_engine/status/1809291686625046816">来自 Qdrant (@qdrant_engine) 的推文</a>：大家好！我们确实发现之前的 BM42 Benchmarks 存在偏差。请不要盲目相信我们，务必在您自己的数据上检查性能。我们纠正该问题的最新努力见此：http...</li><li><a href="https://www.manning.com/books/build-a-large-language-model-from-scratch">从零开始构建大语言模型 (Build a Large Language Model (From Scratch))</a>：通过从头开始构建一个大语言模型，学习如何创建、训练和微调 LLMs！&lt;/b&gt;<p>在《从零开始构建大语言模型》中，你将发现 LLMs 是如何工作的...</p></li></ul></div>

---

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1258858993888137227)** (5 条消息):

> - `与 Reka 的 Yi Tay 合作的新播客集`
> - `关于成功 AI 研究员素质的讨论`
> - `OpenAI、Google Gemini 和 Reka 团队的对比`
> - `播客中涵盖的技术主题`

- **Yi Tay 谈 YOLO 研究员元博弈 (Metagame)**：[新播客集](https://latent.space/p/yitay) 中，来自 **Reka** 的 **Yi Tay** 讨论了其团队从零开始构建新训练栈 (training stack) 并纯粹基于直觉训练前沿模型的历程。**Yi Tay** 对比了 OpenAI 和 Google Gemini 的团队规模，并反思了 **Reka** 的研究文化。
  - *"@sama 曾推测过‘10,000x AI 研究员’的素质，最近 @_jasonwei 描述了‘Yolo run’研究员。"* 详细主题包括 LLM 趋势、RAG 以及开源与闭源模型。
- **现已登上 Hacker News**：与 Yi Tay 合作的 [Latent Space Podcast](https://news.ycombinator.com/newest) 剧集现已在 Hacker News 上线。参与 [讨论](https://news.ycombinator.com/item?id=40886218) 并投票以增加曝光度。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://news.ycombinator.com/newest">New Links | Hacker News</a>：未找到描述</li><li><a href="https://x.com/latentspacepod/status/1809300018907828285">来自 Latent Space Podcast (@latentspacepod) 的推文</a>：🆕 播客：与 @YiTayML 探讨 Yolo 研究员元博弈！https://latent.space/p/yitay OpenAI (约 GPT4 时期)：~600 人 Google Gemini：~950 名共同作者 @RekaAILabs：20 人 @sama 曾推测过素质...</li></ul></div>

---

### **Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1258135475609145376)** (34 条消息🔥):

> - `Discord 音视频 (AV) 问题`
> - `迁移到 Zoom 以获得更好的音视频体验`
> - `Discord 与 Linux 之间已知的兼容性问题`

- **AIEWF 演示期间 Discord 音视频出现故障**：**OpenAI AV** 在 AIEWF 演示期间面临**重大问题**，多名用户无法看到屏幕并遇到中断。*Eugene 和其他人建议切换到 Zoom 以获得更稳定的体验。*
  - *swyxio 补充道：*
- **论文俱乐部 (Paper Club) 切换到 Zoom**：由于持续的音视频问题，小组决定从 Discord 切换到 **Zoom**。[分享了](https://us06web.zoom.us/j/8807908941?pwd=eHBBdk9sWWluSzB2TFdLOVdEN3BFdz09) Zoom 链接，成员们开始迁移。
- **讨论 Discord-Linux 兼容性问题**：几位参与者强调了 Discord 与 Linux 之间**已知的兼容性问题**。*Eugene 补充说 Discord 与 Linux 兼容性不佳*，并建议寻找替代方案。

**提到的链接**：[加入我们的 Cloud HD 视频会议](https://us06web.zoom.us/j/8807908941?pwd=eHBBdk9sWWluSzB2TFdLOVdEN3BFdz09)：Zoom 是现代企业视频通信的领导者，拥有简便、可靠的云平台，可跨移动端、桌面端和会议室系统进行视频和音频会议、聊天及网络研讨会。Zoom ...

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1258874839163207785)** (243 条消息🔥🔥):

> - `用户技术困难与技能幽默`
> - `对工作坊主持人的个人赞赏`
> - `关于模型合并策略的讨论`
> - `LlamaFile vs Ollama 对比`
> - `活动规划与反馈`

- ****用户应对技术问题并分享欢笑****：一位用户在通话中难以听清声音，引发了调侃以及现在流行的短语 **'skill issue tbh'**（坦白说是技能问题）。最终，该用户意识到自己不在通话中，并以幽默的方式解决了问题。
- ****LlamaFile vs Ollama：不同的目标****：社区成员对比了 **LlamaFile** 和 **Ollama**，指出 LlamaFile 的优势在于**便携性和优化**，而 Ollama 则在于**与众多模型的广泛兼容性**。
- ****模型合并策略****：讨论强调了 **LlamaFile 和 Ollama** 之间产品目标的差异，同时提出了潜在的模型合并策略以及双方各自需要改进的方向。
- ****AI 生成笔记与可穿戴技术****：关于可穿戴设备使用的讨论强调了潜在的隐私担忧以及录音中征得同意的重要性。参与者提到希望将可穿戴设备与 AI 生成的笔记集成，以便更轻松地浏览活动内容。
- ****即将到来的活动计划与反馈****：参与者为未来的活动构思了改进方案，考虑为工作坊和社区活动增加天数，并指出目前组织和执行高效 AI 会议的方法取得了成功。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://www.ivanleo.com/blog/ai-conf">AI Engineering World Fair</a>：未找到描述</li><li><a href="https://aie.compasswearable.com/events">AI Engineers World Fair Recaps - Powered by Compass</a>：体验规模最大的 AI 技术会议，包含实时转录和 AI 生成的摘要。</li><li><a href="https://codingwithintelligence.com/p/ai-engineer-world-fair-in-sf">AI Engineer World Fair in SF</a>：Coding with Intelligence 第 26 周</li><li><a href="https://x.com/RickLamers/status/1808705188024439187">Rick Lamers (@RickLamers) 的推文</a>：模型合并太疯狂了，看看这个家谱 :0</li><li><a href="https://x.com/philip_kiely/status/1808589566921879702">Philip Kiely (@philip_kiely) 的推文</a>：这是我在 @aiDotEngineer World's Fair 充满活力的三天中捕捉到的 3 个主题：1. 开源正在缩小差距 2. 推理无处不在 3. Evals 就是一切。详情如下：</li><li><a href="https://docs.google.com/document/d/1TLXkcaNX6cvpiqqyo952_K2a7XTF064R44v3WL9CSbE/edit?usp=sharing">AI Engineering Worlds Fair</a>：AI Engineering Worlds Fair Thomas Dohmke 以人为本的方法 - “co-pilot”。Copilot 帮助开发者保持在软件流程中。使信息获取民主化 - 入职培训。Agent - AI 洗碗机（侧边...</li><li><a href="https://docs.google.com/presentation/d/1A_yLcD6Sy1Nr_v2YesOzvtcg5yAmmrfPR2bU4dyxTzw/edit?usp=sharing">AI in action - 2024-07-05</a>：AI in action AI Engineers World Fair 回顾 2024-07-05</li><li><a href="https://x.com/intertwineai/status/1807060271828975632">Bryan Young (@intertwineai) 的推文</a>：@aiDotEngineer 第 3 天回顾与总结！1/12：#AIEWF 2024 第 3 天结束了，很明显我们才刚刚触及 AI 潜力的皮毛，并正在定义什么是 @aiDotEngineer。这里...</li><li><a href="https://x.com/intertwineai/status/1806270266965889289">Bryan Young (@intertwineai) 的推文</a>：@aiDotEngineer 第 2 天回顾！1/14。第二天以 @YoungPhlo_ 关于 AI 生成音乐的及时会议开始。我们一起制作了一些酷炫的节拍。尽管最近针对...的 RIAA 诉讼...</li><li><a href="https://x.com/intertwineai/status/1805867608593645916">Bryan Young (@intertwineai) 的推文</a>：1/5：@aiDotEngineer 的第 1 天正如我想象的那样令人兴奋！#AIEWF 当日快速回顾：</li></ul></div>

---

### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1258136112522854441)** (157 条消息🔥🔥):

> - `等待升级 LM Studio 的硬件`
> - `Llama3 与 Mistral 模型的对比`
> - `在 LM Studio 中使用 OpenAI 或 Anthropic 的 API keys`
> - `LM Studio 中的 Text embeddings 和本地服务器设置`
> - `在有限硬件上运行 Llama3 70b 等大型模型的挑战`

- ****等待升级 LM Studio 的硬件****：一位用户提到计划等待 2 年后再为 **LM Studio** 购买新笔记本电脑，在此期间倾向于使用目前的配置：**64GB DDR4 RAM**、**Ryzen 5900** CPU 和 **3060 6GB GPU**。
- ****Llama3 与 Mistral 模型的对比****：成员们讨论了偏好，一些人更倾向于 **Llama3 8b** 而非 **Mistral 7b Instruct 0.3**，另一些人则强调了基于 **Mistral** 微调的 **OpenHermes 2.5** 的成功体验。
- ****在 LM Studio 中使用 OpenAI 或 Anthropic 的 API keys****：一位用户询问 **LM Studio** 是否允许使用来自 **OpenAI** 或 **Anthropic** 的 API keys 来加载他们的模型。他们被告知 **LM Studio** 仅支持本地文本模型。
- ****在有限硬件上运行 Llama3 70b 等大型模型的挑战****：一位用户报告了由于显存限制在 **RTX 3090 Ti** 上运行 **Llama3 70b** 时出现的问题，并收到了降低 GPU offload 和 context length 或切换到更小模型的建议。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator">LLM Model VRAM Calculator - a Hugging Face Space by NyxKrage</a>：未找到描述</li><li><a href="https://lmstudio.ai/docs/text-embeddings">Text Embeddings | LM Studio</a>：Text embeddings 是一种将文本表示为数字向量的方法。</li><li><a href="https://llama3.dev/">Llama 3 Chat Meta AI - Llama 3 Chat Online 8B and 70B</a>：Llama 3 是来自 Meta 的最新语言模型。Llama 3 有两种尺寸：8B 和 70B。通过此 Llama 聊天机器人快速在线试用 Llama 3。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1806ksz/information_on_vram_usage_of_llm_model/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/discussions/638">30B model now needs only 5.8GB of RAM? How? · ggerganov/llama.cpp · Discussion #638</a>：(编辑：抱歉，我最初应该澄清我是在 Linux 操作系统上运行的。我没意识到对于非 Linux 用户来说，仅凭截图可能并不明显。所有测试都是在 Ubun...</li></ul></div>

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1258136706755067915)** (130 messages🔥🔥):

> - `关于 Gemma-2-27b 不同量化版本之间模型行为不匹配的讨论`
> - `使用 system prompts 改善编程模型的行为`
> - `比较不同的量化技术及其性能`
> - `Qwen2 模型预设和 ChatML 格式讨论`
> - `使用 Gemma、InternLM 和 Dolphin 等不同大语言模型的经验与问题`

- ****Gemma 2 模型在基准测试中表现不佳****：用户报告称 **Gemma-2-27b** 模型在基准测试中表现糟糕且不稳定，不同量化方法（Q5_K_M 或 Q6_K）之间存在显著的不一致性。一项特定测试显示 **27b** 和 **9b** 模型之间存在**巨大的性能差异**。
- ****System prompts 改善编程响应****：为 **Gemma 2 27B** 等模型定制用于编程指导的 system prompts 提高了响应质量。一种专注于 PEP 8 规范和高效算法的特定方法增强了**代码生成的连贯性和完整性**。
- ****了解 Qwen2 的 ChatML 格式****：由于缺乏关于 ChatML 预设的清晰说明，新用户在使用 **Qwen2 模型**时遇到困难。关于 **ChatML 格式重要性**的详细解释有助于澄清预设配置。
- ****不同量化技术的问题****：用户讨论了 **IQ quants** 在非 CUDA 硬件上的不稳定性，报告了更慢的 token 速度以及随机行为（如死循环和响应不一致）。建议在 **Apple 设备**上避免使用 IQ quants，并考虑其他量化方法以获得更好的性能。
- ****在游戏开发和其他任务中使用各种 LLM 的经验****：成员们分享了在游戏开发和 VFX 工作流等任务中使用 **Gemma、InternLM 和 Dolphin** 等不同大模型的参差不齐的结果。模型在保持上下文和遵循指令方面表现不均，引发了对**实际应用和稳定性**的担忧。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/chat-markup-language">如何使用 Chat Markup Language (预览版) - Azure OpenAI</a>：了解如何使用 Chat Markup Language (预览版)</li><li><a href="https://huggingface.co/facebook/multi-token-prediction">facebook/multi-token-prediction · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/mradermacher/koboldai-erebus-extended-32k-7B-GGUF?not-for-all-audiences=true">mradermacher/koboldai-erebus-extended-32k-7B-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/bartowski/Qwen2-7B-Instruct-GGUF">bartowski/Qwen2-7B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/deepseek-ai/ESFT-vanilla-lite">deepseek-ai/ESFT-vanilla-lite · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/Downtown-Case/internlm2_5-7b-chat-1m-llamafied-Q6K-GGUF/tree/main">Downtown-Case/internlm2_5-7b-chat-1m-llamafied-Q6K-GGUF at main</a>：未找到描述</li><li><a href="https://huggingface.co/KoboldAI/Mistral-7B-Erebus-v3?not-for-all-audiences=true).">KoboldAI/Mistral-7B-Erebus-v3 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/mradermacher/Mistral-7B-Erebus-v3-i1-GGUF?not-for-all-audiences=true).">mradermacher/Mistral-7B-Erebus-v3-i1-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1dvcqt5/checked_180_llms_on_writing_quality_code_for_deep/">Reddit - 深入探索一切</a>：未找到描述</li></ul></div>

---

### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1258451443069751357)** (3 messages):

> - `MacBook Pro M2 上 LM 模型下载的问题`
> - `在 LM 中暂停/停止下载的解决方案`

- ****模型在 MacBook Pro M2 上卡在下载状态****：**msouga** 遇到了 LM 中某些模型在搭载 **M2 芯片的 MacBook Pro** 上无限期卡在下载状态的问题，无法停止这些下载或预估完成时间。
- ****如何在 LM 中暂停/停止下载****：**a_dev_called_dj_65326** 建议查看**下载部分**（底栏）以暂停或停止下载。**msouga** 确认该方案完美解决问题。

---

### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1258814454661906525)** (5 条消息):

> - `Nxcode 7B JSON 请求`
> - `CodeQwen 1.5 7B ChatML 兼容性`
> - `RTX 4060 8GB VRAM 和 16 GB DDR5 RAM 性能问题`
> - `针对中端 GPU 配置的建议模型`

- **Nxcode 7B JSON 请求**: @49206c696b652063757465 询问了 **Nxcode 7B** 或 **CodeQwen 1.5 7B** 的 JSON 配置。
- **CodeQwen 1.5 7B ChatML 兼容性**: @heyitsyorkie 提到 **Nxcode 7B** 和 **CodeQwen 1.5 7B** 都使用 **ChatML**，且 **CodeQwen** 需要启用 **flash attention**。
- **RTX 4060 8GB VRAM 在运行 20B 模型时表现挣扎**: @falconandeagle123 分享了他们的配备 **RTX 4060 8GB VRAM** 和 **16 GB DDR5 RAM** 的笔记本电脑在运行 **q4 量化 20B 模型**时非常吃力，导致笔记本电脑冻结。
- **针对中端 GPU 配置的建议模型**: @niga256_512_1024_2048 建议在中端 GPU 配置中使用更简单的模型，如 **Mistral 7B**、**Open Hermes 2.5**、**Wizard code** 和 **Phi 3 mini**。
  - 他们指出这些模型更适合类似于配备 RTX 4060 的笔记本电脑系统。

---

### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1258297612751081535)** (61 条消息🔥🔥):

> - `搭载 Snapdragon Elite 的 Surface Laptop 性能详情`
> - `Snapdragon 设备中的 NPU 和 GPU 利用率`
> - `Snapdragon 和 Intel 设备上的 CPU 性能对比`
> - `Llama.cpp 对 NPU 的未来支持`
> - `关于 LM Studio 所用硬件的一般讨论`

- \****Snapdragon Elite CPU 表现出色**: *成员讨论了搭载 Snapdragon Elite 的 Surface Laptop 的性能细节，包括 LLaMA3 模型的首字速度和每秒 token 数 (t/s)。其他成员将其与自己的 Intel 四核笔记本电脑进行了对比，发现 Snapdragon 的 CPU 性能令人印象深刻.*\*: 一位成员报告称，在搭载 **Snapdragon Elite** 和 **32 GB RAM** 的 **Surface Laptop** 上，使用 **8bit 精度**运行 **LLaMA3 8b** 时，**首字延迟为 1.5 秒**，速度为 **10 t/s**。他们注意到 **GPU 占用率为 10%** 且没有 NPU 活动，这引发了对未来潜在 NPU 利用率的好奇。
  - 对比显示，**Snapdragon Elite CPU** 的性能**明显快于**旧款 Intel 四核笔记本电脑，甚至可以与典型的云端 AI 速度相媲美。成员们推测，未来的 NPU 支持可能会带来进一步的速度提升。
- \***\*Llama.cpp 未来会支持 NPU 吗？\*\*: \*关于 LM Studio 中的 Llama 模型何时可能获得 NPU 支持的讨论。**\*: 成员们讨论到 **Llama.cpp 目前尚不支持 NPU**，导致 LM Studio 中的 **LLaMA 模型仅能依靠 CPU 运行**。关于何时实现支持的推测不断涌现，人们希望是在 2024 年底或 2025 年初。
  - 对话透露 **Qualcomm 有一个 GitHub 仓库**展示了在 NPU 上运行的 LLaMA2，尽管目前还比较粗糙。社区对*未来的增强功能*表现出极大的热情，特别是随着 Qualcomm 和 Microsoft 推动 NPU 的利用。
- ****NPU 实现面临延迟*\*: \*成员们表达了对当前硬件性能的希望和挣扎。***: 在现有系统中实现 **NPU** 的努力进展缓慢，成员们分享了调查该主题的讨论和仓库链接 ([GitHub repo](https://discord.com/channels/1110598183144399058/1153759714082033735/1257360281936597103))。
  - 成员们对最终的改进持乐观态度，甚至分享了一些幽默的建议，比如用“纸板 NPU”作为占位方案。
- ****Surface Laptop 在 Snapdragon Elite 上展现出潜力*\*: \*用户分享了关于新款 Surface Laptop 制造质量和性能的正面体验。***: *一位成员赞扬了他们搭载 Snapdragon Elite 的 Surface Laptop 的制造质量、键盘和触控板*。他们强调**进行视频编辑和玩游戏**的能力是其突出特点。
  - 总体而言，搭载 Snapdragon Elite 的 Surface Laptop 广受好评，尤其是作为**个人使用的日常主力机**，尽管在受 IT 限制的工作场景下仍需要单独的工作笔记本。

<div class="linksMentioned"><p><strong>提到的链接</strong>:</p><ul><li><a href="https://tenor.com/view/office-michael-scott-thank-you-gif-5278681">Office Michael GIF - Office Michael Scott - 浏览并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/not-funny-haha-not-funny-hahaha-evil-laugh-laughing-gif-17347025675359632212">Not Funny Haha Not Funny GIF - Not funny Haha not funny Hahaha - 浏览并分享 GIF</a>: 点击查看 GIF</li></ul></div>

---

### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1258422461549121566)** (2 条消息):

> - `AppImage not compatible with aarch64 CPUs`
> - `No ARM CPU support on Linux for LM Studio`

- ****AppImage 与 aarch64 CPU 不兼容****：一位用户在尝试于 **aarch64** 系统上执行 [LM_Studio-0.2.27.AppImage](https://link.to/LM_Studio-0.2.27.AppImage) 时遇到了 **Exec format error**，这表明架构不兼容。`lscpu` 命令输出确认了该 CPU 架构为 **aarch64**。
- ****Linux 上不支持 ARM CPU****：讨论强调了 **LM Studio** 在 Linux 上缺乏 **ARM CPU 支持**。一名成员确认道：*"No arm cpu support on linux"*。

---

### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1258509419747086456)** (2 条消息):

> - `7800XT user confirms GPU works`
> - `Problems loading models with GPU offload`
> - `Successful ROCm installation script`

- ****7800XT 用户确认 GPU 可正常工作****：*用户报告其 **7800XT** 运行成功，且不确定是否需要反馈。*
- ****GPU offload 加载模型时出现问题****：除非禁用 GPU offload，否则**加载模型会失败**。用户讨论了解决此问题的安装脚本。
- ****成功的 ROCm 安装脚本****：一位**用户建议了一个安装 ROCm 的脚本**，该脚本帮助解决了 GPU offload 的加载问题。另一位用户确认该脚本运行良好。

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1258257005542375555)** (10 条消息🔥):

> - `Matrix multiplication in CUDA`
> - `Efficient remote development with pay-for-use compute`
> - `New blog post on executive summary process using GPTs`
> - `Paid CUDA/ML system certifications`
> - `Upcoming in-person CUDA mode event in October`

- **CUDA 中的矩阵乘法：为什么是列而不是行？**：一位用户询问为什么在 GPU 矩阵乘法期间，紫色图块（tile）上加载的是 64 元素的列而不是行，另一位用户分享了一篇详细的[博文](https://siboehm.com/articles/22/CUDA-MMM)，介绍如何使用 CUDA 优化此过程。
- **使用 GPTs 简化执行摘要**：一篇新的[博文](https://www.dylandavis.net/2024/07/three-gpts-walk-into-a-bar-and-write-an-exec-summary/)详细介绍了一个涉及三个自定义 GPTs 的流程，以加快执行摘要的撰写，展示了它们如何快速提取见解、起草和修改摘要。
- **高效远程开发的技巧**：成员们讨论了远程开发的解决方案，允许使用按需付费的算力同时保留文件，并提到 **Lightning AI** 和 **AWS S3** 是潜在的选择。
- **CUDA/ML 认证推荐**：一位用户寻求 500 美元以下的付费 CUDA/ML 认证推荐，随后有人建议了 NVIDIA 课程以及可能由社区组织的研讨会。
- **宣布 CUDA Mode 线下活动**：据社区成员透露，**CUDA Mode** 正计划在 10 月举办一场线下活动，并承诺很快会公布更多细节。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://siboehm.com/articles/22/CUDA-MMM">How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog</a>：在这篇文章中，我将迭代优化一个用 CUDA 编写的矩阵乘法实现。我的目标不是构建 cuBLAS 的替代品，而是深入...</li><li><a href="https://www.dylandavis.net/2024/07/three-gpts-walk-into-a-bar-and-write-an-exec-summary/">Three GPTs Walk into a Bar and Write an Exec Summary – D-Squared</a>：未找到描述</li></ul></div>

---

### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1258221254331203666)** (2 条消息):

> - `Triton kernels with multiple CUDA graphs create latency issues`
> - `SRAM contention affecting performance`

- ****并行 CUDA 执行下的 Triton Kernel****：与本地基准测试相比，并行运行带有 Triton kernel 的**多个 CUDA graph 实例**显示出**更差的延迟**。
  - 有人建议，如果多个实例都在执行 `tl.load`，**SRAM 争用**可能是原因。
- ****与 Torch 性能的比较****：尽管存在潜在的 **SRAM 争用**，但在类似条件下，**Torch 似乎不存在**此问题。
  - 这种差异引发了关于 **Triton 和 Torch 在处理 SRAM 逐出（eviction）方式上有何不同**的问题。

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1258584845966249985)** (7 条消息):

> - `torch.compile` 在 Python 3.12 上不受支持
> - Python `bytecode` 兼容性问题
> - `TorchDynamo` 与 Python `frame evaluation API`
> - `TorchDynamo` 在 PyTorch 性能中的作用

- ****Torch 2.3 `.compile` 在 Python 3.12 上不受支持****：对于 **torch 2.3**，由于 Python 内部机制的变化（特别是处理 `bytecode` 的方式），`.compile` 函数在 **Python 3.12 上不受支持**。详细解释可以在[这里](https://dev-discuss.pytorch.org/t/torch-compile-support-for-python-3-12-completed/2054)找到。
- ****Python Bytecode 变化导致支持滞后****：**Python bytecode** 在每个 Python 版本中都会发生变化，这需要 **torch.compile** 等框架花费时间来调整并支持这些新变化。更多关于 `bytecode` 调整的信息可以阅读[此文档](https://pytorch.org/docs/stable/torch.compiler_deepdive.html)。
- ****TorchDynamo 提升 PyTorch 性能****：[TorchDynamo](https://pytorch.org/docs/stable/torch.compiler_deepdive.html) 是一个 **Python 级 JIT compiler**，它挂钩到 CPython 的 `frame evaluation` 中以修改 Python `bytecode`，并将 PyTorch 操作编译为 **FX Graph**。通过使用由 `torch.compile()` 包装的 `torch._dynamo.optimize()`，它可以无缝提升 PyTorch 代码性能。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://pytorch.org/docs/stable/torch.compiler_deepdive.html">TorchDynamo 深度解析 — PyTorch 2.3 文档</a>：暂无描述</li><li><a href="https://dev-discuss.pytorch.org/t/torch-compile-support-for-python-3-12-completed/2054">Torch.compile 对 Python 3.12 的支持已完成</a>：宣布 Python 3.12 的支持已添加到 torch.compile 中，并且已经在 nightly 版本中存在了一段时间。我们预计此功能将包含在 PyTorch 2.4 版本中...</li></ul></div>

---

### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1258827477841743982)** (5 条消息):

> - 训练语言模型预测未来多个 token 的新方法
> - 语言模型中的 `self speculative decoding`
> - `multi-token prediction` 与 `lookahead decoding` 基准的比较
> - `multi-token prediction` 模型中 `n-gram` 生成的有效性

- **新方法提升语言模型效率**：[最新研究论文](https://arxiv.org/abs/2404.19737)建议训练语言模型一次预测多个未来 token，从而在不增加训练时间的情况下，实现更高的样本效率并提升下游能力。**13B 参数模型**显示出显著收益，在 **HumanEval 上多解决了 12% 的问题**，在 **MBPP 上多解决了 17%**。
- **Self Speculative Decoding 获得认可**：一位成员提到该模型执行 **self speculative decoding** 的能力非常出色。
- **质疑 lookahead decoding 基准**：成员们想知道这种新的 `multi-token prediction` 与 **lookahead decoding 基准**相比如何。
- **剖析 n-gram 有效性**：讨论了在 `multi-token prediction` 模型中生成 **n-grams** 的有效性，以及它们与传统 `next-token prediction` 输出的一致性。

**提到的链接**：[Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)：诸如 GPT 和 Llama 等大型语言模型是使用 `next-token prediction` 损失进行训练的。在这项工作中，我们建议训练语言模型一次预测多个未来 token 会导致...

---

### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 条消息):

iron_bound: [https://oimo.io/works/life/](https://oimo.io/works/life/)

---

### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1258442106981187747)** (17 条消息🔥):

> - `Learning path for backend SDEs interested in CUDA and inference optimization`（对 CUDA 和推理优化感兴趣的后端 SDE 的学习路径）
> - `Challenges of finding jobs with open source contributions`（凭借开源贡献寻找工作的挑战）
> - `Recommendation of CUDA Mode GitHub for beginners`（为初学者推荐 CUDA Mode GitHub）
> - `Building a deep learning framework from scratch in C++ using CUDA`（使用 C++ 和 CUDA 从零开始构建深度学习框架）
> - `Using Python for CUDA kernel development vs C++`（使用 Python 还是 C++ 进行 CUDA kernel 开发）

- ****寻找 CUDA 精通之路****：一位后端 SDE 寻求关于转向 **CUDA** 和**推理优化**相关工作的建议。建议包括**观看特定频道**、**阅读相关资源**、向 **GitHub** 贡献代码以及加入**工作组**。
- ****开源贡献并不总是工作的敲门砖****：有人担心尽管做出了显著的**开源贡献**，但仍未能获得工作。社区承认了这一挑战，并讨论了极高的准入门槛。
- ****CUDA Mode GitHub：初学者的宝库****：对于想要深入学习 **CUDA** 的初学者，**CUDA Mode GitHub** 被推荐为一个卓有成效的起点。它被建议作为一个构建有趣项目和高效学习的平台。
- ****使用 C++ 和 CUDA 构建深度学习框架****：一位成员表达了使用 **CUDA** 和 **C++** 实现并行化来构建类似 **tinygrad** 的深度学习框架的兴趣，但在 **C++** 的复杂性上遇到了困难。他们考虑改用 **Python** 以获得更好的可管理性并可能更快完成。
- ****Python vs C++ 进行 CUDA Kernel 开发****：关于是使用 **Python** 还是 **C++** 进行 **CUDA** kernel 开发引发了辩论。共识倾向于在初始尝试中使用 **Python**，而在进行深层系统级工作时转向 **C++**，并引用了 **llama.c** 等仓库进行学习。

---

### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1258429258984849580)** (4 条消息):

> - `Fourth edition released in 2022`（第四版于 2022 年发布）
> - `Differences between third and fourth editions`（第三版和第四版之间的区别）

- ****第四版于 2022 年发布****：第四版于 2022 年发布，而上一版是在 2012 年发布的。
- ****第三版和第四版之间的区别****：一位成员提到没有读过第三版，对两者之间的差异表示好奇。另一位成员查阅了他们手中书籍的封底以获取详细信息。

---

### **CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1258851566044184587)** (4 条消息):

> - `casual conversation`（闲聊）
> - `channel engagement`（频道互动）

- **频道内的随性互动**：一位成员用简单的 *"that's so cool!"* 表达了他们的兴奋，展示了随性的互动和赞赏。
  - 另一位成员回复了 *"thanks"*，展示了频道内友好且相互感激的互动。
- **友好互动**：成员们通过 *"yo"* 和 *"you"* 等简短消息进行随性且友好的交流。
  - 这些互动反映了积极且受欢迎的社区环境。

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1258171846227136622)** (11 条消息🔥):

> - `a.to` 方法识别与功能的处理
> - 移除 PyTorch/ao 中不必要的参数
> - `a.to` 方法当前的限制与权宜之计
> - 在子类中添加对 `device` 和 `dtype` 处理的支持
> - 未来功能及在 Torchbench 模型中的测试

- ****修复 PyTorch 中的 `a.to` 问题****：`a.to(torch.int32)` 方法被识别为 `a.to(device=torch.int32)`，导致非预期行为。为了修复此问题，需要移除 [affine_quantized_tensor.py](https://github.com/pytorch/ao/blob/a8956992191853b13f82ceb3e6929bed7691a3fa/torchao/dtypes/affine_quantized_tensor.py#L262) 中不必要的 `device` 和 `memory_format` 参数。
- ****`a.to(dtype=torch.int32)` 面临的挑战****：讨论强调 `a.to(dtype=torch.int32)` 目前仅更改设备，而不会更改 dtype 或 layout 等其他关键字，这表明目前尚不支持 **dtype 和 memory format 的更改**。
- ****AQT 中的临时函数调整****：建议修改 `affine_quantized_tensor.py` 文件，暂时丢弃 `device`、`dtype` 和 `memory_format` 参数，以应对当前实现中的局限性。
- ****Subclass `a.to` 方法的局限性****：关于 `torchbench` 中子类功能的讨论显示，处理不同 dtype 的 `a.to` 方法并非初衷，因为更改外部表示的 dtype 会带来复杂的挑战。
- ****在 Torchbench 中测试功能****：人们担心当前的设置是否支持 `torchbench` 中各种模型的 `.to` 方法，特别是关于 **subclass 处理** 以及 AQT 实现中所需的功能测试。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://github.com/pytorch/ao/blob/a8956992191853b13f82ceb3e6929bed7691a3fa/torchao/dtypes/affine_quantized_tensor.py#L262">pytorch/ao 中的 ao/torchao/dtypes/affine_quantized_tensor.py</a>：创建并集成自定义数据类型、布局和内核，在推理和训练中可实现高达 2 倍的加速并减少 65% 的 VRAM。</li><li><a href="https://github.com/pytorch/ao/blob/a8956992191853b13f82ceb3e6929bed7691a3fa/torchao/dtypes/affine_quantized_tensor.py#L261:">pytorch/ao 中的 ao/torchao/dtypes/affine_quantized_tensor.py</a>：创建并集成自定义数据类型、布局和内核，在推理和训练中可实现高达 2 倍的加速并减少 65% 的 VRAM。</li></ul></div>

---

### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1258329101425709126)** (3 条消息):

> - `Lightning AI 制作的 Thunder Sessions 播客`
> - `Andrej Karpathy 在 2024 年加州大学伯克利分校 AI 黑客松上的主题演讲`

- ****Thunder Sessions 播客引发关注****：Lightning AI 宣布了由 **Luca Antiga** 和 **Thomas Viehmann** 主持的 [Thunder Sessions 播客](https://x.com/LightningAI/status/1808610408481370205)，内容涵盖编译器和性能优化，将于 **美东时间 7 月 5 日星期五上午 11 点** 播出。
- ****Andrej Karpathy 在伯克利黑客松大放异彩****：2024 年加州大学伯克利分校 AI 黑客松颁奖典礼的 [YouTube 视频](https://www.youtube.com/watch?v=tsTeEkzO9xc) 展示了 **Andrej Karpathy** 发表的鼓舞人心的主题演讲，并重点介绍了参赛者们极具开创性的项目演示。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://x.com/LightningAI/status/1808610408481370205">来自 Lightning AI ⚡️ (@LightningAI) 的推文</a>：我们很高兴向大家介绍 🌩️ Thunder Sessions 🌩️，这是来自 Lightning AI 团队的一个新播客，涵盖编译器和性能优化领域。请在美东时间 7 月 5 日星期五上午 11 点加入我们...</li><li><a href="https://www.youtube.com/watch?v=tsTeEkzO9xc">Andrej Karpathy 的主题演讲及 2024 年加州大学伯克利分校 AI 黑客松颁奖典礼获奖者演示</a>：在 2024 年加州大学伯克利分校 AI 黑客松的颁奖典礼上，气氛非常热烈，OpenAI 创始成员 Andrej Karpathy 发表了鼓舞人心的演讲...</li></ul></div>

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1258149753305301012)** (134 条消息🔥🔥):

> - `CUDA MODE Discord 聊天机器人消息`
> - `GPT-2 训练中的 FP8 梯度问题`
> - `Schedule-Free Optimizer 论文`
> - `GPT-2 训练性能`
> - `GPT-2 训练时长估算`

- ****Schedule-Free Optimizers 的问题****：一位成员注意到使用 [Schedule-Free Optimizers](https://x.com/_clashluke/status/1808590060654108910) 产生了令人惊讶的平滑损失曲线，这在像 ImageNet 这样充满噪声的数据集上似乎不太可能。尽管最初持怀疑态度，但该优化器即使在没有自定义优化的情况下也显示出显著的收敛优势。
- ****FP8 梯度激活影响 GPT-2 训练****：一位成员发现将梯度激活转换为 FP8 会显著增加 GPT-2 测试运行期间的损失。他们指出这种误差会在模型中传播，尝试使用随机舍入（stochastic rounding）来缓解的效果有限，建议将某些操作保留在 BF16 以维持稳定性。
- ****Lambda 服务器编译时间的性能困扰****：一位用户报告 Lambda 服务器上的编译时间比本地机器长得多，可能是由于虚拟化实例上禁用了 CPU Turbo。调查显示 CPU 维持在 2GHz 的基准频率，无法发挥其 3.8GHz Turbo 频率的全部潜力。
- ****超参数扫描与模型缩放****：几项讨论集中在不同模型宽度和深度下扫描不同的超参数，如 LR、`attn_mult` 和 `out_mult`。初步结果表明余弦调度器（cosine schedulers）和 `attn_mult` 为 1 是最优的，但进一步的测试仍在进行中。
- ****奥斯汀科技圈趣闻****：闲聊中透露成员们参加了有 Lex Fridman 等科技界知名人士出席的 7 月 4 日派对。他们还提到了奥斯汀在半导体工程方面的重要性，但强调了其与更广泛科技圈缺乏交集。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://x.com/_clashluke/status/1808590060654108910?s=46&amp;t=Qzf619GMalbD77YmVui2Jw">Lucas Nestler (@_clashluke) 的推文</a>：Schedule-free optimizers (https://x.com/aaron_defazio/status/1776320004465582331) 非常不可思议。我读了论文，研究了数学原理，并试图理解发生了什么。这一切看起来……</li><li><a href="https://github.com/karpathy/llm.c/blob/master/scripts/run_gpt2_1558M.sh">llm.c/scripts/run_gpt2_1558M.sh at master · karpathy/llm.c</a>：使用简单、原生的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号为 karpathy/llm.c 的开发做出贡献。</li></ul></div>

---

### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1258735996489433140)** (3 条消息):

> - `CUDA 中针对 int2*int8 gemm 的优化算子`
> - `发布针对 int2*int8 的自定义 gemv`
> - `用于混合精度矩阵乘法的 BitBLAS 库`

- ****新人询问关于 int2*int8 gemm 的优化算子****：一位新成员询问 **CUDA** 中是否有针对 \**int2*int8 gemm\** 操作的优化算子。
- ****宣布发布自定义 gemv 算子****：一位成员宣布他们制作了一个针对 int2\*int8 的自定义 **gemv kernel**，将在几天内发布。
  - 他们还建议查看 [BitBLAS](https://github.com/microsoft/BitBLAS) 作为另一个选择。

**提到的链接**：[GitHub - microsoft/BitBLAS: BitBLAS 是一个支持混合精度矩阵乘法的库，特别适用于量化 LLM 部署。](https://github.com/microsoft/BitBLAS)：BitBLAS 是一个支持混合精度矩阵乘法的库，特别适用于量化 LLM 部署。 - microsoft/BitBLAS

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1258136049272487966)** (165 条消息🔥🔥):

> - `Perplexity AI 重复回答问题`
> - `实时联网访问问题`
> - `Perplexity Pro 中的数学准确性`
> - `在股市中使用 Perplexity 的经验`
> - `订阅计划与模型使用`

- ****Perplexity AI 重复回答问题****：用户反馈 Perplexity AI 在使用相同提示词时会给出重复的回答，特别是在使用 Llama 3 和 Claude 等模型时。一位用户提到 Alex 回复称他们已意识到该问题并正在修复。
- ****实时联网访问问题****：一位用户描述了 Perplexity AI 在访问实时互联网获取实时数据时出现的问题，导致提供了不准确且过时的信息。尽管关闭并重新打开了 App，问题依然存在，用户已在反馈频道中记录了此事。
- ****Perplexity Pro 中的数学准确性****：用户对 Perplexity Pro 在处理 CAPM beta 计算等数学问题时的不准确性表示沮丧。尽管使用的是 GPT-4o 模型，结果仍有显著偏差，这引发了对该模型在学术计算中效能的质疑。
- ****在股市中使用 Perplexity 的经验****：一位用户分享了他们利用 Perplexity 在股市中赚取了 8000 美元的经历，并对其能力表示赞赏。这引发了关于用户在 Pro 版本中体验到的各种益处的简短讨论。
- ****订阅计划与模型使用****：用户讨论了 Pro 和 Enterprise Pro 计划之间的区别，特别是关于 Sonnet 和 Opus 等模型的使用细节。关于不同订阅计划中模型的可用性和具体细节也出现了一些疑问。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://docs.perplexity.ai/reference/post_chat_completions">Chat Completions</a>：未找到描述</li><li><a href="https://www.mayoclinic.org/diseases-conditions/hyperthyroidism/symptoms-causes/syc-20373659">甲状腺功能亢进症 - 症状与原因</a>：未找到描述</li><li><a href="https://docs.perplexity.ai/docs/model-cards">支持的模型</a>：未找到描述</li><li><a href="https://www.dylandavis.net/2024/07/three-gpts-walk-into-a-bar-and-write-an-exec-summary/">三个 GPT 走进酒吧并撰写执行摘要 – D-Squared</a>：未找到描述</li></ul></div>

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1258270754781528106)** (13 条消息🔥):

> - `Threads 的里程碑`
> - `古代原住民仪式`
> - `核动力数据中心`
> - `火星苔藓`
> - `大胃王比赛`

- ****Threads 达成里程碑****：一段名为 **Discover today: Threads' Milestone, Ancient Aboriginal Rituals, and Nuclear-Powered Data Centers** 的 [YouTube 视频](https://www.youtube.com/embed/Q-jy32fjcSs) 讨论了 Threads 最近取得的成就。
- ****火星苔藓与其他奇观****：另一段名为 **Discover today: Mars Moss, Eating Contests, Tech Titans, and Toxic Green** 的 [YouTube 视频](https://www.youtube.com/embed/PfWwPIB62d8) 探讨了火星上苔藓的存在以及各种不寻常的话题。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://www.youtube.com/embed/Q-jy32fjcSs">YouTube</a>：未找到描述</li><li><a href="https://www.youtube.com/embed/PfWwPIB62d8">YouTube</a>：未找到描述</li></ul></div>

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1258281585703256165)** (15 条消息🔥):

> - `pplx-70b-online 与 llama-3-sonar-large-32k-online 的区别`
> - `在 API 中使用 Google Dorks`
> - `LLM 中的时间感知能力`
> - `LLM 中查询命令的有效性`
> - `Perplexity AI 模型与模型卡片 (model cards)`

- ****Google Dorks 与 API 掌握****：一位用户建议利用 Google Dorks 可以增强 API 的效用，因为它简化了在 Web 产品上有效过滤源域的过程。
- ****LLM 缺乏时间感知****：用户讨论了像 **llama3** 和 **haiku** 这样的模型在没有明确提示的情况下，无法直观理解“最新”或“最近”的概念，从而影响了它们的回答。
- ****LLM 中的查询命令：非官方支持****：会议强调，虽然经常建议使用 **Google Dork 运算符**来限制结果，但它们并未正式集成到 Perplexity 的 LLM 后端。
- ****Perplexity 模型说明****：一位用户参考 Perplexity 的博客和 API 文档，寻求关于 **pplx-70b-online** 和 **llama-3-sonar-large-32k-online** 模型之间区别的澄清。
- ****模型别名与过时问题****：关于模型别名和潜在的过时问题存在困惑；一位用户认为某些模型可能是别名，而另一位用户指出某些模型现在可能会抛出错误。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://docs.perplexity.ai/docs/model-cards">支持的模型</a>：未找到描述</li><li><a href="https://docs.perplexity.ai/reference/post_chat_completions">Chat Completions</a>：未找到描述</li></ul></div>

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1258332502016266342)** (185 条消息🔥🔥):

> - `BUD-E 新功能更新`
> - `Clipdrop NSFW 检测问题`
> - `关于数据集可用性与使用的讨论`
> - `各种 AI 模型性能与训练技术`
> - `Stability AI 模型的商业许可`

- ****BUD-E 更新：支持剪贴板访问****：[最近的一段 YouTube 视频](https://youtu.be/WMcEzVfEXpM)展示了 **BUD-E 的新功能**：从屏幕和剪贴板读取文本，详情见 [GitHub](https://github.com/christophschuhmann/Desktop_BUD-E/tree/main) 上的项目描述。演示视频的分辨率为 240p，引发了一些幽默的批评。
- ****Clipdrop 的 NSFW 检测失败****：一位成员分享了一个幽默事件，Clipdrop 错误地将一张图片标记为 **NSFW 内容**。
- ****数据集获取的困境****：成员们讨论了 **FAL.AI** 在获取新数据集方面面临的困难，评论强调了多个模型对相同数据集的过度依赖。一位用户强调，像 **Chameleon** 这样有趣的突破来自于多样化且集成的模态。
- ****Stability AI 的许可证修正****：[Stability AI](https://stability.ai/news/license-update) 将 **SD3 Medium** 的商业许可修改为 Stability AI Community License，允许个人创作者和小型企业更广泛地免费使用。这一变化是针对社区对原始商业许可的反馈而做出的。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://youtu.be/WMcEzVfEXpM">BUD-E Update: Seeing images &amp; reading text from screen &amp; clipboard</a>：https://github.com/christophschuhmann/Desktop_BUD-E/tree/main</li><li><a href="https://huggingface.co/spaces/LPDoctor/Glyph-SDXL-v2/tree/main">LPDoctor/Glyph-SDXL-v2 at main</a>：未找到描述</li><li><a href="https://tenor.com/view/dog-in-space-dog-i-have-no-idea-i-have-no-idea-what-im-doing-gif-25502378">Dog In Space Dog GIF - Dog In Space Dog I Have No Idea - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://gist.github.com/Nodja/2a97c530b8898affd8fd897a95595ee0">字符级分词 (Letter level tokenization)</a>：字符级分词。GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://stability.ai/news/license-update">Community License — Stability AI</a>：我们新的社区许可证现在对研究、非商业和商业用途免费。只有当您的年收入超过 100 万美元且使用 Stability AI 模型时，才需要付费的企业许可证...</li></ul></div>

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1258681015606575116)** (2 条消息):

> - `scammer alert`（诈骗者警报）
> - `new tokenizer proposal for LLMs`（针对 LLMs 的新 Tokenizer 提案）
> - `T-FREE tokenizer paper`（T-FREE Tokenizer 论文）

- ****用户标记诈骗者****：一名用户提醒社区聊天中存在**诈骗者**。
- ****T-FREE Tokenizer 提案引发 LLMs 变革****：一篇新论文提出了 **T-FREE**，这是一种通过字符三元组（character triplets）上的稀疏激活模式对单词进行嵌入的 Tokenizer，消除了对参考语料库的需求，并在 Embedding 层实现了超过 **85%** 的参数缩减。你可以在[这里查看论文](https://arxiv.org/abs/2406.19223)。
  - 该论文概述了 **T-FREE** 的优势，包括提高低资源语言（underrepresented languages）的性能以及 Embedding 层的显著压缩。

**提到的链接**：[T-FREE: Tokenizer-Free Generative LLMs via Sparse Representations for Memory-Efficient Embeddings](https://arxiv.org/abs/2406.19223)：Tokenizer 对于 Large Language Models 中的信息编码至关重要，但其发展近期停滞不前，且存在固有弱点。主要局限性包括计算开销...

---

### **LAION ▷ #[resources](https://discord.com/channels/823813159592001537/991938328763056168/)** (1 条消息):

khazn: $50 礼品卡 [steamcommunity.com/gift/sd271azjxn2h](https://exi.link/EvuqQq)

---

### **LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/)** (1 条消息):

khazn: $50 礼品卡 [steamcommunity.com/gift/sd271azjxn2h](https://exi.link/EvuqQq)

---

### **LAION ▷ #[paper-discussion](https://discord.com/channels/823813159592001537/1172520224797511700/)** (1 条消息):

khazn: $50 礼品卡 [steamcommunity.com/gift/sd271azjxn2h](https://exi.link/EvuqQq)

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1258135295153537054)** (116 条消息🔥🔥):

> - `Moshi AI 演示`
> - `GPT-2 的问题`
> - `OpenAI 模型中的语音模态`
> - `chatGPT 对孟加拉语 (Bangla) 的支持`
> - `用于 AI 集成的 API 使用`

- \****Moshi AI 演示令人兴奋但也令人沮丧**: 发布了一个新的 [Moshi AI 演示](https://moshi.chat/?queue_id=talktomoshi)，具有实时语音交互功能并承诺开源灵活性。然而，用户遇到了对话中断和循环响应等问题，突显了当前模型的局限性。*\*: 发布了一个新的 [Moshi AI 演示](https://moshi.chat/?queue_id=talktomoshi)，具有实时语音交互功能并承诺开源灵活性。然而，用户遇到了对话中断和循环响应等问题，突显了当前模型的局限性。
- \****AI 缺乏长期记忆**: Hume AI 的 [playground](https://demo.hume.ai/) 提供了可中断的语音 AI，但缺乏长期记忆功能，每次会话后都会重置。这种限制让希望从 AI 交互中获得持续学习的用户感到沮丧。*\*: Hume AI 的 [playground](https://demo.hume.ai/) 提供了可中断的语音 AI，但缺乏长期记忆功能，每次会话后都会重置。这种限制让希望从 AI 交互中获得持续学习的用户感到沮丧。
- \****呼吁增强孟加拉语支持**: 一位用户强调了 chatGPT 在处理孟加拉语时持续存在的问题，敦促改进以提高可访问性。该请求附带了线程 ID 以供特定参考，并强调了更广泛语言支持的必要性。*\*: 一位用户强调了 chatGPT 在处理孟加拉语时持续存在的问题，敦促改进以提高可访问性。该请求附带了线程 ID 以供特定参考，并强调了更广泛语言支持的必要性。
- \****GPT-2 与现代模型的辩论**: 关于是使用较旧的 GPT-2 模型进行文本生成，还是升级到更现代的选择（如 GPT-3.5 Turbo），展开了讨论。虽然一些人主张 GPT-2 的成本效益，但其他人指出新模型的性能要好得多。*\*: 关于是使用较旧的 GPT-2 模型进行文本生成，还是升级到更现代的选择（如 GPT-3.5 Turbo），展开了讨论。虽然一些人主张 GPT-2 的成本效益，但其他人指出新模型的性能要好得多。
- \****通过 API 进行 AI 集成**: 用户讨论了使用 API 集成 AI 模型的各种方法，特别关注通过 Assistant API 端点实现 RAG。对话强调了编程知识对于最大化 AI 效用和定制化是多么至关重要。*\*: 用户讨论了使用 API 集成 AI 模型的各种方法，特别关注通过 Assistant API 端点实现 RAG。对话强调了编程知识对于最大化 AI 效用和定制化是多么至关重要。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://demo.hume.ai/">Voice-to-Voice Demo •&nbsp;Hume AI</a>：与第一个共情 AI 语音对话。</li><li><a href="https://moshi.chat/?queue_id=talktomoshi">moshi.chat</a>：未找到描述</li></ul></div>

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1258211085408796753)** (26 条消息🔥):

> - `ChatGPT 免费版与付费版的区别`
> - `在 GPT 知识库中处理图像和 PDF`
> - `GPT 记忆功能的有效性`
> - `在 GPT 内部访问其他 GPT 模型`
> - `GPT 知识库的外部文件链接与向量数据库`

- **ChatGPT 付费计划权益详解**：一位成员询问了 ChatGPT 付费计划的好处，得到的解释是 **Plus** 提供更高的使用上限、访问 **DALL·E** 的权限以及更大的上下文窗口。更多详情可以在[这里](https://openai.com/chatgpt/pricing)找到。
- **GPT 知识库中的图像与 PDF**：成员们讨论了 GPT 是否使用视觉能力来读取上传到知识库部分的图像和 PDF。结论是 **GPT** 并不使用视觉能力，而是依赖 **OCR** 从图像和 PDF 中提取文本。
- **GPT 记忆功能的有效性受质疑**：一位成员批评了 GPT 的记忆功能，指出它虽然能保存偏好但仍会胡编乱造。另一位成员澄清说，这些记忆的作用是建议而非硬性规则，并建议使用自定义选项来改善行为。
- **链接 GPT 与文档服务**：围绕将 GPT 知识库链接到 Google Drive 和其他类似服务展开了复杂的讨论。**有人指出，如果没有自定义后端，外部文件无法达到向量数据库的优化水平**，部分服务为类似功能提供了实时链接支持。
- **确认 GPT-4 使用冷却时间**：针对 GPT-4 可用性的担忧得到了回应，解释称用户在达到限制后，需要经过一段冷却时间才能再次使用 GPT-4。**Plus 用户在 GPT-4 上每 3 小时最多可发送 40 条消息**，在 GPT-4o 上为 80 条，高峰时段可能会有所减少。

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1258154368767627295)** (16 条消息🔥):

> - `员工认可计划`
> - `培训课程的内容生成脚本`
> - `测试多个 AI 回复的工具`
> - `桌面 RPG 提示词`
> - `交通罚单申诉指南`

- **员工认可计划提升士气**：用户讨论了开发一个**员工认可计划**以提升士气和动力。该计划包括目标、认可方式、标准、实施方案和反馈机制。
- **高效的内容生成脚本**：一位用户正在寻求关于**开发内容生成脚本**的建议，以便根据地点、时长、主题和受众等输入来创建培训课程结构。他们正在考虑将 **prompt engineering**、**RAG** 和网页搜索集成作为潜在技术。
- **测试多个 AI 回复的工具**：一位用户询问了用于**测试和可视化来自相同提示词的多个 AI 回复**的工具，寻求支持文件上传和显示回复差异等功能。建议包括**自定义工具**或 Autogen 等现有选项。
- **桌面 RPG 战斗地图提示词**：一位用户征求**生成桌面 RPG 战斗地图的提示词创意**。未讨论具体的工具或技术。
- **挑战交通罚单的指南**：频道讨论了在法庭上**挑战交通罚单**的结构化方法。指南包括有效抗辩罚单的步骤和陈述案件的策略。

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 条消息):

teknium: [https://x.com/kerstingaiml/status/1809152764649574541?s=46](https://x.com/kerstingaiml/status/1809152764649574541?s=46)

---

### **Nous Research AI ▷ #[datasets](https://discord.com/channels/1053877538025386074/1105324249721356298/1258180446613668012)** (1 条消息):

> - `Replete-AI 发布了两个海量数据集`
> - `Everything_Instruct 和 Everything_Instruct_Multilingual 数据集`
> - `新数据集的大小和特性`
> - `bagel 数据集和 EveryoneLLM AI 模型的影响`

- **Replete-AI 揭晓海量数据集**：<@716121022025302076> 发布了两个新数据集，**Everything_Instruct** 和 **Everything_Instruct_Multilingual**，每个大小为 11-12GB，包含超过 600 万行数据。这些数据集采用 **Alpaca Instruct** 风格格式化，重点在于创建一个全面的指令数据集以训练 AI 模型。
- **用于终极 AI 模型训练的双重数据集**：**Everything_Instruct** 专为纯英文数据设计，而 **Everything_Instruct_Multilingual** 包含多语言翻译，以增强模型的语言能力。这两个数据集都受到了 **bagel 数据集** 和之前的 **EveryoneLLM AI 模型** 的启发。
  - 其目标是将所有可以想象到的指令数据类型整合到一个海量数据集中，以训练顶尖的 AI 模型。欢迎在 [Hugging Face 上体验这些数据集](https://huggingface.co/datasets/Replete-AI/Everything_Instruct)。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://huggingface.co/datasets/Replete-AI/Everything_Instruct">Replete-AI/Everything_Instruct · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/Replete-AI/Everything_Instruct_Multilingual">Replete-AI/Everything_Instruct_Multilingual · Hugging Face 数据集</a>：未找到描述</li></ul></div>

---

### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1258278785409876009)** (4 条消息):

> - `即将出版的 Nous 实体杂志征稿`
> - `StudioMilitary 杂志中的开源/去中心化技术`

- **Nous 实体杂志征稿**：**John0galt** 邀请大家为即将出版的 **Nous 实体杂志** 投稿，提供优秀的文案、有趣的内容或创意。如有兴趣请**联系 John0galt**。
- **StudioMilitary 杂志寻求投稿**：**StudioMilitary** 已开始筹备其第一期杂志，**重点关注开源和去中心化技术**。他们正在征集文字、文章、图片和信息图表方面的投稿，并鼓励感兴趣的人士[与其联系](https://x.com/StudioMilitary/status/1807826564970848691)。

**提到的链接**：[John Galt (@StudioMilitary) 的推文](https://x.com/StudioMilitary/status/1807826564970848691)：我正开始筹备我们杂志的第一期。总主题是开源/去中心化技术。重点展示我们世界中的乐观力量。如果你有兴趣投稿...

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1258565061295997020)** (5 messages):

> - `Achyut Benz 开发的 Apollo 项目`
> - `flask-socketio-llm-completions GitHub 仓库`
> - `foxhop 的聊天室应用演示`
> - `LLM 与 flask-socketio 的集成`

- ****Apollo 项目以 3Blue1Brown 风格可视化 AI 生成的主题****：*Achyut Benz* 推出了 [Apollo](https://x.com/achyut_benz/status/1808969030969274507?s=46)，该项目能以 **3Blue1Brown** 风格的视频可视化主题，且全部由 AI 生成。它使用 **Next.js 框架**、**GroqInc 推理**，并支持集成在 **LangChainAI** 中的 **AnthropicAI 3.5 Sonnet** 和 **OpenAI GPT-4**。
  - 受 Chris Abey 启发，该项目旨在通过 **AI 生成** 的教育视频来增强学习体验。
- ****聊天室应用通过 flask-socketio 向多个 LLM 发送消息****：*foxhop* 分享了 [flask-socketio-llm-completions](https://github.com/russellballestrini/flask-socketio-llm-completions) 的 **GitHub 仓库**，这是一个将消息发送至 **GPT**、**Claude**、**Mistral**、**Together** 和 **Groq AI** 并流式传输到前端的聊天室应用。
  - “该应用经过维护，可与各种 **LLM** 无缝协作，并展示了实时通信能力。”
- ****Foxhop 展示了集成 LLM 的聊天室应用演示****：*foxhop* 提供了一个 [演示链接](http://home.foxhop.net:5001/chat/vllm-hermes-llama-3?username=changeme)，展示集成了 **LLM** 的聊天室应用。该演示说明了消息如何与 **vLLM**、**Hermes** 和 **Llama3** 模型进行交互。
  - 该应用程序是聊天室环境下与 **LLM** 功能进行交互和实验的实用工具。

<div class="linksMentioned"><p><strong>提及的链接</strong>：</p><ul><li><a href="https://x.com/achyut_benz/status/1808969030969274507?s=46">来自 ach (@achyut_benz) 的推文</a>：介绍 apollo，这是我一直在开发的一个新项目，它以 @3blue1brown 风格的视频可视化主题或概念，全部由 AI 生成。@nextjs 框架 @GroqInc 推理支持 @Anthropic...</li><li><a href="https://github.com/russellballestrini/flask-socketio-llm-completions">GitHub - russellballestrini/flask-socketio-llm-completions：将消息发送至 GPT、Claude、Mistral、Together、Groq AI 并流式传输到前端的聊天室应用。</a>：将消息发送至 GPT、Claude、Mistral、Together、Groq AI 并流式传输到前端的聊天室应用。 - russellballestrini/flask-socketio-llm-completions</li></ul></div>

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1258153439620235345)** (110 messages🔥🔥):

> - `Replete-AI 发布的新数据集`
> - `Nomic AI 发布 GPT4ALL 3.0`
> - `InternLM-XComposer-2.5 模型发布`
> - `Claude 3.5 Sonnet 越狱面临的挑战`
> - `关于 LLM 视觉潜空间的讨论`

- **Replete-AI 发布大规模新数据集**：Replete-AI 发布了两个全新的大规模数据集，**Everything_Instruct** 和 **Everything_Instruct_Multilingual**，每个数据集大小为 11-12GB，包含超过 600 万行数据，旨在整合各种 instruct 数据，将 AI 模型训练推向新高度。[详情点击这里](https://huggingface.co/datasets/Replete-AI/Everything_Instruct_Multilingual)。
  - 这些数据集受到 bagel 数据集和 Replete-AI 的 EveryoneLLM 模型的启发，包括一个英文数据集和另一个包含多语言翻译的数据集，以增强模型的多语言能力。
- **Nomic AI 发布 GPT4ALL 3.0**：**Nomic AI** 宣布发布 **GPT4All 3.0**，这是一款开源的本地 LLM 桌面应用，支持主流操作系统上的数千个模型，具有显著的 UI/UX 改进并采用 MIT 许可证。[点击查看](https://home.nomic.ai/gpt4all)，该应用拥有超过 250,000 名月活跃用户，并具备支持本地文件对话的隐私优先功能。
- **InternLM-XComposer-2.5 设定新基准**：InternLM 发布了 **InternLM-XComposer-2.5**，这是一款通用的视觉语言大模型，支持长上下文输入和输出，使用 24K 交错的图文上下文进行训练，并能通过 RoPE 外推扩展到 96K 长上下文。[公告在此](https://x.com/_akhaliq/status/1808747694317261114?s=46)，它在 16 个基准测试中超越了现有的开源模型，并与 GPT-4V 和 Gemini Pro 展开激烈竞争。
- **越狱 Claude 3.5 Sonnet 的挫败感**：用户分享了在越狱 **Claude 3.5 Sonnet** 时面临的挑战，讨论了使用特定 pre-prompts 和角色的尝试，但该 AI 在伦理限制方面表现得非常固执。一些人建议使用 Anthropic 的 workbench 可能会有更高的成功率，但警告可能会导致账号被封禁。
- **探索 LLM 的视觉潜空间能力**：关于让 **LLM** 绘制或展示其视觉潜空间（visual latent space）的讨论不断涌现，考虑到如果经过足够的视觉数据训练，它们是否可以复现化学结构或 3D 空间等视觉元素。一些示例包括模型使用 HTML 和 CSS 生成 3D 城市，这暗示了潜力，但也指出需要包含视觉数据的数据集。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://x.com/nomic_ai/status/1808162955806097767">来自 Nomic AI (@nomic_ai) 的推文</a>：发布 GPT4All 3.0：开源本地 LLM 桌面应用 - 完全私密的体验 - 支持数千个模型和所有主流操作系统 - 重大的 UI/UX 改进 - 本地文件对话 -...</li><li><a href="https://home.nomic.ai/gpt4all">GPT4All</a>：在本地运行大语言模型：隐私优先且无需联网</li><li><a href="https://console.anthropic.com/workbench">Anthropic Console</a>：未找到描述</li><li><a href="https://x.com/localai_api/status/1808975139792425168?s=46">来自 LocalAI (@LocalAI_API) 的推文</a>：🚀 新模型提醒！查看 #internlm2，一个具有出色推理能力和 1M 上下文窗口的 7B 参数对话模型。使用 `local-ai run internlm2_5-7b-chat-1m` 在 LocalAI 中安装它 #AI #N...</li><li><a href="https://x.com/_akhaliq/status/1808747694317261114?s=46">来自 AK (@_akhaliq) 的推文</a>：InternLM-XComposer-2.5 一个支持长上下文输入和输出的多功能视觉大语言模型。我们介绍了 InternLM-XComposer-2.5 (IXC-2.5)，这是一个支持...的多功能视觉大语言模型。</li><li><a href="https://www.codedump.xyz/py/ZfkQmMk8I7ecLbIk**">未找到标题</a>：未找到描述</li><li><a href="https://x.com/_philschmid/status/1808755146190446667">来自 Philipp Schmid (@_philschmid) 的推文</a>：我之前没注意到，但看起来 (claude ai) 上的 Anthropic Claude 3.5 Sonnet 正在向用户隐藏部分回答内容，这些内容并未发送给客户端。你可以通过以下方式测试...</li><li><a href="https://pastebin.com/Gj7CpdSE">Karan4D 的 WorldSim 系统提示词开源 - Pastebin.com</a>：Pastebin.com 是自 2002 年以来排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://github.com/russellballestrini/flask-socketio-llm-completions/pull/1">跨所有聊天室进行关键词搜索以查找对话历史记录，由 russellballestrini 提交 · Pull Request #1 · russellballestrini/flask-socketio-llm-completions</a>：CodeRabbit 的总结：新功能：添加了搜索房间和消息的功能。引入了搜索结果页面以显示搜索结果。重构：精简了聊天界面...</li><li><a href="https://x.com/9mmballpoint/status/1808890582825120219">来自 RednBlackSalamander (@9mmballpoint) 的推文</a>：艺术工具</li><li><a href="https://huggingface.co/internlm/internlm-xcomposer2d5-7b">internlm/internlm-xcomposer2d5-7b · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/vllm-project/vllm/pull/5649#issuecomment-2209429032">支持允许 OpenAI API 风格工具调用和“自动”工具选择的开源模型，由 K-Mistele 提交 · Pull Request #5649 · vllm-project/vllm</a>：草案：OpenAI 工具使用检查清单。此（草案）PR 将以一种对工具使用格式和提示词格式保持最小偏见的方式，添加对 OpenAI 风格工具调用的支持。以下功能...</li><li><a href="https://www.dylandavis.net/2024/07/three-gpts-walk-into-a-bar-and-write-an-exec-summary/">三个 GPT 走进酒吧并写了一份执行摘要 – D-Squared</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/Replete-AI/Everything_Instruct_Multilingual">Replete-AI/Everything_Instruct_Multilingual · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/Replete-AI/Everything_Instruct">Replete-AI/Everything_Instruct · Hugging Face 数据集</a>：未找到描述</li></ul></div>

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1258183587006513254)** (1 条消息):

> - `使用视觉语义信息提升图像分类性能`
> - `CVPR 讨论的零样本/少样本多模态模型`
> - `应用 Florence 2 进行有监督微调`

- **通过视觉语义信息提升图像分类**：一位用户询问如何利用**视觉语义信息**之间的交互来增强细粒度图像分类性能，特别是通过有监督微调（SFT）。他们提到了 **Florence 2** 在此用途上的潜在应用。
- **CVPR 重点关注零样本/少样本多模态模型**：在 **CVPR** 上，大量论文聚焦于零样本/少样本多模态模型，展示了利用视觉和文本数据的兴趣。一位从事计算机视觉工作的用户寻求在实际的有监督环境中使用这些研究的建议。

---

### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1258342665636806678)** (8 messages🔥):

> - `crossover with pipelines, flows, and agents`
> - `rag dataset as 0 shot context ingestion`
> - `context and metadata for llm`
> - `HF tool processing corpus queries against hf datasets`
> - `keyword matching for relevance score and filtering`

- ****Crossover with pipelines, flows, and agents****：**Pipelines, flows, 和 agents** 正在融合，其想法是让 RAG dataset 主要用于 **0 shot context ingestion**，后期侧重于基于 Agent 的处理。
  - *interstellarninja* 提到，即使在 RAG 开发期间，将交叉融合纳入 Agentic flows 也是有益的。
- ****HF Tool Processing and Keyword Matching****：描述了一个 **HF tool**，它可以处理针对 HF datasets 的语料库查询，利用**倒排索引（inverted index）进行关键词匹配**，并将其转换为带有 Metadata 的 **.jsonl** 文件。
  - *@everyoneisgross* 提到该界面允许使用 Gradio 编辑生成内容，关键词搜索在 Toy prompting 中表现良好。

---

### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1258326477343035463)** (10 messages🔥):

> - `Users discussing lack of credits to use WorldSIM`
> - `Issues with using GPT-3.5 on WorldSIM`
> - `Prompt engineering for different models on WorldSIM`
> - `Positive feedback about WorldSIM`
> - `Buddhist world simulation on WorldSIM`

- **WorldSIM 用户额度耗尽**：一位用户建议解释 **WorldSIM** 的额度限制，建议使用类似“额度不足以使用”的标题或使用红色文字标示“无额度（NO CREDITS）”。这有助于避免新用户的困惑。
- **对 WorldSIM 上的 GPT-3.5 感到沮丧**：几位成员表达了在 **WorldSIM** 上使用 **GPT-3.5** 的挫败感，提到它经常在最终正常工作前返回单行答案。一位用户抱怨为了启动而浪费了多条消息的额度。
- **WorldSIM 模型的新 Prompt Engineering**：讨论透露 **WorldSIM** 正在为不同模型开发新的 Prompt engineering。一位成员提到，针对不同模型分离 Prompt 的工作正在进行中（**WIP**）。
- **成员赞扬 WorldSIM**：一位成员表示 **WorldSIM** “非常疯狂（bonkers）”，并祝贺团队做得出色。另一位成员分享了他们在午休时间用完所有额度来创建一个根植于佛教原则的世界的经历。

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1258788212323778652)** (1 messages):

> - `Simple Telegram bot to interface with different AI models`
> - `First 1000 responses free on the bot`

- ****尝试用于 AI 模型交互的 Mysticella Bot****：创建了一个[简单的 Telegram bot](https://t.me/mysticella_bot)来与不同的 AI 模型交互。**前 1000 次响应**免费。
- ****Telegram Bot 前 1000 次响应免费****：查看新的 Telegram bot **Mysticella**，进行免费的 AI 模型交互。**前 1000 次响应**完全免费。

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1258136927434051644)** (107 条消息🔥🔥):

> - `OpenRouter 中已部署 LLM 模型的量化 (Quantisation)`
> - `Microsoft 的 API 变更影响 OpenRouter`
> - `Infermatic 的隐私政策更新`
> - `DeepSeek Coder 公式渲染问题`
> - `Mistral Codestral API 的定价与性能`

- **澄清 LLM 模型量化困惑**：根据用户的解释，除非提供商另有说明，否则 OpenRouter 的 LLM 模型均以 **FP16/BF16** 部署。另一位用户澄清了**量化图标**的存在，该图标指示了模型的量化状态。
- **Microsoft API 变更影响 OpenRouter**：**Microsoft 对其被 OpenRouter 使用的 API 引入了破坏性变更**，但补丁已迅速部署。用户反馈称赞了快速的响应和修复。
- **Infermatic 澄清隐私政策**：正如其修订后的[隐私政策](https://infermatic.ai/privacy-policy/)所述，**Infermatic 不记录任何输入提示词 (prompts) 或模型输出**，仅实时处理数据。与之前暗示可能保留数据的旧政策相比，用户对此感到放心。
- **DeepSeek Coder 公式问题已解决**：用户遇到了 **DeepSeek Coder** 中公式无法正确渲染的问题，尽管一位用户通过使用正则表达式处理输出字符串找到了解决方案。另一位用户报告系统提示词在 TypingMind 的前端无法正确处理，并提交了该问题以供审查。
- **Mistral Codestral API 定价遭到批评**：用户对 **Mistral 的 Codestral API** 定价表示不满，认为对于一个 22B 模型来说价格过高。推荐使用 **DeepSeek Coder** 等替代方案，以获得更好的成本效益和相当的代码性能。

<div class="linksMentioned"><p><strong>提及的链接</strong>：</p><ul><li><a href="https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard">UGI Leaderboard - 由 DontPlanToEnd 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://openrouter.ai/models/sao10k/l3-euryale-70b">sao10k 开发的 Llama 3 Euryale 70B v2.1</a>：Euryale 70B v2.1 是一款专注于创意角色扮演的模型，来自 [Sao10k](https://ko-fi.com/sao10k)。- 更好的提示词遵循能力。- 更好的解剖学/空间意识。- 能更好地适应独特且复杂的...</li><li><a href="https://openrouter.ai/docs/limits">限制 | OpenRouter</a>：设置模型使用限制</li><li><a href="https://docs.mistral.ai/capabilities/code_generation/">代码生成 | Mistral AI 大语言模型</a>：Codestral 是一款尖端的生成式模型，专门为代码生成任务设计和优化，包括 fill-in-the-middle 和代码补全。Codestral 在 80 多种...</li><li><a href="https://www.baseten.co/blog/llm-transformer-inference-guide/">LLM 推理与性能指南</a>：了解 LLM 推理是计算受限还是内存受限，以充分利用 GPU 算力。获取有关更好利用 GPU 资源的见解。</li><li><a href="https://github.com/SillyTavern/SillyTavern/blob/release/src/prompt-converters.js#L86">SillyTavern/src/prompt-converters.js (release 分支) · SillyTavern/SillyTavern</a>：面向高级用户的 LLM 前端。通过在 GitHub 上创建账号为 SillyTavern/SillyTavern 做出贡献。</li><li><a href="https://web.archive.org/web/20240112082806/https://infermatic.ai/privacy-policy/">隐私政策 - Infermatic</a>：未找到描述</li><li><a href="https://aistudio.google.com/app/prompts/new_chat?pli=1">未找到标题</a>：未找到描述</li><li><a href="https://infermatic.ai/privacy-policy/">隐私政策 - Infermatic</a>：未找到描述</li><li><a href="http://llum.chat">lluminous</a>：未找到描述</li></ul></div>

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1258299318939549696)** (42 条消息🔥):

> - `排行榜上的失败任务`
> - `生成式模型的 Checksum`
> - `用于模型指纹识别的拓扑数据分析 (TDA)`
> - `1.58 bit LLM 论文及其实现`
> - `VQ-VAE 对后验崩溃 (posterior collapse) 的免疫性`

- ****排行榜任务问题浮出水面****：一名成员询问了 [Hugging Face 排行榜](https://huggingface.co/datasets/open-llm-leaderboard/requests/blob/8c010a41f0b5f726199183bbad05f1649a362adf/cognitivecomputations/dolphin-2.9.2-qwen2-72b_eval_request_False_bfloat16_Original.json#L9) 上失败的任务以及是否可以重新添加。
- ****关于生成式模型 Checksum 的辩论****：讨论了在使用 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 时，对于像 `LlamaForCausalLM` 这样的生成式模型是否存在类似 Checksum 的指标，并指出了基准测试与 Checksum 之间的差异。
- ****探索用于模型指纹识别的 TDA****：成员们深入研究了使用拓扑数据分析 (TDA) 通过测量拓扑不变量来识别模型指纹，并参考了 [TorchTDA](https://giotto-ai.github.io/gtda-docs/0.5.1/library.html) 等工具。
  - *“你有没有研究过拓扑数据分析？你可能可以通过使用 TDA 根据权重的固有拓扑不变量来对其进行分析。”*
- ****实现 1.58-bit LLM 创新****：一名成员寻求关于采用 [1.58-bit LLM 论文](https://arxiv.org/abs/2402.17764) 中的技术来量化权重和激活以提高成本效益的指导。
  - 他们计划在 Pythia 等预训练模型中用 “BitLinear” 层替换线性层，以测试量化权重训练。
- ****PDF 标记工具的困扰****：一名成员对缺乏具有“搜索 -> 全部标记”功能的 PDF 标记工具表示沮丧，并提到了 Bluebeam 和 PDF Studio 等昂贵的选项。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://giotto-ai.github.io/gtda-docs/0.5.1/library.html">Overview — giotto-tda 0.5.1 文档</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/open-llm-leaderboard/requests/blob/8c010a41f0b5f726199183bbad05f1649a362adf/cognitivecomputations/dolphin-2.9.2-qwen2-72b_eval_request_False_bfloat16_Original.json#L9">cognitivecomputations/dolphin-2.9.2-qwen2-72b_eval_request_False_bfloat16_Original.json · open-llm-leaderboard/requests at 8c010a41f0b5f726199183bbad05f1649a362adf</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2405.18432">On the Origin of Llamas: Model Tree Heritage Recovery</a>：互联网上共享的神经网络模型的快速增长使得模型权重成为一种重要的数据模态。然而，由于权重不可解释，这些信息并未得到充分利用...</li><li><a href="https://arxiv.org/abs/2402.17764">The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits</a>：最近的研究（如 BitNet）正在为 1-bit 大语言模型 (LLMs) 的新时代铺平道路。在这项工作中，我们引入了一个 1-bit LLM 变体，即 BitNet b1.58，其中每一个参数...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness">GitHub - EleutherAI/lm-evaluation-harness: 语言模型少样本评估框架。</a>：语言模型少样本评估框架。- EleutherAI/lm-evaluation-harness</li></ul></div>

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1258310457001775146)** (28 messages🔥):

> - `Diffusion forcing 用于规划`
> - `与 Nathan Frey 的 walk_jump 方法的比较`
> - `讨论新的研究策略`
> - `LLM 的持续预训练 (Continual pre-training)`
> - `具有不同同伦类 (homotopy classes) 的函数逼近`

- **Diffusion Forcing 在规划中展现出潜力**：一位成员分享了一段[视频](https://boyuan.space/diffusion-forcing/static/videos/planning/planning.mp4)，展示了用于规划的 Diffusion forcing，引发了大量关注和积极反馈，*“说实话，结果真的很酷”*。
- **Diffusion Forcing 对比 Walk-Jump 方法**：关于 Diffusion forcing 是否会优于 Nathan Frey 的 [walk_jump 方法](https://www.youtube.com/watch?v=O3YBEnvvPZY) 的讨论得出结论，它们可能是具有不同机制的正交技术。
- **有效的论文阅读策略**：一位成员询问了紧跟新研究的策略，得到的建议是：在 ArXiv 论文发布时进行快速浏览，并有系统地筛选重要论文是关键。
- **大语言模型 (LLM) 的持续预训练**：最近关于持续预训练的研究观察到 LLM 在适应新领域时性能存在 **“稳定性差距” (stability gap)**，并提出了[三种策略](https://arxiv.org/abs/2406.14833)来缓解这一问题。
- **函数逼近中的同伦类 (Homotopy Classes)**：一位成员询问了在函数逼近过程中，使每个基函数的图像属于不同同伦类的好处，特别是在旋转轨迹建模中。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://www.youtube.com/watch?v=O3YBEnvvPZY">使用离散 Walk-Jump 采样进行蛋白质发现 | Nathan Frey</a>：Portal 是 AI 药物发现社区的家园。加入以获取有关此演讲的更多详细信息并与演讲者联系：https://portal.valencelabs.co...</li><li><a href="http://arxiv.org/abs/2306.12360">使用离散 Walk-Jump 采样进行蛋白质发现</a>：我们通过学习平滑的能量函数，使用 Langevin Markov chain Monte Carlo 从平滑的数据流形中采样，解决了离散生成模型在训练和采样方面的困难...</li><li><a href="https://arxiv.org/abs/2406.14833">通过缓解稳定性差距实现高效的持续预训练</a>：持续预训练已日益成为将大语言模型 (LLM) 适应新领域的主流方法。这一过程涉及使用来自...的语料库更新预训练的 LLM。</li></ul></div>

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1258145861884575765)** (5 messages):

> - `Chinchilla 仓库中的 efficientcube.ipynb`
> - `JAX 中的 XLA 能力`
> - `Flax 中 JIT 函数的 FLOPs 估算`
> - `临界 Batch Size 与性能退化`

- ****Chinchilla 中的 EfficientCube Notebook****：一个名为 **efficientcube.ipynb** 的 [Scaling Law 研究工具包](https://github.com/kyo-takano/chinchilla/blob/master/examples/efficientcube.ipynb) 已添加到 Chinchilla 仓库中。该 Notebook 包含与 Scaling 研究活动相关的实用程序。
- ****JAX 新增 AOT 编译能力****：[JAX](https://jax.readthedocs.io/en/latest/aot.html#debug-information-and-analyses-when-available) 现在除了 JIT 编译外，还支持提前编译 (AOT)。这允许用户在执行前编译代码，从而对编译过程提供更多控制。
- ****分享 Flax FLOPs 估算方法****：在 [GitHub 讨论](https://github.com/google/flax/discussions/1854#discussioncomment-4758695) 中分享了一个用于估算 Flax 中 **JIT 函数 FLOPs** 的代码片段。该方法利用了 JAX 内部的 XLA 能力进行精确的性能测量。
- ****重新评估临界 Batch Size 理论****：最近的研究结果表明，在低于某个**最优 Batch Size** 时，性能会下降，这与**传统观点**（即任何低于临界值的 Batch Size 都是好的）相矛盾。这在理论上很有趣，但在大规模应用中并不显著。

<div class="linksMentioned"><p><strong>提及的链接</strong>：</p><ul><li><a href="https://jax.readthedocs.io/en/latest/aot.html#debug-information-and-analyses-when-available">Ahead-of-time lowering and compilation — JAX documentation</a>：未找到描述</li><li><a href="https://github.com/google/flax/discussions/1854#discussioncomment-4758695">How do you access XLA's flop estimate for a jitted program? · google/flax · Discussion #1854</a>：目前的方法如下：In [1]: import jax, jax.numpy as jnp In [2]: m = jax.xla_computation(lambda x, y: x @ y)(jnp.ones((1000, 1000)), jnp.ones((1000,1000))).as_hlo_module() In [3]: clien...</li><li><a href="https://github.com/kyo-takano/chinchilla/blob/master/examples/efficientcube.ipynb">chinchilla/examples/efficientcube.ipynb at master · kyo-takano/chinchilla</a>：一个用于 Scaling Law 研究的工具包 ⚖。通过在 GitHub 上创建账号为 kyo-takano/chinchilla 的开发做出贡献。</li></ul></div>

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1258144662800044155)** (3 messages):

> - `Llama 3 8B 上的 SAEs`
> - `稀疏自编码器 (Sparse autoencoders)`
> - `残差流 (Residual stream) 处理`

- ****Llama 3 8B 上的 SAEs 已训练完成****：[在 Llama 3 8B 残差流上训练的稀疏自编码器 (SAEs)](https://huggingface.co/EleutherAI/sae-llama-3-8b-32x) 现在已可用。这些 SAEs 使用 **RedPajama 语料库**，并可以使用 EleutherAI 的 **`sae` 库**加载。
  - *目前未追踪该模型的下载量。*
- ****使用 SAEs 进行残差流处理****：该项目**按层组织 SAEs**，并将其与 Llama 3 8B 模型集成，以更有效地处理残差流。更多详情请参阅 [Model Card](https://huggingface.co/EleutherAI/sae-llama-3-8b-32x)。

**提及的链接**：[EleutherAI/sae-llama-3-8b-32x · Hugging Face](https://huggingface.co/EleutherAI/sae-llama-3-8b-32x)：未找到描述

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1258161481808351432)** (18 messages🔥):

> - `Preprocessing Function Optimization` (预处理函数优化)
> - `Proof-Pile Config Error` (Proof-Pile 配置错误)
> - `Metric Inconsistencies in Config` (配置中的指标不一致)
> - `Long Model Names Issue` (模型名称过长问题)
> - `Evaluating Model in Parallel` (并行评估模型)

- **预处理缓存替代方案**：一位用户询问是否可以在将预处理后的问题/参数输入模型之前进行保存，以避免每次都重复运行预处理函数。
- **Proof-Pile 配置错误解决**：一位用户在使用特定配置文件运行 `proof-pile` 任务时遇到错误。切换到 `lambada_openai` 后可以正常工作，这表明数据集本身可能存在问题。
- **识别到配置中的指标不匹配**：配置中使用 `loglikelihood_rolling` 但实际调用了 `loglikelihood` 引起了困惑，这很可能是由于指标不一致导致的。**loglikelihood 指标：** `perplexity` vs `word_perplexity`, `byte_perplexity`, `bits_per_byte`。
- **长模型名称导致保存问题**：一位用户遇到了由于模型名称过长导致文件和目录无法正确写入的保存问题。返回的错误为 **OSError(36, 'File name too long')**。
- **并行评估设置咨询**：一位用户询问如何在通过 `pretrained` 参数传递模型时以并行化方式评估模型。**收到的警告：** 'assuming single-process call to evaluate() or custom distributed integration'。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://huggingface.co/spaces/open-llm-leaderboard/GenerationVisualizer">Exploring model generations - a Hugging Face Space by open-llm-leaderboard</a>：暂无描述</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/2010">Added MedConceptsQA Benchmark by Ofir408 · Pull Request #2010 · EleutherAI/lm-evaluation-harness</a>：嗨，我添加了名为 MedConceptsQA 的新基准测试。MedConceptsQA 是一个专门用于医学概念问答的开源基准测试。该基准测试包含各种问题...</li></ul></div>

---

### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/)** (1 messages):

wendlerc: 有人有好的 SDXL latent downscaler 吗？我想从 128x128x4 降采样到 64x64x4。

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1258139491739238462)** (75 条消息🔥🔥):

> - `使用 LangChain 的困难`
> - `OpenAI 与 ChatOpenAI 之间的偏好`
> - `PeopleGPT 和 Juicebox.ai 的功能`
> - `用于预约演示的 RAG 架构`
> - `LangChain 性能问题与改进`

- ****使用 LangChain 的理由与疑虑****：一位成员表达了使用 **LangChain** 的困难并质疑其效用，理由是响应时间长以及处理过程中的不必要步骤，尤其是在本地 CPU 上运行时。
  - 另一位用户指出，这可能是模型的推理性能问题，或者仅仅是因为在没有 GPU 的情况下运行，导致了诸如过度无关搜索之类的低效行为。
- ****OpenAI vs. ChatOpenAI****：对比了 **OpenAI** 和 **ChatOpenAI** 在任务执行中的表现，一位用户询问了优缺点，并指出 **OpenAI** 可能会被弃用，转而支持 **ChatOpenAI**。
  - 几位成员澄清说，根据具体需求和实现上下文，存在不同的使用体验。
- ****Juicebox.ai 中的 PeopleGPT 大放异彩****：一位成员讨论了由 **PeopleGPT** 驱动的 **Juicebox.ai**，这是一个基于自然语言的搜索引擎，用于在不使用布尔值（Booleans）的情况下寻找合格人才，并在此处提供了易于点击的示例 [here](https://juicebox.ai/)。
  - 讨论集中在技术功能上，强调它将过滤器与搜索相结合以增强用户体验。
- ****LangChain 处理 CSV 文件的问题****：一位用户寻求在 **LangChain** 中处理多个 CSV 文件的最新方法，并指出更新后在处理两个以上文件时存在局限性。
  - 该成员回顾了之前模块的有效性，并询问了旨在实现最佳性能和集成的现代替代方案。
- ****使用 LangChain 预约演示的挑战****：一位成员在尝试使用 LangChain 和 RAG 架构在聊天机器人中加入演示预约功能时遇到困难，提到了 **SlackScheduleMessage** 等工具。
  - 讨论了来自 LangChain 社区提供的详细步骤以寻求可能的解决方案，并强调需要进一步的社区投入。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://python.langchain.com/v0.2/docs/concepts/#messages">概念指南 | 🦜️🔗 LangChain</a>：本节包含 LangChain 核心部分的介绍。</li><li><a href="https://api.python.langchain.com/en/latest/_modules/langchain_core/messages/human.html#HumanMessage">langchain_core.messages.human — 🦜🔗 LangChain 0.2.6</a>：未找到描述</li><li><a href="https://github.com/langchain-ai/langchain/issues/23881">Agent 与 GraphQL - 401 Client Error: Unauthorized for url: https://streaming.bitquery.io/eap · Issue #23881 · langchain-ai/langchain</a>：检查了其他资源，我为此 Issue 添加了一个非常详细的标题。我使用集成搜索搜索了 LangChain 文档。我使用 GitHub 搜索来寻找类似的问题并...</li><li><a href="https://api.python.langchain.com/en/latest/agents/langchain.agents.tool_calling_agent.base.create_tool_calling_agent.html">langchain.agents.tool_calling_agent.base.create_tool_calling_agent — 🦜🔗 LangChain 0.2.6</a>：未找到描述</li><li><a href="https://juicebox.ai/">Juicebox (PeopleGPT) - AI 驱动的人才搜索领导者。</a>：探索 PeopleGPT，一个了解你想找谁的搜索引擎。使用自然语言实时搜索超过 8 亿个个人资料。获取联系方式并建立推广活动...</li><li><a href="https://x.com/levelsio/status/1804078191385956668">来自 @levelsio (@levelsio) 的推文</a>：我建议大家不要使用 LangChain，这篇文章解释得很清楚。它在抽象之上又使用了抽象，实际上让你的代码变得不必要的复杂。只需编写 API 调用和...</li><li><a href="https://www.octomind.dev/blog/why-we-no-longer-use-langchain-for-building-our-ai-agents">为什么我们不再使用 LangChain 构建 AI Agent</a>：当抽象弊大于利时——在生产环境中使用 LangChain 的经验教训以及我们本该怎么做</li><li><a href="https://www.dylandavis.net/2024/07/three-gpts-walk-into-a-bar-and-write-an-exec-summary/">三个 GPT 走进酒吧并写了一份执行摘要 – D-Squared</a>：未找到描述</li></ul></div>

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1258353115384774686)** (3 条消息):

> - `Adding demo scheduling feature to chatbot using the RAG architecture and LangChain framework`
> - `Blogpost on creating an E2E Image Retrieval app using Lightly SSL and FAISS`
> - `Beta testing for advanced research assistant and search engine with premium model access`

- **RAG 聊天机器人需要演示预约功能**：一位成员寻求社区帮助，为其基于 **RAG 架构**和 **LangChain 框架**构建的聊天机器人添加**演示预约（demo scheduling）**功能。
- **Lightly SSL 和 FAISS 驱动图像检索应用**：分享了一篇关于使用 **Lightly SSL** 和 **FAISS** 创建 **E2E 图像检索应用**的博客文章，内容包括实现 Vision Transformer 模型和创建向量嵌入（vector embeddings）。详细的博客文章包含一个 [Colab Notebook](https://colab.research.google.com/drive/1n4CwX5T6Ch2v7OYTRe6g1j_QJHxxOvcM) 和一个 [Gradio 应用](https://huggingface.co/spaces/lightly-ai/food101-image-retrieval)。
- **Rubik's AI 提供免费 Beta 测试**：发布了一个**高级研究助手**和搜索引擎的 Beta 测试邀请，提供为期 **2 个月的免费高级访问权限**，可使用 Claude 3 Opus、GPT-4o 等模型。
  - 用户被提示使用促销代码 'RUBIX' [注册](https://rubiks.ai/) 以获取免费试用。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://rubiks.ai/">Rubik's AI - AI 研究助手 &amp; 搜索引擎</a>：未找到描述</li><li><a href="https://www.lightly.ai/post/vector-indexes-and-image-retrieval-using-lightly">使用 lightly 进行向量索引和图像检索</a>：使用 Lightly 提供的预训练 Vision Transformer，在任意数据集上创建向量索引，以便使用 FAISS 进行图像检索</li><li><a href="https://huggingface.co/spaces/lightly-ai/food101-image-retrieval">Food101 图像检索 - lightly-ai 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://x.com/MaheshkarSaurav/status/1808881869829853305">Saurav Maheshkar ☕️ (@MaheshkarSaurav) 的推文</a>：🚀 @LightlyAI 的最新工作。了解如何使用 FAISS (@AIatMeta) 作为向量索引 🗃️、Lightly SSL 软件包的模型实现以及 @weights_biases 来创建图像检索应用...</li><li><a href="https://colab.research.google.com/drive/1n4CwX5T6Ch2v7OYTRe6g1j_QJHxxOvcM?usp=sharing">Google Colab</a>：未找到描述</li></ul></div>

---

### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/)** (1 条消息):

dievas_: [https://www.youtube.com/watch?v=yF9kGESAi3M](https://www.youtube.com/watch?v=yF9kGESAi3M) 试试这个

---

### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1258212529507663915)** (1 条消息):

> - `Next webinar on RAG experimentation/evaluation with LlamaIndex and Weights and Biases`
> - `Announcements about the timing and focus of the upcoming webinar`
> - `Complex challenge of aligning LLM Judge for accurate evaluation`

- **下场网络研讨会：对齐你的 LLM Judge**：下周三太平洋时间上午 9 点，参加关于使用 [LlamaIndex 和 Weights and Biases](https://lu.ma/dywrdye5) 进行 **RAG 实验/评估**的原则性方法的网络研讨会。请通过提供的链接注册以预留名额。
- **对齐 LLM Judge 的复杂挑战**：本次研讨会将探讨各种**评估策略**，重点是以 **RAG Pipeline** 为案例研究来对齐你的 LLM Judge。它还将演示如何利用 **Weights and Biases Weave** 进行系统性评估。

**提到的链接**：[LlamaIndex 网络研讨会：使用 LlamaIndex 和 W&B Weave 对齐你的 LLM Judge · Zoom · Luma](https://lu.ma/dywrdye5)：虽然现在创建一个 RAG Pipeline 非常简单，但对齐你的 LLM Judge 以进行准确评估仍然是一个复杂的挑战。在本次研讨会中，我们将深入探讨……

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1258148071141933167)** (4 条消息):

> - `新网络研讨会：RAG 实验与评估的原则性方法`
> - `Reflection as a Service`
> - `成为顶尖 AI Engineer 和教育者`
> - `Corrective RAG as a Service`

- **Webinar：与 Weights & Biases 合作 RAG**：LlamaIndex 宣布与 **Weights & Biases** 合作举办一场 [网络研讨会](https://twitter.com/llama_index/status/1808589017744880062)，展示如何构建、评估和迭代 RAG 流水线。这总结了 1 年多来的 RAG 开发经验，并指出适当的评估仍然具有挑战性。
- **通过 Reflection as a Service 确保可靠性**：LlamaIndex 讨论了 “Reflection as a Service” 的概念，通过实现反思步骤来在输出错误时进行自我修正，从而解决 Agent 应用中的可靠性问题。该解决方案旨在防止 LLM 产生有问题的输出。
- **顶尖 AI Engineer：@ravithejads 的历程**：LlamaIndex 重点介绍了社区成员 @ravithejads 的历程，他通过热情、OSS 贡献以及紧跟最新 AI 趋势，成为了一名 **Developer Advocate**。分享他的故事旨在激励他人在 AI 工程和教育领域脱颖而出。
- **发布 Corrective RAG as a Service**：LlamaIndex 宣布发布由 **Yan 等人** 开发的 [Corrective RAG (CRAG)](https://twitter.com/llama_index/status/1809282069606068486)，该技术在生成步骤之前使用 Web Search 动态验证检索到的上下文，并在不相关时进行纠正。

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1258263851745546300)** (71 条消息🔥🔥):

> - `包含多模型加载的 Google Cloud Function 推理流水线`
> - `Cohere command r+ 的性能对比`
> - `在 LlamaIndex 中结合 RAG 实现对话记忆`
> - `在不从文件系统存储/加载的情况下使用混合检索器 (hybrid retrievers)`
> - `用于 “穷人版 RLHF” 的 Few-shot 示例技术`

- ****Google Cloud Function 推理流水线中的多模型加载****：一位用户表达了在 Google Cloud Function 上加载阿里巴巴 NLP 嵌入模型和 Llama3 LLM 进行推理时遇到的问题，面临重复加载时间过长的情况。他们询问了直接从 Vertex AI 加载嵌入的替代方案，并收到了建议，但目前尚无具体解决方案。
- ****在 LlamaIndex 中处理对话记忆****：一位用户寻求在 LlamaIndex 中避免过度使用对话记忆的方法，并收到了通过改进 Prompt Engineering 来缓解该问题的建议。他们同意修改系统提示词 (system prompt) 可能会有所帮助。
- ****不使用文件系统存储的混合检索器用法****：一位用户询问了如何在不使用文件系统存储的情况下实现混合检索器 (hybrid retriever)，建议包括为稀疏向量编写 BM25 算法并将其存储在向量数据库中。讨论还提到了未来对 BM42 的探索，以及 LlamaIndex 支持所需的小幅调整。
- ****处理大模型与量化****：一位用户讨论了由于 GPU 限制，在使用 'gte-Qwen2-7B-instruct' 和 'BAAI/bge-large-en-v1.5' 嵌入模型时面临的挑战。他们计划测试量化嵌入模型，并了解到如果维度匹配，这两个模型都可以使用。
- ****本地 LLM、GPT4All 与过时的文档****：用户对文档中过时的示例和链接表示担忧。分享了关于使用本地 LLM 的最新信息，并指出欢迎为更新文档做出贡献。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://Your_url_here',api_key=" ")"="">未找到标题</a>：未找到描述</li><li><a href="https://qdrant.tech/articles/bm42/#">BM42: 混合搜索的新基准 - Qdrant</a>：介绍 BM42 —— 一种新的稀疏嵌入方法，它结合了精确关键词搜索的优势与 Transformer 的智能。</li><li><a href="https://x.com/mathemagic1an/status/1617606970114179072">Jay Hack (@mathemagic1an) 的推文</a>："穷人版 RLHF" 1) 让用户指示模型何时正确 2) 将相关的 (输入, 输出) 存储在嵌入索引中 3) 在推理时，检索最近的 K 个先前输入 4) 将其放入...</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/node_parsers/semantic_double_merging_chunking/">语义双重合并分块 - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/microsoft/graphrag">GitHub - microsoft/graphrag: 一个模块化的基于图的检索增强生成 (RAG) 系统</a>：一个模块化的基于图的检索增强生成 (RAG) 系统 - microsoft/graphrag</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/">入门教程 (本地模型) - LlamaIndex</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/16QMQePkONNlDpgiltOi7oRQgmB8dU5fl?usp=sharing#scrollTo=20cf0152">Google Colab</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/schema/#llama_index.core.schema.IndexNode.from_text_node>).">Index - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_context/">聊天引擎 - 上下文模式 - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/retrievers/recursive_retriever_nodes/#chunk-references-smaller-child-chunks-referring-to-bigger-parent-chunk>).">递归检索器 + 节点引用 - LlamaIndex</a>：未找到描述</li></ul></div>

---

### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1258237090370551909)** (45 messages🔥):

> - `关于参加伦敦活动资格的讨论`
> - `在生产环境中使用 Cohere 的 rerank API 部署应用时遇到的问题`
> - `新成员介绍`
> - `教授 AI 和高级开发`
> - `致力于 AI-Plans，一个用于对齐计划红队测试（red teaming）的同行评审平台`

- ****伦敦活动无需资格限制****：一位成员询问参加伦敦活动是否需要特定资格，其他成员澄清说**没有先决条件要求**，任何人都可以通过填写表格参加。*参加社区活动不需要 PhD 学位*是其中的核心信息。
- ****生产环境中的 Rerank API 错误****：一位成员反映在生产环境中部署使用 **rerank API** 的应用时出现了 **TypeError**，而该应用在本地运行正常。另一位成员指出该问题似乎与 Cohere 无关，并索要 Streamlit 脚本以便进一步诊断。
- ****新成员自我介绍****：几位新成员（包括一名最近毕业的**计算机科学专业毕业生**和一名对教学感兴趣的 AI 开发者）介绍了自己，并表达了加入社区的兴奋之情。他们分享了自己的背景以及希望在 Discord 社区中实现的愿景。
- ****教授 AI 和高级开发****：一位成员表达了对**教授 AI 和高级开发**的浓厚兴趣，并邀请他人联系合作。这得到了积极响应，另一位成员公开表示很快会向其寻求专业建议。
- ****AI-Plans 平台****：一位成员透露正在开发 **AI-Plans**，这是一个用于**对齐计划红队测试（red teaming）**的同行评审平台。这引起了大家的兴趣，并欢迎他们进一步讨论该项目。

---

### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1258377613630505061)** (17 messages🔥):

> - `在 Cohere 博客上推荐教程`
> - `介绍 Command R+，强大的新模型`
> - `使用 Rhea.run 创建趣味小应用（toy apps）`
> - `Rhea.run 中的新功能“保存到项目”`

- ****Cohere 博客上的推荐教程****：一位成员表示有兴趣在 [Cohere 博客上推荐教程](https://cohere.com/blog/build-a-smart-slack-bot-with-language-models)，并分享了一篇旧博文以及 [GitHub](https://github.com/cohere-samples/cohere-slack-starter-app) 上的 Slack 机器人启动代码。另一位成员确认他们将直接跟进。
- ****使用 Rhea.run 开发趣味小应用****：成员们讨论了使用 [Rhea.run](https://rhea.run) 创建趣味小应用，并注意到它可以通过要求其设计 HTML 脚本来生成交互式应用程序的能力。
- ****介绍 Command R+****：Cohere [宣布发布 Command R+](https://cohere.com/blog/command-r-plus-microsoft-azure)，这是 Command R 家族中功能最强大的模型，现在已开放使用。
- ****Rhea.run 的新功能****：[Rhea.run](https://rhea.run) 引入了新的“保存到项目”（Save to Project）功能，允许用户通过对话设计 HTML 脚本来创建交互式应用程序。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://rhea.run">Rhea | Byte Breeze Studios</a>：未找到描述</li><li><a href="https://cohere.com/blog/build-a-smart-slack-bot-with-language-models">使用语言模型构建智能 Slack 机器人</a>：你是否曾想过构建一个智能 Slack 机器人？有许多方法可以为 Slack 或 Discord 机器人注入智能。启动代码：https://github.com/cohere-samples/cohere-slack-starter-ap...</li><li><a href="https://github.com/cohere-samples/cohere-slack-starter-app">GitHub - cohere-samples/cohere-slack-starter-app: 由 Co:here 驱动的 Slack 应用入门项目</a>：由 Co:here 驱动的 Slack 应用入门项目。通过在 GitHub 上创建账号来为 cohere-samples/cohere-slack-starter-app 的开发做出贡献。</li></ul></div>

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1258170247358316655)** (57 条消息🔥🔥):

> - `关于解释器输出的技术问题`
> - `关于新 MacOS Copilot —— Invisibility 的讨论`
> - `Open Interpreter (OI) 安全性的认可与进展`
> - `Open Interpreter 的新调试功能`
> - `每月 House Party 活动`

- ****Invisibility：MacOS Copilot 受到关注****：成员们讨论了新的 [Invisibility MacOS Copilot](https://x.com/sulaimanghori/status/1791113392482377833)，它使用了 GPT-4, Gemini 1.5 Pro 和 Claude-3 Opus，强调了其免费可用性以及无缝上下文吸收等特性。语音、长期记忆和 iOS 版本的开发正在进行中。
  - 有人表达了将类似工具集成到 OI 生态系统的兴趣，一名成员建议将之前的项目 grav.ai 开源。
- ****Open Interpreter 实现调试命令****：一位用户兴奋地报告说，Open Interpreter 现在可以自动将 [VSC 主题从浅色模式切换为深色模式](https://discord.com/invite/YQhmv5pd?event=1258399216078684242)，展示了其在没有显式编程的情况下执行某些操作的能力。该功能被称为 'wtf' 命令，允许在终端中调试错误并建议修复方案。
  - 这一新实现的功能引起了不小的轰动，成员们分享了他们的惊叹，并对持续的改进表示支持。
- ****对 OI 安全措施的认可****：一位成员称赞了 OI 团队对安全性的投入，提到了一次会议，会上讨论了各种改进系统安全模型的想法和建议。团队将安全性作为优先事项的承诺受到了高度赞赏。
  - 提到了未来安全圆桌会议的计划，并承诺向社区更新日期和持续的努力。
- ****每月 House Party 回顾****：社区庆祝了 [OI 7 月 4 日 House Party](https://discord.com/invite/YQhmv5pd?event=1258399216078684242) 的成功，活动展示了新的 Demo、新面孔以及即将发布的更新预览。下一次活动定于 8 月 1 日。
  - 成员们表达了对活动的喜悦和感谢，强调了它在促进社区参与和协作方面的作用。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://x.com/sulaimanghori/status/1791113392482377833">来自 SKG (ceo @ piedpiper) (@sulaimanghori) 的推文</a>：我们在过去几周一直在努力。很高兴终于揭晓 Invisibility：专用的 MacOS Copilot。由 GPT4o, Gemini 1.5 Pro 和 Claude-3 Opus 提供支持，现在免费提供 -&gt; @inv...</li><li><a href="https://web.archive.org/web/20240418151656/https://grav.ai/">Gravity</a>：你的个人 AI。</li></ul></div>

---

### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1258565540935630912)** (2 条消息):

> - `01 Light 发货更新`
> - `01 Light 发货延迟`

- ****01 Light 发货更新****：成员们表达了对 **01 Light 发货** 的期待，其中一人 *希望很快能有更新*。另一位成员表达了挫败感，称他们已经 *等了很久*。
- ****对发货延迟的挫败感****：一位成员表达了对 **01 Light** 漫长等待的不满。另一位成员也表达了同样的看法，反映了集体的挫败感。

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1258137941260238868)** (5 条消息):

> - `Mojo 中的类型转换 (casting) Bug 讨论`
> - `Mojo 对象与 Python 对象的比较`
> - `关于在 EDx 开设 Mojo 基础课程的提议`
> - `学习 Mojo 的资源`

- ****Mojo 中的类型转换 Bug****：一位成员强调了类型转换 Bug，并引用了相关的 GitHub issue [#3065](https://github.com/modularml/mojo/issues/3065) 和 [#3167](https://github.com/modularml/mojo/issues/3167)。
- ****Mojo vs Python 对象讨论****：有推测认为类型转换 Bug 可能与 **Mojo 对象** 和 **Python 对象** 之间的差异有关，引用了 issue [#328](https://github.com/modularml/mojo/issues/328)。
- ****Mojo 基础课程提议****：一位用户提议为 EDx 创建“Mojo Fundamentals”课程，但另一位成员认为这会很快过时。他们建议使用 [Mojo by example](https://ruhati.net/mojo/) 和 [mojo-learning](https://github.com/rd4com/mojo-learning) 作为最新的替代资源。

<div class="linksMentioned"><p><strong>提及的链接</strong>：</p><ul><li><a href="https://github.com/modularml/mojo/issues/328)">Issues · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/3065)">Issues · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/3167)">Issues · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 的开发做出贡献。</li></ul></div>

---

### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1258147458001928233)** (22 条消息🔥):

> - `在 Mojo 中将文件指针转换为结构体`
> - `在 Mojo 中使用 system 或 popen 调用外部程序`
> - `通过字节数组操作处理 Mojo 中的 bitcast 问题`
> - `在 Mojo 中将 List 作为参数传递给函数`
> - `Mojo 中无符号整数转换的 MLIR 问题`

- **在 Mojo 中将文件指针转换为结构体**：一位用户参考另一位用户分享的示例，使用 [bitcast](https://docs.modular.com/mojo/stdlib/memory/unsafe_pointer/UnsafePointer#bitcast) 成功地将 `List` 的 `UnsafePointer` bitcast 为 Mojo 中的结构体。
- **报告 MLIR 无符号整数转换 Bug**：讨论了 **MLIR Issue #3065**，其中转换为无符号整数的行为类似于转换为有符号整数，导致了不一致。该问题影响了多位用户，讨论已从 Discord 转移到 [GitHub Issue #3065](https://github.com/modularml/mojo/issues/3065)。
- **Mojo 中的外部程序**：在 Mojo 中**运行外部程序**可以通过 `external_call` 实现，参考了 [此处示例](https://github.com/modularml/max/blob/main/examples/) 中关于 `system` 和 `popen` 的实现。分享了一个 `popen` 的 Python 示例，详细说明了如何运行。
- **通过字节数组操作处理 Mojo 中的 bitcast 问题**：一位用户在 Mojo 中从文件指针 bitcast 对象时遇到不一致，行为随字节数组查找而变化。怀疑该问题是由于字节被释放导致的，建议保留字节或使用 `Reference` 以避免未定义行为。
- **在 Mojo 中将 List 作为参数传递给函数**：一位用户通过在函数签名中指定类型为 `inout inList:List[String]`，解决了将 `List` 作为参数传递的问题。最初遇到了类型错误，但在修复后成功向列表追加了项。

<div class="linksMentioned"><p><strong>提及的链接</strong>：</p><ul><li><a href="https://docs.modular.com/mojo/stdlib/memory/unsafe_pointer/UnsafePointer#bitcast).">UnsafePointer | Modular Docs</a>：这是一种指针类型，可以指向任何可移动的泛型值。</li><li><a href="https://github.com/modularml/mojo/issues/3065">[BUG] Unsigned integer casting overflowing as if signed when using `int()` or `UInt32()` · Issue #3065 · modularml/mojo</a>：Bug 描述：在 Discord 进行了一些讨论后迁移至此。似乎转换为无符号整数实际上只是转换为有符号整数，但在不同情况下有不同的行为...</li><li><a href="https://github.com/modularml/max/blob/main/examples/">max/examples at main · modularml/max</a>：一系列示例程序、notebook 和工具，展示了 MAX 平台的强大功能 - modularml/max</li></ul></div>

---

### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1258188994445967470)** (10 messages🔥):

> - `nightly build 中的段错误 (segfault) 问题`
> - `提交 Bug 报告`
> - `os.path.expanduser Bug`
> - `新的 nightly Mojo 编译器发布`
> - `更新日志 (changelog) 更新`

- ****Nightly Build 在编译时出现段错误 (Segfaults)****：一位成员在用 nightly build 编译源文件时遇到了段错误，并分享了[有问题的代码文件](https://github.com/Mojo-Numerics-and-Algorithms-group/MojoSci/blob/dynobs/src/diffeq/runga_kutta.mojo)。这促使他们提交了 Bug 报告。
- ****os.path.expanduser Bug 导致 Nightly Build 失败****：由于在测试期间未设置 `HOME` 环境变量，使用 `os.path.expanduser` 引入的一个 Bug 导致 nightly builds 失败。一位成员承认了这一错误，并对造成的不便表示歉意。
- ****新的 Nightly Mojo 编译器发布****：新的 Mojo 编译器版本 `2024.7.416` 已发布，其特性包括为指针类型添加 `exclusive` 参数以及实现 `collections.Counter`。详细变更请参阅 [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 和 [raw diff](https://github.com/modularml/mojo/compare/5b77a66cb42143ffbcf39db635964ae344e63d25...654d07945a5aff2c92e0877153ea5d4b4563dcb6)。
- ****随后的 Nightly Mojo 编译器发布****：另一个 nightly 编译器版本 `2024.7.505` 发布，弃用了 `time.now`，转而推荐使用 `time.perf_counter` 方法。详细变更可在 [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 和 [raw diff](https://github.com/modularml/mojo/compare/654d07945a5aff2c92e0877153ea5d4b4563dcb6...39d95f073592c59b5badeb9740600674540e1235) 中查看。

---

### **Modular (Mojo 🔥) ▷ #[mojo-marathons](https://discord.com/channels/1087530497313357884/1255303604894437388/1258318749841494069)** (17 messages🔥):

> - `Modular 员工对最佳答案的反馈`
> - `对 x86 和 SVE 轮次的兴趣`
> - `需要 MLIR 知识的优化计时器 PR`
> - `Benny 的矩阵乘法解决方案`
> - `测试套件中的编译时间和段错误 (segfaults)`

- **Modular 员工将对最佳答案提供反馈**：Modular 员工将在挑战结束时对最佳答案提供反馈，并提出改进建议。
- **对 x86 和 SVE 基准测试的兴趣**：由于 Graviton 4 预计很快将进入 GA 阶段且支持 SVE，引发了关于进行 x86（带和不带 AMX）以及 SVE 轮次的讨论。
- **Benny 分享矩阵乘法解决方案和提示**：Benny 分享了他最好的矩阵乘法解决方案，并提示通过调整块大小 (block size) 来提高性能。他提到使用 **CPU cache sizes** 作为参数，并建议查阅 UT Austin 的论文以获取更多细节。
- **测试套件中的编译时间和段错误问题**：用户报告称，在使用提供的解决方案运行最新的测试套件时，出现了编译时间过长和内部段错误 (segfault) 问题。
- **参数调优的相关论文**：Benny 引用了几篇 UT Austin 的论文，涉及与 **cache sizes** 相关的参数调优以及矩阵乘法性能提升。他提供了一个 [Google Spreadsheet 链接](https://docs.google.com/spreadsheets/d/1TBz9Lp0JT1Ph7ndfbWqp-B30FQcRYl1959hP2lZ6yH4/edit) 列出了这些论文。

**提及的链接**：[Matrix Multiplication](https://docs.google.com/spreadsheets/d/1TBz9Lp0JT1Ph7ndfbWqp-B30FQcRYl1959hP2lZ6yH4/edit): Sheet1 约束、参数 / 调优向量化、连续访问、Nelts、可展开并行化、可展开循环展开、连续操作分块正方形优化、摊销增加、递归...

---

### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1258529329957306378)** (12 messages🔥):

> - `Usage of LangSmith without LangChain`（在不使用 LangChain 的情况下使用 LangSmith）
> - `Accusation of lack of GPU credits during AI course`（关于 AI 课程期间缺少 GPU 额度的指责）
> - `3rd place solution in AI Mathematical Olympiad`（AI 数学奥林匹克竞赛第三名方案）
> - `Benefits of in-context learning vs. fine-tuning`（In-context learning 与 Fine-tuning 的优势对比）

- ****LangSmith 独立于 LangChain 运行****：一位用户询问 **LangSmith** 是否可以在不使用 **LangChain** 的情况下使用，其他人确认这是可行的，并提供了 [Colab 示例](https://colab.research.google.com/github/langchain-ai/langsmith-cookbook/blob/main/tracing-examples/traceable/tracing_without_langchain.ipynb)和 [GitHub 链接](https://github.com/langchain-ai/langsmith-cookbook/blob/main/tracing-examples/traceable/tracing_without_langchain.ipynb)。**LangSmith** 允许对任何 **LLM** 应用进行**插桩（instrumentation）**，对调试和监控非常有用。
- ****关于 GPU 额度缺失的指责****：针对有说法称课程参与者未收到 **GPU 额度**，引发了激烈辩论。多名成员指出，相关条款在课程平台上已清晰标明且可见。一些人推测这些投诉可能毫无根据，或者是出于其他动机。
- ****AI 数学奥林匹克竞赛前三名方案未进行 Fine-tuning****：一位用户强调，在 AI 数学奥林匹克竞赛中获得 **3.2 万美元**奖金的**第三名方案**没有涉及任何 Fine-tuning。可以在[此处](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/leaderboard)查看排行榜详情。
- ****In-Context Learning 与 Fine-Tuning 的讨论****：一篇比较 **LLM** 的 **In-context learning** 与 **Fine-tuning** 的 **LinkedIn 帖子**引发了有趣的讨论。详细见解可以在[此处](https://www.linkedin.com/posts/zainhas_should-you-finetune-your-llm-or-is-giving-activity-7215029375476383744-ZZ0K)找到。

<div class="linksMentioned"><p><strong>提及的链接</strong>：</p><ul><li><a href="https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/leaderboard">AI Mathematical Olympiad - Progress Prize 1 | Kaggle</a>：未找到描述</li><li><a href="https://docs.smith.langchain.com/old/cookbook/tracing-examples/traceable">Tracing without LangChain | 🦜️🛠️ LangSmith</a>：在 Colab 中打开，在 GitHub 中打开</li></ul></div>

---

### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1258498357983186997)** (7 messages):

> - `Discussion on monthly credits and expiration`（关于每月额度及过期的讨论）
> - `Distributed finetuning issue solutions`（分布式 Fine-tuning 问题的解决方案）
> - `Clarifying the usage and remaining balance of credits`（澄清额度使用情况及余额）

- ****澄清每月额度及过期****：成员们讨论了 **1000 美元的每月额度**及潜在漏洞，澄清了**未使用的额度**可能无法结转，但仍认为这项福利很慷慨。
- ****分布式 Fine-tuning 的问题****：一位成员分享了一个[帖子链接](https://discord.com/channels/1238365980128706560/1247226177257734247)，详细说明了解决**分布式 Fine-tuning** 过程中遇到问题的步骤。
- ****了解额度使用和余额****：讨论集中在成员们注意到的余额上，有人在对 Mistral 进行 Fine-tuning 后报告余额为 **1030 美元**，并询问这是否源于每月 **30 美元**的默认配额。

---

### **LLM Finetuning (Hamel + Dan) ▷ #[jarvis-labs](https://discord.com/channels/1238365980128706560/1241117895740625099/)** (1 messages):

goktrenks: 额度的过期日期是什么时候？（顺便谢谢了！）

---

### **LLM Finetuning (Hamel + Dan) ▷ #[ankurgoyal_textsql_llmevals](https://discord.com/channels/1238365980128706560/1242222674835538012/1258791747300102208)** (2 messages):

> - `Text2SQL use case discussion and appreciation for iterative eval dataset building`（Text2SQL 用例讨论以及对迭代构建评估数据集的赞赏）

- ****迭代构建评估数据集令人印象深刻****：一位成员对关于 **Text2SQL** 的会议表示赞赏，强调了其在**迭代构建评估数据集（eval dataset）**方面的价值。
  - 这种迭代过程特别受到赞赏，被认为对即将到来的用例非常有益。
- ****感谢社区****：成员们向社区表达了谢意，特别是感谢某位个人在**构建 Text2SQL 评估数据集**方面的指导。
  - 成员们认为此类会议和讨论**极具价值**。

---

### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-2](https://discord.com/channels/1238365980128706560/1242223415293644930/1258434662552965223)** (1 messages):

> - `Applying eval framework to unstructured applications`
> - `Challenges of using unit tests/Level 1 evals without structured output`

- **非结构化输出在 Eval Framework 中的挑战**：一位用户询问了 **eval framework** 是否适用于缺乏严格语法规则（**例如查询语言**）的输出。他们对在没有结构化输出的情况下如何实施 **unit tests/Level 1 evals** 表示困惑。
- **非结构化 Eval 应用中缺失的方法论**：该用户询问在考虑如何将 **eval framework** 应用于结构化程度较低的应用时，是否*遗漏了某些环节*，这表明在理解或实践上存在差距。

---

### **LLM Finetuning (Hamel + Dan) ▷ #[jeremy_python_llms](https://discord.com/channels/1238365980128706560/1242224309548875917/1258743017863380992)** (2 messages):

> - `Pushing models to HF_HUB for inference endpoints`
> - `Training models on HF_HUB as endpoints`

- **将模型推送到 HF_HUB 进行推理**：通过将模型推送到 Hub 并使用 **HF_HUB** 的 **credits** 来创建 **inference endpoints**。该建议围绕利用现有资源创建高效的推理流水线展开。
- **训练作为 Endpoints 不可行**：关于**训练可以在 HF_HUB 上作为 endpoint 运行**的想法存疑。讨论认为**训练对于 endpoints 来说可能并不实际**，原因可能是资源或基础设施的限制。

---

### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1258469044718272513)** (3 messages):

> - `Using type: input_output with Meta-Llama-3-8B-Instruct`
> - `Special tokens configuration in Axolotl`
> - `Training outcomes with L3 8B base vs L3 70B Instruct`
> - `Template usage for prompt formatting`
> - `Special tokens setup discrepancies between models`

- **挣扎于 Meta-Llama-3-8B-Instruct 的设置**：一位用户分享了在使用 `type: input_output` 以及为 `Meta-Llama-3-8B-Instruct` 模型配置 `special_tokens` 时遇到的挑战，并对 jsonl 和 yml 文件中的正确设置感到困惑。他们参考了一个 [GitHub 示例](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-3/instruct-lora-8b.yml) 和一篇 [博客文章](https://hamel.dev/notes/llm/finetuning/09_template_free.html) 以获取更多上下文。
- **Special tokens 设置的差异**：讨论涉及需要从 Meta 的 [special_tokens_map.json](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/special_tokens_map.json) 中添加 special tokens，并将其与 Mistral 7B base 的 special tokens 设置进行了比较。他们建议遵循与其他训练设置类似的配置以避免问题。
- **训练结果倾向于 L3 70B Instruct**：一位用户注意到，与 L3 8B base 相比，在 L3 70B Instruct base 上训练的主观效果更好，这是在训练后检查模型配置时才发现的。他们提到，当训练设置默认使用 70B instruct 模型时，得到了一个意外但更好的结果。

**提到的链接**：[Hamel’s Blog - Template-free axolotl](https://hamel.dev/notes/llm/finetuning/09_template_free.html)：在 Axolotl 中使用新的 input_output 格式进行无模板的 prompt 构建。

---

### **LLM Finetuning (Hamel + Dan) ▷ #[credits-questions](https://discord.com/channels/1238365980128706560/1243721538432270388/1258798631255805982)** (1 messages):

> - `Eligibility for credits on all services`
> - `Enrollment date and course catch-up`

- **寻求 Credits 获取资格**：一位成员询问了他们在所有服务上获得 **credits 的资格**，并对任何适用的 credits 表示感谢。
- **课程报名日期**：该成员提到他们是在 **6 月 14 日** 报名参加课程的，最近一直在赶进度。

---

### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1258342772276989964)** (1 messages):

> - `Expired compute credits`
> - `Extension request for compute credits`

- **Compute Credits 过期太快**：一位成员意识到他们的 **compute credits** 在仅一个月后就过期了，导致还有大约 **$70** 未使用。
- **Compute Credits 延期请求**：该成员礼貌地询问是否可以为剩余的 **compute credits** 申请延期。

---

### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/1258803520342069429)** (1 条消息):

> - `额度授予请求`
> - `报名详情`

- ****额度授予请求****：一位用户为其更新的表单申请了额度授予，组织 ID 为 **org-SxGZTlTAAYP5xAswIojG7KI5**。
- ****报名详情****：该用户提到他们在 **6 月 14 日** 报名，最近正在补习课程进度。

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1258171932621148260)** (5 条消息):

> - `对 AI 演示反应平平`
> - `Stability AI 的道歉和许可证更新`

- ****AI 演示批评引发真实性质疑****：@benhylak 在 [X](https://x.com/benhylak/status/1808611023123067357) 上对一个 AI 演示表示失望，质疑其真实性并称 *“这真的非常糟糕……让我怀疑这个演示是不是假的？”*。**响应时间**问题被特别指出。
- ****Stability AI 道歉并更新许可证****：Stability AI 承认 **Stable Diffusion 3 Medium** 未能达到社区预期，并澄清了其商业许可证的 [更新](https://x.com/stabilityai/status/1809274908641489160?s=46)，旨在解决困惑和疑虑。他们承诺今后将发布 **高质量的 Generative AI** 模型。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://x.com/stabilityai/status/1809274908641489160?s=46">来自 Stability AI (@StabilityAI) 的推文</a>：在 Stability AI，我们致力于发布高质量的 Generative AI 模型，并慷慨地与我们的创新者和媒体创作者社区分享。我们承认我们最新的发布...</li><li><a href="https://x.com/benhylak/status/1808611023123067357">来自 ben (@benhylak) 的推文</a>：刚试了一下……真的非常糟糕。让我怀疑演示是不是假的？引用 ben (@benhylak)：世界即将发生飞速变化。</li></ul></div>

---

### **Interconnects (Nathan Lambert) ▷ #[other-papers](https://discord.com/channels/1179127597926469703/1179142630517518397/1258334337158610966)** (8 条消息🔥):

> - `搜索引擎中的 BM42 与 BM25`
> - `Contextual AI 对 RAG 的关注`
> - `Jo Bergum 对 Qdrant 的 BM42 声明的批评`

- **BM42 在搜索技术中挑战 BM25**：**Qdrant Engine** 声称新的搜索模型 **BM42** 在现代 RAG 应用中超越了传统的 **BM25**，如其 [推文](https://x.com/qdrant_engine/status/1808498752107241949) 中所述，它结合了语义搜索和关键词搜索。
- **Jo Bergum - BM42 结果造假**：**Jo Bergum** 批评 **Qdrant Engine** 伪造了关于 BM42 在 Quora 数据集上的结果，指出报告的 Precision@10 高得离谱，并称这些结果为 **

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://x.com/qdrant_engine/status/1808498752107241949">来自 Qdrant (@qdrant_engine) 的推文</a>：40 年来，BM25 一直是搜索引擎的标准。然而，它在现代 RAG 应用中表现不足。向 BM42 问好：语义搜索与关键词搜索的结合</li><li><a href="https://x.com/jobergum/status/1809157587612336402?s=46">来自 Jo Kristian Bergum (@jobergum) 的推文</a>：好吧，摊牌了。@qdrant_engine 在 BM42 帖子中的做法是不可接受的。他们正在严重误导 RAG 社区。1) 将 Quora 作为一个相关的 RAG 问答数据集呈现。我...</li></ul></div>

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1258300980542439425)** (9 messages🔥):

> - `理解 VAEs`
> - `Interconnects 的投资天赋`
> - `AI 驱动的 GDP 增长率时间线`
> - `Anthropic Claude 3.5 Sonnet 隐藏回答`

- **理解 VAEs 导致流鼻血**：*VAEs* (Variational Autoencoders) 引起了困惑，一位用户幽默地提到，在试图理解它们时流了鼻血。
- **Interconnects 的投资天才**：在最近的一篇帖子中，透露了 **interconnects** 展示了他作为“绝对投资天才”的实力。
- **AI 驱动的 GDP 增长需要显著的增长率**：AI 带来的 **GDP 增长** 需要在 **11-15%** 之间才能符合 Stuart 的时间线，具体取决于初始条件。该指标的可行性已通过检查，被认为是合理的。
- **Anthropic Claude 3.5 Sonnet 隐藏回答**：据报道，[Anthropic Claude 3.5 Sonnet](https://fxtwitter.com/_philschmid/status/1808755146190446667) 正在向用户隐藏其部分回答。使用诸如 *§§antThinking§§* 之类的隐藏标签引发了人们对这些 AI 系统**透明度 (transparency)** 的担忧。

**提到的链接**：[来自 Philipp Schmid (@_philschmid) 的推文](https://fxtwitter.com/_philschmid/status/1808755146190446667)：我之前没意识到这一点，但看起来 (claude ai) 上的 Anthropic Claude 3.5 Sonnet 正在向用户隐藏其部分回答，这些内容没有发送给客户端。你可以通过以下方式测试...

---

### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1258407157356040192)** (4 messages):

> - `Gemini web app`
> - `Google AI Studio`
> - `Vertex AI`
> - `Google 的 AI 竞赛`
> - `第一修正案与权重 (weights)`

- **Google 的 AI 竞赛处于落后地位**：根据聊天中的详细讨论，**Google** 在 AI 竞赛中落后于其他公司，需要解决导致用户困惑的清晰度问题。
  - 一位参与者表示，Google **在 AI 竞赛的启动阶段表现缓慢且混乱**，但也承认他们正在改进。
- **了解 Gemini 及其产品**：**Gemini web app** 每月收费 **$20**，与 ChatGPT 竞争，其前身为 Bard，最初由 **PaLM 2** 驱动，现在使用 **Gemini 1.5**。**Google AI Studio** 为开发者提供 API key，以便使用具有 2M 上下文的 Gemini 1.5，而 **Vertex AI** 则为企业提供相同服务。
  - *一位用户表达了困惑*，因为 FAQ 说明不清晰，不确定付费版 Gemini 是否始终使用 Gemini 1.5。
- **第一修正案与权重 (weights)**：一位用户讨论了**第一修正案 (First Amendment)** 在 AI 模型权重 (weights) 上的应用，认为这可能是一个逻辑自洽但乐观的观点。
  - *其核心观点是*，权重应作为已发布的内容受到保护，从而涵盖在**第一修正案**之内。

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1258511792901066842)** (20 messages🔥):

> - `build.nvidia API 的问题`
> - `build.nvidia API 的队列系统`
> - `脚本问题与解决`
> - `使用 YAML 示例的 Pipeline`

- ****build.nvidia API 出现故障****：一位成员注意到 **build.nvidia API** 存在问题。另一位成员指出出现了一个用于处理请求的**队列系统 (queue system)**。
  - 在尝试解决脚本问题时，一位成员发现经过短暂暂停后它又可以工作了，这表明该 API 的可靠性是间歇性的。
- ****Pipeline 接受 YAML 输入****：在关于处理输入的讨论中，一位成员提到他们的 **pipeline** 采用对话的 **YAML 示例** 进行 few-shot 学习。在被问及是否包含教科书数据时，他们澄清了这一点。

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1258454095292010547)** (1 messages):

> - `Gemma2 更新修复问题`
> - `固定版本的 transformers`
> - `CI 捕捉问题`

- **Gemma2 在更新中修复问题**：**Gemma2** 的更新解决了之前遇到的问题。由于我们的 CI 系统可以检测到此类问题，使用固定版本的 transformers 可以确保避免这些问题。
- **CI 确保 Transformers 的稳定性**：**固定版本的 transformers** 应该能避开问题，因为持续集成 (CI) 会捕捉到潜在问题。这保证了更稳定的开发环境。

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/)** (1 messages):

le_mess: 需要更多 VRAM 🙂

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1258154032288239779)** (3 条消息):

> - `test for bug placement`
> - `issue reproduction`
> - `focused test case`
> - `PR management`

- ****Bug 测试位置决策****：一位用户询问了 bug 测试的最佳位置——是在 **test_nn** 还是 **test_ops** 中——并征求了命名建议。
  - 该用户确认已理解，并将任务委派给他人，表示他们将处理此事。
- ****问题复现与 PR 管理****：另一位用户建议保持 PR 开启状态，将其视为带有复现步骤的 issue，并确保修复方案包含一个更具针对性的测试用例。
  - *原始用户的最终确认*暗示他们将处理具体细节。

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1258137471414308945)** (12 条消息🔥):

> - `Contiguous Tensors in Tinygrad`
> - `Tinygrad Training Efficiency Concerns`
> - `Matrix Multiplication Blog Post`
> - `Using Pre-trained PyTorch Models with Tinygrad`

- **Tinygrad 连续 Tensor 困扰用户**：讨论提到 `Tensor.randn/randint` 会创建连续的 Tensor，而 `Tensor.full` 及类似方法则创建非连续的 Tensor，这与 PyTorch 的行为相反。
- **优化 Tinygrad 的大规模训练**：成员们讨论了 Tinygrad 在大规模训练中的低效问题，称其速度慢且不具备成本效益。有人建议使用 BEAM 搜索，但这需要时间。
- **通过科普博客学习 Matmul**：分享了一篇关于 CPU 高性能矩阵乘法的精彩[博客文章](https://salykova.github.io/matmul-cpu)，展示了超过 1 TFLOPS 的性能，并附带了易于理解的[代码](https://github.com/salykova/matmul.c)。
- **在 Tinygrad 上运行 PyTorch 模型推理**：询问如何使用 Tinygrad 运行预训练 PyTorch 模型的推理。提供的答案指向了 `tinygrad.nn.state.torch_load` 的用法。

<div class="linksMentioned"><p><strong>提及的链接</strong>：</p><ul><li><a href="https://gist.github.com/python273/0dc136fbc63559188ab279c07329e891">TinyJit vis WIP</a>：TinyJit 可视化进行中。GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://salykova.github.io/matmul-cpu">Beating NumPy’s matrix multiplication in 150 lines of C code</a>：摘要：教程中的代码可在 matmul.c 中找到。这篇博客文章是我尝试在保持代码简单、可移植的同时，在 CPU 上实现高性能矩阵乘法的结果……</li></ul></div>

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1258157010567037008)** (8 条消息🔥):

> - `Setting evaluation parameters for Torchtune`
> - `Grad norm graph on wandb`
> - `Loss curve optimization in wandb`
> - `Learning rate adjustment impacts`
> - `Missing wandb logging metrics`

- ****设置 Torchtune 的评估参数****：一位用户询问如何在 **Torchtune** 中设置评估参数，另一位用户提到应该有一个“验证数据集”或类似的参数。
- ****wandb 中缺失梯度范数图表****：一位用户寻求关于在 wandb 中获取 **梯度范数（grad norm）图表** 的帮助，因为这是 aoxotl 等其他工具中的默认图表。
- ****wandb 中的损失曲线优化****：建议用户观察 **损失曲线** 的形状以确认下降趋势，并提供了一个带有[链接](https://wandb.ai/salman-mohammadi/torchtune_codellama_testing/runs/zobzkhd3?nw=nwusersalmanmohammadi)的示例。他们注意到损失曲线优化不足，并建议增加初始学习率。
- ****学习率调整的影响****：在收到反馈后，一位用户增加了初始 **学习率** 并修改了多个参数以优化模型，但报告称损失没有显著改善。
- ****缺失 wandb 日志指标****：一位用户质疑 **wandb 日志** 中缺少评估损失和梯度范数，表明指标记录存在问题。

**提及的链接**：[salman-mohammadi](https://wandb.ai/salman-mohammadi/torchtune_codellama_testing/runs/zobzkhd3?nw=nwusersalmanmohammadi)：Weights & Biases，机器学习开发者工具。

---

### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1258176915458752653)** (5 messages):

> - `Investigating system robustness with Python and TypeScript`
> - `Challenges with automatic Docker installation of Convex local backend`

- ****Python 和 TypeScript 面临集成问题****：一位成员分享了集成 **Python** 和 **TypeScript** 时遇到的问题，特别是在未预装 Python 的情况下启动 **Convex** 时遇到的 Bug。
- ****Docker 的 Convex 后端安装很棘手****：另一位成员讨论了在 **Docker** 中实现 **Convex** 本地后端自动安装的挑战，这主要是由于为了便于更新和访问，容器文件夹被设置为 Volume（数据卷）。

---

### **AI Stack Devs (Yoko Li) ▷ #[assets](https://discord.com/channels/1122748573000409160/1176906086368935966/1258699780851241060)** (1 messages):

> - `Collection of sprite sheets`
> - `Aesthetics and style matching with Cloudpunk`
> - `Largest tilemaps on itch.io`

- **寻找匹配 Cloudpunk 美学的 sprite sheets**：一位成员询问了特定 **sprite sheets** 集合的来源，并提到他们在 **itch.io** 上购买了几个大型 **tilemaps**，但与 [Cloudpunk](https://store.steampowered.com/app/746850/Cloudpunk/) 那种**黑暗、未来感、赛博朋克**的美学风格不太匹配。
- **匹配已购买 tilemaps 的美学**：该成员很好奇从哪里可以获得与 **Cloudpunk** 美学相契合的 **spritesheets**，因为他们目前在 **itch.io** 收集和购买的资源尚不理想。

---

### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1258487391300358279)** (1 messages):

> - `Three GPTs Walk into a Bar and Write an Exec Summary blog post by dsquared70`
> - `Utilizing Custom GPTs for creating executive summaries`
> - `Processes for high-frequency, short turnaround executive summaries`

- ****三个 GPT 彻底改变执行摘要****：[Three GPTs Walk into a Bar and Write an Exec Summary](https://www.dylandavis.net/2024/07/three-gpts-walk-into-a-bar-and-write-an-exec-summary) 博文介绍了一种快速创建执行摘要的简单流程。**三个 Custom GPTs** 协同工作：一个提取洞察，一个撰写摘要，第三个负责修订内容。
- ****高频执行摘要策略****：博文详细介绍了这些 **Custom GPTs** 如何应对在总结事件、技术或趋势时的高频和快速周转需求。在任务期限通常很紧的情况下，该流程确保了摘要既快速又有意义。

**提到的链接**：[Three GPTs Walk into a Bar and Write an Exec Summary – D-Squared](https://www.dylandavis.net/2024/07/three-gpts-walk-into-a-bar-and-write-an-exec-summary/)：未找到描述

---

### **DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1258458307484712980)** (2 messages):

> - `Magpie model available on HuggingFace Spaces`
> - `Generating preference data via HuggingFace Spaces`
> - `Duplicated model from davanstrien/magpie`
> - `User feedback on Magpie model performance`

- **Magpie 模型已在 HuggingFace Spaces 上线**：一个 Magpie 模型现在可以在 [HuggingFace Spaces](https://huggingface.co/spaces/sroecker/Elster) 上访问，该模型复制自 [davanstrien/magpie](https://huggingface.co/spaces/davanstrien/magpie)。
  - 目前*效果还不是很好*，但通过 HuggingFace Spaces 生成偏好数据的概念很受欢迎。
- **用户对 Magpie 模型性能的反馈**：一位用户分享说 Magpie 模型运行效果不佳，但对其概念表示赞赏。

**提到的链接**：[Magpie - a Hugging Face Space by sroecker](https://huggingface.co/spaces/sroecker/Elster)：未找到描述

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1258246339028979722)** (2 messages):

> - `Claude hackathon collaboration`
> - `Kafka optimization webinar`

- ****Claude Hackathon Collaboration****：一位成员邀请其他人协作，为下周结束的 [Claude hackathon](https://docs.anthropic.com/en/build-with-claude-contest/overview) 构建一些酷炫的东西。
- ****优化 Kafka 并节省成本！****：参加 **7 月 18 日下午 4 点 (IST)** 的网络研讨会，学习 [优化 Kafka](https://www.meetup.com/futureofdata-bangalore/events/301849238/?notificationId=1389017441959817216) 的最佳实践，包括扩展策略和成本节约技术。
- ****Kafka 研讨会专家讲师****：本次活动将邀请来自 Superstream 的 **Yaniv Ben Hemo** 和来自 Cloudera 的 **Viktor Somogyi-Vass**，他们将分享在构建可扩展、高成本效益 Kafka 环境方面的专业经验。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://docs.anthropic.com/en/build-with-claude-contest/overview">未找到标题</a>：未找到描述</li><li><a href="https://www.meetup.com/futureofdata-bangalore/events/301849238/?notificationId=1389017441959817216">Optimizing Kafka for Cost-Efficiency: Best Practices and Strategies, Thu, Jul 18, 2024, 4:00 PM | Meetup</a>：**活动标题：** **Optimizing Kafka for Cost-Efficiency: Best Practices and Strategies** **活动详情：** 日期：2024 年 7 月 18 日 时间：下午 4:00 (IST)（线上活动）欢迎加入我们</li></ul></div>

---

### **Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1258348238851932220)** (1 messages):

> - `Potential uses for embeddings`
> - `New job title 'Embeddings Engineer'`

- ****Embeddings Engineer 发现更多用途****：一位成员表示他们正在发现更多 **Embeddings 的潜在用途**，并开玩笑说要采用 *Embeddings Engineer* 这个头衔。
- ****新职位名称幽默****：由于 Embeddings 的用途日益增多，**Embeddings Engineer** 被幽默地提议作为一个新职位名称。
  - *我想从现在起称呼自己为 Embeddings Engineer* 😄

---

---

---

---

{% else %}

> 完整的频道细分内容已针对电子邮件进行截断。
> 
> 如果您想查看完整细分，请访问此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！
> 
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}