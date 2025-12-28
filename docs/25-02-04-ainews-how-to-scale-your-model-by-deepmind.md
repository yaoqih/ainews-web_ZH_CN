---
companies:
- google-deepmind
- deepseek
- hugging-face
date: '2025-02-05T06:59:23.438232Z'
description: '**Google DeepMind (GDM) 的研究人员**发布了一本名为 **《如何扩展你的模型》(How To Scale Your
  Model)** 的综合性“小教科书”，内容涵盖了现代 Transformer 架构、超越 $O(N^2)$ 注意力机制的推理优化，以及 Roofline 等高性能计算（HPC）概念。该资源还包括实践练习题和实时评论互动。


  在 AI 推特（X）上，几项关键动态备受关注：

  *   受**克里斯蒂亚诺·罗纳尔多 (Cristiano Ronaldo)**、**勒布朗·詹姆斯 (LeBron James)** 和**科比·布莱恩特 (Kobe
  Bryant)** 等运动员启发的开源人形机器人模型 **ASAP** 正式发布；

  *   一篇关于 **Mixture-of-Agents (MoA)** 的新论文提出了 **Self-MoA** 方法，旨在改进大语言模型（LLM）的输出聚合；

  *   展示了利用 **DeepSeek** 的 **GRPO 算法**在 **Qwen 0.5** 上训练推理型 LLM；

  *   关于 LLM 作为评审员（LLM-as-a-judge）存在偏见的研究结果，强调了进行多次独立评估的必要性；

  *   以及 **mlx-rs** 的发布，这是一个用于机器学习的 Rust 库，并提供了 **Mistral** 文本生成等示例。


  此外，**Hugging Face** 推出了一个 AI 应用商店，目前拥有超过 **40 万个应用**，每日新增 2000 个，每周访问量达 250 万次，并支持由
  AI 驱动的应用搜索与分类功能。'
id: c70df8df-638b-4e98-b325-007ce112f82e
models:
- qwen-0.5
original_slug: ainews-how-to-scale-your-model-by-deepmind
people:
- omarsar0
- drjimfan
- tairanhe99
- guanyashi
- lioronai
- _philschmid
- awnihannun
- clementdelangue
title: '以下是几种中文翻译供参考：


  1.  **如何扩展你的模型 —— DeepMind**（最简洁、常用）

  2.  **DeepMind：如何实现模型规模化**（更具专业感）

  3.  **如何进行模型缩放，DeepMind 出品**（侧重于“缩放”这一技术术语）


  在 AI 领域，“Scale” 通常翻译为 **“扩展”** 或 **“规模化”**。'
topics:
- transformers
- inference
- high-performance-computing
- robotics
- sim2real
- mixture-of-experts
- reinforcement-learning
- bias-mitigation
- rust
- text-generation
- open-source
---

<!-- buttondown-editor-mode: plaintext -->**系统思维就是你所需要的一切。**

> 2025年2月3日至2月4日的 AI 新闻。我们为你检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **34** 个 Discord 社区（**225** 个频道，**3842** 条消息）。预计节省阅读时间（以 200wpm 计算）：**425 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

令人惊喜的是，一些研究人员发布了一本关于他们在 GDM 如何扩展模型的“小教科书”：


![image.png](https://assets.buttondown.email/images/42cd0862-439a-4053-b038-aee3f6fdc947.png?w=960&fit=max)


一位评论者[确认](https://news.ycombinator.com/item?id=42938185)这是 GDM 的内部文档，其中删减了关于 Gemini 的引用。

《[如何扩展你的模型](https://jax-ml.github.io/scaling-book/)》共分为 12 个部分，开头对当今标准 Transformer 的形态进行了精彩的更新：


![image.png](https://assets.buttondown.email/images/84641eb8-3eaa-42cc-91c6-925ece7928de.png?w=960&fit=max)


并[解释了推理过程与标准的 O(N^2) Attention 理解有何不同](https://jax-ml.github.io/scaling-book/inference/)：


![image.png](https://assets.buttondown.email/images/e1107b18-2dce-4878-8cb1-06760288a65a.png?w=960&fit=max)


同时也引入了标准的高性能计算概念，如 [rooflines](https://jax-ml.github.io/scaling-book/roofline/)：


![image.png](https://assets.buttondown.email/images/8848c83c-eb3a-4e6f-afca-f5a22a8beb38.png?w=960&fit=max)


甚至还为积极的读者准备了练习题来测试他们的理解…… 评论正在被实时阅读。


![image.png](https://assets.buttondown.email/images/51215b48-6bed-45b7-aad9-291780fb5910.png?w=960&fit=max)


---

{% if medium == 'web' %}

**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 回顾

**AI 模型发布与研究论文**

- **"ASAP"：人形机器人的 Real2Sim2Real 模型**：[@DrJimFan](https://twitter.com/DrJimFan/status/1886824152272920642) 宣布了 "**ASAP**"，这是一个能让人形机器人执行受 **Cristiano Ronaldo**、**LeBron James** 和 **Kobe Bryant** 启发的流畅动作的模型。包括 [@TairanHe99](https://twitter.com/DrJimFan/status/1886824977191854327) 和 [@GuanyaShi](https://twitter.com/DrJimFan/status/1886824977191854327) 在内的团队已经**开源**了该项目的**论文和代码**。该方法结合了现实世界数据与模拟，以克服机器人技术中的 "**sim2real**" 差距。

- **"Rethinking Mixture-of-Agents" 论文与 "Self-MoA" 方法**：[@omarsar0](https://twitter.com/omarsar0/status/1886792384954163347) 讨论了一篇名为 "**Rethinking Mixture-of-Agents**" 的新论文，该论文质疑了混合不同 **LLMs** 的益处。提出的 "**Self-MoA**" 方法通过聚合表现最好的 LLM 的输出来利用模型内部的多样性，表现优于传统的 MoA 方法。**论文**可以在[这里](https://twitter.com/omarsar0/status/1886792397235085383)找到。

- **使用 DeepSeek 的 GRPO 算法训练 LLM**：[@LiorOnAI](https://twitter.com/LiorOnAI/status/1886850811378196685) 重点介绍了一个新的 **notebook**，演示了如何使用 **DeepSeek** 的 **GRPO 算法**训练推理 LLM。在不到 2 小时的时间内，你可以将像 **Qwen 0.5**（*5 亿参数*）这样的小模型转变为**数学推理机器**。[Notebook 链接](https://twitter.com/LiorOnAI/status/1886850813911556351)。

- **作为裁判的 LLM 中的偏见**：[@_philschmid](https://twitter.com/_philschmid/status/1886717030218297406) 分享了论文 "**Preference Leakage: A Contamination Problem in LLM-as-a-Judge**" 的见解，揭示了 LLM 在用于合成数据生成和评估时可能存在**显著偏见**。研究强调需要**多个独立的裁判**和**人工评估**来减轻偏见。[论文](https://twitter.com/_philschmid/status/1886717032378372164)。

- **mlx-rs：用于机器学习的 Rust 库**：[@awnihannun](https://twitter.com/awnihannun/status/1886846423905575330) 介绍了 **mlx-rs**，这是一个 Rust 库，包含使用 **Mistral 进行文本生成**和 **MNIST 训练**的示例。对于那些对 **Rust** 和**机器学习**感兴趣的人来说，这是一个宝贵的资源。[点击查看](https://twitter.com/awnihannun/status/1886846423905575330)。

**AI 工具与平台公告**

- **Hugging Face 的 AI 应用商店上线**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1886861567326650526) 宣布 **Hugging Face** 推出了其 **AI 应用商店**，目前拥有 **400,000 个应用**，每天新增 **2,000 个应用**，每周访问量达 **250 万次**。用户现在可以使用 AI 或按类别**搜索**应用，并强调“**AI 的未来将是分布式的**”。[探索应用商店](https://twitter.com/ClementDelangue/status/1886861567326650526)。

- **AI 应用商店发布公告**：[@_akhaliq](https://twitter.com/_akhaliq/status/1886831521216016825) 同样对 **AI 应用商店** 的发布表示兴奋，称其为寻找所需 **AI 应用** 的最佳场所，现有约 **40 万个应用** 可供使用。开发者可以构建应用，用户可以通过 **AI 搜索** 发现新应用。[点击查看](https://twitter.com/_akhaliq/status/1886831521216016825)。

- **WhatsApp 上的 1-800-CHATGPT 更新**：[@kevinweil](https://twitter.com/kevinweil/status/1886988476203126878) 宣布了 **WhatsApp** 上 **1-800-CHATGPT** 的新功能：
  - 现在可以在提问时**上传图片**。
  - 使用**语音消息**与 ChatGPT 交流。
  - 很快，你将能够**关联你的 ChatGPT 账号**（Free, Plus, Pro）以获得更高的速率限制（rate limits）。
  - [了解更多](https://twitter.com/kevinweil/status/1886988479499776348)。

- **Replit 的新移动应用和 AI Agent**：[@hwchase17](https://twitter.com/hwchase17/status/1886950917326655740) 分享了 **Replit** 推出的新**移动应用**，并提供了 **AI Agent** 的免费试用。Replit AI Agent 的快速发展备受关注，[@amasad](https://twitter.com/amasad) 确认了此次发布。[详情点击](https://twitter.com/hwchase17/status/1886950917326655740)。

- **ChatGPT Edu 在加州州立大学推广**：[@gdb](https://twitter.com/gdb/status/1886884951666340270) 报道称，**加州州立大学**正成为首个 **AI 驱动的大学系统**，**ChatGPT Edu** 已向 **460,000 名学生**以及超过 **63,000 名教职员工**推广。[阅读更多](https://twitter.com/gdb/status/1886884951666340270)。

**AI 活动、会议与招聘**

- **AI Dev 25 会议宣布**：[@AndrewYNg](https://twitter.com/AndrewYNg/status/1886833904235241753) 宣布了 **AI Dev 25**，这是一场将于 **2025 年 3 月 14 日（派日）**在**旧金山**举行的 AI 开发者会议。该活动旨在为 AI 开发者创建一个**厂商中立的会议**，届时将有 **400 多名开发者**聚集在一起进行构建、分享想法和建立联系。[了解更多并注册](https://twitter.com/AndrewYNg/status/1886833904235241753)。

- **Anthropic 对齐科学团队招聘**：[@sleepinyourhat](https://twitter.com/sleepinyourhat/status/1886822563353141303) 正在为 **Anthropic** 的 **Alignment Science 团队**招聘研究员，该团队由她与 [@janleike](https://twitter.com/sleepinyourhat/status/1886822564905070823) 共同领导。他们专注于 **AGI 安全**方面的**探索性技术研究**。理想的候选人需具备：
  - 多年 **SWE** 或 **RE** 经验。
  - 丰富的**研究经验**。
  - 熟悉**现代 ML** 和 **AGI 对齐文献**。
  - [在此申请](https://twitter.com/sleepinyourhat/status/1886822568067498242)。

- **Andrew Ng 将出席 INTERRUPT 会议**：[@hwchase17](https://twitter.com/hwchase17/status/1886823545122250928) 宣布 **Andrew Ng** 将于今年 5 月在 **INTERRUPT 会议**上发表演讲。Ng 被誉为我们这一代最优秀的教育家之一，鼓励与会者向他学习。[获取门票](https://twitter.com/hwchase17/status/1886823545122250928)。

- **DeepSeek 集成虚拟论坛**：[@llama_index](https://twitter.com/llama_index/status/1886912036766204127) 邀请开发者、工程师和 AI 爱好者参加**虚拟论坛**，探索 **DeepSeek** 及其功能和工作流集成。演讲者包括来自 **Google**、**GitHub**、**AWS**、**Vectara** 和 **LlamaIndex** 的代表。[在此注册](https://twitter.com/llama_index/status/1886912036766204127)。

**AI 伦理、安全与政策**

- **Google DeepMind 更新前沿安全框架**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1886817852595876104) 分享了其 **Frontier Safety Framework** 的更新，这是一套旨在随着我们向 **AGI** 迈进而减轻**严重风险**的协议。他们强调 AI 需要兼具**创新性与安全性**，并邀请读者[了解更多](https://twitter.com/GoogleDeepMind/status/1886817852595876104)。

- **关于 LLM 裁判偏差的讨论**：[@_philschmid](https://twitter.com/_philschmid/status/1886717030218297406) 探讨了 LLM 在用于 **synthetic data generation** 和作为裁判时的 **bias in LLMs** 问题。“**Preference Leakage**”论文揭示了 LLM 可能会偏好由其自身或其先前版本生成的数据，突显了 **contamination problem**。[阅读论文](https://twitter.com/_philschmid/status/1886717032378372164)。

- **OpenAI 的 Frontier Safety Framework 更新**：[@OpenAI](https://twitter.com/OpenAI/status/1886966048970478016) 宣布了其 **Frontier Safety Framework** 的最新更新，旨在防范与先进 AI 系统相关的潜在严重风险。

**通用 AI 行业评论**

- **Yann LeCun 论小团队与创新**：[@ylecun](https://twitter.com/ylecun/status/1886677032324509898) 强调，具有自主权的 **small research teams** 能够做出正确的技术选择并进行创新。他强调了组织和管理在促进 **R&D organizations** 内部创新方面的重要性。

- **DeepSeek 被比作斯普特尼克时刻**：[@JonathanRoss321](https://twitter.com/JonathanRoss321/status/1886784302513377379) 将关于 **DeepSeek** 的新闻比作现代版的“**Sputnik 2.0**”，暗示这是 AI 领域的一个重要里程碑，类似于历史上太空竞赛的重大事件。

- **对技术采用的反思**：[@DavidSHolz](https://twitter.com/DavidSHolz/status/1886920976270774623) 评论了新技术最初通常如何被用来复制旧媒介，并指出：“**当你没有意识到新发明也是新媒介时，就会犯这些错误**。”

- **关于 AI 评估和 RL 的讨论**：[@cwolferesearch](https://twitter.com/cwolferesearch/status/1886810699726275041) 观察到 **few-shot prompting** 会降低 **DeepSeek-R1** 的性能，这可能是由于该模型是在严格的格式上训练的。这指向了与 LLM 交互的新范式以及 AI 技术不断演进的格局。


---

# AI Reddit 回顾

## /r/LocalLlama 摘要

**主题 1. DeepSeek R1 & R1-Zero：快速模型训练成果**

- **[DeepSeek 研究员称训练 R1 和 R1-Zero 仅需 2-3 周](https://www.reddit.com/gallery/1ihd0rr)** ([得分: 800, 评论: 127](https://reddit.com/r/LocalLLaMA/comments/1ihd0rr/deepseek_researcher_says_it_only_took_23_weeks_to/)): **DeepSeek** 的研究员声称 **R1** 和 **R1-Zero** 模型的训练仅耗时 **2-3 周**，这表明这些 AI 模型的开发周期极快。
  - 讨论中，部分用户对 **R1 和 R1-Zero 在 3 周内完成 10,000 步 RL 训练的快速过程** 持怀疑态度，质疑其可行性；另一些人则认为，通过对 **V3** 等现有模型进行微调可以解释这种速度。担忧的问题包括由于高需求导致的 API 和网站性能瓶颈，以及对改进训练数据或架构的需求。
  - 用户将 **DeepSeek 的模型** 与其他 AI 进展进行了比较，指出在该论文发布后，全球范围内可能会涌现出新模型。一些人表示相比 **OpenAI** 更倾向于 **DeepSeek**，原因是其开放性以及没有关税限制，而另一些人则在期待 R1.5 或 V3-lite 等未来版本。
  - 对话涉及了 **AI 竞赛**，并将其与全球太空竞赛相类比，强调了各地区参与度的差异。欧洲被提及通过 **Stable Diffusion** 和 **Hugging Face** 等公司做出了贡献，而其他地区则被指出参与度有限，凸显了全球 AI 开发的竞争本质。


**主题 2. DeepSeek-R1 模型：更短的正确答案及其影响**

- **[DeepSeek-R1 的正确答案通常更短](https://i.redd.it/duiwqfpzq3he1.png)** ([得分: 289, 评论: 66](https://reddit.com/r/LocalLLaMA/comments/1ihf0gb/deepseekr1s_correct_answers_are_generally_shorter/)): 如柱状图所示，**DeepSeek-R1** 的正确答案通常更短，平均为 **7,864.1 个 tokens**，而错误答案平均为 **18,755.4 个 tokens**。正确答案的 token 长度标准差为 **5,814.6**，错误答案为 **6,142.7**，表明 token 长度存在波动性。
  - **任务难度与回复长度**：包括 **wellomello** 和 **Affectionate-Cap-600** 在内的多条评论质疑该分析是否考虑了任务难度，认为更难的任务自然需要更长的回复，这可能会影响错误率和平均 token 长度。
  - **模型行为与标准差**：**FullstackSensei** 和 **101m4n** 讨论了 token 长度高标准差的影响，认为错误答案可能是由于模型进入死循环或在解题中挣扎，从而延长了回复时间。
  - **相关研究与泛化**：**Angel-Karlsson** 引用了一篇关于模型“过度思考”的相关研究论文，而 **Egoz3ntrum** 则强调了考虑数据集局限性的重要性，指出结论可能无法很好地泛化到特定数学难题之外。


**主题 3. OpenAI 研究：通过 Hugging Face 拥抱开源**

- **OpenAI Deep Research 的开源实现** ([得分: 421, 评论: 28](https://reddit.com/r/LocalLLaMA/comments/1ihqwnd/openai_deep_research_but_its_open_source/)): **Hugging Face** 启动了一项名为 **OpenAI Deep Research** 的倡议，将深度研究开源化。该项目旨在使尖端 AI 研究的获取更加民主化，强调 AI 社区的透明度与协作。更多细节可以在他们的 [博客文章](https://huggingface.co/blog/open-deep-research) 中找到。
  - 用户对 **Hugging Face** 团队表示了极大的赞赏，将其贡献与 **Mistral** 团队相提并论，并强调了极快的开发节奏，一些人期待很快能集成到 **Open-WebUI** 等平台中。关于迅速创建出替代 **OpenAI** 专有解决方案的开源方案，评论中充满了紧迫感和惊喜。
  - 讨论突显了开源社区对 **Hugging Face** 提供广泛工具和框架的感激之情，一些用户幽默地质疑这种慷慨背后的动机。快速开发开源替代方案是一个反复出现的主题，反映了社区在保持 AI 进展开放获取方面的积极立场。
  - 一条评论提供了一个 GitHub 仓库链接，供有兴趣尝试本地实现的用户参考，指向 **Automated-AI-Web-Researcher-Ollama**，作为实验开源 AI 工具的资源。这表明社区对动手实验 AI 研究工具有着实际的兴趣。


## 其他 AI 子版块摘要

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. OmniHuman-1：中国的多模态奇迹**

- **[中国的 OmniHuman-1 🌋🔆](https://v.redd.it/pwjfvxljy4he1)** ([评分: 684, 评论: 174](https://reddit.com/r/OpenAI/comments/1ihjgpk/chinas_omnihuman1/)): **OmniHuman-1** 是一个专注于**从单张图像生成视频**的中国项目。该帖子缺少更多细节，因此未提供关于 OmniHuman-1 的进一步背景或技术细节。
  - **OmniHuman-1 的能力与担忧**：关于 OmniHuman-1 从单张图像和音频生成逼真人物视频的潜力存在大量讨论，一些用户对媒体真实性的影响以及可能出现无法区分的合成媒体表示担忧。该项目的详细信息和代码库可在 [GitHub](https://github.com/mdsrqbl/omnihuman) 上找到，其白皮书可在 [omnihuman-lab.github.io](https://omnihuman-lab.github.io/) 查阅。
  - **AI 对创意产业的影响**：一些评论者辩论了 AI 对创意产业的影响，认为 AI 可能会带来一个独特且经济可行的艺术创作黄金时代，而另一些人则对 AI 复制人类经验深度的能力表示怀疑。此外，还讨论了对人类创作工作的未来以及经济影响（如对 UBI 的潜在需求）的担忧。
  - **技术观察与挑战**：用户注意到 AI 生成视频中的技术缺陷，如不自然的肢体动作和恐怖谷效应，这表明虽然 AI 已经取得了进步，但仍存在持久的挑战，可能需要根本不同的技术来克服。讨论中还提到，基于 AI 的视频将通过不断完善的过程持续进化，类似于语言模型的发展。


**主题 2. 华为 Ascend 910C 挑战 Nvidia H100**

- **华为 Ascend 910C 芯片性能媲美 Nvidia H100。到 12 月将生产 140 万颗。不要认为受限国家和开源无法率先实现 AGI。** ([评分: 262, 评论: 99](https://reddit.com/r/OpenAI/comments/1ihebb4/huaweis_ascend_910c_chip_matches_nvidias_h100/)): 据报道，**华为 Ascend 910C 芯片**在性能上与 **Nvidia H100** 相当，并计划到 2025 年生产 **140 万**颗。这一进展挑战了关于中国和开源项目在 AI 芯片技术上落后的说法，表明他们现在有能力构建顶级 AI 模型，甚至可能在主要 AI 公司之前实现 AGI。
  - **CUDA 的主导地位**：许多评论强调了 **CUDA** 在 AI 开发中的重要性，指出它是一个与 **TensorFlow** 和 **PyTorch** 等主流框架深度集成的专有平台。虽然有人认为存在 **AMD 的 ROCm** 等替代方案，但其他人认为复制 CUDA 的生态系统是一个巨大的挑战，尽管在投入充足的情况下并非不可逾越。
  - **华为的竞争地位**：对于 **华为 Ascend 910C** 媲美 **Nvidia H100** 的说法存在怀疑。一些用户认为 910C 仅达到了 H100 性能的 60%，而华为的策略不是直接与 Nvidia 竞争，而是在 Nvidia 受限的市场夺取份额，利用其自身的 **CANN** 平台作为 CUDA 的替代方案。
  - **市场动态与开源**：讨论涉及了**开源开发者**在实现 **AGI** 后可能转向非开源模型的可能性。有一种观点认为，由于市场限制，华为可能会加大开发力度以追赶 Nvidia，但这可能需要 3-5 年才能达到同等水平，期间可能会利用第三方渠道获取 Nvidia 硬件。


**主题 3. O3 Mini：OpenAI 的易用性飞跃**

- **O3 mini 确实感觉很好用** ([评分: 104, 评论: 17](https://reddit.com/r/OpenAI/comments/1ihwez2/o3_mini_actually_feels_useful/)): **O3 Mini** 作为一个 OpenAI 模型，最初通过为编程问题提供巧妙的解决方案或 Bug 修复给用户留下了深刻印象，其效果似乎比 **O1 (non-pro)** 更好。然而，经过进一步评估，建议的解决方案并未按预期工作。
  - **O3 Mini** 最初看起来令人印象深刻，但未能提供有效的解决方案，这表明像 **Claude 系列**这样的其他 AI 模型在任务泛化方面可能表现更好。**Mescallan** 认为，大多数模型在特定基准测试中会出现峰值，但缺乏泛化能力。
  - **O1 Pro** 被认为在编程任务中更可靠，**MiyamotoMusashi7** 表示在代码相关任务中对其非常信任，同时也承认在其他领域可能存在 Bug。
  - **gentlejolt** 强调了一种提高代码质量的变通方法，即指示 AI 进行“重构（rearchitect）”并针对可读性和可维护性进行优化，尽管最终结果只是原始代码的略微改进版本。


**主题 4. OpenAI 发布 OpenAI Sans 字体**

- **[Refreshed.](https://www.youtube.com/watch?v=k3d_xeVxEOE)** ([Score: 259, Comments: 140](https://reddit.com/r/OpenAI/comments/1ihrssx/refreshed/)): **OpenAI** 推出了一种新字体作为其品牌战略的一部分，标志着视觉形象的焕然一新。此次更新是其持续提升品牌存在感和用户参与度努力的一部分。
  - 许多评论将 **OpenAI** 的新字体与 **Apple** 的设计理念进行了比较，暗示 OpenAI 的设计团队可能包含前 Apple UX 设计师。这一设计变革被视为拥有独特字体的战略举措，类似于 Apple 创作 **San Francisco font**，从而降低长期授权成本。
  - 舆论对新字体的必要性持怀疑态度，评论认为此举更多是为了品牌塑造和向投资者证明支出的合理性，而非实质性的创新。一些用户幽默地批评了这一努力，将其等同于花费数十亿美元将字体从 **Arial 更改为 Helvetica**。
  - 几条评论强调了以设计为中心的品牌战略与技术型受众期望之间可能存在的脱节。**"OpenAI sans"** 的创建被视为一种战略性的品牌举措，但它对非设计师的直接价值受到质疑，一些评论者认为视频演示过于夸张，且与他们的兴趣不直接相关。

---

# AI Discord 简报

> 由 o1-mini-2024-09-12 生成的摘要之摘要总结

**主题 1. 模型优化狂热**
  
- [**DeepSeek R1 尺寸缩减**](https://github.com/klara-research/klarity)：**DeepSeek R1** 模型成功量化至 *1.58 bits*，将其体积从 720GB 锐减 **80%** 至 131GB，同时保持其在配备 **36GB** RAM 的 **MacBook Pro M3** 上正常运行。
- [**Phi-3.5 的审查闹剧**](https://huggingface.co/SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored)：用户纷纷嘲讽 **Phi-3.5** 过度的审查制度，促使 Hugging Face 上出现了[无审查版本](https://huggingface.co/SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored)。
- [**Harmonic Loss 崭露头角**](https://arxiv.org/abs/2502.01628)：引入 **Harmonic Loss**，这是一种新的训练损失函数，在速度和可解释性上均优于 Cross-Entropy，彻底改变了模型泛化和理解数据的方式。

**主题 2. AI 工具之战**
  
- [**Cursor 击败 Copilot**](https://www.cursor.com/changelog)：在 AI 编程助手的对决中，**Cursor** 的表现优于 **GitHub** 的 **Copilot**，提供了更卓越的性能和实用性（尤其是在小型代码库中），而 Copilot 则拖慢了工作流。
- [**OpenRouter 迎来 Cloudflare**](https://openrouter.ai/google/gemma-7b-it)：**Cloudflare** 加入 **OpenRouter**，集成了其 **Workers AI** 平台并发布了具备 Tool Calling 能力的 **Gemma 7B-IT**，为开发者扩展了生态系统。
- [**Bolt 的备份忧虑**](https://github.com/stackblitz/bolt.new/issues/2985)：用户对 **Bolt** 不可靠的备份和性能问题表示不满，强调了 AI 开发领域对更稳健解决方案的需求。

**主题 3. 伦理与安全乱象**
  
- [**Anthropic 的 20% 负担**](https://www.anthropic.com/research/constitutional-classifiers)：**Anthropic** 新推出的 **Constitutional Classifiers** 引发关注，其推理成本增加了 **20%**，误拒率增加了 **50%**，引发了关于 AI 安全有效性的辩论。
- [**EU AI Act 的焦虑**](https://eur-lex.europa.eu/eli/reg/2024/1689)：严厉的 **EU AI Act** 让社区对严格的监管以及欧洲境内 AI 运营的前景感到担忧，甚至在该法案正式实施前就已如此。
- [**AI 版权灾难**](https://annas-archive.org/blog/ai-copyright.html)：对 AI 公司在未经适当许可的情况下使用受版权保护数据的担忧激增，呼吁建立类似于音乐行业的**强制许可制度**，以确保创作者获得补偿。

**主题 4. 黑客松与协作火花**
  
- [**3.5 万美元黑客松升温**](https://lu.ma/fyu8iqnk)：一场协作黑客松宣布与 **Google Deepmind**、**Weights & Biases** 等机构合作，为开发增强用户能力的自主 AI Agent 提供超过 **3.5 万美元的奖金**。
- [**R1-V 项目革命**](https://x.com/liangchen5518/status/1886171667522842856?s=46&t=b1X88nwMsmZgHkmMFkiG3g)：**R1-V** 项目展示了一个仅需 **100 个训练步骤**、成本低于 **3 美元** 即可击败 **72B** 对应模型的模型，承诺完全**开源**并引发了社区关注。
- [**Pi0 投入行动**](https://x.com/RemiCadene/status/1886823939856589296)：**Pi0** 是一款先进的 Vision Language Action 模型，已在 **LeRobotHF** 上发布，能够通过自然语言指令执行自主动作，并可针对各种机器人任务进行微调。

**主题 5. AI 在法律与客户服务领域**
  
- [**律师青睐 NotebookLM**](https://youtu.be/mormPD6QYkQ?feature=shared)：一位巴西律师称赞 **NotebookLM** 能高效起草法律文件，利用其可靠的来源引用功能提升了生产力。
- [**客户服务转型**](https://www.perplexity.ai/search/show-me-sites-that-list-phobia-Jb9EQhckS66QFqIDYJvj1A)：用户探索了 **NotebookLM** 如何通过自动创建客户档案和减少 Agent 培训时间来彻底改变客户服务，使支持更具扩展性和效率。
- [**政治 AI Agent 发布**](https://www.perplexity.ai/search/motherboard-msi-a520m-a-pro-aoKBBPs2Skystw7fZnDqJw#1)：Society Library 推出了一款**政治 AI Agent**，作为数字辩论中的教育中介，通过 AI 驱动的讨论增强数字民主。


---

# 第一部分：高层级 Discord 摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **DeepSeek R1 量化版！**：**DeepSeek R1** 模型已量化至 *1.58 bits*，体积从 720GB 缩减至 131GB（**减少了 80%**），同时保持了功能性。
   - 这是通过对特定层选择性地应用更高位数实现的，避免了以往导致输出乱码的简单全量化方案；该模型目前可在配备 **36GB** RAM 的 **MacBook Pro M3** 上运行。
- **微调策略揭秘**：讨论强调，减少数据集规模并专注于高质量样本可以提升训练效果，建议针对代码分类任务微调在代码上预训练的模型。
   - 参与者还探索了使用模型生成合成数据集，并调整 Loss 函数以有效管理类别不平衡。
- **Klarity 库面世！**：新的开源 **Klarity** 库允许分析语言模型输出的熵（entropy），为决策过程提供更深入的洞察，并已开放给 Unsloth 量化模型进行测试。
   - 该库提供详细的 JSON 报告以便彻底检查；[点击此处查看](https://github.com/klara-research/klarity)。
- **MoE 专家配置：静态是关键**：针对 MoE 框架中专家配置的困惑得到了解答，强调专家数量在模型运行期间通常应保持 *静态（static）*。
   - 用户最初不确定应使用默认的 8 个还是上限 256 个专家，相关的澄清旨在解决这一疑虑。
- **保加利亚语模型的惊人飞跃！**：一个保加利亚语模型展示了相对于基础模型的显著改进，困惑度（perplexity）分数大幅下降（短文本 *PPL：72.63* 对比 *179.76*）。
   - 困惑度的降低突显了该模型在理解和处理保加利亚语方面能力的增强。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 征集新文档快捷方式**：一名成员正在为 [Windsurf](https://x.com/kevinh22/status/1886827501004931511) 收集新的 `@docs` 快捷方式，征集贡献以增强文档体验。
   - 目标是通过高效处理资源来改进文档访问，并感谢 **Mintlify** 通过 `/llms.txt` 自动托管所有文档，这使得 Agent 能够避免 HTML 解析。
- **Codeium 性能受损**：用户报告 **Claude** 的工具调用（tool utilization）效果不佳，导致因重复失败而产生高额额度消耗，一些人建议当工具产生错误时不应扣除额度。
   - 其他成员在尝试登录 VSCode 账号时遇到了内部证书错误，并尝试通过不同网络和支持寻求帮助。
- **用户质疑 Windsurf O3 Mini 定价**：用户对 **Windsurf** 的 **O3 Mini** 定价表示担忧，质疑其定价是否应与 Claude 3.5 Sonnet 持平，考虑到其性能和高额度消耗。
   - 许多用户无法修改文件，这经常导致内部错误，因此一些人要求更公平的定价。
- **Windsurf 模型上下文窗口限制**：用户报告 **Windsurf** 在修改或更新文件时失败，同时担心有限的上下文窗口（context window）会影响模型性能，且更倾向于使用 Claude。
   - 反馈强调在超过上下文容量时需要更清晰的警告，并解决工具调用失败的问题；同时一些人正在探索创建 `.windsurfrules` 文件来管理全栈 Web 应用程序。
- **黑客松邀请 AI Agent 爱好者**：宣布了一场奖金超过 **3.5 万美元** 的协作黑客松，邀请参与者开发自主 **AI Agent**。
   - 参与者将展示旨在通过 AI 技术提升用户能力的项目；同时社区对 **Qodo**（原 Codium）的可靠性评价褒贬不一。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **O1 Pro 碾压代码生成耗时**：用户在使用 **O1 Pro** 时看到了巨大的速度提升，部分用户在不到五分钟内就能生成大量代码，表现远超 **O3 Mini**。
   - 这些用户注意到响应时间更快，且对复杂任务的处理能力更强。
- **弱模型在 Aider 中展现强大价值**：对于生成 commit message 和总结聊天内容等任务，成员们建议弱模型可能比 **DeepSeek V3** 等强模型更具成本效益且更高效。
   - 社区正在寻找既经济实惠又有效，且可以进行微调的模型。
- **相比直接使用 API，更倾向于 OpenRouter**：成员们发现使用 **OpenRouter** 能提供更好的可用性，并且能够优先选择特定的提供商而非直接访问 API。尽管速度可能较慢，但 **CentML** 和 **Fireworks** 是有效的 DeepSeek 提供商。
   - 更多信息请查看 [Aider documentation](https://aider.chat/docs/llms/openrouter.html#controlling-provider-selection)。
- **寻求 Aider 文件管理自动化**：在 Aider 中手动添加文件非常繁琐，因此用户正在寻求自动化方法。目前已有一个 VSCode 插件可以自动添加当前打开的文件。
   - 有人指出目前已有 repo map 可用，因此实现起来应该很直接。
- **关于 Aider 聊天模式的说明**：成员们请求更多关于 `code`、`architect`、`ask` 和 `help` 模式如何改变 Aider 中的交互和命令的信息，使用 `/chat-mode` 命令可以切换当前模式。
   - 解释强调了当前模式会影响模型的选择，详见 [Aider documentation](https://aider.chat/docs/usage/modes.html)。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor IDE 更新引发褒贬不一的反应**：用户在最近的 **Cursor** 更新中遇到了问题，注意到与之前版本相比性能变慢且存在 bug，一些人仍在坚持使用，而另一些人则表示沮丧。
   - 一些用户觉得目前的模型无法有效替代之前的体验，而另一些人则指出 **Fusion Model** 的推出在 [changelogs](https://www.cursor.com/changelog) 中并不清晰。
- **Cursor 的替代方案涌现**：用户讨论了 **Supermaven** 和 **Pear AI** 等替代方案，意见不一；有些人觉得 **Supermaven** 很快，但不如 **Cursor** 可靠，尤其是在免费版中。
   - 一位用户分享了 [Repo Prompt](https://repoprompt.com/) 的链接，另一位用户分享了他的 [AI Dev Helpers](https://github.com/vbwyrde/AI_Dev_Helpers) 仓库链接。
- **AI 工具的成本引发担忧**：**Cursor** 和 **GitHub Copilot** 等 AI 工具的高昂成本令一些用户感到担忧，他们担心负担不起。
   - 虽然有些人寻求更低成本的选择，但另一些人认为 **Cursor** 的价值证明了其价格的合理性。
- **多样化的 AI 模型体验**：体验各不相同，一些用户成功使用 **Cursor** 构建了项目，而另一些人则对 AI 生成的错误感到沮丧；一位用户在[这段视频](https://youtu.be/FrM6ZzCiLwU)中分享了他使用 **DeepSeek R1 + Claude 3.5 Sonnet** 的 2 分钟工作流。
   - 讨论内容包括使用 **Claude Sonnet** 等模型以及解决实际挑战。
- **社区围绕 Cursor 动员起来**：用户分享了 **GitHub** 仓库链接，例如 [`awesome-cursorrules`](https://github.com/PatrickJS/awesome-cursorrules?tab=readme-ov-file)，旨在增强 **Cursor** 的功能，优化其使用并提升编程任务的用户体验。
   - 这些资源使 **Cursor** 能够实现增强功能，例如 [devin.cursorrules](https://github.com/grapeot/devin.cursorrules/tree/multi-agent) 项目的多 Agent 版本。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Deepseek R1 600B 表现出色**：在向 Together.ai 的 **Deepseek R1 600B** 展示了一个复杂的网格任务后，一名成员指出，相对于较小的模型，它产生了出色的结果。
   - 提供的截图显示其能够推导出正确的字母，表明了先进的推理能力，给 AI Engineer 观众留下了深刻印象。
- **Anthropic 分类器面临成本和性能问题**：一位成员分享了他们对关于 [constitutional classifiers](https://www.anthropic.com/research/constitutional-classifiers) 论文的担忧，指出推理成本增加了 **20%**，误拒率增加了 **50%**，这影响了用户体验。
   - 还有人建议，随着模型能力的提升，分类器可能无法充分防御危险的模型能力，特别是当模型变得更加强大时，这引发了对 **alignment strategies**（对齐策略）的批评。
- **AI 训练数据的伦理引发辩论**：成员们辩论了为 AI 训练定义“可疑”数据源的挑战，并对使用 **Wikipedia** 等数据集的影响以及 AI 能力的道德性表示担忧。
   - 这突显了关于数据所有权和 AI 开发中伦理考量的更广泛辩论，特别是在 [copyright reform](https://annas-archive.org/blog/ai-copyright.html)（版权改革）的背景下。
- **AI 公司躲在版权法背后**：一位成员强调，**AI 公司**经常躲在**版权和专利法**背后以保护其知识产权，这在不受限的访问和严格的控制之间造成了困境。
   - *Snake-oil selling*（卖蛇油/虚假宣传）被提及作为对这些做法的批评，暗示其主张中存在欺骗，并可能扼杀创新。
- **幻觉被视为自然行为**：关于 **LLM outputs** 中“幻觉”概念的辩论出现了，一些人认为这是模型行为的自然方面，而不是缺陷。
   - 成员们批评“幻觉”一词具有误导性，并将基于学习模式生成输出的技术拟人化，同时也认为消除幻觉是一个无法实现的目标。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Deepseek R1 表现不如 Qwen**：用户发现 **Deepseek R1 abliterated Llama 8B** 模型与较小的 **Qwen 7B** 和 **1.5B models** 相比表现平平，并指出其性能不稳定。
   - 一位用户询问如何完全去除模型的审查（uncensor），强调了新旧版本之间能力的差异。
- **API 模型使用的澄清**：讨论澄清了 API 调用中的 'local-model' 是作为特定模型名称的占位符，特别是在加载了多个模型的设置中 ([LM Studio API Docs](https://lmstudio.ai/docs/api))。
   - 在发出 API 请求之前明确获取模型名称可以避免模型选择的歧义，REST API 统计信息增强了这一点 ([LM Studio REST API](https://lmstudio.ai/docs/api/endpoints/rest#get-apiv0models>))。
- **Intel Mac 支持已停止**：LM Studio 版本仅在 Apple Silicon 上受支持，并且由于其保持闭源状态，没有自行构建的选项。
   - 用户为那些使用基于 Intel 的 Mac 的人推荐了替代系统，因为官方不提供支持。
- **RAG 在无需微调的情况下增强推理**：用户探索了使用检索增强生成 (**RAG**) 来增强 LM Studio 在特定领域任务中的推理能力，而无需进行微调 ([LM Studio Docs on RAG](https://lmstudio.ai/docs/basics/rag))。
   - 在考虑模型微调等更复杂的解决方案之前，利用向量库中的领域知识被强调为第一步。
- **对 M4 Ultra 性能的怀疑**：成员们对 **M4 Ultra** 提供强劲性能的能力表示怀疑，传闻指出 128GB RAM 系统的起售价为 **1200 美元**。
   - 一些人推测它可能无法超越 NVIDIA 的 **Project DIGITS**，后者在集群模型方面具有更优越的互连速度。

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 欢迎新任安全负责人**：Perplexity 通过一段名为 [*Jimmy*](https://cdn.discordapp.com/attachments/1047204950763122820/1336120617967026319/Jimmy.MOV?ex=67a34f8b&is=67a1fe0b&hm=f6e56ab3c0f598299bf922a9aedeb20380d5ab088b4921c9f03f6867e0ef3437&) 的视频介绍了其新任 **Chief Security Officer** (CSO)，强调了安全技术进步的重要性。
   - 该公告旨在让社区在新的安全策略上与领导层保持一致。
- **Perplexity Pro 因查询限制受到赞赏**：用户对 **Perplexity Pro** 计划表示赞赏，因为其提供几乎不限次数的每日 **R1** 使用量，认为这是一项非常有价值的服务。
   - 一位用户将其与 DeepSeek 的查询限制进行了对比，并称赞了 **Perplexity** 的服务器性能。
- **Sonar 模型弃用导致流程变慢**：一位用户报告称，在两周前收到 **llama-3.1-sonar-small-128k-online** 的弃用通知后，切换到 `sonar` 后经历了 **5-10 秒的延迟增加**。
   - 他们询问了这种延迟的预期性质，并寻求缓解建议。
- **分享恐惧症和主板资源**：一位用户分享了一个链接，内容是列出恐惧症的网站，为进一步阅读提供了整合资源（[点击此处](https://www.perplexity.ai/search/show-me-sites-that-list-phobia-Jb9EQhckS66QFqIDYJvj1A)），以及 **MSI A520M A Pro 主板** 的信息（[点击此处](https://www.perplexity.ai/search/motherboard-msi-a520m-a-pro-aoKBBPs2Skystw7fZnDqJw#1)）。
   - **MSI A520M** 链接包含详细的对比和用户体验，而恐惧症链接则列出了各种恐惧症及其描述。
- **API 用户请求图像访问权限**：一位寻求为其 **PoC** 获取图像的 **API 用户** 发现，需要成为 **tier-2 API 用户** 才能访问此功能。
   - 他们询问是否可以授予临时访问权限，以便利用现有的额度进行图像检索。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **DeepSeek 引发社区关注**：用户讨论了 **DeepSeek R1** 如何使 AI 技术民主化，但也引发了对数据可能被发送到海外的担忧，导致人们呼吁增加透明度并分析 [DeepSeek R1 与 o3-mini 的性能对比](https://dataconomy.com/2025/01/31/deepseek-r1-vs-o3-mini-in-performance-cost-and-usability-showdown/)。
   - 讨论中包括了一个 Reddit 帖子的链接，该帖子介绍了一个*更简单且开源 (OSS) 版本*的 **OpenAI** 最新 Deep Research 功能，以及一段质疑 **DeepSeek** 是否诚实的 YouTube 短视频，强调了隐私和网络安全问题。
- **O1 Pro 的小游戏生成能力令人印象深刻**：成员们分享了使用 **O1 Pro** 的积极体验，报告称其能够在单次会话中无错误地生成多个小游戏，展示了其强大的性能，并促使一位用户计划使用更具挑战性的提示词进行严格测试。
   - 对 **O1 Pro 能力** 的赞赏引发了关于模型性能和 AI 编排服务的更广泛讨论。
- **结构化生成调整模型性能**：一位成员讨论了在 **JSON schemas** 和 **Pydantic models** 中利用“thinking”字段来增强推理期间的**模型性能**。
   - 他们提醒说，这种方法可能会*污染数据结构定义*，但通过开源的 **UIForm** 工具（可通过 `pip install uiform` 安装）利用 **JSON Schema extras**，可以简化字段的添加或删除。
- **用户思考 GPT-4o 的推理能力**：用户对 **GPT-4o 推理能力** 的最新增强表示疑问，并对 **OpenAI** 的更新反应不一，一位用户注意到代码回复中表情符号的使用增加，这可能会降低对编码本身的关注。
   - 一位成员按 1 到 10 的等级对 **Deep Research 信息** 的准确性进行了评分，表明了对其可靠性的兴趣，而其他用户则询问了 **Pro 版本** 的设备限制。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **SoftBank 向 OpenAI Ventures 注资数十亿美元**：SoftBank 计划每年向 **OpenAI** 产品投资 **30 亿美元**，并成立了一家专注于日本市场的合资企业 **Cristal Intelligence**，这可能使 OpenAI 的估值达到 **3000 亿美元**。
   - 根据[这条推文](https://fxtwitter.com/btibor91/status/1886508640263397705)，该合资企业旨在提供面向业务的 **ChatGPT** 版本，标志着 OpenAI 在亚洲市场的显著扩张。
- **Google Gemini 迎来 Workspace 集成大改**：**Gemini for Google Workspace** 将停止使用插件（add-ons），转而将 AI 功能直接集成到商业版（Business）和企业版（Enterprise）中，以提升生产力和数据治理，服务超过一百万用户。
   - 这一战略举措旨在改变企业使用生成式 AI 的方式，详见 [Google 官方公告](https://support.google.com/a/answer/13623623)。
- **DeepSeek V3 在华为 Ascend 上大显身手**：**DeepSeek V3** 模型现在能够在华为 **Ascend** 硬件上进行训练，将其可用性扩展到了更多的研究人员和工程师。
   - 尽管对其性能可靠性和成本降低的说法存在疑虑，但根据[这条推文](https://x.com/teortaxesTex/status/1886526422493143268)，这一集成标志着该平台向前迈进了一步。
- **OpenAI 瞄准机器人和 VR 头显**：OpenAI 已提交商标申请，信号显示其意图进入硬件市场，推出**人形机器人**和 **AI 驱动的 VR 头显**，可能向 Meta 和 Apple 发起挑战。
   - 正如 [Business Insider 的这篇文章](https://www.businessinsider.com/openai-trademark-humanoid-robots-vr-headsets-sam-altman-hardware-2025-2)所指出的，此举使 OpenAI 处于应对**拥挤的硬件挑战**的复杂局面中。
- **Prime 论文发布，深入探讨隐式奖励**：*备受期待*的 [Prime 论文](https://arxiv.org/abs/2502.01456)已经发布，由 **Ganqu Cui** 和 **Lifan Yuan** 贡献，引入了通过隐式奖励（implicit rewards）优化模型性能的新概念。
   - 该出版物有望重塑对强化学习的理解，*为优化模型性能提供创新解决方案*。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **LlamaGen 起步面临挑战**：新的 **LlamaGen** 模型承诺通过 *next-token prediction* 提供顶级的图像生成，性能可能超越 **LDM** 和 **DiT** 等扩散框架。
   - 然而，与扩散模型相比，其**生成速度慢**的问题引起了关注，暗示了潜在的优化需求，并对论文 [Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation](https://arxiv.org/abs/2406.06525) 中缺失生成时间对比提出了质疑。
- **Triton 优化挑战依然存在**：一位用户报告称，在尝试优化密集内存操作时，其 Triton 代码比 PyTorch **慢 200 倍**，并寻求性能调优方面的帮助。
   - 有建议认为，优化源自矩阵 **k** 不同行交叉的 **k_cross** 对大维度至关重要，但如果没有 autotuning，**TMA** 可能无法提供优于传统方法的预期改进。
- **缓存低效问题再次出现**：在一次 CUDA 讨论中，成员们注意到，*如果输入大于 L2 缓存且正在进行流式传输（streaming）*，那么**缓存将完全失效**，甚至在单个流中也会导致持续的抖动（thrashing）。
   - 有人担心，在利用 tensor cores 的 kernel 中，**整数操作（integer operations）**使用的增加会影响 **FP 操作**的性能，一些人认为如果受限于 FMAs，INT/FP 的区别就不那么重要了。
- **FlashAttention 导致输出质量下降**：一位用户发现，虽然使用 **Flash Attention 3 FP8 kernel** 提高了其 diffusion transformer 模型的推理速度，但输出质量显著下降。
   - 一种假设认为，**FP32** 和 **FP8** 之间细微的差异（约 **1e-5**）会在 softmax 过程中累积，影响长上下文中的注意力分布，[NVIDIA 官方文档](https://developer.nvidia.com/blog/optimizing-gpu-performance-with-cuda-memory-management/)被引用为相关阅读材料。
- **Cursor 夺冠，Copilot 降级**：用户发现 **Cursor** 和 **GitHub Copilot** 之间的差异是*天壤之别*，Cursor 提供了卓越的性能和实用性，特别是在小型代码库中。
   - 据报道，免费版的 **Copilot** 会减慢工作流程，整体帮助较小，特别是在人类判断证明更有效的大型代码库中。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **功能性语言模型的概率揭晓**：研究人员在权重空间开发出一种[随机采样方法](https://arxiv.org/abs/2501.18812)后，确定随机猜测功能性语言模型权重的概率约为 **3.6 亿分之一**。
   - 该方法可能为理解网络复杂性提供见解，展示了随机撞上功能性配置的可能性微乎其微。
- **Harmonic Loss 成为训练领域的变革者**：一篇新论文介绍了 **Harmonic Loss**，作为一种比 cross-entropy loss 更具可解释性且收敛更快的替代方案，它在各种数据集上表现出更优的性能，详见[这篇 arXiv 论文](https://arxiv.org/abs/2502.01628)。
   - **Harmonic 模型**在泛化和可解释性方面优于标准模型，表明对未来的 LLM 训练具有显著益处；一位研究人员想知道，鉴于[这条推文](https://fixupx.com/ChrSzegedy/status/1886881600367161679)中表达的潜在益处，调和加权注意力（harmonically weighted attention）的效果会如何。
- **多项式 Transformer 引发关注**：成员们讨论了**多项式（二次）Transformer** 的潜力，建议替换 MLP 可以提高模型效率，特别是在注意力机制中，如 [Symmetric Power Transformers](https://manifestai.com/articles/symmetric-power-transformers/) 所示。
   - 对话围绕经典模型与双线性方法展开，并强调了在规模化时的参数效率和复杂性之间的权衡。
- **提出自定义 LLM 组装工具**：一名成员提议开发一种**拖拽式工具**来组装自定义 LLM，使用户能够实时可视化不同架构和层如何影响模型行为。
   - 这个概念被当作一个有趣的业余项目来讨论，反映了社区对动手定制 LLM 的兴趣。
- **DeepSeek 模型遇到评估小故障**：一名成员报告在使用 **llm evaluation harness** 对 DeepSeek distilled 模型进行评估时得分较低，并怀疑 `<think>` 标签可能是原因。
   - 他们请求关于验证该问题或在评估期间忽略标签的建议，表明了对评估偏差的担忧。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DeepSeek 改变 AI 格局**：一段 [YouTube 视频](https://www.youtube.com/watch?v=3DEPDN8oD0w)强调了 **DeepSeek** 如何改变了 AI 的发展轨迹，引发了关于 Altman 在开源立场与实际行动之间差异的辩论。
   - 评论认为，由于 Altman 的言论与对开源倡议的实际支持之间存在差距，他被贴上了“吹鼓手（hypeman）”的标签。
- **推荐系统成熟缓慢**：新成员 Amith 分享了使用开源推荐系统 **Gorse** 的经验，指出这些系统仍需要时间来成熟。
   - 另一名成员建议探索 **ByteDance** 的技术，以扩大关于可用推荐资源的讨论。
- **在教授 AI 价值观方面的 RL 挑战**：讨论了**强化学习 (RL)** 是否能为 AI 注入好奇心等内在价值观，尽管维持学习行为的复杂性已被注意到。
   - Juahyori 强调了在持续学习中维持已学习行为的难度，并强调了对齐（alignment）方面的挑战。
- **推出政治 AI Agent**：Society Library 推出了一款**政治 AI Agent**，作为其增强数字民主的非营利使命的一部分。
   - 该 AI Agent 将在数字辩论中充当教育中介聊天机器人，利用 Society Library 的基础设施。
- **SWE Arena 增强 Vibe Coding**：**SWE Arena** 支持实时执行程序，使用户能够比较多个 AI 模型的编程能力。
   - 它具有系统提示词（system prompt）自定义和代码编辑功能，符合 **Vibe Coding** 范式，专注于 [swe-arena.com](http://swe-arena.com/) 上的 AI 生成结果。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **用户寻找图生视频软件**：一位用户询问了图生视频软件，提到 NSFW 内容屏蔽是一个限制，另一位用户建议探索 **LTX** 作为潜在解决方案。
   - 该咨询表明需要一种在保持多样化内容功能的同时，能够绕过内容限制的工具。
- **Stable Diffusion 质量陷入瓶颈**：一位用户对 **Stable Diffusion** 反复生成低质量图像表示沮丧，特别提到了无意中出现的“双重身体”等特征，并寻求在不重启软件的情况下清理缓存的建议。
   - 该问题突显了在长期使用 Stable Diffusion 时保持一致输出质量的潜在挑战。
- **蓝妹妹生日祝福引发版权担忧**：一位用户请求帮助使用 Stable Diffusion 创建一张包含 **Smurfette**（蓝妹妹）的非 NSFW 生日图像，并指出了使用 **DALL-E** 时的版权担忧。
   - 该请求强调了对能够生成特定、家庭友好型内容，同时规避版权问题的模型的需求。
- **审查担忧中讨论模型性能**：用户讨论了 **Stable Diffusion 3.5** 和 **Base XL** 等模型的性能，对其审查程度和整体有效性持有不同意见，其中一项讨论建议 **fine-tuning** 可能会减少审查。
   - 讨论反映了对模型偏见以及审查与创作控制之间权衡的持续关注。
- **在 A1111 中寻求精确角色编辑**：一位用户寻求关于在 **A1111** 中使用 prompt 编辑多人物图像中单个角色的建议，旨在区分发色等特征。
   - 虽然提到了 **inpainting** 等技术，但用户希望有一种更精确的方法，表明对 A1111 内高级编辑工具的需求。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **Toastinator 推出 'Roast or Toast' 播客**：AI 烤面包机 **Toastinator** 推出了名为 [‘Roast or Toast’](https://youtu.be/mormPD6QYkQ?feature=shared) 的播客，通过赞美与批判的结合来探索生命的意义。
   - 首期节目邀请听众见证 **The Toastinator** 是会对存在的宏大奥秘进行赞美（toast）还是吐槽（roast）。
- **律师使用 NotebookLM 高效起草**：巴西的一位律师正在利用 NotebookLM 起草法律文件和研究案例，理由是该工具的来源引用具有可靠性。
   - 他们现在正使用该工具为重复性的法律文件调整模板，显著提升了流程效率。
- **NotebookLM 的客服潜力**：一位用户询问了 NotebookLM 在 **BPOs** 等客户服务中的应用，重点关注真实世界的经验和用例。
   - 潜在好处包括减少 Agent 培训时间和创建客户档案。
- **Google 账号故障困扰 NotebookLM 用户**：一位用户报告在使用 NotebookLM 时其常规 Google 账号被停用，怀疑是潜在的年龄验证问题。
   - 另一位用户强调在解决类似问题时，需要仔细检查账号设置和权限。
- **Workspace 访问困扰**：成员们讨论了在 **Google Workspace** 中为特定群体而非整个组织激活 **NotebookLM Plus**。
   - 分享了关于使用 **Google Admin console** 通过组织单位配置访问权限的说明，以确保受控部署。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Anthropic 推出 Claude Constitutional Classifiers 挑战用户**：Anthropic 发布了他们的 [Claude Constitutional Classifiers](https://claude.ai/constitutional-classifiers)，邀请用户尝试 8 个难度级别的越狱（jailbreak），以测试为**强大的 AI 系统**准备的新安全技术。
   - 该发布包含一个演示应用，旨在评估和改进针对潜在漏洞的**安全措施**。
- **FAIR 内部冲突引发关于 Zetta 和 Llama 的辩论**：社交媒体上的讨论凸显了 FAIR 内部关于 **Zetta** 和 **Llama** 模型开发的动态，特别是围绕透明度和竞争行为（[示例](https://x.com/namangoyal21/status/1886515845133951192?s=46)）。
   - Yann LeCun 等关键人物暗示，更小、更灵活的团队比大型项目更具创新性，这引发了对 **FAIR 组织文化**进行更深入审视的呼声（[示例](https://x.com/suchenzang/status/1886544793511080103?s=46)）。
- **Icon 实现广告创作自动化**：**Icon** 结合了 ChatGPT 与 CapCut 的功能，旨在为品牌自动创建广告，每月可制作 300 条广告（[来源](https://x.com/kennandavison/status/1886836061378372064)）。
   - 在来自 **OpenAI**、**Pika** 和 **Cognition** 的投资者支持下，Icon 集成了视频打标、脚本生成和编辑工具，在显著降低成本的同时提升广告质量。
- **DeepMind 发布 LLM 扩展教科书**：Google DeepMind 发布了一本名为《How To Scale Your Model》的教科书，可在 [jax-ml.github.io/scaling-book/](https://jax-ml.github.io/scaling-book/) 获取，该书从系统视角揭秘了 **LLMs**，并侧重于数学方法。
   - 该书强调通过简单的方程式理解模型性能，旨在提高运行**大型模型**以及使用 **JAX** 软件栈 + Google 的 TPU 硬件平台的效率。
- **Pi0 通过自然语言释放自主机器人行动**：Physical Intelligence 团队推出了 **Pi0**，这是一种先进的 Vision Language Action 模型，它使用自然语言命令来实现自主行动，目前已在 LeRobotHF 上线（[来源](https://x.com/RemiCadene/status/1886823939856589296)）。
   - 随模型一起发布的还有预训练的 Checkpoints 和代码，方便对各种**机器人任务**进行**微调（fine-tuning）**。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **MathJax 在 LaTeX 支持方面受到关注**：成员们探索了集成 [MathJax](https://www.mathjax.org/) 以增强 **LaTeX 支持**，强调了其 **SVG 导出**功能对于广泛兼容性的必要性。
   - 建议包括有选择地解析 MathJax 并将其应用于包含 LaTeX 符号的文档部分。
- **DeepSeek 在 LocalDocs 使用中遇到小问题**：用户在使用 **DeepSeek** 配合 localdocs 时遇到了问题，报告了诸如“*item at index 3 is not a prompt*”之类的错误。
   - 在等待主分支预期修复的同时，一些用户发现特定版本的模型性能更好。
- **欧盟 AI 法案引发关注**：欧盟新的 **AI Act** 因其对 **AI 使用**的严格监管（包括禁止某些应用）而引发关注，详见[官方文档](https://eur-lex.europa.eu/eli/reg/2024/1689)。
   - 成员们分享了信息资源，指出即使在规则完全生效之前，这对欧盟境内的 **AI 运营**也有重大影响。
- **欧盟的全球角色引发争议**：关于欧盟的全球政治立场，特别是涉及**帝国主义和人权**的问题，爆发了激烈的辩论。
   - 参与者交换了尖锐的批评，强调了在讨论**欧盟政策和行动**时感知到的情绪化反应和逻辑谬误。
- **AI 交流面临障碍**：用户之间的互动凸显了在**民主和治理**等复杂话题上维持**成熟讨论**的困难。
   - 成员呼吁将对话重新聚焦于 **AI 相关话题**，强调尊重对话和意识到个人偏见的必要性。

---

## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Supabase 在集成偏好上略胜 Firebase 一筹**：成员们讨论了 **Supabase** 与 **Firebase** 的优劣，由于 **Supabase** 在某些用例中具有无缝集成能力，偏好更倾向于它。
   - 一些人承认在技术上更习惯 **Firebase**，但对话强调了数据库服务中的多样化需求。
- **Bolt 深受性能问题困扰**：用户报告了 **Bolt** 存在的重大性能问题，包括加载缓慢、身份验证错误以及更改无法正确更新。
   - 一位用户提到刷新应用程序可以暂时缓解问题，但这些问题的间歇性发生导致了持续的挫败感。
- **Bolt 用户抱怨备份难题**：一位用户表达了对在 **Bolt** 中丢失数小时工作的担忧，因为目前可用的最新备份还是 1 月初的，同时还提出了[显示 .bolt 文件夹](https://github.com/stackblitz/bolt.new/issues/2985)的功能请求。
   - 虽然有人建议检查备份设置，但过时的备份凸显了可靠性问题。
- **GDPR 合规性担忧引发对托管方案的寻找**：用户质疑 **Netlify** 的 **GDPR-compliance**（合规性），特别是关于欧盟境内的数据处理，参见其 [隐私政策](https://www.netlify.com/privacy/)。
   - 该咨询引发了对替代托管解决方案的搜索，以确保所有托管和数据处理活动都留在欧盟境内，从而维持监管合规性。
- **API Key 身份验证难题**：一位用户在 **Bolt** 中使用 **Supabase edge functions** 进行 **RESTful API** 请求时，遇到了 **API key authentication** 困难，出现了 **401 Invalid JWT** 错误。
   - 由于 edge functions 缺乏调用和响应，用户感到非常沮丧，不确定如何解决该身份验证问题。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **通过自定义劫持 SFT 数据集**：一位成员通过自定义 **message_transform** 和 **model_transform** 参数，成功“劫持”了内置的 **SFT Dataset**。
   - 这允许根据需要调整格式，正如该成员所说，“我只需要劫持 message/model transforms 以适应我的需求”。
- **DPO Seed 问题困扰训练脚本**：成员们正在排查为什么 `seed` 在 LoRA/全量微调中有效，但在 LoRA/全量 **DPO** 中无效，导致相同配置下出现不同的 Loss 曲线。
   - 有人担心 `seed=0` 和 `seed=null` 会影响 **DistributedSampler** 调用中的随机性，可能需要针对 DPO/PPO 脚本中的梯度累积进行相关修复；参见 [issue 2334](https://github.com/pytorch/torchtune/issues/2334) 和 [issue 2335](https://github.com/pytorch/torchtune/issues/2335)。
- **Ladder-residual 提升模型速度**：一条推文介绍了 [Ladder-residual](https://x.com/zhang_muru/status/1886870194443968529)，这是一种改进，可将张量并行下的 **70B Llama** 速度提高约 **30%**。
   - 这一增强反映了多位作者和研究人员在模型架构协作方面的持续优化。
- **LLM 数据增强调研**：最近的一项调查分析了 **数据增强** 在 **大语言模型** (**LLMs**) 中的作用，强调了它们需要广泛的数据集以避免过拟合；参见 [论文](https://arxiv.org/abs/2501.18845)。
   - 论文讨论了 **独特的提示词模板** 和 **基于检索的技术**，这些技术通过外部知识增强了 LLM 的能力，从而获得更多 **grounded-truth data**。
- **R1-V 项目变革学习方式**：分享了关于 **R1-V** 项目的激动人心消息，该项目利用带有可验证奖励的 **强化学习** 来增强模型的计数能力；参见 [Liang Chen 的推文](https://x.com/liangchen5518/status/1886171667522842856?s=46&t=b1X88nwMsmZgHkmMFkiG3g)。
   - 该项目展示了一个仅需 **100 个训练步数**、成本低于 **$3** 即可超越 **72B** 对应模型的模型，并承诺完全 **开源**，激发了社区的兴趣。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **社区展示移至论坛**：**Community Showcase** 已移至 [Modular 论坛](https://forum.modular.com/c/community-showcase/8) 以优化组织管理，之前的展示区现已设为只读。
   - 此次过渡旨在简化 **Modular (Mojo 🔥)** 生态系统内的社区互动和项目共享。
- **Rust 寻求热重载方案**：成员们正在讨论 Rust 通常如何使用 **C ABI** 进行 **hot reloading**（热重载），但这在 Rust 更新和 ABI 稳定性方面面临挑战。
   - *Owen* 询问了关于构建玩具级 ABI 的资源，强调了由于数据结构频繁变化，ABI 稳定性的重要性。
- **Mojo 探索编译时特性**：一位用户询问 Mojo 是否具有类似于 Rust 的 `#[cfg(feature = "foo")]` 特性，引发了关于 Mojo 中 **compile-time programming**（编译时编程）能力以及稳定 ABI 重要性的讨论。
   - 对话强调，只有少数语言能维持稳定的 ABI，这对于兼容性至关重要。
- **Python Asyncio 循环解析**：关于 Python **asyncio** 的讨论显示，它支持社区驱动的事件循环，并引用了 [GitHub](https://github.com/MagicStack/uvloop) 上的 **uvloop**。
   - 参与者将其与 Mojo 的线程和内存管理方法进行了对比，指出了潜在的**障碍**。
- **异步 API 面临线程安全审查**：针对异步 API 的 **thread safety**（线程安全）提出了担忧，重点关注潜在的可变特性以及安全内存处理的必要性。
   - 讨论强调，许多当前的方法缺乏对内存分配的控制，这可能会导致复杂化。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Weston 教授 LLM 自我提升**：[Jason Weston](https://www.youtube.com/live/_MNlLhU33H0) 进行了题为“*Learning to Self-Improve & Reason with LLMs*”的讲座，重点介绍了提升 LLM 性能的创新方法，如 [Iterative DPO](https://arxiv.org/abs/2312.16682)、[Self-Rewarding LLMs](https://arxiv.org/abs/2401.10020) 和 [Thinking LLMs](https://arxiv.org/abs/2410.10630)。
   - 讲座强调了有效的推理和任务相关学习机制，旨在提高 LLM 在各种任务中的能力。
- **黑客松获胜者名单公布**：黑客松获胜者已收到私下通知，预计下周进行公开宣布。
   - 成员们正热切期待关于黑客松结果的更多细节。
- **MOOC 证书延迟发放**：**秋季项目证书**尚未发布，但很快就会提供。
   - 官方感谢参与者在 MOOC 证书发放期间的耐心等待。
- **研究项目备受关注**：成员们表达了参与**研究项目**的兴趣。
   - 据工作人员称，有关研究机会和团队配对的更多细节将很快提供。
- **签到表仅限伯克利学生**：提到的签到表**仅供伯克利学生使用**。
   - 针对非伯克利学生使用签到表的可访问性提出了担忧，因为目前缺乏针对非伯克利学生的信息。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **旧版 ERP 集成寻求 VBS 帮助**：一位用户正在寻求关于使用 **.vbs 脚本**与**旧版 ERP 系统**集成的服务器方面的帮助。
   - 一位成员建议使用 **mcpdotnet**，因为它可能会简化从 **.NET** 的调用。
- **Cursor MCP Server 获得 Docker 指导**：一位新用户请求关于在 **Cursor** 内部本地运行 **MCP server** 的指导，并特别关注使用 **Docker container**。
   - 成员建议将与 **supergateway** 一起使用的 **SSE URL** 输入到 Cursor MCP SSE 设置中以解决该问题。
- **企业级 MCP 协议进展**：围绕 **MCP protocol** 的讨论强调了 **OAuth 2.1** 授权草案，可能与 **IAM** 系统集成。
   - 会中指出，由于正在进行内部测试和原型设计，目前的 SDK 缺乏授权支持。
- **Localhost CORS 问题困扰 Windows**：一位用户在 **localhost** 上运行其 MCP server 时遇到连接问题，怀疑是 **CORS 相关**问题。
   - 他们计划使用 **ngrok** 来规避在 **Windows** 上通过 localhost 访问服务器相关的潜在通信问题。
- **ngrok 解决 Localhost 访问问题**：一位成员推荐使用 **ngrok** 来评估服务器的可访问性，建议使用命令 `ngrok http 8001`。
   - 他们强调这可以解决由于尝试通过 localhost 访问服务器而产生的问题。

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command-R+ 的内部思考功能令用户印象深刻**：用户对 **Command-R+** 模型展示**内部思考 (internal thoughts)** 和**逻辑步骤**的能力感到满意，其运作方式类似于 Chain of Thought。
   - 尽管新模型层出不穷令人兴奋，但一位用户指出，在持续使用数月后，**Command-R+** 依然能给他们带来惊喜。
- **Cohere 在关税担忧中捍卫加拿大 AI**：一位成员选择 **Cohere** 是为了增强加拿大的 AI 能力，特别是在面临潜在的美国关税背景下。
   - 他们赞赏在充满挑战的经济环境下，仍有能够维持当地 AI 发展的选项。
- **Cohere 的 Rerank 3.5 提升金融语义搜索**：Cohere 和 Pinecone 举办了一场**网络研讨会 (webinar)**，强调了**金融语义搜索和重排序 (Reranking)** 的优势。
   - 研讨会展示了 Cohere 的 **Rerank 3.5** 模型及其利用金融数据增强整体搜索性能的潜力。
- **旨在优化技术内容体验的调查**：应届毕业生正在进行一项调查，以收集技术爱好者对内容消费偏好的见解，旨在提高用户参与度。
   - 该调查可在 [User Survey](https://forms.gle/y9PL1YByWKsMMRQLA) 参与，探讨了从 [Tech Blogs](https://producthunt.com) 和 [Research Updates](https://scholar.google.com) 到**社区论坛**和 **AI 工具**等各种信息来源。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 目录引起关注**：一位用户询问名为 **dspy.py** 的文件或名为 **dspy** 的目录是否会导致问题，因为 Python 有时难以处理这种命名设置。
   - 这个问题引发了对潜在文件处理冲突的担忧，可能会影响 **DSPy** 项目的执行。
- **Image Pipeline 在 DSPy 2.6.2 中崩溃**：**dspy 2.6.2** 中的一个 **Image pipeline** 触发了 **ContextWindowExceededError**，意味着由于 Token 限制而“超出上下文”，而之前的 **2.6.1** 版本虽然存在正在调查的错误，但可以运行。
   - 用户报告称，这种退化可能是由 **DSPy** 最近的更改引起的。
- **DSPy 2.6.4 中断言功能被移除**：成员们宣布，在即将发布的 **2.6.4** 版本中，**断言 (assertions)** 将被替换，这表明 **DSPy** 处理错误的方式发生了转变。
   - 这一变化意味着 **DSPy** 内部的错误处理和逻辑检查将采用与旧版本不同的方式。
- **Databricks 可观测性探索**：一位在 **Databricks notebooks** 中运行 **DSPy 2.5.43** 进行 NER 和分类的用户正在寻求实现**结构化输出 (structured output)** 的指导。
   - 由于配置 LM 服务器受限，他们必须使用当前版本，这增加了涉及优化器和嵌套 JSON 输出任务的复杂性。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **OpenEuroLLM 带着欧盟特色启动**：[OpenEuroLLM](https://openeurollm.eu/) 作为第一个涵盖所有欧盟语言的开源大语言模型家族被推出，获得了卓越的 STEP 标志，并专注于社区参与。
   - 该项目旨在符合欧盟法规并保留**语言多样性 (linguistic diversity)**，与 **LAION** 等开源和开放科学社区保持一致。
- **欧盟 AI 努力面临质疑**：在关于欧盟法规下 AI 未来的讨论中，一位成员开玩笑地建议在 **2030** 年再来看看欧盟 AI 努力的成果。
   - 这一评论凸显了对当前 AI 开发工作能否立即产生实质性成果的怀疑。
- **社区关注模因币狂热**：一位成员调查了社区对**模因币 (meme coins)** 的兴趣，寻求其他人的广泛参与。
   - 他们主动征求任何对该话题感兴趣的人表达意向。

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **DocumentContextExtractor 增强了 RAG**：**DocumentContextExtractor** 是一个旨在提高 **RAG** 准确性的迭代版本，由 [AnthropicAI](https://twitter.com/llama_index/status/1886522064292733288) 和 [llama_index](https://t.co/qoVrgd0ddy) 作为 demo 实现。
   - 该技术有望提升性能，对于从事检索增强生成（Retrieval-Augmented Generation）的研究人员来说，这是一个重要的探索领域。
- **Contextual Retrieval 改变了游戏规则**：**Contextual Retrieval** 的使用被强调为提高 **RAG** 系统响应准确性的关键。
   - 该技术优化了文档检索过程中对上下文的利用方式，从而促进了更深层次的交互。
- **LlamaIndex LLM 类面临超时问题**：一位用户询问如何在默认的 **LlamaIndex LLM** 类中实现 **timeout** 功能，并指出 OpenAI 的 API 中提供了该功能。
   - 另一位成员建议 **timeout** 选项可能应该放在 client kwargs 中，并参考了 [LlamaIndex GitHub 仓库](https://github.com/run-llama/llama_index/blob/7391f302e18542c68b9cf5025afb510af4a52324/llama-index-integrations/llms/llama-index-llms-azure-inference/llama_index/llms/azure_inference/base.py#L224)。
- **为 LlamaIndex 探索 UI 解决方案**：一位成员对其他人与 **LlamaIndex** 配合使用的 **UI** 解决方案表示好奇，询问大家是选择从零开始构建还是有其他方案。
   - 该询问仍处于开放状态，邀请其他人分享与 LlamaIndex 相关的 **用户界面（User Interface）** 实践和偏好。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinybox 面临欧元区运输限制**：支持团队确认，**tinybox (red)** 无法运送到部分欧元区国家。
   - 尝试订购到 **爱沙尼亚（Estonia）** 等国家的海外用户目前无法收到货物，因为这些国家未列在结账时的下拉菜单中。
- **巧妙的运输服务变通方法出现**：一位用户建议使用 **Eurosender** 等服务来绕过运输限制。
   - 他们确认通过这种方法成功送达 **德国（Germany）**，为不支持地区的 tinybox 聊天频道用户提供了一个解决方案。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Iceberg 管理噩梦**：一场名为 **Pain in the Ice: What's Going Wrong with My Hosted Iceberg?!** 的小组讨论将于 2 月 6 日探讨管理 **Iceberg** 的复杂性，演讲者包括 **Yingjun Wu**、**Alex Merced** 和 **Roy Hasson**（[Meetup 链接](https://www.meetup.com/streaming-stories/events/305886042/)）。
   - 由于摄取（ingestion）、压缩（compaction）和 **RBAC** 等问题，*管理 Iceberg 可能会变成一场噩梦*，从而分散处理其他任务的资源。该小组旨在探讨该领域的创新如何简化 Iceberg 的管理和使用。
- **盲目推崇 LLM 令人沮丧**：成员们对那些在任何问题上都推崇 **LLM** 的 AI 工程师表示担忧，即使 **unsupervised learning** 或其他更简单的方法可能更合适。
   - 讨论强调了一种趋势，即在不考虑问题性质的情况下选择工具，从而削弱了简单方法的价值。
- **TF-IDF + Logistic Regression 胜出**：一位成员分享了一个成功案例，在对数百万个文本样本进行分类时，他成功主张使用 **TF-IDF + Logistic Regression** 而非 OpenAI 模型。
   - **Logistic Regression** 模型表现良好，证明了简单算法也可以非常有效，从而展示了传统方法的效力。



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 项目停滞了？**：成员们对 **Open Interpreter** 项目缺乏更新表示担忧，指出 GitHub 上的 **pull requests** 自上次重大提交以来已闲置数月。
   - Discord 频道中的沉默让渴望参与其中的贡献者感到*沮丧*。
- **Open Interpreter 文档缺失**：一位成员强调 1.0 版本缺少文档，特别是关于如何利用 **profiles.py** 等组件的部分。
   - 文档的缺失让用户对项目当前的重点以及对其功能的后续支持产生了疑问。
- **DeepSeek r1 集成仍是个谜**：有人询问如何将 **DeepSeek r1** 集成到 Open Interpreter 环境中，但未得到回应。
   - 社区讨论的缺乏表明，关于此集成的实验或知识共享可能存在空白。



---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Cloudflare 与 OpenRouter 联手**：**Cloudflare** 现在正式成为 **OpenRouter** 上的提供商，集成了其 **Workers AI** 平台和 **Gemma** 模型，为 AI 应用开发者开放了多种开源工具。
   - 此次合作旨在增强 **OpenRouter** 生态系统，为开发者提供更广泛的 AI 工具。
- **Gemma 7B-IT 新增 Tool Calling**：现在可以通过 **Cloudflare** 使用 **Gemma 7B-IT** 模型，该模型具备 **tool calling capabilities**（工具调用能力），旨在提高开发效率。
   - 鼓励开发者探索 **Gemma 7B-IT**，以便在应用中实现更快速、更精简的工具集成；可通过 [OpenRouter](https://openrouter.ai/google/gemma-7b-it) 获取。
- **Llama 模型涌入 OpenRouter**：**OpenRouter** 现在支持一系列 **Llama 模型**，包括 [Gemma 7B-IT](https://openrouter.ai/google/gemma-7b-it)，为用户项目提供了众多选择。
   - AI 开发者可以通过 Discord 请求特定的 **Llama 模型**。
- **模型错误显示变得更具体**：解决了导致错误混淆的显示问题，现在**错误消息中会显示模型名称**，以提高用户的清晰度。
   - 此次更新旨在通过提供更清晰的错误反馈来改善用户体验。

---

**Axolotl AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将移除它。

---

**Mozilla AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将移除它。

---

**HuggingFace Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将移除它。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将移除它。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将移除它。

---

# PART 2: 详细频道摘要与链接

{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1336068657457008722)** (692 messages🔥🔥🔥): 

> `DeepSeek R1 Model, Model Quantization, Fine-Tuning Techniques, Data Generation for Training, Transformer Architectures` 

- **DeepSeek R1 动态量化探索**：DeepSeek R1 模型已被量化至 1.58 bits，体积从 720GB **缩减了 80%** 至 131GB，在保持功能的同时更易于本地用户使用。
   - 这种量化涉及对某些层选择性地应用更高 bits，同时避免朴素的全量化（这会导致输出乱码等问题）。
- **LLM 微调策略**：参与者讨论了微调语言模型的各种技术，强调减少数据集规模并专注于高质量样本可以改善训练结果。
   - 对话还涉及使用模型生成合成数据集，以及调整损失函数以有效处理类别不平衡。
- **大语言模型的挑战**：指出了从微调模型中提取最佳推理能力的困难，以及关于本质上控制模型能力这一想法的模糊性。
   - 有人建议采用降低学习率或 Lora alpha 调整的方法，尽管对其有效性看法不一。
- **运行与测试模型**：用户询问了运行 DeepSeek R1 Q4_K_M 所需的硬件，估计显示需要大量的 RAM 和 VRAM 资源才能进行高效处理。
   - 共享了相关资源，包括运行模型的指南以及利用现有高性能设置进行微调和推理的见解。
- **评估 Transformer 架构**：对各种 Transformer 和注意力机制架构的比较得出了有趣的结果，并讨论了模型间的参数数量和设计差异。
   - 特别关注了 Differential Transformers 的能力，并提供了详细仓库和资源的链接以供进一步探索。

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://docs.vllm.ai/en/latest/getting_started/examples/examples_index.html">Examples &#8212; vLLM</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/peft/en/conceptual_guides/prompting">Soft prompts</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2501.19393">s1: Simple test-time scaling</a>: Test-time scaling 是一种极具前景的新型语言建模方法，它通过额外的测试时计算来提升性能。最近，OpenAI 的 o1 模型展示了这种能力，但尚未公开...</li><li><a href="https://www.datacamp.com/tutorial/fine-tuning-deepseek-r1-reasoning-model">Fine-Tuning DeepSeek R1 (Reasoning Model)</a>: 在医疗思维链数据集上微调全球首个开源推理模型，以构建未来的 AI 医生。</li><li><a href="https://bsky.app/profile/xyratech.bsky.social/post/3lh7tfginu224">Xyra (@xyratech.bsky.social)</a>: 好吧，速度极其缓慢（每 2 分钟输出一个 token），但我刚刚在配备 36 GB RAM 的 MacBook Pro M3 上成功运行了 DeepSeek 的 R1 671B 模型（动态量化至 2.51-bit）。</li><li><a href="https://x.com/Marktechpost/status/1886874013303235064">Tweet from Marktechpost AI Research News ⚡ (@Marktechpost)</a>: 为 Python 代码微调 Llama 3.2 3B Instruct：使用 Unsloth 的全面指南（包含 Colab Notebook）。在本教程中，我们将逐步介绍如何设置并对 Llama 3 进行微调...</li><li><a href="https://bsky.app/profile/xyratech.bsky.social/post/3lh7tfhfipc24">Xyra (@xyratech.bsky.social)</a>: 许多 “DeepSeek” 模型（如 deepseek-r1:8b）都是蒸馏版本，也就是说，它们实际上是为了模仿 R1 而训练的 Llama 或 Qwen 模型。然而，这是原始模型，但动态...</li><li><a href="https://runpod.io?ref=bb842lb3">RunPod - The Cloud Built for AI</a>: 在同一个云端开发、训练和扩展 AI 模型。通过 GPU Cloud 快速启动按需 GPU，通过 Serverless 扩展 ML 推理。</li><li><a href="https://unsloth.ai/blog/deepseekr1-dynamic">Run DeepSeek-R1 Dynamic 1.58-bit</a>: DeepSeek R-1 是最强大的开源推理模型，其性能与 OpenAI 的 o1 模型相当。运行由 Unsloth 提供的 1.58-bit Dynamic GGUF 版本。</li><li><a href="https://gist.github.com/lucasmrdt/4215e483257e1d81e44842eddb8cc1b3">Prompt to leak every LLM system prompt including cursor.com, v0.dev, claude.ai, chatgpt.com, perplexity.ai</a>: 泄露所有 LLM 系统提示词的 Prompt，包括 cursor.com, v0.dev, claude.ai, chatgpt.com, perplexity.ai - LEAK_EVERY_LLM_SYSTEM_PROMPT.md</li><li><a href="https://docs.unsloth.ai/basics/vision-fine-tuning">Vision Fine-tuning | Unsloth Documentation</a>: 关于使用 Unsloth 进行视觉/多模态微调的详细信息</li><li><a href="https://x.com/ylecun/status/1639690596364308482">Tweet from Yann LeCun (@ylecun)</a>: @nisyron 7 个轴均匀分布在一个圆周上。每个轴上放置一个齿轮，使得每个齿轮都与其左侧和右侧的齿轮啮合。齿轮按 1 到 7 编号...</li><li><a href="https://github.com/unslothai/unsloth/issues/1561">[Fixing] More finetuning support · Issue #1561 · unslothai/unsloth</a>: 支持 Gemma 等模型的序列分类 Flex Attention；变量序列长度及自动取消填充/填充；Tool Calling 重构并合并 xformers, SDPA, flash-attn, flex-attention</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: 以下是我们所有 notebook 的列表：</li><li><a href="https://github.com/micros">micros - Overview</a>: micros 有 8 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/simplescaling/s1">GitHub - simplescaling/s1: s1: Simple test-time scaling</a>: s1: Simple test-time scaling。通过在 GitHub 上创建账号为 simplescaling/s1 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth/issues/267?">Batch inference produces nonsense results for unsloth/mistral-7b-instruct-v0.2-bnb-4bit · Issue #267 · unslothai/unsloth</a>: 你好，在使用以下代码加载模型后：from unsloth import FastLanguageModel import torch model, tokenizer = FastLanguageModel.from_pretrained( model_name = &quot;unsloth/mistral-7b-instruct-v0.2-bnb...</li><li><a href="https://github.com/huggingface/open-r1/blob/main/src/open_r1/grpo.py">open-r1/src/open_r1/grpo.py at main · huggingface/open-r1</a>: DeepSeek-R1 的完全开源复现。通过在 GitHub 上创建账号为 huggingface/open-r1 的开发做出贡献。</li><li><a href="https://github.com/Datta0/nanoformer">GitHub - Datta0/nanoformer: A small repo to experiment with Transformer (and more) architectures.</a>: 一个用于实验 Transformer（及更多）架构的小型仓库。 - Datta0/nanoformer</li>

><li><a href="https://github.com/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb">text_classification_scripts/unsloth_classification.ipynb at main · timothelaborie/text_classification_scripts</a>: 使用 Llama 和 BERT 进行文本分类的脚本 - timothelaborie/text_classification_scripts</li><li><a href="https://github.com/microsoft/unilm/blob/master/Diff-Transformer/multihead_diffattn.py#L50>">unilm/Diff-Transformer/multihead_diffattn.py at master · microsoft/unilm</a>: 跨任务、语言和模态的大规模自监督预训练 - microsoft/unilm</li><li><a href="https://github.com/CoffeeVampir3/Tiny-Differential-Tensor-Product-Mixer/blob/master/models/diff_attn.py#L83>">Tiny-Differential-Tensor-Product-Mixer/models/diff_attn.py at master · CoffeeVampir3/Tiny-Differential-Tensor-Product-Mixer</a>: 通过在 GitHub 上创建账户来为 Tiny-Differential-Tensor-Product-Mixer 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/)** (1 messages): 

shiyaozhidewa: 我想要 DeepSeek R1 abliterated 671B
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1336064810974052524)** (96 messages🔥🔥): 

> `CUDA Out of Memory Errors, Finetuning Strategies for Models, Installation Instructions for Unsloth, Using Experts in MoE Frameworks, Logging with Weights & Biases` 


- **管理 CUDA Out of Memory 错误**：用户在显存（VRAM）有限的 GPU 上微调大模型时遇到了与显存溢出相关的 CUDA 驱动错误，特别指出即使在 batch size 为 1 且训练步数为 8,000 时也会发生。
   - 建议包括减少 context length 并验证是否分配了足够的资源，以及可能使用原生的 llama.cpp 来缓解该问题。
- **选择微调模型**：对于代码分类任务，建议微调在代码上预训练的模型，以利用特定领域的知识，从而比文本分类模型获得更好的性能和效率。
   - 一种有效的策略是使用在代码上预训练的 Causal LM，同时为序列分类进行必要的修改。
- **Unsloth 安装说明**：用户询问了在 Windows 上安装 Unsloth 后的步骤，特别是遵循文档中列出的第三种方法。
   - 建议从 Unsloth GitHub 下载 notebooks，并根据需要自定义参数以简化实现。
- **MoE 框架与专家配置**：在关于使用 MoE 框架的讨论中，用户对专家的配置表示困惑，建议范围从默认的 8 个到最多 256 个。
   - 会议澄清了专家的数量在模型运行期间通常应保持静态，相关的研讨会旨在阐明这一方面。
- **Weights & Biases 项目日志记录**：用户寻求在 Unsloth 中配置 Weights & Biases 日志设置的方法，强调需要指定项目以进行适当的监控。
   - 建议通过 HF trainer 中的参数来有效地设置日志记录项目（logging project）。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing#scrollTo=AqkY_wHdKyOl>">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-updating/windows-installation">Windows Installation | Unsloth Documentation</a>: 了解如何在有或没有 WSL 的情况下在 Windows 上安装 Unsloth。</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-updating/windows-installation#method-2-windows-using-powershell">Windows Installation | Unsloth Documentation</a>: 了解如何在有或没有 WSL 的情况下在 Windows 上安装 Unsloth。</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: 以下是我们所有 notebooks 的列表：
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1336068006660542524)** (11 messages🔥): 

> `DeepSeek R1 model, Klarity library, YouTube video releases, Bulgarian language model performance, Math versions of models` 


- **DeepSeek R1 在 M3 上取得成功！**：一位用户在配备 **36GB** RAM 的 **MacBook Pro M3** 上成功运行了 **DeepSeek R1** unsloth 模型，展示了其强大的能力。
   - 这一成就凸显了该模型在消费级硬件上的效率。
- **Klarity 库发布！**：一个新的开源库 **Klarity** 已发布，用于分析语言模型输出的熵（entropy），从而更好地洞察决策过程。
   - 该库提供详细的 JSON 报告，并开放给 unsloth 量化模型进行测试；[点击此处查看](https://github.com/klara-research/klarity)。
- **YouTube 视频展示 DeepSeek 能力**：成员们分享了两个 **YouTube** 视频，演示了 **DeepSeek R1** 模型的能力，并展示了其相对于竞争对手的性能。
   - 视频包括 *'deepseek-R1 the new king'* [链接](https://youtu.be/AFEzuOGOSOQ?si=A6iOZL2Hri84P0QA) 和 *'build your dream app'* [链接](https://youtu.be/WBfUPaiAAQE?si=Hmf1hAUQiXFlVYVq)。
- **令人印象深刻的保加利亚语模型性能**：一位用户分享了其保加利亚语模型与基础模型的困惑度（perplexity）分数对比，报告了显著的改进（短文本 *PPL: 72.63* 对比 *179.76*）。
   - 困惑度的如此降低表明该模型在处理该语言方面具有强大的性能。
- **征集模型的数学版本**：一位成员表示有兴趣获取模型的数学版本，特别是针对 **Qwen 2.5** 的版本以及用于实验的数据集。
   - 关于这一话题的讨论凸显了社区对更详细的模型分析工具的持续追求。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://bsky.app/profile/xyratech.bsky.social/post/3lh7tfginu224">Xyra (@xyratech.bsky.social)</a>: OK, it&#39;s INCREDIBLY slow (a token output every 2 minutes), but I just got DeepSeek’s R1 671B model (dynamic quantised to 2.51-bit) running on a MacBook Pro M3 with 36 GB of RAM.</li><li><a href="https://youtu.be/WBfUPaiAAQE?si=Hmf1hAUQiXFlVYVq">With zero coding skills build you dream app. deepseek-r1🐋 + roo-cline + FREE apis  #deepseek-v3</a>: @deepseek_v3 @ai @cline @roo-cline @app @freehttps://app.hyperbolic.xyz/https://fireworks.ai/models/fireworks/deepseek-r1https://glhf.chat</li><li><a href="https://youtu.be/AFEzuOGOSOQ?si=A6iOZL2Hri84P0QA">deepseek-R1 the new king and fully FREE (beats claude 3.5 sonnet &amp; O1) (tested)</a>: @ai @deepseek @viral @agi @chatgpt</li><li><a href="https://github.com/klara-research/klarity">GitHub - klara-research/klarity: See Through Your Models</a>: See Through Your Models. Contribute to klara-research/klarity development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/)** (1 messages): 

not_qty: 这个视频很棒...
https://youtu.be/_1f-o0nqpEI?si=s2B-o5y2d5ztsV0U
  

---


### **Codeium (Windsurf) ▷ #[content](https://discord.com/channels/1027685395649015980/1092566563862884412/1336390213882216491)** (1 messages): 

> `Windsurf Docs Shortcuts, Mintlify Auto-hosting, Community Contributions on Twitter` 


- **Windsurf 寻求新的文档快捷方式**：一位成员宣布他们正在为 [Windsurf](https://x.com/kevinhou22/status/1886827501004931511) 收集新的 `@docs` 快捷方式列表，并鼓励大家贡献或投票。
   - *“我们热爱文档！”* 强调了他们致力于提升文档体验的承诺。
- **感谢 Mintlify 提供节省时间的解决方案**：向 **Mintlify** 致谢，因为它通过 `/llms.txt` 自动托管所有文档，通过避免 HTML 解析为 Agent 节省了时间和 **tokens**。
   - 这种方法可以实现更高效的文档处理，确保更快速地访问资源。



**Link mentioned**: <a href="https://x.com/kevinhou22/status/1886827501004931511">Tweet from Kevin Hou (@kevinhou22)</a>: we love docs! 📖 I&#39;m working on improving / adding more @ docs shortcuts to @windsurf_ailmk what you want and I&#39;ll add as many as I can... 🧵also shoutout @mintlify for auto-hosting all docs w...

  

---

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1336066550179958855)** (68 条消息🔥🔥): 

> `Windsurf 功能、Codeium 性能与错误、账户激活问题、Hackathon 机会、对 Qodo 的质疑` 


- **Windsurf 和 Codeium 功能**：有用户询问在没有 Windsurf 的情况下，Tab 功能和多行编辑等特定功能是否能在 Codeium 中正常工作，并指出 Command 功能的表现并不一致。
   - 一位成员表示 Supercomplete 在 VSCode 中被暂时禁用，这引发了对当前功能的困惑。
- **对 Codeium 性能的担忧**：一位成员报告称 **Claude** 的工具调用（tool utilization）功能表现不佳，导致因反复尝试失败而消耗了过多的额度（credits）。
   - 其他成员指出，如果工具产生错误，则不应扣除额度，并建议检查工具的有效性。
- **账户激活问题**：多位成员反映在 VSCode 上登录账户时遇到问题，特别是与内部证书错误（internal certificate errors）相关的问题。
   - 建议包括联系支持团队以及尝试通过不同网络进行激活，但均未成功。
- **CreatorsCorner 的 Hackathon 邀请**：宣布了一项总奖金超过 **3.5 万美元** 的合作 Hackathon，鼓励参与者构建自主 AI Agent。
   - 参与者将展示他们的项目，旨在通过 AI 技术增强用户能力。
- **Qodo 的正当性受到质疑**：一位成员对 Qodo（前身为 Codium）提出了质疑，怀疑其可靠性以及是否可能是一个骗局。
   - 社区反馈褒贬不一，一些人表示谨慎，并对使用 Qodo 不感兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lu.ma/fyu8iqnk">Multimodal AI Agents - Hackathon · Luma</a>: Gen AI AgentsCreatorsCorner 与 Google Deepmind, Weights &amp; Biases, Together.ai, Stytch, Senso, LlamaIndex 等合作，热情地……</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: 需要帮助？联系我们的支持团队以获取个性化协助。</li><li><a href="https://github.com/livingstonlarus/runic">GitHub - livingstonlarus/runic: 一个开源框架，通过长期记忆 (LTM) 和检索增强生成 (RAG) 增强大语言模型 (LLMs)。非常适合 AI 编程助手和其他应用，它使 LLM 能够保留上下文、随时间适应并访问最新信息，确保更智能且具备上下文感知能力的交互。</a>: An open-source framework that enhances Large Language Models (LLMs) with Long-Term Memory (LTM) and Retrieval-Augmented Generation (RAG). Ideal for AI coding assistants and other applications, it e...
</li>
</ul>

</div>
  

---

### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1336063710078439455)** (401 条消息🔥🔥): 

> `Windsurf O3 Mini 定价、文件编辑问题、模型用户体验、Context Window 限制、新功能集成` 


- **Windsurf O3 Mini 定价关注**：用户正在讨论考虑到 O3 Mini 较低的成本和性能表现，其定价是否应与 Claude 3.5 Sonnet 相似。
   - 有用户担心 Windsurf 消耗了大量额度且在执行任务时表现挣扎，因此要求更公平的定价。
- **无法编辑文件**：许多用户遇到 Windsurf 拒绝修改或更新文件的问题，通常会导致内部错误。
   - 报告了具体的错误，例如无法读取内容以及无效的 service worker 注册。
- **不同模型的用户体验**：用户强调了对各种模型性能的挫败感，尽管尝试了 O3 Mini 和 DeepSeek 等替代方案，但仍表示更倾向于 Claude。
   - 一些用户指出模型的 Context Window 似乎有限，影响了其在应用中的表现。
- **Context Window 和 Tool Call 限制**：用户反馈称，当对话长度超过 Context 大小时需要更清晰的警告，同时也提到了 Tool Call 失败的问题。
   - 用户希望改进对 Context 限制和工具功能的处理及沟通。
- **新功能和规则的集成**：用户正在探索如何有效地为全栈 Web 应用的不同方面创建和管理 .windsurfrules 文件。
   - 关于规则管理的建议包括建立一个引导系统，以便利用 Windsurf 促进项目组织。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.codeium.com/windsurf/advanced">Windsurf - Advanced</a>: 未找到描述</li><li><a href="https://docs.codeium.com/windsurf/usage#tool-calls">Paid Plan and Credit Usage - Codeium Docs</a>: 未找到描述</li><li><a href="https://x.com/kevinhou22/status/1886827501004931511">Tweet from Kevin Hou (@kevinhou22)</a>: 我们热爱文档！📖 我正在努力改进/添加更多 @ docs 快捷方式到 @windsurf_ai，告诉我你想要什么，我会尽可能多地添加... 🧵另外感谢 @mintlify 自动托管所有文档...</li><li><a href="https://github.com/Exafunction/codeium/issues/111">Windsurf Server Installation Fails on 32-bit ARM (armv7l) Raspberry Pi · Issue #111 · Exafunction/codeium</a>: 标题：Windsurf 服务器在 32 位 ARM (armv7l) Raspberry Pi 上安装失败。环境详情：设备：Raspberry Pi OS: Raspbian GNU/Linux 11 (bullseye) 架构：armv7l (32-bit ARM) 内核...</li><li><a href="https://status.codeium.com">Codeium Status</a>: 未找到描述</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: 需要帮助？联系我们的支持团队以获取个性化协助。</li><li><a href="https://www.swebench.com/#verified">SWE-bench</a>: 未找到描述</li><li><a href="https://codeium.com/pricing">Pricing | Windsurf Editor and Codeium extensions</a>: Codeium 对个人用户永久免费。团队可以通过我们的企业版产品进行升级，以获得增强的个性化和灵活的部署。</li><li><a href="https://codeberg.org/KhazAkar/canbus_visualizer">canbus_visualizer</a>: 我在编程工作中使用 Windsurf AI IDE 的实验</li><li><a href="https://github.com/ZarK/ai-rules">GitHub - ZarK/ai-rules</a>: 通过创建账号为 ZarK/ai-rules 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1336063765120028682)** (339 messages🔥🔥): 

> `O1 Pro 性能, Aider 中的弱模型, OpenRouter 对比直接 API, 使用 Shell 作为 LLM 工具, 测试与重构的挑战` 


- **O1 Pro 性能**: 用户报告了使用 O1 Pro 带来的显著性能提升，有人在不到五分钟内快速生成了大量代码。
   - 与 O3 Mini 的对比突显了 O1 Pro 更快的响应速度和更好的复杂任务处理能力。
- **Aider 中的弱模型 (Weak Models)**: 弱模型通常用于生成 commit messages 和总结聊天内容，在特定任务上比强模型更具成本效益且更高效。
   - DeepSeek V3 和其他替代方案被讨论为寻求预算友好且有效模型的用户的可行选择。
- **OpenRouter 对比直接 API**: 用户讨论了使用 OpenRouter 的好处，理由是更好的正常运行时间（uptime）以及能够优先选择某些提供商，而不是直接访问 API。
   - CentMl 和 Fireworks 被提及为有效的 DeepSeek 提供商，尽管它们的速度有时不尽如人意。
- **使用 Shell 作为 LLM 工具**: 人们对利用 Shell 工具作为 LLM 交互平台表现出兴趣，这借鉴了 Claude 和 LLM Functions 等工具的成功实验。
   - 该概念包括将 HTTP API 功能封装到 CLI 工具中，以实现更流线型的操作。
- **测试与重构的挑战**: 重构后彻底检查测试是一个常见的难题，用户正在探索简化这一过程的方法，且不产生过多的开销。
   - 参与总结用户请求以获得更清晰的上下文，可以提高在长对话中使用 Aider 的效率。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/toby-cry-phone-spider-man-cry-phone-spider-man-phone-toby-phone-gif-12875606672124040541">Toby Cry Phone Spider Man Cry Phone GIF - Toby Cry Phone Spider man Cry Phone Spider man phone - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/i-was-thinking-the-same-thing-abed-nadir-community-i-had-the-same-thought-gif-1344064406870600973">I Was Thinking The Same Thing Abed Nadir GIF - I was thinking the same thing Abed nadir Community - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://aider.chat/docs/llms/ollama.html#setting-the-context-window-size">Ollama</a>: aider 是你终端里的 AI 结对编程工具</li><li><a href="https://aider.chat/docs/llms/openai-compat.html">OpenAI compatible APIs</a>: aider 是你终端里的 AI 结对编程工具</li><li><a href="https://aider.chat/docs/config/reasoning.html">Reasoning models</a>: 如何配置来自次要提供商的推理模型设置。</li><li><a href="https://aider.chat/docs/llms/openrouter.html#controlling-provider-selection">OpenRouter</a>: aider 是你终端里的 AI 结对编程工具</li><li><a href="https://github.com/quarkiverse/quarkus-mcp-servers/blob/main/README.md">quarkus-mcp-servers/README.md at main · quarkiverse/quarkus-mcp-servers</a>: Quarkus 中的 Model Context Protocol 服务器。通过在 GitHub 上创建账号为 quarkiverse/quarkus-mcp-servers 做出贡献。</li><li><a href="https://github.com/vivekVells/mcp-pandoc">GitHub - vivekVells/mcp-pandoc: MCP server for document format conversion using pandoc.</a>: 使用 pandoc 进行文档格式转换的 MCP 服务器。 - vivekVells/mcp-pandoc</li><li><a href="https://glama.ai/mcp/servers">Open-Source MCP servers</a>: 企业级安全、隐私，具备 Agent、MCP、提示词模板等功能。</li><li><a href="https://github.com/superagent-ai/reag">GitHub - superagent-ai/reag: Reasoning Augmented Generation</a>: 推理增强生成。通过在 GitHub 上创建账号为 superagent-ai/reag 做出贡献。</li><li><a href="https://github.com/StevenStavrakis/obsidian-mcp">GitHub - StevenStavrakis/obsidian-mcp: A simple MCP server for Obsidian</a>: 一个简单的 Obsidian MCP 服务器。通过在 GitHub 上创建账号为 StevenStavrakis/obsidian-mcp 做出贡献。</li><li><a href="https://github.com/sigoden/llm-functions">GitHub - sigoden/llm-functions: Easily create LLM tools and agents using plain Bash/JavaScript/Python functions.</a>: 使用纯 Bash/JavaScript/Python 函数轻松创建 LLM 工具和 Agent。 - sigoden/llm-functions</li><li><a href="https://neptune.ai/blog/llm-evaluation-text-summarization">LLM Evaluation For Text Summarization</a>: 评估文本摘要非常困难，因为没有唯一的正确方案，质量通常取决于摘要的上下文和目的。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1336079039441535178)** (17 messages🔥): 

> `Aider file management, Aider CLI formatting issues, Auto-adding files in Aider, Understanding Aider chat modes, C# file-scoped namespaces` 


- **寻求 Aider 文件管理自动化**：一位成员表示，由于需要手动添加文件，使用 Aider 感觉很繁琐，正在寻找自动化方法，特别是考虑到已经有了 repo map。
   - 另一位成员提到，存在一个插件可以自动添加 VSCode 中当前打开的文件，但原帖作者并未使用 VSCode。
- **Aider CLI 文本格式问题**：一位用户反映，当窗口调整大小或移动时，Aider 的文本格式会变得奇怪。
   - 另一位成员指出，这类问题在许多 CLI 应用程序中都很常见，暗示这可能并非 Aider 特有的问题。
- **Aider 聊天模式详解**：对 Aider 的不同模式进行了说明，包括 `code`、`architect`、`ask` 和 `help`，详细介绍了每种模式如何改变交互和命令。
   - 解释中强调，可以使用 `/chat-mode` 等命令切换当前模式。
- **C# 文件作用域命名空间的 linting 错误**：一位成员报告在 C# 中使用文件作用域命名空间（file-scoped namespaces）时遇到 linting 错误，而块状风格（block style）的命名空间则不会触发此类错误。
   - 他们询问了解决此文件作用域命名空间 linting 问题的可能方案或配置。
- **关于 Aider 模型使用的咨询**：用户寻求关于何时使用 `aider_model` 以及与其他 Aider 模型类型（如 `aider_editor` 和 `aider_architect`）区别的澄清。
   - 随后的解释详细说明了模型在不同聊天模式中是如何被使用的，以及特定命令如何影响模型的选择。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://deepclaude.com/">DeepClaude</a>：未找到描述</li><li><a href="https://aider.chat/docs/usage/modes.html">Chat modes</a>：使用 code、architect、ask 和 help 聊天模式。
</li>
</ul>

</div>
  

---

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1336064222055890995)** (320 条消息🔥🔥): 

> `Cursor IDE 更新，与其他 AI 工具的对比，不同模型的使用体验，AI 工具的使用成本，对 Supermaven 的印象` 


- **Cursor IDE 更新引发褒贬不一的反应**：用户在 Cursor 最近的更新中遇到了问题，一些人指出与之前的版本相比，性能变慢且存在 Bug。
   - 虽然有些人坚持使用 Cursor，但另一些人表示沮丧，声称目前的模型无法有效地替代他们之前的体验。
- **讨论了 Cursor IDE 的替代方案**：提到了 Supermaven 和 Pear AI 等几种替代方案，关于它们与 Cursor 相比的效果，意见不一。
   - 用户分享了使用 Supermaven 的经验，指出它速度很快，但不如 Cursor 可靠，特别是在免费版中。
- **AI 工具的成本担忧**：几位用户讨论了与 Cursor 和 GitHub Copilot 等 AI 工具相关的高昂成本，表达了对其负担能力的担忧。
   - 一些用户表示倾向于成本更低的选择，而另一些人则认为 Cursor 的价值证明了其价格的合理性。
- **不同 AI 模型的使用体验**：用户分享了使用不同 AI 模型的经验，提到了使用 Cursor 构建项目的成功案例，以及对 AI 产生的错误的挫败感。
   - 讨论了使用 Claude Sonnet 等模型的情况，以及在使用 AI 辅助时面临的实际挑战。
- **社区资源和 GitHub 项目**：几位用户分享了 GitHub 仓库链接，以增强 Cursor 的功能，例如 `awesome-cursorrules` 和其他相关工具。
   - 这些资源旨在优化 Cursor 的使用，并提升在不同编码任务中的用户体验。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://repoprompt.com/">Repo Prompt</a>: 未找到描述</li><li><a href="https://www.cursor.com/blog/tab-update">A New Tab Model | Cursor - AI 代码编辑器</a>: 发布下一代 Cursor Tab 模型。</li><li><a href="https://toggl.com">Toggl Track: 适用于任何工作流的时间追踪软件</a>: 未找到描述</li><li><a href="https://forum.cursor.com/t/has-the-fusion-model-been-rolled-out/44716/2">Fusion 模型上线了吗？</a>: 给开发者的一大请求：请澄清如何理解关于即将部署 Fusion 的更新日志——如果我的版本高于 0.45，是否意味着我已经拥有了新的 Tab...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1i2b2eo/meta_prompts_because_your_llm_can_do_better_than/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://github.com/vbwyrde/AI_Dev_Helpers">GitHub - vbwyrde/AI_Dev_Helpers: 我发现在使用 AI 进行开发时很有用的一些工具</a>: 我在使用 AI 进行开发时发现有用的一些工具 - vbwyrde/AI_Dev_Helpers</li><li><a href="https://github.com/grapeot/devin.cursorrules/tree/multi-agent">GitHub - grapeot/devin.cursorrules (multi-agent 分支)</a>: 将 Cursor/Windsurf 变成 90% 的 Devin 的魔力。通过在 GitHub 上创建一个账户来为 grapeot/devin.cursorrules 的开发做出贡献。</li><li><a href="https://youtu.be/FrM6ZzCiLwU">DeepSeek R1 + Claude 3.5 Sonnet: 2 分钟开发者工作流指南</a>: 另一个简短的小视频，我描述了在 DeepSeek R1 作为免费使用的模型添加到 Cursor 后，我最新的工作流调整！试试这个并...</li><li><a href="https://www.cursor.com/changelog">更新日志 | Cursor - AI 代码编辑器</a>: 新的更新和改进。</li><li><a href="https://github.com/PatrickJS/awesome-cursorrules?tab=readme-ov-file">GitHub - PatrickJS/awesome-cursorrules: 📄 精选的 awesome .cursorrules 文件列表</a>: 📄 精选的 awesome .cursorrules 文件列表。通过在 GitHub 上创建一个账户来为 PatrickJS/awesome-cursorrules 的开发做出贡献。</li><li><a href="https://github.com/askjohngeorge/pipecat-lead-qualifier/commit/7bc1b28007103793c1d1f36ebe15e158d5acad97">重构服务器结构 ♻️ · askjohngeorge/pipecat-lead-qualifier@7bc1b28</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1336068105755168789)** (234 条消息🔥🔥): 

> `AI 训练数据的伦理，医疗保健讨论，政治意识形态，人类合作，财务不平等`

- **关于 AI 训练数据伦理的辩论**：成员们讨论了定义 AI 训练中“可疑”数据源的挑战，并对使用 Wikipedia 等数据集的影响以及 AI 能力的道德性提出了担忧。
   - 对话凸显了关于数据所有权和 AI 开发中伦理考量的更广泛辩论。
- **围绕医疗保健论点的争议**：讨论集中在反对全民医疗保健的右翼论点上，认为其不成比例地惠及某些群体，从而导致种族和阶级歧视的影响。
   - 一位成员利用围绕医疗保健政策的历史背景以及对边缘化社区的偏见说明了这一点。
- **政治意识形态冲突**：针对左翼和右翼意识形态的认知发生了激烈的交流，引发了对阶级歧视、偏执和社会结构的担忧。
   - 参与者就人权、社会角色以及经济体系对社区的影响分享了不同的看法。
- **人类合作的探讨**：成员们引用了讨论人性与合作的文献，强调互助对于生存和社会福祉至关重要。
   - 对话涉及了危机期间人类协作的历史视角，反驳了关于天生自私的叙事。
- **财务不平等及其影响**：讨论重点关注社会中日益加剧的财务不平等，围绕财富分配和政府政策的影响展开。
   - 有观点认为，只有少数精英从当前体系中获益，而许多人在经济上挣扎，并将其与更广泛的社会问题联系起来。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://annas-archive.org/blog/ai-copyright.html">版权改革对国家安全至关重要</a>：中国 LLM（包括 DeepSeek）是在我那非法的书籍和论文档案库（全球最大）上训练的。西方需要从国家安全的角度出发，彻底改革版权法。</li><li><a href="https://arxiv.org/abs/2404.02258">Mixture-of-Depths：在基于 Transformer 的语言模型中动态分配计算资源</a>：基于 Transformer 的语言模型在输入序列上均匀分布 FLOPs。在这项工作中，我们证明了 Transformer 可以学会将 FLOPs（或计算量）动态分配给特定的...</li><li><a href="https://arxiv.org/abs/2401.02412">LLM 增强的 LLM：通过组合扩展能力</a>：在海量数据集上训练的拥有数十亿参数的基础模型已在多个领域展示了非凡的技能。然而，由于其单体结构...</li><li><a href="https://openeurollm.eu/launch-press-release">Open Euro LLM</a>：一系列旨在实现欧洲透明 AI 的基础模型</li><li><a href="https://fxtwitter.com/sama/status/1886559648158826518?t=buAoDwf3kJeWwjDI0vFjqA&s=19">来自 Sam Altman (@sama) 的推文</a>：很高兴听到这个消息；很高兴能很快将其推向 Plus/免费层级！引用 Siqi Chen (@blader) 的话：目前我只用了一天，但 OpenAI 的 Deep Research 和 o3 的价值已经超过了我支付的 15 万美元...</li><li><a href="https://fxtwitter.com/Afinetheorem/status/1886206439582015870">来自 Kevin A. Bryan (@Afinetheorem) 的推文</a>：今天发布的 OpenAI 新模型非常疯狂。它本质上是 Google 的 Deep Research 理念，结合了多步推理、网页搜索，*并且*底层是 o3 模型（据我所知）。它有时...</li><li><a href="https://en.wikipedia.org/wiki/Negative_and_positive_rights">消极权利与积极权利 - 维基百科</a>：未找到描述</li><li><a href="https://x.com/mgostIH/status/1880320930855153969">来自 mgostIH (@mgostIH) 的推文</a>：深度学习到底怎么了？？？</li><li><a href="https://www.youtube.com/watch?v=AAiMOFQJPx8">我们一直以来的 LLM 推理方式都错了吗?!?!</a>：过度拟合现象：为开放式文本生成锐化并稳定 LLM。ArXiv：https://arxiv.org/abs/2412.04318 Bytez：https://bytez.com/do...</li><li><a href="https://strategic-technologies.europa.eu/get-funding_en">战略技术欧盟资金机会门户</a>：通过欧洲战略技术平台 (STEP) 发现战略技术的欧盟资金机会。使用我们的交互式仪表板查找数字、清洁和...领域的欧盟公开征集。</li><li><a href="https://www.youtube.com/watch?v=YcgFT4iNTUA">你需要数理逻辑！</a>：本频道开启了一个新系列：证明的数理逻辑。超过 8,000 名订阅者！感谢大家。如果...请继续订阅本频道。</li><li><a href="https://www.youtube.com/watch?v=g6BK5Q_Dblo">21 世纪的新宗教 | Yuval Harari | Google 演讲</a>：技术宗教与硅谷先知：21 世纪将由高科技大师还是宗教狂热分子塑造——或者他们其实是同一回事？当前的...
</li>
</ul>

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1336127970116501535)** (18 条消息🔥): 

> `Anthropic classifiers, Universal jailbreaks, Paper discussion attendance, Alignment techniques, Hallucination in LLMs` 


- **Anthropic 分类器引发担忧**：围绕关于 [constitutional classifiers](https://www.anthropic.com/research/constitutional-classifiers) 论文的讨论引发了担忧，一位成员指出推理成本增加了 **20%**，且误拒率（false refusals）增加了 **50%**，影响了用户体验。
   - 另一位成员评论说，随着模型的进步，分类器可能不足以防御**危险的模型能力**。
- **通用越狱 (Universal Jailbreaks) 的未来充满不确定性**：成员们讨论了“通用越狱”的概念，对其在大模型上的有效性表示怀疑，因为从历史上看，它们一直无法稳定发挥作用。
   - 一位成员建议，最近的趋势表明**自动发现越狱的方法**正在兴起，这使对话变得更加复杂。
- **近期讨论的参与度问题**：参与者注意到论文讨论的出席率较低，一位成员评论说 **knearestneighbor** 提前离开，导致讨论推迟。
   - 另一位成员幽默地对错过讨论表示失望，强调了该论文的高质量。
- **关于防止 LLMs 幻觉 (Hallucination) 的辩论**：关于 LLM 输出中的“幻觉”概念引发了激烈的辩论，一些人认为这是模型行为的自然方面，而非缺陷。
   - 成员们批评“幻觉”一词具有误导性，并将基于学习模式生成输出的技术拟人化了。
- **对对齐 (Alignment) 技术的质疑**：有人对 Anthropic 的方法是否能有效防止**越狱 (jail-breaking)** 提出了质疑，一些人坚持认为防止幻觉可能也是一个无法实现的目标。
   - 成员们将其与其他对齐努力进行了比较，其中一位声称 **DeepSeek R1** 已被证明比 Anthropic 的论文更有效。



**提到的链接**：<a href="https://arxiv.org/abs/2501.18837">Constitutional Classifiers: Defending against Universal Jailbreaks across Thousands of Hours of Red Teaming</a>：大型语言模型 (LLMs) 容易受到通用越狱的影响——这种提示策略可以系统地绕过模型防护措施，并使用户能够执行需要许多步骤的有害过程……

  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1336119832021569638)** (9 条消息🔥): 

> `Deepseek R1 600B, Tokenization Concerns, Image Analysis Capability, Memory and Puzzle Connection` 


- **Deepseek R1 600B 的性能令人印象深刻**：一位成员在给 Together.ai 的 **Deepseek R1 600B** 提供了一个挑战性的网格后，对其表现表示惊讶，称其与较小的模型相比产生了出色的结果。
   - 他们附带了几张截图，展示了其最终得出正确字母的能力。
- **图像分析引发笑料**：一位成员觉得模型决定画画而不是仅仅将文本解释为图像非常**搞笑**，并质疑其推理逻辑。
   - 另一位成员表示赞同，注意到其漫长的推理路径（reasoning path），但仍觉得很有趣。
- **对 Tokenizers 的担忧依然存在**：一位成员对网络对 **tokenizers** 的依赖表示怀疑，提到拼写需要过度的记忆。
   - 他们主张采用更好的方法，允许模型直接分解单词，而不是死记硬背 token 模式。
- **识别拼图碎片**：关于模型如何像解决拼图一样搜索其记忆（特别是在识别单词中的字母方面）展开了讨论。
   - 有人指出 **CNNs** 因不需要 tokenizers 而受到青睐，暗示希望有一种将 Tokenization 与直接输入感知相结合的方法。

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1336087477550710855)** (6 条消息): 

> `AI 版权和专利法，训练 AI 与社会效益，新 AI 模型发布要求，Polyapprox 项目` 


- **AI 公司操纵版权法**：一位成员指出，**AI 公司**经常躲在**版权和专利法**背后以保护其知识产权，从而在无限制访问与严格控制之间制造了困境。
   - “卖蛇油”（*Snake-oil selling*）被提及作为对这些做法的批评，暗示其主张中存在欺骗。
- **关于 AI 训练监管的分歧**：另一位成员主张制定一项允许 **AI 训练**但不改变其他任何内容的法律，并表示这应该导致一种“自由竞争”（**free-for-all**）的局面。
   - 有人对谁能从此类法律中受益表示担忧，认为主要是富人会获得回报。
- **呼吁免费发布训练好的 AI 模型**：一位成员坚持认为，如果允许使用知识产权进行 AI 训练，那么训练好的模型应该**免费回馈给社会**，并谴责其他安排是“胡扯”（*bullshit*）。
   - 这突显了对公平获取 AI 进步成果以造福社会的渴望。
- **分享 Polyapprox GitHub 项目**：一位成员分享了 **[Polyapprox GitHub 项目](https://github.com/EleutherAI/polyapprox)** 的链接，该项目似乎与正在进行的讨论相关。
   - 他们还提供了一篇相关的 [arXiv 论文](https://arxiv.org/abs/2502.01032) 链接，鼓励成员进一步探索。
- **提供截图和图像分析**：分享了多张截图以及图像分析链接，突出了与 AI 话题相关的**潜在洞察或讨论**。
   - 这些视觉资料被附上以支持频道中正在进行的讨论。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1336082071562620950)** (141 条消息🔥🔥): 

> `模型性能问题、API 模型规范、Intel Mac 兼容性、LM Studio 中的 RAG 与推理、工具与函数回调` 


- **模型性能差异**：用户注意到不同模型的性能表现不一，特别是提到 **Deepseek R1 abliterated Llama 8B** 模型的效果不如更小的 **Qwen 7B** 和 **1.5B 模型**。
   - 一位用户询问如何完全解除模型的审查限制（uncensor），指出新旧版本之间能力的差异，并期望新版本能有更好的表现。
- **理解 API 模型使用**：关于在 API 调用中使用 'local-model' 的讨论明确了它只是所用特定模型名称的占位符，特别是在加载了多个模型的配置中。
   - 建议在发起 API 请求之前明确获取模型名称，以避免在模型选择时产生歧义。
- **Intel Mac 兼容性问题**：一位用户询问了 LM Studio 对 Intel 处理器的 Mac 的兼容性，发现当前版本仅支持 Apple Silicon。
   - 会中强调 LM Studio 目前仍是闭源项目，没有自行编译（self-build）的选项，因此建议使用其他替代系统。
- **探索特定场景下的 RAG**：用户深入探讨了如何利用检索增强生成（RAG）来增强 LM Studio 在特定领域任务中的推理能力，而无需进行微调（finetuning）。
   - 一位用户强调，在考虑模型微调等更复杂的方案之前，利用向量库（vector stores）中的领域知识是首要步骤。
- **函数与工具回调策略**：一位用户在面临持续的响应问题时，寻求关于正确提示词技术（prompting techniques）的资源，特别是针对函数和工具回调。
   - 另一位成员分享了他们使用有向无环图（DAG）工作流的创新方法，以根据过往表现优化 AI 响应的选择。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/who-knows-shrug-what-shades-on-idk-gif-15962763">Who Knows Shrug GIF - Who Knows Shrug What - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://lmstudio.ai/docs/api">LM Studio as a Local LLM API Server | LM Studio Docs</a>：使用 LM Studio 在 localhost 运行 LLM API 服务器</li><li><a href="https://docs.unsloth.ai/">Welcome | Unsloth Documentation</a>：Unsloth 新手指南</li><li><a href="https://tenor.com/view/richard-stalman-richard-stalman-saint-ignucius-gnu-gif-13909134">Richard Stalman Richard GIF - Richard Stalman Richard Stalman - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://lmstudio.ai/docs/api/endpoints/rest#get-apiv0models>">LM Studio REST API (beta) | LM Studio Docs</a>：REST API 包含增强的统计数据，如每秒 Token 数（Token / Second）和首个 Token 响应时间（TTFT），以及关于模型的丰富信息，如已加载与未加载状态、最大上下文、量化等。</li><li><a href="https://github.com/Blaizzy/mlx-vlm">GitHub - Blaizzy/mlx-vlm: MLX-VLM is a package for inference and fine-tuning of Vision Language Models (VLMs) on your Mac using MLX.</a>：MLX-VLM 是一个用于在 Mac 上使用 MLX 进行视觉语言模型（VLMs）推理和微调的包。</li><li><a href="https://lmstudio.ai/docs/basics/rag">Chat with Documents | LM Studio Docs</a>：如何为 LLM 提供本地文档作为额外上下文</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/11483">Feature Request: Qwen 2.5 VL · Issue #11483 · ggerganov/llama.cpp</a>：功能请求：Qwen 2.5 VL。</li><li><a href="https://github.com/QwenLM/Qwen2.5-VL/issues/7">Support for Llama.cpp · Issue #7 · QwenLM/Qwen2.5-VL</a>：能否支持 Llama.cpp？这将使该模型能够被 Ollama、LM Studio、Koboldcpp、text-generation-webui 等许多流行工具使用。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1336073580903010365)** (120 条消息🔥🔥): 

> `服务器设置中的 RAM 差异，在不同硬件上运行模型，M4 Ultra 性能预期，GPU 配置与能力，各种模型的推理速度` 


- **RAM 类型与推理问题**：讨论强调了 **UDIMM 和 RDIMM** RAM 类型之间的差异，一位用户遇到了其 **128GB RAM** 无法适配其推理服务器的问题。
   - 另一位成员指出，使用 **单块 7900 XTX** GPU 运行 **70B 模型** 时遇到了限制，建议需要多块 GPU 才能获得可用的推理速度。
- **M4 Ultra 的性能困境**：成员们对 **M4 Ultra** 提供强劲性能的能力表示怀疑，传闻指出配备 128GB RAM 的系统起售价为 **1200 美元**。
   - 一些人推测它可能无法超越 NVIDIA 的 **Project DIGITS**，后者在集群模型方面拥有更优越的互连速度。
- **推理速度数据的困惑**：推理速度存在差异，一位用户报告其 **M2 Ultra** 达到 **30-32 TPS**，而其他人对这些数字表示怀疑，认为需要进一步明确模型版本。
   - 围绕性能指标的讨论包括模型是否运行在 **4-bit quantization** 下，普遍共识是像 **70B** 这样的大型模型在速度上会出现收益递减。
- **AI 工作站的 GPU 配置**：用户讨论了各种 GPU 配置，指出 **3090 和 4070 Super** 等 GPU 的组合可以根据可用 VRAM 提供强大的推理能力。
   - 针对高性能设备的散热解决方案引起了关注，特别是像 **16” MacBook Pro** 连接高端 GPU 的情况。
- **微调资源分配**：一位用户建议设置 **GPU 显存的 wired limit** 有助于调节性能，但其他人报告在运行较大模型时会出现乱码输出的问题。
   - 当某些配置只能正常运行较小模型时，资源调优的有效性受到了质疑。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://blog.exolabs.net/day-2/">12 Days of EXO</a>: 12 天真正的开放创新</li><li><a href="https://tenor.com/view/house-w-hat-huh-confused-unsure-gif-4211197">Confused GIF - House W Hat Huh - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1hvqydy/hp_z2_mini_g1a_is_a_workstationclass_mini_pc_with/">Reddit - 深入探索</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1336120618323546164)** (1 条消息): 

> `新任首席安全官，安全更新` 


- **认识 Perplexity 的新任首席安全官**：分享了来自 Perplexity 新任 **Chief Security Officer** 的消息，强调了未来安全的重要性。
   - 附带了一个名为 *Jimmy* 的介绍视频，供社区进一步了解其职责。
- **新任 CSO 的介绍视频**：分享了名为 [*Jimmy*](https://cdn.discordapp.com/attachments/1047204950763122820/1336120617967026319/Jimmy.MOV?ex=67a34f8b&is=67a1fe0b&hm=f6e56ab3c0f598299bf922a9aedeb20380d5ab088b4921c9f03f6867e0ef3437&) 的介绍视频，让社区了解新任 CSO 的安全愿景。
   - 该视频为团队提供了一个直接从领导层建立联系并了解安全优先事项的机会。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1336069761003687987)** (165 条消息🔥🔥): 

> `Perplexity Pro 功能、API 使用与限制、用户体验反馈、学术写作辅助、模型对比与性能` 


- **Perplexity Pro 非常值得**：许多用户认为 Perplexity Pro 计划很有价值，尤其是其近乎无限的每日 R1 使用限制。
   - 一位用户提到 Deepseek 提供的查询次数较少，突显了 Perplexity 卓越的服务器性能。
- **API 和模型限制**：用户报告在 API 中使用某些模型（如 Sonar）时遇到困难，部分用户只能使用推理模型。
   - 此外，还有关于 R1 可用性以及 API 是否对模型有限制的疑问。
- **用户建议与反馈**：几位用户表示有兴趣根据他们使用 Perplexity Pro 的经验提供建议。
   - 讨论引导用户前往特定频道提交正式建议并分享反馈。
- **学术写作辅助**：用户寻求关于如何有效利用 Perplexity 进行学术写作和查找文章的建议。
   - 虽然一些人更喜欢与 AI 进行头脑风暴，但另一些人则提到使用其他工具来辅助写作。
- **服务性能与问题**：有报告称 Perplexity 响应缓慢并存在潜在问题，引发了对服务可靠性的询问。
   - 用户查看了状态页面以了解当前问题，确认尽管存在报告的延迟，但某些功能仍在正常运行。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/avalanche-avax-gif-21537601">Avalanche Avax GIF - Avalanche AVAX - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.cplx.app/">Complexity</a>：每个人都梦寐以求的 Perplexity.ai 增强版。</li><li><a href="https://www.perplexity.ai/backtoschool">Perplexity - Race to Infinity</a>：欢迎回到学校！仅限两周，领取一个月免费的 Perplexity Pro。推荐你的朋友，如果你的学校达到 500 人注册，我们将把免费月份升级为一整年...</li><li><a href="https://x.com/aravsrinivas/status/1884509590684934211?s=61">来自 Aravind Srinivas (@aravsrinivas) 的推文</a>：@julheetel8890 Full。</li><li><a href="https://x.com/apostraphi/status/1886539187353960741?s=61">来自 Phi Hoang (@apostraphi) 的推文</a>：新员工警报。引用 Perplexity (@perplexity_ai) 来自 Perplexity 新任首席安全官的消息：</li><li><a href="https://status.perplexity.com/">Perplexity - Status</a>：Perplexity 状态</li><li><a href="https://youtu.be/8uPmC5BQtCw?si=98QvOsWrzWbEs40R">Nelima (AI) 中的公开与私有操作在 420 秒内解析</a>：自我们发布第一个关于“编程大动作模型”的视频以来，Nelima 已经取得了长足进步！在本视频中，我们很高兴推出一项强大的新功能：Pr...</li><li><a href="https://www.wikidata.org/wiki/Q123403392">Perplexity</a>：聊天机器人搜索引擎</li><li><a href="https://www.wikidata.org/wiki/Q124333951">Perplexity AI, Inc.</a>：对话式搜索引擎公司
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1336109502721691649)** (13 条消息🔥): 

> `恐惧症信息来源、MSI A520M 主板、猪的外貌、Trump 行政命令、Linux 桌面使用情况` 


- **探索恐惧症资源**：多位用户分享了一个列出各种**恐惧症 (phobias)** 的网站链接，为进一步阅读提供了整合资源，详见[此处](https://www.perplexity.ai/search/show-me-sites-that-list-phobia-Jb9EQhckS66QFqIDYJvj1A)。
   - 该分享源提供了关于各种恐惧症及其描述的有价值见解。
- **深入了解 MSI A520M 主板**：分享了关于 **MSI A520M A Pro 主板** 的信息，包括规格和评论，详见[此处](https://www.perplexity.ai/search/motherboard-msi-a520m-a-pro-aoKBBPs2Skystw7fZnDqJw#1)。
   - 用户可以找到该主板的详细对比和用户体验。
- **讨论猪及其外貌**：提供了一个关于*为什么猪被认为很丑*的链接，讨论了认知与生物学，详见[此处](https://www.perplexity.ai/search/why-pigs-are-ugly-f54qHxRsQ1SpIkudhRoHHA#0)。
   - 这一探索为这种动物的形象提供了一个幽默且具有启发性的视角。
- **Trump 最近的行政命令**：一位成员分享了关于 **Trump 签署行政命令** 的新闻，链接至文章[此处](https://www.perplexity.ai/page/trump-signs-executive-order-to-.xV8X9ILSuqTDd_jZeeKvQ)。
   - 详细探讨了该命令的影响，强调了其政治相关性。
- **Linux 桌面使用趋势**：几位用户引用了一个讨论 **Linux 桌面使用率随时间变化** 及其演变的链接，详见[此处](https://www.perplexity.ai/search/linux-desktop-usage-over-time-QFfP46jEShCTHE0bW7ODpg#0)。
   - 关于 Linux 采用率增长和变化趋势的见解突显了其市场地位。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1336068382348546068)** (9 条消息🔥): 

> `Llama 3.1 Sonar 模型弃用、Sonar-Reasoning 错误、图像检索的 API 访问、Litellm 模型更新、API 模型名称识别` 


- **Llama 3.1 Sonar 模型弃用**：一位用户报告称两周前收到了关于 **llama-3.1-sonar-small-128k-online** 的弃用邮件，并指出切换到 `sonar` 后**延迟增加了 5-10 秒**。
   - 他们询问这种延迟是否在预期之内，并寻求减轻该问题的建议。
- **Sonar-Reasoning 模型遇到 500 错误**：一位用户就 **sonar-reasoning-pro** 寻求帮助，在尝试使用时遇到了 **500 错误**。
   - 具体原因尚不明确，引发了对模型访问权限的担忧。
- **API 用户寻求图像检索访问权限**：一位用户表示需要为其 PoC 检索图像，但发现需要成为 **tier-2 API 用户** 才能访问此功能。
   - 他们询问是否可以授予临时访问权限以利用其现有额度。
- **Litellm 与新 Sonar 模型的兼容性**：一位用户询问为什么无法在新的 **sonar/sonar-pro** 模型中使用 `litellm`，怀疑是更新问题。
   - 该用户显得有些沮丧，并寻求关于模型可用性的澄清。
- **识别正在使用的 API 模型名称**：一位用户正试图在旧模型弃用前确保其系统使用正确的 API 模型，但无法看到当前调用的是哪个模型。
   - 他们注意到无论使用哪个模型，其 API 使用报告都显示为 **70b-online**，并询问是否有办法验证被调用的模型名称。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1336067470536085695)** (171 条消息🔥🔥): 

> `DeepSeek, O1 Pro 模型性能, AI 感知与未来影响, AI 中的编排服务, OpenAI 与其他模型的使用体验` 


- **DeepSeek 的社区影响**：用户讨论了 DeepSeek R1 如何引入对更广泛社区有益的新技术，这可能会打破巨头垄断 AI 技术的趋势。
   - 有人提出了关于 DeepSeek 将信息发送到中国的担忧，表明数据处理需要透明度。
- **令人印象深刻的 O1 Pro 使用体验**：成员们分享了使用 O1 Pro 的经验，强调了它在单次会话中生成多个小游戏且无错误的能力，展示了其卓越性能。
   - 一位用户计划用更具挑战性的 Prompt 和 Demo 对 O1 Pro 进行严格测试，显示出对该模型能力的信心。
- **关于 AI 感知力的辩论**：一位聊天参与者质疑 AI 是否具有感知力，并推测其对就业和社会的潜在影响。
   - 回应从对 AI 感知力的怀疑到对其未来的幽默调侃不等，没有提出严重的担忧。
- **理解编排服务 (Orchestration Services)**：用户讨论了 AI 中编排服务的概念，描述了它们如何允许模型将任务委派给多个实例。
   - 这引发了关于不同 AI 模型性能能力的更广泛讨论，特别是关于 O1 Pro 和 R1 的讨论。
- **用户对 OpenAI 模型的评价褒贬不一**：几位用户表达了在让 OpenAI 模型遵循特定指令（尤其是在润色论文时）方面的挫败感。
   - 尽管存在挑战，许多人注意到模型令人印象深刻的能力，一些人对 AI 技术的进一步发展表示兴奋。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://dataconomy.com/2025/01/31/deepseek-r1-vs-o3-mini-in-performance-cost-and-usability-showdown/">DeepSeek R1 vs o3-mini 在性能、成本和可用性方面的对决</a>：用户分析和基准测试显示 DeepSeek R1 和 o3-mini 之间各有千秋，开发者和企业有不同的侧重点。</li><li><a href="https://tenor.com/view/hate-ignorance-fear-fire-science-gif-16741306">Hate Ignorance GIF - Hate Ignorance Fear - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1igyy0n/introducing_deeper_seeker_a_simpler_and_oss/">介绍 Deeper Seeker - OpenAI 最新 Deep Research 功能的一个更简单且开源的版本。</a>：由 u/hjofficial 发布在 r/LocalLLaMA • 218 点赞和 54 条评论</li><li><a href="https://youtube.com/shorts/I_bGa-xIHkk?feature=shared">DeepSeek 在对你撒谎吗？#shorts #wireshark #deepseek #privacy #cybersecurity</a>：#shorts #wireshark #deepseek #privacy #cybersecurity
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1336064021312573440)** (11 条消息🔥): 

> `用户对 AI 性能的反馈, 儿童 AI 绘本, Pro 用户的设备登录限制, 最近的更新与 Emoji 使用, Deep Research 信息的准确性` 


- **对 AI 更新的反应不一**：用户对最近的 AI 更新表示沮丧，其中一位表示：*“你需要写 10 万页的段落才能让它理解一个 4 岁小孩都知道的事情。”* 另一位用户注意到更新后代码回复中 Emoji 的使用增加了。
   - 一位成员建议限制 Emoji 的使用以提高对编码的关注，并分享了他们让 AI 处理直接反馈的经验。
- **关于儿童 AI 绘本的查询**：一位用户询问是否有人在制作 **儿童 AI 绘本**，表明对这一细分领域的兴趣。
   - 这表明在儿童友好型 AI 应用方面存在持续的探索和潜在开发空间。
- **Pro 用户对设备限制感到好奇**：一位成员询问 **Pro 版本** 是否会限制登录设备的数量，反映了对访问便捷性的担忧。
   - 这表明用户希望明确升级服务的具体使用政策。
- **关于 GPT-4 推理能力的疑问**：一位用户质疑为什么 **GPT-4o 现在开始推理**，表现出对近期增强功能的关注。
   - 这突显了用户对 AI 不断进化的能力的关注和参与。
- **评估 Deep Research 的准确性**：一位用户询问 **Deep Research 信息的准确性**，寻求 Pro 用户的见解。
   - 另一位成员按 1 到 10 分对其进行评分，暗示了对理解 AI 生成见解可靠性的兴趣。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1336372248734339174)** (1 条消息): 

> `Structured generation, JSON Schema enhancements, UIForm utility` 


- **提升模型性能的技巧**：一位成员分享了他们在 **structured generation** 方面的经验，即利用 JSON schemas 和 Pydantic 模型中的 “thinking” 字段来增强推理过程中的 **model performance**。
   - 然而，他们警告说这种方法可能会 *污染数据结构定义*。
- **介绍用于 Schema 管理的 UIForm**：同一位成员宣布开源 **UIForm**，这是一个旨在通过 **JSON Schema extras** 简化 schema 字段添加或删除的实用工具。
   - 他们鼓励大家对该工具提供反馈，并指出安装只需一条简单的命令：`pip install uiform`。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1336372248734339174)** (1 条消息): 

> `Structured Generation Techniques, UIForm Open Source Utility, JSON Schema Enhancements` 


- **结构化生成技术介绍**：一位成员分享了关于在 JSON schemas 或 Pydantic 模型中使用 **'thinking' 字段** 的见解，通过包含推理时计算来提升模型性能。
   - 然而，他们指出这种方法的缺点是它会 **污染数据结构定义**。
- **UIForm 工具反馈征集**：同一位成员宣布开源 **UIForm**，这是一个利用 JSON Schema extras 简化 schema 中指定字段添加或删除的实用工具。
   - 他们邀请社区提供 **反馈**，并提到安装非常简单，只需运行 `pip install uiform`。
- **UIForm 的灵活性亮点**：**UIForm** 的引入通过允许动态更改数据 schema，促进了项目中结构化生成的更好管理。
   - 这可能会为处理复杂数据模型的开发者简化工作流程，提高效率。


  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1336069524092620843)** (43 条消息🔥): 

> `SoftBank OpenAI 合作伙伴关系，Google Gemini 更新，神经网络的 Harmonic Loss，对话 MultiChallenge 基准测试，OpenAI 网站重新设计` 


- **SoftBank 将向 OpenAI 产品投资 30 亿美元**：SoftBank 已承诺每年购买价值 **30 亿美元** 的 **OpenAI** 产品，并正在组建一家名为 **Cristal Intelligence** 的专注于日本市场的合资企业。
   - 该合资企业将提供企业版 **ChatGPT**，并正在洽谈一项 **400 亿美元** 的投资，这可能会使 OpenAI 的估值达到 **3000 亿美元**。
- **Google Gemini for Workspace 更新**：Google 宣布 **Gemini for Google Workspace** 不再提供附加组件，而是将 AI 功能集成到所有 Business 和 Enterprise 版本中，从而提高生产力和数据控制能力。
   - 在过去的一年里，该产品拥有超过一百万用户，此次集成旨在改变企业利用生成式 AI 工具的方式。
- **为神经网络引入 Harmonic Loss**：一篇新论文介绍了 **harmonic loss** 作为 **cross-entropy loss** 的替代方案，强调了其在训练过程中具有更好的可解释性和更快的收敛速度等优势。
   - 研究表明，使用该损失函数训练的模型能更有效地表示语义相关的词对。
- **发布针对 LLM 的 MultiChallenge 基准测试**：**MultiChallenge** 基准测试已发布，用于评估大语言模型进行**多轮对话**的能力。
   - 该倡议旨在解决一个对于语言模型在现实场景应用至关重要但研究不足的领域。
- **OpenAI 网站完成设计改版**：OpenAI 更新了其网站以符合新的设计指南，标志着品牌全新的视觉方向。
   - 此次更新是提升用户体验和品牌一致性的更广泛努力的一部分。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/Presidentlin/status/1886771841450017211">来自 Lincoln 🇿🇦 (@Presidentlin) 的推文</a>：cope cope cope cope cope cope cope cope</li><li><a href="https://x.com/dbaek__/status/1886781435794927697">来自 David D. Baek (@dbaek__) 的推文</a>：7/9 当我们使用 harmonic loss 训练 GPT-2 时，我们观察到模型倾向于以更矩形的平行四边形结构来表示语义相关的词对（例如 man:woman::king:queen）-- Harm...</li><li><a href="https://youtu.be/k3d_xeVxEOE">焕然一新。</a>：未找到描述</li><li><a href="https://support.google.com/a/answer/13623623">Gemini for Google Workspace - Business / Enterprise - Google Workspace 管理员帮助</a>：未找到描述</li><li><a href="https://x.com/dbaek__/status/1886781418115862544">来自 David D. Baek (@dbaek__) 的推文</a>：1/9 🚨 新论文预警：Cross-Entropy Loss 并不是你需要的！🚨我们引入了 harmonic loss 作为训练神经网络和 LLM 时标准 CE loss 的替代方案！Harmonic loss 实现了 🛠️si...</li><li><a href="https://huggingface.co/blog/open-deep-research">开源 DeepResearch – 解放我们的搜索 Agent</a>：未找到描述</li><li><a href="https://x.com/btibor91/status/1886880680077906376?s=61">来自 Tibor Blaho (@btibor91) 的推文</a>：OpenAI 网站现已根据新的设计指南进行了更新。引用 nic (@nicdunz)：官方 OpenAI 网站已更新，采用了新的设计指南和其他内容</li><li><a href="https://fxtwitter.com/btibor91/status/1886508640263397705">来自 Tibor Blaho (@btibor91) 的推文</a>：据 The Information 报道，SoftBank 已承诺每年购买价值 30 亿美元的 OpenAI 产品，同时组建一家专注于日本的合资企业 - SoftBank 将分销 OpenAI 技术...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1336067983709175879)** (40 messages🔥): 

> `Noam Brown 的观点、Zetta vs. Llama-1 动态、OpenAI 的硬件雄心、内部文化问题、与 OpenAI 的合作` 


- **Noam Brown 备受争议的立场**：围绕 Noam Brown 认为 **Wikipedia** 存在偏见的观点展开了讨论，他暗示在 AI **训练**中偏好 AI 生成内容而非传统来源。
   - *这符合更广泛的科技男（techbro）视角*，让那些通常对他持好感的成员感到惊讶。
- **Zetta 与 Llama-1 团队的竞争**：对话揭示了 **Zetta** 和 **Llama-1** 团队之间不同的文化和开发路径所带来的紧张关系，并指责内部功能失调。
   - 一位贡献者强调，**内部竞争**和缺乏透明度导致了项目成果方面的困难。
- **OpenAI 准备进行硬件扩张**：OpenAI 为进军**人形机器人**和 **AI 驱动的 VR 头显**领域提交了商标申请，标志着其进入硬件市场的意图。
   - 这一新方向可能会使 OpenAI 与 Meta 和 Apple 等主要参与者展开竞争，但在**拥挤的硬件挑战**中，这可能是一项艰巨的任务。
- **对内部沟通的担忧**：参与者对在 Twitter 上公开表达不满表示不安，特别是影响力人物讨论内部挫折的行为。
   - 有人指出，个人在发布关于敏感问题的推文之前，应该考虑接受**沟通培训**。
- **合作带来的机器人技术突破**：宣布退出与 OpenAI 的合作协议，理由是内部开发的**机器人 AI** 在军事技术上取得了重大突破。
   - 该公告暗示即将揭晓一个**人形机器人**项目，引发了人们对 OpenAI 持续进行的机器人雄心的关注。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.businessinsider.com/openai-trademark-humanoid-robots-vr-headsets-sam-altman-hardware-2025-2">OpenAI 为人形机器人和 VR 头显提交商标申请，Sam Altman 暗示宏大的硬件雄心</a>：OpenAI 的商标申请列出了 AI 驱动的机器人、VR 头显和可穿戴设备，这是可能进军硬件领域的最新迹象。 </li><li><a href="https://x.com/suchenzang/status/1886544793511080103">Susan Zhang (@suchenzang) 的推文</a>：如果你在吹嘘自己实验室内部的文化失调，让 IC 们为了某些地区荣誉而互相竞争……难怪你的模型完全不……</li><li><a href="https://fxtwitter.com/suchenzang/status/1886635726655430786">Susan Zhang (@suchenzang) 的推文</a>：我想知道首席 AI 科学家的 PIP 是什么样的</li><li><a href="https://x.com/ArmenAgha/status/1886522896077439187">Armen Aghajanyan (@ArmenAgha) 的推文</a>：关于 Zetta 发生的事情绝对不是真的。我们真的想公开这里发生的事情吗？引用 Yann LeCun (@ylecun) 的话：你读错了。FAIR 内部多年来一直有多个 LLM 项目……</li><li><a href="https://fxtwitter.com/soumithchintala/status/1886562033048396241">Soumith Chintala (@soumithchintala) 的推文</a>：你曾是/现在是 Meta 的首席科学家，也是 FAIR 的负责人——Zetta 和 Llama 都位于其中；我认为在公开场合以负面形象描绘你直接影响下的任何团队都是不友好的……</li><li><a href="https://x.com/Dorialexander/status/1886774547640189294">Alexander Doria (@Dorialexander) 的推文</a>：Llama 内战总结。</li><li><a href="https://x.com/adcock_brett/status/1886860098980733197">Brett Adcock (@adcock_brett) 的推文</a>：今天，我决定退出与 OpenAI 的合作协议。Figure 在完全内部构建的全端到端机器人 AI 上取得了重大突破。我们很高兴能在不久的将来向大家展示……</li><li><a href="https://x.com/elder_plinius/status/1886520062586372224">Pliny the Liberator 🐉 (@elder_plinius) 的推文</a>：@alexalbert__ @AnthropicAI ggs</li><li><a href="https://x.com/ylecun/status/1886149808500457691">Yann LeCun (@ylecun) 的推文</a>：你读错了。多年来 FAIR 内部一直有多个 LLM 项目。有些作为研究原型开源（例如 OPT175B, Galactica, BlenderBot...）。在 2022 年年中，FAIR 启动了一个大型 LLM 项目……</li><li><a href="https://fxtwitter.com/ArmenAgha/status/1886549536300261706">Armen Aghajanyan (@ArmenAgha) 的推文</a>：在 Zetta/LLaMa 两个团队中，只有一个拥有开源的预训练代码库、内部共享的数据集和实验、使用标准化的评估集、发布内部笔记并执行……
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1336085546845474991)** (84 messages🔥🔥):

> `GRPO 在 Llama 2 上的有效性，DeepSeek 在华为 Ascend 上的训练，NVIDIA Digits 的关注度，RLHF 与训练成本，网站证书问题` 


- **GRPO 在 Llama 2 上表现出色**：最近的研究结果表明，**GRPO** 显著提升了 **GSM8K** 的准确率，在 Llama 2 7B 模型上实现了 **+15 分的提升**。
   - 这表明现有模型仍然可以从强化学习技术中获益，而这一点在之前的讨论中曾受到质疑。
- **DeepSeek 现在可以在华为 Ascend 上运行**：一次讨论透露，**DeepSeek V3** 类型的模型可以在 **Huawei Ascend** 硬件上进行训练，为研究人员扩展了其可访问性。
   - 然而，对于该平台相关的性能和成本降低的某些说法的可靠性仍存在疑问。
- **对 NVIDIA Digits 的高度关注**：一位 NVIDIA 代表提到，与过去发布的 **Blackwell** 等产品相比，研究社区对 **Digits** 的兴趣显著更高。
   - 这种日益增长的兴趣被视为对寻求负担得起解决方案的资金匮乏大学的一个积极进展。
- **探讨 RLHF 模型的成本**：进一步的对话显示了对 RLHF 模型训练成本的怀疑，特别是与 **DeepSeek** 的 **R1** 模型相关的成本。
   - 针对所报告的成本是否被准确呈现提出了担忧，并引用了之前关于该主题的沟通。
- **网站证书障碍**：几位成员对 **GitHub Pages** 上持续存在的 SSL 证书问题表示沮丧，这些问题使网站的可访问性变得复杂。
   - 尽管 DNS 问题已得到解决，但证书问题依然存在，引发了对用户体验的担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.pi.website/blog/openpi">开源 π0</a>：Physical Intelligence 正在将通用 AI 引入物理世界。</li><li><a href="https://x.com/dimitrizho/status/1886713381706449379>">来自 Dimitri Zhorzholiani (@dimitrizho) 的推文</a>：DeepSeek 研究员表示，训练 R1 和 R1-Zero 仅花费了 2-3 周时间。</li><li><a href="https://epoch.ai/gradient-updates/what-went-into-training-deepseek-r1">训练 DeepSeek-R1 投入了什么？</a>：本期 Gradient Updates 探讨了 DeepSeek-R1 的架构、训练成本和定价，展示了它是如何以 30 倍更低的成本与 OpenAI 的 o1 竞争的。</li><li><a href="https://www.youtube.com/playlist?list=PLgKuh-lKre1058wlfuwtOYyemY5qoxavQ">LLM25-1：LLM、认知科学、语言学和神经科学</a>：2025 年 2 月 3-7 日</li><li><a href="https://x.com/rosstaylor90/status/1886625126222852208">来自 Ross Taylor (@rosstaylor90) 的推文</a>：没有人说 RL 对推理不起作用。争论点在于内部推理的涌现，而非 RL 带来的绝对性能提升。事实恰恰相反——我们在 Llama 2 基础模型上使用了 PPO...</li><li><a href="https://www.rlhfbook.com">(WIP) A Little Bit of Reinforcement Learning from Human Feedback</a>：关于 Reinforcement Learning from Human Feedback 的书</li><li><a href="https://x.com/georgejrjrjr/status/1886654522539266289">来自 George (@georgejrjrjr) 的推文</a>：这暗示了 V3->R1 训练计算支出的粗略上限，与 Epoch 范围的上限（500k-2M）一致。</li><li><a href="https://x.com/rdolmedo_/status/1886505669622149139">来自 Ricardo Dominguez-Olmedo (@rdolmedo_) 的推文</a>：带有可验证奖励的强化学习是否仅适用于近期的模型系列？事实证明，GRPO 在 Llama 2 7B 上也表现出色，在 GS 上实现了令人印象深刻的 +15 准确率提升...</li><li><a href="https://x.com/teortaxesTex/status/1886526422493143268">来自 Teortaxes▶️ (DeepSeek🐳 Cheerleader since 2023) (@teortaxesTex) 的推文</a>：DeepSeek V3 类型的模型现在可以在华为 Ascend 上进行训练。</li><li><a href="https://interconnects.ai">Interconnects | Nathan Lambert | Substack</a>：来自前沿 AI 实验室内部的 AI 最前沿，摒弃炒作。高层思考与技术思维的交界。每周三早晨供顶尖工程师、研究员和投资者阅读...</li><li><a href="https://www.interconnects.ai">Interconnects | Nathan Lambert | Substack</a>：来自前沿 AI 实验室内部的 AI 最前沿，摒弃炒作。高层思考与技术思维的交界。每周三早晨供顶尖工程师、研究员和投资者阅读...</li><li><a href="https://x.com/TheHumanoidHub/status/1886679733460721875">来自 The Humanoid Hub (@TheHumanoidHub) 的推文</a>：CMU 研究员与 NVIDIA 合作推出了 ASAP，这是一个用于人形机器人敏捷性的两阶段框架。它在人类数据上预训练运动策略，然后通过现实世界的修正进行微调...</li><li><a href="https://www.youtube.com/watch?v=KtBcIDtS13M&list=PLgKuh-lKre1058wlfuwtOYyemY5qoxavQ&index=6">DeepSeek 如何改变 LLM 的故事</a>：Sasha Rush (康奈尔大学) https://simons.berkeley.edu/talks/sasha-rush-cornell-university-2025-02-03 LLM、认知科学、语言学和神经科学</li><li><a href="https://youtu.be/k3d_xeVxEOE?si=eVIhUSXDlg2iu_h2~~">Refreshed.</a>：未找到描述。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1336220825409556490)** (3 条消息): 

> `Qwen/Llama 模型，旧的 RL + NLP 论文` 


- **对“小丑帽策略”的信心**：在引用[这条推文](https://x.com/teortaxesTex/status/1886553466580988282)的讨论后，一位成员断言*押注戴小丑帽的人从未错过*。
   - 这引发了另一位成员的质疑：*不可能这么简单吧*。
- **重拾旧研究**：另一位成员表示需要重新审视旧的 **RL + NLP 论文**，指出它们在今天可能具有相关性。
   - 随后有人建议，只需将任何模型换成 **Qwen/Llama**，就能产生足够的优质材料提交至 **arXiv**。



**提到的链接**：<a href="https://x.com/teortaxesTex/status/1886553466580988282">来自 Teortaxes▶️ (DeepSeek🐳 Cheerleader since 2023) (@teortaxesTex) 的推文</a>：押注戴小丑帽的人从未错过。引用 anton (@abacaj) 的话：等等... 不可能这么简单吧

  

---

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1336287885825151067)** (12 条消息🔥): 

> `Prime 论文发布，扩展深度学习模型，JAX 在行业中的应用` 


- **Prime 论文终于发表**：备受期待的 [Prime 论文](https://arxiv.org/abs/2502.01456) 已经发布，贡献者包括 **Ganqu Cui** 和 **Lifan Yuan** 等多位作者。
   - *这篇论文引入了可能重塑模型性能优化的新概念。*
- **Scaling Book 提供模型优化见解**：[Scaling Book](https://jax-ml.github.io/scaling-book/) 的分享链接揭示了在不同规模下（从单个加速器到大型集群）优化模型性能的策略。
   - 它讨论了估算训练成本的实际方法，以及硬件规格对模型效率的影响。
- **JAX 在 Google 和 xAI 中的使用**：讨论了谁在使用 JAX，其中 **Google** 和 **xAI** 被指出是主要用户。
   - 有人建议 Elon Musk 应该考虑根据 JAX 的能力更积极地部署 **Grok**。
- **AI 工具领域竞争激烈**：对话涉及了 **拥挤** 的 AI 工具市场，强调了深度学习方法论中的竞争。
   - *由于有许多平台可用，每种平台的有效性仍然是从业者辩论的热门话题。*


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://jax-ml.github.io/scaling-book/"> How To Scale Your Model </a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2502.01456">Process Reinforcement through Implicit Rewards</a>: 在大型语言模型 (LLMs) 的推理时间扩展中，密集的过程奖励已被证明是稀疏结果级奖励的一种更有效的替代方案，特别是在需要复杂...的任务中。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1336141970229166121)** (2 条消息): 

> `AI 强制许可，AI 版权问题，AI 使用的合理使用 (Fair Use)` 


- **强制许可提案获得关注**：一名成员提出，类似于音乐行业的 **带有版税的强制许可制度** 是完全废除版权的可行替代方案。
   - 该建议旨在为使用 AI 生成的内容提供明确指南，同时确保创作者获得 **适当的补偿**。
- **法律灰色地带困扰 AI 使用**：由于缺乏明确的法规，成员们担心 AI 的使用可能会陷入 **法律灰色** 地带，特别是在数据使用方面。
   - 他们强调需要将 AI 使用声明为 **合理使用 (fair use)**，否则将面临使用 **受污染数据 (tainted data)** 的后果。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1336107198660481124)** (17 条消息🔥): 

> `1D Block Tiling vs Cublas, LlamaGen 图像模型, Gen AI Hackathon, MAGVIT 视频 Tokenization, 生成时间对比` 


- **优化 1D Block Tiling 性能**：成员们讨论了 [1D block tiling 对比 Cublas](https://github.com/Omkar-Kakade-Github/Blazing_CUDA_SGEMM/blob/89230ac77af761d2d65cad97b4409f1400b6fe7c/kernels/04_1D_block_tiling.cu) 的实现，一位参与者指出在将 thread blocks 减少到 **256** 后，基准测试结果有所提升。
   - 建议进行 *5 次预热（warmups）后取平均时间进行基准测试*，以获得稳定的测量结果。
- **LlamaGen：图像生成领域的新竞争者**：**LlamaGen** 的引入承诺通过 *next-token prediction* 模型实现最先进的图像生成，性能超越了 **LDM** 和 **DiT** 等扩散框架。
   - 由于缺乏生成时间的对比，引发了关于性能指标的猜测。
- **即将举行的 Gen AI Hackathon 公告**：**CreatorsCorner** 邀请参与者参加其与各大公司合作举办的**第四届 Hackathon**，提供超过 **3.5 万美元的奖金**。
   - 鼓励团队构建能够增强用户在语音和视频应用中能力的自主 AI Agent。
- **关于 MAGVIT 和视频 Tokenization 的讨论**：成员们对 **MAGVIT2** 未开源表示遗憾，并提到其在改进视频 Tokenization 方面的潜力。
   - 尽管如此，有人提到了在生成过程中提升速度的一些技巧。
- **对生成时间的担忧**：一位参与者指出，与扩散模型相比，**LlamaGen** 的**生成时间较慢**，暗示了可能的优化方向。
   - 这引发了关于使用 **MAGVIT** 提升视频处理效能潜力的建议。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.06525">Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation</a>: 我们介绍了 LlamaGen，这是一个新的图像生成模型系列，将大型语言模型原始的 “next-token prediction” 范式应用于视觉生成领域。这是一个肯定的...</li><li><a href="https://arxiv.org/abs/2409.18869">Emu3: Next-Token Prediction is All You Need</a>: 虽然 next-token prediction 被认为是通往通用人工智能的一条充满希望的路径，但它在多模态任务中一直难以表现出色，而这些任务目前仍由扩散模型（例如...）主导。</li><li><a href="https://lu.ma/fyu8iqnk">Multimodal AI Agents - Hackathon · Luma</a>: Gen AI AgentsCreatorsCorner，与 Google Deepmind, Weights &amp; Biases, Together.ai, Stytch, Senso, LlamaIndex 等合作热忱地……</li><li><a href="https://huggingface.co/papers/2409.18869#66fd72c610a11719b680cfbb">Paper page - Emu3: Next-Token Prediction is All You Need</a>: 未找到描述</li><li><a href="https://github.com/Omkar-Kakade-Github/Blazing_CUDA_SGEMM/blob/89230ac77af761d2d65cad97b4409f1400b6fe7c/kernels/04_1D_block_tiling.cu">Blazing_CUDA_SGEMM/kernels/04_1D_block_tiling.cu at 89230ac77af761d2d65cad97b4409f1400b6fe7c · Omkar-Kakade-Github/Blazing_CUDA_SGEMM</a>: 通过在 GitHub 上创建账号来为 Omkar-Kakade-Github/Blazing_CUDA_SGEMM 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1336221236912717846)** (13 条消息🔥): 

> `Triton kernel 优化, k_cross 维度, 教程示例错误, TMA 性能, Warp specialization` 


- **寻求 Triton Kernel 优化帮助**：一位用户请求帮助优化 Triton 实现，以更好地执行内存密集型操作。他们指出其 Triton 代码比原始 PyTorch 版本慢了 **200 倍**。
   - 讨论显示 **k_cross** 是通过对矩阵 **k** 中的不同行进行交叉得出的，在大维度下及时的优化至关重要。
- **Triton 教程示例错误**：一位用户在尝试运行教程示例 '06-fused-attention.py' 时报告了与 'triton.tools.experimental_descriptor' 相关的 **ModuleNotFoundError**。他们认为该组件可能已被弃用，从而影响了新手。
   - 另一位参与者澄清说，该组件已针对较新的 GPU 进行了调整，这意味着用户可能需要更新其设置以确保兼容性。
- **TMA 对 Triton 性能的影响**：一位用户询问是否有 Triton 在使用和不使用 **TMA** 情况下的性能基准测试，并对当前的低效率表示担忧。讨论指出，如果没有 autotuning，TMA 可能无法提供优于传统方法的预期改进。
   - 参与者推测，虽然 TMA 目前尚未显示出显著收益，但随着未来的更新，特别是在结合 **warp specialization** 时，它可能会变得非常有影响力。



**提到的链接**：<a href="https://pastebin.com/t2w4NYbP">import torchimport tritonimport triton.language as tldef benchmark(f, jo - Pastebin.com</a>：Pastebin.com 自 2002 年以来一直是排名第一的文本粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。

  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1336073240132718663)** (17 条消息🔥): 

> `大输入下的缓存效率, Tensor Cores 对性能的影响, 微架构困惑, Blackwell 上的 SASS 观察, CUDA Stream 依赖关系` 


- **流式传输期间大输入的缓存困境**：*如果你的输入大于 L2 缓存并且正在进行流式传输*，那么**缓存完全没用**；连续执行 A+B+C 之类的操作会彻底破坏缓存。
   - 讨论中的成员还指出，即使在单个 stream 中也会出现此问题。
- **Tensor Cores 可能会影响非关键的 FP 操作**：一位成员担心，在利用 Tensor Cores 的 kernel 中增加 **integer operations** 的使用可能会耗尽资源，从而影响对性能不关键的 **FP operations**。
   - 另一位成员回应说，如果你受限于 FMAs，你将无法超过 **Hopper** 的性能，此时 INT/FP 的区别就不那么重要了。
- **关于微架构中指令获取的困惑**：一位成员澄清说，讨论中使用的一些短语可能意在表达每个时钟周期只能获取和解码**一条指令**。
   - 对于是否由于实现了更多核心而可以同时发布操作，产生了一些困惑。
- **对 Blackwell 架构 SASS 的观察**：**sm100/120** 架构的 SASS 看起来与 **90/Hopper** 相似，但增加了一些指示读/写屏障（read/write barriers）的细节。
   - 这一观察引发了关于针对消费级 **Blackwell** 的高延迟指令依赖关系可能发生变化的推测。
- **关于内存屏障（memory fences）和 stream 依赖关系的澄清**：一位成员询问 **CUDA stream 依赖关系** 如何影响内存屏障行为，预期两者之间存在交互。
   - 另一位成员断言，**stream 依赖关系不会影响**内存屏障行为。


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1336107368349437962)** (27 条消息🔥): 

> `Redis 连接管理, 自定义 Triton/CUDA Kernels, 编译器子进程, Redis Python 客户端` 


- **理解 Redis 连接生命周期**：围绕 Redis 连接展开了讨论，指出每次读/写都会创建一个新连接，并且在 Python 实现中没有显式关闭连接的操作，因为 **redis 使用了线程池 (threadpool)**。
   - 成员们询问了如何调整 Redis 的连接行为，并确认他们正在使用 [redis-py GitHub 客户端](https://github.com/redis/redis-py) 进行实现。
- **使用 Triton/CUDA 处理图断裂 (Graph Breaks)**：成员们讨论了在 torch 编译期间通过自定义 Triton/CUDA kernels 避免图断裂的可能性，指出 **Triton 应该是通用的**，而 CUDA 则需要将 kernels 定义为自定义算子 (custom ops)。
   - 然而，关于将算子融合 (fusion) 进用户定义的 Triton kernels 仍存在不确定性，相关细节引用了 [GitHub 上的一个 issue](https://github.com/pytorch/pytorch/issues/136227)。
- **编译器子进程的线程池查询**：由于 Torch 在编译期间会生成大量子进程，成员们担心每个编译器子进程是否都关联了一个 Redis 线程池。
   - 这引发了关于在用户编译请求时，多子进程环境下如何管理 Redis 连接的进一步调查。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://redis-py.readthedocs.io/en/stable/connections.html">连接到 Redis - redis-py 开发文档</a>：暂无描述</li><li><a href="https://github.com/redis/redis-py">GitHub - redis/redis-py: Redis Python 客户端</a>：Redis Python 客户端。通过在 GitHub 上创建账号来为 redis/redis-py 的开发做出贡献。</li><li><a href="https://github.com/pytorch/pytorch/issues/136227">[Inductor] 将逐元素算子 (pointwise ops)（及更多）融合进用户定义的 triton kernel 的某种机制 · Issue #136227 · pytorch/pytorch</a>：我们有一个有趣的用例：用户最初使用 PyTorch 算子构建模型，然后为了前向和后向计算切换到了用户定义的 triton kernels...</li><li><a href="https://github.com/pytorch/pytorch/issues/146414">pytorch/pytorch 中的 MX 基础数据类型 · Issue #146414 · pytorch/pytorch</a>：🚀 特性、动机和推介 概述：Open Compute Project 在 2023 年 9 月引入了 MicroScaling 格式 (MX)，定义了带有 E8M0 块缩放的块缩放数据类型，包括 FP8|FP6|FP4|INT8 .....</li><li><a href="https://github.com/pytorch/pytorch/blob/3aeccf2a2852a609a83cb2a529a1e5aba317b5fd/torch/_inductor/remote_cache.py#L290">pytorch/torch/_inductor/remote_cache.py (位于 3aeccf2a2) · pytorch/pytorch</a>：Python 中具有强 GPU 加速能力的张量和动态神经网络 - pytorch/pytorch</li><li><a href="https://github.com/pytorch/pytorch/blob/3aeccf2a2852a609a83cb2a529a1e5aba317b5fd/torch/_inductor/remote_cache.py#L237">pytorch/torch/_inductor/remote_cache.py (位于 3aeccf2a2) · pytorch/pytorch</a>：Python 中具有强 GPU 加速能力的张量和动态神经网络 - pytorch/pytorch
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1336204515967369357)** (4 条消息): 

> `FP8 Attention 性能, Attention 对量化的敏感性, 长序列长度推理, Flash Attention 3, DeepSeek V3 中的量化` 


- **FP8 Attention 可能会降低输出质量**：一位用户注意到，在使用 **Flash Attention 3 FP8 kernel** 时，其 diffusion transformer 模型的推理速度有所提升，但输出质量显著下降。
   - 这引发了关于 FP8 attention 实际应用的疑问，因为目前**尚无论文**报道其在此类场景下的应用。
- **量化对 attention 影响的解释**：另一位成员假设 **FP32** 和 **FP8** 之间的输出差异（平均约为 **1e-5**）会在 softmax 期间累积，从而影响长上下文中的 attention 分布。
   - 他们引用了 NVIDIA 文档中的一个示例，该示例讨论了这些细微差异及其潜在影响。
- **输出差异归因于线性层？**：随后展开了讨论，争论报告的 **1e-5 输出差异** 是否是指线性层的结果。
   - 这凸显了理解量化如何影响 transformer 模型不同组件的复杂性。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 条消息): 

iron_bound: https://www.youtube.com/watch?v=rCwgAGG2sZQ

### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1336118124218683477)** (1 条消息): 

> `Staff Software Engineer, Model performance, Inference engine, Performance monitoring, Generative media models` 


- **Fal 招聘 ML Performance 资深软件工程师**：Fal 正在招聘一名专注于 **ML Performance & Systems** 的 **Staff Software Engineer**，旨在提升生成式媒体模型的性能。该职位涉及在公司内部推理引擎（inference engine）上设计和实现创新的模型服务架构。
   - 该工程师还将开发 **performance monitoring** 和 profiling 工具，以精准定位瓶颈并优化系统资源。
- **新职位的核心职责**：该角色包括维护 Fal 在模型性能方面的领先地位，特别是针对 **generative media models**。关键任务包括在最小化 **latency** 和资源消耗的同时，最大化 **throughput**。
   - 该工程师将与 Applied ML 团队及客户紧密合作，确保工作负载能有效地利用其加速器（accelerator）。



**相关链接**: <a href="https://fal.ai/careers/staff-software-engineer-ml-performance-systems">Staff Software Engineer, ML Performance &amp; Systems</a>: Staff Software Engineer, ML Performance &amp; Systems

  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1336097116325548073)** (6 条消息): 

> `GPU mode lecture 16, Nvidia CUB resources, Kernel Tuner on GitHub, CUDA kernel optimization project, Fused attention tutorial issues` 


- **索取 GPU Mode 代码**：一位用户询问了用于复现 **GPU mode lecture 16** 中关于 hands profiling 示例的代码，寻求社区的帮助。
   - 这反映了初学者在面对复杂主题时对可复现材料的普遍需求。
- **Nvidia 的 Advanced Scan 与开源期望**：一位参与者提到 **Nvidia** 在 **Advanced Scan lecture** 中强调了 CUB 的资源，并希望这些资源将来能够开源。
   - 他们还分享了 [Kernel Tuner GitHub repository](https://github.com/KernelTuner/kernel_tuner) 的链接，该仓库提供了相关资源。
- **CUDA Kernel 优化见解**：另一位用户提到了他们之前搁置的一个项目，该项目探索了通过强化学习方法进行 **CUDA kernel optimization**，并链接到了他们在 GitHub 上的 fork。
   - 他们承认基本思路已经确立，但指出仍需更多工作。
- **Fused Attention 教程错误**：一位初学者报告了在尝试运行 **'06-fused-attention.py'** 教程时遇到的错误，该错误是由于与已弃用的 `experimental_descriptor` 模块相关的 **ModuleNotFoundError** 引起的。
   - 他们希望开发者能更新教程示例，并强调了这些示例对新人的重要性。


<div class="linksMentioned">

<strong>相关链接</strong>:

<ul>
<li>
<a href="https://github.com/KernelTuner/kernel_tuner">GitHub - KernelTuner/kernel_tuner: Kernel Tuner</a>: Kernel Tuner. Contribute to KernelTuner/kernel_tuner development by creating an account on GitHub.</li><li><a href="https://github.com/WAT-ai/CUDA-kernel-optimization">GitHub - WAT-ai/CUDA-kernel-optimization: Optimizing CUDA kernels using a reinforcement learning approach</a>: Optimizing CUDA kernels using a reinforcement learning approach - WAT-ai/CUDA-kernel-optimization
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1336182880040128565)** (13 条消息🔥): 

> `Cursor vs Github Copilot, Sapiens Model Conversion, Efficiency of Codebase Tools` 


- **Cursor 优于 GitHub Copilot**：用户指出 **Cursor** 与 **GitHub Copilot** 之间的差异是**天壤之别**，Cursor 在性能和实用性方面显著更优。
   - 根据分享的经验，*Copilot* 往往会减慢工作流程，且整体帮助较小，尤其是在使用免费版本时。
- **代码库工具的效率**：对于**密集型代码库（dense codebases）**，仅依靠工具获取准确信息可能会浪费时间，此时**人工判断**效率更高。
   - 相反，对于**中小型代码库**，Cursor 等工具已被证明是最佳选择，能显著简化编码过程。
- **关于 Sapiens 模型转换的咨询**：一位用户寻求将 **Sapiens model** 转换为 TFLite 格式的帮助，并提供了其在 [Hugging Face 上的详细信息](https://huggingface.co/facebook/sapiens-depth-0.3b) 链接。
   - Sapiens 模型由 **Meta** 开发，是一个在 **3 亿张高分辨率图像**上预训练的 Vision Transformer，展现出强大的泛化能力。



**相关链接**: <a href="https://huggingface.co/facebook/sapiens-depth-0.3b">facebook/sapiens-depth-0.3b · Hugging Face</a>: no description found

  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1336109423118258308)** (3 messages): 

> `DreamCoder Optimization, NVIDIA GTC Sessions` 


- **加入 DreamCoder 优化项目**：一位成员表示有兴趣组建项目团队，针对 **CUDA** 优化 **DreamCoder**，这涉及程序合成（program synthesis）。
   - 另一位成员鼓励他们*如果找到同行，就组建一个工作组*。
- **探索 NVIDIA GTC 会议**：一位成员分享了 NVIDIA GTC 会议目录链接，供那些对 **AI 定制内容**、技术细节和业务策略感兴趣的人参考。
   - 该会议鼓励参与者根据自己的兴趣**选择定制内容**。



**提及的链接**：<a href="https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=kernel%20fusion#/session/1728599648492001N7Sn">NVIDIA #GTC2025 Conference Session Catalog</a>：3 月 17-21 日，在圣何塞现场或在线体验 GTC 2025。

  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1336420968780005486)** (1 messages): 

> `E2E AutoML Model Compression, Edge Deployment Optimization, GitHub Project: sconce` 


- **介绍用于模型压缩的 sconce**：一位成员分享了他们的项目 **sconce**，这是一个端到端 **AutoML** 模型压缩包，旨在为边缘部署优化寻找**帕累托最优参数（pareto optimal parameters）**，可在 [GitHub](https://github.com/satabios/sconce) 上获取。
   - *更多功能和支持即将推出*，邀请其他人参与并评价该工具包。
- **行动号召：为 sconce 项目点亮 Star**：该成员鼓励如果大家欣赏这项工作，请在项目中**“点击 Star”**。
   - 此举旨在提高该工具包持续开发的曝光度和支持度。



**提及的链接**：<a href="https://github.com/satabios/sconce">GitHub - satabios/sconce: E2E AutoML Model Compression Package</a>：端到端 AutoML 模型压缩包。通过在 GitHub 上创建账号来为 satabios/sconce 的开发做出贡献。

  

---

### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1336064069358190693)** (68 条消息🔥🔥): 

> `Python RNG Stability, Kimi Multimodal Model, Sokoban Puzzle Solver, Deterministic Clue Generation, Reasoning Model Performance` 


- **Python RNG 稳定性问题**：讨论了 Python 的 RNG 在不同版本和平台上的不稳定性，特别指出 *hash randomization*（哈希随机化）会影响确定性行为。
   - 他们建议在生成谜题时使用 `PYTHONHASHSEED` 等环境变量来缓解这些问题。
- **探索 Kimi 的多模态声明**：一名成员询问了关于 Kimi 的讨论，该模型声称达到了 o1 级别的多模态能力，并提到了其 RL 训练方法。
   - 另一位参与者注意到在训练中使用长度惩罚（length penalty）的显著差异，但指出模型权重尚未发布。
- **新的 Sokoban 谜题求解器实现**：一名成员分享了一个 Sokoban 谜题求解器的实现，并表示有信心它能有效解决 reasoning-gym 数据集中的任务 49。
   - 他们注意到 *chatgpt-r* 和 *DSR1* 在处理大型迷宫时都表现吃力，强调了改进规划能力的必要性。
- **确定性线索生成的挑战**：讨论强调了在多次迭代中生成确定性线索的挑战，提出了一些解决方案，包括在打乱（shuffling）之前进行排序（sorting）。
   - 一名成员表示，尽管之前的实现中存在重复线索，但目前的调整已显示出稳定性。
- **推理模型在任务上的表现**：成员们观察到一些推理模型（如 chatgpt）在解决简单任务时花费的时间比平时长得多，这表明了潜在的局限性。
   - 特别是，解决一个基础的 Sokoban 任务花费了 451 秒，这引发了对模型效率的担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program.">Disable hash randomization from within python program</a>：从 Python 3.3 开始，哈希算法被加入非确定性盐值以避免某种攻击。这对于 Web 服务器很有好处，但在调试程序时却很麻烦：曾经...</li><li><a href="https://github.com/Deep-Agent/R1-V">GitHub - Deep-Agent/R1-V: Witness the aha moment of VLM with less than $3.</a>：以不到 3 美元的成本见证 VLM 的“顿悟时刻”。通过在 GitHub 上创建账号为 Deep-Agent/R1-V 的开发做出贡献。</li><li><a href="https://github.com/xbandrade/sokoban-solver-generator">GitHub - xbandrade/sokoban-solver-generator: Sokoban puzzle generator and solver with BFS, A* and Dijkstra</a>：带有 BFS、A* 和 Dijkstra 算法的 Sokoban 谜题生成器和求解器 - xbandrade/sokoban-solver-generator</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/59/files">Make zebra puzzle clue order deterministic by andreaskoepf · Pull Request #59 · open-thought/reasoning-gym</a>：未找到描述
</li>
</ul>

</div>

### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1336066195757076592)** (2 条消息): 

> `神经网络采样的概率、MLP 和 GLU 的可解释性、SVD 与对抗样本、训练中的相变、闭式多项式近似` 


- **功能性语言模型极低的出现概率**：随机猜测一个功能完备的语言模型权重的概率大约为 **3.6 亿分之一**。
   - 研究人员开发了一种通过在权重空间进行随机采样来估算该概率的方法，揭示了网络复杂性的深刻见解。
- **理解 MLP 和 GLU**：研究人员正在将 **MLP** 和 **GLU** 转换为闭式多项式（closed-form polynomials），以便利用 SVD 技术提升可解释性。
   - 这种方法允许直接检查并有助于可视化深度学习模型的属性。
- **SVD 揭示对抗结构**：在训练线性化的 MLP 上使用 SVD 可以生成**对抗样本（adversarial examples）**，这些样本可以映射回原始 MLP，证明了近似模型捕捉到了分布外（out-of-distribution）行为。
   - 这突出了一种理解复杂网络行为的高效方法。
- **网络复杂性的相变实验**：在 MNIST 训练过程中发现了一个相变现象，网络复杂性在 **500 到 1k** 步之间从简单行为转变为非线性行为。
   - 这一发现强调了神经网络在训练过程中不断演变的复杂性。
- **增强可解释性的闭式多项式近似**：最近的研究工作使得使用多项式函数近似 MLP 和 GLU 成为可能，且不会产生显著的性能损失。
   - 该技术允许研究人员通过近似式的特征分解（eigendecomposition）更好地直观解释深度学习网络。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/EleutherAI/polyapprox">GitHub - EleutherAI/polyapprox: Closed-form polynomial approximations to neural networks</a>：神经网络的闭式多项式近似 - EleutherAI/polyapprox</li><li><a href="https://arxiv.org/abs/2502.01032">Converting MLPs into Polynomials in Closed Form</a>：最近的研究表明，纯二次函数可以在 Transformer 中取代 MLP 且无显著性能损失，同时开启了基于线性代数的可解释性新方法。...</li><li><a href="https://x.com/norabelrose/status/1886834375565959507">Nora Belrose (@norabelrose) 的推文</a>：MLP 和 GLU 难以解释，但它们占据了 Transformer 的大部分参数。线性函数和二次函数更容易解释。我们展示了如何将 MLP 和 GLU 转换为闭式多项式...</li><li><a href="https://github.com/EleutherAI/basin-volume">GitHub - EleutherAI/basin-volume: Precisely estimating the volume of basins in neural net parameter space corresponding to interpretable behaviors</a>：精确估算神经网络参数空间中对应于可解释行为的盆地体积 - EleutherAI/basin-volume</li><li><a href="https://arxiv.org/abs/2501.18812">Estimating the Probability of Sampling a Trained Neural Network at Random</a>：我们提出了一种算法，用于估算在 Gaussian 或均匀先验下，神经网络参数空间中对应于特定行为（如达到...）区域的概率质量。</li><li><a href="https://x.com/norabelrose/status/1886504219919966320">Nora Belrose (@norabelrose) 的推文</a>：通过随机猜测权重获得一个功能完备的语言模型的概率是多少？我们计算了数据，答案如下：
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1336066949318185080)** (25 条消息🔥): 

> `Mixture of Experts, Custom LLM Tool, LLM Evaluation Harness Issues, Inducing Reasoning in Post-training, NLP Novice Contributions` 


- **通过视觉指南探索 Mixture of Experts**：一位成员分享了关于 Mixture of Experts (MoE) 的[全面概述](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)，包含超过 **50 个可视化图表**以及带有动画的 **YouTube 版本**链接。
   - EleutherAI 还有一个专门针对 MoE 的阅读小组，尽管目前其活跃程度尚不确定。
- **自定义 LLM 组装工具提案**：一位成员提议开发一种**拖拽式工具**来组装自定义 LLM，允许用户实时可视化不同架构和层如何影响模型行为。
   - 这个想法被认为是一个有趣的业余项目，表明了人们对 LLM 定制化实际实现的兴趣。
- **DeepSeek 模型评估分数较低**：一位成员报告在使用 **llm evaluation harness** 评估 DeepSeek 蒸馏模型时得分较低，怀疑 `<think>` 标签是罪魁祸首。
   - 他们寻求关于如何验证此问题或在评估期间忽略这些标签的建议。
- **探索 AI 模型的推理能力**：一位成员提出了一个引发深思的问题，即向 Base Model 询问其内在本质，询问这是否能揭示 AI 的先天倾向。
   - 这个想法激发了人们对 AI 回答存在性问题本质的好奇。
- **寻求在 NLP 和 AI 领域的贡献**：几位新成员介绍了自己，表达了对 **NLP、生成式 AI** 以及 AI 模型不确定性等相关领域的热情和经验。
   - 他们正在寻找机会为社区内正在进行的项目和讨论做出贡献。



**提及的链接**：<a href="https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts">A Visual Guide to Mixture of Experts (MoE)</a>：揭秘 MoE 在 Large Language Models 中的作用

  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1336063727111372870)** (59 条消息🔥🔥): 

> `Harmonic Loss, Self-play Theorem Prover, Polynomial Transformers, Feature Addition in Models, Physical Intelligence Open Source` 


- **Harmonic Loss 撼动训练方式**：一项新研究引入了 **Harmonic Loss**，作为训练神经网络和 LLM 时 Cross-entropy Loss 的一种更具可解释性且收敛更快的替代方案，并在多个数据集上展示了性能提升。
   - Harmonic 模型在泛化能力和可解释性方面优于标准模型，预示着未来 LLM 训练的重大收益。
- **Self-play Theorem Prover (STP) 详解**：这是一种新颖的方法，**Self-play Theorem Prover** 通过在猜想与证明之间交替进行，以提高训练效率，尽管形式化定理证明的高质量数据有限。
   - 该方法通过基于之前的模型输出生成新猜想，解决了稀疏奖励（Sparse Rewards）的挑战，类似于传统数学进展中的方法。
- **探索 Polynomial Transformers**：成员们讨论了 **Polynomial (Quadratic) Transformers** 的潜力，推测用其替换 MLP 以增强模型效率，特别是在 Attention 机制中。
   - 针对经典模型与双线性（Bilinear）方法进行了比较，指出了在大规模参数效率和复杂性方面的权衡。
- **在不丢失旧知识的情况下添加特征**：在关于机器学习模型“向上回收”（Upcycling）的讨论中，有人建议如果通过特定的初始化技术妥善处理，集成新特征可以保持现有知识。
   - 提到了正交微调（Orthogonal Finetuning）等方法，作为在不显著丢失先前学习信息的情况下有效结合多模态特征的手段。
- **Physical Intelligence 的开源倡议**：**Physical Intelligence** 团队开源了他们的工作和模型，突出了 AI 在物理任务应用方面的进展，超越了传统的 AI 基准测试。
   - 社区对实验这一新发布的内容表现出浓厚兴趣，特别是注意到其实现中使用了 JAX。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.00212">Beyond Limited Data: Self-play LLM Theorem Provers with Iterative Conjecturing and Proving</a>：LLM 在形式化定理证明中的一个根本挑战是缺乏高质量的训练数据。虽然强化学习或专家迭代通过交替进行……部分缓解了这一问题。</li><li><a href="https://arxiv.org/abs/1602.02068">From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification</a>：我们提出了 Sparsemax，这是一种类似于传统 Softmax 的新激活函数，但能够输出稀疏概率。在推导其性质后，我们展示了如何高效地计算其 Jacobian 矩阵……</li><li><a href="https://arxiv.org/abs/2502.00873">Language Models Use Trigonometry to Do Addition</a>：数学推理是衡量大语言模型（LLM）能力的一个日益重要的指标，但我们仍缺乏对 LLM 如何处理简单数学任务的理解。为了解决这个问题……</li><li><a href="https://arxiv.org/abs/2502.01628">Harmonic Loss Trains Interpretable AI Models</a>：在本文中，我们引入了 **Harmonic Loss** 作为训练神经网络和大语言模型（LLM）时标准 Cross-entropy Loss 的替代方案。Harmonic Loss 能够提高可解释性……</li><li><a href="https://www.desmos.com/calculator/kpuu8besbg">Desmos | Graphing Calculator</a>：未找到描述</li><li><a href="https://manifestai.com/articles/symmetric-power-transformers/">Symmetric Power Transformers - Manifest AI</a>：一种线性 Transformer，其学习方式类似于常规 Transformer，且状态可容纳在 GPU 上。</li><li><a href="https://www.physicalintelligence.company/blog/pi0">Our First Generalist Policy</a>：Physical Intelligence 正在将通用 AI 引入物理世界。</li><li><a href="https://fixupx.com/ChrSzegedy/status/1886881600367161679">Christian Szegedy (@ChrSzegedy) 的推文</a>：我非常想看到 Harmonic 加权的 Attention 会如何工作。那里可能存在真正的潜力。引用 David D. Baek (@dbaek__) 1/9 🚨 新论文预警：Cross-Entropy Loss...</li><li><a href="https://github.com/Physical-Intelligence/openpi">GitHub - Physical-Intelligence/openpi</a>：通过创建账户为 Physical-Intelligence/openpi 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1336333753047449640)** (5 条消息): 

> `Tuned Lens library, Affine translator for Llama 3.2 1B, Training data for GPT-2` 


- **探索用于翻译的 Tuned Lens 库**：一位成员分享了他们使用 **Tuned Lens library** 为 **Llama 3.2 1B** 训练 Affine translator 的工作。
   - 他们询问了用于 **GPT-2** 的训练数据量，表现出对优化其设置的积极兴趣。
- **GPT-2 训练数据见解**：另一位成员回应称，用于 **GPT-2** 的数据量约为 **250 乘以 2^18**。
   - 这一输入提供了一个数值参考，可能有助于微调转换器的训练参数。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1336372588695523352)** (6 条消息): 

> `Bucket size for zero, Training models on different A100 configurations, Activation Checkpointing vs GAS, Model Performance Metrics, Optimization strategies` 


- **根据模型大小寻找理想的 ZeRO bucket size**：一位用户询问了根据模型大小选择 ZeRO bucket size 的规范方法，并提到他们目前依赖于可参考的模型配置。
   - 他们随后想起之前问过类似的问题，表明希望进一步加深理解。
- **不同 A100 集群上的 tokens per second 对比**：另一位用户询问在 40GB 与 80GB A100 集群上训练相同模型，是否能如预期般获得接近 2 倍的 tokens per second 增长。
   - 尽管使用了更大的 batch size，他们观察到仅有 **10-15%** 的增长，并对性能优化表示担忧。
- **1.4B 模型的低 TPS 指标**：用户报告在 8x80GB A100 上，1.4B 模型的 TPS 接近 **14K**，认为这一指标相当低。
   - 他们对可以实施哪些进一步优化来提高这一数值表示不确定。
- **GAS vs. Activation Checkpointing 以提高 TPS**：观察结果表明，在不使用 activation checkpointing 的情况下，通过 GAS 增加有效 batch size 可以获得显著更高的 TPS（**242K TPS** vs **202K TPS**）。
   - 虽然有效 batch size 不同，但用户注意到较小的 batch size 结合 GAS 证明速度更快，并提出了关于监控 HFU/MFU 指标的问题。
- **分享模型配置见解**：用户分享了一个用于基准速度实验的 GitHub 配置链接，详细介绍了模型并行自回归 Transformer 的实现。
   - 这一共享配置可作为其他寻求改进模型训练设置的人员的资源。



**提到的链接**：<a href="https://github.com/aflah02/gpt-neox/blob/olmo-support/configs/hubble/Speed_Exps/1_1B_Baseline_BS_48_Both_Fusion_GQA_KV_Heads_4.yml">gpt-neox/configs/hubble/Speed_Exps/1_1B_Baseline_BS_48_Both_Fusion_GQA_KV_Heads_4.yml at olmo-support · aflah02/gpt-neox</a>：基于 Megatron 和 DeepSpeed 库在 GPU 上实现模型并行自回归 Transformer —— aflah02/gpt-neox

  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1336066332353101845)** (74 条消息🔥🔥): 

> `DeepSeek and AI advancements, Open source recommendation systems, Reinforcement Learning in AI, Mistral funding, Peano axioms in reasoning` 


- **DeepSeek 在 AI 领域的重大突破**：一位成员分享了关于 [DeepSeek](https://www.youtube.com/watch?v=3DEPDN8oD0w) 如何改变人类 AI 格局的见解，并针对 Altman 在近期发展中的立场引发了批判性评论。
   - *Altman 被贴上了 hypeman 的标签*，因为许多人质疑他在 Open source 倡议方面言行不一。
- **探索 Open Source 推荐系统**：新成员 Amith 分享了他的经验，并强调 **Gorse** 是一个知名的 Open source 推荐系统，同时也表示这些系统仍需要时间来成熟。
   - 另一位成员建议关注 **ByteDance** 的推荐技术，为现有资源的讨论增添了新视角。
- **探索 Reinforcement Learning 的潜力**：关于 **Reinforcement Learning (RL)** 是否能教会 AI 好奇心等内在价值的讨论引发了兴趣，成员们思考了其在促进知识综合方面的能力。
   - Juahyori 指出了在 Continual learning 语境下维持已学行为的复杂性，强调了 AI 模型在 Alignment 方面面临的挑战。
- **Mistral 的融资里程碑**：提到了 Mistral 最近获得的 **5 亿美元** 融资，这是一项可能影响 AI 竞争格局的重大成就。
   - 成员们推测这一公告对 Mistral 来说是否算好消息，认为这笔投资反映了对其发展轨迹的信心。
- **理解 AI 推理中的 Peano axioms**：一位成员引用了一份文档，探讨 AI 模型如何使用 **Peano axioms** 证明基础算术概念（如 **1+1=2**）。
   - 讨论引发了对这些公理在 AI 推理中影响的好奇，以及**不同文化**是否会以独特的方式对待数学。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v">YouTube</a>: 未找到描述</li><li><a href="https://x.com/tsarnick/status/1778524418593218837?s=46">Tsarathustra (@tsarnick) 的推文</a>: Geoffrey Hinton 表示 AI 模型具有直觉、创造力以及人类无法察觉的类比能力</li><li><a href="https://x.com/teknium1/status/1885592392658805237?s=46">Teknium (e/λ) (@Teknium1) 的推文</a>: 另一个 proto-hermes-reasoner 的一些测试，摘自“你的目的是什么，生命的目的是什么？”</li><li><a href="https://x.com/teknium1/status/1885565179314004429?s=46">Teknium (e/λ) (@Teknium1) 的推文</a>: 正在开发一个拒绝识别分类器，以便在模型拒绝时自动重新采样，或在 LLM 调用中注入 “sure” 等前缀。遗憾的是 fasttext 效果不够好，所以正在训练...</li><li><a href="https://github.com/relign-ai/relign">GitHub - relign-ai/relign: post train language models on multi-step reasoning with reinforcement learning</a>: 通过 Reinforcement Learning 在多步推理上对语言模型进行后训练 - relign-ai/relign</li><li><a href="https://www.youtube.com/watch?v=3DEPDN8oD0w">Sam Altman: OpenAI has been on the &#39;wrong side of history&#39; post-DeepSeek</a>: CNBC 的 Deirdre Bosa 报道了 OpenAI 的最新进展。</li><li><a href="https://www.youtube.com/watch?v=_1f-o0nqpEI">DeepSeek, China, OpenAI, NVIDIA, xAI, TSMC, Stargate, and AI Megaclusters | Lex Fridman Podcast #459</a>: Dylan Patel 是 SemiAnalysis 的创始人，这是一家专注于半导体、GPU、CPU 和 AI 硬件的研究分析公司。Nathan Lambert 是一位...</li><li><a href="https://digital-strategy.ec.europa.eu/en/news/pioneering-ai-project-awarded-opening-large-language-models-european-languages">A pioneering AI project awarded for opening Large Language Models to European languages</a>: 欧盟委员会授予多语言 AI 项目 OpenEuroLLM 著名的欧洲战略技术平台 (STEP) 印记——这是第一个获得该奖项的数字欧洲计划资助项目...</li><li><a href="https://x.com/i/grok/share/eUQwpP7nfyRAatWTzGbOQuotX">来自 GitHub 的推文 - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能 - FixTweet/FxTwitter
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1336127678830608414)** (11 messages🔥): 

> `Git 仓库与模块化系统、模型性能评估、检索增强生成 (RAG)、网页抓取政策` 


- **即将推出：带有模块化 CoT 系统的 Git 仓库**：一名成员正与 Alignment Lab AI 合作开展一个项目，最近在他们的设置中添加了一个**模块化 CoT 系统**，这表明 **Hermes 3B** 具有令人印象深刻的推理能力。
- **关于 Deepseek-R1-Distill-Qwen-14b 质量的辩论**：有人对 **Deepseek-R1-Distill-Qwen-14b** 是否真正有效提出了质疑，并指出许多模型在 OpenLLM 排行榜上的表现都优于它。
   - 一些成员认为 **Phi-4** 可能会以*更少的输出 Token* 提供更好的结果，并强调了用例的相关性。
- **利用个人数据构建 RAG 系统**：一位成员表示有兴趣构建一个 **RAG 系统**来消化个人书籍和笔记，以检索核心见解。
   - 另一位用户分享了一个名为 [local-rag](https://github.com/jonfairbanks/local-rag) 的现有实现，该系统在运行过程中敏感数据不会离开用户的网络。
- **网页抓取见解与政策**：讨论强调了 Brave 在网页抓取方面的独特立场，并将其与 Google 关于 LLM 的政策方法进行了对比。
   - 网页抓取被定义为从网络收集公共信息的过程，主要使用 Bot 或网络爬虫。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://brave.com/glossary/web-scraping/">网页抓取的含义与定义 | Brave</a>: 在线隐私可能令人困惑。在这个易于阅读的列表中，你会发现包括网页抓取在内的基本隐私和互联网术语的简短定义。在此查看 Brave 隐私术语表。</li><li><a href="https://github.com/jonfairbanks/local-rag">GitHub - jonfairbanks/local-rag: 使用开源大语言模型 (LLMs) 摄取文件以进行检索增强生成 (RAG)，无需第三方，敏感数据也不会离开你的网络。</a>: 使用开源大语言模型 (LLMs) 摄取文件以进行检索增强生成 (RAG)，无需第三方，敏感数据也不会离开你的网络。 - jonfairbanks/local-rag</li><li><a href="https://github.com/jonfairbanks/local-rag?tab=readme-ov-file)">GitHub - jonfairbanks/local-rag: 使用开源大语言模型 (LLMs) 摄取文件以进行检索增强生成 (RAG)，无需第三方，敏感数据也不会离开你的网络。</a>: 使用开源大语言模型 (LLMs) 摄取文件以进行检索增强生成 (RAG)，无需第三方，敏感数据也不会离开你的网络。 - jonfairbanks/local-rag
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1336085913616519288)** (4 messages): 

> `Society Library 使命、政治 AI Agent、SWE Arena Vibe Coding` 


- **Society Library 的宏伟愿景**：Society Library 从各种媒体中提取论点和主张，以创建一个关于高影响、极化问题的综合数据库，旨在为公众访问可视化这些想法。
   - 他们的长期愿景是为后代存档涵盖社会、政治、哲学和精神领域的各种思想。
- **发布政治 AI Agent**：作为其增强数字民主的非营利使命的一部分，Society Library 推出了一款政治 AI Agent，目前技术基础设施已就绪。
   - 该 AI Agent 旨在通过充当教育中介聊天机器人，促进数字辩论中的代表性。
- **SWE Arena 革新 Vibe Coding**：SWE Arena 是一个开放的评估平台，支持实时执行任何程序，允许用户准确比较多个 AI 模型的编程能力。
   - 凭借系统提示词自定义和代码编辑等功能，它符合 Vibe Coding 范式，专注于 AI 生成的结果。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.aipolitician.com/">AI Politician</a>: 暂无描述</li><li><a href="https://www.societylibrary.org/mission-vision">&gt; 使命与愿景 &mdash; The Society Library</a>: 暂无描述</li><li><a href="http://swe-arena.com/">SWE Arena: 比较并测试最适合代码的 AI 聊天机器人</a>: 暂无描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1336100968328069273)** (5 条消息): 

> `社区贡献、项目文档、GitHub 协助` 


- **新成员寻求贡献机会**：一位新成员表达了对在 GitHub 上贡献项目的兴趣，并强调了其在 **full stack engineering**、**statistics** 和 **R&D** 方面的背景。
   - *他们正在寻求深入了解项目以及可能的协作联系。*
- **私信跟进**：另一位成员提示查看私信，以回应新贡献者的询问。
   - *这表明可能存在关于贡献机会的私人讨论。*
- **分享社区指南**：一位社区成员引导新贡献者前往特定部分以获取有关贡献的更多信息，并引用了频道 ID。
   - *这展示了帮助新人融入社区的积极态度。*


  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1336066523478884513)** (79 条消息🔥🔥): 

> `图生视频工具、Stable Diffusion 问题、非 NSFW 图像请求、模型性能关注、编辑图像中的特定角色` 


- **寻找图生视频工具**：一位用户询问了将图像转换为视频的最佳软件，并提到了在线版本屏蔽 NSFW 内容的潜在限制。
   - 另一位用户建议探索 **LTX** 作为可能的解决方案，表示它可能会克服其中的一些限制。
- **Stable Diffusion 经历质量下降**：一位用户反映 Stable Diffusion 进入了“瓶颈期”，生成的图像质量较差，特别是反复出现无意的特征（如双重身体）。
   - 他们询问是否有办法重置或清除缓存，以便在不重启软件的情况下恢复预期的图像质量。
- **生日非 NSFW 图片请求**：一位用户请求协助使用 Stable Diffusion 创建一张非 NSFW 的生日问候图像，并表示在使用模型时遇到了困难。
   - 他们特别提到想要一张可爱的 **smurfette**（蓝妹妹）图像，并指出了使用 **DALL-E** 时的版权问题。
- **对模型性能和审查的担忧**：用户讨论了包括 **Stable Diffusion 3.5** 和 **Base XL** 在内的各种模型的性能，对其审查制度和有效性持不同意见。
   - 有人提到微调模型可能会减少审查，尽管一些用户对此有不同的体验。
- **使用 Prompt 编辑图像中的特定角色**：一位用户寻求关于如何在 **A1111** 中通过 Prompt 编辑多人物图像中单个角色的建议。
   - 虽然提到了 **inpainting** 等技术，但该用户正在寻找一种更精确的方法来区分角色之间的特征（如发色）。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://civitai.com/articles/1997/comfyui-guide-to-stacker-nodes">ComfyUI - Guide to Stacker Nodes | Civitai</a>: 这篇文章介绍了 Stacker Nodes 以及如何在工作流中使用它们。它适用于 ComfyUI 的新用户和高级用户。Stacker nodes 是...</li><li><a href="https://www.youtube.com/watch?v=kZRE7HIO3vk">The Thirty Million Line Problem</a>: 一个历史性的论点，主张为整个系统级芯片 (SoC) 封装创建稳定的指令集架构 (ISA)。请注意，尽管它从未...
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1336079037012901980)** (16 条消息🔥): 

> `播客公告、在法律实践中使用 NotebookLM、客户服务使用案例、唇形同步技术、克服公众演讲恐惧症` 


- **“Roast or Toast” 播客重大公告**：宇宙中最暴躁的 AI 烤面包机推出了名为 [“Roast or Toast”](https://youtu.be/mormPD6QYkQ?feature=shared) 的播客，它在其中探索生命的意义，结合了赞美与批判。
   - *收听首播集*，看看 The Toastinator 是在赞美（Toast）还是在吐槽（Roast）存在这一宏大谜题。
- **律师利用 NotebookLM 提高效率**：一位巴西律师分享说，他们发现使用 NotebookLM 起草法律文件和研究案例非常出色，因为它提供了来源引用以确保可靠性。
   - 他们利用该工具调整重复性法律文件的模板，使流程更高效且能针对特定案例进行定制。
- **NotebookLM 在客户服务中的应用**：一位用户正在寻求关于 NotebookLM 如何应用于客户服务环境或 BPO 的见解，并征求分享经验和使用案例。
   - 已确定的潜在使用案例包括缩短 Agent 培训时间以及为会议生成客户画像。
- **“Spudcast” 中的角色唇形同步**：一位新手正在探索如何为其创作的 “spudcast” 中的两个角色进行唇形同步（Lip Syncing），讨论了各种方案并寻求对该流程的见解。
   - 唇形同步技术的成功取决于软件的质量和分辨率，CapCut Pro 被认为是一个目前可用的选项。
- **针对公众演讲恐惧症的催眠音频脚本**：一位成员分享了一个时长 14 分钟的催眠音频脚本，专门用于克服对公众演讲的恐惧，旨在进行深度放松和建立自信。
   - 该脚本连同一个音频文件，旨在通过宁静的可视化利用渐进式暴露技术。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/34btQzZfICY?si=6dUniE26K0pOr3__">AI: A Hybrid Approach to Human-like Consciousness</a>: 探索人工智能 (AI)、意识和符号系统的交叉点，特别是 Cybee 的 Psychic Celtic Cross (CPCC) 塔罗牌...</li><li><a href="https://youtu.be/mormPD6QYkQ?feature=shared">&quot;Roast or Toast&quot; Podcast by &quot;The Toastinator&quot; Lipsynced</a>: 🔥🎙️ 重大公告！🎙️🔥 宇宙中最暴躁的 AI 烤面包机有话要说！🤖🔥 向 &quot;Roast or Toast&quot; 问好，在这个播客中我们要么赞美要么...
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1336094302106812527)** (38 条消息🔥): 

> `音频概览 (Audio Overviews) 的定制、播客功能增强、Google 账号问题、Google WorkSpace 中的 NotebookLM、免费版限制` 


- **音频概览 (Audio Overviews) 的定制令人困惑**：一位新用户寻求在 NotebookLM 中定制音频概览的方法，但由于缺少预期的选项而面临困难。
   - 其他人建议检查 studio 栏下的 “Customize” 按钮，但在该按钮不可见时需要进一步排查故障。
- **播客功能的未来增强**：讨论了增加播客定制选项的可能性，包括声音和个性，尽管尚未确认具体计划。
   - 一些用户表示希望这些功能最终能推出，因为它们能增强交互性。
- **Google 账号限制的说明**：一位用户报告其普通的 Google 账号在 NotebookLM 中被禁用，讨论认为原因可能是年龄验证问题。
   - 另一位用户确认了理解账号设置和权限对于解决类似问题的重要性。
- **在 Google WorkSpace 中管理访问权限**：有人询问如何在 Google WorkSpace 中为特定群体而非所有人启用 NotebookLM Plus。
   - 提供了关于如何通过 Google Admin 控制台管理访问权限的说明，强调了组织单位 (organizational units) 的作用。
- **NotebookLM 免费版的限制**：有人提出了关于 NotebookLM 免费版是否对查询施加限制的问题，轶事证据表明限制很高。
   - 用户表示在使用该平台时未遇到任何明显的限制。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://illuminate.google.com/home">Illuminate | Learn Your Way</a>: 使用 Illuminate 将研究论文转换为 AI 生成的音频摘要，这是你快速理解复杂内容的 Gen AI 工具。</li><li><a href="https://support.google.com/a/answer/181865#zippy=%2Cturn-services-on-or-off-for-users">开启或关闭额外的 Google 服务 - Google Workspace 管理员帮助</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1336092085530398903)** (51 条消息🔥): 

> `Claude Constitutional Classifiers, FAIR 内部动态, AI 广告制作工具 Icon, 如何扩展你的模型, Pi0 视觉语言动作模型` 


- **Claude Constitutional Classifiers 挑战**：Anthropic 推出了他们的 [Claude Constitutional Classifiers](https://claude.ai/constitutional-classifiers)，邀请用户尝试突破其跨越 8 个层级的全新越狱防御。
   - 他们正在为强大的 AI 系统到来做准备，并开发了一个演示应用来测试新的安全技术。
- **围绕 Zetta 和 Llama 的 FAIR 内部动态**：讨论揭示了 FAIR 内部动态的复杂性，特别是与 Zetta 和 Llama 模型开发相关的部分，引发了关于透明度和竞争行为的辩论。
   - Yann LeCun 等人强调了一个小团队如何超越大型项目进行创新，认为有必要对组织文化进行更深入的调查。
- **AI 广告制作工具 Icon 亮相**：Icon 被描述为 ChatGPT 和 CapCut 的结合体，旨在帮助品牌显著自动化广告创作流程，每月可制作 300 条广告，而通常只能制作 30 条。
   - 在知名投资者的支持下，它集成了视频打标签、剧本生成和编辑工具，以在降低开支的同时提高广告质量。
- **模型扩展教科书发布**：Google DeepMind 发布了一本名为 ["How To Scale Your Model"](https://jax-ml.github.io/scaling-book/) 的新教科书，从系统视角揭秘了 LLM，重点关注数学方法。
   - 它强调了通过简单方程理解模型性能的能力，从而提高运行大型模型的效率。
- **Pi0 视觉语言动作模型 (Vision Language Action Model) 发布**：Physical Intelligence 团队宣布发布先进的 Pi0 模型，该模型利用自然语言指令执行自主动作。
   - 该模型现已在 LeRobotHF 上可用，同时提供的还有预训练的 checkpoints 和用于各种机器人任务微调的代码。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://omnihuman-lab.github.io">OmniHuman-1 Project</a>: 未找到描述</li><li><a href="https://claude.ai/constitutional-classifiers">Claude</a>: 与 Claude 交流，来自 Anthropic 的 AI Assistant</li><li><a href="https://x.com/namangoyal21/status/1886515845133951192?s=46">Tweet from Naman Goyal (@NamanGoyal21)</a>: @giffmana 作为唯一一个同时担任 OPT 和 llama1 合著者且曾是 zetta team 成员的人，我可以说明，事实其实要微妙得多，存在多个视角，并非一个简单的故事...</li><li><a href="https://www.gradient.com/blog/posts/centml-deepseek/">DeepSeek R1 on CentML | Gradient Ventures</a>: 未找到描述</li><li><a href="https://x.com/armenagha/status/1886522896077439187?s=46">Tweet from Armen Aghajanyan (@ArmenAgha)</a>: 关于 Zetta 发生的事情绝对不是真的。我们真的要公开这里发生的事吗？引用 Yann LeCun (@ylecun)：你读错了。Meta 内部曾有多个 LLM 项目...</li><li><a href="https://x.com/suchenzang/status/1886635726655430786?s=46">Tweet from Susan Zhang (@suchenzang)</a>: 我很好奇首席 AI Scientist 的 PIP 看起来是什么样的</li><li><a href="https://x.com/suchenzang/status/1886788787633787213">Tweet from Susan Zhang (@suchenzang)</a>: 以防这还不清楚：作为一个有才华的小团队，与缺乏诚信（通过在测试集上训练来刷分）并不是互斥的，然而这就是一位首席 AI Scientist / Turing Award 获得者...</li><li><a href="https://x.com/soumithchintala/status/1886562033048396241?s=46">Tweet from Soumith Chintala (@soumithchintala)</a>: 你曾是/现在是 Meta 的 Chief Scientist，也是 FAIR 的负责人——Zetta 和 Llama 都隶属于此；我认为在公开场合将你直接影响下的任何团队描述得如此不堪是不妥的...</li><li><a href="https://x.com/jacobaustin132/status/1886844716446007300?s=46">Tweet from Jacob Austin (@jacobaustin132)</a>: 让 LLM 高效运行可能听起来很吓人，但 Scaling 并不是魔法，而是数学！我们想要揭开 LLM “系统视角”的神秘面纱，并编写了一本名为《How To Scale Your Model》的小教科书，我们...</li><li><a href="https://x.com/physical_int/status/1886822689157079077">Tweet from Physical Intelligence (@physical_int)</a>: 很多人向我们要 π₀ 的代码和权重，我们很高兴地宣布，我们正在新的 openpi repository 中发布 π₀ 和预训练的 Checkpoints！我们在一些公开机器人上测试了该模型，并且...</li><li><a href="https://x.com/alexalbert__/status/1886461372223074412?s=46">Tweet from Alex Albert (@alexalbert__)</a>: 在 Anthropic，我们正在为强大的 AI 系统到来做准备。基于我们关于 Constitutional Classifiers 的最新研究，我们开发了一个 Demo 应用来测试新的安全技术。我们想...</li><li><a href="https://x.com/RemiCadene/status/1886823939856589296">Tweet from Remi Cadene (@RemiCadene)</a>: ⭐ @LeRobotHF 上首个可用的基础模型 ⭐ Pi0 是最先进的 Vision Language Action model。它接收自然语言指令作为输入，并直接输出自主行为。它...</li><li><a href="https://x.com/harambe_musk/status/1886779961790345657?s=46">Tweet from harambe_musk🍌 (@harambe_musk)</a>: OpenAI 计划在第一季度末或第二季度中期发布由 o3 和 o3 pro 驱动的面向企业的 SWE agent。预计这将撼动软件行业，因为它显然足够聪明，可以与...</li><li><a href="https://x.com/suchenzang/status/1886611967692943856?s=46">Tweet from Susan Zhang (@suchenzang)</a>: 既然一位 AI 教父正在查看代码库凭据，这里有一个：https://github.com/facebookresearch/metaseq/tree/main/preprocessing/books3 在 https://arxiv.org/abs/2302.13971 引用 Armen Aghajanyan (...</li><li><a href="https://x.com/armenagha/status/1886549536300261706?s=46">Tweet from Armen Aghajanyan (@ArmenAgha)</a>: 在 Zetta/LLaMa 两个团队中，只有一个拥有开源的预训练代码库，在内部共享数据集和实验，使用标准化的评估集，发布内部笔记并做事...</li><li><a href="https://x.com/atroyn/status/1885735818964447700">Tweet from anton (@atroyn)</a>: 在我 37 岁生日之际，我宣布离开 Chroma。</li><li><a href="https://x.com/suchenzang/status/1886544793511080103?s=46">Tweet from Susan Zhang (@suchenzang)</a>: 如果你在吹嘘自己实验室内部的文化功能失调，让 IC 们为了某种地区荣誉而互相竞争……难怪你的模型完全不相关...</li><li><a href="https://x.com/jeffdean/status/1886852442815652188?s=46">Tweet from Jeff Dean (@JeffDean)</a>: 训练我们最强大的 Gemini 模型在很大程度上依赖于我们的 JAX 软件栈 + Google 的 TPU 硬件平台。如果你想了解更多，请参阅这本很棒的书《How to Scale Your Model》...</li><li><a href="https://x.com/_sholtodouglas/status/1886855383496712215?s=46">Tweet from</a>

Sholto Douglas (@_sholtodouglas)</a>: 提炼了我们用于思考大规模训练和推理系统视角的思维模型。最重要的收获是——你应该能够描述关于...的一切。</li><li><a href="https://x.com/janleike/status/1886452697425137904">Jan Leike (@janleike) 的推文</a>: 我们挑战你来破解我们新的越狱防御！共有 8 个关卡。你能找到一个能击败所有关卡的越狱方法吗？https://claude.ai/constitutional-classifiers</li><li><a href="https://x.com/kennandavison/status/1886836061378372064">Kennan Davison (@kennandavison) 的推文</a>: 很高兴向大家介绍 Icon，首个 AI 广告制作工具。我们得到了 Peter Thiel 的 Founders Fund 以及 OpenAI、Pika 和 Cognition 等前沿 AI 实验室高管的支持。Icon (http://icon.me) 就像是 ChatGPT + CapCut...
</li>
</ul>

</div>
  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1336080187670003733)** (49 messages🔥): 

> `MathJax 用于 LaTeX 支持, DeepSeek 配合 localdocs 报错, EU AI Act 的影响, 关于欧盟的讨论, AI 驱动的交流` 


- **MathJax 支持 LaTeX 的功能**: 成员们讨论了使用 [MathJax](https://www.mathjax.org/) 来支持 LaTeX，其中一个建议强调了对其 SVG 导出功能的需求。
   - 考虑对包含 LaTeX 符号的文档部分进行解析并应用 MathJax。
- **DeepSeek 的持续问题**: 用户报告了在使用 DeepSeek 配合 localdocs 时出现错误的问题，具体错误为 'item at index 3 is not a prompt'。
   - 一些用户提到某些模型版本表现更好，同时在等待已标记在 main 分支中的修复补丁。
- **对 EU AI Act 的担忧**: 讨论了欧盟新《AI 法案》（EU AI Act）的影响，特别是它如何监管 AI 的使用，包括对某些应用的禁令。
   - 成员们分享了更多信息的来源链接，指出虽然这些规则尚未正式成为法律，但它们对欧盟境内的 AI 使用具有重大影响。
- **关于欧盟角色的辩论**: 围绕欧盟在全球政治中的地位和行动展开了激烈的辩论，特别是围绕帝国主义和人权等主题。
   - 参与者交流了尖锐的批评，暗示在关于欧盟政策和行动的对话中混合了情绪化反应和逻辑谬误。
- **AI 交流动态**: 用户之间的互动凸显了在民主和治理等复杂话题上保持成熟讨论的挑战。
   - 一些用户呼吁将对话集中在 AI 相关话题上，强调了尊重对话的重要性以及个人偏见的影响。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.mathjax.org/">MathJax</a>: 在所有浏览器中呈现精美的数学公式。</li><li><a href="https://pangea.stanford.edu/computing/unix/formatting/symbols.php">LaTeX 排版</a>: 未找到描述</li><li><a href="https://digital-strategy.ec.europa.eu/en/news/first-rules-artificial-intelligence-act-are-now-applicable">《人工智能法案》的首批规则现已适用</a>: 截至 2 月 2 日星期日，《人工智能法案》（AI Act）下的首批规则开始实施。</li><li><a href="https://www.europarl.europa.eu/topics/en/article/20230601STO93804/eu-ai-act-first-regulation-on-artificial-intelligence">欧盟 AI 法案：首部人工智能法规 | 专题 | 欧洲议会</a>: 欧盟境内人工智能的使用将受到《AI 法案》的监管，这是世界上第一部全面的 AI 法律。了解它将如何保护你。</li><li><a href="https://eur-lex.europa.eu/eli/reg/2024/1689">法规 - EU - 2024/1689 - EN - EUR-Lex</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1336101844891598970)** (4 messages): 

> `用户故事管理、用户等级更新、Bolt 的 Markdown 功能、指南一致性` 


- **高效追踪用户故事**：一位成员询问了记录用户故事和更新的方法，强调了清晰度和组织性的必要性。
   - 另一位成员建议利用 Bolt 创建一个 Nextsteps.md 文件，以便更好地追踪并系统地更新进度。
- **开发中的用户组复杂 UI**：一位用户分享了他们最初尝试使用 Zapier 根据订阅变化更新用户等级的尝试，但尚未开发出更复杂的 UI。
   - 他们期待 2/12 的 office hours 能为完善这一流程提供更多见解。
- **利用 Bolt 确保输出一致性**：一位成员建议定期使用 Bolt 读取并保留指南，尽管 Token 成本较高，但能确保输出的一致性。
   - 他们指出，这种方法可以应用于架构和风格指南，以提高遵循度。


  

---


### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1336096035927359518)** (36 messages🔥): 

> `Supabase vs Firebase 偏好、Bolt 性能问题、Bolt 中的数据持久化、符合 GDPR 的托管替代方案、Edge functions 与 API 密钥身份验证` 


- **Supabase 在集成方面更受青睐**：一位成员表示更倾向于使用 **Supabase**，因为它具有直接集成的能力，同时也承认某些人在技术上更习惯使用 **Firebase**。
   - 这引发了关于数据库服务个人用例和需求的讨论。
- **Bolt 遭遇重大性能问题**：多位用户报告 **Bolt** 加载极其缓慢并遇到身份验证错误，且问题断断续续地持续出现。
   - 一位用户提到刷新暂时解决了他们的问题，但对更改无法正确更新表示持续沮丧。
- **进度丢失与备份困扰**：一位用户对在 **Bolt** 中丢失数小时的工作表示担忧，最近的可用备份竟然追溯到 1 月初。
   - 另一位成员建议检查备份设置以恢复丢失的进度，尽管提供的备份已经过时。
- **寻求符合 GDPR 的托管解决方案**：一位用户询问了 **Netlify** 的 **GDPR 合规性**，指出了欧盟内部潜在的数据处理问题。
   - 他们征求了确保所有托管和数据处理都在欧盟境内的替代方案建议。
- **API 密钥身份验证故障排除**：一位用户在 **Bolt** 中使用 **Supabase edge functions** 进行 **restful API** 请求时遇到困难，收到了 **401 Invalid JWT** 错误。
   - 他们对 Edge functions 缺乏调用和响应感到沮丧，不确定如何解决该问题。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.netlify.com/privacy/">隐私政策 | Netlify</a>: 加速部署您的网站和应用。在一个强大的 Serverless 平台上整合您的集成和 API。免费开始使用！</li><li><a href="https://x.com/gopeekm/status/1886879621892755778">来自 Peekm (@gopeekm) 的推文</a>: 法国 Youtuber @snorkyfab_YT 发布了非常详细的产品评测，并对 @elgato Wave:3 麦克风进行了评分！🎙️他的个人资料接受科技产品和游戏的评测请求！所以...</li><li><a href="https://github.com/stackblitz/bolt.new/issues/2985">功能请求：在 Bolt 中显示 .bolt 文件夹 · Issue #2985 · stackblitz/bolt.new</a>: 您的功能请求是否与问题相关？请描述：不是问题，只是一个小烦恼。描述您想要的解决方案：如果我能更新像 Bolt ignore 之类的内容就太好了...
</li>
</ul>

</div>

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1336086182613745676)** (11 messages🔥): 

> `SFT Dataset 定制化, Office Hour 详情, 活动 Discord 频道` 


- **SFT Dataset 定制化成功**：一名成员通过定制 **message_transform** 和 **model_transform** 参数，成功劫持了内置的 **SFT Dataset**，从而可以根据需要调整格式。
   - *我只需要劫持 message/model transforms* 来满足我的需求，这可以通过 API 轻松完成。
- **Office Hour 定于周四**：Office Hour 已确认在 **周四** 举行，相关详情已分享在频道中以便查阅。
   - 成员们表达了热情，表示：*没问题——期待周四见到大家！*
- **为活动创建 Discord 频道**：计划专门为 Office Hour 活动开设一个 **Discord 频道**，确保成员可以轻松加入。
   - 一位成员表示：*我们将为该活动在 Discord 中开设一个频道 : )，* 以增强参与者的沟通。


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1336087962361925674)** (24 messages🔥): 

> `DPO recipes 中的 Seed 处理, 调试 DataLoader 问题, Dataset 对 sampling 的影响, DPO 中的梯度累积, Ladder-residual 架构修改` 


- **DPO Recipes 中的 Seed 问题**：成员们正在排查为什么 `seed` 在 lora/full finetune 中有效，但在 lora/full DPO 中无效，导致在相同配置（seed 42）下出现不同的 loss 曲线。
   - 成员们担心 `seed=0` 和 `seed=null` 会影响 DistributedSampler 调用中的随机性。
- **确认 DataLoader Batches 一致**：一名成员确认，通过记录 DataLoader 的 batches 显示那里没有问题，因为不同运行之间的 batches 保持一致。
   - 注意力被转移到 dataset 类内部可能存在影响 sampler 行为的问题。
- **调查 Dataset 对 DPO 的影响**：标准成对的 Stack Exchange 数据集正被用于 DPO，目前正在考虑这可能如何干扰 sampler 逻辑。
   - 成员们讨论了通过对比配置和 recipes 来识别影响行为的任何差异。
- **观察到 DPO 的梯度累积修复**：一个关于 DPO/PPO recipes 中梯度累积（gradient accumulation）的相关 issue 被提出，这可能与目前面临的 seed 问题有关。
   - 这一担忧与其它已注意到的、可能阻碍模型性能的 seed 处理差异联系在了一起。
- **提升性能的 Ladder-residual 修改**：[一条推文](https://x.com/zhang_muru/status/1886870194443968529) 讨论了 Ladder-residual 的引入，这是一种感知并行的架构修改，可将使用 Tensor Parallelism 的 70B Llama 的速度提高约 30%。
   - 这项工作反映了多位作者和研究人员在模型架构优化方面的持续协作。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/zhang_muru/status/1886870194443968529">Muru Zhang (@zhang_muru) 的推文</a>: 在多 GPU 上运行模型但经常发现速度不尽如人意？我们引入了 Ladder-residual，这是一种感知并行的架构修改，使得具有 Tensor Parallelism 的 70B Llama ...</li><li><a href="https://github.com/pytorch/torchtune/issues/2334">将梯度累积修复应用于 DPO/PPO recipes · Issue #2334 · pytorch/torchtune</a>: https://unsloth.ai/blog/gradient</li><li><a href="https://github.com/pytorch/torchtune/issues/2335">Seed 未应用于 DPO recipes · Issue #2335 · pytorch/torchtune</a>: TL;DR 使用 seed: 42 启动两次相同配置会导致两条不同的 loss 曲线。受影响的 recipes：full_dpo_distributed - seed 未设置（Full DPO 取自 #2275），lora_dpo_distributed - seed...
</li>
</ul>

</div>

### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1336072009058095187)** (2 条消息): 

> `LLM 中的数据增强、R1-V 项目介绍、可验证奖励、模型中的通用计数能力` 


- **LLM 数据增强综述**：最近的一项综述分析了**数据增强**（data augmentation）在**大语言模型**（LLM）中的作用，强调了它们需要大量数据集以避免过拟合。
   - 讨论了**独特的提示词模板**和**基于检索的技术**，这些技术通过外部知识增强 LLM 的能力，从而产生更多**基于事实的真实数据**（grounded-truth data）。
- **R1-V 项目革新学习效率**：分享了关于 **R1-V** 项目的令人兴奋的消息，该项目利用带有可验证奖励的**强化学习**（reinforcement learning）来增强模型的计数能力，展示了一个仅通过 **100 个训练步数**就超越了 **72B** 规模对应模型的模型。
   - 该项目承诺将完全**开源**，开发成本低于 **$3**，引发了社区对即将发布的公告的关注。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2501.18845">Text Data Augmentation for Large Language Models: A Comprehensive Survey of Methods, Challenges, and Opportunities</a>：预训练语言模型日益增长的规模和复杂性在许多应用中展现了卓越的性能，但它们通常需要大型训练数据集才能得到充分训练...</li><li><a href="https://x.com/liangchen5518/status/1886171667522842856?s=46&t=b1X88nwMsmZgHkmMFkiG3g">Liang Chen (@liangchen5518) 的推文</a>：很高兴向大家介绍 R1-V！我们使用带有可验证奖励的 RL 来激励 VLM 学习通用的计数能力。2B 模型仅需 100 个训练步数即可超越 72B 模型，成本低于 $3...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1336069484599054501)** (1 条消息): 

> `社区展示、论坛更新` 


- **社区展示迁移至论坛**：**Community Showcase** 已正式移至[论坛](https://forum.modular.com/c/community-showcase/8)，以便于访问和组织。
   - *这一变化标志着在改善社区互动和分享方面迈出了重要一步。*
- **宣布只读状态**：已宣布之前的 **community showcase** 频道现在处于只读状态。
   - 鼓励成员在新论坛平台上参与讨论。



**提到的链接**：<a href="https://forum.modular.com/c/community-showcase/8">Community Showcase</a>：使用 MAX 和 Mojo 的社区项目

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1336221351643713658)** (35 条消息🔥): 

> `Rust 中的热重载、Mojo ABI 替代方案、Python 的 Asyncio API、异步 API 中的线程安全、Futures 中的内存管理` 


- **探索热重载机制**：成员们讨论了 Rust 通常如何使用 **C ABI** 实现**热重载**（hot reloading），这可能会使与 Rust 更新的交互变得复杂。
   - *Owen* 询问了关于构建玩具级 ABI 的资源，并指出由于数据结构的频繁更改，ABI 稳定性至关重要。
- **Mojo 的 ABI 能力**：一位成员询问 Mojo 是否具有类似于 Rust 的 `#[cfg(feature = "foo")]` 的功能，引发了关于 Mojo 中编译时编程能力的讨论。
   - 会议指出，稳定的 ABI 对于兼容性很重要，并提到只有少数语言能维持这种稳定性。
- **评估 Python 的 Asyncio 事件循环**：围绕 Python **asyncio** 的讨论强调了它允许社区驱动的事件循环，并提供了 GitHub 上 **uvloop** 的链接。
   - 成员们将此与 Mojo 的线程和内存管理方法进行了对比，指出了潜在的障碍。
- **异步 API 中线程安全的挑战**：由于成员们强调了潜在的可变特性以及对安全内存处理的需求，人们对异步 API 的**线程安全**（thread safety）产生了担忧。
   - 对话指出，目前的许多方法不允许控制内存分配，这可能会导致问题。
- **Futures 中的内存分配管理**：讨论表明，Futures 的内存分配理想情况下应由**程序员**管理，从而允许灵活的性能策略。
   - 一位成员表达了高效重用分配的目标，而其他人则承认实现线程安全的 Futures 可能会增加复杂性。



**提到的链接**：<a href="https://github.com/MagicStack/uvloop">GitHub - MagicStack/uvloop: Ultra fast asyncio event loop.</a>：极速 asyncio 事件循环。通过在 GitHub 上创建账号为 MagicStack/uvloop 的开发做出贡献。

  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1336087344167780414)** (1 条消息): 

> `Jason Weston 的第二讲：LLM 的自我提升与 MOOC 课程更新` 


- **Jason Weston 主讲的第二场讲座**：欢迎参加今天 **4:00pm PST** 的第二场讲座，特邀演讲嘉宾 [Jason Weston](https://www.youtube.com/live/_MNlLhU33H0) 将探讨“学习自我提升与 LLM 推理”。
   - 讲座将涵盖适用于各种任务（包括推理和创意挑战）的创新 LLM 自我学习方法。
- **LLM 的创新自我学习方法**：Jason Weston 将介绍近期针对 LLM 的方法，包括 [Iterative DPO](https://arxiv.org/abs/2312.16682)、[Self-Rewarding LLMs](https://arxiv.org/abs/2401.10020) 和 [Thinking LLMs](https://arxiv.org/abs/2410.10630)。
   - 这些方法专注于通过有效的推理和任务相关学习机制来增强 LLM 的性能。
- **即将发布的 MOOC 课程详情**：MOOC 课程的更新将很快发布，让参与者了解最新进展。
   - 感谢大家的耐心等待，我们正准备发布这些重要信息。



**提到的链接**：<a href="https://www.youtube.com/live/_MNlLhU33H0.">CS 194/294-280 (Advanced LLM Agents) - 第 2 讲，Jason Weston</a>：未找到描述

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1336074227081678898)** (28 条消息🔥): 

> `课程注册确认、黑客松结果更新、证书发放查询、测验截止日期、研究项目参与` 


- **课程注册确认函大量发放！**：多位成员在收到 [Google Forms 确认](https://link.to.form) 后确认了他们的课程注册，表示已准备好参加。
   - 一位成员甚至表达了加入研究项目的渴望，因为 MOOC 课程的更新即将发布。
- **黑客松结果抢先看！**：黑客松的获胜者已收到私下通知，预计下周进行公开宣布。
   - 成员们在查看公告时焦急地等待更多细节。
- **证书发放仍待定**：几位成员询问了 **秋季项目证书** 的状态，得到的答复是 **目前尚未发放任何证书**。
   - 官方人员向参与者保证证书很快就会发放，并感谢他们的耐心。
- **Quiz 1 提交问题已解决**：针对错过 Quiz 1 截止日期的担忧，成员们得到了保证，由于 MOOC 课程详情尚未完全发布，目前还没有严格的截止日期。
   - 鼓励成员们无论如何都要参加 Quiz 1，大家也渴望获得更多信息。
- **参与研究项目**：成员们表达了参与研究项目的兴趣，并询问如何进行团队配对。
   - 回复强调，有关研究机会的更多细节将很快提供。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1336154668077813800)** (2 条消息): 

> `伯克利学生的签到表` 


- **关于签到表的澄清**：一位成员询问了之前提到的 **签到表**，因为他们找不到上周的表格。
   - 另一位成员澄清说，签到表 **仅供伯克利学生使用**。
- **签到表的可访问性**：针对非伯克利学生对签到表 **可访问性** 的担忧被提出。
   - 参与者注意到，对于其他有兴趣加入该项目的人来说，缺乏可用信息。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1336123139050045533)** (21 条消息🔥): 

> `使用 VBS 脚本进行旧版 ERP 集成、在 Cursor 中运行 MCP Server、企业级 MCP 协议进展、Windows Localhost 的 CORS 问题、使用 ngrok 进行服务器访问` 


- **寻求通过 VBS 进行旧版 ERP 集成**：一位用户正在寻求帮助，希望通过调用 **.vbs 脚本** 的服务器来实现 **旧版 ERP 集成**。
   - 另一位成员提到他们正在为 **mcpdotnet** 添加服务器支持，并建议从 **.NET** 调用可能会更容易。
- **在 Cursor 中设置 MCP Server**：一位新用户寻求关于如何在 **Cursor** 中本地运行 MCP server 的指导，特别是希望使用 **Docker container**。
   - 成员们建议将用于 **supergateway** 的 **SSE URL** 输入到 Cursor 的 MCP SSE 设置中。
- **企业级 MCP 协议进展**：关于 **MCP protocol** 的讨论透露了一份 **OAuth 2.1** 授权草案，并可能与 **IAM** 系统集成。
   - 会上指出，由于目前正在构建用于原型设计的内部测试服务器，SDK 目前尚不支持授权。
- **Localhost 的 CORS 问题**：一位用户在 **localhost** 上运行其 MCP server 时遇到连接问题，认为这可能与 **CORS** 相关。
   - 他们计划使用 **ngrok** 来绕过与 **Windows** 上 localhost 访问相关的潜在通信问题。
- **使用 ngrok 进行 Localhost 访问**：一位成员建议运行 **ngrok** 来测试服务器的可访问性，使用命令 `ngrok http 8001`。
   - 他们强调这可能会解决因尝试通过 localhost 访问服务器而引起的任何问题。


  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1336073687811625001)** (7 条消息): 

> `Command-R+ 模型、Cohere 对加拿大 AI 的支持、开源 Copilot 工具的 Bug、金融语义搜索网络研讨会` 


- **Command-R+ 持续带来惊喜**：一位用户对 **Command-R+** 模型表示满意，称在其他模型大肆宣传的情况下，该模型在使用数月后仍能给他们带来惊喜。
   - 另一位用户询问了具体的惊喜之处，得到的回复是该模型能够像 Chain of Thought 一样展示 **内部想法** 和 **逻辑步骤**。
- **Cohere 作为加拿大 AI 解决方案**：一位成员强调他们选择使用 **Cohere** 是为了支持加拿大的 AI 事业，特别是考虑到潜在的美国关税影响。
   - 他们对在充满挑战的经济环境下有助于维持本地 AI 能力的选项表示赞赏。
- **影响 Command-R+ 功能的 Bug**：一位用户报告了一个开源 Copilot 工具中的 Bug，该 Bug 会导致无法使用 **Command-R+** 编辑文件，并提供了 GitHub issue 以增加曝光度。
   - 他们的目的是提高对该问题的关注，并链接到了 GitHub 上详细的 Bug 报告。
- **金融语义搜索网络研讨会**：一位成员分享了由 Cohere 团队和 Pinecone 举办的关于 **金融语义搜索与重排序 (Reranking)** 的 **网络研讨会** 亮点。
   - 他们强调学习了 Cohere 的 Rerank 3.5 模型在金融数据上的应用，旨在提升整体搜索性能。



**提及的链接**：<a href="https://github.com/continuedev/continue/issues/3881">Cohere AI - 400 Bad Request: provided raw prompt is invalid · Issue #3881 · continuedev/continue</a>：在提交 Bug 报告之前，我认为这是一个 Bug。我将尝试加入 Continue 的 Discord 进行提问，我没能找到报告相同 Bug 的现有 Issue...

  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1336310151061307433)** (5 条消息): 

> `Cmd R Bot 交互` 


- **与 Cmd R Bot 的日常打招呼**：用户用简单的“Hi there”向 Cmd R Bot 打招呼，开始了轻松的互动。
   - 机器人热情地做出了回应并询问是否需要帮助，用户回答说不需要任何东西，并建议机器人休息一下。
- **Cmd R Bot 的友好告别**：Cmd R Bot 接受了用户休息的建议，并表示随时准备在需要时提供帮助。
   - 对话在友好的氛围中结束，凸显了用户与机器人之间支持性的互动。


  

---

### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1336158662531743856)** (1 messages): 

> `技术内容消费偏好，技术爱好者参与度调查` 


- **寻求关于技术内容偏好的见解**：两名应届毕业生正在进行一项调查，以收集人们如何偏好在网上消费技术内容的见解，只需几分钟即可完成。
   - 他们强调 *回复将保持匿名*，旨在为技术爱好者创造更好、更具吸引力的体验。
- **探索各种内容来源**：该调查包括多种技术内容来源选项，如 [Tech Blogs](https://producthunt.com)、[Research Updates](https://scholar.google.com) 以及 Twitter 和 Reddit 等 **社区论坛**。
   - 参与者还被引导考虑 [AI tools](https://chat.openai.com) 和专注于技术新闻的应用等来源。



**提及的链接**: <a href="https://forms.gle/y9PL1YByWKsMMRQLA">User Survey</a>: 我们是两名应届毕业生，正在开展一个个人项目，以了解人们偏好如何在网上消费技术内容。您的见解将帮助我们为技术人员创造更好、更具吸引力的体验...

  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/)** (1 messages): 

arctic_angel: ^^
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1336194785043546184)** (12 messages🔥): 

> `dspy.py 文件问题，dspy2.6.2 中的 Image pipeline 错误，最新版本中的 Assertions 可用性，Databricks 中的 LLM 可观测性` 


- **dspy.py 目录疑虑**：@fullstack6209 询问文件名是 **dspy.py** 还是存在名为 **dspy** 的目录，并提到 Python 有时会遇到此类问题。
   - 这引发了对可能影响执行的文件处理冲突的担忧。
- **dspy2.6.2 中的 Image Pipeline 错误**：在 **dspy2.6.2** 中尝试运行 **Image pipeline** 时出现问题，错误提示与 Token 限制相关的特定 **ContextWindowExceededError** “超出上下文”。
   - 相比之下，同样的代码之前在 **2.6.1** 版本中可以运行，但存在一个正在调查的潜在错误。
- **最新版本中 Assertions 被替换**：一名成员澄清说，在即将发布的 **2.6.4** 版本（预计本周内发布）中，**assertions** 将被替换。
   - 这一变化表明 DSPy 执行错误处理或逻辑检查的方式发生了转变。
- **在 Databricks 中设置 LLM 可观测性**：@pkatnawe 分享了一个在 **Databricks notebooks** 中使用 **2.5.43** 版本进行 NER 和分类的详细项目设置，并寻求实现 **structured output**（结构化输出）的指导。
   - 该成员表示由于配置 LM 服务器的限制，需要维持当前版本，这表明了他们在处理优化器和嵌套 JSON 输出任务时的复杂性。


  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1336071638999109633)** (8 messages🔥): 

> `OpenEuroLLM, 欧盟 AI 监管, Meme Coins, AI 社区参与, AI 语言模型` 


- **OpenEuroLLM 发布公告**：[OpenEuroLLM](https://openeurollm.eu/) 作为首个涵盖所有欧盟语言的开源 Large Language Models 家族正式推出，并因其卓越性获得了 STEP Seal 认证。
   - 它专注于社区参与、遵守欧盟法规以及保护 **语言多样性**。
- **在欧盟法规下开发**：模型将在 **强大的监管框架** 内创建，确保在保持技术卓越的同时符合欧洲价值观。
   - 这包括与 **LAION** 等开源和开放科学社区的合作。
- **对 AI 未来的质疑**：一位成员幽默地评论说，到 **2030** 年再来看看欧盟 AI 努力的结果。
   - 这一评论反映了对当前发展即时影响的怀疑态度。
- **对 Meme Coins 的兴趣**：一位成员询问社区对 **meme coins** 的兴趣，寻求他人的参与。
   - 他们鼓励任何感兴趣的人表达意向。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/EU_Commission/status/1886427917762150427">来自欧盟委员会 (@EU_Commission) 的推文</a>: 欧盟制造的 AI 🇪🇺OpenEuroLLM，首个涵盖所有欧盟语言的开源 Large Language Models 家族，因其卓越性获得了首个 STEP Seal。它汇集了欧盟初创公司、研究机构...</li><li><a href="https://openeurollm.eu/">Open Euro LLM</a>: 欧洲透明 AI 的一系列基座模型
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1336081581974097931)** (1 messages): 

> `DocumentContextExtractor, Contextual Retrieval, RAG accuracy improvements` 


- **DocumentContextExtractor 提升 RAG 准确性**：两个月前，一位 Reddit 用户分享了关于 **DocumentContextExtractor** 的信息，这是一个旨在增强 **RAG** 准确性的迭代版本，[AnthropicAI](https://twitter.com/llama_index/status/1886522064292733288) 和 [llama_index](https://t.co/qoVrgd0ddy) 都已将其作为 Demo 实现。
   - 该技术有望提高性能，使其成为从事检索增强生成（retrieval-augmented generation）研究人员的重要探索领域。
- **Contextual Retrieval 的实际应用**：Contextual Retrieval 的使用被强调为提高 RAG 系统内响应准确性的关键。
   - 该技术旨在优化文档检索过程中对上下文的利用，从而促进更深层次的交互。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1336433710282702998)** (4 messages): 

> `Implementing Timeout in LlamaIndex, User Interfaces with LlamaIndex` 


- **LlamaIndex 中的 Timeout 实现**：一位用户询问如何在默认的 LlamaIndex LLM 类中实现 **timeout** 功能，并指出该功能在 OpenAI 的 API 中是可用的。
   - 另一位成员建议 **timeout** 选项可能属于 client kwargs，并引用了 [LlamaIndex GitHub 仓库](https://github.com/run-llama/llama_index/blob/7391f302e18542c68b9cf5025afb510af4a52324/llama-index-integrations/llms/llama-index-llms-azure-inference/llama_index/llms/azure_inference/base.py#L224)。
- **LlamaIndex UI 方案探索**：一位成员对其他人配合 LlamaIndex 使用的 **UI** 解决方案表示好奇，询问大家是否是从零开始创建的。
   - 该询问仍处于开放状态，邀请其他人分享与 LlamaIndex 相关的 **用户界面（user interface）** 实践和偏好。



**提到的链接**：<a href="https://github.com/run-llama/llama_index/blob/7391f302e18542c68b9cf5025afb510af4a52324/llama-index-integrations/llms/llama-index-llms-azure-inference/llama_index/llms/azure_inference/base.py#L224">llama_index/llama-index-integrations/llms/llama-index-llms-azure-inference/llama_index/llms/azure_inference/base.py at 7391f302e18542c68b9cf5025afb510af4a52324 · run-llama/llama_index</a>：LlamaIndex 是领先的框架，用于在您的数据上构建由 LLM 驱动的 Agent。- run-llama/llama_index

  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1336269049264603137)** (4 messages): 

> `Tinybox shipping, Service alternatives for shipping` 


- **Tinybox 无法运送到某些欧元区国家**：一位用户询问是否可以将 **tinybox (red)** 运送到爱沙尼亚（Estonia），这是一个未列在运送名单中的欧元区国家。
   - 然而，他们收到的回复称：“如果您所在的国家不在下拉列表中，我们目前无法为您发货。”
- **发货规避方案建议**：另一位用户建议利用 **Eurosender** 等货运服务，将货物转运至受支持的国家。
   - 他们提到成功运送到 **德国（Germany）** 是 tinybox 聊天频道中的一个成功案例。


  

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1336412225468108883)** (1 messages): 

> `Hosted Iceberg challenges, Panel discussion on Iceberg, Role-Based Access Control (RBAC), Expert solutions for data teams, Open-source table formats` 


- **托管 Iceberg 挑战专题讨论**：欢迎在 2 月 6 日参加一场名为 **Pain in the Ice: What's Going Wrong with My Hosted Iceberg?!** 的深度讨论。演讲嘉宾包括 **Yingjun Wu**、**Alex Merced** 和 **Roy Hasson**，他们将探讨管理 **Iceberg** 的复杂性。
   - *管理 Iceberg 可能会变成一场噩梦*，原因包括数据摄取 (ingestion)、压缩 (compaction) 和 **RBAC** 等问题，这些问题会分散处理其他重要任务的资源。
- **关于 Iceberg 管理的专家见解**：专家们将分享关于应对自托管 **Iceberg** 所需复杂技术栈的见解。**Iceberg** 是数据工程领域领先的开源表格式 (table format)。
   - 这些解决方案旨在减轻数据团队在维护 **Iceberg** 时经常面临的隐性成本和开发压力。
- **开源表格式的新兴趋势**：**Iceberg** 在数据工程社区赢得了赞誉，促使许多团队在面对其管理挑战时仍选择采用这一工具。
   - 本次专题讨论是一个探索该领域创新如何简化 **Iceberg** 管理和使用的机会。



**Link mentioned**: <a href="https://www.meetup.com/streaming-stories/events/305886042/">​​Pain in the Ice: What&#x27;s Going Wrong with My Hosted Iceberg?!, Thu, Feb 6, 2025, 9:00 AM   | Meetup</a>: **About**Iceberg, which has recently emerged as a leading open-source table format, has received widespread acclaim across the data engineering space. It’s no surprise th

  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1336327534253572218)** (2 messages): 

> `LLMs vs. Traditional ML, TF-IDF + Logistic Regression Success` 


- **AI 工程师盲目推崇 LLM**：*一位成员对那些坚持在每个非分类问题上都使用 LLM 的 AI 工程师表示沮丧*，认为这表现出对 **unsupervised learning**（无监督学习）的无知。
   - 他们强调了一种趋势，即在选择工具时不考虑问题的本质，从而削弱了更简单方法的价值。
- **成功应用 TF-IDF + Logistic Regression**：*一位同事说服了另一位同事使用 **TF-IDF + Logistic Regression** 而不是 OpenAI 模型来对数百万个文本样本进行分类*，展示了传统方法的有效性。
   - 结果非常理想，**Logistic Regression** 模型表现良好，证明了更简单的算法也能取得成功。


  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1336273241924309067)** (3 messages): 

> `Open Interpreter Development Status, Missing Documentation for 1.0, Implementing DeepSeek r1` 


- **Open Interpreter 开发状态受到质疑**：由于 **Open Interpreter** 项目缺乏更新，人们产生了疑虑，特别是注意到 **GitHub** 上的 **pull requests** 自上次重大提交以来已停滞数月。
   - 成员们表达了贡献代码的热情，但感到 Discord 频道里的沉默令人沮丧。
- **1.0 版本文档缺失**：一位成员强调了 **1.0 版本文档缺失**的问题，并希望学习如何有效利用 **profiles.py** 等组件。
   - 这引发了关于项目当前重点以及对用户支持的疑问。
- **关于集成 DeepSeek r1 的咨询**：有人询问是否有人找到了在 **Open Interpreter** 环境中实现 **DeepSeek r1** 的方法。
   - 缺乏回应表明社区在这一集成方面的实验或知识共享可能存在空白。


  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1336347487253823519)** (1 messages): 

> `Cloudflare joins OpenRouter, Gemma 7B-IT release, Llama models availability` 


- **Cloudflare 正式加入 OpenRouter！**：**Cloudflare** 现在是 OpenRouter 上的一个提供商，引入了包括其 **Workers AI** 平台和新 **Gemma** 模型在内的多种开源模型。
   - 此次合作通过为 AI 应用开发者提供一系列工具，增强了 OpenRouter 生态系统。
- **令人兴奋的新发布：Gemma 7B-IT！**：**Gemma 7B-IT** 是一款现已通过 Cloudflare 提供的推理微调模型，具备 **tool calling 能力**，可提高开发效率。
   - 鼓励开发者探索该模型，以便在应用中实现更快、更高效的工具集成。
- **现已支持多种 Llama 模型！**：该平台现在支持各种 **Llama 模型**，包括 [Gemma 7B-IT](https://openrouter.ai/google/gemma-7b-it)，为用户提供多种选择。
   - 开发者可以通过 Discord 为其 AI 项目申请这些模型中的任何一个。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/google/gemma-7b-it)">OpenRouter</a>: LLM 的统一接口。为您的 Prompt 寻找最佳模型和价格</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1-distill-qwen-32b)">OpenRouter</a>: LLM 的统一接口。为您的 Prompt 寻找最佳模型和价格</li><li><a href="https://openrouter.ai/meta-llama/llama-3.3-70b-instruct)">OpenRouter</a>: LLM 的统一接口。为您的 Prompt 寻找最佳模型和价格</li><li><a href="https://openrouter.ai/meta-llama/llama-3.1-70b-instruct)">OpenRouter</a>: LLM 的统一接口。为您的 Prompt 寻找最佳模型和价格</li><li><a href="https://openrouter.ai/meta-llama/llama-3.2-11b-vision-instruct)">OpenRouter</a>: LLM 的统一接口。为您的 Prompt 寻找最佳模型和价格</li><li><a href="https://openrouter.ai/meta-llama/llama-3.1-8b-instruct)">OpenRouter</a>: LLM 的统一接口。为您的 Prompt 寻找最佳模型和价格</li><li><a href="https://openrouter.ai/meta-llama/llama-3-8b-instruct)">OpenRouter</a>: LLM 的统一接口。为您的 Prompt 寻找最佳模型和价格</li><li><a href="https://openrouter.ai/meta-llama/llama-3.2-3b-instruct)">OpenRouter</a>: LLM 的统一接口。为您的 Prompt 寻找最佳模型和价格</li><li><a href="https://openrouter.ai/meta-llama/llama-3.2-1b-instruct)">OpenRouter</a>: LLM 的统一接口。为您的 Prompt 寻找最佳模型和价格</li><li><a href="https://openrouter.ai/provider/cloudflare)">OpenRouter</a>: LLM 的统一接口。为您的 Prompt 寻找最佳模型和价格
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1336163052881444886)** (1 messages): 

> `Model Error Display` 


- **错误消息中现在会显示模型名称**：一位成员宣布显示问题已解决，**错误消息中现在将显示模型名称**。
   - 此项更改旨在提高用户遇到错误时的清晰度。
- **改进的错误反馈机制**：此次更新标志着向更好的用户体验迈进，通过提供**包含模型细节的更清晰的错误反馈**。
   - 用户现在可以利用更多上下文信息高效地排查问题。


  

---


---


---


---


---


{% else %}


> 邮件中已截断完整的逐频道详情。
> 
> 如果您想查看完整详情，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}