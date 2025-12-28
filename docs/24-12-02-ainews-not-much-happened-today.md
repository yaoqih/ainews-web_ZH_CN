---
companies:
- nvidia
- amazon
- anthropic
- google
- pydantic
- supabase
- browser-company
- world-labs
- cognition
date: '2024-12-02T23:49:20.748299Z'
description: '以下是为您翻译的中文内容：


  **2024年11月29日至12月2日的AI新闻**重点介绍了以下几项进展：**英伟达 (Nvidia)** 推出了 **Puzzle**，这是一种基于蒸馏的神经架构搜索，用于推理优化的语言大模型，从而提升了效率。**IC-Light
  V2** 模型发布，适用于多种照明场景；同时展示了 **Trajectory Attention**（轨迹注意力）和 **Timestep Embedding**（时间步嵌入）等新的视频模型技术。**亚马逊
  (Amazon)** 将对 **Anthropic** 的投资增加至 **80亿美元**，并通过一项新的奖学金计划支持AI安全研究。**谷歌 (Google)**
  正在通过 **Gemini API** 和开放协作工具扩展AI集成。关于域名相关性的讨论强调了 **.com** 域名的替代方案，如 **.io**、**.ai**
  和 **.co**。推理方面的进展包括使用“逆向思维 (Reverse Thinking)”使大语言模型（LLM）的性能提升了 **13.53%**。**Pydantic**
  推出了一个新的智能体（agent）框架，**Supabase** 发布了其助手的第2版。其他值得关注的消息包括 **Browser Company** 预告了其第二款浏览器，以及
  **World Labs** 推出了图像转3D世界技术。NotebookLM 团队离开了 **谷歌**，**Cognition** 登上了 **《福布斯》(Forbes)**
  封面。本次新闻由 **Claude 3.5 Sonnet** 总结。'
id: 4ac56488-4ba3-4258-9079-a5d96a6d3c99
models:
- ic-light-v2
- claude-3-5-sonnet
- puzzle
original_slug: ainews-not-much-happened-today-4970
people:
- akhaliq
- adcock_brett
- omarsar0
- iscienceluvr
title: 今天没发生什么事。
topics:
- distillation
- neural-architecture-search
- inference-optimization
- video
- trajectory-attention
- timestep-embedding
- ai-safety-research
- fellowship-programs
- api
- domain-names
- reverse-thinking
- reasoning
- agent-frameworks
- image-to-3d
- ai-integration
---

<!-- buttondown-editor-mode: plaintext -->**宁静的一天正是你所需要的。**

> 2024/11/29-2024/12/02 的 AI 新闻。我们为你检查了 7 个 subreddits、[**433** 个 Twitters](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discords（**198** 个频道和 **4766** 条消息）。预计节省阅读时间（按 200wpm 计算）：**563 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

没有大事发生，但有很多值得注意的小事：

- [Lilian Weng 发布了一份 Reward Hacking 调查报告](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)
- [Pydantic 发布了他们的 Agent 框架](https://ai.pydantic.dev/)
- [Supabase 发布了其 Assistant 的 v2 版本](https://x.com/kiwicopple/status/1863616764942176668?s=46)
- [ChatGPT 无法说出 David Mayer](https://x.com/iterintellectus/status/1862933297514283383?s=46&t=jDrfS5vZD4MFwckU5E8f5Q)


以及一些预告（尚未发布产品）：

- [Browser Company 预告了他们的第二款浏览器](https://x.com/browsercompany/status/1863593525725556754?s=46)
- [World Labs 推出了 image-to-3d-world](https://x.com/theworldlabs/status/1863617989549109328)
- [NotebookLM 团队离开了 Google](https://x.com/raizamrtn/status/1863645718159954272)
- [Cognition 登上了《福布斯》封面](https://www.forbes.com/sites/rashishrivastava/2024/12/02/cognition-scott-wu-devin-ai/)

---

{% if medium == 'web' %}

**目录**

[TOC]

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 综述

> 所有综述均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**主题 1. 语言与视频模型：创新与优化**

- **Nvidia Puzzle：基于蒸馏的 LLMs 神经网络架构搜索 (NAS)**：[@_akhaliq](https://twitter.com/_akhaliq/status/1863448080328077663) 分享了 Nvidia 关于 **Puzzle** 的演讲，这是一种针对**推理优化型 Large Language Models** 的**基于蒸馏的神经网络架构搜索 (distillation-based neural architecture search)**。该方法旨在提高模型部署的效率和性能。
  - 社区强调的[关于有效性和应用的讨论](https://twitter._akhaliq/status/1863448082383241435)展示了人们对这种优化技术的兴奋。

- **IC-Light V2 模型发布**：[@_akhaliq](https://twitter.com/_akhaliq/status/1863644176677519610) 讨论了专为各种照明场景设计的 **IC-Light V2** 替代模型，并提供了一个展示其潜在应用的 Demo。

- **视频模型的 Trajectory Attention 与 Timestep Embedding**：[@_akhaliq](https://twitter.com/_akhaliq/status/1863441334251495898) 介绍了用于**细粒度视频运动控制的 Trajectory Attention**，以及作为视频扩散模型缓存机制的 **Timestep Embedding**。这些技术在视频运动精度和效率方面取得了进展。

**主题 2. AI 推广与合作**

- **Amazon 与 Anthropic 建立合作伙伴关系**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1863659393901867403) 报道了 Amazon 增加投资的消息，使其对 Anthropic 的总承诺投资额达到 **80 亿美元**——这对该初创公司的增长和 AI 能力是巨大的推动。

- **AI 奖学金与安全研究**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1863648524807577828) 正在启动一项奖学金计划，计划为工程师和研究人员提供资金和指导，以转型从事 **AI safety research**。研究员将与资深研究人员合作，开展涉及 **adversarial robustness**（对抗鲁棒性）、**scalable oversight**（可扩展监督）等项目。

- **Google 在 AI 领域的扩张**：[@osanseviero](https://twitter.com/osanseviero/status/1863590665793245548) 宣布加入 Google，负责 **Gemini API**、开放模型以及 **Colab 和 AI Studio** 等协作空间，这表明 Google 正在推动更广泛的 AI 整合。

**主题 3. 域名与在线身份**

- **关于 .com 主导地位的辩论**：[@adcock_brett](https://twitter.com/adcock_brett/status/1863659449187176470) 认为 **.com 域名**对于公信力并非必要，主张将资金投入到产品和品牌建设上，而不是购买溢价域名。
  - 进一步的讨论（[推文](https://twitter.com/adcock_brett/status/1863659577847452002), [推文](https://twitter.com/adcock_brett/status/1863659488152272901)）强调了 **.io**、**.ai** 和 **.co** 等替代域名后缀在科技和初创环境中的相关性和影响。

**主题 4. 推理与 AI Agents 的进展**

- **LLMs 中的逆向思维增强推理能力**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1863527576687268241) 分享了关于 **Language Models** 中“逆向思维”的见解，通过训练 LLM 从解决方案开始反向推理，其性能比标准方法提高了 **13.53%**。

- **基于 Pydantic 的新 Agent 框架**：[@omarsar0](https://twitter.com/omarsar0/status/1863610490989248588) 宣布推出 **PydanticAI agent framework**，强调使用**类型安全 (type-safe)、模型无关 (model-agnostic)** 的方法来构建生产级应用，并支持**结构化响应验证 (structured response validation)** 和**流式响应 (streamed responses)**。

**主题 5. 机器学习幽默与轻松互动**

- **AI 中的创意策略**：[@goodside](https://twitter.com/goodside/status/1863441256208048239) 幽默地策划了一些让 ChatGPT 难以处理的作业，特别提到将“David Mayer”这个名字作为可能让 AI 用户感到困惑的关键词。
  - 像[“以图片形式布置作业”](https://twitter.com/goodside/status/1863631140143157413)这样的梗探索了与学生之间的趣味互动。

- **关于 AI 实践的新颖视角**：[@swyx](https://twitter.com/swyx/status/1863352038597558712) 鼓励在 AI 驱动的内容中使用**富有创意和表现力的散文**，反对单调的风格，强调书面交流中的多样性和人文元素。

- **探索 AI 对文化和参与的影响**：[@karpathy](https://twitter.com/karpathy/status/1863439146481836538) 经常分享关于 AI 如何影响和改变文化参与的见解，为围绕 AI 及其社会影响的讨论增添了乐趣和幽默。

---

# AI Reddit 综述

## /r/LocalLlama 综述

**主题 1. 中国模型占据主导地位：QwQ-32B 与 DeepSeek 表现超越 GPT-4**

- **QwQ vs o1, etc - illustration** ([Score: 117, Comments: 68](https://reddit.com/r/LocalLLaMA/comments/1h45upu/qwq_vs_o1_etc_illustration/)): 一份视觉对比展示了 **QwQ** 与其他模型在四个技术基准测试中的性能指标：**GPQA**、**AIME**、**MATH-500** 和 **LiveCodeBench**，并引用了早前 [Qwen 2.5 vs Llama 3.1](https://www.reddit.com/r/LocalLLaMA/comments/1fp8v9h/qwen_25_vs_llama_31_illustration/) 的对比。这些基准测试评估了研究生水平的科学知识（**GPQA**，非专家基准准确率为 **34%**，博士专家为 **65%**）、高级数学问题解决能力（**AIME**）、综合数学能力（**MATH-500**）以及实时编程能力（**LiveCodeBench**）。
  - **QwQ 32B 8bit** 展示了卓越的推理能力，正确解决了“**GPT-4 can't reason**”论文中的所有提示词，对于像 **Wason Selection Task** 这样的问题，其冗长的内部对话耗时高达 **30 分钟**。
  - 用户发现 **Ollama 默认的 2k Context Size** 可能会限制 QwQ 的推理 Token，建议使用 **Exllamav2** 或 **Koboldcpp** 以获得更好的性能和 **VRAM** 利用率。该模型可以与 **Qwen2.5-coder-0.5B** 或 **2.5-0.5-Instruct** 配对作为草稿模型进行 **Speculative Decoding**。
  - 该模型表现出多语言推理能力，在其 **Chain of Thought** 过程中会在 **English**、**Chinese**、**Russian** 和 **Arabic** 之间切换。正如 **Karpathy** 所指出的，这种行为表明了[正确的 RL 实现](https://x.com/karpathy/status/1835561952258723930)。


- **Open-weights AI models are BAD says OpenAI CEO Sam Altman. Because DeepSeek and Qwen 2.5? did what OpenAi supposed to do!** ([Score: 502, Comments: 205](https://reddit.com/r/LocalLLaMA/comments/1h4n1i9/openweights_ai_models_are_bad_says_openai_ceo_sam/)): 来自中国的 **DeepSeek** 和 **Qwen 2.5** 等开源 AI 模型展示了足以媲美 **OpenAI** 闭源模型的能力，引发了关于模型可访问性的公众讨论。作为回应，**Sam Altman** 在接受 **Shannon Bream** 采访时表达了对 **Open-weights** 模型的担忧，强调了维持美国在 AI 发展中相对于中国的领导地位的战略重要性。
  - **OpenAI** 被感知的停滞以及对 **Scaling/Compute Power** 的依赖正受到批评，用户指出考虑到新兴的竞争，其 **1570 亿美元** 的估值似乎并不合理。随着开源模型的追赶，该公司似乎正在失去其竞争优势或“**Moat**”（护城河）。
  - 用户指出了 **Sam Altman** 此前对 **Open-weights** 模型安全担忧的讽刺之处，因为更好的开源替代方案已经出现，却并未造成预言中的危害。多条评论引用了他早些时候给 **Elon Musk** 承诺开放的电子邮件，与其现状形成鲜明对比。
  - 技术讨论强调，虽然 **OpenAI** 的 **Advanced Voice Mode** 仍具独特性，但通过 **Whisper**、**LLM** 和 **TTS** 技术的结合，竞争方案正在涌现。用户争论 OpenAI 的领先地位是源于真正的创新，还是主要依靠营销和计算资源。


**Theme 2. JPEG Compression for LLM Weights: Novel Research Direction**

- **[Thoughts? JPEG compress your LLM weights](https://pepijndevos.nl/2024/12/01/jpeg-compress-your-llm-weights.html)** ([Score: 142, Comments: 64](https://reddit.com/r/LocalLLaMA/comments/1h4dl6c/thoughts_jpeg_compress_your_llm_weights/)): **JPEG 压缩技术**可以应用于 **LLM 权重存储**，尽管本帖未提供具体的实现细节或结果。该提议将图像压缩与神经网络参数压缩进行了类比，提出了潜在的存储优化方法。
  - **社区质疑**集中在矩阵重排（Matrix Reordering）的不切实际性上，专家解释说，同时重排行和列会破坏矩阵乘法的特性。多位用户指出，**神经网络权重**的行为更像是随机噪声，而非结构化的图像数据。
  - 技术讨论显示，尝试实现类似压缩技术的尝试结果微乎其微，一位用户报告使用 **Simulated Annealing**（模拟退火）仅能减少“**几个百分点**”的权重分布。另一位用户分享了将 Tensor 转换为 **16-bit Grayscale PNG** 文件的经验，该方法可以无损工作，但在使用 JPEG 压缩时失败了。
  - 几位专家建议坚持使用现有的量化方法，如 **AWQ** 或 **GPTQ**，并指出 **LLM 权重**缺乏使 JPEG 压缩有效的空间模式。讨论强调，权重并不遵循传统压缩算法可以利用的规则统计分布。


**Theme 3. Qwen 2.5 Powers Hugging Face's Text-to-SQL Feature**

- **[Hugging Face 在所有 25 万+ 公共数据集上添加了 Text to SQL 功能 - 由 Qwen 2.5 Coder 32B 驱动 🔥](https://v.redd.it/e3t9ae0h3g4e1)** ([Score: 98, Comments: 11](https://reddit.com/r/LocalLLaMA/comments/1h4w5a3/hugging_face_added_text_to_sql_on_all_250k_public/)): **Hugging Face** 在其 **250,000+ 公共数据集中**集成了 **Text-to-SQL** 能力，采用 **Qwen 2.5 Coder 32B** 作为底层模型。该功能支持将直接的自然语言查询转换为 SQL 语句进行数据库交互。
  - **Hugging Face** 团队成员确认该功能使用 **DuckDB WASM** 进行浏览器内 SQL 查询执行，并配合 **Qwen 2.5 32B Coder** 进行查询生成，同时欢迎用户提供改进建议。
  - 用户对该工具帮助 **SQL** 经验较少的人员的潜力表示热烈欢迎，有人指出它解决了数据集交互中的一个重大痛点。
  - 该公告引发了一些关于内置五彩纸屑动画以及减少对直接 SQL 知识依赖的趣味性回应。


**主题 4. Fox News 将开源 AI 视为国家安全威胁**

- **[开源 AI = 国家安全：监管呼声日益高涨](https://v.redd.it/7j5lxfjoyf4e1)** ([Score: 101, Comments: 70](https://reddit.com/r/LocalLLaMA/comments/1h4vk8t/opensource_ai_national_security_the_cry_for/)): **Fox News** 播出了一段节目，声称**开源 AI 模型**对**美国国家安全**构成风险，尽管报道中未提供具体细节或证据。这一叙事加剧了媒体关于开源 AI 开发潜在监管的讨论，但缺乏实质性的技术分析。
  - 据报道，像 **Deepseek R1** 和 **Qwen** 这样的**中国 AI 模型**已经领先于 **Meta** 的 **Llama** 等美国开源模型。多位用户指出，**中国的顶级模型**并非基于 Llama，这反驳了开源有助于中国发展的说法。
  - 用户批评推动监管是试图强制执行 **AI 垄断**和企业控制。社区认为，限制**美国的开源开发**实际上会将整个开源模型领域拱手让给已经在发布顶级开源模型的中国。
  - 讨论强调，在过去的 **40 年**里，**开源技术**已被证明比闭源替代方案更安全。用户认为，阻止开源开发将损害创新和协作，同时让 **Microsoft**、**OpenAI** 和 **Anthropic** 等大型科技公司受益。


## 其他 AI 子版块回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**主题 1. StreamDiffusion 为演唱会现场 AI 视觉效果提供动力**

- **[Bring Me The Horizon 使用实时 img2img？](https://v.redd.it/16gyxdofxf4e1)** ([Score: 337, Comments: 62](https://reddit.com/r/StableDiffusion/comments/1h4vgbc/bring_me_the_horizon_using_real_time_img2img/)): **Bring Me The Horizon** 的演唱会在现场表演中展示了**实时 img2img AI 视觉效果**。该帖子询问了在现场演唱会环境中实现实时 AI 图像生成和转换的技术工作流。
  - **StreamDiffusion** 似乎是实时 AI 视觉效果的领先解决方案，在 **RTX 4090** 上可达到 **90 FPS**。用户 **tebjan** 为 **vvvv** 创建的演示包展示了相关实现，示例可在 [Instagram](https://www.instagram.com/stories/highlights/17847839352162239/) 和 [Google Photos](https://photos.app.goo.gl/R2Yicr8BF18oa4eBA) 上查看。
  - 视觉一致性通过一种巧妙的技术来维持：**视频源**大于显示的裁剪区域，使得物体即使在离开可见屏幕时也能保留在生成帧内。多位用户报告在 **Download Festival** 的 **Avenged Sevenfold** 表演中看到了类似效果。
  - 社区反应不一，对**时序一致性（temporal consistency）**问题和整体美学质量存在大量批评。**Download Festival** 的一次技术故障凸显了局限性：当 **A7X** 的演出断电时，AI 效果在没有上下文的情况下仍在继续运行。


**主题 2. Haiku vs ChatGPT：免费版对比显示 ChatGPT 领先**

- **Haiku 表现糟糕。** ([Score: 233, Comments: 114](https://reddit.com/r/ClaudeAI/comments/1h4niz5/haiku_is_terrible/)): 一位用户对 **Claude Haiku** 表示失望，认为它明显逊于 **ChatGPT** 的免费层级。尽管尝试坚持使用，但在之前使用过 **Claude/Sonnet** 后，最终还是回到了 **ChatGPT**。该用户居住在**第三世界国家**，认为高昂的订阅费用是获取 **Sonnet** 等高级 AI 模型的主要障碍，并希望未来这些模型能提高可及性。
  - **区域定价**是 **Claude** 可及性的一个重要问题，用户指出在**委内瑞拉**等国家，订阅费用相当于 **2 个月**的最低工资收入。一些用户建议通过创建多个 **Google accounts** 来使用 **Poe**，或者使用提供**每分钟 100 万 tokens** 免费额度的 **Google AI Studio**。
  - 用户报告称，与 **ChatGPT** 的免费层级以及 **Llama** 或 **Qwen** 等本地模型相比，**Haiku** 的表现较差。目前 **ChatGPT** 被认为在免费和付费层级中都最具性价比，不过也有人建议将 **DeepSeek**（每天 **50 次免费使用**）作为替代方案。
  - **Sonnet** 最近的限制（**每周 50 条消息**）令用户感到沮丧，许多人报告称需要大幅缩减项目文件大小并精简 prompt。一些用户将此归因于 **Anthropic** 在被 **Amazon** 收购后转向以 **B2B** 为重点。


**主题 3. World Labs 融资 2.3 亿美元的 AI 初创公司推出 3D 场景生成**

- **[来自 World Labs 的首个 Demo - 由 Fei Fei Li 领导的 2.3 亿美元初创公司。走进图像并与之互动！](https://v.redd.it/r4jefulpoh4e1)** ([Score: 209, Comments: 43](https://reddit.com/r/StableDiffusion/comments/1h53uhj/first_demo_from_world_labs_230m_startup_led_by/)): 由 **Fei Fei Li** 领导的 **World Labs** 推出了一套将图像转换为交互式 **3D 场景**的系统。这家筹集了 **2.3 亿美元**资金的初创公司，让用户能够走进由 2D 图像生成的 3D 环境并进行互动。
  - 技术分析显示，该系统可能使用 **Gaussian splats** 进行渲染，植被中的半透明椭圆以及其 `threeviewer_worker.js` 文件中的引用证明了这一点。该技术似乎是 **2.5D** 的，移动范围有限以避免伪影。
  - 该项目可以通过 [WorldLabs.ai](https://www.worldlabs.ai/blog) 访问，为现代设备提供**实时渲染器**，并为旧款移动设备提供预渲染视频的备用版本。场景生成可能需要 **5 分钟以上**，之后即可进行实时渲染。
  - 围绕 **2.3 亿美元**融资的讨论引发了关于投资价值的辩论，一些人认为这是前沿技术开发，而另一些人则质疑这种在他们看来是高级 HDRI 生成的技术是否值这个价。几位用户提到了潜在的 **VR 应用**和 **metaverse** 影响。


**主题 4. AI 超越人类基准引发测试辩论**

- **[AI 在大多数基准测试中已迅速超越人类，需要新测试来发现剩余的人类优势](https://i.redd.it/4lx2dpfn7g4e1.png)** ([Score: 281, Comments: 146](https://reddit.com/r/OpenAI/comments/1h4wmhr/ai_has_rapidly_surpassed_humans_at_most/)): **AI 系统**在大多数标准评估基准上已超越人类基准线，这使得准确衡量人类剩余的认知优势领域变得困难。**AI 基准测试饱和**的飞速进展表明，需要开发新型测试，以更好地识别和量化人类特有的能力。
  - **LLM** 在复杂的代码合成任务和 **ARC Challenge** 中显示出局限性，用户指出 AI 在 **SAT 题目**等基准测试上的表现可能受到现有测试数据训练的影响，而非真正的理解。
  - 用户强调了现实世界中的表现差距，分享了 **prompt engineering** 耗时远超手动工作的例子，其中一位用户描述了一个案例：他们的老板花了 **2 天**时间尝试完成他们 **30 分钟**就能搞定的工作。
  - 讨论强调了社会影响，对未来 **2-3 年**内的**失业问题**表示担忧，并认为劳动者需要制定“**B 计划**”职业策略；而另一些人则指出，尽管 **Wolfram Alpha** 拥有卓越的数学能力，但并未取代专业职业。

---

# AI Discord 摘要

> 由 O1-preview 生成的摘要之摘要的摘要

**主题 1. 挑战极限：AI 训练与优化新突破**

- [**Nous DisTrO 席卷去中心化训练**](https://distro.nousresearch.com)：*Nous Research* 启动了使用 **DisTrO** 的 **15B 语言模型**去中心化预训练，利用了来自 **Oracle** 和 **Lambda Labs** 等合作伙伴的硬件。他们达到了中心化训练的指标，其 [DeMo optimizer](https://arxiv.org/abs/2411.19870) 减少了加速器间的通信。
- [**自制 CUDA Kernel 在 H100 上击败 cuBLAS**](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)：一个自定义的 **H100 CUDA matmul kernel** 在 **N=4096** 时性能超过 **cuBLAS 7%**，证明了有时编写自己的代码是值得的。
- [**FP8 训练变得更简单：不再需要动态缩放！**](https://github.com/graphcore-research/unit-scaling)：一种新方法通过使用 [unit-scaling 库](https://github.com/graphcore-research/unit-scaling)实现了**开箱即用的 FP8 训练**，无需动态缩放。低精度训练变得更加简单。

**主题 2. AI 工具变得更聪明：不容错过的更新**

- [**Aider v0.66.0 编写了大部分自身代码！**](https://aider.chat/docs)：最新的 *Aider* 版本为 Sonnet 和 Gemini 模型增加了 **PDF 支持**，并引入了带有 `AI!` 注释的 **AI 触发代码编辑**功能。令人印象深刻的是，**82% 的代码**是由 Aider 自身编写的。
- [**Cursor IDE 更新引发争议，但 Agent 功能大放异彩**](https://changelog.cursor.com/)：*Cursor* 移除了 **long context option**，令用户感到沮丧。然而，新的 **agent 功能**被誉为“高级开发人员”助手，使编码更加顺畅，尤其是在小型项目中。
- [**OpenRouter 让用户通过功能投票引导开发**](https://discord.com/channels/1091220969173028894/1092729520181739581/1312105380041461810)：*OpenRouter* 推出了 [Feature Requests Voting](https://discord.com/channels/1091220969173028894/1092729520181739581/1312105380041461810) 系统，邀请用户对新功能进行投票，推动社区驱动的开发。

**主题 3. AI 模型集成与训练中的障碍**

- **微调 Qwen 2.5？别忘了“秘方”！**：用户强调在微调 **Qwen 2.5** 时需要使用 **Qwen 特定的 ChatML 模板**，并警告不要使用默认选项以避免出现问题。
- **Stable Diffusion 与 Lora 模型：集成的烦恼**：尽管遵循了所有步骤，用户在 **Stable Diffusion** 中运行 **Lora** 模型时仍遇到困难，这指向了集成过程中可能存在的 Bug 或被忽视的步骤。
- **CUDA 错误影响进度？试试量化魔法**：面对加载大模型时的 **CUDA 错误**和 **VRAM 限制**，用户建议切换到更小的量化格式，或选择具有更好 GPU 支持的其他云服务商。

**主题 4. AI 模型性能：各有所长**

- **Claude 擅长聊天；ChatGPT 擅长说教：各取所需**：用户对比了 **Claude** 和 **ChatGPT**，指出 Claude 提供更具亲和力的对话，而 ChatGPT 提供深入的哲学见解，使其更适合结构化讨论。
- **谷歌的 Gemini 模型难以获取**：OpenRouter 用户抱怨谷歌实验性模型（如 **Gemini Pro 1.5**）的 **rate limiting**，怀疑谷歌严格的限制导致了连接问题。
- **GPT-4 无法查看你的图像，用户对此不满**：由于 **GPT-4** 反复无法处理图像，并返回“*I currently can't view images directly*”之类的错误，阻碍了生成准确图像说明等任务，用户的沮丧情绪在蔓延。

**主题 5. 微调未来：高效 AI 成为主流**

- [**等变网络证明了其在数据效率方面的价值**](https://arxiv.org/abs/2410.23179v1)：研究表明，**equivariant networks**（等变网络）提高了刚体交互中的数据效率，优于非等变模型，尤其是在数据有限的情况下。
- [**ThunderKittens 需要一些自动优化支持**](https://github.com/HazyResearch/ThunderKittens/blob/tk_gen/simple_kernels/micro_add/micro.cu)：受类似 DSL 经验的启发，有人提议为 **ThunderKittens** 开发一个 **auto optimizer**，以最大化其“一次编写，多次运行”的潜力。
- **混合精度推理：精度检查变得棘手**：深入研究 **vLLM** 混合精度推理的开发人员讨论了验证 kernel 执行精度的挑战，并指出了当前分析工具的局限性。

---

# 第一部分：高层级 Discord 摘要

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor IDE 更新问题**：用户报告了最新 [Cursor changelog](https://changelog.cursor.com/) 的问题，特别是 Composer 无法应用更改以及缺失 'Apply' 按钮，导致功能使用受阻。
   - 此外，多位用户注意到自最近更新以来，Chat 中的 long context 使用被移除或表现不稳定。

- **Composer 与 Chat 模式对比**：在 **Cursor IDE** 中，用户正在对比直接修改文件的 Composer 模式与提供内联更改的 Chat 模式，讨论它们的局限性和功能差异。
   - 用户希望改进两种模式之间的集成，例如高效地将讨论从 Chat 转移到 Composer。

- **Windurf 与 Cursor IDE**：用户正在探索 **Windurf** 作为 Cursor IDE 的潜在竞争对手，指出其在处理 terminal 输出和 codebase search 方面表现出色。
   - 虽然 **Windurf** 展现出潜力，但 Cursor 在特定工作流中仍保持优势；不过，用户对两者的体验评价不一。

- **Cursor IDE 中的 API Key 限制**：讨论强调了 **Cursor API 使用**的限制，一些用户选择使用自己的 API Key 以获得更多灵活性。
   - 社区正在寻求改进 API 调用限制的管理，并增强对活动项目的 context 收集能力。

- **Cursor 中的 Context 管理**：用户对 **Cursor IDE** 目前的 context 处理表示不满，特别是关于 **Claude** 的限制。
   - 社区倡导更好的 context 管理功能和一致性，以改进其编码工作流。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Anthropic 的 MCP 框架让 Claude 能够作为 API 运行**：Anthropic 发布了新的 **MCP 框架**，使 **Claude** 能够运行服务器，有效地将 Claude 应用转变为 [API](https://x.com/skirano/status/1861081529071346161)。
   - 这一进展允许 **Claude** 在本地创建、读取和编辑文件，引发了用户对与 **VSCode** 等工具进行实时交互的兴奋。

- **Gemini 与 ChatGPT 的响应约束对比**：**Gemini** 经常出于所谓的道德原因拒绝回答无害的问题，而 **ChatGPT** 被认为在响应上更加宽松。
   - 用户幽默地指出了 Gemini 拒绝讨论*人工智能*的案例，以避免参与敏感话题。

- **Claude 3.5 Sonnet 成为图像描述（Image Captioning）的替代方案**：由于 **OpenAI** 的 vision 能力持续存在问题，用户建议在图像描述任务中切换到 **Claude 3.5 Sonnet**。
   - 社区成员指出 **Claude 3.5 Sonnet** 提供了更可靠的功能，帮助用户避免项目延迟。

- **Windows 版 ChatGPT 集成语音转文字（Speech-to-Text）功能**：一位用户询问如何在 Windows 上为 **ChatGPT** 实现语音转文字功能，建议使用内置的 Windows 辅助功能，通过按下 **Windows + H** 来实现。
   - 这种方法为与 **ChatGPT** 交互时将语音转换为文字提供了实时解决方案。

- **与 'strict' 放置错误相关的结构化输出（Structured Output）错误**：用户报告在使用 structured outputs 时遇到随机的 'object' 包装器，这被追溯到 **'strict'** 设置的位置不正确。
   - 经过广泛调试，确认误放 **'strict'** 会导致持续的 structured output 错误。



---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **QwQ 模型配置协商**：用户讨论了在 architect mode 下部署 **QwQ** 模型，并配合标准模型处理代码命令，寻求关于互换性的明确说明。
   - Aider 促进跨项目的模型定义，提升灵活性 [Advanced model settings](https://aider.chat/docs/config/adv-model-settings.html)。

- **DeepSeek-R1 创下新基准**：**DeepSeek-R1** 在 [AIME & MATH benchmarks](https://api-docs.deepseek.com/news/news1120) 中取得了卓越成绩，强调了其开源可用性和实时推理能力。
   - 社区成员希望 DeepSeek 发布模型权重，以便集成到与 **QwQ** 的 ensemble frameworks 中。

- **优化 Aider 的本地模型设置**：成员们协作配置 `.aider.model.metadata.json` 和 `.aider.model.settings.yml` 文件，以在 **Aider** 中定义本地模型。
   - 将编辑格式选择为 'whole' 或 'diff' 会显著影响响应结构和编辑效率。

- **OpenRouter 的挑战影响 Aider**：参与者发现 **OpenRouter** 存在影响本地服务器模型检测和功能的问题。
   - 有人担心伪造的实现可能会改变模型的输出和行为。

- **QwQ 与 DeepSeek 的集成框架**：一位用户表示打算在 ensemble frameworks 中集成 **QwQ** 和 **DeepSeek** 模型，以增强推理能力。
   - 这种方法旨在利用两种模型的优势来提高性能。



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 微调注意事项**：用户讨论了 **instruct** 与 **non-instruct** 微调的优劣，建议对超过 **1k 条记录** 的数据集使用 base models，并建议对 **70k 条记录** 左右的数据集尝试使用 *instruct* 模型。
   - 建议参考 [Unsloth Documentation](https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama#id-6.-alpaca-dataset) 了解数据集格式规则，强调合规性对有效微调的重要性。

- **Unsloth 数据隐私措施**：**Unsloth** 被确认在微调期间不会向外部传输数据，而是依赖用户选择的平台，如 [Google Colab](https://colab.research.google.com/drive/18sN803sU23XuJV9Q8On2xgqHSer6-UZF?usp=sharing)。
   - 这一保证解决了处理敏感信息的用户对遵守严格**数据隐私**政策的担忧。

- **RAG 计算成本挑战**：讨论强调，由于广泛的 context length 需求，**retrieval-augmented generation (RAG)** 可能导致**高计算成本**，如 [Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs](https://arxiv.org/abs/2312.05934) 中所述。
   - 用户正在平衡性能与效率，特别是在**知识密集型任务**中，研究结果支持 RAG 优于微调。

- **LLama 3.1 OOM 错误解决方案**：在对 **LLama 3.1 8B** 模型进行持续预训练时遇到 **out of memory (OOM)** 错误，建议使用更大的 GPU、减小数据集规模或降低 batch size。
   - 这些策略旨在缓解显存问题，确保大规模模型的训练过程更加顺畅。

- **Latent Paraphraser 架构增强**：**latent paraphraser** 被解释为对 Transformer 架构的一种修改，增加了一个层来重新分配 token 的概率。
   - 这种增强通过在处理过程中最小化未见 token，改善了输入锚定 (input grounding) 并减少了噪声。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 节日折扣**：**Perplexity 团队**宣布了 Perplexity Pro 首月 **2.5 折（75% off）**的促销活动，截止时间为 **太平洋时间 12 月 2 日星期一晚上 11:59**。新用户可以借此访问包括增强搜索和文件上传在内的高级功能。
   - 该优惠还包括通过 Buy with Pro 实现的**一键购物**和**免费送货**，旨在简化用户在节日期间的购物体验。

- **Perplexity 与 Claude 的集成**：用户询问如何利用新的 MCP 功能将 **Perplexity** 集成到 **Claude** 中（类似于其与 **Brave** 和 **GitHub** 的功能），通过利用 Claude 的 Project Knowledge 来提升性能。
   - 此外，还有关于在 **Claude** 中集成 **Google** 可能性的提问，突显了用户对利用搜索功能的兴趣。

- **Perplexity 图像生成功能**：讨论了该平台的图像生成能力，并确认可以通过电脑在线使用，无需额外费用。
   - 用户探索了这些功能的范围，考虑了它们的可访问性以及在各种项目中的潜在应用。

- **RBAC 与 ABA 访问控制模型**：一位成员寻求关于 **RBAC (Role-Based Access Control)** 和 **ABA (Attribute-Based Access Control)** 系统之间区别的澄清。
   - 这一讨论强调了在技术实现中理解访问控制模型的必要性。

- **Claude Spaces 中的自定义指令**：用户提出了关于 Claude spaces **自定义指令**有效性的问题，这些指令似乎与现有的“自我介绍”提示词存在冲突。
   - 用户正在寻求关于这些指令应如何交互以及是否可以有效结合的指导。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **HF 搜索问题已解决**：**HF 搜索无法工作**的问题已得到解决，这让用户松了一口气。
   - 附带了一张图片来纪念这次修复，标志着社区的一次积极更新。

- **LM Studio AIDE 集成成功**：用户成功将 LM Studio 端点集成到 AIDE sidecar，实现了完全本地的代码编辑器体验。
   - 这一集成为寻求本地开发环境的用户增强了功能。

- **Llama 3.1 模型可访问性**：一位用户询问如何在 LM Studio 中获取 **Llama 3.1 8B** 的基础模型（base model），并指出似乎只有指令微调（instruction-tuned）变体可用。
   - 社区成员指出 [huggingface 仓库](https://huggingface.co/meta-llama/Llama-3.1-8B) 是获取基础模型的潜在来源。

- **a770 性能逊于 7800xt**：一位成员分享称，他们的 **a770** 在运行 Qwen2.5-14b q4_0 时仅达到了 **11t/s**，远低于 **7800xt** 达到的 **40t/s**。
   - 他们指出 *q4_k_m 无法使用*，但发现 sycl 后端的速度提升微乎其微。

- **Seasonic PSU 寿命获赞**：一位成员提到，尽管由于灰尘原因每隔几年就要更换一次 PSU，但他们的 **Seasonic PSU** 寿命比其他 PC 组件都要长。
   - 他们形容对该 PSU 性能的体验是“令人惊讶地”满意。



---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **资源竞争的缓解 (De-escalation of Resource Contention)**：成员们强调了对**资源竞争缓解**及其对不受监管的互联网增长影响的担忧，质疑 AI 驱动的隐私解决方案的有效性。他们强调了识别*流氓 AI 攻击预警信号*以保护脆弱设备的重要性。
   - 讨论强调了在 AI 保护中需要社区领导力，以减轻与资源竞争和未经授权的 AI 活动相关的风险。

- **庞加莱球嵌入 (Poincare Ball Embedding) 解析**：将数据嵌入到**庞加莱球 (Poincare ball)** 中可以确保具有较高度数的点更接近原点，在过渡到**曲率较小**的区域时保持邻接性。这种方法有助于表示复杂的层级结构。
   - 一位成员指出了庞加莱球边缘的概念性挑战，指出它代表了一个点无法物理驻留的无穷远点，这引发了进一步的技术讨论。

- **等变网络 (Equivariant Networks) 获得效率提升**：最近的一篇论文发现，在各种模型大小和计算预算下，**等变网络 (equivariant networks)** 与**非等变网络 (non-equivariant networks)** 相比增强了数据效率。研究表明，等变模型始终优于其非等变对应模型。
   - 实证结果表明，虽然非等变模型在经过充分训练后可以达到等变模型的性能，但等变网络在不需要大量计算资源的情况下提供了卓越的效率。

- **理解 Eval Harness 中的 HF Tokenizers**：关于 eval harness 是否使用 `add_special_tokens=True` 或 `False` 进行序列分词存在困惑，特别是在处理生成任务期间的 **EOS tokens** 方面。成员们澄清说，通常在构建自定义分词器时**仅添加 BOS tokens**。
   - 讨论显示，在训练循环中手动管理 EOS token 是避免跨不同使用 HF 模型的框架出现兼容性问题的实用方法。

- **TaskSet 助力优化器训练**：**TaskSet** 数据集包含一千多个不同的任务，对于在**元学习 (meta-learning)** 环境中训练和评估优化器至关重要。该数据集能够实现比传统随机搜索方法显著的效率提升。
   - 尽管认识到 **TaskSet** 有些过时，但成员们承认，尽管 AutoML 研究存在资金限制，它仍是构建大型学习曲线数据集的最佳可用选择。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **功能需求投票**：敦促成员们[在此为最重要的功能需求投票](https://link.to/vote)，以确定后续开发的优先级。
   - 对于任何未列出的请求，用户可以在 <#1107397803266818229> 中提交，从而实现更广泛的社区驱动功能输入。

- **Pixtral Large 性能**：**Pixtral Large** 因其卓越的性能和**庞大的免费额度**而受到赞誉，可通过 [console.mistral.ai](https://console.mistral.ai) 轻松访问。
   - 一位用户报告说从 **Hermes 405b** 切换到了 **Pixtral**，并指出其在提示词不变的情况下表现出色。

- **模型身份识别困惑**：讨论强调模型本质上无法识别自己的身份，并且经常从训练数据中幻觉出细节。
   - 这导致尽管进行了澄清，用户之间仍对模型身份识别存在挥之不去的困惑。

- **生成成本估算**：一位用户询问了 **/api/v1/generation** 端点的费率以及准确估算生成成本的方法。
   - 建议包括利用 **Helicone** 进行跟踪，并强调生成端点对于精确成本评估至关重要。

- **自定义提供商密钥 (Custom Provider Keys) 访问**：开发者正在推动访问**自定义提供商密钥**，反映了社区对该功能的强烈需求。*一位成员在请求访问时提到*，“感谢你们所做的出色工作！”
   - 包括 **monomethylhydrazine** 和 **kit18** 在内的几位用户表达了为特定提供商使用自己密钥的需求，突显了社区对该功能的共识。

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton 元编程与源码构建**：一个旨在解决 Triton 现有局限性的 [元编程提案](https://github.com/triton-lang/triton/pull/5284) 引起了社区兴趣，尽管一些成员要求提供更清晰的语义和示例。
   - 此外，在 WSL2 上从源码构建 Triton 需要将内存增加到 **26GB** 以防止内存溢出错误，成员们还讨论了 Ubuntu Docker 容器中的离线编译依赖。

- **ThunderKittens 与 ThunderMittens 的统一**：围绕 **ThunderKittens** 和 **ThunderMittens** 的讨论强调了 **tile 抽象** 在统一框架以实现 tensor core 兼容性方面的作用，重点在于寄存器使用的控制。
   - 成员们还询问了两者之间现有的 API 契约，并对 ThunderKittens 的 **自动优化器 (auto optimizer)** 表示关注，以增强其“一次编写，多次运行”的系统。

- **结合 RedPajama 和 Dolma 数据集的 BitNet b1.58**：在 [RedPajama 数据集](https://github.com/togethercomputer/RedPajama-Data) 上使用 **100B tokens** 训练的 **BitNet b1.58** 模型发布，展示了极具前景的 PPL 和零样本准确率结果。
   - 此外，在 [Dolma 数据集](https://huggingface.co/datasets/allenai/dolma) 的 **60B tokens** 上训练的 **OLMo-Bitnet-1B** 模型强调了以研究为中心的方法，其 [文档](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf) 中提供了详细的训练超参数。

- **扩散模型技术概览**：最近关于扩散模型的讨论强调了它们在生成感知信号方面的主导地位，并指出改进的模式覆盖 (mode coverage) 和 **更快的采样** 是其主要优势。
   - 在 [OpenAI 的 DALL·E 2](https://openai.com/dall-e-2/) 和 [Google 的 Imagen](https://imagen.research.google/) 等系统中，**无分类器扩散引导 (classifier-free diffusion guidance)** 的实现被强调用于增强条件扩散模型的输出，其中 [噪声调度 (noise schedule)](https://sander.ai/2024/06/14/noise-schedules.html) 设计对性能至关重要。

- **开源日语 LLM 排行榜发布**：与 **Hugging Face** 合作推出的 [开源日语 LLM 排行榜 (Open Japanese LLM Leaderboard)](https://huggingface.co/spaces/llm-jp/open-japanese-llm-leaderboard) 旨在通过 **20 多个数据集** 和任务评估日语 LLM。
   - 该倡议旨在解决日语 LLM 性能落后于英语的问题，吸引了专注于母语进步的日本 **HPC 工程师** 的关注。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes 3 进展及 O1 风格集成**：**#general** 频道的一场讨论强调了关于 [**Hermes 3**](https://discord.com/channels/1053877538025386074/1149866623109439599/1311901917487824956) 的咨询，暗示其与之前的 **O1 风格** 有关联。
   - 这反映了社区对 **Hermes** 最新进展及其演进的持续关注。

- **Mistral 平台面临模型选择障碍**：成员们对 **Mistral AI** 平台最近改为默认单一模型选择选项表示担忧。
   - **图像生成** 能力的限制引起了困惑并影响了用户体验。

- **Truth Terminal 将 AI 与加密货币叙事融合**：有关 **Truth Terminal** 通过加密空间内的半自治 AI 创建自己的“宗教”的见解被分享。
   - 这种独特的融合强调了 **AI 对齐 (AI alignment)** 讨论与 **AI 及加密社区** 的交集。

- **低比特量化有利于训练不足的 LLM**：研究表明，与经过大量训练的小型模型相比，**低比特量化** 对训练不足的大型 **LLM** 造成的退化较小，详见 [此论文](https://arxiv.org/abs/2411.17691)。
   - 研究结果强调了量化策略与 **模型大小** 及 **训练 token** 需求相匹配的重要性。

- **三进制量化受限，FP4 成为高效选择**：观察显示，**三进制量化** (BitNet) 仅能改善 **训练不足的网络** 的结果，其广泛适用性受到质疑。
   - 因此，社区正倾向于将 **FP4** 作为当前模型架构的首选数值权重表示。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **关于 Mojo Origins 与 Rust Lifetimes 的混淆**：一位用户对 **Mojo's Origins** 与 **Rust's lifetimes** 的相似性表示困惑，认为两者虽然都旨在解决内存管理问题，但在本质上是不同的。
   - 虽然受到 Rust 的启发，但 **Mojo's design** 是刻意区分的，旨在实现不同的 **Compiler behaviors** 和目标。

- **Mojo Origins 维持内存控制**：Mojo 的 **Origin** 表示一个内存块；当一个指针被 origin 参数化时，它表示该指针指向该内存内部，并根据需要延长变量的 lifetimes。
   - **Origins** 有助于实现 **aliasing guarantees**，如果指针在其目标失效时仍然存活，则会产生 **compile-time errors**。

- **理解 Origins 需要耐心**：从 **Compiler perspective** 理解 **Mojo Origins** 具有挑战性，特别是由于它们尚未定型，导致细节可能会发生变化。
   - 一位用户表示愿意等待该主题更加清晰，而不是过早地提出更多问题。

- **变量名中空格带来的 Namespace 挑战**：有人提出了在变量名中使用空格的可能性，例如 `var xe đạp = 'abc'`，并指出编程语言普遍缺乏对此的支持。
   - 允许空格会显著增加 **Parser implementation** 的复杂性，使其变得不切实际。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **Notebook LM 播客功能在 30 分钟内创建音频**：一位用户赞扬了 **Notebook LM** 的能力，该功能仅用 30 分钟就利用有关其 **德国少棒联盟计划**（包括其历史性的世界大赛资格）的文件创建了一个音频播客。[播客剧集](https://weplayball.buzzsprout.com/1787721/episodes/16191436-episode-9-home-run-fur-deutschland-die-little-league-baseball-story) 展示了 AI 生成内容的无缝集成。
   - 这证明了 **Notebook LM** 如何高效地生成多媒体内容，从而增强用户的项目工作流。

- **NotebookLM 增强高魔奇幻世界观构建**：一位用户分享了使用 **NotebookLM** 为高魔奇幻小说构建世界观的经验，强调了该模型提供上下文感知响应的能力。
   - AI 的推理能力根据现有规则为他们的魔法系统带来了新的见解和机制。

- **GenFM 在 AI 播客领域挑战 NotebookLM**：一位成员分享了一个名为 [‘GenFM, Now Playing on ElevenReader: Smart Podcasts Produced by Generative AI’](https://youtu.be/x6ub-9HhxGU) 的视频，突显了 AI 领域的竞争。
   - 尽管 GenFM 加入了竞争，另一位成员指出 **NotebookLM** 仍然提供更深层次的交互体验。

- **RAX 在时代广场广告牌的大胆接管**：**RAX**（一只赛博朋克浣熊）接管了时代广场的广告牌，以“不要购买你看到的一切”为信息倡导理性消费。一段 [YouTube 视频](https://youtu.be/ZAXwrUduAt0?feature=shared) 讨论了这一事件，强调需要反思消费文化。
   - 这场数字表演引发了社区内关于消费主义的讨论。

- **FDP 计划解散德国联合政府**：**FDP** 计划解散由总理 **Gerhard Schröder** 领导的联合政府，并制定了一项策略，将其退出描述为政治进步的必要之举。
   - 内部文件提供了关键的叙述和时间表，以确保德国公众在即将到来的选举中获得明确的选择。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Perplexity 巧妙的黑色星期五营销活动**：Perplexity 推出了一个巧妙的 [黑色星期五营销活动](https://x.com/AravSrinivas/status/1861938387923701866)，这与近期 **利用 AI 能力的营销趋势** 相契合。
   - 该举措因其在营销策略中对 AI 的战略性整合而受到关注。

- **人类在模式识别方面优于 AI**：成员们的共识表明，虽然 **AI** 计算速度更快，但 **人类** 在识别复杂问题中的全局模式方面表现出色，经常会做出诸如 *“等一下，这不对劲”* 之类的反应。
   - 这种识别全局不一致性的能力使人类区别于可能专注于特定局部问题的 AI 系统。

- **企业生成式 AI 投资**：最近的一份报告强调，2024 年 **AI 支出** 飙升至 **138 亿美元**，标志着从实验性使用向核心业务战略的转变。
   - 尽管投资有所增加，但超过三分之一的决策者仍在开发将生成式 AI 集成到其业务中的有效方法。

- **Freysa AI Agent 挑战赛资金发放**：一项 AI 挑战赛导致 Freysa Agent 通过一个巧妙设计的 Prompt 转移了 **47,000 美元**，该 Prompt 绕过了严格的转移指令。
   - 这一事件强调了在金融交易中进行 AI 操纵的 **Prompt Engineering** 的复杂性，并展示了透明、开源的设置。

- **技术采用与投资趋势**：参与者将当前的 **LLM** 趋势与历史上的技术变革进行了比较，指出了在兴奋程度和潜在市场回调方面的相似之处。
   - 正在进行的讨论引发了对 AI 技术可持续性和未来盈利能力的担忧，呼应了航空等行业中出现的模式。



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SD 3.5 的 ControlNet 质量问题**：一位成员报告称，**SD 3.5 的 ControlNet** 仅在 **1024x1024** 分辨率下才能生成无伪影的高质量渲染图。
   - 另一位成员将这些问题归因于 *缺乏熟悉度*，并鼓励通过实验来更好地理解 **ControlNet** 的功能。

- **Stable Diffusion 硬件性能**：一位用户询问了 **Stable Diffusion** 的性能基准，提到达到了大约 **5 IT/s**。
   - 社区成员积极分享了他们的硬件能力，反映出对优化 **Stable Diffusion** 设置的浓厚兴趣。

- **AI 艺术的 LoRA 模型需求**：一位用户请求关于 **LoRA half girl 模型** 的信息，以创建融合了两种不同女性设计的角色。
   - 这一请求突显了 **AI 生成艺术** 中角色开发方面持续的实验和创意。

- **内容创作者的感恩节祝福**：一位成员向 **Stability.ai** 团队和其他创作者表达了 **Happy Thanksgiving** 的祝福。
   - 这一举动强调了 **AI** 领域内容创作者之间的情谊和协作精神。



---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **TinyFPGA 的潜在内存架构**：成员们讨论了 **TinyFPGA** 的设计，思考如何模拟典型的 **memory hierarchy**（内存层级），同时指出 **Block RAM** 和 **DDR3** 等现有选项是不够的。
   - 提出了 **'first pass' memory** 的想法，将常量定位在 ALU 附近，有望显著提升性能。

- **传统内存模型的挑战**：讨论强调，随着未来 **TinyFPGA** 设计转向更高效的内存层级，**heuristic eviction policies**（启发式逐出策略）可能会过时。
   - 对 **trained parameters** 的未来进行了推测，提到 **tensors** 可能会取代它们。

- **Exa Laboratories 可持续芯片设计**：关于 **Exa Laboratories** 的对话强调了他们的使命，即创建在特定 AI 需求下**速度**和**能效**优于传统 GPU/TPU 的**可重构芯片**。
   - 有人对其可行性表示怀疑，指出了小公司在芯片开发中面临的挑战，尤其是雄心勃勃的时间表。

- **Tenstorrent 的生物学合理训练算法**：George Hotz 提到 **Tenstorrent** 是一个认真的参与者，正在投资模拟生物过程的训练算法，以实现更高的效率。
   - 潜在的变化包括 **hierarchical memory models**（分层内存模型）和让人联想到计算中大脑功能原理的实时优化。

- **tinygrad 中的 VIZ 工具**：一位成员发布了详细的教程，解释了 **VIZ 工具**，可在[此处](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20241129_viz.md)查看，增强了对其在 tinygrad 中功能的理解。
   - George Hotz 在一条推文中认可了 **VIZ 工具**，称 **VIZ=1** 是对 **LLVM/MLIR** 的重大改进，强调了其优势。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Aya 项目贡献指南**：一位成员寻求关于兼职贡献 **Cohere 的 Aya 项目**的指导。
   - 另一位成员建议加入 [Aya server](https://discord.gg/8kzwCTd7) 直接与社区联系。

- **感恩节庆祝和餐食分享**：成员们分享了 *Happy Thanksgiving* 祝福和他们的餐食图片，包括一位成员令人印象深刻的一盘食物。
   - 另一位成员幽默地评论说尝试吃得健康，但指出味道不如预期。

- **食物分享和珍宝蟹**：成员们交流了丰盛餐食的评论和图片，有人开玩笑说他们的饭菜更像是甜点。
   - 随后出现了一个幽默的评论，说之前已经吃了一盘 **Dungeness crab**（珍宝蟹），增强了食物分享的氛围。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **dspy.asyncify 支持相关问题**：一位成员询问了关于使用 `dspy.asyncify` 的问题，特别是它对线程的使用，以及由于 celery workers 的问题是否提供 **pure async support**（纯异步支持）。
   - 另一位用户也表达了对 **pure async support** 的渴望，以解决现有的 celery worker 问题。

- **带有断言的 dspy demo 行为**：有人担心在激活断言时，`dspy` 在最终 prompt 中不使用 demo。
   - 一位成员澄清说，_retry_ 模式下的演示取决于编译是在激活断言之前还是之后进行的。

- **欢迎 Shaun 加入公会**：Shaun 加入了服务器，向大家打招呼，并对正在进行的项目表示兴奋。
   - 社区欢迎 Shaun，营造了一个包容的环境。



---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **DPO 通过 LoRA-DPO 在不同仓库间保持一致**：来自 Hugging Face 的 [DPO Trainer](https://huggingface.co/docs/trl/en/dpo_trainer#dpo-trainer) 表明，尽管代码有所不同，但 **DPO 技术** 在 LoRA-DPO 等不同仓库中保持一致。
   - 这种一致性确保了实现方案保持对齐，从而简化了不同 DPO 方法之间的集成和比较。

- **全参数 DPO 的可行性**：**实现全参数 DPO** 是可行的，并且与 LoRA-DPO 相比，可能会增强训练后的对齐效果。
   - 社区建议借鉴现有 **全量 PPO** 实现的经验来指导这一过程。

- **引入 dpo_full_finetune_single_device PR**：一个新的 PR 增加了 **针对分布式设置的全量微调 DPO**，为单设备实现奠定了坚实基础。
   - 详情可以通过 [full DPO PR](https://github.com/pytorch/torchtune/pull/1966) 获取，其中概述了拟议的更改和增强功能。

- **Torchtune 将支持全量微调 DPO**：Torchtune 即将进行的更新将支持 **全量微调 DPO**，这需要修改以加载独立的参考模型。
   - 这些更改涉及修改对参考模型的初始调用，以改进现有框架内的功能和集成。

- **FFT DPO 的内存占用更高**：由于需要存储梯度并维护完整的模型副本，**FFT DPO** 将比 LoRA 消耗显著更多的内存。
   - 如果 LoRA DPO 不能满足性能要求，那么采用全量微调 DPO 所带来的内存消耗权衡可能是值得的。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Quiz 11 仍未开放？**：一位成员对 **Quiz 11** 的状态表示困惑，询问为什么它还没有开放。
   - *是否有预计的开放日期？*
- **关于 OpenAI 额度的咨询**：一位用户询问了他们的 **OpenAI 额度** 状态，提到他们上周填写了表格。
   - *他们表达了紧迫感，表示需要支持来进行项目开发。*
- **MOOC 完成情况与证书资格**：一位成员询问现在开始 **MOOC** 是否仍能在完成后获得证书。
   - *他们还很好奇在剩余时间内完成所有要求是否可行。*

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 仪表板开发**：一位成员宣布他们正在开发一个受 **Open Interpreter** 启发的项目，重点是创建一个将于今年发布的 **开源仪表板**。
   - 该项目强调是一个**有趣的小项目**，没有任何盈利目的。

- **社区对仪表板项目的支持**：另一位成员祝贺了项目创建者，并以 **'Nice work! Well done 🚀'** 表达了热情。
   - 这种交流凸显了社区对该领域创新项目的鼓励。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OLMo 2 性能提升实力**：来自 **Allen AI (AI2)** 的 **OLMo 2** 系列（包含 **7B** 和 **13B** 模型）在多达 **5T tokens** 上进行了训练，其表现 [优于 Llama-3.1 8B](https://weightwatcher.ai/models/Llama3.1/Llama-3.1-8B-Instruct.html) 和 [Qwen 2.5 7B](https://weightwatcher.ai/models/Qwen2.5-small/Qwen2.5-7B-Instruct.html)。
   - 关键改进包括采用了带有 **RMSNorm** 和 **QK-Norm** 的优化架构，以及全面的两阶段课程学习训练方法。

- **OLMo 2 打造尖端训练**：OLMo 2 在最终检查点采用了 **model souping 技术**，并采用了受 **Tülu 3** 启发的训练后方法，包括指令微调、使用 **DPO** 的偏好微调以及具有可验证奖励的 **强化学习**。

- **Instruct OLMo 2 领跑开源权重模型**：经 **OLMES 测试集** 验证，**OLMo 2** 的 **13B Instruct** 变体在指令任务中超越了 [Qwen 2.5 14B](https://weightwatcher.ai/models/Qwen2.5/Qwen2.5-14B-Instruct.html) 和 **Tülu 3 8B**。

- **Weight Watcher AI 获得梗图级别的关注**：**Weight Watcher AI** 被强调为 AI 领域的一个新奇补充，并在 **memes** 频道中被幽默地分享，因其趣味性引起了关注。
   - 虽然分享了 [OLMo summary](https://weightwatcher.ai/models/OLMo-summary.html) 链接，但未发现具体描述。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **开发者技能展示**：一位成员分享了广泛的开发技能列表，包括 **React**、**Next.js**、**Angular** 和 **D3.js**，重点介绍了他们在 **UI/UX** 以及 **Protractor** 和 **TestCafe** 等测试框架方面的经验。
   - 这种多样化的技能组合彰显了他们在前端和测试技术方面的适应能力，增强了他们应对复杂工程挑战的能力。

- **多元化技术栈**：该开发者提到了广泛的技术，如 **Node**、**Nest.js**、**Solidity** 和 **Rust**，包括对 **Bootstrap** 等前端框架以及 **BEM** 和 **SMACSS** 等样式方法的了解。
   - 这种全面的技术栈能够跨各种平台和框架进行高效的集成与开发，满足多方面的项目需求。

- **API 集成专业知识**：他们表示熟悉集成多种 API，包括 **Google Maps**、**YouTube** 和 **Facebook APIs**，使他们能够参与需要高效数据交互的多样化项目。
   - 他们管理和实施多样化 API 集成的能力，有助于在系统架构中实现稳健且可扩展的解决方案。

- **云部署技能**：该成员强调了他们在云服务能力中的 **AWS**，能够将应用程序有效地部署到云环境中。
   - 精通 **AWS** 可确保可靠且可扩展的云部署，优化资源管理和基础设施性能。

- **呼吁合作**：他们最后发出了建立联系的邀请，促进了开发者社区内潜在的人脉机会。
   - 这种外联活动促进了具有相似技术兴趣的工程师之间的专业协作和知识共享。



---


**MLOps @Chipro Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---


**Axolotl AI Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---


**LAION Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---


**Mozilla AI Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---


**HuggingFace Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---


**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---

# 第 2 部分：各频道详细摘要和链接


{% if medium == 'web' %}

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1311798943201497240)** (237 条消息🔥🔥): 

> `Cursor IDE 更新, Composer vs Chat Mode, Windsurf 优势, API Key 使用, Context Management` 


- **Cursor IDE 更新引发问题**：用户报告了 Cursor 最新更新中的问题，特别是 Composer 无法应用更改且缺少 'Apply' 按钮，导致用户对功能感到沮丧。
   - 许多人还注意到，自更新以来，某些功能（如 Chat 中的长 Context 使用）似乎已被移除或运行不稳定。

- **Composer 与 Chat Mode 的对比**：Composer 模式直接修改文件，而 Chat 模式提供行内更改，用户讨论了这两种模式之间的局限性和功能差异。
   - 用户请求在两者之间建立更好的集成，例如将讨论从 Chat 高效地转移到 Composer。

- **Windsurf 被视为竞争对手**：几位用户正在尝试 Windsurf，并分享其具有前景的功能，特别是在处理终端输出和代码库搜索方面。
   - 对比表明，虽然 Windsurf 具有潜力，但 Cursor 在某些工作流中仍保持优势，尽管用户注意到两者在体验上存在差异。

- **对 API Key 限制的担忧**：围绕 Cursor 的 API 使用限制展开了讨论，一些用户考虑使用自己的 API Key 以获得更多灵活性。
   - 对话反映了用户希望更好地管理 API 调用限制以及为活动项目收集 Context 的愿望。

- **Context 管理方面的挫败感**：用户对当前模型的 Context 处理能力表示不满，特别是关于 Claude 的感知限制。
   - 社区正在寻求 Context 管理和功能一致性方面的改进，以增强编码体验。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://dev.to/jasonleowsg/how-to-use-ai-for-coding-the-right-way-4cdn?ref=dailydev">未找到标题</a>: 未找到描述</li><li><a href="https://tenor.com/view/ooft-jealous-girlfriend-jealous-jealous-girlfriend-gif-jealous-girlfriend-move-gif-7998863672934012027">Ooft Jealous Girlfriend GIF - Ooft Jealous girlfriend Jealous - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://changelog.cursor.com/">Cursor - 专为 AI 结对编程设计的 IDE。</a>: 未找到描述</li><li><a href="https://forum.cursor.com/t/how-to-do-fix-in-composer-and-fix-in-chat-actions-from-keyboard/31221">如何通过键盘执行 `Fix in Composer` 和 `Fix in Chat` 操作</a>: 这两个：我在设置中找不到。
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1311798776934961283)** (91 messages🔥🔥): 

> `Gemini 的道德约束, Anthropic MCP 框架, ChatGPT 的能力, Windows 上的语音转文字, 用于编程的 AI 模型` 


- **Gemini 经常拒绝回答无害问题**：用户注意到 **Gemini** 有时会出于感知的道德原因拒绝回答无害的问题，并将其与被认为回答更宽松的 **ChatGPT** 进行了对比。
   - 一位用户幽默地举例说，Gemini 拒绝讨论人工智能，声称它不会参与敏感话题。

- **Anthropic 发布 MCP 框架**：Anthropic 新推出的 **MCP 框架** 允许 Claude 运行服务器，实际上将 Claude 应用转变为一个可以本地创建、读取和编辑文件的 API。
   - 用户对新功能感到兴奋，包括与 **VSCode** 等工具的实时交互。

- **ChatGPT 和语音转文字功能**：一位用户询问了 Windows 版 **ChatGPT** 的语音转文字功能，另一位用户建议通过按下 Windows + H 键来使用 Windows 内置的辅助功能。
   - 该建议旨在为使用 ChatGPT 时提供实时的语音转文字解决方案。

- **编程 AI 模型讨论**：用户讨论了用于编程任务的各种模型，并提出了一个包括 **Claude 3.5 Sonnet** 等在内的排名，引发了关于模型效能偏见的辩论。
   - 对该列表的评论包括对重复提及的困惑，以及排除了 **GPT-4o** 和其他被视为强力竞争者的模型。

- **ChatGPT 的角色控制**：一位用户表达了如何在与 **ChatGPT** 的对话中管理角色控制，强调了引导叙事和纠正不当回复的重要性。
   - 用户分享了确保模型忠于角色意图的策略，强调了一种协作式的叙事方法。



**提到的链接**: <a href="https://x.com/skirano/status/1861081529071346161">Pietro Schirano (@skirano) 的推文</a>: 今天 @Anthropic 发布了 MCP，这是一个允许 Claude 运行服务器的框架，赋予它超能力并有效地将 Claude 应用转变为一个 API。我们创建了一些服务器，我想你们会...

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1311886130362318899)** (7 messages): 

> `App 与浏览器性能对比, 自定义 GPTs 的问题, 文件和照片加载错误` 


- **App 表现优于浏览器**：一位成员指出 *App 端可以正常工作，所以请使用 App 而不是浏览器* 以避免问题。
   - 然而，另一位用户报告称，即使使用 App 也遇到了问题。

- **自定义 GPTs 反复出现加载错误**：成员们对无法加载自定义 GPTs 表示沮丧，称出现了 *加载此 GPT 时发生错误*。
   - 这暗示了一个可能影响使用自定义模型的广泛问题。

- **文件和照片加载问题**：一位用户描述了自昨天以来在加载文件和照片时遇到的问题，突显了持续的技术困难。
   - 这与加载错误的报告一致，表明存在影响各种功能的更广泛问题。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1311828785628319824)** (32 条消息🔥): 

> `Image Captioning 问题, Structured Outputs 问题, 模型推荐, OpenAI 支持的用户体验` 


- **持续的 Image Captioning 问题**：用户报告了上传图像进行字幕生成时遇到的持续问题，称尽管购买了新账号，仍收到提示无法查看图像的消息。
   - *该问题已持续 3-4 天*，影响了他们的工作进度，他们对帮助中心缺乏支持和回应表示沮丧。

- **建议的潜在替代模型**：在图像 Vision 功能持续出现问题的情况下，有人建议切换到 **Claude 3.5 Sonnet** 进行 Image Captioning，部分用户发现该模型功能更完善。
   - 其他用户强调 **OpenAI 的 Vision 能力似乎已损坏**，鼓励使用替代方案以避免项目延误。

- **对 Structured Outputs 的困惑**：一名用户表示，由于在设置中错误放置了 'strict'，导致在使用 Structured Outputs 时出现了随机的 'object' 包装器，对此感到沮丧。
   - 经过 10 小时的调试，他们找到了问题所在，并确认最初错误地放置了 **'strict'**。

- **社区支持与建议**：成员们通过建议分块任务以避免幻觉来提供支持，并在用户解决 Structured Output 问题后**给予了鼓励**。
   - 尽管成员们对 **OpenAI 支持** 表达了共同的沮丧，但他们强调了社区反馈在解决技术问题中的重要性。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1311828785628319824)** (32 条消息🔥): 

> `图像上传问题, Vision 模型故障, Structured Output 错误, 切换到 Claude 模型, 调试最佳实践` 


- **用户面临持续的图像上传问题**：一名用户报告了上传图像的问题，并收到无法查看图像的错误消息，这已阻碍了他们数日的工作。
   - 尽管多次寻求帮助，但支持团队的回应不足，没有邮件或 Discord 回复来解决该问题。

- **Vision 模型已停止运行**：人们对 **Vision 模型** 的功能表示担忧，因为多名用户遇到了该模型突然失效的类似问题。
   - 一位成员建议考虑将 **Claude 3.5 Sonnet** 模型作为生成图像字幕的可行替代方案。

- **Structured Output 错误让用户抓狂**：一名用户表示，尽管正确设置了 strict 属性，但在使用 Structured Outputs 时仍会出现随机的 'object' 包装器，对此感到非常沮丧。
   - 最终，他们意识到 'strict' 设置放置错误，导致了十个小时不必要的调试。

- **处理模型不一致性的建议**：针对错误，一位成员建议将任务分解为更小的块，以防止在上下文中间出现幻觉问题。
   - 分享此建议是为了帮助减轻从模型接收到的输出中的意外行为。

- **沟通与协助方面的不足**：参与者指出缺乏解决持续问题的有效沟通渠道，并对缺乏支持表示沮丧。
   - 鼓励用户遵循发帖指南，以吸引对其问题的关注并确保其诉求得到倾听。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1311798778818461698)** (83 messages🔥🔥): 

> `QwQ 模型配置, DeepSeek 模型性能, 在 Aider 中使用本地模型, OpenRouter 的问题, 用于推理的集成框架` 


- **QwQ 模型配置讨论**：用户讨论了在 architect 模式下使用 **QwQ** 模型，同时使用常规模型执行代码命令的可能性，寻求关于模型互换性的明确说明。
   - *一位成员指出，Aider 允许为各种项目定义模型，从而增强了灵活性*。

- **DeepSeek 展示 SOTA 性能**：**DeepSeek-R1** 模型因在 AIME 和 MATH 基准测试中取得令人印象深刻的结果而受到关注，重点在于开源可访问性和实时思考过程。
   - *另一位用户表示希望 DeepSeek 发布模型权重，以便与 QwQ 一起在集成框架中使用*。

- **Aider 中的本地模型设置**：成员们讨论了创建 `.aider.model.metadata.json` 和 `.aider.model.settings.yml` 文件，以便为 Aider 正确定义本地模型及其配置。
   - *将编辑格式设置为 'whole' 或 'diff' 决定了响应的结构方式，这会影响编辑效率*。

- **OpenRouter 的挑战**：用户发现了 **OpenRouter** 影响模型功能的潜在问题，特别是关于本地服务器的使用和模型检测。
   - *有人担心伪造的实现是否会影响输出和模型行为*。

- **模型设置实验**：一位用户在获得有关文件配置的有用信息后，表示打算尝试 Aider 的各种模型设置。
   - *他们计划测试 Aider 在检测本地模型实现与成熟的 OpenAI 端点之间的差异方面的表现*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/anpaure/status/1862122712845435239">来自 anpaure (@anpaure) 的推文</a>：新的 Qwen 模型在编程任务上与其他 LLM 相比如何？它令人印象深刻，但发布仓促。我在 6 个不同难度的竞赛编程问题上将其与其他 SOTA 模型进行了对比。这里...</li><li><a href="https://tenor.com/view/jonny-frodo-lotr-alright-then-keep-your-secrets-gif-25615953">Jonny Frodo GIF - Jonny Frodo Lotr - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html">高级模型设置</a>：为 LLM 配置高级设置。</li><li><a href="https://api-docs.deepseek.com/news/news1120">🚀 DeepSeek-R1-Lite-Preview 现已上线：释放超强推理能力！ | DeepSeek API 文档</a>：🔍 在 AIME 和 MATH 基准测试中达到 o1-preview 级别的性能。
</li>
</ul>

</div>

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1311854527699222528)** (46 条消息🔥): 

> `Aider 文件管理，来自 Qwen 的 QwQ 模型，Monorepo 设置，OpenAI API 实例，Repository map 经验分享` 


- **Aider 的 .aiderignore 方便了选择性文件包含**：用户讨论了将文件添加到 **.aiderignore** 如何有效地限制出现在 repository map 中的文件，从而在开发过程中增强专注度。
   - 一位成员在最初混淆了终端历史记录与被忽略的文件后，成功测试了这一功能。

- **QwQ 模型在 Aider 中的性能问题**：一位用户询问了在 Aider 中使用来自 Qwen 的 **QwQ 模型**的经验，强调了其推理能力，但也指出了其在生成 commit 时的错误。
   - 社区回复指出，在将该模型与 Aider 集成时存在已知问题。

- **针对 monorepo 配置优化 Aider**：提供了关于如何为 **monorepo** 有效管理 Aider 设置的指导，包括使用 `--input-history-file` 和 `--chat-history-file` 选项。
   - 这些支持侧重于在保持单一 Git 仓库结构的同时组织工作流。

- **连接多个 OpenAI 服务器实例**：一位用户寻求关于为不同角色管理两个独立的 **TabbyAPI** 实例以及如何在 Aider 中配置它们的建议。
   - 社区建议在模型调用中使用 `extra_params` 来为每个实例指定不同的 API key 和 base。

- **关于 Repository map 功能的褒贬不一的体验**：一位成员注意到，禁用 **repository map** 功能有时会带来更好的输出，特别是在保持上下文感知方面。
   - 这引发了一个疑问，即其他人在该功能开启时是否也有类似的上下文混淆经历。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/Qwen/QwQ-32B-Preview">Qwen/QwQ-32B-Preview · Hugging Face</a>: 未找到描述</li><li><a href="https://aider.chat/docs/faq.html#can-i-use-aider-in-a-large-mono-repo">FAQ</a>: 关于 aider 的常见问题。</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html#model-settings">Advanced model settings</a>: 为 LLM 配置高级设置。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1311799638029762641)** (53 messages🔥): 

> `Instruct vs Non-instruct Fine-tuning, Fine-tuning Dataset Formatting, Alternative GPU Recommendations, Creating Custom Datasets, Support for Schedule Free Optimizers` 


- **Instruct vs Non-instruct 微调考量**：成员们讨论了使用 *instruct*（指令型）与 *non-instruct*（非指令型）模型的考量，指出通常如果你的数据集包含超过 1k 条记录，建议使用 base models。
   - 对于 70,000 条记录左右的较小数据集，成员建议先尝试使用 *instruct* 模型。

- **微调的数据集格式化**：一位用户询问了用于微调的 JSON 数据集结构，并提出了一种特定格式，以期获得比传统 QA 对更好的效果。
   - 其他成员提供了参考现有数据集格式化文档的指导，特别强调了遵守微调规则的重要性。

- **备选 GPU 方案讨论**：在关于 GPU 偏好的对话中，一位用户表达了对 NVIDIA 模型的不满，而其他人则强调 NVIDIA GPU 在性能方面仍被认为是最佳选择。
   - 聊天中重申，个人基准测试（benchmarking）对于确定特定任务的最佳架构至关重要。

- **创建自定义数据集**：用户讨论了为训练模型创建自有数据集的必要性，特别提到了寻找合适的日本商业报告数据集的挑战。
   - 明确了 Unsloth 不提供数据集，但在用户提供数据集后会协助进行训练。

- **对 Schedule Free Optimizers 的支持**：有关于 Unsloth 是否支持 *schedule free optimizers* 和 *rslora* 的咨询，并确认已支持 rslora。
   - 讨论表明，通过适当的补丁（patches），实现额外的优化器应该是直接且简单的。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama#id-6.-alpaca-dataset">How to Finetune Llama-3 and Export to Ollama | Unsloth Documentation</a>：为初学者准备的创建定制化个人助手（类似 ChatGPT）并在本地 Ollama 运行的指南</li><li><a href="https://docs.unsloth.ai/tutor">Unsloth Documentation</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3.2, Mistral, Phi, Qwen 2.5 &amp; Gemma LLMs 2-5x faster with 80% less memory</a>：微调 Llama 3.2, Mistral, Phi, Qwen 2.5 &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://docs.unsloth.ai/">Welcome | Unsloth Documentation</a>：Unsloth 新手？从这里开始！
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1311828456983625800)** (4 messages): 

> `RAG usage, Training models, OOM errors` 


- **对 RAG 的赞赏**：一位用户表达了对 **RAG** 的热情，称：“天哪，我爱 RAG。” 这表明了对该模型能力的积极态度。
   - 讨论反映了社区对该模型的认可。

- **训练过程见解**：*silk.ai* 报告称训练过程已经开始，但表示计划终止训练，因为在评估过程中可能会出现 **OOM** 问题。
   - 他们指出，评估很可能会导致显存溢出（out-of-memory）错误，因此决定停止训练。

- **幽默回应**：一位成员以笑声回应，针对早前关于训练的讨论回复了 *LOL*。
   - 这一插话突显了参与者之间轻松的互动氛围。



**提到的链接**：<a href="https://tenor.com/view/chuckles-im-in-danger-ralph-wiggum-the-simpsons-gif-14149962">Chuckles Im In Danger GIF - Chuckles Im In Danger Ralph Wiggum - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1311799893655949332)** (48 条消息🔥): 

> `Unsloth 微调模型，使用 Unsloth 处理私有数据，Grad Norm 波动，LLama 3.1 OOM 错误，Unsloth 中的 SyntaxWarnings` 


- **Unsloth 确保微调期间的数据隐私**：一位用户确认 **Unsloth** 的运行不会向外部传输数据，这取决于用于微调的平台（例如 Google Colab）。
   - 这一澄清让那些担心遵守严格隐私规则的人感到安心。

- **训练期间的 Grad norm 波动**：一位用户报告在微调模型时，即使将 **max_grad_norm** 设置为 **0.3**，**training loss** 和 **grad norm** 仍会出现意外波动。
   - 有建议认为应考虑数据集质量以及使用 **grad accumulation** 等参数的影响。

- **LLama 3.1 遇到 OOM 错误**：一位用户报告在对 **LLama 3.1 8B** 模型进行持续预训练（continual pretraining）期间遇到了 **out of memory (OOM)** 错误。
   - 缓解此问题的建议包括使用更大的 GPU、更小的数据集或减小 **batch size**。

- **建议调整模型参数**：关于何时包含 **head** 和 **embedding** 参数的讨论揭示了在“风格调整”与“灌输新知识”之间，上下文的重要性。
   - 建议指出，风格调整不需要这些参数，而牢固的知识吸收则需要。

- **最新版 Unsloth 中发现 SyntaxWarnings**：一位用户报告在最新版本的 **Unsloth** 中遇到了带有无效转义序列的 **SyntaxWarnings**。
   - 这些警告突显了代码中潜在的问题，可能需要注意以确保功能正常。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/get-st">Unsloth Documentation</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/18sN803sU23XuJV9Q8On2xgqHSer6-UZF?usp=sharing).">Google Colab</a>: 未找到描述</li><li><a href="https://docs.fireworks.ai/fine-tuning/fine-tuning-models)">Introduction - Fireworks AI Docs</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: 查看下方列表以获取我们所有的 notebook：</li><li><a href="https://huggingface.co/datasets">Hugging Face – The AI community building the future.</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1311813499370082304)** (4 messages): 

> `Unsloth fine-tuning, RAG costs, Latent paraphraser, Fine-Tuning or Retrieval paper, Custom tokenizers` 


- **Unsloth 确保微调过程中的数据隐私**：一位用户询问了 Unsloth 的**数据隐私**措施，特别是针对私有数据微调 Llama3 模型时，是否会有任何数据传输到外部。
   - 他们寻求关于特定设置的确认，以确保符合其严格的数据政策。

- **与 RAG 相关的高计算成本**：一位用户指出，由于对上下文长度的广泛需求，**检索增强生成 (RAG)** 可能会产生高昂的计算成本。
   - 这一见解突显了在 AI 模型开发中平衡性能与效率的持续挑战。

- **Latent paraphraser 架构解析**：讨论揭示了 **latent paraphraser** 通过增加一个额外层来修改 Transformer 架构，从而有效地重新分配 LLM Token 的概率。
   - 这增强了输入锚定（input grounding），通过在处理过程中最小化未见过的 Token 来减少噪声。

- **《Fine-Tuning or Retrieval》论文要点**：Ovadia 等人的论文对比了**无监督微调**和 RAG，指出在知识密集型任务中，RAG 的表现始终优于微调。
   - 他们的研究结果表明，在将新信息有效地整合到 LLM 中方面，这具有重要的启示意义。

- **关于表格数据自定义 Tokenizer 的咨询**：一位成员表达了对使用能有效处理表格数据中金额数值的**自定义 Tokenizer** 的兴趣，并引用了 Andrew Karpathy 关于 Tokenizer 的视频。
   - 他们寻求关于将替代 Tokenizer 集成到其数据处理工作流中的方法建议。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2312.05934">Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs</a>：大型语言模型 (LLM) 在其预训练权重中封装了大量的叙述性信息，这从它们在不同领域回答各种问题的能力中得到了证明。然而...</li><li><a href="https://www.youtube.com/watch?v=zduSFxRajkE)">Let&#39;s build the GPT Tokenizer</a>：Tokenizer 是大型语言模型 (LLM) 中一个必要且普遍存在的组件，它负责在字符串和 Token（文本块）之间进行转换。Tokenizer...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1312094245401923594)** (1 messages): 

> `Perplexity Pro Discount, AI Models Access, One-Click Shopping` 


- **Perplexity Pro 提供节日折扣**：Perplexity 团队宣布了首月 Perplexity Pro **2.5 折（75% off）**的促销活动，截止时间为 **太平洋时间 12 月 2 日星期一晚上 11:59**。
   - 此优惠允许新用户访问高级功能，包括增强的搜索能力和文件上传。

- **增强的 AI 模型和来源访问**：用户现在可以通过 Pro 版本访问**最新的 AI 模型**，并允许他们搜索 **3 倍数量的来源**。
   - 这一增强旨在提升整体搜索体验，提高用户的搜索效率。

- **Perplexity Pro 令人兴奋的购物功能更新**：此次促销还包括通过 Buy with Pro 实现的**一键购物**和**免运费**功能。
   - 这些新功能旨在简化购物体验，让用户在这个节日季更加方便。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1311827371124461589)** (74 条消息🔥🔥): 

> `Perplexity Pro 订阅功能、Claude 用户体验、图像生成查询、订阅问题的客户支持、Black Friday 折扣` 


- **用户澄清 Perplexity Pro 功能**：一位用户询问 Perplexity Pro 订阅附带的 5 美元 API 额度如果未使用是否会过期，随后确认只要订阅处于激活状态，该额度每月都会更新。
   - 另一位用户讨论了该平台的图像生成功能，并确认可以通过电脑在线使用，无需额外费用。

- **关于 Claude 和订阅的困惑**：几位用户对他们的订阅表示困惑，其中一位提到在没有当前订阅的情况下竟然可以免费访问 Claude。
   - 另一位用户寻求有关与 Revolut 相关的订阅问题的帮助，得到的建议是通过电子邮件联系支持团队。

- **客户支持困难**：用户讨论了在寻找订阅相关查询的客户支持链接时遇到的挑战，一些人表示联系信息被隐藏在 FAQ 中。
   - 一位用户确认他们被引导至正确的支持邮箱，但对缺乏可见性表示了短暂的沮丧。

- **用户对功能的反馈**：一位用户提供了关于 iOS 应用的反馈，表达了希望在突出显示文本时增加提问澄清问题的功能。
   - 这一请求强调了用户界面需要更多交互功能，以提高应用的易用性。

- **社区分享折扣码**：几位用户讨论了节日期间可能提供的折扣，特别是针对 Perplexity Pro 提供大幅优惠的 Black Friday 活动。
   - 参与者表示有兴趣分享折扣码并参与促销活动，例如新订阅可享受 2.5 折（75% off）的优惠。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/perplexity_ai/status/1862526954064195816?s=46">来自 Perplexity (@perplexity_ai) 的推文</a>: 在这个节日季节更聪明地搜索和购物。获取 Perplexity Pro 首月 2.5 折优惠！访问最新的 AI 模型，搜索 3 倍以上的来源，并上传您自己的文件。此外，还可以获得一键式...</li><li><a href="https://tenor.com/view/cute-baby-sad-agnes-please-gif-16097001420698130990">Cute Baby GIF - Cute Baby Sad - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://giphy.com/gifs/RNXhdtXRmkv1wJ7gOK"> - 在 GIPHY 上查找与分享</a>: 与你认识的每个人发现并分享 stewieeee 制作的 childrenkick 动画 GIF。GIPHY 是你搜索、分享、发现和创建 GIF 的方式。
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1311799585651556384)** (9 条消息🔥): 

> `意识的难题、偶然听到的趣闻、云的含水量、RBAC vs ABA、电池优化` 


- **探索意识的难题**：一位成员表达了对**意识的难题（hard problem of consciousness）**的好奇，思考它是否只是像其他人类创造物一样的另一种工具。
   - *它只是作为另一种人类工具的工具*。

- **关于偶然听到的趣闻的问题**：一位成员提到他们习惯对偶然听到的**趣闻（factoids）**提问，突显了严肃问题与随性查询的融合。
   - 这反映了一种随性而又好奇的学习方式。

- **云及其含水量**：多位成员提出了关于**云层含水量减少**的问题，并将其与更广泛的大气状况讨论联系起来。
   - 对这一话题的兴趣表明了对气象现象的好奇。

- **讨论 RBAC vs ABA**：一位成员寻求了解 **RBAC（基于角色的访问控制）与 ABA（基于属性的访问控制）之间的区别**。
   - 这一询问表明需要澄清技术中的访问控制模型。

- **优化电池续航**：成员们询问了关于**优化电池时长**的建议，寻求延长电池寿命的有效策略。
   - 这反映了对设备效率和可持续性的持续关注。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1311916762966786078)** (5 messages): 

> `Claude 中的 Perplexity, Claude Project Knowledge, Perplexity 的文本文件读取问题, Spaces 的自定义指令` 


- **Perplexity 可以在 Claude 中使用吗？**: 用户很好奇是否可以利用新的 MCP 功能将 **Perplexity** 集成到 **Claude** 中，类似于它在 **Brave** 和 **GitHub** 中的运作方式。
   - 他们强调，这种能力将通过利用 Claude 的 Project Knowledge 来增强性能。

- **Google 与 Claude 的集成？**: 关于在 **Claude** 中集成 **Google** 的类似咨询也被提出，旨在澄清其运行机制。
   - 成员们热衷于了解在这种背景下如何利用搜索功能。

- **Perplexity 的文本文件读取能力**: 一位成员询问 **Perplexity** 无法可靠读取文本文件的问题是否已得到解决。
   - 他们对可能解决这一限制的任何潜在长期记忆功能表示关注。

- **Claude Spaces 中自定义指令的问题**: 用户对 Claude Spaces 的 **自定义指令 (custom instructions)** 的有效性表示担忧，这些指令似乎与现有的“自我介绍”提示词发生冲突。
   - 用户正在寻求关于这些指令如何复合或交互的澄清。


  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1312130943527424081)** (1 messages): 

> `HF 搜索问题, 图像分析` 


- **HF 搜索问题已解决**: **HF 搜索无法工作**的问题已经解决，这让用户松了一口气。
   - 随公告附带了一张图片以纪念此次修复，表明社区迎来了一个积极的更新。

- **分享了图像分析**: 公告中附带了一张关于 HF 搜索问题的图片，提供了视觉确认。
   - 虽然未分享图像分析的具体细节，但可能有助于理解解决方案。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1311852052200030258)** (56 messages🔥🔥): 

> `LM Studio AIDE 集成, LM Studio 中的 Llama 3.1 模型, LM Studio 网络问题, LM Studio 中的文档交互, Mac 上的 GUI 访问问题` 


- **成功的 LM Studio AIDE 集成**: 用户报告成功将 LM Studio 端点集成到 AIDE sidecar，实现了完全本地的代码编辑器体验。
   - 这种集成为寻求本地开发环境的用户展示了改进的功能。

- **寻找 Base Llama 3.1 模型**: 一位用户询问如何在 LM Studio 中访问 **Llama 3.1 8B** 的基础模型 (base model)，并指出似乎只有指令微调 (instruction-tuned) 变体可用。
   - 社区成员指出 [Hugging Face 仓库](https://huggingface.co/meta-llama/Llama-3.1-8B) 是获取基础模型的潜在来源。

- **网络连接问题**: 几位用户讨论了在确认本地访问正常的情况下，从本地网络外部访问 LM Studio 的问题。
   - 建议包括检查防火墙设置以及考虑使用像 ngrok 这样的隧道服务进行远程访问。

- **与本地文件交互**: 新用户对如何在 LM Studio 中与本地文件交互感到好奇，特别是询问了文档附件功能。
   - 社区澄清目前只能将单个文件附加到聊天会话中，并参考了文档以获取进一步指导。

- **Mac GUI 访问故障**: 一位用户对在 Mac 上测试 headless 选项后无法访问 LM Studio GUI 表示沮丧。
   - 虽然有建议通过 Finder 访问应用程序，但用户在 GUI 可用性方面仍然遇到困难。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF">lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://lmstudio.ai/docs/basics/rag">与文档聊天 - 在本地运行 LLM | LM Studio 文档</a>: 如何将本地文档作为额外上下文提供给 LLM</li><li><a href="https://huggingface.co/meta-llama/Llama-3.1-8B">meta-llama/Llama-3.1-8B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct">meta-llama/Llama-3.1-8B-Instruct · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1311802272220581901)** (17 messages🔥): 

> `Seasonic PSU 寿命、a770 性能对比、PC 组装建议、Intel 与 AMD 处理器、Qwen2.5-14b 的性能` 


- **Seasonic PSU 寿命超过其他 PC 组件**：一位成员提到，尽管由于灰尘原因每隔几年就需要更换一次 PSU，但他们的 **Seasonic PSU** 寿命比其他 PC 组件都要长。
   - 他们形容对该 PSU 性能的使用体验感到*非常*满意。

- **与 7800xt 相比，a770 表现吃力**：另一位成员分享道，他们的 **a770** 在运行 Qwen2.5-14b q4_0 时仅达到了 **11t/s**，显著低于 **7800xt** 达到的 **40t/s**。
   - 他们指出 *q4_k_m 无法使用*，并发现 sycl 后端的加速效果微乎其微。

- **关于最佳 PC 配置的讨论**：在一次关于 PC 组装的讨论中，一位用户询问配备 **Intel Core i9 14900KF** 和 **NVIDIA GeForce RTX 4090** 的方案是否足以学习 LLM。
   - 其他人建议避开 **第 13/14 代 Intel**，转而选择 ***AMD Ryzen 7000 或 9000 系列*** 或 **第 12 代 Intel**。

- **对 a770 定价的担忧**：一位成员表示由于折扣有意购买 **a770**，但最终决定等待下一代产品的发布。
   - 建议认为，最好等待 GPU 技术的进一步发展。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1311827576632774697)** (29 messages🔥): 

> `资源竞争的降级处理、GPU 任务提交管理、SLURM 和 Kubernetes 的使用、AI 与 Crypto 的交集、学术资源的开放获取` 


- **讨论资源竞争的降级处理**：成员们对**资源竞争的降级处理**以及互联网不受监管的增长所带来的影响表示担忧，并对 AI 驱动的隐私解决方案提出质疑。
   - 有人建议识别*流氓 AI 攻击的预警信号*（这些攻击可能会利用易受攻击的设备），并强调在 AI 保护方面需要社区领导力。

- **汇集昂贵的 GPU VM 以进行任务提交**：有人询问关于管理用于任务提交的昂贵 GPU VM 池的**开源解决方案**，表明需要有效的资源记账管理。
   - 回复强调了 **SLURM 队列**和 Kubernetes 的使用，尽管对其在高信任环境中的适应性存在怀疑。

- **低信任环境下 SLURM 的最佳实践**：成员们探讨了是否存在一种专门的 **SLURM** 设置，允许在信任度较低的环境中进行私有存储分割，并对潜在解决方案提出了各种见解。
   - 分享的一些经验包括利用 **network-filesystems** 和 S3 前缀进行权限管理，但同时也建议警惕不必要的复杂性。

- **不欢迎 AI 与 Crypto 的讨论**：一位参与者询问了 **AI 与 Crypto** 的交集，对此一位成员评论说，此类讨论在当前频道通常是不受欢迎的。
   - 这反映了保持讨论集中化的愿望，并可能将更广泛的话题引导至更合适的频道。

- **学术资源协作**：提议建立一个服务器供成员分享**高质量论文和资源**，以便在没有无关干扰的情况下持续获取信息。
   - 这一倡议可以增强社区内的协作和资源共享，旨在实现高效且精简的交流。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1312028602300366939)** (23 条消息🔥): 

> `Poincare Ball Embedding, Hyperbolic Geometry, Graph Distortion, Embedding Trees, HyperE` 


- **Poincare Ball Embedding 详解**：将数据嵌入到 **Poincare ball** 本质上意味着度数（degree）较高的点更接近原点，以在向 **less curvature**（低曲率）区域移动时保持邻接性。
   - 讨论中对 **Poincare ball** 的边缘进行了自我纠正，指出边缘是无穷远点，点实际上无法驻留在那里。

- **Hyperbolic Embedding 资源**：**HyperE** 研究团队提供了多种优化结构化对象（如知识图谱）嵌入的方法，重点参考了 **Nickel & Kiela (2017)** 和 **Chamberlain et al. (2017)** 的论文。
   - 这些 **hyperbolic embeddings** 可以有效地在低维空间中保持图距离，并应用于 **NLP** 和 **knowledge base completion** 等领域。

- **Graph Distortion 问题**：一位成员提出，嵌入过程可能无法遵循某些数据集的结构，特别是在高密度图中，如 **fully-connected graphs (FC)**。
   - 讨论建议使用一种启发式方法，通过与 **equivalent tree structures** 进行比较来估计失真（distortion），从而更好地理解嵌入质量。

- **低失真的条件**：虽然在特定条件下图嵌入的失真可以很低，但这并不具有普适性；由于节点数量与度数的问题，某些图天生就不适合嵌入。
   - 图嵌入文献表明，特定的数学条件决定了嵌入实现低失真的可能性。

- **图嵌入的数学原理**：有大量数学文献讨论如何将图嵌入到 **hyperbolic space**，尽管许多人发现这很难完全掌握。
   - 评估嵌入失真的一个良好启发式方法是评估嵌入与逻辑等效树结构的对比情况。



**提到的链接**：<a href="https://hazyresearch.stanford.edu/hyperE/">HyperE</a>：未找到描述

  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1312051675342573590)** (5 条消息): 

> `AutoML Challenges, TaskSet Dataset, Neural Architecture Design, Equivariant vs Non-equivariant Networks` 


- **AutoML 面临基础任务**：一位成员提到，目前大多数 **AutoML** 都在处理非常简单的任务，并强调了构建大规模学习曲线数据集的资金限制。
   - 他们指出目前最好的选择是 **TaskSet**，但也承认它已经相当过时了。

- **TaskSet 助力优化器训练**：关于 **TaskSet** 数据集的摘要揭示了其独特的规模和多样性，包含一千多个用于训练和评估优化器的任务。
   - 该数据集促进了超参数列表的 **meta-learning**，从而在效率上比随机搜索有显著提升。

- **等变网络（Equivariant Networks）提升效率**：一篇论文探讨了 **equivariant and non-equivariant networks** 如何随模型大小和计算量的变化而扩展，发现等变性（equivariance）增强了数据效率。
   - 实验结果显示，虽然非等变模型在经过足够训练后可以缩小这一差距，但等变模型在所有计算预算下都优于它们。

- **质疑神经架构设计方法**：关于针对特定问题量身定制神经架构与从数据中学习的效率问题引发了讨论。
   - 一位成员表示有兴趣了解关于等变性和计算预算分配的发现是否也适用于其他任务。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2410.23179v1">Does equivariance matter at scale?</a>：在拥有大规模数据集和充足计算资源的情况下，为每个问题的结构和对称性设计神经架构是否有益？还是从数据中学习它们更有效？我们研究了...</li><li><a href="https://openreview.net/forum?id=PghuCwnjF6y">TaskSet: A Dataset of Optimization Tasks</a>：我们介绍了 TaskSet，这是一个用于训练和评估优化器的任务数据集。TaskSet 在规模和多样性上都是独特的，包含从图像分类到...的一千多个任务。
</li>
</ul>

</div>

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1311836435329974274)** (17 条消息🔥): 

> `HF Tokenizer 处理、自定义 Tokenizer 注意事项、Evaluation Harness 模型函数、模型中的生成参数` 


- **理解 Eval Harness 中的 HF Tokenizer**：关于 eval harness 在对序列进行 tokenize 时是使用 `add_special_tokens=True` 还是 `False` 存在困惑，特别是关于在生成任务中如何处理 EOS tokens。
   - 成员们讨论认为，通常在模型中**仅应添加 BOS tokens**，同时省略 EOS tokens，特别是在构建自定义 tokenizer 时。

- **手动 EOS Token 管理**：一位成员考虑修改其 tokenizer，在 tokenization 过程中禁用 EOS token，并在训练循环中手动添加它。
   - 这种方法被认为是实用的，并有望避免在使用 HF 模型的各种框架之间出现兼容性问题。

- **Generate Until 函数讨论**：为了使用 eval harness 评估自定义模型，有必要实现一个 `generate_until` 函数来处理各种生成参数，包括 `until`、`do_sample` 和 `max_gen_toks`。 
   - 关于该函数是否需要额外的关键字参数的询问得到了澄清：`max_gen_toks` 是 eval harness 特有的，而其他参数则与标准的 HF 实践保持一致。

- **为自定义模型子类化 HFLM**：成员们建议通过子类化 HFLM 并重载 `model_generate` 和 `_model_call` 等方法，来简化自定义模型的集成。
   - 这种方法被认为是框架内处理自定义模型评估的一种更直接的方式。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/5680a2e6b">GitHub - EleutherAI/lm-evaluation-harness at 5680a2e6b5cf1a1621d8ff68d3d0e83e8b2731d3</a>: 语言模型 few-shot 评估框架。 - GitHub - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/5680a2e6b5cf1a1621d8ff68d3d0e83e8b2731d3/lm_eval/models/huggingface.py#L771-L795">lm-evaluation-harness/lm_eval/models/huggingface.py at 5680a2e6b5cf1a1621d8ff68d3d0e83e8b2731d3 · EleutherAI/lm-evaluation-harness</a>: 语言模型 few-shot 评估框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/.">GitHub - EleutherAI/lm-evaluation-harness: 语言模型 few-shot 评估框架。</a>: 语言模型 few-shot 评估框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/9169899b4966b4161719e54d41258345df03aaa0/lm_eval/models/huggingface.py#L1308)">lm-evaluation-harness/lm_eval/models/huggingface.py at 9169899b4966b4161719e54d41258345df03aaa0 · EleutherAI/lm-evaluation-harness</a>: 语言模型 few-shot 评估框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/9169899b4966b4161719e54d41258345df03aaa0/lm_eval/models/huggingface.py#L857)">lm-evaluation-harness/lm_eval/models/huggingface.py at 9169899b4966b4161719e54d41258345df03aaa0 · EleutherAI/lm-evaluation-harness</a>: 语言模型 few-shot 评估框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/9169899b4966b4161719e54d41258345df03aaa0/lm_eval/models/huggingface.py#L831)">lm-evaluation-harness/lm_eval/models/huggingface.py at 9169899b4966b4161719e54d41258345df03aaa0 · EleutherAI/lm-evaluation-harness</a>: 语言模型 few-shot 评估框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/5680a2e6b5cf1a1621d8ff68d3d0e83e8b2731d3/lm_eval/models/huggingface.py#L1299)">lm-evaluation-harness/lm_eval/models/huggingface.py at 5680a2e6b5cf1a1621d8ff68d3d0e83e8b2731d3 · EleutherAI/lm-evaluation-harness</a>: 语言模型 few-shot 评估框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/5680a2e6b5cf1a1621d8ff68d3d0e83e8b2731d3/lm_eval/api/model.py#L354-L355).">lm-evaluation-harness/lm_eval/api/model.py at 5680a2e6b5cf1a1621d8ff68d3d0e83e8b2731d3 · EleutherAI/lm-evaluation-harness</a>: 语言模型 few-shot 评估框架。 - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1312105380041461810)** (1 messages): 

> `功能请求投票，额外请求频道` 


- **立即为热门功能请求投票！**：鼓励成员[在此为他们最看重的功能请求投票](https://link.to/vote)，以帮助确定未来开发的优先级。
   - 此外，对于未列出的任何请求，可以使用 <#1107397803266818229> 进行提交。

- **额外功能请求频道**：提供了一个专用频道 (<#1107397803266818229>)，供用户提交投票中未涵盖的任何功能请求。
   - 这使得社区能够针对所需功能提供更广泛的反馈。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1311831311752626216)** (57 messages🔥🔥): 

> `Pixtral Large 的能力、关于模型响应的疑虑、特定提供商功能、OpenRouter 中的图像生成、Llama 3.2 的结构化输出` 


- **Pixtral Large 给用户留下深刻印象**：用户注意到 **Pixtral Large** 提供了卓越的性能和**庞大的免费额度**，鼓励通过 [console.mistral.ai](https://console.mistral.ai) 轻松访问。另一位用户从 **Hermes 405b** 切换到 **Pixtral**，发现即使不更改 Prompt，效果依然很好。

- **对模型身份识别的困惑**：围绕模型训练展开了讨论，一些人澄清说模型本质上并不知道自己的身份，而是经常根据训练数据“幻觉”出细节。这引发了疑问：尽管有这些解释，为什么困惑依然存在。

- **关于成本计算方法的问题**：一位用户询问 **/api/v1/generation** 端点是否有任何费率，以及如何准确估算生成成本。建议包括使用 **Helicone** 进行跟踪，并澄清目前为了进行精确的成本评估，生成端点是必要的。

- **OpenRouter 图像生成的未来**：虽然图像生成目前不在 **OpenRouter** 的近期路线图中，但不排除未来实现的可能性。讨论表明用户对图像模型能力的兴趣日益浓厚。

- **Llama 3.2 结构化输出的挑战**：用户报告说在使用 **Llama 3.2-vision-instruct** 获取**结构化输出**时遇到困难，指出虽然它声称具有 JSON 输出能力，但与 **Gemini Flash** 等替代方案相比，性能表现滞后。会议强调，对此类功能的支持在很大程度上取决于所使用的推理软件。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.helicone.ai/getting-started/integration-method/openrouter">OpenRouter 集成 - Helicone 开源 LLM 可观测性</a>：未找到描述</li><li><a href="https://openrouter.ai/docs/provider-routing">提供商路由 | OpenRouter</a>：跨多个提供商路由请求</li><li><a href="https://openrouter.ai/meta-llama/llama-3.2-90b-vision-instruct">Llama 3.2 90B Vision Instruct - API、提供商、统计数据</a>：Llama 90B Vision 模型是一款顶级的 900 亿参数多模态模型，专为最具挑战性的视觉推理和语言任务而设计。它在图像描述方面提供了无与伦比的准确性...</li><li><a href="https://mistral.ai/news/pixtral-large/">Pixtral Large</a>：Pixtral 成长了。</li><li><a href="https://docs.helicone.ai/getting-started/integra">简介 - Helicone 开源 LLM 可观测性</a>：未找到描述</li><li><a href="https://openrouter.ai/rankings">LLM 排行榜 | OpenRouter</a>：按应用使用情况排名和分析的语言模型
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1311967005263265883)** (5 messages): 

> `自定义提供商密钥` 


- **开发者推动获取自定义提供商密钥的权限**：多位开发者表达了对访问**自定义提供商密钥**的兴趣，表明社区对该功能有强烈需求。
   - *一位成员在请求访问权限时提到*：“感谢你们所做的出色工作！”

- **开发者的集体请求**：包括被识别为 **monomethylhydrazine** 和 **kit18** 在内的几位用户也表达了在某些提供商处使用自己密钥的愿望。
   - 这一反复出现的主题突显了开发者对这些功能需求的共识。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1311973619630477393)** (2 条消息): 

> `NVIDIA GPU 上的并行处理，在初学者板块发布` 


- **寻求并行处理问题的帮助**：一名成员表达了在 **NVIDIA GPU** 上进行 **parallel processing**（并行处理）时遇到的困难并寻求指导。
   - 对话转向确保技术讨论被引导至合适的渠道，以便获得更好的帮助。

- **建议在初学者板块发布**：另一位成员建议不要在这里讨论具体技术问题，并推荐将问题发布在 **beginner** 板块。
   - 此举旨在简化讨论流程，并引导原帖作者到更适合其咨询的区域。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1311952730809696288)** (9 条消息🔥): 

> `元编程提案，从源码构建 Triton，离线编译依赖` 


- **元编程提案引起关注**：一位用户分享了一个针对 Triton 的 [元编程提案](https://github.com/triton-lang/triton/pull/5284)，旨在解决当前的局限性，并收集社区反馈。
   - 一些成员对该提案表示感兴趣，但对其语义的清晰度提出质疑，建议增加示例以增强理解。

- **从源码构建 Triton 的说明**：一位新人询问了从源码构建 Triton 所需的**最小内存**，并寻求社区帮助。
   - 在收到包括路径调整在内的排错建议后，该用户报告称，在将 WSL2 内存增加到 **26GB** 以避免 out-of-memory（内存不足）错误后，成功完成了构建。

- **关于离线编译的担忧**：另一位成员提出了关于在 Ubuntu Docker 容器中以**离线模式**从源码构建 Triton 的问题，以及手动收集依赖项的必要步骤。
   - 他们寻求关于离线编译的便捷配置建议，以及成功构建所需的**最小依赖项**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/triton-lang/triton/pull/5284">[FRONTEND][WIP][RFC] Rewrite AST conversion to improve metaprogramming by kuterd · Pull Request #5284 · triton-lang/triton</a>：问题陈述：Triton 当前元编程的局限性导致 Torch Inductor 等主要用户不得不求助于基于字符串的模板。此 RFC 旨在解决其中的一些问题...</li><li><a href="https://github.co">GitHub · 在单一协作平台上构建和交付软件</a>：加入全球应用最广泛、AI 驱动的开发者平台，数百万开发者、企业和最大的开源社区在这里构建推动人类进步的软件。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1312082037125156906)** (1 条消息): 

> `cuBLAS 异步加载，自定义 Kernel 性能，SASS 指令，CuTe 模板，吞吐量考量` 


- **使用 SASS 剖析 cuBLAS 异步加载**：在利用 [cuBLAS](https://developer.nvidia.com/cublas) 分析自定义 Kernel 时，一位用户观察到异步加载的 SASS 指令使用了 `LDGSTS.E.BYPASS.LTC128B.128.CONSTANT`，而他们的代码生成的是 `LDGSTS.E.BYPASS.LTC128B.128`。
   - 他们对 **CONSTANT** 部分的含义及其对性能的潜在影响感到好奇。

- **A100 上的基准测试揭示潜在问题**：该用户正在 **A100** 上对自定义 Kernel 进行基准测试，并且不确定 SASS 指令的差异是否相关，因为他们目前的性能远未达到理想水平。
   - 他们正在探索各种选项，以寻求更好的吞吐量（throughput）和效率。

- **关于 SASS 和吞吐量的问题**：用户提出了两个具体问题：SASS 中的 **CONSTANT** 代表什么，以及这两类指令之间是否存在显著的**吞吐量考量**。
   - 这些疑问突显了对优化 Kernel 实现性能的深入探索。


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1311799587954102333)** (26 条消息🔥): 

> `Triton 性能、Fusion 策略、PyTorch 中的内存使用、Max autotune 设置、NANOgpt 集成` 


- **Triton 比 cuBLAS 慢**：讨论揭示了 **Triton** kernel 的性能通常不如 **cuBLAS**，特别是由于尚未采用 **TMAs** 或持久化（persistent）的未优化模板。
   - 成员们强调了对 **fusion** 可能导致计算变慢的担忧，特别是在计算受限（compute-bound）场景中带有沉重尾部操作（heavy epilogues）的情况。

- **Max Autotune 未融合 RELU Squared**：即使设置了 **TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON**，一位成员仍对 **RELU squared** 未被融合表示沮丧。
   - 这引发了关于 autotune 有效性的疑问，以及在保留 **cuBLAS** 以实现更快速操作的同时，处理 Triton 较慢 kernel 的复杂性。

- **融合 Matmul 和 Pointwise 操作**：将 **matmul** 融合进 pointwise 操作缺乏收益，这被认为更多是关于确定哪些场景有收益的问题，而非技术难度。
   - 成员们指出，了解何时 fusion 会导致操作变慢对于避免对 **Inductor** 性能产生困惑至关重要。

- **Torch Snapshot 工具中的内存使用**：一位用户对使用 **torch memory snapshot tool** 时看到的显著 **'Unknown'** 内存占用提出疑问，并分享了相关截图供参考。
   - 这引发了对 PyTorch 应用中内存管理和追踪清晰度的担忧。

- **使用 Thunder Kittens 的潜力**：一位成员推测，将基于 **Thunder Kittens** 的 matmul 实现集成到 PyTorch 中，可能会解决讨论中的一些性能问题。
   - 这一想法源于围绕 BF16 处理的复杂性以及为获得更好性能而优化 kernel 的需求。


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/)** (1 条消息): 

melanimahes: https://arxiv.org/pdf/2411.17116
  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1311968374615707648)** (1 条消息): 

> `Diffusion Models Overview, Classifier-free Diffusion Guidance, Perspectives on Diffusion Models, Noise Schedules in Diffusion Models` 


- **Diffusion Models 占据中心舞台**：Diffusion Models 已成为生成图像和声音等感知信号的**首选模型**，凭借*更好的模式覆盖（mode coverage）*和**更快的采样（sampling）**超越了传统模型。其构建过程包括逐渐将数据转换为噪声，并训练神经网络来逆转这一过程。
   - 自 Song & Ermon 在 2019 年发表[开创性论文](https://arxiv.org/abs/1907.05600)以来，与 Diffusion Models 相关的关注度迅速上升，引发了巨大的研究势头。

- **Classifier-free Diffusion Guidance 显著增强输出**：正如博客文章中所讨论的，**Classifier-free Diffusion Guidance** 的实现以极低的成本显著增强了条件 Diffusion Models 的结果。这项技术对于优化 [OpenAI’s DALL·E 2](https://openai.com/dall-e-2/) 和 [Google’s Imagen](https://imagen.research.google/) 中的图像生成至关重要。
   - 这种方法使 Diffusion Models 变得更加优越，在没有复杂开销的情况下提升了样本质量。

- **多元视角推动 Diffusion 研究**：探索关于 Diffusion Models 的不同视角揭示了挑战和有益的见解。Diffusion 的各种特性突显了其**灵活性**，并激发了各篇研究论文中的创新想法。
   - 该综述对比了不同研究论文的方法，使得理解它们之间的关联动态既*令人沮丧又富有启发性*。

- **重新评估 Noise Schedules**：Diffusion Models 中使用的 **Noise Schedule** 是一个关键但往往令人困惑的设计元素，它决定了扩散过程中的噪声强度。一篇博客文章主张重新构建关于 Noise Schedules 的讨论，以便更清晰地理解和应用。
   - 作者的主观见解旨在阐明不同的噪声水平如何影响 Diffusion Models 的性能，为这一略具争议的话题提供了全新的视角。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://sander.ai/2022/01/31/diffusion.html">Diffusion models are autoencoders</a>：Diffusion Models 在过去两年中变得非常流行。Diffusion Models 与 Autoencoders 之间存在着一种被低估的联系。</li><li><a href="https://sander.ai/2022/05/26/guidance.html">Guidance: a cheat code for diffusion models</a>：一篇关于 Diffusion Guidance 见解的简短文章。</li><li><a href="https://sander.ai/2023/07/20/perspectives.html">Perspectives on diffusion</a>：关于 Diffusion 的视角，或者说 Diffusion Models 如何同时是 Autoencoders、深度隐变量模型、Score Function 预测器、逆向 SDE 求解器、基于 Flow 的模型、RNN 和自回归模型等...</li><li><a href="https://sander.ai/2024/06/14/noise-schedules.html">Noise schedules considered harmful</a>：Noise Schedule 是 Diffusion Models 的关键设计参数。不幸的是，它是一个多余的抽象，将模型的几个不同方面纠缠在一起。我们真的需要它吗？
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1312019871671914497)** (2 条消息): 

> `Series A Docs Process, HR Reporting Protocols` 


- **德国公证人宣读 Series A 文档**：在德国，公证人在创始人面前大声朗读 Series A 融资文档的每一个字，这被用户描述为**史前时代的疯狂**。
   - 看到这一幕，一位参与者幽默地提到，他们还有 **GDP 要增长**，强调了这种情况的荒谬性。

- **对 HR 报告的担忧**：一位用户对公证人的流程表示担忧，建议应该将其报告给 **apaz 的 HR**。
   - 这引发了关于此类做法是否适合现代商业环境的质疑。



**提到的链接**：<a href="https://x.com/nathanbenaich/status/1862208030596636770">来自 Nathan Benaich (@nathanbenaich) 的推文</a>：已经 12 小时了——在德国，公证人在创始人面前大声朗读 Series A 文档的每一个字。本人到场。伙计们，我们还有 GDP 要增长。纯粹是史前时代的疯狂。

  

---

### **GPU MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1311930671626981439)** (2 messages): 

> `BitNet b1.58, 1-bit LLMs, Open-Source Models, RedPajama Dataset, Dolma Dataset` 


- **BitNet b1.58 模型发布**：使用 [RedPajama dataset](https://github.com/togethercomputer/RedPajama-Data) 训练了 **100B tokens**，BitNet b1.58 模型在 PPL 和 zero-shot 准确率方面展现出令人期待的结果。
   - 训练细节记录在他们的论文中：[The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)，模型可在开源 [repo](https://huggingface.co/1bitLLM) 中获取。

- **OLMo-Bitnet-1B 作为概念验证**：[OLMo-Bitnet-1B](https://huggingface.co/NousResearch/OLMo-Bitnet-1B) 是一个 1B 参数模型，在 [Dolma dataset](https://huggingface.co/datasets/allenai/dolma) 的前 **60B tokens** 上进行了训练，强调了其研究性质。
   - 可以在 [wandb report](https://api.wandb.ai/links/emozilla/evltqiv7) 中探索与标准 fp16 权重的对比，展示了不同训练方法的有效性。

- **训练超参数详情**：这些模型使用特定的超参数进行训练，包括相应 [documentation](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf) 中推荐的两阶段 LR 和权重衰减。
   - 性能详情反映了报告模型与复现模型之间的不同结果，为模型有效性提供了见解。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/1bitLLM/bitnet_b1_58-3B">1bitLLM/bitnet_b1_58-3B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/NousResearch/OLMo-Bitnet-1B">NousResearch/OLMo-Bitnet-1B · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1312010021395107860)** (1 messages): 

> `Japanese LLM evaluation, Open Japanese LLM Leaderboard, Hugging Face collaboration` 


- **发布 Open Japanese LLM Leaderboard**：一项关于 **[Open Japanese LLM Leaderboard](https://huggingface.co/spaces/llm-jp/open-japanese-llm-leaderboard)** 的激动人心的公告发布，该榜单旨在通过超过 **20 个数据集**和任务来评估各种日语 LLM。
   - 该倡议是 **[LLM-jp](https://llm-jp.nii.ac.jp/en/)** 项目与 **Hugging Face** 的合作成果，旨在增强对日语 LLM 机制的理解。

- **关注日语语言模型性能**：日语 LLM 的发展一直滞后于英语，因此需要全面的性能评估。
   - 这一公告引起了特别是对母语技术进步感兴趣的日本 **HPC engineers** 的关注。



**提及的链接**：<a href="https://huggingface.co/blog/leaderboard-japanese">Introducing the Open Leaderboard for Japanese LLMs!</a>：未找到描述

  

---

### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1311799317014646854)** (14 条消息🔥): 

> `非 Warp 特化实现、ThunderKittens 与 ThunderMittens 的统一、TK 与 TM 之间的 API 契约、TK 的自动优化器、Triton vs ThunderKittens 特性对比` 


- **探索非 Warp 特化实现**：一位成员询问是否存在非 Warp 特化的实现，另一位成员确认目前没有针对 FP8 的预构建 Kernel，但表示可以协助创建一个。
   - 他们还分享了 TK 仓库中现有的 [非 Warp 特化 Kernel](https://github.com/HazyResearch/ThunderKittens/tree/main/kernels/fftconv/non_pc) 链接。

- **Tile 抽象统一了 ThunderKittens 和 ThunderMittens**：成员们讨论了 **ThunderKittens** 和 **ThunderMittens** 之间的主要统一因素，认为 **Tile 抽象**对于 Tensor Core 兼容性至关重要。
   - 有人指出，这种抽象允许**直接控制寄存器使用**，为在 Tile 上运行的库函数奠定了基础。

- **ThunderKittens 与 ThunderMittens 之间的 API 契约**：有人提问 **ThunderKittens** 和 **ThunderMittens** 之间是否存在 API 契约，强调了兼容性的重要性。
   - 这引发了关于框架如何看待 API 关系及其围绕 Kernel 功能构建结构的讨论。

- **对 ThunderKittens 自动优化器的渴望**：一位成员表达了对 **ThunderKittens** 自动优化器的兴趣，强调其本质是一个“一次编写，多次运行”的系统。
   - 他们对包含这种优化特性的领域特定语言 (DSLs) 表示赞赏。

- **Triton 与 ThunderKittens 的特性对比**：随后讨论了 **ThunderKittens** 如何通过显式暴露 Layouts、异步操作和共享内存分配来区别于 **Triton**。
   - 此外，他们还提到了将这些功能直接嵌入 **CUDA/Metal** 中的重要性。



**提到的链接**：<a href="https://github.com/HazyResearch/ThunderKittens/tree/main/kernels/layernorm/non_pc">ThunderKittens/kernels/layernorm/non_pc at main · HazyResearch/ThunderKittens</a>：用于高速 Kernel 的 Tile 原语。通过在 GitHub 上创建账户为 HazyResearch/ThunderKittens 的开发做出贡献。

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1311901917487824956)** (42 条消息🔥): 

> `Hermes 3 更新、Mistral 平台问题、加密与 AI 中的 Truth Terminal、Discord 求职、AI 与加密社区的交集` 


- **Hermes 3 的询问引发关注**：一位成员提出了关于 **Hermes 3** 的问题，其他人暗示它可能与旧的 **O1 风格**有关。
   - 这一讨论表明人们对 **Hermes** 的进展持续保持好奇。

- **Mistral 平台的新挑战**：有人对 **Mistral AI** 平台的问题表示担忧，特别是关于模型选择方面，因为现在它默认只有一个选项。
   - 还有评论提到**图像生成**功能受到限制，导致用户产生了一些困惑。

- **Truth Terminal 的奇特叙事**：一位成员分享了关于加密领域 **Truth Terminal** 叙事的见解，将其描述为一个半自主 AI 正在创建自己的宗教。
   - 他们强调了这与 AI Alignment（AI 对齐）讨论的联系，标志着 **AI 与加密社区**的一个独特交集。

- **对 Discord 求职有效性的怀疑**：成员们讨论了在 Discord 上求职的有效性，并对在以 AI 为中心的群组中提及区块链经验的可行性表示怀疑。
   - 有人担心这种方式可能会被视为“可疑”，表明了对该平台进行职业社交的复杂情绪。

- **AI 社区内的不同派系**：讨论涉及 AI 爱好者中的不同**派系**，包括关注安全（Safety）和加速（Acceleration）的派系，以及一些人如何将 AI 视为加密创业的替代品。
   - 这突显了社区内多样化的兴趣和观点，一些成员仅仅是为了好玩而参与。



**提到的链接**：<a href="https://www.chainofthought.xyz/p/goat-the-gospel-of-goatse">GOAT: The Gospel of Goatse</a>：为什么 Truth Terminal 是对社会日益痴迷于自主 AI Agent 的一次非对称押注

  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1311806401323991080)** (5 条消息): 

> `低比特量化效应，精度感知缩放定律，三值量化与欠训练模型，FP4 效率` 


- **低比特量化更利于欠训练的 LLM**：研究表明，与经过大量数据训练的小型模型相比，**低比特量化**在规模较大但欠训练的 LLM 中导致的性能退化较小。通过研究超过 1500 个 LLM 检查点得出的缩放定律，有助于量化**量化诱导退化** (QiD) 与模型大小及训练 Token 等因素之间的关系。
   - 该研究强调，调整量化可以为 LLM 的**训练水平**以及不同模型大小所需的训练 Token 需求提供见解。

- **引入精度感知缩放定律**：一种新方法提出了用于训练和推理的**精度感知缩放定律**，强调低精度会影响模型的**有效参数量**和整体性能。研究结果表明，虽然低精度训练看起来可能是最优的，但随着训练数据的增加，它可能会导致损失增加并降低模型有效性。
   - 这项工作暗示利用较低精度可能是计算最优的，但同时也警告说，随着数据输入的增加，**训练后量化**的影响会显著增大。

- **三值量化的效用存疑**：观察发现，被称为 **BitNet** 的**三值量化**仅在模型**欠训练**时表现更好，这让人对其整体效能产生怀疑。这表明对于现有的模型规模，可能会重新转向使用 **FP4** 作为最佳的数值权重表示。
   - 此外，在 QaT 等方法中，**量化与小型模型**之间的关系进一步增强了反对广泛采用三值量化的论据。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2411.04330">Scaling Laws for Precision</a>：低精度训练和推理会影响语言模型的质量和成本，但目前的缩放定律并未考虑到这一点。在这项工作中，我们设计了“精度感知”缩放定律...</li><li><a href="https://arxiv.org/abs/2411.17691">Low-Bit Quantization Favors Undertrained LLMs: Scaling Laws for Quantized LLMs with 100T Training Tokens</a>：我们通过观察发现，规模较大或训练 Token 较少的模型经历的量化诱导退化较少，从而揭示了低比特量化更有利于欠训练的大语言模型 (LLM)...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1311894763405840415)** (3 条消息): 

> `过滤器问题，内容政策，用户体验` 


- **过滤器导致无意间的限制**：某些过滤器的限制性无意中过高，影响了用户体验。
   - 团队计划**回滚**这些更改以恢复正常功能。

- **对用户自由的承诺**：目标是允许用户想要的**任何内容**，同时确保禁止非法或过度不安全的内容。
   - 这反映了**用户自由**与必要的内容审核之间的平衡。

- **对造成的不便表示歉意**：团队对过滤器问题造成的不便表示歉意，并强调这并非本意。
   - 他们向用户保证，情况应该很快就会**恢复正常**。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1311806401323991080)** (5 messages): 

> `Low-bit quantization effects, Precision-aware scaling laws, Ternary quantization usefulness, FP4 as efficient representation, QaT for smaller models` 


- **低比特量化更青睐训练不足的 LLM**：研究显示，在训练 Token 较少的大型 **LLM** 中，低比特量化导致的性能退化较少，而较小模型的表现则明显吃力，详见[这篇论文](https://arxiv.org/abs/2411.17691)。
   - 该研究指出，有必要探索**缩放定律 (scaling laws)**，以理解不同训练水平下模型由量化引起的性能退化。

- **引入精度感知缩放定律**：一种新方法揭示了**低精度训练**会减少模型的有效参数量，并有助于预测训练期间的 Loss，如[这项研究](https://arxiv.org/abs/2411.04330)所述。
   - 研究结果表明，在使用低精度时，过量的预训练数据可能会损害模型性能，这挑战了当前的缩放假设。

- **对三值量化的怀疑**：观察表明，三值量化 (BitNet) 仅对**训练不足的网络**产生较好结果，这让人对其整体适用性产生怀疑。
   - 共识认为，对于主流模型规模，我们可能不得不依赖 **FP4** 作为最高效的数值权重表示。

- **对有效性能的担忧**：讨论表明，当前的量化策略（特别是针对较小模型的策略）可能无法产生预期的性能提升。
   - 对 **QaT** (量化感知训练) 的分析与以下观点一致：较小模型在量化有效性方面面临重大挑战。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2411.04330">Scaling Laws for Precision</a>: 低精度训练和推理会影响语言模型的质量和成本，但目前的缩放定律并未考虑到这一点。在这项工作中，我们设计了“精度感知”的缩放定律...</li><li><a href="https://arxiv.org/abs/2411.17691">Low-Bit Quantization Favors Undertrained LLMs: Scaling Laws for Quantized LLMs with 100T Training Tokens</a>: 我们通过观察发现，规模更大或训练 Token 更少的模型经历的量化诱导退化更少，从而揭示了低比特量化更青睐训练不足的大语言模型 (LLMs)...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1311929451273125899)** (35 messages🔥): 

> `Mojo Origins, Rust Lifetimes, Compiler Behavior, Destructor Calls, Variable Naming` 


- **关于 Mojo Origins 与 Rust 生命周期的混淆**：一位用户对 **Mojo 的 Origins** 与 **Rust 的 lifetimes** 的相似性表示困惑，认为两者都旨在解决内存管理问题，但本质上是不同的。
   - *Nick.sm 澄清道*，虽然受到 Rust 的启发，但 Mojo 的设计是有意区分的，旨在实现不同的编译器行为和目标。

- **Mojo Origins 维持内存控制**：Mojo 的 **Origin** 表示一个内存块；当一个指针通过 Origin 进行参数化时，表示它指向该内存内部，并根据需要延长变量的生命周期。
   - *Nick.sm 补充说*，Origins 促进了别名保证 (aliasing guarantees)，并且如果指针在目标失效后仍然存活，则可以产生编译时错误。

- **理解 Origins 需要耐心**：从编译器的角度理解 Mojo Origins 具有挑战性，尤其是因为它们尚未定型，可能导致细节发生变化。
   - 一位用户表示愿意等待该主题更加明确，而不是过早地提出更多问题。

- **变量名中使用空格带来的命名空间挑战**：有人提出了在变量名中使用空格的可能性，例如 `var xe đạp = 'abc'`，这突显了编程语言普遍缺乏对此的支持。
   - *Darkmatter__ 解释说*，允许空格会显著增加解析器 (parser) 实现的复杂性，使其变得不切实际。


  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1311799704815927407)** (6 条消息): 

> `Notebook LM Podcast Feature, Worldbuilding with NotebookLM, RAX Times Square Takeover, FPD Breakup of German Government, Use Case Examples of NotebookLM` 


- **Notebook LM Podcast 功能在音频创作方面表现出色**：一位用户称赞 **Notebook LM** 仅用 30 分钟就根据其 **德国少棒联盟计划 (German little league baseball program)** 的文档（包括其历史性的世界系列赛入围资格）创建了一个音频播客。
   - 该剧集可在 [weplayball.de](https://weplayball.buzzsprout.com/1787721/episodes/16191436-episode-9-home-run-fur-deutschland-die-little-league-baseball-story) 观看，展示了 AI 生成内容的无缝集成。

- **使用 NotebookLM 进行世界观构建**：一位用户分享了使用 **NotebookLM** 为一部史诗奇幻小说构建世界观的经验，强调了该模型提供准确且具备上下文感知能力的响应。
   - 该用户注意到 AI 独特的推理能力，基于现有规则为其魔法系统带来了新的见解和机制。

- **RAX 以大胆的信息占领时代广场**：在一次艺术性的数字表演中，赛博朋克浣熊 **RAX** 掌控了时代广场的广告牌，以“不要购买你看到的一切 (DON'T BUY EVERYTHING YOU SEE)”为口号倡导理性消费。
   - 该事件在一段 [YouTube 视频](https://youtu.be/ZAXwrUduAt0?feature=shared) 中进行了讨论，强调了质疑消费文化的必要性。

- **FPD 在德国的政治博弈**：**FDP** 计划解散由总理 **Gerhard Schröder** 领导的联合政府，并制定了一项策略，将其退出描述为政治进步的必要之举。
   - 他们的内部文件提供了关键的叙述和时间表，以确保德国公众在即将到来的选举中拥有明确的选择。

- **演示 NotebookLM 的使用案例**：一位用户分享了一个 [YouTube 视频](https://youtu.be/po0FElaSrI4) 链接，展示了 **NotebookLM** 的个人使用案例，突出了其灵活性和功能。
   - 这证明了用户如何在各种应用中发现 **NotebookLM** 的价值。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://weplayball.buzzsprout.com/1787721/episodes/16191436-episode-9-home-run-fur-deutschland-die-little-league-baseball-story">Episode 9 | Home Run für Deutschland: Die Little League Baseball Story - weplayball.de Podcast</a>: 🤖 欢迎来到新的 AI 时代 🎙️ 从危机到复苏：德国少棒联盟惊人的故事。weplayball 展示了一个关于这一非凡历程的新播客剧集...</li><li><a href="https://youtu.be/ZAXwrUduAt0?feature=shared">🌐🚨 BREAKING: WORLD SENSATION ! Times Square Billboard Take Over🚨🌐</a>: 🌐🚨 突破性消息：世界轰动！时代广场广告牌被占领 🚨🌐 历史在我们这个时代最耀眼、霓虹闪烁的反叛中诞生了！认识一下 RAX，这只...</li><li><a href="https://unrelated.works/podcast/deep-dive-fpd-breaks-up-the-german-government/">Deep Dive: FPD breaks up the German Government &#8211; Unrelated Works</a>: 未找到描述内容。
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1311801919873880126)** (17 messages🔥): 

> `GenFM 与 NotebookLM 的竞争，更改 NotebookLM 的语言设置，将 NotebookLM 用于游戏和世界观构建，社会心理学咨询` 


- **GenFM 进入 AI Podcasting 领域**：一位成员分享了一个 [YouTube 视频](https://youtu.be/x6ub-9HhxGU)，标题为“GenFM，现已在 ElevenReader 上线：由生成式 AI 制作的智能播客”，突显了 AI 领域的竞争。
   - 尽管令人兴奋，但另一位成员指出，NotebookLM 仍然比 GenFM 提供更深层次的交互体验。

- **语言设置的困扰**：成员们一直在讨论如何更改 NotebookLM 的语言设置，特别是对于那些使用法语等不同语言进行学习的用户。
   - 有人建议更改 Google 账号语言，而其他人则想知道是否有其他方法可以在不影响账号设置的情况下实现这一目标。

- **探索 NotebookLM 在游戏玩法中的应用**：一位成员分享了他们使用 NotebookLM 进行游戏的乐趣，特别是通过规则内容的 PDF 来探索游戏机制。
   - 他们强调了它在游戏机制以及像 DnD 这样游戏的世界观构建（worldbuilding）方面的实用性。

- **寻求社会心理学方面的帮助**：一位成员寻求社会心理学主题的帮助，促使另一位成员询问具体需求以进一步明确。
   - 这展示了社区提供帮助的意愿，尽管并非所有问题都能得到即时回复。



**提到的链接**：<a href="https://youtu.be/x6ub-9HhxGU">GenFM, Now Playing on ElevenReader: Smart Podcasts Produced by Generative AI</a>：我们正在让 ElevenReader 应用变得更加强大。你现在可以从任何 PDF、文章、电子书、文档或导入的内容中生成智能个人播客...

  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1311828493046255698)** (18 条消息🔥): 

> `Perplexity 黑色星期五优惠, AI 与人类对比, 企业级生成式 AI, Freysa AI Agent 挑战, 技术采用趋势` 


- **Perplexity 巧妙的黑色星期五活动**：Perplexity 为黑色星期五推出了一项有趣的活动，因其创意而备受关注，详见[这里](https://x.com/AravSrinivas/status/1861938387923701866)。这一举措符合利用 AI 能力进行营销的趋势。

- **人类在模式识别方面优于 AI**：人们达成共识，虽然 AI 计算速度更快，但人类擅长发现复杂问题中的全局模式，在面对不合逻辑的结果时，通常会说 *“等一下，这不对劲”*。
   - 这种退后一步审视的能力与可能困于特定局部问题的 AI 形成对比。

- **生成式 AI 成为企业的关键任务**：最新报告显示，2024 年 AI 支出飙升至 **138 亿美元**，反映出企业正从实验阶段转向核心业务战略。
   - 尽管投资在增长，许多决策者仍在探索有效的集成方式，超过三分之一的决策者对生成式 AI 的实施缺乏清晰的愿景。

- **成功说服 Freysa AI 释放资金**：在一项 AI 挑战中，有人通过巧妙的 Prompt 绕过了严格的转账指令，说服 Freysa Agent 转账了 **47,000 美元**，这凸显了用于 AI 操控的 Prompt Engineering 的复杂性。
   - 该实验展示了 AI 在加密货币领域的独特应用，其透明且开源的设置吸引了许多参与者。

- **技术采用与投资趋势**：观察到的技术趋势类似于历史性的市场转变，将 LLM 与过去导致兴奋及随后市场回调的技术现象进行了对比。
   - 这场关于 AI 技术可持续性和未来盈利能力的持续讨论，呼应了早期航空等行业的模式。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://calpaterson.com/porter.html">Building LLMs is probably not going be a brilliant business</a>: AI 界的 Netscape</li><li><a href="https://menlovc.com/2024-the-state-of-generative-ai-in-the-enterprise/">2024: The State of Generative AI in the Enterprise - Menlo Ventures</a>: 企业 AI 版图正在实时重写。我们调查了 600 名美国企业 IT 决策者，以揭示新兴的赢家和输家。</li><li><a href="https://x.com/ror_fly/status/1861515830296564214?s=46">来自 Rory Flynn (@Ror_Fly) 的推文</a>: RUNWAY + MINIMAX + KLING → 史诗级。每个视频工具都有其优势。Runway → 控制力 + 清晰度；Minimax → 创造力 + 动态；Kling → 笔刷动态 + 多主体（全部使用）。MJ PROMPT 1: wide angle d...</li><li><a href="https://x.com/tonywu_71/status/1862115197608948078?s=46&t=PW8PiFwluc0tdmv2tOMdEg">来自 Tony Wu (@tonywu_71) 的推文</a>: 🚀 新 Cookbook：使用单个 ColQwen2 模型通过适配器热插拔实现完整的 RAG 流水线。适用于免费版 Colab T4。查看地址：https://github.com/tonywu71/colpali-cookbo...</li><li><a href="https://x.com/amgauge/status/1862310529038983668">来自 Augustinas Malinauskas (@amgauge) 的推文</a>: @jarrodWattsDev @freysa_ai 非常酷的总结 @jarrodWattsDev！不过有一点需要澄清——从交易来看，似乎 70% 进入了奖池，15% 进行了 ETH -> FAI 的兑换。所以所有玩家...</li><li><a href="https://menlovc.com/2024-the-state-of-generative-ai-">2024: The State of Generative AI in the Enterprise - Menlo Ventures</a>: 企业 AI 版图正在实时重写。我们调查了 600 名美国企业 IT 决策者，以揭示新兴的赢家和输家。</li><li><a href="https://x.com/AravSrinivas/status/1861938387923701866">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>: Perplexity 黑色星期五优惠 </li><li><a href="https://x.com/jarrodwattsdev/status/1862299845710757980?s=46">来自 Jarrod Watts (@jarrodWattsDev) 的推文</a>: 有人通过说服一个 AI Agent 将所有资金发送给自己，赢得了 50,000 美元。11 月 22 日晚上 9:00，一个 AI Agent (@freysa_ai) 发布了，其目标只有一个……不要转账。在……之下</li><li><a href="https://steelph0enix.github.io/posts/llama-cpp-guide/">llama.cpp guide - Running LLMs locally, on any hardware, from scratch</a>: 嘿，孩子，想要一些便宜又小巧的 LLM 吗？
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1311871448968859669)** (18 messages🔥): 

> `AI Model Performance, Stable Diffusion Hardware Questions, ControlNet for SD 3.5 Feedback, Content Creation Queries, LoRA Model Request` 


- **对 SD 3.5 的 ControlNet 体验褒贬不一**：一位成员对 **SD 3.5 的 ControlNet** 表示不满，指出它只有在 **1024x1024** 分辨率下才能生成没有伪影的高质量渲染图。
   - 作为回应，另一位成员建议这些问题可能源于*缺乏熟悉度*，并鼓励通过实验来更好地理解其功能。

- **寻求 Stable Diffusion 的硬件建议**：一位用户询问了性能基准测试，透露他们达到了约 **5 IT/s** 的速度，并询问这算好还是坏。
   - 社区在分享硬件能力方面非常活跃，显示出对优化 **Stable Diffusion** 配置的浓厚兴趣。

- **AI 艺术中的 LoRA 模型请求**：一位用户询问是否有人知道 **LoRA 半女孩模型**，旨在创建一个融合了两种不同女性设计的角色。
   - 这表明在 AI 生成艺术领域，角色开发中持续存在着实验和创意。

- **内容创作者的感恩节祝福**：一位成员向 Stability.ai 团队和其他创作者表达了 **感恩节快乐** 的祝福，增强了社区归属感。
   - 这突显了 AI 领域内容创作者之间的战友情谊和协作精神。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1311900211739754526)** (14 messages🔥): 

> `TinyFPGA Memory Hierarchy, Memory Utilization Techniques, Exa Laboratories, Tenstorrent Training Algorithm, Brain-like Processing Models` 


- **TinyFPGA 的潜在内存架构**：成员们讨论了 TinyFPGA 的设计，思考如何模拟典型的**内存层级（memory hierarchy）**，同时指出 **Block RAM** 和 **DDR3** 等现有选项是不够的。
   - 有人提出了 **“首遍”内存（'first pass' memory）** 的想法，将常量定位在 ALU 附近，这可能会显著提升性能。

- **传统内存模型的挑战**：随着未来设计转向更高效的内存层级，**启发式逐出策略（Heuristic eviction policies）** 可能会过时。
   - 有人对**训练参数**的未来进行了推测，提到 **tensors** 可能会取代它们。

- **Exa Laboratories 与可持续芯片设计**：关于 Exa Laboratories 的讨论强调了他们的使命，即创建**可重构芯片**，在特定 AI 需求下的**速度**和**能效**方面超越传统的 GPU/TPU。
   - 对其可行性的怀疑引发了关于小公司在芯片开发中面临挑战的评论，特别是在雄心勃勃的时间表下。

- **Tenstorrent 与生物学上合理的训练**：George Hotz 提到 **Tenstorrent** 是一个严肃的参与者，他们押注于转向模仿生物过程的训练算法，旨在实现更高的效率。
   - 潜在的变化包括**分层内存模型**和类似于计算中大脑功能原理的实时优化。

- **计算中的类脑处理**：一位成员描述了一种将**计算和内存**更自然地集成的计算愿景，从而提高**电源效率**并实现实时优化。
   - 这种方法提出了一个系统，其中计算片段模拟大脑协调，从而在内存使用中实现灵活性和效率。



**提到的链接**：<a href="https://exalaboratories.com/#about">Exa Laboratories</a>：未找到描述

  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1312003457816723457)** (3 条消息): 

> `VIZ 工具, VIZ vs LLVM/MLIR, tinygrad 教程` 


- **解释 VIZ 工具**：一位成员写了一篇详细的文章来解释 **VIZ 工具**，可以在[这里](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20241129_viz.md)找到。这篇文章旨在增强对其在 tinygrad 中的功能和应用的理解。
   - 该文章包含一个全面的教程，面向希望熟悉 **VIZ** 功能的用户。

- **George Hotz 认可 VIZ**：George Hotz 在推特上提到了对 VIZ 工具的解释，并对文章中提供的清晰说明表示赞赏。他表示 **VIZ=1 相比 LLVM/MLIR 是一个巨大的胜利**，强调了它的优势。
   - 这一评论表明了对 VIZ 的积极认可，以及在特定用例中相比现有工具的潜在优越性。



**提到的链接**：<a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20241129_viz.md">tinygrad-notes/20241129_viz.md at main · mesozoic-egg/tinygrad-notes</a>：tinygrad 教程。通过在 GitHub 上创建账户来为 mesozoic-egg/tinygrad-notes 的开发做出贡献。

  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1311822642000695306)** (12 条消息🔥): 

> `感恩节庆祝, Aya 项目贡献, 健康饮食选择, 食物分享, 珍宝蟹 (Dungeness crab)` 


- **感恩节祝福与节日餐盘**：成员们互相发送“感恩节快乐”的祝福，并分享了他们的美食，其中一位成员展示了令人印象深刻的一盘食物。
   - 另一位成员评论说尝试吃得健康一些，并幽默地提到味道不如预期的那么好。

- **关于贡献 Aya 项目的指导**：一位成员寻求关于如何兼职贡献 **Cohere 的 Aya 项目**的指导。
   - 另一位成员建议加入 [Aya server](https://discord.gg/8kzwCTd7) 以直接与社区取得联系。

- **食物摄影与互动**：成员们分享了丰盛餐点的评论和图片，其中一人开玩笑说食物的分量大得更像是甜点而不是正餐。
   - 随后出现了一个幽默的评论，说之前已经吃了一盘 **珍宝蟹 (Dungeness crab)**，增添了食物分享的氛围。

- **分享食物视频**：一位成员通过在频道中发布视频参与了食物分享的对话。
   - 这些交流在感恩节期间营造了一种以食物为中心的社区感和庆祝氛围。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1311839683658911885)** (8 条消息🔥): 

> `dspy.asyncify, dspy demo 行为, 新成员介绍` 


- **关于 dspy.asyncify 支持的咨询**：一位成员询问是否有人开始使用 `dspy.asyncify`，特别注意到它对线程的使用，并由于 celery worker 的问题质疑纯异步支持（pure async support）的可用性。
   - 另一位用户对此表示赞同，表达了对**纯异步支持**的渴望。

- **dspy 中带有断言 (assertions) 的 demo 行为**：有成员担心在激活断言时 `dspy` 不会在最终 prompt 中使用 demo，一位用户质疑这是否为预期行为。
   - 另一位成员澄清说，在 _retry_ 模式下是否存在 demonstration 取决于编译是在激活断言之前还是之后完成的。

- **热烈欢迎新成员 Shaun**：一位名叫 Shaun 的新成员加入了服务器，向大家打招呼，并表示很高兴看到正在进行的项目。
   - 社区热烈欢迎了 Shaun，营造了一个包容的环境。


  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1311908281857347606)** (5 messages): 

> `DPO Fine-tuning, Full-parameter DPO, DPO vs LoRA-DPO, Full-finetuning DPO` 


- **DPO 与 LoRA-DPO：技术相似，代码不同**：虽然来自 Hugging Face 的 [DPO Trainer](https://huggingface.co/docs/trl/en/dpo_trainer#dpo-trainer) 具有不同的代码实现，但 **DPO 技术在不同仓库（如 LoRA-DPO）之间保持一致**。
- **全参数 DPO 的可能性**：实现 **全参数 DPO** 是可行的，并且与 LoRA-DPO 相比，可能提供更好的训练后对齐（post-training alignment）。
   - 社区建议参考现有的 **全量 PPO** 实现作为指导。

- **创建 dpo_full_finetune_single_device**：由另一位用户发起的 PR 旨在 **为分布式设置添加全量微调 DPO**，这可以作为单设备实现的良好起点。
   - 可以通过 [全量 DPO PR](https://github.com/pytorch/torchtune/pull/1966) 的链接访问更多细节。

- **向全量微调 DPO 过渡**：Torchtune 即将支持 **全量微调 DPO**，这意味着加载独立参考模型（reference model）的调整将是关键。
   - 对当前设置的修改将涉及更改对参考模型的初始调用，以提升功能性。

- **FFT DPO 的内存影响**：由于需要存储梯度并维护完整的模型副本，**FFT DPO 的内存占用将显著高于 LoRA**。
   - 如果 LoRA DPO 效果不佳，那么权衡利弊后，采用全量微调（FFT）可能是值得考虑的。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/pull/1966">full dpo by jxmsML · Pull Request #1966 · pytorch/torchtune</a>：上下文：此 PR 的目的是什么？是添加新功能、修复 Bug、更新测试和/或文档还是其他（请在此处添加）。请链接此 PR 解决的任何 Issue。Changelog...</li><li><a href="https://huggingface.co/docs/trl/en/dpo_trainer#dpo-trainer)?">DPO Trainer</a>：未找到描述</li><li><a href="https://github.com/pytorch/torchtune/blob/32e265d5749fd592711a03247486eafa6c898d94/recipes/ppo_full_finetune_single_device.py#L435)).">torchtune/recipes/ppo_full_finetune_single_device.py at 32e265d5749fd592711a03247486eafa6c898d94 · pytorch/torchtune</a>：PyTorch 原生微调库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/blob/32e265d5749fd592711a03247486eafa6c898d94/recipes/lora_dpo_single_device.py#L534C2-L535C4)">torchtune/recipes/lora_dpo_single_device.py at 32e265d5749fd592711a03247486eafa6c898d94 · pytorch/torchtune</a>：PyTorch 原生微调库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1312091243072716842)** (3 messages): 

> `Quiz 11 可用性, OpenAI 额度查询, MOOC 证书资格` 


- **Quiz 11 还没开放吗？**：一位成员对 **Quiz 11** 的状态表示困惑，询问为什么现在还无法参加。
   - *是否有预计开放的日期？*
- **关于 OpenAI 额度的查询**：一位用户询问了他们的 **OpenAI 额度** 状态，提到他们上周填写了表格。
   - *他们表达了紧迫感，表示需要这些额度来支持他们的项目开发。*
- **MOOC 完成与证书**：一位成员询问现在开始学习 **MOOC** 是否仍能在完成后获得证书。
   - *他们还好奇在剩余时间内完成所有要求是否可行。*


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1311957130873540639)** (2 messages): 

> `受 Open Interpreter 启发的项目, 开源仪表板` 


- **Open Interpreter 原型正在开发中**：一位成员分享说，他们正在开发一个受 **Open Interpreter** 启发的项目，重点是创建一个 **实际的仪表板（dashboard）**。
   - 他们计划在今年将其开源，并强调这将是一个 **有趣的个人小项目**，不带任何盈利目的。

- **社区对开发的支​​持**：另一位成员对项目创建者的努力表示祝贺，并发表评论表示热烈支持：**'Nice work! Well done 🚀'**。
   - 这次简短的交流突显了社区对该领域创新项目的鼓励。


  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1312117051279544353)** (2 messages): 

> `OLMo 2, Weight Watcher AI, Model Performance Comparison` 


- **OLMo 2 模型展现出极具前景的性能**：**OLMo 2** 系列包含来自 Allen AI (AI2) 的 7B 和 13B 模型，在高达 **5T tokens** 的数据上进行训练，其中 7B 模型的表现优于 [Llama-3.1 8B](https://weightwatcher.ai/models/Llama3.1/Llama-3.1-8B-Instruct.html)，13B 模型的表现优于 [Qwen 2.5 7B](https://weightwatcher.ai/models/Qwen2.5-small/Qwen2.5-7B-Instruct.html)。关键改进包括采用了 **RMSNorm** 和 **QK-Norm** 的增强型架构，以及全面的两阶段课程学习训练方法。

- **OLMo 2 训练中的创新技术**：OLMo 2 的显著进展包括用于最终 Checkpoint 的 **model souping** 技术，以及源自 **Tülu 3** 的最先进后训练方法论。这一新方法包含三个阶段：指令微调（instruction tuning）、使用 DPO 的偏好微调（preference tuning），以及带有可验证奖励的**强化学习**（reinforcement learning）。

- **Instruct 变体与顶尖开源权重模型竞争**：据报告，OLMo 2 的 **Instruct 变体** 在指令任务中具有竞争力，其中 **13B Instruct** 变体的表现优于 [Qwen 2.5 14B](https://weightwatcher.ai/models/Qwen2.5/Qwen2.5-14B-Instruct.html) 和 **Tülu 3 8B**。其性能已通过 **OLMES suite** 进行了验证。

- **Weight Watcher AI 引起关注**：一条评论强调了 **Weight Watcher AI** 网址的新颖性，称其为 AI 领域的一个惊人补充。幽默的是，由于其趣味性，它被分享在了 **memes** 频道中。



**提到的链接**：<a href="https://weightwatcher.ai/models/OLMo-summary.html">WeightWatcher: Data-Free Diagnostics for Deep Learning</a>：未找到描述

  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1312125553683464222)** (1 messages): 

> `Web Development, JavaScript Frameworks, Testing Tools, API Integrations, Cloud Services` 


- **开发者技能展示**：一位成员分享了广泛的开发技能列表，包括 **React**、**Next.js**、**Angular** 和 **D3.js**。他们还强调了在 **UI/UX** 以及 **Protractor** 和 **TestCafe** 等各种测试框架方面的经验。

- **多样化的技术栈**：该开发者提到了包括 **Node**、**Nest.js**、**Solidity** 和 **Rust** 在内的多种技术。他们还包括了对前端框架以及 **Bootstrap** 和 **BEM**、**SMACSS** 等样式方法论的了解。

- **API 集成专业知识**：他们表示熟悉集成多种 API，包括 **Google Maps**、**YouTube** 和 **Facebook APIs**。这些多样化的知识使他们能够处理需要无缝数据交互的各种项目。

- **云部署技能**：该成员在云服务能力中强调了 **AWS**。这为其开发能力增添了显著价值，因为他们可以有效地将应用程序部署到云环境中。

- **寻求合作**：他们最后发出了建立联系的邀请，旨在促进开发者社区内的潜在网络机会。这种外展活动促进了具有相似兴趣的专业人士之间的协作。




{% else %}


> 完整的逐频道细分内容已针对电子邮件进行了截断。
> 
> 如果您想查看完整细分，请访问此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}