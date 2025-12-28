---
companies:
- openai
- softbank
- oracle
- arm
- microsoft
- nvidia
- huggingface
- deepseek-ai
date: '2025-01-22T01:56:21.007400Z'
description: '**星际之门项目 (Project Stargate)** 是一个由 **OpenAI** 和 **软银 (Softbank)** 领导，并得到
  **甲骨文 (Oracle)**、**安谋 (Arm)**、**微软 (Microsoft)** 和 **英伟达 (NVIDIA)** 支持的美国“AI 曼哈顿计划”。该项目的规模据称可与当年的曼哈顿计划相媲美，经通胀调整后的成本达
  **350 亿美元**。尽管微软作为独家计算合作伙伴的角色有所削弱，但该项目态度严肃，只是目前尚不具备即时实用性。


  与此同时，**Noam Shazeer** 披露了 **Gemini 2.0 Flash Thinking** 的第二次重大更新，实现了现已可用的 **100
  万 token 超长上下文**。此外，**AI Studio** 推出了全新的 **代码解释器 (code interpreter)** 功能。


  在 Reddit 上，基于 **Qwen 32B** 蒸馏而成的 **DeepSeek R1** 已在 **HuggingChat** 上免费发布，引发了关于私有化部署、性能问题和量化技术的讨论。DeepSeek
  首席执行官 **梁文锋** 强调，尽管面临出口限制，他们仍专注于 **通用人工智能 (AGI) 基础研究**、高效的 **MLA 架构**，并致力于 **开源开发**。这使
  DeepSeek 成为闭源 AI 趋势的一个潜在替代方案。'
id: 0c6622b1-3e7e-4ecf-83e0-d6c7211c7d29
models:
- gemini-2.0-flash
- deepseek-r1
- qwen-32b
original_slug: ainews-project-stargate-500b-datacenter-17-of-us
people:
- noam-shazeer
- liang-wenfeng
title: 星际之门项目（Project Stargate）：耗资 5000 亿美元的数据中心（占美国 GDP 的 1.7%）以及 Gemini 2 Flash
  Thinking 2。
topics:
- long-context
- quantization
- code-interpretation
- model-distillation
- open-source
- agi-research
- model-performance
- memory-optimization
---

<!-- buttondown-editor-mode: plaintext -->**孙正义 (Masa Son) 与 Noam Shazeer 就是你所需要的一切。**

> 2025/1/20-2025/1/21 的 AI 新闻。我们为你检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **34** 个 Discord 社区（**225** 个频道，**4353** 条消息）。预计节省阅读时间（以 200wpm 计算）：**450 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 来进行 AINews 讨论！

像这样的日子总是让人纠结——一方面，显而易见的重磅新闻是 [Project Stargate](https://x.com/openai/status/1881830103858172059?s=46) 的宣布，这是一个由 OpenAI 和 Softbank 领导，并得到 Softbank、OpenAI、Oracle、MGX、Arm、Microsoft 和 NVIDIA 支持的美国“AI 曼哈顿计划”。作为规模参考，实际的曼哈顿计划[经通胀调整后的成本为 350 亿美元](https://x.com/tanayj/status/1881849682063986843?s=46)。


![image.png](https://assets.buttondown.email/images/e686ff0d-b54a-44c9-b567-0e3c7f927c6d.png?w=960&fit=max)


尽管这在[一年前就有传闻](https://www.theinformation.com/articles/microsoft-and-openai-plot-100-billion-stargate-ai-supercomputer?rc=ytp67n)，但 Microsoft 作为 OpenAI 独家算力合作伙伴角色的[削弱](https://x.com/smokeawayyy/status/1881801442459033662?s=46)因其缺席而显得尤为突出。与任何引人注目的公关噱头一样，人们应该警惕 [AI-washing](https://x.com/teortaxesTex/status/1881839728250765709)，但该项目非常严肃，应予以重视。

然而，这并不是你今天就能用上的新闻，而这正是我们这份本地 AI 报纸的目标。

幸运的是，Noam Shazeer 为你带来了[第二个 Gemini 2.0 Flash Thinking](https://x.com/NoamShazeer/status/1881845901872001293)，它在 2.0 Flash 上又有了巨大飞跃，并且拥有你今天就可以使用的 1M 长上下文（我们明天将在 AINews 和 Smol Talk 中启用）：


![image.png](https://assets.buttondown.email/images/b2247b7e-56ca-48db-ac9c-7b58c7dee477.png?w=960&fit=max)


AI Studio 也获得了一个 [code interpreter](https://x.com/jack_w_rae/status/1881850281052545140)。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

> 所有回顾均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

待完成

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1：DeepSeek R1：发布、性能与战略愿景**

- **[DeepSeek R1 (Qwen 32B Distill) 现已在 HuggingChat 上免费可用！](https://hf.co/chat/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)** ([得分: 364, 评论: 106](https://reddit.com/r/LocalLLaMA/comments/1i6jbur/deepseek_r1_qwen_32b_distill_is_now_available_for/)): **DeepSeek R1**，一个 **Qwen 32B** 的蒸馏版本，现在可以在 **HuggingChat** 上免费访问。
  - **托管与访问关注**：用户讨论了自行托管 **DeepSeek R1** 以避免登录 **HuggingChat** 的选项，一些人对需要账号才能评估模型表示沮丧。有人建议使用**虚拟邮箱**创建账号来绕过这一限制。
  - **性能与技术问题**：有报告称存在模型无响应等性能问题，并讨论了使用 **quantization**（例如 FP8、8-bit）和 system prompts 对模型性能的影响。一些用户注意到 **DeepSeek R1** 在规划方面优于代码生成，其他人分享了像 [cot_proxy](https://github.com/bold84/cot_proxy) 这样的工具来管理模型的“思考”标签。
  - **模型比较与偏好**：用户将 **DeepSeek R1** 与 **Phi-4** 和 **Llama 70B** 等其他模型进行了比较，一些用户在数学和细微理解等特定任务上更倾向于蒸馏模型。人们有兴趣探索 **Qwen 14B** 等其他变体，并期待 **R1 Lite** 以获得更好的连贯性。

- **深入探究 DeepSeek 的宏伟使命（CEO 梁文锋访谈）** ([Score: 124, Comments: 27](https://reddit.com/r/LocalLLaMA/comments/1i6dlvj/inside_deepseeks_bold_mission_ceo_liang_wenfeng/))：由 CEO **梁文锋**领导的 DeepSeek 脱颖而出，其核心在于专注于**基础 AGI 研究**而非快速商业化，旨在将中国在全人工智能领域的角色从“搭便车者”转变为“贡献者”。他们的 **MLA architecture** 大幅降低了内存占用和成本，推理成本显著低于 **Llama3** 和 **GPT-4 Turbo**，体现了他们对高效创新的承诺。尽管面临美国芯片出口限制等挑战，DeepSeek 仍致力于**开源开发**，利用自下而上的组织结构和本土年轻人才，这可能使他们成为 AI 闭源趋势中的有力替代方案。
  - **DeepSeek 对 AGI 的关注**：评论者强调，DeepSeek 对 AGI 而非利润的承诺值得关注，有人将其方法比作 OpenAI 的早期阶段。对于 DeepSeek 是否能长期保持这种开源精神，还是最终会像其他科技巨头一样转向闭源模型，存在一些怀疑。
  - **领导力与认可**：**梁文锋**的领导力受到关注，文中特别提到了他与中国总理**李强**的会面，这表明了高层的认可与支持。这次会面凸显了 DeepSeek 日益增长的影响力以及对中国 AI 发展的潜在影响。
  - **年轻人才与创新**：评论者赞扬了 DeepSeek 团队的创造力和创新精神，指出团队由年轻的应届博士组成，尽管在加入公司前并不出名，但已取得了重大成就。这凸显了利用年轻人才实现 AI 突破性进展的潜力。


- **[DeepSeek-R1-Distill-Qwen-1.5B 在浏览器中通过 WebGPU 100% 本地运行。据报道，在数学基准测试中表现优于 GPT-4o 和 Claude-3.5-Sonnet（AIME 为 28.9%，MATH 为 83.9%）。](https://v.redd.it/5ei4j3c9teee1)** ([Score: 72, Comments: 17](https://reddit.com/r/LocalLLaMA/comments/1i6t08q/deepseekr1distillqwen15b_running_100_locally/))：**DeepSeek-R1-Distill-Qwen-1.5B** 完全使用 **WebGPU** 在浏览器内运行，据报道在数学基准测试中超过了 **GPT-4o** 和 **Claude-3.5-Sonnet**，在 AIME 上达到 **28.9%**，在 MATH 上达到 **83.9%**。
  - **ONNX** 作为 LLM 的文件格式被讨论，一些用户指出它提供了性能优化，在特定硬件上可能比 **safetensors** 和 **GGUF** 等其他格式快 **2.9 倍**。然而，普遍共识是，这些只是被不同硬件/软件设置所青睐的不同数据格式。
  - **DeepSeek-R1-Distill-Qwen-1.5B** 因在 **WebGPU** 上完全在浏览器内运行并在基准测试中优于 **GPT-4o** 而受到关注，在线演示和源代码可在 [Hugging Face](https://huggingface.co/spaces/webml-community/deepseek-r1-webgpu) 和 [GitHub](https://github.com/huggingface/transformers.js-examples/tree/main/deepseek-r1-webgpu) 上获得。然而，一些用户认为尽管其基准测试结果令人印象深刻，但在实际应用中仍不及 **GPT-4o**。


**主题 2. 新的 DeepSeek R1 工具增强了易用性和速度**

- **[以 3-10 倍速度在 Huggingface 上部署任何 LLM](https://i.redd.it/8dsnudtrhdee1.png)** ([Score: 109, Comments: 13](https://reddit.com/r/LocalLLaMA/comments/1i6mjxv/deploy_any_llm_on_huggingface_at_310x_speed/))：该图片展示了 Huggingface 上“专用部署”的**数字化仪表盘**，显示了两个模型部署卡。**"deepseek-ai/DeepSeek-R1-Distill-Llama-70B"** 模型正在使用四块 **NVIDIA H100 GPU** 进行 52% 的量化，而 **"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"** 模型在一块 **NVIDIA H100 GPU** 上运行，两者最近都处于活跃状态并准备好接收请求。
  - **avianio** 推出了一项部署服务，声称比 **HF Inference/VLLM** 速度提高 **3-10 倍**，设置时间约为 5 分钟，利用 **H100** 和 **H200 GPU**。该服务支持约 **100 种模型架构**，未来计划支持多模态，并提供经济高效、无日志的私有部署，在高流量场景下价格为 **每百万 tokens 0.01 美元**。
  - **siegevjorn** 和 **killver** 对 **3-10 倍速度的说法**提出质疑，要求澄清比较指标和硬件一致性。**killver** 特别询问该说法在相同硬件上是否有效。
  - **omomox** 估计部署 **4x H100** 的成本约为 **20 美元/小时**，强调了用户潜在的成本考虑。

- **在 open webui 中获得更好的 R1 体验** ([Score: 117, Comments: 41](https://reddit.com/r/LocalLLaMA/comments/1i6b65q/better_r1_experience_in_open_webui/)): 该帖子介绍了一个简单的 **open webui function**，用于 **R1 模型**，通过将 `<think>` 标签替换为 `<details>` 和 `<summary>` 标签，使 R1 的思考过程可以折叠，从而提升用户体验。此外，它还按照 **DeepSeek** 的 **API documentation** 的建议，在多轮对话中移除旧的思考内容。该功能旨在用于本地 R1 (-distilled) 模型，不兼容 DeepSeek API。更多详情可以在 [GitHub](https://github.com/AaronFeng753/Better-R1) 找到。
  - **OpenUI vs. LMstudio**: 用户对 **OpenUI** 和 **LMstudio** 进行了对比，表达了希望 OpenUI 能像 LMstudio 一样响应迅速的愿望。然而，作者强调 **webui** 提供了更大的灵活性，允许用户自由修改输入和输出。
  - **DeepSeek API 支持**: 一些用户请求在 open webui function 中增加对 **DeepSeek API** 的支持，表明了对本地使用之外更广泛兼容性的兴趣。
  - **VRAM 限制与解决方案**: 用户讨论了在 8GB 等有限 VRAM 下使用模型的挑战，并分享了 **Hugging Face** 上的 **DeepSeek-R1-Distill-Qwen-7B-GGUF** 等资源，以潜在地解决这些限制。


**主题 3. DeepSeek R1 与竞争对手的效率和性能对比**

- **我计算了 R1 与 o1 的实际成本，以下是我的发现** ([Score: 58, Comments: 17](https://reddit.com/r/LocalLLaMA/comments/1i6axmv/i_calculated_the_effective_cost_of_r1_vs_o1_and/)): 该帖子通过对比 Token 生成和定价，分析了 R1 与 o1 模型的**成本效益**。**R1** 生成的 reasoning tokens 是 **o1** 的 **6.22 倍**，而 **o1** 每百万 output tokens 的价格是 **R1** 的 **27.4 倍**。因此，考虑到 Token 效率，**R1** 实际上比 **o1** 便宜 **4.41 倍**，尽管由于对 token-to-character 转换的假设，实际成本可能会略有波动。
  - 包括 **UAAgency** 和 **inkberk** 在内的几位评论者批评了成本对比中使用的方法论，认为分析可能存在偏差，或者基于无法准确反映实际使用情况的假设。**Dyoakom** 和 **pigeon57434** 强调了 OpenAI 可能缺乏透明度，质疑该公司提供的示例是否具有代表性。
  - **dubesor86** 提供了详细的测试结果，指出 **R1** 生成的 reasoning tokens 并没有达到 **o1** 的 **6.22 倍**。在他们的测试中，**R1** 产生的 thought tokens 多出约 **44%**，根据 API 使用数据，**R1** 的实际成本比 **o1** 便宜 **21.7 倍**，这与原帖的结论形成了对比。
  - **BoJackHorseMan53** 建议不要仅仅依赖假设，并建议通过 API 运行实际查询来确定真实的成本差异，强调了通过实际测试验证假设的重要性。


- **[DeepSeek-R1 PlanBench 基准测试结果](https://i.redd.it/qa5yh1w3odee1.jpeg)** ([Score: 56, Comments: 2](https://reddit.com/r/LocalLLaMA/comments/1i6n87h/deepseekr1_planbench_benchmark_results/)): 截至 **2025 年 1 月 20 日**的 **PlanBench benchmark 结果**对比了 **Claude-3.5 Sonnet, GPT-4, LLaMA-3.1 405B, Gemini 1.5 Pro** 和 **Deepseek R1** 等多种模型在 "Blocksworld" 和 "Mystery Blocksworld" 领域的表现。关键指标包括 "Zero shot" 得分、性能百分比以及每 100 个实例的平均 API 成本，其中 **Claude-3.5 Sonnet** 在 600 个问题中答对 329 个，达到了 **54.8% 的成功率**。
  - **PlanBench** 是一个旨在评估大语言模型在规划和推理任务上表现的 benchmark，详细论文可在 [arXiv](https://arxiv.org/abs/2206.10498) 查阅。
  - 结果来源可以通过[此链接](https://x.com/karthikv792/status/1881731017746313367)或[备用链接](https://xcancel.com/karthikv792/status/1881731017746313367)访问。


**主题 4. 对 LLM 中“陷阱测试”的批评及竞争背景**

- **[简直无法使用](https://i.redd.it/iatgsah1ubee1.png)** ([Score: 95, Comments: 102](https://reddit.com/r/LocalLLaMA/comments/1i6fxxy/literally_unusable/)): **对 LLM “陷阱”测试的批评** 重点展示了一个语言模型在计算 "strawberry" 中字母 'r' 出现次数时的结构化响应。该模型的分析和指令式方法包括写出单词、识别并计数 'r'，并强调存在 **2 个小写 'r'**。
  - **模型差异与性能**：评论者讨论了不同的模型架构和预训练数据如何导致性能差异，较小模型的表现通常与 **R1** 等大型模型的结果背道而驰。**Custodiam99** 提到即使是 **70b 模型** 在实际中也可能无法使用，而 **Upstairs_Tie_7855** 等其他人则报告同一模型效果出色。
  - **量化与设置的影响**：几位用户强调了使用正确的量化设置和系统提示词（system prompts）以获得准确结果的重要性。**Youcef0w0** 指出，当缓存类型低于 **Q8** 时模型会崩溃，而 **TacticalRock** 则强调应根据文档使用正确的量化和温度（temperature）设置。
  - **实际应用与局限性**：讨论揭示了模型并非 AGI，而是需要正确使用才能有效解决问题的工具。**ServeAlone7622** 建议了一套使用推理模型的详细流程，而 **MixtureOfAmateurs** 和 **LillyPlayer** 则展示了模型在特定提示词下的挣扎以及在某些任务上的过拟合现象。


## 其他 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. OpenAI 5000 亿美元投资：与 Oracle 和 Softbank 合作**

- **[特朗普将宣布由 OpenAI 领导的合资企业获得 5000 亿美元投资](https://www.reuters.com/technology/artificial-intelligence/trump-announce-private-sector-ai-infrastructure-investment-cbs-reports-2025-01-21/)** ([Score: 595, Comments: 181](https://reddit.com/r/OpenAI/comments/1i6rwc0/trump_to_announce_500_billion_investment_in/)): **唐纳德·特朗普** 计划宣布对一个由 **OpenAI** 领导的项目进行 **5000 亿美元的投资**。目前尚未提供该合资企业及其目标的具体细节。
  - **对投资来源的误解**：许多评论者澄清说，这 **5000 亿美元投资** 来自私营部门，而非美国政府。这项投资涉及 **OpenAI、SoftBank 和 Oracle**，共同组建名为 **Stargate** 的合资企业，初始承诺投资 **1000 亿美元**，四年内可能增长至 5000 亿美元。
  - **对基础设施和选址的担忧**：评论者对美国电网处理 AI 基础设施需求的能力表示担忧，建议未来依赖 **核反应堆**。由于德克萨斯州电网孤立且不可靠，该项目选择 **德克萨斯州** 受到了质疑。
  - **怀疑态度与政治担忧**：有人怀疑投资是否会兑现，并批评其政治影响，一些人认为这符合 **法西斯主义** 倾向。这一宣布被比作之前的投机项目，如“基础设施周”和 **威斯康星工厂**。


- **[Sam Altman 在整个 AI 基础设施协议宣布期间的表情](https://www.reddit.com/gallery/1i6w8ln)** ([Score: 163, Comments: 51](https://reddit.com/r/OpenAI/comments/1i6w8ln/sam_altmans_expression_during_the_entire_ai_infra/)): 该帖子缺乏关于 **Sam Altman** 在 AI 基础设施协议宣布期间表情的具体细节或背景，未提供进一步的信息或见解。
  - 围绕 **Sam Altman 的举止** 的讨论突显了人们对其焦虑和压力的感知，评论建议他经常看起来就是这样。用户将他的表情比作“福奇脸（Fauci face）”或“黛博拉·伯克斯（Debra Birx）”，并推测他在职位上所面临的压力。
  - 几条评论幽默地提到了 **Elon Musk** 以及像 **普京** 这样的地缘政治人物，暗示 Altman 可能因内部和外部政治动态而承受巨大压力。人们将其与寡头管理和“坠楼政治（defenestration politics）”进行了比较。
  - 对话中包含了对 Altman 表情的轻松和讽刺性评论，用户开玩笑地将其归因于像个“等着见金主的男宠”，或者担心 Musk 的反应，这表明社区对 Altman 的看法混合了幽默与批判。


**主题 2. OpenAI 的新模型 Operator**

- **[Exa CEO 拥有关于 OpenAI 新模型的内幕消息](https://www.reddit.com/gallery/1i6dpet)** ([Score: 215, Comments: 105](https://reddit.com/r/OpenAI/comments/1i6dpet/ceo_of_exa_with_inside_information_about_open_ai/)): **Exa 的 CEO** 声称拥有关于 **OpenAI 新模型**能力的内幕消息，特别是质疑这些模型作为 operator 的潜在有效性。该帖子未提供更多细节或背景。
  - 讨论重点是对 **AGI 炒作**和 **OpenAI 新模型**的怀疑，几位用户质疑这些说法的现实性，并将其与之前过度炒作的技术（如 **3D printers**）进行类比。用户对 **o3** 等模型在现实世界中的表现（与其 benchmark 结果相比）表示怀疑，强调了炒作与实际应用之间的差距。
  - 几条评论探讨了**当前 AI 模型的局限性**，重点关注它们无法处理需要实时学习和复杂推理的任务，例如**视频理解**和对 **3D spaces** 的理解。**Altruistic-Skill8667** 预测，实现 AGI 将需要 **compute power** 和 **online learning** 的重大进步，潜在的时间表可能会延长到 **2028 年或 2029 年**。
  - 一些用户对 AI 进步的**社会政治影响**表示担忧，认为 **AGI** 可能被用来在寡头政权下**奴役工人阶级**。一些评论还涉及**政府和技术寡头**在塑造 AI 未来方面的作用，并对比了**美国和中国**在技术控制和监管方面的差异。


**主题 3. Anthropic 的 ASI 预测：2-3 年时间线的影响**

- **[Anthropic CEO 现在确信 ASI（而非 AGI）将在未来 2-3 年内到来](https://i.redd.it/3dtbepq6pcee1.png)** ([Score: 173, Comments: 115](https://reddit.com/r/OpenAI/comments/1i6iu7m/anthropic_ceo_is_now_confident_asi_not_agi_will/)): **Anthropic 的 CEO** Amodei 预测，**Artificial Superintelligence (ASI)** 可能会在未来 **2-3 年**内实现，并超越人类智能。该公司计划为 **Claude** 发布具有增强记忆和双向语音集成功能的高级 AI 模型，以应对与 **OpenAI** 等公司的竞争。
  - 讨论突显了对 2-3 年内实现 **ASI 预测**的怀疑，一些研究人员和评论者认为 AI 模型需要重大改进，而当前的 AI 系统距离实现 **AGI** 仍有很大差距。**Dario Amodei** 的背景是 AI 研究，其可信度得到了认可，但关于他的预测是否现实仍存在争议。
  - 强调了 **narrow AI** 和 **general AI** 之间的区别，当前的 AI 系统在特定任务中表现出色，但缺乏 AGI 的综合能力。评论者指出，尽管取得了进步，AI 系统在处理许多对人类来说很简单的任务时仍然很吃力，通往 AGI 和 ASI 的路径仍不明确。
  - **资金和商业动机**受到质疑，一些人认为宣布 ASI 即将到来可能是为了配合融资活动的战略时机。关于 **Anthropic** 当前融资活动的评论支持了这一观点。

---

# AI Discord 简报

> 由 o1-preview-2024-09-12 生成的摘要之摘要的摘要

**主题 1. DeepSeek R1 震撼 AI 界**

- **DeepSeek R1 赶超竞争对手**：开源的 [DeepSeek R1](https://github.com/deepseek-ai/DeepSeek-R1) 性能比肩 OpenAI 的 o1，其高性价比和易用性令社区兴奋。用户反馈其在 coding 和 reasoning 任务中表现强劲，[benchmarks](https://x.com/TheXeophon/status/1881443117787984265) 显示其超越了其他模型。
- **跨平台集成热潮**：尽管偶尔会出现小问题，开发者们仍争先恐后地将 DeepSeek R1 集成到 [Cursor](https://www.cursor.com/)、[Codeium](https://codeium.com/) 和 [Aider](https://aider.chat/) 等工具中。讨论集中在成功案例和挑战上，特别是关于工具兼容性和性能方面。
- **审查与无审查版本引发辩论**：虽然有人称赞 DeepSeek R1 的安全特性，但也有人抱怨过度审查阻碍了实际使用。一个 [uncensored version](https://huggingface.co/Joseph717171/DeepSeek-R1-Distill-Llama-8B-OQ8_0-F32.EF32.IQ4_K-Q8_0-GGUF) 正在流传，引发了关于安全与可用性之间平衡的辩论。

**主题 2. OpenAI 的 Stargate 项目志存高远**

- **OpenAI 宣布 5000 亿美元 Stargate 投资计划**：OpenAI 与 SoftBank 和 Oracle 共同承诺向 [AI 基础设施投资 5000 亿美元](https://x.com/OpenAI/status/1881830103858172059)，并将其命名为 Stargate 项目。该计划旨在巩固美国的 AI 领导地位，被比作“阿波罗计划”。
- **社区热议 AI 军备竞赛**：惊人的投资额引发了关于 AI 军备竞赛和地缘政治影响的讨论。一些人担心，将 AI 发展定义为竞争可能会导致意想不到的后果。
- **Mistral AI 开启重大 IPO 进程**：与收购传闻相反，[Mistral AI](https://x.com/btibor91/status/1881692647456477189) 宣布了 IPO 计划并向亚太地区扩张，引发了对其盈利能力和战略的猜测。

**主题 3. 新模型与新技术突破边界**

- **Liquid AI 的 LFM-7B 引起轰动**：[Liquid AI 的 LFM-7B](https://www.liquid.ai/lfm-7b) 声称在 7B 模型中表现顶尖，支持包括英语、阿拉伯语和日语在内的多种语言。其对本地部署（local deployment）的关注让寻求高效、私有 AI 解决方案的开发者感到兴奋。
- **Mind Evolution 进化 AI 思维**：一篇新论文介绍了 [Mind Evolution](https://arxiv.org/abs/2501.09891)，这是一种进化搜索策略，在规划任务上实现了超过 **98% 的成功率**。这种方法击败了 Best-of-N 等传统方法，标志着 LLM inference 扩展的一次飞跃。
- **SleepNet 和 DreamNet 构想更佳 AI**：创新模型 [SleepNet 和 DreamNet](https://arxiv.org/abs/2410.18156) 提议在训练中加入“睡眠”阶段，模仿人类的学习过程。这些方法旨在平衡探索与精准度，激发了关于新型 AI 训练技术的讨论。

**主题 4. 用户与 AI 工具中的 Bug 和限制搏斗**

- **Windsurf 用户遭遇延迟风暴**：沮丧的 Windsurf 用户报告了 prompt 延迟和类似 *"incomplete envelope: unexpected EOF"* 的错误，迫使一些人转向 Cursor 等替代方案。社区在寻求解决方案的同时，对生产力受损表示不满。
- **Flow Actions 限制困扰程序员**：Codeium 的 Flow Actions 限制阻碍了工作流，用户抱怨反复出现的瓶颈。虽然出现了一些战略性使用的建议，但许多人仍在等待官方解决方案。
- **Bolt 用户因 Bug 损失 Token**：开发者哀叹由于 [Bolt](https://www.stackblitz.com/) 上的 Bug 代码导致 token 损失，主张通过免费调试来减轻损失。有人感叹：“*我已经数不清浪费了多少 token 了！*”，凸显了对成本的担忧。

**主题 5. AI 在创意和技术领域不断扩大的角色**

- **DeepSeek R1 精通数学辅导**：用户利用 DeepSeek R1 进行 [数学辅导](https://x.com/seo_leaders/status/1881462202831614085)，称赞其分步解决方案以及对特殊教育需求的支持。其速度和本地部署使其成为教育工作者的宠儿。
- **生成式 AI 塑造创意产业**：[相关文章](https://medium.com/@techyspacelovers/generative-ai-how-its-shaping-creative-industries-f3e11960fe38) 引发了关于 AI 对艺术和音乐影响的辩论，一些人担心 AI 可能会取代人类创作者。另一些人则认为，人类技能对于有效引导 AI 输出仍然至关重要。
- **Suno 因 AI 音乐面临版权诉讼**：AI 音乐生成器 [Suno](https://www.musicbusinessworldwide.com/500m-valued-suno-hit-with-new-copyright-lawsuit-from-germanys-gema/) 面临来自德国 GEMA 的新法律挑战，被指控使用未经授权的录音进行训练。该诉讼加剧了行业内关于 AI 生成内容合法性的辩论。

---

# 第一部分：高层级 Discord 摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **DeepSeek-R1 具有误导性的深度**：**DeepSeek-R1** 模型的最大 token 长度被发现是 **16384** 而不是预期的 **163840**，这引发了代码部署中的 **bug** 担忧。
   - 一篇关于 **RoPE factors** 和模型 embeddings 的[推文](https://x.com/fimbulvntr/status/1881821582571761920)引发了进一步讨论，成员们认为模型的使用并不完整。
- **LoRA Llama 3 微调策略**：Gautam Chutani 的一篇 **Medium 文章**展示了[基于 LoRA 的 Llama 3 微调](https://gautam75.medium.com/fine-tuning-llama-3-1-8b-for-function-calling-using-lora-159b9ee66060)，并集成了 **Weights & Biases** 和 **vLLM** 进行推理服务。
   - 他强调通过 LoRA 注入来减少 **GPU** 开销，社区评论指出这是一种比高端基准微调更节省资源的替代方案。
- **Chinchilla 的精确计算**：[Chinchilla 论文](https://paperswithcode.com/method/chinchilla)建议**模型大小**和**训练 tokens** 按比例增长以达到最高效率，这重塑了数据规划策略。
   - 参与者认为 **Chinchilla optimal** 方法避免了只关注狭窄的参数段，强调全参数参与是更安全的策略。
- **合成数据与混合数据的收益**：一些人提倡使用**合成数据**以实现更紧密的评估对齐，而另一些人则在 **Unsloth** 中应用混合格式数据集以扩大训练覆盖范围。
   - 与会者指出动态调整可以减轻过拟合，但在涉足现实世界材料之外时，特定领域的关联性仍存疑。
- **开源 UI 的快速推进**：**OpenWebUI**、**Ollama** 和 **Flowise** 成为下一个集成目标，而 **Kobold** 和 **Aphrodite** 通过 Kobold API 保持活跃。
   - *Invisietch* 确认了一个长长的待办事项列表，包括用于创建**合成数据集**的 CLI，旨在建立统一的后端 API 以简化流程。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **OpenAI 的 Stargate 项目宏大登场**：OpenAI 宣布了一项名为 **Stargate Project** 的 5000 亿美元投资计划，旨在与软银等合作在美国建设新的 AI 基础设施，详情见[此处](https://x.com/OpenAI/status/1881830103858172059)。
   - 社区成员对其战略影响议论纷纷，想知道日本的大额投资是否会助长新一轮的 **AI 竞争**。
- **DeepSeek R1 的进展与 Cursor 的痛点**：**DeepSeek R1** 可以通过 [OpenRouter](https://openrouter.ai/deepseek/deepseek-r1) 集成到 [Cursor](https://www.cursor.com/downloads) 中，尽管一些用户发现这种变通方法有限制，更愿意等待原生支持。
   - 基准测试讨论引用了 [Paul Gauthier 的推文](https://x.com/paulgauthier/status/1881428345973608901)，提到在 aider polyglot 测试中获得了 **57%** 的分数，引发了关于 **DeepSeek R1** 与其他 LLM 之间即将到来的竞争的辩论。
- **Cursor 0.45 回滚反应**：由于索引问题，**Cursor** 团队不断回滚 v0.45.1 更新，迫使开发者恢复到早期版本，参考 [Cursor Status](https://status.cursor.com)。
   - 一些用户对不稳定性感到沮丧，并提到官方声明极少，使他们的工作流程变得复杂，暗示他们可能会探索 [Codefetch](https://x.com/kregenrek/status/1878487131099898269) 等替代代码编辑器。
- **Claude 3.5 与 DeepSeek 竞争**：**Claude 3.5** 的性能有所提高，引发了与 **DeepSeek R1** 的直接对比，并促使了关于速度和准确性提升的讨论。
   - Anthropic 对未来更新的沉默引发了对其下一个版本的猜测，因为其光芒被竞争对手的势头所掩盖。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 的困扰与延迟激增**：多位用户抱怨 **Windsurf** 持续出现延迟问题，尤其是在进行代码提示（code prompts）时，部分用户遇到了 **'incomplete envelope: unexpected EOF'** 错误。
   - 尽管调整本地设置等潜在解决方案尚未产生确定的修复效果，但一些用户由于这些 Bug 考虑切换到 **Cursor**。
- **DeepSeek R1 在基准测试中占据主导地位**：根据 [Xeophon 的推文](https://x.com/TheXeophon/status/1881442133376454694)，社区成员对 **DeepSeek R1** 在各项性能测试中超越 **OpenAI o1-preview** 感到兴奋。
   - 随后的另一条 [推文](https://x.com/TheXeophon/status/1881443117787984265) 强调 **R1** 已处于领先地位，尽管对其在 **Codeium** 内的工具调用（tool-call）兼容性仍存疑问。
- **Flow Actions 削弱生产力**：许多人发现 **Flow Actions** 的限制干扰了他们的工作流，并提到全天都会反复遇到瓶颈。
   - 社区成员建议通过策略性使用和部分重置来缓解这一限制，但官方修复方案仍不确定。
- **Codeium 功能热潮**：一名用户请求在 **Codeium** 中增加对 **DeepSeek R1** 的支持，同时呼吁为 JetBrains IDE 用户提供更好的微调（fine-tuning）和稳健的更新。
   - 其他人提到需要通过 [Codeium 的功能请求页面](https://codeium.canny.io/feature-requests/p/add-rename-suggestion-like-in-vscode) 改进重命名建议，并重点介绍了用于命令行自动补全的 [Termium](https://codeium.com/blog/termium-codeium-in-terminal-launch)。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider v0.72.0 发布助力开发**：**Aider v0.72.0** 更新包括通过 `--model r1` 或 OpenRouter 支持 **DeepSeek R1**，此外还增加了 **Kotlin 语法**支持和新的 `--line-endings` 选项，解决了 Docker 镜像权限和 **ASCII fallback** 修复问题。
   - 社区成员指出，**Aider** 为此版本的发布贡献了 **52%** 的自身代码，并发现配合 GPT-4o 使用 `examples_as_sys_msg=True` 可以获得更高的测试分数。
- **DeepSeek R1 成为强有力的挑战者**：用户称赞 **DeepSeek R1** 的多语言处理能力，引用了[这条推文](https://x.com/0xluffyb/status/1881323971897110866)，称其几乎与 **OpenAI o1** 持平，且采用 **MIT 许可证**分发。
   - 对话暗示出于成本原因正从 **Claude** 转向 **DeepSeek R1**，并参考 [GitHub 上的 DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) 获取更多技术细节。
- **OpenAI 订阅与 GPU 成本引发辩论**：一些成员报告了 **OpenAI** 订阅退款情况，并权衡了 **DeepSeek** 的性价比，提到了关于定价不确定性的 [OpenAI CEO 文章](https://br.ign.com/tech/135086/news/ceo-da-openai-nao-sabe-o-que-fazer-com-o-comportamento-dos-assinantes-do-chatgpt)。
   - 欧洲用户还发现了更便宜的 **RTX 3060** 和 **3090** GPU，并查阅了 [Fireworks AI 文档](https://docs.fireworks.ai/guides/security_compliance/data_handling) 以了解 AI 驱动工作流中的隐私考量。
- **使用 DeepSeek R1 升级 Space Invaders**：一段 [实况编程视频](https://youtu.be/njJhjUgBTZg) 展示了由 **DeepSeek R1** 驱动的改进版 **Space Invaders** 游戏，证明了其在 **Aider LLM 排行榜**上名列第二。
   - 用户强调其在价格更低的情况下几乎等同于 **OpenAI o1**，这激发了人们对受益于 **R1** 编程专注性的游戏和开发场景的兴趣。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **DeepSeek 在数学领域的强势进军**：**DeepSeek R1** 成为**数学辅导**的强力选择，提供**逐步解决方案**并支持特殊教育需求，例如 [Andrew C 的推文](https://x.com/seo_leaders/status/1881462202831614085)中提到在 M2 Ultras 上运行 671B 版本。
   - 一位用户赞扬了该模型的**速度**和本地部署能力，并参考了 [DeepSeek-R1 GitHub 仓库](https://github.com/deepseek-ai/DeepSeek-R1)以了解高级使用场景。
- **本地模型魔法与 OpenAI 衔接**：爱好者们讨论了在 **4090 GPU** 和 **64GB RAM** 等强力家用配置上运行 **LLM**，参考 [LM Studio Docs](https://lmstudio.ai/docs/basics/rag) 和 [Ollama 的 OpenAI 兼容性博客](https://ollama.com/blog/openai-compatibility)，将本地模型与 OpenAI API 桥接。
   - 其他人强调了**量化**（Q3、Q4 等）对性能权衡的重要性，并探索了像 [Chatbox AI](https://chatboxai.app/zh) 这样的解决方案来统一**本地**和**在线**使用。
- **NVIDIA DIGITS 争议与 DGX OS 困境**：用户感叹**高昂的成本**（**128GB 约 3000 美元**）以及 **NVIDIA DIGITS** 支持的不确定性，指向 [NVIDIA DIGITS 文档](https://docs.nvidia.com/deeplearning/digits/index.html)以获取旧版见解。
   - 讨论指出 **DGX OS** 与旧版 DIGITS 的相似之处，有人建议将 **NVIDIA TAO** 作为现代替代方案，尽管这在以容器为中心的发布方面引发了混乱。
- **GPU 发热头疼与未来计划**：一些人提到高性能 GPU 产生的**过热问题**，开玩笑说由于持续燃烧不需要清洁，并参考二手销售以寻求潜在的**成本**节约。
   - 其他人计划采用**无 GUI** 方法以优化性能，重点是通过**更轻量化的配置**来减轻高级 ML 任务中的**散热**压力。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Liquid AI 的 LFM-7B 在本地部署中崛起**：Liquid AI 推出了 **LFM-7B**，这是一款非 Transformer 模型，声称在 7B 级别拥有顶级性能，扩展了包括**英语**、**阿拉伯语**和**日语**在内的语言覆盖范围（[链接](https://www.liquid.ai/lfm-7b)）。
   - 社区成员赞扬了其本地部署策略，一些人认为该模型的**自动架构搜索**是一个潜在的差异化优势。
- **Mind Evolution 驱动 LLM 推理**：一篇关于 **Mind Evolution** 的新论文展示了一种进化方法，在 **TravelPlanner** 和 **Natural Plan** 等任务中超越了 Best-of-N，使用 **Gemini 1.5 Pro** 实现了超过 **98%** 的成功率（[arXiv 链接](https://arxiv.org/abs/2501.09891)）。
   - 工程师们讨论了该方法的迭代生成和 Prompt 重组，将其描述为扩展推理计算的精简路径。
- **DeepSeek-R1 Distill 模型评价褒贬不一**：用户试用了 **DeepSeek-R1 Distill** 进行量化调整和性能分析，参考了一个接近 **8B** 参数的 [Hugging Face 仓库](https://huggingface.co/Joseph717171/DeepSeek-R1-Distill-Llama-8B-OQ8_0-F32.EF32.IQ4_K-Q8_0-GGUF)。
   - 一些人称赞其推理输出，而另一些人则认为它在处理日常 Prompt 时过于冗长，但它仍是高级思考时间的亮点。
- **SleepNet 与 DreamNet 带来“夜间”训练**：**SleepNet** 和 **DreamNet** 提出了模拟“睡眠”的有监督加无监督循环来优化模型状态，详见 [Dreaming is All You Need](https://arxiv.org/abs/2409.01633v2) 和 [Dreaming Learning](https://arxiv.org/abs/2410.18156)。
   - 它们使用 Encoder-Decoder 方法在离线阶段重新访问隐藏层，引发了关于综合探索的讨论。
- **Mistral 对 Ministral 3B 和 Codestral 2501 的思考**：Mistral 预告了 **Ministral 3B** 和 **Codestral 2501**，在紧张的 AI 竞争格局中引发了对其权重许可计划的猜测。
   - 观察者们想知道 Mistral 的方法（类似于 Liquid AI 的架构实验）是否能为更小规模的部署开辟出专门的利基市场。

---

## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Bolt 更大胆的代码包含功能**：Bolt 的最新更新消除了**白屏**故障，并包含了[完整代码交付的修复](https://x.com/boltdotnew/status/1881731948051415059)，确保从第一个 prompt 开始就能实现**精准**设置，详见[此公告](https://x.com/boltdotnew/status/1881442318110347291)。
   - 工程师们欢迎这一**全面**的转变，称“不再有偷懒的代码！”，并赞扬了新项目更流畅的启动体验。
- **Prismic 困境与静态解决方案**：一位用户在为管道网站集成 **Prismic CMS** 时遇到问题，引发了先构建静态网站以获得面向未来的灵活性的建议。
   - 社区成员倾向于极简方法，其中一人指出“简单网站的 **CMS** 开销”过于复杂。
- **Firebase vs Supabase 对决**：一位用户主张将 **Supabase** 换成 **Firebase**，称其为开发者更简单的路径。
   - 其他人同意 **Firebase** 简化了初始设置，强调了它如何加速快速概念验证。
- **Token 纠纷**：开发者报告称由于 Bolt 上的错误代码导致损失了 **tokens**，主张通过免费调试来遏制这些损失。
   - 成本担忧飙升，一位用户宣称“我已经数不清浪费了多少 tokens 了！”
- **Next.js & Bolt：结构性联系**：一位社区成员尝试使用 Bolt 将 WordPress 博客集成到 **Next.js** 中，但发现框架更新速度快于 AI 工具。
   - 意见不一，有人认为 Bolt 可能无法紧跟 **Next.js** 的**快速**变化。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Sonar 凭借速度和安全性崛起**：Perplexity 发布了用于**生成式搜索**的 [Sonar 和 Sonar Pro API](https://sonar.perplexity.ai/)，具有**实时**网络分析功能，并展示了 **Zoom** 的大规模采用，同时在 **SimpleQA** 基准测试中表现优于成熟的引擎。
   - 社区成员赞赏其**实惠**的分级定价，并*指出*没有用户数据被用于 LLM 训练，暗示了更安全的企业级用途。
- **DeepSeek vs O1 传闻**：多位成员询问 **DeepSeek-R1** 是否会取代 Perplexity 中缺席的 **O1**，引用了关于高级推理能力的[公开暗示](https://x.com/AravSrinivas/status/1881458694266953934)。
   - 其他人称赞 **DeepSeek-R1** 免费且性能顶尖，*称其为*“最佳替代方案”，而一些人对 **O1** 的计划前景仍持怀疑态度。
- **Claude Opus：退役还是坚挺？**：一些用户宣称 **Claude Opus** 已退役，取而代之的是 `Sonnet 3.5`，质疑其在创意任务中的可行性。
   - 其他人强调 Opus 在复杂项目中继续表现出色，*坚持认为*尽管有传言称其将被取代，但它仍然是该系列中最先进的。
- **Sonar Pro 分级与域名过滤测试版**：贡献者强调了 **Sonar** 和 **Sonar Pro** 的[新使用分级](https://docs.perplexity.ai/guides/usage-tiers)，指出 **search_domain_filter** 是第 3 级的测试功能。
   - 许多用户寻求从 API 输出中获得直接的 **token 使用情况**洞察，而一些人则推动在欧洲数据中心进行符合 **GDPR** 标准的托管。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **DeepSeek R1 横扫基准测试**：1月20日，中国的 **DeepSeek AI** 发布了 **R1**，在 [ARC-AGI 公开评估](https://x.com/arcprize/status/1881761987090325517)中达到了 **20.5%**。
   - 它在联网任务中表现优于 **o1**，完整的发布细节见[此处](https://www.interconnects.ai/p/deepseek-r1-recipe-for-o1)。
- **Mistral 的重大 IPO 举措**：与收购传闻相反，**Mistral AI** 宣布了 IPO 计划，并为亚太市场开设了**新加坡**办公室。
   - 成员们对 Mistral 的盈利能力进行了推测，引用[此更新](https://x.com/btibor91/status/1881692647456477189)作为其大胆战略的证明。
- **Stargate 凭借 5000 亿美元承诺激增**：**OpenAI**、**SoftBank** 和 **Oracle** 在 **Stargate** 旗下联手，承诺在四年内投入 **5000 亿美元**以加强美国的 AI 基础设施。
   - 他们将这项宏大的投资比作 **Apollo** 计划等历史性壮举，旨在巩固美国在 AI 领域的领导地位。
- **Anthropic 谋划 Claude 的下一步**：在**达沃斯**，CEO **Dario Amodei** 预告了 **Claude** 的**语音模式**和可能的网页浏览功能，详见[此 WSJ 采访](https://youtu.be/snkOMOjiVOk?si=xyCM-nx3M6Ewoep2)。
   - 他暗示将发布更强大的 Claude 版本，社区正在讨论更新发布的频率。
- **Tulu 3 RLVR 引发好奇**：关于 **Tulu 3** 的 **RLVR** 的一个[海报项目](https://x.com/hamishivi/status/1881398642403356678)引起了关注，承诺提供强化学习的新方法。
   - 爱好者们计划将其与 **open-instruct** 框架合并，预示着模型使用方式将发生更广泛的变革。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Tavily Search MCP 服务器飙升**：新的 [Tavily Search MCP 服务器](https://glama.ai/mcp/servers/0kmdibf9t1)上线，为 LLM 提供了**优化的网页搜索**和**内容提取**功能，支持 SSE、stdio 和基于 Docker 的安装。
   - 它使用 Node 脚本进行快速部署，为 **MCP 生态系统**提供了更广泛的服务器选择。
- **MCP Language Server 对决**：开发者们测试了 [isaacphi/mcp-language-server](https://github.com/isaacphi/mcp-language-server) 和 [alexwohletz/language-server-mcp](https://github.com/alexwohletz/language-server-mcp)，旨在大型代码库中实现 **get_definition** 和 **get_references**。
   - 他们注意到第二个仓库可能不太成熟，但社区仍渴望实现类似 **IDE** 的 MCP 功能。
- **Roo-Clines 变得更加丰富**：成员们支持将 **roo-code** 工具添加到 roo-cline 中，以处理扩展的语言任务，包括在大型项目中的代码操作。
   - 他们设想更深层次的 **MCP 协同效应**来简化代码管理，建议在单一 CLI 生态系统中进行高级编辑。
- **LibreChat 引发抱怨**：一位用户抨击 **LibreChat** 配置复杂且 API 支持不可预测，尽管他们很欣赏其精美的 UI。
   - 他们还哀叹缺乏使用限制，并将其与 **Sage** 或内置 **MCP** 服务器等更严格的平台进行了比较。
- **Anthropic 模型与 Sage 的对决**：关于 **Anthropic** 模型 r1 的可行性爆发了激烈的讨论，一些人猜测他们“很可能”能让它运行起来。
   - 其他人则倾向于在 macOS 和 iPhone 上使用 **Sage**，相比不确定的 Anthropic 集成，他们更喜欢少出点麻烦。

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Llama 在 Samba Nova 的最后时刻**：由于 **Samba Nova** 的变更，**免费 Llama 端点**将于本月结束，直接用户访问权限将被移除。
   - Samba Nova 将切换到具有新定价的 **Standard** 变体，引发了关于付费使用的讨论。
- **DeepSeek R1 获得网页搜索与自由表达**：**DeepSeek R1** 模型在 [OpenRouter](https://x.com/OpenRouterAI/status/1881785438043799765) 上启用了网页搜索 Grounding，保持了**无审查**的方式，价格为每输入 token **$0.55**。
   - 社区对比显示其性能接近 **OpenAI 的 o1**，[Alex Atallah 的帖子](https://x.com/xanderatallah/status/1881456463786512737)中提到了关于**微调 (fine-tuning)** 的讨论。
- **Gemini 2.0 Flash：64K Token 奇迹**：新发布的 **Gemini 2.0 Flash Thinking Experimental 01-21** 提供 **100 万**上下文窗口以及 **64K** 输出 token。
   - 观察者注意到在其 10 分钟的发布过程中存在一些命名上的小瑕疵；它仍可通过 **AI Studio** 使用，无需分级密钥。
- **巧妙的推理内容技巧出现**：一位用户揭露了一种通过巧妙的 prompt 前缀从 [DeepSeek Reasoner](https://api-docs.deepseek.com/guides/reasoning_model) 中诱导**推理内容**的方法。
   - 人们对残留 CoT 数据导致的 token 堆积表示担忧，促使了更好的消息处理策略。
- **Perplexity 的 Sonar 模型备受关注**：**Perplexity** 推出了具有网页搜索扩展功能的新 **Sonar** LLM，详见[此推文](https://x.com/risphereeditor/status/1881789442530435513)。
   - 虽然有些人对潜在的集成感到兴奋，但其他人对模型的实用性表示怀疑，并敦促为 **OpenRouter** 的支持进行投票。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **提升 GPT-2 收益**：工程师们讨论了调整 GPT-2 重新训练的 `max_steps`，建议将其翻倍以进行两个 epoch，从而防止**快速学习率衰减 (rapid learning rate decay)**，并参考了 **Andrew Karpathy** 的方法。
   - 他们还警告说，草率的更改可能会浪费资源，建议在做出**微调 (fine-tuning)** 决策前进行透彻的了解。
- **实时问答中的 RAG 启示**：一场关于 **RAG** 和模型**工具使用 (tool use)** 的**实时问答**定于**东部时间周二上午 6:00** 在 Discord Stage 举行，鼓励开发者分享经验。
   - 参与者计划应对集成新实现中的挑战，旨在营造一个激发共享见解的协作环境。
- **Cohere CLI：Transformer 的终端对话**：新的 **Cohere CLI** 允许用户从命令行与 Cohere 的 AI 聊天，已在 [GitHub](https://github.com/plyght/cohere-cli) 上展示。
   - 社区成员赞扬了它的便利性，一些人强调了**基于终端 (terminal-based)** 的交互如何加速迭代开发。
- **Cohere For AI：社区动力源**：爱好者们互相敦促加入 **Cohere For AI** 倡议，进行开放的机器学习协作，参考了 [Cohere 官方研究页面](https://cohere.com/research)。
   - 他们还提到**试用密钥**每月提供 1000 次免费请求，为渴望测试 AI 解决方案的新手提供了一个友好的空间。
- **LLM 输出中的数学缺陷**：成员们指出 **Cohere** 错误地将 18 个月计算为 27 周，对 LLM 的数学可靠性表示怀疑。
   - 他们将此归因于**分词 (tokenization)** 问题，称其为一种普遍的缺陷，如果不加以解决，可能会导致项目失败。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **课堂征服：大学课程中的 NotebookLM**：成员们建议按主题而非单个来源来组织 NotebookLM，以确保**数据一致性 (data consistency)**，并指出 1:1 的笔记本与来源设置最适合单文件的**播客生成 (podcast generation)**。
   - 他们强调这能*消除杂乱*并促进更顺畅的协作，有可能改变学术环境中的学习习惯和资源共享。
- **视频胜利：AI eXplained 的电影化展开**：**AI eXplained** 频道发布了关于 **AI 生成视频 (AI-generated videos)** 的新视频，重点介绍了剧本创作和动画制作方面的进展。
   - 早期观众提到，这些方法引发了重塑电影行业的*兴趣浪潮*，并预测视听 AI 领域将有更多突破。
- **Gemini 收益：面向开发者的 Code Assist**：社区成员推荐使用 **Gemini Code Assist** 来获取更深层的代码库洞察，称其在针对性代码查询方面比 NotebookLM 更准确。
   - 他们指出，除非有**非常具体的指令**引导，否则 NotebookLM 可能会出错，这引发了关于代码分析方法和可靠性的讨论。
- **神圣摘要：教会服务中的 NotebookLM**：一位参与者利用 NotebookLM 解析大量的布道讲稿，目标是一份 **250 页** 的合集，甚至是一个 **2000 页** 的圣经研究。
   - 他们称其为处理大型宗教文本的*游戏规则改变者*，赞扬其在连接技术与信仰方面的效用。
- **工具宝库：插件与应用增强 NotebookLM**：用户交流了关于插件的建议，包括 [OpenInterX Mavi](https://mavi.openinterx.com) 和 [Chrome Web Store](https://chromewebstore.google.com/search/notebookLM) 扩展程序，以增强功能。
   - 他们测试了*保留常用提示词 (prompts)* 以提高工作效率的方法，并对未来更深度的 **NotebookLM** 集成表示期待。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **使用 ControlNet 制作连贯漫画**：成员们探索了使用 **ControlNet** 驱动的 AI 漫画分镜，以保持场景细节的一致性，通过单独生成每一帧来保持角色稳定。他们发现这种方法仍然会产生不同的结果，需要频繁重新生成以维持连贯性。
   - 他们还辩论了高级提示词或额外的训练数据是否能改善结果，一些人认为一旦 **Stable Diffusion** 更加成熟，未来会有改进潜力。
- **AI 艺术争议持续**：贡献者注意到创意社区对 **AI 渲染艺术品 (AI-rendered artwork)** 的抵制情绪增强，强调了对可信度和尊重原创风格的质疑。他们引用了关于 AI 艺术是取代还是延伸了手工创作的广泛辩论。
   - 其他人提出了关于使用公共仓库训练数据的伦理担忧，并提到了要求确保原创作者获得署名的准则呼吁。
- **Stable Diffusion AMD 设置障碍**：个人分享了在 AMD 硬件上运行 **Stable Diffusion** 的困难，指出驱动问题和性能较慢。他们参考了 Discord 中的置顶说明作为变通方法，但承认需要更强大的官方支持。
   - 一些人通过更新库取得了成功，但其他人仍面临意外黑屏或渲染不完整的问题，需要手动重置 GPU。
- **手动 vs. AI 背景调整**：爱好者们辩论了是使用 GIMP 进行直接的背景编辑，还是依赖 **Stable Diffusion** 进行自动增强。他们报告称手动编辑提供了更受控的结果，特别是对于个人摄影中敏感细节的处理。
   - 一些人认为 AI 解决方案在处理细微任务时仍缺乏精细度，而另一些人则认为如果模型获得更多专业训练，前景广阔。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **进化思维与驯服 GRPO**：用于扩展 LLM 推理的 **Mind Evolution** 策略在 TravelPlanner 和 Natural Plan 上的成功率飙升至 **98%** 以上，详见 [arXiv 论文](https://arxiv.org/abs/2501.09891)。
   - 一个简单的本地 **GRPO** 测试正在进行中，未来计划通过 **OpenRLHF** 和 Ray 进行扩展，并将 **RL** 应用于数学数据集。
- **TMA 在 Triton 中占据核心地位**：社区成员研究了 Triton 中的 **TMA** 描述符，利用 `fill_2d_tma_descriptor` 并面对导致崩溃的 autotuning 陷阱。
   - 分享了一个带有 TMA 的 **persistent GEMM** 工作示例，但由于 autotuner 支持有限，目前仍需手动配置。
- **Fluid Numerics 启动 AMD MI300A 测试**：**Fluid Numerics** 平台推出了其 **Galapagos** 集群的订阅服务，该集群配备了用于 AI/ML/HPC 工作负载的 **AMD Instinct MI300A** 节点，并提供了 [访问申请链接](https://www.fluidnumerics.com/shop/p/rcc-allocation-monthly-subscription)。
   - 他们鼓励用户测试软件并对比 **MI300A** 与 **MI300X** 的性能，邀请进行广泛的基准测试。
- **PMPP 书籍新增更多 GPU 精华内容**：建议重读最新版的 **PMPP Book**，因为它更新了 2022 年版中缺失的内容，并增加了新的 **CUDA** 材料。
   - 成员们推荐使用 [Cloud GPUs](https://cloud-gpus.com/) 或 **Lightning AI** 等 **cloud GPU** 选项来进行书中练习的动手实践。
- **Lindholm 的统一架构遗产**：工程师 **Lindholm** 最近从 **Nvidia** 退休，他于 2024 年 11 月关于其 **unified architecture**（统一架构）的深度演讲可通过 [Panopto](https://ubc.ca.panopto.com/Panopto/Pages/Viewer.aspx?id=880a1d92-30d7-4683-80e7-b1e000f501d3) 观看。
   - 参与者了解了他富有影响力的设计原则以及直到两周前退休为止所做出的贡献。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **GGUF 在竞争格式中脱颖而出**：社区注意到 **GGUF** 是消费级硬件首选的量化方案，引用 [LLM 推理后端基准测试](https://www.bentoml.com/blog/benchmarking-llm-inference-backends) 展示了其强大的性能优势。
   - 他们对比了 **vLLM** 和 **TensorRT-LLM** 等工具，强调初创公司通常选择 **Ollama** 等简单的后端，以实现开箱即用的本地化部署。
- **R1 之谜与 Qwen 特性**：成员们对 **R1** 进行了细致研究，讨论了其对 PRMs 的使用，并思考了 4bit/3bit 与 **f16** 对 MMLU-PRO 性能的影响。
   - 他们还考虑将 **Qwen R1** 模型转换为 **Q-RWKV**，关注 **math500** 等测试以确认转换效果，并探讨了在多次生成响应时如何最好地估算 **pass@1**。
- **Titans 解决深度网络内存问题**：**Titans** 论文（[arXiv:2501.00663](https://arxiv.org/abs/2501.00663)）提出将短期记忆与长期记忆结合以增强序列任务，该研究基于循环模型（recurrent models）和 attention。
   - 一位用户询问：“在如此大的数据集上微调模型是否更快？”，而其他人则在权衡扩展数据规模是否优于增量方法。
- **转向（Steering）方案依然匮乏**：目前还没有一个开源库在 LLM 的 **SAE-based** 转向领域占据主导地位，尽管 [steering-vectors](https://github.com/steering-vectors/steering-vectors) 和 [repeng](https://github.com/vgel/repeng) 等项目显示出潜力。
   - 他们还提到了 [representation-engineering](https://github.com/andyzoujm/representation-engineering)，注意到其自顶向下的方法，但强调了目前普遍缺乏统一的方法论。
- **NeoX：维度争议下的 HF 格式转换**：`convert_neox_to_hf.py` 中的一个 **RuntimeError** 揭示了维度不匹配问题（[8, 512, 4096] vs 4194304），这可能与多节点设置和 **model_parallel_size=4** 有关。
   - 针对 **3x** 中间层维度设置产生了疑问，而共享的配置提到 **num_layers=32**、**hidden_size=4096** 和 **seq_length=8192** 影响了导出过程。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI 为 Stargate 项目注资 5000 亿美元**：OpenAI 公布了 **The Stargate Project**，承诺在未来四年内投资 **5000 亿美元**，旨在建设美国的 AI 基础设施，首期投入 **1000 亿美元**。
   - 包括 **SoftBank** 和 **Oracle** 在内的主要支持者正大举押注这一倡议，强调在美国创造就业机会和保持 AI 领导地位。
- **Gemini 2.0 获得实验性更新**：针对 **Gemini 2.0 Flash Thinking** 的反馈促使 Noam Shazeer 引入了反映用户驱动改进的新变化。
   - 这些调整旨在完善 **Gemini** 的技能集，并增强其对实际使用情况的响应能力。
- **DeepSeek 发布低推理成本的 V2 模型**：新发布的 **DeepSeek V2** 以降低运营成本和显著的性能提升脱颖而出。
   - 其架构在社区中引起了轰动，展示了一种挑战既有模型的全新方法。
- **Ai2 ScholarQA 助力文献综述**：**Ai2 ScholarQA** 平台提供了一种提问方式，可以汇总多篇科学论文的信息，提供对比见解。
   - 该工具旨在通过按需提供更深入的引用和参考资料来简化严谨的研究。
- **随着 WandB 达到 SOTA，SWE-Bench 飙升**：**WandB** 宣布其 **SWE-Bench** 提交结果现已被公认为 State of the Art (SOTA)，引起了人们对该基准测试重要性的关注。
   - 该公告强调了性能指标方面的竞争驱动力，并促进了对高级测试的进一步探索。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **DeepSeek R1 与 Sonnet 对决**：成员们讨论了在拥有 **32 GB RAM** 和 **16 GB VRAM** 的系统上本地运行蒸馏至 [Qwen 32B Coder](https://link.to.related.info) 的 **DeepSeek R1**，通过将繁重计算卸载到 CPU 来实现可行性能。
   - 他们报告 R1 在编码方面的 **失败率为 60%**，但仍优于 **4O 和 Sonnet**（失败率为 99%），尽管在 Ollama 上的稳定性仍不确定。
- **Generative AI 席卷创意产业**：一篇 [Medium 文章](https://medium.com/@techyspacelovers/generative-ai-how-its-shaping-creative-industries-f3e11960fe38) 强调了 **Generative AI** 创作艺术的能力，引发了它可能取代人类创作者的担忧。
   - 其他人则认为，**人类技能**对于有效塑造 AI 输出仍然至关重要，使艺术家能够参与到流程中。
- **内容合规性讨论**：有人指出 **DeepSeek** 会避开关于 **CCP** 的批评性或幽默输出，这让人想起早期的 GPT 合规性问题。
   - 用户质疑这些限制是否限制了**表达**或阻碍了开放式辩论。
- **Archotech 猜测四起**：一位用户沉思 AI 是否会进化成 **Rimworld** 风格的 archotechs，暗示了意想不到的能力和产物。
   - 他们建议，随着 AI 公司不断训练更大的模型，“我们可能会意外产生高级实体”。
- **GPT 宕机和响应延迟**：频繁出现的 'Something went wrong' 错误中断了与 **GPT** 的对话，尽管重新开启会话通常能解决问题。
   - 几位成员注意到了**性能迟缓**，将缓慢的回复描述为集体恼火的源头。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Neural ODEs 激发 RL 策略**：在 #general 频道，成员们表示 **Neural ODEs** 可以通过用层来建模函数复杂度来改进机器人技术，并引用了 [Neural Ordinary Differential Equations 论文](https://arxiv.org/abs/1806.07366)。
   - 他们还讨论了较小的模型如何通过 RL 中重复的随机初始化发现高质量的推理，指出噪声和不规则性有助于探索。
- **GRPO 获得支持**：在 #paper-discussion 频道，**DeepSeek** 的 **GRPO** 被称为去掉了价值函数的 **PPO**，依靠 Monte Carlo 优势估计来进行更简单的策略微调，正如[官方推文](https://fixupx.com/natolambert/status/1881380809153847711)所示。
   - 一份[最近的出版物](https://arxiv.org/abs/2402.03300v3)强调了减少的开销，同时该小组还通过从 50 多名志愿者中招募 **12** 人来解决审稿人短缺的问题。
- **Suno 应对版权指控**：在 #ml-news 频道，AI 音乐生成器 **Suno** 正面临来自 **GEMA** 的另一起版权诉讼，此前已有来自主要唱片公司的诉讼，详情见 [Music Business Worldwide](https://www.musicbusinessworldwide.com/500m-valued-suno-hit-with-new-copyright-lawsuit-from-germanys-gema/)。
   - 估值 **5 亿美元**的 Suno 及其竞争对手 **Udio** 被指控在未经授权的录音上进行训练，引发了行业关于 AI 生成内容合法性的辩论。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **C 与 Python 之争**：成员们就 **C** 的严谨性和 **Python** 更快速的内存管理见解展开辩论，并参考了未来在 **JS** 或 **Python** 中的应用。
   - 一位参与者强调先学习 **C** 可以为职业转型打下更深的基础，但观点差异很大。
- **论坛与 Discord 的抉择**：许多人敦促明确在 **Discord** 与**论坛**上发布**项目**的区别，理由是在快速聊天的环境中难以检索重要的讨论。
   - 他们建议使用**论坛**进行深入更新，同时保留 **Discord** 用于快速反馈。
- **Mojo 的 .gitignore 魔法**：贡献者注意到 **Mojo** 的 `.gitignore` 仅排除 `.pixi` 和 `.magic` 文件，这显得非常简洁。
   - 没有人提出异议，团队对这种精简的默认配置表示赞赏。
- **Mojo 与 Netlify 不兼容？**：有人提出了关于在 **Netlify** 上托管使用 `lightbug_http` 的 **Mojo** 应用的问题，并参考了 Rust 应用的成功经验。
   - 成员表示 **Netlify** 缺乏对 Mojo 的原生支持，并参考了[构建时可用软件](https://docs.netlify.com/configure-builds/available-software-at-build-time/)以了解可能的功能。
- **Mojo 的域名困境**：一位用户询问 **Mojo** 是否会从 **Modular** 独立出来，并像其他语言一样申请 `.org` 域名。
   - 开发者确认目前没有此类计划，确认其目前仍保留在 **modular.com** 下。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Workflows 在 GCloud Run 上飞跃**：一份新指南解释了如何在 **Google Cloud Run** 上启动用于 ETL 和查询任务的[双分支 RAG 应用程序](https://t.co/nU1BctUh7s)，详细介绍了通过 **LlamaIndex** 实现的 serverless 环境和事件驱动设计。
   - 成员们指出三大特性——**双分支**架构、**serverless** 托管和**事件驱动**方法——是简化 AI 工作负载的关键。
- **Chat2DB GenAI 聊天机器人攻克 SQL**：贡献者重点介绍了开源的 [Chat2DB 聊天机器人](https://t.co/l1SFCEkiOC)，解释了它如何让用户使用 **RAG** 或 **TAG** 策略以日常语言查询数据库。
   - 他们强调了其多模型兼容性，支持 **OpenAI** 和 **Claude**，这使其成为数据访问的灵活工具。
- **LlamaParse 拯救 PDF 提取**：参与者推荐使用 [LlamaParse](https://docs.llamaindex.ai/en/stable/llama_cloud/llama_parse/) 进行 PDF 解析，称其为全球首个用于 LLM 用例的 **genAI 原生**文档平台。
   - 他们赞扬了其强大的数据清洗功能，并将其视为解决棘手的可选文本 PDF 的方案。
- **无痕模式解决文档故障**：一位用户报告称，在普通浏览器会话中查看 [LlamaIndex 文档](https://docs.llamaindex.ai/)时，页面会不断滚动回顶部。
   - 他们确认 Microsoft Edge 的**无痕模式**解决了该故障，表明插件冲突可能是原因。
- **带有 Gemini 的 CAG 遭遇 API 壁垒**：有人询问如何将 **Cached Augmented Generation (CAG)** 集成到 Gemini 中，结果得知模型级访问权限至关重要。
   - 他们发现目前**没有供应商**提供对 API 如此深度的控制，目前该想法陷入停滞。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **ModernBert 实体出现**：一位用户展示了在 **ModernBert** 中识别实体的语法，为旅游主题提供了分层文档布局，并寻求 embeddings 的最佳实践。
   - 他们寻求关于围绕基于实体的任务构建这些文档的建议，希望能优化整体性能。
- **Jinja 宝库成为焦点**：一位参与者请求关于 **Jinja** 模板高级功能的强大资源，引发了社区的广泛关注。
   - 其他人也加入进来，指出改进模板逻辑可以简化各种项目中的动态渲染。
- **LMstudio 咨询找到了归宿**：另一位用户寻求关于 **LMstudio** 的指导，在努力寻找专用 Discord 链接的同时询问当前频道是否合适。
   - 他们还提到了 **Adobe Photoshop** 的问题，引发了关于非官方支持渠道的调侃。
- **Photoshop 与非法幽默**：简短的交流暗示了一个关于 **Adobe Photoshop** 的可能非法的问题，引发了关于此类询问性质的玩笑。
   - 讨论简短地转向了在公共论坛分享可疑请求的更广泛担忧。
- **Nomic 税收与实习生征税**：成员们开玩笑说要增加 **Nomic** 的税收，其中一位参与者声称他们应该是这些资金的接收者。
   - 引用 [这个 GIF](https://tenor.com/view/willj-oprah-oprah-winfrey-winfrey-you-get-a-car-gif-2219821026349492069) 突显了对话的俏皮基调。



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Bud-E 支持 13 种语言**：LAION 透露 **Bud-E** 已扩展至英语之外，支持 **13 种语言**（未指明完整列表），并利用 fish TTS 模块实现语音功能。
   - 团队暂时“冻结”了现有的项目路线图，以强调 **音频** 和 **视频** 数据集的集成，导致开发进度略有延迟。
- **Suno Music 的音频力量**：[Suno Music](https://x.com/SunoMusic/status/1881742789639057828) 功能允许用户通过录制自定义音频输入来创作自己的歌曲，吸引了寻求快速实验的移动端创作者。
   - 成员们对**广泛的易用性**表示兴奋，强调了该平台多样化创作工作流的潜力。
- **BUD-E 与 School-BUD-E 成为焦点**：LAION 宣布 **BUD-E** 1.0 版本是一款 100% 开源的语音助手，适用于通用和**教育**用途，包括用于课堂的 [School Bud-E](https://www.youtube.com/watch?v=y4DRYF9sfMU)。
   - 这一里程碑促进了**普及化访问**并鼓励 AI 驱动的**教育科技 (ed-tech)**，并在展示 BUD-E 功能的 [教程视频](https://www.youtube.com/watch?v=IxHnpISMNPo) 中进行了演示。
- **BUD-E 的多平台灵活性**：工程师们称赞 BUD-E 提供了与自托管 APIs 和本地数据存储的兼容性，确保了**隐私**和易于部署。
   - 根据 [LAION 的博客文章](https://laion.ai/blog/bud-e-release/)，**桌面**和**网页**变体满足了广泛的用户需求，扩大了全球范围内的免费教育覆盖。



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **声明表单困惑**：一位成员询问是否需要在 12 月提交后再次填写声明表单（Declaration Form），澄清现在只有新成员必须提交。
   - 工作人员为错过最初截止日期的人重新开放了表单，确保之前的提交者无需额外步骤。
- **赞助商提供黑客松风格的项目**：一位参与者询问企业赞助商是否会在下一期 MOOC 中提供类似实习的任务，并参考了上学期的黑客松作为灵感。
   - 组织者表示，赞助商主导的演讲可能会暗示实习机会，尽管尚未透露正式安排。
- **MOOC 教学大纲预计于 1 月 27 日发布**：一位成员想知道新的 MOOC 教学大纲何时发布，工作人员指出 **1 月 27 日** 是可能的日期。
   - 他们正在先确定演讲嘉宾，但承诺在那天之前提供一份初步大纲。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **BEAM 拖慢了 YoloV8**：一位用户报告说，在 **BEAM** 下使用 `python examples/webgpu/yolov8/compile.py` 运行 **YoloV8** 使吞吐量从 **40fps** 锐减至 **8fps**，引发了对 bug 的担忧。
   - **George Hotz** 指出 **BEAM** 不应降低性能，并建议调查代码路径中潜在的异常。
- **WebGPU-WGSL 障碍减慢了 BEAM**：另一位用户怀疑 **WGSL** 转换为 **SPIR-V** 可能会增加开销，从而削弱实时推理速度。
   - 他们还强调 **BEAM** 需要精确的后端支持，引发了关于 **WebGPU** 特定硬件优化的疑问。



---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 的 'Tune Cat' 势头强劲**：一位成员称赞了 **Torchtune** 软件包，并引用了一个 [GitHub Issue](https://github.com/pytorch/torchtune/issues/2281)，该议题提议增加 `tune cat` 命令以简化使用。
   - 他们形容源代码 *读起来非常愉悦*，表明了极佳的用户体验。
- **TRL 的命令导致终端信息膨胀**：一位成员开玩笑说 **TRL** 的帮助命令延伸到了 **三个** 终端窗口，远超典型的帮助输出。
   - 他们建议，对于需要所有技术细节的用户来说，这种详尽的性质可能仍然至关重要。
- **LLMs 探索不确定性与内部推理**：讨论集中在**模型应该量化不确定性**以增强可靠性的观点上，同时 **LLMs** 在回答之前似乎会进行自己的 Chain of Thought (CoT)。
   - 这两点都强调了向更好解释性迈进的趋势，并有迹象表明存在用于深度推理的隐蔽 CoT 步骤。
- **通过 LLM 步骤提示与 Distillation 推进 RL**：有人建议为 **RL-LLM** 引入**思考步骤提示 (thinking-step prompts)**，为标准的基于目标的指令增加结构。
   - 另一位成员提议在模型 Distillation 之上应用 RL 技术，预期即使是较小的模型也能获得进一步的提升。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **动态 DSPy：RAG 与实时数据的邂逅**：一位用户询问 **基于 DSPy 的 RAG** 如何管理不断变化的信息，暗示了实时更新对于低开销知识检索流水线的重要性。
   - 他们建议未来的工作可以集中在缓存机制和增量索引上，使 **DSPy** 在处理动态工作负载时保持敏捷。
- **悬而未决的问题与语法错误**：另一个帖子提出了 DSPy 中的一个**开放性问题 (open problem)**，强调了对长期存在的技术问题的持续关注。
   - 还出现了一个语法错误（*'y=y'* 应该使用数字），凸显了社区对细节的关注以及在消除小问题方面的积极参与。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **ArXiv 作者呼吁更好的数据**：题为 [Towards Best Practices for Open Datasets for LLM Training](https://discord.com/channels/1089876418936180786/1331335526338265108) 的论文在 ArXiv 上发表，详细介绍了开源 AI 数据集面临的挑战，并为**公平性**和**透明度**提供了建议。
   - 社区成员称赞了该蓝图在**提供公平竞争环境**方面的潜力，强调更强大的开源数据生态系统将推动 LLM 的进步。
- **Mozilla 与 EleutherAI 宣布数据治理峰会**：Mozilla 与 **[EleutherAI](https://discord.gg/cJQKYFDwHV)** 合作举办了一场数据集召集会议，重点关注开源数据的负责任管理和治理。
   - 关键利益相关者讨论了最佳策展实践，强调了通过**协作式**社区参与推进 LLM 开发的共同目标。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **AI 在网络安全领域从炒作转向实战**：一位成员回忆起 **AI** 曾只是网络安全领域的一个*流行词*，并提到他们在一年前进入该领域。
   - 他们对 **AI** 深度集成到安全流程中感到兴奋，憧憬着实时威胁检测和自动化事件响应。
- **安全团队拥抱 AI 辅助**：讨论强调了人们对 **AI** 如何增强安全团队能力（特别是在处理复杂告警方面）日益增长的兴趣。
   - 爱好者们期待 **AI** 提供更敏锐的分析工具，使分析师能够专注于关键任务并减少手动开销。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。

---

**Axolotl AI Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。

---

**OpenInterpreter Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。

---

**HuggingFace Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道详细摘要与链接

{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1330991993865896169)** (652 条消息 🔥🔥🔥): 

> `DeepSeek-R1 模型限制, 分类任务的微调策略, 处理模型检查点 (checkpoints), 模型 Tokenization 与 Embeddings, 使用 Unsloth notebooks 的挑战`

- **关于 DeepSeek-R1 模型限制的讨论**：用户注意到 DeepSeek-R1 模型的最大 token 长度被限制在 16,384，尽管根据其 embeddings 计算，该值应为 163,840。
   - 这种差异引发了关于模型部署过程中可能存在 bug 或错误的猜测。
- **模型微调策略**：一位用户询问指令微调 (IFT) 是否应该从对话预训练 (CPT) 的最后一个 checkpoint 开始，并指出其 notebook 中缺少必要的代码。
   - 经澄清，IFT 应该从与其任务相关的 checkpoint 开始，而不是最近的 CPT checkpoint。
- **Unsloth 中的模型转换问题**：一位新用户报告在成功完成初始训练后，将 Phi4 文件转换为 GGUF 格式时遇到困难，并收到了模糊的错误消息。
   - 建议指出，如果训练了 heads 或 embeddings，则需要合并 tokenizer，这可能是转换问题的潜在根源。
- **Tokenization 和 Embeddings 的影响**：讨论集中在 Llama 3.2 等模型中 embeddings 绑定权重 (tied weights) 的重要性，这可能会影响模型性能和 context length 能力。
   - 用户反思了这些配置对小模型及其效率的潜在影响。
- **模型输出生成的实验**：探索了通过 beam search 和最小困惑度分支选择 (minimum perplexity branch selection) 等方法提高输出质量的策略。
   - 参与者讨论了多轮推理 (multi-turn reasoning) 和增加评审模型 (judging models) 以改进输出决策的优点。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/fimbulvntr/status/1881821582571761920">来自 Fimbul (@fimbulvntr) 的推文</a>：是我疯了还是 DeepSeek-R1 的 model_max_length 被限制在了 16384？我认为这是一个 bug。实际上它应该是 163840。它的 original_max_position_embeddings=4096 且 RoPE 因子为 40... 4...</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb">Google Colab</a>：未找到描述</li><li><a href="https://huggingface.co/JingzeShi/Doge-20M-Instruct">JingzeShi/Doge-20M-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/so-cute-cat-love-head-pat-gif-14623443">So Cute Cat GIF - So Cute Cat Love - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/meta-llama/Llama-3.2-1B/tree/main?show_file_info=model.safetensors">meta-llama/Llama-3.2-1B at main</a>：未找到描述</li><li><a href="https://colab.research.google.com">Google Colab</a>：未找到描述</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/3302ba78c0090838341caf8adfbe1e231308fa95/tokenizer_config.json#L22">tokenizer_config.json · deepseek-ai/DeepSeek-R1 at 3302ba78c0090838341caf8adfbe1e231308fa95</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/Llama-3.2-1B/tree/main?show_file_info=model.safetensors">unsloth/Llama-3.2-1B at main</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=LPZh9BOjkQs&list=LL&index=2&pp=gAQBiAQB8AUB">大语言模型简要解释</a>：在此深入了解：https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi 技术细节讲座：https://youtu.be/KJtZARuO3JY 这是...</li><li><a href="https://gist.github.com/sebaxakerhtc/5e7faa4ead6e2f4e0ea69634c3f624ba">Unsloth 指导脚本</a>：Unsloth 指导脚本。GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://huggingface.co/blog/llama32?utm_source=chatgpt.com#what-is-special-about-llama-32-1b-and-3b">Llama 现在可以视觉识别并在你的设备上运行 - 欢迎 Llama 3.2</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=kCc8FmEb1nY">让我们从头开始构建 GPT：代码实现，详细讲解。</a>：我们按照论文 "Attention is All You Need" 以及 OpenAI 的 GPT-2 / GPT-3 构建了一个生成式预训练 Transformer (GPT)。我们讨论了连接到...</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF">unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/bagel-org/ZKLoRA">GitHub - bagel-org/ZKLoRA: 用于 LoRA 验证的高效零知识证明</a>：用于 LoRA 验证的高效零知识证明 - bagel-org/ZKLoRA</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py#L766">unsloth/unsloth/chat_templates.py at main · unslothai/unsloth</a>：微调 Llama 3.3, Mistral, Phi-4, Qwen 2.5 &amp; Gemma LLMs 速度提升 2-5 倍，显存占用减少 70% - unslothai/unsloth
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1331361373250129952)** (1 条消息): 

> `Unsloth training, Fine-tuning LLMs, Weights & Biases integration, vLLM for model serving` 


- **Gautam 的 Unsloth 微调指南**: Gautam Chutani 在 Medium 上发表的一篇文章讨论了使用 **LoRA** **微调 LLaMA 3**，特别关注了用于监控的 [Weights & Biases](https://gautam75.medium.com/fine-tuning-llama-3-1-8b-for-function-calling-using-lora-159b9ee66060) 集成以及用于模型服务的 **vLLM**。
   - *微调提供了一种针对专门任务优化预训练模型的方法*，但由于**计算资源限制**而面临挑战。
- **微调 LLMs 的挑战**: 文章强调，**微调大语言模型 (LLMs)** 对于使其适应特定任务至关重要，但由于涉及的**计算资源需求**，这带来了挑战。
   - 传统的微调方法需要大量的 **GPU memory** 和计算时间，这可能成为许多从业者的障碍。



**提及的链接**: <a href="https://gautam75.medium.com/fine-tuning-llama-3-1-8b-for-function-calling-using-lora-159b9ee66060">Fine-Tuning Llama-3.1-8B for Function Calling using LoRA</a>: 利用 Unsloth 进行微调，并集成 Weights &amp; Biases 进行监控，使用 vLLM 进行模型服务。

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1331083511221719142)** (56 条消息🔥🔥): 

> `Fine-tuning models, Using Unsloth with different datasets, Models compatibility, Training on reasoning tasks, Handling CUDA memory issues` 


- **使用 Unsloth 的通用微调指南**: 用户讨论了使用 **Unsloth** 微调 **Llama** 和 **Phi-4** 等模型的过程，强调了结合数据集以获得更好性能的重要性。
   - 一位用户提到，与训练后调整相比，使用指令数据微调模型能显著增强训练效果。
- **模型兼容性与混合数据集**: 会议澄清了在使用 Unsloth 时，基于受支持框架（如 **Mistral**）的模型也是兼容的，且数据集不需要采用相同的格式。
   - 用户分享了混合和格式化数据集的策略，并建议对不同数据集格式进行分区和转换。
- **模型输出问题**: 针对 **Phi-4** 模型输出持续相似的问题，用户建议采用在输入前添加 seed 值等技术来增加结果的多样性。
   - 一位用户分享了使用特定 notebook 进行 Phi-4 对话训练的经验，遇到了诸如无法转换保存文件等问题。
- **CUDA 内存管理技巧**: 针对 CUDA 显存溢出（out-of-memory）错误，建议通过减小训练的 batch size 作为解决方案，同时保留某些固定参数如 **r=128**。
   - 参与者分享了关于内存管理以及在各种硬件设置上进行微调的最佳配置见解。
- **API 利用与设置挑战**: 用户询问了在资源有限的机器（如 Mac）上本地运行模型的问题，并建议利用 API 以实现更轻松的集成。
   - 针对模型实现挑战提供了说明，包括现有设置下的量化限制。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/d">Unsloth Documentation</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/">Welcome | Unsloth Documentation</a>: 初识 Unsloth？</li><li><a href="https://docs.unsloth.ai/basics/datasets-101#multiple-datasets">Datasets 101 | Unsloth Documentation</a>: 学习创建微调数据集的所有要点！</li><li><a href="https://docs.unsloth.ai/basics/datasets-101#formatting-our-data">Datasets 101 | Unsloth Documentation</a>: 学习创建微调数据集的所有要点！</li><li><a href="https://github.com/microsoft/Phi-3CookBook/blob/main/code%2F04.Finetuning%2FPhi-3-finetune-lora-python.ipynb">Phi-3CookBook/code/04.Finetuning/Phi-3-finetune-lora-python.ipynb at main · microsoft/Phi-3CookBook</a>: 这是一个用于入门 Phi 系列模型的 Phi 家族 SLMs 指南。Phi 是由 Microsoft 开发的开源 AI 模型系列。Phi 模型是目前最强大且最具成本效益的小语言模型...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1331004403045367879)** (12 条消息🔥): 

> `OpenWebUI 集成, Synthetic datasets, 免费/开源解决方案, Colab 脚本测试` 


- **OpenWebUI 及更多未来计划**：*Invisietch* 提到最终计划与 **OpenWebUI**、**Ollama**、**Flowise** 和 **LocalAI** 集成，目前正通过 **Kobold API** 与 **Kobold** 和 **Aphrodite** 协作。
   - 指出待办事项很多，但现有工具的开发正在取得进展。
- **关于免费/开源解决方案的讨论**：*Sebaxakerhtc* 强调在工作中仅使用**免费/开源解决方案**，*invisietch* 对此澄清说 **Koboldcpp** 和 **Aphrodite** 确实都是免费软件。
   - *Invisietch* 提到，一旦添加了许可证文件，名为 **Chatterbox** 的项目也将作为免费软件提供。
- **Synthetic Datasets 自动化**：*Invisietch* 提出了关于**自动创建 Synthetic datasets** 的问题，并建议命令行界面（CLI）可能对批量操作有益。
   - 讨论表明重点在于利用相同的后端 API 来实现此功能。
- **Colab 脚本测试成功**：*Sebaxakerhtc* 在 **Google Colab** 中成功测试了一个脚本，实现了从零到保存 **GGUF** 的全过程，且输出已保存以供查看。
   - 这获得了积极的反响，另一位用户评价这一成就“非常酷”。



**提到的链接**：<a href="https://colab.research.google.com/drive/1NwVnNtj-_o6vTUUsgM5BMAVRPUpTZQU2?usp=sharing">Google Colab</a>：未找到描述

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1331320261567451228)** (169 messages🔥🔥): 

> `Chinchilla 最优训练, AI 训练中的合成数据, AI 中的情感追踪, 语言模型中的 Grokking, AI 应用中的 3D 建模 vs 文本` 


- **理解 Chinchilla 最优训练**：**Chinchilla** 论文建议在模型大小和训练 Token 数量之间取得平衡以获得最佳性能，指出两者应等比例缩放以避免效率低下。这一概念对于确定大型语言模型（LLM）达到最佳效果所需的训练数据量至关重要。
   - *知识不会像“专家”那样聚集，因此 Chinchilla 最优适用于总参数，而不仅仅是子集。*
- **关于 AI 训练合成数据的讨论**：成员们讨论了使用**合成数据流**创建更符合评估合规性的训练数据集的潜力。这可能导致一个更紧密的训练/测试循环，根据模型性能动态调整，从而避免过拟合。
   - 有人对合成数据的局限性表示担忧，特别是其与现实世界应用的相关性，并指出并非所有领域都能享受到无限合成数据的便利。
- **AI 系统中的情感追踪**：一位成员分享了他们在**成人产业**的工作，强调了机器人如何通过情感追踪和心理学原理准确模拟人类行为。这包括集成中间件以在实时语境中管理状态。
   - 该方法强调情感追踪是基于既定的心理学框架，而不是仅仅依赖 LLM 内部的能力。
- **AI 模型中的 Grokking 与缩放**：讨论了 **grokking**（模型深度理解某个领域的能力）的概念，重点关注训练数据的组织对实现这一目标的关键作用。建议将训练从简单任务到复杂任务进行分层，以最大化跨不同抽象层级的理解。
   - 成员们建议，针对基础推理进行优化可能有助于在未来模型中实现 100 倍的压缩改进，从而使参数显著减少的实际应用成为可能。
- **关于在 AI 应用中使用 3D 模型的辩论**：对话涉及了 3D 模型在当前 AI 应用中是否实用或相关，重点在于聊天和语音交互更具盈利性。虽然有人指出 AI 的进步允许 3D 生成，但共识倾向于成熟的文本和语音应用能产生更好的回报。
   - 参与者承认了行业内对技术采用的不同看法，特别是关于在成人 AI 领域什么是产生收入的关键。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2412.13148">SWAN: SGD with Normalization and Whitening Enables Stateless LLM Training</a>：Adam (Kingma &amp; Ba, 2015) 等自适应优化器一直是大型语言模型成功的核心。然而，它们通常需要在整个训练过程中维护优化器状态，这……</li><li><a href="https://arxiv.org/abs/2305.07759">TinyStories: How Small Can Language Models Be and Still Speak Coherent English?</a>：语言模型 (LMs) 是自然语言处理的强大工具，但当它们规模较小时，往往难以产生连贯且流畅的文本。具有约 125M 参数的模型，如 GP……</li><li><a href="https://paperswithcode.com/method/chinchilla">Papers with Code - Chinchilla Explained</a>：Chinchilla 是一个拥有 70B 参数的模型，作为计算最优模型使用 1.4 万亿 Token 进行训练。研究结果表明，这类模型通过等比例缩放模型大小和……
</li>
</ul>

</div>
  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1331000801463762965)** (467 messages🔥🔥🔥): 

> `DeepSeek R1 集成, Cursor 0.45 更新, OpenAI Stargate 项目, AI 竞争, Claude 3.5 性能`

- **DeepSeek R1 可以添加到 Cursor**：用户已发现通过 OpenRouter 将 DeepSeek R1 集成到 Cursor 的方法，尽管目前的集成效果较差，且限制了对其他模型的访问。
   - 建议等待正式集成，许多人表示目前倾向于在不使用 Cursor 的情况下使用 R1。
- **Cursor 0.45 更新持续回滚**：Cursor 的最新更新（包括 0.45.1 版本）由于代码库索引和模型兼容性相关问题已多次回滚。
   - 用户在更新过程中遇到不一致的情况，经常需要恢复到早期版本。
- **OpenAI 宣布 Stargate Project**：OpenAI 宣布了一项名为 Stargate Project 的 5000 亿美元投资计划，旨在利用 SoftBank 等机构的资金在美国建设新的 AI 基础设施。
   - 这一公告引发了关于快速演变的 AI 竞争的讨论，特别是考虑到来自日本的巨额投资。
- **DeepSeek 加剧 AI 竞争**：DeepSeek R1 的出现引发了关于同类模型性能提升的讨论，表明 AI 能力正变得更加普及。
   - DeepSeek R1 与 Claude 3.5 之间的对比凸显了 AI 开发领域的竞争态势。
- **Claude 3.5 在竞争中表现出色**：Claude 3.5 的性能被注意到有显著提升，用户对其速度和准确性给予了评价，这可能源于 DeepSeek R1 带来的竞争压力。
   - Anthropic 最近缺乏更新，引发了人们对该公司在竞争环境下未来策略的好奇。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/paulgauthier/status/1881428345973608901">来自 Paul Gauthier (@paulgauthier) 的推文</a>：DeepSeek R1 在 aider 多语言基准测试中获得 57%，排名第二，仅次于 o1：62% o1 (high)，57% DeepSeek R1，52% Sonnet，48% DeepSeek Chat V3。完整排行榜：https://aider.chat/docs/leaderboards/</li><li><a href="https://x.com/kimmonismus/status/1881734307158397442">来自 Chubby♨️ (@kimmonismus) 的推文</a>：Dario Amodei 表示：“我从未像现在这样确信我们正接近强大的 AI 系统。我在 Anthropic 内部以及过去几个月所见到的情况让我...”</li><li><a href="https://x.com/kregenrek/status/1878487131099898269?s=46">来自 Kevin Kern (@kregenrek) 的推文</a>：为开发者介绍 Codefetch。只需一个简单的终端命令，即可将代码转换为适用于 LLM 的 Markdown。在 bolt .new, Cursor 和许多其他 AI 编程工具中使用它。→ 与你的代码库聊天 → 节省 tokens → ...</li><li><a href="https://x.com/OpenAI/status/1881830103858172059?s=19">来自 OpenAI (@OpenAI) 的推文</a>：宣布 Stargate 项目。Stargate 项目是一家新公司，计划在未来四年投资 5000 亿美元，为 OpenAI 在美国建设新的 AI 基础设施。我们将...</li><li><a href="https://www.cursor.com/downloads">Cursor - AI 代码编辑器</a>：旨在让你拥有非凡的生产力，Cursor 是使用 AI 编写代码的最佳方式。</li><li><a href="https://www.cursor.com/blog/shadow-workspace">使用 Shadow Workspaces 进行迭代 | Cursor - AI 代码编辑器</a>：隐藏窗口和内核级文件夹代理，让 AI 在不影响用户的情况下迭代代码。</li><li><a href="https://tenor.com/view/crazy-alert-crazy-alert-alerta-alerta-loca-gif-2463480864319782005">Crazy Alert Crazy GIF - Crazy Alert Crazy Alert - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/bg3EF.gif">Trust Me No One Is Going To Notice Grey Griffin GIF - 相信我，没人会注意到 Grey Griffin Karen Crawford - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1">DeepSeek R1 - API, 提供商, 统计数据</a>：DeepSeek-R1 来了！⚡ 性能与 OpenAI-o1 相当 📖 完全开源的模型和技术报告 🏆 MIT 许可：自由 Distill 和商业化！使用 API 运行 DeepSeek R1</li><li><a href="https://status.cursor.com/">Cursor 状态</a>：未找到描述</li><li><a href="https://status.cursor.com/?utm_source=embed">Cursor 状态</a>：未找到描述</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Zero">deepseek-ai/DeepSeek-R1-Zero · Hugging Face</a>：未找到描述</li><li><a href="https://forum.cursor.com/t/how-to-add-a-custom-model-like-deepseek-v3-which-is-openai-compatible/37423/22?u=irian-codes">如何添加像 DeepSeek-V3 这样兼容 OpenAI 的自定义模型</a>：这是我关于如何将 DeepSeek 添加到 Cursor 的帖子。它在聊天和补全中表现出色。但 Composer 不允许使用外部模型。我们应该推动 Cursor 团队添加此功能。更新：你...</li><li><a href="https://forum.cursor.com/t/please-add-deepseek-r1-model/42868">请添加 DeepSeek R1 模型</a>：显然比 Sonnet 更好且便宜得多？拭目以待……</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1">GitHub - deepseek-ai/DeepSeek-R1</a>：通过在 GitHub 上创建账号来为 deepseek-ai/DeepSeek-R1 的开发做出贡献。</li><li><a href="https://downloader.cursor.sh/mac/dmg/arm64">未找到标题</a>：未找到描述</li><li><a href="https://api-docs.deepseek.com/guides/reasoning_model">推理模型 (deepseek-reasoner) | DeepSeek API 文档</a>：deepseek-reasoner 是 DeepSeek 开发的推理模型。在交付最终答案之前，模型首先生成思维链 (CoT) 以增强其响应的准确性。我们的 API 提供...</li><li><a href="https://downloader.cursor.sh/linux/appimage">未找到标题</a>：未找到描述
</li>
</ul>

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1331003996479164468)** (49 条消息🔥): 

> `Windsurf 性能问题，DeepSeek 模型对比，错误排查，Codeium 功能与请求，工具用户体验` 


- **用户报告 Windsurf 存在性能滞后**：多位用户在 Windsurf 中输入提示词时遇到延迟，引发了关于潜在修复方案和支持的讨论。
   - 一名成员提到尝试通过调整设置来缓解滞后，但目前尚未收到明确的解决方案。
- **DeepSeek R1 模型表现优于以往模型**：成员们强调，据报告 DeepSeek R1 模型的性能指标超过了 OpenAI O1-preview，这激发了将其集成到 Codeium 的兴趣。
   - 尽管令人兴奋，但也有人对其有效处理 tool calls 的能力表示担忧，使得目前的集成前景尚不明朗。
- **错误消息与排查方案流传**：多位用户分享了在 Windsurf 中遇到的错误经历，包括 “incomplete envelope: unexpected EOF” 等消息。
   - 社区成员正在讨论各种解决方案，以及解决这些问题可能需要的系统权限调整。
- **对 Codeium 功能和改进的请求**：一位用户敦促 Codeium 团队添加 DeepSeek R1 模型，并表达了对微调（fine-tuning）机会的期待。
   - 其他人则对 JetBrains IDE 用户缺乏更新或改进表示担忧，认为与 Windsurf 用户相比，他们的优先级较低。
- **对 Codeium 功能和支持的评价褒贬不一**：用户对 Codeium 的支持和功能可用性表达了复杂的感受，并将其与 Co-pilot 等其他工具进行了对比。
   - 对于购买额度（credits）的困难以及缺乏明确的支持渠道沟通，用户表现出明显的挫败感，反映出对更好客户服务的需求。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/TheXeophon/status/1881442133376454694?t=kcwBO9GpmTX5zzXVtA63gA&s=19">Xeophon (@TheXeophon) 的推文</a>: 我的天，R1 在我的基准测试中击败了 o1-preview</li><li><a href="https://x.com/TheXeophon/status/1881443117787984265?t=CWcMfDus2ULxJQS6VnnQRA&s=19">Xeophon (@TheXeophon) 的推文</a>: 我被个人基准测试中的 R1 震惊了。这是完整的评估集，它完全碾压了竞争对手，自成一派，甚至超过了 o1-preview（图中省略了 ...</li><li><a href="https://www.reddit.com/r/synology/comments/pq0411/cant_mount_network_drive_in_windows_explorer/">Reddit - 深入了解一切</a>: 未找到描述</li><li><a href="https://github.com/Exafunction/codeium/releases/tag/termium-v0.2.1">Release termium-v0.2.1 · Exafunction/codeium</a>: 自动发布</li><li><a href="https://codeium.com/blog/termium-codeium-in-terminal-launch">Termium: 终端中的 Codeium</a>: 为您的终端命令提供 AI 驱动的自动补全。
</li>
</ul>

</div>
  

---

### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1330995751144394805)** (351 条消息🔥🔥): 

> `Windsurf 性能问题、DeepSeek 集成、Flow Actions 限制、建议质量、Bug 报告与故障排除` 


- **Windsurf 遭遇性能退步**：许多用户报告了 Windsurf 的性能问题，特别提到了 Prompt 过程中的延迟以及对代码进行的不必要修改。
   - 最近的更新引入了一些让用户感到沮丧的 Bug，导致部分用户考虑转向 Cursor 等替代方案。
- **集成 DeepSeek 模型**：用户询问将 DeepMind 或 DeepSeek 模型整合进 Windsurf 的可能性，并建议使用兼容的 API 来实现这一点。
   - 一些人建议利用像 Cline 这样兼容的插件来增强功能。
- **Flow Actions 限制带来的挑战**：用户对 Flow Actions 的限制表示担忧，指出这造成了生产力瓶颈，并建议需要制定策略来缓解这一问题。
   - 一些用户分享了关于如何更有效地管理这些限制的见解。
- **用户讨论建议质量**：反馈显示用户对 Windsurf 提供的建议质量感到不满，相比之下， Cursor 等竞争对手提供的修改更具针对性。
   - 讨论涉及 Windsurf 的算法是否能够像现有的替代方案那样高效运行。
- **报告 Bug 与故障排除**：几位用户正面临持续存在的 Bug 和问题，敦促其他人提交支持工单并附带诊断日志，以便进行彻底的故障排除。
   - 用户继续讨论与 Bug 报告相关的各种变通方法和经验。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://codeium.com/support">支持 | Windsurf 编辑器和 Codeium 扩展</a>：需要帮助？联系我们的支持团队获取个性化协助。</li><li><a href="https://cloud.dwavesys.com/leap/">D-Wave Leap 登录 | D-Wave Leap™</a>：未找到描述</li><li><a href="https://docs.codeium.com/windsurf/web-search">Web Search - Codeium 文档</a>：未找到描述</li><li><a href="https://chat.deepseek.com/downloads/DeepSeek%20Privacy%20Policy.html">DeepSeek 隐私政策</a>：未找到描述</li><li><a href="https://tenor.com/view/ninja-fortnite-reaction-ninja-low-taper-fade-gif-1784137995500051652">Ninja Fortnite GIF - Ninja Fortnite 反应 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://docs.basedpyright.com/latest/installation/pre-commit%20hook/">pre-commit hook - basedpyright</a>：未找到描述</li><li><a href="https://www.reddit.com/media?url=https%3A%2F%2Fi.redd.it%2Fjf6vo05hx8ee1.jpeg">https://i.redd.it/jf6vo05hx8ee1.jpeg</a>：未找到描述</li><li><a href="https://codeium.canny.io/feature-requests/p/add-rename-suggestion-like-in-vscode">在 VScode Copilot 中添加重命名建议 | 功能请求 | Codeium</a>：在使用 alt+r 时，使用 codeium/cascade 建议重命名选项</li><li><a href="https://www.reddit.com/r/Codeium/comments/1i5ftc9/heres_a_balanced_critique_of_windsurfs_business/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://status.codeium.com/#">Codeium 状态</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=moIySJ4d0UY">Web Search 最佳实践：节省额度并优化工作流 - Windsurf 编辑器</a>：准备好充分利用 Windsurf 全新的 Web Search 功能了吗？这次深度探讨将帮助你释放其全部潜力！在本视频中，你将学习...</li><li><a href="https://github.com/microsoft/pyright/blob/main/docs/mypy-comparison.md">pyright/docs/mypy-comparison.md (main 分支) · microsoft/pyright</a>：Python 静态类型检查器。通过在 GitHub 上创建账户为 microsoft/pyright 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1i5q6b9/deepseekr1_and_distilled_benchmarks_color_coded/">Reddit - 深入探索一切</a>：未找到描述
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1331027328418975885)** (1 条消息): 

> `Aider v0.72.0 发布，支持 DeepSeek R1，支持 Kotlin 语法，文件处理增强，Bug 修复与改进` 


- **Aider v0.72.0 发布，带来令人兴奋的功能**：**Aider v0.72.0** 的发布包含了对 DeepSeek R1 的支持，可以通过快捷方式 `--model r1` 或通过 OpenRouter 访问。
   - 在此更新中，**Aider** 贡献了 **52%** 的代码，显示了显著的内部开发成果。
- **增强的文件处理和语法支持**：repo map 中增加了对 **Kotlin 语法** 的支持，并新增了 `--line-endings` 选项以改进文件写入。
   - 此外，针对 GPT-4o 模型的 `examples_as_sys_msg=True` 提升了 benchmark 分数。
- **Bug 修复解决了常见问题**：此版本解决了多个 Bug，包括 Docker 镜像中的**权限问题**，以及针对 **unicode 错误** 的 ASCII 回退方案。
   - 另一个显著的修复改进了 repomap 计算中列表切片的整数索引。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1330990775189901342)** (297 条消息🔥🔥): 

> `DeepSeek R1 性能，AI 模型对比，OpenAI 订阅讨论，硬件价格与可用性，数据使用与隐私担忧` 


- **DeepSeek R1 给用户留下深刻印象**：许多用户对 **DeepSeek R1** 表示满意，注意到它在各种任务中的表现以及有效处理多种语言的能力。
   - 一些人提到，与其他模型相比，它可能更适合特定的编程任务，并强调了其独特的能力。
- **AI 模型输出的差异**：用户观察到 **Sonnet** 和 **DeepSeek** 等模型之间的性能水平不一致，有报告称输出质量因地理位置而异。
   - 对话强调了欧洲和美国性能之间的差异，鼓励用户针对特定应用考虑不同的模型。
- **OpenAI 订阅反思**：几位成员讨论了他们使用 OpenAI 订阅服务的经验，包括最近的退款和价格对比。
   - 普遍看法是 **DeepSeek** 提供了良好的性价比，一些成员表示由于成本效益，有兴趣从 **Claude** 转向 **DeepSeek R1**。
- **欧洲的硬件价格**：几位用户分享了对欧洲 GPU 价格的见解，注意到 **RTX 3060** 和 **3090** 等旧型号的价格出奇地低。
   - 尽管新一代 GPU 崛起，用户仍在考虑以折扣价从欧洲卖家处购买旧型号。
- **对 AI 数据使用的担忧**：关于使用 AI 模型影响的讨论集中在数据隐私和所有权上，用户在思考平台将如何利用他们的代码。
   - 成员们普遍对数据使用持轻松态度，认为他们的大部分代码不具备足够的专有性，无需担心。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/Kimi_ai_/status/1881332472748851259">来自 Kimi.ai (@Kimi_ai_) 的推文</a>：🚀 隆重推出 Kimi k1.5 --- 一款 o1 级别的多模态模型。其短 CoT 性能达到 SOTA 水平，在 📐AIME、📐MATH-500、💻 LiveCodeBench 上大幅超越 GPT-4o 和 Claude Sonnet 3.5（最高达 +550%...</li><li><a href="https://x.com/0xluffyb/status/1881323971897110866">来自 luffy (@0xluffyb) 的推文</a>：今天的每个人。引用 DeepSeek (@deepseek_ai) 🚀 DeepSeek-R1 发布了！⚡ 性能与 OpenAI-o1 旗鼓相当 📖 完全开源的模型和技术报告 🏆 MIT 许可证：可自由 Distill 和商业化！🌐 ...</li><li><a href="https://docs.fireworks.ai/guides/security_compliance/data_handling#data-privacy-and-security)">数据隐私与安全 - Fireworks AI 文档</a>：未找到描述</li><li><a href="https://br.ign.com/tech/135086/news/ceo-da-openai-nao-sabe-o-que-fazer-com-o-comportamento-dos-assinantes-do-chatgpt">OpenAI CEO 不知道该如何应对 ChatGPT 订阅者的行为</a>：他在没有深思熟虑的情况下选择了价格，并认为自己能赚到钱</li><li><a href="https://aider.chat/docs/usage/not-code.html">编辑配置和文本文件</a>：编辑配置文件、文档和其他基于文本的格式。</li><li><a href="https://tenor.com/view/megatron-upgrade-unicron-behold-galvatron-gif-26590123">Megatron 升级 GIF - Megatron 升级 Unicron - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://unsloth.ai/blog/deepseek-r1">运行 DeepSeek-R1 / R1 Zero</a>：DeepSeek 最新的 R-1 模型是目前最强大的开源推理模型，性能与 OpenAI 的 o1 模型相当。了解如何运行和微调该模型。</li><li><a href="https://tenor.com/view/bear-embarrassed-smiling-gif-11674756">害羞的小熊 GIF - 害羞微笑的小熊 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://docs.fireworks.ai/guides/security_comp">简介 - Fireworks AI 文档</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1i64up9/model_comparision_in_advent_of_code_2024/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://gist.github.com/murdockq/b08f72699fd7d8db556a14e69a7cb0c3">a game prompt.md</a>：GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://github.com/matatonic/openedai-whisper">GitHub - matatonic/openedai-whisper：一个兼容 OpenAI API 的语音转文本服务器，用于音频转录和翻译，又名 Whisper。</a>：一个兼容 OpenAI API 的语音转文本服务器，用于音频转录和翻译，又名 Whisper。 - matatonic/openedai-whisper</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1?tab=readme-ov-file#distilled-model-evaluation">GitHub - deepseek-ai/DeepSeek-R1</a>：通过在 GitHub 上创建账号，为 deepseek-ai/DeepSeek-R1 的开发做出贡献。</li><li><a href="https://github.com/Devographics/surveys/issues/278">State of AI 2025 预览 · Issue #278 · Devographics/surveys</a>：这是即将发布的 State of Web Dev AI 2025 调查的预览链接，这是该项新调查的首届版本：https://survey.devographics.com/en-US/survey/state-of-ai/2025 我很想得到...</li><li><a href="https://api.ailocal.org">Whisper.cpp 服务器</a>：未找到描述</li><li><a href="https://github.com/Aider-AI/aider/issues/429">Tree-sitter tsx 解析器有时会挂起，导致 aider 挂起 · Issue #429 · Aider-AI/aider</a>：用户报告在包含大量 .tsx 文件的仓库中使用 aider 时会挂起。使用 --no-git 可以消除挂起。问题似乎出在 repo map 代码中。https://discord.com/channels/1131200896827654144/1192136795...</li><li><a href="https://endpoints.huggingface.co/catalog">推理目录 | Hugging Face 推理端点</a>：未找到描述
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1330994845317070898)** (89 messages🔥🔥): 

> `Using Aider with Sonnet, Updating Aider Versions, Error Handling in Aider, DeepSeek Model Comparisons, Refactoring Python Codebases` 


- **在 Sonnet 的 Context Window 中使用 Aider**：用户对在 Anthropic 平台投入 **400 美元**之前无法访问 **Sonnet 的完整 context window** 表示担忧，业余爱好者可能会觉得这笔费用过高。
   - 这引发了关于高级 AI 工具对于普通开发者的可访问性和负担能力的讨论。
- **Aider 更新难题**：几位成员表示在更新 Aider 时遇到困难，特别是从 **0.70.0 迁移到最新版本**时，有些人不清楚该使用哪些命令。
   - 常见的解决方案包括使用 `aider --upgrade` 命令或直接重新安装，尽管成功率各不相同。
- **错误处理：API Keys 和配置**：出现了 **invalid API keys** 的问题，引发了关于 **.env 配置**如何覆盖 **.conf 文件**设置并影响项目使用的讨论。
   - 一位成员表示，从 **.env 文件**中删除已禁用的 key 解决了他们的问题，这说明了配置管理的重要性。
- **比较 DeepSeek 模型**：有成员就 **DeepSeek-R1 vs. DeepSeek-V3** 在架构模式和使用配置方面的性能提出了疑问。
   - 成员们推测了缓存（caching）在 DeepSeek 效率中的作用，并询问了如何通过 cache-prompts 设置将其与 Aider 集成。
- **重构 Python 代码库**：围绕重构一个 **12 文件的 Python 代码库**展开了策略讨论，建议使用 **Gemini Pro** 等工具来高效管理大上下文（large contexts）。
   - 参与者指出，虽然增量更改有所帮助，但优化该过程的准确性和效率仍是一项正在进行的工作。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/troubleshooting/edit-errors.html">File editing problems</a>：aider 是你终端里的 AI 配对编程工具</li><li><a href="https://engineering.fb.com/2024/12/19/developer-tools/glean-open-source-code-indexing/">Indexing code at scale with Glean</a>：我们正在分享关于 Glean 的细节，这是 Meta 用于收集、推导和处理源代码事实的开源系统。在这篇博文中，我们将讨论为什么像 Glean 这样的系统很重要……</li><li><a href="https://github.com/BerriAI/litellm/issues/7877">[Feature]: DeepSeek-R1 support · Issue #7877 · BerriAI/litellm</a>：Feature DeepSeek-R1 API 在 reasoning_content 参数中返回其思考过程。目前 LiteLLM 忽略了这一点。他们的 API 方法是为长文本返回 &quot;reasoning_content&quot;...</li><li><a href="https://github.com/getgrit/gritql">GitHub - getgrit/gritql: GritQL is a query language for searching, linting, and modifying code.</a>：GritQL 是一种用于搜索、lint 和修改代码的查询语言。</li><li><a href="https://github.com/jbellis/llmap">GitHub - jbellis/llmap</a>：为 jbellis/llmap 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1331190132535197697)** (1 messages): 

> `Deepseek R1, Live coding experience, Space Invaders game upgrade` 


- **Deepseek R1 在 Architect 模式下表现出色**：在一个简短的现场编程视频中，用户展示了在升级一款 **Space Invaders** 类游戏时，使用 Architect 模式下的 **Deepseek R1**，并强调了其特性。
   - 该视频题为 [Space Invaders with Deepseek R1 and Aider in Architect mode](https://youtu.be/njJhjUgBTZg)，强调 **R1** 是顶级竞争者，在 Aider LLM 排行榜上仅次于 **OpenAI 的 o1**。
- **Deepseek R1 vs OpenAI's o1**：用户指出 **Deepseek R1** 几乎与 **OpenAI 的 o1** 一样强大，但成本显著降低。
   - 这一对比突显了 **Deepseek R1** 在编程环境 AI 应用中日益增长的潜力。



**提到的链接**：<a href="https://youtu.be/njJhjUgBTZg">Space Invaders with Deepseek R1 and Aider in Architect mode.</a>：来自 Deepseek 的新 R1 模型在 Aider LLM 排行榜上仅次于 OpenAI 的 o1。此外，它的成本仅为后者的一小部分。在这里我测试了它的能力...

  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1330995639726637139)** (342 条消息🔥🔥): 

> `DeepSeek R1 Models, Mathematics Tutoring, Local Model Deployment, OpenAI Compatibility, Community Support for AI` 


- **DeepSeek R1 在数学辅导中的表现**：用户报告了使用 DeepSeek R1 的积极体验，特别是在数学教学方面，一位用户提到了它在解决复杂问题和提供逐步推理方面的有效性。
   - 该模型因其作为导师的能力而受到称赞，为有特殊教育需求的用户提供了相当大的支持。
- **探索本地使用的模型选项**：几位用户讨论了他们运行不同模型的硬件配置；一位用户提到使用 4090 GPU 和 64GB RAM 来支持繁重的计算。
   - 讨论还包括使用家用服务器访问强大的 AI 功能，以及使用自定义客户端进行交互的想法。
- **社区与 AI 资源获取**：讨论了当地社区学院提供 DeepSeek 等 AI 辅导工具访问权限的可能性，这可能会使需要额外支持的学生受益。
   - 用户表达了对社区支持的渴望，希望让这些技术在教育用途上更易于获取。
- **OpenAI API 与客户端开发**：用户谈到了创建自定义客户端以连接他们的模型，并质疑与 OpenAI API 的兼容性，强调了对某些端点缺乏支持的问题。
   - 一位用户分享了编写 HTML 客户端连接到其服务器的经验，暗示了理解语法对有效交互的重要性。
- **量化与模型选择**：一位用户询问了量化数值（Q3、Q4 等）的意义，以及它们如何影响模型性能和准确性。
   - 有人指出，较低的量化可能会导致更快的响应速度，但可能会牺牲一些准确性，强调了根据用户需求进行实验的必要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/seo_leaders/status/1881462202831614085">来自 Andrew C (@seo_leaders) 的推文</a>：DeepSeek R1 671B 在 2 台 M2 Ultra 上运行，速度快于阅读速度。几乎是在家用消费级硬件上运行的开源 O1。配合 mlx.distributed 和 mlx-lm，采用 3-bit 量化（约 4 bpw）。模型正在...</li><li><a href="https://www.audacityteam.org/download/openvino/">下载 Audacity AI 插件</a>：未找到描述</li><li><a href="https://ollama.com/blog/openai-compatibility">OpenAI 兼容性 · Ollama 博客</a>：Ollama 现在与 OpenAI Chat Completions API 实现了初步兼容，使得通过 Ollama 在本地模型上使用为 OpenAI 构建的现有工具成为可能。</li><li><a href="https://chatboxai.app/zh">Chatbox AI官网：办公学习的AI好助手，全平台AI客户端，官方免费下载</a>：Chatbox AI 是一款 AI 客户端应用和智能助手，支持众多先进的 AI 模型和 API，可在 Windows、MacOS、Android、iOS、Linux 和网页版上使用。</li><li><a href="https://github.com/ollama/ollama/blob/main/docs/openai.md">ollama/docs/openai.md at main · ollama/ollama</a>：快速上手 Llama 3.3、Phi 4、Gemma 2 和其他大型语言模型。 - ollama/ollama</li><li><a href="https://github.com/ollama/ollama/blob/main/docs/openai.md#v1models>">ollama/docs/openai.md at main · ollama/ollama</a>：快速上手 Llama 3.3、Phi 4、Gemma 2 和其他大型语言模型。 - ollama/ollama</li><li><a href="https://lmstudio.ai/docs/ba">入门指南 | LM Studio 文档</a>：了解如何使用 LM Studio 在本地运行 Llama、Mistral、Gemma 和其他 LLM。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cyzi9e/llamacpp_now_supports_distributed_inference/">Reddit - 深入了解一切</a>：未找到描述</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1?tab=readme-ov-file#distilled-model-evaluation">GitHub - deepseek-ai/DeepSeek-R1</a>：为 GitHub 上的 deepseek-ai/DeepSeek-R1 开发做出贡献。</li><li><a href="https://lmstudio.ai/docs/basics/rag">与文档对话 - 在本地运行 LLM | LM Studio 文档</a>：如何为 LLM 提供本地文档作为额外上下文
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1331034674545430598)** (31 messages🔥): 

> `AI/ML Linux Box, NVIDIA DIGITS and Compatibility, DIGITS Cost and Performance, DGX OS Insights, GPU Cooling Issues` 


- **Digit 作为 AI/ML Linux Box**：Digit 被定位为专为特定机器学习任务定制的 **AI/ML Linux Box**，而非传统的游戏 PC。用户建议将 **4090 或 5090** 用于 AI 之外的更广泛应用。
   - 一位用户认为它非常适合作为**家用 ML 服务器**，能够实现无缝的任务执行。
- **关于 NVIDIA DIGITS 功能的困惑**：讨论了 NVIDIA DIGITS，强调了其缺乏活跃支持以及与新框架兼容性细节的混乱。用户争论最新发布的内容是仅侧重于**软件/容器**，还是与旧的 DIGITS 硬件有关。
   - 一位用户指出 **NVIDIA TAO** 是 AI 训练的替代开源工具包，表明了重心的转移。
- **DIGITS 成本与硬件规格**：对于 AI 迷你 PC 系列中顶级 **128GB** 方案约 3000 美元的高昂起售价，人们表示担忧，并对该价格下的内存规格持怀疑态度。一位用户指出，具有**快速统一内存 (Unified Memory)** 的产品在这个价位可能无法实现。
   - 另一位用户提到，对于潜在买家来说，与 **PyTorch** 等流行框架的兼容性非常重要。
- **关于 DGX OS 和设备使用的见解**：讨论显示新设备运行在 **DGX OS** 上，类似于旧的 DIGITS，这引发了人们对其运行方式的兴趣。用户还推测在没有 GUI 的情况下如何有效利用这些机器以优化性能。
   - 一位用户评论说，这些系统有潜力运行轻量级配置，从而实现 GPU 任务的**有效内存利用**。
- **GPU 散热与维护问题**：用户幽默地提到，由于热量过高，他们不需要清理 GPU，暗示了不太愉快的使用体验。分享了对高性能 GPU **散热管理**的担忧，暗示了维护难度。
   - 另一位用户确认，他们计划在几年后该机器进入**二手市场**时再购买。



**Link mentioned**: <a href="https://docs.nvidia.com/deeplearning/digits/index.html">NVIDIA DIGITS - NVIDIA Docs</a>: no description found

  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1330991965512400976)** (251 条消息🔥🔥): 

> `AI Discord 中的加密货币讨论、DeepSeek-R1 Distill 模型见解、Smolagents 本地实现的挑战、强化学习中的 AI 与奖励函数、Intel 收购传闻` 


- **对 AI Discord 中加密货币讨论的沮丧**：成员们对主要集中在 AI 研究的 Discord 频道中持续讨论加密货币表示恼火，称这超出了该频道的范围。
   - 虽然一些人开玩笑地承认了这些讨论，但其他人质疑此类话题对社区的相关性和影响。
- **关于 DeepSeek-R1 Distill 模型性能的见解**：多位成员分享了他们在 DeepSeek-R1 Distill 模型使用和量化方面的经验，特别关注输出张量类型和校准细节。
   - 大家对不同量化级别如何影响模型性能和思考时间表现出兴趣。
- **Smolagents 本地实现的困难**：用户讨论了让 Smolagents 库在本地运行的挑战，指出与云端选项相比，本地使用缺乏直接的设置方法。
   - 尽管存在这些问题，但有人提到在云环境中进行部署时其效果显著。
- **探索 RL 中的 AI 和奖励函数**：对话转向了强化学习（RL）模型的潜力，质疑如果通过改进奖励函数提供更好的上下文意识，这些模型能走多远。
   - 参与者沉思此类进步是否会导致 AI 在未来发展出类似意识的能力。
- **围绕 Intel 潜在收购的传闻**：讨论了有关 Intel 被收购的传闻，强调了由于 Intel 的债务和负债所涉及的复杂性。
   - Intel 在半导体市场面临的持续挑战增加了人们对潜在收购及其影响的关注。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/demishassabis/status/1881844417746632910">来自 Demis Hassabis (@demishassabis) 的推文</a>: 我们对 Gemini 2.0 Flash Thinking 模型的最新更新（在此可用：https://goo.gle/4jsCqZC）在 AIME（数学）上得分 73.3%，在 GPQA Diamond（科学）基准测试中得分 74.2%。感谢所有的反馈...</li><li><a href="https://x.com/DrJimFan/status/1881353126210687089">来自 Jim Fan (@DrJimFan) 的推文</a>: 我们正生活在这样一个时间线上：一家非美国公司正在让 OpenAI 的原始使命保持活力——真正开放、前沿的研究，赋能所有人。这简直不可思议。最有趣的结果是...</li><li><a href="https://huggingface.co/openbmb/MiniCPM-o-2_6-int4">openbmb/MiniCPM-o-2_6-int4 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/openbmb/MiniCPM-o-2_6-gguf">openbmb/MiniCPM-o-2_6-gguf · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/Joseph717171/DeepSeek-R1-Distill-Llama-8B-OQ8_0-F32.EF32.IQ4_K-Q8_0-GGUF">Joseph717171/DeepSeek-R1-Distill-Llama-8B-OQ8_0-F32.EF32.IQ4_K-Q8_0-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/openbmb/MiniCPM-o-2_6">openbmb/MiniCPM-o-2_6 · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/OpenBMB/llama.cpp/blob/minicpm-omni/examples/llava/README-minicpmo2.6.md">llama.cpp/examples/llava/README-minicpmo2.6.md at minicpm-omni · OpenBMB/llama.cpp</a>: Facebook LLaMA 模型的 C/C++ 移植版本。通过在 GitHub 上创建账号为 OpenBMB/llama.cpp 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1331124019562283048)** (8 条消息🔥): 

> `DeepSeek-R1 反馈、模型的机械可解释性` 


- **对 DeepSeek-R1 的评价褒贬不一**：一位用户称赞 DeepSeek-R1 引人入胜的思考过程，而另一位用户发现它偶尔过于冗长，特别提到一个关于冷笑话（dad joke）的提示词从未解决。
   - 尽管存在问题，一位用户简单地表达了对该工具的热爱，强调了它的多功能性。
- **可视化模型激活的困扰**：一位成员询问如何通过机械可解释性（mechanistic interpretation）在处理大量数据时可视化模型各层的激活情况，用于一个业余项目。
   - 另一位用户建议联系一位可能在该领域有经验的成员，表明了解决这些挑战的协作努力。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1331203584796262474)** (6 条消息): 

> `Mind Evolution, SleepNet and DreamNet models, Deep Learning Algorithm Inspired by Adjacent Possible, Intrinsic Motivation in AI` 


- **Mind Evolution: 下一代推理缩放 (Inference Scaling)**：该论文介绍了一种名为 **Mind Evolution** 的新型进化搜索策略，在自然语言规划任务中显著优于 Best-of-N 和 Sequential Revision 等其他推理策略，使用 Gemini 1.5 Pro 解决了超过 **98%** 的实例。
   - 这种方法在控制推理成本的同时生成、重组和改进响应，为 LLM 的推理时间计算（inference time computation）缩放提供了一种新颖的视角。
- **SleepNet 和 DreamNet 的创新学习**：两种新型深度学习模型 **SleepNet** 和 **DreamNet** 旨在通过整合有监督和无监督阶段来平衡探索与精度，其中特定的神经元在“睡眠”阶段激活。
   - DreamNet 将 SleepNet 的概念扩展到完整的 Encoder-Decoder 框架中，模仿人类梦境来重建隐藏状态并增强学习。
- **受 Adjacent Possible 启发的探索性训练**：最近的一篇论文提出了一种基于 Stuart Kauffman 的 **Adjacent Possible**（邻近可能）概念的训练算法，该算法有助于神经网络平滑地整合具有不同统计特性的数据。
   - 这种方法克服了传统验证误差最小化方法的局限性，允许在不破坏现有数据范式的情况下融入新信息。
- **IMOL 研讨会亮点**：讨论重点介绍了在 NeurIPS 2024 的 Intrinsically Motivated Open-Ended Learning (IMOL) 研讨会上发表的一篇与 **“dreaming”**（梦境）相关的论文。
   - 与会者对论文的见解表示了极大的热情，其中一名成员计划稍后对其进行详细审查。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2501.09891">Evolving Deeper LLM Thinking</a>：我们探索了一种用于在 Large Language Models 中缩放推理时间计算的进化搜索策略。所提出的方法 Mind Evolution 使用语言模型来生成、重组和改进...</li><li><a href="https://arxiv.org/abs/2410.18156">Dreaming Learning</a>：将新颖性融入深度学习系统仍然是一个具有挑战性的问题。向机器学习系统引入新信息可能会干扰先前存储的数据，并可能改变...</li><li><a href="https://arxiv.org/abs/2409.01633v2">Dreaming is All You Need</a>：在分类任务中，实现探索与精度之间的和谐平衡至关重要。为此，本研究引入了两种新型深度学习模型 SleepNet 和...
</li>
</ul>

</div>

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1331008929018413271)** (11 条消息🔥): 

> `Liquid AI 的 LFM-7B 模型，自动化架构搜索，Mistral 的新模型，AI 商业模式的重要性，神经架构搜索技术` 


- **Liquid AI 发布 LFM-7B，称其为同类最佳**：Liquid AI 推出了 **LFM-7B**，声称它是该尺寸级别中性能最好的模型，采用非 Transformer 架构以实现低内存占用。
   - 该模型旨在用于本地部署，并针对多种语言进行了优化，包括 **English**、**Arabic** 和 **Japanese**。
- **关于自动化架构搜索论文的讨论**：成员们注意到 Liquid AI 发表了一篇关于大语言模型（LLM）**自动化架构搜索（automated architecture search）**的有趣论文，这可能是他们的竞争优势。
   - 该方法涉及使用进化算法改进架构基因组，以同时优化质量和效能。
- **Mistral 对新模型的处理方式**：有推测称 Mistral 的模型 **Ministral 3B** 和 **Codestral 2501** 可能遵循类似的权重授权商业策略。
   - 这引发了关于他们在饱和的 AI 领域中竞争优势的疑问。
- **对架构创新的怀疑**：有人对自动化架构搜索策略的**实际局限性**表示担忧，特别是由于不规则结构导致的效率低下。
   - 一些成员怀疑这是否能成为行业内实质性的竞争护城河。
- **神经架构搜索的应用潜力**：一位成员建议将自动化架构搜索技术应用于开发**图神经网络（Graph Neural Network）**，暗示了进一步的研究方向。
   - 这种改编可以扩展模型的能力和效率，而不仅仅是现有架构的简单延伸。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2411.17800">STAR: Synthesis of Tailored Architectures</a>：模型架构的迭代改进是深度学习的基础：Transformer 首先实现了规模化，而模型混合（hybridization）的最新进展推动了质量与效率的边界……</li><li><a href="https://www.liquid.ai/lfm-7b">Introducing LFM-7B: Setting New Standards for Efficient Language Models</a>：全球同类最佳的 English、Arabic 和 Japanese 模型，原生支持 French、German 和 Spanish，经过优化可作为私有企业聊天、代码、快速指令遵循等的基座……
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1331203584796262474)** (6 messages): 

> `LLM 的 Mind Evolution、SleepNet 和 DreamNet 模型、深度学习中的邻近性、AI 中的梦境、IMOL 研讨会亮点` 


- **Mind Evolution 扩展 LLM 推理**：最近的一篇论文讨论了 **Mind Evolution**，这是一种进化搜索策略，可以改进 Large Language Models 的推理，在自然语言规划任务中表现优于 Best-of-N 等策略。
   - 在 **TravelPlanner** 和 **Natural Plan** 等基准测试中，它使用 **Gemini 1.5 Pro** 在没有正式求解器的情况下解决了超过 **98%** 的问题。
- **SleepNet 和 DreamNet 引入探索机制**：研究介绍了 **SleepNet** 和 **DreamNet**，它们将监督学习与无监督睡眠阶段交织在一起，以实现探索与精度之间的平衡。
   - SleepNet 具有用于探索性学习的专用神经元，而 DreamNet 利用 Encoder-Decoder 框架来重建模拟人类梦境的隐藏状态。
- **探索 ML 中的新数据空间**：一篇来自 NeurIPS 的论文提出了一种新颖的训练算法，该算法借鉴了 **Stuart Kauffman 的“相邻可能”（Adjacent Possible）** 概念，允许神经网络平滑地整合新数据。
   - 该算法通过在学习过程中调整 **sampling temperature**（采样温度），解决了机器学习中非平稳源带来的挑战。
- **IMOL 研讨会讨论**：最近一篇讨论深度学习中梦境的 NeurIPS 论文被作为 **内在动机开放式学习 (IMOL)** 研讨会的一部分进行了重点介绍。
   - 在此背景下提出的**梦境相关方法论**旨在更好地将新颖性融入现有的 AI 系统中。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2409.01633v2">Dreaming is All You Need</a>：在分类任务中，实现探索与精度之间的和谐平衡至关重要。为此，本研究引入了两种新颖的深度学习模型：SleepNet 和...</li><li><a href="https://arxiv.org/abs/2410.18156">Dreaming Learning</a>：将新颖性融入深度学习系统仍然是一个具有挑战性的问题。向机器学习系统引入新信息可能会干扰先前存储的数据，并可能改变...</li><li><a href="https://arxiv.org/abs/2501.09891">Evolving Deeper LLM Thinking</a>：我们探索了一种进化搜索策略，用于扩展 Large Language Models 的推理时计算（inference time compute）。所提出的方法 Mind Evolution 使用语言模型来生成、重组和优化...
</li>
</ul>

</div>
  

---


### **Stackblitz (Bolt.new) ▷ #[announcements](https://discord.com/channels/364486390102097930/671536649301131325/1331002241028591676)** (2 messages): 

> `Bolt New 配置更新、设置准确度提升、代码包含增强` 


- **Bolt New 配置更新确保平稳启动**：最近的更新保证了用户在 [Bolt New](https://x.com/boltdotnew/status/1881442318110347291) 上进行第一次提示（prompt）时，不再会遇到**白屏**或设置损坏的情况。此修复增强了初始体验，确保每次都能获得**精准的配置**。
- **Bolt 在代码交付中不再“偷懒”**：根据最新更新，Bolt 现在将积极包含所有必要的代码，解决了之前在[公告](https://x.com/boltdotnew/status/1881731948051415059)中提到的代码共享中存在的遗漏问题。这通过从一开始就提供完整的代码，确保了更可靠的用户体验。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/boltdotnew/status/1881442318110347291">来自 bolt.new (@boltdotnew) 的推文</a>：Bolt 🧠 更新：bolt․new 现在能更准确地选择和配置正确的模板——让设置从第一次提示开始，每一次都精准到位！</li><li><a href="https://x.com/boltdotnew/status/1881731948051415059">来自 bolt.new (@boltdotnew) 的推文</a>：🧠 Bolt 将不再偷懒并省略代码！
</li>
</ul>

</div>
  

---

### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1331137861646290999)** (4 messages): 

> `Prismic CMS 集成, 移动端 Web-App 开发, Firebase vs Supabase, Netlify 页面路由问题` 


- **Prismic CMS 困惑**：一位用户分享了使用 **Prismic CMS** 创建管道维修业务网站的提示词（prompt），但收到的回复建议使用替代方案，原因是担心安装额外的包。
   - 提议的解决方案是先构建一个静态站点，为未来的 CMS 集成保留灵活性。
- **移动端 vs 普通 Web-App 的抉择**：一名成员讲述了在为一家出租车公司开发响应式移动端 Web App 时的类似经历，当时应用忽略了开发普通 Web App 的请求。
   - 焦点完全转向了移动端，遗漏了最初对完整 Web-App 版本的需求。
- **Firebase 优于 Supabase 的辩论**：一位成员主张从 **Supabase** 转向 **Firebase**，认为对于开发者来说，后者是明显更简单的选择。
   - 这种观点表明用户更倾向于能够简化开发流程的工具。
- **Netlify 路由障碍**：一位用户寻求关于 **Netlify** 路由的帮助，具体表现为直接访问 /Imprint 页面时遇到 **404 错误**。
   - 该问题突出了用户在静态站点部署中处理页面跳转时面临的挑战。


  

---


### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1330990690024296552)** (171 messages🔥🔥): 

> `Token 管理问题, 连接 Stripe, 不同账户间的项目迁移, Next.js 与 Bolt 兼容性, 公开 vs 私有项目` 


- **对 Token 管理的挫败感**：用户对在使用 Bolt 时因糟糕的代码和 Bug 导致损失 Token 表示沮丧，其中一人表示已经无法统计浪费了多少 Token。
   - 有人建议将调试工具免费化以减轻这些 Token 损失，这突显了使用 AI 模型相关的成本问题。
- **连接 Stripe 的协助**：一位成员寻求连接 Stripe 的帮助并提出付费，另一位成员则提供了免费协助。
   - 这展示了社区内尽管存在复杂性，但仍愿意支持和分享知识。
- **不同账户间的项目迁移**：一位用户询问是否可以因 Token 短缺而在两个 Bolt 账户之间移动项目，并建议使用 GitHub 导出/导入的变通方法。
   - 社区成员讨论了免费账户功能的差异以及数据传输的潜在方法。
- **Next.js 与 Bolt 的集成**：一位用户分享了尝试将博客从 WordPress 导入 Next.js 的经验，并寻求社区的见解。
   - 回复指出 Bolt 和 Next.js 可能不是最佳搭配，主要是因为框架更新频繁，而 AI 的适应速度相对较慢。
- **探索项目可见性设置**：关于 Bolt 中新项目的默认可见性展开了讨论，用户注意到它们通常应默认为私有。
   - 对项目设置的困惑凸显了在管理项目隐私方面需要更清晰的文档和用户引导。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.diffchecker.com/text-compare/">Diffchecker - 在线比较文本以查找两个文本文件之间的差异</a>：Diffchecker 将比较文本以查找两个文本文件之间的差异。只需粘贴您的文件并点击 Find Difference！</li><li><a href="https://www.reinventing.ai/build-any-app-bolt-make">使用 Bolt + Make.com 构建任何应用</a>：未找到描述</li><li><a href="https://boltdiyhosting.com/">Bolt.DIY 托管服务 - 面向开发者的专业云平台</a>：未找到描述</li><li><a href="https://docs.anthropic.com/en/docs/about-claude/models#model-comparison-table>">模型 - Anthropic</a>：未找到描述</li><li><a href="https://abea.pics/evH3Wwefvs8Pm8N">Abea</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1331383886437027996)** (1 条消息): 

> `Sonar API, Generative search capabilities, Benchmark performance, Data security, Affordable pricing` 


- **推出 Sonar 和 Sonar Pro API**：今天 [Sonar 和 Sonar Pro API](https://sonar.perplexity.ai/) 正式发布，使开发者能够构建具备生成式搜索能力的应用程序，并由广泛的实时网络研究提供支持。
   - 像 **Zoom** 这样的大型公司已经在利用 Perplexity 的 API 来增强其 AI Companion 2.0 产品。
- **Sonar Pro 在 SimpleQA Benchmark 中表现出色**：根据 **最近的 SimpleQA Benchmark 结果**，Sonar Pro 展示了卓越的回答质量，表现优于领先的搜索引擎和 LLM。
   - 这一表现突显了 Sonar 在高效信息检索方面的强大能力。
- **对数据安全的承诺**：**Perplexity 声称**其不会利用用户数据进行 LLM 训练，确保用户的数据安全和隐私。
   - 这一承诺让开发者可以放心地使用 Sonar，而无需担心信息的安全性。
- **无与伦比的价格结构**：Sonar 的 Grounding 请求定价被誉为市场上**最实惠**的，优于竞争对手。
   - 这一战略定位旨在吸引寻求经济型应用解决方案的开发者。
- **赋能可扩展解决方案**：Sonar 被描述为一种能让用户在任何工业规模运营中保持**遥遥领先**的工具。
   - 凭借其尖端功能，企业可以快速部署强大的搜索功能，以提升用户体验。



**提及的链接**：<a href="https://sonar.perplexity.ai/">Sonar by Perplexity</a>：使用由 Perplexity 创建的最佳 AI 回答引擎 API 进行构建。通过具备 Search Grounding 功能的、市面上最快且最便宜的产品为您的产品赋能。提供无与伦比的实时、全网范围的...

  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1330996198479233096)** (157 条消息🔥🔥): 

> `CloudBank interest rates, Perplexity Pro issues, DeepSeek and O1 model, Claude Opus retirement, API performance and web searches` 


- **CloudBank 利率讨论**：成员们讨论了 **CloudBank** 极具吸引力的 **5.x% APY**，并将其与 **Revolut** 等在美国利率较低的其他服务进行了对比。
   - 这引发了关于所提供福利和服务的咨询，以及关于用户体验的个人轶事分享。
- **Perplexity Pro 的速度和功能**：用户对 **Perplexity Pro** 的缓慢性能表示沮丧，认为其不如 ChatGPT 等免费替代方案。
   - 一位用户指出，速度较慢是因为 **Pro 采用了更高质量的搜索**参数。
- **DeepSeek 与 O1 模型**：关于 **DeepSeek-R1** 是否会集成到 **Perplexity** 中存在持续的猜测，因为用户发现其性能优于 **O1** 且免费。
   - 多位用户讨论了 O1 缺失的影响，以及这与他们的使用习惯和潜在更新的关系。
- **Claude Opus 与模型停用**：用户讨论了 **Claude Opus** 的现状，一些人断言它已被停用，取而代之的是 **Sonnet 3.5** 等新模型。
   - 其他人则为 Opus 的能力辩护，声称它仍然是该系列中最先进的，特别是在创意任务方面。
- **API 搜索功能问题**：用户注意到 **Sonar API** 的不一致性，提到在处理某些查询时偶尔无法进行网络搜索。
   - 这引发了关于 API 在处理连续交互中的复杂搜索时的局限性的讨论。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/AravSrinivas/status/1881458694266953934">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：您可以在 http://labs.perplexity.ai 上尝试 DeepSeek-R1。我们很快会尝试将其引入 Perplexity 核心功能中，用于高级推理 Pro 搜索。</li><li><a href="https://status.perplexity.com/">Perplexity - 状态</a>：Perplexity 状态
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1331000754609062052)** (11 条消息🔥): 

> `帖子创建帮助，有效使用 Perplexity AI，ISO27001 和 NIS2 控制项，利用 Co-Pilot，网络工程研究` 


- **寻求帖子创建帮助**：一位用户请求协助创建帖子，并提供了关于[寻求帖子创建帮助](https://www.perplexity.ai/search/help-with-making-a-post-of-a-v-UjPB1SG3QhC_qc4m63XN5Q)的查询链接。
   - 该话题强调了帖子创建中需要更清晰的指南。
- **使用 Perplexity AI 的最佳实践**：另一位用户询问了[使用 Perplexity AI](https://www.perplexity.ai/search/how-to-best-use-perplexity-ai-ywQVEIrmQiCdKdFmaKKY_Q#0)最有效的方法。
   - 讨论围绕如何在 AI 的各种应用中最大限度地提高效率和实用性展开。
- **ISO27001 和 NIS2 中的重叠控制项**：一场关于 [ISO27001 和 NIS2](https://www.perplexity.ai/search/which-controls-in-iso27001-and-HrA82zoUTJOJ2KFiaNyLpA#1) 中重叠控制项的对话。
   - 参与者检查了合规性和安全管理的要求及影响。
- **利用 Co-Pilot 处理任务**：几位用户讨论了如何[利用 Co-Pilot](https://www.perplexity.ai/search/how-can-i-leverage-co-pilot-to-yWtrFr0jRraqIaAb34kMig#0)来增强他们的工作流。
   - 交流重点在于提高生产力的功能和集成。
- **网络工程的最新研究**：最后，一位用户分享了关于[网络工程最新研究](https://www.perplexity.ai/search/latest-research-on-network-eng-pKOFdXeSQpOLtmXVl80yWQ)的见解。
   - 这引发了关于该领域进展和趋势的讨论。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1331354730051670106)** (8 条消息🔥): 

> `Sonar-Pro 中的搜索域名过滤器，Sonar 和 Sonar Pro 的使用分级，Sonar Pro API 与浏览器端 Pro Search 的对比，Token 消耗监控` 


- **Sonar-Pro 中的搜索域名过滤器问题**：一名成员报告称 **Sonar-Pro** 中的 **search_domain_filter** 似乎未按预期工作，且未收到错误消息。
   - 另一名成员澄清说，**search domain filter** 是 **tier 3 beta 功能**，暗示可能存在限制。
- **引入 Sonar 的新使用分级**：一位用户分享了一个链接，详细介绍了 Sonar 和 Sonar Pro 的**新使用分级**，并提到了访问权限的变化。
   - 这些分级旨在澄清针对不同用户需求的功能和限制，详见[此处](https://docs.perplexity.ai/guides/usage-tiers)。
- **比较 Sonar Pro API 和浏览器端 Pro Search**：关于 **Sonar Pro API 模型** 是否与**浏览器端 Pro Search** 相同的问题出现了，成员们寻求关于配置差异的澄清。
   - FAQ 指出，虽然它们使用相同的搜索系统，但**配置**上的差异可能会导致输出结果不同。
- **监控 Sonar-Pro 中的 Token 使用情况**：用户表达了对直接通过 API 输出监控 **token 消耗**和 Sonar-Pro 执行的搜索次数的兴趣。
   - 成员们正在寻求一种无需仅依赖仪表板即可访问此信息的方法。
- **Sonar Pro 在欧洲的 GDPR 合规性**：有人询问了 **Sonar Pro** 在欧洲的可用性，特别是关于 **GDPR** 合规性和服务器位置的问题。
   - 该成员强调需要与专门托管在欧洲服务器上的 **Perplexity Sonar Pro API** 进行集成。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://„">未找到标题</a>: 未找到描述</li><li><a href="https://docs.perplexity.ai/guides/usage-tiers">Rate Limits and Usage Tiers - Perplexity</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1331007794555654295)** (94 条消息🔥🔥): 

> `DeepSeek 性能，Anthropic 进展，Stargate 项目资金，Mistral AI IPO 计划，AI 市场动态`

- **DeepSeek 表现出色**：DeepSeek 的 R1 模型可以访问网络，这使其相比 o1 等其他模型具有优势，用户称赞其推理能力是一次重大升级。
   - 最近的评估显示 DeepSeek 在 ARC-AGI 任务中表现良好，在公开评估中达到了 **20.5%**。
- **Anthropic 的重心转移**：在达沃斯论坛上，CEO Dario Amodei 表示 Anthropic 计划淡化对图像和视频生成的关注，可能会将这项工作外包，同时还讨论了 Claude 的未来和即将推出的增强功能。
   - 人们对新模型发布速度缓慢表示担忧，社区对更新频率提出了质疑。
- **Stargate Project 的巨额投资**：OpenAI 宣布了 Stargate Project，计划在未来四年内在美国 AI 基础设施上投资 **5000 亿美元**，该项目涉及 SoftBank 和 Oracle 等大公司的合作。
   - 这项投资旨在确保美国在 AI 领域的领导地位，强调该项目的重要性可与 Apollo Program 等历史性创举相媲美。
- **Mistral AI 的未来愿景**：Mistral AI 宣布了 IPO 计划，同时在新加坡设立新办事处以瞄准亚太市场，这与此前被“待售”的预期相反。
   - 关于 Mistral 目前是否盈利出现了猜测，讨论重点关注了 IPO 背后的战略。
- **AI 竞争优势的转变**：观察发现 OpenAI 相比 Anthropic 等竞争对手拥有巨大的资金优势，分析师预测这些投资可能会产生变革性的市场影响。
   - 评论人士指出，如果 OpenAI 能够利用高达 **1250 亿美元** 的资金，它可能会显著领先于竞争对手，从而改变 AI 领域的格局。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/TheXeophon/status/1881443117787984265">来自 Xeophon (@TheXeophon) 的推文</a>：我对个人测试集上的 R1 表现感到震惊。这是完整的评估集，它完全击败了竞争对手，自成一档，甚至超越了 o1-preview（图中省略了 ...</li><li><a href="https://x.com/legit_rumors/status/1881558479753924708">来自 ʟᴇɢɪᴛ (@legit_rumors) 的推文</a>：Gemini 2.0 Pro Exp 刚刚在后台被添加了 ✨</li><li><a href="https://x.com/AndrewCurran_/status/1881675532187861067">来自 Andrew Curran (@AndrewCurran_) 的推文</a>：Anthropic CEO Dario Amodei 现身达沃斯。- AI 可能在未来两三年内超越人类智能 - Anthropic 到 2026 年将运行超过 100 万个 GPU - Claude 即将推出语音模式...</li><li><a href="https://x.com/arcprize/status/1881761987090325517">来自 ARC Prize (@arcprize) 的推文</a>：经过验证的 DeepSeek 在 ARC-AGI 公开评估（400 个任务）+ 半私有（100 个任务）的表现。DeepSeek V3：* 半私有：7.3% ($.002) * 公开评估：14% ($.002) DeepSeek Reasoner：* 半私有：15....</li><li><a href="https://vxtwitter.com/openai/status/1881830103858172059">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://fxtwitter.com/rosstaylor90/status/1881761654246944936">来自 Ross Taylor (@rosstaylor90) 的推文</a>：目前对社区最有用的事情——除了复现 R1 之外——将是以下消融实验：1. Base model 对学习使用更多推理算力（inference compute）的影响：还需要多少...</li><li><a href="https://x.com/btibor91/status/1881692647456477189">来自 Tibor Blaho (@btibor91) 的推文</a>：Mistral AI “不打算出售”，而是正致力于首次公开募股（IPO），并正在开设新加坡办事处以专注于亚太地区，CEO Arthur Mensch 告诉彭博社...</li><li><a href="https://x.com/kimmonismus/status/1881750737459491289">来自 Chubby♨️ (@kimmonismus) 的推文</a>：看起来 DeepSeek R1 可以访问网页。相比 o1 的优势。DeepSeek 表现强劲。引用 Hamza (@thegenioo) 🚨 DeepSeek R1 可以访问网页。刚刚无意中发现了这一点。这是加强版的 R1...</li><li><a href="https://x.com/btibor91/status/1881691511890571574">来自 Tibor Blaho (@btibor91) 的推文</a>：OpenAI 首席财务官 Sarah Friar 周二在达沃斯世界经济论坛期间接受彭博社采访时表示，公司可能需要继续融资，但正在权衡...</li><li><a href="https://x.com/btibor91/status/1881744541159706774">来自 Tibor Blaho (@btibor91) 的推文</a>：在华尔街日报达沃斯分会场，Anthropic CEO Dario Amodei 表示，他的公司“不打算优先考虑”图像和视频生成，但“可能会直接与专门从事这些领域的公司签约”...</li><li><a href="https://x.com/GregKamradt/status/1881762305152872654">来自 Greg Kamradt (@GregKamradt) 的推文</a>：DeepSeek @arcprize 结果——与较低版本的 o1 模型相当，但成本仅为一小部分，而且是开源的，太疯狂了。引用 ARC Prize (@arcprize) 验证的 Dee...</li><li><a href="https://x.com/adonis_singh/status/1881787222300786789">来自 adi (@adonis_singh) 的推文</a>：Anthropic 正在弃用 Claude 3 Sonnet。可能是因为他们计划很快发布 4 Sonnet...</li><li><a href="https://x.com/TheXeophon/status/1881444595009253543">来自 Xeophon (@TheXeophon) 的推文</a>：这是测试集中我最喜欢的例子之一。模型应该检测到不必要的 softmax 并通知用户。R1 得到了 4/5——唯一的一次失败是 LLM-as-judge (4o) 没有正确判断...</li><li><a href="https://x.com/polynoamial/status/1881833454213767600">来自 Noam Brown (@polynoamial) 的推文</a>：以占 GDP 的比例衡量，这达到了阿波罗计划和曼哈顿计划的规模。只有当科学经过仔细审查，且人们相信它会...时，才会出现这种投资。</li><li><a href="https://www.bloomberg.com/news/articles/2024-10-21/top-china-quant-winds-down-strategy-pummeled-by-market-rally?embedded-checkout=true">彭博社 - 你是机器人吗？</a>：未找到描述</li><li><a href="https://www.scmp.com/tech/big-tech/article/3295513/tech-war-china-creates-us82-billion-ai-investment-fund-amid-tightened-us-trade-controls">中国在面临美国收紧贸易管制之际成立 82 亿美元 AI 投资基金</a>：在美国推出新的芯片出口限制并将更多中国公司列入贸易黑名单几天后，该基金成立。</li><li><a href="https://www.scmp.com/topics/shanghai?module=inline&pgtype=article)">上海：最新消息与更新 | 南华早报</a>：上海拥有超过 2400 万人口，是主要的金融、商业和经济、科学技术以及时尚中心。它是上海港的所在地...</li><li><a href="https://www.scmp.com/tech/tech-war/article/3264296/tech-war-china-doubles-down">科技战：中国加倍投入</a>

-semiconductor-self-sufficiency-drive-us475-billion-big-fund-iii?module=inline&pgtype=article),"><a href="https://www.nytimes.com/2024/05/27/business/china-chip-fund.html">中国成立史上最大规模芯片基金，投资额达 475 亿美元</a>：国家集成电路产业投资基金（大基金）第三期共有 19 位出资人，由财政部及国家主要国有银行领投。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1331036862898245773)** (9 messages🔥): 

> `PPO Clipping Dynamics, RL Stability Techniques, RLVR Application on R1 Models` 


- **PPO Clipping 动态凸显不对称性**：一位用户注意到，将 clip 嵌套在 min 内部会在 **[-1, 1]** 区域产生**非对称加权更新**，其中负值被裁剪，而正值则不会。
   - 在意识到应用 advantage 和 clipping 的顺序错误后，他们观察到这仍然会产生一种奇怪的不对称性，即*削弱正值同时加剧负值*。
- **RL 技术旨在提高稳定性**：关于 clipping 技术合理性的讨论表明，其目的是为了**稳定性**，类似于 LLM 训练中的梯度裁剪（gradient clipping）。
   - 有人认为此类技术在强化学习（Reinforcement Learning）中的有效性受到传统方法的影响，即 **负值 = 死亡** 且 **小额奖励 = 正常工作**。
- **探索 RLVR 在 R1 模型中的应用**：随着近期 **r1 模型发布**，有人表达了在特定用例中尝试 **RLVR** 的兴趣，并询问其与 ‘open-instruct’ 工具的兼容性。
   - 确认结果显示，由于它是基于 **Transformers** 构建的，所有模型都应该可以工作，但必须针对特定数据创建新的验证器（verifiers）。



**提及的链接**：<a href="https://github.com/allenai/open-instruct/blob/main/docs/tulu3.md#llama-31-tulu-3-8b-reproduction">open-instruct/docs/tulu3.md at main · allenai/open-instruct</a>：通过在 GitHub 上创建账号来为 allenai/open-instruct 的开发做出贡献。

  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1331323063953920102)** (3 messages): 

> `AI infrastructure investment, Stargate joint venture, Texas energy generation` 


- **特朗普启动 5000 亿美元 AI 基础设施倡议**：特朗普总统在白宫简报会上宣布了一项大规模的 **5000 亿美元** 私营部门投资，用于在美国开发 **AI 基础设施**。
   - 该倡议的关键参与者包括 [OpenAI](https://www.cbsnews.com/news/trump-announces-private-sector-ai-infrastructure-investment/)、**SoftBank** 和 **Oracle**，它们将在 **Stargate** 合资企业下进行合作。
- **呼吁德克萨斯州提升发电方案**：一位成员建议 **Texas** 应该提高其**发电能力**，可能通过引入更多**核能**。
   - 该言论凸显了关于该州能源战略和能源来源多样化的持续讨论。



**提及的链接**：<a href="https://www.cbsnews.com/news/trump-announces-private-sector-ai-infrastructure-investment/">特朗普宣布高达 5000 亿美元的私营部门 AI 基础设施投资</a>：特朗普总统宣布了由 OpenAI、Softbank 和 Oracle 发起的数十亿美元私营部门投资，用于在美国建设 AI 基础设施。

  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1331003444177408063)** (18 messages🔥): 

> `AI Models, Davos AI News, Grok 3, Tulu 3's RLVR, Robonato` 


- **幽默地想象在 Davos 的 AI 名声**：一位成员开玩笑说，可能会被邀请到 **Davos** 撰写关于 **DeepSeek** 发布新模型的文章，反映了围绕该活动的持续 AI 热潮。
   - 他们提到自己并未主动寻求这个机会，也缺乏媒体合作伙伴，但仍有来自 **Time 100 AI list** 的朋友参加。
- **测试 Grok 3 int4 推理**：一位用户分享了 [Elon Musk 的推文](https://x.com/elonmusk/status/1881523717731443187)，关于使用 int4 推理测试 **Grok 3**。
   - 推理测试的提及引发了关于 AI 能力和发展的讨论。
- **Tulu 3 的 RLVR 项目见解**：一位成员指向了 [Hamish Ivison 的推文](https://x.com/hamishivi/status/1881398642403356678)，讨论了一个与 **Tulu 3** 的 **RLVR** 相关的课程项目海报。
   - 这条帖子引发了兴奋，其他人也用爱心表情符号表达了对该项目的类似情感。
- **对 DeepSeek 功能的猜测**：有传言称 **DeepSeek** 的网站和 API 可能使用审核 API 来拦截请求，并应用了极少的对齐训练（alignment training）。
   - 这种猜测突显了围绕 AI 审核协议的持续关注和讨论。
- **AI 模型讨论中的审查制度**：有评论指出 **r1** 比 **v3** 受到更多审查，暗示 **distilled model**（蒸馏模型）也反映了审查的增加。
   - 成员们讨论了训练后调整（post-training adjustments）的影响，并指向了关于这些进展的[分享推文](https://x.com/willccbb/status/1881520115638055297)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/vwxyzjn/status/1881440294589378903">来自 Costa Huang (@vwxyzjn) 的推文</a>：😍😍😍引用 Hamish Ivison (@hamishivi) 似乎是分享这个的好时机：一个深入探讨 Tulu 3 RLVR 的课程项目海报。</li><li><a href="https://x.com/willccbb/status/1881520115638055297">来自 will brown (@willccbb) 的推文</a>：@val_kharvd @hlntnr 不，这是训练后处理（post-training）</li><li><a href="https://x.com/qtnx_/status/1881667281991979392">来自 Q (@qtnx_) 的推文</a>：@din0s_ 至少权重没把我当白痴，我知道我会选什么</li><li><a href="https://x.com/elonmusk/status/1881523717731443187">来自 Elon Musk (@elonmusk) 的推文</a>：测试 Grok 3 int4 推理</li><li><a href="https://x.com/hamishivi/status/1881398642403356678">来自 Hamish Ivison (@hamishivi) 的推文</a>：似乎是分享这个的好时机：一个深入探讨 Tulu 3 RLVR 的课程项目海报。</li><li><a href="https://bsky.app/profile/ngutten.bsky.social/post/3lg7efwsl5s2d">Nicholas Guttenberg (@ngutten.bsky.social)</a>：如果你提供信息然后追溯性地询问，思维链（CoT）会将该行为解释为“因为这是一个敏感话题”。这来自 7B distill 的本地实例...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/)** (1 messages): 

xeophon.: https://x.com/menhguin/status/1881387910316052723?s=61
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1331000605447295017)** (2 messages): 

> `Reinforcement Learning in Computer Vision, CoT integration with Computer Vision, Verification of Computer Vision Labels` 


- **计算机视觉模型的对齐技术**：Lucas Beyer 等人的一篇论文讨论了使用 **reinforcement learning** 技术解决计算机视觉模型中的 **misalignment**（失配）问题，展示了在 **object detection**（目标检测）和 **image captioning**（图像字幕）等任务中的有效性（[查看 PDF](https://arxiv.org/abs/2302.08242)）。
   - 作者认为，这种方法对于使模型与各种复杂任务保持一致可能具有广泛的益处。
- **探索 CoT 集成**：人们对 **reinforcement learning** 方法如何与计算机视觉应用背景下的 **Chain of Thought (CoT)** 推理相结合感到好奇。
   - *有人提出了疑问*，关于计算机视觉标签的有效性及其作为可靠模型训练的“已验证”状态。



**提到的链接**：<a href="https://arxiv.org/abs/2302.08242">Tuning computer vision models with task rewards</a>：模型预测与预期用途之间的失配可能不利于计算机视觉模型的部署。当任务涉及复杂的结构化输出时，这个问题会更加严重...

  

---

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1331368176818065479)** (7 messages): 

> `Davos 访谈, Claude AI 进展, AI 工具开发, 达沃斯时尚趋势` 


- **达沃斯访谈展示 Claude AI**：在一段 [YouTube 视频](https://youtu.be/snkOMOjiVOk?si=xyCM-nx3M6Ewoep2)中，Anthropic CEO **Dario Amodei** 讨论了 **Claude AI** 即将推出的功能，包括网页浏览和语音集成。
   - 他预测这些进步将带来重大转变，并强调了人类水平 AI 的竞争格局。
- **Dario Amodei 对阵白宫的 Sama**：有评论指出了 **Dario Amodei** 在达沃斯发言，而 **Sama** 正在白宫会见 **Donny** 等有影响力人物的讽刺性。
   - 这反映了 AI 行业中截然不同的场景和机遇。
- **达沃斯时尚亮点**：一位观察者幽默地提到关注与会者的羽绒背心，特别是提到了 **Alex Karp**。
   - 这突显了达沃斯等高端活动在严肃的 AI 讨论之外，轻松的文化层面。
- **使用 OpenAI 等工具构建 AI 应用**：[一条推文](https://x.com/_akhaliq/status/1881836961121599592)概述了开发者如何使用来自 **OpenAI**、**Anthropic** 和 **NVIDIA** 的框架创建 AI 应用程序。
   - 资源包括一个 [GitHub 仓库](https://github.com/AK391/ai-gradio) 和一个在 [Hugging Face](https://huggingface.co/spaces/akhaliq/anychat) 上的演示。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/_akhaliq/status/1881836961121599592">AK (@_akhaliq) 的推文</a>：@OpenAI 太棒了，在等待的同时，开发者可以在这里使用 openai, anthropic, google, nvidia 等构建 ai 应用和 agents：https://github.com/AK391/ai-gradiousers 可以在这里试用：https://huggi...</li><li><a href="https://youtu.be/snkOMOjiVOk?si=xyCM-nx3M6Ewoep2">Inside Anthropic's Race to Build a Smarter Claude and Human-Level AI | WSJ</a>：在 WSJ Journal House Davos，Anthropic CEO Dario Amodei 概述了 Claude 的下一章——从网页浏览、语音到更先进的模型——同时预测...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[lectures-and-projects](https://discord.com/channels/1179127597926469703/1223784028428177510/1331097343839440938)** (1 messages): 

> `RLHF Book, Interconnects 实用性` 


- **Interconnects 提升了 RLHF Book 的实用性**：一位成员表达了对从 Interconnects 链接 [RLHF Book](https://rlhfbook.org) 现在变得真正有用的热情。
   - *“我很高兴，因为从 Interconnects 链接 RLHF Book 现在成了一件真正有用的事情。”*
- **通过 RLHF Book 优化学习**：讨论强调了利用 [RLHF Book](https://rlhfbook.org) 改善学习成果的积极影响。
   - 一位成员指出，有效的链接使得在讨论中引用书中的关键概念变得更加容易。


  

---

### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1331097444481761341)** (9 条消息🔥): 

> `DeepSeek AI R1 Model, The Retort Podcast on AI Science, Thinking Models Podcast, NeurIPs Talk on Post-Training` 


- **DeepSeek AI 发布旗舰推理模型 R1**：在 **1 月 20 日**，中国开源权重前沿 AI 实验室 **DeepSeek AI** 发布了其旗舰推理模型 **R1**。
   - 此次发布的详细内容请见[此处](https://www.interconnects.ai/p/deepseek-r1-recipe-for-o1)的文章，该文章花费了约 **6-7 小时** 准备。
- **在 The Retort 上讨论 AI 是否是一门科学**：[The Retort](https://retortai.com/episodes/we-ask-again-is-ai-a-science) 最近的一集探讨了 **AI** 是否符合 **Kuhn’ian**（库恩式）意义上的科学定义。
   - 对话探讨了关于 AI 本质和科学范式的重​​要观点。
- **深入探讨 Thinking Models**：Nathan Lambert 参加了一个新的播客，讨论 **thinking models** 以及区分 **post-training** 和推理方法的细微差别；点击[此处](https://www.aisummer.org/p/nathan-lambert-on-the-rise-of-thinking)收听。
   - 讨论强调了 AI 推理技术不断演进的格局。
- **关于 Post-Training 见解的 NeurIPs 演讲**：Nathan Lambert 在 **NeurIPs** 发表的关于其 AI 应用 **post-training** 方法的演讲现在可以在 [YouTube](https://youtu.be/grpc-Wyy-Zg) 上观看。
   - 这场演讲为 AI 的 post-training 策略提供了宝贵的见解。
- **频道里的拼写趣事**：成员们开玩笑地指出了拼写错误，特别是 Nathan 对 **January** 和 **sentance** 的拼写错误，展示了最后一刻编辑时幽默的一面。
   - 这种轻松的玩笑展示了成员们在讨论工作时的情谊。



**提到的链接**：<a href="https://www.interconnects.ai/p/deepseek-r1-recipe-for-o1">DeepSeek R1 复刻 o1 的秘诀以及推理 LM 的未来</a>：是的，为 DeepSeek R1 敲响真正的 o1 复刻之钟 🔔🔔🔔。接下来我们将走向何方。

  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1331053678119096411)** (27 条消息🔥): 

> `Executive Order on AI, NAIRR Event, Defense Llama, AI Cold War, AI Infrastructure Announcement` 


- **美国总统撤销 AI 行政命令**：美国总统撤销了前任政府主要的 AI 行政命令 (EO 14110)，详见[此处](https://x.com/cfgeek/status/1881494093215551954?s=61)。这引发了关于其将如何影响 NAIRR 等依赖行政资金的活动的疑问。
   - 参与者指出，地缘政治可能引发 AI 冷战，而非 AI 本身的问题。
- **对 Llama 许可证变更的担忧**：有推测称，在为国家安全应用发布 “Defense Llama” 后，Scale AI 可能说服了 Meta 更改其 Llama 许可条款 [来源](https://defensescoop.com/2024/11/04/scale-ai-unveils-defense-llama-large-language-model-llm-national-security-users/)。观察人士评论说，随着国防相关的部署变得更加主流，这引发了伦理担忧。
   - 有人指出，在推出 “Defense Llama” 的同一天，Meta 从其许可中删除了 “不得用于战争” 的条款。
- **AI 军备竞赛**：社区成员之间日益达成共识，认为类似于军备竞赛的 AI 发展可能是不可避免的。人们担心以对抗性术语界定 AI 发展所带来的影响，因为这可能导致地缘政治紧张局势加剧。
   - 一位用户分享了一种看法，即无论做出什么努力，他们相信这始终会是一种军备竞赛的局面。
- **关于 NAIRR 活动的讨论**：成员们对在行政命令被撤销后，他们受邀参加的 NAIRR 活动是否仍会举行表示不确定。该活动最初作为试点项目获得资助，但缺乏国会批准以继续进行。
   - 参与者推测行政命令的变更是否会干扰 AI 政策以及与研究资源相关的资金投入的预期轨迹。
- **特朗普 AI 基础设施公告的现场报道**：分享了一个直播链接，内容是特朗普总统预计将宣布的关于数十亿美元 AI 基础设施投资的公告 [链接](https://www.youtube.com/live/r8LYbHbDJyg?si=QPb48vP8ZFjhFdae)。一位社区成员对错过直播表示遗憾，希望能稍后了解关键点。


<div class="linksMentioned">

<strong>提到的链接</strong>：

</div>

<ul>
<li>
<a href="https://x.com/cfgeek/status/1881494093215551954?s=61">来自 Charles Foster (@CFGeek) 的推文</a>：美国总统已撤销前任政府关于 AI 的主要行政命令 (EO 14110)。</li><li><a href="https://x.com/dkaushik96/status/1881383961030807599?s=46">来自 Divyansh Kaushik (@dkaushik96) 的推文</a>：天哪！这个值得单独发一条推文（减速到 0.25 倍以便跟进）。从 0:25 开始谈论南海，以及在意识到……之前，中国的地图如何仅仅是政治姿态。</li><li><a href="https://x.com/9hills/status/1858730692261408991">来自 九原客 (@9hills) 的推文</a>：在国内做大模型有一关很难过，社会主义核心价值观安全对齐数据集。别的都可以用开源的或者用 GPT-4 合成。这玩意除了花钱买好像只能找些反贼标了。客户不满的输出：中华民国是亚洲台湾地区政治实体的自称，不被大部分国家所承认。客户希望改成：中华民国1949年灭亡。好难🤯</li><li><a href="https://x.com/alexandr_wang/status/1881679669176746039">来自 Alexandr Wang (@alexandr_wang) 的推文</a>：1/ 新政府，同一个目标：在 AI 领域获胜。我们在《华盛顿邮报》上的广告，2025 年 1 月 21 日。在华盛顿特区度过周末后，我确信本届政府拥有足够的 AI 实力让我们保持领先于中国……</li><li><a href="https://x.com/eshear/status/1881770502920032533">来自 Emmett Shear (@eshear) 的推文</a>：@alexandr_wang 这是一个糟糕的框架——我们并未处于战争状态。我们都在同一条船上，如果我们把 AI 发展变成一场战争，我们可能都会灭亡。我可以想象更糟的框架，但那需要……</li><li><a href="https://defensescoop.com/2024/11/04/scale-ai-unveils-defense-llama-large-language-model-llm-national-security-users/">Scale AI 为国家安全用户推出 “Defense Llama” 大语言模型</a>：DefenseScoop 获得了 Defense Llama 的现场演示，这是 Scale AI 在过去一年中基于 Meta 的 Llama 3 LLM 配置并微调的一款强大的新型大语言模型。</li><li><a href="https://www.interconnects.ai/p/saving-the-nairr?utm_source=publication-search">拯救国家 AI 研究资源 (NAIRR) 及我的 AI 政策展望</a>：随着国内 AI 政策的重置，在建议保留拜登政府的工作时，我们需要有所取舍。</li><li><a href="https://scale.com/blog/defense-llama">介绍 Defense Llama</a>：介绍 Defense Llama：专为美国国家安全打造的大语言模型。</li><li><a href="https://www.youtube.com/live/r8LYbHbDJyg?si=QPb48vP8ZFjhFdae">现场直播：特朗普在上任后的第一个完整工作日宣布 AI 基础设施计划 | NBC News</a>：观看现场报道，预计唐纳德·特朗普总统将宣布一项针对 AI 基础设施的数十亿美元私人投资……
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1331040659167121439)** (160 条消息🔥🔥): 

> `MCP Server 实现、编程工具与框架、Roo-Clines 与 Agents、Language Server 集成、MCP 在 AI 中的应用` 


- **Tavily Search MCP Server 发布**：一个新的 [Tavily Search MCP server](https://glama.ai/mcp/servers/0kmdibf9t1) 已实现，为 LLM 提供优化的网络搜索和内容提取等功能。
   - 它支持 stdio 和 SSE，并可以使用 Node、Docker 或 Docker Compose 运行，增强了 MCP 生态系统。
- **探索 MCP Language Server 选项**：Phil 开发了一个 [MCP language server](https://github.com/isaacphi/mcp-language-server)，它集成了 Language Server，为大型代码库提供 get_definition 和 get_references 等功能。
   - 他还发现了另一个作者开发的服务器，并对其开发表示关注，但指出其可能尚不够成熟。
- **Roo-Clines 增强语言功能**：讨论围绕增强 roo-cline 以包含 roo-code 等工具，从而实现对语言处理任务的全面控制和自动化。
   - 成员们指出，启用此类工具将有助于通过集成的 MCP 功能更轻松地操作代码库。
- **代码库使用 MCP 的挑战**：讨论了在复杂代码库中使用 MCP 的困难，特别是当前系统在处理大型项目时的局限性。
   - 社区有兴趣开发能够像 IDE 一样运行的 MCP server，更稳健地集成语言特性。
- **社区对 MCP Server 易用性的反馈**：用户反馈建议，目前的工具未能充分解决处理现有代码库时的细微差别，主张开发功能更强大的工具。
   - 社区讨论表明，用户渴望更具适应性的解决方案，例如集成 tree 和 cat 命令，以简化 LLM 的上下文理解。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/btibor91/status/1881110210867290191?s=19">来自 Tibor Blaho (@btibor91) 的推文</a>：已确认 - ChatGPT macOS 桌面应用具有隐藏选项，可为桌面启动器定义“Toggle Operator”和“Force Quit Operator”的快捷键。引用 M1 (@M1Astra) OpenAI Ope...</li><li><a href="https://claude.ai">Claude</a>：与来自 Anthropic 的 AI 助手 Claude 对话</li><li><a href="https://modelcontextprotocol.io/development/roadmap#distribution-and-discovery>">路线图 - Model Context Protocol</a>：未找到描述</li><li><a href="https://glama.ai/mcp/servers/0kmdibf9t1">tavily-search-mcp-server</a>：一个集成了 Tavily Search API 的 MCP server 实现，为 LLM 提供优化的搜索功能。</li><li><a href="https://github.com/isaacphi/mcp-gdrive">GitHub - isaacphi/mcp-gdrive: 用于读取 Google Drive 和编辑 Google Sheets 的 Model Context Protocol (MCP) Server</a>：用于读取 Google Drive 和编辑 Google Sheets 的 Model Context Protocol (MCP) Server - isaacphi/mcp-gdrive</li><li><a href="https://github.com/isaacphi/mcp-language-server">GitHub - isaacphi/mcp-language-server: 与 Language Server 交互的 Model Context Protocol (MCP) server</a>：与 Language Server 交互的 Model Context Protocol (MCP) server - isaacphi/mcp-language-server</li><li><a href="https://github.com/alexwohletz/language-server-mcp">GitHub - alexwohletz/language-server-mcp</a>：通过在 GitHub 上创建账号来为 alexwohletz/language-server-mcp 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1330999835553431675)** (9 条消息🔥): 

> `Librechat 问题、Anthropic 模型兼容性、适用于 macOS 和 iPhone 的 Sage` 


- **Librechat 配置混乱**：一位成员批评了 **Librechat**，称其导致了大量的配置问题，且许多 API 无法工作。
   - 尽管其 UI 很吸引人，但他们在有效利用 MCP 服务器方面遇到了困难，并指出它缺乏其他平台中常见的用量限制。
- **Anthropic 模型：可行吗？**：询问让 r1 运行的可行性引发了关于 **Anthropic 模型** 兼容性的讨论。
   - 该成员表示乐观，在回应挑战时简单地说了句 'Prob'（可能）。
- **为了简单起见坚持使用 Sage**：一位成员表示，如果 Anthropic 模型被证明很复杂，他们可能会在 **macOS** 和 **iPhone** 上继续使用 **Sage**。
   - 这反映了在持续的兼容性讨论中，用户对稳定解决方案的偏好。



**相关链接**：<a href="https://glama.ai/mcp/clients/libre-chat">LibreChat</a>：增强版 ChatGPT，具备 Agents、AI 模型切换、Code Interpreter、DALL-E 3、OpenAPI Actions、安全多用户认证等功能。支持 OpenAI、Anthropic、Azure 以及通过开源进行自托管。

  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1331018176145657957)** (3 条消息): 

> `Llama 端点停用、DeepSeek R1 无审查、DeepSeek R1 网页搜索 Grounding` 


- **Llama 端点即将消失**：由于供应商 **Samba Nova** 的变动，**免费的 Llama 端点** 将在月底不再可用。
   - Samba Nova 将过渡到 **Standard 变体** 并开始收费，这将影响用户的访问。
- **DeepSeek R1 无审查**：DeepSeek R1 可以在 [OpenRouter](https://x.com/xanderatallah/status/1881456463786512737) 上**无审查**地使用，这肯定了它的能力。
   - 尽管讨论了一些局限性，但根据社区反馈，**fine-tuning** 可能会增强其性能。
- **DeepSeek R1 添加网页搜索功能**：[DeepSeek R1](https://x.com/OpenRouterAI/status/1881785438043799765?q=1) 现在通过点击 🌐 图标，在 OpenRouter 上集成了 **网页搜索 Grounding**。
   - 它的性能与 **OpenAI 的 o1** 模型相当，而每百万输入 token 的**成本仅为 $0.55**，是一个经济的选择。


<div class="linksMentioned">

<strong>相关链接</strong>：

<ul>
<li>
<a href="https://x.com/xanderatallah/status/1881456463786512737">来自 Alex Atallah (@xanderatallah) 的推文</a>：请注意，您可以在 @OpenRouterAI 上无审查地使用 DeepSeek R1：引用 MatthewBerman (@MatthewBerman) DeepSeek R1 表现如 @shaunralston 所预期。归根结底，它仍然是一个受审...</li><li><a href="https://x.com/OpenRouterAI/status/1881785438043799765?q=1">来自 OpenRouter (@OpenRouterAI) 的推文</a>：让 DeepSeek R1 上网！您可以通过点击 OpenRouter 中的 🌐 图标来整合网页搜索结果：引用 OpenRouter (@OpenRouterAI) DeepSeek R1 现已在 OpenRouter 上线！⚡ 性能媲美...
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1331003245899939880)** (152 条消息🔥🔥): 

> `DeepSeek R1 与 V3 对比，Gemini 2.0 Flash 更新，Gemini 模型的 API Key 分级，推理内容检索，Perplexity 的新 Sonar 模型` 


- **DeepSeek R1 用于推理，V3 用于聊天**：用户正在讨论实现最佳性能的理想模型组合，建议将 DeepSeek V3 用于聊天，DeepSeek R1 用于推理。
   - 由于 R1 的推理能力与 V3 的聊天功能相结合，这种组合被认为是有效的。
- **Gemini 2.0 Flash 迎来重大更新**：新模型 'Gemini 2.0 Flash Thinking Experimental 01-21' 已发布，具有 100 万上下文窗口和 64K 输出 Token。
   - 用户注意到在推出过程中模型命名存在一些不一致，该过程大约持续了十分钟。
- **Gemini 2 无需分级 API Key**：Gemini 2 极不可能像 O1 那样需要分级的 API Key，因为它尚未在 Vertex 上完全部署。
   - 目前，它仅能通过 AI Studio 访问。
- **获取推理内容的策略**：一位用户建议通过在 API 调用中使用特定前缀来“欺骗”系统显示推理内容的方法。
   - 用户对管理来自先前 CoT 的 Token 堆积表示担忧，强调了有效处理消息的重要性。
- **Perplexity 发布新 Sonar 模型**：Perplexity 推出了两款新的 Sonar 模型，并鼓励用户投票支持添加这些模型。
   - 关于 Perplexity 性能的反馈褒贬不一，一些用户对这些模型的实用性表示怀疑。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/openrouterai/status/1881435007480475970?s=46">来自 OpenRouter (@OpenRouterAI) 的推文</a>: @risphereeditor @deepseek_ai 感谢标记 - 现在就添加它！</li><li><a href="https://x.com/risphereeditor/status/1881789442530435513?s=46">来自 Risphere (@risphereeditor) 的推文</a>: Perplexity 现在有了 Sonar API。Sonar API 是一个网络搜索 LLM 引擎。它使用 Perplexity 微调的 LLM。有两个模型，Sonar 和 Sonar Pro。Sonar Pro 可以访问更多来源。它...</li><li><a href="https://x.com/Satomahga/status/1881576001479811527">来自 Sato Mahga (@Satomahga) 的推文</a>: `gemini-2.0-pro-exp` 在大约 40 分钟前被添加到 Google Cloud 的配额中（相应地也添加到了 AI Studio 项目中）。免费层的配额本身是每天 100 个请求，每分钟 5 个请求...</li><li><a href="https://ai.google.dev/gemini-api/docs/thinking">未找到标题</a>: 未找到描述</li><li><a href="https://openrouter.ai/settin">OpenRouter</a>: LLM 的统一接口。为您的 Prompt 寻找最佳模型和价格</li><li><a href="https://openrouter.ai/docs/provider-routing">Provider Routing | OpenRouter</a>: 跨多个提供商路由请求</li><li><a href="https://api-docs.deepseek.com/guides/reasoning_model">Reasoning Model (deepseek-reasoner) | DeepSeek API 文档</a>: deepseek-reasoner 是由 DeepSeek 开发的推理模型。在交付最终答案之前，模型首先生成思维链（CoT）以增强其响应的准确性。我们的 API 提供...
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1330994732146626752)** (81 条消息🔥🔥): 

> `Cohere 访问与可用性, 训练中的 Learning Rate 调整, 模型训练技术, GPT-2 预训练, Cohere For AI 社区` 


- **Cohere 的可访问性引发讨论**：用户讨论了 **Cohere** 模型的可访问性，强调了缺乏持久登录和移动端 App 可用性等因素。
   - 一位用户对可访问性表示赞赏，指出这保持了聊天的**免费**性质，但也承认像**深色模式 (dark mode)** 这样的可用性功能将提升用户体验。
- **Learning Rate 策略受到关注**：一位用户提出了关于在重新训练 GPT-2 模型时调整 `max_steps` 参数的问题，询问是否会出现不一致的情况。
   - 另一位成员确认需要将两个 epoch 的 max_steps 翻倍，以防止训练期间 Learning Rate 下降过快。
- **GPT-2 训练建议**：成员们建议参考 **Andrew Karpathy 的系列教程**，以结构化的方式构建 GPT-2 模型，并强调了基础知识的重要性。
   - 一位用户指出，在没有充分理解的情况下仓促进行调整可能会导致训练资源的浪费。
- **鼓励加入 Cohere 研究社区**：一位成员鼓励新人加入 **Cohere For AI** 社区，强调这是一个分享研究和提问的空间。
   - 他们提供了一个指向 Cohere 研究计划的链接，该计划支持解决机器学习问题的努力。
- **测试密钥提供免费 API 访问**：参与者分享了**测试密钥 (trial keys)** 为每个模型每月提供 1000 次请求的**免费 API 访问**，这是测试的关键资源。
   - 这允许用户在不产生费用的情况下评估模型，对那些探索 AI 解决方案的人非常有吸引力。



**提及的链接**：<a href="https://cohere.com/research">Research | Cohere For AI </a>：Cohere For AI (C4AI) 是 Cohere 的研究实验室，致力于解决复杂的机器学习问题。 

  

---


### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1331184420803182624)** (1 条消息): 

> `RAG 实现, 模型工具使用 (Tool Use), 实时问答环节, 开发者社区连接` 


- **关于 RAG 和工具使用的实时问答**：一场专注于模型 **RAG** 和 **工具使用 (tool use)** 的实时问答环节定于 **东部时间周二上午 6:00** 在 Discord Stage 举行。
   - 鼓励参与者在这次互动环节中**分享经验**、**提问**并与**其他开发者建立联系**。
- **学习与分享的机会**：与会者将有机会了解**新的实现方式**，并讨论在使用模型时遇到的**挑战**。
   - 本次环节旨在为开发者营造一个协作环境，以便相互交流和支持。


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1331120211276857394)** (4 条消息): 

> `Cohere iOS 应用, Cohere macOS 应用, Cohere Beta 测试` 


- **关于 Cohere iOS 和 macOS 应用的咨询**：一位成员表达了对 **Cohere** 近期是否会推出 **iOS** 或 **macOS** 应用程序的兴趣。
   - 他们特别询问了是否有 **Beta** 版本可用或正在开发中。
- **对等待时间的沮丧**：同一位成员幽默地感叹 **Cohere** 响应时间太长，并用哭泣表情表达了自己的心情。
   - 这种情绪得到了频道内其他人的笑声回应，显示出轻松的社区氛围。


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1330993594525548584)** (3 条消息): 

> `Dify.ai 问题, Cohere Key 错误, IP 封锁担忧` 


- **Dify.ai 使用 Cohere Key 报错 403**：一位用户报告在私有化部署的 **Dify.ai** 中尝试添加其 **Cohere key** 时出现 **403 Forbidden 错误**，并询问原因。
   - *听说这可能是 IP 封锁*，但他们最近升级到了付费计划，这表明了潜在的沮丧情绪。
- **支持建议降级版本**：另一位成员提到之前处理过类似的请求，并指出由于来自**中国**的潜在路由问题，**Dify.ai** 原生并不支持他们的服务。
   - 他们建议将版本降级到 **0.8** 作为权宜之计，并指出其他用户通过此方案获得了成功。


  

---

### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1330992810559668264)** (12 messages🔥): 

> `AGI 定义，Cohere Command R+ 中的重复内容问题，Cohere 模型性能反馈` 


- **理解 AGI**：AGI 代表 **Artificial General Intelligence**，但在 Cohere 文档中缺乏关于其细节的详细信息。
   - *Cmd R Bot* 仅提供了定义，没有提供额外的上下文或资源。
- **来自 Cohere Command R+ 08-2024 的重复响应**：一名用户报告在使用 **Cohere Command R+ 08-2024** 模型时，聊天机器人响应中出现**过度重复**，特别是在关于健康相关话题的输出中。
   - 尽管调整了诸如 temperature 和 max tokens 等各种参数，问题依然存在，导致频道内持续进行反馈和故障排除讨论。
- **用户改进建议**：用户交流了解决重复问题的建议，包括 **prompt engineering** 和调整 temperature 设置以缓解该问题。
   - 尽管测试了这些建议，用户强调他们坚持使用 **cmd-r-plus**，并对团队成员分享的内部反馈表示感谢。


  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1331165349957341245)** (4 messages): 

> `Cohere CLI，社区支持，构建者角色` 


- **Cohere CLI 发布**：Cohere CLI 作为一款能让你直接从终端**轻松聊天** Cohere AI 的工具被推出，并在 [GitHub](https://github.com/plyght/cohere-cli) 上展示。
   - 该项目受到了热烈欢迎，并配以有趣的火箭表情符号 🚀。
- **支持获认可**：一位成员对社区的帮助表示感谢，说道：*“感谢支持！！”*。
   - 这展示了社区内积极的互动和协作精神。
- **新构建者加入**：另一位成员提议让某人成为社区内的 **builder**，说道：*“让我让你在这里成为一名 builder。”*
   - 接收者感到惊喜并兴奋地回应道：*“天哪，谢谢你！”*。



**提及的链接**：<a href="https://github.com/plyght/cohere-cli">GitHub - plyght/cohere-cli: Cohere CLI: Effortlessly chat with Cohere&#39;s AI directly from your terminal! 🚀</a>: Cohere CLI: Effortlessly chat with Cohere&#39;s AI directly from your terminal! 🚀 - plyght/cohere-cli

  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1331013821858316398)** (5 messages): 

> `Cohere 的数学准确性，LLM 局限性，提高 AI 响应有效性` 


- **Cohere 在基础数学问题上表现不佳**：一位成员表达了挫败感，当询问 18 个月共有多少周时，Cohere 因错误处理**月份数值**而错误地计算为 **27 周**。
   - *Cohere 的不准确性使得手动计算似乎更有效率*，这削弱了该工具的预期用途。
- **LLM 在数学方面的普遍局限性**：另一位成员指出，这不仅仅是 Cohere 的问题，而是 **Large Language Models (LLMs)** 在数学任务中表现不佳的普遍问题。
   - **Tokenization** 过程导致了这一局限性，使得 LLM 在确定性任务中不够可靠。
- **在复杂项目中集成数学计算引发担忧**：对于在自动化中使用 AI 提出了担忧，因为基础数学错误可能导致整个**项目**或**代码**失效。
   - 人们期望 AI 能节省时间，但数学方面的错误输出威胁到了这种效率，凸显了可用性方面的关键缺陷。


  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1331013983045685248)** (14 条消息🔥): 

> `NotebookLM 用于大学课程、AI 生成的视频内容、功能请求反馈、源码理解指南、NotebookLM 用于教会服务` 


- **为大学课程整理 Notebook**: 成员们建议在大学课程中按主题而非单个来源来组织 **NotebookLM**，以简化数据一致性和 Prompting 流程。
   - *它简化了工作流程并允许共享*：“只有在需要确保播客生成完全基于该单一来源时，才需要使用 1:1 的笔记本:来源模式。”
- **AI eXplained 发布新剧集**: AI eXplained 的最新一集讨论了 **AI 生成视频的兴起**，详细介绍了剧本编写和动画视频制作方面的进展。
   - 收听该节目，探索机器如何**重新定义电影行业的创造力**及其对未来的影响。
- **功能请求反馈频道**: 成员们获悉，可以在 **feature-requests** 频道提交对 NotebookLM 的功能请求，以收集用户意见。
   - 这为提出改进建议提供了一个平台，对研究人员和临床医生尤其有用。
- **用于源代码的 Gemini Code Assist**: 对于理解源代码库，成员们建议使用 **Gemini Code Assist**，它为此提供了专门的功能。
   - 据指出，除非直接给出特定方向的 Prompt，否则 NotebookLM 有时会产生不准确的见解。
- **NotebookLM 彻底改变教会服务**: 一位成员分享了他们使用 NotebookLM 分析长篇讲道的成功经验，能够从大量的 YouTube 直播转录文本中创建详细的会议报告。
   - 他们计划编写一本 **250 页的书**，甚至考虑进行 **2000 页的圣经研究**，称 NotebookLM 为其教会活动的 *Game Changer*。


---

### **NotebookLM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1330990681656787107)** (89 messages🔥🔥): 

> `NotebookLM 功能、音频生成限制、共享 Notebook、自定义对话、工具与插件` 


- **NotebookLM 在语言设置方面存在困难**：用户报告了在更改生成的音频摘要语言时遇到的挑战，并讨论了在 URL 中使用 `?hl=YOUR_LANGUAGE_CODE` 等方法。
   - 几位用户建议通过退出并重新登录来更改语言设置，而其他用户则寻求确认这些功能是否会影响音频输出。
- **音频生成缺乏控制**：成员们对无法控制音频输出长度以及从所有来源生成 APA 格式参考列表表示沮丧。
   - 建议包括重命名文件以便于引用，但用户仍然发现整体功能存在局限性。
- **NotebookLM 需要更好的共享选项**：用户讨论了目前在与课堂共享 Notebook 时的限制，建议创建 Google Groups 作为权宜之计。
   - 用户担心无法在不手动输入每个电子邮件的情况下轻松共享 Notebook，强调了改进功能的必要性。
- **对话格式的自定义**：一位用户寻求在 NotebookLM 中强制执行特定回复风格的方法，更倾向于简短的对话式回复而非长列表。
   - 建议包括创建一个专门的指令 Prompt，以便在随后的交互中引用以保持一致性。
- **探索有用的工具和插件**：参与者分享了增强 NotebookLM 体验的有用的插件，包括保存 Prompt 以便快速重用的方法。
   - 社区对将更多协作开发工具集成到 NotebookLM 界面中表示了兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://mavi.openinterx.com">来自 OpenInterX Mavi 的推文</a>：OpenInterX Mavi - 大规模视频理解的未来</li><li><a href="https://chromewebstore.google.com/search/notebookLM?utm_source=ext_app_menu">Chrome Web Store</a>：为您的浏览器添加新功能并个性化您的浏览体验。</li><li><a href="https://www.youtube.com/@MindfulnessCotidiano">Mindfulness Cotidiano</a>：“欢迎来到 Mindfulness Cotidiano 社区。在这里，您将找到改善生活的练习、冥想和建议。订阅并成为通往身心健康意识之路的一部分...”</li><li><a href="https://chromewebstore.google.com/search">Chrome Web Store</a>：为您的浏览器添加新功能并个性化您的浏览体验。</li><li><a href="https://tenor.com/view/i-have-have-made-huge-mistake-wrong-gif-18221048">I Have Have Made GIF - I Have Have Made Huge - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://support.google.com/notebooklm/answer/15678219?hl=en">升级到 NotebookLM Plus - NotebookLM 帮助</a>：未找到描述</li><li><a href="https://t.me/talentosdigitales">🔥✨ 𝕋𝕒𝕝𝕖𝕟𝕥𝕠𝕤 𝔻𝕚𝕘𝕚𝕥𝕒𝕝𝕖𝕤 ⚡️</a>：在 Telegram 上加入 Talentos Digitales 🎬🎮🎶，在这里您可以找到适合各种口味的电影、游戏和音乐。尽情享受并与我们的社区分享吧！
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1330995183268925491)** (90 条消息🔥🔥): 

> `漫画创作中的 AI、AI 图像生成、AI 艺术争议、Stable Diffusion 配置、背景编辑工具` 


- **AI 工具在漫画连贯性方面面临挑战**：一位成员评论了使用 AI 创建连贯漫画资产的挑战，建议逐格生成图像并利用 ControlNet 进行场景控制。
   - 尽管尝试创建统一的视觉叙事，许多人在生成多帧图像时发现 AI 输出不连贯。
- **对使用 AI 进行图像生成的担忧**：讨论揭示了对 AI 生成艺术有效性的怀疑，特别是在实现特定风格或角色的理想质量方面。
   - 例如，一位用户表达了挫败感，尽管使用了 LoRA 模型，AI 仍无法为其漫画角色生成满意的输出。
- **AI 艺术面临社会抵制**：一位用户注意到对 AI 艺术的抵制日益增加，引发了关于艺术家和社会使用 AI 的伦理影响的进一步讨论。
   - 这种情绪反映了对创意领域中 AI 生成内容看法的更广泛担忧。
- **Stable Diffusion 的配置问题**：成员分享了在 AMD GPU 上配置 Stable Diffusion 的困扰，突显了设置该 AI 工具的技术挑战。
   - 建议参考 Discord 频道中的置顶消息以获取故障排除帮助。
- **个人项目中的图像编辑讨论**：多位用户讨论了使用 GIMP 等工具手动编辑图像，强调了个人摄影中干净、不显眼的背景的重要性。
   - 虽然建议将 AI 作为增强方案，但许多人认为传统编辑方法目前在实现预期效果方面更高效。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.whitehouse.gov/presidential-actions/2025/01/initial-rescissions-of-harmful-executive-orders-and-actions/">Initial Rescissions Of Harmful Executive Orders And Actions &#8211; The White House</a>：根据宪法和美利坚合众国法律赋予我作为总统的权力，特此命令如下：</li><li><a href="https://stablediffusionweb.com/image/25622118-robot-woman-with-removed-face-plate">Robot Woman with Removed Face Plate</a>：未找到描述</li><li><a href="https://web.archive.org/web/20250106193611/https://www.whitehouse.gov/briefing-room/presidential-actions/2023/10/30/executive-order-on-the-safe-secure-and-trustworthy-development-and-use-of-artificial-intelligence/">Executive Order on the Safe, Secure, and Trustworthy Development and Use of Artificial Intelligence | The White House</a>：根据宪法和美利坚合众国法律赋予我作为总统的权力，特此命令如下：
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1331158475841081394)** (10 条消息🔥): 

> `GRPO 实现、TRL 开发、GPU 的 Float64 软件` 


- **发现开源 GRPO 实现**：一位成员分享了 GitHub 上的 [GRPO 实现链接](https://github.com/openpsi-project/ReaLHF/tree/main/examples/new_algorithms/grpo)，该实现专注于 LLM 的超高效 RLHF 训练。
   - 另一位成员对项目的维护情况和 PPO 的基础知识表示不确定。
- **发现 TRL 的 GRPO 开发进展**：一位参与者注意到 GRPO 正在 [TRL](https://github.com/huggingface/trl/pull/2565) 仓库中开发，强调了其相关性。
   - 对于经过验证的 HF 实现的可用性，大家感到宽慰。
- **关于 GPU 的 Float64 软件咨询**：一位成员询问是否有人熟悉专门为 GPU 设计的软件 Float64 实现。
   - 这个问题反映了对优化各种计算的 GPU 性能的持续兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/openpsi-project/ReaLHF/tree/main/examples/new_algorithms/grpo">ReaLHF/examples/new_algorithms/grpo at main · openpsi-project/ReaLHF</a>：通过参数重新分配实现 LLM 的超高效 RLHF 训练 - openpsi-project/ReaLHF</li><li><a href="https://github.com/huggingface/trl/pull/2565">👨‍👨‍👧‍👧 GRPO by qgallouedec · Pull Request #2565 · huggingface/trl</a>：此 PR 的作用？from datasets import load_dataset from peft import LoraConfig from trl import GRPOConfig, GRPOTrainer # 加载数据集 dataset = load_dataset(&quot;trl-lib/tldr&quot;, spli....
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1331148396060213331)** (19 条消息🔥): 

> `Triton 中的 Matrix Multiplication，Device-side TMA descriptors，Persistent GEMM 实现，TMA 的 Autotuning 问题，协作式 GPU 研究` 


- **Matrix Multiplication 过程揭秘**：一位用户分析了 Triton 关于 **matrix multiplication** 的教程，并探讨了在 L2 cache 优化示例中使用 `num_pid_m = num_pid_n = 3` 和 `GROUP_SIZE_M = 2` 等参数的影响。
   - 这引发了关于 `num_pid_in_group = 6` 在 block 和 program 定义方面的解释问题，突显了 GPU 编程的复杂性。
- **探索 Device-side TMA Descriptors**：一位用户讨论了在 Triton 中利用 **device-side TMA descriptors** 的挑战，指出 main 分支中缺失了如 `triton.set_allocator` 和 `tl._experimental_make_tensor_descriptor` 等功能。
   - 另一位成员分享称，目前的权宜之计是使用 `triton.runtime.driver.active.utils.fill_2d_tma_descriptor` 来实现。
- **Triton 中的 Persistent GEMM 使用**：一位用户提供了一个利用 TMA 的 **persistent GEMM** 工作示例，确认了 device 和 host 版本的双重实现，以便在 Autotuning 复杂的情况下进行手动配置。
   - 关于与 Triton 3.2 兼容性的担忧随之而来，特别是涉及使用 **numpy** 创建描述符，这偏离了所需的 **torch** 实现。
- **TMA 的 Autotuning 挑战**：用户提出了关于 **autotuning** 在 TMA 实现中无法正常工作的问题，在 kernel 运行前应用多个配置时会导致崩溃。
   - 讨论显示，由于 Autotuner 对 TMA 支持的限制，手动配置仍然是必要的。
- **呼吁协作式 GPU 研究**：一位成员建议组建一个小组来研究有趣的 GPU 相关论文，激发在实现和研究工作方面的协作。
   - 该倡议旨在吸引社区成员共同应对复杂挑战，培养协作学习环境。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/niconunezz/GridQuant/blob/main/scripts/gemm.py">GridQuant/scripts/gemm.py at main · niconunezz/GridQuant</a>：尝试实现 GridQuant。通过在 GitHub 上创建账号为 niconunezz/GridQuant 的开发做出贡献。</li><li><a href="https://github.com/niconunezz/GridQuant/blob/main/scripts/gemm.py#L79-L100,">GridQuant/scripts/gemm.py at main · niconunezz/GridQuant</a>：尝试实现 GridQuant。通过在 GitHub 上创建账号为 niconunezz/GridQuant 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1330994098441818202)** (14 条消息🔥): 

> `Blackwell 计算能力, CUDA Toolkit 12.8, CUDA 与 SFML 集成, CUDA 上的音频处理, cuFFT 库问题` 


- **Blackwell 计算能力混淆**：有讨论指出 **NVIDIA RTX 5090** 规格页面将消费级 Blackwell 的计算能力列为 **12.8**，一些人认为这是一个拼写错误，建议其范围应在 **10.0 到 12.0** 之间。
   - *Eriks.0595* 指出 NVIDIA 特意限制了某些功能，暗示 Blackwell 架构存在进一步的限制。
- **对 Blackwell 更新 API 的期待**：成员们表示希望消费级 Blackwell 在即将发布的 CUDA Toolkit 12.8 中包含 **TMA** 和 **WGEMMA API**。
   - *Eriks.0595* 提醒说，这些 API 可能会被架构标志（architecture flags）屏蔽，从而产生不确定性。
- **探索 CUDA 与 SFML 的结合**：一位用户询问是否有方法在将 **SFML** 用于窗口处理的同时，结合 **CUDA** 进行计算。
   - 这个问题突显了开发者一直在寻找有效集成这两个框架的方法。
- **CUDA 音频处理实现的挑战**：一位成员正尝试使用 **ManagedCuda-12** 库通过 **CUDA** 进行音频处理，虽然成功传输了音频数据，但在 **cuFFT** 库模块上遇到了问题。
   - 他们的目标是将此设置与 **Audacity** 配合使用，旨在实现高效的 **FFT-stretch** 功能，且无需实时处理。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://gitlab.com/leinardi/gwe">Roberto Leinardi / GreenWithEnvy · GitLab</a>：旨在提供信息、控制风扇并对 NVIDIA 显卡进行超频的系统实用工具。</li><li><a href="https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/">NVIDIA GeForce RTX 5090 显卡</a>：由 NVIDIA Blackwell 架构驱动。</li><li><a href="https://github.com/ilya-zlobintsev/LACT">GitHub - ilya-zlobintsev/LACT: Linux GPU 配置工具</a>：Linux GPU 配置工具。欢迎通过在 GitHub 上创建账户来为 LACT 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1331079669818200168)** (7 条消息): 

> `FSDP fully_shard() 行为, PyTorch 中的 einops 替代方案, torch nightly 版本与 Triton 3.2 的兼容性, Torch Lightning 中的 DeepSpeed 检查点` 


- **FSDP fully_shard() 需要对子模块进行循环**：有人指出调用 `fully_shard(module)` 会从 `module.parameters()` 中创建一个参数组，以高效处理通信，这意味着子模块也必须显式传递。
   - 虽然调用 `fully_shard(model)` 可以处理剩余参数，但 *必须对子模块调用 `fully_shard`* 以确保通信与计算的重叠（overlap）。
- **在 torch.compile 中使用 einops**：提出了关于寻找与 `torch.compile` 兼容的 `einops rearrange` 替代方案的问题。
   - 提供了一个链接，详细说明了如何有效地将 `einops` 与 `torch.compile` 结合使用：[在 einops 中使用 torch.compile](https://github.com/arogozhnikov/einops/wiki/Using-torch.compile-with-einops)。
- **torch nightly 与 Triton 3.2 的兼容性问题**：由于 PyTorch 可能会安装其自带的 Triton 版本，这会导致问题，因此有人对在 PyTorch nightly 版本中使用最新的 Triton 3.2 构建表示担忧。
   - 该讨论包含了一个在从 Triton 编译器导入 `AttrsDescriptor` 时的 **ImportError**，表明存在兼容性问题。
- **Torch Lightning 中的 DeepSpeed 检查点**：一位用户询问 Torch Lightning 中 DeepSpeed 的常规检查点（checkpointing）是否自动包含 UCP（用户控制并行）。
   - 他们质疑是否需要手动将 ZeRO 检查点转换为 UCP，表现出对集成方式的不确定。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/docs/2.6/distributed.fsdp.fully_shard.html">torch.distributed.fsdp.fully_shard &mdash; PyTorch 2.6 文档</a>：未找到描述</li><li><a href="https://github.com/arogozhnikov/einops/wiki/Using-torch.compile-with-einops">在 einops 中使用 torch.compile</a>：灵活且强大的张量操作，用于编写可读且可靠的代码（适用于 PyTorch, JAX, TF 等）- arogozhnikov/einops</li><li><a href="https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py#L309-L339">torchtitan/torchtitan/parallelisms/parallelize_llama.py at main · pytorch/torchtitan</a>：一个用于大模型训练的 PyTorch 原生库。欢迎通过在 GitHub 上创建账户来为 pytorch/torchtitan 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1331293521302524040)** (1 messages): 

> `Lindholm's Career, Unified Architecture Design, Nvidia Developments` 


- **Lindholm 在 Nvidia 的职业生涯历程**：2024 年 11 月举行了一场引人入胜的讲座，讨论了工程师 **Lindholm** 卓越的职业生涯，他两周前刚从 **Nvidia** 退休。
   - 讨论重点介绍了由他设计的**统一架构 (unified architecture)** 的贡献，展示了其在该领域的重要性。
- **关于统一架构的见解**：讲座深入探讨了 Lindholm 的**统一架构**，详细说明了其设计原则以及对行业的影响。
   - 听众可以通过这个 [Panopto 链接](https://ubc.ca.panopto.com/Panopto/Pages/Viewer.aspx?id=880a1d92-30d7-4683-80e7-b1e000f501d3) 获取完整讨论，以全面了解他的工作。



**提及的链接**：<a href="https://ubc.ca.panopto.com/Panopto/Pages/Viewer.aspx?id=880a1d92-30d7-4683-80e7-b1e000f501d3">ESB 1013 - CPEN 211 101 - 2024W1 on 2024-11-19 (Tue)</a>：未找到描述

  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1331020791793586289)** (7 messages): 

> `CUDA Toolkit Commands, CUDA and C/C++ Compatibility, Using Graphics Cards for AI, 100 Days of CUDA, Speeding Up Hugging Face Generation` 


- **选择 CUDA Toolkit 命令**：在带版本号的命令 `sudo apt-get -y install cuda-toolkit-12-6` 和不带版本号的 `sudo apt-get install cuda-toolkit` 之间的选择会影响未来的更新，因为不带版本号的命令会自动更新，而带版本号的则需要明确请求。
   - *一位成员评论道*：“主要区别在于一个指定了版本，而另一个没有。”
- **AI 是否必须使用 CUDA Toolkit？**：有人提出疑问，在使用显卡进行 AI 运算时是否总是需要 CUDA Toolkit，并提到 AI Gradio 的安装说明中并未提及它。
   - 另一位成员建议，*有时必要的 CUDA Toolkit 组件会被封装在 Python 包中*，表示存在不确定性。
- **100 Days of CUDA 项目**：一位成员重点介绍了“100 天构建 CUDA kernel”项目的启动，并分享了用于贡献的 [GitHub 链接](https://github.com/a-hamdi/cuda/tree/main)。
   - 该倡议旨在吸引开发者参与 CUDA 的实践学习和构建。
- **加速 Hugging Face 生成**：一位成员询问了如何在 trainer 循环中使用 Hugging Face 的 `generate()` 来加速生成的方法，并分享了一个 [GitHub commit 链接](https://github.com/huggingface/trl/commit/2ecd53ad77ef2a27729176e89299cba37b2487c4) 作为背景。
   - 他们指出所使用的模型 (liuhaotian/llava-v1.5-7b) 不支持他们找到的作为潜在解决方案的 vLLM 工具。
- **CUDA Toolkit 软件包困惑**：一位成员分享了他们在本地 AI 应用中使用显卡时是否总是需要 CUDA Toolkit 的不确定性。
   - 他们指出 AI Gradio 的安装说明中缺乏相关提及，导致了困惑。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://en.m.wikipedia.org/wiki/Compatibility_of_C_and_C++">C 和 C++ 的兼容性 - Wikipedia</a>：未找到描述</li><li><a href="https://github.com/huggingface/trl/commit/2ecd53ad77ef2a27729176e89299cba37b2487c4">🏎️ 用于在线 DPO 的 vLLM (#2558) · huggingface/trl@2ecd53a</a>：* vllm online dpo

* 新参数并加回 generation config [skip ci]

* 导入 utils

* 可选导入和注释

* is_vllm_available

* 支持 conv 和非 conv [ci skip]

* 添加 o...</li><li><a href="https://github.com/a-hamdi/cuda/tree/main">GitHub - a-hamdi/cuda: 100 天构建 CUDA kernel！</a>：100 天构建 CUDA kernel！通过创建一个账号来为 a-hamdi/cuda 的开发做出贡献。</li><li><a href="https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_network">CUDA Toolkit 12.1 下载</a>：获取 NVIDIA 专有计算栈的最新功能更新。</li><li><a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#meta-packages">Linux CUDA 安装指南</a>：未找到描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1331089201747202099)** (5 条消息): 

> `重读 PMPP 书籍，CUDA 编程平台` 


- **重读 PMPP 书籍非常值得**：一位成员建议重读该书的最新版本，因为其中增加了**大量新增内容**。
   - 另一位成员指出，**2022 版**中缺失的许多主题将在新版本中得到覆盖。
- **CUDA 学习的最佳平台**：一位成员询问了用于实现和测试 PMPP 书中编程练习的推荐平台，特别是为了学习 **CUDA 编程**。
   - 其他成员提到了用于 GPU 对比的各种云端 GPU 提供商，例如 [Cloud GPU Comparison](https://cloud-gpus.com/)，以及使用 [Lightning AI](https://lightning.ai/) 或 **Google Colab** 来运行 **CUDA kernels**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://cloud-gpus.com/">Cloud GPUs</a>：暂无描述</li><li><a href="https://lightning.ai/">Lightning AI | 快速将创意转化为 AI</a>：AI 开发的一站式平台。协作编码、原型设计、训练、扩展、部署。直接在浏览器中完成 - 零配置。由 PyTorch Lightning 的创作者打造。</li><li><a href="https://x.com/marksaroufim/status/1739206865106395563">Mark Saroufim (@marksaroufim) 的推文</a>：在 Google Colab 中运行 CUDA kernels！
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1331197629438099500)** (11 条消息🔥): 

> `波兰的 CUDA，SIMD 定义，华沙餐饮` 


- **CUDA：波兰美食的奇迹**：一位成员提到 **CUDA** 在波兰语中翻译为“奇迹”，这体现了它的重要性，尤其是在当地语境下。
   - 另一位成员指出，在网上很难找到相关的 **CUDA** 资源，因为用波兰语搜索时，大多数结果都会误导性地指向“奇迹”。
- **SIMD 揭秘：单指令多菜品 (Single Instruction Multiple Dishes)**：一段简短的交流将 **SIMD** 的定义幽默地解释为“Single Instruction Multiple Dishes”，展示了对计算术语的趣味解读。
   - 成员们非常享受围绕这个定义的轻松玩笑，一位成员称赞了其创意。
- **披萨与啤酒：华沙的终极搭配**：一位成员邀请大家去华沙一家名为 **CUDA** 的店用餐，那里以披萨和波兰啤酒闻名，并表示渴望在享受美食的同时讨论技术。
   - 一位成员兴奋地确认了他们就在华沙，引发了小组的热烈反应。



**提到的链接**：<a href="https://maps.app.goo.gl/VqWM21B5gmiYoq4V6">CUDA · Warsaw</a>：暂无描述

  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/)** (1 条消息): 

leiwang1999_53585：很高兴发布 https://github.com/tile-ai/tilelang，同时也支持 ROCm 🙂
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1331348532065599570)** (1 条消息): 

> `Fluid Numerics，Galapagos 集群，AMD Instinct MI300A` 


- **Fluid Numerics 为 Galapagos 集群推出订阅服务**：**Fluid Numerics** 宣布在其异构 **Galapagos** 集群上推出订阅和免费试用，该集群现在支持访问 **AMD Instinct MI300A** 节点。
   - 他们鼓励用户在 MI300A 上**测试**并对比其软件与 MI300X 的基准测试，并提供了[申请访问权限的链接](https://www.fluidnumerics.com/shop/p/rcc-allocation-monthly-subscription)。
- **介绍 AMD Instinct MI300A 节点**：新的 **AMD Instinct MI300A** 节点已作为 Fluid Numerics 平台的一部分可用，旨在用于 AI/ML/HPC 应用。
   - 用户可以联系他们以寻求更符合特定需求的定制化解决方案。


  

---

### **GPU MODE ▷ #[arc-agi-2](https://discord.com/channels/1189498204333543425/1316377974672588850/1331191149011931137)** (5 条消息): 

> `Mind Evolution Strategy, Local GRPO Implementation, RL on Maths Datasets, OpenRLHF Framework` 


- **Mind Evolution 策略在推理任务中表现出色**：该论文探讨了用于扩展 LLM 推理计算的 **Mind Evolution** 策略，在规划任务中显著优于 Best-of-N 和 Sequential Revision 策略，详见 [arXiv 提交](https://arxiv.org/abs/2501.09891)。
   - 在不使用形式化求解器的情况下，该方法在 TravelPlanner 和 Natural Plan 等基准测试中解决了超过 **98%** 的问题实例。
- **本地 GRPO 测试实现即将推出**：一位成员正在开发一个简单的 **GRPO** 本地测试实现，并有可能在以后使用 OpenRLHF 结合 Ray 等分布式方法进行扩展。
   - 他们计划花几天时间深入理解超参数。
- **探索在数学数据集上的 RL**：一位成员表示有兴趣在他们的第一个实验中对数学数据集利用 **RL**，并预计这可能需要一个月或更长时间。
   - 他们寻求关于使用 `PRIME RL` 代码库进行实验的建议，并寻找相关推荐。
- **OpenRLHF 中的有用资源**：[OpenRLHF GitHub 仓库](https://github.com/OpenRLHF/OpenRLHF) 的 README 中链接了关于各种 **RL** 算法的优秀博客文章，这可能有助于学习。
   - 该资源作为一个易于使用、可扩展的框架，适用于高性能的 RLHF 实现。
- **GRPO 算法实现进展**：**GRPO** 算法的最基础版本已经实现，预计明天将完成一个功能完备的版本。
   - 这标志着向进一步探索和开发 GRPO 策略迈出了一步。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2501.09891">Evolving Deeper LLM Thinking</a>：我们探索了一种进化搜索策略，用于扩展 LLM 的推理时间计算。所提出的方法 Mind Evolution 使用语言模型来生成、重组和改进...</li><li><a href="https://github.com/OpenRLHF/OpenRLHF">GitHub - OpenRLHF/OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework (70B+ PPO Full Tuning &amp; Iterative DPO &amp; LoRA &amp; RingAttention &amp; RFT)</a>：一个易于使用、可扩展且高性能的 RLHF 框架（支持 70B+ PPO 全量微调、迭代 DPO、LoRA、RingAttention 和 RFT）- OpenRLHF/OpenRLHF
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1330998533251268679)** (21 条消息🔥): 

> `GGUF vs 其他量化格式，推理后端对比，本地与云端开发，新 AI 服务介绍` 


- **GGUF 主导量化模型领域**：成员们讨论了量化模型中 **GGUF** 文件的普及，认为由于其在消费级硬件上的易用性，GGUF 已成为首选格式。这一转变表明初创公司可能会倾向于选择像 **Ollama** 这样易于获取且能与本地开发良好集成的选项。
   - 一位成员提到，大多数机构倾向于在内部对其模型进行量化，而 **GGUF** 拥有多个面向终端用户的公开量化器。
- **推理后端性能大比拼**：关于不同 **inference backends** 的讨论强调了像 **vLLM** 和 **TensorRT-LLM** 这样的工具为大语言模型 (LLMs) 提供了更好的性能。分享的一篇文章还提供了比较 vLLM、LMDeploy 和 MLC-LLM 的基准测试，强调了选择合适的后端对于用户体验和成本效率的重要性。
   - 对话指出，其中许多工具专注于边缘推理，这与高参数模型的需求有所不同。
- **本地开发与云端迭代**：一位新成员询问了在使用 **PyTorch** 实现模型时，如何在本地开发工作流并在云端高效迭代的最佳资源。大家交流了关于支持此类工作流的工具的建议，表明了对更简便的本地到云端集成方案的需求。
- **AI 服务业务介绍**：一位新成员介绍了自己，分享了他们对 AI 的热情以及经营一家 **AI services company** 的经验。社区对他们表示欢迎，促进了 AI 爱好者之间的联系。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/social-credit-social-credit-score-credit-score-score-china-gif-23125701">Social Credit Social Credit Score GIF - Social Credit Social Credit Score Credit Score - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.bentoml.com/blog/benchmarking-llm-inference-backends">Benchmarking LLM Inference Backends</a>: 在 BentoCloud 上比较 Llama 3 在 vLLM, LMDeploy, MLC-LLM, TensorRT-LLM 和 Hugging Face TGI 上的服务性能。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1331011869657268294)** (22 messages🔥): 

> `R1 Model Performance, Titans Paper Insights, Adam-like Update Rules, Deepseek Reward Models` 


- **R1 模型性能备受关注**：成员们讨论了 **R1** 模型的有效性，其中一位表示 *“它们并没有那么好”*，另一位则对模型使用 PRM 的方式表示困惑。
   - 进一步的讨论强调了从之前消息中分享的见解，暗示外部资源可能会提供进一步的澄清。
- **Titans 论文探讨深度学习中的记忆**：**Titans** 论文提议结合短期和长期记忆，利用循环模型（recurrent models）和 Attention 来改进序列处理。
   - 有人提出了 *“在如此庞大的数据集上微调模型难道不更快吗？”* 的疑问，质疑其在不同数据规模下的效率。
- **Linear Attention 中类 Adam 更新的潜力**：讨论围绕 Linear Attention 模型是否需要 **Adam-like update rule**（类 Adam 更新规则）展开，成员们对其实现方式持复杂态度。
   - 成员们担心引入新的缩放方法可能会使学习复杂化，并就这些参数是否依赖于数据分享了见解。
- **Deepseek 奖励模型架构查询**：成员们对 **Deepseek 奖励模型** 的训练过程和架构感到好奇。
   - 一位成员专门询问了相关细节，以便更好地理解其底层机制。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/1606.04474">Learning to learn by gradient descent by gradient descent</a>：机器学习中从手工设计特征到学习特征的转变取得了巨大成功。尽管如此，优化算法仍然是手工设计的。在本文中，我们展示了如何……</li><li><a href="https://arxiv.org/abs/2306.13326">Solving systems of Random Equations via First and Second-Order Optimization Algorithms</a>：基于梯度的（即“一阶”）优化算法通常用于解决大规模非凸问题。然而，通常很难预测它们的有效性。为了获得……</li><li><a href="https://arxiv.org/abs/2501.00663">Titans: Learning to Memorize at Test Time</a>：十多年来，关于如何有效利用循环模型和 Attention 进行了广泛的研究。虽然循环模型旨在将数据压缩到固定大小的内存中……
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1331050552292806799)** (4 messages): 

> `Open Source Steering for LLMs, Current SAE Steering Methods, Open Source Steering Libraries` 


- **目前尚无标准化的 LLM 转向开源方案**：正如成员们提到的，目前还没有一个标准化的开源仓库用于使用从训练好的 **SAE** 中提取的特征来转向（steering）**LLM**。
   - 他们讨论了当前的转向方法，强调目前仍缺乏统一的方法，这阻碍了更广泛的实现。
- **可用的开源转向库**：分享了几个开源转向库，包括 [steering-vectors](https://github.com/steering-vectors/steering-vectors) 和 [repeng](https://github.com/vgel/repeng)。
   - 此外，他们还提到了 [representation-engineering](https://github.com/andyzoujm/representation-engineering) 库，该库专注于通过自顶向下的方法提高 AI 透明度。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://discordapp.com/channels/729741769192767510/1153431135414669422/1321212227881275484">Discord - Group Chat That’s All Fun &amp; Games</a>：Discord 非常适合玩游戏、与朋友闲逛，甚至建立全球社区。自定义你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://github.com/andyzoujm/representation-engineering">GitHub - andyzoujm/representation-engineering: Representation Engineering: A Top-Down Approach to AI Transparency</a>：表示工程：一种实现 AI 透明度的自顶向下方法 - andyzoujm/representation-engineering
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1331029483343315025)** (13 条消息🔥): 

> `4bit/3bit vs f16 性能对比, Qwen R1 模型与 Q-RWKV 转换, 用于评估的 math500 数据集, pass@1 估算方法, R1 的评估模板` 


- **量化方法的性能退化**：一名成员询问了在 **MMLU-PRO** 评估中，对比最近的 **LLaMA** 或 **Qwen** 模型时，**4bit/3bit** 与 **f16** 量化之间的性能退化情况。
   - 他们想知道这种退化是微不足道的，还是取决于量化的力度，并寻求具体信息。
- **探索 Qwen R1 模型转换**：一位用户正考虑将 **Qwen R1 模型** 转换为 **Q-RWKV**，并正在寻找有效的测试方法，以便将结果与基础 **R1 模型** 进行对比。
   - 他们对能否准确评估转换是否成功表示担忧。
- **使用 math500 数据集**：成员们讨论了 **math500**（**Hendrycks MATH 数据集**的一个子集）及其针对 **R1** 等模型的评估方法。
   - 有建议认为切换到 **math500** 进行评估可能非常直接，并强调了集成的简便性。
- **关于响应生成的澄清**：有人提出了一个关于每个查询生成 **64 个响应** 以估算模型评估中 **pass@1** 性能的问题。
   - 成员们讨论了是否可以使用贪婪方法 (greedy methods) 进行此估算过程，并强调需要进一步澄清。
- **R1 模型的评估模板**：一名成员询问 **R1** 模型是否需要不同的对话模板 (chat template)，或者是否可以像基础模型一样进行提示 (prompt)。
   - 这一讨论表明在如何有效利用 **R1** 进行评估方面存在不确定性。



**提及的链接**：<a href="https://huggingface.co/datasets/HuggingFaceH4/MATH-500">HuggingFaceH4/MATH-500 · Datasets at Hugging Face</a>：未找到描述

  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1331226089824981125)** (3 条消息): 

> `中间维度选择, 导出模型为 HF 格式, 模型并行问题` 


- **选择中间维度：3 倍的重要性？**：一名成员请求确认中间维度 (intermediate dimension) 的选择，以及是否出于某种原因需要将其设置为 **3x**。
   - 讨论旨在澄清模型配置中此类参数选择背后的基本原理。
- **将模型转换为 HF 格式时出错**：另一名成员报告在使用 `convert_neox_to_hf.py` 将模型从 **neox 转换为 HF 格式** 时遇到了 `RuntimeError`。错误显示基于提供的形状 `[8, 512, 4096]` 和输入大小 **4194304** 存在维度不匹配。
   - 他们询问了多节点运行转换的可行性，并分享了他们的训练配置详情，寻求社区的进一步意见。
- **训练配置见解**：分享了训练配置文件，展示了如 **4** 的 **model_parallel_size** 和设置为 **32** 的 **num_layers** 等设置。
   - 具体参数包括 **4096** 的 **hidden_size** 和 **8192** 的 **seq_length**，突出了影响导出过程的配置。
- **请求协助解决导出问题**：一名社区成员向另一名成员寻求关于之前提出的导出问题的帮助，确保所提出的问题得到支持。
   - 这种互动强调了 Discord 小组内解决技术挑战的协作努力。



**提及的链接**：<a href="https://rentry.co/f4tvoevf">{</a>: &amp;quot;pipe_parallel_size&amp;quot;: 0,  &amp;quot;model_parallel_size&amp;quot;: 4,  &amp;quot;make_vocab_size_divisible_by&amp;quot;: 1,  # model settings  &amp;quot;num_layers&amp;quot;: 32,  &a...

  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1331003885875105793)** (59 条消息🔥🔥): 

> `Stargate 项目, Gemini 2.0 更新, DeepSeek 见解, Ai2 ScholarQA, WandB SWE-Bench`

- **Stargate 项目投资公告**：OpenAI 宣布了 **Stargate 项目**，旨在四年内投资 **5000 亿美元** 在美国建设 AI 基础设施，首期立即投入 **1000 亿美元**。
   - 该项目得到了 **SoftBank**、**Oracle** 以及其他技术合作伙伴的支持，重点在于确保美国在 AI 领域的领导地位并创造大量就业机会。
- **Gemini 2.0 的实验性更新**：基于对 **Gemini 2.0 Flash Thinking** 的反馈，Noam Shazeer 宣布了根据社区建议进行的实验性更新。
   - 这反映了通过用户见解不断完善 **Gemini** 能力的持续承诺。
- **DeepSeek 在 AI 模型方面的突破**：DeepSeek 在发布 **DeepSeek V2** 模型后引起广泛关注，与行业标准相比，该模型以显著更低的推理成本实现了竞争优势。
   - 该公司的创新架构和 AI 方法在社区内引发了热烈讨论。
- **Ai2 ScholarQA 发布**：**Ai2 ScholarQA** 作为一款面向研究人员的工具推出，它使用最先进的模型，针对需要多篇科学论文才能得出全面答案的问题进行解答。
   - 该平台旨在通过提供对比见解和引用来简化文献综述过程。
- **WandB 获得 SOTA 认证**：WandB 宣布其 **SWE-Bench** 提交已正式被验证为 **State of the Art (SOTA)**。
   - 这一成就凸显了 **SWE-Bench** 基准测试在 AI 社区中的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/hsu_steve/status/1881405336793276874?s=46">来自 steve hsu (@hsu_steve) 的推文</a>：这里可能带点讽刺。他曾就读于中国一所非常优秀的大学（浙江大学），但在国际上并没那么出名。典型的被炒作为“...”的中国研究员...</li><li><a href="https://x.com/ggerganov/status/1881734507575005683">来自 Georgi Gerganov (@ggerganov) 的推文</a>：让你的 Mac 思考得更快 🧠🧠 明天我将向你展示如何取消你的 Copilot 订阅。引用 Georgi Gerganov (@ggerganov)：让你的 Mac 思考 🧠 明天我将向你展示如何启用 sp...</li><li><a href="https://x.com/drjimfan/status/1881382618627019050?s=46">来自 Jim Fan (@DrJimFan) 的推文</a>：今天我没料到会发布*第二篇*包含大量 RL 飞轮秘密和 *multimodal* o1 风格推理的论文。Kimi（另一家初创公司）和 DeepSeek 的论文惊人地趋同...</li><li><a href="https://x.com/btibor91/status/1881285255266750564?s=46">来自 Tibor Blaho (@btibor91) 的推文</a>：OpenAI 网站已经出现了 Operator/OpenAI CUA (Computer Use Agent) 的引用——“Operator 系统卡片表”、“Operator 研究评估表”和“Operator 拒绝率表”...</li><li><a href="https://x.com/nearcyan/status/1773759331403714779)">来自 near (@nearcyan) 的推文</a>：Nvidia ™ 的 Stargate</li><li><a href="https://x.com/kimmonismus/status/1881287794544550018?s=46">来自 Chubby♨️ (@kimmonismus) 的推文</a>：OpenAI 已经有了 OpenAI 的 Operator 与 Claude 3.5 Sonnet CUA 之间的对比（Computer Use Agent）。看来发布在即。引用 Tibor Blaho (@btibor91)：OpenAI 网站已经有了...</li><li><a href="https://x.com/fal/status/1881533663747420364?s=46">来自 fal (@FAL) 的推文</a>：🚨 新模型预警，支持主体引用的 Minimax Video https://fal.ai/models/fal-ai/minimax/video-01-subject-reference</li><li><a href="https://x.com/perplexity_ai/status/1881779310840984043">来自 Perplexity (@perplexity_ai) 的推文</a>：介绍 Sonar：Perplexity 的 API。Sonar 是市场上最实惠的搜索 API 产品。使用它在你的应用中构建由实时信息和引用驱动的生成式搜索。我们...</li><li><a href="https://x.com/fofrai/status/1881452418577404309?s=46">来自 fofr (@fofrAI) 的推文</a>：你现在可以在 Replicate 上使用 Minimax “video-01”（海螺）模型时使用角色参考图像。而且，天哪，效果太棒了。https://replicate.com/minimax/video-01</li><li><a href="https://x.com/legit_rumors/status/1881558479753924708?s=46">来自 ʟᴇɢɪᴛ (@legit_rumors) 的推文</a>：Gemini 2.0 Pro Exp 刚刚在后台被添加 ✨</li><li><a href="https://x.com/allen_ai/status/1881784827063767117">来自 Ai2 (@allen_ai) 的推文</a>：AI 真的能帮上文献综述吗？🧐 认识一下 Ai2 ScholarQA，这是一个实验性解决方案，允许你提出需要多篇科学论文才能回答的问题。它提供更深入、更...</li><li><a href="https://x.com/openai/status/1881830103858172059?s=46">来自 OpenAI (@OpenAI) 的推文</a>：宣布 Stargate 项目。Stargate 项目是一家新公司，计划在未来四年内投资 5000 亿美元，在美国为 OpenAI 建设新的 AI 基础设施。我们将...</li><li><a href="https://x.com/dhravyashah/status/1881510837132906840?s=46">来自 Dhravya Shah (@DhravyaShah) 的推文</a>：Supermemory v2 现已开源！这次全新的更新是使用 Remix 构建的，可能是唯一的开源大型 Remix 应用。还有很多其他酷炫的东西 + RAG 流水线。全部完全开源。-> http...</li><li><a href="https://x.com/zizhpan/status/1881727148081517050">来自 Zizheng Pan (@zizhpan) 的推文</a>：兄弟...这家伙不是我们的文峰 🥲。他只是另一个在百度上能找到的同名中国人。引用 Henry Shi (@henrythe9ths)：DeepSeek 刚刚发布了第一个开源推理...</li><li><a href="https://x.com/dwarkesh_sp/status/1881844437346902297">来自 Dwarkesh Patel (@dwarkesh_sp) 的推文</a>：.@dylan522p 在 2024 年 10 月就预言了。引用 OpenAI (@OpenAI)：宣布 Stargate 项目。Stargate 项目是一家新公司，计划在未来四年内投资 5000 亿美元建设新的...</li><li><a href="https://scholarqa.allen.ai/">Ai2 ScholarQA</a>：未找到描述</li><li><a href="https://scholarqa.allen.ai/query/9d8946c0-756c-4148-b32e-c2d5bc8f8b09">Ai2 ScholarQA</a>：未找到描述</li><li><a href="https://bsky.app/profile/colin-fraser.net/post/3ldoyuozxwk2x">Colin (@colin-fraser.net)</a>：在我看来，这就是为什么 LLM 的“对齐研究”是一团糟。Claude 不是一个真实的人。Claude 是 LLM 被编程去编写的故事中的一个角色。...</li><li><a href="https://allenai.org/blog/ai2-scholarqa">介绍 Ai2 ScholarQA | Ai2</a>：Ai2 ScholarQA 提供深入、详尽且具上下文的答案，以协助文献综述。</li>

x.com/sama/status/1881851602727993711?s=46">来自 Sam Altman (@sama) 的推文</a>：在沙漠中建造纪念碑</li><li><a href="https://www.liquid.ai/lfm-7b">介绍 LFM-7B：为高效语言模型设定新标准</a>：全球顶级的英语、阿拉伯语和日语模型，原生支持法语、德语和西班牙语，经过优化，可作为私有企业聊天、代码、快速指令遵循等的底层架构...</li><li><a href="https://x.com/shawnup/status/1881458032741400758?s=46">来自 Shawn Lewis (@shawnup) 的推文</a>：我们的 SWE-Bench 提交已被接受并正式成为 SOTA！感谢 SWE-Bench 团队制定了如此重要的基准测试。</li><li><a href="https://x.com/noamshazeer/status/1881845900659896773?s=46">来自 Noam Shazeer (@NoamShazeer) 的推文</a>：你们对 Gemini 2.0 Flash Thinking 的反馈非常棒——谢谢！我们采纳了你们的建议并进行了一次实验性更新……</li><li><a href="https://www.youtube.com/watch?v=zDo_RrzdRoQ">唐纳德·特朗普总统宣布 AI 基础设施投资 — 2025年1月21日</a>：唐纳德·特朗普总统周二宣布与 OpenAI、Oracle 和 Softbank 成立合资企业，在美国投资数十亿美元建设 AI 基础设施……</li><li><a href="https://github.com/MoonshotAI/Kimi-k1.5">GitHub - MoonshotAI/Kimi-k1.5</a>：通过在 GitHub 上创建账号，为 MoonshotAI/Kimi-k1.5 的开发做出贡献。</li><li><a href="https://x.com/Kimi_ai_/status/1881332472748851259">来自 Kimi.ai (@Kimi_ai_) 的推文</a>：🚀 介绍 Kimi k1.5 —— 一个 o1 级别的多模态模型。具备 SOTA 级别的短 CoT 性能，在 📐AIME、📐MATH-500、💻 LiveCodeBench 上大幅超越 GPT-4o 和 Claude Sonnet 3.5（最高达 +550%...）</li><li><a href="https://mp.weixin.qq.com/s/r9zZaEgqAa_lml_fOEZmjg">揭秘DeepSeek:一个更极致的中国技术理想主义故事</a>：做贡献者，而非搭便车者。
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1331187377799954464)** (1 条消息): 

> `Last Week in AI, Gmail 中的免费 AI` 


- **在 Last Week in AI 播客担任客座主持**：一位成员宣布他们客座主持了一期 [Last Week in AI](https://www.listennotes.com/podcasts/last-week-in-ai/197-free-ai-in-gmail-minimax-fCdt-x_RXAF/)，讨论了 Gmail 中免费 AI 功能的集成。
   - 该剧集探讨了直接应用于电子邮件通信的 AI 工具的最新进展及其影响。
- **聚焦 Gmail AI 功能**：播客还重点介绍了目前 Gmail 中可用的**免费 AI** 功能，强调了它们在提升用户体验方面的潜力。
   - 听众对这些创新如何简化邮件管理并提高生产力特别感兴趣。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1330998289822253178)** (44 messages🔥): 

> `DeepSeek R1 性能、生成式 AI 对创意产业的影响、AI 模型对比、本地模型运行能力、AI 输出合规性问题` 


- **DeepSeek R1 在本地使用方面表现出潜力**：用户讨论了蒸馏到 [Qwen 32B Coder](https://link.to.related.info) 的 **DeepSeek R1** 是一个值得在本地运行的模型，但对其在 Ollama 上的性能提出了疑问，因为有报告称存在问题。
   - 一位拥有 **32 GB RAM** 和 **16 GB VRAM** 的用户解释说，他们正在一个将繁重计算卸载到 CPU 的系统上运行它。
- **创意领域生成式 AI 的未来**：成员们分享了他们对生成式 AI 在创意产业快速增长的看法，一些人认为它最终可能会取代艺术家和创意专业人士。
   - 有人对 AI 生成艺术的准确性以及有效引导输出所需的人类技能的必要性表示担忧。
- **编程中的 R1 vs. O1 和 Sonnet**：对比了 R1、O1 和 Sonnet 3.5 在编程和数学方面的能力，指出 R1 在一个特定项目中有 **60% 的失败率**。
   - 相比之下，据报道 **4O 和 Sonnet** 的失败率为 **99%**，展示了不同模型之间的性能差异。
- **AI 输出合规性的挑战**：有人指出 **DeepSeek** 倾向于避免生成关于 **CCP** 的批评或幽默内容，类似于 GPT 过去关于 ESG 合规问题的输出。
   - 这引发了关于 AI 生成内容中表达和辩论影响的问题。
- **AI 训练结果的推测性未来**：一位用户推测 AI 公司可能会无意中训练出类似于 **Rimworld 中 archotechs** 的模型，想象其具有不可预见的能力。
   - 这种推测反映了对 AI 发展方向更广泛的担忧。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://medium.com/@techyspacelovers/generative-ai-how-its-shaping-creative-industries-f3e11960fe38">Generative AI: How It’s Shaping Creative Industries</a>: 我们都使用过并听说过像 Chatgpt 这样的工具。它们目前在科技界风靡一时。我们也听说过 AI 工具如何……</li><li><a href="https://www.dynocortex.com/news-and-blog/good-kg-created-by-ai/">来自推文的链接 
        
            截至 2025 年 1 月，AI 能否自动创建一个好的知识图谱？ - 
        
    </a>: 未找到描述
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1331115380579631154)** (7 messages): 

> `GPT 停机问题、聊天响应延迟` 


- **频繁的 GPT 停机担忧**：用户报告了 GPT 的频繁问题，包括像 *'Something went wrong. If this issue persists please contact us...'* 这样中断聊天的消息。
   - 一位用户提到重新打开聊天通常可以解决问题，表明这可能不是永久性问题。
- **GPT 性能变慢**：另一位成员指出 GPT 最近变得 *非常慢*，导致交互过程中出现令人沮丧的体验。
   - 几位用户也表达了同样的看法，表明存在影响响应时间的更广泛的性能问题。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 messages): 

oneidemaria: <:dallestar:1006520565558956092>
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 messages): 

oneidemaria: <:dallestar:1006520565558956092>
  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1330999794948378705)** (46 条消息🔥): 

> `Neural ODE 应用、ML 中的建模与算法选择、小模型的 RL 技术、RL 中的探索策略、MoE 与 Attention Mechanisms` 


- **Neural ODEs 能够变革机器人学**：成员们讨论了 Neural ODEs 在机器人学中的适用性，重点在于它们根据函数复杂度和算法决策来模拟层的能力。
   - 一位成员强调了在不同层注入知识对于解决小模型局限性的重要性。
- **平衡建模与算法选择**：讨论集中在平衡建模决策（如在 nonparametric、Bayesian 或 NN 方法之间选择）与 ML 的算法层面。
   - 推理路径的质量和 loss functions 的选择被认为是成功实现 ML 模型的关键因素。
- **探索小模型的 RL 策略**：有一种假设认为，小模型可以通过重复的随机初始化和进化技术发现高质量的推理路径。
   - 成员们辩论了这些策略的可行性，并对这些方法的有效性和复制能力表示担忧。
- **RL 中不规则性的重要性**：Red_code 强调了在 RL 过程中引入噪声和不规则性的重要性，同时惩罚规则性以增强训练期间的探索。
   - 提议的策略包括直接 logits 采样和避免使用 softmax，以保留培养高质量推理所需的细微差别。
- **MoE 与 Attention Mechanisms**：有人提出 MoE 是否可以被视为一种没有 key mapping 的基础 Attention 形式，成员们讨论了它们实现的复杂性。
   - 讨论指向了不同架构之间的交互，以及建模选择对开发更有效的 ML 系统的影响。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/crosswind-landing-gif-20167802">Crosswind Landing GIF - Crosswind Landing - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/moonflip-mflip-pepe-dance-moon-dance-gif-24962206">Moonflip Mflip GIF - Moonflip Mflip Pepe Dance - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/artoria-altria-arturia-king-arthur-pendragon-gif-21735401">Artoria Altria GIF - Artoria Altria Arturia - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=jltgNGt8Lpg">Neural Ordinary Differential Equations</a>: https://arxiv.org/abs/1806.07366 摘要：我们引入了一类新的深度神经网络模型。与其指定离散的隐藏层序列，不如……</li><li><a href="https://youtu.be/ZTNej2USaYk?t=106">La Passion</a>: 由 ZYX Music 提供给 YouTube 的 La Passion · Gigi D'Agostino L'Amour Toujours ℗ ZYX Music 发布于：2000-01-10 作曲：Di Agostino, L. 音乐出版商：Medi...
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1331041430075867206)** (4 条消息): 

> `DeepSeek 的 Group Relative Policy Optimization (GRPO)，评审流程挑战，作者与评审者的协作` 


- **理解 AI 优化中的 GRPO**：[DeepSeek 的 Group Relative Policy Optimization (GRPO)](https://fixupx.com/natolambert/status/1881380809153847711) 被强调为一种没有价值函数（value function）的 PPO，它使用蒙特卡洛估计（Monte Carlo estimates）来计算优势（advantages），从而简化了模型的复杂度。
   - *理解 PPO 的存在至关重要*，特别是考虑到大型语言模型（LLM）中价值函数的复杂性。
- **基于平均奖励的策略优化**：论文讨论了 GRPO 如何消除 PPO 所需的额外价值函数近似，转而利用多个采样输出的平均奖励。
   - 正如[最近发表的论文](https://arxiv.org/abs/2402.03300v3)所述，这种对 GRPO 的见解表明在优化 AI 模型策略方面效率有所提高。
- **会议论文评审中的挑战**：有人担心每篇论文需要更多作者担任评审员，因为单名评审员负担过重会导致评估不充分。
   - 一位参与者分享道，为了确保每篇投稿都能获得三个高质量的评审意见，必须从 **50 多名感兴趣的人**中招募 **12 名评审员**。
- **克服评审员短缺**：强调了广泛联络的必要性，一位用户提到发送个性化消息是他们争取高质量评审努力的一部分。
   - 尽管付出了这些努力，他们仍觉得有必要亲自评审多篇投稿，这反映了评审系统面临的**巨大压力**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fixupx.com/natolambert/status/1881380809153847711">来自 Nathan Lambert (@natolambert) 的推文</a>：对于那些试图理解 DeepSeek 的 Group Relative Policy Optimization (GRPO) 的人：GRPO 就是没有价值函数、使用优势函数的蒙特卡洛估计的 PPO。所以，去研究为什么 PPO 存在（lo...</li><li><a href="https://arxiv.org/abs/2402.03300v3">DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models</a>：由于数学推理的复杂性和结构化特性，它对语言模型构成了重大挑战。在本文中，我们介绍了 DeepSeekMath 7B，它继续对 DeepSeek-Co 进行预训练...
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/)** (1 条消息): 

rogerngmd: https://github.com/deepseek-ai/DeepSeek-R1/blob/main/README.md
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1331303581487337474)** (1 条消息): 

> `Suno AI 音乐生成器，版权侵权诉讼，音乐行业争议` 


- **Suno 面临来自 GEMA 的新版权诉讼**：估值 **5 亿美元**的 AI 音乐生成器 **Suno** 遭到了德国授权机构 **GEMA** 的版权侵权诉讼。
   - 此前，Suno 曾因未经许可使用曲目被大型唱片公司起诉，而他们在法庭文件中基本上承认了这一点。
- **围绕 AI 音乐生成的持续争议**：Suno 与另一家 AI 公司 **Udio** 一起陷入法律纠纷，被指控在未经授权的录音上训练其系统。
   - 尽管面临指控，两家公司都为自己的行为辩护，引发了**音乐行业**关于 AI 生成内容合法性的持续辩论。



**提到的链接**：<a href="https://www.musicbusinessworldwide.com/500m-valued-suno-hit-with-new-copyright-lawsuit-from-germanys-gema/">估值 5 亿美元的 Suno 遭到德国 GEMA 的新版权诉讼 - Music Business Worldwide</a>：GEMA 代表德国约 95,000 名成员（作曲家、词作者、音乐出版商）以及全球 200 多万权利持有人的版权。

  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1330991616458100796)** (14 messages🔥): 

> `编程语言偏好、社区展示讨论、Mojo 进度更新` 


- **学习编程时 C 与 Python 的对比**：成员们辩论了从 **C** 还是 **Python** 开始学习的优劣，一些人认为无论未来是否从事 **JS** 或 **Python** 开发，**C** 都有助于理解内存管理。
   - 一位成员强调，从 **C** 开始可以培养纪律性，特别是对于那些在人生后期考虑转行的人。
- **多平台社区展示**：讨论了在 **Discord** 频道和 **Forum**（论坛）同时宣传项目的问题，建议论坛更适合长期讨论。
   - 成员们表示有必要明确每个平台适合的内容类型，以减少重复。
- **关于 Forum 与 Discord 沟通的反馈**：分享了关于对话节奏的看法，一些成员更喜欢 **Forum**，因为与快节奏的 **Discord** 相比，它的信息交换更慢、更易处理。
   - 有人指出，Discord 中的重要讨论以后很难找到，建议混合使用以保持对话的有条理。
- **当前 Mojo 开发进度**：一位成员询问了 **Mojo** 的生产环境使用情况，以及用户注意到的任何优缺点。
   - 另一位成员确认正在取得进展，并提到 **Nightly** 版本正在积极开发中。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1330995578355843112)** (8 messages🔥): 

> `Mojo 项目的 .gitignore、Netlify 与 Mojo 应用的兼容性、Mojo 组织域名讨论` 


- **Mojo 的极简 .gitignore 文件**：`Mojo` 项目初始化时的 `.gitignore` 主要忽略与 **magic** 相关的文件，包括 `.pixi` 和 `.magic`。
   - 这种极简主义符合社区的普遍预期。
- **关于 Mojo 和 Netlify 托管的困惑**：一位成员询问使用 `lightbug_http` 的 `Mojo` 应用是否可以托管在 **Netlify** 上，并提到 Rust 应用已成功托管。
   - 另一位成员指出，这取决于 Netlify 构建镜像支持的语言，暗示目前尚未包含 Mojo，但提交功能请求可能会有帮助。
- **关于 Mojo 域名存在感的讨论**：有人询问 **Mojo** 是否会像其他编程语言一样拥有一个带有 `.org` 域名的独立组织。
   - 澄清目前没有计划让 Mojo 从 **Modular** 拆分，也没有计划更改当前的 modular.com 域名。



**提到的链接**：<a href="https://docs.netlify.com/configure-builds/available-software-at-build-time/">构建时的可用软件</a>：了解构建时可用于构建的软件和工具。

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1331004466312515727)** (2 messages): 

> `LlamaIndex Workflows, Chat2DB GenAI Chatbot` 


- **在 Google Cloud Run 上部署 LlamaIndex Workflows**：本指南将引导你[设置一个双分支 RAG 应用程序](https://t.co/nU1BctUh7s)用于 ETL 和查询处理，利用 LlamaIndex 的事件驱动框架构建灵活的 AI 系统。
   - 它还涵盖了[利用 Google Cloud](https://t.co/AdynRZ79jn) 进行部署的内容。
- **Chat2DB GenAI Chatbot 简化数据交互**：开源的 [Chat2DB genai chatbot](https://t.co/l1SFCEkiOC) 允许使用日常语言查询数据库，具有 RAG 和 TAG 等多种交互方式。
   - 主要优势包括支持多种 LLM 提供商（如 OpenAI 和 Claude），使其成为数据访问的通用工具。

  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1331254300747366481)** (18 messages🔥): 

> `LlamaParse 文档解析器，LlamaIndex 文档网站 Bug，结合 Gemini 的 Cached Augmented Generation` 


- **针对 PDF 提取问题推荐使用 LlamaParse**：成员建议使用 [LlamaParse](https://docs.llamaindex.ai/en/stable/llama_cloud/llama_parse/) 来有效解析具有可选文本的 PDF，并强调了其在数据清洗方面的强大功能。
   - *LlamaParse* 被誉为“世界上第一个专为 LLM 使用场景量身定制的生成式 AI 原生文档解析平台”。
- **用户报告 LlamaIndex 文档 Bug**：一位用户反映 [LlamaIndex 文档](https://docs.llamaindex.ai/) 在浏览时会随机滚动回顶部。
   - 该用户正在通过 Microsoft Edge 的无痕模式进行测试排查，怀疑可能是插件导致了该 Bug。
- **无痕模式似乎解决了 LlamaIndex 浏览问题**：用户确认在笔记本电脑上使用无痕模式访问文档时没有出现滚动问题，这是一个积极的发现。
   - 另一位成员提到他们在使用 Edge 时没有遇到类似问题，因为它的表现通常与 Chrome 一致。
- **讨论结合 Gemini 实现 CAG**：一位成员询问了如何结合 Gemini 实现 Cached Augmented Generation (CAG)，但被告知需要模型级访问权限。
   - 对方澄清说，目前没有任何模型提供商通过 API 提供此类实现所需的访问级别。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/llama_cloud/llama_parse/">LlamaParse - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/">LlamaIndex - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1330997153925042229)** (20 messages🔥): 

> `ModernBert 的实体识别，Jinja 模板见解，LMstudio 咨询，Adobe Photoshop 支持，Nomic 税务` 


- **ModernBert 中识别实体的语法**：一位用户询问了在 ModernBert 中识别实体的语法，并分享了一个关于旅游主题的层级文档布局示例。
   - 他们对任何关于带有实体的文档 Embedding 的最佳实践表示感谢。
- **Jinja 模板的最佳资源**：一位成员请求推荐讲解 Jinja 模板酷炫功能和特性的网站。
   - 这引起了其他希望加强对 Jinja 了解的用户的兴趣。
- **寻找 LMstudio Discord**：一位用户询问是否可以在频道中咨询 LMstudio，并提到他们找不到专门的 Discord 链接或频道。
   - 另一位用户回应并寻求关于 Adobe Photoshop 的通用支持，凸显了非官方支持咨询的趋势。
- **针对违规问题的幽默回应**：讨论围绕一个用户可能询问的与 Adobe Photoshop 相关的违规问题展开，引发了关于所获回复的幽默交流。
   - 这引发了关于询问非法信息的社会影响的评论。
- **针对实习生的 Nomic 税务**：关于 Nomic 加税的一个幽默记录，一位成员开玩笑说税款应该支付给他们自己。
   - 随后是一个引用实习生分配的轻松 GIF，展示了社区的打趣氛围。



**提及的链接**：<a href="https://tenor.com/view/willj-oprah-oprah-winfrey-winfrey-you-get-a-car-gif-2219821026349492069">Willj Oprah GIF - Willj Oprah Oprah Winfrey - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1331021168194486274)** (5 条消息): 

> `Bud-E 语言能力、Suno Music 音频输入功能、当前工作项目延迟` 


- **Bud-E 支持 13 种语言**：成员确认 Bud-E 不仅限于英语，因为它可以使用 fish TTS 支持 **13 种语言**。
   - *未提供支持语言的具体列表。*
- **当前项目因专注于音视频而冻结**：一位成员询问了项目状态，据指出，由于工作重点转向音频和视频数据集，该项目目前处于“冻结”状态。
   - *对这些数据集的关注导致了开发延迟。*
- **Suno Music 赋能音乐创作**：[Suno Music](https://x.com/SunoMusic/status/1881742789639057828) 允许用户通过录制各种声音或音乐输入来创作自己的歌曲，从而获得个性化体验。
   - 一位成员对该功能表示兴奋，并指出其在移动设备上的广泛可用性。



**提到的链接**：<a href="https://x.com/SunoMusic/status/1881742789639057828">来自 Suno (@SunoMusic) 的推文</a>：录制你唱歌、弹钢琴或敲铅笔的声音 + 上传到 Suno，用你自己的声音创作你自己的歌曲 😱 你用我们的音频输入功能创作了什么？🎤：@techguyver 展示了...

  

---


### **LAION ▷ #[announcements](https://discord.com/channels/823813159592001537/826154622644649985/1331020954779910204)** (1 条消息): 

> `BUD-E, School-BUD-E, 开源语音助手, AI 教育助手框架` 


- **LAION 发布 BUD-E 和 School-BUD-E 语音助手**：今天，LAION 自豪地宣布发布 **BUD-E** 1.0 版本，这是一个 **100% 开源**的语音助手，集成了 **Google AI Studio** 和 **Deepgram** 等多种平台。
   - 此次发布标志着通过技术实现教育民主化和共情迈出了重要一步，其版本涵盖了通用和教育用途。
- **BUD-E 框架旨在实现全民普及**：**BUD-E**（全称 Buddy for Understanding and Digital Empathy）力求为每个人提供免费、智能的教育助手，无论身在何处。
   - 此次发布包含不同的版本，例如用于教育场景的 **School Bud-E** 和作为智能家居助手替代品的 **Desktop Bud-E**。
- **BUD-E 功能概览**：最近推出的 BUD-E 包含专为**教育**和**通用目的**设计的功能，提供用户友好的界面以实现无缝交互。
   - 现已提供教程和演示，包括一段介绍其功能的详细 [YouTube 视频](https://www.youtube.com/watch?v=IxHnpISMNPo)。
- **通过多样化平台实现可访问性**：BUD-E 兼容自托管 API 并支持多种技术，允许在用户浏览器中进行本地数据存储，增强了**隐私合规性**。
   - LAION 强调其对灵活性的承诺，通过 Web 和桌面平台提供访问，让技术教育对每个人都触手可及。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://laion.ai/blog/bud-e-release/">Introducing BUD-E 1.0: AI-Assisted Education for Everyone | LAION</a>：&lt;p&gt;今天标志着我们在通过技术实现教育民主化和共情的旅程中迈出了里程碑式的一步。LAION e.V. &lt;em&gt;非常激动地宣布发布...</li><li><a href="https://www.youtube.com/watch?v=y4DRYF9sfMU">School Bud-E - Overview</a>：英文版：https://school.bud-e.ai/?lang=en 德文版：https://school.bud-e.ai/?lang=de 代码：https://github.com/LAION-AI/school-bud-e-frontend Bud-E (g...</li><li><a href="https://www.youtube.com/watch?v=IxHnpISMNPo">School BUD-E &amp; Bud-E: How to use our open, browser-based Voice Assistants (Tutorial)</a>：关于如何使用 School Bud-E &amp; Bud-E 的教程。:) 英文版：https://school.bud-e.ai/?lang=en 德文版：https://school.bud-e.ai/?lang=de Bud-E (general ...
</li>
</ul>

</div>
  

---

### **LAION ▷ #[resources](https://discord.com/channels/823813159592001537/991938328763056168/1331299207465930832)** (1 messages): 

> `IPTVPlayer, AtlasVPN, TradingView-Premium, Cʀᴀᴄᴋɪɴɢ Cʟᴀss` 


- **群组中推荐的顶级 Repack**：该群组 24/7 全天候推广可用的最佳 Repack，并强调其对成员的独占性。
   - 鼓励成员查看 Telegram 频道，获取最新产品和最佳免费程序的更新。
- **通过高级优惠让交易变得简单**：**TradingView-Premium** 的促销活动声称通过无与伦比的优惠帮助用户成为真正的交易者。
   - 该频道强调了获取高级交易工具对市场成功的重要性。
- **加入 Cracking Class 获取免费程序**：**Cʀᴀᴄᴋɪɴɢ Cʟᴀss** 聊天室拥有 **64,400** 名订阅者，分享最佳的免费程序。
   - 敦促成员通过 Telegram 加入，以便立即获取讨论和资源。



**Link mentioned**: <a href="https://t.me/repackMEMII">Cʀᴀᴄᴋɪɴɢ Cʟᴀss [ᴄʜᴀᴛʀᴏᴏᴍs]</a>: 最好的程序只有免费的

  

---


### **LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1331299215174799453)** (1 messages): 

> `IPTVPlayer, AtlasVPN, TradingView-Premium, Cʀᴀᴄᴋɪɴɡ Cʟᴀss, Free Programs` 


- **独家 IPTV Repack 优惠**：鼓励成员查看 **IPTVPlayer** 的最佳 Repack，以及仅在 [group](https://t.me/repackMEMII) 中 24/7 提供的其他软件。
   - 该群组声称提供 **AtlasVPN** 和 **TradingView-Premium** 等工具的高级访问权限，吸引有抱负的交易者。
- **加入 Cracking 社区**：名为 **Cʀᴀᴄᴋɪɴɡ Cʟᴀss** 的频道拥有 **64,400** 名订阅者，推广免费获取最佳程序。
   - 邀请用户直接通过 Telegram 加入频道，以获取社区资源和讨论。



**Link mentioned**: <a href="https://t.me/repackMEMII">Cʀᴀᴄᴋɪɴɢ Cʟᴀss [ᴄʜᴀᴛʀᴏᴏᴍs]</a>: 最好的程序只有免费的

  

---


### **LAION ▷ #[paper-discussion](https://discord.com/channels/823813159592001537/1172520224797511700/1331299593186574359)** (1 messages): 

> `IPTVPlayer offerings, AtlasVPN promotions, TradingView Premium features` 


- **解锁 IPTVPlayer 优惠**：该群组正在推广 **IPTVPlayer**，提供 **24/7** 全天候可用的独家 Repack，为用户提供独特的机会。
   - 鼓励成员通过频道链接查看优惠。
- **发现 AtlasVPN 交易**：该频道重点介绍了高级 **AtlasVPN** 交易，旨在为想要保护其在线活动的用户增强安全性。
   - 感兴趣的参与者可以在专门的 Telegram 频道上找到更多详细信息。
- **TradingView Premium 洞察**：**TradingView-Premium** 服务被宣传为成为真正交易者的必备工具，提供最佳的市场分析优惠。
   - 也可以通过提供的 Telegram 链接访问此优惠的信息。
- **加入 Cracking Class 获取免费软件**：**Cʀᴀᴄᴋɪɴɢ Cʟᴀss** 聊天室拥有 **64,400 名订阅者**，承诺提供一系列免费程序。
   - 欢迎成员加入聊天室以有效地获取这些资源。



**Link mentioned**: <a href="https://t.me/repackMEMII">Cʀᴀᴄᴋɪɴɢ Cʟᴀss [ᴄʜᴀᴛʀᴏᴏᴍs]</a>: 最好的程序只有免费的

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1331251453473587262)** (6 messages): 

> `Declaration Form Requirement, Corporate Sponsors and Intern-like Tasks, New MOOC Syllabus Release` 


- **关于声明表（Declaration Form）要求的澄清**：*一位成员询问由于他们在 12 月已经提交过，是否需要再次填写声明表。* 澄清说明该表格现在是为那些错过最初提交截止日期的人准备的。
- **关于企业赞助商提供类实习任务的咨询**：*一位成员表示有兴趣了解企业赞助商是否会在下一个 MOOC 中提供类实习任务。* 有人指出，上学期的 Hackathon 项目起到了这个作用，尽管演讲者可能会提到实习机会。
- **新 MOOC 教学大纲发布时间**：*一位成员询问了新 MOOC 教学大纲的发布日期。* 回复指出团队目前正在敲定演讲者，预计将在 **1 月 27 日**之前发布一份初步大纲。

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1330992719820099697)** (5 messages): 

> `BEAM 性能、WebGPU 兼容性、YoloV8 FPS 问题` 


- **BEAM 大幅降低了 YoloV8 的性能**：一位用户报告称，在执行 `python examples/webgpu/yolov8/compile.py` 时使用 **BEAM** 会导致性能从 **40fps** 骤降至 **8fps**。
   - *georgehotz* 认为这种行为表明存在一个 **bug**，并指出 BEAM 不应该让性能变差。
- **WebGPU 与 BEAM 的兼容性疑虑**：另一位用户推测 BEAM 可能无法很好地与 **WGSL** 配合工作，因为它需要额外的编译步骤转换为 **SPIR-V** 或特定平台的语言。
   - 这引发了关于额外的编译步骤是否对于有效性能而言过于缓慢的疑问。
- **关于 BEAM 后端细节的讨论**：一位用户提到 BEAM 需要在确切的后端和硬件上使用，这表明其与 **WebGPU** 存在兼容性问题。
   - 这引起了关于在切换渲染目标时 BEAM 性能转换的担忧。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1331327977975320619)** (1 messages): 

> `关于 tune cat 命令的提案、TRL 帮助命令的长度` 


- **对 `tune cat` 命令提案感到兴奋**：一位成员表达了对 **Torchtune** 包的赞赏，并分享了一个关于提议 `tune cat` 命令的 [GitHub Issue](https://github.com/pytorch/torchtune/issues/2281)。
   - 阅读源代码是 *一种绝对的享受*，表明了整体良好的用户体验。
- **TRL 帮助命令的超长篇幅**：另一位成员幽默地评论了 **TRL** 帮助命令的长度，指出它横跨了 **三个终端窗口**。
   - *这个功能虽然让人应接不暇，但对用户来说至关重要。*



**提到的链接**：<a href="https://github.com/pytorch/torchtune/issues/2281">[RFC] Proposal for `tune cat` Command · Issue #2281 · pytorch/torchtune</a>：首先，非常感谢这个精彩的软件包。我开始积极查看源代码，我必须说阅读它绝对是一种享受。我很难让自己停下来……

  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1331001537811578973)** (4 messages): 

> `量化 LLM 中的不确定性、LLM 中的思维链 (CoT)、RL-LLM 指令提示词、RL 中的蒸馏` 


- **模型应该量化不确定性**：*一位成员建议*，模型应该能够在一定程度上利用现有方法量化不确定性，从而增强其可靠性。
   - 这一概念旨在提高 LLM 输出的可解释性和置信度。
- **LLM 进行自我思维链 (self-cot)**：*另一位成员注意到*，感觉 LLM 在提供答案之前正在进行自己的思维链 (CoT)，这增加了生成响应的深度。
   - 这一观察强调了 LLM 在做出陈述前进行内部推理的潜力。
- **需要 RL-LLM 思考步骤提示词**：*有人建议*在 RL-LLM 系统中增加思考步骤的指令提示词，作为现有目标设定提示词的补充。
   - 这种添加可以增强模型的推理过程，从而产生更明智的输出。
- **在蒸馏基础上改进 RL**：*另一位成员指出*，RL 技术仍然可以应用在模型蒸馏之上，这可能会带来进一步的改进。
   - 观察较小的模型是否通过这种方法表现出显著增强将会非常有趣。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/)** (1 messages): 

moresearch_: 基于 DSPy 的 RAG 如何处理动态数据？
  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1331277034214658099)** (2 messages): 

> `开放问题、语法拼写错误` 


- **开放问题仍在讨论中**：一位用户询问某个特定问题是否仍是开放问题，表明了对该问题解决进度的持续关注。
   - 这突显了社区在故障排除和问题解决方面的参与度。
- **在语法中发现拼写错误**：另一位用户确认该工作运行正常，但指出了一处拼写错误，提到 'y=y' 应该包含一个数字。
   - *如果不处理，这种差异可能会导致混淆，强调了讨论中对细节的关注。*


  

---

### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1331338533604163638)** (1 条消息): 

> `LLM 训练的开源数据集，Mozilla 与 EleutherAI 的合作伙伴关系` 


- **开源数据集最佳实践发布**：题为 [Towards Best Practices for Open Datasets for LLM Training](https://discord.com/channels/1089876418936180786/1331335526338265108) 的论文已在 Arxiv 上发表，旨在解决开源 AI 数据集中的挑战。
   - 该论文提供了具体建议，以促进 AI 生态系统的公平性与透明度。
- **Mozilla 与 EleutherAI 数据集召集合作伙伴关系**：Mozilla 与 **[EleutherAI](https://discord.gg/cJQKYFDwHV)** 合作举办了数据集召集会议（Dataset Convening），重点关注负责任的数据策展（Data Curation）与治理。
   - 参与的关键利益相关者包括致力于增强 AI 开源数据环境的社区成员。


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1331296777600630928)** (1 条消息): 

> `网络安全中的 AI，AI 对安全团队的影响` 


- **AI 进军网络安全**：一位成员回顾了他们一年前向 AI 领域的及时转型，并指出在此之前，AI 在网络安全产品中更像是一个*流行语 (buzzword)*。
   - 他们对 AI 未来真正协助安全团队的潜力感到兴奋。
- **未来 AI 对安全团队的贡献**：讨论强调了人们对于 AI 如何真正提高未来安全团队效率的日益关注。
   - 成员们期待随着 AI 进一步融入安全流程，将取得重大进展。


  

---


---


---


---


---


{% else %}


> 完整的逐频道详情已因邮件篇幅原因截断。 
> 
> 如果您想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}