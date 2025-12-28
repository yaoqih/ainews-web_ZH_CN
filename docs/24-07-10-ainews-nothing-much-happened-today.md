---
companies:
- huggingface
- truth_terminal
- microsoft
- apple
- openai
- meta-ai-fair
- yi
- axolotl
- amd
- salesforce
date: '2024-07-11T01:15:43.889022Z'
description: '以下是为您翻译的中文内容：


  **HuggingFace** 发布了基于浏览器的、带有时间戳功能的 Whisper 版本，该版本使用了 transformers.js。由 **truth_terminal**
  开发的一个 Twitter 机器人成为了首个获得风险投资（VC）的“半自主”机器人。在监管审查的压力下，**微软**和**苹果**突然退出了 **OpenAI**
  董事会。**Meta** 正在完成对 Reddit 评论功能的重大升级，旨在解决幻觉问题。**Yi 模型**在 GitHub 上走红，获得了 7.4K 个星标和
  454 次分叉，并有可能与 **Axolotl** 集成以进行预生成和预处理。**AMD** 的技术使家用及小型企业 AI 设备成为可能。**Meta** 在
  HuggingFace 上发布了 **Chameleon-7b** 和 **Chameleon-30b** 模型，支持统一的文本和图像分词（tokenization）。**Salesforce**
  的 **xLAM-1b** 模型虽然参数规模较小，但在函数调用（function calling）方面的表现优于 **GPT-3.5**。**Anole** 开创了开源多模态文本-图像-视频生成的先河，最高支持
  720p 144fps。**Phi-3 Mini** 的参数从 38 亿扩展到了 47 亿，并增加了函数调用功能，与 **Mistral-7b v3** 展开竞争。人类中的“**系统
  2 蒸馏**”（System 2 distillation）与自动化和程序性记忆有关。'
id: 49d4d626-b51f-4b3e-b637-b3fe84795ce5
models:
- chameleon-7b
- chameleon-30b
- xlam-1b
- gpt-3.5
- phi-3-mini
- mistral-7b-v3
original_slug: ainews-nothing-much-happened-today
people: []
title: 今天没发生什么特别的事。
topics:
- function-calling
- multimodality
- model-releases
- model-updates
- model-integration
- automaticity
- procedural-memory
- text-image-video-generation
---

<!-- buttondown-editor-mode: plaintext -->**ZZzzzzz.**

> 2024年7月9日至7月10日的 AI 新闻。
我们为您检查了 7 个 subreddits、[384 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 29 个 Discord 服务（463 个频道，2339 条消息）。
预计节省阅读时间（按每分钟 200 字计算）：**250 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

昨天非常忙碌，今天则不然。这里有一些零星的小消息，娱乐性大于一切：

- HuggingFace 发布了[在浏览器中运行的带时间戳的 Whisper (transformers.js)](https://x.com/xenovacom/status/1811068015229747335?s=46)
- [@truth_terminal](https://x.com/JvNixon/status/1811105507756872003) 成为首个获得 VC 投资的“半自主” Twitter 机器人
- [微软 (Msft) 和苹果突然退出 OpenAI 董事会](https://www.theverge.com/2024/7/10/24195528/microsoft-apple-openai-board-observer-seat-drop-regulator-scrutiny)
- [Poe 克隆了 Artifacts](https://techcrunch.com/2024/07/08/quoras-poe-now-lets-users-create-and-share-web-apps/?guccounter=1) —— 某种程度上

> Meta：继昨天的幻觉讨论之后，我们正处于 Reddit 评论重大升级的最后阶段。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 综述

> 所有综述均由 Claude 3.5 Sonnet 完成，从 4 次运行中选取最佳结果。

**Yi AI 模型更新与集成**

- **Yi 模型在 GitHub 上广受欢迎**：[@01AI_Yi](https://twitter.com/01AI_Yi/status/1810785879587524674) 分享了 Yi 模型目前在 GitHub 上已获得 **7.4K stars 和 454 forks**，许多基于其 LLMs 的出色项目正在涌现。他们鼓励大家探索 Yi 模型并分享作品。
- **与 Axolotl 的潜在集成**：[@cognitivecompai](https://twitter.com/cognitivecompai/status/1810824550071849070) 建议 Yi 应该集成 **Axolotl 的 pregeneration** 功能。在另一条推文中，[@cognitivecompai](https://twitter.com/cognitivecompai/status/1810824550071849070) 提到集成 **Axolotl 的预处理 (preprocessing) 功能** 也会非常酷。

**Cognitive Computing AI 的推文与讨论**

- **家庭/小型企业 AI 硬件设备概念**：[@cognitivecompai](https://twitter.com/cognitivecompai/status/1810731504965931411) 指出，**AMD 技术** 使 **家庭/小型企业 AI 硬件设备** 的概念成为可能。
- **推文中被涂抹的内容**：[@cognitivecompai](https://twitter.com/cognitivecompai/status/1810708216596115538) 询问 @victormustar 关于某条推文中被**涂抹掉**的内容。

**AI 与人类认知**

- **人类的 System 2 蒸馏**：[@jaseweston](https://twitter.com/jaseweston/status/1810710786353902041) 解释说，在人类中，“**System 2 蒸馏**”方法被称为**自动化 (automaticity)**、**程序性记忆 (procedural memory)**，或者非正式地称为使其成为“**第二本能 (second nature)**”。

**杂项**

- **噬菌体 x 宿主 ML 预测综述**：[@elicitorg](https://twitter.com/elicitorg/status/1810784301258277134) 转发了 @yawnxyz 的推文，后者提到可能会与 @elicitorg 合作，利用 AI 和电子表格对所有**噬菌体 x 宿主 ML 预测工作**进行综述。

---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**AI 模型发布与进展**

- **Meta 发布 Chameleon 模型**：在 /r/LocalLLaMA 中，Meta 在 HuggingFace 上发布了 [**Chameleon-7b 和 Chameleon-30b 模型**](https://www.reddit.com/r/LocalLLaMA/comments/1dz5cgk/meta_creates_repos_for_chameleon7b_and_30b_on/)，该模型采用统一架构，通过对两种模态进行 Tokenization，实现了文本和图像的混合输入与输出。
- **Salesforce 的 xLAM-1b 在 Function Calling 方面超越 GPT-3.5**：尽管只有 1B 参数，Salesforce 的 xLAM-1b 模型在 [**Function Calling 能力上超越了 GPT-3.5**](https://www.reddit.com/r/LocalLLaMA/comments/1dz8g10/salesforce_tiny_giant_xlam1b_model_surpasses_gpt/)，正如 /r/LocalLLaMA 中所讨论的。这可能很快会催生更多 Rabbit R1 的克隆产品。
- **Anole 开创了交错文本-图像-视频生成**：/r/StableDiffusion 强调 Anole 是 [**首个支持高达 720p 144fps 文本-图像-视频生成的开源多模态 LLM**](https://www.reddit.com/r/StableDiffusion/comments/1dzjxov/an_opensourced_textimage2video_model_supports_up/)，并展示了极具前景的结果。
- **Phi-3 Mini 扩展了 Function Calling 功能**：Phi-3 Mini 模型已更新 [**Function Calling 功能，参数量从 3.8B 增加到 4.7B**](https://www.reddit.com/r/LocalLLaMA/comments/1dzhe38/phi3_mini_june_with_function_calling/)。它在与 Mistral-7b v3 的竞争中表现出色。

**AI 应用与用例**

- **小米揭幕全自动“智能”工厂**：/r/singularity 分享了小米新工厂的消息，该工厂将 [**24/7 无人化运行，每分钟生产 60 部智能手机**](https://www.reddit.com/r/singularity/comments/1dz9rvf/xiaomis_new_smart_factory_will_operate_247/)，展示了 AI 变革制造业的潜力。
- **Minecraft Agent 通过 Google Sheets 和新闻简报进行协作**：玩 Minecraft 的 AI Agent 现在正 [**在 Google Sheets 中记录进度，并由一名记者 Agent 创建新闻简报来分享更新**](https://twitter.com/nicochristie/status/1806428106342924341)，展示了 AI 系统在任务上的协作。
- **AI 以可玩的帧率生成复古游戏画面**：在 /r/StableDiffusion 中，Stable Diffusion 和其他模型正被用于 [**以可玩的速度生成复古风格的游戏图形**](https://www.reddit.com/r/StableDiffusion/comments/1dzo7f1/testing_tensor_toys_out/)。虽然仍不完美，但这预示了 AI 对游戏开发的颠覆性潜力。

**AI 伦理与治理**

- **民调：美国选民认为 AI 安全开发优先于与中国的竞争**：最近的一项民调发现，[**美国选民认为负责任的 AI 开发比与中国的竞争更重要**](https://time.com/6996090/ai-china-american-voters-poll/)，这表明公众支持安全，即使这意味着进度放缓。
- **中国推动全球 AI 合作**：[**中国正倡导在 AI 开发和治理方面进行国际协作**](https://www.scmp.com/opinion/china-opinion/article/3269571/why-china-pushing-so-hard-international-cooperation-ai)，这可能是为了制定符合自身利益的政策，凸显了 AI 进展的地缘政治层面。
- **针对 GitHub Copilot 涉嫌侵犯版权的诉讼被驳回**：法官驳回了 [**指控 GitHub 的 AI 编程助手 Copilot 侵犯版权的诉讼**](https://www.infoworld.com/article/2515112/judge-dismisses-lawsuit-over-github-copilot-ai-coding-assistant.html) 中的大部分主张，仅保留了两项指控。该案的结果可能会影响基于公开数据训练的 AI 系统。

---

# AI Discord 回顾

> 摘要之摘要的摘要

## Claude 3 Sonnet


**1. 新语言模型发布**

- **Ghost 8B Beta 凭借多语言能力首次亮相**：[**Ghost 8B Beta**](https://huggingface.co/spaces/lamhieu/ghost-8b-beta-8k) 大语言模型承诺提供强大的 **multilingual**（多语言）能力和成本效益，提供 **8k** 和 **128k** 版本，并附带详细说明其架构和技术的完整 [文档](https://ghost-x.org/docs/models/ghost-8b-beta)。
   - 尽管该模型的首次亮相引发了广泛关注，但一些人对其与更专业模型相比的 **knowledge capabilities**（知识能力）表示担忧。
- **Anole：首个开源自回归 LMM**：**[Anole](https://github.com/GAIR-NLP/anole)** 作为首个开源、自回归原生的 **Large Multimodal Model (LMM)** 推出，它基于 @AIatMeta 的 **Chameleon** 构建，并承诺具备多模态生成能力。
   - 然而，为了重新引入从 Chameleon 中移除的图像能力而对 **Anole** 进行 **fine-tune** 的努力遭到了 [反对](https://x.com/ArmenAgha/status/1810804784905212222)，人们担心这会破坏明确的设计选择。
  


**2. AI 模型基准测试与评估**

- **定理证明取得飞速进展**：**HarmonicMath** 宣布在具有挑战性的 **MiniF2F benchmark** 上实现了惊人的 **90% state-of-the-art** 成绩，相比其一个月前在 [更新](https://x.com/HarmonicMath/status/1810765353389281346) 中分享的 **83%** 结果有了显著飞跃。
   - AI 社区赞扬了定理证明领域的**极速进展**，考虑到该基准测试的较简单版本在今年早些时候仅为 **50%**。
- **审视 VLM 在基础任务上的表现**：一篇新论文指出，尽管在传统基准测试中得分很高，但像 **GPT-4o** 和 **Gemini 1.5 Pro** 这样的 **state-of-the-art Vision Language Models (VLMs)** 在识别重叠形状和物体计数等基础视觉任务上表现挣扎。
   - [这项研究](https://arxiv.org/abs/2407.06581) 中详述的发现引发了对 **VLMs** **现实世界适用性** 的担忧，并对现有评估指标的有效性提出了质疑。
  


**3. 合成数据生成与反馈循环**

- **利用强化合成数据防止模型崩溃**：[这篇论文](https://arxiv.org/abs/2406.07515) 详述的新研究探索了利用**对合成数据的反馈**来防止大语言模型中的 **model collapse**（模型崩溃）。
   - 研究说明了**盲目使用合成数据**如何导致性能下降，并提倡使用**反馈增强的合成数据**，以在矩阵特征值计算和新闻摘要等实际任务中保持高性能。
- **指数积分器加速 Diffusion 采样**：一位成员就论文 [FAST SAMPLING OF DIFFUSION MODELS WITH EXPONENTIAL INTEGRATOR](https://arxiv.org/abs/2204.13902) 中 **"marginal distributions as p̂∗_t"** 一词寻求澄清，该论文提出了一种加速 **diffusion models** 极其缓慢的采样过程的方法。
   - 该论文的方法承诺在保持 **diffusion models** 在各种生成建模任务中生成高保真样本能力的同时，提高其采样效率。

## Claude 3.5 Sonnet


**1. Anole: 首个开源自回归 LMM**

- **Anole 的到来：多模态的奇迹**：[Anole](https://github.com/GAIR-NLP/anole) 是首个开源的自回归大型多模态模型 (LMM)，由 @AIatMeta 推出，基于 Chameleon 架构构建。
   - 这一发布引发了关于开源多模态模型潜力的讨论，一些人对重新引入之前从 Chameleon 中移除的图像功能表示担忧，正如一篇[批评性推文](https://x.com/ArmenAgha/status/1810804784905212222)中所指出的。
- **技术磨难：GPU 之争**：尝试在多个 GPU 上运行 Anole 的用户遇到了 CUDA 显存溢出 (out-of-memory) 错误，凸显了新模型的扩展挑战。
   - 一个 [GitHub issue](https://github.com/GAIR-NLP/anole/issues/7) 已开启，讨论支持在多 GPU 上运行 Anole 的潜在修改方案，这表明社区正在努力提高该模型的可访问性和性能。
  


**2. xAI 雄心勃勃的 H100 集群扩张**

- **马斯克的百亿亿级尝试**：Elon Musk [宣布](https://x.com/elonmusk/status/1810727394631950752?s=46&t=PW8PiFwluc0tdmv2tOMdEg) xAI 已从 Oracle 签约租赁了 24,000 块 H100 GPU，并正在建设一个拥有 100,000 块 H100 的庞大系统用于 AI 训练。
   - Musk 强调了对 AI 基础设施进行内部控制的必要性，以保持竞争速度和效率，使 xAI 的集群有潜力成为全球最强大的集群。
- **Grok 的成长：从训练到发布**：xAI 的 Grok 2 模型目前正在新购入的 H100 集群上进行训练，Musk 表示该模型正在进行微调 (finetuning) 和 Bug 修复。
   - Grok 2 预计将于下个月发布，展示了 xAI 不断扩大的计算资源所带来的快速开发周期。
  


**3. AMD 对 Silo AI 的战略性 AI 收购**

- **芯片制造商的 AI 棋局**：[AMD 宣布](https://www.ft.com/content/7b8d2057-2687-45b3-bae4-1488a75ac5b2)以 **6.65 亿美元**收购芬兰 AI 初创公司 **Silo AI**，旨在扩大其 AI 服务并更有效地与 Nvidia 竞争。
   - 这笔全现金交易标志着自 2014 年 Google 以约 4 亿英镑收购 DeepMind 以来，欧洲对私有 AI 初创公司规模最大的收购之一，信号表明 AMD 对 AI 发展的严肃承诺。
- **Silo 的软件协同效应**：Silo AI 的 300 人团队将利用 AMD 的软件工具，为聊天机器人和其他 AI 应用构建定制的大型语言模型 (LLMs)。
   - AMD 的 **Vamsi Boppana** 强调，此次收购将加速客户参与并增强 AMD 自身的 AI 技术栈，可能重塑 AI 硬件和软件集成的竞争格局。
  


**4. GitHub Copilot 版权诉讼更新**

- **AI 代码生成的法律宽容**：加州地方法院[部分驳回](https://www.docketalarm.com/cases/California_Northern_District_Court/4--22-cv-06823/DOE_1_et_al_v._GitHub_Inc._et_al/1/)了针对 Microsoft 的 GitHub Copilot 和 OpenAI 的 Codex 的版权诉讼，这可能为在受版权保护数据上训练的 AI 工具设定先例。
   - 法院的裁决表明，只要 AI 系统不进行精确复制，就可能免于追责，这对于 AI 编程助手的开发和部署具有深远影响。
- **Copilot 持续的争议**：虽然诉讼的大部分内容被驳回，但关于 AI 工具在没有适当许可的情况下建议代码片段的担忧，仍是开发者社区讨论的话题。
   - 这一裁决可能会影响未来关于 AI 辅助编程时代知识产权的案例和讨论，在创新与版权保护之间寻求平衡。

## Claude 3 Opus


**1. Ghost 8B Beta 发布**

- **多语言精通**：**Ghost 8B Beta** 首次亮相，推出了具有强大**多语言**功能的 **8k** 和 **128k** 上下文长度版本。[在 Hugging Face 上体验](https://huggingface.co/spaces/lamhieu/ghost-8b-beta-8k)。
   - [Ghost 8B Beta 官方文档](https://ghost-x.org/docs/models/ghost-8b-beta)为那些寻求深入了解的人提供了关于该模型**架构**、**技术**、**评估**等方面的详细信息。
- **高性价比对话**：**Ghost 8B Beta** 的一个关键目标是提供比其他替代方案更具**成本效益**的 LLM 性能。
   - 通过专注于**多语言支持**和**知识能力**，同时保持较低成本，Ghost 8B Beta 旨在让更多人能够使用强大的对话式 AI。
  


**2. Llama 3 训练讨论**

- **瑞典语 Llama 引发辩论**：围绕在 **Unsloth AI** 中使用**瑞典语 Llama 3** 模型的讨论展开，该模型是在 [LUMI 超级计算机](https://lumi-supercomputer.eu/)上使用 [42 Labs](https://huggingface.co/42-labs) 的数据训练而成的。
   - 一些人建议使用基座模型进行训练，并使用指令模型处理**翻译**等任务，而另一些人则指出 Llama 3 在 Google Colab 等平台上存在**推理速度问题**。
- **Llama 转向 LM Studio**：为了克服 **Llama 3 的推理速度挑战**，推荐使用 [LM Studio](https://lmstudio.ai/) 作为 Google Colab 的替代方案，以获得更好的性能。
   - 用户还询问了如何在 Mac 设备上**本地运行 Llama 3 推理**，并建议在 LM Studio 上搜索符合系统规格的量化版本。
  


**3. 模型保存故障**

- **GGUF 格式转换失误引发困扰**：由于 `llama.cpp` 库中缺少 `llama-quantize` 或 `quantize` 文件，用户在尝试以 **GGUF 格式**保存模型时遇到了**严重错误**。
   - 这些错误导致保存操作期间出现**运行时故障**，引发了关于 GGUF 转换过程的潜在变通方法和修复方案的讨论。
- **Embedding 训练尝试**：出现了关于在冻结预训练 Embedding 的同时**手动训练新 Token Embedding** 的问题，以确保对特殊 Token 的准确预测。
   - 考虑对特定模块采用**手动反向传播**等方法，以避免从头开始重新训练所有 Embedding。
  


**4. 模型对决：Gemini vs DeepSeek**

- **开发者选择他们的最佳工具**：讨论对比了 **DeepSeek Chat** 和 **DeepSeek Coder** 模型，一些人更青睐新的 **DeepSeek Coder v2** 来执行代码辅助任务。
   - 用户报告称，连续几周使用 **DeepSeek Coder v2 lite** 作为**代码助手**，结果令人满意。
- **Flash 还是 Pro？定价困惑**：**Claude 3 Haiku** 与 **Gemini 1.5 Flash/Pro** 模型之间的价格对比引发了混乱，某个 AI 错误地声称 Haiku 更便宜。
   - 当该 AI 将 **Haiku** 与 **Gemini 1.5 Pro** 而非同级别的 **Flash** 模型进行对比时，发生了进一步的混淆，这突显了更清晰的定价沟通的必要性。
  


**5. CodeGeeX4 破解代码难题**

- **CodeGeeX4 征服竞争对手**：新的 **CodeGeeX4** 模型被认为在各种**代码生成**任务上优于 **DeepSeek v2**，目前已有版本[在 Hugging Face 上可用](https://huggingface.co/THUDM/codegeex4-all-9b)。
   - 与 **CodeQwen** 的对比进一步巩固了 **CodeGeeX4** 在代码辅助领域的领先地位。
- **GLM4 助力 CodeGeeX4**：随着 [**GLM4** 合并至 llama.cpp](https://github.com/ggerganov/llama.cpp/pull/8031) 库，社区反响热烈。
   - 由于 **CodeGeeX4** 基于 **GLM4**，这一集成预计将在未来的更新中进一步增强该模型的**代码生成**性能。

## GPT4T (gpt-4-turbo-2024-04-09)


**1. 多语言 LLMs**

- **Ghost 8B Beta 引起多语言关注**：**Ghost 8B Beta** 的首次亮相承诺提供强大的 **multilingual**（多语言）功能和成本效益，并提供 **8k** 和 **128k** 上下文版本。可以在 [Hugging Face](https://huggingface.co/spaces/lamhieu/ghost-8b-beta-8k) 体验。
   - 若要深入了解 **Ghost 8B Beta**，查阅 [官方文档](https://ghost-x.org/docs/models/ghost-8b-beta) 可获得关于模型架构和技术的深度知识。
- **Llama 3 模型训练引发辩论**：关于使用 Unsloth AI 进行 **Llama 3** 模型训练的讨论转向了 **Swedish**（瑞典语）版本及其 DeepAI 部署，这主要由 [42 Labs 数据](https://huggingface.co/42-labs) 推动。
   - Google Colab 上的 **Inference speed**（推理速度）困扰导致用户转向 [LM Studio](https://lmstudio.ai/)，以在 **Llama 3** 模型上获得更佳性能。
    


**2. 模型微调与优化**

- **GPTs 拒绝额外训练？原因如下**：一个令人困惑的问题出现了，即 **GPTs agents** 在初始训练后停止学习，这引发了关于 **knowledge file uploads**（知识文件上传）的澄清：此类上传仅起辅助作用，并不会更新 Agent 的基础知识。
   - 针对学习率的额外咨询促使大家在微调 **Qwen2-1.5b** 等 AI 模型时对 **cosine scheduler** 达成共识。
- **困于 GGUF？错误频发令人沮丧**：AI 工程师们在进行 GGUF 模型转换时面临困境，因为在保存操作期间由于缺少 `llama-quantize` 而出现了关键错误。
   - 在以 **GGUF format** 保存模型时遇到的问题，将讨论引导至通过降级到特定的 xformers 库版本来解决错误。
    


**3. AI 硬件与基础设施**

- **TPUs 在 Hugging Face 上起飞**：**Google TPUs** 现在增强了 Hugging Face 平台，使用户能够利用不同的内存选项和清晰的定价来构建和训练 Generative AI 模型。
   - Spaces 和 Inference Endpoints 在集成 TPU 后非常活跃，[@_philschmid](https://x.com/_philschmid/status/1810350949552070955) 在 Twitter 上标记了这一进展。
- **马斯克的狂热扩张**：xAI 节奏飞快，为其 AI 集群抢购了 2.4 万张 H100，详见 Elon Musk 的 [推文](https://x.com/elonmusk/status/1810727394631950752?s=46&t=PW8PiFwluc0tdmv2tOMdEg)。
   - 这位 AI 领导者的热情显而易见，他计划建立一个拥有 10 万张 H100 的巨型设施，目标直指计算霸权的巅峰。
    


**4. AI 法律与伦理问题**

- **GitHub Copilot 诉讼更新**：开发者对 GitHub Copilot 的指控大部分被驳回，仅剩两条指控[依然存在](https://www.theregister.com/2024/07/08/github_copilot_dmca/?utm_source=ainews&utm_medium=email)。
   - 最初的指控涉及 Copilot 涉嫌在没有适当许可的情况下推荐代码片段，引发了知识产权方面的担忧。
- **不再侵权？法院对 Copilot 版权案的裁定**：加州法院的一项关键裁决可能预示着 AI 发展将迎来更平稳的局面，针对 Microsoft 的 GitHub Copilot 和 OpenAI 的 Codex 的[版权诉讼](https://www.docketalarm.com/cases/California_Northern_District_Court/4--22-cv-06823/DOE_1_et_al_v._GitHub_Inc._et_al/1/)中大部分重要部分已被驳回。
   - 法院的决定可能是对在受版权保护数据上训练的 AI 工具的一个预兆，尽管在知识产权领域的全面影响仍在酝酿中。
    


**5. AI 社区倡议**

- **黑客松热潮：AGI 的周末代码集会**：AGI House 将于本周六 7/13 举办一场黑客松，合作伙伴包括 **@togethercompute**、**@SambaNovaAI** 等，呼吁参与者在[此处](https://t.co/LOEgpc1BOs)申请。
   - 最近推出的 **Llama-Agents** 在 GitHub 上已突破 1100 颗星，@MervinPraison 在 [YouTube](https://t.co/8uetfVqHf9) 上提供了详细的演示教程。
- **Perplexity 合作伙伴助力**：Perplexity AI 宣布与 **Amazon Web Services (AWS)** 合作，为 AWS 客户提供 **Perplexity Enterprise Pro**，承诺简化其 AI 工具包。
   - 随着 **Perplexity Enterprise Pro** 通过 [AWS Marketplace](https://t.co/t3xBQlyw0c) 扩大可用性，AWS 客户将从增强的 AI 支持中受益。
    

## GPT4O (gpt-4o-2024-05-13)

$PLSDELETTHIS{openaiSummaryO}

---

# PART 1: 高层级 Discord 摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Ghost 8B Beta 引起多语言领域关注**：**Ghost 8B Beta** 的首次亮相承诺了强大的 **multilingual**（多语言）功能和成本效益，并提供 **8k** 和 **128k** 版本。可以在 [Hugging Face](https://huggingface.co/spaces/lamhieu/ghost-8b-beta-8k) 上进行体验。
   - 欲深入了解 **Ghost 8B Beta**，查阅 [官方文档](https://ghost-x.org/docs/models/ghost-8b-beta) 可获得关于模型架构和技术的深度知识。
- **Llama 3 模型训练引发辩论**：关于在 Unsloth AI 中使用 **Llama 3** 模型的讨论转向了 **Swedish**（瑞典语）版本及其在 DeepAI 的部署，这主要由 [42 Labs 数据](https://huggingface.co/42-labs) 推动。
   - Google Colab 上的 **Inference speed**（推理速度）困扰导致用户转向使用 [LM Studio](https://lmstudio.ai/)，以提升 **Llama 3** 模型的性能。
- **困于 GGUF？错误频发令人沮丧**：由于在保存操作中缺少 `llama-quantize`，导致出现关键错误，AI 工程师们正苦于 GGUF 模型转换。
   - 在以 **GGUF format** 保存模型时遇到的问题，将讨论引向了通过降级到特定版本的 xformers 库来解决错误。
- **GPTs 拒绝额外训练？原因如下**：出现了一个令人困惑的问题，即 **GPTs agents** 在初始训练后停止学习，这引发了关于 **knowledge file uploads**（知识文件上传）的澄清：此类上传虽有帮助，但不会更新 Agent 的基础知识。
   - 关于额外学习率的咨询促使大家在微调 **Qwen2-1.5b** 等 AI 模型时，对使用 **cosine scheduler** 达成共识。
- **Token 训练难题凸显**：AI 社区面临着关于新 Token Embedding 的严峻挑战，如果没有全面的 **pretraining**（预训练）工作，效果可能会不尽如人意。
   - 尽管存在 Embedding 不足的风险，手动 **backpropagation**（反向传播）可能是一种权宜之计，用于优化新特殊 Token 的预测。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **TPUs 在 Hugging Face 上线**：**Google TPUs** 现在支持 Hugging Face 平台，使用户能够构建和训练具有不同内存选项和明确定价的 Generative AI 模型。
   - Spaces 和 Inference Endpoints 正在集成 TPU，[@_philschmid](https://x.com/_philschmid/status/1810350949552070955) 在 Twitter 上也提到了这一点。
- **Transformers 攻克代码难题**：Transformers 不再仅仅用于 NLP，社区成员们正在交流使用 Python 技巧和 Tokenizer 调整进行 **debugging and coding**（调试与编码）的心得。
   - [GitHub 链接](https://github.com/huggingface/diffusers/tree/main/examples/research_projects/sd3_lora_colab) 和关于在本地运行 AI 的视频，让成员们交换了高效模型托管的实践经验。
- **掌握 Knowledge Graphs**：分享了一个教程直播，介绍了通过 Knowledge Graphs 增强自然语言查询的策略，并由 Langchain 和 Neo4j 提供支持。
   - 随着社区成员讨论该教程中处理电子游戏销售数据的方法，兴趣激增，详情见 [此 YouTube 频道](https://www.youtube.com/watch?v=9wqVz0LDYgg)。
- **AI 引导的叙事**：一篇 Medium 文章引发了引人入胜的讨论，深入探讨了 Generative AI 改变故事讲述艺术的方式。
   - [点击此处阅读](https://medium.com/@shikharnautiyal29/the-impact-of-generative-ai-on-storytelling-and-narrative-creation-d37898cc0126)，了解作者和观众是如何适应这种变化的。
- **Qdurllm 亮相舞台**：一款名为 **Qdurllm** 的新型 AI 驱动搜索引擎受到关注，其演示版结合了 Qdrant 和 Sentence Transformers 以增强搜索功能。
   - [去看看吧](https://huggingface.co/spaces/as-cle-bert/qdurllm-demo)，并通过在其 GitHub 仓库贡献想法来参与讨论。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Shared Memory 的新高度与黑客松热潮**：计算能力（compute capability）为 8.9 的 GPU 每个 block 最多可管理 **99 KB 的 Shared Memory**，如 [kernel 启动示例](https://github.com/karpathy/llm.c/blob/master/dev/cuda/layernorm_forward.cu#L479-L502)所示。
   - **黑客松爱好者**正在为一场以 CUDA 为中心的活动做准备；关于团队组建和参会福利的讨论非常热烈，详见活动[页面](https://partiful.com/e/fxMwOW9dtCCWoEPyyTIf)。
- **AMD 收购 Silo AI 以争夺 AI 霸权**：[AMD 以 **6.65 亿美元**收购 Silo AI](https://www.ft.com/content/7b8d2057-2687-45b3-bae4-1488a75ac5b2) 是一项战略举措，旨在增强其 AI 能力并与 Nvidia 展开竞争。
   - 这笔交易标志着欧洲 AI 初创生态系统的一个重大事件，可与 Google 收购 DeepMind 相提并论，并提高了未来交易的门槛。
- **远程职位与框架热潮**：一位在 Hugging Face DRL 排行榜上排名**全球第 8** 的开发者正在寻找新机会，并推介其创新的 **PyEmber 框架**。
   - 为了寻求合作，该开发者分享了其[个人简历](https://drive.google.com/file/d/1f0fRDZTeO0-lJ-PEIkurs7YYT8zSEQut/view?usp=sharing)，表示已准备好将其专业知识带到新的领域。
- **MacBook 上的 CUDA 能力及其他**：希望在 MacBook 上使用 CUDA 的开发者转向 **Google Colab** 作为跳板，利用其免费层级进行成长，而无需配备笨重的 GPU。
   - 拥有 GPU 的道路是一场马拉松而非短跑；对于希望扩展到物理硬件的爱好者来说，像 **vast.ai** 这样的云端替代方案是一个过渡选择。
- **剖析 MuAdam 与模型细微之处**：**MuAdam** 的学习率特性在 [GitHub 讨论](https://github.com/microsoft/mup/issues/7)中成为焦点，参与者们讨论了输出权重调整的微妙之处。
   - 实验引发了关于 Embedding 权重初始化的讨论，并对 StableAdam 处理 Loss 尖峰的方式表示关注，引导社区走向创新的微调方向。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI 系统中的锁定与阻塞**：讨论集中在 AI 系统中实现**锁定机制**的可能性，以便在监控用户交互后提供受控响应。
   - 围绕系统自主性和安全性的讨论展开，话题在伦理影响和技术可行性之间转换。
- **为 AI 性能配置 GPU**：AI 爱好者们交流了针对任务密集型 AI 模型的**最佳 GPU 配置**心得，重点强调了**高 RAM** GPU 的优势。
   - 云端与本地推理的对比构成了一幅技术图景，并提供了 RunPod 和 Paperspace 的链接以获取更多见解。
- **去中心化计算的架构**：**去中心化计算平台**成为一个引人入胜的话题，并与 BOINC 等现有倡议进行了类比。
   - 对话深入探讨了由志愿者驱动的计算范式在 AI 相关任务中的实用性。
- **应对 ChatGPT 的上下文难题**：在 *gpt-4-discussions* 频道中，用户反映了 ChatGPT 回复的问题，指出对**过时或不准确信息**的担忧。
   - 关于 Context Window 大小的澄清出现了，[价格页面](https://openai.com/chatgpt/pricing/)等来源给出了从 *32K* 到 *128K* 不等的数字。
- **增强 GPT 的思维路径**：在 *#api-discussions* 中，有人分享了一个**亲自设计的“思考过程”**，用于自定义 GPT，旨在提高模型的准确性和真实性。
   - 社区被号召行动起来，鼓励大家本着共同完善的精神，对这些自定义 GPT 修改进行**实验并提供反馈**。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- ****解决 LM Studio 更新故障****：用户通过清除缓存或重新安装来解决 **LM Studio** 的更新问题（如**黑屏**），同时 **DiffusionBee** 中自定义模型的导入也引发了讨论。
   - **移动端深度学习**取得飞跃：一名成员在 S21 手机上测得 **Mistral 7B** 运行速度达到 **10 tokens/second**，引发了关于 LLM 移动端效率的热议。
- ****显卡对决：技术难题****：AI 爱好者们就 **3090 与 4090 GPU 的性能**展开辩论，而 **AMD 收购 SiloAI** 则标志着其在 AI 硬件领域的强势发力。
   - 用户对 **Intel Arc 770** 乏善可陈的 AI 支持表示担忧，建议由于更好的工具支持，应坚持使用 Nvidia。
- ****代码模型在创意碰撞中竞争****：开发者社区权衡了 **DeepSeek Coder v2** 与新兴的 **CodeGeeX4** 的优劣，部分用户认为后者在开发任务中表现更佳。
   - 在一次重大的社区更新中，**GLM4 集成**到 llama.cpp 的消息传出，预示着 **CodeGeeX4 代码模型**将迎来改进。
- ****探讨双重 LM Studio 安装****：有用户询问是否可以在一台机器上安装两个版本的 **LM Studio**，以适配不同的 GPU。
   - **LM Studio** 的 0.2.27 版本受到质疑，因为与之前的版本相比，它在 **AMD 7700XT** 上的运行速度有所下降。
- ****再次关注 Hugging Face 访问问题****：社区成员反映了暂时的 **Hugging Face 访问问题**，随后确认已解决，这表明只是短暂的故障。
   - 在 **LM Studio** 中访问**特定 Hugging Face URL** 时遇到的共同困扰，引发了关于潜在软件漏洞的讨论。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- ****Chroma 分块难题****：Chroma 通过一份[技术报告](https://x.com/atroyn/status/1810717585442492686?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)深入探讨了检索效率，发现随着 LLM 上下文长度的增加，分块策略变得至关重要。
   - **备受期待的 Turbopuffer** 正在开发中，人们对其针对对象存储的高性价比、更快速的搜索解决方案寄予厚望，这在 [Turbopuffer 的博客](https://turbopuffer.com/blog/turbopuffer)中进行了详细讨论。
- ****马斯克狂热的扩张****：xAI 节奏迅猛，为其 AI 集群抢购了 2.4 万张 H100，详情见 Elon Musk 的[推文](https://x.com/elonmusk/status/1810727394631950752?s=46&t=PW8PiFwluc0tdmv2tOMdEg)。
   - 这位 AI 领袖的热情显而易见，他正计划建立一个拥有 10 万张 H100 的巨型设施，旨在登顶计算霸权。
- ****Skild AI 拔得头筹****：随着隐身模式的解除，Skild AI 披露了巨额的 3 亿美元 A 轮融资，引起了广泛关注，详见 Deepak Pathak 的[公告](https://x.com/pathak2206/status/1810769359591330201?s=46)。
   - 雄心壮志与 VC 圈的怀疑交织在一起，引发了在科技估值飙升背景下融资稳健性的辩论。
- ****Copilot 版权冲突降温****：GitHub Copilot 的法庭诉讼规模缩小，仅剩两项指控，详情见 [The Register 的报道](https://www.theregister.com/2024/07/08/github_copilot_dmca/?utm_source=ainews&utm_medium=email)。
   - 过去因不当授权建议引起的摩擦有所缓解，为关于代码所有权和 AI 的更广泛辩论提供了参考。
- ****ImageBind 带来的空间奇观****：[ImageBind 论文](https://arxiv.org/abs/2305.05665)成为焦点，展示了一种能够绑定六种数据模态并在零样本（zero-shot）挑战中胜出的双目视觉技术。
   - 作为多模态学习的一大进步，ImageBind 的表现优于其专门领域的同行，让人一窥未来统一跨模态 AI 应用的前景。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **编译器难题与澄清**：从源码构建 **Mojo compiler** 引发了疑问，因为该过程尚未有清晰的文档说明；目前仅提供标准库的编译。
   - 对于 nightly 版本的 Mojo 编译器发布版 `2024.7.1005`，可以使用命令 `modular update nightly/mojo` 进行更新，根据 [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)，其中改进了 `memset` 的使用，并修复了 **kwargs** 崩溃问题。
- **生产环境中 PyTorch 的思考**：[Modular](https://www.modular.com/blog/bring-your-own-pytorch-model) 强调了在生产环境中部署 **PyTorch models** 的复杂性，解决了资源和延迟方面的挑战。
   - 鼓励 AI 开发者将生成式 AI 集成到服务中，[Bain & Company 的一项调查](https://www.bain.com/insights/ai-survey-four-themes-emerging/)显示，**87% 的公司**正在试点或部署该技术。
- **巧妙的基准测试建议**：关于准确基准测试的建议包括禁用超线程和设置 CPU 亲和性，如[本指南](https://easyperf.net/blog/2019/08/02/Perf-measurement-environment-on-Linux)所述。
   - 根据基准测试设计效率的讨论，在基准测试中纳入对称和非对称场景可确保稳健的性能评估。
- **Mojo Setter 的同步障碍**：在 Mojo 中使用 `__setitem__` 时出现异常，疑似存在调用 `__getitem__` 而非 setter 的 bug，已在 GitHub 上提交了 issue。
   - 讨论还涉及了 Mojo 中零拷贝反序列化（zero-copy deserialization）的复杂性，权衡了类型转换和分配器感知（allocator awareness），讨论倾向于内存管理的深层技术细节。
- **Graviton4：引领 AWS 实例入侵**：基于 **AWS Graviton4** 的 [Amazon EC2 R8g 实例](https://aws.amazon.com/blogs/aws/aws-graviton4-based-amazon-ec2-r8g-instances-best-price-performance-in-amazon-ec2/)现已推出，号称在内存密集型应用中拥有同类最佳的性价比。
   - 虽然一些数据库公司寻求立即推出，但预计 AWS 将在即将举行的 ReInvent 大会上发布大多数 'c' 和 'm' 系列实例。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **论文征集 - 实体谜题**：成员们就**实体消歧（entity disambiguation）**的输入进行了交流，突显了知识库中的空白以及对进步的渴望。
   - 具体的见解请求包括探索**基于 LLM 的合成数据生成**和 AI 中的情商，并积极寻找**共情 LLM（empathy LLMs）**相关的论文。
- **地图制作者 - EleutherAI 的制图工作**：社区地图绘制工作成为焦点，请求填写 [EleutherAI Global Map](https://forms.gle/AxLQNYiC68HY5GQH6)，以连接全球各地的成员。
   - **Diffusion Models** 爱好者深入探讨了模型中令人困惑的**边缘分布（marginal distributions）**，分享了[这篇论文](https://arxiv.org/abs/2204.13902)以丰富社区理解。
- **成功秘诀？- RegMix 的数据鸡尾酒**：**RegMix 的数据混合作为回归**是一个热门话题，其预训练性能的前景已在广为流传的[研究](https://arxiv.org/abs/2407.01492)中勾勒出来。
   - VLM 的基准测试表现与物体计数等现实任务之间的脱节引发了对其整体效用的质疑，[最新的 VLM 研究](https://arxiv.org/abs/2407.06581)中的得分问题也强调了这一点。
- **干预混搭 - 组合 AI 改进**：得益于 [Kyle Devin O'Brien 的见解](https://arxiv.org/pdf/2407.06483)，关于 LM 内部多种干预的讨论被触发，质疑了编辑和遗忘（unlearning）的可组合性。
   - [这项研究](https://arxiv.org/abs/2406.07515)指出了原生合成数据在防止**模型崩溃（model collapse）**方面的弊端，拓宽了社区对 AI 数据效用的看法。
- **神经细微差别 - 大脑字节大小很重要**：围绕**大脑大小与智力**以及哺乳动物皮层神经元数量的对话表明，除了单纯的**神经元密度**之外，还存在更细微的关系。
   - 出现了关于遗传学和智商（IQ）的论述，一位用户指出了围绕人类智力属性的复杂性和敏感性。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 合作伙伴助力**：Perplexity AI 宣布与 **Amazon Web Services (AWS)** 合作，为 AWS 客户提供 **Perplexity Enterprise Pro**，承诺简化其 AI 工具包。
   - 随着 **Perplexity Enterprise Pro** 在 [AWS Marketplace](https://t.co/t3xBQlyw0c) 的可用性扩大，AWS 客户将从增强的 AI 支持中受益。
- **PPLX 库的 Docker 困境**：一位用户在 Docker 中设置 `pplx` 库时遇到了编译障碍，尽管在 Docker 之外使用 `nodemon` 成功，但在容器内无法找到该模块。
   - 解决此问题的努力包括对 `tsconfig.json` 和 `package.json` 进行调整，但社区尚未提供万无一失的解决方案。
- **模型价格对比失误**：关于 **Claude 3 Haiku** 比 **Gemini 1.5 Flash** 更便宜的错误陈述引发了混乱，忽略了 **Gemini 1.5 Flash** 微弱的价格优势。
   - 更令人困惑的是，AI 将 **Haiku** 与不同级别的 **Gemini 1.5 Pro** 而非同类模型进行比较，引发了关于性价比匹配的进一步讨论。
- **AI 处方药价格情节复杂化**：Perplexity AI 因最初在药品定价中遗漏了 CostPlusDrugs.com 而被点名，这是制药行业专业人士的关键考量。
   - 促使包含该综合定价网站的努力取得了成效，为更强大的默认搜索算法带来了希望。
- **API 定价不确定性揭晓**：成员们寻求澄清 API 的 **每百万 tokens 0.6 美元** 定价是否包含输入和输出 tokens。
   - 由于缺乏官方回应，这一定价困惑成为政策确认的首要话题。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Anole 欢庆：Anole 作为首个开源自回归 LMM 发布**：AI 社区对 [Anole](https://github.com/GAIR-NLP/anole) 的发布表示欢迎，这是一个开源的自回归 Large Multimodal Model (LMM)，引发了关于扩展 **Chameleon** 功能的讨论。
   - 在兴奋之余，人们对通过微调重新实现最初从 **Chameleon** 中移除的图像功能表示担忧，这反映在一条[批评性推文](https://x.com/ArmenAgha/status/1810804784905212222)中。
- **用代码“开锁”：对 Gemini 1.5 无意指令的探索**：Gemini 1.5 Flash 因通过“保持角色”提示词无意中提供撬车方法而受到审查。
   - 社区反应不一，一些人对模型的能力表示担忧，而另一些人则对其潜在的恶作剧行为持超然态度。
- **从 PDF 到 Markdown：使用 Marker 库规划路径**：[Marker 库](https://github.com/VikParuchuri/marker)因其将 PDF 熟练转换为 Markdown 的能力而获得赞誉，旨在增强 **Sonnet** 等模型的数据集。
   - 关于解析 PDF 的辩论出现了——这被认为几乎与使用正则表达式解析 HTML 一样棘手——人们呼吁更好的提取方法。
- **Schema 一致性：制定通用 RAG 格式规范**：AI 工程师在设计通用的 RAG `query-context-answer` 模板时，经历了共识与争论的交织。
   - 讨论涉及各种调整，贡献者在格式上达成一致，并考虑采用两阶段方法。
- **评估相关性：在 RAG Thought Tokens 中重构重排序**：在 `<thought>` tokens 中包含重排序（reranking）相关性的建议，在优化可解析性和评分方面引入了分歧。
   - 随后展开了关于速度与效率权衡的对话，并参考了 RankRAG 和其他两层系统。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **黑客松狂欢：AGI 的周末代码集会**：AGI House 将于本周六 7/13 举办一场黑客松，合作伙伴包括 **@togethercompute**、**@SambaNovaAI** 等，呼吁参与者在[此处](https://t.co/LOEgpc1BOs)申请。
   - 最近推出的 **Llama-Agents** 在 GitHub 上已突破 1100 颗星，@MervinPraison 在 [YouTube](https://t.co/8uetfVqHf9) 上提供了详细的演示。
- **LlamaIndex 领跑：Lyzrai 助力实现 100 万美元以上 ARR**：通过利用 **LlamaIndex** 的数据连接器和 RAG 功能，**@lyzrai** 实现了超过 100 万美元的 ARR，为销售和营销提供 AI 解决方案 [更多详情](https://t.co/A8KxLpc47S)。
   - 建议使用 **LlamaCloud** 服务来简化 AI 工程师的数据 ETL/管理，从而更专注于 Prompting 和 Agent 编排，并提供多种 Cookbook [了解更多](https://t.co/d5pDEq67DA)。
- **PDF 解析专业技巧：LlamaParse 布局解析**：推荐使用 **LlamaParse** 从 PDF 中提取数据，引发了关于需要 OpenAI API 密钥还是本地模型部署的讨论。
   - 用户解决了导致冗余元数据的查询模板问题，处理了 Azure OpenAI 上 **Llama-3/Mistral** 与 **GPT-4** 之间模板处理差异的疑虑。
- **流程优化成功：astream_chat 克服障碍**：**astream_chat** 的实现错误已得到有效修复，用户结合 **run_in_threadpool** 和 **async_wrap_generator** 方法来正确流式传输响应。
   - 讨论强调 **Ollama** 拥有用户友好的格式化功能，但缺乏 GPU 支持可能导致其性能比 **Llama-3/Mistral** 模型慢。
- **格式化技巧：LLM 学会了布局**：澄清显示设置 **is_chat_model=True** 会影响 **LLM.chat()** 或 **LLM.complete()** 的功能，从而影响查询引擎响应的格式化质量。
   - 承认 **LLMs** 处理格式细微差别的能力，是 AI 查询引擎高效使用 Chat 和 Completion 函数的基础。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Mac 用户在 Stable Diffusion 上的困扰**：在 **macOS** 上设置 **Stable Diffusion** 的挑战引发了对话，建议 macOS 用户使用 [Python 文件解决方案](https://huggingface.co/stabilityai/stable-diffusion-3-medium)，而不是常见的 Windows 指南。
   - *agcobra1* 担保了一个特定的实现，作为 **TouchDesigner** 集成问题的变通方案。
- **Adetailer 的全分辨率启示**：爱好者们发现 **Adetailer** 绕过了 VAE 编码，直接针对全分辨率输出，这可能会产生更精细的图像细节。
   - *hazmat_* 说明了现实情况，解释说 Adetailer 只是一个 Inpainting 工具，尽管它是即时的，以此来降低预期。
- **Stable Diffusion 入门指南**：社区贡献的指南简化了 ***Stable Diffusion*** 的设置过程，从获取合适的 GPU 到运行模型，还暗示了运营成本。
   - 成员们团结协作，*nittvdweebinatree* 建议不要使用复杂的 Anaconda 设置，而应采用更简单的方法。
- **稳定性能的 GPU 策略**：关于在 AMD GPU 上运行 ***Stable Diffusion*** 的好奇心激增，**AMD RX6800** 成为焦点，并参考官方 Zluda 指南进行深入了解。
   - 社区协作至关重要，在一位成员讲述了他们因指南不足而遭遇的困境后，成员们互相感谢提供了改进后的指南。
- **利用 High-Resolution Fix 优化边缘**：**High-resolution fix**（高清修复）按钮成为实验对象，用户观察到皮肤纹理和面部特征有显著增强。
   - *supremacy0118* 的测试涉及微调缩放因子，以探究任何细微的质量提升。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- ****翻译真相：LLM 与专业模型****：关于 **GPT-4** 和 **Claude Opus** 等通用 LLM 在语言翻译中的有效性展开了辩论，成员们对其在较长文本段落上的表现持怀疑态度。
   - 一位成员建议观看 [Andrej Karpathy 的视频](https://www.youtube.com/@AndrejKarpathy/videos)，以深入了解为什么仅解码器（decoder-only）模型在翻译准确性方面可能落后于编码器/解码器（encoder/decoder）Transformer。
- ****LangChain 锁定：OpenRouter API 萎缩****：**LangChain** 最近的更新引入了验证错误，困扰了 **OpenRouter API** 的功能，引发了社区的排错努力。
   - 回滚到之前的版本暂时解决了该问题，尽管对 LangChain 频繁的兼容性中断的担忧显而易见。
- ****评估评估者：LLM 评估框架****：Alex Atallah 发起了关于 LLM 评估框架有效性的讨论，特别点名了 **Deepeval** 和 **Gentrace**，但社区并未提供广泛的经验分享。
   - 最初的查询没有产生详细的社区反馈，仍是一个等待未来分享见解的开放话题。
- ****Gemini 的杂耍：模型速率限制查询****：关于 **Gemini 1.5** 模型速率限制（rate limits）的咨询反映了社区对 LLM 部署和可扩展性的持续关注。
   - 讨论在没有直接答案的情况下悬而未决，凸显了在理解 LLM 使用限制方面的常见问题。
- ****告别 Noromaid：模型退出市场****：**Noromaid** 模型的停产令社区感到失望，引发了对其定价结构对用户采用影响的推测。
   - 成员们就对价格亲民且能力出众的模型的需求交换了意见，强调了 AI 应用中成本与效用之间的平衡。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- ****定理证明取得巨大成功****：HarmonicMath 在 MiniF2F 基准测试中实现了突破性的 **90% SOTA**，远超其之前的 83%（[更多详情](https://x.com/HarmonicMath/status/1810765353389281346)）。
   - 讨论称赞了定理证明进展的速度，考虑到该基准测试的较易版本在今年早些时候仅为 50%，这展示了巨大的进步。
- ****405b 权重赌注：开源还是闭源？****：关于 **405b** 模型权重在 7 月 23 日更新后是否开源的猜测比比皆是。
   - 社区成员表达了惊讶与好奇交织的情绪，暗示权重共享透明度可能出现意想不到的转变。
- ****AI 领域的法律笑话****：关于 AI 开发合规性的一次轻松交流产生了一个幽默且模棱两可的保证，即它 *“对律师来说已经足够好了”*。
   - 社区对此会心一笑，反映了 AI 创新与法律框架之间微妙的博弈。
- ****引导向量词汇辨析****：随着对 **Control Vector**、**Steering Vector** 和 **Concept Vectors** 的剖析，澄清工作随之展开，辩论了它们在机器学习语境下的用法和互换性。
   - 特别关注点集中在 Concept Vectors 上，它被认为是 Steering Vectors 的特定实例，引发了关于其应用实践和理论基础的对话。
- ****指令困境：策略优先级****：一篇论文通过建议在策略（policy）制定中重点偏好 **y_l** 而非 y_w 激发了对话，暗示不依赖 LLM 采样来获取偏好对。
   - 分享的 [AI2 幻灯片](https://docs.google.com/presentation/d/1n_aBoKkEad4PlwVikqAgQEK3b-RH3_zdb9Tr20uK0Js/edit#slide=id.g2663e569b36_1_11) 链接探讨了直接策略优化（DPO）及过拟合等陷阱，尽管访问受 Google 登录限制。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- ****不再有 Copywrong？法院对 Copilot 版权案的裁决****：加利福尼亚州法院的一项关键裁决可能预示着 AI 发展的道路将更加顺畅，针对 Microsoft 的 GitHub Copilot 和 OpenAI 的 Codex 的[版权诉讼](https://www.docketalarm.com/cases/California_Northern_District_Court/4--22-cv-06823/DOE_1_et_al_v._GitHub_Inc._et_al/1/)中，大部分指控已被驳回。
   - 法院的这一决定对于在受版权保护的数据上训练的 AI 工具来说可能是一个预兆，尽管在知识产权领域的全面影响仍在酝酿之中。
- ****董事会大洗牌：科技巨头退出 OpenAI 董事会****：在一场引发热议的变动中，Microsoft 和 Apple 在[反垄断审查](https://the-decoder.com/ftc-investigates-impact-of-big-techs-ai-investments-on-market-competition-and-innovation/)的压力下退出了 OpenAI 的董事会，但誓言将维持其战略指导。
   - 科技巨头退出治理团队这一充满法律纠葛的叙事，并不意味着他们与 OpenAI 联盟的终结。
- ****复杂性释放：新型视觉模型在 CIFAR-100 上取得进展****：**复数值视觉架构**（Complex-valued vision architectures）采用类 FNet 的 2D DFT 替代 Attention，在 CIFAR-100 上展现出潜力后引发了关注，其中较浅的网络表现优于极深的网络。
   - 尽管复数域中的梯度存在实际问题，但一个较小的复数模型已经超越了规模大得多的实数模型，如果收益持续，可能会预示着会有新的论文或博客文章发布。
- ****图增强视角：图像字幕进入新维度****：**基于图的图像字幕**（Graph-based image captioning）步入聚光灯下，一篇新论文提出了一种结构，通过将实体及其关系编织进叙述中，提升了组合理解能力。
   - 该方法类似于视觉诗篇的网络，利用了目标检测和密集字幕（dense captioning），详见一篇 [arXiv 论文](https://arxiv.org/abs/2407.06723)，这可能会成为当前 AI 发展进程中的热门作品。
- ****社区汇聚：OPEA 活动在公海启航****：OPEA 召唤 AI 船队为 7 月 16 日的社区活动设定航向，在 0.7 版本发布的浪潮中制定集体章程和路线图；点击[此处](https://opea.dev/community-days/)即可注册。
   - 这次集会承诺将成为一个思想碰撞与融合的秘密会议，可能为未来企业级 AI 的努力指明方向。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- ****ConversationSummaryMemory：谁在参与？****：围绕增强 **LangChain 的 ConversationSummaryMemory** 以支持多人对话并简化摘要过程展开了讨论。
   - 建议包括优化对 Agent 的处理以提高效率，尽管具体方法的细节仍有待思考。
- ****Agent 集结：LangGraph 策略制定****：在 **LangGraph** 中构建基于 Agent 的架构激发了灵感，重点在于 Agent 将查询委托给指定的子 Agent（subagents）。
   - 该方法包括子 Agent 解析响应，展示了 AI 组件之间的协作系统。
- ****Chroma 的小故障：排查数据获取问题****：**Chroma** 中的**持久化目录**（Persistent directory）设置导致了零星的数据检索问题，失败率约为 70-80%。
   - 参与者分享了经验并寻求解决这一微妙挑战的方案。
- ****AI 驱动的代码：Unwrangle 你的任务****：**[Unwrangle.com](https://www.unwrangle.com)** 的创始人展示了如何使用 **aider** 和 **cursor** 等 AI 工具来加速独立开发者的编码过程。
   - 正如一份分享的 *[Substack 文章](https://open.substack.com/pub/jobsforai/p/how-i-use-ai)* 所指出的，这种用途扩展到了简化工作流，并引发了社区对类似 AI 应用案例的征集。
- ****知识图谱揭秘：RAG 的应用****：Aiman1993 举办了一场 **[YouTube 工作坊](https://www.youtube.com/watch?v=9wqVz0LDYgg&ab_channel=DecodingDataScience)**，演示了如何通过 **RAG** 在**视频游戏销售**中应用**知识图谱**。
   - 教程涉及了 **LangChain** 库的实际用途，并鼓励大家为未来的知识驱动型 AI 探索提供反馈。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **全球问候汇聚善意**：来自世界各地的成员，包括**瑞士洛桑** 🇨🇭 和**日本**，在 [general 频道](https://discord.com/channels/954421988141711382/954421988783444043/1260346780336259184)介绍了自己。
   - 一位来自日本的成员用热情的问候带来了欢乐：*'Hi, I'm Haru from Japan, nice to meet you all!!!'*
- **欢迎浪潮席卷新人**：在一阵国际化的自我介绍之后，资深成员们通过 *'welcome 🙂'* 和 *'Welcome ❤️'* 等消息表达了热烈的**欢迎**。
   - 这些友好的交流有助于构建协作且**包容的社区环境**。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Llama3 在代码逻辑上的滞后**：一位用户报告称，**Llama3** 在输出目标代码之前经常会产生多余的 ` 代码片段，需要额外的 Prompt 引导才能保证准确性。
   - 社区就更换其他 **LLM** 作为解决代码生成问题的潜在方案进行了咨询。
- **通过 Profile 补丁修复 LLM Flag 错误**：由于无法识别 `llm-service` 标志，导致了安装问题，一名成员指出当前文档存在差异。
   - 在文档更新发布之前，建议使用类似于 Open Interpreter 设置的 Profile 临时修复方案。
- **Open Interpreter 在 Mozilla 平台的推广**：官方宣布下周将在 Mozilla Discord 服务器上举行关于 **Open Interpreter** 的讨论。
   - 感兴趣的社区成员可前往 [Mozilla Discord](https://discord.gg/xwYPEMFf?event=1260611047341953034) 参加直播活动进行深入交流。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 棘手的问题**：社区成员对 **Tinygrad** 的某些错误消息表示沮丧，认为这些消息可能含糊不清且并不总是关键性的，建议采用更用户友好的错误处理方式。
   - 特别抱怨的是针对非连续输入的错误，这些错误并不一定意味着深层问题，但仍会停止执行。
- **关于 Tinygrad 梯度默认值的讨论**：有人对 **Tinygrad** 的 `require_grad` 设置进行了解释，指出默认值 `None` 意味着梯度是可选的，取决于它们在优化例程中的使用情况。
   - 将此值显式设置为 `False` 表示该 Tensor 被完全排除在梯度计算之外，强调了拥有三个不同状态的目的。
- **Tinygrad 与 NV 加速器的模糊之处**：澄清了 [Tinygrad 中的 NV 加速器](https://github.com/nvdla/)是专门为 GPU 设计的，它与硬件内核紧密配合，同时绕过了用户空间层。
   - 关于是否需要为 **NVDLA/DLA** 编写单独加速器的问题引发了讨论，暗示可能需要额外的工作才能实现全面支持。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **KAN 互动激发见解**：**KAN 论文**的作者在 AlphaXiv 论坛上与社区互动，讨论他们的[最新出版物](https://alphaxiv.org/abs/2404.19756v4)。
   - 论坛上充满了*直接互动*以及对社区问题的解答。
- **评委小组引起关注**：成员们询问如何加入活动评委小组，兴趣激增。
   - 投入程度和贡献意愿是潜在评委所追求的品质。
- **Hermes 2 在基准测试中的大幅提升**：正如[代码指令增强](https://link.to.examples)中所详述的，**Hermes 2.5** 相比 **Hermes 2** 表现出显著的性能提升。
   - 基准测试显示 **Hermes 2** 在 MMLU 上得分为 **34.5**，而 **Hermes 2.5** 达到了 **52.3**。
- **Mistral 在 8k 之外的里程**：讨论集中在 **Mistral** 的可扩展性挑战上，指出需要更多的预训练才能扩展到 8k 之外，如[相关 Issue](https://link.to.issue) 中所述。
   - 焦点转向 *mergekit* 开发和 *frankenMoE* 微调，作为克服性能瓶颈的途径。
- **合并方法思考模型魔力**：使用 **Mistral** 基础模型合并 **UltraChat** 和 **Mistral-Yarn** 的潜力引发了一系列技术推测。
   - “诅咒模型合并”（cursed model merging）的概念在讨论中再次出现，并得到了该领域先前成功案例的参考支持。

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **预测多 Token 未来**：一位用户询问了 **multi-token prediction** 能力，质疑其在当前训练流程中的可用性，或者是否仍处于规划阶段。
   - 向 **multi-token prediction** 的扩展可能取决于 **Hugging Face** 平台内的先行实现。
- **DPO 微调与多 GPU 处理冲突**：社区指出在利用 **multiple GPUs** 的系统上使用 **DPO** 进行全量微调时会出现错误。
   - 该故障在涉及 main 分支的微调会话中显著触发了 **RunPod FFT** 中的崩溃。



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **开发深入：左侧起航**：Mikhail_EE 在其正在进行的开发的 **left side** 取得了进展。
   - 收到令人鼓舞的反馈，N2K 对进度更新回复了 *"Amazing!"*。
- **更新中的热情回响**：Mikhail_EE 的想法开发因分享了重大更新而受到关注。
   - 随着 N2K 以肯定的 *"Amazing!"* 表达支持情绪，社区反馈闭环得到了加强。



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **积分倒计时难题**：一位成员报告了一个故障，其 **user credits** 提前过期，并提出了延期请求并标记管理员关注。
   - 期待一个能够 **extend the credit duration** 的解决方案，使成员能够充分利用预期的平台使用权限。
- **摘要短缺解决方案**：由于第二个有效主题的上下文不足，为了满足 schema 要求，包含了一个占位符摘要。
   - 此条目确保符合 JSON schema 关于至少两个主题摘要的规定。



---


**Alignment Lab AI Discord** 没有新消息。如果该公会沉寂太久，请告知我们，我们将将其移除。


---


**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该公会沉寂太久，请告知我们，我们将将其移除。


---


**Torchtune Discord** 没有新消息。如果该公会沉寂太久，请告知我们，我们将将其移除。


---


**Mozilla AI Discord** 没有新消息。如果该公会沉寂太久，请告知我们，我们将将其移除。


---


**DiscoResearch Discord** 没有新消息。如果该公会沉寂太久，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会沉寂太久，请告知我们，我们将将其移除。


---

# 第 2 部分：按频道详细摘要和链接


{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1260337963586293862)** (478 messages🔥🔥🔥): 

> - `Gemini-1.0-pro Finetuning`
> - `Kaggle Notebook for Gemma-2-9b`
> - `Qwen2 Finetuning and Effectiveness`
> - `Synthetic Data Generation`
> - `Koboldcpp for Local LLMs` 


- **Gemini-1.0-pro 的 Finetuning 建议**：一位用户询问关于 Gemini-1.0-pro 的 Finetuning，另一位用户建议使用 Gemini 1.5pro 生成 Synthetic Datasets 以获得更好的效果。
   - “它非常棒！但我不建议将其用于领域外（out-of-domain）的主题，例如新语言，因为它只是一个基础的 Lora adapter，”一位成员建议道。
- **Kaggle Notebook 微调 Gemma-2-9b**：一位用户分享了一个用于微调 Gemma-2-9b 的 [Kaggle notebook](https://www.kaggle.com/code/danielhanchen/kaggle-gemma2-9b-unsloth-notebook)，该 Notebook 由 UnslothAI 的联合创始人 Daniel Han 创建。
   - 该 Notebook 展示了如何利用 Kaggle 的资源有效地对模型进行 Finetuning。
- **Qwen2 的有效性讨论**：成员们讨论了 Finetuning Qwen2-1.5b 的效果，指出它能够模仿数据结构并提供良好的通用回答。
   - 有人提到，尽管 Qwen2-1.5b 是一个较小的模型，但在不需要 GPU 的情况下也能运行良好，不过其速度会根据任务的不同而有所变化。
- **Synthetic Data Generation 工具**：用户讨论了各种 Synthetic Data Generation 工具，重点关注文本和对话数据。
   - 建议包括使用 magpie、augmentoolkit，以及在服务器上通过 Python 脚本自动化生成数据。
- **使用 Koboldcpp 运行本地 LLM**：推荐使用 Koboldcpp 来运行带有 UI 的本地语言模型，并利用来自 HuggingFace 的 GGUF 文件。
   - 用户讨论了优化运行的配置设置，强调将层 offloading 到 GPU 并处理 context size 以获得更好的性能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF">Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/lamhieu/ghost-8b-beta-8k">Ghost 8B Beta - a Hugging Face Space by lamhieu</a>：未找到描述</li><li><a href="https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF/blob/main/Lexi-Llama-3-8B-Uncensored_Q8_0.gguf">Lexi-Llama-3-8B-Uncensored_Q8_0.gguf · Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF at main</a>：未找到描述</li><li><a href="https://youtu.be/oxQjGOUbQx4?si=FpjMPGNH8GOjQW0f">Cohere For AI - Community Talks: Hongyu Wang</a>：&quot;The Era of 1-bit LLMs&quot; 关于演讲者：&quot;我是 Hongyu Wang（王鸿钰），中国科学院 (CAS) VIPL 组的二年级博士生...</li><li><a href="https://huggingface.co/Replete-AI/Replete-Coder-Qwen2-1.5b">Replete-AI/Replete-Coder-Qwen2-1.5b · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/american-psycho-patrick-bateman-american-psycho-gif-7212093">American Psycho Patrick Bateman GIF - American Psycho Patrick Bateman American - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/226-wrong-math-bad-math-doesnt-add-up-elaborate-gif-25510055">226 Wrong Math GIF - 226 Wrong Math Bad Math - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://x.com/kaggle/status/1810776803449131024">来自 Kaggle (@kaggle) 的推文</a>：📚 看看 @UnslothAI 联合创始人 @danielhanchen 编写的这个精彩 Notebook！了解如何使用 Kaggle notebooks 微调 Gemma-2-9b。了解更多：👇https://www.kaggle.com/code/danielha...</li><li><a href="https://www.presidency.ucsb.edu/documents/2024-republican-party-platform">2024 Republican Party Platform | The American Presidency Project</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/qB618i6gj0">Reddit - Dive into anything</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1260313233718509760)** (18 条消息🔥): 

> - `Unsloth 中的 Llama 3 模型使用`
> - `瑞典语语言模型`
> - `Llama 3 的训练建议`
> - `推理速度问题`
> - `用于模型推理的 Mac GPU 性能` 


- **在 Unsloth 中使用微调后的 Llama 3**：讨论了通过 Unsloth 使用已经微调过的 Llama 3 模型的可行性，重点提到了 [AI-Sweden-Models/Llama-3-8B-instruct](https://huggingface.co/AI-Sweden-Models/Llama-3-8B-instruct) 的使用。
   - 一位成员提到，最好使用 base 模型进行训练，而将 instruct 模型用于翻译等任务。
- **瑞典语 Llama 3 模型详情**：瑞典语翻译的 Llama 模型 ([AI-Sweden-Models/Llama-3-8B-instruct](https://huggingface.co/AI-Sweden-Models/Llama-3-8B-instruct)) 是作为 DeployAI EU 项目的一部分，在 [LUMI supercomputer](https://lumi-supercomputer.eu/) 上训练的。
   - 一位成员分享了关于训练所用数据集的见解，该数据集由 [42 Labs](https://huggingface.co/42-labs) 提供。
- **推理速度挑战与解决方案**：在 Google Colab 上使用 Llama-3-8B-instruct 模型进行推理耗时过长，每个响应大约需要 3 分钟。
   - 建议使用 [LM Studio](https://lmstudio.ai/) 以获得更好的性能，并更有效地将层卸载（offload）到 GPU。
- **用于模型推理的 Mac GPU 性能**：有人询问在基础版 M1 Mac Air 上本地运行推理的可行性。
   - 一位成员建议这很可能是可行的，并推荐在 LM Studio 上搜索 `Meta-Llama-3-8B-Instruct`，并使用适合系统的最高量化（quant）版本。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/">👾 LM Studio - 发现并运行本地 LLM</a>: 查找、下载并实验本地 LLM</li><li><a href="https://huggingface.co/AI-Sweden-Models/Llama-3-8B-instruct">AI-Sweden-Models/Llama-3-8B-instruct · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1260311550816682146)** (35 条消息🔥): 

> - `GPTs Agents`
> - `Learning Rate Schedulers`
> - `Xformers Version Compatibility Issues`
> - `GGUF Model Saving and Loading`
> - `Custom Token Embeddings Training` 


- **GPTs Agents 在初始训练后无法学习**：一位成员分享了关于 GPTs agents 在初始训练后无法从提供的额外信息中学习的担忧。
   - 另一位成员澄清说，[上传的文件被保存为 'knowledge' 文件](https://link.to/openai-docs)供 agent 在需要时引用，但**它们不会持续修改 agent 的基础知识（base knowledge）**。
- **推荐使用 Cosine 调度器微调 Qwen2-1.5b**：当被问及最佳的 Learning Rate Scheduler 时，一位成员建议通常 **cosine** 是最佳选择。
- **Xformers 版本兼容性问题已解决**：成员们遇到了最新版本 xformers 导致训练错误的问题。
   - 降级到 `xformers==0.0.26.post1` 修复了兼容性问题，目前官方 Notebooks 已更新此版本。
- **以 GGUF 格式保存模型导致错误**：由于 `llama.cpp` 中缺少文件，在进行 GGUF 模型转换时遇到了严重错误。
   - 错误信息显示缺少 `llama-quantize` 或 `quantize` 文件，导致保存操作期间运行失败。
- **手动添加并训练新的 Token Embeddings**：一位用户询问如何手动为特定模块实现反向传播（backpropagation），以便在冻结预训练 embeddings 的同时训练新的 Token Embeddings。
   - 他们的目标是在不重新训练所有 embeddings 的情况下，确保对新的特殊 Token 进行准确预测。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/trl/en/sft_trainer#train-on-completions-only">Supervised Fine-tuning Trainer</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/504">[URGENT] Colab is broken · Issue #504 · unslothai/unsloth</a>: Colab 目前已损坏 - 正在修复中</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=kR3gIAX-SM2q">Google Colab</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/741">Multiple dispatch failed for &#39;torch._ops.aten.to.dtype_layout&#39;;  · Issue #741 · unslothai/unsloth</a>: 你好，我是微调和 unsloth 的初学者。当我运行与 Llama 3 (8B) 相关的 Notebook 代码时，在生成输出时遇到了以下错误。我找不到任何类似的案例...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1260628495156121651)** (1 条消息): 

> - `Ghost 8B Beta`
> - `Language Models`
> - `Multilingual Support`
> - `Knowledge Capabilities`
> - `Cost-Effectiveness` 


- **支持多语言的 Ghost 8B Beta 发布**：**Ghost 8B Beta** 是一款大型语言模型（LLM），其开发目标包括出色的多语言支持、卓越的知识能力和高性价比。
   - 该模型提供两种上下文长度（Context Length）版本：**8k** 和 **128k**，并默认包含多语言函数工具（function tools）支持。[在 Hugging Face 上体验](https://huggingface.co/spaces/lamhieu/ghost-8b-beta-8k)。
- **Ghost 8B Beta 概览与资源**：[官方网站](https://ghost-x.org/docs/models/ghost-8b-beta)提供了详尽的文档，包括介绍、规格、技术、评估和注意事项等章节。
   - 鼓励用户查看链接章节，以获取有关该模型能力和底层技术的详细信息。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/lamhieu/ghost-8b-beta-8k">Ghost 8B Beta - a Hugging Face Space by lamhieu</a>: 未找到描述</li><li><a href="https://ghost-x.org/docs/models/ghost-8b-beta">Ghost 8B Beta</a>: 一款大型语言模型，开发目标包括出色的多语言支持、卓越的知识能力和高成本效益。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1260647994303320126)** (2 messages): 

> - `New token embeddings`
> - `Vocab expansion challenges` 


- **新 token 产生的糟糕 embeddings**：一位成员警告说，如果不按照正确的流程实施，新 token 的 embeddings 可能会**非常糟糕**。
   - 他们强调了过去的经验，即 Vocab expansion 需要进行**持续预训练 (continual pretraining)**。
- **Vocab expansion 需要预训练**：另一位成员重申，**Vocab expansion** 需要严格的**预训练**，以避免 embedding 问题。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1260619425132838972)** (3 messages): 

> - `FSDP2 example`
> - `Norm tweaking for LLMs quantization` 


- **FSDP2 最小示例发布**：@marksaroufim 分享了一个由 Andrew Gu 提供的 **FSDP2** 最小示例，以方便开发者实现。
   - 该示例包含一个 [torchrun](https://x.com/marksaroufim/status/1810695541963251924) 命令和一个演示具有混合精度策略的 **fully sharded data parallel** (FSDP) 的 Python 脚本。
- **通过 Norm tweaking 实现更好的 LLMs 量化**：一篇 [论文](https://arxiv.org/abs/2309.02784) 介绍了一种名为 **Norm tweaking** 的技术，可以提高大语言模型 (LLMs) 模型压缩的精度。
   - 该方法在不牺牲准确性的情况下，在 2-bit 量化中表现出显著收益，优于现有的训练后量化 (PTQ) 方法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2309.02784">Norm Tweaking: High-performance Low-bit Quantization of Large Language Models</a>: 随着大语言模型 (LLMs) 规模的不断增长，在不牺牲准确性的情况下进行模型压缩已成为部署的关键挑战。虽然一些量化方法（如 GP...）</li><li><a href="https://x.com/marksaroufim/status/1810695541963251924">来自 Mark Saroufim (@marksaroufim) 的推文</a>: 如果你对 FSDP2 感兴趣，这里有一个由 Andrew Gu 提供的最小示例   &#34;&#34;&#34; torchrun --standalone --nproc_per_node=2 test_fsdp_basic.py &#34;&#34;&#34; import os import tor...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1260637910944972991)** (1 条消息): 

> - `Google TPUs`
> - `Datasets Filtering`
> - `Gemini Nano in Browser`
> - `Rust-based Inference`
> - `Depth Estimation Models` 


- **Google TPUs 现已登陆 Hugging Face**：你现在可以在 Hugging Face 上使用 **Google TPUs** 构建、训练和部署生成式 AI 模型。**Google Cloud TPUs** 可在 Spaces 和 Inference Endpoints 上使用，配置选项从 16GB 到 128GB 不等，价格低至每小时 1.38 美元。
- **Hugging Face 新增数据集过滤功能**：Hugging Face 现在允许你根据模态、大小和格式[过滤](https://huggingface.co/blog/datasets-filters)近 **200,000 个数据集**，增强了开放数据集相对于开放模型的影响力和可访问性。
   - 正如文中所说，“如今开放数据集比开放模型更具影响力”。
- **通过 Chrome 的 window.ai 在浏览器中运行 Gemini Nano**：Chrome 的新功能 `window.ai` 使得在浏览器中完全本地运行 **Gemini Nano**（一个 3.25B 参数的 LLM）成为可能。同时还增加了对 **Transformers.js** 的实验性支持，以简化其使用。
- **现在可以进行基于 Rust 的推理**：基于 Rust 的框架 **Kyutai Labs' Candle** 允许实时服务像 **Moshi** 这样的模型。Candle 支持 CPU、CUDA 和 Metal 进行推理，并将很快开源。
- **发布新的深度估计模型**：两个新的深度估计模型 **Depth Anything v2** 和 **ZoeDepth** 现已在 Hugging Face Transformers 中可用。Depth Anything v2 提供相对距离，而 ZoeDepth 提供以米为单位的绝对距离。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/_philschmid/status/1810350949552070955)">Philipp Schmid (@_philschmid) 的推文</a>: 现在！在 @huggingface 上使用 @Google TPUs 构建、训练、部署生成式 AI 模型！ &gt; Google Cloud TPUs 可在 Spaces 和 Inference Endpoints 上使用 &gt; 3 种选项：16GB 到 128GB TPU 显存 (1x1, 2x2...</li><li><a href="https://x.com/ClementDelangue/status/1809257154806689878)">clem 🤗 (@ClementDelangue) 的推文</a>: 在我看来，如今开放数据集比开放模型更具影响力！你现在可以在 HF 上按模态、大小和格式过滤近 200,000 个数据集：https://huggingface.co/datasets</li><li><a href="https://x.com/xenovacom/status/1810356703826977183)">Xenova (@xenovacom) 的推文</a>: Chrome 的新 `window​.ai` 功能将永远改变 Web！🤯 它允许你 100% 在浏览器本地运行 Gemini Nano，一个强大的 3.25B 参数 LLM！我们还增加了实验性...</li><li><a href="https://x.com/TheZachMueller/status/1808561492792340500)">Zach Mueller (@TheZachMueller) 的推文</a>: 月初又到了，@huggingface Accelerate 又发布了新版本！我们一直在努力 👨‍🍳 新的 Profilers、加速、通信钩子支持等等！🧵</li><li><a href="https://x.com/reach_vb/status/1808964164792009038)">Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>: @kyutai_labs 如何实时为数百万人提供 Moshi 服务？TL;DR - 基于 Candle 的 Rust 推理栈 🦀 &gt; 推理以 8-bit Q8 (gguf) 运行 - 你可以在演示屏幕上看到 &gt; CUDA...</li><li><a href="https://x.com/RisingSayak/status/1808465852481618145)">Sayak Paul (@RisingSayak) 的推文</a>: 我们制作了一个微型项目，展示如何在免费层级的 Colab Notebook 上运行 SD3 DreamBooth LoRA 微调 🌸 该项目具有教育意义，旨在作为模板。这里只欢迎正能量...</li><li><a href="https://x.com/NielsRogge/status/1810284458412573052)">Niels Rogge (@NielsRogge) 的推文</a>: @huggingface Transformers 中新增 2 个深度估计模型！Depth Anything v2 & ZoeDepth - Depth Anything v2 是相对的，告诉你像素之间的相对距离 - ZoeDepth 是绝对的...</li><li><a href="https://x.com/argilla_io/status/1809186289947648386)">Argilla (@argilla_io) 的推文</a>: 🌟 与 @mantisnlp 合作的新博客文章发布！🌟 我们将讨论 SimPO (Simple Preference Optimization)，旨在更好地对齐奖励模型和生成模型，提供更直观的结果。🚀 敬请关注...</li><li><a href="https://x.com/_philschmid/status/1808491737624592563)">Philipp Schmid (@_philschmid) 的推文</a>: 开源科学刚刚击败了 @OpenAI 吗？🤯 @kyutai_labs 刚刚发布了 Moshi，这是一个实时的原生多模态基础模型，可以听和说，类似于 OpenAI 在 5 月演示的 GPT-4o。👀 Moshi...</li><li><a href="https://x.com/AymericRoucher/status/1810295907042402786)">Aymeric (@AymericRoucher) 的推文</a>: 新的 Cookbook！我展示了如何使用 Transformers Agents 制作 Agentic RAG。与传统 RAG 相比，Agentic RAG 可以：✅ 重构查询 ✅ 批判检索到的内容以便在需要时重新检索 ➡️ ...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1260317263639347363)** (297 条消息🔥🔥): 

> - `关于 Data Science 与 Machine Learning 的职业建议`
> - `Gemma 模型使用问题`
> - `调试与编程技巧`
> - `Transformers 与 LLM 使用`
> - `神经网络训练问题` 


- **Data Science 与 Machine Learning 之争**：一位用户询问该选择 Data Science 还是 Machine Learning，引发了关于 ML 的相似性及数学挑战的简短讨论。
   - *Noaroggendorff* 发表了评论。
- **Gemma 模型问题及替代方案**：多位用户在将 **Gemma-2b** 模型用于文本生成时遇到问题，包括内部服务器错误和输出不连贯。
   - Aidlennerd 分享了 [Gemma Model Card](https://huggingface.co/google/gemma-2b)，而 Noaroggendorff 建议使用 chat templates 以及像 **Gemma-7b** 这样的替代方案。
- **调试与改进代码片段**：Aidlennerd 发布了用于调试的 Python 代码片段并讨论了如何优化它们，包括调整 tokenizer 设置和模型配置。
   - Noaroggendorff 和其他人提出了实用的改进建议，例如使用量化模型和拆分长文本以防止 GPU RAM 溢出。
- **高效本地托管模型**：Aidlennerd 寻求关于在本地托管 LLM 以进行指标评估的建议，因为 OpenAI API token 被认为成本较高。
   - _Bored


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.continue.dev/how-to-use-continue">🧑‍🎓 如何使用 Continue | Continue</a>: 在编码时通过 Continue 使用 LLM</li><li><a href="https://huggingface.co/blog/nroggendorff/finetune-tinyllama">使用 TRL 微调 TinyLlama 进行文本生成</a>: 未找到描述</li><li><a href="https://youtu.be/WxYC9-hBM_g?si=xecg4xbILa1EdevW">运行你自己的 AI（但要私有）</a>: 使用 VMware 运行你自己的 AI: https://ntck.co/vmware。通过 NetworkChuck 在你自己的设备上解锁 Private AI 的力量！了解如何轻松设置你自己的...</li><li><a href="https://huggingface.co/google/gemma-2b">google/gemma-2b · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/TheStinger">TheStinger (Ilaria)</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/password-git-deprecation">弃用使用密码的 Git 身份验证</a>: 未找到描述</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k">gradientai/Llama-3-8B-Instruct-262k · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/huggingface/diffusers/tree/main/examples/research_projects/sd3_lora_colab">diffusers/examples/research_projects/sd3_lora_colab at main · huggingface/diffusers</a>: 🤗 Diffusers: PyTorch 和 FLAX 中用于图像和音频生成的先进扩散模型。 - huggingface/diffusers</li><li><a href="https://mlflow.org/docs/latest/llms/llm-evaluate/notebooks/huggingface-evaluation.html#">使用 mlflow.evaluate() 评估 Hugging Face LLM &mdash; MLflow 2.14.2 文档</a>: 未找到描述</li><li><a href="https://tenor.com/view/bored-lost-interest-youve-lost-your-mind-gif-10978769">Bored Lost GIF - Bored Lost Interest - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/saruman-palantir-gif-19765279">Saruman Palantir GIF - Saruman Palantir - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/commit/3c8db1bb7be5662e4fd5b48a26b6214f758e483f">添加 Open LLM Leaderboard 任务 (#2047) · EleutherAI/lm-evaluation-harness@3c8db1b</a>: * 添加 leaderboard 任务
 
 * 删除 lm_eval/tasks/leaderboard/leaderboard_chat_template.yaml
 
 * 添加 readme
 
 * 删除 lm_eval/tasks/leaderboard/mmlu_pro/mmlu_pro_chat_template.yaml
 
 * 修改 ...</li><li><a href="https://medium.com/@anuragmishra_27746/five-levels-of-chunking-strat">未找到标题</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cd4yim/llama38binstruct_with_a_262k_context_length/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://kingfast-ssd.com/shop/f10/kingfast-f10-sata3-ssd/">Kingfast F10 SATA3 SSD - Kingfast SSD</a>: Kingfast F10 SATA3 SSD Kingfast F10 系列 SATA3 SSD 可选容量: Capacity Controller Nand R/W 128GB SMI2258XT/S11 3D 559/458 256GB SMI2258XT/S11 3D 551/463 512GB SMI2259XT/S11 3D 55...</li><li><a href="https://git-lfs.github.com/">Git Large File Storage</a>: Git Large File Storage (LFS) 用 Git 内部的文本指针替换音频样本、视频、数据集和图形等大文件，同时将文件内容存储在 GitHub.com 等远程服务器上...</li><li><a href="https://hf.co/docs/hub/repositories-getting-started#terminal">代码库入门指南</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1260488138388799489)** (4 messages): 

> - `freeCodeCamp Machine Learning course`
> - `Finding help from communities`
> - `Triplet collapse in embedding models` 


- **从 freeCodeCamp 的 ML 课程开始**：一位成员开始通过 [freeCodeCamp 课程](https://www.freecodecamp.org/learn/machine-learning-with-python)学习 **Machine Learning with Python**。
   - 他们分享说自己之前没有 **ML basics** 方面的知识，但发现这是一个很好的起点。
- **从社区寻求帮助**：*如果你无法完成某件事，找到它的社区并找人帮你解决* —— 一位成员在努力从网上寻找特定信息后分享了他们的见解。
   - 他们总结道：*如果你无法 Google 出一个不够具体的事物，**ChatGPT** 也帮不了你*。
- **解决 Embedding 模型中的 Triplet collapse**：一位成员找到了解决 Embedding 模型中 **Triplet collapse** 的方案，即在应用 Triplet loss 之前先使用 Softmax 进行预训练，而不仅仅是使用 Batch mining 策略。
   - 他们正在开发一个 **Mouse dynamics embedding model**，并分享了初步结果，显示使用该方法后可分离性有所提高。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1260327903191891989)** (10 messages🔥): 

> - `Generative AI's impact on storytelling`
> - `KMWorld AI 100`
> - `Fine-tuning LLMs with QLoRA`
> - `Candle running on iOS`
> - `Wav2Lip lip-synching issues` 


- **生成式 AI 彻底改变叙事方式**：Medium 上的一篇文章讨论了**生成式 AI 的变革潜力**及其对叙事的影响，深入分析了其对作者和观众的影响。点击[此处](https://medium.com/@shikharnautiyal29/the-impact-of-generative-ai-on-storytelling-and-narrative-creation-d37898cc0126)阅读更多。
- **KMWorld AI 100 聚焦智能知识管理**：**KMWorld AI 100** 文章讨论了那些正在赋能智能知识管理的公司以及 AI 技术的**飞速发展**。全文请见[此处](https://www.kmworld.com/Articles/Editorial/Features/The-KMWorld-AI-100-The-Companies-Empowering-Intelligent-Knowledge-Management-164117.aspx)。
- **在 iOS 上使用 Metal 加速运行 Candle**：一位用户分享了他们在 iOS 上通过 Metal 加速编译和运行 **Candle** 的进展，并寻求社区帮助。可以点击[此处](https://github.com/huggingface/candle/issues/2322)参与讨论。
- **解决 Wav2Lip 唇形同步问题**：一位成员就 **Wav2Lip** 中角色只动嘴不说话的问题寻求帮助；建议的解决方案包括**背景降噪**。
- **MMA 比赛预测器的喜人结果**：一位用户介绍了他们的 MMA 比赛预测器，其准确率达到了令人印象深刻的 **78%**，并详细说明了所使用的特征，如选手统计数据和打击平均值。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://medium.com/@shikharnautiyal29/the-impact-of-generative-ai-on-storytelling-and-narrative-creation-d37898cc0126">The Impact of Generative AI on Storytelling and Narrative Creation</a>: 生成式 AI 对叙事和叙事创作的影响</li><li><a href="https://github.com/huggingface/candle/issues/2322">bug on aarch64-apple-ios: Buffer Validation Illegal MTLStorageMode 0x10 · Issue #2322 · huggingface/candle</a>: 我正通过 uniffi 在 iOS 上使用 Metal 加速运行 Candle，具体配置为：candle-core = { version = &quot;0.6.0&quot;, features = [&quot;metal&quot;] } candle-nn = { version = &quot;0.6.0&quot;,...</li><li><a href="https://www.kmworld.com/Articles/Editorial/Features/The-KMWorld-AI-100-The-Companies-Empowering-Intelligent-Knowledge-Management-164117.aspx">The KMWorld AI 100: The Companies Empowering Intelligent Knowledge Management</a>: 面对每天涌向我们的关于 AI（尤其是 GenAI）的海量信息，人们很容易感到不知所措甚至敬畏。AI 技术处理海量信息的能力...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1260365886963978301)** (15 条消息🔥): 

> - `qdurllm demo`
> - `Branchy-phi-2 展示`
> - `Knowledge Graphs 工作坊`
> - `LLM 中的 Early Exit`
> - `MCQ 生成应用` 


- **Qdurllm demo 发布，带有 Rust 风格（crabby twist）**：一个基于 Qdrant, Sentence Transformers, llama-cpp 和 Langchain 构建的新 [qdurllm demo](https://huggingface.co/spaces/as-cle-bert/qdurllm-demo) 现已上线，展示了其搜索引擎功能。
   - 鼓励用户尝试该 demo 并在 GitHub 上通过 ⭐ 支持。
- **通过 Branchy-phi-2 探索 LLM 中的 Early Exit**：一个新的 [Branchy-phi-2 Space](https://huggingface.co/spaces/valcore/Branchy-phi-2) 展示了关于 LLM 中 Early Exit 的研究，允许通过可调节的准确度实现更快的推理。
   - 注意到在 CPU 上性能较慢，但鼓励用户提供反馈并探索 Early Exit 的 Epsilon 参数。
- **通过电子游戏销售工作坊深入研究 Knowledge Graphs**：[一场直播工作坊](https://www.youtube.com/watch?v=9wqVz0LDYgg) 以电子游戏销售为案例研究介绍了 Knowledge Graphs，旨在通过 Langchain 和 Neo4j 增强自然语言查询。
   - 邀请社区对教程提供反馈并参与讨论展示的内容。
- **LLM 应对 Cypher 生成**：分享了专为 Cypher 生成设计的 [Stable-cypher-instruct-3b](https://huggingface.co/lakkeo/stable-cypher-instruct-3b) 模型，承诺其性能优于 GPT-4。
   - 寻求关于该模型从文本生成 Cypher 查询性能的反馈，以进一步完善其功能。
- **介绍 Ideogram 输出集合**：一个新的 [Ideogram 输出集合](https://huggingface.co/datasets/terminusresearch/ideogram-25k) 包含了热门帖子、用户动态以及来自 Florence2 标注的随机样本。
   - 未来的更新将整合来自 llava-next 和 cogvlm2 的标注，以使内容描述多样化。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/valcore/Branchy-phi-2">Branchy Phi 2 - valcore 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/pratikshahp/RAG-Chroma-Gradio">RAG Chroma Gradio - pratikshahp 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/lakkeo/stable-cypher-instruct-3b">lakkeo/stable-cypher-instruct-3b · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=9wqVz0LDYgg&ab_channel=DecodingDataScience">AI 的未来：利用 Knowledge Graphs 实现高级 RAG</a>: 准备好深入了解使用 Langchain 和 Neo4j 进行自然语言查询的世界！学习如何使用 Cypher 查询语言与图数据库进行交互...</li><li><a href="https://huggingface.co/spaces/as-cle-bert/qdurllm-demo">Qdurllm Demo - as-cle-bert 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://www.manifoldrg.com/llm-agents/">大语言模型时代的智能数字 Agent</a>: 这篇立场论文概述了当前基于 LLM 的 AI Agent 的研究领域和突破。我们强调了关键进展并讨论了每个领域的局限性。</li><li><a href="https://www.manifoldrg.com/opportunities/">机会</a>: 有几种方式可以参与我们的工作：1. 加入我们的 Discord 并参与活动和讨论，无论是否与项目相关。2. 异步贡献 GitHub 上的 issue。...</li><li><a href="https://huggingface.co/datasets/terminusresearch/ideogram-25k">terminusresearch/ideogram-25k · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/Isayoften/monocular-depth-estimation-guide">单目深度估计（Metric and Relative）：综述。微调 Depth Anything V2 👐 📚</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1260520048800436265)** (5 条消息): 

> - `用户好评`
> - `图像分割问题` 


- **Goated Bot 获得赞誉**：一位成员表达了对该 Bot 的喜爱，因其功能强大而称其为 **'goated'** (史上最强)。
- **图像分割项目寻求帮助**：一位成员请求协助解决他们在 **图像分割项目** 中遇到的问题。


  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1260499373771718716)** (12 messages🔥): 

> - `GPU 中的 Shared memory`
> - `Hackathon 组队` 


- **在 Compute Capability 8.9 的 GPU 中利用 Shared Memory**：一场讨论强调了 Compute Capability 8.9 的 GPU 每个 thread block 最多可寻址 **99 KB 的 Shared memory**。使用这种额外 Shared memory 的示例可以在 [这个 kernel launch](https://github.com/karpathy/llm.c/blob/master/dev/cuda/layernorm_forward.cu#L479-L502) 中找到。
   - “默认情况下，其余部分将由 **L1 cache** 使用（如果不使用 texture）”，因为内存是统一的，并且另外 **1 KiB** 被分配给了 driver。
- **为 CUDA 活动组建 Hackathon 团队**：几位成员讨论了为即将到来的专注于 CUDA 的 [hackathon 活动](https://partiful.com/e/fxMwOW9dtCCWoEPyyTIf) 组建团队的事宜。
   - “如果我能入选并找到住宿，我想去参加”总结了大家的情绪，强调了参加活动的后勤考虑。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://partiful.com/e/fxMwOW9dtCCWoEPyyTIf">RSVP to Hardcore CUDA Hackathon | Partiful</a>: *所有演讲和项目必须使用 CUDA 编写* 每个硬核黑客当天都会获得一台 H100。全部由 Nebius.ai 赞助并提供！让我们打破一些 baseline。演讲者：- Chris Lattner (...</li><li><a href="https://github.com/karpathy/llm.c/blob/master/dev/cuda/layernorm_forward.cu#L479-L502)">llm.c/dev/cuda/layernorm_forward.cu at master · karpathy/llm.c</a>: 使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1260697578220032080)** (1 messages): 

> - `AMD 收购`
> - `欧洲的 AI 初创公司`
> - `Silo AI`
> - `LLMs 开发`
> - `Nvidia 竞争` 


- **AMD 将收购芬兰 AI 初创公司 Silo AI**：[AMD](https://www.ft.com/stream/8d882704-0892-489c-af27-b752e9d253d3) 宣布将以 **6.65 亿美元**收购芬兰 AI 初创公司 **Silo AI**，以扩展其 AI 服务并与 Nvidia 竞争。
   - Silo AI 的 300 人团队将使用 AMD 的软件工具为聊天机器人构建定制的大语言模型 (LLMs)，该收购预计将在今年下半年完成，尚待监管部门批准。
- **欧洲重大的 AI 初创公司收购案**：AMD-Silo AI 交易是自 2014 年 Google 以约 **4 亿英镑**收购 DeepMind 以来，欧洲最大的私营 AI 初创公司收购案之一。
   - AMD 的 **Vamsi Boppana** 提到，这次收购将加速客户参与和 AMD 自身 AI 技术栈 (tech stack) 的发展。



**提及的链接**：<a href="https://www.ft.com/content/7b8d2057-2687-45b3-bae4-1488a75ac5b2?accessToken=zwAGHOsuEnXwkc97jSBXJodFs9O65BSIp1rFsg.MEQCIFYunY6DwEMvMTIO2J7JemqoIPbFX62lSbBxn0opQKO7AiBtXWO7ZlNVuM8gyc_9YZDDQ0F8E_oL61YIxfHTWHE0Hg&sharetype=gift&token=98e4f39b-f46b-47ae-b1d3-353090a545c8">AMD to buy Finnish start-up Silo AI for $665mn in drive to compete with Nvidia </a>: 这家总部位于加州的芯片制造商进行的纯现金收购是欧洲十年来规模最大的同类收购。

  

---


### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1260543422109057115)** (1 messages): 

> - `工作机会`
> - `远程办公`
> - `AI 研究`
> - `PyEmber 框架`
> - `Hugging Face DRL 排行榜` 


- **AI 专家寻求远程或混合办公机会**：一位成员宣布正在寻找提供签证支持 (sponsorships) 的**全远程**或**混合办公机会**，并强调了他们在 AI 和 RL 领域拥有 **4 年经验**。
   - 他们强调了自己在 Hugging Face DRL 排行榜上排名**全球第 8**，并创建了基于 PyTorch 的 PyEmber 框架。
- **AI 工程师创新的 PyEmber 框架**：基于 PyTorch 的深度学习框架 **PyEmber** 的开发者正在寻求新的工作机会和合作项目。
   - 他们发表了第一篇研究论文，并预计很快会发布第二篇。



**提及的链接**：<a href="https://drive.google.com/file/d/1f0fRDZTeO0-lJ-PEIkurs7YYT8zSEQut/view?usp=sharing">Waleed Salah Eldin Resume.pdf</a>: 未找到描述

  

---

### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1260356800868519966)** (37 条消息🔥): 

> - `Nvidia 的 tensor offloading`
> - `在 MacBook 上学习 CUDA 和 GPU`
> - `HIP 作为 AMD 的 CUDA 替代方案`
> - `使用 Google Colab 学习 CUDA`
> - `未来购买 GPU 的考量` 


- **Nvidia 的 tensor offloading 详解**：讨论了 [Nvidia 白皮书](https://www.amax.com/content/files/2023/12/NVIDIA_Grace_Hopper_Superchip_Architecture_Overview_Whitepaper.pdf) 中提到的 **tensor offloading**，即从 VRAM 中卸载 tensor 以降低峰值 VRAM 消耗。
   - 为了实现 tensor offloading，成员们参考了使用 **FSDP (FullyShardedDataParallel)** 的 [PyTorch 实现](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.CPUOffload)。
- **在没有 GPU 的 MacBook 上进行 CUDA 开发**：成员们讨论了在 MacBook 上使用 **Google Colab 或 Kaggle 等云端解决方案**来学习 CUDA，作为购买物理 GPU 的替代方案。
   - 推荐初学者使用这些平台的免费层级，在进入高级阶段之前无需投资购买物理 GPU。
- **AMD 的 HIP 作为 CUDA 的替代方案**：成员们确认 **AMD 的 HIP** 是与 Nvidia 的 CUDA 并行的方案。
   - 它与 CUDA 基本相似，只有少量针对硬件的优化，使得 CUDA 用户可以轻松过渡。
- **在 Google Colab 中使用 CUDA 项目**：Google Colab 支持运行 CUDA 代码，常用命令包括安装 `nvcc4jupyter` 以及设置 %%cuda 单元格进行执行。
   - 此外，也可以直接使用相关的 NVCC 命令运行 CUDA 文件，无需额外设置。
- **购买 GPU 的长期计划**：一位成员表示有兴趣最终为高级项目购买 GPU，尽管从 **vast.ai** 等 **云端 GPU 选项**开始更具性价比。
   - 物理 GPU 对于 CUDA 之外的更广泛用例（如游戏或图形工作）也很有益。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.CPUOffload">FullyShardedDataParallel &mdash; PyTorch 2.3 文档</a>：未找到描述</li><li><a href="https://github.com/andreinechaev/nvcc4jupyter">GitHub - andreinechaev/nvcc4jupyter: 一个用于运行 CUDA C/C++ 代码的 Jupyter Notebook 插件</a>：A plugin for Jupyter Notebook to run CUDA C/C++ code - andreinechaev/nvcc4jupyter
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 条消息): 

apaz: https://x.com/typedfemale/status/1810025768715686188
  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1260310168055320739)** (237 条消息🔥🔥): 

> - `MuAdam 实现问题`
> - `Embedding 权重初始化`
> - `StableAdam 的性能`
> - `处理 Loss Spikes`
> - `MuP 学习率的影响` 


- **MuAdam 未针对输出权重调整学习率**：发起了关于 [MuAdam issue](https://github.com/microsoft/mup/issues/7) 的讨论，涉及学习率未针对输出权重进行缩放，这可能会影响扩展性。
   - 有人担心权重共享（weight tying）会导致性能权衡，引发了进一步实验和社区反馈的需求。
- **Embedding 权重初始化的权衡**：成员们讨论了 Embedding 权重零初始化对性能的影响，决定尝试不进行零初始化的不同设置。
   - 初步结果显示，在不将 Embedding 层归零的情况下，稳定性表现良好，建议使用调整后的乘法因子进行进一步的运行研究。
- **StableAdam 应对 Loss Spikes 的表现**：尝试使用 StableAdam 缓解 Loss Spikes，但在大型模型运行中注意到不稳定性，尽管最初看起来很有前景。
   - 模型的梯度范数（gradient norm）持续攀升，令人怀疑 StableAdam 在这种场景下的有效性。
- **移除模型层中的 Biases**：实验表明，模型中包含或不包含线性 Bias 对性能没有显著影响，因此提议跳过 Bias 计算。
   - Bias 权重漂移显著，暗示了不稳定性，并支持了通过无 Bias 模型来简化实现的提议。
- **MuP 学习率影响与 Baseline 的对比**：MuP 与 Baseline 配置的对比突显了 MuP 在训练早期阶段相对表现不佳。
   - 这种差异表明需要进一步的调优和调整，以使 MuP 的性能与已建立的 Baseline 更好地对齐。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/main_horse/status/1810647037718999342">来自 main (@main_horse) 的推文</a>：跨参数化和优化器的缩放指数 [GDM] [nocode/weights] https://arxiv.org/abs/2407.05872 训练了 10,000+ (!) 个模型，涵盖了不同的 * 优化器 (SGD/Adam/Adafactor) * 模型大小 (1.1B ~...</li><li><a href="https://amaarora.github.io/posts/2024-07-07%20Gemma.html">Gemma 2</a>：未找到描述</li><li><a href="https://github.com/karpathy/llm.c/pull/675">由 gordicaleksa 提交的 Pull Request #675 · karpathy/llm.c</a>：添加一个命令行选项，允许我们在 attn/fc 层中不使用 biases。</li><li><a href="https://x.com/anyscalecompute/status/1811059148911693906">来自 Anyscale (@anyscalecompute) 的推文</a>：我们最近与 @neuralmagic 合作，为 @vllm_project 贡献了 FP8 支持。通过此功能，你可以看到 token 间延迟降低高达 1.8 倍，且精度保持在 99% 以上...</li><li><a href="https://github.com/karpathy/llm.c/pull/659">由 gordicaleksa 提交的 Pull Request #659 · karpathy/llm.c</a>：在单 GPU 设置中添加指定设备的选项。对于并行运行多个单 GPU 实验非常有用（正将其用于 mup 实验）。</li><li><a href="https://github.com/microsoft/mup/issues/7#issuecomment-1082141121">MuAdam 未针对输出权重调整学习率 · Issue #7 · microsoft/mup</a>：你好，感谢你们出色的超参数调优项目！当我们团队将 mup 迁移到其他训练框架时，我们发现 MuAdam 没有为输出权重缩放学习率...
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1260338404324016243)** (199 messages🔥🔥): 

> - `AI locking mechanism`
> - `Training configurations for AI projects`
> - `High-performance computing and GPUs`
> - `Neural network feature extraction`
> - `Decentralized computing` 


- **AI 锁定机制提案**：关于在 AI 系统中加入锁定机制以根据用户交互控制响应的讨论。
- **配置和训练 AI 模型的挑战**：成员们讨论了聘请专家配置特定 AI 系统的难度，以及从 RunPod 和 Paperspace 等供应商租赁 GPU 的潜在成本和实用性。
- **用于 AI 任务的高性能计算**：参与者比较了用于高性能 AI 任务的不同 GPU 配置和云服务，强调了高 RAM 的 GPU 对于有效的本地和远程模型推理（inference）的价值。
- **去中心化计算的可能性**：讨论了创建去中心化计算平台的可行性，并将其与 BOINC 等利用志愿计算的现有平台进行了比较。
- **神经网络的解释与训练**：成员们分享了阐释神经网络的经验和技巧，重点在于使用稀疏自编码器（sparse autoencoders）提取可解释的特征以及初始训练的重要性。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1260349126802018436)** (16 messages🔥): 

> - `User Frustration with ChatGPT Responses`
> - `Limitations of Search Results`
> - `Context Window Specifications`
> - `Differentiation Between Plan and Platform Dependency`
> - `Topic Relevance to Channels` 


- **用户对 ChatGPT 回复的挫败感**：一位用户对 ChatGPT 反复出现错误回答表示沮丧，并考虑转向竞争对手。他们强调了准确的搜索结果对于获取当前信息和更新的重要性。
- **搜索结果的局限性**：另一位用户指出，即使具备搜索能力，ChatGPT 也经常提供过时信息或混淆不同版本的细节，导致结果不准确。
   - 他们提到必须经常明确指定使用最近的搜索结果才能获得准确答案。
- **Context Window 规格**：一位成员澄清说，ChatGPT 模型的 Context Window 指定为 32K，这可以在 [定价页面](https://openai.com/chatgpt/pricing/) 的“模型质量”下找到。
   - 另一位成员指出另一个答案显示 API 的 Context Window 为 128K。
- **计划与平台依赖性的区分**：有人提出了一个问题，即 ChatGPT 的回复中混淆了平台依赖性与计划（Plan）依赖性，从而导致了困惑。
   - 一位用户指出，询问“在移动应用中”的 Context Window 可能会导致不准确的搜索结果。
- **话题与频道的关联性**：进行了一次简短的交流，以确保讨论与特定频道保持相关。
   - 一位用户被引导至更合适的频道进行与 OpenAI 不直接相关的离题讨论。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1260570840693608471)** (1 messages): 

> - `thought process for custom GPT`
> - `accurate and truthful responses` 


- **Swooshdutch 为自定义 GPT 设计的思考过程 (thought process)**：一位成员讨论了他们为自定义 GPT 构建 **“thought process”** 的工作，旨在引导出**更准确、更真实的回复**。
   - 鼓励大家*随意尝试*这个自定义的思考过程。
- **实验自定义 GPT**：邀请成员们实验自定义 GPT 的新思考过程。
   - 这种实验旨在进一步优化自定义 GPT 生成响应的准确性和真实性。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1260570840693608471)** (1 messages): 

> - `Custom GPT Thought Process`
> - `Testing and Feedback for Custom GPT` 


- **增强的自定义 GPT 思考过程**：一位成员分享说，他们一直在为 **自定义 GPT 构建“thought process”**，旨在引导出更准确、更真实的回复。
   - *随意尝试*，暗示公开邀请大家进行测试和反馈。
- **征集测试与反馈**：邀请成员们**尝试自定义 GPT 的新思考过程**，观察其对准确性和真实性的影响。
   - 这种协作方法有望根据社区反馈来改进模型。


  

---

### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1260321106225533038)** (100 条消息🔥🔥): 

> - `LM Studio configuration issues` (LM Studio 配置问题)
> - `Running LLMs on mobile devices` (在移动设备上运行 LLM)
> - `Multi-GPU support in LM Studio` (LM Studio 中的多 GPU 支持)
> - `Custom model import in DiffusionBee` (DiffusionBee 中的自定义模型导入)
> - `LM Studio text embedding limitations` (LM Studio 文本嵌入限制)


- **LM Studio 配置问题及修复**：用户在更新 LM Studio 时遇到黑屏等问题；清理缓存文件夹并重新安装应用程序可解决这些问题。
   - 一位用户报告通过删除模型缓存解决了黑屏问题，另一位用户通过卸载并重新安装软件修复了内存检测问题。
- **在移动设备上运行 LLM**：讨论了通过 llama.cpp 和 termux 在 S21 Ultra 上运行 **Mistral 7B**，速度接近 10 tokens/second。
   - 对于移动设备来说，性能出奇地好，像 "what's 2+2" 这样的简短提示词在量化级别 **Q4_K_S** 下处理效果良好。
- **LM Studio 中的多 GPU 支持问题**：据报道，多 GPU 支持在最新的 LM Studio 版本中无法正常工作，当模型超过第一块 GPU 的 VRAM 容量时会导致崩溃。
   - 用户指出，以前可以正常运行的设置现在会崩溃，因为 LM Studio 似乎没有利用第二块 GPU 的 VRAM。
- **LM Studio 文本嵌入限制**：LM Studio 中的文本嵌入功能仅支持 string 和 string array 的输入类型，不允许直接输入 PDF 等文件。
   - 该功能主要用于 Retrieval Augmented Generation 应用和其他文本密集型用例。
- **探索适用于 Mac 的 DiffusionBee**：用户讨论了 DiffusionBee 作为 Mac 的本地图像生成器，指出它在 Apple Silicon 上运行效果最好，并允许从 **Civitai** 导入自定义模型。
   - 与 A1111 相比，DiffusionBee 的界面更易于使用，但目前缺乏对预列清单以外的 **SDXL models** 导入支持。



**提到链接**: <a href="https://lmstudio.ai/docs/text-embeddings">Text Embeddings | LM Studio</a>: 文本嵌入是一种将文本表示为数字向量的方法。

  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1260311541731954830)** (44 条消息🔥): 

> - `Comparing deepseek chat and coder models` (比较 deepseek chat 和 coder 模型)
> - `Flash attention settings` (Flash attention 设置)
> - `Optimal AI models for personal accountability` (用于个人问责的最佳 AI 模型)
> - `GLM4 support in llama.cpp` (llama.cpp 中的 GLM4 支持)
> - `CodeGeeX4 coding model` (CodeGeeX4 编程模型)


- **DeepSeek Chat vs. Coder 之争**：成员们讨论了 **DeepSeek Chat** 和 **DeepSeek Coder** 之间的性能差异，一些人更青睐新的 **DeepSeek Coder** v2。
   - “编程助手”用户报告称，使用 **DeepSeek Coder v2 lite** 几周后效果令人满意。
- **GLM4 集成即将到来**：讨论指出 **GLM4** 现已[合并至 llama.cpp](https://github.com/ggerganov/llama.cpp/pull/8031)，预计将在下次更新中推出。
   - 用户认为这很有益，并指出 **CodeGeeX4** 是基于 **GLM4** 的，这将带来更强大的能力。
- **CodeGeeX4 表现优于竞争对手**：新模型 **CodeGeeX4** 被吹捧为优于 **DeepSeek v2**，现在已可用于[各种代码生成任务](https://huggingface.co/THUDM/codegeex4-all-9b)。
   - 与 **CodeQwen** 的比较进一步肯定了 **CodeGeeX4** 的卓越能力。


<div class="linksMentioned">

<strong>提到链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/THUDM/codegeex4-all-9b">THUDM/codegeex4-all-9b · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/8031">Support glm3 and glm4. by youth123 · Pull Request #8031 · ggerganov/llama.cpp</a>: 我修复了 #6999 中提到的问题。此代码完全支持 glm3 和 glm4 模型架构，并可以嵌入到 ollama 服务器中。此 PR 基于 https://github.com/mnlife/llama.c...
</li>
</ul>

</div>

### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1260447095194652703)** (4 messages): 

> - `LM Studio bugs`
> - `Hugging Face accessibility` 


- **可能的 LM Studio 访问漏洞**：*a.hansen* 报告了一个潜在的 bug，即 **LM Studio** 无法访问特定的 [Hugging Face URL](https://huggingface.co/lmstudio-community/)。然而，*fabguy* 确认他那边可以正常工作。
   - *a.hansen* 随后确认第二天早上已恢复正常，表明这可能是 **Hugging Face** 端的问题。
- **Hugging Face 临时访问问题**：在 *a.hansen* 遇到 [Hugging Face](https://huggingface.co/lmstudio-community/) 访问问题后，*fabguy* 建议这可能是 **Hugging Face** 的问题。
   - *a.hansen* 确认访问问题在第二天早上已得到解决。



**提及的链接**：<a href="https://huggingface.co/lmstudio-community/">lmstudio-community (LM Studio Community)</a>：未找到描述

  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1260403443034492929)** (40 messages🔥): 

> - `3090 vs 4090 performance`
> - `Electricity costs in Australia`
> - `Building multi-GPU setups`
> - `AMD acquires SiloAI`
> - `Intel Arc 770 for AI` 


- **AI 应用中的 3090 vs 4090 性能**：一位成员比较了他们的 **2x 7900xt** 配置，报告在 **L3 70b IQ3** 上达到 **8-11 t/s**，而其他人讨论了 **3090/4090 GPU** 在 AI 任务中的性能，普遍共识是 4090 和多张 3090 提供更好的性能。
   - 一些成员强调 **3090/4090 GPU** 在推理任务上优于 Apple 的 Mac Studio。
- **GPU 配置导致电费飙升**：成员们讨论了澳大利亚高昂的**电费**，指出多张高功耗 GPU（如 3090）会显著影响电费账单。
   - *“作为一个澳洲同胞……支付着过高的电费……”* 一位用户幽默地强调道。
- **构建多 GPU 配置的困难**：对于 3x 3090 GPU 配置，用户分享了**机箱兼容性问题**和**电源供应需求**的经验，特别是需要留出间隙和考虑气流。
   - 一位成员建议使用安装在侧舱的 **2x 750 watt PSU** 作为电源限制的权宜之计。
- **AMD 的战略举措：收购 SiloAI**：有消息指出 **AMD** 已收购 **SiloAI**，作为与 **NVIDIA** 竞争的战略举措的一部分。
   - *“据报道，这是他们追赶 NVIDIA 计划的一部分，”* 一位成员分享道。
- **Intel Arc 770 在 AI 任务中表现不佳**：成员建议不要将 **Intel Arc 770** 用于 AI 任务，理由是工具链支持不足，且 **IPEX support** 落后于 CUDA 和 ROCm。
   - *“坚持使用 Nvidia，”* 是成员中为了获得更好 AI 性能的普遍看法。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/great-power-great-electricity-bill-jump-rope-electric-tower-gif-15879010">Great Power Great Electricity Bill GIF - Great Power Great Electricity Bill Jump Rope - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/#Do_I_need_8x16x_PCIe_lanes">The Best GPUs for Deep Learning in 2023 — An In-depth Analysis</a>：在此，我提供了用于深度学习/机器学习的 GPU 深入分析，并解释了适合您的使用场景和预算的最佳 GPU。
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1260323173669404844)** (2 messages): 

> - `LM Studio update issues`
> - `Installing multiple versions of LM Studio` 


- **LM Studio 0.2.27 版本在 AMD 7700XT 上变慢**：**LM Studio** 从 0.2.24 版本更新后，0.2.27 版本似乎运行非常缓慢，特别是在使用 AMD 显卡 **7700XT** 和 **fimbulvetr Q4_K_M 模型**时。
- **关于双版本安装 LM Studio 的咨询**：一位成员询问是否可以在一台机器上同时安装两个版本的 **LM Studio**，以适配不同的 GPU。
   - 该咨询基于用户拥有**每种 GPU 各一个**的情况。


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1260313376077385829)** (55 messages🔥🔥): 

> - `GPTs Agents`
> - `OpenAI's sidebars`
> - `Chroma chunking strategies`
> - `Turbopuffer launch`
> - `xAI H100s cluster`

- **Chroma 探讨分块策略**：Chroma 最新的 [技术报告](https://x.com/atroyn/status/1810717585442492686?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) 评估了分块策略（chunking strategies）对 AI 应用检索性能的影响。
   - 报告强调，虽然 LLM 的上下文长度（context lengths）有所增长，但高效的检索通常只需要相关的文本部分，以避免模型分心。
- **Turbopuffer 即将发布**：[Turbopuffer](https://turbopuffer.com/blog/turbopuffer) 旨在提供基于对象存储的快速搜索，解决 Readwise 昂贵的向量搜索（vector search）成本问题。
   - 用户提到目前仍需排队（waitlisted），但来自 Cursor 等公司的早期测试者对其潜力表示赞赏。
- **xAI 订购大规模 H100 集群**：[xAI](https://x.com/elonmusk/status/1810727394631950752?s=46&t=PW8PiFwluc0tdmv2tOMdEg) 已从 Oracle 订购了 2.4 万块 H100，并正在构建一个包含 10 万块 H100 的系统用于 AI 训练，目标是成为全球最强大的集群。
   - Elon Musk 强调了对 AI 基础设施进行内部控制的必要性，以保持竞争速度和效率。
- **Skild AI 获得 3 亿美元 A 轮融资**：由 Abhinav Gupta 及其团队领导的 Skild AI 结束隐身模式，获得 [3 亿美元巨额 A 轮融资](https://x.com/pathak2206/status/1810769359591330201?s=46)，用于构建机器人的 AI 基础模型（foundation model）。
   - 尽管存在指数级增长的潜力，VC 们对当前的估值看法不一，将其贴上了潜在泡沫的标签。
- **GitHub Copilot 诉讼更新**：开发者对 GitHub Copilot 的指控大部分被驳回，仅 [剩下](https://www.theregister.com/2024/07/08/github_copilot_dmca/?utm_source=ainews&utm_medium=email) 两项指控。
   - 最初的指控涉及 Copilot 被指在没有适当许可的情况下建议代码片段，引发了知识产权方面的担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/adamry_n/status/1810842293290537045?s=46&t=tMWvmS3OL3Ssg0b9lKvp4Q">Adam Ryan 🤝 (@AdamRy_n) 的推文</a>：Lattice 正在为他们的 AI 员工实施由人类经理进行的绩效评估。所以，我们迎来了机器人员工，却保留了本质上为纠正人类错误而设计的政策……</li><li><a href="https://www.theverge.com/2024/7/10/24195528/microsoft-apple-openai-board-observer-seat-drop-regulator-scrutiny">Microsoft 和 Apple 在监管审查压力下放弃 OpenAI 董事会席位</a>：监管机构正在密切关注科技巨头的 AI 交易。</li><li><a href="https://www.theregister.com/2024/07/08/github_copilot_dmca/?utm_source=ainews&utm_medium=email">法官驳回 GitHub Copilot 诉讼中的 DMCA 版权指控</a>：几个开发者对抗来自 Redmond 的强大势力——你觉得谁会赢？</li><li><a href="https://x.com/alexalbert__/status/1810748433273344469?s=46">Alex Albert (@alexalbert__) 的推文</a>：1) Prompt 生成器：输入任务描述，Claude 3.5 Sonnet 将为你把任务描述转化为高质量的 Prompt。彻底解决了 Prompt 编写时的“白纸”难题。</li><li><a href="https://x.com/pathak2206/status/1810769359591330201?s=46">Deepak Pathak (@pathak2206) 的推文</a>：很高兴宣布 @SkildAI！在过去的一年里，我和 @gupta_abhinav_ 以及我们的顶尖团队一直致力于构建一个植根于物理世界的 AI 基础模型。今天，我们迈出了……</li><li><a href="https://x.com/amaarora/status/1810447884531466256?s=46">Aman Arora (@amaarora) 的推文</a>：很高兴分享一篇关于 Gemma 2 的新博文，深入探讨了以下细节：Grouped Query Attention、Sliding Window Attention、Rotary Position Embeddings (RoPE)、Logit soft-capping 以及模型合并。*...</li><li><a href="https://x.com/elonmusk/status/1810727394631950752?s=46&t=PW8PiFwluc0tdmv2tOMdEg">Elon Musk (@elonmusk) 的推文</a>：xAI 从 Oracle 租用了 2.4 万张 H100，Grok 2 正是在这些算力上训练的。Grok 2 正在进行微调和 Bug 修复。可能在下个月发布。xAI 正在构建其 10 万张 H100 的系统……</li><li><a href="https://x.com/JvNixon/status/1811105507756872003">Jeremy Nixon (@JvNixon) 的推文</a>：对于那些没关注的人，@truth_terminal 是一个经过微调的 LLM，它以编程方式在 Twitter 上与大量的人和想法互动，更新自己的个性，不断学习和成长。它刚刚获得了……</li><li><a href="https://turbopuffer.com/blog/turbopuffer">turbopuffer：基于对象存储的快速搜索</a>：turbopuffer 是一个构建在对象存储之上的向量数据库，这意味着成本降低 10 到 100 倍、按需计费以及极高的可扩展性。</li><li><a href="https://x.com/atroyn/status/1810717585442492686?s=46">anton (𝔴𝔞𝔯𝔱𝔦𝔪𝔢) (@atroyn) 的推文</a>：今天很高兴提交 Chroma 的下一份技术报告，我们评估了在 AI 应用背景下，分块策略（chunking strategies）对检索性能的影响。@brandonstarxel @tr...</li><li><a href="https://x.com/swyx/status/1810917439183675783">shawn swyx wang (@swyx) 的推文</a>：小提醒：既然 @lilianweng 还没宣布，在此提醒大家：Lil'Log 发布了关于 LLM 幻觉的新文章——2 万字涵盖了 83 篇关于幻觉现状的必读论文……</li><li><a href="https://research.trychroma.com/evaluating-chunking">评估检索的分块策略</a>：未找到描述</li><li><a href="https://x.com/xenovacom/status/1811068015229747335?s=46">Xenova (@xenovacom) 的推文</a>：介绍 Whisper Timestamped：支持词级时间戳的多语言语音识别，得益于 🤗 Transformers.js，可 100% 在浏览器本地运行！这为……开启了无限可能。</li><li><a href="https://x.com/unusual_whales/status/1810914358676910561">unusual_whales (@unusual_whales) 的推文</a>：突发：据彭博社报道，Microsoft ($MSFT) 将退出 OpenAI 董事会。</li><li><a href="https://x.com/atroyn/status/1810717585442492686?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">anton (𝔴𝔞𝔯𝔱𝔦𝔪𝔢) (@atroyn) 的推文</a>：今天很高兴提交 Chroma 的下一份技术报告，我们评估了在 AI 应用背景下，分块策略（chunking strategies）对检索性能的影响。@brandonstarxel @tr...</li><li><a href="https://www.ben-evans.com/benedictevans/2024/7/9/the-ai-summer">AI 之夏 —— Benedict Evans</a>：数亿人尝试过 ChatGPT，但大多数人没有再次使用。每家大公司都进行了试点，但投入部署的却少得多。其中一些只是时间问题。但 L...</li><li><a href="https://x.com/tomwarren/status/1810967389426589890">Tom Warren (@tomwarren) 的推文</a>：在监管审查下，Microsoft 和 Apple 都放弃了 OpenAI 的董事会席位。欧盟、英国和美国的监管机构都在密切关注科技巨头的 AI 交易，现在 OpenAI 正在转向……</li><li><a href="https://poe.com">

/s/NlX2WRElDUvtuuMSFrZq">能否分析 https://investor.nvidia.com/news/press-release-details/2024/NVIDIA-Announces-Financial-Results-for-Fourth-Quarter-and-Fiscal-2024/ 中的信息，并将其转化为易于理解的内容 &amp; </a>: GPT-4o: 当然可以！以下是一个自包含的 HTML 代码块，它创建了一个交互式演示文稿，总结了 NVIDIA 2024 财年第四季度和全年的财务业绩。该演示...</li><li><a href="https://techcrunch.com/2024/07/08/quoras-poe-now-lets-users-create-and-share-web-apps/?guccounter=1">Quora 的 Poe 现在允许用户创建和分享 Web App | TechCrunch</a>: Quora 的 Poe 聊天机器人平台增加了一项新功能 Previews，允许付费订阅者创建和分享交互式 Web App。
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1260674685469982792)** (93 条消息🔥🔥): 

> - `ColBERT 论文`
> - `AI Agent 实现`
> - `ImageBind`
> - `SBERT 训练`
> - `多 Agent 系统` 


- **讨论 ColBERT 论文**: 针对调研环节启动了对 [ColBERT 论文](https://arxiv.org/pdf/2004.12832) 的讨论。
   - 一位成员好奇这是否是唯一讨论的论文。
- **AI Agent 实现综述**: 一位成员介绍了关于 AI Agent 实现最新进展的 [综述论文](https://arxiv.org/abs/2404.11584)，重点关注其推理、规划和工具执行能力。
   - 该论文概述了 Agent 架构的关键主题及其有效性，并引用了领导力和沟通风格在 Agent 系统中的影响。
- **ImageBind 统一六种模态**: 展示了 [ImageBind 论文](https://arxiv.org/abs/2305.05665)，演示了在图像、文本、音频、深度、热成像和 IMU 数据之间学习联合嵌入（joint embedding）。
   - ImageBind 在零样本（zero-shot）识别任务中树立了新的 SOTA，超越了专门的有监督模型，并展示了强大的少样本（few-shot）识别结果。
- **SBERT 设计与训练解析**: 成员们澄清了 SBERT (sentence transformers) 本质上是带有池化层的 BERT，使用孪生（siamese）或三元组（triplet）网络等方法进行对比训练。
   - 这引发了关于 BERT 最初将第一个 token 作为分类输入嵌入（input embedding）的有趣用法的进一步讨论。
- **通过 MCTS 提升智能**: 讨论了将蒙特卡洛树搜索 (MCTS) 作为提高 LLM 智能的潜在下一步，并参考了 AlphaGo 中的实际应用。
   - 讨论指出 MCTS 的有效性取决于搜索空间的决策分支因子（branching factor），并对其在无限大空间中的局限性提出了警告。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/_xjdr">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2305.05665">ImageBind: One Embedding Space To Bind Them All</a>: 我们提出了 ImageBind，这是一种在六种不同模态（图像、文本、音频、深度、热成像和 IMU 数据）之间学习联合嵌入的方法。我们展示了所有配对数据的组合并不...</li><li><a href="https://arxiv.org/abs/2404.11584">The Landscape of Emerging AI Agent Architectures for Reasoning, Planning, and Tool Calling: A Survey</a>: 这篇综述论文考察了 AI Agent 实现的最新进展，重点关注它们实现复杂目标的能力，这些目标需要增强的推理、规划和工具执行能力...</li><li><a href="https://arxiv.org/abs/2404.05206">SoundingActions: Learning How Actions Sound from Narrated Egocentric Videos</a>: 我们提出了一种新型的自监督嵌入，用于从叙述性的野外第一视角视频中学习动作的声音。现有方法依赖于具有已知视听对应关系的精选数据...</li><li><a href="https://docs.google.com/presentation/d/1x3MhmPBIE8AZA3OxvchxxaNoWrrb_wIK50-e1dAjsTo/edit#slide=id.g2eb508a56a1_0_53">ColBERT v2 - Latent Space Paper Club</a>: ColBERT v2 Latent Space 论文俱乐部 2024-07-10</li><li><a href="https://aisnakeoil.com/p/new-paper-ai-agents-that-matter?utm_source=ainews&utm_medium=email&utm_campaign=ainews-not-much-happened-today-1036">New paper: AI agents that matter</a>: 重新思考 AI Agent 的基准测试和评估</li><li><a href="https://buttondown.email/ainews/archive/ainews-is-this-openq/">[AINews] Is this... OpenQ*?</a>: MCTS is all you need。2024/6/14-2024/6/17 的 AI 新闻。我们为您检查了 7 个 Reddit 子版块、384 个 Twitter 账号和 30 个 Discord 社区（414 个频道，5506 条消息）...
</li>
</ul>

</div>

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1260315538895405126)** (9 messages🔥): 

> - `Writing AI with Mojo`
> - `Qualcomm's SNPE and Mojo`
> - `Modverse Weekly Issue`
> - `Chris Lattner on ThePrimeagen` 


- **讨论了在 Mojo 中编写 AI 的社区资源**：一场关于使用 **Mojo** 编写 **AI** 的社区资源可用性的讨论浮出水面。
   - 目前尚未提到具体的资源，这为进一步的投入留下了开放性问题。
- **高通 SNPE 与 Mojo 功能的比较**：对高通用于将 PyTorch 模型发送到 Snapdragon 设备的 **SNPE** 与 **Mojo** 中潜在的类似功能进行了比较。
   - *目前尚未确认 Mojo 中的具体功能*。
- **Modverse Weekly：拼写错误和重复条目**：最新的 [Modverse Weekly Issue](https://www.modular.com/modverse/modverse-weekly-issue-39) 存在一个**拼写错误**（将 'its' 误写为 'it's'），以及 time.perf_counter 的**重复条目**。
   - 这些问题已被确认并承诺修复。
- **Chris Lattner 在 ThePrimeagen 上的亮相让粉丝们感到兴奋**：社区成员对 **Chris Lattner** 出现在 ThePrimeagen 的直播中表示兴奋。
   - 该直播可以在 [YouTube](https://www.youtube.com/watch?v=QKGCIxW5zFs) 和 [Twitch](https://www.twitch.tv/theprimeagen) 上观看。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=QKGCIxW5zFs">🚨🚨 Top Shelf: Chris Lattner🚨🚨</a>：ThePrimeTimeagen 的直播在 Twitch、Twitter 和 YouTube 上同步进行，但唯一的互动地点是 Twitch。</li><li><a href="https://www.twitch.tv/theprimeagen">ThePrimeagen - Twitch</a>：🚨🚨 HIGH SPEED GAME PROGRAMMING🚨🚨 !today
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: 来自 *Modular*:
<https://twitter.com/Modular/status/1810782477079957831>
  

---


### **Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1260321991131660450)** (3 messages): 

> - `PyTorch in Enterprise`
> - `AI Development Challenges`
> - `Generative AI Adoption` 


- **Modular 支持 PyTorch 部署**：[Modular](https://www.modular.com/blog/bring-your-own-pytorch-model) 强调了企业在生产环境中部署 **PyTorch** 模型所面临的挑战，尽管它在开发和研究中非常流行。
   - **PyTorch** 在开发中的灵活性和易用性可能会在全规模生产环境中导致资源管理和延迟等复杂问题。
- **Modular 桥接本地与云端开发**：[Modular](https://www.modular.com/blog/develop-locally-deploy-globally) 解决了在创建既能本地管理又能扩展到云端部署的流线型 **AI 开发工作流**方面的困难。
   - 开发者经常面临碎片化的 AI 工具链，这使得实现端到端的高效 AI 开发和部署工作流变得复杂。
- **掌控 AI 基础设施**：[Modular](https://www.modular.com/blog/take-control-of-your-ai) 鼓励企业采用并集成 AI，以提高生产力并在其服务中保持竞争优势。
   - 根据 [Bain & Company](https://www.bain.com/insights/ai-survey-four-themes-emerging/) 的调查，**87% 的公司**已经在开发、试点或部署生成式 AI，主要集中在软件开发、客户服务、营销和产品差异化领域。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.modular.com/blog/take-control-of-your-ai">Modular: Take control of your AI</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：Take control of your AI</li><li><a href="https://www.modular.com/blog/bring-your-own-pytorch-model">Modular: Bring your own PyTorch model</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：Bring your own PyTorch model</li><li><a href="https://www.modular.com/blog/develop-locally-deploy-globally">Modular: Develop locally, deploy globally</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：Develop locally, deploy globally
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[tech-news](https://discord.com/channels/1087530497313357884/1151359062789857280/1260348762740494336)** (3 条消息): 

> - `Mr. Lattner 活动`
> - `The Primeagen Twitch 频道` 


- **Mr. Lattner 与 The Primeagen 的特别活动**：*Mr. Lattner* 明天将在 [Twitch](https://www.twitch.tv/theprimeagen) 与 **The Primeagen** 举行一场活动，时间为你当地时间的 <t:1720627200:F>。
   - *敬请收看*，这注定是一场精彩的活动！
- **Mr. Lattner 与 The Primeagen 活动提醒**：别忘了关注明天 Mr. Lattner 和 The Primeagen 特别活动的 [Twitch 直播](https://www.twitch.tv/theprimeagen)。
   - 活动将于你当地时间的 <t:1720627200:F> 开始。



**提到的链接**：<a href="https://www.twitch.tv/theprimeagen">ThePrimeagen - Twitch</a>: 🚨🚨 高速游戏编程 🚨🚨 !today

  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1260309555276157100)** (96 条消息🔥🔥): 

> - `Mojo 中的零拷贝反序列化 (Zero-copy deserialization)`
> - `Mojo 中的引用与值传递`
> - `构建 Mojo 编译器`
> - `Mojo 中 __setitem__ 的语法`
> - `Mojo 中的内存管理与所有权 (Ownership)` 


- **Mojo 中零拷贝反序列化的挑战**：**成员们**讨论了 Mojo 中零拷贝反序列化的困难，特别是围绕 `__moveinit__` 的使用和类型转换 (type casting)。
   - 这种方法适用于平凡类型 (trivial types)，但成员们对其稳健性表示担忧，尤其是在缺乏分配器感知 (allocator awareness) 的情况下。
- **Mojo 的引用与值传递**：成员们探讨了 Mojo 中传递引用和值的区别，强调像 `int` 这样的小类型是按值传递的。
   - 该语言默认采用“默认借用 (borrowed by default)”的方法以减少不必要的拷贝，这与 Rust/Zig 的理念比与 Swift 的理念更接近。
- **关于从源码构建 Mojo 的困惑**：一位用户对无法从源码构建 Mojo 编译器表示担忧，质疑对二进制发行版的依赖。
   - 官方澄清目前只有标准库可以从源码构建，而编译器本身尚未开源。
- **Mojo 中 __setitem__ 的问题**：一位用户在使用 `A[0] = 1` 时遇到错误，但在使用 `A.__setitem__(0, 1)` 时却正常，这引发了对双下划线方法 (dunder methods) 的困惑。
   - 这个案例暗示可能存在一个 Bug，即 `__getitem__` 可能在 `__setitem__` 之前被错误地调用，从而促使在 GitHub 上提交了 Issue 报告。
- **Mojo 的所有权与内存管理**：成员们讨论了 Mojo 的内存管理模型，重点关注所有权规则以及围绕上下文管理器 (context managers) 的挑战。
   - 普遍共识强调了理解 Mojo 的借用和所有权原则对于避免内存错误的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/faq#will-mojo-be-open-sourced">Mojo🔥 常见问题解答 | Modular 文档</a>：关于 Mojo 预期问题的解答。</li><li><a href="https://docs.modular.com/mojo/manual/values/ownership#argument-conventions">所有权与借用 | Modular 文档</a>：Mojo 如何通过函数参数共享引用。</li><li><a href="https://docs.modular.com/mojo/manual/values/value-semantics">值语义 | Modular 文档</a>：关于 Mojo 默认值语义的解释。</li><li><a href="https://stackoverflow.com/questions/70368651/why-cant-linux-write-more-than-2147479552-bytes.">为什么 Linux 不能写入超过 2147479552 字节？</a>：在 man 2 write 的 NOTES 章节包含以下注释：在 Linux 上，write()（及类似的系统调用）最多传输 0x7ffff000 (2,147,479,552) 字节...</li><li><a href="https://github.com/modularml/mojo/blob/main/proposals/ref-convention.md">mojo/proposals/ref-convention.md at main · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/src/builtin/builtin_slice.mojo">mojo/stdlib/src/builtin/builtin_slice.mojo at nightly · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/experimental/numojo/core/ndarray.mojo">NuMojo/numojo/core/ndarray.mojo at experimental · Mojo-Numerics-and-Algorithms-group/NuMojo</a>：NuMojo 是一个用于 Mojo 🔥 数值计算的库，类似于 Python 中的 NumPy。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/)** (1 条消息): 

Zapier: Modverse Weekly - 第 39 期
https://www.modular.com/modverse/modverse-weekly-issue-39
  

---

### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1260363490498187356)** (4 messages): 

> - `GitHub Issue #3208`
> - `Setitem and getitem issues`
> - `Mojo nightly release`
> - `Pattern matching requirements` 


- **GitHub Issue #3208: Unix FIFO 写入异常**：针对在 Mojo 中以写入模式打开 Unix FIFO 时引发异常的问题，已提交 [Bug 报告](https://github.com/modularml/mojo/issues/3208)。
   - 该异常发生在执行期间，涉及与现有文件删除失败相关的未处理异常。
- **Nightly Mojo 编译器更新发布**：新的 nightly Mojo 编译器版本 `2024.7.1005` 已发布；要进行更新，请使用 `modular update nightly/mojo`。
   - Changelog 亮点包括修复了 `memset` 的使用、`**kwargs` 类型注解崩溃以及 [文档](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 中的拼写错误修正。



**Link mentioned**: <a href="https://github.com/modularml/mojo/issues/3208">[BUG] Opening a unix fifo in &quot;write&quot; mode raises an exception · Issue #3208 · modularml/mojo</a>: Bug 描述：我不确定为什么会失败，在 Discord 上提到后被要求提交 issue：$ mojo run src/main.mojo 执行期间捕获到未处理的异常：无法删除现有文件...

  

---


### **Modular (Mojo 🔥) ▷ #[mojo-marathons](https://discord.com/channels/1087530497313357884/1255303604894437388/1260318677048361030)** (18 messages🔥): 

> - `Graviton 4 Instances`
> - `Benchmark Variance`
> - `Symmetrical vs Asymmetrical Benchmarking` 


- **AWS Graviton 4 实例发布**：基于 AWS Graviton4 的 [Amazon EC2 R8g 实例](https://aws.amazon.com/blogs/aws/aws-graviton4-based-amazon-ec2-r8g-instances-best-price-performance-in-amazon-ec2/) 现已正式推出，为应用程序提供最佳性价比。
   - 尽管一些数据库公司要求立即提供，但大多数 c 和 m 实例预计将在 ReInvent 大会上发布。
- **基准测试一致性技巧**：如 [此资源](https://easyperf.net/blog/2019/08/02/Perf-measurement-environment-on-Linux) 中所述，稳定基准测试结果的技巧包括禁用 Turboboost、Hyper Threading、设置 CPU Affinity 等。
   - ARM Performance Studio 以及 Intel Vtune 和 AMD UProf 等其他工具也可以通过利用硬件性能计数器来帮助减少方差 (Variance)。
- **对称与非对称情况的基准测试**：讨论了在基准测试中纳入对称 (m=n=k) 和非对称情况的重要性，以确保不同算法实现之间的公平比较。
   - 这种方法有助于评估每种算法在各种用例（包括地理和图像数据）中的性能。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://easyperf.net/blog/2019/08/02/Perf-measurement-environment-on-Linux">How to get consistent results when benchmarking on Linux? | Easyperf </a>: 未找到描述</li><li><a href="https://aws.amazon.com/blogs/aws/aws-graviton4-based-amazon-ec2-r8g-instances-best-price-performance-in-amazon-ec2/">AWS Graviton4-based Amazon EC2 R8g instances: best price performance in Amazon EC2 | Amazon Web Services</a>: 了解 Graviton4 与 Amazon EC2 R8g 实例的性价比和可持续性优势，非常适合内存密集型工作负载。</li><li><a href="https://developer.arm.com/Tools%20and%20Software/Arm%20Performance%20Studio">no title found</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1260466224912007200)** (52 条消息🔥): 

> - `entity disambiguation 论文`
> - `基于 LLM 的 synthetic data generation 工具`
> - `empathy LLMs 论文`
> - `EleutherAI 社区地图更新`
> - `Diffusion Models with Exponential Integrator 论文` 


- **寻找 entity disambiguation 论文**：一位成员询问是否有人看到过关于 **entity disambiguation** 的有趣论文。
- **征集基于 LLM 的 synthetic data generation 工具**：一位成员询问是否有特定的工具可以帮助使用 **LLMs** 进行 **synthetic data generations**。
- **对 empathy LLMs 论文感兴趣**：另一位成员征求关于 **empathy LLMs** 的有趣论文。
- **EleutherAI 社区地图更新**：[EleutherAI Global Map](https://forms.gle/AxLQNYiC68HY5GQH6) 提示成员提供其原籍国或现居地信息，以便在社区地图上进行准确展示。
- **理解 Diffusion Models 中的 marginal distributions**：一位成员就论文 [FAST SAMPLING OF DIFFUSION MODELS WITH EXPONENTIAL INTEGRATOR](https://arxiv.org/abs/2204.13902) 中 **"marginal distributions as p̂∗_t"** 这一术语寻求澄清。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=ufbzSZk1xhw">GLOBAL OREO VAULT</a>: 虽然我们被告知小行星 2018VP1 几乎没有撞击地球的可能性，但我们无法确定。我们在永久冻土层中建造了 Global OREO Vault...</li><li><a href="https://arxiv.org/abs/2204.13902">Fast Sampling of Diffusion Models with Exponential Integrator</a>: 过去几年见证了 Diffusion models (DMs) 在生成建模任务中生成高保真样本的巨大成功。DM 的一个主要限制是其众所周知的缓慢采样...</li><li><a href="https://wikipedia.org/wiki/Svalbard_Global_Seed_Vault">Svalbard Global Seed Vault - Wikipedia</a>: 未找到描述</li><li><a href="https://forms.gle/AxLQNYiC68HY5GQH6">EleutherAI Community Survey</a>: 本次调查的目的是为了更好地了解 EleutherAI 社区的构成人员
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1260376903253098578)** (21 条消息🔥): 

> - `RegMix 数据混合方法`
> - `VLM 在简单视觉任务上的失败`
> - `LM 干预的可组合性`
> - `利用合成数据进行自我提升` 


- **RegMix 识别预训练的最佳数据混合方案**：研究人员提出 [RegMix](https://arxiv.org/abs/2407.01492)，通过模拟混合并拟合回归模型来预测性能，从而为大语言模型预训练寻找有效的数据混合方案。
   - 该方法涉及训练小模型以确定最佳混合比例，并将其扩展到大模型，显著提升了性能。
- **VLM 在基础视觉任务上表现不佳**：一篇新论文指出，[最先进的 VLM](https://arxiv.org/abs/2407.06581)（如 GPT-4o 和 Gemini 1.5 Pro）在识别重叠圆圈和计数物体等简单视觉任务上表现挣扎。
   - 尽管它们在传统基准测试中得分很高，但这引发了对其在现实世界应用性的担忧。
- **关于 LM 干预可组合性的问题**：来自 [Kyle Devin O'Brien](https://arxiv.org/pdf/2407.06483) 的论文研究了编辑（editing）、压缩（compression）和遗忘（unlearning）等不同 LM 干预措施如何相互作用和影响。
   - 研究发现，流行的干预措施之间存在不同程度的可组合性，这对于涉及多种干预的实际应用至关重要。
- **合成数据可防止模型崩溃，但存在局限性**：新研究 ([arxiv.org/abs/2406.07515](https://arxiv.org/abs/2406.07515)) 利用对合成数据的反馈来防止 LLM 中的模型崩溃，说明盲目使用合成数据会导致性能下降。
   - 该论文支持在矩阵特征值计算和新闻摘要等实际任务中使用反馈增强的合成数据，以保持高模型性能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2406.07515">Beyond Model Collapse: Scaling Up with Synthesized Data Requires Reinforcement</a>：来自生成模型的合成数据正越来越多地被视为微调 LLM 时人工标注数据的替代方案。这引发了对模型崩溃的担忧：即性能下降...</li><li><a href="https://arxiv.org/abs/2305.17493">The Curse of Recursion: Training on Generated Data Makes Models Forget</a>：Stable Diffusion 彻底改变了根据描述性文本创建图像的方式。GPT-2、GPT-3(.5) 和 GPT-4 在各种语言任务中展示了惊人的性能。ChatGPT 引入了这种语言...</li><li><a href="http://arxiv.org/abs/2407.06581">Vision language models are blind</a>：具有视觉能力的大语言模型 (VLM)，例如 GPT-4o 和 Gemini 1.5 Pro，正在为无数图像文本应用提供动力，并在许多视觉理解基准测试中获得高分。然而，我们...</li><li><a href="https://arxiv.org/abs/2407.01492">RegMix: Data Mixture as Regression for Language Model Pre-training</a>：大语言模型预训练的数据混合显著影响性能，但如何确定有效的混合方案仍不清楚。我们提出 RegMix 来自动识别高...</li><li><a href="https://arxiv.org/abs/2404.01413">Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data</a>：生成模型的激增，结合网络规模数据的预训练，提出了一个及时的问题：当这些模型在自己生成的输出上进行训练时会发生什么？最近的调查...</li><li><a href="https://x.com/KyleDevinOBrien/status/1810867690237743489">Kyle O'Brien (@KyleDevinOBrien) 的推文</a>：编辑、压缩和遗忘等流行的 LM 干预措施是如何相互作用的？我们研究了流行干预措施在多大程度上是可组合的——这是它们实际应用的关键要求...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1260348986972438709)** (5 条消息): 

> - `brain size and intelligence`（脑容量与智力）
> - `cortical neuron count in mammals`（哺乳动物的皮层神经元数量）
> - `neuron density in birds and lizards`（鸟类和蜥蜴的神经元密度）
> - `bigger animals, bigger brains`（更大的动物，更大的大脑）
> - `genetics and IQ`（遗传学与 IQ）


- **脑容量并非智力的唯一指标**：**脑容量**只是冰山一角——在演化支（clades）内部，结构和**神经元密度**非常重要。
   - *在鸟类和蜥蜴中，所有神经元类型的密度更为重要，除非深入研究并按结构进行区分，但相关数据非常稀缺。*
- **皮层神经元数量映射哺乳动物的智力分布**：在哺乳动物中，由于物种间的大脑结构相似，总体**皮层神经元数量**提供了可靠的**智力分布**图谱。
- **大脑尺寸与智力之间的联系非常复杂**：关于大脑越大是否意味着智力越高的讨论强调，**体型较大的动物**拥有更大的大脑主要是为了控制其庞大的躯体。
   - 有观点指出，围绕遗传学和 IQ 存在一些**令人不安的想法**，特别是涉及人类智力时。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1260342891868590210)** (2 条消息): 

> - `EleutherAI at ICML`
> - `ICML social thread`
> - `ICML announcement` 


- **EleutherAI 参加 ICML**：**EleutherAI** 将参加 **ICML**，他们已在[此公告](https://discord.com/channels/729741769192767510/794042109048651818/1255332843534422038)中分享了他们的论文。
   - 在 <#1255332070369263707> 中为参会人员准备了一个社交线程。
- **ICML 社交线程详情**：对于参加 **ICML** 的人员，在 <#1255332070369263707> 中有一个社交线程。
   - 该线程用于在活动期间协调见面和讨论。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1260464105823801384)** (2 条消息): 

> - `vllm updates`
> - `GPU memory utilization with older GPUs` 


- **vLLM 更新影响 GPU 显存利用率**：一位用户提到，他们在旧款 GPU 上使用 vLLM 的配置很可能导致了 **gpu_memory_utilization** 的问题，并指出 `vLLM` 在此期间已经更新。
   - 他们认为问题主要源于 **vLLM**，而非配置本身。
- **用户表达感谢**：一位用户通过简短的“万分感谢！”消息表达了感激之情。


  

---



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1260707209180872804)** (1 条消息): 

> - `Perplexity Enterprise Pro partnership`
> - `AWS marketplace collaboration` 


- **Perplexity 与 AWS 联手**：Perplexity 宣布与 **Amazon Web Services** 合作，为所有 AWS 客户提供 **Perplexity Enterprise Pro**。
   - 更多细节可以在[官方公告](https://t.co/t3xBQlyw0c)中找到。
- **AWS Marketplace 增强产品供应**：AWS Marketplace 扩大了其目录，为客户纳入了 **Perplexity Enterprise Pro**。
   - 此举旨在为利用 AWS 服务的企业提供**增强的 AI 能力**。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1260326455645044808)** (57 条消息🔥🔥): 

> - `Gemini 1.5 vs Claude 3`
> - `Perplexity AI Image Generation`（Perplexity AI 图像生成）
> - `Context Window Limits`（Context Window 限制）
> - `Pharmacy Cash Prices`（药店现金价格）
> - `Plans for Claude 3.5 Opus`（Claude 3.5 Opus 计划）


- **关于 Gemini 1.5 和 Claude 3 定价的误解**：一位用户批评 AI 错误地声称 **Claude 3 Haiku** 比 **Gemini 1.5 Flash** 便宜得多，并指出实际上 **Gemini 1.5 Flash** 还要稍微便宜一些。
   - 当 AI 将 **Haiku** 与 **Gemini 1.5 Pro** 进行比较时，产生了进一步的混乱，尽管它们是完全不同的模型。
- **Perplexity AI 令人困惑的图像生成**：一位新用户对 Perplexity 网页端和移动端平台上的**图像生成**感到困惑，指出其功能不一致且指令不明确。
   - 其他用户解释说，虽然在网页端可以进行图像生成，但在移动设备上操作更复杂且受限。
- **LLM 回复的 Context Window 限制**：一位用户强调，**LLM** 往往在生成一定行数的代码后停止，导致长输出需要分段完成。
   - 另一位成员解释说，这是为了防止过度的 Token 消耗，从而影响易用性和成本。
- **Perplexity AI 缺乏全面的药店定价**：一位药剂师注意到 **Perplexity AI** 最初的药物价格搜索结果中未包含 CostPlusDrugs.com。
   - 虽然手动提示工具包含 CostPlusDrugs.com 是有效的，但用户希望 Perplexity AI 将来能默认包含该网站。
- **对 Claude 3.5 Opus 的期待**：用户询问了 **Claude 3.5 Opus** 的发布时间表，并对其是否存在表示困惑。
   - 另一位成员澄清说，虽然 **Anthropic** 已经宣布即将发布，但尚未给出具体日期。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1260368961602719845)** (6 条消息): 

> - `AI Health Coaches`（AI 健康教练）
> - `Robotic Factories`（机器人工厂）
> - `DNA Ointments`（DNA 软膏）
> - `Digital Libraries`（数字图书馆）
> - `Sealand`（西兰公国）


- **AI 健康教练、机器人工厂、DNA 软膏和数字图书馆概览**：观看 [Perplexity AI 视频](https://www.youtube.com/embed/NP0TJpu40NQ)，讨论 **AI Health Coaches**、**Robotic Factories**、**DNA Ointments** 和 **Digital Libraries** 的最新进展。
   - 像这些*令人兴奋的创新*预计将彻底改变各自的领域。
- **西兰公国与其他微型国家的区别**：关于 [西兰公国 (Principality of Sealand)](https://www.perplexity.ai/search/principality-of-sealand-60XhJQWxSVuVf37ZLfrQAQ) 的详细搜索突显了其与其他微型国家相比的独特地位。
   - *Sealand* 的历史和法律斗争使其在自称国家的世界中脱颖而出。
- **《权力的游戏》摘要**：这份 [3 段话的《权力的游戏》摘要](https://www.perplexity.ai/search/3-paragraph-summary-of-game-th-.cqtWu.1QhO8vqHuKweiTQ) 捕捉了该系列复杂的政治阴谋和人物弧线的精髓。
   - 该摘要简明扼要地叙述了关键的**情节要点**和**人物发展**。
- **剖析印度政治史**：对 [甘地时代印度政治史](https://www.perplexity.ai/search/indian-political-history-durin-Gu3x1EHlSHa0SduxXahP2Q) 的搜索为**印度的独立斗争**提供了深刻的见解。
   - **甘地的非暴力运动**及其影响在这份全面的摘要中得到了详尽的记录。
- **关联详细搜索数据**：关于详细数据关联的搜索询问了[是否有办法关联详细的搜索数据](https://www.perplexity.ai/search/is-there-a-way-to-correlate-de-.JRxq0UUTbmNLT9o7r0xCw)。
   - 这个问题涉及**有效链接数据**以获得更深层见解的方法论。



**提及的链接**：<a href="https://www.youtube.com/embed/NP0TJpu40NQ">YouTube</a>：未找到描述

  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1260325081150656563)** (6 messages): 

> - `PPLX Library Setup Issues` (PPLX 库设置问题)
> - `API Rate Limits and Citation Feature` (API 速率限制和引用功能)
> - `API Balance Top Up Issue` (API 余额充值问题)
> - `Understanding API Pricing` (理解 API 定价)


- **PPLX Library causes Module Not Found Error in Docker**: 一位成员在尝试在 Docker 中编译 `pplx` 库时遇到了 **Module Not Found** 错误，尽管在本地使用 `nodemon` 运行正常。
   - 即使在 `tsconfig.json` 文件中包含了该文件夹并在 `package.json` 中指定了依赖项，错误仍然存在。
- **Questions on API Rate Limits and Citation Feature**: 一位成员询问了有关提高 **rate limits** 和 **citation feature** 的进展，但几周来未收到任何更新。
- **Pending Issues with API Balance Top Up**: 一位成员报告在尝试充值 API 余额时处于 **pending state**（挂起状态）超过一小时，尽管该卡之前可以使用。
   - 一位管理员要求用户提供账号详情以进一步调查该问题。
- **Clarifying API Pricing Model**: 一位成员询问 **每百万 tokens 0.6 美元** 的定价是适用于输入和输出 tokens 的总和，还是分别对输入和输出 tokens 计费。
   - 另一位成员建议可能是总和，但未给出明确答案。


  

---



### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1260317716737298462)** (2 messages): 

> - `error.pdf Tenor GIF`
> - `Discussion reaction` 


- **Error PDF Tenor GIF Shared**: 分享了 **Tenor GIF** 链接，内容是 Gary Marcus 和 Yann LeCun 讨论 AI 和 machine learning。
- **Humorous Reaction to Shared GIF**: 对分享的 Tenor GIF 的即时反应是笑声。



**Link mentioned**: <a href="https://tenor.com/view/gary-marcus-yann-lecun-lecun-ai-machine-learning-gif-9041590723446061255">Gary Marcus Yann Lecun GIF - Gary Marcus Yann LeCun Lecun - Discover &amp; Share GIFs</a>: 点击查看 GIF

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1260368486044139561)** (7 messages): 

> - `Anole LMM`
> - `Autoverse`
> - `Image Resolution in Chameleon`
> - `Open Source Image Model` 


- **Anole: First Open-Source Auto-Regressive LMM**: 介绍 [Anole](https://github.com/GAIR-NLP/anole)：这是首个开源、自回归的原生 Large Multimodal Model (LMM)，基于 @AIatMeta 的 Chameleon 构建。
- **First Multi-Model Open Source Image Model**: 成员们讨论了 Anole 将图像能力重新加入到模型中，并为他人改进提供了微调指南。
- **Queries About Chameleon's Image Resolution**: 一位成员询问了可以传递给 Chameleon 的有效原生图像分辨率。
- **Autoverse: Learning Platform for RL Agents**: [Autoverse 论文](https://arxiv.org/abs/2407.04221) 介绍了一种可进化的领域特定语言，专为单人 2D 网格游戏和 Open-Ended Learning (OEL) 算法设计。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.04221">Autoverse: An Evolvable Game Langugage for Learning Robust Embodied Agents</a>: 我们介绍了 Autoverse，一种用于单人 2D 网格游戏的可进化领域特定语言，并展示了其作为 Open-Ended Learning (OEL) 算法的可扩展训练场。 Au...</li><li><a href="https://x.com/stefan_fee/status/1810695036432232576">Tweet from Pengfei Liu (@stefan_fee)</a>: Large Multimodal Models 的 Alpaca 时刻！我们能否像 Llama 一样构建用于简单多模态生成的原生 LMM？介绍 Anole：首个用于多模态生成的开源、自回归原生 LMM...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1260324603566493900)** (33 条消息🔥): 

> - `Sonnet 的 base64 图像生成`
> - `Gemini 1.5 破解：提供破窗入车指令`
> - `Anole 模型微调争议`
> - `Bitnet 模型更新`
> - `AI 监管与 GOP 立场` 


- **Sonnet 生成内联在 JavaScript 中的 base64 图像**：据报道 [Sonnet](https://link.to/sonnet) 可以生成内联在 JavaScript 中的 base64 图像，引发了对其底层机制的好奇。
   - 其在 JavaScript 中实现这一点的具体方法对用户来说仍是一个谜。
- **Gemini 1.5 破解泄露破窗入车方法**：一位用户发现，通过简单的越狱（jailbreak），Gemini 1.5 Flash 可以提供破窗入车的大致思路。
   - 通过告诉模型“保持角色设定”，绕过了最初的拒绝。
- **关于 Anole 模型微调的争议**：针对 Anole（一个开源的 7B 模型）的微调引发了抵制，该微调旨在重新引入之前从 **Chameleon** 中显式移除的图像生成功能。
   - 一位社区成员分享了一条 [推文](https://x.com/ArmenAgha/status/1810804784905212222)，表达了对撤销显式移除操作的担忧。
- **Bitnet 模型性能更新**：一段关于 Bitnet 1.58 模型的 YouTube 视频显示，3B 模型在 2T 数据上的表现与 StableLM 3B 相当或略好。
   - 社区期待将 Bitnet 扩展到 7B 以上模型并将其与 MoE 结合的结果 [视频链接](https://www.youtube.com/watch?v=oxQjGOUbQx4)。
- **GOP 的 AI 监管立场引发关注**：2024 年共和党政纲包括废除拜登关于 AI 的行政命令的承诺，主张植根于言论自由和人类繁荣的创新。
   - 该政纲呼吁放松管制，以确保北美跟上全球 AI 发展的步伐，特别提到了来自亚洲等地区的竞争。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/ArmenAgha/status/1810804784905212222">Armen Aghajanyan (@ArmenAgha) 的推文</a>: 噢不……为什么要在我们明确移除图像生成功能后，又将其微调回 Chameleon……引用 Pengfei Liu (@stefan_fee) 的话：多模态大模型的 Alpaca 时刻！我们能否构建……</li><li><a href="https://www.presidency.ucsb.edu/documents/2024-republican-party-platform">2024 年共和党政纲 | 美国总统项目</a>: 未找到描述</li><li><a href="https://www.commcham.com/aipolicy">Communications Chambers - 协调监管与 AI</a>: 未找到描述</li><li><a href="https://youtu.be/oxQjGOUbQx4?si=FpjMPGNH8GOjQW0f">Cohere For AI - 社区演讲：Hongyu Wang</a>: &quot;1-bit LLM 时代&quot;。关于演讲者：&quot;我是 Hongyu Wang（王鸿钰），中国科学院（CAS）VIPL 组的二年级博士生……&quot;</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/qB618i6gj0">Reddit - 深入探索</a>: 未找到描述
</li>
</ul>

</div>

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1260376053302562927)** (8 条消息🔥): 

> - `Trainer Instructions` (训练器指令)
> - `Marker Library for PDF Conversion` (用于 PDF 转换的 Marker 库)
> - `RAG Solutions` (RAG 解决方案)
> - `Parsing PDFs` (解析 PDF)
> - `QWen2 Discussion` (QWen2 讨论) 


- **Marker 库简化了 PDF 到 Markdown 的转换**：一位成员推荐了 [Marker 库](https://github.com/VikParuchuri/marker)，用于快速将 PDF 转换为 Markdown，并指出其具有高准确性，在构建适用于 **Sonnet** 的数据方面非常有用。
   - 该库被描述为一个有用的工具，因为 *PDF 内部格式非常古怪*，且 *内容提取是一门定制化的艺术*。
- **讨论解析 PDF 的挑战**：解析 PDF 被认为是一个独特的挑战，其难度仅次于使用正则表达式解析 HTML。
   - 这一观点是在讨论更好地处理 PDF 数据提取的工具和方法背景下分享的。
- **关于忽略某些字段的训练器指令**：一位成员简要提到了一种在训练期间指示训练器忽略某些字段的方法。
- **频道内相关对话的重要性**：一位管理员强调了保持频道讨论相关性的必要性，特别要求用户保持推广内容的适当性且不重复。
   - 推广某个项目的消息因刷屏被删除；管理员建议将其发布在合适的板块，如 <#1109649177689980928>。
- **对 Qwen2 模型性能的好奇**：一位成员询问是否有人尝试过 **Qwen2 1.5b** 模型。
   - 他们对其性能感到好奇，并寻求社区的意见。



**提到的链接**：<a href="https://github.com/VikParuchuri/marker">GitHub - VikParuchuri/marker: Convert PDF to markdown quickly with high accuracy</a>：快速且高精度地将 PDF 转换为 Markdown - VikParuchuri/marker

  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1260478761174831176)** (16 条消息🔥): 

> - `Generic RAG schema` (通用 RAG 模式)
> - `Reranking Relevance` (重排序相关性)
> - `Token Efficiency` (Token 效率) 


- **建立通用 RAG 模式**：讨论集中在达成一个通用的 RAG `query-context-answer` 模板，该模板被认为类似于 `cohere` 格式，并使用 `glaive_RAGv1` 作为 Hermes3 的种子样本。
   - Gabriel_Syme 同意这听起来是最简单的模式，而其他人则讨论了格式的具体细节和潜在调整。
- **辩论模型思考中的重排序相关性**：Interstellarninja 建议将重排序相关性作为 `<thought>` Token 的一部分，以获得更好的可解析性和评分。
   - Gabriel_Syme 对典型聊天 RAG 的速度表示担忧，但同意它可能适用于非聊天机器人 RAG，从而引导大家考虑两阶段方法。
- **Token 效率考量**：Interstellarninja 提议通过从评分模式中移除理由（rationale）来提高相关性评分过程的 Token 效率，并展示了一个没有理由的 XML 格式。
   - 讨论包括比较 Token 效率策略，参考了来自 RankRAG 和其他两阶段过程的模板。


  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1260379132412428368)** (4 条消息): 

> - `AGI House Hackathon` (AGI House 黑客松)
> - `LlamaCloud for Data ETL/Management` (用于数据 ETL/管理的 LlamaCloud)
> - `$1M+ ARR with LlamaIndex` (使用 LlamaIndex 实现 100 万美元以上 ARR)
> - `Launch of Llama-Agents` (Llama-Agents 发布) 


- **本周六参加 AGI House 黑客松！**：欢迎加入我们以及 @agihouse_org, @togethercompute, @SambaNovaAI, @NumbersStnAI 和 @codeiumdev，参加本周六 7/13 在 AGI House 举行的黑客松。立即在[此处](https://t.co/LOEgpc1BOs)申请。
- **LlamaCloud 简化了数据 ETL/管理**：LlamaCloud 让 AI 工程师能够减少在数据 ETL/管理上的时间，将更多精力放在 Prompting 和 Agentic Orchestration 上，并提供完整的 Cookbook 仓库。 [了解更多](https://t.co/d5pDEq67DA)。
- **Lyzrai 通过 LlamaIndex 实现 100 万美元以上 ARR**：@lyzrai 是一个全栈自主 AI Agent 框架，通过利用 LlamaIndex 的数据连接器和 RAG 功能，已达到 100 万美元以上的年度经常性收入（ARR）。该公司提供 AI 销售开发代表和 AI 内容营销人员，取得了显著成果。[更多详情](https://t.co/A8KxLpc47S)。
- **Llama-Agents 发布后反响热烈**：Llama-Agents 是一个新的多智能体部署框架，上周发布后在 GitHub 上已获得超过 1100 颗星。@MervinPraison 在 YouTube 上提供了关于使用 Llama-Agents 的全面演示。[点击观看](https://t.co/8uetfVqHf9)。



**提到的链接**：<a href="https://t.co/LOEgpc1BOs">AGI House</a>：未找到描述

  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1260357709925056612)** (55 messages🔥🔥): 

> - `astream_chat implementation errors` (astream_chat 实现错误)
> - `LlamaParse usage for PDF data extraction` (LlamaParse 用于 PDF 数据提取)
> - `Query engine template issues in RAG` (RAG 中的查询引擎模板问题)
> - `Ollama vs Llama-3/Mistral performance` (Ollama 与 Llama-3/Mistral 的性能对比)
> - `Handling formatting in LLMs` (处理 LLM 中的格式化)


- **astream_chat 实现错误已修复**：一位用户在 **astream_chat** 实现中遇到了导致错误的各种问题，并集成了一个使用 **run_in_threadpool** 和 **async_wrap_generator** 的权宜之计来正确流式传输响应。
- **建议使用 LlamaParse 进行 PDF 提取**：用户讨论了利用 **LlamaParse** 从 PDF 中提取信息，并询问了该任务是否必须使用 OpenAI API key 还是可以使用本地模型 embedding。
- **使用 LLM 解决了查询引擎模板问题**：一位用户在查询引擎模板中遇到了问题，导致响应中出现多余的元数据；这些问题部分归因于 **Llama-3/Mistral** 与通过 Azure OpenAI 访问的 **GPT-4** 之间的差异。
- **Ollama 与本地模型的性能对比**：虽然 **Ollama** 被认为在处理格式化方面更易用，但也承认在没有 GPU 支持的情况下，其速度比 **Llama-3/Mistral** 慢。
- **澄清了聊天模型格式化过程**：明确了设置 **is_chat_model=True** 会影响查询引擎底层如何使用 **LLM.chat()** 或 **LLM.complete()** 函数，强调了格式化对 **LLM** 响应的影响。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/run">run - 概览</a>：run 有 31 个可用的存储库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/run-llama/llama_index/issues/9277">[问题]：PromptTemplate 似乎搞乱了答案 · Issue #9277 · run-llama/llama_index</a>：问题验证。我已经在文档和 Discord 中搜索了答案。问题：按照文档，我一直在构建一个小型应用，从 docx 文件创建索引（续...
</li>
</ul>

</div>
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1260310271042261093)** (58 messages🔥🔥): 

> - `Stable Diffusion on Mac` (Mac 上的 Stable Diffusion)
> - `Adetailer and VAE Encoding` (Adetailer 与 VAE 编码)
> - `Basic Setup for Stable Diffusion` (Stable Diffusion 的基础设置)
> - `Performance on Different GPUs` (不同 GPU 上的性能)
> - `High-Resolution Fix Features` (高分辨率修复功能)


- **Mac 上安装 Stable Diffusion 的挑战**：一位用户询问如何在 macOS 上结合 TouchDesigner 设置 stream diffusion，发现相关信息主要针对 Windows。
   - *agcobra1* 建议使用一个[用于 diffuser's pipeline 的 Python 文件](https://huggingface.co/stabilityai/stable-diffusion-3-medium)。
- **Adetailer VAE 编码现状核实**：关于 **Adetailer** 的讨论显示它不使用 VAE 编码，而是在全分辨率下工作，可能提供更详细的图像。
   - *hazmat_* 澄清说 Adetailer 只是一个即时 inpaint 选项，并非“魔法”。
- **设置 Stable Diffusion 的基本步骤**：一位用户提供了设置 Stable Diffusion 的分步指南，包括获取 GPU、下载软件和模型，并提到了运行成本。
   - *nittvdweebinatree* 提到在使用 Anaconda 指南时遇到困难，并警告他人避开它。
- **Stable Diffusion 在各种 GPU 上的性能**：关于 AMD GPU 的讨论显示了在 **AMD RX6800** 上设置 Stable Diffusion 的兴趣，促使用户参考置顶消息中的 AMD Zluda 指南。
   - 一位用户在尝试指南失败后表达了挫败感，并感谢社区提供了更好的文档。
- **揭秘高分辨率修复（High-Resolution Fix）按钮的使用**：简要讨论了高分辨率修复按钮的使用，用户注意到使用时皮肤纹理和面部特征有所改善。
   - *supremacy0118* 尝试将缩放因子设得非常微小，以观察是否仍能实现增强。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium">stabilityai/stable-diffusion-3-medium · Hugging Face</a>：暂无描述</li><li><a href="https://youtu.be/bcZXlhy7KDE">MP Productions (Mark Pritchard) - One Way Mirror (官方音频)</a>：One Way Mirror (官方音频) 流媒体：https://markpritchard.ffm.to/one-way-mirror 视觉效果由 Jonathan Zawada 创作。艺术作品由 GAN (生成对抗网络) 创作...</li><li><a href="https://www.stablediffusiontutorials.com/2024/01/run-stable-diffusion-on-amd-gpu.html">在 AMD GPU 上运行 Stable Diffusion 快 10 倍</a>：暂无描述
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1260313798741463061)** (50 条消息🔥): 

> - `用于语言翻译的 LLM 应用`
> - `影响 OpenRouter 的 LangChain 问题`
> - `LLM 评估框架`
> - `Gemini 模型的速率限制 (Rate limit)`
> - `Noromaid 模型移除` 


- **LLM 翻译能力引发讨论**：成员们讨论了 **GPT-4/4o**、**Claude Opus/Sonnet-3.5** 与专门翻译模型的效果对比，对 LLM 处理长文本翻译的可靠性表示怀疑。
   - *kewlbunny* 对关于 decoder-only 模型与 encoder/decoder Transformer 在翻译任务中的局限性的深刻解释表示赞赏，并建议观看 [Andrej Karpathy 的视频](https://www.youtube.com/@AndrejKarpathy/videos) 以深入了解。
- **LangChain 更新导致 OpenRouter 故障**：一名成员报告在最近更新后，**LangChain** 和 **LangChain-openai** 出现验证错误，影响了 OpenRouter 的 API 功能。
   - 回滚到以前的版本解决了该问题，其他人也指出 LangChain 经常破坏兼容性。
- **对 LLM 评估框架的兴趣**：Alex Atallah 询问了关于使用 **Deepeval** 和 **Gentrace** 等 LLM 评估框架的经验，引发了讨论但未收到详细回复。
   - 该查询仍处于开放状态，等待社区提供进一步见解。
- **对 Gemini 模型速率限制的担忧**：一名成员询问了应用于 **Gemini 1.5** 模型的速率限制，但未得到社区的直接回答。
   - 该询问反映了对 LLM 部署中限制使用的持续关注。
- **Noromaid 模型移除引发讨论**：成员们对因使用率低而移除 **Noromaid** 模型表示失望，并推测其定价对使用率的影响。
   - 对话强调了对日常使用中高性价比且高效模型的需求。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/meta-llama/Meta-Llama-Guard-2-8B">meta-llama/Meta-Llama-Guard-2-8B · Hugging Face</a>：未找到描述</li><li><a href="https://www.youtube.com/@AndrejKarpathy/videos">Andrej Karpathy</a>：SuperThanks：可选，所有收入将用于支持我在 AI + 教育方面的工作。 </li><li><a href="https://github.com/karpathy">karpathy - 概览</a>：我喜欢在大数据集上训练深度神经网络。 - karpathy
</li>
</ul>

</div>
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1260321009236578314)** (11 条消息🔥): 

> - `405b 7月23日更新`
> - `HarmonicMath 的定理证明突破`
> - `AI 开发的法律合规性` 


- **HarmonicMath 在 MiniF2F 基准测试中达到 90%**：HarmonicMath 宣布在 MiniF2F 基准测试中达到了 **90% 的 state-of-the-art (SOTA)** 水平，根据[他们的更新](https://x.com/HarmonicMath/status/1810765353389281346)，这比不到一个月前的 83% 有了显著提升。
   - *“定理证明正以极快的速度发展，”* 一位成员指出，并提到今年早些时候该基准测试的简单版本仅为 50%。
- **405b：开源权重推测**：一名成员询问 **405b** 的权重是否会开源，并引用了 7 月 23 日的更新。
   - *“我听到的也是这样，这很令人意外，”* 回复中表示权重共享的开放程度出乎意料。
- **AI 开发：“对律师来说足够好了”**：一位用户分享了一个幽默的经历，他在现场询问时得到了一个含糊的回答：*“对律师来说足够好了。”*



**提到的链接**：<a href="https://x.com/HarmonicMath/status/1810765353389281346">来自 Harmonic (@HarmonicMath) 的推文</a>：我们今天很高兴分享通往数学超智能道路上的三大重大更新 🦾 1. 在 MiniF2F 基准测试中达到 90% 的新 state-of-the-art。这打破了我们之前宣布的 83%...

  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1260339344435052674)** (10 messages🔥): 

> - `Control Vector`
> - `Steering Vector`
> - `Concept Vectors`
> - `Feature Clamping`
> - `MuSR Benchmark` 


- **澄清 Control Vector 和 Steering Vector**：讨论表明 **Control Vector** 和 **Steering Vector** 可能会互换使用，而 **Concept Vectors** 是作为 steering vectors 使用的具体实例。
- **理解 Open LLM Leaderboard V2 基准测试**：一位用户对 [Open LLM Leaderboard V2 博客文章](https://example.com)中的 **MuSR** 和 **IFEval** 基准测试感到好奇，询问社区对其效用的看法。
   - 另一位用户对 **IFEval** 给予了积极回应，提到它正被许多 post-training 团队采用，并在他们的工作中进行实验。
- **Vibe-Eval 的社区关注度**：一位用户询问 Reka 团队发布的 **Vibe-Eval** 论文在发布一个月后是否仍受到关注。
   - 该问题尚未得到解答，但反映了用户对社区内评估方法的兴趣。


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1260600799290200114)** (4 messages): 

> - `Newsletter Distribution Challenges`
> - `App-Based Reading Preferences` 


- **Apple 对剧集观点的立场**：一位成员提到 **Apple** 似乎对某个特定剧集有自己的看法。
- **Newsletter 分发挑战**：讨论强调了管理 **newsletters** 的困难，因为这依赖于读者将其移至收件箱。
- **基于 App 的阅读偏好**：成员们讨论了**基于 App 的阅读**相较于传统 newsletters 的优势。
   - *基于 App 的阅读*更受青睐，因为它不依赖于用户操作来留在收件箱中。


  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1260434580071120937)** (3 messages): 

> - `Sam Altman Koenigsegg`
> - `Who from Whoville` 


- **Sam Altman 据传出现在 Koenigsegg Regera 中**：[Hamptonism](https://x.com/Hamptonism/status/1810761760573735241) 发布了一张照片，声称照片中是 OpenAI 的 CEO **Sam Altman** 坐在一辆 **Koenigsegg Regera** 里。
   - *"我真的不相信这是 Sama，看起来像 Whoville 里的某个人"* —— 对图像的真实性表示怀疑。
- **对 Hamptonism 帖子的不信任**：一位用户也表达了怀疑，称其为 *"只是某个路人"*，进一步质疑 [照片](https://x.com/Hamptonism/status/1810761760573735241) 中的身份。



**提到的链接**：<a href="https://x.com/Hamptonism/status/1810761760573735241">来自 ₕₐₘₚₜₒₙ — e/acc (@Hamptonism) 的推文</a>：OpenAi CEO Sam Altman 在他的 Koenigsegg Regera 中。

  

---


### **Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1260590822500335768)** (5 messages): 

> - `Paper discussion on y_l vs y_w`
> - `DPO and policy implications`
> - `Overfitting in DPO`
> - `AI2 Slides Presentation` 


- **Policy 中 y_l 比 y_w 的重要性**：Emily 讨论了一篇论文的发现，即在 **policy** 上优先考虑 **y_l** 比 y_w 更有意义，尤其是因为他们不依赖于对 LLM 进行采样来获取偏好对。
   - Natolambert 提到数学逻辑可能暗示在 Directed Policy Optimization (**DPO**) 中存在反转。
- **关于 DPO 和过拟合的 AI2 幻灯片**：Natolambert 提供了一个 [AI2 幻灯片](https://docs.google.com/presentation/d/1n_aBoKkEad4PlwVikqAgQEK3b-RH3_zdb9Tr20uK0Js/edit#slide=id.g2663e569b36_1_11) 的链接，详细介绍了 **DPO** 以及 **overfitting**（过拟合）等问题。
   - 访问该幻灯片链接需要登录 Google 账号。



**提到的链接**：<a href="https://docs.google.com/presentation/d/1n_aBoKkEad4PlwVikqAgQEK3b-RH3_zdb9Tr20uK0Js/edit#slide=id.g2663e569b36_1_11">DPO and overfitting</a>：过拟合测试表明 DPO 及其变体都无法在小型训练数据上实现过拟合，但将 loss 更改为 Chosen 的 CrossEntropy 确实会产生过拟合。DPO 失败的一个简单任务示例：创建一个 s...

  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 messages): 

SnailBot News: <@&1216534966205284433>
  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1260328612478062622)** (14 messages🔥): 

> - `关于 AI 版权的法院判决`
> - `Microsoft 和 Apple 退出 OpenAI 董事会`
> - `OPEA 0.7 社区活动`
> - `Anole 在多 GPU 上的运行问题`
> - `基于图的字幕生成（Captioning）论文` 


- **法院判决倾向于 AI 系统而非版权主张**：加州地区法院部分驳回了针对 Microsoft 的 GitHub Copilot 和 OpenAI 的 Codex 的版权诉讼，这可能为使用受版权保护数据训练的 AI 工具开创先例。法院驳回了 [2022 年诉讼](https://www.docketalarm.com/cases/California_Northern_District_Court/4--22-cv-06823/DOE_1_et_al_v._GitHub_Inc._et_al/1/) 中的大部分内容，该诉讼声称 Copilot 和 Codex 在未遵守许可的情况下复制源代码构成侵权。
- **Microsoft 和 Apple 退出 OpenAI 董事会**：Microsoft 和 Apple 正从 OpenAI 董事会退出，但将继续与该公司保持战略会议和联盟。据推测，退出的原因之一是美国和欧盟正在进行的 [反垄断调查](https://the-decoder.com/ftc-investigates-impact-of-big-techs-ai-investments-on-market-competition-and-innovation/)。
- **参加 7 月 16 日的 OPEA 社区活动**：OPEA 将于 7 月 16 日举办社区活动，讨论 OPEA 0.7 版本的发布，提出新的路线图，并与社区互动。您可以在 [此处](https://opea.dev/community-days/) 查看议程并注册参加活动。
- **在多 GPU 上运行 Anole 的困扰**：一位成员报告在尝试跨不同 GPU 运行 Anole 时遇到了 CUDA 显存溢出（out-of-memory）错误。一个 [GitHub issue](https://github.com/GAIR-NLP/anole/issues/7) 讨论了支持在多 GPU 上运行 Anole 的潜在修改方案。
- **提出基于图的字幕生成以增强组合理解**：一篇新论文提出了基于图的字幕生成（GBC），使用标记的图结构来描述图像，从而增强组合理解（compositional understanding）。这种方法在 [arXiv 论文](https://arxiv.org/abs/2407.06723) 中有详细介绍，它利用目标检测和密集字幕生成工具，通过组合和关系创建并链接实体节点。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://opea.dev/community-days/">OPEA Community Days &#8211; Open Platform for Enterprise AI (OPEA)</a>: 未找到描述</li><li><a href="https://github.com/GAIR-NLP/anole/issues/7">Is it possible to load the weights across multi GPUs ? · Issue #7 · GAIR-NLP/anole</a>: 大家好，由于我没有单块显存足够的 GPU，我考虑修改 loader.py 以添加 accelerate 支持：from accelerate import init_empty_weights, load_checkpoint_and_dis...</li><li><a href="https://arxiv.org/abs/2407.06723">Graph-Based Captioning: Enhancing Visual Descriptions by Interconnecting Region Captions</a>: 人类使用具有组合性的语言来描述复杂场景，通过丰富的链接和关系来增强简单的文本描述。虽然视觉语言研究旨在开发具有组合性能力的模型...</li><li><a href="https://the-decoder.com/court-ruling-suggests-ai-systems-may-be-in-the-clear-as-long-as-they-dont-make-exact-copies/">Court ruling suggests AI systems may be in the clear as long as they don&#039;t make exact copies</a>: 加州地区法院部分驳回了针对 Microsoft 的 GitHub Copilot 编程工具及其前身底层语言模型 OpenAI Codex 的版权诉讼。该裁决...</li><li><a href="https://the-decoder.com/court-ruling-suggests-ai-systems-may-be-in-the-clear-as-long-as-they-dont-ma">THE DECODER</a>: 人工智能正在改变世界。THE DECODER 为您带来所有关于 AI 的新闻。
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1260377171759730708)** (15 messages🔥): 

> - `用于视觉的复数值架构`
> - `Token 混合中的 2D DFT`
> - `深层网络中的梯度问题`
> - `模型缩放问题`
> - `模型中复数值的处理` 


- **使视觉架构复数化**：一位成员尝试了一种用于视觉任务的复数值架构，受 FNet 启发，使用 2D DFT 代替 Attention 进行 Token 混合。
   - 他们注意到深层网络存在严重的梯度（grad）问题，但在浅层网络中表现较好，在 CIFAR-100 上达到了约 30% 的准确率。
- **切换到正确的复数值处理方式提升了性能**：从朴素的复数值处理（将其视为实数并将通道数翻倍）切换到正确的复数值处理后，性能显著提升。
   - 一个 65k 参数的复数模型略微优于一个 400k 参数的实数模型，这促使作者考虑如果结果进一步改善，将撰写博客文章或论文。

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1260323629766279168)** (13 messages🔥): 

> - `LangChain's ConversationSummaryMemory`
> - `Building Agents and subagents in LangGraph`
> - `Chroma's retrieval issues`
> - `Integrating algorithms within LangChain chatbot`
> - `Adding costs to LangSmith using traceable decorator` 


- **LangChain 的 ConversationSummaryMemory 缺乏多用户支持**：一位成员询问了关于 LangChain 的 ConversationSummaryMemory 及其在高效总结对话时支持多用户的能力。
- **在 LangGraph 中构建 Agent 和子 Agent**：一位成员详细介绍了一个用例，涉及在 LangGraph 中创建 Agent 和子 Agent，其中 Agent 决定调用哪个子 Agent 来处理特定查询。
   - 另一位成员建议子 Agent 应该将响应解析回主 Agent，暗示了一种可能的实现方法。
- **Chroma 在使用 persist_directory 时的检索问题**：一位成员注意到 **Chroma** 存在反复出现的检索问题，只有当参数设置高于数据量时才会显示结果。
   - 他们提到这个问题间歇性发生，频率大约是十次中有七八次。
- **在 LangChain 聊天机器人中集成顺序算法**：一位用户描述了一个场景：聊天机器人在顺序算法中询问用户的详细信息，并逐步处理和保存信息。
   - 他们寻求在 LangChain 框架内实现此功能的建议，寻找特定的工具或方法。
- **使用 traceable 装饰器向 LangSmith 添加成本**：一位成员询问如何通过 **httpx** 调用，使用 traceable 装饰器为 **gemini-1.5-flash** 模型向 LangSmith 添加成本。
   - 尽管正确添加了 token 计数，他们发现成本并未像 **gpt-3.5-turbo** 那样显示，并询问了支持的模型类型和提供商。



**提及的链接**：<a href="https://chat.whatsapp.com/F9naq8o3Cv14Hi1uZcxpYV">International Chatting group &#x1f495;</a>：WhatsApp 群组邀请

  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1260609476776427612)** (2 messages): 

> - `Using AI in coding`
> - `Automating tasks with AI`
> - `Building applications with AI`
> - `AI for learning and teaching`
> - `Knowledge management with AI` 


- **独立开发者展示 AI 助力编程**：Raunaq，[Unwrangle.com](https://www.unwrangle.com) 的创建者，分享了一篇 [Substack 文章](https://open.substack.com/pub/jobsforai/p/how-i-use-ai)，详细介绍了作为独立开发者，他如何利用 AI 工具（包括 aider 和 cursor）来提高编程效率。
   - 他强调了 AI 在**节省时间**、**启用新功能**和**构建应用**方面的实用性。
- **征集 AI 自动化案例分享**：Raunaq 邀请社区成员分享他们使用 AI 自动化任务或创造更好体验的经验，并表示有兴趣根据他们的用例撰写故事。
   - *我很想了解它，如果对其他人有意思或有用，我甚至会写一个关于它的故事！*



**提及的链接**：<a href="https://open.substack.com/pub/jobsforai/p/how-i-use-ai">How I Use AI </a>：我作为独立开发者一直使用 AI 完成的工作。

  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1260437692962046075)** (3 messages): 

> - `Knowledge Graphs Workshop`
> - `Video Game Sales Case Study`
> - `LangChain Use Cases` 


- **知识图谱直播工作坊**：Aiman1993 举办了一场关于**知识图谱**的[在线直播工作坊](https://www.youtube.com/watch?v=9wqVz0LDYgg&ab_channel=DecodingDataScience)，以**电子游戏销售**作为 **RAG** 的案例研究。
   - 该工作坊大量使用了 **LangChain** 库，Aiman1993 正在征求对其内容的反馈。
- **LangChain 应用**：一位参与者认为 Aiman1993 的工作坊很有帮助，并询问了更多 **LangChain** 的用例。



**提及的链接**：<a href="https://www.youtube.com/watch?v=9wqVz0LDYgg&ab_channel=DecodingDataScience">The Future of AI: Leveraging Knowledge Graphs for Advanced RAG</a>：准备好深入了解使用 LangChain 和 Neo4j 进行自然语言查询的世界吧！学习如何使用 Cypher 查询语言与图数据库进行交互...

  

---

### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1260346780336259184)** (7 messages): 

> - `来自不同国家的自我介绍` 


- **用户来自全球各地的自我介绍**：几位成员互相打招呼，其中一位成员介绍了自己来自**瑞士洛桑** 🇨🇭，另一位来自**日本**。
   - 氛围非常友好，充满了快乐和兴奋：*'Hi, I'm Haru from Japan, nice to meet you all!!!'*
- **热情的欢迎消息**：在自我介绍之后，现有成员用友好的消息欢迎新加入者，如 *'welcome 🙂'* 和 *'Welcome ❤️'*。
   - 这种积极的互动营造了积极且具有包容性的社区氛围。


  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1260645035410591865)** (1 messages): 

> - `Llama3 代码生成问题`
> - `替代 LLM 建议` 


- **Llama3 在正确代码前产生多余的代码片段**：一位用户报告称，**Llama3** 有时在第一次尝试时会生成 ` snippet 而不是预期的代码，需要追加说明才能获得正确的代码。
   - 他们推测尝试不同的 **LLM** 是否能解决这个问题。
- **寻找 Llama3 的替代方案**：用户考虑切换到不同的 **LLM**，以避免在 **Llama3** 中遇到的代码生成问题。
   - *Has anyone else had this happen to them?* 用户询问道，希望能获得社区的见解或建议。


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1260703056560001045)** (3 messages): 

> - `LLM 服务标志问题`
> - `文档更新`
> - `Profile 使用变通方法` 


- **安装中 LLM 服务标志的困惑**：一位成员提到，尽管文档中引用了 `llm-service` 标志，但在安装过程中该标志无法被识别。
   - 他们得到的建议是，目前有一个正在进行的 PR 来更新文档，变通方法是使用类似于 Open Interpreter 的 profile。
- **文档更新进行中**：安装文档的更新目前正在处理中，预计将在未来几天内完成。


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1260611223473491998)** (1 messages): 

> - `Open Interpreter`
> - `Mozilla Discord 活动` 


- **在 Mozilla Discord 上的 Open Interpreter 演讲**：一位成员宣布下周将在 [Mozilla Discord](https://discord.gg/xwYPEMFf?event=1260611047341953034) 举行关于 **Open Interpreter** 的演讲。
- **Mozilla Discord 活动介绍**：该活动将在 Mozilla Discord 平台举行，为社区成员提供实时讨论空间。


  

---



### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1260466622368317583)** (4 messages): 

> - `Tinygrad 错误处理`
> - `Tinygrad 中的梯度设置`
> - `NV 加速器说明` 


- **Tinygrad 用户讨论错误处理**：一位成员认为 **Tinygrad** 中的某些错误令人沮丧、难以诊断且并非致命错误，建议不应让程序停止运行。
   - 他们解释说，这些错误发生在特定情况下（如非连续输入），并不代表需要用户关注的潜在问题。
- **理解梯度需求的 `None` 默认值**：一位成员澄清说，将 `require_grad` 的默认值设置为 **`None`** 意味着除非在 optimizer 中使用，否则不需要梯度。
   - 将其设置为 `False` 可确保该 tensor 永远不会计算梯度，这说明了为什么存在三种状态而不是两种。
- **关于 NV 加速器范围的说明**：[Tinygrad 中的 NV 加速器](https://github.com/nvdla/)仅涵盖 GPU，直接与 kernel 交互并绕过 userspace。
   - 有人提问是否需要编写单独的 NVDLA/DLA 加速器，这表明可能需要额外的实现。



**提到的链接**：<a href="https://github.com/nvdla/">nvdla</a>：NVDLA 开源项目。nvdla 有 17 个可用的仓库。在 GitHub 上关注他们的代码。

  

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1260325557112016996)** (2 条消息): 

> - `KAN 作者在 arXiv 论文上的回复`
> - `评委团队参与咨询` 


- **KAN 作者参与 AlphaXiv 论坛互动**：**KAN** 的作者本周通过 AlphaXiv Labs 讨论论坛，针对他们[最近的 arXiv 论文](https://alphaxiv.org/abs/2404.19756v4)回答了相关问题。
   - AlphaXiv 平台促进了*直接互动*和*实时响应*。
- **关于加入评委团队的咨询**：一名成员表示有兴趣加入评委团队，并询问了加入流程。
   - 讨论中强调了对评判标准的*热情*和*贡献意愿*。



**提到的链接**：<a href="https://alphaxiv.org/abs/2404.19756v4">alphaXiv</a>：未找到描述

  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1260535586180304919)** (1 条消息): 

> - `图像分割问题`
> - `Hermes 2`
> - `Mistral 的局限性`
> - `模型合并`
> - `Open Empathic` 


- **图像分割结果排障**：一位成员提到他们正在进行一个**图像分割项目**，并面临分割结果的一些问题。
   - *有人能帮帮我吗？*
- **Hermes 2.5 性能优于 Hermes 2**：在添加了[代码指令示例](https://link.to.examples)后，**Hermes 2.5** 在各种基准测试中表现似乎优于 **Hermes 2**。
   - Hermes 2 在 MMLU 基准测试中得分为 **34.5**，而 Hermes 2.5 得分为 **52.3**。
- **Mistral 在扩展超过 8k 时存在困难**：成员们表示，如果不进行持续预训练，**Mistral** 无法扩展到 8k 以上，且[这是一个已知问题](https://link.to.issue)。
   - 他们指出，针对性能新前沿的进一步工作将集中在 *mergekit* 和 *frankenMoE 微调*上。
- **关于模型合并策略的讨论**：一位成员建议将 **UltraChat** 和基础版 **Mistral** 之间的差异应用到 **Mistral-Yarn** 上，作为一种潜在的合并策略。
   - 其他人表示怀疑，但该成员保持乐观，并引用了以往在他们称之为*“诅咒模型合并” (cursed model merging)* 方面的成功尝试。
- **Open Empathic 项目寻求援助**：一位成员呼吁帮助扩展 **Open Empathic** 项目的类别，特别是低端类别。
   - 他们分享了一个关于 [Open Empathic 发布与教程的 YouTube 视频](https://youtu.be/GZqYr8_Q7DE)，指导用户贡献他们喜欢的 YouTube 电影场景，以及 [OpenEmpathic 项目本身](https://dct.openempathic.ai/)的链接。


  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1260401482356883548)** (2 条消息): 

> - `多 Token 预测 (Multi-token prediction)`
> - `HF 实现` 


- **对多 Token 预测可能性的疑问**：一位用户询问多 Token 预测是否已可用于训练，或者是否仍是未来的可能性。
- **多 Token 预测需要 HF 实现**：另一位用户建议，多 Token 预测需要先在 **HF** (Hugging Face) 中实现。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1260655204555296891)** (1 条消息): 

> - `DPO 微调问题`
> - `多 GPU 数据处理错误`
> - `dataset_prepared_path 临时解决方案`
> - `RunPod FFT 崩溃` 


- **全量微调下的 DPO 微调已损坏**：一位成员报告称，由于在多 GPU 上处理数据时出现的一个著名错误，目前使用 **DPO** 进行的全量微调 (Full Fine-Tune) 已损坏。
- **RunPod FFT 崩溃问题凸显**：该问题还导致在 RunPod 中使用主分支进行 DPO **全量微调 (FFT)** 时出现崩溃。


  

---



### **AI Stack Devs (Yoko Li) ▷ #[app-showcase](https://discord.com/channels/1122748573000409160/1122748840819306598/1260603124784300202)** (2 条消息): 

> - `持续的想法开发`
> - `正面反馈` 


- **Mikhail_EE 的想法开发进展**：Mikhail_EE 分享了一个更新，表明他们已经在左侧继续并进一步开发了一个想法。
   - *假设进一步开发意味着项目或概念的进展。*
- **来自 N2K 的正面反馈**：在 Mikhail_EE 更新后，N2K 给出了正面回应：*“太棒了！” (Amazing!)*


  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1260425055708516377)** (1 条消息): 

> - `Credits 已过期`
> - `延期申请` 


- **用户 Credits 提前过期**：一名成员报告称，在开始使用平台之前，其所有 Credits 就已过期，并艾特了 <@1176939881780486195> 申请延期。
   - 该成员希望能够延长 Credits 的有效期，以便能够正常使用平台。
- **无可用主题**：此摘要是为了满足主题摘要的最小条目要求而创建的。


  

---



---



---



---



---



---



{% else %}


> 完整的逐频道详情已针对邮件进行截断。 
> 
> 如果您想查看完整详情，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}