---
companies:
- langchain
- meta
- microsoft
- openai
- pytorch
- ray
- claude
date: '2025-10-22T05:44:39.731046Z'
description: '以下是该文本的中文翻译：


  **LangChain 与 LangGraph 1.0** 正式发布，带来了针对可靠、可控智能体的重大更新及统一文档，并强调了“智能体工程”（Agent Engineering）的概念。**Meta**
  推出了 **PyTorch Monarch** 和 **TorchForge**，分别用于分布式编程和强化学习，旨在支持大规模智能体系统。**Microsoft
  Learn MCP** 服务器现已与 **Claude Code** 和 **VS Code** 等工具集成，支持即时文档查询，从而加速了基于事实的智能体工作流。**vLLM**
  通过支持 Token ID 返回和批次无关推理提升了推理的正确性，并与 **Ray** 合作在 PyTorch 基金会框架下进行任务编排。**OpenAI**
  推出了浏览器智能体 **ChatGPT Atlas**，具备上下文问答和高级安全功能，但早期用户指出其在成熟度方面仍面临挑战，并对凭据访问权限持谨慎态度。'
id: MjAyNS0x
models:
- vllm
- chatgpt-atlas
people:
- hwchase17
- soumithchintala
- masondrxy
- robertnishihara
- cryps1s
- yuchenj_uw
title: 今天没发生什么特别的事。
topics:
- agent-frameworks
- reinforcement-learning
- distributed-computing
- inference-correctness
- serving-infrastructure
- browser-agents
- security
- middleware
- runtime-systems
- documentation
---

平静的一天。

> 2025/10/21-2025/10/22 的 AI 新闻。我们为您检查了 12 个 subreddits、544 个 Twitter 账号和 23 个 Discord 社区（198 个频道，7314 条消息）。预计节省阅读时间（以 200wpm 计算）：528 分钟。我们的新网站现已上线，支持全元数据搜索，并以精美的 vibe coded 风格展示所有往期内容。详见 https://news.smol.ai/ 并通过 @smol_ai 向我们提供反馈！

如果你对在工作中使用 AI 编程充满热情或积极倡导，AIE CODE 的演讲者（原 Workshop）[今天已公布](https://x.com/aiDotEngineer/status/1981062300254818356)。完整名单请见[官网](https://www.ai.engineer/code/2025)，最后一轮申请已于今日开始。赞助席位已售罄，门票也即将售罄。

[AI Engineer Code Summit 2025 演讲者名单，汇集了来自各家公司和研究机构的专家。](https://resend-attachments.s3.amazonaws.com/gMAzQRqiIlzI2KY)

---

# AI Twitter 回顾

**Agent 框架、编排和 RL 工具（LangChain/LangGraph 1.0, PyTorch Monarch + Forge, MCP 生态系统）**

- **LangChain & LangGraph 1.0 (Python + TypeScript)**：重大重写，专注于可靠且可控的 Agent。亮点：新的 `create_agent` 模板；与供应商无关的“标准内容块”；用于可控性和上下文工程的中间件；以及通过 LangGraph 运行时实现的持久化、人机回环（human-in-the-loop）执行。LangChain、LangGraph 和 LangSmith 的统一文档已上线，团队正明确向“Agent 工程”转型。公告与深度解析：[@hwchase17](https://twitter.com/hwchase17/status/1981030005229670438), [@LangChainAI](https://twitter.com/LangChainAI/status/1981030195873333269), [圆桌会议回顾](https://twitter.com/bromann/status/1981076440780013666)。
- **PyTorch 的新分布式与 RL 栈**：Meta 推出了两个用于大规模 Agent 系统的构建块：[Monarch](https://twitter.com/PyTorch/status/1981020264474231030)（一个用于编排集群、调试和预训练的分布式编程框架）和 [TorchForge](https://twitter.com/PyTorch/status/1981035379126890748)（一个具有高性能组件和示例的 PyTorch 原生 RL 库）。此举强调了 Agent 工作负载从研究到生产的端到端路径。预告：[@soumithchintala](https://twitter.com/soumithchintala/status/1980812457301160196)。
- **MCP 走向主流**：Microsoft Learn MCP 服务器使官方文档在 Claude Code 和 VS Code 等工具中可立即查询——无需身份验证，兼容 OpenAI——加速了基于事实的 Agent 工作流：[@code](https://twitter.com/code/status/1981076900471562579)。LangChain 文档现在内置了 MCP：[@masondrxy](https://twitter.com/masondrxy/status/1981003281603428670)。

**推理正确性和服务基础设施（vLLM + Ray）**

- **消除 Agent RL 中的重新分词偏移**：vLLM 的 OpenAI 兼容端点现在可以直接返回 token ID——添加 `"return_token_ids": true`——防止导致 RL 不稳定的细微 string→token 不匹配（例如 JSON 重新格式化、模板差异）。这是与 Agent Lightning/MSR 的出色合作，对于任何构建自我进化 Agent 的人来说都值得一读：[@vllm_project](https://twitter.com/vllm_project/status/1981017184769061153)。
- **Batch 不变量推理**：vLLM 引入了一个单标志开关，用于在不同 batch size（包括 prefill）下获得位级等效的结果：设置 `VLLM_BATCH_INVARIANT=1`。这极大地简化了服务栈的调试和可复现性：[@vllm_project](https://twitter.com/vllm_project/status/1981088861506982041)。
- **vLLM x Ray，现已加入 PyTorch 基金会**：随着推理变得复杂，协调和放置变得至关重要。PyTorchCon 上的演讲强调了跨节点并行、prefill-decode 分离、前缀感知路由和广泛的专家并行——由 Ray 提供编排，vLLM 作为引擎：[@robertnishihara](https://twitter.com/robertnishihara/status/1981112722361372924), [@vllm_project](https://twitter.com/vllm_project/status/1981045521671393441)。

**浏览器 Agent 与安全（OpenAI Atlas 发布及反响）**

- **OpenAI 的 ChatGPT Atlas**：浏览器集成了一个可以在页面上执行操作的 Agent，并引入了 “Ask ChatGPT”（上下文页面问答）以及纵深防御安全措施：针对无凭据操作的登出模式、针对敏感网站的 “Watch Mode”，以及对提示词注入攻击的快速响应。OpenAI 详细介绍了广泛的红队测试和旨在忽略恶意指令的新训练——同时指出攻击仍然是一个未解决的前沿领域：[@cryps1s](https://twitter.com/cryps1s/status/1981037851279278414), [@OpenAI](https://twitter.com/OpenAI/status/1981098271901962439)。
- **来自从业者的现实反馈**：早期用户报告 “Agent 模式” 经常过度思考并停滞；建议在授予凭据或电子邮件访问权限时保持谨慎。预计会有较长的成熟曲线：[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1980846874904219932), [后续](https://twitter.com/Yuchenj_UW/status/1980847565819302116), [密码风险](https://twitter.com/Yuchenj_UW/status/1980855677397659869)。

**多模态浪潮：OCR/VLM 以及 3D/视频**

- **OCR 走红（开源、快速、廉价）**：AI2 发布了采用 Apache-2.0 许可的 olmOCR 2，带来了新数据集、经过单元测试的合成训练，并声称达到 SOTA——成本约为每 100 万页 178 美元左右；模型、FP8 版本及公开 Demo 已发布：[@allen_ai](https://twitter.com/allen_ai/status/1981029159267659821), [概览](https://twitter.com/mervenoyann/status/1981040748133826918)。据报道，DeepSeek-OCR 在社区测试中领先于 Qwen3-VL；部署模板和端点正在激增（[Baseten](https://twitter.com/basetenco/status/1980924381217104338), [HF Endpoints 目录](https://twitter.com/ErikKaum/status/1980965155145216336)）。这里整理了一份具有竞争力的 OCR/VLM 短名单：[@HarveenChadha](https://twitter.com/HarveenChadha/status/1981055277408669934)。
- **新的 VLM 和数据集**：Qwen3-VL 登陆 HF，支持 1M 上下文和更强的 GUI/视频推理能力：[@HuggingPapers](https://twitter.com/HuggingPapers/status/1980809413045940553)。Liquid AI 的小型 VLM **LFM2-VL-3B** 在 MM-IFEval 上达到 51.8%，在 RealWorldQA 上达到 71.4%，具有多语言 OCR 优势和低幻觉率：[@LiquidAI_](https://twitter.com/LiquidAI_/status/1980985540196393211)。Hugging Face 推出了 **FineVision**（跨 185 个子集的 2400 万个精选多模态样本），以标准化 VLM 预训练：[@HuggingPapers](https://twitter.com/HuggingPapers/status/1981093262912819418)。
- **3D/视频生成**：腾讯开源了 **Hunyuan World 1.1 (WorldMirror)**，这是一个单次前馈的视频/多视图转 3D 重建模型，可在单张 GPU 上数秒内输出点云、深度、法线、相机参数和 3D Gaussians——具有灵活的几何先验以保证一致性：[@TencentHunyuan](https://twitter.com/TencentHunyuan/status/1980930623536837013)。在视频生成方面，请关注 UltraGen 和 MoGA 中新的长文本注意力机制方法：[@_akhaliq](https://twitter.com/_akhaliq/status/1980952631544799705), [MoGA](https://twitter.com/_akhaliq/status/1980952993563349127)。

**前沿模型与方法（DeepSeek v3.2、记忆层、Token 效率、生物医学）**

- **DeepSeek v3.2 (685B MoE) 专注于长上下文成本/速度**：关注“最相关的 Token”，长上下文推理速度比 v3.1 快 2-3 倍，处理成本比 v3.1 低 6-7 倍。提供 MIT 许可权重；API 定价为每 100 万输入/缓存/输出 Token 分别为 0.28/0.028/0.42 美元；针对华为/国产芯片进行了优化。性能与 v3.1 大致相似，在编程/Agent 任务上有小幅提升，在某些数学/科学任务上略有下降：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1980846573681520824)。
- **持续学习“记忆层”**：一种提出的与输入无关的 KV 记忆层，仅在具有高 TF-IDF 的槽位上进行微调，引起了人们对可扩展持续学习的浓厚兴趣（[推文 + 论文摘要](https://twitter.com/giffmana/status/1980869216149619009)）。后续研究提出了两个实际点：包含一个“汇（sink）”槽位以允许选择“不使用记忆”，并注意内循环中随机内存访问带来的性能/吞吐量损失：[@BlackHC](https://twitter.com/BlackHC/status/1981022197415068129), [@gallabytes](https://twitter.com/gallabytes/status/1981038852539371969)。
- **通过图像实现 Token 效率 + 生物医学分辨率**：研究人员继续探索将文本编码为图像，以使多模态 LLM 的 Token 计数几乎减半（[论文/代码](https://twitter.com/iScienceLuvr/status/1980942325573648703)），而“原生分辨率”训练/推理显著改善了生物医学 MLLM（[论文](https://twitter.com/iScienceLuvr/status/1980944519001727281)）。同样值得注意的还有：针对 MEG 神经影像数据的 Transformer 基础模型 MEG-GPT 的首次尝试（[摘要](https://twitter.com/iScienceLuvr/status/1980945270369399234)）。

**相关计算与数据集**

- **可验证的量子优越性 (Google)**：通过在 Willow 芯片上使用 “Quantum Echoes” (OTOC) 测量，Google 报告了首个可验证的量子优越性——比顶级超级计算机上的最佳经典算法快 13,000 倍——在基于 NMR 的材料/药物发现分子建模中具有潜在应用。该研究已在 Nature 经过同行评审；通过在其他量子设备/实验上的重复进行了验证：[@sundarpichai](https://twitter.com/sundarpichai/status/1981013746698100811), [@GoogleQuantumAI](https://twitter.com/GoogleQuantumAI/status/1981016219340648778)。
- **大规模 Agent 训练数据**：IBM + 华盛顿大学在 Hugging Face 上发布了一个包含 150 万个任务场景的数据集，以推动 Agent 评估和“完成任务”工作流：[@IBMResearch](https://twitter.com/IBMResearch/status/1981066891062817274)。此外，斯坦福大学新的 CME295 (Transformers & LLMs) 课程上线，DeepMind + UCL 发布了免费的 AI Research Foundations 课程：[@omarsar0](https://twitter.com/omarsar0/status/1981030346037612847), [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1980962352775176637)。

**热门推文（按互动量排序）**

- Google 在 Willow 芯片上通过 “Quantum Echoes” 声称实现可验证的量子优越性（提速 13,000 倍）：[@sundarpichai](https://twitter.com/sundarpichai/status/1981013746698100811)。
- OpenAI 的 Atlas 添加了 “Ask ChatGPT” 功能，可读取当前页面以提供即时回答：[@OpenAI](https://twitter.com/OpenAI/status/1981098271901962439)。
- 对 ChatGPT 的 Agent/浏览用户体验及安全担忧的早期反应：[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1980846874904219932)。
- 腾讯 Hunyuan World 1.1 开源了前馈视频到 3D 世界重建技术：[@TencentHunyuan](https://twitter.com/TencentHunyuan/status/1980930623536837013)。
- Higgsfield Popcorn 推出了具有一致性角色编辑功能的 AI 分镜脚本工具：[@higgsfield_ai](https://twitter.com/higgsfield_ai/status/1981110992630341928)。

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Qwen 团队对 llama.cpp 的贡献

- [**Qwen 团队再次助力 llama.cpp**](https://www.reddit.com/r/LocalLLaMA/comments/1oda8mk/qwen_team_is_helping_llamacpp_again/) (热度: 1035): **这张图片是来自 GitHub 的截图，展示了 Qwen 团队成员的一篇帖子，详细说明了他们对** `llama.cpp` **项目的贡献。帖子提到了具体的背景技术更新，例如修复 Vision Transformer (ViT) 的位置嵌入 (positional embeddings) 以及修正 DeepStack 的实现。这表明了在** `llama.cpp` **项目中的持续协作与改进，该项目是在消费级硬件上高效运行大语言模型 (LLM) 的流行实现。带有边界框 (bounding boxes) 的纸杯蛋糕图片可能是为了直观展示软件中与目标检测或图像处理功能相关的特性。** 评论反映出一种观点，即非中国 AI 实验室的产出速度已经放缓，而像阿里巴巴这样的中国公司正在迅速推进。此外，人们对 Qwen 团队亲力亲为的代码编写方式表示赞赏，并建议他们协助完成 Qwen3-Next 架构。
    - -p-e-w- 的评论强调了 Google、Meta 和 Microsoft 等主要的非中国实验室在 AI 模型发布方面表现出的停滞感，并将其与 DeepSeek 和阿里巴巴等中国公司的快速发展步伐进行了对比。这表明 AI 领域正在发生转变，中国公司在推动 AI 技术边界方面变得更加突出。
    - YearZero 讨论了 Qwen 团队协助 Qwen3-Next 架构的可能性，并引用了 GitHub 上的一个特定 Pull Request ([链接](https://github.com/ggml-org/llama.cpp/pull/16095))。该评论暗示项目已接近完成，并可能从同行评审中受益以最终确定架构，体现了协作开发的模式。
    - GreenPastures2845 提供了一个 GitHub Issue 评论的链接 ([链接](https://github.com/ggml-org/llama.cpp/issues/16207#issuecomment-3432273713))，其中可能包含与 llama.cpp 项目相关的进一步技术见解或讨论。这表明社区围绕该项目有着持续的参与和技术交流。
- [**嘿 [Z.ai](http://z.ai/)，两周期限昨天就到了**](https://www.reddit.com/r/LocalLLaMA/comments/1od1hw4/hey_zai_two_weeks_was_yesterday/) (热度: 514): **这张图片是一个迷因 (meme)，强调了 [Z.ai](http://z.ai/) 在 Twitter 交流中提到的 “GLM 4.6 Air” 发布延迟。Ivan Fioravanti 用一个 GIF 幽默地期待发布，而 [Z.ai](http://z.ai/) 则以典型的开发者承诺“两周内准备好”作为回应，这是软件开发中表示无限期延迟的常见套路。评论反映了社区支持的态度，强调了开源贡献的自愿性质，并表达了测试新模型的渴望，特别是与来自 REAP GLM 4.6 的现有版本（如 q4 GGUF 或 AWQ）进行对比。** 评论者普遍表示理解和耐心，承认 [Z.ai](http://z.ai/) 工作的自愿和开源性质，并对将即将发布的版本与现有模型进行比较表现出兴趣。
    - Leflakk 表达了对测试 REAP GLM 4.6 模型的 Q4 量化格式（特别是 GGUF 或 AWQ）的兴趣。这表明人们关注这些量化方法之间的性能和效率对比，这对于在资源受限的环境中优化模型部署至关重要。
    - nuclearbananana 提到“两周”的时间线是大概的，暗示软件开发的时间表可能是流动的且易于变化。这强调了项目管理中灵活性的重要性，特别是在贡献通常是自愿的开源项目中。
    - inkberk 强调了 [Z.ai](http://z.ai/) 对开源社区的重大贡献，暗示他们的工作对 AI 技术的开发和普及产生了实质性影响。这凸显了社区驱动项目在推动技术创新方面的价值。

## 非技术性 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Google 在量子计算领域的突破

- [**Google 在利用量子计算进行药物发现和材料科学方面的突破**](https://www.reddit.com/r/singularity/comments/1odbbbr/google_breakthrough_in_using_quantum_computing/) (活跃度: 1125): **Google 宣布其 Willow chip 在量子计算方面取得了重大突破，利用一种名为 Quantum Echoes 的算法实现了可验证的量子优越性 (quantum advantage)。据报道，该算法比经典算法快** `13,000 倍`**，能够通过核磁共振解释分子相互作用。这一进展对于药物发现和材料科学具有重大的潜在影响，标志着量子计算在现实世界应用中迈出了关键一步。更多详情请参阅 [Google 博客文章](https://blog.google/technology/research/quantum-echoes-willow-verifiable-quantum-advantage/)。** 评论者正在讨论 Google 量子计算里程碑的时间表，一些人对未来的里程碑和面临的挑战表示好奇。这些里程碑的路线图可以在 [Google 的 Quantum AI roadmap](https://quantumai.google/roadmap) 上找到。

### 2. 人形机器人与 AI 交互

- [**AheadForm 发布新款男性人形机器人面部 Origin M1**](https://www.reddit.com/r/singularity/comments/1od7n5c/aheadform_unveils_their_new_male_humanoid_robot/) (活跃度: 667): **AheadForm 推出了** `Origin M1`**，这是一款全新的男性人形机器人面部，正如在 [X (原 Twitter)](https://x.com/XRoboHub/status/1980886176845517175) 上宣布的那样。该设计旨在通过提供更具亲和力和表现力的界面来增强人机交互。此次发布突显了机器人领域日益增长的趋势，即创造更逼真、更具情感吸引力的机器，尽管公告中未披露具体的材料、表情范围或底层的 AI 技术等技术细节。** 评论反映了对机器人设计的怀疑和批评，一些用户质疑审美选择和性别化外观的必要性，这表明了关于机器人拟人化 (anthropomorphism) 的更广泛辩论。
- [**正在想念 Honda ASIMO 🥀**](https://www.reddit.com/r/singularity/comments/1odab1q/thinking_about_honda_asimo_rn/) (活跃度: 448): **图片展示了 Honda 的 ASIMO，这是一款以先进的移动性和交互能力著称的人形机器人，正站在一个人旁边。由 Honda 开发的 ASIMO 是人形机器人领域的先驱项目，展示了先进的行走、跑步和交互能力。尽管取得了技术成就，但 ASIMO 通常被视为一个超越时代的项目，类似于 Segway，并未导致人形机器人的广泛采用。评论反映了一种观点，即日本尽管在工业机器人领域实力雄厚，但在现代人形机器人的开发中并未保持领先地位，而这些机器人目前正由日本以外的公司推进。** 评论者对日本尽管在机器人领域拥有历史领先地位，却未能继续在人形机器人领域领跑表示失望，现代进展正由非日本公司驱动。
    - Distinct-Question-16 强调了 Honda ASIMO 的运行限制，指出它只能运行 '30 分钟 - 1 小时' 且需要 '3 小时充电'。这反映了人形机器人要在现实应用中变得更加实用和自主，在电池技术和能源效率方面仍需取得重大进展。

### 3. 影响 ChatGPT 的 Meta 政策变化

- [**Lol OpenAI 在这里完胜 Meta**](https://www.reddit.com/r/OpenAI/comments/1od4xmy/lol_openai_cooked_meta_here/) (活跃度: 1057): **这张图片是一个梗图，重点展示了 OpenAI 针对 Meta 政策变动发布的一条推文。该政策变动将从 2026 年 1 月 15 日起影响 WhatsApp 上 1-800-ChatGPT 的功能。推文向用户保证，ChatGPT 仍可通过 App、网站和浏览器等其他平台访问。这反映了各大科技公司之间持续的紧张关系和竞争动态，特别是在他们如何管理第三方集成和平台政策方面。** 评论反映了对该帖子的批判性观点，用户对盲目崇拜某一家大科技公司而贬低另一家表示怀疑，并认为该帖子无关紧要。
- [**是的，真的很幸运**](https://www.reddit.com/r/ChatGPT/comments/1od4yqn/yea_truly_luckily/) (活跃度: 502): **这张图片是一个梗图，重点展示了 OpenAI 关于 Meta 政策变动的推文，该变动将在 2026 年 1 月 15 日前禁用 WhatsApp 上的 1-800-ChatGPT 服务。OpenAI 向用户保证，ChatGPT 仍可通过 App、网站和浏览器访问。这一变化反映了影响 AI 服务集成的平台政策的持续调整。** 评论者对通过 WhatsApp 访问 ChatGPT 的实用性提出质疑，认为这可能是为了无障碍访问。此外，人们对浏览器的功能也持怀疑态度，担心可能会出现 OpenAI 惯有的限制。
    - *si1endeath* 对浏览器的限制表示担忧，指出它阻止了对许多网站的访问，这与 **OpenAI** 的内容审查（content moderation）方法一致。这可能会影响那些依赖不受限浏览进行研究或其他目的的用户。
    - Erik-AmaltheaFairy 询问了集成 GPT 的浏览器的运行模式，推测其对免费用户可能存在的限制，例如搜索上限或升级到付费订阅的提示。这反映了人们对 AI 服务变现策略的广泛关注。

---

# AI Discord 摘要

> 由 Gemini 2.5 Pro Exp 生成的摘要之摘要之摘要
> 

**1. 野生 AI 模型：性能、成本与怪癖**

- **Gemini 3 发布传闻落空**：最初在 **LMArena** 和 **Perplexity** 上的猜测指向 **Gemini 3** 将于 10 月底发布，但根据 [AI.google.dev](https://ai.google.dev/) 的最新报告，现在建议在 12 月进行预览，并在 1 月正式发布。同时，在 **Lithiumflow**（据称是 **Gemini 3 Pro**）上的实验显示了令人印象深刻的代码编写能力，但在更具体的 Prompt 上失败了。
- **Claude 模型掏空开发者钱包**：**Cursor Community** 和 **MCP Contributors** Discord 的工程师报告称，使用 **Claude 4 Sonnet** 等模型非常昂贵，在最高模式下成本达到 **每次请求 7 美元**。在多 Agent（multi-agent）设置中，成本激增至 **每次操作 7-8 美元**，迫使一些人放弃该平台而转向自定义 API 解决方案。
- **Sora 2 及其同类展示实力，但有限制**：**OpenAI** 和 **Nous Research** 社区的用户正在使用 **Sora 2** 生成视频，但报告称每天有 **30 个视频** 的限制。讨论还强调 **Veo 3** 视频生成缺乏声音和模型选择，而用户注意到 **GPT-4o** 仍然能成功完成 **GPT-5** 无法实现的口音。

**2. 开发者体验：工具、IDE 和 API**

- **Cursor IDE 充斥着安全漏洞和 Bug**：在 **Cursor Community** 中流传的一篇 [BleepingComputer 文章](https://www.bleepingcomputer.com/news/security/cursor-windsurf-ides-riddled-with-94-plus-n-day-chromium-vulnerabilities/) 指出，由于 **Chromium** 引擎过时，该 IDE 存在超过 **94 个 n-day 安全问题**。这一消息传出时，正值 [Cursor 大规模停机](https://status.cursor.com/) 以及有报告称 Bug 导致 **"apply"** 功能失效，令编码人员感到沮丧。
- **OpenRouter 通过 :exacto 端点强化 Tool-Calling**：**OpenRouter** 推出了 **:exacto** 端点，通过将请求路由到具有卓越结构化输出（structured-output）性能的提供商，来提高 Tool-Calling 的准确性，详见其 [公告帖子](https://openrouter.ai/announcements/provider-variance-introducing-exacto)。此举旨在解决性能差异问题，初步基准测试显示，`qwen/qwen3-coder:exacto` 等模型的 **Tool-Calling 成功率有了实质性提升**。
- **工程师们正与庞大的 Tool Contexts 搏斗**：在 **MCP Contributors** Discord 中，管理拥有 **60 多个工具** 的服务器的开发者正因冗长的工具描述而触及上下文限制（context limits）。一位工程师设计了一个仅含 **3 个工具**（list, describe, invoke）的工作流来管理超过 **50 个 CLI 操作**，展示了对精简方法的需求，以避免模型过载并产生高昂成本。

**3. 硬件与系统优化：挑战性能极限**

- **GPU 辩论：云端租赁 vs 实体设备**：在 **Unsloth AI** 的 Discord 中，工程师们就租用一台拥有 **200GB VRAM**、价格为 **$4k** 的 **DGX Spark**，还是以类似价格购买一台 **RTX 6000 Pro** 的经济性展开了辩论。而在 **Yannick Kilcher** 的 Discord 中，没有本地 GPU 的研究人员建议在 [vast.ai](https://vast.ai/) 上以低至 **每小时 $0.15** 的价格租用 **RTX 3090**，以大幅缩短实验运行时间。
- **PyTorch Helion Kernel 接受基准测试并遭到质疑**：在 **PyTorch Helion** 发布（[博客文章](https://pytorch.org/blog/helion/)）后，**GPU MODE** 社区的成员对其报告的 **14 倍加速**提出了挑战，认为这不切实际。他们认为，[int4_gemm 实现](https://github.com/pytorch/helion/blob/main/examples/int4_gemm.py#L166-L178) 应该与 **Marlin** 等融合 Kernel 或简单的 **Triton gemm kernel** 进行基准测试对比，以实现更公平的比较。
- **Mojo 语言在高端硬件上出现 Segfault**：**Modular** 社区的开发者报告称，最新 nightly 版本的 **Mojo** 在 **H100** GPU 上加载 `Llama-3.1-8B-Instruct-GGUF` 时会导致 Segfault。该问题似乎源于 GPU 尝试运行模型的 **bf16** 版本，而 CPU 则正确运行了 **q4_k** 反量化 Kernel。

**4. 快速变化的 AI 格局：新发布与大厂动态**

- **Google 量子芯片声称比超级计算机快 13,000 倍**：根据 [X 上的帖子](https://x.com/googleai/status/1981022228801307035)，**GoogleAI** 宣布了一个重大的量子计算里程碑，其 **65-qubit Willow 芯片**利用 **Quantum Echoes 算法**执行任务的速度比顶级超级计算机快 **13,000 倍**。这一消息在 **Latent Space** 中分享后，引发了关于其对密码学和科学建模影响的讨论。
- **Unsloth 与 PyTorch 宣布量化合作**：根据 [X 上的公告](https://x.com/UnslothAI/status/1981021761782317368)，**Unsloth AI** 和 **PyTorch** 正在合作开展一项新的 **Quantization Aware Training (QAT)** 计划。这次合作引发了工程师们关于其实现的各种技术问题，特别是它是否会保持 vision encoder 不受影响。
- **面向 Agent 和 LLM 系统的开源工具受到关注**：**HuggingFace** 社区见证了 **Fenic** 的发布，这是一个直接与 **Hugging Face Datasets** 集成以通过版本化 context 创建 Agent 的新工具，其 [仓库位于 GitHub](https://github.com/typedef-ai/fenic)。在 **GPU MODE** 中，一个新的 [Awesome LLM Systems](https://github.com/romitjain/awesome-llm-systems) 仓库被引入，用于整理关于 LLM 部署和优化的论文及资源。

**5. 用户困扰与平台问题：Bug、计费与糟糕的支持**

- **Perplexity 推荐计划因欺诈标记遭抨击**：**Perplexity AI** Discord 的用户报告了推荐计划的重大问题，声称合法的引流未被计算在内，且部分账户在被标记为欺诈后收到了 **0 分钱** 的支付。这导致了关于“优质引流”标准的广泛猜测，以及对该计划不可靠性的沮丧。
- **OpenAI 用户对神秘的每日扣费感到困惑**：一位 **OpenAI** 用户报告称，自 10 月 9 日以来，尽管已删除了所有 API keys 和项目，但仍被收取每天 **$15 USD** 的固定费用。他们分享了 [截图](https://cdn.discordapp.com/attachments/998381918976479273/1430552464108556308/Captura_de_Tela_2025-10-22_as_10.43.57.png?ex=68fa314d&is=68f8dfcd&hm=ac45c8620b631401aa0bc85ff59c1098a590e604ce27d207135a45798d95ad4c&) 并向社区寻求答案，因为官方支持尚未解决该问题。
- **Manus.im 用户谴责积分制度的“诱导转向”并遭遇账号封禁**：**Manus.im** 社区对平台的积分系统感到愤怒，用户觉得他们在 **Pro Plan** 上遭遇了“诱导转向（bait-and-switch）”，因为该计划不再提供无限积分。除了对积分的沮丧外，新用户还报告称他们的账号在输入支付信息后立即被封禁，且没有明确的原因或解决途径。

---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **推荐计划结果陷入争议**: 用户报告 Perplexity AI 推荐计划存在问题，包括潜在客户未被计入，以及系统可能将合法的推荐标记为欺诈，导致审核后支付金额为 **0 元**。关于其策略以及旧推荐计划价值的猜测层出不穷。
   - 许多人讨论了“优质潜在客户”的标准，以及通过 VPN 和虚拟机获取更多推荐的潜在手段；一些人报告成功，而另一些人则表达了不满；此外，一些用户遇到了 **Comet 浏览器**重置账户和 Cloudflare 检查的问题。
- **Perplexity Pro 的争议：无限还是有限？**: 在一名用户分享了一张显示使用限制的图片后，用户们对 **Perplexity Pro** 的“无限访问”限制展开了辩论，引发了关于订阅条款和限制可能更新的讨论。
   - 一位用户询问 **Perplexity API** 是否可以访问 **ChatGPT5** 和 **Claude**，另一位成员回答说该 API 仅限于 **Sonar**。
- **Gemini 初现，中国可能正在追赶**: 关于 **Gemini 3** 发布日期的猜测不断，有人提到可能在 12 月发布，同时也讨论了西方 AI 对其产品施加的限制可能为中国占据领先地位铺平道路。
   - 一些人担心东方可能会接管软件和 AI 开发。
- **Android 应用问题困扰用户**: 用户报告了 Perplexity 的技术问题，包括 **Android 应用 UI 回退到旧版本**，以及移动应用、Comet 浏览器和网页版上持续出现的 *something went wrong* 错误。
   - 这些问题特别涉及文件生成，一些用户发现应用在编写代码时崩溃；一位用户建议使用网页版作为替代方案。
- **数学思考与艺术美学的融合**: 一位成员分享了[他们关于艺术家 0thernes 数学艺术的已发表文章链接](https://generativeai.pub/the-math-art-of-artist-0thernes-not-the-typical-96e009060bc1)并征求反馈。
   - 此外，他们还分享了几个 **Perplexity AI** 页面链接，包括关于 [Time-Based Researcher](https://www.perplexity.ai/search/time-base-researcher-LxNZL3iFRXamL0kYZV3RiA#0) 和 [特朗普的谎言与事实否认](https://www.perplexity.ai/page/trump-s-lies-and-fact-denials-cgZjiGKUTvuj6sTWzTkm0A) 的页面。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini 3 发布希望破灭**: 最初的猜测指向 **Gemini 3** 将于 10 月底发布，但随后的报告显示，[根据 AI.google.dev](https://ai.google.dev/)，12 月将进行预览，1 月正式发布。
   - 许多人曾期待 2024 年发布，但最终对更新的时间线感到失望。
- **Veo 3 生成视频，无模型选择或声音**: 用户正在使用 **Veo 3** 生成视频，并指出生成过程中缺乏模型选择，且可能没有声音。
   - 当有声音时，生成视频中的声音似乎是随机选择的，这引发了人们对 *Ling-1t* 潜力的好奇。
- **Lithiumflow 的 Gemini 3 代码表现出潜力与缺陷**: 在 **Lithiumflow**（据称是 **Gemini 3 Pro**）上进行的代码生成实验显示，它在体素艺术和魔方求解等领域具有令人印象深刻的能力，但在更具体的提示词上也会失败。
   - 尽管存在问题，它通常优于 **Sonnet 4.5** 和 *OpenAI* 等模型，但在更具体的提示词上表现不佳。
- **OpenAI 因数学发现面临欺诈指控**: **OpenAI** 面临欺诈指控，批评者指责其重复利用了现有研究的数据。
   - 评论驳斥了任何 AI 真正做出“发现”的观点，将 **OpenAI** 称为“现代史上最大的骗子”。
- **Lithiumflow 和 Orionmist：同一模型，不同访问权限？**: 有假设认为 **Lithiumflow** 和 **Orionmist** 是同一个模型，只是 **Lithiumflow** 拥有 Google Search 访问权限，但这一说法尚未得到证实。
   - 一位用户建议 **Gemini 2.5 Pro** 可能优于两者，此外 *Bytedance* 的 **Seed** 具有非常好的图像理解能力。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Claude 4 Sonnet 银行账户收割机**：成员们抱怨使用 **Claude 4 Sonnet** 非常昂贵，据报告在 **max mode** 下每次请求的费用约为 **7 美元**。
   - 尽管拥有更多请求额度并使用了 **Claude 4 Sonnet**，高昂的成本依然发生了。
- **Cursor "Apply" 工具踩下刹车**：用户报告称，新更新导致 Agent 错误地认为自己处于 *ask mode*，从而禁用了 **"apply"** 功能。
   - 沮丧的成员不得不通过查看完整输出并使用 *show code* 来手动复制代码。
- **Cursor 和 Windsurf IDE 充斥着安全漏洞**：一篇 [BleepingComputer 文章](https://www.bleepingcomputer.com/news/security/cursor-windsurf-ides-riddled-with-94-plus-n-day-chromium-vulnerabilities/) 指出，由于 **Chromium 和 V8 引擎** 过时，**Cursor 和 Windsurf IDE** 极易受到攻击，导致超过 **94 个 n-day 安全问题**。
   - 这些漏洞可能导致拒绝服务甚至远程代码执行（Remote Code Execution）。
- **2024 年 Cursor 大停机导致编码停滞**：用户经历了 **Cursor** 的广泛问题，包括连接失败和无法发送消息；[Cursor 状态页面](https://status.cursor.com/) 确认了此次 **outage**（停机）。
   - 一位用户哀叹他们的 *threejs 游戏处于停滞状态*。
- **Cursor 免费层级：从 Claude Sonnet 降级到 Haiku**：用户确认在标准的 **$20 plan** 中，一旦 **Claude Sonnet** 额度耗尽，系统会自动切换到 **free Haiku model**。
   - 一位用户提到，当处于“极端贫困”状态运行时，他们宁愿使用第三方 API key。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth SLA 费用不可用**：一位用户询问 **Unsloth SLA** 的费用，但获知目前为 **N/A**（不可用）。
   - 该用户无法确定任何具体的定价或服务水平协议。
- **互联网 IPv4 扫描导致蜜罐封禁**：一名成员提到扫描了整个 **IPv4 internet**，并警告会被 **international honeypots**（国际蜜罐）举报，需要避开某些网络。
   - 其他成员开玩笑地建议使用 botnet 进行扫描，但也承认扫描互联网是有害的。
- **LLM 无法破解加密数据**：一位用户想使用小型 LLM 对加密消息进行分类，但另一名成员指出，良好的加密会将数据变成**纯噪声**，使分类几乎不可能。
   - 一名成员开玩笑说，*要使用 ML 分析现代加密算法以寻找模式，你可能需要在未来一百万年里动用整个星球的 GPU 资源*。
- **PyTorch 与 Unsloth 联手进行 QAT**：有一个新的与 **PyTorch** 合作的 **Quantization Aware Training (QAT)** 项目，这是公告的 [链接](https://x.com/UnslothAI/status/1981021761782317368)。
   - 一名成员引用了 [这篇论文](https://www.arxiv.org/abs/2509.11986) 和 [这条 X 帖子](https://x.com/SravanthiSinha/status/1980867770498838560?) 来询问该合作是否保持 vision encoder 不变。
- **DGX Spark vs RTX 6000**：成员们辩论了购买具有 **200 GB VRAM**、售价 **4000 美元** 的 **DGX Spark** 与在 eBay 上以 **3500-4000 美元** 购买 **RTX 6000 Pro** 的成本权衡。
   - 配备了 **8x H100** 的 **DGX Spark** 被赞誉为能够实时进行 **2-4B model** 的 full SFT（全量微调）。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Meta 终止 ChatGPT WhatsApp 访问权限**：根据 [openai.com](https://openai.com/index/chatgpt-whatsapp-transition/)，Meta 的政策变更将于 **2026 年 1 月 15 日**后关闭 **WhatsApp** 上的 **1-800-ChatGPT**。
   - 团队提醒用户，**ChatGPT** 仍可通过 App、网站和浏览器使用。
- **Sora 2 视频生成设有每日限制**：用户报告称 **Sora 2** 将视频生成限制为**每天 30 个视频**，部分用户通过使用 VPN 绕过限制。
   - 一位用户感叹只需 *1-2 小时我的 Sora 就用完了 xD*。
- **莫名账单困扰 Bot 开发者**：一位用户报告称，自 10 月 9 日以来，即使删除了所有 API 密钥，其 OpenAI 账户仍产生 **15 美元的固定每日费用**，并附带了 [截图](https://cdn.discordapp.com/attachments/998381918976479273/1430552464108556308/Captura_de_Tela_2025-10-22_as_10.43.57.png?ex=68fa314d&is=68f8dfcd&hm=ac45c8620b631401aa0bc85ff59c1098a590e604ce27d207135a45798d95ad4c&)。
   - 该用户正在 OpenAI 社区寻求答案。
- **AI OS：未来趋势？**：一位成员推测，未来的**操作系统**可能会由 **LLM** 辅助驱动，利用 AI 处理系统进程和用户交互。
   - 这将是与现有操作系统设计的背离。
- **规避版权难题**：成员们讨论了如何规避版权问题，其中一人建议描述受版权保护的角色（*穿着红蓝相间、带有黑色蜘蛛网符号制服的男人*）。
   - 另一位成员警告说这可能还不够，并指出这是**受版权保护的 IP**，他们无法提供帮助。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Andromeda-alpha 正式发布**：一个新的隐身模型 **Andromeda-alpha** 专注于**图像和视觉理解**，已通过 [OpenRouter.ai](https://openrouter.ai/openrouter/andromeda-alpha) 发布以获取社区反馈。
   - 所有 Prompt 和输出都会被记录以用于模型改进，该模型仅供试用，不建议用于生产环境。
- **OpenRouter 推出 Exacto 端点**：OpenRouter 推出了 **:exacto** 端点以实现**更高的 Tool-calling 准确率**，将请求路由到具有更好结构化输出性能的提供商，详见 [博客文章](https://openrouter.ai/announcements/provider-variance-introducing-exacto)。
   - 首批上线模型包括 `moonshotai/kimi-k2-0905:exacto`、`deepseek/deepseek-v3.1-terminus:exacto`、`z-ai/glm-4.6:exacto`、`openai/gpt-oss-120b:exacto` 和 `qwen/qwen3-coder:exacto`，基准测试显示 **Tool-calling 成功率有实质性提升**。
- **Objective-AI 展示 AI 置信度**：[Objective-AI](https://objective-ai.io/) 的 CEO Ronald 介绍了一种针对符合 OpenAI 标准的补全选项的**置信度评分 (Confidence Score)**，该评分源于更智能的机制，而非 AI 的直接评估。
   - Objective-AI 提升了**成本效益**，允许用户最大限度地利用小型模型，并利用 **OpenRouter** 访问广泛的 **Large Language Models (LLMs)**。
- **RooCode 缓解资源压力**：**RooCode** 提供免费模型，如 **Grok Code Fast 1**、**Grok 4 Fast**、**Supernova 1 million** 和 **Deepseek Chat 3.1**，为代币售卖平台提供了替代方案。
   - 一位用户表示，他们在过去几个月里在 Roo Cloud 上*消耗了数十亿个 Token，它一直运行顺畅，从未被降级或限流*。
- **OpenRouter：警惕 Chutesflation**：用户报告在 OpenRouter 上遇到了意料之外的高额成本，一位用户称在极少使用的情况下，仅 **23 天**就消耗了 **5 美元**，引发了对 *chutesflation*（Chutes 导致的费用通胀）的担忧。
   - 一位用户表示：*你别无选择，你很可能已经在用 Chutes 了*，暗示 OpenRouter 默认将用户路由到该提供商。

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **在 Vast.ai 上租用 GPU 优于本地运行**：对于没有 GPU 的独立研究人员，在 [vast.ai](https://vast.ai) 上租用 **RTX 3090** 的成本约为 **每小时 0.15 美元**，可大幅缩短实验运行时间。
   - 成员指出，本地实验运行时间可能长达 **5-8 小时**，并建议由于资源限制，应使用云端 GPU 而非 Google Colab。
- **探讨函数调用（Function Calling）微调**：成员讨论了针对函数调用微调小型语言模型，并建议使用 **lm-format-enforcer** 或 **Lark Grammar** 来约束输出，使其遵循工具调用的 **JSON** 格式。
   - 他们补充说，**llguidance** 库提供了构建工具调用的语法，且延迟极低、准确度高，其结构为：`[Thinking] I will now call tool {tool_name}. [tool JSON]`。
- **Transformer Circuits 论文获得好评**：一位成员将 [Transformer Circuits 文章](https://transformer-circuits.pub/2025/linebreaks/index.html) 描述为本周 *最值得阅读的论文*，并指出其在 *架构框架方面具有重大意义*。
   - 该成员提到，该论文对他 *自己的愚蠢和无知* 进行了严谨的分析，而另一位成员尽管如此也表示赞同。
- **DeepSeek 工程师标准化 Jsonnet 配置**：DeepSeek 在其项目中使用 **jsonnet 方案**。
   - 一位成员表示 *如果 DeepSeek 都在用它，我也应该尝试一下，看看它有多好*。
- **亚马逊发布 Vibe Code IDE**：亚马逊的 **Vibe Code IDE** 结束了邀请制内测，为用户提供 **500 积分** 作为起步，其 [设计基于“规范”（spec based）](https://kiro.dev/blog/waitlist-is-over/)，围绕功能和实现的规范进行工作，而不仅仅是 Prompt。
   - 成员还指出，**Kiro** 与许多 **AI IDE** 一样，也是一个 **VScode fork**。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **新仓库 Awesome LLM Systems 走红**：一位成员介绍了一个新仓库 [Awesome LLM Systems](https://github.com/romitjain/awesome-llm-systems)，其中收录了专注于 **Large Language Models** 系统层面的精选论文、博客和资源。
   - 该仓库旨在为那些对 **LLM 部署与优化** 相关的工程挑战和解决方案感兴趣的人提供全面指南。
- **CuPy GPU 指针性能碾压 PyTorch GPU 指针**：一位成员强调了在自定义 **MatMul kernel** 中使用 **CuPy GPU 指针** 与 **PyTorch GPU 指针** 时的显著性能差异，指出存在性能差距，且在将 CuPy 数组转换为 PyTorch 张量的 **DLPack** 转换过程中存在瓶颈，详见[此截图](https://cdn.discordapp.com/attachments/1189607595451895918/1430323860464734360/Screenshot_from_2025-10-21_17-32-52.png?ex=68faade6&is=68f95c66&hm=d84234753a2510107fb4d7ecd73bbf01b7e07a92a430333bafa04d79be3e8bd3)。
   - 尽管数值结果相同，但性能滞后依然存在，这表明 **DLPack 转换** 过程可能存在低效。
- **warpGroup.arrive 注入引发关注**：详细编译显示注入了 `warpgroup.arrive` 以允许在 **GMMA functions** 中使用寄存器，这引发了人们对所有 **wgmma** 可能共享相同寄存器的担忧。
   - 该成员认为，注入是为了 *允许在 GMMA 中使用寄存器*。
- **Helion 基准测试基准受到质疑**：在 **Helion 博客文章** 发布后，一位成员对报道的 **14 倍加速** 提出了挑战，认为即使是与 **fp16** 相比的内存受限操作，这也是不切实际的，并引用 [int4_gemm 实现](https://github.com/pytorch/helion/blob/main/examples/int4_gemm.py#L166-L178) 作为一个非融合（non-fused）kernel。
   - 建议与 **Marlin**/**Machete** 等专门的 kernel 或使用 `tl.dot_scaled` 的简单 **Triton gemm kernel** 进行基准测试，以进行更公平的比较；**Helion** 团队邀请大家参加“会见 **Helion devs**”活动。
- **排行榜产生新的 sort_v2 之王**：一位用户凭借提交 ID `66238` 夺得 `sort_v2` 排行榜 **第一名**，在 B200 上耗时 **8.68 ms**，在 A100 上耗时 **16.3 ms**，此外在 H100 上成功运行耗时 **6.60 ms**，在 L4 上耗时 **52.7 ms**。
   - 另一位用户多次提交，凭借提交 ID `66241` 在 L4 上以 **52.6 ms** 和 A100 上以 **16.0 ms** 获得 **第一名**。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **AI 被称为“恶魔机器”**：成员们将生成式 **AI** 称为“恶魔机器”，这是由于艺术家群体对图像生成的抵触情绪，以及对**批判性思维能力**退化的担忧。
   - 针对这种情绪，一位成员认为 **AI** 产生幻觉的倾向实际上促使他们进行了*更多的批判性思考*。
- **Qwen3 Embedding 准确率大幅飞跃**：**Qwen3 embedding 8b** 的新量化版本在 *roocode* 代码索引方面表现出更高的准确性。
   - 据报道，与 **mxbai-embed-large:latest** 相比，相关查询的置信度分数更高，而不相关查询的分数显著降低。
- **LM Studio 插件支持第三方 LLM**：成员们讨论了使用 **LM Studio** 通过 **API key** 与第三方 **LLM** 通信的可能性。
   - 已确认通过目前处于封闭测试阶段的插件，将支持通过 [OpenAI-compatible endpoint](https://lmstudio.ai/fuutott/openai-compat-endpoint-v2) 实现此功能。
- **Chatterbox 成为本地 AI TTS 解决方案**：针对寻找 **AI 语音** 解决方案的需求，**Chatterbox** 被推荐为一款*非常出色*的**本地 AI TTS** 选项。
   - 已确认 **Chatterbox** 支持多种语言，使其成为一个多功能的方案。
- **越南并非 4090 的廉价天堂**：尽管有人询问在越南购买**廉价 3090/4090** 的事宜，但一位成员指出越南是向中国供应 **4090** 的货源地。
   - 因此，不太可能出现捡漏价。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **架构师利用 AI 进行重构**：一位成员通过提示词“作为一名专注于模块化代码和可维护性的资深架构师”来促使 **AI** 像“正常人”一样重构代码。对于变更，他们使用的提示词包括：`do not make changes or write code, answer question: do you have enough info to make these updates?` 以及 `please create the minimal changeset (no tests)`。
   - 这样，**AI** 就能以人类易于理解和扩展的方式**重构代码**，并避免重写或重新设计。
- **Fenic 连接至 Hugging Face Datasets**：开源项目 **Fenic** 现在直接集成了 **Hugging Face Datasets**，允许用户直接从 Hub 注入版本上下文，并安全地为 **Agent** 提供工具支持，详见[文档](https://huggingface.co/docs/hub/datasets-fenic)。
   - **Fenic** 类似于用于计算的 **e2b**，支持**数据快照**、**Agent** 上下文创建，并通过类似于 **pandas** 的 **dataframe API** 暴露 **MCP tools**，其 [Fenic 仓库已在 GitHub 上线](https://github.com/typedef-ai/fenic)。
- **多模型协作提升请求质量**：**多模型协作**的评估正在进行中，重点是降低单用户每次请求的幻觉率并提高整体请求质量；更多详情请见[此博客](https://facilitair.ai)。
   - 通过**顺序协作**已取得成果，目前有两个使用协作的开源仓库至少已达到 v1 版本。
- **Databomz 工作区正式发布**：一位成员介绍了 **Databomz**，这是一个用于保存、组织和共享提示词的工作区和 Chrome 扩展程序，具有标签、版本和文件夹等功能，更多信息请访问 [www.databomz.com](http://www.databomz.com/)。
   - **Forever Free 计划**包含大部分核心功能，创作者正在寻求活跃提示词用户的反馈，也可在 [github.com/Lnrchaos/NeSy-CML](https://github.com/Lnrchaos/NeSy-CML) 查看。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Firebase Studio 获得 Lovable 全栈升级**：Logan Kilpatrick 确认 [Firebase Studio 集成](https://ai.studio/build) 将引入新的 **AI Studio**，支持 vibe-coding 应用并整合其语音转文字功能，但目前使用的是 **Gemini 2.5 Pro** 模型。
   - 用户请求更简单的数据库、身份验证（auth）和存储集成，Logan 邀请针对 Chrome 扩展和 SolidJS 等尚未支持的用例提供反馈。
- **Lovable 发布 AI Shopify 全店集成**：Lovable 宣布了 **Shopify 集成**，允许用户通过简单的提示词（prompts）快速生成完整的在线商店，并演示了启动其自身周边商店（[lovable.dev/merch](https://lovable.dev/merch)）的流程。
   - 该功能向所有人开放，并附带 **30 天 Shopify 试用期**，不过目前尚不支持导入或修改现有的 Shopify 商店。
- **Project Mercury 聘请投资银行家训练 AI**：[Project Mercury](https://www.entrepreneur.com/business-news/openai-is-paying-ex-investment-bankers-to-train-its-ai/498585) 正以 **每小时 150 美元** 的薪酬聘请承包商将金融模型输入 **OpenAI**，以扩展 AI 在金融和技术等商业领域的实际应用。
   - 该项目展示了在特定专业领域训练 AI 的持续需求。
- **GoogleQuantumAI 运行《孤岛危机》（Crysis）**：**GoogleAI** 宣布了一个新的里程碑，利用 **65-qubit Willow 芯片** 和 **Quantum Echoes (OTOC) 算法** 运行一项可验证任务，速度比顶级超级计算机快 **13,000 倍**（[X 平台帖子](https://x.com/googleai/status/1981022228801307035)）。
   - 团队讨论了其对密码学（SHA-256 安全性）、结果可验证性、药物研发和气候建模的现实时间表，甚至运行《孤岛危机》/《毁灭战士》（Crysis/Doom）的影响。
- **Next.js 对框架进行“考试”**：Guillermo Rauch 宣布了 [Next.js Evals](https://xcancel.com/rauchg/status/1981037270624076092)，这是一套开源“考试”，让任何 LLM/Agent 证明其能够正确使用 **Next.js** 和其他支持的框架进行构建。
   - **GPT-5-codex** 和 **Claude Sonnet 4.5** 等模型目前的得分在 40% 左右；社区要求增加实际任务、公开追踪（traces）和成本列。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **诈骗警报震惊 Manus 社区**：一名用户被指控为“诈骗犯”，据称其索要付费 **Manus** 账号的登录权限，引发了关于账号安全的讨论。
   - 被指控的用户声称已找到“另一种方式”解决问题，而其他用户则警告不要分享账号凭据，以防个人和银行信息被盗。
- **Manus 额度系统压力**：用户对 **Pro Plan** 额度系统表示困惑和沮丧，一些人因为之前承诺无限额度的帮助页面现已消失而感到被“诱导转向”（bait and switched）。
   - 一些用户愿意为无限计划支付每月 200 美元以上的费用，而另一些用户则指出需要不断的额度管理，并参与改进计划以赚取免费额度。
- **账号封禁困扰新用户**：一名用户报告称，他和女友的账号在输入银行卡信息后不久就被封禁，怀疑是因为邀请了过多员工触发了 **Manus** 的封禁机制。
   - 尽管账号被封，他们仍寻求支持以保留账号中已经创建的项目。
- **用户建议 Manus 探索本地算力利用**：一名用户建议利用本地计算资源（local compute）来增强 **Manus** 的能力。
   - 这将支持构建大型原生应用、在本地处理海量数据集、运行资源密集型 AI 模型，并利用本地机器性能实现更快的构建。

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Comet 邀请函引发辩论**：成员们就 [Perplexity Comet](https://www.perplexity.ai/) 的免费 **1 年 Pro 邀请函**是否有价值展开了辩论。
   - 讨论质疑了 **Perplexity Comet Pro 版本**目前的收益和功能。
- **Pacific-Prime 吹捧环保架构**：来自法国的 Boris 介绍了 [Pacific-Prime 架构](https://huggingface.co/Pacific-Prime/pacific-prime)，声称其 **25 层**且每层 **5 次迭代**可以在无误差的情况下收敛。
   - 该架构可以在 **CUDA** 上运行，拥有 **1B** 参数且仅需 **6GB VRAM**，据称比 *llamacasual 架构环保两倍*。
- **“Regular” 角色仍在发放**：成员们确认 “Regular” 角色仍根据具体情况发放。
   - 该角色授予那些*一直为社区做出贡献*的人。
- **寻找带有 System Messages 的数据集**：一位成员请求推荐**带有 System Messages 的聊天数据集**，特别是寻找近期的资源。
   - 该用户表示很难找到此类最新的数据集。
- **Claude 展现出“变形”能力！**：一位成员在 X 上分享了关于 [Claude 变形 (shapeshifting)](https://fxtwitter.com/wesg52/status/1980680563582538099) 的帖子。
   - 未讨论进一步细节。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **使用灵活 Schema 编码 Tool Call 属性**：工程师们正寻求在 `inputSchema` 中编码一个属性，该属性在 Tool Call 中既可以是 **array** 也可以是 **object**，因为所使用的 **JSONSchema** 子集不支持 `oneOf`。
   - 挑战在于定义一个 LLM 可以推理且客户端可以接受的灵活 Schema，尽管受限于 **JSONSchema** 子集的限制，这使得工具属性的 Schema 定义变得复杂。
- **应对 Tool Call 中 JSONSchema 子集的约束**：**JSONSchema** 的一个子集限制了 Tool Call 的 `inputSchema` 定义，省略了如 `oneOf` 等允许属性具有多种有效 Schema 类型的功能。
   - 当属性可以是数组或对象时，这种限制带来了挑战，影响了为 Tool Call 指定灵活 Schema 的能力。
- **包含大量工具描述的 MCP Server 触及上下文限制**：工程师报告称，拥有 **60 个工具**的 MCP Server 很快就会达到上下文限制（即使是最高级方案），这是由于工具描述的大小导致的。
   - 客户并不青睐将工具拆分到多个 Server 中，这导致了对高效上下文管理和成本优化方案的探索，因为在 Multi-agent 实现中，每次操作的成本高达 **$7-8**。
- **简化工具工作流以优化上下文和准确性**：一位工程师通过一个由 **3 个工具**（涉及操作列表、描述和调用）组成的精简工作流，在 Server 指令的引导下管理了 **50 多个 CLI 操作**。
   - 为大量工具使用描述和输入/输出 Schema 可能会使模型不堪重负并超出上下文限制，因此需要精简工作流来提高效率和准确性。
- **开发自定义聊天以降低成本**：为了降低成本，一位工程师在注意到运行 Claude 桌面版或 ChatGPT 会迅速累积费用后，通过直接 API 连接和优化的模型选择创建了自定义聊天。
   - 具有模型切换功能的 Multi-agent MCP 客户端每次操作会产生 **$7-8** 的成本，导致他们放弃了该设置。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **NPC 语音通过 DSPy-ElevenLabs 合成技术萌芽**：一名成员开发了一个用于游戏 **NPC** 的语音生成系统，利用 **DSPy** 解析 wiki 内容，并集成 **ElevenLabs** 进行语音合成，详见 [该项目的 GitHub 仓库](https://github.com/Gielinor-Speaks/voiceover-mage) 和 [这段开发日志视频](https://youtu.be/z3DQm2PiKpo)。
   - 开发者旨在通过 **自动化评审循环（automated judging loop）** 来自动化语音筛选，使用人工选择作为训练信号，以提高生成语音的质量。
- **微软的 Trace 程序胜过 DSPy？**：根据一名成员分享的截图，微软的 [Trace](https://microsoft.github.io/Trace/) 程序声称比同等的 **DSPy** 程序 **准确率提升了 8%**。
   - 尽管有此说法，一些成员仍计划测试 **Trace** 以进行公平比较，并期望在使用 **DSPy** 时能保留更多细粒度的控制。
- **异步 DSPy 解决 OCR 障碍**：一名成员探索了用于并行执行的 **异步 DSPy 模块**，专门用于涉及 **Google Cloud Vision**、**DSPy Attachments 库** 以及通过 bboxes 进行布局的高吞吐量 OCR 任务。
   - 挑战包括重复循环的 bug，导致其开始探索 **paddleocr-vl** 或 **nanonets** 等替代方案以提升性能。
- **AI 与区块链工程师展示专业实力**：一名成员介绍自己为 **高级 AI 与区块链工程师**，专注于 AI 与区块链交叉领域的自主系统，拥有在 **Base**、**Solana** 和 **EVM chains** 上的经验。
   - 他们的专长在于 **链上 AI Agent**、**带有实时数据的 LLM 流水线**，以及使用 **LangChain/AutoGen** 进行 **AI Agent 编排**。
- **可训练装饰器（Trainable Decorator）引起兴趣**：一位成员表达了对可训练装饰器的欣赏。
   - 目前尚不清楚是指哪个可训练装饰器，但他们似乎对此非常兴奋，简单地评价道：*噢，我喜欢这个可训练装饰器！非常棒的主意*。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Sora 生成的视频大放异彩**：一名成员分享了一段由 **Sora** 生成的视频，展示了该工具的能力并让人们一窥其潜力，视频展示在 [这里](https://cdn.discordapp.com/attachments/1149866623109439599/1430403237084659774/20251022_0850_01k850gmktfant3bs18n3hbd79.mp4?ex=68fa4f13&is=68f8fd93&hm=ac5d3464bd020d568c770a6fa89ef8ccb2db40420fd45bee25dc3be64c60807f&)。
   - 视频的发布引发了关于 **Sora** 带来的真实感和艺术可能性的热烈讨论。
- **请求 Nous Research 智囊团协助**：一名成员请求 **Nous Research** 协助完成一篇旨在帮助对 **AI** 感兴趣的青少年的研究论文。
   - 这一请求突显了社区对 AI 教育倡议的兴趣。
- **GPT-3.5 Sonnet 迎来终曲**：一名成员分享了来自 fixvx.com 的链接（[点击此处](https://fixvx.com/dbreunig/status/1980733694634770710)），表达了对 **GPT-3.5 Sonnet** 终结的告别感。
   - 该帖子引发了对该模型相对于其他模型的优缺点的反思。
- **GPT-5 进行了态度调整**：根据最新分享的基准测试结果，**GPT-5** 变得不那么友好了，这可能是由于目前关于 AI 模型中 **sycophancy（谄媚/迎合）** 现象的持续讨论。
   - 这一转变引发了成员们关于“帮助性”与“客观性能”之间权衡的辩论。
- **GPT-4o 依然能模仿口音**：一位用户注意到 **GPT-4o** 在语音模式下仍能表演牙买加口音，相比之下，**GPT-5** 声称可以做到但随后失败了。
   - 该用户指出，这是他们关心的 *少数几个指标之一*。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi 用户发现没有支持渠道**：一位用户报告称，他们获得的 **Kimi** 支持为 *零*，且未收到支持团队的任何回复。他们被告知这不是一个支持服务器。
   - 一位社区成员告诉他们私信（DM）特定用户以寻求帮助。
- **出现关于 Moderato 和 Allegretto 付费方案的问题**：一位用户询问有关升级到 **Moderato** 或 **Allegretto** 的事宜，以及在付费方案下可以使用 **OK Computer** 的次数。
   - 另一位用户澄清说 **Moderato** 每月允许使用 **20** 次，并链接到了 [相关的推文](https://x.com/togethercompute/status/1980337943169651055)。
- **K2 在 Together 上速度极快**：一位用户评论了 **K2** 在 **Together** 平台上使用时令人印象深刻的速度。
   - 讨论中未提供更多细节。
- **探讨与 Kimi 的合作机会**：一位用户询问关于联系谁来洽谈与 **Kimi** 的合作机会。
   - 另一位用户建议私信特定个人以跟进潜在的合作。

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 缺失的关键特性**：成员们讨论了 **Mojo** 目前缺失的最重要的东西：一个*完善的类型系统*，此外还有一系列其他特性的愿望清单。
   - 其他期望的特性包括完善 **standard library datatypes**（标准库数据类型）、**proper IO**、一个优秀的 **async runtime**、**effect system**、**static reflection**、**compiler plugins**、处理更多 **restrictive targets**（受限目标）的能力、**cluster compute**、**device/cluster modeling**，以及 **Erlang's OTP** 的某种克隆版。
- **Llama-3.1 在 H100 上遇到障碍**：一位成员报告称，在 **H100** GPU 上使用 nightly 版本加载 `modularai/Llama-3.1-8B-Instruct-GGUF` 时出现 segfault（段错误），这与 *llama3/model.py* 中的 `session.load` 函数有关。
   - 该问题似乎是 GPU 特有的，因为该模型在 CPU 上运行正常，特别是在使用 `modular/max-nvidia-full:nightly` 基础镜像时，但在推送到 GPU 时会发生 segfault。
- **GGUF 模型引发段错误排查**：一位成员推测，segfault 可能会在 **bfloat16** 模型中持续存在，因为 **GGUF**（*q4_k* 等）权重的反量化 kernels 到目前为止仅针对 CPU 执行进行了构建。
   - 该成员确认，虽然 CPU 运行的是 **q4_k** 版本，但 GPU 尝试运行 **bf16** 版本，从而导致了 segfault。
- **tensor_internal 变为 tensor**：成员们宣布 `tensor_internal` 包已重命名为 `tensor`（[Discord 链接](https://discord.com/channels/1087530497313357884/1224434323193594059/1430580184305635339)）。
   - 这是 **Mojo API** 最新 nightly 版本中的一项关键更新。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Vulkan Sine 函数的问题**：**Vulkan sine 函数的不准确性**导致了错误，一位成员报告由于此问题导致*几个错误检查失败*。
   - 这需要自定义 sine 函数，但这可能会减慢处理速度。
- **需要重构渲染器**：如果你的渲染器没有实现 **sin/log/exp**，**tinygrad/uop/decompositions.py** 会自动触发。
   - 一位成员之前没意识到会发生这种情况，并据此修改了他们的渲染器，以确保所有 **transcendental functions**（超越函数）都能通过。
- **JITBEAM 的 Jitted 珍宝**：澄清：**JITBEAM** 特指 jitted kernels 的 **BEAM**。
   - JITBEAM 代表了在 tinygrad 框架内专门为 jitted kernels 量身定制的 BEAM。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 被误用作基准测试工具**：一位成员承认*从未将 Aider 用于其原始用途，仅用于基准测试*，更倾向于使用 **Cline** 和 **RooCode** 等本地模型。
   - 这表明一些用户正在发现 **Aider** 在其预期设计之外的替代用途。
- **LLM 历史记录检索证明很困难**：一位用户寻求*评估日志和 LLM 历史记录*（完整聊天记录），但另一位成员确认他们最近在检索这些日志方面也*运气不佳*。
   - 缺乏可访问的日志可能会阻碍 **LLM** 交互的调试和分析工作。
- **分享了可疑的 Uniswap 门户链接**：一位用户分享了一个与 **Uniswap** 相关的可疑链接（[uniswap-portal.web.app](https://uniswap-portal.web.app)）。
   - 与该链接关联的图片附件被图像分析工具标记为 **spam**（垃圾信息），表明存在潜在的安全风险。
- **Aider 的项目维护受到质疑**：鉴于自 **8 月 10 日**以来没有版本更新且活动有限，一位成员质疑该项目的维护状态。
   - 这一询问引发了对该项目持续支持和开发的担忧，尽管目前尚未给出官方答复。
- **Chutes AI 廉价 LLM 模型列表上线**：一位成员分享了一个指向廉价 LLM 的 **Chutes AI 模型**列表的[链接](https://wuu73.org/r/chutes-models/)，认为这对社区有潜在用处。
   - 随着社区对模型列表贡献的增加，预计在不久的将来 **Chutes AI** 的使用案例将会增加。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**Windsurf Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

您收到这封邮件是因为您通过我们的网站选择了订阅。

想更改接收这些邮件的方式吗？
您可以从该列表中[取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord：各频道详细摘要与链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1430269580735152391)** (1269 条消息🔥🔥🔥): 

> `推荐计划问题、Comet 浏览器、Perplexity Pro、Gemini 3 发布日期、Perplexity 技术问题` 


- ****推荐计划公投：可疑的判定引发激烈讨论****：用户讨论了 Perplexity AI 推荐计划的问题，包括线索（leads）未被计入，以及系统可能将合法的推荐标记为欺诈，导致审核后支付额为 **0 派尼（penny）**；许多人推测“优质线索”的标准，以及使用 VPN 和虚拟机获取更多推荐的策略，部分用户表示成功，另一些则表示沮丧。
   - 一些用户还报告了 Perplexity Pro 额度限制的问题，部分用户抱怨即使朋友通过他们的链接下载并使用 Comet，他们也没有获得线索奖励。
- ****Comet 热潮：用户争论浏览器福利还是套路？****：用户分享了 **Comet 浏览器**的推荐链接，一些人提供一个月的 Pro 会员以换取下载量，他们遇到了 Comet 浏览器重置账户、遇到 Cloudflare 检查以及推荐难以计入等问题。
   - 一些用户推测旧推荐计划的价值已经降低。
- ****Pro 纠纷：无限访问还是虚假宣传？****：用户讨论了对 Perplexity Pro “无限访问”的感知限制，一名用户上传了一张显示使用限制的图片，引发了关于订阅条款和限制可能更新的讨论。
   - 曾有一个奖励系统，成员可以通过提供线索和报告 Bug 获得报酬。
- ****Gemini 焦虑：Google 的微光还是矩阵中的故障？****：关于 **Gemini 3** 的发布日期出现了推测，提到了可能在 12 月发布，并讨论了西方 AI 对其产品施加的限制可能为中国在新品上取得领先地位铺平道路。
   - 还讨论了东方是否会接管软件和 AI 领域，或者西方是否会继续保持领先。
- ****Android 异常：UI 回退令用户不满****：用户报告了 Perplexity 的技术问题，包括 **Android 应用 UI 回退到旧版本**，以及移动应用、Comet 浏览器和网页版 Perplexity（特别是在生成文件时）持续出现“something went wrong”错误。
   - 一名用户报告应用在编写代码时崩溃，其他人建议使用网页版作为替代方案。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1430299672295247944)** (6 条消息): 

> `可分享线程、基于时间的研究员、特朗普的谎言、信息战、数学艺术` 


- ****可分享线程**提醒**：提醒一名成员确保其线程已设置为**可分享（Shareable）**。
   - 附带了一个附件作为如何设置线程为可分享的可视化指南。
- **发表关于数学艺术的内容**：一名成员分享了[他们发表的关于艺术家 0thernes 数学艺术的文章链接](https://generativeai.pub/the-math-art-of-artist-0thernes-not-the-typical-96e009060bc1)并征求反馈。
   - 文章讨论了该艺术家融合数学与艺术的独特方法。
- **Perplexity 页面链接**：一名成员分享了几个 Perplexity AI 页面的链接，包括 [Reconciling Atomic Curvature](https://www.perplexity.ai/page/reconciling-atomic-curvature-a-27py5fYSRiyMr.91kDpiDAPerplexity)、[Time-Based Researcher](https://www.perplexity.ai/search/time-base-researcher-LxNZL3iFRXamL0kYZV3RiA#0) 以及 [Trump's Lies and Fact Denials](https://www.perplexity.ai/page/trump-s-lies-and-fact-denials-cgZjiGKUTvuj6sTWzTkm0A)。
   - 另一个分享的 Perplexity AI 页面描述了[修辞轨迹和复杂信息战的逻辑终点](https://www.perplexity.ai/page/-g5fwV1GzRlqm_q32kk3y9g)。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1430316660950302812)** (4 条消息): 

> `Perplexity API, ChatGPT5, Claude, Sonar` 


- **Perplexity API 仅支持 Sonar**：一名成员询问 **Perplexity API** 是否可以访问 **ChatGPT5** 和 **Claude**，还是只能访问 **Sonar**。
   - 另一名成员回答说 API 仅限于 **Sonar**。
- **用户表达对 API 的兴奋**：一名用户表达了兴奋，但未说明原因。
   - 该用户写道 *idk why , hehehehe*。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1430269427261116426)** (1034 messages🔥🔥🔥): 

> `Gemini 3 发布日期推测，使用 Veo 3 进行图生视频，Gemini 3 性能基准测试，OpenAI LibGen 欺诈指控，Lithiumflow 与 orionmist` 


- **Gemini 3 发布预期升温后破灭**：成员们推测了 **Gemini 3** 的发布时间，有人建议在 10 月 25 日发布，但其他人表示这不太可能，11 月是一个更现实的时间框架，这取决于 AI 潜在的新版本命名系统 [这可能并不准确](https://ai.google.dev/)。
   - 最终据报道，实际发布时间可能要晚得多，建议在 12 月进行预览，并在 1 月正式发布，这打破了许多人对 2025 年发布的幻想。
- **使用 Veo 3 进行图生视频**：成员们讨论了使用 **Veo 3** 生成视频，但据说在生成视频时无法选择模型，一位用户表示他们生成的视频没有声音。
   - 还有说法称视频中生成的音频似乎是随机选择的。此外，一些人发现 *Ling-1t* 非常有趣。
- **Gemini 3 表现出性能提升，但也伴随着失败**：成员们在多个测试用例上实验了 **Gemini 3** 的代码生成能力，包括体素艺术（voxel art）、风洞、魔方求解器以及其他编程问题和非 Web GUI 库。
   - 有建议认为，虽然它令人印象深刻，但 **Lithiumflow** 的代码（据称是 **Gemini 3 Pro**）在某些情况下无法运行，有报告称 *Lithiumflow 在更具体的 Prompt 下失败了*，但它通常优于 **Sonnet 4.5** 和 *OpenAI* 模型等其他选项。
- **OpenAI 在数学发现中被指控欺诈**：针对 **OpenAI** 的欺诈指控被提出，称其只是在重复现有研究中的数据。
   - 一位用户表示：*到目前为止，尽管 YouTube 上有各种阴谋论，但还没有 AI 做出任何形式的“发现”，因此 OpenAI 将作为现代历史上最大的骗子载入史册*。
- **Lithiumflow 和 Orionmist 是 Gemini 3 的不同版本**：一位用户提出 **Lithiumflow** 和 **Orionmist** 是同一个模型，但前者可以访问 Google Search，但这一说法尚未得到证实，且有人持不同意见。
   - 一位用户建议 **Gemini 2.5 Pro** 可能更优，此外还有人指出来自字节跳动（Bytedance）名为 **Seed** 的模型具有非常出色的图像理解能力。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1430270761175744582)** (540 messages🔥🔥🔥): 

> `Claude 4 Sonnet 成本分析，Cursor 新的 "apply" 问题，Cursor IDE 安全漏洞，2024 年 Cursor 大停机，额度耗尽后的 Cursor 免费模型` 


- **Claude 4 Sonnet 的昂贵代价**：成员们注意到使用 **Claude 4 Sonnet** 可能会变得非常昂贵，一位用户提到他们在 **max mode** 下为**单次请求支付了 7 美元**，尽管请求次数更多，但总体成本仍然更高。
- **Cursor 新的 "apply" 问题出现**：最新的更新导致 Agent 告诉用户它们处于询问模式，在聊天时禁用了 "apply" 功能；用户现在可以在聊天提供代码后直接在聊天窗口中应用代码。
   - 一位成员开玩笑说 *apply 工具大部分时间都会失败*，所以他们会查看完整输出并选择 *show code*。
- **Cursor 和 Windsurf IDE 因旧版 Chromium 受到威胁**：一篇 [BleepingComputer 文章](https://www.bleepingcomputer.com/news/security/cursor-windsurf-ides-riddled-with-94-plus-n-day-chromium-vulnerabilities/) 证实，由于 **Chromium 和 V8 引擎**版本过旧，**Cursor 和 Windsurf IDE** 存在超过 **94 个 n-day 安全漏洞**，带来了拒绝服务或远程代码执行等风险。
- **2024 年 Cursor 大停机**：用户报告了 **Cursor** 的广泛问题，包括连接失败和无法发送消息。
   - [Cursor 状态页面](https://status.cursor.com/) 确认了此次**停机**，一位用户幽默地哀叹他们的 *threejs 游戏陷入了停滞*。
- **付费额度耗尽后的 Cursor 免费模型使用**：用户讨论了在 **20 美元标准套餐**的 **Claude Sonnet** 额度用完后会发生什么，确认系统会切换到**免费的 Haiku 模型**。
   - 一位用户表示在*极端贫困*时仍倾向于使用第三方 API Key。


  

---

### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1430560153760960512)** (2 条消息): 

> `Cursor + Linear 集成，UI 错误信息` 


- **Cursor-Linear 集成间歇性工作**：一位用户报告说，**Cursor 与 Linear 的集成**有时在重试后可以工作，即使没有进行任何明显的更改。
   - 用户尝试在 **Linear** 中 ping **Cursor**，重试后集成恢复正常。
- **UI 错误信息具有误导性**：一位用户建议在本地分支未推送到远程仓库时更新 **UI 错误信息**。
   - 用户建议将错误信息替换为：*"The branch {branch name} does not exist in the remote repository."*（分支 {branch name} 在远程仓库中不存在），因为他们之前误以为是环境配置出了问题。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1430270902234255541)** (175 条消息🔥🔥): 

> `Unsloth SLA 成本，IPv4 扫描，用于分析加密消息的 LLM，与 Pytorch 合作的量化感知训练 (QAT)，连贯性奖励函数` 


- **Unsloth SLA 成本暂无 (N/A)**：一位用户询问了 **Unsloth SLA** 的费用，但回复显示目前为 **N/A**（暂无）。
   - 目前没有关于具体定价或服务水平协议的进一步讨论。
- **成员讨论 IPv4 互联网扫描的风险**：一位成员提到了扫描整个 **IPv4 互联网**，但警告说可能会被**国际蜜罐 (honeypots)** 举报，并需要避开某些网络以尽量减少滥用报告带来的问题。
   - 其他成员开玩笑地建议使用僵尸网络进行扫描，但也承认扫描互联网会让*每个人的处境都变得更糟*。
- **LLM 分析加密消息？没那么快！**：一位用户想使用小型 LLM 来对加密消息和二进制/十六进制代码进行分类，但另一位成员指出，良好的加密会将数据变成**纯噪声**，从而使分类变得极其困难。
   - 一位成员开玩笑说，*要使用机器学习分析现代加密密码以寻找模式，你可能需要动用整个地球未来一百万年以上的 GPU 资源*。
- **与 Pytorch 合作的量化感知训练 (QAT) Colab**：推出了一个新的与 **Pytorch** 合作的 **量化感知训练 (QAT)** Colab，这是公告的 [链接](https://x.com/UnslothAI/status/1981021761782317368)。
   - 一位成员询问该合作是否保持视觉编码器（vision encoder）不变，并引用了 [这篇论文](https://www.arxiv.org/abs/2509.11986) 和 [这条 X 帖子](https://x.com/SravanthiSinha/status/1980867770498838560?)。
- **连贯性奖励函数的难题**：一位成员询问如何创建一个奖励函数来在 RL（强化学习）期间判断文本的连贯性，建议使用 LLM 对连贯性进行 1 到 5 分的评分。
   - 另一位成员建议使用 **Flesch-Kincaid 可读性测试** 或微调 **ModernBert/Colbert**，并指出使用 LLM 成本太高。此外还提供了一个 **language_tool_python** 的 [链接](https://github.com/jxmorris12/language_tool_python)。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1430272012256940153)** (4 条消息): 

> `自我介绍，频道指南` 


- **新频道自我介绍**：一位名为 motithewizerd_16739 的成员在频道中打招呼说 *"Hi all new here. Great stuff"*。
   - 另一位成员 theyruinedelise 对 motithewizerd_16739 表示欢迎。
- **发布频道指南**：Unsloth AI 欢迎大家来到新频道，并提醒大家遵守指南。
   - 即：由于这是一个介绍频道，请不要发布推广内容。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1430282757690495177)** (207 messages🔥🔥): 

> `DGX Spark vs RTX 6000, OSS VSCode Fork for Local LLM Usage, Training completion timing, Token Initialization Importance` 


- **DGX Spark vs RTX 6000 Pro 之争**：成员们讨论了是购买一台拥有 **200 GB VRAM**、售价 **$4k** 的 **DGX Spark**，还是在 eBay 上购买 **$3500-4k** 的 **RTX 6000 Pro**。
   - 配备 **8x H100s** 的 **DGX Spark** 被吹捧为能够实时进行 **2-4B 模型** 的全量 SFT，有人开玩笑说这*肯定是个骗局*。
- **用于本地 LLM 的开源 VSCode**：一名成员请求创建一个 **VSCode** 的开源分支或任何围绕本地 LLM 使用构建的 IDE，并具备良好的 FIM 和聊天功能。
   - 他们正在寻找一种*免费且不会把我的数据发给 Sam Altman* 的工具，但又对需要*浪费 3 个月时间自己造轮子*感到绝望。
- **Pod 空转成本焦虑**：一名成员表达了对训练在不方便的时间结束导致 Pod 处于闲置状态并产生不必要运行成本的挫败感。
   - 另一名成员表示，他们会*调整 num of steps 以匹配自己的作息*，从而避免 Pod 的运行成本。
- **Token 初始化**：一名成员强调了良好的 **Token 初始化** 的重要性，尤其是当不打算投入很长的一个阶段来训练它们时。
   - 他们还链接了一个关于音频编码的*有趣的小讨论* [Karpathy Tweet](https://x.com/karpathy/status/1980397031542989305)。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1430281447347982460)** (63 messages🔥🔥): 

> `Unsloth with custom TRL Trainer, Qwen3-30B-A3B-Instruct-2507 KeyError, Fine-tuning Vision Models w/ Unsloth, Llama 3.1 Data Augmentation, FastLanguageModel Crash` 


- **自定义 TRL Trainer 的梯度故障？**：一名博士生在使用带有 Unsloth 的自定义 TRL Trainer 变体时遇到了梯度传播问题，特别是在修改了 `generation_and_score` 和 `compute_loss` 函数后。
   - 他们正在寻求任何在 Unsloth 中有自定义 TRL Trainer 实现经验的人的建议。
- **Qwen3 的快速下载失败了！**：一名用户在使用 Unsloth 的 `FastModel.from_pretrained` 下载 `unsloth/Qwen3-30B-A3B-Instruct-2507` 模型时遇到了 `KeyError`。
   - 从回溯信息来看，错误发生在尝试获取 16 个文件之后。
- **不使用聊天模板的视觉模型微调？**：一名用户询问是否可以在不使用聊天模板或用户-助手对数据集的情况下，使用 Unsloth 微调视觉模型，旨在采用更接近文本补全风格的微调方法。
   - 该用户没有得到明确的答复，但正在寻找以类似于文本补全任务的方式进行视觉模型微调的方法。
- **多个 QA 版本：数据增强还是过拟合？**：一名正在训练 Llama 3.1 Instruct 版本的用户想知道，在他们的 Q/A 数据集中包含同一问题的多个版本是否是数据增强的好主意。
   - 一名成员表示，*与简单地对相同数据进行多个 epoch 的训练相比，这有助于减少过拟合*。
- **FastLanguageModel 在并行处理时受挫！**：一名用户报告称 `FastLanguageModel` 在并行请求期间崩溃，而 `FastVisionModel` 功能正常。
   - 该用户正在寻求帮助以解决此语言模型的问题。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1430351853694226583)** (4 messages): 

> `External GPUs on ARM MacBooks, Thunderbolt 5` 


- **Nvidia GPU 现在可以在 ARM MacBook 上运行**：据 [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/tiny-corp-successfully-runs-an-nvidia-gpu-on-arm-macbook-through-usb4-using-an-external-gpu-docking-station) 报道，TinyCorp 成功通过 **USB4** 使用外置 GPU 扩展坞在 **ARM MacBook** 上运行了 **Nvidia GPU**。
- **Thunderbolt 5 端口登陆 Pro 机型**：Mac 'Pro' 机型现在配备了 **Thunderbolt 5**。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1430362843701706793)** (1 messages): 

> `Meta, WhatsApp, ChatGPT, Policies` 


- **Meta 修改政策，破坏了派对**：Meta 修改了其政策，因此 **1-800-ChatGPT** 在 **2026 年 1 月 15 日**之后将无法在 **WhatsApp** 上运行。
- **OpenAI 为 ChatGPT 成瘾者提供救生索**：幸运的是，我们有应用程序、网站和浏览器，你可以改用它们来访问 **ChatGPT**。
   - 更多信息可以在 [chatgpt-whatsapp-transition 页面](https://openai.com/index/chatgpt-whatsapp-transition/)找到。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1430270607379140659)** (345 messages🔥🔥): 

> `Sora 2 视频生成限制, Sora 2 在英国的可用性, AI 动画化小猪佩奇, OpenAI 账户每日扣费, Sora 2 中受版权保护内容的接受与拒绝情况` 


- **Sora 2 视频生成达到每日上限**：用户报告 Sora 2 将视频生成限制为**每天 30 个视频**，导致服务很快耗尽，并引发了关于使用 VPN 绕过限制的讨论。
   - 一些用户开玩笑说限制*太低了*，其中一人表示*只需 1-2 小时我的 Sora 就用完了 xD*。
- **英国用户渴望通过 VPN 访问 Sora 2**：用户询问了 **Sora 2 在英国的可用性**，并建议使用 VPN，尽管据报道该服务目前仅限美国。
   - 一位用户分享了一条 [推文](https://x.com/hamdfakhr8/status/1980693381585285182?t=zxNAhQv-3PuBDxGIDQpeTQ&s=19)，暗示 Sora 2 仅在美国可用。
- **AI 版《小猪佩奇》动画集即将到来？**：一位用户寻求推荐 **AI 程序**来制作 10 分钟一集的**《小猪佩奇》**动画，并为角色生成 AI 语音，并链接到了 [Opus Pro agent](https://www.opus.pro/agent?ref_id=5XHUEZ3WP)。
   - 讨论中未提供具体的程序推荐。
- **神秘的每日 15 美元 OpenAI 扣费困扰用户**：一位用户报告称，自 10 月 9 日以来，其 OpenAI 账户每天都有 **15 美元的固定扣费**，即使在删除了所有 API keys 后依然如此，并提供了 [截图](https://cdn.discordapp.com/attachments/998381918976479273/1430552464108556308/Captura_de_Tela_2025-10-22_as_10.43.57.png?ex=68fa314d&is=68f8dfcd&hm=ac45c8620b631401aa0bc85ff59c1098a590e604ce27d207135a45798d95ad4c&)。
   - 该用户正在寻求帮助以确定这些持续扣费的原因。
- **Sora 2 的版权处理让《蔬菜总动员》中招**：用户讨论了 **Sora 2 对受版权保护材料的处理方式**，注意到存在不一致性：某些涉及 *VeggieTales* 等 IP 的提示词被接受，而其他的则被拦截。
   - 一位成员表示，*通常情况下即使检查环节恰好漏掉了，你也不被允许侵犯版权*。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1430287585472938024)** (27 messages🔥): 

> `LLM OS, ChatGPT 延迟, Gemini, Sora, Realtime API` 


- **LLM 将驱动未来的操作系统？**：一位成员推测，如果 **AI** 继续进步，我们可能会看到完全由 **LLM** 辅助驱动的**操作系统**。
   - 这可能涉及 AI 以比当前操作系统设计更动态的方式处理系统进程和用户交互。
- **自定义 RPG GPT 对话延迟解决方案**：一位用户报告称，在浏览器中与自定义 **RPG GPT** 进行长时间对话会出现延迟和卡死，但在移动端 App 中运行正常。
   - 另一位用户建议使用 **scrcpy** 作为潜在的变通方案，并认为这可能是**硬件问题**。
- **懒加载（Lazy Loading）实现**：一位成员提到 **Gemini** 通过实现*懒加载*修复了延迟问题，即在用户滚动时才加载消息。
   - 另一位成员分享了一个 **ChatGPT Lightsession** 扩展程序，可能有助于改善体验。
- **Sora 2 故事模式难题**：一位用户询问 **Sora 2** 的**故事模式**在哪里。
   - 另一位用户尝试解释在哪里可以找到**分镜脚本生成（storyboard generation）**的**编辑选项**，但承认在此期间失去了 **Sora 2** 的访问权限，描述为*进入你在 Sora 2 的个人资料，生成你的第一个视频，点击它，编辑选项应该会出现在某个地方*。
- **质疑 Realtime API 的新颖性**：一位成员质疑 **GPT-realtime 模型**有什么新奇之处，因为 **Realtime API** 在一年多前就已经发布了，并链接到了 [Introducing the Realtime API](https://openai.com/index/introducing-the-realtime-api/)。
   - 讨论未给出明确答案，而是转向了其他话题。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1430271122355786001)** (15 条消息🔥): 

> `AI 生成内容的版权问题，针对受版权保护材料的 Prompt engineering，拼写错误对 AI 模型输出的影响，在 AI Prompt 中构建上下文，通过文本 Prompt 使用 AI 生成视频` 


- **避免 AI 生成内容的版权索赔**：成员们讨论了如何在 AI 生成内容时避免版权问题，一位用户建议通过描述受版权保护的角色（*穿着红蓝相间、带有黑色蜘蛛网符号服装的男子*）而不是直接命名该角色来规避问题。
   - 另一位成员警告说这可能还不够，并指出这是**版权 IP**，他们无法提供帮助。
- **Prompt 拼写错误影响 AI 输出质量**：一位用户询问 Prompt 中的拼写错误（例如将 *"create a hangman game"* 写成 *"crete a hagman gam"*）是否会对输出产生负面影响。
   - 一位成员表示，虽然简单的 Prompt 受影响较小，但**拼写错误可能会在复杂的 Prompt 中引起混淆**，尤其是在存在歧义或拼写错误改变了原意的情况下。
- **在 AI Prompt 中构建上下文和记忆**：成员们讨论了在 AI Prompt 中**构建上下文**的策略，重点在于*周密的层级结构和相关内容*。
   - 另一位成员指出，拼写错误会迫使模型进行猜测，除非通过额外的引导加强请求，否则可能导致输出更加模糊。
- **使用 AI 生成电影感视频预告片 - Gemini**：一位成员请求在 Prompt engineering 方面提供帮助，以使用 Gemini 生成 AI 电影感视频预告片。
   - 该用户提供了 [prompt](https://chatgpt.com/share/68f92672-a0f8-8011-a8a7-1d86973ff476) 和一段 [视频](https://cdn.discordapp.com/attachments/1046317269069864970/1430677767673876551/gemini_generated_video_D4CFBE60.mov?ex=68faa600&is=68f95480&hm=8172c60da7184e89c71e1c664a3e08ddee0bdda44c3f728669e041aefeed96e5&) 作为输出示例。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1430271122355786001)** (15 条消息🔥): 

> `版权侵权，规避版权，Prompt 中的拼写错误，构建上下文，忍者题材电影预告片` 


- **版权警察说“禁止蛛丝荡！”**：一位用户询问关于为 Sora AI v2 生成“终极蜘蛛侠在纽约市荡秋千”的内容，但被告知[生成受版权保护的 IP 是不允许的](https://community.openai.com/tos)。
   - 作为回应，一位成员建议使用描述性词汇，如*“穿着红蓝相间、带有黑色蜘蛛网符号服装的男子”*，以规避版权问题。
- **拼写错误会引发 AI 的胡言乱语吗？**：一位成员质疑 Prompt 中的拼写错误是否会对 AI 模型的输出产生负面影响，例如将 *"create a hangman game"* 写成 *"crete a hagman gam"*。
   - 有人建议，拼写错误会迫使模型进行猜测，可能影响输出质量并导致更模糊的回答，特别是当拼写错误引入歧义时。
- **上下文是关键吗？**：一位用户询问了[在 Prompt 中构建上下文的最佳实践](https://community.openai.com/tos)，以及记忆和上下文发挥重要作用的典型用例。
   - 回复强调了*周密的层级结构和相关内容*的重要性，以及根据个人偏好和特定工作流定制上下文的重要性。
- **精巧的忍者服务器需要视频**：一位用户分享了一个 [prompt 和 Gemini 生成的视频](https://cdn.discordapp.com/attachments/1046317269069864970/1430677767673876551/gemini_generated_video_D4CFBE60.mov?ex=68faa600&is=68f95480&hm=8172c60da7184e89c71e1c664a3e08ddee0bdda44c3f728669e041aefeed96e5)，用于发布一个受忍者和传说启发的角色扮演服务器的短篇电影感预告片。
   - 该用户寻求关于改进 Prompt 的建议，以更好地呈现竖屏预告片（9:16 格式，时长约 10 秒）的概念，并具有黑暗、神秘且充满情感的日系审美风格。


  

---

### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1430293802685562983)** (2 条消息): 

> `Andromeda-alpha, :exacto` 


- **全新隐身模型 "Andromeda-alpha" 发布**：一款专注于**图像和视觉理解**的新型小型推理模型 **Andromeda-alpha** 已发布，可通过 [OpenRouter.ai](https://openrouter.ai/openrouter/andromeda-alpha) 获取社区反馈。
   - 所有提示词和输出都将被记录以改进模型，该模型仅供试用，请勿上传任何个人、机密或敏感信息，且不适用于生产环境。
- **推出 `:exacto`：tool-calling 端点**：OpenRouter 推出了名为 `:exacto` 的新一类端点，通过将请求路由到在**结构化输出性能方面有显著提升**的提供商，从而实现**更高的 tool-calling 准确率**。详情记录在新的 [博客文章](https://openrouter.ai/announcements/provider-variance-introducing-exacto) 中。
   - 首批上线的模型包括 `moonshotai/kimi-k2-0905:exacto`、`deepseek/deepseek-v3.1-terminus:exacto`、`z-ai/glm-4.6:exacto`、`openai/gpt-oss-120b:exacto` 以及 `qwen/qwen3-coder:exacto`。内部和外部基准测试显示，其 **tool-call 成功率有实质性提升**。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1430340923639988436)** (1 条消息): 

> `AI Confidence Scores, Cost-Efficient AI, OpenRouter LLMs, AI Agents and Workflows` 


- **Objective-AI：AI 置信度分数解析**：[Objective-AI](https://objective-ai.io/) 的 CEO Ronald 介绍了一种针对符合 OpenAI 标准的补全选项的**置信度分数 (Confidence Score)**，该分数并非源自 AI 的直接评估，而是通过更智能的机制得出。
   - Objective-AI 利用 **AI 模型**的多样性来提供**透明的统计数据**。
- **利用 Objective-AI 实现高性价比 AI**：Objective-AI 的置信度分数系统增强了**成本效益**，允许用户在有效使用小型模型时最大化其效用。
   - Ronald 提到，如果使用得当，你可以从那些微型模型中获得巨大的价值。
- **Objective-AI 集成 OpenRouter**：Objective-AI 使用 **OpenRouter** 访问广泛的 **Large Language Models (LLMs)**，增强了平台的通用性和能力。
   - Ronald 表示，他们使用 **OpenRouter** 是为了获取其提供的各种 LLMs。
- **Objective-AI 构建免费 AI Agents 和工作流**：Ronald 宣布将免费开发**可靠的 AI Agents、工作流 (Workflows) 和自动化 (Automations)**（不含运行成本），以展示其能力的实际应用。
   - 他表示，他个人正在免费构建可靠/稳健的 AI Agents、工作流和自动化，不包含运行成本。他还提到已经存在 *n8n* 集成，相关的文档和示例即将发布。


  

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1430274752639140062)** (106 messages🔥🔥): 

> `免费图像生成模型, Stealth 模型, MCP 设置, GPT-5 Codex 配额, 模型过度计费` 


- ****RooCode** 提供免费模型**: **RooCode** 提供免费模型，如 **Grok Code Fast 1**、**Grok 4 Fast**、**Supernova 1 million** 和 **Deepseek Chat 3.1**，为代币销售平台提供了另一种选择。
   - 一位用户表示，他们在过去几个月里在 **roo cloud** 上消耗了数十亿个 token，系统始终运行顺畅，从未出现过降速或频率限制（rate limited）。
- ****Chutesflation**（Chutes 引起的通胀）是真实存在的**: 用户在 OpenRouter 上遇到了意料之外的高额费用，一位用户报告称在极少使用的情况下，仅 **23 天** 就消耗了 **$5**，引发了对 *chutesflation* 的担忧。
   - 另一位用户指出：*你别无选择，你很可能已经在通过 Chutes 使用了*，暗示 OpenRouter 默认将用户路由到该提供商。
- ****OpenRouter API** 有区域限制**: 用户遇到了区域限制问题，表现为来自 OpenAI 提供商的 **403 错误**，提示信息为 *Country, region, or territory not supported*（国家、地区或领土不受支持）。
   - 据解释，Cloudflare 收集的位置数据会被转发给提供商，而 [OpenRouter 本身并不施加区域限制](https://support.cloudflare.com)。
- ****Exacto** 端点使用最昂贵的提供商**: **Qwen3 coder** 的 **exacto** 端点默认开启，但它始终使用 **Google Vertex** 和 **Alibaba Open Source** 等昂贵的提供商，导致费用翻倍。
   - 一位社区成员建议，费用的增加与发布无关，用户必须更改模型才能选择加入 exacto。
- ****GPT-5** 是一个优秀的数学模型**: 用户讨论了最佳数学模型，并指出 **GPT-5** 是一个优秀的模型，并附带了 [matharena.ai](https://matharena.ai/) 的链接。
   - 有观点认为 **Grok 4** 也相当不错，**DeepSeek** 表现平平，而任何 **Claude** 模型在数学方面表现都不佳。


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - 新模型**
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1430275337345826928)** (37 messages🔥): 

> `Qwen 模型尺寸, 角色扮演废话解决方案, ExactoNow 端点, Nebius AI Studio` 


- ****Qwen 离奇的量化困惑！****: 成员们讨论了 [Alibaba_Qwen 模型](https://x.com/Alibaba_Qwen/status/19806659326253838682) 与其视觉编码器相关的异常 **1.7B** 尺寸，以及更标准的 **32B** 版本。
   - 爱好者们表达了获取这些模型的兴趣，注意到 **Qwen 聊天网站** 的表现不错，且该模型具有作为本地视觉模型的潜力，同时强调了如此小的模型竟有惊人的评分。
- ****用明智的选择消除废话（Slop）！****: 社区探索了解决**角色扮演（roleplays）**中严重“废话问题”的方案，建议采用提高 Temperature 进行多次生成，并使用低 Temperature 模型作为裁判来挑选最佳结果的方法。
   - 一位用户提议使用双阶段流水线（dual-pass pipeline）来重写散文并避免重复短语，并提到 Sam Peach 之前的工作，即提供禁止 LLM 使用的短语列表，这些列表在 Kobold.ccp 上很容易实现。
- ****ExactoNow 令端点爱好者感到兴奋！****: 社区对新的 **ExactoNow** 端点做出了反应，这是一项旨在筛选提供商并过滤掉性能较差模型的计划。
   - 用户建议在非 Exacto 页面上显示模型的统计数据，或添加表示 Exacto *质量* 的徽章。
- ****Nebius AI 在新 Qwen 模型上的出色表现！****: 一位成员报告称，[OpenRouter.ai Qwen2.5-coder-7b-instruct](https://openrouter.ai/qwen/qwen2.5-coder-7b-instruct) 模型通过 **Nebius AI Studio (Fast)** 运行效果极佳。
   - 他们指出，这是目前该模型的唯一提供商。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1430286254066503812)** (61 条消息🔥🔥): 

> `无 GPU 的独立研究、论文发表挑战、小语言模型的函数调用、数字孪生城市的校准、受限采样与 logit 偏置` 


- **低成本独立研究**：一位成员正在进行没有 GPU 的独立研究，导致 **5-8 小时的实验运行时间**并打乱了睡眠。
   - 另一位成员建议使用 [vast.ai](https://vast.ai) 租用 GPU（例如 **RTX 3090** 每小时约 **$0.15**），但前一位成员的目标是让实验通过 Google Colab 即可进行。
- **论文发表的艰难挑战**：一位成员将发表论文描述为一场*艰苦的战斗*，特别是在没有 GPU 的情况下，只能选择偏重数学的项目，并通过撰写论文来帮助理解现有概念，并参考了[这篇论文](https://arxiv.org/abs/2208.11970)。
   - 他们还提到了 **NeurIPS** 投稿的漫长准备时间，截止日期可能在 2026 年 3 月，但建议将 1 月份的 **ICML** 作为替代方案。
- **探索函数调用微调**：一位成员询问了使用 **litellm/openai** 等库为函数调用（function calling）微调小语言模型的内部机制，特别是底层如何处理 **JSON schema**。
   - 建议包括使用 **lm-format-enforcer** 或 **Lark Grammar** 来约束输出遵循工具调用的 **JSON** 格式，以及使用 llguidance 的语法来构建工具调用，以实现极低的延迟和高准确度，并将结构化输出表示为 `[Thinking] I will now call tool {tool_name}. [tool JSON]`。
- **实时 Prompting 的威力**：成员们讨论了来自 **OpenAI/Google** 的实时功能，重点关注了 **Gemini** 移动应用中的视频聊天以及 **OpenAI** 的实时语音聊天 Prompting 指南。
   - 一位成员分享了 [OpenAI 实时 Prompting 指南](https://cookbook.openai.com/examples/realtime_prompting_guide)的链接，对该技术预测用户行为的潜力表示既兴奋又谨慎。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1430318141795336354)** (34 条消息🔥): 

> `Transformer Circuits、VinePPO 实验、Jsonnet 配置` 


- **Transformer Circuits 被评为“最佳论文”**：一位成员分享了 [Transformer Circuits 文章链接](https://transformer-circuits.pub/2025/linebreaks/index.html)，称其为本周*最值得阅读的论文*，另一位成员也认为它在*框架层面具有重大意义*。
   - 他们在推荐时特别指出，作者对他*自己的愚蠢和无知*进行了严谨的分析。
- **VinePPO 实验配置详情披露**：一位成员分享了 [VinePPO 实验的配置](https://github.com/McGill-NLP/VinePPO/tree/main/configs)，指出其使用了相当复杂的 **jsonnet** 设置。
   - 如果有人感兴趣，他愿意就此进行演示，尽管他承认这有些*过度设计*，并且他*本可以用不同的方式来实现*。
- **DeepSeek 的 Jsonnet 配置备受赞赏**：一位成员正考虑在一个项目尝试 **jsonnet 方案**。
   - 他解释了自己的兴趣，理由是*如果 DeepSeek 都在用它，我也应该亲自尝试一下，看看它到底有多好*。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044755510/1430287286565863445)** (13 messages🔥): 

> `Amazon vibe code IDE, Kiro spec based IDE, LLM Brain Rot, Apple AI researchers` 


- **Amazon 的 Vibe Code IDE 结束 Beta 测试**: Amazon 的 **Vibe Code IDE** 已结束仅限邀请的 Beta 测试，为用户提供 **500 credits** 初始额度。它[被设计为“基于规格（spec based）”](https://kiro.dev/blog/waitlist-is-over/)，围绕功能和实现的规格说明进行工作，而不仅仅是依赖 prompts。
- **Kiro，基于规格的 IDE**: 一位成员指出，**Kiro** 与许多 **AI IDEs** 一样，也是一个 **VScode fork**，并被设计为“基于规格”，即围绕功能和实现的规格说明进行工作，而非单纯依靠 prompts。
   - 另一位成员建议，将规格文本转换为代码是一种任何平台都可以采用的模式，他们制作的任何特殊 VScode UI 都可以被轻松克隆。
- **“LLM Brain Rot” 论文被驳回！**: 成员们分享了一篇题为 [LLM Brain Rot](https://llm-brain-rot.github.io/) 的论文链接，但发现其质量较低。
   - 一位成员评论道：*“我本以为这会是一篇有趣且轻松的论文，但对其质量之差感到惊讶”*，另一位成员也表示同意：*“这个标题可以写出一篇好论文，但显然不是这一篇。”*
- **Apple 为推理寻找 AI 研究员**: 成员们链接了一篇关于 [Apple 正在寻找推理（reasoning）方向的 AI 研究员](https://the-decoder.com/apple-seeks-ai-researchers-for-reasoning-even-as-its-own-study-questions-current-models/)的文章。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1430369914685755564)** (15 messages🔥): 

> `NCU wrapper scripts, Custom NCU metrics, PyTorch Conference AI Infra panel discussion, L2 cache hit rates, LLM Systems` 


- **NCU 封装脚本寻求者希望自定义指标**: 一位成员正在寻找 [NCU 封装脚本的 GitHub 仓库](https://github.com/NVIDIA/nsight-compute)，以便通过 `--metrics` 传入需要分析的指标列表。
   - 另一位成员建议使用 [NVIDIA Nsight Compute Customization Guide](https://docs.nvidia.com/nsight-compute/CustomizationGuide/index.html#section-files) 创建包含自定义指标的自定义部分/集合。
- **GPU Kernel 大佬齐聚 AI Infra**: [PyTorch Conference AI Infra 关于 GPU kernels 的小组讨论](https://aiinfrasummit2025.sched.com/event/28FoW/panel-discussion-the-ai-kernel-revolution-rethinking-execution-from-the-ground-up-robert-lange-sakanaai-simran-arora-stanford-university-nathan-lambert-allen-institute-moderated-by-mark-saroufim-gpu-mode) 达成共识，认为使用 **PTX/assembly** 可以实现 kernel 的峰值性能。
- **L2 缓存难题：命中率超过 100%**: 一位成员询问是否有人在 **NCU** 上见过 **L2 缓存命中率 >100%** 的情况，以及该如何解释。
   - 另一位成员分享了一个可能有所帮助的帖子：[NVIDIA Forum](https://forums.developer.nvidia.com/t/weird-number-for-l2-cache-hitrate/120341/2)
- **Awesome LLM Systems 正式发布**: 一位成员发布了一个新仓库 [Awesome LLM Systems](https://github.com/romitjain/awesome-llm-systems)，这是一个精选的列表，涵盖了大型语言模型系统侧的关键论文、博客和资源。
- **数值稳定性知识点**: 在阅读了关于 **FA4 的正确性优化**和确定性推理（deterministic inference）的博客后，一位成员寻求更多关于**数值稳定性（numerical stability）**的资源，并指向了 [stochastic rounding](https://arxiv.org/pdf/2207.10321)。
   - 另一位成员分享了 David Bindel 教授提供的一份很好的总结：[epubs.siam.org](https://epubs.siam.org/doi/book/10.1137/1.9781611971491)


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1430323860888223825)** (1 messages): 

> `CuPy GPU Pointer vs PyTorch GPU Pointer, DLPack Conversion Performance, MatMul Kernel Performance` 


- **CuPy GPU 指针性能优于 PyTorch GPU 指针**: 一位成员询问在自定义 **MatMul kernel** 中使用 **CuPy GPU 指针**与 **PyTorch GPU 指针**时的性能差异，并注意到显著的性能差距。
- **DLPack 转换瓶颈**: 该成员观察到，使用 **DLPack** 将 **CuPy array** 转换为 **PyTorch tensor** 再转回 CuPy，尽管数值结果相同，但会导致性能下降。
   - 他们质疑这种性能差异是否存在内在原因，并展示了一张性能对比的[截图](https://cdn.discordapp.com/attachments/1189607595451895918/1430323860464734360/Screenshot_from_2025-10-21_17-32-52.png?ex=68faade6&is=68f95c66&hm=d84234753a2510107fb4d7ecd73bbf01b7e07a92a430333bafa04d79be3e8bd3)。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1430516998030364702)** (9 messages🔥): 

> `warpGroup.arrive injection, Nvidia Stream Benchmark, CUDA video upscale project` 


- **`warpGroup.arrive` 注入引发关注**：详细编译模式（Verbose compilation）提示一条消息，称 *`warpgroup.arrive` 已被编译器注入在第 2580 行左右，以允许在函数的 GMMA 中使用寄存器*，这表明 **所有的 wgmma 都使用相同的寄存器**。
   - 进一步澄清，发生这种情况是为了 *允许在 GMMA 中使用寄存器*。
- **探索 Nvidia 的 Stream Benchmark 选项**：一位成员询问有关 **Nvidia Stream benchmark** 的信息，以衡量可实现的带宽，并参考了 Decoupled Look-back 讲座。
   - 另一位成员分享了几个相关链接，包括 [cpplinks](https://github.com/MattPD/cpplinks/blob/master/performance.tools.md#memory-benchmarking)、[jeffhammond/STREAM](https://github.com/jeffhammond/STREAM) 以及支持 GPU 的 [UoB-HPC/BabelStream](https://github.com/UoB-HPC/BabelStream)。
- **CUDA 项目在视频超分辨率速度上遇到困难**：一位成员就一个 CUDA 项目寻求帮助，该项目旨在实现实时视频超分辨率，尽管使用了 GPU 加速，但仅达到了 **0.5 FPS**。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1430412919752884235)** (2 messages): 

> `CPU affinity, taskset command, nice command` 


- **使用 Taskset 和 Nice 优化 CPU 使用率**：为了优化 CPU 使用率，特别是当节点未运行其他任务时，请使用 `taskset` 命令绑定线程。
   - 此外，使用 `nice` 命令确保这些线程在 CPU 上保持优先级，防止它们被降低优先级。
- **Taskset 命令详解**：`taskset` 命令允许你将进程或线程绑定到特定的 CPU 核心或核心集合。
   - 这确保了进程或线程仅在指定的内核上运行，从而通过减少上下文切换（context switching）和缓存未命中（cache misses）来潜在地提高性能。


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1430648973412139068)** (2 messages): 

> `Relevant Algorithms for Optimization, Real-time Video Upscaling Projects` 


- **亟需优化的算法备受关注**：一位成员询问了目前有哪些相关的算法可以从优化中显著受益。
   - 他们还询问了适合实验的经典算法，旨在紧跟最新标准并寻求项目灵感，特别是针对实时视频超分辨率（real-time video upscaling）的项目。
- **实时超分辨率项目构思**：该成员正在寻找开发实时视频超分辨率的项目。
   - 他们希望紧跟最新标准，并寻找一个好的算法进行实验。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1430295615459299420)** (1 messages): 

> `SIG Hiring, PyTorch Conference` 


- **SIG 招募量化人才！**：来自量化交易公司 **Susquehanna International Group (SIG)** 的 Jacob 宣布，他们正在 [招聘多个职位](https://sig.com/careers/quant/)。
   - 鼓励感兴趣的候选人私信 Jacob 进行进一步讨论，或在 [PyTorch 会议](https://calendly.com/jacob-baumbach-sig/pytorch-2025) 上与他面谈。
- **量化机会来袭**：**SIG** 正在积极寻求公司内各个职位的量化（Quant）人才。
   - 鼓励感兴趣的人士探索职业机会，并联系 Jacob 了解更多详情。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1430646355851870279)** (1 messages): 

> `GPU Rental, Acer Nitro 5, New vs Used GPUs, Utilizing Newest GPU Features` 


- **租用 GPU 来学习入门？**：一位经验有限的成员询问，**租用廉价 GPU** 是否是初学者入门和提高技能的好方法。
   - 他们质疑，在还没学会充分利用 GPU 特性之前，购买最新、最强大的 GPU 是否有意义；该成员目前使用的是 **Acer Nitro 5 游戏笔记本电脑**。
- **新特性对初学者来说是不必要的吗？**：一位参与者发表观点认为，如果无法充分利用新特性，那么购买最新的 GPU 硬件就没什么意义。
   - 这一疑问建议了一种策略：在投资高端 GPU 之前，先在**价格合理的硬件上学习基础知识**。


  

---

### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1430574442614099969)** (3 messages): 

> `C++ 更新，Python 版本` 


- **C++ 版本面临发布延迟**：C++ 版本正在开发中，正面临来自出版商的延迟。
   - 由于出版商相关的延迟，预计不会在年底前发布。
- **Python 版本仍在讨论中**：目前正在讨论一个 Python 友好版本。
   - 然而，尚未建立具体的计划或时间表。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1430346894260965582)** (3 messages): 

> `Hackathon 项目，工作组想法` 


- **工作组想法成为 Hackathon 素材**：一位用户询问 mobicham 创建的工作组想法是否用于 Hackathon。
   - Mobicham 回复说这些只是想法，团队在 Hackathon 期间可以选择进行其他项目。
- **鼓励替代性 Hackathon 项目**：Mobicham 澄清说提供的主题仅仅是建议。
   - 团队在 Hackathon 期间可以灵活地追求符合其兴趣和技能的其他项目。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

bobmarleybiceps: 我也在 OC (Orange County)
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/)** (1 messages): 

snektron: https://www.thingiverse.com/thing:7179241 我终于费心上传了这个
  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1430341094968660099)** (3 messages): 

> `Profiling 日志，NCU 分析` 


- **来自 Profiling 日志的可操作见解**：一位成员建议将海量的 **profiler 日志**和**指标**转化为可操作的见解。
   - 另一位成员建议这应该作为一个 **kernel generating agent** 的子组件。
- **NCU 能消除 Profiling 瓶颈吗？**：一位成员询问 **NCU** 和类似工具是否已经能够精准定位问题和**瓶颈**。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1430561313859571742)** (3 messages): 

> `sort_v2 排行榜，B200 性能，H100 性能，L4 性能，A100 性能` 


- **Sort_v2 排行榜冠军揭晓**：一位用户凭借提交 ID `66238` 获得 `sort_v2` 排行榜第一名，在 B200 上表现为 **8.68 ms**，在 A100 上为 **16.3 ms**。
   - 该提交在 H100 上也取得了 **6.60 ms** 的成绩，在 L4 上为 **52.7 ms**。
- **另一个 sort_v2 条目入榜**：ID 为 `66239` 的用户向 `sort_v2` 排行榜提交了结果，展示了 B200 **8.83 ms**、L4 **52.6 ms**、H100 **6.60 ms** 以及 A100 **16.5 ms** 的性能。
   - 该提交展示了在不同硬件配置下的一致性能。
- **sort_v2 第一名再次出击！**：一位用户通过提交 ID `66241` 在 `sort_v2` 排行榜上取得了 L4 **52.6 ms** 和 A100 **16.0 ms** 的第一名成绩。
   - 他们在 B200 上也跑出了 **8.69 ms**，在 H100 上跑出了 **6.58 ms**。


  

---


### **GPU MODE ▷ #[tpu](https://discord.com/channels/1189498204333543425/1351571759761068082/1430427470817005701)** (2 messages): 

> `Jax 转换，TPU，PyTorch，vLLM` 


- **PyTorch 模型实现 JAX 化**：一个新的库允许通过这个 [链接](https://google.github.io/torchax/) 使用 **Jax 转换层**运行 **PyTorch 模型**。
- **vLLM 登陆 TPU**：根据 [这篇博文](https://blog.vllm.ai/2025/10/16/vllm-tpu.html)，新版本的 **vLLM** 显然将使用 **Jax 转换**在 **TPU** 上运行。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1430294254919614544)** (3 messages): 

> `GymAgent，推理模型，Action->Response agent，MCP 集成` 


- **GymAgent 解析**：**GymAgent** 是一个 **Action->Response agent**，它在每一轮观察环境，并能在编写代码在游戏中运行之前进行自然语言推理。
   - 与具有 **MCP 集成**的纯推理模型（如 Claude Code）不同，它不能选择何时观察或以其他方式与状态交互。
- **Action vs 推理**：**GymAgent** 使用 **Action->Response**，这允许它每轮观察环境。
   - 具有 **MCP 集成**的**推理模型**可以选择何时观察或与状态交互。


  

---

### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1430312950366736544)** (8 messages🔥): 

> `CUTLASS Cmake, TiledCopy, Thread-Value layouts` 


- **CUTLASS 获得 CMake 示例**：一位成员分享了一个 [CUTLASS CMake 示例](https://github.com/leimao/CUTLASS-Examples)，用于构建简单的 CUTLASS 代码。
   - 这可能是关于如何将 CMake 与 CUTLASS 结合使用的*优秀入门示例*。
- **TiledCopy 线程是否在复制零值？**：一位成员询问在文档的 `TiledCopy` 示例中，值 **0** 是否被多个线程重复复制。
   - 图片展示了一个代码片段，使用了 `make_tiled_copy` 以及 `Copy_Atom`、`Layout<Shape<_32,_8>>` 和 `Layout<Shape< _4,_1>>` 来定义线程和值布局，引发了关于数据重复的疑问。
- **Thread-Value 布局反转**：一位成员指出，图片显示了两个*逆* **Thread-Value 布局**，它们将数据坐标映射到 **(Thread, Value)** 坐标。
   - 他们澄清道，*T32V0 表示从线程 32 的视角（POV）来看的第 0 个数据项*。


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1430289283943501865)** (15 messages🔥): 

> `Mojo Language, Apple Silicon, GPU Algorithms, Metal Toolchain` 


- **Modular 的 Mojo 目标远大**：一位成员表示 **Modular** 和 **Mojo** 语言的目标非常宏大，如果能取得成功将令人兴奋。
- **Mojo 在 Apple Silicon 上的表现：喜忧参半**：一位用户报告在 **Apple Silicon** 机器上用 **2-3 小时**完成了前 8 个问题，并指出这些问题很简单，但有些问题无法在他们的电脑上运行。
   - 另一位用户确认 **Mojo** 确实可以在 **Apple Silicon** 的 alpha 版本上运行，但需要安装 **Metal toolchain** 和 **Xcode**，他们觉得这很麻烦。
- **GPU Algorithm 问题成为焦点**：一位用户希望即将到来的关于 **GPU algorithms** 的问题能更深入地探讨 **GPU layouts, blocks, and threads**。
   - 另一位用户建议第 **25-34** 题会很酷，但第一位用户无法在自己的机器上运行，并开玩笑说需要一台 **DGX**。


  

---


### **GPU MODE ▷ #[irl-accel-hackathon](https://discord.com/channels/1189498204333543425/1416087968933740688/1430363465423519775)** (5 messages): 

> `Porting Qwen 3 to Burn, Mega Kernel Compilation, Hackathon Team Formation, Hackathon Registration Status` 


- **Qwen 3 瞄准 Burn 移植**：一位成员计划在 IRL 黑客松期间将 **Qwen 3** 移植到 [Burn](https://burn.dev/)，目标是将 **0.6B 变体**编译成单个 [mega kernel](https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17)。
- **Mega Kernel 探索开启**：尽管是 GPU 编程和 Burn 的新手，一位成员正在探索使用 Burn 进行严肃工作以及将 **LLMs** 编译成 **megakernel** 的可行性。
- **黑客松小队寻找 Kernel 高手**：一位精通 **Rust** 的成员正在寻找黑客松队伍，特别是擅长 kernel 的队友，以协作完成 **I/O** 或 **通信相关项目**，例如 **KV/weight transfers** 或 **基于磁盘的 KV cache**。
- **黑客松申请者等待批准**：一些成员仍在等待黑客松的批准以最终确定旅行计划。
   - 一位组织者回应称，黑客松的报名人数已超额约 **6 倍**。


  

---


### **GPU MODE ▷ #[opencl-vulkan](https://discord.com/channels/1189498204333543425/1418990184367919267/1430575051484430358)** (1 messages): 

> `coopVecMatMulAddNV in GLSL, GLSL Cooperative Vector Extension, VkConvertCooperativeVectorMatrixInfoNV, Hand-prepacked buffers with coopVec, Row-major order in GLSL` 


- **探索在 GLSL 中结合手动预打包缓冲区使用 coopVec**：一位成员询问如何在 GLSL 中将 `coopVecMatMulAddNV` 与手动预打包的缓冲区结合使用，目前面临 Vulkan 驱动程序不允许通过 `VkConvertCooperativeVectorMatrixInfoNV` 更改布局的问题。
   - 他们有一个使用 `coopVecMatMulAddNV` 的简单 MLP，但输出与非 coop-vec 版本不一致，使用的是 RowMajor 顺序的 float16_t 数组。
- **GLSL Coop-Vec 输出差异**：一位成员在 GLSL 中使用 `coopVecMatMulAddNV` 实现了一个 MLP，但观察到输出与非 coop-vec 版本不同。
   - 他们在权重和偏置中使用了 RowMajor 顺序的 float16_t 数组，并询问是否可以手动打包数据以使 `gl_CooperativeVectorMatrixLayoutRowMajorNV` 正常工作。


  

---

### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1430281207177936916)** (17 条消息🔥): 

> `Helion 博客文章，int4_gemm 参考，Helion vs Triton 性能` 


- **Helion 博客文章上线**：**Helion 博客文章**现已在 [pytorch.org](https://pytorch.org/blog/helion/) 发布，展示了**性能数据**以及与 **Gluon**、**CUTEst** 和 **Cutlass** 的**对比**。
   - 团队邀请大家参加“会见 **Helion 开发者**”活动。
- **int4_gemm 参考请求**：一名成员询问了 **int4_gemm** 的参考实现。
   - 另一名成员指向了 [仓库中的 examples 文件夹](https://github.com/pytorch/helion/blob/main/examples/int4_gemm.py)，该文件夹使用 **Liger kernels** 和 **torch.compile kernels** 作为参考。
- **Helion 的加速主张受到质疑**：一名成员对报道的 **14 倍加速**表示怀疑，认为即使是与 **fp16** 相比的内存受限（memory-bound）操作，这也不太现实，并指出 [int4_gemm 实现](https://github.com/pytorch/helion/blob/main/examples/int4_gemm.py#L166-L178) 并非融合算子（fused kernel）。
   - 建议应与 **Marlin** 或 **Machete** 等融合算子进行对比，或者至少与带有 `tl.dot_scaled` 的简单 **Triton gemm kernel** 进行对比。
- **对更优基准测试基准的渴望**：一名成员表示，这其实不是 bug，而是预期不匹配：读者期望看到相对于专门的 **cutlass/cute kernels** 的**加速对比**。
   - 一名成员同意更新参考实现，并鼓励在 GitHub 上创建 issue。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1430298738399903755)** (51 条消息🔥): 

> `AI “恶魔机器”，批判性思维能力，Qwen3 embedding，LM Studio API key，本地 AI TTS` 


- **AI 是“恶魔机器”？**：成员们讨论了生成式 AI 是“恶魔机器”的情绪，指出由于图像生成，艺术家群体产生敌意是可以理解的，并担心**人们正在丧失批判性思维能力**。
   - 一名成员反驳说，由于 AI 容易产生幻觉（hallucinate），他们实际上*进行了更多的批判性思考*。
- **Qwen3 Embedding 得到改进**：一名成员表示，**Qwen3 embedding 8b** 的*较新量化版本（已修复的版本）*在配合 roocode 代码索引工作时，比他们之前使用的版本*准确得多*。
   - 他们澄清说，与 **mxbai-embed-large:latest** 相比，相关查询的置信度分数要高得多，而不相关查询的分数则低得多。
- **LM Studio 支持第三方 LLM**：一名成员询问，如果他们有 API key，是否可以使用 **LM Studio** 与第三方 LLM 进行通信。
   - 另一名成员回答说，*通过插件（目前处于封闭测试阶段），你将能够实现这一点*，并分享了一个 [OpenAI 兼容端点的链接](https://lmstudio.ai/fuutott/openai-compat-endpoint-v2)。
- **探索本地 AI TTS**：在一名成员寻找 **AI voice** 解决方案后，另一名成员推荐了 *chatterbox*，认为它是一个*非常棒的* **本地 AI TTS** 选择。
   - 确认 **Chatterbox** 支持多种语言。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1430280534151663716)** (34 条消息🔥): 

> `ATX power supplies, Ground loop issues, GPU powering issues, Vietnam cheap 3090/4090, Mi50 blower egpu` 


- **ATX 电源绿线接地**：成员们讨论了大多数 **ATX** 电源上的**绿线**共享如何允许其在接地时开启，但可能会导致**地环路问题 (ground loop issue)**。
- **多个 PSU 可能相互倒灌**：提到使用与主板 PCIE 供电不同的独立 **PSU** 为 **GPU** 供电可能会导致 PSU 之间相互倒灌、主板虚假供电以及地环路问题，但可以通过使用同步线 (sync cables) 来避免。
   - 他们建议不要并联 **12V 导轨 (12V rails)**，因为在将一个 **GPU** 分配到多个 **PSU** 时可能会引发问题。
- **越南 4090 捡漏不太可能**：一位成员询问在越南哪里可以买到**便宜的 3090/4090**，但另一位成员表示，越南是向中国供应 **4090** 的地区之一，因此不指望会有什么大便宜。
- **Mi50 涡轮 eGPU 重新涂抹硅脂并刷入固件**：一位成员用 Arctic 硅脂为他们的 **Mi50 涡轮 eGPU** 重新涂了硅脂，报告在 3/4 风扇转速下的推理温度为 **50°C**，并计划刷入 v420.rom 以在 Vulkan 上获得完整的 **32GB** 显存并研究完整的温度读取。
   - 另一位成员建议使用 [274474 rom](https://gist.github.com/evilJazz/14a4c82a67f2c52a6bb5f9cea02f5e13)，因为 V240 有 **178W** 的功耗限制。
- **AMD Instinct MI50 导风罩可打印**：一位成员提到，在风扇 100% 转速下，他们的显卡**结温 (junction)** 从未超过 **90°C**，而另一位成员链接了一个 [Printables.com 模型](https://www.printables.com/model/1421067-amd-instinct-mi50-shroud)，用于 **AMD Instinct MI50** 的导风罩。
   - 他们还考虑再次使用 **PTM7950** 重新涂抹。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1430272074861248592)** (53 条消息🔥): 

> `RAG implementation, AI Fact Checking, HF download with Xet, AI Agent Course` 


- **RAG 没那么难？**：成员们讨论了如何构建 RAG 系统：*请求 -> RAG 服务 -> AI 服务 -> RAG 服务 -> 响应*，建议使用带有向量插件的 **PostgreSQL** 数据库。
   - 一位成员指出，虽然基础的 RAG 设置很简单，但在广泛的主题中调整它以获得相关结果是很棘手的；另一位成员补充说，在每一步前后使用一个小模型作为**完整性检查 (sanity check)** 是一种有效的措施。
- **用于可靠事实核查的 AI Agents？**：成员们讨论了实现一个用于可靠核查新闻事实的 **AI agent**，建议该 agent 需要具备网页搜索能力。
   - 一位成员指出，它可能会给出虚假与真实的概率可能性，但不能保证 *100% 确定*。
- **`hf download` 保存到 `hub`？**：一位成员询问为什么 `hf download repo/id` 将文件保存到 `hub` 而不是 `xet`，以及如何确保 CLI 使用 `xet` 下载。
   - 另一位成员解释说，**HF** 有自己的保存到 hub 的设置，而 `xet` 是完全不同的设置。
- **AI Agent 课程的成果与收益**：一位成员询问参加 **AI Agent 课程** 的成果和收益，因为他们是初学者并感到好奇。
   - 一位成员回答说，成果和收益是*学习、满足你对技术的好奇心，并了解这些知识能带你去向何方*，而且它可以让人为自己制作 AI agents，或者向其他人/公司提供此类服务。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1430284183174905896)** (2 条消息): 

> `AI Refactoring, vibecoder.buzz` 


- **AI 暂停以重构合理的代码**：一位成员学会了暂停并提示 AI *像个正常人一样* 重构代码，使用的提示词是：`as a senior architect focused on modular code and maintainability`。
   - 对于更改，他们的提示词是：`do not make changes or write code, answer question: do you have enough info to make these updates?` 以及 `please create the minimal changeset (no tests)`。
- **vibecoder.buzz 上线了！**：一位成员报告说他们的项目已在 [vibecoder.buzz](https://vibecoder.buzz) 上线。
   - 该项目花费了 **$2** 购买域名以启用电子邮件验证，偏离了他们最初花费 **$0** 的目标。


  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1430286618723352666)** (2 条消息): 

> `Databomz, Chrome extension for prompts, Prompt organization` 


- **Databomz：Prompt 工作空间正式发布**：一名成员介绍了 **Databomz**，这是一个用于保存、组织和共享 Prompt 的工作空间和 Chrome 扩展，具有标签、版本和文件夹等功能，更多信息请访问 [www.databomz.com](http://www.databomz.com/)。
- **永久免费层级吸引 Prompt 资深用户**：**Forever Free 计划**包含大部分核心功能，开发者正在寻求活跃 Prompt 用户的反馈。
   - 该项目的 GitHub 仓库位于 [github.com/Lnrchaos/NeSy-CML](https://github.com/Lnrchaos/NeSy-CML)。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1430318686987751496)** (4 条消息): 

> `Fenic integrates with Hugging Face Datasets, Multi-model collaboration, Prompt optimization tool` 


- **Fenic 接入 🤗 Datasets**：开源项目 **Fenic** 现在直接集成了 **Hugging Face Datasets**，允许用户直接从 Hub 填充版本上下文，并安全地为 Agent 提供工具化支持，详见[文档](https://huggingface.co/docs/hub/datasets-fenic)。
   - Fenic 类似于用于计算的 e2b，它支持数据快照、Agent 上下文创建，并能通过类似于 pandas 的 dataframe API 暴露 MCP 工具，[Fenic 仓库已在 GitHub 上线](https://github.com/typedef-ai/fenic)。
- **多模型协作评估**：关于多模型协作的评估正在进行中，重点是降低每个请求中单用户的幻觉率并提高整体请求质量；更多细节可在[此博客](https://facilitair.ai)中找到。
   - 顺序协作已取得良好效果，目前有两个使用协作的开源仓库至少已达到 v1 版本。
- **Datafrosch Newsletter 实验**：[Datafrosch newsletter](https://datafrosch.fun/blog/rss-newsletter.html) 正在进行持续实验，鼓励社区分享见解。
   - 鼓励社区讨论哪些方案有效，哪些无效。
- **Genie Prompt Optimizer Chrome 扩展**：这是一个作为 Chrome 浏览器扩展创建的工具，旨在改进 Prompt，可在 [Chrome Web Store](https://chromewebstore.google.com/detail/genie-prompt-optimizer/eejkodpbdljgoiidoekiiogpehoghnip) 试用。
   - 非常欢迎对该工具提出反馈。


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1430411424315736145)** (1 条消息): 

> `Diffusion Models, DDPM` 


- **简化 Diffusion Models 背后的数学**：一名成员分享了他们的文章，旨在为初学者简化 **Diffusion Models (DDPM)** 背后的数学原理：[The Math Behind Diffusion Models (DDPM)](https://joydeep31415.medium.com/the-math-behind-diffusion-models-ddpm-9fabe9c9f1d9)。
- **Diffusion Models 初学者资源**：提到的另一个资源是一篇关于理解 **Denoising Diffusion Probabilistic Models (DDPM)** 背后数学的初学者友好文章。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1430295971765424210)** (2 条消息): 

> `MBZUAI K2Think, OpenAI text-embedding-3-large dataset` 


- **MBZUAI K2Think 竞赛开启**：一名成员分享了 **MBZUAI K2Think** 竞赛的 [LinkedIn 帖子](https://www.linkedin.com/posts/mbzuai_mbzuai-mbzuai-k2think-activity-7383761114959876097-0R7f?utm_source=share&utm_medium=member_desktop&rcm=ACoAAD53GRUB60-DZ9YvQ9NaG-LySvMdcC2QJzI)，并邀请他人组队。
- **寻求 OpenAI text-embedding-3-large 训练数据集**：一名成员询问用于训练 OpenAI `text-embedding-3-large` 嵌入模型的数据集是否公开。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1430467324099694635)** (2 条消息): 

> `Cogito 14b, AI Agent Course, Benefits of the AI Agent Course` 


- **Cogito:14b 在 Agent 课程中表现出色**：一名成员使用 **Cogito:14b** ollama 模型完成了 Agent 课程，并在 [LinkedIn 上分享了个人见解](https://www.linkedin.com/posts/duhyeon-kim-6623082b1_aiagents-huggingface-cogito-activity-7386672913896067072-YDbx?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEr5baoBwnyzQRLN-2xbgBSLqVXBm-f_i_QHi)。
   - 他们邀请对 **AI Agent** 和该课程感想感兴趣的人进行互动。
- **初学者寻求课程成果**：一名新成员对参加 **AI Agent 课程**的动力和优势表示好奇。
   - 他们询问了关于参与者预期收益和成果的见解。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1430295667871449169)** (50 messages🔥): 

> `AI Studio 升级, Lovable Shopify 集成, OpenAI Project Mercury, Langchain 社区, GoogleQuantumAI 加速` 


- **AI Studio 获得 Firebase 全栈升级**：Logan Kilpatrick 确认 [Firebase Studio 集成](https://ai.studio/build) 即将引入新的 **AI Studio**，该工具因其 vibe-coding 应用能力和语音转文字功能而备受赞誉。
   - 用户请求更便捷地集成数据库、身份验证（auth）和存储，Logan 邀请用户针对不支持的用例（如 Chrome 扩展和 SolidJS）提供反馈，并澄清 **Gemini 2.5 Pro** 是当前使用的模型。
- **Lovable 发布 AI Shopify 全店集成**：Lovable 宣布了一项 **Shopify 集成**，允许用户通过简单的 Prompt 快速创建完整的在线商店，并演示了启动其自身周边商店（[lovable.dev/merch](https://lovable.dev/merch)）的流程。
   - 该功能对所有用户开放，并附带 **30 天 Shopify 试用期**，不过目前尚不支持导入或修改现有的 Shopify 商店。
- **Project Mercury 支付投资银行家以训练 OpenAI**：[Project Mercury](https://www.entrepreneur.com/business-news/openai-is-paying-ex-investment-bankers-to-train-its-ai/498585) 以每小时 **150 美元** 的报酬聘请承包商将金融模型输入 AI，从而扩大 AI 在金融和科技等商业领域的实际应用。
- **GoogleQuantumAI 的 Willow 芯片实现大幅加速**：**GoogleAI** 宣布了一个新的里程碑，利用 **65 量子比特 Willow 芯片** 和 **Quantum Echoes (OTOC) 算法** 运行一项可验证任务，速度比顶级超级计算机快 **13,000 倍**（[X 平台帖子](https://x.com/googleai/status/1981022228801307035)）。
   - 团队讨论了其对密码学（SHA-256 安全性）、结果可验证性、药物研发和气候建模的实际时间表，以及运行《孤岛危机》/《毁灭战士》的影响。
- **Next.js Evals 考察框架的 AI 适配能力**：Guillermo Rauch 宣布了 [Next.js Evals](https://xcancel.com/rauchg/status/1981037270624076092)，这是一套开源“考试”，让任何 LLM/Agent 证明其能够正确使用 **Next.js** 和其他支持的框架进行构建。
   - 像 **GPT-5-codex** 和 **Claude Sonnet 4.5** 这样的模型目前的得分在 40% 左右；社区要求增加真实世界的任务、公开追踪记录和成本列。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1430331836764131479)** (8 messages🔥): 

> `Sesame AI 融资, Sora 路线图` 


- **Sesame 获得 2.5 亿美元 B 轮融资**：[Sesame](https://x.com/AnjneyMidha/status/1980705692253331624) 开启测试版，并完成了由 **Sequoia & Spark** 领投的 **2.5 亿美元 B 轮** 融资。
- **Sora 的社交功能、Android 版发布及 Cameos 即将推出**：**OpenAI 的 Bill Peebles** 透露了 [Sora 的路线图更新](https://x.com/billpeeb/status/1981118483607032050)，包括将在几天内推出的 **角色客串 (character cameos)**、剪辑拼接编辑器、即将推出的群组 **社交频道**、Feed 流改进、减少过度审核、性能提升以及 **Android 版发布**。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1430272461353521272)** (38 messages🔥): 

> `Scammer Alert, Manus Credit System Stress, Account Suspensions, Local Compute Utilization, Pro Plan Credit Limits` 


- **指控满天飞：用户被贴上“骗子”标签！**：一名用户因涉嫌为法学院考试研究征求付费账号的登录权限，被公开指责为*诈骗犯（fraudster scammer）*，引发了激烈讨论。
   - 指控者警告不要分享账号凭据，理由是存在个人和银行信息被盗的潜在风险，但被指控方回应称已找到*另一种方式*来解决问题。
- **用户分享建站经验**：一名用户询问社区是否有使用 Manus 构建网站的示例，并请大家分享在该平台上的使用体验。
   - 另一名成员通过私信（DM）回答了这个问题。
- **积分困惑：Pro 计划承诺无限量？**：用户对积分系统表示困惑和沮丧，特别是关于 **Pro Plan**。一名用户在购买该计划后感到被*诱导转向（bait and switched）*，因为之前承诺无限积分的帮助页面现在消失了。
   - 一些用户表示愿意为无限计划支付每月 200 美元以上的费用；还有人指出，尽管进行了调整，积分系统仍需要不断的管理，并需要参与改进计划才能获得免费积分。
- **账号停用困扰新用户**：一名用户报告称，他和女友的账号在输入银行卡信息后不久就被停用，怀疑可能是邀请了过多的员工触发了停用机制。
   - 尽管如此，他们已经在账号上创建了项目，目前正在寻求支持，以了解如何保留这些优秀的项目。
- **集思广益本地算力（Local Compute）**：一名用户建议利用本地算力资源来增强 Manus 的功能。
   - 该用户建议以此支持构建大型原生应用、在本地处理海量数据集、在本地硬件上运行资源密集型 AI 模型，并利用机器的全部性能实现更快的构建。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1430283392334958633)** (23 messages🔥): 

> `Perplexity Comet, Pacific-Prime model, Regular Role, Chat Datasets with System Messages, Pretraining Interview` 


- **Perplexity Comet 邀请引发关注**：一名成员询问现在领取 [Perplexity Comet](https://www.perplexity.ai/) 的 **1 年免费 Pro 邀请**是否值得。
   - 这个问题引发了大家对 Perplexity Comet Pro 版本的当前价值和功能的关注。
- **Pacific-Prime 宣称环保**：来自法国的 Boris 介绍了 [Pacific-Prime 架构](https://huggingface.co/Pacific-Prime/pacific-prime)，强调其 **25 层**中的每一层都执行 **5 次迭代**，并学习无误差收敛。
   - 该模型可以在支持 **CUDA** 的硬件上运行，拥有 **1B** 参数和 **6GB VRAM**，作者声称它比 *llamacasual 架构环保两倍*。
- **“Regular”角色依然存在，按个案授予**：一名成员询问“Regular”角色是否仍在发放，怀疑这已是过时的做法。
   - 另一名成员确认他们最近刚获得该角色，第三名成员确认该角色是根据个案授予那些*一直为社区做出贡献*的人。
- **推荐预训练访谈**：一名成员分享并推荐了[一段关于预训练（pretraining）的访谈](https://spotify.link/O1llYVO8FXb)。
   - 未提供关于该访谈的更多细节。
- **寻找带 System Message 的聊天数据集**：一名成员征求**带有 System Messages 的聊天数据集**推荐。
   - 他们表示很难找到最近的相关资源。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1430554105486639224)** (2 messages): 

> `Claude shapeshifting, Geometric structures in AI tasks` 


- **Claude 变形了！**：一名成员在 X 上分享了一篇关于 [Claude 变形（shapeshifting）](https://fxtwitter.com/wesg52/status/1980680563582538099)的帖子。
- **AI 任务中的几何结构？**：一名成员想知道，在多少任务中可以发现类似于 [Transformer Circuits Thread on Linebreaks](https://transformer-circuits.pub/2025/linebreaks/index.html) 中所展示的那种几何结构。


  

---

### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1430574403174793398)** (1 messages): 

> `Tool Call Property Encoding, JSONSchema Subsets, LLM Reasoning with Schemas` 


- **在工具调用中编码数组/对象属性**：一位成员询问如何在 `inputSchema` 中编码一个既可以是 **array** 也可以是 **object** 的属性，因为在所使用的 JSONSchema 子集中不支持 `oneOf`。
   - 讨论围绕如何定义一个灵活的 schema 展开，使得 LLM 能够进行推理，且在属性具有多种数据类型时客户端也能接受。
- **JSONSchema 子集限制**：用户在为工具调用定义 `inputSchema` 时，受到 **JSONSchema** 受限子集的约束。
   - 诸如 `oneOf` 之类的功能（允许为属性指定多个有效的 schema 类型）在该环境中不受支持，这给既可以是数组也可以是对象的属性带来了挑战。


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1430596327594786988)** (23 messages🔥): 

> `MCP Server Context Limits, Optimizing MCP Costs, Tool Call Context Management, Workflow-Based Tool Execution, Subagents for Task Isolation` 


- **MCP Server 快速达到上下文限制**：一位用户在使用包含 **60 个工具** 的 MCP server 时遇到了上下文限制问题（即使使用了最高级套餐），并怀疑工具描述导致了上下文膨胀。
   - 将工具拆分到多个 server 的做法并未得到客户的认可，这促使该用户探索更高效的上下文和成本管理方案。
- **工程师构建自定义聊天界面以应对成本问题**：一位工程师通过直接 API 连接和优化模型选择构建了自定义聊天界面来管理成本，并指出由于套餐限制，在 Claude 桌面端或 ChatGPT 中运行时成本会迅速累积。
   - 他们发现，使用带有模型切换的多 Agent MCP 客户端会导致 **每次操作 $7-8** 的成本，因此放弃了该方法。
- **精简工具工作流以提高准确性和节省上下文**：一位工程师实现了一个由 **3 个工具** 组成的工作流来管理 **50 多个 CLI 操作**，包括：列出操作、描述操作和调用操作，并使用 server instructions 来引导 LLM。
   - 他们强调，为大量工具使用描述和 input/output schemas 会使模型不堪重负并超出上下文限制，因此需要精简的工作流。
- **工具调用上下文应过期以提高效率**：一位工程师提出了一种向 Agent 发出信号的方法，表明上下文中不再需要某个工具调用以防止膨胀，并建议在 `tools/list` 调用中添加 `hint` 或字段。
   - 其核心思想是，一旦 LLM 使用了来自工具调用的信息（例如列出文档章节），该调用就可以从上下文中移除，从而提高效率。
- **Subagents 在 MCP 中隔离任务**：讨论涉及使用 subagents 来隔离任务，尽管 MCP 目前还没有正式的 subagents 或 Agent 间通信的概念。
   - 有人建议可以使用 server `instructions` 来引导客户端在可能的情况下优先为某些任务使用 subagents，但客户端的合规性仍是一个问题。


  

---

### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1430385505270497361)** (1 messages): 

> `Voice Generation, Automated Judging Loop, ElevenLabs Integration, DSPy Optimization` 


- **DSPy 系统生成 NPC 语音**：一位成员使用 **DSPy** 构建了一个为游戏 **NPC** 生成语音的系统，该系统通过解析 wiki 内容并生成角色语音提示词，并集成 **ElevenLabs** 为每个角色生成三个候选语音，详见该项目的 [GitHub 仓库](https://github.com/Gielinor-Speaks/voiceover-mage)。
   - 该成员目前正在手动筛选语音，但目标是通过自动评审循环来实现自动化，并寻求关于利用 **DSPy** 优化和编译特性的建议，相关内容在[这段开发日志视频](https://youtu.be/z3DQm2PiKpo)中有所展示。
- **自动语音评审循环**：该语音生成系统的开发者计划添加一个**自动评审循环**以减少人工筛选，通过收集人工选择作为训练信号来创建示例。
   - 目标是让系统学习什么是针对不同角色原型的“优质”语音匹配，而无需对每个候选语音进行人工判断。
- **带有情感映射的 ElevenLabs 合成**：该系统利用 **ElevenLabs** 进行语音合成，并包含一个**情感映射层**，将游戏内动画 ID 与合成参数绑定。
   - 这使得同一个角色语音可以根据游戏内上下文听起来愤怒、快乐或恐惧。
- **寻求 DSPy 优化技巧**：该成员正在寻求关于构建角色分析流水线以及更有效地利用 **DSPy 优化和编译特性**的建议，希望能提高生成语音的主观质量判断。
   - 由于该游戏的 subreddit 社区对 AI 持反对态度（*anti-AI*），缺乏社区支持，因此该成员将此视为利用 **DSPy** 进行学习的机会。


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1430274415668494556)** (2 messages): 

> `DSPy, ArXiv Papers` 


- **DSPy 亮相新 ArXiv 论文**：一位成员分享了一篇新论文的 [ArXiv 链接](https://arxiv.org/abs/2510.13907v1)，该论文在其仓库中使用了 **DSPy**。
   - 然而，该论文的实际代码尚未发布。
- **代码尚未发布**：上述提到的 ArXiv 论文虽然使用了 **DSPy**，但实际代码尚未公开。
   - 成员们正热切期待代码发布，以便研究实现细节。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1430278974042542276)** (19 messages🔥): 

> `Microsoft Trace vs DSPy, Async DSPy Modules for OCR, Enforcing Providers with OpenRouter, AI & Blockchain Engineer Specialization` 


- **微软 Trace 程序声称准确率优于 DSPy**：根据一位成员分享的截图，微软的 [Trace](https://microsoft.github.io/Trace/) 程序声称比等效的 **DSPy** 程序**准确率提高了 8%**。
   - 另一位成员对此表示关注，计划对其进行测试以进行公平比较，但预计 **DSPy** 能保持更细粒度的控制。
- **异步 DSPy 模块应对 OCR 挑战**：一位成员询问了用于并行执行的 **DSPy 模块异步版本**，特别是针对 OCR 任务。
   - 另一位成员确认了异步能力，并分享了他们在涉及 **Google Cloud Vision**、**DSPy Attachments** 库和通过 bboxes 进行布局的高吞吐量任务中遇到的挑战，提到了重复循环的 bug，并正在探索 **paddleocr-vl** 或 **nanonets** 等替代方案。
- **OpenRouter 提供商强制指定仍是难题**：一位成员询问如何在 **DSPy** 中使用 **OpenRouter** 时**强制指定特定的提供商**。
   - 目前的对话中尚未提供解决方案。
- **AI 与区块链工程师展示 Agent 架构实力**：一位成员介绍自己是**高级 AI 与区块链工程师**，专注于在 AI 与区块链的交汇点构建智能自主系统。
   - 他们的专业领域包括**链上 AI Agent**、**带有实时数据的 LLM 流水线**、**使用 LangChain/AutoGen 进行 AI Agent 编排**，并拥有在 **Base**、**Solana** 和 **EVM 链**等多个区块链平台上的经验。
- **可训练装饰器（Trainable Decorator）备受青睐**：一位成员表达了对可训练装饰器的喜爱。
   - 目前尚不清楚具体指哪个可训练装饰器，但他们似乎对此非常兴奋，简单地评价道：*“噢，我喜欢这个可训练装饰器！非常棒的主意。”*


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1430269558203224124)** (20 messages🔥): 

> `Sora video, Nous research help, GPT-3.5 Sonnet, GPT-5 personality change, GPT-4o voice mode` 


- ****Sora** 视频亮相！**: 一位成员分享了使用 **Sora** 制作的视频，展示了其强大的功能，[点击此处查看](https://cdn.discordapp.com/attachments/1149866623109439599/1430403237084659774/20251022_0850_01k850gmktfant3bs18n3hbd79.mp4?ex=68fa4f13&is=68f8fd93&hm=ac5d3464bd020d568c770a6fa89ef8ccb2db40420fd45bee25dc3be64c60807f&)。
- **寻求 **Nous** Research 协助！**: 一位成员请求 Nous Research 协助完成一篇旨在帮助对 AI 感兴趣的青少年的研究论文。
- **再见，**3.5 Sonnet**！**: 一位成员分享了来自 fixvx.com 的链接（[点击此处](https://fixvx.com/dbreunig/status/1980733694634770710)），对 **GPT-3.5 Sonnet** 的终结表达了一种告别感。
- ****GPT-5** 的冷遇！**: 一位成员分享了基准测试结果，指出 **GPT-5** 变得不那么友好了，并将其归因于关于“谄媚性”（sycophancy）的辩论。
- ****GPT-4o** 依然能完美驾驭牙买加口音！**: 一位成员指出 **GPT-4o** 在语音模式下依然能出色地模仿牙买加口音，而 **GPT-5** 虽然声称可以，但未能改变音色。
   - 他们表示这是他们关心的*少数几个指标*之一。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1430338693037691073)** (2 messages): 

> `Microsoft Trace, Microsoft Winsock` 


- **发现 Microsoft Trace 工具**: 一位成员分享了 [Microsoft Trace](https://microsoft.github.io/Trace/) 的链接，这是一个实用工具，并提到*显然它并不全是新鲜玩意*。
- **用户态下的 Winsock Kernel？**: 另一位成员分享了一个关于在 **Userspace** 运行 [Winsock Kernel](https://github.com/microsoft/Windows-driver-samples/tree/main/network/winsock/userspace) 的项目，由 Microsoft 实现。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1430281050692648983)** (19 messages🔥): 

> `Kimi support, Moderato and Allegretto, K2 on together, partnership with Kimi` 


- **Kimi 被指“零支持”**: 一位成员抱怨称 **Kimi** 得到了*零*支持，且未收到支持团队的任何回复。
   - 另一位成员澄清说这*不是*一个支持服务器，并建议私信特定用户。
- **Moderato 和 Allegretto 付费方案**: 一位成员表示有兴趣升级到 **Moderato** 或 **Allegretto**，但找不到关于在付费方案中可以使用多少次 **OK Computer** 的信息。
   - 另一位成员分享说 **Moderato** 每月允许使用 **20** 次，并附上了[相关推文](https://x.com/togethercompute/status/1980337943169651055)链接。
- **K2 在 Together 上运行极快**: 一位成员评论了 **K2** 在 **Together** 上的运行速度。
   - 未提供进一步信息。
- **关于与 Kimi 合作的讨论**: 一位成员询问应该私信谁来商讨与 **Kimi** 的合作事宜。
   - 另一位成员建议私信特定用户。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1430377338834522247)** (2 messages): 

> `Mojo features, Mojo missing type system, Mojo standard library datatypes, Mojo async runtime, Mojo effect system` 


- **Mojo 的类型系统：仍在开发中**: 一位成员表示 **Mojo** 目前最缺失的是一个*完善的类型系统*。
- **Mojo 的愿望清单项**: 除了类型系统，一位成员还提到了 **Mojo** 应该具备的其他功能：完善**标准库数据类型**（standard library datatypes）、**原生 IO**、良好的 **async runtime**、**effect system**、**静态反射**（static reflection）、**编译器插件**（compiler plugins）、处理更具**限制性目标**（restrictive targets）的能力、**集群计算**（cluster compute）、**设备/集群建模**（device/cluster modeling），以及某种 **Erlang OTP** 的克隆版。


  

---

### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1430593668888264704)** (13 messages🔥): 

> `H100 上 Llama-3.1 的 Segfault，GGUF 与 bfloat16 模型对比，tensor_internal 包重命名为 tensor` 


- **Llama-3.1 在 H100 上使用 Nightly Build 时出现 Segfault**：一名成员报告在 **H100** GPU 上使用 Nightly Build 加载 `modularai/Llama-3.1-8B-Instruct-GGUF` 时出现 Segfault，并将问题缩小到 *llama3/model.py* 中的 `session.load` 函数。
   - 该问题似乎是 GPU 特有的，因为模型在 CPU 上运行正常，特别是在使用 `modular/max-nvidia-full:nightly` 基础镜像时，但推送到 GPU 时会发生 Segfault。
- **GGUF 模型引发 Segfault 推测**：一名成员询问 Segfault 是否在 **bfloat16** 模型中依然存在，并指出 **GGUF**（*q4_k* 等）权重的反量化算子（dequantization kernels）目前仅针对 CPU 执行构建。
   - 该成员确认，虽然 CPU 运行的是 **q4_k** 版本，但 GPU 尝试运行 **bf16** 版本，从而导致了 Segfault。
- **tensor_internal 演变为 tensor**：一名成员强调了 **Mojo API** 最新 Nightly 版本中的一项关键更新：`tensor_internal` 包已重命名为 `tensor`（[Discord 链接](https://discord.com/channels/1087530497313357884/1224434323193594059/1430580184305635339)）。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1430290731641340057)** (6 messages): 

> `Vulkan，正弦函数，渲染器，超越函数` 


- **Vulkan 不准确的正弦函数导致错误**：**Vulkan sine 函数的不准确性**会导致错误，因此需要自定义正弦函数，但这可能会降低速度。
   - 一名成员报告由于此问题导致 *未能通过几个错误测试*。
- **Tinygrad 的渲染器要求**：如果你的渲染器没有实现 **sin/log/exp**，**tinygrad/uop/decompositions.py** 会自动触发。
   - 一名成员之前没有意识到会发生这种情况，并相应地修改了他们的渲染器，以确保所有**超越函数（transcendental functions）**都能通过测试。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1430668420646572123)** (2 messages): 

> `BEAM, JITBEAM, jitted kernels` 


- **BEAM 与 JITBEAM 概念澄清**：一名成员询问 **BEAM** 和 **JITBEAM** 之间的区别，并质疑两者是否都可用。
   - 另一名成员澄清说，**JITBEAM** 特指 jitted kernels 的 **BEAM**。
- **JITBEAM**：它是 jitted kernels 的 BEAM。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1430427902549295135)** (6 messages): 

> `Aider 用于基准测试，使用本地模型编码，评估日志与 LLM 历史记录，排行榜排名，Uniswap Portal 垃圾信息` 


- **Aider 被滥用于基准测试**：一名成员提到他们*从未将 Aider 用于其原始目的，仅用于基准测试。*
   - 他们澄清说，他们一直只使用 **Cline** 和 **RooCode** 等本地模型（Local models）进行编码。
- **LLM 历史记录困扰**：一位用户正在寻找*评估日志和 LLM 历史记录*，并澄清他们指的是*完整对话*。
   - 另一名成员确认他们最近在检索这些日志方面*也没有成功*。
- **Uniswap Portal 垃圾信息**：一名用户分享了一个看起来很可疑的链接（[uniswap-portal.web.app](https://uniswap-portal.web.app)）。
   - 一个图片附件被图像分析标记为**垃圾信息（spam）**。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1430541841698127902)** (1 messages): 

> `项目维护，版本更新` 


- **项目维护状态查询**：鉴于自 **8 月 10 日**以来缺乏版本更新且活动有限，一名成员询问了项目的维护状态。
   - 目前尚未对项目状态给出答复。
- **无更新**：在问题提出后，没有提供任何更新。