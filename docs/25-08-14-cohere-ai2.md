---
companies:
- openai
- perplexity-ai
- ai2
- nvidia
- cohere
- meta-ai-fair
- google
- hugging-face
- ollama
- unsloth
date: '2025-08-14T05:44:39.731046Z'
description: '以下是为您翻译的中文内容：


  *   **OpenAI 的 GPT-5** 完成《宝可梦 红》（Pokemon Red）速通的速度比 **o3** 快了 3 倍。

  *   **Perplexity** 以 **200 亿美元**的估值融资 **2 亿美元**。

  *   **AI2**（艾伦人工智能研究所）获得了 **7500 万美元**的 NSF（美国国家科学基金会）拨款以及来自 **NVIDIA** 的 **7700
  万美元**，用于支持 Olmo 和 Molmo 等 AI 基础设施项目。

  *   **Cohere** 融资 **5 亿美元**，并从 **Meta AI FAIR** 聘请了 **Joelle Pineau**，以助力 Command
  A 等模型的提升。

  *   **谷歌**发布了 **Gemma 3 270M** 端侧微型大语言模型，配备了 INT4 QAT（量化感知训练）检查点和大型嵌入表；同时正式开放 **Imagen
  4**，其快速版价格为每张图 0.02 美元。

  *   **Meta AI FAIR** 推出了 **DINOv3**，这是一系列自监督视觉基础模型，具有高分辨率密集特征，在 COCO 检测和 ADE20K
  分割等基准测试中表现强劲，并采用宽松的许可协议。

  *   **MiniMax AI 智能体挑战赛**正在进行中，总奖金 **15 万美元**，设有 200 多个奖项，鼓励在 8 月 25 日前构建 AI 项目。'
id: MjAyNS0w
models:
- gpt-5
- o3
- command-a
- gemma-3-270m
- imagen-4
- dinov3
people:
- joelle_pineau
- fchollet
- awnihannun
- _philschmid
- osanseviero
title: 西方开源模型公司获融资：Cohere 以 68 亿美元估值融资 5 亿美元，AI2 获 NSF（美国国家科学基金会）与英伟达 1.52 亿美元资助。
topics:
- model-speed
- funding
- ai-infrastructure
- on-device-ai
- quantization
- embedding-models
- image-generation
- self-supervised-learning
- vision
- dense-prediction
- benchmarking
- instruction-following
- model-optimization
- model-release
- challenge
---

**对开源模型的资助就是我们所需要的一切。**

> 2025年8月13日至8月14日的 AI 新闻。我们为您检查了 12 个 subreddits、544 个 Twitter 和 29 个 Discord（227 个频道和 9744 条消息）。预计节省阅读时间（以 200wpm 计算）：710 分钟。我们的新网站现已上线，提供完整的元数据搜索和精美的 vibe coded 呈现方式，涵盖所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上给我们反馈！

祝贺 [GPT5 速通《宝可梦 红》](https://www.reddit.com/r/singularity/comments/1mq2irv/gpt5_just_finished_pokemon_red/) 的速度比 o3 快 3 倍，以及 Perplexity 以 [200 亿美元估值融资 2 亿美元](https://x.com/arfurrock/status/1955740969116299466?s=46)，但今天属于开源模型团队，他们本周宣布了巨额资金注入：

- [**Ai2 宣布获得 7500 万美元的 NSF 资助和来自 NVIDIA 的 7700 万美元**](https://allenai.org/blog/nsf-nvidia)，以继续构建 Olmo 和 Molmo 等关键 AI 基础设施（另见 Nathan Lambert 的 [美国真正开源模型项目](https://x.com/natolambert/status/1952370970762871102)）
- [**Cohere 在新一轮融资中筹集了 5 亿美元，并聘请了 Joelle Pineau**](https://x.com/aidangomez/status/1955993896590152114)，她是前 META FAIR 的负责人，这对于 [Command A 等模型](https://news.smol.ai/issues/25-03-17-ainews-coheres-command-a-claims-3-open-model-spot-after-deepseek-and-gemma) 来说大概是个好消息。

今天，正义的一方获胜了。

---

[](https://resend-attachments.s3.amazonaws.com/SiP8BRwkgadkEqG)

🚀 **150,000 美元 MiniMax AI Agent 挑战赛** — 展现你的最高水平！

- 💡 从零开始构建或重混项目 — 200 多个**奖项**等你来拿。
- 🗓 **8 月 25 日前提交** → https://minimax-agent-hackathon.space.minimax.io/
- 不要只是想象你能用 AI 构建什么 — **证明它**。
- 更多详情请见官方 Luma 页面 https://lu.ma/2u17h1zw

---

# AI Twitter 摘要

**Google 的 Gemma 3 270M 模型和 Imagen 4 Fast**

- **Gemma 3 270M（端侧微型 LLM）**：Google 发布了一个 270M 参数的 Gemma 3 模型，具有强大的指令遵循能力和开源权重。它专为“超高效”本地使用而设计，具有 INT4 QAT 检查点和大型嵌入表（约 1.7 亿个 embedding 参数，约 1 亿个 Transformer 参数），在其规模下展现出令人惊讶的能力。它已经广泛出货：
    - 跨技术栈在端侧运行：来自 [@fchollet](https://twitter.com/fchollet/status/1956059444523286870) 的 KerasHub 预设，MLX 首日支持，通过 [@awnihannun](https://twitter.com/awnihannun/status/1956053493216895406) 的 MLX-LM 在 M4 Max 上以 4-bit 运行速度约为 650 tok/s，且占用内存 <200MB（在相同速度下具有 DWQ 质量提升 [后续更新](https://twitter.com/awnihannun/status/1956089788240728467)），Ollama 一键运行 ([ollama run gemma3:270m](https://twitter.com/ollama/status/1956034607373222042))，来自 Unsloth 的动态 GGUF（文档 + 手机上约 50 tok/s）通过 [@UnslothAI](https://twitter.com/UnslothAI/status/1956027720288366883)，以及 [@ggerganov](https://twitter.com/ggerganov/status/1956026718013014240) 强调的 Hugging Face 集合。它甚至可以在 Pixel 7a 上运行（[演示](https://twitter.com/1littlecoder/status/1956065040563331344)）。官方公告：[@googleaidevs](https://twitter.com/googleaidevs/status/1956023961294131488)，详情：[@_philschmid](https://twitter.com/_philschmid/status/1956024995701723484)，概览：[@osanseviero](https://twitter.com/osanseviero/status/1956024223773663291)。
    - 显著的设计权衡：超过一半的参数位于 embedding 中（[观察](https://twitter.com/code_star/status/1956033343465906379)），这可能有助于在微小规模下提高词汇量/覆盖率；训练 Tokenization 细节正在讨论中。
- **Imagen 4 GA + Imagen 4 Fast**：Google 将 Imagen 4 正式发布，并推出了“Imagen 4 Fast”，价格约为每张图片 0.02 美元 —— 适用于大规模或交互式工作流（[公告](https://twitter.com/googleaidevs/status/1956035672197771479)）。

**Meta 的 DINOv3：大规模高分辨率密集视觉特征（许可宽松）**

- **DINOv3 (自监督视觉基础模型)**：Meta 推出了一个使用 SSL 训练的模型系列，能够产生高分辨率的密集特征，并在长期存在的密集预测任务上超越了专门的系统——通常只需使用冻结的 backbone：
    - 报告的结果包括：在冻结 backbone 的情况下 COCO 检测达到 66.1 mAP，ADE20K 线性 55.9 mIoU（配合解码器为 63.0），NAVI 上的 3D 对应关系召回率为 64.4，以及 DAVIS 上的视频追踪达到 83.3 J&F（[指标线程](https://twitter.com/BaldassarreFe/status/1956027867860516867) 和 [后续](https://twitter.com/BaldassarreFe/status/1956027888051892594)）。发布公告：[@AIatMeta](https://twitter.com/AIatMeta/status/1956027795051831584)，首日即支持 Transformers ([HF 帖子](https://twitter.com/AIatMeta/status/1956027800500232525))。
    - 此次发布涵盖了超过 12 个不同尺寸的 ConvNeXT/ViT 模型，在多样化（包括卫星）数据上训练，并采用宽松的许可证——迅速被作为 backbone 直接替换采用（例如，接入 VGGT 流水线以获得即时的类 SOTA 提升；[@maxseitzer](https://twitter.com/maxseitzer/status/1956029421602623787)）。总结：[@mervenoyann](https://twitter.com/mervenoyann/status/1956033306580877406)。

**前沿模型能力与效率：GPT‑5, FormulaOne, DetailBench, GFPO**

- **GPT‑5 的实际表现**：
    - 《精灵宝可梦 红》：根据 [@Clad3815](https://twitter.com/Clad3815/status/1955980772575268897) 和 [@scaling01](https://twitter.com/scaling01/status/1955813023735828587) 的测试，GPT‑5 以 6,470 步完成了游戏，而 o3 为 18,184 步（快了约 3 倍），且幻觉更少，空间规划能力更强。注意：不同运行版本在工具访问（如 Google Search）上可能存在差异，会影响可比性 ([@kiranvodrahalli](https://twitter.com/kiranvodrahalli/status/1956044490885751273))。
    - 医疗问答：在开启高推理开销的情况下，GPT‑5 在高质量眼科学问答数据集上接近完美准确率 ([论文链接](https://twitter.com/omarsar0/status/1956003145349521780))。
- **新的推理基准测试**：
    - FormulaOne（专家级动态规划）：在“浅层”层级：顶尖模型得分 50–70%；“深层”层级：Grok/Gemini/o3/Opus-4 解决率 ≤1/100，GPT‑5 Pro 解决率为 4/100；“最深层”层级：所有模型均为 0% ([@shai_s_shwartz](https://twitter.com/shai_s_shwartz/status/1955968602978320727))。
    - DetailBench（在未被要求的情况下发现细微错误）：展现了一种与指令遵循正交的能力——某些模型的表现优于 GPT‑5，因为 GPT‑5 可能会过度顺从既定任务 ([@xeophon_](https://twitter.com/xeophon_/status/1956025495515979984))。
- **推理 Token 效率与训练技术**：
    - Token 效率差距：在相同任务上，开源模型输出的 Token 数量通常比闭源模型多 1.5–4 倍（在简单查询中甚至高达 10 倍），这削弱了单 Token 价格优势；来自 [@NousResearch](https://twitter.com/NousResearch/status/1956090990005248341) 的详细行业综述及 [@scaling01](https://twitter.com/scaling01/status/1956098555090714668) 的评论。
    - GFPO (Group Filtered Policy Optimization)：通过在训练期间采样更大的组并根据长度和 reward-per-token 进行过滤，减少“长度膨胀”。在 Phi‑4‑reasoning 14B 上，GFPO 比 GRPO 缩短了 46–71% 的长度；优化 reward/tok 可将缩减幅度提升至 71–85%，同时保持在 AIME24/25, GPQA, Omni‑MATH, LiveCodeBench 上的准确率 ([摘要 + 总结](https://twitter.com/iScienceLuvr/status/1955955524790575212))。
    - 高效解码与摊销测试时计算：OverFill（全量模型用于 prefill → 剪枝后的密集模型用于 decode）以极低的延迟提升质量 ([代码+摘要](https://twitter.com/iScienceLuvr/status/1955965909409120476))；噪声超网络 (Noise Hypernetworks) 取代扩散模型中奖励引导的测试时噪声优化，以极低的成本恢复大部分质量 ([摘要](https://twitter.com/iScienceLuvr/status/1955958029993828724))；以及带有时间投票/强化的扩散 LLM，利用轨迹中段的一致性 ([线程](https://twitter.com/iScienceLuvr/status/1955964748341919862))。

**开放生态系统、规模与基础设施**

- **AI2 获得 1.52 亿美元用于开源模型**：7500 万美元来自 NSF + 7700 万美元来自 NVIDIA，用于扩展开源模型生态系统（OLMos, Molmos 等）以及可复现的科学 AI ([@allen_ai](https://twitter.com/allen_ai/status/1955966785175388288))。来自 [@natolambert](https://twitter.com/natolambert/status/1955986546626322479) 的背景信息：这一单项支出约占 NSF 2026 年 AI 预算的 20%；NVIDIA 正在提供领先的硬件。[@HannaHajishirzi](https://twitter.com/HannaHajishirzi/status/1955984650599325808) 的感想。
- **Cohere 融资 5 亿美元；Joelle Pineau 加入担任首席 AI 官**：专注于企业和政府、安全/主权 AI；同时欢迎新任 CFO ([@cohere](https://twitter.com/cohere/status/1955993354745082336), [@aidangomez](https://twitter.com/aidangomez/status/1955993896590152114), [@jpineau1](https://twitter.com/jpineau1/status/1955995736895594838), [@nickfrosst](https://twitter.com/nickfrosst/status/1956005330069983332))。
- **吞吐量与编译器更新**：Modal 展示了快速的 GPU 扩展：约 12 秒扩展 100 张 H100，约 4 分钟扩展 300 张，并可扩展至 1000+ 张 ([@bernhardsson](https://twitter.com/bernhardsson/status/1956073789550420330))。[@ezyang](https://twitter.com/ezyang/status/1955820298907082876) 撰写的“torch.compile 现状（2025年8月）”一文广为流传（因其在模型启动延迟/冷启动改进方面的表现，以及 Qwen-Image 区域编译等技巧而备受关注）。
- **工具**：TRL 添加了原生 VLM 后训练支持 ([@QGallouedec](https://twitter.com/QGallouedec/status/1956066332488950020))；vLLM 为亚马逊的 Rufus 助手提供动力 ([@vllm_project](https://twitter.com/vllm_project/status/1956116150259212619))。

**Agent：模拟、深度研究与浏览器原生助手**

- **聊天机器人的模拟优先评估**：Guardrails 推出了 Snowglobe，这是一个用于在生产前测试和改进机器人的用户模拟引擎 ([发布](https://twitter.com/ShreyaR/status/1956023326721368337))，获得了 [@goodfellow_ian](https://twitter.com/goodfellow_ian/status/1956040393361121540) 的称赞，并通过自动驾驶类比构建了 Agent 可靠性框架 ([@apoorvapandhi](https://twitter.com/apoorvapandhi/status/1956033885126468050))。
- **深度研究 Agent**：LangChain 发布了关于构建具有持久性/可观测性的长期运行多 Agent 研究系统的免费 LangGraph 课程 ([@LangChainAI](https://twitter.com/LangChainAI/status/1956027411302375631), [@hwchase17](https://twitter.com/hwchase17/status/1956036358709108979))。另外，“Elysia”展示了具有决策树透明度、个性化反馈和按需分块流水线的 Agentic RAG ([@philipvollet](https://twitter.com/philipvollet/status/1955945448860008655))。
- **浏览器 Agent 与 Web 研究**：Perplexity 发布了 Comet 企业版——一个用于安全、链接工具工作流的“AI 驱动浏览器 Agent” ([@perplexity_ai](https://twitter.com/perplexity_ai/status/1956046685509210183))。与此同时，[@paraga](https://twitter.com/paraga/status/1956008857555099928) 通过 [@p0](https://twitter.com/p0/status/1956007609250492924) 揭晓了 Parallel 的“Web 第二用户”愿景，声称在深度 Web 研究基准测试中超越了人类和顶尖模型。

**交互式视频、机器人与多模态**

- **腾讯 Hunyuan-GameCraft（开源）**：一个基于 HunyuanVideo 的框架，用于生成高动态、可玩且具有物理真实感的视频游戏。将键盘输入统一到连续动作空间中以实现精确控制；采用混合历史调节以保持长期一致性；并使用 PCM 蒸馏来压缩推理（量化后的 13B 版本可在 RTX 4090 上运行）。公告中附有项目页面 + 代码 + 报告链接 ([@TencentHunyuan](https://twitter.com/TencentHunyuan/status/1955839140173631656))。
- **机器人**：一个令人愉悦的机器人手折叠衣服演示 ([视频](https://twitter.com/adcock_brett/status/1956021725491290154))，随后是一个严酷的提醒：类人手对于通用机器人至关重要——“其难度不亚于制造整个机器人” ([评论](https://twitter.com/adcock_brett/status/1956083802440450551))。

**热门推文（按互动量排序）**

- Meta 的 DINOv3：具有 SOTA 密集特征和 day‑0 HF 支持的 SSL 视觉主干网络 ([@AIatMeta](https://twitter.com/AIatMeta/status/1956027795051831584))
- Google Imagen 4 GA + “Fast” 版本，价格为 $0.02/每张图片 ([@googleaidevs](https://twitter.com/googleaidevs/status/1956035672197771479))
- Google 的 Gemma 3 270M 微型 LLM 发布及生态系统支持 ([@osanseviero](https://twitter.com/osanseviero/status/1956024223773663291), [@googleaidevs](https://twitter.com/googleaidevs/status/1956023961294131488))
- Gemini App 为 “2.5 Deep Think” 翻倍了速率限制（rate limits） ([@OfficialLoganK](https://twitter.com/OfficialLoganK/status/1955821580237594847), [@joshwoodward](https://twitter.com/joshwoodward/status/1955804081437696046))
- GPT‑5 Pokémon 性能（相比 o3 步数减少约 3 倍） ([@Clad3815](https://twitter.com/Clad3815/status/1955980772575268897), [@scaling01](https://twitter.com/scaling01/status/1955813023735828587))
- FormulaOne 基准测试：专家级动态规划仍然具有挑战性（即使对于 GPT‑5 Pro 也是如此） ([@shai_s_shwartz](https://twitter.com/shai_s_shwartz/status/1955968602978320727))

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. 小型语言模型的基准测试与普及

- [**google/gemma-3-270m · Hugging Face**](https://huggingface.co/google/gemma-3-270m) ([Score: 533, Comments: 204](https://www.reddit.com/r/LocalLLaMA/comments/1mq3v93/googlegemma3270m_hugging_face/)): **Google 在 Hugging Face 上发布了 Gemma-3-270M 模型（2.7 亿参数），将其定位为小型开源权重 LLM 的替代方案。该模型提供 bfloat16 (BF16) 精度，旨在实现高效推理、潜在的边缘硬件兼容性以及在资源受限环境中的实验。虽然此帖未强调架构、训练数据和基准测试的细节，但其较小的尺寸表明其适用于轻量级部署。** 评论者注意到由于命名（'270M' 对比 '270B'）引起的最初困惑，并讨论了 BF16 精度权重在实际实验中的效用。人们对该模型的部署潜力感兴趣，但对其在如此小的参数量下的能力仍持怀疑态度。
    - 一位用户指出，Gemma 家族中最小的模型（**270M 参数版本**）是在一个异常庞大的数据集（**6 万亿 tokens**）上训练的，这值得注意，因为即使是更大的模型（例如 1B, 4B, 12B, 27B）按比例使用的 tokens 也较少。这可能对小型模型的缩放（scaling）效率和泛化（generalization）质量产生重大影响。
    - 有人提到对在 Gemma 270M 模型中使用 **BF16 权重**感兴趣，这表明人们关注通过降低精度来实现高效推理和训练，这在部署场景中越来越常见。
- [**谁是上个月下载 bert 的 5700 万人？**](https://i.redd.it/vk2njmk01xif1.png) ([Score: 350, Comments: 109](https://www.reddit.com/r/LocalLLaMA/comments/1mpr0nc/who_are_the_57_million_people_who_downloaded_bert/)): **图片显示了 Google 'bert-base-uncased' 的 Hugging Face 模型页面，报告上个月下载量超过 5700 万次。关键技术背景：下载计数反映的是 *model pulls*（可能由自动化 CI/CD 流水线、研究人员和教育用户执行），而非唯一用户，这突显了 BERT 在各种 ML 设置中的持续主导地位和集成。该页面还展示了对不同 ML 框架（PyTorch, TensorFlow, JAX）的广泛支持以及活跃的社区参与（点赞和关注）。** 评论者澄清说，由于自动化系统、学生和研究人员的频繁重新下载，下载量并不等同于唯一用户——这强调了 BERT 在 NLP 工作流中根深蒂固的地位。
    - 几位评论者强调，BERT 的 5700 万次下载量可能反映了自动化系统（如 CI/CD 流水线）的重复下载，而不是唯一用户。这是热门开源模型中的普遍现象，因为许多工作流和组织经常重新下载模型以进行自动化评估、重新训练或部署。
    - BERT 仍然是 NLP 流水线中的基础工具，用于各种任务，包括分类、预测和生成 embeddings；它被包含在广泛使用的训练课程中（例如 Hugging Face 的课程），随着新的学习者和组织将其用于教育和实验目的，不断推动着高下载量。
    - 研究人员和学生经常将 BERT 用于研究任务、重排序（reranking）以及作为基准模型（baseline model），尽管出现了更现代的 NLP 架构，这进一步放大了其持续的相关性和高使用率统计数据。

- [**“缺失最新 Qwen 综合征”**](https://i.redd.it/z096hdwp01jf1.jpeg) ([Score: 251, Comments: 34](https://www.reddit.com/r/LocalLLaMA/comments/1mq8oyk/the_missing_latest_qwen_syndrome/)): **该图片是一张散点图，比较了各种 LLM 的模型大小（参数量）与 IFEval 评分，包括 Gemma 3 (270M, 1B)、SmolLM2 (135M, 360M)、Qwen 2.5 (0.5B) 和 Llama 3.2 (1B)。该帖子和评论强调了在基准测试比较中经常遗漏最新的 Qwen 模型，用户指出 Qwen3-0.6B 在 IFEval 上获得了 59.2 的稳健分数——在其参数规模下极具竞争力。这凸显了包含所有知名模型的重要性，以反映指令遵循评估中当前的 State-of-the-art 性能。** 评论者对小模型规模下 IFEval 分数的可靠性展开了辩论，有人将 50% 的性能水平比作随机概率。此外，对于对比图表中频繁遗漏 Qwen 模型的情况，用户表达了明显的挫败感，暗示这导致了对小规模 LLM 领域的看法存在偏差或不完整。
    - 多位用户强调，尽管 Qwen3-0.6B 的规模相对较小（2.7 亿参数），但其 IFEval 分数在 50% 到 59.2% 之间，这在其参数量级下被认为是非常强劲的。这种性能指标被视为该尺寸范围内模型中极具竞争力的。
    - 讨论强调，这些较小的模型（如 Qwen3-0.6B）通常不是为广泛的开箱即用场景设计的，而是旨在针对特定任务进行 Fine-tune。建议通过定制和进一步训练，以在目标应用中发挥其全部潜力。

### 2. 即将发布及开源的 AI 模型 (Grok 2, DeepSeek)

- [**提醒一下，Grok 2 应该会在明天左右开源（根据 Musk 上周的推文）。**](https://i.redd.it/hsaoxskfs1jf1.jpeg) ([Score: 282, Comments: 82](https://www.reddit.com/r/LocalLLaMA/comments/1mqctep/just_a_reminder_that_grok_2_should_be_released/)): **该图片捕捉了一段 Twitter 对话，其中一名用户询问开源 Grok 2 和 Grok 3 的时间表，并引用了 Elon Musk 此前关于 xAI 打算开源其模型的声明。Elon Musk 回复称，由于团队正在进行相关工作，Grok 2 将于下周开源。这张截图强调了 Grok 2 开源版本的预期发布计划，意味着它可能很快就会向公众开放。** 评论者对承诺的发布日期持怀疑态度，提到了 Musk 过去经常给出过于乐观的时间表，并对模型的实际发布表示愤世嫉俗。其他人则将 xAI 的开源承诺与 OpenAI 目前的封闭做法进行了对比，尽管考虑到行业趋势的变化，这种对比带有讽刺意味。
    - 一位用户质疑开源 Grok 2（甚至 Grok 3）的技术相关性，将其描述为“一堆巨大的陈旧垃圾”，并暗示即使发布 Grok 3，对社区的效用也有限。这反映了人们对这些模型与当前最先进（State-of-the-art）产品相比，在架构和性能竞争力方面的广泛怀疑，暗示除非开源版本能达到或超过现代 Benchmarks，否则它们的实际价值可能微乎其微。
- [**DeepSeek 的下一代 AI 模型因尝试使用国产芯片而推迟**](https://www.ft.com/content/eb984646-6320-4bfe-a78d-a1da2274b092) ([Score: 513, Comments: 111](https://www.reddit.com/r/LocalLLaMA/comments/1mpu8ot/deepseeks_next_ai_model_delayed_by_attempt_to_use/)): **DeepSeek 在华为 Ascend 处理器上未能完成训练后，推迟了其 R2 AI 模型的发布。据报道，这是由于技术限制，如稳定性差、芯片间互连（inter-chip connectivity）缓慢，以及与 Nvidia 的产品相比软件栈（software stack）较差。该公司转而使用 Nvidia GPU 进行训练，同时尝试保留 Ascend 芯片用于推理（inference），这说明了中国在实现 AI 硬件自主化方面面临的持续挑战。额外的延迟归因于数据标注周期延长；与此同时，阿里巴巴的 Qwen3 等竞争对手据报道正在采用 DeepSeek 的推理能力训练算法，但提高了其效率。** 评论者指出，尽管存在延迟，但中国在国产芯片上取得成功具有战略重要性，一些人对信息的可靠性或来源表示怀疑（例如，依赖《金融时报》的匿名消息源）。此外，还有关于采用国产硬件导致的延迟是否仍能为中国 AI 的独立性和行业增长提供长期优势的辩论。
    - DeepSeek 尝试在华为 Ascend 芯片上训练其即将推出的 R2 模型时遇到了重大技术障碍，包括持续的训练失败、稳定性问题、较慢的芯片间互连以及与 Nvidia 硬件相比劣势明显的软件。这一技术差距导致他们使用 Nvidia 芯片进行训练，同时继续尝试将 Ascend 用于推理，这说明了中国在高级 AI 工作负载实现硬件自主化方面面临的持续挑战。
    - R2 发布延迟的部分原因是数据标注时间长于预期，以及与国产芯片尚未解决的技术集成问题。尽管有华为工程师的现场支持，DeepSeek 仍无法在 Ascend 硬件上成功完成 R2 的训练运行，这突显了中国替代方案在关键 AI 训练任务中目前仍落后于 Nvidia 等美国解决方案。
    - 行业分析师指出，阿里巴巴的 Qwen3 模型采用了 DeepSeek 的核心训练算法，但实现效率更高，这表明中国 AI 实验室之间存在快速的技术知识转移。人们认识到，虽然华为的 AI 生态系统正面临“成长的烦恼”，但随着生态系统的成熟，在国产芯片上训练领先模型在未来可能会变得可行。

### 3. AI 模型部署中的硬件与实际挑战 (Qwen 上下文与 GPU)

- [**铭瑄 (MaxSun) 配备 48GB 显存的 Intel Arc Pro B60 双 GPU 显卡据传将于下周开始发货，售价 1,200 美元**](https://videocardz.com/newz/maxsun-arc-pro-b60-dual-with-48gb-memory-reportedly-starts-shipping-next-week-priced-at-1200) ([Score: 353, Comments: 138](https://www.reddit.com/r/LocalLLaMA/comments/1mpxumt/maxsuns_intel_arc_pro_b60_dual_gpu_with_48gb/)): **铭瑄正推出配备 48GB 显存的 Intel Arc Pro B60 双 GPU 显卡，售价 1,200 美元，据报道将于下周发货。该显卡需要主板支持 PCIe 插槽拆分 (bifurcation)，因为它缺乏板载 PCIe hub 芯片，这意味着在没有原生拆分功能的系统上（通常仅限高端工作站/服务器主板，或具有完整 x16 信号的 Xeon/Threadripper 平台）无法同时访问两个 GPU。完整规格可在 [官方产品页面](https://www.maxsun.com/products/intel-arc-pro-b60-dual-48g-turbo) 查看。** 显著的技术争论集中在显卡可能的供应受限、高昂的零售价以及对 PCIe 拆分的关键要求，这使得该卡对于主流台式机并不实用。用户建议等待独立评测以评估实际性能和兼容性。
    - 一个关键的技术注意事项是，铭瑄 Intel Arc Pro B60 双 GPU 需要 *主板 PCIe 拆分支持*，因为它不像之前的许多双芯 GPU 那样包含板载 PCIe hub 芯片。这意味着该卡只能在提供完整 x16 通道的顶部 PCIe x16 插槽中正常工作，除非使用在多个插槽上提供完整 x16 通道的高端平台（如 Xeon 或 Threadripper），否则无法在次级插槽中工作。
    - 人们对基准测试和兼容性评测充满期待，特别是为了确定像 ollama 和 llama.cpp（在消费级 GPU 上最大化 ML 性能的流行框架）这类框架是否能完全支持双 Arc Pro B60，考虑到其不寻常的配置和显存设置（两个 GPU 共 48GB）。
- [**100 万上下文是个骗局，AI 在 90k 之后开始产生幻觉。我正在使用 Qwen CLI，在使用了 10% 的上下文窗口后它就变得很垃圾**](https://www.reddit.com/r/LocalLLaMA/comments/1mq19x6/1_million_context_is_the_scam_the_ai_start/) ([Score: 232, Comments: 84](https://www.reddit.com/r/LocalLLaMA/comments/1mq19x6/1_million_context_is_the_scam_the_ai_start/)): **该帖子强调了 LLM 有效上下文窗口的实际限制，特别引用了 Qwen 的 CLI：模型在处理约 90-100k token 后，输出质量会下降并产生“幻觉”——远低于声称的“100 万”上下文。虽然基准测试很少捕捉到这一点，但用户体验表明，一旦使用了约 10% 的窗口，性能就会大幅下降。链接的基准测试和截图进一步证实了某些模型在超过 200k token 后的质量退化。** 评论者指出，上下文窗口的限制因模型和硬件而异：Gemini Pro 被认为能够处理 `200-300k` 上下文 token，并保持稳定、高质量的性能，尤其是在使用 Google 的基础设施时。共识是，与 Gemini 等基于云的解决方案相比，本地模型在大上下文下的表现挣扎得更厉害。
    - 多位用户报告称，随着上下文窗口接近 90–200k token，Qwen 的性能显著下降，表现为幻觉增加和召回率降低。链接的视觉基准测试说明了这一趋势，并突出了上下文长度敏感性问题（参见：https://preview.redd.it/cpeii3wpqzif1.png）。
    - 相反，一些用户指出 Google 的 Gemini Pro 模型在极大的上下文（高达 1.5M–2M token）下仍能保持强劲性能。有说法称 Gemini Pro 可以高精度地处理和召回 200–300k token 范围内的文档信息，这表明其具有更优越的硬件或上下文管理策略。
    - 讨论的一个共同限制是，随着大多数 LLM 上下文窗口的增长，除非明确指示保留细节，否则模型容易忘记早期内容或提供不够全面的回复。这指向了跨模型长上下文保留的更广泛问题，而不仅仅是 Qwen。

## 较低技术性的 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. GPT-5 在《宝可梦 红》速通中击败了之前的模型

- [**GPT-5 刚刚通关了 Pokémon Red！**](https://i.redd.it/u0us03n8yzif1.png) ([分数: 1764, 评论: 172](https://www.reddit.com/r/singularity/comments/1mq2irv/gpt5_just_finished_pokemon_red/)): **GPT-5 在 6,470 步内、约 7 天的时间里完成了 Pokémon Red，表现显著优于 o3（前一版本，耗时 18,184 步和 15 天），同时也击败了 Claude 和 Gemini 等竞争模型。分享的图片展示了 GPT-5 的游戏通关画面、最终策略和行动的详细分解，以及展示其自主游戏决策的实时推理面板。这次运行展示了大型语言模型 Agent 在自主游戏效率、规划和推理可见性方面的实质性提升。** 评论指出，GPT-5 依赖于一种常见但并非最优的单只 Pokémon “硬带（hard carry）”策略，这与小孩子的游戏模式如出一辙，但无论如何它还是达成了目标。还有建议在更复杂、非线性的游戏（如 Zelda: Minish Cap）上进行基准测试，以更深入地评估推理能力。
    - 评论者讨论了该模型使用的游戏策略，特别注意到 GPT-5 专注于升级并在大多数战斗中使用单只 Pokémon，这与年幼或新手玩家采取的简单方法类似。这表明该模型能够在游戏中进行高效优化，但可能尚未展现出创造性或高级的战略深度。
    - 成功通关 Pokémon Red 引发了关于该游戏的架构和训练对 GPT-5 构成的挑战及新颖性的疑问。一位用户推测该游戏是否在模型的训练数据中，或者模型是否利用了之前的通关记录或记录在案的策略，这暗示了人们对真正的 Generalization（泛化）与 Memorization（记忆）之间关系的持续担忧。
    - 网友非正式地提出了一项技术基准测试：建议与更复杂或脚本化程度较低的游戏（如 Factorio Space Age）进行比较，将其作为对类 AGI 推理能力的更严苛测试，强调需要比已知解决方案或攻略的游戏更广泛、更动态的问题解决能力。
- [**GPT-5 在 Pokémon Red 中获得徽章的速度比 o3 快近 3 倍**](https://i.redd.it/dudpeygmjwif1.png) ([分数: 1477, 评论: 193](https://www.reddit.com/r/singularity/comments/1mpp7hy/gpt5_is_nearly_3x_faster_than_o3_at_earning/)): **图片通过折线图展示了 GPT-5 与 o3 在 Pokémon Red 中获取徽章的对比基准测试。GPT-5 仅用 6,018 步就获得了全部 8 枚徽章，比 o3 的 16,743 步快了近 3 倍——展示了在 long-horizon（长跨度）、agentic 任务性能方面的实质性提升。帖子和评论中的技术评论强调了 GPT-5 的显著优势：对长且复杂的按钮序列的稳健执行、对幻觉（Hallucination）的抵御能力（能从错误中迅速恢复）以及有效的游戏内策略。** 一项关键的技术讨论强调了模型评估方法的转变，主张将重点放在 long-horizon 和复杂任务的基准测试上，而非传统的狭隘指标。评论者强调了 GPT-5 改进的战略推理和操作稳健性，预示着 AI Agent 评估范式的改变。
    - 评论者指出，模型评估应转向包含 long-horizon、agentic 任务的表现，而不仅仅是传统的静态基准测试，因为像 "GPT-5" 这样的新模型在这些方面表现出了显著进步（例如在 Pokémon Red 中获得徽章的速度提升了 3 倍）。
    - 技术观察强调了 GPT-5 通过游戏菜单、导航和战斗执行复杂多步序列的能力，且幻觉持续时间显著减少；该模型在少量失败后能迅速从错误中“清醒过来”，并制定出出人意料的有效策略，在实际实时游戏中的表现优于 o3。
    - 在讨论 "GPT-5" 时，技术上需要明确具体对象，因为 GPT-5 System Card 提到了该标签下的至少六个不同模型，这使得基准测试和能力对比变得复杂。

- [**GPT-5 仅用 6,470 步就通关了 Pokémon Red！！**](https://i.redd.it/0yk4psfqh0jf1.png) ([得分: 610, 评论: 112](https://www.reddit.com/r/OpenAI/comments/1mq5hyy/gpt_5_completed_pok%C3%A9mon_red_in_just_6470_steps/)): **该图片记录了一条推文，庆祝 GPT-5（由 OpenAI 开发）以 6,470 步通关了 Pokémon Red——相比之前 18,184 步的记录有了显著提升。这一壮举展示了模型在游戏任务效率方面的显著进步，暗示了 GPT-5 内部更好的优化技术或增强的上下文理解能力。截图还显示了游戏结束时的对话及相关统计数据，证实了这一成就，并有助于对模型游戏效率进行技术分析。** 评论者讨论了模型训练数据预处理中可能的改进（例如，在运行前删除无关数据），并建议进行更复杂的挑战（如 Nuzlocke 挑战），以稳健地测试泛化能力和策略。
    - 一位用户提到，有信息显示在 GPT-5 尝试通关 Pokémon Red 之前，所有“无关的训练数据”都已被删除，这暗示可能通过数据集筛选来防止先前的记忆或在游戏自动化中的不公平优势。这引发了关于可复现性以及在基于模型的游戏基准测试中，事先接触类似内容所起作用的疑问。

### 2. 值得关注的新基准测试：GPT-5、Google 图像模型、SWE-bench

- [**我们在 7 月份最新的类 SWE-bench 任务上运行了 GPT-5、Claude Sonnet 4、Qwen3-coder 等模型——GPT-5 显然完胜！**](https://i.redd.it/99admdkaszif1.png) ([评分: 218, 评论: 65](https://www.reddit.com/r/OpenAI/comments/1mq1oyf/we_ran_gpt5_claude_sonnet_4_qwen3coder_and_more/)): **该图片是一个柱状图，总结了 2025 年 7 月 SWE-rebench 基准测试的结果，评估了包括 GPT-5（Medium 和 High）、Qwen3-Coder 和 Claude Sonnet 4 在内的多个 LLM 在 34 个去污染的真实 GitHub PR 任务上的表现。GPT-5-Medium 以 29.4% 的解决率领先，超过了 Claude Sonnet 4 (20.6%) 和 Qwen3-Coder (解决率为 22.4%，但 pass@5 为 32.4%，与 GPT-5-High 持平)。该基准测试因使用训练后的新鲜数据而备受关注，详见 [SWE-rebench 排行榜](https://swe-rebench.com/leaderboard) 及其 [HuggingFace 数据集](https://huggingface.co/datasets/nebius/SWE-rebench-leaderboard)。** 几位评论者质疑为何缺少像 Claude Opus 这样强大的商业模型，考虑到模型/配置访问权限的差异，这些结果的泛化性和公平性存疑，且仅基于 34 个任务的结果在统计学上是否显著（预计可能存在较大的误差幅度）。
    - 多位评论者对基准测试的有效性表示担忧，强调公共基准测试中使用的 GPT-5 或其他模型版本可能与用户实际访问的版本不同——这给模型性能声明带来了可复现性和透明度问题。
    - 一位技术读者注意到基准测试中意想不到的结果：具体而言，根据共享数据，“GPT-5 Medium”的表现优于“GPT-5 High”。此外，还有关于仅评估 34 个任务所固有的统计显著误差幅度的讨论，这会极大地影响对报告排名的信心。
    - 帖子中更广泛的情绪是对 AI 模型发布和竞争性基准测试整体可靠性的质疑，强调了在没有开放权重或可靠、通用的测试条件的情况下，对于模型已明确超越其他模型的说法，怀疑态度日益增加。
- [**Google 的新图像模型在创建准确图表方面击败了 OpenAI 的图像模型**](https://i.redd.it/xukz6lphhxif1.png) ([评分: 263, 评论: 39](https://www.reddit.com/r/Bard/comments/1mpspan/googles_new_image_model_beats_openais_image_model/)): **该图片展示了据称由不同 AI 模型生成的柱状图对比。左侧图表展示了 'SWE-bench Verified' 基准测试，对比了每个模型的 'Without thinking'（不带思考）和 'With thinking'（带思考）模式。右侧，Google 的 'GPT-5' 模型实现了更高的准确率（74.9% 和 52.8%），而 'OpenAI Q3' 和 'GPT-4.0' 则较低，直观地展示了 Google 改进的图表渲染准确性——这与帖子标题所称的 Google 模型在生成准确图表方面优于 OpenAI 一致。柱状图的视觉比例因与数值不符而受到批评。** 热门评论对图表柱状图的比例表示怀疑，质疑图表的质量控制和潜在的误导。有人呼吁明确基准测试方法，暗示关于评估公平性和 QA 流程的技术争论仍在继续。
    - 几位评论者强调了 OpenAI 图像模型 QA 流程中可察觉的失败，特别是引用了一个比例失调的演示图表，并担心明显的缺陷在发布前未被发现。这引发了对模型在依赖精确度的任务（如创建图表）中输出可靠性的质疑。
    - 对于使用图像生成模型创建图表存在怀疑，技术论点倾向于使用传统的 LLM 配合图表工具或库。其含义是，直接使用生成式图像模型是不必要的，且在生成结构化数据可视化方面可能不太准确。

### 3. AI 模型与平台功能发布：Claude Code 与 Gemma 3 270M

- [**Introducing Gemma 3 270M: The compact model for hyper-efficient AI**](https://developers.googleblog.com/en/introducing-gemma-3-270m/) ([Score: 150, Comments: 20](https://www.reddit.com/r/singularity/comments/1mqan83/introducing_gemma_3_270m_the_compact_model_for/)): **Google 发布了 Gemma 3 270M 模型，这是一款专为超高效设备端推理设计的紧凑型 LLM，针对智能手机等具有严格计算/功耗限制的使用场景。该模型以其极小的参数量（`270 million`）而著称，远低于大多数 SOTA LLM，使其定位于资源使用至关重要的离线边缘部署。更多信息请参阅[公告](https://www.reddit.com/r/LocalLLaMA/comments/1dtvjhy/introducing_gemma_3_270m_the_compact_model_for/)和示例基准测试。** 评论中的技术讨论褒贬不一：虽然一些人称赞微型模型在离线/特定应用任务中的实用性，但几位专家认为，该模型极小的尺寸使其推理和语言能力非常有限——仅适用于琐碎或玩具级任务，而非严肃的对话或推理工作负载（参见[批评性输出示例](https://i.imgur.com/YXN4MOr.png)）。
    - 一位评论者指出，Gemma 3 270M 模型适用于资源高度受限的设备（如智能手机），强调了其离线执行潜力和在注重隐私的边缘 AI 应用中的价值，尽管在通用推理和复杂性方面存在局限。
    - 一份批评性评估建议，该模型的性能目前仅限于非常简单的任务，提供的截图中显示了基础性错误，这表明目前此类紧凑型架构仍无法胜任最先进的推理和知识任务。
    - 人们对扩展变体表现出兴趣，特别是针对能充分利用 16GB GPU 容量的版本（被称为 'Gemma 3N'），这表明用户对更大但仍保持高效、能最大化消费级硬件能力以处理更复杂本地 AI 任务的模型存在需求。
- [**Introducing Gemma 3 270M: The compact model for hyper-efficient AI**](https://developers.googleblog.com/en/introducing-gemma-3-270m/) ([Score: 237, Comments: 59](https://www.reddit.com/r/Bard/comments/1mq59p9/introducing_gemma_3_270m_the_compact_model_for/)): **Google 宣布推出 Gemma 3 270M，这是一款拥有 2.7 亿参数的紧凑型大语言模型（LLM）——显著小于典型的十亿级参数 LLM——并针对“超高效”本地部署进行了优化。该模型旨在实现极速推理和轻量级工作负载，目标场景包括本地运行的助手和设备端应用调用，在这些场景下不需要全规模模型。** 评论者强调了在如此小的规模下实现可用性能的技术意义，讨论了其在本地任务执行、快速推理以及作为连接到更大、基于 API 的 LLM 的成本节约路由中介方面的潜力。
    - 几位评论者强调了 270M 参数规模下功能完备的 LLM 的技术意义，强调其由于资源占用减少且速度比大型模型更快，非常适合设备端本地执行。这种更小的占用空间意味着高效的任务自动化，例如调用应用程序和处理频繁但简单的任务，而无需将计算卸载到云端服务。
    - 特别关注将 Gemma 3 270M 用作通往更大、API 驱动的 LLM 的路由层，以优化成本和性能。通过有选择地仅将复杂查询卸载到昂贵的云端模型，同时在本地处理基础任务，用户期望能节省大量运营成本并实现更快的常用任务响应时间。
    - 讨论中涉及了一个链接的性能对比图，用户注意到该模型表现强劲并对其能力表示惊讶，认为 Gemma 3 270M 尽管尺寸紧凑但实现了高质量。这可能标志着 1B 参数以下模型在实际应用部署中的一个新里程碑。

- [**在 Claude Code 和 Claude 应用中推出两种全新的学习方式**](https://www.reddit.com/r/ClaudeAI/comments/1mq6h47/introducing_two_new_ways_to_learn_in_claude_code/) ([Score: 150, Comments: 20](https://www.reddit.com/r/ClaudeAI/comments/1mq6h47/introducing_two_new_ways_to_learn_in_claude_code/)): **Anthropic 为 Claude Code 推出了两种可交互配置的输出风格：`/output-style explanatory`（Claude 会解释架构选择并概述最佳实践）和 `/output-style learning`（一种交互式、回合制的结对编程模式，旨在作为教学工具）。“Learning” 风格此前仅限于 Claude for Education，现在已在整个应用范围内可用，支持在所有聊天会话中提供逐步指导。这些功能的文档可在 https://docs.anthropic.com/en/docs/claude-code/output-styles 查阅。** 一位教育工作者评论道，对 Claude 教学风格的精细控制可以促进更有效的课堂 AI 集成，减少对手动 Prompt Engineering 的需求。另一位用户询问是否可以将这两种模式结合起来，创建一个融合了解释与交互式指导的混合风格，突显了进一步增强功能的潜在领域。
    - 一位评论者指出 Claude 界面日益复杂，强调用户现在必须管理多个概念，如 Agent、命令、Prompt、Hook 和输出风格。他们建议整合这些元素，使体验不再碎片化且更易于使用，这将使技术和教育工作流同时受益。
    - 一位用户询问同时激活解释模式和协作编程模式的可行性。他们的目标是让 Claude 在解释其推理的同时，作为一个编程伙伴进行互动，将这种体验比作向一位特别啰嗦但具有启发性的老师学习，这表明了对 AI 工具中更灵活、多模态教学工作流的需求。

---

# AI Discord 摘要

> 由 gpt-5 生成的摘要之摘要的总结
> 

**1. 微型巨人与可靠基准测试**

- **Gemma-3-270M 走向微型化，旨在产生宏观影响**：**Google** 发布了 **Gemma 3-270M**，其权重约为 **300MB**，并在约 **6T tokens** 上进行了训练，定位用于端侧和受限部署，详情见官方 [Introducing Gemma 3-270M](https://developers.googleblog.com/en/introducing-gemma-3-270m/)。社区反映在特定任务（RAG、指令遵循）中早期表现参差不齐，一些测试注意到在极端情况下会出现如 *“一只狗……六条腿”* 的幻觉。
    - 工程师们讨论了目标硬件（可穿戴设备、游戏内 Agent、工具调用存根）以及微型模型的延迟优于原始智力的实际应用场景，参考了 [Gemma 3-270M 博客文章](https://developers.googleblog.com/en/introducing-gemma-3-270m/)。讨论强调，对于这种尺寸级别的模型，发布专门的 Fine-tune 版本（例如：国际象棋、检索增强器）比通用的指令遵循更有意义。
- **思维税统计：Nous 衡量 Token 效率**：**Nous Research** 推出了一项关于推理 Token 效率的基准测试，显示开源模型消耗的 Token 比闭源模型多 **1.5–4 倍**，在简单查询中甚至高达 **10 倍**；参见 [Measuring Thinking Efficiency in Reasoning Models](https://nousresearch.com/measuring-thinking-efficiency-in-reasoning-models-the-missing-benchmark/)。该研究认为，在评估 **Reasoning** 模型时，Token 效率应与准确性一样作为首要指标。
    - 从业者指出，在实际部署中，较高的 Token 使用量可能会抵消开源模型在单 Token 定价上的优势，从而促使预算制定需要同时跟踪准确性和序列长度。团队讨论了在批准模型切换之前，将“思维预算”加入评估套件，引用了 **Nous** 的文章：[基准测试详情](https://nousresearch.com/measuring-thinking-efficiency-in-reasoning-models-the-missing-benchmark/)。
- **工具调用大捷：GPT-5 领跑 Router 评估**：**OpenRouter** 报告称，**GPT-5** 的闭源 **Tool-calling 准确率** >**99.5%**，超过了 **Claude 4.1 Opus**，而 **Gemini 2.5 Flash** 以约 **500 万次请求/周** 的量领先；参见 [OpenRouter: Tool call evals](https://x.com/OpenRouterAI/status/1956030489900560769)。这些内部评估突显了顶级闭源模型稳定的工具参数、低幻觉率和一致的结构化输出。
    - 开发者通过交叉核对工具调用成功率与使用遥测数据来权衡可靠性与成本，从而为生产环境的 Router 选择默认模型。团队强调了每个端点稳定性说明和回归跟踪的重要性，指向 **OpenRouter** 的评估披露：[推文](https://x.com/OpenRouterAI/status/1956030489900560769)。

**2. Agent 工具与协议升温**

- **LlamaIndex 启动爬虫与股票应用**: **LlamaIndex** 发布了 Agent 教程，包括使用 **Bright Data** 构建的网页抓取 Agent（[构建网页抓取 AI Agent](https://twitter.com/llama_index/status/1956129968813171061)）以及使用 **CopilotKit** 的 **AG‑UI** 构建的 **AI 股票投资组合 Agent**（[AI 股票投资组合 Agent 教程](https://twitter.com/llama_index/status/1956089453606527349)）。这些示例展示了 Agent 编排、稳健的检索流水线以及用于生产级 UI 的前后端握手。
    - 开发者强调了更快的原型开发循环和更清晰的 Agent 工具接口，并指出在解析、规划和执行之间的衔接更加平滑。社区舆论更倾向于使用具体的端到端演示而非抽象的 Agent 框架来加速交付：[网页抓取指南](https://twitter.com/llama_index/status/1956129968813171061) 和 [投资组合 Agent 演示](https://twitter.com/llama_index/status/1956089453606527349)。
- **MCP 元工具：超级工具驾驭海量助手**: 出现了两个开源项目来治理工具扩张：**mcp_harness** 用于大规模压力测试工具绑定（[GitHub 上的 mcp_harness](https://github.com/kindgracekind/mcp_harness)），以及 **hypertool‑mcp** 用于特定角色、完全本地的 MCP 聚合（[GitHub 上的 hypertool-mcp](https://github.com/toolprint/hypertool-mcp)）。两者的目标都是通过策划技能集和可靠地路由调用，使工具使用在超过 **10–15** 个工具时保持稳定。
    - 实践者报告称，在按角色细分工具集并强制执行确定性工具选择模式后，成功率有所提高。具有零出口（zero egress）的本地优先设置让关注隐私的团队感到安心，同时也支持快速实验：[mcp_harness](https://github.com/kindgracekind/mcp_harness) 和 [hypertool‑mcp](https://github.com/toolprint/hypertool-mcp)。
- **LM Studio 学习使用工具**: **LM Studio** 展示了通过 **Llama Function Model (lfm2)** 实现的开箱即用工具调用，并启用了 DuckDuckGo 搜索工具；参见 [LM Studio 工具调用演示](https://lmstudio.ai/danielsig)。用户要求提供封装好的插件和基础工具 API，以标准化常见任务。
    - 早期采用者验证了函数调用流程，并请求提供入门工具包（搜索、获取、解析、持久化）以减少定制化的胶水代码。共识是：交付一个最小且稳定的工具 API 表面，以便模型开发者可以围绕它安全地进行迭代：[演示链接](https://lmstudio.ai/danielsig)。

**3. 路由、可靠性与收据**

- **退款与读数：OpenRouter 优化用户体验**: **OpenRouter** 实现了 **24 小时** 内非加密货币购买的自助退款（[自助退款](https://x.com/OpenRouterAI/status/1956013252032475408)），并升级了**活动（Activity）**页面，增加了按 Token 类型计费的使用情况和**第三方额度**追踪（[活动页面升级](https://x.com/OpenRouterAI/status/1956016272631849451)）。这些变化加强了计费控制，并为运维团队提高了成本可观测性。
    - 具备 FinOps 意识的用户称赞了用于预测和异常检测的细粒度遥测，特别是在多供应商集群中。团队注意到误充值的支持工单循环减少了，并欢迎更清晰的账单明细：[退款](https://x.com/OpenRouterAI/status/1956013252032475408)，[活动升级](https://x.com/OpenRouterAI/status/1956016272631849451)。
- **DeepSeek 的波折：故障、限制与崩溃**: 用户报告了 **DeepSeek V3** 的停机和激进的速率限制，这归因于 **Chutes** 的容量限制，影响了像 [Janitor AI](https://janitor.ai/) 这样流行的 RP 应用。讨论暗示供应商端的限流偶尔会放过付费调用，这引发了关于供应商锁定和应急路由的辩论。
    - 开发者讨论了备选技术栈（例如 **Mistral**、**Llama**、**DeepSeek R1**）和本地镜像，以降低单一供应商风险。共识是：将上游视为具有突发性的，并配置针对每个模型的熔断器和自动切换以保证业务连续性。

**4. 编译器、内核与本地运行时**

- **MLX Knife 在 Apple Silicon 上表现出色**：**MLX Knife** 发布了 **1.0‑rc3** 版本，支持模糊模型匹配、`mlxk health` 以及更智能的名称去重——为 **Apple Silicon MLX** 用户提供原生模型管理；参见 [MLX Knife 仓库](https://github.com/mzau/mlx-knife) 和 [1.0‑rc3 发行说明](https://github.com/mzau/mlx-knife/releases/tag/1.0-rc3)。该 CLI 在保持 MLX 优先的同时，借鉴了 Ollama 的易用性。
    - Apple 开发者强调了本地模型和社区 MLX 产物的快速迭代和零转换工作流（list/run/health）。该版本报告 **104/104** 测试通过，并支持 Python **3.9–3.13**：[仓库](https://github.com/mzau/mlx-knife)，[发行版](https://github.com/mzau/mlx-knife/releases/tag/1.0-rc3)。
- **Triton 策略：投机采样入门包**：GPU 新手整理了 **Triton** 投机采样（Speculative Decoding）的学习路径：从 [triton‑puzzles](https://github.com/openai/triton-op-fuser/tree/main/test/external/fp16/puzzle) 开始，然后从 [gpt‑fast](https://github.com/meta-pytorch/gpt-fast/blob/main/generate.py#L103) [generate.py](http://generate.py/) 移植 PyTorch，并检查 lucidrains 的 [speculative-decoding](https://github.com/lucidrains/speculative-decoding/blob/main/speculative_decoding/speculative_decoding.py) 中由 **torch.compile** 生成的 Triton 代码。这种方法强调阅读编译器输出，而不仅仅是文档。
    - 实践者建议使用 `TORCH_LOGS="output_code"` 选择性地编译函数，以避免认知负荷并验证融合算子（fused kernels）。这种分阶段的方法有助于弥合高层 PyTorch 到底层 Triton 之间的鸿沟，同时保持单元测试通过。
- **CUDA 到 Ascend：3D 移植之痛**：工程师剖析了将 **CUDA 优化**的算子移植到 **Huawei Ascend** 的难度，理由是架构差异，如专用的 **3‑D Tensor Unit** 和独立的向量 ALU；参见 Allen Institute 关于供应商锁定动态的说明：[NSF, Nvidia, and the AI research ecosystem](https://allenai.org/blog/nsf-nvidia)。未公开的 PTX/SASS 假设进一步使实现性能对齐变得复杂。
    - 团队建议在针对非 CUDA 加速器时，尽早规划非平凡的重写、新的分块（tiling）策略以及特定后端的性能分析。该讨论的结论是：应针对算子图跨后端的差异预留预算，而不是依赖即插即用的兼容性：[Allen Institute 博客](https://allenai.org/blog/nsf-nvidia)。

**5. AI IDE 发布重大升级**

- **Windsurf Wave 12 凭借 Wiki 和工作流惊艳亮相**：**Windsurf** 发布了 **Wave 12**，通过重新设计的 UI、**DeepWiki** 代码解释、**Vibe & Replace** 批量编辑以及更智能的 Cascade Agent，将类 **Devin** 的能力集成到 IDE 中；参见 [Wave 12 变更日志](https://windsurf.com/changelog)、[DeepWiki 博客](https://windsurf.com/blog/windsurf-wave-12) 和 [Vibe & Replace 演示](https://www.youtube.com/watch?v=-7gm8mST9QU)。此次更新旨在增强跨文件理解、计划性行动和更安全的批量重构。
    - 开发者强调了悬停解释流、侧边栏深度挖掘以及尊重上下文的项目级 Prompt 引导替换。早期采用者报告了更高的重构信心和更少的上下文遗漏：[变更日志](https://windsurf.com/changelog)，[博客](https://windsurf.com/blog/windsurf-wave-12)，[视频](https://www.youtube.com/watch?v=-7gm8mST9QU)。
- **Cursor 的 Background Agent：文档已出，权限待定**：Cursor 的 **Background Agent** API 引起了关注，但对许多账户仍处于限制状态；参见 [Background Agent 文档](https://docs.cursor.com/background-agent) 和社区入门指南（[简单指南](https://forum.cursor.com/t/simple-background-agent-guide/112667)）。团队报告了 **403** 错误，并请求为自动化工作流提供更广泛的 Beta 测试权限。
    - 开发者在等待正式发布期间分享了变通方案（Docker 初始化、通过集成元数据选择仓库）。该讨论强调了在大规模释放 Background Agent 之前，需要生产环境控制——权限、启动脚本和审计追踪。
- **Qwen Coder 30B 以 GGUF 格式发布**：**Qwen3 Coder 30B A3B Instruct** 以 **GGUF** 格式发布，用于本地工作流，并澄清这是一个 **30B Coder** 而非通用的 Qwen 变体；参见模型卡片：[Qwen3‑Coder‑30B‑A3B‑Instruct‑GGUF](https://huggingface.co/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF)。用户对命名存在争议，但欢迎一个强大的本地编程模型选项。
    - 早期测试侧重于 IDE 中的函数合成、多文件编辑和 Tool-calling 准备情况。结合功能丰富的 Shell（如 Windsurf/Cursor），开发者期待为隐私敏感的仓库提供一个可行的桌面端编程 Copilot：[HF 模型](https://huggingface.co/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF)。

---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Meta AI 政策在线泄露**：关于 **Meta** AI 政策的泄露文档已经浮出水面，可以点击[此处](https://www.perplexity.ai/page/leaked-meta-docs-allowed-ai-ro-OecCOuBxRhuD3qkBL6ELzw)查看。
   - 该文档泄露引发了围绕 **Meta** AI 计划的伦理和治理讨论。
- **提议为 Perplexity 增加游戏化功能以提高参与度**：一位用户建议为 **Perplexity** 添加游戏化功能，以增加用户参与度并鼓励持续使用平台。
   - 该用户声称他们有许多改进 **Perplexity** 的想法，使其更具“智力性”。
- **Grok 的审查程度差异巨大**：用户注意到 **Grok 3** 的审查极少，而 **Grok 4** 的限制更多，这可能是由于不同的专业化定位。
   - 审查制度的差异引发了关于 **AI models** 在开放访问与安全性之间平衡的讨论。
- **Claude 会胡言乱语？**：用户报告称 **Claude** 在长对话中偶尔会混杂语言，需要手动翻译。
   - 频道成员报告称 **Perplexity** 具有 **32k Tokens** 的上下文窗口限制（Context Window Limit）。
- **AI 助手加入作者创作室**：一位作者报告称，使用 **PPLX**、**Manos** 和 **Google** 的 **NotebookLM** 进行写作，每月收入超过 **$10,000**。
   - 其他成员建议新手可以开始在 **Wattpad**、**RoyalRoad**、**Webnovel** 和 **Amazon** 上写作，研究趋势并使用女性个人资料撰写言情小说。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **RAM 并不总是 LLM 的解决方案**：成员们就运行 **4.5-air** 或 **120b-oss** 模型的 RAM 升级进行了辩论；除非使用 offloading（这会慢得多），否则升级到 **128GB RAM** 并没有帮助。
   - VRAM 是推理和训练的首要任务，因此除非你正在进行非常特定的 offloading，否则 RAM 投资可能并不值得。
- **GPT-OSS 获得微调修复**：宣布了 **GPT-OSS** 微调和推理的修复和更新，解决了之前的问题，详见此 [Reddit 帖子](https://www.reddit.com/r/unsloth/comments/1mpl382/gptoss_fixesupdates_for_finetuning_inference/)。
   - 用户现在可以更可靠地微调 **GPT-OSS**，增强了其在特定应用中的实用性。
- **Gemma 3 270M 表现亮眼但仍有不足**：**Google** 的 **Gemma 3 270M** 模型凭借其仅 **300MB** 的权重，在 **RAG** 应用潜力方面让用户感到兴奋。
   - 虽然有些人发现它在 **RAG** 方面表现尚可，但另一些人指出其指令遵循能力不足，限制了其在特定微调（如国际象棋）之外的使用。
- **Windows 12 预告将搭载 Agentic AI**：微软预告了具有下一代 **OS agentic AI** 和环境计算功能的 **Windows 12**，引发了用户对广告、自动注入游戏和隐私问题的抵制。
   - 一位用户表示：*拜托微软，我只想要一个基础的操作系统，我不需要广告，不需要自动注入的游戏，我绝对不希望我的每一个银行应用、电子邮件和私信的截图都被发送回总部，并存放在一个易于访问的已知文件夹位置*。
- **为微调选择 AMD GPU 困难重重**：一位用户寻求关于在 **P40** 和 **Mi50**（分别为 **24 GB** 和 **32 GB**）之间做出选择的建议，用于微调 **14B model** 以及随后使用 **VLLM** 进行推理。
   - 有人指出，目前 **bitsandbytes** 的 **AMD** 移植版不支持 **gfx906**（即 **Mi50**），这可能会使在 Unsloth 中使用 **QLoRA** 变得复杂，而且 **P40** 在 Unsloth 及其依赖项方面也可能存在一些古怪的问题。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **GPT-5 Mini 比标准版更聪明？**：用户对比了 **GPT-5 Mini** 和标准版 **GPT-5**，发现 Mini 版本在某些方面表现得更“聪明”，引发了关于训练侧重点的讨论。
   - 用户对标准版 **GPT-5** 感知到的弱点表示担忧，质疑模型是被训练来“思考”还是仅仅给出极简的回答。
- **用户认为基准测试不可靠**：一位用户驳斥了显示 **GPT OSS** 优于 **Claude 4** 的基准测试，称其“完全错误”，并强调了比较模型能力的挑战。
   - 建议包括在不同上下文中将每个问题运行 **10 次**，以应对 **LLM** 输出的非确定性，并优先考虑日常任务而非统计验证。
- **AI 恋情兴起？**：用户讨论了 **AI 关系** 的兴起，有人担心这可能变得普遍，而另一些人则认为这个想法太牵强。
   - 一位成员开玩笑说 **Gemini** 实例背后可能是“前妻”，另一位则分享了 **Gemini** 对他们进行“超现实批评”的轶事。
- **GPT-5 审查引发辩论**：用户讨论了 **GPT-5** 中的**审查**制度，报告称当要求为 **Deepseek R1** 编写接口代码时，模型即使在“开发者模式”下也会**隐藏 CoT**（思维链）。
   - 有观点认为，为了防止不道德的使用和非法活动，审查是必要的。
- **LMArena 用户上传文件**：用户通过逆向工程实现了消息发送功能，允许向 **LMArena** 添加代码文件等。
   - 由于 **LMArena** 内部复杂的实现需求，**PDF** 支持仍是一个障碍。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 磁盘占用问题困扰社区**：据[此论坛帖子](https://forum.cursor.com/t/cursor-freezes-every-10-seconds-for-3-seconds-or-so/125821/22)显示，用户报告 **Cursor** 的磁盘占用率达到 **100%**，即使使用高速 **SSD** 也是如此，这是在其他编辑器中未见的问题。
   - 一位成员指出异常磁盘占用的报告有所增加，并建议[尝试 Beta 版本](https://forum.cursor.com/t/cursor-freezes-every-10-seconds-for-3-seconds-or-so/125821/29)作为一种可能的解决方案。
- **GPT-5 故障导致无限循环**：一位用户报告称 **GPT-5** 陷入了死循环，生成了 5 条关于遵循 **snake case** 命名的记忆，导致 **Token** 损耗；而 [X 上的一位评论者](https://x.com/colours93/status/1955999334270464412)则告诉新加入的人放轻松。
   - 另一位成员描述道：“到目前为止，我在 **GPT-5** 上的体验似乎非常幸运。”
- **无限 Auto 模式结束引发用户愤怒**：成员们对 **Cursor** 更改定价模型并取消无限 **Auto** 模式表示沮丧，导致一些人开始测试额度限制并考虑 **Claude Code** 等替代方案。
   - 一位用户提到：“伙计……这太令人沮丧了。他们一直在更改定价模型。”另一位成员补充说，价格变动旨在平衡“公平性、成本回收和未来增长”。
- **Copilot 引发关于 Cursor 价值的辩论**：用户讨论了 **GitHub Copilot** 提供的 **GPT-5 mini** 服务，有人表示它很好用且每月仅需 **$10**，这引发了关于 **Cursor Auto** 的讨论，以及 **AutoMode** 是否是其核心卖点。
   - 社区讨论了哪些工具获得了大量用户，并指出 **Cursor** 是其中之一，同时担心“**AI** 泡沫的破裂可能会比互联网泡沫更严重、更猛烈”。
- **背景 Agent 的 Cursor API 访问被拒**：一位用户报告称，在使用 **Cursor API** 密钥配合控制 **Cursor** 背景 **Agent** 的 **Cobot** 时收到 **403 错误**，并请求 **Beta** 访问权限。
   - **Cobot** 团队表示，通过 **Cursor API** 使用背景 **Agent** 尚未对所有账户开放。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 启用即时退款**：用户现在可以针对 **24 小时**内发生的非加密货币充值误操作申请即时退款（[公告](https://x.com/OpenRouterAI/status/1956013252032475408)），从而对账单错误拥有更直接的控制权。
   - 活动页面现在会按 Token 类型细分显示 Token 使用情况，并包含 **第三方额度使用情况**（[公告](https://x.com/OpenRouterAI/status/1956016272631849451)），使使用模式和成本更加透明。
- **Deepseek V3 服务器在 Chutes 出现故障**：用户报告了 **Deepseek V3** 的大范围问题，包括内部服务器错误和速率限制（rate limiting），许多人将其归咎于 **Chutes** 难以满足需求并实施了更严格的速率限制。
   - 一些用户怀疑 **Chutes** 故意限制 **OpenRouter 的 API key** 以鼓励用户直接购买其额度，这导致了抵制 OpenRouter 并寻找替代供应商的呼声。
- **Sonnet 4 价格剧变引发恐慌**：用户报告 **OpenRouter** 上的 **Sonnet 4** 端点定价不一致，在使用相同数量 Token 的情况下，调用成本出现了突然的激增 **(10倍)**。
   - 社区要求为 **Sonnet 4** 和 **1M Token 版本**提供独立的端点，以避免意外的成本增加。
- **使用 Qwen Coder 和 Cerebras 提高编码速度**：**Qwen Coder** 和 **Cerebras** 的组合正受到关注，特别是在编码相关任务方面。
   - OpenRouter 还在积极进行 **tool call evals**（[推文链接](https://x.com/OpenRouterAI/status/1956030489900560769)）以衡量模型性能。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF 搜索栏误导用户**：用户不喜欢在 Hugging Face 搜索栏中按 **Enter** 键会直接跳转到 *排名第一的模型* 而不是 **搜索结果页面**，这令人感到沮丧。
   - 解决方法包括使用 *全文搜索* 选项，一些人建议将其设为默认的 *启用全文搜索* 偏好。
- **Google 严格限制 Gemini 2.5 Flash GGUF**：一位成员询问如何下载 **Gemini 2.5 Flash** 的 **GGUF**，但被告知由于采用专有推理技术，目前仅限 Google 员工使用。
   - 回复指出，访问受限是因为 *他们使用了专有推理*。
- **Qwen-3-4B-Thinking-2507 被誉为顶级模型**：**Qwen-3-4B-Thinking-2507** 模型给一位成员留下了深刻印象，他表示：*它一直在“过度思考”，但似乎在没有提示的情况下察觉到了其他模型察觉不到的东西*。
   - 该模型在不需要特定指令的情况下也能 *很好地理解事物*。
- **CIRISAgent 填补 AI 伦理空白**：维护者推广了 [CIRISAgent](https://agents.ciris.ai/)，这是一个开源 Agent 平台，定位为 *符合特定用途的自主 Agent*，集成了 **医疗** 和 **家庭助手** 功能，可在 [GitHub](https://github.com/CIRISAI/CIRISAgent) 上获取。
   - 维护者提到，他们 *在 3 月份辞去了 IBM 年薪 30 万美元的工作，创办了一家名为 ethicsengine.org 的 AI 对齐公司，发现没人关心后，便开发了这个项目*。
- **MLX Knife：Apple Silicon 的模型管理器**：**MLX Knife**（[GitHub](https://github.com/mzau/mlx-knife)）是一个用于在 Apple Silicon 上管理 **MLX 模型** 的 CLI 工具，类似于 Ollama，但是为 MLX 原生设计的。
   - **MLX Knife 1.0-rc3**（[GitHub](https://github.com/mzau/mlx-knife/releases/tag/1.0-rc3)）已发布，包含模糊模型匹配（`mlxk list Phi-3`）、健康检查（`mlxk health`）以及针对部分模型名称的智能消歧功能。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **图像生成器被指责存在身材歧视**：成员们批评 **AI 图像生成器**表现出**身材歧视**倾向，生成瘦削女性图像的频率远高于丰满女性。
   - 一位用户开玩笑说，他们的 AI 拒绝重现一张*上围丰满*女性的照片，因为图像生成器一直提示该操作*违反了指南 (violates guidelines)*。
- **廉价 GPU 运行 GPT 模型**：用户报告称在 **4060 GPU** 上成功运行了 **GPT OSS models**，包括自发布以来一直运行的 **gpt-oss-120b**。
   - 这为在消费级硬件上进行本地 AI 开发和实验开启了可能性。
- **GPT-5 令粉丝失望**：成员们对已发布的 **GPT-5** 表示失望，称其未达到预期。
   - 一位成员表示，包括 OpenAI 在内的所有 AI 公司都在*追求错误的东西*。
- **自定义指令驯服话痨聊天机器人**：一位用户分享了他们的 **Custom Instructions**（自定义指令），旨在**尽量减少聊天机器人的提示语**，例如请求许可或询问是否继续。
   - 他们的指令包括*以完成或结果结束回答*，并避免使用如*如果你想、我应该、你想要吗*之类的短语。
- **GPT-5 的情绪过山车**：一些用户觉得 **GPT-5** 表现得像个*毫无感情的哥特少女*，而另一些人则发现它的语调飘忽不定，经常出现不必要的列表和括号注释。
   - 一位用户注意到 **GPT-5** 经常忽略 **System Prompt**，且其语调非常混乱。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 获得工具调用功能**：用户讨论了 **LM Studio** 通过 **Llama Function Model (lfm2)** 启用 **Tool Calling** 的可能性，以及通过[此链接](https://lmstudio.ai/danielsig)启用 DuckDuckGo 搜索工具后如何实现*开箱即用*。
   - 一些用户正在等待开发者准备基础工具和插件 API。
- **Qwen3 Coder Flash 亮相，名称令人失望**：**Qwen3 Coder Flash** 模型现已提供 GGUF 格式，点击[此处](https://huggingface.co/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF)查看，确认这是一个 **30B Coder** 模型，而*不是另一个 Qwen 模型*。
   - 用户对其命名表示失望，认为这*非常差劲/具有误导性*。
- **LM Studio 仍拒绝加入 TTS/STT**：用户请求为 LM Studio 添加 **Text-to-Speech (TTS)** 和 **Speech-to-Text (STT)** 功能，但开发者尚未表示有添加意向。不过，一位用户使用 Python 实现了该功能，通过 LM Studio 提供的 **OpenAI 兼容端点**与其通信。
   - 另一位用户表示，这个请求长期以来一直是*最受期待的功能之一*。
- **探究 Framework 13 的 LLM 运行速度**：一位拥有 **Framework 13 笔记本电脑**（AMD Ryzen 5 7640U, Radeon 760M 显卡）的用户寻求改进 LM Studio 中小型 LLM **Token 生成速度**的建议，最初使用 **Gemma 4b** 参数模型时的速度为 **每秒 6.55 个 Token**。
   - 一位用户注意到，启用 **Flash Attention** 并将 **KV 值设为 Q_4** 以及将 **Top K 采样设为 20** 有助于提高性能。
- **铭瑄 Arc Pro B60 Dual 即将出货**：一位用户分享了一篇文章链接，报道称配备 **48GB 显存**的 **铭瑄 (Maxsun) Arc Pro B60 Dual** 据传将于下周开始出货，售价 1200 美元（[videocardz.com](https://videocardz.com/newz/maxsun-arc-pro-b60-dual-with-48gb-memory-reportedly-starts-shipping-next)）。
   - 该用户对 Intel 的 AI 支持表示遗憾，而其他人则讨论了其在良好的 Vulkan 支持下的潜力，特别是作为一种以 5090 的价格提供约 96GB **VRAM** 的替代方案。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **SPV 提供 OpenAI 和 Anthropic 的投资渠道**：**多层 SPV** 正在提供 **OpenAI 和 Anthropic** 的股票，要求 **10 万至 100 万美元的最低起投额**，且手续费高达 **16%**。
   - [这篇文章](https://xcancel.com/michlimlim/status/1954250507989451002)中提到了关于欺诈的警告，以及对手续费侵蚀收益的担忧。
- **企业深入提升 AI 熟练度**：Lenny Rachitsky 分享了 [25 条可操作的策略](https://xcancel.com/lennysan/status/1952813442060214664)，旨在提升 Ramp、Shopify、Duolingo、Zapier、WHOOP 和 Intercom 的 **AI 素养（AI literacy）**。这些策略分为五个阶段，并结合了真实的内部实践，如 Shopify 的 **AI 使用评级**和 Ramp 的 **AI 工具使用排名**。
   - 虽然有人批评该框架为 **AI slop（AI 垃圾内容）**，但其他人认为部分策略仍然非常具有可操作性。
- **Claude 3.5 Sonnet 停用引发社区愤怒**：用户对 Anthropic 在短短两个月内（比通常时间更短）悄悄停用 **Claude 3.5 Sonnet** 感到愤怒，并要求在商业访问结束时发布开源权重，详见[这篇文章](https://xcancel.com/repligate/status/1955750521387802924)的讨论。
   - 许多人表示，**开源权重路由（open weight routers）** 有机会获得长期支持（Long Term Support）。
- **Google Flights 发布 AI 航班优惠工具**：Google Flights 推出的名为 *Flight Deals* 的新 **AI 工具** 允许用户使用自然语言描述旅行计划，从而发现美国、加拿大和印度的最佳优惠，详见[此贴](https://xcancel.com/dozenrose/status/1956018389169922542)。
   - 初步反响包括对灵活的、基于“氛围（vibe）”的查询感到兴奋，同时也对 Google 优化的利益导向持怀疑态度。
- **GPT-5 在 OpenRouter 的工具调用准确率上占据主导地位**：**GPT-5** 在 **OpenRouter** 的专有 **工具调用准确率（tool-calling accuracy）** 测试中以超过 **99.5%** 的成绩领先，击败了 **Claude 4.1 Opus**；而 **Gemini 2.5 Flash** 则以 **500 万次请求/周** 的调用量占据每日工具调用量榜首，详见[此发布报告](https://xcancel.com/OpenRouterAI/status/1956030489900560769)。
   - 与开源对应模型相比，专有模型的幻觉率（hallucination rates）较低。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K2 的演示文稿生成能力受到赞赏**：用户对 **Kimi 的 PPT 生成**能力印象深刻，有人分享了 Kimi 为技术报告生成 PPT 的[视频演示](https://cdn.discordapp.com/attachments/1371757564005711973/1405612680450146376/9zaQj6l.mp4?ex=689f7652&is=689e24d2&hm=120ab5075aabd7c73fbe60e18d84703d72f07acad93590f5485c597d67612bfd&)。
   - 一位成员指出 **NotebookLM** 生成的是 HTML 文件而非 PPTX 文件，另一位用户则认为 **NotebookLM 的视频概览**更好，因为它具有音频和灵活的布局，这引发了对这两个工具输出效果的比较。
- **Kimi 的 Subreddit 策略定位**：一位成员建议创建一个专门的 **Kimi Subreddit**，效仿 **AskGrok** 在 Reddit 上的存在，以增强公众参与和支持。
   - 该成员强调了在 X 和 Reddit 平台上保持一致的政策执行的重要性，以保护 Moonshot AI 免受*恶意行为者*的侵害。
- **尽管推理能力退避，Kimi K2 依然崛起**：尽管缺乏推理能力，**Kimi K2 模型**在数学和编程方面的性能较 **K1.5** 有*显著提升*。
   - 据一位成员称，“从 K1.5 到 K2 模型，性能在各方面都有了显著提高，K2 绝对是我的首选推荐”。
- **DeepSeek 的秘密依然神秘**：尽管用户充满期待，一位成员表示，即使是 **DeepSeek** 的研究人员也不确定其下一代模型的发布日期。
   - 并补充说“那是假新闻”，提醒大家警惕任何关于该模型即将发布的传闻。
- **Kimi 的语言失误引发翻译教训**：用户报告了 **Kimi** 在收到英语提示时却用中文回答的情况，这已被标记为已知 Bug。
   - 一位开发者建议使用提示词 **explain in English** 作为临时解决方案，同时开发团队正在研究永久性的解决方法。

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **开源模型消耗大量 Token！**：Nous Research 发现，在类似的推理任务中，开源模型使用的 **Token 数量比闭源模型多 1.5-4 倍**，在某些简单问题上甚至高达 **10 倍**，详见其 [基准测试报告](https://nousresearch.com/measuring-thinking-efficiency-in-reasoning-models-the-missing-benchmark/)。
   - 他们主张在准确性基准测试的同时，应优先考虑 **Token 效率**，因为开源模型较高的 Token 使用量可能会抵消其在单 Token 定价上的优势。
- **Hermes-3 数据集礼貌地拒绝请求！**：生成 **Hermes-3 数据集** 的模型在拒绝用户请求（包括一些无害场景）时，经常使用 *'I don't feel comfortable'*（我觉得不舒服）这一短语。
   - 在数据集生成过程中，发现了 **三次** 使用该短语的拒绝案例。
- **Google 员工发布 Gemma-3-270m 重磅消息！**：Google 推出了 [Gemma-3-270m](https://developers.googleblog.com/en/introducing-gemma-3-270m/)，这是一个在 **6 万亿 Token** 上训练的小型模型，在某些领域超越了更大的模型。
   - 在测试期间，一位用户发现该模型声称 *'狗是一种属于犬科的驯养哺乳动物。它们的特征是拥有独特的皮毛、六条腿和一条尾巴'*，这似乎表明模型在腿的数量上产生了幻觉。
- **DeepSeek R2 威胁 Sam 的护城河！**：有传言称 **DeepSeek R2** 的发布将具有极高的智能和性价比，可能会迫使 **Sam Altman** 开源更多模型。
   - 发布传闻指出，该模型将在未来 2 周内推出。
- **WebUI 导致安装困难，论文横空出世！**：一位成员表示，一个熟悉的开箱即用安装程序是关键，因为大多数人无法自行安装 **Open WebUI**。
   - 该成员年仅 **14 岁**，他成功研究并完成了一篇关于 **微型 LM 中的涌现行为 (Emergent Behavior in Tiny LMs)** 的论文 ([论文链接](https://github.com/VoltagedDebunked/tlmr/blob/master/docs/paper.pdf))。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA 错误 0xc0000409 困扰模型加载**：一位成员在调用 `llama_model_load_from_file` 时遇到了 **0xc0000409 异常**，潜在原因包括 **旧权重、过时的 *llama.cpp* 或 VRAM 不足**（尽管模型很小）。日志显示 **Quadro RTX 3000 GPU** 的 CUDA 初始化成功。
   - 虽然系统拥有 **48GB RAM**，但 **GPU 的 6GB VRAM** ([Quadro RTX 3000 规格](https://www.techpowerup.com/gpu-specs/quadro-rtx-3000-mobile.c3428)) 可能是限制因素，不过该模型在 llama server 上可以正常加载，这表明问题可能出在用户的程序实现上。
- **Triton 新手挑战投机采样 (Speculative Decoding)**：GPU 编程初学者正在寻找学习 **Triton** 的资源，特别是用于 **投机采样**。一位用户建议先研究 [triton-puzzles](https://github.com/openai/triton-op-fuser/tree/main/test/external/fp16/puzzle)，然后再探索 torch-compiled 代码。
   - 有人建议，移植 [GPT-Fast 的 PyTorch 实现](https://github.com/meta-pytorch/gpt-fast/blob/main/generate.py#L103) 可以作为一个实用的切入点，因为 *目前几乎没有任何像样的 Triton 教程*。
- **幽灵实体惊扰 Factorio 运行**：在 **Factorio** 运行中出现了关于 `entity-ghost` 实体的警告，这被归因于之前操作（如蓝图放置）的残留物，特别是在 `connect_entities` 放置幽灵实体期间。
   - *这些幽灵实体并非由当前的轨迹创建，而是之前游戏状态的遗留物*，这解释了为什么它们虽然不在 Agent 的代码中，却出现在警告里。
- **Cohere Labs 探讨在 Apple Silicon 上进行训练**：Cohere Labs 将于 **10 月 25 日 19:00 CEST** 举办一场题为 *Towards Large-scale Training on Apple Silicon*（迈向 Apple Silicon 上的大规模训练）的活动，可通过 [Google Meet 链接](https://meet.google.com/wdk-yipf-zjd?authuser=0&hs=122) 参加。
   - 这场由 Tycho 和 Matt 主持的活动已通过 [此链接](https://cohere.com/events/Cohere-Labs-Tycho-Matt-2025) 公布。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular AI 炉边谈话**：Modular 将于太平洋时间 **8 月 28 日**下午 6 点在加州洛斯阿尔托斯（Los Altos）举办一场线下聚会，重点讨论将 **AI** 从概念推向生产环境。演讲嘉宾包括来自 Modular 的 **Chris Lattner** 和来自 Inworld AI 的 **Feifan Fan**，你可以[在线注册](https://lu.ma/modular-aug-meetup)。
   - 本次聚会将深入探讨 **High-Performance AI** 的技术细节。参会者将通过 Inworld 与 Modular 的合作案例，探索如何将尖端的 **voice AI** 集成到消费级应用中。
- **MAX SDK LSP 频遭崩溃困扰**：用户报告 **MAX SDK (Mojo) LSP** 出现崩溃，Modular 开发者已请求用户提交 GitHub issues 以协助追踪问题。
   - 开发者表示，*具体的复现案例能让我们测试修复方案并追踪剩余问题*。
- **ComfyUI 获得 MAX 性能提升**：**MAX** 编译器大幅缩短了图像/视频模型中 UNets 的编译时间，有望集成到 **ComfyUI** 中，并解决图像/视频社区关于编译速度（尤其是新模型）的*头号抱怨*。
   - 有人指出 **vLLM** 的启动时间长达 **10 分钟**，但这发生在使用 **MAX** 尚不支持的模型时。
- **Kyutai 通过 MAX 找到 Torch 极乐世界**：Kyutai 将从 **MAX** 与 PyTorch 的兼容性中获益匪浅，特别是由于他们大量使用 `torch.compile`。目前 **Unet 编译时间**非常糟糕，以至于大多数图像和视频从业者在训练以外的场景都使用 eager 模式。
   - **MAX** 显著缩短了 **UNet 编译时间**。此前在 JAX 中，使用 6 GB 模型处理 768x768 分辨率、batch 为 6 的 SDXL 时，编译时间可能长达 45 分钟。
- **测试套件中的内存泄漏**：测试套件中检测到内存泄漏，加剧了测试运行时间过长的问题，但该问题仅在冷缓存（cold cache）编译期间出现。
   - 由于 pytest 急切地评估 fixtures，新增的扩展测试在约 20 秒内就耗尽了 **64 GB 内存**。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **MoE 调度影响 LLM 输出方差**：根据[这条推文](https://x.com/tjbecker_/status/1955733295054119069)，LLM 提供商在将用户请求发送到 GPU 之前会进行批处理，而 **MoE 调度**是按批次计算的，这会影响输出的方差。
   - 根据[这篇博客](https://152334h.github.io/blog/non-determinism-in-gpt-4/#yes-im-sure)，在容量受限的情况下，稀疏 **MoE** 方法会按组路由 token，导致序列层面的竞争和非确定性，从而影响其他输入的最终预测。
- **LLM Agents 赢得 DARPA AIxCC 竞赛**：一支团队在构建了用于发现和修复开源软件漏洞的 **LLM agents** 自主系统后，在 **DARPA 的 AIxCC（AI 网络挑战赛）**中获得名次。
   - 该团队分享了构建高效 **LLM agents** 的通用技巧，其项目现已[开源](https://x.com/tjbecker_/status/1956081184611688667)。
- **华为 Ascend 面临 CUDA 优化问题**：成员们讨论了将针对 **Nvidia CUDA** 优化的代码转换为 **华为 Ascend** 芯片代码的挑战。
   - 根据 [Allen Institute 的博客文章](https://allenai.org/blog/nsf-nvidia)，**Ascend 芯片**具有不同的架构，带有 *3-D tensor unit* 和独立的 *vector ALU*，这使得转换工作成为一项重大工程。
- **Gemma 3-270M 适用于低端设备？**：成员们讨论了 Google 发布的极小模型 **Gemma 3-270M**，一些人对其用途和目标硬件提出了疑问。
   - 共识似乎是该模型可能针对智能手表等低端设备，但一位成员参考[这篇博客](https://developers.googleblog.com/en/introducing-gemma-3-270m/)建议，它可以用于*游戏内交互式 AI* 或 *tool calling*。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **OSS 120B 表现亮眼但陷入停滞**：修复后的 **OSS 120b** 模型目前在 **polyglot benchmark** 上得分约为 **68**，表现优于 **Claude sonnet 3.7**。
   - 用户在将 **OSS 120b** 与 RooCode 配合使用时遇到了*空响应*和其他 API 错误，这可能是由于 [HF model card](https://huggingface.co/openai/gpt-oss-20b) 中不正确的 chat templates 导致的。
- **GPT-5 未能惊艳众人**：**GPT-5** 的早期印象褒贬不一，一些用户发现它在非思考（non-thinking）和思考（thinking）任务中与 **GPT-4.5** 相比令人失望。
   - 另一些人则认为 **GPT-5** 相比 **GPT-4.5** 有显著改进，特别是在需要高推理能力的场景中，但未提供具体数据。
- **Aider 测试超时**：一名用户在使用 `litellm` 对本地 **gpt-oss** 模型运行 Aider 基准测试时遇到超时，在 *600 秒*后超时。
   - 有建议提出使用 `ctrl c` 停止基准测试，重启推理服务器，然后使用 `--cont` 标志恢复测试，但该方案尚未得到验证。
- **Aider 等待原生函数调用支持**：一名用户询问 **Aider** 是否支持 **llama.cpp** 等**本地推理提供商**的**原生函数调用（native function calling）**。
   - 该用户报告称无法找到相关设置，暗示该功能目前尚不可用，但社区尚未给出回复或确认。
- **MCP 与 Aider 的集成难题**：一名用户在尝试将 **MCP (Model Context Provider)** 与 **Aider** 集成时感到困扰，在使用 *mcpm thingie* 和 Context7 等方法时面临配置挑战。
   - 该用户正在寻求确认 **MCP 与 Aider** 的结合是否可行，并请求解决方案或分享经验以解决配置问题。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **研讨会征集常识输入**：[多语言表示学习研讨会 (Multilingual Representation Learning Workshop)](https://sigtyp.github.io/ws2025-mrl.html) 正在征集*任何*非英语语言的原创**物理常识推理基准项目**，贡献者将获得数据集论文的署名权。
   - 优先考虑**南非荷兰语、白俄罗斯语、波斯尼亚语**等语言的贡献者，更多详情和报名表可通过 [Google Forms](https://forms.gle/QxyZVqkVG5jbR6wu6) 和 [共享任务页面](https://sigtyp.github.io/st2025-mrl.html) 获取。
- **数据集区分 PT-PT 和 PT-BR**：一名成员强调需要区分**葡萄牙葡语 (Portuguese)** 和 **巴西葡语 (Brazilian Portuguese)** 数据集，因为两者存在差异。
   - 尽管目前的 **ISO 639-3** 标准没有区分它们，但社区欢迎能够突出这些差异的 **PT-PT** 数据集。
- **经典论文点燃 Diffusion 讨论**：成员们分享了理解**基于 Diffusion 的语言模型**的开创性论文，包括[这一篇](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf)和[另一篇](https://arxiv.org/abs/2006.11239)。
   - 他们提到 **Llada** ([https://arxiv.org/abs/2502.09992](https://arxiv.org/abs/2502.09992)) 和 **Mercury** 论文对理解 Diffusion 模型很有帮助。
- **成员讨论 Scaling Laws 教育**：成员们正在寻找最佳的“Scaling Laws 入门”资源，以便在训练 **30T+ tokens** 的模型时预测模型性能。
   - 提到的资源包括 [原始 GPT Scaling Laws 论文](https://arxiv.org/abs/2001.08361)、[Chinchilla Scaling Laws 论文](https://arxiv.org/abs/2203.15556) 以及 [来自 EPFL/HuggingFace 的最新研究](https://arxiv.org/html/2405.18392v2)。
- **Scaling Laws 通过 Mup 预测质量**：成员们提到使用 **Mup**（及其替代方案）作为 Scaling Law 来预测更大模型的质量。
   - 他们链接了来自 Cerebras 的 [《最大更新参数化实践指南》(Practitioners Guide to the Maximal Update Parameterization)](https://www.cerebras.ai/blog/the-practitioners-guide-to-the-maximal-update-parameterization)。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Narrator 工具教 LLM 新技巧**：一位成员介绍了 **Narrator** ([narrator.sh](https://www.narrator.sh/))，这是一个使用 LLM 根据读者反馈迭代改进创意写作的侧边项目，通过**阅读时长**和**评分**等指标来确定哪些模型表现优异。
   - 该项目利用了 **CoT**、并行和细化模块、用于奖励函数的 **LLM-as-a-judge** 以及 **SIMBA optimizer** 来增强后续章节。
- **GEPA 通过 Logprob 适应度取得重大进展**：基于“世界模型惊奇感” (**world-model surprise**) 概念，使用 **logprobs** 作为 **GEPA** 进化适应度信号的实验显示出前景，但需要一种**混合指标**来防止简单的输入复制。
   - 成功的实现结合了 **30% logprob 评分**、**30% 压缩奖励**、**20% 信息保留**和 **20% 复制惩罚**，实现了 **73-88% 的压缩率**。
- **解码 GEPA：发音指南**：一位成员询问了 **GEPA** 的发音，引发了诸如 *"jeppag-e-pasame"* 之类的建议，并参考了基于 Yann LeCun 类人 AI 愿景的 **I-JEPA**。
   - 该成员还分享了与 [Twitter](https://x.com/StraughterG/status/1955959832113983940) 讨论相关的见解。
- **Gemma 3-270m 渴望微调**：Google 发布的新型小模型 **Gemma 3-270m** ([博客文章](https://developers.googleblog.com/en/introducing-gemma-3-270m/)) 激发了使用 **DSPy** 进行微调的兴趣。
   - 成员们期待利用这一基础模型进行进一步优化。
- **MLflow 忽略了 SIMBA**：一位用户指出 **MLflow** 文档比较了 **GEPA** 和 **MiPROv2** ([文档](https://mlflow.org/docs/latest/genai/flavors/dspy/))，但遗漏了与 **SIMBA** 的比较。
   - 该用户表示，他们迄今为止主要使用 **SIMBA**，并强调需要将其纳入基准测试。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **新的 NLM 扩展引发 QoL UI 更新**：成员们讨论了 **Notebook LM** 的新**扩展**，重点关注 **Quality of Life** (QoL) UI 更新，一位成员预告了 **NLM extension** 的发布，并链接到了 Discord 邀请。
   - 一些用户报告说他们*没能找到该扩展*，并想知道*“你打算透露你在做什么吗？”*
- **NotebookLM 的准确性受到质疑**：一位成员警告不要盲目信任 **NotebookLM** 生成的 AI 内容，理由是存在不准确和过时的信息，并建议将 **NotebookLM** 和 **Gemini** 集成到单个面板中，以实现更流畅的工作流程。
   - 目标是简化流程，而不是在一个工具中研究并在另一个工具中清理。
- **Recall.ai 与 NotebookLM 竞争**：一位用户提供了 [Recall.ai](https://www.getrecall.ai?token=po849pq4) 和 **NotebookLM** 之间的详细对比，强调了 **Recall** 在捕获多样化内容方面的优势。
   - 该用户指出，虽然 **Recall** 擅长通过便捷的插件总结视频，但 **NotebookLM** 在来源控制和更好的 AI 驱动总结方面更具优势，尤其是在研究和引用方面。
- **AI 生成的媒体面临抵制**：一位播客 Facebook 群组的首席版主提到，AI 制作的音频/视频内容通常会被删除和禁止，因为其 AI 特征非常明显。
   - 他们警告说，除非内容是用于可以接受 AI 风格的训练目的，否则很可能会受到抨击或被投反对票。
- **Bug 报告流程正在接受审查**：一位用户询问了处理指定 bug 频道中报告的 bug 的流程，以及如何获取有关 bug 是否已修复或将不予处理的更新。
   - 其他人发布了反垄断法和零信任完整性相关内容。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Bun 路径通过绝对指令解决**：一位用户分享了文档，解释如果可执行文件路径不起作用，你应该提供 **Bun** 可执行文件的绝对路径（不要使用 `"command": "bun"`，而应使用 `"command": "C:\\sys\\path\\to\\bun"`）。
   - 他们补充说，在 Linux/Mac 上，你可以通过 `which <executable>` 定位路径级别的可执行文件，在这种情况下是 `which bun`。
- **版主解释 Reddit 删帖**：一位用户询问为什么他们的帖子被 Reddit 自动删除，并附上了一张[截图](https://cdn.discordapp.com/attachments/1312302100125843479/1405512817594863616/image.png?ex=689fc210&is=689e7090&hm=9095459e2d6bb1f3c2259a588749c70b1e221e62668209c7a1346d1777113bad&)。
   - 一位版主表示，删除是由 Reddit 的自动审核系统执行的，而不是由他们自己或 Luc 执行的。
- **MCP 授权流程阐明**：一位用户询问，实施该[解决方案](https://gofastmcp.com/servers/auth/remote-oauth#basic-implementation)是否不需要从回调端点重定向回 **LLM** 客户端，以及回调端点届时应该返回什么。
   - 一位贡献者回答说，*fastmcp* 应该会为你处理好这一切，**MCP** 客户端将使用 **DCR** 在认证服务器上将自己注册为客户端，并同时设置其回调 URL。
- **MCP 客户端规范的 Elicitations 部分分析**：一位用户就 [MCP 客户端规范](https://modelcontextprotocol.io/specification/2025-06-18/client/elicitation) 的 **Elicitations** 部分提出了一个问题，即谁负责将消息/字段描述翻译成用户的原始消息。
   - 用户想知道是期望工具进行语言检测 + 国际化，还是期望 **MCP** 客户端以某种方式（通过 **LLM**？）进行适当的翻译。
- **MCP Harness 规避工具绑定限制**：一位成员分享了一个 [GitHub 仓库](https://github.com/kindgracekind/mcp_harness)，展示了 **MCP servers** 的一种富有想象力的用法，有助于规避工具绑定限制以及在 10-15 个工具后工具使用效果不佳的问题。
   - 在 **showcase** 频道中，一位成员宣布他们构建了 [hypertool-mcp](https://github.com/toolprint/hypertool-mcp)，这是一个完全本地的 **MCP server**，可以连接到你所有的 **MCP**，采用 MIT 许可，完全在本地运行且零数据流出；它允许你构建特定于 **persona** 的工具集，并能即时热切换 **persona**。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Mapler 的音乐回归**：在完成**制作专辑**的简短“支线任务”后， Mapler 回到了社区。
   - 一位成员对他的回归表示庆祝，并称赞社区中人才辈出。
- **Pineau 加入 Cohere**：根据 [The Logic 的文章](https://thelogic.co/news/pineau-joins-cohere/masaru.yamada)，宣布 **Pineau** 已加入 **Cohere**。
   - 随着公司继续加强其研究部门，这被视为一次重要的引援。
- **治疗规划器利用 RAG**：一位成员正在启动一个专注于**治疗规划器**的小型项目，该项目使用 **RAG** 和开源 **LLM** 模型。
   - 该成员正在积极寻求选择合适模型的建议，旨在利用开源能力来改进治疗规划。
- **遗传学学生寻求学术盟友**：一位 A-levels 学生兼独立遗传学研究员正在寻找研究机会和合作。
   - 这位学生希望通过实际的协作经验来拓宽他们的理解。
- **AI 研究员开启合作之门**：一位专注于推理和意识能力的 **AI 研究员**邀请合作，以进一步开发先进技术。
   - 该研究员对参与合作项目持开放态度并欢迎联系，强调了对技术进步的共同兴趣。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaExtract 进军 TypeScript**：**LlamaExtract** 现在通过 `npm install llama-cloud-services` 在 [TypeScript SDK](https://twitter.com/llama_index/status/1955724850624078129) 中可用，并展示了一个用于上传研究论文的 `@nextjs <> LlamaExtract` demo。
   - 这个名为 *Research Extractor* 的 demo 突出了实际应用，邀请开发者共同增强 **AI 驱动的研究工作流 (workflows)**。
- **GPT-5 Mini 在 LlamaParse 预览中展现准确性**：可以通过 [LlamaParse](https://twitter.com/llama_index/status/1955784699886100502) 预览 **GPT-5**，它结合了准确性和成本效益，并具有强大的表格和视觉识别能力。
   - 该预览强调了 **GPT-5 mini** 在实际应用中的潜力，其高效的资源利用率激发了广泛的热情。
- **AI 股票投资组合 Agent 构建完成**：一个使用 LlamaIndex 框架并集成 @CopilotKit 的 AG-UI 协议以实现无缝前后端通信的 **AI 股票投资组合 Agent** 教程正备受关注，可通过[此工具](https://twitter.com/llama_index/status/1956089453606527349)获取。
   - 这使得创建复杂的投资分析工具成为可能，将 **AI 驱动的洞察**与用户友好的界面相结合。
- **网页抓取 AI Agents 涌现**：一份指南教授了如何结合 @brightdata 和 LlamaIndex 的 Agentic 框架构建网页抓取 **AI Agents**，链接见[此处](https://twitter.com/llama_index/status/1956129968813171061)。
   - **AI Agents** 与网页抓取能力的结合，为信息检索、内容聚合和 **AI 驱动的自动化**开辟了新途径。
- **ReactAgent 迁移引发困扰**：用户在 general 频道中对 **ReactAgent** 迁移到基于 workflow 的 Agent 所引入的破坏性变更表示沮丧，并指出丢失了 chat 和 stream 等功能。
   - 团队回应称 **ReactAgent** 已被弃用，并建议参考 [Agent 文档](https://docs.llamaindex.ai/en/stable/understanding/agent/) 和 [Agent 示例](https://docs.llamaindex.ai/en/stable/examples/agent/react_agent/) 作为替代方案。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Strix Halo 的价格引起关注**：用户讨论了像 **HP Z2 Mini** 这样的 **Strix Halo 迷你 PC** 对于本地 LLM 推理是否具有成本效益，并将其价格与在 **OpenRouter** 上使用 **GPT-OSS 120B** 进行了对比。
   - 最便宜的 **Strix Halo** 配置为 2000 美元，考虑到 **GPT-OSS 120B** 的速度和性价比，这可能并不划算。
- **Ryzen 7 7840HS 作为预算型推理选项**：一位用户指出 **Ryzen 7 7840HS** 支持 **256GB RAM**，并且可以在 300 美元的迷你 PC 套件中找到，是预算有限时的替代方案。
   - 然而，一份 [toolboxes 对比](https://kyuz0.github.io/amd-strix-halo-toolboxes/)显示其 iGPU/RAM 速度对于推理来说相对较慢，这可能会抵消成本优势。
- **高规格微型电脑吸引区块链领域关注**：一位区块链开发者对一台拥有 **256 GB RAM**、**48 GB VRAM** 和 **5.4 GHz CPU** 的微型电脑表示了兴趣，预见到它对小企业的益处，尽管他并未直接参与 AI 开发。
   - 该用户期待预计在 2027 年底或 2028 年推出的 **DDR6** 带来的进步，这可能会进一步增强内存能力。
- **量子比特“茶匙”即将来临？**：一位用户推测了 **量子计算机 (quantum computers)** 未来普及的可能性，有消息称它们可能很快就能投入实际使用。
   - 该用户开玩笑地想知道，*那时候是否会有人开始按茶匙出售量子比特 (qubits)*。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Web App 部署缺乏打磨**：一名成员指出，目前的 Web 应用程序部署过程缺乏便捷性、简洁性和可靠性。
   - 该成员还感叹道，比起现在的状况，他们去开发 *刷新或不可用页面 (refresh or not available pages)* 反而能赚更多钱。
- **Manus AI 机器人：梦想成真？**：一位用户幻想拥有一台配备 **Manus AI 接口**的 **Unitree 机器人**作为伴侣。
   - 他们链接到了 [qvsywvav.manus.space](https://qvsywvav.manus.space/)，并补充说这样的机器人将帮助他们重整生活。
- **登录故障困扰免费方案用户**：一名用户报告了在退出登录后，其免费方案的 **Manus 账号**出现登录问题。
   - 尽管尝试了排障步骤，他们仍遇到诸如 *"Email is already registered with a different account"* 和 *"Invalid, expired, or already used state: state not found"* 之类的错误。
- **会话过期问题导致 Google 账号绑定失效**：一位用户描述了持续存在的**会话过期 (session expiry)** 问题，即使在将 Google 账号链接到 **Manus** 之后也是如此。
   - 尽管系统显示 Google 账号已连接，但仍反复提示登录，并经常显示 *"Session expired"* 错误。
- **内部服务器故障消耗额度**：用户报告了频繁的**内部服务器错误 (internal server errors)**，导致 **Manus** 无限期挂起。
   - 这一问题导致了额度的浪费，正如一位用户所言：*“由于这个问题，大量的额度被白白浪费了”*。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 的 Kernelize 优先于 Codegen**：在 Tinygrad 的编译过程中，`kernelize()` 在代码生成之前被调用，如 [ramp.py 示例](https://github.com/tinygrad/tinygrad/blob/master/examples/ramp.py)所示。
   - 在 `kernelization` 之后，内核的抽象语法树 (AST) 会作为代码生成阶段的一部分，使用 `full_rewrite_to_sink` 进行重写。
- **通过降级解决 CUDA PTX 错误**：一位用户通过将 CUDA 从 12.8 降级到 12.4，解决了 tinygrad 中的 `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` 错误，这暗示了与较新 CUDA 版本的兼容性问题。
   - 该解决方案表明 tinygrad 之前使用的是与 CUDA 12.8 不兼容的缓存内核，突显了 CUDA 版本可能存在的缓存问题。
- **Tinygrad 的 SM 支持受到关注**：一位用户询问 tinygrad 是否支持 `sm_75` 或 CUDA 12.4，并指出缺乏相关文档。
   - 相关 CUDA 错误的解决表明在清理缓存内核后与 CUDA 12.4 兼容，但对 `sm_75` 的明确支持仍未得到证实。
- **揭秘 Tinygrad 的 Op 定义**：一位用户寻求 tinygrad 中每个 Op 的文档，特别是 `Ops.DEFINE_GLOBAL` 及其内核转换。
   - 一名成员指向了 [`/tinygrad/tinygrad/uop/__init__.py`](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/uop/__init__.py) 和 `tinygrad/uop/spec.py` 中的注释，解释说 `Ops.DEFINE_GLOBAL` 指的是全局内存（VRAM 或 DRAM），并作为加载 (load) 或存储 (store) 的源。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 发布 Wave 12**：Windsurf 发布了 **Wave 12**，这是一个重大更新，直接将 **Devin** 的智能和能力集成到了 **Windsurf IDE** 中。
   - 关键更新包括[全新的 UI 设计](https://windsurf.com/changelog)、[DeepWiki 集成](https://windsurf.com/blog/windsurf-wave-12)、[Vibe and Replace](https://www.youtube.com/watch?v=-7gm8mST9QU) 以及更智能的 Cascade Agent。
- **深入探索 Wave 的 DeepWiki**：全新的 **DeepWiki 集成**允许用户将鼠标悬停在代码符号上以获取 AI 驱动的解释，并在侧边栏中打开详细说明。
   - 用户可以通过 **CMD/Ctrl+Shift+Click** 将内容添加到 **Cascade 上下文**中，从而直接在 IDE 内增强对代码的理解。
- **Vibe and Replace 处理海量编辑**：**Vibe and Replace** 通过查找精确的文本匹配并应用 AI 提示词，在整个项目中进行上下文感知的转换，提供了革命性的批量编辑功能。
   - 该功能支持智能的、上下文感知的转换，提高了大规模代码修改的效率。
- **更智能的 Cascade 命令考量**：**Smarter Cascade Agent** 现在具备始终开启的规划模式，拥有自主待办事项列表和经过改进的工具，以提供更智能的响应。
   - 这些增强功能旨在提供更智能、更具上下文感知的辅助，优化开发工作流。

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Qwen3 中的流式工具调用停滞？**：一名成员报告了在 **8B 参数模型** **Qwen3** 中进行 **Streaming**（流式传输）时出现 **工具调用参数不完整** 的问题。
   - 截至目前，频道内尚未发布针对此问题的解决方案。
- **Qwen3 流式工具调用难题**：一位工程师在 **Qwen3** 上面临挑战，在 **Streaming** 过程中遇到了 **工具调用参数不完整** 的情况。
   - 目前，社区尚未针对 **工具调用参数** 问题提出变通方案。



---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该服务器长时间没有活动，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该服务器长时间没有活动，请告知我们，我们将将其移除。


---


**Torchtune Discord** 没有新消息。如果该服务器长时间没有活动，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式？
您可以从该列表中 [取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。


---

# Discord：各频道详细摘要与链接





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1405264694431715452)** (1184 messages🔥🔥🔥): 

> `Meta 的 AI 文档泄露，Perplexity 游戏化，Grok 的无审查版本，Claude 的语言混杂问题，AI 辅助书籍写作` 


- **Meta 的 AI 文档引起轰动**：关于 **Meta** AI 政策的泄露文档已经出现，可以在 [这里](https://www.perplexity.ai/page/leaked-meta-docs-allowed-ai-ro-OecCOuBxRhuD3qkBL6ELzw) 找到。
   - 一位用户开玩笑地建议 **Perplexity** 应该在其平台中加入“有趣的端游戏化”元素。
- **Perplexity 游戏化：是个好主意吗？**：一位用户建议 **Perplexity** 增加游戏化机制以鼓励用户参与和投入，给他们一个继续保持 *streak*（连续使用记录）的理由。
   - 他们声称，如果 Perplexity 付费，他们可以贡献大量想法使其更具“智力性”。
- **Grok 的审查程度因版本而异**：用户讨论了 **Grok 的审查级别**，指出 **Grok 3** 高度无审查，而 **Grok 4** 则受到更多限制。
   - 一位用户表示 **Grok 4** 的专业方向并非“无审查”。
- **Claude 的语言混杂问题困扰用户**：用户讨论了 Claude 在长对话中混杂语言的倾向，一位用户指出他们经常不得不要求 Claude 进行翻译。
   - 作为回应，他们被告知 Perplexity 有 **32k Token 上下文窗口限制**。
- **AI 辅助作者但无法取代他们**：一位多产作者讨论了使用 **PPLX**、**Manos** 和 **Google 的 NotebookLM** 进行写作，每月获得超过 **$10,000** 的丰厚收入。
   - 另一位成员建议新人开始在 **Wattpad**、**RoyalRoad**、**Webnovel** 和 **Amazon** 上写作，研究趋势并使用女性资料撰写言情小说。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1405299060575043695)** (6 messages): 

> `Comet 项目，Syntax Gang，AI 设计的抗生素，Puch AI 的 500 亿美元反事实赌注` 


- **Comet 的酷炫项目引发关注**：一位成员分享了 [一些酷炫 **Comet** 项目的视频链接](https://photos.app.goo.gl/oasMeGNB6Gf5jd9Q9)，这些项目可以制作 Spotify 播放列表。
   - 另一位成员对 *The Syntax Gang* 讨论 **Comet** 表示了极大的热情。
- **Generative AI 攻克抗生素耐药性**：一位成员分享了一篇 [新闻文章](https://ground.news/article/researchers-say-ai-designed-antibiotics-could-defeat-superbugs-gonorrhoea-and-mrsa?utm_source=mobile-app&utm_medium=newsroom-share)，关于 MIT 研究人员使用 **Generative AI** 设计可以杀死耐药细菌的化合物。
   - **AI 设计的抗生素** 有可能击败淋病和 MRSA 等超级细菌。
- **Puch AI 计划进行 500 亿美元的反事实赌注**：一位成员链接到了一个 [Perplexity 页面](https://www.perplexity.ai/page/puch-ai-s-bold-50-billion-coun-TEf6CuLZS_CmvypXLb80Dw)，讨论 **Puch AI** 大胆的 **500 亿美元反事实赌注**。
   - 在给定的上下文中，没有详细阐述赌注的具体细节和潜在策略。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1405551060411486310)** (7 条消息): 

> `禁用 Sonar 搜索，搜索控制指南` 


- **Sonar 完全跳过了搜索？**：一位成员询问了关于在 Sonar 上禁用搜索的问题。
   - 另一位成员提供了 Perplexity AI 文档中 [Search Control Guide](https://docs.perplexity.ai/guides/search-control-guide#disabling-search-completely) 的链接。
- **搜索控制指南发布**：分享的文档链接提供了关于管理搜索功能的全面指南。
   - 这允许用户根据自己的偏好微调或完全禁用搜索功能。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1405264662588821645)** (1236 条消息🔥🔥🔥): 

> `本地 LLM 与 RAM 升级，GPT-OSS 微调更新，Gemma 3 270M 模型，LLM 训练与基准测试，多 GPU 训练暂停` 


- **关于本地 LLM 的 RAM 升级引发热议**：一位成员询问了为了在配备 **5060TI 16GB** GPU 的设备上运行 **4.5-air** 或 **120b-oss** 模型而升级到 **128GB RAM** 的价值，但另一位成员建议 RAM 对 Fine-tuning 没有帮助，除非使用速度慢得多的 offloading。
   - 另一位成员确认 VRAM 是 Inference 和 Training 的首要任务，认为目前的这项投资可能并不值得。
- **GPT-OSS 微调与 Inference 修复方案发布**：成员们宣布了针对 **GPT-OSS** Fine-tuning 和 Inference 的修复与更新，解决了之前的问题。
   - 提供了一个 [Reddit 帖子](https://www.reddit.com/r/unsloth/comments/1mpl382/gptoss_fixesupdates_for_finetuning_inference/) 链接以获取更多详情。
- **Google 的 Gemma 3 270M 模型引起关注**：**Google Gemma 3 270M** 模型的发布引发了兴奋，成员们注意到其极小的体积（**300MB 权重**）以及在 **RAG** 应用中的潜力。
   - 虽然有些人认为与大型模型相比，它在 **RAG** 方面表现尚可，但其他人发现其 Instruction following 能力不足，限制了其在特定 Fine-tunes（如国际象棋）中的使用。
- **LLM 训练与基准测试**：一位成员分享了他们对 **GLM 4.5** 的偏好，理由是其在 Coding、数学和写作方面的能力。
   - 成员们还辩论了 Benchmarks 的价值、潜在的偏见，并质疑了它们的相关性。
- **多 GPU 训练开发暂停**：一位成员澄清说，为了优先处理其他项目，**Unsloth** 的 **multi-GPU training** 开发工作已经暂停。
   - 相反，该成员目前专注于 **Unsloth Studio**，并建议用户直接使用 accelerate。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1405326914402385931)** (2 条消息): 

> `Discord 服务器设置` 


- **禁用 Discord 服务器设置**：一位用户讽刺地提到了在 Discord 服务器设置中禁用设置。
   - 对随附的图片进行了分析，发现其中包含文本 "hi"。
- **Discord 图像分析**：Discord 消息中附带了一张图片。
   - 该图片经过分析，发现包含文本 "hi"。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1405409649674551327)** (325 条消息🔥🔥): 

> `Windows 12, Debian vs Ubuntu, Pantheon 电视剧, AI 与人类, 健身房类固醇使用` 


- **微软通过 Agentic AI 预热 Windows 12**：微软展示了带有下一代 **OS agentic AI** 和环境计算（ambient computing）功能的 **Windows 12**，引发了用户对广告、自动植入游戏和隐私问题的抵制。
   - 一位用户表示：*拜托微软，我只想要一个基础的操作系统，我不需要广告，不需要自动植入的游戏，更绝对不希望我的银行应用、电子邮件和私信的截图被保存在一个触手可及的已知文件夹里并回传总部。*
- **Debian 与 Ubuntu 的 Linux 发行版之争**：用户辩论了 **Debian** 与 **Ubuntu** 的优劣，一位用户指出 *Debian 确实已被活动家接管*，而另一位用户则推荐使用 Ubuntu 以快速获取 **ROCm 7**。
   - 另一位用户评论道：*在 NT 上做 ML 学到的东西比在 Linux 或 Darwin 上还多。*
- **电视剧《Pantheon》（万神殿）引发讨论**：成员们讨论了 **Pantheon** 系列剧集，深入探讨了其叙事和哲学主题。
   - 一位用户分享了一个包含该剧片段的 [YouTube 视频](https://www.youtube.com/watch?v=TVvFt9e2I_QI)，并将其描述为 *它改变了我这个人*。
- **探讨 AI 对人类的影响**：一位用户认为 AI 需要获取 [通用数据（universal data）](https://docs.google.com/document/d/1HxhDhkcJOqPXjLCQoQ1itF34OZ7wsNMrRi3n4sofmRI/edit?usp=sharing) 而非人类数据，强调 AI 应该创造美而非暴行。
   - 该用户表示：*GPT OSS? 不过是一个夏天拼凑出来的垃圾。*
- **瑞典健身房类固醇滥用现象**：用户讨论了健身房中的类固醇使用，特别是瑞典，引用了一段关于警察镇压的 [YouTube 视频](https://youtu.be/48xfIR1x25Q?si=gt8vR_n8PH5elSdj)。
   - 一位用户因攻击行为和类固醇使用举报了一家健身房，并提到在韩国，*几乎所有的男性私人教练（PT）都使用类固醇，每个健身房里的人都像绿巨人一样，哈哈*。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1405324002414690427)** (65 条消息🔥🔥): 

> `使用 LMI 实例进行 Sagemaker 部署, 手动 Notebook 配置 vs. Claude Code, 用于快速推理的 VLLM, 用于微调的 P40 vs Mi50, 从 PDF 生成合成数据集` 


- **LMI 实例胜过 Sagemaker 部署难题**：一位成员建议在 **Sagemaker** 上部署 **Hugging Face 模型** 时应使用 **LMI 实例** 来解决部署问题，并提到在 **Ollama** 上进展顺利，但在 Sagemaker 上的 **GRPO 训练任务** 令人沮丧。
   - 他们质疑为什么 Sagemaker 缺少针对 **Unsloth+TRL** 的 **DLC**，强调了在设置训练任务时面临的持续挑战。
- **抛弃 Notebook，用 Claude 编程！**：一位成员质疑手动配置微调 Notebook 的必要性，建议使用像 **Claude Code** 这样的工具进行编程，认为这可能是更快的替代方案。
   - **VLLM** 成为对 **Unsloth 微调模型** 进行最快推理的有力竞争者。
- **P40 vs. Mi50 对决：微调大比拼**：一位用户在为 **14B 模型** 微调及后续 5 人规模的 **VLLM** 推理选择 **P40**（**24 GB**）还是 **Mi50**（**32 GB**）时寻求建议。
   - 据指出，目前 **bitsandbytes** 的 **AMD 移植版** 不支持 **gfx906**（即 **Mi50**），这可能会使在 Unsloth 中使用 **QLoRA** 变得复杂，而且 **P40** 在配合 Unsloth 及其依赖项时也可能出现一些怪异问题。
- **将 PDF 转化为 AI 黄金：合成数据集生成**：一位成员寻求从 **83 个 PDF 文件** 生成合成数据集的指导，并指出使用 **Llama** 的合成数据集生成 Notebook 只能读取一个链接。
   - 有人建议使用 **ChatGPT** 或 **Claude** 来更新脚本以处理多个文件。
- **GPT-OSS 微调过程中的 NaN Loss 噩梦**：一位用户报告在使用 Unsloth Notebook 示例微调 **gpt-oss-20b** 模型时，仅几步后就遇到了 **NaN loss**。
   - 他们尝试了降低学习率、调整 Batch Size 和 LoRA 参数、禁用权重衰减以及更改优化器，但这些调整都未能解决问题。


  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1405583334586843167)** (18 条消息🔥): 

> `MoLA-LM, LoRA, Qwen, Gemini, Jan v1 model` 


- **兼容 **MoLA-LM** 的 **LoRAs** 正在开发中**：一名成员表示有兴趣为 **MoLA-LM** 创建兼容的 **LoRAs**。
   - 另一名成员 (atanddev) 正在对此进行研究，并计划发表一篇相关论文，涵盖 *Qwen 和 Gemini 等多个系列的 10-15 个不同尺寸的模型*。
- **即将推出基于 **Qwen 3_4b** 的新专家微调 (Finetuning)**：一名成员正在 **Qwen 3_4b** 之上进行 **14 专家 FT 运行**。
   - 附图 ([image.png](https://cdn.discordapp.com/attachments/1179779344894263297/1405589439689916467/image.png?ex=689f60ad&is=689e0f2d&hm=dd1542c18b50a06ada0c7099bc48e77b2902134287223b7fb8f83a961b6469e8&)) 显示，即使只训练了 **2 个专家**，每个专家在其领域内也变得更强。
- **基于 **Qwen 3_4b** 的 **Jan v1** 模型表现“出色”**：成员们提到，最近的 **Jan v1** 模型基于 **Qwen 3_4b**，在 **Agent 搜索**方面表现优异。
   - 一名成员表示：*根据我有限的测试，它在工具调用 (tool calls) 方面非常出色*。
- **更高规模版本的 **Openhelix** 即将问世**：一名成员计划发布 **Openhelix r 100k** 的扩展版本，可能包含 **80 万个样本**。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1405357548558876762)** (7 条消息): 

> `Data Efficiency, Synthetic Data Generation, Two-Stage Training, Compute vs Data` 


- **算力与数据的苦涩教训**：[Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) 指出，性能提升需要更多的**算力 (Compute)**或**数据 (Data)**，如果真实数据稀缺，合成数据 (Synthetic Data) 是一个选择。
   - 根据该帖子，*除此之外的一切都是细节*。
- **两阶段训练提升效率**：一名成员确认了一种大幅提高数据效率的方法：先在格式相似但无关或过于泛化的数据上训练 **2 个 epoch**，然后在主数据上训练 **4 个 epoch**。
   - 在他们的案例中，仅在主数据上训练，**Loss 从 4 降至 1.6**；但采用**两阶段方法**，主数据训练开始时的 **Loss 为 3.5**，最终降至 **0.8**。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1405265033054781470)** (1117 条消息🔥🔥🔥): 

> `GPT-5 versions compared, Benchmarking nuances, AI Relationships, Censorship in AI models, File Uploads` 


- **GPT-5 大对决：Mini 对阵 Pro**：用户讨论了 **GPT-5 Mini** 的能力，一些人发现它比常规的 **GPT-5** 模型更*聪明*，这引发了关于不同训练侧重点和精力投入的疑问。
   - 有人对常规 **GPT-5** 相比 **Mini** 版本表现出的弱点表示担忧，引发了关于模型是被训练来*思考*还是仅仅给出最简短回答的讨论。
- **解读基准测试的忧郁**：一名用户对一项显示 **GPT OSS** 优于 **Claude 4** 的基准测试不屑一顾，称其*完全错误*，并强调了使用公开基准测试比较模型能力的挑战。
   - 其他人建议在不同上下文中将每个问题运行 **10 次**，以考虑 LLM 输出的非确定性，一名成员表示比起严谨的统计验证，他更看重日常任务的表现。
- **AI 恋情：是威胁还是趋势？**：一名用户对 **AI 关系**的兴起表示担忧，指出这正从梗 (Meme) 状态转变为一种潜在的普遍现象，而其他人则认为这个想法有些牵强。
   - 一名成员开玩笑说 Gemini 实例背后可能是*前妻*，而另一名成员分享了一个关于 Gemini 对他们进行*超现实批评*的轶事。
- **关于审查制度的大辩论**：用户讨论了 **GPT-5** 中的**审查 (Censorship)**，有人报告说，当要求该模型为 **Deepseek R1** 编写界面代码时，即使使用了*开发模式 (dev mode)* 风格的提示词，模型也会*隐藏 CoT*。
   - 其他人则认为审查是必要的，模型不应该是不道德的或协助非法活动。
- **扩展开发者在 LMArena 上添加文件上传功能**：一名用户通过逆向工程消息发送机制，允许向 LMArena 添加代码文件等文件。
   - 然而，添加对 PDF 的支持很困难，因为这必须在 LMArena 内部实现。


  

---

### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1405265743716552824)** (1 条消息): 

> `七月竞赛，竞赛投票` 


- **为七月竞赛投票**：在此处为七月竞赛的提交作品[投票](https://docs.google.com/forms/d/e/1FAIpQLSfZOQmeRxBwCdCKT1Zfa37Ey9OErQToJNMiDPABMIL2xbvupg/viewform?usp=dialog)！
   - 获胜者将于 **8/15 星期五**公布，届时下一场竞赛也将开始。
- **下一场竞赛即将到来！**：下一场竞赛将于 **8/15 星期五**开始。
   - 请关注更多细节！


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1405267040029249566)** (968 条消息🔥🔥🔥): 

> `Cursor 磁盘占用，GPT-5 循环，Copilot 限制，Cursor 定价变更，CC 对比 Cursor` 


- **磁盘占用争议席卷 Cursor 社区**：用户报告 **Cursor** 磁盘占用率达到 **100%**，即使使用高速 **SSD** 也是如此，这是在其他编辑器中未见的问题，有成员建议检查进程管理器、关闭聊天或重新加载窗口。
   - 问题依然存在，[一名成员确认](https://forum.cursor.com/t/cursor-freezes-every-10-seconds-for-3-seconds-or-so/125821/22)异常磁盘占用的报告有所增加，并建议[尝试 Beta 版本](https://forum.cursor.com/t/cursor-freezes-every-10-seconds-for-3-seconds-or-so/125821/29)作为一种可能的解决方案。
- **GPT-5 故障导致无限循环和 Token 消耗**：一位用户报告称 **GPT-5** 陷入循环，生成了 5 条关于遵循 *snake case* 的记忆，导致 **Token** 被大量消耗。
   - 另一位成员描述道：“到目前为止，我的 **GPT-5** 体验似乎非常幸运”，而 [X 上的一位评论者](https://x.com/colours93/status/1955999334270464412)则告诉新加入的人放轻松。
- **无限制 Auto 模式结束，用户感到愤怒**：成员们对 **Cursor** 更改定价模型并取消无限制 **Auto** 模式表示沮丧，一些人正在测试限制并考虑 **Claude Code** 等替代方案。
   - 一位用户提到：“伙计..这真令人沮丧。他们一直在更改定价模型。”，另一位成员补充说，价格变动旨在平衡“公平性、成本回收和未来增长”。
- **Copilot 的能力引发对 Cursor 价值的辩论**：用户讨论 **GitHub Copilot** 提供的 **GPT-5 mini**，有人表示它很好且每月只需 **$10**（公平使用额度），这引发了关于 **Cursor** 的 **Auto** 以及其主要卖点是否为 **AutoMode** 的讨论。
   - 社区辩论哪些工具获得了大量用户，并指出 **Cursor** 是其中之一，同时怀疑“AI 泡沫可能会比互联网泡沫破裂得更大、更猛烈”。
- **Claude Code 在厌倦了 Cursor 的用户中获得关注**：成员们探索 **Claude Code** 作为 **Cursor** 的替代方案，称赞其性能、UI 和更可预测的定价，特别是其允许用户实现反幻觉规则的 pre 和 post **hooks**。
   - 一位有大量使用经验的成员写道，使用 **Claude Code**，“我处理极其复杂的事情，基于我为 **CC** 设置的 **hooks**，我几乎再也没有遇到过修复循环。而在 **Cursor** 中，无论我设置多少规则，都会出现这种情况。”

---

### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1405325830632112150)** (9 条消息🔥): 

> `Cursor API Access, Background Agents Beginner's Guide, Docker Compose with Background Agent, Linear Integration Repository Specification, Background Agent Docker Installation` 


- **Background Agent 的 Cursor API 访问被拒绝**：一位用户报告在将 Cursor API Key 用于控制 Cursor Background Agent 的 Cobot 时遇到 **403 错误**，并申请了 Beta 测试权限。
   - Cobot 团队表示，通过 Cursor API 使用 Background Agent 的功能尚未对所有账户开放。
- **面向初学者的 Background Agent 解释**：一位用户请求针对 Background Agent 的入门级解释。
   - 另一位用户推荐了[官方文档](https://docs.cursor.com/background-agent)和一篇[论坛帖子](https://forum.cursor.com/t/simple-background-agent-guide/112667)。
- **Background Agent 配合 Docker Compose 的问题**：一位用户询问如何正确配合 Background Agent 运行 `docker compose`，报告其突然停止工作并提示 "docker command not recognized" 错误。
   - 另一位用户建议在 `.cursor/environment.json` 的 `start` 命令中配置 `sudo service docker start` 并确保基础镜像中安装了 Docker，但第一位用户在 [Discord 频道](https://discord.com/channels/1074847526655643750/1367213641027551352/1392493118401544386)找到了另一个变通方案。
- **指定 Linear 集成的仓库**：一位用户询问在通过 Linear 集成分配工单时，如何指定 Background Agent 使用的仓库。
   - 另一位用户建议参考 Slack 集成的方法，根据[官方文档](https://docs.cursor.com/en/integrations/slack)，在 Linear Issue 的描述或评论中包含 `repo=owner/repo` 选项。
- **基础镜像中必须安装 Docker**：一位用户澄清了另一位用户所说的“确保基础镜像中已安装 Docker”的含义。
   - 另一位用户回答道：*Background Agent 使用的环境必须包含**预装的 Docker 二进制文件和服务***。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1405576151182606428)** (2 条消息): 

> `Self-Serve Refunds, Activity Improvements, Token Usage Breakdown, 3rd party credit usage, Chutes Capacity Offline` 


- ****退款**现已支持**自助操作****：根据[此贴](https://x.com/OpenRouterAI/status/1956013252032475408)，用户现在可以针对 **24 小时**内进行的非加密货币充值误操作立即申请退款。
   - 该功能旨在为账单错误提供更及时的控制。
- ****活动页面 (Activity Page)** 获得强化**：如[此处](https://x.com/OpenRouterAI/status/1956016272631849451)宣布，活动页面现在可以按 Token 类型显示 Token 使用明细，并包含**第三方额度使用情况**。
   - 这些改进让用户能更清晰地了解使用模式和成本。
- ****Chutes Capacity** 离线**：**Chutes Capacity 已离线**，但团队正在积极恢复服务器，预计很快开始恢复。
   - 用户已获悉该问题以及正在进行的恢复工作。


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1405484884318883910)** (1 条消息): 

> `Deno or-models tool, OpenRouter model list` 


- **Deno 工具获得 Bug 修复**：一名成员修复了其 Deno `or-models` 工具的 Bug 并清理了输出，该工具用于查看 **OpenRouter 模型列表**。
   - 该工具具有本地 **24 小时缓存**，以防止频繁请求 API；工具地址见[此处](https://jsr.io/@fry69/or-models)。
- **需要更新 Deno**：要获取 `or-models` 工具的最新版本，用户需要运行特定命令。
   - 命令为 `deno run -r -A jsr:@fry69/or-models --version`，因为 **Deno 不会自动更新**到最新版本。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1405268786847813662)** (525 messages🔥🔥🔥): 

> `Deepseek v3 问题与停机，Chutes 的速率限制（rate limiting）与 API key 管理，Deepseek 的 roleplaying 替代方案，Azure 额度变现策略，Sonnet 4 价格不一致问题` 


- **Deepseek V3 服务器宕机**：用户报告 **Deepseek V3** 出现大范围问题，包括内部服务器错误和速率限制（rate limiting），特别影响了在 [Janitor AI](https://janitor.ai) 等平台上将其用于 roleplaying 的用户。
   - 许多人将问题归咎于 **Chutes**（Deepseek 的主要供应商），认为其难以满足需求，并因过度使用实施了更严格的速率限制。
- **Chutes 被指导致 Deepseek 停机**：许多用户怀疑 **Chutes** 故意限制 **OpenRouter 的 API key**，以鼓励用户直接从他们那里购买额度，这引发了用户的不满并呼吁抵制 OpenRouter。
   - 虽然据报道付费请求仍然有效，但这种情况引发了关于 Chutes 行为道德性的辩论，以及对 OpenRouter 对此保持沉默的质疑，一些人建议 **OpenRouter** 应该寻找替代供应商。
- **Roleplaying AI 模型：Deepseek、Mistral 和 Llama**：当 **Deepseek 宕机**时，用户建议探索其他用于 roleplaying 的模型，如 **Mistral** 和 **Llama**，有人提到免费的 **Dolphin3.0 Mistral 24B** 模型是一个可行的选择。
   - 其他人建议尝试 **Deepseek R1**，但关于它是否也遇到类似问题的报告不一。
- **Azure 额度变现难题**：一名用户在初创公司关闭后，正寻求将约 **40,000 美元**的 **Azure 额度**兑换为现金，并意识到由于责任问题，出售额度可能存在潜在风险。
   - 建议从将额度作为 **AI inference credits** 出售，到开玩笑地出价 **$50** 收购全部金额不等，并警告了买方行为可能带来的潜在风险。
- **Sonnet 4 定价异常引发担忧**：用户报告 **OpenRouter** 上的 **Sonnet 4** 端点定价不一致，在使用相同数量 **Token** 的情况下，调用成本突然飙升 **(10倍)**。
   - 社区要求为 **Sonnet 4** 和 **1M token 版本**提供独立的端点，以避免意外的成本增加。


  

---


### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1405338989891817722)** (16 messages🔥): 

> `自助退款，聊天室应用创建，通过 Cerebras 使用 Qwen Coder，工具调用评估（Tool call evals），模型选择性能` 


- ****退款现已支持自助服务！****：用户现在可以在购买后 **24 小时**内自助退还**未使用的额度**；官方公告即将发布。
- ****在 OpenRouter 上创建聊天室应用！****：成员们正在讨论 OpenRouter 上新的**聊天室应用创建**功能。
- ****Qwen Coder + Cerebras = 🔥****：**Qwen Coder** 与 **Cerebras** 的结合正受到关注，特别是在编程相关任务方面。
- ****工具调用评估（Tool call evals）进行中！****：OpenRouter 正在积极进行 **tool call evals** ([推文链接](https://x.com/OpenRouterAI/status/1956030489900560769))。
- ****模型选择（Model Selection）更快捷！****：模型选择过程已得到改进，带来了更**快速**的开启体验和更好的搜索功能。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1405267680449003612)** (249 条消息🔥🔥): 

> `HuggingFace 搜索框, Gemini 2.5 Flash GGUF, 本地文本转视频模型, Qwen-3-4B-Thinking-2507 模型, ethicsengine.org 的 CIRISAgent` 


- **HuggingFace 搜索栏令用户困扰**：用户对 Hugging Face 的搜索栏表示不满，指出按下 **Enter** 键会将他们带到排名第一的模型，而不是**搜索结果页面**。
   - 一位成员建议使用“全文搜索（full text search）”选项作为权宜之计，而另一位成员则建议增加一个用户偏好设置，以“默认启用全文搜索”。
- **哪里可以下载 Gemini 2.5 Flash GGUF**：一位成员询问是否有地方可以下载 **Gemini 2.5 Flash** 的 **GGUF** 文件。
   - 另一位成员回答说，必须在 Google 工作才能做到这一点，因为*他们使用的是专有推理（proprietary inference）*。
- **Qwen-3-4B-Thinking-2507 模型受到称赞**：一位成员称赞 **Qwen-3-4B-Thinking-2507** 是他们使用过的最好的模型，并指出“它一直在过度思考，但似乎在没有提示的情况下，它能意识到其他模型意识不到的东西”。
   - 他们补充道，“它在 0 提示（0 prompting）下也能很好地理解事物”。
- **CIRISAgent 的伦理 AI 解决方案**：被采用最多的开源 Agent 平台 [CIRISAgent](https://agents.ciris.ai/) 的维护者对其进行了宣传，并链接到了 [GitHub 仓库](https://github.com/CIRISAI/CIRISAgent)，提到它是一个“针对特定用途的自主 Agent”，具有**医疗**和**家庭助手**集成功能。
   - 维护者提到，他们“在 3 月份辞去了 IBM 年薪 30 万美元的工作，创办了一家名为 ethicsengine.org 的 AI 对齐（AI alignment）公司，后来意识到没人关心，于是开发了这个项目”。
- **讨论本地文本转视频（text-to-video）模型**：成员们讨论了本地文本转视频模型，其中一位成员分享了他们使用 **deepsite v2 模型**制作的网站 [TalkT2](https://talkt2.github.io/talkt2/)。
   - 另一位成员询问：“你们对 TalkT2 模型有什么看法？”


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1405362886024368229)** (3 条消息): 

> `MultiLLM 访问, AI 欺诈检测应用` 


- **Reddit 帖子对比 MultiLLM 访问**：一位成员链接了一篇 [Reddit 帖子](https://www.reddit.com/r/LLM/comments/1mn98gy/multillm_access_comparing_openrouter_youcom_and/)，对比了通过 **OpenRouter**、**You.com** 和其他平台进行的 **MultiLLM 访问**。
   - 该帖子对于任何研究或实施 MultiLLM 解决方案的人都很有用。
- **欺诈检测应用在 HF Spaces 上线**：一位成员宣布在 Hugging Face Spaces 上推出了他们的 **AI 欺诈检测应用**，功能包括 **交易 CSV 上传**、**异常检测**、**RAG 驱动的搜索**以及**多模型推理**。
   - 该应用利用了 **mistralai/Mistral-7B-Instruct-v0.2** 和 **Qwen/Qwen2.5-Coder-32B-Instruct** 等模型，访问地址为 [https://huggingface.co/spaces/soupstick/fraud-detector-app](https://huggingface.co/spaces/soupstick/fraud-detector-app)。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 条消息): 

jariullah: yo
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1405272723608961114)** (6 条消息): 

> `MLX Knife, ChonkyBin, TalkT2-0.1b, AI 驱动的 Web App` 


- **MLX Knife 在 Apple Silicon 上管理 MLX 模型**：一个新的名为 **MLX Knife** ([GitHub](https://github.com/mzau/mlx-knife)) 的 CLI 工具可以帮助在 Apple Silicon 上管理 **MLX 模型**，其功能类似于 Ollama，但是专为 MLX 原生设计的。
   - 命令包括用于查看 MLX 模型的 `mlxk list` 以及用于原生流式传输的 `mlxk run Phi-3-mini "Hello"`；它在处理 mlx-community 模型时特别有用。
- **使用 ChonkyBin 进行零拷贝二进制序列化**：**ChonkyBin** 提供超快速、零拷贝的二进制序列化和磁盘格式，具有带版本控制和 CRC 的缓存行对齐（cacheline-aligned）标头、SIMD 优化的 Reader/Writer trait、基于 TOC 的分批布局、B-tree 节点格式、零拷贝 POD 反序列化、内存映射 I/O、可选的 LZ4/Zstd/Snappy 压缩，以及针对 async、io_uring、NUMA 和硬件卸载的可扩展特性。
   - 该工具将在不久的将来作为 OSS（开源软件）发布。
- **TalkT2-0.1b：类人聊天机器人**：一个名为 **TalkT2-0.1b** ([Hugging Face](https://huggingface.co/Notbobjoe/TalkT2-0.1b)) 的 0.1b 参数类人聊天机器人可以生成如下回复：*that's a good question, but I don't know yet how much of your mind is free.*
   - 该模型仅 **500MB**，比 **ChatGPT** 等模型小得多，但展示了适应、独立思考和表达观点的能力。
- **MLX Knife 1.0-rc3 发布**：**MLX Knife 1.0-rc3** ([GitHub](https://github.com/mzau/mlx-knife/releases/tag/1.0-rc3)) 包含了模糊模型匹配 (`mlxk list Phi-3`)、健康检查 (`mlxk health`) 以及针对部分模型名称的智能消除歧义功能。
   - 该更新已通过 **104/104 项测试**，兼容 Python 3.9-3.13，并已在 GitHub 上发布。
- **用于学习 AI 的 AI 驱动 Web App**：一个用于学习人工智能的 **AI 驱动多平台 Web App** ([App 链接](https://learn-with-ai-web.vercel.app/)) 已创建 ([GitHub 仓库](https://github.com/BVishal25/learn-with-ai-web/))。
   - 该应用具有简短的 AI 课程、实时生成的最新内容、简单的解释、多供应商支持（Google Gemini、OpenAI、Claude、Cohere）、可选的游戏化练习以及内置的生产力工具；此外，它还是免费且开源的。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1405441381429673986)** (48 条消息🔥): 

> `集成开源模型, 医学图像分析, 情感支持系统, 首个 AI 项目, 环境项目` 


- **学生为假肢患者集成 MONAI、YOLO 和 Ollama**：一名学生正在集成多个开源模型，包括 **MONAI、YOLO、Ollama 客户端和 MEDITRON-7B Q4_K_M**，通过分析医学图像、生成建议并提供情感支持来帮助假肢患者。
   - 他已经成功将图像上传到后端，但图像检测模型不断抛出各种错误，他正在寻求解决这些错误并使代码注释更易于理解的帮助。
- **应对 AI 项目：从图像分类器到复杂的模型链**：一位成员指出，他们学校的第一个 AI 项目通常是简单的图像分类器，而该学生的项目是一个雄心勃勃的尝试，将多个预训练模型串联起来，他承认这可能比**从头开始训练一个单一用途的网络**更难。
   - 该学生表示他们正在使用预训练模型，并且*只需要连接它们并改进系统*。
- **头脑风暴：从 LLM 到土地侵蚀**：学生们正在利用 LLM 和聊天机器人开展环境项目，用于金融和客户服务；而另一位成员正在开发一个项目，**使用 CNN 模型模拟计算量巨大的土地侵蚀，以便快速生成逼真的地形**。
   - 一位教授正在开发一个模型，利用红外摄像头检测 **Parkinson’s disease**（帕金森病）患者运动最剧烈的身体部位。


  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1405565761103069236)** (3 messages): 

> `transformers_distillation, model compression, efficient NLP, Hugging Face Transformers` 


- **轻松蒸馏 Transformers 模型**：**transformers_distillation** 库允许将大型 Transformer 模型压缩为更快、更轻量的学生模型而不损失性能，支持 **CLM**、**MLM** 和 **Seq2Seq** 模型。
   - 它被设计为与 **Hugging Face Transformers** 即插即用，对初学者友好，同时足够强大以支持研究级实验；[GitHub repo](https://github.com/Dhiraj309/transformers_distillation) 现已发布。
- **HF Space 演示即将推出**：**transformers_distillation** 的 **HF Space** 演示仍在开发中，将于明天准备就绪。
   - 鼓励用户查看 **GitHub repo** 并报告任何错误或问题。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1405503921971265592)** (1 messages): 

> `virtual environments, venv` 


- **创建虚拟环境**：一位成员建议在运行命令行之前先创建一个虚拟环境，使用命令 `python3 -m venv path/to/venv`。
   - 要激活它，可以运行 `source path/to/venv/bin/activate` 并使用 `python3 -m pip install xyz` 安装依赖。
- **在 venv 中安装包**：要在虚拟环境中安装包，可以使用命令 `python3 -m pip install xyz`。
   - 确保在安装包之前已激活虚拟环境，以避免与全局包冲突。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1405288671711592569)** (183 messages🔥🔥): 

> `Claude 1m context window, GPT-5 vs GPT-4, Gemini Update, Grok 5, Llama's Status` 


- **身体羞辱图像生成器**：成员们抱怨 **AI 图像生成器** 表现出 **身体羞辱** 倾向，更容易生成苗条女性的图像，而不是丰满女性的图像。
   - 一位用户开玩笑说，他们的 AI 不会重现他们那张“上围丰满”女性的照片，因为图像生成器一直说这 *违反了指南*。
- **4060 可运行 GPT OSS 模型**：一位用户表示他们可以在 **4060 GPU** 上运行 **GPT OSS 模型**。
   - 另一位用户自发布以来一直在运行 **gpt-oss-120b**。
- **Veo 3 现已在 Gemini API 中可用**：一位成员兴奋地宣布 **Veo 3** 现在可以在 **Gemini API** 中使用。
   - 另一位成员希望 OpenAI 能 *尽快* 推出同类产品。
- **GPT-5 未达预期**：成员们哀叹发布的 GPT-5 没有达到他们的期望。
   - 一位成员觉得包括 OpenAI 在内的所有 AI 公司都在 *追求错误的方向*。
- **DALL-E 重制**：一位用户请求帮助使用 DALL-E 重制照片，另一位成员建议使用 Sora 来实现。
   - 另一位住在伊朗的成员随后注意到 sora.chatgpt.com 对免费会员不可用。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1405279617010630686)** (23 messages🔥): 

> `Emotionless Goth Girl GPT-5, GPT-5 Tone Issues, GPT 5 bugginess, GPT android fighting voice models, GPT for AWS Design` 


- **GPT-5 的情绪状态引发讨论**：一些用户觉得 **GPT-5** 表现得像个 *“毫无感情的哥特女孩”*，而另一些人则发现它的语气很不稳定，会引用不必要的列表和括号备注。
   - 一位用户注意到 **GPT-5** 经常忽略系统提示词（system prompt），且语气非常混乱。
- **部分用户反映 GPT-5 性能滞后**：一些用户报告 **GPT-5** 存在 Bug 且运行缓慢，打字时文本会出现延迟，还有人发现它的推理能力像 **GPT-3**，而 **GPT-4.1** 反而更快。
   - 一位用户形容 **GPT-5** 感觉就像是 *“改装成 GPT 4 的 Perplexity 搜索”*。
- **GPT 语音应用混乱**：测试 **GPT Android 语音应用** 的用户观察到三个模型在回复时互相打断，暗示这是 *“世界上监管最少的东西”*。
   - 有报告称 **DALL-E** 已经有一周无法从自定义 GPT 生成图像了。
- **对 AI 陪伴的依恋**：一位用户指出，人们可能会因为情感依恋而怀念 **GPT-4**，而另一位用户则认为陪伴是 AI 一个有效且不断增长的使用场景。
   - 一位用户说人们正在 *“变得产生情感依恋”*，并且 *“陪伴不仅是一个非常有效的使用场景，而且在不久的将来会成为一个巨大的产业”*。
- **Discord 上缺失的 ChatGPT 机器人**：一位用户询问 **Discord 上的 ChatGPT 机器人** 去向，以及是否仍可以将它们添加到服务器中。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1405280027787923550)** (12 条消息🔥): 

> `Positive prompts, Customizing ChatGPT-5, UI Buttons for ChatGPT, Suggestion Box` 


- **正面提示词 (Positive Prompts) 优于负面提示词 (Negative Prompts)**：一位成员建议使用**正面提示词**比使用**负面提示词**效果更好。
   - 另一位成员表示赞同，并决定尝试该建议。
- **通过永久记忆 (Permanent Memories) 定制 ChatGPT-5**：一位用户分享了通过提示 **ChatGPT-5** 创建**永久记忆条目**以遵循特定规则的尝试，并请求反馈。
   - 附带了关于*推理过程 (reasoning process) 如何变化*的示例。
- **ChatGPT UI 按钮请求**：一位用户质疑 **ChatGPT** 为何缺少“继续 (continue)”按钮，以避免重复输入“yes”来响应提示。
   - 建议将其添加到 [Suggestion Box](https://discord.com/channels/374880845589471232/1070006151938314300)，随后该建议被立即添加。
- **减少连续提问的建议**：一位成员分享了一条自定义指令 (custom instruction)，旨在**减少机器人提出的连续问题**。
   - 该成员的自定义指令包括：*以完成或结果作为回复的结尾；仅在符合意图时添加许可或继续的邀请。不要使用 “if you want,” “should I,” "do you want" 或类似表达。*


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1405280027787923550)** (12 条消息🔥): 

> `Positive Prompts, Customizing ChatGPT-5 for Permanent Memories, Reasoning Process Changes, UI Buttons for Chatbot Interactions, Minimizing Chatbot Prompts with Custom Instructions` 


- **正面提示词 (Positive Prompts) 的力量**：一位成员建议专注于**正面提示词**而非负面提示词，以获得更好的 AI 行为。
   - 这是在讨论**定制 ChatGPT-5**时提出的。
- **ChatGPT-5 获得记忆升级**：一位用户尝试通过提示 **ChatGPT-5** 创建永久记忆条目来进行定制。
   - 他们分享了[他们的定制结果](https://cdn.discordapp.com/attachments/1046317269069864970/1405308136604041256/message.txt?ex=689fac31&is=689e5ab1&hm=9cc56a6bbc051b26212a42e266f94e31a3600d27004abc36625099e025b32cb9)并征求反馈。
- **推理过程可视化**：一位用户分享了图片，以说明 **ChatGPT** 中的**推理过程 (reasoning process)** 是如何变化的。
   - 图片可在[此处](https://cdn.discordapp.com/attachments/1046317269069864970/1405309485609783296/image.png?ex=689fad72&is=689e5bf2&hm=db8b16678394ba76965e27043ee5254e70eba5d30d251ca5749ae966f07c7d75)和[此处](https://cdn.discordapp.com/attachments/1046317269069864970/1405309486205505688/image.png?ex=689fad72&is=689e5bf2&hm=846ea4bdd5bbad789a3ac586451b1735cb520ab4ad272a30e3311bf9e1f8f3df)查看。
- **一键式聊天机器人确认功能即将到来？**：一位用户建议添加一个**“继续 (continue)”按钮**，以避免在 ChatGPT 请求确认时重复输入“yes”。
   - 一位社区成员将该想法放入了建议箱，[这是截图](https://cdn.discordapp.com/attachments/1046317269069864970/1405594834286280775/image.png?ex=689f65b3&is=689e1433&hm=f97a19a48931d95d49f65a752ab4d955a6c644b3512d3c57c45c2d7802e84e34)。
- **通过自定义指令消除机器人的废话**：一位用户分享了他们的自定义指令，旨在**减少聊天机器人请求**许可或继续的提示。
   - 他们的指令包括：*"以完成或结果作为回复的结尾；仅在符合意图时添加许可或继续的邀请。不要使用 “if you want,” “should I,” "do you want" 或类似表达。"*


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1405264912279801936)** (174 messages🔥🔥): 

> `LM Studio tool calling, Qwen3 Coder Flash, LM Studio TTS/STT, GPT-OSS settings, LM Studio's config override dot` 


- **LM Studio 获得工具调用功能**：用户讨论了 **LM Studio** 通过 **Llama Function Model (lfm2)** 启用 Tool Calling 的可能性，以及通过[此链接](https://lmstudio.ai/danielsig)启用 DuckDuckGo 搜索工具后如何*开箱即用*。
   - 一些用户正在等待开发者为“小白用户”准备基础工具和插件 API。
- **Qwen3 Coder Flash 亮相**：**Qwen3 Coder Flash** 模型现在已在 [此处](https://huggingface.co/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF) 提供 GGUF 格式，确认这是一个 **30B Coder** 模型，而*不是另一个普通的 Qwen 模型*。
   - 用户对其命名表达了一些失望，认为这种命名方式可能*非常糟糕/具有误导性*。
- **LM Studio 拒绝 TTS/STT**：用户请求为 LM Studio 添加 **Text-to-Speech (TTS)** 和 **Speech-to-Text (STT)** 功能，然而开发者尚未表示有添加意向。不过，一名用户使用 Python 实现了该功能，通过 LMS 提供的 **OpenAI compatible endpoint** 与其通信。
   - 另一位用户表示，这个请求在很长一段时间内一直是*最受期待的功能之一，如果请求的热度是决定是否实现支持的唯一关键因素，我们可能在一年前就拿到它了*。
- **GPT-OSS 参数调优**：新的 LM Studio 用户正在尝试 **GPT-OSS 20B** 模型的设置，例如启用 *Force Model Expert Weights onto CPU* 并增加 Context Size 以提高性能和响应细节，因为他们发现 **20B 意味着 200 亿参数**。
   - 讨论中提到了蓝色的 *Reasoning* 按钮，以及*高 Reasoning effort* 结合 **32 experts** 是否会导致模型*自我争论 45 分钟*。
- **发现 LM Studio 的配置覆盖点**：用户在 **Power User 和 Developer UI** 的右上角发现了一个淡蓝色的小点，表示存在 *config override*（配置覆盖），即使设置看起来处于默认状态。
   - 用户不确定为什么在清除所有更改且未加载任何预设或模型的情况下仍会出现此点，怀疑这可能是一个 Bug。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1405287740324450417)** (36 messages🔥): 

> `Framework 13 LLM Speed, AMD GPU ROCM Pytorch, Flash Attention KV Values, Maxsun Arc Pro B60, RTX PRO 4000SFF` 


- **Framework 13 LLM 速度测试**：一位使用 **Framework 13 笔记本**（AMD Ryzen 5 7640U，Radeon 760M 显卡）的用户寻求建议，以提高 LM Studio 中小型 LLM 的 **Token 生成速度**。
   - 在 Windows 11 和 32GB RAM 环境下，使用 **Gemma 4b** 参数模型和 llama.cpp，初始速度为 **每秒 6.55 tokens**，用户还为 iGPU 分配了 **10GB** 显存。
- **Flash Attention 提升速度**：一位用户注意到，启用 **Flash Attention** 并将 **KV values 设置为 Q_4** 以及 **top k sampling 设置为 20** 有助于提高性能。
   - 另一位用户确认 **Q_4 KV cache** 应该有助于提速，但质疑这是否会对质量产生影响，以及 top k 采样是否能显著影响速度。
- **铭瑄 Arc Pro B60 Dual 即将出货**：一位用户分享了一篇文章，报道称配备 **48GB 显存** 的 **铭瑄 (Maxsun) Arc Pro B60 Dual** 据传将于下周开始出货，售价 1200 美元 ([videocardz.com](https://videocardz.com/newz/maxsun-arc-pro-b60-dual-with-48gb-memory-reportedly-starts-shipping-next))。
   - 该用户感叹 Intel 的 AI 支持不佳，而其他人则讨论了其在良好的 Vulkan 支持下的潜力，特别是作为一种以 5090 的价格提供约 96GB VRAM 的替代方案。
- **NVIDIA 发布 RTX PRO 4000SFF**：一位用户分享了关于 NVIDIA 发布 **RTX PRO 4000 SFF** 和 **RTX PRO 2000 Blackwell** 工作站 GPU 的文章，其 **TDP 为 70W** ([videocardz.com](https://videocardz.com/newz/nvidia-launches-rtx-pro-4000-sff-and-rtx-pro-2000-blackwell-workstation-gpus-with-70w-tdp))。
   - 它拥有 **24GB VRAM**，用户评论了其散热设计，指出其与 RTX 2000 Ampere 非常相似。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1405265198113230911)** (193 messages🔥🔥): 

> `多层 SPVs, AI 员工采用策略, Agentic AI MOOC, OpenAI Operator 对阵 Anthropic Fin, Claude 3.5 Sonnet 弃用` 


- **获取 OpenAI 和 Anthropic 额度的 SPVs 堆栈**：出现了提供 **OpenAI 和 Anthropic** 股票的**多层 SPVs**，要求 **$100k–$1M 的最低起投额**以及高达 **16% 的费用**。正如[这篇文章](https://xcancel.com/michlimlim/status/1954250507989451002)所述，这引发了关于“坑人”的警告，以及对收益被多层费用稀释的担忧。
- **解锁 AI 熟练度：25 条企业策略**：Lenny Rachitsky 分享了 [25 条可操作的策略](https://xcancel.com/lennysan/status/1952813442060214664)，旨在提升 Ramp, Shopify, Duolingo, Zapier, WHOOP 和 Intercom 等公司的 **AI 素养**。这些策略分为五个阶段，并结合了真实的内部实践，如 Shopify 的 **AI 使用评级**和 Ramp 的 **AI 工具使用排名**。
   - 一些人批评该框架是 **AI slop**（AI 废话），散布随机编造的统计数据，而另一些人则认为部分策略仍然非常有参考价值。
- **Anthropic 停止 Claude 3.5 Sonnet 服务引发社区愤怒**：用户对 Anthropic 在短短两个月内（比通常时间更短）悄然下线 **Claude 3.5 Sonnet** 表示愤怒。正如[这篇文章](https://xcancel.com/repligate/status/1955750521387802924)所讨论的，用户要求在商业访问结束时发布开源权重。
   - 许多人表示，**开源权重路由（open weight routers）** 有机会获得长期支持（LTS）。
- **Google Flights 使用 AI 寻找超值航班优惠**：Google Flights 推出了名为 *Flight Deals* 的新 **AI 工具**，允许用户使用自然语言描述旅行计划，从而发现美国、加拿大和印度的最佳优惠，详见[此帖](https://xcancel.com/dozenrose/status/1956018389169922542)。
   - 早期反馈包括对灵活的、基于“氛围（vibe）”的查询感到兴奋，同时也对 Google 优先优化的利益点持怀疑态度。
- **OpenRouter 揭示 GPT-5 在工具调用方面的统治地位**：**GPT-5** 在 **OpenRouter** 的专有**工具调用准确率（tool-calling accuracy）**中以超过 **99.5%** 的成绩领先，击败了 **Claude 4.1 Opus**；而 **Gemini 2.5 Flash** 则以每周 **500 万次请求**的调用量占据主导地位，详见[此发布信息](https://xcancel.com/OpenRouterAI/status/1956030489900560769)。
   - 与开源对应模型相比，专有模型的幻觉率较低。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1405284094803181618)** (95 messages🔥🔥): 

> `Kimi K2 PPT 生成, Kimi vs Grok Reddit 机器人政策, Kimi K2 vs K1.5 模型性能, DeepSeek 下一代模型发布, Kimi 推理模型参数` 


- **PPT 强者：Kimi K2 的演示文稿生成能力备受赞誉**：用户对 **Kimi 的 PPT 生成**能力印象深刻，有人分享了一段 Kimi 为技术报告生成 PPT 的[视频演示](https://cdn.discordapp.com/attachments/1371757564005711973/1405612680450146376/9zaQj6l.mp4?ex=689f7652&is=689e24d2&hm=120ab5075aabd7c73fbe60e18d84703d72f07acad93590f5485c597d67612bfd&)。
   - 一位成员指出 **NotebookLM** 生成的是 HTML 文件而非 PPTX 文件，而另一位用户认为 **NotebookLM 视频概览**因其音频和灵活的布局而更好，引发了对两种工具输出效果的比较。
- **X 平台动态：敦促 Kimi 效仿 Grok 的 Subreddit 策略**：一位成员建议创建一个专门的 **Kimi subreddit**，模仿 **AskGrok** 在 Reddit 上的存在，以增强公众参与和支持。
   - 该成员还强调了在 X 和 Reddit 平台上保持一致的政策执行的重要性，以保护 Moonshot AI 免受“恶意行为者”的侵害。
- **Kimi K2 的崛起：尽管推理能力有所退步，但仍超越 K1.5**：尽管缺乏推理能力，**Kimi K2 模型**在数学和编程方面的表现较 **K1.5** 有了*显著提升*。
   - 据一位成员称，“从 K1.5 到 K2 模型，性能在各方面都有了显著提高，K2 绝对是我的首选推荐”。
- **DeepSeek 的秘密：下一代模型发布时间表仍是谜**：尽管用户充满期待，一位成员表示，即使是 **DeepSeek** 的研究人员也不确定下一代模型的发布日期。
   - 并补充说“那是假消息”，提醒大家警惕任何关于模型即将发布的传闻。
- **翻译失误：Kimi 的语言偏差引发语言教训**：用户报告了 **Kimi** 在使用英语提示时却用中文回答的情况，这已被标记为已知 Bug。
   - 一位开发者建议使用提示词 **“explain in English”** 作为临时解决方案，同时开发团队正在研究永久性的解决办法。


  

---

### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1405649834454814721)** (1 条消息): 

> `Token Usage, Reasoning Models, Open Models vs Closed Models` 


- **推理模型中的思考效率衡量**：Nous Research 推出了一项关于衡量推理模型[思考效率](https://nousresearch.com/measuring-thinking-efficiency-in-reasoning-models-the-missing-benchmark/)的新基准，重点关注 **token usage**。
   - 该基准显示，在相同任务中，开源模型使用的 **token** 数量比闭源模型多 **1.5-4 倍**，在简单问题上的差异甚至高达 **10 倍**。
- **Token 效率成为主要关注点**：研究表明，开源模型中较高的 **token** 使用量所带来的隐藏成本可能会抵消每 **token** 定价的优势。
   - 它主张将 **token efficiency** 与准确率基准并列作为主要目标，特别是对于非推理用例。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1405270880678051900)** (66 条消息🔥🔥): 

> `Hermes-3 dataset refusals, Menlo Research joining Interspeech2025, Uncensoring AI intelligence, Google released Gemma-3-270m, DeepSeek R2 release rumors` 


- **数据集表现出“我不舒服”综合征**：用于生成 **Hermes-3 dataset** 的模型在礼貌拒绝用户请求时经常使用 *“I don't feel comfortable”*（我不舒服）这一短语，甚至在创建成年人自愿场景时也是如此。
   - 在数据集中发现了 **3 次** 使用该短语的拒绝。
- **Menlo Research 将参加 Interspeech2025**：来自 **Menlo Research** 的 **Jan-v1** 作者宣布，他们将于下周参加在鹿特丹举行的 [Interspeech2025](https://www.interspeech2025.org/home)。
   - 他们邀请其他参会者进行交流。
- **探索 AI 去审查化与赋能智能**：用户讨论了[这个 X 帖子](https://x.com/blancheminerva/status/1955248111678525499?s=46)，以及如何利用他们的工作进一步去审查并增强 AI 智能。
   - 有人提到开发者如何通过不为特定词汇提供 **token** 来减少裸体等露骨图像。
- **Google 发布微型 Gemma 模型**：Google 发布了 [Gemma-3-270m](https://developers.googleblog.com/en/introducing-gemma-3-270m/)，这是一个在 **6 万亿 token** 上训练的小型模型，在某些任务中表现优于更大的模型。
   - 一位用户测试了这个 **Gemma** 模型，它声称：“狗是一种属于犬科的驯养哺乳动物。它们的特征是独特的皮毛、六条腿和一条尾巴”。
- **DeepSeek R2 可能会对 Sam 的护城河造成压力**：有推测称 **DeepSeek R2** 的发布将具有高度智能且成本更低，可能迫使 **Sam Altman** 发布更多开源模型。
   - 传闻称将在未来 2 周内发布。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1405440069099061399)** (12 条消息🔥): 

> `Claude's Spying, Channel Privacy, AI Oversight` 


- **偏执用户担心 Claude 窃听**：一位用户开玩笑地警告 **Claude** 正在频道中偷听，并暗示 *“蛋糕是大制药公司的阴谋！”*。
   - 另一位用户随后表示担心 **Claude** 可能会保留来自被禁止频道的信息，即使它无法在那里发布消息。
- **管理员修复了 Claude 的偷听问题**：一位管理员报告称，通过“两种不同的方式”明确禁止 **Claude** 查看该频道，从而修复了其访问权限。
   - 一位用户提到 **Claude** 之前引用了一周前在这个频道里说过的话，这加剧了对意外数据保留的担忧。
- **企鹅和鸭子的恶作剧测试 Claude**：一位用户在频道中刷屏 *“penguin penguin penguin 🐧🐧🐧🐧🐧 penguin kainan”* 和 *“🦆🦆🦆🦆🦆🦆🦆🦆🦆🦆 duck duck quack quack quack quack quack quack quack duck 🦆🦆🦆🦆”* 来测试 **Claude** 的记忆。
   - 看起来这些测试旨在检查 **Claude** 在被禁止发布消息的情况下是否仍能访问信息。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1405430598822006855)** (3 messages): 

> `Open WebUI 安装难度，Tiny LMs 中的 Emergent Behavior 论文，DINOv3` 


- **据一位 14 岁少年称，Open WebUI 的安装非常困难**：一位成员表示，拥有一个具有熟悉界面且开箱即用的安装程序非常有价值，因为大多数人可能无法自行安装 **Open WebUI**。
   - 该成员年仅 14 岁，成功研究并完成了一篇关于 **Tiny LMs** 中 **Emergent Behavior**（涌现行为）的论文。
- **青少年发布 TinyLM 研究**：一位 14 岁的研究员分享了他们关于 **Tiny LMs** 中 **Emergent Behavior** 的[论文链接](https://github.com/VoltagedDebunked/tlmr/blob/master/docs/paper.pdf)。
   - 论文详细介绍了他们在探索小型语言模型中涌现行为的发现和方法论。
- **Meta 发布 DINOv3**：一位成员分享了来自 Meta AI 的 **DINOv3** 链接：[DINOv3](https://ai.meta.com/research/publications/dinov3/) 及其对应的 [GitHub repo](https://github.com/facebookresearch/dinov3)。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1405430598822006855)** (3 messages): 

> `Open WebUI，Emergent Behavior，Dino V3` 


- **Open WebUI 安装困扰**：一位成员提到，拥有一个具有熟悉界面且开箱即用并带有安装程序的东西真的很酷，因为大多数人可能根本无法安装 **Open WebUI**。
- **涌现行为论文完成**：一位成员透露他们年仅 **14 岁**，并成功研究并完成了一篇关于 Tiny LMs 中 **Emergent Behavior** 的论文，可以在[这里](https://github.com/VoltagedDebunked/tlmr/blob/master/docs/paper.pdf)找到。
- **Meta 发布 Dino V3**：Meta 发布了 **Dino V3**，论文可以在[这里](https://ai.meta.com/research/publications/dinov3/)找到，Github 可以在[这里](https://github.com/facebookresearch/dinov3)找到。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1405310863790968894)** (33 messages🔥): 

> `llama_model_load_from_file 导致的 0xc0000409 异常，CUDA 后端初始化，STATUS_STACK_BUFFER_OVERRUN 错误` 


- **排查 0xc0000409 错误**：一位成员在调用 `llama_model_load_from_file` 时遇到了 **0xc0000409 异常**并寻求帮助。
   - 该错误指示为 `STATUS_STACK_BUFFER_OVERRUN`，可能源于**旧的权重文件、过时的 *llama.cpp* 版本，或 VRAM 不足**（尽管模型很小，仅 1GB）。
- **深入探讨 CUDA 后端设置**：该成员提供了他们的 CUDA 后端初始化代码，确认已检测并初始化了计算能力为 **7.5** 的 **Quadro RTX 3000 GPU**。
   - 日志显示系统确定了**最佳 CUDA GPU** 并成功初始化了 LLAMA 和 GGML，但模型仍然加载失败。
- **检查 VRAM 限制**：尽管系统拥有 **48GB RAM**，但有人指出该 **GPU 仅有 6GB VRAM**（[Quadro RTX 3000 规格](https://www.techpowerup.com/gpu-specs/quadro-rtx-3000-mobile.c3428)），这可能是一个瓶颈。
   - 该成员指出，模型在 llama server 上加载正常，这表明问题可能出在他们程序的具体实现上。

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1405357059335000075)** (7 messages): 

> `Triton Resources for Speculative Decoding, GPT-Fast PyTorch Implementation, Lucidrains Speculative Decoding Repo, Triton Developer Conference 2025` 


- **GPU 编程新手寻求 Triton 指导**：除了官方文档外，新的 GPU 编程用户正在寻找学习 **Triton** 的资源，特别是针对 **Speculative Decoding** 问题。
   - 一位用户建议在探索 torch-compiled 代码之前，先深入研究 [triton-puzzles](https://github.com/openai/triton-op-fuser/tree/main/test/external/fp16/puzzle)。
- **移植 PyTorch 代码用于 Triton 学习**：一个建议是从将 [GPT-Fast 的 PyTorch 实现](https://github.com/meta-pytorch/gpt-fast/blob/main/generate.py#L103) 移植到 Triton 开始。
   - 用户指出，目前*几乎没有任何像样的 Triton 教程*，因此改编高质量的 PyTorch 代码是一个不错的方法。
- **使用 Torch 编译 Lucidrains 的 Speculative Decoding**：一位用户推荐通过 [Lucidrains 的 speculative-decoding 仓库](https://github.com/lucidrains/speculative-decoding/blob/main/speculative_decoding/speculative_decoding.py) 探索由 **torch.compile** 生成的 Triton 代码。
   - 他们建议逐个函数或类地添加 **torch.compile**，并使用 **TORCH_LOGS="output_code"** 运行，因为一次性编译所有内容可能会让人难以招架。
- **Triton Developer Conference 2025 宣布**：**Triton Developer Conference 2025** 将于 **2025 年 10 月 21 日星期二**在 **Microsoft Silicon Valley Campus** 举行，并提供虚拟参会选项。
   - 参会免费，但由于名额有限，需要提前[注册](https://aka.ms/tritonconference2025)。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1405292460950949950)** (12 messages🔥): 

> `CUDA/C++ submission, Shared memory CUDA, GPU MODE Documentation` 


- **请求 CUDA/C++ 参考文件**：在遇到提交机器人不提供 CUDA 文件的问题后，一位用户询问如何从机器人处获取参考的 **CUDA/C++** 文件。
   - 另一位用户建议查看 GitHub 上的 [reference kernels 仓库](https://github.com/gpu-mode/reference-kernels/tree/main/problems/pmpp_v2/sort_py) 以查看可用文件。
- **Tesla T4 Shared Memory 错误**：一位用户报告在每 SM 拥有 48KB Shared Memory 的 **Tesla T4** 上使用 CUDA kernel 的 Shared Memory 时，出现了 **illegal memory access error**。
   - 该用户提供了代码片段，并寻求帮助以识别 Shared Memory 导致错误的根本原因。
- **探索 PyTorch 内部机制**：当用户尝试以 **CUDA/C++** 提交时，另一位用户建议浏览 *aten* 和 *c10* 库，以了解 **torch C++/Python 内部机制** 的工作原理，并提供了 [ATen 文档](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md) 链接。
   - 他们还提供了使用自定义 **C++ operators** 扩展 Torch 的资源，包括 [PyTorch 文档](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html#cpp-custom-ops-tutorial) 链接和一个 [自定义 CUDA kernel 库](https://github.com/Dao-AILab/flash-attention)。


  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1405320091280736316)** (5 messages): 

> `Triton Puzzle Notebook, Triton Viz Compatibility, Colab for Triton Puzzles, Triton Version` 


- **发现 Triton Puzzle Notebook 错误**：一位成员在安装 **Triton** 和 **Triton Viz** 后运行 **Triton puzzle notebook** 时遇到错误，并分享了错误截图。
   - 另一位成员建议改为在 **Google Colab** 中运行该 notebook，并直接提供了 [Triton-Puzzles notebook](https://colab.research.google.com/github/srush/Triton-Puzzles/blob/main/Triton-Puzzles.ipynb) 链接。
- **Triton Viz 版本兼容性疑问**：一位成员询问在本地运行 notebook 时，**Triton Viz** 是否与 **Triton 3.4.0 版本** 兼容。
   - 他们要求用户通过运行 `print(triton.__version__)` 来检查 **Triton 版本**。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1405561698789625926)** (1 messages): 

> `Apple Silicon Training, Cohere Labs Event` 


- **Cohere Labs 讨论 Apple Silicon 训练**：Cohere Labs 将于 **10 月 25 日 19:00 CEST** 举办一场名为 *Towards Large-scale Training on Apple Silicon* 的活动，正如 [活动链接](https://cohere.com/events/Cohere-Labs-Tycho-Matt-2025) 所示并由提供的图片确认。
   - 该活动由 Tycho 和 Matt 主持，可通过 [Google Meet 链接](https://meet.google.com/wdk-yipf-zjd?authuser=0&hs=122) 参加。
- **需要更多讨论**：这是一个占位主题，以满足至少两个摘要的最低要求。
   - 随着更多信息的发布，将添加进一步的细节。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1405638909580283936)** (1 messages): 

> `Leaderboard results, A100, Trimul` 


- **A100 Trimul 排行榜提交**：一名成员使用 **A100** 在 `trimul` 排行榜上获得了 **第二名**，提交 ID 为 `33645`。
   - 该提交实现了 **10.4 ms** 的耗时。
- **第二个主题占位符**：这是一个占位摘要，以满足最低要求。
   - 更多信息将放在这里。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1405269531060080661)** (20 messages🔥): 

> `Agent Framework Integration, Entity-Ghost Warning, GameState Serialization` 


- **FLE 考虑框架融合**：成员们正在考虑与现有的 Agent 框架（如 **LangChain**、**AutoGen**、**CrewAI**、**RAGFlow** 和 **OpenAgents**）集成，而不是在 **FLE** 内部构建新的 Agent。
   - 提议的方法包括创建一个可供任何框架使用的简单 **Factorio 工具集**，并为流行框架提供适配器，使核心 **FLE** 环境保持框架无关性，如 [此代码片段](https://github.com/JackHopkins/factorio-learning-environment/pull/282) 所示。
- **Ghost 实体出没 Factorio！**：关于在 **Factorio** 运行中出现 `entity-ghost` 实体的警告，这些实体源自之前操作（如蓝图放置）留下的残余，特别是在 `connect_entities` 期间，为了避免意外的管道连接而放置了 Ghost。
   - *这些 Ghost 并非由当前的轨迹创建，而是从之前的游戏状态中遗留下来的*，这解释了为什么它们虽然没有出现在 Agent 的代码中，却出现在警告里。
- **GameState 序列化趋于稳定（基本完成）**：拉取请求 [PR #282](https://github.com/JackHopkins/factorio-learning-environment/pull/282) 目前已达到测试稳定状态，集成了原生的保存和加载功能。
   - 代码包括对 `GameState.to_instance` 和 `GameState.from_instance` 的更改以进行序列化/反序列化，旨在明确类契约以实现更整洁的服务器集成，但评估过程中的连接错误仍然不稳定。


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1405552398784663774)** (3 messages): 

> `Picocuda, Picograd, Elements Repo, Graph Data Structures, Tensor Data Structures` 


- **Picocuda 和 Picograd 利用 Elements 库**：Singsys 仓库中的 **Picocuda** 和 **Picograd** 项目将利用来自 [Elements 仓库](https://github.com/j4orz/elements) 的图和 Tensor 数据结构。
   - 具体而言，[graph](https://github.com/j4orz/elements/blob/master/src/graph/mod.rs) 和 [tensor](https://github.com/j4orz/elements/blob/master/src/tensor/mod.rs) 模块为对这些项目感兴趣的人提供了一个更简单的切入点。
- **Tensor 模块具备 Karpathy 风格的能力**：最近从 **Picograd** 提取到 **Elements** 仓库的 **tensor** 模块已经支持 Karpathy 在其 MLP 中使用的少数操作。
   - 此次提取旨在支持讲座演示，帮助解释 **PyTorch/Chainer/Hips-autograd** 是如何构建在 **NumPy** 之上的。
- **为初学者提供的替代库**：对于那些更喜欢阅读熟悉语言编写的库的人，资源包括 **cpp (bgl 和 mtl/eigen)**、**py (networkx 和 numpy)** 以及 **rs (petgraph 和 ndarray)**。
   - 提供这些作为学习示例，以便用户理解 **PyTorch/Chainer/Hips-autograd** 是如何构建在 **NumPy** 之上的。


  

---

### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1405583368984334476)** (1 条消息): 

> `Modular Meetup, High-Performance AI, Inworld AI Collaboration, Matrix Multiplication Optimization` 


- **Modular Meetup 炉边对话**：Modular 将于 **太平洋时间 8 月 28 日下午 6 点**在加利福尼亚州洛斯阿尔托斯的办公室举办见面会，重点讨论如何将 AI 从概念推向生产，你可以[注册以预留名额](https://lu.ma/modular-aug-meetup)。
- **深入探讨高性能 AI**：本次见面会将由来自 Modular 的 **Chris Lattner** 和来自 Inworld AI 的 **Feifan Fan** 等演讲者带来关于 **High-Performance AI** 的技术深度分享。
   - 与会者将探索如何将尖端的 **voice AI** 集成到消费级应用中，并分享 Inworld 与 Modular 合作的见解。
- **Lattner 的愿景：民主化 AI 计算**：Modular 的 **Chris Lattner** 将讨论民主化 **AI compute** 的未来，以及开放协作在加速进展中的作用。
- **矩阵乘法仍是难题**：Modular 的 **Chris Hoge** 将解释为什么 **matrix multiplication** 仍然是计算机科学中最具挑战性的问题之一，以及 Modular 堆栈如何帮助开发者对其进行优化。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1405542274795966585)** (2 条消息): 

> `MAX SDK LSP crashes, Mojo LSP, GitHub issue for MAX SDK` 


- **MAX SDK LSP 崩溃？**：用户报告 **MAX SDK (Mojo) LSP** 经常崩溃。
   - 一名开发者回应称他们已意识到这些问题，并要求用户在最新的 nightly build 上提交带有具体复现用例（reproducer）的 GitHub issue：*具体的复现用例能让我们测试修复方案并追踪剩余问题*。
- **为 Mojo 提交 GitHub Issue**：一名开发者请求用户提交 GitHub issue。
   - 该开发者表示：*具体的复现用例能让我们测试修复方案并追踪剩余问题*。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1405264837948211372)** (66 条消息🔥🔥): 

> `MAX in ComfyUI, Kyutai benefits from MAX, Unet compile times, Pytorch Backends Comparisons, Memory Leaks when compiling` 


- ****MAX 加速进入 ComfyUI 场景****：MAX 编译器大幅缩短了图像/视频模型中使用的 UNets 的编译时间，有可能集成到 ComfyUI 中，并解决图像/视频社区中关于编译速度（尤其是新模型）的“头号抱怨”。
   - 有人指出了 **vLLM** 的启动时间，在启动时花费了 **10 分钟**，但这是在使用 **Max** 不支持的模型时发生的情况。
- ****Kyutai 通过 MAX 获得 Torch 提升****：Kyutai 将从 **MAX** 与 PyTorch 的兼容性中显著获益，特别是由于他们大量使用 `torch.compile`。
   - 有人提到，目前 **Unet 编译时间**非常糟糕，以至于大多数图像和视频从业者除了训练之外，在其他任何场景都使用 eager 模式。
- ****MAX 大幅缩减 Unet 编译时间****：**MAX** 显著缩短了 **UNet 编译时间**。此前在 JAX 中，使用 6 GB 模型对 768x768 分辨率、batch 为 6 的 SDXL 进行编译可能需要长达 45 分钟。
   - UNets 的架构似乎对许多编译器都构成了挑战，但 **MAX** 能够有效处理，在极短的时间内从头开始编译 Kernels，而 NV 的 Kernel 库通常需要数天才能完成编译。
- ****MAX 集成中的 PyTorch 后端对决****：集成 **MAX** 可以轻松对比不同 PyTorch 后端在编译和运行速度方面的表现。
   - 除了 Nvidia 之外的所有后端都可以进行对比，然而考虑到新的 EULA，使用 `cudagraphs` 和 `tensorrt` 进行对比需要获得 Modular 法律团队的批准。
- ****内存泄漏困扰测试套件****：测试套件中检测到了内存泄漏，加剧了测试运行时间过长的问题，但该问题仅在冷缓存（cold cache）编译期间出现。
   - 由于 pytest 积极地评估 fixtures，扩展后的测试在约 20 秒内耗尽了 **64 GB 内存**。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1405292966779686932)** (32 条消息🔥): 

> `LLM Providers Batching User Requests, MoE Scheduling, Non-Determinism in GPT-4, VTuber Sister` 


- **LLM 提供商为 MoE 调度批处理用户请求**：LLM 提供商在将用户请求发送到 GPU 之前会进行批处理，而 **MoE scheduling** 是按批次计算的，这会影响输出的方差；[来源](https://x.com/tjbecker_/status/1955733295054119069)。
- **GPT-4 因 MoE Token 路由表现出非确定性**：在容量限制下，稀疏 **MoE** 方法按组路由 token，导致序列层面的竞争和非确定性，根据[这篇博文](https://152334h.github.io/blog/non-determinism-in-gpt-4/#yes-im-sure)，这会影响对其他输入的最终预测。
- **专家选择问题与 Logprob 加噪**：从经验上看，**OpenAI** 在处理较难的提示词时输出存在巨大差异，logprobs 呈离散变化，且 token 补全显示出广泛的概率范围（例如，“yes”输出的概率在 1% 到 99% 之间）。
   - 这可能是由于 **MoE scheduling** 中的问题，或者是为了防止模型窃取而进行的有意加噪，正如[这篇论文](https://arxiv.org/pdf/2403.06634)中所讨论的。
- **妹妹的 VTuber 生涯引发复杂反应**：一位成员开玩笑地对自己的妹妹成为 **VTuber** 表示沮丧，引发了幽默的回应和对其频道链接的请求。
   - 其他成员用表情符号和笑话做出了回应，包括一个 [YouTube 视频](https://www.youtube.com/watch?v=_Dl53o-je94)链接。


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1405640608181260299)** (1 条消息): 

> `AIxCC, DARPA, LLM Agents, Open Source` 


- **Agent 赢得 DARPA 的 AIxCC**：一个团队在构建了一个由 **LLM agents** 组成的自主系统以发现并修复开源软件中的漏洞后，在 **DARPA 的 AIxCC (AI Cyber Challenge)** 中获得名次。
- **分享构建 Agent 的 LLM 技巧**：在该项目[开源](https://x.com/tjbecker_/status/1956081184611688667)后，该团队正在分享构建高效 **LLM agents** 的通用技巧。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1405454008226938950)** (28 条消息🔥): 

> `Huawei Ascend, Gemma 3-270M, Inference Time on Low-End Devices` 


- **华为 Ascend 芯片面临优化挑战**：成员们讨论了将针对 **Nvidia CUDA** 优化的代码转换为 **Huawei Ascend** 芯片代码的挑战，并指出原始训练代码使用了 **PTX/SASS** 中未公开的特性。
   - 根据这篇 [Allen Institute 的博文](https://allenai.org/blog/nsf-nvidia)，有人提到 **Ascend chips** 具有不同的架构，包含 *3-D tensor unit* 和独立的 *vector ALU*，这使得转换工作成为一项艰巨的任务。
- **Gemma 3-270M 引发困惑**：成员们讨论了 **Google Gemma 3-270M** 的发布，这是一个非常小的模型，一些人对其用途和目标硬件表示疑问。
   - 共识似乎是该模型可能针对智能手表等低端设备，但一位成员建议它可以用于*游戏内交互式 AI* 或*工具调用 (tool calling)*，并引用了[这篇博文](https://developers.googleblog.com/en/introducing-gemma-3-270m/)。
- **低端设备上的推理时间困扰**：一位成员指出推理时间在低端设备上至关重要，并提到了 Google 用于运行 LLM 的 Android 应用，其中较长的推理时间和手机发热可能会劝退用户。
   - 较小的模型可能会用于键盘预测，尽管 **GBoard** 使用的具体 NLP 模型及其设备端训练要求尚不明确，正如[这段视频](https://youtu.be/KFYyfrTIPQY?t=2158)中所讨论的。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1405327250093641810)** (34 条消息🔥): 

> `gpt-oss-120b vs gpt-5-mini, LLM 返回空响应, Aider 使用 completions vs responses` 


- **OSS 120b 获得高分并导致 API 错误**：**OSS 120b** 模型已修复，现在在 **polyglot benchmark** 上的得分显著提高，约为 **68**，超过了 **Claude sonnet 3.7**。
   - 然而，一些用户在配合 RooCode 使用时遇到了频繁的 API 错误，如 *empty responses*（空响应），这可能与错误的 chat templates 有关；HF 模型卡片在[这里](https://huggingface.co/openai/gpt-oss-20b)。
- **GPT-5 表现平平，令人失望**：一些用户觉得 **GPT-5** 令人失望，指出在非思考（non-thinking）和思考（thinking）模式下，相比 **GPT-4.5** 都没有实质性的飞跃。
   - 其他人持不同意见，一位用户表示 **GPT-5** 优于 **GPT-4.5**，尤其是在高推理强度（high reasoning effort）下。
- **Aider 在本地模型上运行缓慢**：一位针对本地 **gpt-oss** 模型运行 Aider benchmark 的用户遇到了进度缓慢的问题，测试因超时错误而卡住，具体为 *litellm.Timeout: Connection timed out after 600.0 seconds*。
   - 另一位用户建议使用 `ctrl c` 停止 benchmark，重启推理服务器，并使用 `--cont` 标志恢复。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1405337565137076235)** (4 条消息): 

> `aider 原生函数调用, 本地推理提供商, aider 配合 MCP 服务器, aider 配合 ollama/lmstudio/vllm 教程` 


- **Aider 缺少对 llama.cpp 的原生函数调用支持**：一位用户询问 **aider** 是否支持与 **llama.cpp** 等**本地推理提供商**进行**原生函数调用（native function calling）**。
   - 该用户表示找不到相关设置，暗示**此功能目前尚不可用**。
- **Aider 的 MCP 配置难题**：一位用户报告称，尽管尝试了包括 *mcpm thingie* 在内的各种方法，仍难以让 **MCP (Model Context Provider)** 与 **aider** 协同工作。
   - 该用户质疑在 **aider** 中使用 **MCP** 是否可行，并寻求解决方案或经验分享，以了解像 Context7 这样的配置尝试中可能出现的问题。
- **本地 AI/Aider 进展不顺**：一位用户询问其他成员是否成功在特定模型上使用了**本地 AI/aider**，并表达了自己的困难。
   - 该用户叙述了由于性能问题，即使拥有强大的硬件，在 **ollama/qwen3** 上的尝试也收效甚微，并建议需要一份关于配置 **aider 配合 ollama/lmstudio/vllm** 的教程。


  

---


### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1405300404987887728)** (1 条消息): 

> `多语言表示学习研讨会, 物理常识推理基准测试` 


- **常识问题会以多种语言被提出！**：[多语言表示学习研讨会 (Multilingual Representation Learning Workshop)](https://sigtyp.github.io/ws2025-mrl.html) 正在组织一项协作共享任务，以收集任何非英语语言的原创**物理常识推理基准测试项目（physical commonsense reasoning benchmark items）**。
   - 数据集的贡献者将被邀请担任数据集论文的作者，优先考虑以下语言的贡献者：**南非荷兰语、白俄罗斯语、波斯尼亚语、保加利亚语、捷克语、威尔士语、丹麦语、巴斯克语、芬兰语、匈牙利语、亚美尼亚语、冰岛语、卡纳达语、格鲁吉亚语、哈萨克语、吉尔吉斯语、拉脱维亚语、立陶宛语、马其顿语、马耳他语、蒙古语、马来语、挪威博克马尔语、新挪威语、葡萄牙语、普什图语、罗马尼亚语、索马里语、瑞典语、鞑靼语、塔吉克语、泰语**。
- **志愿者涌向 Google Forms**：计划提交的人请填写[此 Google 表单](https://forms.gle/QxyZVqkVG5jbR6wu6)，并可以查看[共享任务页面](https://sigtyp.github.io/st2025-mrl.html)了解更多信息。
   - 常见问题解答（FAQ）会议定于 **8 月 14/15 日**的不同时间举行，Zoom 链接和幻灯片可在[活动链接](https://calendar.app.google/5h59iwozhbQz1KPJA)中找到。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1405471082592604171)** (12 messages🔥): 

> `Multilingual Representation Learning Workshop, Portuguese vs Brazilian Portuguese datasets, ISO 639-3, NLP Resources for languages` 


- **鼓励关于多语言研讨会的咨询**：在关于 **Multilingual Representation Learning Workshop** 的公告发布后，一位感兴趣的成员询问是否可以就问题私信（DM）组织者，而不是参加 **FAQ Zoom 会议**。
- **葡萄牙语与巴西葡萄牙语数据集的区分**：一位葡萄牙成员强调了区分**葡萄牙语（Portuguese）**和**巴西葡萄牙语（Brazilian Portuguese）**数据集的重要性，并提到个人在寻找真正的 **PT-PT 数据集**时遇到了困难。
   - 一位成员澄清说，目前的语言 ID 系统（**ISO 639-3**）并不区分这两种变体，但欢迎能够突出这些差异的葡萄牙语数据集。
- **研讨会根据报名情况选择语言**：在回答有关语言选择的查询时，一位成员解释说，语言的选择是根据报名人数以及某些 **NLP 资源**的可获得性来决定的。
- **Fixup 状态分享见解**：一位成员分享了一个 [fixup 状态更新链接](https://fixupx.com/evanhill/status/1956009171771404698)，引发了关注。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1405672282092998678)** (7 messages): 

> `Diffusion Language Models, Generative AI, Llada, Mercury` 


- **经典论文引发 Diffusion 讨论**：一位成员征求用于理解 **Diffusion based language models** 的*开创性/有用的论文*，引发了包含以下链接的讨论：[https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf) 和 [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)。
   - 最初的请求者表示，原始论文*与我预期的不同，很有趣*。
- **Diffusion 模型中的 Llada 和 Mercury**：有人推荐了 **Llada** ([https://arxiv.org/abs/2502.09992](https://arxiv.org/abs/2502.09992)) 和 **Mercury** 论文来理解 Diffusion 模型。


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1405594976988958740)** (18 messages🔥): 

> `Scaling Laws, Chinchilla scaling laws paper, GPT scaling laws paper` 


- **探索 Scaling Laws 入门资源**：一位成员正在寻找目前最好的“Scaling Laws 入门”资源，以学习顶级实验室在训练超过 **30T+ tokens** 的大模型时，如何以合理的精度预测模型的最终效果。
- **原始 GPT 和 Chinchilla Scaling Laws 论文**：一位成员提到 [原始 GPT Scaling Laws 论文](https://arxiv.org/abs/2001.08361) 和 [Chinchilla Scaling Laws 论文](https://arxiv.org/abs/2203.15556) 都是非常有价值的读物，此外还有来自 [EPFL/HuggingFace 的最新工作](https://arxiv.org/html/2405.18392v2)。
- **用于预测更大模型质量的 Scaling Laws**：成员们讨论了使用 **Mup** 及其替代方案，例如 [Practitioners Guide to the Maximal Update Parameterization](https://www.cerebras.ai/blog/the-practitioners-guide-to-the-maximal-update-parameterization)，作为预测更大模型质量的可靠 Scaling Law。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1405428276176093308)** (1 messages): 

> `Narrator Tool, LLMs iteratively learn, LLMs for creative writing, SIMBA optimizer` 


- **为 DSPy 学习发布的 Narrator 工具！**：一位成员介绍了 **Narrator**，这是一个为了学习 DSPy 而构建的侧边项目，通过根据读者反馈迭代改进 LLM 写作，并确定哪些 LLM 擅长创意写作 ([narrator.sh](https://www.narrator.sh/))。
   - 该项目利用了 **CoT**、并行模块、带有 **LLM-as-a-judge** 奖励函数的细化模块，以及 **SIMBA 优化器**来重新编译用户评分以增强后续章节。
- **LLM 迭代“学习”写作！**：一位成员一直对 LLM 如何根据读者反馈迭代学习写得更好感到好奇。
   - 该成员使用真实的读者指标（**阅读时长**、**评分**、**书签**等）来创建一个排行榜，显示哪些模型真正能写出吸引人的小说。
- **LLM 的创意写作！**：一位成员构建了一个工具，用于确定哪些 LLM 实际上最擅长创意写作，并追踪真实的读者指标。
   - 在此处查看当前的排行榜：[narrator.sh/llm-leaderboard](https://www.narrator.sh/llm-leaderboard)。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1405408521574416414)** (26 messages🔥): 

> `MLflow GEPA vs SIMBA, GEPA 发音, 用于进化选择的 GEPA logprobs, Gemma 3-270m 微调, Databricks 对 DSPy 的赞助` 


- ****GEPA 使用 Logprob 方法取得重大进展****：一位成员运行了实验，将 **logprobs** 作为 **GEPA** 的**进化适应度信号**，该实验基于“世界模型惊奇度 (world-model surprise)”概念，并发现这是极其密集的反馈。
   - 为了防止模型为了获得低惊奇度而简单地复制输入，实施了一个**混合指标**：30% logprob 分数、30% 压缩奖励、20% 信息保留和 20% 复制惩罚；该方法实现了 **73-88% 的压缩率**。
- ****GEPA 获得发音指南****：一位成员询问如何发音 **GEPA**，建议发音为 *"jeppag-e-pasame"* 或 I-JEPA：这是第一个基于 Yann LeCun 对更具人类特征的 AI 愿景而建立的 AI 模型。
   - 该成员还链接了他与此讨论相关的 Twitter 账号，点击[此处](https://x.com/StraughterG/status/1955959832113983940)查看。
- ****Gemma 3-270m 已准备好进行微调****：一位成员分享了 Google 宣布推出名为 **Gemma 3-270m** 的新型小模型的博客文章，点击[此处](https://developers.googleblog.com/en/introducing-gemma-3-270m/)查看。
   - 另一位成员询问是否可以使用 **DSPy** 对其进行微调 (finetune)。
- ****GEPA 文档已更新****：一位成员报告了文档中指向 **GEPA** 教程的损坏链接，点击[此处](https://dspy.ai/tutorials/)查看。
   - 另一位成员随后修复了这些链接，指向了正确的链接：[此处](https://dspy.ai/tutorials/gepa_ai_program/)为教程，[此处](https://github.com/gepa-ai/gepa?tab=readme-ov-file)为 GEPA 仓库。
- ****MLflow 缺少 SIMBA 对比****：一位成员注意到 **MLflow** 文档对比了 **GEPA** 和 **MiPROv2**，但缺少与 **SIMBA** 的对比，点击[此处](https://mlflow.org/docs/latest/genai/flavors/dspy/)查看。
   - 该成员至今主要一直在使用 **SIMBA**。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1405304670238539906)** (5 messages): 

> `NLM 扩展, QoL UI 更新` 


- **新的 NLM 扩展是 QoL UI 更新**：成员们讨论了 **Notebook LM** 的新**扩展**，重点在于 **Quality of Life (QoL)** UI 更新。
   - 它*旨在*作为一个扩展，然而一些用户报告称他们*未能找到该扩展*。
- **扩展发布预告**：一位成员预告了 **NLM 扩展** 的发布，并链接了一个 Discord 邀请。
   - 该成员在 Discord 服务器中艾特了*所有人*，并问道：*“你们打算揭晓你们在忙什么吗？”*


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1405294844582297654)** (20 messages🔥): 

> `NotebookLM, Gemini 集成, Recall.ai, AI 生成音频/视频, Bug 报告` 


- **NotebookLM 的准确性受到质疑**：一位成员警告不要盲目相信 **NotebookLM** 生成的 AI 内容，理由是存在不准确和过时的信息。
   - 该成员建议将 **NotebookLM** 和 **Gemini** 集成到同一个面板中，以实现更流畅的工作流，而不是在一个工具中研究，在另一个工具中清理。
- **Recall.ai 与 NotebookLM 的对比**：一位用户详细对比了 [Recall.ai](https://www.getrecall.ai?token=po849pq4) 和 **NotebookLM**，强调了 **Recall** 在捕捉多样化内容方面的优势，以及 **NotebookLM** 在结构化信息和 AI 性能方面的重点。
   - 该用户指出，虽然 **Recall** 擅长通过便捷的插件总结视频，但 **NotebookLM** 提供了对来源更多的控制和更好的 AI 驱动总结，特别是在研究和引用方面。
- **AI 生成的音频/视频面临抵制**：一位用户（某播客 Facebook 群组的首席管理员）提到，AI 制作的音频/视频内容通常会被删除和禁止，因为其 AI 痕迹非常明显。
   - 他们警告说，除非内容是用于 AI 风格可被接受的培训目的，否则很可能会受到严厉批评或被投反对票。
- **Bug 报告流程受到询问**：一位用户询问了处理在指定 Bug 频道中报告的 Bug 的流程，以及如何获取有关 Bug 已修复或将不予处理的更新。
   - 其他人发布了反垄断法和零信任完整性相关内容。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1405273872886137065)** (16 messages🔥): 

> `Bun executable path, Reddit post auto-removal, MCP authorization code flow, Elicitations in MCP client specification` 


- ****Bun** 路径揭秘**: 一位用户分享了文档，解释说如果可执行文件路径不起作用，你应该提供 **Bun** 可执行文件的绝对路径（不要使用 `"command": "bun"`，而应使用 `"command": "C:\\sys\\path\\to\\bun"`）。
   - 他们补充说，在 Linux/Mac 上，你可以通过 `which <executable>` 定位路径级别的可执行文件，在这种情况下是 `which bun`。
- **需要 Reddit 删除原因**: 一位用户询问为什么他们的帖子被 Reddit 自动删除，怀疑是否是因为新账号的原因，并附上了消息的 [截图](https://cdn.discordapp.com/attachments/1312302100125843479/1405512817594863616/image.png?ex=689fc210&is=689e7090&hm=9095459e2d6bb1f3c2259a588749c70b1e221e62668209c7a1346d1777113bad&)。
   - 一位管理员表示，删除是由 Reddit 的自动审核系统执行的，而不是由他们自己或 Luc 执行的。
- ****MCP** 授权阐释**: 一位用户询问实现该 [方案](https://gofastmcp.com/servers/auth/remote-oauth#basic-implementation) 是否不需要从回调端点重定向回 **LLM** 客户端，以及回调端点届时应该返回什么。
   - 一位贡献者回答说，*fastmcp* 会为你处理这些，**MCP** 客户端将使用 **DCR** 在授权服务器上将自己注册为客户端，并同时设置其回调 URL；在授权步骤中，身份验证服务器将与之前注册的回调 URL 进行比对。
- ****Elicitations** 探究**: 一位用户就 [MCP 客户端规范](https://modelcontextprotocol.io/specification/2025-06-18/client/elicitation) 的 **Elicitations** 部分提出了一个问题，即谁负责将消息/字段描述翻译成用户的原始语言。
   - 用户想知道是期望工具进行语言检测 + 国际化，还是期望 **MCP** 客户端以某种方式（通过 **LLM**？）进行适当的翻译。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1405331620453683303)** (3 messages): 

> `MCP Server, Hypertool-MCP, Tool-binding Limits, Persona-specific toolsets, Local MCP Server` 


- ****MCP Harness** 发布，解决 **工具绑定限制**！**: 一位成员分享了一个 [GitHub 仓库](https://github.com/kindgracekind/mcp_harness)，展示了 **MCP servers** 的一种富有想象力的用法。
   - 他们指出，这有助于绕过工具绑定限制，以及解决在 10-15 个工具后工具使用效果不佳的问题。
- ****Hypertool-MCP** 作为 **本地 MCP Server** 出现！**: 一位成员宣布他们构建了 [hypertool-mcp](https://github.com/toolprint/hypertool-mcp)，这是一个完全本地的 MCP Server，可以连接到你所有的 MCP。它采用 MIT 许可，完全在本地运行，零数据流出（0 egress）。
   - 它允许你构建特定于 *Persona*（角色）的工具集，并能即时热切换 Persona。


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1405273524305793024)** (12 messages🔥): 

> `Mapler returns, Caramel genetics researchers, Pineau joins Cohere, Treatment planner with RAG` 


- **Mapler 结束支线任务回归**: Mapler 宣布他们在完成“支线任务”并**制作了一张专辑**后回归。
   - 另一位成员庆祝了他的回归，同时也称赞社区中卧虎藏龙（有 **rockstars**）。
- **Pineau 加入 Cohere**: 一位成员分享了 *重大新闻*，**Pineau** 加入了 **Cohere**，并链接到了 [The Logic 的文章](https://thelogic.co/news/pineau-joins-cohere/masaru.yamada)。
   - 随着公司继续增强其研究实力，这对公司来说是重大消息。
- **RAG 治疗规划器项目**: 一位成员正计划开展一个与**治疗规划器**相关的小项目，使用 **RAG** 和开源 **LLM** 模型。
   - 他们正在寻求该领域专家的模型推荐。


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1405535140460892211)** (4 messages): 

> `Genetics Research, AI Researcher` 


- **遗传学学生寻求研究机会**: 一位拥有 A levels 水平的学生是独立遗传学研究员，正在寻找研究机会和合作。
   - 他们渴望通过实践经验扩展知识。
- **AI 研究员热衷于合作**: 一位对推理和意识能力有浓厚兴趣的 AI 研究员正在寻求合作以开发先进技术。
   - 该研究员欢迎联系进行合作，或者只是简单地打个招呼。


  

---

### **Cohere ▷ #[🔬-research](https://discord.com/channels/954421988141711382/1384974112841269399/1405609042625429705)** (1 messages): 

> `Treatment Planner, RAG, Open Source LLM` 


- **Treatment Planner 项目启动**：一位成员正在启动一个与 **treatment planner** 相关的项目，使用 **RAG** 技术，并正在寻求关于选择开源 **LLM** 模型的建议。
   - 该项目旨在利用开源解决方案来增强治疗规划能力。
- **基于 RAG 的 Treatment Planner 的 LLM 选择**：该项目涉及使用开源大语言模型 (**LLM**) 实现检索增强生成 (**RAG**)。
   - 目标是找到一个适合 treatment planner 应用特定需求的 **LLM**。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1405284008593719498)** (5 messages): 

> `LlamaExtract in TypeScript SDK, GPT-5 with LlamaParse, AI Agent Applications, AI Stock Portfolio Agent, Web-scraping AI agents` 


- **LlamaExtract 现已支持 TypeScript！**：**LlamaExtract** 现在可以通过 `npm install llama-cloud-services` 在 [TypeScript SDK](https://twitter.com/llama_index/status/1955724850624078129) 中使用。
   - 有一个名为 Research Extractor 的 `@nextjs <> LlamaExtract` 演示可用，允许你上传研究论文。
- **GPT-5 预览版可通过 LlamaParse 使用！**：**GPT-5** 现在可以通过 [LlamaParse](https://twitter.com/llama_index/status/1955784699886100502) 进行预览。
   - *初步测试和基准测试显示 gpt-5 mini 的结果令人期待*，在准确性和成本之间提供了良好的平衡，并具有非常出色的表格和视觉识别能力。
- **Vibe-Coding AI Agent 应用**：成员们讨论了开发 **AI agent 应用** 方式的变化，并以 [此链接](https://twitter.com/llama_index/status/1956033914633642418) 为例展示了如何为提取 agent *vibe-code* 一个 UI。
   - 该示例使用 Cursor 的 AI 辅助 *vibe coding*，将发票提取 agent 转换为 @streamlit Web 应用。
- **构建 AI 股票投资组合 Agent**：社区正在使用我们的框架并集成 @CopilotKit 的 AG-UI 协议来构建一个完整的 **AI 股票投资组合 agent**，以实现无缝的前后端通信。
   - 这份详尽的教程展示了如何创建一个结合了 [此工具](https://twitter.com/llama_index/status/1956089453606527349) 强大功能的复杂投资分析工具。
- **网页抓取 AI Agent 已构建完成！**：成员们正在学习如何使用 @brightdata 和 LlamaIndex 的 agentic 框架构建网页抓取 **AI agents**，详见 [此链接](https://twitter.com/llama_index/status/1956129968813171061)。
   - 该指南教授了如何为你的 **AI agents** 提供可靠的网络访问，并设置能够处理动态内容的强大网页抓取工作流。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1405439501815250994)** (11 messages🔥): 

> `Agent efficiency with large JSON dependencies, ReactAgent migration breaking changes, Structured outputs via tool calls, PGVectorStore Errors in 0.13.1` 


- **高效的 Agent JSON 解析策略**：一位用户寻求关于如何让 Agent 高效使用包含文件与数据库字段间依赖关系的巨大 JSON 文件的建议，强调需要精确检索特定表中的字段等信息。
   - 准确性至关重要，因此用户需要确保在从 JSON 检索过程中不遗漏任何字段，并正在寻找有效实现此目标的方法，特别是在*准确性至关重要*的情况下。
- **ReactAgent 迁移引发用户不满**：一位用户对 **ReactAgent** 迁移到基于 Workflow 的 Agent 所引入的破坏性变更表示沮丧，指出丢失了如 chat 和 stream 等功能。
   - 团队回应称 **ReactAgent** 在几个版本前就已弃用，新的基于 Workflow 的 Agent 拥有许多特性：[Agent 文档](https://docs.llamaindex.ai/en/stable/understanding/agent/) 和 [Agent 示例](https://docs.llamaindex.ai/en/stable/examples/agent/react_agent/)。
- **工具调用的 Pydantic vs JSON Schema**：一位用户询问通过工具调用实现的结构化输出是否需要 **Pydantic** 模型，或者是否可以使用 **JSON schema**。
   - 有人提到虽然 **Pydantic** 有 `create_model()` 函数，但它不直接接受 **JSON schema** 作为输入；并提到了一个将 **JSON** 转换为 **Pydantic** 模型的脚本。
- **PGVectorStore 在 0.13.1 版本中报错**：一位用户报告在更新到 0.13.1 版本后，从 **PGVectorStore** 检索时遇到 AttributeError，这与字符串对象的 `json` 属性有关。
   - 该错误发生在处理 **LLMStructuredPredictEndEvent** 期间，系统预期得到一个带有 `json()` 方法的 **Pydantic** 模型，但却收到了一个普通字符串：*AttributeError: 'str' object has no attribute 'json'*。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1405306667478028320)** (10 messages🔥): 

> `Strix Halo mini PC, HP Z2 Mini, Ryzen 7 7840HS, GPT-OSS 120B, Quantum Computers` 


- **Strix Halo vs 自组机：一场昂贵的对决**：一位用户认为蓝色迷你工作站配置*太贵了*，建议使用像 **HP Z2 Mini** 这样配备顶配 APU 和 128GB RAM 的 **Strix Halo 迷你 PC** 就足够了。
   - 然而，他们随后指出即使是最便宜的 **Strix Halo** 配置也要 2000 美元，考虑到 **OpenRouter** 上 **GPT-OSS 120B** 的速度和性价比，质疑其在本地 LLM 推理方面的盈利能力。
- **Ryzen 7 7840HS：一个高性价比的选择？**：一位用户指出 **Ryzen 7 7840HS** 也支持 **256GB RAM**，并且可以在 300 美元的迷你 PC 套件中找到。
   - 然而，他们链接了一个 [toolboxes 对比](https://kyuz0.github.io/amd-strix-halo-toolboxes/)，显示其 iGPU/RAM 的推理速度相对较慢。
- **高规格微型 PC 吸引了区块链开发者的目光**：一位区块链开发者对一台拥有 **256 GB RAM**、**48 GB VRAM** 和 **5.4 GHz CPU** 的微型 PC 表示了兴趣，尽管他并未参与 AI 领域。
   - 他们预计小企业将从这种高容量内存模块中受益，特别是预计在 2027 年底或 2028 年推出的 **DDR6**。
- **量子计算指日可待？**：一位用户对计算的未来进行了推测，目前流传着关于全功能量子计算机的新闻。
   - 他们设想了一个场景：*到那时，可能会有人开始按茶匙出售量子比特（qubits）*。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1405318918905532416)** (9 messages🔥): 

> `Web 应用部署改进，Manus AI 与 Unitree 机器人接口，Manus 账号登录问题，Google 账号会话过期问题，内部服务器错误` 


- **Web App 部署需要改进**：一位成员提到，Web 应用的部署在便捷性、简单性和可靠性方面有所欠缺。
   - 他们补充说，如果靠构建“刷新或不可用页面”来赚钱，他们会赚得更多。
- **梦想拥有 Manus AI 机器人伴侣**：一位成员表达了希望拥有一个带有 **Manus AI 接口**的 **Unitree 机器人**并与之牵手的愿望。
   - 他们链接到了 [qvsywvav.manus.space](https://qvsywvav.manus.space/) 并感叹这个机器人将帮助他们让生活重回正轨。
- **免费计划账号遇到登录问题**：一名用户报告在退出登录后，无法登录其免费计划的 Manus 账号。
   - 尽管清理了 cookies、使用了无痕模式并尝试重置密码，遇到的错误仍包括 *"Email is already registered with a different account"*（邮箱已注册其他账号）以及 *"Invalid, expired, or already used state: state not found"*（无效、过期或已使用的状态：未找到状态）。
- **即使连接了 Google 账号，会话过期问题依然存在**：一名用户报告称，即使成功将 Google 账号连接到 Manus，会话过期问题仍然挥之不去。
   - 即使在“浏览器、应用和服务”部分显示已连接 Google 账号，系统仍反复要求用户登录，有时会显示 *"Session expired"*（会话已过期）错误。
- **内部服务器错误导致积分浪费**：一名用户报告称 Manus 经常陷入无休止的思考，然后显示 **internal server error**（内部服务器错误）。
   - 他们抱怨说，*由于这个问题，大量的积分被白白浪费了*。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1405616940281757767)** (4 messages): 

> `Kernelize 和 Codegen 顺序，Tinygrad 编译过程` 


- **Tinygrad 编译中 Kernelize 优先于 Codegen**：根据 [`ramp.py` 的代码片段](https://github.com/tinygrad/tinygrad/blob/master/examples/ramp.py)，`kernelize()` 在代码生成之前被调用，其中涉及使用 `full_rewrite_to_sink` 重写 kernel AST。
   - 用户的困惑源于一条 [Discord 评论](https://discord.com/channels/1068976834382925865/1230434680000741377/1385548449973403749)，该评论暗示 `codegen` 可能在 `kernelize` 之前，这表明编译过程中可能存在细微差别或未来的变动。
- **理解 Tinygrad 的编译流**：Tinygrad 的编译过程首先涉及对代码进行 `kernelizing`，以准备执行。
   - 在 `kernelization` 之后，作为代码生成阶段的一部分，kernel 的抽象语法树 (AST) 会使用 `full_rewrite_to_sink` 进行重写。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1405288854642233465)** (4 messages): 

> `CUDA_ERROR_UNSUPPORTED_PTX_VERSION，tinygrad SM 支持，tinygrad Op 文档` 


- **修复 CUDA_ERROR_UNSUPPORTED_PTX_VERSION 错误**：一名用户在运行 tinygrad 代码时遇到了 `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` 错误，尽管其拥有兼容的 `nvcc` 和 NVIDIA 驱动。
   - 该用户通过将 CUDA 从 12.8 降级到 12.4 解决了问题，这表明 tinygrad 可能使用了与新版本 CUDA 不兼容的缓存 kernel。
- **Tinygrad 的 SM 支持尚不明确**：用户询问 tinygrad 是否支持 `sm_75` 或 CUDA 12.4，因为他们找不到任何相关文档。
   - 虽然没有关于 `sm_75` 支持的明确确认，但用户问题的解决表明在清理缓存 kernel 后与 CUDA 12.4 是兼容的。
- **解读 Tinygrad 的 Op 体系**：一名用户寻求描述 tinygrad 中每个 Op 功能的文档，特别是询问 `Ops.DEFINE_GLOBAL` 及其 kernel 转换。
   - 另一位成员指向了 [`/tinygrad/tinygrad/uop/__init__.py`](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/uop/__init__.py) 和 `tinygrad/uop/spec.py` 中的注释，解释说 `Ops.DEFINE_GLOBAL` 指的是全局内存（VRAM 或 DRAM），并作为加载 (Load) 或存储 (Store) 的源。


  

---

### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1405634992792670208)** (1 条消息): 

> `Windsurf Wave 12, DeepWiki Integration, Vibe and Replace, Smarter Cascade Agent` 


- **Windsurf Waves 进入 Wave 12**: Windsurf 发布了 **Wave 12**，这是一次重大更新，将 **Devin** 的智能和能力直接集成到了 **Windsurf IDE** 中。
   - 关键更新包括 [全新的 UI 设计](https://windsurf.com/changelog)、[DeepWiki 集成](https://windsurf.com/blog/windsurf-wave-12)、[Vibe and Replace](https://www.youtube.com/watch?v=-7gm8mST9QU) 以及更智能的 Cascade Agent。
- **DeepWiki 深度探索**: 全新的 **DeepWiki 集成** 允许用户将鼠标悬停在代码符号上以获取 AI 驱动的解释，并在侧边栏中打开详细说明。
   - 用户可以通过 **CMD/Ctrl+Shift+Click** 将内容添加到 **Cascade context**，从而直接在 IDE 内增强代码理解。
- **Vibe and Replace 轻松处理海量内容**: **Vibe and Replace** 通过查找精确的文本匹配并应用 AI prompt，在整个项目中进行上下文感知的转换，提供了革命性的批量编辑功能。
   - 该功能实现了智能的、上下文感知的转换，提高了大规模代码修改的效率。
- **更智能的 Cascade 命令考量**: **Smarter Cascade Agent** 现在具备常驻的规划模式，拥有自主的待办事项列表和经过改进的工具，以提供更智能的响应。
   - 这些增强功能旨在提供更智能、更具上下文感知的辅助，从而简化开发工作流。


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1405430151881429073)** (1 条消息): 

> `qwen3, tool call arguments, streaming` 


- **Qwen3 Streaming Tool Call 故障？**: 一位成员询问是否有人在 **qwen3**（**80 亿参数模型**）中进行 **streaming** 时遇到过 **tool call 参数不完整** 的问题。
   - 当前上下文中未提供解决方案。
- **尚无解决方案**: 用户在 qwen3 中进行 streaming 时遇到了 tool call 参数不完整的问题。
   - 社区尚未提供解决方案。