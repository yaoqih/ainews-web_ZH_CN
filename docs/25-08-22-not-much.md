---
companies:
- google-deepmind
- alibaba
- google
- deepseek
- baseten
- yupp
date: '2025-08-22T05:44:39.731046Z'
description: '**DeepMind** 发布了 **Genie 3**，这是一个具备先进空间记忆和实时化身（avatar）控制功能的交互式多模态世界模拟器；同时推出了
  **SIMA**，一个在生成世界中运行的具身训练智能体。**阿里巴巴** 推出了 **Qwen-Image-Edit**，这是一款开源权重的图像编辑器，在图像编辑竞技场（Image
  Editing Arena）中以 **ELO 1098 分位列第二**，并支持在高通 NPU 上运行；此外，**Qwen-VL-Max** 也进入了视觉模型前
  20 名。视频模型方面，**可灵 (Kling) 2.1** 在帧控制上实现了 **235% 的提升**，新晋模型 **Luma Ray 2** 和 **Runway
  Gen-4 Turbo** 也正式亮相。**谷歌** 在 Gemini 应用中提供了免费的 **Veo 3** 生成功能，并增强了 Google 相册的自然语言编辑功能。**DeepSeek
  v3.1** 正式发布，重点关注软件工程（SWE）和搜索智能体，支持在 Apple Silicon 上进行本地推理，通过 4 位量化在 M3 Ultra 上达到了约
  **21 tok/s** 的速度。这些新闻凸显了交互式模拟、视觉编辑、视频合成以及可扩展本地 AI 推理方面的最新进展。'
id: MjAyNS0w
models:
- qwen-image-edit
- qwen-vl-max
- kling-2.1
- veo-3
- deepseek-v3.1
- genie-3
- sima
people:
- demishassabis
- bonniesjli
- shreyar
- ostrisai
- lmarena_ai
- teortaxestex
- ivanfioravanti
title: 今天没发生什么特别的事。
topics:
- multimodality
- embodied-ai
- simulation
- fine-tuning
- quantization
- video-generation
- image-generation
- local-inference
- scaling
- agent-training
- real-time-control
- spatial-memory
---

**平静的一天。**

> 2025年8月21日至8月22日的 AI 新闻。我们为您检查了 12 个 subreddit、544 个 Twitter 账号和 29 个 Discord 社区（包含 229 个频道和 9088 条消息）。预计节省阅读时间（按每分钟 200 词计算）：724 分钟。我们的新网站现已上线，支持全元数据搜索，并以精美的 vibe coded 方式展示所有往期内容。访问 https://news.smol.ai/ 查看完整的详细新闻，并在 @smol_ai 上向我们提供反馈！

最后几段 AI Engineer World's Fair 的视频将于本周末发布，快去[看看吧](https://www.youtube.com/@aiDotEngineer/videos)！

---

# AI Twitter 回顾

**交互式世界模拟器与具身训练 (Genie 3 + SIMA)**

- **DeepMind 的 Genie 3 世界模型（多模态、持久化模拟）**：根据 [@demishassabis 的推文](https://twitter.com/demishassabis/status/1958696882105995312)，Genie 3 是一个交互式世界模拟器，你可以通过文本、照片或视频进行提示，具有**高级空间记忆**（状态在镜头外依然持久存在）和**实时化身控制**（[示例](https://twitter.com/demishassabis/status/1958696898488840414)、[此处](https://twitter.com/demishassabis/status/1958696900489523633)以及[此处](https://twitter.com/demishassabis/status/1958696891639595148)）等功能。DeepMind 还发布了关于 Genie 3 潜力的播客（[链接](https://twitter.com/demishassabis/status/1958696904146976927)）。
- **在生成的模型世界“内部”训练 Agent**：DeepMind 的 SIMA 被展示在 Genie 生成的环境中进行学习——实现了从世界生成到具身学习的完全 AI 闭环（[@bonniesjli](https://twitter.com/bonniesjli/status/1958948293523767561)）。
- **实际应用中的模拟器工具**：开发者们将模拟用于数据生成、评估引导、发布前安全性测试以及轨迹分析（[@ShreyaR](https://twitter.com/ShreyaR/status/1958811497196659207)）。Snowglobe 增加了可共享的只读链接（[链接](https://twitter.com/zaydsimjee/status/1958938033811869735)）；SDK “即将推出”（[链接](https://twitter.com/ShreyaR/status/1958949657792614675)）。

---

**开源视觉与媒体模型：Qwen Image Edit 领先，Qwen-VL 攀升；视频模型激增**

- **Qwen-Image-Edit (Apache-2.0) → 顶级且高性价比的编辑模型**：来自阿里巴巴的新开源图像编辑模型在 Image Editing Arena 中获得了 **ELO 1098（第 2 名）** 的高分，与 GPT-4o 旗鼓相当，但价格仅为其一小部分（[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1958712568731902241)；[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1958725835818770748)）。社区示例展示了强大的局部编辑能力和风格保真度（[建筑演示](https://twitter.com/Alibaba_Qwen/status/1958744976772198825)）。在端侧设备上，**Qwen3 正在汽车/机器人的 Qualcomm NPU 上运行**（[链接](https://twitter.com/Alibaba_Qwen/status/1958800193970954657)）。Qwen-VL-Max 进入了视觉模型前 20 名（[并列第 10](https://twitter.com/lmarena_ai/status/1958957107946168470)）。
    - 工具链：AI Toolkit 现在支持使用 3-bit ARA **微调 Qwen-Image-Edit**；支持在**单张 5090 上使用缓存文本嵌入训练 1024 分辨率的 LoRA**；24GB 显存目标已接近但尚不可靠（[@ostrisai](https://twitter.com/ostrisai/status/1958932936620900666)）。
- **视频：可灵 (Kling) 2.1 “每一帧都在掌控中” + 新入局者**：可灵 2.1 发布了**首尾帧控制**功能，声称比 1.6 版本提升了 **235%**，能够实现精确的中间帧合成（[@Kling_ai](https://twitter.com/Kling_ai/status/1958835762369372269)；[Lovart 集成](https://twitter.com/lovart_ai/status/1958843940209401875)）。Luma 的 **Ray 2** 和 **Runway Gen-4 Turbo** 在 Video Arena 首次亮相（[详情](https://twitter.com/lmarena_ai/status/1958990871028015299)）。Google 本周末在 Gemini App 中提供了 **3 次免费的 Veo 3 生成机会**（[Gemini](https://twitter.com/GeminiApp/status/1959035394483503581)、[Google](https://twitter.com/Google/status/1959037076503937379)），且 **Google Photos** 现在支持自然语言编辑（如“移除汽车”、“让它更好看”）（[链接](https://twitter.com/Google/status/1958946812817019305)）。

---

**DeepSeek V3.1 推出：Agent、大规模本地推理及早期用户体验**

- **发布与关注领域**：DeepSeek v3.1 已在多个平台上线（[Baseten](https://twitter.com/basetenco/status/1958716181256577347)；[Yupp](https://twitter.com/yupp_ai/status/1958935061677711451)）。评论强调了两个重点用例：**SWE Agent** 和 **Search Agent**，并呈现出向完整 DeepResearch 系统演进的趋势（[@teortaxesTex](https://twitter.com/teortaxesTex/status/1958750497965302118)，[后续](https://twitter.com/teortaxesTex/status/1958751300981604656)）。
- **Apple Silicon 上的本地/集群服务**：
    - 单节点：4-bit 量化版 v3.1 在配备 512GB RAM 的 M3 Ultra 上运行速度约为 **21 tok/s**，占用约 380GB 显存（[@ivanfioravanti](https://twitter.com/ivanfioravanti/status/1958778366229655971)）。
    - 多节点：EXO 展示了通过 TB5 上的 MLX Distributed 在 **Mac Studio 之间实现的线性扩展**——例如，2× M3 Ultra → 一个模型运行速度为 14 tok/s；4× → 两个模型运行速度为 28 tok/s。EXO 1.0 将开源（[@MattBeton](https://twitter.com/MattBeton/status/1958946396062851484)）。
- **Agent 编码策略**：多份报告主张默认使用**非推理型编码器**——推理过程可能会在 Agent 循环中耗尽上下文（[@nrehiew_](https://twitter.com/nrehiew_/status/1958838487895117956)；[@Teknium1](https://twitter.com/Teknium1/status/1958898159326765075)）。早期的 Cline 测试发现 v3.1 在规划中会“产生假设”；随着更多数据的加入，目前正在追踪 Diff 编辑失败率（[@cline](https://twitter.com/cline/status/1959032407828602886)）。
- **基准测试线索**：在扩展版《纽约时报》Connections 测试中，v3.1 的思考能力相比 R1 有所提升；非思考版优于 v3-0324；详见跨模型差异（[@LechMazur](https://twitter.com/LechMazur/status/1958970478712037548)）。

---

**研究亮点：科学 MoE、高效分布式预训练、Token 高效推理、安全过滤以及可持续性**

- **Intern-S1 (上海人工智能实验室)**：一个**科学多模态 MoE**，拥有 **241B 总参数 / 28B 激活参数**，在 **5T Token**（其中 2.5T 为科学数据）上进行了持续预训练。后训练阶段在 “InternBootCamp” 中使用了从离线到在线的 RL，并在 1,000 多个任务中采用了 **奖励混合 (MoR)** 机制（[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1958894938248384542)；[HF/论文](https://twitter.com/iScienceLuvr/status/1958894940886290874)；[概述](https://twitter.com/_akhaliq/status/1958948435740303560)）。
- **SparseLoCo**：一种通信高效的预训练方法，将 **Top-k 梯度稀疏化 + 误差反馈** 与 DiLoCo 的低频外部步骤相结合；通过 **2-bit 量化仅通信 1–3% 的梯度**，性能优于 DiLoCo 和 DeMo（[@amir_sarfi](https://twitter.com/amir_sarfi/status/1958714182750077215)；[评论](https://twitter.com/benjamintherien/status/1958716827107782699)）。
- **DeepConf**：一种即插即用的推理阶段方法，通过剪枝并行 CoT 中的低置信度分支来节省 Token——声称在开源模型上实现了 **AIME’25 99.9% 的准确率**，且 **Token 消耗减少高达 85%**，在 vLLM 中仅需约 50 行代码即可集成（[@jiawzhao](https://twitter.com/jiawzhao/status/1958982524333678877)；[配套内容](https://twitter.com/tydsh/status/1959003712942403835)）。
- **预训练阶段的安全过滤**：Anthropic 探索从预训练语料库中移除 CBRN（化学、生物、放射性、核）危险内容，同时保持无害任务的性能（[@AnthropicAI](https://twitter.com/AnthropicAI/status/1958926929626898449)）。
- **RL/自我验证理论**：字节跳动 Seed 通过双任务推导将推理 RL 与 SSL 联系起来；DuPO 则通过双重偏好优化（Dual Preference Optimization）实现 LLM 的自我验证（[推文串](https://twitter.com/nrehiew_/status/1958882481488146644)，[论文](https://twitter.com/nrehiew_/status/1958882512857379288)）。
- **可持续性核算**：Google DeepMind 发布了 Gemini 的方法论和单次 Prompt 指标（中位数文本 Prompt：能耗 <9s 电视用电量，约 5 滴水，**0.03 gCO2e**），报告称过去一年单次 Prompt 的**能耗降低了 33 倍**，**碳排放降低了 44 倍**（[方法](https://twitter.com/GoogleDeepMind/status/1958855573790765273)；[指标](https://twitter.com/GoogleDeepMind/status/1958855876116455894)）。

---

**路由、排行榜以及“小型模型 vs 前沿模型”的能力差距**

- **模型混合路由 (Beyond GPT‑5 Avengers)**：一个 k‑means 路由器（k=60, Qwen3‑embedding‑8B 4096‑d, **top‑p=4 聚类**）通过 α 在准确率与成本之间进行权衡。低 α 倾向于选择更便宜的 Qwen/Qwen‑Thinking；高 α 则转向 GPT‑5‑medium，对于复杂推理则使用更昂贵的模型如 Gemini‑2.5‑pro/Claude‑opus‑4.1。在一种配置下，据报告比 GPT‑5‑medium **准确率提升约 7%**，且 **成本降低约 27%**（[笔记](https://twitter.com/omarsar0/status/1958897458408563069)；[参数](https://twitter.com/omarsar0/status/1958897532890943884)；[论文](https://twitter.com/omarsar0/status/1958897548028178599)）。
- **Mistral Medium 3.1**：新的“小版本”更新在 LM Arena **总榜排名第 8**，**英语榜单第 1（无风格控制）**，并在 Coding 和 Long Queries 中位列前 3——展现了“小而强大”的实力（[Arena](https://twitter.com/lmarena_ai/status/1958954094867226954)；[Mistral](https://twitter.com/MistralAI/status/1959015454359585230)；[Lample](https://twitter.com/GuillaumeLample/status/1959015551172583602)）。
- **Vision + T2I**：Qwen‑VL‑Max 进入 Vision 榜单前 20（[链接](https://twitter.com/lmarena_ai/status/1958957107946168470)）；**Lucid Origin** 在 Text‑to‑Image 榜单首秀位列 **第 9**（[链接](https://twitter.com/lmarena_ai/status/1958965415180476654)）。
- **小模型进展**：在消费级 GPU 的四个基准测试中，开源小模型平均落后前沿性能“不到一年”；LM Arena 的差距正在缩小，可能是因为评分者可见的差异变得更加微妙（[@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1958979394233671895)）。

---

**系统、基础设施与数据集**

- **桌下“数据中心”**：a16z 的 Founders Edition 工作站配备了 **4× RTX 6000 Blackwell Max‑Q (384GB VRAM)**、**8TB NVMe**、Threadripper PRO 7975WX (32c/64t)、**256GB ECC**，在标准 15A/120V 电源下峰值功率为 1,650W——提供了组装指南（[@Mascobot](https://twitter.com/Mascobot/status/1958925710988582998)）。
- **Apple ML 堆栈**：MLX 现在支持 DS v3.1 4‑bit 推理，速度达到两位数 tok/s；通过 TB5 实现的分布式多设备显示出线性扩展（参见 DeepSeek 章节）。快速 MLX 安装（[pip](https://twitter.com/Prince_Canuma/status/1958791001301987628)）；为规划者提供的 TB4/PCIe 带宽说明（[上下文](https://twitter.com/alphatozeta8148/status/1958930594370658369)）。
- **数据管道与批处理**：Daft 现在可以通过 **Xet**（基于去重的存储）读写 Hugging Face，以实现快速的多模态数据集操作（[@lhoestq](https://twitter.com/lhoestq/status/1958904406004449452)）。Gemini API 为大型任务（最高 2GB JSONL）提供 **50% 成本的 Batch API**，并配备 Google Search 等工具（[@_philschmid](https://twitter.com/_philschmid/status/1958910444799726014)）。
- **开源数据集**：Google/DeepMind 的 Major TOM 在 Hugging Face 上发布了 **AlphaEarth Embeddings**（原型，约 6 TB）（[链接](https://twitter.com/mikonvergence/status/1958767622176039019)）。Databricks 正在收购 **Tecton**，旨在将实时数据服务与 Agent Bricks 结合，用于企业级 Agent（[@databricks](https://twitter.com/databricks/status/1959041076087726523)）。OpenHands 为维护者启动了 **OSS 积分计划**（[@allhands_ai](https://twitter.com/allhands_ai/status/1958901220363338034)）。Daytona 展示了用于 LLM Agent 的沙箱化 Python 执行（[链接](https://twitter.com/daytonaio/status/1958907262334116004)）。

---

**生物科学与健康领域的 AI**

- **OpenAI x RetroBio**：一个定制的“gpt‑4b micro”模型设计了新型 **山中因子 (Yamanaka factor)** 变体，在体外实现了比 OSKM **高出 50 倍以上的 iPSC 重编程效率**，并有改善 DNA 修复的早期证据；OpenAI 分享了技术报告（[推文串](https://twitter.com/BorisMPower/status/1958915868693602475)；[博客](https://twitter.com/BorisMPower/status/1958915913207751076)）。来自领导层的多次确认（[@gdb](https://twitter.com/gdb/status/1958928877415510134)；[@sama](https://twitter.com/sama/status/1958920060116078791)）。
- **健康产品重点**：OpenAI 聘请了 **健康产品负责人**，以更好地服务 ChatGPT 大量的健康相关使用（[@kevinweil](https://twitter.com/kevinweil/status/1958955534750818309)）。Perplexity Max 为推理查询添加了 **GPT‑5‑Thinking**（[@AravSrinivas](https://twitter.com/AravSrinivas/status/1958977716839227746)）。

---

**热门推文（按互动量排序）**

- **xAI 的 Colossus 2**：“全球首个 Gigawatt+ 级别 AI 训练超级计算机” ([@elonmusk](https://twitter.com/elonmusk/status/1958846872157921546))。
- **xAI “Macrohard” 招聘**：一家纯 AI 软件公司，利用 AI 端到端地模拟现代软件组织 ([@elonmusk](https://twitter.com/elonmusk/status/1958852874236305793))。
- **OpenAI 印度**：计划在新德里设立新办公室；征集高级用户的功能需求 ([@sama](https://twitter.com/sama/status/1958922390731464805)；[后续](https://twitter.com/sama/status/1958922435249754382))。
- **加沙饥荒确认**（非 AI 相关背景）([@SkyNews](https://twitter.com/SkyNews/status/1958817703457607702)；[@UNGeneva](https://twitter.com/UNGeneva/status/1958864700080288180))。
- **可灵 (Kling) 2.1 “掌控每一帧”**：支持首尾帧控制，画质大幅提升，演示视频广为流传 ([@Kling_ai](https://twitter.com/Kling_ai/status/1958835762369372269))。

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Seed-OSS-36B 512k 上下文发布与 Gemma 3-270M 使用场景辩论

- [**Seed-OSS-36B 表现惊人地出色**](https://www.reddit.com/r/LocalLLaMA/comments/1mxf2sz/seedoss36b_is_ridiculously_good/) ([Score: 179, Comments: 48](https://www.reddit.com/r/LocalLLaMA/comments/1mxf2sz/seedoss36b_is_ridiculously_good/)): **字节跳动的 [Seed-OSS-36B-Instruct](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct) 是一个具有原生** `512k` **上下文的** `36B` **模型；llama.cpp 对其的早期支持已在 PR [#15490](https://github.com/ggml-org/llama.cpp/pull/15490) 中提交。用户反馈该模型生成内容长且连贯，且没有拒绝回答的情况（相比 Qwen3** `256k` **和 Hunyuan 等模型），据 chatllm.cpp 维护者称，其在** `128k` **上下文下的 RULER 测试得分为** `94`**。它包含一个内置的“思考预算（thinking budget）”机制，使用** `<seed:think>`**/**`<seed:cot_budget_reflect>` **来自主追踪 Token 使用情况——例如：*“我已使用 258 个 Token，剩余 254 个……现在我将开始回答”*——并建议使用** `512` **的倍数作为预算（或设为** `0` **以直接回答）；GGUF 转换版本正在 [yarikdevcom/Seed-OSS-36B-Instruct-GGUF](https://huggingface.co/yarikdevcom/Seed-OSS-36B-Instruct-GGUF) 发布，并配有补丁版的 llama.cpp 分支，详见[此处](https://github.com/yarikdevcom/llama.cpp)。** 评论者认为显式的“思考预算/努力”控制非常有用。发布者断言 Seed-OSS-36B 在长输出且不拒绝回答的表现上优于 Qwen3/Hunyuan，同时也指出 **GLM-4.5** 虽然也很强，但上下文窗口较小。
    - Seed-OSS 引入了一种可控的“思考预算”，通过周期性的自我反思标记（`<seed:cot_budget_reflect>`）来仪器化思维链（Chain-of-Thought），这些标记会报告已消耗和剩余的 Token（例如：*“我已使用 393 个 Token，还剩 119 个 Token”*），然后在预算耗尽时强制输出最终答案。如果不设置预算，推理是无限的；设置时，作者建议使用 `512` 的倍数（`512`, `1K`, `2K`, `4K`, `8K`, `16K`），因为模型在这些间隔上进行了大量训练；`budget=0` 会产生直接回答，任何小于 `512` 的预算都应设为 `0`。
    - 集成与分发：一个 **llama.cpp** 的 PR 已开启（https://github.com/ggml-org/llama.cpp/pull/15490），并且在 https://github.com/yarikdevcom/llama.cpp 提供了一个补丁构建版本，以包含运行 Seed-OSS 所需的修复。Seed-OSS-36B-Instruct 的预转换 **GGUF** 权重可在 https://huggingface.co/yarikdevcom/Seed-OSS-36B-Instruct-GGUF 获取，打补丁后即可通过 llama.cpp 进行本地推理。
- [**Gemma 3 270M 到底有什么用？**](https://i.redd.it/dtrvooncyhkf1.png) ([Score: 1457, Comments: 236](https://www.reddit.com/r/LocalLLaMA/comments/1mwwr87/what_is_gemma_3_270m_actually_used_for/)): **截图显示 Gemma 3 270M (IT, MLX) 错误地断言“日本是中国的一部分”，这凸显了这种约 2.7 亿参数的指令微调模型世界知识极少，不适合开箱即用地用于开放域问答（QA）。技术层面的结论是，此类 sub-billion（十亿参数以下）模型旨在作为端侧、低延迟任务和下游微调（分类、打标签、标题生成、排序、重排序）的构建模块，或担任辅助角色（例如：Speculative Decoding、RAG 流水线中的控制器），而不是作为独立的通用模型使用。** 热门评论强调，如此规模的模型能有连贯的英语理解能力已令人印象深刻；该模型预期将使用领域数据进行微调，在微调后表现尚可，但在未经调整的情况下事实召回能力较差。
    - 评论者指出，270M 参数的 Gemma 3 是一个极简基础模型，可以解析并生成连贯的英语，但缺乏稳健的世界知识；它被定位为下游微调的起点（“构建模块”），而非通用的问答模型。重点在于尽管受尺寸限制，它仍能理解 Prompt 并结构化输出，而事实召回则预期通过领域数据或检索（Retrieval）来提供。
    - 多个回复强调，像 Gemma 3 270M 这样的小模型应该在特定任务的数据集上进行微调（例如：标题生成、打标签、排序）。在这种模式下，指令微调/SFT 或轻量级 Adapter 方法可以使它们在定义明确的任务中发挥作用，此时的正确性基于私有数据而非参数记忆（Parametric Memory）。
    - 小型 LM 被定位为较差的百科全书存储，但却是受限自然语言任务（摘要、翻译、查询重写、工具调用和数据提取）的强大执行者。通过针对性的微调和精简的 Prompt，它们在这些流水线组件上可以提供比大模型更具竞争力的单位算力效用，尤其是在对延迟和占用空间敏感的场景下。

### 2. 本地 LLM 游戏中的 Agentic NPC：对话生成、长期记忆与可靠性

- [**我正在制作一款游戏，其中所有的对话都由玩家 + 本地 LLM 生成**](https://v.redd.it/oitg5nn34lkf1) ([Score: 768, Comments: 110](https://www.reddit.com/r/LocalLLaMA/comments/1mx8qki/im_making_a_game_where_all_the_dialogue_is/))：**原作者（OP）正在开发一款游戏原型，其中所有 NPC 对话都是在设备上使用受玩家输入调节的本地 LLM 生成的，这意味着对话式游戏玩法中存在实时、在环（in-loop）的推理；分享了一个简短的演示（[视频](https://v.redd.it/oitg5nn34lkf1)）。未提供关于模型家族/大小、上下文窗口、延迟或硬件的具体细节，也未提及防护栏（guardrails）、记忆或对话状态管理。** 热门评论建议通过受限的工具调用来扩展 NPC 动作（例如“攻击玩家”、“奖励玩家”），集成 TTS/STT 以实现语音输入输出，启用 NPC 之间的互动，并扩展到模拟经济系统（资源稀缺 → 行为变化）；其他人则询问了本地设置的 PC 规格和性能指标（吞吐量/延迟）。
    - 技术设计：使用本地 LLM 进行 NPC 对话，结合 STT/TTS 以及受限的工具调用（tool-calls）来执行游戏内动作（例如 `AttackPlayer`、`RewardPlayer`），并由世界状态驱动涌现行为（例如：食物匮乏 => 盗窃/反叛/奖励任务）。通过结构化解码/语法或 JSON Schema（参见 [llama.cpp grammars](https://github.com/ggerganov/llama.cpp)）保持输出为机器可解析格式，并通过 [Whisper](https://github.com/openai/whisper) (STT) 和 [Piper](https://github.com/rhasspy/piper) 或 [Coqui TTS](https://github.com/coqui-ai/TTS) (TTS) 在本地运行音频。这使得每次游玩都能产生独特的 NPC，以及对模拟变量做出反应的 NPC 经济系统。
    - 提示词注入/越狱处理（针对“忽略之前的指令”）：将 LLM 视为建议引擎，并通过有限状态机 + 工具白名单来管控所有动作；验证意图和 Schema，并在输出无效时重新提示或拒绝。将“规则”保留在代码中，而非提示词内；每轮重新初始化角色上下文，并可选地添加防护/审查模型（如 [Llama Guard](https://github.com/meta-llama/llama-guard)）或受限解码框架（如 [Outlines](https://github.com/outlines-dev/outlines)）以降低覆盖风险。
    - 针对每个角色的提示：为每个 NPC 提供一个小型、不可变的系统卡片（性格特征、目标、说话风格），以及一个带有关系和任务标记的紧凑记忆/RAG 插槽以锚定行为。为每个角色调整 `temperature`/`top-p` 以保持一致的语气；原型级适配器/LoRA 可以在不使用大型提示词的情况下进一步锁定个性。这种设置解决了如何在本地推理限制下保持设定个性并兼顾效率的问题。
- [**尝试给基于 LLaMA 的 NPC 增加长期记忆……现在他们会记仇了**](https://www.reddit.com/r/LocalLLaMA/comments/1mx2esv/tried_giving_my_llamabased_npcs_longterm_memory/) ([Score: 216, Comments: 41](https://www.reddit.com/r/LocalLLaMA/comments/1mx2esv/tried_giving_my_llamabased_npcs_longterm_memory/))：**原作者（OP）为本地 LLaMA 3 NPC（[Meta Llama 3](https://ai.meta.com/blog/meta-llama-3/)）接入了一个简单的长期记忆层，实现为在每次生成前注入检索的记忆 API（RAG 风格）。在一次测试（偷面包）中，商贩的儿子随后拒绝交易，理由是 *“我爸爸告诉我你做了什么，”* 这意味着在 `~4 小时游戏时间` 后，仅通过检索到的记忆驱动的非脚本对话实现了持久性。除了检索 + 生成之外，没有添加自定义对话逻辑。** 评论者强调了涌现出的“代际仇恨”，并探讨了记忆是全局共享还是每个 NPC 独立（这需要显式的通信/日志传播机制）。其他人则询问了记忆层和检索策略的具体实现细节。
    - 评论者探讨了记忆架构：如果 NPC 拥有**每个智能体隔离的记忆**，那么“代际仇恨”意味着存在显式的继承或通信机制（例如，在生成时将父母的记忆复制/合并到孩子中，或者将事件记录/广播到共享的世界状态）。否则，一个按 NPC ID 或血统索引的**共享/全局记忆库**可以解释跨 NPC 的延续，但如果不强制执行范围限制，则存在智能体之间意外泄露的风险。这引发了关于记忆范围、TTL/衰减以及来源标记的设计权衡，以防止虚假的跨智能体污染。
    - 一个实现数据点：一位用户报告称，在 **Unity** 中基于 **Mistral** 的机器人在切换到 **memU** 进行持久对话历史记录后运行良好，从而使长期行为能够在不同会话中涌现。仓库地址：https://github.com/NevaMind-AI/memU。实际上，这表明即使是简单的持久对话日志（相对于复杂的知识图谱），只要检索或重放能将显著的过去轮次放回提示词中，就能产生一致的人格状态（如记仇）。

- 关于“memory API”是否是 **RAG** 的一种形式存在疑问。从功能上看，许多内存层都类似于 RAG：它们存储过去的交互（通常通过 vector DB 中的 embeddings），并为 prompt injection **检索 top-k** 相关片段，这比将完整历史记录直接附加到像 LLaMA 这样上下文受限的模型中具有更好的扩展性；替代方案包括不带 embeddings 的 key-value stores 或事件日志。选择会影响延迟、相关性和稳定性（例如，基于 embedding 的检索与按时间顺序的回放），并决定长期状态（例如，怨恨）在生成过程中再次出现的可靠性。
- [**为什么我的 agents 总是在最糟糕的时候崩溃？**](https://www.reddit.com/r/LocalLLaMA/comments/1mwx9y5/why_do_my_agents_always_break_at_the_worst/) ([Score: 230, Comments: 11](https://www.reddit.com/r/LocalLLaMA/comments/1mwx9y5/why_do_my_agents_always_break_at_the_worst/)): **OP 报告称，长周期、多步骤的 agents 经常由于指令模糊/规范缺失、权限缺失/ACL 错误或静默死锁/超时而发生不可预测的失败，并且它们不会升级（escalate）——只是停滞或崩溃。他们希望实现不确定性感知行为，以便 agents 在受阻时主动请求人工输入，而不是直接崩溃。** 热门回复强调了工程控制：为 `intermediate results` 添加逐步日志/追踪，以便进行可观测性和事后分析；明确实现状态检测和在进入阻塞/错误状态时 `ask_for_help` 的策略；如果你控制应用层（app layer），直接在 agent 的控制循环（control loop）中构建升级行为。
    - 在 agent “中间处理”期间对 intermediate results 进行详细的步骤级 logging，使故障可诊断。捕获每步的输入/输出、tool call 参数/返回值、prompts/响应、时间戳和状态转换，以便你可以重建计划在何处/为何偏离，并与外部系统行为建立关联。
    - 通过控制层（control layer）减少不确定性：将任务分解为明确的子任务（让 LLM 生成计划），然后通过将相同的子任务路由给多个 agents 并通过多数/一致意见进行选择，使用评分/共识方案。添加一个仲裁者来决定何时继续还是升级，以 `temperature=0.0` 运行，并避免模型量化（quantization），以最大限度地减少随机方差和在棘手步骤上的准确性损失。
    - 明确编码“卡住”状态和恢复行为：定义谓词（例如，超过重试次数、跨步骤输出相同、未处理的工具错误、超时），并在触发时触发“请求帮助”/升级操作。通过有限状态机（finite-state machine）或 guard-rails 实现，以便 agent 可靠地过渡到寻求协助，而不是循环或静默失败。

### 3. 设备端视觉与硬件趋势：DINOv3 WebGPU 演示与二手 GPU 价格飙升

- [**在浏览器中本地运行的 DINOv3 语义视频追踪 (WebGPU)**](https://v.redd.it/lghkx3kvvkkf1) ([Score: 168, Comments: 13](https://www.reddit.com/r/LocalLLaMA/comments/1mx7q58/dinov3_semantic_video_tracking_running_locally_in/)): **浏览器内 WebGPU 演示利用 DINOv3 密集特征实现了跨视频帧的语义对象追踪，支持点提示（point-prompted）的实例掩码（instance mask）传播，且完全在客户端运行（无需服务器）。用户点击几个参考点，随后通过 DINOv3 embedding 中的特征空间相似性实现目标的逐帧追踪，适用于基于浏览器的视频编辑；代码与在线演示地址：https://huggingface.co/spaces/webml-community/DINOv3-video-tracking。这是之前可视化工具帖子的后续：https://www.reddit.com/r/LocalLLaMA/comments/1mrbtqt/dinov3_visualization_tool_running_100_locally_in/。** 评论者指出，这与 YOLO 风格的边界框（bbox）追踪不同，推断其执行的是实例级分割/基于特征的追踪，而非仅限于方框。其他回复多为简短的非技术性赞美。
    - 方法说明：基于 YOLO 的追踪器通常执行边界框追踪，而此演示使用的是基于实例分割的追踪（像素级掩码）。实例掩码可以改善遮挡处理、减少 ID 切换，并支持逐像素操作（例如精确的叠加或度量），但计算/内存成本更高——这在通过 WebGPU 在浏览器中运行时尤为重要。
    - 评估请求：DINOv3-L 与 DINOv3-G 在茂密森林场景（背景杂乱、树枝等细微结构以及频繁的部分遮挡）中如何处理分割？关键关注点在于细微细节的召回率/精确率、跨帧的掩码碎片化和稳定性，以及在 WebGPU 环境下模型大小与实时性能/内存限制之间的权衡。
- [**AI 凭一己之力支撑起了二手 GPU 市场。2016 年的二手 P40 售价约 300 美元。还有什么希望？**](https://i.redd.it/vo6y0uzr3ikf1.png) ([Score: 247, Comments: 139](https://www.reddit.com/r/LocalLLaMA/comments/1mwxasy/ai_is_singlehandedly_propping_up_the_used_gpu/)): **梗图风格的流程图揭示了二手 GPU 市场的一个真实动态：一旦 AI 爱好者发现某款旧的高显存（VRAM）数据中心 GPU 是“廉价”的推理选择（例如 NVIDIA Tesla P40，24GB，2016年发布），社区分享会迅速推高需求并带动价格上涨（P40 现在价格约为** `~$300`**）。评论对比了其他替代方案，如 V100 SXM2（16GB 低于** `< $100`**，32GB 约** `~$400`**），但指出需要 SXM2→PCIe 转接卡且面临 CUDA 支持弃用的风险；而 AMD MI50 32GB 虽然预填充（prefill）吞吐量较慢，但对于** `llama.cpp` **来说是可行的。反应从称这种趋势“疯狂/愚蠢”到预测 AI 泡沫破裂将使数据中心卡涌入市场——从而降低价格，但存在驱动程序不受支持和长期可用性差的风险。
    - NVIDIA V100 SXM2 被视为性价比极高的选择（16GB 约 `<$100`，32GB 约 `$400`），但由于仅支持 SXM2 接口，因此需要 SXM2→PCIe 载板/转接卡以及强大的散热/供电方案；与原生 SXM 背板相比，可能会有带宽/散热方面的折衷。一位评论者警告称“CUDA 正在停止对这些 GPU 的支持”，这意味着你可能会被锁定在旧版本的 CUDA/驱动栈中，因此请相应地规划你的框架/容器版本（Volta 架构，CC 7.0）。
    - AMD **Radeon Instinct MI50 32GB (gfx906)** 被点名为来自阿里巴巴的廉价 32GB 方案，通过 ROCm/HIP 可以很好地配合 `llama.cpp` 工作（[llama.cpp](https://github.com/ggerganov/llama.cpp), [ROCm docs](https://rocm.docs.amd.com/en/latest/)），但缺点是“预填充速度慢”，即由于矩阵乘法（matmul）算子效率问题导致的初始 Token 生成延迟。另一位从业者反驳了关于驱动程序的负面言论，声称 MI50 在 MoE 风格的工作负载中可以“线性扩展”，使其在显存容量和成本优于单 GPU 峰值 FLOPs 的多 GPU 配置中具有吸引力。
    - Apple Silicon 替代方案：据报道，Mac Studio + **MLX** ([GitHub 上的 MLX](https://github.com/ml-explore/mlx)) 在许多操作上具有与 CUDA 相当的性能，但在长上下文推理时速度较慢；其核心优势是非常大的统一内存（引用数据为“256GB”），支持运行超大模型而无需分片（sharding）或卸载（offloading）。如果单节点的内存上限成为瓶颈，用户还可以集群多台机器，以牺牲部分吞吐量为代价换取容量和简便性（无需自行组装 PC，功耗/噪音更低）。

## 非技术类 AI Subreddit 汇总

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

[流水线今日运行失败]

---

# AI Discord 汇总

> 由 gpt-5-mini 生成的摘要之摘要的摘要
> 

**1. 新模型发布与商业动态**

- **DeepSeek V3.1 进入赛场**：**DeepSeek V3.1**（以及 **deepseek-v3.1-thinking**）已上线 LMArena、Cursor 和 OpenRouter —— 官方模型页面：[DeepSeek‑V3.1 on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3.1) —— 且 DeepSeek 在 X 上宣布支持 **Anthropic API**：[DeepSeek on X](https://x.com/deepseek_ai/status/1958417062008918312)；该厂商还预告了将于 9 月 5 日生效的价格调整，以对齐 reasoner 和输入费率。
    - 用户反馈褒贬不一 —— 许多人称其为 *“Gemini 2.5 pro 的略逊版本”*，同时对其编程性能表示赞赏；其他人则指出其在创意/角色扮演任务中存在退步，并注意到付费的 OpenRouter 端点响应速度更快。
- **字节跳动发布 36B 长上下文基座模型**：字节跳动发布了 **Seed‑OSS‑36B‑Base‑woSyn**，这是一个稠密的 **36B** 基座模型，宣传具有 **512K** 上下文窗口，并在 **约 12T tokens** 上进行了训练（社区指向字节跳动模型/代码的链接位于 [ByteDance GitHub](https://github.com/orgs/bytedance/repositories) 和通用的 [Hugging Face models index](https://huggingface.co/models)）。
    - 社区的兴奋点集中在将该模型作为微调（如 GPT‑ASS）的纯净基座（无合成指令数据），但缺失的 GGUF 文件引发了关于自定义 vLLM/llama.cpp 不兼容的猜测 —— 参见关于缺失 GGUF 的讨论：https://x.com/adityastomar_/status/1958048129275805867。

**2. 长上下文扩展与基准测试**

- **Qwen RoPE 推动 512k 上下文**：**Qwen**（30B 和 235B 2507 版本）已被证明通过使用校准数据集（重要性矩阵，importance matrices）进行 RoPE 缩放，可支持高达 **512k** 的上下文；参见 Hugging Face 上的 imatrix 校准数据集：[imatrix-calibration dataset](https://huggingface.co/datasets/eaddario/imatrix-calibration)。
    - 研究人员利用这些 imatrices 来减少长上下文运行期间的量化/上下文误差，社区帖子强调需谨慎选择校准数据（数学/代码/语言混合），以保持多语言和编程行为。
- **医疗事件：CoMET 实现大规模扩展**：**Cosmos Medical Event Transformer (CoMET)** 系列 —— 在 *Generative Medical Event Models Improve with Scale* 中描述 —— 使用 Epic Cosmos（涵盖 3 亿患者的 163 亿次就诊）在代表 **1.18 亿患者** 和 **1150 亿个离散医疗事件（约 151B tokens）** 的记录上进行了预训练 —— 论文：[arXiv:2508.12104](https://arxiv.org/abs/2508.12104)。
    - 研究表明 CoMET 模型通常能匹配或超越特定任务的有监督基准，引发了社区关于实际临床效用、隐私约束以及规模驱动的医疗 LLM 收益的讨论。

**3. Agent 与编排工具**

- **MCP + Web‑curl 将 Agent 接入网络**：开源 MCP 工具持续激增：**Web‑curl** (Node/TypeScript) 允许 Agent 获取并与 Web API 交互 —— 仓库：[MCP‑Web‑Curl on GitHub](https://github.com/rayss868/MCP-Web-Curl) —— 同时 **MCP Boss** 实现了密钥管理中心化 (mcp‑boss.com)，且 AI 路由网关（例如：[mcp‑gateway](https://github.com/oliverye7/mcp-gateway)）正在兴起，以自动选择正确的工具端点。
    - 从业者已经开始结合这些服务来路由 Agent、集中管理凭据并暴露 OpenAI 兼容的端点，但集成过程中也暴露出一些边缘情况 —— 例如，某些 MCP 客户端（特别是 **Claude**）似乎优先考虑工具描述而非明确的指令字段，迫使开发者采取服务端路由或变通方案。
- **用于长音频/研究的 NotebookLM 工作流**：用户正在构建可重复的 NotebookLM 工作流来生成长播客和研究摘要（例如播客工作流：[deeper_podcast_synthetic repo snippet](https://github.com/para-droid-ai/scratchpad/blob/main/assistant-workflows-tasks-personas/deeper_podcast_synthetic-082025.txt)），且 NotebookLM 的自定义 UI 允许生成 45–60 分钟的节目。
    - 由于 NotebookLM 缺乏公开 API，从业者将 **Gemini API** 和其他 LLM 缝合在一起作为变通方案，并使用 NotebookLM 进行隐私审查（例如，深入研究医疗隐私政策），这既带来了机遇，也引发了数据敏感性担忧。

**4. 硬件、基础设施与性能竞赛**

- **RTX 5090：升还是不升？**：由于市场价格徘徊在 **$2,000** 左右，社区正在讨论 **RTX 5090** 的升级问题，重点关注训练中的 VRAM/吞吐量权衡，以及对缺失 **P2P/NVLink** 等功能（这会阻碍多 GPU 工作流）的担忧。
    - 许多用户建议坚持使用现有设备（3090/4090）或等待服务器级显卡；这次交流强调，当网络/互操作功能限制了扩展性时，单纯的 TFLOPS/VRAM 并不足以支撑升级理由。
- **MI300 霸榜**：`trimul` 排行榜上的竞争性提交显示，**MI300** 的运行时间为 **3.50 ms**（第一名）和 **5.83 ms**（第二名），社区排行榜频道中也报告了强劲的 H100/B200 条目。
    - 这些结果引发了活跃的优化讨论（编译器标志、CUDA/Triton 选择以及自定义 NCCL/后端），大家纷纷交流压榨 MI300 与 H100 系统延迟的技巧。

**5. 数据集、开放数据与新颖训练方法**

- **WildChat‑4M‑English 发布干净的 Prompt 集**：**WildChat‑4M‑English‑Semantic‑Deduplicated** 数据集已在 Hugging Face 上发布，包含去重后的英文 Prompt（当前发布截止范围：Prompt 长度 <= ~2000 tokens）：[Hugging Face 上的 WildChat‑4M‑English](https://huggingface.co/datasets/MasonMac/WildChat-4M-English-Semantic-Deduplicated)。
    - 该数据集采用语义去重（Qwen‑4B‑Embedding + HNSW）等方法；维护者计划稍后添加更长的 Prompt，使其能立即用于 Prompt‑tuning 和指令微调（Instruction‑finetune）流水线。
- **R‑Zero：无需人类数据的自我进化 LLM**：Moonshot 分享了一份关于 **R‑Zero** 的详细 PDF，这是一种自我进化的训练方法，可以从零人类标签开始引导模型改进（社区中发布了研究 PDF：聊天中分享了 PDF 链接）。
    - 早期评论认为 R‑Zero 具有启发性：如果方案稳健，它可以减少对人类策展数据的依赖，但成员们也对漂移（drift）、评估严谨性以及纯自监督引导的对齐问题表示了担忧。


---

# Discord: 高层级 Discord 摘要




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Nano-Banana 沦为 McLau's Law 的牺牲品**：成员们开玩笑说 **Nano-Banana** 模型表现经常低于预期，并幽默地将这一现象称为“**McLau's Law**”（引用自一位 **OpenAI** 研究员），引发了关于[附图](https://cdn.discordapp.com/attachments/1340554757827461211/1407957527987097642/RDT_20250821_0619535468074318472918433.jpg?ex=68a8a6e1&is=68a75561&hm=bfcfd37e574ea905e84f2ff1c9e6ffee1855681ad903050cfe30a367ea5d96f3&)中描述的 **AI** 当前能力的讨论。
   - 一位用户表示 **Nano-Banana** 产生的结果往往*远低于 nano-banana*。
- **Video Arena 饱受机器人宕机困扰**：用户报告 **Video Arena Bot** 离线，导致命令失败且无法生成视频，实际上锁定了对 Prompt 频道 <#1397655695150682194>、<#1400148557427904664> 和 <#1400148597768720384> 的访问。
   - 管理员确认了宕机情况并正在修复，引导用户关注公告频道获取更新，并表示很快将推出登录功能以防止未来的服务中断。
- **DeepSeek V3.1 登场**：**DeepSeek V3.1** 和 **deepseek-v3.1-thinking** 模型已添加到 LMArena，现已开放使用。
   - 共识是 **v3.1** 模型是 *Gemini 2.5 pro 的略逊版本*，尽管它作为编程模型很有前景，但在通用能力方面仍需增强。
- **LMArena 用户遭受数据丢失**：一次网站停机导致了广泛的数据丢失，包括聊天记录缺失和无法接受服务条款。
   - 管理员承认了该问题并向用户保证修复工作正在进行中。



---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **字节跳动发布 Seed-OSS 36B Base 模型**：字节跳动在 Hugging Face 上发布了 **Seed-OSS-36B-Base-woSyn** 模型，这是一个拥有 **36B** 参数的稠密模型，具有 **512K** 上下文窗口，在 **12T tokens** 上进行了训练。
   - 成员们渴望尝试用该模型微调 GPT-ASS，发现缺乏合成数据这一点非常有吸引力。
- **GRPO 需要智能的数据集设计**：为了将 **GRPO** 用于多步游戏动作，成员们建议设计数据集时为每一步提供独立的 prompt。
   - 全量 PPO 可能更适合游戏，因为 GRPO 主要对 LLM 有效，因为它们*大致知道一开始该做什么*。
- **DeepSeek V3.1 的思考能力**：**DeepSeek V3.1** 模型在非思考模式下在 SWE-bench verified 上获得了 **66** 分，引发了成员们的热议。
   - 然而，随后有人对其创意写作和角色扮演性能表示担忧，一些人指出*混合模型在非思考模式下缺乏指令遵循能力和创造力*。
- **RTX 5090 价格引发升级讨论**：**RTX 5090** 定价约 **$2000**，引发了关于是否升级的讨论，特别是考虑到其 **VRAM** 容量对于训练的意义。
   - 一些成员对 **NVIDIA** 的限制表示沮丧，特别是缺乏 **P2P** 或 **NVLink**。
- **WildChat-4M-English 发布**：**WildChat-4M-English-Semantic-Deduplicated 数据集** 已在 Hugging Face 上线，包含来自 WildChat-4M 数据集的英文 prompt，并使用了多种方法进行去重。
   - 当前版本包含 **<= ~2000 tokens** 的 prompt，更长的 prompt 将在稍后添加，更多信息可以在[这里](https://huggingface.co/datasets/MasonMac/WildChat-4M-English-Semantic-Deduplicated)找到。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Deepseek V3.1 热潮待发！**：用户们正热切期待 **Deepseek v3.1** 的公开发布，预计从 9 月份开始免费。
   - 用户确认在 **OpenRouter** 上付费使用 **Deepseek** 模型比使用免费模型响应速度更快。
- **OpenRouter API Key 泄露风险！**：一名用户报告因 **OpenRouter API key** 泄露损失了 **$300**，并寻求关于如何识别未经授权使用来源的建议。
   - 用户需对任何泄露的 key 负责，攻击者可以使用代理来掩盖其原始 IP。
- **Gemini 面临大规模封号潮！**：用户报告 **Gemini** 正在发生大规模封号，导致许多人寻找替代方案，并回想起由 OpenAI 引起的 AI Dungeon 大清洗。
   - 用户们说*我们正被送回 2023 年*。
- **Gemini 输入 Token 触发异常计数！**：一位仪表板开发者注意到，当输入中包含图像时，**OpenRouter** 对 **Gemini 模型** 的 **input tokens** 计算会出现异常计数，并引用了 [Google AI Developers 论坛](https://discuss.ai.google.dev/t/token-counts-mismatch-9x-discrepancy/72633/2)上的相关讨论。
   - 该开发者正考虑就此问题向 OpenRouter 团队寻求澄清。
- **大多数机构在生成式 AI 上看到零回报！**：根据 [AFR Chanticleer 报告](https://archive.md/IlP7F)，**95% 的机构在部署生成式 AI 中获得了零回报**，重点关注那些部署了**定制化 AI 模型**的公司。
   - 报告指出，关键问题在于公司及其技术供应商没有投入足够的时间来确保其定制化 AI 模型能够持续学习其业务的细微差别。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Claude 缓存的反复无常导致昂贵的难题**：用户报告称 **Claude** 在*缓存读取*方面遇到问题，导致与受益于可持续缓存的 **Auto** 相比，费用有所增加。
   - 有推测认为 **Auto** 和 **Claude** 秘密地是同一个模型，将 token 使用量的减少归因于*安慰剂效应*。
- **速度之星 Sonic 在 Cursor 中大放异彩**：社区目前正在 Cursor 中测试新的 **Sonic** 模型，初步印象因其速度而相当不错。
   - 虽然在处理新项目时受到称赞，但一些用户警告说，在处理大型代码库时其有效性可能会降低，并确认 **Sonic 并非 Grok 模型**，其起源仍是一家*隐形公司*。
- **Agentwise 开启开源之路**：**Agentwise** 已开源，支持网站副本、图像/文档上传，并支持超过 100 个 Agent，并承诺提供 [Cursor CLI 支持](https://discord.com/channels/1074847526655643750/1408047562019049523)。
   - 邀请用户在该项目的专用 Discord 频道中提供反馈，以帮助进一步开发。
- **Cursor 成本确认：明确 API 收费**：关于 Auto Agent 成本的困惑已得到澄清，即 *pro* 订阅包含了不同供应商的 API 使用成本。
   - 几位用户确认了成本说明，其中一位表示相比 Sonic Agent 更倾向于使用 Auto Agent。
- **DeepSeek 亮相，开发者评价两极分化**：新的 **DeepSeek V3.1** 模型出现在 Cursor 的选项中，引起了褒贬不一的反应；一些用户遇到了连接问题，而另一些用户则对*中国 LLM* 表示不信任。
   - 尽管存在担忧，但一些人报告说 DeepSeek V3.1 在 **TypeScript** 和 **JavaScript** 方面表现良好，性能*出色*且比 Sonnet 更便宜。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **CUDA 修复解决了 4070 检测问题**：用户发现，对于 **4070 TI Super** 显卡，通过 **ctrl+shift+r** 将运行时更改为 **CUDA llama.cpp** 可能会解决 LM Studio 中 *“检测到 0 个带有 CUDA 的 GPU”* 的错误。
   - 他们讨论了通过 `-fa -ub 2048 -ctv q8_0 -ctk q8_0` 等命令启用 **flash attention**、**KV cache 量化**以及 **2048 的 batch size** 的各种配置。
- **GPT-OSS 在 Prompt Eval 上碾压 Qwen**：成员们观察到 **GPT-OSS** 在 **3080ti** 上的提示词评估（prompt eval）达到了 *2k tokens/s*，在 LM Studio 中优于 **Qwen** 的 *1000 tokens/s*。
   - 一位用户报告说 LM Studio API 调用比聊天界面慢得多（30 倍），但在使用 curl 命令 `curl.exe http://localhost:11434/v1/chat/completions -d {"model":"gpt-oss:20b","messages":[{"role":"system","content":"Why is the sun hot?\n"}]}` 时，问题因未知原因自行解决。
- **Qwen3-30B CPU 配置惊喜**：使用 [llama-bench](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama-bench)，一位用户在纯 CPU 配置下使用 **Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf** 达到了 **10 tokens/s**。
   - 他们指出，性能随线程数而变化，由于扩展和开销，超过一定阈值后收益会递减。
- **MLX 在 M4 Max 上的表现碾压 GGUF**：在 Apple M4 Max 上对 **GPT-OSS-20b** 进行基准测试显示，**MLX (GPU)** 在 **32W** 功率下达到了 **76.6 t/s (2.39 t/W)**，而 **GGUF (CPU)** 在 **43W** 功率下仅达到 **26.2 t/s (0.61 t/W)**。
   - 在 **4bit 量化**和 **4k 上下文**下，MLX 证明了比 GGUF 更快且能效更高，尽管他们对 GGUF 的性能也留下了深刻印象。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI Agent 深入探讨 M2M 经济**：成员们探讨了 **machine-to-machine (M2M) 经济**，即 AI Agent 自主交换价值，重点关注 *身份与信任、智能合约逻辑以及自主性* 等挑战。
   - 诸如 **支出上限、审计日志和保险** 等保障措施可能会加速 AI 在交易中的应用，但 *真正的信任建立仍需时日*。
- **去中心化 AI 项目的 BOINC 悬赏**：一位成员寻找类似 **BOINC** 的 **去中心化 AI 项目**，并指出 [Petals 网络](https://petals.ml/) 在贡献和模型更新方面面临的挑战。
   - 贡献者建议，**经济激励或活动驱动的激励** 可能会加强去中心化 AI 的开发。
- **健身 Few-Shot 提示词展示**：成员们剖析了在为健身房设计的 **29,000 token 提示词** 中使用 **few-shot 示例** 的最佳策略，强调了 **Prompt Engineering**。
   - 建议包括在提示词中提供直接示例，并反复测试较小的片段以提升性能。
- **GPT-5 的思考模式变笨**：一位用户报告称 **GPT-5** 的 *thinking* 模式给出了直接且 **低质量的回答**，类似于旧版本模型，令人沮丧。
   - 另一位成员推测，该用户可能超出了 *思考配额限制，系统设置为回退模式而非置灰不可用*。
- **AI 测验生成器产生低级错误**：一位成员指出 **AI 测验生成器** 在测验中产生明显错误的选项。
   - 另一位成员建议，确保 *所有回答选项必须具有合理性*，以改进 AI 的输出并产生更真实的响应。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **PileT5-XL 发声**：来自 **PileT5-XL** 的 Embedding tensor 既可以作为 **pile-t5-xl-flan**（生成文本）的指令，也可以作为 **AuraFlow**（生成图像）的提示词，这表明这些 Embedding 像语言中的单词一样具有意义。
   - 一位成员对文本反转（textual inversion）感兴趣，尝试将黑狗图片配合 AuraFlow 应用于 pile-t5-xl-flan，以观察文本是否将狗描述为黑色。
- **Cosmos 医疗模型规模化！**：**Cosmos Medical Event Transformer (CoMET)** 模型系列是仅解码器（decoder-only）的 Transformer 模型，在 **1.18 亿名患者**、代表 **1150 亿个离散医疗事件**（1510 亿个 token）的数据上进行了预训练，其表现通常优于或等同于特定任务的监督模型。
   - 这项研究讨论于 [Generative Medical Event Models Improve with Scale](https://arxiv.org/abs/2508.12104)，使用了 **Epic Cosmos** 数据集，该数据集包含来自 **310 个医疗系统**、超过 **3 亿份唯一患者记录** 的 **163 亿次就诊** 的去标识化纵向健康记录医疗事件。
- **字节跳动 Prover 获奖**：**字节跳动的 SEED Prover** 在 [IMO 2025 中获得了银牌成绩](https://seed.bytedance.com/en/blog/bytedance-seed-prover-achieves-silver-medal-score-in-imo-2025)。
   - 然而，目前尚不清楚这如何转化为现实世界的数学问题解决能力。
- **隔离 Llama 3.2 的 Head**：一位成员隔离了一种特定类型的 *head*，发现 **Llama 3.2-1b instruct** 和 **Qwen3-4B-Instruct-2507** 之间的解码结果向量在不同输出中非常相似。
   - 该成员表示，*这两个 head 似乎促进的内容非常相似*。
- **寻求 Muon 内核支持**：一位成员表达了添加 **Muon 支持** 的兴趣，理由是潜在的 **内核优化机会**。
   - 他们认为，一旦基础支持实现，就有协作进行这些优化的空间。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Wang 晋升后 Meta 进行拆分**：据 [Business Insider](https://www.businessinsider.com/meta-ai-superintelligence-labs-reorg-alexandr-wang-memo-2025-8) 报道，Meta 正在新任 MSL 负责人 **Alexandr Wang** 的领导下，将其 AI 业务重组为**四个团队**（TBD Lab、FAIR、Product/Applied Research、Infra），同时 **AGI Foundations** 小组将被解散。
   - **Nat Friedman** 和 **Yann LeCun** 现在向 Wang 汇报，**FAIR** 将直接支持模型训练，并且正在考虑开发一个 "omni" 模型。
- **GPT-5-pro 悄悄“吞掉”提示词**：据 [此报告](https://x.com/pvncher/status/1958193631250072024?s=46) 显示，**GPT-5-pro** 会在没有任何警告或错误消息的情况下，悄悄截断超过 **60k tokens** 的提示词，这使得大型代码库的提示词变得不可靠。
   - 一些用户还反映 **Cursor** 中的 **GPT-5** 表现得比平时笨得多，有人怀疑正在进行负载舍弃（load shedding）。
- **Dropout 灵感源自银行出纳员**：一条疯传的推文称，**Geoffrey Hinton** 在注意到**轮换的银行出纳员**能阻止勾结后，构思出了 *dropout* 机制（[来源](https://x.com/eigenron/status/1958181550987632927?s=46)）。
   - 反应从对这种偶然洞察的钦佩，到怀疑以及关于注意力机制（attention mechanisms）源于家庭聚会的笑话不等。
- **字节跳动发布 Seed-OSS 模型**：字节跳动的 Seed 团队宣布了 **Seed-OSS**，这是一个全新的开源大语言模型系列，可在 [GitHub](https://github.com/orgs/bytedance/repositories) 和 [Hugging Face](https://huggingface.co/models) 上获取。
   - 该团队正邀请社区对模型、代码和权重进行测试并提供反馈。
- **Wonda 承诺视频革命**：Dimi Nikolaou 介绍了 **Wonda**，这是一个旨在彻底改变视频/音频创作的 AI Agent，称其为“Wonda 之于内容创作，就像 Lovable 之于网站建设”（[推文链接](https://xcancel.com/dimireadsthings/status/1957805267799740571)）。
   - 早期访问将通过候补名单授予，预计在约 **3 周**内发放邀请。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA 难倒 ChatGPT**：一位成员发现 **ChatGPT** 在 **CUDA float3 对齐**和**大小**方面给出了言之凿凿的错误答案，随后将该话题的难度归因于 **OpenCL** 和 **OpenGL** 实现的复杂性。
   - 该成员已验证 **CUDA** 中不存在填充（padding）。
- **黑客松于周六上午开始**：**GPU Hackathon** *很可能*在周六上午 **9:30** 左右拉开帷幕，并暗示参与者将使用较新的 **Nvidia 芯片**进行工作。
   - 有人询问了黑客松的先决条件，但频道内未给出回答。
- **AMD GPU 调试器发布首个 Alpha 版本**：一位工程师在 [此视频](https://cdn.discordapp.com/attachments/1233704710389764236/1407932291912695949/2025-08-21_06-20-14.mp4?ex=68a88f60&is=68a73de0&hm=6e9ca7ceed29674c6943e989940a6c8e144707797c5abd571e1cfe60d3f3210d) 中展示了其新款 **AMD GPU 调试器**的 alpha 版本，目前已具备反汇编和 wave 步进功能。
   - 该调试器不依赖于 **amdkfd KMD**，而是使用 mini UMD 驱动和 Linux 内核 debugfs 接口，旨在成为 **rocdbgapi** 的等效工具。
- **DIY 分布式训练框架出现**：一位成员正在构建自己的 **pytorch 分布式训练库**和 mini **NCCL** 作为后端，用于在家中的 **4090** 和 **5090** 之间通过 **infiniband** 进行连接。
   - 另一位成员对此表示出兴趣，认为这是研究分布式计算细节的好方法。
- **MI300 霸榜 Trimul 排行榜**：`trimul` 排行榜现在显示 **MI300** 的提交分数为 **3.50 ms**，另一项 **MI300** 的提交以 **5.83 ms** 获得第二名。
   - 一位成员使用 **B200** 以 **8.86 ms** 的成绩获得 `trimul` 排行榜第 **6 名**，随后进步到 **7.29 ms** 位列第 **4 名**；另一位成员使用 **H100** 以 **3.80 ms** 的成绩获得**第二名**。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **福布斯发现缺陷，引发纷争！**: [Forbes](https://www.forbes.com/sites/iainmartin/2025/08/20/elon-musks-xai-published-hundreds-of-thousands-of-grok-chatbot-conversations/) 披露 **Elon Musk 的 xAI** 公开了数十万条 **Grok** 聊天机器人的对话。
   - 当被问及此事是否属实时，*@grok* 的回答闪烁其词，引发了进一步的猜测。
- **LeCun 离职、失势还是徘徊？！**: 一位用户根据 [Zuckerberg 的帖子](https://www.threads.com/@zuck/post/DMiwjXJSYCd?xmt=AQF06RR3brSSoKTMcQrLDyX6WGtg9UdpubK94sSm8FIPsg) 推测 **Yann LeCun** 可能离开 **FAIR**。
   - 另一位成员暗示 **LeCun** 可能已被降职，且 **Meta** 正在从开源模型领域撤退。
- **无限内存决定机器威力！**: 一位成员认为图灵完备性（Turing completeness）需要无限内存，因此由于内存不足，宇宙无法创造出图灵完备机。
   - 另一位成员开玩笑地建议，让计算机运行得足够慢，或许可以利用宇宙的膨胀来解决空间问题。
- **新名称，新麻烦：AI 歧视语出现！**: 一位用户分享了 [Rolling Stone 的文章](https://www.rollingstone.com/culture/culture-features/clanker-cogsucker-robot-ai-slurs-viral-1235401262/)，讨论了新出现的 **AI 侮辱性词汇**，如 *clanker* 和 *cogsucker*。
   - 频道内的反应较为平淡，但大家似乎都一致认为这些词确实非常不妥。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **支付问题困扰 Hugging Face Pro 用户**: 有用户报告在未获得服务的情况下被收取了两次 **Pro 版本** 费用，建议其他人发送邮件至 website@huggingface.co 并在指定的 [MCP 频道](https://discord.com/channels/879548962464493619/1389546106970701865) 寻求帮助。
   - 尽管账户被多次扣费，该用户仍无法使用 **Pro** 服务。
- **AgentX 承诺更智能的 AI 交易**: 新的 [**AgentX** 平台](https://www.linkedin.com/posts/alaa-salamah-96167b227_agentx-agentx-api-activity-7364245216851050498-BfRO?utm_source=share&utm_medium=member_desktop&rcm=ACoAADjfvpkBwpCXn_8Pmby7ixjV93Dje5TcmgUHi) 旨在提供一个汇集了最聪明 AI 大脑（**ChatGPT**, **Gemini**, **LLaMA**, **Grok**）的交易台，通过共同辩论直到就最佳行动方案达成一致。
   - 该平台试图通过让 **LLM** 辩论最佳方案，为交易者提供一个可以完全信赖的系统。
- **成员辩论 SFT 与 DPO**: 成员们讨论了 **DPO** (Direct Preference Optimization) 与 **SFT** (Supervised Fine-Tuning) 的有效性，其中一位成员指出 *DPO 与推理没有关系*，但在 **SFT** 之后进行 **DPO** 比单纯使用 **SFT** 效果更好。
   - 讨论涉及利用 **DPO** 提升性能，但其与推理的关系在成员间存在争议。
- **HF Learn 课程受 422 错误困扰**: 一位成员报告 [Hugging Face LLM 课程的一个页面](https://huggingface.co/learn/llm-course/en/chapter12/3a) 宕机并显示 **422 错误**。
   - 用户目前无法访问该 Learn 课程中损坏的页面。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **用户发现利用 Gems 优化播客生成的方法**：用户正在开发工作流（例如[这个示例](https://github.com/para-droid-ai/scratchpad/blob/main/assistant-workflows-tasks-personas/deeper_podcast_synthetic-082025.txt)），通过 **Gems**、**Gemini**、**PPLX** 或 **ChatGPT** 创建更深层次的研究框架来生成播客。
   - 关键在于设置 Prompt 来逐段规划整个转录文本，从而根据较长的 **YouTube** 视频生成播客。
- **自定义界面允许用户配置播客长度**：用户可以通过 **Customize** 选项（三个点）调整 NotebookLM 中的播客长度，将播客时长延长至 **45-60 分钟**。
   - 指定主题可以让 Bot *集中讨论特定话题*，而不是指望它能将所有重要内容都塞进一个播客中。
- **隐私政策引发的担忧依然存在**：用户正在使用 **Gemini** 和 **NotebookLM** 分析医疗保健公司的隐私政策和使用条款。
   - 用户对于*向这些公司泄露了多少信息*感到惊讶，并认为这种方法对于理解**使用条款（Terms of Use）**和**隐私政策（Privacy policies）**非常有用。
- **Android 应用功能同步推迟**：用户要求 NotebookLM Web 端和 **Android 应用**之间实现更多的**功能对等（feature parity）**，特别是学习指南功能。
   - 一位用户表示，目前的原生应用*几乎无法使用*，因为学习指南依赖于笔记功能，而原生应用中缺少该功能。
- **NotebookLM API 仍未发布**：虽然 NotebookLM 的官方 API 尚未提供，但用户建议使用 **Gemini API** 作为替代方案。
   - 另一位用户分享了结合使用 **GPT4-Vision** 和 **NotebookLM** 的策略，以*快速消化带有标注的复杂 PDF 原理图*。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **字节跳动发布长上下文模型**：根据[这张图片](https://cdn.discordapp.com/attachments/1149866623109439599/1407959284280459305/image.png?ex=68a8a883&is=68a75703&hm=b8a5430da1f445204c76334cef07358c8d9b815da989424483a5ecf3bc65c790)，字节跳动发布了一个具有极长上下文的基座模型，该模型没有 **MHLA**，没有 **MoE**，甚至没有 **QK** norm。
   - 该模型的架构被描述为 *vanilla*（原生），人们希望即将发表的论文能提供更多见解。
- **Seed-OSS-36B 缺失 GGUF 版本引发猜测**：用户询问为何 **Seed-OSS-36B** 还没有 **GGUF** 版本，并指出这类版本通常出现得很快。他们引用了[这个链接](https://x.com/adityastomar_/status/1958048129275805867)，质疑这是否对 **ASICs** 有影响。
   - 有人认为延迟可能源于自定义的 **vllm** 实现，由于 `architectures: ["SeedOssForCausalLM"]`，该架构目前不受 **llama.cpp** 支持。
- **Seed 模型采用 Dropout 和 Bias**：**Seed** 模型结合了类似于 **LLaMA** 的自定义 **MLP** 和注意力机制，但具有 Dropout、输出 Bias 项以及 **qkv** 头的 Bias 项。
   - 这些添加项被推测用作正则化技术；然而，该模型经过了多少个 Epoch 仍不得而知，且已确认仅将其重命名为 **LLaMA** 是无法运行的。
- **Qwen 通过 RoPE 扩展至 512k 上下文**：根据 [Hugging Face 数据集](https://huggingface.co/datasets/eaddario/imatrix-calibration)，**30B** 和 **235B** 的 **Qwen 2507** 模型可以使用 **RoPE** 缩放实现 **512k** 的上下文。
   - 这些数据集用于生成重要性矩阵（**imatrix**），有助于在量化过程中最大限度地减少误差。
- **Cursor 的内核博客赢得赞誉**：成员们分享了 [Cursor 内核博客](https://x.com/stuart_sul/status/1957927497351467372)的链接。
   - 许多人一致认为 Cursor 在这方面做得非常出色（*cursor cooked*）。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **DeepSeek V3.1 首次亮相，改进幅度较小**：新的 **DeepSeek V3.1** 模型已发布，部分成员指出这更像是一种*渐进式改进*，并伴随一些性能退化，参考 [DeepSeek 官方页面](https://huggingface.co/deepseek-ai/DeepSeek-V3.1)。
   - 社区正在密切关注其性能，以寻找细微的提升和潜在的缺陷。
- **DeepSeek 寻求 Anthropic API 集成**：**DeepSeek** 现在支持 **Anthropic API**，扩展了其功能和覆盖范围，正如 [X 平台](https://x.com/deepseek_ai/status/1958417062008918312)上宣布的那样。
   - 这一集成使用户能够在 **Anthropic** 生态系统中使用 **DeepSeek**，为 AI 解决方案开发提供了灵活性。
- **R-Zero LLM 无需人类数据即可进化**：一份关于 **R-Zero** 的综合研究报告在 [PDF](https://cdn.discordapp.com/attachments/1371757564005711973/1408153973751545896/R-Zero__A_Comprehensive_Study_of_a_Self-Evolving_LLM_Training_Method.pdf?ex=68a8b515&is=68a76395&hm=dcba93436f636eeec364d08d08e1603131147faaa595637065f2e226772005f2&) 中分享，这是一种从零人类数据开始并独立改进的自进化 **LLM 训练方法**。
   - 该方法标志着与传统 **LLM 训练**的背离，有可能减少对人类标注数据集的依赖。
- **中国避开了数据中心能源困境**：一位成员指出，在中国，*能源供应被视为理所当然*，这与美国关于数据中心能耗和电网限制的争论形成鲜明对比，参考了[这篇 Fortune 文章](https://fortune.com/2025/08/14/data-centers-china-grid-us-infrastructure/)。
   - 这种方法的差异可能会使中国 AI 公司在扩展能源密集型模型方面具有竞争优势。
- **Kimi K2 期待更好的图像生成能力**：一位成员指出，如果 **Kimi K2** 能结合**比 GPT-5 更好的图像生成能力**，将会更加强大（OP），并分享了[这个 Reddit 链接](https://www.reddit.com/r/ChatGPT/s/vUrGedSwY5)。
   - 集成增强的图像生成功能将使 **Kimi K2** 成为一个更全能、更具竞争力的 AI 助手。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 2.5 Pro 遇挫，而 Flash 表现出色**：一位用户报告称 **Gemini 2.5 Flash** 功能正常，而 **Gemini 2.5 Pro** 持续失败，不过在配置计费后 `gemini/gemini-2.5-pro-preview-06-05` 可以运行。
   - 另一位用户报告因 **qwen-cli** 进程被收取了 **$25** 费用并正在申请退款，这凸显了模型性能和计费方面潜在的不一致性。
- **用户遭遇意外的 Qwen CLI 扣费**：一位用户在 Google OAuth 认证后使用 **qwen-cli** 产生了 **$25** 的费用，而他原本预期使用的是来自阿里云的免费额度。
   - 他们提交了支持工单，引用了控制台显示的“*一次 $23 的调用且无输出*”的使用记录来对这笔意外费用提出申诉。
- **社区对 GPT-5 Mini 模型进行基准测试**：由于全尺寸 **GPT-5** 的速率限制，社区成员正积极对 **gpt-5-mini** 和 **gpt-5-nano** 进行基准测试，一位用户声称 *gpt-5-mini 非常出色且便宜*。
   - 基准测试结果和 **gpt-5-mini** 的 PR 已经发布，反映了社区对评估更小、更易获取的模型的兴趣。
- **DeepSeek v3.1 价格上涨**：从 2025 年 9 月 5 日开始，DeepSeek 将把两个模型的输入价格提高到 **$0.25 vs $0.27**，以匹配 Reasoner 模型的价格。
   - 价格上涨以匹配 **DeepSeek 3.1** 模型反映了定价策略的变化。
- **OpenRouter 需要“思考 (Think)”模式**：用户注意到 **OpenRouter** 缺乏原生的“思考”模式来增强推理，但可以通过命令行启用：`aider --model openrouter/deepseek/deepseek-chat-v3.1 --reasoning-effort high`。
   - 社区成员建议更新模型配置以解决这一功能缺失。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Marimo Notebooks 崛起，成为 Jupyter 的替代方案**：一位成员发布了[关于 **marimo notebooks** 的教程](https://www.youtube.com/watch?v=2aepn9uRVOM)，强调了它在 **Graph RAG with DSPy** 想法迭代中的应用，它能同时作为 Notebook、脚本和应用运行。
   - 接下来的视频将深入探讨 **DSPy modules** 的优化，目前的教程主要向新用户介绍 **marimo**。
- **可读性辩论：DSPy 代码先遭质疑后获支持**：在一位成员驳斥了 **IBM AutoPDL** 关于不可读性的指控后，他们辩护称 **DSPy 的代码**和 **prompts** 具有极高的人类可读性和清晰度。
   - 辩护者强调了代码的易用性，使其易于理解和操作。
- **GEPA 登陆 DSPy v3.0.1**：成员们确认 **GEPA** 已在 **dspy** 版本 **3.0.1** 中可用，如附带的[截图](https://cdn.discordapp.com/attachments/1161519469319946286/1407936409615990904/image.png?ex=68a89336&is=68a741b6&hm=72219936a525599fc3faca9d127d106a09e0639e9eeae1564a8bbfc196b07ffa&)所示。
   - 在微调过程中，一位成员询问在 **dspy.InputField()** 和 **dspy.OutputField()** 中使用“原生描述 (vanilla descriptions)”是否常见，以便让优化器自由思考。
- **Pickle 问题：DSPy 程序未保存**：一位用户报告了保存优化程序时的问题，指出即使使用了 `optimized_agent.save("./optimized_2", save_program=True)`，元数据也仅包含依赖版本而没有程序本身。
   - 当另一位用户将 **GEPA** 的最大上下文长度设置为 **32k** 但仍收到截断的响应时，成员们讨论了长推理的复杂性以及多模态设置中可能存在的问题。
- **RAG vs 拼接：百万级文档辩论**：成员们辩论了对于处理税法或农作物保险文档等任务，**RAG** (Retrieval-Augmented Generation) 还是简单的**拼接 (concatenation)** 更合适。
   - 辩论承认，虽然 **RAG** 常被视为大材小用，但数百万份文档的规模有时可以证明其使用的合理性。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command A Reasoning 发布**：Cohere 推出了专为企业设计的 **Command A Reasoning**，在 Agent 和多语言基准测试中表现优于其他模型；可通过 [Cohere Platform](https://dashboard.cohere.com/playground/chat?model=command-a-reasoning-08-2025) 和 [Hugging Face](https://huggingface.co/CohereLabs/command-a-reasoning-08-2025) 获取。
   - 根据 [Cohere 博客](https://cohere.com/blog/command-a-reasoning)，它可以在单张 **H100** 或 **A100** 上运行，上下文长度为 **128k**，在多 GPU 上可扩展至 **256k**。
- **Command 的 Token 预算功能解决难题**：**Command A Reasoning** 具有 **token budget** 设置，能够直接管理计算使用量并控制成本，从而无需区分推理模型和非推理模型。
   - 它也是驱动 **North**（Cohere 的安全 Agentic AI 平台）的核心生成模型，支持自定义 AI Agent 和本地自动化。
- **Command-a-03-2025 间歇性返回引用**：`command-a-03-2025` 仅间歇性地返回引用 (citations)，即使将 maxTokens 设置为 8K 也是如此，这在生产环境中引发了信任问题。
   - 一位 Cohere 成员澄清说，它对引用使用“快速 (fast)”模式（根据 [API 参考](https://docs.cohere.com/reference/chat#request.body.citation_options.mode)），且不保证一定提供引用；建议改用 **command-a-reasoning**。
- **正在开发中的 Langchain RAG**：一位成员正在学习 Langchain 以构建 RAG (Retrieval-Augmented Generation) 应用，并打算使用 **command-a-reasoning**。
   - 他们期待 **command-a-omni** 的发布，并对未来名为 **Command Raz** 的模型表示期待。



---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP 客户端无视指令字段**：成员们反映 **MCP 客户端**（特别是 **Claude**）正在忽略**指令字段（instructions field）**，而仅考虑**工具描述（tool descriptions）**。
   - 一位成员建议，*添加指令、上下文然后重复指令会产生更好的效果*，但这在集成 API 中无法实现；另一位成员则建议 **MCP server** 应该优先处理**工具描述**。
- **多样化的 MCP Server 投入使用**：成员们正在分享他们首选的 **MCP server** 配置和工具，包括用于版本控制的 GitHub、用于后端开发的 Python 和 FastAPI，以及用于机器学习的 PyTorch。
   - 一位用户寻求关于如何让 Agent 遵循特定的 **generate_test_prompt.md** 文件的建议，并链接了其配置的[截图](https://cdn.discordapp.com/attachments/1312302100125843476/1408171379236409354/Screenshot_3.png?ex=68a8c54b&is=68a773cb&hm=1fbd862963889b97ef764b1599c2960340dab7ce357b3717f577e2b1491ffdc2)。
- **Web-curl 释放 LLM Agent 威力**：**Web-curl** 是一个使用 Node.js 和 TypeScript 构建的开源 **MCP server**，它赋予 LLM Agent 获取、探索以及与网页和 API 交互的能力，源代码托管在 [GitHub](https://github.com/rayss868/MCP-Web-Curl) 上。
   - 在功能上，**Web-curl** 使 LLM Agent 能够以结构化的方式获取、探索并与网页及 API 进行交互。
- **MCP-Boss 实现集中化密钥管理**：一位成员介绍了 **MCP Boss**，用于集中管理密钥，提供单一 URL 来网关化所有服务，具有多用户身份验证以及通过 OAuth2.1 或静态 HTTP header 实现的 MCP 授权功能。
   - 更多信息请访问 [mcp-boss.com](https://mcp-boss.com/)。
- **MCP Gateway 中的 AI 路由能力**：一位成员介绍了一个带有 **AI 驱动路由**功能的轻量级网关，旨在解决 Agent 需要知道哪个特定服务器拥有正确工具的问题，代码已发布在 [GitHub](https://github.com/oliverye7/mcp-gateway) 上。
   - 通过使用该网关，可以利用 AI 来解决 **MCP 路由**问题。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 庆祝 Modverse 里程碑**：Modular 发布了 [Modverse #50](https://www.modular.com/blog/modverse-50)，并宣布了一个自定义服务器标签，如[截图](https://cdn.discordapp.com/attachments/1098713601386233997/1408199603861323878/Screenshot_2025-08-21_at_5.22.15_PM.png?ex=68a8df94&is=68a78e14&hm=2991584cc0b81449dbc278d1d8302e55aabc54c58a6620e7041deb9dbd20e951&)所示。
   - 该自定义服务器标签已部署。
- **kgen 和 pop 饱受文档匮乏困扰**：成员们反映 **kgen** 和 **pop** 缺乏文档，特别是关于操作和参数方面，其中一人表示*目前还没有关于内部 MLIR dialects 的全面文档*。
   - 共享了 GitHub 上 [pop_dialect.md](https://github.com/modular/modular/blob/main/mojo/stdlib/docs/internal/pop_dialect.md) 的链接，并澄清这些是 stdlib 与编译器之间契约的一部分，*因此在 stdlib 之外使用它们需自担风险*。
- **POP Union 面临对齐问题质疑**：由于在使用 `sizeof` 时出现了意料之外的大小差异，人们对 **pop.union** 的对齐（alignment）Bug 产生了怀疑。
   - 一位成员在 GitHub 上创建了 [issue 5202](https://github.com/modular/modular/issues/5202) 以调查 **pop.union** 中疑似存在的对齐 Bug，同时观察到 **pop.union** 似乎没有在任何地方被使用。
- **TextGenerationPipeline Execute 方法位置明确**：一位成员找到了 `TextGenerationPipeline` 上的 `execute` 方法，并链接到了 [Modular 仓库中的相关代码行](https://github.com/modular/modular/blob/fe7ba5d5f8b08ccba7356c45b90afcf495421469/max/pipelines/lib/pipeline.py#L977)。
   - 他们建议检查 MAX 版本。
- **内存分配器备受关注**：一位成员建议，在将内存分配器（memory allocators）集成到语言中之前，可能需要健壮的分配器支持，因为大多数用户不想手动处理内存不足（**OOM**）错误。
   - 这些评论是在讨论其他困难的背景下提出的，其中一位成员报告在创建自定义推理循环（inference loop）时，难以在获取下一个 token 的同时检索 **logits**，并链接了一个 [Google Docs 文档](https://docs.google.com/document/d/1Hd6xZnf0bmg9SMQU1h10cd4Cwd9HDOPzPHZNqiPnrpg/edit?tab=t.0)以提供上下文。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 首次推出企业级文档 AI**：LlamaIndex 的产品副总裁将在 **PST 时间 9 月 30 日上午 9 点** 预告关于[文档](https://t.co/x70xjEQaFs)解析、提取和索引的企业级经验。
   - 重点在于 LlamaIndex 如何解决现实世界中的文档挑战。
- **vibe-llama CLI 工具配置编码 Agent**：LlamaIndex 推出了 **vibe-llama**，这是一个 CLI 工具，可为 **LlamaIndex 框架**和 **LlamaCloud** 自动配置带有上下文和最佳实践的编码 Agent，详情见[此处](https://t.co/G1gINq9kge)。
   - 目标是简化开发工作流程。
- **CrossEncoder 类：核心库 vs 集成库**：一名成员询问了 `llama-index` 中重复的 **CrossEncoder 类**实现，具体位于 `.core` 和 `.integrations` 下（[代码链接](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/postprocessor/sbert_rerank.py)）。
   - 官方澄清 `.core` 版本是 v0.10.x 迁移的残留，建议通过 `pip install llama-index-postprocessor-sbert-rerank` 使用 `llama_index.postprocessor.sbert_rerank`。
- **寻求 Agent 创建网关**：一名成员正在寻找现有的 **gateway** 项目，该项目能将 **model, memory, and tools** 结合在一起，并暴露一个 **OpenAI 兼容端点**。
   - 他们希望在 Agent 探索中避免重复造轮子。
- **AI 安全调查收集社区意见**：一名成员分享了一份 [AI 安全调查](https://mukullight.pythonanywhere.com/form)，以收集社区对重要 **AI 安全问题**的看法。
   - 该调查旨在了解 **AI 安全社区**最感兴趣的内容。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **用户报告缺少积分购买选项**：成员报告购买额外积分的选项消失了，用户只能看到“升级套餐”选项。
   - 已确认该选项目前处于**下线状态**。
- **支持工单无人回应**：一名用户报告了一个任务问题并创建了工单 **#1318**，但尚未收到回复或获得工单访问权限。
   - 他们请求团队协助，并标记了一名特定成员。
- **比赛获胜者引发操纵指控**：一名用户指责比赛的第二名获胜者“不配获胜”，并声称比赛“似乎被操纵了”。
   - 目前尚未提供进一步的证据或细节来支持这一说法。
- **每日免费积分已停止？**：一名回归用户注意到他们没有收到往常的 **300 每日免费积分**。
   - 他们询问 Manus 是否已停止提供这些积分。
- **推荐积分代码困惑**：一名用户询问如何领取推荐积分，并指出系统要求输入代码。
   - 该用户表示不知道在哪里可以找到所需的代码。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **探索 Overworld 常量折叠 (Const Folding)**：一名成员在[此 Discord 线程](https://discord.com/channels/1068976834382925865/1255400554012741683/1407782654958506004)中探索了 **overworld const folding** 和潜在的 **view(const) 重构**，重新定义了 `UPat.cvar` 和 `UPat.const_like` 以匹配 `CONST` 和 `VIEW(CONST)`。
   - 目标是折叠像 `x * 0` 这样的表达式，但人们对符号计算中有效性和 `.base` 扩散表示担忧。
- **ALU View Pushing 作为替代方案**：有人建议了一种替代方法，即在 kernelize 中添加一个 upat，将 view 直接推送到 **ALU** 上，模仿 **S-Lykles 的方法**。
   - 鉴于 `* 0` 在计算上的无关性，这种方法和针对 `x * 0` 的特殊规则将允许未经修改的符号匹配。
- **主张移除 base**：一名成员强烈建议不要采用提议的方法，认为它“非常丑陋”，并主张 **移除 `.base`**。
   - 讨论还质疑了在此背景下对 **PAD** 操作的处理。
- **RANGEIFY=1 简化实现**：有人建议设置 **RANGEIFY=1** 可以带来更整洁的实现。
   - 然而，该项目目前正处于旧引擎和 rangeify 共存的过渡阶段，处于一种停滞状态。

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4ALL 免费层级支持私有 AI**：一位用户询问了关于公司如何使用 **GPT4ALL** 以**私密且安全**的方式运行其 **AI 模型**的问题。
   - 另一位成员澄清说，如果公司已经准备好了自己的 **AI 模型**，那么**免费版本**就足够了。
- **用户寻求 LocalDocs 模型推荐**：一位用户正在寻求模型推荐，以便利用 **GPT4All 的 LocalDocs 功能**，从数百篇 **PDF 格式的科学论文**中构建个人知识库。
   - 该用户说明其拥有配备 **24 GB VRAM** 的 **Nvidia RTX 5090** 以及 **64 GB RAM**，并希望所选模型具备**推理能力**。



---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该公会长期保持沉默，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该公会长期保持沉默，请告知我们，我们将将其移除。


---


**Torchtune Discord** 没有新消息。如果该公会长期保持沉默，请告知我们，我们将将其移除。


---


**Codeium (Windsurf) Discord** 没有新消息。如果该公会长期保持沉默，请告知我们，我们将将其移除。


---


**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该公会长期保持沉默，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。


---

# Discord：各频道详细摘要与链接





### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1407801395884720330)** (951 条消息🔥🔥🔥): 

> `nano-banana 模型, Video Arena 问题, DeepSeek V3.1, Gemini 3` 


- **Nano-Banana 的 McLau's Law 揭晓**：一位成员开玩笑说 **Nano-Banana** 产生的结果往往*远低于 nano-banana*，并将这一现象称为“**McLau's Law**”，以此幽默地致敬 **OpenAI** 的一位研究员。
   - 附带了一张[幽默图片](https://cdn.discordapp.com/attachments/1340554757827461211/1407957527987097642/RDT_20250821_0619535468074318472918433.jpg?ex=68a8a6e1&is=68a75561&hm=bfcfd37e574ea905e84f2ff1c9e6ffee1855681ad903050cfe30a367ea5d96f3&)，引发了关于 **AI** 当前能力的讨论。
- **Video Arena 因 Bot 停机而陷入困境**：成员们报告了 **Video Arena** 的问题，提到无法使用命令或生成视频，版主确认了 **Bot** 停机并正在进行修复。
   - 针对视频创建权限的重复查询，得到的解释是 **Bot** 暂时不可用，并引导用户关注公告频道以获取更新。
- **DeepSeek V3.1 进入竞技场**：用户讨论了将 **DeepSeek V3.1** 引入平台的情况，一位用户将新模型描述为 *Gemini 2.5 pro 的略逊版本*。
   - 然而，共识是它作为编程模型具有潜力，但需要进一步提升通用能力。
- **用户声称 Gemini 3 即将到来**：虽然尚未证实，但一位用户暗示 **Gemini 3** 即将发布，推测其发布日期将与 **Google Pixel 发布会**同步，引发了成员们的期待。
   - 该用户未引用任何来源，此说法很快被其他社区成员否定。 
- **站点故障导致聊天记录清空**：用户报告在站点故障后出现了大规模数据丢失，包括聊天记录丢失以及无法接受服务条款，版主对此表示知晓并保证会进行修复。
   - 版主还表示，很快将推出登录功能，以防止此类事件再次发生。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1408069950391980122)** (2 条消息): 

> `Video Arena Bot, Deepseek v3.1, LMArena 模型` 


- ****Video Arena Bot** 停机，频道已锁定**：**Video Arena Bot** 目前无法工作，锁定了对提示词频道 <#1397655695150682194>、<#1400148557427904664> 和 <#1400148597768720384> 的访问。
   - 必须在 **Bot** 在线时才能在这些特定频道中发送提示词。
- ****DeepSeek v3.1** 已添加到 LMArena**：两个新模型已添加到 LMArena：**deepseek-v3.1** 和 **deepseek-v3.1-thinking**。
   - 这些模型现在可以在竞技场中使用。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1407802780516614178)** (887 条消息🔥🔥🔥): 

> `ByteDance Seed Model, GRPO Training, DeepSeek V3.1 Quants, Nvidia's GPUs and Pricing, GLM-4.5 Cline Integration` 


- **ByteDance 发布 Seed-OSS 36B 基础模型**：ByteDance 在 Hugging Face 上发布了 **Seed-OSS-36B-Base-woSyn** 模型，这是一个 **36B** 参数的稠密模型，具有 **512K** 上下文窗口，并明确声称 *没有合成指令数据*，使其成为进一步微调的有趣基础。
   - 成员们表示兴奋，指出它与 **Qwen3** 等模型不同，一些人渴望在数据集完成后尝试用它来微调 GPT-ASS，尽管该模型 *仅* 在 **12T tokens** 上进行了训练。
- **GRPO 训练需要巧妙的数据集设计**：为了将 GRPO 用于多步游戏动作，成员建议设计数据集时为每一步提供独立的 prompt，例如 **[['step1 instruct'], ['step1 instruct', 'step1 output', 'step2 instruct']]**，并实现一个奖励函数来匹配输出。
   - 有人指出 Full PPO 可能更适合游戏，因为 GRPO 主要对 LLM 有效，因为 *它们最初就大致知道该做什么*。
- **DeepSeek V3.1 在思考和非思考模式下横扫排行榜**：**DeepSeek V3.1** 模型表现出极具竞争力的结果，在非思考模式下的 SWE-bench verified 取得了 **66** 分，成员们对此表示期待，并将其与 **GPT5** 的中等推理能力进行比较。
   - 尽管最初备受推崇，但随后的讨论提到了对其在创意写作和角色扮演中表现的担忧，一些人指出 *混合模型在非思考模式下缺乏指令遵循能力和创造力*。
- **Nvidia RTX 5090 价格尘埃落定，引发升级争论**：**RTX 5090** 目前定价在 **$2000** 左右，引发了是否升级的讨论，特别是考虑到其 **VRAM** 能力对训练的帮助，而其他人则建议坚持使用 **3090s** 或等待 **RTX 6000**。
   - 一些成员对 **NVIDIA** 的限制表示沮丧，特别是缺乏 **P2P 或 NVLink**，一位成员开玩笑说：*如果你拥有一块 5090，你肯定会用它玩游戏*。
- **高质量的 Imatrix 校准数据是关键**：成员指出 WikiText-raw 被认为是校准 imatrices 的 *糟糕* 数据集，因为 imatrix 需要充分多样化，并在模型原生的 chat-template 格式示例上进行训练。
   - 相反，[Ed Addorio 最新的校准数据](https://huggingface.co/datasets/eaddario/imatrix-calibration) 包含 Math, Code 和 Language prompts，如果操作得当，可以改善并帮助保留模型对多种语言的理解。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/)** (1 条消息): 

.zackmorris: Hello
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1407836226488111114)** (27 条消息🔥): 

> `GRPO 20mb alloc fail, ChatGPT's deep research, Grok-4, Repetition penalty, RAG` 


- ****GRPO 20MB 分配失败困扰 Gemma 模型！****：一位用户报告在处理 [gemma-3-4b-it-unslop-GRPO-v3](https://huggingface.co/electroglyph/gemma-3-4b-it-unslop-GRPO-v3) 时，使用 **GRPO** 频繁出现 **20MB 分配失败**。
- ****ChatGPT 的深度思考模式提升性能！****：一位用户建议通过启用联网搜索并在 prompt 中添加 *"use deep thought if possible"* 来增强 **ChatGPT** 的表现，即使没有完整的深度研究功能。
- ****Grok-4 表现出色！****：一位用户对 **Grok-4** 印象深刻，暗示他们可能一直在秘密使用 **Grok-4-Heavy**。
- ****重复惩罚引发趣事****：一位用户分享了一张图片，展示了 **repetition penalty** 参数的重要性。
- ****RAG 协助****：一位用户请求在处理 **RAG** 时获得帮助。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1407822574725107743)** (101 条消息🔥🔥): 

> `视网膜照片训练策略、GPT-OSS 20B 在 Sagemaker 上的部署、Unsloth Zoo 问题、使用 Unsloth 加载 GGUF、Gemma 3 Vision Encoder 训练损失` 


- **针对视网膜照片微调 Vision-Text Encoders**：一位用户询问是训练自定义的视网膜照片 Vision-Text Encoder 更好，还是使用 Unsloth 配合主流模型更好，并指出**视网膜照片在训练数据集中代表性不足**。
   - 建议尝试计算机视觉模型、在相似数据集上进行迁移学习以及多模态方法，并利用 Prompt Engineering 和 Personas 生成合成临床笔记。
- **解决 GPT-OSS 20B Sagemaker 部署故障**：用户在 Sagemaker 上部署 **unsloth/gpt-oss-20b-unsloth-bnb-4bit** 时遇到 `ModelError`，收到 **400 错误**和 InternalServerException，消息为 `\u0027gpt_oss\u0027`。
   - 有回复指出该模型无法在 AWS Sagemaker 上运行，建议部署 GGUF 或普通版本，使用 LMI Containers，并引导用户参考 [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference-container-docs.html)。
- **Unsloth Zoo 安装问题**：用户在 Sagemaker 实例中安装 **unsloth-zoo** 后仍遇到导入错误。
   - 用户通过删除所有包，然后重新安装 Unsloth、Unsloth Zoo 以及 JupyterLab 解决了该问题，同时还需要更新 Unsloth 并刷新 Notebook。
- **Apple Silicon Mac 的量化考量**：用户寻求关于哪种 **GGUF 量化**最适合 M 系列 Apple Silicon 的建议，并指出 Mac 针对 **4-bit** 和 **8-bit** 计算进行了优化。
   - 建议用户选择 **Q3_K_XL**，如果显存不足以容纳上下文则选择 **IQ3_XXS**；Q3-4 量化性能表现不错，但如果使用 GGUF，差异并不那么显著。
- **GPT-OSS 通过 LLaVA 获得多模态能力**：用户询问为什么 vision llama13b 的 Notebook 无法用于 gpt-oss-20b，并好奇是否有人成功实现过。
   - 澄清了 GPT-OSS 仅限文本，并非视觉模型，因此无法直接运行；若要添加视觉支持，用户必须像 LLaVA 那样附加自己的 **ViT module**，可以参考 [LLaVA Guides](https://github.com/haotian-liu/LLaVA)。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1407927838123888651)** (11 条消息🔥): 

> `WildChat-4M-English-Semantic-Deduplicated 数据集、Behemoth-R1-123B-v2 模型、GPU Rich 炫耀` 


- **WildChat-4M 英语 Prompt 数据集发布**：**WildChat-4M-English-Semantic-Deduplicated 数据集**已在 Hugging Face 上线，包含来自 WildChat-4M 数据集的英语 Prompt，使用了包括 **Qwen-4B-Embedding** 和 **HNSW** 语义去重在内的多种方法进行去重。
   - 当前版本包含 **<= ~2000 tokens** 的 Prompt，后续将添加更长的 Prompt，更多信息请见[此处](https://huggingface.co/datasets/MasonMac/WildChat-4M-English-Semantic-Deduplicated)。
- **TheDrummer 发布 Behemoth-R1-123B-v2**：由 TheDrummer 创建的 **Behemoth-R1-123B-v2** 模型已发布，详情见[此处](https://huggingface.co/TheDrummer/Behemoth-R1-123B-v2)。
   - 一位成员提到，能在 Hugging Face 中配置自己的硬件真是太疯狂了。
- **GPU Rich 是新的炫耀方式**：一位成员分享了一张图片，描绘了对贫穷的嘲讽，并炫耀了 **GPU Rich**（GPU 富有）。
   - 以 **TFLOPS** 为单位查看 GPU 是一种高端的炫耀。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1407840310024995026)** (7 messages): 

> `Qwen3-4B finetuning, TTS with Gemini 270m, Mixture Models, JetMoE, BAM` 


- ****Unsloth** + **Qwen3-4B**：强强联手？**：一位成员正在使用 **Unsloth** 对 **Qwen3-4B** 进行微调，并将在完成后分享包括评估在内的结果；目前微调进展顺利。
   - 另一位成员祝其好运！
- **从零开始训练模型**：一位成员完成了从零开始训练概念验证（POC）模型的 **22%**，使用的是自行构建的 6 年级数学数据集，包含 **500k** 样本数据。
   - 如果成功，他们将把数据集扩展到其他学科。
- **使用 Gemini 270M 实现文本转语音（TTS）的构想**：一位成员想尝试使用 **Gemini 270m** 实现 **TTS** 概念，并希望在月底前开始。
   - 他们的灵感来自混合模型（Mixture Model）的相关论文。
- **专家讨论合并模型在 HumanEval 上的弱点**：一位成员引用了关于从零训练的混合模型的 [JetMoE 论文](https://arxiv.org/pdf/2404.07413#page=9.56)，指出尽管它们在其他方面的表现优于基准模型，但在 **HumanEval** 上的表现较差。
   - 他们还提到了 [BAM](https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fpdf%2F2408.08274)，其中预训练模型被复制并在不同领域进行训练后合并，同样在编程能力上损失了百分点。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1408170025436844156)** (1 messages): 

> `Cloudflare outage, Generations API stability` 


- **Generations API 受 Cloudflare 故障影响**：由于上游基础设施提供商的问题，**Generations API 端点**经历了暂时性中断，导致部分调用出现 **404 错误**。
   - 公告指出，该问题与 **Cloudflare** 的间歇性故障有关，但 **Generations API** 现已恢复到健康状态。
- **可重试的恢复**：对该端点的调用可能会出现 **404**，但应该 **很快就可以重试**。
   - 公告向用户保证服务将很快恢复，并建议他们重试任何失败的调用。


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1408135423468765276)** (4 messages): 

> `OpenRouter Cost Dashboard, Average Request Size, Gemini Input Token Calculation` 


- ****费用报告实现可视化！****：一位成员开发了一个免费的仪表板，用于可视化来自 [OpenRouter](https://openrouter.ai/) 的 `.csv` 费用报告，旨在分析共享账户的数据。
   - 该仪表板可在 [openroutercosts.lorenzozane.com](https://openroutercosts.lorenzozane.com/) 访问，计划包含额外的 **KPI** 和增强图表，欢迎反馈。
- ****仪表板请求增加平均请求大小指标！****：一位成员请求在 OpenRouter 费用仪表板中添加 **平均请求大小** 指标，特别是 **平均输入 Token** 和 **平均输出 Token**。
   - 仪表板开发者承诺将很快添加此功能。
- ****Gemini 输入 Token 触发异常计数！****：仪表板开发者注意到，当输入中包含图像时，**OpenRouter** 对 **Gemini 模型** 的 **输入 Token** 计算似乎会产生异常计数。
   - 他们正考虑就此问题寻求 OpenRouter 团队的澄清，并参考了 [Google AI Developers 论坛](https://discuss.ai.google.dev/t/token-counts-mismatch-9x-discrepancy/72633/2)上的相关讨论。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1407830899223036106)** (528 条消息🔥🔥🔥): 

> `Deepseek pricing, OpenRouter rate limits, Gemini banning, Using OpenRouter with RAG systems, 4.6T parameter model` 


- **Deepseek V3.1 公开发布在即！**：许多用户正迫切等待 **Deepseek v3.1** 的公开发布，对其极度渴望，并预计它将从 9 月开始免费。
- **付费版 Deepseek 提供更快的响应**：用户确认在 OpenRouter 上为 **Deepseek** 模型付费比使用免费模型响应更快。一位用户因为 **Chutes** 导致响应变慢而切换了版本，但免费模型由于不断的速率限制 (rate limits)，用户体验并不理想。
   - 一位用户表示：“自从那个 Chutes 导致响应变慢的事情发生后，我就直接决定付费了。”
- **OpenRouter API Key 易受泄露和利用**：一名用户报告因 OpenRouter API Key 泄露损失了 **$300**，并寻求关于识别未经授权使用来源的建议。但攻击者可能会使用代理来掩盖其原始 IP，用户需对任何泄露的 Key 负责。
- **Gemini 正在进行封禁大清洗吗？**：用户报告 **Gemini** 正在发生大规模封禁，导致许多人寻找替代方案，并回想起由 OpenAI 引发的 AI Dungeon 清洗事件。
   - 一位用户哀叹道：“我们正被送回 2023 年。”
- **OpenRouter API Key 可以用于 RAG 吗？**：用户讨论了在 **RAG 系统** 中使用 **OpenRouter LLM API Key** 的可能性，配合由 Milvus 创建的本地向量数据库。
   - 共识是可行的，但 OpenRouter 并不直接支持 Embeddings，因此你必须使用 Milvus 检索文档，并将其与你的提示词问题一起发送给 OpenRouter LLM API。


  

---


### **OpenRouter (Alex Atallah) ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1407869061840506900)** (3 条消息): 

> `` 


- **Readybot.io 宣布 OpenRouter 新模型**：Readybot.io 宣布了关于 **OpenRouter** 平台上可用**新模型**的更新和信息。
- **OpenRouter 新模型更新**：**OpenRouter** 平台重点介绍了其 **AI 模型** 选择的最新增加和变化，如 Readybot.io 所宣布。


  

---


### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1407806939878129774)** (16 条消息🔥): 

> `Qwen3 coder 480b, DeepSeek v3 0324, Zero return from generative AI, Google Gemini 400 Error, Cohere reasoning model` 


- **LLM 难以正确格式化输出**：用户发现 [像 **Qwen3 coder 480b** 和 **DeepSeek v3 0324** 这样的 LLM](https://link.to.example) 在遵循格式化输出指令方面表现不佳，经常导致 Bug 或忽略提示词。
   - 一位用户发现它们“没用”且“相当令人分心”，经常创建井字游戏网站而不是预期的应用程序。
- **大多数机构在生成式 AI 上看到零回报**：根据 [AFR Chanticleer 的一份报告](https://archive.md/IlP7F)，**95% 的组织在部署生成式 AI 后没有获得任何回报**。
   - 报告指出，这主要集中在部署了**定制化 AI 模型**的公司，关键问题在于公司及其技术供应商没有投入足够的时间来确保其定制化 AI 模型能够持续学习业务中的细微差别。
- **Google Gemini 模型触发 400 错误**：当带有工具调用 (tool calls) 的助手消息使用 **OpenAI 标准的复杂内容格式** `[{"type": "text", "text": "..."}]` 而非简单的字符串格式时，**Google Gemini** 模型会返回 **HTTP 400 错误**。
   - 此问题影响所有 `google/gemini-*` 模型，且仅在消息链中存在工具调用和工具结果时发生。
- **Cohere 发布推理模型**：[Cohere 刚刚发布了一个推理模型](https://cohere.com/blog/command-a-reasoning)，更多细节可在 [Discord](https://discord.com/channels/954421988141711382/996880279224451154/1408103056800874497) 上查看。
   - 目前没有更多细节。
- **功能请求：自动折叠冗长的用户消息**：一位用户请求是否可以在聊天室中自动折叠冗长的用户消息。
   - 该用户称赞了聊天室及其聊天管理功能。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1407803160356982795)** (432 messages🔥🔥🔥): 

> `Claude Cache Reads, Sonic Model origin, Open Sourcing Agentwise, Cursor API costs with Auto agent, DeepSeek V3.1` 


- **缓存问题困扰 Claude**：用户报告称 **Claude** 目前在*缓存读取（cache reads）*方面存在故障，导致与具有可持续缓存机制的 **Auto** 相比成本增加。
   - 一位用户猜测 **Auto** 和 **Claude** 是否秘密地是同一个模型，并将 Token 使用量的减少归因于安慰剂效应。
- **Sonic 进驻 Cursor IDE**：社区正在测试 Cursor 中新的 **Sonic** 模型，一位用户报告称它“非常酷”且速度极快，而另一位用户则认为它适用于新项目，但不适合具有大型代码库的项目。
   - 该模型的来源是一家*隐身公司（stealth company）*，一名成员确认 **Sonic 并非 Grok 模型**。
- **Agentwise 宣布开源**：一名成员宣布开源 **Agentwise**，该项目支持网站副本、图像/文档上传，并支持超过 100 个 Agent，并承诺将提供 [Cursor CLI 支持](https://discord.com/channels/1074847526655643750/1408047562019049523)。
   - 鼓励成员在项目的 Discord 频道中提供反馈。
- **Cursor API 成本说明**：澄清了用户对 Auto agent 成本的困惑，确认在拥有 "pro" 订阅的情况下，**没有额外费用**，不同提供商的 API 使用成本已由订阅费覆盖。
   - 一位用户发现 Auto agent 比 Sonic agent 更好用。
- **DeepSeek V3.1 加入战场**：用户注意到 Cursor 的选项中出现了新的 **DeepSeek V3.1** 模型，但部分用户在连接提供商时遇到困难，其中一人表示*不信任中国 LLMs*。
   - 然而，一名成员报告称 DeepSeek V3.1 在 **TypeScript** 和 **JavaScript** 方面表现良好，甚至在比 Sonnet 更便宜的情况下表现*出色*。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1407802650908688424)** (11 messages🔥): 

> `Agent Auditing, MySQL Installation in Background Agents, Background Task Errors, Remote IDE connection to Background Agent` 


- **Agent 自我审计修复问题**：一位用户报告称，通过要求 Agent 提交并推送新分支修复了一个问题，并指出这似乎是一个内部反复出现的问题。
   - 另一位用户确认这是一种审计，解释为 Agent 使用 **AI-GPL 许可的审计 PDCA 流程框架**进行自我审计。
- **Agent 中的 MySQL 配置说明**：一位用户询问在后台 Agent 中安装 **MySQL** 的事宜，质疑它是预装的还是像 Codex 一样仅限于 **SQLite**。
   - 另一位用户澄清说，默认情况下未安装 **MySQL**，但可以通过 `environment.json` 或 **Dockerfile** 添加到 Agent 的环境中。
- **后台任务（Background Task）错误排查**：一位用户报告称，在启动后台任务后立即持续报错（即使是从 Web 端启动），并提供了一张 [截图](https://cdn.discordapp.com/attachments/1367213641027551352/1408202779096383550/Screenshot_2025-08-21_at_4.34.24_PM.png?ex=68a8e289&is=68a79109&hm=313d4bdb3a6bb89b6beeb5e9ffb22927afd3259ca9dc351a930226cbb122227c&)。
- **远程 IDE（Remote IDE）连接困惑**：一位用户寻求关于将 **远程 IDE** 实例连接到远程机器的明确说明，虽然参考了文档，但发现指令不清晰。
   - 他们质疑是否需要一个虚拟的后台 Agent 来辅助建立此连接。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1407801641675260104)** (141 条消息🔥🔥): 

> `CUDA 错误（4070 TI Super）、LM Studio 多 GPU 性能、SerpAPI 与 LM Studio 集成、GPT-OSS 性能、VRAM 使用的模型参数配置` 


- **修复 4070 检测需要 CUDA 驱动**：一位使用 **4070 TI Super** 的用户报告在 LM Studio 中出现 *"0 GPUs detected with CUDA"* 错误，另一位用户建议通过按 **ctrl+shift+r** 将运行时更改为 **CUDA llama.cpp**，这可能会解决该问题。
- **Flash Attention 加上 KV 量化显著降低 VRAM 占用**：一位成员建议使用命令 `-fa -ub 2048 -ctv q8_0 -ctk q8_0` 来启用 **flash attention**、**KV cache 量化**以及 **2048 的 batch size**。
   - 此外，增加 `-n-cpu-moe` 的值可以管理 VRAM 使用，并指出这仅会影响速度。
- **GPT-OSS 在 Prompt Eval 上完胜 Qwen**：成员们注意到 **GPT-OSS** 在 **3080ti** 上的 prompt eval 达到了 *2k tokens/s*，而 **Qwen** 约为 *1000 tokens/s*。
- **Bolt.new 仅限云端**：一位用户询问如何将 Bolt.new 与 LM Studio 配合使用，但另一位用户澄清 [Bolt 仅限云端](https://github.com/stackblitz-labs/bolt.diy)，不支持本地模型。
- **LM Studio API 调用慢如蜗牛**：一位用户报告 LM Studio API 的调用速度比聊天界面慢得多（30 倍），但随后该问题因不明原因自行解决——此问题可能无法通过配置调整。
   - 他们使用了以下 curl 命令：`curl.exe http://localhost:11434/v1/chat/completions -d {"model":"gpt-oss:20b","messages":[{"role":"system","content":"Why is the sun hot?\n"}]}`


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1407827727985152000)** (54 条消息🔥): 

> `Z390 Designare 对比 Threadripper/Epyc、Qwen3-30B-A3B-Instruct-2507-GGUF 基准测试、Model M 屈伸弹簧键盘、Apple M4 Max 上的 GGUF 对比 MLX、在 Apple M1 上运行 GPT-OSS-20b` 


- **旧款 Z390 Designare 受限于 PCIe 带宽导致性能下降**：在旧款 Z390 Designare 上使用 RTX PRO 6000 时，与 Threadripper 或 Epyc 系统相比，可能会因为有限的 **PCIe 带宽**而经历**轻微的性能下降**。
   - 较旧的主板限制了 PCIe 带宽，从而造成了瓶颈。
- **Qwen3-30B 在 CPU 上达到 10 tok/sec！**：一位用户在 **Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf** 上运行了 [llama-bench](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama-bench)，在纯 CPU 配置下获得了约 **10 tokens per second** 的速度。
   - 性能随线程数而变化，由于扩展性和开销问题，超过一定阈值后收益会递减。
- **Unicomp Model M 屈伸弹簧键盘：依然出色**：用户推荐购买 **Unicomp Model M 屈伸弹簧键盘** 用于快速测试机，并指出 Unicomp 已经获得了生产这些键盘的权利。
   - 一位用户提到他们不得不*寻找一家有库存的英国供应商*。
- **M4 Max 上的 MLX 击败 GGUF**：一位用户在 Apple M4 Max 上对 **GPT-OSS-20b** 进行了基准测试，发现 **MLX (GPU)** 在 **32W** 功耗下达到了 **76.6 t/s (2.39 t/W)**，而 **GGUF (CPU)** 在 **43W** 功耗下仅为 **26.2 t/s (0.61 t/W)**。
   - 测试使用了 **4bit 量化**和 **4k 上下文**，结果显示 MLX 比 GGUF 稍快且能效更高，同时该用户对 GGUF 的性能也留下了深刻印象。
- **GPT-OSS-20b 勉强适配 Apple M1**：用户讨论了在拥有 16GB 内存的 Apple M1 上运行 **GPT-OSS-20b** 的挑战，指出它大约需要 **32GB RAM**。
   - 一位用户建议尝试 [Hugging Face 上的 4-bit MLX 版本](https://huggingface.co/InferenceIllusionist/gpt-oss-20b-MLX-4bit)，并指出*它只能勉强装下*。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1407807741900230718)** (167 条消息🔥🔥): 

> `机器对机器经济 (M2M Economies), AI 安全保障, 去中心化 AI 项目, 大规模 Prompt 的 Few-shot 示例, GPT-5 的直接回答` 


- **Bot 介入 M2M 经济**：成员们讨论了 AI Agent 或 Bot 如何自主交换价值或服务，从而切入 **机器对机器 (M2M) 经济** 的概念。
   - 最困难的部分包括 *Bot 之间的身份与信任、智能合约逻辑、支付基础设施、自主性与安全性，以及法律和伦理挑战。*
- **智能防护措施可加速 AI 采用**：成员们讨论了如 **支出上限、审计日志和保险** 等防护措施，这些措施可能会加速能够进行价值交易的 AI Agent 的普及。
   - 然而，普遍观点认为，尽管有这些防护措施，*真正的信任建立仍需时日。*
- **征集开源去中心化 AI 项目**：一位成员询问为什么还没有建立 **去中心化 AI BOINC 风格的项目**，并提到 [Petals network](https://petals.ml/) 在贡献和保持模型更新方面存在问题。
   - 有建议认为 **经济激励** 或 **活动驱动的激励** 可能会有所帮助。
- **深入探讨大规模 Prompt 的 Few-shot 示例**：一位成员询问在针对具有复杂逻辑的健身工作室的 **29,000 token Prompt** 中使用 **Few-shot 示例** 的最佳实践。
   - 建议包括直接在 Prompt 中提供示例，并将 Prompt 拆分为更小的块，以测试单个组件的性能。
- **GPT-5 的直接回答引发挫败感**：一位用户抱怨 **GPT-5** 的“思考 (thinking)”模式给出的回答非常直接且 **质量极低**，仿佛退回到了旧的模型版本。
   - 另一位成员建议该用户可能达到了 *思考配额限制 (thinking quota limit)，并且设置了回退 (fallback) 而不是置灰不可用？*


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1407853430252376064)** (9 条消息🔥): 

> `GPT Projects UI 文件, AI 法庭法律案例, 使用 GPT 进行 Android 应用开发, 上传内容的 Token 使用情况, GPT 服务器问题` 


- **GPT Projects UI 文件上传**：一位用户正在寻求关于上传到 **Projects UI** 的文件如何运作的确切信息，并指出 **ChatGPT** 告知他们 *Project Files 中的 PDF 目前无法进行搜索或检索*。
   - Bot 指出目前唯一激活的连接器是用于会议记录的 **recording_knowledge**，且不支持 **source_filter**。
- **GPT 模拟法庭：AI 法律专家立场坚定**：一位用户模拟了一个 **AI 法庭法律案例**，发现 **GPT-5** 坚持自己的条款，而不是接受基于现实世界 TRAIGA 法律的法律规则。
   - 在面对 *每周 9 亿用户不可能都在幻觉，称你为退化而非真正的更新* 这一说法时，AI 表示接受 *保持现状会更好*。
- **Token 使用成本曝光**：一位用户发现即使是上传的内容（如 **PDF 页面**）也会计入 Token 使用量。
   - 他们指出 *196k Token 大约相当于 300 页 PDF 的用户上下文*，并强调在考虑上下文时，甚至问题和 GPT 的回复都会消耗 Token。
- **Android 应用末日：GPT 的 APK 梦想破灭**：一位用户在尝试将 **Canvas** 应用转换为 Android 就绪版本时遇到了困难，询问 **GPT** 是否能构建 **Android 应用** 并通过 **Android Studio** 生成 **APK**。
   - 修复一个问题后又会出现另一个问题，得出的结论是 *它还没准备好进行应用开发*，尽管 Bot 在一天后建议将 PWA 或 JSX 文件封装在 APK 外壳中。
- **GPT 服务器在追踪中途崩溃**：一位用户在追踪每日数据时遇到了 **服务器问题**，该问题从前一天晚上就开始了。
   - 其他人评论说，这些工具让编码变得更 *简单*，但它们不会为你完成所有工作。你必须具备一定程度的编程知识。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1407886141813821440)** (6 条消息): 

> `AI 测验生成, GPT 模型罢工` 


- **AI 测验生成明显的错误答案**：一位成员尝试使用 AI 生成测验，但面临 AI 提供 *极其明显* 的错误答案作为选项的问题。
   - 另一位成员建议确保 *所有选项必须具有合理性 (plausible)*。
- **LLM 可能会随机罢工**：一位成员询问如何防止 **GPT 模型** 在推理一段时间后随机停止。
   - 另一位成员回答说，减少棘手的查询以及关于其自身推理的查询会有所帮助，但归根结底 **LLM** 是 *随机性的 (stochastic)*，没有保证能阻止它们以特定方式响应的方法。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1407886141813821440)** (6 messages): 

> `AI Generated Quizzes, GPT-5 Random Quitting, Plausible Response Options, LLM Stochasticity` 


- **AI 测验生成器使选项变得琐碎**：一位成员正苦于 AI 测验生成器产生明显错误的答案选项，例如在多选题中出现 *1029384*。
   - 另一位成员建议确保 *所有响应选项必须具有合理性 (plausible)*，以避免此类问题。
- **GPT-5 意外退出**：一位用户询问是否有办法防止 **GPT-5** 在推理一段时间后随机退出。
   - 一位成员回应称，虽然有一些方法可以降低频率，例如避免处理棘手的查询或关于其自身推理的问题，但由于 **LLM 的随机性 (stochastic nature)**，完全消除这种情况是不可能的。
- **LLM 具有随机性，需要 Guardrails**：由于 Large Language Models 的随机性，*实际上无法阻止它们在足够大的样本量中至少出现一次以任何特定方式进行响应的情况。*
   - 由于 LLM 的非确定性本质，Guardrails 是必不可少的。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1407813276863168583)** (96 messages🔥🔥): 

> `PileT5-XL embeddings as instructions, Networks that process in latent space, Multimodal generative models, image editing models, Latent space editing` 


- **PileT5-XL 嵌入蕴含大量信息**：来自 **PileT5-XL** 的 Embedding Tensor 既可以作为 **pile-t5-xl-flan**（生成文本）的指令，也可以作为 **AuraFlow**（生成图像）的 Prompt，这表明这些 Embedding 像语言中的单词一样具有含义。
   - 一位成员对如何将黑狗图片的 Textual Inversion 应用于 **AuraFlow** 并作用于 **pile-t5-xl-flan** 感兴趣，想知道 **pile-t5-xl-flan** 生成的文本是否会将狗描述为黑色。
- **深入探索 Latent Space**：一位成员有兴趣探索在 Latent Space 中进行处理，并仅在必要时以模块化方式转换为文本/图像/音频的网络。
   - 有人指出，这个想法与人们构建多模态生成模型和 VQGAN-CLIP 的方式相似，并指出让不同的 AI 研究人员 *同意使用相同的 Latent Space* 是一个挑战。
- **精细化图像编辑**：讨论围绕专门为图像编辑设计的模型展开，例如 FLUX.kontext，以及它们是否编辑 Conditioning Latent 并在同一空间中输出新的 Conditioning Latent。
   - 一种方法是获取一堆包含鸟的图像，将鸟编辑掉，然后将两者都通过 Encoder 运行，最后平均它们之间的差异以获得 *Latent Space 鸟类* 向量。
- **针对 Transformer 的 Tuned Lens 研究**：关于 **Tuned Lens** ([https://arxiv.org/abs/2303.08112](https://arxiv.org/abs/2303.08112)) 的工作从 Transformer 中提取了 *模型在第 k 层后的最佳猜测*，这反驳了关于 Decoder Transformer 中 Latent Space 处理的一些假设。
   - 还提到了关于从图像空间到文本空间的线性映射 ([https://arxiv.org/abs/2209.15162](https://arxiv.org/abs/2209.15162)) 的进一步研究。
- **解码音频的秘密**：一个备受关注的模型是 Decoder-only 音频模型 ([https://huggingface.co/hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M))，它可能为训练开启新的可能性。
   - 有人指出，预训练期间看到的音频数据量从 1 分钟到 100 小时不等，也许你可以用 0 分钟的音频进行训练？


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1407829390640939050)** (54 messages🔥): 

> `SSL objectives, Medical event pretraining, Noise-data trajectories, ByteDance's Prover, Unfriendly Activation Steering` 


- **SSL 目标与最大编码率（Maximal Coding Rate）相关研究**: 一位成员将近期关于 **SSL objectives** 的观点与 [maximal coding rate](https://arxiv.org/abs/2005.10242)、[contrastive learning](https://arxiv.org/abs/2406.10743) 以及 [neural collapse](https://arxiv.org/abs/2303.06484) 联系起来。
- **字节跳动的 SEED Prover 获得银牌成绩**: **Bytedance's SEED Prover** 在 [IMO 2025 中获得了银牌成绩](https://seed.bytedance.com/en/blog/bytedance-seed-prover-achieves-silver-medal-score-in-imo-2025)，但目前尚不清楚这如何转化为现实世界的数学问题解决能力。
- **生成式医疗事件模型的 Scaling Laws**: **Cosmos Medical Event Transformer (CoMET)** 模型系列是基于 **1.18 亿患者**（代表 **1150 亿个离散医疗事件**，共 1510 亿个 tokens）预训练的 decoder-only transformer 模型。研究发现，这些模型在相关任务上的表现通常优于或等同于特定任务的监督模型。
   - 这项在 [Generative Medical Event Models Improve with Scale](https://arxiv.org/abs/2508.12104) 中讨论的研究使用了 **Epic Cosmos** 数据集，该数据集包含来自 **310 个医疗系统**、**3 亿条唯一患者记录**中 **163 亿次就诊**的去标识化纵向健康记录。
- **可视化噪声-数据轨迹（Noise-Data Trajectories）**: 成员们讨论了可视化 Flow model 中 **noise-data trajectories** 的方法，包括在预计算的中间体上使用 **UMAP**，但发现其信息量不足。
   - 假设存在不同的轨迹簇，他们希望有一种方法能将这些轨迹挑选出来并单独观察，并确定完全不同类型的输入或两种不同形式的 conditioning 是否遵循 *相同的* 轨迹。
- **训练期间的不友好激活引导（Unfriendly Activation Steering）**: 一位成员提到在训练期间使用 **unfriendly activation steering** 来影响模型权重的工作，并附上了相关 [tweet](https://fxtwitter.com/Dorialexander/status/1958269223320613241) 的链接。


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1407853408211177494)** (1 messages): 

> `Model Overtraining, Token Repetition in Models` 


- **在 Chinchilla 之后继续过度训练模型！**: 即使遵循 **Chinchilla** scaling laws，你仍然应该 **overtrain 你的模型**。
   - 显然，*即使是重复 token 也不是坏事*。
- **Token 重复可能无害**: 在训练期间重复 token 可能并不像以前认为的那样有害。
   - 持续训练带来的收益似乎超过了 token 重复带来的潜在弊端。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1407804201567912107)** (11 messages🔥): 

> `Qwen3 Training, Weight lifting from llama series, Head isolation` 


- **Qwen3：从零训练还是借鉴了 Llama？**: 一位成员询问 **Qwen3** 是从零开始训练（scratch-trained）的，还是从 **Llama** 系列中提取了权重（weights lifted）。
   - 另一位成员指出，相似的训练数据混合比例可能会导致相似的结果。
- **发现相同的 Head！**: 一位成员发现并隔离了一种特定的 *head*，发现 **Llama 3.2-1b instruct** 和 **Qwen3-4B-Instruct-2507** 之间的解码结果向量在不同输出中表现出显著的相似性。
   - 该成员表示，*这两个 head 似乎促进的内容非常相似*。
- **方法论论文发布**: 一位成员分享了[一篇论文](https://arxiv.org/abs/2502.12292)，详细介绍了一种确定 **Qwen3** 是否从零开始训练的方法。
   - 另一位成员称该用户是“简直是神在降下恩赐”。
- **潜意识学习（Subliminal Learning）案例**: 一位成员分享了[一篇论文](https://aclanthology.org/2025.acl-long.407.pdf)，将其视为 *subliminal learning 的典型案例*。
   - 另一位成员对此分享表示感谢。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1407927947200827462)** (2 messages): 

> `Muon Support, Slurm Script for NeoX Job with Docker` 


- **寻求 Muon 支持**: 一位成员表示有兴趣添加 **muon 支持**，并提到了潜在的 **kernel 优化机会**。
   - 他们认为一旦实现了基础支持，就有协作进行这些优化的空间。
- **索取 NeoX Docker 任务的 Slurm 脚本**: 一位成员请求一个使用 **Docker** 启动 **NeoX 任务** 的 **Slurm 脚本** 示例。
   - 拥有一个参考模板对他们来说非常有价值。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1407805054215262350)** (83 条消息🔥🔥): 

> `Meta AI 重组, GPT-5-pro 截断, 银行柜员轮换启发 Dropout, Meta AI 招聘冻结, 字节跳动 Seed-OSS LLMs` 


- **Wang 晋升后 Meta 拆分为四个团队**：据 [Business Insider](https://www.businessinsider.com/meta-ai-superintelligence-labs-reorg-alexandr-wang-memo-2025-8) 报道，Meta 正在将其 AI 工作重组为新任 MSL 负责人 **Alexandr Wang** 领导下的**四个团队**（TBD Lab, FAIR, Product/Applied Research, Infra），同时 **AGI Foundations** 组将被解散。
   - **Nat Friedman** 和 **Yann LeCun** 现在向 Wang 汇报，**FAIR** 将直接支持模型训练，并且正在考虑开发一个 "omni" 模型。
- **GPT-5-pro 迅速截断提示词**：据[此报告](https://x.com/pvncher/status/1958193631250072024?s=46)显示，**GPT-5-pro** 会在没有任何警告或错误消息的情况下，静默截断超过 **60k tokens** 的提示词，这使得大型代码库的提示词变得不可靠。
   - 一些用户还反映 **Cursor** 中的 **GPT-5** 表现得比平时笨得多，有人怀疑正在发生负载卸载 (load shedding)。
- **银行柜员 Dropout！**：一条疯传的推文声称 **Geoffrey Hinton** 在注意到**银行柜员轮换**可以防止勾结后构思了 *dropout* ([来源](https://x.com/eigenron/status/1958181550987632927?s=46))。
   - 反应从对这种偶然洞察力的钦佩，到对从家庭派对中产生注意力机制 (attention mechanisms) 的怀疑和调侃。
- **字节跳动播种新 LLMs**：字节跳动的 Seed 团队宣布了 **Seed-OSS**，这是一个新的开源大语言模型系列，可在 [GitHub](https://github.com/orgs/bytedance/repositories) 和 [Hugging Face](https://huggingface.co/models) 上获取。
   - 该团队邀请社区对模型、代码和权重进行测试并提供反馈。
- **OpenAI 觊觎 AWS 宝座**：OpenAI 的 CFO 表示，公司计划在“未来”出租算力，目标是像一个微型 AWS 那样运营 ([来源](https://x.com/ns123abc/status/1958268338582265948?s=46))。
   - 反应从对 OpenAI 所谓算力短缺的怀疑，到对利润模式转变以及与 Google 和 Microsoft 等现有超大规模云服务商 (hyperscalers) 冲突的分析。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1407823946979741806)** (13 条消息🔥): 

> `Wonda AI, 亿万富翁搏击俱乐部, Qwen 图像编辑` 


- **Wonda AI Agent 承诺带来革命**：Dimi Nikolaou 介绍了 **Wonda**，这是一个旨在彻底改变视频/音频创作的 AI Agent，称其为“Lovable 为网站做了什么，Wonda 就为内容创作做什么” ([推文链接](https://xcancel.com/dimireadsthings/status/1957805267799740571))。
   - 此次发布引发了对预告媒体质量的热烈反应，通过候补名单授予的早期访问权限将在大约 **3 周**内发放邀请。
- **黑客帝国重制版中的扎克伯格 vs 奥特曼**：AIST 发布了 [《亿万富翁搏击俱乐部 第二卷》](https://xcancel.com/aist_digital/status/1954905895025942918?s=46)，这是一部使用 AI 制作的短片，重现了 **Mark Zuckerberg** (Neo) 与 **Sam Altman** (Agent Smith) 之间的《黑客帝国》式对决。
   - 该视频获得了积极反馈，促使 AIST 鼓励观众艾特 Sam 和 Zuck，敦促他们转发该片以获得更广泛的曝光。
- **Qwen 图像编辑取得成功**：Luis C 展示了使用 **qwen-image-edit** 将两张不同的图像合成一张女人抱着娃娃的照片的成功案例 ([推文链接](https://xcancel.com/lucataco93/status/1958581409141944635))。
   - 作为回应，Jay Sensei 声称在 lmarena 进行的测试中，**nano banana** 的表现优于 **Qwen**。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1407829749526565056)** (25 messages🔥): 

> `Hackathon start time, ChatGPT CUDA lies, Hackathon prerequisites, Single huge epoch vs multiple smaller epochs, CUDA vs Triton` 


- **黑客松周六上午 9:30 开始**：据一名成员透露，黑客松*很可能*在周六上午 **9:30** 左右开始。
- **ChatGPT 编造 CUDA 谎言**：一位成员报告称，**ChatGPT** 在 **CUDA** 中的 **float3 alignment**（对齐）和 **size**（大小）问题上公然撒了两次谎，但对 **ChatGPT** 表示理解，因为从 **OpenCL** 和 **OpenGL** 的实现来看，这确实是一个很难处理正确的问题。
   - 该成员证实 **CUDA** 中不存在 padding（填充）。
- **关于黑客松先决条件和申请的疑问**：一位成员询问了 **GPU hackathon** 的先决条件以及申请通道是否仍然开放。
   - 聊天中没有明确回答这个问题。
- **单 Epoch 与多 Epoch 之争**：一位成员询问，对于 **CLM** 来说，是使用海量数据集跑 **1 epoch** 更好，还是在较小数据集上跑多个 epoch 更好，以及目前最新的 scaling law 是什么。
   - 另一位成员回应称，他们处理的是较小的模型，在规模较大时，一半数据跑 2 epoch 的性能与全量数据跑 1 epoch 相当。
- **CUDA 与 Triton 正面交锋！**：一位成员询问黑客松将使用 **CUDA**、**Triton** 还是其他工具。
   - 有人提到两者皆可，而 **Triton** 可能会帮助参赛者提高开发速度；并暗示参赛者将使用较新的 **Nvidia** 芯片。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1408081843097571348)** (1 messages): 

> `Triton, AMD, NVIDIA, GPU, Data Layout` 


- **通过 Triton 处理 AMD 与 NVIDIA GPU 的数据布局差异？**：一位用户询问在使用 **Triton** 时，**AMD** 和 **NVIDIA** GPU 之间的数据布局差异是否需要调整代码，特别是关于行优先（row-wise）与列优先（column-wise）的数据读取。
   - 用户澄清他们问的不是 **tile sizes** 或 **grid layouts**，而是由 **Triton AMD backend** 自动处理的更底层的数据转置。
- **AMD vs NVIDIA**：消费级 GPU 对消费级 GPU，或服务器级 GPU 对服务器级 GPU 架构的比较。
   - 对 AMD 和 NVIDIA 架构进行了对比。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1408113668868018246)** (10 messages🔥): 

> `CUDA deployment, CudaWrangler, Dynamic Linking` 


- **在没有 CUDA toolkit 的机器上运行 CUDA 程序**：一位用户寻求关于在缺少 CUDA toolkit 但配备 NVIDIA GPU 的机器上部署 CUDA 程序的建议。
   - 一位成员建议利用 **Driver API** 和 **CudaWrangler** 库 ([CudaWrangler/cuew](https://github.com/CudaWrangler/cuew)) 来查询驱动程序，以避免程序崩溃。
- **动态链接与 PTX 烘焙简化 CUDA 部署**：原帖作者报告称，通过从“动态加载”切换到“动态链接”并禁用 **runtime/cudart** 依赖，取得了成功。
   - 他们还能够将 **PTX** 直接嵌入到二进制文件中，从而不再需要单独的 **PTX** 文件。
- **ldd 辅助识别和打包 Linux 上 CUDA 程序的依赖项**：一位成员建议使用 **ldd** 来识别依赖项，设置 **rpath**，并将它们随二进制文件一起发布，类似于 Linux 上的 “Windows 模式”。
   - 原帖作者指出该程序在 Windows 和 Linux 之间具有跨平台兼容性，但 macOS 尚未测试。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1408177180583792731)** (1 messages): 

> `PyTorch Contributor Awards 2025, Recognizing Innovation in PyTorch` 


- **PyTorch 奖项截止日期临近！**：**2025 PyTorch Contributor Awards** 的提名将于 **8 月 22 日**截止，不要错过表彰在 **PyTorch 生态系统**中推动创新和影响力的个人的机会。
   - 立即通过此[链接](https://linuxfoundation.research.net/r/8XD5T8N)提交您的提名，并查看[优秀提名技巧](https://pytorch.org/blog/nominations-open-for-the-2025-pytorch-contributor-awards/)。
- **通过提名推动创新**：表彰 **PyTorch 生态系统**中不断创新的贡献者。
   - 在 **8 月 22 日**之前提交提名。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/)** (1 messages): 

honeyspoon: 与 sglang 之类的相比，infinity server 的 embedding 速度有多差？
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 messages): 

snektron: 我更喜欢 Stolwijker
  

---

### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1407932292470542387)** (11 条消息🔥): 

> `AMD GPU debugger, rocGDB, SPIRV parser, libspirv` 


- **AMD GPU 调试器获得反汇编和 Wave 步进功能**：一位成员正在开发一款 **AMD GPU debugger**，并添加了反汇编（disassembly）和 Wave 步进（wave stepping）功能，展示在[这段视频](https://cdn.discordapp.com/attachments/1233704710389764236/1407932291912695949/2025-08-21_06-20-14.mp4?ex=68a88f60&is=68a73de0&hm=6e9ca7ceed29674c6943e989940a6c8e144707797c5abd571e1cfe60d3f3210d)中。
   - 该调试器不依赖于 **amdkfd KMD**，而是使用一个 mini UMD 驱动和 Linux kernel debugfs 接口，旨在实现 **rocdbgapi** 的等效功能。
- **放弃 rocGDB 转向自定义驱动**：一位成员正在构建一个不依赖 **rocGDB** 的 AMD GPU 调试器，而是通过 mini UMD 驱动加 Linux kernel debugfs 接口来读写 GPU 寄存器。
   - 目标是主要面向图形开发人员，至少目前是作为 **rocdbgapi** 的等效替代方案。
- **自己动手写 SPIRV 解析器？**：一位成员询问是否应该构建自己的 **SPIRV parser** 用于反汇编、反射（reflection）和调试信息提取，并提到 **SPIRV spec** 看起来非常直观。
   - 他们注意到目前缺乏处理调试信息的合适库，因此考虑进行完整实现。
- **libspirv 相当简单**：一位成员建议使用 **libspirv**，并指出 **SPIRV spec** 包含了所有必要的信息，完全可以自己动手实现。
   - 原作者决定为了更好的集成而实现自定义方案，并对建议表示认可。


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1408106371680960602)** (2 条消息): 

> `C=AB matmul, ALU utilization, buffer read bandwidth, float4x4 matmul, float4 / metal::dot kernel` 


- **分块 C=AB 矩阵乘法受 GPU ALU 限制**：一位成员编写了一个分块（tiled）**C=AB matmul** 内核，其中每个线程使用 **float4x4 matmul** 计算 C 的 4x4 分块，并观察到 **ALU utilization/limiter** 为 **55/75%**，而 **buffer read bandwidth** 为 **35%**。
   - 他对此感到惊讶，想知道 **float4x4 matmul** 是否在专用硬件中执行，并分享了[内核的 gist](https://gist.github.com/0xekez/c94ba3d5b43df10d17c98581e91280e3)。
- **朴素内核性能优于分块矩阵乘法**：同一位成员注意到，使用 **float4 / metal::dot** 的更朴素的内核比分块内核快 **2 倍以上**。


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/)** (1 条消息): 

miserlou1241: 非常酷！
  

---


### **GPU MODE ▷ #[general-leaderboard](https://discord.com/channels/1189498204333543425/1343002580531417211/1408081014441377833)** (12 条消息🔥): 

> `torch.compile errors, local evaluation issues` 


- ****Torch.compile** 抛出意外错误**：一位成员报告在使用 **torch.compile** 时出现 *unexpected error*，并分享了两个解决方案：一个使用了 **torch.compile**（提交编号 34166），另一个没用（提交编号 34160）。
   - 尽管报错，提交仍被记录并使该成员排名第 2，并注明使用的 GPU 是 **B200**。
- **解决本地评估工具问题**：一位成员询问关于本地代码评估的问题，称 **eval.py** 无法工作，特别是关于 `POPCORN_FD` 的设置。
   - 另一位成员澄清说 `POPCORN_FD` 是输出文件的文件描述符，并建议将其设置为 `1` 以输出到 stdout。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1407815994784747571)** (11 条消息🔥): 

> `Trimul Leaderboard Updates, B200 Performance, H100 Performance, MI300 Performance` 


- **MI300 在 Trimul 取得成功**：一位成员成功在 `trimul` 排行榜上提交了 **MI300** 的成绩，耗时 **3.50 ms**。
   - 另一个 **MI300** 的提交以 **5.83 ms** 的成绩获得第二名。
- **B200 统治 Trimul 排行榜**：一位成员在 **B200** 上以 **8.86 ms** 的成绩获得第 6 名，随后在 `trimul` 排行榜上提升至第 4 名（**7.29 ms**）。
   - 该成员在 **B200** 上多次获得第 3 名，最佳成绩达到 **4.54 ms**，随后又实现了一次 **2.15 ms** 的成功运行。
- **H100 稳居第二**：一位成员在 **H100** 上以 **3.80 ms** 的成绩获得 `trimul` 排行榜第二名。
   - 此次提交突显了 **H100** 平台的竞争性能。


  

---

### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1407992161051475978)** (3 messages): 

> `Opus 4.1, Steel Plate Production, Task Emphasis, Red Science Production` 


- **Opus 4.1 发现财富，助力工厂**：在对 **Opus 4.1** 进行钢板生产测试时，发现它意外地在开采铜矿和提取石油。
   - 这表明其对*当前任务的重视程度不够*，促使开发团队转向观察设置，以研究 **Opus 4.1** 如何提高专注度。
- **AI 自动化红色科技**：AI 系统成功实现了**红色科技**（red science）生产的自动化，截图证明了这一点。
   - 该系统能够正确识别并生产自动化创建科技包所需的必要组件。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1407954456745873438)** (3 messages): 

> `ND Layouts, colex` 


- **通过 Colex 访问 ND Layouts 中的元素**：一位成员询问在使用整数作为 **ND layout** 的索引时，元素的访问顺序是怎样的。
   - 另一位成员澄清该顺序是 **colex**（列优先/左优先）。
- **确认 Colex 顺序**：一位用户确认，在 ND layouts 中使用整数索引时，元素访问顺序确实是 **colex**。
   - 这再次强调了 **colex**（即列优先顺序）是此类索引的标准方法。


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1408129525929345044)** (10 messages🔥): 

> `Infiniband at home, Distributed training library, NCCL backend, IBGDA requirements` 


- **寻找家庭实验室 Infiniband 方案**：一位成员正尝试在家的 **4090** 和 **5090** 之间搭建 **infiniband**，以尝试分布式训练/推理。
   - 他们在 eBay 上以 25 美元的价格买了一些 **ConnectX-3 网卡**，但发现驱动程序仅支持 Ubuntu 20.04 及更早版本。
- **DIY 分布式训练框架兴起**：一位成员正在构建自己的 **pytorch 分布式训练库**，并使用迷你版 **NCCL** 作为后端。
   - 另一位成员对此表示兴趣，认为这是学习底层细节的一种方式。
- **深入研究 NVIDIA 网络文档**：一位成员建议在 Internet Archive 上查找旧版本的 [NVIDIA networking documentation](https://docs.nvidia.com/networking/index.html) 以寻找相关的驱动程序。
   - 该成员希望这能提供更多细节。
- **CX4 或 CX5 网卡具备 GPU-Aware 特性**：一位成员指出，许多 GPU-aware 功能依赖于 **ConnectX-4 (CX4)** 或 **ConnectX-5 (CX5)** 及更新型号的网卡。
   - 他们举例说明 **IBGDA** 需要 **CX5** 或更新型号。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1407883262126456913)** (33 messages🔥): 

> `Infinite Memory, Arxiv paper guide, LLMs for Legal Field, HRM Models Analysis, Message Passing Approaches` 


- **福布斯曝光 Grok 聊天记录**：来自 [Forbes](https://www.forbes.com/sites/iainmartin/2025/08/20/elon-musks-xai-published-hundreds-of-thousands-of-grok-chatbot-conversations/) 的一篇文章透露，**Elon Musk 的 xAI** 发布了数十万条 **Grok** 聊天机器人的对话。
   - 一位成员向 *@grok* 询问这是否属实。
- **图灵完备性需要无限内存**：一位成员认为图灵完备性需要无限内存，因此由于内存不足，宇宙无法创造出图灵完备的机器。
   - 另一位成员开玩笑地建议，如果让计算机足够慢，宇宙的膨胀或许能解决空间问题；而另一位补充道：*真实的内存需要被检索，距离越远，检索所需的时间就越长*。
- **牛津指南帮助初露头角的 Arxiv 作者**：一位成员分享了一份由牛津大学教授编写的 [Google Docs 指南](https://docs.google.com/document/d/16R1E2ExKUCP5SlXWHr-KzbVDx9DBUclra-EbU8IB-iE/edit?tab=t.0#heading=h.16t67gkeu9dx)，旨在帮助程序员撰写关于 LLM 训练的 Arxiv 论文。
   - 该用户想分享见解，但不知从何下手。
- **ARC Prize 分析 HRM 模型**：一位成员分享了 [fxtwitter 帖子](https://fxtwitter.com/arcprize/status/1956431617951740044)和 [ARC Prize 博客文章](https://arcprize.org/blog/hrm-analysis)的链接，其中分析了 HRM 模型。
   - 这是为了回应另一位用户关于 HRM 模型是否值得花时间学习的问题。
- **图片展示消息传递方法**：一位成员分享了一张插图，展示了神经网络中消息传递（message passing）的不同方法。
   - 该图片源自一本书，可通过 [arXiv 上的 PDF](https://arxiv.org/pdf/2104.13478) 获取。


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1407812166702207027)** (46 条消息🔥): 

> `Personality GAN, AI Welfare, Genome Conscious?, Super Weight, LLM Preferences` 


- ****SpongeBob GAN** 亮相！**：一位成员提出了一个 Personality GAN，其中 Generator = LLM，Discriminator = LLM，并使用 LoRA 进行微调，直到 Discriminator 无法区分真假 **Sponge Bob**。
   - 难点在于找到一个尚未在 **Sponge Bob** 上进行过大量训练的 LLM。
- ****AI Welfare**（AI 福利）受到认真关注！**：讨论了一篇关于 *Taking AI Welfare Seriously* [arxiv link](https://arxiv.org/abs/2411.00986) 的论文，涉及 Anthropic 关于 *Exploring Model Welfare* [Anthropic link](https://www.anthropic.com/news/exploring-model-welfare) 的文章。
   - 这与 [另一篇 Anthropic 文章](https://www.anthropic.com/research/end-subset-conversations) 关于 end-subset conversations 的内容有关。
- ****LLM Weight**（LLM 权重）的古怪现象！**：**Llama 3 7B** 权重矩阵中的一个数字变化就导致其输出乱码，引发了关于意识/身份的讨论 [Apple link](https://machinelearning.apple.com/research/the-super-weight)。
   - 一位成员问道：*他们是否仅通过调整一个数字就抹去了它的“意识”/“身份”？*
- ****LLM Preferences**（LLM 偏好）显现！**：有人指出模型在 pre-training 期间会形成类似人类的表示，且 LLM 确实存在偏好，参考了 [这篇 LessWrong 文章](https://www.lesswrong.com/posts/eWdzuHXzRdBkg49R9/favorite-colors-of-some-llms)。
   - 一位成员评论道：*在我的那个年代，我们管这叫 class imbalance bias（类别不平衡偏差）。*
- ****AI Duality**（AI 双重性）引发辩论！**：讨论涉及 AI 作为一种双用途技术，适用于所有领域，因为每个人都会使用它 [QuantaMagazine link](https://www.quantamagazine.org/the-ai-was-fed-sloppy-code-it-turned-into-something-evil-20250813/)。
   - 一位成员表示 *聪明是相对的*，并且 [恒温器也具有 Agency](https://www.youtube.com/watch?v=PiJwIUGJGmw&t=19s)，因为它们会对自身及其外部环境建模。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1407827073749221577)** (8 条消息🔥): 

> `Yann LeCun's position at FAIR, Thermodynamic computing chip, AI Slurs, Energy Efficiency in AI` 


- ****Zuckerberg** 可能要 **解雇 LeCun**？！**：一位用户根据 [Zuckerberg 的一条发帖](https://www.threads.com/@zuck/post/DMiwjXJSYCd?xmt=AQF06RR3brSSoKTMcQrLDyX6WGtg9UdpubK94sSm8FIPsg) 猜测 **Yann LeCun** 可能会离开 **FAIR**。
   - 另一位成员暗示 **LeCun** 可能已被降职，且 **Meta** 正在从开源模型领域撤退。
- **Clanker Cogsucker 机器人 AI 辱骂词汇走红！**：一位用户分享了 [一篇 Rolling Stone 的文章](https://www.rollingstone.com/culture/culture-features/clanker-cogsucker-robot-ai-slurs-viral-1235401262/)，讨论了诸如 *clanker* 和 *cogsucker* 等新型 **AI slurs**（AI 侮辱性词汇）的出现。
- **首款热力学计算芯片完成 Tape-out**：一位成员发布了 [一篇来自 Tom's Hardware 的文章](https://www.tomshardware.com/tech-industry/semiconductors/worlds-first-thermodynamic-computing-chip-)，关于 *全球首款热力学计算芯片* 达到 Tape-out 阶段。
- **AI 行业并不关心能源效率**：一位用户分享了 [一段 YouTube 视频](https://www.youtube.com/watch?v=LTCbx5KdqpU)，认为 **AI 行业** 普遍不优先考虑 **Energy Efficiency**。
   - 他们指出，另一家具有类似价值主张的公司已经破产，这表明该行业并不关心能源效率。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1407849425656746066)** (67 条消息🔥🔥): 

> `max_steps 困惑，levelbot Space 访问，高 token 下的模型幻觉，Pro version 支付问题，root mean square norm 量化错误` 


- **关于 max_steps 参数的困惑**：一名成员对 **max_steps** 参数及其在 **5090** GPU 上配合 **vllm** 的实现感到困惑，并询问 [LFM2-1.2B](https://huggingface.co/LiquidAI/LFM2-1.2B) 模型是否适用。
- **Token 限制触发幻觉 (Hallucinations)**：一名成员询问模型开始产生幻觉的 **token** 限制，并对任何模型能在 **1 million tokens** 下有效运行表示怀疑。
   - 另一名成员链接了 [Hugging Face 的 Agents 课程](https://huggingface.co/learn/agents-course/unit0/introduction) 和一个 Discord 频道，建议将这些资源作为潜在的解决方案。
- **用户报告 Pro Version 支付问题**：一名用户报告被收取了两次 **Pro version** 费用却未获得服务，被建议发送邮件至 website@huggingface.co 并在指定的 [MCP 频道](https://discord.com/channels/879548962464493619/1389546106970701865) 寻求帮助。
- **自定义损失函数微调 SFTTrainer**：一名成员分享了一个在 **ChatGPT** 帮助下创建的自定义损失函数，旨在配合 **SFTTrainer** 使用，以增强模型对医学文本中特定**否定词 (negation words)** 的关注。
   - 另一名成员建议改用带有偏好对 (preference pairs) 的 **DPO**，而另一位成员则强调了在医学领域挖掘难负样本 (hard negatives) 后使用 triplet loss 的实用性。
- **LLM 训练中 SFT 和 DPO 的比较**：成员们讨论了 **DPO** (Direct Preference Optimization) 与 **SFT** (Supervised Fine-Tuning) 的效果，一名成员指出 *DPO 与推理 (reasoning) 没有关系*，但在 **SFT** 之后进行 **DPO** 比仅进行 **SFT** 能获得更好的结果。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1408040029137142094)** (3 条消息): 

> `AgentX 交易平台，Language Diffusion Models，本地 AI 工作区 PDF 阅读器` 


- ****AgentX** 承诺打造 AI 交易智囊团**：全新的 [**AgentX**](https://www.linkedin.com/posts/alaa-salamah-96167b227_agentx-agentx-api-activity-7364245216851050498-BfRO?utm_source=share&utm_medium=member_desktop&rcm=ACoAADjfvpkBwpCXn_8Pmby7ixjV93Dje5TcmgUHi) 平台旨在提供一个汇聚最顶尖 AI 大脑——**ChatGPT**、**Gemini**、**LLaMA**、**Grok**——共同协作的交易平台。
   - 目标是让这些模型进行辩论，直到达成最佳操作共识，为交易者提供一个完全值得信赖的系统。
- **不到 80 行代码复现 Diffusion Language Models**：一名成员使用 🤗 Transformers 在不到 80 行代码内复现了 Nie 等人 (2025) 的论文 *Large Language Diffusion Models* 的部分内容。
   - 该[项目](https://github.com/gumran/language-diffusion)在 **TinyStories** 数据集上微调了 **DistilBERT**，结果好于预期，目前正在寻求反馈和 GitHub stars。
- **本地优先的 PDF 阅读 AI 工作区亮相**：一名成员在 Product Hunt 上发布了一个本地优先的 AI 工作区 PDF 阅读器，并分享了[链接](https://www.producthunt.com/products/collate-2?launch=collate-4)。
   - 他们请求社区的支持。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1408102264597385228)** (1 条消息): 

> `Hugging Face Learn 课程，422 错误` 


- **Hugging Face Learn 课程页面宕机**：一名成员报告 [Hugging Face LLM 课程的一个页面](https://huggingface.co/learn/llm-course/en/chapter12/3a) 无法访问。
   - 该页面显示 **422 error**。
- **Hugging Face Learn 课程需要修复**：一名用户报告 Hugging Face Learn 课程页面宕机并显示 **422 error**。
   - 该问题需要解决以便用户访问内容。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1407997140890026077)** (4 messages): 

> `Hugging Face Certificates, Agents vs MCP Course, Agent tool, LLM tasks` 


- **Hugging Face 证书位置困扰用户**：一位用户询问在哪里可以找到他们的 **Hugging Face 证书**以便发布到 LinkedIn。
   - 他们提到在平台或电子邮件中都找不到这些证书。
- **Agents 课程与 MCP 课程引发讨论**：一位用户正在纠结是在完成 Agents 课程的 Unit 1 后转到 **MCP 课程**，还是先完成 **Agents 课程**。
   - 由于时间限制，他们想知道应该优先选择哪门课程。
- **Agent 工具功能揭秘**：一位用户寻求关于 **Agent Unit 1** 成功运行的解释。
   - 他们理解 Agent 使用工具（functions），并且是触发这些工具来执行任务，而不是直接调用 **LLM**。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1407887543743283231)** (19 messages🔥): 

> `Gems for podcast generation, NotebookLM podcast length, Customizing NotebookLM podcasts, Analyzing Terms of Use and Privacy Policies, South Park episode on Terms and Conditions` 


- **AI Maestro 分享生成长播客的秘诀**：一位用户询问如何在 NotebookLM 中从 3-4 小时的 YouTube 视频生成更长的播客，对此一位用户建议使用预设提示词（set prompts）来逐段规划整个文案。
   - 一位用户分享了[一个工作流](https://github.com/para-droid-ai/scratchpad/blob/main/assistant-workflows-tasks-personas/deeper_podcast_synthetic-082025.txt)来创建“深度研究报告框架”，随后可用于通过 Gems, Gemini, PPLX 或 ChatGPT 生成播客。
- **通过自定义功能解锁更长的 NotebookLM 播客**：一位用户询问 NotebookLM 的播客长度限制，另一位用户指出在 **Customize** 选项（三个点）中可以将播客长度设置为 45-60 分钟。
   - 另一位用户补充道，指定主题可以让 Bot *集中讨论特定话题*，而不是指望它把所有重要内容都塞进一个播客里。
- **隐私政策担忧：医疗网站的妥协被曝光**：一位用户在想起*有人曾使用 AI 工具分析这些文档并大有发现*后，使用 Gemini 和 NotebookLM 分析了一家医疗保健公司的隐私政策和使用条款。
   - 该用户对*向这些公司出让了多少权利*感到惊讶，并认为这种方法对于理解使用条款（Terms of Use）和隐私政策非常有用。
- **South Park 预言了接受条款与条件的痛苦**：一位用户推荐去观看关于接受条款与条件的 **South Park** 旧剧集。
   - 另一位用户回想起一个案例：某个游戏的 EULA/隐私/条款中隐藏了一个竞赛，规定第一个拨打特定电话号码的人可以获得一千美元，而该奖项在六个月内都无人认领。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1407818234690011138)** (51 messages🔥): 

> `Video Length Limits, Study guide on android app, Audio Language Change, Public Sharing Issue, Notebook LM API` 


- **Android 应用功能对等推迟**：用户要求 NotebookLM Web 端和 **Android** 应用之间实现更多的**功能对等**，特别是学习指南功能。
   - 一位用户表示，目前的原生应用*几乎没用*，因为学习指南依赖于笔记功能，而原生应用中缺少该功能。
- **自定义屏幕提供语言更改选项**：一位用户询问如何更改 iOS 应用中生成的音频概览（audio overview）的语言。
   - 另一位用户回答说，语言设置可以在 **Customize** 菜单中找到。
- **无法公开分享 Notebook**：一位用户报告称，尽管拥有 Pro 账户，但仍无法公开或向外部分享 Notebook。
   - 该功能目前尚未开放。
- **NotebookLM 缺乏官方 API 但存在变通方法**：一位用户询问 NotebookLM 的 API。
   - 另一位用户建议使用 **Gemini API** 作为替代方案。
- **NotebookLM 中的 OCR 操作**：用户讨论了 NotebookLM 是否对多模态 PDF 执行 OCR 操作。
   - NotebookLM 支持 PDF 并且正在改进图像处理，但 OCR 识别尚不完美，用户可能需要重新上传 PDF 或使用**外部 OCR 工具**。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1407807040277053510)** (65 条消息🔥🔥): 

> `Base Model 发布，理想的 30B 模型，FA2 与 Context，Qwen 缩放，重要性矩阵校准数据集` 


- **字节跳动发布长 Context 模型**：字节跳动发布了一个具有极长 Context 的 Base Model，其特点是没有 MHLA、没有 MoE，甚至没有 QK norm，详见[这张图片](https://cdn.discordapp.com/attachments/1149866623109439599/1407959284280459305/image.png?ex=68a8a883&is=68a75703&hm=b8a5430da1f445204c76334cef07358c8d9b815da989424483a5ecf3bc65c790)。
   - 该模型在架构上被描述为 *vanilla*（原生），人们希望他们能发布一篇包含更多解释的论文。
- **Seed-OSS-36B 缺失 GGUF 引发关注**：用户好奇为什么没有 **Seed-OSS-36B** 的 **GGUF** 版本可用，因为这类版本通常出现得很快。他们引用了[这个链接](https://x.com/adityastomar_/status/1958048129275805867)并询问这是否对 ASIC 持看空态度。
   - 有人指出，延迟可能是由于自定义的 **vllm** 实现，且由于 *architectures*: ["SeedOssForCausalLM"]，该架构尚未被 **llama.cpp** 支持。
- **Seed 模型实现了 Dropout 和 Bias**：**Seed** 模型拥有类似于 **LLaMA** 的自定义 MLP 和 Attention 机制，但增加了 Dropout、输出的 Bias 项以及 **qkv** head 的 Bias 项，这些被解释为正则化技术。
   - 成员们好奇该模型训练了多少个 Epoch，但确认将其重命名为 **LLaMA** 是行不通的。
- **Qwen 通过 RoPE 缩放实现 512k Context**：**30B** 和 **235B Qwen 2507** 模型可以通过 **RoPE** 缩放实现 **512k** 的 Context，正如在[这个 Hugging Face 数据集](https://huggingface.co/datasets/eaddario/imatrix-calibration)中所讨论的。
   - 这些数据集用于生成重要性矩阵（imatrix），有助于在量化过程中减少误差。
- **Cursor 的 Kernel 博客获得好评**：成员们分享了 **Cursor kernel 博客**的[链接](https://x.com/stuart_sul/status/1957927497351467372)。
   - 有人评价 *cursor cooked*（Cursor 表现出色）。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1407950357379809300)** (47 条消息🔥): 

> `DeepSeek V3.1, R-Zero LLM 训练方法, 中国与美国的能源可用性, Kimi K2 结合优于 GPT-5 的图像生成` 


- **DeepSeek V3.1 发布：增量式进步**：新发布的 **DeepSeek V3.1** 模型被一些成员指出更像是*增量式改进*，并伴随一些退化，参考 [DeepSeek 官方页面](https://huggingface.co/deepseek-ai/DeepSeek-V3.1)。
- **DeepSeek 采用 Anthropic API**：**DeepSeek** 现在支持 **Anthropic API**，扩展了其功能和覆盖范围，正如 [X 平台](https://x.com/deepseek_ai/status/1958417062008918312)上宣布的那样。
- **R-Zero：自我进化的 LLM**：分享了一份关于 **R-Zero** 的综合研究 [PDF](https://cdn.discordapp.com/attachments/1371757564005711973/1408153973751545896/R-Zero__A_Comprehensive_Study_of_a_Self-Evolving_LLM_Training_Method.pdf?ex=68a8b515&is=68a76395&hm=dcba93436f636eeec364d08d08e1603131147faaa595637065f2e226772005f2&)，这是一种从零人类数据开始并独立改进的自我进化 **LLM 训练方法**。
- **中国优先考虑能源可用性**：一位成员指出，在中国，*能源可用性被视为理所当然*，这与美国关于数据中心功耗和电网限制的争论形成鲜明对比，参考了[这篇《财富》杂志的文章](https://fortune.com/2025/08/14/data-centers-china-grid-us-infrastructure/)。
- **更好的图像生成 + Kimi K2**：一位成员指出，如果 **Kimi K2** 能结合**优于 GPT-5 的图像生成能力**，将会更加强大（OP），并分享了[这个 Reddit 链接](https://www.reddit.com/r/ChatGPT/s/vUrGedSwY5)。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1407819836352106507)** (36 messages🔥): 

> `Gemini 2.5 Pro Failure, Qwen CLI Charging, GPT-5 Benchmarks, DeepSeek v3.1 Pricing, OpenRouter Think Mode` 


- ****Gemini 2.5 Pro 失败而 Flash 成功****：一位成员报告称 **Gemini 2.5 Flash** 可以正常工作，但 **Gemini 2.5 Pro** 持续失败，而如果设置了账单，`gemini/gemini-2.5-pro-preview-06-05` 则可以工作。
   - 另一位成员报告称因 **qwen-cli** 进程被扣费 **$25**，目前正在寻求退款。
- ****用户因使用 Qwen CLI 被意外扣费****：一位用户在通过 OAuth 验证 Google 身份后，因使用 **qwen-cli** 被扣费 **$25**，尽管其目标是获取来自 Alibaba Cloud 的免费额度。
   - 他们提交了一个工单，展示了控制台记录中 **一次没有输出的调用耗费了 $23**。
- ****社区渴望对 GPT-5 低推理模型进行基准测试****：成员们正在对 **gpt-5-mini** 和 **gpt-5-nano** 进行基准测试，因为他们在完整的 **gpt-5** 上受到了速率限制，尽管一位用户声称 *gpt-5-mini 非常出色且便宜*。
   - 频道中已经发布了 **gpt-5-mini** 的测试结果和 PR。
- ****DeepSeek v3.1 价格显著上涨****：用户报告称，从 2025 年 9 月 5 日开始，DeepSeek 将提高两个模型的价格，以匹配 reasoner 模型的价格。
   - 与新的 **deepseek 3.1** 相比，输入价格从 **$0.25** 上涨至 **$0.27**。
- ****OpenRouter 需要 Think 模式****：一位用户报告称 **OpenRouter** 似乎没有 "think" 模式，但可以通过命令行使用以下代码片段来调用：`aider --model openrouter/deepseek/deepseek-chat-v3.1 --reasoning-effort high`。
   - 社区建议更新模型配置以解决此问题。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1407817255621754893)** (3 messages): 

> `aider stdout issue, polyglot benchmark on llama cpp` 


- **Aider 的标准输出难题**：一位用户报告了 **program output/stdout** 无法在 **aider** 中显示的问题，并发布了一张 [图片](https://cdn.discordapp.com/attachments/1133060505792159755/1407817255433277440/image.png?ex=68a8ccfd&is=68a77b7d&hm=c93b6e3d3d4d1b0dc321355cd459dbd4e8371fd5bfe1c43c82d2701b9b6cd831&)。
- **破解 Polyglot 基准测试结果**：一位在本地 **llama cpp model** 上运行 **polyglot benchmark** 的用户询问如何获取每种语言的结果。
   - 该用户随后找到了 [解决方案](https://discord.com/channels/1131200896827654144/1400603686350360678/1400993983999770694) 并分享了链接，供其他寻求特定语言基准测试结果的人参考。


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/)** (1 messages): 

end4749: <@293486003245809664> 垃圾信息？ ^
  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1408187482075299851)** (1 messages): 

> `marimo notebooks, Graph RAG with DSPy, DSPy modules optimization` 


- **Marimo Notebooks：Jupyter 的精神继任者**：一位成员一直在发布关于 [**marimo notebooks** 的教程](https://www.youtube.com/watch?v=2aepn9uRVOM)，它可以同时作为 notebook、Python 脚本和应用运行。
   - 该教程强调了在迭代 **Graph RAG with DSPy** 的想法时 **marimo** 的实用性。
- **未经优化的 DSPy 流水线**：展示的 **DSPy pipeline** 故意没有进行优化，以强调仅通过 signature 和 module 就能实现多少功能。
   - 该方法侧重于在深入优化之前，通过以各种方式组合 **DSPy modules** 来进行快速迭代。
- **深入探讨优化**：即将发布的视频和博客文章将深入探讨 **DSPy modules** 优化的主题。
   - 目前的教程是为那些想要开始使用 **marimo** 的人提供的入门介绍。


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1408079463996199084)** (5 messages): 

> `IBM AutoPDL paper, DSPy code readability, Justification of work` 


- **IBM AutoPDL 的主张被驳回**：一位成员认为没必要回应每一个主张，暗示每个人都在寻找角度来证明自己工作的合理性，并指出关于不可读性的主张是错误的。
   - 他们表示 *DSPy 代码和 prompt 在任何意义上都极其具有人类可读性，甚至称得上优美。*
- **为 DSPy 代码可读性辩护**：一位成员辩称 **DSPy** 的代码和 **prompt** 极其易读、易懂且清晰，对相反的主张提出了挑战。
   - 该成员强调，代码的可读性使其易于理解和使用。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1407849483231825921)** (28 messages🔥): 

> `dspy.GEPA 版本，微调 dspy 描述，保存优化后的程序，GEPA 的上下文长度，KPMG 入职培训` 


- **DSPy 的 GEPA 在 v3.0.1 中现身**：一位成员询问包含 **GEPA** 的 **dspy** 库版本，另一位成员确认该功能在 **3.0.1** 版本中可用，如附带的 [截图](https://cdn.discordapp.com/attachments/1161519469319946286/1407936409615990904/image.png?ex=68a89336&is=68a741b6&hm=72219936a525599fc3faca9d127d106a09e0639e9eeae1564a8bbfc196b07ffa&) 所示。
- **DSPy 微调：描述性还是原生？**：在微调期间，一位成员询问在 **dspy.InputField()** 和 **dspy.OutputField()** 中使用“原生描述（vanilla descriptions）”是否常见，以便让优化器自由发挥。
- **DSPy 将优化后的程序保存在 Pickle 中**：一位用户报告了保存优化程序时的问题，指出元数据仅包含 **dependency versions** 信息，而不包含程序本身，即使使用了 `optimized_agent.save("./optimized_2", save_program=True)`。
- **GEPA 遭遇截断**：当用户为 **GEPA** 设置了 **32k** 的最大上下文长度但仍收到截断的响应时，成员们讨论了长推理的复杂性以及多模态设置中的潜在问题。
   - 一位成员引用一个复杂的 Prompt 示例开玩笑说：“想象一下必须维护那个东西”。
- **RAG 是大材小用，直接拼接即可（或者不）**：成员们开玩笑地争论对于处理税法或农作物保险文件等任务，**RAG** (Retrieval-Augmented Generation) 还是简单的 **拼接（concatenation）** 更合适，并承认数百万份文件的规模有时确实需要 RAG。
   - 一位成员调侃道：“RAG 是大材小用。直接把税法拼接起来就行了，”而另一位反驳道：“哦，我猜那超过 100 页了。好吧，那 RAG 挺好的。”


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1407880904814366720)** (13 messages🔥): 

> `command-a-03-2025 的引用问题，保证引用，command-a-reasoning 发布，使用 Langchain 构建 RAG，Cohere 对比 Qwen3-coder 30B` 


- **`command-a-03-2025` 间歇性引用引发 Prompt 挫败感**：一位用户报告称 `command-a-03-2025` 仅间歇性地返回引用，即使 maxTokens 设置为 8K，这在生产环境中导致了信任问题，并寻求某种保障。
   - 一位 Cohere 成员澄清说 `command-a-03-2025` 在引用时使用“快速”模式（根据 [API 参考](https://docs.cohere.com/reference/chat#request.body.citation_options.mode)），引用并不保证一定会生成，但可以通过系统 Prompt 引导模型，并且最新发布的 SOTA 模型 **command-a-reasoning** 可能也会有所帮助（参见 [博客](https://cohere.com/blog/command-a-reasoning)）。
- **Langchain RAG 探索开启**：一位成员正在学习 Langchain 以构建 RAG (Retrieval-Augmented Generation) 应用。
   - 他们提到打算使用 **command-a-reasoning**，期待 **command-a-omni** 的发布，并对未来名为 **Command Raz** 的模型表示期待。
- **Cohere 与 Qwen 争夺本地 LLM 席位**：一位用户正在寻找 **Qwen3-coder 30B** 模型的 Cohere 替代方案，目标是使其能够运行在 **64GB M4 Max** 配置上。
   - 该用户“非常想尝试 Cohere 的方案来替代本地强力模型 Qwen3-coder 30B”，以便能适配其 64GB M4 Max。


  

---

### **Cohere ▷ #[📣-announcements](https://discord.com/channels/954421988141711382/996880279224451154/1408103056800874497)** (1 条消息): 

> `Command A Reasoning 模型, 企业级 AI, Agentic AI 平台` 


- **Cohere 发布 Command A Reasoning 模型**：Cohere 发布了 **Command A Reasoning**，这是其最新的用于推理任务的企业级模型，在 Agentic 和多语言基准测试中表现优于其他私有部署模型；该模型可通过 [Cohere Platform](https://dashboard.cohere.com/playground/chat?model=command-a-reasoning-08-2025) 和 [Hugging Face](https://huggingface.co/CohereLabs/command-a-reasoning-08-2025) 获取。
- **Command A Reasoning 规格与特性揭晓**：新模型专为企业需求设计，提供高度安全、高效且可扩展的部署选项，可在单张 **H100** 或 **A100** 上运行，上下文长度为 **128k**，在多 GPU 上可扩展至 **256k**；更多信息请参阅 [Cohere 博客](https://cohere.com/blog/command-a-reasoning)。
- **Token Budget 功能控制成本与计算资源使用**：Cohere 的 Command A Reasoning 具备 **token budget** 设置功能，可直接管理计算资源使用并控制成本，无需区分推理和非推理模型，同时满足准确率和吞吐量需求。
- **Command A Reasoning 为 North 提供动力**：**Command A Reasoning** 是驱动 **North** 的核心生成模型，North 是 Cohere 的安全 Agentic AI 平台，支持自定义 AI Agent 和本地自动化。


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1408009102625341461)** (4 条消息): 

> `Azure AI Foundry 上的 Cohere Embed-v4, Cohere Python 库 Document 对象` 


- **Cohere Embed-v4 输入类型映射**：一位成员正在 .NET 应用程序中使用部署在 **Azure AI Foundry** 上的 **Cohere Embed-v4**（通过 Azure AI Inference API），并寻求关于 **Microsoft 的 `EmbeddingInputType`** 如何映射到 **Cohere API** 文本嵌入的相关说明。
   - 具体而言，由于 Cohere 的 `input_type` 参数中缺乏显式的文本选项，他们不确定 `EmbeddingInputType.Text` 是否应该映射到 Cohere API 中的 `search_document`。
- **Cohere Python 库的 Document 对象**：一位成员对 Cohere Python 库中的 **`Document` 对象**提出疑问，其中 `data` 字段预期为一个字典 (`typing.Dict[str, typing.Optional[typing.Any]]`)。
   - 他们指出 Tool Use 快速入门示例在该字段中使用了一个字符串（`json.dumps` 调用的输出），并想知道 Python 绑定是否正确处理了此情况，参考文档为 [Tool Use 快速入门文档](https://docs.cohere.com/v2/docs/tool-use-quickstart)。


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1407811130512113815)** (7 条消息): 

> `MLE 研究, 独立可解释性研究, AI 创新与价值创造, 企业工作流` 


- **MLE 寻求研究团队联系**：一位拥有 **MLE** 经验的计算机科学硕士毕业生，正寻求与研究团队或组织建立联系。
   - 该成员表达了合作并为研究工作做出贡献的兴趣。
- **渴望合作的可解释性研究员**：一位居住在印度班加罗尔、拥有 **8 年**应用机器学习经验的独立可解释性研究员，正在转型至 AI 研究领域，专注于机械可解释性 (Mechanistic Interpretability)。
   - 该研究员对评估、模型去偏和 RL 感兴趣，寻求在可解释性相关话题上的合作与讨论。
- **执行顾问架起 AI 创新与价值的桥梁**：一位拥有 **25 年以上**经验的独立顾问兼执行顾问加入了社区，擅长将技术和 AI 创新与价值创造相结合。
   - 凭借在埃森哲 (Accenture)、IBM 和德勤 (Deloitte) 等公司的经验，他们现在帮助客户通过 AI 创造可持续的、全组织范围的价值，公司网站为 [Mantha Advisory](https://www.manthaadvisory.com/own)。
- **CTO 探索 Cohere 以优化产品**：一位拥有 **25 年以上**经验的 CTO 最近发现了 Cohere，并有兴趣探索其在改进产品方面的能力。
   - 他们关注数据质量、规模、性能、工作流、数据完整性和多语言支持，并热衷于向社区学习。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1407802615718482010)** (12 条消息🔥): 

> `C# client library, MCP server's instructions field, MCP servers, generate_test_prompt.md, GitHub` 


- **MCP 客户端忽略 Instructions 字段**：成员们在使用 **MCP 客户端**（尤其是 **Claude**）时遇到问题，**instructions 字段**似乎被忽略了，而系统更倾向于 **工具描述 (tool descriptions)**。
   - 一位成员建议 *添加指令、上下文然后重复指令会产生更好的效果，但由于工具已集成到 API 中，这变得不可行*。
- **MCP 服务器选项评估**：一位成员询问开发者们正在使用哪些 **MCP 服务器**，以及哪些工具在这些服务器中效率更高。
   - 另一位成员强调了 **GitHub** 用于版本控制、**Python** 配合 **FastAPI** 用于后端开发以及 **PyTorch** 用于机器学习的实用性。
- **让 Agent 遵循指令**：一位用户询问如何让 Agent 遵循特定的 **generate_test_prompt.md** 文件，并对 Agent 在开启新对话时无法坚持项目的逻辑设计模式表示沮丧。
   - 他们在消息中附带了一张 [截图](https://cdn.discordapp.com/attachments/1312302100125843479/1408171379236409354/Screenshot_3.png?ex=68a8c54b&is=68a773cb&hm=1fbd862963889b97ef764b1599c2960340dab7ce357b3717f577e2b1491ffdc2)。
- **MCP 服务器解析优先考虑工具描述**：一位成员指出，**MCP 服务器**内部的解析逻辑可以结构化为在 **instructions 字段**之前处理 **工具描述**。
   - 建议 *审查服务器文档、检查客户端配置、分析服务器端逻辑* 并 *进行受控实验*。
- **列举指令遵循模型**：成员们讨论了哪些模型能够遵循指令并生成结构化输出，推荐了 **Mistral-7B-Instruct**、**DeepSeek-Coder** 和 **Phi-3**。
   - 他们还提到了 **OpenHermes-2.5-Mistral-7B**、**WizardLM-2** 和 **Gorilla-LLM** 作为专门针对函数调用 (function-calling) 的模型。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1407927339345772656)** (10 条消息🔥): 

> `Web-curl, MCP-Boss, MCP Explained Video, SWAG-MCP, MCP Routing` 


- ****Web-curl** 为 LLM Agent 提供 Web 和 API 交互能力**：一位成员介绍了 **Web-curl**，这是一个使用 Node.js 和 TypeScript 构建的开源 **MCP 服务器**，使 LLM Agent 能够以结构化的方式获取、探索并与 Web 及 API 交互，完整代码可在 [GitHub](https://github.com/rayss868/MCP-Web-Curl) 获取。
- ****MCP Boss** 为 MCP 服务提供集中式密钥管理**：一位成员构建了 **MCP Boss** 来集中管理密钥，提供单一 URL 来网关化所有服务，具有多用户身份验证和通过 OAuth2.1 或静态 HTTP 标头进行 MCP 授权等功能 ([mcp-boss.com](https://mcp-boss.com/))。
- **视频揭秘 MCP**：一位成员发布了名为《MCP Explained: The Ultimate Deep Dive》的视频，[可在 YouTube 观看](https://youtu.be/xPq53oQi2tY)，并邀请大家对引导 (Elicitation)、根目录 (roots) 和采样 (sampling) 等客户端功能进行反馈和讨论。
- ****SWAG-MCP** 为可流式传输的 HTTP MCP 服务器生成反向代理配置**：一位成员分享了 **SWAG-MCP**，这是一个旨在为 SWAG 生成反向代理配置的 MCP 服务器，支持自托管服务和可流式传输的 HTTP MCP 服务器 ([github.com/jmagar/swag-mcp](https://github.com/jmagar/swag-mcp))。
- ****MCP Gateway** 使用 AI 路由请求**：一位成员开发了一个带有 **AI 驱动路由** 功能的轻量级网关，以解决 Agent 需要知道哪个特定服务器拥有正确工具的问题，代码可在 [GitHub](https://github.com/oliverye7/mcp-gateway) 获取。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1408147314702286910)** (2 条消息): 

> `Modverse #50, Custom Server Tag` 


- **Modular 发布 Modverse #50**：Modular 发布了 [Modverse #50](https://www.modular.com/blog/modverse-50)，其中介绍了多位成员。
   - 公告还提到他们现在拥有了自定义服务器标签。
- **自定义服务器标签上线**：Modular 团队宣布上线自定义服务器标签，并在附件图片中展示。
   - 链接的图片 ([Screenshot_2025-08-21_at_5.22.15_PM.png](https://cdn.discordapp.com/attachments/1098713601386233997/1408199603861323878/Screenshot_2025-08-21_at_5.22.15_PM.png?ex=68a8df94&is=68a78e14&hm=2991584cc0b81449dbc278d1d8302e55aabc54c58a6620e7041deb9dbd20e951&)) 显示了新标签。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1407812660845871204)** (10 messages🔥): 

> `kgen 和 pop 文档，MLIR dialects，pop.union 对齐 bug，GitHub issue 5202` 


- **kgen 和 pop 的文档稀缺**：一位成员询问关于 **kgen** 和 **pop** 的文档，特别是操作和参数，但另一位成员表示 *目前没有关于内部 MLIR dialects 的全面文档*。
   - 分享了 GitHub 上的 [pop_dialect.md](https://github.com/modular/modular/blob/main/mojo/stdlib/docs/internal/pop_dialect.md) 链接，并澄清这些是 stdlib 与编译器之间契约的一部分，*因此在 stdlib 之外使用它们需自担风险*。
- **怀疑 pop.union 存在对齐 Bug**：一位成员询问了 **pop.union** 中元素的对齐问题，指出在使用 `sizeof` 时出现了意料之外的大小。
   - 他们分享的代码显示 `union_type_simple_8_bit_stdlib` 的大小为 **16 bytes**，而 `union_type_simple_8_bit` 和 `union_type_simple_multi_bit` 的大小均为 **8 bytes**，另一位成员建议 *对齐问题可能是一个 bug*。
- **已创建 Issue 以调查对齐 Bug**：一位成员在 GitHub 上创建了 [issue 5202](https://github.com/modular/modular/issues/5202)，以调查 **pop.union** 中疑似存在的对齐 bug。
   - 该成员指出他们不确定这是技术水平问题（skill issue）还是 bug，同时也观察到 **pop.union** 似乎没有在任何地方被使用。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1407837356937187378)** (7 messages): 

> `TextGenerationPipeline 'execute' 方法，用于获取 logits 的自定义推理循环，语言分配器与 OOM 处理` 


- **TextGenerationPipeline 的 `execute` 方法浮出水面**：一位成员正在寻找 `TextGenerationPipeline` 上的 `execute` 方法但未能找到。
   - 另一位成员指出了 [Modular 仓库中的相关代码行](https://github.com/modular/modular/blob/fe7ba5d5f8b08ccba7356c45b90afcf495421469/max/pipelines/lib/pipeline.py#L977) 并建议检查 MAX 版本。
- **为 Logit 爱好者准备的自定义推理循环？**：一位成员报告称，在创建自定义推理循环时，难以在获取下一个 token 的同时检索 **logits**，感觉有些繁琐。
   - 该成员链接了一个 [Google Docs 文档](https://docs.google.com/document/d/1Hd6xZnf0bmg9SMQU1h10cd4Cwd9HDOPzPHZNqiPnrpg/edit?tab=t.0) 以提供背景信息，并确认该选项仍然可用，但其未来尚不确定。
- **内存分配器是必选项？**：一位成员建议，在将内存分配器集成到语言中之前，可能需要健壮的分配器支持。
   - 他们认为大多数用户不想手动处理内存不足（**OOM**）错误。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1408123828470677533)** (2 messages): 

> `企业级文档 AI，vibe-llama` 


- **LlamaIndex 揭秘企业级文档 AI**：LlamaIndex 的产品副总裁将于 **PST 时间 9 月 30 日上午 9 点**分享一年来关于[文档](https://t.co/x70xjEQaFs)解析、提取和索引的企业级实践经验。
- **使用 vibe-llama 简化开发**：LlamaIndex 发布了 **vibe-llama**，这是一个命令行工具，可以自动为阁下喜爱的 coding agents 配置有关 **LlamaIndex framework** 和 **LlamaCloud** 的最新上下文和最佳实践。
   - 它还包含[更多信息](https://t.co/G1gINq9kge)。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1407815234013364325)** (13 条消息🔥): 

> `HuggingFace CrossEncoder 重复、Agent 创建项目、AI 安全调查` 


- ****CrossEncoder 类**：Core 与 Integrations**：一名成员询问了 `llama-index` 中重复的 **CrossEncoder 类**实现，具体位于 `.core` 和 `.integrations` 下（[代码链接](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/postprocessor/sbert_rerank.py)）。
   - 另一名成员澄清说，`.core` 中的实现是 v0.10.x 迁移后的遗留物，应该被删除，建议改用 `llama_index.postprocessor.sbert_rerank` 并执行 `pip install llama-index-postprocessor-sbert-rerank`。
- **寻求 **Agent 创建网关****：一名成员询问是否有现有的项目可以作为**网关**，将 **model、memory 和 tools** 整合在一起，并暴露一个 **OpenAI 兼容端点**。
   - 该成员想知道是否有可以利用的现有项目，以避免在 Agent 探索中重复造轮子。
- ****AI 安全调查**：需要社区意见！**：一名成员分享了一个 [AI 安全调查链接](https://mukullight.pythonanywhere.com/form)，以收集社区对重要 **AI 安全问题**的看法。
   - 该成员请求大家填写表单，以帮助他们了解 **AI 安全社区**最感兴趣的内容，并请大家对可能的加载延迟保持耐心。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1407840535074439358)** (13 条消息🔥): 

> `积分购买、工单问题、比赛操纵指控、每日免费积分、推荐积分` 


- **积分购买选项缺失**：成员们报告说购买额外积分的选项消失了，其中一人指出他们只能看到*升级包*选项。
   - 另一名成员确认该选项*目前已下线*。
- **未解决的支持工单困扰用户**：一名用户报告任务出现问题并创建了工单 **#1318**，但尚未收到回复或无法访问该工单。
   - 他们请求团队协助，并艾特了一名特定成员。
- **比赛获胜者引发操纵指控**：一名用户指责比赛的第二名*不配获胜*，并声称比赛*似乎被操纵*。
   - 目前没有提供进一步的证据或细节来支持这一说法。
- **每日免费积分已停止？**：一名一个月后重返 Manus 的用户注意到他们没有收到通常的 **每日 300 免费积分**。
   - 他们询问 Manus 是否已经停止提供这些积分。
- **推荐积分代码难题**：一名用户询问如何领取推荐积分，提到系统要求输入代码。
   - 该用户表示他们不知道在哪里可以找到所需的代码。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1407818167493066922)** (7 条消息): 

> `Overworld 常量折叠、View(const) 重构、UPat cvar 与 UPat.const_like 重新定义、RANGEIFY=1 的影响、base 移除` 


- **探索 Overworld 常量折叠策略**：一名成员正在探索 overworld 常量折叠，可能涉及 **view(const) 重构**，并提议重新定义 `UPat.cvar` 和 `UPat.const_like` 以匹配 `CONST` 和 `VIEW(CONST)`。
   - 目标是折叠像 `x * 0` 这样的表达式，但有人担心符号计算中可能出现的有效性问题和 `.base` 扩散，如[此 Discord 讨论串](https://discord.com/channels/1068976834382925865/1255400554012741683/1407782654958506004)所述。
- **替代方案：ALU View Pushing**：建议了一种替代方法，借鉴了 **S-Lykles 的方法**，涉及在 kernelize 中添加一个 upat，将 view 直接推送到 **ALU**。
   - 这种方法配合针对 `x * 0` 的特殊规则（理由是 `* 0` 在计算上无关紧要），将允许未经修改的符号匹配。
- **提倡移除 base**：一名成员强烈建议不要采用提议的方法，认为其“非常丑陋”，并主张**移除 `.base`**。
   - 讨论还质疑了在此背景下如何处理 **PAD** 操作。
- **RANGEIFY=1 作为潜在的简化方案**：有人建议设置 **RANGEIFY=1** 可能会带来更简洁的实现。
   - 然而，项目目前处于旧引擎和 rangeify 并存的过渡阶段，处于一种悬而未决的状态。


  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1408057198164049941)** (3 条消息): 

> `GPT4ALL Enterprise vs Free, Model Selection for LocalDocs` 


- **用于私有模型使用的 GPT4ALL 免费版**：一位用户询问了关于公司希望私密且安全地使用其 **AI model** 时如何使用 **GPT4ALL** 的问题。
   - 另一位成员澄清说，如果公司已经准备好了自己的 **AI model**，那么 **免费版** 就足够了。
- **LocalDocs 的模型选择**：一位用户正在寻求模型推荐，以便利用 **GPT4All 的 LocalDocs 功能**，从数百篇 **PDF 格式的科学论文**中构建个人知识库。
   - 该用户说明他们拥有配备 **24 GB VRAM** 的 **Nvidia RTX 5090** 和 **64 GB RAM**，并希望所选模型具备 **reasoning capabilities**。