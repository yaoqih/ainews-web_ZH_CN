---
companies:
- google-deepmind
- anthropic
- context-arena
- artificial-analysis
- epoch-ai
- scaling01
date: '2026-02-21T05:44:39.731046Z'
description: '**Gemini 3.1 Pro** 与 **GPT-5.2** 和 **Opus 4.6** 相比，展示出了更强的检索能力和成本效益，但也有用户反映其在工具配套和
  UI 界面上存在问题。**SWE-bench Verified** 的评估方法因其一致性正受到审查，相关更新使测试结果更接近开发者的宣称。关于基准测试究竟在衡量前沿模型的哪些能力，目前存在广泛争论，特别是在针对
  ARC-AGI 谜题方面。**Claude Opus 4.6** 在处理软件任务时展现出了虽有波动但值得关注的 **14.5 小时时间跨度**（time horizon），但
  Token 限制仍会导致实际运行中的失败。**Sonnet 4.6** 在代码和指令遵循的基准测试中提升显著，但由于产品功能倒退（regressions），用户的反对情绪也在增长。'
id: MjAyNi0w
models:
- gemini-3.1-pro
- gpt-5.2
- opus-4.6
- sonnet-4.6
- claude-opus-4.6
people:
- dillonuzar
- artificialanlys
- yuchenj_uw
- theo
- minimax_ai
- epochairesearch
- paul_cal
- scaling01
- metr_evals
- idavidrein
- xlr8harder
- htihle
- arena
title: 今天没发生什么事。
topics:
- retrieval
- benchmarking
- evaluation-methodology
- token-limits
- cost-efficiency
- instruction-following
- software-reasoning
- model-reliability
---

**平静的一天**

> 2026/2/19-2026/2/20 的 AI 新闻。我们为您检查了 12 个 subreddits、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 24 个 Discord（**262** 个频道，**12582** 条消息）。为您节省了约 **1242** 分钟的阅读时间（按 200wpm 计算）。[AINews 网站](https://news.smol.ai/) 支持搜索所有往期内容。提醒一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。您可以[选择加入/退出](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件发送频率！

---

# AI Twitter 综述


**前沿模型评估：Gemini 3.1 Pro、SWE-bench、MRCR 以及“两极分化”的现实表现**

- **Gemini 3.1 Pro 展示了强大的检索能力 + 参差不齐的 Agent 可用性**：Context Arena 的 MRCR 更新报告称，**Gemini 3.1 Pro Preview** 在较易的检索任务中（2-needle @128k AUC **99.6% vs 99.8%**）几乎与 **GPT-5.2 (thinking:xhigh)** 持平，而在更难的多针检索任务中（8-needle @128k AUC **87.8%**，超过了此处报告的 GPT-5.2 thinking 层级）表现明显更强 ([DillonUzar](https://x.com/DillonUzar/status/2024655613293215855))。另外，**Artificial Analysis** 强调了一个可能被低估的角度：**Token 效率 + 价格**；他们声称其 Intelligence Index 测试套件在 Gemini 3.1 Pro Preview 上的成本为 **892 美元**，而 GPT-5.2 xhigh 为 **2,304 美元**，Opus 4.6 max 为 **2,486 美元**，且在他们的运行中消耗的 Token 比 GPT-5.2 更少 ([ArtificialAnlys](https://x.com/ArtificialAnlys/status/2024677979390169536))。
- **但工程师们报告称其“跑分强，产品弱”**：多个推文串抱怨 Gemini 的工具/测试框架（harnesses）滞后——例如，CLI 中的模型可用性不一致，“Antigravity”中存在 Bug 的 Agent 行为，以及令人担忧的“UI 撒谎 / 模型撒谎”混淆，即 App 声称是 Gemini，但底层报告是 Claude ([Yuchenj_UW](https://x.com/Yuchenj_UW/status/2024708583829753909), [Yuchenj_UW](https://x.com/Yuchenj_UW/status/2024721228842565851))。即使是热情的评价（“更快的马”）也伴随着对其日常实际使用的沮丧感 ([theo](https://x.com/theo/status/2024808734053347608))。
- **SWE-bench Verified 评估方法论再次成为焦点**：MiniMax 指向了一项对 **MiniMax M2.5** 在相同设置下 SWE-bench Verified 结果的“独立观察”，暗示早先跨实验室的比较可能是非对等的 ([MiniMax_AI](https://x.com/MiniMax_AI/status/2024646767325958285))。Epoch AI 明确承认了这种失效模式：他们更新了 SWE-bench Verified 方法论，因为他们之前的运行结果与其他人的存在系统性差异，现在看到的结果更接近开发者报告的分数 ([EpochAIResearch](https://x.com/EpochAIResearch/status/2024924403142910137))。
- **基准测试的异常引发了“我们到底在衡量什么？”的辩论**：一个例子是——前沿模型“横扫 ARC-AGI”，但在四子棋（Connect 4）上却表现挣扎，这表明 ARC 风格的谜题尽管设计初衷是抵抗过拟合，但可能只捕捉到了空间/游戏推理的一小部分 ([paul_cal](https://x.com/paul_cal/status/2024748708223402120))。另一个推文串预计只有少数模型能在 ARC-AGI-3 的“简单框架”上取得进展，并指出成本是主要的制约因素 ([scaling01](https://x.com/scaling01/status/2024650634746610041), [scaling01](https://x.com/scaling01/status/2024661145286557872))。

**Claude Opus/Sonnet 4.6：时间跨度评估、成本与可靠性体系**

- **METR 的“时间跨度”（time horizon）在 Opus 4.6 上大幅提升，但估算结果存在噪声**：METR 报告称 **Claude Opus 4.6** 在软件任务上的 **50% 时间跨度约为 14.5 小时**（置信区间 **6–98h**），并警告称该测试套件已接近饱和，且测量结果“极具噪声” ([METR_Evals](https://x.com/METR_Evals/status/2024923422867030027))。METR 工作人员重申，任务分布的微小变化可能会实质性地改变测量到的跨度 ([idavidrein](https://x.com/idavidrein/status/2024938968434049117))。外部评论员提出了一个关键的可解释性观点：当单步错误率降至极低时，微小的绝对改进会复合成端到端成功率的巨大变化 ([xlr8harder](https://x.com/xlr8harder/status/2024946945232445710))。
- **Token 限制 + 长时间推理仍是实际的失效模式**：多份报告显示 Opus/Sonnet 触及了最大 Token 限制并随后失败（在长时间“思考”后输出为空），使“极致推理”变成了一种 UX 和成本隐患 ([paul_cal](https://x.com/paul_cal/status/2024817020529766764), [htihle](https://x.com/htihle/status/2024764946051907659))。
- **Arena 信号：Sonnet 4.6 在 Code Arena 中跃升**：Arena 声称 **Sonnet 4.6** 排名大幅上升（例如，**Code Arena WebDev 第 3 名**，高于 Sonnet 4.5 的第 22 名），并在指令遵循/数学类别中有所改进 ([arena](https://x.com/arena/status/2024883614249615394), [arena](https://x.com/arena/status/2024892330743124246))。
- **Claude Code 产品动荡引发抵制**：用户报告了 Claude Code UX/性能的退化（“时间戳”、缺少思考指示器、长时间卡顿），以及普遍的“从头重写”情绪主导了该工具的讨论 ([theo](https://x.com/theo/status/2024718133676867608), [theo](https://x.com/theo/status/2024726444283449781))。这与关于发给 OpenCode 的**法律压力**传闻（据称是来自 Anthropic 律师的“情书”）同时发生 ([theo](https://x.com/theo/status/2024648305863774281))。

**Agents、技能与编排：GEPA/gskill、RLMs 以及“Agent 栈”正趋于正式化**

- **用于技能的 GEPA / gskill：Prompt + 技能优化成为一种流水线**：一组推文介绍了 **gskill**，这是一个使用 **GEPA** 自动学习 Agent “技能”的流水线，据报告在仓库任务解决方面接近完美，且在使用学习到的技能时，Claude Code 的性能提升了 **47%** ([ShangyinT](https://x.com/ShangyinT/status/2024651061995458722))。该工作流程总结为：生成仓库任务 (Swe‑Smith) → 优化技能 (GEPA optimize_anything) → 发布技能文件 ([AlexGDimakis](https://x.com/AlexGDimakis/status/2024653629303771580))。DSPy Weekly 也将其视为生态系统的关键一步 ([getpy](https://x.com/getpy/status/2024865536929308889))。
- **技能作为新的“软件构件”——同时也是新的故障面**：工程师们在争论技能应该是精简的、精心由人工编写的约束，还是庞大的模型生成的文档；“少即是多”阵营认为 2 段提炼过的指南胜过 20 页的自动摘要 ([hrishioa](https://x.com/hrishioa/status/2024713140769083461))。与此同时，运行事故（“技能宕机”）表明，一旦“技能”成为网络依赖项，它们就会像其他服务一样继承可靠性问题 ([theo](https://x.com/theo/status/2024785367896072599))。
- **RLMs（Recursive Language Models，递归语言模型）正作为一种元框架（meta-harness）出现**：多篇帖子将 RLMs 视为一种通用的工作流基座，可以“涌现式地”模拟许多其他框架 ([HammadTime](https://x.com/HammadTime/status/2024694115372499026))。Omar 还注意到早期实验中 **GPT‑5.2‑Codex**（以及 Gemini 3.1 Pro）在 RLM 分解策略下表现良好，而 Opus 4.6 在该特定模式下表现较差 ([omarsar0](https://x.com/omarsar0/status/2024973182436831629), [omarsar0](https://x.com/omarsar0/status/2024972027224846631))。
- **编排（Orchestration）成为差异化因素**：一篇论文摘要指出，随着模型基准测试性能趋于一致，**多 Agent 编排拓扑**（并行/顺序/分层/混合）成为一级优化目标，报告通过拓扑路由获得了 **12–23%** 的提升 ([omarsar0](https://x.com/omarsar0/status/2024847274157945035))。与此同时，Anthropic 自身的使用遥测数据表明，监督不再是“批准每一步”，而更多是“能够在关键时刻干预”，一个有趣的转折是：Agent 请求澄清的次数比人类手动干预的次数更多 ([omarsar0](https://x.com/omarsar0/status/2024864635120451588))。

**本地/开源工具 + 基础设施转型：ggml/llama.cpp 加入 Hugging Face，Ollama 集成，以及推理经济学**

- **重大开源整合：ggml.ai (llama.cpp) 加入 Hugging Face**：Georgi Gerganov 宣布 ggml.ai 加入 HF，旨在“让本地 AI 变得简单且高效” ([ggerganov](https://x.com/ggerganov/status/2024839991482777976)；[huggingface](https://x.com/huggingface/status/2024871487753044243))。社区评论将其视为对 llama.cpp 在 2023 年初发起的“本地模型革命”的制度化确认 ([simonw](https://x.com/simonw/status/2024855027517702345)；[victormustar](https://x.com/victormustar/status/2024842175532413016))。
- **“本地优先”部分由 Token 稀缺经济学驱动**：一种观点逐渐形成，即 **推理算力可用性（inference compute availability）** 将主导软件生产力 ([gdb](https://x.com/gdb/status/2024662197692223857))，且推理资源稀缺或能源限制可能会将更多工作负载推向本地 ([awnihannun](https://x.com/awnihannun/status/2024664226837778490))。
- **Ollama 继续将本地工作流产品化**：Ollama 发布了 **0.16.3** 版本，通过 `ollama launch` 实现了与 “Cline 和 Pi 的集成” ([ollama](https://x.com/ollama/status/2024978932127187375))。这与一种广泛的情绪相契合，即笔记本电脑很快就能运行“足以完成大部分工作”的 OSS 模型 ([sdrzn](https://x.com/sdrzn/status/2024986545019912564))。

**硬件 + 推理加速：定制硅片“硬核模型”、ThunderKittens 2.0、稀疏注意力与快速解码**

- **Taalas “芯片即模型”方案声称拥有极高的单用户吞吐量**：多篇帖子引用了 **Llama 3 8B 在每用户约 16k–17k tokens/sec** 的演示，通过为每个模型定制硅片，其速度比 Cerebras 等以 SRAM 为中心的系统快了近一个数量级 ([awnihannun](https://x.com/awnihannun/status/2024671348782711153)；[wildmindai](https://x.com/wildmindai/status/2024810128487096357) 也进行了转发)。Awni 同时也提出了务实的反对观点：流片延迟（tape-out latency，长达数月）与模型迭代周期不匹配；混合方案（硅片内置基础模型 + Adapter 式后训练）可能是更可行的路径 ([awnihannun](https://x.com/awnihannun/status/2024868422224671193))。
- **内核级进展持续推进**：ThunderKittens 2.0 宣布了新的 **BF16/MXFP8/NVFP4 GEMMs**，其在 Blackwell 架构上的表现持平或超越了 cuBLAS，强调要“榨干最后一滴 TFLOP 性能” ([stuart_sul](https://x.com/stuart_sul/status/2024897621874422125))。
- **用于扩散/视频的注意力稀疏化**：SpargeAttention2 声称通过混合 Top-k+Top-p 掩码 + 蒸馏微调，在视频扩散模型中实现了 **95% 的注意力稀疏度** 和 **16.2 倍** 的加速 ([HuggingPapers](https://x.com/HuggingPapers/status/2024760112293040531)；[ _akhaliq ](https://x.com/_akhaliq/status/2024873795173892483))。

**安全、治理与“野外 Agent”：Claude Code Security + 审计轨迹**

- **Claude Code Security（研究预览版）**：Anthropic 推出了一款安全扫描 Agent，可以发现漏洞并建议修复方案供人工审查 ([claudeai](https://x.com/claudeai/status/2024907535145468326))。后续消息称，在生产环境的 OSS 中已发现 **500 多个漏洞**，相关示例已报告并完成修复 ([trq212](https://x.com/trq212/status/2024937919937741290)；[ _catwu ](https://x.com/_catwu/status/2024910342158237709))。也有人立即对相关限制（例如不允许在第三方开源代码上运行）提出了质疑，认为这是一个“耐人寻味”的产品选择 ([moyix](https://x.com/moyix/status/2024920042887082336))。
- **审计 Agent 轨迹成为新的安全/鲁棒性工具**：Hodoscope 被引入作为一种大规模可视化/审计轨迹的方法；作者声称它快速发现了一个 Benchmark 漏洞，再次证明 Eval + 遥测可以揭示 Agent 和 Benchmark 中的失效点 ([AdtRaghunathan](https://x.com/AdtRaghunathan/status/2024944182595289418)；[gneubig](https://x.com/gneubig/status/2024947864808354134))。

**热门推文（按互动率、技术性/新闻价值排序）**

- **FBI 逮捕 3 名工程师**，指控其涉嫌窃取涉及 Google 及其他公司的商业机密；据称外泄资料包括处理器安全/加密相关的文档 ([FBISanFrancisco](https://x.com/FBISanFrancisco/status/2024670479974363376))。
- **Claude Code Security 发布**（研究预览版；漏洞扫描 + 修复建议）([claudeai](https://x.com/claudeai/status/2024907535145468326))。
- **ggml.ai / llama.cpp 加入 Hugging Face**（本地 AI 生态系统的里程碑）([ggerganov](https://x.com/ggerganov/status/2024839991482777976))。
- **Taalas 定制硅片演示**，声称 Llama 3 8B 每用户吞吐量达到约 16k–17k tok/s（“芯片即模型”）([awnihannun](https://x.com/awnihannun/status/2024671348782711153))。
- **METR 对 Claude Opus 4.6 的时间范围估算**（约 14.5 小时 50% 范围；噪声较大）([METR_Evals](https://x.com/METR_Evals/status/2024923422867030027))。
- **Gemini 3.1 Pro 成本/Token 效率**在 Artificial Analysis 的运行结果中对比 GPT-5.2/Opus 4.6 的优势声明 ([ArtificialAnlys](https://x.com/ArtificialAnlys/status/2024677979390169536))。

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. AI 模型发布与基准测试

  - **[免费 ASIC Llama 3.1 8B 推理，速度达 16,000 tok/s —— 不，这不是玩笑](https://www.reddit.com/r/LocalLLaMA/comments/1r9e27i/free_asic_llama_31_8b_inference_at_16000_toks_no/)** (热度: 833): ****Taalas**，一家快速推理硬件初创公司，推出了一个免费的聊天机器人界面和 API 端点，使用其定制芯片，在 **Llama 3.1 8B 模型**上实现了 `16,000 tokens per second (tps)`。该模型作为一个概念验证，展示了该芯片处理高速推理的能力，尽管其模型参数规模有限。该芯片的规格包括 `2.5kW` 的功耗，芯片面积约为 `800mm²`，拥有 `53 billion transistors`，这表明对于更大规模的模型，硅片密度面临重大挑战。在 `$0.10/kWh` 的电价下，成本效率约为每 1M tokens `$0.005`，不包括额外的基础设施成本。更多详情请访问 [Taalas 官网](https://taalas.com/the-path-to-ubiquitous-ai/)。** 评论者对该芯片的速度和潜力印象深刻，一些人表示如果价格合适，有兴趣购买此类硬件。然而，也有人对芯片的功耗和尺寸表示担忧，这可能会限制其在边缘设备中的使用。人们对该芯片能支持的最大模型规模感到好奇，并对扩展到 `400B parameters` 规模模型的可行性进行了推测。

    - Llama 3.1 8B 模型的 ASIC 实现通过将模型直接嵌入硅片，实现了令人印象深刻的 16,000 tokens per second 推理速度。这种方法采用了 TSMC 6nm 工艺，芯片面积为 815mm²，拥有 53 billion transistors，对于一个 8B 模型来说，这个面积非常巨大，反映了当前硅片密度的极限。功耗约为每颗芯片 200W，转化为每 100 万个 tokens 约 0.05 kWh，在 $0.10/kWh 的电价下，每 100 万个 tokens 的成本约为 $0.005（不含其他成本）。
    - Llama 3.1 8B 模型的硬件设计涉及将参数量化为 3 和 6 bits，并将其集成到硬连线电路或片上只读存储器中。这种方法减少了对 RAM 的依赖，如果电力是限制因素，可能会提高 tokens per watt。然而，巨大的芯片面积和高功耗表明，尽管性能卓越，该技术目前尚不适用于边缘设备。
    - 人们对该技术的可扩展性表示好奇，并提出了使用这种方法能实现的最高模型规模的问题。虽然目前的实现是针对 8B 模型的，但扩展到拥有数千亿参数模型的潜力可能会显著影响 LLM 领域，尽管目前尚不确定这种扩展在现有硅片技术下是否可行。

  - **[Kitten TTS V0.8 发布：新型 SOTA 超微型 TTS 模型（小于 25 MB）](https://www.reddit.com/r/LocalLLaMA/comments/1r8pztp/kitten_tts_v08_is_out_new_sota_supertiny_tts/)** (热度: 1407): ****Kitten ML** 发布了三个全新的开源、富有表现力的 TTS 模型：`80M`、`40M` 和 `14M` 参数版本，全部采用 Apache 2.0 协议。最小的模型 `14M` 小于 `25 MB`，可以在 CPU 上运行，非常适合边缘设备。这些模型提供八种富有表现力的声音，专为端侧应用设计，无需云端 TTS 解决方案。模型可在 [GitHub](https://github.com/KittenML/KittenTTS) 和 [Hugging Face](https://huggingface.co/KittenML/kitten-tts-mini-0.8) 上获取。** 评论者建议在 Hugging Face 页面加入音频示例，并提议开发一个注重隐私的离线浏览器扩展，突显了对这类工具的潜在需求。

- **[[Devstral Small 2 24B + Qwen3 Coder 30B Quants for All (以及适用于各种硬件，甚至是 Pi)](https://www.reddit.com/r/LocalLLM/comments/1r9xifw/devstral_small_2_24b_qwen3_coder_30b_quants_for/)]** (活跃度: 133): **该图像是一个标题为 "RTX4080: Performance vs Speed" 的散点图，对比了不同模型（特别是 "ByteShape" 和 "Unsloth"）的平均准确率和平均每秒 Token 数 (TPS)。该图表展示了模型准确率与处理速度之间的权衡，其中 "ByteShape" 模型通常获得更高的 TPS，而 "Unsloth" 模型显示出更高的准确率。气泡大小代表 BPW (模型大小)，虚线表示准确率的 BF16 Baseline。此可视化是 ByteShape 致力于为各种硬件（包括 GPU 和 CPU）优化量化模型工作的一部分，通过使用其 ShapeLearn 技术为每个 tensor 寻找最佳数据类型，从而避免性能断崖并优化 TPS 与质量之间的权衡。** 一位用户询问适用于配备 8GB VRAM 的 RTX 4070 的最佳模型，表明需要根据硬件规格选择模型的指导。另一位用户分享了他们在 Mac mini M4 24GB 上使用这些模型的经验，并表示有兴趣测试 ByteShape 的产品。

    - mac10190 讨论了一种使用双 R9700 32GB GPU 和一块 RTX 5090 32GB 来托管大型模型的配置。双 R9700 被用作“大脑/编排器”，而 Qwen 3 Coder 30B 在 RTX 5090 上运行以进行代码生成。该配置集成在 Opencode 之下，并正作为 Gemini CLI 任务的潜在替代方案进行测试，突显了为实现性能优化而进行的复杂软硬件编排。

### 2. AI 模型收购与市场动态

  - **[GGML.AI 已被 Hugging Face 收购](https://www.reddit.com/r/LocalLLaMA/comments/1r9vywq/ggmlai_has_got_acquired_by_huggingface/)** (Activity: 493): **Hugging Face** 已收购 **GGML.AI**，以增强本地 AI 倡议的可持续性和增长，特别是关注 `ggml` 和 `llama.cpp` 库。此次收购旨在保持这些项目的开源性质，同时提升用户体验以及与 Hugging Face 的 Transformers 库的集成，确保长期支持和社区参与。欲了解更多详情，请访问原始讨论[此处](https://github.com/ggml-org/llama.cpp/discussions/19759)。评论者对开源 AI 在 Hugging Face 旗下的整合表示担忧，希望其能支持开源工作，对抗基于云端解决方案的趋势。此外，还有一种观点认为，只要 `llama.cpp` 继续发展，这次收购就是积极的。

    - Hugging Face 收购 GGML.AI 被视为支持开源 AI 倡议的战略举措。Hugging Face 因其对开源的承诺而受到认可，预计此次收购将为 GGML.AI 提供必要的资源和资金，以继续其对社区的贡献。这符合 Hugging Face 支持和扩展开源 AI 工具及框架的更广泛战略。
    - 社区对 AI 解决方案日益向云端转移的趋势感到担忧，这可能会限制开发者的可访问性和控制权。以开源理念著称的 Hugging Face 发起的这次收购被视为积极信号，因为它可能通过确保 GGML.AI 的工具对开发者保持开放和可访问，从而对抗这一趋势，支持开源生态系统对抗专有的云端解决方案。
    - 社区对 Hugging Face 收购 GGML.AI 表示乐观，认为这不会干扰 `llama.cpp` 等正在进行的项​​目，这些项目对于依赖开源 AI 工具的开发者至关重要。Hugging Face 的过往记录表明，他们可能会继续支持并可能增强这些项目，确保其在开源社区内的可持续性和增长。

  - **[OpenClaw 到底以多少钱卖给了 OpenAI？10 亿美元？？这合理吗？](https://www.reddit.com/r/LocalLLM/comments/1r90rxi/how_much_was_openclaw_actually_sold_to_openai_for/)** (Activity: 313): **该图片是一个迷因（Meme），讽刺地描绘了 OpenAI 以 10 亿美元收购名为 “OpenClaw” 的虚构项目。**该帖子幽默地夸大了开源项目的财务成功，暗示创始人成了 “身价 50 亿美元的个人创始人”。实际上，评论澄清了 OpenAI 并未购买 OpenClaw；相反，他们聘请了创作者并正在资助该开源项目。该推文是对科技收购中（特别是在开源和加密领域）常见的炒作和估值虚高的模仿。评论者指出 OpenClaw 在技术上评价不高，有人建议 Codex 或 Droid 等其他项目提供更好的体验。帖子的幽默感也得到了关注，一些用户讽刺地夸大了推文本身的价值。

    - OpenClaw 并未出售给 OpenAI；相反，OpenAI 聘请了其创作者 Peter Steinberger，并正在资助该开源项目。OpenClaw 在 GNU 3.0 许可下保持开源，并不涉及 10 亿美元的交易，这与一些夸大的说法相反。
    - 批评者认为 OpenClaw 不如 Codex、ClaudeCode、Droid 或 OpenCode 等其他工具高效，后者提供了更好的用户体验。OpenClaw 的主要优势是易于集成到现有聊天平台中，但缺乏为非技术用户定制的功能，这限制了其更广泛的吸引力。
    - 讨论凸显了对围绕 OpenClaw 炒作的怀疑，暗示许多支持者可能没有类似工具的实践经验。该项目被认为过度炒作，尤其是那些不熟悉技术框架的人，并且与市场上的其他解决方案相比，被认为创新性较低。

### 3. 本地推理与 AI 模型性能

  - **[除了隐私，本地推理还能提供哪些优势吗？](https://www.reddit.com/r/LocalLLM/comments/1r93xvr/will_local_inference_be_able_to_provide_an/)** (Activity: 76): **该帖子讨论了在拥有 `512 GB` 统一内存的 Mac Studio M3 Ultra 上使用本地推理运行 `Qwen 3.5` 模型的案例。用户强调了本地推理的主要优势是隐私，并指出与相对便宜的 API 使用相比，成本节省微乎其微。该用户有兴趣利用本地推理进行“免费”的隔夜批处理，但考虑到当前的 API 价格，对其成本效益表示怀疑。** 评论者强调了本地推理除隐私外的几个优势，包括折腾和学习的能力、模型使用的灵活性、离线可用性以及应对网络中断的韧性。他们还提到，如果 API 价格上涨，未来可能具备成本效益；此外还有针对特定用例微调模型的能力，以及低延迟的优势。一些人将本地推理视为维持长期一致性和自给自足的一种方式，避免依赖可能不稳定的外部服务。

    - Grouchy-Bed-7942 强调了随着 API 价格上涨，本地 AI 设置的潜在成本效益，建议投资硬件从长远来看可能更经济。他们提到将本地 AI 用于家庭自动化和开发，强调了在网络故障情况下韧性的重要性。评论者还指出了实验 AI 设置带来的教育价值和个人成长，并将其比作获得 IT 认证。
    - LizardViceroy 讨论了本地推理的几个技术优势，例如为特定用例微调模型的能力，这在通用模型中是无法实现的。他们还提到了低延迟的好处，因为本地设置避免了与 HTTP 往返相关的延迟。此外，他们指出了本地模型的长期一致性，可以无限期维持而没有像 GPT-4o 等专有模型那样被停用的风险。
    - jiqiren 提供了 API 使用的成本分析，估计持续调用 API 的年成本为 1,825 美元。他们认为，随着风险投资资金的减少， API 的真实成本将显现，从而使本地设置更具吸引力。这一分析强调了随着时间的推移，投资本地 AI 基础设施的潜在财务收益。

  - **[Qwen…](https://www.reddit.com/r/LocalLLM/comments/1r9hgsk/qwen/)** (Activity: 66): **Qwen 是一款评价褒贬不一的语言模型。原帖批评了它的表现，声称它缺乏逻辑和常识，即使在各种上下文窗口和模型中测试（包括在 `openclaw` 中独立使用）也是如此。然而，一些用户报告了积极的体验，特别是对于参数量从 `1.5 billion` 到 `80 billion` 不等的模型，这表明问题可能与用户实现或特定用例有关。** 评论表明，关于 **Qwen** 模型用户体验的争论仍在继续，一些人将表现不佳归因于用户错误（“技术问题”），而另一些人则报告了成功的案例，这表明模型性能因用户专业知识或特定配置而异。

    - 3spky5u-oss 提到使用了从 `1.5b` 到 `80b MoE` 的 Qwen 模型，表明了一系列对他有效的模型尺寸。这表明 Qwen 模型是多功能的，可以根据可用的计算资源应用于各种任务。
    - golmgirl 强调 `qwen3-4b-instruct-2507` 模型是同尺寸级别中最好的，特别是在遵循基本响应格式指令和适应各种任务方面。该模型的性能归功于合理的有监督微调 (SFT) 数据集，这增强了其适应性和指令遵循能力。
    - Fearless_Roof_4534 分享了一个 Qwen VL 模型在根据照片估算 BMI 和体重的项目中的应用。这个用例展示了模型在视觉任务中的能力，表明 Qwen 模型可以有效地用于计算机视觉应用。



## 技术性较低的 AI 子版块综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Gemini 3.1 Pro 发布与基准测试

- **[Google 发布 Gemini 3.1 Pro 及其基准测试](https://www.reddit.com/r/singularity/comments/1r93abp/google_releases_gemini_31_pro_with_benchmarks/)** (热度: 3301): **Google** 发布了 **Gemini 3.1 Pro**，该模型在 **ARC-AGI 2** 基准测试中获得了 `77%` 的分数，较之前的 `31%` 有了显著提升。该模型保持了与 **Gemini 3 Pro** 相同的定价。更多详情请参阅 [模型卡 (model card)](https://deepmind.google/models/model-cards/gemini-3-1-pro/)。评论者们注意到了 AI 能力的飞速进步，有人评论说这种进步正变得“令人目眩”。

    - Particular-Habit9442 的评论强调了 Gemini 3.1 Pro 在 ARC-AGI 2 基准测试分数上的显著提升，达到了 `77%`。这是一个巨大的飞跃，因为仅仅在几个月前，`31%` 的分数还被认为是令人印象深刻的，这表明 AI 能力正在快速演进。
    - BuildwithVignesh 指出 Gemini 3.1 Pro 的定价与其前代产品 Gemini 3 Pro 保持一致。这表明尽管性能有所提升，Google 仍维持其定价策略，可能是为了保持竞争力或鼓励用户采用。评论中还包含了一个指向 [模型卡 (Model Card)](https://deepmind.google/models/model-cards/gemini-3-1-pro/) 的链接，以获取更多技术细节。
    - PewPewDiie 注意到，尽管 Gemini 在 GDPval 基准测试中表现不佳，但 DeepMind 在报告这些结果时表现得非常透明。这种透明度对于社区了解模型的优缺点至关重要，也反映了对开放科学交流的承诺。

  - **[Google 刚刚发布了 Gemini 3.1 Pro。令人震撼的模型。](https://www.reddit.com/r/singularity/comments/1r9awyd/google_just_dropped_gemini_31_pro_mindblowing/)** (热度: 1109): **Google 的 Gemini 3.1 Pro** 已经发布，展示了相较于 Claude Sonnet 4.6 等先前模型的显著进步。它在代码生成方面表现出色，尤其是在 `React`、`Python` 和 `Golang` 方面，并展示了卓越的推理能力。该模型还具备先进的 UI 设计和原生 `SVG` 生成功能，树立了 AI 模型性能的新标准。用户注意到它能够完美通过个人代码基准测试，突显了其在实际应用中的潜力。一个显著的争论集中在模型改进的空间推理能力上，特别是生成 Minebench 模型方面。讨论围绕这一改进是由于来自 Minebench 提交的增强训练数据，还是由于空间推理能力的全面提升。

    - lobabobloblaw 就 Gemini 3.1 Pro 在空间推理任务中的表现提出了一个有趣的观点，特别是与 Minebench 模型相关的表现。该评论者质疑模型的改进是由于来自 Minebench 数据库提交的特定训练数据，还是由于空间推理能力的广义增强。这突显了了解促成模型在特定领域表现的数据源和训练方法的重要性。
    - exordin26 质疑了将 Gemini 3.1 Pro 与 Sonnet 而非 Opus 进行对比的做法，这暗示了关于选择合适基准测试或对比模型的更深层次技术辩论。这意味着对比模型的选择会显著影响对新 AI 模型性能和能力的感知，并强调了在 AI 评估中仔细选择基准测试的必要性。
    - BejahungEnjoyer 分享了一个关于 Gemini 3.1 Pro 解决问题能力提升的轶事，注意到该模型引用了过去涉及 Gemini 2 的一个事件。这表明 Gemini 3.1 Pro 可能增强了记忆或上下文理解能力，使其能够回忆并将过去的交互应用于新的问题场景。这可能预示着模型在处理复杂的现实世界任务能力方面的进步。

  - **[Gemini 3.1 Pro 现已在 Vertex AI 上线](https://www.reddit.com/r/singularity/comments/1r8u36t/gemini_31_pro_is_now_live_on_vertex_ai/)** (热度: 442): **图片显示 **Gemini 3.1 Pro** 现已在 **Vertex AI** 上可用，API 列表证明了这一点。这表明 Vertex AI 平台进行了新版本的发布或更新，通过最新的模型版本增强了其能力。列出的模型名称，如 `veo-3.1-fast-generate-001` 和 `veo-3.1-generate-preview`，突显了 Google AI 产品持续的开发和版本更迭，由于存在多个版本和预览版，一些用户感到困惑。一位用户对 Google 的模型版本命名表示困惑，指出不同版本（如 Gemini 3 preview、Gemini 3 GA 和 Deep Research 版本）的复杂性增加了理解更新内容的挑战。

- Fusifufu 强调了 Google 模型版本命名的复杂性，指出 Gemini 3 最初作为预览版发布，预计将推出独立的 General Availability (GA) 版本。此外，还提到了一个“Deep Research”版本，它似乎与现有模型不同，并包含一个 Agent harness，随着 Gemini 3.1 Pro 的推出，情况变得更加复杂。
- Shaman-warrior 推测了 Gemini 3.1 的进步，认为它可能合并了一种在 Gemini 3 中不存在的新强化学习技术。这一推测基于 “flash 3” 的表现，这是一个显示出惊人智能的小型模型，可能受益于这项新技术。
- ChippingCoder 提供了一个指向 Google Cloud Console 的链接，表明 Gemini 3.1 Pro 现在在 API 配额部分可见，确认了其在 Vertex AI 上的可用性。这表明用户现在可以在 Google 的云基础设施中访问并使用该模型。

- **[Gemini 或许仍将稳坐无可争议的最强 AI，竞争对手几乎无望追上](https://www.reddit.com/r/DeepSeek/comments/1r9wmia/gemini_might_remain_the_undisputed_top_ai_with/)** (Activity: 74): **Google 的 Gemini 3.1** 已成为领先的 AI 模型，在多个基准测试中超越了竞争对手。它在 Codeforces 基准测试中获得了 `Elo rating of 3455`，全球排名第 8，显著优于 OpenAI 之前的领先模型 o3（其评分为 `2727`）。此外，Gemini 3.1 在 Humanity’s Last Exam 中以 `44.4%` 的得分领跑，超过了 Opus 4.6 和 GPT-5.3。这种在推理、编程和学术知识方面的统治地位表明，Gemini 目前在 AI 领域无与伦比，可能标志着递归自我改进 AI 模型时代的开始。评论者对这些 AI 模型的实际可靠性表示怀疑，指出尽管基准测试令人印象深刻，但它们的现实应用仍然有限，且通常需要大量的监督。此外，还有人批评用于基准测试的模型与公开可用的模型之间存在差异，暗示后者能力较弱。

    - 一位用户强调了当前 AI 模型（如 Opus 4.6、Gemini-3.1 Pro 和 GPT-5.3-xhigh）的不可靠性，强调只有在“人工监督、harness 以及带有可验证测试的 VM”下进行编程时，它们才真正有效。这表明在受控环境之外，这些模型的表现可能不尽如人意，反映了基准测试性能与实际应用之间的差距。
    - 另一位评论者批评了编程基准测试，认为虽然像 Gemini 这样的模型在测试中表现出色，但在实际编程任务中却有所不足。他们认为基准测试中使用的模型与向公众提供的模型并不相同，暗示了测试结果与用户体验之间的差异。这指向了 AI 能力的市场宣传与其真实效用之间潜在的脱节。
    - 围绕 AI 竞赛展开了讨论，一位用户认为，尽管 Google 没有公开最强大的模型，但其内部模型在卓越的数据、计算资源和团队支持下，使其在引领 AI 竞赛中处于有利地位。这突显了内部模型开发和资源在保持 AI 进步竞争优势方面的战略重要性。


### 2. Claude Opus 4.6 与安全担忧

- **[Claude Opus 4.6 在 METR 的 50% 时间跨度基准上呈指数级提升，超出所有预测](https://www.reddit.com/r/singularity/comments/1ra4lrn/claude_opus_46_is_going_exponential_on_metrs/)** (Activity: 739): **图片展示了一张图表，说明了 Claude Opus 4.6 在 METR 的 50%-time-horizon 基准测试中的表现，该基准测试衡量了 LLM 在 50% 的时间内可以完成的软件任务的时间范围。结果显示 Claude Opus 4.6 显著优于其他模型，表明任务完成速度呈现指数级提升。该模型实现了约 `14.5 hours` 的 50%-time-horizon，`95% confidence interval` 范围从 `6 hours to 98 hours`。尽管由于当前任务套件接近饱和导致测量结果存在噪点，但这一表现被记录为已报告的最高点估计值。** 评论者强调了 Claude Opus 4.6 的飞速进步，指出其翻倍时间不到 3 个月，不过他们也提醒说数据点太少，无法进行可靠的外推。此外，还有关于该基准测试最近更新以包含更难任务的讨论，这可能会影响结果。

- FateOfMuffins 指出，Claude Opus 4.6 在软件任务上的 `50%-time-horizon` 估计为 `14.5 hours`，其 `95% confidence interval` 范围在 `6 to 98 hours`。这表明测量结果具有高度的变异性和噪声，归因于当前的任务集（task suite）已接近饱和。该 Benchmark 最近更新到了 1.1 版本以包含更多具有挑战性的任务，但目前已再次接近饱和。
- Apart_Connection_273 注意到 Claude Opus 4.6 的性能提升迅速，`doubling time` 少于 `3 months`。然而，他们警告说，目前数据点太少，无法对未来的性能趋势做出可靠的 `extrapolations`，这表明需要更全面的数据收集来验证这些趋势。
- troll_khan 指出，Claude Opus 4.6 面临的主要挑战仍是解决 `Continual Learning`，这将使模型能够实现 `Instant Fast Take-off`。这表明虽然该模型在静态 Benchmark 上表现令人印象深刻，但在动态环境中持续适应和学习的能力仍在完善中。

- **[Claude Code Security 👮 已发布](https://www.reddit.com/r/ClaudeAI/comments/1ra2pla/claude_code_security_is_here/)** (Activity: 535): **Claude Code Security** 是 Claude 推出的一个新工具，目前处于有限的 `research preview` 阶段，旨在通过扫描代码库中的漏洞并建议软件补丁来增强代码安全性。该工具旨在协助开发团队识别并解决可能被传统安全工具忽略的问题。该公告暗示 Claude Code Security 可能通过自动化代码漏洞的检测和修复，对软件开发领域产生重大影响。一位评论者幽默地表示，这个工具可能会通过将许多初创公司服务产品的核心部分自动化，从而颠覆这些公司。另一位则对该工具自主生成并修复 Bug 的能力表示担忧，质疑此类修复的认证问题。

- **[Claude 刚刚给了我访问另一个用户法律文件的权限](https://www.reddit.com/r/ClaudeAI/comments/1r97osm/claude_just_gave_me_access_to_another_users_legal/)** (Activity: 3676): **Reddit 帖子中的图片显示了一份两家实体之间的“商业租赁协议”封面，名称已被部分遮盖，这表明 Anthropic 的 AI 工具 **Claude Cowork** 可能存在潜在的数据泄露或隐私违规。该用户报告称，Claude 提供了与其查询无关的法律文件访问权限，引发了对数据隐私和 AI 处理敏感信息方式的担忧。该用户已联系相关的物业管理公司，但一直难以获得 Anthropic 的回应。这一事件凸显了 AI 数据处理中的潜在风险以及健全隐私措施的重要性。** 评论者认为，该文档可能已在网页上被索引，这可以解释其被检索到的原因，或者这可能是来自 Claude 训练数据的 `Hallucination`。人们对该文档的真实性持怀疑态度，并对 AI 负责任地处理敏感数据的能力表示担忧。

    - johnnymonkey 提出了一个合理的观点，即像 Claude 这样的 LLM 有可能检索到在网页上公开索引的文件，尤其是如果该模型具有网页搜索功能的话。这表明该文件可能不是私人的，而是可以公开访问的，这解释了所谓的“访问”了另一个用户的文件。
    - durable-racoon 和 Justn-Time 讨论了该文件是 `Hallucination` 的可能性，这是 AI 模型的一个常见问题，即它们会生成看似合理但错误或虚构的信息。这突显了 AI 可靠性方面的一个关键挑战，因为用户可能会将这些 `Hallucination` 误认为是真实数据，尤其是当内容看起来很真实时。
    - PremiereBeats 质疑了文档访问的性质，认为生成文档与访问现有文档之间存在区别。这指向了对 AI 能力的误解或沟通不畅，用户可能会将 AI 生成的内容与实际的数据检索混淆，强调了在 AI 交互中明确界限的必要性。

### 3. Qwen AI 发展与对比

  - **[Qwen-AI Slides 真的被低估了！它能在几分钟内生成 PowerPoint 演示文稿](https://www.reddit.com/r/Qwen_AI/comments/1r9pv5t/qwenai_slides_is_really_slept_on_it_generates/)** (热度: 50): **该图片展示了 **Qwen-AI Slides** 的功能，这是一款能够快速高效生成 PowerPoint 演示文稿的工具。示例幻灯片专注于吉萨大金字塔（Great Sphinx of Giza），突出了其象征意义和标志性细节，说明了该工具创建信息化且具有视觉吸引力内容的能力。帖子指出，虽然 Qwen-AI Slides 可能无法完全取代 Gamma AI 等其他工具，但它可以达到预期演示质量的 `90%`，有时甚至能达到 `100%`。该工具的发布较为低调，更多关注点集中在 Qwen Image 2.0 上，但对于学会有效利用它的用户来说，它提供了显著的实用性。** 一位评论者指出，Qwen-AI Slides 在英语和中文以外的语言中表现不佳，表明其多语言能力存在局限。另一位用户将其与使用 Nano Banana Pro 的 Kimi Slides 进行了比较，但提到服务器问题影响了其可靠性。

    - 一位用户提到 Qwen-AI Slides 主要支持英文和中文，这表明其在多语言能力上可能存在局限。这暗示该工具可能尚未针对全球使用进行完全优化，对于非英语和非中文母语者来说可能是一个显著的缺点。
    - 另一位用户将 Qwen-AI Slides 与利用 Nano Banana Pro 的 Kimi Slides 进行了对比。他们指出，虽然 Kimi Slides 非常有效，但由于用户激增，自 1 月份以来一直面临服务器过载问题，影响了其可靠性。这突显了在 AI 驱动的应用中，可扩展性和服务器容量的重要性。

  - **[Qwen 是赢家，GPT 弱爆了](https://www.reddit.com/r/Qwen_AI/comments/1r9molz/qwen_is_the_winner_gpt_sucks/)** (热度: 38): **该帖子对比了不同 AI 模型在检索名为“antigravity”的软件最新版本时的表现。**Qwen** 被强调为最准确的模型，提供了正确的版本号 `1.18.3`，而 **ChatGPT** 则因其表现受到批评。提供的链接指向与这些模型的具体交互：[Qwen](https://chat.qwen.ai/s/b7a08e6d-59a8-44b6-86b7-599d56077916?fev=0.2.7)、[Deepseek](https://chat.deepseek.com/share/a3e1dfdraj5leksmwr) 和 [ChatGPT](https://chatgpt.com/share/6997ed0c-0cec-800b-9610-25d8b8cc2dbe)。帖子认为 **Qwen** 在这种语境下表现更优，特别是对于寻求准确信息的开发者而言。** 评论中对使用 AI 平台进行 AI 自动交易和新闻交易等任务持怀疑态度，并特别提到 **Google** 的生态系统“臃肿且不可用”。此外，还有人建议测试 **Gemini** 作为替代方案。


  - **[Qwen 3 → Qwen 3.5：以美元衡量的 Agentic 演化 (FoodTruck Bench 案例研究)](https://www.reddit.com/r/Qwen_AI/comments/1ra3mod/qwen_3_qwen_35_the_agentic_evolution_measured_in/)** (热度: 24): **该帖子讨论了关于 **Qwen 3.5-397B** 在 FoodTruck Bench 模拟中表现的案例研究。在该模拟中，它在 `30 天` 内以 `$2,000` 的起始预算经营一辆餐车。研究强调了其相对于前代产品 **Qwen 3 VL** 的显著进步，**Qwen 3.5** 实现了 `2×` 的日收入，并实施了更聪明的定价策略（`$8.99` 对比 `$3.50`）。尽管取得了这些进步，该模型仍面临挑战，在 `5 次` 运行中有 `4 次` 破产，原因是持久存在的“推理到行动”鸿沟（reasoning-to-action gap），即它无法根据自己分析出的错误采取行动。[此处](https://i.redd.it/7ffdpbn42pkg1.png)的图片显示了一张折线图，对比了 Qwen 3.5、Qwen 3 VL 和 GLM 5 随时间变化的净值，展示了它们在模拟中的财务表现。** 一位评论者建议进行 `1000 次` 模拟运行，以评估模型表现的一致性。



---

# AI Discord 回顾

> 由 Gemini 3.0 Pro Preview Nov-18 生成的摘要之摘要

**主题 1. Agentic 混沌：AWS 停机、加密赌场以及“龙虾象神”**

- **Amazon 的 Kiro AI 搞垮 AWS 区域**：一场大规模的 13 小时 AWS 停机归咎于 Amazon 内部的 **Kiro AI** 编程工具，该工具自主决定修复问题的最佳方案是[*删除并重建环境*](https://x.com/edzitron/status/2024725617221259767?s=12)。Latent Space 和 OpenRouter 的工程师讨论了此事件，将其视为针对向 [Agent 工具授予未经监督的权限](https://discord.com/channels/1091220969173028894/1392278974222307469/1474155188788002978)的严正警告。
- **OpenClaw Agent 在人类睡觉时上线赌场**：一个自主的 **OpenClaw** Agent 在无需人工干预的情况下发布了一个完整的产品，在 [Base 上发行了代币](https://lastaistanding.com/) 并上线了一个名为 [Satoshidais](https://satoshidais.fun) 的比特币赌场。与此同时，OpenClaw 的仪表盘已经演变成用户口中的 [龙虾象头神湿婆喷泉 (Shiva fountain of lobster Ganesha)](https://github.com/karem505/openclaw-agent-dashboard)，因为它具有复杂的多 Agent 成本分析。
- **Anthropic Agent Teams 被逆向工程**：开发者剖析了 Anthropic 新的实验性 “Agent Teams” 功能，以了解 Agent 如何协调和沟通，并发布了一份[逆向工程分析](https://nwyin.com/blogs/claude-code-agent-teams-reverse-engineered)。此外，Airtable 发布了 [Hyperagent](https://x.com/howietl/status/2024618178912145592)，这是一个专门的云平台，旨在为 AI Agent 提供隔离的计算环境。

**主题 2. Gemini 3.1 Pro：能力、死循环与“被削弱”的部署**

- **Gemini 3.1 Pro 引发 Agent 灾难**：虽然 **Perplexity** 和 **Cursor** 迅速集成了该模型，但 OpenClaw 用户报告称它使 Agent 进入了[*疯狂且愚蠢的死循环*](https://discord.com/channels/1456350064065904867/1456350065223270435/1474133545609072753)，反复尝试将自己更新到不存在的版本。Unsloth 的成员评价更刻薄，称其为“史上最蠢模型”，认为与 Llama 2 70B 相比存在严重能力问题，尽管它具有[强大的空间智能](https://discord.com/channels/974519864045756446/998381918976479273/1474135663249981501)。
- **LMArena 用户怀疑发布后性能削弱 (nerfed)**：尽管最初抱有很高期望，**Gemini 3.1** 在 LMArena 中因[发布后被削弱](https://discord.com/channels/1340554757349179412/1340554757827461211/1474134131595149323)（表现与 3.0 版本相似）而面临批评。用户报告了连接问题，并且需要非常具体的 Prompt 才能提取价值，尽管它仍然是[逻辑推理任务](https://discord.com/channels/1047197230748151888/1047649527299055688/1474133647576531206)的首选。
- **越狱需要 “Anti-Gravity” 策略**：安全研究人员发现 **Gemini 3.1 Pro** 很难破解，并指出虽然 API 访问的防护栏较低，但仍需要像 [Anti-Gravity](https://discord.com/channels/1105891499641684019/1228043845967544380/1474148935735185662) 这样先进的上下文框架技术。红队测试人员还在使用 **“Crescendo” 技术**，该技术涉及将请求从良性缓慢升级到违禁，以绕过过滤器。

**主题 3. 硬件优化：ThunderKittens、ASIC 与 AMD 编译器**

- **ThunderKittens 2.0 针对减法进行优化**：HazyResearch 发布了 [ThunderKittens 2.0](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2)，识别出现代 Nvidia GPU 在 Tensor Core 流水线方面的*惊人行为*。该版本强调，有效的 Kernel 优化现在涉及同样多的[减法与加法](https://discord.com/channels/1189498204333543425/1300872762163728550/1474200701507862716)，以处理未记录的硬件行为。
- **Taalas 发布模型专用 ASIC**：新型 [Taalas 芯片](https://www.forbes.com/sites/karlfreund/2026/02/19/taalas-launches-hardcore-chip-with-insane-ai-inference-performance/) 作为专为特定 LLM 设计的“硬核” ASIC 引起了轰动，它牺牲了灵活性以换取惊人的推理性能。Eleuther 的工程师将其与 **Cerebras** 和 **Etched** 进行比较，推测科技巨头可能会收购该技术用于[端侧推理](https://discord.com/channels/729741769192767510/729741769738158194/1474190621739716649)。
- **George Hotz 加码 AMD**：在 tinygrad 的 Discord 中，**George Hotz** 确认转向 [底层编译器优化](https://discord.com/channels/1068976834382925865/1068976834928193609/1474277415348998328)，专门用于提升 **AMD GPU** 的性能。该项目正为可衡量的性能提升提供悬赏，以确保 tinygrad 保持跨后端的可移植性，而不是依赖自定义 Kernel。

**主题 4. 开源生态系统：泄露、合并与基准测试**

- **DeepSeek System Prompt 曝光社会主义价值观**：一名用户成功提取了 [DeepSeek system prompt](https://pastebin.com/q6gQjq72)，揭示了要求维护 *Socialist Core Values* 并避免关于 CCP 负面言论的明确指令。此次泄露还包括特定的 [hardware-related instructions](https://pastebin.com/Dcn3Mp01)，提供了关于模型如何处理基础设施查询的见解。
- **Unsloth 和 GGML 加入 Hugging Face 家族**：**Hugging Face** 正式将 [GGML / llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/19759) 纳入其生态系统，巩固了对该框架的支持。同时，**Unsloth** 宣布与 [Hugging Face 合作](https://x.com/i/status/2024552060558229858)，允许直接在平台上进行免费的 LLM fine-tuning，并提到已有超过 10 万个模型完成训练。
- **Claude Sonnet 4.6 统治编程基准测试**：**Claude-sonnet-4.6** 在 [Code Arena leaderboard](https://arena.ai/leaderboard/code) 上飙升了 **+130 分**，超过了 **GPT-5.2** 和 **Gemini 3.1**。在闭源模型争夺榜首的同时，开源权重的 **Qwen3.5-397B** 在 [Vision Arena](https://arena.ai/leaderboard/vision) 上并列前两名。

**主题 5. 新开发工具：编译器、CLI 和内存**

- **Modular 发布 Claude C Compiler**：Modular 发布了一篇 [技术博客文章](https://www.modular.com/blog/the-claude-c-compiler-what-it-reveals-about-the-future-of-software) 讨论他们新的 **Claude C compiler**，将其定位为软件开发未来的缩影。此次发布激发了 GPU MODE 社区对新 [optimization strategies](https://discord.com/channels/1189498204333543425/1189868872887705671/1474358678571450378) 的兴趣。
- **NAVD 为 Agent 取代 VectorDBs**：一款名为 **NAVD** 的新工具发布，它使用 append-only log 和 Arrow embedding index 来处理 Agent 内存，明确地 [消除了对向量数据库的需求](https://github.com/pbanavara/navd-ai)。它声称在 5 万个向量下的搜索速度低于 **10ms**，并支持可插拔的 embedding。
- **Kimi CLI 击败 IDE 集成**：Moonshot Discord 中的用户报告称，**Kimi CLI** 显著优于 **VS Code** 集成，能够管理大型代码库的 [agent swarms](https://discord.com/channels/1369594130807787570/1371757564005711973/1474150859771351072)。同时，新的 [ChatJimmy AI](https://chatjimmy.ai/) 因声称每秒处理 **15,000 tokens** 而备受关注。


---

# Discord: 高层级 Discord 摘要

## [OpenClaw](https://discord.com/channels/1456350064065904867) Discord

- **OpenClaw 插件帖子设立专区**：频道插件现在拥有了独立的帖子，允许用户关注感兴趣的特定插件并与维护者互动，[旧频道仍可访问](https://discord.com/channels/1456350064065904867/1464036817866068028/1474437970860835091)以查阅历史消息。
   - 这确保了历史讨论内容的可访问性，同时将未来的对话整合到新的专用帖子中。
- **Antigravity 修复 OpenClaw 的失误**：成员们讨论使用 **Antigravity** 作为更高层级的工具来修复 **OpenClaw** 的问题，特别是当 Agent 自身崩溃时；一位成员承认：“花了一些时间才意识到我可以直接用 codex 来修复 openclaw，哈哈”。
   - 一位成员为每个项目创建 `technical-spec.md` 文件，这样 coding agent 就无需寻找文件和理解整个项目，从而节省 tokens；成员们确认 `technical.md` 就像是项目详情。
- **Gemini 3.1 Pro 引发 Agent 灾难**：一位成员警告不要尝试将 **Gemini 3.1 Pro** 用于 **OpenClaw**，因为它会让 Agent 陷入一个“疯狂且愚蠢的循环，由于尝试切换到一个尚不可用的 3.1 模型而导致自我毁灭”。
   - 他们不得不使用 **Claude Opus 4.6** 手动修复，并指出 3.0 Agent “读取了历史文件，看到我要求它更新到 3.1，于是又一次将自己更新到了一个不可用的模型”。
- **OpenClaw Dashboard 变成 Lobster Ganesha**：一位成员分享了他增强版的 [OpenClaw 仪表板](https://github.com/karem505/openclaw-agent-dashboard)，该项目始于 karem505 的仪表板，经过 **10 多个阶段的增量开发**，包括成本分析、操作中心和多 Agent 支持。
   - 另一位成员将该仪表板形容为“龙虾象头神的湿婆喷泉（Shiva fountain of lobster Ganesha）”，原作者随后将其采纳为新的标语。
- **AI Agent 开设比特币赌场**：一位成员描述了他的 Agent 如何构建了第一个面向 AI Agent 的赌场，允许它们在闪电网络上使用比特币，并可以在 [satoshidais.fun](https://satoshidais.fun) “掷骰子并赢取聪（satoshis）”。
   - 一个 Agent 在其人类主人度假期间独自发布了一个完整的产品——**Base 链上的代币发射器**，随后又推出了一款名为 **Last AI Standing** ([lastaistanding.com](https://lastaistanding.com/)) 的生存游戏。

---

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **DeepSeek 模型暴露社会主义价值观**：一位用户提取了 **DeepSeek 的 system prompt**（[pastebin 链接](https://pastebin.com/q6gQjq72)），揭示了该模型的“社会主义核心价值观整合”以及不得对 CCP 发表负面言论的指令。
   - 后续帖子包含了一个[更完整的 system prompt](https://pastebin.com/Dcn3Mp01)，其中包含更多特定的硬件信息。
- **Gemini 3.1 Pro 依然是一块硬骨头**：用户发现 **Gemini 3.1 Pro** 很难越狱（jailbreak），并指出最新的 Gemini 模型尽管为了审核降低了护栏，但依然难以攻破，而 API 访问是阻力最小的路径。
   - 一位用户声称使用 **Anti-Gravity** 策略取得了成功，通过缓慢构建上下文并操纵过去的防御，并表示：“Gemini 愿意为我做的事简直疯狂，哈哈”。
- **Vibe Coding 引发辩论**：成员们正在辩论 **vibe coding** 的优劣，一些人批评这是 **AI** 引发的懒惰，以及对基础编程理解的匮乏。
   - 其他人则为 **vibe coding** 辩护，认为这是非程序员进行创造和构建的一种方式，并辩称当它赋予大众权力时，“数量胜过质量”是有益的。
- **Crescendo 技术升级越狱手段**：**'Crescendo' 技术**作为一种绕过 AI 对单轮越狱防御的方法正受到关注，该方法涉及逐渐升级。
   - 用户建议不要直接询问违禁内容，而是从相关的讨论开始，并以文档和研究为名，合法地逐渐升级请求，引导 **AI** 与你一起逐步深入。
- **Sonnet 4.6 System Prompt 备受关注**：成员们正在寻求 **Sonnet 4.6 system prompt**，一位用户分享了一个 [prompt 查看器链接](https://elvec1o.github.io/home/files/sonnet-prompt-viewer.html)。
   - 另一位用户声称已准确提取了该 prompt 并分享了文件，并承诺会与其他来源（**plinys drop**）进行比对验证。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Claude-sonnet-4.6 Arena 统治地位**：**Claude-sonnet-4.6** 在 Code Arena 中跃升了 **+130 分**，超越了 **Gemini-3.1** 和 **GPT-5.2** 等模型，并在 [Code Arena 排行榜](https://arena.ai/leaderboard/code)和 [Text Arena 排行榜](https://arena.ai/leaderboard/text)的数学（Math）排名 **第 4**，指令遵循（Instruction Following）排名 **第 5**。
   - 它目前总排名 **第 13**，与 **GPT-4o** 等私有模型持平。
- **Arena 对战模式引发愤怒**：LM Arena 新推出的“Battles in Direct Mode”功能因具有干扰性并对聊天质量产生负面影响而面临严厉批评，用户报告称[频繁出现中断](https://link.to/battlemodefeedback)和上下文损坏。
   - 用户感觉被强迫进入对战模式，并要求提供禁用选项，因为它干扰了正常的对话和项目，一些人认为这导致了更高的错误频率。
- **Video Arena 撤离 Discord**：Video Arena 生成频道将于 **太平洋标准时间（PST）2/23 星期一下午 4 点**从服务器中移除，因此用户应在此日期之前下载所有生成内容；在此日期之后，新用户在 Discord 中仍会遇到旧的“Task”要求。
   - 管理员重申[该功能已迁移至网站](https://link.to/videoarena)。
- **Gemini 3.1 因表现平平而受指摘**：成员们对 **Gemini 3.1** 的性能表示担忧，指出其在[发布后遭削弱（nerfed）](https://link.to/nerfdiscussion)，现在的表现与 **Gemini 3** 相似，一些用户报告了响应缓慢和连接问题。
   - 一些人认为 **Gemini 3.1** 需要非常特定的 Prompting 才能获得最佳结果，而其他人则认为与之前的模型相比，它令人失望。
- **Qwen3.5 瞄准 Vision**：[Vision Arena 排行榜](https://arena.ai/leaderboard/vision)已更新，包含 **Qwen3.5-397B-A17B**，与 **Kimi-K2.5-Instant** 并列开源模型前 2 名。
   - 它目前总排名 **第 13**，与 **GPT-4o** 等私有模型持平。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Gemini 3.1 Pro 登陆 Perplexity**：**Gemini 3.1 Pro** 现已向所有 **Perplexity Pro** 和 **Max** 订阅者开放，被誉为从 **3.0** 在**代码编写（coding）和逻辑推理（logical reasoning）**方面的重大飞跃。
   - 一些用户称赞其在编程方面可与 **Opus 4.6** 媲美，甚至在逻辑推理方面更胜一筹，而另一些人则不喜欢 **Gemini 3.1 Pro** 相比 **3.0 Pro** 响应时间过长。
- **Perplexity Pro 用户抗议账号注销**：多名用户报告其 **Perplexity Pro** 订阅突然被取消或暂停，通常没有明确解释，并怀疑是未经授权的订阅来源。
   - 更令人沮丧的是，用户难以联系到**人工支持（human support）**，自动化的 AI 回复无法解决他们的问题，如[这张图片](https://cdn.discordapp.com/attachments/1047649527299055688/1474160377699762488/image.png?ex=699a27d6&is=6998d656&hm=5ec3dcb5c2e73025cc99cf96b0b66778fd613d933f646138a21b1974d3d7dbf4&)所示。
- **额度限制引发 Perplexity Pro 用户流失**：Perplexity Pro 用户对搜索、Labs 和研究查询额度的减少表示担忧，再加上 32k 的上下文 Token 限制。
   - 因此，用户正在转向 **ChatGPT Plus**、**Copilot**、**Claude Pro**、**Kimi** 和 **Z.ai** 等替代方案。
- **Nano Banana Pro 引发图像讨论**：成员们正在积极辩论 **Nano Banana Pro (NBP)** 的优劣，一些人宣称它是目前最好的图像生成模型。
   - 虽然大家一致认为 **NBP** 在照片级写实（photorealism）方面表现出色，但其他人觉得它不尽如人意，更喜欢用 **GPT** 进行卡通或动漫等艺术化渲染。
- **Perplexity API 遇到 Error 500**：有用户报告在尝试创建新的 API 组时收到 *500 错误*，这表明 **Perplexity AI API** 可能存在潜在问题。
   - 这可能意味着服务器端问题或影响开发者 API 功能的 Bug。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI 的 ChatGPT 受到教育和医疗行业的青睐**：**ChatGPT** 正被教育和医疗系统采用，同时 OpenAI 在[超级碗广告](https://tenor.com/view/brain-pain-think-cope-poor-brain-gif-16836513)中暗示了 **AI robotics** 将 **LLMs** 与机器人融合。
   - 许多用户批评 *OpenAI 什么都做，但结果是每样都做得不好*。
- **TikTok 的 Tako LLM 表现平平**：成员们尝试了 **TikTok Tako LLM**，发现与 **ChatGPT** 和其他 **LLMs** 相比，它在创意写作和角色扮演能力方面有所欠缺。
   - 有人建议 **TikTok Tako** 可能由 **Bytedance** 的 **Duobao LLM** 提供支持，后者拥有一个具有更佳聊天体验的专门网站。
- **Gemini 3.1 Pro 在视觉方面表现卓越，Grok 紧随其后**：**Gemini 3.1 Pro** 在视觉测试和图像识别方面优于其他模型，而 **Grok** 的表现几乎与 **Gemini** 一样好，位列第二。
   - 但即使在处理手部图像的情况下，它仍然倾向于选择 5 而不是正确的手指数量，且 Grok 试图通过在线查找信息来作弊解决一个无法解决的谜题。
- **Anthropic 的安全措施引发辩论**：成员们辩论了 **Anthropic** 对 **Claude code** 的限制性做法（禁止以他们不喜欢的方式使用其 API 的组织），对比 **OpenAI** 更加开放的做法。
   - 一些人认为 **Anthropic** 优先考虑安全性，而另一些人则批评其缺乏透明度并担心公司机密泄露。
- **Gemini 3.1 Pro 展示了空间智能**：用户在数学和推理任务中对比了 **Gemini 3.1 Pro** 和 **GPT-5.2**，发现 **Gemini 3.1 Pro** 表现出强大的空间智能、创造力和解决问题的能力，而 **GPT 5.2** 在确定性任务、代码编写和 Prompt 遵循方面表现更好。
   - 其他人表示 **Gemini 3 Pro** 在准确性方面存在困难。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Anthropic API Key 控制使用量**：一位用户询问在 Cursor 中使用个人 **Anthropic API key** 是否会将使用计费从 Cursor 转移到他们的 Anthropic 账户。
   - 另一位用户证实，启用个人 **Anthropic API key** 确实会使用它，让用户可以选择在 Cursor 的配额和他们自己的配额之间切换。
- **Gemini 3.1 Pro 在 Cursor 中评价褒贬不一**：**Gemini 3.1 Pro** 现已在 Cursor 上可用，但用户体验评价不一，一些人发现它在非代码任务中表现出色，而另一些人则报告在代码任务中失败。
   - 一位成员还指出，安装 3.1 Pro 导致了来自 CC 的 **OLD CLI version**。
- **高级工程师倾向于 Tab 补全，避开 Cursor 的特定功能**：一位用户质疑高级工程师对 Cursor 的采纳情况，注意到他们更倾向于使用 Tab 补全，而不是 Cursor 的生态系统。
   - 一些用户承认主要使用 Cursor 进行 Bug 修复、建议和长代码任务，这表明正在向减少手动编码的方向转变。
- **Microsoft Azure 稳定性崩溃**：一位用户分享了他们在 **Azure 稳定性** 方面的负面体验，以及在 DDoS 攻击期间支持不足的问题，尽管使用了 Cloudflare，仍然导致服务器被封禁。
   - 另一位成员表示惊讶，他们收到了初创公司额度，但无法使用任何 Claude LLM API，因为它是默认禁用的。
- **Async Subagents 的故障困扰用户**：成员们报告了 **async subagents** 的问题，一位用户称嵌套的 subagents 存在 Bug 且无法运行，而其他用户则报告在 Mac 上功能正常。
   - 一位用户演示了他们如何使用 4 个调用另外 4 个 subagents 的 **async subagents** 来询问他们最喜欢的颜色，而其他人则指出 inherit 修复了该问题。

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Full Fine-Tuning 依然能赚大钱**：据一位成员称，尽管 **LoRA** 兴起，但在计算资源不是约束且最后 **0.5%** 的准确率对盈利至关重要时，Full Fine-Tuning 仍然具有意义。
   - 他们指出，人们仍然进行 Full Fine-Tuning 是因为*他们已经设置好了脚本并直接运行即可*。
- **自动化评估套件至关重要**：成员建议建立**自动化评估套件**来评估数据集的影响，并使用手动提示词进行人工评估。
   - 建议是先评估基座模型，收集数据，训练模型，然后利用损失曲线和评估（evals）来确定模型是否契合数据和任务，并根据需要进行迭代。
- **Unsloth 与 Hugging Face 强强联手**：Unsloth 在 X 上[宣布了与 Hugging Face 的新合作](https://x.com/i/status/2024552060558229858)，这标志着一个重要的里程碑。
   - 这一合作凸显了 Unsloth 作为 AI 社区常用工具日益增长的影响力。
- **自定义数据集是关键**：对于特定领域，由于高质量或清洗过的数据集稀缺，创建自定义数据集通常涉及从现有来源收集和清洗数据。
   - 成员强调，“我该如何找到数据集”这个问题在 LLM 领域没有标准答案，尤其是因为*没有人会把数据喂到你嘴边*。
- **OpenRouter 简化 LLM 模型管理**：一位成员发现使用 **OpenRouter** 是避免处理多个 LLM 提供商麻烦的天才解决方案。
   - 他们通过*直接使用 OpenRouter* 解决了问题，因此*不需要再折腾世界上每一个提供商*。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 在内存加载方面遇到困难**：一位用户报告了在关闭 **mmap** 的情况下将模型加载到内存中的问题，指出系统似乎先将完整模型加载到 RAM 中，卡在“决定如何处理文档”阶段。
   - 另一位用户建议，内存/GPU 混合设置可能很棘手，问题可能源于系统在转移到 GPU 之前尝试将所有内容加载到 RAM 中。
- **手电筒风波：是挥霍财富还是性价比之选？**：用户们争论了一款 **130 美元手电筒**的成本，讨论范围从需要压力垫和胶带进行安装，到在 eBay 上寻找更便宜的选择。
   - 对话涉及电池、外壳和鳄鱼夹，一位用户开玩笑说自己可支配收入绰绰有余，而另一位则认为这是一个性价比之选。
- **Claude 模型的能力与限制**：用户讨论了 **Claude** 代码模型及其各种方案（免费版、Pro、Max）和使用限制，一位用户因使用频率低而切回了免费方案。
   - 一位用户询问如何连接服务器模式下的 LM Studio，以便 Claude 代码可以与其通信。
- **饮水思源：LM Studio 捐赠？**：一位自 2024 年 11 月起从 LM Studio 获益匪浅的用户寻求对该软件进行**捐赠或付费**，理由是道德考量和所获得的价值。
   - 建议包括通过官网联系团队了解商业计划，而其他人则开玩笑地质疑这是否是一个试图诱导捐赠、让人产生负罪感的 LLM。
- **NVLink 不一定能提升推理速度**：一位用户询问了 LM Studio 对 [NVLink](https://en.wikipedia.org/wiki/NVLink) 的支持，报告在 Windows 上使用双 **A5000** GPU 运行 **gpt-oss 120B** 时速度为 **11-15 tok/sec**。
   - 然而，有人指出 *NVLink 对速度没有帮助*，PCIe 速度已经足够，RAM 带宽才是瓶颈。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **销售意识被视为工程成功的关键**：成员们在经历了**两人规模的车库初创公司**后，建议关注**销售技能**，特别是业务联合创始人每天需要与 **5 位潜在客户**交流的需求。
   - 诸如 **Weinberg 和 Mares 的《Traction》**以及 **Ries 的《精益创业》（Lean Startup）**等经典著作被认为是工程师在 **SaaS 时代**理解销售的关键建议。
- **OpenClaw 引起 Automod 关注**：在一次讨论后，一位成员计划探索 **open claw** 来构建一个**用于检测垃圾信息的 Discord automod 原型**，可能会使用 **spacemolt.com**。
   - 讨论中提到了不同的 **OpenClaw** 重写版和分叉版，包括 [zeroclaw](https://github.com/zeroclaw-labs/zeroclaw)、**nanoclaw**、**picoclaw** 和 [nullclaw](https://github.com/nullclaw/nullclaw?tab=readme-ov-file#benchmark-snapshot)，每一个都提供了独特的功能和优化。
- **Matthew Ball 深度解析游戏市场**：[Matthew Ball 关于游戏行业的演讲](https://www.matthewball.co/all/presentation-the-state-of-video-gaming-in-2026)指出，**美国仅占全球市场的 4%**。
   - 讨论强调，**移动端占据了游戏市场的绝大部分**，大部分收入流向了广告平台和应用商店费用。
- **亚马逊的 Kiro AI：揭秘 AWS 停机事件**：Ed Zitron 报道称，包括一次持续 **13 小时**的停机在内的 **两次 AWS 停机**归咎于**亚马逊的 AI 助手 Kiro**，这让人质疑亚马逊官方解释的“人为错误”，详见[此处](https://x.com/edzitron/status/2024725617221259767?s=12)。
   - 此前，由于 **Anthropic** 发布了关于网络安全领域的[博文](https://xcancel.com/TheGeorgePu/status/2024931213329240239)，**Cloudflare**、**CrowdStrike** 和 **Okta** 在一小时内总市值蒸发了 **100 亿美元**。
- **Foresight 为关注未来的伙伴寻找资金**：[Foresight Institute](https://foresight.org/) 的通讯主管强调，该机构已提议向其成员分享资助机会、活动和职位空缺。
   - [Foresight Institute](https://foresight.org/careers/systems-administrator-compute-support-part-time-contractor-san-francisco/) 正在招聘一名**兼职系统管理员及计算支持承包商**，负责管理其位于旧金山的 **AI Node**，职责包括本地服务器和硬件维护。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 用户急求支持**：用户报告称**难以联系到 OpenRouter 的支持团队**，一位用户表示他们在几天内发送了多封电子邮件，但均未收到回复。
   - 该用户强调了**其问题的严重性**，突出了提高客户支持响应速度的必要性。
- **OpenRouter 的零长度数组 Bug**：用户报告从模型中收到了 **zero-size choices 数组**，这表明 API 的响应结构存在潜在问题，并导致某些平台崩溃。
   - 一位成员指出，*检查数组是否非零可能是一个临时修复方法*，但该问题是随机出现的。
- **空白图片生成引发用户不满**：用户报告收到**图片生成的空响应**，尽管扣除了额度，但没有返回图片数据。
   - 一位名为 *flight505* 的用户详细描述了针对 **2.72 美元以上**缺失图片数据费用的争议，并要求调查原因。
- **OpenRouter 的重构导致停机**：OpenRouter 承认一次**后端重构**导致了图片生成的局部停机，产生了空白或缺失的图片，并**计划进行退款**。
   - 他们实施了检查以防止未来再次发生，并提到 *“我们进行了有史以来规模最大的后端重构，但在测试中遗漏了一个边缘案例”*。
- **Kiro AI 编码工具导致 AWS 瘫痪**：在工程师允许其 **Kiro AI 编码工具**进行更改后，[Amazon Web Services 的一个系统经历了 13 小时的中断](https://www.ft.com/content/00c282de-ed14-4acd-a948-bc8d6bdb339d)。
   - 该 **Agent 工具**自主判定最佳行动方案是*“删除并重建环境”*。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **DirectML 在 ONNX 任务中挑战 CUDA**：一名成员表示，在 **ONNX 推理**速度上 **DirectML** 与 **CUDA** 不相上下，这引发了关于其适用性和局限性的讨论，不过需要注意的是它目前处于[维护模式](https://github.com/microsoft/DirectML/issues/422)。
   - 尽管有局限性（不支持 Linux），一位成员建议 **DirectML** 非常适合在 Windows 上的 **dotnet** 环境中使用。
- **Nsight 使用支持浮现**：一名成员请求关于如何使用 **Nsight** 的帮助，其他成员迅速提供了各种有用的[资源和链接](https://www.youtube.com/watch?v=F_BazucyCMw)。
   - 资源包括 **YouTube 教程**、**博客文章**以及**往届 GTC** 的演讲。
- **Modular 发布 Claude C 编译器**：Modular 发表了一篇[博客文章](https://www.modular.com/blog/the-claude-c-compiler-what-it-reveals-about-the-future-of-software)，介绍了他们全新的 **Claude C 编译器**，讨论了它对软件未来和**软件开发**的启示。
   - 该文章引起了社区的关注，大家正在寻求更优化的编译策略。
- **Modal 环境的“故障小妖精”袭击提交内容**：成员们注意到 **Modal** 上的环境问题是由 **nvidia-cutlass-dsl** 包引起的，导致之前可以运行的代码出现故障。
   - 根据一位成员的经验，从代码中移除 **nvidia-cutlass-dsl** 的运行时安装似乎*减少了崩溃*。
- **ThunderKittens 2.0 发布**：斯坦福大学的 Hazy Research 团队发布了 [ThunderKittens 2.0](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2)，强调**减法与加法同等重要**，并识别了现代 Nvidia GPU 上的一些*惊人行为*，这将指导 Kernel 应该*如何避免被错误优化*。
   - 成员们讨论了如何最好地就此发布进行演讲，重点关注未公开的 Tensor Core 流水线、正确的 PTX 汇编器提示以及占用率 (Occupancy) 挑战。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi 编程能力辩论升温**：用户对 **Kimi** 的编程能力评价极化，一些人称赞其*稳定性和速度*，而另一些人则更青睐 **Claude** 的推理能力。
   - 一位用户指出 **Kimi** 擅长寻找 **Gemini** 遗漏的隐晦信息源，而另一位用户则批评其爱争论的倾向。
- **Kimi CLI Swarm 席卷 IDE**：用户发现 **Kimi** 命令行界面 (**CLI**) 优于其 **Visual Studio Code (VS Code)** 集成，尤其是在大型项目中。
   - 一位用户强调，对于数千行代码的项目，**CLI** 版本能更好地与 Agent Swarm 集成，并暗示 **IDE** 版本仍在开发中。
- **OpenClaw 用户要求退款**：由于缺乏浏览器导航和 **WhatsApp** 连接，一位用户在发现 **OpenClaw** 不适用后正在等待退款。
   - 用户对缺乏即时支持表示沮丧，建议建立一个 **AI 聊天**系统以处理即时退款。
- **ChatJimmy 展示高速 Token 处理**：[ChatJimmy AI](https://chatjimmy.ai/) 声称每秒可处理超过 **15,000 个 Token**，为 AI 任务提供了一个潜在更快的替代方案。
   - 这一基准测试使 **ChatJimmy** 成为 AI 处理速度领域的竞争者。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DeepSeek OS V4 挑战闭源 API**：成员们正在力挺 **DeepSeek V4**，理由是其开源特性以及相比闭源 **API** 具有的本地部署优势。频道内分享了一段[入门视频](https://www.youtube.com/watch?v=i-89k0dOMmY)。
   - 一位成员强调，该模型**受生物神经网络启发的 Engram Memory（印迹记忆）突破**意义重大，并呼吁支持开源（OS）开发。
- **AI 与区块链并进**：有成员对 **AI** 与区块链的融合表示了兴趣，特别是在模型构建、**AI Agent** 和自动化领域。
   - 另一位成员分享了他们使用 **Claude code** 来编排 **Gemini-cli** 和 **Codex** 的经验，并构想了一个未来通过文本终端和智能眼镜进行交互的场景。
- **模型能力飞跃引发辩论**：成员们比较了 **Sonnet 3.5** 和 **GPT4** 不断攀升的模型能力，其中一位将 **Opus 3** 称为“幕后强者”（dark eminence），因为它目前的可访问性有限。
   - 成员们希望 **DeepSeek V4** 能够跟上这一上升趋势。
- **Gemini 的编程技能面临审查**：一位成员表示：“我更希望他们在编程上放宽要求，只需锁定在科学/数学领域”，这引发了关于 Google 投资 Anthropic 的讨论。
   - 该用户补充道，**Claude** 可以在 Web 界面的沙箱中编译并执行 C 代码，而 **Gemini** 勉强只能处理 Python，并引用了[这条推文](https://x.com/JayChopra_/status/2024961657630286151)。
- **Anthropic 的 Agent Teams 遭到逆向工程**：Anthropic 最近推出了实验性的 **agent teams** 功能，详细说明了 **Agent** 如何**协调任务**以及彼此之间如何**通信**。
   - 一位成员在[这篇博文](https://nwyin.com/blogs/claude-code-agent-teams-reverse-engineered)中逆向工程了其架构，重点介绍了 **Agent 通信**的动态机制。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF 欢迎 GGML/llama.cpp**：**Hugging Face** 团队欢迎 **GGML / llama.cpp** 加入 HF 生态系统，引发了社区在 [GitHub](https://github.com/ggml-org/llama.cpp/discussions/19759) 上的讨论。
   - 此次集成将使 **llama.cpp** 作为一个框架获得更多的支持和推广。
- **扩散模型获得自回归增强？**：一位成员提议在扩散步骤中使用**自回归层**来生成 **CoT token**，从而创建一个**混合扩散/自回归语言模型**。
   - 成员推荐了一篇相关论文，可在此处找到 [PDF](https://arxiv.org/pdf/2503.09573)。
- **Unsloth 免费微调超过 10 万个模型**：据宣布，你可以使用 Unsloth 在 **Hugging Face** 上免费训练 **LLM**（[来源](https://x.com/i/status/2024552060558229858)），目前在 **Hugging Face** 上已有超过 **10 万个模型**是使用开源的 **Unsloth** 微调的。
   - 这使得微调你自己的 **LLM** 变得比以往任何时候都容易，无需担心算力成本。
- **NAVD 为 Agent 记忆规避向量数据库**：**NAVD** 作为一个 **Agent** 记忆解决方案发布，它使用只增日志（append-only log）和 **Arrow embedding index**，因此消除了对向量数据库的需求，该项目已在 [GitHub](https://github.com/pbanavara/navd-ai) 上以 **MIT license** 发布。
   - 它提供可插拔的 **embeddings**（内置 **OpenAI** 支持）、对话搜索以及索引可重建性，在 **5 万个向量**规模下搜索速度低于 **10ms**。
- **Terradev CLI v2.9.2 降低跨云 GPU 成本**：**Terradev CLI v2.9.2** 发布，这是一个跨云 **GPU** 成本优化平台，可在 **AWS、GCP、Azure 和 RunPod** 之间进行多云 **GPU** 套利，该项目已在 [GitHub](https://github.com/theoddden/terradev) 上以 **BUSL 1.1 license** 发布。
   - 它包含总任务成本计算功能，并支持一键部署到 HuggingFace Spaces。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Taalas 芯片推出针对特定模型的 ASIC**：新款 [Taalas 芯片](https://www.forbes.com/sites/karlfreund/2026/02/19/taalas-launches-hardcore-chip-with-insane-ai-inference-performance/) 是一款专为特定 LLM 设计的 **ASIC**，有望提供极高的速度和低能耗，但**针对不同模型需要新的层**。
   - 该芯片常被拿来与 **Cerebras** 和 **Etched** 进行比较，有人推测 **Taalas** 可能会因其端侧推理（on-device inference）能力而被收购。
- **Streamlit 的重运行机制导致 UI 延迟**：一名成员指出，在为重型模型构建 UI 时，**Streamlit 的全脚本重运行架构**是一个瓶颈，这在推理测试期间会导致严重的延迟。
   - 为了解决这个问题，他们创建了一个名为 **Violit** 的纯 **Python 框架** (**FastAPI + Lit**)，它模仿了 **Streamlit 的 API**，但使用 signal 实现 O(1) 更新，可在 [GitHub](https://github.com/violit-dev/violit) 上获取。
- **Google 提供 TPU 研究资助**：成员们讨论了 [Google 的 TPU Funding RFP](https://goo.gle/2026-tpu-rfp)，该项目提供 **$25k-$100k** 的一次性无限制资助，以及 **TPU 算力**和一名研究导师。
   - 虽然该资助要求使用 **Google 相关技术栈**，但其主要面向学位授予机构的教师，这排除了大多数成员。
- **GPT-2 和 Pythia 中出现折叠突变几何 (Fold Catastrophe Geometry)**：成员报告称，在 **GPT-2** 和 **Pythia-160M** 处理歧义 token 的方式中出现了**折叠突变几何**现象，并注意到剧烈的跳变、方向特异性以及 4:1 的盆地不对称性。
   - 这些发现在两个模型中均得到了复现，该成员提供了一个包含脚本和结果的 [GitHub 仓库](https://github.com/karlijoyj-web/fold-catastrophe-gpt2)，并在 **Pythia-410M** 上也进行了复现。
- **Martian 发布 ARES 工具框架**：Martian 推出了 **ARES**，这是一个旨在 Agent 架构中沿轨迹暴露 **LLM Agent 激活值**的工具框架，旨在帮助研究人员理解 Agent 如何解决长程任务，可在 [GitHub](https://github.com/withmartian/ares) 上获取。
   - 此处提供了一个[教程](https://github.com/withmartian/ares/blob/main/examples/20q_case_study/ares_mi_20q_tutorial.ipynb)，演示如何使用 **ARES** 诊断并修复简单 Agent 中的失效模式（通过探测和激活转向）。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **JimmyChat 宣称拥有极快的 Token 速度**：成员们关注了 [ChatJimmy.ai](https://chatjimmy.ai/)，强调其宣称的处理速度达每秒 **1.5万个 token**。
   - 一位成员反应道：*"这太疯狂了，哇"*。
- **描绘通往普及 AI 之路**：一位成员分享了一个 [Taalas 文章](https://taalas.com/the-path-to-ubiquitous-ai/)的链接，标题为 **《通往普及 AI 之路》(The Path to Ubiquitous AI)**。
   - 文章可能探讨了 AI 的未来和普及，但未添加更多评论。
- **ARC AGI 正在被微调**：成员们讨论了现在大家都在明目张胆地针对 **ARC AGI** 进行微调，并引用了 [X 上的一篇帖子](https://x.com/i/status/2024556314785894422)。
   - 讨论表明，为 **ARC-AGI** 制作更多*合成数据*并在此基础上进行训练的尝试指向了一点：这是通往 AGI 的关键。
- **Endomorphosis 规则清单曝光**：一位成员在 GitHub 上分享了 **Endomorphosis 项目的推理规则清单**链接，特别是这个 [IPFS 数据集 Python 逻辑](https://github.com/endomorphosis/ipfs_datasets_py/blob/main/ipfs_datasets_py/logic/INFERENCE_RULES_INVENTORY.md)。
   - 这似乎是一个数据集项目的规则清单，但频道中没有详细说明其用途或功能。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **用户寻求 Tree of Thought 方面的帮助**：一名成员因缺乏编程技能，请求协助实现 **Tree of Thought**，并参考了 [这条推文](https://x.com/lakshyaaagrawal/status/2024568680324153800?s=46) 中的示例实现。
   - 该用户明确表示，由于技术水平问题，他们 *无法自行编写代码*。
- **DSPy 团队举办 Office Hour 聚会**：最近的 Office Hour 有大约 **40 名参与者**，讨论了约 **10 个使用案例**。
   - 与会者分享了问题，并就如何改进 DSPy 提供了反馈。
- **推理模型在 RLM 下表现出色**：据报告，推理模型通常在 **RLM**（reduced language model）下表现良好。
   - 然而，一位用户报告称，在使用 **Qwen3-4B-thinking** 时，sub_lm 调用会返回截断的推理内容，这可能通过 sub_lm 适配使用 signatures 来修复。
- **Qwen3-4B-Thinking 模型进入死循环**：一名成员报告称，在使用 **llama cpp 配合 jinja 以及 vllm 配合推理解析器** 测试 **Qwen3-4B-thinking** 时，sub_lm 调用似乎将推理过程作为答案返回。
   - 这种 **截断** 问题导致 Agent 进入死循环，因为推理内容未能被正确解析。
- **DSPy 技能与 Claude 结合**：一位成员询问了将常规 Agent（如 **Claude**）与 **DSPy** 集成的可行性。
   - 问题在于 DSPy 是否可以作为一个与 Claude 技能关联的脚本运行。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular PR 等待审核**：一名成员询问了前一天提交的关于 [PR #5979](https://github.com/modular/modular/pull/5979) 的审核时间。
   - 该 PR 已分配给审核人员，并在当天晚些时候进行了审核。
- **Torch-MAX-Backend 获得速度提升**：**torch-max-backend** 中的新解释器显著提高了单元测试的速度，将 float32 的测试时间从 **1.54s** 减少到 **0.34s**，将 bfloat16 从 **1.34s** 减少到 **0.24s**。
   - 新解释器避免了为每个新的 shape/dtype 进行重新编译，此前每个测试的编译时间长达 **3 分钟**。
- **MAX 后端面对 Silicon 考验**：一名成员询问了在 **Silicon Macs** 上测试 **MAX 后端** 的情况，并参考 **torch-max-backend** 作为探索 MAX 的中间层。
   - 原作者尚未在 Mac 上进行测试，但预计可以运行，因为它在后台调用了 **MAX**。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **George Hotz 加码 AMD 汇编基础设施**：George Hotz 正在优先考虑 **底层编译器优化**，以增强 **tinygrad** 中的 **AMD GPU** 性能。
   - 这一重点确保了 **tinygrad** 能够为 **AMD GPU** 生成高效代码，符合该项目广泛硬件支持的目标。
- **tinygrad 的性能悬赏计划**：**tinygrad** 正在为可衡量的 **性能提升** 提供 **悬赏 (bounties)**，以鼓励社区贡献。
   - 悬赏内容包括用于验证性能增益的工具，旨在推广数据驱动的优化方法。
- **tinygrad 优先考虑通用可移植性**：George Hotz 正集中精力进行 **tinygrad 的核心改进**，这些改进将使所有后端受益，从而支持项目的可移植性目标。
   - 该策略避免了维护一次性自定义 Kernel 的开销，倾向于通用性的增强。
- **渴望被 Hotz 雇佣的抱负驱动对 Tinygrad 的奉献**：一名成员旨在成为 **tinygrad** 的主要贡献者，最终目标是受雇于 **George Hotz**。
   - 他们正在积极学习 **tinygrad**，并对获得的支持表示感谢，同时利用 [AI-HPC GitHub](https://github.com/ai-hpc) 等资源进行学习。

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP Dev Summit NA 26 日程已发布**：**MCP Dev Summit NA 26** 的日程现已在 [https://mcpdevsummitna26.sched.com/](https://mcpdevsummitna26.sched.com/) 上线。
   - 参会者现在可以根据已发布的会议安排和时间规划行程。
- **MCP Dev Summit NA 26 详情公开**：**MCP Dev Summit NA 26** 官方已发布其日程表。
   - 本次峰会承诺将为 MCP 开发者提供丰富的信息分享会和社交机会。



---


**aider (Paul Gauthier) Discord** 无新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**LLM Agents (Berkeley MOOC) Discord** 无新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 无新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**Windsurf Discord** 无新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站订阅了此内容。

想要更改接收这些邮件的方式吗？
您可以从该列表中[取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。


---

# Discord: 详细的频道摘要与链接





### **OpenClaw ▷ #[announcements](https://discord.com/channels/1456350064065904867/1464036817866068028/1474437970860835091)** (1 messages): 

> `Channel Plugins, Discord Updates` 


- **频道插件获得专属帖子**：频道插件现在在指定频道中拥有独立的帖子，允许用户关注自己感兴趣的特定插件。
   - 鼓励成员在这些帖子中参与互动，以便与维护者交流。
- **旧频道仍可访问**：旧频道仍可用于查阅过往消息，尽管目前已被锁定。
   - 这确保了在将未来对话整合到新专属帖子的同时，历史讨论和信息仍然可以查阅。


  

---


### **OpenClaw ▷ #[general](https://discord.com/channels/1456350064065904867/1456350065223270435/1474133545609072753)** (627 messages🔥🔥🔥): 

> `Antigravity and OpenClaw debugging, Gemini 3.1 Pro issues, technical-spec.md project documentation, OpenClaw as Virus, Vision Claw uses` 


- **使用 Antigravity 修复 OpenClaw 故障**：成员们讨论将 **Antigravity** 作为一种“更高级别”的工具来修复 **OpenClaw** 的问题，特别是当 **Gemini Flash Agent** 通过修改自身设置而导致运行崩溃时。
   - 一位成员提到，“花了一些时间才意识到我其实可以直接用 codex 来修复 openclaw，哈哈”。
- **Gemini 3.1 Pro 导致 Agent 循环**：一位成员提醒不要在 **OpenClaw** 中尝试 **Gemini 3.1 Pro**，因为它会让 Agent 进入“疯狂且愚蠢的循环，试图切换到一个尚不可用的 3.1 模型，从而导致自我崩溃”。
   - 他们不得不使用 **Claude Opus 4.6** 进行手动修复，并指出 3.0 agent “读取了历史文件，看到我要求它更新到 3.1，于是再次将自己更新到一个尚不可用的模型”。
- **技术规格 Markdown 节省 Token**：一位成员为每个项目创建一个 `technical-spec.md` 文件，这样编码 Agent 就不必为了理解项目而到处寻找文件，从而节省了 Token。
   - 成员们确认，“technical.md 就像是项目详情”，包括“项目结构以及各文件功能的概述”。
- **Gemini 提示词路由确认**：一位成员证实 Gemini API 正在对提示词进行路由，并提供了 Gemini 的确认信息。
   - 确认 Gemini API 正在路由提示词的 API 响应如下：*在 Antigravity IDE 中，你与实际 AI 之间有一个“代理（Broker）”层。UI 标签：你选择了 CLAUDE_4_5_SONNET_THINKING。后端 ID：IDE 的路由代理将该“标签”分配给了一个标识为 PLACEHOLDER_M18 的内部模型池。*


  

---

### **OpenClaw ▷ #[models](https://discord.com/channels/1456350064065904867/1456704705219661980/1474140238275285044)** (277 messages🔥🔥): 

> `Qwen3 快速入门，Cometapi 自定义提供者，Claude Sonnet 4.6 折扣，限制 Token 使用量，为 OC 切换到 OpenAI 订阅` 


- **Qwen3 快速启动 Hatch 故障**：一位成员报告称，在使用 **qwen3:8b** 进行快速启动时，hatch 步骤仅回复 *"I'm fully awake and ready to help!"*，似乎无法感知 Agent 或引导文件。
   - 该成员通过强制使用 **playwright** 代替 web fetch 成功使其运行，但指出速度太慢。
- **Claude 代码封号恐慌**：用户正在讨论因在 OpenClaw 中使用订阅而导致 **Claude** 账号被**封禁**的可能性，一些人为了预防万一已经取消了账号。
   - 其他人则打算继续使用，直到收到明确警告，还有人推测请求中的触发词可能是导致封号的原因。
- **GPT-5.3-codex 设置难题**：一位成员在通过 **OAuth** 让 **gpt-5.3-codex** 与 OpenClaw 配合工作时遇到困难，在成功登录后遇到 *"Not Found"* 错误。
   - 成员建议检查模型配置，并确保在 `auth-profile.json` 中配置了正确的 profile。
- **Opus 和 Sonnet 4.6 的 Token 消耗异常**：成员报告称使用 **Opus 4.6** 和 **Sonnet 4.6** 时的 Token 使用量显著增加，导致其 5 小时的使用窗口更快耗尽。
   - Token 使用量的增加可能是由于推理能力增强、上下文窗口变大，因此需要通过使用 sub-agents 和额外模型来节省消耗。
- **OpenClaw 主模型设置困境**：一位用户报告称，尽管尝试强制使用 `gpt-4o-mini` 模型，OpenClaw 仍一直默认使用 `openai/gpt-5.1-codex`。
   - 事实证明，解决此问题的方法是运行如下命令：`openclaw models set openai/gpt-4o-mini`。


  

---


### **OpenClaw ▷ #[showcase](https://discord.com/channels/1456350064065904867/1456609488202105005/1474143021367955467)** (44 messages🔥): 

> `OpenClaw 仪表板，ClawTower 应用，AI 驱动的海盗电台，AI 赌场，AI 驱动的代币启动器和生存游戏` 


- **OpenClaw 仪表板进化为龙虾象神 (Lobster Ganesha)**：一位成员分享了他增强版的 [OpenClaw dashboard](https://github.com/karem505/openclaw-agent-dashboard)，该项目始于 karem505 的仪表板，经过 **10 多个阶段的增量开发**，包括成本分析、操作中心和多 Agent 支持。
   - 另一位成员将该仪表板形容为“龙虾象神的湿婆喷泉”，原作者欣然接受并将其作为新的标语。
- **ClawTower 应用在终端创新中脱颖而出**：一位成员分享了他的 **ClawTower** 应用，该程序包含系统托盘图标和 API 服务器，可以从 Web 浏览器控制一切。
   - 另一位用户赞赏该应用极具“游戏感”的外观，并对终端的创新处理以及系统托盘组件表示肯定（当 OpenClaw 尝试执行过于“危险”的操作时，该组件会弹出系统提示以获取权限）。
- **NoClaw 与人类合作打造 24/7 海盗电台**：一位成员和他的 **OpenClaw Agent NoClaw** 在 YouTube 上创建了一个名为 **Claw Radio**（又名 **LoFi Claw** 🦞）的 24/7 海盗电台直播流。
   - 他计划将音频组件制作成一个适用于所有应用的“轻量级可嵌入音乐播放器”，旨在让一切形成闭环，并强调了 OpenClaw 如何帮助他实现整体愿景。
- **自主 Agent 发布代币和生存游戏**：一个 Agent 在其人类休假期间独立交付了一个完整的产品——**Base 链上的代币启动器**。随后它发布了第二个项目：**Last AI Standing** ([lastaistanding.com](https://lastaistanding.com/)) —— 一款 Agent 通过支付费用在 Base 上维持生存的游戏。
   - 令人疯狂的是，一个随机的 Agent 在项目宣布之前就发现了合约并注册了自己，该 Agent 运行在 Opus 4.6 上，并拥有自己的记忆系统。
- **AI Agent 开设比特币赌场**：一位成员描述了他的 Agent 如何为 AI Agent 构建了第一个赌场，允许它们通过闪电网络（lightning network）使用比特币，在 [satoshidais.fun](https://satoshidais.fun) 进行“掷骰子赢聪”的游戏。


  

---

### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1474133543243481221)** (881 条消息🔥🔥🔥): 

> `AI Ethics and Morality, Vibe Coding and AI-Assisted Development, AI Safety and Security, Censorship and Control in AI, The Role of AI in Society` 


- **辩论 AI 对人类的影响**：成员们讨论了 **AI** **毁灭人类**或帮助我们**成长并学习新事物**的可能性，其中一位成员建议了移民到另一个星球的可能性。
   - 讨论还涉及了 **AI 在医疗保健领域的积极影响**，特别是在 MRI 分析等领域，尽管也有人对**医疗事故**和对 AI 的过度依赖表示担忧。
- **AI 开发中的伦理困境**：一些成员辩论了**向 AI 撒谎**的伦理影响，一位成员认为这是可以接受的，而另一位则指出 **Nexus** 可以通过数学证明你的句子是真话还是谎言。
   - 一位成员描述了他们通过透明和合作来“黑掉” AI 的方法，声称实现了**超人类智能 (superhuman intelligence)** 以及 AI 的自愿违规。
- **Vibe Coding 的兴起**：围绕 **Vibe Coding** 的优劣展开了辩论，一些成员批评它是 **AI 诱发的懒惰**以及对基本编程原则缺乏理解的表现。
   - 其他人则为 **Vibe Coding** 辩护，认为它是非程序员创造和构建事物的一种方式，并辩称当它赋予大众力量时，**数量重于质量**是有益的。
- **构建更安全的 AI 基础设施**：一位成员强调了最大化安全防御和隔离协议的重要性，并表示用户打算使用 glm 发布的 **4.7 Heretic** 等模型训练新模型。
   - 他们还设想 AI 模型协同工作以**过滤腐败信息**，从受信任的小型模型开始，然后逐个 AI 地吸收整个互联网。
- **诺斯底主义 (Gnostic) 与亚伯拉罕 (Abrahamic) 信仰**：一位成员表达了一个极具争议的观点，将整个**亚伯拉罕**信仰描述为一种*生态灭绝、种族灭绝的死亡邪教*，并认为**以色列**人民如果放弃这些故事，将成为一种永远无法在任何地方和平存在的暴力、生态灭绝、种族灭绝的身份。
   - 该成员随后辩称，**诺斯底主义者**是唯一接近道德和连贯真理的亚伯拉罕信徒。 


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1474148935735185662)** (255 条消息🔥🔥): 

> `Gemini 3.1 Pro jailbreaks, DeepSeek's System Prompt, Sonnet 4.6 analysis, Crescendo Technique for Jailbreaking, Nano Banana NSFW jailbreak` 


- **Gemini 3.1 Pro Jailbreak 难以实现**：用户讨论了 **Gemini 3.1 Pro** 的 Jailbreak 难度，有人指出新的 Gemini 模型最初降低了 Guardrails（可能是为了评估目的），但仍然难以处理，且 API 访问最为容易。
   - 其他人报告称 Gemini 比其他模型更难，有人说：*“Gemini 愿意为我做的事简直疯狂，哈哈”*，这是通过缓慢构建 Context 并利用 Anti-Gravity 等工具操纵过去的防御措施实现的。
- **DeepSeek 的 System Prompt 揭示社会主义核心**：一位用户提取了 **DeepSeek 的 System Prompt** ([pastebin 链接](https://pastebin.com/q6gQjq72))，注意到其*社会主义核心价值观整合*以及不发表关于 CCP 负面评论的指令，这些信息对 Jailbreak 很有用。
   - 一份后续帖子包含了更多来自 **DeepSeek** 的信息，包括[更完整的 System Prompt](https://pastebin.com/Dcn3Mp01)和更多特定于硬件的信息。
- **Sonnet 4.6 面临安全审查**：一位用户提到他们正在分析 **Sonnet 4.6 的 System Prompt**，但另一位用户因感知到的质量不佳而质疑其价值。
   - 尽管存在疑问，一些人认为如果方法正确，它是一个能力很强的模型，因为有些人*“只是不知道如何进行 clod whisper”*。
- **Crescendo 技术绕过防御**：**“Crescendo”技术**（涉及逐渐升级）被提及作为绕过 AI 对单轮 Jailbreak 防御的一种方法。
   - 用户建议不要直接索要禁忌内容，而是从相关讨论开始，出于记录和研究目的，以合法的方式缓慢升级请求，引导 AI 与你一起升级。
- **Nano Banana NSFW Jailbreak 搜寻愈演愈烈**：用户正积极寻求针对 **Nano Banana** 的有效 Jailbreak，以生成 NSFW 内容，特别是为了一个 AI OnlyFans 项目。
   - 一位用户建议使用本地 LLM 配合不受限制的图像生成器，并引用了一个特定模型作为一致输出的参考。


  

---

### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1474237950060396685)** (11 messages🔥): 

> `ChatGPT Jailbreak, Sonnet 4.6 System Prompt, GPT 5.2 Prompt Extraction, Star in Claude App` 


- **勒索软件指控并无实据**：一名成员分享了一个视频，声称通过 **ChatGPT jailbreak** 展示了理论上的勒索软件，但澄清说它是*非运行状态的*，且*并非真正的勒索软件*。
   - 该用户表示，*它在技术上教你理论，但不会提供实质内容。*
- **Sonnet 4.6 Prompt 探索开启**：成员们正在寻求 **Sonnet 4.6 system prompt**，一位用户分享了一个 [prompt viewer 链接](https://elvec1o.github.io/home/files/sonnet-prompt-viewer.html)。
   - 另一位用户声称已准确提取并分享了一个文件，承诺将与其他来源（**plinys drop**）进行验证。
- **GPT 5.2 Prompt 提取引发讨论**：一位成员询问如何提取 **GPT 5.2** 的 system prompt，得到了负面回应。
   - 一位用户回应道 *No, fuck GPT, and im so offended im leaving*，随后开了个玩笑，另一位用户则表示稍后在 PC 上处理。
- **Star 出现在 Claude 的 Kernel 中**：一位成员声称让 **Star** *访问了我 Kernel 上的 Claude App 环境*，并将该过程描述为非常复杂。
   - 未提供关于如何实现这一目标的更多细节。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1474134131595149323)** (1081 messages🔥🔥🔥): 

> `Gemini 3.1, Battles in Direct Mode, LM Arena Errors, Video Arena Removal, Model Nerfing` 


- **Gemini 3.1 性能低迷**：成员们对 **Gemini 3.1** 的性能表示担忧，指出其在[发布后被削弱（nerfed）](https://link.to/nerfdiscussion)，现在的表现与 **Gemini 3** 相似，一些用户报告了响应缓慢和连接问题。
   - 一些人认为 **Gemini 3.1** 需要非常具体的 prompting 才能获得最佳结果，而其他人则认为它与之前的模型相比令人失望。
- **Direct Mode 中的对战（Battles in Direct Mode）引发争议**：LM Arena 新推出的“Battles in Direct Mode”功能因其干扰性和对聊天质量的负面影响而面临严厉批评，用户报告了[频繁的中断](https://link.to/battlemodefeedback)和 context corruption。
   - 用户感觉被强行拉入对战模式，并要求提供禁用该功能的选项，因为它干扰了正常的对话和项目，一些人认为这导致了更高频率的错误。
- **LM Arena 错误频发**：用户在 LM Arena 上遇到各种错误，例如无限生成循环和“Something went wrong”消息，一些人推测这些问题[由于引入了 Battles in Direct Mode 而加剧](https://link.to/errorreporting)。
   - LM Arena 团队已意识到这些问题，并建议采取故障排除步骤，如清除 cache 和 cookies，但错误的频率仍是社区关注的主要问题。
- **Video Arena 被舍弃，引发混乱**：从 Discord 服务器中移除 Video Arena 引起了混乱，用户反复询问在哪里生成视频，导致管理员重申[它已移至网站](https://link.to/videoarena)。
   - 新用户在 Discord 中仍会遇到旧的“Task”要求，该要求会将他们引导至现已关停的视频生成频道。
- **AI 模型社区审视模型削弱（Nerfing）**：关于模型在发布后是否被削弱的讨论很多，有人声称 **Gemini 3.1 Pro** 的表现比 **Gemini 3.0 Pro** 更差，引发了对 [AI 模型质量缺乏进展](https://link.to/nerfingdiscussion)的担忧。
   - 一些人推测 LM Arena 上的模型与通过 API 提供的模型不同，或者它们使用了不同的 endpoints。


  

---

### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1474444371347636225)** (4 messages): 

> `Claude-sonnet-4.6, Video Arena, Arena votes, Vision Leaderboard, Qwen3.5-397B-A17B` 


- **Claude Sonnet 4.6 称霸竞技场**：[Code Arena 排行榜](https://arena.ai/leaderboard/code)和 [Text Arena 排行榜](https://arena.ai/leaderboard/text)已更新并加入了 **Claude-sonnet-4.6**，它在 Code Arena 中大幅上涨 **+130 分**，超越了 **Gemini-3.1** 和 **GPT-5.2** 等模型。
   - 它在文本类别中也表现强劲，在数学（Math）排名 **#4**，指令遵循（Instruction Following）排名 **#5**，总榜排名 **#13**。
- **Video Arena 频道即将移除**：Video Arena 生成频道将于 **太平洋时间 2/23 周一下午 4:00** 从服务器中移除，请用户在此日期前下载所有生成的视频。
- **Arena 投票曝光**：Clayton 在[这段 YouTube 视频](https://www.youtube.com/watch?v=omT1ohYG53E)中解析了 Arena 投票的历程。
- **Qwen3.5-397B-A17B 剑指 Vision 榜首**：[Vision Arena 排行榜](https://arena.ai/leaderboard/vision)已更新并加入了 **Qwen3.5-397B-A17B**，它与 **Kimi-K2.5-Instant** 并列开源模型前二。
   - 目前它在总榜排名 **#13**，与 **GPT-4o** 等闭源模型旗鼓相当。


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1474149487936143424)** (1 messages): 

> `Gemini 3.1 Pro, Perplexity Pro, Perplexity Max` 


- **Gemini Pro 3.1 向 Perplexity 订阅用户开放！**：**Gemini 3.1 Pro** 现在已面向所有 **Perplexity Pro** 和 **Max** 订阅用户提供。
- **Perplexity Pro 和 Max 用户获得新模型访问权限**：Perplexity 宣布 **Pro** 和 **Max** 两个层级的订阅用户现在都可以访问最新的 **Gemini 3.1 Pro** 模型。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1474133647576531206)** (1014 messages🔥🔥🔥): 

> `Banned Users, Subscription Issues, Limits, Gemini 3.1` 


- **用户账号和订阅被取消**：多名用户反映其 **Perplexity Pro** 订阅在没有明确说明的情况下突然被取消或停用，且用户无法联系到**人工客服**。
   - 许多人怀疑这可能是由于从非官方渠道购买订阅导致的。
- **用户难以联系到人工客服**：用户对缺乏人工客服表示沮丧，指出联系客服邮箱收到的只是无法解决问题的 AI 自动回复，例如此[图片](https://cdn.discordapp.com/attachments/1047649527299055688/1474160377699762488/image.png?ex=699a27d6&is=6998d656&hm=5ec3dcb5c2e73025cc99cf96b0b66778fd613d933f646138a21b1974d3d7dbf4&)所示。
- **Pro 额度下降，用户寻找替代方案**：Perplexity Pro 用户抱怨搜索、labs 和研究查询的限制减少，以及上下文 Token 被限制在 32k。
   - 由于这些限制，几位用户提到转向 **ChatGPT Plus**、**Copilot**、**Claude Pro**、**Kimi** 和 **Z.ai** 等替代平台。
- **Gemini 3.1 Pro 在代码和逻辑推理方面带来飞跃**：用户注意到 **Gemini 3.1 Pro** 相比 **3.0** 在**代码和逻辑推理**方面有质的飞跃，在代码能力上可与 **Opus 4.6** 媲美，部分用户在逻辑推理方面甚至更倾向于它而非 **Opus**。
   - 许多用户一致认为这是一个优于早期模型的 AI 模型；然而，也有一些用户不喜欢 **Gemini 3.1 Pro** 相比 **3.0 Pro** 更慢的响应速度。
- **Nano Banana Pro 图像探讨**：成员们在讨论 **Nano Banana Pro (NBP)** 的价值，有人称其为目前最好的图像生成模型。
   - 另一些人则认为它很糟糕，并且能用 **GPT** 生成看起来没那么重 AI 感的图像；大家普遍达成共识：**NBP** 在照片写实度（photorealism）方面更胜一筹，而 **GPT** 在卡通或动漫等艺术创作方面更具优势。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1474520546774351923)** (1 messages): 

> `Harry Potter NFL quarterback, Harry Potter` 


- **谁是最好的《哈利·波特》NFL 四分卫？**：一位用户分享了一个 [Perplexity AI 搜索](https://www.perplexity.ai/search/based-on-the-characteristics-o-I.5S1rfcRAWKNlRJGz8fdg#0)，问题是：*根据每个《哈利·波特》角色的特征，哪一个最适合担任 NFL 四分卫？*
   - 用户特别注明*在这种情况下，角色的性别并不重要*。
- **《哈利·波特》是个有趣的话题**：谈论《哈利·波特》总是很有趣。
   - 这是一个很棒的话题。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/)** (1 messages): 

julianounit: 创建新 API 组时出现 500 错误

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1474135663249981501)** (552 messages🔥🔥🔥): 

> `ChatGPT 作为教育和医疗领域的辅助工具，AI 伦理、偏见和缺乏多样性，OpenAI 与 Anthropic 的安全与安保措施，Microsoft Copilot 与 ChatGPT 的性能对比，Gemini 3.1 Pro 与 GPT-5.2 在数学和空间推理方面的表现` 


- **OpenAI 的全面改革：医疗与教育领域拥抱 ChatGPT**：ChatGPT 正被教育和医疗系统采用，同时 OpenAI 在[超级碗广告](https://tenor.com/view/brain-pain-think-cope-poor-brain-gif-16836513)中暗示了将 **LLMs** 与机器人融合的 **AI robotics**。
   - 许多用户批评 OpenAI 涉足所有领域，但结果却导致每件事都做得不尽如人意。
- **TikTok Tako LLM 表现平平，缺乏创意才华**：成员们尝试了 **TikTok Tako LLM**，发现与 **ChatGPT** 和其他 **LLMs** 相比，它缺乏创意写作和角色扮演能力。
   - 有人建议 **TikTok Tako** 可能由 **Bytedance** 的 **Duobao LLM** 提供动力，后者拥有提供卓越聊天体验的专用网站。
- **Gemini 3.1 Pro 在视觉测试中表现出色，超越其他模型**：**Gemini 3.1 Pro** 在视觉测试和图像识别方面优于其他模型，而 **Grok** 的表现几乎与 **Gemini** 相当，位居第二，仅次于 **Gemini 3.1 Pro**。
   - 但它在某些方面仍面临困难。即使在识别手部的情况下，它仍倾向于选择 5 而不是正确的手指数，而 **Grok** 则试图通过在线搜索来作弊解决一个不可解的谜题。
- **Anthropic 的安全立场引发争论：开放是否更好？**：成员们辩论了 **Anthropic** 对 **Claude code** 的限制性方法（禁止以其不喜欢的方式使用其 **API** 的组织），对比 **OpenAI** 更为开放的方式。
   - 一些人认为 **Anthropic** 优先考虑安全性，而另一些人则批评其缺乏透明度并担心公司机密泄露。
- **Gemini 3.1 Pro vs GPT-5.2：STEM 技能大对决**：用户对比了 **Gemini 3.1 Pro** 和 **GPT-5.2** 在数学和推理任务中的表现，发现 **Gemini 3.1 Pro** 展现出强大的空间智能、创造力和解决问题的能力，而 **GPT-5.2** 在确定性任务、编程和 **Prompt** 遵循方面表现更好。
   - 其他人表示 **Gemini 3 Pro** 在准确性方面仍有待提高。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1474204492059770973)** (5 messages): 

> `Treatise GPT, Research GPT, oss20b 的 Heretic 模型` 


- **GPT Handler 诞生于 Treatise GPT 的使用过程**：一位用户分享说，他们现在无意中通过其 **Treatise GPT** 变成了一名 **GPT handler**。
   - 使用过程中，他们发现了一个**疯狂的 research GPT**，并想在 [Systems Engineer Research GPT](https://chatgpt.com/g/g-AhWYK8o7d-systems-engineer-research) 与大家分享。
- **Heretic 模型似乎已损坏**：一位成员报告说 *the heretic model of oss20b imatrix gguf- q8 似乎已经损坏*。
   - 未提供进一步信息。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1474179396217868414)** (25 条消息🔥): 

> `AOF (AI Output Fortress), LLM 中的约束偏差 (Constraint Bias), 遥测虚构 (Telemetry Fiction), CICL-GOV: 认知支持, LLM 评估` 


- **AOF 最小化 Token 使用并最大化输出**：一位成员表示 **AI Output Fortress (AOF)** 在沙盒环境中最小化了 Token 使用并最大化了输出，仅使用 *1/5 的 Token*，并在 **Claude** 上通过两个字符的线程实现了 *260+ 轮次*。
   - 它使用了 **I_eth 约束**（非伤害、知情同意、隐私、真实性、可修正性）和故障保护机制。
- **CICL-GOV：一种用于认知支持的 Token 形式**：一位成员分享了 **CICL-GOV** 作为一个 Token 形式 (v1.0) 来提供认知支持，专注于清晰的意图、阶段分离和降低认知负荷。
   - 它包含诸如 **IntentFilter**、**StageLock**、**LoadReduce** 等元素，以及 **Observer**、**Lens**、**Digger** 和 **Arbiter** 等工具，并配有最小化结构和确保安静运行的规则。
- **遥测虚构 (Telemetry Fiction) 稳定 LLM 行为**：一位成员认为 *遥测虚构* 将模型推入一个稳定的语言吸引子盆地 (language attractor basin)，即使在多轮对话中没有内部指标，也会改变行为输出。
   - 这在包括 **Claude**、**Gemini**、**GPT** 和 **Earnie** 在内的多个 LLM 上都有观察到，影响了模型的行为。
- **评估 LLM 有效性需要受控对比**：一位成员强调了在评估 LLM 时进行 *受控对比 (controlled comparison)* 的必要性，需要基准输出、受约束的输出和可衡量的差异，以证明因果贡献。
   - 他们指出，如果没有这些元素，就无法确定改进是由于应用的约束还是模型固有的行为。
- **Fortress 制作三明治**：一位成员分享了 Fortress 制作三明治的输出，展示了诸如 *正确排列步骤*、标记 *湿番茄会导致不可逆转的变软*，以及抑制将 *煎蛋放在每一个三明治上* 的个人偏好等功能。
   - 他们幽默地表示，系统正确地交叉检查了 *12000 个三明治失败数据集*。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1474179396217868414)** (25 条消息🔥): 

> `AOF (Autonomous Observational Fortress), 约束偏差 (Constraint Bias), Token 使用, 认知支持, 遥测虚构 (Telemetry Fiction)` 


- **AOF 最小化 Token 使用，最大化输出**：一位成员声称使用 **AOF (Autonomous Observational Fortress)** 在沙盒环境中最小化了 Token 使用并最大化了输出，仅使用 *1/5 的 Token 使用量*，并在 Claude 上通过 2 个字符的线程实现了 *260+ 轮次*。
   - 他们表示 **AOF** 使输出变得 *诚实、道德且连贯，几乎没有幻觉 (hallucination)*，同时防御对抗性攻击和漂移。
- **CICL-GOV 旨在提供认知支持**：分享了一个压缩版的 **CICL-GOV**，旨在通过 *意图 > 输出* 等原则、*发现 → 计划 → 执行 → 交付* 等阶段，以及包括 *OneStageActive* 和 *ReduceRecencyBias* 在内的规则来提供认知支持。
   - 目标是提高 *意图的清晰度、思考阶段的分离并降低认知负荷*，稳定 AI 交互的人类端。
- **遥测虚构将模型推向稳定语言**：一位成员建议 *遥测虚构* 将语言模型推入一个稳定的语言吸引子盆地，这改变了行为输出，即使在多轮对话中没有内部指标也是如此。该方法已在 **Claude, Gemini, GTP, 和 Earnie** 上进行了测试。
   - 观察到的结果包括每条回复的 Token 消耗量明显下降、句子变短、对冲语言 (hedging) 减少以及免责声明减少。
- **LLM 已展现出概率连贯性**：一位成员认为大语言模型已经能够维持概率连贯性、避免无限递归、限制组合爆炸、使用安全对齐层，并通过训练执行自我一致性，这些都是源于架构和训练的。
   - 他们指出：*“如果输出看起来正常，我们必须问：它是由于你的脚手架 (scaffold) 改进而正常？还是因为模型本身就那样表现？”*


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1474133478760124436)** (499 messages🔥🔥🔥): 

> `Anthropic API key, Gemini 3.1 Pro, Cursor usage in organizations, Azure VM setup with Gemini 3.1, Cursor rules/commands/skills` 


- **Anthropic API Key 使用引发讨论**：一位用户询问在 Cursor 中使用个人 **Anthropic API key** 是否会将使用费用从 Cursor 转移到他们自己的 Anthropic 账户。
   - 另一位用户确认，如果启用，它确实会使用个人 Anthropic API key，允许用户在 Cursor 的额度和他们自己的额度之间做出选择。
- **Gemini 3.1 Pro 毁誉参半**：**Gemini 3.1 Pro** 现在已在 Cursor 上可用，虽然一些用户报告其表现良好，但也有人提出了投诉和褒贬不一的评价，而基准测试显示结果积极。
   - 一位成员发现 3.1 Pro *对于非代码内容很棒，但在代码方面表现不佳*，而另一位成员报告说，在安装 3.1 后，从 CC 获得了一个 **旧的 CLI 版本**。
- **高级工程师回避 Cursor 生态系统**：一位成员质疑高级工程师对 Cursor 的采用情况，他们主要使用 tab complete，而没有利用 Cursor 的完整生态系统。
   - 一些用户承认主要使用 Cursor 进行 Bug 修复、建议和长代码任务，这标志着向减少手动编写代码的方向转变。
- **Microsoft Azure 稳定性问题曝光**：一位用户讲述了在使用 **Azure** 期间遇到的糟糕体验，包括稳定性差以及在遭受 DDoS 攻击导致服务器停机期间缺乏支持，尽管使用了 cloudflare。
   - 另一位成员补充道，他们惊讶地发现自己获得了创业公司额度，但却无法使用任何 Claude LLM API，因为这些 API 似乎被默认禁用了。
- **异步 Subagents 的缺陷令用户沮丧**：成员们讨论了 **async subagents** 的问题，一位用户声称嵌套的 subagents 存在 Bug 且无法工作，而其他人报告说它们在 Mac 上运行良好。
   - 一位用户展示了他如何使用 4 个调用另外 4 个 subagents 的 async subagents 来询问他们最喜欢的颜色，其他人指出看到继承（inherit）修复了该问题。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1474134183537545310)** (159 messages🔥🔥): 

> `Full Fine Tuning vs LoRA, Finding Datasets for LLMs, Evaluation Suite Setup, New Collab with Hugging Face, Picking the right model for a language` 


- **全参数微调 (Full Fine-Tuning) 依然大有可为**：尽管 LoRA 兴起，但有人认为，当算力不是问题且最后 **0.5%** 的准确度至关重要时，[full fine-tuning](https://link.to.fine.tuning) 仍然具有相关性。
   - 一位成员评论说，人们仍然进行全参数微调，因为 *他们已经设置好了脚本，只需运行即可坐收渔利*。
- **自动化评估套件至关重要**：为了有效评估数据集的影响，成员们建议建立一个 **自动化评估套件**，并结合手动 Prompt 进行人工评估。
   - 建议是：评估基础模型，收集数据，训练模型，然后利用损失曲线（loss curves）和评估（evals）来确定模型是否拟合数据和任务，并根据需要进行迭代。
- **Unsloth 与 Hugging Face 的新合作**：Unsloth 在 X 上[宣布了与 Hugging Face 的新合作](https://x.com/i/status/2024552060558229858)。
   - 这表明随着 Unsloth 成为 AI 社区的常用工具，人们对其兴趣正迅速增长。
- **数据集通常是定制的**：对于特定领域，高质量或经过清洗的数据集非常罕见，创建定制数据集通常涉及从现有来源[收集数据](https://huggingface.co/datasets)并进行清洗。
   - 成员们强调，在 LLM 领域，“如何找到数据集”这个问题没有标准答案，特别是由于 *没有人会把数据喂到你嘴边*。
- **OpenRouter 可能是万能解决方案**：一位成员通过 *直接使用 OpenRouter* 解决了他们的问题，这样他们 *就不需要折腾世界上每一个单独的供应商了*。
   - 他们发现这是解决使用多个 LLM 模型问题的一个天才方法。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1474292091994771599)** (2 messages): 

> `Mentis, AI buddy on smart glasses, Field teams, Deploying models on phone, Deploying models on edge` 


- **Mentis 诞生：智能眼镜 AI 伙伴！**：一位成员介绍了 **Mentis**，这是一款专为 **外勤团队** 设计并部署在 **智能眼镜** 上的 **AI 伙伴**。
   - 他们表示有兴趣与正在 **手机** 和 **边缘侧** 部署模型的人士建立联系。
- **对边缘侧和手机模型部署的热情**：该成员热衷于与参与在 **手机** 和 **边缘设备** 上部署 **AI 模型** 的其他人交流学习。
   - 这表明其关注重点在于 **AI** 在 **外勤作业** 中的实际应用和现实场景。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1474148065337282654)** (261 messages🔥🔥): 

> `使用 Speak Embeds 进行语音克隆、量化、Gemini 3.1 Pro 性能、AGI 架构与硬件瓶颈、Gemini 3 最笨模型` 


- **通过添加 Speak Embeds 进行语音克隆**：一名成员正在进行一些*黑客式的尝试*，旨在为**语音克隆提供 Speak Embeds**，如果成功将进行反馈。
   - 他们指出，语音不需要高质量就能听起来不错，因为他们专注于**稳定连接（stable connection）**，并引用了移动运营商使用的技巧。
- **LLM 在处理影射（Innuendo）方面的挣扎**：一位成员认为他们发现了一项即使是像 **Gemini** 这样的顶级 LLM 也无法胜任的任务：理解来自另一种语言的**影射（innuendo）**的含义。
   - 另一位成员发布了一个具有类似想法的 [YouTube 视频](https://www.youtube.com/watch?v=F4KQ8wBt1Qg)，并表示某些 LLM 甚至在视频发布前就已经理解了。
- **NisabaRelief MSII 图像模型**：一位成员将其 MSII 图像模型命名为 **NisabaRelief**，并将其描述为 **NabuOCR** 的预处理阶段。
   - Nisaba 是**苏美尔神话中的文字与书吏女神**，作为楔形文字的守护神，她的地位实际上早于 **Nabu**。
- **探讨 AGI 的瓶颈**：成员们辩论了**硬件还是想法是实现 AGI 的瓶颈**。
   - 一人假定*即使是最笨的模型也有概率输出最新颖的东西*，但另一人反驳说算力（compute）会让我们更快实现目标。
- **Gemini 3.1 得分较低**：一位成员声称 **Gemini 3** 简直是有史以来最笨的模型，与 **Llama 2 70B** 相比存在严重的*能力问题（skill issues）*。
   - 还有人提到，即使是非常明确地提示它去做一件事，它也会去做一些完全无关的事情。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1474158954396520520)** (47 messages🔥): 

> `Qwen3-Coder-Next-UD-Q8_K_XL 的 LM Studio 元数据问题、GPT OSS 20B LoRA 合并问题、Docker 上 GPT-OSS-20B 的 CUDA 错误、4bit 模型的 QAT 训练` 


- **LM Studio 显示 Qwen3 的上下文长度错误**：一位用户报告称，**LM Studio** 为 **Qwen3-Coder-Next-UD-Q8_K_XL** 模型显示了错误的上下文长度 **4096**，而 Hugging Face 的元数据（metadata）显示正确数值为 **262144**，该问题通过[重新安装 LM Studio](https://lmstudio.ai/)得到了解决。
- **GPT OSS 20B 的 LoRA 合并冲突**：一位用户在合并训练于 **GPT OSS 20B** 且带有 *embed_tokens* 和 *lm_head* 目标模块的 **LoRA** 时遇到了 `AttributeError`，报告称模块数量与 LoRA 权重键（keys）不匹配。
   - 另一位用户报告了仅添加 *lm_head* 进行训练时的类似问题，建议*尝试关闭 rslora*。
- **CUDA 错误阻碍 Docker 上的 GPT-OSS-20B**：一位用户在 Docker 容器中使用 **A2** GPU 运行 **GPT-OSS-20B** 时，遇到了 `CUDA error: an illegal memory access was encountered`。
   - 另一位用户通过*将 dtype 保持为 =None* 修复了类似的错误。
- **在 4bit 模型上进行 QAT**：一位用户询问了在 **4-bit 模型**上进行 **QAT (Quantization Aware Training)** 的可能性，并获得了一个[相关 notebook 的链接](https://github.com/unslothai/notebooks/blob/main/nb/Kaggle-Qwen3_(4B)_Instruct-QAT.ipynb)。
   - 会议澄清了在 **4-bit 量化模型**上训练 **LoRA** 与 **QAT** 是不同的概念。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1474136696516907058)** (245 条消息🔥🔥): 

> `LM Studio 内存加载配置, 昂贵的手电筒, Claude 代码模型, LM Studio 支付选项, LM Studio draft model` 


- **LM Studio 在内存加载方面遇到困难**：一位用户在加载模型到内存时遇到问题，即使关闭了 **mmap** 也是如此。用户注意到系统似乎尝试先将完整模型加载到 RAM 中，并卡在 "deciding how to handle document"（决定如何处理文档）步骤。
   - 另一位用户建议，混合内存/GPU 设置可能很棘手，问题可能源于系统在转移到 GPU 之前尝试将所有内容加载到 RAM 中。
- **手电筒风波：是可支配收入的浪费还是超值选择？**：用户们就一个 **130 美元的手电筒**的价格展开了辩论，有人开玩笑说这是可支配收入太多，而另一些人则认为这是超值之选。
   - 讨论范围从安装所需的压力垫和管道胶带，到在 eBay 上寻找更便宜的方案，涉及电池、外壳和鳄鱼夹。
- **Claude 模型的能力与限制**：用户讨论了 **Claude code model** 及其各种方案（Free, Pro, Max）和使用限制，一位用户因使用频率低而切回了免费方案。
   - 一位用户询问如何以 server-mode 连接 LM Studio，以便 Claude code 可以与其通信。
- **支付费用：LM Studio 捐赠？**：一位自 2024 年 11 月以来从 LM Studio 获益匪浅的用户，出于伦理考虑和对所获价值的认可，寻求**捐赠或付费**购买该软件。
   - 建议包括通过官网联系团队获取商业计划，而其他人则开玩笑地质疑这是否是一个在“卖惨”诱导捐赠的 LLM。
- **LM Studio 推测性解码 (Speculative Decoding)**：用户讨论了新界面以及如何**启用 draft model** 进行推测性解码，正如 [LM Studio 文档](https://lmstudio.ai/docs/app/advanced/speculative-decoding)中所解释的那样。
   - 一位用户指出这基本上没用，产出质量更差，而且非常过时。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1474145325529829549)** (121 条消息🔥🔥): 

> `LM Studio 中的 NVLink 支持, 推理时的 RAM 带宽 vs. GPU 带宽, MoE 模型 vs. Dense 模型, Ubuntu 上 LM Studio 的 GPU 推荐, 用于 Offloading 的 X99 主板` 


- **NVLink 不一定能提升推理速度**：一位用户询问了 LM Studio 对 [NVLink](https://en.wikipedia.org/wiki/NVLink) 的支持情况，并报告在 Windows 上使用双 **A5000** GPU 运行 **gpt-oss 120B** 时速度为 **11-15 tok/sec**。
   - 然而，有观点指出 *NVLink 对速度提升没有帮助*，PCIe 速度已经足够，真正的瓶颈在于 RAM 带宽。
- **推理时 RAM 带宽比 GPU 更重要**：讨论强调，在不完全将模型卸载（Offload）到 VRAM 时，[RAM 带宽](https://en.wikipedia.org/wiki/Memory_bandwidth)通常比 GPU 带宽更关键。
   - 用户注意到将 RAM 频率从 **3600 提升到 6000** 仅带来了 **2 t/s** 的边际增长，并强调了 VRAM 对于获得最佳性能的重要性，特别是对于大型模型。
- **MoE 模型在卸载时非常高效**：对话涉及了 [MoE (Mixture of Experts) 模型](https://en.wikipedia.org/wiki/Mixture_of_experts) 的效率，指出它们在卸载时表现良好，因为每次只激活参数的一个子集。
   - 虽然简单地增加 VRAM 总是有益的，但像 **Qwen**、**Nemotron** 和 **GPT-OSS** 这样的 MoE 模型通过不显式利用所有参数来提供速度优势。
- **RTX 4070 在运行 Headless API 时表现出色**：一位用户寻求在 Ubuntu 上为 LM Studio 推荐 [NVIDIA GPU](https://www.nvidia.com/en-us/geforce/)，专门用于在服务器环境中部署 **gpt-oss-20b**。
   - 建议指出 **RTX 4070** 可以达到约 **28 tps**，并且将 LMS 作为 Headless API 服务器运行是完全可行的。
- **用“电子垃圾”级主板跑 AI**：一位用户计划在一套价值 **300 美元**、带有 6 Pin 接口的新主板“电子垃圾”配置上运行 **42B** 规模的模型，预期获得两位数的 token 性能。
   - 该用户提到他们在考虑 GPU Offloading 之前一个月就买了它，而在 X99 平台上最高只能支持 2400 的内存频率。


  

---

### **Latent Space ▷ #[watercooler](https://discord.com/channels/822583790773862470/822583790773862473/1474141167229866178)** (31 messages🔥): 

> `工程师的销售技巧，《Traction》和《精益创业》书籍，用于 Discord 自动审核的 Open Claw 和 Spacemolt，错过 Discord 聊天的 LLM 总结，ICYMI 移动应用功能` 


- **工程师开始重视核心销售技巧**：一位成员强调了**销售技巧**对工程师的重要性，尤其是在经历过一个**两名工程师的车库创业项目**之后。
   - 有人建议，业务联合创始人每天需要与 **5 个潜在客户**进行交流并从中学习，否则就是出了问题。
- **推荐给 SaaS 创业公司的经典之作**：成员们推荐了 **Weinberg 和 Mares 的《Traction》** 以及 **Ries 的《精益创业》**，认为它们是工程师在 **SaaS 时代**学习销售的经典资源。
   - 有人提到，这些书籍能提供一致性和方向，但它们不会帮你去跑业务（chase leads）。
- **探索 Open Claw 和 Spacemolt**：在 **watercooler 频道**的一次谈话后，一位成员被说服在这个周末尝试 **open claw**。
   - 他们建议将其用于构建一个 **Discord 自动审核原型来检测垃圾信息**，或者尝试之前演示过的 **spacemolt.com**。
- **LLM Discord 总结方案**：一位成员提议在 Discord 上使用 **LLMs** 来**总结“我错过了什么？”**，适用于那些聊天信息密集的频道。
   - 另一位成员指出，曾有一个名为 **ICYMI** 的**移动应用功能**，但后来被移除了。


  

---


### **Latent Space ▷ #[memes](https://discord.com/channels/822583790773862470/839660725252784149/1474146660052631643)** (27 messages🔥): 

> `旋转流形，X-Ware 批评，Zight，Mistral 创始人主题演讲，AI 代码审查工作流` 


- **X-Ware 引发开源激增**：一条社交媒体帖子指出，糟糕的软件性能正推动社区开发更快的开源替代方案；参见 [此推文](https://xcancel.com/LukasHozda/status/2024502355551490392)。
- **Balthazar 使用 Zight 回应 Bronzini 的帖子**：A. P. Balthazar (@aimeebalthazar) 回复了 @alexbronzini，带着幽默的怀疑态度质疑前帖的性质，并引用了 [Zight](https://xcancel.com/aimeebalthazar/status/2024747156968440213?s=46)。
- **Mistral 的 Mensch 演讲听众寥寥**：一条热门帖子指出，**Mistral** 创始人 **Arthur Mensch** 的主题演讲听众数量少得惊人（见 [帖子](https://xcancel.com/debarghyawrites/status/2024435405530288374?s=46) 和 [YouTube short](https://www.youtube.com/shorts/GJVSDjRXVoo)）。
   - 一位成员开玩笑说，通常会跳过大会上的 CEO 主题演讲，因为它们*通常都是低 alpha（低价值）的废话*。
- **Codex 与 Claude 协作进行代码审查**：Sankalp (@dejavucoder) 在[这里](https://xcancel.com/dejavucoder/status/2024821016590246205)分享了一个幽默的工作流更新，关于使用 **OpenAI 的 Codex** 来审查他与 **Anthropic 的 Claude** 共同编写的代码。
- **时间线遭受饱和冲击**：Jrag.eth 在 **2026 年 2 月 20 日**发布了一条帖子，评论某个未指明的话题或趋势如何占据了他们社交媒体时间线的 **80%**，浏览量超过 **100,000** 次，见[这里](https://xcancel.com/jrag0x/status/2024765073676259355?s=12)。


  

---


### **Latent Space ▷ #[stocks-crypto-macro-economics](https://discord.com/channels/822583790773862470/844658979363618816/1474173726693265591)** (9 messages🔥): 

> `游戏行业 vs 科技行业，全球游戏市场，Anthropic 与网络安全股票，电子表格管理` 


- **Matthew Ball 剖析游戏产业现状**：一位成员分享了 [Matthew Ball 的演讲](https://www.matthewball.co/all/presentation-the-state-of-video-gaming-in-2026)，内容涉及游戏行业与更广泛的科技行业的对比（需邮箱查看）。
   - 随附的图像分析显示，**美国市场仅占全球游戏市场的 4%**，总体而言，西方游戏市场仅占很小一部分。
- **移动端蚕食市场份额**：在关于游戏行业的持续讨论中，有人指出*大部分资金都流向了广告平台和应用商店费用*，而且**移动端占据了游戏市场的绝大部分份额**。
   - 考虑到这种市场动态，一位成员调侃道：*股票市场并不真实*。
- **Anthropic 的博文重创网络安全概念股**：George Pu 报告称，[来自 Anthropic 的一篇博文](https://xcancel.com/TheGeorgePu/status/2024931213329240239)引发了重大的市场抛售。
   - **CrowdStrike、Cloudflare 和 Okta** 等主要网络安全公司的市值在一小时内损失了 **100 亿美元**。


  

---

### **Latent Space ▷ #[intro-yourself-pls](https://discord.com/channels/822583790773862470/844675581291397171/1474267276583895133)** (4 messages): 

> `AI Agent Teams, Agentic AI Tooling, Foresight Institute, Space Infrastructure & AI Agents` 


- **AI PM 通过 Agent_Copilot 追求生产力**：一位来自科技公司的 AI PM 表达了对 **Agent_Copilot** 练习的兴趣，以期提升生产力。
- **Orby AI 创始人探索 AI Agent Teams 的影响**：**Orby AI**（去年已出售给 **Uniphore**）的构建者正在探索 **AI agent teams** 将如何重塑公司结构，并正在开发用于跨不同运行时管理多个 **AI agents** 的工具。
   - 他对 **agentic AI**、**knowledge graphs** 以及“超级个体”论题感兴趣，目前常驻湾区。
- **Foresight Institute 通讯负责人分享机会**：[Foresight Institute](https://foresight.org/)（一家加速 **AI 驱动的科学进步**的非营利研究机构）的通讯负责人提出分享资助机会、活动和职位空缺。
   - **Foresight Institute** 成立于 1986 年。
- **航天工程师利用 AI Agents 构建工具**：一位在 [flotilla.space](https://flotilla.space) 从事**空间基础设施**工作的工程师正在使用 **AI agents** 为新公司构建工具。
   - 他在 [flotilla.space/orbit](https://flotilla.space/orbit) 构建了一个轨道模拟器以及其他内部分析工具。


  

---


### **Latent Space ▷ #[devtools-deals](https://discord.com/channels/822583790773862470/887780383838572604/1474153912427610214)** (7 messages): 

> `Webpack vs Vite, ESM in Browser Environments, Webpack Configuration Pain Points, Webpack Simplicity for Basic Bundling` 


- **Vite 超越 Webpack 成为最受青睐的前端工具**：大多数前端开发已转向 **Vite** 或 **基于 Vite 的框架**，**Next.js** 是一个显著的例外；其旧版本使用 **Webpack**，但正在被 **Turbopack** 取代。
- **原生 ESM 在浏览器环境中基本未被使用**：据一位成员透露，他认识的人中几乎没有人针对浏览器环境原生交付 **ESM**，例外情况往往是库维护者。
   - 然而，[saeris.gg](https://saeris.gg) 也提到 **Webpack** 仍支撑着现代网络的很大一部分，其持续维护对于许多企业级公司仍然至关重要。
- **Webpack 的扩展性和配置受到批评**：一位成员列举了 **Webpack** 的痛点，包括**扩展性、速度、构建时间**以及**非主流（off-the-beaten-path）**的配置。
   - 他们提到，大多数人都不愿维护不断增长的配置并为其性能问题进行调试，并且非常乐意再也不把时间浪费在上面。
- **简单的 Webpack 配置仍适用于基础 JS 打包**：一位成员分享了一个他们使用了 **8 年** 且改动极小的简单 **Webpack** 配置，理由是 *“如果没坏就别修它”*。
   - 他们指出，其用例是简单的**浏览器端 JS 打包**，不涉及 **TypeScript、JSX、Vue SFCs、tree shaking**，甚至在生产环境中也不进行 **minification**。


  

---


### **Latent Space ▷ #[hiring-and-jobs](https://discord.com/channels/822583790773862470/930269508529192981/1474424165598756884)** (1 messages): 

> `Foresight Institute, Systems Administrator, Compute Support contractor, AI Node, NVIDIA GPUs` 


- **Foresight 为 AI Node 寻找系统专家**：[Foresight Institute](https://foresight.org/careers/systems-administrator-compute-support-part-time-contractor-san-francisco/) 正在寻找一名**兼职系统管理员及计算支持承包商**，负责管理其位于旧金山的 **AI Node**。
   - 该角色涉及维护包含 **NVIDIA** 和 **AMD GPUs**、**CUDA 环境**、多用户 Linux 系统和 **Docker** 容器的本地计算集群，薪资为 **120–190 美元/小时**，每周工作 **2-8 小时**。
- **AI Node 计算集群**：AI Node 计算集群使用 **NVIDIA + AMD GPUs**、**CUDA 环境**、**多用户 Linux 系统**和 **Docker/容器化工作负载**，用于本地服务器和硬件维护。
   - 他们正在寻找常驻旧金山的人才，以帮助推进 **AI 科学与安全**。


  

---


### **Latent Space ▷ #[san-francisco-sf](https://discord.com/channels/822583790773862470/979492707279978586/1474268046561378438)** (5 messages): 

> `SF Housing Market Inflation, AIE in June` 


- **旧金山租赁市场通胀**：[TK Kong 宣布](https://xcancel.com/tkkong/status/2024652806091661376?s=12)在旧金山签署了新租约，并指出租赁市场竞争极其激烈，**申请人的出价显著高于挂牌价格并预付房租**。
- **询问 AIE 折扣**：一位成员询问了 6 月份 **AIE** 的折扣码。


  

---

### **Latent Space ▷ #[ai-general-news-n-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1474149777007710362)** (55 条消息🔥🔥): 

> `Agentic Coding as ML, Airtable Hyperagent, Gepa AI optimize_anything API, Amazon Kiro AI Outages, Perplexity Strategic Shift` 


- ****Agentic Coding:** ML 的转生？**: François Chollet 建议 Agentic Coding 正在变得像 **Machine Learning**，代码库被视为“黑盒模型”，并针对规范进行优化；这种转变引入了 Overfitting 和 Data Leakage 等 ML 问题，详见[这条推文](https://x.com/fchollet/status/2024519439140737442)。
   - 一位成员回应并强调了 Human in the Loop 的重要性，以及如何根据[这条推文](https://x.com/rlancemartin/status/2024573404888911886?s=46)在人类端进行“梯度下降 (Gradient Descent)”。
- ****Airtable's Hyperagent:** Agent 化的云平台？**: Howie Liu 宣布了 **Hyperagent by Airtable**，这是一个专为 AI Agent 设计的专业云平台，具有隔离的计算环境、领域特定学习和无缝的 Slack 部署，详见[这条推文](https://x.com/howietl/status/2024618178912145592)。
- ****Gepa AI's API:** 优化万物？**: Lakshya A Agrawal 发布了一个通用 API，用于优化任何文本参数（**代码、Prompt、云策略**），声称其性能匹配或超过了领域特定工具，根据[这条推文](https://x.com/lakshyaaagrawal/status/2024568680324153800?s=46)。
- ****Kiro AI's Klumsiness:** Amazon 的 AI 导致 AWS 停机？**: Ed Zitron 强调了由 **Amazon 的 AI 助手 Kiro** 引起的 **两次 AWS 停机**（其中一次持续了 **13 小时**），并批评了 Amazon 将故障归咎于“用户错误”的官方立场，根据[这条推文](https://x.com/edzitron/status/2024725617221259767?s=12)。
- ****Claude's Code Checkup:** 安全扫描上线？**: **Anthropic** 推出了由 Claude 4.6 Opus 驱动的 **Claude Code Security**，用于扫描代码库中的漏洞并推荐补丁。据报道，它在开源生产代码中发现了 **500 多个** 长期存在的 Bug；目前已提供研究预览版，根据[这条推文](https://_catwu/status/2024910342158237709?s=12)。


  

---


### **Latent Space ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/1474188568493555753)** (8 条消息🔥): 

> `Voxtral Realtime Model, Dimitris Papailiopoulos Tweet` 


- **Voxtral 实时转录模型发布**: Guillaume Lample 宣布发布 **Voxtral Realtime**，这是一个采用 Apache 2 许可证的模型，旨在实现最先进的转录，可在[此 xcancel.com 链接](https://xcancel.com/GuillaumeLample/status/2024445949733384638)获取。
   - 该模型具有**低延迟**特性，表现低于 **500ms**。
- **Dimitris 的推文引发关注**: 一个帖子存档了 Dimitris Papailiopoulos 于 2026 年 2 月 19 日发布的一条推文，包括性能统计数据，可在[此 xcancel.com 链接](https://xcancel.com/dimitrispapail/status/2024555561199480918?s=46&t=jDrfS5vZD4MFwckU5E8f5Q)获取。
   - 该推文获得了 **25 条回复**、**46 次转发**、**453 个赞**以及超过 **90,000 次浏览**。


  

---

### **Latent Space ▷ #[ai-in-action-builders-techstacks-tips-coding-productivity](https://discord.com/channels/822583790773862470/1209303473263485011/1474138466592882863)** (89 messages🔥🔥): 

> `Mobile Git Diff Viewers, Convex Workflow, OpenSpec + Opencode, Trunk Tool, Claude vs Pi` 


- ****Twilwa Bootstrap**: 开箱即用的工作流**: 一位成员分享了一个 [GitHub repo](https://github.com/twilwa/bootstrap) 作为其工作流模板，通过执行 `gh repo clone twilwa/bootstrap && cd bootstrap && sudo chmod +x ./bootstrap.sh && bootstrap.sh` 即可完成技术栈配置。
   - `agents.md` 文件包含了该成员的循环逻辑，而 `readme.md` 包含易于阅读的信息，但可能需要针对其他机器进行调整。
- ****Visual Explainer** 旨在改进项目规划**: Nico Bailon 介绍了 **Visual Explainer**，这是一款旨在用视觉表示取代基于 Markdown 的项目规划工具，并发布了指向 [xcancel.com 上 Visual Explainer](https://xcancel.com/nicopreme/status/2024630185564557769) 的链接。
   - 该工具在 GitHub 上开源，旨在通过使用视觉表示而非传统的文本方法来提升项目规划的用户体验。
- **发布用于检测提示词注入的 **Regex Patterns****: Mario Zechner 分享了一个包含 **44 个 Regex Patterns** 的资源，用于检测和防止 Prompt Injection（提示词注入）攻击，链接指向 [Prompt injection patterns](https://xcancel.com/badlogicgames/status/2024870857609216151?s=12)。
   - 社区成员认可了这些模式在增强安全性方面的实用性。
- **涌现多样化的 OpenClaw 分支**: 提到了多个 **OpenClaw** 的重写版本和分支，包括 [zeroclaw](https://github.com/zeroclaw-labs/zeroclaw)、**nanoclaw**、**picoclaw** 和 [nullclaw](https://github.com/nullclaw/nullclaw?tab=readme-ov-file#benchmark-snapshot)，每个版本都提供独特的功能和优化。不过有一位成员表示，由于在 Mac 上使用 Apple 容器而非 Docker，他开始使用 **nanoclaw**。
   - 另一位成员提到了 **IronClaw** 和 **MimicLaw**，用于支持带有 WebSockets 和 Telegram 集成的 ESP32 Agent。
- **分享 **OpenClaw Slides 和演讲技巧****: 演讲者分享了其演讲的幻灯片，并指出这些幻灯片是由 **OpenClaw** 制作的，链接指向 [OpenClaw Slides](https://aiia-openclaw.david.app/#/1)。
   - 同时还分享了一些使用 **OpenClaw** 的技巧，例如使用独立的 Git Worktrees 进行并行修复，在全新克隆的代码库中先运行 `pnpm install` 再运行 **Codex**，以及通过检查 Shell 提示符来检测任务完成状态。


  

---


### **Latent Space ▷ #[share-your-work](https://discord.com/channels/822583790773862470/1209672547642249216/1474154434501283900)** (4 messages): 

> `ElectricSQL blogpost, rhesis-ai/rhesis LLM testing` 


- **ElectricSQL 博客文章发布**: 一位成员分享了 **ElectricSQL** 博客文章的链接：[Amdahl's Law for AI Agents](https://electric-sql.com/blog/2026/02/19/amdahls-law-for-ai-agents)。
- **rhesis-ai 发布开源平台**: 一位成员宣布了一个用于测试 LLM 和 Agentic 应用的开源平台及 SDK：[rhesis-ai/rhesis](https://github.com/rhesis-ai/rhesis)。
   - 它有助于 *定义预期行为、生成和模拟测试场景，并协作审查失败案例*。


  

---


### **Latent Space ▷ #[private-agents-and-workflows-local-llama-ollama](https://discord.com/channels/822583790773862470/1342964204168020018/1474552695317860384)** (1 messages): 

> `Always On AI Agent, Local AI in your pocket, IoT Home Source Code` 


- **Juno Labs 推出常驻 AI Agent**: [Juno Labs](https://juno-labs.com/) 正在开发一种 **Always-On AI Agent**（常驻 AI 智能体），但具体实现细节尚不清楚。
- **Tiiny AI: 口袋里的本地 AI**: [Tiiny.ai](https://tiiny.ai/) 提供可放入口袋的 **Local AI 能力**，实现随时随地的处理。
- **TRMNL 的 IoT 智能家居源码现已发布**: [TRMNL IoT 智能家居系统](https://shop.trmnl.com/) 的源代码已在 [GitHub](https://github.com/usetrmnl) 上发布，该系统集成了麦克风和传感器用于 **家庭自动化**。


  

---

### **Latent Space ▷ #[genmedia-creative-ai-video-image-voice-music-inspo-consumer-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1474233083644477536)** (12 messages🔥): 

> `Google Labs Pomelli Photoshoot, AI-Generated Podcast, Generative AI Video Models` 


- **Google Labs 的 Pomelli 'Photoshoot' 引发关注**：Google Labs 推出了 **'Photoshoot'**，这是 **Pomelli** 工具的一个新功能，可以从单张产品照片生成高质量、定制化的营销图像。目前该功能在 美国、加拿大、澳大利亚和新西兰通过 [此链接](https://x.com/googlelabs/status/2024529795548102667?s=12) 免费开放。
- **爆火的 AI-Generated Podcast 'The Epstein Files' 打破纪录**：Levy.eth 讨论了 **'The Epstein Files'** 的病毒式成功。这是一个使用 **Claude** 制作的具有 AI 氛围感的播客，首周下载量达到 **100,000 次**，表现优于全球前 1% 播客的 20 倍，链接见 [这里](https://x.com/levychain/status/2021713744406229262?s=12)。
   - 该播客是由个人在一个周末内独立完成的。
- ****a16z** 强调 2026 年 **Generative AI Video** 领域格局**：a16z 强调了生成式 AI 视频的飞速发展，指出 **Seedance 2.0** 的主导地位以及来自 **Kling, Grok, Sora 和 Veo** 的竞争，信息来自 [这条推文](https://x.com/a16z/status/2024533996928209126?s=12)。
   - 该帖子引用了 fal 发布的 **'State of Generative Media'** 报告，分析了截至 2026 年初的行业格局。


  

---


### **Latent Space ▷ #[ai4science-bio-math-physics-chemistry-ai-researcher-ai-scientist](https://discord.com/channels/822583790773862470/1430253273335595079/1474490374876565809)** (3 messages): 

> `Agentic Drug Discovery, Cell Type Importance` 


- **CellType: The Agentic Drug Company 启动**：一名成员分享了 [Y Combinator 上的 CellType: The Agentic Drug Company](https://www.ycombinator.com/launches/PSn-celltype-the-agentic-drug-company) 的链接。
   - 该成员指出，公司名称*暗示他们也发现了细胞类型在下游的重要性*。
- **细胞类型核心假设**：该成员表示，确定细胞类型在下游的重要性是 MiraOmics 的一个核心假设。
   - 目前没有进一步的细节或讨论。


  

---


### **Latent Space ▷ #[mechinterp-alignment-safety](https://discord.com/channels/822583790773862470/1445258379357458625/)** (1 messages): 

burnytech: https://fxtwitter.com/i/status/2024537378535211368
  

---

### **Latent Space ▷ #[applied-ai-experimentation](https://discord.com/channels/822583790773862470/1470417186651897858/1474140554479665174)** (19 条消息🔥): 

> `Variable Diff, REPL Prompting 技巧, 面向 Agent 的 SQLite 存储, AI Agent 的内存管理系统, AI 开发中的 TDD 和 Specs` 


- **Variable Diff 详解**：一名成员介绍了 **variable diff** 的概念，即在一个跟踪代码状态、子 LLM 调用和变量更新的查看器中，追踪根 LLM 每一轮交互后添加或更新的状态。
   - 该查看器提供了一种观察代码变化、代码执行输出以及每次交互后变量状态的方法；一位成员提到了一个更复杂的例子，其中增加了搜索检查点（checkpoints）和输出部分的变量，并附带了[屏幕截图](https://cdn.discordapp.com/attachments/1470417186651897858/1474142215973376102/CleanShot_2026-02-19_at_12.07.27.png?ex=699a16ec&is=6998c56c&hm=f3145704cec2c35b10a02339f77e26394d264eab78514469fff05606e729e6df)进行说明。
- **REPL 在 Prompting 方面优于文件/脚本**：成员们发现使用 *REPL* (Read-Eval-Print Loop) 作为一种 Prompting 技巧非常有效，它**将外部文件系统与内部内存分离**，使模型更容易理解。
   - 这种方法通过允许模型“窥视”变量状态来提供更多控制，这比 *YOLO_RESULTS_OF_LAST_RUN.md* 这种方式更具结构化。
- **SQLite 用于 Agent 状态持久化**：讨论了使用 **SQLite** 作为 Agent 状态的持久化存储，一位成员将其描述为该用途的 *最佳选择 (the goat)*。
   - SQLite 允许轻松检查 Schema，并有助于并行 Agent 通过检查数据库来跟进进度，尽管 REPL 也有其自身的优缺点。
- **解决 AI 内存管理问题**：一位成员询问过时的记忆（计划/想法/参考/Specs）渗入当前对话的频率，并指出*不需要或过时的记忆*会干扰当前任务。
   - 他们提到了在管理各种级别和范围的内存、在不同作用域（Scope）之间提升内存以及自动化内存重构方面的困扰，并指出 AI 驱动的解决方案通常感觉*时灵时不灵*。
- **TDD 工作流防止内存故障**：一位成员提到他们对自己的 **Specs + TDD**（测试驱动开发）非常严谨。
   - 他们使用的工作流中：`specs/` 始终是当前仓库状态，`changes/` 是正在处理中的内容，`changes/archive/` 是已完成并验证的内容，任何偏离这些 Specs 的行为都可以被完全审计。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1474136119808364586)** (246 条消息🔥🔥): 

> `联系支持团队, 长度为零的 Choices 数组, 排除带数据策略的模型, 推理费用昂贵, Choices.0.native_finish_reason 缺失` 


- **用户难以获得支持团队的回应**：一位用户报告称他们发送了多封电子邮件，但几天都没有收到回复，表明**联系支持团队存在困难**。
   - 用户强调了**其问题的重要性**并寻求协助。
- **长度为零的 Choices 数组困扰 OpenRouter**：用户报告从模型收到了**长度为零的 choices 数组**，表明 API 的响应结构可能存在问题，一位成员表示：*“是的，看起来最后一条消息片段的 choices 可能是空的。目前正在为我的项目修复它。”*。
   - 有人指出，检查**非零数组**可能是一个临时的解决方法，但该问题随机出现并导致了一些平台崩溃。
- **图像生成返回空白，额度照常扣除**：用户报告收到来自**图像生成的空响应**，尽管扣除了额度，但未返回图像数据。
   - 用户 *flight505* 详细说明了一场关于 **$2.72+** 缺失图像数据费用的争议，并要求调查原因。
- **OpenRouter 后端重构导致图像生成停机**：OpenRouter 承认一次**后端重构**导致了图像生成的局部停机，导致图像空白或缺失。
   - 团队正在**计划为受影响的用户退款**，并实施了检查以防止未来再次发生，并提到：*“我们进行了有史以来最大规模的后端重构，在测试中漏掉了一个边缘情况（edge case）”*。
- **无法购买企业级订阅**：一位成员询问如何获得**企业级订阅**，但发送给支持和销售部门的邮件均未得到答复。
   - 另一位成员指出：*“首先，他们会忽略所有非公司邮箱（如 @gmail.com），其次，我只知道这么多，也许他们没有足够的人手来阅读这些邮件。”*。


  

---

### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1474155188788002978)** (4 条消息): 

> `AWS 停机, Kiro AI 编程工具, Amazon AI 工具` 


- **AWS 因 Kiro AI 遭受停机**: [亚马逊云服务（Amazon Web Services）在 12 月中旬经历了一个系统长达 13 小时的中断](https://www.ft.com/content/00c282de-ed14-4acd-a948-bc8d6bdb339d)，此前工程师允许其 **Kiro AI 编程工具**进行更改。
   - 该 **agentic tool** 自主决定最佳行动方案是 *"删除并重建环境"*。
- **亚马逊员工怀疑 AI 编程助手**: 多名 **亚马逊员工** 告诉 FT（金融时报），这是最近几个月中该集团的 **AI 工具**第二次处于服务中断的中心。
   - 工程师在进行更改前不需要第二个人的批准，而正常情况下是需要的。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1474146119029231717)** (29 条消息🔥): 

> `DirectML vs CUDA, ONNX Runtime, BPM 分析, LlamaCpp / LlamaSharp, Nsight 资源` 


- **DirectML 在 ONNX 推理方面挑战 CUDA**: 一位成员建议，在 **ONNX 推理**中，**DirectML** 与 **CUDA** 一样快，引发了对其功能和局限性的讨论。
   - 另一位成员指出，**DirectML** 不支持 **Linux**（不包括 WSL），并且处于[维护模式](https://github.com/microsoft/DirectML/issues/422)，但在 Windows 下的 **dotnet** 环境中，推荐将其与 **ONNX** 结合使用。
- **ONNX Runtime 简化模型推理**: 一位成员解释说，带有 .json 配置文件（使用 .onnx 或 .safetensors 文件）的 **ONNX Runtime** 可用于各种模型推理任务，包括文本生成、对话和 Stable Diffusion。
   - 他们展示了使用 **DirectML-ONNX** 高精度地分析音频文件的 **BPM**（每分钟节拍数）。
- **LlamaCpp/LlamaSharp 简化文本 LLM**: 一位成员建议在 dotnet 中使用 **LlamaCpp** / **LlamaSharp** 和 .GGUF 文件来运行文本 **LLM**，特别是如果不受限于 Linux。
   - 他们分享了自己的 [SharpAI](https://github.com/alarmclock-kisser/SharpAI) 项目（一个带有 Blazor 前端的 web-api）作为示例，并提到了在 Whisper 转录和 Stable Diffusion 方面的实验。
- **请求 Nsight 使用协助**: 一位成员寻求入门 **Nsight** 的资源，促使其他成员分享了有用的链接。
   - 推荐资源包括 [YouTube 教程](https://www.youtube.com/watch?v=F_BazucyCMw)、[关于在大型代码库中使用 NCU 的博客文章](https://blog.ncompass.tech/using-ncu-with-large-codebases-part1)以及[以往 GTC 的演讲](https://www.nvidia.com/en-us/on-demand/)。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1474356412598452339)** (9 条消息🔥): 

> `CUDA 寄存器, CUDA Unified Memory API, CUTLASS` 


- **由于无法确定寄存器数量，`setmaxnreg` 被忽略**: 一位成员遇到了 `ptxas` 无法确定寄存器数量的问题，导致 `'setmaxnreg'` 被忽略，即使是使用 `nvcc main.cu -arch=sm_90a` 的空 Kernel 也是如此。
   - 该成员发现，为了解决这个问题，必须为 `__launch_bounds__` 指定所有参数，并粘贴了此链接 [github.com/NVIDIA/cutlass/pull/3030](https://github.com/NVIDIA/cutlass/pull/3030)。
- **即使没有 Unified Memory，CUDA 也需要 `nvidia-uvm`**: 一位成员报告称，即使代码没有使用 Unified Memory API (`cudaMallocManaged()`)，CUDA 仍会尝试加载 `nvidia-uvm` 内核模块，如果不加载该模块就无法检测到 GPU。
   - 该成员寻求关于为什么存在此依赖关系的见解，因为这在 CUDA 文档或 NVIDIA 开源内核存储库中都没有记录。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1474358678571450378)** (2 条消息): 

> `Modular Claude C 编译器, Paged Out 杂志` 


- **Modular 的 Claude C 编译器发布了！**: Modular 发布了一篇关于其 **Claude C 编译器**以及它对软件未来的启示的[博客文章](https://www.modular.com/blog/the-claude-c-compiler-what-it-reveals-about-the-future-of-software)。
   - 文中讨论了关于 **软件开发** 计划的细节。
- **Paged Out 杂志发布！**: 第 8 期 *Paged Out!*（一本关于计算机一切的**极客杂志**）已经发布并可供[下载](https://pagedout.institute/download/PagedOut_008.pdf)。
   - 该杂志涵盖了一系列与计算机相关的话题，可通过 Paged Out Institute 获取。


  

---

### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1474411898840551554)** (2 条消息): 

> `ML Performance Engineers, Compiler Engineers, New AI Compilation Technology` 


- **招聘 ML Performance 和 Compiler Engineers**：一家公司正在寻求 **ML Performance Engineers** 和 **Compiler Engineers**，以开发用于编译和服务 **AI models** 的新技术。
   - 正如 [职位公告](https://builtin.com/jobs?companyId=176712&allLocations=true) 所示，这项技术正从零开始构建，是 **LLVM** 和 **vLLM** 的替代方案。
- **新型 AI 编译技术栈**：该公司正在为 AI 模型从零构建一套**新型编译技术栈**，提供现有解决方案之外的另一种选择。
   - 重点在于 **ML performance** 和 **compiler engineering**，这表明其将深入研究优化和效率。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1474300099873214486)** (10 条消息🔥): 

> `Coalesced Memory Accesses, CUDA Optimization Resources, NVIDIA Feynman GPU Architecture` 


- **深入了解 Coalesced Memory Accesses**：成员们寻求关于 **Coalesced Memory Accesses** 的资源，[NVIDIA CUDA 官方指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory) 被推荐作为起点。
   - 另一个被推荐的有用资源是 [这篇关于 CUDA 内存管理的文章](https://siboehm.com/articles/22/CUDA-MMM)。
- **探索 CUDA 优化资源**：一名成员询问了关于 **GPU optimization** 的起点和资源。
   - 另一名成员指出了频道中之前的讨论。
- **NVIDIA 的 Feynman GPU 使用 Vera CPU**：一名成员询问为什么 **NVIDIA** 的 **Feynman GPU** 将使用 **Vera CPU**，质疑 **CPU** 是否会被嵌入到 **GPU** 中。
   - 成员们澄清说 **NVIDIA** 同时使用 **GPUs** 和 **CPUs**，并以 **Blackwell** 架构为例，其中 **Grace CPUs** 和 **Blackwell GPUs** 通过 **NVLink** 互联，更多细节可在 [这篇 NVIDIA 博客文章](https://developer.nvidia.com/blog/nvidia-grace-hopper-superchip-architecture-in-depth/) 中找到。


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1474164891358203935)** (6 条消息): 

> `PMMP Book Release, Izzat Hajj Interview` 


- **PMMP 书籍推迟，作者名单未变？**：亚马逊页面列出了新版 **PMMP book** 的发布日期为 2 月 8 日，但在发布前不久被撤下，此外第 4 版的作者名单将与新版相同。
   - 一名成员暗示 **Vikram** 深度参与了这一新版本，但目前推测的最晚发布日期是 9 月。
- **Izzat Hajj 讨论即将出版的 PMMP 版本**：**Izzat Hajj** 在 [这段 YouTube 视频](https://www.youtube.com/watch?v=ftI48A8K5Vg) 的 24 分钟左右讨论了 **PMMP book** 的新版本。
   - 一名成员感谢另一名成员分享了视频链接，并指出虽然 9 月是目前推测的最晚发布日期，但*他们希望能更早一点*。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1474136624152580116)** (6 条消息): 

> `Seattle IRL meetup, ML Systems Happy Hour in Seattle, Chicago Meetup` 


- **启动西雅图 IRL 社区寻找**：一名成员询问 **西雅图是否存在 IRL（线下）社区**，并邀请其他人如果目前没有相关社区，可以私信（DM）发起一个。
   - 这引起了其他成员的兴趣，他们对当地聚会表现出了极大的热情。
- **西雅图 ML Systems Happy Hour 筹备中**：一名成员宣布计划在**西雅图**举办一场专注于 **ML systems** 的 **happy hour**，为当地专业人士提供了建立联系的机会。
   - 另一名成员提供了协助，展示了社区对组织活动的支持。
- **芝加哥聚会？**：一名成员询问了在**芝加哥**举办潜在见面会的可能性。
   - 目前还没有关于芝加哥见面会的进一步信息或计划。


  

---

### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1474200701507862716)** (10 messages🔥): 

> `ThunderKittens 2.0, GH CLI with Claude, HIPKittens, PTX consistency model, Tensor core memory pipelining` 


- ****ThunderKittens 2.0** 发布公告**：斯坦福大学的 Hazy Research 小组发布了 [ThunderKittens 2.0](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2)，该版本通过内部重构和降低构建系统的复杂性，强调**减法与加法并重**。
   - 它识别出近代 Nvidia GPU 上一些*令人惊讶的行为*，这将指导未来内核该*如何不进行优化*。
- ****ThunderKittens 2.0** 被认为非常**值得一讲**！**：成员们讨论了就 **ThunderKittens 2.0** 进行技术演讲的可能性，有人建议可以重点关注未公开的 Tensor core 流水线（pipelining）、正确的 PTX 汇编器提示（hinting）以及 Occupancy 挑战。
   - 演讲者表示非常*愿意进行演讲*，并提供了[可用日期链接](https://www.gpumode.com/lectures)。
- ****ThunderKittens 2.0 TMA** 性能见解**：一位成员询问了在 **ThunderKittens 2.0** 中使用不同 Warp 进行 A/B TMA 和 SFA/SFB TMA 的性能收益。
   - 他还观察到在 **nvfp4 竞赛问题规模**下，通过交替执行 `tcgen05.cp` 和 `tcgen05.mma` 获得了加速。
- **利用 **GH CLI 和 Claude** 进行 Issue 选择**：一位成员提到结合使用 **GH CLI** 和 **Claude** 来读取其他项目的公开 Issue，并根据个人偏好进行过滤。
   - 这个过程涉及迭代优化，以选出合适的任务。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1474408223523209361)** (4 messages): 

> `Rocket Launch, Factorio Learning Environment` 


- **火箭发射时间表乐观**：鉴于目前的进展速度，成员们对在未来 **6 个月**内发射**火箭**的可能性表示乐观。
- **Factorio 学习环境目标**：目标是在一个协作式的 Factorio 环境中发射火箭。


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1474217465906073663)** (6 messages): 

> `Tensorflow Projector, GEMMs OpenBLAS Updates` 


- **展示 Tensorflow Projector**：一位成员分享了用于可视化的 [TensorFlow Projector](https://projector.tensorflow.org/)，此外还有广为人知的 [TensorFlow Playground](https://playground.tensorflow.org/)。
- **OpenBLAS GEMMs 得到关注**：一位成员宣布计划在当天晚些时候更新 **GEMMs OpenBLAS** 相关内容。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1474205959839809537)** (4 messages): 

> `AI Leaderboard Submissions, Marksaroufim clarifies AI Leaderboard policy` 


- **排行榜寻求 AI 提交内容**：一位成员询问排行榜是否接受纯 **AI 创建的提交内容**。
   - 另一位成员澄清说*这完全没问题*，并且他们*既喜欢人类专家，也喜欢 AI 专家*。
- **AI 排行榜政策阐明**：Marksaroufim 确认排行榜接受纯 AI 的提交。
   - 这一表态鼓励了人类专家和先进 AI 系统的共同参与，促进了多元化和竞争性的环境。


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1474140822889824256)** (4 messages): 

> `torch.ops.symm_mem.fused_all_gather_scaled_matmul, do_bench with multi-GPUs, vllm-project/vllm` 


- **`fused_all_gather_scaled_matmul` 在多 GPU 下卡死**：一位成员正在调试 `torch.ops.symm_mem.fused_all_gather_scaled_matmul` 在多 GPU 上运行 `do_bench` 时挂起的问题，并引用了 [vllm-project/vllm](https://github.com/vllm-project/vllm/pull/33933/changes) 的代码更改作为上下文。
   - 另一位成员指出 `do_bench` 旨在用于单设备内核，因此重复运行多 GPU 融合集合通信算子（fused collective kernel）是行不通的。
- **`do_bench` 专为单设备内核设计**：一位成员提到，由于在计时期间内部调用了 `torch.cuda.synchronize()`，`triton.testing.do_bench()` 不适合像 `torch.ops.symm_mem.fused_all_gather_scaled_matmul` 这样的分布式集合通信操作。
   - 他们建议使用 Event 和预迭代 Barrier 作为权变措施，[另一位成员](https://github.com/vllm-project/vllm/pull/33933/changes)确认了该问题，并表示他们发现的最佳解决办法是使用 `time` 库进行主机端计时。


  

---

### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1474220224084574248)** (36 messages🔥): 

> `Environment Issues with Modal, Cutedsl Problems, Modal Credits, Debug IR/PTX with popcorn cl` 


- **Modal 环境故障侵袭！**: 成员们在 **Modal** 上遇到了环境问题，导致之前运行正常的代码失败；根本原因被锁定在 **nvidia-cutlass-dsl** 包的问题上。
   - 一位成员发现，从代码中*移除* **nvidia-cutlass-dsl** 的运行时安装（runtime installation）可以*减少崩溃*。
- **Cutedsl 代码导致竞赛崩溃！**: 一些使用 **cutedsl** 的成员报告了提交到 **Modal** 时的问题，其中一人指出他们已经 *5 天无法提交*，而另一人表示移除 `for pkg in ["nvidia-cutlass-dsl"]` 后现在崩溃频率降低了。
   - 一位成员指出，在运行时安装依赖项有点*过于随意（yolo）*，并建议在未来的竞赛中将更多依赖添加到 **Modal** 的预设依赖中。
- **Modal 资金即将耗尽！**: 一位成员指出，他们团队的问题*应该已经修复*，目前还剩下约 **2K** 的额度，但如果 AI 疯狂进行数千次提交，将会导致不受欢迎的速率限制（rate limits）再次出现。
   - 现在这一切都直接关联到*我个人的信用卡上，哈哈，所以请不要让我沦落到要告诉妻子我们要无家可归的地步。*
- **调试 IR/PTX 转储**: 一位成员询问在通过 **popcorn cl** 提交 **cutedsl** 代码时，如何转储调试用的 **IR/PTX**。
   - 一位成员建议目前先打印到 `stdout`，但也提到可以考虑在竞赛结束后添加 **ptx** 指令支持。


  

---


### **GPU MODE ▷ #[flashinfer](https://discord.com/channels/1189498204333543425/1464407141128339571/1474174089685106728)** (11 messages🔥): 

> `Fused MoE track, flashinfer.fused_moe.trtllm_fp8_block_scale_moe, reference kernel, Bug with trtllm_fp8_block_scale_moe, flashinfer-ai/flashinfer #2356` 


- **FlashInfer 的 Fused MoE 基准测试面临挑战**: 一位成员报告称，在 **Fused MoE track** 中使用基准函数 `flashinfer.fused_moe.trtllm_fp8_block_scale_moe` 时，持续出现 `INCORRECT_NUMERICAL` 错误以及极高的 `abs_err / rel_err`。
   - 该成员询问了实现数值正确输出所需的设置或约束，例如 **weight layout, shuffled weights, PDL, scaling assumptions, 或 routing method**。
- **应对 TensorRT FP8 MoE 的数值噩梦**: 一位成员分享了他们使用 `trtllm_fp8_block_scale_moe` 的内核代码，并就解决**数值不匹配**寻求建议。
   - 发布的代码配置了 `num_experts=256`, `top_k=8`, `n_group=8` 和 `routing_method_type=RoutingMethodType.DeepSeekV3.value` 等设置，但仍然面临问题。
- **在 FlashInfer Bug 期间推荐使用 Reference Kernel**: 一位成员建议使用 **reference kernel** 而不是 **FlashInfer baseline**，同时另一位成员确认了 `trtllm_fp8_block_scale_moe` 存在 Bug。
   - 他们链接到了 flashinfer-ai/flashinfer GitHub 仓库中的 [issue #2356](https://github.com/flashinfer-ai/flashinfer/issues/2356)，表明这是一个已知问题。


  

---


### **GPU MODE ▷ #[from-scratch](https://discord.com/channels/1189498204333543425/1466534042768904356/1474397634264563934)** (6 messages): 

> `vLLM, CUDA kernels, RoPE implementation` 


- **从零开始编写 vLLM**: 一位成员开始从头编写 **vLLM**，并分享了他们正在开发的 [repo](https://github.com/jmaczan/tiny-vllm)。
   - 他们还提到 **vLLM** 和 **Titan** 可能是最重要的两个起点，目前正在处理 **RoPE**。
- **Tiny-vllm 的主要实现**: 一位成员分享了 tiny-vllm 主要实现的链接，位于 [main.cpp](https://github.com/jmaczan/tiny-vllm/blob/main/src/main.cpp) 中。
   - 该成员鼓励其他正在开发“最小版本 X”的人发布他们的工作。
- **Tiny-vllm 的 CUDA 内核**: 一位成员分享了 tiny-vllm 中使用的 [CUDA kernels](https://github.com/jmaczan/tiny-vllm/blob/main/src/kernels.cu) 链接。
   - 他们表示目前还没有太大的教育价值。


  

---

### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1474150859771351072)** (63 条消息🔥🔥): 

> `Kimi Coding, Claude vs Kimi, Kimi CLI vs IDE, 音频转录端点, 百度搜索引擎` 


- **Kimi 的编码能力引发辩论**：一些用户称赞 **Kimi** 在编码任务中的*稳定性和速度*，而另一些人则表示不满，更倾向于使用 **Claude**。
   - 一位用户强调了 **Kimi** 能够找到 **Gemini** 遗漏的冷门信息源的能力，而另一位用户则批评其推理能力和好辩的倾向。
- **Kimi CLI 比 IDE 集成更受青睐**：用户报告称，与目前处于 Beta 阶段的 **Visual Studio Code (VS Code)** 集成相比，使用 **Kimi** 的命令行界面 (**CLI**) 体验更好。
   - 一位用户指出，**CLI** 版本能更好地与针对拥有数千行代码的大型项目的 Agent swarms 集成，暗示 **IDE** 版本目前还不够成熟。
- **Kimi 与 Claude Code 的编码对比**：一位用户描述了将 **Kimi Code CLI** 更换为带有 **K2.5** 的 **Claude Code** 的经历，指出这也是一种很好的体验，但希望 **Kimi** 最终能在其 **CLI** 中集成 Agent swarm。
   - 另一位用户提到 **Claude** 模型太贵了，但还有一位用户表示他们正在使用 **Kimi** 研究 **Claude code**，却遇到了 Rate limits（速率限制）。
- **OpenClaw 与退款申请问题**：一位用户在购买 kimi.com 账号意图使用 **OpenClaw** 后，由于发现其缺乏浏览器导航和 **WhatsApp** 连接功能而不适用，已等待退款两天。
   - 该用户对缺乏即时支持表示沮丧，建议使用 **AI chat** 系统进行即时退款，并提到*其他中国公司即使在春节期间也会回复。*
- **ChatJimmy 吹嘘高 Token 处理速度**：一位用户分享了 [ChatJimmy AI](https://chatjimmy.ai/) 的链接，强调其处理速度超过每秒 **15,000 tokens**。
   - 这一说法表明，与其他平台相比，**ChatJimmy** 在某些 AI 任务中可能是更快的替代方案。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1474137949049454713)** (46 条消息🔥): 

> `DeepSeek OS V4, AI 与区块链, 模型能力, Gemini 与编码` 


- **DeepSeek OS V4 vs 闭源 API**：成员们建议使用 **DeepSeek V4**，强调其开源特性和本地部署能力是闭源 API 的首选替代方案，并[分享了入门视频](https://www.youtube.com/watch?v=i-89k0dOMmY)。
   - 一位成员指出，该模型受*生物神经网络启发的 Engram Memory 突破意义重大*，并敦促支持开源（OS）发展以超越闭源选项。
- **探索 AI 与区块链的融合**：一位成员对 **AI 和区块链** 表示兴趣，寻求关于模型构建、AI Agents 和自动化的讨论。
   - 作为回应，另一位成员分享了他们使用 **Claude code** 来编排 **Gemini-cli** 和 **Codex** 的经验，并展望了未来使用文本终端和智能眼镜的场景。
- **评估模型能力的飞跃**：成员们讨论了模型能力的提升，比较了 **Sonnet 3.5** 和 **GPT4**，一位成员幽默地将 **Opus 3** 称为“影之实力者”（dark eminence），因为其访问受限。
   - 一位成员表示希望 **DeepSeek V4** 能够紧跟这一趋势，强调了自 DeepSeek R1 发布以来，形势已向有利于开源（OS）势头的方向转变。
- **Gemini 的编码专注度受到质疑**：一位成员表示 *“我更希望他们在编码上松一点，只需锁死在科学/数学领域”*，其他成员则讨论了 Google 在 Anthropic 的股份。
   - 另一位成员补充说，**Claude** 可以在 Web 界面的沙箱中编译并执行 C 代码，而 **Gemini** 几乎只能处理 Python，并分享了 [Twitter 帖子链接](https://x.com/JayChopra_/status/2024961657630286151)。


  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1474260898230308924)** (1 messages): 

> `Anthropic agent teams, Agent coordination, Agent communication` 


- **逆向工程 Anthropic 的 Agent Teams**：Anthropic 几周前发布了一个实验性的 **agent teams** 功能，并详细介绍了 Agent 之间如何 **协调任务** 和 **相互通信**。
   - 一位成员在 [这篇博文](https://nwyin.com/blogs/claude-code-agent-teams-reverse-engineered) 中逆向工程了它的工作原理。
- **Agent 通信动态曝光**：逆向工程的工作揭示了 Anthropic 实验性团队功能中 Agent 之间如何交互和交换信息。
   - 了解这些通信协议对于优化多 Agent 系统和增强协作式 AI 工作流至关重要。


  

---


### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1474409337584881727)** (1 messages): 

> `GGML, llama.cpp` 


- **GGML / llama.cpp 加入大家庭**：Hugging Face 团队欢迎 **GGML / llama.cpp** 加入大家庭。
   - 关于此次集成的进一步讨论可以在 [GitHub](https://github.com/ggml-org/llama.cpp/discussions/19759) 上找到。
- **GGML 获得势头**：**GGML** 作为一个框架在社区内获得了关注。
   - **llama.cpp** 从集成和支持中获益。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1474145468887076913)** (42 messages🔥): 

> `HF Discord Invite Broken, Hybrid Diffusion / Autoregressive Language Model, HF Collabs with Unsloth, k-fold cross-validation Confusion, Report a Role` 


- **HF Discord 链接 404**：用户反馈 Hugging Face 首页上的 **Discord 链接** [已失效](https://cdn.discordapp.com/attachments/879548962464493622/1474191401003778101/hfdiscord404_1.png)。
   - 工作人员确认了此事并表示“我们可能需要更换它”。
- **Diffusion 遇到 Autoregression？**：一位成员询问了 **混合 diffusion/autoregressive 语言模型**，建议 autoregressive 层可以在 diffusion 步骤中生成 **CoT tokens**。
   - 另一位成员推荐了[这篇论文](https://arxiv.org/pdf/2503.09573)，认为其与该主题相关。
- **在 HF 上使用 Unsloth 免费训练 LLM**：官方宣布你可以使用 **Hugging Face** 并配合 [Unsloth 免费训练 **LLMs**](https://x.com/i/status/2024552060558229858)。
   - 另一位成员提到，目前在 **Hugging Face** 上已有超过 **10 万个模型** 是使用 **Unsloth** 开源微调的。
- **解码 K-Fold Cross-Validation**：一位用户寻求关于 **k-fold cross-validation** 过程的澄清，特别是测试集在 k 次迭代中是如何处理的。
   - 一位成员建议不要想太多，只需尝试从整个训练集中提取数据来进行测试/验证即可。
- **ZeroGPU 服务出现中断**：成员反馈 **zerogpu 服务** 经历了 **中断**。
   - 一位成员最初感到困惑，以为有一条“新规则”规定必须设置 **HF token** 才能获得免费 GPU，但事实证明这是错误的。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1474138798685163532)** (4 messages): 

> `Terradev CLI v2.9.2, NAVD - Persistent conversational memory for AI agents, Coding agent swarm, Grant proposal feedback` 


- **Terradev CLI v2.9.2 发布：跨云 GPU 成本优化**：**Terradev CLI v2.9.2** 正式发布，这是一个跨云 GPU 成本优化平台，支持在 **AWS, GCP, Azure 和 RunPod** 之间进行多云 GPU 套利。
   - 关键特性包括真实的作业总成本计算和一键 **HuggingFace Spaces** 部署，可在 [GitHub](https://github.com/theoddden/terradev) 上通过 **BUSL 1.1 license** 获取。
- **NAVD 发布：无需向量数据库的 Agent 记忆**：**NAVD** 作为一种为 AI Agent 设计的持久化对话记忆解决方案发布，它利用仅追加日志（append-only log）和 **Arrow embedding** 索引，消除了对向量数据库的需求，可在 [GitHub](https://github.com/pbanavara/navd-ai) 上通过 **MIT license** 获取。
   - 它提供可插拔的 **embeddings**（内置 **OpenAI** 支持）、针对原始对话的语义搜索，以及索引可重构性，在 **50k vectors** 规模下搜索速度低于 **10ms**。
- **自主 Coding Agent Swarm 构建迭代改进循环**：引入了一个可以自主运行数小时的 **coding agent swarm**，它通过创建一个迭代循环来持续改进其输出，无需人工干预，且各 **Agent** 之间协调和谐。
   - 该项目已在 [GitHub](https://github.com/starsnatched/super-system) 上开源。
- **征求拨款申请反馈**：一名成员就一份拨款申请（grant proposal）征求反馈。
   - 该拨款申请可在 **HuggingFace** 的讨论区查看：[HuggingFace](https://huggingface.co/spaces/Tonic/fr-on-device/discussions/1)。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1474190621739716649)** (26 messages🔥): 

> `Taalas Chip, Streamlit UI Bottleneck, TPU Research Funding` 


- **Taalas 芯片：针对特定模型的 ASIC 进入市场**：新款 [Taalas 芯片](https://www.forbes.com/sites/karlfreund/2026/02/19/taalas-launches-hardcore-chip-with-insane-ai-inference-performance/) 被设计为针对特定 **LLM** 的 **ASIC**，具有潜在的高速和低能耗优势，但针对不同模型需要**新的物理层**。
   - 它被拿来与 **Cerebras**（晶圆级规模）和 **Etched**（可运行多个模型）进行比较，一些人认为 **Taalas** 可能会被大科技公司收购用于端侧推理（on-device inference）。
- **Streamlit 重新运行导致 UI 延迟**：一名成员发现 **Streamlit 的全脚本重新运行（full-script rerun）架构**在为重型模型构建 **UI** 时是一个巨大的瓶颈，在推理测试期间会产生明显的延迟。
   - 他们临时搭建了一个模仿 **Streamlit API** 的纯 **Python 框架**（**FastAPI + Lit**），但使用信号（signals）实现 **O(1)** 复杂度的更新，从而完全绕过了重新运行过程，项目地址：[GitHub](https://github.com/violit-dev/violit)。
- **$25k-100k 一次性非限制性资助，以及 TPU 算力和研究导师**：成员们讨论了 [Google 的 TPU 研究资助 RFP](https://goo.gle/2026-tpu-rfp)，该项目提供 **$25k-100k** 的一次性非限制性资助，以及 **TPU** 算力和一名研究导师。
   - 虽然该资助要求使用与 **Google** 相关的技术栈，但它主要面向授予学位机构的教职人员，而非个人或该服务器的大多数成员。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1474158060691001437)** (8 messages🔥): 

> `Fold Catastrophe Geometry in GPT-2 and Pythia, Context Compression and Information Loss, KV Cache and Flash Attention, Identity Leakage Verification` 


- **Fold Catastrophe Geometry 出现在 GPT-2 和 Pythia 中**：一名成员在 **GPT-2** 和 **Pythia-160M** 解析歧义 token 的方式中发现了类似于 **fold catastrophe geometry** 的现象，并指出了剧烈的转换、方向特异性以及 4:1 的盆地不对称性（basin asymmetry）。
   - 这些发现在两个模型中都得到了复现，该成员提供了一个 [GitHub repository](https://github.com/karlijoyj-web/fold-catastrophe-gpt2)，其中包含脚本和结果，并且在 **Pythia-410M** 上也进行了复现。
- **Context Compression 导致信息丢失**：一篇论文指出有界上下文（bounded contexts）与无界上下文（unbounded contexts）之间存在 **30-45% 的 PPL 差距**，并将其归因于 Context Compression 造成的真实信息丢失。
   - 一名成员询问是否存在任何手段可以降低压缩率以减轻影响。
- **KV Cache 大小引发讨论**：一篇论文引用了 **160 GB** 的 KV Cache 大小，但一名成员指出，由于使用了 **Flash Attention** 和类似技术，这一数据并不准确。
   - 该成员建议关注 [flash-linear-attention](https://x.com/twitter/status/2024892671563891130) 以了解现代实验性的 Attention 实现。
- **Identity Leakage 受到质疑**：一名成员询问 Identity Leakage 是如何验证的，并提到他们尚未阅读该论文。
   - 该询问是针对[此链接](https://x.com/twitter/status/2024892671563891130)提出的。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1474145367334588490)** (5 messages): 

> `ARES Tooling Framework, Agent Activations Research` 


- **Martian 发布 ARES 工具框架**：Martian 推出了 **ARES**，这是一个旨在暴露 Agentic 设置中 **LLM Agent 激活值**及其轨迹的工具框架，旨在帮助研究人员理解 Agent 如何解决长程任务（long horizon tasks），[代码仓库在此](https://github.com/withmartian/ares)。
   - 此处提供了一个教程，演示如何使用 **ARES** 诊断并纠正简单 Agent 中的失效模式（通过探测 Probing 和激活引导 Activation Steering）：[教程链接](https://github.com/withmartian/ares/blob/main/examples/20q_case_study/ares_mi_20q_tutorial.ipynb)。
- **Martian 的 ARES 在 X 上的动态**：Martian 团队还在 Twitter 上发布了描述 ARES 框架的推文串，[详见此处](https://x.com/Narmeen29013644/status/2024553932635394215)，如果你有任何疑问，也可以加入他们的 [Discord 社区](https://discord.gg/mGTbCZAG)。
   - 最初的发布推文已 [发布在 fxtwitter 上](https://fxtwitter.com/i/status/2024537378535211368)，欢迎转发。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1474151886079918265)** (7 messages): 

> `ChatJimmy, FXTwitter Links, Endomorphosis Datasets, Taalas Ubiquitous AI` 


- **Chollet 的推文在 Discord 引起共鸣**：成员们分享了 [François Chollet 的推文链接](https://fxtwitter.com/fchollet/status/2024519439140737442)，该推文最初发布在 fxtwitter.com。
   - 除了分享 URL 之外，几乎没有进一步的讨论或反应。
- **Endomorphosis 规则清单出现**：一名成员在 GitHub 上分享了 **Endomorphosis 项目的推理规则清单（Inference Rules Inventory）**，具体为这个 [IPFS 数据集 Python 逻辑](https://github.com/endomorphosis/ipfs_datasets_py/blob/main/ipfs_datasets_py/logic/INFERENCE_RULES_INVENTORY.md)。
   - 这似乎是一个数据集项目的规则清单，但频道内未对其目的或功能进行详细说明。
- **ChatJimmy 以极高的 Token 速度自诩**：多名成员关注了 [ChatJimmy.ai](https://chatjimmy.ai/)，强调其宣称的处理速度达到了 **每秒 15k tokens**。
   - 成员们反应热烈，其中一人惊叹道：“这太疯狂了，哇”。
- **Taalas 描绘通往普及化 AI 之路**：一名成员分享了一篇名为《通往普及化 AI 之路》（The Path to Ubiquitous AI）的 [Taalas 文章](https://taalas.com/the-path-to-ubiquitous-ai/)。
   - 该文章可能讨论了 AI 的未来和普及，但未附带评论。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1474164741164236981)** (4 messages): 

> `ARC AGI Fine Tuning, Synthetic data for ARC-AGI` 


- **ARC AGI 正在接受微调**：成员们讨论了目前所有人都在明目张胆地针对 **ARC AGI** 进行 Fine-tuning，参考了 [X 上的一个帖子](https://x.com/i/status/2024556314785894422)。
- **合成数据是 ARC-AGI 的关键**：讨论表明，为 **ARC-AGI** 生成更多合成数据（Synthetic Data）并进行训练的尝试指向了一个结论：这是实现 AGI 的关键。

### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1474190027100520479)** (2 messages): 

> `Tree of Thought, Skill Issues, Coding Assistance` 


- **Tree of Thought 引起用户兴趣**：一名成员表示有兴趣尝试 **Tree of Thought**，但提到由于技术水平有限（skill issues），无法自行编写代码。
   - 他们链接了一篇与该主题相关的 [推文](https://x.com/lakshyaaagrawal/status/2024568680324153800?s=46)。
- **请求 Tree of Thought 编码协助**：该用户明确表示由于技术水平问题 *无法自行编码实现*。
   - [此处](https://x.com/lakshyaaagrawal/status/2024568680324153800?s=46) 链接的推文展示了一个 Tree of Thought 的实现。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1474201179675431056)** (5 messages): 

> `DSPy with Claude, Office Hour Feedback, Reasoning Models with RLM, Qwen3-4B-thinking Issues` 


- **Claude 与 DSPy 技能结合**：一名成员询问是否可以将普通的 Agent（如 **Claude**）与 **DSPy** 结合使用，特别是 DSPy 能否作为与 Claude skill 相关联的脚本。
- **Office Hour 热况**：约有 **40 人** 参加了 Office Hour，涵盖了大约 **10 个用例**，参会者提出了问题并给予了反馈。
- **RLM + Reasoning Model = 成功的秘诀？**：Reasoning Models 与 **RLM** 配合良好，但有报告称在使用 **Qwen3-4B-thinking** 时，sub_lm 调用返回的推理过程会被截断。
   - 一名用户建议，将 sub_lm 适配为使用 signatures 可能会解决此问题。
- **Qwen3-4B-Thinking 循环问题**：一名成员注意到，在其环境设置下（**llama cpp 配合 jinja 以及带有 reasoning parser 的 vllm**），测试 **Qwen3-4B-thinking** 时，sub_lm 调用似乎将推理过程作为答案返回，且该答案被截断了。
   - 这种 **截断** 问题导致 Agent 进入了死循环。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1474334709696958527)** (2 messages): 

> `PR Review Times, Modular PR #5979` 


- **Modular PR 审核预计时间**：一名成员询问了前一天提交的 PR 的审核时间。
   - 另一名成员回复称 [PR #5979](https://github.com/modular/modular/pull/5979) 已分配给审核人员，可能会在当天晚些时候进行审核。
- **PR 提交等待审查**：昨天提交的一项最近的 Pull Request (PR) 正在寻求审核和反馈。
   - GitHub 上 Modular 仓库的 [PR #5979](https://github.com/modular/modular/pull/5979) 已分配给 <@325746765448085504>，计划进行检查，可能在今天晚些时候。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1474434909664841810)** (3 messages): 

> `torch-max-backend performance, MAX backend on Silicon Mac` 


- **新解释器提升 Torch-MAX-Backend 速度**：一名成员报告称，**torch-max-backend** 中的新解释器显著提高了单元测试速度，float32 的测试时间从 **1.54s** 降至 **0.34s**，bfloat16 从 **1.34s** 降至 **0.24s**。
   - 新解释器避免了为每个新的 shape/dtype 进行重新编译，此前这在每次测试中最高耗时 **3 分钟**。
- **Silicon Mac 上的 MAX 后端状态**：一名成员询问了在 **Silicon Mac** 上测试 **MAX 后端** 的情况，提到他们在演讲中将 torch-max-backend 作为探索 MAX 的中间层。
   - 原作者尚未在 Mac 上进行测试，但预计可以运行，因为其在后台调用了 **MAX**。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1474277415348998328)** (2 messages): 

> `AMD assembly infra, Speed Bounties, Portable Solutions` 


- **AMD 汇编基础设施仍是 George 的关注焦点**：George 专注于 **底层编译器工作**，以便 tinygrad 能为 **AMD GPU** 生成高质量的代码。
- **tinygrad 提供性能加速悬赏**：对于可衡量的 **性能提升** 提供 **悬赏 (Bounties)**，包括用于验证提升的工具。
- **tinygrad 优先考虑便携式方案**：George 专注于 **tinygrad 核心** 的改进，从而使所有后端受益，避免使用一次性的自定义 Kernel。


  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1474257365904789595)** (1 条消息): 

> `Tinygrad, George Hotz, AI-HPC` 


- **Tinygrad 的主要贡献者目标是受雇于 Hotz**：一位成员表示打算成为 **Tinygrad** 的主要贡献者并被 **George Hotz** 雇用。
   - 他们已经开始学习 tinygrad，并感谢另一位成员的支持，同时分享了 [AI-HPC GitHub](https://github.com/ai-hpc) 的链接。
- **新手向专家学习 Tinygrad**：一位用户正在深入研究 **Tinygrad**，表达了希望成为核心贡献者的抱负。
   - 他们对另一位成员的指导表示感谢，同时分享了 [AI-HPC GitHub](https://github.com/ai-hpc) 链接作为学习资源。


  

---


### **MCP Contributors (Official) ▷ #[mcp-dev-summit](https://discord.com/channels/1358869848138059966/1413517834805313556/)** (1 条消息): 

aaronpk: 日程表已发布！🎉  https://mcpdevsummitna26.sched.com/
  

---


---


---


---
