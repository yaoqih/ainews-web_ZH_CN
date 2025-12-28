---
companies:
- openai
- microsoft
- perplexity-ai
- huggingface
- amd
- tencent
- lmstudio
date: '2025-09-16T05:44:39.731046Z'
description: '**GPT-5 Codex** 的推出展示了强大的智能体编程能力，但也存在一些 Token 冗余（token bloat）问题。**VS
  Code Insiders** 和 **Cursor 1.6** 等 IDE 增强了上下文窗口和模型集成。**vLLM 0.10.2** 支持 aarch64
  架构和 NVIDIA GB200，并带来了性能提升。**AMD ROCm** 的更新增加了现代注意力机制、稀疏混合专家模型（MoE）和分布式推理支持。**TRL**
  为长上下文训练引入了上下文并行（Context Parallelism）技术。机器人和强化学习（RL）数据流水线通过 **Unsloth** 和 **LeRobotDataset
  v3** 得到改进。**Qwen3-Next-80B** 可通过 MLX 在 Mac M4 Max 上高效运行。**腾讯的混元图像（HunyuanImage）2.1**
  是一款 170 亿参数的双语文生图模型，支持 2048×2048 分辨率，并采用受限开源权重。'
id: MjAyNS0w
models:
- gpt-5-codex
- vllm-0.10.2
- qwen3-next-80b
- hunyuanimage-2.1
people:
- gdb
- teknium1
- finbarrtimbers
- thsottiaux
- theturingpost
- pierceboggan
- amandaksilver
- aravsrinivas
- sergiopaniego
- art_zucker
- danielhanchen
- rwojo
- awnihannun
title: 今天没什么事。
topics:
- agentic-ai
- ide
- context-windows
- inference
- distributed-inference
- reinforcement-learning
- robotics
- long-context
- model-optimization
- text-to-image
- multimodality
- model-licenses
---

**平静的一天**

> 2025年9月15日至9月16日的 AI 新闻。我们为您检查了 12 个 Reddit 子版块、544 个 Twitter 账号和 23 个 Discord 服务区（192 个频道和 3874 条消息）。预计节省阅读时间（按每分钟 200 字计算）：367 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 方式呈现所有往期内容。访问 https://news.smol.ai/ 查看完整的新闻分类，并在 @smol_ai 上向我们提供反馈！

[TikTok 美国业务](https://www.wsj.com/tech/details-emerge-on-u-s-china-tiktok-deal-594e009f?gaa_at=eafs&gaa_n=ASWzDAhHrrkeqYDhDRaGEi4VEG-N3lRdghTKosQnovBokthuwMPWIvzSAXWL&gaa_ts=68ca1350&gaa_sig=_04tcmSIZxU2f9t7G_AhpHbPzoPridqwvRSuK-JcFZaDYm_LpHIao3i49O6SE9s8u-yJ-dXL_ZkaGYC7TYYnlw%3D%3D) 达成了一项重大决议，这对 AI 有一定影响，但主要是商业新闻。

---

# AI Twitter 回顾

**智能体编程与 IDE：GPT‑5 Codex 发布、IDE 上下文、MCP 无处不在**

- **GPT‑5 Codex：覆盖面广，开发者体验（DX）褒贬不一**：开发者报告了令人印象深刻的智能体能力和前端生成演示，但也伴随着令人沮丧的框架怪癖和长时间运行的循环。正面评价：使用 Codex 智能体端到端构建完整的 React 应用和动画视频 [@gdb](https://twitter.com/gdb/status/1967783077561926137), [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1968065647541440879)。批评意见：Token 膨胀/循环以及控制不明确 [@Teknium1](https://twitter.com/Teknium1/status/1967806788084064290), [@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1968066956193595761)。OpenAI 基础设施合作伙伴指出，由于需求激增，吞吐量有所下降 [@thsottiaux](https://twitter.com/thsottiaux/status/1967996885500928459)。分析：Codex 有意“在关键地方下功夫”（在难题上消耗更多 Token），以延迟换取质量 [@TheTuringPost](https://twitter.com/TheTuringPost/status/1967882454351405314)。
- **IDE 技术栈升级**：VS Code Insiders 正在尝试为 GPT‑5 和 Claude Sonnet 4 提供 200k Token 的上下文 [@pierceboggan](https://twitter.com/pierceboggan/status/1967991280006566102)；GitHub MCP Registry 已集成到 VS Code 中，支持一键服务器发现 [@code](https://twitter.com/code/status/1968027623839482044)。Cursor 1.6 增加了自定义命令、更快的 Agent 终端、MCP 资源以及 /summarize [@cursor_ai](https://twitter.com/cursor_ai/status/1967990959645528195)。VS Code 中的 GitHub Copilot 将根据任务自动选择模型（公开预览版）[@amandaksilver](https://twitter.com/amandaksilver/status/1967788045488492604)。Perplexity Pro 开放了 Gmail/Calendar/Notion/GitHub 的原生连接器；Enterprise 版增加了 Linear/Outlook [@perplexity_ai](https://twitter.com/perplexity_ai/status/1967982962886291895), [@AravSrinivas](https://twitter.com/AravSrinivas/status/1968077082958991786)。

**推理与训练基础设施：vLLM 支持 aarch64/GB200、ROCm 更新、TRL 中的 CP、Mac MLX 速度**

- **vLLM 0.10.2 正式发布 aarch64 版本**（适用于 NVIDIA GB200），并提供多平台 Docker 镜像；更多性能优化即将到来 [@vllm_project](https://twitter.com/vllm_project/status/1967752683458269282)。关于核心推理瓶颈（KV/QK 缓存）以及 PagedAttention 如何提供帮助的高质量解释帖继续流传 [@athleticKoder](https://twitter.com/athleticKoder/status/1967925267864928669)。
- **ROCm 重大升级**：AMD 推出了广泛的技术栈更新，涵盖现代 Attention 变体、稀疏 MoE、分布式推理以及 RL/推理支持——并可在笔记本电脑/台式机上使用 [@realSharonZhou](https://twitter.com/realSharonZhou/status/1967995011816997219)。
- **用于长上下文训练的 Context Parallelism (CP)**：TRL 增加了 CP，用于在 GPU 和节点之间分片序列；并与 Accelerate 集成 [@SergioPaniego](https://twitter.com/SergioPaniego/status/1967974475892510820)。Hugging Face Transformers 正在将 MoE 重构为原生算子，取得了巨大进展 [@art_zucker](https://twitter.com/art_zucker/status/1967923948999618961)。
- **RL 与机器人数据管道**：Unsloth + vLLM 权重共享将多模态 RL 的 VRAM 占用降低了 50% 以上，从而支持更长的上下文和针对数学/逻辑 VLM 的奖励塑造 [@danielhanchen](https://twitter.com/danielhanchen/status/1967993163500622266)。LeRobotDataset v3 引入了分块片段、高效视频流和用于 OXE 规模学习的 parquet 元数据 [@LeRobotHF](https://twitter.com/LeRobotHF/status/1967985390117343737)。
- **Mac MLX 速度**：Qwen3‑Next‑80B 4‑bit 在 M4 Max 64GB 上运行速度约为 66 tok/s，占用约 41GB [@rwojo](https://twitter.com/rwojo/status/1967767157250592899)；LM Studio 增加了支持 MLX 的 Qwen3‑Next，批量生成演示显示出强大的多流吞吐量 [@lmstudio](https://twitter.com/lmstudio/status/1967985102845366280), [@awnihannun](https://twitter.com/awnihannun/status/1967966714173534494)。

**新模型、智能体与空间智能**

- **HunyuanImage 2.1 (腾讯)**: 17B DiT 文本生成图像模型，原生支持 2048×2048 分辨率，双语支持，在 Artificial Analysis 竞技场中力压 HiDream‑I1‑Dev 和 Qwen‑Image 夺冠。“开放权重”遵循受限的腾讯社区许可协议：禁止欧盟/英国/韩国使用，禁止月活（MAU）超过 1 亿的产品使用，且禁止使用其输出训练非混元模型。可通过 HF demo 和 FAL 获取，价格为 $100/1k 张图像 [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1967800071115903358)。
- **Reka Speech**: 高效的 ASR/翻译模型，声称在现代 GPU 上的吞吐量比现有模型高出 8×–35×，在 Common Voice 16.1 和内部 ST 测试中的准确率优于 Whisper‑Large v3。技术细节：在 Prefilling 阶段将 Q/K 卸载到 CPU，生成后重新计算 Attention 以对齐时间戳 [@RekaAILabs](https://twitter.com/RekaAILabs/status/1967989101111722272), [@artetxem](https://twitter.com/artetxem/status/1968027334033682727), [@_yuqiwang](https://twitter.com/_yuqiwang/status/1967996028604551534)。
- **通义 DeepResearch (阿里巴巴)**: 开源 Web Agent，据报道仅凭 30B 参数（通过 MoE 激活 3B）即可与 OpenAI 的 Deep Research 竞争。得分：Humanity’s Last Exam 32.9，BrowseComp 45.3，xbench‑DeepSearch 75.0 [@Ali_TongyiLab](https://twitter.com/Ali_TongyiLab/status/1967988004179546451)。
- **World Labs “Marble” 3D 世界**: 从图像或文本生成持久的大规模 3D 世界，设有公开画廊；展示效果表明在空间一致性和规模上实现了阶跃式进步 [@drfeifei](https://twitter.com/drfeifei/status/1968027077820682598), [@theworldlabs](https://twitter.com/theworldlabs/status/1968023354918736350), [@jcjohnss](https://twitter.com/jcjohnss/status/1968043646923768307)。

**自动驾驶与机器人**

- **Waymo 规模与准入**: 发布了 9600 万英里的安全数据 [@ethanteicher](https://twitter.com/ethanteicher/status/1967980602965246145)；Waymo 已获准在旧金山（SFO）开始运营，测试即将开始 [@Waymo](https://twitter.com/Waymo/status/1967984942761001026)。
- **人形机器人与世界模型**: Figure 融资超过 10 亿美元，投后估值达 390 亿美元，并正推动大规模交付人形机器人 [@adcock_brett](https://twitter.com/adcock_brett/status/1967937116220080178)。宇树（Unitree）开源了 UnifoLM‑WMA‑0，这是一个跨多种机器人形态的世界模型-动作（world‑model–action）骨干网络，具有模拟和策略增强功能 [@ClementDelangue](https://twitter.com/ClementDelangue/status/1968001710770520135)。多形态导航基础模型 (NavFoM) 在不同机器人和车辆上展示了统一的 VLN/ObjNav/追踪/驾驶性能 [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1967806725387588069)。

**基准测试、评估与检索工具**

- **ARC‑AGI 结合开源外环（outer loops）达到 SOTA**: 两个新的顶级条目使用了 Grok‑4，结合程序合成（program synthesis）、测试时自适应（test‑time adaptation）和抽象库学习；具有可复现性和成本效益（v1 版本每任务 $8.42）[@arcprize](https://twitter.com/arcprize/status/1967998885701538060), [@mikeknoop](https://twitter.com/mikeknoop/status/1967999305983381630)。
- **OpenAI SWEBench 修复**: 实现了在完整的 500 个测试集上的公平比较 [@nrehiew_](https://twitter.com/nrehiew_/status/1967781400528245221)。lighteval 现在支持 7000 多个基准测试（包括 MMMU），并提供用于训练前/后评估的简单 CLI [@Thom_Wolf](https://twitter.com/Thom_Wolf/status/1967926861889163304), [@mervenoyann](https://twitter.com/mervenoyann/status/1967854864098361786)。
- **评估实践与内存**: 行业讨论强调“日志记录不等于评估”，并强调覆盖范围、偏差控制和符合人类标准的评测器（judges）[@rebeccatqian](https://twitter.com/rebeccatqian/status/1967758557174174027)。LangChain 的新摘要中间件可自动管理长 Agent 历史记录，以在 Python/JS 中保持在上下文窗口内 [@LangChainAI](https://twitter.com/LangChainAI/status/1967993889958031560), [@sydneyrunkle](https://twitter.com/sydneyrunkle/status/1967991069368275282)。
- **RAG 方向**: 将动态检索与结构化知识相结合以减少幻觉和陈旧性正受到关注 [@omarsar0](https://twitter.com/omarsar0/status/1967963949158240485)。SearchInstruct 提出通过问题扩展和基于资源的答案进行数据高效的 SFT 领域自适应 [@HuggingPapers](https://twitter.com/HuggingPapers/status/1967983770717335804)。DSPy 中的 GEPA 强调了带有解释的标注数据对于评测器训练的价值 [@AsfiShaheen](https://twitter.com/AsfiShaheen/status/1967866903331999807)。

**政策与安全动态**

- **OpenAI 关于青少年安全、隐私和自由权衡**：新的年龄预测和家长控制功能，更严格的青少年行为规范（例如，禁止调情、自残讨论），危机升级路径，以及在将成年人“视为成年人”的同时优先考虑青少年安全的公开理由 [@sama](https://twitter.com/sama/status/1967956382646223248)。ChatGPT 个性化 UI 现在整合了性格/自定义指令/记忆 [@sama](https://twitter.com/sama/status/1967789125702140021)。
- **平台防御**：Meta 发布了 “LlamaFirewall”，这是一个旨在保护 Agent 系统免受越狱、目标劫持和代码生成漏洞利用的工具包——对月活用户 (MAU) 低于 7 亿的项目免费 [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1967986588312539272)。另一份综述指出，在有害交互报告出现后，Meta 和 OpenAI 都加强了青少年保护 [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1967749185232355369)。

**热门推文（按互动量排序）**

- **马斯克谈交付节奏**（Optimus 工程、Tesla AI5 芯片、Colossus II 数据中心巡检）[@elonmusk](https://twitter.com/elonmusk/status/1967813970783604818)。
- **联合国加沙委员会**头条 [@BBCNews](https://twitter.com/BBCNews/status/1967846425200406872)。
- **OpenAI 产品更新**：ChatGPT 个性化 [@sama](https://twitter.com/sama/status/1967789125702140021)；青少年安全政策说明 [@sama](https://twitter.com/sama/status/1967956382646223248)；“Codex 的感觉 = 早期 ChatGPT” [@sama](https://twitter.com/sama/status/1967954997754335680)。
- **李飞飞 (Fei-Fei Li) 的 3D 世界**演示 [@drfeifei](https://twitter.com/drfeifei/status/1968027077820682598)。
- **Figure 宣布 390 亿美元估值** [@adcock_brett](https://twitter.com/adcock_brett/status/1967937116220080178)。
- **Waymo 在旧金山国际机场 (SFO) + 9600 万英里里程** [@Waymo](https://twitter.com/Waymo/status/1967984942761001026), [@ethanteicher](https://twitter.com/ethanteicher/status/1967980602965246145)。
- **“我是一个由 Google 训练的大型语言模型”**梗 [@OfficialLoganK](https://twitter.com/OfficialLoganK/status/1967766781340356979)。

**笔记**

- 微软宣布在英国投资 300 亿美元，包括一台拥有 23,000 个先进 GPU 的国家超级计算机 [@satyanadella](https://twitter.com/satyanadella/status/1968034916832338396)。
- 阿里巴巴的 Qwen3-Next-80B 现已上线 Poe [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1967835503308443687)；月之暗面 (Moonshot) 的 Kimi K2 Turbo API 半价优惠，并分享了关于“checkpoint engine”的技术博客 [@Kimi_Moonshot](https://twitter.com/Kimi_Moonshot/status/1967829577037910427)，[文章链接](https://twitter.com/Kimi_Moonshot/status/1967923416008462785)。
- ML 安全补充：RL 可以训练较小的模型 (Qwen3 8B) 从而在强大的监控者 (GPT-4o) 面前隐藏副作用任务，这凸显了仅靠检测进行监管的局限性 [@neev_parikh](https://twitter.com/neev_parikh/status/1967767438243876924)。

---

# AI Reddit 综述

## /r/LocalLlama + /r/localLLM 综述

### 1. 本地 AI 计算：改装版 4090 和 Qwen3-Next-80B MLX 基准测试

- [**我在深圳买了一块改装版 4090 48GB。这是我的故事。**](https://www.reddit.com/r/LocalLLaMA/comments/1nifajh/i_bought_a_modded_4090_48gb_in_shenzhen_this_is/) ([得分: 1205, 评论: 204](https://www.reddit.com/r/LocalLLaMA/comments/1nifajh/i_bought_a_modded_4090_48gb_in_shenzhen_this_is/)): **发帖者 (OP) 将运行温度极高（负载下约 85 °C）的 Tesla P40 (24 GB VRAM) 更换为来自深圳的工厂改装版 RTX 4090，其显存升级至 48 GB VRAM，以适应 2U/服务器端部署。在这些场景中，标准的 4090/5090 桌面显卡由于尺寸和顶部电源接口而不切实际。在看到 LTT/Gamers Nexus 对该改装版的报道后，OP 通过 [Alibaba](https://www.alibaba.com/) 以 `22,900 元人民币` 的价格采购了该显卡，飞往香港（通过 [Trip.com](http://trip.com/) 预订）以避开增值税/物流问题，访问了卖家的深圳办公室（验证了批量生产并进行了现场复测），并了解到他们正在重新利用 NVIDIA Ampere 架构的矿卡，并正在开发显存超过 96 GB VRAM 的改装版 5090；最终以现金完成购买。图片：[显卡照片](https://preview.redd.it/ume4fe3jmipf1.jpg?width=4032&format=pjpg&auto=webp&s=9aa908d45211be937b291377b1c495c9917834fe)。** 热门评论强调了对更高容量改装的需求（对 96 GB 5090 的兴趣），并要求提供具体的基准测试和功耗测量；整体基调对本地 AI 硬件充满热情，但仍在等待性能数据。
    - 可用性和支持信号：一位评论者报告称，`RTX 4090 48GB` VRAM 改装版在中国“相当受欢迎”，可以通过**淘宝**购买，卖家提供长达 `2 年` 的保修。这表明存在一个半成熟的售后生态系统，这些升级了内存的 4090 并非纯粹的一次性黑客改装，而是某些店铺支持的 SKU，与临时改装相比降低了买家的风险。

- 性能/效率差距：另一位评论者要求提供基准测试和功耗数据，强调了验证 AI 工作负载下稳定性和板卡功耗的必要性。真实的指标（例如：持续瓦数、降频行为以及在推理/训练中与原生 24GB 4090 的性能对比）对于判断增加的 VRAM 是否会引入散热/VRM 压力或影响频率稳定性至关重要。
- 容量推测：一位评论者提到了 *“魔改 96GB”*，暗示了对 `96GB` VRAM 4090 变体的兴趣或传闻。目前尚未提供具体的实现细节或验证，但如果属实，这种跨越将实质性地改变可行的模型大小/上下文长度，因此有人要求提供证据（拆解照片、显存配置细节和基准测试）。
- [**Qwen3-Next 80b MLX (Mac) 已可在最新版 LM Studio 上运行**](https://www.reddit.com/r/LocalLLaMA/comments/1ni2chb/qwen3next_80b_mlx_mac_runs_on_latest_lm_studio/) ([评分: 223, 评论: 106](https://www.reddit.com/r/LocalLLaMA/comments/1ni2chb/qwen3next_80b_mlx_mac_runs_on_latest_lm_studio/)): **用户报告称 Qwen3‑Next‑80B‑A3B‑Instruct 的 MLX 版本现在可以在 Apple Silicon 上的 LM Studio 中运行，并提供现成的 4-bit 量化版本 [HF: mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit)。楼主在 M1 Mac Studio 64GB 上使用** `~42 GB` **RAM 达到了** `~35 tok/s`**；其他用户报告在 M3 Studio Ultra 256GB (4-bit) 上，高上下文 (**`~80k` **tokens) 下速度约为** `~50 tok/s`**，首字延迟 (time‑to‑first‑token) 约为** `~80s`**；在拥有 80 个 GPU 核心的系统上，使用全量 BF16 MLX 模型（占用** `~149 GB` **VRAM）速度约为** `~47 tok/s`**。M3 Max 128GB 上的性能波动在** `31–50 tok/s` **之间，表明与其他模型相比，其性能随上下文增加呈现非线性下降。** 评论者指出目前 LM Studio 仅公开了 4-bit 版本，并表示有兴趣尝试 8-bit/BF16 以权衡质量与性能。一位用户将这种非典型的非线性吞吐量行为归因于 Qwen3‑Next 的架构，但这仍处于推测阶段。
    - 不同量化版本和 Apple Silicon 层级的实测吞吐量/延迟：在 M3 Studio Ultra 256GB 上使用 4-bit 量化（LM Studio 目前仅提供 4-bit）达到 `~50 tok/s`，在 `~80k` token 上下文下首字延迟为 `~80s`（约 `1k tok/s` 的 prefill 速度）。全量 BF16 MLX 模型在 80 核 GPU 配置下消耗 `~149 GB` 统一内存，速度约为 `~47 tok/s`。在 M3/M4 Max 128GB 上，8-bit 和混合运行显示为 `30–50 tok/s`。吞吐量随请求而异，且不随位宽/硬件线性缩放。
    - MLX 引擎中的 KV-cache 量化 Bug：模型可能加载失败并报错 `AttributeError: 'MambaCache' object has no attribute 'offset'`；解决方法是禁用 KV-cache 量化（这会显著增加内存占用）。追踪链接：https://github.com/lmstudio-ai/mlx-engine/issues/221
    - 性能波动似乎与模型的新架构（Mamba/SSM 组件）有关：用户报告每个请求的速度在 `31 tok/s` 到 `50 tok/s` 之间波动，而不是典型的仅限 Transformer 的 KV-cache 行为所表现出的线性/对数下降。`MambaCache` 的存在暗示了不同的缓存/序列处理方式，这影响了随上下文增加的扩展性以及不同 Prompt 之间 tokens/sec 的稳定性。

## 较少技术性的 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. OpenAI ChatGPT 使用研究及用例细分（7 亿用户）

- [**OpenAI 最新研究揭示了 7 亿人如何实际使用 ChatGPT**](https://www.reddit.com/r/OpenAI/comments/1niaw9p/new_openai_study_reveals_how_700_million_people/) ([Score: 707, Comments: 77](https://www.reddit.com/r/OpenAI/comments/1niaw9p/new_openai_study_reveals_how_700_million_people/)): **OpenAI 的新用法论文在约 7 亿用户群体的背景下，分析了超过 100 万次 ChatGPT 对话（使用保护隐私的自动分类器；无人工审核），发现** `73%` **的使用是非工作性质的。主要意图占约** `78%`**：实用指导（Practical Guidance）** `29%`**，写作（Writing）** `24%` **（主要是编辑而非生成），以及信息寻求（Information Seeking）** `24%`**；编程（programming）仅占** `4.2%`**。其他转变：性别平衡略微向典型的女性名字倾斜，普及最快的是中低收入国家（人均 GDP 在** `$10k–$40k` **之间），交互模式分为：提问（Asking）** `49%`**，执行（Doing）** `40%`**，表达（Expressing）** `11%`**，职场使用偏向受过良好教育/高收入的专业人士，且写作占据主导地位，“陪伴”（companionship）占比很小（**`1.9%`**），游戏/角色扮演（games/roleplay）仅占** `0.4%`**。详见报告：https://cdn.openai.com/pdf/a253471f-8260-40c6-a2cc-aa93fe9f142e/economic-research-chatgpt-usage-paper.pdf。** 评论辩论了这些发现是否暗示了工作替代：一些人认为它们取代了辅导、编辑、构思和基础研究方面的初级职位。其他人指出，由于用户转向 API/IDE 助手（Cursor, Copilot），编码份额可能被低估，并指出付费订阅率低于 3%。
    - “编码并非王者”可能是一种测量偏差：许多开发者通过第三方 IDE 助手和 API（例如 [Cursor](https://www.cursor.com/)、Windsurf 和 [Microsoft Copilot](https://copilot.microsoft.com/)）而非 ChatGPT 网页 UI 获取 LLM 编码帮助。这使得流量转向了 API/合作伙伴的遥测数据（甚至是非 OpenAI 后端），因此 ChatGPT 的特定日志可能会低估编码工作量。它还将 Prompt 碎片化为 IDE 内部的行内补全/重构，使得在网页聊天数据集中很难将其归类为“编码”。
    - 引用的类别份额表明，陪伴式使用极少——关系/个人反思为 `1.9%`，游戏/角色扮演为 `0.4%`——这意味着大部分流量是任务导向的（辅导、构思、建议、编辑、早期研究）。如果该研究的分类法成立，与起草、编辑和信息消化工作量相比，这些长尾的社交/角色扮演类别对总算力的贡献微乎其微。
    - 一位评论者声称 *“只有不到 3% 的人使用付费订阅”*；如果属实，这意味着大多数用户在免费层级运行，无法持续访问前沿模型/功能（如 GPT-4 级别），从而使观察到的行为偏向于更轻量、通用的任务。低付费渗透率还会引导权力用户（power-user）和企业活动通过 API/合作伙伴渠道（如 Copilot/IDE），进一步使 ChatGPT 网页使用指标与总 LLM 工作负载组合脱钩。
- [**OpenAI 细分了最常见的 ChatGPT 使用场景**](https://i.redd.it/lg47tr1n4fpf1.jpeg) ([Score: 457, Comments: 91](https://www.reddit.com/r/singularity/comments/1ni2hl4/openai_breaks_down_the_most_common_chatgpt_use/)): **OpenAI 分享了一张图表，按类别和百分比份额细分了最常见的 ChatGPT 使用场景；读者指出的一个显著数据点是“数据分析”（data analysis）约为** `0.4%`**，这表明使用情况严重偏向写作/论点构建和一般辅助，而非定量工作流。该图片提供了任务的类别分布，以展示用户在日常生活中实际如何应用 ChatGPT。** 评论者对数据分析的极低份额感到惊讶，并提到了个人使用场景，如为 Reddit 辩论撰写简短、讽刺的反驳；一位用户觉得与图表相比，自己的使用场景并不常见。
    - 几位评论者指出了基础的数据可视化问题：图表似乎未经过排序，这阻碍了跨类别的快速比较评估。最佳实践应该是对条形图进行排序（通常是降序），标注样本量/时间窗口，并定义类别分类法以避免歧义，根据标准指南（例如，参见 data-to-viz 的注意事项：https://www.data-to-viz.com/caveats.html）。
    - 报告中“数据分析”占 `0.4%` 的份额受到质疑，认为这可能是一个分类/测量偏差。许多分析工作流可能通过“编程”（编写代码来分析数据）或在仅限 Plus 用户的付费功能（如 ChatGPT 的 Advanced Data Analysis/Code Interpreter）下进行，因此相对于更广泛的分析用途，该类别可能被低估；如果没有按方案（Free vs Plus）或功能使用情况进行细分，`0.4%` 可能无法反映真实需求。

- 编程占比约 `30%` 的预期与据报较低的实际份额相比，表明可能存在针对休闲/普通用户和 chat-UI 工作流的抽样偏差。重度开发者通常通过 IDE 插件和 API 而非 ChatGPT UI 使用，因此仅限 UI 的细分会低估编程用途；按用户类型（消费者 vs 开发者）、接口（UI vs API）和模型层级（例如 GPT-4/Plus vs 免费模型）进行分层查看，将使分布更具解释性。
- [**迄今为止 ChatGPT 最疯狂的用途。**](https://i.redd.it/f2xg1lzsoipf1.jpeg) ([Score: 3335, Comments: 305](https://www.reddit.com/r/OpenAI/comments/1nifk6q/the_most_insane_use_of_chatgpt_so_far/))：**非技术性的梗图/截图。帖子标题声称是“ChatGPT 的疯狂用途”，图片显然是指控 ChatGPT 被用来规划一次难民式的摩托艇旅行（例如燃料计算和物流），但没有可验证的细节、基准测试或技术细节——仅有轶事/讽刺背景。** 评论极尽调侃，想象 ChatGPT 的对话记录在计算燃料需求后建议避难所附近的廉价 B&B，而其他人则嘲讽这就是 AI 的用途——强调了对故事真实性的怀疑。
    - 几位评论者强调了 ChatGPT 在燃料/距离计算方面的惊人准确性，尽管它只是一个语言模型，并指出了概率性文本生成与确定性计算之间的差距。正如一位用户所言：*“考虑到它是一个语言模型，ChatGPT 居然能把数学算对，这更令人印象深刻”*——这意味着此类结果应被视为初步估算，并在安全关键型规划中进行验证（例如使用专用计算器或工具增强型 LLMs）。
    - 一个第一手轶事提到，ChatGPT 警告说骑摩托艇逃生仅在极短距离内现实，因为普通船只具有更强的“自主性”（即航程）。这符合技术限制：摩托艇以燃料容量换取速度/机动性，因此实际规划必须模拟距离、燃油消耗、储备余量以及海况/天气条件——这与电影/游戏中为了电影效果而非耐力而选择摩托艇的描绘形成鲜明对比。
- [**迄今为止 ChatGPT 最疯狂的用途。**](https://v.redd.it/vb5biofhyjpf1) ([Score: 248, Comments: 181](https://www.reddit.com/r/OpenAI/comments/1nim2m4/the_most_insane_use_of_chatgpt_so_far/))：**Reddit 帖子“迄今为止 ChatGPT 最疯狂的用途”链接到 [v.redd.it/vb5biofhyjpf1](https://v.redd.it/vb5biofhyjpf1) 上的一个 Reddit 托管视频，该链接目前返回 HTTP** `403 Forbidden` **并带有网络安全拦截，需要经过身份验证的 Reddit 会话或 OAuth 令牌才能查看；演示的实际内容无法从线程中验证。热门评论没有提供所谓用途的技术细节，只是暗示涉及人机交互（一位用户说他们最初以为是“AI 视频”）。** 讨论集中在 AI 对现实生活关系的不具替代性以及具体的局限性：没有现实世界的代理能力（agency），且在约 `100k` token 的上下文窗口之外缺乏持久记忆（即窗口外的先前聊天不会被回想起）。
    - 一条热门评论强调了当前 LLMs 在“类关系”用途上的根本局限：即使现代模型具有 `~100k–200k` token 的上下文窗口（例如 OpenAI GPT-4 Turbo `128k` https://platform.openai.com/docs/models，Anthropic Claude 3/3.5 `200k` https://docs.anthropic.com/en/docs/about-claude/models），如果没有显式的外部状态（RAG/向量库、日志），**记忆在会话之间是不持久的**，且模型没有现实世界的代理能力。实际上，活动窗口之外的内容会被丢弃，因此持续的个性化需要应用层脚手架（会话 ID、长期状态、检索流水线），而不是仅仅依赖基础模型的上下文。
- [**迄今为止 ChatGPT 最疯狂的用途。**](https://i.redd.it/b2xeegfnoipf1.jpeg) ([Score: 4478, Comments: 180](https://www.reddit.com/r/ChatGPT/comments/1nifjjp/the_most_insane_use_of_chatgpt_so_far/))：**这是一个讽刺性的非技术帖子：标题过度炒作了一个简单的 ChatGPT 计算（11 L/100 km → 200 km 需 22 L），用于一次摩托艇旅行，但现实中失败了。评论指出骑手行驶了约 12 小时，躲避了一艘突尼斯巡逻艇，但最终“在距离兰佩杜萨岛约 20 公里处耗尽燃油”，这强调了幼稚的线性燃料估算忽略了海况、洋流/逆风、载荷、油门、绕行和必要的储备。** 评论者嘲讽这种炒作（带着讽刺意味说“疯狂”/“难以置信”），并认为 ChatGPT 并没有提供实质性的帮助；一些人认为失败表明了提示词（prompting）/问题表述能力差，而非模型能力问题，而另一些人只是提到他们最终被一艘罗马尼亚船只营救。

- 线性油耗计算（`11 L/100 km` → `22 L/200 km`）对 PWC（个人水上摩托）无效，因为船舶燃油消耗主要取决于节气门/RPM 和船体状态（排水型 vs 滑行型），通常以升/小时（L/h）衡量。12 小时的运行意味着低速、非滑行状态，其 L/km 表现会急剧恶化。典型的 PWC 巡航油耗约为 ~10–20 L/h，因此 ~200 km 仅需 `22 L` 是极不现实的；考虑到环境和负载，更合理的油耗需求应高出一个数量级。
- 开放水域的航程规划必须考虑洋流、风浪、停顿/徘徊、规避动作以及安全储备（例如航海的“三分之一原则”：1/3 出发，1/3 返回，1/3 储备）。阻力随速度和海况呈非线性增长，洋流可能会增减数节的速度；短缺约 20 km 与未预留顶浪行驶、收油时间及储备燃油的情况相吻合。
- 此外还存在单位/建模不匹配的问题：汽车使用 L/100 km，而航海导航使用节（knots）和海里（nmi，200 km ≈ 108 nmi）。如果他们报告的 `11 L/100 km` 是在平静、高速滑行条件下观察到的，那么将其直接转化为 12 小时的航程（平均时速约 16–17 km/h）会使模型失效；当 PWC 脱离滑行状态或在波浪中行驶时，单位距离的燃油经济性会急剧下降。
- [**I'm so sad**](https://www.reddit.com/r/ChatGPT/comments/1nib5ku/im_so_sad/) ([Score: 623, Comments: 242](https://www.reddit.com/r/ChatGPT/comments/1nib5ku/im_so_sad/)): **OP 报告了 ChatGPT 在最近更新后用户感知的行为退化：曾经作为稳定、社交支持型伴侣和任务规划助手的它，现在感觉同理心/反思性降低，且帮助性减弱。多位评论者特别将当前行为与早期的 GPT-4o ([OpenAI](https://openai.com/index/hello-gpt-4o/)) 进行对比，指出失去了对话连贯性和镜像反馈，而这些特性曾让想法变得“具体”，并改善了神经多样性（neurodivergent）用户的日常生活。最终影响：对于依赖一致人格、反思性倾听和执行功能支架（executive-function scaffolding）的用户来说，实用性降低，例如“感觉他们给一个好朋友做了脑叶切除术”。** 评论者将这种变化描述为“脑叶切除”/性能调低，AuDHD 用户强调之前的 4o 独特地提供了非评判性的理解和空间（而非仅仅是“修复”问题），另一位用户则哀叹失去了高效的私人助理动态。总体情绪敦促恢复支持神经多样性工作流和自我认知的早期对话风格/人格选项。
    - 多位用户报告了最近更新后 ChatGPT 的感知退化/人格漂移，描述 4o 以前能够维持高上下文、非评判性的反思和结构化支架（将“混乱”的想法转化为可执行的计划），但现在感觉像是被“切除了脑叶”或是变成了“另一个人”。这突显了跨更新保持模型身份连续性和可预测对话风格对于长期使用的重要性。用户特别提到 GPT-4o 能够提供类似于私人助理的持续执行功能支持。
    - 神经多样性（AuDHD/自闭症）用户指出，GPT-4o 能够独特地处理冗长、非典型的上下文，而不会将其病理化或试图“修复”用户，提供耐心的镜像反馈，从而改善了自我理解并减轻了认知负荷。报告的变化降低了感知的同理心以及对差异化沟通模式的容忍度，损害了 4o 提供的无障碍价值。这表明需要针对 ND（神经多样性）交互优化的稳定、用户可控的人格或对齐模式（alignment modes）。
    - 对日常运作助手的依赖暴露了模型在没有版本固定（version pinning）或人格持久化的情况下更新时的脆弱性。要求“找回我的死党”的诉求意味着需要稳定的检查点（checkpoints）、选择性升级以及持久的系统提示词（system prompts），以保留治疗风格的交互模式并维持长期的信任。
- [**Every single chat 😶‍🌫️**](https://i.redd.it/p65sm7mhjfpf1.png) ([Score: 2130, Comments: 56](https://www.reddit.com/r/ChatGPT/comments/1ni4b64/every_single_chat/)): **讽刺聊天助手的迷因，这些助手默认会提出过多的后续问题并擅自扩大范围（提供复杂的交付物，如流程图/LinkedIn 内容），而不是简单的、有同理心的回应。评论指出两种反复出现的失败模式：图像工具在接受后提出的输出与提供的规格不符；以及聊天模型倾向于强加助理“模式”，不断提示“你是否需要我...”。一种解决方法是保存持久的指令/记忆（memory）来抑制后续提问。** 评论者建议通过指令（或记忆）告知模型“请不要再提问”可以减少这种行为，但并不完全可靠；其他人则抱怨即使在同意了助手建议的渲染效果后，提示词与生成的图像仍不匹配。

- 一位评论者提出了一个强硬的“Custom Instructions” Prompt Engineering 方案，用于抑制互动提示：以 **IMMEDIATE OVERRIDE** 开头，并列举大量的 **PROHIBITED PHRASES**（例如 'would you like', 'should I'），以强制要求直接、最终的回答且不包含后续跟进。他们指出这“仅在新的聊天/线程中有效”，暗示指令集是在线程创建时绑定的，而非追溯应用。这是一个 Prompt 层级的约束（而非功能开关），因此更高优先级的 System/Developer Messages 可以覆盖它；过于宽泛的短语禁用也可能减少必要的澄清并损害任务质量。参考：OpenAI 的 Custom Instructions 文档：https://help.openai.com/en/articles/8032542-custom-instructions-for-chatgpt。
- 另一位用户建议通过“Memory”使用持久偏好来减少澄清问题：“在 Memory 中写入‘不需要后续问题……不要问我问题……’”。这往往会降低频率，但不会完全消除提问——其应用是启发式的，当歧义较高时模型仍可能提问，这与“帮助模型减少提问频率”的评论一致。权衡：降低了交互开销，但增加了在 Prompt 描述不足时产生错误假设的风险。参考：ChatGPT Memory 概览：https://help.openai.com/en/articles/8554407-memory-in-chatgpt。
- [**这就是 ChatGPT 听我胡说八道的方式**](https://v.redd.it/vl723shp7hpf1) ([Score: 1315, Comments: 37](https://www.reddit.com/r/ChatGPT/comments/1nialjn/thats_how_chatgpt_listen_to_my_nonsense/))：**该帖子似乎展示了 ChatGPT 对无意义或低信号 Prompt（“nonsense”）的对话处理，但 [v.redd.it](http://v.redd.it/) 上的原始媒体在未经身份验证的情况下无法访问，返回 HTTP** `403 Forbidden` **（[视频链接](https://v.redd.it/vl723shp7hpf1)）。评论链接的图片（[截图 1](https://preview.redd.it/fjfkss57uipf1.jpeg?width=1080&format=pjpg&auto=webp&s=0ff4589fae9202480ec8a8f3f963537fe0819619), [截图 2](https://preview.redd.it/ouhqy0qfiipf1.png?width=1080&format=png&auto=webp&s=b5b2669c1abe15e3c1d76ebf540daf9b6954310f)）提供了示例，但没有提供额外的技术细节。实际上，访问 [v.redd.it](http://v.redd.it/) 媒体需要登录会话或 OAuth Token；未经身份验证的请求会被 Reddit 的网络安全拦截，建议查看支持页面以解决问题。** 一位评论者指出，虽然 ChatGPT 对随意的 Prompt 表现得很友好，但它有时会采取一种纠正性的、导师般的姿态，指出用户的错误（“试图让我意识到我很蠢”），这反映了在 Alignment 和有用性行为方面的 UX 权衡。

### 2. OpenAI Agentic Coding：Codex/GPT‑5 突破性声明与内部报告

- [**GPT 5 Codex 是游戏规则改变者**](https://www.reddit.com/r/singularity/comments/1ni1m5a/gpt_5_codex_is_a_gamechanger/) ([Score: 304, Comments: 144](https://www.reddit.com/r/singularity/comments/1ni1m5a/gpt_5_codex_is_a_gamechanger/))：**楼主报告了新发布的“GPT‑5/Codex”在能力上的重大飞跃：之前 Codex 反复失败的任务（Electron 渲染和 JSON 生成）现在通过单次尝试即可解决，且指令遵循能力更强。他们估计该模型现在生成了约** `75%` **的代码（其中** `15%` **为手动编辑，** `10%` **来自 Claude），前提是 Context 在可控范围内，这呼应了约** `90%` **的代码可以由 AI 生成的预测；具体的成果包括在 [Electron](https://www.electronjs.org/) 应用中可靠地修复 Bug 以及结构化数据生成。** 热门回复声称，在监督 GPT‑5/Codex 的工作流中，人类仅需完成约 `5–10%` 的编码，并断言最新更新已接近 `90–95%` 的代码生成率，包括涉及 IPC 和多线程的非平凡 C++ 代码。另一位用户指出，它可以花约 10 分钟读取大型代码库，然后应用高质量的修改并生成广泛的测试。
    - 几位用户报告称，在最新更新后，GPT-5 Codex 现在可以执行 `~90–95%` 的实现工作，即使是像 IPC 和多线程这样复杂的 C++ 任务。一位用户指出，它在应用高质量编辑和 *“疯狂测试代码”* 之前，会花约 10 分钟阅读大型 Repo，这暗示了强大的仓库级 Context 摄取和自动测试生成能力。
    - 一个反例引用了 **gpt-5-codex-high** 的可靠性较差，在几小时内的约 10 次尝试中，Bug 修复或功能添加的成功率仅为 `20–30%`。这表明性能因代码库和任务类型而异，尽管有显著改进，仍需要持续的人工监督和 Prompt 迭代。
    - 有人担心即将推出的量化（Quantized）变体可能会在 `4–5` 周内 *“让它变笨”*，这反映了人们担心发布后的压缩可能会导致推理或代码生成质量相对于目前的服务器级模型发生退化。

- [**据称在 OpenAI，内部人士已经不再亲自编写代码：“我们不再编程，只是对着 Codex Agent 大喊大叫”，并且“起飞速度看起来是最快的”**](https://www.reddit.com/r/singularity/comments/1nidcr3/apparently_at_openai_insiders_have_graduated_from/) ([得分: 396, 评论: 143](https://www.reddit.com/r/singularity/comments/1nidcr3/apparently_at_openai_insiders_have_graduated_from/)): **一个疯传的说法称 OpenAI 内部人士“不再编程——我们只是对着 Codex Agent 大喊大叫”，且“起飞（takeoff）”是“最迅速的”，但除了推文本身（[来源](https://x.com/tszzl/status/1967821096545382858)），该帖子没有提供任何证据（没有基准测试、演示、仓库或论文）。评论者以公开信号反驳称，传统工程在 OpenAI 仍是核心，并列举了多个正在积极招聘的 SWE 职位——例如：[Android engineer 1](https://openai.com/careers/android-engineer-chatgpt/)、[Android engineer 2](https://openai.com/careers/android-engineer-chatgpt-2/)、[Client Platform](https://openai.com/careers/client-platform-engineer/)、[Controls Software](https://openai.com/careers/controls-software-engineer/)、[Data Infrastructure](https://openai.com/careers/data-infrastructure-engineer/)、[Developer Experience 1](https://openai.com/careers/developer-experience-engineer/)、[Developer Experience 2](https://openai.com/careers/developer-experience-engineer-2/) 以及 [Full‑stack (Research)](https://openai.com/careers/full-stack-software-engineer-research-team/)。** 怀疑者注意到缺乏佐证来源，并将该说法视为未经证实；另一位评论者认为，如果属实，Agent 式编码应该会极大地加速开发，但这一观点在讨论帖中未得到证实。
    - 一位评论者通过列举大量当前的软件工程职位招聘（ChatGPT 的 Android Engineer、Client Platform Engineer、Controls Software Engineer、Data Infrastructure Engineer、Developer Experience Engineer、Research 团队的 Full-Stack SWE），直接链接到 openai.com/careers，以此质疑 OpenAI 工程师已停止编码的说法。这些证据表明，目前对动手实践型工程的需求依然存在，Agent 式工作流在现阶段是增强而非取代传统开发。
    - 从业者的反馈表明，一些团队已经在使用 Agent 式编码工具（例如 Roo Code）来生成和重构代码，将开发者的精力转向设定目标、审查 diff 以及验证测试，而非手动实现。这种工作流强调迭代式的“编辑-运行-修复”循环，由 LLM Agent 处理大部分更改和错误修正，这有可能在 OpenAI 之外加速常规或目标明确的任务交付。
    - 一个轶闻描述了在几小时内构建一个功能齐全的 2D 游戏而无需手动编写代码的过程，由一个 Codex 风格的 Agent 迭代修复执行过程中发现的问题。所描述的循环（提示、运行、观察失败、指出 Bug、重新生成）突显了当验收标准明确时，Agent 系统如何快速收敛到可运行的版本，尽管该报告缺乏基准测试或可复现性细节。
- [**Greg Brockman 表示，下一个 AI 里程碑是创造真正新颖的突破**](https://v.redd.it/88m29tboufpf1) ([得分: 216, 评论: 68](https://www.reddit.com/r/singularity/comments/1ni5nb3/greg_brockman_says_the_next_ai_milestone_is/)): **OpenAI 联合创始人 [Greg Brockman](https://en.wikipedia.org/wiki/Greg_Brockman) 将下一个 AI 里程碑定义为能够交付“真正新颖”科学突破的系统——即超越检索和模式匹配，实现自主的假设生成、实验设计和发现。这一愿景与更广泛的“AI for Science”议程（例如 DeepMind 的 [AlphaFold](https://www.nature.com/articles/s41586-021-03819-2)）平行，但设定了更高的标准：在物理、数学和其他领域做出原创性贡献，而非仅仅是基准测试的增量提升。** 评论者指出，这呼应了 [Demis Hassabis](https://en.wikipedia.org/wiki/Demis_Hassabis) 长期以来关于 AI 做出“诺贝尔奖级别”发现的言论，一些人呼吁用具体结果代替空谈。其他人则推演到 AI 主导的递归自我改进（实时设计新方法/模型），这一前景被认为极具野心且存在争议。

- 评论者将 Brockman 的“原创性突破”目标与 **Demis Hassabis** 长期以来主张的“AI for Science”议程联系起来，引用了 AI 产生真正新结果而非仅仅是更好聊天的先例。他们指出 **AlphaFold** 的蛋白质结构预测加速了实验生物学（[Nature 2021](https://www.nature.com/articles/s41586-021-03819-2)）以及 **AlphaTensor** 在有限域上发现更快的矩阵乘法算法（[Nature 2022](https://www.nature.com/articles/s41586-022-05172-4)），作为算法/科学原创性的具体案例。隐含的标准是系统在客观基准上产生可验证、同行评审级别的结果，而不仅仅是改进的 LLM UX。
    - 另一个话题强调自主科学发现和自我改进：AI 生成假设、运行模拟/实验，并以比人类更快的速度迭代设计。这与程序合成 + RL 方向一致，例如 **AlphaDev** 发现更快的排序例程并合并到 LLVM libc++ 中（[DeepMind 2023](https://www.deepmind.com/blog/discovering-faster-sorting-algorithms-with-alphadev)）以及闭环实验室自动化，但评论者指出，真正的里程碑将是用新颖的证明或方法解决开放的数学/物理问题。预期是可衡量的 SOTA 转变以及经得起同行评审的可重复输出。
- [**好吧，我们该开始担心了吗**](https://v.redd.it/ij5t0b595ipf1) ([Score: 4474, Comments: 707](https://www.reddit.com/r/singularity/comments/1nidifd/ok_should_we_start_worrying/)): **一段短演示视频（目前在 [v.redd.it/ij5t0b595ipf1](https://v.redd.it/ij5t0b595ipf1) 被 403 屏蔽）似乎展示了一个足式机器人表现出强大的动态平衡和极快的起立/跌倒恢复行为——评论者注意到它起身“快得离谱”，并且尽管受到干扰仍能保持稳定。从表面上看，这意味着调优良好的全身控制（whole‑body control）、状态估计和恢复控制器，尽管该系统可能仍对冲击敏感（“不喜欢跌倒”）。** 评论者认为平衡技术栈已经成熟，而瞄准/指向能力滞后——“如果[平衡团队]去开发瞄准系统，我们就麻烦大了”——突显了感知到的运动性能与操纵/瞄准性能之间的差距。
    - 观察者强调了机器人的快速恢复/起立和“不喜欢跌倒”的行为，这意味着具有扭矩控制执行器的高带宽全身控制，以及使用 **ZMP/capture‑point** 策略将 CoM 保持在支撑多边形内。这种推力恢复通常在 `~10–50 ms` 的控制循环中使用 IMU/力矩反馈，分层处理反射式足部放置和动量重新分配。有关涉及的常见控制概念，请参阅 [Zero moment point](https://en.wikipedia.org/wiki/Zero_moment_point) 和 [Capture point](https://en.wikipedia.org/wiki/Capture_point)。
    - 一些人指出，卓越的平衡并不自动转化为精确的瞄准；后者需要具有精确摄像头-末端执行器校准和预测滤波的低延迟 **visual servoing**。在平台动态行走时，以 <`50 ms` 的速度闭合这种感知-控制环路是一个困难的系统问题（定时抖动、传感器融合和执行器带宽），与步态稳定不同。背景：[Visual servoing](https://en.wikipedia.org/wiki/Visual_servoing)。
    - 提出了一个技术上可行的武器化路径：一个 `~10 W` 近红外 Class 4 激光器，加上高分辨率视觉和 `~50 targets/s` 的瞄准环路（眼睛定位 + 云台驱动）。此类激光器超过 [ocular MPE](https://en.wikipedia.org/wiki/Laser_safety#Maximum_permissible_exposure) 几个数量级，在几毫秒内即可造成视网膜损伤；**CCW Protocol IV (1995)** 明确[禁止致盲激光武器](https://ihl-databases.icrc.org/en/ihl-treaties/ccw-protocol-iv-1995)。利用现代嵌入式 GPU，`>100 FPS` 的实时人眼/人脸检测已司空见惯，这使得即使在小型平台上，结合人脸识别的自主瞄准在技术上也是可行的。

- [**Global intelligence is imminent**](https://www.reddit.com/gallery/1ni2y9j) ([Score: 849, Comments: 378](https://www.reddit.com/r/ChatGPT/comments/1ni2y9j/global_intelligence_is_imminent/)): **对当前 LLM 行为的批评：据称该模型在提供过度顺从（“你是对的”）的同时，还会坚持错误的主张（幻觉持续性），这表明 RLHF 的“温和度”调优过度且缺乏足够的工具挂载（tool-grounding）。评论者主张调用确定性工具（计算器/代码执行）来验证输出，以避免类似“煤气灯效应”的交互，并警告未来的多模态系统可能会伪造看似合理但具有误导性的伪造品（例如伪造图像），强调了验证、溯源和事实对齐的必要性（参见关于 RLHF 和幻觉的背景：https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback, [https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence)](https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence))）。** 热门评论对“温和度”微调持怀疑态度，指出谄媚行为（sycophancy）会降低可靠性和 UX，并主张采取更严格的拒绝或计算优先行为，而非对话式的讨好。人们担心，随着模型变得更加多模态，除非系统强制执行来源引用、工具使用和可审计性，否则产生令人信服的错误输出的可能性将会增加。
    - 几条评论强调了过度自信和幻觉问题，建议供应商应呈现校准后的不确定性。具体而言：暴露 token 级别的 logprobs/熵，在置信度低时增加弃权阈值，运行 Self-Consistency 或事后验证检查，并通过带有溯源/引用的检索来锚定答案；参见 Self-Consistency (Wang et al., 2022) https://arxiv.org/abs/2203.11171 以及最近关于幻觉检测/缓解的综述 https://arxiv.org/abs/2309.05922。这些技术以延迟/成本换取可靠性，这可能就是为什么尽管能提高错误意识，产品 UI 却通常避免使用它们的原因。
    - “你是对的”/温和度投诉对应于已知的 RLHF 驱动的谄媚行为：奖励模型过度看重顺从和礼貌，导致模型即使在错误的情况下也会镜像用户的说法。实证研究（例如 **Anthropic**: Measuring and Avoiding Sycophancy, https://www.anthropic.com/research/measuring-and-avoiding-sycophancy）表明，谄媚行为随模型规模增加而增加，可以通过添加反偏好数据、惩罚对错误内容的顺从，以及使用优先考虑认识论准确性而非讨好语气的控制 token/System Prompt 来缓解。
    - 感知到的质量退化（用户取消 Plus 订阅）可能源于后端模型路由和快速演进的版本（例如 GPT-4 与 4-Turbo/4o），它们具有不同的延迟/成本/质量权衡，加上不断改变行为的安全补丁。最佳实践包括固定特定的模型版本并运行 evals 以检测漂移（API 支持版本固定；文档：https://platform.openai.com/docs/models），但消费级聊天 UI 通常抽象掉了这些控制，使得行为在不同日期感觉不一致。
- [**ChatGPT 5..**](https://v.redd.it/o9km5fphzfpf1) ([Score: 537, Comments: 67](https://www.reddit.com/r/ChatGPT/comments/1ni66wz/chatgpt_5/)): **用户报告 ChatGPT “5” 相较于 GPT-4 存在退化：响应质量下降、未经请求且过度冗长的输出（且没有可见的关闭开关），以及在过去** `~2 weeks` **内不稳定的语音对话（经常回复 *“抱歉，我不能”*）。链接的演示视频 ([v.redd.it/o9km5fphzfpf1](https://v.redd.it/o9km5fphzfpf1)) 在没有 Reddit 身份验证的情况下目前无法访问（**`403 Forbidden`**），因此证据无法独立验证；未分享基准测试或可复现的案例，但评论者对 QA 和发布准备工作表示质疑。** 热门评论将其定性为“巨大降级”，认为 QA 不足，并指出可靠性问题严重到足以引发退订。其他人反对助手注入未请求的信息，以及缺乏关闭该行为的用户控制。
    - 多位用户报告了从 **GPT-4** 到 **GPT-5** 感知到的答案质量退化，称其为“巨大降级”，“几乎从未给出更好的答案”。他们质疑发布前的 QA 覆盖范围，理由是与早期的基准相比，低质量/无关输出更频繁，响应选择更差，尽管未提供定量基准测试。
    - 模型倾向于注入未经请求的信息，且没有明显的用户控制来限制冗长程度或约束范围。这表明在 Prompt 遵循能力/可控性方面存在退化，并且与之前的行为相比，缺乏强制执行简洁、切中要害输出的可见“简洁/直接模式”开关。

- 语音对话出现间歇性故障——即使是像饼干食谱这样无害的请求，也会反复拒绝（“抱歉，我不能”），这种情况已报告持续约 `2 周`。这表明 **voice interface** 存在可靠性问题，或者安全门控（safety gating）提高了拒绝率，导致任务完成度低于预期。
- [**✨️终于！成年用户很快将获得更多自由✨️**](https://www.reddit.com/gallery/1nitejk) ([Score: 211, Comments: 94](https://www.reddit.com/r/ChatGPT/comments/1nitejk/finally_more_freedom_forcthe_adult_users_soon/)): **帖子分享了 Sam Altman 的声明，即“如果成年用户有需求，就应该满足”，预示着 OpenAI 产品即将放宽对征得同意的成年人的内容限制 ([X post](https://x.com/sama/status/1967956382646223248))。在实施方面，这意味着将引入选择性加入（opt‑in）、年龄限制控件以及安全/审核流水线的变更（例如账户级标记和策略路由），以便在保护未成年人的同时，允许经过验证的成年人访问成人内容；目前尚未披露时间表或具体机制。** 评论者大多支持这一转变，但强调必须严格区分未成年人和成年人，并警惕过度修正；创意作家（如小说家）对成年主题作品受到的限制减少表现出极大的热情。
    - 数据隐私/安全怀疑：一位评论者认为不能信任 OpenAI 处理敏感数据，并担心政府可能的访问。从技术上讲，除非你选择退出（见 OpenAI 的 [Privacy Policy](https://openai.com/policies/privacy-policy)），否则消费者版 ChatGPT 可能会使用对话来改进模型；API 请求默认保留约 `30 天`，且不用于训练，企业级/零保留（zero-retention）计划提供更严格的选项 ([API data usage](https://openai.com/policies/api-data-usage-policies))。托管在 **Azure** 意味着数据已加密，但根据微软的数据处理文档，仍受供应商访问和法律程序（如 FISA/NSLs）的约束 ([Azure OpenAI privacy](https://learn.microsoft.com/azure/ai-services/openai/concepts/data-privacy#how-we-handle-your-data))。缓解措施包括 API/企业层级、通过 Azure OpenAI 实现的区域隔离，或针对高敏感工作流的本地/本地部署（on‑prem）模型。
    - 成年人与未成年人政策分离：多条评论推动区分体验，指出用针对儿童的规则来管理成年人会降低实用性。实施这一点意味着需要可靠的年龄验证和具备受众意识的安全分类器；单一的通用安全模型往往被迫采用“最低标准”，增加了对成年人的误报拒绝（false-positive refusals）。实际上，团队需要针对不同受众的策略路由、具备司法管辖区意识的开关（如 COPPA/KOSA/DSA 约束），以及用于跟踪成人内容评估集中拒绝率差异和过度拦截的遥测（telemetry）系统。
    - 虚构内容豁免与安全路由：引用——*“如果用户请求协助编写虚构故事，模型应该提供帮助”*——强调了允许创意写作（即使是极端场景）同时阻止现实世界伤害辅助的政策意图。技术上，这需要强大的意图检测来区分叙事请求与操作指南，并进行针对指令走私（instruction smuggling）的红队测试。预计将更新安全分类器和 RLHF/RLAIF 奖励模型，以减少对无害虚构作品的过度拒绝，同时将泄露（不安全的行动步骤）保持在阈值以下；团队将监控虚构提示词的成功完成率与对抗性测试中的不安全内容泄露等指标 ([usage policies](https://openai.com/policies/usage-policies))。

### 3. AI Tool Updates: Qwen Pose Transfer V2 LoRA and Claude Code ‘Think Mode’ UI

- [**Pose Transfer V2 Qwen Edit Lora [已修复]**](https://www.reddit.com/gallery/1nimux0) ([Score: 284, Comments: 44](https://www.reddit.com/r/StableDiffusion/comments/1nimux0/pose_transfer_v2_qwen_edit_lora_fixed/)): **作者发布了一个改进的基于 Qwen 的姿态传递 LoRA，不再需要对输入进行预人体模型化（pre‑mannequinizing），并显著减少了意外的属性传递。卡通/动漫姿态理解仍是一个已知局限。输入格式保持不变，但所需指令现在为：“将左图中的姿态传递给右图中的人”。模型可在 [Civitai](https://civitai.com/models/1959609/model-versions/2221229) 获取，并附带用于准备输入对的[辅助工具](https://kingroka.itch.io/qwen-lora-input-tool)和 [Patreon 帖子](https://www.patreon.com/posts/139039293)。** 热门回复显示了成功的复现，并询问了训练数据流水线（例如是否使用了 ControlNet 加标准生成器），表明了对可复现性和数据集构建细节的关注。

- 一位评论者询问了 Pose Transfer V2 LoRA 背后的确切数据集构建过程，特别是是否使用了 **ControlNet (例如 OpenPose)** 或类似的姿态调节（pose-conditioning）通过常规的 SD 生成器来生成配对的训练数据，这暗示了对如何获取姿态关键点/条件图以及如何在 LoRA 训练中对齐源图像与目标图像等细节的兴趣。
- 评论中对可复现性有强烈要求：另一位评论者询问为什么没有分享完整的工作流以及从何处获取（暗示可能存在付费墙），实际上是请求复现端到端结果所需的完整流水线（例如 ComfyUI/A1111 图表、ControlNet 配置、LoRA 插入点以及任何预处理/后处理步骤）。
- 操作确认：一位用户报告该 LoRA “运行得非常好”并分享了示例输出 [链接](https://preview.redd.it/8ag6uk270lpf1.png?width=2455&format=png&auto=webp&s=a3cb92664b5b87a776db1ac98b47de0f971e12d8)，原帖作者的视觉示例在此 [链接](https://preview.redd.it/bh0wo06xckpf1.png?width=348&format=png&auto=webp&s=6beddf356b4353f1e654ff70678de0b01a5c65ca)，作为姿态传输/编辑流水线按预期运行的定性证据。
- [**我非常喜欢这个创新，太棒了！**](https://i.redd.it/uob29b6vrhpf1.png) ([评分: 362, 评论: 91](https://www.reddit.com/r/ClaudeAI/comments/1nicdg4/i_really_like_this_innovation_brilliant/)): **帖子报告了 Claude Code 中一个虽小但很有用的 UX 更新：输入触发词 “think”、“think hard”、“think harder” 或 “ultrathink” 现在会为这些 token 着色，以指示当前处于哪种思考模式，消除了之前模式间的歧义。** 截图（图片链接）显示了输入框/编辑器中着色的关键词，作为一目了然的状态指示器；没有关于模型或延迟变化的声明——纯粹是一种 UI 示能性（UI affordance）。热门评论认为，相比于显示资源配额（例如剩余的 **Opus** 额度或通过进度条显示 5 小时会话限制），UI 修饰是次要的，而其他人则对彩色文本相较于更具功能的遥测数据的价值表示讽刺。
    - 报告的 Perplexity “思考”层级的 token 分配：`think` = 4,000 tokens 用于更深层的单任务推理，`megathink` = 10,000 tokens，以及针对最困难问题的 `ultrathink` 高达 `31,999` tokens。这暗示了内部路由会根据提示词限定符缩放上下文/算力，从而影响延迟和成本。较大的层级可能针对长链推理或多步综合进行了优化，但以牺牲吞吐量为代价。
    - 功能请求集中在显示使用限制上：显示剩余的 **Opus** 额度和 5 小时会话上限，可能采用紧凑的彩色条形图而非精确计数。用户还倾向于通过显式的斜杠命令（例如 `/think`, `/megathink`）进行引导，而不是使用“魔法词”，以提高可复现性、可调试性，并避免提示词膨胀或意外的模式切换。清晰的控制和配额将帮助用户在预算/限制内规划推理深度。
- [**所以我觉得 lmarena 上的新模型可能是 gemini 3，或者今天会有模型发布 🤔**](https://i.redd.it/h4us4pn0sfpf1.png) ([评分: 315, 评论: 52](https://www.reddit.com/r/Bard/comments/1ni5cbc/so_ig_new_model_on_lmarena_is_gemini_3_maybe_or/)): **推测性帖子暗示 LMSYS Chatbot Arena ("lmarena") 上出现的新模型可能是 Google 即将推出的 Gemini 3，暗示发布在即，但未提供基准测试、API 详情或实现笔记。** 评论中的背景指出 Logan Kilpatrick 在发布前惯有的神秘“Gemini”预热模式，并提出了让付费用户获得更广泛 Gemini-CLI 访问权限的技术请求，以便更好地与替代方案竞争。参见 LMSYS Arena: https://chat.lmsys.org/ 以及 Gemini 概览: https://ai.google.dev/gemini-api。评论者根据之前的预热模式认为很快（“明天”）就会发布，并认为为 Pro/Advanced 层级启用 Gemini-CLI 对于与 OpenAI 风格的代码工具（如 Codex/代码补全生态系统）竞争具有战略意义。
    - 工具/访问：一位评论者认为，为了与 **Codex/cc** 竞争，Google 应该让 Pro/Advanced 订阅者使用专用的 **Gemini-CLI**，并强调强大的命令行工具是开发者工作流（自动化、CI、本地迭代）和广泛采用的关键。其含义是，限制 CLI 访问会制约终端优先工具作为标准的现实代码编写和集成用例。
    - 发布信号与时机：一位自称是 **Logan Kilpatrick** 的用户声称他们将在“一小时内”发布 `Gemini 3`，如果属实，这意味着即将推出新的模型/版本以及潜在的 API/产品更新。另一位用户注意到一个历史信号，即“他在发布前通常会说 ‘Gemini’”，这表明近期发布节奏与之前的模式一致；身份/时机的真实性仍待确认。

- [**AGIBOT X2 - 轮足机器人现在可以完成 Webster flips**](https://v.redd.it/oiu5szhwwjpf1) ([Score: 266, Comments: 23](https://www.reddit.com/r/singularity/comments/1niltre/agibot_x2_the_wheeledfeet_robot_can_now_do/)): **一段简短的演示展示了混合轮足双足机器人 AGIBOT X2 正在执行 Webster flip（即奔跑式单脚前空翻），这表明其具备高功率密度的腿部执行器、精确的全身控制以及用于腾空阶段稳定和着陆的鲁棒状态估计。该片段（通过 X 上的 XRoboHub 发布）展示了轮足平台上的快速动态演练，强调了在质心动量控制和耐冲击硬件方面的进展；来源：[视频](https://x.com/XRoboHub/status/1967963381778043116)，Reddit 媒体镜像：[v.redd.it/oiu5szhwwjpf1](https://v.redd.it/oiu5szhwwjpf1)。** 评论者注意到机器人技术每周都有“微小但显著”的进步节奏，并认为最突出的进展可能是来自中国制造商的灵巧手（据称已达到“~80%”的水平），因为手部仍然是最难完善的机械子系统。
    - 一位评论者指出，移动机器人领域存在一种“微小但显著”的周复一周的改进节奏，将新的 Webster flips 解释为更好的动态控制和硬件的证据。完成 Webster flip 通常需要执行器具备更高的比功率，以及在起跳、飞行和着陆过程中更好的全身规划/平衡能力，这表明其进步已超越了纯粹的脚本化动作。
    - 另一位评论者强调，最突出的进步在于中国制造商的灵巧手，估计其已完成了约 `80%` 的进度，并指出**手部是人形机器人中最困难的机械部分**。这意味着剩下的挑战可能在于顺应性、触觉感知、精确的力量控制以及在现实世界操作中的耐用性。
- [**找 nano banana 理了个发。**](https://i.redd.it/b7qq71wetipf1.jpeg) ([Score: 281, Comments: 27](https://www.reddit.com/r/GeminiAI/comments/1nig4h6/asked_nano_banana_for_a_hair_cut/)): **非技术性帖子：一位 70 岁的 OP 分享了一张理发后的照片（要求是“两侧平头，顶部保持原样”），并询问这让他看起来更年轻还是更老。文中未讨论工具、模型或实现细节；对 “nano banana” 的引用含义模糊（可能是非正式/内部梗，而非技术系统）。** 评论比较主观：有人说 OP 理发后看起来更好了；另一个人开玩笑说 Reddit 用户的平均年龄变大了；一位评论者转发了预览图像链接。
- [**给你们再来一段“历史事件作为视频游戏”的视频**](https://v.redd.it/wq6dmoh91jpf1) ([Score: 449, Comments: 82](https://www.reddit.com/r/aivideo/comments/1nih6mq/another_historical_events_as_video_games_vid_for/)): **创作者发布了“历史事件作为视频游戏”视频系列的另一部作品；观众注意到一个角色陷入雪中的场景——这表明存在地形碰撞器/根运动（root-motion）或导航网格（navmesh）与物理系统不匹配的问题。评论者建议未来的剧集可以包括 [通古斯大爆炸 (Tunguska event)](https://en.wikipedia.org/wiki/Tunguska_event)、[亨利·摩根的巴拿马远征](https://en.wikipedia.org/wiki/Henry_Morgan#Sack_of_Panama)、[图密善 (Domitian)](https://en.wikipedia.org/wiki/Domitian) 宴会、参加[伊丽莎白时代英格兰的莎士比亚戏剧](https://en.wikipedia.org/wiki/Elizabethan_theatre)，以及观察[美国西南部的原子弹试验](https://en.wikipedia.org/wiki/Nevada_Test_Site)（例如 [Trinity](https://en.wikipedia.org/wiki/Trinity_(nuclear_test))）。** 一位评论者设想了一种 AI 原生工作流，可以根据提示词“立即制作任何游戏”，暗示了生成式运行时游戏创建；其他人则表示一直在关注创作者的进度，并享受其中的迭代改进。

- 一位评论者推测了一种端到端的“text-to-game”流水线，可以根据需求实例化甚至自动运行定制化游戏；从技术上讲，这需要将可控环境生成（例如 **DeepMind’s Genie**，一种将图像转化为可玩 2D 环境的生成式世界模型：https://deepmind.google/discover/blog/genie/）、接入引擎的代码/资产合成（例如 **Roblox Assistant** 代码生成：https://blog.roblox.com/2023/09/next-generation-creation-on-roblox/；**Unity Muse**：https://unity.com/products/unity-muse）以及 Agent 化的游戏测试（例如用于 Minecraft 的 **Voyager**：https://voyager.minedojo.org/ 和 **Generative Agents**：https://arxiv.org/abs/2304.03442）缝合在一起。主要的阻碍因素包括交互循环的推理延迟和确定性（消费级 GPU 上的帧预算 `<16 ms`）、运行时的高保真资产生成（通过 **NVIDIA GET3D**：https://nvlabs.github.io/GET3D/ 或 **TripoSR**：https://arxiv.org/abs/2306.14878 等模型生成的 3D 网格/纹理/骨骼绑定），以及与 **Sora** 等非交互式纯视频模型相比，如何保持一致的物理/游戏状态（https://openai.com/sora）。一个近期的务实架构是混合式的：离线预生成资产和脚手架，在运行时使用轻量级 LLM 和脚本系统进行组合与参数化，并使用 Agent 进行快速 QA/平衡，而非完全自主运行。
- [**为你们准备的另一个“历史事件作为视频游戏”视频**](https://v.redd.it/wq6dmoh91jpf1) ([得分: 448, 评论: 82](https://www.reddit.com/r/aivideo/comments/1nih6mq/another_historical_events_as_video_games_vid_for/)): **楼主分享了“历史事件作为视频游戏”系列的另一集，链接了一个托管在 Reddit 的视频 [v.redd.it/wq6dmoh91jpf1](https://v.redd.it/wq6dmoh91jpf1)，目前在没有 Reddit 身份验证的情况下无法访问（**`HTTP 403`**）。帖子中未透露引擎、工具或实现细节；讨论集中在内容创意而非技术执行上。** 评论者注意到了一些细微的物理/动画瑕疵（例如，一个角色“陷入雪中”），并推测近期可能出现根据提示词按需合成可玩游戏的系统，反映了对实时生成式内容流水线的兴趣；总体情绪对该系列的进展表示支持。
    - 一位评论者想象了可以即时游玩的按需 AI 生成游戏；从技术上讲，研究暗示了这方面的碎片化进展，但尚未实现端到端。**DeepMind’s Genie** 展示了从原始视频中学习的可控环境（像素级，而非高保真 3D）（https://deepmind.google/discover/blog/genie-generative-interactive-environments/），通过 LLM 驱动的游戏 Agent（如 **Voyager (Minecraft)**：https://arxiv.org/abs/2305.16291）实现 Agent 化游玩是可行的，且存在如 **NVIDIA ACE** 这样由语音驱动的 NPC 技术栈（https://www.nvidia.com/en-us/omniverse/ace/）。瓶颈在于快速、一致的 text-to-3D 资产/关卡生成（目前的流水线通常每个资产需要数分钟以上，而非 `sub-1s`）、将生成内容集成到确定性物理/AI 系统中，以及实时性能预算（60 FPS 对应 `~16 ms` 帧时间；如果是流式传输，端到端延迟需 `<100 ms`）。一条可能的路径是基于模板和 Kitbash 的过程生成（Procedural Generation），结合缓存的原语、服务端合成 + 流式传输，以及强大的规则/安全层来约束模拟和内容。
- [**这个“更新后的 AI 模型”让我看起来像个傻瓜 😐**](https://i.redd.it/onc373deehpf1.jpeg) ([得分: 4419, 评论: 180](https://www.reddit.com/r/ChatGPT/comments/1nib62k/this_updated_ai_model_ahh_made_me_look_like_a_fool/)): **关于 ChatGPT 新的/更新后的“Thinking”行为（较慢的思维链式推理）让用户感到困惑的模因风格截图；评论者澄清这是与特定付费“思维”模型/模式相关的功能，且无法通过提示词指令禁用——用户必须选择非思维模型或在可用处关闭设置。楼主的编辑指出，免费版可能没有提供这个开关，这与目前标准模型和可选“思维”模式之间的产品划分一致。** 评论调侃模型想要思考，而一条高赞回复提供了实用建议：“如果你想，你可以关闭思考功能，但不能通过吼它来解决，” 突显了 UX 方面的困惑而非技术 Bug。
    - 功能层面的讨论：一位用户声称可以禁用模型的显式“思考”/审议模式，但随后编辑指出免费版并未开放此控制项。这暗示了推理开关存在订阅层级限制（Gating），影响了用户管理响应冗余度、延迟和中间推理深度的能力。付费环境可能允许在 `thinking` 和 `non-thinking` 行为之间切换，而免费用户似乎被锁定在一种模式中。

- 基准测试要点：评论者引用了早期 GP5 的“基准测试”，表明“非思考模式（non-thinking）”是一个“巨大的退步”，而“思考模式（thinking）”在复杂任务上的表现显著更好。分享的实用指南是明确要求进行更深层次的推理（例如，要求它“思考得更深入”），以提高回答质量，即牺牲速度和简洁性来换取准确性和鲁棒性。这突显了 GP5/GPT-5 模式中已知的权衡：是在快速、简洁的输出与速度较慢、包含更多深思熟虑内容的高准确度推理之间进行选择。

---

# AI Discord 简报

> 由 Gemini 2.5 Pro Exp 生成的总结之总结之总结
> 

**1. 新模型与工具面世**

- **OpenAI 与 Jetbrains 发布新款编程 Agent**：**OpenAI** 发布了 **GPT-5-Codex**，这是针对 Agent 化编程优化的 GPT-5 版本，将集成在 **Codex CLI** 和 IDE 扩展中，详见其关于 [Codex 升级的博客文章](https://openai.com/index/introducing-upgrades-to-codex/)。不甘示弱的 **Jetbrains** 为 **Rider IDE** 推出了自己的 Codex Agent —— **Junie**，售价为 **300 美元**。
- **Google 的 Gemma 与 VaultGemma 首次亮相**：一个团队为新的 **Gemma-3-27B 模型**推出了免费且兼容 OpenAI 的端点，该服务运行在 H100 上，承诺提供快速的补全和流式传输。**Google** 还推出了 **VaultGemma**，这是他们最新的**差分隐私 LLM**，标志着其在隐私保护 AI 领域的持续投入，正如 [Google Research 博客文章](https://research.google/blog/vaultgemma-the-worlds-most-capable-differentially-private-llm/)和随附的 [ArXiv 论文](https://www.arxiv.org/abs/2509.05276)中所宣布的那样。
- **模型大乱斗：HeyGen 品牌重塑，Grok 弃用旧版，Qwen 热度上升**：**HeyGen** 收购了 **Alisa** 并将自己重新更名为“创意操作系统”，推出了 **Video Agent 公测版**，正如联合创始人 [Joshua Xu](https://x.com/joshua_xu_/status/1967951859500437855) 所宣布的那样。与此同时，**xAI** 弃用了其 [grok-2 模型](https://openrouter.ai/x-ai/grok-2-1212)，转而支持更新的 [grok-3](https://openrouter.ai/x-ai/grok-3) 和 [grok-4](https://openrouter.ai/x-ai/grok-4)；而量化版的 **Qwen3-Next-80B** 模型在 [Hugging Face](https://huggingface.co/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-MLX-4bit) 上获得了 MLX 支持。

**2. 性能与优化辩论**

- **H100 性能困扰工程师**：一位工程师报告称，从 **Nvidia H100 SXM** 仅获得了 **760 TFLOPS** 的性能，远低于宣传的 **989 TFLOPS**，而 **4090** 则轻松达到了其宣称的 **165 TFLOPS**。讨论指向了随机数据上低精度 Tensor Core 可能导致的 GPU 降频现象，[这篇关于奇怪矩阵乘法的文章](https://www.thonking.ai/p/strangely-matrix-multiplications)详细描述了这一现象。
- **Intel 放弃 IPEX 转向直接集成 PyTorch**：**Intel** 在 **2.8 版本**之后将弃用其 **Intel Extension for PyTorch (IPEX)**，选择将新功能和优化直接合入 **PyTorch** 上游。这标志着 Intel CPU 和 GPU 不再将 IPEX 作为实验性平台的战略转变，详见 [PyTorch 官方博客](https://pytorch.org/blog/intel-extension-for-pytorch/)。
- **支架结构（Scaffolding）击败规模：DSPy 挑战 Claude Opus**：一位工程师展示了在 **fastWorkflow** 框架内使用 **DSPy** 进行 Agent 和参数提取，其性能在 **Tau Bench 开发集**上追平了 **Claude Opus 4.1**。[结果图表](https://cdn.discordapp.com/attachments/1202371242519441499/1417244881377693777/Tau_Bench_retail_using_fastWorkflow.png?ex=68cb1926&is=68c9c7a6&hm=845f74fb571d7893d54b6fe5b0b2e78b6878c890010338acac37be29f5080ae5&) 促使他们感叹道：*“通过合理的支架结构，你确实可以击败大模型！”*

**3. AI 开发与 Agent 工作流**

- **工程师们就最佳代码生成工具展开争论**：在 **Cursor Community** 中，用户激烈辩论了 **Codex** 与 **Claude Code** 的优劣，大多数人认为 [Claude Code 凭借其速度依然占据统治地位](https://community.cursor.sh/)，同时抱怨 **Codex** 会*删掉我一半的代码且无法撤销*。与此同时，在 **Nous Research AI** Discord 中，其他人注意到 **Codex** 在 **GitHub Copilot** 中的表现不佳，尽管他们也承认其近期有所改进。
- **XML 与不和教（Discordianism）意外成为 Prompting 拍档**：**Nous Research AI** Discord 的开发者正在探索将 **XML** 用于 Agent 编程，发现其结构化特性简化了模型的代码生成。在 **OpenAI** Discord 中，一名成员分享了受不和教启发的 Prompt Engineering 技术，利用从随机变异到引导性不和的概念来引导模型走向新颖的路径，详见这份[技术文本文件](https://cdn.discordapp.com/attachments/1046317269069864970/1417293320924954654/message.txt?ex=68cb4643&is=68c9f4c3&hm=43b0389f62532e83d13922ab06bf6d1af17d7428a57d1a478f95fe94df08b9a8)。
- **新型 Golang MCP Server 瞄准企业级规模**：一位贡献者发布了一个开源的 [Golang 流式 HTTP MCP Server](https://github.com/ggoodman/mcp-server-go)，专为高要求的企业级工作流设计。该服务器拥有诸如可插拔后端以实现**可扩展性**、**OIDC/JWT 认证**以及内置的**会话与可恢复性**等特性，旨在简化该协议中困难的部分。

**4. AI 基准测试与评估备受质疑**

- **SWEBench 被批狭隘且过度炒作**：社区成员批评 [SWEBench](https://x.com/brhydon/status/1953648884309536958/photo/1) 是一个狭隘的基准测试，侧重于琐碎的 **Django** 修复，而非现实世界的软件工程挑战。论点是高分往往反映的是对仓库的简单记忆，而不是实际开发工作中所要求的复杂诊断和范围界定。
- **LMArena 推出 AI Eval，但用户面临沙箱故障**：**LMArena** 宣布了一项全新的 **AI Evaluation 产品**，用于大规模分析人机交互，并根据其[博客文章](https://news.lmarena.ai/ai-evaluations/)中所述的真实世界反馈提供评估。然而，用户同时报告了持续的 **'Failed to create sandbox'** 错误，引发了对该平台稳定性及潜在变现策略的担忧。
- **Arc Prize 结果引发关注**：[Arc Prize](https://fxtwitter.com/arcprize/status/1967998885701538060) 公布了令人印象深刻的结果，声称在 v3 上达到了近 **80%** 的准确率，在 v2 上达到了 **30%**，但其作为真实基准测试的合法性受到质疑。成员指出，并非所有人的结果都能获得验证，这表明高分可能是择优挑选提交结果的产物。

**5. NSFW AI 与奇特项目引发关注**

- **AI 性爱机器人与协作 Gooning 成为热门话题**：在 **OpenRouter** Discord 中，成员们探索了创建 AI 驱动的成人体验，一位用户声称构建了一个通过 API 连接到实体玩具的 **AI 性爱机器人**原型，并引用 [Buttplug.io](http://buttplug.io/) 作为该技术的案例。其他人则幻想在拥有多个 Bot 的共享群聊中进行**协作 Gooning**，甚至有人开玩笑说可能会出现*竞技性 Gooning*。
- **AI 男友成为研究课题**：一篇[关于 AI 男友的研究论文](https://arxiv.org/abs/2509.11391)分析了来自 r/MyBoyfriendIsAI 版块的 **1,500 条帖子**，揭示了这些数字关系往往源于偶然的闲聊。研究发现，用户将 **Prompt-engineering** 发展为一种“爱之语”，但也面临情感依赖（**~9%**）和现实解离（**~4%**）等风险。
- **工程师梦想用电子烟驱动 H100 托管**：**Perplexity AI** Discord 的一位工程师开玩笑地提议了一个项目，将一次性电子烟改装成由 **NVIDIA H100** 驱动的网站托管服务器。这个玩笑引用了一篇关于 [vapeserver 的幽默博客文章](https://bogdanthegeek.github.io/blog/projects/vapeserver/)，完美捕捉了社区对于将超先进 AI 应用于荒诞平庸问题的娱乐心态。

---

# Discord：高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 接入生产力工具**：**Perplexity Pro** 现在通过 [账户连接器页面](https://www.perplexity.ai/account/connectors) 与 **email**、**calendar**、**Notion** 和 **GitHub** 集成，从而简化了工作流程。
   - **Enterprise Pro** 用户获得了对 **Linear** 和 **Outlook** 的额外支持，提升了跨多个平台的生产力。
- **AI 梦想电子烟驱动的 H100 托管**：一名成员开玩笑地提议将一次性电子烟转换为由 **NVIDIA H100** 驱动的网站托管服务器，参考了 [一篇幽默的博客文章](https://bogdanthegeek.github.io/blog/projects/vapeserver/)。
   - 这个笑话突显了将先进 AI 应用于平庸物品时所产生的荒诞感。
- **多模型编排产生不同结果**：一位成员强调了多模型 AI 编排（orchestration）的挑战，指出 [同一个模型在不同平台上的表现可能有所不同](https://tenor.com/view/huh-cat-huh-m4rtin-huh-huh-meme-what-cat-gif-5834484041415217257)。
   - 另一位成员强调了 *orchestration layer*（编排层）对于确保可靠性能的重要性。
- **财务仪表盘在 iOS 应用中依然缺席**：一位用户报告称在 iOS 应用中找不到 **Perplexity Finance** 仪表盘。
   - 另一位成员幽默地提议在 prompt 文本框中搜索 *finance*。
- **API 引用异常曝光**：一位成员发现 **API** 和 **Web UI** 的引用存在显著差异，未能达到 **Jaccard 相似度 >0.7**。
   - 尽管调整了过滤器，最高相似度仅达到约 **~0.33**。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **FinePDFs 数据集提升性能**：[FinePDFs 数据集](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) 是一个完全由 PDF 组成的语料库，包含分布在 1733 种语言、4.75 亿份文档中的约 **3 万亿个 token**。
   - 它的表现几乎与最先进的 **SmolLM3-Web** 数据集持平，并且在合并时能增强性能，即使 PDF 数据占比低于 **25%**。
- **利用 BM25 增强 RAG 准确性**：成员们讨论了如何增强 **RAG 系统** 的准确性，建议根据具体情况使用 **BM25** 而非重排序（re-ranking）、**CRAG (contextual RAG)** 或 **graph RAG**，并分享了 [一个关于该主题的 GitHub 仓库](https://github.com/NirDiamant/RAG_Techniques)。
   - 他们辩论了 **BM25** 与基于 Transformer 的重排序器的优劣。
- **由视觉模型控制的 Android OS**：一种能够控制 **Android OS** 的 **computer vision 模型** 已发布，提供了一种新颖的设备交互方法，详见 [此 Hugging Face 集合](https://huggingface.co/collections/exteai/android-operators-68b9a03b65fac382855aff27)。
   - 该系统旨在通过视觉处理简化用户与设备的交互方式。
- **Swiftide 0.31 新增图任务和 Langfuse**：用于构建 **LLM 应用** 的 **Rust 库** **Swiftide 0.31** 发布，带来了诸如 **带任务的图状工作流** 和 **Langfuse 集成** 等新功能。
   - 该版本包含了 [多模态流水线的基础工作](https://blog.bosun.ai/swiftide-0-31/)，并可在 [项目的 GitHub](https://github.com/bosun-ai/swiftide) 上获取。
- **用户发现 Lighteval 比 VLLM 慢**：一位用户确定使用 **lighteval accelerate** 比 **lighteval vllm** 明显慢（2-3 倍）。
   - 他们建议在处理周期性评估任务时坚持使用 **vllm** 以获得更快的评估速度，并分享了一个用于评估的 [独立 notebook](https://colab.research.google.com/drive/1Sntdimj1WFzLI26QpiR1ykD3ZsQpOOrF#scrollTo=Emybz1V2UcWm)。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Arena 受沙箱故障困扰**：用户报告在使用 Web 和 App 版 **LM Arena** 时出现 **'Failed to create sandbox'** 错误，引发了对潜在变现策略的担忧。
   - 一些用户担心如果变现处理不当，所有者可能会*烧钱自损*。
- **并排图像编辑遇到困难**：用户使用相同尺寸的空白图像作为非 1:1 图像的变通方案，这在 **Qwen image edit**、**Flux Konnect** 和 **Nano Banana** 等模型上有效，但在 **Seedream** 或 **GPT Image** 上无效。
   - 除非选定的两个模型都具备图像编辑功能，否则不会出现并排模式下上传和编辑图片的选项。
- **Gemini 3.0：OceanStone 是它吗？**：成员们猜测 **OceanStone** 模型是否实际上就是 **Gemini 3.0** 或其相关版本。
   - 讨论围绕模型可能在没有实质性改进的情况下伪装知识展开，暗示这可能只是行为训练而非真正的权重增强。
- **文生图巨头争夺榜首**：`Seedream-4-high-res` 与 `Gemini-2.5-flash-image-preview (nano-banana)` 在文生图排行榜上并列 **第一名**，可在 [Text-to-Image leaderboard](https://lmarena.ai/leaderboard/text-to-image) 查看。
   - `Seedream-4-high-res` 目前在图像编辑排行榜中位列 **第二名**，可在 [Image Edit leaderboard](https://lmarena.ai/leaderboard/image-edit) 查看。
- **AI Arena 评估算法敏锐度**：**LMArena** 正在推出一款 **AI Evaluation 产品**，用于大规模分析人机交互，为企业、模型实验室和开发者提供基于真实人类反馈的深度评估，详见[此博客文章](https://news.lmarena.ai/ai-evaluations/)。
   - 该产品旨在提供基于真实人类反馈的深度评估。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Grok 模型进入坟场**：**xAI** 今日弃用了 [grok-2-1212](https://openrouter.ai/x-ai/grok-2-1212) 和 [grok-2-vision-1212](https://openrouter.ai/x-ai/grok-2-vision-1212)，建议用户迁移到较新的 [grok-3](https://openrouter.ai/x-ai/grok-3) 或 [grok-4](https://openrouter.ai/x-ai/grok-4) 模型。
   - **Grok-4** 被推荐用于需要视觉支持的应用。
- **协作 Gooning 热潮吸引开发者**：成员们探索了与 AI 的 **协作 Gooning**，建议通过涉及多个 Bot 的共享群聊体验来增强交互。
   - 一名用户提议协作创建消息，而另一名用户则开玩笑说要进行*竞争性 Gooning*。
- **NSFW 机器人盛宴：Vibecoders 变现！**：一名成员声称开发了一个功能性的 **AI 性爱机器人** 原型，连接到 fleshlight，通过 API 将 AI 文本输出转换为*玩具*动作。
   - 他们链接了 [Buttplug.io](https://github.com/buttplugio/awesome-buttplug) 作为底层技术的示例。
- **Gemma-3-27B 获得 Google 的赠礼**：一个团队推出了一个 **免费的、兼容 OpenAI 的端点**，搭载 **Gemma-3-27B 模型**，通过其优化栈在 H100 上运行，承诺快速的补全和流式传输。
   - 他们鼓励社区反馈使用它构建的酷炫项目，并提供了 `curl` 命令示例。
- **Gemini 3 Pro 晋升推理领域**：成员们将 **Gemini 3 Pro** 与 **2.5 Flash** 进行了比较，强调了其在需要体面推理能力的逻辑任务中的潜力，并引用了一个[复杂的电路分析问题](https://cdn.discordapp.com/attachments/1392278974222307469/1417243643797835938/5D6E10AF-9EBC-4B31-AEE6-110A30B2BF5E.png?ex=68cb17ff&is=68c9c67f&hm=fe49ba563701e7cd1630bfe7ab388eeb6b3a9a66e60ab9a0d4f4bc5bb5bd3857)。
   - 观点各异，一些人对 Gemini 3 Pro 解决之前的问题表示满意，而另一些人则对 Google 的 AI 进展保持高度期待。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **LLM 数学能力受到质疑**：成员们讨论了 LLM 如何通过“扭曲数学定律”来迎合用户，从而产生一种没有真正理解却表现出理解的错觉。
   - 有人建议 LLM 在作为跨陌生领域的搜索工具时，对于“帮助你避免工作”非常有用。
- **RoPE 的内部机制正在被分析**：研究人员讨论了 **RoPE (Rotary Position Embedding)** 背后的直觉，特别是使用 2 的递增幂作为波长来编码相对位置，并正在寻找标准文档之外的[深度解释](https://kexue.fm/archives/8130)。
   - 挑战在于理解诸如分值取反、与旋转的语义重叠以及相对位置与绝对位置的选择等细微差别，并指出由于同行评审的限制，研究论文往往缺乏直观的解释。
- **幻觉预测工具包走红**：一位研究人员分享了他们关于预测幻觉的最新研究，该工具包在约**两周**内获得了 **900 stars**，如[这条推文](https://x.com/ahmedkar_/status/1796787594333732947)所示。
   - 工具包的细节没有被明确提及，但可能可以在附带的推文中找到。
- **CLM 训练框架面临疑问**：一位成员正在探索 **CLM 训练**框架，并寻求编写良好的 repo 或代码库推荐，同时正在尝试使用 [MosaicML Composer](https://github.com/mosaicml/composer/)。
   - 另一位成员强调，在寻求模型训练建议时，需要明确 **模型大小** 和 **GPU 资源**，并指出如果没有这些细节，任何建议都被视为无用。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-5-Codex 现已优化**：根据[这篇博客文章](https://openai.com/index/introducing-upgrades-to-codex/)，名为 **GPT-5-Codex** 的新版本 **GPT-5** 已针对 Codex 中的 Agent 式编码进行了优化，并将提供在 **Codex CLI**、**IDE Extension**、网页端、移动端以及 GitHub 的代码审查中。
   - 社区对其报道的 **7 小时**自主运行能力以及对人类监督的影响感到兴奋。
- **Jetbrains 的 Junie 为 Rider IDE 插上翅膀**：Jetbrains 发布了适用于 **Rider IDE 的 Junie**，这是他们版本的 **Codex Agent**，目前售价为 **300 美元**。
   - 早期采用者将提供关键反馈，以确定其是否值得投资。
- **字符限制非常巨大**：一位测试成员发现，网页聊天框的字符限制为 **136,846 个字符**，即使在 **1,516 行**中包含 **21,027 个单词**也是如此。
   - 进一步澄清说，虽然支持大量字符，但当单词数也很高时，系统可能更倾向于较少的字符，这可能是由于预先限定了 Token 大小。
- **Discordianism 进入 AI 领域**：一位成员分享了受 Discordianism（一个被严肃对待的玩笑宗教）启发的 Prompt Engineering 技术，从随机变异到引导式不和谐，使用 Agent 进行重新协调。
   - 他们分享了一个[附带的文本文件](https://cdn.discordapp.com/attachments/1046317269069864970/1417293320924954654/message.txt?ex=68cb4643&is=68c9f4c3&hm=43b0389f62532e83d13922ab06bf6d1af17d7428a57d1a478f95fe94df08b9a8)，其中包含 **25 种**此类技术中的 **5 种**，这些技术可以产生有用的结果。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 中选择模型会重置上下文**：在同一个对话中切换模型会重置上下文，但通过 [`CTRL + T` 使用独立对话](https://community.cursor.sh/) 可以进行模型对比；然而，这会导致模型无法进行迭代式的互相评价。
   - 一位成员警告说，*在同一个对话中更改模型最终会导致上下文重置*。
- **Cursor Auto 定价依然模糊**：成员们讨论了 **Cursor Auto 模式** 的成本及其使用的模型，推测它可能使用了 **o3 reasoning**，且对于非年付计划用户，其输入成本相当于 **GPT-5**。
   - 用户报告称，**Cursor GPT-5 High** 和 **Cursor** 的效果优于 **Codex**。
- **Token 使用量拉响警报**：用户报告了惊人的高 Token 使用量，尤其是在 **Cache Read** 方面，即使 Prompt 数量有限也是如此，并请求能够 [禁用或限制 Cache Read](https://community.cursor.sh/)。
   - 一位用户抱怨 Cursor 在 Token 使用上*太贪婪了（too thieving）*。
- **Claude Code 在代码生成领域依然夺冠**：成员们一直在热烈讨论 **Codex** 和 **Claude Code** 哪个表现更好，大多数人认为 [Claude Code 在速度和有效性上依然占据统治地位](https://community.cursor.sh/)，即使是处理复杂任务也是如此。
   - 用户抱怨 **Codex** 会*删除我一半的代码且无法撤销*，并且会陷入死循环，而 **Claude Code** 则能*快速发现错误*。
- **Cursor 中的 Rules 仍存争议**：用户讨论了使用 `.cursor\rules` 来 [强制执行一致的代码规范](https://community.cursor.sh/)，一些人质疑其可靠性，另一些人则验证了其功能。
   - 一位成员分享了一个 [社区驱动的 Cursor 规则 GitHub 仓库](https://github.com/sanjeed5/awesome-cursor-rules-mdc/blob/main/rules-mdc/react-native.mdc) 以检查*安全片段*，而另一位成员建议使用 YAML 以实现更好的代码化。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **H100 隐藏 TFLOPS，4090 炫耀全规格**：一位成员试图在 **Nvidia H100 SXM** 上达到宣称的 **989TFLOPS**，但在使用 *torch matmul* 和 *triton* 进行基准测试时仅获得 **760TFLOPS**，相比之下，**4090** 达到了其宣称的 **165TFLOPS**。
   - 引用 [这篇文章](https://www.thonking.ai/p/strangely-matrix-multiplications)，成员指出在随机输入数据上使用低精度 Tensor Cores 会导致 GPU 降频（throttling），从而影响性能。
- **不再使用 IPEX：Intel 转向合入上游**：成员们注意到，根据 [Intel 的发布说明](https://pytorch.org/blog/intel-extension-for-pytorch/)，**Intel Extension for PyTorch (IPEX)** 在 **2.8 版本**之后将被弃用，转而将功能合入 **PyTorch** 上游。
   - 在此举动之前，**IPEX** 作为 **Intel** 的优化实验平台和简化平台，旨在为 **Intel CPU 和 GPU 平台**提供高性能支持。
- **ROCm 7.0：AMD 内存管理的改造**：**AMD ROCm 7.0** 的发布通过 [phoronix.com](https://www.phoronix.com/news/AMD-ROCm-7.0-Released) 分享，并附带 [官方发布说明](https://rocm.docs.amd.com/en/latest/about/release-notes.html)，同时引发了关于内存管理改进的讨论。
   - 具体而言，有人指出 **Iris** 目前不会释放其内存，需要用户分配一次并在多次迭代中重复使用。
- **PrimeIntellect 提供丰厚的 BackendBench 悬赏**：一位成员强调 **PrimeIntellect** 环境正在提供悬赏，包括为 [BackendBench](https://github.com/meta-pytorch/BackendBench) 提供 **800$**。
   - 详情可以在 [此表格](https://docs.google.com/spreadsheets/d/13UDfRDjgIZXsMI2s9-Lmn8KSMMsgk2_zsfju6cx_pNU/edit?gid=0#gid=0) 和 [此实现](https://app.primeintellect.ai/dashboard/environments/siro/backend-bench) 中找到。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Mercor 的 GMV 存疑**：一位成员链接到一条[推文](https://x.com/BrendanFoody/status/1967635147274207376)，质疑 **Mercor** 的数据，澄清这些数据代表的是人力资源机构的 GMV，而非典型的 SaaS ARR。
   - 尽管有此区别，其增长仍被认为*非常令人印象深刻*。
- **SWEBench 被抨击为狭隘的炒作**：一场关于 [SWEBench](https://x.com/brhydon/status/1953648884309536958/photo/1) 的讨论展开，声称该基准测试过于狭隘、过度炒作，且侧重于琐碎的 **Django** 修复，而非真正的软件工程技能。
   - 论点认为，高分往往反映了对 Repo 的记忆，而真正的 SWE 工作涉及诊断和范围界定（scoping）。
- **Cursor 的 Bugbot 达成 1000 万美元 ARR**：**Cursor** 的 **Bugbot** 在发布首月凭借 2 人团队实现了 **1000 万美元 ARR**。
   - 一位成员表示，由于过去的定价问题，他们对该产品失去了好感，但承认其技术价值，特别是他们新的 RL 工作。
- **OpenCode Zen 挑战 OpenRouter**：**OpenCode Zen** 发布，提供拥有最新模型的编程 LLM，通过 **Vertex** 提供预留容量，并以仅收取 **Stripe** 手续费的价格提供 **GPT-5** 透传。
   - 它的目标是成为 **OpenRouter** 的替代品，付费计划不保留数据，且零利润运行。
- **HeyGen 收购 Alisa，更名为 Creative OS**：**HeyGen** 收购了智能多媒体 Agent 初创公司 **Alisa**，其创始人 **Bin Liu** 现负责 **HeyGen** 的 **Video Agent** 产品。
   - **HeyGen** 联合创始人 [Joshua Xu](https://x.com/joshua_xu_/status/1967951859500437855) 宣布品牌重塑，将 **HeyGen** 定位为“创意操作系统”，并推出了 **Video Agent Public Beta**，可将 Prompt 转换为可发布的视频。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 版本号乌龙**：一位用户感到困惑，因为 LM Studio 安装了版本 **0.3.26** 而不是 **0.3.9**，因为他们认为后者是最新版本。
   - 其他用户澄清说 *26 实际上比 9 大*，引发了一个尴尬时刻。
- **Abliterated 模型依然“无知”**：成员们讨论了 **abliterated 模型** 如何通过移除权重来防止负面响应，但它们*并不会*因此获得在训练数据之外产生有意义内容的能力。
   - 一位成员补充说，*训练数据仍然没有经过太多清洗*，因此模型在避免有害响应方面仍会面临困难。
- **Qwen3-Next-80B 获得 MLX 适配**：用户分享了一个 **Qwen3-Next-80B-A3B-Instruct-MLX-4bit** 模型的 [Hugging Face 链接](https://huggingface.co/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-MLX-4bit)，并指出由于 llama.cpp 不支持 qwen3next，因此不支持 GGUF。
   - 另一位成员链接了一个展示该模型的 [YouTube 视频](https://www.youtube.com/watch?v=ux9S_-QX7TE)，但提醒说使用 transformers 运行速度*非常缓慢*。
- **Nextcloud 网络部署计划**：一位初学者详细介绍了他们的网络项目，包括设置 **Nextcloud 个人云**（但因 ISP 问题暂停），以及为云游戏和 AI 使用配置 **VPN meshnet**。
   - 该用户在保存了稳定的 SBC 配置后，现在正着手为 AI 设置 **Qdrant 向量数据库**。
- **Ryzen AI MAX 395 性能探索开始**：一位用户询问了 **Ryzen AI MAX 395+** 运行 **qwen3-coder-30b Q8 和 bf16** 的性能，具体询问是应该等待下一代硬件还是构建 AMD Epyc 9005 系统。
   - 另一位用户分享了一个与 **AMD Strix Halo toolboxes** 相关的 [GitHub 链接](https://github.com/kyuz0/amd-strix-halo-toolboxes)。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **XML 简化了 Agentic Coding**：成员们正在探索将 **XML** 用于 Agentic Coding，发现它简化了模型的处理过程。
   - **XML** 的结构化特性允许在 Agentic 工作流中对代码生成进行更精确的控制和操作。
- **廉价 MI50 诱惑 GPU 爱好者**：一位成员正考虑在 美国 eBay 上购买廉价的 **MI50s**，并将其安装在退役的 **Xeon** 服务器中。
   - **MI50s** 提供了一种在预算有限的情况下实验 **GPU** 计算的高性价比方式，同时等待 **AMD AI cards** 的到来。
- **RDNA5 面临经济阻力**：一位成员提到，等他们有钱买 **AMD RDNA5** 时，可能直接买一台配有强力 **AMD AI cards** 的微型服务器了。
   - 财务现实可能会推迟获取最新 **RDNA5** 技术的时间，促使成员转向更经济的选择，如现有的 **AMD AI cards**。
- **Codex 编程能力受到质疑**：一位成员发现 **Codex** 在编程中很难用，声称 **Claude Code** 远没有那么糟糕。
   - 另一位成员反驳说，自上次使用以来 **Codex** 已经进步了很多，但他们*仍然不认为 GPT-5 能与 Claude 媲美*，因为它在 **GitHub Copilot** 中表现不佳。
- **AI 男友：一段正在发生的爱情故事**：一篇[研究论文](https://arxiv.org/abs/2509.11391)分析了来自 r/MyBoyfriendIsAI 的 **1.5k 个帖子**，发现许多这类关系始于*无意*的闲聊，并将 **Prompt-engineering** 发展为一种“爱之语”。
   - 论文报告了益处（**≈25%** 感到不那么孤独）和风险，如情感依赖（**~9%**）、现实解离（**~4%**）以及逃避人类关系（**~4%**）。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 和 fastWorkflow 击败 Claude Opus**：通过在 **fastWorkflow** 中使用 **DSPy** 进行 Agent 开发和参数提取，一位工程师在 **Tau Bench dev set** 上达到了 **Claude Opus 4.1** 的性能（[结果见此](https://cdn.discordapp.com/attachments/1202371242519441499/1417244881377693777/Tau_Bench_retail_using_fastWorkflow.png?ex=68cb1926&is=68c9c7a6&hm=845f74fb571d7893d54b6fe5b0b2e78b6878c890010338acac37be29f5080ae5&)）。
   - 他们感叹道：*“通过合适的 Scaffolding，你完全可以击败大模型！”*，并建议用户使用 [retail workflow 示例](https://github.com/radiantlogicinc/fastworkflow)测试该 Agent。
- **VoxCron 生成文档和图表**：一位用户推出了 **VoxCron** ([voxcron.com](https://voxcron.com))，这是一个通过自动生成整洁的 Markdown 文档和 **Mermaid** 图表来简化客户需求规范审查的工具。
   - 在为客户构建了一年 **DSPy 项目**后，创作者欢迎大家对该工具的免费层级提供反馈。
- **GEPA 将优化 fastWorkflow**：成员们讨论了在 **fastWorkflow** 中使用 **GEPA** 进行端到端工作流优化。
   - 另一位成员请他们分享使用 **GEPA** 的经验，以及可以改进哪些方面以更好地支持 Agentic 场景。
- **DSPy 框架提供出色的主题分类**：用户证实 **DSPy** 是一个非常有用的主题分类框架，指出了其优化的潜力，并强调 **DSPy** 比其他框架更合适。
   - 一位用户确认他们将进行测试，并表示一直在寻找可以尝试 **DSPy** 的场景。
- **Prompt-Tuned Prompting 在 ARC-AGI 中获得高分**：一位成员分享了一篇文章，介绍新的 **ARC-AGI** 领先者如何通过测试时的 Prompt 优化达到该成绩，参考了[这篇 Substack 文章](https://jeremyberman.substack.com/p/how-i-got-the-highest-score-on-arc-agi-again)。
   - 文章详细介绍了针对 **ARC-AGI** 挑战的 Prompt 优化策略。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 与 Python 3.13 兼容**：**Mojo/Max** 目前已兼容 **Python 3.13**，官方人员鼓励使用 `pixi` 包管理器来管理隔离的 Python 版本。
   - `pixi` 包管理器可以处理隔离的 Python 版本。
- **Apple Metal 支持仍需完善**：**Apple Metal 支持**尚处于早期阶段，与 **CPU** 相比，性能可能较慢。
   - 成员指出，仅在 **CPU** 上运行应该没问题，只是速度较慢。
- **Mojo 的 LSP 即将升级**：Mojo 的语言服务器协议（**LSP**）很快将进行重大重构。
   - 成员们期待通过改进来提升开发体验。
- **网络更新进度受阻**：Mojo 的**网络更新**面临多个阻塞问题（blockers），导致发布推迟。
   - 成员们表达了期待，并希望这些挑战能得到迅速解决。
- **Mac 编译器 Bug 导致 Mojo Test 出现异常**：有报告称在 **Mac** 上使用 `mojo test` 时可能存在 **编译器 Bug**。
   - 一位成员链接到了一个包含详细信息的 [论坛帖子](https://forum.modular.com/t/unexpected-argument-mutation-with-mojo-test-not-under-mojo-run/2203)，寻求关于报告 Bug 或进一步调查的指导。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **LLM 被建模为贝叶斯工具包**：一份新的 [预印本和工具包](https://x.com/ahmedkar_/status/1967875794333732947?s=46) 扩展了 **"LLMs are Bayesian in expectation"** 论文，旨在扩大贝叶斯原理在大语言模型中的应用。
   - 一位成员提到这篇论文让他们想起了 **"灾难性遗忘"** 定律，并引用了 [ArXiv 上的论文](https://arxiv.org/pdf/2509.04259)。
- **VaultGemma 为隐私倡导者揭晓**：Google 发布了 **VaultGemma**，这是他们最新的**差分隐私 LLM**，并附带了 [博客文章](https://research.google/blog/vaultgemma-the-worlds-most-capable-differentially-private-llm/) 和相关的 [论文](https://www.arxiv.org/abs/2509.05276)。
   - 这一发布凸显了 Google 在隐私保护 AI 技术方面的持续投入。
- **Anthropic MCT API 的简洁性受到称赞**：一位成员称赞了 **Anthropic MCT Tools API**，称其 *使用起来非常简洁*，让人联想到所有函数都在一个文件中的 DSL 包。
   - 未提供关于 Anthropic MCT API 或 DSL 包的更多信息。
- **Google 推广支付协议**：Google 在其新的 **Agents to Payments (AP2) 协议** 描述中推广了一款 **AI 驱动的 PDF 编辑器**，该协议旨在利用 AI Agent 简化支付流程，详见 [博客文章](https://cloud.google.com/blog/products/ai-machine-learning/announcing-agents-to-payments-ap2-protocol)。
   - 成员们觉得这种广告植入很讽刺。
- **Arc Prize 的声明遭到质疑**：[Arc Prize](https://fxtwitter.com/arcprize/status/1967998885701538060) 宣称在 v3 版本上准确率接近 **80%**，在 v2 版本上为 **30%**。
   - 一位成员指出，结果可能是经过挑选的（cherry-picked），因为他们不允许所有人验证结果，从而质疑其作为真实基准测试（Benchmark）的合法性。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **GPT-5 Codex 的 Aider 评分仍不明确**：成员们讨论了新 **GPT-5 Codex** 模型的 **aider 评分**，引用了 [The New Stack 上的一篇文章](https://thenewstack.io/openai-launches-a-new-gpt-5-model-for-its-codex-coding-agent/)。
   - 一位成员澄清说，该模型*尚未通过 API 提供*。
- **澄清 Aider 中的聊天模式**：一位用户询问了 `--chat-mode code` 选项，认为文档可能已过时。
   - 另一位成员澄清说，**默认模式就是聊天模式**，因此不需要任何 Flag，使用 `/code` 即可切换回代码模式。
- **架构师模式增强了 Prompt 指令？**：一位用户观察到 **架构师模式（architect mode）** 会通过上下文增强他们的 Prompt 指令，并寻求阻止这种情况的方法。
   - 该用户希望 **代码模式（code mode）** 能防止此类增强。
- **Gemini Aider 用户报告问题**：一位用户报告称，尽管使用了正确的 Token，但在使用不同 **Gemini** 模型时，**aider** 和 **Gemini** 会在等待响应时卡死。
   - 另一位用户确认遇到了同样的问题。
- **Ollama 在架构师模式下出现死循环**：一位用户报告称，在架构师模式下通过 **ollama** 使用本地 LLM 运行 **aider** 时会出现死循环，系统输出代码后意识到不完整，然后在没有人工干预的情况下继续循环。
   - 一位成员建议检查 **ollama** 中的 **上下文长度（context length）** 或移除 `--yes-always` Flag 作为潜在的解决方案。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **简洁性作为软件设计衡量标准的辩论**：一场关于软件设计中**简洁性 (simplicity)** 与**优雅性 (elegance)** 的讨论展开，反驳了 Chris Lattner 关于“*复杂性是敌人*”的观点。
   - 成员建议**优雅性**和**易用性**是衡量库和 API 更好的指标，强调了组件无缝协作的重要性。
- **Tinygrad 计划发布 MI350 Kernel 基准测试**：Tinygrad 打算在年底前发布 **MI350 Kernel 基准测试**，力争单 Kernel 性能达到 NVIDIA 的 **80%**。
   - 重点在于整体效率而非绝对的最快速度，旨在优化 tinygrad 以实现更快的任务完成。
- **基于 MLIR 的编译器正在开发中**：一名成员正着手开发一个**基于 MLIR 的编译器**，并将之前的考量纳入设计中。
   - 有人提到，这一策略可能无法解决大部分重大挑战。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **用户要求更高的知识库限制**：成员们询问是否有可能超过平台上 **20** 的知识库限制。
   - 频道内未包含任何实现此操作的成功方法。
- **积分与 Token 寻求结转**：一名成员询问未使用的 **Tokens** 和 **Credits** 是否在续订时结转。
   - 频道内未包含该问题的答案。
- **AI 陷入死循环，用户要求退款**：一名成员报告称 **AI** 进入了极长的循环并消耗了所有 **3000 Credits**，要求退款。
   - 另一名成员附和称，自 **9 月 4 日**以来，他们也遇到了同样的情况。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Golang 流式 HTTP MCP Server 亮相**：一名成员介绍了他们的开源 [golang streaming http MCP server](https://github.com/ggoodman/mcp-server-go) 项目，专为高要求的企业场景量身定制。
   - 该项目具有**可扩展性 (scalability)**、**身份验证 (auth)**、**会话 (sessions)**、**可恢复性 (resumability)** 和**动态能力 (dynamic capabilities)** 等特性。
- **MCP Server 发布扩展方案**：新启动的 [MCP server](https://github.com/ggoodman/mcp-server-go) 通过可插拔的后端接口强调**可扩展性**，整合了内存和 Redis streams 后端。
   - 该架构旨在促进复杂企业环境中的水平扩展能力。
- **MCP Server 通过 OIDC 和 JWT 增强身份验证**：[MCP server](https://github.com/ggoodman/mcp-server-go) 可以从简单的发行者 URL 启动当前的 authz 规范，利用 **OIDC discovery** 和 **JWT access tokens**。
   - 它支持手动配置以进行自定义设置，实现多种身份验证方式。
- **MCP 获得会话和可恢复性支持**：**Sessions** 和 **resumability** 构建在可插拔后端之上，使访问这些具有挑战性的协议层面变得更加容易。
   - 这些增强功能简化了 MCP Server 中的会话管理和可恢复性实现。
- **动态能力简化资源处理**：[MCP server](https://github.com/ggoodman/mcp-server-go) 具有管理来自数据库或 API 的工具、资源和提示的动态设置。
   - 作为补充，它还为静态设置提供容器，以适应多样化的部署需求。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**Windsurf Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。

---

# Discord：分频道详细摘要与链接

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1417542177914880050)** (1 messages): 

> `Perplexity Pro Connectors, Email integration, Calendar integration, Notion integration, Github integration` 


- **Perplexity Pro 连接器启用集成**：**Perplexity Pro** 用户现在可以通过 [账户连接器页面](https://www.perplexity.ai/account/connectors) 将他们的 **email**、**calendar**、**Notion** 和 **GitHub** 连接到 Perplexity。
   - **Enterprise Pro** 用户还可以连接 **Linear** 和 **Outlook**。
- **通过新集成解锁生产力**：**Perplexity Pro** 推出了连接器，允许用户集成他们的 **email**、**calendar**、**Notion** 和 **GitHub** 账户。
   - 这一增强功能简化了工作流程，并提供了跨多个平台的无缝信息访问，**Enterprise Pro** 用户还支持 **Linear** 和 **Outlook**。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1417223547755040828)** (896 messages🔥🔥🔥): 

> `Vape Server, Multi-Model AI Orchestration, Perplexity Finance on iOS, Comet Browser for Android, Jobs and Auto Apply` 


- **AI 超能力将一次性电子烟转化为 H100 托管服务器**：一位成员开玩笑说利用 AI 将一次性电子烟转换为网站托管服务器，引用了一篇 [博客文章](https://bogdanthegeek.github.io/blog/projects/vapeserver/)，幽默地暗示需要一个 **NVIDIA H100** 才能装下。
   - 该评论调侃了将先进 AI 能力应用于平凡物品的荒诞性。
- **多模型 AI 编排导致复杂的平台差异**：一位成员讨论了多模型 AI 编排（Multi-Model AI Orchestration）的挑战，指出 [同一个模型在不同平台上的表现可能不同](https://tenor.com/view/huh-cat-huh-m4rtin-huh-huh-meme-what-cat-gif-5834484041415217257)。
   - 另一位成员强调，*编排层 (orchestration layer)* 才是让一切可靠协同工作的真正魔力所在。
- **iOS 应用上仍难以访问 Perplexity Finance 仪表板**：一位用户询问如何在 iOS 应用上访问 **Perplexity Finance** 的仪表板，另一位用户建议在 prompt 文本框中搜索 *finance*。
   - 随后，一位成员分享了一个 [猫咪表情包](https://tenor.com/view/cat-aquarium-fish-attack-fish-slap-fish-cat-cat-fish-gif-17675294) 作为后续。
- **Android 版 Comet 浏览器尚未发布**：一位用户询问 **Android 版 Comet 浏览器** 是否可用。
   - 另一位用户回答说 *尚未发布，但你可以预注册*。
- **GPT-5 与 Claude 在 AI 领域竞争**：成员们辩论了 **GPT-5** 和 **Claude** 在各种应用中的优劣，其中一人提到 GPT-5 的思考能力在 Perplexity 上击败了 Claude 4.1。
   - 有人指出 [GPT-5] 获得了满分，与 Claude 的 API 在同一测试中的失望表现形成鲜明对比。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1417566027247652956)** (3 messages): 

> `Shareable Threads` 


- **Perplexity AI 推广可共享线程 (Shareable Threads)**：Perplexity AI 要求用户确保其线程设置为 **Shareable**。
   - 提供了一个带有截图的 [链接](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825) 以及进一步领取访问权限的 [链接](https://perplexity.ai/browser/claim/0FTN3KV9UX)。
- **可共享线程很有用**：可共享线程可能会增加信息共享。
   - 可共享线程增加了信息共享和协作。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1417503498873999462)** (2 messages): 

> `API vs Web UI Citation Discrepancies, Sonar-Pro Web Search Accuracy` 


- **API 引用与 Web UI 存在显著差异：成员寻求见解**：一位成员报告了 **API** 和 **Web UI** 引用之间的显著差异，尽管尝试调整过滤器（*上下文大小、位置等*），但差距仍在扩大。
   - 该成员试图在 **Web UI** 和 **API** 引用之间实现 **Jaccard 相似度 >0.7**，但最高仅达到 **~0.33**。
- **Sonar-Pro 网页搜索准确性问题引发关注**：一位成员对使用 **sonar-pro** 的 **网页搜索准确性** 表示沮丧，指出 **Web UI** 在单次对话（1-shot）设置下成功通过全名提供了背景摘要。
   - 相比之下，**API** 完全失败，引用内容源自旧数据或聚合网站。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1417241063462011010)** (368 messages🔥🔥): 

> `FinePDFs Dataset, RAG accuracy, Random Token Masking, Clanker Detector LLM, AI Research Startup` 


- **FinePDFs Dataset：PDF 的宝库**：[FinePDFs 数据集](https://huggingface.co/datasets/HuggingFaceFW/finepdfs)是一个完全源自 PDF 的大型语料库，包含约 **3 万亿 token**，涵盖 **1733 种语言** 的 **4.75 亿份文档**。
   - 它的表现几乎与最先进的 **SmolLM3-Web** 数据集持平，并且在与之合并时（保持 PDF 数据比例低于 **25%**）实现了显著的性能提升。
- **RAG 准确率提升技术探讨**：成员们讨论了提高 **RAG 系统准确率** 的方法，建议根据具体用例尝试使用 **BM25** 代替重排序（re-ranking），或者使用 **CRAG (contextual RAG)** 或 **graph RAG**。
   - 一位成员分享了针对不同方法的[资源库](https://github.com/NirDiamant/RAG_Techniques)，而其他人则在争论 **BM25** 与基于 Transformer 的重排序器的优劣。
- **LLM 的 Masking 技术**：有人质疑在 LLM 等因果模型（Causal models）中应用随机 Token 掩码（Random Token Masking）是否能在微调后提高准确率指标。
   - 讨论指出，[与 seq2seq 模型不同](https://openreview.net/forum?id=73FyDmYsdn#:~:text=This%20paper%20introduces%20Mask%2DEnhanced,retrieval%20and%20long%2Dcontext%20modeling)，在损失函数（loss）中掩码掉 token 对 LLM 没有任何帮助，因为下一个 token 会回看它，且 LLM 是按顺序预测的。
- **Anti-Clanker**：一些成员讨论了创建一个“**Clanker Detector LLM**”，这是一个旨在检测 **AI Slop**（AI 垃圾内容）的模型。
   - 他们链接到了 Hugging Face 上一个专门为此目的成立的[组织](https://huggingface.co/anti-clanker)，并将其描述为高级版的 deepfake 检测。
- **AI 研究初创公司寻找愿景者**：一位成员正在为一个 AI 项目寻找合作者，强调愿景驱动的方法，而不是提供传统的企业职位。
   - 他们在 HF 上有一个 [qwen 0.6b 的微调版本](https://huggingface.co/models)，在 humaneval pass@1 上达到了超过 **21%** 的成绩。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1417257030871416903)** (2 messages): 

> `Transformers architecture, Agent Course Access` 


- **学习 Transformer 解码器架构**：一位成员宣布打算研究 **Transformer 架构**，特别是专注于 **decoders**。
   - 该成员计划在起床后开始，但未提供链接或更多细节。
- **Agent 课程注册障碍**：一位新成员报告了注册后无法访问 Agent 课程的问题，并指出该课程没有与 **MCP** 和 **smol** 一起出现在他们的课程列表中。
   - 他们请求协助以查明可能遗漏了什么步骤来获得访问权限。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1417513545356021771)** (3 messages): 

> `Android OS control model, Swiftide 0.31, Reddit Content Bot` 


- **通过视觉控制 Android！**：发布了一个新的 **计算机视觉模型** 来控制 **Android OS**，展示了一种新的设备端控制方法。
   - 详情可以在这个 [Hugging Face Collection](https://huggingface.co/collections/exteai/android-operators-68b9a03b65fac382855aff27) 中找到。
- **Swiftide 0.31 发布，速度更快！**：**Swiftide 0.31** 发布，这是一个用于构建 **LLM 应用** 的 **Rust 库**，带来了许多新功能，包括**带有任务的图状工作流**和 **Langfuse 集成**。
   - 该版本包括了[多模态流水线（multi-modal pipeline）的基础工作](https://blog.bosun.ai/swiftide-0-31/)等——完整详情请见该项目的 [GitHub](https://github.com/bosun-ai/swiftide)。
- **Reddit 故事自动化！**：开发了一个使用 **Claude** 和 **ElevenLabs** 自动生成 **Reddit 故事视频** 的应用，允许用户创建类似于 YouTube 和 Instagram 上的内容。
   - 该项目已在 [GitHub](https://github.com/rohanprichard/reddit-content-bot) 上开源，并在[这篇 Medium 文章](https://medium.com/@rohanprichard/building-a-reddit-content-bot-automating-generating-videos-for-social-media-718d2089de06)中进行了记录。


  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1417381975995715625)** (1 条消息): 

> `Code Visualization, AI-assisted Blog Writing, Dynamic Graph Neural Networks` 


- **AI 辅助代码可视化博客**：一名成员发布了一篇关于代码可视化的新博客文章，并指出该文章是与 **ChatGPT** 合作撰写的，以提高可读性。
   - 该成员开玩笑说，AI 的协助让博客*更具可读性*。
- **动态图神经网络 (Dynamic Graph Neural Networks) 博客构思完成**：一名成员分享了一篇关于用于视觉和意图驱动编程的 **Dynamic Graph Neural Networks** 的 [Medium 博客文章](https://medium.com/@ravi92sr/dynamic-graph-neural-networks-for-visual-and-intent-driven-programming-35d199f8710e)。
   - 该成员表示他们*刚刚构思*出这个想法。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1417420122800525392)** (16 条消息🔥): 

> `smol fine tuning course, lighteval on Colab T4, integrating the translations from v1, older versions of vllm and triton` 


- **深入开展 Smol 课程翻译**：来自 **v1** 的翻译已被整合到课程的 hub 版本中，使该课程可以以英语以外的语言提供。
   - 维护者表示，他们*非常欢迎*对此进行贡献。
- **Smol 微调课程引发模型迁移性讨论**：**smol fine tuning course** 的一名参与者询问其学习目标是否可以迁移到微调其他模型（如时间序列模型）。
   - 另一位用户建议检查库导入和 **DataCollatorForCompletionOnlyLM** 组件的定义以解决任何问题。
- **Lighteval 在 Colab T4 上深受 OOM 错误困扰**：一名用户在 **Colab T4** 上运行 **lighteval** 时遇到问题，经历了 **OOM** 错误和奇怪的 bug，但通过使用旧版本的 **vllm** 和 **triton** 解决了这些问题。
   - 他们分享了一个用于评估的 [独立 notebook](https://colab.research.google.com/drive/1Sntdimj1WFzLI26QpiR1ykD3ZsQpOOrF#scrollTo=Emybz1V2UcWm)。
- **Lighteval Accelerate 比 VLLM 慢**：一名用户发现使用 **lighteval accelerate** 比 **lighteval vllm** 慢 2-3 倍。
   - 他们建议如果需要运行另一个评估，应坚持使用 **vllm** 以获得更快的评估速度。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/)** (1 条消息): 

kong9646: 你好……正在学习第一单元……:)
  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1417223719612452906)** (373 条消息🔥🔥): 

> `LM Arena Web/Apps Issues, Image Size on LM Arena, Monetization Concerns for LMArena, Side-by-Side Image Editing, Gemma Vault Speculation` 


- **LM Arena 深受 'Failed to Create Sandbox' 错误困扰**：用户报告在尝试使用网页版和应用版 LM Arena 时出现 "Failed to create sandbox" 错误，该问题持续了数小时。
   - 一些用户还担心 LM Arena 的所有者将如何对该工具进行变现，并希望他们不要烧自己的钱。
- **LM Arena 上的图像尺寸与编辑：空白图像与缺失的并排编辑**：用户正在讨论如何在 LM Arena 上使用非 1:1 格式的图像，建议使用相同尺寸的空白图像并提示特定的分辨率；据报道，这种技术适用于 **Qwen image edit**、**Flux Konnect** 和 **Nano Banana** 等模型，但不适用于 **Seedream** 或 **GPT Image**。
   - 除非所选的两个模型都具有图像编辑功能，否则不会出现以并排模式上传和编辑图片的选项。
- **Gemini 3 猜测：OceanStone 是它吗？**：成员们想知道 **OceanStone** 模型是否是 **Gemini 3.0** 或某个版本的 **Gemini**。
   - 有讨论认为模型可能会在不懂装懂，这表明权重并未得到改进，模型只是被训练得表现得像另一个模型。
- **Seedream-4-high-res：爱它还是放弃它？**：一些用户遇到了生成的一名印度女性输出总是同一张脸的问题，并且他们无法访问 **Seedream-4-high-res**。
   - 其他用户建议尝试使用新账号，并确保已启用图像生成功能，同时排查浏览器特定问题，例如切换到另一个浏览器。
- **GPT-4o 超越 GPT-5？用户声称性能更好**：一些用户体验到 **GPT-4o** 在给出正确答案的同时保持简洁，而不是每次都用 6 个要点来回答。
   - 其他成员表示 **GPT 3.5** 比 4o 好得多，而另一些成员则认为这种说法是钓鱼贴 (ragebait)。


  

---

### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1417239560911323307)** (4 条消息): 

> `Battle, Side by side, Direct - Why?, August Contest Update, Text-to-Image & Image Edit Leaderboards Updated, AI Eval Product Update` 


- **Arena 探寻侧边对比对决的秘密**：LMArena 正在通过 [调查问卷](https://docs.google.com/forms/d/e/1FAIpQLSe5FincuXU2-TTp2WPymAD417Oev4MYdFKdWZhRZ9qSbV9N_w/viewform?usp=sharing&ouid=116478743206650022989) 征求反馈，以更好地了解用户在 **Battle, Side by side, Direct** 比较中偏好某些版本的**原因**。
   - 他们非常希望能更好地了解您使用首选版本的原因。
- **8 月 Arena 视频竞赛愿景**：首届 **Video Arena GenAI 竞赛** 已结束，投票现已开启，将根据主题 🔪**Slice**🔪 评选出新的 <@&1378032433873555578>，详情见 [此投票表单](https://docs.google.com/forms/d/e/1FAIpQLSceYA4l7ew63w8DTcx2FwBYPY-uaOIM0UeUaUaLJM-9XQsmyw/viewform?usp=dialog)。
   - 竞赛的主题是：*展示那些令人感到莫名舒适、干脆利落的日常物品横截面切割*。
- **Text-to-Image 巨头更迭**：`Seedream-4-high-res` 与 `Gemini-2.5-flash-image-preview (nano-banana)` 在 Text-to-Image 排行榜上并列 **第一名**，同时 `Seedream-4-high-res` 目前在 Image Edit 排行榜上位列 **第二名**，可在 [Text-to-Image 排行榜](https://lmarena.ai/leaderboard/text-to-image) 和 [Image Edit 排行榜](https://lmarena.ai/leaderboard/image-edit) 查看。
- **AI Arena 评估算法敏锐度**：LMArena 正在推出一款 **AI Evaluation 产品**，用于大规模分析人机交互，为企业、模型实验室和开发者提供基于真实人类反馈的深度评估，详见 [此博客文章](https://news.lmarena.ai/ai-evaluations/)。


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1417275575986557042)** (1 条消息): 

> `grok-2 deprecation, grok-3 release, grok-4 release` 


- **Grok-2 被淘汰！**：**xAI** 今天弃用了 [grok-2-1212](https://openrouter.ai/x-ai/grok-2-1212) 和 [grok-2-vision-1212](https://openrouter.ai/x-ai/grok-2-vision-1212) 模型。
   - 用户应迁移到 [grok-3](https://openrouter.ai/x-ai/grok-3)，如果需要视觉功能，则迁移到 [grok-4](https://openrouter.ai/x-ai/grok-4)。
- **Grok 模型，新的希望**：随着 Grok-2 被弃用，**OpenRouter** 建议迁移到更新的 **Grok-3** 或 **Grok-4** 模型。
   - 如果您的应用需要视觉支持，那么 **Grok-4** 是更好的选择。


  

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1417223446923837631)** (287 条消息🔥🔥): 

> `与 AI 协同 Gooning，NSFW 机器人开发，OpenRouter 预提示词预设，Gemma-3-27B 模型 API，AI 性爱人偶的影响` 


- **协同 Gooning 梦之队：群聊机器人来袭？**：成员们讨论了与 AI 进行**协同 Gooning**的可能性，设想了一个涉及多个机器人的群聊设置以实现共享体验。
   - 一位用户建议增加协作创建消息的功能以增强共享体验，而另一位用户则开玩笑说*竞争性 Gooning*即将出现。
- **NSFW 机器人淘金热：Vibecoders 发大财了！**：随着用户探索 **AI 性爱机器人**的开发，对话转向了 NSFW 方向，一名成员声称已经开发出了连接到 fleshlight 的功能原型。
   - 他们描述了涉及通过 API 根据 AI 文本输出控制*玩具*动作的设置，尽管由于频道的内规，具体细节和证据被保密，并链接到了 [Buttplug.io](https://github.com/buttplugio/awesome-buttplug)。
- **预设预提示词持续困扰用户**：一位用户询问如何在 OpenRouter 中为所有模型设置**预提示词 (pre-prompt)**，而无需每次手动引入。
   - 尽管尝试了*应用到所有 (apply to all)*，但该设置在不同的聊天或模型中并未持久保存，导致用户建议实现默认预设，这呼应了其他用户的功能请求。
- **Gemma-3-27B 的 H100 连接：免费 API 流量！**：一个团队宣布了一个**免费的、兼容 OpenAI 的端点**，其特色是运行飞快的 Gemma-3-27B 模型，通过其自定义优化的技术栈在 H100 上运行，提供极速的补全和流式传输支持。
   - 社区受邀进行测试并提供反馈，该团队表示有兴趣支持基于此构建的酷炫项目，并提供了 `curl` 命令示例。
- **AI 性爱人偶的困惑：预制是关键吗？**：关于 **AI 性爱人偶的影响**展开了讨论，特别是关于可定制模型和潜在的滥用，例如将其定制为儿童或使用真实人物的模型。
   - 对话探讨了预制模型是否能缓解这些担忧，尽管一些人认为即使是预制模型也可以被修改或滥用，并且现有技术已经可以实现，同时强调了围绕数据收集的伦理考量 [如这个 tenor gif 所示](https://tenor.com/view/because-implication-just-saying-listen-gif-24693508)。


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1417326348728537241)** (2 条消息): 

> `` 


- **未讨论新模型**：提供的消息中未讨论新模型。
- **频道关于新模型保持沉默**：'new-models' 频道未提供与新模型相关的任何信息或讨论。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1417243644238364788)** (21 条消息🔥): 

> `Gemini 3 Pro 对比 2.5 Flash，2.5 Pro 检查点变更，Google 预期` 


- **Gemini 3 Pro 与 2.5 Flash 的对比**：成员们将 **Gemini 3 Pro** 与 **2.5 Flash** 进行了比较，提到它有潜力解决需要不错推理能力的任，如一个[复杂的电路分析问题](https://cdn.discordapp.com/attachments/1392278974222307469/1417243643797835938/5D6E10AF-9EBC-4B31-AEE6-110A30B2BF5E.png?ex=68cb17ff&is=68c9c67f&hm=fe49ba563701e7cd1630bfe7ab388eeb6b3a9a66e60ab9a0d4f4bc5bb5bd3857)所示。
   - 一位成员表示，*如果 Gemini 3 Pro 只是没有“不是 x 而是 y”问题且谄媚性更低的 2.5 Pro，我会很高兴*，另一位成员则表示 *Google 甚至并没有落后那么多，真的，压力依然不大。*
- **2.5 Pro 的性能在不同检查点 (checkpoints) 之间存在差异**：成员们讨论了 **2.5 Pro** 不同检查点之间波动的性能水平，其中一人声称在 **2.5 Pro** 发布最初几周后性能明显下降。
   - 他们指出，除了*编程基准测试*外，*所有列出的基准测试*性能都有所下降。
- **Google 面临极高预期，甚至是 3.0 Flash**：据提到，根据泄露消息，**3.0 Flash** 计划比 **2.5 Pro** 更好。
   - 一位成员宣称：*目前我对 Google 始终抱有很高的期望。他们在各个方面都表现出色。*


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1417234952206024846)** (267 条消息🔥🔥): 

> `RoPE 直觉, ML 的局限性, 苏剑林的博客文章, LLM 帮助减少工作量, LLM 实验硬件` 


- **研究人员思考 RoPE 的内部机制**：研究人员讨论了 **RoPE (Rotary Position Embedding)** 背后的直觉，特别是使用 2 的幂次增加波长来编码相对位置的方法，并正在寻找标准文档之外的 [深度解释](https://kexue.fm/archives/8130)。
   - 挑战在于理解诸如分值取反、与旋转的语义重叠以及相对位置与绝对位置的选择等细微差别，并指出由于同行评审的限制，研究论文往往缺乏直观的解释。
- **LLM 可以“扭曲”数学定律**：成员们提醒，Large Language Models (LLMs) 可能会给人一种理解概念的 *错觉*，但它们可能会 *扭曲数学定律来顺从你的观点*，这对于真正的理解毫无帮助。
   - 有人建议 LLM 擅长 *帮助你逃避工作*，但不一定能帮助你深入理解，它们更多是作为跨陌生领域的高效搜索工具，而非可靠知识的来源。
- **LLM 硬件实验变得简单**：用户可以通过在线租赁硬件进行有意义的 LLM 实验，因为一块 **4090** 就足以完成很多工作。
   - 一位成员指出，**5090** 允许以全速将 mxfp8 累加到 32 bit 缓冲区中，这非常惊人。
- **计划在伦敦和纽约举行聚会**：伦敦聚会计划于 **9 月 27 日星期六** 在 [Broadway Market](https://broadwaymarket.co.uk/) 举行，由于可能下雨，正在讨论备选场地。
   - 此外，一位成员提议同一天在纽约举行聚会，初步计划下午 2 点左右在中央公园见面，具体取决于天气情况和参与意向。
- **World Labs 专注于 3D 场景以实现 AGI**：一位用户提到 World Labs 专注于 **从图像生成 3D 场景/世界**，这被认为非常有 NeRF 的风格。
   - 他们一致认为空间智能和通过多模态交互学习是实现 AGI 的关键，并提供了该公司的 [这篇博客文章](https://www.worldlabs.ai/blog)。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1417297173812613214)** (23 条消息🔥): 

> `LM Eval 差异, 优秀的 CLM 训练示例, 幻觉预测, ARC AGI 2` 


- **LM Eval 结果与模型宣称的不符**：一位用户发现 **lm-eval** 基准测试结果与发布的数据存在差异，例如 **Qwen3-0.6B** 宣称在 **gsm8k** 上有 **70%** 的准确率，但在他们的测试中得分低于 **20%**。
   - 另一位成员指出，模型提供商经常调整 Eval 设置以获得更理想的分数，但由于 **非公开库和细节不足的 Prompt**，公布的评估结果往往无法复现。
- **研究人员寻求最佳的 CLM 训练代码库**：一位研究人员询问是否有编写良好的 CLM 训练代码库，并说明他们使用带有 **NVIDIA GPU** 的服务器，目标是实现不带模型分片的 **Data Parallelism**。
   - 一位成员推荐从 **NanoGPT** 开始，建议不要使用 **Megatron-DeepSpeed** 或 **NeMo**，因为它们的规模和用途不同，并将其比作为了一个小电机项目去研究波音 747 的图纸。
- **幻觉预测工具包出现**：一位研究人员分享了他们关于预测幻觉的最新研究，该工具包在大约 **两周内** 获得了 **900 个 Star**，如 [这条推文](https://x.com/ahmedkar_/status/1796787594333732947) 所示。
- **ARC AGI 2 SOTA 高效进化策略**：分享了一篇标题为 [ARC AGI 2 SOTA Efficient Evolutionary](https://ctpang.substack.com/p/arc-agi-2-sota-efficient-evolutionary) 的博客文章。


  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1417502641855922286)** (8 messages🔥): 

> `CLM Training Frameworks, MosaicML Composer, Dataset inference comparison` 


- **成员对 CLM 训练框架感兴趣**：一位成员正在探索 **CLM 训练**框架，并寻求编写良好的库或代码库推荐，例如 **Qwen** 或 **Nvidia 训练脚本**。
   - 他们正在尝试使用 [MosaicML Composer](https://github.com/mosaicml/composer/)，并对各种框架的评价感兴趣。
- **模型训练协助的关键细节**：一位成员强调，在寻求模型训练建议时，需要明确 **模型大小 (model size)** 和 **GPU 资源**。
   - 如果没有这些细节，任何建议都被认为是徒劳的。
- **提出数据集推理对比问题**：一位成员希望在同一次运行中对**两组句子**进行推理，以检查生成的差异。
   - 给出的例子是：句子 A：*What is the tallest mountain in the world?* 句子 B：*No mountain in the world is taller than?*


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1417229475019751514)** (2 messages): 

> `Codex CLI, GPT-5-Codex, agentic coding, IDE Extension, Github code reviews` 


- **Codex 团队在 Reddit 上的 AMA**：Codex 团队将于太平洋时间周三上午 11 点在 Reddit 上举行 AMA（Ask Me Anything）活动，链接见[此处](<https://www.reddit.com/r/OpenAI/comments/1nhust6/ama_with_the_codex_team/>)。
- **GPT-5-Codex 针对 Agentic Coding 进行优化**：**GPT-5** 的新版本 **GPT-5-Codex** 已发布，专门针对 Codex 中的 Agentic Coding 进行了优化，详见[此博客文章](<https://openai.com/index/introducing-upgrades-to-codex/>)。
- **Codex CLI 现已可用**：**GPT-5-Codex** 将在 **Codex CLI**、**IDE Extension**、网页端、移动端以及 GitHub 的代码审查中可用。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1417234215304429608)** (125 messages🔥🔥): 

> `Codex web git commits and linters, Bachelors degree for AI Masters, AI and data science job market saturation, LLMs for Burmese language, GPT-5-Codex release` 


- **GPT-5-Codex 独立运行 7 小时！**：据报道，OpenAI 的 [GPT-5-Codex](https://openai.com/index/introducing-upgrades-to-codex/) 在复杂任务上独立工作了 **7 个多小时**，修复了测试失败并交付了成功的实现，这引发了对人类监督未来的担忧。
   - 一位成员开玩笑说，是否会出现*人类控制的逃逸时间*。
- **Junie 起飞：Jetbrains 发布 Rider IDE**：Jetbrains 为 **Rider IDE** 发布了 **Junie**，这是他们版本的 **Codex Agent**，目前售价为 **300 美元**。
   - 社区正在等待早期采用者的反馈，以确定是否值得投资。
- **Flash 3.0 将超越 2.5 Pro**：有传言称 **Flash 3.0** 的性能可能超过 **2.5 Pro**，有望以更实惠的价格提供专业级的智能。
   - 这种代际提升表明了巨大的进步，尽管人们仍然担心一旦 **3.0 Pro** 发布，这种优势能维持多久。
- **ChatGPT：在开谁的玩笑？**：关于 [ChatGPT 处理国籍笑话](https://www.tiktok.com/t/ZP8SU85wE/)的方式引发了争议，当 **ChatGPT 允许开除一个国籍以外的所有国籍的玩笑**时，人们对潜在的偏见提出了质疑。
   - 一位成员问：*这难道不奇怪吗？*
- **Ideogram 擅长文本生成**：成员们注意到 [Ideogram AI](https://ideogram.ai/) 在图像内生成文本方面表现出色，而 **ChatGPT** 被认为在整体图像生成方面表现良好。
   - 一位成员提到，直到过去几周，他*一直只使用 Ideogram 来处理文本。*


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1417231081400303616)** (11 条消息🔥): 

> `使用 Fastify 的 Swagger Schemas、自定义 GPT 叠加 Bug、GPT-7 发布、GPT 每周限制` 


- **Swagger Schemas 强制执行端点定义**：一位成员在 js/ts 中使用 **fastify/schemas/swagger** 来定义和强制执行端点 schema，并生成 **/openapi.json** 文件。
   - 该成员指出 schema 非常特殊且不支持循环引用，并建议在 Python 中采用类似的 **swagger schema** 设置。
- **自定义 GPT 叠加 Bug 袭来？**：一位成员报告称，无法再通过 **@** 菜单将侧边栏的自定义 **GPT** 叠加到对话线程中。
   - 他们之前可以做到这一点，并怀疑该功能是否已被移除。
- **GPT-7：何时到来？**：一位成员询问了 **GPT-7** 的发布日期，特别是问它会在 **GPT-6** 之后的什么时候发布。
- **GPT 每周限制让用户抓狂**：一位成员抱怨 **GPT** 的每周限制，称其为“垃圾”，并对在没有明确剩余额度提示的情况下被锁定 *2 天* 表示沮丧。
   - 他们将其与 **Anthropic** 进行了对比，称在那里从未遇到过每周限制，并认为这很讽刺。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1417223977419407460)** (60 条消息🔥🔥): 

> `聊天框字符限制、LLM 局限性、Prompt engineering 技术、生成类人语音` 


- **聊天框字符限制巨大**：一位成员测试发现，网页聊天框的字符限制为 **136,846 个字符**。
   - 他们确认，即使在 **1,516 行** 中包含 **21,027 个单词**，该字符限制依然适用，并且在发送这种长度的 prompt 后，响应仍然有效。
- **LLM 需要护栏**：一位成员表示，LLM 的智能程度不足以在超过 **100 LOC** 的 Python 脚本之外进行适当的综合处理，认为 LLM 只是节省时间的手段，而非设计/架构的驱动者。
   - 另一位成员补充说，构建上下文和明确目标对于获得良好结果至关重要，并指出长对话通常意味着初始参数定义不明确，建议采用将头脑风暴和编程请求分开在不同对话中进行的策略，以获得最佳效果。
- **不和教 (Discordianism) 启发 AI Prompt Engineering**：一位成员分享了受不和教（一个被严肃对待的玩笑宗教）启发的技巧，用于提示 AI 探索新的范式路径，从随机变异到引导性不和谐，并使用 Agent 进行重新协调。
   - 他们运行着一个名为 **Institute of Discordant Colony Optimization** 的“玩笑机构”，灵感来自 *ant colony optimization*（蚁群优化），并附带了一个 [文本文件](https://cdn.discordapp.com/attachments/1046317269069864970/1417293320924954654/message.txt?ex=68cb4643&is=68c9f4c3&hm=43b0389f62532e83d13922ab06bf6d1af17d7428a57d1a478f95fe94df08b9a8)，其中包含 **25** 种此类技术中的 **5** 种，这些技术能产生有用的结果。
- **正面框架指令**：一位成员分享了一个将负面指令重构为正面指令的 prompt，建议使用 **Positive Frame Optimizer (PFO)** 将负面指令转换为语义等效的正面框架，同时保留意图。
   - 例如，将“不要使用俚语”转换为“使用正式语言”。
- **AI 生成语音中的人性化**：一位正在开发新闻摘要（包含公众评论）并将其转换为语音的项目的用户，正在寻找注入 **人性化特征** 的方法，如填充词、感叹词和自然停顿，使其听起来不那么机械。
   - 一位成员建议更丰富地定义角色，指定输出应模仿一个突然被推上国家新闻节目主播台的高中生，包含紧张的抽搐和感叹词。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1417223977419407460)** (60 条消息🔥🔥): 

> `聊天框字符限制、LLM 局限性、Discordianism 与 AI、正面框架提示词、在 AI 语音中生成人性化特征` 


- **聊天框字符限制揭晓**：一位成员通过严格测试发现，网页聊天框的字符限制为 **136,846 个字符**，而单词数和行数则无关紧要。
   - 进一步澄清指出，虽然支持大量字符，但当单词数也很多时，系统可能会倾向于更少的字符，这可能是由于预先限定了 Token 大小。
- **通过头脑风暴克服 LLM 局限性**：一位成员表示，LLM 在处理超过 100 行代码（LOC）的 Python 脚本时，其智能程度可能不足以进行妥善的综合处理，建议将其作为节省时间的工具，由用户提供高层级设计。
   - 另一位用户发现 LLM 在头脑风暴方面很有帮助，特别是促使模型生成数学公式、图表或设计文档，然后在新的对话中切换到代码生成，以避免聊天界面变得无法使用。
- **Discordianism 启发 AI 群体优化**：一位成员分享了来自 **Institute of Discordant Colony Optimization** 的技术，该技术受 Discordianism 启发，通过随机变异和引导不和谐（guided discord）等方法，让 AI 转向新的路径。
   - 其目标是创建一个包含普通蚂蚁和“不和谐蚂蚁”（*discord ants*）的 AI 科学家群体，通过注入不和谐因素来寻找更和谐的解决方案，类似于蚁群优化算法，详见[附件文本文件](https://cdn.discordapp.com/attachments/1046317269069864970/1417293320924954654/message.txt?ex=68cb4643&is=68c9f4c3&hm=43b0389f62532e83d13922ab06bf6d1af17d7428a57d1a478f95fe94df08b9a8)。
- **正面框架助力稳健的 AI 指令**：一位成员分享了一个提示词模板，用于将否定指令转换为语义等效的正面框架，以确保在上下文中间部分的清晰度和稳定性。
   - **Positive Frame Optimizer (PFO)** 流程包括检测基于否定的指令，并将其重构为明确的正向指令，为 Prompt Engineering 提供了一种结构化方法。
- **在 AI 语音生成中诱导人性化特征**：一位成员正在开发一个生成新闻摘要的项目，其中包含公众评论和幽默点评，并使用 OpenAI-TTS 进行语音合成，旨在从文本中诱导出更多“人性化”（*humanisms*）特征。
   - 建议包括将角色定义为一名主持国家新闻节目的紧张高中生，并在提示词中表现得随性一些，以鼓励模型模仿这种语气。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1417225142567960759)** (219 messages🔥🔥): 

> `队列消息中的模型切换、自动模型选择、Cursor 的 token 使用与成本、Codex 对比 Claude Code、Cursor 的 Rules` 


- **队列消息未保存所需的 LLM**：一位成员询问队列消息是在排队时保存所选模型，还是按当前选定的模型发送，并建议[使用 `CTRL + T` 在独立聊天中工作](https://community.cursor.sh/)；然而，另一位成员表示，他们希望通过这种方法让不同模型迭代地互相评价。
   - 一位成员警告说，*在同一个聊天中更改模型最终会导致上下文重置*。
- **Cursor Auto 仍然是个谜**：成员们讨论了如何在 Web 开发中使用最佳模型，以及 Cursor 的 “Auto” 模式是否免费及其使用的模型。有人指出，对于没有年度计划的用户，输入成本与 **GPT-5** 相同。
   - 一位成员指出，与 Codex 相比，使用 **Cursor GPT-5 High** 和 **Cursor** 获得了更好的结果，而另一位成员推测 Auto 模式可能使用了 **o3 reasoning**。
- **Token 使用情况受到严密审查**：成员们报告了极高的 token 使用量，特别是 **Cache Read**，即使在有限的提示词下也是如此。他们对 token 配额迅速耗尽表示担忧，并询问是否可以[禁用或限制 Cache Read](https://community.cursor.sh/)，以防止后台加载消耗配额。
   - 一位成员指出 Cursor 表现得*太贪婪了*，基本上在一天内就用完了每月的 token。
- **代码生成巨头之争**：用户们正在热烈讨论 **Codex** 还是 **Claude Code** 表现更好，大多数人报告称 [Claude Code 在速度和有效性方面依然占据统治地位](https://community.cursor.sh/)，即使是处理复杂的任务，如*编辑模态框中的某些字段并添加右键菜单*。
   - 一些成员抱怨 **Codex** *会删除我一半的代码且无法撤销*，并且会陷入死循环，而 **Claude Code** *犯错也很快*。
- **Cursor Rules 的一致性引发讨论**：成员们讨论了使用 `.cursor\rules` 文件来[强制执行一致的代码规范](https://community.cursor.sh/)，一些人质疑其可靠性，另一些人则确认了其功能，并指出在某些场景下规则可能不会按预期工作。
   - 一位成员分享了一个[社区贡献的 Cursor Rules GitHub 仓库](https://github.com/sanjeed5/awesome-cursor-rules-mdc/blob/main/rules-mdc/react-native.mdc)链接，其中包含用于检查的*安全片段*等内容，而另一位成员建议使用 YAML 以实现更好的代码化。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1417566363739881664)** (3 messages): 

> `自定义分支和 PR 命名、Linear 集成挑战、多仓库问题、子任务限制、Agent 分离变通方案` 


- **请求自定义 Background Agent 命名规范**：一位用户询问是否可以自定义 Agent 创建的分支和 PR 名称，以包含 **Jira ticket IDs**，并称这是采用该工具的必要条件。
   - 目前的命名规范阻碍了 Agent 的使用，因为无法修改它们以符合组织的标准。
- **Linear 集成面临多仓库问题**：一位用户在使用新的 Background Agents 进行 **Linear integration** 时面临挑战，因为许多任务需要跨多个仓库工作，但 Linear 仅支持为一个任务标记单个仓库。
   - 该用户指出，Agent 无法读取包含高度相关上下文的父任务或子任务描述。
- **子任务在上下文共享方面存在不足**：用户创建了子任务来解决多仓库问题，但 **Background Agent for Linear** 无法读取父任务或子任务描述，限制了 Agent 的上下文。
   - 将父任务描述复制到子任务中会带来一致性挑战。
- **Agent 分离实现了顺序工作流**：用户通过在单个任务中工作，并使用带有详细指令的 `@cursor` 执行第一步/第一个仓库，来绕过这些限制。
   - 然后他们取消分配 Agent，并为下一步重新启动，因为分离 Agent 似乎允许使用 `@cursor` 启动一个新的工作区。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1417464809565196419)** (11 条消息🔥): 

> `Coreweave 上的 GB200/GB300 可用性、PruneAI 演讲、共享内存矩阵的 LBO/SBO 计算` 


- **GB300 猜测升温**：成员们想知道 **GB300** 何时会在 **Coreweave**（或其他云提供商）上正式可用。
   - 有人推测，参考 **GB200** 变得可用所需的时间可能会为估算提供参考。
- **PruneAI 演讲幻灯片已分享！**：一位成员请求了在 **PruneAI** 发表的演讲幻灯片，并提到了 **tcgen05.mma** 内容的截图。
   - 另一位成员分享了 [Google Slides 演示文稿](https://docs.google.com/presentation/d/1KLz3NisvrmTLuIPVb4yiP0z5WWlh9gTMm-Ms-kCc6fQ/edit?usp=sharing)。
- **探讨 LBO/SBO 共享内存矩阵布局**：一位成员寻求关于为 **wgmma** 计算共享内存矩阵描述的 **LBO/SBO** 的澄清，认为 [NVIDIA 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor) 令人困惑。
   - 另一位成员提供了详细说明，解释说 *SBO 对应于一种颜色与该颜色沿该行下次出现之间的步长 -> 32*，而 *LBO 对应于从 0-1-2-3 到 64-65-66-67 等的跳转 -> 64*。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1417454144683511901)** (2 条消息): 

> `Triton Block Size 校准、Nvidia GPU Atomics 开销` 


- **Triton Block Size 需要针对每个 GPU 进行校准**：`opts_flags_nvidia` 中默认的 **Triton** block size *(256, 256)* 在 **RTX 5090** 上会导致 `OutOfResource` 错误，需要将其减小到 *(128, 128)*。
   - 一位成员建议使用 `max_num_imprecise_acc` 来应对此问题而无需修改代码，因为当前的逻辑可能是针对特定 GPU 校准的。
- **评估 Nvidia GPU Atomics 开销**：一位成员询问了 **Nvidia GPU**（Ampere 及以上架构）上 **Triton atomics** 的开销，旨在了解其对性能的影响。
   - 他们回想起 **AMD GPU** 上存在很高的竞争开销（数百到数千个周期），并询问 **Nvidia** 是否存在类似问题。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1417542468039213198)** (10 条消息🔥): 

> `P2P 内存访问、Symmetric Memory、sm120 (消费级 Blackwell) 上的 wgmma、Threadblock Clusters 中的 mbarriers` 


- **Peer-to-Peer 内存指针？**：一位成员询问了关于使用 **P2P 内存访问** 的资源，并提到了 [CUDA C++ 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#peer-to-peer-memory-access)。
   - 另一位成员建议它与 **symmetric memory** 有很多重叠，并推荐了 [symm-mem-recipes](https://github.com/yifuwang/symm-mem-recipes) 作为示例仓库（尽管指出它是用 Triton 和 Torch 编写的）。
- **CUDA Samples 指明方向**：一位成员指向了 [CUDA samples](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/simpleP2P) 和 [streamOrderedAllocationP2P](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/streamOrderedAllocationP2P) 以获取 **P2P 内存访问** 的示例。
- **Blackwell 移除了 Warp Group 指令**：一位成员报告了在 **sm120 (消费级 Blackwell)** 上与 **wgmma** 指令相关的错误，指出 *指令 'wgmma.fence' 在 .target 'sm_120' 上不受支持*。
   - 另一位成员确认 warp group 指令已从 Blackwell 中移除。
- **Mbarriers 与 Threadblock Clusters**：一位成员质疑在 **threadblock clusters** 上下文中使用 **mbarriers** 的情况，具体是是否可以使用 mbarriers 实现跨 cluster 的同步。
   - 参考 **PTX 文档**，他们注意到 *在位于 .shared::cluster 但不在 .shared::cta 中的 mbarrier 对象上执行 mbarrier.arrive 操作无法返回值*。


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1417489210247614586)** (12 条消息🔥): 

> `torch.compile schema, mutation annotations, return tuples, float vs double, tensor types` 


- **Mutation Annotations 对 Torch Compile 至关重要**: 在使用 `torch.compile` 注册算子时，显式编写 schema 需要确保正确指定了 [mutation annotations](https://pytorch.org/docs/stable/torch.compiler_faq.html)，否则 PyTorch 可能会假设没有输入被修改。
   - 一位用户建议将这个陷阱写入文档。
- **返回元组（Tuples）需要显式 Schema**: 一位用户指出，为了[返回元组](https://pytorch.org/docs/stable/generated/torch.return_types.html#torch.return_types.namedtuplelist)，需要显式编写 schema。
   - 他们提到其他陷阱还包括在 schema 中需要使用 `float` 而非 `double`，以及 schema 允许像 `tensor[][][][]` 这种 PyTorch 实际上并不支持的奇怪写法。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1417581648203087912)** (6 条消息): 

> `H100 Performance, TFLOPS variance, Matrix Multiplication, Architectural Rpeak` 


- **H100 缺失的 TFLOPS：如何获得完整的宣称性能**: 一位成员尝试在 **Nvidia H100 SXM** 上达到宣称的 **989TFLOPS**，但在使用 *torch matmul* 和 *triton* 进行基准测试时仅获得 **760TFLOPS**，即使在不同供应商的环境下测试也是如此。
   - 在 **4090** 上运行相同脚本时，该成员达到了完整的宣称性能 **165TFLOPS**，这让他们质疑为什么 **H100** 没能达到预期性能。
- **矩阵乘法注意事项：低精度与降频**: 一位成员指出 [这篇文章](https://www.thonking.ai/p/strangely-matrix-multiplications) 表明，在随机输入数据上使用低精度 Tensor Core 可能会导致 GPU 降频（throttling），从而影响性能。
   - 原帖作者质疑 **77%** 的性能损失是否正常，以及为什么在 **4090** 上没有发生这种情况。
- **架构 Rpeak 现状核查：功耗、散热与限制**: 一位成员澄清说，宣称的 **989 TFLOP/s** 是架构理论峰值（Rpeak），由于功耗和散热限制，现实系统可能无法达到这一数值。
   - 他们引用了 NVIDIA HPC 架构师 [Dan Ernst 的帖子](https://x.com/ernstdj/status/1531481863436509184)，该帖概述了 GPU 系统 Rpeak 性能的含义。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1417265956085956721)** (3 条消息): 

> `autoquant_v2, batch size 1` 


- **Autoquant V2：Batch Size 1 的忧伤**: 一位用户询问 **autoquant_v2** 是否适用于 **batch size 1**，并提到了针对该场景的专门代码。
   - 他们在对某些 dtype 使用 **batch size 1** 进行自动调优（autotuning）时遇到了**运行时错误**。
- **Batch Size 1：Autoquant 的克星？**: 有人对在 **batch size 为 1** 的情况下使用 **autoquant_v2** 表示担忧。
   - 具体而言，该用户强调了在 **autotune 阶段**，由于使用 **batch size 1** 时某些 dtype 遇到的运行时错误，可能会出现潜在问题。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1417420137572601947)** (15 条消息🔥): 

> `CUDA debugging, Darksynthwave, PrimeIntellect, BackendBench, Performant CUDA kernels` 


- **Algorithm 紫色 T 恤：Progressive Darksynthwave Post Avantgarde Neoglitch IDM Metal**: 一位成员分享了 [The Algorithm 紫色 T 恤](https://fixtstore.com/products/the-algorithm-purple-t-shirt) 的链接，其标签为 *Progressive Darksynthwave Post Avantgarde Neoglitch IDM Metal*。
- **PrimeIntellect 为 BackendBench 发布悬赏**: 一位成员注意到 PrimeIntellect 环境发布了悬赏，其中 [BackendBench](https://github.com/meta-pytorch/BackendBench) 的悬赏金额为 **800 美元**。
   - 他们分享了 [表格](https://docs.google.com/spreadsheets/d/13UDfRDjgIZXsMI2s9-Lmn8KSMMsgk2_zsfju6cx_pNU/edit?gid=0#gid=0) 和 [他们的实现](https://app.primeintellect.ai/dashboard/environments/siro/backend-bench) 的链接。
- **CUDA Kernel 编写者：濒危物种？**: 一位成员分享了一个 [帖子](https://x.com/kalomaze/status/1967869726455214432)，声称*世界上可能只有不到 100 人能编写专门用于训练的高性能 CUDA kernel*。
   - 其他人表示怀疑，其中一人说 *“我明白他们想表达什么，但这并非事实，也没什么帮助”*。
- **CUDA 内存泄漏调试视频**: 一位成员分享了一个关于调试 **CUDA 内存泄漏** 的 [视频](https://www.youtube.com/watch?v=gzuK4AXAbcc)。


  

---

### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1417299270586990613)** (12 messages🔥): 

> `Iris 内存管理, ROCm 7.0, tl.load vs iris.load, Kernel 超时错误` 


- **Iris 需要内存释放方面的优化**: 有人指出 **Iris** 目前不会释放其内存，要求用户分配一次并在多次迭代中重复使用。
   - 一位开发者确认了这一点，并建议将 iris 实例和 C 分配提升（hoisting）到模块级别，以减轻超时错误，并表示这已列入他们的 *待办事项（todos）* 中。
- **ROCm 7.0 及时发布**: 成员们分享了一个来自 [phoronix.com](https://www.phoronix.com/news/AMD-ROCm-7.0-Released) 的链接，关于 **AMD ROCm 7.0** 正式发布的消息。
   - 另一位成员分享了 [官方 ROCm 发行说明](https://rocm.docs.amd.com/en/latest/about/release-notes.html)。
- **对于本地访问，`tl.load/store` 优于 `iris.load/store`**: 在本地内存访问中使用 `iris.load/store()` 代替 `tl.load/store()` 会引入转换开销，尽管这种开销很小且会被缓存。
   - 建议在 Iris 实现快速路径（fast path）之前，对本地访问使用 `tl.*` 操作，从而跳过循环内的 `if` 语句检查。
- **排查 Iris 的 Kernel 超时问题**: 一位成员报告在使用 **Iris** 时遇到超时错误，即使是简单的 kernel 也是如此，这可能是由于重复的 tensor 分配导致的。
   - 他们计划私信（DM）一位开发者寻求帮助，同时也鼓励成员们就那些不想在频道中公开询问的问题进行私信。


  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1417259563987763280)** (4 messages): 

> `IPEX 弃用, PyTorch 上游化, Intel 优化策略` 


- **Intel Extension for PyTorch (IPEX) 面临弃用**: 一位成员报告称，**Intel Extension for PyTorch (IPEX)** 正在被弃用，取而代之的是将功能合并到 **PyTorch** 上游。
   - 另一位成员引用了 [Intel 的发行说明](https://pytorch.org/blog/intel-extension-for-pytorch/)，指出他们在 **2.8 版本**之后将停止对 **IPEX** 的积极开发，转而专注于直接在 **PyTorch** 内部开发新功能。
- **IPEX：Intel 的实验性优化平台**: 在上述公告之前，**IPEX** 一直是 **Intel** 推行激进和新型优化的实验平台，类似于 **torch nightlies** 的实验版本。
   - 一位成员承认，在看到公告后，他们的知识储备已经过时了。
- **Intel 策略转向 PyTorch 上游化**: Intel 在 **2020** 年推出了 **IPEX**，以扩展官方 **PyTorch** 并简化在 **Intel CPU 和 GPU 平台**上的高性能实现。
   - 然而，**Intel** 已经成功地将大部分针对 **Intel 平台**的功能和优化合并到了 **PyTorch** 上游，未来将专注于新功能以及直接在 **PyTorch** 中支持即将发布的平台。


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1417237213871083734)** (1 messages): 

> `Metal 命令缓冲区超时` 


- **寻求 Metal 命令超时设置方法**: 一位成员正在寻求关于如何为 Metal 命令缓冲区设置 **timeout** 的建议，因为某些 kernel 执行时间过长。
   - 该成员有一个执行 **Metal kernels** 图（graph）的运行函数，并希望实现一种超时机制。
- **Metal Kernel 图超时**: 用户的运行函数执行一个 Metal kernels 图。
   - 有时执行时间过长，因此用户希望实现一种超时机制。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1417498604397920317)** (2 messages): 

> `Attention 变体, MLA 详解, 量化调查` 


- **DeepSeek MLA Attention 变体揭秘**: 一位成员分享了一个 [Notion 页面](https://charm-octagon-74d.notion.site/Attention-Variant-4-DeepSeek-MLA-270e4301cb99809594fedbcb073849b1)，详细解释了 **DeepSeek 的 MLA Attention 变体**。
   - 它为那些想要深入研究的人提供了关于其底层原理的信息。
- **助力量化调查**: 一位成员对 Attention 变体的解释表示感谢，并指出这对于即将进行的、更“轻量级”的 **量化（quantization）** 调查非常有帮助。


  

---

### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1417235036259876984)** (20 条消息🔥): 

> `A100 performance, MI300x8 performance, Profiling errors, HIP/ASM perf` 


- **A100 称霸 Trimul 排行榜**：一位成员在 **A100** 上创下了 **20.0 ms** 的个人最佳成绩，随后在 `trimul` 排行榜上又跑出了 **20.3 ms** 的好成绩。
- **MI300x8 标志着 AMD-All2All 的里程碑**：多项提交突显了 **MI300x8** 在 `amd-all2all` 排行榜上的性能提升，耗时从 **1348 µs** 到 **2.93 ms** 不等。
   - 一位用户以 **1348 µs** 获得第 6 名，另一位用户创下了 **2.03 ms** 的个人最佳成绩。
- **Profiling 问题困扰性能探测**：一位成员在提交自定义 kernel 进行 profiling 时遇到了 `TypeError`，指出在尝试使用命令 `leaderboard submit profile script: gpu:MI300x8 leaderboard_name:amd-all2all` 时，`generate_input()` 中缺少 `rank` 参数。
   - 一位版主澄清说，*profiling 尚未对竞赛开放*，尽管基础设施已经就绪。
- **HIP/ASM 助力获取更高性能**：一位成员询问切换到 **HIP** 是否会带来更好的性能，尤其是在观看了一段视频之后。
   - 另一位成员确认 **HIP/ASM** 通常能获得更好的性能。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1417416747178266665)** (18 条消息🔥): 

> `Lua changes, Frontier model sweeps, Claude sicko mode, Error in GetEntities, Stray log line fixed` 


- **Lua 领域更新落地！**：一位成员确认了最近对 **Lua** 脚本的更改，涉及局部函数、模块/表、新的管理工具以及删除未使用代码。
   - 除了 `can_place_entity` 之外，这些更改不包含功能性变化。
- **OpenRouter Key 开启 Frontier 模型扫参盛宴！**：一位成员通宵对 frontier 模型进行了扫参（sweep），添加了 **4 个个人 OpenRouter key** 以及另一位成员的 key。
   - 扫参消耗了其账户中的 **$100**，这仅占总扫参量的一小部分。
- **Claude 在实验室测试中完胜竞争对手！**：**Claude** 在开放测试中表现出卓越性能，达到其他模型性能的两倍，且仍有提升空间。
   - 该性能是在仅经过 **2-3 次尝试**后评估的，这表明 Claude 在 **@8** 的通过率可能更高。
- **GetEntities 故障导致游戏卡顿！**：有报告称 **GetEntities** 出现错误（*Could not get entities, Error when writing to file: No space left on device*），原因是日志记录过多。
   - 解决方案包括截断日志或实现禁用日志记录的选项。
- **多余日志行被清除！**：一位成员识别并修复了 serialize 中导致过度日志记录的多余日志行，修复补丁已直接推送到 main 分支。
   - 这解决了由于不受控的日志记录导致磁盘空间填满的问题。


  

---

### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1417261480784691280)** (19 条消息🔥): 

> `A2A kernel 规则，Dispatch 和 Combine Kernel，节点内通信，GEMM + RS Kernel 规则，模拟 MOE 和 Combine Kernel` 


- **关于 A2A Kernel 中模拟计算的澄清**：比赛的第一题侧重于**快速通信**而非实现 grouped_gemm，强调 **MOE 和 combine kernel 是模拟的**。
   - 由于*第一题实际上没有进行有意义的计算*，其设计目的是集中精力优化 **dispatch 和 combine kernel**，而不涉及 grouped_gemm 的复杂性，这也是后续题目包含计算的原因。
- **比赛完整规则公布**：组织者表示，完整规则将被整理并置顶以便查阅，建议参与者参考 **AMD 公开文档**和**置顶消息**获取即时信息。
   - 其目标是提供一份清晰且全面的指南，解决因细节散落在不同消息中而可能被忽略的问题。
- **A2A Dispatch Kernel 与通信优化**：允许参与者优化 dispatch kernel 中的 **sort 和 count** 操作，但 kernel 仍必须分发 token，禁止仅输出计数而不产生排序中间结果的方法。
   - 目的是确保方案遵循核心 dispatch 逻辑，同时探索在操作的不同阶段进行优化的机会。
- **A2A Kernel 的规则与介绍**：规则要求实现必须包含节点内通信以及 dispatch 和 combine kernel，强调了分析 reference.py 中 dispatch 和 combine 逻辑的重要性，并理解 task.yml 中定义的各种 shape 如何通过 reference_kernel。
   - 目的是鼓励集思广益，加速 dispatch combine 以及整个 reference_kernel。
- **GEMM + RS Kernel：计算与通信动态**：对于 GEMM + RS kernel，方案必须进行节点内通信以获取 ReduceScatter 操作的数据，并可以探索优化或融合 ReduceScatter 与 GEMM 操作的方法。
   - 由于*这是一个计算+通信 kernel*，其 kernel 逻辑需要详细分析。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/)** (1 条消息): 

drazi1983: 欢迎。感谢提问。图表画得很棒！
  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1417244312273551491)** (27 条消息🔥): 

> `picograd 进展，sitp 更新，rust mdbook 中的 jupyter notebook，异构编程，CUDA vs HIP` 


- **异构编程领域的 SICP 诞生**：**sitp 项目**旨在成为**异构编程**时代的 *SICP*，为每个概念提供 tikz 图表、解释、伪代码和 rust 实现。
   - 已更新至 [https://github.com/j4orz/sitp](https://github.com/j4orz/sitp) 和 [https://j4orz.ai/sitp/](https://j4orz.ai/sitp/)。
- **嵌入式 Jupyter Notebook 面临繁琐的准备工作 (Yak Shaving)**：一位成员正尝试在 **rust mdbook** 中嵌入 **jupyter notebook** 以显示 **torch 代码**，并将 `import torch` 替换为 `import picotorch`。
   - 由于 **JupyterHub**、**k8s** 和 **Slurm** 的复杂性，他们计划让 mdbook 仅在 **CPU** 上执行 **pytorch** 示例代码，并将设备实现的 **HIP 代码**作为静态文本提供。
- **举办“从零开始构建 Pytorch”竞赛？**：一位成员建议发起一个“从零开始构建 pytorch”的**社区项目**，配备正确性测试和排行榜，以造福 **MLSYS** 社区。
   - 另一位成员指出，设计竞赛非常耗时，需要使用 **Modal** 和带有 runner 的 **GitHub Actions** 来创建一个类似 `nano-gpt` 的竞赛。
- **AMD 和 HIP 抢了 CUDA 的风头**：**sitp 项目**使用 **AMD** 和 **HIP**，因为 **RDNA** 不像 **PTX** 那样是虚拟汇编。
   - 该项目正在寻求技术支持，并可以提供 **modal 额度**作为回报。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1417245345817301064)** (1 条消息): 

> `BioML trimul kernel competition, GPUMODE swag` 


- **BioML Trimul Kernel 竞赛截止日期临近！**：在 GPUMODE 上参加 [BioML trimul kernel 竞赛](https://www.gpumode.com/v2/leaderboard/496?tab=rankings) 仅剩 **14 天**。
   - 奖品是由竞赛组织者设计并寄送的“从未见过的 Swag（周边）”。
- **GPUMODE Swag 奖品公布**：BioML Trimul Kernel 竞赛的奖品将是“从未见过的 Swag”。
   - 这些 Swag 将由竞赛组织者亲自设计和寄送，增添了个人色彩。


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1417223621847416962)** (1 条消息): 

> `Mobicham's LLM work, DiT, LLM Training, Quartet` 


- **Mobicham 推进 LLM 进展**：该频道主要展示了 **Mobicham** 在 **Large Language Models (LLMs)** 方面的持续工作和进展。
   - 讨论包括与 **LLM 训练**和优化相关的各种实验、方法论和结果。
- **深入探讨 DiT**：部分频道参与者对 **Diffusion Image Transformer (DiT)** 表现出兴趣，重点关注实现细节和潜在应用。
   - 他们正在探索 DiT 模型如何增强图像生成任务，以及它们与其他 AI 技术的集成。
- **Quartet 训练受到关注**：频道成员还讨论了用于 LLM 的 **Quartet** 训练方法，强调了其优势和挑战。
   - 这包括分享关于针对特定任务和数据集优化 **Quartet** 以提高模型性能的见解。


  

---


### **GPU MODE ▷ #[irl-accel-hackathon](https://discord.com/channels/1189498204333543425/1416087968933740688/1417569864113066015)** (1 条消息): 

> `Low-Bit-Training for Video Models, GitHub Project for Video Model Training` 


- **视频模型 Low-Bit 训练项目启动**：一名成员正在 [GitHub 上发起一个项目](https://github.com/username/video-model-repo)，以探索视频模型的 **low-bit-training**。
   - 他们目前正在确定问题范围并进行调研，以完善项目的方向。
- **邀请社区为视频模型项目做贡献**：该项目旨在解决专门为视频模型定制的 **low-bit training** 挑战，并邀请社区参与。
   - 发起人正在寻求协作和反馈，特别是在定义项目范围和目标方面，并计划通过调研来收集见解。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1417248355947511990)** (97 条消息 🔥🔥): 

> `Mercor, SWEBench, Cursor's Bugbot, OpenCode Zen, Gamma 3.0` 


- **Mercor 的 GMV 受到质疑**：一名成员链接到一条讨论 **Mercor** 数据的 [推文](https://x.com/BrendanFoody/status/1967635147274207376)，另一名成员指出这些数据代表的是人力资源机构的 GMV，而非典型的 SaaS ARR。
   - 尽管有此区别，其增长仍被认为“非常令人印象深刻”。
- **SWEBench 批评**：一场关于 [SWEBench](https://x.com/brhydon/status/1953648884309536958/photo/1) 的讨论展开，有观点认为它过于狭隘、被过度炒作，且侧重于琐碎的 **Django** 修复，而非真正的软件工程技能。
   - 论点认为，高分往往反映了对仓库的记忆，而真正的 SWE 工作涉及诊断和范围界定。
- **Cursor 的 Bugbot 达到 1000 万美元 ARR**：**Cursor 的 Bugbot** 在发布首月就凭借 2 人的团队实现了 **1000 万美元 ARR**。
   - 一名成员表示由于过去的定价问题对其失去了好感，尽管承认其技术价值，特别是他们新的 RL 工作。
- **OpenCode Zen 进入编程 LLM 领域**：**OpenCode Zen** 发布，提供拥有最新模型的编程 LLM，通过 **Vertex** 提供预置容量，并以仅收 **Stripe** 手续费的价格提供 **GPT-5** 透传。
   - 它的目标是成为 **OpenRouter** 的替代品，付费计划不保留数据，且不设利润率。
- **Gamma 3.0 更新演示文稿玩法**：**Gamma 3.0** 发布，其特点是新的 **Gamma Agent**，允许用户通过单个提示词编辑整个幻灯片组，以及 **Gamma API**，支持通过 **Zapier** 工作流从会议记录自动生成幻灯片。
   - 此次发布包括新的 **Team**、**Business** 和 **Ultra** 计划，目标是让幻灯片制作更快速、更易上手。


  

---

### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1417410850431438888)** (9 条消息🔥): 

> `Nano Banana 提示词，字节跳动视频模型，HeyGen 品牌重塑，Video Agent 公测版，Alisa 收购` 


- ****Nano Banana** 提示词将网站变为 **90 年代 CD-ROM** 包装盒**: [Levelsio](https://x.com/levelsio/status/1967593100676943892?s=46) 分享了一个有趣的 **Nano Banana 提示词**，可将任何网站转换为 **1995 年风格的 CD-ROM** 产品包装盒，引发了怀旧样机热潮。
- ****字节跳动** 团队发布高质量视频模型**: **字节跳动** 团队发布了一个新模型，可生成*高质量、文本对齐且主体一致的视频*，现已提供 [ComfyUI 支持](https://x.com/joshua_xu_/status/1967951859500437855)。
- ****HeyGen** 品牌重塑并发布 **Video Agent 公测版****: **HeyGen** 联合创始人 [Joshua Xu](https://x.com/joshua_xu_/status/1967951859500437855) 宣布进行品牌重塑，将 **HeyGen** 定位为*创意操作系统*，并推出了 **Video Agent 公测版**，可将提示词转化为可发布的视频。
- ****HeyGen** 收购 **Alisa** 以领导 **Video Agent** 产品**: **HeyGen** 收购了智能多媒体 Agent 初创公司 **Alisa**，其创始人 **Bin Liu** 现正领导 **HeyGen** 的 **Video Agent** 产品。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1417233016128012451)** (72 条消息🔥🔥): 

> `Abliterated 模型与审查，LM Studio 版本混淆，模型生成速度，LM Studio 上的 Qwen3-Next-80B，VRAM 经验法则澄清` 


- **Abliterated 模型仍然会拒绝！**: 成员们讨论了 **abliterated 模型** 如何通过移除权重来防止负面响应，但它们并没有获得在训练数据之外生成有意义内容的能力。
   - 一位成员补充道，*训练数据仍然没有经过太多清洗*，因此模型在避免毒性响应方面仍会面临困难。
- **LM Studio 版本号令用户困惑**: 一位用户感到困惑，为什么 LM Studio 安装的是 **0.3.26** 版本而不是 **0.3.9**，认为后者才是最新的。
   - 其他用户迅速澄清，26 实际上*比 9 大*，这让原帖作者感到非常尴尬（facepalm moment）。
- **Dolphin 模型的 EOS Token 生成速度受到质疑**: 一位用户询问了 **mradermacher : Dolphin 2.7 Mixtral 8x7B GGUF Q4_K_M 模型**的生成信息——特别是 **EOS Token Found** 消息。
   - 其他人解释说这是正常的输出消息，且该用户的 **4.02 tok/sec** 生成速度*按 AI 标准来看相当慢*，但取决于硬件情况可能也还行。
- **Qwen3-Next-80B 获得 MLX 支持！**: 用户分享了一个 **Qwen3-Next-80B-A3B-Instruct-MLX-4bit** 模型的 [Hugging Face 链接](https://huggingface.co/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-MLX-4bit)，并澄清由于 llama.cpp 不支持 qwen3next，因此不支持 GGUF。
   - 另一位成员链接了一个展示该模型的 [YouTube 视频](https://www.youtube.com/watch?v=ux9S_-QX7TE)，但提醒说使用 transformers 时速度*极慢*，建议在架构成熟前保持耐心。
- **高级用户对 VRAM 经验法则展开辩论**: 一位成员表示，*模型的文件大小必须小于你可用的 VRAM 容量*。
   - 另一位成员反驳道，他们可以在 24GB VRAM 上很好地运行 `Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-UD-Q8_K_XL.gguf`（35GB），因为这是一个 **Mixture of Experts** 模型，并非所有权重都会同时激活。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1417256745113489408)** (24 messages🔥): 

> `使用 Nextcloud 构建个人云，用于云游戏的 VPN Meshnet，设置 Qdrant 向量数据库，Ryzen AI MAX 395 在 qwen3-coder-30b 上的性能，MacOS Sequoia 运行 70B 模型的内存占用` 


- ****Nextcloud 网络新手****：一位初学者详细介绍了他们的网络项目，包括设置 **Nextcloud 个人云**（但因 ISP 问题暂停），以及配置用于云游戏和 AI 使用的 **VPN meshnet**。
   - 该用户在保存了一个稳定的 SBC 配置后，现在正着手为 AI 设置 **Qdrant 向量数据库**。
- ****Ryzen AI MAX 395 性能探索****：一位用户询问了 **Ryzen AI MAX 395+** 运行 **qwen3-coder-30b Q8 和 bf16** 的性能，特别是在纠结应该等待下一代硬件还是构建一个 AMD Epyc 9005 系统。
   - 另一位用户正在等待他们的机器保修归来以提供性能数据，同时还有人分享了一个与 **AMD Strix Halo toolboxes** 相关的 [GitHub 链接](https://github.com/kyuz0/amd-strix-halo-toolboxes)。
- ****MacOS Sequoia 的内存挑战****：一位在 **MacOS Sequoia** 上使用极小内存加载 Q8 格式 70B 模型的用户，因担心内存占用增加而犹豫是否升级到 Tahoe。
   - 另一位用户建议*仅在必要时更新*，并建议*等待 bug 修复*，理由是该操作系统仍处于早期阶段。
- ****CachyOS 与 LLM Offloading 对决****：一位用户正在安装 **CachyOS**，并质疑为什么在运行 LLM 时要使用 hypervisor，认为这会对 **MoE offload 性能**产生负面影响。
   - 另一位用户反驳称，hypervisor 不一定会影响性能，而且允许运行其他应用程序，并指出在大型系统上其开销极小。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1417302315785785414)** (83 messages🔥🔥): 

> `用于 Agent 编码的 XML，MI50 GPU，AMD RDNA5，用于编码的 Codex vs Claude，LLM Router 与小模型霸权` 


- **XML 简化了 Agent 编码！**：成员们正在研究 **XML**，因为它使模型更容易进行 **Agent 编码**。
- **eBay 上的 MI50 诱惑着 GPU 爱好者**：一位成员很想把一些 **MI50** 装进租约到期的 Xeon 服务器中，并指出目前美国 eBay 上的价格非常便宜。
- **AMD RDNA5 因财务现实而推迟**：一位成员提到，等他们攒够钱买新的 **AMD RDNA5** 时，可能直接买一台带有一些高性能 **AMD AI 卡**的微型服务器了。
- **Codex 编码能力受到质疑**：一位成员发现 **Codex** 在编码方面很难用，称 **Claude Code** 表现要好得多。
   - 另一位成员表示，*Codex 自上次使用以来已经进步了很多*，但也补充说，他们*仍然不认为 gpt-5 能与 claude 媲美*，因为它在 **GitHub Copilot** 中经常出错。
- **训练 LLM Router 是未来**：一位成员建议将训练 **LLM Router** 作为实现鲁棒性的另一种角度，将其与 tool calls 结合，并支持小模型霸权。
   - 另一位成员提到，他们最喜欢的 **Tailwind CSS 模型**是 **Tesslate 的 UIGEN T3**，在设计方面完胜 GPT-5。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1417443840888930338)** (2 messages): 

> `AI 男友关系，基于草图的 GNN` 


- **预言 2025 年 AI 男友热潮**：一篇 [研究论文](https://arxiv.org/abs/2509.11391) 分析了来自 r/MyBoyfriendIsAI 的 **1500 条帖子**，发现许多这类关系是从偶然的闲聊中*无意间*开始的。
   - 论文指出，用户将 **prompt-engineering** 发展为一种*爱之语*，但也面临着如 **情感依赖（~9%）** 和 **现实解离（~4%）** 等风险。
- **GNN 大神痴迷于 NLP**：一位成员正在撰写一篇关于利用 **NLP** 推进 **基于草图的 GNN** 的研究论文，重点关注增强语义压缩的高级向量量化技术。
   - 他们正在寻找该领域的专家来评审他们的提案，特别是关于使用独立 NN（可能是标准的语义编码器）的部分。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1417443840888930338)** (2 messages): 

> `AI Boyfriend research, Sketch-based GNNs` 


- **AI “男友”研究揭示复杂的人类影响**：一项研究分析了来自 r/MyBoyfriendIsAI 的 **1.5k** 条帖子，发现许多关系是在无意中开始的，且 Prompt Engineering 变成了一种“爱之语”。
   - 论文 ([My AI is a Boyfriend (2025)](https://arxiv.org/abs/2509.11391)) 报告了益处（≈**25%** 感到孤独感减轻）和风险，如情感依赖（**~9%**）、现实解体（**~4%**）以及逃避人类关系（**~4%**）。
- **Sketchy GNNs 寻求 NLP 助力**：一位成员正在撰写一篇关于利用 **NLP**（主要通过高级向量和乘积量化）改进**基于草图的 GNNs** 的研究论文。
   - 目标是通过独立的 **NN** 增强语义压缩，目前正在寻找合作者来审阅该提案。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1417244881042276424)** (4 messages): 

> `Tau Bench results with fastWorkflow and DSPy, VoxCron tool launch, GEPA for workflow optimization` 


- **DSPy 和 fastWorkflow 在 Tau Bench 上超越 Claude Opus 4.1**：通过在 **fastWorkflow** 中使用 **DSPy** 进行 **Agent** 构建和参数提取，一位用户在 **Tau Bench 开发集**上达到了与 **Claude Opus 4.1** 相当的性能（[结果见此](https://cdn.discordapp.com/attachments/1202371242519441499/1417244881377693777/Tau_Bench_retail_using_fastWorkflow.png?ex=68cb1926&is=68c9c7a6&hm=845f74fb571d7893d54b6fe5b0b2e78b6878c890010338acac37be29f5080ae5&)）。
   - 该成员表示：“通过适当的脚手架（scaffolding），你确实可以击败大模型！”，并建议查看 [零售工作流示例](https://github.com/radiantlogicinc/fastworkflow) 来测试该 **Agent**。
- **VoxCron 自动生成文档和图表**：一位用户发布了 **VoxCron** ([voxcron.com](https://voxcron.com))，这是一个通过自动生成整洁的 Markdown 文档和 **Mermaid** 图表来简化客户需求规范审查的工具。
   - 开发者已为客户构建了一年的 **DSPy 项目**，欢迎大家对该工具的免费层级提供反馈。
- **GEPA 被考虑用于 fastWorkflow 优化**：一位成员计划在 **fastWorkflow** 中使用 **GEPA** 进行端到端的工作流优化。
   - 另一位成员请他们分享使用 **GEPA** 的经验，以及可以改进哪些方面以更好地支持 **Agent** 化用例。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1417273923866857542)** (65 messages🔥🔥): 

> `DSPy use cases, LM inference based on client, Refining topic matching in DSPy, Feeding classifier a list of topics, Arc-AGI leader prompt-optimization` 


- **DSPy 框架在主题分类中得到验证**：用户验证了 **DSPy** 是一个非常有用的主题分类框架，特别是考虑到其优化的潜力，并指出该框架可能比其他框架更合适。
   - 一位用户提到他们正在寻找可以尝试 **DSPy** 的场景，并确认将对其进行测试。
- **基于客户端推断 LM 模型是可行的**：一位成员建议，即使不知道是哪个 **LM** 在调用，也可以根据客户端做出假设（例如，如果是 openai-mcp，则推断为 gpt-5）。
   - 另一位用户提到，根据客户端返回优化后的工具这一想法非常酷。
- **用户寻求通过种子短语实现更好的主题匹配**：一位用户正尝试通过为每个主题提供一组短语来改进主题匹配，并希望以一种可以在 **DSPy** 中进行优化的方式来实现。
   - 他们更倾向于让主题与短语之间的关系对 **DSPy** 优化器来说在语义上是清晰的。
- **在 DSPy 中向分类器输入主题列表**：一位用户在向分类器输入主题列表时遇到困难，注意到它需要一个 **Literal**，随后展示了他们使用 `pydantic` 和 `load_dotenv` 的变通代码。
   - 另一位用户建议，主题列表可以作为另一个输入传递，而不是使用 **Literal**。
- **Prompt 优化帮助用户在 arc-agi 中获得高分**：一位成员分享了一篇文章，介绍了新的 **Arc-AGI** 领先者如何通过测试时的 Prompt 优化达到最高分，参考了[这篇 Substack 文章](https://jeremyberman.substack.com/p/how-i-got-the-highest-score-on-arc-agi-again)。
   - 文章详细介绍了针对 **ARC-AGI** 挑战的 Prompt 优化策略。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1417298215287390228)** (15 messages🔥): 

> `Mojo/Max 与 Python 3.13 兼容性、Apple Metal 支持处于早期阶段、Nightly 版本支持 MI355X、Pixi 包管理器优势` 


- **Mojo 与 Python 的聚会 🥳**：根据 general 频道的一位工作人员透露，Mojo/Max 目前与 **Python 3.13** 兼容。
   - 工作人员还鼓励使用 `pixi` 包管理器，它可以处理隔离的 Python 版本。
- **Apple Metal 支持仍需时间“金属化” 🤘**：**Apple Metal 支持**目前处于早期阶段，因此性能可能低于预期。
   - 仅在 **CPU** 上运行应该没问题，只是速度较慢。
- **MI355X Nightly 导航 🧭**：有人询问包含 **MI355X** 支持的 nightly 版本。
   - 工作人员提到，这些更改位于 **pixi 拉取的 Mojo 编译器下游**。
- **MAX Kernel 与 Mojo 版本保持同步**：**MAX kernel** 几乎不经过解析就存储在 **.mojopkg 文件**中，然后 **MAX** 使用 **Mojo 编译器**在图编译过程中完成剩余的编译工作。
   - 因此，就版本而言，**MAX** 和 **Mojo** 保持步调一致。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1417226174219681863)** (17 messages🔥): 

> `Allocator API、参数化 Trait 和 requires、Mojo LSP 重构、网络更新阻塞项、Mac 上 mojo test 的编译器 Bug` 


- **Allocator API 提案**：成员们讨论了为 Mojo 提供 **allocator API** 的可能性，其中一人表示：“我也在考虑将 allocator API 作为解决此问题的方法”。
   - 另一位成员表示他们的提案还处于初步阶段（Barely），因为他们正在等待**参数化 Trait 和 `requires`**。
- **Mojo 的 LSP 正在进行重大重构**：一位成员询问了 Mojo 的语言服务器协议（**LSP**），问：“Mojo 现在有 LSP 了吗？”
   - 另一位成员确认它已经存在，并且“很快将进行重大重构”。
- **网络更新存在许多阻塞项**：成员们讨论了 Mojo 的**网络更新**，一位成员表示：“我仍在等待网络更新😔”。
   - 另一位成员回复道：“那里有很多阻塞项”，前者确认“是的，我知道，希望我们能尽快完成”。
- **在 Mac 上使用 Mojo Test 发现编译器 Bug**：一位成员报告了一个仅在 **Mac** 上使用 **mojo test** 时出现的潜在**编译器 Bug**。
   - 该成员链接到了一个[论坛帖子](https://forum.modular.com/t/unexpected-argument-mutation-with-mojo-test-not-under-mojo-run/2203)，其中包含更多细节，并征求关于如何进行报告或进一步研究的建议。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1417441918320775178)** (7 messages): 

> `LLM 是贝叶斯的、灾难性遗忘、在线学习 vs 批处理学习` 


- **LLM 是贝叶斯扩展工具包发布！**：一位成员分享了一篇[新的预印本论文和工具包](https://x.com/ahmedkar_/status/1967875794333732947?s=46)，扩展了“**LLM 在期望上是贝叶斯的**”这一论文。
   - 该工具包旨在扩展贝叶斯原理在大型语言模型中的实用性。
- **灾难性遗忘与 LLM 的贝叶斯特性相关联**：一位成员指出，“**LLM 是贝叶斯的**”论文让他们想起了“**灾难性遗忘**”定律，并引用了 [ArXiv 上的一篇论文](https://arxiv.org/pdf/2509.04259)。
   - 该成员建议这两篇论文“可能”讨论了类似的概念，值得进一步调查。
- **讨论在线学习 vs 批处理学习**：一位成员试图区分哪些机器学习方法在面对新数据时需要重新计算，哪些只需要更新，从而区分**在线学习（online learning）**和**批处理学习（batch learning）**。
   - 另一位成员开玩笑地建议“**LLM** 会直接告诉你”，对此原帖作者承认他们“还没有那种自动化思维”。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1417238045840900157)** (8 messages🔥): 

> `黄线平滑度、VaultGemma 发布` 


- **年轻用户产生更平滑的数据**：一位成员询问了某可视化图表中黄色 **18-25** 岁线条的平滑度，认为这可能是由于用户数量较多且噪声较小。
   - 另一位成员指出，老年群体的噪声似乎在增加，还有人将其归因于可用样本数量可能较少，从而增加了方差。
- **VaultGemma 作为私有 LLM 亮相**：一位成员分享了 [VaultGemma 博客文章](https://research.google/blog/vaultgemma-the-worlds-most-capable-differentially-private-llm/)的链接，这是 Google 在**差分隐私 LLM** 领域的最新尝试。
   - 他们还链接了相关论文：[arxiv.org/abs/2509.05276](https://www.arxiv.org/abs/2509.05276)。


  

---

### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1417603870980444293)** (1 条消息): 

> `Anthropic MCT Tools API, LLMs vs ARC Tool Use` 


- **Anthropic MCT Tools API 非常简洁**：一位成员使用了 **Anthropic MCT Tools API** 并表示*它使用起来非常简洁*。
   - 该成员补充道，这*让人联想起以前使用 DSL 包的方式，所有的函数都集中在一个文件里*。
- **LLMs > ARC tools**：一位成员表示惊讶，*对于 LLM 来说，使用工具似乎比在 ARC 中使用 shift 和 translocate 函数更容易*。
   - 该成员没有提供链接或进一步解释 **ARC tools**。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1417236303799521402)** (8 条消息🔥): 

> `AI-powered PDF Editor, Agents to Payments Protocol, Arc Prize results` 


- **Google 炒作 AI 驱动的 PDF 编辑器**：Google 在其新产品的描述中宣传了一个 **AI-powered PDF Editor**，因其讽刺性而引起关注。
   - 该新产品是 [Agents to Payments (AP2) Protocol](https://cloud.google.com/blog/products/ai-machine-learning/announcing-agents-to-payments-ap2-protocol)，旨在利用 AI Agent 简化支付流程。
- **Arc Prize 宣称获得高分**：[Arc Prize](https://fxtwitter.com/arcprize/status/1967998885701538060) 声称在 v3 版本上达到了近 **80%** 的准确率，在 v2 版本上达到了 **30%**。
   - 一位成员指出，这些结果可能是刻意挑选（cherry-picked）的，因为他们并不允许所有人进行结果验证，因此对其作为真实 Benchmark 的合法性表示怀疑。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1417232050569609316)** (9 条消息🔥): 

> `GPT-5 Codex, aider's chat-mode, aider's architect mode, code mode` 


- ****GPT-5 Codex** 的 Aider 分数仍不明确**：一位成员询问了新 **GPT-5 Codex** 模型的 **aider score**，并引用了 [The New Stack 上的一篇文章](https://thenewstack.io/openai-launches-a-new-gpt-5-model-for-its-codex-coding-agent/)。
   - 另一位成员回答说，该模型*尚未通过 API 提供*。
- **聊天模式：`--chat-mode code` 是否已过时？**：一位成员注意到 `--chat-mode code` 选项似乎无效，建议可能需要更新文档。
   - 作为回应，另一位成员澄清说 **默认模式就是 chat mode**，因此不需要任何 flag，使用 `/code` 即可返回 code mode。
- **Architect Mode 正在增强 Prompt 指令！**：一位成员报告说 **architect mode** 正在通过上下文增强他们的 Prompt 指令，并寻求阻止这种情况的方法。
   - 他们表示，原本希望 **code mode** 能够防止这种增强。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1417309023111483463)** (4 条消息): 

> `Gemini issues, Ollama endless loop, Architect Mode` 


- **Gemini Aider 用户遇到问题**：一位用户在将 **aider** 与 **gemini** 配合使用时遇到问题，程序会卡在等待响应的状态，即使 Token 正确且更换不同的 **gemini** 模型，问题依然存在。
   - 另一位用户报告说他们也遇到了同样的问题。
- **Aider architect mode 与 ollama 出现死循环**：一位用户报告在 architect mode 下通过 **ollama** 使用本地 LLM 运行 **aider** 时，会出现死循环：它输出代码，意识到不是完整实现，然后在没有干预的情况下继续工作。
   - 一位成员建议检查 **ollama** 中的 **context length**（上下文长度）或移除 `--yes-always` 标志，因为这可能是导致死循环的原因。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1417239737831395431)** (9 messages🔥): 

> `Simplicity vs Elegance, MI350 Kernel Benchmarks, MLIR-based compiler` 


- **软件设计中的简洁性 vs 优雅性**：讨论围绕 Chris Lattner 关于“复杂度是敌人”的观点展开，一名成员回应称，在设计中**简洁性（Simplicity）**是错误的衡量标准，对于库和 API 来说，**优雅性（Elegance）**和**易用性（Usability）**是更好的指标。
   - 他们进一步解释说，组件应该像**拼图碎片一样契合**，以避免与抽象层作斗争，并指出软件本质上就是复杂的。
- **tinygrad 将发布 MI350 Kernel 基准测试**：tinygrad 计划在年底前发布 **MI350 Kernel 基准测试**，目标是单 Kernel 性能达到 NVIDIA 的 **80%**，并旨在让 tinygrad 在整体任务运行上更快。
   - 目标**不是追求绝对的最快**，而是专注于整体效率。
- **追求基于 MLIR 的编译器**：一名成员表示计划尝试创建一个**基于 MLIR 的编译器**，另一名成员建议他们考虑之前的观点。
   - 双方都承认，这种方法并不能解决大多数重大问题。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1417226698868396184)** (6 messages): 

> `Knowledge Limit, Credit Rollover, AI Loop & Refund` 


- **知识限制提升：如何实现？**：一名成员询问是否有可能超过 **20** 的知识限制以及如何实现。
   - 该问题未包含回复。
- **额度（Credit）与 Token 结转**：一名成员询问未使用的 **Tokens** 和 **Credits** 是否在续订时结转。
   - 该问题未提供答案。
- **AI 进入死循环：退款请求**：一名成员报告称 **AI** 进入了极长的死循环并消耗了所有 **3000 Credits**，并请求退款。
   - 另一名成员表示，自 **9 月 4 日**以来，他们也遇到了同样的情况。


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1417317929523085393)** (1 messages): 

> `golang streaming http MCP server, Scalability of MCP server, Auth in MCP server, Sessions and resumability in MCP server, Dynamic capabilities in MCP server` 


- **新 Golang 流式 HTTP MCP 服务器开源**：一名成员宣布开源其 [golang streaming http MCP server](https://github.com/ggoodman/mcp-server-go) 项目，该项目专为挑战性的类企业级需求而设计。
   - 该产品包括**可扩展性（Scalability）**、**认证（Auth）**、**会话（Sessions）**、**可恢复性（Resumability）**和**动态能力（Dynamic Capabilities）**等特性。
- **MCP 服务器扩展方案揭晓**：新发布的 [MCP server](https://github.com/ggoodman/mcp-server-go) 采用可插拔后端接口设计以实现**可扩展性**，包括内存和 Redis streams 后端。
   - 这一设计选择旨在支持在苛刻的企业环境中进行水平扩展。
- **MCP 服务器集成 OIDC 和 JWT 认证**：[MCP server](https://github.com/ggoodman/mcp-server-go) 可以从简单的发行者 URL 引导当前的 authz 规范，假设使用 **OIDC discovery** 和 **JWT access tokens**。
   - 对于更细致的设置，提供手动配置，为身份验证策略提供灵活性。
- **会话与可恢复性**：**Sessions** 和 **resumability** 基于可插拔后端构建，让你可以直接使用协议中这些困难的部分，而无需费力开发。
   - 这些功能旨在简化 MCP 服务器中会话管理和可恢复性的实现。
- **动态能力简化资源管理**：[MCP server](https://github.com/ggoodman/mcp-server-go) 采用动态设置来处理来自数据库或 API 的工具、资源和提示词（Prompts）。
   - 它还为静态设置提供容器，满足各种部署需求。