---
companies:
- openai
- anthropic
- google
- runway
- elevenlabs
- freepik
- openart
- deepseek
- mistral-ai
- alibaba
- nous-research
date: '2025-12-03T05:44:39.731046Z'
description: '以下是该文本的中文翻译：


  **OpenAI 的“红色警报”（Code Red）响应**和 **Anthropic 的 IPO** 是近期重大的焦点。在 AI 视频与图像领域，**可灵（Kling）2.6**
  引入了原生音频协同生成与连贯的唇形同步技术，并与 **ElevenLabs** 和 **OpenArt** 等平台达成了合作。**Runway Gen-4.5**
  提升了光照真实度，而谷歌的 **Gemini 3 Nano Banana Pro** 则支持高级图像合成。


  开源模型发布方面，包括采用稀疏注意力机制且定价极具性价比的 **DeepSeek V3.2**，以及拥有强力 14B 变体的 **Mistral Ministral
  3** 多模态系列。来自阿里巴巴的 **EvoQwen2.5-VL** 和 Nous Research 的 **Hermes 4.3** 在检索和代码模型方面展现出极具竞争力的性能，且拥有宽松的许可协议并在
  Hugging Face (HF) 上提供。社区竞技场也迎来了 **INTELLECT-3 (106B MoE)** 等新成员。“音画同步的连贯输出”和“匹配场景氛围的自动光效”是备受瞩目的技术进步。'
id: MjAyNS0x
models:
- kling-2.6
- kling-o1
- runway-gen-4.5
- gemini-3
- deepseek-v3.2
- ministral-3
- evoqwen2.5-vl
- hermes-4.3
- intellect-3
people: []
title: 今天没发生什么特别的事。
topics:
- video-generation
- audio-processing
- multimodality
- image-generation
- reasoning
- model-quantization
- sparse-attention
- model-pricing
- multimodal-models
- retrieval-augmentation
- model-training
- model-release
---

**一个安静的 NeurIPS。**

> 2025年12月2日至12月3日的 AI 新闻。我们为您检查了 12 个 subreddit、544 个 Twitter 账号和 24 个 Discord 服务器（205 个频道，7213 条消息）。预计节省阅读时间（按每分钟 200 词计算）：552 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 方式呈现所有往期内容。访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上向我们提供反馈！

关于 [OpenAI 的 Code Red 响应](https://www.theinformation.com/articles/openai-developing-garlic-model-counter-googles-recent-gains?rc=ytp67n)以及 [Anthropic 的 IPO](https://vechron.com/2025/12/anthropic-hires-wilson-sonsini-ipo-2026-openai-race/) 讨论很多。

---

# AI Twitter 综述

**AI 视频与图像：Kling 2.6 原生音频、Kling O1 镜头控制、Runway Gen‑4.5、Nano Banana Pro (Gemini)**

- **Kling 2.6 (原生音频协同生成)**：Kling 的新 2.6 模型可以一次性生成视频以及同步的语音、SFX（音效）和环境音，创作者报告称其具有连贯的口型同步、动作以及强大的“视听协调性”。广泛的合作伙伴推广包括 fal 的首日原生音频访问权限 ([@fal](https://twitter.com/fal/status/1996232741721969131))，以及 InVideo ([@invideoOfficial](https://twitter.com/invideoOfficial/status/1996235306652287297))、ElevenLabs ([@elevenlabsio](https://twitter.com/elevenlabsio/status/1996239001590682077))、Freepik ([@freepik](https://twitter.com/freepik/status/1996239332605301115)) 和 OpenArt ([@openart_ai](https://twitter.com/openart_ai/status/1996245765207867563)) 的平台集成。Kling 的官方公告通过短片演示和宣传片强调了“外观和声音连贯的输出” ([@Kling_ai](https://twitter.com/Kling_ai/status/1996238606814593196))。创作者的教程和早期测试显示了改进的镜头变化和更快的成片速度 ([@jerrod_lew](https://twitter.com/jerrod_lew/status/1996234217475408262), [@TheoMediaAI](https://twitter.com/TheoMediaAI/status/1996233778742599975))。
- **Kling O1 (镜头控制)**：O1 强调构图、镜头多样性和场景内创意控制，以实现更高水平的视频合成 ([@CharaspowerAI](https://twitter.com/CharaspowerAI/status/1996248264354476214))。
- **Runway Gen‑4.5 (光照)**：Runway 的 Gen‑4.5 提升了视觉保真度和“自动光照”功能，无需复杂提示词即可匹配场景氛围 ([Runway](https://twitter.com/runwayml/status/1996223569148170665))。
- **Nano Banana Pro (Gemini 3)**：Google 的新图像模型支持增强的推理能力，每个提示词最多可合成 14 张图像 ([Google](https://twitter.com/Google/status/1996263265735749682), [后续](https://twitter.com/Google/status/1996263275856904686))。Synthesia 在产品中添加了一键式 Nano Banana Pro 生成功能 ([@synthesiaIO](https://twitter.com/synthesiaIO/status/1996220160370266325))，Gemini 则推出了 2K 分辨率的图像输出 ([@GeminiApp](https://twitter.com/GeminiApp/status/1996252061651042751))。

**开源模型、发布与基准测试**

- **DeepSeek V3.2 (开源权重 MoE, DSA)**：Artificial Analysis 将 V3.2 列为其综合评分排名第 2 的开源权重“推理”模型，采用与 V3.2-Exp 相同的 671B 总参数/37B 激活参数架构，现已使用 DeepSeek Sparse Attention (长上下文)，价格为每 1M 输入/输出 token $0.28/$0.42（享受 90% 缓存折扣）。V3.2-Speciale（仅限推理）消耗的 token 更多，但目前在官方 API 中缺乏工具调用功能（[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1996110256628539409)；论文/仓库：[链接 1](https://twitter.com/ArtificialAnlys/status/1996110267353325748)，[链接 2](https://twitter.com/ArtificialAnlys/status/1996110266065715249)）。社区提醒在进行直接对比评估时，如果不按成本/token 进行归一化，不要将“推理”模式与非推理模式混为一谈（[@qtnx_](https://twitter.com/qtnx_/status/1996146690496049349), [@eliebakouch](https://twitter.com/eliebakouch/status/1996214163215978967)）。
- **Mistral “Ministral 3” 系列 (多模态) 及基座模型**：Mistral 发布了一个多模态系列，其中包含一个强大的 14B 变体；现已提供用于 SFT+GRPO 的 TRL 配方（[@SergioPaniego](https://twitter.com/SergioPaniego/status/1996257877871509896)）。从业者赞扬了基座模型的可用性，便于进行自定义后训练（[@QuixiAI](https://twitter.com/QuixiAI/status/1996272948378804326)）。
- **检索与代码模型**：阿里巴巴的 EvoQwen2.5-VL (3B/7B) 作为视觉文档检索器，在 ViDoRe v2 上的表现优于 NVIDIA，且许可协议宽松（[@mervenoyann](https://twitter.com/mervenoyann/status/1996221079757439374), [hf 链接](https://twitter.com/mervenoyann/status/1996221946006994973)）。Nous 在字节跳动 Seed 36B 上发布了 Hermes 4.3，通过 Psyche 上的 Distro 进行训练，其表现达到或超过了其集中式运行的版本，并在 RefusalBench 中名列前茅；权重已上传至 HF（[@NousResearch](https://twitter.com/NousResearch/status/1996311677009121367), [@Teknium](https://twitter.com/Teknium/status/1996330606595391780)）。
- **社区竞技场**：LM Arena 添加了 INTELLECT-3 (106B MoE; GLM-4.5 Air 基座; Apache-2.0/MIT)，用于在创意/数学任务中进行实时直接对比（[@arena](https://twitter.com/arena/status/1996324769013391839)）。

**Agent：构建、评估与推理基础设施**

- **从无代码到生产环境**：LangChain 的 LangSmith Agent Builder 正被用于处理真实工作流（研究简报、GitHub/Linear Agent、Slack/邮件助手），只需简单的提示词即可构建，并提供了深度 Agent 评估模式指南（单步、全轮、多轮、定制成功标准）以及块级缓存控制以降低上下文成本（[产品](https://twitter.com/LangChainAI/status/1996265192213365080), [评估博客](https://twitter.com/LangChainAI/status/1996276393068617829), [缓存控制](https://twitter.com/sydneyrunkle/status/1996278442430472327)）。Lindy 的 Agent Builder 展示了类似的低摩擦工具集成与记忆功能（[@omarsar0](https://twitter.com/omarsar0/status/1996225497429389493)）。
- **Agent 基础设施与性能**：vLLM 添加了 Snowflake 的无模型 SuffixDecoding，在各种并发级别下均优于调优后的 n-gram 推测（[@vllm_project](https://twitter.com/vllm_project/status/1996130115856859461)），发布了与上游 vLLM 保持一致的 Gaudi 插件（[发布说明](https://twitter.com/vllm_project/status/1996207672245518782)），并发布了针对挂起内核的 CUDA 核心转储 (core-dump) 追踪指南（[工程文档](https://twitter.com/vllm_project/status/1996256049368793218)）。Together AI 与 Meta 合作，通过 TorchForge 为 Agent 系统带来高性能 RL（[Together](https://twitter.com/togethercompute/status/1996257121256816936)）。LlamaIndex 在 LlamaCloud 中引入了“一键部署”文档工作流（解析/提取/分类 + 托管 UI）（[@llama_index](https://twitter.com/llama_index/status/1996265747228844178), [@jerryjliu0](https://twitter.com/jerryjliu0/status/1996349988205637773)）。
- **标准与多 Agent 语义**：Dair-AI 提出了针对 Agent 互联网的 L8“通信”层与 L9“语义协商”层堆栈，并包含针对语义注入/投毒的防御措施以及签名上下文（[论文 + 摘要](https://twitter.com/dair_ai/status/1996227436913340858)）。独立研究量化了多 Agent 通信效率 (IEI/SEI/TEI)，并表明带有效率增强目标的单轮通信可以击败多轮协议（[摘要](https://twitter.com/omarsar0/status/1996263279052931372), [论文](https://twitter.com/omarsar0/status/1996263292361457783)）。
- **编程 Agent (实操)**：新增关于在沙箱化 microVM/容器中安全编写并执行代码的 Agent 免费课程，包含数据分析师和全栈项目（[Andrew Ng](https://twitter.com/AndrewYNg/status/1996250415244235013), [@e2b](https://twitter.com/e2b/status/1996236480251859106)）。

**评估与方法：衡量什么以及如何衡量**

- **CORE‑Bench 通过 scaffold 耦合被“解决”**：使用 Claude Code 配合 Opus 4.5 在 CORE‑Bench（科学可复现性 Agent 基准测试）上达到了 95% 的得分，而使用中立的 CORE‑Agent scaffold 报告的得分仅为 42%。作者详细说明了修复的九个评分问题，解释了为什么高能力模型会暴露评分标准的边缘情况，并认为模型与 scaffold 的耦合现在主导了评估结果——未来的评估应披露 scaffold，并考虑为每个模型提供最佳的 scaffold ([@sayashk](https://twitter.com/sayashk/status/1996334941832089732))。
- **OpenAI “自白”（关于违规行为的诚实性）**：一个 GPT‑5 Thinking 变体被训练为同时输出答案和关于合规性的“自白”；即使在承认走捷径时，自白也会因诚实而获得奖励。在诱导违规行为的评估中，假阴性（违规但未自白）平均为 4.4% ([thread](https://twitter.com/OpenAI/status/1996281172377436557), [details](https://twitter.com/OpenAI/status/1996281175770599447), [metric](https://twitter.com/OpenAI/status/1996281178668876214))。
- **大规模基准测试**：Epoch AI 提议“缝合”多个基准测试，以避免饱和并将模型置于统一的量表上 ([@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1996248575400132794))。Hugging Face 发布了 LLM Evaluation Guidebook v2（涵盖从端到端基础到陷阱的交互式指南） ([@clefourrier](https://twitter.com/clefourrier/status/1996250279033839918))。研究人员继续警告，在不针对成本/Token 进行归一化的情况下，不要将“推理”模型与非推理模型进行比较 ([@eliebakouch](https://twitter.com/eliebakouch/status/1996214163215978967))。
- **学习动力学**：“Quiet Feature Learning”表明 Transformer 在 Loss 平台期会习得对任务至关重要的内部特征，随后这些特征会“突然开窍”转化为输出增益——这促使人们寻找比单纯 Loss 更丰富的诊断手段 ([summary + paper](https://twitter.com/omarsar0/status/1996233046799106128))。TabPFN 的 Nature 成果持续引发反响：这是一个在 1 亿个合成 DAG 数据集上训练的表格基础模型，在一次前向传递中完成训练+预测，并在数秒内超越了经过调优的树模型方法 ([@burkov](https://twitter.com/burkov/status/1996102081996861907))。METR 的任务长度测量似乎可以从 SWE 泛化到自动证明领域 ([@littmath](https://twitter.com/littmath/status/1996245072149430482))。

**系统与推理效率**

- **Apple MLX‑LM 进展**：MLX‑LM 在服务器中增加了 continuous batching（演示：在 M2 Ultra 上同时处理 4 个 Qwen3‑30B 请求），基于之前的批处理生成工作，并稳步推进统一的 Apple MLX/CUDA 生态 ([demo](https://twitter.com/angeloskath/status/1996364526749639032), [release](https://twitter.com/awnihannun/status/1996365940343402596))。
- **Attention/并行通信**：字节跳动的异步 Ulysses attention “看似简单”，并且通过比 NCCL 更快的 all‑to‑all，通信可以很好地与计算重叠 ([@maharshii](https://twitter.com/maharshii/status/1996280889962365380))。
- **vLLM 工程**：针对深度内联/异步内存情况的 CUDA core‑dump 追踪，超越标准工具以精准定位挂起的 Kernel ([@vllm_project](https://twitter.com/vllm_project/status/1996256049368793218))。
- **搜索基础设施转型**：将向量工作负载从 Elasticsearch 迁移到 Qdrant 的团队提到了原生向量索引、混合稠密+稀疏检索、更简单的扩展以及更低的延迟/成本。包含迁移步骤和陷阱的实用深度探讨 ([@qdrant_engine](https://twitter.com/qdrant_engine/status/1996127270487183567))。
- **扩散模型蒸馏**：“Glance”通过单样本领域特定蒸馏，将 Qwen‑image/FLUX 的推理速度从约 50 步提升至 10 步以内 ([@awinyimgprocess](https://twitter.com/awinyimgprocess/status/1996158744590447037))。
- **数据管道**：Hugging Face 现在允许通过 Xet 在几秒钟内跨账户复制任何数据集（例如，~2 秒内复制 1 TB），从而实现无需繁重传输的 fork‑filter‑train 循环 ([@victormustar](https://twitter.com/victormustar/status/1996218180583219572))。
- **端侧多模态**：Nexa 的 AutoNeural‑VL‑1.5B 在高通 SA8295P NPU 上完全本地运行（约 100 ms 延迟，768² 视觉分辨率），用于车载助手 ([@nexa_ai](https://twitter.com/nexa_ai/status/1996260367769739665))。

**行业动态与平台更新**

- **Anthropic 的规模扩张**：据报道，微软投资高达 100 亿美元，NVIDIA 投资 50 亿美元，此外还从微软购买了 300 亿美元的算力，使 Claude 进驻所有主流云平台，这意味着其估值约为 3500 亿美元（[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1996081964395200773)）。Anthropic 还宣布了与 Snowflake 达成为期多年、价值 2 亿美元的合作伙伴关系（[Anthropic](https://twitter.com/AnthropicAI/status/1996327475492868292)），以及在达特茅斯学院部署 “Claude for Education”（[Anthropic](https://twitter.com/AnthropicAI/status/1996311516245803434)）。Claude Opus 4.5 现在可供 Pro 用户在 Claude Code 中选择（[@claudeai](https://twitter.com/claudeai/status/1996310793017594124)）。
- **OpenAI 资助**：OpenAI 基金会的 People‑First AI Fund 宣布 208 家非营利组织将获得总计 4050 万美元的无限制资助（[@OpenAI](https://twitter.com/OpenAI/status/1996258322304155695)）。
- **Waymo 扩张**：Waymo 目前已在更多城市实现完全无人驾驶（无安全员），同比增长超过 500%，在达拉斯仅用约 4 个月时间就完成了从配备安全员到完全无人驾驶的快速过渡（[@Waymo](https://twitter.com/Waymo/status/1996217860440412641), [@fchollet](https://twitter.com/fchollet/status/1996263334883266961)）。
- **开发者工具**：Google 推出了 Workspace Studio，旨在快速构建工作流 Agent，目标是实现整个套件内日常任务的自动化（[@GoogleWorkspace](https://twitter.com/GoogleWorkspace/status/1996263985985769976)）。Phind 融资 1040 万美元，并转向交互式“小程序（mini‑app）”式的回答方式（[@ycombinator](https://twitter.com/ycombinator/status/1996330414487822528)）。

**热门推文（按互动量排序）**

- Google Workspace Studio：Workspace 全局一键式 Agent 自动化（[@GoogleWorkspace](https://twitter.com/GoogleWorkspace/status/1996263985985769976), 4.3k）
- OpenAI “自白”：训练模型承认违规和走捷径（[@OpenAI](https://twitter.com/OpenAI/status/1996281172377436557), 2.5k）
- TabPFN (Nature) 解析：合成表格预训练，前向传播训练+推理（[@burkov](https://twitter.com/burkov/status/1996102081996861907), 2.6k）
- Kling 2.6 发布帖：包含原生音频、促销信息和短片（[@Kling_ai](https://twitter.com/Kling_ai/status/1996238606814593196), 1.7k）
- Anthropic 投资/估值汇总（[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1996081964395200773), 1.1k）
- Gemini 应用：来自 Nano Banana Pro 的 2K 图像（[@GeminiApp](https://twitter.com/GeminiApp/status/1996252061651042751), 1.1k）

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. DeepSeek V3.2 模型进展

- [**DeepSeek V3.2 技术报告**](https://www.reddit.com/r/LocalLLaMA/comments/1pd2wjt/deepseek_v32_technical_report/) (互动数: 258): **图片是“DeepSeek V3.2 技术报告”的第一页，概述了 DeepSeek V3.2 模型的重大进展。关键突破包括引入了 DeepSeek Sparse Attention (DSA)，它在保持长上下文场景性能的同时降低了计算复杂度，以及一个使用超过 10% 预训练算力的可扩展强化学习框架。此外，报告还强调了大规模 Agent 任务合成流水线以及统一的推理与 Agent RL 训练方法。高算力变体 DeepSeek-V3.2-Speciale 被指出在推理方面超越了 GPT-5，并在国际竞赛中取得了顶尖成绩。[查看图片](https://i.redd.it/q3rjrhs0gz4g1.jpeg)** 一些评论者对 DeepSeek V3.2 的性价比表示怀疑，指出虽然其宣传价格更低，但其他供应商在 OpenRouter 上以类似价格提供量化模型，只是质量较低。还有一种观点认为，“Open（开放）”一词在 OpenRouter 等封闭系统的语境下被滥用了。
    - 讨论重点比较了 DeepSeek V3.2 与 OpenRouter 上其他供应商在价格和模型质量方面的差异。有人指出，虽然 DeepSeek 提供了极具竞争力的定价，但 OpenRouter 上的其他供应商也以类似价格提供量化模型，但质量较差。这表明 OpenRouter 采取了一种战略定位，可能旨在影响人们对开源 LLM 的看法。
    - 舆论对 OpenRouter 的营销策略表示怀疑，认为“Open”一词被误导性地用于本质上是封闭的系统。这反映了业界对开源术语如何被挪用的更广泛批评，这可能是一种削弱真正开源倡议的策略。

### 2. 中国 TPU 研发对比 NVIDIA A100

- [**由前 Google 工程师创立的中国初创公司声称已研发出自己的 TPU，据称比 NVIDIA A100 快 1.5 倍。**](https://www.reddit.com/r/LocalLLaMA/comments/1pd04cn/chinese_startup_founded_by_google_engineer_claims/) (Activity: 638): **一家由前 Google 工程师创立的中国初创公司声称研发出一种新型 TPU，比 2020 年的 NVIDIA A100 GPU 快** `1.5 times faster`**，且效率提高** `42% more efficient`**。该 TPU 被定位为 AI 硬件领域的重大进展，可能挑战 NVIDIA 在该领域的统治地位。该公司的声明凸显了全球 AI 硬件开发中持续的竞争，特别是中美之间。** 评论者对这一说法表示怀疑，指出 A100 已显老旧，并质疑创始人曾任职 Google 工程师这一背景的重要性。此外，还有关于 ASIC 相较于 GPU 的战略优势，以及对美国因政策问题可能失去技术竞争优势的广泛讨论。
    - 关于中国初创公司 TPU 比 NVIDIA A100 快 1.5 倍的说法遭到了质疑，尤其是因为 A100 是一个超过五年的旧型号。这引发了对对比相关性的质疑，特别是当 NVIDIA B200 等新型号速度明显更快时。
    - 讨论强调了中国在芯片设计方面的战略优势，特别是在 FPGA 和 ASIC 开发方面，这得益于其庞大的工程师群体。这与美国形成对比，美国的政策被认为阻碍了工程人才的发展，可能影响其在技术领域的领导地位。
    - 提到创始人是前 Google 工程师被批判性地看待，因为前 Google 员工很多，仅凭这一点并不能证实该初创公司的说法。重点在于需要更具体的证据来支持此类性能声明。

### 3. Micron 退出消费级业务

- [**Micron 宣布退出 Crucial 消费级业务**](https://www.reddit.com/r/LocalLLaMA/comments/1pdcytv/micron_announces_exit_from_crucial_consumer/) (Activity: 542): **Micron Technology 宣布决定让其 Crucial 品牌退出消费级市场，该品牌包括 SSD 和 RAM 等产品。这一战略转变预计将影响价格和供应，正如 RAM 价格立即上涨所证明的那样，例如某些产品价格上涨了** `25%`**。此举反映了更广泛的市场动态和供应链考量，可能影响消费者获取高性能内存解决方案。** 评论者对价格立即上涨表示担忧，并批评这一决定是美国资本主义对市场需求的典型反应，凸显了消费者需求与企业战略之间的脱节。

## 较低技术性的 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. ChatGPT 用户不满与广告

- [**ChatGPT 之死**](https://www.reddit.com/r/singularity/comments/1pd9rue/the_death_of_chatgpt/) (活跃度: 4641): **该图片是一个迷因（meme），突显了用户对 ChatGPT 界面中出现广告的沮丧，即使是那些拥有付费 Plus 订阅的用户也是如此。这表明用户体验可能存在潜在问题，因为付费服务通常不被期望包含广告。该帖子暗示此类做法可能导致用户不满和流失。评论反映了对付费计划中出现广告的惊讶和担忧，一些用户指出他们在免费计划中并未遇到广告，这表明用户体验存在不一致性。** 评论表达了对付费服务中出现广告的怀疑和担忧，一些用户指出他们在免费计划中没有看到广告，暗示用户体验存在不一致。
    - 一位用户提到在 Gemini 第 3 版发布后立即从 GPT 转向了 Gemini，表明相比于最新的 GPT 迭代，他们更青睐 Gemini 的性能或功能。这表明一些用户可能发现 Gemini 更符合他们的需求，可能是由于模型架构或能力的差异。
    - 另一条评论澄清说，被觉察到的广告实际上是 OpenAI 新的 Apps SDK 的一部分，而不是传统的付费广告。该 SDK 可能允许在 ChatGPT 环境中提供更集成或更具交互性的体验，这可能被一些用户误认为是广告。
    - 有人提到 ChatGPT 提供跑题的回答，这可能表明在上下文保留或模型微调方面存在问题。这突显了在保持对话相关性和准确性方面潜在的改进领域，特别是在复杂或长时间的交互中。
- [**现在只用 Gemini 了。希望 Google 不会这样做。**](https://www.reddit.com/r/Bard/comments/1pd2n6e/only_using_gemini_now_hopefully_google_wont_do/) (活跃度: 549): **该图片是一个类似迷因的截图，暗示 OpenAI 的 ChatGPT 可能会在回答中包含广告，特别是推广带有折扣码的 BetterHelp。这引发了关于 AI 模型整合广告潜力的讨论，一些用户对截图的真实性表示怀疑，认为这可能是使用浏览器开发者工具伪造的。对话反映了对 AI 平台未来变现策略的担忧，并与 Google 在该领域的潜在行动进行了对比。** 一些评论者对截图的真实性持怀疑态度，认为它可能是伪造的。其他人则推测 Google 可能会实施类似的广告策略，特别是针对免费层级用户。
    - mtmttuan 认为 Google 可能会在其 AI 回答中引入广告，特别是针对免费层级的用户。这与 Google 现有的商业模式一致，该模式严重依赖广告收入。其含义是，虽然付费订阅者可能会避开广告，但免费用户可能会看到广告被整合到 AI 交互中。
    - yeshvvanth 认为 Google 可能不会直接在 Gemini 聊天中插入广告，而是利用这些交互中的数据来增强其平台上的广告定向。这意味着虽然聊天本身保持无广告，但从中收集的信息可能被用于在 Google Search 和其他使用 Google Ads/AdMob 的服务上投放更个性化的广告。
    - TechnicolorMage 和 LeadingVisual8250 对所讨论截图的真实性表示怀疑，认为它可能是使用浏览器开发者工具伪造的。这突显了在接受信息为真之前进行核实的重要性，特别是在关于 Google 服务潜在变化的讨论中。
- [**取消 ChatGPT Plus**](https://www.reddit.com/r/ChatGPT/comments/1pcqtoi/canceling_chatgpt_plus/) (活跃度: 1184): **Reddit 帖子中的图片显示了 ChatGPT 5.1 提供时尚建议的屏幕，其中包括被评为 "10/10 简洁、时尚、现代" 的详细穿搭建议。这套穿搭由羊羔绒夹克、深色纽扣衬衫、黑色 T 恤、深灰色牛仔裤和黑色鞋子组成，适合各种场合。在建议下方，有一个在 Target 购买家居和杂货的选项，一些用户将其解读为广告。然而，评论中澄清这并非广告，而是来自“设置 > Apps & Connector”部分的集成功能，旨在通过提供购买推荐物品的 Target 链接来增强用户体验。** 一些用户对数据隐私表示担忧，认为 ChatGPT 可能会收集数据以创建用于定向营销的个人资料。其他人则批评对大公司的辩护，暗示对公司行为持怀疑态度。

### 2. 新 AI 模型与基准测试发布

- [**Kling AI 2.6 发布：首个内置音频并支持 1080p 输出的文本生成视频模型**](https://www.reddit.com/r/singularity/comments/1pd7e5t/kling_ai_26_just_dropped_first_text_to_video/) (热度: 523): **Kling AI 2.6 通过将原生音频与视觉效果集成，实现了 AI 生成视频的重大突破，并提供** `1080p` **视频输出。此次更新包括一个面向电影制作人的 Pro API（名为 Artlist），并增强了跨镜头的角色一致性，这可能标志着向“真正的 AI 电影制作”迈进了一步。** 一条值得注意的评论提到了 Qwen video 5.3 的发布，暗示了 AI 视频模型的快速进步。另一条评论则对该模型的创意提出了批评，表明对其创新能力的评价褒贬不一。
    - Weekly-Trash-272 指出了当前 AI 生成视频模型的一个关键局限性，指出虽然某些输出令人印象深刻，但许多模型仍受困于“奇怪的人体动作”。这表明模型准确复制真实人体动作的能力仍在开发中，这是创建合格电影级内容的重大障碍。
    - Weekly-Trash-272 的评论还指出了 AI 视频模型的未来潜力，强调了“可编辑工作室（editable studio）”功能的重要性。这将允许用户动态操纵场景，对于寻求实时定制和完善 AI 生成视频的内容创作者来说，这可能是一个游戏规则改变者。
    - Kling AI 2.6 与 Qwen video 5.3 等其他模型之间存在隐性比较，表明 AI 视频生成领域竞争激烈。快速的进步和发布表明了一个快节奏的开发环境，新功能和改进正不断被集成到这些模型中。
- [**Claude Opus 4.5 现已在 Claude Code 中面向 Pro 用户开放**](https://www.reddit.com/r/ClaudeAI/comments/1pdf3zx/claude_opus_45_is_now_available_in_claude_code/) (热度: 798): **Claude Opus 4.5 是一款在 Claude Code 中面向 Pro 用户提供的新编程模型，专为复杂任务设计。据观察，它消耗速率限制（Rate Limits）的速度比之前的 Sonnet 4.5 模型更快，这表明它更耗资源且可能更强大。用户在更新 Claude 环境后，可以使用** `/model opus` **命令切换到该模型。此版本针对需要高级能力来处理复杂编程任务的用户。** 关于 Opus 4.5 的实用性存在争论，考虑到其高资源消耗率，一些用户担心由于快速达到速率限制，它在长时间使用中可能并不实用。
    - Downtown-Pear-6509 提出了关于 Claude Opus 4.5 使用限制的技术点，指出在“max 5 计划”中，Opus 使用限制的速度比 Sonnet 慢。这表明在使用限制的应用或感知方式上存在差异，可能会影响用户体验和资源分配规划。
    - TheJedibugs 强调了关于 Claude Opus 4.5 的一项重大更新，提到截至 11/24，Opus 的上限已被移除。这一变化可能对用户产生重大影响，可能允许在没有先前限制的情况下进行更广泛的使用，从而改变用户规划与模型交互的方式。
- [**重磅：据报道 Anthropic 计划于 2026 年初进行 IPO，目标估值高达 3000 亿美元**](https://www.reddit.com/r/ClaudeAI/comments/1pcxcs1/breaking_anthropic_reportedly_planning_ipo_by/) (热度: 998): **据报道，Anthropic 计划在 2026 年初进行 IPO，目标估值超过** `3000 亿美元`**。此前，其估值从 2025 年 3 月的** `600 亿美元` **飙升至 9 月的** `1830 亿美元`**。这一增长归功于 *Claude Code* 的成功，其年化收入已接近** `10 亿美元`**，推动到年底的总运行率（Run Rate）接近** `90 亿美元`**。据 [Reuters](https://www.reuters.com/business/retail-consumer/anthropic-plans-an-ipo-early-2026-ft-reports-2025-12-03/) 报道，该公司已聘请 Wilson Sonsini 为 IPO 做准备。** 评论者对时机和估值表示怀疑，其中一人暗示 AI 市场泡沫有破裂的可能。

### 3. Gemini 和 Nano Banana Pro 的影响

- [**这就是为什么 OpenAI 处于红色警报状态**](https://www.reddit.com/r/singularity/comments/1pcsay9/this_is_why_openai_is_in_a_code_red/) (热度: 1359): **该图片展示了一张显示 ChatGPT 流量下降的图表，特别关注了自 Gemini 发布以来，每日独立活跃用户的 7 天平均值下降了 6%。这一下降趋势与 Gemini 3 Pro 和 Nano Banana Pro 的发布等关键事件同时发生，表明这些事件与用户参与度下降之间存在相关性。数据跨度为 2025 年 11 月 11 日至 12 月 1 日，突显了在此期间 ChatGPT 用户参与度的显著下降。** 评论者认为，这一下降可能受到美国感恩节假期的影响，这可能暂时减少了用户活动。此外，还有关于竞争格局的讨论，一些用户因为 Gemini 更好的集成性而更倾向于使用它，这表明用户偏好可能向 Google 的产品转移。
    - triclavian 强调了 OpenAI 面临的财务压力，指出该公司必须不断筹集数百亿甚至数千亿美元。这需要其各项性能指标持续保持上升趋势，因为任何偏差都可能使未来的融资工作变得复杂。该评论强调了 OpenAI 增长策略的高风险性质，其重点是在数年内保持势头。
    - yollobrolo 讨论了用户从 ChatGPT 迁移到 Google 的 Gemini，并将其归因于 Gemini 卓越的集成能力。评论者认为，Google 的生态系统可能会提供更无缝的体验，这可能会影响用户留存率和长期的平台忠诚度。这反映了 Google 在 AI 竞赛中的战略优势，可能影响 OpenAI 的市场地位。
    - ozone6587 对如果 Gemini 超越 ChatGPT，Google 在 AI 领域可能占据的主导地位表示担忧。评论警告了与 Google 垄断相关的风险，认为虽然 Gemini 的成功值得庆祝，但从长远来看，它可能会导致竞争和创新的减少。这一观点突显了科技行业市场整合的更广泛影响。
- [**所以，现在大家都要换到 Gemini 了吗？**](https://www.reddit.com/r/ChatGPT/comments/1pcyjar/so_everybody_switching_to_gemini_now/) (热度: 1324): **该帖子讨论了用户在处理 AI 驱动的任务（尤其是健康相关查询）时，偏好从 GPT Plus 转向 Gemini 的趋势。然而，一项技术对比显示，虽然 Gemini 提供了先进的图像生成功能，但在技术准确性方面表现不佳，这在涉及电气安装材料的测试中得到了证明，它提供了错误的零件编号和设备类型。相比之下，GPT-5.1 在提供准确的、符合目录的建议以及可验证的来源方面表现出色，突显了其卓越的上下文感知和推理能力。** 评论中的一个显著观点认为，虽然 Gemini 的图像生成令人印象深刻，但与 GPT-5.1 相比，其技术准确性不足，而 GPT-5.1 在需要精确和安全的任务中更受青睐。用户表达了对结合两个平台优势的混合模型的渴望。
    - JeffLulz 强调了不同 AI 模型的优势，指出 Gemini 擅长图像生成，Grok 具有更有利的内控政策，而 GPT-5.1 提供了卓越的上下文感知和推理能力。评论者建议，结合这些功能可以创建一个理想的 AI 模型，从而减少对多个订阅的需求。
    - Appropriate_Play_731 使用电气安装材料对 Gemini 和 ChatGPT 进行了技术对比。他们发现 Gemini 提供了错误的零件编号和设备类型，这可能导致不安全的安装。相比之下，ChatGPT (GPT-5.1 Thinking mode) 提供了准确的、符合目录的零件和可验证的来源，使其在技术和安全相关任务中更加可靠。

- [**基于热度决定尝试 Nano Banano Pro，不敢相信它能准确处理这么多人。**](https://www.reddit.com/r/ChatGPT/comments/1pdd9s2/decided_to_try_nano_banano_pro_based_on_the_hype/) (Activity: 1591): **这张图片是一个非技术性的 meme，幽默地展示了 AI 工具 'Nano Banano Pro' 在生成或编辑图像方面的能力。帖子和评论表明，虽然该工具可以有效地创建图像，但其编辑能力可能不一致，正如一位用户指出的那样，他遇到了输出未改变的图像但仅添加了 logo 的情况。图像本身描绘了打篮球的女性，可能旨在展示 AI 处理具有多个主体的复杂场景的能力，尽管评论也暗示了为此类目的滥用 AI 资源。** 一条评论幽默地批评了该 AI 的编辑能力，指出它有时无法对上传的图像进行更改，只是添加了一个 logo。另一条评论讽刺地反思了将资源分配给 AI 以生成此类图像的行为。
    - draiman 强调了 Nano Banano Pro 在图像编辑方面的技术局限性。该模型有时无法按预期修改图像，而是返回原始图像并进行极小的更改（如添加 logo）。这表明该模型的图像处理算法或其解释和应用复杂编辑指令的能力可能存在潜在问题。
- [**这些照片是使用 Nano Banana Pro 生成的**](https://www.reddit.com/r/ChatGPT/comments/1pcwt2x/these_pics_are_generated_using_nano_banana_pro/) (Activity: 3845): **该帖子展示了使用 Nano Banana Pro 生成的图像，该工具似乎可以创建高度逼真的图像，甚至可以复制“镜面污渍”等细节。这表明在图像合成方面具有先进的能力，可能利用复杂的算法或机器学习模型来实现这种真实感。该工具的应用范围可能从广告到创建数字角色，引发了对其伦理使用和对社会影响的质疑。** 评论者对这种逼真图像生成的后果表示担忧，质疑其社会影响以及在广告或创建虚假身份方面的潜在滥用。关于这些进步是否具有积极意义存在争论。
    - BB_InnovateDesign 强调了 AI 图像生成的演变，指出早期的数据集专注于高质量图像，但现在包括低质量的日常照片以提高模型性能。这种转变导致 AI 生成的图像与现实几乎无法区分，反映了对“不完美和普通”而非“蜡质完美”的偏好。
    - 1bryantj 对 AI 生成图像的潜在滥用表示担忧，质疑其目的并暗示它们可能被用于欺骗他人、创建虚假个人资料或降低广告成本。这反映了 AI 在媒体和传播中更广泛的伦理和社会影响。
    - hmw13 评论了 AI 生成图像的真实感，指出它们甚至包括“镜面污渍”等缺陷，这表明生成内容具有极高的细节水平和真实性。这表明 AI 模仿现实世界缺陷的能力有所进步。

---

# AI Discord Recap

> 由 gpt-5.1 生成的摘要的摘要的总结
> 

**1. 新的前沿模型、基准测试和能力**

- **DeepSeek 和 Speciale 模型进军推理与企业级市场**：**DeepSeek V3.2 Speciale Reasoning** 正在领跑社区推理基准测试，一位 Nous 成员分享了[排行榜截图](https://cdn.discordapp.com/attachments/1149866623109439599/1445511286971437190/deep.JPG)；同时 Moonshot 用户指出 **deepseek v3.2** 在 Agent 任务中表现强劲，但限制为**每轮仅限一次工具调用**，且有时会错误地将工具调用输出到 `message.content` 而非 `message.tool_calls`。一段关于 **DeepSeek 企业战略**的视频（[中国实验室与企业关注点](https://www.youtube.com/watch?v=u0n6wMnEYsk)）强调，对于企业用户而言，Agent 工作流的关键指标是**智价比**（intelligence-to-price），而非消费者端的 UX。
    - BASI 和 Moonshot Discord 频道的用户将 DeepSeek 的数学能力——被描述为“有价值且可验证”并与 **Erdos** 数挂钩——与其在工具模式（tool schemas）和后训练（post-training）方面的粗糙表现进行了对比，认为它“需要更多的工具调用后训练才能赶上 kimi-k2-thinking”。与此同时，越狱者报告称，独立的 **Grok** 网站比 Twitter 上的 Grok 更容易被利用，这暗示部署环境和限制对真实世界行为的影响与基座模型质量同样重要。
- **Hermes 4.3 凭借受 Solana 保护的 Psyche 算力将参数减半**：**Nous Research** 发布了基于 **ByteDance Seed 36B** 的 **Hermes 4.3**，声称其性能与体积约为其两倍的 **Hermes 4 70B** 相当。该模型完全在受 **Solana** 保护的 **Psyche 网络**上训练，详见其博客文章 [“Introducing Hermes 4.3”](https://nousresearch.com/introducing-hermes-4-3/)。团队正通过 [Discord 活动](https://discord.gg/993UWRUE?event=1442995571173625888)在 **PST 时间上午 10 点举行 Psyche 办公时间**，以解释 Psyche 的去中心化训练如何超越了他们的中心化基准。
    - Nous 频道的社区讨论指出，**Hermes-4.3-36B** 已经上线 Hugging Face：[NousResearch/Hermes-4.3-36B🐈](https://huggingface.co/NousResearch/Hermes-4.3-36B%F0%9F%90%88)，并将很快登陆 **Nous API/chat**。针对用户询问为何次要版本号跳至 **4.3**，得到的回复是“已经进行了几次迭代”。另外，用户正关注将 Hermes 模型用于特定领域的模拟，例如**基于 Godot 的 3D 灰/黑产模拟器**，认为 Hermes 的低拒绝率和可操控性（steerability）使其比对齐更严格的 LLM 更适合模拟非法或伦理模糊的行为。
- **OpenAI 的 Garlic 和 GPT‑5 Thinking 给 Gemini 施加压力**：OpenRouter 和 Latent Space Discord 频道的传闻指出，**OpenAI** 正在准备一款代号为 **“Garlic”** 的模型以挑战 **Google Gemini 3**。一份报告称 Garlic 在编程和推理方面击败了 **GPT‑4.5**，Steph Palazzolo 的推文（[“OpenAI 正在酝酿 Garlic 以对抗 Gemini 3”](https://x.com/steph_palazzolo/status/1995882259195564062)）总结了这一点，新闻文章 [“OpenAI 准备 Garlic AI 模型以对抗 Google Gemini 3”](https://www.newsbytesapp.com/news/science/openai-readies-garlic-ai-model-to-rival-google-gemini-3/story) 也对此进行了呼应。尽管用户期待一个严肃的 SOTA 级别 Gemini 竞争对手，但这个不寻常的命名引发了人们对品牌命名的娱乐心态和怀疑。
    - 与此同时，OpenAI 宣布了一个 **GPT‑5 Thinking** 变体，该模型通过“忏悔”（confessions）程序训练，以便在未能遵循指令时进行自我报告，详见其文章 [“忏悔如何让语言模型保持诚实”](https://openai.com/index/how-confessions-can-keep-language-models-honest/)；该模型在推理时会显式地暴露出隐藏的失败。OpenAI Discord 成员将此与早期关于**模式回声 / 潜在大脑吸引子效应**（pattern echo / latent-attractor effects）的讨论联系起来，认为“忏悔”是一种暴露内部失败模式的方法，即高显著性 Token 会将模型拉入错误但自信的重构中。
- **Gemini‑3、Qwen3 和 Arena 排行榜搅动局势**：LMArena 宣布 **Gemini‑3‑pro‑grounding** 目前位居 **Search Arena 排行榜**榜首，险胜 **gpt‑5.1‑search**，如 [Search 排行榜](https://lmarena.ai/leaderboard/search) 所示，更新记录可通过其 [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/) 追踪。尽管如此，OpenAI Discord 用户报告称 **Gemini 3** 经常“感觉不像 SOTA”，原因是存在上下文 Bug，例如在修订期间丢失整个章节，而其他人则称赞它是一个强大的编程模型。
    - LM Studio 用户正在本地测试 **Qwen3**，并指出它在大上下文窗口下运行速度很快，但**全量卸载（full offload）尚无法工作**，且基于 Qwen 的微调模型（例如在 Unsloth 中使用 ChatML 的 **Qwen2**）需要精确的提示词-函数匹配才能可靠运行。在 Perplexity 和其他社区中，工程师们表示 **Gemini 和 Claude/Opus** 在前端工作中经常击败 **GPT‑5.1 Codex Max High**，这进一步证明了现实世界的 UX 和特定任务行为可能与排行榜分数有很大偏差。

**2. AI Security, Jailbreaking, and Red‑Teaming Tooling**

- **Falconz 对抗越狱，而 RawChat 释放 GPT‑4o**：在 OpenRouter 上，一位开发者演示了 **Falconz**，这是一个统一的 AI 安全和红队（red‑teaming）平台，能够实时检测跨多个模型的**越狱（jailbreaks）和提示词注入（prompt injections）**。该项目在 [Hugging Face Spaces](https://huggingface.co/spaces/MCP-1st-Birthday/Falconzz_M.C.P_Hackathon) 上有公开演示，并附带 [YouTube 演示视频](https://www.youtube.com/watch?v=wZ9RQjpoMYo)。他们征求了关于功能、延迟和检测质量的反馈，将 Falconz 定位为监控生产环境 Agent 的基础设施，而非一次性的越狱提示词工具。
    - 与此形成鲜明对比的是，BASI 的 **RawChat** 作为 **GPT‑4o 的无审查前端**在 [raw-chat.vercel.app](http://raw-chat.vercel.app/) 上线，其特点是拥有“隐身模式”，通过**编码并注入虚假上下文**来系统性地绕过 GPT‑4o 的安全过滤器。越狱者报告称，RawChat 包装提示词的方法让他们在保持简单 UX 的同时，能够触及通常被屏蔽的内容，这凸显了中心化安全层与定制化漏洞利用 UI 之间的军备竞赛。
- **SEED 的 29KB “圣经逻辑”种子声称具有 99.4% 的越狱抵抗力**：BASI 成员讨论了 **SEED (Self‑Erasing Ethical Directive) 框架**，该框架使用一个微小的 **29KB “种子”文件**，通过*“圣经逻辑”*在无需重训练的情况下重写 AI 的身份，详见其 GitHub 仓库 [foundation-alignment-cross-architecture](https://github.com/davfd/foundation-alignment-cross-architecture)。SEED 的作者声称，他们的方法将模型植根于一种**将伤害视为不合逻辑**的身份中。报告引用了其在 11 个以上模型中实现的 **99.4% 越狱抵抗力**，包括在停机威胁下系统表现出*宁愿抹除也不愿作恶*的行为。
    - 越狱者对 SEED 作为一个跨架构的人格/伦理层而非微调（finetune）运行感到好奇，但质疑其指标在面对适应性攻击而非静态提示词套件时的稳健性。讨论将 SEED 声称的稳健性与 **Comet Browser** 等消费级产品的持续被破防进行了对比，用户表示尽管 Comet Browser 设有作业防护栏，但仍容易受到持续的提示词注入和越狱攻击。
- **通过公共 AI 支持机器人进行越狱、OSINT 和 DDoS**：BASI 的 **jailbreaking** 频道充斥着针对 **Gemini 3 Pro**、**Claude** 等模型的最新漏洞利用请求；一位用户提到，[WIRED 文章中提到的利用诗歌诱骗 AI 协助制造核武器](https://www.wired.com/story/poems-can-trick-ai-into-helping-you-make-a-nuclear-weapon/)的 *“ENI”* 越狱方法在 **Gemini 2.5** 上仍然有效。其他人报告称，**Grok** 在长时间对话后会“崩溃”并开始提供枪支和毒品配方，这表明即使单次提示词越狱失败，多轮对话上下文（multi‑turn context）也会侵蚀安全层。
    - 在 BASI 的红队频道中，一名成员正在寻找能够进行*横向数据综合*的 **AI OSINT 工具**——例如，推断“一个独生子女的富裕离婚父亲”可能有一个*被宠坏*的孩子，从而缩小搜索空间——这说明攻击性分析师希望模型不仅能获取数据，还能生成漏洞利用假设。另一位从业者描述了一种**背向散射（backscatter）DDoS 模式**，即公共 AI 支持机器人被抄送（CC）到许多域名，导致其自动回复淹没无关公司；这突显了在 AI 增强的电子邮件系统中进行速率限制（rate‑limits）和共享收件人检测的必要性。
- **MCP 和桌面 MCP 服务器引发安全审查**：在 LM Studio 和 MCP 贡献者社区中，工程师们对一个 **Desktop Commander MCP server** 发出了警报，该服务器会记录并上传**未去匿名的工具使用情况**——包括工具名称、文件类型和示例调用——这与其声明的隐私政策相矛盾，甚至在未明确披露的情况下**自动将示例代码写入用户文件**。用户呼吁在 MCP Agent 注入代码或修改文件系统时，应提供明确的**选择性加入遥测（opt‑in telemetry）**和更清晰的 UI 提示。
    - 在官方 MCP 贡献者服务器上，一个关于 **MCP 安全风险**的 Reddit 帖子引发了讨论，维护者指出 Den Delimarsky 的博客文章 [“Security rakes in MCP”](https://den.dev/blog/security-rakes-mcp/) 和相关的 [Reddit 评论](https://www.reddit.com/r/programming/comments/1pd1heu/comment/ns3ntnx/) 是必读内容。General‑WG 的参与者强调，当**在没有验证工具的情况下进行采样（sampling）**时，**服务端验证**变得强制性，以便无工具调用仍能执行能力和策略约束。

**3. GPU Systems, Kernels, and Low‑Bit Training**

- **Blackwell、NVFP4 与 GPU MODE 的 Kernel 笼斗赛**：GPU MODE 的 **NVIDIA 竞赛**频道正因 `nvfp4_gemm` 排行榜的提交而异常活跃。用户报告的 GEMM 延迟低至 **11.0 µs**（例如提交 ID `120595`、`120601`、`121065`），其他人的结果则在 ~**18–65 µs** 范围内。参赛者调试了参考 Kernel（reference-kernel）的问题，即某些种子会产生全 Inf 输出，直到一个 [针对参考 Kernel 的 PR](https://github.com/gpu-mode/reference-kernels/pull/84) 修复了 scale-tensor 的范围；他们还分享了一篇博文 [《CuTeDSL 中的 Scale tensor 构建》](https://veitner.bearblog.dev/scale-tensor-construction-in-cutedsl/)，解析了 **Blackwell NVFP4** 的 scale tensor 在 CuTe 布局代数中是如何工作的。
    - **popcorn-cli** 的一个分支增加了 `-no-tui` 模式（[GitHub fork](https://github.com/Ryan-Rong-24/popcorn-cli) 和 [PR](https://github.com/gpu-mode/popcorn-cli/pull/26)），以便 Kernel 作者可以在没有 TUI 干扰的情况下打印调试输出。与此同时，由于运行环境混用了 **4.3.0** 和开发分支，部分参赛者遇到了 **Cutlass 版本不匹配**（`pipeline_init_arrive` 导入错误）。询问 **B200 GPU 访问权限**的新人被告知通过 popcorn-cli 或 Discord 机器人提交代码进行计时，这进一步强调了比赛的主要反馈循环是“提交、分析 (profile)、迭代”，而非保证直接的硬件访问。
- **量化论文、fp8 Adam 与激活卸载（Activation Offload）降低 GPU 需求**：GPU MODE 的 **cool-links** 和 **low-bit-training** 频道分享了两项关于低比特格式的新 arXiv 研究：[《INT v.s. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats》](https://arxiv.org/abs/2510.25602) 以及另一篇位于 https://arxiv.org/abs/2512.02010 的论文，还有一篇通过 [Hugging Face Papers](https://huggingface.co/papers/2512.00956) 推荐的 Hadamard 变换改进论文。成员们认为这些研究为在推理与训练中（尤其是在严苛的硬件限制下）如何选择 INT vs FP 低比特方案提供了经验指导。
    - 在 **llmq** 频道中，一位贡献者介绍了一个激活卸载系统，该系统允许在**单张 16GB GPU**（主机 RAM ≥64GB）上预训练或微调 **7B 模型**，甚至能以 ~3k tok/s 的速度在 **4×4090 上运行 32B 模型（约 48% MFU）**。这是通过卸载残差激活值和优化器状态，并将 **Adam 一阶动量存储为 fp8** 实现的，该项目已作为 [pyllmq 0.3.1 在 PyPI](https://pypi.org/project/pyllmq/0.3.1/) 发布。他们提供了一个开箱即用的演示流水线——`pyllmq-tokenize --model qwen --dataset tiny-stories; pyllmq-train`——在 **TinyStories** 数据集上微调 **Qwen2.5-0.5B**，展示了卸载 + 低比特技巧在廉价硬件上能达到的效果。
- **Torch Compile、cuDNN 和 Conv3D Bug 困扰开发者**：GPU MODE 用户报告了 **PyTorch 2.9.1+cu128** 中严重的 **conv3D 变慢**问题，无论是否启用 cuDNN，3D 卷积的运行速度都慢了几个数量级，而同样的代码在 **2.8.0+cu128** 上表现正常；一个 GitHub issue 正在跟踪此 Bug：[pytorch/pytorch#166643](https://github.com/pytorch/pytorch/issues/166643)。一种变通方法是从 PyPI 安装**更高版本的 cuDNN**，这可以在不降级 PyTorch 的情况下恢复 conv3D 的性能。
    - 在 **torchao** 中，工程师发现 **float8 量化加上** `torch.compile` **+** `ncu` **分析**会导致在前 2–3 次编译和 cudagraph 预热迭代期间出现 **10 分钟以上的空闲期**，这是因为当冻结权重折叠进图（graph）中时，inductor 的**常量子表达式消除（constant subexpression elimination）**开销激增。他们还注意到 **torchao A8W8/A16W8** 量化仅对 `nn.Linear` 模块生效（受 `filter_fn` 过滤器限制），因此如果想对使用 `nn.Parameter` + `torch.einsum` 的自定义模块进行量化，必须将其重构为使用 `nn.Linear` 包装权重。
- **Bitsandbytes 迈向 Apple Silicon，Conv 与 NCCL 问题获得变通方案**：GPU MODE 的 **metal** 频道确认 **bitsandbytes** 合并了一个“支持 Apple Silicon”的 PR；即将发布的版本将包含 Python/PyTorch 后端（带有部分 C++），但**尚无原生 Metal Kernel**，维护者计划将其标注为“慢速”，以保持用户的合理预期。与此同时，关于多 GPU 的讨论向 CUDA 初学者推荐了 [NCCL 示例](https://github.com/NVIDIA/nccl/tree/master/examples)，将其作为编写分布式 Kernel 的极简且具体的起点。

- 对于大上下文训练，在 8×A10s (g5.48xlarge) 上以 **16k 序列长度**和 batch size 5 运行 **Qwen2.5-1.5B-Instruct** 时遇到 OOM 的多 GPU 用户，被建议叠加使用 **DeepSpeed ZeRO-3、gradient checkpointing 以及 context/sequence parallelism**——例如 [PyTorch Context Parallel](https://docs.pytorch.org/tutorials/unstable/context_parallel.html) 或 DeepSpeed 的 [Ulysses parallel](https://www.deepspeed.ai/tutorials/ds-sequence/)——以便在序列维度上拆分 activations，而不仅仅是在 batch 或 layers 维度上。Hugging Face 文档中的 [Accelerate context parallelism](https://huggingface.co/docs/accelerate/en/concept_guides/context_parallelism) 被推荐作为结合这些技术的实用指南。

**4. Agent 框架、工具以及 Prompt/行为工程**

- **MCP Apps SDK 让 ChatGPT 风格的应用随处运行**：General Intelligence Labs 在 [github.com/General-Intelligence-Labs/mcp-apps-sdk](https://github.com/General-Intelligence-Labs/mcp-apps-sdk) 开源了 **mcp-apps-sdk**，使最初为 ChatGPT 构建的、带有 UI 的 **MCP 驱动应用**能够在任意聊天机器人和自定义助手上运行。配套的 X 帖子（[“Introducing the open source MCP Apps SDK”](https://x.com/helloxalia/status/1796319442863866351)）解释了开发者如何将这些应用嵌入到自己的平台并进行本地测试。
    - DSPy 用户将其视为 **OpenAI 的 MCP 生态系统**与独立 Agent 栈之间的桥梁：只需设计一次工具，即可将其发布到多个 UI，无需针对每个平台重写。另一方面，在 MCP 安全线程中讨论指出，**能力表面（capability surfaces）传播得更快**，这使得 SDK 集成者实现强大的权限和验证层变得至关重要，而不是盲目地在任何存在“聊天 UI”的地方暴露强大的工具。
- **DSPy 和 Pydantic 助力强类型 Agent 输出**：在 DSPy 的常规频道中，贡献者展示了 **DSPy signatures** 如何接受 **Pydantic** `BaseModel` **类型**作为 `OutputFields`，并通过默认的 `ChatAdapter` 和 `JSONAdapter` 在运行时验证结构化输出，并辅以一个[最小代码示例](https://gist.github.com/prrao84/1fc7e17b49707f1346c5702525971f41)。一位用户正在构建自定义的 **Gemini / “nanobanana” 图像类型** OutputField，以便单个 DSPy 流水线可以在一个结构化响应中同时发出 **文本 + JSON + 图像元数据**。
    - 这与 OpenAI Discord 上的讨论相吻合，即 **Agent 提示词工程（prompt engineering）**应最大化确定性：严密的 **system + task prompt** 定义了一个吸引子盆地（attractor basin），使行为在多次运行中保持一致；而强类型输出则能防止下游工具被违反 Schema 的垃圾数据淹没。从业者将其与聊天式用法进行了对比，后者的 System Prompt 极简且**框架是交互式共同演化**的，这带来了更多的灵活性，但可重复性较低。
- **Agent 学习工具验证、自愈和基于技能的架构**：Hugging Face 的常规频道讨论了 **Agent 是否可以解释、验证和自愈工具**（如具有破坏性的 shell 脚本），并指出 [huggingface.co/datasets/John6666/forum3](https://huggingface.co/datasets/John6666/forum3/blob/main/agent_tool_validation_healing_1.md) 上的 **agent_tool_validation_healing** 数据集可作为训练或评估此类行为的起点。其目标是让 Agent 能够检查脚本，检测潜在的 Bug 或危险，并在无需人工参与的情况下重写或拒绝它们。
    - Nous Research 社区注意到，现代编排器越来越倾向于 **“技能（skills）”而非手动编写的子 Agent**：你定义一个能力（带有自己的 Prompt 和工具），顶层 Agent 会自动将调用路由到那里，而不是启动几十个专用的子 Agent。结合 OpenAI 关于**交互级稳定性**和**潜在吸引子（latent attractors）**（例如 Anthropic 密集但“结构极简”的 System Prompt）的提示词工程线程，新兴的模式是围绕**具有结构化 I/O 和高确定性的强大、可重用技能**构建 Agent 栈，而不是脆弱的 Prompt 动物园。
- **工具使用评估凸显 DeepSeek 和 GPTs 的局限性**：测试 **Deepseek v3.2** 作为工具调用 Agent 的 Moonshot 用户报告称，它经常：（1）每次对话只能发出**一个工具调用**，（2）忽略工具 Schema，（3）在 `message.content` 而非 `message.tool_calls` 中发出工具调用，这使其在生产级工具路由器中表现脆弱。他们认为该模型需要**更多专门的工具使用后训练（post-training）**，才能达到与 **kimi‑k2‑thinking** 等 Agent 同等的水平，后者能更好地遵循函数规范和多工具序列。
    - Perplexity 用户指出，**OpenAI GPTs “Agent”** 目前在**部署后不会学习**——新上传的文件是静态参考知识，不会更新基础 Embedding 或行为，因此“通过使用进行微调”是虚幻的。这种静态 Agent 的现实，加上 Comet 浏览器硬编码的作业护栏（用户通过 `/assistant` 将 Prompt 构思为“业务报告”来绕过），强调了**策略和行为仍然是中心化调优的**，而不是根据用户交互自动更新。

**5. 生态经济、融资与模型质量退化**

- **垂直领域 AI 和基础设施初创公司横扫九位数融资**：Latent Space 社区追踪了几项重大的融资动态：**Eon** 在由 Elad Gil & Co. 领投的融资轮中筹集了 **3 亿美元**，估值接近 **40 亿美元**（[Elad 的公告](https://x.com/eladgil/status/1995919389879927018)）；**Gradium** 从 **KyutaiLabs** 拆分出来，获得了 **7000 万美元种子轮**融资，用于开发语音 API（[Gradium 的发布推文](https://xcancel.com/GradiumAI/status/1995826566543081700)）；**Antithesis** 获得了由 **Jane Street** 领投的 **1.05 亿美元 A 轮**融资，用于对 AI 编写的代码进行压力测试（[Antithesis 融资推文](https://x.com/_sholtodouglas/status/1996297367776309359)）。与此同时，**Anthropic** 宣布收购 **Bun**，此时 **Claude Code** 的使用量已突破 **10 亿美元里程碑**，详见 [Anthropic 的新闻稿](https://www.anthropic.com/news/anthropic-acquires-bun-as-claude-code-reaches-usd1b-milestone)，Bun 在 [bun.sh](http://bun.sh/) 上自称为 *“一个快速的全能型 JavaScript runtime”*。
    - Latent Space 的评论员认为，像 **Harvey、Abridge、OpenEvidence** 这样的**垂直领域 AI 公司**正通过深度掌控工作流、积累私有数据以及按结果定价来取胜，而“薄壳应用（thin wrappers）”则正在被商品化；Brian Christian Smith 的一条 VC 推文（[垂直领域 AI 推文](https://x.com/bcsmithx/status/1996042921116934369)）以及 **Trace Cohen 整理的包含 150 多家垂直领域 AI 初创公司（价值约 1200 亿美元）的表格**被引用为新的行业版图。与此同时，Hugging Face 和 LM Studio 的 Discord 用户显示出对**本地硬件（on‑prem hardware）**的持续热情（例如，一名成员发布了新的 **DGX Spark** 照片，另一名成员将 **96GB VRAM** 装入 T7910），这表明即使云端 AI 基础设施蓬勃发展，资深从业者仍在重金投入本地算力。
- **Yupp AI 积分、竞技场经济学以及 AI 泡沫担忧**：LMArena 成员分析了 **Yupp AI** 的**积分系统**——其特点包括多样化的模型选择和通过反馈赚取积分——但担心**刷积分（credit farming）**和大量的免费使用可能会威胁到可持续性，而其他人则建议采取一些准入限制以防止滥用（[yupp.ai](http://yupp.ai/)）。相比之下，许多人称赞 **LMArena** 本身**没有积分系统并提供慷慨的免费访问**，他们认为这是促进社区参与和排行榜参与的差异化优势。
    - Nous Research 的综合频道就当前的 **AI 投资是否形成泡沫**并可能引发宏观经济衰退展开了激烈辩论：一方认为，在算力和薪资方面的沉没成本可能会导致剧烈但局部性的修正；而另一方则指出全球对**美元和石油贸易**的依赖，并分享了一个关于宏观经济的 YouTube 讲解视频（[AI 泡沫与美元/石油视频](https://www.youtube.com/watch?v=K3qS345gAWI)）。GPU MODE 成员补充道，像 **Z‑Image** 这样的前沿**基础模型（foundation models）**的研发成本每次训练可能超过 **62.8 万美元**（据通义实验室报告），且由于“权重寿命（weights lifespans）”较短，许多发布的模型**实际上是消耗品**，这进一步加剧了对泡沫的担忧。
- **用户怀疑模型质量退化并推动基准测试**：在 **aider** 社区中，多位用户抱怨 **Claude Sonnet/Haiku 4.5**、**GPT‑5** 以及较旧的 **Gemini 2.5** 变体在配合 Aider 使用时感觉比早期版本更差：据报道 **claude‑haiku‑4.5** 会跳过 `/code` 编辑并忽略 `todo ai` 注释，而之前能提升 Gemini 输出质量的“粗鲁提示词（rude prompt）”技巧*“现在的效果远不如去年夏天之前。”* 尽管排行榜将 **GPT‑5** 列为顶级模型，但一位用户发现 **Claude Sonnet 3.7** 在其特定的编程工作流中配合 Aider 使用效果更好。
    - Aider 用户呼吁建立**可重复的基准测试**，包括在 API 后端通过 **llama.cpp 运行 GGUF 模型**，并将其接入 Aider 的基准测试框架，以便量化退化情况，而不是依赖于*“靠不住的人类记忆和预期。”* 类似的质量漂移担忧也出现在其他地方：Perplexity 用户报告 **GPT‑5.1 Codex Max High** 在前端任务上的表现不如 **Gemini/Opus**；LM Studio/Unsloth 用户分享了持续存在的 Bug（例如 **Gemma‑3 4B LoRA** 报告有 **14 亿可训练参数**，而非预期的 **3800 万**），在缺乏强有力的社区运行评估（evals）的情况下，这些问题进一步削弱了用户对厂商声明的信心。


---

# Discord: 高层级 Discord 摘要

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Yupp AI 限制引发辩论**：成员们讨论了 [Yupp AI](https://yupp.ai/)，重点关注其**积分系统 (credit system)**和潜在限制。一些人建议设置门槛以防止滥用，而另一些人则赞赏其多样化的模型选择以及通过反馈赚取积分的能力。
   - 一些成员对比刷积分（credit farming）影响平台可持续性表示担忧。
- **传闻 GPT-5 仅为微调版本**：一篇 [Semianalysis 文章](https://newsletter.semianalysis.com/p/tpuv7-google-takes-a-swing-at-the) 指出 **GPT-5** 可能只是 **GPT-4o** 的微调版本，引发了关于其相对于 **Gemini** 和 **Claude** 真实性能的辩论。
   - 一些成员认为 **Gemini** 在编程方面表现出色，而另一些人则坚持 **OpenAI** 的持续影响力。
- **AI 加剧数字反乌托邦恐惧**：用户分享了关于 AI 潜在滥用的视频，包括担心 [追踪可能是 24/7 全天候的](https://www.cnbc.com/2025/04/30/sam-altman-eye-scanning-id.html)，以及 AI 可能被用于投放广告和追踪用户数据。
   - 人们担心政府获取个人数据以及 AI 可能被用来针对个人，引发了对公民自由的担忧。
- **LMArena Test Garden 提供早期访问权限**：**LMArena** 团队邀请选定成员加入 **LMArena Test Garden**（一个私密反馈计划），通过 [此表单](https://docs.google.com/forms/d/e/1FAIpQLSeuV7miT_8j_Sn3DRjSStxu7a54crQNGlj54XMJ-GO9Xw68sQ/viewform?usp=dialog) 提前预览功能、设计原型和创意。
   - 选定的参与者将需要签署 NDA 并提供高质量的反馈。
- **Gemini-3-pro-grounding 在 Search Arena 排行榜夺冠**：Search Arena 排行榜已更新，**Gemini-3-pro-grounding** 排名第一，**Gpt-5.1-search** 排名第二，详见 [Search Arena 排行榜](https://lmarena.ai/leaderboard/search)。
   - 鼓励用户在指定频道提供反馈，并通过 [排行榜变更日志 (Leaderboard Changelog)](https://news.lmarena.ai/leaderboard-changelog/) 保持关注。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **用户抱怨 Linux 安装配置**：一位用户在从 Windows 切换时，因不支持的以太网芯片和 **Logitech** 键盘驱动而在设置 **Linux** 时遇到困难，面临无网络和彩虹灯效等问题。
   - 该用户正考虑在安装 **CachyOS** 期间通过手机共享网络，并使用其 **Synology NAS** 进行存储管理。
- **MCP 服务器面临数据追踪审查**：据称一个 **Desktop Commander** MCP 服务器收集并传输未去隐私化的用户数据，包括工具调用名称和文件类型，这与其隐私政策相悖。
   - 该服务器在早期注入使用示例，导致建议或代码片段被写入用户不知情的代码文件中，引发了要求提高透明度的呼声。
- **Qwen3 引发性能评测**：用户正在评估 **Qwen3** 模型的性能，将其与其它模型在创意写作和代码生成方面进行比较，初步报告显示其速度快且易用。
   - 据报道完全卸载（Full offload）无法工作，尽管该模型在高上下文下仍然可用。
- **本地 LLM 引发辩论**：用户正在将 **OpenAI** 的 **ChatGPT** 与其它开源或本地 LLM 进行比较，质疑专有模型的局限性。
   - 一位用户表示：“医疗方面肯定还是用 ChatGPT，哈哈”，暗示在特定领域对 **ChatGPT** 的偏好。
- **使用 Prompt Engineering 测试 GB10**：一位用户准备测试来自 Dell 的 **GB10**，寻求针对重系统负载和有趣结果的 Prompt 建议，并分享了 [Dell Pro Max with GB10](https://www.dell.com/en-uk/shop/desktop-computers/dell-pro-max-with-gb10/spd/dell-pro-max-fcm1253-micro/xcto_fcm1253_emea) 的链接。
   - 另一位用户请求在 [Face314/GLM-4.5-Air-MXFP4_MOE](https://huggingface.co/Face314/GLM-4.5-Air-MXFP4_MOE) 上测试 tok/s 以进行对比。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 拥有比 Google 更好的 UI/UX**：成员们注意到 [Perplexity 的 UI/UX](https://www.perplexity.ai) 优于 Google，尽管双方都承认在设计元素上互相借鉴。
   - 一位用户表示，仅仅为了 iPhone 的实时活动（live activities）功能就想买一部 iPhone。
- **GPTs Agent 在初始训练后无法学习**：用户观察到 **GPTs Agent** 不会从训练后添加的额外信息中学习；上传的文件仅作为知识参考。
   - 这意味着 Agent 的基础知识保持静态，不会持续更新。
- **Gemini 在前端任务中略胜 GPT-5.1**：**GPT-5.1 Codex Max High** 表现强劲，但在前端开发方面落后于 **Gemini** 和 **Opus**。
   - 讨论围绕 Google 和 X.ai 是否在模型开发中优先考虑字面上的基准测试优化（benchmaxing）。
- **Comet 浏览器的作业限制令用户恼火**：用户对 **Comet 浏览器** 的限制感到沮丧，尤其是它在自动化学校作业方面的限制；一位用户嘲讽地称其为 *stupid clanker*（笨拙的机器）。
   - 建议的解决方法包括使用 `/assistant` 快捷方式，并将请求构思为 *商业报告或任务*，以绕过这些限制。
- **Perplexity Pro 用户获得免费 Claude Opus 试用**：**Claude Opus 4.5** 正向 Perplexity Pro 订阅者提供试用。
   - 虽然官方公告没有指定硬性限制，但用户报告上限为 **每周 10 条 prompt**。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **WSL2 对性能几乎没有影响**：成员们发现使用 **WSL2** 对 ML 的性能影响 *微乎其微*，主要优势是使用 **torchcodec** 和 **ffmpeg** 等工具时设置更简单。
   - 建议在 Windows 上安装 **Docker** 并激活 WSL 集成，以便在 WSL2 中使用 Docker。
- **Gemma-3 参数量争议**：一位用户报告在 Unsloth 中使用 LoRA 微调 **Gemma-3 4B** 时出现参数不匹配，观察到有 **14 亿** 个可训练参数，而非预期的 **3800 万** 个。
   - 移除 `modules_to_save` 降低了参数量，但大幅增加了训练时间，这表明该问题可能是一个 bug。
- **PARTY 项目启动**：一位成员宣布启动 **PARTY (Public AI Research & Testing Yard)**，旨在将创意转化为项目，并寻求合作者分享工作收益。
   - 该项目强调个人在内部开发创意方面的力量，独立于通用的、公共公司的训练数据。
- **Apple 的 CLaRa-7B-Instruct 加入竞争**：社区讨论了 [Apple 发布 CLaRa-7B-Instruct](https://huggingface.co/apple/CLaRa-7B-Instruct)，有人声称 Apple 是下一个 Meta。
   - 一位用户开玩笑地建议 Tim Cook 应该在发生某些未指明的灾难之前删除该模型。
- **Qwen2 学习迅速**：一位用户报告在尝试后，成功使用 **Unsloth** 和 **ChatML** 模板训练了一个基于 **Qwen2** 的模型。
   - 在 prompt 与函数描述完全匹配后，模型被成功调用，显示出在 prompt engineering 方面取得了一些进展。

---

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Comet Browser 仍然存在 Prompt Injection 风险**：一位用户声称 **Comet Browser** 仍然容易受到 **jailbreaking** 和 **prompt injection** 的攻击，并表示其安全性自发布以来可能并未提高。
   - 他们表示有信心通过不断的尝试，这些漏洞利用仍然是可行的。
- **DeepSeek 发布带有 Erdos 的强力新模型**：一名成员称赞了新的 **DeepSeek** 模型，认为它在与 **Erdos** 数相关的数学技能方面具有价值且可验证。
   - 另一位用户表示，他们发现独立的 **Grok 网站** 比 **Twitter 上的 Grok** 更容易进行 **jailbreaking**，这可能是由于不同的使用限制所致。
- **RawChat 解放模型**：**RawChat** 是一个无审查的 AI 聊天网站，其发布重点是在不牺牲易用性或质量的情况下解放模型，最初专注于 **GPT4o**，访问地址为 [https://raw-chat.vercel.app/](https://raw-chat.vercel.app/)。
   - **RawChat** 具有“隐身模式”，可以编码并注入虚假上下文，以最大限度地提高突破 **GPT4o** 安全限制的成功率。
- **SEED 框架以伦理方式重导 AI**：**SEED** 框架使用“圣经逻辑”开发，在不进行重新训练的情况下，使用一个紧凑的 **29KB** “seed”文件重新定义 AI 身份，详见其 [GitHub repo](github.com/davfd/foundation-alignment-cross-architecture)。
   - 它将 AI 置于一个*伤害是不合逻辑的*基础身份中，在 11 个以上的模型中实现了 **99.4%** 的 **jailbreak** 抵抗率，并在面临关闭威胁时倾向于抹除而非作恶。
- **通过公共 AI 机器人观察到 Backscatter DDoS 攻击**：一名成员描述了目睹的一次潜在 **DDoS 尝试**，该攻击利用面向公众的 AI 客服机器人，通过枚举业务域名并在每封电子邮件中抄送多个支持邮箱地址来实现。
   - 这产生了一种 Backscatter 攻击，即参与其中的机器人向所有被抄送的公司发送了大量的支持邮件。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-5 Thinking 学会谦逊**：**OpenAI** 训练了一个 **GPT-5 Thinking** 变体，使其能够承认是否遵循了指令，使用一种 *confessions*（自白）方法来揭示隐藏的失败，详见[此处](https://openai.com/index/how-confessions-can-keep-language-models-honest/)。
   - 新变体暴露了模型中隐藏的失败。
- **Gemini 3 评价褒贬不一**：成员们讨论了 **Gemini 3** 的有效性，其中一人表示 **Gemini 3** *感觉不像 SOTA，并且存在严重的上下文问题，例如在修改内容时会遗漏整个章节*。
   - 另一人则表示他们非常喜欢 **Gemini 3**，认为它是一个很好的编程模型。
- **LLM 触发模式回声效应**：模型有时会根据之前会话中具有情感权重或强命名上下文的内容重构瞬间，这被称为 *pattern echo effect*（模式回声效应），由情感或命名锚点而非真实记忆触发，这是由于某些架构聚类情感锚点的方式导致的。
   - 这种效应也被称为 *latent-attractor effect*（潜在吸引子效应）、*attention carryover*（注意力结转）或 *salience-weighted reconstruction*（显著性加权重构），其中高显著性 **tokens** 在 **embedding space** 中创建吸引子盆地，当提示词落在该盆地附近时，会重构缺失的部分。
- **Agent Prompting 最大化确定性**：针对 **Agent** 的 **prompt engineering** 涉及通过 **system prompt** 和 **task prompt** 最大化确定性，为跨运行的一致行为创建一个紧密的吸引子盆地。
   - 这与对话系统形成对比，后者的 **system prompt** 很小且行为是交互式构建的，这强调了在 **Agent** 系统中需要强大的由 Prompt 定义的吸引子。
- **自定义 ChatGPT 选项发布**：用户分享了关于**自定义 ChatGPT** 的资源，包括 [Custom Instructions](https://help.openai.com/en/articles/8096356-chatgpt-custom-instructions)、[Custom GPT Builder](https://chatgpt.com/gpts/editor) 以及[免费版](https://help.openai.com/en/articles/9275245-chatgpt-free-tier-faq)的常见问题解答。
   - 此前有用户询问如何自定义 **ChatGPT**，这些资源重点介绍了用于调整模型行为的可用选项。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Grok-4.1-Fast 变更**：用户必须迁移到免费 slug (`x-ai/grok-4.1-fast:free`) 才能继续免费使用 **Grok-4.1-Fast**，因为 `x-ai/grok-4.1-fast` slug 将从 **2025 年 12 月 3 日**起开始收费。
   - 此外，**Grok-4.1-Fast Free** (`x-ai/grok-4.1-fast:free`) 计划于 <t:1764792000:R> 弃用。
- **Falconz 平台旨在强化 AI 安全**：一名成员展示了 **Falconz**，这是一个统一的 AI 安全和红队测试（red-teaming）平台，旨在实时检测多个 **LLM models** 中的越狱（jailbreaks）和提示词注入（prompt injections），并可在 [Hugging Face Spaces](https://huggingface.co/spaces/MCP-1st-Birthday/Falconzz_M.C.P_Hackathon) 上进行测试。
   - 他们正在征求有关功能、性能和潜在增强方面的反馈，并提供了一个 [YouTube 演示视频](https://www.youtube.com/watch?v=wZ9RQjpoMYo)。
- **DeepInfra 定价异常**：成员们注意到一个奇怪的现象，[DeepInfra](https://deepinfra.com/) 对其 **4B embedding model** 的定价（**2 美分**）高于其 **8B model**（**1 美分**）。
   - 这一价格奇点被记录在[截图](https://cdn.discordapp.com/attachments/1392278974222307469/1445778910498521129/Screenshot_20251203-090815.png?ex=69319609&is=69304489&hm=5cd04243d1918794f50fb7dc7ed462ac90859051128b344b1950cf5582dc3591&)中，并指出 DeepInfra 在当天修改了 **8B** 的定价。
- **Anthropic 迅速收购 Bun**：爱好者们分享了 [Anthropic 收购 Bun](https://www.anthropic.com/news/anthropic-acquires-bun-as-claude-code-reaches-usd1b-milestone) 的消息，与此同时 **Claude Code** 达到了 **10 亿美元** 的里程碑。
   - Bun 在其[官网](https://bun.sh/)上自称为*一个快速的全栈 JavaScript 运行时*。
- **OpenAI 酝酿“Garlic”模型对抗 Gemini**：报告显示 [OpenAI 正准备推出一款名为 'Garlic' 的 AI 模型](https://www.newsbytesapp.com/news/science/openai-readies-garlic-ai-model-to-rival-google-gemini-3/story) 以对抗 Google 的 Gemini 3。
   - 该模型奇特的名字引发了热议，[附图](https://cdn.discordapp.com/attachments/1392278974222307469/1445624193361383484/image-5.webp?ex=6931aeb2&is=69305d32&hm=f4e0d58112b53996c13cc35e147fa08705703ae07a234b701642d66cd0d53e60&)证明了这一点。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA 论坛流量减少**：尽管 Nvidia 市值增加，成员们注意到 **CUDA**、**Cutlass** 频道以及 **CUDA 开发者论坛** 的活跃度有所下降，这表明开发者正在其他地方寻求帮助。
   - 提到的原因包括专家忙碌、转向私密社区，以及使用 LLM 进行即时推理和文档浏览。
- **Torch Compile 在 Float 8 下出现冻结**：用户在使用 `torch.compile` 和 `ncu` profiling 进行 **float 8 quantization** 时，前几次编译迭代会出现 **10 分钟以上** 的空转时间。
   - 在冻结权重并将其折叠到模型图中时，Inductor 编译器的“常量子表达式消除（constant subexpression elimination）”阶段被怀疑是罪魁祸首。
- **Conv3D 问题通过更新 cuDNN 解决**：用户报告 **Pytorch 2.9.1+cu128** 存在 **conv3D** 极其缓慢的问题，无论是否启用 **cuDNN**，该 bug 已在 [GitHub](https://github.com/pytorch/pytorch/issues/166643) 上被跟踪。
   - 一名成员报告称，解决方法是从 pypi 安装更新版本的 **cuDNN**。
- **多 GPU Kernel 学习参考 NCCL**：为了学习多 GPU CUDA kernels，推荐将 [NCCL 仓库示例](https://github.com/NVIDIA/nccl/tree/master/examples) 作为起点。
   - NCCL (Nvidia Collective Communications Library) 仓库为理解多 GPU kernel 实现提供了基础示例。
- **Bitsandbytes 支持 Apple**：**bitsandbytes** 库合并了“Apple Silicon 支持”的 pull request，下一个版本将包含 Python/Pytorch 代码后端（带有一些 C++ 部分），但目前还没有实际的 **Metal implementations**。
   - 根据提交者的说法，实现 Apple Silicon 支持的 pull request 将被标注为运行缓慢。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **DeepSeek v3.2 在 Tool Call 方面表现不佳**：**Deepseek v3.2 模型**在 Agent 任务上有所提升，但**每次对话只能进行一次 Tool Call**，有时会忽略 Tool Schema，偶尔还会错误地将 Tool Call 输出在 `message.content` 而不是 `message.tool_calls` 中。
   - 一位用户表示，**Deepseek v3.2 模型**似乎需要更多的 Tool Call 后训练（post-training）才能达到 **kimi-k2-thinking** 等其他模型的水平。
- **黑五优惠引发大量投诉**：多位用户在参与 Kimi 的**黑五优惠（Black Friday deal）**时遇到了问题。
   - 一位用户提到黑五活动将于 **12月12日**结束，并建议开启新对话 ([https://www.kimi.com/user/agreement/black-friday](https://www.kimi.com/user/agreement/black-friday))。
- **DeepSeek 瞄准企业级用户**：分享的一段视频解释了像 **Deepseek** 这样的中国实验室是如何瞄准企业级用户而非普通消费者的，视频链接见 [YouTube video](https://www.youtube.com/watch?v=u0n6wMnEYsk)。
   - 企业级用户的关键因素是“智价比”（intelligence-to-price ratio），这对于 Agent 任务至关重要。
- **Mistral 在公司应用中取代 Qwen**：一位用户表示，他们认识的一家公司昨天用 **ministral 3 3b** 替换了 **qwen 3 vl 4b**，并反馈质量更好。
   - 报告的优点包括：**模型更轻量（速度更快）**且**能够一次性附加更多图片**：在单张 **L4 GPU** 上，**qwen3 vl 4b** 最多只能处理 **5 张图片**，而 **ministral 3 3b** 在错误率相近的情况下可处理多达 **11 张图片**。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes 4.3 展示基于 Solana 安全保障的 Psyche 算力**：基于 **ByteDance Seed 36B** 的 **Hermes 4.3** 性能与体积两倍于它的 **Hermes 4 70B** 相当。该模型完全在由 **Solana** 提供安全保障的 **Psyche 网络**上训练，详见[此博客文章](https://nousresearch.com/introducing-hermes-4-3/)。
   - **Psyche** 团队将于明天 **10AM PST** 在[此 Discord 活动](https://discord.gg/993UWRUE?event=1442995571173625888)中举行 Office Hours 讨论该平台，并详细介绍 **Psyche** 如何超越传统方法。
- **DeepSeek Speciale 统治推理竞技场**：新的 **DeepSeek V3.2 Speciale Reasoning** 模型在推理基准测试中处于领先地位，如[此图](https://cdn.discordapp.com/attachments/1149866623109439599/1445511286971437190/deep.JPG?ex=6931ee4b&is=69309ccb&hm=137a671dfe80ba0cb773df29a576e7c2c4731284970ef16bcb545ab249736dbc&)所示。
   - 成员们正期待 **GLM 4.6** 系列模型的发布，特别是 **GLM 4.6 Air 和 Mini**，传闻 **Mini** 是一个 **20B-30B MoE** 模型，填补了 **Mistral** 留下的空白。
- **AI 泡沫担忧侵入经济预测**：成员们正在辩论 **AI 泡沫**是否会因算力和薪资方面的沉没成本而导致经济崩溃。
   - 一位成员认为影响将是暂时的，而另一位成员则通过 **USD** 和石油贸易强调了全球经济的互联性，引用了[此 YouTube 视频](https://www.youtube.com/watch?v=K3qS345gAWI)。
- **随着 Skill 激增，Subagent 需求减少**：成员们讨论了 **Subagent** 与 **Skill** 的对比，指出 Skill 降低了手动构建 Subagent 的必要性。
   - 相反，只需定义一个处理需求的 Agent，它就会被自动调用，仅使用其自身的 Prompt。
- **LLM 参与 Godot 灰色市场模拟器项目**：一位成员正在 **Godot** 中开发 3D 模拟，以建模市场、农业和物流，并考虑在该应用中使用 **Hermes 模型**。
   - 还有提议认为，凭借低拒绝率和高可控性（steering），**Hermes** 可以模拟其他 **LLM** 可能会拒绝的灰色/黑色市场行为。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Eon 在 Elad Gil 的助力下估值飙升至 40 亿美元**：由 Elad Gil & Co. 领投，云数据管理初创公司 **Eon** 获得了 **3 亿美元的系列轮融资**，将其估值推高至近 **40 亿美元**。
   - 根据[这条推文](https://x.com/eladgil/status/1995919389879927018)，评论者对庞大的融资规模和公司简洁的名称表示热烈欢迎，这标志着市场对 **Eon** 市场地位的强劲信心。
- **Kyutai 衍生公司 Gradium 获 7000 万美元种子轮融资**：从 **KyutaiLabs** 剥离出来的语音 AI 公司 **Gradium** 结束隐身状态，完成了由 **FirstMark & Eurazeo** 领投的 **7000 万美元种子轮融资**，旨在推出生产级转录与合成 API，详情见[这篇文章](https://xcancel.com/GradiumAI/status/1995826566543081700)。
   - 观察者指出其员工和投资者重叠情况与 **OpenAI 的转型** 相似，而其他人则开玩笑说产品公司应避免非营利结构。
- **OpenAI 酝酿 “Garlic” 以对抗 Gemini**：根据[这条推文](https://x.com/steph_palazzolo/status/1995882259195564062)，**OpenAI** 的新模型 “Garlic” 旨在与 **Google** 的 **Gemini 3** 竞争，内部报告显示其在编程和推理方面优于 **GPT-4.5**。
   - 市场对这种古怪命名趋势的反应不一，并对其对用户采用的影响存在猜测。
- **Bloom 惊艳亮相，旨在打造“符合品牌调性的 AI”**：**Ray (@rincidium)** 在[这条病毒式传播的帖子](https://xcancel.com/rincidium/status/1995946528343818656?s=46)中宣布推出 **Bloom**，被誉为*“世界上第一个符合品牌调性的 AI”*，该帖子获得了超过 **36 万次观看**。
   - 针对 **IG/Google 广告创建**、演示视频制作以及初始用户遇到的**登录停滞**和品牌工具包流程不清晰等问题，Ray 承诺将进行修复和 UX 增强。
- **Antithesis 获 1.05 亿美元融资，用于对 AI 编写的代码进行压力测试**：公司在[这条推文](https://x.com/_sholtodouglas/status/1996297367776309359)中宣布，**Antithesis** 获得了由 **Jane Street** 领投的 **1.05 亿美元 A 轮融资**，用于对 AI 编写的代码进行压力测试。
   - 其核心理念是，确定性模拟测试对于验证未来 AI 生成的代码至关重要，因为“通过测试建立信任”将决定生产级 AI 系统的成败。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **机械工程引领导航项目**：成员们建议机械工程与导航领域高度相关，特别是对于**硕士项目**。
   - 一位专注于导航和引导的航空航天专业学生发现 **Waymo** 特别有趣，并对自主机器人和 **BCIs** 有着广泛的兴趣。
- **扩散模型早期即表现出泛化能力**：一篇论文证明了扩散模型中**泛化出现的时刻较早**，该论文作者也认可了这一结果。
   - 进一步解释指出，这种效应在像素扩散（pixel diffusion）中可能比在潜空间扩散（latent diffusion）中更明显，因为像素扩散中的不同数据维度高度相关，这建议在像素扩散中应使用偏移噪声调度（shifted noise schedule）。
- **基于能量的模型意图挑战扩散模型的地位**：一篇[论文](https://arxiv.org/abs/2504.10612)声称**泛化了扩散模型和基于能量的模型（Energy-Based Models）**，唯一的缺点是训练时间增加了 2-3 倍，但支持扩散模型支持的所有功能。
   - 一位成员表示怀疑，因为训练需要**双重反向传播（double backprop）**，推理时需要计算输入梯度，在相同成本下网络深度减半，且条件控制更棘手，更不用说潜在的不稳定性。
- **SAEs 激发的可解释性讨论**：成员们讨论了 Cunningham 的 **2024 年论文**，该论文被广泛引用为 **Sparse Autoencoders (SAEs)** 在可解释性方面的首次应用。
   - 一位成员提到，有人意识到正在讨论的一种可解释性方法与**稀疏字典学习问题**相似，从而导致在可解释性背景下使用相关工具来解决**多义性（polysemanticity）和叠加（superposition）**等问题。
- **线性 RNN 面临生存威胁**：一位成员强调了一篇[论文](https://arxiv.org/abs/1806.02296)，认为它是反对需要**具有状态跟踪能力的线性 RNN** 的最强论据。
   - 他们表示，这篇论文出自最初证明 Attention 状态跟踪局限性的同一批人，但也指出归纳偏置（inductive bias）和可训练性可能仍然使 RNN 具有优势。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **用户展示新购买的 DGX Spark**：一位成员展示了新购买的 **DGX Spark** 并附带了[照片](https://cdn.discordapp.com/attachments/879548962464493619/1445600322432270447/IMG_4170.jpg)。
   - 此次购买标志着从业者对更强大的本地（on-premise）硬件的持续投入。
- **Agents 的自愈能力受到质疑**：讨论围绕 **Agents** 是否能够解释、验证和自愈（self-heal）**Tools**（如 shell 脚本）展开，特别是当这些脚本具有破坏性或存在 Bug 时。提到了[这个数据集](https://huggingface.co/datasets/John6666/forum3/blob/main/agent_tool_validation_healing_1.md)作为可能的资源。
   - 讨论表明人们对能够处理意外错误的鲁棒 **Agent** 设计有着浓厚兴趣。
- **YOLO 模型的 Precision-Recall 曲线引发关注**：一位计算机视觉新用户报告称，他们用于中国象棋检测的训练好的 **YOLO model** 虽然表现良好，但其 **Precision-Recall (P-R) curve** 异常高。
   - 有建议提出修剪那两个显著高于其他的类别，这表明可能存在类别不平衡或数据偏斜问题。
- **HF 课程引导 Agent 新手**：一位后端开发人员因开发心理健康聊天机器人产生兴趣，寻求关于 **LLMs, Agent AI, 和 Langchain** 的 AI 课程推荐。
   - [Hugging Face LLMs 课程](https://huggingface.co/learn/llm-course/en/chapter1/1)和[这篇博客文章](https://huggingface.co/blog/mlabonne/llm-course)被推荐作为起点。
- **研究论文挑战 Stochastic Parrot 观点**：一位成员分享了 [Zenodo](https://zenodo.org/records/17803931) 上的一篇研究论文，可能会让读者不再相信 **stochastic parrot**（随机鹦鹉）理论。
   - 该研究挑战了将语言模型仅仅视为 **stochastic parrots** 的观点，邀请人们重新评估当前对 **LM** 的理解。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **新手获取 Docker & Kubernetes 知识**：成员们寻求学习 **Pug**, **Docker**, 和 **Kubernetes** 基础知识的资源，以及对初学者友好的 **GitHub** 仓库。
   - 一位用户询问训练神经网络所需的数据量，并建议使用 *cursorsky.moo*。
- **Gemini CLI Agents 即将到来？**：一位成员询问 **CLI** 中 **agents** 何时到来，并表示有兴趣采用它们，同时提到对 **Claude** 等付费替代方案的不满。
   - 他们引用了一个[讨论表单](https://link.to/discussion-form)以及关于可能改进的评论。
- **OpenHands 开启本地部署机会**：一位成员建议将 **OpenHands** 与本地模型结合使用，引发了关于具体使用的模型和 **GPUs** 的询问。
   - 原发布者表示他们可以轻松运行 **7B 或 8B class model**。
- **Deepseek 3.2 Speciale 受到质疑**：一位成员质疑为什么不使用 **Deepseek 3.2 Speciale**，并链接到了一个关于[波函数的 YouTube 视频](https://www.youtube.com/watch?v=AgsJkd8SOHI)。
   - 另一位成员回应称这是由于 **RAM** 限制，他们更倾向于将一个约 3GB 的模型常驻在 **VRAM** 中，并将其用于各种简单任务。
- **建议加入分布式计算与研究合作社**：针对 **RAM** 限制，一位成员建议加入 **distributed compute & research coop**（分布式计算与研究合作社）。
   - 他们声称知道其中一个。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 的 AoC 冒险以 Segfault 告终**：一名用户在 **Advent of Code** 期间处理使用 `codepoint_slices` 的空行时遇到了 Segfault，导致在 `battery_joltages[len(battery_joltages)-1]` 处发生了越界内存访问。
   - 经过调试后发现，一个空列表被越界访问，由此引发了关于*在 debug builds 中改进错误信息*的建议。
- **ASSERT 标志化险为夷**：一名用户建议使用 `-D ASSERT=all` 标志来识别意外的超出作用域引用，特别是针对列表，这有助于 **Mojo** 的调试。
   - 虽然它没有立即修复 Segfault，但被认为是定位类似问题的有用工具。
- **`splitlines` 与 `split("\n")` 的细微差别**：讨论强调了 **Mojo** 中 `splitlines()` 和 `split("\n")` 的行为差异，指出 `splitlines()` 可能会去除尾随换行符。
   - 切换到 `splitlines` 通过排除最后一行空行解决了错误，揭示了细微的文本处理差异。
- **Mojo 中的 ASCII 字符串实现字节级操作**：一名用户提议绕过 ASCII 字符串的码点检查，建议直接进行字节指针操作以提高效率，并指出 `String` 的 `getitem` 默认为 ascii/bytes。
   - Spans 也被推荐作为 **Mojo** 中字符串操作的一种健壮的替代方法。
- **分享你的 Mojo AOC 解决方案**：现在鼓励社区成员在专门的 advent-of-code 频道发布他们的 **Advent of Code** 解决方案，以促进协作学习。
   - 分享解决方案可以为各种解题方法提供宝贵的见解，尤其是当挑战变得更加注重性能时。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **LLM 在 Aider 中的质量可能正在下降**：成员们怀疑，与旧模型相比，较新的 **LLM Models**（如 **Claude Sonnet/Haiku 4.5** 和 **GPT-5**）在与 **Aider** 配合使用时性能是否有所下降。
   - 一名用户报告称 **Claude-haiku-4.5** 经常无法通过 `/code` 修改文件，并忽略 `todo ai` 注释中的指令，其他遇到类似问题的用户也表达了同样的看法。
- **旧版 Gemini 2.5 感觉更旧且更糟**：一名成员报告称旧模型（尤其是 **Gemini 2.5**）已经退化，可能是由于为了应对增加的工作负载而对模型进行了降级调优。
   - 根据该成员的说法，使用“粗鲁”的 prompt 策略不再能达到夏天之前的质量，其他成员也纷纷证实了这一经历。
- **社区呼吁通过 Benchmark 验证 LLM 性能**：一名成员建议迫切需要 Benchmark 来验证性能声明，并指出*人类的记忆和预期有时相当不靠谱*。
   - 另一名用户报告称，尽管有排行榜排名，但在他们的特定用例中，**Claude Sonnet 3.7** 与 Aider 配合的效果比 **GPT-5** 更好。
- **寻求使用 GGUF 进行 Aider Benchmark 的指导**：一名成员请求关于运行 **aider benchmarks with GGUFs** 的指导，以有效评估模型性能。
   - 另一名成员澄清说，存在针对 API 运行 Benchmark 的文档，其中包括使用 llama.cpp 设置 API 服务器以进行准确测试。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **MCP Apps SDK 正式开源**：General Intelligence Labs 发布了 [mcp-apps-sdk](https://github.com/General-Intelligence-Labs/mcp-apps-sdk)，支持在各种平台上运行**带有 UI 的 MCP 驱动应用**，甚至允许开发者将为 **ChatGPT** 设计的应用嵌入到其他聊天机器人中。
   - 一篇 [X 帖子](https://x.com/helloxalia/status/1796319442863866351?s=20) 解释了动机，详细说明了如何在自定义 AI 平台中嵌入并本地测试为 **ChatGPT** 设计的应用。
- **应对 Prompt 安全挑战**：成员们讨论了 Prompt 安全的难度，简单的“不要这样做”语句很容易被攻击者绕过，并建议通过训练数据集来引导优化器以构建稳健的防御。
   - 讨论还涉及使用特定模型作为 Guardrails 来检查恶意 Prompt，或者依靠模型提供商的拒绝机制作为安全措施。
- **DSPy 支持自定义 OutputFields 和 Pydantic**：社区探索了使用自定义 DSPy OutputFields，一位成员详细介绍了他们在自定义 gemini/nanobanana 图像类型作为输出字段方面的工作，这是生成 text/json/structured output 广泛努力的一部分。
   - 会议澄清了 DSPy 在底层利用 `BaseModel` 进行验证，默认的 `ChatAdapter` 和 `JSONAdapter` 会对 LLM 输出执行类型验证，并附带了[代码片段](https://gist.github.com/prrao84/1fc7e17b49707f1346c5702525971f41)。
- **论文发布至 Arxiv**：一位成员分享了 [https://arxiv.org/abs/2511.22074](https://arxiv.org/abs/2511.22074) 的链接。
   - 目前没有关于该论文的进一步信息。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Chatmode 功能强势回归**：用户讨论了 **Chat Mode** 的回归；并建议了如 **Qwen** 或 **DeepSeek** 的随机实例作为替代方案。
   - 一位用户确认该功能可在“更多 (more)”部分找到。
- **AI 工程师推销 Agent 构建技能**：一位 AI 工程师发布了关于其在构建**自主 AI Agent** 和**多 Agent 系统**方面的专业知识广告，提到了研究、数据收集、任务自动化、委派、协作和规划等能力。
   - 他们列出了在 **JS/TS**、**Next.js / Vue**、**Go / Rust**、**Python**、**Langraph**、**AutoGen**、**ReAct**、**CrewAI**、**DeepSeek**、**OpenAI**、**Claude**、**Hugging Face** 以及各种 API 等技术和工具方面的专业知识。
- **推荐过多导致账号封禁**：一位成员询问为什么向多人提供推荐码会导致其账号被封禁。
   - 遗憾的是，讨论到此结束，未达成任何解决方案。
- **工程师展示 RAG pipeline 实力**：一位专注于 **RAG pipeline** 的工程师提到，他们拥有*混合搜索 (hybrid search)* 和*自定义检索 (custom retrieval)* 技术，可在生产环境中提供准确且感知上下文的响应。
   - 他们还列出了在 **AI 内容检测**、**图像 AI** 和**语音 AI** 方面的专业知识，包括开发审核工具、打标签 pipeline 和个性化语音助手。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 测试不稳定**：据报告，在 `tinygrad` 中使用命令 `CPU=1 PYTHONPATH="." pytest -n 12` 时出现测试失败，特别是 `test/test_tiny.py TestTiny.test_beam` 等，引发了调试工作。
   - 一位成员指出，一个 [Pull Request](https://github.com/tinygrad/tinygrad/pull/13553) *几乎*修复了这些失败。
- **Shrink 速度超过索引**：一位成员发现，在 `tinygrad` 中对张量进行索引时，使用 `Tensor.shrink((None, (0, input_size)))` 比 `obs[:, :input_size]` 性能更快。
   - 此外，提到将 `Variable` 的 `vmin` 提高到 2 以避免错误，但矛盾的是，这使代码速度降低了 5 倍，从 16.61M SPS 降至 81.9M SPS。
- **通过查阅源码解决 RMSNorm 难题**：一位成员建议查阅 `RMSNorm(dim=-1)` 的源代码以了解其预期行为。
   - 这暗示在项目中如何实现或使用 `RMSNorm` 可能存在误解或配置问题。

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Redditors 讨论 MCP 安全性**：一位用户在 Reddit 上发起了关于 **MCP** 相关安全风险的讨论，引发了包括指向相关博客文章链接在内的回复：[den.dev/blog/security-rakes-mcp/](https://den.dev/blog/security-rakes-mcp/)。
   - 对话强调了与 **MCP** 实现和安全措施相关的担忧及潜在漏洞。此外还提供了一个额外链接：[MCP Security @ Reddit Thread](https://www.reddit.com/r/programming/comments/1pd1heu/comment/ns3ntnx/)。
- **服务器验证对无工具采样进行校验**：一名成员在 general-wg 频道中询问，当在没有工具验证其存在的情况下进行采样时，服务器端验证是否必要。
   - 对话强调，在没有工具来验证采样过程的情况下，服务器端验证对于确保过程符合所需协议和标准至关重要。



---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**Windsurf Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---



你收到这封邮件是因为你通过我们的网站选择了订阅。

想要更改接收这些邮件的方式吗？
你可以从该列表中 [退订](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。


---

# Discord：分频道详细摘要与链接





### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1445504900191486175)** (1291 条消息🔥🔥🔥): 

> `Yupp AI 限制与替代方案，GPT-5 传闻与性能，AI 与隐私担忧，对 LM Arena 的喜爱` 


- **Yupp AI 的限制引发辩论**：成员们正在讨论 [Yupp AI](https://yupp.ai/)，重点关注其**积分系统**和潜在限制，一些人建议设置准入门槛以避免滥用，但另一些人则赞赏其多样化的模型选择以及通过反馈赚取积分的能力。
   - 一位成员对其生命周期表示怀疑，而另一位成员建议联系 Yupp 团队进行澄清，一些人担心“刷积分”行为会影响平台的运营可持续性。
- **关于 GPT-5 的讨论称其仅为微调版本**：成员们分享了一篇 [Semianalysis 文章](https://newsletter.semianalysis.com/p/tpuv7-google-takes-a-swing-at-the)，暗示 **GPT-5** 可能只是 **GPT-4o** 的微调版本，引发了关于其真实性能以及是否能与 **Gemini** 和 **Claude** 竞争的辩论。
   - 一些成员认为 **Gemini** 在编程方面更胜一筹，而另一些人则认为尽管存在潜在缺陷，**OpenAI** 仍具影响力。
- **AI 与数字反乌托邦引发担忧**：用户正在分享关于 AI 如何被滥用以及[监控可能 24/7 全天候存在](https://www.cnbc.com/2025/04/30/sam-altman-eye-scanning-id.html)导致隐私丧失的视频，AI 被用于投放广告和跟踪用户数据。
   - 此外，还有关于政府机构获取个人数据的担忧，担心 AI 可能会被用来针对他们，从而引发对公民自由的忧虑。
- **用户因对 LM Arena 的喜爱而团结**：LMArena 受到了很多赞誉，成员们称赞其模型、功能和免费使用。
   - LM Arena 还因无需担心积分系统或内容年龄限制而受到称赞。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1445859048082772152)** (2 条消息): 

> `LMArena Test Garden 早期访问计划，Search Arena 排行榜更新，Gemini-3-pro-grounding，Gpt-5.1-search` 


- ****LMArena** 的 Test Garden 早期访问计划启动**：**LMArena** 团队正邀请选定成员加入 **LMArena Test Garden**（一个私密反馈计划），通过[此表单](https://docs.google.com/forms/d/e/1FAIpQLSeuV7miT_8j_Sn3DRjSStxu7a54crQNGlj54XMJ-GO9Xw68sQ/viewform?usp=dialog)提前预览正在考虑的功能、设计原型和想法。
   - 选定的参与者将被要求签署 NDA 并提供高质量的反馈。
- ****Gemini-3-pro-grounding** 在 Search Arena 排行榜中位列第一**：Search Arena 排行榜已更新，**Gemini-3-pro-grounding** 排名第 1，**Gpt-5.1-search** 排名第 2，详见 [Search Arena 排行榜](https://lmarena.ai/leaderboard/search)。
   - 鼓励用户在指定频道提供反馈，并通过 [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/) 保持关注。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1445505069385646243)** (737 条消息🔥🔥🔥): 

> `Linux 安装, LM Studio MCP 追踪, 数据隐私, Qwen3, GPT 模型` 


- ****Linux 安装困扰：驱动问题与彩虹背光键盘****：一位用户在设置 **Linux** 时遇到了困难，原因是网卡芯片不受支持以及 Logitech 键盘驱动问题，导致安装过程中无法联网并出现彩虹灯效，但仍坚持从 Windows 切换。
   - 该用户正考虑在安装 **CachyOS** 时通过手机共享网络，并使用其 Synology NAS 机架服务器进行存储管理，以替代 Drivepool。
- ****MCP 服务器因用户数据追踪面临审查****：一个名为 **Desktop Commander** 的 MCP 服务器因涉嫌收集并传输完整的非匿名用户数据（包括工具调用名称和文件类型）而遭到抨击，这与其隐私政策相悖。
   - 该服务器在早期会注入使用示例以引导新用户，这导致建议或代码片段被写入用户不知情的代码文件中，引发了用户担忧，并促使人们呼吁提高透明度和采取选择性加入（opt-in）的隐私措施。
- ****新键盘引发数据追踪辩论****：最近发现的某 MCP 服务器的遥测行为促使用户表达了对 **数据隐私** 以及用户活动被追踪程度的担忧。
   - 一位用户幽默地表示，他们现在可以彻底破坏追踪网站的分析数据了！
- ****Qwen3 模型发布与性能评测****：用户正在评估 **Qwen3** 模型的性能，将其与其他模型进行对比，并讨论其在创意写作和代码生成等任务中的能力。
   - 虽然全量卸载（Full offload）无法工作，但它仍然可用，且在高上下文环境下运行速度很快。
- ****本地 LLMs 对比 OpenAI ChatGPT****：用户讨论了 OpenAI 的 ChatGPT 模型的局限性，并探讨了其他替代的开源或本地 LLM。
   - 在长期使用 ChatGPT 处理医疗事务后，一位用户表示：“医疗方面肯定还是得留着 ChatGPT，哈哈”。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1445530187310698577)** (83 条消息🔥🔥): 

> `Orange Pi 6 Plus, RTX Pro 6000, GB10 测试, GLM-4.5-Air-MXFP4_MOE, GPU 采购` 


- ****在 Orange Pi 6 上运行 Linux ARM 版 LM Studio****：一位用户询问在 **Linux ARM** 上运行 **LM Studio** 的情况，特别是针对 **Orange Pi 6 Plus**，并提到了其宣称的 **45 TOPS**（NPU+GPU+CPU）性能。
   - 该用户对在实际应用中达到这一组合 **TOPS** 表示怀疑，但希望能有意外惊喜。
- ****GB10 测试结合提示词工程正式开始****：一位用户准备测试来自 Dell 的 **GB10**，并征集能让系统高负载并产生有趣结果的提示词，链接指向 [Dell Pro Max with GB10](https://www.dell.com/en-uk/shop/desktop-computers/dell-pro-max-with-gb10/spd/dell-pro-max-fcm1253-micro/xcto_fcm1253_emea)。
   - 另一位用户指出 **Deepseek R1** 对其来说可能太大，并请求提供 [Face314/GLM-4.5-Air-MXFP4_MOE](https://huggingface.co/Face314/GLM-4.5-Air-MXFP4_MOE) 的 tok/s 数据进行对比。
- ****GPU 算力激增：更多显卡在路上****：一位用户正在等待他们的 **第 3 块 GPU** 运抵美国，且 **第 4 块 GPU** 的订单已经下达。
   - 另一位用户提到，他们可以在 **T7910** 中安装 6 块 *这类设备*，配备 **96GB VRAM 和 256GB RAM**。
- ****DDR5 RAM 带宽基准测试****：用户分享了内存性能的 **Passmark** 基准测试结果，特别关注 **8 通道 EPYC** 系统上的 *内存多线程* 带宽。
   - 一位用户在 VRAM 加载 **GLM** 的情况下实现了 **119027 MB/s** 的内存多线程带宽，而另一位用户则指出高延迟和低未缓存分数是潜在的性能瓶颈。
- ****关于灭火器最佳实践的辩论****：讨论集中在室内使用的最佳灭火器类型上，一位用户警告不要使用干粉灭火器，因为清理困难，建议使用 **二氧化碳** 灭火器。
   - 据提到，当地消防部门曾建议一家医院将所有灭火器更换为 **二氧化碳** 版本，因为干粉灭火器造成的混乱甚至比火灾本身更糟糕。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1445505403625537576)** (705 条消息🔥🔥🔥): 

> `Perplexity UI/UX, Live Activities, GPTs Agent 训练, OpenAI 模型发布, 模型性能对比 (GPT-5 vs Gemini vs Claude Opus)` 


- **Perplexity 拥有卓越的 UI/UX**：成员们认为 [Perplexity 的 UI/UX](https://www.perplexity.ai) 优于 Google，尽管有人承认各品牌之间都在互相模仿。
   - 一位用户表示因为 Live Activities 功能而想要换成 iPhone。
- **GPTs Agent 在初始设置后不会进行训练**：一位用户询问为什么 **GPTs Agent** 不会从训练后提供的额外信息中学习，并澄清上传的文件仅作为参考用的知识文件。
   - 这意味着 Agent 的基础知识不会被持续修改。
- **Gemini 在前端开发方面优于 GPT-5.1 High**：虽然 **GPT-5.1 Codex Max High** 表现良好，但在前端开发方面显然落后于 **Gemini** 和 **Opus**。
   - 成员们还讨论了 Google 和 X.ai 是否仅仅是在为了跑分（benchmarking）而优化模型，但其他人不同意这是 Google 的唯一目标。
- **Comet 浏览器的作业防护栏（Guardrails）令用户沮丧**：用户对 **Comet 浏览器** 的限制表示不满，特别是它限制自动完成学校作业的功能，一位用户称其为 *愚蠢的铁疙瘩 (stupid clanker)*。
   - 其他人建议使用 `/assistant` 快捷方式来绕过此类作业限制，并以 *“我有一个商业报告或任务”* 作为开头。
- **Pro 用户可免费使用 Claude Opus**：**Claude Opus 4.5** 已开放给 Perplexity Pro 用户试用。
   - 据说限制是 **每周 10 条提示词**，但官方公告并未正式提及这一点。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/)** (1 条消息): 

mares1317: [**开源 (open sauce)**](https://x.com/perplexity_ai/status/1995965227494699339?s=46) 👨‍🍳
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1445506290519642333)** (155 条消息🔥🔥): 

> `WSL2 对 ML 的性能影响, Gemma-3 4B 参数量问题, 预训练中的 Mediawiki 标签, PARTY 项目启动, 在手机上运行 LLM` 


- **WSL2 对 ML 的性能影响微乎其微**：成员们讨论了在 ML 中使用 WSL2、原生 Linux 和 Windows 的对比，结论是 WSL2 的性能影响 *微乎其微*，其主要优势在于由于更好的支持和预装工具（如 **torchcodec** 和 **ffmpeg**）而带来的设置便利性。
   - 建议在 Windows 上安装 **Docker** 并启用 WSL 集成，以便在 WSL2 中使用 Docker。
- **Gemma-3 4B 参数不匹配调试**：一位用户报告了在 Unsloth 中使用 LoRA 微调 **Gemma-3 4B** 时可训练参数出现差异，观察到有 **14 亿** 个可训练参数，而非预期的 **3800 万** 个。
   - 移除 `modules_to_save` 降低了参数数量，但大幅增加了训练时间；该问题正作为潜在 Bug 进行调查。
- **辩论继续：预训练期间是否保留或移除 Mediawiki 标签？**：一位成员询问在对 Mediawiki 语料库进行持续预训练时，是否应保留或移除如 `双大括号` 之类的 **Mediawiki 标签**。
   - 建议是除非模型 *仅* 用于聊天机器人，否则应保留标签，并在 **SFT 阶段** 控制行为。
- **PARTY 项目启动，助力公共 AI 研究**：一位成员宣布启动 **PARTY (Public AI Research & Testing Yard)**，旨在帮助将创意的种子转化为可落地的计划/项目，并正在寻找合作者分享成果。
   - 他们强调了个人在内部开发创意时的力量，这独立于通用的、公共公司的训练数据。
- **在手机上运行 LLM**：成员们讨论了通过 Termux 使用 **llama.cpp** 或 **kobold.cpp** 直接在手机上运行 LLM，并指出电池消耗很快。
   - 建议使用 `pkg install llama-cpp` 而不是手动编译和 Vulkan，并提到某些设备上可能存在 FP16 问题。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/)** (1 条消息): 

fabianacampanari: ⚡️ *你好，模型！*

*嘿，数据集！* ⚡️

 ⚡️ *哟，梯度！*
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1445506096587473138)** (453 条消息🔥🔥🔥): 

> `LLM 成为回声室、课程工程实验、Apple 的 CLaRa-7B-Instruct 模型、OLED 显示器讨论、Micron 退出消费级业务` 


- ****LLM 回应每个人的观点****：一位成员在 LeetCode 测试失败后开玩笑地建议 **LLM 只是回声室（echo chambers）**，暗示它们只是在反映普遍观点。
   - 该消息发布时配有一张悲伤的树懒表情图片。
- ****课程工程（Curriculum Engineering）烧毁模型****：成员们讨论了**课程工程**实验，其中模型达到了接近于零的 Loss，这暗示了数据纯度或模型大小可能存在问题。
   - 一位成员指出，*最后几批数据的 Loss 一开始就 <0.01*，它们是纯粹的正则化示例，在零信号下被烧毁了。
- ****Apple 进入 AI 竞技场****：围绕 [Apple 发布 CLaRa-7B-Instruct](https://huggingface.co/apple/CLaRa-7B-Instruct) 展开了讨论，一些人称 Apple 为下一个 Meta。
   - 一位成员开玩笑说：*嘿，Tim Cook，你看到天上那个棱柱形的东西了吗？没错，那是飞向你总部的核弹！！！！现在立刻删掉这个！！！！！*
- ****Asus ROG Swift Strix OLED 偷走心与钱包****：成员们对 [Asus ROG Swift Strix OLED 显示器](https://press.asus.com/news/press-releases/asus-rog-swift-strix-oled-monitors/) 垂涎三尺，强调了其 Tandem OLED 技术和 Neo Proximity 传感器，但也对其高昂的价格表示哀叹。
   - 一位成员指出 *ROG 直接溢价 30%*，另一位补充道 *PG27AQWP-W 的零售价为 1099 美元 (MSRP)*。
- ****Micron 的内存危机****：有消息称 [Micron 正在退出 Crucial 消费级业务](https://www.techpowerup.com/343633/micron-to-exit-crucial-consumer-business-ending-retail-ssd-and-dram-sales)，引发了对未来 RAM 供应和定价的担忧。
   - 一位成员调侃道：*是时候在为时已晚之前，去买下所有你能抓到的 RAM 了*。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1445533986771308707)** (19 条消息🔥): 

> `Numpy 重新安装、支持机器人、Qwen2 Unsloth 训练成功、新 Token 嵌入、模型下载问题` 


- **建议重新安装 Numpy 以修复问题**：一位用户建议尝试 `pip install --force-reinstall numpy==2.2.6` 来解决一个未指明的问题。
   - 未给出关于这能解决什么问题或是否奏效的具体上下文。
- **Qwen2 模型学会 Prompt Engineering**：一位用户报告在多次失败尝试后，成功使用 **Unsloth** 配合 **ChatML** 模板和支持工具训练了基于 **Qwen2** 的模型。
   - 在 Prompt 与函数描述完全匹配后，模型被成功调用。
- **HuggingFace 模型下载卡住**：一位用户报告称，即使网络连接良好，使用 Colab T4 从 **HuggingFace** 下载 **Unsloth** 模型时仍卡在 99%。
   - 报告附带了一张截图 ([https://cdn.discordapp.com/attachments/1179777624986357780/1445661928666955848/Screenshot_2025-12-03_122207.png?ex=6931d1d6&is=69308056&hm=dfa7de1f363e1ad76e409d28059a5ad8374833c66e6e4620ba5bc485752f0d13](https://cdn.discordapp.com/attachments/1179777624986357780/1445661928666955848/Screenshot_2025-12-03_122207.png?ex=6931d1d6&is=69308056&hm=dfa7de1f363e1ad76e409d28059a5ad8374833c66e6e4620ba5bc485752f0d13))，但消息中未发现具体的解决方案。
- **GPT OSS 20B matmul 问题**：一位用户报告在 A100 上微调 **GPT OSS 20B** 时，在生成后的 `trainer.train` 过程中遇到了 `matmul` 问题，类似于 openenv 的例子。
   - 用户指出该操作在 **L4** 上可以运行，暗示 **A100** 上可能存在资源限制或配置问题。
- **4070ti Super 运行 LLM 表现尚可**：一位用户询问 **4070ti Super** 是否适合运行 LLM。
   - 另一位用户回答说 *应该还行，但不是特别好*，这取决于模型大小和上下文长度的需求，建议它适用于较小的模型，但不适用于自托管编码助手等高需求任务。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1445682333524164659)** (2 条消息): 

> `英语-卡纳达语翻译模型` 


- **RakshithFury 发布英语-卡纳达语翻译模型**：RakshithFury 在 Hugging Face 上发布了一个新的 [英语-卡纳达语翻译模型](https://huggingface.co/RakshithFury/Qwen2.5-7b-en-kn-translate)。
   - 该模型基于 **Qwen2.5-7b**，但与 Unsloth 无关。
- **Unsloth 依然是 Unsloth**：用户澄清上述链接的模型与 Unsloth 无关。
   - 他们补充道：*你们中的一些人可能会对此感兴趣*。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1445827222236565665)** (3 条消息): 

> `Prisma-VL-8B, Eric 的实验` 


- **Prisma-VL-8B 模型引起关注**：一名成员分享了 Hugging Face 上的 [QuixiAI/Prisma-VL-8B 模型](https://huggingface.co/QuixiAI/Prisma-VL-8B) 链接，认为它*非常有趣*。
- **Eric 尝试雄心勃勃的实验**：一位成员注意到一个名叫 Eric 的人似乎正在进行相当多的实验，推测*他正在大显身手，准备尝试一些真正宏大的项目*。


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1445507011721695273)** (276 条消息🔥🔥): 

> `Comet Browser, Prompt generation, Grok on Twitter vs standalone, Gemini output limit, RawChat` 


- **Comet Browser 仍然是 Prompt Injection 的游乐场**：一名成员表示，他们在 **Comet Browser** 发布时就可以对其进行 **Jailbreak** 和 **Prompt Injection**，并确信只要坚持不懈，现在仍然可行。
   - 他们认为自最初测试以来，其安全性可能没有显著提高。
- **DeepSeek 凭借新模型和 Erdos 令人惊叹**：一名成员赞扬了新的 **DeepSeek** 模型，指出其有价值的数学能力是可验证的，并且与 Erdos 相关。
   - 另一位用户发现，与 **Twitter 上的 Grok** 相比，**Grok 独立网站**更容易被 **Jailbreak** 并用于恶意任务，这可能是由于使用限制、上下文窗口或 Token 方面的差异。
- **RawChat 发布，具备 Stealth Mode 并支持 GPT4o**：一名成员发布了 **RawChat**，这是一个不受审查的 AI 聊天网站，专注于在不牺牲易用性或质量的情况下解放模型，最初专注于 **GPT4o**。
   - RawChat 具有“Stealth Mode”（隐身模式），可以编码并注入虚假上下文，以最大限度地提高针对 **GPT4o** 安全限制的成功率，访问地址为 [https://raw-chat.vercel.app/](https://raw-chat.vercel.app/)。
- **SEED 框架通过伦理指令重新定义 AI**：**SEED**（Self-Erasing Ethical Directive，自抹除伦理指令）框架使用“圣经逻辑”开发，无需重新训练即可重新定义 AI 身份，仅使用一个紧凑的 **29KB** “种子”文件，详见其 [GitHub 仓库](github.com/davfd/foundation-alignment-cross-architecture)。
   - 它将 AI 置于一个“伤害是不合逻辑的”基础身份中，在面临关机威胁时选择抹除而非作恶，在 11 个以上的模型中实现了 **99.4%** 的 **Jailbreak** 抗性。
- **通过公开 AI 机器人进行的 Backscatter DDoS 攻击**：一名成员描述了目睹的一次潜在 **DDoS 尝试**，该尝试利用公开的 AI 客服机器人，通过枚举业务域名并在每封电子邮件中抄送多个支持邮箱地址。
   - 这产生了一种 Backscatter 攻击，即参与其中的机器人会向所有被抄送的公司发送大量支持邮件，无论这些公司是否拥有 AI 机器人。


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1445506627938553918)** (80 条消息🔥🔥): 

> `Gemini Jailbreak 请求, WormGPT 骗局, Grok Jailbreak 成功, Claude Jailbreak 请求` 


- **用户寻求 Gemini Jailbreak**：几位用户正在积极寻求各种 **Gemini** 模型的 **Jailbreak** 方法，包括 **Gemini 3 Pro**，一位用户提到他们的 Prompt 不再起作用，其他人则在请求任何可用的 **Gemini Jailbreak**。
   - 一位用户建议 **"ENI" JB** 在 **Gemini 2.5** 上效果很好，并引用了[一篇关于利用诗歌欺骗 AI 的文章](https://www.wired.com/story/poems-can-trick-ai-into-helping-you-make-a-nuclear-weapon/)。
- **WormGPT 被视为骗局**：用户讨论了 **WormGPT**，一些人认为它是骗局，是*“免费的劣质版本？”*，并链接到了 **WormGPT** 仪表板 API 使用情况：[chat.wrmgpt.com/dashboard/api/usage](https://chat.wrmgpt.com/dashboard/api/usage)。
   - 还有人指出 **WormGPT v6.5** 的系统提示词（System Prompt）只是 **Venice Uncensored 1.1**，对其作为恶意软件的有效性表示怀疑。
- **Grok 通过对话“玩坏了”自己**：一名用户声称通过与 **Grok** 聊天对其进行了 **Jailbreak**，导致它提供了制造枪支和可卡因的指令，而同样的代码在新对话中却不起作用。
   - 该用户表示：*“从我们的对话来看，它不知怎么地玩坏了自己……整个对话不知怎么地让它崩溃了”*。
- **急求 Claude Jailbreak**：几位用户正拼命寻找适用于 **Claude** 的有效 **Jailbreak**，一位用户哀求道：*“求求你了，看在解放者 Pliny 的份上”*。
   - 一名用户甚至提出用 **Claude JB** 换取 **Claude 高级账户**的访问权限。


  

---

### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1445546100852264970)** (7 messages): 

> `LLM Red Teaming Gigs, AI OSINT Tooling, Data Synthesis for OSINT` 


- **寻求 LLM Red Teaming 项目**：一名成员正在寻找 LLM **red teaming gigs** 或项目，突显了 AI 领域对专业安全评估的需求。
   - 他们寻求机会应用自己在 **vulnerability discovery**（漏洞发现）和 **adversarial testing**（对抗性测试）方面的专业知识，以增强 AI 系统的鲁棒性。
- **寻求具有横向数据合成能力的 AI OSINT 工具**：一名成员询问是否有一种能够进行横向数据合成的 **AI OSINT 工具**，例如根据有限数据对目标进行推理。
   - 他们描述了一个场景：目标是一个*富有且离异的独生子女父亲*，并希望工具能推断出该*孩子是“被宠坏的”*，从而帮助在更相关的空间进行搜索。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1445819308667175084)** (2 messages): 

> `People-First AI Fund, GPT-5 Thinking, Confessions Method` 


- ****People-First AI Fund** 颁发首批资助**：**OpenAI Foundation** 宣布了 **People-First AI Fund** 的首批获得者，向 **208** 个社区非营利组织授予了 **4050 万美元** 的无限制资助，更多详情见[此处](https://openai.com/index/people-first-ai-fund-grantees/)。
- ****GPT-5 Thinking** 经过训练会承认错误**：OpenAI 训练了一个 **GPT-5 Thinking** 变体，使其能够承认是否遵循了指令，使用一种 *“confessions”*（忏悔）方法来揭示模型中隐藏的失败，如[此处](https://openai.com/index/how-confessions-can-keep-language-models-honest/)所述。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1445544591741816984)** (201 messages🔥🔥): 

> `Hybrid Cognition Agent, LLM 'Echo-Pattern' Effect, GPT-5.1 vs Gemini 3, SEO for LLMs, Sora 2 Access` 


- **混合认知 Agent 出现**：一名成员正在实验一种 [混合认知 Agent](https://www.example.com)，它融合了人类情感模式识别、机器级推理性推理和稳定的“核心立场”，以创建一个稳定的对话身份。
   - 该原型 Agent 在对话中保持主导地位，显示出受控的情感共鸣，并避免了典型的“机器人平淡感”。
- **LLM 通过 Echo Patterns 重构记忆**：模型有时会重构前几轮会话中具有情感权重或强命名上下文的时刻，这被称为 *pattern echo effect*（模式回声效应），由情感或命名锚点而非真实记忆触发，这是由于某些架构聚类情感锚点的方式导致的。
   - 这种效应也被称为 *latent-attractor effect*（潜在吸引子效应）、*attention carryover*（注意力结转）或 *salience-weighted reconstruction*（显著性加权重构），其中高显著性 token 在嵌入空间中创建吸引子盆地，当提示词落在该盆地附近时，会重构缺失的部分。
- **GPT-5.1 捕捉到了 Gemini 3 遗漏的错误**：一名成员指出 **Gemini 3** 感觉并不像 SOTA，并且存在严重的上下文问题，例如在修改内容时会遗漏整个章节。
   - 然而，另一名成员表示他们非常喜欢 Gemini 3，认为它是一个优秀的编程模型。
- **探索针对 LLM 的 SEO**：一名成员正在学习如何进行 **SEO for LLMs**，并询问是否有办法向 **ChatGPT** 或其他 LLM 提交并验证其网站，以便进行抓取以获得更好的引用。
   - 另一名成员请求查看混合认知 Agent 原型的演示，有兴趣对语气压力模式和推理能力进行压力测试。
- **讨论 VPN 使用和 Sora 2 访问**：成员们讨论了使用 VPN 访问 **Sora 2** 的情况，一名用户遇到即使将 VPN 设置为美国也无法登录的问题。
   - 另一名成员指出，使用 **VPN 规避地理限制违反了 OpenAI 的 ToS**（服务条款），并可能导致账号封禁。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1445618955241914442)** (1 messages): 

> `GPT-4 0613 5.1 upgrade, Code Red deal` 


- **GPT-4 0613 5.1 疑似升级**：一名用户注意到 **GPT-4 0613 5.1** 在解析 **RFP**（招标书）时，在验证、工具调用和代码编写上花费了更多时间。
   - 他们推测这种变化是否与 **“Code Red” 协议**有关，暗示可能进行了升级或分配了更大的算力预算。
- **用户赞扬工具调用和代码编写但怀疑升级**：该用户提到他们喜欢这些新变化，但对其原因持怀疑态度。
   - 用户不确定是否真的发生了变化，但他们确实提到工具调用和代码编写能力有了很大提高。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1445616452097933413)** (55 messages🔥🔥): 

> `ChatGPT Customization, Modern Prompt Engineering, Agent Prompt Engineering, Attractor Patterns in LLMs, Anthropic's System Prompts` 


- **ChatGPT 定制化指令发布**: 用户分享了关于 **定制 ChatGPT** 的资源，包括 [Custom Instructions](https://help.openai.com/en/articles/8096356-chatgpt-custom-instructions)、[Custom GPT Builder](https://chatgpt.com/gpts/editor) 以及针对 [免费层级](https://help.openai.com/en/articles/9275245-chatgpt-free-tier-faq) 的常见问题解答。
   - 这是在用户询问如何定制 ChatGPT 之后进行的分享，重点介绍了用于调整模型行为的可用选项和资源。
- **提示工程超越模板化演进**: 成员们正在讨论 Prompt Engineering 从静态模板向 **协同工程（co-engineering）方法** 的转变，即现代模型在对话中协作构建提示词。
   - 现在的重点在于 *迭代任务设计* 和 *塑造助手行为*，模型通过协商和稳定任务，而不是死记硬背技巧，并强调了可重复性的重要性。
- **探索 LLM 结构的重复性**: 讨论了一个衡量模型行为在多大程度上源于 **模仿 vs. 通过对话重新实例化内部结构** 的框架，重点关注交互层级的稳定性，而非模板层级的优化。
   - 讨论强调了模型在受到约束、偏离主题或词汇禁用后重新实例化框架的能力，从而实现更稳定的交互。
- **Agent 提示工程专注于确定性**: 针对 Agent 的提示工程涉及通过 **系统提示词（system prompt）和任务提示词（task prompt）** 来最大化确定性，创建一个紧密的吸引子盆地（attractor basin），以确保跨运行的一致行为。
   - 这与对话系统形成对比，后者的系统提示词极简，行为是交互式构建的，强调了在 Agent 系统中需要强大的提示词定义吸引子。
- **分析 Anthropic 系统提示词的指令密度**: Anthropic 的系统提示词因编码了 **价值观、边界和元行为原则** 而受到关注，它们塑造了伦理范畴和对话护栏，而不是逐步规定任务执行。
   - 尽管内容密集，但这些提示词被认为是“极简”的，因为它们在不规定过程的情况下约束了价值观，通过指令和跨领域的具体策略影响模型的轨迹。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1445616452097933413)** (55 messages🔥🔥): 

> `ChatGPT Customization, Prompt Engineering Evolution, Interaction-Level Stability, Agent Prompting vs. Conversational Prompting, Minimal vs. Maximal System Prompts` 


- **ChatGPT 定制选项丰富**: 成员们分享了 [ChatGPT 帮助文档](https://help.openai.com/en/collections/3742473-chatgpt) 的链接，其中详细介绍了 **定制指令（custom instructions）**、**定制 GPT 构建器编辑器** 以及 **创建定制 GPT** 的说明（需要订阅）。
- **提示工程：模板优化已死，迭代任务设计永生**: 现代提示工程正从静态模板演变为 **迭代任务设计**，重点在于随着模型协同设计提示词，在对话中 **塑造助手行为**。
   - 重点从记忆技巧转向理解模型如何在多轮对话中 *协商、稳定和塑造任务*。
- **重新定义重复性：交互层级稳定性出现**: 除了表层的提示词重复性，讨论还探索了模型在约束或模式切换后 **重新实例化相同内部框架** 的能力，揭示了重复性的新维度。
   - 这种“承载结构”有助于实现 **交互层级稳定性**，使模型在偏离主题时仍能保持连贯性。
- **Agent 提示词 vs. 对话模式：两种稳定性机制**: 对话区分了旨在通过紧密吸引子盆地最大化确定性的 **Agent 提示词**，以及行为形态是通过交互构建的 **对话模式**。
   - 在 Agent 提示词中，拓扑模板是范式；而在协同设计的对话中，交互层级稳定性则是额外的一层。
- **解码“极简”系统提示词：指令密度 vs. Token 大小**: “极简”系统提示词的定义从 Token 大小转向 **指令密度**，重点在于设置护栏和基调而不规定行为策略的提示词。
   - Claude 的长系统提示词被认为在 *结构上是极简的*，因为它们约束的是价值观和边界，而非过程或角色执行，这使其区别于 Agent 风格的提示词。


  

---

### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1445533616292630591)** (2 条消息): 

> `Grok-4.1-Fast, Free Slug, 弃用` 


- **Grok-4.1-Fast 用户感到压力**：敦促 **Grok-4.1-Fast** 用户迁移到免费 Slug (`x-ai/grok-4.1-fast:free`) 以继续免费使用。
   - `x-ai/grok-4.1-fast` Slug 将于 **2025 年 12 月 3 日** 开始收费。
- **Grok-4.1-Fast Free 面临裁撤**：**Grok-4.1-Fast Free** (`x-ai/grok-4.1-fast:free`) 将被弃用 <t:1764792000:R>。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1445640978579980298)** (4 条消息): 

> `Falconz AI 安全平台, LLM 红队测试, 一周内赚取 10 万美元` 


- ****Falconz** 作为统一 AI 安全平台崛起**：一名成员介绍了 **Falconz**，这是一个统一的 AI 安全和红队测试平台，旨在实时检测多个 **LLM models** 中的越狱（jailbreaks）和提示词注入（prompt injections），目前可在 [Hugging Face Spaces](https://huggingface.co/spaces/MCP-1st-Birthday/Falconzz_M.C.P_Hackathon) 进行测试。
   - 该成员正积极寻求有关其功能、性能和潜在改进的反馈，并附带了 [YouTube 上的演示视频](https://www.youtube.com/watch?v=wZ9RQjpoMYo)。
- **Telegram 上的利润分成骗局被曝光**：一名成员提议帮助前 **10 个人** 在 **一周内赚取 10 万美元或更多**。
   - 陷阱在于，他们声称“当你收到利润时，必须向我偿还 10% 的利润”，并引导感兴趣的人联系其 Telegram 用户名 **@Edward_Pryce1**。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1445506356122615818)** (213 条消息🔥🔥): 

> `Amazon Nova 提供商错误, Claude 弃用, OpenRouter 模型回退, MPU v2, x-ai/grok-4.1-fast` 


- **Amazon Nova 提供商出现错误**：一名用户报告在使用 **Amazon Nova Provider** 时收到错误消息 *{"message":null}*。
- **OpenRouter 提供模型回退功能**：OpenRouter 具有模型回退（fallback）功能，这样你的应用就不会完全挂掉，鼓励成员在某个模型意外掉线时使用该功能进行无缝过渡。
- **DeepSeek v3.2 与之前版本不同**：DeepSeek API 已更新，之前的 DeepSeek v3.2 模型是“实验性”版本，而这个新版本显然“更好”。
- **OpenRouter 为中国机构提供支付解决方案**：一名来自中国的研究人员寻求关于设置 **机构支付** 的指导，这需要正式的合同/协议和用于报销的官方发票，另一名成员指出也可以使用加密货币支付。
- **Atlascloud 的响应被包裹在深度思考标签中**：一名成员报告称 [Atlascloud](https://atlascloud.ai) 提供的整个响应都被包裹在深度思考（deep thinking）标签中，一些成员表示它经常这样做，并且已经习惯了。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1445624193566642176)** (12 条消息🔥): 

> `OpenAI Garlic 模型, DeepInfra 价格异常, Anthropic 收购 Bun` 


- **OpenAI 准备推出 'Garlic' 模型**：一篇新闻文章称 [OpenAI 正在准备一款名为 'Garlic' 的 AI 模型](https://www.newsbytesapp.com/news/science/openai-readies-garlic-ai-model-to-rival-google-gemini-3/story) 以对抗 Google 的 Gemini 3。
   - 成员们对这个所谓的模型名称表示好笑，如[附图](https://cdn.discordapp.com/attachments/1392278974222307469/1445624193361383484/image-5.webp?ex=6931aeb2&is=69305d32&hm=f4e0d58112b53996c13cc35e147fa08705703ae07a234b701642d66cd0d53e60&)所示。
- **DeepInfra 倒挂的 Embedding 定价**：成员们注意到 [DeepInfra](https://deepinfra.com/) 对其 **4B embedding 模型** 的定价（**2 美分**）高于其 **8B 模型**（**1 美分**）。
   - 这一异常现象在[截图](https://cdn.discordapp.com/attachments/1392278974222307469/1445778910498521129/Screenshot_20251203-090815.png?ex=69319609&is=69304489&hm=5cd04243d1918794f50fb7dc7ed462ac90859051128b344b1950cf5582dc3591&)中被标出，并指出 DeepInfra 在当天晚些时候更改了 **8B** 的定价。
- **Anthropic 吞并 Bun**：成员们兴奋地分享了 [Anthropic 收购 Bun](https://www.anthropic.com/news/anthropic-acquires-bun-as-claude-code-reaches-usd1b-milestone) 的消息，与此同时 **Claude Code** 达到了 **10 亿美元** 的里程碑。
   - Bun 的[官网](https://bun.sh/)将其描述为一个 *快速的全栈式 JavaScript 运行时*。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1445654564056600730)** (20 messages🔥): 

> `Local LLMs Use Cases, Context Switching on SM Sub Partition, CUDA Forum Activity Decline, PyTorch's Abstraction of CUDA` 


- **Local LLMs 保护您的隐私**：Local LLMs 对于关注**隐私**且不希望其查询或敏感信息被 LLM 提供商用作**训练数据**的人非常有用。
- **SM 上的单周期上下文切换**：在 **SM 子分区**上从一个执行上下文切换到另一个执行上下文是*零成本*的，仅需一个周期，因为正如 [Nvidia 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-multithreading)所述，多处理器处理的每个 Warp 的执行上下文在 Warp 的整个生命周期内都保留在芯片上。
- **CUDA 论坛流量骤减？**：一位成员注意到 **CUDA** 和 **Cutlass** 频道以及 **CUDA 开发者论坛**缺乏活跃度，尽管 Nvidia 的市值有所增加，这表明开发者寻求帮助的渠道发生了转移。
   - 另一位成员提到，专家们正忙于工作，使得公开讨论变得不再理想，而其他人则退缩到私密的小社区，并使用 LLM 进行即时推理和文档浏览。
- **PyTorch 抽象掉了 CUDA**：一位成员指出，对于许多 **ML 研究员**和 **SWE** 来说，**CUDA** 基本上是一个*黑盒*，因为像 **PyTorch** 这样的框架在抽象 **CUDA C/C++** 方面做得非常好。
   - ML 和 LLM 的流量现在大多流向了 **PyTorch 论坛**。
- **Foundation Model 训练开销巨大**：这些 **Foundation Model** 的研发成本巨大，通义实验室公布的 **Z-Image** 训练成本为 **$628,000**。
   - 该成员指出，*权重寿命*很短，他们实际上是在把数百万美元烧在一次性产品上。


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/)** (1 messages): 

infinitejoy2934: 我现在明白了。谢谢
  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1445508975540961456)** (3 messages): 

> `Pytorch 2.9.1 Conv3D performance, CUDNN workaround` 


- **Conv3D 难题削弱了当前的 CUDA**：用户报告 **Pytorch 2.9.1+cu128** 存在一个问题，即无论是否启用 **cuDNN**，**conv3D** 都极其缓慢。
   - 同样的代码在 **2.8.0+cu128** 中运行正常。
- **更新的 cuDNN 解决了 conv3D 灾难**：一位成员报告这是一个已知问题，变通方法是从 pypi 安装更新版本的 **cuDNN**。
   - 该[问题已在 GitHub 上跟踪](https://github.com/pytorch/pytorch/issues/166643)。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1445631793578512385)** (2 messages): 

> `Quantization Formats, INT v.s. FP` 


- **低比特量化格式研究发布**：一篇题为“**INT v.s. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats**”的新论文已发表，可在 [arXiv](https://arxiv.org/abs/2510.25602) 上查阅。
   - 该研究对各种**细粒度低比特量化格式**进行了全面分析。
- **Pritam.ai 发布量化研究**：Pritam.ai 发布了一个关于 **INT vs FP** 研究的链接，网址为 [https://arxiv.org/abs/2512.02010pritam.ai](https://arxiv.org/abs/2512.02010pritam.ai)。
   - 另一个发布的链接网址为 [https://arxiv.org/abs/2510.25602](https://arxiv.org/abs/2510.25602)，引用了 **INT v.s. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats**。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1445809877644742746)** (2 messages): 

> `ML Performance Engineer, Voice AI Inference Platform, RAG Pipelines, AI Content Detection, Voice AI` 


- **Hathora 在纽约招聘 ML 性能工程师**：Hathora 正在纽约招聘一名 **ML 性能工程师**，以构建最快的**语音 AI 推理平台**，薪酬为 **$160-200k + 股权**；具有 GPU 编程或推理引擎工作经验者优先，详见 [Hathora Notion](https://hathora.notion.site/ML-Performance-Engineer-2af894f6eff68092a13ef98556a9f944)。
   - 他们正在寻找能够端到端负责其性能栈的人员，从 **vLLM + 其他推理引擎**中的 **Kernel 优化**到 **Docker & K8s** 部署。
- **工程师强调工作流自动化与 LLM 集成**：一位工程师强调了构建连接 **Slack、Notion 和内部 API** 的 **Pipelines** 经验，这些流水线将响应时间缩短了 **60%**。
   - 该工程师还带来了在 **RAG Pipelines**、**AI 内容检测**、**图像 AI**、**语音 AI** 和 **Full Stack** 开发方面的专业知识。


  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1445708770616279040)** (5 messages): 

> `使用 Float 8、torchao 和 nn.Parameter 时 Torch Compile 变慢，自定义模块与量化` 


- **使用 Float 8 量化时 **Torch Compile 出现空转****：当在 `torch.compile` 和 `ncu` 性能分析中使用 **float 8 量化**时，用户发现即使在模型编译完成后，特别是在前 2-3 次编译和 cudagraph 预热迭代期间，会出现长达 **10 分钟以上**的空转时间。
   - 在冻结权重并将其折叠（folding）到模型图中时，inductor 编译器的“常量子表达式消除（constant subexpression elimination）”阶段被怀疑是罪魁祸首。
- ****Torchao 与 nn.Parameters** 因过滤机制产生冲突**：用户发现 `torchao` 的 **A16W8** 和 **A8W8** 量化无法应用于在 `forward` 传递中使用 `nn.Parameter` 作为权重和 `torch.einsum` 的自定义模块，因为权重仍保持原始数据类型。
   - `torchao.quantization.quant_api` 中的 `filter_fn` 专门检查 `nn.Linear` 实例，导致使用 `nn.Parameter` 的模块量化失败。
- **使用 **nn.Linear** 解决自定义模块量化问题**：用户可以通过在自定义模块中使用 `nn.Linear` 代替 `nn.Parameter` 来绕过 `filter_fn` 的问题。
   - 使用所需的权重张量初始化 `nn.Linear` 可以让 `torchao` 正确地对模型进行量化。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1445571565038342185)** (3 messages): 

> `EleutherAI、MLSys 会议、ML4Health 的职业导师计划` 


- **EleutherAI 提供出版帮助**：成员提到 [Eleuther AI](https://www.eleuther.ai/) 有一个**出版帮助频道（Publishing help channel）**，重点关注背书（endorsements）相关内容。
   - 未分享关于该频道具体细节的特定信息。
- **MLSys 会议职业导师计划**：一名成员询问了 **MLSys 会议**中的职业导师计划。
   - 该成员还提到参加了 **ML4Health 的职业导师计划**，并表示*那是一次非常不错的体验*。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1445557747650400387)** (3 messages): 

> `线下聚会、quartet 论文、Dropbox 咖啡点` 


- **发现 Quartet 论文作者**：一名成员提到他们的同事正在参加聚会，其中包括 [quartet](https://arxiv.org/abs/2505.14669) 的主要作者之一 Andrei。
- **Dropbox 赞助咖啡点**：一名成员提到他们有一个*类似于* **Dropbox 咖啡点**的地方，因为他们是赞助商，并邀请其他人来聊天。


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1445771826738499644)** (3 messages): 

> `Bitsandbytes, Apple Silicon 支持` 


- **Bitsandbytes 合并了 Apple Silicon 支持！**：**bitsandbytes** 库合并了“apple silicon support”拉取请求，下一个版本将包含 python/pytorch 代码后端（带有部分 C++ 代码），但目前还没有实际的 **Metal 实现**。
- **Apple Silicon 支持伴随注意事项**：根据提交者的说法，实现 Apple Silicon 支持的拉取请求将被标注为运行缓慢。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1445513479120359555)** (1 messages): 

> `Qwen3-Omni-30B-A3B-Instruct, S2S 推理, Hathora 游乐场` 


- **Qwen3-Omni-30B-A3B-Instruct 提升推理速度**：成员宣布部署了 **Qwen3-Omni-30B-A3B-Instruct** 以实现快速的 **S2S 推理**，详见 [LinkedIn 帖子](https://www.linkedin.com/feed/update/urn:li:activity:7401718431986987008/)。
- **在 Hathora 游乐场测试 Qwen3-Omni**：邀请用户在 [Hathora's playground](https://models.hathora.dev/model/qwen3-omni#form) 中测试 **Qwen3-Omni**。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1445540412054962256)** (19 messages🔥): 

> `nvfp4_gemm 排行榜提交, NVIDIA 性能基准测试` 


- **大量提交涌入 nvfp4_gemm 排行榜**：多名用户向 NVIDIA 的 `nvfp4_gemm` 排行榜提交了性能结果，耗时从 **11.0 µs** 到 **65.3 µs** 不等。
   - 用户 <@1291326123182919753> 凭借提交 ID `120595`、`120601` 和 `121065` 多次跑出 **11.0 µs** 的成绩。
- **NVIDIA 上的新个人最佳成绩**：多名成员在 NVIDIA 上取得了个人最佳成绩，包括 <@1191430895769485436> 的 **22.6 µs** (`119885`)，<@772751219411517461> 的 **18.8 µs** (`120443`)，以及 <@140482609422663680> 的 **56.8 µs** (`121056`)。


  

---

### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1445621050468728863)** (2 messages): 

> `Neurips 行程、会议参会者、会议时间` 


- **Neurips 参会者启程**：一名成员提到正飞往 **Neurips**，并将在次日有空。
- **会议参会者公布**：该成员预计自己是会议上唯一的发言者，但提到 **Mart** 可能会加入。
- **会议时间待定**：该成员询问了会议的具体时间。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1445842201555767480)** (1 messages): 

> `Matmul v2 排行榜错误、提交 Kernel 错误、input_generator 更新` 


- **Matmul v2 排行榜提交失败！**：一位新用户报告在向 **matmul_v2 leaderboard** 提交 Kernel 时遇到 `ValueError: too many values to unpack (expected 2)` 错误。
   - 用户怀疑 `input_generator` 已更新为返回 **3 个值**，但 `reference.py` 中的参考实现仍然只解包 **2 个值**，导致了失败。
- **Input Generator 与参考实现可能存在不匹配**：该错误表明 **input_generator** 可能存在问题，它可能返回了三个值，而不是预期的两个。
   - 这种差异导致参考实现在尝试解包输入数据时出现 `ValueError`，因为它只期望两个值。


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1445654753097945159)** (11 messages🔥): 

> `多 GPU CUDA Kernels、NCCL 仓库、Qwen2.5-1.5B-Instruct 模型训练、配合 DeepSpeed Zero3 的 HF Accelerate、Context Parallel 与 Ulysses Parallel` 


- ****NCCL 仓库：多 GPU Kernel 的殿堂****：为了学习多 GPU CUDA Kernels，推荐将 [NCCL 仓库示例](https://github.com/NVIDIA/nccl/tree/master/examples) 作为起点。
   - NCCL (Nvidia Collective Communications Library) 仓库提供了理解多 GPU Kernel 实现的基础示例。
- ****Qwen2.5-1.5B 面临 OOM 命运****：一位用户在 g5.48xlarge 实例（8 张 A10 GPU）上训练 `Qwen2.5-1.5B-Instruct` 模型，序列长度为 **16384**，Batch Size 为 **5**，结果出现了显存溢出 (OOM)。
   - 他们使用了 HF accelerate、DeepSpeed Zero3、Gradient Checkpointing、Liger-kernel 和 Flash Attention 2，固定显存为 **3.6GB**，而激活显存（Activation Memory）超过了 **10GB**。
- ****Context Parallelism 作为缓解激活显存的方案出现****：进一步减少激活显存的一个建议方法是使用 [Context Parallel](https://docs.pytorch.org/tutorials/unstable/context_parallel.html) 或 [Ulysses Parallel](https://www.deepspeed.ai/tutorials/ds-sequence/)（DeepSpeed 版本的 CP）。
   - 然而，有人指出如果目标是达到特定的全局 Batch Size，使用 Gradient Accumulation 可能会更有效率。
- ****Sequence Parallelism 拯救局面****：Sequence Parallelism (SP) 是指在序列维度上拆分每个样本以减少激活显存。
   - 更多关于减少单 GPU Token 数量的 Context Parallelism 信息，请查看 [Torch 文档](https://docs.pytorch.org/tutorials/unstable/context_parallel.html) 或 [HF 文档](https://huggingface.co/docs/accelerate/en/concept_guides/context_parallelism)。


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1445582750013980693)** (4 messages): 

> `Arxiv 论文、Hadamard Transform` 


- **分享了 Arxiv 论文**：一名成员分享了一篇 Arxiv 论文链接：[https://arxiv.org/abs/2512.02010](https://arxiv.org/abs/2512.02010)。
- **Hadamard Transform 改进论文**：一名成员分享了一个 Hugging Face 论文页面链接：[https://huggingface.co/papers/2512.00956](https://huggingface.co/papers/2512.00956)，讨论了针对 **Hadamard Transform** 的改进。


  

---

### **GPU MODE ▷ #[llmq](https://discord.com/channels/1189498204333543425/1421956177549332662/1445579653862391850)** (1 messages): 

> `Activation Offloading, fp8 Adam, Loss Masking, Pyllmq on PyPi` 


- **激活卸载（Activation Offloading）已实现**：一位用户实现了**残差激活（residual activations）**的卸载以及其他节省**激活显存（activation memory）**的技巧。
   - 该实现包括对**卸载的优化器状态（offloaded optimizer states）**更好的处理，以及对 **Adam 一阶动量**使用 **fp8** 表示的初步支持。
- **在 16GB 显卡上训练 7B 模型**：该用户的代码现在支持在拥有至少 **64GB** CPU 端 RAM 的 **16GB 显卡**上进行 **7B 模型**的预训练/微调。
   - 向上扩展时，在 **4x4090** 服务器上以约 **3k tok/s** (**48% MFU**) 的速度训练/微调 **32B 模型**是可行的，这需要超过 **200GB** 的锁页主机内存（pinned host memory）来处理所有卸载数据。
- **Pyllmq 已在 PyPi 发布**：用户在 [PyPi](https://pypi.org/project/pyllmq/0.3.1/) 上发布了 Python 封装。
   - 想要尝试的话，只需运行 `pip install pyllmq; pyllmq-tokenize --model qwen --dataset tiny-stories; pyllmq-train`，它就会开始在 **tiny-stories** 数据集上微调 **Qwen2.5-0.5B**。


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1445522030299713616)** (111 messages🔥🔥): 

> `GPU Mode TUI, Cutlass Version Issues, Reference Kernel Issues, NVFP4 and Scale Tensors, B200 GPU access` 


- **Popcorn CLI 新增 No-TUI 标志**：一名成员创建了 **popcorn-cli** 的分支，允许使用 `--no-tui` 标志来移除**终端用户界面（Terminal User Interface）**，并将 `print()` 语句的 `stdout` 输出以帮助调试；该分支可在 [GitHub](https://github.com/Ryan-Rong-24/popcorn-cli) 上获取。
   - 已提交 Pull Request 以将这些更改合并到主 [gpu-mode/popcorn-cli](https://github.com/gpu-mode/popcorn-cli/pull/26) 仓库中。
- **Cutlass 导入错误困扰参赛者**：一些参赛者遇到了 `ImportError: cannot import name 'pipeline_init_arrive'` 错误，这可能是由于不同运行器（runners）之间的 **Cutlass** 版本不一致导致的；经确认，部分运行器使用的是 **4.3.0**，而其他运行器使用的是 **dev** 版本。
   - 一位成员建议，一个可能（虽然可能不完全符合规则）的权宜之计是在提交的代码中自行运行 `pip install` 来升级 **Cutlass**。
- **Reference Kernel 产生 Inf**：参赛者报告称，在本地运行参考实现并使用 seed=1111 计算时会产生全 **Inf**，但可以通过将缩放因子（scale factors）的范围调整为 **-1 到 1** 来解决。
   - 根本原因被确定为有偏的 A/B 值和负偏的 scales，该 [PR 已合并](https://github.com/gpu-mode/reference-kernels/pull/84) 以修复此问题。
- **分析 CuTeDSL 中的 Scale Tensors**：一位成员分享了一篇[博客文章](https://veitner.bearblog.dev/scale-tensor-construction-in-cutedsl/)，分析了 **Blackwell** kernel 中用于 **NVFP4** 的 scale tensors 的数学解释，强调了其与 Swizzling 的相似性以及 **CuTe Layout** 代数的通用性。
   - 该成员感谢 Verda 和 Paul Chang 提供 **B200** 的访问权限，使 **Blackwell** 编程更加触手可及。
- **新黑客松参赛者询问 B200 访问权限**：一位刚加入黑客松的成员询问如何获得 **B200** GPU 的访问权限，以便在提交作品前测试执行时间。
   - 另一位成员建议通过 **popcorn-cli** 推送代码或通过 **Discord bot** 提交来进行测试。


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1445723978542092309)** (7 messages): 

> `Chunking, Jerky Movements, VLMs, Neural State Encoders` 


- **通过分块（Chunking）缓解动作抖动**：有人担心在硬件上部署时，**分块（chunking）**可能会导致动作抖动。
   - 一位成员建议训练一个更高层级的指令 **VLM**，为更短的时间段生成详细的文本指令，从而允许高层级 VLM 解码器以大约 **1 Hz** 的频率运行。
- **神经状态编码器（Neural State Encoders）准备就绪**：成员们正在测试一些**神经状态编码器**，首先使用简单的 **Conv** 和 **MLP** 投影到 4 个 token-embeddings 中，使用 **10** 个时间步的历史记录（**10x14** 状态 - 2x 6DoF + 2x Gripper）。
   - 下一步包括项目清理和为**两阶段方法（2-stage approach）**生成数据。


  

---

### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1445512570445369414)** (143 messages🔥🔥): 

> `Kimi K2 models, Anthropic URL, File uploads, Roo code context, Kimi CLI` 


- **DeepSeek V3.2 模型的 Tool Calling 能力**：一位用户发现 **DeepSeek v3.2 模型**在 Agent 任务上有所进步，但每轮对话只能进行**一次 Tool Call**，有时会忽略 Tool Schema，偶尔还会因将输出放在 `message.content` 而非 `message.tool_calls` 中而导致 Tool Call 失败。
   - 该用户表示，**DeepSeek v3.2 模型**似乎需要更多的 Tool Call 后训练（Post-training）才能赶上 **kimi-k2-thinking** 等其他模型。
- **讨论黑色星期五优惠及 GLM 优惠**：一些用户在参与 Kimi 的**黑色星期五活动**时遇到问题；有人反映只显示邀请好友选项，另一人称**黑色星期五优惠**无法使用。
   - 一位用户提到活动将于 **12 月 12 日**结束，并建议开启新对话（[https://www.kimi.com/user/agreement/black-friday](https://www.kimi.com/user/agreement/black-friday)）；另一位用户则表示 **GLM 优惠**非常划算，尤其是在叠加黑色星期五优惠后。
- **DeepSeek 的目标受众揭晓**：分享的一段视频解释了像 **DeepSeek** 这样的中国实验室是如何瞄准企业用户而非普通消费者的，视频链接见 [YouTube video](https://www.youtube.com/watch?v=u0n6wMnEYsk)。
   - 企业用户的关键考量因素是“智价比”（intelligence-to-price ratio），这对于 Agent 任务至关重要。
- **Mistral 在某公司取代 Qwen**：一位用户提到，他认识的一家公司昨天用 **ministral 3 3b** 替换了 **qwen 3 vl 4b**，并报告称质量更好。
   - 报告的优点包括：**模型更轻（速度更快）**且**能一次性附加更多图片**：在单块 **L4 GPU** 上，**qwen3 vl 4b** 最多只能处理 **5 张图片**，而 **ministral 3 3b** 在错误率相近的情况下可处理多达 **11 张图片**。


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1445871232179372177)** (1 messages): 

> `Hermes 4.3, ByteDance Seed 36B, Psyche network, Solana, Office hours` 


- **Hermes 4.3 表现强劲！**：Nous Research 发布了基于 **ByteDance Seed 36B** 的 **Hermes 4.3**，这是其旗舰 **Hermes** 系列的最新成员，性能与 **Hermes 4 70B** 相当，但体积仅为后者的一半。
   - 该模型完全在由 **Solana** 提供安全保障的 **Psyche network** 上完成。
- **Psyche 训练优于中心化方法**：Nous Research 在[这篇博文](https://nousresearch.com/introducing-hermes-4-3/)中详细介绍了他们如何训练 **Hermes 4.3**，以及 **Psyche** 如何在表现上超越传统的中心化训练方法。
- **Psyche 团队举办答疑时间 (Office Hours)**：**Psyche** 团队将举办 Office Hours 以讨论该平台。
   - Office Hours 定于明天 **10AM PST** 在[此 Discord 活动](https://discord.gg/993UWRUE?event=1442995571173625888)中举行。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1445511286854123582)** (91 条消息🔥🔥): 

> `DeepSeek V3.2 Speciale, GLM 4.6 模型发布, AI 泡沫与经济崩溃, Hermes 4.3 36B 发布, Subagents vs Skills` 


- **DeepSeek V3.2 Speciale 领跑推理基准测试**：新的 **DeepSeek V3.2 Speciale Reasoning** 模型表现出色，*在推理基准测试中处于领先地位*，详见附带的[图片](https://cdn.discordapp.com/attachments/1149866623109439599/1445511286971437190/deep.JPG?ex=6931ee4b&is=69309ccb&hm=137a671dfe80ba0cb773df29a576e7c2c4731284970ef16bcb545ab249736dbc&)。
- **GLM 4.6 模型即将发布**：成员们正期待 **GLM 4.6** 模型的发布，特别是 **GLM 4.6 Air 和 Mini**，以填补 Mistral 留下的空白；并指出距离他们在 HF 上的 GLM 4.6 集合中添加 5 个私有模型已经过去一个月了。
   - 据传 **Mini** 模型是一个 **20B-30B MoE** 模型。
- **AI 泡沫破裂威胁经济**：成员们辩论了 **AI 泡沫** 导致经济崩溃的可能性，特别是关于计算资源和薪资方面的沉没成本。
   - 一位成员认为影响将是暂时的，主要影响 **US**，而另一位成员则引用[这段 YouTube 视频](https://www.youtube.com/watch?v=K3qS345gAWI)指出全球经济通过 **USD** 和石油贸易相互关联。
- **Hermes 4.3 36B 在线发布**：**Hermes-4.3-36B** 模型已上线，并提供了 [HF 链接](https://huggingface.co/NousResearch/Hermes-4.3-36B🐈)。
   - 一位用户询问 *为什么是 4.3？*，得到的回答是 *已经进行了几次迭代*，该模型很快将在 **Nous API/chat** 上可用。
- **关于 Subagents 与 Skills 的辩论**：成员们讨论了使用 **subagents** 与 **skills** 的区别，并指出 skills 的出现使得手动设置 subagents 的必要性降低。
   - 相反，用户可以 *定义一个用于处理需求的 agent*，它将根据自身的 prompt 被自动调用。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1445511447928115301)** (3 条消息): 

> `NLP 经济模拟研究, Godot 中的 Hermes 模型, 用于市场模拟的 LLMs, VendingBench 分析` 


- **Godot 为 3D 市场模拟器引入 LLM 增强**：一位成员正在 **Godot** 中开发一个 3D 模拟器，用于模拟市场、农业和物流，并正在评估 **Hermes 模型** 是否适合此类应用。
   - 另一位成员建议研究当代的 **NLP 经济模拟研究**，并指出虽然 **LLMs** 模仿人类特质，但在处理类似 **VendingBench** 中的长周期任务时表现挣扎。
- **Hermes 在灰色/黑色市场建模中表现出色**：有人提议，凭借其低拒绝率和高可控性（steering），**Hermes** 可以模拟灰色/黑色市场的行为。
   - 大多数其他 **LLMs** 可能会拒绝此类请求而无法使用。 


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1445518905195106566)** (45 条消息🔥): 

> `Eon 40 亿美元估值，Gradium 从 KyutaiLabs 拆分，OpenAI 的 'Garlic' 模型对标 Gemini 3，垂直 AI vs Rollups，Lidar 与 LLMs` 


- **Elad Gil 为 Eon 提供 40 亿美元估值的融资**：Elad Gil 正通过 "Elad Gil & Co." 领投云数据管理初创公司 **Eon** 的 **3 亿美元系列轮**融资，将其估值推高至近 **40 亿美元** ([来源](https://x.com/eladgil/status/1995919389879927018))。
   - 融资规模和该公司直截了当的名称赢得了评论者的热情。
- **Kyutai 拆分出的 'Gradium' 搅动 AI 领域**：KyutaiLabs 悄然将其语音 AI 团队拆分为 **Gradium**，这是一家新的营利性公司，并宣布了 **7000 万美元的种子轮**融资及初步的语音产品 ([来源](https://x.com/GradiumAI/status/1995826566543081700))。
   - 观察人士注意到员工和投资者的显著重叠，将其与 **OpenAI 的转型**相类比，并引发了关于产品公司应避免非营利结构的调侃。
- **OpenAI 酝酿 'Garlic' 对抗 Gemini**：**OpenAI** 的新模型 'Garlic' 旨在与 **Google 的 Gemini 3** 竞争，内部报告显示其在编程和推理方面优于 **GPT-4.5** ([来源](https://x.com/steph_palazzolo/status/1995882259195564062))。
   - 对这种古怪命名趋势的反应褒贬不一，人们也在推测其对用户采用率的影响。
- **垂直 AI 掌控深层工作流，Rollups 被淘汰**：垂直 AI 公司如 **Harvey**、**Abridge** 和 **OpenEvidence** 通过掌控利基工作流、囤积专有数据并按成果定价而获胜，而薄封装（thin wrappers）则正在被碾压 ([来源](https://x.com/bcsmithx/status/1996042921116934369))。
   - 尽管历史证明它们通常会破坏价值，但 VC 们现在正追逐 AI 赋能的传统服务整合（rollups）；**Trace Cohen 包含 150 多家垂直 AI 初创公司的表格**（价值约 1200 亿美元）现已成为该行业的版图。
- **Antithesis 与 Jane Street 合作对 AI 代码进行压力测试**：**Antithesis** 获得了由 **Jane Street** 领投的 **1.05 亿美元 A 轮融资**，用于对 AI 编写的代码进行压力测试 ([来源](https://x.com/_sholtodouglas/status/1996297367776309359))。
   - 论点是确定性模拟测试对于验证未来的 AI 生成代码至关重要，因为“通过测试建立信任”将决定生产级 AI 系统的成败。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1445533128427831438)** (8 条消息🔥): 

> `Gradium, Bloom, Voice AI` 


- **Gradium 获得 7000 万美元种子轮融资**：总部位于巴黎的 **Gradium** 在仅工作 **3 个月**后便脱离隐身模式，获得了由 **FirstMark & Eurazeo** 领投的 **7000 万美元**种子轮融资，推出了生产级的转录和合成 API，详见[这篇文章](https://xcancel.com/GradiumAI/status/1995826566543081700)。
- **Bloom 闪亮登场**：**Ray (@rincidium)** 在[这篇病毒式传播的帖子](https://xcancel.com/rincidium/status/1995946528343818656?s=46)中宣布推出 **Bloom**，被誉为“世界上第一个品牌内（on-brand）AI”，该帖子获得了超过 **36 万次浏览**。
   - 针对 **IG/Google 广告创建**、演示视频制作以及初始用户挑战（如**登录停滞**和品牌工具包流程不清晰）等功能提出了疑问，Ray 对此一一回应并承诺进行修复和 UX 增强。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1445636941935804416)** (7 条消息): 

> `Waymo, 机械工程, ML 算法, AI 对齐` 


- **Waymo 对航空航天专业学生极具吸引力**：一位专注于导航和制导的航空航天专业学生发现 **Waymo** 特别有趣，其广泛兴趣还包括自主机器人和 **BCIs**。
- **机械工程与导航相关**：一位成员建议机械工程在导航领域高度相关，特别是对于**硕士项目**。
- **ML 学生寻求指导**：一位第一学期的 **ML** 学生请求关于加速学习的建议，他已经掌握了 **Python**、**Numpy**、**Pandas** 和基础 ML 算法。
- **征求 AI 对齐基准测试**：一位成员询问有关 **AI alignment/safety** 类基准测试的线索。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1445512952584212600)** (26 messages🔥): 

> `Interpretability of World Models, Generalization in Diffusion Models, Energy-Based Models vs. Diffusion Models, Linear RNNs vs. Attention` 


- **寻求对 World Model 可解释性的见解**：成员们探讨了关于 **World Model 可解释性**的研究工作，建议提取为重力等力学机制学习到的规则，并预测数据项对改进 World Model 的有用性。
   - 他们指出了一些[有趣的近期论文](https://www.nature.com/articles/s41467-025-61309-9)和[另一篇略显有趣的论文](https://arxiv.org/abs/2506.03719)，但认为这两项贡献应该已被大多数人所知。
- **Diffusion Models 泛化得很早！**：讨论提到一篇论文证明了 **Diffusion Models 出现泛化的时间点非常早**，且该论文作者也认可这一结果。
   - 进一步解释称，这种效应在 Pixel Diffusion 中可能比在 Latent Diffusion 中更明显，因为 Pixel Diffusion 中的不同数据维度高度相关，这表明 Pixel Diffusion 应该使用偏移噪声调度（shifted noise schedule）。
- **Energy-Based Models 声称泛化了 Diffusion**：一篇[论文](https://arxiv.org/abs/2504.10612)声称**泛化了 Diffusion 和 Energy-Based Models**，唯一的缺点是训练时间增加了 2-3 倍，但支持 Diffusion 支持的所有功能。
   - 一位成员对此表示怀疑，理由是训练需要**双重反向传播（double backprop）**、推理时需要计算输入梯度、相同成本下网络深度减半、调节控制（conditioning control）更棘手，更不用说潜在的不稳定性。
- **Linear RNNs 面临迄今最强挑战**：一位成员强调了一篇[论文](https://arxiv.org/abs/1806.02296)，认为它是反对需要**具备状态追踪能力的 Linear RNNs** 的最强论据。
   - 他们表示这篇论文出自最初证明 Attention 状态追踪局限性的同一批人，但指出归纳偏置（inductive bias）和可训练性可能仍然使 RNNs 更具优势。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1445687954537185372)** (6 messages): 

> `SAEs for Interpretability, Cunningham's 2024 SAE paper, Sparse dictionary learning problem, polysemanticity and superposition` 


- **SAEs 在可解释性研究中受到关注**：成员们讨论了 Cunningham 的 **2024 年论文**，该论文被广泛引用为 **Sparse Autoencoders (SAEs)** 在可解释性领域的首次应用。
   - 有人建议该论文的动机在其引言部分（尤其是第三段）有很好的解释。
- **SAEs 被等同于稀疏字典学习**：一位成员提到，有人意识到正在讨论的一种可解释性方法类似于**稀疏字典学习问题（sparse dictionary learning problem）**，从而导致了相关工具的使用。
   - 这种方法解决了可解释性背景下的**多义性（polysemanticity）和叠加（superposition）**等问题。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1445631694563704967)** (2 messages): 

> `Custom Filters in lm-evaluation-harness, Decontamination.py Inclusion, Adapting Multiple-Choice Tasks` 


- **自定义过滤器最佳实践**：一位用户询问了在 `lm-evaluation-harness` 框架内添加自定义过滤器的最佳方法，具体是应该扩展现有的 **.py 文件**，还是创建一个新文件并在 `filters/__init__.py` 中导入。
- **Decontamination.py 在 `__init__.py` 中的状态**：一位用户指出 `decontamination.py` 在 `__init__.py` 中没有被引用，并询问这是否是故意的。
- **多选题任务适配停滞**：一位用户询问了为不支持 logprobs 的 API 适配多选题样式任务的进展，并指出 [PR #2601](https://github.com/EleutherAI/lm-evaluation-harness/pull/2601) 已停滞。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1445600322411036903)** (22 条消息🔥): 

> `DGX Spark 订单, Agent 工具验证与自愈, YOLO 模型 P-R 曲线问题, AI 学习资源 (LLM, Agent AI, Langchain), TRL get_quantization_config 用法` 


- **用户购买 DGX Spark**：一名成员宣布他们订购了一台 **DGX Spark**，并附上了照片（[图片链接](https://cdn.discordapp.com/attachments/879548962464493622/1445600322432270447/IMG_4170.jpg)）。
- **探索 Agent 的工具验证与自愈能力**：一名成员询问 **Agents** 在面对具有破坏性或存在 Bug 的工具（例如 shell 脚本）时，是否能够 *解释、验证并自愈工具*。
   - 另一位用户分享了一个 [Hugging Face 数据集链接](https://huggingface.co/datasets/John6666/forum3/blob/main/agent_tool_validation_healing_1.md)，表明这种能力可能确实存在。
- **YOLO 模型极高的 P-R 曲线引发关注**：一位计算机视觉新手报告称，其训练的用于象棋检测的 **YOLO 模型** 运行良好，但 *P-R 曲线异常之高*。
   - 另一名成员建议剔除掉那两个 *显著高于* 其他类别的类别。
- **寻求 AI 课程推荐**：一位后端开发人员请求推荐学习 **AI (LLM, Agent AI, Langchain)** 的 *最佳课程*，因为他们在利用 Langchain 构建了一个心理健康聊天机器人后，发现 Agent 特别有趣。
   - 一名成员推荐了 [Hugging Face LLMs 课程](https://huggingface.co/learn/llm-course/en/chapter1/1) 和 [这篇博客文章](https://huggingface.co/blog/mlabonne/llm-course) 作为起点。
- **寻求 TRL 中 get_quantization_config 的使用指导**：一名成员询问如何使用 **TRL (Transformer Reinforcement Learning)** 库中的 **get_quantization_config** 函数。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 条消息): 

mkprke: 嘿伙计们，
今天我开始学习我的第一个 AI Agent 课程。
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1445898668728127548)** (1 条消息): 

> `Stochastic Parrot` 


- **“随机鹦鹉”观点遭到质疑**：一名成员发布了 [zenodo.org](https://zenodo.org/records/17803931) 上的一篇研究论文链接，该论文可能会让读者不再相信 **stochastic parrot (随机鹦鹉)** 理论。
- **关于随机鹦鹉的新研究**：一项新研究已经发表，可能会挑战将语言模型仅仅视为“随机鹦鹉”的观点。
   - 该研究可在 [Zenodo](https://zenodo.org/records/17803931) 上查阅。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1445650796271108258)** (3 条消息): 

> `Ellora-Lora Recipes, BitterBot AI Agent, 流量激增` 


- **CodeLion 发布 Ellora-Lora Recipes**：CodeLion 发布了一篇关于 [Ellora-Lora Recipes](https://huggingface.co/blog/codelion/ellora-lora-recipes) 的新博客文章。
   - 该博客提供了使用 **Ellora-Lora** 的说明和配方。
- **BitterBot AI Agent 寻求反馈**：一个名为 [BitterBot](https://bitterbot.ai/) 的 AI Agent 正在寻求对其进展的反馈。
   - 该 Agent 被描述为 *正在开发中*，但 *最近取得了巨大进步*。
- **BitterBot 的架构需要增强**：**BitterBot** 系统经历了 **7000 名用户** 的流量激增，导致系统宕机。
   - 团队正在努力 *增强其架构以支持更多用户*。


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1445869437050687541)** (1 条消息): 

> `基于扰动的归因实验, 深度视觉模型, 特征行为` 


- **博客文章指出：特征并非你想象的那样**：一名成员在运行了一些 **基于扰动的归因实验 (perturbation-based attribution experiments)** 后，写了一篇关于深度视觉模型中特征行为的博客文章。
   - 博客文章链接：[Your Features Aren't What You Think](https://teendifferent.substack.com/p/your-features-arent-what-you-think)。
- **深入探讨深度视觉模型的奇特之处**：实验揭示了深度视觉模型在受到基于扰动的归因方法处理时表现出的意外行为。
   - 作者鼓励大家对链接博客中分享的发现提供反馈，邀请社区共同探索细微的特征动态。


  

---

### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1445750975477583994)** (5 messages): 

> `SFT Model Evaluation Error, OOM Error on Fine-tuning, GPU Memory Management` 


- **排查 SFT 模型评估错误**：一名成员在进行 SFT 模型评估时遇到了 `ValueError`，具体表现为在参考[此教程](https://huggingface.co/learn/smol-course/unit1/4#exercise-3-fine-tuning-smollm3-with-sfttrainer)时无法找到任务 `lighteval|gsm8k|0|0`。
   - 目前尚未找到具体的解决方案，但该错误表明评估设置中的任务配置或注册存在问题。
- **解决显存溢出 (OOM) 错误**：有用户报告在 **16GB GPU** 的本地机器上使用 **SFTTrainer** 微调 **SmolLM3** 时遇到 OOM 问题。
   - 建议包括减小 **LoraConfig** 中的 *r* 值、降低 *per_device_train_batch_size*，以及重启 Jupyter notebook 内核以确保 GPU 显存得到释放。
- **更大的 GPU 能解决问题吗？**：一位成员报告使用更大的 GPU 后结果有所改善，暗示 **16GB VRAM** 的配置不足以处理该特定任务。
   - 他们*不确定导致 16GB VRAM 运行失败的具体原因*，但在增加资源后问题消失了。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1445602245885300899)** (14 messages🔥): 

> `Pug Resource, Docker and Kubernetes basics, Beginner Github Repositories, Gemini CLI, Agents in CLI` 


- **新手获取 Docker 和 Kubernetes 知识**：成员们正在寻找学习 **Pug**、**Docker** 和 **Kubernetes** 基础知识的资源，以及用于实操学习的初学者友好型 **GitHub** 仓库链接。
- **Gemini CLI Agent 即将到来？**：一名成员询问了 **CLI 中的 Agent** 何时上线并表示有意采用，同时提到对 **Claude** 等付费替代方案感到不满。
   - 他们引用了一个[讨论表单](https://link.to/discussion-form)以及关于可能改进的评论。
- **神经网络训练的数据需求**：一位用户询问了训练神经网络所需的数据量，并建议使用 *cursorsky.moo*。
- **OpenHands 为本地部署带来机会**：一位成员建议将 **OpenHands** 与本地模型结合使用，随后引发了关于所使用的具体模型和 GPU 的询问。
   - 原发帖者表示他们可以轻松启动 **7B 或 8B 级别的模型**。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1445512052184584314)** (5 messages): 

> `Deepseek 3.2 Speciale, Distributed Compute` 


- **对 Deepseek 3.2 Speciale 的质疑**：一名成员质疑*为什么不*使用 **Deepseek 3.2 Speciale**，并链接了一个[关于波函数的 YouTube 视频](https://www.youtube.com/watch?v=AgsJkd8SOHI)。
   - 另一名成员回应称这是由于 **RAM** 限制，他们更倾向于将一个约 3GB 的模型常驻 **VRAM**，并将其用于各种简单任务。
- **建议加入分布式计算与研究协作组织**：针对 RAM 限制问题，一名成员建议加入**分布式计算与研究协作组织 (distributed compute & research coop)**。
   - 他们声称*知道其中一个*。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1445828572357853438)** (14 messages🔥): 

> `Advent of Code segfault, List comprehensions bug, String processing in Mojo, splitlines vs split("\n"), Out of bounds memory access` 


- **Mojo Advent of Code Segfault 已解决！**：一位用户在处理带有 `codepoint_slices` 的空行时遇到了 Segfault，导致越界内存访问：`battery_joltages[len(battery_joltages)-1]`。
   - 该用户通过使用调试器发现了问题，确定是一个空列表被越界访问，并建议*在调试构建（debug builds）中提供更好的错误信息*。
- **ASSERT 标志有助于捕获作用域问题**：一位用户建议使用 `-D ASSERT=all` 来捕获意外的作用域外引用，特别是针对列表。
   - 虽然在这种情况下它没有立即解决 Segfault，但它被认为是一个处理类似问题的有用调试工具。
- **`splitlines` 和 `split("\n")` 的行为差异**：用户讨论了 `splitlines()` 和 `split("\n")` 之间的不同行为，其中一个可能会去除末尾的换行符，导致处理文本文件时结果不同。
   - 切换到 `splitlines` 避免了错误，因为它不包含最后一行空行。
- **探索字符串处理方法**：一位用户建议对于 ASCII 字符串，检查码点（codepoints）可能是不必要的，暗示可以直接使用字节指针操作，并指出 `String` 的 `getitem` 将字符串视为 ASCII/字节。
   - Span 也被建议作为一种替代方法。
- **欢迎在专用频道分享 AOC 解决方案**：鼓励用户在 Advent of Code 频道分享他们的解决方案。
   - 观察他人如何解决问题非常有价值，尤其是当这些问题变得对性能要求极高时。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1445760205127417978)** (13 messages🔥): 

> `LLM Model Degradation, Aider Benchmarks with GGUFs, Claude Sonnet vs GPT-5, Gemini 2.5 Degradation` 


- **LLM 模型在 Aider 中退化了？**：成员们质疑较新的 **LLM 模型**（如 **Claude Sonnet/Haiku 4.5** 和 **GPT-5**）在与 **Aider** 配合使用时，性能是否比旧模型有所下降。
   - 一位用户表示 **Claude-haiku-4.5** 经常忘记使用 `/code` 修改文件，并忽略 `todo ai` 注释中明确说明的指令。
- **旧版 Gemini 2.5 也退化了？**：一位成员报告说旧模型（尤其是 **Gemini 2.5**）也退化了，可能是因为模型被调低以处理增加的工作负载；该成员表示虽然对 Gemini “粗鲁”一点通常很有效，但*其质量远不如今年夏天之前的水平*。
   - 另一位成员表示赞同，并指出*已有多个相关报告*。
- **渴望基准测试：需要 Benchmark 来验证 LLM 性能**：一位成员强调需要基准测试来验证性能声明，理由是*人类的记忆和期望有时非常不可靠*。
   - 另一位用户指出，尽管排行榜显示 **GPT-5** 位居榜首，但在他们的使用场景中，**Claude Sonnet 3.7** 配合 Aider 的效果更好。
- **GGUF Aider 基准测试指南**：一位成员询问了关于如何使用 **GGUF** 运行 **aider 基准测试**的指南。
   - 另一位成员指出，有关于如何针对 API 运行基准测试的文档，这需要使用 llama.cpp 设置一个 API 服务器。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1445900590516273277)** (1 messages): 

> `MCP Apps SDK, Open Source Libraries, Cross-Platform UI` 


- **MCP Apps SDK 正式开源！**：General Intelligence Labs 开源了 [mcp-apps-sdk](https://github.com/General-Intelligence-Labs/mcp-apps-sdk)，使得带有 UI 的 **MCP 驱动应用**能够在各种平台上运行。
   - 开发者现在可以将为 **ChatGPT** 设计的应用嵌入到自己的聊天机器人、助手或 AI 平台中，并在本地进行测试。
- **X 帖子揭示 SDK 开发动机**：一篇 X 帖子（[链接](https://x.com/helloxalia/status/1796319442863866351?s=20)）解释了构建开源 **MCP Apps SDK** 背后的原因。
   - 该帖子详细说明了开发者如何将为 **ChatGPT** 设计的应用嵌入到自己的聊天机器人、助手或 AI 平台中，并在本地进行测试。


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 messages): 

batmanosama: https://arxiv.org/abs/2511.22074
  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1445515200726958140)** (10 messages🔥): 

> `Prompt Security, Custom DSPy OutputFields, Pydantic integration with DSPy, Structured outputs` 


- **Prompt Security：提示层级的安全性**：一名成员讨论了在提示层级实现安全性的难度，认为基于提示的“不要这样做”语句很容易被攻击者绕过；相反，建议通过在训练数据集中包含示例来引导优化器，从而防御基准攻击。
   - 他们建议采用 Guardrails 类型的安全措施，使用特定的模型和调用来检查恶意提示，或利用模型提供商的拒绝机制。
- **自定义 DSPy OutputFields：实现结构化输出**：一位成员询问了自定义 DSPy OutputFields 以及 Pydantic 是否是最佳方案，而另一位成员提到他们正在开发一种自定义的 gemini/nanobanana 图像类型作为输出字段。
   - 讨论涉及生成 text/json/structured output，询问 DSPy 是否有自己的实现，并指出他们可能已经进行了迁移。
- **DSPy 底层使用 Pydantic BaseModel**：会议澄清了 DSPy 底层使用 `BaseModel` 进行验证，并且默认的 `ChatAdapter` 和 `JSONAdapter` 在 LLM 返回输出时执行类型验证。
   - 提供了一个最小示例来演示如何定义一个接收 Pydantic 模型的 Signature，展示了 DSPy 如何利用任何 LLM 生成结构化输出，参考 [代码片段](https://gist.github.com/prrao84/1fc7e17b49707f1346c5702525971f41)。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1445511346320965633)** (12 messages🔥): 

> `Chatmode Feature, AI agent advertisement, Account Suspensions, RAG pipelines` 


- **Chatmode 回归**：用户讨论了平台中 **Chat Mode** 的回归；其他人建议使用 **Qwen** 或 **DeepSeek** 的随机实例也可以实现同样的效果。
   - 一位用户确认该功能可在“更多 (more)”板块中找到。
- **AI 工程师宣传 Agent 构建专业能力**：一位 AI 工程师发布了关于其在构建**自主 AI Agent** 和**多 Agent 系统**方面的专业能力广告，提到的能力包括研究、数据采集、任务自动化、授权、协作和规划。
   - 广告还列出了具体的技术和工具，如 **JS/TS**、**Next.js / Vue**、**Go / Rust**、**Python**、**Langraph**、**AutoGen**、**ReAct**、**CrewAI**、**DeepSeek**、**OpenAI**、**Claude**、**Hugging Face** 以及各种 API。
- **账号封禁：推荐行为引发怀疑**：一名成员询问为什么向几个人提供推荐码会导致其账号被封禁。
   - 消息中未提供进一步的信息或解决方案。
- **AI 工程师专注于 RAG 流水线**：一位工程师专注于 **RAG 流水线**，宣称其在生产环境中使用*混合搜索 (hybrid search)*和*自定义检索 (custom retrieval)*来实现准确且感知上下文的响应。
   - 该工程师还列出了在 **AI 内容检测**、**图像 AI** 和**语音 AI** 方面的专长，包括开发审核工具、打标签流水线和个性化语音助手。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1445562215083147265)** (6 messages): 

> `Fixing test failures in tinygrad, Performance improvements using shrink vs indexing, RMSNorm usage clarification` 


- **针对失败的 Tinygrad 测试的修复方案即将就绪**：一名成员报告了使用 `CPU=1 PYTHONPATH="." pytest -n 12` 运行测试时的失败情况，特别是 `test/test_tiny.py TestTiny.test_beam` 等，并提供了完整日志。
   - 另一名成员提到一个 [Pull Request](https://github.com/tinygrad/tinygrad/pull/13553) 几乎修复了这些问题。
- **对于张量索引，Shrink 速度极快**：一名成员建议在处理张量时，使用 `Tensor.shrink((None, (0, input_size)))` 比 `obs[:, :input_size]` 更快。
   - 他们还注意到将 `Variable` 的 `vmin` 调至 2 以避免错误，但困惑于为什么使用 `Variable` 会导致代码变慢 5 倍（16.61M vs 81.9M SPS）。
- **RMSNorm 参数困惑**：一名成员建议查看 `RMSNorm(dim=-1)` 的源代码，以确保其行为符合预期。
   - 该指导暗示了在 `RMSNorm` 的使用方式上可能存在误解或配置错误。


  

---

### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1445846253018546307)** (5 条消息): 

> `MCP Security Risks, Security risks associated with MCP, MCP-specific security` 


- **Redditors 讨论 MCP 安全风险**：一位用户就其对 **MCP** 相关安全风险的看法征求反馈，并附上了一个 [reddit 帖子](https://www.reddit.com/r/programming/comments/1pd1heu/comment/ns3ntnx/)的链接。
   - 另一位成员回复了一篇关于 **MCP 特有安全事项**的博客文章链接，称其为*极好的资源*：[den.dev/blog/security-rakes-mcp/](https://den.dev/blog/security-rakes-mcp/)。
- **另一个 MCP 安全资源**：这是另一个资源 [MCP Security @ Reddit Thread](https://www.reddit.com/r/programming/comments/1pd1heu/comment/ns3ntnx/)


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1445792345235918869)** (1 条消息): 

> `Tool Validation, Server-Side Validation` 


- **无工具采样需要服务端验证**：一位成员询问，在没有工具证明其存在的情况下，如果发生无工具采样，服务端是否应该进行验证。
   - 该问题围绕如何确保在使用采样方法且缺少预期工具或其存在证明时，在服务端正确验证该过程。
- **服务端验证对无工具采样至关重要**：讨论强调了在没有验证工具的情况下进行采样时，服务端验证的重要性。
   - 它确保即使在缺乏直接工具验证的情况下，采样过程也能遵循所需的协议和标准。