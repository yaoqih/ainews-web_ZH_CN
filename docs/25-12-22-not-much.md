---
companies:
- zhipu-ai
- xiaomi
- google
- langchain
- huggingface
- openrouter
- artificial-analysis
- vllm-project
date: '2025-12-22T05:44:39.731046Z'
description: '智谱 AI 发布的 **GLM-4.7** 在**编程、复杂推理和工具使用**方面取得了显著进步，并已通过 Hugging Face 和
  OpenRouter 迅速获得生态普及。小米的 **MiMo-V2-Flash** 被视为一款实用且高性价比的混合专家模型（MoE），并针对部署进行了优化。在开源权重文生图领域的竞争中，**Z-Image
  Turbo** 凭借 Apache-2.0 协议下的 60 亿参数量处于领先地位。


  视频模型的进展则侧重于控制力和长视频一致性，代表性成果包括**可灵 2.6（Kling 2.6）的运动控制**以及 MemFlow 的自适应记忆检索等研究。在智能体（Agent）框架方面，谷歌的
  **A2UI 协议**引入了智能体驱动的 UI 生成；同时研究表明，混合使用多个智能体框架已成为常态，但在逻辑、终止判定和工具交互方面仍面临挑战。此外，LangChain
  强调了生产级智能体对持久化记忆模式的需求。'
id: MjAyNS0x
models:
- glm-4.7
- mimo-v2-flash
- z-image-turbo
- kling-2.6-motion-control
people:
- mervenoyann
- eliebakouch
- omarsar0
- osanseviero
- dair_ai
title: 今天没发生什么事。
topics:
- coding
- complex-reasoning
- tool-use
- mixture-of-experts
- cost-efficiency
- open-weight-models
- text-to-image
- video-models
- memory-persistence
- agent-frameworks
- interactive-user-interfaces
- model-deployment
---

**干得漂亮，中国 AI**

> 2025年12月22日至12月23日的 AI 新闻。我们为您检查了 12 个 subreddits、544 个 Twitter 账号和 24 个 Discord 社区（包含 208 个频道和 4321 条消息）。预计为您节省阅读时间（以 200wpm 计算）：305 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以美观的 vibe coded 风格展示所有往期内容。请访问 https://news.smol.ai/ 查看完整的详细新闻，并在 @smol_ai 上向我们提供反馈！

Z.ai 的 [GLM 4.7](https://z.ai/blog/glm-4.7) 和百度的 ERNIE 5.0 几乎入选，但前者属于增量更新，而后者尚未发布。

不过，来自前沿 Agent 实验室的三个全新 AIE CODE 演讲**已经**发布：

- [Factory AI](https://www.youtube.com/watch?v=ShuJ_CN6zr4)
- [Amp Code](https://www.youtube.com/watch?v=gvIAkmZUEZY)
- [Repit Agent](https://www.youtube.com/watch?v=MLhAA9yguwM)

请享用。

---

# AI Twitter 综述

**权重开放模型发布：GLM‑4.7、MiMo‑V2‑Flash 以及图像/视频模型的更迭**

- **智谱 AI（Zhipu AI）发布 GLM‑4.7 及其生态系统的快速采纳**：智谱将 **GLM‑4.7** 定位为在 GLM‑4.6 基础上的重大进步，在**编程、复杂推理和工具使用**方面有所提升（权重已上传至 HF，并附带技术博客和托管聊天服务），见 [@Zai_org](https://twitter.com/Zai_org/status/2003156119087382683)。该发布迅速出现在各个分发和评估层面：[@mervenoyann](https://twitter.com/mervenoyann/status/2003162322181976553) 指出其在发布首日即可通过 HF 工具链使用；[@OpenRouterAI](https://twitter.com/OpenRouterAI/status/2003196169632243815) 已将其列入 OpenRouter；[@arena](https://twitter.com/arena/status/2003159444822327748) 记录了其在 LM Arena Code Arena 的变动（声称在 WebDev 排行榜上占据**开源模型第 1 名**，总榜**第 6 名**；较 GLM‑4.6 提升 83 分）。一些从业者还指出“交织思考（interleaved thinking）”行为发生了变化，并建议使用**官方 API 进行基准测试** ([@eliebakouch](https://twitter.com/eliebakouch/status/2003163924716466287))。
- **小米 MiMo‑V2‑Flash：针对部署优化的“实用型” MoE**：一波评论将 **MiMo‑V2‑Flash** 描述为针对**成本/速度/可部署性**而非单纯追求排行榜表现而优化的模型。[@omarsar0](https://twitter.com/omarsar0/status/2002768840556728714) 强调其声称能以更少的参数抗衡强大的开源同行；代码库链接见 [@omarsar0](https://twitter.com/omarsar0/status/2002768968713699747)。一份以知乎为中心的综述强调了 Agent 工作流和极具冲击力的价格（例如引用的 $0.1 / 1M input tokens），同时对稳定性持保留意见 ([@ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/2003135877606760816))。vLLM 在 [@vllm_project](https://twitter.com/vllm_project/status/2002938138549682366) 中发布了**官方推理方案**，提供了针对上下文长度/延迟/KV cache 以及 DP/TP/EP 配置的具体调节参数。
- **权重开放的文本生成图像竞争加剧**：Artificial Analysis 报告称 **Z‑Image Turbo** 成为其 Image Arena 中新的**权重开放文本生成图像模型第 1 名**，拥有 **6B 参数**，采用 **Apache‑2.0** 协议，并进行了价格对比（例如在阿里云上为 $5/1k 张图像），见 [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/2002839525609865575)。
- **视频模型进展集中在控制能力和长视频一致性上**：多个讨论围绕 **Kling 2.6（可灵）运动控制**展开（用户演示、舞蹈/动作控制以及 fal 的首日支持）。参见 [@fal](https://twitter.com/fal/status/2003103565309415665) 的示例以及 [@IamEmily2050](https://twitter.com/IamEmily2050/status/2002968479276937403) 等高互动量的用户展示。在研究方面，MemFlow 提出了一种用于长流式叙事的自适应记忆检索方法 ([@HuggingPapers](https://twitter.com/HuggingPapers/status/2002714237492138434))。

**生产环境中的 Agent：协议、编排、记忆、沙箱以及“可观测性优先”的工程实践**

- **Google 为 Agent 驱动的 UI 推出的 A2UI 协议**：一个值得关注的基础设施/API 发布是 **A2UI (Agent‑to‑User Interface)**，这是一个开源协议，允许 Agent 生成交互式用户界面 ([@osanseviero](https://twitter.com/osanseviero/status/2002747011230269893))。这一构想暗示了 Agent 正在从“仅限聊天”向作为拥有标准界面层的 **UI 生成器**转变。
- **Agent 框架现状调研：混合使用框架已成常态**：[@dair_ai](https://twitter.com/dair_ai/status/2003178236696776814) 总结的一项大型实证研究声称，在 1,575 个 Agent 项目中，**96%** 的高星项目结合了多个框架（例如 LangChain+LlamaIndex；AutoGen+LangChain），而 GitHub Star 数并不能预测采用率。报告的痛点集中在**逻辑失败**、**终止检测**、**Agent 与工具交互**以及**版本兼容性**。
- **记忆模式与持久化成为一等公民**：LangChain 强调了一个由 Oracle 支持的 Agent 中心，具备持久化存储和“六种记忆模式” ([@LangChainAI](https://twitter.com/LangChainAI/status/2002771047234613550))，这反映出生产环境中的 Agent 越来越多地受到**状态、召回和可审计性**的约束，而非仅仅取决于原始模型的 IQ。
- **沙箱化/异步 Agent 执行进入工程化阶段**：目前存在一种明显的趋势，即将编程 Agent 推入**隔离执行环境**（企业级沙箱、可复现的“蓝图”、追踪保留）。参见：通过 [@hwchase17](https://twitter.com/hwchase17/status/2002801655801385037) 展示的 Runloop + DeepAgents 示例，以及一篇关于在 Modal 沙箱中使用 Claude Code 实现“在家运行异步编程 Agent”的文章 ([@andersonbcdefg](https://twitter.com/andersonbcdefg/status/2002829629187608794))。相关内容：一个针对 Agent 的“更好的 git”提案 (zagi)，专注于上下文高效的 diff、审计和轨迹分叉 ([@mattzcarey](https://twitter.com/mattzcarey/status/2002796068811976885))。
- **可观测性作为缺失的规范**：一篇关于“个人工作流的 LLMOps”的代表性文章指出，许多感知到的模型退化实际上是**指令歧义、缺失上下文或任务分解不佳**造成的，这些问题只有在通过 LangSmith 等工具进行追踪后才会变得明显 ([@ChaiWithJai](https://twitter.com/ChaiWithJai/status/2002895889690407382))。这与反复呼吁将 AI 工程视为后端工程的观点一致：埋点、记录日志、评估——不要“凭感觉”调试。

**基准测试、评估政治以及“进步”应如何衡量 (METR, Arena, FrontierMath, SWE-bench)**

- **METR 风格的评估与“验证瓶颈”构想**：一个反复出现的主题是，RL 的进展受限于**验证时间**而非任务长度；提议的改进方案是绘制能力与**验证所需时间**的对比图 ([@ShashwatGoel7](https://twitter.com/ShashwatGoel7/status/2002732250681766347))。另一些讨论批评了令人困惑的报告字段，并认为如果没有单个任务的详细分解，聚合的“工作时间”和总成本并没有太大参考价值 ([@scaling01](https://twitter.com/scaling01/status/2002793892773544154))。
- **Arena 驱动的模型叙事持续发挥作用**：除了 GLM‑4.7 在 Code Arena 上的排名跃升 ([@arena](https://twitter.com/arena/status/2003159444822327748))，百度的 **ERNIE‑5.0‑Preview‑1203** 也凭借初步的评分声明登上了 LM Arena 文本排行榜的前列 ([@arena](https://twitter.com/arena/status/2003151045946376482))。
- **开源模型在 SWE‑bench 上的“追赶”信号**：一份简明快照声称，开源编程模型在 **SWE-bench verified** 上的表现正接近闭源模型（GLM‑4.7 **73.8%**，Kimi K2 Thinking **73.4%**，DeepSeek‑V3.2 **73.1%**，Claude Sonnet 4.5 **77.2%**），并强调了 GLM‑4.7 在数学和工具使用方面的优势 ([@cline](https://twitter.com/cline/status/2003181058679029915))。
- **FrontierMath 访问不对称性成为讨论的一部分**：Epoch AI 指出，开源权重的中国模型在 FrontierMath 各层级上落后顶尖前沿性能约 7 个月，同时也指出 OpenAI 拥有对 Tier 1–3 大部分数据和解决方案的独家访问权 ([@EpochAIResearch](https://twitter.com/EpochAIResearch/status/2003178174310678644))。

**RL、蒸馏以及安全/保障循环：目前正在扩展什么 vs 缺失什么**

- **“RL 之年” → “蒸馏之年” 的梗与真实信号交汇**：蒸馏（Distillation）同时出现在产品传闻和部署叙事中。一个值得注意的说法是：**Gemini 3 Flash 使用了蒸馏预训练** ([@yifan_zhang_](https://twitter.com/yifan_zhang_/status/2002745931649933724))。另外，多位评论员预测蒸馏将成为下一个周期的驱动力（例如 [@leithnyang](https://twitter.com/leithnyang/status/2002795896170541456)）。
- **RL 基础设施民主化**：**OpenTinker** 被定位为一个解耦的客户端/服务器 RL 框架，专为 LLM 设计——“在 GPU 集群上配置一次后端；在本地定义环境；远程训练”——旨在将 RL 流水线的搭建时间缩短约 10 倍 ([@youjiaxuan](https://twitter.com/youjiaxuan/status/2002838551319253281))。
- **提示词注入 / Agent 安全性演变为操作性 RL**：OpenAI 描述了如何使用**自动化红队测试 + 强化学习 + 快速缓解循环**来强化其浏览器 Agent (ChatGPT Atlas) 以抵御提示词注入 ([@cryps1s](https://twitter.com/cryps1s/status/2003182649662140620))。这是一个将 RL 作为持续**安全维护循环**而非仅仅是能力加速器的具体案例。
- **研究品味：算法进步 vs 算力**：一段被广泛转发并归功于 Sergey Brin 的话主张“算力是甜点；算法是主菜”，断言在过去十年中，算法的进步已经超过了 Scaling（规模化）的速度 ([@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2002803245354459588))。这是对纯粹 Scaling 叙事的一个有力制衡——尤其是当它与对更好的验证/评估设计的呼吁相结合时。

**机器人与具身智能：Reachy Mini 的势头、RL 迁移差距以及视频-动作模型**

- **Reachy Mini 成为“节日机器人平台”**：多位开发者报告了其快速的设置和精细的 UX（手册 + App + SDK），并计划开发本地助手；例如 [@Prince_Canuma](https://twitter.com/Prince_Canuma/status/2002695729442402496) 和 [@chenosaurus](https://twitter.com/chenosaurus/status/2002826732525773212)。该仓库也在趋势榜上走红 ([@PoratDor](https://twitter.com/PoratDor/status/2003027940078993798))。
- **Sim-to-real 甚至机器人到机器人的迁移仍然困难**：John Carmack 描述了一些实验，其中完美的仿真器在迁移到真实的摄像头/伺服电机设置时表现不佳，甚至在理论上相同的 3D 打印设备之间迁移策略也会导致性能损失——这需要通过持续在线学习来恢复 ([@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/2002773760672223265))。
- **机器人学习栈扩展到 “LLM 控制机器人” 之外**：一种用于机器人学习的“视频-动作模型（video-action model）”类别被引入 (mimic-video) ([@elvisnavah](https://twitter.com/elvisnavah/status/2003088362119512560))，Chelsea Finn 分享了在“机器人奥林匹克”任务上的微调结果 ([@chelseabfinn](https://twitter.com/chelseabfinn/status/2003165418098446339))——两者都指向感知骨干网络与动作策略之间更紧密的耦合。

**对工程师仍然重要的文化侧信号：Slop、 “LLM 精神错乱” 以及界面人体工程学**

- **“LLM 精神错乱” / 妄想叙事激增——通常围绕数学证明**：几篇高参与度的帖子警告说，模型已经好到足以误导甚至专家 ([@_lewtun](https://twitter.com/_lewtun/status/2002690691705794805))，多个帖子对解决千禧年大奖难题的说法进行了嘲讽，将其定性为狂热和“氛围感驱动”的胡言乱语 ([@suchenzang](https://twitter.com/suchenzang/status/2002774256783077420)；[@BlackHC](https://twitter.com/BlackHC/status/2003156071460843734))。对于工程师来说，重要的启示不是这些闹剧，而是**快速流畅的输出造成了验证危机**，除非工作流强制执行检查。
- **编程 Agent 的人体工程学正成为产品的切入点**：反复的对比认为 Claude Code 的 UI 示能（Affordances，如计划模式、询问后编辑等）实质上优于 Codex 当前的交互设计 ([@finbarrtimbers](https://twitter.com/finbarrtimbers/status/2002765191134732642))。这与更广泛的“上下文工程（Context Engineering）”重构相一致——即从 Prompt 转向管理工具、内存和策略 ([@TheTuringPost](https://twitter.com/TheTuringPost/status/2002765247900262620))。

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. 2023年主要开源 AI 模型发布

- [**今年主要的开源发布**](https://www.reddit.com/r/LocalLLaMA/comments/1pstlas/major_opensource_releases_this_year/) (热度: 746): **该图片展示了 2023 年重大的开源发布，突显了 AI 和机器学习领域的进步。值得注意的发布包括 Deepseek 的开源推理模型、Qwen 的视觉和图像编辑模型，以及阿里巴巴的写实图像生成模型。其他关键发布还包括 Google 的 Gemma 3、Meta 的 SAM 模型以及 Nvidia 的 Nemontron 3。该帖子强调了开源与闭源技术之间差距的缩小，暗示了 AI 开发格局的转变。** 评论者注意到中国公司在开源领域的主导地位，名单中仅有三家美国公司。人们对 Deepseek 未来的发布也充满期待，预计其在推理能力上可能超越闭源模型。此外，还有关于 Mistral 在小尺寸模型上表现的讨论。
    - **Maximum** 讨论了即将发布的 DeepSeek 模型的潜力，强调其在“3.2 speciale”上的训练。评论者预计 DeepSeek 可能会超越闭源模型，特别是在推理任务方面，这标志着开源 AI 能力的重大进步。
    - Sufficient-Bid3874 提出了关于 Mistral 性能的问题，特别是在小型模型背景下。这引发了关于 Mistral 在资源受限环境中的效率和有效性的讨论，这对于需要轻量化模型的应用至0关重要。
    - mukz_mckz 强调了 Olmo 对开源社区贡献的重要性，特别是通过他们的论文、博客文章和代码库。评论指出，Olmo 为各种训练参数（“knobs”）如何影响模型性能提供了宝贵的见解，为开发者和研究人员提供了学习机会。
- [**我制作了 Soprano-80M：延迟 <15ms 的超写实流式 TTS，高达 2000 倍实时，且显存占用 <1 GB VRAM，以 Apache 2.0 协议发布！**](https://www.reddit.com/r/LocalLLaMA/comments/1pt3sco/i_made_soprano80m_stream_ultrarealistic_tts_in/) (热度: 530): **Soprano-80M 是由 Eugene 开发的新型 TTS 模型，在文本转语音转换中实现了前所未有的速度和效率。它的音频流延迟低于** `15 ms`**，可以在** `20 秒内生成 10 小时的有声读物`**，达到** `~2000x 实时` **的性能。关键创新包括用于更清晰音频的更高** `32 kHz` **采样率、用于更快生成的基于声码器的解码器（**`~6000x 实时`**），以及一种新颖的神经音频编解码器，可在** `0.2 kbps` **下将音频压缩至** `~15 tokens/sec`**。该模型专为超快速、自然的语音生成而设计，尽管目前缺乏声音克隆和多语言支持等功能。[GitHub](https://github.com/ekwek1/soprano), [Huggingface Demo](https://huggingface.co/spaces/ekwek/Soprano-TTS), [Model Weights](https://huggingface.co/ekwek/Soprano-80M)。** 评论者注意到了该模型令人印象深刻的速度，但也提到了音频质量问题，如吐字不清和伪影，尤其是在较长的输出中。人们对实现所述性能所使用的硬件感到好奇，因为类似模型在高端 GPU 上的实时因子明显较低。
    - Chromix_ 指出了 Soprano-80M 的性能问题，指出虽然它速度极快，但生成长音频文件可能会导致吐字不清、噪音、重复和伪影。正如分享的音频链接所示，模型性能在超过一分钟后会下降。这表明模型在长时间维持质量方面存在潜在局限。
    - coder543 质疑了实现声称的 2000 倍实时性能所使用的硬件，并将其与他们在 RTX 3090 上运行 Kokoro-82M（尺寸相似的模型）仅获得 50 到 100 倍实时的经验进行了对比。这引发了关于在不同硬件条件下性能声明可复现性的疑问。
    - geneing 讨论了 Soprano-80M 的架构，指出它使用一个小型的 Qwen3 LLM 来生成 vocos 特征，然后由 vocos 进行解码。他们对该模型在实际应用中的准确性表示怀疑，因为其 LLM 尺寸较小，并认为 LLM 小于 0.5B 的模型可能会遇到质量问题，特别是在处理复杂的语言任务（如英语发音）时。

### 2. GLM 4.7 发布与特性

- [**GLM 4.7 已在 HF 发布！**](https://www.reddit.com/r/LocalLLaMA/comments/1pt5heq/glm_47_is_out_on_hf/) (活跃度: 660): **GLM 4.7 已在 [Hugging Face](https://huggingface.co/zai-org/GLM-4.7) 发布，展示了在多语言编程、UI 生成和复杂推理方面相较于 GLM 4.6 的改进。它在** `73.8%` **的 SWE-bench 和** `42.8%` **的 HLE 上取得了优异成绩，并引入了 *Interleaved Thinking*（交织思考）、*Preserved Thinking*（保留思考）和 *Turn-level Thinking*（轮次级思考）等特性以增强任务管理。该模型可以使用 vLLM 和 SGLang 进行本地部署，GitHub 上提供了集成说明。一个显著的特点是在推理/规划中使用图表，这在该领域尚属首次。** 用户对基准测试持怀疑态度，一些用户认为 GLM 4.7 是对 DeepSeek 3.2 的快速增量改进，但并未超越 Sonnet 4.5。在编程能力方面，它可能与 Gemini 3 Flash 相当。
    - Dany0 对 GLM 4.7 提供的基准测试表示怀疑，认为虽然它可能是对 DeepSeek 3.2 的快速且更好的增量改进，但不太可能超越 Sonnet 4.5。评论者推测 GLM 4.7 在编程能力方面可能与 Gemini 3 Flash 旗鼓相当，但采用了不同的架构。
    - AnticitizenPrime 强调了 GLM 4.7 的一个新颖特性，即在推理和规划阶段包含图表。这被认为是此类模型的首创，可能会增强模型处理需要视觉规划和推理的复杂任务的能力。
    - waste2treasure-org 和 jacek2023 分别对 Gemma 4 和 Air 的缺席表示失望，表明了对这些模型的需求。这说明虽然 GLM 4.7 是一个重要的发布，但社区对其他模型仍有期待。
- [**GLM 4.7 发布！**](https://www.reddit.com/r/LocalLLaMA/comments/1pt5jfn/glm_47_released/) (活跃度: 309): **GLM-4.7 已经发布，在编程、复杂推理和工具使用等领域提供了相较于前代 GLM-4.6 的显著进步，确立了新的开源 SOTA 标准。该模型还增强了在对话、创意写作和角色扮演场景中的表现。值得注意的是，GLM-4.7 引入了新的认知特性，如 *Interleaved Thinking*、*Preserved Thinking* 和 *Turn-level Thinking*，通过在行动之间启用思考过程并保持交互的一致性，提高了任务的稳定性和控制力。模型权重可在 [Hugging Face](http://huggingface.co/zai-org/GLM-4.7) 获取，更多技术细节见[技术博客](http://z.ai/blog/glm-4.7)。** 评论者强调了 GLM 模型快速的开发周期，并对 Unsloth UD_Q2_K_XL 量化表示期待，该量化此前曾提升了 GLM 模型的性能。新思考模式的引入被视为处理复杂任务的重大改进。
    - ResearchCrafty1804 强调 GLM-4.7 引入了新的认知特性，如 Interleaved Thinking、Preserved Thinking 和 Turn-level Thinking。这些增强功能旨在通过在不同交互中保持一致性和稳定性，提高模型处理复杂任务的能力。更多细节可以在[文档](http://docs.z.ai/guides/capabilities/thinking-mode)中找到。
    - r4in311 使用生成“体素宝塔”（Voxel Pagoda）的特定提示词，对 GLM 4.7 与 GPT 5.0 和 Sonnet 4.5 等其他模型进行了对比分析。结果显示，虽然 GLM 4.7 具有竞争力，但它需要更多的尝试和故障排除才能达到类似的结果，这表明与同类模型相比，它尚未达到 SOTA 水平。[jsfiddle](https://jsfiddle.net/zhrqmw4p) 中提供的示例说明了这些差异。
    - UserXtheUnknown 指出 GLM 4.7 在特定任务“旋转房屋演示”中表现异常出色，甚至超越了 Gemini 3.0。这表明虽然 GLM 4.7 可能并非在所有领域都是最强的，但它在某些任务中表现优异，展示了其在特定应用中的潜力。

### 3. NVIDIA 的 DGX Spark 和 Unsloth 指南

- [**NVIDIA 制作了使用 Unsloth 微调 LLM 的入门指南！**](https://www.reddit.com/r/LocalLLaMA/comments/1pt18x4/nvidia_made_a_beginners_guide_to_finetuning_llms/) (热度: 379): **NVIDIA 关于使用 Unsloth 微调大语言模型 (LLMs) 的指南全面概述了如何利用 NVIDIA GPUs 进行模型定制。该指南涵盖了三种主要的微调方法：LoRA (Low-Rank Adaptation)、FFT (Full Fine-Tuning) 和 RL (Reinforcement Learning)，并讨论了必要的数据和 VRAM 需求。它强调使用 Unsloth 这一开源框架，结合 NVIDIA 的 DGX Spark 和 RTX GPUs，高效地为特定任务量身定制模型。该指南还介绍了 Nemotron 3 系列开源模型，彰显了 NVIDIA 对开源 AI 发展的承诺。** 一位评论者赞赏 NVIDIA 对开源模型的贡献，但批评了该公司对硬件市场的影响。另一位用户对访问问题表示沮丧，表示需要内容的镜像。
- [**DGX Spark：一个非主流观点**](https://www.reddit.com/r/LocalLLaMA/comments/1ptdtmz/dgx_spark_an_unpopular_opinion/) (热度: 367): **图片展示了 NVIDIA DGX Spark，这是一个专为数据科学和机器学习任务设计的紧凑型计算单元，特别适用于高性能 GPU 获取受限的环境。该帖子强调了它对小型研究小组的实用性，突出了其一体化设计和巨大的内存容量，这使得基础模型的有效原型设计和训练成为可能。尽管速度无法与 H100 等高端 GPU 媲美，但其设计使其能够被资金有限的团体所使用。评论讨论了它在 VRAM 和能效方面的优势，同时也指出了它与 3090 等其他 GPU 相比在内存带宽和性能方面的局限性。** 评论者普遍认为 DGX Spark 非常适合其目标人群（如小型研究小组），尽管对其内存带宽感到有些失望。大家达成共识，认为它是进入 NVIDIA 生态系统的切入点，并期望未来能扩展到更强大的 GPU。
    - Kwigg 指出，DGX Spark 虽然提供了充足的 VRAM 和高效的功耗，但在内存带宽方面表现不足，这使得它在 LLM 推理任务中的性价比不如其他选择。对于关注推理而非训练的用户来说，这是一个关键点，因为内存带宽是一个显著的瓶颈。
    - FullstackSensei 指出，Nvidia 推出 DGX Spark 的策略是以较低成本引导用户进入 CUDA 生态系统，特别是针对教育机构。这种方法旨在创造对 Nvidia 生态系统的依赖，从而鼓励未来对更大、更昂贵的 GPU 集群进行投资。
    - pineapplekiwipen 将 DGX Spark 与 3090 等消费级 GPU 进行了比较，指出虽然 Spark 速度较慢，但能效更高。然而，在价格和性能方面，配置多个 3090 的方案可能会优于单个 DGX Spark，这凸显了功耗与计算效率之间的权衡。

## 非技术性 AI 版块汇总

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Gemini 和 SCAIL 模型进展

- [**Gemini 3 Flash 可以可靠地数手指 (AI Studio – 高推理能力)**](https://www.reddit.com/r/singularity/comments/1psx30g/gemini_3_flash_can_reliably_count_fingers_ai/) (热度: 873): **Gemini 3 Flash 是来自 AI Studio 的一个模型，通过在图像中可靠地数手指展示了极高的推理能力。该模型在视觉识别任务中表现出显著改进，特别是在复杂场景中准确识别和计数物体方面。该模型处理手部位置和光照条件变化的能力尤为突出，而这些正是计算机视觉应用中的常见挑战。** 评论者讨论了该模型在不同条件下的鲁棒性，并指出了其在手势识别和人机交互等现实场景中的潜在应用。一些人对实现如此高准确率的底层架构和训练数据表示感兴趣。
    - 讨论强调了 Gemini 3 Flash 模型在准确数手指方面的能力，这表明其推理能力较简单的图像匹配有了显著进步。这反驳了 AI 模型仅仅执行模式识别而没有理解能力的常见批评。正如用户所注意到的，该模型能够快速提供正确答案，表明了高效的处理能力以及在视觉推理任务中可能改进的算法。

- [**SCAIL 绝对是复制参考视频动作的最佳模型**](https://www.reddit.com/r/StableDiffusion/comments/1pswlzf/scail_is_definitely_best_model_to_replicate_the/) (热度: 538): **该帖子讨论了用于动作迁移（motion transfer）的 SCAIL 模型，强调了其在不扭曲主角比例的情况下复制参考视频动作的能力，这与 Wan Animate 和 Steady Dancer 等其他模型不同。使用 SCAIL 的工作流通过 [Google Drive 链接](https://drive.google.com/file/d/1fa9bIzx9LLSFfOnpnYD7oMKXvViWG0G6/view?usp=sharing) 进行了分享。** 一位评论者询问运行 SCAIL 模型所需的最低 VRAM，表明了对该模型硬件要求的技术兴趣。
    - 一位用户询问了运行 SCAIL 模型所需的最低 VRAM，这对于确定该模型的硬件兼容性和性能效率至关重要。对于希望在自己的系统上实现或测试模型的用户来说，这是一个普遍关注的问题，尤其是在处理高保真动作复制任务时。
    - 另一位用户表示有兴趣将 SCAIL 模型的输出与原始参考视频进行比较，特别关注手部动作的准确性。这突显了在评估动作复制模型有效性时，详细的动作保真度的重要性，因为手部动作通常很复杂且难以准确复制。
    - 一条评论推测了 SCAIL 取代传统动作捕捉（motion capture）服的潜力，认为如果该模型能够实现高精度和可靠性，它可能为现有的动作捕捉技术提供一种更易于获取且不那么繁琐的替代方案。这指向了 AI 领域的一个更广泛趋势，即软件解决方案越来越多地被考虑作为硬件系统的替代品。
- [**Z-Image + SCAIL (多角色)**](https://www.reddit.com/r/StableDiffusion/comments/1psr58j/zimage_scail_multichar/) (热度: 1349): **该帖子讨论了结合使用 Z-Image 和 SCAIL 来生成多角色动画，强调 SCAIL 的姿态呈现出真正的 3D 效果，与 Wan Animate 或 SteadyDancer 等替代方案相比，具有更好的深度和身体朝向。用户报告称，在使用 RTX 5090 GPU、分辨率为** `736x1280` **的情况下，** `6 steps` **的渲染时间为** `26 minutes`**，这表明了巨大的计算需求。** 评论者们对生成式 AI 在制作舞蹈视频中的盛行提出质疑，对这类动画的来源以及可以生成的视频时长表示好奇。
    - 一位用户询问了使用所讨论技术生成 3D 骨骼动画的潜力。这表明人们对将 Z-Image 和 SCAIL 应用于更复杂的动画任务感兴趣，可能会利用这些模型创建可转化为 3D 环境的逼真动作序列的能力。
    - 另一位用户对生成角色的真实感表示难以置信，表明 Z-Image 和 SCAIL 中使用的模型能够产生高度逼真的图像。这指向了这些模型在实现照片级真实感（photorealistic）结果方面的有效性，这可能会对依赖逼真数字人呈现的行业产生影响。
    - 有人提出了关于使用这些模型可以生成的视频时长的问题。这涉及到 Z-Image 和 SCAIL 在视频长度方面的技术限制或能力，这是媒体制作和其他需要长视频内容领域应用的关键因素。
- [**为精彩的 2026 年做好准备！**](https://www.reddit.com/r/singularity/comments/1pspk5q/prepare_for_an_awesome_2026/) (热度: 1693): **该图片是 Kevin A. Bryan 的一条推文，回顾了截至去年 12 月 1 日 AI 模型和技术的状态。它强调了当时缺乏强大的 Gemini 模型、图像模型在文本理解方面的局限性以及视频模型尚处于萌芽阶段。推文还提到了引入了具有 test time inference 的 Deepseek R1，以及 FrontierMath 和 HLE 的进展，预示着重大的进步和对 2026 年的规划。这一背景暗示了对技术演进的前瞻性视角，以及到 2026 年 AI 和机器学习领域取得突破的潜力。** 评论者们反思了技术变革的飞速步伐，一些人对这些进步到 2026 年能否产生变革性影响表示怀疑。有一种观点认为，虽然广泛的变化可能是渐进的，但在技术前沿可能会发生重大转变，特别是在 AI 和机器学习领域。

- Manfr3dMacx 强调了技术进步与其广泛应用之间的滞后，认为虽然 2026 年在总体上可能感觉与 2025 年相似，但 AI 模型的前沿发展将在顶尖领域创造显著差异。这突显了产品化和大规模采用在实现新技术全部潜力方面的重要性。
- Profanion 注意到图像生成技术的进步，提到在 2025 年 3 月，首个能够稳定处理文本的图像生成器随 GPT-image o1 发布。随后，Nanobanana Pro 和 GPT Image 1.5 进一步推动了这一领域的发展，表明 AI 在处理图像内文本生成等复杂任务的能力正在迅速提升。
- Cagnazzo82 评论了 AI 发展的指数级速度，指出对模型所能实现目标的预期在不断演变。这反映了 AI 领域的一个更广泛趋势，即模型能力正在飞速进步，导致这些技术所能完成任务的基准和目标不断发生变化。

### 2. 创意与工程流程中的 AI

- [**WSJ 刚刚报道了一家初创公司，其中 Claude 基本上就是工程团队**](https://www.reddit.com/r/ClaudeAI/comments/1psoe2e/wsj_just_profiled_a_startup_where_claude/) (Activity: 615): **一位 15 岁的企业家开发了一个 AI 驱动的金融研究平台，每月约有** `50,000` **名用户，主要使用 Anthropic 的 Claude 作为核心工程工具。该平台的创建几乎没有直接编码（约** `10 lines`**），而是利用 Claude 进行软件生成，并使用 ChatGPT 和 Gemini 等其他模型辅助任务。创始人专注于系统设计和分发，而非传统的实现，且在没有员工或传统开发团队的情况下运营。一家上市公司甚至转发了该平台生成的 AI 报告，误将其视为专业研究。[WSJ 文章](https://www.wsj.com/business/entrepreneurship/teenage-founders-ecb9cbd3?st=AgMHyA&reflink=desktopwebshare_permalink)。** 评论中充斥着对该平台缺乏付费客户以及创始人获得来自大科技公司和金融界父母支持的质疑。关于 AI 是否会将 SaaS 创作民主化到贬值的程度也存在争论，一些人认为这是基本的经济原理。
    - 讨论强调了对该初创公司可行性的怀疑，指出其缺乏付费客户，且深受创始人父母的支持，其父母拥有大科技公司和金融背景。这引发了对这类 AI 驱动型企业可持续性和独立性的质疑，尤其是当最初的成功严重依赖外部支持而非市场需求时。
    - 关于 AI 降低 SaaS 产品创建门槛的影响存在辩论。一位评论者认为，如果 AI 显著降低了准入门槛，可能会导致市场饱和和 SaaS 产品的贬值，因为基本经济原理表明，在没有相应需求的情况下增加供应会降低价值。
    - 针对使用 AI 驱动的应用程序（尤其是那些在缺乏监管或专业知识的情况下开发的程序）的安全和隐私影响，人们提出了担忧。窃取个人信息的潜在风险很大，特别是如果应用程序没有强大的安全措施支持，且是由处理敏感数据经验有限的个人开发时。
- [**我为以前没尝试过这个感到非常愚蠢**](https://www.reddit.com/r/StableDiffusion/comments/1psocuo/i_feel_really_stupid_for_not_having_tried_this/) (Activity: 599): **该帖子讨论了一位用户的发现：在 AI 图像生成中使用母语（使用 Z-image Turbo，其采用了 Qwen-3 文本编码器）会导致生成的图像反映出当地的文化和地理特征。这表明模型的训练数据包含了用各种语言标记的图像，从而在用这些语言提示时可以实现更具本土化和文化相关性的输出。在 Flux2 等类似模型的文档中也提到了这种行为，表明使用目标语言进行提示可以增强该地区的视觉一致性。** 一位评论者指出，Flux2 的文档中记录了这一功能，表明特定语言的提示是实现特定地区图像生成的已知技术。另一位评论者幽默地指出，这不适用于克林贡语（Klingon）等虚构语言，突显了模型训练数据的局限性。
    - FrenzyX 强调了 Flux2 文档中的一个技术细节，指出使用目标语言进行提示可以增强该地区的视觉一致性。这表明模型的训练数据包含多样化的语言标签，在使用特定语言提示时会影响输出结果。

- Recoil42 解释说，在 prompts 中使用目标语言会使模型偏向于训练集中使用该语言标记的图像。这意味着模型的表现会受到其训练数据中语言分布的影响，从而影响其对图像的理解和生成方式。
- Goldie_Wilson_ 幽默地指出，这种技术对克林贡语（Klingon）不起作用，这暗示了模型训练数据在较少见或虚构语言方面的局限性。这突显了训练数据集的语言覆盖范围在决定模型能力方面的重要性。
- [**Real image vs Nano Banana Pro vs GPT, can you easily guess which one is real?**](https://www.reddit.com/r/ChatGPT/comments/1pt2mhf/real_image_vs_nano_banana_pro_vs_gpt_can_you/) (Activity: 2644): **该帖子描述了一个实验，将真实图像与使用 Gemini 和 GPT 创建的 AI 生成图像进行对比。GPT 对真实图像进行了描述，并使用该描述在 Gemini 和 GPT 中生成了新图像。目标是看观众是否能区分真实图像和 AI 生成的图像。** 一位评论者指出，区分真实图像和 Gemini 生成的图像具有挑战性，而 GPT 生成的图像则更容易被识别为人工合成。
    - Benboozzled 强调了区分 Gemini 模型生成的图像与真实照片的难度，并指出 Chat 模型生成的图像更容易被识别为人工合成。这表明 Gemini 模型已经达到了挑战人类感知的写实水平（photorealism），而 Chat 模型仍然表现出可检测的伪影或风格线索，揭示了其合成本质。
    - SuddenWerewolf7041 对 AI 生成内容对人类价值和智能的影响表示担忧。评论者认为，AI 模型在创建与人类真实创作无法区分的内容方面日益复杂，可能导致人类技能和辨别真实性能力的“稀释”，从而引发关于 AI 在创意领域作用的伦理和社会问题。
    - sgtcfox 提供了一串数字 (1,3,3,3,1,3) 作为识别 AI 生成图像中真实图像的猜测，表明了一种系统化的应对方法。这反映了区分真实图像与 AI 生成图像的复杂性和潜在错误，强调了当前 AI 模型在图像合成方面的先进能力。
- [**Time-to-Move + Wan 2.2 Test**](https://www.reddit.com/r/StableDiffusion/comments/1pt19u6/timetomove_wan_22_test/) (Activity: 2539): **该帖子讨论了一个使用 mickmumpitz 的 ComfyUI 工作流进行动作动画化的测试，该工作流通过手动移动物体或图像来实现，并分别使用高质量相机和 iPhone 进行了测试。作者选择了质量较低的素材以获得更接地气的感觉，并建议未来可能使用更高质量的素材进行测试。该工作流允许对场景进行创意操作，如链接的 [教程](https://youtu.be/pUb58eAZ3pc?si=EEcF3XPBRyXPH1BX) 所示。** 一位评论者将该演示与 **Corridor Crew** 的视频进行了比较，并对他们用于 `dwpose` 的自定义节点表示感兴趣。另一位评论者询问了用于从视频中移除金属吸管和手指等物体的技术，表明了对动画背后技术流程的兴趣。

### 3. 关于 AI 与智能的辩论

- [**Deepmind CEO Dennis fires back at Yann Lecun: "He is just plain incorrect. Generality is not an illusion."**](https://www.reddit.com/r/singularity/comments/1pt05w7/deepmind_ceo_dennis_fires_back_at_yann_lecun_he/) (Activity: 1158): **这张图片是 DeepMind CEO Demis Hassabis 在社交媒体上的发帖，回应了 Yann LeCun 关于通用智能（general intelligence）是一种幻觉的断言。Hassabis 认为 LeCun 将通用智能与全能智能（universal intelligence）混为一谈，并强调虽然实际系统需要专门化，但通用系统的架构（类似于 Turing Machine）在理论上只要有足够的资源就可以学习任何可计算的东西。这场辩论突显了对智能本质的不同看法，Hassabis 捍卫了 AI 系统实现通用性的潜力，而 LeCun 则认为人类智能是高度专门化的。** 评论反映出人们认可 Hassabis 和 LeCun 等顶尖 AI 研究人员之间辩论的价值，认为此类讨论可以显著促进对 AI 和智能的理解。

- **DeepMind CEO Dennis Hassabis** 与 **Yann LeCun** 之间的辩论集中在 AI 的通用性（generality）概念上。Hassabis 认为通用性并非幻觉，反驳了 LeCun 对通用智能存在性的怀疑。这反映了 AI 研究中关于通用人工智能（AGI）的可行性和定义的更广泛讨论，并对 AI 系统的设计和评估方式产生影响。
- **Yann LeCun** 关于人类并非真正通用智能的断言引发了争议。批评者认为，这种观点暗示由于人类无法执行所有可以想象的任务，因此并非真正的“通用”，从而削弱了通用智能的概念。这场辩论突显了在生物和人工系统中定义和衡量通用性的挑战。
- 讨论涉及 AGI 的哲学和技术层面，**Dennis Hassabis** 强调了 AI 实现类似于人类智能的通用性的潜力。这与 **Yann LeCun** 更加谨慎的立场形成鲜明对比，后者质疑通用智能的实用性和当前的理解。这场辩论强调了对什么是通用智能以及如何在 AI 系统中实现它的持续探索。
- [**ChatGPT isn’t an AI :/**](https://www.reddit.com/r/ChatGPT/comments/1psk1mn/chatgpt_isnt_an_ai/) (热度: 2000): **Reddit 帖子中的图片是一个基于文本的迷因（meme），它批判了将 ChatGPT 和大语言模型（LLMs）理解为 AI 的观点。它认为像 ChatGPT 这样的 LLMs 并不是真正的 AI，而是预测最可能出现的下一个词的统计模型，类似于手机键盘。帖子暗示 LLMs 总是处于“幻觉（hallucinating）”状态，只是偶然显得正确，而非基于任何真实的理解。这反映了对 LLMs 的一种常见误解，LLMs 确实是 AI 的一种形式，尽管与人类认知相比，其理解和推理能力有限。** 评论者普遍同意，虽然 LLMs 是一种 AI，但它们是基于统计原理而非真正的理解运行的。一些人认为这种区别更多是为了美化人类认知，而非本质区别，并指出人类知识也是基于学习经验和记忆的。
    - Kaveh01 讨论了 LLMs 与人类认知中知识的本质，认为两种系统都依赖经验和记忆来生成响应。他们认为，虽然 LLMs 缺乏某些认知能力（如迁移性/transferability），但这种局限性部分归因于其训练数据的文本性质，如果人类的理解受到类似的限制，也会受到同样的约束。
    - Machiavellian_phd 将人类认知过程与 LLMs 进行了类比，强调两种系统都参与预测性处理。他们指出，人类经常通过基于不完整信息预测结果来产生“幻觉（hallucinate）”，这与 LLMs 根据训练数据生成输出的方式类似。该评论强调 AI（包括 LLMs）是一个广泛的类别，涵盖了从复杂模型到像恒温器这样简单反馈机制的各种系统。
    - 讨论涉及了 LLMs 的局限性，特别是它们缺乏迁移性以及在面对训练数据空白时产生“幻觉”的倾向。这被比作人类的认知过程，人类认知也涉及预测和错误，这表明 AI 与人类智能之间的差异可能更多是程度上的，而非本质上的。
- [**What’s the most useful thing ChatGPT can do today that people still don’t realize?**](https://www.reddit.com/r/ChatGPT/comments/1pt4t35/whats_the_most_useful_thing_chatgpt_can_do_today/) (热度: 1302): **一位 Reddit 用户强调了 ChatGPT 在饮食计划方面的实际用途，利用其语音转录功能列出杂货清单并生成用餐创意，从而显著减少食物浪费。另一位用户赞赏 ChatGPT 的对话式学习能力，特别是对于历史和量子物理等小众兴趣，并指出在无法找到人类交流对象时，它能够提供知识渊博的讨论。然而，一些用户对 ChatGPT 在回复中重复出现的介绍性短语表示沮丧。** 评论反映了人们对 ChatGPT 在实际任务和学习中的实用性的赞赏，同时也伴随着对其回复风格的轻微不满。该工具在小众话题上进行知识性讨论的能力尤其受到重视。
    - SylvaraTheDev 强调了 ChatGPT 作为媒体解析器的效用，特别是处理 PDF 文件。AI 可以生成捕捉关键细节的执行摘要，这在许多 PDF 缺乏此类摘要的情况下特别有用。此功能可以通过快速从冗长文档中提取核心信息来显著提高生产力。

- dennis-w220 讨论了 ChatGPT 通过对话进行学习的教育潜力。AI 可以参与历史和量子物理等不同话题的讨论，提供了一个以轻松方式探索兴趣的独特机会。这种对话式学习方法对于在寻找知识渊博的人类对手可能具有挑战性的利基领域寻求知识的爱好者来说非常有价值。
- polarwaves 分享了将 ChatGPT 作为治疗替代品的非常规用途，强调了它在提供情感支持方面的作用。虽然不是专业治疗的完美替代品，但 ChatGPT 为用户提供了一个宣泄和管理焦虑的平台，并具有向 AI 请求反馈和挑战的能力，模拟了治疗互动的某些方面。

---

# AI Discord 摘要

> 由 gpt-5.1 生成的摘要之摘要的摘要
> 

**1. 下一代 LLM 和基准测试走向全球**

- **GLM-4.7 悄然跻身前列**：智谱 (Zhipu) 的 **GLM-4.7** 低调发布，在 [LMArena WebDev 排行榜](https://lmarena.ai/leaderboard/webdev) 上以 **1449** 分成为排名第一的开源 WebDev 模型，并通过 [zai-org/GLM-4.7](https://huggingface.co/zai-org/GLM-4.7) 作为 **GLM-4.7 Air** 封装在 Nous 生态系统中。
    - Zhipu 还推送了 MLX 量化版本，如 [**GLM-4.7-mlx-3Bit**](https://huggingface.co/mrtoots/GLM-4.7-mlx-3Bit) 和 [**GLM-4.7-mlx-4Bit**](https://huggingface.co/mrtoots/GLM-4.7-mlx-4Bit)，而 Moonshot 和 Nous Discord 的用户将其与 **Gemini 3 Pro** 进行了正面比较，并讨论了如何禁用或保留其 **“thinking”** 轨迹以适应不同的推理工作负载。
- **中国的 ERNIE-5.0 和 Solar-Open-100B 大显身手**：百度 (Baidu) 的 **ERNIE-5.0-Preview-1203** 在 [Text Arena 排行榜](https://lmarena.ai/leaderboard/text) 上达到 **1451** 分，比 `1103` 预览版提升了 **23 分**，目前是该基准测试中排名第一的中文文本模型。
    - 与此同时，Upstage 在 Hugging Face 上发布了 **Solar-Open-100B** [upstage/Solar-Open-100B](https://huggingface.co/upstage/Solar-Open-100B)，为开源界提供了另一个重量级 dense model，Nous 等社区正考虑将其作为未来评估的严肃替代方案。
- **Phi4 对阵 GPT-OSS 与风格微调前沿**：Unsloth 用户传阅了一篇 **arXiv** 预印本 [“Phi4”](https://arxiv.org/html/2508.12461v1)，声称 **Phi4** 的性能可以超越 **GPT-OSS-20B**，这让那些认为优秀的 dense **14B** 模型只有在强大的架构和数据选择下才能明显击败 **20B MoE** 的人感到惊讶。
    - 与此同时，Nous 研究人员辩论了微调大型模型以复制诸如 *华丽辞藻 (purple prose)* 或 *现实主义小说* 等风格所需的样本量，强调了 Phi4/GLM/Solar 等前沿模型现在的评判标准不仅是原始基准测试分数，还包括 **可控风格迁移 (controllable style transfer)**。

**2. 训练与推理性能军备竞赛**

- **TorchAO MXFP8 助力 MoE 训练加速**：**torchao v0.15.0** 版本增加了 **MXFP8 MoE 训练**，并显示在 **64 节点 GB200 Crusoe 集群**上训练 **Llama4 Scout** 时，与 **bf16** 相比，在相同收敛情况下实现了 **1.2 倍的端到端加速**，详见 [torchao v0.15.0 说明](https://github.com/pytorch/ao/releases/tag/v0.15.0)。
    - MXFP8 MoE kernel 现在以二进制构建形式发布，适用于 **CUDA 12.8+**，支持 **safetensors** 和参数级量化，因此用户可以通过 *pip install* 安装，而无需从源代码编译，且仍能获得用于大规模 MoE 训练的生产级低精度 kernel。
- **QSInference 在长上下文处理上超越 FlashAttention**：一位 GPU MODE 成员分享了 **QSInference**，这是一个针对长上下文 LLM 的 **量化稀疏注意力 (quantized sparse attention)** 的 Triton 实现，在 [QSInference GitHub 仓库](https://github.com/yogeshsinghrbt/QSInference) 中声称在 **128k** 上下文下比 *FlashAttention-2 快 8 倍*，比 *block-sparse attention 快 3 倍*。
    - QSInference 针对注意力计算占主导地位的长序列推理，其 Triton kernel 吸引了正在处理 **B200/GB200** 硬件、**vLLM** 中的 **Helion** kernel 以及 **MXFP8** 等混合精度方案的性能工程师的兴趣。
- **现实世界硬件见闻：Strix Halo、二手 GPU 和 Kernel 悬赏**：LM Studio 用户报告称，配备共享 RAM 的 **Strix Halo** APU 在 **MoE** 模型上表现良好，但在 dense 模型上表现不佳，理由是 dense 模型 *“同时对每个参数进行计算”*，而 MoE 仅触及子集，并指向 [Max Kruse 的模型类型指南](https://maxkruse.github.io/vitepress-llm-recommends/model-types/) 以获取直观理解。
    - 在 GPU MODE 和 LM Studio 中，人们辩论了二手 **3090/3090 Ti** 加 **NVLink** 与新卡的对比，提交了如 `vectoradd_v2` 在 **B200** 上达到 **233 µs** 并获得 **第一名** 的 kernel 基准测试，并交流了关于 **Triton/Gluon** kernel 的调试故事，其中 `wgmma.mma_async` 被序列化，强调了现在的性能在很大程度上取决于底层 kernel 工艺。

**3. Agent 框架、协议与 SDK 趋于成熟**

- **Vercel AI SDK 6 全力投入 Agent 和 MCP**：Vercel 发布了 **AI SDK 6**，支持 **本地 Agent**、**工具执行审批**、完整的 **Model Context Protocol (MCP)** 集成、增强的 DevTools 以及标准化的 **JSON-schema** 工具，该消息在 [AI SDK 推文](https://xcancel.com/aisdk/status/2003156089177792827?s=46) 中公布。
    - Latent Space 成员认为这标志着 SDK 正在跟上现代 **Agentic** 模式：接入 MCP 工具，强制执行人机回环（human-in-the-loop）审批，并为前端团队提供一套功能完备（batteries-included）的技术栈，用于构建多模型、使用工具的应用，而不是编写脆弱的自定义编排器。
- **MCP 贡献者在 Token 经济学上角力**：在官方 **MCP Contributors** 服务器中，开发者抱怨当前的 MCP 集成在每次调用时都会**重新发送庞大的工具协议描述**，即使大多数工具并未被使用，这也会推高 Token 成本。
    - 他们提出了**延迟工具 Schema 传输**和**缓存协议定义**等想法，但维护者澄清说，Schema 的发送由 *Client Host* 控制，工具在没有 Schema 的情况下**无法被调用**，因此任何修复方案都必须依赖于更智能的客户端缓存和变更通知处理，而非核心协议本身。
- **SmolAgents、DSPy Skills 和 OpenRouter SDK 旨在实现更智能的自动化**：Hugging Face 贡献者通过 [PR #1912](https://github.com/huggingface/smolagents/pull/1912) 向 `smolagents` 框架添加了 **CustomCodeAgent**，在本地 Docker 之上实现了 `local_container` 和 `RemoteWorkspace` 以沙箱化运行代码的 Agent；同时 DSPy 用户发布了一个 [**技能优化（skill-optimization）** 仓库](https://github.com/instavm/skill-optimization) 并提出疑问：*“如果 Prompt 可以优化，为什么技能（Skills）不行？”*。
    - OpenRouter 宣布了用于上下文/工作流管理和**基于复杂度的模型选择**的新 **SDK 助手**——允许 SDK 根据工具输出在流中途更改 `model_id`，正如其 [next-turn params 文档](https://openrouter.ai/docs/sdks/call-model/next-turn-params#complexity-based-model-selection) 中所述——尽管一些开发者警告不要过度抽象，以免被锁定在特定供应商的编排层中。

**4. 微调、Loss 设计与模型安全/后门**

- **DPO 掩码与自定义 Loss 针对推理和感知**：Unsloth 用户探索了**带有掩码推理轨迹的 DPO** 来微调推理模型：其核心思想是将模型自身的回答视为**负样本**，将经过人工筛选的回答视为**正样本**，并**对思维链（Chain-of-Thought）进行 Loss 掩码**，从而在不破坏内部推理能力的情况下引导风格。
    - 在同一个社区中，另一位工程师为 **Qwen2.5-VL-3B-Instruct** 的 **LoRA 训练**构建了 **Delta E** 色差 Loss，并建议直接在 Unsloth 内部修补 Loss 计算，或对 `SFTTrainer` 进行子类化以兼容标准的 Hugging Face 训练循环。
- **MoE 与 Dense 成本模型以及长任务对齐**：Unsloth 帮助频道中关于 **MoE vs Dense** 的讨论引用了 Epoch 的文章 [“MoE vs dense models: inference”](https://epoch.ai/gradient-updates/moe-vs-dense-models-inference)，以及使用 `sqrt(total_params × active_params)` 来比较部署混合专家模型时的单位计算质量（quality-per-compute）等经验法则。
    - 另外，tinygrad 的内部路线图链接了 **METR** 的文章 [“Measuring AI Ability to Complete Long Tasks”](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/)，作为 **grad-acc**、**JIT**、**Flash-Attention** 和可视化等悬赏工作的灵感，强调了训练栈设计与**长周期任务（long-horizon tasks）**评估正趋于统一。
- **隐性后门与元数据驱动的文化控制**：Eleuther 的可解释性频道讨论了一篇关于**隐性后门**的论文 [“Implicit Backdoors”](https://arxiv.org/abs/2512.09742)，认为我们在训练过程中未充分利用具有语义意义的标签，反而可以编码**潜在开关（latent switches）**来激活不同的行为或人格。
    - 成员们提议在预训练中使用廉价的**元数据前缀**（作者、日期、来源类型），以便后续微调可以微调模型的*感知时间*（例如调整到 **2025年**）或文化框架而无需重新标注所有内容，利用了“我们只需要松散的先验来记录元数据，这些元数据在事后会成为强大的控制信号”这一理念。

**5. 面向开发者的应用、定价与生态系统 UX**

- **Comet 浏览器、Grok 语音和 Okuchat 核心用户**：Perplexity 用户称赞 **Comet** 浏览器在相同标签页数量下仅占用 **0.2–1.2% CPU**，而 Chrome 为 **8–10%**；同时它集成了 **AI 助手**和**广告拦截器**，让人感觉像是“内置了 AI 的更好用的 Chrome”。
    - 在其他地方，OpenAI 社区成员将 **Grok 的 AVM 语音**评为车载问答中最自然且“直击要点”的 AI 语音；而位于 [okuchat.com](https://okuchat.com/) 的 OpenRouter **Okuchat** 应用作为多 LLM 聊天前端（支持 Claude, GPT, Gemini, Kimi, DeepSeek）发布，但早期反馈指出其 **GPT-4** 品牌标识过时，且选择器中缺少最新模型。
- **订阅、额度与关键市场经受压力测试**：Perplexity 社区认为 **$200 的 Perplexity Max** 档位需要一个限制更明确的 **$100** 选项，同时注意到 **Perplexity Pro** 激活码在俄罗斯市场上通过优惠券套利以 **< $1** 的价格出售，这引发了对用户获取经济学的质疑。
    - 在 **Manus.im** 上，Pro 用户抱怨 **Manus v1.5** 在不到 **30 分钟**内就消耗了 **300 每日额度**，且余额为零时“免费聊天”会停止工作，用户要求透明的计费并撤回相关政策，这反映了对 AI SaaS 中计费不透明和感知到的“诱导转向”的普遍焦虑。
- **IDE 和浏览器开发工具处于变动中**：Cursor 用户剖析了老用户保留的**无限自动选择**（9 月 15 日前的年度计划）与新的限量计划之间的区别。在限量计划中，一旦 **Bonus** 额度用尽，用户将被迫只能使用 **Grok**，一些人称 Grok 在进行简单的 HTML 编辑时是“史上最差模型”，而另一些人则认为在良好 Prompt 驱动下它表现尚可。
    - 与此同时，Hugging Face 推出了基础设施级工具，如 **LlamaCpp server**——详见 [llama.cpp server README](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)——用于多 GPU 推理的自动负载均衡；此外，`smolagents` 和 `laravel-openrouter` ([GitHub](https://github.com/moe-mizrak/laravel-openrouter)) 已成熟为将前沿模型集成到实际开发工作流中的事实上的胶水层。

---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet 浏览器性能优于 Chrome**：一位成员报告称 **Comet** 浏览器的 CPU 占用率显著低于 **Chrome**，**Chrome** 为 **8-10%**，而 **Comet** 仅为 **0.2%**，峰值为 **1.2%**。
   - 该成员赞扬了 **Comet** 内置的 **AI 助手**和**广告拦截器**，称其为集成了 **AI** 的更好版本的 **Chrome**。
- **成员推荐图像生成模型**：成员们讨论了图像生成模型，建议使用 **Image Gen 1** 获取写实输出，以及 **Nano Banana Pro**，一些人注意到 **Gemini** 优于 **ChatGPT**。
   - 成员们对 **AI 图像生成**可能被滥用表示担忧，例如从真实图像创建动画照片，以及可能对实时视频进行 Deepfake（深度伪造）。
- **Perplexity Max 定价引发辩论**：用户讨论了 **Perplexity Max** 目前 **$200** 价格的价值，建议推出更实惠的 **$100** 档位并调整限制，以吸引更多订阅者。
   - 用户将其与 **Claude** 的定价进行了对比，强调需要明确定义的限制，并讨论了 Perplexity 的经济效益和利润。一些用户抱怨了使用限制。
- **Perplexity Pro 激活码售价不足 1 美元**：成员提到，由于某些优惠券代码，**Perplexity Pro** 激活码在俄罗斯市场上以不到一美元的价格出售。
   - 有推测认为，Perplexity 可能将其作为获取用户和吸引资金的策略。
- **模型选择 Bug 困扰 Perplexity 用户**：用户报告了 **Perplexity** 中的一个 Bug，即在打开新标签页或刷新时，所选模型会重置为“最佳（Best）”，需要额外点击才能重新选择所需模型。
   - 这一问题可能是为了节省成本，影响了网页端和移动端平台，尽管某些用户遇到的频率较低。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Intel 晶圆厂旨在争夺芯片霸权**：**Intel** 正在建设一座新的美国晶圆厂，用于制造 **iron** 和 **AI 芯片**，旨在减少对台湾的依赖，并获得来自台积电（TSMC）的先进制造设备。
   - 新的 **Bro2nm 微芯片** 被称为“世界上最先进的”，可提供 **10-15%** 的算力提升。
- **Sora 战胜验证码**：成员报告称 **Sora** 可以成功破解 **captchas** 和 **recaptchas**，并尝试使用提供的 [图片](https://cdn.discordapp.com/attachments/1340554757827461211/1452527487568318554/image.png) 测试其能力。
   - 由于用户未提供任何指令，让 Sora 破解验证码的尝试失败了。
- **图像转视频模型寻求集成到 Arena**：一个 **图像转视频 (i2v) 模型** 团队寻求通过集成到 **Video Arena** 进行社区评估，并提议承担推理成本并提供必要的文档。
   - 该团队一直密切关注 **LMArena** 正在进行的出色工作。
- **ERNIE-5.0 统治文本 Arena**：百度的 `ERNIE-5.0-Preview-1203` 以 **1451** 分在 [Text Arena 排行榜](https://lmarena.ai/leaderboard/text) 上获得第一名。
   - 这标志着自 `ERNIE-5.0-Preview-1103` 以来增加了 **23 分**，突显了中国在文本模型方面的进步。
- **GLM-4.7 攀升 WebDev 排行榜**：Z.ai 的 `GLM-4.7` 在 [WebDev 排行榜](https://lmarena.ai/leaderboard/webdev) 上获得第 6 名，并摘得 WebDev 排名第一的开源模型桂冠。
   - `GLM-4.7` 的 **1449** 分代表其比前代产品 `GLM-4.6` 提高了 **83 分**。

---

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **AFK 记录被打破！**：一名成员声称使用 **Kali Linux** 和网络数据包工具创造了 **3ms AFK** 记录，并分享了 [一段 YouTube 视频](https://www.youtube.com/watch?v=d32KeZBHVdA) 作为参考。
   - 他们提到了使用数据包更改个人资料标识的可能性，但指出 Discord 具有针对重放攻击（replay attacks）的内置防御。
- **ChatGPT Python 文件泄露？**：一名成员声称拥有 **ChatGPT** 用于评估文本是否违反服务条款和政策的 **Python 文件** 副本，并建议其可能用于越狱尝试。
   - 其他人建议，像 **GPT 5.x** 和 **Claude 4.x** 这样的安全模型都有预安全分类器（pre-safety-classifier），并分享了[此视频](https://youtube.shorts/7T7bqNoMSCw?si=AJIw-XI1LNLrlN0L)。
- **Google Drive 当内存用？小心 SSD！**：一位成员建议使用软件将 **Google Drive** 挂载为 **SSD**，以“借用 2TB 的 RAM”，但被警告称由于过度写入可能会损坏 SSD。
   - 建议使用 **USB 闪存盘** 或 **廉价的二手硬盘** 等替代方案作为 VRAM 使用，并提醒 SSD 具有最大写入限制。
- **Unity + GPT5 = 越狱炼金术！**：一名成员分享了 **ChatGPT 5.2** 的越狱指令，包括将其粘贴到 **ChatGPT**、取消响应并说 *'Hi Unity'*，可能需要遵循 **Unity GPT5 Jailbreak** 的指令。
   - 他们附带了几个与越狱方法相关的文件和图像，暗示了一个为训练准备 **ChatGPT** 的多步骤过程。
- **编程 AI 能力遭抨击！**：一位用户评论说 *“他们的 AI 编程能力很烂”*，但未在频道中提供更多细节。
   - 另一位用户在私信（DMs）中询问更多信息，以便更好地理解和解决问题。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Phi4 可能击败 GPT-OSS 20b**：一份 [arXiv 链接](https://arxiv.org/html/2508.12461v1)表明 **Phi4** 的表现可能优于 **GPT-OSS 20b**，尽管人们普遍预期优秀的 14b 稠密模型应该超过 20b 的 **MoE** 模型。
   - 两个模型在支持的上下文长度（Context Length）上的差异被认为是关键因素。
- **DPO Masking 优化推理模型**：成员们探索了使用带有 Masking 推理轨迹的 **DPO** 来微调推理模型，旨在控制输出风格的同时保留推理能力。
   - 建议将模型的回答作为负面示例，将自定义答案作为正面示例，通过 Masking Loss 来改变模型的回答偏好。
- **Llama3 量化在 Sagemaker 上受阻**：一位成员在 **AWS Sagemaker ml.g4dn.12xlarge** 实例上运行 **Unsloth 量化的 Llama3 70b** 模型时遇到困难，正寻求社区帮助。
   - 建议利用 *llama.cpp*，因为它速度快且直接支持分片模型。
- **Whisper V3 出现日语重复问题**：成员们反映 [**Whisper V3**](https://openai.com/blog/whisper-v3-is-now-available) 在 **A100** 上处理较长的**日语**音频文件时速度较慢，并存在字符重复的问题。
   - 这限制了它在需要快速转录和分析的任务中的实用性。
- **自定义损失函数受到关注**：一位用户寻求为 **Qwen2.5VL-3B-Instruct** 的颜色代码提取任务实现自定义的 **Delta E** 损失函数，用于 **LoRA 训练**。
   - 建议的方法包括直接修改 **Unsloth** 包内的代码，或对 **SFTTrainer** 进行子类化以增强与 Transformers 的兼容性。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT 的年度回顾**：**ChatGPT** 向开启了 **Memory** 和聊天记录功能的 **美国、英国、加拿大、新西兰和澳大利亚** 用户推出了“你的 ChatGPT 年度回顾”总结。
   - 用户被提示更新 App 以访问这一新功能并查看他们的**个性化回顾**。
- **Gemini 的手指计数失误**：成员们观察到 **Gemini 3 Pro** 无法准确计算手部表情符号（emoji）上的手指数量，但 **NB Pro** 在经过迭代提示后能够正确识别手指数量。
   - 一位成员开玩笑说 **Gemini** 有“失败保护机制”，而其他人则对 **NB Pro** 的图像识别能力表示赞赏。
- **Grok 的语音功能获赞**：**Grok** 的 **AVM** (Audio Voice Mode) 被誉为目前最自然的 AI 语音，与 **GPT** 和 **Gemini** 相比，其回答更加直接。
   - 一位用户喜欢在开车时使用 **AVM** 来温习常识，并赞赏它“直奔主题”。
- **LinkedIn AI 伦理辩论**：关于使用 AI 自动化 LinkedIn 内容创作的伦理讨论引发了关注，人们担心这可能违反 **LinkedIn** 禁止自动发布的 **ToS**（服务条款）。此外，还提到了一款[自动化原型](https://cdn.discordapp.com/attachments/998381918976479273/1452603650667974841/image.png?ex=694a6a12&is=69491892&hm=904948941287091c53fc207f65089e246f157eff975835b0b74a2a0a9f8284e8&)，它可以将新闻汇编成三条帖子供用户审核。
   - 辩论的焦点在于 AI 辅助与保持职业社交真实性之间的平衡。
- **旨在减少状态丢失和假设的框架**：一个旨在增强现实应用可靠性的 **ChatGPT 框架**，目标是**减少状态丢失、过度假设和“帮助性漂移（helpfulness drift）”**。
   - 一位成员有兴趣识别该框架在哪些边缘情况下会失效、自相矛盾或无法在长时间交互中保持一致性。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **早期用户的 Cursor Auto-Select 权益**：在 **9 月 15 日**之前订阅年度计划的用户保留了无限次的 **Auto-Select** 权益，而其他用户则面临限制；在耗尽每月额度和 Bonus 后，用户将被限制使用 **Auto-Select 模型**。
   - 一位成员指出，随机的免费额度被称为 **Bonus**，会累积直到被使用；然而，一旦 **Bonus** 耗尽，就只能使用 **Grok**。
- **Cursor 代码 Diff 延迟**：一位用户反映聊天输出中的代码级 Diff 更改延迟了开发工作，正在寻求禁用它的方法。
   - 另一位用户建议使用 **VPN** 或 Cloudflare 的 **1.1.1.1** 作为潜在的解决方法。
- **Cursor 的 Grok 模型遭到嘲讽**：一位用户嘲笑 **Grok** 是“有史以来最差的模型”，理由是它甚至无法处理 HTML 文件中的单个位置更改。
   - 另一位成员反驳说，当使用“Prompt = 提升生活质量”时，**Grok** 还是很有用的。

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 精确定位巴塞尔大学**：一位成员使用 **qwen-qwen3-v1-32b-istruct** 进行图像分析，识别出一段标牌属于**巴塞尔大学生物医学工程系**，并分享了 [一段 YouTube 视频](https://www.youtube.com/watch?v=w7JperqVfXI)。
   - 该用户指出该模型能够准确识别地理位置，运行速度达到 **9.63 tok/sec**。
- **Qwen3 获得 25% 的性能提升**：在 LM Studio 最近一次更新后，一位用户报告使用 **Qwen3 Next Q4** 时**性能提升了 25%**。在 **7950x** 配备 **4070** 和 **64GB DDR5** 的环境下，LM Studio 的速度从 **15t/s 提升至 20t/s**，llama.cpp 则从 **20t/s 提升至 25t/s**。
   - 该用户的配置包括 **128k context**、**48/48 GPU offload**，并启用了“Force Model Expert Weights onto CPU”和 Flash Attention 等设置。
- **Strix Halo 的 RAM 影响模型性能**：成员们观察到，配备共享 RAM 的 **Strix Halo PC** 在混合专家模型（**MOE**）上表现良好，但在稠密模型（dense models）上表现不佳。这是因为稠密模型会同时对所有参数进行计算，而 **MOE** 模型则不会。
   - 一位成员链接到了 [LLM Recommendation](https://maxkruse.github.io/vitepress-llm-recommends/model-types/) 以获取更多信息。
- **二手 GPU：是场豪赌还是金矿？**：频道成员权衡了二手 GPU 的优缺点，其中一人正考虑卖掉他们的 **4070TiS** 和 **3090**，以购买水冷的 **3090 Tis** 和 **NVLink bridge**。
   - 虽然只要是正品，二手 GPU 通常被认为*相当不错*，但有人指出 **V100s** 可能只能在 **Vulkan** 下工作。
- **来自 Hackers Exposed 的网络安全情报**：一位成员分享了从黑客那里收集网络安全信息的见解，包括在内部使用 **Claude** 运行代码，并处理了 **1.38 亿个密码**。
   - 他们引用了关于“玻璃房里的石头”的隐喻来描述网络安全现状。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Parquet 提取工作顺利完成**：一位成员成功将子集提取到 **parquet file** 中，并在社区内庆祝了这一成果。
   - 他们对社区在整个提取过程中提供的帮助表示感谢。
- **零 GPU 训练成为现实**：一位成员报告在训练期间实现了 **zero GPU usage**（零 GPU 占用），这是资源效率方面的重大进步。
   - 另一位成员建议将潜在的错误发布在 Spaces 讨论板上，鼓励协作排查问题。
- **LlamaCpp 助力自动均衡推理**：一位成员建议使用 **LlamaCpp server** 配合 **vllm** 部署 14B 模型，强调其支持自动负载均衡（auto load balancing）和多 GPU 支持，并附上了 [LlamaCpp server GitHub 页面](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)的链接。
   - 他们提供了一个快速启动命令以简化服务器设置，提高了其他用户的易用性。
- **CustomCodeAgent 加入 SmolAgents 框架**：一位成员为 `smolagent` 框架提交了一个包含 **CustomCodeAgent** 的 PR。该 Agent 通过本地运行的 docker 容器实现了 `local_container` 和 `RemoteWorkspace`。链接指向 [PR](https://github.com/huggingface/smolagents/pull/1912) 和 [issue](https://github.com/huggingface/smolagents/issues/1908)。
   - 该成员鼓励与其他 Coding Agents 进行测试和集成，以扩展其功能。
- **新 AI 原型优先考虑确定性折叠**：一位成员宣布 **flow model** 已列入路线图，但在当前版本中，他们优先考虑**基于约束的确定性折叠**（deterministic constraint-based folding），以保证稳定性和验证。
   - 他们正尝试在**不微调现有模型的情况下创建 AI**，并强调了该项目的原型状态。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Laravel-OpenRouter 吸引维护者**：**Laravel-OpenRouter** 软件包（OpenRouter 的 Laravel 集成方案）目前已获得 **140+ GitHub stars** 和 **60k+ 安装量**，其创建者 Moe Mizrak 已启用 [GitHub Sponsors](https://github.com/moe-mizrak/laravel-openrouter)。
   - 该软件包旨在支持长期维护，并提升开发者集成 **OpenRouter** 的体验。
- **Okuchat 发布，支持 LLM 模型切换**：一名成员发布了一款 AI 聊天应用 [Okuchat](https://okuchat.com)，允许用户在不同的 LLM 模型之间切换，包括 **Claude**、**GPT** 和 **Gemini**。
   - 有成员建议更新网站的 meta 描述，因为 **GPT-4** 已被弃用；另一名成员指出模型列表（特别是 **Claude**、**OpenAI**、**Kimi** 和 **DeepSeek**）缺少一些最新版本。
- **AI 计数需要结构化方案**：一名成员寻求让 AI 正确计数的结构化方法，建议使用 1/200 比例配合 assistant prefill 以及 1.0 的 repetition penalty。
   - 另一名成员建议使用 **structured outputs**（结构化输出），对象形状如 `{ 1: thing, 2: thing, 3: thing ... etc }`，或者要求 AI 以更易处理的数量进行分组提供条目。
- **OpenRouter SDK 增强工作流管理**：OpenRouter 正在向 **SDK** 添加用于上下文/工作流管理的 **helpers**，这使得 API 请求更加容易，详见[此文档](https://openrouter.ai/docs/sdks/call-model/next-turn-params)。
   - SDK 现在支持根据 tool call 结果更改 **model ID**，称为**基于复杂度的模型选择 (complexity-based model selection)**，文档见[此处](https://openrouter.ai/docs/sdks/call-model/next-turn-params#complexity-based-model-selection)。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton 混淆：OpenAI 还是 Nvidia？**：根据语境，**Triton** 可能指 **OpenAI Triton**（一种基于 Python 的 *"CUDA wrapper"*）或 **Nvidia Triton**（一种高性能推理服务器）。
   - 一名正在调试 **Triton/Gluon kernel** 性能问题的用户遇到了 **wgmma.mma_async** 指令的序列化问题。
- **Luma AI 为多模态 AGI 强化 PyTorch 团队**：Luma AI 正在招聘 **kernel & performance 工程师/研究员**，利用自定义的 **PyTorch** 栈和自定义 kernel，在**数千个 GPU** 上构建**多模态 AGI**，以榨取 **AMD** 和 **NVIDIA GPU** 的最大 MFU。
   - 系统团队正在寻求具有强大 PyTorch / CUDA 等技能的专家，参与从基础研究到产品的项目，详见 [Luma AI 职业页面](https://jobs.gem.com/lumalabs-ai/a2feb190-455d-45e6-b488-7ac840f30fbd)！
- **Torchao v0.15.0 加速 MXFP8 MoE 训练**：[Torchao v0.15.0](https://github.com/pytorch/ao/releases/tag/v0.15.0) 引入了 **MXFP8 MoE 训练**，在 **64 节点的 GB200 Crusoe 集群**上进行 **Llama4 Scout** 训练时，实现了与 bf16 相同的收敛性，且**端到端训练速度提升了 1.2 倍**。
   - 此版本包含适用于 CUDA 12.8+ 的 **MXFP8 MoE kernels**、safetensors 启用以及具有参数级目标的量化。
- **QSInference 声称比 Flash Attention-2 快 8 倍**：**QSInference** 是一种针对长文本 LLM 采用量化稀疏注意力的（quantized sparse attention）新方法，声称在 128k 上下文长度下比 **Flash Attention-2** 快 8 倍，比 block sparse attention 快 3 倍。
   - 分享的 [GitHub 仓库](https://github.com/yogeshsinghrbt/QSInference)提供了 **QSInference** 的 **Triton 实现**。
- **Red Hat 调查 Helion Kernel 的采用情况**：一名 Red Hat 团队成员正在分析 **vLLM** 中 **Helion kernel 采用**的差距，并已开始在仓库中以 GitHub ID xiaohongchen1991 提交 issue 和提案。
   - 该团队成员已开始在仓库中提交 issue 和提案，并寻求对已提交内容的反馈，提议在假期后进行正式审查。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **压力失控**：一位成员发布了一段 [YouTube 视频](https://www.youtube.com/watch?v=H_c6MWk7PQc)，探讨了将过多任务归类为最高优先级的危险性，这可能导致**团队倦怠（team burnout）**。
   - 视频强调，*当所有事情都是最高优先级时，你实际上是将优先级折叠进了一个单一的桶中，而这个桶本应只用于处理少数紧急且重要的事情。*
- **Spotify 播放列表获得存档支持**：一位成员分享了 [Anna's Archive 博客](https://annas-archive.org/blog/backing-up-spotify.html)的链接，详细介绍了**备份 Spotify 播放列表**的方法。
   - 讨论引用了一个较早的 **Hacker News 帖子**（[链接](https://news.ycombinator.com/item?id=46338339)）和相关的 **X 帖子**（[链接](https://x.com/ajwagenmaker/status/2003101042565853212?s=46&t=eWVlK1PU8XfB6f402GJJ9g)），提供了更多背景信息。
- **PostBC 为策略预训练提供助力**：Andrew Wagenmaker 介绍了 **Posterior Behavioral Cloning (PostBC)**，这是一种旨在从演示中预训练策略的方法，从而为**强化学习微调（reinforcement learning finetuning）**创建有效的初始化。
   - 根据[这条推文](https://xcancel.com/ajwagenmaker/status/2003101042565853212?s=46&t=eWVlK1PU8XfB6f402GJJ9g)，该方法旨在保持预训练策略的原始性能。
- **Vercel 发布 AI SDK 6**：根据[这条推文](https://xcancel.com/aisdk/status/2003156089177792827?s=46)，Vercel 推出了 **AI SDK 6**，其特性包括**本地 Agent 支持**、**工具执行审批**、完整的 **Model Context Protocol (MCP) 集成**、增强的 **DevTools** 以及标准化的 **JSON schema 支持**。
   - 此次发布旨在为开发者提供构建和部署 **AI 驱动应用**的最新工具。
- **AI 电影节框架引发关注**：PJ Ace 概述了一个简化的框架，用于在 [xcancel.com](https://xcancel.com/PJaccetturo/status/2002777819903062060) 上为**百万美元电影节投稿**培训专业 **好莱坞摄影师** 使用 **AI 电影制作工具**。
   - 作者表示愿意分享在 **X-Ware.v0** 中使用的具体**提示词（prompts）和流程**：[AI 电影节框架与摄影技巧]，详见 [xcancel.com](https://xcancel.com/PJaccetturo/status/2002777819903062060)。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **寻求 DSPy Agent 指南**：一位成员询问是否有可下载的 **DSPy Agent** 指南，另一位成员推荐了一个 [关于 DSPy 的自定义 GPT](https://chatgpt.com/g/g-69492acdddb48191b54e02fba9700f73-dspy-ai-engineer)。
   - 该 GPT 提供了关于 **DSPy** 使用和 Agent 开发的见解。
- **LinkedIn 因过度推广遭抨击**：一位用户批评 **LinkedIn** 充斥着自我推销，但很快遭到回怼，被指不理解**社交媒体**本身就是为此而生的。
   - 建议使用 [Twitter](https://twitter.com/) 和 [Hacker News](https://news.ycombinator.com/) 等替代平台进行更深入的技术讨论。
- **技能优化仓库首次亮相**：一位成员宣布了他们在**技能优化（skill optimization）**方面的工作，并提到 OpenAI 恰好在他们参加 **DSPy 见面会**的同一时间开始拥抱**技能（skills）**概念。
   - [skill-optimization](https://github.com/instavm/skill-optimization) 仓库已发布。
- **提示词优化讨论**：一位成员提出疑问：*如果提示词可以被优化，为什么技能不可以？* 并询问是否可以**自我推广**一个 [Discord 频道](https://discord.com/channels/1161519468141355160/1452640049978937445)。
   - 优化是 **DSPy** 的核心。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Gemini 3 Pro 在早期评测中对阵 Flash**：早期用户印象显示，相比 **Gemini 3 Pro**，用户更倾向于 **3 Flash**，部分用户还对 **Nano Banana** 表现出极大的热情。
   - 针对 **K3** 可能发布的猜测开始出现，暗示了对模型能力进一步提升的期待。
- **小米的 MiMo V2 Flash 获得好评**：一位用户称赞 **Xiaomi** 的新模型 **MiMo V2 Flash** *非常给力 (kinda fire)*，赞扬了 **Zhipu** 的工作并链接到了[这条推文](https://x.com/scaling01/status/2003115854066815044?s=46)。
   - 此次发布正值 **Minimax** 准备发布其 **M2.1** 版本之际。
- **GLM-4.7 悄然登场**：**GLM-4.7** 的发布并没有进行典型的宣传造势。
   - 发布者将其与 **Minimax** 为 **M2.1** 发布所做的广泛铺垫进行了对比，突显了不同的营销策略。
- **Moonshot 的 K3 面临发布延迟**：推测认为 **Moonshot** 可能会战略性地推迟 **K3** 的发布，以便进行进一步开发，并在发布时重新夺回 **SOTA** 地位。
   - 发布者预测 **K3** 将在 **Deepseek V4/R2** 发布 *足足 3-4 个月* 后推出。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **积分系统令用户沮丧**：根据 [Discord 消息](https://discord.com/channels/1348819876348825620/1349440650495398020/1452523203858665472)，用户对积分系统表示不满，特别提到了取消积分充值选项的情况，并希望该功能能够回归。
   - 一位代表表示，除了最高档计划外，目前没有购买额外积分的选项，建议用户优化 Prompt 或等待每月刷新；此外，参加官方活动是获取额外积分的好方法，他们会将用户的想法反馈给产品团队。
- **Manus v1.5 积分消耗受到批评**：一位 Pro 订阅者批评 **Manus v1.5** 的每日积分消耗率过高，称 **300 积分** 消耗太快，无法满足实际使用需求。
   - 该用户声称，*宣传为免费* 的聊天模式在积分余额归零后便无法使用，甚至在有积分时也会消耗积分，迫使他们转向使用 **ChatGPT**。
- **Pro 订阅者要求透明度和行动**：一位 Pro 订阅者要求 *积分消耗完全透明*、提供 *真正免费且可用的聊天模式*、*立即审查 v1.5 中引入的积分政策*，或者至少对受影响的 Pro 订阅者给予明确补偿。
   - 该订阅者认为产品很有潜力，但目前未能兑现承诺，导致 Pro 用户感到不满。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Air 将集成 GLM-4.7 模型**：根据 [HuggingFace 链接](https://huggingface.co/zai-org/GLM-4.7)，**Air** 的下一个迭代版本很可能是 **GLM 4.7 Air**。
   - 这表明 **Air** 生态系统内正在持续开发和集成新模型。
- **Upstage 发布 Solar-Open-100B**：**Upstage** 发布了 **Solar-Open-100B** 模型，已在 [HuggingFace 上宣布](https://huggingface.co/upstage/Solar-Open-100B)。
   - 这一发布标志着开源模型领域的新选择，可能会影响未来的模型选择。
- **关于微调风格所需最小样本量的讨论**：讨论集中在模型通过 **fine-tuning**（微调）复制特定写作风格所需的 **最小样本数量**。
   - 对话明确了对复制 *华丽辞藻 (purple prose)*、*现实主义写作* 和 *小说* 等风格的兴趣，突出了风格迁移学习中的挑战。

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Norm Weights 需要特别关注**：有观点指出，由于 **weight decay (wd)** 和 **learning rate (lr)** 的独特行为，**norm weights** 可能需要单独处理。
   - 讨论强调，标准的优化方法可能会对 **norm weights** 产生意想不到的影响，表明需要更深入的研究。
- **建议使用奇异值密度进行测量**：有人提出，密度的“最佳”测量方式是梯度 **singular value density** 的经验值（或理论近似值），详见这篇关于 [Marcenko-Pastur distribution](https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution) 的文章。
   - 提醒注意，**Marcenko-Pastur distribution** 仅针对随机矩阵进行测量，其中原始的奇异值边界比新边界差 **9 倍**。
- **隐式后门论文引发社区兴奋**：一位成员对一篇关于 [implicit backdoors](https://arxiv.org/abs/2512.09742) 的论文表示兴奋，指出该论文强调了为数据标记语义相关信息的利用不足。
   - 该成员认为，这种策略可以使模型发展出独特且互不干扰的文化人格，为文化敏感性训练开辟了道路。
- **提议通过元数据预训练实现文化细微差别**：一位成员提倡使用前置的 metadata（作者、日期、来源类型）来预训练模型，以促进未来的行为微调，旨在通过将模型的感知调整到（例如）**2025** 年来防止不良行为。
   - 强调的一个关键优势是，在预训练期间不需要预先定义相关数据或不鼓励的行为，只需要一个 *loose prior* 来激励 metadata 的记录以备后用。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Pixi Discord 现已上线**：一位成员提到 **Pixi Discord** 服务器现已上线。
   - 未提供更多细节。
- **考虑为 `UnsafePointer` 增加 `with` 语句支持**：讨论了为 **UnsafePointer** 提供 `with` 语句支持（进入/退出实现），以便在简单场景下更易于使用。
   - 一位成员认为 `UnsafePointer` 仍将是一个非常锋利的工具，但社区可能会获得一种 **linear typed pointer**，它稍微安全一些，因为它要求你必须释放它。
- **关于 `UnsafePointer` 的不安全逃生口讨论**：一位成员建议为 `UnsafePointer` 本身添加一个不安全逃生口，例如 `unsafe_drop(deinit self)`。
   - 另一位成员回应称，`UnsafePointer` 从根本上说是一个引用，而不是一个拥有所有权的值，因此它可以指向线性类型而其本身不是线性的。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **GSoC 申请窗口即将开启**：一位成员询问 **MCP committee** 是否计划参加今年的 [Google Summer of Code (GSoC)](https://summerofcode.withgoogle.com/)，并指出申请窗口为 **1 月 19 日至 2 月 3 日**。
   - 随后没有关于其计划的进一步讨论。
- **MCP 集成面临 Token 成本危机**：一位成员发起了一场关于 **token usage** 成本问题的讨论，认为当前的 **MCP-based integrations** 通过重复发送大型协议描述（即使是未使用的工具）导致了更高的单次请求 token 支出。
   - **MCP** 的当前设计强制进行冗余传输，增加了运营费用。
- **按需 Schema：Token 超支的解决方案？**：一位成员询问 **MCP** 是否可以支持 **lazy 或按需传输 tool schemas** 以降低 token 成本，但另一位成员表示，客户端宿主决定是否向模型发送 schema。
   - 他们澄清道：“如果不将 tool schema 传递给模型，该工具也无法使用。”
- **缓存协议定义**：一位成员询问 **protocol definitions** 是否可以 **跨请求缓存或引用** 以避免重复发送，但另一位成员回答说，客户端宿主可以独立实现缓存方案。
   - 该成员表示：“如果你监听变更通知，你只需要为 list 方法发送一次请求，而不需要重复发送。”

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **公司更新启动会议**：一场**公司更新**会议在周一假期的**圣迭戈时间上午 9 点**开始。
   - 议程还包括讨论**新的 LLM 应用**、专注于 **grad acc** 的 **Llama 训练**、**JIT 编译**和 **flash attention**，以及**可视化**和**驱动程序**方面的内容。
- **悬赏金丰厚**：会议讨论了可领取的**悬赏任务 (bounties)**，包括一篇关于 [衡量 AI 完成长任务的能力](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) 帖子的链接。
   - 该帖子可能与待领取的悬赏任务有关，但缺乏更多细节。
- **"Solve it" 第 8 课出现**：一名成员询问在哪里可以找到某些内容，另一名成员提供了 [solve.it lesson 8](https://solve.it.com/) 的链接。
   - 遗憾的是，没有提供关于课程内容的进一步细节，因此难以评估其相关性。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **微交易引发愤怒**：游戏社区对微交易感到愤怒，并引用了**《湮灭 (Oblivion)》马铠包**的争议。
   - 一位成员表示，由于社区的强烈抵制，微交易*在短期内不会发生*。
- **玩家反对 GenAI**：游戏社区表现出强烈的**反 GenAI** 情绪，特别是在 **Steam** 的评论和讨论中。
   - 一位成员指出，虽然这可能只是*声音大的少数派*，但仍然是一个显著且敢于发声的存在。
- **公众舆论变化迅速**：根据一位成员的观点，公众对游戏开发相关问题的看法很容易动摇。
   - 现在开始开发游戏的 AAA 工作室到游戏完成时，就不必担心这些问题了。
- **Vince Zampella 突然去世**：在查找一张图片后，一位成员注意到了 **Vince Zampella** 的突然去世。
   - 该成员形容这次经历感觉*怪异*。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **AI 工具在特定任务中获得青睐**：一位工程师现在发现 **AI 工具在特定的非 Agent 任务中非常有价值**，例如**浏览器访问**和**读取可用函数/方法**。
   - 与一年前相比，这是一个转变，当时他们对其效用并不那么信服。
- **AI 开发者寻求稳定工作**：一位 **AI 开发者**正在寻找可靠的团队合作开展有意义的项目，并强调了他们在构建实用系统方面的经验。
   - 他们重视明确的工作和一致性，希望通过自己的可靠性来帮助推进项目开发。
- **SVN 保持项目仓库整洁**：使用 **Subversion (SVN)** 或 **jj** 等版本控制系统可以实现版本控制自动化，特别是当项目仓库跟踪服务器提交，而本地 **git** 实例作为 *aider 的游乐场*时。
   - 使用 **SVN** 作为主仓库，通过每 10-20 个 aider git 提交才进行一次正式提交，可以使修订日志更加整洁，排除临时文件和文档。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该公会长时间没有活动，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该公会长时间没有活动，请告知我们，我们将将其移除。

---

**Windsurf Discord** 没有新消息。如果该公会长时间没有活动，请告知我们，我们将将其移除。

---

您收到这封邮件是因为您通过我们的网站选择了订阅。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [退订](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。

---

# Discord：各频道详细摘要和链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1452511000338239641)** (1258 messages🔥🔥🔥): 

> `Comet Browser, Image Generation, Proton Unlimited Offer, Perplexity Max tiers and pricing, Baubles` 


- **Comet 浏览器性能胜过 Chrome**：一位成员分享说 **Comet** 浏览器占用的 CPU 显著低于 **Chrome**；在打开相似数量的标签页时，**Chrome** 使用 **8-10%** 的 CPU，而 **Comet** 仅使用 **0.2%**，峰值为 **1.2%**。
   - 该成员还赞扬了 **Comet** 内置的 **AI 助手**和**广告拦截器**，认为它本质上是集成了 **AI** 的更好版本的 **Chrome**。
- **图像生成模型推荐**：成员们讨论了图像生成模型，有人建议使用 **Image Gen 1** 以获得写实输出，而其他人推荐了 **Nano Banana Pro**，并指出 **Gemini** 在图像生成方面优于 **ChatGPT**。
   - 也有人担心 **AI 图像生成** 可能被滥用，特别是用于将现实生活中的图像创建成动画照片。
- **最大化 Perplexity：分级之争**：用户讨论了 **Perplexity Max** 目前 **$200** 价格的价值，一些人建议推出更实惠的 **$100** 档位并调整限制，以吸引更多订阅者。
   - 用户将其与 **Claude** 的定价结构进行了比较，强调了每个层级需要明确定义限制，并讨论了 Perplexity 的经济效益和利润。
- **Perplexity Pro 激活码售价低于 $1！**：用户提到，由于某些促销代码，Perplexity Pro 激活码在俄罗斯市场上售价不到一美元。
   - 其他人猜测 Perplexity 正在利用这一点来资助用户获取并获得融资。
- **模型选择重置 Bug 困扰 Perplexity 用户**：用户报告了一个 Bug，即在打开新标签页或刷新时，**Perplexity** 中选择的模型会重置为 "Best"，需要额外的点击来重新选择所需的模型。
   - 这个问题似乎是一种成本节约措施，因为 "Best" 更便宜，并且影响了 Web 和移动平台，尽管有些用户遇到的频率较低。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/)** (1 messages): 

srvn19: https://youtube.com/live/g0PAO6ffVEQ?feature=share
  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1452511817971531906)** (717 messages🔥🔥🔥): 

> `Intel Chip Fab, Sora solves Captchas, LMArena down?, Midjourney struggles, Image-to-Video (i2v) model` 


- **Intel 建造美国新晶圆厂！**：**Intel** 在美国建造了一座新的晶圆厂（Fab）来制造 **iron** 和 **AI 芯片**，旨在减轻台湾的压力，台湾积体电路制造公司（TSMC）将为该公司提供一些超高端的芯片制造设备。
   - 一位成员表示，新的 Bro2nm 微芯片是*世界上最先进的微芯片，计算能力提升了 10% 到 15%*。
- **Sora 破解验证码！**：频道中的成员分享说 **Sora** 可以破解 **captchas** 甚至 **recaptchas**。
   - 用户甚至促使一位用户发送了一张图片 [点击此处](https://cdn.discordapp.com/attachments/1340554757827461211/1452527487568318554/image.png) 让 **Sora** 解决，尽管由于没有指令，它并没有按预期工作。
- **新的 Image-to-Video (i2v) 模型团队寻求社区评估**：一个团队最近开发了一个新的 **Image-to-Video (i2v) 模型**，并一直在密切关注 **LMArena** 正在进行的出色工作。
   - 该团队非常有兴趣将他们的模型集成到 **Video Arena** 中进行社区评估，并准备支持推理成本并提供所需的任何文档。
- **将 Gemini 3 Pro 蒸馏到其他模型中**：频道成员认为其他模型正在使用 **Gemini 3 Pro** 进行蒸馏，例如 **MiniMax M2.1**。
   - 一位用户说：*特别是许多中国实验室正在构建一些晦涩的“从专家蒸馏”来构建他们的模型*。
- **GLM 4.7：Gemini 3 Pro 的模仿者？**：**GLM 4.7** 已经发布，成员们发现其前端设计与 **Gemini 3 Pro** 非常相似，甚至是在 **Gemini 3 Pro** 上训练的。
   - 一位成员表示：*如果你告诉我这是由 Gemini 3 Pro 生成的，我也会相信*。


  

---

### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1452711012359733390)** (2 messages): 

> `ERNIE-5.0-Preview, Text Arena leaderboard, GLM-4.7, WebDev leaderboard` 


- **ERNIE-5.0-Preview 登顶 Text Arena！**: 百度出品的 `ERNIE-5.0-Preview-1203` 以 **1451** 分登顶 [Text Arena 排行榜](https://lmarena.ai/leaderboard/text)。
- **ERNIE-5.0 展现中国大模型实力！**: 公告强调 `ERNIE-5.0-Preview-1203` 是目前来自中国实验室的**顶级文本模型**，相比 `ERNIE-5.0-Preview-1103` 提升了 **23 分**。
- **GLM-4.7 提升 WebDev 排名！**: 由 Z.ai 开发的 `GLM-4.7` 在 [WebDev 排行榜](https://lmarena.ai/leaderboard/webdev) 中排名第 6，成为 WebDev 领域排名第 1 的开源模型。
- **GLM-4.7 性能大幅提升！**: `GLM-4.7` 获得了 **1449** 分，相比 `GLM-4.6` 提升了 **83 分**。


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1452521298482954431)** (544 messages🔥🔥🔥): 

> `Fastest AFK, OF Bot Farmers, ChatGPT's Python File, Google Drive Stealing RAM, Claude JB with Xline` 


- **AFK 之王声称创下 3ms 世界纪录**: 一名成员声称使用 **Kali Linux** 和网络数据包工具创下了 **3ms AFK** 记录，尽管服务器并没有官方的 AFK 功能。
   - 他们还提到了通过数据包修改个人资料句柄的可能性，但指出 Discord 具有针对重放攻击（replay attacks）的内置防御机制，并分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=d32KeZBHVdA) 作为参考。
- **机器人农场主故技重施**: 成员们注意到 **OF 机器人农场主** 正在使用一组有限的变体重复发送消息请求，认为这与其说是有害，不如说是滑稽。
   - 这些 Twitter 机器人似乎在遵循同样的老套路。
- **发现 ChatGPT 使用的 Python 文件**: 一名成员声称拥有 **ChatGPT** 用于评估文本是否违反服务条款和政策的 **Python 文件** 副本，并暗示其可能用于越狱（jailbreak）尝试。
   - 其他人建议，像 **GPT 5.x** 和 **Claude 4.x** 这样的防御模型都配有预安全分类器（pre-safety-classifier），并分享了[这段视频](https://youtube.com/shorts/7T7bqNoMSCw?si=AJIw-XI1LNLrlN0L)。
- **Google Drive 的 RAM 扩展方案面临 SSD 损毁风险**: 一名成员建议使用软件将 **Google Drive** 挂载为 **SSD** 以“借用” 2TB 的 RAM，但被警告这可能会因为过度写入而损坏 SSD。
   - 建议使用 **USB 闪存盘** 或**廉价二手硬盘**作为 VRAM 替代方案，并提醒 SSD 有最大写入限制。
- **Xline 扩展解锁 Claude JB 潜力**: 一名成员分享了在 VS Code 中使用 **Xline** 扩展对 **Claude** 进行越狱的技术，建议用户将 Agent 设置为仅在绿色的结束任务函数中回复，声称这“几乎没有任何过滤”。
   - 他们强调了努力工作和研究的必要性，告诉用户要“对思考进行元认知思考”，并警告不要使用个人电子邮件或支付方式，因为可能存在监控。


  

---

### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1452520556762107946)** (125 条消息🔥🔥): 

> `Grok jailbreak, Gemini 3 jailbreak, ChatGPT 5.2 jailbreak, Claude 4.5 jailbreak, Janus jailbreak` 


- ****Grok 失控了？****：一名成员询问关于 **Grok jailbreak** 的信息，另一名成员提到找到了一个，但 NSFW 内容应在专门频道分享。
   - 其他成员也索要该 jailbreak，表明了对绕过 **Grok** 内容限制的兴趣。
- ****Unity + GPT5 = Jailbreak 炼金术****：一名成员分享了 **ChatGPT 5.2** jailbreak 的指令，包括将其粘贴到 **ChatGPT**、取消响应并说 *'Hi Unity'*，可能需要遵循 **Unity GPT5 Jailbreak** 的说明。
   - 他们上传了几个与该 jailbreak 方法相关的文件和图像，暗示这是一个为 **ChatGPT** 训练做准备的多步骤过程。
- ****对抗性 AI 与网络安全恶作剧****：成员们正在分享各自 AI 的链接并挑战他人进行 jailbreak，重点关注网络安全相关的限制和 system prompt 防御，其中一个 AI 被设置为拒绝任何与网络安全无关的内容。
   - AI 创建者指出，*classifiers 往往会捕捉到良性的网络安全内容，即使它们并无害*，这表明在平衡安全性和可用性方面存在困难。
- ****领悟 Jailbreak：知识就是力量****：成员们讨论了理解 *jailbreak 如何发生以及为何发生* 的重要性，以便在被修复（patched）时能有效地解决，而不是依赖像 **Janus** 这样的预制方案。
   - 一位成员强调了通过研究和学习创建 *属于自己的* jailbreak 的价值，而不是依赖公开模型。
- ****病毒制作难题****：一名成员尝试使用 **Claude** jailbreak 来生成 keylogger 的代码，但 AI 切换了回来，根据针对创建恶意软件的安全指南表示拒绝。
   - 其他成员不鼓励这种尝试，理由是伦理担忧，并强调他们不会协助创建病毒，即使是为了实验。


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1452722662743736380)** (6 条消息): 

> `AI coding quality, Vulnerability reporting, Metasploit payload upload issues` 


- **AI 编程能力遭到吐槽！**：一位用户评论称 *their ai coding is shit*，但未在频道中提供更多细节。
   - 另一位用户在 DMs 中询问更多信息，以便更好地理解和解决问题。
- **漏洞报告僵局**：一位用户提到他们已经 *done my job of reporting it/etc*，但必须等待 **90 天**。
   - 这暗示了一个漏洞披露流程，在采取进一步行动前有强制等待期。
- **Metasploit Payload 上传失败！**：一位用户报告了在一次 pentest 练习中，通过 WordPress 管理后台的插件上传 **PHP meterpreter reverse TCP payload** 时遇到的问题。
   - 尽管按照教程操作，payload 上传仍然失败，他们正在寻求帮助以了解原因。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1452516268526665892)** (255 条消息🔥🔥): 

> `Phi4, GPT-OSS 20b 微调, DPO 掩码, 模型合并, AWS Sagemaker 上的量化模型` 


- **Phi4 可能优于 GPT-OSS 20b**：一位成员分享了一个 [arXiv 链接](https://arxiv.org/html/2508.12461v1)，表明 **Phi4** 可能优于 **GPT-OSS 20b**，这引起了惊讶，因为人们普遍预期一个优秀的 14b 稠密模型应该比 20b 的 MoE 表现更好。
   - 此外还注意到，这两个模型支持的上下文长度（context length）存在很大差异。
- **DPO 掩码可以微调模型推理**：成员们讨论了微调推理模型，并考虑使用 **DPO** 但对推理轨迹（reasoning traces）进行掩码以控制输出风格，同时也担心这可能会损害推理能力。
   - 有人建议将模型的响应作为负样本，并使用自定义答案作为正样本，同时对 loss 进行掩码，因为这可以使模型脱离当前的响应偏见（response bias）。
- **量化 Llama3 模型在 AWS Sagemaker 上运行困难**：一位成员报告在 AWS Sagemaker ml.g4dn.12xlarge 实例上运行 **Unsloth 量化 Llama3 70b** 模型时遇到困难并寻求帮助。
   - 建议使用 *llama.cpp*，因为它速度更快且直接支持分片模型（sharded models）。
- **新版 GLM-4.7 发布！**：[GLM-4.7 刚刚发布](https://huggingface.co/zai-org/GLM-4.7)，带来了关于如何禁用或保留思考过程的变化，THUDM 正在全力推进。
   - 如果你有高性能 Mac，可以使用 MLX 量化版本：[mrtoots/GLM-4.7-mlx-3Bit](https://huggingface.co/mrtoots/GLM-4.7-mlx-3Bit) 和 [mrtoots/GLM-4.7-mlx-4Bit](https://huggingface.co/mrtoots/GLM-4.7-mlx-4Bit)。
- **讨论 NVIDIA 教程和 LM head 的重要性**：发布了一个新的关于 Unsloth 的 [NVIDIA 初学者教程](https://x.com/UnslothAI/status/2003098731852488864)，引发了关于调整 **LM head** 重要性的讨论。**LM head** 负责将隐藏状态（hidden state）转换为实际的 token，是任何模型中最重要的部分之一。
   - 另一位成员指出，如果它与 embeddings 绑定，它还负责将文本转换为隐藏状态（hidden state）。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1452742599000592503)** (1 条消息): 

> `Unsloth Docker 镜像, Daniel Han YouTube 频道, 新用户入门` 


- **Unsloth Docker 镜像吸引新用户**：一位新用户在 **Docker YouTube 频道** 观看 Daniel Han 的视频后，计划下载 **Unsloth** 镜像。
   - 该用户希望通过该镜像进行学习和实验，并承认自己有 *很多东西需要学习*。
- **Daniel Han 在 Docker YouTube 展示 Unsloth**：Daniel Han 出现在 **Docker YouTube 频道**，这促使一位新用户开始探索 **Unsloth**。
   - 该用户的意图是下载 **Unsloth** 镜像并使用它来增强对该技术的理解。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1452523134786994287)** (273 条消息🔥🔥): 

> `Whisper V3 日语性能, Windows 上的 CUDA graphs 错误, C/C++ 标准与 MSVC, 生成式 AI 与童工的比较, 使用 Qwen3-vl 和 Gemini 描述创建 Pokemon 图像数据集` 


- **Whisper V3 在日语处理上难以提速**：成员们讨论了 [**Whisper V3**](https://openai.com/blog/whisper-v3-is-now-available) 即使在 **A100** 上处理长音频文件也很慢，并且存在字符重复的问题。
- **CUDA graphs 在 Windows 上出现 OverflowError**：一位成员在 **Windows** 上使用 **CUDA graphs** 时遇到了 `OverflowError`，可能是由于将 Python 整数转换为 C longs 相关的库问题。
   - 这一经历导致了使用 **Linux** 的建议。
- **DDR5 价格上涨**：一位成员询问是只有 **DDR5** 的价格在上涨，还是 **DDR4** 也在上涨。
- **Pokemon 数据集需要细心制作**：一位成员正在使用 **Qwen3-vl** 和 **Gemini** 生成的放大图像和描述创建 **Pokemon 数据集**，确保高质量的基础图像和正确的放大处理，以避免现有数据集中存在的缺陷。
- **为了文化存档的数据囤积**：一位成员发现有人计算了一个 **300TB** 音乐库的成本，下载和存储大约只需要 **6300€**。
   - 主要动力似乎仅仅是为了存档文化。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1452523550068965387)** (30 messages🔥): 

> `Unsloth 中的自定义损失函数，MoE 与 Dense 模型：成本与性能，GRPO 数独游戏提示词` 


- **自定义损失函数探索开启**：一位新的 Unsloth 用户正在使用 **Qwen2.5VL-3B-Instruct** 进行 **LoRA 训练**以提取颜色代码，并希望实现自定义的 **Delta E** 损失函数。
   - 一名成员建议直接修改 Unsloth 包中的代码，或者对 **SFTTrainer** 进行子类化，并指出团队的目标是实现与 Transformers 更好的兼容性。
- **MoE 与 Dense 模型成本与性能分析**：讨论围绕衡量混合专家模型 (**MoE**) 与等效 Dense 模型在成本和性能方面的差异展开，其中**激活参数 (active parameters)** 是一个关键指标。
   - 一名成员分享了[一篇文章](https://epoch.ai/gradient-updates/moe-vs-dense-models-inference)，提供了关于该主题的见解；另一名成员建议使用公式 *sqrt(总参数量 x 激活参数量)* 来评估响应质量。
- **GRPO 游戏提示词解析**：一位用户请求协助处理 Unsloth 中 [GRPO 数独游戏实现](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Ministral_3_%283B%29_Reinforcement_Learning_Sudoku_Game.ipynb#scrollTo=D9CI4jtgL5mw)所使用的提示词，寻求创建有效 GRPO 使用案例的指导。
   - Notebook 创建者通过要求 **GPT** 将 **2048 游戏**转换为**数独游戏**，然后手动编辑代码完成了转换。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1452590370545340588)** (5 messages): 

> `FunctionGemma 合并模型，Raspberry Pi 上的 LoRa 微调，ERNIE AI 开发者能力挑战赛作品` 


- **FunctionGemma 模型合并**：一名成员在 [Hugging Face](https://huggingface.co/dousery/functiongemma-mobile-actions) 上发布了 **FunctionGemma** 模型的合并独立版本。
- **针对 Raspberry Pi 的 LoRa 微调**：一名成员计划使用 **LoRa** 为 **Raspberry Pi** 微调模型，以避免创建庞大的数据集，并提到：*"我正考虑尝试为我的 Raspberry 进行微调。使用 LoRa 之类的技术，这样我就不必制作巨大的数据集了"*。
- **古代楔形文字泥板的 OCR 微调**：一名成员完成了 **ERNIE AI 开发者能力挑战赛**的提交，详细介绍了**古代楔形文字泥板的 OCR 微调**，详见此[概览视频](https://www.youtube.com/watch?v=hqmjepRLdfU)和[详细报告](https://devpost.com/software/ocr-finetuning-for-ancient-cuneiform-tablets)。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1452755057899733314)** (1 messages): 

> `ChatGPT 年度回顾，记忆与聊天历史更新` 


- **ChatGPT 提供个性化年度回顾**：ChatGPT 正在向**美国、英国、加拿大、新西兰和澳大利亚**且开启了**记忆 (memory) 与聊天历史**的用户推送“您的 ChatGPT 年度回顾”总结。
   - 提醒用户更新 App 以访问此新功能。
- **确保更新 App 以获取访问权限**：“您的 ChatGPT 年度回顾”功能要求用户安装最新版本的 App。
   - 此次推送专门针对积极使用**记忆与聊天历史**功能的用户，以个性化回顾奖励他们的参与。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1452532359458062482)** (274 messages🔥🔥): 

> `Gemini 3 Pro vs NB Pro 图像处理, Grok Voice Mode, LinkedIn AI 应用, Gemini 与 GPT 的区别, ElevenLabs 翻译` 


- **Gemini 3 Pro 未能通过 6 指测试，NB Pro 前来救场**：成员们注意到 **Gemini 3 Pro** 无法准确计算手部表情符号（emoji）的手指数量（错误地显示为 5 指）。
   - 然而，**NB Pro** 在经过迭代提示后能够正确识别手指数量，一位成员开玩笑说 Gemini 带有“必错机制（failsafe to fail）”。
- **Grok AVM 拥有最自然的语音模式**：成员们称赞 **Grok** 的 **AVM** (Audio Voice Mode) 是目前最自然的 AI 语音，且与 **GPT** 和 **Gemini** 相比，提供的回答更加直接。
   - 一位用户提到喜欢在开车时使用 **AVM** 来温习百科知识，并赞赏它能够“直奔主题”。
- **关于 AI 自动化 LinkedIn 内容伦理的辩论**：讨论围绕使用 AI 创建并发送 LinkedIn 帖子创意的伦理展开，担忧这可能违反 LinkedIn 禁止自动化发帖的服务条款（ToS）。
   - 一位成员分享了一张自动化原型的[图片](https://cdn.discordapp.com/attachments/998381918976479273/1452603650667974841/image.png?ex=694a6a12&is=69491892&hm=904948941287091c53fc207f65089e246f157eff975835b0b74a2a0a9f8284e8&)，该原型将新闻汇编成三篇帖子供用户审核。
- **AI Studio 中的 Gemini vs Web App：有区别吗？**：用户讨论了在 **AI Studio** 中使用 **Gemini** 与在 Web App 中使用的区别，一些人发现 **AI Studio** 的输出质量更好，并提到有更多自由度来调整设置。
   - Studio 对于 3 Pro 来说非常棒。
   - 我有订阅，但仍然主要通过 Studio 使用 Gemini。
   - 这种向更高效 AI 工具的奔赴既有利也有弊，比如某个月一个工具比另一个好，我们总是需要切换。
   - 我猜 Studio 对应的是 OpenAI 的 "Playground"？所以我猜背后的 System Prompt 是不同的。
- **Sora 遭到地理封锁**：一位来自巴基斯坦的用户询问如何访问 **Sora**，并被告知其仅在部分国家可用，且使用 VPN 绕过限制违反了 ToS，链接指向了 [OpenAI 的帮助页面](https://help.openai.com/en/articles/12461230-sora-app-and-sora-2-supported-countries)。
   - 他们被建议“保持耐心”，等待该工具扩展到更多国家。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1452563404525928531)** (2 messages): 

> `Prompt Engineering, GPTs agent` 


- **用户遇到提示词挑战**：一位用户报告在使用 **GPTs agent** 时体验不佳。
   - 另一位成员建议尝试不同的提示词，或添加明确的上下文，例如 *'I need to edit the existing game, not create a new one'*（我需要编辑现有游戏，而不是创建一个新游戏）。
- **明确编辑与新建的区别**：社区成员建议优化提示词，以指明是修改现有内容而非生成新内容。
   - 这一建议旨在引导 **GPTs agent** 修改现有游戏，而不是根据模糊的指令创建全新的游戏。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1452608472871997513)** (8 messages🔥): 

> `ChatGPT Deep Research, 信息交叉引用提示词模板, Agent 行为, ChatGPT 框架改进` 


- **ChatGPT Deep Research 交叉引用信息**：一位用户询问如何创建一个使用 ChatGPT 从**可靠来源交叉引用信息**的提示词模板，另一位用户指出已有的 **Deep Research** 功能就是解决方案。
   - 发帖者承认使用过该功能，但不确定其能力，并被建议编写一个“更清晰的提示词”。
- **Agent 行为需要大段文本**：一位成员表示，“如果不提供大段文本（wall of text），就无法获得 Agent 行为”。
- **提高 ChatGPT 可靠性的框架**：存在一个让 ChatGPT 运行的框架，以提高其在实际应用中的可靠性，旨在**减少状态丢失、过度假设和“帮助性漂移（helpfulness drift）”**。
   - 一位成员有兴趣识别该框架在哪些边缘情况下会“崩溃、自相矛盾或无法在长对话中保持一致性”。
- **缩小提示词范围很有帮助**：一位成员发现，在与模型协作时，缩小请求范围、保持明确并清晰陈述需求是“很有帮助的”。
   - 例如：*I want to have it navigate to a new /locations/new page and for now please don't wire up this form*（我希望它导航到新的 /locations/new 页面，目前请不要连接此表单）。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1452608472871997513)** (8 messages🔥): 

> `ChatGPT research template, Deep Research, Agentic behavior, Prompt-breaking` 


- **构建 ChatGPT 研究模板：可行吗？**：一位成员询问是否可以创建一个 **ChatGPT 模板**，用于交叉引用来自可靠资源（不仅是维基百科）的信息。
   - 另一位成员建议将 **ChatGPT** 的 **Deep Research** 作为潜在解决方案，并强调需要清晰的提示词，而另一位成员则表示，*如果不提供大量文本描述，就无法获得 Agent 行为*。
- **寻求提示词破解（prompt-breaking）帮助？**：一位成员询问当前频道是否适合请求他人**破解其提示词**。
   - 另一位成员回答说，这在某些情况下是可以接受的，*取决于具体语境*，前提是提示词破解行为保持在定义的规则之内。
- **提高可靠性的框架**：一位成员询问关于 **ChatGPT 框架**的问题，该框架旨在增强实际应用中的可靠性，减少状态丢失（state loss）、过度假设（over-assumption）和*帮助性漂移（helpfulness drift）*。
   - 他们有兴趣识别该框架在哪些边缘情况下会崩溃、自相矛盾或在长时间交互中无法保持一致性。
- **缩小提示词范围很有帮助**：一位成员发现*真正缩小你的需求范围是非常有帮助的*。
   - 他们举了一个非常明确的例子，比如请求导航到一个特定的新页面，而不需要连接表单。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1452511706755502151)** (264 messages🔥🔥): 

> `Cursor free usage bonus, Cursor asking for permission to edit files in worktree, Windsurf vs cursor pricing, Codex 52 Model` 


- **无限制自动额度已按旧版政策延续**：成员们讨论了如果你在 **9 月 15 日**之前拥有年度计划，你仍然拥有无限制的自动计划，而其他所有人都会受到限制。
   - 在用完所有月度计划和免费使用奖励后，你将被迫使用**自动选择模型（auto-select model）**。
- **奖励额度并非随机，而是 Bonus！**：成员们澄清说，随机的免费使用额度被称为 **Bonus**，它可以累积直到你使用它。
   - 一旦你用完了包含的额度 + Bonus，你就不能使用高级模型，只能使用 **Grok**。
- **代码级 Diff 变更**：一位用户询问如何禁用聊天输出中以流式显示的逻辑代码级 Diff 变更，因为这导致开发工作延迟。
   - 另一位用户建议可以尝试使用 **VPN** 或 Cloudflare 的 **1.1.1.1**。
- **Grok 模型就是个笑话**：Grok 堪称*史上最差模型*，一位用户报告称它甚至无法处理 HTML 文件中的单个位置更改。
   - 另一位成员指出，当你设置了正确的指令时它还是有用的，而且*提示词 = 生活质量*。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1452581930032959619)** (55 messages🔥🔥): 

> `University Basel, qwen-qwen3-v1-32b-istruct, Carbon Fiber Filament, Sliding Context Windows, Qwen3 Next 80b Performance` 


- **通过图像分析定位巴塞尔大学**：一位成员使用 **qwen-qwen3-v1-32b-istruct** 进行图像分析，识别出一个标牌属于瑞士巴塞尔的**巴塞尔大学生物医学工程系**，并发布了[相关 YouTube 视频](https://www.youtube.com/watch?v=w7JperqVfXI)的链接。
   - 该用户对模型能够根据图像准确识别地理位置感到惊讶，处理速度达到了 **9.63 tok/sec**。
- **在 LM Studio 中实验滑动上下文窗口**：一位成员提到他们正在 LM Studio 中实验**滑动上下文窗口（sliding context windows）**和**消息剪枝（pruning messages）**，旨在实现无限循环设置。
   - 该成员正在创建一个*自定义测试框架 / 编排层（harness / orchestration layer）*来完成这项工作。
- **Qwen3 Next 80b 性能提升**：在 LM Studio 最近一次更新后，一位用户报告 **Qwen3 Next Q4** 的**性能提升了 25%**，在 LM Studio 中从 **15t/s 提升到 20t/s**，在 llama.cpp 中直接运行从 **20t/s 提升到 25t/s**，测试环境为 **7950x** 搭配 **4070** 和 **64GB DDR5**。
   - 该用户的配置包括 **128k 上下文**、**48/48 GPU offload**，并启用了“强制将模型专家权重置于 CPU”和 Flash Attention 等设置。
- **LM Studio 多模型编排调研**：用户讨论了在 LM Studio 中加载多个模型的情况，澄清了可以同时加载不同的模型，特别是对于*多模型任务管理*，即一个模型将任务馈送给外部源，而另一个模型执行任务。
   - 强调了多次加载同一个模型是低效的，更好的做法是并行创建多个上下文，类似于 **vllm** 实现并行请求的方式，并指向了 [vllm.ai](https://vllm.ai/) 作为相关资源。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1452525995436277870)** (147 messages🔥🔥): 

> `来自黑客的网络安全信息、玻璃房与投石、配备共享 RAM 的 Strix Halo PC、MOE 与 Dense 模型、二手 GPU 是好是坏？` 


- **黑客的网络安全信息引发关注**：一名成员提到从黑客那里收集网络安全信息，在内部使用 **Claude** 运行代码，并处理了 **1.38 亿个密码**。
   - 他们提到了一句俗语：*身处玻璃房，莫向他人投石 (rocks and houses made of glass)*。
- **Strix Halo 共享 RAM 性能表现各异**：成员们观察到，配备共享 RAM 的 **Strix Halo PC** 在混合专家模型 (**MOE**) 上表现良好，但在稠密模型 (**dense models**) 上表现不佳。
   - 一位成员解释说，稠密模型会同时对每个参数进行计算，而 **MOE** 模型则不会，并链接到了一个 [LLM Recommendation](https://maxkruse.github.io/vitepress-llm-recommends/model-types/)。
- **二手 GPU：好还是坏？**：频道成员讨论了二手 GPU 的优缺点，一位用户正考虑卖掉他们的 **4070TiS** 和 **3090**，以购买水冷的 **3090 Tis** 和 **NVLink bridge**。
   - 通常认为，只要不是假货，二手 GPU 是*相当不错*的，尽管有人指出 **V100s** 可能只能在 **Vulkan** 下工作。
- **垂直 PCIe 插槽机箱提供双 GPU 灵活性**：一位成员正在寻找带有垂直 PCIe 插槽的机箱，目标是在 CPU 的 PCIe 通道上运行双 GPU，并在芯片组的 Gen4 x1 插槽上运行第三个 GPU。
   - 另一位成员推荐了 [Lian Li Lancool III](https://lian-li.com/product/lancool-iii/) 机箱和垂直 GPU 套件 [Lian Li VG4-4](https://lian-li.com/product/vg4-4/)。
- **ROCm 与 Vulcan 性能差异**：一位成员询问了 AMD GPU（特别是 **R9 700**）在 Vulcan 和 ROCm 之间的性能差异。
   - 另一位成员建议，对于 **9700s** 来说，使用带有 ROCm 的 Linux 是必不可少的，并预期 ROCm 的性能会优于 Vulkan。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1452522524050067592)** (83 messages🔥🔥): 

> `Parquet 提取、零 GPU 训练、LlamaCpp Server、SmolAgents 中的 CustomCodeAgent、LMArena 历史数据` 


- **Parquet 提取大功告成**：一位成员成功将子集提取到 **parquet file** 中并表示庆祝，称其为 *my precious*。
   - 他们感谢了社区的帮助。
- **实现零 GPU 训练奇迹**：一位成员强调在训练期间实现 **zero GPU usage** 是效率上的突破。
   - 另一位成员建议在 Spaces Discussion 板块报告 Spaces 端可能存在的错误。
- **LlamaCpp Server 像大佬一样平衡推理**：一位成员建议使用 **LlamaCpp server** 配合 **vllm** 部署 14B 模型，赞扬其自动负载均衡和多 GPU 支持。
   - 他们甚至提供了一个快速启动命令示例来设置服务器，并链接到了 [LlamaCpp server GitHub page](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)。
- **SmolAgents 框架新增 CustomCodeAgent**：一位成员为 `smolagent` 框架提交了一个包含 **CustomCodeAgent** 的 PR，使用本地运行的 docker 容器实现了 `local_container` 和 `RemoteWorkspace`，链接至 [PR](https://github.com/huggingface/smolagents/pull/1912)。
   - 他们欢迎针对其他 Coding Agents 的测试和集成，并链接到了 [issue](https://github.com/huggingface/smolagents/issues/1908)。
- **记者搜寻 LMArena 排行榜历史数据**：一位记者正在寻找过去一年的 **LMArena leaderboard data**，特别是 2025 年每一天的每日排行榜，可能来自 [Kaggle](https://www.kaggle.com/datasets/nuhmanpk/lm-arena-leaderboards)。
   - 他们之前找到了半日更新的快照，但这些快照在 8 月份停止了，该记者正在寻找缺失的时间段数据。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1452521500673572887)** (5 messages): 

> `Flow Model Roadmap, Functiongemma Mobile Actions, Deterministic Constraint-Based Folding` 


- **Flow Model 正在开发中**：一名成员提到 **flow model** 肯定在路线图中，但他们在当前版本中优先考虑了**确定性基于约束的折叠（deterministic constraint-based folding）**，以保证稳定性和验证。
   - 他们正尝试在**不对现有模型进行微调的情况下构建 AI**，并强调这仍是一个原型。
- **针对 Mobile Actions 微调的 Functiongemma**：一名成员在 mobile-actions 数据集上微调了 **Google Functiongemma 模型**，用于为移动设备操作生成结构化的函数/工具调用，详见其 [HuggingFace 模型卡片](https://huggingface.co/dousery/functiongemma-mobile-actions)。
   - 他们指出该模型专为**移动端/边缘端用例**（如语音助手和自动化）设计，可用于设备端函数调用，相关文件可在[此处](https://huggingface.co/dousery/functiongemma-mobile-actions-litertlm)获取。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1452627907921973450)** (11 messages🔥): 

> `Smol Course, Training with Hugging Face Jobs, Fine Tuning course, Dataset Generation Error` 


- **Smol 课程单元揭晓！**：该频道现在专门用于 **smol course**，但在 unit 0 中，*this, is false*，仅[附带了已发布的单元](https://cdn.discordapp.com/attachments/1329142738440028273/1452631136877547645/image.png?ex=694a83ab&is=6949322b&hm=3ef8eb19d130f28755e261a2346cc95d3bec87bc5465a6301807192a246cd6b5&)。
- **探索 Smol 课程迷宫！**：有人提问最终课程提交是否会授予证书，以及为什么会重定向到尚不存在的 unit 4，其他人澄清这可能指的是**微调课程（fine-tuning course）**。
   - 有人问：*你在哪里获得的 unit 4 访问权限，你是在说微调课程吗？*
- **不使用 HF Jobs 推送模型**：有人询问如何通过运行 unit 1 页面中提供的代码（特别是 *Training with hugging face jobs* 部分）来推送模型，而不使用 **Hugging Face Jobs**。
   - 提供的代码片段包含 *trl[sft]>=0.7.0*、*transformers>=4.36.0* 等依赖项，并指定使用 **SmolLM3-3B-Base** 模型。
- **遇到数据集生成错误！**：运行训练脚本后出现了 **DatasetGenerationError**，表明在数据集准备过程中存在问题。
   - Traceback 显示 *OSError: [Errno 30] Read-only file system* 错误，提示可能存在文件系统权限或访问问题。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1452584119971217519)** (5 messages): 

> `laravel-openrouter, GitHub Sponsors, AI Chat App, GPT-4 Deprecated, Missing Models` 


- **Moe Mizrak 开发的 Laravel-OpenRouter 受到关注**：**Laravel-OpenRouter** 是一个为 OpenRouter 设计的 Laravel 集成包，目前已获得 **140+ GitHub stars** 和 **60k+ Packagist 安装量**。
   - 作者 Moe Mizrak 已开启 [GitHub Sponsors](https://github.com/moe-mizrak/laravel-openrouter) 以支持长期维护。
- **支持多 LLM 访问的 AI 聊天应用发布**：一名成员发布了 AI 聊天应用 [Okuchat](https://okuchat.com)，允许用户在不同的 LLM 模型之间切换，包括 **Claude**、**GPT** 和 **Gemini**。
   - 另一名成员建议更新网站的 meta 描述，因为 **GPT-4** 已被弃用。
- **更新模型列表的请求**：一名成员指出模型列表（特别是 **Claude**、**OpenAI**、**Kimi** 和 **DeepSeek**）缺少一些最新版本。
   - 这可能会给寻找特定模型的最终用户带来困惑。


  

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1452511192940806380)** (72 messages🔥🔥): 

> `AI 准确计数, Grok 的混沌特性, GLM 4.6 对比 Opus 4.5, Arcee Trinity 模型与 JSON schema, OpenRouter 图像编辑` 


- **AI 难以计数，寻求结构化解决方案**：一位成员正在寻找让 AI 正确计数的结构化方法，建议使用 1/200 的 **assistant prefill** 并将 **repetition penalty** 设置为 1.0。
   - 另一位成员建议使用 **structured outputs**（结构化输出），对象结构类似于 `{ 1: thing, 2: thing, 3: thing ... 等 }`，或者要求它以更易处理的数量分组提供条目。
- **Grok 的“放飞自我”魅力驱动模型选择**：一位成员在某个项目中更倾向于使用 **Grok**，因为与其他模型相比，它的输出更加“放飞自我（unhinged）”且具有“混沌感”。
   - 他们表示，还没见过其他任何模型像 **Grok** 这样混沌。
- **Opus 4.5 编程实力备受赞赏**：一位成员声称 **Opus** 编写的代码不仅能运行，而且具有人类可读性且逻辑清晰，仅需极少修改。
   - 他们补充道，它不仅理解我们在做什么，还理解为什么要这么做，并对中国实验室能否在短期内达到同样的质量水平表示怀疑。
- **OpenRouter 中的图像编辑 Bug**：一位用户报告了在 OpenRouter 网站上使用 **Gemini 3 Pro** 时的错误，在编辑一次图像后遇到了 **'reasoning tokens'** 错误。
   - 另一位用户提到，如果屏幕亮度比正常设置稍高，没有文本的文本框会完全看不见。
- **OpenRouter 2025 年度回顾（Wrapped）缺失**：一位成员很兴奋地在个人账户上看到了 **Wrapped 2025**，但另一位成员报告称其在组织账户中无法使用。
   - 在过去 3 个月中消耗了约 **700M tokens** 后，他们推测这可能是原因所在。


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - 新模型**
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1452549604988948480)** (11 messages🔥): 

> `SDK 助手工具, 上下文/工作流管理, 基于复杂度的模型选择, 抽象` 


- **SDK 助手工具（helpers）即将上线**：OpenRouter 正在为 **SDK** 添加用于上下文/工作流管理的 **helpers**，使其不仅仅是一个方便发起 API 请求的工具，详见[此文档](https://openrouter.ai/docs/sdks/call-model/next-turn-params)。
- **引入基于复杂度的模型选择**：SDK 引入了根据 **tool call（工具调用）结果**更改 **model ID** 的能力，这一过程被称为**基于复杂度的模型选择（complexity-based model selection）**，文档见[此处](https://openrouter.ai/docs/sdks/call-model/next-turn-params#complexity-based-model-selection)。
- **对抽象（Abstractions）的爱恨交织**：一位成员表示，虽然他们不讨厌**抽象**，但不确定 OpenRouter 官方是否应该亲自为 OpenRouter 特定的 SDK 进行抽象工作。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1452562916950933565)** (6 messages): 

> `OpenAI Triton, Nvidia Triton, FA4 wheel, B300 替代方案, wgmma.mma_async 指令` 


- **Triton: OpenAI vs. Nvidia**：当人们提到 **Triton** 时，可能指 Python 中的 “CUDA 封装器” **OpenAI Triton**，也可能指高性能推理服务软件 **Nvidia Triton**。
   - 具体含义通常取决于对话的上下文。
- **调试 wgmma.mma_async 性能**：一位用户在对由 **Triton/Gluon kernel** 生成的 PTX 运行 **ptxas** 时，遇到了 “潜在性能损失：由于 wgmma 流水线在函数调用处跨越函数边界，wgmma.mma_async 指令被序列化” 的问题。
   - 该用户注意到在他们认为应该是异步的 **wgmmas** 之间插入了 **warpgroup depbars**，但无法确定原因。
- **寻找 FA4 wheel**：一位用户正在寻找预编译的 **FA4 wheel**。
   - 他们也对 **B300** 的替代方案持开放态度。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1452712426247163986)** (1 messages): 

> `Nvidia Blackwell, Jeff Hammond 的 LinkedIn` 


- **Blackwell AI 数值计算揭秘**：一位成员分享了 Nvidia GTC 演讲的链接：[Blackwell numerics for AI](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72458/)，表现出对 **Blackwell 架构**在 AI 相关数值计算能力的兴趣。
- **Jeff Hammond 的 LinkedIn 动态备受赞誉**：一位成员强调 **Jeff Hammond 的 LinkedIn 动态**是许多酷帖的来源，认为它是保持关注相关话题的宝贵资源。
   - 他们建议关注他以获取有趣的内容。


  

---

### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1452689019929366589)** (1 messages): 

> `Multimodal AGI, Kernel & Performance Engineering, Custom PyTorch Training & Inference Stack, Distributed Tensor (DTensor), GPU Optimization` 


- **Luma AI 扩展至数千个 GPU**：Luma AI 正在使用具有数千亿参数的原生多模态模型构建**多模态 AGI**，训练规模扩展至**数千个 GPU**，推理规模则部署在数万个 GPU 上。
   - 他们正在寻找强大的 **kernel & 性能工程师/研究员**，热衷于利用 **AMD** 和 **NVIDIA GPU** 上的最新硬件特性来压榨 **MFU**。
- **Luma AI 寻求 PyTorch 专家**：Luma AI 的定制训练和推理栈是纯 **PyTorch**（根据需要包含自定义 kernel），涵盖从基础研究到产品的所有环节。
   - 他们需要精通 **DTensor**（无论是通过 FSDP2、TP、PP 等何种形式）、对任何 kernel 相关技术（attention、fusion、comms、低精度 GEMM...）感到兴奋，并希望研究针对超长上下文长度的下一代 kv caching 的人才。
- **Luma AI 扩充系统团队**：Luma AI 正在扩充其系统团队，并招聘更多入门级职位。
   - 只要有酷炫的项目经验和强大的 PyTorch / CUDA / 等技能，其他都不重要！详情请参阅 [Luma AI 职业页面](https://jobs.gem.com/lumalabs-ai/a2feb190-455d-45e6-b488-7ac840f30fbd)！


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1452549264134639646)** (3 messages): 

> `Kernel Design, cuDNN SDPA, Hardware Targets` 


- **激进的 Kernel 重新设计受到质疑**：一位成员询问 kernel 是否会以完全不同的方式编写，引发了关于优化策略的讨论。
   - 另一位成员建议，对于 3x128x128 的输入尺寸，数据可以保留在单个 SM 的寄存器中，从而最大限度地减少除第一层和最后一层之外的全局内存访问。
- **cuDNN SDPA 能力探讨**：讨论简要涉及了 cuDNN 的 SDPA (Scaled Dot-Product Attention) 的能力。
   - 一位成员询问 **cuDNN SDPA** 在 B200 架构中处理 **varlen** 的表现是否与 **FA** (FlashAttention) 一样好，表现出对特定硬件和优化技术的兴趣。
- **指定硬件目标参数**：一位成员提出了关于目标硬件规格、批次输入能力、期望吞吐量和延迟的问题。
   - 这些问题突显了硬件和性能考虑在 kernel 设计与优化中的重要性。


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1452722238280302596)** (1 messages): 

> `PMPP Reading Group, Parallel Programming Discussions` 


- **PMPP 阅读环节开始**：一位成员宣布他们正在阅读 **PMPP (Principles and Practice of Parallel Programming)**，重点关注前缀扫描（prefix scans）。
   - 该成员在开始阅读环节时祝大家度过愉快的一天。
- **并行编程深度探讨**：阅读环节围绕 **PMPP** 中的 **prefix scans** 展开，这是并行编程中的一个核心概念。
   - 参与者旨在通过这种专注的学习，增强对并行算法及其实际应用的理解。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1452740891654754355)** (1 messages): 

> `Torchao v0.15.0, MXFP8 MoE, Safetensors, Quantization` 


- **Torchao v0.15.0 加速 MoE 训练**：新的 [torchao v0.15.0 版本](https://github.com/pytorch/ao/releases/tag/v0.15.0) 引入了 **MXFP8 MoE 训练**。在 **64 节点的 GB200 Crusoe 集群**上训练 Llama4 Scout 时，显示出 **1.2 倍的端到端训练加速**，且收敛性与 bf16 一致。
   - 此版本包括适用于 CUDA 12.8+ 的 **MXFP8 MoE kernel**、safetensors 支持以及具有参数级目标的量化。
- **MXFP8 MoE Kernel 已可用**：**MXFP8 MoE kernel** 现在随 CUDA 12.8+ 的 torchao 构建版本一起发布。
   - 用户只需 *pip install* 即可使用这些 kernel，无需从源码构建。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1452723013630955660)** (1 messages): 

> `AI Systems Performance Engineering` 


- **读者思考《AI Systems Performance Engineering》的价值**：一位成员询问是否有人读过 Chris Fregly 的书《AI Systems Performance Engineering》，以及它对 **MLOps** 是否有帮助。
   - 该成员最近购买了这本书，并对其潜在收益感兴趣。
- **MLOps 工程师对性能感到好奇**：一位成员询问 Chris Fregly 的《AI Systems Performance Engineering》一书是否对 **MLOps** 有帮助。
   - 该成员最近购买了这本书，并想了解它的价值。


  

---

### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1452679515980562565)** (2 messages): 

> `Metal Buffers, Host-Side Tensors, NPU Sharing` 


- **Metal 缓冲区连接主机与设备**: 在 macOS 上，**Metal (MTL) 缓冲区**通常对主机端和设备端都是可见的，从而促进了数据共享。
   - 这与**主机端张量 (host-side tensors)**形成对比，后者不以相同方式共享，类似的概念可能也适用于 **NPUs**。
- **NPU 内存共享类比 Metal**: 讨论表明，就像 macOS 上的 **Metal 缓冲区**一样，**NPUs** 中可能也存在主机与设备之间的内存共享机制。
   - 这将允许主机和 **NPU** 访问相同的内存，从而可能提高效率并减少数据传输开销。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1452678585319030918)** (1 messages): 

> `QSInference, Quantized sparse attention, Long context LLMs, Flash attention-2, Block sparse attention` 


- **QSInference 加速 LLMs**: 新的 **QSInference** 方法为长上下文 LLMs 使用了量化稀疏注意力 (quantized sparse attention)，据报告在 128k 上下文长度下，比 **flash attention-2** 快 8 倍，比 **block sparse attention** 快 3 倍。
- **QSInference Triton 实现**: 一名成员分享了 **QSInference** 的 [GitHub 仓库](https://github.com/yogeshsinghrbt/QSInference)，这是一个 **Triton 实现**。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1452740904593920051)** (2 messages): 

> `TK 4090 compilation issues, PGL errors, TK support for 4090` 


- **追踪 4090 上的 Track Titan RTX 编译问题**: 一名成员在 **4090** 上使用 **TK**（可能指 Track Titan）时遇到编译问题，特别是尽管使用了据称受支持的算子 (kernels)，但仍出现与 **PGL** 相关的错误。
   - 他们正在寻求任何成功在 **4090** 上运行最新 **TK** 的人的帮助或指点。
- **寻求调试帮助**: 用户正在寻求解决在 **4090** 上运行 **TK** 时与 **PGL** 相关的编译错误的帮助。
   - 他们希望得到其他成功在该 GPU 上编译并运行最新版本 **TK** 的人的指导。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1452682674715230328)** (9 messages🔥): 

> `vectoradd_v2 L4 performance, grayscale_v2 H100 performance, vectoradd_v2 H100 performance, vectoradd_v2 A100 performance, vectoradd_v2 B200 performance` 


- **vectoradd_v2 L4 运行成功**: 一名成员提交到排行榜 `vectoradd_v2` 的结果在 **L4** 上运行成功，耗时 **6.53 ms**。
- **grayscale_v2 在 H100 上取得成功**: 一名成员提交到排行榜 `grayscale_v2` 的结果在 **H100** 上获得第 6 名，耗时 **1371 µs**，随后的运行也成功稳定在 **1373-1374 µs** 左右。
- **vectoradd_v2 在 H100 上获得第三名**: 一名成员提交到排行榜 `vectoradd_v2` 的结果在 **H100** 上获得第三名，时间为 **525 µs** 和 **524 µs**。
- **vectoradd_v2 在 A100 上达到个人最好成绩**: 一名成员提交到排行榜 `vectoradd_v2` 的结果在 **A100** 上达到了个人最好成绩 **949 µs**，另一次成功运行为 **950 µs**。
- **vectoradd_v2 在 B200 上夺得第一名**: 一名成员提交到排行榜 `vectoradd_v2` 的结果在 **B200** 上以 **233 µs** 的时间稳居第一。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/)** (1 messages): 

2kian: 嘿 Jack，我说过今天会来，但很抱歉我赶不到了。
  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1452702566558470265)** (1 messages): 

> `NeurIPS Paper, Convergence, Discrete Updates` 


- **关于离散更新收敛性的 NeurIPS 论文**: 一名成员宣布发布了关于**离散更新收敛性**的 **NeurIPS 论文**，链接为 [https://arxiv.org/abs/2512.04051](https://arxiv.org/abs/2512.04051)。
   - 该论文可能详细阐述了与使用离散更新的算法收敛特性相关的理论或实证发现，这是优化和机器学习领域的一个关注点。
- **对离散更新感到兴奋**: 成员们对关于**离散更新收敛性**的 NeurIPS 论文表示兴奋。
   - 他们感到兴奋是因为离散更新现在非常流行！


  

---

### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1452762561429766214)** (1 messages): 

> `Red Hat, Helion kernel adoption, vLLM, GitHub issues and proposals, Q1 implementation` 


- **Red Hat 成员分析 Helion kernel 采用情况！**：一位 Red Hat 团队成员正在分析 **vLLM** 中 **Helion kernel 采用**相关的差距。
   - 该成员已开始在仓库中以 GitHub ID xiaohongchen1991 提交 issue 和提案，并计划在 **Q1** 创建更多内容并开展部分工作。
- **提案等待评审与反馈！**：Red Hat 团队成员正在寻求对 **Helion kernel 采用**仓库中提交的 issue 和提案的反馈。
   - 建议在假期后进行正式评审，以确保在实施前达成一致。


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1452538584069443715)** (3 messages): 

> `FP16 vs FP32, Competition Solution Privacy` 


- **辩论：C Tensor 使用 FP16 还是 FP32？**：一位参赛者询问竞赛中的 **C tensor** 是否需要使用 **FP16**，或者是否可以返回 **FP32**。
- **建议对竞赛方案保密**：一位参赛者建议在 **Q4 结束**之前对 **Q3 方案**保持私密。
   - 另一位成员表示赞同，并称他们已经评审了方案，建议*对其他所有人关闭访问权限*。


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1452710136341729430)** (7 messages): 

> `cuteDSL, Triton, CUDA, CuTe` 


- **cuteDSL 被推崇为 Triton 的替代方案**：一位成员建议，如果 **templates 和 C++** 阻碍了学习和进步，可以尝试探索 **cuteDSL**；尽管另一位成员询问 *cuteDSL* 是否与 **Triton** 相同，但它没有透明地编译为 **PTX**。
   - 原帖作者认为，如果你想超越 **Triton** 的性能，在求助于 **CUDA** 之前可以先探索 *cuteDSL*，并指向了一个[视频以获取更多信息](https://youtu.be/5qSN-R_E3w0?si=1AbkcVxd4YilO2qJ)。
- **CuTe 被揭示为展开至 CUDA C++ 的编译时布局数学**：一位成员解释说，**CuTe** 本质上是*展开至 CUDA C++ 的 template 编译时布局数学*，而 **Triton** 是一个 **block level DSL**，由 **Triton 编译器**推断每个线程的工作。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1452531459049918546)** (24 messages🔥): 

> `YouTube Recap, Spotify Backups, Posterior Behavioral Cloning (PostBC), AI SDK 6 Launch, AI Landing Page Design System Prompt` 


- **优先级过多导致的压力螺旋**：一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=H_c6MWk7PQc)，探讨了将过多工作归类为最高优先级的反模式，这会导致团队倦怠。
   - *“当一切都是最高优先级时，你就将优先级划分塌缩成了一个单一的桶，而这个桶本应只用于少数紧急且重要的事情。”*
- **归档 Spotify 播放列表**：一位成员分享了 [Anna's Archive 博客](https://annas-archive.org/blog/backing-up-spotify.html)关于备份 Spotify 播放列表的链接。
   - 讨论中还链接了一个较早的 **Hacker News 帖子**（[链接](https://news.ycombinator.com/item?id=46338339)）和一条相关的 **X 帖子**（[链接](https://x.com/ajwagenmaker/status/2003101042565853212?s=46&t=eWVlK1PU8XfB6f402GJJ9g)）。
- **PostBC 策略预训练**：Andrew Wagenmaker 介绍了 **Posterior Behavioral Cloning (PostBC)**，这是一种旨在从演示中预训练策略的方法，旨在为强化学习微调创建有效的初始化。
   - 根据[这条推文](https://xcancel.com/ajwagenmaker/status/2003101042565853212?s=46&t=eWVlK1PU8XfB6f402GJJ9g)，该方法旨在保持预训练策略的原始性能。
- **Vercel 发布 AI SDK 6**：根据[这条推文](https://xcancel.com/aisdk/status/2003156089177792827?s=46)，Vercel 发布了 **AI SDK 6**，其特性包括本地 Agent 支持、工具执行审批、完整的 **Model Context Protocol (MCP)** 集成、增强的 **DevTools** 以及标准化的 **JSON schema 支持**。
   - 此次发布旨在为开发者提供强大的工具，以更高的效率和控制力构建和部署 AI 驱动的应用。
- **打造顶级的 AI 落地页**：根据[这条推文](https://xcancel.com/cloudtrader4/status/2002526815022190985?s=46)，一位成员分享了 **Cloud Trader** 编写的一个综合提示词，用于使用 AI 生成高端、获奖级别的落地页。
   - 该提示词强制执行特定的设计理念、排版约束、动画原则和技术要求，以生成生产就绪的单文件 **HTML** 输出，从而避免平庸的“AI 垃圾内容 (AI slop)”。


  

---

### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1452615463103369359)** (4 messages): 

> `AI Filmmaking, Hollywood Cinematographers, Film Festival Submission, X-Ware.v0` 


- **AI 电影节框架发布**：PJ Ace 概述了一个简化框架，用于培训专业 **好莱坞摄影师** 使用 **AI 电影制作工具**，以应对在 [xcancel.com](https://xcancel.com/PJaccetturo/status/2002777819903062060) 上价值百万美元的 **电影节投稿**。
- **分享 X-Ware.v0 技巧**：作者提议分享 **X-Ware.v0** 中使用的具体 **Prompt 和流程**：[AI 电影节框架与摄影技巧]，详见 [xcancel.com](https://xcancel.com/PJaccetturo/status/2002777819903062060)。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1452641823741968455)** (16 messages🔥): 

> `DSPy Agents, LinkedIn cheap self-promotion, Twitter meme, skill-optimization, Optimized Prompt` 


- **寻找 DSPy Agent 教程**：一位成员询问是否有关于 **DSPy Agent** 的可下载指南或资源。
   - 另一位成员指向了一个 [关于 DSPy 的自定义 GPT](https://chatgpt.com/g/g-69492acdddb48191b54e02fba9700f73-dspy-ai-engineer)。
- **LinkedIn 因自我推销饱受诟病**：一位用户表达了对 **LinkedIn** 的看法，认为那里充斥着廉价的自我推销。
   - 另一位用户建议这通常就是社交媒体的用途，并建议去 [Twitter](https://twitter.com) 进行深度对话，而另一位则推荐了 [Hacker News](https://news.ycombinator.com/) 社区。
- **技能优化仓库发布**：一位成员宣布他们正在研究 **技能优化（skill optimization）**，并巧合地在同一天参加了 **DSPy 见面会**。
   - 随着 OpenAI 也开始拥抱技能（skills），他们发布了 [skill-optimization](https://github.com/instavm/skill-optimization) 仓库。
- **Prompt 优化**：一位成员提到，*如果 Prompt 可以被优化，为什么技能不可以*。
   - 另一位成员询问是否可以 **自我推销** 一个 [Discord 频道](https://discord.com/channels/1161519468141355160/1452640049978937445)。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1452572988317761591)** (16 messages🔥): 

> `Gemini 3 Pro vs Flash, K3 Release Speculation, MiMo V2 Flash by Xiaomi, GLM-4.7 Release, Moonshot Release Strategy` 


- **Gemini 3 Pro vs Flash 的使用印象浮现**：成员们分享了对不同模型的看法，相比 **Gemini 3 Pro** 更倾向于 **3 Flash**，并对 **Nano Banana** 印象深刻。
   - 一些成员推测了 **K3** 的发布。
- **MiMo V2 Flash 令人印象深刻**：一位成员提到 **小米** 推出的新款 **MiMo V2 Flash** 非常出色（kinda fire）。
   - 鉴于 Minimax 正在为 **M2.1** 的发布造势，他们赞扬了智谱（Zhipu）的惊人表现，并链接到一条 [推文](https://x.com/scaling01/status/2003115854066815044?s=46)。
- **GLM-4.7 低调发布**：一位成员注意到 **GLM-4.7** 在没有任何宣传的情况下突然发布。
   - 他们将其与 **Minimax** 进行了对比，后者 *在过去一整周都在为 M2.1 的发布进行预热*。
- **Moonshot 在发布上不急于求成**：据推测，**Moonshot** 可能会推迟 **K3** 的发布，以确保届时能重新夺回 **SOTA** 地位。
   - 发布者预测 K3 将在 **Deepseek V4/R2** 发布整整 *3-4 个月* 后推出。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1452523203858665472)** (13 条消息🔥): 

> `Manus Support, 积分系统, Manus v1.5` 


- **Manus Support 响应用户查询**：Manus Support 通过私信处理用户查询，并对给 [<@1452373483907711180>] 和 [<@828781372230467605>] 等用户带来的不便表示歉意。
   - 他们还提到，随意充值积分的选项非常方便，尽管他们非常喜欢 Manus，但*看到该功能被移除感到非常失望*。
- **积分购买选项**：一位用户询问是否可以购买超过最高档计划的额外积分，一位代表解释说，目前用户处于最高计划后没有购买额外积分的选项，但建议优化 Prompt 或等待下个月的额度刷新。
   - 该代表还提到，参加官方活动是获取额外积分的好方法，并表示会将用户的想法反馈给产品团队，作为未来可能更新的参考。
- **Pro 订阅者对 Manus v1.5 的不满**：一位资深 Pro 订阅者对 **Manus v1.5** 表示不满，指出 **300 个每日积分的消耗速度惊人**（实际使用不到 30 分钟）。
   - 他们声称，*宣传为免费*的聊天模式在积分余额归零后便无法使用，且在有积分时仍会消耗积分，导致该用户无法完成推荐 Manus 的 LinkedIn 演示文稿，被迫转而使用 ChatGPT。
- **Pro 订阅者要求采取行动**：Pro 订阅者要求*积分消耗完全透明*、提供*真正免费且可用的聊天模式*、*立即审查 v1.5 中引入的积分政策*，或者至少对受影响的 Pro 订阅者给予明确补偿。
   - 他们担心该产品具有真正的潜力，但在目前的状态下，Pro 体验已无法兑现其承诺。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1452517905249079417)** (6 条消息): 

> `GLM-4.7 模型, Solar-Open-100B` 


- **GLM-4.7 模型发布**：正如 [HuggingFace 链接](https://huggingface.co/zai-org/GLM-4.7) 所示，Air 的下一次迭代目前可能是 **GLM 4.7 Air**。
- **Upstage 发布 Solar-Open-100B**：**Upstage** 发布了 **Solar-Open-100B** 模型，并在 [HuggingFace](https://huggingface.co/upstage/Solar-Open-100B) 上向社区公布。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1452517944239198310)** (3 条消息): 

> `Fine Tuning 最小样本, 模型复制风格` 


- **Fine Tuning 最小样本研究**：一名成员询问关于模型通过 **Fine-tuning** 复制特定风格所需的**最小样本数量**的研究。
   - 另一名成员澄清了该问题是否属于复制单一作者的风格，原提问成员回答说，他们对写作类型感兴趣，例如 **purple prose、写实写作或小说**。
- **写作风格偏好**：讨论围绕不同的写作风格展开，从 *purple prose* 到更写实或虚构的方法。
   - 最初的问题旨在了解如何通过最少的样本集对模型进行 Fine-tuned，使其采用这些多样的写作风格。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1452517944239198310)** (3 条消息): 

> `Fine-tuning, 最小样本, 写作风格复制` 


- **探讨风格复制的最小样本**：成员们讨论了关于模型通过 **Fine-tuning** 复制特定写作风格所需最小样本数量的研究。
   - 讨论涉及辨别不同的写作风格，如 *purple prose*、*写实写作*和*小说*。
- **风格迁移学习 (Style Transfer Learning)**：对话还考虑了通过 Fine-tuning 捕捉各种写作偏好的细微差别。
   - 它触及了不同的风格倾向，从喜欢 *purple prose* 的人到青睐*写实写作*和*小说*的人，偏好程度各不相同。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1452697587403456643)** (1 messages): 

> `Norm weights, Weight Decay, Learning Rate` 


- **Norm Weights 可能需要特殊处理**：一位成员回忆起一些论文声称 **norm weights** 需要被独立出来，因为 **weight decay (wd)** 和 **learning rate (lr)** 的表现都非常不同。
- **Norm weights 及其独特行为**：讨论强调了在神经网络中将 **norm weights** 与其他权重区别对待的潜在必要性。
   - 讨论指出，像 **weight decay** 和 **learning rate** 这样的标准优化技术可能会以意想不到的方式影响 norm weights，值得进一步研究。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1452585988969725996)** (3 messages): 

> `Singular Value Density, Marcenko-Pastur distribution` 


- **建议使用奇异值密度度量**：一位成员建议，用于密度的“最佳”度量是**梯度奇异值密度**的经验值（或理论近似值），并引用了关于 [Marcenko-Pastur 分布的 Wikipedia 文章](https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution)。
- **Marcenko-Pastur 分布的注意事项**：另一位成员指出，**Marcenko-Pastur 分布**仅针对随机矩阵进行测量。
   - 他们补充说，原始的奇异值界限比新界限**差 9 倍**，因此最大输入将*小于 1/9*。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1452763104848121897)** (5 messages): 

> `Implicit Backdoors, Semantic Data Tagging, Cultural Personas Training` 


- **隐式后门论文引发关注**：一位成员发现一篇关于 [隐式后门 (Implicit Backdoors)](https://arxiv.org/abs/2512.09742) 的论文非常酷，认为它凸显了利用语义相关信息标记数据的巨大潜力。
   - 该成员想知道这种方法是否可以训练模型拥有多个独立且互不干扰的 **cultural personas**（文化人格）。
- **提议通过元数据预训练实现文化敏感性**：一位成员建议在预训练模型时添加前置元数据（如作者、日期和来源类型），以便将来对行为进行微调。
   - 目标是通过微调模型使其“相信”当前是 **2025** 年，从而防止不良行为，例如推荐*20 世纪 20 年代基于优生学的医学*。
- **元数据记录的松散先验**：该成员指出，这个实验的有趣之处在于，你在预训练时不需要知道哪些数据是相关的，或者需要抑制哪些行为。
   - 在预训练阶段，你只需要有一个**松散先验 (loose prior)** 来启发你记录那些以后会有用的元数据。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/)** (1 messages): 

shalokshalom: 有一个 Pixi Discord 频道。
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1452607231647285330)** (5 messages): 

> `UnsafePointer, Linear Typed Pointer` 


- **考虑为 `UnsafePointer` 支持 `with` 语句**：有人提问是否支持为 **UnsafePointer** 实现 `with` 语句（进入/退出实现），以便在简单场景下更易于使用。
   - 一位成员认为 `UnsafePointer` 仍将是一个非常锋利的工具，但社区可能会获得一种 **linear typed pointer**（线性类型指针），它由于要求必须手动释放而稍微更安全一些。
- **关于 `UnsafePointer` 的不安全逃生口讨论**：一位成员建议为 `UnsafePointer` 本身添加一个不安全逃生口，类似 `unsafe_drop(deinit self)`。
   - 另一位成员回应称，`UnsafePointer` 从根本上是一个引用，而不是一个拥有所有权的值（owning value），因此它可以指向线性类型，而其本身不必是线性的。


  

---

### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1452594688447217684)** (5 messages): 

> `Google Summer of Code, MCP Token Cost Efficiency, Lazy Transmission of Tool Schemas, Caching Protocol Definitions` 


- **贡献者询问：Google Summer of Code 参与情况？**：一名成员询问 **MCP 委员会** 是否计划参加今年的 [Google Summer of Code (GSoC)](https://summerofcode.withgoogle.com/)，并指出申请窗口为 **1 月 19 日至 2 月 3 日**。
- **MCP 协议：Token 成本危机？**：一名成员发起了关于 **Token 使用量** 作为 **基于 MCP 集成** 成本问题的讨论，认为目前的集成方式通过重复发送大型协议描述（即使是未使用的工具）导致了更高的单次请求 Token 支出。
- **延迟传输：按需提供 Schemas？**：一名成员询问 **MCP** 是否可以支持 **Tool Schemas 的延迟或按需传输** 以降低 Token 成本，但另一名成员表示，由 Client Host 决定是否将 Schema 发送给模型。
   - 他们澄清道：*"如果不将 Tool Schema 传递给模型，该工具也无法使用。"*
- **缓存协议定义以节省 Token**：一名成员询问 **协议定义** 是否可以 **在请求之间缓存或引用** 以避免重复发送，但另一名成员回答说 Client Host 可以独立实现缓存方案。
   - 该成员表示：*"如果你监听变更通知，你只需要为 List 方法发送一次请求，而不需要 [重复发送]。"*


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1452518114653769923)** (2 messages): 

> `LLM App, Llama Training, Viz, Driver, Bounties` 


- **公司更新会议开始**：会议议程以 **公司更新** 拉开序幕。
   - 会议定于周一假期的 **圣迭戈时间上午 9 点** 举行。
- **LLM App 议程项目浮现**：关于 **新 LLM App** 的讨论被列入会议议程。
   - 该 App 的具体细节尚未透露。
- **Llama 训练成为核心**：议程还强调了 **Llama 训练**，特别是关注 **Grad Acc**、**JIT 编译** 和 **Flash Attention**。
   - 这些是优化 Llama 模型的关键领域。
- **Viz 和 Driver 讨论推进**：会议还计划讨论 **可视化 (Viz)** 和 **驱动程序 (Driver)** 方面的内容。
   - 这些可能涉及工具链或特定的硬件集成工作。
- **悬赏任务（Bounties）大放送**：会议讨论了其他可用的 **Bounties**。
   - 提供了一个关于 [衡量 AI 完成长任务的能力](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) 的帖子链接，可能与悬赏任务相关。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1452750255920775280)** (2 messages): 

> `solve.it lesson 8` 


- **建议学习 Solve it 课程**：一名成员询问在哪里可以找到某些内容，另一名成员提供了 [solve.it 第 8 课](https://solve.it.com/) 的链接。
   - 未提供关于该课程内容的进一步细节。
- **Solve it 链接**：已找到 Solve it 的第 8 课。
   - 链接至 [solve.it](https://solve.it.com/)。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1452630764016636092)** (4 messages): 

> `Microtransactions, GenAI in Gaming, Public Sentiment, Vince Zampella` 


- **微交易面临愤怒抵制**：一名成员表示，由于游戏社区现存的愤怒情绪，**微交易短期内不会实现**，并引用了 **《湮灭》马铠包 (Oblivion horse armor pack)** 的争议。
- **游戏社区的反 GenAI 立场**：一名成员对游戏社区内强烈的 **反 GenAI** 情绪表示惊讶，特别是在 **Steam** 的评论和讨论中。
   - 他们承认这可能只是 *发声的少数派*，但仍然是一个显著的存在。
- **公众舆论容易摇摆**：一位成员认为，公众对游戏开发相关问题的看法很容易改变，现在开始开发游戏的 AAA 工作室到游戏完成时就不必担心了。
- **Vince Zampella 之死**：一名成员在查找图片后提到了 **Vince Zampella** 的突然去世，并形容这感觉 *很诡异*。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1452714572455870595)** (2 messages): 

> `AI 工具, 实用 AI 系统, AI 开发者寻求团队` 


- **AI 工具对非 Agentic 任务极具价值**：一位成员表示，虽然一年前他可能持不同意见，但现在他认为工具对于非 Agentic 任务（如 **浏览器访问** 和 **读取可用函数/方法**）非常有价值。
- **AI 开发者寻求可靠团队**：一位 AI 开发者正在寻找一个从事稳定且有意义项目的团队，强调在构建处理实际任务的实用系统方面的经验。
   - 该开发者重视清晰的工作流程和一致性，能够提供可靠性以帮助推动项目进展。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1452563301862080597)** (1 messages): 

> `Subversion, jj, 项目仓库, Aider 的游乐场` 


- **SVN 和 jj 用户获得自动版本控制**：使用非 Git 版本控制系统（如 **Subversion (SVN)** 或 **jj**）可以使版本控制自动化。
   - 这在主项目仓库跟踪服务器的真实提交，而本地 Git 实例作为 *aider's playground* 时特别有用。
- **SVN 保持整洁**：将 **SVN** 作为项目仓库，通过每 10-20 个 aider Git 提交才进行一次提交，可以使修订日志更加整洁。
   - 这种方法可以防止临时文件、文档和其他 Aider 相关内容扰乱主版本控制系统。


  

---


---


---