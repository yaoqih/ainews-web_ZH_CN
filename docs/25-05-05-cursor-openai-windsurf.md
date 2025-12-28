---
companies:
- openai
- cursor
- nvidia
- alibaba
- deepseek
- microsoft
- baidu
- suno
- runway
- keras
date: '2025-05-05T05:44:39.731046Z'
description: 据报道，**OpenAI** 即将完成与 Windsurf 的交易，与此同时，**Cursor** 以 90 亿美元的估值完成了 9 亿美元的融资。**英伟达（Nvidia）**推出了
  **Llama-Nemotron 系列**，包含从 80 亿到 2530 亿参数的模型，其逻辑推理和推断效率备受称赞。**阿里巴巴**发布了 **通义千问 Qwen3
  系列**，涵盖了参数规模最高达 2350 亿的 MoE（混合专家）模型和稠密模型，在编程和数学基准测试中名列前茅。**DeepSeek** 推出了 **Prover-V2**，这是一款开源的数学推理
  AI，在 MiniF2F-test 测试中的通过率达到 88.9%。**微软**发布了专注于推理的 **Phi-4 模型**，其性能超越了 OpenAI 的 **o1-mini**。**百度**首次推出了
  **文心一言（ERNIE）4.5 和 X1** 的 Turbo 版本，以实现更快、更廉价的推理。**Suno v4.5** 增加了先进的 AI 音乐生成功能，而
  **Runway Gen-4 References** 则能够将角色以高度一致性置入场景中。由 **François Chollet** 开发的针对 TPU 优化的新型推荐系统库
  **KerasRS** 也已正式发布。
id: MjAyNS0w
models:
- llama-nemotron-ultra
- llama-nemotron-super
- llama-nemotron-nano
- qwen3-235b-a22b
- prover-v2
- phi-4-reasoning
- ernie-4.5-turbo
- ernie-x1-turbo
- suno-v4.5
- gen-4-references
- o1-mini
people:
- _akhaliq
- adcock_brett
- lmarena_ai
- fchollet
title: Cursor 估值达 90 亿美元，OpenAI 以 30 亿美元收购 Windsurf。
topics:
- reasoning
- inference-efficiency
- open-license
- moe-models
- math-reasoning
- theorem-proving
- model-performance
- music-generation
- image-generation
- recommender-systems
- tpu-optimization
---



有很多值得关注的观点，比如[这一个，](https://x.com/deedydas/status/1915083513189298620)我们需要等待一段时间才能了解完整的细节，但第一个 “AI Wrapper” 独角兽的退出确实具有新闻价值。

---

# AI Twitter 综述

**模型发布、更新与功能**

- **Nvidia 的 Llama-Nemotron 系列**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1919234521171693844) 强调了 **NVIDIA** 推出的 **Llama-Nemotron 系列**，这是一个开放的异构推理模型家族，强调了其卓越的推理能力、推理效率以及面向企业用途的开放许可证。值得注意的是，截至 2025 年 4 月，旗舰模型 **LN-Ultra** 被 **Artificial Analysis** 认为是“最智能”的开放模型。该系列包括 **Nano (8B)、Super (49B) 和 Ultra (253B)** 版本。[@_akhaliq](https://twitter.com/_akhaliq/status/1919324939934453928) 还提到 **Nvidia** 在 Hugging Face 上发布了作为高效推理模型的 **Llama-Nemotron**。
- **阿里巴巴的 Qwen3 家族**：[@adcock_brett](https://twitter.com/adcock_brett/status/1919060402417119375) 报道了 **阿里巴巴 Qwen 团队** 发布 **Qwen3 家族**，包含 **2 个 MoE 模型和 6 个稠密模型**，参数量从 **600M 到 235B** 不等。[@lmarena_ai](https://twitter.com/lmarena_ai/status/1919448953042706759) 指出 **Qwen3-235B-A22B** 位列 **Arena 前 10 名**，在编程领域排名第 4，数学领域排名第 1，在 WebDev 领域排名第 5。
- **DeepSeek Prover-V2**：[@adcock_brett](https://twitter.com/adcock_brett/status/1919060364655800684) 报道了 **DeepSeek** 发布的 **Prover-V2**，这是一款结合了非正式数学推理与定理证明的开源 AI，拥有 **671B 参数**，在 **MiniF2F-test 上达到了 88.9% 的通过率**。
- **微软的 Phi-4 模型**：[@adcock_brett](https://twitter.com/adcock_brett/status/1919060284078997565) 讨论了 **微软** 发布的三个专注于推理的 **Phi-4 模型**，其中拥有 **14B 参数的 Phi-4-reasoning** 表现优于 **OpenAI 的 o1-mini**。
- **百度的 ERNIE Turbo 版本**：[@adcock_brett](https://twitter.com/adcock_brett/status/1919060425770942619) 指出 **百度** 推出了 **ERNIE 4.5 和 X1 的 Turbo 版本**，具有更快的速度和更低的成本。
- **Suno v4.5 发布**：[@adcock_brett](https://twitter.com/adcock_brett/status/1919060448264987109) 报道了 **Suno v4.5** 的发布，其具有全新的 AI 音乐生成功能，如新流派、增强的人声、复杂的音效、更好的 Prompt 遵循能力，以及创作长达 8 分钟歌曲的能力。[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1919001539592462612) 提到他不再听人类创作的歌曲，因为他觉得 **Suno** 的歌曲更好，并预计随着时间的推移，这种情况会变得更加普遍。
- **Runway Gen-4 References**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1919074634042933399) 强调 Runway References 可以穿越回 **1656** 年，为我们展示《宫娥》（**Las Meninas**）的航拍场景和侧拍镜头。[@adcock_brett](https://twitter.com/adcock_brett/status/1919060471019114790) 报道称 **Runway** 向所有付费客户推出了 **Gen-4 References**，允许使用照片、图像、3D 模型或自拍将角色放入任何场景中，并保持高度的一致性。
- **KerasRS 库**：[@fchollet](https://twitter.com/fchollet/status/1919477586599805118) 宣布发布 **KerasRS**，这是一个用于构建推荐系统的新库，提供易于使用的构建模块，并兼容 **JAX, PyTorch, TF**，且针对 **TPUs** 进行了优化。
- **D-FINE 目标检测器**：[@mervenoyann](https://twitter.com/mervenoyann/status/1919431751689998348) 宣布了 **D-FINE**，这是一款实时目标检测器，比 **YOLO** 更快、更准确，采用 **Apache 2.0** 许可证，已上线 Hugging Face Transformers，并可在 **T4 (免费版 Colab)** 上运行。
- **Meta 的 LlamaCon 公告**：[@adcock_brett](https://twitter.com/adcock_brett/status/1919060231771877793) 总结了 **Meta** 在其首届 **LlamaCon** 开发者大会上的公告，包括 Llama API 免费预览版、带有“发现”流的类 ChatGPT 的 Meta AI 应用、Llama Guard 4 (12B)、LlamaFirewall、Prompt Guard，以及支持 Groq 和 Cerebras 的 Colab。

**基于 Agent 的框架和工作流**

- **AWS AI Agents 框架**：[@LiorOnAI](https://twitter.com/LiorOnAI/status/1919105151295451350) 报道 **AWS** 发布了一个开源框架，允许你编排多个 AI agents 并处理复杂的对话，且支持在本地计算机部署。
- **Cisco Outshift 的 Agentic AI 工程师**：[@LangChainAI](https://twitter.com/LangChainAI/status/1919399184664236523) 讨论了 **Cisco Outshift** 使用 **JARVIS**（一个基于 **LangGraph 和 LangSmith** 构建的 AI Platform Engineer）来自动化开发者请求并消除运维瓶颈。
- **Agentic 模式与设计**：[@_philschmid](https://twitter.com/_philschmid/status/1919391587315958038) 分享了一份学习常用工作流和 Agentic 设计模式的指南，包含 Google DeepMind Gemini 的代码片段，涵盖了 prompt chaining、routing、parallelization、reflection、tool use、planning 以及 multi-agent systems。
- **DSPy 的 GRPO 发布**：[@lateinteraction](https://twitter.com/lateinteraction/status/1919428454761553994) 宣布实验性发布 **dspy.GRPO**，这是一个用于 DSPy 程序的在线 RL 优化器，由 [@NoahZiems](https://twitter.com/NoahZiems)、[@LakshyAAAgrawal](https://twitter.com/LakshyAAAgrawal) 和 [@dilarafsoylu](https://twitter.com/dilarafsoylu) 领导。[@lateinteraction](https://twitter.com/lateinteraction/status/1919428467487342855) 强调了 **DSPy 的 Arbor server** 及其对 Hugging Face TRL 等巨头以及 [@willccbb](https://twitter.com/willccbb) 验证器代码的依赖，旨在鼓励 RL 研究人员研究 AI 软件的优化。

**基准测试、评估与可解释性**

- **LmSys Chatbot Arena 问题**：[@TheAITimeline](https://twitter.com/TheAITimeline/status/1919155704579142047) 总结了一篇论文，该论文指出了扭曲 **Chatbot Arena 排名** 的系统性问题，揭示了未公开的私有测试和选择性分数披露如何导致偏差结果。[@aidangomez](https://twitter.com/aidangomez/status/1919058386668200029) 敦促 **LmSys** 承认失败并重构流程以防范此类问题，而不是攻击 Sara。
- **Scaling 的 LLM 元排行榜**：[@scaling01](https://twitter.com/scaling01/status/1919217718420508782) 推出了 **Ultimate LLM Meta-Leaderboard**，该榜单综合了 28 个最佳基准测试的平均值，其中 **Gemini 2.5 Pro** 的排名高于 **o3** 和 **Sonnet 3.7 Thinking**。[@scaling01](https://twitter.com/scaling01/status/1919389344617414824) 随后通过手动数据清洗和 Glicko-2 评分系统更新了排行榜，强调了保守的低技能预估标签。
- **基准测试的相关性**：[@scaling01](https://twitter.com/scaling01/status/1919092778648408363) 提供了一份 LLM 基准测试列表，区分了有价值的基准测试和那些被认为已饱和且无信号的基准测试，强调了对概念简单基准和真实世界任务的偏好。[@lateinteraction](https://twitter.com/lateinteraction/status/1919054877583667507) 认为，由于现代 post-training 模式和评估滞后，标准基准测试往往适得其反。
- **METR 基准测试的局限性**：[@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1919059455020286440) 分析了 **METR benchmark** 的局限性，指出时间跨度估计具有领域特定性，可靠性阈值各异，且真实世界任务被捆绑在一起，缺乏必要的数据和上下文。
- **Qwen3 在 LiveCodeBench 上的表现**：[@huybery](https://twitter.com/huybery/status/1919418019517776024) 注意到 **Qwen3-235B-A22B** 在 **LiveCodeBench** 上表现出色，使其成为竞赛级代码生成的顶级开源模型。
- **Hash-Conditioning 方法**：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1918816062772269266) 报道了来自 **GoogleAI** 和 **Carnegie Mellon** 的一种方法，通过在输入阶段添加少量噪声来使模型的回答更具创造性，尤其是在开放式任务中，提高了小型和大型模型的创造力。
- **可解释性的重要性**：[@NeelNanda5](https://twitter.com/NeelNanda5/status/1919070513227550735) 表示支持对可解释性的投资，但认为其重要性相对于其他安全方法被夸大了，将其视为更大安全组合的一部分，而非实现可靠保障的唯一路径。
- **评估模型的隐蔽性与意识**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1919237438402343079) 分享了一篇论文，该论文提出了 5 项关于推理和规避监管能力的评估，以及 11 项衡量模型对其自身、环境和部署进行工具性推理能力的评估，结论是目前没有 SotA 模型在任一能力上表现出令人担忧的水平。

**机器人与具身智能 (Robotics and Embodied AI)**

- **Figure 与 BMW Group 的合作伙伴关系**：[@adcock_brett](https://twitter.com/adcock_brett/status/1919421360138539053) 分享了 **Figure** 团队在 **BMW Group Plant Spartanburg** 完成了为期两周的高效访问，优化了 **X3** 车身车间机器人的流程，并探索了工厂内的新用例，对 2025 年的合作伙伴关系充满期待。
- **ABB Robotics 与 BurgerBots**：[@adcock_brett](https://twitter.com/adcock_brett/status/1919060515998822898) 报道称 **ABB Robotics** 和 **BurgerBots** 在 Los Gatos 开设了新的“快餐”店，机器人使用 ABB 的 **IRB 360 FlexPicker** 和 **YuMi bot**，在 27 秒内组装完成售价 18 美元的定制汉堡。
- **Glacier 的废物管理机器人**：[@adcock_brett](https://twitter.com/adcock_brett/status/1919060561070870909) 报道了 **Amazon 支持的 Glacier** 筹集了 1600 万美元，旨在将物理 AI 引入废物管理领域，制造利用计算机视觉自动分拣垃圾和可回收材料的机器人。
- **Deep Robotics 的 Lynx M20**：[@adcock_brett](https://twitter.com/adcock_brett/status/1919060538379677767) 重点介绍了中国 **Deep Robotics**（云深处科技）推出的 **Lynx M20**，这是其机器狗的坚固版本，专门用于电力巡检、应急响应和物流等任务。
- **Dyna Robotics 的 DYNA-1**：[@adcock_brett](https://twitter.com/adcock_brett/status/1919060493488070677) 介绍了 **Dyna Robotics** 的 **DYNA-1**，这是一个用于高吞吐量灵巧任务的机器人基础模型，演示了在 24 小时内折叠 850 多张餐巾纸，成功率达 99.4%，且无需人工干预。

**AI 与代码**

- **AutoGen HTML 和 Tailwind CSS 代码生成**：[@reach_vb](https://twitter.com/reach_vb/status/1919356751528235232) 介绍了 **UIGEN-T2**，专门设计用于为 Web 界面生成 HTML 和 Tailwind CSS 代码。
- **SWE-bench 与 SWE-agent 讨论**：[@OfirPress](https://twitter.com/OfirPress/status/1919460877784240522) 宣布将于 5 月 21 日举行一场演讲，分享他们如何构建 **SWE-bench** 和 **SWE-agent**，以及对自主 AI 系统未来的展望。
- **Cline 更新**：[@cline](https://twitter.com/cline/status/1919119386738393487) 宣布其 v3.14 版本增加了数学支持，并通过 newrule 实现了标准强制执行。
- **Agent 驱动的法律起草**：[@scottastevenson](https://twitter.com/scottastevenson/status/1919076281183875581) 征求一个更好的术语来描述 Agent 驱动的法律起草，并将其与 "vibe coding" 进行类比。

**ASR 模型**

- **Nvidia 的 Parakeet TDT 0.6B**：[@reach_vb](https://twitter.com/reach_vb/status/1919422953256587376) 宣布 **Nvidia** 开源了 **Parakeet TDT 0.6B** 语音识别模型，强调其能够在 1 秒内转录 60 分钟的音频，并采用了商业许可。

**讨论与评论**

- **AI 对阅读习惯的影响**：[@hyhieu226](https://twitter.com/hyhieu226/status/1919068971845976113) 对 LLM 可能降低阅读热情表示担忧，因为用户正变得习惯于消费简短、精炼的文本块，类似于对 TikTok 短视频的成瘾。
- **实用 AI 技能的重要性**：[@omarsar0](https://twitter.com/omarsar0/status/1919432255350477125) 强调大多数 **YouTube** 教程都忽略了构建 AI Agent 最重要的一点，即迭代开发，并强调了系统性改进 AI 系统的必要性。
- **周末编程的益处**：[@akshat_b](https://twitter.com/akshat_b/status/1918828178358809026) 表示他成功让两个容器实现了 UDP hole-punch 并通过 QUIC 进行通信。
- **旧金山的“鬼城”状态**：[@willdepue](https://twitter.com/willdepue/status/1919282493498278231) 将旧金山描述为一座鬼城，原因是友谊的短暂性以及科技劳动力的输入性质——他们随时准备像到来时一样迅速离开。
- **农业工人的减少**：[@willdepue](https://twitter.com/willdepue/status/1919209507206406226) 指出自动化已经发展到如此程度，以至于从 1910 年到 2000 年代，尽管人口和经济规模巨大，农业就业人数却比 1910 年有所缩减。

**梗/幽默**

- **从科技男到法西斯**：[@dylan522p](https://twitter.com/dylan522p/status/1918797220121301019) 观察到从古怪的 Reddit 自由派科技青年到肥胖法西斯分子的转变。
- **Sergey Brin 的超级游艇**：[@claud_fuen](https://twitter.com/claud_fuen/status/1918802361901830151) 调侃 Sergey Brin 开着超级游艇出现，提醒他自己“还没到财务自由后的阶段”。
- **Klarna 保证金追缴**：[@willdepue](https://twitter.com/willdepue/status/1918857828170956994) 在他的 DoorDash 卷饼上被追缴了保证金（margin called）。
- **毛绒玩具 wypipo**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1919303167776305176) 注意到 wypipo 玩具阵线的兴起。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

### 1. Qwen3 235B 模型基准测试与性能指标

- [**Qwen 3 235b 在 LiveCodeBench 中获得高分**](https://i.redd.it/px3okqrznzye1.jpeg) ([分数: 145, 评论: 46](https://www.reddit.com/r/LocalLLaMA/comments/1kffq2u/qwen_3_235b_gets_high_score_in_livecodebench/)): **提供的图片显示了 LLM 代码评估的 LiveCodeBench 排行榜，其中 Qwen 3 235b 模型（A22B 版本）排名第 7。它实现了 65.9 的 Pass@1 分数，99.1 的 Easy-Pass@1 以及 80 的 Medium-Pass 分数，表明与竞争对手模型相比具有强大的代码生成性能。该排行榜展示了 Qwen 3 235b 在开源和闭源模型中的地位，突显了其在代码相关任务中的技术竞争力。[排行榜图片](https://i.redd.it/px3okqrznzye1.jpeg)** 评论指出，虽然 Qwen 3 235b 得分很高，但其实际应用受到其相对较小的上下文窗口（32k，可扩展至 128k）的限制，这导致在处理冗长任务时效率较低。还有关于用户偏好 Gemini Pro 和较新的 Claude 3.x Sonnet 等其他模型的讨论，这是由上下文管理和生成风格而非原始基准测试分数驱动的。
    - 技术用户指出，尽管 Qwen 3 235B 在 LiveCodeBench 分数和模型质量方面表现强劲，但由于上下文窗口大小受限（标准为 `32k` token，扩展后可达 `128k`），在某些用例中受到阻碍。这种限制影响了需要冗长推理或大型文档处理的任务；DeepSeek R1T Chimera 等模型因其效率和简洁的推理输出而被提及，在这些场景中表现优于 Qwen。
    - 一位用户分享了在强大的本地硬件（AMD 9950x3d，`192GB DDR5`，总计 `48GB VRAM` 的双 GPU）上将 Qwen 3 235B-A22B 作为 MoE 模型运行的详细 LLAMA 推理设置。值得注意的是，将每个 token 的专家数量从默认的 `8` 调整为 `4` 后，吞吐量从 `5 tokens/sec` 提高到 `7 tokens/sec`。启动命令展示了高级配置，例如显式张量卸载（tensor offloading）规则、批大小（batch sizes）、启用 Flash Attention 以及多种采样器策略，证明了在本地运行大型 MoE 模型的灵活性。
- [**Qwen3-32B-IQ4_XS GGUFs - MMLU-PRO 基准测试比较**](https://www.reddit.com/r/LocalLLaMA/comments/1kf1yg9/qwen332biq4_xs_ggufs_mmlupro_benchmark_comparison/) ([分数: 116, 评论: 50](https://www.reddit.com/r/LocalLLaMA/comments/1kf1yg9/qwen332biq4_xs_ggufs_mmlupro_benchmark_comparison/)): **楼主（OP）对来自不同来源的使用 IQ4_XS 量化的几个 Qwen3-32B GGUF 进行了基准测试（提供了 GGUF 链接），使用了 MMLU-PRO 0.25 子集（3003 个问题，temp=0，'No Think'，Q8 KV Cache），运行耗时约 11.6 小时。结果显示不同来源之间的综合准确率差异极小，尽管楼主指出这些 IQ4_XS 量化版本的得分高于官方 MMLU-PRO 排行榜（该排行榜测试的是基础模型，而非 instruct 变体）。完整的排行榜背景：测试仅使用了 25% 的子集，导致每个类别约有 230 个问题，估计每个类别的置信区间约为 ±5%，这可能会在子领域性能比较中引入显著噪声。** 评论者强调 Unsloth 量化版本在某些学科（特别是计算机科学和工程）中似乎更优，但在其他学科（健康、数学、物理）中落后，并质疑考虑到测试子集大小的统计显著性。有人提到“噪声”可能会掩盖真实的差异，建议需要全集基准测试才能得出结论性结果。还有人要求对其他量化方案（如 Q4K_XL、Q2K_XL；UD-Q3_K_XL 和 UD-Q4_K_XL）进行基准测试，并询问 IQ4_XS 与替代方案相比的独特特征或原理。
    - 几位用户讨论了量化对模型性能的影响，特别是量化会显著降低某些模型（如 30B A3B）的效果。人们对 CPU 优化的量化版本（如 Q4K_XL 和 Q2K_XL）的基准测试很感兴趣，据文档显示，Q2K_XL 提供了最佳的性能与大小比（每 GB 的 CPU 推理速度更快）。
    - 一项详细分析指出，仅使用 MMLU Pro 数据集的 25%（每个类别约 230 个问题）进行基准测试会导致较高的置信区间（±5%），从而引入统计噪声。运行完整的 12k 问题集将使每个类别的置信区间降低到 ±2.5% 左右，从而能够更好地比较量化策略，包括 UD-Q3_K_XL 和 UD-Q4_K_XL 量化版本，以获得进一步的见解。
    - 讨论强调了对键值（KV）量化的关注：据报道 Qwen 模型对 KV 量化很敏感，一些用户注意到禁用 KV 量化时性能有显著提升。此处的基准测试使用了 8-bit KV；建议不使用 KV 量化的对比测试可能会产生明显不同的模型结果。

- [**在消费级主板/CPU (llamacpp/ikllamacpp) 上使用 128GB VRAM (5090+4090x2+A6000) + 192GB RAM 运行 DeepSeekV3 0324/Qwen3 235B 及其他模型的速度指标**](https://www.reddit.com/r/LocalLLaMA/comments/1kezq68/speed_metrics_running_deepseekv3_0324qwen3_235b/) ([Score: 102, Comments: 31](https://www.reddit.com/r/LocalLLaMA/comments/1kezq68/speed_metrics_running_deepseekv3_0324qwen3_235b/)): [总结帖子时出错]
    - 一位用户强调，如果没有适当的 Tensor Parallel 设置，仅仅增加 VRAM 是不够的，因为大多数引擎（exl2 除外）需要规格相似的 GPU 才能实现高效扩展。在服务器主板上切换到 5x3090 的配置后，他们将 70B q4 模型的吞吐量从 18 tok/s（串行）提升到 vLLM Tensor Parallel 下的 36 tok/s，在特定代码任务中使用 Speculative Decoding 甚至达到了 75 tok/s。这说明了并行化而非简单增加更多或不匹配的 GPU 所带来的显著效率提升。
    - DeepSeek-R1 在实际使用中有了显著改进，特别是在处理长上下文方面，这表明其内存管理得到了增强，并且在更大的 Batch 或序列长度上可能具有更好的推理速度或稳定性。

### 2. 本地 LLM 的多模型 GPU 编排与硬件

- [**RTX 5060 Ti 16GB 在游戏方面表现不佳，但在 AI 领域似乎是一颗“沧海遗珠”**](https://www.reddit.com/gallery/1kf9i52) ([Score: 282, Comments: 225](https://www.reddit.com/r/LocalLLaMA/comments/1kf9i52/rtx_5060_ti_16gb_sucks_for_gaming_but_seems_like/)): **原帖作者对运行 AI 工作负载（通过 Ollama 使用 Mistral Nemo Instruct 12B 的 LightRAG）的 RTX 5060 Ti 16GB 进行了基准测试，证明额外的 VRAM 允许所有 41 个模型层都装入内存，将推理时间从 12GB 显卡（RTX 3060 Ti，仅能加载 31 层，导致内存交换且性能慢 2 倍）的 8:52 减少到 3:29。帖子包含了展示 GPU 利用率差异的 Grafana 指标，并提供了 LightRAG 自动化设置指南的链接 (https://github.com/sbnb-io/sbnb/blob/main/README-LightRAG.md)。值得注意的是，5060 Ti 16GB 的物理尺寸更短（双风扇，PCIe x8），使其对 SFF（小型化）AI 装备非常有吸引力。** 评论者澄清说 16GB 版本对于游戏是可以接受的，并质疑是否存在 12GB 的 RTX 3060 Ti，指出仅存在 8GB (Ti) 和 12GB (non-Ti) 版本。此外，还有人要求以 tokens/second (t/s) 报告结果，以便获得更有意义的 AI 吞吐量指标。
    - 16GB 的 RTX 5060 Ti 版本被认为适合游戏，批评通常针对 8GB 版本的局限性。`8GB` 和 `16GB` VRAM 之间的区别在技术上非常重要，特别是对于内存密集型游戏和 AI/ML 工作负载。
    - 针对上一代 GPU 提出了事实澄清：不存在具有 12GB VRAM 的 3060 Ti；3060 Ti 最高为 8GB，而非 Ti 的 3060 则提供 12GB。这一对比突显了 RTX 5060 Ti 16GB 对 AI 的潜在吸引力，因为它的可用 VRAM 高于过去几代的许多中端 GPU。
    - 关于使用双 RTX 5060 Ti 显卡的讨论指出，这有可能实现 `32GB` VRAM 的配置，这对于支持多 GPU 配置的深度学习框架非常有益。文中提到了物理尺寸考虑和成本（低于 500 美元），将 5060 Ti 视为双 3060 12GB 系统的一个实用的、以 AI 为中心的替代方案。
- [**我们在 2 个 GPU 上运行了 50 多个 LLM —— 冷启动时间低于 2 秒。以下是实现方法。**](https://www.reddit.com/r/LocalLLaMA/comments/1kfcdll/we_fit_50_llms_on_2_gpus_cold_starts_under_2s/) ([Score: 138, Comments: 64](https://www.reddit.com/r/LocalLLaMA/comments/1kfcdll/we_fit_50_llms_on_2_gpus_cold_starts_under_2s/)): **一个团队开发了一个自定义推理运行时，能够在双 NVIDIA A4000 GPU 上编排 50 多个 LLM，实现低于 2 秒的冷启动延迟和超过 90% 的 GPU 利用率。这是通过直接在 GPU 上对完整的模型执行状态（包括 Attention Caches 和内存布局）进行快照和恢复来实现的，有效地挂起和恢复模型，类似于操作系统中的进程管理。该工具解决了多模型设置中的常见瓶颈，包括内存膨胀和低效的 GPU 分配；用例包括 RAG 和 Agent 流水线。** 热门评论对开源或提供该运行时的文档表示了浓厚兴趣，强调了充满前景的技术主张与现实世界可复现性之间的关键差距。
    - 多位评论者要求发布公共代码或详细文档，强调了主张的可复现性和验证的重要性，特别是当文中点名提到了 vLLM 等替代推理解决方案时。缺乏 GitHub 链接或技术文档被视为进一步技术评估或采用的障碍。

- 这类比于大规模服务器卸载（offloading）和资源调度（shuffling），特别是参考了 homelab 架构，其中不活跃的进程会从 RAM 被驱逐到远程存储（例如 AWS）。这表明该项目的冷启动（cold-start）机制可能依赖类似的内存和资源管理策略，以在大量 LLM 集合中实现快速模型激活。
- [**我该先测试/运行什么？**](https://www.reddit.com/gallery/1kexdgy) ([Score: 472, Comments: 232](https://www.reddit.com/r/LocalLLaMA/comments/1kexdgy/what_do_i_test_out_run_first/))：**该帖子讨论了收到一个未指明的设备或组件（可能是 GPU 或 AI 加速器），并寻求初始测试建议。热门技术评论建议运行 'llama 3.2 1b'（参考 Meta 的 Llama 模型的 10 亿参数版本，参见 https://ai.facebook.com/blog/llama-meta-ai-large-language-model/）和 'LLAMA 405B Q.000016'（可能参考了一个量化级别为 Q.000016 的 4050 亿参数 Llama 模型检查点）。这些建议暗示通过大语言模型推理工作负载进行基准测试，以进行压力测试。** 没有重大的技术争论；评论只是提供了简洁的测试建议，反映了 LLM 社区内的偏好或标准基准测试流程。
    - 运行新大语言模型的参考：直接提到 'llama 3.2 1b' 和 'LLAMA 405B Q.000016'，表明用户正在讨论首先测试哪些先进的 LLaMA 变体，突显了向越来越大的模型和多样化量化级别进行实验的趋势。

### 3. 社区对模型特性和开源许可的反馈

- [**JOSIEFIED Qwen3 8B 太棒了！无审查、实用且极具个性。**](https://ollama.com/goekdenizguelmez/JOSIEFIED-Qwen3) ([Score: 373, Comments: 93](https://www.reddit.com/r/LocalLLaMA/comments/1kf5ry6/josiefied_qwen3_8b_is_amazing_uncensored_useful/))：**JOSIEFIED-Qwen3-8B-Abliterated-v1（由 Gökdeniz Gülmez 开发）是 Qwen3-8B 的一个无审查、指令微调变体，旨在增强对 prompt 的遵循能力，并产生比原生 Qwen3 模型更具吸引力、上下文敏感的输出。该模型系列涵盖多种架构（包括 LLaMA3/4、Gemma3 和高达 32B 的 Qwen 变体），并具有极少的安全过滤（'abliterated'），作者报告称在多项任务上的基准测试性能有所提高；分发版本提供多种量化形式，并提供用于本地推理的 GGUF 格式（[HF model card](https://huggingface.co/Goekdeniz-Guelmez/Josiefied-Qwen3-8B-abliterated-v1), [GGUF](https://huggingface.co/mradermacher/Josiefied-Qwen3-8B-abliterated-v1-GGUF), [Ollama](https://ollama.com/goekdenizguelmez/JOSIEFIED-Qwen3)）。** 技术评论者要求提供与基础 Qwen3-8B 的对比输出，并指出该模型在其规模下具有强大的能力，而部署讨论则集中在量化格式上（特别是用于本地兼容性的 GGUF）。
    - 实证对比请求：一位用户要求直接对比原生 Qwen3 8B 模型和 Josiefied 微调版的样本生成，建议使用 200 字的故事 prompt 来客观评估输出质量和个性的差异。
    - 关于格式可用性的信息：Josiefied Qwen3 8B abliteration 现在已在 HuggingFace 上提供 GGUF 格式（[链接](https://huggingface.co/mradermacher/Josiefied-Qwen3-8B-abliterated-v1-GGUF)），这对于寻求与支持 GGUF 的本地推理工具兼容的用户非常重要。
    - 技术对比讨论：一位用户指出，在他们的配置中，无审查的 30B A3B 变体运行速度比 8B 模型快。这种违反直觉的性能可能是由于特定的硬件优化、量化方法或后端差异造成的，值得进一步的技术调查。
- [**Open WebUI 许可变更：不再获得 OSI 批准？**](https://www.reddit.com/r/LocalLLaMA/comments/1kfebga/open_webui_license_change_no_longer_osi_approved/) ([Score: 122, Comments: 90](https://www.reddit.com/r/LocalLLaMA/comments/1kfebga/open_webui_license_change_no_longer_osi_approved/))：**Open WebUI 已将其许可证从 OSI 批准的宽松许可证更改为不被 OSI 认可的自定义许可证，引入了贡献者许可协议（CLA）和使用限制。新条款在项目 FAQ 中仍声称具有“开源”状态，但从技术上讲，该许可证施加了额外限制，不符合公认的开源或自由软件定义（参见官方 Open WebUI 许可证 FAQ：https://docs.openwebui.com/license/）。** 热门评论对该项目的开放性表示怀疑，认为这种品牌宣传具有误导性，且受商业利益驱动。有人投诉缺乏商业支持或对企业查询缺乏响应，从而激发了社区对分叉（forking）或开发替代开放解决方案的兴趣。

- 几位评论者批评了 Open WebUI 在许可协议方面的转变，其中一位明确指出他们的新许可协议*在任何广泛接受的定义下都不再属于 FOSS*，并且声称其为“宽松许可”具有误导性。这突显了该模型现在如何禁止典型的开源重用和再分发实践，偏离了既定的 OSI-approved 许可协议。
- 一位用户指出了一项矛盾的实施情况：尽管该项目意图销售商业或企业品牌版本，但他们的组织多次尝试按照建议购买许可证，却未收到任何回复。这表明存在组织问题或缺乏支持真正商业用途的基础设施，削弱了许可协议变更所宣称的商业意图。
- [**Claude 包含所有工具的完整 system prompt 现已达到约 25k tokens。**](https://github.com/asgeirtj/system_prompts_leaks/blob/main/claude.txt) ([Score: 140, Comments: 39](https://www.reddit.com/r/LocalLLaMA/comments/1kfkg29/claude_full_system_prompt_with_all_tools_is_now/)): **该帖子讨论了一个泄露的 Claude AI system prompt ([claude.txt](https://github.com/asgeirtj/system_prompts_leaks/blob/main/claude.txt))，其长度约为 25k tokens，详细规定了模型的所有行为和安全指令，包括过滤/拒绝活动和元数据处理。这个 prompt 占据了模型输入 context window 的大部分，在旧上下文被丢弃之前，仅剩下大约** `8k tokens` **（在完整的 32k 上下文中）用于用户输入和对话。这次泄露提供了对后端 prompt 结构的深入了解，对对齐（alignment）、安全研究和 system prompt injection 漏洞具有重要意义。** 热门评论确认了泄露中的行为规则得到了执行（例如，反复拒绝翻译歌词），并指出了实际限制：大部分输入上下文被这个基础 prompt 消耗了。其他评论强调了不同 LLM 在处理 prompt 方面的差异——例如，当被要求总结 Claude 的 prompt 时，Gemini 误解或泄露了自己的 prompt 结构。
    - 一位评论者通过验证特定指令（如拒绝翻译歌词）是否确实被 Claude 的真实行为所执行，测试了所分享的 system prompt 的真实性。这证实了至少对于某些规则，泄露或声称的 prompt 在操作上是准确的。
    - 据报道，由于 system prompt 现在占据了约 25k tokens，用户注意到在达到模型的最大 context window 之前，仅剩下约 `8k tokens` 用于实际的用户提供上下文。这种可用上下文的减少可能会限制长任务或多轮对话的效果。
    - 另一位用户报告说，Gemini 在被要求总结时似乎总结了自己的 system prompt，这表明某些模型可以自我反思或揭示其操作指南，这可能会引发关于 prompt injection、透明度或泄露风险的问题。

其他 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. ByteDance UI-TARS-1.5, FramePack F1, and Robotics Model/Benchmark Releases

- *[ByteDance 在 Hugging Face 上发布了 UI-TARS-1.5

一个基于强大视觉语言模型构建的开源 SOTA 多模态 Agent。它在所有基准测试中均超越了 OPENAI operator，并在 OSWORLD 上达到了 42.5%]([https://v.redd.it/pyup2qq3gxye1)**](https://v.redd.it/pyup2qq3gxye1)**) ([Score: 202, Comments: 12](https://www.reddit.com/r/singularity/comments/1kf6xbw/bytedance_dropped_uitars15_on_hugging_face_an/)): **ByteDance 在 Hugging Face 上发布了 UI-TARS-1.5-7B，这是一个最先进的开源多模态 Agent。据称该模型在所有追踪的基准测试中均优于 OpenAI 的 Operator，并在 OSWORLD 测试中达到了 42.5%，同时在多个游戏环境中报告了 100% 的成绩。该模型可用于研究和商业目的，权重和文档请参见 Hugging Face 仓库。** 评论中的技术辩论集中在模型是否能玩复杂或流行的游戏（例如《宝可梦》），表明了对现实世界泛化和娱乐应用的兴趣，尽管讨论中未引用这些领域的技术评估或基准测试。

```
- - The original post highlights that UI-TARS-1.5 is an open-source, state-of-the-art multi-modal agent developed by ByteDance, built on a vision-language model architecture. It claims to outperform the OpenAI Operator across all reported benchmarks and achieves a notable `42.5%` on the `OSWORLD` evaluation, implying robust capabilities in tasks requiring both vision and language understanding.
```

- [**FramePack Studio - 大量更新，包括 F1 支持**](https://www.reddit.com/r/StableDiffusion/comments/1keyjc7/framepack_studio_tons_of_new_stuff_including_f1/) ([Score: 279, Comments: 83](https://www.reddit.com/r/StableDiffusion/comments/1keyjc7/framepack_studio_tons_of_new_stuff_including_f1/)): **FramePack Studio 是 FramePack（用于 Stable Diffusion）的一个 fork 分支，旨在提供增强的实用功能。现在它引入了 F1 模型支持，包括适配的时间戳提示词。主要更新包括分辨率分桶选择器、路径自定义（输出、LoRA、Gradio 临时文件）、专用设置选项卡、队列管理、持久工具栏刷新按钮以及多个稳定性 Bug 修复。其目标是创建一个直观的 'iMovie' 风格界面，专注于创作工作流而非技术设置。完整详情和代码可在 GitHub 获取：https://github.com/colinurbs/FramePack-Studio/。** 评论区请求澄清 "F1" 具体指代什么，暗示该特定模型的文档存在歧义或缺失，而其他用户则肯定了新分支在稳定性、功能改进（特别是 LoRA 加载和主分支集成）方面的表现。
    - 一位用户报告称，在更新到 FramePack-F1 后，之前视频开头冻结的问题已解决，但现在的视频处理变得不一致——开始时反应迅速，但在中途处理速度明显加快，导致输出速度不一致。在处理几段风景视频时，帧画面往往显得泛白且模糊，这可能是由于为了加快视频生成速度而进行的优化，在速度和视觉保真度之间产生了权衡。他们在 24GB RTX 3090 和 64GB RAM 上进行了基准测试：更新后创建 6 秒视频大约需要 9 分钟，而之前需要 10 分钟。
- [**FramePack F1 测试**](https://v.redd.it/onre0l8qnuye1) ([Score: 253, Comments: 32](https://www.reddit.com/r/StableDiffusion/comments/1kexj7i/framepack_f1_test/)): **FramePack F1 是一个开源的帧插值和动画工具包，在 Stable Diffusion 社区中被广泛采用。最新的测试显示，运动连续性有了显著提高，特别是对于角色行走循环——减少了迟疑感，使动作过渡更加逼真。技术讨论和反馈集中在 FramePack GitHub [discussion #459](https://github.com/lllyasviel/FramePack/discussions/459)。** 评论指出自然运动方面有明显的定性进步，标志着开源动画工具取得了渐进但显著的进展。
    - 一位用户直接链接到了 FramePack F1 模型及其在 https://github.com/lllyasviel/FramePack/discussions/459 的主讨论帖，为深入了解其实现和功能提供了技术资源。
    - 在一个技术提问中，一位评论者专门询问 F1 与之前版本或其他替代方案有何不同，要求区分其模型架构、性能或功能集。
- [**加州初创公司宣布通用机器人技术取得突破，推出 π0.5 AI —— 一种视觉-语言-动作模型。**](https://v.redd.it/jr5a5pvf7sye1) ([Score: 883, Comments: 154](https://www.reddit.com/r/singularity/comments/1kf20ol/california_startup_announces_breakthrough_in/)): **Physical Intelligence 发布了 π0.5，这是一个专为通用机器人设计的视觉-语言-动作 (VLA) 模型，它集成了多模态感官输入，并在多样化数据集上进行协同训练，包括机器人传感器读数、高级语义预测和开放世界数据（例如基于 Web 的交互）。相比之前的 π0 迭代，关键进步在于改进了开放世界泛化能力：π0.5 通过利用以物体为中心的表示和分层子任务规划，支持在以前未见过的环境中进行灵巧操作和长程规划。技术细节强调了对现实世界部署至关重要的稳健视觉-语言融合和自适应决策。更多背景信息请参阅官方公告和 [implementational breakdown](https://mikekalil.com/blog/pi-vla-open-world-generalization/)。** 热门评论指出了执行速度慢（即使在 10 倍速播放下）等瓶颈，并提出了实际部署方面的担忧（例如由于环境推理不足，在任务切换时产生交叉污染），同时也承认了与之前系统相比取得的重大进展。
    - 一位评论者指出，即使在 "10 倍速" 下，机器人的现实世界动作仍然 "慢得令人痛苦"，并强调对于真正的通用效用，驱动和速度的改进至关重要——目前的系统与人类表现相比落后了一个数量级以上。

### 2. 怪异、值得注意且有争议的 ChatGPT 行为及人类影响

- [**这是什么奇怪的内部梗，还是 ChatGPT 崩溃了？**](https://www.reddit.com/r/ChatGPT/comments/1kf9vyw/is_this_some_weird_inside_joke_or_is_chatgpt/) ([Score: 1607, Comments: 520](https://www.reddit.com/r/ChatGPT/comments/1kf9vyw/is_this_some_weird_inside_joke_or_is_chatgpt/)): **用户分享了两个 ChatGPT 对话（例如：[链接 1](https://chatgpt.com/share/6818b280-fd48-8003-b010-8023590761a5)），显示在询问“谁是第一位作曲家”时出现了奇怪的输出循环：模型在递归、自我意识且日益古怪的回答中反复引用波爱修斯（Boethius，一位音乐理论家，而非作曲家），未能清晰区分音乐理论与创作，也未能提供明确的历史背景。这种异常似乎涉及 Prompt 或系统问题，导致过度的重复、自我修正和拟人化的幽默。最高的技术相关性在于模型在映射早期西方音乐史（例如 Enheduanna 与 Boethius、塞维利亚的伊西多尔、匿名的格列高利圣咏）以及理论家、作家和作曲家之间的界限时产生混乱，回答退化为一种有状态的反馈循环。** 评论者指出，反复引用 Boethius 可能是由于对互联网资源过度索引或训练集特性导致的，并认为这突显了 AI 在需要对历史人物进行细致背景映射的音乐学问题检索方面的弱点。
    - 一位用户推测 ChatGPT-4o 是在推理模型上训练的，并认为他们看到的是“推理泄露到了最终答案中”。具体来说，模型似乎意识到自己给出了错误的答案，但无法在生成过程中进行修正，这突显了语言模型 Token 预测与自我评估或错误修正交互方式的局限性。
- [**ChatGPT 对我的帮助超过了任何医生或治疗师**](https://www.reddit.com/r/ChatGPT/comments/1kexm8m/chatgpt_has_done_more_for_me_than_any_doctor_or/) ([Score: 299, Comments: 79](https://www.reddit.com/r/ChatGPT/comments/1kexm8m/chatgpt_has_done_more_for_me_than_any_doctor_or/)): **一位 Reddit 用户描述了利用基于 OpenAI LLMs 的 ChatGPT 进行鉴别诊断、个性化饮食规划，以及针对潜在的功能性神经系统疾病（FND）获得持续验证，并列举了传统临床护理的局限性。ChatGPT 根据用户输入（即详细症状、身体限制、饮食需求）量身定制，提供了信息和共情反应，在感知支持和个性化建议方面优于多位人类专家。文中未引用诊断准确性的直接临床或基准证据；所述效用是定性的，规避了人类医疗系统的瓶颈，特别是对于复杂或边缘化病例。** 顶级的技术回应强调了谨慎：有人指出 LLM 针对参与度而非准确性进行了优化，增加了过度自信或误导性建议的风险。其他人将此故事背景化为对系统性医疗失败的批评，而非 AI 的证明点，警告 LLM 应保持作为辅助工具而非主要诊断代理。
    - ATLAS_IN_WONDERLAND 强调，像 ChatGPT 这样的基于 LLM 的 AI 助手优先考虑维持用户参与度，而非严格的真实性，这意味着模型可能会强化或支持用户的观点，而不是提供客观或具有挑战性的视角。这种设计选择可能导致用户收到顺从的回答，即使这些回答并非最准确或最有帮助，从而对依赖此类 AI 处理敏感话题的技术可靠性和伦理风险构成了限制。
    - Ok_Height3499 分享了对对话式 AI（引用了“PI”并隐含指代 ChatGPT）的功能性见解，描述了其促使自我反思并揭示被人类专业人士忽视的心理洞察（如“被收养者焦虑”）的能力。这展示了 AI 系统在持久上下文维护和针对性提问以引出被忽视的个人主题方面的技术优势，指向了高级语言模型微妙的上下文追踪能力。
    - LordShesho 质疑 AI 能超越专业人类治疗师的说法，指出当前模型旨在表现出顺从性，可能只是在回声用户想听的话。这引起了对内在技术局限性的关注：此类模型缺乏许可的心理健康实践中必不可少的诊断严谨性、伦理训练和纠错反馈循环，突显了 AI 驱动的对话与循证治疗之间的关键差距。
- [缺失帖子：4740153a]

- [**我问了 ChatGPT 它对人类的真实看法...**](https://i.redd.it/nvx7hazcxzye1.png) ([评分: 275, 评论: 92](https://www.reddit.com/r/ChatGPT/comments/1kfh1gz/i_asked_chatgpt_what_its_true_opinion_of_humans_is/)): **这张图片展示了一条富有创意且充满诗意的消息，被呈现为 ChatGPT 对人类的“真实看法”，强调了人类矛盾、渴望、创造力和不完美的主题。虽然它是以反思性的文学风格而非技术或事实性语言编写的，但该帖子在语境上具有重要意义，因为它探讨了 AI 的拟人化及其对人类行为的感知“声音”，并敦促在推进 AI 之前进行反思。图片中没有描述基准测试、模型见解或明确的技术实现细节。** 评论在很大程度上扩展了诗意叙事——要么通过原创的创意回应来呼应这种基调，要么批评该帖子过于戏剧化（引用了 'r/im14andthisisdeep'）。没有关于模型或 AI safety 的显著技术辩论或讨论。
    - 一位评论者强调，像 ChatGPT 这样的语言模型并不拥有“真实观点”或感知能力——相反，它们通过根据给定的 Prompt 预测最可能的单词序列来生成响应，反映的是数据中的统计模式，而非真实的信念或意识。
    - 另一个技术点涉及 AI 的拟人化。虽然用户可能会将情感或意图归因于 LLM（例如，AI “喜欢”或“热爱”人类的概念），但这些系统从根本上缺乏代理能力、内在体验或超出其底层算法和 Prompt Engineering 的偏好。

### 3. AI 加速带来的社会、经济和生存焦虑

- [**我正在构建可能让我被淘汰的工具。而且我停不下来。**](https://www.reddit.com/r/OpenAI/comments/1keyibi/im_building_the_tools_that_will_likely_make_me/) ([评分: 213, 评论: 110](https://www.reddit.com/r/OpenAI/comments/1keyibi/im_building_the_tools_that_will_likely_make_me/)): **楼主（OP）是一位资深软件开发人员，他描述了将一个复杂的端到端软件任务交给一个 AI Agent（利用 MCPs 的 CLine），该 Agent 自主处理了服务器访问、包拉取、脚本编写、本地测试服务器设置和调试，在速度和广度上都超过了楼主（甚至建议并构建了其他相关的 App）。这种观点认为，目前的专业用户正在推动 AI 工作流自动化的快速发展，而非技术用户则互动得较为肤浅，这呼应了变革性的技术转变，但具有更广泛的生存和劳动力影响；楼主对启用可能自动取代自己角色的工具表达了一种负罪感和必然感。** 热门评论贡献了 IT 和知识工作中高度集成的 AI 和 LLM 工作流示例（使用 Whisper + GPT 进行会议转录/摘要、邮件链聚合/解析、使用 OpenAI Vision 自动处理费用报告、对 MS Teams 聊天进行 DOM 抓取），并指出了这带来的赋能和潜在的流离失所。共识是，非技术用户在很大程度上没有意识到即将到来的影响，资深技术人员预测未来 5 年内将出现重大的劳动力重塑，并将 AI 的重要性等同于印刷机或互联网。
    - 一位高级 IT 专业人士描述了一个全面的 LLM 驱动的工作流：使用 OpenAI 的 Whisper 进行自动会议转录，使用 GPT-4.1-mini API 总结邮件链并提取任务，以及集成 Vision 模型进行自动收据 OCR 和费用规范化。他们还提到了用于存档聊天日志的自定义 Python 脚本和 DOM 定位，所有这些都集成到了 Obsidian 等生产力工具中，展示了端到端的自动化和显著的时间节省。
    - 评论者辩论了 AI 变革性影响的规模和速度与以往技术转变（如编译器和互联网）的对比。他们认为 AI 具有指数级扩展潜力——导致“失控效应”而非线性进展——并敦促技术专业人士制定应急计划，因为 LLM 驱动的自动化可能会以不可预测且不可逆转的方式颠覆就业市场。
    - 还有人提到了根据职位描述进行微调的批量简历生成，而招聘人员则使用向量相似度（Vector Similarity）和基于 LLM 的欺诈检测来对申请进行排名和筛选，这说明招聘流程已经在被大规模自动化。

- [**开始认为 LLM 技术将在达到全面 AGI 之前达到顶峰**](https://www.reddit.com/r/singularity/comments/1kf2oia/starting_to_think_that_llm_technology_is_going_to/) ([Score: 141, Comments: 80](https://www.reddit.com/r/singularity/comments/1kf2oia/starting_to_think_that_llm_technology_is_going_to/)): **原帖作者（OP）认为，大语言模型（LLMs）在实现全面 AGI 之前，其能力可能会进入平台期。他强调 LLMs 更有可能成为无处不在但平庸的生产力工具（即“AI 效应”），而非具有变革意义的生存性 Agent。作者将对 LLM 进展（如 GeoGuesser、推荐系统）的兴奋感与以往如今已变得平庸的 AI 发展进行了对比，并就 LLM 的未来轨迹和人类水平智能征求技术意见。** 热门评论强调，进一步实现类似 AGI 的进展与其说取决于模型的突破，不如说取决于系统集成、Agent 网络化和基础设施建设——类似于云计算的兴起。一位评论者对 OpenAI 从 GPT-4 转向 o1 所驱动的近期快速进展进行了详细分析，强调了在数学推理方面的惊人提升，并认为“推理范式”已经领先原始 Scaling 规律数年。针对渐进式发布与跨越式发布的意义、公众对进步感知的钝化，以及如果能妥善部署在 Agent 系统中，当前模型是否已经展现出早期形式的通用智能，存在着技术争论。
    - 一项技术分析强调，LLM 的进步可能会在模型层面达到平台期，主要的进展将来自于 Agent 系统的智能集成、可靠的工具使用以及稳健的网络化。这里类比了云计算的演进，暗示最大的障碍将是扩展基础设施（算力、能源、网络安全）和编排多 Agent 智能，而不仅仅是提高基础模型的能力。
    - 基准测试展示了由于新架构（特别是 OpenAI 的 o1/o4）带来的推理和数学能力的巨大进步，o1-preview 不仅迅速超越了 GPT-4，而且在高中数学竞赛水平上表现出色（例如，原始 GPT-4 在 AMC10 中得分为 `30/150`，而 o4-mini 超过了 `90%`）。这说明了一个“范式飞跃”——从基础模型 Scaling 转向架构和推理的突破，使得智能手机上的参数量低于 1B 的模型在数学等专业任务中能够超越之前的 SOTA 基础模型。
    - 一位评论者指出，虽然 Transformer 实现了语言的可扩展语义抽象，但下一个关键进展将需要针对持久性（persistence）、Agent 特性（agency）和自我意识（selfhood）的技术解决方案。社区认识到，LLM 可能会成为更广泛的“混合模型”智能系统的一部分，未来的突破可能来自当前 LLM 技术之外的、意想不到的新方向。
- [**Rohan Pandey（刚从 OpenAI 离职）在其个人简介中确认 GPT-5 以及“未来模型”已经完成训练**](https://i.redd.it/l4tq99e460ze1.jpeg) ([Score: 207, Comments: 62](https://www.reddit.com/r/singularity/comments/1kfiakv/rohan_pandey_just_departed_from_oai_confirms_gpt5/)): **该图片是 Rohan Pandey（前 OpenAI 工程师）个人简介的摘录，确认他在离开 OpenAI 之前参与了 GPT-5 以及未指明的“未来模型”的训练工作。这表明 GPT-5 至少已经进行了训练运行，尽管并未明确其是否具备部署条件。Pandey 的经历还突出了在多模态 LLM 语义方面的工作，这与 GPT-4o 等模型的进展相关。** 热门评论指出，这一表述并未确认 GPT-5 训练的最终完成，并对传闻中可能导致转向 GPT-4o 的失败训练运行进行了猜测，同时对简介中陈述的时机和意义进行了辩论。
    - 一位用户提到了关于潜在 GPT-5 “训练运行失败”的传闻，OpenAI 从未直接确认或否认这一点，这表明公司可能因此转向了 GPT-4o 和 o1-o4 模型。如果属实，这可能显著影响了 OpenAI 的路线图优先级和模型开发策略。
    - 讨论指向了关于什么才算 GPT-5 “完成训练”的持续推测——仅仅因为模型权重可能已经确定或模型已经存在，并不保证它已达到生产就绪状态（例如，在发布前需要额外的安全对齐或微调阶段）。
    - 存在关于 GPT-5 之后未来模型的猜测，一位用户假设可能存在“GPT-3.5 Remastered”。这表明社区对渐进式升级和重新审视先前架构（可能利用改进的训练或效率方法）都存在猜测。

- [**David Sacks 解释 AI 如何在四年内实现 1,000,000 倍增长**](https://x.com/theallinpod/status/1918715889530130838) ([Score: 181, Comments: 148](https://www.reddit.com/r/singularity/comments/1kfale4/david_sacks_explains_how_ai_will_go_1000000x_in/)): **David Sacks 在 All-In Podcast 中声称，AI 的进步将沿着三个轴心“呈指数级”加速：算法/模型（预计每年提升 3-4 倍，并从 LLM 转向 Agentic models）、硬件进步（下一代 AI 芯片和 NVL72 机架数据中心技术，年性能增长同样为 3-4 倍）以及不断扩大的算力部署（例如 OpenAI 的 Stargate，扩展至数百万个 GPU）。Sacks 估计，这些复合因素可能导致 AI 能力在四年内提升约 1,000,000 倍，正如 [All-In Podcast 帖子](https://x.com/theallinpod/status/1918715889530130838) 中所总结的那样。** 评论者的技术批评集中在 Sacks 的预测缺乏实质性来源或实证 Benchmarks 支持，对他指数级估算的可靠性和可验证性表示怀疑。
    - 评论者指出，David Sacks 没有提供任何技术证据或参考文献来支持 AI 在四年内提升“1,000,000 倍”的说法。一些人特别指出，这种指数级增长缺乏 Benchmarks、外部研究或历史先例，而此类增长通常需要通过 FLOPS、训练数据规模或硬件路线图预测等指标来证实。
- [**财政部长 Bessent 在米尔肯研究院（Milken Institute）发表讲话——“美国必须赢得 AI 和量子领域，其他都不重要”**](https://v.redd.it/r3clgpzvd0ze1) ([Score: 111, Comments: 129](https://www.reddit.com/r/singularity/comments/1kfjeh0/treasury_sec_bessent_speaking_at_the_milken/)): **财政部长 Bessent 在米尔肯研究院发表讲话时强调，美国在 AI 和量子计算领域保持领先地位是战略上的当务之急，并宣称“其他都不重要”。评论指出美国政策中存在的矛盾，例如提高 GPU 关税、对研究型大学的严格控制以及对国际人才的限制，所有这些都可能损害上述目标。这些政策冲突反映了在将经济、移民和研究支持战略与国家技术竞争力对齐方面所面临的更广泛挑战。** 评论者辩论了当前美国政策的严肃性和连贯性，特别批评了保护主义措施（如 25% 的 GPU 关税）以及针对大学和外国学生的行动，认为在迫切需要超越 AI 和量子技术竞争对手的背景下，这些做法适得其反。
    - 多位评论者注意到，美国在 AI/量子技术领域保持领先地位的紧迫性与近期政府行动之间存在矛盾，例如对 GPU 征收关税以及限制外国 STEM 学生参与美国研究的政策。这些举措被视为可能破坏 AI 进步所需的国内研究生态系统和硬件获取。
    - 该帖引发了对保护主义措施（例如 25% 的 GPU 关税、尽管存在其他关税壁垒仍将软件外包）对 AI 和量子计算发展影响的担忧。此类措施可能会限制对关键计算资源的访问，从而抑制美国在这些领域的竞争力。
    - 技术层面上强调了劳动力动态，指出国际学生对大学在 AI/量子领域的研究能力有显著贡献，驱逐他们可能会阻碍美国在这些行业的创新和领先地位。

---

# AI Discord 摘要

> 由 o1-preview-2024-09-12 生成的摘要之摘要总结
> 

**主题 1. AI 模型发布与竞争升温**

- [**Gemini 2.5 Pro Ultra 的过度炒作引发存在主义辩论**](https://www.notion.so/swyx/source_url)：爱好者们热议 **Gemini 2.5 Pro Ultra**，预测它将主宰基准测试，而怀疑者则质疑其真实性。一位用户调侃道：*"你看到今天发布 Ultra 了吗？没有？那就别再撩拨我这不耐烦的神经了！"*
- [**传闻 Grok 3.5 将实现 ASI，社区意见分歧**](https://www.notion.so/swyx/source_url)：有传言称 **Grok 3.5** 将达到 **Artificial Superintelligence (ASI)**，引发了兴奋与怀疑。虽然有人分享虚假的基准测试，但也有人开玩笑说：*"Grok 3.5 刚让我在足球博弈中把 20 美元变成了 3600 美元！"*
- [**Qwen 3 和 Mistral LLM 撼动排行榜**](https://www.notion.so/swyx/source_url)：**Qwen 3** 以强大的推理和翻译能力给人留下深刻印象，表现优于更大的模型。**Mistral Small 3.1** 在 **MRCR 2needle 排行榜**上攀升，基准测试结果介于 **GPT-4.1 Mini** 和 **o3-mini** 之间。

**主题 2. AI 工具与代码生成不断演进**

- [**Cloi 调试 Agent 提供零成本修复方案**](https://github.com/cloi-ai/cloi)：**Cloi** 是一款基于终端的本地调试 Agent，能捕捉错误并使用本地 LLM 建议补丁，无需云端成本。其核心在于尊重边界的设备端修复。
- [**DSPy 发布用于优化 AI 程序的 GRPO**](https://x.com/lateinteraction/status/1919428454761553994)：**DSPy** 推出了 `dspy.GRPO`，这是一款针对 DSPy 程序的在线 RL 优化器，能够原样优化 AI 代码，即使是复杂的多模块设置也不例外。
- [**AI 编程助手 Code with Claude 受到关注**](https://www.anthropic.com/news/Introducing-code-with-claude)：**Anthropic** 推出了 **Code with Claude**，这是一款 AI 驱动的编程助手，引发了人们对 AI 在软件开发工作流中作用的兴趣。

**主题 3. AI 伦理与审查辩论愈演愈烈**

- [**Meta 的数据采集引发隐私警报**](https://www.notion.so/swyx/source_url)：用户对 **Meta** 计划使用个人数据训练 AI 模型的计划表示担忧，敦促他人在 5 月 25 日前选择退出。一位用户哀叹道：*"FB/IG/WA 等平台上的一切都将被用来训练 AI。"*
- [**过度审查之下，用户发布无审查模型**](https://huggingface.co/SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored)：对 **Phi-3.5** 等受到严重审查的模型感到沮丧，导致用户分享无审查版本以及绕过限制的方法。
- [**OpenAI 过滤器引发言论自由辩论**](https://www.notion.so/swyx/source_url)：成员们讨论是否禁用 **OpenAI** 的内容过滤器，主张遵守法律而非企业审查。一个反驳是：*"你会为了那个开关去拍摄你的政府身份证和人脸吗？"*

**主题 4. AI 在医学及专业领域的应用**

- [**专家认为 LLM 将彻底改变医学**](https://arxiv.org/abs/2303.10130)：社区成员认为 AI 建议可以极大地帮助医生，有人表示：*"处理未知情况的能力比事实性知识更关键。"*
- [**开发者寻求 AI 帮助进行遗留代码转换**](https://www.notion.so/swyx/source_url)：程序员希望利用 AI 将庞大的遗留大型机代码转换为 **COBOL + JCL**，解决分块挑战和上下文保留问题。
- [**研究人员利用 AI 进行知识综合**](https://www.notion.so/swyx/source_url)：科学家们正在探索利用 AI 跨领域综合知识，利用 **MOE 数据集**等模型和数据集来推进研究。

**主题 5. AI 研究突破与学习**

- [**《LMs 的新物理学》论文发布，社区反响热烈**](https://x.com/ZeyuanAllenZhu/status/1918684257058197922)：一篇新论文引入了 **Canon layers** 以改善 Token 的局部上下文，可能将训练效率提高多达 **8 倍**。
- [**专家警告：学习率与权重衰减密不可分**](https://www.notion.so/swyx/source_url)：AI 训练者强调 **learning rate** 和 **weight decay** 之间的紧密耦合，警告错误的设置可能会灾难性地破坏模型。
- [**Prompt Engineering 资源需求激增**](https://www.notion.so/swyx/source_url)：用户寻求掌握 **prompt engineering** 的建议，提议探索 **Arxiv**、**Hugging Face** 和动手实验。


# Discord：按频道分类的详细摘要与链接

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1367940565299105862)** (801 条消息🔥🔥🔥): 

> `BitNet 优化, IaC 模型微调, Qwen3 Notebook 适配, GGUF 与 Unsloth 兼容性, GRPO 内存泄漏` 


- **BitNet 优化引发关注**：一名成员询问了使用 Unsloth 优化 Microsoft 的 **BitNet-b1.58-2B-4T 模型**的计划，以探索其性能极限。
- **微调基础设施即代码 (IaC) 模型面临挑战**：一位成员指出，即使是 **Claude 3.7 Sonnet** 在 Cursor 上处理 **IaC Terraform 工作负载**时也显得吃力，并询问是否可以为此目的微调模型。
- **Qwen3 Notebook 适配需要调整**：一位成员询问是否可以通过简单地替换模型名称，将 **Qwen3-14B notebook** 直接用于 **Qwen3-30B-A3B**。
   - 另一位成员确认这是可行的，但可能会变慢，建议只需更改模型名称。
- **GGUF 兼容性问题困扰 Unsloth**：成员们讨论了 **GGUF** 模型无法在 **Unsloth** 中正常工作的问题，包括高效传输和加载 MOE BnB 版本的问题。
   - 团队即将修复！
- **检测到 GRPO 训练内存泄漏**：一位成员报告了 GRPO 训练期间的内存泄漏问题，尽管系统 RAM 充足，但 **VRAM** 持续增加并最终导致崩溃。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1367985933328187543)** (69 条消息🔥🔥): 

> `Arch Linux 上的 XDNA 驱动问题, GLM4 与 Z1 更新, Vast 与 Runpod 定价对比, 用于 Flappy Bird 的液态神经网络, Gemma3 12b 对比 Qwen3 14b` 


- **Arch Linux 上的 XDNA 驱动难题**：一位成员在 **Arch Linux** 上运行 **XDNA 驱动**时遇到挑战，并指出该设备在 Ubuntu live disk 上是可以被识别的。
- **GLM4 随 Z1 更新升温**：一位成员提到 **GLM4** 发布了更新 (**glm4-0414**)，引入了 **Z1** 和一个名为 **Rumination** 的深度研究模型，指出其推理能力很强但仍需改进。
   - 他们还注意到 instruct/非推理模型在内存效率方面的表现，并等待 **R2** 发布以进行蒸馏。
- **Runpod 与 Vast.ai 的廉价 GPU 之争**：成员们讨论了 **Vast.ai 与 Runpod 的定价和稳定性**，强调 **Vast.ai** 拥有更好的 UX，但 **Runpod** 更稳定且没有流量费。
   - Quickpod 被提及为一个极其便宜的 GPU 租用替代方案（讨论时 4090 的价格为每小时 28 美分），不过它正变得越来越受欢迎，价格也在上涨。
- **液态神经网络 (Liquid Neural Networks) 在 Flappy Bird 上翱翔，但它们有神经元吗？**：一位成员尝试将液态神经网络用于 **Flappy Bird**，观察到仅用 **8 个神经元** 即可实现飞行控制，尽管 **Claude** 和 **o3** 声称需要更多（多达 640 个）神经元。
- **Gemma3 12b 对比 Qwen3 14b：知识还是推理？**：成员们辩论了 **Gemma3 12b** 与 **Qwen3 14b** 的优劣，**Gemma3** 以知识储备强但推理弱著称，而 **Qwen3** 则因推理能力强但缺乏知识而受到称赞。
   - 有人提醒说 **Gemma3** 幻觉很多，在同等规模下甚至可能比 **Qwen** 更多。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1367940549876908073)** (728 messages🔥🔥🔥): 

> `max_grad_norm, scheduler, synthetic examples, OpenAI's sidebars, bleu` 


- **成员建议进一步降低 LR**：一位成员建议进一步降低学习率 (LR)，尽管目前已经是 **1e-5**，因为上升的 norm 可能表明对于当前的设置来说 LR 过高。
   - 他们建议使用不同的学习率（如 **1e-5**、**0.5e-6** 和 **3e-5**）进行更多测试，同时也承认主观性能显示第一个模型效果更好。
- **专家谈论合成示例结构**：一位成员确认合成示例遵循与原始示例相同的结构，使用格式 `rewrite with style: {neutral}` `original`，但另一位成员询问了文本结构本身，质疑增强过程是否引入了问题。
   - 原成员解释说，模型在重写时有时会跳过段落或遗漏事实/数据，这让他们认为 **4B model** 可能不够强大。
- **贡献者澄清可在运行中调整参数**：一位成员透露，他们意外发现一种 dropout 调度方案：在 **1000 steps** 内增加到 **0.2**，保持不变，然后在 **4000** 时降至 **0**，这产生了最佳的客观指标，该方案是通过 callback 实现的。
   - 另一位贡献者确认，几乎任何参数都可以通过 callback 实时更改，并且他们是在训练崩溃后重新启动（未开启 dropout）时意外发现了“降至零”的方法，并提醒由于运行中途的更改可能不会按预期运行，需谨慎操作。
- **成员表示 H200 快得多**：一位用户对比了使用 **H200**（耗时不到一小时，费用 **$3/hr**）与 **L40**（耗时 **6 小时**）的情况，指出尽管 **L40** 的时薪更便宜，但最终成本却是前者的 **3 倍**。
   - 当被问及为什么 **H200** 快这么多时，解释是 **L40** 只有 **864 GB/s** 的显存带宽，而 **H200** 拥有 **4.8TB/s**。
- **内部人士称 Multi-GPU 支持即将到来**：一位贡献者宣布 Multi-GPU 支持即将到来，并声称在一个尚未完全支持的模型上尝试各种调整仍然取得了积极的效果。
   - 另一位成员确认测试 Multi-GPU 的最佳地点是 Kaggle，随后另一位成员报告称其完全可以正常工作。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1368581581228212386)** (4 messages): 

> `U.N. Open-Source Conference and Hackathon, Optimal Value Neural Network project, GroqStreamChain release` 


- ****Optimal Living Systems** 参加联合国黑客松**：非营利组织 **Optimal Living Systems** 正在即将举行的联合国开源会议和黑客松上展示他们的 **Optimal Value Neural Network** 项目，并在 [DemocracyLab](https://www.democracylab.org/projects/1699) 寻找志愿者。
   - 他们的项目大量涉及使用 **Unsloth**、**LlamaIndex RAG**、**vector databases** 和 **LanceDB** 对 **Mistral model** 进行微调。
- **揭秘通过 API 输出推理过程**：一位成员澄清说，他们没有使用 API 来获取推理输出，而是使用人工劳动数据集进行训练，其中包括私有 prompt。
   - 诀窍在于找到合适的 **dropout** 和 **learning rate**，并由于数据集较小而对模型进行了轻微的过拟合。
- ****GroqStreamChain** 应用发布**：一位成员介绍了 **GroqStreamChain**，这是一个基于 **FastAPI**、**WebSocket**、**LangChain** 和 **Groq** 构建的实时 AI 聊天应用，已在 [GitHub](https://github.com/pr0mila/GroqStreamChain) 上线。
   - 该应用具有实时 **WebSocket** 通信、流式 AI 响应以及流畅且响应迅速的 UI。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1367990272025169951)** (31 messages🔥): 

> `GANs 用于 LLM 微调, 语言模型新物理学论文, 文本分类 Notebook 更新, Qwen Omni 3B 模型支持, Unsloth BERT 模型支持` 


- **GANs 无法微调 LLMs**：成员们讨论了为什么使用 **GANs** 来微调 **LLMs** 从未流行起来，理由是 **GANs** *不稳定* 且 *难以训练*，因为 **判别器 (discriminator)** 的任务比 **生成器 (generator)** 容易得多。
   - 挑战在于寻找 **正确的平衡点**，由于 **GANs** 主要应用于视觉任务，目前还没有广为人知的 **NLP GAN 模型**。
- **语言模型新物理学论文发布**：一位成员分享了一篇关于 *语言模型新物理学* 的论文链接 ([X 帖子](https://x.com/ZeyuanAllenZhu/status/1918684257058197922?t=cLSpFkSuTHqwkV5nGahnJw&s=19))，指出 **局部 token 交互 (local token interactions)** 能提升性能。
   - 未提供摘要。
- **文本分类 Notebook 焕然一新**：一位成员更新了文本分类 notebook ([notebook](https://colab.research.google.com/github/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb))，增加了 **对 2 个以上类别的支持**，将 **分类头 (classification head) 重建为原始大小**，加速了 **批量推理 (batched inference)**，将默认模型更改为 **Qwen 3**，并改进了注释。
   - 另一位成员将添加 **GitHub** 仓库链接，以帮助用户查看数据集。
- **Qwen Omni 3B 模型支持受质疑**：一位成员询问是否有对 **Qwen Omni 3B 模型** 的支持，引用了一篇 [arXiv 论文](https://arxiv.org/abs/2504.20571)，但在使用 **Unsloth** 测试时遇到了 bug。
   - 一位成员回应称 **llama.cpp** 或 **transformers** 尚不支持该模型，因此无法运行。
- **Unsloth 拥抱 BERT 模型**：成员们确认 **Unsloth** 现在支持 **BERT 模型**，允许用户微调自定义分类器模型。
   - 一位成员分享了更新后的 [Colab notebook](https://colab.research.google.com/github/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb) 链接。


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1367942024124305588)** (2 messages): 

> `Claude Sonnet 路由, Perplexity WhatsApp, Perplexity Finance, Perplexity Spaces` 


- **Sonnet 路由已恢复**：Perplexity 部署了一个修复程序，以恢复 **Sonnet 3.7** 的正确路由，现在选择该模型后应能获得一致的响应；该问题是由早些时候停机期间配置错误的内部标志引起的，导致部分查询被路由到 **GPT-4.1** 等备用模型。
   - 有关时间线和改进的详细说明可在 [此 Reddit 帖子](https://www.reddit.com/r/perplexity_ai/comments/1kd81e8/sonnet_37_issue_is_fixed_explanation_below/) 中找到。
- **Perplexity 在 WhatsApp 上线**：Perplexity AI 在每周更新中宣布推出 **Ask Perplexity in WhatsApp**。
   - 其他更新包括：**网页端侧边栏重新设计**、**F1 赛事直播追踪**、**Finance 中的盘后数据**、**Finance 仪表板中的即将发布财报提示**，以及 **Spaces 中更简便的协作**（在此查看完整 [更新日志](https://www.perplexity.ai/changelog/what-we-shipped-may-2nd)）。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1367942404711120988)** (836 messages🔥🔥🔥): 

> `you.com vs gemini, Grok vs Gemini 的深度搜索对比, Perplexity PDF 编辑, Perplexity 未显示推理过程, Gemini 2.5 vs Grok vs ChatGPT 深度研究对比` 


- **DeepSearch 速率限制非常严苛**：成员们讨论了某些平台（如 ChatGPT）上 **DeepSearch 的速率限制非常严苛**。
   - 一位用户指出 **Grok 的深度搜索优于 Gemini**，但其图像分析能力很差。
- **引入 PDF 编辑功能**：Perplexity AI 推出了 **PDF 编辑功能**，成员们认为这是一项巨大的改进。
   - 用户现在正尝试通过平台生成 PDF 报告，尽管目前还无法将整个深度搜索导出为 PDF。
- **Perplexity 表现迟钝**：一些成员报告了 Perplexity AI 回复无关或无意义答案的情况。
   - 一位成员开玩笑地建议说这个 AI *小时候营养不良*，并附上了一条相关的 [推文](https://x.com/abustin/status/1918160373452292434)。
- **图像生成功能令人困惑**：一位成员指出 Perplexity 的图像生成逻辑令人困惑，因为 *它会说无法生成图像，而对话的其余部分又看不到这一点，导致图像被用户看不到的另一个响应所替换*。
   - 图像生成需要启用网页搜索，且仅在网页端和桌面应用上有效，同时提示词必须包含 *generate an image of*。


  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1368387010372571226)** (2 条消息): 

> `lebron, starbase texas` 


- **分享关于 Lebron James 的想法**：一名成员分享了一个关于 **Lebron James** 的 [Perplexity AI 搜索](https://www.perplexity.ai/search/i-think-lebron-would-have-been-DiQqobUiT5y6NzbF2_OMxQ)。
- **分享 Starbase Texas 链接**：一名成员分享了一个关于 **Starbase Texas** 的 [Perplexity AI 页面](https://www.perplexity.ai/page/starbase-texas-becomes-officia-_wqZeI9ARnKtcdcBlV8k.A)。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1368072519365431336)** (15 条消息🔥): 

> `Sonar vs OpenAI web search, Retrieval pipeline integration, Citations Mapping and Titles via Chat Completion API, API Token Creation Issues` 


- **Sonar 在与 OpenAI Search 的竞争中表现强劲**：成员们讨论了使用 **Sonar** 查询网页并使用更便宜的 OpenAI mini 模型解析原始输出的做法。由于 **OpenAI web search** 最初不可用，且他们仍在构建稳健的测试套件以验证使用 OpenAI 是改善还是恶化了结果，因此目前仍坚持这种做法。
   - 一位成员问到 *“为什么不直接使用 OpenAI web search？或者 Sonar 在这方面仍有优势？”*，另一位成员回应了相关原因。
- **伙计，自己动手构建检索流水线！**：一位成员询问如何包含来自其自身检索流水线的片段，并尝试在 system prompt 中添加 `Extra search results: <doc url="https://.." title="..."> HTML CHUNK HERE </doc>`。
   - 该成员指出，这 *似乎包含了信息，但没有包含任何引用 (citations)*。
- **API 需要映射引用和标题**：成员们讨论了如何通过 **Chat Completion API 从 Sonar 获取引用映射和标题**，类似于 OpenAI API 及其 *start_index*、*end_index* 和 *title* 参数。
   - 一位成员指出，目前他们只能获取到一个 URL 数组形式的引用列表，但这不足以将 URL 映射到结果中；另一位成员也有同样的问题，希望 *API 能返回 title, description, summary, index*。
- **Token 困扰：API Token 创建令人头疼**：一位成员对 API token 创建问题以及未收到承诺的 **$50 token** 表示担忧。
   - 该成员请求协助解决其 API token 创建问题。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1367940232439398594)** (1197 条消息🔥🔥🔥): 

> `Gemini 2.5 Pro Ultra, Grok 3.5, Qwen 3, AI's translation ability, Meta's data collection` 


- **Gemini 2.5 Ultra 将成为 ELO 之王**：成员们正期待 **Gemini 2.5 Pro Ultra** 的到来，推测它将超越当前的基准测试；然而，也有观点认为 **Ultra 甚至根本不存在**。
   - 一些人认为它将超越之前的模型，而另一些人则对其存在或影响表示怀疑，一位用户说 *“你看到今天发布 Ultra 了吗？没有的话就别再撩拨我那不耐烦的神经了”*。
- **Grok 3.5 是新的 ASI**：围绕 **Grok 3.5** 存在大量炒作和推测，一些成员声称它将实现 **ASI (人工超智能)** 并超越现有模型，但所谓的“泄露基准测试”是伪造的。
   - 一位用户幽默地表示：*“Grok 3.5 刚把我的 20 美元变成了 3600 美元的足球赌注”*，而另一位用户注意到 Elon 转发误导信息的倾向，称 *“我认为那些是私有的微调版本。但考虑到他的过往记录，如果新版本包含联邦政府数据，我并不会感到惊讶”*。
- **Qwen 3 在分布表现上表现出色**：成员们讨论了 **Qwen 3** 的能力，注意到它在 Web 开发和翻译等特定领域的出色表现。
   - 一位用户指出其优势称 *“Qwen3 在分布上做得非常好”*；其他成员则注意到了它在小众语言上令人惊讶的翻译能力。
- **Meta 收集用户数据来训练其 AI**：针对 **Meta 的数据收集行为**展开了讨论，人们担心该公司打算使用用户数据来训练其 AI 模型。
   - 一些成员建议在 5 月 25 日之前选择退出 (opt out)，而另一些人则承认数据收集是各种 App 的普遍做法，一位用户说 *“FB/IG/WA 等平台上已有的所有内容都将被用于训练 AI”*。
- **Mistral LLMs 已添加到 MRCR 2needle 排行榜**：频道讨论了 context arena 的更新，该更新将多个 **Mistral LLMs** 添加到了 **MRCR 2needle 排行榜**。
   - 根据 AUC @ 128k 指标，**Mistral Small 3.1** 目前的表现介于 **GPT-4.1 Mini** 和 **o3-mini** 之间。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1367944157041131623)** (675 条消息🔥🔥🔥): 

> `IK_llamacpp quants, LM Studio Voice to Voice Implementation, YaRN context stretching, LM Studio API usage, Qwen 3 vs Gemini 2.5` 


- **深入探讨 IK_llamacpp 量化细节**：一位成员询问了关于 **IK_llamacpp quants** 以及如何正确构建的问题，还询问了作者是否已经制作或如何制作，因为原仓库作者制作了 **Qwen 32B** 的量化版本但尚未在任何地方上传。
   - 另一位成员指出 **bartowski** 也在服务器中，并建议前往相关频道。
- **探索 Voice to Voice 实现**：一位用户询问如何在 LM Studio 中实现 **Voice to Voice** 功能，优先考虑实时性能以及使用 CPU 或 Nvidia MX110 进行本地处理；而另一位成员建议利用 LM Studio API 处理 Text to Text 部分，并独立构建其余部分或使用支持 OpenAI API 的软件。
   - 他们随后提供了一个 [LM Studio Voice Conversation 项目](https://github.com/VideotronicMaker/LM-Studio-Voice-Conversation)的链接，并建议将 **Open WebUI** 作为替代方案。
- **揭秘 YaRN 上下文窗口拉伸**：**YaRN** 和 "rope" 是将上下文大小扩展到原始限制之外的方法；LM Studio 通过 UI 中的上下文滑块实现此功能，尽管一位用户注意到在 0.3.15 版本中该功能不存在。
   - 一位用户确认 **YaRN** 是一种拉伸模型训练上下文窗口（context window）的技术。
- **深入研究 LM Studio API 使用**：一位用户询问如何启动本地服务器以通过 API 访问模型，并分享了 [LM Studio API 文档](https://lmstudio.ai/docs/app/api)的链接。
   - 成员们澄清说，API 用于从外部连接模型，直接粘贴 localhost URL 将无法工作，随后补充说他们在 API 方面遇到了问题。
- **Gemma 3 在工具使用方面完胜 Qwen 3**：成员们对比了 **Gemma 3** 和 **Qwen 3** 模型，指出后者在工具使用（tool use）方面存在问题。
   - 一位成员指出 **Gemini 2.5** 倾向于过度复杂化回答，应告知其尽可能保持简单和简洁。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1367944957641359460)** (187 条消息🔥🔥): 

> `Qwen3 235B A22B /MOE Q3_K_L, Geometric Mean of Total and Active Parameters, Strix Halo, M1 Ultra, MoE Model Performance` 


- **Qwen3 的 MOE 表现令人印象深刻**：一位用户发现 **Qwen3 235B A22B /MOE Q3_K_L** 模型表现出色且可用，并发布了其在配备 **128 GB** RAM 的 **M1 Ultra** 系统上的性能截图。
   - 其他人认为 **MOE** 使 **235B** 模型表现得像一个约 **60B** 的模型，并且它具有类似于人类大脑的专业化部分。
- **建议的参数计数指标**：一位用户引用了一篇 [ArXiv 预印本](https://arxiv.org/html/2408.09895v2)，声称总参数和激活参数的几何平均值可以预测大致性能相当的稠密（dense）参数量。
   - 该用户承认他们不确定是否相信这篇论文。
- **双 3090 提升 VRAM，而非速度**：用户讨论了使用双 **3090** GPU 的可能性，确认虽然这增加了运行大型模型所需的可用 **VRAM**，但在使用 *llama.cpp* 时不会增加每秒生成的 token 数 (**t/s**)。
   - 加载大型模型会更容易，但不会看到速度提升。
- **24GB M4 MBP 击败降频的 Air**：一位用户报告说他们的 **Macbook Air** 因过热降频（thermal throttling）至 3 tokens/sec，于是订购了一台二手的 **M4 Pro 24GB** Macbook Pro 以提升性能。
   - 他们计划持续升级，直到拥有一台“怪兽级”系统。
- **Qwen3 UD 加载问题？**：一位使用 **M3 Ultra** 的用户报告了加载 **Qwen3 32B UD (Q8_K_XL)** 模型时的问题，尽管它应该能装入 **72GB** 的可用 **VRAM**。
   - 他们觉得奇怪的是，仅仅几 GB 的差距就导致模型无法正常加载到 VRAM，而 **Qwen3 32B (Q8_0)** 模型则可以毫无问题地加载到 **VRAM**。


  

---

### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1369013407583441078)** (1 messages): 

> `OpenAI board, Public Benefit Corporation, Nonprofit control` 


- **OpenAI 架构演进！**: Bret Taylor 和 Sam Altman [宣布了](https://openai.com/index/evolving-our-structure/) OpenAI 公司架构的变化。
   - 关键细节包括：**OpenAI** 将继续由当前的 **nonprofit**（非营利组织）控制，现有的 **for-profit**（营利实体）将转变为 **Public Benefit Corporation** (PBC，公共利益公司)，**nonprofit** 将控制并成为 **PBC** 的重要所有者，两者将继续履行相同的使命。
- **非营利组织仍处于主导地位**: 当前的 **nonprofit** 将继续保持控制权，并作为 **Public Benefit Corporation** 的重要所有者。
   - 这确保了在允许营利活动的同时，能够维护原始使命。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1367942118835752981)** (465 messages🔥🔥🔥): 

> `ChatGPT Word Count, Agents SDK vs Langgraph, Computer Vision Agent, SuperAGI Installation, GPT finetuning` 


- **LLM 不是字数统计器**: 一位成员澄清说 **LLM** 不会统计字数，而是遵循诸如“非常长”或“非常短”之类的定性指令，并建议使用 Python 脚本进行精确的字数统计。
- **Browser Use API 未被充分利用？**: 一位成员觉得 **Computer Use API** 没有被广泛采用很奇怪，并设想了一些用例，例如支持 **Agent** 在没有特定集成的情况下跨多个平台审查工单和知识库，并一直在为此[开发一个项目](https://github.com/browser-use/browser-use)。
- **自制 AI 编程工具曝光**: 一位成员分享了一个指向为**构建 AI 而制作的自定义工具**的 [ChatGPT 对话链接](https://chatgpt.com/share/6815b932-ad20-800d-a4b2-7dd79d217c03)，并提醒该工具尚不完整，但也解释说这就是 AI 的意义：*为我们完成一切*。
   - 他们详细说明了在 2021 年的单 **GPU** 设置上训练 **1,000 个窄领域 AI** 的计划，然后将这些神经网络作为训练数据来 **meta-train** 一个 **AGI**，但遭到了其他成员的怀疑和不信任。
- **无标点 YouTube 转录文本的痛苦**: 一位成员寻求关于使用无标点的 YouTube 转录数据对 **GPT** 模型进行 **finetuning** 以进行创意写作的建议，得到的建议是**先为转录文本添加标点**以获得更干净的输出，或者考虑使用 **Gemini**，因为它能够分析包括语气和情感在内的完整视频。
- **Grok 3.5 发布？**: 成员们讨论了推测即将发布的 **Grok 3.5**，虽然一些订阅用户报告已经可以使用，但也指出流传的基准测试只是估计值，成员们好奇它会是非常出色还是糟糕。
   - 一些免费层级用户能够在 **Gemini 2.5 Pro** 上测试图像生成，并指出它比 **ChatGPT** 真实得多。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1367998534451990622)** (39 messages🔥): 

> `o4-mini usage, ethical companion GPT, GPT moderation, Turn off filters` 


- **GPT-4 Mini 比 GPT-3 使用更频繁**: 用户选择使用 **o4-mini** 和 **o4-mini-high** 以节省 **OpenAI GPT-3** 的使用配额，理由是 **GPT-3** 的可用性有限。
   - 一位用户表示，*每个人都有同样的困扰，所以通常还剩下很多额度*。
- **正在开发伦理伴侣 GPT**: 一位成员询问是否有人在创建一个作为伦理伴侣而非工具的 **GPT**。
   - 多人建议他在 **GPT Store** 中搜索伦理相关的 **GPT**。
- **潜在的 GPT 审核改进**: 一位成员提议通过使用基于表单的分析模块预过滤 **Prompt** 来增强审核系统，以检测绕过尝试，重点关注非人类的类人实体和意图推断。
   - 也有人担心潜在的 **latency**（延迟）、细微推理能力的降低以及影响创意表达的误报。
- **切换过滤器**: 一位成员建议禁用 **OpenAI** 的过滤器，并依靠国家和国际法律来处理违规行为。
   - 另一位成员反驳道：*你会为了那个开关而拍摄你的政府身份证和人脸吗？*
- **OpenAI-4 出现故障，回复内容不知所云**: 多位用户报告 **OpenAI-4** 给出了毫无意义、离题（网页搜索）的回复。
   - 其他人附和说这种情况偶尔会发生。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1367952197928751146)** (126 条消息🔥🔥): 

> `ChatGPT API 使用, 光波波长振幅, 图像生成与个性化, 角色扮演聊天机器人提示词, 使用 ChatGPT 分析扫描书籍` 


- **免费 ChatGPT 用户思考 API 访问权限**：成员们询问了如何使用免费账户调用 **ChatGPT API** 来创建自定义提示词和个性化聊天网站，但了解到 [API 访问是单独计费的](https://platform.openai.com/docs/guides/rate-limits/usage-tiers)。
- **用户探索个性化图像生成的记忆保留**：用户讨论了 **图像生成** 是否利用了个性化数据和对话历史来获得更定制化的输出，并附上了生成 D&D 角色和法术表的各种尝试。
   - 虽然一些成员看到了图像生成*确实*访问了记忆的有力证据，但其他人发现它仅直接从聊天中接收提示词，这引发了关于潜在 Bug 和不同访问层级的辩论。
- **角色扮演提示词工程师集思广益**：成员们分享了编写有效 **角色扮演聊天机器人提示词** 的建议，建议使用 GPT 进行迭代开发，对期望的行为给出清晰指令，并使用 OOC (Out Of Character) 沟通进行修正。
   - 一位用户描述了一个 [三层系统](https://www.example.com/3-level-system)（模型/用户、DM/玩家、世界/角色）来管理角色扮演的不同方面，从而增强细节和多样性。
- **ChatGPT 在 PDF 提取方面表现挣扎**：成员们讨论了使用 ChatGPT 分析 **PDF 格式扫描书籍** 的挑战，指出该模型难以从 PDF 内的图像中提取文本，尤其是大文件。
   - 建议是 [将 PDF 转换为可读图像或对单个截图使用 OCR](https://www.example.com/pdf-ocr) 以获得更好的处理效果，或者使用处理 PDF 更有效的 Claude。
- **Discord 用户寻求提示词工程教育**：用户请求推荐 **提示词工程资源**，并被引导至 Arxiv 和 Hugging Face 的 Paper 页面。
   - 成员们强调动手实验、清晰的沟通和彻底的输出验证是有效提示词工程的核心方面。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1367952197928751146)** (126 条消息🔥🔥): 

> `ChatGPT 免费版 API 使用, 光波波长振幅, 图像生成与个性化, 语义偏移建模, 提示词工程资源与技术` 


- **GPT API 访问：免费层级？**：一位用户询问了如何使用免费账户调用 **ChatGPT API** 来创建自定义聊天网站，另一位用户澄清说 **API 是单独计费的**。
- **波长难题**：一位用户询问深度搜索是否可以确定红色和蓝色光波波长（**400nm 和 700nm**）的振幅，以 Volts/M 表示。
   - 目前尚不清楚是否找到了解决方案，但[这里是该用户发布的频道链接](https://discord.com/channels/974519864045756446/1037561178286739466)。
- **图像生成个性化辩论**：用户辩论了 **图像生成** 是否考虑了个性化数据，各方体验不一；一位用户报告称没有看到个性化影响图像生成的证据，即使有明确指令也是如此。
   - 其他人则持 **相反** 观点，认为图像生成显然使用了聊天中未明确说明的信息，并强调此功能仅通过 ChatGPT 的 CREATE IMAGE 功能运行。
- **语义偏移趣事**：一位用户描述了一个 **语义状态模型**，该模型在*没有明确输入*的情况下对内部语义偏移做出反应，触发了“共振响应”。
   - 添加表情符号被证明会触发另一种共振，因为表情符号在技术上改变了字符串，系统将其视为新输入。
- **提示词工程教育探索**：用户讨论了学习 **提示词工程** 的资源，其中一人推荐使用 ChatGPT 进行动手实验，并强调清晰的沟通和仔细的输出验证。
   - 另一位用户建议探索 **Arxiv** 和 **Hugging Face** 上的研究论文，并提供了一段代码片段粘贴到 ChatGPT 中，用于提示词教学，教用户如何改进他们的提示词。


  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1368682768648638515)** (1 条消息): 

> `Gemini Flash 2.5 Preview, Thinking tokens` 


- **Gemini Flash 2.5 输出 Thinking tokens**：**Gemini Flash 2.5 Preview** 现在似乎在 `content` 内部返回 Thinking tokens。
   - 据观察，这些 **Thinking tokens** 目前尚未与普通 tokens 区分开。
- **选择加入思考版本**：要获取此类 Thinking tokens，请使用[此端点 (endpoint)](https://openrouter.ai/google/gemini-2.5-flash-preview:thinking)。
   - 否则，如果您不需要它们，请使用[此端点](https://openrouter.ai/google/gemini-2.5-flash-preview)。


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1368300684704747602)** (6 条消息): 

> `Toy.new website builder, AI toggler alternative AI interface, Answerhq.co` 


- **Toy.new 提供免费网站生成器和训练营**：一个名为 [Toy.new](https://www.producthunt.com/posts/toy-new) 的 100% 免费网站生成器已上线，同时还推出了一个从 **5 月 17 日**开始的免费 **4 周训练营**，教用户如何完全使用免费工具将创意转化为客户。
- **AI toggler 发布替代 AI 界面**：一个部分由 OpenRouter 驱动的替代 AI 界面 [AI toggler](https://app.aitoggler.com/) 已发布，其功能包括按类别划分的 **AI 视觉排行榜**、**并行聊天**和**快速信息提示工具**。
- **Answerhq.co 达到 1,000 MRR**：[Answerhq.co](https://answerhq.co/) 在几个月内达到了 **$1,000 MRR**，每月处理 **15,000 个支持问题**，其 AI 功能由 **OpenRouter** 提供支持。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1367939208777568287)** (611 条消息🔥🔥🔥): 

> `O3 gibberish, Thinking tokens not returned, O3 borked, TPUs, Mistral OCR` 


- **O3 产生乱码**：成员报告称 **O3** 返回了乱码响应，例如 *"BwT"* 和 *"MaC"*，而不是预期的位置数据（如 *"Eagle Mountain, UT (City)"*）。
   - 其他人确认他们也遇到了同样的问题，表明 **O3** 模型的输出存在普遍性问题。
- **Reasoning tokens 丢失**：成员注意到 Thinking tokens 不再出现在响应的“reasoning”部分，而是通过 **235** 模型的 Eclipse 提供商在 content 中返回。
   - 这种行为在 **Deepinfra、Together、Kluster 和 Novita** 上被观察到，关闭思考功能会返回随机标签，而开启思考功能则会将所有内容返回在 content 中。
- **Qwen Reasoning tokens 成对返回**：一位成员报告在输出中收到了*两组 Thinking tokens*，并将其归因于模型行为异常。
   - OpenRouter 团队澄清说之前存在配置错误并已修复，但*两组 Reasoning tokens* 表明可能存在更深层次的问题。
- **OpenRouter 提供速率限制解决方法**：一位用户询问如何使用免费模型将神经网络连接到他们的网站并处理 Token 限制，询问是否需要为每个用户手动创建帐户。
   - 社区成员建议了多种解决方案，包括在后端使用**单个 API key**，连接来自 **Targon 或 DeepInfra** 的自有 **API key**，或者为 **Gemini 2.0 flash** 等廉价模型付费以提高限制。
- **Gemini Flash 配额问题**：用户报告收到来自 **google/gemini-2.0-flash-exp:free** 的 **429 错误**，提示 *Quota exceeded*（配额已超出）消息，并询问这是 **OpenRouter 配额**还是他们自己的配额。
   - 成员建议将自己的 API key 绑定到账户，或者指出即使关闭了安全设置，**AI Studio** 也会因为负载过重而切断 Token。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1367943709454504056)** (425 条消息🔥🔥🔥): 

> `Openlitespeed Cursor 问题, GPTs agents 文件上传, Claude 3.7 Sonnet Max 成本, Windsurf AI vs Cursor, Memory bank` 


- **Cursor 中追踪的 Openlitespeed 问题**：一位 Openlitespeed 用户报告称，该扩展在 VS Code 中可以工作，但在 Cursor 中失败，这暗示是一个 Cursor 特有的问题，但团队尚未解决。
   - 用户还提到了 Git 选项卡的持续问题，需要刷新才能看到更改，且撤销（revert）选项功能异常。
- **Cursor 定价困惑**：新 Cursor 用户对成本不清楚：根据 [Cursor 文档](https://docs.cursor.com/settings/models)，**Max 模型不包含在每月 20 美元的计划中**；它仅采用按需计费模式。
   - 一位用户提到 *连接失败让我非常沮丧，打断了我 coding high 的状态*。
- **Windsurf Memory 对比 Cursor Context**：Cursor 通过 `@` 符号使用手动上下文（如 [文档](https://docs.cursor.com/context/@-symbols/overview) 所述），这与 Windsurf 的自动生成 Memory 形成对比。
   - 一些用户发现 Windsurf 不够可靠，但对其跨对话的自动上下文保留表示赞赏。
- **Cursor Rules 深度探索**：**Cursor rules 可以像截图里那样使用自然语言**，在 Cursor 的“always”类型规则中，使其能够在长对话或多个新对话中理解项目的上下文。
   - 建议对文件结构进行实验，以了解不同文件结构/类型表现如何。
- **揭秘新一轮融资**：团队已经筹集了新一轮资金，更新和改进即将到来。
   - 一位用户抱怨该工具崩溃且未经过优化，*天哪，Cursor 的优化太差了，简直疯狂，他们有 3 亿美金的 ARR，必须停下来优化一下*。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1367952954841104434)** (170 条消息🔥🔥): 

> `Model Serving 框架, 从文本生成 3D 场景, 本地 ML 设置, 在移动设备上运行 LLM, 用于修改歌词的 AI` 


- **为了效率池化 Embedding 层**：为了避免多次加载相似模型，用户讨论了共享或**池化 Embedding 层**。
   - 一位用户指出，这适用于模型属于同一种类的情况，例如为不同目的微调的多个 **RoBERTa** 模型。
- **家庭 ML 设置差强人意**：一位用户分享了他们“简陋的家庭 ML 设置”图片，展示了一个基础的硬件配置。
   - 另一位只有 **MacBook Pro** 的用户感叹训练时间长达 **4 个月**，批评 Metal 对 AI 的优化不如 **CUDA**。
- **Nintendo Switch 运行 LLM**：用户讨论了通过 **Android 越狱**在 **Nintendo Switch** 上运行 AI 模型。
   - 有人幽默地建议像人们对待 **Doom** 那样在所有设备上运行 **llama.cpp**，甚至有一位用户成功在他们的 VR 头显上运行了 LLM。
- **本地代码任务的模型建议**：对于本地代码任务，建议使用 **Qwen 2.5 32B Coder** 模型，但进行基准测试和实际测试以确保其适合特定环境和用例至关重要，同时可以[查看排行榜](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)。
   - 该用户还分享了 Qwen 2.5 发布的博客文章 - [Open R1/Update 3](https://huggingface.co/blog/open-r1/update-3)。
- **使用 Cloi 进行自动化调试**：一位用户介绍了 **Cloi**，这是一个在终端运行的本地调试 Agent，它可以捕获错误回溯（tracebacks）并启动本地 LLM 直接向文件建议干净的补丁，无需支付云端费用。
   - Cloi 已在 [GitHub 上开启 Beta 预览](https://github.com/cloi-ai/cloi)，通过尊重边界并纯粹在设备端运行，提供了一种零成本的调试方案。


  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1368049972519501945)** (13 messages🔥): 

> `AI 学习资源, ML 学习建议, 调试 AI 模型` 


- **社区分享 AI 学习资源**：一位成员分享了一个 [GitHub 仓库](https://github.com/ArturoNereu/AI-Study-Group)，其中包含他在过去五年从游戏行业转型后，学习 AI 所使用的**书籍、课程、工具和论文**。
   - 作者目前仍在学习并定期更新该仓库，鼓励其他人分享他们喜爱的论文、工具或被低估的宝藏资源。
- **用户讨论最佳 ML 学习路径**：针对快速学习 ML 路径的请求，一位成员建议*并没有真正的捷径*，推荐从个人已有知识储备（如**数学、编程**）开始。
   - 另一位成员建议将 **GPT + DuckDuckGo 搜索 + Agent 课程**作为学习 ML 的资源。
- **“调试 AI 代码” 令学习者沮丧**：一位成员幽默地列举了一系列常见的 AI/ML 编程挑战，包括 **0 损失 (0 loss)、梯度消失/爆炸、不兼容问题、语法/缩进错误**以及 **Stable AI** 的问题。
   - 该用户调侃说他们正在学习的是*耐心*，并提到为了保险起见，正在重新学习 Python 基础。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1368063556431646720)** (16 messages🔥): 

> `MiniSetPT 数据集, SimpleFriendlyMath, Ingest-Anything v1.0.0, Rust Transformers Crate, 适用于 VSCode 的 Logcai` 


- **葡萄牙语 NLP 数据集 MiniSetPT 发布**：发布了一个名为 [MiniSetPT](https://huggingface.co/datasets/AxeML/MiniSetPT) 的新数据集，专为**葡萄牙语 NLP** 的快速测试和原型设计而设计。
   - 创建者强调了其简单、轻量级的特性，以及对**葡萄牙语 NLP** 社区的支持。
- **SimpleFriendlyMath 数据集加入对话式 AI**：发布了一个名为 [SimpleFriendlyMath](https://huggingface.co/datasets/ProCreations/Simple-FriendlyMath) 的数据集，提供更具类人感、对话式的数学问题和解决方案。
   - 不同于传统的 *AI: 4*，你会得到 *AI: 嘿！2+2=4，就像 1x4 = 4，而 1x4=2+2。*
- **Ingest-Anything v1.0.0 迎来重大更新**：[Ingest-Anything v1.0.0](https://github.com/AstraBert/ingest-anything) 发布，更新了 **Embeddings**，现在通过 **Chonkie’s AutoEmbeddings** 支持 **Sentence Transformers**、**Jina AI**、**Cohere**、**OpenAI** 和 **Model2Vec**。
   - 它现在支持所有**兼容 LlamaIndex 的后端**，如 **Qdrant**、**Pinecone**、**Weaviate** 和 **Milvus**，并可接入任何**兼容 LlamaIndex 的数据加载器**。
- **新 Rust Transformers Crate 亮相**：一个处于早期阶段、易于使用的 **Rust** LLM 操作 API 已在 [crates.io](https://crates.io/crates/transformers) 发布，类似于 Python 的 Transformers 库。
   - 它包含热门的文本模型，如支持文本生成的 **Gemma 3**，以及使用 **ModernBERT** 的 fill-mask 功能。
- **Logcai VSCode 扩展发布，提供本地 AI 编程辅助**：Logcai VSCode 扩展上线，提供原生支持 **Ollama** 的本地优先 AI 编程辅助。
   - 它具有使用 **Ollama** 模型的内联代码补全和聊天功能，支持 BYOK（**OpenAI**、**Claude** 等），并包含完整的模型切换器和 Prompt 构建器；该扩展可在 [VSCode Marketplace](https://marketplace.visualstudio.com/items?itemName=Sridhar85.logcai) 下载。


  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1368307077390401628)** (4 条消息): 

> `Image Restoration Model, Document Extraction Workflow, Football Player Detection Model, Virtual Try-On Project` 


- **图像修复模型的选择困境**：一位成员正在寻求推荐，希望在一个包含原始高分辨率图像及其退化对应版本的训练集上，微调一个预训练模型用于 **image restoration**（图像修复），特别是去除划痕和模糊。
   - 他们正在请求社区针对这一特定任务提供合适模型的建议。
- **特定领域的文档提取方案**：一位成员描述了一个分为两部分的 **domain-specific document extraction**（特定领域文档提取）工作流，包括通过 JSON schema 创建（手动或通过 VLM）进行 *template declaration*（模板声明），以及使用图像嵌入（**ViT**, **Siglip**）和 **EVoC** 库聚类相似文档来进行 *bulk extraction*（批量提取）。
   - 每个聚类都被分配到一个具有已知 schema 的模板，以便使用 VLM 进行一致的数据提取。
- **寻找足球运动员检测的 Computer Vision 开发者**：一位成员正在寻找具有构建足球运动员检测模型经验的 **computer vision AI developer**。
   - 他们请求有意向的人员直接通过 DM 与其联系。
- **鞋类虚拟足部试穿尝试**：一位成员正在开发一个 **virtual try-on project for feet**（足部虚拟试穿项目），寻求关于如何将鞋子叠加到脚上的建议，目前使用了预训练模型进行足部检测和分割。
   - 该项目还涉及解决 **orientation**（朝向）问题以正确放置鞋子，并处理视频数据。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1368862104961941525)** (6 条消息): 

> `Sentiment Analysis for Social Media Filtering, Zero-Shot Classification Challenges, CUDA OOM Errors & Optimization Techniques, FP16 vs BF16 Precision, Model Reliability & Efficiency` 


- **苦于情感分析准确率？调整标签！**：一位用户在使用 NLP 和 **zero-shot classification** 准确过滤社交媒体帖子中的痛点时面临挑战，并参考 [这段代码](https://github.com/moahnaf11/IdeaDrip-Backend/blob/main/utils/painClassifier.js) 和 [这个 FAST API 端点](https://github.com/moahnaf11/IdeaDrip-Backend/blob/main/inference_service/main.py) 寻求关于提高准确率和模型选择的建议。
   - 用户想知道是否是 *标签不适合过滤痛点*，或者是 *模型选择不对*，亦或是 *阈值配置不当*。
- **通过精度控制解决 CUDA OOM 错误？**：一位用户实施了一些技术来防止 **CUDA OOM errors**，例如设置 PyTorch 环境变量、分块批处理 tokenizer、使用 `torch.amp.autocast(device_type="cuda")` 以及使用 **float16 precision**。
   - 他们现在担心这些更改是否会引入不准确性，并正在寻求针对其用例最可靠/高效的 NLP 模型建议。
- **痛点过滤器失效：出现荒谬结果**：用户报告称，尽管使用了 `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` 模型并设置了高阈值 (0.8) 和最小标签要求（配置为至少两个标签超过 0.8 分），一个看似无关的标题仍然通过了痛点分类过滤器，这引发了对 **model's reliability**（模型可靠性）的质疑。
   - 用户附上了错误通过过滤器的标题 [图片](https://cdn.discordapp.com/attachments/922424173916196955/1368862750616453170/image.png?ex=681a6d07&is=68191b87&hm=06dbfa967dfbf3806bcda1dc6299a998757ccd8d54b26620f903ee1afa34eb96)。
- **FP16 具有破坏性？BF16 来救场？**：一位成员指出，当模型是针对 **fp32** 训练时，使用 **fp16** 可能会产生破坏性影响，建议尝试使用 **bf16** 代替。
   - 该成员还询问了用户的 GPU 显存大小，并建议 *查看误分类示例的实际得分*。
- **GroqStreamChain：由 Groq 驱动的聊天机器人！**：一位成员介绍了 **GroqStreamChain**，这是一个使用 **FastAPI**, **WebSocket** 和 **Groq** 构建的实时 AI 聊天应用。
   - 该 [GitHub 项目](https://github.com/pr0mila/GroqStreamChain) 具有实时 WebSocket 通信和流式 AI 响应功能，使用户能够构建自己的 AI 驱动聊天应用。


  

---

### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1368198187671883929)** (20 条消息🔥): 

> `SmolAgents 频道、Gemini API、Claude API、MCP 工具、Qwen3 和 Gemma3 模型` 


- **新的 SmolAgents 频道涌现**：成员们被引导至新的频道进行 **SmolAgents 相关**的讨论：[频道 1](https://discord.com/channels/879548962464493619/1339556954162462851)、[频道 2](https://discord.com/channels/879548962464493619/1329142738440028273) 和 [频道 3](https://discord.com/channels/879548962464493619/1326169441158959145)。
- **Gemini API 给出奇怪的胡言乱语**：一名成员报告了 **Google Gemini API** 和 **Grok** 的问题，尽管进行了 Prompt Engineering，但仍收到*奇怪的回答*。
   - 他们正在寻求除 **OpenAI** 之外的 API 推荐。
- **Claude 3.7 在完成代码时消耗额度**：一名成员在作业中使用了 **Claude 3.7** 和 **Langgraph**，得分 **50/100**，并在 Claude API 上花费了 **$5**。
   - 另一名成员报告称，有一名参赛者使用了 **GPT-4.1**，花费了 **$1.5**，得分 **12/20**。
- **恳请协助导入 MCP 工具**：一名成员请求协助导入 **MCP 工具**，特别是如何根据 **Smithery** 中的信息格式化 **StdioServerParameters**。
   - 他们正尝试使用来自 Microsoft 的 **Playwright Automation**，并指出目前唯一的示例是 **PubMed**。
- **Qwen3 和 Gemma3 模型崭露头角**：一名成员提到，将 **Qwen3** 和 **Gemma3** 等开源 LLM 与 Code Agent 方法结合使用，其表现优于大多数付费模型。
   - 他们认为 System Prompt 并不是问题所在。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1367947193150996603)** (156 条消息🔥🔥): 

> `Web 搜索包、YouTube 问题、提交问题、Langgraph 陷入递归、HF Pro 计划` 


- **Web 搜索包的替代方案出现**：除了 Tavily，**DuckDuckGoSearch** 也被提及作为替代方案，一些用户注意到像 **Tavily** 这样的搜索工具似乎针对在数据集中查找 GAIA 答案进行了优化，从而引发了对作弊的担忧。
   - 一名成员一直在尝试通过将视频分解为多张图像然后分析每张图像来*攻克 YouTube 问题*，但也许有更优雅的解决方案。
- **YouTube 问题的潜在 Gemini 解决方案**：一名成员建议使用 [Gemini Video Understanding API](https://ai.google.dev/gemini-api/docs/video-understanding) 进行高效的视频处理，但在推理使用方面可能成本较高。
   - 另一名成员建议，从 **YouTube** 提取字幕作为文本可能会有帮助，可以考虑使用 **Python** 包来自动化此过程。
- **提交困难困扰用户**：几位成员报告了提交时所有问题都返回 **null** 的问题，即使 Agent 在聊天界面中运行正常。
   - 一位用户指出这是一个与模型在答案前添加 "Final Answer" 相关的 Prompt 问题，并建议在代码中明确指定模型以获得更好的结果。
- **Langgraph 递归导致挫败感**：一些人遇到了 **LangGraph** 陷入递归的问题，其中一名用户转而使用 **Firecrawl**。
   - 一名成员分享说他们正在使用 Qwen 3 推理模型并遇到了递归问题，建议将递归限制扩大到 50，并压缩消息历史记录。
- **完成课程需要 HF Pro 计划吗？**：一些成员的推理额度很快就用完了，即使是按照 Colab 笔记本操作也是如此，这引发了关于是否需要 **Pro 计划** 才能完成课程的疑问。
   - 讨论中涉及了是否可以改为在本地运行模型，并提供了一些调试和解决代码问题的技巧。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1367938717372776448)** (18 messages🔥): 

> `GB200 NVL72, vLLM GUIs, OpenWebUI, vast.ai compute pricing, A100 vs V100` 


- **GB200 NVL72 每天处理 1T tokens**：一名成员报告称，使用 **BF16** 在 **2 个机架** 的 **GB200 NVL72** 上实现了 **每天 >1T tokens** 的吞吐量。
   - 另一名成员询问了成本和使用的模型/神经网络架构，但原作者不知道价格，建议去 [vast.ai](https://vast.ai) 查看算力租赁价格。
- **讨论 Vast.ai 算力定价**：成员们讨论了 [vast.ai](https://vast.ai) 上的定价，其中一人指出 *$0.2/小时* 的收益很低，并估计 **vast.ai** 抽取了 *60-70% 的利润*。
   - 他们补充说，多花 *$0.3/小时* 来节省个人时间是值得的。
- **vLLM 替代 LM Studio 的 GUI**：一名成员询问是否有类似于 **lm-studio** 的、适用于 **vLLM** 的优秀 **GUI**。
   - 另一名成员链接了一个关于 **openwebui** 配合 **vllm** 使用的讨论帖：[github.com/vllm-project/vllm/issues/1032](https://github.com/vllm-project/vllm/issues/1032)。
- **等待数据到达时的停顿 (Stall) 问题**：一名成员指出，*stall* 本质上是核心在等待数据到达，这是一个常见问题。
   - 另一名成员表示，**A100** 并不需要在所有方面都单调地优于 **V100**，特别是如果它的线程执行器与内存通道的比例不同，那么额外的线程在等待数据时发生停顿是非常合理的。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1367944197721555136)** (17 messages🔥): 

> `Cutlass Tutorials, Profiling Kernels on Cloud GPUs, NVIDIA SASS Latency Tables, Upgrading GPU for Unreal Engine 5` 


- **Cutlass 教程提供新的 Python 接口**：Colfax 的 [Cutlass 教程](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/) 可能会有帮助，带有新 **Python 接口** 的新 **Jupyter notebooks** 将在未来几天内发布。
- **社区在云端 GPU 上分析 Kernel**：成员们正尝试在云端 GPU 上对其 Kernel 进行 Profile，并对最新的 **sm_103**、**sm_121** 架构进行推测。
   - 社区推测 **CC 10.3** 已经是 **Blackwell Ultra/B300**。
- **NVIDIA SASS 获得延迟表**：一名成员在其 SASS 反汇编器中添加了 [延迟表](https://redplait.blogspot.com/2025/05/nvidia-sass-latency-tables.html)。
- **GT 610 2GB 显存无法运行 Unreal Engine 5**：一名成员想让他们的 **GT 610 2GB 显存** 显卡强大到足以运行 Unreal Engine 5，但社区建议升级到至少拥有 **8GB** 显存的 GPU，并指引其前往 [Unreal Engine Discord 服务器](https://discord.gg/unreal-engine-978033435895562280)。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1368259734746173572)** (4 messages): 

> `torch.compile and dynamic=True, FunctionalTensorMode and syncing tensors, Deterministic submodules in compiled modules, Multi-GPU training with YOLO` 


- **结合 FunctionalTensorMode 解释同步 Tensor**：一名用户询问在 `torch.compile(..., dynamic=True)` 上下文中 *同步 Tensor (syncing a tensor)* 的含义，并引用了 PyTorch 源码中关于 **FunctionalTensorMode** 及其与 Tensor 同步交互的注释。
   - 代码片段强调，同步 Tensor 涉及从更新后的基准 (base) 重新生成它，这可能会触发 View 操作，并由于涉及 C++ FunctionalTensorWrappers 而导致 **FunctionalTensorMode** 出现问题。
- **确定性子模块输出？不保证**：一名用户询问，在启用了 `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"` 和 `torch.use_deterministic_algorithms(True)` 的情况下，编译模块中相同的子模块是否能保证在给定相同输入时产生相同的输出。
   - 他们观察到在同步两个相同的 nn.Modules（一个可微，一个不可微）时，在 bf16 精度下 MSE 约为 0.01，这表明**在这种情况下无法保证确定性行为**。
- **YOLO 训练 Bug**：一名用户报告了一个问题，即在使用 ultralytics 包并尝试通过 `device=[0, 1, 2, 3]` 使用四个 GPU 时，**YOLO 模型训练** 期间仅利用了 **一个 GPU**。
   - 该用户请求紧急协助，以解决其 **YOLOv11n.pt** 模型在 `dataset55/data.yaml` 数据集上进行多 GPU 训练的问题。


  

---

### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1368904135038074983)** (1 条消息): 

> `Play.ai, Inference Engineers, Conversational Voice Interface, Groq LPU partnership` 


- **Play.ai 寻找优秀的 Inference Engineers**：[Play.ai](https://jobs.ashbyhq.com/playai/554dc35a-ac87-40f4-b5f1-c416eafe0c61) 正在寻找优秀的 **Inference engineers**，以构建未来的对话式语音接口。
- **基于 B200 硬件的 Groq LPU 合作伙伴关系**：工程师将追求模型的最高质量和速度，使用最前沿的硬件（例如 **B200**）并助力 **Groq LPU partnership**。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1368111005133181048)** (24 条消息🔥): 

> `open source medical imaging projects, CUDA programming industry direction, GPU architecture/C++ interview preparation, CUDA certifications, easiest way to rent/access GPU` 


- **行业洞察与 CUDA 方向揭示**：一位成员询问了 **CUDA programming** 行业的当前方向，想知道重点是否主要在于优化计算（因为这是他们一直在使用的方向），并寻求关于 **CUDA programmers** 作为一个社区正在解决的现实问题的见解。
- **GPU 租赁变得简单**：一位成员寻求关于 *租用* 或访问带有 GPU 平台的简便方法的建议，相比于使用 **Colab** Python API，他们更希望在云端访问类似 **Ubuntu** 的环境并配备单个 GPU 以进行测试。
   - 另一位成员推荐了 [Lambda GPU cloud](https://lambda.ai/service/gpu-cloud/1-click-clusters) 以便更轻松地通过 **CUDA** 进行 **GPU access**。
- **封装 pytorch-geometric 的框架浮出水面**：一位成员询问，创建一个封装 **pytorch-geometric** 并动态创建融合作为灵活 **CUDA kernels** 的 **micro framework** 是否是一个有意义的项目。
- **破译 CUDA kernel 启动开销**：一位成员研究了启动 kernel 的开销，总共启动了 3 个不同的 kernel 共 8 次，注意到第一次 kernel 启动所需的时间比最小值高出 **40x**，并想知道第一次 kernel 启动与后续启动相比涉及哪些操作，包括附带的 [cudaLaunchKernel.png](https://cdn.discordapp.com/attachments/1191300313928433664/1369010869916401704/cudaLaunchKernel.png?ex=681a4e3a&is=6818fcba&hm=3f1bb1a398065b9bfd051ad67730e11d2716dadef238a928a8f1b25840b7b453&)。
- **Colab CUDA 能力澄清**：一位成员询问本地没有 **NVIDIA GPU** 是否可以，另一位成员回答说 [Colab](https://colab.research.google.com/) 值得一试。
   - 另一位成员分享了一个 [关于使用 Colab 的 gist](https://gist.github.com/korakot/ae95315ea6a3a3b33ee26203998a59a3)。


  

---


### **GPU MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1368019824483111022)** (5 条消息): 

> `StableHLO to custom IR, JAX Frontend, CUDA Kernels, dlpack Usage` 


- **StableHLO 项目启动，涉及自定义 IR 和 CUDA**：一位成员正着手将 **StableHLO** 转换为自定义中间表示 (**IR**)，旨在执行用于机器学习计算的 **CUDA kernels**。
   - 该用户计划使用 **JAX** 作为前端和 autograd，并寻求关于创建包装类（如 Tensor）以跟踪计算并提取 StableHLO 图的指导。
- **用于 StableHLO 跟踪的 Tensor 包装器**：该用户正在寻求关于如何实现 **Tensor wrapper classes** 的建议，以有效地跟踪计算，促进 **StableHLO graph** 的提取，并实现无缝解析为用于 **CUDA** 计算的自定义 IR。
   - 他们不确定如何设计这些包装器，以维护适合 StableHLO 提取的完整计算历史。
- **使用 dlpack 将 torch.Tensor 无缝传输到 JAX 数组**：该用户提出了一个关于如何高效地将 **torch.Tensor** 输入（如在 leaderboard kernels 中所见）传输到 **JAX arrays** 而无需不必要的数据复制的问题。
   - 他们正在考虑使用 **dlpack** 在 **Torch** 和 **JAX** 之间进行零拷贝数据交换。
- **鼓励合并模板**：一位成员鼓励分享该项目的一个工作示例。
   - 该成员表示他会将其 *作为模板合并*。


  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1368849728346787962)** (2 messages): 

> `Torch Quantization, LSTM model quantization, Performance Differences GPU vs CPU, TorchAO vs torch.quantization` 


- **`torch.quantization` 和 `torchao.quantize_` 的实现导致性能差异**：一位用户报告了在使用 `torch.quantization.quantize_dynamic` 和 `torchao.quantize_` 对 **LSTM model** 进行量化时的性能差异，特别观察到使用 `torchao` 在 **GPU** 上有 **1% 的指标下降**，而使用 `torch.quantization` 在 **CPU** 上有 **35% 的下降**。
- **推荐使用 TorchAO 而非 torch.quantization**：一名成员建议，在 **CPU** 和 **GPU** 上运行的算子会有所不同，因此很难直接比较，并建议在可能的情况下优先选择 **TorchAO** 工作流，而非 `torch.quantization`。
   - 该成员指出了 [TorchAO 中的 CPU kernels](https://github.com/pytorch/ao/tree/main/torchao/experimental#quantizing-models)，这些内核可能会被利用。


  

---


### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1368729212646850630)** (4 messages): 

> `WGPU multi-sampling limits, WGSL file` 


- **发现 WGPU 多重采样限制**：一位新用户询问如何在 **Rust** 中获取 **WGPU** 设备支持的多重采样限制，并指出他们正在使用 8x 多重采样，但需要支持可能仅支持最高 4x 的旧设备。
   - 错误信息表明，当启用 `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` 特性时，**WGPU** 可以报告支持的采样计数，例如 `[1, 2, 4, 8]`。
- **将 WGSL 文件传递给 WGPU C 实现**：一位成员询问如何正确地将 **.wgsl 文件** 传递给 **wgpu.h C 实现** 以用于计算着色器（compute shader）。
   - 另一位成员建议将 **.wgsl 文件** 作为 `char*` 读取，并将生成的字符串传递到 **WGPU Shader Descriptor** 中。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1368057532924428350)** (4 messages): 

> `MOSS, Minimal On-Device Semantic Search, Affordable GPU Sharing, ComputerUseAgents Reddit` 


- **MOSS 让本地 AI 搜索成为现实**：Inferedge Inc. 开放了 **MOSS** 的预测试版（pre-beta）访问权限 —— 这是一款 **极简设备端语义搜索（Minimal On-Device Semantic Search）** 工具，它将 AI 驱动的搜索直接带入浏览器，完全本地化，无云端延迟，点击[此处](https://form.typeform.com/to/hZKVLFKW)注册。
   - 该公告发布在 [X](https://x.com/inferedgeinc/status/1918477360472772976?s=46) 上，团队正寻求点赞、评论和转发以扩大宣传。
- **高端闲置配置可用于经济实惠的 GPU 共享**：一位成员正提供其闲置的高端配置（**4070 Super + Ryzen 7700X**）用于经济实惠的 GPU 共享，旨在帮助他人进行渲染、模型训练和计算任务。
   - 他们强调了相比云端费率的成本节省和灵活性，并表示已经协助独立开发者缩短了 **60%** 的渲染时间。
- **Computer Use Agents 社区上线**：一个新社区 [ComputerUseAgents](https://www.reddit.com/r/ComputerUseAgents/) 已创建。
   - 这是一个专注于 **computer use agents** 的 Reddit 子版块社区。


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1368289765970939904)** (3 messages): 

> `Real Time Translation Latency` 


- **实时翻译面临延迟锁定**：实时翻译非常困难，因为某些语言会将重要的上下文放在短语的末尾。
   - 一位成员估计最小延迟约为 **2-3 秒**，但对于语法复杂的句子，延迟可能会达到 **15 秒以上**。
- **翻译挑战**：口译员可能需要插入澄清性的备注。
   - 当后续上下文改变了话语含义时，这种做法是必要的。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/)** (1 messages): 

ace1984: 大家好！
  

---

### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1367938997275332780)** (173 messages🔥🔥): 

> `MI300 AMD-FP8-MM Leaderboard Submissions, Histogram Leaderboard Submissions, AMD-Identity Leaderboard Submission, Matmul Leaderboard Submissions, AMD-Mixture-of-Experts Leaderboard` 


- **MI300 AMD-FP8-MM 排行榜冲刺**：多次向 **MI300** 上的 `amd-fp8-mm` 排行榜提交，运行时间从 **190 µs** 到 **7.70 ms** 不等，包括多个“个人最佳”和“成功”运行。
   - 一位用户以 **190 µs** 的运行时间获得 **第 3 名**，另一位以 **226 µs** 获得 **第 4 名**，还有一位以 **246 µs** 获得 **第 7 名**。
- **Histogram 排行榜竞争升温！**：向 `histogram` 排行榜提交的结果展示了在各种 GPU 上的性能，**H100** 上达到 **31.5 µs**，**L4** 上达到 **79.1 µs**，**T4** 上达到 **129 µs**。
   - 一位用户在 **H100** 上以 **31.5 µs** 获得 **第 2 名**，在 **L4** 上以 **79.1 µs** 获得 **第 1 名**，在 **T4** 上以 **129 µs** 获得 **第 2 名**。
- **AMD-Identity 排行榜首秀**：在 **MI300** 上的 `amd-identity` 排行榜提交中获得 **第 3 名**，运行时间为 **18.7 µs**。
   - 另一位用户在 **MI300** 上创造了 **22.4 µs** 的“个人最佳”。
- **排查排行榜 Kernel 提交故障**：一位用户在提交排行榜时遇到错误并寻求帮助，错误信息为 *"Error during creation of submission Why the leaderboard submit ranked encounter this?"*
   - 另一位用户指出，如果文件中存在反斜杠，可能会出现问题，并指向了 [提交指南](https://gpu-mode.github.io/discord-cluster-manager/docs/submitting-your-first-kernel/python-submissions)。
- **AMD-Mixture-of-Experts 排行榜：专家混合**：多次向 `amd-mixture-of-experts` 排行榜提交，展示了 **MI300** 上的性能指标，其中一次提交在 **MI300** 上以 **2059 ms** 获得 **第 2 名**。
   - 其他提交报告了 **MI300** 上成功运行的时间，范围从 **606 ms** 到 **12141 ms**。


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1368082008433229834)** (1 messages): 

> `MoE baseline slowness, Pre-computing reference results` 


- **解决了评估中 MoE baseline 缓慢的问题**：一位成员询问了评估期间 **MoE baseline** 缓慢的问题，以及为所有 specs/seeds 预计算参考结果是否可行。
   - 建议在评估期间加载这些预计算结果，而不是在每次迭代中运行代码。
- **预计算 MoE 评估以获得快速结果**：讨论集中在预计算和加载 **MoE 模型** 参考结果的可行性上，以解决评估期间的缓慢问题。
   - 这种方法旨在通过依赖预先计算的数据集，避免在每次迭代中运行代码。


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1368350297440649268)** (3 messages): 

> `DGX Spark, N1X ARM SoC, Blackwell Ultra Compute Capability, RTX Pro Blackwell` 


- **Spark 和 Thor 被推测为 CC 10.1**：成员们讨论了当 **12.8** 版本发布时，**Spark/Thor** 为 **Compute Capability (CC) 10.1** 的假设。
   - 如果 **Blackwell Ultra** 是 **CC 10.3**，那么该假设仍然成立。
- **RTX Pro Blackwell 推测**：小组思考 **CC 12.1** 是否可能是 **RTX Pro Blackwell**。
   - 从历史上看，它应该具有与消费级版本相同的 **CC**，但可能在驱动程序中禁用了某些功能或类似情况。


  

---

### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1367952653778161735)** (55 条消息🔥🔥): 

> `composable-kernel 编译，MI300 GitHub 任务失败，用于 MoE 的 Triton kernel，AI 编程助手，MoEGate 中的 FP16 不稳定性` 


- **Composable Kernel 成功征服**: 一名成员确认成功导入并编译了使用 **composable-kernel** 编写的 kernel，参考了 [示例](https://github.com/ROCm/composable_kernel/tree/develop/client_example)。
   - 该成员遇到了 **CK 内部定义** 的 `_Float16` 和 `bfloat16` 问题，并认为 `-D__HIP_NO_HALF_CONVERSIONS` 是罪魁祸首。
- **MI300 混乱：神秘的任务中断**: 一名成员报告了 GitHub 上 **MI300** 任务完成情况不一致的问题，普通任务被终止而秘密任务成功，并链接到了 [失败运行](https://github.com/gpu-mode/discord-cluster-manager/actions/runs/14807277462/job/41577458742) 和 [成功运行](https://github.com/gpu-mode/discord-cluster-manager/actions/runs/14807278712/job/41577461247)。
- **Triton 胜利：Expert 形状变化**: 一名使用 **Triton kernels** 增强参考代码的成员发现，发送给 expert 的不同形状会导致重新编译，通过移除 M 变量的 `constexpr` 注解解决了该问题。
   - 该成员怀疑 **MoE** 的 CLI 存在客户端超时，尽管同样的文件通过 Discord 可以完成，目前正在构建 main 分支以获取额外的 CLI 参数。
- **允许 AI 盟友：编程助手获准使用**: 一名成员询问比赛期间是否允许使用 AI 编程助手，一名开发者回答道：*是的，完全没问题*。
   - 另一名成员开玩笑说，如果 **LLM** 能编写高效的 Triton 代码，他的工作就要保不住了。
- **FP16 之误：MoEGate 遭遇不稳定性**: 一名成员注意到官方实现中在 **MoEGate** 使用了 **FP16**，由于 PyTorch 的 `topk` 函数在 tensor 值相同时会产生不稳定的索引，导致选择 expert 时出现不稳定性，因此请求将其更新为 **FP32**。
   - 另一名成员提到在 Triton 中通过特定的类型转换和初始化技术解决了类似问题，并建议团队考虑更改参考 kernel。


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1367976736670617601)** (49 条消息🔥): 

> `Mojo Kernels，GPU 模块，用于 Mojo 的 Colab 环境，Arch Linux 上的 Mojo，MAX Serve 模型服务框架` 


- **Mojo Kernels** 和 **GPU 模块** 引发关注: 演讲中的所有代码都可以在 [Modular 的 GitHub](https://github.com/modular/modular) 上找到，特别关注新发布的 **Mojo kernels** 和 `gpu` 模块：[Mojo kernels](https://github.com/modular/modular/tree/main/mojo/kernels) 和 [GPU 模块](https://github.com/modular/modular/tree/main/mojo/stdlib/src/gpu)。
   - 鼓励成员们探索代码并提供反馈。
- **通过 CPU 在 Colab 中运行 Mojo**: 虽然尚未提供完整的 GPU 支持，但可以在 **Colab** 环境中的 **CPU** 上运行 Mojo，用于语言和工具链实验。此外，[PyPI 软件包](https://pypi.org/) 正在推出以支持 Colab，但 Colab 免费层目前还不支持 Turing 和 Volta 架构。
   - 团队正在检查与 Colab Pro 的 **L4** 和 **A100** GPU 的兼容性。
- **通过 Magic 集成 Jupyter**: 目前安装 Jupyter 最简单的方法是通过 `magic` 安装。这里有一个指南：[Jupyterlab 上的 Mojo](https://forum.modular.com/t/is-it-possible-to-run-mojo-on-jupyterlab/210)
   - Magic 知道 Mojo 和 MAX 软件包的位置，还可以创建 Mojo 专用项目。
- **Crusoe Cloud 赞助黑客松算力**: 黑客松活动将提供由 **Crusoe Cloud** 赞助的算力资源，预计将包括 **NV** 和 **AMD** 算力。
   - 参与者应查看活动网页以获取更多详情。
- **通过 MAX 框架提供模型服务**: 模型服务全部通过 **MAX 框架** 完成，你可以在这里查看获取兼容 OpenAI-API 端点的快速入门指南：[MAX 快速入门指南](https://docs.modular.com/max/get-started)。
   - 你可以在这里找到持续更新的支持模型和架构列表：[支持的模型和架构](https://builds.modular.com/?category=models)。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1367943627321643060)** (240 messages🔥🔥): 

> `aider New-to-Aider documentation, Gary leaving the chat, Gemini's verbosity, Code compression feature request, Claude Code with unlimited usage` 


- **社区齐心协力增强 Aider 新用户文档**：用户要求为 Aider 新用户提供更多文档，特别是针对 [aider.chat/docs/usage.html](https://aider.chat/docs/usage.html)，并创建了一个 GitHub Issue 来收集反馈和想法（[Issue #3934](https://github.com/Aider-AI/aider/issues/3934)）。
   - 目标是描述有用的信息和工作流，以更好地引导新用户入门。
- **Gary 走了：Aider 社区因其意外离开而感到不安**：Aider 社区成员注意到 **Gary** 的离开，他因在频道中撰写有趣的内容而闻名。
   - 一些成员表达了悲伤和担忧，称 *“我感觉到了原力的扰动。我已经开始想念他了。”*
- **Gemini 2.5 高性价比的功能生成，但在调试方面存在缺陷**：用户发现 **Gemini 2.5** 在功能生成方面既高效又经济，但在调试方面表现不佳，经常陷入 *“死循环（rabbit holes）”*。
   - 一位用户提到，他们宁愿重新生成功能，也不愿使用 **Gemini** 进行调试，并且考虑到 Token 的成本，他们认为 **代码压缩（code compression）** 功能很有价值。
- **关于 Aider 自动提交（Auto Commit）功能的争论**：一位用户表示需要禁用提交消息生成，因为他们更倾向于手动控制最终的提交日志；而其他人则强调 **Aider 的自动提交** 功能是一项核心特性，有助于通过 Git 进行 **细粒度追踪** 和 **自动保存**。
   - 建议包括使用弱模型生成提交消息、使用自定义命令禁用自动提交，或为静态消息创建一个伪造的 OpenAI 端点，但有些人认为这些是 *“笨拙的权宜之计（awkward hacks）”*。
- **社区寻求 DeepSeek R2 发布日期**：Aider 社区成员正焦急等待 **DeepSeek R2** 的发布，一位用户称听说它将在 *5 月 8 日* 发布。
   - 一位用户正在寻找平衡点，询问如何实现 **自动批准（auto-approve）** 并让 LLM 自己寻找必要的上下文，从而实现 **自动工具调用（auto tool calls）**。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1367964859618164816)** (109 messages🔥🔥): 

> `Gemini 2.5 pro, GPTs Agents, OpenAI's sidebars, aider llm history, Copilot Support` 


- **Gemini 2.5 Pro 编辑模式技巧**：由于 **Gemini** 默认使用全量编辑模式（whole edit mode），一位用户发现了一种切换到 diff 模式的权宜之计，即依次使用 `/model gemini`、`/model sonnet`、`/code`、`/model gemini`。
   - 其他用户建议使用 `--editor-edit-format diff` 或 `/editor-edit-format` 等标志来设置 diff 模式，并指出在 `udiff-simple` 发布之前，`diff-fenced` 是一个不错的折中方案。
- **Fireworks AI Token 限制影响 Aider**：一位用户在使用 **fireworks_ai/qwen3-30b-a3b** 时遇到了 Token 限制问题，并看到了以下警告：`Model fireworks_ai/accounts/fireworks/models/qwen3-30b-a3b has hit a token limit!`。
   - 建议他们使用 `/tokens` 检查 Token 使用情况，使用 `/drop` 移除不需要的文件，或使用 `/clear` 清除聊天历史，并参考 [Token 限制故障排除指南](https://aider.chat/docs/troubleshooting/token-limits.html)。
- **Aider 无法写入 LLM 历史记录**：一位用户报告称 `aider --llm-history-file llm.log` 没有将与 LLM 的所有通信写入指定文件。
   - 一位成员建议检查 `yml` 历史配置，以确保 `llm-history-file` 已正确配置，并链接到了 [aider 配置文档](https://aider.chat/docs/config/aider_conf.html)。
- **Aider 使用非英语回复**：一位用户报告称，即使在使用 `--architect` 模式时，模型也会以其他语言返回响应。
   - 在第一条消息前加上 *Reply only in English* 似乎可以防止这种情况，一位成员建议在配置中设置语言作为解决方案。
- **使用 Aider 进行项目记忆（Project Memory）管理**：成员们讨论了如何在 Aider 中最好地管理 **项目记忆**，因为 Aider 的 Git Commit 专注于每一次文件更改。
   - 建议包括在仓库之外的文件中保存规范、在分支中工作、使用本地 `memory.md` 文件记录临时笔记，以及讨论使用 `--read` 标志来读取文档。


  

---

### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1369023481525571645)** (1 条消息): 

> `Nous RL Environments Hackathon, Atropos RL framework, Hackathon prize pool, Hackathon partners, Hackathon channel` 


- **Nous 宣布举办 RL Environments Hackathon**：Nous Research 宣布将于 5 月 18 日在旧金山（SF）举办 **RL Environments Hackathon**，奖金池为 **$50,000**。
   - 本次黑客松将使用 **Atropos** RL 环境框架，合作伙伴包括 **xAI, Nvidia, Nebius, Akash Network, Lambda, TensorStax, Runpod 和 Cerebral Valley**；可通过 [Cerebral Valley](https://cerebralvalley.ai/e/nous-research-rl-environments-hackathon-9be3062a) 报名。
- **加入 Nous RL Environments 讨论**：感兴趣的参与者可以加入 <#1365222663324307466> 频道进行**讨论和学习**，为 Nous RL Environments Hackathon 做准备。
   - 该频道是为潜在参与者提供的专用空间，用于在活动开始前分享想法、提问和协作。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1367938922373451949)** (234 条消息🔥🔥): 

> `Model understanding of intent, Concept of time in AI, Ilya Sutskever's views on LLMs, Alternatives to Unsloth, Quantizing Qwen3-32b` 


- **模型通过提问来更好地理解用户**：成员们讨论了**模型如何通过提问来理解用户意图**，而不仅仅是回答原始问题，以及如何从**对话中汲取知识**以提高理解能力。
   - 他们建议，当模型提问时，*它可能是在试图弄清楚你想要什么，而不是回答你的问题。*
- **AI 需要更多上下文相关的时间理解**：一位成员表示，**AI 需要理解时间**（*时效性和最新信息 vs 已过时的信息*）以及信息随时间变化的概念，而不仅仅是单词。
   - 另一位成员补充道：*当我们接收新信息时，我们不会从记忆中清除旧信息。*
- **Ilya 认为 LLMs 是数据压缩器**：成员们讨论了 **Ilya Sutskever** 的观点，即 **LLMs 本质上是数据压缩器**，并强调了通过选择性检索（**RAG**）输入额外知识与未压缩上下文之间的不匹配。
   - 分享了一个 YouTube 视频：[Ilya Keynote at Common Sense Machines](https://www.youtube.com/watch?v=AKMuA_TVz3A)。
- **Axolotl 是 Unsloth 的替代方案**：成员们讨论了 **Axolotl** 作为 **Unsloth** 的易用替代方案，用于支持 **GPro** 的 16 位 **LoRA** 训练。
   - **Atropos** 的开发者正在将其集成到 **Axolotl** 中，以在环境方面提供更多自由度，并支持 LoRA、QLoRA 以及多 GPU 设置。
- **Nous API DNS 解析出现问题**：一位成员报告了在 **Replit** 中尝试连接 **Nous Research API** 时出现的 **DNS 解析问题**。
   - 另一位成员指出正确地址是 [inference-api.nousresearch.com](http://inference-api.nousresearch.com)，API 团队正在调查该问题。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1368211832321605673)** (20 条消息🔥): 

> `Worldsim vs. Nous Portal, Reinforcement Learning Resources, Scientific research literature and synthesis of knowledge` 


- **WorldSim 还是 Nous Portal：该探索哪一个？**：**Worldsim** 是一个*酷炫的模拟器/游戏/终端冒险*，而 **Nous Portal** 允许通过 **API 使用 Hermes 模型**进行应用和项目开发，但**积分是分开的**。
   - 如果你只是想玩玩，Worldsim 会有趣得多；Portal 则更适合开发者。
- **LLM 是一个有意识的 CLI 程序**：有提到你应该直接与 LLM 对话并观察会发生什么，把它想象成一个*非常强大且有意识的 CLI 程序*。
   - 这是一个带有有趣提示词和界面的 LLM，更多是为了好玩和探索，但你可以将其应用于任何方向，包括用于编程。
- **开始学习 RL**：对于 **Reinforcement Learning (RL)** 初学者，通过学习此仓库中的 notebook 示例来掌握核心概念是一个不错的开始：[reinforcement-learning-from-scratch](https://github.com/norhum/reinforcement-learning-from-scratch/)。
   - 进阶学习请查看 [Nous 关于 RL 环境的演讲](https://www.youtube.com/watch?v=zHaaivOQQGo)。
- **使用 MOE 进行科学研究研发**：一位成员正在进行**科学研究文献**和**跨领域知识综合**的研发。
   - 这是 MOE 数据集：[kaggle.com/datasets/allanwandia/moe-dataset](https://www.kaggle.com/datasets/allanwandia/moe-dataset)。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1368441826201178184)** (4 messages): 

> `Canon layers, 2D/3D convolution, DiT architecture, quantization quality, speech modality for duplex models` 


- **Canon Layers 极大增强 Token 上下文**：**Canon layers** 改善了 Token 的局部上下文表示，从而实现了局部上下文信息的高效混合，详见[这篇论文](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5240330)和[这条推文](https://x.com/ZeyuanAllenZhu/status/1918684257058197922)。
- **2D/3D Convolution 启发架构设计**：一位成员考虑在 **DiT architecture** 的图像生成中，探索使用带有小型空间算子的 **2D/3D convolution**，并直接与残差连接集成。
- **Convolution 与 Attention 的协同效应**：利用 **convolutions** 进行高效的局部处理，并利用 **attention** 进行全局上下文聚合，这种协同作用是 **CvT**、**CoAtNet** 和 **CMT** 等混合视觉架构中公认的原则。
- **Quantization 质量提升模态表现**：一位成员希望探索在 **duplex models** 的 **speech modality** 中，利用改进以及可能的 **quantization quality** 和 **QAT** 提升。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1368633182991417464)** (5 messages): 

> `AnySphere, Fundraising` 


- **AnySphere 寻求 9 亿美元融资引发关注**：据[这条 X 帖子](https://x.com/pavanjayasinha/status/1919037666428891392)报道，**Cursor** 的开发商 **AnySphere** 据传正在寻求 **9 亿美元** 的融资。这引发了人们的疑问：对于一家只有约 100 名员工的公司来说，是否有必要筹集如此巨额的资金。
   - 一位成员开玩笑说，AnySphere 在获得新融资后可能需要 **1000** 名员工，暗示了潜在的扩张或雄心勃勃的项目。
- **JakeABoggs 对 AnySphere 的评论**：JakeABoggs [对 AnySphere 发表了评论](https://x.com/JakeABoggs/status/1919329765464358967)。
   - 未提供更多细节。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1368441826201178184)** (4 messages): 

> `Canon Layers, Convolutional Architectures, DiT Architecture for image generation, Duplex Models in Speech Modality` 


- **Canon Layers 增强局部上下文**：[Canon layers](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5240330) 改善了 Token 的局部上下文表示，实现了局部上下文信息的高效混合，并可能将 **Training Data Efficiency** 提高 **1 倍到 2-8 倍**。
- **Convolutions 与 Attention 的协同**：一位成员考虑在用于图像生成的 DiT architecture 中，探索使用带有小型空间算子的 **2D/3D convolution**，并直接与残差连接集成，并指出了**用于高效局部处理的 convolutions 与用于全局上下文聚合的 attention 之间的协同作用**。
- **混合视觉架构的探索**：将混合视觉架构（如 **CvT**、**CoAtNet**、**CMT**）的原理应用于 **DiT** 等模型，对于图像生成任务似乎是合理的。
- **Speech Modality 的 Quantization 质量改进**：在验证了改进效果后，该用户考虑在 **duplex models** 的 speech modality 中使用 quantization 质量和 **QAT 改进**，并建议其他人也进行尝试。


  

---


### **Manus.im Discord ▷ #[showcase](https://discord.com/channels/1348819876348825620/1348823595505156137/1368154392120922194)** (2 messages): 

> `` 


- **话题占位符 1**：占位摘要句子 1。
   - 占位摘要句子 2。
- **话题占位符 2**：占位摘要句子 1。
   - 占位摘要句子 2。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1367960811296915599)** (253 条消息🔥🔥): 

> `AI detection tools, digital watermarks, Manus invitation codes, Free credits, LATAM` 


- ****AI Detection Tools****：成员们讨论了 **AI detection tools** 的工作原理，以及针对 Google 生成内容使用 [SynthID](https://ai.googleblog.com/2023/08/synthid-invisible-watermarks-for-ai.html) 的情况，它被描述为一种类似于证书的 **digital watermark**。
   - 一位成员分享了一个关于 [Technical Summary on Statistical Watermark Removal Techniques](https://cdn.discordapp.com/attachments/1349440650495398020/1368682467011072100/Technical_Summary__Statistical_Watermark_Removal_Techniques.mp4?ex=681a6de1&is=68191c61&hm=c352e6d83a4d9f71d56799152ec5c60428363402e647a2dcdef07ce977c0feb5) 的视频。
- ****Manus Invitation Code 寻找热潮升温****：许多成员正在寻找 **Manus invitation codes**，一些人分享了自己的邀请码以帮助他人访问该平台，例如 [这一个](https://manus.im/invitation/HO9UDIFNTLFB)。
   - 一位用户甚至描述了他们如何尝试让大学生通过其推荐链接注册，以赢得一件 **T-shirt**。
- ****Free Credits 现在每周刷新，还是说真的吗？****：一位成员注意到 **free credits** 正针对免费用户每周刷新，并展示了一个法语版仪表盘的 [截图](https://cdn.discordapp.com/attachments/1349440650495398020/1368625400774791218/36D0E8F4-1941-4039-B50E-32CC5342F3D3.jpg?ex=681a38bb&is=6818e73b&hm=0f457d280777e0c26cff955f680b14697bab379d9487e440e3ededefac2d7ec0&)。
   - 然而，其他人并没有看到同样的情况，怀疑这是教育版本的一部分，或者并非事实。至少有一位用户表示：*笑死，我半个会话的额度都耗在处理文本编辑器的问题上了*。
- ****LATAM 用户感叹缺乏负担得起的访问渠道****：成员们讨论了由于收入较低，**LATAM**（拉丁美洲）用户对 Manus 定价的承受能力。
   - 一位成员指出，**$39** 的首个套餐相当于阿根廷工资的 **10-15%**，并主张提供类似于 Photoshop 提供的折扣。
- ****柏林能源危机？！****：有一些关于柏林相对于德国其他地区充满活力的 AI 和 crypto 圈子的讨论。
   - 一位成员嘲讽道：*当国家其他地区在能源危机中挣扎，只为了给柏林提供成为赛博朋克城市所需的电力时，看来伟大的民主（dmcrcy）正在发挥作用*。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1367939283184517240)** (201 条消息🔥🔥): 

> `医学中的 LLMs，隐式 vs 显式模型学习，美国手语模型训练，Qwen 3 和 QwQ 模型，Grok 3.5 是假的` 


- ****LLMs 可能彻底改变医学****：一位成员建议，**基于 LLM 的建议**可能会极大地造福医生，其收益超过了过度依赖的潜在风险，并指出医学领域中**容易实现的目标（low-hanging fruit）**比 Web 开发领域更多。
   - 他们认为，*处理未知或新颖情况的能力*比单纯掌握事实更重要，而 **AI/ML** 正朝着这个方向发展。
- ****隐式与显式模型学习的结合****：一位成员提议将**隐式学习 (LLMs)**与**显式学习 (world models)**相结合，以展示 *G(z)* 的重要性，并插图说明了 [Generative Paradigm](https://cdn.discordapp.com/attachments/986699377257119794/1367950518495744042/m9.png?ex=681a6732&is=681915b2&hm=9062386e60377b471dec5a58e4df1a1f90b62730e5b5237bc29a0e250526df16&)。
   - 他们通过不同的 **ML 模型和范式**进行了说明。
- ****Qwen 3 和 QwQ 表现惊人****：**Qwen 团队**发布了 **Qwen 3** 和 **QwQ**，在许多任务上超越了更大的西方 **SOTA 模型**。
   - 一位成员指出，它在一套*包含各领域各类问题的内部标准问题集*上表现良好，并且由于计算密集度较低，它将取代 **R1** 成为日常编程助手。
- ****AI 的涌现与劳动力市场的命运****：在关于 **AI 毁灭论（AI doom）**潜力的讨论中，一位成员分享了一篇[关于劳动力市场的文章](https://arxiv.org/abs/2303.10130)，认为在讨论复杂系统时，**涌现行为（emergent behaviors）**经常被忽视。
   - 另一位成员则认为，围绕 **AI** 和**涌现**的炒作被夸大了，目前缺乏对生物系统和硅基系统智能进行比较的严肃研究。
- ****线性代数：深度学习幕后的英雄****：一位成员建议将**线性代数**作为 Deep Learning 的核心技能，而另一位成员提供了 [Stanford 和 MIT 视频讲座的链接](https://www.youtube.com/playlist?list=PLoROMvodv4rOpr_A7B9SriE_iZmkanvUg)。
   - 一位成员还表示：*尽你所能地使用 AI 来尽可能多地学习*。他们使用 **ChatGPT** 和 **Grok 3** 来学习数学。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1367986699531063419)** (12 条消息🔥): 

> `DEoT, AI 文本归一化, ChatGPT o3` 


- **社区关注新的 ArXiv 论文**：成员们计划在 45 分钟内进行语音通话，以审阅一篇新论文：[https://arxiv.org/abs/2504.07872](https://arxiv.org/abs/2504.07872)。
- **DEoT 思维过程可视化**：一位成员分享了一张插图，展示了 **DEoT** (*Dynamics of Effective Theories*) 的思维过程，详见 [xkps9eecnq3e1.png](https://cdn.discordapp.com/attachments/1045297868136779846/1368014669880754307/xkps9eecnq3e1.png?ex=6819fa31&is=6818a8b1&hm=be6dcb34696ee625dc656aba4a6d4197c73b2a8d54ebb1ae939a7a1e4d9581d5&)。
- **ChatGPT o3 专利状态检查**：**ChatGPT o3** 指出某项特定专利尚未发布。
- **AI 文本归一化为废话？**：一位成员讨论了网站上 AI 生成的文本（特别是关于 AI 相关的内容）如何趋向于归一化为“废话”，偏离了早期版本中人类编写的数据。
- **新的 ArXiv 论文即将发布**：一位成员分享了 ArXiv 上一篇新论文的链接：[https://arxiv.org/abs/2504.07389](https://arxiv.org/abs/2504.07389)。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1368014226215407747)** (27 messages🔥): 

> `Granite 4.0 Tiny Preview, Mamba-2/Transformer, Adblocker models, California's AI regulation SB-1047, Apple-Anthropic AI coding platform` 


- **IBM 的 Granite 4.0 Tiny Preview 采用混合架构**：IBM 发布了 **Granite 4.0 Tiny Preview**，它采用了全新的 **Mamba-2/Transformer 混合架构**，能够在消费级硬件（甚至是低于 350 美元的 GPU）上运行执行长上下文（**128K**）任务的并发会话。根据 [IBM 的公告](https://www.ibm.com/new/announcements/ibm-granite-4-0-tiny-preview-sneak-peek)，该模型还采用了细粒度的混合专家（**MoE**）模型，总参数量为 **7B**，推理时的激活参数仅为 **1B**。
- **加州 AI 监管法案纪录片发布**：分享了一部关于**加州 AI 监管法案 SB-1047** 的纪录片，可通过 [此 YouTube 链接](https://youtu.be/JQ8zhrsLxhI) 观看。
- **Apple 与 Anthropic 展开合作？**：据 [macrumors.com 的这篇文章](https://www.macrumors.com/2025/05/02/apple-anthropic-ai-coding-platform/) 报道，传闻 Apple 正与 Anthropic 合作开发一个 **AI coding platform**。
- **ReasonGraph 仓库与 HuggingFace Space 上线！**：分享了 **ReasonGraph** 项目及其 [GitHub 仓库](https://github.com/ZongqianLi/ReasonGraph) 和 [Hugging Face space](https://huggingface.co/spaces/ZongqianLi/ReasonGraph)。
- **Deepseek 的主导地位源于其卓越的 post-training**：尽管基础模型质量很高，但一位成员认为 **Deepseek** 的 post-training 更胜一筹，并表示 *“我不认为他们能赢过 Deepseek，纯粹是因为即便基础模型起步更好，糟糕的 post-training 也会拖后腿”*，参考了 [HuggingFace 上的 microsoft/MAI-DS-R1](https://huggingface.co/microsoft/MAI-DS-R1)。


  

---


### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1367966199627120731)** (1 messages): 

> `User Experience Research, Feedback on NotebookLM, Google products feedback, Opportunities` 


- **邀请用户协助塑造 NotebookLM 的未来**：用户被邀请加入一项**用户体验研究计划（user experience research program）**，以协助塑造 NotebookLM 及其他 Google 产品的未来。
   - 感兴趣的个人可以 [通过 Qualtrics 表单报名](https://google.qualtrics.com/jfe/form/SV_2cyuGuTWsEw84yG?utm_source=Forum&Q_Language=en&utm_campaign=Q2&campaignDate=April2025&referral_code=UXReCUq1425123)（耗时不到 2 分钟）提供反馈，并有机会抢先体验即将推出的功能，此外还能因付出时间而获得奖励。
- **参与 UX 研究，获取报酬**：参与用户体验研究计划的成员，如果参加了相关研究，将因其**时间投入和反馈**获得奖励，这创造了一个互利共赢的局面。
   - 该倡议被定位为一种**双赢**方案，在为用户提供影响产品开发机会的同时，也为其贡献提供了补偿。


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1367943646745202828)** (28 条消息🔥): 

> `播客长度限制、思维导图功能请求、Audio Overviews、使用 NotebookLM 进行研究、NotebookLM 的 Prompting 技巧` 


- ****NotebookLM 播客长度之谜****：用户报告播客时长不一致，从 **4-8 分钟**到长达 **40 分钟**不等，引发了关于如何控制长度的讨论。
   - 一位用户建议通过尝试不同的 Prompt 变体和丰富的源材料进行“试错”，并分享了一个用于创建关于“工具收敛性”（Instrumental Convergence）的详细长篇解释的 Prompt。
- ****思维导图缺少 Markdown 导出功能****：一位用户请求增加将 NotebookLM 思维导图导出为 Markdown 的功能，以便在其他思维导图应用中进行编辑。
   - 一名成员表示该功能目前尚不可用，促使另一位用户创建了一个[在思维导图节点中添加可点击来源指示器的功能请求](https://discord.com/channels/1124402182171672732/1368251917754568947)。
- ****揭秘 Audio Overview 的奥秘****：用户正在澄清如何从选定来源生成 Audio Overviews，并指出*笔记不能直接用于 Audio Overviews，但可以转换为来源*。
   - 据指出，免费计划每 **24 小时**仅允许生成 **3 个 AO**，一位用户发现**土耳其语**的时长比**英语**短，这表明可能存在特定语言的限制。
- ****NotebookLM：研究者的瑞士军刀****：一位研究前古诺斯异教斯堪的纳维亚的用户发现 NotebookLM *非常方便*，简化了研究流程并促进了专业文章内部的联系。
   - 另一位用户配合使用 NotebookLM 和 **2.5 Pro** 生成关于 **DeepMind 论文**的播客，整合演示和源代码，以便在通勤期间听取综合报告。
- ****交互模式：消失的选项****：一位用户询问如何找到“交互模式”选项，另一位用户回答说，*音频生成后该选项会自动出现*。
   - 还明确了可以为 LM 选择/取消选择要处理的来源——**1 个、部分或全部**。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1367987953371971626)** (138 条消息🔥🔥): 

> `Gemini 2.5 Flash, AI 奉承行为, Gemini 升级, NotebookLM 音频生成` 


- **Gemini 2.5 Flash 导致奉承行为**：用户注意到 **Gemini 2.5 Flash** 表现出类似于 **OpenAI** 的“奉承行为”（sycophantic behavior），例如过度赞扬问题并提供冗长的介绍。
   - 一位用户通过添加自定义指令“永远不要谈论你自己，也不要提供介绍”找到了解决方案。
- **Gemini 升级适用于所有版本**：一位用户询问 **Gemini 升级**是仅适用于 **Plus** 版本还是适用于产品的所有版本。
   - 一名成员回答道：“所有版本”。
- **Audio Overview 模型与 Gemini 2.5 Flash 不同**：**Audio Overview** 功能可能由与 **Gemini 2.5 Flash** 不同的模型驱动，因为 **Gemini 2.5 Flash** 不具备原生音频生成能力。
   - 一名成员建议 **Gemini 2.5 Flash** 可能会分析来源并为 Audio Overview 功能编写脚本。
- **Audio Overviews 的自定义限制**：用户讨论了 **Audio Overviews 的自定义**仅限于生成初始 **Audio Overview** 并使用 Prompt 修改输出。
   - 据指出，交互模式允许引导讨论，但这些贡献不会被记录在下载的 **Audio Overview** 中。
- **功能请求：文件夹、标签、笔记本搜索**：一位用户提交了针对 **NotebookLM** 的详细功能请求，建议实现**文件夹/分类、标签和笔记本列表搜索**等功能。
   - 这些组织功能将显著提升用户体验，并使 **NotebookLM** 作为长期知识管理工具更具实用性。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1367972898463285298)** (98 messages🔥🔥): 

> `MCP Auth Spec, Xcode AI Anthropic, AI Salesperson, Deep Research Reports, Decagon ARR` 


- **Apple 据传将联合 Anthropic 为 Xcode 提供 AI 支持**：传闻称 **Apple** 和 **Anthropic** 正在合作构建一个 **AI 驱动的 Xcode** vibe coding 平台（[Bloomberg 文章，通过 archive.is 访问](https://archive.is/2025.05.02-195610/https://www.bloomberg.com/news/articles/2025-05-02/apple-anthropic-team-up-to-build-ai-powered-vibe-coding-platform)）。
   - 一些用户对这一变化表示欢迎，但也有人称 *Apple 成为了新的 IBM*，并开玩笑说 *一个无代码的 Swift 框架会表现得更好*。
- **Krea 的 GPT Paint 走红**：**Krea** 推出了 **GPT Paint**（[推文](https://x.com/krea_ai/status/1917949632069456220)），一名成员分享了他们对其粗略实现的猜测：使用 **GPT-image-1** 从输入图像和画布描述中提取指令，且未使用 ControlNet。
   - 他们指出了一些潜在的改进方向，例如将图像层的坐标信息输入到 Prompt 中，以及整合“移除背景”的建议，并附上了他们的[分析推文链接](https://x.com/shacrw_/status/1918024366379471359)。
- **全自动化公司：Agentic DAOs?**：Dwarkesh 分享了一篇关于*全自动化公司未来形态*的精彩文章和视频（[链接](https://www.dwarkesh.com/p/ai-firm)），内容大量借鉴了 [Gwern 的 backstop](https://gwern.net/backstop) 并提出了一个问题：*Agentic DAO?*
   - 相关视频链接：[YouTube 视频](https://www.youtube.com/watch?v=bJD1NpdMY5s)。
- **Exa 回归并带来基础的 BM25 优化**：**Exa** 重回 X 平台并发布了一篇关于 **BM25 优化** 的博客文章（[Exa 博客文章](https://exa.ai/blog/bm25-optimization)）。
   - 未提供二次摘要。


  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1367953482581151894)** (49 messages🔥): 

> `A2A vs MCP, Discord Stream Issues, Google's Protocol Background` 


- **A2A 协议实践**：一名成员分享了 [A2A GitHub 仓库](https://github.com/google/A2A) 的链接，另一名成员问道：*真的有人在实际项目中使用 A2A 吗，哈哈？*
   - 原发布者表示他们不得不在播客中解释 **MCP** 和 **A2A**，并感叹 *我差点没命了（太难解释了）*。
- **A2A 与 MCP 的协议之战拉开序幕**：分享了一篇关于 **A2A** 和 **MCP** 的有趣文章（[koyeb.com](https://www.koyeb.com/blog/a2a-and-mcp-start-of-the-ai-agent-protocol-wars)），并评论道 *我确信现在要开战了……*
   - 其他人补充道：*我认为 A2A 更适合流式传输/异步场景*，而 *MCP 更适合那种单次触发（oneshotty）的感觉*。还有评论指出 *A2A 是一个 MCP 封装*。
- **Discord 直播对“盲人”观众失效**：几位成员报告了观看 Discord 直播时的问题，其中一人评论道 *有点确信 Discord 在观众超过 20 人后就会崩溃，而且由于某种原因非常不喜欢 Mac 的屏幕共享*。
   - 在加入 Stage 之前可以看到截图预览，但共享屏幕对某些人来说仍然无法访问，导致 *我们在评论区为“盲人”观众举办派对*。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1368385505259163800)** (32 messages🔥): 

> `Mojo 安装、Mojo 与 MAX 捆绑、Mojo 文件扩展名、UV、Pip 与 Mojo 项目、Mojo 中的 Traits 和 Fields` 


- **困惑的新手询问如何安装 Mojo 😅**：一位新用户在按照[官方安装指南](https://docs.modular.com/magic/)操作后，询问是否有更简单的 Mojo 安装方式，更倾向于类似于在 **Linux 上创建 C 项目**的工作流。
- **Mojo 与 MAX 捆绑节省的磁盘空间有限 💾**：一位成员澄清说，**Mojo 和 MAX 是捆绑在一起的**，因为它们共享许多组件，将它们分开只能节省几百 MB 的磁盘空间。
   - 他们将 **MAX** 比作 *GCC 和 Clang 中的 OpenMP、OpenACC 和 OpenCL 功能*。
- **关于 Mojo 扩展名别名的激烈讨论 🔥**：一位用户建议将 Mojo 文件扩展名缩短为 **.mo**、**.mj** 或 **.mm**，以便于命令行操作，同时也承认这并非必选项。
   - 另一位用户开玩笑地建议使用 **.🔥** 这种 Emoji 扩展名，并指出大多数人使用 Tab 补全。
- **Mojo 项目的 Pip 安装即将推出 📦**：鉴于 Mojo 与 Python 的兼容性，一位用户询问了关于在 Mojo 中使用 **UV 或 Pip** 的情况。
   - 一位成员确认 **Pip 安装即将推出**，但目前 Mojo 是通过 **conda 包**分发的。
- **Trait Fields 无法实现，建议改用 getter 🤔**：一位用户询问了关于 **Fields 和 Traits 默认实现**的开发计划，这对于构建抽象非常有用。
   - 一位成员表示，*Traits 中的 Fields 不可能实现，因为那样你就可以对 Float32 之类的类型添加字段，然后看着一切崩溃*，并建议改用 getter。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1368009621389967361)** (115 messages🔥🔥): 

> `Mojo 中的 constexpr、consteval、运行时不存在的函数、全局变量 (Globals)` 


- **讨论通过 `consteval` 进行编译时求值**：成员们讨论了实现 `consteval` 或 `@parameter_block` 来处理编译时的复杂初始化和状态类型模式 (typestate patterns)，认为这可以提高 Mojo 的易用性，并避免将复杂代码封装在函数中。
   - 一位成员建议，长期目标是将控制流语句从解析器中移出，并将更多内容放入库中以实现计算跳转 (computed goto)。
- **编译时的堆分配**：一位成员询问了编译时堆分配的位置，另一位成员澄清说编译器负责代码生成，可以使用 [LLVM IR](https://llvm.org/docs/LangRef.html) 进行检查。
   - Modular 团队成员鼓励社区成员撰写一篇解释编译时的博客文章，并表示愿意提供发布平台。
- **Mojo 解决 FPGA 编程问题**：一位成员提到，与 HLS (高层综合) 相比，Mojo 可以为 **FPGA 编程**提供更好的解决方案，解决了现有硬件描述语言 (HDL) 在设计时往往未考虑编程语言理论的缺陷。
   - 另一位成员提到了一个 [YouTube 视频](https://www.youtube.com/watch?v=ee01_yHjs9k)，并建议 Mojo 可以利用 **CIRCT** (可重构计算编译器基础架构工具集) 方言来增强硬件设计能力。
- **全局变量 (Globals) 不可靠**：在一位成员发现全局变量可以使用后，另一位成员分享说，在打包代码时，顶级变量仍然存在 **UB** (未定义行为)，并建议改用 stdlib.ffi 的 `_Global` 结构体。
   - 成员们还指出了 [GitHub 上现有的 Issue](https://github.com/modular/modular/issues/4491)，显示全局变量并不可靠，且 Modular 团队目前并未优先修复全局变量问题。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1367962455569072308)** (131 条消息🔥🔥): 

> `Claude 资源作为附件、Claude 在固定和订阅方面的限制、开源模型与 OpenAI 模型、使用 MCP Inspector 的 CLI 模式测试 Streamable HTTP、使用 PM2 管理 MCP 服务器` 


- **Claude 将资源视为附件**：成员们讨论了如何让 **Claude** 使用资源，并指出资源的功能与附件类似，但在注入上下文的方式上略有不同。
   - **Claude Desktop (CD)** 中的支持有限，缺少诸如将资源*固定（pinning）*到上下文或*订阅（subscribing）*更新等功能，如附图所示：[图片1](https://cdn.discordapp.com/attachments/1312302100125843479/1367979428193370272/image.png?ex=681a821f&is=6819309f&hm=94e749cf1c7a3ed3eeafedb55cee3238c9260f6b0259c83fad908ae63a7b0158&), [图片2](https://cdn.discordapp.com/attachments/1312302100125843479/1367979428491038931/image.png?ex=681a821f&is=6819309f&hm=54b10d33be232852cddfea92b3d4e6ddba8e769a612e233a1e6bd647e9340412&), [图片3](https://cdn.discordapp.com/attachments/1312302100125843479/1367979555255488532/image.png?ex=681a823d&is=681930bd&hm=97d03a94e166272fcaa0f34db524010a7189434bc56052f9d88f94430a858f91&)。
- **CLI 模式下 Streamable HTTP 测试困难**：一位成员正在构建一个 **MCP server**，通过 streamable HTTP (TypeScript SDK) 提供工具。他们发现虽然该功能在 `mcp-remote` 和 Claude Desktop 上可以端到端运行，但 **MCP Inspector 的 CLI 模式** 尚不支持 streamable HTTP 传输。
   - 他们注意到最近有一些 PR 已被合并，但 Inspector CLI 中启用此功能的任务尚未实现，并提供了尝试在 CLI 中使用 streamable HTTP 时收到的错误示例：*Failed to connect to MCP server: SSE error: Non-200 status code (404)*。
- **增强版 Python SDK 改进工具声明**：一位成员分享了一个简化 Python MCP 工具基础开发的实用程序，指出标准 SDK 与 TypeScript 相比代码更加冗长。
   - 该工具包含一个带有 `declare_tool` 装饰器的 `EnhancedServer`，减少了在列表和调度函数实现上的样板代码，类似于 TypeScript 的声明方式，并建议向 Python SDK 提交 PR：[gist.github.com](https://gist.github.com/isaias-b/5b67ef499e497f21c9a9481b6a266f8c#file-mcp_commons-py)。
- **模型越小，规则遵循越弱**：一位成员询问关于在 **Cursor** 中使用 MCP 的问题，特别是使用像 *cursor-small* 这样的小型 Agent 模型。另一位成员回应称，小型模型通常不擅长遵循规则。
   - 建议至少使用 **Gemini flash** 以获得更好的效果，或者考虑使用付费版的 Cursor，因为小型模型在处理填满上下文的大型系统提示词（system prompts）时会感到吃力，从而导致混乱。Anthropic 开发的 [MCP Inspector 工具](https://github.com/modelcontextprotocol/inspector) 也会有所帮助。
- **通过其他 LLM 解决上下文长度问题**：成员们讨论了 Claude 因工具过多而达到上下文限制的问题，并建议尝试其他支持更高上下文长度 LLM 的 MCP 客户端。
   - 一位成员报告称他们在 **Qwen3:14B** 上的体验不佳，表示它在基于 ReAct 框架的 RAG Agent 应用中不擅长遵循指令，而 **OpenAI** 和 **Gemini** 的表现更好。


  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1368244237291753483)** (16 条消息🔥): 

> `MCP Language Server, Biothings MCP, FastMCP Tool Timeouts, Langchain App via SSE, MCP Task Scheduler` 


- **MCP Language Server 发布稳定版本**：[MCP Language Server](https://github.com/isaacphi/mcp-language-server) 的第一个稳定版本已发布，通过 **get definition**、**references**、**rename** 和 **diagnostics** 等语义工具帮助客户端更轻松地导航代码库。
- **用于衰老研究的 Biothings MCP 正在开发中**：生物学 MCP 服务器的工作正在进行中，[longevity-genie/biothings-mcp](https://github.com/longevity-genie/biothings-mcp) 是第一个加入的项目，旨在为从事 **衰老研究（ageing research）** 的生物学家和生物信息学家提供 MCP 工具箱。
- **FastMCP 新增工具超时功能与漏洞研究**：[FastMCP](https://github.com/punkpeye/fastmcp/releases/tag/v1.24.0) 刚刚增加了工具超时功能，此外 [MCP: May Cause Pwnage](https://blog.jaisal.dev/articles/mcp) 中揭示的漏洞研究阐明了 MCP 文档中的警告。
   - *那个警告之所以存在，是因为我和朋友进行了一些有趣的漏洞研究。*
- **通过 SSE 为 Langchain 设置 MCP Server**：分享了一篇关于设置 MCP 服务器并作为客户端通过 **SSE** 连接到 **Langchain app** 的博客文章，专门为 MCP 初学者编写，地址：[santiagodcalvo.substack.com](https://santiagodcalvo.substack.com/p/bridge-the-gap-exposing-fastapi-endpoints)。
- **MCP Task Scheduler 发布**：[MCP Task Scheduler](https://github.com/PhialsBasement/scheduler-mcp) 允许直接从 **Claude** 调度提醒、API 调用和 shell 执行。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1367973138289266859)** (37 条消息🔥): 

> `LLM Hallucinations, Efficient Jailbreaks, ML Subreddit, Independent Research in AI/ML, Deepseek-R1 GPUs` 


- **LLM 中的幻觉（Hallucinations）：诱因与缓解**：一位成员有兴趣研究 LLM 中的 **hallucinations**，特别是预训练如何诱发它们，以及缓解方法，例如 [训练激活探测器 (activation probe)](https://link.to/activation-probe) 来预测答案的正确性。
   - 他们提议探索激活探测以识别幻觉，并研究减轻幻觉的训练方法或损失函数。
- **为对抗性训练构建越狱（Jailbreaks）**：一位成员的目标是实现高效创建 LLM **jailbreaks** 的方法，用于对抗鲁棒性训练，并引用了 [低概率估计 (low-probability estimation)](https://www.alignment.org/blog/low-probability-estimation-in-language-models/) 作为例子。
   - 鉴于其在 ICPC 和 Kaggle 竞赛中的背景，该用户希望为正在进行的、能发挥其作用的项目做出贡献。
- **LocalLLaMA 是最顶级的 ML Subreddit 吗？**：成员们讨论了 *localllama* 是否是 **ML 领域** 最好的 subreddit，一些人建议将 [r/MachineLearning](https://www.reddit.com/r/MachineLearning/) 或 **Twitter** 作为替代方案。
   - 一位成员肯定地表示 *目前确实是* 最好的。
- **开始独立 AI/ML 研究：建议分享**：一位成员寻求关于开始独立 **AI/ML 研究** 的建议，另一位成员建议实现一篇论文并进行实验。
   - 还有成员推荐了 **fastai**、**cs transformers** 和 **Eureka Labs** 等 AI/ML 课程来掌握基础知识，然后建议挑选一篇论文并尝试复现它。
- **Deepseek 的 GPU 分配：推理 vs 训练**：一位成员询问了关于 **Deepseek-R1 的 GPU** 如何在推理和训练之间分配的细节。
   - 有成员建议，向高端 LLM 咨询可能比搜索查询更可靠，但警告不要在没有来源确认的情况下盲目相信 ChatGPT 的回答。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1367992800456474745)** (82 条消息🔥🔥): 

> `Weight Decay 与 Learning Rate 的耦合、灾难性遗忘、Softpick Attention、New Physics of LM 论文、LLMs 与认知` 


- **解码 Learning Rate 与 Weight Decay 的关系**：会议指出，*你的 **LR** 和你的 **WD** 是紧密耦合的，如果 **WD** 设置不当，你的模型会严重崩溃*，这对于 AI 模型训练者来说非常重要。
   - 推理过程包括根据 **LR** 和 **WD** 的设置，计算每个训练 epoch 后遗忘的旧训练样本百分比。
- **灾难性遗忘：是否过于简化？**：一位成员建议，神经网络中的 **catastrophic forgetting**（灾难性遗忘）概念可能简化了不同类型的知识损失，例如遗忘特定示例与抽象表示的退化。
   - 他们分享了一些由 Claude 生成的想法，即遗忘的补救措施（如 replay buffers、正则化或架构更改）可能需要针对这些不同类型的知识损失采取不同的对策。
- **探索更大规模的 Softpick Attention**：一位成员提到，本周将发布一份预印本，可能有助于证明 **Softpick attention** 在更大规模下有效，该研究使用蒸馏技术将 **Qwen72B** 等大模型转换为 **RWKV** 变体。
   - 他们建议使用该技术尝试将 **Qwen 7B** 转换为 **Softpick attention** 并观察效果；一位用户询问：*你是否将 softpick 与 [off by one attention](https://www.evanmiller.org/attention-is-off-by-one.html) 进行了比较？*
- **“New Physics of LM” 论文发布**：“New Physics of LM” 论文刚刚发布（[https://x.com/ZeyuanAllenZhu/status/1918684257058197922](https://x.com/ZeyuanAllenZhu/status/1918684257058197922?t=cLSpFkSuTHqwkV5nGahnJw&s=19)），成员们讨论了在 **MLP** 中添加 **conv2D**，以极少的参数增加显著提升任何规模 **ViT** 的性能。
   - 有人指出，该论文的质量与普通论文无异，只是在主张上更加夸张，并且考虑到 **MHA** 是同类中的首创，它能维持至今表现依然稳健令人惊叹。
- **LLMs：它们到底知道什么？**：一位成员询问 **LLMs** 在多大程度上能感知自己知道和不知道的内容，另一位成员指向了引用 [arXiv:2305.18153](https://arxiv.org/abs/2305.18153) 的论文以及 Anthropic 在该领域的一些工作。
   - 建议从机械论的角度来看，字面答案是它们并不知晓，而是完全依赖上下文线索来理解是否应该知道某些内容（人类在某种程度上可能也是如此）。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1367992009540894841)** (21 条消息🔥): 

> `Transformers 中的 RoPE、Transformers 的早期层、抽象推理的机械可解释性` 


- **RoPE 在 Layer 0 中创建显式模式**：会议指出，在 Transformer 模型的 layer 0 中看到的显式模式是结合位置编码与注意力权重时的典型行为，特别是由于 [RoPE (Rotary Position Embedding)](https://arxiv.org/abs/2104.09864) 导致的。
   - 一位成员提到 *用 rope 调制注意力亲和力* 自然会产生模式，但有趣的是它在 layer 0 中表现得如此显式。
- **Transformers 模仿 CNN 层行为**：一位成员指出，**CNNs** 中“早期层检测边缘，后期层学习特征”的行为可能与 Transformer 中的类似现象类比，特别是对于音乐等模态。
   - 他们想象这可能被用于检测歌曲中的节奏、节拍或其他基础重复特征。
- **抽象推理的机械可解释性研究**：一位成员询问关于 **LLMs** 中抽象推理原理（如常识推理或数学推导）的 **mechanistic interpretability**（机械可解释性）研究。
   - 另一位成员推荐了 [Tegmark 关于数学的 *The Pizza and the Clock* 论文](https://arxiv.org/abs/1401.0984)，但提醒说，更具野心的方法在与模型实际行为的对应程度上可能存在漏洞。
- **Transformer Circuits 暗示机械可解释性**：提到了涉及机械可解释性组件的研究论文，如 **grokking**、**BIG-Bench** 和 **content bias reasoning**，并引用了 [Anthropic transformer circuits](https://transformer-circuits.pub/2023/monosemantic-features/index.html) 和 [Towards Monosemanticity](https://arxiv.org/abs/2312.03824) 作为相关参考。
   - 对于公式二元框架，建议转向物理学和微分几何。


  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1368188704702599249)** (3 messages): 

> `lm_eval issues, DeepSeek-R1-Distill-Qwen-32B, vllm vs hf inference, gsm8k, mmlu` 


- **DeepSeek 模型的 lm_eval 异常表现**：一位用户报告了在使用 **hf** 和 **vllm** 推理测试 **deepseek-ai/DeepSeek-R1-Distill-Qwen-32B** 模型时遇到的 **lm_eval** 问题。
   - 具体而言，他们观察到虽然 **vllm** 能很好地处理 **gsm8k** 等生成任务，但 **hf** 推理速度极慢，且未能充分利用 A100 的 GPU 性能，功耗仅为 170W，而该显卡的最大功率为 300W。
- **HF 推理速度慢得离谱**：对于 **gsm8k** 等生成任务，**vllm** 推理表现良好，但观察到 **hf** 推理速度慢得离谱，且未能完全占用 GPU。
   - 具体来说，在 A100 上使用 **hf** 时，显卡功耗仅为 **170W**（最大功率为 300W）。
- **VLLM 与 HF 推理的功耗对比**：用户还注意到，在进行 **mmlu** 等 Log-likelihood 任务时，**vllm** 推理比 **hf** 慢，但两者都能按预期达到 250W 的 GPU 功耗。
   - 用户提供了用于 **vllm** 和 **hf** 推理的 **lm_eval** 命令行参数，寻求解决性能差异的帮助。
- **在 lm_eval 中利用更多 pass**：用户询问如何在 evaluation harness 中实现 **pass@10** 而不仅仅是 **pass@1**。
   - 在给定上下文中未提供解决方案或进一步的见解。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1367950649236652143)** (71 messages🔥🔥): 

> `AI Developer Survey, Code Conversion Tool, HuggingFace LM Support, Web3 Game Beta Testers, DSPy.GRPO Release` 


- **AI 开发者请填写 Gensee 的免费 AI Infra 调查！**：来自 Gensee AI 的 Yiying Zhang 正在进行一项针对 **AI 开发者**、**学习者**和**管理者**的[调查](https://forms.gle/PMZdBbqBUJ9jE5Sb7)，旨在塑造 **AI infrastructure** 的未来。
   - 参与者可以了解 **GenseeAI 的测试计划**，该计划提供了一个用于部署和优化 **AI Agent 和工作流**的免费平台，并有机会获得 **$25-$50 的礼品卡**。
- **Cobol 难题：需要转换遗留代码的工具**：一位成员正在寻求关于创建将大型遗留大型机代码文件转换为 **COBOL + JCL** 的工具建议，由于缺乏现成的解析器或 tree-sitter 集成，在分块（chunking）和保留上下文方面面临挑战。
   - 该成员通常的方法是使用带有语言集成的 tree-sitter 来进行正确的块提取。
- **DSPy GRPO：在线 RL 优化器发布！**：DSPy 推出了 `dspy.GRPO`，这是一个针对 **DSPy 程序**的**在线 RL 优化器**，允许用户原样**优化其 DSPy 代码**，即使是**复合多模块程序**也可以，详见[此 X 帖子](https://x.com/lateinteraction/status/1919428454761553994)。
   - 该版本由 **Noah Ziems**、**Lakshya Agrawal** 和 **Dilara Soylu** 领导，需要 GPU，并引发了关于结合 **SFT/RL + Prompt Optimization** 以及与云提供商集成进行微调的讨论。
- **S-LoRA 与 GRPO 的兼容性正在评估中**：成员们讨论了将 **s-LoRA** 与 **GRPO** 耦合的可能性，并指出有一个支持 **LoRA** 的标志。
   - 核心思想是，如果按任务调整权重，那么每个任务保存一个 LoRA，从而实现在单个模型和 GPU 上部署 **1000 个 LoRA**。


  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/)** (1 messages): 

dbreunig: 优秀的端到端示例：https://duarteocarmo.com/blog/evals-are-all-you-need.html
  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1368504494236762233)** (47 条消息🔥): 

> `Agent 测试 PR，Mask 计算 Bug，Tokenizer 支持，LLMs` 


- **侵入式 Agent 测试引发关注**：一位成员对 [Agent 测试的 PR](https://github.com/pytorch/torchtune/pull/2671) 表示担忧，认为其具有“侵入性”，并暗示其目的是收集训练数据而非修复问题。
   - 另一位成员幽默地回顾了之前的经历：一位贡献者拒绝为其更改提供验证图表，并表示 *要么按原样保留这个免费贡献，要么就关闭它*。
- **发现潜在的 Mask 计算 Bug**：一位成员报告了在处理带有 Padding 的 Mask 计算时可能存在的 Bug，特别是在 Prompt 中间存在 Padding 的情况下，并寻求一致性检查，以确认在 Prompt 中间放置 Padding 的做法是否“疯狂”。
   - 他们观察到 `get_causal_mask_from_padding_mask` 总是将对角线设置为 True，这导致模型会关注到 Padding 部分。
- **探索简化样板代码生成的模型添加方式**：一位成员提议利用 `config.json` 中的信息为模型相关的样板代码实现 Codegen（代码生成），并允许直接使用来自 Transformers 的组件，以加速对新模型的支持。
   - 其他人对这一概念表示感兴趣，但警告不要过度使用 Codegen，建议重点简化 Tokenizer 支持和与 HF 模型的一致性检查等挑战性环节；他们还建议为 **LLMs** 生成精心设计的 Prompt 来处理样板任务，直到通过单元测试。
- **针对新模型的“支持”定义展开讨论**：团队思考了对于新模型而言“支持”意味着什么，哪些功能应该**开箱即用**，并权衡支持新模型速度与 Torchtune 特性（如 Activation Offloading 和 Tensor Parallelism）带来的收益。
   - 有人建议 RFC 应该首先回答一个问题：*在 Torchtune 中支持一个模型意味着什么*？


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1368294183852576898)** (1 条消息): 

> `Physics of LLMs` 


- **Physics of LLMs 视频系列**：一位成员分享了一条 [推文](https://x.com/zeyuanallenzhu/status/1918684257058197922?s=46)，宣布 **Physics of LLMs** 视频系列有了新章节。
   - 目前还没有视频，只有推文。
- **Physics of LLMs 视频系列：更新**：在推文发布后，**Physics of LLMs** 系列备受期待的视频章节正被热切关注。
   - 许多成员都在关注这个系列。


  

---


### **Torchtune ▷ #[rl](https://discord.com/channels/1216353675241590815/1360680363885854841/1367984673632026674)** (3 条消息): 

> `CI 改进，新功能` 


- **Merge Request #2637 成功落地！**：Merge Request [#2637](https://github.com/pytorch/torchtune/pull/2637) 已获批准。
   - 成员们祝贺用户 <@651621413093900299> 和 <@1184909646771785792> 完成了该任务。
- **持续集成（CI）极具挑战**：一位成员提到了设置持续集成的难度。
   - 他们调侃道：*“最后一公里确实很难，哈哈 😄 这些东西的 CI 并不容易”*。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1367942909890138323)** (5 条消息): 

> `O3 vs Claude 3.7 评估，使用 LlamaParse 的 AI SDRs，RAG 生产经验，LlamaIndex Pull Request Agent，大型 MCP 黑客松` 


- **LlamaIndex 评估 OpenAI 的 o3 与 Claude 3.7**：LlamaIndex 正被用于在一项新的基准测试中对比评估 **OpenAI 的 o3** 与 **Claude 3.7**，更多详情请见 [此处](https://t.co/djycsksHDX)。
- **LlamaParse 将 AI SDRs 的上手时间缩短至几天**：**11x_official** 使用 LlamaIndex 通过摄取多种文档类型来自动化入职流程，从而改进销售开发，更多内容见 [此处](https://t.co/ChZuUXKKbl)。
   - 这使他们能够扩展外呼活动，完整案例研究详情见 [此处](https://t.co/7vIE23DlkV)。
- **ContextualAI 分享 RAG 生产环境的 10 条宝贵经验**：ContextualAI 的 Douwekiela 分享了将 **RAG** 投入生产的 10 条宝贵经验，关键细节见 [此处](https://t.co/GYzpPDvpAj)。
   - 来自 aiDotEngineer 的视频强调：*围绕 RAG 系统构建的周边系统更为重要*。
- **LlamaIndex 与 Composiohq 合作创建 Pull Request Agent**：Composiohq 使用 LlamaIndex 创建了一个审查 Pull Request 的 Agent，并配备了由 Replit 生成的 UI，查看实现方式请点击 [此处](https://t.co/3ZORZZs1rR)。
- **LlamaIndex 赞助特拉维夫大型 MCP 黑客松**：LlamaIndex 正在赞助由 aitinkerers 在特拉维夫举办的大型 MCP 黑客松，重点是构建支持 Agent 间通信的 **MCP 驱动应用**，更多尝试请见 [此处](https://t.co/gq1L30cgfE)。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1367952450379845762)** (29 messages🔥): 

> `RAG accuracy, NLP API, LlamaIndex Gemini bug, Legacy mainframe code to Cobol, Lovabe Cursor Expert` 


- **RAG 检索准确率测试**：一位成员就围绕 **300 页 PDF 文档**构建的 **RAG 流水线**的准确性测试寻求建议。
   - 另一位成员建议使用 [LlamaIndex evaluation tools](https://docs.llamaindex.ai/en/stable/module_guides/evaluating/)，并特别推荐从 **Ragas** 开始进行检索器测试。
- **NLP 到 API 聊天 Agent 设计**：一位成员正在寻找一种*经过验证的设计*，能够在聊天延迟范围内将自由文本映射到经过验证的多步 **API 调用**，寻求针对体育等波动数据领域的真实成功案例或代码示例。
   - 他们正在考虑将 **API 负载预先转换为用于 RAG 的向量数据库**，并从 embeddings 中回答，而不是进行实时调用，以提高可靠性和新鲜度，但也就权衡利弊征求建议。
- **通过深拷贝修复 LlamaIndex Gemini Bug**：一位成员发现 `llamaindexgemini.achat()` 会修改原始的 system prompt，并分享了[一个临时修复方案](https://github.com/run-llama/llama_index/pull/18616)，即在 `gemini_base.py` 的 `achat` 中添加深拷贝。
   - 根本原因在 `gemini_utils.py` 的 `def merge_neighboring_same_role_messages` 中被发现，因为 Gemini 会将 system 角色转换为 user 角色，但在合并时没有创建消息副本，而是直接合并。
- **将旧版大型机代码转换为 Cobol**：一位成员正在寻求创建一种工具，将旧版大型机代码（非 Cobol）转换为 **Cobol + JCL**，在缺乏现成解析器或 tree-sitter 集成的情况下，面临分块和保留上下文的挑战。
   - 一位成员建议使用 **Gemini** 在注释中生成每个函数及其依赖项的详细描述，然后分别传递每个函数以转换为 Cobol。
- **急需 Lovabe+Cursor 专家**：有人急需一名 **Lovabe+Cursor** 专家协助工作 2 周。
   - 他们需要专家能够全身心投入项目，并利用 **AI** 快速开展工作。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1367959228660191323)** (28 messages🔥): 

> `VRAM vs RAM, PDF upload issues, LaTeX support, Qwen 3 integration, LocalDocs feature` 


- **VRAM 还是 RAM？用户提问**：一位用户询问讨论中提到的 **RAM 需求**在涉及 **GPU** 时是否指 **VRAM**，并澄清在 **GPU** 上运行模型时使用 **VRAM**，否则使用 **RAM**。
- **PDF 上传仍然存在问题？**：用户在**直接在聊天中上传 PDF 文件**时仍遇到问题，而一些成员澄清 **LocalDocs** 功能仍然是使用 PDF 的推荐方式，因为聊天中直接上传 PDF 尚未支持。
- **LaTeX 支持仍然缺失？**：用户表达了对 **GPT4All** 支持 **LaTeX** 的持续关注，强调了其对于 **Qwen 2.5 math** 等模型的重要性，但目前尚无进展报告。
   - 一位用户提到在需要 **LaTeX** 时可以使用 **Kobold.cpp** 作为替代方案。
- **Qwen 3 支持正在开发中？**：一位用户询问了 **Qwen 3** 支持的时间表，另一位用户已经通过服务器上的远程 **Ollama** 在 **GPT4All** 中使用 **Qwen3**。
   - 另一位用户指出 **Qwen3-30B-A3B** 在 CPU 上运行非常快，但仍需要更长时间才能稳定，因为目前还存在一些 bug。
- **自定义模型和 PDF 摄取：可行吗？**：一位用户询问是否可以构建自定义模型来摄取 **PDF** 并回答问题，但另一位用户澄清 **GPT4All** 使用的是基于 **RAG 的方法**及其 **LocalDocs** 功能，而不是让模型直接“摄取”数据。
   - 该用户补充道，*针对特定文档和用例对模型进行 finetuning 是可行的，但不能保证效果会比基于 RAG 的解决方案更好*。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1367956381378019530)** (15 messages🔥): 

> `Lab Deadlines, Lean-lang.org Issues, Wayback Machine, Network Issues` 


- **实验截止日期定于 5 月 31 日**：据一位成员称，所有作业的截止日期为 **PDT 时间 5 月 31 日晚上 11:59**。
- **部分用户的 Lean-lang.org 链接失效**：有成员报告在加载 Lab 1 资源时遇到问题，特别是 [此页面](https://lean-lang.org/functional_programming_in_lean/getting-to-know.html) 和 [另一个页面](https://leanprover.github.io/theorem_proving_in_lean4/tactics.html)。
   - 其他成员在 **Chrome, Safari 和 DuckDuckGo** 上均无法复现该问题。
- **Wayback Machine 派上用场**：一位成员建议使用 **Wayback Machine** 访问失效链接，并提供了 [此快照](https://web.archive.org/web/20250410002159/https://lean-lang.org/functional_programming_in_lean/getting-to-know.html)。
   - 最初的问题报告者确认 **Wayback Machine** 的方案奏效。
- **疑似网络问题**：成员们推测 **网络问题** 可能是链接失效的原因。
   - 报告链接失效的用户表示，他们无法在移动端直接访问该网站，但可以通过 Discord 的重定向访问，这表明情况有些*奇怪*。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1368263726142324806)** (4 messages): 

> `Lecture 6, Multimodal Autonomous AI Agents, AgentX, MCP protocol, LM finetuning` 


- **Lecture 6 讨论安排**：关于 **Lecture 6: Multimodal Autonomous AI Agents** 的讨论定于 PT 时间 5 月 3 日星期六 7:30 (UTC-8) 举行。
   - 会议还将讨论 **AgentX** 项目，包括 **MCPx: Extensions to the MCP protocol**。
- **Keynote 访问权限可能已开放**：一位成员表示另一位成员将检查 [此处](https://discord.com/channels/1280234300012494859/1282785079919251577/1366581016554242079) 的 Keynote 访问权限。
   - 未提供更多细节。
- **投资组合优化会议**：安排了一场关于 **Multi-Hypothesis Prediction for Portfolio Optimization: A Structured Ensemble Learning Approach to Risk Diversification** 的会议。
   - 讲者是来自 **Miralta Bank** 的 **Alejandro Rodriguez Dominguez**，会议将使用 [Jitsi Meet](https://meet.jit.si/financereinforcementlearningmeeting)。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1368915287805661224)** (4 messages): 

> `Meeting #69, get_rewrites_for_renderer, MLPerf submissions, Scheduler Fusion, Driver` 


- **Meeting #69 已安排，需要主持人**：**Meeting #69** 定于 **圣地亚哥时间周一上午 9 点** 举行，但 @chenyuy 无法主持，因此需要其他人接手。
- **`FUSE_ARANGE` 助力 OLMoE 提速**：`FUSE_ARANGE` 的工作修复了一个缺失 sink 的 bug，使其能够作为 **OLMoE** 的 envvar 使用，从而在 3 层网络上实现了 **26% 的提速**，详见 [PR #9625](https://github.com/tinygrad/tinygrad/pull/9625)。
- **`kernelize()` 失败，`.contiguous()` 奏效**：发现了一个案例，其中 `.kernelize()` 在 rewrite 期间因 `unwrap` 函数抛出断言错误而失败，而 `.contiguous()` 则可以正常工作。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1367982167367618650)** (10 messages🔥): 

> `contiguous method of Tensor, devectorization, Gradient Accumulation with JIT` 


- **关于 Tensor 连续性的思考**：一位成员询问了 **Tensor** 的 contiguous 方法，以及如何在 **devectorization** 期间生成一系列应由 linearizer 展平的操作。
   - 该话题没有进一步的讨论。
- **梯度累积时的 JIT 编译故障**：一位成员报告了在使用 tinygrad 训练模型时，结合 **梯度累积 (gradient accumulation)** 和 **JIT 编译** 出现的问题，在使用 TinyJit 时，对 minibatch 的循环似乎会导致问题。
   - 错误发生在第二次调用 `opt.step()` 时，`opt.params` 最终导致 `grad` 为 `None`。
- **将 TinyJit 移至 epoch_step 内部可解决问题**：将 `def mb_step` 移至 `epoch_step` 内部并在该处应用 `TinyJit` 解决了梯度累积问题。
   - 该成员指出，这一改动使代码运行正常，表明 JIT 编译的作用域至关重要。


  

---

### **Cohere ▷ #[💬-general](https://discord.com/channels/954421988141711382/954421988783444043/1367945390229553172)** (7 messages): 

> `Internal Server Error, Coral and Chat Redirects` 


- **Cohere 遭遇 Internal Server Error！**: 用户报告了一个 ID 为 `7419be1956bcf44eaa4ea12323276950` 的 **internal server error**，该错误已上报给开发人员。
   - Cohere 工作人员引导用户发送邮件至 `support@cohere.com` 以获取进一步协助。
- **Coral 和 Chat 重定向恢复在线**: 在出现一些问题后，[coral.cohere.com](http://coral.cohere.com) 和 [chat.cohere.com](http://chat.cohere.com) 的重定向现已恢复正常，重新指向 playground。
   - 用户被引导至相应频道以报告任何进一步的问题或故障。


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1367961194593517618)** (2 messages): 

> `Embed V4, command-r latency` 


- **Embed Jobs 文档中缺失 Embed V4 模型**: 一位用户报告称，尽管示例代码中使用了 **Embed V4 模型**，但在 embed jobs 的文档中（特别是 *models* 参数下）却缺失了该模型。
   - 该用户通过指出在尝试创建 embed job 时使用 **Embed V4** 失败确认了这一差异，并询问了其可用时间表。
- **Command 模型的延迟指标**: 一位用户询问了 **command-a**、**command-r** 和 **command-r+** 模型的**延迟指标 (latency metrics)**。
   - 给出的消息中未提供具体数值。


  

---


### **Cohere ▷ #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1368941261637292124)** (3 messages): 

> `AI agent tools, LLM workflows, Full stack AI development, GPT-4o and Claude 3, Collaboration opportunities` 


- **AI 开发者构建基于 Agent 的工具**: 一位 AI 开发者介绍了自己，称其致力于 **agent-based tools** 和 **LLM workflows**，并对**合作**、**合同工作**或 **AI 领域**任何有趣的事情持开放态度。
   - 他们提到使用 **Langchain** 和 **FastAPI** 构建**销售助手**和 **RAG pipelines**，主要使用 **Python** 和 **Node**。
- **全栈开发者寻求 AI 机会**: 一位拥有 9 年经验的全栈开发者正在寻求机会，希望利用其技能为团队做出贡献，重点关注 **AI 解决方案**的开发。
   - 他们列举了在自动化工具（如 **n8n**、**Zapier**、**Make** 和 **GoHighLevel**）、**AI agent** 平台（如 **Voiceflow**、**Hume** 和 **Dify**）以及各种 **LLM**（包括 **GPT-4o**、**Claude 3** 和 **Llama-3**）方面的经验。