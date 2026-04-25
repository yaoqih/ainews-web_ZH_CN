---
companies:
- deepseek
- nvidia
- openai
- lambdaapi
- togethercompute
- xiaomi
date: '2026-04-24T05:44:39.731046Z'
description: '**DeepSeek-V4** 技术发布：采用 **1.6万亿（1.6T）参数的混合专家模型（MoE）**架构，其中 **激活参数为 490亿（49B）**，并支持
  **100万（1M）token 的上下文**。该模型展示了混合注意力机制（hybrid attention）和压缩 KV 方案，从而大幅降低了显存占用。在开源权重推理模型中，它排名
  **第二**，仅次于 **Kimi K2.6**，但存在幻觉率较高和推理服务成本较高的问题。


  该发布强调了硬件与模型的协同设计，利用 **NVIDIA Blackwell Ultra** 可实现每用户 **150+ TPS** 的吞吐量，并支持 **FP4
  和 FP8 量化**，从而实现单节点部署。在国产开源模型中，其定位与 **GLM-5.1** 和 **小米 MiMo V2.5 Pro** 具有竞争关系。


  与此同时，**OpenAI 发布了 GPT-5.5 和 GPT-5.5 Pro API**，拥有 **100万 token 的上下文窗口**，重点提升了长路径工作流的性能和
  token 效率，并已迅速集成到 **GitHub Copilot** 和 **Cursor** 等工具中。“GPT-5.5 能以更少的重试次数处理复杂、重度依赖工具且含糊不清的工作流”，这突显了其快速分发和智能体（agent）集成的优势。'
id: MjAyNS0x
models:
- deepseek-v4
- deepseek-v4-pro
- deepseek-v4-flash
- kimi-k2.6
- glm-5.1
- xiaomi-mimo-v2.5-pro
- gpt-5.5
- gpt-5.5-pro
people:
- scaling01
- ben_burtenshaw
- artificialanlys
title: DeepSeek v4 (或者翻译为：**深度求索 v4**)
topics:
- long-context
- mixture-of-experts
- model-quantization
- memory-optimization
- hardware-model-co-design
- inference-speed
- agent-integration
- token-efficiency
- model-deployment
- open-weights
- reasoning
- hallucination-detection
---

**平静的一天。**

> 2026年4月23日至4月24日的 AI 新闻。我们检查了 12 个 subreddits、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，未检查更多 Discord 频道。[AINews 网站](https://news.smol.ai/) 允许您搜索所有往期内容。友情提示，[AINews 现已成为 Latent Space 的一个板块](https://www.latent.space/p/2026)。您可以[选择加入或退出](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件发送频率设置！

---

# AI Twitter 综述


**DeepSeek-V4 的长上下文架构、开源权重定位及推理部署**

- **DeepSeek-V4 的技术发布**占据了讨论的主导地位：多位分析师将 **V4 Pro** 描述为拥有 **1.6T 参数、49B 激活参数的 MoE 模型**，并搭配了 **284B / 13B 激活参数**的 **V4 Flash**，两者均具备 **1M token 上下文**并采用 **MIT 许可证**。最核心的技术解读强调了全新的长上下文技术栈：混合注意力（hybrid attention）、压缩 KV 方案以及大幅度的内存占用降低。[@scaling01](https://x.com/scaling01/status/2047618271310926151) 称该技术报告为“重大进展”，突出了其高效的长上下文设计，并暗示其他开源实验室可能会借鉴该架构的部分设计；[@ben_burtenshaw](https://x.com/ben_burtenshaw/status/2047646980139016560) 将其实际影响总结为“1M 上下文”加上 **~10 倍更小的 KV cache**；[@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2047735160544841953) 提供了更完整的基于基准测试的分解：在智力指数（Intelligence Index）中排名 **#2 的开源权重推理模型**，仅次于 **Kimi K2.6**，在 **Agent 任务 GDPval-AA** 中表现强劲，但也存在**极高的幻觉率**，且推理服务成本显著高于前几代 DeepSeek。
- **基础设施的影响与模型质量同样重要**。此次发布引发了关于硬件-模型协同设计（hardware-model co-design）异常详尽的讨论。[@NVIDIAAI](https://x.com/NVIDIAAI/status/2047765637808664759) 表示 **Blackwell Ultra** 在 DeepSeek-V4-Pro 的 Agent 工作流中可提供 **150+ TPS/用户**，预计通过 **Dynamo、NVFP4 和高级并行技术**还将获得进一步提升；[@NVIDIAAI](https://x.com/NVIDIAAI/status/2047823093578518758) 随后发布了首日性能曲线。[@LambdaAPI](https://x.com/LambdaAPI/status/2047654086263320965) 指出其 Checkpoint 将**专家权重（expert weights）存储为 FP4**，其余权重为 **FP8**，使其能够适配单个 **8xB200** 节点。[@SemiAnalysis_](https://x.com/SemiAnalysis_/status/2047726025748930687) 和 [@togethercompute](https://x.com/togethercompute/status/2047743446522224987) 均宣布了首日支持。对于本地/开源部署，[@Prince_Canuma](https://x.com/Prince_Canuma/status/2047685898163147125) 在一台 **256GB Mac** 上运行了 **DeepSeek4-Flash**，随后在[此处](https://x.com/Prince_Canuma/status/2047847095466385899)发布了 **MLX 量化版本**。来自 [@ZhihuFrontier](https://x.com/ZhihuFrontier/status/2047664976215839021) 的一份实用技术解读分析了其压缩方案，声称在 1M 上下文下，其 **KV 压缩率较 V3.2 提高了 8.7 倍**。
- **与其它中国开源模型的竞争愈发激烈**。[@arena](https://x.com/arena/status/2047714237502677405) 报告称，排名前三的开源文本模型分别是 **GLM-5.1**、**DeepSeek-V4-Pro** 和 **Kimi-K2.6**，它们各自在不同的真实世界类别中领先。[@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2047799218828665093) 还强调了**小米 MiMo V2.5 Pro** 在其智力指数上得分为 **54**——与 Kimi K2.6 持平，如果能按承诺交付权重，它可能成为又一个顶级开源权重竞争者。[@scaling01](https://x.com/scaling01/status/2047626000091971811) 的宏观观点是：DeepSeek 仍然落后于前沿闭源实验室约 **3-6 个月**，但在编程和 Agent 领域正日益展现出竞争力，而这些领域正是开源实验室尚能跟上步伐的地方。

**GPT-5.5 API 发布、代码 Agent 性能及广泛的工具链支持**

- **OpenAI 将 GPT-5.5 和 GPT-5.5 Pro 推送至 API**，具备 **1M 上下文窗口**，此次发布的重点在于更好的长时运行工作和 Token 效率，而不仅仅是原始的 Benchmark 差异。核心公告来自 [@OpenAIDevs](https://x.com/OpenAIDevs/status/2047742566410736090)、[@OpenAIDevs](https://x.com/OpenAIDevs/status/2047742589982654915) 和 [@OpenAI](https://x.com/OpenAI/status/2047743592278745425)。OpenAI 的定位是 GPT-5.5 能够以更少的重试次数处理复杂、重工具使用且具有歧义的工作流，多家下游集成商立即验证了这一说法。
- **最强烈的信号是分发和 Agent 集成速度**。GPT-5.5 在发布当天即上线了 [Cursor](https://x.com/cursor_ai/status/2047744579127185843)、[GitHub Copilot](https://x.com/github/status/2047747243617460482)、[Devin](https://x.com/cognition/status/2047743153461936257)、[OpenRouter](https://x.com/OpenRouter/status/2047744317415141787)、[Perplexity](https://x.com/perplexity_ai/status/2047748486767272243)、[Cline](https://x.com/cline/status/2047769312514257148)、[Factory/Droid](https://x.com/FactoryAI/status/2047772874879193464)、[Hermes Agent](https://x.com/Teknium/status/2047791512210293067)，以及微软的技术栈，包括通过 [@satyanadella](https://x.com/satyanadella/status/2047743651053556126) 发布的 **Copilot, M365 Copilot, Copilot Studio 和 Foundry**。这使其成为近年来模型推广速度最快、范围最广的一次。
- **Benchmarks 和从业者反馈对编程/Agentic 工作表现出强烈的正向评价**。[@cursor_ai](https://x.com/cursor_ai/status/2047744579127185843) 表示 GPT-5.5 在 **CursorBench 上以 72.8%** 的成绩登顶；[@cline](https://x.com/cline/status/2047769312514257148) 报告其在 **Terminal-Bench 上以 82.7 的成绩排名第一**，领先于 Anthropic 尚未发布的 Mythos 预览版；[@j_dekoninck](https://x.com/j_dekoninck/status/2047788742434287622) 表示它成为了 **MathArena 的冠军**；[@scaling01](https://x.com/scaling01/status/2047818395970904229) 展示了 **LisanBench** 的提升，以及实质上更好的 Token 效率。在感性反馈方面，工程师们报告了定性上更好的代码：[@almmaasoglu](https://x.com/almmaasoglu/status/2047745168141324559) 表示“防御性冗余代码（defensive slop code）”问题消失了；[@KentonVarda](https://x.com/KentonVarda/status/2047788670728495142) 强调了其在数年前的背景信息中进行深度 bug 发现的能力；[@ChrisHayduk](https://x.com/ChrisHayduk/status/2047817267065258436) 观察到 Codex 中出现了更多自主的“等待/行动”行为；[@omarsar0](https://x.com/omarsar0/status/2047768166126809512) 则认为 GPT-5.5 在真实的编程 Agent 工作流中比早期的 GPT 模型更“懂行”。
- **Token 效率成为主要的竞争杠杆**。多篇帖子指出，GPT-5.5 的价值在于更低的总 Token 消耗，而不仅仅是顶尖的 Benchmark 胜出。[@OpenAIDevs](https://x.com/OpenAIDevs/status/2047772632150675593) 引用了 Perplexity 的数据，显示在处理相同的复杂任务时，**Token 使用量减少了 56%**；[@AravSrinivas](https://x.com/AravSrinivas/status/2047788775468908840) 表示将 Computer 的编排器切换到 GPT-5.5 降低了额度消耗；[@sarahmsachs](https://x.com/sarahmsachs/status/2047797374454747140) 声称 GPT-5.5 在 Notion 的知识工作基准测试中，比 Opus 4.7 **快 33%** 且仅使用其 **一半的 Token**。这种关于 Token 效率的叙事也解释了为什么一些开发者发现其实际性能比早期头条评估所暗示的更强。

**Agent Frameworks, Open ML Tooling, and Parallel Workflow UX**

- **Hugging Face 的 “ML Intern”** 是当天除了模型发布之外最重大的开源工具发布。来自 [@MillieMarconnni](https://x.com/MillieMarconnni/status/2047639632859500691) 的热门总结将其描述为一个 CLI Agent，它可以研究论文、搜索 HF 数据集和 GitHub、运行实验、启动任务，并推送最终模型，其中包含审核检查点和多达 **300 次迭代**。比热度更有趣的是后续数据：[@akseljoonas](https://x.com/akseljoonas/status/2047737429507944481) 表示，该发布立即推动了 **500 多个自主 AI 研究项目在 Space 上并发运行**，其中包括结合了 Recurrent Transformer、低比特权重和新 Attention 理念的架构实验。
- **Hermes Agent 迎来了一个重要的发布周期**。[@WesRoth](https://x.com/WesRoth/status/2047646749427216385) 将 **v0.11.0** 总结为迄今为止最大的更新，包括重写的 **基于 React 的 TUI v2**、插件仪表板、新 Provider、图像后端以及 QQBot 集成。周边生态系统也动作迅速：[@ShaneRobinett](https://x.com/ShaneRobinett/status/2047692184518787185) 发布了用于 Obsidian 项目协作的 **Hermes Kanban 1.5.0**，而 [@mr_r0b0t](https://x.com/mr_r0b0t/status/2047673600900010044) 则强调了 Nous 门户对 DeepSeek-V4 的首日支持。从业者的对比评价也很高；[@LoicBerthelot](https://x.com/LoicBerthelot/status/2047690512199540959) 认为 Hermes 在内存、安全性、模型支持和部署灵活性方面均优于 OpenClaw。
- **Cursor 的异步 subagent 用户体验正趋向于真正的多任务 IDE Agent**。[@cursor_ai](https://x.com/cursor_ai/status/2047764651363180839) 在 Cursor 3 中引入了 **/multitask**，允许异步 subagent 并行处理请求而非串行排队，此外还支持 **multi-root workspaces**（多根工作区）以进行跨仓库更改。多位高级用户强调，“不再被子代理阻塞”感觉是 Agent 编程工作流的一次质变。
- **DeepAgents 和沙箱基础设施趋于成熟**。LangChain 相关的帖子减少了对 Demo 的关注，更多转向生产工作流：[@sydneyrunkle](https://x.com/sydneyrunkle/status/2047645786020749590) 和 [@Vtrivedy10](https://x.com/Vtrivedy10/status/2047714471439737003) 描述了使用 **deepagents** 保持文档代码片段在库变更时得到测试并保持最新；同时 [@nu_b_kh](https://x.com/nu_b_kh/status/2047775326412136574) 发布了一个**原生 Linux 沙箱后端**，使用 **bubblewrap + cgroups v2** 来实现 Agent 代码的隔离本地执行。

**研究亮点：蒸馏、序列模型、工具路由和长周期个性化**

- **On-policy Distillation 中的 Token 选择** 受到关注，因为它提供了一种纯粹的效率提升。[@TheTuringPost](https://x.com/TheTuringPost/status/2047617791709282405) 总结的一篇论文表明，并非所有 Token 都同样重要：**高不确定性 Token** 和 **过度自信的错误** 携带最强的训练信号。据报道，仅使用由不确定性选出的 **~50% Token** 即可匹配或超越全量训练效果，同时减少 **~47%** 的内存占用；甚至仅关注 **<10%** 的“自信但错误”的 Token 也能接近全量性能。
- **Google 预览了 MesaNet**，这是一种序列模型替代方案，被定位为一种新的线性层，“在固定内存预算下能实现最优的上下文学习”。[@GoogleResearch](https://x.com/GoogleResearch/status/2047630714145776053) 的公告细节较少，但其脱颖而出是因为它被明确定义为 **Transformer 的替代方案**，而不仅仅是又一个效率优化。
- **动态工具门控 / 减少 “MCP 税”** 正在成为一个正式的系统级课题。[@omarsar0](https://x.com/omarsar0/status/2047725276851994639) 重点介绍了一篇使用意图-模式重叠（intent-schema overlap）加状态感知门控和延迟模式加载的论文；在模拟的 **120 个工具** 基准测试中，据报道工具 Token 从 **每轮 47.3k 下降到 2.4k**，上下文利用率从 **24% 提升至 91%**。
- **长周期个性化基准测试正在赶上实际部署需求**。[@StellaLisy](https://x.com/StellaLisy/status/2047645651324821998) 引入了 **HorizonBench**，旨在追踪长对话历史中的用户偏好，即使生活事件在无声中改变了这些偏好——这对于下一代持久化助手来说是一个有用的抽象。
- 另外两篇偏向理论的论文也备受关注：[@TheTuringPost](https://x.com/TheTuringPost/status/2047720038342476187) 关于 **Hyperloop Transformers** 的研究，该架构在多次传递中复用中间块并增加“超级连接”（hyper-connections），同时减少了 **~50%** 的参数；以及 [@learning_mech](https://x.com/learning_mech/status/2047723849874330047) 关于 **“Learning Mechanics”** 的研究，试图将 Scaling Laws、玩具模型和训练动力学统一成一种更类似物理学的深度学习理论。

**算力、主权与生态重组**

- **Meta 通过 AWS Graviton 扩展了其计算组合**。[@AIatMeta](https://x.com/AIatMeta/status/2047647617681957207) 宣布达成协议，将 **数千万个 AWS Graviton 核心** 引入 Meta 的基础设施，用于支持全球范围内的 Meta AI 和 agentic 系统。
- **Cohere 和 Aleph Alpha 宣布了以主权为核心的合作伙伴关系**，将自己定位为面向企业和政府工作负载的加拿大-德国“跨大西洋 AI 巨头”。参见 [@cohere](https://x.com/cohere/status/2047631725426000268)、[@aidangomez](https://x.com/aidangomez/status/2047651054381052086) 和 [@nickfrosst](https://x.com/nickfrosst/status/2047704679878996253)。虽然在技术层面较轻，但在战略上非常重要：它强化了区域控制的企业级 AI 技术栈的趋势。
- **据报道，谷歌对 Anthropic 的追加投入是信息流中最大的融资项目**。据 FT 报道，尽管谷歌通过 Gemini 与 Anthropic 直接竞争，但已承诺 **现在投入 100 亿美元，未来还将投入 300 亿美元**；参见 [@FT](https://x.com/FT/status/2047715653553942997)。即使考虑到融资结构的变数，对于基础设施观察者来说，信号是明确的：前沿模型（frontier-model）合作伙伴关系日益受到资本获取和算力承诺的影响，其程度不亚于模型 API 的竞争。

**热门推文（按互动量排序）**

- **GPT-5.5 的推出及其对编程的影响**：[@cursor_ai 关于 GPT-5.5 在 CursorBench 登顶](https://x.com/cursor_ai/status/2047744579127185843)、[@OpenAI 宣布 API 可用性](https://x.com/OpenAI/status/2047743592278745425) 以及 [@satyanadella 关于 GPT-5.5 贯穿 Copilot/Flex 产品线](https://x.com/satyanadella/status/2047743651053556126) 是信号最强的发布推文。
- **DeepSeek-V4 开源模型时刻**：[@NVIDIAAI 关于 Blackwell Ultra 性能](https://x.com/NVIDIAAI/status/2047765637808664759)、[@ArtificialAnlys 的基准测试明细](https://x.com/ArtificialAnlys/status/2047735160544841953) 以及 [@scaling01 的技术解读](https://x.com/scaling01/status/2047618271310926151) 最好地捕捉了该版本在技术和竞争方面的意义。
- **开源工具 / 本地 AI**：[@MillieMarconnni 关于 Hugging Face ML Intern](https://x.com/MillieMarconnni/status/2047639632859500691) 以及 [@julien_c 在 MacBook Pro 上通过 llama.cpp 运行本地 Qwen3.6-27B 进行编程](https://x.com/julien_c/status/2047647522173104145) 是互动量最高的从业者帖子，指向了近期工作流的变化，而不仅仅是模型排行榜的更迭。

---

# AI Reddit 总结

## /r/LocalLlama + /r/localLLM 总结

### 1. Deepseek V4 及相关发布

  - **[Deepseek V4 AGI 确认](https://www.reddit.com/r/LocalLLaMA/comments/1suolda/deepseek_v4_agi_comfirmed/)** (活跃度: 1138): **该图片是一个梗（meme），不包含任何技术内容。标题“Deepseek V4 AGI confirmed”暗示了对某个 AI 模型的幽默或夸张说法，可能是在引用通用人工智能（AGI）的进展。评论进一步暗示了讽刺基调，提到了未审查的数据集和军事应用，这些很可能不是认真的主张。** 评论反映了对 AI 能力的讽刺看法，提到了未审查的数据集和军事应用，表明了怀疑或幽默，而非严肃的技术讨论。

    - 用户 XtheUnknown 讨论了 Deepseek V4 的一个测试场景，强调了它过度思考问题的倾向。该模型将“仅使用一把刀”等约束条件解读为强制性而非可选，这影响了其解决问题的方法。这反映了对任务约束的细微理解，但也指出了在处理隐含指令方面潜在的改进空间。

  - **[Deepseek V4 Flash 和 Non-Flash 版本已在 HuggingFace 发布](https://www.reddit.com/r/LocalLLaMA/comments/1su3hdo/deepseek_v4_flash_and_nonflash_out_on_huggingface/)** (活跃度: 1393): **DeepSeek V4** 已在 [HuggingFace](https://huggingface.co/collections/deepseek-ai/deepseek-v4) 上发布，包含两个模型：**DeepSeek-V4-Pro** 拥有 `1.6T 参数`（其中 `49B` 被激活），以及 **DeepSeek-V4-Flash** 拥有 `284B 参数`（其中 `13B` 被激活）。两个模型都支持 `一百万 token` 的上下文长度，这对于处理长序列具有重要意义。这些模型是在 **MIT license** 下发布的，允许广泛的使用和修改。一条显著的评论强调了在处理此类大型模型时硬件限制（尤其是 RAM）的挑战。另一条评论建议使用 `0.01bit quantization`（量化）可能会更有利于管理模型大小。

- DeepSeek-V4 模型以其庞大的参数规模而闻名，Pro 版本拥有 1.6 万亿参数（激活 490 亿），Flash 版本拥有 2840 亿参数（激活 130 亿）。两款模型均支持高达 100 万 tokens 的超长上下文，这对于处理大规模数据输入和复杂任务具有重要意义。
- 一位用户对 DeepSeek-V4 模型的 0.01-bit 量化表达了兴趣，这表明用户关注在保持性能的同时减小模型体积和计算需求。量化是将模型优化以便在资源受限的硬件上部署的常用技术。
- 提及 MIT 许可证表明 DeepSeek-V4 是开源的，允许社区广泛使用和修改。这种许可选择可以促进协作和创新，因为开发者可以自由地将模型集成并适配到他们自己的项目中。

- **[隐藏的核心新闻：Deepseek v4 Flash 的官方 API 价格在其权重级别中低得令人难以置信](https://www.reddit.com/r/LocalLLaMA/comments/1su5gj5/buried_lede_deepseek_v4_flash_is_incredibly/)** (活跃度: 404): **图片展示了 "deepseek-v4-flash" 和 "deepseek-v4-pro" 两个模型之间的对比，强调了 "deepseek-v4-flash" 模型在输入和输出 token 成本方面明显更具性价比。尽管价格低廉，该模型在非思考和思考模式下均支持 JSON 输出、tool calls 和 chat prefix completion 等高级功能。围绕该图片的讨论表明，虽然 "deepseek-v4-flash" 以低价为卖点，但一些用户认为，考虑到参数规模的缩放，它实际上比之前的版本贵，"V3.2" 模型的单参数成本更低。** 评论者讨论了 GPU 短缺对当前定价的影响，认为随着 GPU 产量的增加，价格可能会下降。此外，关于定价策略也存在争议，一些用户注意到新模型与旧版本相比，单参数价格更高。

    - DistanceSolar1449 强调了 DeepSeek V3.2 和 V4 Flash 之间的价格对比，指出 V3.2 的 `671b` 参数定价为 `$0.26/0.38`（输入/输出），而 V4 Flash 的 `284b` 参数定价为 `$0.14/$0.28`。这表明如果价格随参数线性缩放，V4 Flash 实际上更贵，从而挑战了其性价比的说法。
    - jwpbe 对 DeepSeek V4 Flash 的 API 成本进行了对比分析，指出其 `14 cents in / 28 cents out` 的价格远低于竞争对手，例如 Minimax 2.7 的成本是其 `3x`，而 Qwen 的同类产品价格更高。他们还提到 Trinity Thinking Large 的价格是其两倍，表明 V4 Flash 在市场上具有竞争性的价格优势。
    - Worried-Squirrel2023 讨论了华为芯片发展的战略影响，认为 DeepSeek 的定价策略涉及用 NVIDIA 的利润空间换取 Ascend 的供应。他们预测一旦 `950 supernodes` 规模化，DeepSeek 可能会利用华为的技术进步来优化成本，从而在 open weights 级别中击败竞争对手。

- **[Deepseek 已发布 DeepEP V2 和 TileKernels。](https://www.reddit.com/r/LocalLLaMA/comments/1ste9zs/deepseek_has_released_deepep_v2_and_tilekernels/)** (活跃度: 396): ****Deepseek** 发布了 **DeepEP V2** 和 **TileKernels**，这是 AI 模型优化和并行化方面的重大进步。**DeepEP V2** 专注于提升模型效率和准确性，而 **TileKernels** 引入了一种新型并行化技术，据称可以实现线性扩展，这意味着计算能力翻倍将导致处理速度翻倍。此次发布是开源的，促进了 AI 研究的透明度和协作。更多详情请参见 [DeepEP V2 pull request](https://github.com/deepseek-ai/DeepEP/pull/605) 和 [TileKernels 仓库](https://github.com/deepseek-ai/TileKernels)。** 一位评论者强调，**Deepseek** 正在履行 **OpenAI** 曾被期望扮演的角色，即推进研究并公开分享成果，尽管存在专有技术，这仍建立了良好的声誉。另一位评论者质疑并行化技术是否真的能线性扩展，如果属实，这将是一个重大的技术突破。

- **DeepSeek 的 DeepEP V2 和 TileKernels** 因其在并行化技术方面的潜在进步而受到关注。有用户推测，这些技术可能会实现线性扩展（linear scaling），即计算能力翻倍可直接使处理速度翻倍。这可能代表模型训练和推理效率的显著提升。
- 关于 DeepSeek 的硬件使用存在猜测，特别是关于 SM100 和 Blackwell GPU。一位评论者建议 DeepSeek 可能正在使用 Blackwell GPU 进行训练，可能是通过在 Vast.ai 上租赁 B200 单元。这种硬件选择可能会影响其模型的性能和功能。
- DeepSeek 下一个模型（可能命名为 v4）的潜在创新受到关注。重点在于 Engram 和 mHC 技术的整合，预计这些技术将在模型性能中发挥关键作用。这些创新的成功可能取决于 DeepSeek 开发的新数据集。


### 2. Qwen 3.6 模型性能与基准测试

  - **[这就是我们现在的处境，LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1suqfba/this_is_where_we_are_right_now_localllama/)** (热度: 1755): **图中展示了一台 MacBook Pro 通过 Llama.cpp 运行 Qwen3.6 27B 模型，展示了即使在飞行模式下也能在本地执行复杂 AI 模型的能力。这突显了本地 AI 模型通过独立于云服务运行，在提高效率、安全性、隐私和主权方面的潜力。该帖子强调了使强大的 AI 模型在个人设备上可用的技术进步，并强调了本地执行对于隐私和控制的重要性。** 评论者对 Qwen3.6-27B 模型能力的夸大表示怀疑，认为虽然该尺寸的模型令人印象深刻，但仍无法与 Sonnet 或 Opus 等更先进的模型性能相匹配。人们担心夸大的说法可能会导致用户失望，并引发对更广泛 LLM 社区的反弹。

    - **ttkciar** 强调了用户对 Qwen3.6-27B 模型产生失望的可能性，指出虽然它的大小令人印象深刻且适用于 Agent 式代码生成，但它无法与 Sonnet 或 Opus 等更先进模型的能力相匹配。担忧在于，过度炒作其能力可能会导致对整个 LLM 社区的反弹，而不仅仅是针对发布言论的个人。
    - **sooki10** 同意该模型在本地编码任务方面令人印象深刻，但将其与 Opus 等更先进的模型进行比较具有误导性，可能会损害所称说法的可信度。这表明需要更准确的基准测试和关于模型能力的沟通，以有效管理用户预期。
    - **Melodic_Reality_646** 指出了资源上的差距，将使用高端 128GB RAM 的 m5max 系统与更普及的配置进行了比较。这强调了在评估模型性能时考虑硬件限制的重要性，因为并非所有用户都能使用如此强大的系统，这可能会扭曲对模型能力的认知。

  - **[DS4-Flash vs Qwen3.6](https://www.reddit.com/r/LocalLLaMA/comments/1sub71w/ds4flash_vs_qwen36/)** (热度: 470): **该图展示了 **DS4-Flash Max** 与 **Qwen3.6** 模型（特别是 `35B-A3B` 和 `27B` 版本）之间的基准测试对比。图表显示，**DS4-Flash Max** 在各个类别中普遍优于 Qwen 模型，尤其在 'LiveCodeBench' 和 'HLE' 基准测试中表现出色。这表明 DS4-Flash Max 在编码和推理任务中可能具有更卓越的能力。评论中的讨论暗示了 Qwen3.6 可能会有更大的版本（如 `122B`），并强调了 `1M token context` 功能的重要性，这可能会影响其在 'omniscense' 等其他基准测试中的表现。** 评论者指出，尽管 DS4-Flash Max 的体积更大，但其性能仅略优于 Qwen3.6，这引发了关于效率与规模之间关系的疑问。`1M token context` 被视为一个显著特性，可能会影响未来的基准测试结果。

- **Rascazzione** 强调了 Qwen 3.6 在上下文长度方面的显著增加，指出其具备处理 100万 token 上下文的能力。这是对以往模型的重大改进，对于需要处理长文本的任务（如文档摘要或复杂对话系统）可能产生深远影响。
- **LinkSea8324** 指出了模型规模的差异，DS4-Flash 拥有 2840亿参数，而 Qwen 3.6 为 270亿参数。这引发了关于模型规模与能力之间效率和性能权衡的讨论，特别是在计算资源和推理速度方面。
- **madsheepPL** 讨论了基准测试（benchmark）提升的非线性特征，认为即使模型在基准测试中看起来只是略有进步，其实际应用的影响可能更为显著。他们强调分数的提高并不成正比，且对现实应用的影响程度各异。

- **[Qwen 3.6 27B 在 Artificial Analysis 的智能体能力（Agency）上取得巨大进步 - 与 Sonnet 4.6 持平](https://www.reddit.com/r/LocalLLaMA/comments/1strodp/qwen_36_27b_makes_huge_gains_in_agency_on/)** (热度: 964): **Qwen 3.6 27B** 在 Artificial Analysis 的 **Agentic Index**（智能体指数）上已与 **Sonnet 4.6** 持平，超越了 **Gemini 3.1 Pro Preview**、**GPT 5.2 and 5.3** 以及 **MiniMax 2.7** 等模型。该模型在各项指数上均有提升，尽管在 **Coding Index**（编程指数）上的进步不那么明显，原因是该指数依赖于 **Terminal Bench Hard** 和 **SciCode** 等被认为不常规的基准测试。训练重点似乎放在了针对 **OpenClaw/Hermes** 的智能体应用上，凸显了较小规模模型接近前沿能力的潜力。人们对即将推出的 **Qwen 3.6 122B** 模型充满期待。评论者对 Qwen 3.6 27B 等小型模型的潜力感到兴奋，指出了显著的改进和未来版本的潜力。然而，也有人对这些提升的程度表示怀疑，认为部分进步可能归因于 "benchmaxxing"（针对基准测试的过度优化），而非模型内在能力的提升。

    - **Iory1998** 强调了 Qwen 3.6 27B 模型的出色表现，指出它超越了去年的 670B 模型。他们提到在 RTX 3090 和 RTX 5070ti 上运行 Q8 版本，使用 40GB VRAM 并开启 FP16 格式的 KV cache，上下文达到 170K，这凸显了该模型的效率和强大。
    - **AngeloKappos** 讨论了基准测试差距的缩小，分享了他们在 M2 芯片上运行 Qwen3-30b-a3b 模型的经验。他们注意到该模型能够有效处理多步工具调用（tool calls），并暗示如果 27B 稠密模型表现如此出色，即将推出的 122B 模型可能会凭借其潜在性能给 API 提供商带来挑战。
    - **Velocita84** 针对 Qwen 3.6 27B 模型报告的性能提升提出了关于潜在 "benchmaxxing" 的观点，暗示某些改进可能归功于优化的基准测试过程，而非模型本身的固有能力。这表明在评估模型性能声明时需要进行仔细审查。

- **[比较 QWEN 3.6 35B 与 QWEN 3.6 27B 的编程原语表现](https://www.reddit.com/r/LocalLLaMA/comments/1styxdy/compared_qwen_36_35b_with_qwen_36_27b_for_coding/)** (热度: 491): **该帖在配备 64GB RAM 的 MacBook Pro M5 MAX 上比较了两个版本的 **QWEN 3.6** 模型，即 `35B` 和 `27B` 参数版本。`35B` 模型达到了 `72 TPS`（每秒 token 数），而 `27B` 模型为 `18 TPS`。尽管速度较慢，但 `27B` 模型在编程任务中产生的结果更精确、更正确，而 `35B` 模型速度虽快但准确性较低。测试涉及生成一个模拟具有视差效果的移动汽车的单个 HTML 文件，且不使用任何外部库。模型通过 [Atomic.Chat](http://Atomic.Chat) 托管，源代码可在 [GitHub](https://github.com/AtomicBot-ai/Atomic-Chat) 上获取。** 一条评论强调了 `Qwen 3.6 27B FP8` 模型使用 opencode 的输出，耗时约 `52 秒`。另一条评论将其与 `Qwen 3.5 27B Q3` 模型进行了视觉对比，提示了输出质量的差异。

- 用户 'sacrelege' 分享了 Qwen 3.6 27B 模型在 FP8 精度下的性能结果，指出使用 'opencode' 完成一项任务大约耗时 52 秒。这表明其关注点在于通过精度调整来优化模型性能，这会显著影响计算效率和速度。
- 用户 'nikhilprasanth' 提供了 Qwen 3.5 27B Q3 模型的视觉对比，表明用户对比较 Qwen 模型不同版本和量化级别具有潜在兴趣。这突显了理解不同模型配置如何影响性能和输出质量的重要性。
- 'Technical-Earth-3254' 询问了测试中使用的量化方法，这对于理解模型大小、速度和准确性之间的权衡至关重要。量化可以极大地影响 Qwen 等大型模型在资源受限环境下的效率。

- **[Qwen 3.6 27B 性能强悍](https://www.reddit.com/r/LocalLLaMA/comments/1steip4/qwen_36_27b_is_a_beast/)** (热度: 1239): **该帖子讨论了 **Qwen 3.6 27B** 模型在配备 **RTX 5090 GPU** 和 `24GB VRAM` 的高端笔记本电脑上的性能，强调了它在 **pyspark/python** 以及数据转换调试任务中的有效性。用户使用 **llama.cpp** 并配合 `q4_k_m` (处于 `q4_0` 级别)，正在探索使用 `200k q8_0` 的 **IQ4_XS** 进行进一步优化。用户尚未实现 speculative decoding（投机采样）。配置包括一台配备 `64GB DDR5 RAM` 的 **ASUS ROG Strix SCAR 18**。** 评论建议在编程任务中避免将 kv cache 设置为 q4，推荐在 `130k` 上下文中使用 `q8`。另一条评论预告了 **z-lab** 即将发布的版本以及一个特定的 [GitHub pull request](https://github.com/ggml-org/llama.cpp/pull/22105) 所带来的性能提升，该 PR 承诺将解码速度提高 `2x`。此外，还有人对该模型在配备 `16GB VRAM` 和 `32GB DDR5 RAM` 且开启 offloading 的系统上的表现感到好奇。

    - sagiroth 强调了在将 Qwen 3.6 27B 用于编程任务时的一个技术考量，建议由于限制原因不要将 KV cache 用作 q4，而是建议使用 q8 以实现 `130k` 的上下文窗口，这可以显著提升大上下文任务的性能。
    - inkberk 指出解码速度即将得到提升，并引用了 `llama.cpp` 仓库的 pull request [#22105](https://github.com/ggml-org/llama.cpp/pull/22105)。这一更新，连同 z-lab 预期发布的 'dflash drafter'，有望带来 `2x` 的解码速度提升，这将极大惠及追求效率的用户。
    - Johnny_Rell 询问了 Qwen 3.6 27B 在配备 `16 GB VRAM` 和 `32 GB DDR5` 的系统上的性能，特别是关于 offloading 的效果。这表明其关注点在于优化资源分配以满足模型需求，这对于在消费级硬件上高效运行大型模型至关重要。


### 3. 本地 AI 模型实现与创新

- **[已经使用 PI Coding Agent 搭配本地 Qwen3.6 35b 一段时间了，效果简直疯狂](https://www.reddit.com/r/LocalLLaMA/comments/1stjwg5/been_using_pi_coding_agent_with_local_qwen36_35b/)** (热度: 656): **该帖子讨论了在实际项目中使用 **PI Coding Agent** 搭配 **Qwen3.6 35b a3b q4_k_xl 模型** 的情况，强调了一个自定义“计划优先”技能文件的有效性。该文件通过在执行任何代码前要求 `TODO.md` 审批来强制执行结构化工作流，确保任务按计划有序完成。模型在本地运行，展示了本地模型能力的重大进步。技能文件包括项目分析、澄清问题、创建 TODO.md、修订循环和任务执行等阶段，强调了编程任务中严谨的方法。该设置在配备 `8GB VRAM 和 32GB RAM` 的笔记本电脑上实现了 `15-30 tokens per second`，展示了该模型在适度硬件配置下的效率。** 评论者分享了类似的配置，其中一位使用配备 48GB RAM 的 Macbook Pro M4 Pro，注意到该模型的速度和智能，从而取消了 IDE 和 Claude 的订阅。另一位用户指出，“计划模式”已作为官方示例中的扩展提供，表明了社区的兴趣和采用。

- SoAp9035 分享了他们使用 `llama.cpp` 运行 Qwen3.6-35B 模型的配置，重点介绍了 `--temp 0.6`、`--top-p 0.95` 和 `--top-k 20` 等特定参数。他们在 `8GB VRAM` 和 `32GB RAM` 的配置下实现了 `15-30 tokens per second` 的性能，展示了本地模型推理中资源的高效利用。
- ibishitl 提到在配备 `48GB RAM` 的 Macbook Pro M4 Pro 上使用了类似的设置，并指出系统在完成任务时的速度和智能程度非常出色。他们已经取消了 IDE 和 Claude 的订阅，这表明带有 Qwen3.6-35B 的本地设置既具成本效益，又足以满足他们的需求。
- audiophile_vin 讨论了在本地使用 Qwen3.6 27B 模型的情况，并认为其表现令人印象深刻。他们提到了 GitHub 官方示例中一个名为“Plan mode”的扩展插件，它可以增强 coding Agent 的功能。这凸显了本地设置的灵活性和可扩展性。

- **[Qwen-3.6-27B, llamacpp, speculative decoding - appreciation post](https://www.reddit.com/r/LocalLLaMA/comments/1stcer1/qwen3627b_llamacpp_speculative_decoding/)** (Activity: 402): **该帖子讨论了在 Qwen-3.6-27B 模型上使用 Speculative Decoding 的实验，展示了 Token 生成速度从 `13.60 t/s` 到 `136.75 t/s` 的显著提升。用户将其归功于 `llama-server` 命令中的特定设置，特别是使用了 `--spec-type ngram-mod --spec-ngram-size-n 24 --draft-min 12 --draft-max 48`。该配置包括一台配备 `40GB VRAM` 和 `128GB DDR5 RAM` 的 Linux PC，使用了 `RTX3090` 和 `RTX4060ti` GPU。用户注意到了 `llama.cpp` 最近的变化，并提供了 [文档](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md#n-gram-cache-ngram-cache) 和 [Pull Request](https://github.com/ggml-org/llama.cpp/pull/19164) 的链接以供进一步阅读。** 评论者讨论了 Speculative Decoding 是否必须使用 `--no-mmproj-offload` 参数，一些人在不同的硬件设置上没有观察到速度提升。还有人对使用哪个模型进行草拟（drafting）感到好奇，并对不同用例下的速度提升持怀疑态度。

    - EatTFM 质疑 Speculative Decoding 是否需要 `--no-mmproj-offload` 标志，并指出在他们的 RTX5090 配置下没有获得速度提升。他们提供了使用 Qwen-3.6-27B 模型的 `llama.cpp` 详细命令行配置，强调了 `--spec-type ngram-mod` 和 `--spec-ngram-size-n 24` 等参数。他们怀疑可能是与其他参数的不兼容导致了问题。
    - kiwibonga 指出了在 Speculative Decoding 中使用 n-grams 的局限性，特别提到它“不适用于编程”，并且可能“破坏 Tool Calls”。这表明虽然 n-grams 可能对某些文本生成任务有益，但在需要精确工具集成或代码生成的语境中可能会引入问题。
    - nunodonato 分享了他们的经验，指出在他们的用例中，Speculative Decoding 没有带来明显的速度差异。这意味着 Speculative Decoding 的收益可能取决于具体语境，可能会随着不同的硬件设置或特定的模型配置而变化。

- **[just wanted to share](https://www.reddit.com/r/LocalLLM/comments/1su6vtx/just_wanted_to_share/)** (Activity: 1336): **用户开发了一个名为“Chappie”的分布式 AI 系统，该系统使用由四台 Mac Mini M4 Pro 组成的集群，每台机器都贡献于一个拥有 `256GB 统一内存`、`56 个 CPU 核心`、`80 个 GPU 核心` 和 `64 个 Neural Engine 核心` 的统一节点集群。该系统利用 [Exo](https://github.com/exo-explore/exo) 将节点汇集到分布式推理集群中，并采用 Qdrant 向量数据库进行内存共享和复制。Chappie 能够自主生成问题、阅读 arXiv 论文，并根据发现开发新技能。它具有一个用于任务分发的子 Agent 框架，以及一个由评审模型组成的“委员会（council）”以确保输出质量。该 AI 的架构包含了 Qwen 3.6 35B、Qwen 3.6 27B 等多种模型的组合，用于执行各种任务，重点在于自主探索，而非仅仅作为一个工具或助手。**

    - bionicdna 提出了一个技术改进建议，即在集群中使用基于 Thunderbolt 的 RDMA，Apple 现在已支持该功能。与使用 10G Ethernet 相比，这可能会进一步提升性能，因为 RDMA（远程直接内存访问）允许数据从一台计算机的内存直接传输到另一台计算机，而无需涉及任何一方的操作系统，从而实现更快的传输速度。

## 非技术性 AI 版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. GPT-5.5 发布与基准测试

- **[介绍 GPT-5.5](https://www.reddit.com/r/singularity/comments/1stqev3/introducing_gpt55/)** (活跃度: 1407): **OpenAI** 已发布 **GPT-5.5**，其定价为 `每 100 万输入 token 5 美元` 和 `每 100 万输出 token 30 美元`，价格是其前代产品 GPT-5.4 的两倍。该模型针对编程和知识工作等任务进行了优化，在复杂工作流中提供行业领先的准确性，并具有低延迟和低 token 使用率。它包含防止滥用的高级安全防护措施，并向 Plus、Pro、Business 和 Enterprise 用户开放，随后将提供 API 访问。更多详情请参阅 [原文](https://openai.com/index/introducing-gpt-5-5/)。** 评论中对新安全防护措施的有效性持怀疑态度，正如评论所言：*"我们正在发布带有迄今为止最强安全防护措施集的 GPT-5.5" 🫪 噢伙计*，暗示了对其鲁棒性的怀疑。

    - MapForward6096 强调了 GPT-5.5 的定价结构，指出其成本为 `每 100 万输入 token 5 美元` 和 `每 100 万输出 token 30 美元`，是 GPT-5.4 价格的两倍。这表明用户的成本将显著增加，可能会影响依赖该模型的项目的预算分配。
    - spryes 批评了 GPT-5.5 在 SWE-Bench Pro 基准测试中的表现，其分数为 `58.6%`，而 Mythos 达到了 `78%`。这一对比表明 GPT-5.5 在某些技术基准测试中可能不具备竞争力，引发了对其相对于其他模型效能的质疑。
    - mph99999 对 GPT-5.5 表示失望，将其描述为“微小的进步”而非预期的重大突破。这种情绪表明 GPT-5.5 的改进可能未达到此前公告或营销所设定的预期，尤其是在创新或性能提升方面。

  - **[GPT-5.5 基准测试结果已发布](https://www.reddit.com/r/singularity/comments/1stqk81/gpt55_benchmark_results_have_been_released/)** (活跃度: 779): **该图片展示了 AI 模型在各种基准测试中表现的对比分析，重点介绍了 **GPT-5.5** 及其变体。**GPT-5.5** 显示出比其前代产品 **GPT-5.4** 以及 **Claude Opus 4.7** 和 **Gemini 3.1 Pro** 等其他模型更好的性能。值得注意的是，**GPT-5.5 Pro** 在 BrowseComp 基准测试中达到了 `90.1%` 的分数，表明其在浏览能力方面取得了显著进步。然而，**SWE-Bench Pro** 的结果不尽如人意，仅从 `57.6%` 微幅增长至 `58.6%`，而 **Mythos** 的分数为 `77.8%`。** 评论者注意到某些基准测试的改进非常有限，特别是批评了 SWE-Bench Pro 分数的微小增长，并认为结果是被选择性突出以利于 GPT-5.5。还有一种观点反对在没有实际使用的情况下，仅凭基准测试分数就过早评判模型。

    - MapForward6096 和 spryes 指出 GPT-5.5 在 SWE-Bench Pro 基准测试中仅有微小进步，从 `57.6%` 提升至 `58.6%`，而 Mythos 模型获得了显著更高的 `77.8%` 分数。这表明 GPT-5.5 在这一特定基准测试中与 Mythos 相比可能缺乏竞争力。
    - TuteliniTuteloni 指出了 GPT-5.5 一个可能被忽视的优势：它能以明显更少的 token 提供更好的结果。这种 token 使用效率对于计算资源或处理时间受限的应用可能是关键因素，尽管基准测试提升有限，但提供了实际的益处。
    - BrennusSokol 对 GPT-5.5 表示怀疑，质疑它代表的是重大进步还仅仅是增量更新。这反映了社区内对于 AI 能力实现实质性飞跃而非微小改进的渴望。

  - **[Chat GPT 5.5 发布了，我们听到 Sam Altman 一些非常大胆的言论。有什么想法？](https://www.reddit.com/r/singularity/comments/1str6al/chat_gpt_55_got_launched_and_we_got_some_really/)** (活跃度: 784): **图片是 **Sam Altman** 讨论 **GPT-5.5** 发布的一条推文，强调了迭代部署对于快速改进和 AI 民主化以确保平等准入的重要性。Altman 强调了该平台对网络安全的关注及其支持广泛用户（包括公司和企业家）的能力。据报道，新版本使用更少的 token 并在更低的延迟下运行，这可能会增强性能和可访问性。** 评论反映了怀疑与支持交织的情绪，一些用户对过度积极的信息表示不信任，而另一些用户则对这些进展表现出热情。

- **[对 GPT 5.5 的看法](https://www.reddit.com/r/OpenAI/comments/1su1ikc/thoughts_on_gpt_55/)** (Activity: 1414): **该图片是一个迷因（meme），幽默地评论了新版本（可能是 GPT 5.5）的发布，通过讽刺地庆祝版本号的增加来进行调侃。这种俏皮的语气反映了对“版本号生意”的兴奋，暗示了对版本更新的一种轻松诙谐的看法。[查看图片](https://i.redd.it/3zudtu3yi1xg1.png)** 评论者表达了对 GPT 5.5 改进语音模式的渴望，并将其与 Claude 进行了正面比较，表明用户正在寻找特定的增强功能，且总体上对更新持积极态度。

    - One_Internal_6567 强调 GPT-5.5 Pro 明显优于其前代产品，并指出从 5.2 版本到 5.4 版本有可见的改进。这表明这些迭代在性能和功能上有着持续的增强，可能包括更好地处理复杂查询或更高效的处理。
    - hardworkinglatinx 将 GPT-5.5 与 Claude 进行了对比并看好前者，暗示 GPT-5.5 提供了更优越的性能或功能。这可能涉及响应准确性、速度或更有效地处理多样化主题的能力。
    - blownaway4 对 GPT-5.5 表达了正面看法，将其描述为“棒极了”。虽然缺乏具体的技术细节，但这种情绪可能反映了对该版本中引入的模型改进或新功能的总体满意度。

  - **[ChatGPT 5.5 🔥🔥🔥](https://www.reddit.com/r/OpenAI/comments/1stzivt/chatgpt_55/)** (Activity: 1359): **图片幽默地描绘了与 ChatGPT 5.5 的一段对话，AI 建议步行而不是开车去 50 米外的洗车店。这展示了模型根据上下文提供实际建议的能力，强调了能源效率和便利性。对话突显了 AI 的推理能力，因为它考虑了不必要的引擎启动以及为了这么短的距离挪车的麻烦等因素。这反映了模型在情境理解和决策过程方面的改进。** 一位评论者指出，AI 的响应质量随其“thinking”模式而变化，暗示延长思考时间会带来更准确的响应。另一条评论幽默地暗示，该问题在互联网上的流行程度可能影响了 AI 的训练数据。

    - Successful-Earth678 讨论了“extended thinking”模式对 ChatGPT 性能的影响，指出当模型被设置为思考更长时间时，它始终能提供正确的答案。这表明可以通过允许更多的处理时间来提高模型的准确性，突显了 AI 响应中速度与准确性之间的潜在权衡。
    - Portatort 认为某些问题在互联网上的广泛存在可能会影响 ChatGPT 的训练数据，从而可能影响其准确回答这些问题的能力。这引发了关于模型接触常见查询及其如何影响其学习和响应生成的思考。
    - ---0celot--- 提供了 ChatGPT 关于是否步行或驾车短距离决策场景的详细且实用的回复。回复中包含了对实用性、安全性和环境条件的考虑，展示了模型根据语境提供细致建议的能力。


### 2. DeepSeek V4 发布与基准测试

  - **[DeepSeek V4 已发布](https://www.reddit.com/r/singularity/comments/1su3lj9/deepseek_v4_has_released/)** (Activity: 1407): ****DeepSeek V4** 已在 [HuggingFace](https://huggingface.co/collections/deepseek-ai/deepseek-v4) 上发布，它采用了创新的 **manifold-constrained hyper-connections** (MHC) 技术，该技术在[最近的一篇论文](https://www.reddit.com/r/LocalLLaMA/comments/1q0zk1u/deepseek_new_paper_mhc_manifoldconstrained/)中进行了详细介绍。这种方法通过优化神经网络流形（manifold）空间内的连接来增强模型性能，有可能以极具竞争力的价格提供卓越的结果。** 一位评论者强调了该模型相对于其成本的惊人性能，认为它提供了显著的价值。另一位则指出 MHC 技术的实现是一项值得关注的进步。

- FaceDeer 强调 DeepSeek V4 实现了最近一篇论文中详述的 'manifold-constrained hyper-connections' 技术。这种方法可能有助于提升模型的性能，因为它通过在流形（manifold）内约束连接来优化神经网络架构，从而可能同时提高效率和准确性。[阅读更多](https://www.reddit.com/r/LocalLLaMA/comments/1q0zk1u/deepseek_new_paper_mhc_manifoldconstrained/)。
- InterstellarReddit 指出 DeepSeek V4 令人印象深刻的性价比，并暗示如果报告的数据属实，该模型可能会显著颠覆美国市场。这意味着与竞争对手相比，DeepSeek V4 以更低的成本提供了巨大的计算能力或准确性提升，使其成为企业和研究人员的一个极具竞争力的选择。
- cryyingboy 注意到 DeepSeek 持续交付新模型，这与可能更注重营销或理论讨论的竞争对手形成了鲜明对比。这表明 DeepSeek 频繁、务实的更新策略可能是其市场成功的关键因素，可能导致更快的采用并集成到各种应用中。

- **[DeepSeek V4 基准测试！](https://www.reddit.com/r/singularity/comments/1su5bwp/deepseek_v4_benchmarks/)** (热度: 466): **该图展示了各种模型的基准测试对比，包括 DS-V4-Pro Max 和 DS-V4-Flash Max，涵盖了 'Reasoning Effort'（推理能力）、'Knowledge & Reasoning'（知识与推理）、'Long Context'（长上下文）和 'Agentic'（智能体能力）等类别。使用的基准测试包括 MMLU-Pro、SimpleQA-Verified 和 Codeforces，突出了每个模型的优缺点。值得注意的是，DS-V4-Flash Max 因其高性价比而受到赞誉，在人工分析任务中的表现与 Gemini 3 Flash 相当，但成本显著降低，在典型使用场景下每月费用估计仅为 50 美分左右。** 评论者指出，虽然 V4 模型在编程任务中表现出色，但缺乏图像分析能力。DS-V4-Flash Max 被强调为一个极具成本效益的选项，以其他模型一小部分的成本提供了具有竞争力的性能。

    - Dangerous-Sport-2347 强调 DeepSeek V4 Flash 模型特别具有成本效益，在人工分析任务中的表现与 Gemini 3 Flash 相当，但成本显著降低——大约低 5 倍。这使得它对于关注成本效益的用户，特别是那些经常进行 AI 搜索和编程任务的用户来说，是一个极具竞争力的选择。据估计，中度使用的每月 API 费用约为 50 美分。

- **[DeepSeek V4 发布：1.6T 参数和 1M 上下文，未使用 Nvidia GPU。这是相关数据。](https://www.reddit.com/r/DeepSeek/comments/1su7rzr/deepseek_v4_dropped_16t_params_and_1m_context/)** (热度: 470): ****DeepSeek-V4** 推出了一款拥有 `1.6 trillion` 参数的模型，具有 `1 million` Token 的上下文窗口，在没有 Nvidia GPU 的情况下运行，使用了 **Huawei Ascend 950PR** 芯片。该模型有两个层级：拥有 `49B` 激活参数的 V4-Pro 和拥有 `13B` 激活参数的 V4-Flash。它采用了 **Engram Conditional Memory** 以实现高效的上下文管理，将推理开销降低了 `85%`。API 定价预计在每百万 Token `$0.14 到 $0.28` 之间，显著低于竞争对手。该模型的架构利用了参数稀疏性和原生内存检索，挑战了 Nvidia GPU 的垄断地位，并可能改变 AI 经济学。** 评论者注意到价格可能会进一步降低，并对 Nvidia 市场地位的影响持怀疑态度。还有人观察到模型在身份自我识别和知识截止日期（knowledge cutoff）方面存在不一致，表明模型更新可能存在问题。

    - Neo_Shadow_Entity 强调了 DeepSeek V4 在自我识别和知识截止日期方面的潜在问题。该模型仍自称为 DeepSeek-V3，并且知识截止日期似乎在 2025 年，导致在讨论 2025 年之后的事件或版本时出现混乱。这表明模型的内部数据或更新机制可能尚未与最新版本完全同步，导致其对 2026 年的 DeepSeek V4 信息产生误解或幻觉。
    - smflx 指出了关于 DeepSeek V4 背景下 'Engram' 一词的误解。与某些预期相反，'Engram' 与 KV-cache 无关，而是与模型的权重（weights）有关。评论者指出 Huggingface 页面缺少对 'Engram' 的描述，表明需要进一步调查以了解其在模型中的作用或存在。
    - Wickywire 强调了 DeepSeek V4 定价策略的重要性，指出该模型以极具竞争力的价格提供了强大的能力。这种定价可能会显著改变 AI 用户的格局，特别是在像 Openclaw 这样的环境中，高性价比、高容量的模型可以提供竞争优势。

- **[Deepseek-v4 flash and v4 pro](https://www.reddit.com/r/DeepSeek/comments/1su3bya/deepseekv4_flash_and_v4_pro/)** (Activity: 549): **该图片提供了两个 AI 模型 **deepseek-v4-flash** 和 **deepseek-v4-pro** 的详细对比，突出了它们的功能和定价。关键区别包括 context length 和最大输出能力，其中 v4-pro 提供了增强功能，如 JSON output 和 tool calls。此外还对比了 input 和 output token 的定价结构，为潜在用户提供了成本效益分析。评论中的一个显著点是 Deepseek Reasoner 已弃用，转而采用 v4 flash thinking mode，这虽然影响了性能，但仍保持了具有竞争力的能力。**

    - 讨论指出，Deepseek Reasoner 正在被弃用，取而代之的是 Deepseek v4 Flash 模型，尽管它是一个 “flash” 模型，但其性能令人印象深刻。用户对其能力感到惊讶，因为它的表现几乎与之前的 Deepseek Reasoner 持平，尽管存在一些限制。这种转变可能是近期 API 观察到的性能提升的一个因素，因为 Flash 模型比其前身 Deepseek v3 小得多。
    - 提到与 Deepseek v4 Pro 模型相关的成本增加，表明定价策略发生了变化，这可能会影响以前享受质量和性价比平衡的用户。这一变化意味着虽然性能可能有所提高，但访问这些模型的经济门槛也增加了，可能限制了某些用户的可访问性。
    - 评论还涉及 Deepseek 更广泛的战略举措，例如与其他实体联手，这可能会影响这些模型部署和定价的变化。这可能表明公司正转向更加集成或协作的 AI 开发方法。


### 3. Claude Code 问题与更新

  - **[Anthropic 刚刚发布了一篇 postmortem（故障复盘），详细解释了为什么 Claude 在过去一个月里感觉变笨了](https://www.reddit.com/r/ClaudeCode/comments/1str8gi/anthropic_just_published_a_postmortem_explaining/)** (Activity: 3991): ****Anthropic** 发布了一篇 postmortem，详细介绍了导致 **Claude Code** 性能感官下降的三个 bug。第一个 bug 涉及 3 月 4 日 reasoning effort 从 `high` 静默降级到 `medium`，该问题已于 4 月 7 日恢复。第二个 bug 是 3 月 26 日发生的 caching 问题，导致 Claude 忘记了其 reasoning history，引起 cache misses 并导致 usage limit 更快耗尽。第三个 bug 是 4 月 16 日的 system prompt 更改，将 tool calls 之间的回复限制在 25 个单词以内，影响了 coding 质量，该问题已于 4 月 20 日恢复。这些影响不同流量片段的问题已于 4 月 20 日（v2.1.116）修复，订阅者的 usage limits 正在重置。[阅读完整 postmortem](https://www.anthropic.com/engineering/april-23-postmortem)。** 评论者指出，这些问题与用户的怀疑相吻合，表明用户反馈与公司承认之间存在脱节。尽管一些用户对最初缺乏沟通表示沮丧，但 postmortem 的透明度得到了认可。

    - Direct-Attention8597 提供了 Anthropic postmortem 的直接链接，其中详细说明了导致 Claude 感知性能下降的技术问题。该 postmortem 是了解 Anthropic 实施的具体工程挑战和解决方案的宝贵资源。[点击此处阅读更多](https://www.anthropic.com/engineering/april-23-postmortem)。
    - Jack_Dnlz 强调了 Anthropic 在周末前重置 usage limits 的战略决策，认为由于许多用户在此时段活跃度较低，这将对用户的影响降至最低。这暗示了一种管理用户体验和资源分配的计算方法，可能会减轻其系统的即时负载。
    - Sufficient-Farmer243 评论了社区在官方确认之前诊断 Claude 问题的能力，表明用户的反馈和观察是准确的。这突显了社区见解在识别和理解 AI 性能问题方面的重要性。

- **[Claude Code 质量问题导致的使用额度重置](https://www.reddit.com/r/ClaudeCode/comments/1stpywt/usage_reset_due_to_claude_code_quality_issues/)** (活跃度: 615): **图片是来自 **ClaudeDevs** 的一条推文，解释了由于 Claude Code 的质量问题导致的使用额度重置。在收到用户报告后，他们进行了调查，并发布了一份关于已确定的三个问题的复盘报告 (post-mortem)，这些问题已在版本 `2.1.116+` 中修复。因此，所有订阅者的使用额度都已重置。[图片](https://i.redd.it/v0euvm9d9zwg1.png)** 一些用户注意到这次重置有些不寻常，剩余时间限制各不相同，并希望这些修复能解决缓存未命中 (cache misses) 和不寻常的使用额度过快消耗问题。

    - YatzyNanimous 强调了对 Claude 缓存未命中 (cache misses) 和异常使用额度消耗问题的担忧，认为重置可能会解决这些技术问题。缓存未命中会导致数据检索效率低下，影响性能，而意外的使用额度消耗可能预示着底层的资源管理问题。
    - dwight-is-right 提到了 GPT 5.5 的发布，并提及了最近的开源权重 (open weight) 发布，如 Kimi 2.6、GLM 5.1 和 qwen 3.6。这些发布具有重要意义，据报道它们缩小了不同 AI 模型之间的性能差距，表明了竞争格局中一个模型的改进会促使其他模型的进步。
    - 讨论涉及了 AI 模型更新和重置的技术影响，重点关注这些变化如何影响性能和资源分配。对特定模型版本及其对竞争激烈的 AI 领域影响的提及，突显了发展的快速步伐以及紧跟最新发布的重要性。

  - **[Claude 限制不再按小时取整](https://www.reddit.com/r/ClaudeAI/comments/1sue09c/claude_limits_no_longer_round_to_the_nearest_hour/)** (活跃度: 494): **该图片展示了 AI 服务 Claude 管理其使用限制方式的变化，从取整到最近的小时改为更精确的基于分钟的系统。这一调整可能是为了应对用户在整点前发送消息以最大化利用额度的行为。通知还建议了升级到 Pro 版本的选项，表明了分层服务模式。** 一条评论认为之前的系统存在缺陷，将限制视为“小时桶 (hourly buckets)”，这可能导致效率低下的使用。另一条评论幽默地指出了快速达到使用限制的挫败感，强调了更好地管理消息限制的必要性。

    - jake_that_dude 认为 Claude 限制的问题在于被构想为“小时桶 (hourly bucket)”，这会导致效率低下的使用。对于较长的任务，建议将工作拆分为较小的对话 (chats)，并包含详细的状态、阻碍因素和后续步骤的交接笔记，以避免在上下文损耗 (context churn) 而非生产性输出上浪费额度。
    - idiotiesystemique 强调了通过开启新对话和创建交接文件来有效管理对话会话的重要性。这种方法有助于保持连续性和效率，尤其是在处理复杂或长时间的交互时。
    - KronosDeret 提到了“燃料管理插件 (fuel management plugin)”的变化，暗示了可能影响系统内资源或限制管理方式的技术更新或修改。这对于需要适应新配置或设置的用户可能具有参考价值。

  - **[Claude 为所有人重置了限制](https://www.reddit.com/r/ClaudeAI/comments/1stozsr/claude_reset_limits_for_everyone/)** (活跃度: 2094): **图片描绘了一个服务的仪表盘，可能与 AI 或机器学习使用有关，显示所有类别的限制已重置为 0%，包括“当前会话 (Current session)”、“所有模型 (All models)”和“Claude Design”。这次重置暗示了服务使用政策的变化或额度的临时重置，可能与新功能或更新有关，例如传闻中的 GPT-5.5 发布。正如评论中所指出的，重置对接近使用限制的用户非常有益。** 一条评论幽默地暗示计费系统是“看心情的 (vibes-based)”，暗示了限制管理方式的不可预测性或不一致性。另一条评论指出，重置对接近限制的用户有利，但也提到重置后额度消耗似乎更快，表明使用追踪或模型效率可能发生了变化。

- National-Data-3928 指出了使用额度重置的一个重大问题，并提到他们消耗额度的速度比以前快得多。这表明底层的 usage tracking 或计费算法可能发生了变化，可能会影响那些重度依赖该服务的用户。
- DispensingLCQP 对重置时间表示沮丧，因为重置周期意外地从周四变成了周五。这一变化打乱了他们预定的使用模式，特别是对于那些根据特定日期安排使用计划的用户。该评论还批评了 Opus 4.7 在创意写作任务中的表现，表示对其与其他模型相比的能力感到不满。


# AI Discords

不幸的是，Discord 今天关闭了我们的访问权限。我们不会再以这种形式恢复它，但我们很快就会发布新的 AINews。感谢阅读到这里，这是一段美好的历程。