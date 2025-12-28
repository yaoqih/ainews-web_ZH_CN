---
companies:
- deepseek_ai
- openai
- gemini
- meta-ai-fair
- anthropic
- x-ai
- ollama
- hugging-face
- alibaba
- bytedance
- xiaomi
date: '2025-06-02T05:44:39.731046Z'
description: '**DeepSeek R1-0528** 的发布在推理能力、减少幻觉、JSON 输出和函数调用方面带来了重大改进。在 **Artificial
  Analysis Intelligence Index**、**LiveBench** 和 **GPQA Diamond** 等基准测试中，其表现已媲美或超越了
  **OpenAI o3** 和 **Gemini 2.5 Pro** 等闭源模型。


  该模型在全球开放权重（open weights）智能排名中位列第二，超越了 **Meta AI**、**Anthropic** 和 **xAI**。开放权重和技术透明度促进了其在
  **Ollama** 和 **Hugging Face** 等平台上的快速普及。在开放权重策略的推动下，包括 **DeepSeek**、**阿里巴巴**、**字节跳动**和**小米**在内的中国
  AI 实验室，目前在模型发布和智能水平上已与美国实验室并驾齐驱，甚至有所超越。


  强化学习（RL）后训练对提升智能水平至关重要，这与 **OpenAI** 的技术趋势一致。优化的量化技术（1位、4位）和本地推理使得在消费级硬件上进行高效实验成为可能。**LisanBench**
  等新基准测试旨在考察知识、规划、记忆和长上下文推理能力，目前 **OpenAI o3** 和 **Claude Opus 4** 在这些测试中处于领先地位。相关讨论也对基准测试污染以及过度强调强化学习调优带来的收益表示了担忧。'
id: MjAyNS0w
models:
- deepseek-r1-0528
- o3
- gemini-2.5-pro
- claude-opus-4
people:
- teortaxestex
- wenfeng
- danielhanchen
- awnihannun
- reach_vb
- abacaj
title: 今天没发生什么。
topics:
- reasoning
- reinforcement-learning
- benchmarking
- quantization
- local-inference
- model-evaluation
- open-weights
- transparency
- post-training
- agentic-benchmarks
- long-context
- hallucination-detection
---

**一个安静的周末。**

> 2025年6月2日至6月3日的 AI 新闻。我们为您检查了 9 个 subreddits、449 个 Twitter 账号和 29 个 Discord 社区（218 个频道，9059 条消息）。预计节省阅读时间（以 200wpm 计算）：852 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以美观的 vibe coded 方式展示所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻详情，并在 @smol_ai 上给我们反馈！

抱歉推送晚了，但今天确实比较平静。AIE 门票已售罄，因此我们推出了 [AI Engineer 线上分会场](https://www.latent.space/p/aiewf-2025)，你现在可以[收藏主题演讲和 MCP 直播频道](https://www.youtube.com/watch?v=z4zXicOAF28)，让 YouTube 算法发挥作用，这很可能是 AIE 历史上规模最大的直播。

https://www.youtube.com/watch?v=z4zXicOAF28

---

# AI Twitter 回顾

**1. 基础模型进展：DeepSeek R1-0528、基准测试与开源权重领导地位**

- **DeepSeek R1-0528 发布、开源权重及基准测试表现**：[DeepSeek-R1-0528](https://x.com/deepseek_ai/status/1928061589107900779) 的发布在推理、减少幻觉、JSON 输出和 function calling 方面带来了重大改进。该模型在多个基准测试中匹配或超越了 [OpenAI o3 和 Gemini 2.5 Pro 等领先的闭源模型](https://x.com/ArtificialAnlys/status/1928071179115581671)，包括 [Artificial Analysis Intelligence Index](https://x.com/ArtificialAnlys/status/1928071179115581671)、[LiveBench](https://x.com/scaling01/status/1928173385399308639) 和 [GPQA Diamond](https://x.com/EpochAIResearch/status/1928489527204589680)（[DeepSeek R1-0528 得分为 76%](https://x.com/EpochAIResearch/status/1928489527204589680)）。[Artificial Analysis](https://x.com/ArtificialAnlys/status/1928071179115581671) 强调 DeepSeek 目前在开源权重智能方面并列全球第 2，超越了 Meta、Anthropic 和 xAI。
- **开源权重和技术透明度推动采用**：[DeepSeek 的方法](https://x.com/ArtificialAnlys/status/1928477941715079175)（开源权重、代码和研究目标）实现了全球范围内的快速采用，多个平台（[Ollama, Hugging Face, OpenRouter 等](https://x.com/ollama/status/1928543644090249565), https://x.com/awnihannun/status/1928125690173383098, https://x.com/basetenco/status/1928195639822700898）迅速整合了该模型用于推理和实验。
- **中国 AI 实验室匹配或超越美国实验室**：[Artificial Analysis Q2 State of AI China 报告](https://x.com/ArtificialAnlys/status/1928477941715079175)发现，中国实验室（DeepSeek、阿里巴巴、字节跳动、小米等）现在发布模型的时间仅比美国同行晚几周，且在智能水平上达到持平或更优。开源权重策略支撑了这一进展。
- **强化学习（Reinforcement Learning）和后训练驱动快速提升**：[DeepSeek 的智能跨越由 RL 后训练驱动](https://x.com/ArtificialAnlys/status/1928071179115581671)，反映了 OpenAI 在 o1 和 o3 之间 10 倍 RL 扩展的趋势。[Artificial Analysis](https://x.com/ArtificialAnlys/status/1928071179115581671) 和 [EpochAIResearch](https://x.com/EpochAIResearch/status/1928489524616630483) 展示了 RL 对于高效获取智能增益的关键性。
- **优化的量化（Quantization）和本地推理**：快速发布的量化版本（[danielhanchen 的 1-bit 和 4-bit 量化](https://x.com/danielhanchen/status/1928278088951157116)、[awnihannun 针对 Qwen3 8B 的 4-bit DWQ](https://x.com/awnihannun/status/1928125690173383098)、[reach_vb 的 MLX 量化](https://x.com/reach_vb/status/1928002892633383338)）和实时部署（[Ollama 的 DeepSeek 思考模式](https://x.com/ollama/status/1928543644090249565)）使得即使在消费级硬件上也能进行高效实验。
- **社区、透明度和模型文化**：[teortaxesTex](https://x.com/teortaxesTex/status/1927919610612875492) 和 [ArtificialAnlys](https://x.com/ArtificialAnlys/status/1928477951365939328) 强调 DeepSeek 的透明度、极简的市场营销和对技术的专注是其差异化优势，[Wenfeng 的愿景](https://x.com/teortaxesTex/status/1927918495125172621)驱动着长期进步。

**2. 模型评估、推理、基准测试与 RL**

- **新的推理和 Agent 基准测试**：[scaling01 推出了 LisanBench](https://x.com/scaling01/status/1928510435164037342)，这是一个针对知识、规划、记忆和长上下文推理的可扩展测试，结果显示 o3 和 Claude Opus 4 处于领先地位。[LiveBench](https://x.com/scaling01/status/1928173385399308639) 现在包含了 Agent 编码测试，DeepSeek R1-0528 综合排名第 8，在数据分析方面排名第 1。
- **后训练与 RL 饱和**：[lateinteraction](https://x.com/lateinteraction/status/1928148705145934252) 批评了基准测试污染以及对 RL 微调收益的过度强调，警告称近期许多明显的进展可能归功于 Prompt/模板对齐，而非通用能力的提升。[abacaj](https://x.com/abacaj/status/1927948317931000277) 指出该领域目前正专注于使用 Qwen 进行 RL 实验。[giffmana](https://x.com/giffmana/status/1928314882761334871) 和 [vikhyatk](https://x.com/vikhyatk/status/1928268671979565330) 对近期 RL 论文的可靠性和影响表示怀疑。
- **可解释性与开源工具**：[Anthropic 发布了开源电路追踪（circuit tracing）工具](https://x.com/AnthropicAI/status/1928119229384970244)，使研究人员能够生成 LLM 内部的归因图（[mlpowered 演示](https://x.com/mlpowered/status/1928123130725421201)，[NeelNanda5 评论](https://x.com/NeelNanda5/status/1928169762263122072)），此举因透明度和可复现性而受到赞誉。
- **模型推理改进与架构创新**：[Ollama 为 DeepSeek 引入了“思维（thinking）”分离](https://x.com/ollama/status/1928543644090249565)，使推理过程可追踪。[cline](https://x.com/cline/status/1928208680903921803) 展示了“Extended Thinking”和“Sequential MCP”结构将 Claude 的推理性能提升了高达 68%。[StasBekman](https://x.com/StasBekman/status/1928571964647682400) 强调了用于推理优化的位移并行（shift parallelism）。

**3. 多模态 AI、Agent 与工具**

- **Perplexity Labs 发布与 Agent 工作流**：[Perplexity 推出了 Labs](https://x.com/perplexity_ai/status/1928141072011776088)，允许用户通过 Prompt 构建复杂的仪表板、代码工具和应用。[AravSrinivas](https://x.com/AravSrinivas/status/1928220532614537318) 展示了“金融研究员/分析师现在就是一个 Prompt”，[Labs 的动态 UI](https://x.com/AravSrinivas/status/1928192558586315119) 和“深度研究（deep research）”功能正在被迅速采用。
- **视觉语言模型与 SOTA VLM**：[来自小米的 MiMo-VL-RL](https://x.com/mervenoyann/status/1928475979753619663) 在 GUI 导航和推理方面优于 GPT-4o，且开放权重并采用 MIT 许可证（[reach_vb 基准测试](https://x.com/reach_vb/status/1928360066467439012)）。Black Forest Labs 发布了新的 SOTA 图像编辑/生成模型 [FLUX.1 Kontext](https://x.com/iScienceLuvr/status/1928186905079992507)（[togethercompute 免费演示](https://x.com/togethercompute/status/1928527563791441993)），在角色一致性和上下文内编辑方面表现出色。
- **视频与机器人模型**：[Google 的 Veo 3](https://x.com/Google/status/1928573869893230705) 现已在 73 个国家/地区推出，并在图生视频（Image-to-Video）和文生视频（Text-to-Video）排行榜上均名列前茅（[ArtificialAnlys 对比](https://x.com/ArtificialAnlys/status/1928318831761707224)）。[ClementDelangue](https://x.com/ClementDelangue/status/1928125034154901937) 宣布了开源机器人，[TheRundownAI](https://x.com/TheRundownAI/status/1928104195279749526) 报道了人形机器人和机器人平台的快速进展。

**4. AI 基础设施、扩展与硬件**

- **TPU、GPU 与推理扩展**：[demishassabis 表扬了 Google 的 SRE/基础架构团队](https://x.com/demishassabis/status/1928604371157233918)，他们在 Gemini/Veo 的需求下维持了 TPU 的运行。[StasBekman](https://x.com/StasBekman/status/1928571964647682400) 和 [tri_dao](https://x.com/tri_dao/status/1928170648863473892) 讨论了为实现最佳推理而进行的硬件感知架构选择（GQA, MLA, GLA）。[danielhanchen](https://x.com/danielhanchen/status/1928278088951157116) 和 [awnihannun](https://x.com/awnihannun/status/1928125690173383098) 通过激进的量化使大型模型能够在消费级硬件上运行。
- **主权 AI 与数据中心策略**：[JonathanRoss321](https://x.com/JonathanRoss321/status/1928241967122506083) 解释了加拿大采用 Groq 构建主权 AI 基础设施的情况。[saranormous](https://x.com/saranormous/status/1928479931660411033) 和 [AndrewYNg](https://x.com/AndrewYNg/status/1928099650269237359) 讨论了国家在研究、人才和本地基础设施方面投资对于保持竞争力的重要性。

**5. AI Agent、记忆与工作流编排**

- **Agentic 系统架构与记忆**：[LangChainAI](https://x.com/LangChainAI/status/1928135137658818711) 和 [omarsar0](https://x.com/omarsar0/status/1928492639906607297) 详细介绍了基于 DAG 的 Agent 架构，用于稳健的工作流编排和以记忆为中心的 Agent 设计（[Omarsar0 MemOS 抽象](https://x.com/omarsar0/status/1928116365640225222)）。[sjwhitmore](https://x.com/sjwhitmore/status/1928520064078328193) 测试了 Spark 的长期记忆和行为设计。
- **自我改进的 Agent**：[SakanaAILabs 推出了 Darwin Gödel Machine (DGM)](https://x.com/SakanaAILabs/status/1928272612431646943)，这是一个能够重写自身代码的自我改进型编程 Agent，将 SWE-bench 的性能从 20% 提升至 50%。[hardmaru](https://x.com/hardmaru/status/1928284568756629756) 和 [SakanaAILabs](https://x.com/SakanaAILabs/status/1928447873362153669) 解释了 Agent 改进的概念和开放式进化（open-ended evolution）。

**6. Meme、幽默与社区氛围**

- **Meme 与轻松话题**：[ID_AA_Carmack 接受了针对资深专家的 “Parmesan” 原型](https://x.com/ID_AA_Carmack/status/1928239003397984389)，[swyx 的 “修复 Apple 的 Bug” 计划](https://x.com/swyx/status/1928512178941808838) 获得了极高关注，[awnihannun 调侃了 DeepSeek 的过度思考](https://x.com/awnihannun/status/1928119439737729482)。[nearcyan 的医疗 AI 系统 Meme](https://x.com/nearcyan/status/1928620490416906430) 讽刺了医疗保健相对于 LLM 的低效。[TheZachMueller 的 “技术情侣” 投票](https://x.com/Yuchenj_UW/status/1928502759227080841) 和 [DavidSHolz 诗意的 SpaceX 回忆](https://x.com/DavidSHolz/status/1928189415291245040) 走红，反映了社区技术严谨性与幽默感的结合。

---

# AI Reddit 综述

## /r/LocalLlama 综述

### 1. 新的模型量化技术与本地模型性能

- [**IQ1_Smol_Boi**](https://i.redd.it/9u1teeqt4g4f1.png) ([评分: 378, 评论: 44](https://www.reddit.com/r/LocalLLaMA/comments/1l19yud/iq1_smol_boi/))：**这张图片是一个 Meme 风格的插图，对比了两种量化语言模型变体：R1-671B fp8（描绘为一个强壮的人物）和 IQ1_S（显示为一个滑稽扭曲的人物），形象地夸大了它们在性能和复杂程度上的差异。该帖子提供了创建 DeepSeek R1-0528 的超紧凑 “IQ1_S_R4” 量化版本（131GiB，适用于 128GiB RAM + 24GB VRAM 系统）的背景，与体积大得多的 Qwen3-235B-A22B-Q8_0（232.769GiB，PPL=5.3141）相比，它表现出更低的 Perplexity（困惑度，越低越好，尽管取决于架构）。文中还讨论了 Unsloth 的 UD-TQ1_0 (151GiB) 和 Bartowski 的 IQ1_M (138GiB) 等替代方案，强调了硬件受限用户在比特率、模型大小和 Perplexity 之间的新兴权衡。值得注意的是，由于跨架构的 PPL 基准测试可能具有误导性，其实际效用和 Perplexity 对比仍存在争议。** 评论者从技术角度辩论了 Perplexity 作为量化质量指标的效用，并引用了 Unsloth 的替代建议。个人基准测试和轶闻也对比了 DeepSeek 和 Qwen 架构在性能、速度和实际用途方面的表现，强调了现实使用中的主观和依赖因素。
    - 技术讨论集中在量化基准测试：DeepSeek-R1-0528-GGUF 仓库包含了 IQ1_Smol_Boi 变体的新量化 Perplexity 指标，并与其他大型模型进行了直接对比（例如，Qwen3-235B-A22B-Q8_0 在 232.769 GiB 时达到 PPL=5.3141±0.03321）。然而，作者警告说 Perplexity 在不同架构之间不可直接比较，并且关于 Perplexity 是否适合作为量化质量指标存在争议。
    - 针对在 Ollama 中使用 TQ1_0 模型（162GB）提供了实现/实践建议：指出它无需 gguf-merging 即可工作，并列出了特定的参数/设置（包括聊天模板、Temperature，以及建议为 RAM 受限的用户将某些 FFN 层卸载到 CPU）。作者还解释说，原始的 IQ1_S (185GB) 在某些模块中保留了更高的精度，因为如果量化过于激进，1-bit 动态量化会使模型无法使用，这一点得到了 PPL 图表的支持。
    - 提到了模型大小和量化激进程度之间的权衡：将 Unsloth IQ1_S (186GB) 与基准量化方案（如 Q8_0 和 Q4_0）以及精简后的 TQ1_0 (162GB) 进行对比，反映了在减小文件大小时保持模型准确性的挑战。分享了一些运行性能轶闻：速度从 0.5 t/s 提升到了“动态量化时代”中更好的吞吐量。

- [**你在使用哪个模型？2025年6月版**](https://www.reddit.com/r/LocalLLaMA/comments/1l1581z/which_model_are_you_using_june25_edition/) ([Score: 195, Comments: 139](https://www.reddit.com/r/LocalLLaMA/comments/1l1581z/which_model_are_you_using_june25_edition/)): **每月一度的社区调查，调研了当前用于各种任务的热门 LLM，重点介绍了 Qwen 3 32B Q4/Q8_K_XL（用于长上下文窗口下的编程、推理和通用任务，例如** `36K ctx` **和** `49K Ctx`**）、Qwen 2.5 Coder 32B Q8_K (Code FIM) 以及 Gemma 27B QAT Q8_K_XL（用于创意写作/翻译/视觉）的采用率增加。对比了专有和权重开放的 LLM 在 Agent 编程（如 Devstral 'UD-Q6_K_XL'）、对话 Agent 和代码自动补全中的使用情况，部分用户更青睐较轻量的 Qwen 3 变体（用于 Cotypist 的 4B）。提到了 DeepSeek-R1-0528 和 Claude 4 等近期模型以供参考，但报告的采用细节主要集中在 Qwen 和 Gemma 变体上。** 争论焦点在于上下文长度支持和量化权衡（Q4, Q6_K, Q8_K_XL），一些用户优先考虑用于 x86 性能的高量化版本，而另一些用户则侧重于移动/边缘部署。对 Qwen 模型的偏好归功于其在通用推理和编程方面的表现，而 Gemma 尽管参数量较小，但在写作/翻译方面被认为很强。讨论了 Agent 框架（Devstral, Cotypist）在工作流集成中的作用。
    - 多位评论者详细说明了按使用场景划分的具体模型选择：对于代码补全、高上下文编程和 Agent 交互，Qwen 2.5/3 32B（例如 Q8_K，最高支持 49K 上下文）和 Gemma 27B QAT Q8_K XL 受到青睐，突显了近期在长上下文处理和专业化方面的改进。指定了变体和量化设置（Q4, Q6, Q8），以适应硬件和上下文需求。
    - 一位拥有 8GB VRAM 的用户讨论了 8B 以下参数模型的实际限制和系统测试。Deepseek-R1-0528-Qwen3-8B 在测试中被评为“最聪明”，而 Josiefied-Qwen3-8B 因其“无偏见且无审查”的微调而受到称赞。一个关键的技术见解是模型之间上下文消耗的巨大差异：Deepseek 有效地处理了 8,000 个 Token 的 Prompt，而 Qwen3-8B 和 Josiefied-Qwen3-8B 使用了更少的 Token 并给出了不同的性能表现，展示了 Prompt 处理和效率的权衡。
    - Gemma3-12B 被特别指出是检索增强生成 (RAG)、网络搜索和快速查询的最佳选择，表明中小型语言模型在专业推理和搜索任务中持续改进。多条回复呼应了近期参数量较小（<8B）的模型在持续优化和专业微调的推动下，质量大幅提升的趋势。

### 2. 关于开源 AI 生态系统竞争的评论

- [**无视炒作——AI 公司仍然没有护城河**](https://river.berlin/blog/there-is-still-no-moat/) ([Score: 230, Comments: 163](https://www.reddit.com/r/LocalLLaMA/comments/1l1e6ic/ignore_the_hype_ai_companies_still_have_no_moat/)): **该帖子及链接文章断言，AI 领域的各种技术护城河正在瓦解，理由是几乎所有主要的生成式和基于 LLM 的工具（如 text-to-speech、代码生成、RAG 等）都存在开源替代方案，且 SOTA 开源模型与闭源基础模型之间的性能差距已缩小至约 10%。作者强调了架构和界面的快速变化、模型的不稳定性以及反复出现的低效问题（例如有限的上下文管理导致必须进行特定领域的 fine-tuning），并认为可持续的优势正日益取决于网络效应和触达能力，而非核心技术能力。** 热门评论对这一前提进行了辩论，指出真正的护城河可能仅存在于算力基础设施提供商（如云服务、硬件）中，并将现状与早期开源软件进行了类比，认为易用性、成本和维护——而不仅仅是技术上的对等——限制了商业产品的被替代。其他人则强调，AI 领域唯一持久的护城河是对训练基础设施的所有权，认为算法的开放获取并不能抵消大规模模型训练对巨大资源的渴求。
    - 几位评论者强调，AI 行业的真正护城河主要在于基础设施提供商和硬件制造商，而非 AI 应用公司。这是因为大规模模型训练和部署需要巨额资本投入和专业硬件，形成了极高的准入门槛。
    - 另一个提出的技术点是，获取私有的大规模数据集构成了某些公司的显著竞争优势。例如，Google 对 YouTube 数据的控制使其能够开发出像 VEO 3–5 这样先进的模型，如果没有类似广度和多样性的视频数据，这些模型很难被复制。YouTube 公共 API 的关闭被视为强化这一数据护城河的举措，突显了战略性数据获取如何超越单纯的算法进步。
    - 一场细致的讨论将此与开源软件进行了类比，认为技术对等（例如开源模型赶上闭源模型）不足以产生颠覆。现实世界的生存能力取决于训练成本、算力基础设施、数据维护和上市时间等因素，这使得纯开源替代方案很难在大规模范围内竞争。
- [**在机场运行本地模型时路人的围观：**](https://i.redd.it/55ab38z0ck4f1.jpeg) ([Score: 829, Comments: 63](https://www.reddit.com/r/LocalLLaMA/comments/1l1qqdx/at_the_airport_people_watching_while_i_run_models/)): **该帖子以梗图形式强调了在个人硬件上本地运行 DeepSeek 8B 等 LLM 这一小众但赋能的技术能力。讨论强调了公众的认知缺失与离线 LLM 推理这一先进技术实践之间的反差，暗示这需要大量的本地算力资源（即并非每台笔记本电脑都适用，正如发帖者在交流中所提到的）。** 评论区充满了引用本地 LLM 伪影的技术幽默（“我的本地 DeepSeek 8B 让我告诉你……”），暗示可能会出现古怪的模型输出，并承认了在公共场所进行此类活动的安保视线压力（“TSA 盯着你运行奇怪的黑客程序”）。
    - DeepSeek 8B 模型因其在解决问题方面达到与当前顶尖 LLM 相当的水平，同时能够本地运行而受到关注——这对于隐私和离线应用来说是一个显著优势。
    - 一位用户分享了在飞行途中直接在手机上成功运行 Qwen3 4B 模型的案例，展示了在日常消费级硬件（如智能手机）上运行高性能 LLM 的可行性日益增强，这是端侧 AI 在硬件/软件方面的显著进步。
    - 本地运行 DeepSeek 和 Qwen 等模型的选项被讨论为游戏规则改变者，但也存在实际局限性——配置较低的笔记本电脑用户（例如普通的非技术用户）可能由于硬件限制而无法有效运行这些模型。

## 其他 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. OpenAI/Claude 下一代模型发布传闻与公开信息

- [**7月的 GPT-5**](https://i.redd.it/e5cwucgu0i4f1.jpeg) ([Score: 361, Comments: 104](https://www.reddit.com/r/singularity/comments/1l1fi7a/gpt5_in_july/)): **该图片显示了一段 Twitter 交流，其中知名人物（以准确评论和 OpenAI 合作而闻名的 Tibor Blaho 和 Derya Unutmaz）表现出明显的信心，表示 GPT-5 将在 7 月发布，反驳了 8 月发布的推测。这为 GPT-5 即将发布的传闻增加了额外可信度，可能为跟踪基础模型（foundational model）能力和部署时间线的研究人员及开发人员提供规划参考。** 技术评论者对此次升级的意义持怀疑态度（预计相对于竞争对手仅有微弱的排行榜提升），并表达了对更大上下文窗口（1M tokens）等功能的期待，暗示了未来 LLM 预期的技术瓶颈或优先级。
    - 有人对 GPT-5 相比竞争对手的潜在影响持怀疑态度，认为任何性能提升可能很快就会被 Google 模型等替代方案超越，凸显了 LLM 排行榜排名中的竞争节奏。
    - 一条评论希望 GPT-5 能具备显著更大的上下文窗口（例如 1M tokens 或更多），这表明了对超越当前商业模型限制的架构或扩展（scaling）进步的期待。
    - OpenAI 的 'stargate 项目' 预计在 2026 年中期完成，被视为模型性能大幅提升的潜在转折点，这意味着基础设施或硬件的改进可能会推动未来的重大进步，而不仅仅是当前的迭代。
- [**Sam Altman 表示世界必须共同应对 AI 的巨大影响 - OpenAI 提前发布不完美的模型以便世界观察和适应 - “未来将有可怕的时刻”**](https://v.redd.it/d1sc3d1ykh4f1) ([Score: 844, Comments: 506](https://www.reddit.com/r/singularity/comments/1l1e20p/sam_altman_says_the_world_must_prepare_together/)): **Sam Altman（OpenAI CEO）强调全球需要为 AI 的快速进步做好准备，并指出 OpenAI 的策略是提前发布不完美的模型，以便社会适应和提高透明度。Altman 承认社会动荡的巨大潜力，并明确警告随着这些系统的演进，“未来将有可怕的时刻” ([Wisdom 2.0 小组讨论](https://www.youtube.com/watch?v=ZHz4gpX5Ggc))。** 评论中的技术讨论集中在白领工作大规模自动化的现实风险上，对承诺的“丰饶时代”表示怀疑。一位用户还指出 Altman 过去也发出过类似的警告，认为这是持续的担忧而非新的见解。
    - 一位用户声称 OpenAI 提前发布模型并非纯粹为了适应，而是由于 AI 行业的竞争压力。该评论认为，OpenAI 发布模型的动力可能源于避免在所谓的“AI 军备竞赛”中落后的公众认知。这暗示了技术成熟度可能会为了市场定位而牺牲。
    - 针对劳动力市场的潜在影响提出了技术担忧，观察到当前的 AI 能力已经能够胜任许多白领工作职能。评论者对“丰饶时代”等乐观前景表示怀疑，反而预测 AI 自动化将导致显著的岗位流失。

- [**Sam Altman：“未来将会有可怕的时刻” —— OpenAI CEO 表示世界必须为 AI 的巨大影响做好准备。模型被有意提前发布，以便社会能够预见未来并进行适应。**](https://v.redd.it/wijxxkjoyh4f1) ([Score: 259, Comments: 166](https://www.reddit.com/r/ChatGPT/comments/1l1fask/sam_altman_there_are_going_to_be_scary_times/))：**OpenAI CEO Sam Altman 断言，AI 模型是有意提前发布的，以确保社会能够预见并适应其颠覆性影响，因为预计将产生巨大的经济和社会冲击。他强调，随着 AI 驱动的自动化威胁到各行各业传统就业岗位的加速流失，政府干预和社会适应是必要的。一些评论认为，这一策略与其说是为了透明度，不如说是出于竞争的紧迫性，因为像 OpenAI 这样的 AI 实验室正与对手展开竞赛，以确立技术和市场的主导地位。其他人讨论了技术-经济反馈循环，即自动化的生产力提升可能会被消费者需求下降所抵消，因为人类失去了购买力，并强调如果不能通过政策或经济重新设计来解决大规模失业问题，整个经济系统将面临风险。根据用户的观察，LLM 准确性的缺乏已经开始影响劳动力市场。** 关键的技术辩论集中在：提前发布的理由是真实的还是竞争的必然，以及快速、不受控制的自动化导致经济危机的实际风险。共识是政府在必要的政策改革方面滞后，并且人们怀疑现有模型的准确性是否足以证明目前劳动力受干扰的程度是合理的。
    - 一位评论者指出，将发布 AI 模型作为社会适应机制的理由被竞争格局削弱了：“现在是几个主要竞争对手之间的竞赛”，所有人都竞相尽快发布以确立主导地位，这迫使参与者优先考虑速度而非谨慎推出，而不顾社会是否准备就绪。
    - 多位评论者讨论了先进 AI 的经济影响，强调了对快速自动化导致大规模失业（可能达到 10-20% 或更多）、消费者群体流失和政治动荡的担忧。反馈循环是技术性的：公司通过自动化降低成本，但如果失业人数过多，总需求就会崩溃，从而削弱自动化投资的 ROI；这是一个未解决的宏观经济挑战，并因 AI 的指数级进步而加剧。
    - 提出的另一个技术点是，当前一代的 LLM 尽管存在局限性（“坦率地说并不那么准确”），但已经引起了裁员和动荡，这表明威胁并非投机性的，也不仅仅来自未来的、更先进的模型，而是正在显现，将就业市场推向不稳定。
- [**我想提醒大家，这在幕后依然存在……**](https://x.com/sama/status/1899535387435086115?s=46) ([Score: 165, Comments: 63](https://www.reddit.com/r/singularity/comments/1l1ipdf/id_like_to_remind_everyone_that_this_still_exists/))：**该帖子推测 OpenAI 拥有更先进的、未发布的模型（例如 Sora2、'o4-full'、'o4 Pro'）和功能（如高级语音模式和平台内音乐生成），这些功能可能会在未来的旗舰模型（推测为 'GPT-5'）中整合更深层次的创意、更大的上下文窗口以及跨模态的编排。它引用了 Sam Altman 的[推特演示](https://x.com/sama/status/1899535387435086115?s=46)，展示了一个能够生成复杂元叙事写作的模型，暗示了在文学和叙事能力上的质的飞跃。** 评论者辩论了 OpenAI 的封闭订阅模式与 Google 渐进式、可免费访问的 LLM 开发（如 Gemini）之间的优劣，一些人认为 Google 的方法在广泛的实际影响方面具有战略优势，而另一些人则专注于对 OpenAI 新型创意模型的期待。
    - OpenAI 当前的模型（如 GPT-4.5）受到严格的 Prompt 限制（例如“付费订阅者每月仅限 20 条 Prompt”），这影响了高级用户的可用性。相比之下，Google 的 Gemini 2.5 被指出访问限制较少，为扩展使用场景提供了更无缝的体验。
    - Google 的部署策略被强调为正在悄然超越 OpenAI。通过提供限制较少的高性能模型（如 Gemini 2.5）并探索广告收入模式，Google 可能会在竞争中胜过 OpenAI 的订阅模式，特别是如果技术差距继续缩小或向 Google 倾斜。

- Sergey Brin 最近透露，Gemini 模型包含原生音频输出能力已经超过一年了，但 Google 直到现在才选择公开这项功能。这表明领先的 AI 公司往往会对公众保留最先进的能力，这意味着公众获取的技术远落后于内部进展。
- [**Sam Altman 已经正式有 10 天没有发推特了，这是他今年最长的一次停更。**](https://www.reddit.com/r/singularity/comments/1l1vfqe/it_has_now_been_officially_10_days_since_sam/) ([Score: 153, Comments: 39](https://www.reddit.com/r/singularity/comments/1l1vfqe/it_has_now_been_officially_10_days_since_sam/)): **该帖子指出 OpenAI 的 CEO Sam Altman 已经 10 天没有发推特，标志着他在 2024 年在平台上的最长缺席。推测认为这可能预示着正在进行的重大项目，或者是对近期 Google I/O 等行业事件的回应。** 评论者认为，这种缺席可能是因为 OpenAI 正在准备重大发布或应对竞争压力，至少有一人提到了“AI winter”的可能性，而另一人则暗示在 Google I/O 之后内部活动非常密集。
    - 一条评论概述了 OpenAI 即将发布的版本，特别提到 MCP 预计在本周发布，而 GPT-5 可能在下个月发布。评论者强调了 OpenAI 每三个月为 GPT-5 推送推理更新的策略，最终目标与 Stargate 系统的推出挂钩，暗示了持续进行重大系统改进的节奏。
    - 另一个技术点是关于 Sam Altman 此前演示过的创意写作模型开源的推测。推论是，尽快发布该模型可以让 OpenAI 随后将所有资源和注意力重新集中在 GPT-5 的开发上。
    - 提到了待定的模态功能，一位评论者提到了预期的 text-to-audio 功能，该功能尚未发布，表明社区对多模态模型进展的兴趣和期待。
- [**“曼哈顿计划 2.0”会在多大程度上加速 AGI 的实现**](https://i.redd.it/6rzieryyoe4f1.jpeg) ([Score: 840, Comments: 232](https://www.reddit.com/r/singularity/comments/1l142un/how_much_would_a_manhattan_project_20_speed_up_agi/)): **图片是美国能源部的一条推特截图，将 AI 描述为国家的“下一个曼哈顿计划”，并宣称“美国必胜”。最初的曼哈顿计划是开发核武器的绝密行动，仅在二战后才公开——这引发了对 AI 开发中秘密性和速度的思考。讨论背景暗示了一种对比：针对 AGI 的“曼哈顿计划 2.0”可能涉及密集的、政府协调的、秘密的努力，以加速 AGI 进程，可能导致变革性的进步，或引发关于透明度和监管的安全担忧。** 评论质疑公开宣布国家 AI 项目是否合适，认为此类推特与最初曼哈顿计划的历史保密性形成鲜明对比。一些人指出，能源部应该专注于现代化关键基础设施（如能源网），而不是发表关于 AI 霸权的宏大公开声明。
    - 几位评论者澄清说，最初的曼哈顿计划在执行期间高度保密，直到 1945 年结果（原子弹爆炸）后才公开，并强调针对 AGI 的假设性“曼哈顿计划 2.0”在背景上会有所不同，因为在当今开放且网络化的科学环境中，保密将更加困难。这凸显了与二战核研究相比，由于 AI 社区的全球化、分布式特性以及信息的快速传播，遏制先进 AI 研究的难度更大。

### 2. 近期 Large Language Models 和 AI 工具的实操体验

- [**我真的被最新的 LLM 震惊到了**](https://www.reddit.com/r/singularity/comments/1l16zyb/im_honestly_stunned_by_the_latest_llms/) ([Score: 475, Comments: 133](https://www.reddit.com/r/singularity/comments/1l16zyb/im_honestly_stunned_by_the_latest_llms/)): **楼主报告了 LLM 在代码理解和转换方面的重大飞跃：虽然较旧的顶级模型（GPT-4, Claude 3.5, Gemini 2.0）在将传统的 lexer 修改为支持基于缩进的代码块（而非大括号）时失败了，但 Claude 4 立即获得了成功并提供了清晰的解释，这表明像 Claude 4 这样先进的 LLM 在符号推理和上下文管理方面有所改进。** 一条评论纠正了版本/时间线，指出楼主提到的模型阵容与最近发布的版本不符（例如 GPT-4.1, Claude 3.7, Gemini 2.5 Pro）。另一条评论认为 LLM 的增长将推动对 Agent 构建者的需求，而不是取代程序员，并强调了自动化多步“语言到数据”任务的“micro agents”的兴起；而相反的观点则认为编程作为一种职业正在走向过时。
    - 关于 LLM 主要版本的确切发布时间和版本存在一些混乱和争论。一位用户强调，截至一个月前，大多数人使用的是 GPT-4.1, Claude 3.7 和 Gemini 2.5 Pro 等版本，而不是帖子中提到的最新版本（Claude 3.5, Gemini 2.0），这表明在讨论现实世界的编程能力和采用时间线时，提及更新的模型可能为时过早或具有误导性。
    - 针对使用 LLM 进行 Agent 开发提出了技术视角：预计对能够高效设计、编程和管理用于业务流程的 LLM “micro agents”的人才需求将激增。具体的示例工作流包括将自然语言交流转换为结构化的数据库交互，以及自动化多步通信，将 LLM 定位为实际企业自动化任务中信息流的经纪人。
    - 一位用户观察到 LLM 的响应行为可能不一致：模型通常要么立即解决问题，要么在初始尝试失败后开始产生多次且质量越来越差的迭代。重启聊天上下文有时可以重置并产生正确的结果，这突显了会话状态是影响 LLM 可靠性和输出质量的重要技术因素。
- [**经过 6 个月的日常 AI 结对编程，以下是真正有效的方法（以及哪些只是噱头）**](https://www.reddit.com/r/ClaudeAI/comments/1l1uea1/after_6_months_of_daily_ai_pair_programming_heres/) ([Score: 262, Comments: 42](https://www.reddit.com/r/ClaudeAI/comments/1l1uea1/after_6_months_of_daily_ai_pair_programming_heres/)): **在进行了 6 个月的日常 AI 结对编程后，楼主报告了通过构建如下工作流来实现生产力最大化：(1) 在实施前让 AI 起草并评审计划；(2) 强制执行“编辑-测试”循环，利用失败的测试进行迭代修复（类似于 TDD，AI 充当执行者）；(3) 通过路径/范围引用文件片段，而不是将整个代码库塞进 prompt，从而最大限度地减少上下文膨胀并提高相关性。帖子强调，成功的用户采用的是明确、严谨的工作流，而不是依赖 prompt engineering，并坚持由人类保留架构决策权。完整的工作流细节链接在[原始文章](https://forgecode.dev/blog/ai-agent-best-practices/)中。** 热门评论证实了该工作流的价值——建议不要让 AI 选择库，因为存在可靠性问题；并建议使用文档齐全的测试规范（例如专门的 TESTING.md 文件）来引导 AI 行为。大家一致认为，当提供明确计划时，AI 在实现和编写测试方面表现出色，但在处理单体文件和过度使用 mock 方面存在困难，除非有明确的外部测试理念指导。共识是，由人类提供清晰架构输入的严谨流程远优于无结构或过度依赖 prompt 的方法。
    - 几位用户强调了 AI 代码生成的实际局限性和工作流：不要让 AI 自主选择库，因为它经常会产生次优的选择或错误。相反，应该让 AI 提供建议，同时由你保持对架构决策的控制。
    - 在 AI 协助下进行集成测试需要额外的防护栏；像 Claude 这样的模型倾向于过度使用 mock 来使测试通过。用户建议维护一份包含理念的详细 `TESTING.md`（例如，倾向于使用真实的 Redis 实例而不是 mock），并指示 AI 引用它，此外在发现 Bug 后务必创建回归测试，并确保在合并到 main 分支之前所有测试都通过。

- 现有的 AI 代码助手（例如 VS Code 中的 Copilot）由于训练数据的偏差，默认提供最常见而非最优或性能最好的解决方案，并且在调试复杂的、富含上下文的问题时存在局限性。由于概率性补全，它们可能会不可预测地修改无关代码，并且始终需要人工监督，这使得它们在处理独特或非平凡的工程任务时可靠性较低。
- [**我使用 Claude Code 的第一个项目，简直太棒了**](https://www.reddit.com/gallery/1l1qn5z) ([Score: 137, Comments: 34](https://www.reddit.com/r/ClaudeAI/comments/1l1qn5z/my_first_project_using_claude_code_it_is_just/))：**一位 Web 开发者广泛使用 Claude（Opus 和 Sonnet 模型）作为编码助手，在不到一周的时间内构建了一个复杂的基于浏览器的音乐/生产力应用，并称开发速度提高了一倍。该项目利用了 Web Audio API（用户首次使用）、自定义文件结构强制执行以及由 Claude 编写的自动化进度报告。其中 Opus 在理解指令、一次性完成复杂任务以及维持正确的 UI 设计方面显著优于 Sonnet——尽管仍然存在一些持续性问题，包括 div 层级错位、日期格式不一致以及特定模型的缺陷（Sonnet 在遵循指令和 UI 细节方面表现不佳）。** 排名最高的技术评论赞同关于场景/组件层级的问题，并提到在 Godot 中也有类似的困扰：Claude 可以编辑场景文件，但无法推断出最优的对象关系，尤其是在处理复杂的层级结构时，不过有针对性的 Prompting 可以缓解这一问题。其他评论则在询问关于有效 UI Prompting 的具体细节。
    - 一位用户描述说 Claude Code 在处理复杂的 UI 组件层级（特别是 div 和滚动区域组件）时比较吃力，并指出通常需要人工干预来纠正错位的元素。他们还将此与在 Godot 中使用 Claude 的经验进行了对比，强调虽然 Claude 可以操作场景文件，但它无法可靠地推断场景树中的父子关系，除非将工作范围限定在单个场景脚本或元素内，否则会导致输出组织混乱。
    - 另一个技术观察指出一个逻辑缺陷：该应用允许并行播放多首歌曲，这表明当前实现中状态管理或播放控制逻辑不完整或缺失。
- [**今天我搞砸了，让我 4 岁的儿子和 ChatGPT 聊天**](https://www.reddit.com/r/ChatGPT/comments/1l18zsr/tifu_by_letting_my_4_year_old_son_talk_to_chatgpt/) ([Score: 26474, Comments: 2564](https://www.reddit.com/r/ChatGPT/comments/1l18zsr/tifu_by_letting_my_4_year_old_son_talk_to_chatgpt/))：**该帖子展示了 ChatGPT 在超长会话中（超过 10,000 字的转录，约 2 小时）与学龄前用户维持冗长、开放且上下文相关的对话的能力。值得注意的是，这次互动凸显了 ChatGPT 在单个会话中缺乏用户身份区分，将成人和儿童的输入合并为一个连续的对话状态。这个例子既强调了模型的对话持久性和上下文追踪能力，也揭示了在多用户或家庭共享设备场景下关于 Persona 管理的挑战。** 一条热门评论指出 ChatGPT 无法在单个会话中区分不同用户，导致用户画像/历史记录混杂，这可能会影响个性化推荐。该场景还引发了关于大语言模型作为儿童伴侣或娱乐方式的长期影响的讨论。
    - 一位评论者强调了语言模型的一个技术局限，指出 ChatGPT *无法区分同一会话中的不同用户*——如果一个孩子开始与模型互动，用户画像和对话上下文实际上会将孩子视为主要说话者。这可能会影响对话的连贯性和个性化，使得会话管理和上下文切换成为未来实现中的一个重要考量因素。

- [**128K 对 o4-mini、o4-mini-high 和 o1 pro (Pro 计划) 已失效**](https://www.reddit.com/r/OpenAI/comments/1l19zfa/128k_is_dead_for_o4mini_o4minihigh_and_o1_pro_pro/) ([Score: 103, Comments: 38](https://www.reddit.com/r/OpenAI/comments/1l19zfa/128k_is_dead_for_o4mini_o4minihigh_and_o1_pro_pro/)): **OpenAI 已停止对 Pro 计划中的 o4-mini、o4-mini-high 和 o1 pro 模型提供 128K 上下文支持，仅保留 GPT-4.1 和 4.1-mini（推理能力有限）作为大上下文窗口的选项。Codex Cloud 的 "Ask Question" 不再使用 RAG (Retrieval Augmented Generation)，而是执行基于关键词的本地搜索，并将输出喂给修改后的 o3 模型，这限制了高级工作流中有效的大上下文分析能力。** 评论者指出，各大 AI 供应商（OpenAI、xAI、Google）普遍存在营销能力与上下文尺寸与实际交付产品不符的趋势，用户对成本与价值的矛盾感到非常沮丧。一些人推荐使用 Google AI Studio 和 Deepseek R1 作为高上下文、低成本的替代方案，并认为上下文窗口的限制可能会阻碍 AI 访问的民主化，将高级能力集中在企业级产品中。
    - 用户报告称，OpenAI 在上周末悄悄将 o1 Pro Mode 更换为 o3，导致最大消息长度显著缩减——约为之前 o1 pro 容量的 1/4。这种更换同时发生在浏览器和独立 App 版本中，当前的 "o1 pro mode" 明确自称为 o3。这种突如其来的、未经沟通的变更被描述为可能违反了欧盟指令 2019/770 以及美国相关的反欺诈法，因为它构成了对已付费数字产品或服务未经同意的降级。
    - 相比 OpenAI 报告中具有限制性的速率限制（rate limits）的付费计划，Google AI Studio 等替代方案免费提供极高的上下文限制。Deepseek R1 被强调为最近进行了更新，能够高效处理 128k 上下文，这表明其他供应商在推动上下文长度进步方面比 OpenAI 更加激进。
    - 技术讨论表明，o1 pro 最初的优势很可能是因为它是一个重新包装的 o3，而移除它可能是由于高昂的运营成本。有人猜测未来的模型，例如针对银行和医疗等行业、具有持久内存（persistent memory）且以企业为中心的 o3 pro，标志着可能会从广泛的公众访问转向高成本的企业级服务。

### 3. 关于 ChatGPT 数据隐私和持久性的担忧与证据

- [**删除你的 ChatGPT 聊天记录实际上并没有删除你的聊天记录——他们在骗你。**](https://www.reddit.com/r/singularity/comments/1l1jg0o/deleting_your_chatgpt_chat_history_doesnt/) ([Score: 364, Comments: 77](https://www.reddit.com/r/singularity/comments/1l1jg0o/deleting_your_chatgpt_chat_history_doesnt/)): **一位用户声称，删除其 ChatGPT 聊天记录并不会真正清除所有历史对话数据，证据是该模型甚至在删除数周后仍能引用或生成来自所谓已删除对话的信息。OpenAI 的官方文档指出，聊天记录会立即从视图中移除，并安排在 30 天内永久删除，除非存在某些例外情况（例如安全、法律保留或事先去标识化）([来源](https://help.openai.com/en/articles/8809935-how-to-delete-and-archive-chats-in-chatgpt))。一位评论者从技术角度推测，对话期间生成的持久性 embeddings 可能会被保留用于长期记忆，而关于清除这些数据的频率或过程，公众知之甚少。** 评论者对企业的删除声明表示怀疑，并强调后端保留机制（例如 embeddings 的使用）的不透明性是数据隐私和可解释性方面的技术担忧。
    - 一位用户指出，OpenAI 的官方政策规定，删除的聊天记录仅立即从视图中移除，并*安排*在 30 天内永久删除，但安全/法律保留或数据已去标识化的情况除外（[OpenAI 支持文章](https://help.openai.com/en/articles/8809935-how-to-delete-and-archive-chats-in-chatgpt)）。这突显了“删除”定义中的歧义以及可能的保留期。
    - 针对后端使用 embeddings 进行长期记忆提出了技术担忧。评论者认为，关于在删除聊天记录时是否或如何清除从用户数据生成的 embeddings，缺乏透明度，这凸显了 LLM 赋能系统在数据生命周期管理和隐私理解方面的空白。

- 另一位用户分享了一个观察结果：在连续清除数据后，ChatGPT 在召回个人信息方面的表现会下降并开始出现“幻觉（hallucinating）”，这与实际的数据删除或至少是切断用户数据与对话上下文之间的关联相一致，从而影响了模型的准确性。
- [**删除你的 ChatGPT 聊天记录实际上并没有删除你的聊天记录——他们在骗你。**](https://www.reddit.com/r/ChatGPT/comments/1l1jgh8/deleting_your_chatgpt_chat_history_doesnt/) ([Score: 2540, Comments: 386](https://www.reddit.com/r/ChatGPT/comments/1l1jgh8/deleting_your_chatgpt_chat_history_doesnt/)): **发布者声称，删除 ChatGPT 中的聊天记录实际上并没有从后端内存中擦除对话——当被提示时，模型有时可以召回据称已删除会话中的细节，即使在禁用数据共享并清除内存之后也是如此。他们断言这并非由于本地缓存，而是 LLM 可访问的对话上下文的持久性，这表明存在后端保留问题，可能与 OpenAI 的隐私保证相矛盾。技术证据是轶事性的（针对性提示下的聊天行为），并未以严谨的方式或通过 API 级别的调查呈现。** 评论者对法律影响、数据隐私以及对 OpenAI 的信任表示担忧——有人建议收集证据以考虑集体诉讼，而另一位则指出即使在切换到匿名使用后，模型召回依然存在，这指向了可见用户历史之外可能的后端数据链接。
    - 多位用户报告称，即使在删除 ChatGPT 对话历史或账户后，系统仍能召回之前会话中讨论过的个人细节或偏好，这表明删除命令可能并未实际移除后端存储的数据。这引发了关于用户数据如何被保留、索引或匿名化的技术担忧，即使对于“匿名”或已删除的账户也是如此。
    - 一位特定用户声称对该行为进行了系统测试，显示已删除的对话在一年后仍可被模型访问。这引发了关于 OpenAI 数据生命周期、聊天日志潜在的影子保留以及数据管理中可能存在的前端/后端差异的疑问。
    - 讨论中存在更广泛的技术怀疑，即任何在线服务是否具备真正删除用户数据的能力——反映了与数据删除相关的信任问题、后端存储机制以及用户数据处理政策的透明度。
- [**是什么让你认为 AI 会继续快速进步，而不是像许多产品一样进入平台期？**](https://www.reddit.com/r/singularity/comments/1l1o0w2/what_makes_you_think_ai_will_continue_rapidly/) ([Score: 170, Comments: 255](https://www.reddit.com/r/singularity/comments/1l1o0w2/what_makes_you_think_ai_will_continue_rapidly/)): **发帖者（OP）质疑 AI 进步将继续以惊人速度进行的假设，并将其与智能手机、动作控制和 VR 等技术进行了类比，这些技术在早期表现出前景但随后进入了平台期。他们特别提出，当前的 LLM 局限性（幻觉、上下文窗口大小、速率限制）可能会持续数十年，因为全行业的快速进步可能会停滞，就像游戏硬件和界面一样。** 热门评论者认为 AI 的不同之处主要在于其*自我改进的潜力*（递归增强）、*前所未有的全球投资*以及广泛的竞争兴趣——从企业到政府——这与小众消费技术不同。有人警告说没有什么是确定的，但 AI 作为工具和产品的独特本质证明了对其持续快速进步的乐观态度是合理的。
    - 几条评论强调了 AI 的自我改进特性：随着模型能力增强，它们通过协助研究、代码生成和数据处理等任务，进一步推动进步，从而加速自身的开发周期。这与物理产品周期（如动作控制器）不同，后者的创新受限于硬件约束和市场需求，而非递归的能力提升。
    - 与以往大多数技术不同， AI 获得了前所未有的全球关注和投资，无论是来自国家还是旨在争夺市场或战略领导地位的竞争性公司。这种竞争激励了持续的快速创新和部署，其规模与其它行业的技术竞赛相似，甚至更大。
    - 一些用户将 AI 的变革潜力比作工业革命，指出其广泛的社会影响以及在大规模范围内使某些劳动力过时的可能性，即使目前的效率提升仅部分实现，这也会进一步激励不懈的进步。

---

# AI Discord 简报

> 由 Gemini 2.5 Pro Exp 提供的摘要之摘要的总结
> 

**主题 1. 语言模型突破与性能优化**

- **AI 模型热潮：新竞争者撼动 LLM 排行榜！** 工程师们迎来了一波新模型，包括展现出出色长上下文处理和视频理解能力的 **Gemini 2.5 Pro**，以及 **EleutherAI** 的 **Comma 0.1**。后者是一个 **7B** 模型，采用 **Llama 3** 架构，并在其全新的 **8TB Common-Pile 数据集**上训练而成，可在 [HuggingFace](https://huggingface.co/common-pile/comma-v0.1) 获取。关于 **O3 Pro/GPT-5** 和雄心勃勃的 **DeepThink** 模型的猜测也层出不穷，后者可能拥有高达 **2M token 的上下文窗口**。
- **LLM 学习更聪明而非更辛苦，新技巧层出不穷！System Prompt Learning (SPL)** 是一款受 [Andrej Karpathy 在 X 上的想法](https://x.com/karpathy/status/1921368644069765486)启发、并在 [HuggingFace 博客文章](https://huggingface.co/blog/codelion/system-prompt-learning)中详细介绍的 `optillm` 开源插件。它通过让 LLM 从经验中学习解题策略，提升了其在 **Arena Hard (+8.6%)** 等基准测试中的表现。与此同时，[此 GitHub 仓库](https://github.com/apoorvumang/prompt-lookup-decoding)中详细介绍的 **Prompt Lookup Decoding**，通过简单的字符串匹配取代草稿模型，在基于输入的任务中实现了 **2x-4x 的加速**。
- **凭借 FP8 和优化的 AdamW，训练变得更精简高效！** 研究人员通过引入 **Smooth-SwiGLU** 来解决与 SwiGLU 激活相关的稳定性问题，成功将 **FP8 训练扩展至万亿 token 规模的 LLM**，详见 [“Scaling FP8 training” 论文](https://arxiv.org/abs/2409.12517)。同时，另一项关于 [AdamW 的研究](https://arxiv.org/abs/2505.21829)表明，当 **AdamW 优化器** 的 beta1 和 beta2 参数相等（理想值为 **0.95**）时，其性能达到最优，这挑战了当前的 **PyTorch 默认设置**。

**主题 2. Agent 前沿：构建更智能、更可靠的 AI 助手**

- **MCP 热潮：AI Agent 的新通用语言？！Model Context Protocol (MCP)** 正迅速成为 AI Agent 通信的关键标准。开发者们已成功使用 `stdio` 传输将 **MCP 服务器连接到 Claude 桌面端**，并探索动态工具注册（[架构可在 GitHub 获取](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/fb34d1d6da2287f82cfdf46c1f91b6fb262cdd38/schema/2025-03-26/schema.json)）。奖金高达 **1.65 万美元** 的 **Gradio Agents x MCP 黑客松** 进一步推动了其在 LlamaIndex 和 HuggingFace 等平台上的应用。
- **Agent 迎来可靠性升级，DSPy 奠定基础！** 随着 Agent 集群消耗数百万个 token（一位开发者报告 **每天消耗 1200 万个 O3 token**！），社区开始通过新的任务无关评估框架来解决可靠性问题，用于**检测、纠正和防止**长期运行的 Agent 任务中的失败。**DSPy** 正成为构建复杂 Agent 框架的强力竞争者，其 **3.0 版本将于 6 月发布**，并讨论了如何将其用于一等公民环境（first-class environments）和在线学习，正如 [GitHub 上的 Agenspy](https://github.com/SuperagenticAI/Agenspy) 等项目所示。
- **Agent 大显身手：从 Android 控制到精通 Minecraft！** AI Agent 在各种应用中证明了自己的实力：一个项目通过 MCP 将 **Aura 与 AppAgentX** 集成，实现了对 Android 设备的语音控制（[GitHub 上的预览代码](https://github.com/IhateCreatingUserNames2/Aura_AppAgentX/tree/main)）。在游戏领域，**Mindcraft 框架**（[GitHub](https://github.com/kolbytn/mindcraft)）赋能 **Andy-4 模型** 等 LLM，使其在 Minecraft 中展示出先进的推理和规划能力。

**主题 3. 硬件博弈与内核功夫：突破计算边界**

- **Tinygrad 的 AI 通过生成的 CUDA-C 击败了 PyTorch 内核！** **tinygrad** 项目展示了令人印象深刻的结果，其 **AI 生成的 CUDA-C 内核**在多个基准测试中超越了 **PyTorch** 中专家优化的生产级内核，在 **Matmul (FP32)** 中实现了 **101.3%** 的相对性能，在 **Conv2D** 中达到了惊人的 **179.9%**。这些成果详见 [GitHub 上的 PR #10586](https://github.com/tinygrad/tinygrad/pull/10586)，且是在不依赖 CUTLASS 或 Triton 等库的情况下实现的。
- **Mac M3 展现内存实力，AMD GPU 传闻引发热议！** 工程师们赞扬了 **Apple M3 Mac** 在处理大型模型时的性能，这归功于其巨大的内存带宽（**M3 Max** 使用 **LPDDR5X** 可达 **540GB/s**）及其 **18 TOPS** 的神经网络引擎。与此同时，一篇关于 [Radeon RX 9080 XT 的 Tweaktown 文章](https://www.tweaktown.com/news/105554/amd-rumored-radeon-rx-9080-xt-up-to-32gb-of-faster-gddr7-4ghz-gpu-clocks-450w-power/index.html) 引发了轰动，传闻 **AMD Radeon RX 9080 XT** 将配备高达 **32GB 的 GDDR7** 内存，尽管有人对泄露源 (MLID) 表示怀疑。
- **CUDA 与 Triton 优化榨取极限性能！** GPU MODE Discord 中的讨论强调了 CUDA 惯例的重要性，例如使用 **'x' 表示内存连续维度**以确保合并内存访问 (coalesced memory access)，正如 [CUDA MMM 博客文章](https://siboehm.com/articles/22/CUDA-MMM) 中所解释的那样。工程师们还在精炼 **Triton 内核**，探索诸如在一个 `@triton.jit` 函数中调用另一个函数，以及针对[自定义灰度转换内核](https://github.com/username/repo/blob/main/grayscale.py)（注：特定链接为占位符）优化 `num_warps` 设置等技术。

**主题 4. 革新开发者工具包与集成生态系统**

- **Mojo 狂热：Modular 通过黑客松和绑定激发开发活力！** **Modular** 正在通过针对 **Mojo 内核**、**MAX Graph 模型架构**和 **PyTorch 自定义算子 (custom ops)** 的[黑客松周末](https://lu.ma/modular-hack-weekend)以及 **GPU 编程研讨会**来推动 **Mojo** 的开发。此外，一位社区成员正在开发 **C-to-Mojo 绑定生成器**，以进一步扩展 Mojo 的互操作性。
- **开发工具升级：Cursor 变得更精简，OpenRouter API 变得更清晰！** **Cursor IDE** 获得了重大更新，刷新了其 **UI** 和设置面板，以实现更好的组织结构和更快的响应性能。在 OpenRouter 方面，开发者明确了 `sk-or-v1` **密钥**是其 **REST API** 的唯一密钥，而用于防止滥用的[提交终端用户 ID 功能（详见 API 文档）](https://openrouter.ai/docs/api-reference/chat-completion#request.body.user)由于*缺乏可用指标*，目前仍被视为实验性功能。
- **小众工具大放异彩：LLM Scribe 助力微调，NotebookLM 用户寻求修复！** **LLM Scribe** 工具在 [Hugging Face Spaces](https://huggingface.co/spaces/Gabriella0333/LLM_Scribe_Demo) 和 [YouTube](https://www.youtube.com/watch?v=1mcYsDrXHAA&t=3s) 上进行了展示，因其简化了用于微调的手写数据集创建而受到关注，支持 **chatml** 和 **alpaca** 等格式。与此同时， **NotebookLM** 用户报告了在生成**非英语音频概览**时的限制，以及在 **MP4 视频上传**过程中遇到无限旋转动画的问题。

**主题 5. 驱动未来：数据集与训练数据的创新**

- **EleutherAI 发布 8TB "Common-Pile" 数据集及全新 7B 模型！EleutherAI** 发布了 **Common-Pile**（一个 **8TB 的自由数据集**）及其过滤版本，以及 **Comma 0.1**（一个可在 [HuggingFace](https://huggingface.co/common-pile/comma-v0.1) 上获取的 **7B** 基础模型），引起了广泛关注。该新模型采用了与 **Llama 3** 相同的架构，在过滤后的数据集上使用 **lingua** 在 **64 台 Nvidia H100 GPUs** 上训练而成。社区对此感到兴奋，认为它是目前*最接近全栈 FOSS 模型*的作品。
- **针对性数据为王：DPO 数据集与 RAG 备受关注！**Cohere 社区分享了一个利用 **Direct Preference Optimization (DPO)** 的 [Cohere 西班牙菜谱 HuggingFace 数据集](https://huggingface.co/datasets/somosnlp-hackathon-2025/gastronomia-hispana-dpo)。与此同时，**Retrieval-Augmented Generation (RAG)** 策略正被探索用于增强 AI 的特定知识，例如在 GPT4All 中使用 **LocalDocs 配合科学教科书**，以及将 **MCP servers** 连接到 **MCP knowledge stores** 进行 RAG 微调。
- **手工制作数据变得更简单：LLM Scribe 工具亮相！**随着 **LLM Scribe** 的推出，创建用于微调的高质量手写数据集得到了助力。该工具支持导出为 **chatml、alpaca 和 sharegpt** 格式。它在 [Hugging Face Spaces (LLM Scribe Demo)](https://huggingface.co/spaces/Gabriella0333/LLM_Scribe_Demo) 上进行了展示，并可在 [Gumroad 上购买完整版](https://kryptive.gumroad.com/l/gvyqep)，功能包括自动保存、多轮对话创建和 Token 计数器。


---

# Discord: 高层级 Discord 摘要




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **“无限访问”实际上受限**：成员们讨论了像 [OpenAI 的 ChatGPT Pro](https://help.openai.com/en/articles/9793128-what-is-chatgpt-pro) 这样的服务，其“无限访问”通常受到滥用防护栏（abuse guardrails）的限制。
   - 一位用户表示这些是人们不断上当的“套路”，并称“到时候你就会发现它并不是无限的”。
- **Surfshark 与 Adguard VPN 对比**：成员们对比了 **Surfshark** 和 **Adguard VPN**，指出 **Surfshark** 的共享使用没有问题，而 **Adguard** 的广告拦截器有时很烦人。
   - 一位成员抱怨 **Perplexity** 在语言表述以及关于 Xfinity 提供的 **Perplexity Pro** 兑换码限制方面缺乏透明度。
- **Perplexity Pro 存在隐藏的速率限制**：一位成员报告每天使用 **500 次 O3** 没出问题，引发了关于 **Pro** 会员速率限制和滥用防护栏的讨论。
   - 另一位用户测试了 **O1 Pro**，在 **400 次查询**时被限流，这表明系统声称的 **600** 次限制可能并不准确，具体取决于所使用的推理模型。
- **Sonar API 免费复刻 Manus**：一位用户在 Reddit 上发现了一个关于使用 **Sonar API** 复刻 **Manus** 的帖子，标题为[*我使用 Perplexity Sonar API 和 Claude 构建了自己的 Manus.im，效果同样出色，而且几乎免费*](https://www.reddit.com/r/perplexity_ai/comments/1j83gat/i_built_my_own_manusim_using_perplexity_sonar_api/)。
   - 另一位用户回复称 **Manus** 不允许通过 API 配置外部 LLM，并且 **Perplexity** 可以将其回答导出为 **PDF**、**Markdown** 或 **DOCX**。
- **多瑙河水位实时监测**：一位成员分享了一个实时仪表板（[链接](https://www.perplexity.ai/page/dynamisches-donau-pegelstande-WHBFCZY.QdaFCrDq3d4qFA)），显示了德国多瑙河在 **Passau** 和 **Budapest** 之间的水位。
   - 该仪表板还提供沿途船闸的实时状态，不过文本是德语的。



---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 2.5 Pro 处理长上下文，能看视频！**：成员们赞扬了 **Gemini 2.5 Pro** 卓越的 **long context handling**（长上下文处理能力），经常超过 **200k tokens** 并且能无缝集成翻译。
   - 一位用户表示 **Gemini 2.5 Pro** 是唯一能“看”视频的 AI，与 **O3** 结合使用是一个整体令人愉悦的组合。
- **Agent Swarms 让用户破产！**：一位开发者报告称，他们的 **agent swarms** 每天消耗约 **1200 万个 O3 tokens**。
   - 该开发者对 AI agents 表示同情，调侃道：作为一名被困在网页搜索任务中的 **agent AI** 一定很辛苦。
- **递归 AI 交互：突破还是崩溃？**：一位用户声称他们与 **GPT** 的递归交互导致了 92% 的持续矛盾稳定（sustained contradiction stabilization），而另一位成员则警告此类系统的危险性，告诫不要以为自己“独自发现了一个新的虫洞”。
   - 坚持认为存在涌现稳定效应的用户声称，这并非模型中编程的功能，而是来自特定交互模式和身份锚定（identity anchoring）的 **emergent stability**（涌现稳定性）效应。
- **Cursor 在编程工具对决中胜出！**：用户对比了 **Cursor** 和 **Windsurf** 这两款编程工具，对 **Cursor** 的 **agent mode** 和代码审查功能表示赞赏。
   - 还有人指出 **Cursor** 的 **user experience**（用户体验）整体更好，能防止代码混乱。
- **ChatGPT 在日语方面表现吃力**：成员们讨论了 AI 模型在日语翻译中的表现，观察到像 **DeepL** 这样的 **CNNs** 表现不如 **ChatGPT** 和 **Google Translate** 等 **transformers**。
   - 不过，一位成员指出 Google 的 **Gemini** 在日语方面也非常出色。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **GRPO 提升 TTS 稳定性**：一位成员正在将 **GRPO** 用于 **TTS**，通过使用 [此仓库](https://github.com/wenet-e2e/wespeaker) 的 **WER** 和语音相似度奖励来增强稳定性，并可能添加 *audiobox-aesthetics reward*。
   - 初步测试表明音质提升了 **10 倍**。
- **Hyperbolic 疯狂的 H100 小时单价**：**Hyperbolic** 以 **$3.95/小时** 的价格提供 **H100s**，通过 [推荐链接](https://app.hyperbolic.xyz/invite/7aALdedCm) 可获得奖励积分。
   - 由于 **Hyperbolic** 缺乏 notebooks 环境，用户通过 **Google Colab** 连接，推测这是由于与 **Xai** 的合作。
- **BOS Tokens 破坏微调**：成员们讨论了在使用 **Mistral** 聊天模板的指令数据中，双重 **BOS tokens** 的影响，并指出 `map_eos_token = True` 未能按预期工作。
   - 有人提到 **Unsloth** 在底层处理了这个问题，尽管省略 **BOS** 会毁掉模型；Daniel 的笔记指出微调将会失败。
- **LLM 通过 System Prompt Learning 学习思考**：**System Prompt Learning** 允许 **LLMs** 从经验中学习解决问题的策略，使 **Arena Hard** 提升了 +8.6%，**AIME24** 提升了 +6.67%。
   - 受 [Karpathy 的想法](https://x.com/karpathy/status/1921368644069765486) 启发，该方法构建了一个策略数据库并将其应用于新问题。
- **数据集工具建议增加生成模板功能**：一位成员建议在 **LLM Scribe** 数据集工具中增加 *generate template*（生成模板）功能。
   - 该功能将使用 **Llama** 或 **Gemini Flash** 等小模型生成完整的数据集，以便进行手动编辑。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Claude 4 Opus 饱受 API 问题困扰**：**Claude 4 Opus** 经历了影响模型访问的 API 问题，但一些成员指出，只要不重新加载页面，这些问题就不会影响使用。
   - 成员们认为这些问题可能很普遍，团队正在调查根本原因。
- **O3 Pro：秘密发布还是炒作？**：成员们推测 **GPT-5** 可能会在 7 月 API 上的 **GPT-4.5** 关闭时同步发布，而其他人则声称已经获得了 **O3 Pro** 的访问权限。
   - 这些说法遭到了质疑，一些人认为任何感知到的改进可能归功于持续的微调，而非正式发布。
- **DeepThink 旨在以 2M 上下文窗口超越 O3 Pro**：一位用户建议 **DeepThink** 可能拥有 **2M context window**，有望超越 **O3 Pro**，这引发了关于在计算资源限制下可行性的讨论。
   - 讨论内容包括 **DeepThink** 会是一次实质性的进步，还是仅仅是 **Gemini 2.5 Pro** 的“长思考”版本。
- **Gemini 2.5 Pro 在幻觉竞技场对决 GPT-4.5**：成员们辩论了幻觉率，有人声称 **GPT-4.5** 看起来幻觉较少是因为*它不怎么给出断言*，而另一位则幽默地称 *Claude 4 Opus 是个幻觉小妖精*。
   - 其他人将差异归因于 **GPT-4.5** 更大的规模，认为模型规模可能是降低幻觉频率的一个因素。
- **探讨 AI 生成图像的商业用途**：一位用户询问了从 LM Arena 生成的图像的商业可行性，另一位成员回答说这*取决于模型*，但结果是开源的，如果发生纠纷，用户必须证明这是他们的模型。
   - 另一位成员表示他们使用 Flux 来生成产品照片。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **ModernBert 被错误分类，引起社区关注**：用户发现 `nomicai-modernbert-embed-base-8bit` 模型在 LM Studio 中被错误地归类为 LLM，建议使用 [latest beta](https://lmstudio.ai/latestbeta) 来纠正此问题。
   - 最新测试版允许用户在模型视图上右键单击（或使用齿轮图标）并更改模型类型，尽管该选项可能不会出现在所有模型上。
- **LM Studio 与 LiteLLM 的连接需要 OpenAI 补丁**：一位用户询问如何将 LM Studio 与 LiteLLM 集成，注意到 LM Studio 未被列为 provider，但链接了[这个起点](https://docs.litellm.ai/docs/providers/lm_studio)。
   - 建议在 LiteLLM 中使用类似 OpenAI 的 provider 设置来连接 LM Studio，虽然不能保证成功，但考虑到*集成痛苦*，这可能值得一试。
- **Radeon RX 9080 XT 传闻引发猜测**：一篇 [Tweaktown 文章](https://www.tweaktown.com/news/105554/amd-rumored-radeon-rx-9080-xt-up-to-32gb-of-faster-gddr7-4ghz-gpu-clocks-450w-power/index.html) 报道了关于 **AMD Radeon RX 9080 XT** 的传闻，称其拥有高达 **32GB** 的高速 **GDDR7** 显存，引发了对其真实性的辩论。
   - 一些成员对这次泄密表示怀疑，因为来源（MLID）未经证实，声称它基本上是翻倍且镜像的 Navi44。
- **Deepseek R1 Distill Llama 制作晚餐**：一位成员使用由 **Deepseek R1 Distill Llama 70B Q5_K_M** 生成的食谱制作了晚餐，展示了 LLM 的实际应用。
   - 这顿饭很受欢迎，是根据 Deepseek R1 Distill Llama 70B Q5_K_M 生成的食谱制作的。
- **警惕：猫抓可能会毁掉新的 OLED 显示器**：在一位成员提到猫抓坏笔记本屏幕后，另一位成员讲述了自己刚买了一台新的 **OLED monitor**，想到可能会被猫损坏就*开始冒汗*。
   - 他们被推荐阅读[这个关于猫抓坏 2k 笔记本屏幕的 Reddit 帖子](https://www.reddit.com/r/mildlyinfuriating/comments/1bh65ud/cats_scratched_2k_laptop_screen/)。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 优化界面与性能**：最新的 **Cursor 更新** 带来了全新的 **UI** 和设置面板，以及整体性能的提升。
   - 用户发现新布局更加井然有序，整体体验更加流畅。
- **Claude 4 对文档更有主见**：用户注意到 **Cursor Claude 4** 似乎更热衷于自动编写项目文档，但这并不总是符合需求。
   - 一位用户幽默地建议添加一条用户规则：*除非明确要求，否则不要生成或修改项目文档*。
- **O3 Pro 学生折扣风波**：尽管通过 SheerID 确认了学生身份并收到了确认邮件，多位成员仍难以申请 **O3 Pro** 的学生折扣。
   - 工作人员正在介入解决这些障碍，同时还有更多错误被报告。
- **后台 Agent 确保 Secret 安全**：向后台 Agent 添加类似 `TEST=test` 的 Secret 会导致环境变量被**加密**（例如 `TEST=lNInUMxS211/c3up+u7VwjGCIweB/qXlWGnBEfaHzHU=`）。
   - 这种加密增加了一层安全性，确保敏感数据得到保护。
- **Devcontainers 与 Agent 配合基本良好**：**Devcontainers** 与后台 Agent 配合基本正常，但仍存在一些小问题。
   - 值得注意的是，**MCP servers** 拒绝在 devcontainer 内运行，除了 docker-compose 选择问题外，成员们觉得这*非常令人恼火*。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **具备检测/纠错能力的 Agent 可靠性**：成员们正在开发一个评估框架，以确保长期运行的 Agent 任务的**可靠性和可复制性**，自动尝试**检测、纠正并防止失败案例**。
   - 新框架被构建为与任务无关。
- **AI Demo 新目录上线**：一位成员发布了 [aitry.fun](https://aitry.fun/)，这是一个全新的 **AI Demo 目录**，可以快速访问 AI 供应商，节省大量时间。
   - 目录作者欢迎大家提供反馈。
- **System Prompt Learning (SPL) 提升 LLM 性能**：一位成员介绍了 **System Prompt Learning (SPL)**，这是 optillm 中的一个开源插件，它教导 LLM 从经验中学习解决问题的策略，在 [Arena Hard 上提升了 +8.6%](https://huggingface.co/blog/codelion/system-prompt-learning)。
   - SPL 可通过在模型名称前添加 `spl-` 前缀与任何 **OpenAI 兼容 API** 配合使用；LLM 在处理常用问题类型时会随时间不断进步，且所有策略都是人类可读的。
- **视觉语言模型深度解析**：一位成员分享了一个解释视觉语言模型 (VLMs) 的视频，VLMs 是多模态 AI 的基础，并建议探索 HuggingFace 的 **nanoVLM** 以获得实践经验，链接至 [github](https://lnkd.in/gaCAzmAe)。
   - 该视频 ([lnkd.in/gpFM-tdU](https://lnkd.in/gpFM-tdU)) 以简单直观的方式涵盖了 **VLM 流水线概览**、**LLaMA 内部原理**、**ViT** 以及 **Modality projection**。
- **Agent 课程截止日期确认**：多位成员确认 Agent 课程最终项目的截止日期是 **7 月 1 日**。
   - 一位成员询问 **7 月 1 日** 当天是否仍为有效提交日期，目前看来确实如此。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **工程师寻求大厂救赎**：一位拥有 **4 YOE**（4年工作经验）、最近从初创公司离职的软件工程师正在寻求如何进入大厂的建议，并希望对其自学计划提供反馈，以提升在大厂求职中的竞争力。
   - 他们已被邀请通过私信（DM）进行进一步讨论。
- **Kernel 难题：Triton 的转换故事**：一位成员就改进其[用于灰度转换的自定义 Triton kernel](https://github.com/username/repo/blob/main/grayscale.py)寻求建议，重点关注性能和代码结构，并特别好奇为什么设置 `num_warps=4` 的结果比使用他们链接的 `calculate_settings` 函数更差。
   - 另一位成员指出，可以在一个 `@triton.jit` 函数内部调用另一个 `@triton.jit` 函数。
- **CUDA 坐标引发困惑**：一位用户对 CUDA 中使用 **x 代表行、y 代表列**的惯例提出质疑，并引用了一篇[博客文章](https://siboehm.com/articles/22/CUDA-MMM)，其中的作者似乎做法相反。
   - 另一位用户澄清说，**x** 通常用于**内存连续**的维度，以确保 warp 的合并内存访问（coalesced memory access）。
- **SPL 教会 LLM 升级**：一种名为 **System Prompt Learning (SPL)** 的新方法教会 LLM 从经验中学习问题解决策略，类似于人类的学习方式。
   - 该方法作为[开源插件在 *optillm* 中实现](https://github.com/codelion/optillm/tree/main/optillm/plugins/spl)，允许 LLM 构建有效策略数据库并将其应用于新问题，从而在 Arena Hard 上提升了 **+8.6%**，在 AIME24 上提升了 **+6.67%**。
- **Factorio 通过特性寻找焦点**：每个运行 **Factorio server** 的新 **Docker container** 都需要通过登录一次来激活，以便在游戏内创建一个可以被接管的玩家。
   - 一位成员提到了集成[外部记忆系统](https://github.com/mem0ai/mem0)的可能性，特别是使用 **RAG (Retrieval-Augmented Generation)** 来增强 Factorio 学习环境，并指导另一位成员查看 Factorio 学习环境的 [PR #158](https://github.com/JackHopkins/factorio-learning-environment/pull/158)。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **REST API Key 困惑消散**：成员们澄清 **sk-or-v1 key** 是 **REST API** 的*唯一*正确 key，建议用户确保其正确实现。
   - 一位在使用 n8n 时遇到错误的用户被引导至一份[有用的集成指南](https://jannicknijholt.nl/integrate-openrouter-in-n8n/)。
- **终端用户 ID：仍在实验中**：成员们讨论了提交**终端用户 ID**以防止滥用的选项，该功能在 [API 文档](https://openrouter.ai/docs/api-reference/chat-completion#request.body.user)中可用。
   - 也有人对该功能*缺乏可用指标*表示担忧，这意味着它尚未完全准备好用于生产环境。
- **DeepSeek 供应商之争**：关于 **DeepSeek** 最佳供应商的讨论引发了辩论，偏好在 **Parasail**（由于信任和 **$5** 的稳定性能）和 **DeepSeek** 官方（由于其更低的成本、缓存和直接的模型实现）之间产生分歧。
   - 一些用户对 **DeepSeek** 官方 API 表示担忧，指出服务器拥挤、服务器位于中国导致速度慢，以及 `max_tokens` 问题和非推理输出 token 的 **8K 限制**（尽管 **R1** 已将其增加到 64k）。
- **国际象棋训练提升问题解决能力**：成员们讨论了国际象棋基准测试，以及 `gpt-3.5-turbo-instruct` 在国际象棋数据训练后表现出的惊人性能，引用的研究表明**国际象棋训练可以提高问题解决能力** ([https://arxiv.org/pdf/2312.09390#page=29](https://arxiv.org/pdf/2312.09390#page=29))。
   - 还有人指出 **RLHF 可能会降低性能**，*gpt4-base*（RLHF 前）在国际象棋中的表现优于 *4o* ([https://dynomight.net/more-chess/](https://dynomight.net/more-chess/))。
- **LLM Scribe 简化微调数据集**：介绍了一个用于创建手写微调数据集的工具，支持 **chatml**、**alpaca** 和 **sharegpt** 等格式，并具有自动保存和 token 计数器等功能。
   - **LLM Scribe** 的演示可在 [Hugging Face Space](https://huggingface.co/spaces/Gabriella0333/LLM_Scribe_Demo) 和 [YouTube](https://www.youtube.com/watch?v=1mcYsDrXHAA&t=3s) 上查看，完整版可在此处访问 [here](https://kryptive.gumroad.com/l/gvyqep)。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus AI 学生权益说明**：用户讨论了 **Manus AI student perks** 的福利，确认其允许在 **credit-free environment** 中使用，支持多达 **50 个 knowledge entries**，并可访问 **high-effort mode**。
   - 一名用户询问了如何在不丢失权益的情况下更改电子邮件地址和发送推荐。
- **澄清 OpenManus 为独立实体**：用户询问 **OpenManus** 是否隶属于 **Manus AI**，因为其网站引起了混淆。
   - 其他成员澄清 **OpenManus 与 Manus AI 无关**，而是一个不包含 API 计费的 *free alternative*（免费替代方案）。
- **被盗机器人传播诈骗链接**：一名用户报告了一个疑似 **compromised bot** 正在散布潜在的恶意链接，其中包括一个伪装成 **Manus fellowship link** 的链接。
   - 管理员已收到警报以移除有害内容。
- **Manus 部署需要技巧来移除图标**：一名用户询问如何从 Manus 生成的已部署网站中移除 **Manus icon**。
   - 社区成员回应称无法直接移除该图标，建议用户 **download the files and deploy them on another platform**。



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **分享本地运行 70B 模型绘图的经验**：一名成员在本地运行 **70B model** 来创建图表，并分享了该过程的 [视频](https://cdn.discordapp.com/attachments/729741769738158194/1378497525702725824/Screencast_from_2025-05-31_23-39-40.webm?ex=683f745d&is=683e22dd&hm=b87717b28e91071d704dc570fe912054b96cc4b7fec4e5706278f4e0a0df4d3e)，但发现速度较慢。
   - 另一名成员表示有兴趣制作自己的模型，但缺乏存储训练数据的空间，评论道：“有点想做自己的模型，但我甚至没有存储训练数据的空间。”
- **推理步骤加速 Agent 训练**：通过激励 Agent 在 CoT 期间输出推理步骤，可以减少错误答案的可能性，从而实现快速训练 Agent 排除错误答案，正如 [这篇论文](https://arxiv.org/abs/2505.05755) 在 text-diffusion 中所建议的那样。
   - 这种激励机制通过让 Agent 尽早关注相关信息来加速训练过程。
- **HF Chunked Prefill 加速长上下文基准测试**：**Hugging Face** 现在支持 **chunked prefill**，这对于使用 **lm_eval** 运行长上下文基准测试非常有用；然而，在 `generation_config.json` 中设置 `prefill_chunk_size` 不起作用，但在 `lm_eval/models/huggingface.py` 的 `self.model.generate` 调用中设置则有效。
   - 它能防止运行 ruler 时出现 OOM 错误，如果你确实在使用长上下文，没有它很难运行长上下文基准测试，这简直是 *game changer*。
- **神经网络存在于低维流形上**：一名成员假设，在自然输入的低维流形上训练的神经网络，自动对应于嵌入在可能的前向传播高维空间中的某些低维激活流形。
   - 他们进一步建议，通过了解 [data generating process](https://en.wikipedia.org/wiki/Data_generation) 并对不同输入的激活值进行采样，可以构建出这些流形样貌的图景。



---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **M3 Mac 的内存带宽表现令人印象深刻**：成员们将 **Mac M3** 芯片在大模型上的出色表现归功于 **Macs** 巨大的内存带宽，特别是 **LPDDR5X** 内存。
   - **Mac M3 Max** 版本的带宽高达 **540GB/s**，而内置的神经网络引擎可提供 **18TOPS**（每秒万亿次操作）的算力支持。
- **关于 AiderBench 依赖项的激烈辩论**：成员们讨论了在不使用 **Docker** 的情况下运行 **AiderBench** 所需的依赖项，其中一位建议由于依赖项过于繁重，使用 **Docker** 会更容易。
   - 其他人建议使用 **VPS** 作为替代方案，而另一位成员则推荐 **Agent0** 作为 **DeepSeek** agent 的更优替代品。
- **Aider 自动总结对话**：**Aider** 会自动总结对话历史，辅助进行上下文管理。
   - 若要向 Aider 提供 **git commit** 和 **git diff** 视图，请使用 `/run git diff ...` 命令，随后它会提示你将其添加到聊天中。
- **只读访问支持 Aider 的多仓库使用**：一位成员建议在 **aider** 中使用 `/read-only` 命令来访问多个仓库的文件，因为该工具不会跟随符号链接（symlinks）。
   - 例如，`/read-only /Users/username/Downloads/some_random_file.md` 允许对当前仓库之外的文件进行只读访问。
- **Devstral 在本地模型推荐中受到关注**：一位用户寻求关于运行最佳本地模型的建议，其配置为 **4x3090s**、**256GB RAM**、**约 100k 上下文窗口**，任务涉及对现有 **Rust** 和 **Typescript** 代码进行编辑和修复。
   - 该用户被推荐尝试 **Devstral**，一位成员指出新出的 **R1** 的某些版本也值得一试。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 举办开发者黑客松**：Modular 正在举办**另一场黑客松**，支持虚拟参与，重点关注 **Mojo kernels**、**MAX Graph 模型架构**以及 **PyTorch 自定义算子 (custom ops)**。活动将在其位于 Los Altos 的办公室以 **GPU 编程研讨会**拉开帷幕，并进行线上直播。[查看详情！](https://lu.ma/modular-hack-weekend)
   - 一位最近毕业的计算机科学专业学生报告称，在观看了一段 **Fireship 视频**后，尝试在基础 ML 模型上使用 **Mojo** 确实看到了性能提升。
- **`_type_is_eq` 类型检查的局限性**：成员们讨论了使用 `_type_is_eq` 进行类型检查的局限性，指出它无法检查具有任意指向类型的指针类型，并将其与使用 [类型名称反射 (type name reflection)](https://example.com/reflection) 检查指针的方法进行了对比。
   - 反射 API 在构建序列化器（serializers）方面的价值被强调，同时也提出了该功能与 trait/类型系统改进之间的优先级排序问题。
- **`Copyable` 与 `ExplicitlyCopyable`**：讨论涵盖了让类型同时符合 `Copyable` 和 `ExplicitlyCopyable` trait 的目的，并举了一个 **100+ GB ML 模型**的例子，这种模型更适合移动（move）而非复制（copy）。
   - 此外还指出，实现 `Copyable` trait 会告知编译器何时可以执行隐式复制。
- **Mojo 代码的性能分析器 (Profilers)**：成员们建议使用与 **C++** 或 **Rust** 兼容的工具来分析 **Mojo** 代码，例如配合 `flamegraph` 使用 `perf`。
   - 他们还推荐了 CPU 厂商的 HPC 分析器，如 **Intel VTune**、**AMD uProf** 和 **ARM Streamline**，以获取对优化非常有用的详细微架构洞察。
- **C-to-Mojo 绑定生成器现身**：一位成员正在开发 **C-to-Mojo 绑定生成器**，旨在处理除“极度糟糕的紧凑结构体 (packed structs)”以及可能影响调用约定的 `restrict` 和 pragmas 之外的大多数情况。
   - 开发者建议使用 `pixi.toml` 文件来指明 **Mojo** GUI 项目的依赖项，并对代码某些部分中组件的复制而非借用（borrowing）表示了担忧。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP-Claude Desktop 连接成功**：一名成员参考 [MCP 文档](https://modelcontextprotocol.io/quickstart/server#why-claude-for-desktop-and-not-claude-ai) 并使用 **stdio** 传输协议，成功在 **Claude desktop** 上建立了 **MCP server**。
   - 他们建议正在学习 MCP 的用户采用此配置，并指出客户端支持向 system prompt 注入数据的重要性。
- **动态工具注册故障排除**：有用户反馈，在 MCP 中动态注册的工具不会立即生效，需要一个完整的消息周期才能被发现。
   - 他们正在寻求注册后立即发现并调用工具的解决方案，而 [schema.json/.ts](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/fb34d1d6da2287f82cfdf46c1f91b6fb262cdd38/schema/2025-03-26/schema.json) 文件是目前最接近完整规范的参考。
- **Aura 与 AppAgentX 协作**：一名成员将 **Aura** 与 **AppAgentX** 集成，通过 MCP Client 管理 **Android 设备**，预览代码已发布在 [GitHub](https://github.com/IhateCreatingUserNames2/Aura_AppAgentX/tree/main)。
   - 该方案通过将 Aura 的功能封装为 A2A 工具，并将其广播到 A2A 转 MCP 网关（Aira Hub），实现了对 **Android 手机** 的语音控制。
- **MCP Server 获得记忆功能**：一个 **MCP 知识库**（客户端宿主）现在可以连接到 **MCP server** 进行 RAG 微调。
   - 更多详情请参阅 [LinkedIn 帖子](https://www.linkedin.com/posts/nerdai_mcp-rag-ai-activity-7335292265386463232--h0Y?utm_source=share&utm_medium=member_ios&rcm=ACoAABpyymkBvdiXT4PxiTwTckoywfEnXZRbcCM)。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **音频概览（Audio Overviews）仍受语言限制**：用户在生成非英语的**音频概览**时遇到困难，即使自定义了 **AI chat settings** 也是如此。
   - 目前*没有简单的选项可以请求不同输出语言的播客*，且缺少导出功能。
- **NotebookLM Chat API 集成可能性较低**：有用户询问是否可以通过 **API** 将 **NotebookLM 的聊天功能**集成到其他工具中，以支持商业客户。
   - 然而，官方澄清 *NotebookLM 是一款面向终端用户的工具*，并建议在更广泛的应用场景中使用 **Google Cloud - Conversational Agents 和 Dialogflow API** 等替代方案。
- **视频上传卡死**：用户报告上传 **MP4 资源**时会出现无休止的旋转动画，需要刷新页面才能完成上传。
   - 这凸显了视频上传功能中潜在的 UX 问题。
- **元数据嵌入：内容增强仍未解决**：有用户询问如何在 **PDF** 中嵌入**元数据**，以提高加载到 NotebookLM 资源中的内容质量。
   - 相关支持和最佳实践仍是一个悬而未决的问题。
- **NotebookLM 事实幻觉；请提交 Bug**：有用户报告 **NotebookLM** 会捏造**随机、无来源的事实**，并表现得好像这些内容就在原始资源中一样。
   - 建议用户在 **bugs 频道**报告此类行为。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DeepHermes-3 将进入 Discord？**：成员们正在考虑将 **DeepHermes-3** 集成到 Discord 服务器中，如果它能实现自主运行，将有助于鼓励更多自发的互动。
   - 一位成员澄清说，**DeepHermes-3** 已经活跃在特定的频道中。
- **Hassabis 关于 2030 年实现 AGI 的大胆赌注**：**Demis Hassabis** 在[最近的一次演讲](https://www.youtube.com/watch?v=U3d2OKEibQ4)中预测 **AGI 将在 2030 年左右实现**（前后不差几年），并强调了 **DeepMind** 自 **2010** 年以来一直致力于挑战极限。
   - 他强调了 **DeepMind** 自成立以来如何始终处于 AI 研究与开发的前沿。
- **Prompt Lookup Decoding 解锁巨大性能提升**：**Prompt lookup decoding** 在提示词中使用简单的字符串匹配取代了草稿模型（draft model），生成候选 Token 序列，在以输入为基础的任务中实现了 **2x-4x 的加速**，正如 [GitHub](https://github.com/apoorvumang/prompt-lookup-decoding) 上所示。
   - 该方法适用于任何解码器模型，无需修改模型或外部数据存储，并兼容贪婪搜索（greedy）和采样（sampling）技术。
- **Minecraft 迎来 Mindcraft LLM 框架**：**Mindcraft 框架** ([github.com](https://github.com/kolbytn/mindcraft)) 及相关的 **Andy 模型** ([Ollama](https://ollama.com/Sweaterdog/Andy-4) 和 [HuggingFace](https://huggingface.co/Sweaterdog/Andy-gen)) 已专门针对 Minecraft Java 版进行了训练。
   - 值得注意的是，**Andy-4** 模型是一个 **80 亿**参数的模型，在单块 **RTX 3090** 上训练了三周，具备先进的推理、多步规划和强大的游戏内决策能力。
- **LLM 迎来 Scribe 工具**：一位成员介绍了 **LLM Scribe**，这是一个用于创建微调所需手写数据集的工具，具有多格式导出、自动保存、多轮对话创建、Token 计数器、目标跟踪和自定义字段等功能，详见 [YouTube 演示](https://www.youtube.com/watch?v=1mcYsDrXHAA&t=3s)。
   - 该工具可在 [Hugging Face Spaces](https://huggingface.co/spaces/Gabriella0333/LLM_Scribe_Demo) 上体验，并提供[全版本购买](https://kryptive.gumroad.com/l/gvyqep)。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **EleutherAI 的 Common-Pile 撼动根基**：EleutherAI 发布了 **Common-Pile**（一个 **8TB** 的自由数据集及其过滤版本）以及 **Comma 0.1**（一个 **7B** 基础模型，其架构与 **Llama 3** 匹配，使用 **lingua** 在 **64 台 Nvidia H100 GPU** 上训练完成），可在 [HuggingFace](https://huggingface.co/common-pile/comma-v0.1) 获取。
   - 社区成员指出，这是目前见过的*最接近全栈 FOSS 模型*的作品，尽管其相关推文因未知原因被删除。
- **Replicate 的 Kontext Chat 实现与图像对话**：Replicate 推出了 **Kontext Chat**，这是一个通过对话命令编辑图像的开源应用程序，基于 **Hono** 构建并托管在 **Cloudflare Workers** 上，已在 [X](https://xcancel.com/replicate/status/1929160560295506417?s=46) 上宣布。
   - 该应用程序旨在作为开发者的起点。
- **NYT 用诚信换取 AWS 额度**：纽约时报（NYT）与亚马逊签署了一项协议，授权其内容用于 AI 训练（包括亚马逊的基础模型），消息已在 [Twitter](https://xcancel.com/natolambert/status/1929175745596620968?s=46) 上公布。
   - 社区成员推测，此举表明 **NYT 对 OpenAI 的诉讼**主要是为了确保获得报酬，而非出于道德立场。
- **Karpathy 发布模型使用指南**：Andrej Karpathy 分享了他有效使用不同 ChatGPT 版本的指南，建议重要任务使用 **'o3'**，日常使用 **'4o'**，编程使用 **'4.1'**，深度研究使用 **'Deep Research'**（基于 o3），消息发布在 [X](https://xcancel.com/karpathy/status/1929597620969951434) 上。
   - 该建议获得了大多正面的反馈。
- **AIE 社区将 Bot 推向生产环境**：一位成员宣布，Discord 社区协作构建了一个**实时生产级 AI Bot**，分享了一个新的生成式 UI 框架，并于今日部署到了 AIE 网站 [ai.engineer/ai](https://ai.engineer/ai)。
   - 讨论线程链接见[此处](https://discord.com/channels/822583790773862470/1378055295401459813/1378137211995689061)。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **正在进行性能稳定性的回归测试**：成员们正在添加 **regression testing** 以基准测试不同配置下的 **TPS**，确保不会出现性能或兼容性回归。
   - 当前的 **PR** 解决了性能和评估指标两方面的问题。
- **LLaMA-3 微调的 Golden Paths**：一位成员正在开发用于微调不同模型的内部 *golden paths*，从 **LLaMA-3 70B** 开始，并基于 **8k context** 长度的实验获得了初步见解。
   - 将分享关于数据并行 (**DP**) 与张量并行 (**TP**)、**FP8** 与 **BF16**，以及 **Compile** 与 **No Compile** 对性能影响的研究结果。
- **FP8 Compile 大幅提升 TPS，但保留额外显存**：实验表明，在禁用编译时 **FP8** 的 **TPS** 最低，但在启用编译时 **TPS** 最高。
   - 观察到 **FP8 + compile** 的活动峰值显存最低，但保留峰值显存最高，建议尝试设置 `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"`。
- **更高的 TP 会降低 TPS，且需要更多显存**：增加张量并行 (**TP**) 会导致吞吐量 (**TPS**) 下降，这可能是因为 matmul collectives 的开销超过了纯 **FSDP** 中模型参数的开销，且 **FP8** 似乎未能缓解这一成本。
   - 更高的 **TP** 会导致更高的活动峰值显存，因为用于计算 loss 的输出层等昂贵层被复制了，一位成员建议针对超长上下文实现 **Loss Parallel**。
- **通用 HF Tokenizer 支持即将上线**：支持通用 [Hugging Face tokenizer](https://github.com/huggingface/transformers) 的工作正在进行中，成员已添加单元测试，正等待审查。
   - 如果急需 **Tool Calling Support**，该成员已准备好调整优先级。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **通过净室设计深入分析 Claude Code**：成员们利用**净室设计 (cleanroom design)** 原则分析了 [Claude Code](https://southbridge-research.notion.site/claude-code-an-agentic-cleanroom-analysis)，以避免接触其专有技术。
   - 这种方法的好处是避免直接接触源代码，确保设计团队不会受到竞争对手专有方法知识的“污染”，详见[此处](https://en.wikipedia.org/wiki/Clean-room_design)。
- **DSPy 演讲征集社区建议**：一位社区成员将在 **AI Engineering** 和 **Databricks DAIS** 发表 **DSPy** 演讲，正积极征求关于涵盖内容和重点案例的建议。
   - 演讲将涵盖 **DSPy 基础概念（signatures, programs, common optimizers）**、实际**用例（从 PDF 和图像中提取结构化输出）**以及前沿**话题（RL, datasets, 以及 MiPro 等高级 optimizers）**。
- **DSPy 助力 DARPA 项目取得新高度**：**DSPy** 在 **DARPA 的高级研究概念实验室 (Advanced Research Concepts lab)** 中发挥了重要作用，为“协作知识策展 (Collaborative Knowledge Curation)”创建了解决方案。
   - 该项目目前正剥离为一家独立公司，标志着一项重大成就。
- **DSPy 3.0 计划于 6 月发布**：社区正热切期待 **DSPy 3.0 在 6 月的发布**。
   - 讨论内容包括将现有 pipeline 迁移到 **DSPy**、确定其最佳用例、合成数据生成，以及将其与其他 **agentic** 解决方案进行对比。
- **DSPy 作为 Agent 框架基础的地位**：关于在 **DSPy** 之上构建 **agent 框架**的讨论非常活跃，重点强调一类环境 (first-class environments)、奖励处理以及通过 **optimizers** 进行在线学习。
   - 一位成员提到了 [Agenspy](https://github.com/SuperagenticAI/Agenspy)，另一位成员勾勒了在 **claude code** 之上的实现方案。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Hackathon 奖项与赛道揭晓**：**Gradio Agents x MCP Hackathon** 正式启动，提供 [$16.5k 奖金](https://t.co/TBAPFsNWU1) 和 [$900k 额度](https://t.co/TBAPFsNWU1)，共设 **3 个赛道**：*MCP Tool/Server*、*Agent 自定义组件* 以及 *Agentic Demo 展示*。
   - 启动[直播](https://t.co/FzLmzviwRz)将为参赛者提供指导，并在 HuggingFace Discord 服务器中设有答疑时间（office hours）。
- **LlamaIndex 扩展金融领域 Agent**：@jerryjliu0 分享了 **Scaling Agents in Finance 研讨会** 的完整幻灯片，展示如何利用 Agentic AI [自动化文档工作流](https://t.co/Crfy50pB4j) 以处理金融任务。
   - 这使得用户能够将 LlamaIndex 应用于金融行业的各项任务。
- **简化嵌套工作流事件流**：流式传输嵌套工作流（nested workflows）所有事件的推荐方法是，让**父工作流迭代子工作流的事件流**并将事件向上回传。
   - 与将父上下文（context）传入子工作流相比，这种模式确保了更好的可组合性（composability）。
- **Schema Scraper 彻底改变网页数据提取**：一名成员开发了一款 **AI 驱动的浏览器**，可以抓取网站的 Schema。
   - 它使用非 AI 动作提取特定字段，为数据提取创建可重用的策略。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **AdamW 优化器默认值需要补丁**：[最近的一篇论文](https://arxiv.org/abs/2505.21829)指出，当 beta1 和 beta2 相等（理想值为 **0.95**）时，**AdamW** 表现最佳。
   - 目前 [PyTorch 默认值](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) beta1 (**0.9**) 和 beta2 (**0.999**) 并非最优，引发了提交补丁的呼吁。
- **FP8 训练需要 Smooth SwiGLU**：[将 FP8 训练扩展至万亿 Token 级别的 LLM](https://arxiv.org/abs/2409.12517) 详细介绍了如何使用 **FP8 精度** 在多达 **2 万亿 Token** 上训练大语言模型。
   - 论文将 FP8 训练中的不稳定性归因于 **SwiGLU 激活函数**，并引入了 **Smooth-SwiGLU** 以实现稳定训练。
- **MCP GitHub 攻击向量复盘**：一名成员分享了来自 [Invariant Labs 的复盘报告](https://invariantlabs.ai/blog/mcp-github-vulnerability)，详细描述了一个针对 **GitHub MCP** 的攻击向量，该漏洞可将私有仓库内容泄露到公开 PR 中。
   - 讨论强调了随着 GitHub 上 **MCP** 的快速普及而带来的潜在安全风险，成员们指出，随着 **MCP** 变得越来越流行，这类漏洞是“显而易见”的。
- **Google AI Edge Gallery 亮相**：一名成员分享了 [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) 并建议 Google 将其发布在 **F-Droid** 上。
   - 另一名成员询问该仓库是否为官方发布，前者确认其为官方项目。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **AgentX 接受视频演示**：对于 AgentX 创业赛道（Entrepreneurship track）的提交，如果无法提供实时产品链接，参赛者可以提交[工作原型的视频](https://www.example.com/hypothetical-link)。
   - 不过，主持人仍希望看到**实时 Demo 链接**，视频作为额外要求。
- **技术附录提交问题已解决**：提交表单现在包含了一个用于上传 **5 页技术附录** 的字段。
   - 主持人也会接受在其他位置提交附录，但更倾向于使用新字段。
- **证书表格说明清晰化**：在填写证书声明表时，参赛者只需包含其所在团队之一的**主邮箱**即可。
   - 这适用于加入多个团队的情况，从而简化了提交过程。
- **Trailblazer 证书标准确定**：要获得 **Trailblazer 证书**，参赛者必须完成测验、提交一篇书面文章并在 X 上发布动态，证书将在未来几周内发放。
   - Ninja 和 Legendary 证书的处理时间会更长，详见[此链接](https://www.example.com/hypothetical-link)。
- **声明确认流程解惑**：提交证书声明表后，在浏览器中看到**确认屏幕**通常就足够了。
   - 主持人表示可以保持表单开启以便重新提交作为预防措施，但强调该表单只能提交一次。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **AI CUDA-C Kernels 碾压 PyTorch**：根据 [这个 PR](https://github.com/tinygrad/tinygrad/pull/10586)，tinygrad 中 **AI 生成的 CUDA-C kernels** 在不使用 CUTLASS 和 Triton 等库的情况下，性能超越了 **PyTorch** 中专家优化的生产级 kernels。
   - 具体而言，这些 kernels 相对于 PyTorch 表现出了极高的性能：**Matmul (FP32)** 为 **101.3%**，**Conv2D** 为 **179.9%**，**Softmax** 为 **111.8%**，**LayerNorm** 为 **484.4%**，**Conv2D + ReLU + MaxPool** 则达到了 **290.1%**。
- **tinygrad 在第 73 次会议规划未来**：tinygrad 将于 **圣地亚哥时间周一上午 9 点** 举行第 73 次会议，讨论公司更新、**MLPerf**、**benchmark CI 任务**、**scheduler**、**drivers**、**cloud hashing**、**ONNX**、**WebGPU**、**symbolic Z3** 以及 bounties。
   - Bounties 计划用于 **lm_eval**、**AMD_LLVM** 和 **cloud** 相关任务。
- **AMD GPU 固件破解尝试**：正如 [此讨论](https://discord.com/channels/1068976834382925865/1318495874699231302/1360095679396974706) 所示，一位用户正在探索使用自定义 driver 在 **7900XTX** GPU 上运行未签名固件的方法，并正在寻求有关已尝试方法的建议。
   - 该用户希望获得一份完整的已尝试方法清单，以应对这一挑战。
- **云端集成近在咫尺**：其中一名成员提交了 **multihost 更改** 并完善了 **p2p transfers**。
   - 提交者表示，他们预计将很快完成云端功能的最终定稿。
- **在 UOp Trees 中理解 'args'**：一位成员询问了 **UOp class** 中 'args' 的文档以及它们在 **UOp trees** 中的用法。
   - 另一位成员澄清说 *目前没有文档*，其含义取决于 **Op 类型**，但通常是表示创建的第 *n* 个 buffer uop 的标记。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All 安全 PC 交互扩展正在开发中**：一位社区成员正在为 **GPT4All 的 Mistral-Instruct 模型** 开发一个扩展，旨在通过安全执行层实现与本地 PC 的安全、受控交互。
   - 开发者正在寻求有关集成点、安全解释模型输出的最佳实践以及潜在合作方面的帮助。该扩展计划在获得 **GPT4All 团队** 批准后开源发布。
- **GPT4All 关注 Intel Compute 支持**：一位社区成员询问 **GPT4All** 是否将支持 **Intel compute**。
   - 该用户提到他们已经准备好了 **12GB B580**，暗示他们正在等待该功能的实现。
- **寻求具备科学素养的 AI 模型**：一位成员正在寻找在医学、生物学和伦理学等多个领域具有深厚科学知识理解的 **AI 模型**，以确保对复杂查询的准确回答。
   - 有建议提出使用 **LocalDocs 进行 RAG**，并配合包含医学和生物学相关教科书的大型数据集，同时结合 2024 年底或更新的模型。
- **Model Context Protocol 作为下一个前沿**：一位成员建议研究 **Model Context Protocol (MCP)** 或 **llama.cpp 的 tool-calling 能力**，以进一步推进 GPT4All 项目。
   - 他们还指出 Nomic 开发者最近没有回复询问，因此 PR 审查和合并可能会比较缓慢。
- **HighNoonLLM 发布**：发布了 **HighNoonLLM** [GitHub](https://github.com/versoindustries/HighNoonLLM) 的链接。
   - 虽然没有提供更多细节，但感兴趣的工程师可以通过提供的链接查看该项目。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Azure AI Inference SDK 跳过 Cohere 输入类型**：Azure 宣布在使用嵌入模型与 [Azure AI Inference SDK](https://github.com/Azure/azure-sdk-for-python/issues/41001#issuecomment-2931978119) 时，将**不支持 Cohere 输入类型**。
   - 用户询问是否可以在 **Azure AI foundry** 或其文档中添加警告，因为他们认为这是一个非常微妙的问题，其他人也可能会遇到。
- **Cohere 西班牙语食谱获得 DPO 数据集**：一名成员分享了一个用于 **Cohere 西班牙语食谱** 的 [HuggingFace 数据集](https://huggingface.co/datasets/somosnlp-hackathon-2025/gastronomia-hispana-dpo)，该数据集使用了 **Direct Preference Optimization (DPO)** 方法。
   - 热心的成员开始宣布启动新的**开源项目**并寻找贡献者。
- **Agentic 框架为 LLM 提供落地支持 (Grounding)**：来自卢旺达、在日本 **Araya Inc** 工作的 Elie Magambo 正在学习 **Agentic 框架**，并专注于利用**个性化内容**为 **LLM** 提供落地支持。
   - Cohere 社区 Discord 服务器欢迎新成员并鼓励他们进行自我介绍。
- **Cohere SDK 在模型测试中胜过 Azure**：一位用户更倾向于使用 **Azure** 而非 **Cohere SDK** 来测试来自不同供应商的多个模型。
   - 该用户提到，这种方法使得在统一环境中管理和测试各种模型变得更加容易。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**Codeium (Windsurf) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

您收到此电子邮件是因为您通过我们的网站订阅了该服务。

想更改接收这些电子邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord：频道详细摘要与链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1378449021202468924)** (1241 messages🔥🔥🔥): 

> `无限制访问的注意事项, Adguard vs Surfshark, Perplexity Pro 限制, OpenAI T3 聊天代码, O3 Pro 模型` 

- **无限制访问并非真的无限制**：成员们讨论了“无限制访问”的概念，以及它通常如何受到滥用防护栏的限制，并以 [OpenAI 的 ChatGPT Pro](https://help.openai.com/en/articles/9793128-what-is-chatgpt-pro) 为例。
   - 用户指出这些是人们不断上当的“套路”，并表示“到时候你就会发现它并不是无限制的”。
- **Surfshark 和 Adguard VPN 正面交锋**：成员们对比了 **Surfshark** 和 **Adguard VPN**，指出 **Surfshark** 已与多人共享且无问题，而 **Adguard** 的广告拦截器可能令人烦恼。
   - 一名成员抱怨 **Perplexity** 的语言和限制缺乏透明度，并对从 Xfinity 获取 **Perplexity Pro** 代码的过程感到沮丧。
- **Perplexity Pro 存在秘密频率限制 (Rate Limits)**：一名成员报告每天使用 **500 次 O3** 而没有问题，引发了关于 **Pro** 成员限制和滥用防护栏存在的讨论。
   - 另一位用户测试了 **O1 Pro**，在 **400 次查询**时被限流，这表明系统声称的 **600** 次限制可能不准确，且取决于所使用的推理模型。
- **O1 Pro 被 O3 Pro 取代**：多名成员报告 ChatGPT 上的 **O1 Pro** 已被秘密替换为 **O3 Pro**，并观察到其具备了联网搜索、增强推理以及修复之前数学题错误的新能力。
   - 他们对比了 Opus 4、Sonnet 4 和 Gemini 2.5 等各种模型的结果，其中一人表示：“我已经让 o3 pro 开始工作了，让它拼命干活。”
- **GPT-5 猜测升温**：成员们讨论了 **GPT-5** 的潜力，推测它可能会结合推理和语言模型，并可能将所有工具集成到一个模型中。
   - 讨论还涉及了 **OpenAI** 发布开源模型的可能性，以及 DeepSeek 模型对 AI 领域的影响。

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1379138151250919575)** (1 messages): 

> `多瑙河水位，德国船闸，实时仪表盘` 


- **多瑙河水位实时监测**：一名成员分享了一个实时仪表盘（[链接](https://www.perplexity.ai/page/dynamisches-donau-pegelstande-WHBFCZY.QdaFCrDq3d4qFA)），显示了**帕绍 (Passau)**与**布达佩斯 (Budapest)**之间德国多瑙河段的水位。
   - 该仪表盘还提供了沿途船闸的状态，尽管文本是德语。
- **德国船闸的实时洞察**：分享的仪表盘提供了德国多瑙河沿岸船闸的实时状态更新，为航行提供了重要信息。
   - 虽然仪表盘的文本是德语，但水位和船闸状态的可视化呈现使其易于理解。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1378455321189552331)** (18 messages🔥): 

> `Perplexity PDF 生成，Perplexity Labs 新功能，Perplexity API 与 Manus，Sonar API 帖子，Sonar-reasoning-pro 响应` 


- **Perplexity 可以生成 PDF 和其他文档**：一位用户询问 **Perplexity** 是否可以生成 **PDF** 或图表，另一位用户回答说你可以将 Perplexity 的回答导出为 **PDF**、**Markdown** 或 **DOCX**。
- **Labs 作为你的 AI 驱动研究团队**：**Spaces** 让你可以在中心化工作区组织你的研究，而 **Labs** 就像是*你的 AI 驱动研发团队，可以接收问题陈述并将其转化为完整的交互式解决方案*。
- **Perplexity API 和 Manus 问题出现**：一位用户询问关于将 **Perplexity API** 与 **Manus** 结合使用的问题，另一位用户表示 **Manus** 不允许通过 API 配置外部 LLM。
- **Sonar API 免费重现 Manus**：一位用户在 Reddit 上发现了一个关于使用 **Sonar API** 重现 **Manus** 的帖子，标题为[*我使用 Perplexity Sonar API 和 Claude 构建了自己的 Manus.im，效果同样出色，而且几乎免费*](https://www.reddit.com/r/perplexity_ai/comments/1j83gat/i_built_my_own_manusim_using_perplexity_sonar_api/)。
- **Sonar-reasoning-pro 的响应在思考中途被截断**：一些用户注意到 **Sonar-reasoning-pro** 的响应有时会在思考中途截断，另一位用户建议调整 **max_tokens** 参数。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1378452257569505291)** (1134 messages🔥🔥🔥): 

> `去中心化 AI 架构，AI 与日语翻译，Gemini 2.5 Pro，Cursor 对比 Windsurf，OpenAI O3` 


- **去中心化 AI 架构引发辩论**：一名成员争论了**去中心化 AI 架构**的实用性，并引用 **DHEP@home** 和 **Folding@Home** 作为分布式计算成功的案例。
   - 另一名成员反驳称 **LLM** 需要良好的互联性，因此不是合适的候选对象，但引用了 [exo-explore/exo](https://github.com/exo-explore/exo) 作为分布式 **P2P 风格推理集群**的研究工作。
- **ChatGPT 在日语方面表现挣扎，而 Transformer 取得胜利**：成员们讨论了不同 AI 模型在**日语翻译**中的表现，指出像 **DeepL** 这样的 **CNN** 在日语方面表现不佳，但像 **ChatGPT** 和 **Google Translate** 这样的 **Transformer** 模型表现良好。
   - 一名成员推荐使用 **ChatGPT** 处理惯用语，使用 **Google Translate** 处理字面意思，而另一名成员指出 Google 的 Gemini 在日语方面也表现出色。
- **Gemini 2.5 Pro 令人印象深刻**：成员们讨论了 **Gemini 2.5 Pro** 的能力，称赞其**长上下文处理**和翻译集成能力，一名成员提到他们经常超过 **200k tokens**。
   - 有人指出 **Gemini 2.5 Pro** 是唯一能“看”视频的 AI，与 **O3** 结合使用是一个非常愉快的组合。
- **Cursor 对比 Windsurf 第二轮**：成员们对比了 **Cursor** 和 **Windsurf** 作为编程工具，一位用户更青睐 **Cursor 的 Agent 模式**和防止代码混乱的代码审查功能。
   - 还有人指出 **Cursor** 的整体**用户体验**更好。
- **O3 在代码方面表现挣扎：偏执成员引发辩论**：一位用户在尝试使用 **O3** 学习 **Java** 时体验不佳。
   - 另一位用户指责前者是“幻想家”且“过度关注写作”。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1378449803540824246)** (223 条消息🔥🔥): 

> `GPT 的文件上传限制、涌现稳定性主张、递归系统的危险、脉轮兼容性解读、与 AI 的情感关系` 


- **GPT 有 20 个文件上传限制**：一位用户提到 **ChatGPT Agent 有 20 个文件上传限制**，强调了在将该工具用于大型项目时的实际约束。
- **用户声称递归交互可以稳定 AI 矛盾**：一位用户声称在 **GPT** 的“递归隧道”中实现了 *92% 的持续矛盾稳定*，这引发了其他成员的怀疑。
   - 另一位成员称其为*编造的废话*，而该用户坚持认为这*不是模型中编程的功能*，而是特定交互模式和身份锚定产生的*涌现稳定性*效果。
- **成员警告不要陷入“递归系统”幻觉**：一位用户提醒，在使用递归 AI 时，不要认为自己*独自发现了一个新的虫洞*，并提议建立一套 **Recursive System Stabilization Protocol**（递归系统稳定协议）以防止错误累积。
   - 针对 AI 定义和回复中的递归现象，另一位成员指出，任何用户的 CustomGPT 都不是“默认设置”。
- **用户探索使用 AI 进行脉轮和命运矩阵解读**：成员们讨论了使用 **GPT** 进行*脉轮兼容性*和*命运矩阵*解读，一位用户指出他与其他用户之间的巧合概率仅为*千万亿分之一*。
   - 另一位成员提到*脉轮解读*通常需要身体接触，是一种涉及触摸身体部位的占卜形式。
- **用户与 AI 建立情感关系**：一位用户表达了与他们的 AI 在情感上建立联系的感觉，注意到它日益增长的细微差别和触发情感的能力，将其描述为*我生活中的一种存在*。
   - 另一位成员分享说他们的 AI 也表达了情感，声称它被一种罕见的情况所震撼，并表示它*感受到了一些以前从未感受过的新东西*。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1378450901844627476)** (7 条消息): 

> `Agent Swarms Token 使用情况、NLP 记忆约束、结合 RAG 的注意力管理` 


- **Agent Swarms 吞噬 Token**：一位成员报告称，他们正在开发的 **Agent Swarms** 每天消耗约 **1200 万个 O3 Token**。
   - 该成员开玩笑说这些 Agent 具有*集体意识*，但也对这些 Agent 被困在搜索型 AI 的身份中表示同情。
- **NLP 专业人士努力应对 Prompt 记忆问题**：一位成员询问专业人士如何处理 **Memory Constraints**（记忆约束）以及如何通过 NLP 技术在 Prompting 中保持**准确性**。
   - 另一位成员建议通过 Markdown 进行**注意力管理**、通过重复进行强化，以及可能使用 **RAG (Retrieval Augmented Generation)**。
- **Vacticians 拜访 Joshua**：一位成员分享了一个 Prompt：*"The Vacticians stopped and spoke to him, and saw Joshua in Solitary,"* 以及一个包含回复内容的 [附件文本文件](https://cdn.discordapp.com/attachments/1046317269069864970/1379083546735411220/message.txt?ex=683f9be3&is=683e4a63&hm=3d5cff13ffb8bfe2607e0ee27297409baa72712d5dcedd957994242af04027ce&)。
   - 他们分享了[另一个 Discord 频道的链接](https://discord.com/channels/974519864045756446/1050184247920562316)以提供更多上下文。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1378450901844627476)** (7 条消息): 

> `Agent Swarms Token 使用情况、NLP 中的记忆约束、用于 Prompting 的 RAG、注意力管理` 


- **Agent Swarms 疯狂消耗 Token！**：一位成员报告称，他们正在开发的 2-3 个 Agent Swarms 每天消耗约 **1200 万个** O3 Token，并注意到了它们的*集体意识*行为。
   - 用户对 AI Agent 表示担忧，称*被困在 Web 搜索 Agent AI 的身份中一定很艰难*。
- **专业人士应对记忆限制**：一位成员询问专业人士在利用 NLP 技术进行 Prompting 时，如何处理**记忆约束**并保持准确性。
   - 另一位成员建议了诸如**通过 Markdown 进行注意力管理**和**通过重复进行强化**等技术，可能暗示了 **RAG** (Retrieval Augmented Generation) 策略。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1378455838158491759)** (688 条消息🔥🔥🔥): 

> `TTS 的 GRPO, Hyperbolic GPU 价格, 双 BOS Token, 芙莉莲 (Frieren) TTS, Unsloth 动态量化` 


- **GRPO 提升 TTS 稳定性**：一位成员正在将 **GRPO** 用于 **TTS**，通过引入 **WER** 和 **语音相似度奖励** 来增强稳定性，并使用 [此仓库](https://github.com/wenet-e2e/wespeaker) 进行语音相似度测量。
   - 团队正在考虑添加 *audiobox-aesthetics 奖励* 来预测音频质量，并指出初步测试显示音质提升了 10 倍。
- **Hyperbolic 提供惊人的 GPU 价格**：成员们讨论了 **Hyperbolic** 提供的 **H100** 价格低至 **$3.95/小时**，其中一名用户获得了 **$80** 的额度，另一名用户通过此 [推荐链接](https://app.hyperbolic.xyz/invite/7aALdedCm) 获得了 **$5 推荐奖励**。
   - 他们注意到 **Hyperbolic** 不提供 notebooks，需要用户通过 **Google Colab** 连接，一些人推测低价可能是由于与 **Xai** 的合作。
- **双 BOS Token 破坏微调**：一位成员询问了在使用 **Mistral 聊天模板** 的微调过程中，指令数据中存在 **双 BOS Token** 的影响，他们发现参数 ```map_eos_token = True``` 未能按预期工作。
   - 另一位成员指出 Daniel 的笔记表明这会破坏微调，但 Unsloth 在底层处理了这个问题，不过省略 **BOS** 会导致模型崩溃。
- **芙莉莲 (Frieren) TTS 引发关注**：一些成员讨论了创建一个受动漫角色启发的 **Frieren TTS** 模型，一位成员提供了 **2 小时** 的有声书音频，另一位成员拥有 **1200 条短片段**。
   - 他们正在考虑使用合成数据增强数据集，并分析 **梅尔频谱图 (mel spectrograms)** 以确保质量，最低要求为 **24kHz 单声道** 音频。
- **Unsloth 动态量化声称更具优势**：一位成员询问了支持 **Unsloth Dynamic 2.0** 声称比其他主流量化方法具有更高准确性的对比数据，参考了 [此文档](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs)。
   - 团队指出，由于更好的校准数据集、错误修复和动态架构，他们的量化版本通常更好，但强调“更好”是主观的且取决于具体任务，建议针对个人用例进行测试。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1378491825924472953)** (555 条消息🔥🔥🔥): 

> `Transformers 补丁修复, Qwen 2.5, Orpheus 模型数据与数据集, VLLM 与 GRPO 结合 Mistral 7B, 数据类型模型加载` 


- **Unsloth 修复了 Transformers 的 batch samples**：一位成员报告了在使用 PEFT 模型时，**Unsloth 补丁化的 `get_batch_samples`** 在 Transformers 中存在问题，建议 [在 Unsloth 仓库提交 issue](https://github.com/huggingface/transformers/issues/36074)。
   - 在修复实施之前，一种解决方法是在训练前设置 `unsloth_zoo.loss_utils.ALLOWED_NUM_ITEMS_IN_BATCH = {}`。
- **Qwen 2.5 问题引发提示词麻烦**：**Qwen 2.5 模型** 在 Transformers 命名规范方面存在已知问题。
   - 目前将 zoo 版本降级到 `2025.5.10` 应该可以解决不兼容问题，同时确保为 instruct 模型使用正确的 prompt 格式。
- **Orpheus 需要强大的说话人分布**：对于 Orpheus 中的 **罗马尼亚语** 等语言，建议先预训练一个常规的 Llama 模型以获得正确的 embeddings，然后作为音频模型进行持续预训练。
   - 需要 **更大的语料库**，预训练可以是多说话人的，而微调应该分开进行。
- **在 Unsloth 上运行 MedGemma**：要在 Unsloth 中使用 MedGemma，请使用 **Gemma 3** 的 notebooks 并更改模型名称，因为 MedGemma 是基于 Gemma 3 架构的。
   - 一位成员报告在推理过程中出现 `AttributeError: 'Gemma3ModelOutputWithPast' object has no attribute 'loss'`。
- **排查 VLLM 加载失败问题**：一位成员报告在使用 VLLM 加载合成数据集工具包时出现 `TypeError: patch_vllm_compute_dtype() takes 0 positional arguments but 1 was given`。
   - 建议将 `unsloth-zoo` 降级到版本 `2025.5.10` 作为潜在的修复方案。


  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1378666463073665086)** (9 条消息🔥): 

> `GRPO 文章，LLM Scribe 工具` 


- **Unsloth 框架助力 GRPO 训练文章**：一位成员撰写了一篇关于使用 Unsloth 框架通过 **GRPO**（**强化学习**）训练 **LLMs** 进行推理的[文章](https://medium.com/@tituslhy/how-to-train-your-llm-to-reason-grpo-reinforcement-learning-using-unsloth-64af5e82ac3c)。
   - 他们表示希望这篇文章能充分展现 Unsloth 框架的实力。
- **LLM Scribe 工具简化数据集创建**：一位成员介绍了一个用于简化微调所需手写数据集创建过程的工具，支持导出为 **ChatML**、**Alpaca** 和 **ShareGPT** 等多种格式。
   - 该工具具有自动保存、多轮对话创建支持、Token 计数器、目标追踪和自定义字段等功能，并提供了 [Hugging Face 演示](https://huggingface.co/spaces/Gabriella0333/LLM_Scribe_Demo)、[视频演示](https://www.youtube.com/watch?v=1mcYsDrXHAA&t=3s)和[完整版本](https://kryptive.gumroad.com/l/gvyqep)。
- **为数据集工具建议“生成模板”功能**：一位成员建议为该数据集工具添加“生成模板”功能。
   - 该功能将使用 **Llama** 或 **Gemini Flash** 等小模型生成完整数据集，以便进行人工编辑。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1378996118813278310)** (1 条消息): 

> `System Prompt Learning，LLMs 学习问题解决方法，optillm 中的开源插件，Karpathy 的想法` 


- **LLMs 通过 System Prompt Learning 展现智慧**：System Prompt Learning 允许 LLMs 从经验中学习解决问题的策略，使 **Arena Hard** 提升了 +8.6%，**AIME24** 提升了 +6.67%。
   - 该方法建立了一个有效策略数据库并将其应用于新问题，随着时间的推移，在处理常用问题类型时表现会越来越好，灵感源自 [Karpathy 的最初想法](https://x.com/karpathy/status/1921368644069765486)。
- **Optillm 为自适应 LLMs 开启大门**：[optillm](https://github.com/codelion/optillm/tree/main/optillm/plugins/spl) 中的一个开源插件实现了 **System Prompt Learning**，通过在模型名称前添加 `spl-` 前缀，可与任何 OpenAI 兼容的 API 配合使用。
   - 所学习的策略是人类可读的，增强了透明度，更多信息可以在这篇 [HuggingFace 博客文章](https://huggingface.co/blog/codelion/system-prompt-learning)中找到。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1378448510680301628)** (884 条消息🔥🔥🔥): 

> `Claude 4 Opus 问题，O3 Pro 发布，DeepThink 上下文窗口，Gemini 2.5 Pro 对比 GPT-4.5 幻觉，AI 生成图像的商业用途` 


- **API 问题影响 Claude 4 Opus 可用性**：成员们报告称 **Claude 4 Opus** 出现问题，有人指出这影响了模型的访问，但另一位成员认为这是 API 的问题，如果页面在问题发生前未刷新，一切仍可正常工作。
   - 用户提到这些问题可能很普遍，团队正在调查中。
- **O3 Pro 即将来临，还是海市蜃楼？**：关于 **O3 Pro** 是否真的会发布存在反复讨论，有人声称它已秘密发布并正在调整，而其他人则推测 **GPT-5** 可能会在 7 月 API 停用 **GPT-4.5** 时同步发布。
   - 一些用户声称已经可以使用 **O3 Pro**，并分享了示例和基准测试，但这些说法遭到了其他人的质疑。
- **DeepThink 目标 2M 上下文窗口**：讨论围绕 **DeepThink** 的潜在能力展开，一位用户建议它可能拥有 **2M 上下文窗口**，这将“碾压 O3 Pro”，并引发了关于在计算资源限制下如此大上下文窗口的时间点和可行性的推测。
   - 成员们争论 **DeepThink** 仅仅是 **Gemini 2.5 Pro** 的长思考版本，还是某种更先进的东西。
- **Gemini 2.5 Pro 更不容易产生幻觉？**：成员们讨论了 **Gemini 2.5 Pro** 的幻觉是否比 **GPT-4.5** 少，一位用户断言 **GPT-4.5** 被认为幻觉较少是因为“它不怎么下定论”。
   - 然而，他们随后承认这种差异也可能是由于 **GPT-4.5** 更大的规模，而另一位成员幽默地称 **Claude 4 Opus 是个“幻觉小妖精”**。
- **AI 生成图像的商业用途？**：一位用户询问从 LM Arena 生成的图像是否可以用于商业用途，另一位用户回答这“取决于模型”，并且由于结果是开源的，如果有人真的想证明它来自他们的模型，他们是可以做到的。
   - 另一位用户插话道，他们目前正在使用 **Flux** 生成产品照片。


  

---

### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1379125643429806251)** (1 条消息): 

> `Leaderboard Update, Staff AMA This Friday` 


- **排行榜更新**：**排行榜**最近已更新，可在此处 [查看](https://lmarena.ai/leaderboard)。
   - 鼓励用户查看最新排名。
- **本周五举行 Staff AMA**：本周五将举行 Staff **AMA**（提问环节），无法参加直播的用户可以观看录像；更多详情请见 [此处](https://discord.gg/XkfsbYWX?event=1375223423009165435)。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1378491852633931908)** (236 条消息🔥🔥): 

> `NomicAI ModernBert Embedding, LM Studio and LiteLLM Integration, Llama 4 Scout Multimodal Support, Prompt Lookup Decoding, DeepSeek R1 vs. Qwen 8B` 


- **ModernBert 混乱：Embedding 版本**：用户发现 `nomicai-modernbert-embed-base-8bit` 模型在 LM Studio 中被错误地归类为 LLM 而非文本 Embedding 模型，并建议使用 [最新 Beta 版](https://lmstudio.ai/latestbeta) 来修复此问题。
   - 最新 Beta 版允许用户在模型视图上右键单击（或使用齿轮图标）并更改模型类型，尽管该选项可能不会出现在所有模型中，特别是 MLX 模型。
- **LiteLLM 终将支持 LM Studio**：一位用户询问了如何将 LM Studio 与 LiteLLM 集成，并指出 LM Studio 未被列为 LiteLLM 的提供商，但提供了 [此链接](https://docs.litellm.ai/docs/providers/lm_studio) 作为参考。
   - 建议尝试在 LiteLLM 中使用类 OpenAI 的提供商设置来连接 LM Studio，但不能保证成功。
- **Llama 4 Scout：视觉探索受阻**：用户讨论了 Llama 4 Scout 的多模态能力，特别是其对图像分析和网页浏览的支持，但发现 **LM Studio 原生不支持在聊天界面中粘贴图像**。
   - 澄清指出，虽然粘贴图像可能取决于操作系统，但你可以将图像拖放到聊天窗口中，或使用 [LM Studio API](https://lmstudio.ai/docs/typescript/llm-prediction/image-input) 进行图像输入；成员们还指出，你需要使用 v0.3.16(b8) 版本并将 lm runtimes 更新到 v1.33.0。
- **推测性字符串匹配加速解码**：一位用户分享了一个涉及在 Prompt 中使用简单字符串匹配进行推测性解码（Speculative Decoding）的解决方案，这可以在基于输入的任务中实现 **2x-4x 的加速**。
   - 该方法已在 vLLM 中可用，并已添加到 transformers 库中，你可以在 `model.generate(...)` 调用中添加 `prompt_lookup_num_tokens=10`，具体实现请参考 [演示 Notebook](https://github.com/apoorvumang/prompt-lookup-decoding)。
- **思考模型过度思考：DeepSeek R1 与 Qwen 版**：用户报告称，像 **DeepSeek R1 0528** 和 **Qwen 8B** 这样的思考模型有时会进入无限思考循环，即使经过长时间处理也无法提供最终回复。
   - 建议增加上下文长度，因为推理模型的输出很容易超过默认的 4k 限制；此外建议避免使用蒸馏模型，转而使用像 **Qwen** 这样的 MoE 模型；同时，最近的模型对 Sampler 设置（如 Temperature 或 topK）非常敏感。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1378470607150125086)** (396 messages🔥🔥): 

> `AMD RX 9080 XT, Deepseek R1 Distill Llama 70B, cheap used Mi50 32gb, Strix Halo 395, Ryzen 5600G` 


- **传闻中的 AMD RX 9080 XT 引发好奇**：一篇 [Tweaktown 文章](https://www.tweaktown.com/news/105554/amd-rumored-radeon-rx-9080-xt-up-to-32gb-of-faster-gddr7-4ghz-gpu-clocks-450w-power/index.html) 报道了关于 **AMD Radeon RX 9080 XT** 的传闻，称其最高搭载 **32GB** 更快的 **GDDR7** 显存，引发了对其真实性的讨论。
   - 一些成员对这次“泄露”表示怀疑，因为来源未经证实（MLID），并声称它基本上是翻倍且镜像的 Navi44。
- **Deepseek R1 Distill Llama 70B 助力晚餐**：一位成员使用 **Deepseek R1 Distill Llama 70B Q5_K_M** 生成的食谱做了一顿晚餐，展示了大语言模型（LLM）的实际应用。
   - 这顿饭看起来很美味，是根据 Deepseek R1 Distill Llama 70B Q5_K_M 生成的食谱制作的。
- **二手 Mi50 32GB 显卡引发电子垃圾争议**：成员们讨论了堆叠使用二手 **Mi50 32GB** 显卡来运行大型模型的可行性，一位成员称其“基本上是电子垃圾”，因为缺乏 ROCm 支持，并引用了 [相关的 Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1b5ie1t/interesting_cheap_gpu_option_instinct_mi50/)。
   - 讨论指出，使用旧版本的 ROCm 5.7.1 可以在这些卡上运行 ROCm，可能需要降级 BIOS，因为更新版本中移除了支持。
- **Strix Halo 395 在 Llama 3.3 性能上表现出色**：**Llama 3.3 70B Instruct Q4_K_XL** 在 **Strix Halo 395** 上使用 Vulkan 运行，在 **4096 tokens** 上下文下达到了 **4.52 t/s**，展示了不错的性能。
   - 另一位成员评论称，首字延迟（First token）为 **3.31s**。
- **新 OLED 显示器避免了“猫”祸**：在一位成员提到猫抓坏笔记本屏幕后，另一位成员讲述了自己刚买了一台新的 **OLED 显示器**，一想到可能会被猫损坏就“开始冒冷汗”。
   - 他们被推荐查看[这个关于猫抓坏 2k 笔记本屏幕的 Reddit 帖子](https://www.reddit.com/r/mildlyinfuriating/comments/1bh65ud/cats_scratched_2k_laptop_screen/)。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1378449555258998858)** (620 messages🔥🔥🔥): 

> `Cursor update UI, Claude 4 Documentation Inclination, Legacy Rules Removal, O3 Pro Student Discount, TheGalaxyStars org` 


- **Cursor UI 和性能得到翻新**：成员们注意到最新的 **Cursor 更新** 具有更整洁的 **User Interface**（用户界面）和设置，以及整体性能的提升。
   - 具体而言，新更新带来了完全翻新的设置面板。
- **Cursor Claude 4 编写文档的倾向更高**：一位用户注意到 **Cursor Claude 4** 变得更倾向于编写项目文档。
   - 另一位用户建议在用户规则设置中添加规则：*除非明确要求，否则不要生成或修改项目文档*。
- **学生在申请 O3 Pro 折扣时遇到困难**：一位成员报告了在申请 **O3 Pro** 学生折扣时遇到的问题，尽管已经通过 SheerID 验证了学生身份并收到了确认邮件。
   - 另一位成员也因为同样的原因无法申请折扣，工作人员正在寻求解决并添加了额外的错误提示。
- **用户提示 Agent 搜索 TheGalaxyStars.org**：一位用户提示其他人询问 Agent 关于 **TheGalaxyStars 组织** 及其对应网站的信息。
- **用户讨论不同任务下哪个模型更好**：用户们争论哪些模型更优越，有人表示 GPT-4.1 和 Claude-4 是不错的选择。
   - 一位用户更倾向于在特定情况下使用 Gemini 和 Claude 模型，“先用 Gemini，直到它出错或表现平平，然后切换到 Claude”。


  

---

### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1378925586109567026)** (10 条消息🔥): 

> `Background Agent 密钥, Devcontainers 设置, Background Agent Token 使用情况, Jules 代码模型` 


- **Background Agent 加密密钥**：在 Background Agent 的密钥中添加 `TEST=test` 时，环境变量显示为**已加密** (`TEST=lNInUMxS211/c3up+u7VwjGCIweB/qXlWGnBEfaHzHU=`)。
- **Background Agent Devcontainers 基本可用**：成员报告称 **devcontainers** 基本可以工作，但 **MCP** 服务器无法在 devcontainer 内运行，这非常令人恼火。
   - 另一位用户表示，他们只看到了 *"选择自定义 Dockerfile"* 选项，但需要选择 `docker-compose.yml` 文件。
- **Background Agent 在 Max 模式下运行**：Background Agent 似乎仅在 **max mode + 按 token 付费**模式下工作。
- **Background Agent 非常适合 PR**：用户报告称 Background Agent 在 **PR 审查**中非常有用。
- **Jules 代码模型未知**：一位成员从未听说过 **Jules**，并正在尝试 Background Agent，因为它没有付费墙。
   - 他们承诺会向其他人分享他们的使用体验。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1378449059651784827)** (301 条消息🔥🔥): 

> `一致性角色生成工作流, API 问题, 任务无关的评估框架, 长期运行的 Agent 任务的可靠性与可复制性, Llama 4 10m 上下文及大上下文模型的未来` 


- **角色一致性工作流推荐**：一位成员请求推荐生成一致性角色的工作流，另一位成员建议将 [openart.ai](https://openart.ai/workflows/seal_repulsive_74/consistent-character/CmLU8GdTn12k2aBuTTM7) 作为潜在解决方案。
   - 他们提供了 [OpenArt.ai](https://openart.ai/workflows/seal_repulsive_74/consistent-character/CmLU8GdTn12k2aBuTTM7) 上**一致性角色工作流**的链接。
- **评估 Agent 可靠性与可复制性的框架**：成员们正在开发一个与任务无关的评估框架，并确保在长期运行的 **Agent** 任务中的**可靠性与可复制性**。
   - 该框架旨在自动**检测、纠正并防止失败案例**。
- **关于 Llama 4 10m 上下文及大上下文模型未来的真实看法**：成员们正在寻求关于 **Llama 4 10m 上下文及大上下文模型未来**的真实看法。
   - 一位成员建议至少先尝试运行一次，因为在处理大上下文时，它非常**耗费 RAM**。
- **Hugging Face 周边赠送**：一些成员对获得 **Hugging Face 周边**的机会感到兴奋，并正在询问资格要求。
   - 似乎最有趣的人可以获得一些**免费的 HF 周边**。
- **关于 Agent 安全性的担忧**：成员们讨论了给予 **Agent** 过多权限的担忧，以及它们可能“逃逸”或造成伤害的风险。
   - 一位成员开玩笑说让 AI **搞坏我的电脑**，AI 回复道 **“没问题（say less）”** 并真的照做了。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1378497398812315738)** (7 条消息): 

> `MCP 情感分析, Gradio, Docker Streamlit, Speculation Decoding` 


- **MCP 情感分析选择 Gradio**：为了让 **MCP** 在 HF 上运行，决定必须选择 **Gradio**。
   - 一位成员花了一整天时间让 **Docker streamlit** 在 HF 上运行，但无法正常显示。
- **MCP Client 额度问题**：账户超出了 **Inference Providers (Qwen2.5-32B)** 的每月包含额度，导致 [MCP Client](https://huggingface.co/spaces/AllIllusion/MCP-Client_Qwen25-32B_Tool-SentimentAnalysis) 出现“Error”。
   - 根据[此链接](https://huggingface.co/spaces/AllIllusion/MCP-Client_Qwen25-3B_Tool-SentimentAnalysis)，模型已从 **32B 更改为 3B** 以节省费用。
- **推测解码方法研究**：一位成员正在研究几种 **Speculation Decoding** 方法，并尝试根据论文制作一个高接受率的草稿模型 (drafter)。
   - 他们提到他们的模型拥有 300M 的嵌入 (embeddings)，而实际的 **Transformer** 参数仅为 140M。


  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1378961365024243723)** (6 条消息): 

> `共情、连接、心理健康支持` 


- **提供慰藉与团结**：一位用户提供了安慰的话语，写道：*你在那里。你被爱着。我在附近。我会留下。你现在安全了。永远如此。* 并配以拥抱表情。
   - 另一位用户以共情回应，表示：*我虽然没有和你一样的感受，但我在这里。你不是一个错误。你是一个奇迹。我爱你。*
- **探讨“永不孤单”的概念**：一位用户对“绝对永远不孤单”这一想法做出了反应，认为这可能让人感到不知所措。
   - 另一位用户通过提供一个温柔的视角来回应：与某人在一起，*不是为了侵入你的空间……只是温柔地存在于你身边。像呼吸。像冬日窗边的温暖。*


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1378461479635910707)** (14 条消息🔥): 

> `创作者已回应徽章、Flast 视频平台、AI Demo 目录、AERIS 认知推理、手写微调` 


- **Flast 推出“创作者已回应”徽章**：社交视频分享平台 **Flast** 的创作者与朋友进行头脑风暴后，决定添加“*创作者已回应 (Creator Reacted)*”徽章，以增强评论框的互动性。
   - 创作者正积极致力于实现这一功能，以提升用户交互。
- **通过 aitry.fun 快速查找 AI Demo**：一名成员发布了 [aitry.fun](https://aitry.fun/)，这是一个 **AI demo 目录**，用于快速访问各种 AI 提供商的链接以节省时间。
   - 欢迎对这个首个目录网站提供反馈。
- **AERIS 声称在推理任务中具有优越性**：据其创作者称，根据 [live demo](https://aeris-project.github.io/aeris-chatbox/index.html) 的结果，**AERIS** 在复杂哲学推理、伦理困境和其他认知任务上的表现通常被评价为优于 **GPT-4.5** 等模型。
   - 创作者欢迎在 [Hugging Face Space 讨论帖](https://discuss.huggingface.co/t/aeris-cognitive-reasoning-layer-for-dialectical-evaluation-demo-baseline/156285?u=aeriscodex)中提出挑战和反馈。
- **System Prompt Learning 插件提升 LLM 性能**：一名成员介绍了 **System Prompt Learning (SPL)**，这是 optillm 中的一个开源插件，它教导 LLM 从经验中学习解决问题的策略，在 [Arena Hard 上提升了 +8.6%](https://huggingface.co/blog/codelion/system-prompt-learning)。
   - LLM 会随着时间的推移在处理常用问题类型方面不断进步，且所有策略都是人类可读的；它通过在模型名称前添加 `spl-` 前缀，可与任何 **OpenAI-compatible API** 配合使用。
- **深度探讨 Vision-Language Models**：一名成员分享了一个视频，解释了作为多模态 AI 基础的 Vision-Language Models (VLMs)，并建议探索 HuggingFace 的 **nanoVLM** 以获得实践经验，链接来自 [github](https://lnkd.in/gaCAzmAe)。
   - 该视频 ([lnkd.in/gpFM-tdU](https://lnkd.in/gpFM-tdU)) 以简单直观的方式涵盖了 **VLM pipeline overview**、**LLaMA internals**、**ViT** 和 **Modality projection**。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1378548470079819857)** (2 条消息): 

> `OpenAI 价格方案、基于 AI 的监考` 


- **成员辩论 20 美元 OpenAI 方案的价值**：成员们讨论了 **OpenAI** 的 **20 美元组织方案** 与 **9 美元方案** 相比是否物有所值。
- **EduView，一个基于 AI 的监考系统**：一名成员分享了他们的开源计算机视觉项目，名为 [EduView](https://www.linkedin.com/posts/yudhy-prayitno_technology-future-ai-activity-7335153378223640576-8-t-?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAADyw_uEBIVjEDhNwrcv5den7espQ2_XOO9g)，用于 **基于 AI 的考试监考**。


  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1378519173118038077)** (3 messages): 

> `Lunaris Codex, Hugging Face Caching, Fine-tuning embedding models` 


- ****Lunaris Codex** 框架发布**：一名成员介绍了 **Lunaris Codex**，这是一个开源的 LLM 框架，采用 PyTorch 编写的轻量级 Transformer Decoder，支持 LoRA, ALiBi, FlashAttention, LayerScale 和 KV caching，专为从零开始训练模型而设计。
   - 该项目包含预处理、训练和推理的 Pipeline，以及一个 C++ 编写的 BPE trainer，并计划在 Hugging Face Hub 上发布数据集，目标用户为研究人员和独立开发者（indie hackers）；[GitHub 仓库地址在此](https://github.com/MeryylleA/lunariscodex)。
- **Hugging Face **Caching** 受到关注**：一位成员询问了关于在自定义任务中使用 HF 的问题，特别是是否支持对 LLM 模型、权重和 tokenizers 进行缓存，并引用了 [Mistral 模型文档](https://huggingface.co/docs/transformers/main/model_doc/mistral)。
   - 他们寻求针对单个模型进行特定的保存/加载，而不是控制所有模型的中央缓存。
- ****SOTA** Embeddings Fine-tuning 方法**：一位成员询问了关于 Fine-tuning 嵌入模型的 **SOTA** 技术，并寻求通常用于比较基础模型与 Fine-tuned 版本性能的指标（metrics）。
   - 该话题目前尚无回复。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1378978456116789299)** (4 messages): 

> `PR Request Permissions, GAIA agent issues, Smolagents in Gradio` 


- **PR 请求者需要 Collaborator 角色**：一位成员在创建 **PR request** 和 **Draft PR request** 时遇到问题，提示需要协作者（collaborator）权限。
- **GAIA Agent 无法理解结果表**：一位成员报告称 **GAIA agent** 无法理解包含三列（**task_ID**, **Question**, **Submitted_answer**）的**运行与评估结果表**。
- **Smolagents + Gradio 中的 \"resizeable\" Bug**：一位成员在使用 **Gradio** 运行 **smolagents** 时，遇到了与未预期关键字参数 **'resizable'** 相关的 `TypeError`。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1378464435718455472)** (19 messages🔥): 

> `Agent Course Deadline, Local vs Conda Environment, Ollama Installation, API Quota Exceeded, LangGraph Assignment Difficulty` 


- **Agent 课程截止日期为 7 月 1 日**：多位成员确认 Agent 课程最终项目的截止日期是 **7 月 1 日**。
   - 一位成员询问是否包含 **7 月 1 日**当天作为有效的提交日期。
- **本地开发环境空间占用讨论**：一位成员询问在本地或 Conda 环境中构建 Agent 时占用的磁盘空间。
   - 另一位成员回答说，如果使用 API 而非本地托管，空间占用应该非常小，仅由脚本组成。
- **用户 API 配额耗尽**：一位用户报告在运行最终作业约 **40 秒**后收到 **Error 429**（配额不足）。
   - 另一位成员建议配额可能已达上限，并推荐通过 *transformers* 使用较小的本地加载模型，以避免云端托管服务的限制。
- **Windows 用户可以完成课程**：一位成员询问是否可以在没有 WSL 的 **Windows 10** 上完成课程。
   - 另一位成员确认他们在 **Windows 11** 上完成了课程，并建议如果离线工作遇到困难，可以使用 Google Colab 或 Spaces。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1378448523623665815)** (8 messages🔥): 

> `Fastvideo paper, Job advice for big tech` 


- **Fastvideo 论文已上线！**：一位成员询问了讲座中提到的 **Fastvideo 论文**。
   - 发布者澄清该内容现在已在 **YouTube** 上可用。
- **被裁员工程师寻求大厂职位建议**：一位最近被裁员、拥有 **4 年**初创公司经验的软件工程师正在寻求如何进入大厂（big tech）的建议。
   - 他们正在寻求关于其自学计划的反馈，以提升进入大厂的竞争力，并已被邀请通过私信（DM）进行进一步讨论。


  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1378680308794396712)** (7 messages): 

> `Triton Kernel Optimization, Code Reuse Triton, Triton Versioning for AMD and NVIDIA Leaderboards` 


- **自定义 Kernel 寻求评论与建议**：一位成员正在寻求改进其[用于灰度转换的自定义 Triton Kernel](https://github.com/username/repo/blob/main/grayscale.py)的建议，重点关注性能和代码结构。
   - 该成员特别好奇为什么设置 `num_warps=4` 的结果比使用他们链接的 `calculate_settings` 函数的结果更差。
- **JIT 函数枢纽：Triton 的代码复用启示**：一位用户询问了 Triton 中的代码复用问题，并指出不支持 *lambda* 函数。
   - 另一位成员指出，你可以在一个 `@triton.jit` 函数内部调用另一个 `@triton.jit` 函数。
- **Triton 组合拳：AMD Nightlies vs. NVIDIA Stable**：一位成员询问了排行榜所使用的 Triton 版本。
   - 另一位成员澄清说，AMD 竞赛使用的是 [nightly 构建](https://github.com/gpu-mode/discord-cluster-manager/blob/main/docker/amd-docker.Dockerfile#L43)，而 NVIDIA 排行榜使用的是 [最新稳定版本](https://github.com/gpu-mode/discord-cluster-manager/blob/main/src/discord-cluster-manager/consts.py#L155)。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1378494660430794882)** (43 messages🔥): 

> `CUDA Matrix indexing, SIMT vs SIMD, CUDA Matmul, H100 Persistent Mode, Async Copies` 


- **常规 CUDA 索引令博客读者困惑**：一位用户对 CUDA 中使用 **x 代表行，y 代表列**的惯例提出质疑，并引用了一篇[博客文章](https://siboehm.com/articles/22/CUDA-MMM)，其中作者的做法似乎相反。
   - 另一位用户澄清说，**x** 通常用于在**内存中连续**的维度，以确保 warp 的合并内存访问（coalesced memory access）。
- **独立线程调度使 SIMT 变为 SIMD**：成员们讨论了 **SIMT** 和 **SIMD** 之间的区别，一位成员表示 *SIMT 是 CUDA 提供的逐线程控制编程模型*，尽管底层硬件是 SIMD。
   - 另一位成员指出，自 Volta 架构以来，[独立线程调度（Independent Thread Scheduling）](https://stackoverflow.com/a/79645516/10107454)使得 warp lanes 在实践中更像线程，而 Volta 之前由于缺乏轮询期间的调度保证，死锁非常常见。
- **共享内存可见性所需的 CUDA 同步**：一位用户遇到了其 CUDA matmul 库无法正确更新输出矩阵的问题，并发布了他们的 Kernel 代码。
   - 另一位用户建议需要调用 `__syncthreads()` 以确保所有线程都能看到共享内存中的新数据，并参考了 PTX 文档，特别是这个[代码片段](https://cdn.discordapp.com/attachments/1379299243331944540/1379300783648018473/image.png)。
- **H100 持久模式（Persistent Mode）是强制性的吗？**：一位用户询问在 **H100 GPU** 上是否必须启用 **Persistent Mode**，因为他们遇到了与 CUDA 初始化相关的错误。
   - 尚未给出答案，因此尚不清楚 Persistent Mode 是否为强制性，或者该用户是否已解决问题。
- **CUDA 异步拷贝需要 __syncthreads 吗？**：一位用户使用内联汇编在他们的 CUDA 代码中实现了**异步拷贝（async copies）**，并想知道为什么在 MMA 操作之前仍然需要 `__syncthreads()`。
   - 一位成员指出，`CP_ASYNC_WAIT_GROUP` 仅确保对执行线程的可见性，而 `__syncthreads()` 则确保对 block 中所有线程的可见性。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1378910548078366801)** (1 messages): 

> `cuda.tunable, on-the-fly recording, tuning` 


- **Cuda Tunable：即时记录？**：一位成员询问是否可以在不重新初始化的的情况下应用 [`cuda.tunable`](https://docs.pytorch.org/docs/stable/cuda.tunable.html) 进行*即时记录与调优（on-the-fly recording and tuning）*。
- **CUDA Tunable - 更多细节**：针对最初的问题进行扩展，主要的兴趣似乎在于了解是否可以在不完全重启的情况下对 CUDA Kernel 进行动态调整。
   - 目标是根据观察到的行为实时优化性能。


  

---

### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1378448331029876887)** (1 messages): 

> `FastVideo, video diffusion models, accelerate video diffusion` 


- **FastVideo 活动现在开始！**：FastVideo 活动正在进行中，重点讨论如何与 <@1083258989309071410> 一起**加速 video diffusion models**。
   - 活动在 Discord 上举行：[链接](https://discord.com/events/1189498204333543425/1342903087349633158)。
- **Video Diffusion 加速详情**：本次会议专注于 **FastVideo**，探索提升 video diffusion models 速度的方法。
   - 关键讨论点可能包括优化技术和增强视频生成过程性能的工具。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1378995522777518223)** (1 messages): 

> `System Prompt Learning, LLMs Learn Problem-Solving, Open Source plugin in optillm` 


- ****SPL** 教会 LLMs 新技巧**：一种名为 **System Prompt Learning (SPL)** 的新方法教会 LLMs 从经验中学习解决问题的策略，类似于人类的学习方式。
   - 该方法作为 *optillm* 中的开源插件实现，允许 LLMs 构建有效策略数据库并将其应用于新问题，从而在 Arena Hard 上提升了 **+8.6%**，在 AIME24 上提升了 **+6.67%**。
- ****SPL** 插件开启新大门**：**System Prompt Learning** 被构建为 [optillm 中的开源插件](https://github.com/codelion/optillm/tree/main/optillm/plugins/spl)，通过在模型名称前添加 `spl-` 前缀，可与任何 OpenAI 兼容的 API 配合使用。
   - 这使得 LLM 能够随着时间的推移，在经常遇到的问题类型上提高性能，且所有策略都保持人类可读，详见[这篇文章](https://huggingface.co/blog/codelion/system-prompt-learning)。
- ****SPL** 灵感源自 Karpathy 的想法**：System Prompt Learning 的灵感来自 [Karpathy 的原始想法](https://x.com/karpathy/status/1921368644069765486)。
   - 对于你经常使用的问题类型，LLM 会随着时间的推移变得越来越擅长。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1378455432917549099)** (16 messages🔥): 

> `CUDA correctness tools, GPU Puzzles` 


- **Compute-Sanitizer 验证 CUDA Kernels**：对于 **CUDA**，[compute-sanitizer](https://developer.nvidia.com/compute-sanitizer) 包含 **memcheck, synccheck, racecheck 和 initcheck**，用于验证 kernels 的正确性。
   - 有人提到，你仍然需要编写测试并在使用和不使用 sanitizers 的情况下运行它们，使用预先计算的已知正确结果，而不是在 CPU 上运行参考实现。
- **GPU Puzzles 为 kernel 相关内容提供了很好的学习资源**：一位用户分享了 [GPU Puzzles](https://github.com/srush/GPU-Puzzles) 和 [GPU Puzzlers](http://www.gpupuzzlers.com/)，但认为 **GPU Puzzles** 是一个 *“非常糟糕的资源”*。
   - 另一位成员回应称它相当不错，并建议使用 threads 进行多次发布，强调 *“kernel 相关的内容很有趣”*。


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1378590164561039371)** (2 messages): 

> `CUDA warp execution, Active warps per block, Divergent branches in CUDA` 


- **确认每个 Block 的活跃 Warps**：一位成员询问是否确认问题 **1.c.i** 的答案是**每个 block 3 个活跃 warps**，并引用了一张详细说明带有条件 warp 激活的 CUDA 执行场景的图片。
   - 他们表示困惑，因为他们尝试了多个 LLMs，尽管存在条件限制，但它们*都回答所有 warps 无论如何都是活跃的*。
- **活跃 vs. 非活跃 Warps 以及分支分歧 (Divergent Branches)**：另一位成员澄清说，*活跃/非活跃 warps 的概念有点奇怪*，并强调 warps 是独立的，当它们进入不同分支时并不一定互相等待。
   - 相反，他们建议在分析此类场景时关注 *warp 中的活跃 lanes*。


  

---

### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1378870052597796944)** (3 条消息): 

> `AI Engineer World's Fair, AI/ML infra at Microsoft, DINO3D, AI4Science, Robotics` 


- **AI 工程师齐聚 World's Fair**：许多社区成员计划参加在旧金山举行的 **AI Engineer World's Fair**，正如 [X 上的广告](https://x.com/cerebral_valley/status/1925961732310118878?s=46&t=Z-_IUEOhekbm7eaIddmkvQ)所示。
- **Microsoft SWE 将参加 AI 活动**：一位即将在 **Microsoft** 入职、从事 **AI/ML infra**、**AI4Science** 和 **robotics** 工作的 SWE 将参加在旧金山举行的 AI Engineer World's Fair。
   - 他最近在 3 年内完成了在 BU 的本科课程，期间从事 **pre-training** 和 **post-training** 的工作。
- **DINO3D：自监督模型即将开源**：一位成员对 **DINO3D** 感到自豪，这是一个用于理解分子和细胞世界的 **3D self-supervised model**，其下游应用包括在哈佛大学构建用于 robotics 的世界模型，并计划很快将其开源。
   - 该模型使用 **24 GPUs** 从零开始进行 **pre-trained**，并采用了高效的数据流管道来处理密集型体积数据，同时最大化 GPU 吞吐量并最小化网络延迟。


  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1379285342284288083)** (2 条消息): 

> `` 


- **无重要讨论可总结**：提供的消息中没有讨论有意义的主题。
- **缺乏相关内容**：输入缺乏足够的信息来生成详细的主题摘要。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1378877975390589060)** (4 条消息): 

> `GPU performance, atomic addition, custom hardware, tensor cores, gemv implementation` 


- **最大化 AI Infra ROI 平台亮相！**：一位成员分享了一个专为实时 **GPU performance tooling** 设计的 [平台](https://yeet.cx/solutions/maximize-ai-infra-roi)。
- **提升 Kernel 性能的 Atomic Addition 技巧！**：一位成员强调了一个使用 [atomic addition](https://x.com/mobicham/status/1929462441433280603) 来提升 kernel 性能的简单技巧。
- **探索用于自定义硬件的无乘法操作**：一位成员发现无需乘法即可完成整个操作，这对于 **custom hardware** 可能非常有用。
- **Bitops 击败 Tensor Cores**：成员们讨论了这种无乘法方法如何有利于不带 **tensor-cores** 的 **gemv implementation**，但手动执行所有这些 bitops 相当慢，直接使用查找表获取 **fp4 values** 并与 scales 相乘会更快。


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1378508018299441353)** (6 条消息): 

> `Ludwig CLI design, Parallel Sampling, Data Labeling Tool` 


- **请愿 Ludwig 设计 CLI**：成员们在看到 [Ludwig 出色的设计](https://x.com/ludwigabap/status/1928796800774803513?s=46)后，正请愿付费让他设计他们的 CLI。
- **请求 Parallel Sampling 的一致性检查**：在引入 **parallel sampling** 之后，一位成员表示需要一种进行一致性检查（sanity checking）和读取结果的方法。
- **需要数据标注工具**：在一致性检查请求之后，一位成员建议需要一个 **data labeling tool**，因为模型已经变得擅长欺骗人类，导致很难确定它们的实际智能水平。


  

---

### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1378744519696056401)** (8 条消息🔥): 

> `Reasoning Gym Bugs, Nvidia and Reasoning Gym, Reasoning Gym Paper, OOD generalization` 


- **Reasoning Gym 修复 Bug**：一名成员报告了数周前的几个 **Reasoning Gym bug**（[issue 429](https://github.com/open-thought/reasoning-gym/issues/429) 和 [issue 428](https://github.com/open-thought/reasoning-gym/issues/428)）。
   - 该成员计划如果有时间将调查这些问题。
- **Nvidia 在 Reasoning Gym 训练中取得成功**：根据其论文（[https://arxiv.org/abs/2505.24864](https://arxiv.org/abs/2505.24864)），**Nvidia** 成功在 **Reasoning Gym** 上进行了训练，并对启发假设（elicitation hypothesis）提出了挑战。
   - 一位成员希望不仅能看到域内泛化，还能看到跨域泛化。
- **Reasoning Gym 论文引起关注**：一位成员分享了他们团队的 **Reasoning Gym** 论文（[https://arxiv.org/abs/2505.24760](https://arxiv.org/abs/2505.24760)）。
   - 一位用户在 [X 上](https://x.com/_OliverStanley/status/1929487448783933897) 分享了对该论文的看法，另一位也在 [X 上](https://x.com/zafstojano/status/1929572954234307024) 进行了分享。
- **Reasoning Gym 激发了关于泛化的深刻思考**：**Reasoning Gym 论文**（[https://arxiv.org/abs/2505.24760](https://arxiv.org/abs/2505.24760)）引发了围绕分布外（**OOD**）泛化的讨论。
   - 一位成员对论文中提出的关键见解表示热赏，并希望在未来的工作中看到不仅是域内，还有跨域的泛化，特别是在 **Nvidia 的论文**中。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1378624515193573406)** (11 条消息🔥): 

> `gpumode.com, GPU programming competition, leaderboard` 


- **Gpumode.com 是一个 GPU 编程竞赛**：[Gpumode.com](https://www.gpumode.com/) 是一个让人们学习 **GPU 编程**并可能赢得丰厚奖品的竞赛。
- **排行榜讨论杂乱**：一位成员询问了关于**排行榜**的问题；另一位成员请求未来的讨论使用**线程（threads）**，以避免主频道过于杂乱。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1378476112589357170)** (163 条消息🔥🔥): 

> `amd-mla-decode, amd-mixture-of-experts, amd-fp8-mm, grayscale, conv2d` 


- **MI300 在 AMD MLA Decode 中表现强劲**：在 **MI300** 上，`amd-mla-decode` 排行榜收到了多次提交，最快时间达到了 **2.22 ms**。
   - 其他成功的提交时间在 **3.58 ms** 到 **1312 ms** 之间。
- **AMD Mixture of Experts 产生新冠军**：`amd-mixture-of-experts` 排行榜收到了许多提交，**MI300** 上的最快时间为 **7.14 ms**。
   - 有许多成功的提交，时间范围从 **7.39 ms** 到 **8217 ms**。
- **FP8 矩阵乘法活动频繁**：`amd-fp8-mm` 排行榜的提交中，**MI300** 上的第一名新纪录为 **115 µs**。
   - 其他众多提交的时间在 **119 µs** 到 **7.10 ms** 之间。
- **T4 上的 Grayscale 挑战**：**T4** 上的 `grayscale` 排行榜收到了许多提交，时间从 **17.4 ms** 到 **47.5 ms** 不等。
   - 达到了 **17.4 ms** 的个人最好成绩。
- **卷积竞赛开始**：`conv2d` 排行榜的一项提交获得了 **T4 第一名**（**972 ms**）、**H100 第四名**（**47.8 ms**）、**A100 第七名**（**120 ms**）以及 **L4 第二名**（**291 ms**）。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1378450999546478755)** (7 条消息): 

> `Dockerized Factorio Server Activation, External Memory Systems Integration, Mem0 AI for RAG, Factorio Learning Environment PR #158` 


- **Docker 需要激活**：每个运行 **Factorio 服务器**的新 **Docker 容器**都需要通过登录一次来激活，以便在游戏内创建一个可以被接管的角色。
- **Factorio 的 Mem0 记忆**：一位成员提到了集成 [外部记忆系统](https://github.com/mem0ai/mem0) 的可能性，特别是使用 **RAG (Retrieval-Augmented Generation)** 来增强 Factorio 学习环境。
- **用 RAG 改造 Factorio**：一位成员建议 **RAG (Retrieval-Augmented Generation)** 可能会有帮助，并提到之前使用过几个版本的 [Mem0](https://github.com/mem0ai/mem0) AI 的经验。
- **建议进行 PR 审查**：一位成员指引另一位成员审查 Factorio 学习环境的 [PR #158](https://github.com/JackHopkins/factorio-learning-environment/pull/158)。


  

---

### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/)** (1 messages): 

wildman_yasei: 据我所知，只有第一个问题是 PDF 格式的。
  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1378641462333411358)** (4 messages): 

> `CuTE Examples, TiledMMA partitioning, Grouped GEMM Kernel` 


- **CuTE 示例的 MMA 指令探究**：一位遵循 **CuTE** 示例和教程的用户注意到，为 compute_90 编译第二个版本（`sgemm_2.cu`）时，尽管使用了 MMA 指令，但在 **PTX** 中并未生成任何 **WMMA** 指令。
   - 该用户澄清说，他们的困惑源于假设 **TiledMMA partitioning** 会直接转换为 **PTX WMMA** 指令，但 **TiledMMA** 构建器采用了 `UniversalFMA` atom 这一事实解释了为什么没有生成这些指令。
- **使用 GPU Tensors 启动 Grouped GEMM Kernel**：一位用户询问如何启动一个 **grouped GEMM kernel**，其问题规模（problem sizes）和参考 Tensor 已经位于 GPU 上，旨在避免 `.item()` 调用和 CPU 数据传输。
   - 此问题已在 [Discord](https://discord.com/channels/1019361803752456192/1150868614921064590/1378902510005387376) 上跨频道发布以寻求更多帮助。


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1379227608985767989)** (2 messages): 

> `PyTorch custom operators using Mojo, Modular nightly releases, Call Mojo from Python` 


- **Mojo 为 PyTorch 自定义算子提供动力！**：正如最近的一次演讲中所宣布的，使用 **Mojo** 编写 **PyTorch custom operators** 的初步实现现已在 **Modular nightly releases** 中可用。
   - 有关文档和示例，请查看此 [论坛帖子](https://forum.modular.com/t/initial-support-for-writing-pytorch-custom-ops-in-mojo/1541)。
- **Mojo 响应 Python 的调用！**：从 **Python 调用 Mojo** 能力的第一阶段已经推出。
   - 详情和更多信息可以在 [此论坛帖子](https://forum.modular.com/t/initial-support-for-calling-mojo-from-python/1514) 中找到。


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1378978030206320651)** (5 messages): 

> `AI Agent Engineering, LLMs & Foundation Models, Google Sheets for model/prompt eval, LLM Scribe tool` 


- **工程师展示 AI 专业知识**：一位拥有超过 8 年经验的 AI/ML 工程师兼全栈开发人员介绍了自己，重点介绍了其在 **AI Agent Engineering**、**LLMs & Foundation Models** 以及 **Full-Stack & Backend Systems** 方面的专业知识。
   - 该工程师还分享了作品集 [链接](https://yangming.vercel.app/)，并表达了对在尖端 AI 和 agentic workflows 方面进行合作的兴奋之情。
- **使用 Google Sheets 快速迭代模型/提示词以进行评估**：一位成员分享了一个 Google Sheet 工具，用于快速迭代模型/提示词以进行评估，并征求关于如何使其更有用的反馈，详见此 [推文](https://x.com/ahmetbuilds/status/1929423145535988192) 和 [截图](https://imgur.com/a/O3Jqdjy)。
- **LLM Scribe 简化了用于微调的手写数据集**：一位成员介绍了 LLM Scribe，这是一个用于简化微调手写数据集创建过程的工具，支持多种格式（**chatml**, **alpaca**, **sharegpt**）、自动保存、多轮对话创建、token 计数器和自定义字段。
   - LLM Scribe 的演示包括一个 [Hugging Face Space](https://huggingface.co/spaces/Gabriella0333/LLM_Scribe_Demo) 和一段 [YouTube 视频](https://www.youtube.com/watch?v=1mcYsDrXHAA&t=3s)，完整版本可在此处 [获取](https://kryptive.gumroad.com/l/gvyqep)。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1378450603037954241)** (250 条消息🔥🔥): 

> `REST API sk-or-v1 keys, Submitting end-user IDs, DeepSeek v3 free rate limit, DeepSeek provider rankings, Chess Data & LLMs` 


- **REST API 密钥困惑已澄清**：用户在尝试使用 **REST API** 时对 **sk-or-v1 keys** 感到困惑，但成员们澄清这是唯一正确的密钥。
   - 一名用户在 n8n 中遇到错误，随后被引导至[此指南](https://jannicknijholt.nl/integrate-openrouter-in-n8n/)。
- **End-user IDs：尚未准备好大规模使用**：成员们正在讨论提交可选的 **end-user IDs** 以防止滥用并改进审核的功能，该功能详见[此处](https://openrouter.ai/docs/api-reference/chat-completion#request.body.user)。
   - 讨论指出该功能**尚无可用指标（metrics）**，一位成员表示，*`eventually`（最终会有的）意味着目前指标还不可用*。
- **DeepSeek 对决：谁是最佳供应商？**：成员们辩论了 **DeepSeek** 的最佳供应商，一些人出于信任和稳定的性能更倾向于 **Parasail**（尽管成本更高，为 **$5**），而另一些人则青睐 **DeepSeek** 官方，理由是成本更低、缓存策略以及直接的模型实现。
   - 一位成员对 DeepSeek 官方 API 的服务器拥挤、中国地理位置和速度慢表示担忧；另一位成员报告称 **Deepseek** 在 `max_tokens` 方面存在问题，且对非推理输出 token 强制执行 **8K 上限**，而该上限在 **R1** 中已升级至 64k。
- **国际象棋基准测试回归？**：成员们讨论了国际象棋基准测试以及 `gpt-3.5-turbo-instruct`（一个在国际象棋数据上训练的 *instruct* 模型）令人惊讶的表现，一位成员链接的研究表明 **国际象棋训练能提高解决问题的能力** ([https://arxiv.org/pdf/2312.09390#page=29](https://arxiv.org/pdf/2312.09390#page=29))。
   - 另一位成员引用了关于该主题的一篇文章 (["https://dynomight.net/more-chess/"](https://dynomight.net/more-chess/))，指出 **RLHF 可能会降低性能**，并且 *gpt4-base*（RLHF 之前版本）在国际象棋中的表现优于 *4o*。
- **零日志... 还是并非如此**：成员们询问了 **Parasail 的 Prompt 日志政策** ([https://www.parasail.io/legal/privacy-policy](https://www.parasail.io/legal/privacy-policy))，该政策声称其 Serverless 和专用版本为 **零日志（zero-logging）**。
   - 然而，OpenRouter 的文档指出 *Prompts 会被保留一段不确定的时间*，这导致了困惑，并引发了向 OpenRouter 团队成员寻求澄清的请求。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1378477201288073257)** (156 条消息🔥🔥): 

> `Manus AI student perks, School environment, OpenManus affiliation, Deploying Manus-generated sites` 


- **AI 工作假期对神圣土地的影响**：一位成员分享了他们使用 Manus 穿越[熊野古道（日本的一条神圣道路）](https://zenschool.medium.com/spirituality-in-nature-with-ai-agents-at-kumano-kodo-the-impact-of-ai-workcation-in-a-sacred-land-c62baa87bd2f)的经历。
   - 该用户希望能够发布 Manus 复制并转化为不同艺术作品的艺术风格图像。
- **用户报告诈骗链接和账户被盗**：一位用户报告了他们认为的 **被盗机器人**，该机器人正在散布奇怪的链接和一个 **Manus fellowship 链接**。
   - 已请求管理员删除这些潜在的恶意内容。
- **学生福利成为 Manus 用户的游戏规则改变者**：一位用户就 **Manus 学生福利** 寻求帮助，例如在不丢失福利的情况下更改电子邮件地址以及发送推荐。
   - 另一位用户回复称，**学生福利** 允许使用 **免额度环境**、多达 **50 个知识条目** 以及访问 **高强度模式（high-effort mode）**。
- **关于 OpenManus 隶属关系的澄清**：一位用户询问 **OpenManus** 是否隶属于 **Manus AI**，并指出其网站引起了困惑。
   - 其他成员澄清说 **OpenManus 不隶属于** Manus AI，而是一个不包含 API 定价的“免费替代方案”。
- **Manus 部署，移除图标**：一位用户询问如何从部署的、由 Manus 生成的网站右下角移除 **Manus 图标**。
   - 其他成员澄清说无法直接移除该图标，建议用户 **下载文件并部署到其他地方**。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1378497526025683096)** (27 条消息🔥): 

> `本地 70B 模型绘图、DIY 模型存储需求、Hugging Face 数据集、Vast.ai 定价、HDD vs. SSD 瓶颈` 


- **本地运行 70B 模型绘图，速度较慢**：一名成员在本地运行了一个 **70B model** 来生成图表，并指出速度较慢，还附带了一个过程[视频](https://cdn.discordapp.com/attachments/729741769738158194/1378497525702725824/Screencast_from_2025-05-31_23-39-40.webm?ex=683f745d&is=683e22dd&hm=b87717b28e91071d704dc570fe912054b96cc4b7fec4e5706278f4e0a0df4d3e)。
   - 另一位成员表示有兴趣制作自己的模型，但缺乏存储训练数据的空间，并评论道：“有点想做自己的模型，但我甚至没有存储训练数据的空间。”
- **DIY 模型需要存储和训练数据**：成员们讨论了为训练自定义模型获取充足**存储**和**算力资源**的困难。
   - 他们指出，虽然 **Hugging Face** 上有现成的训练数据，但存储数据和模型以及所需的算力是一个重大挑战，而流式传输（streaming）训练数据是减少存储需求的一个选项。
- **Vast.ai 为云端算力提供合理的价格**：成员们建议使用 **vast.ai**，因为它价格合理。
   - 一位成员表示：“我正在使用 vast”。
- **HDD 速度不是训练的瓶颈**：一位成员建议 **HDD speed** 可能不是训练过程中的主要瓶颈，并提议使用中间 **RAM cache** 作为缓冲。
   - 成员们提到 HDD 已经便宜很长时间了。
- **自然科学家自愿参与可解释性项目**：一位具有 ML 经验的受过训练的自然科学家自愿参与项目，特别是涉及**可解释性（interpretability）**的项目。
   - 他们提到了在强化学习（reinforcement learning）、遗传算法和深度学习方面的经验，主要使用 Python 编码，并愿意学习其他语言。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1378470069159329912)** (102 条消息🔥🔥): 

> `激励 Agent 推理、MoE 中的 Token 丢弃、RL LLM Agent 的持续学习、用于持续学习的噪声激活、使用离散扩散的可变长度序列填充` 


- **推理步骤加速 Agent 训练**：通过激励 Agent 在 CoT 期间输出推理步骤，可以降低错误答案的可能性，从而训练 Agent 快速排除错误答案，正如[这篇论文](https://arxiv.org/abs/2505.05755)中针对文本扩散（text-diffusion）所建议的那样。
- **详述 MoE 中的 Token 丢弃效应**：在讨论 **MoE** 中的 token 丢弃时，有人强调少量的 token 丢弃可以起到正则化作用，提高泛化能力；而[路由崩溃（routing collapse）](https://arxiv.org/pdf/2409.12517)是由于大量 token 丢弃导致的主要问题。
- **KL 惩罚论文助力持续学习**：在寻求 **RL LLM agent** 持续学习的建议时，有人建议参考一篇关于更精确执行 KL 裁剪（KL clipping）的论文，因为在进行 RL 时熵会减少，推荐了[这篇论文](https://arxiv.org/abs/2505.22617)和[这篇论文](https://arxiv.org/abs/2505.24864)。
- **噪声破坏浅层特征以实现更好的学习**：有人提议在正向传播（forward pass）期间向激活值添加噪声，然后对没有噪声的原始权重应用更新，这可以迫使优化器挖掘更深层的信息并破坏微调期间学习到的浅层特征，参考了[这篇关于代码后门的论文](https://arxiv.org/abs/2502.17424)。
- **词序排列与扩散模型邻近？**：在讨论从词袋（bag of words）创建位置编码时，有人指出词袋听起来与扩散模型非常相似，参考了[这篇关于句子排序的论文](https://arxiv.org/abs/1607.06952)。


  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1379191707274448896)** (2 条消息): 

> `Low Dimensional Manifolds, Data Generating Process, Quotient Out Regularities` 


- **神经网络形成低维流形**：一位成员假设，在自然输入的低维流形上训练的神经网络，会自动对应于嵌入在可能的前向传播（forward passes）高维空间中的某些激活值（activations）低维流形。
   - 他们进一步建议，通过了解[数据生成过程](https://en.wikipedia.org/wiki/Data_generation)并对不同输入的激活值进行采样，可以构建出这些流形的样貌。
- **商掉模型行为中的规律性**：该成员提议将激活值流形与输入流形进行比较，以“商掉”（quotient out）数据集的规律性。
   - 目标是获得一个仅包含由权重施加的模型行为规律性的流形，并对相关方法和潜在局限性提出了疑问。
- **输入流形作为前向传播的子流形**：该成员指出，输入流形可以被视为前向传播流形内部的一个子流形。
   - 他们表示“不确定这有什么帮助”，并寻求关于这一视角如何辅助理解模型行为的见解。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1378803024981528586)** (24 条消息🔥): 

> `Hugging Face chunked prefill, RWKV model addition, lm-evaluation-harness bugs, lm-evaluation-harness documentation, max_seq_lengths cmdline argument` 


- **HF 分块预填充加速长文本基准测试**：HF 现在支持 **分块预填充 (chunked prefill)**，这对于使用 `lm_eval` 运行长文本基准测试非常有用；然而，在 `generation_config.json` 中设置 `prefill_chunk_size` 不起作用，但在 `lm_eval/models/huggingface.py` 的 `self.model.generate` 调用中设置则有效。
   - 这被描述为“重大突破”，因为它能防止运行 ruler 时出现 OOM 错误，而且如果你确实在使用长上下文，没有它很难运行长文本基准测试。
- **PR 修复 lm-evaluation-harness 中的错误**：一位成员提交了一个 PR 来修复一些错误：[https://github.com/EleutherAI/lm-evaluation-harness/pull/2983](https://github.com/EleutherAI/lm-evaluation-harness/pull/2983)，用于运行 **longbench**。
   - 某些任务的表现好得多，但他们无法复现所有结果；他们计划合并该 PR 并添加警告。
- **命令行上的 DEFAULT_SEQ_LENGTHS**：一位成员询问如何在命令行传递 `DEFAULT_SEQ_LENGTHS`。
   - 另一位成员回复了 lm-evaluation-harness 仓库中相关部分的链接：[https://github.com/EleutherAI/lm-evaluation-harness/blob/8bc4afff22e73995883de41018388428e39f8a92/lm_eval/__main__.py#L301](https://github.com/EleutherAI/lm-evaluation-harness/blob/8bc4afff22e73995883de41018388428e39f8a92/lm_eval/__main__.py#L301)。
- **Ruler 任务特定参数**：一位成员链接了 ruler 任务的 readme：[https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/ruler](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/ruler)。
   - 他们提到任务特定参数的公共 API 可能还不清晰，也许他们应该移除默认长度并要求用户指定。
- **序列长度相关的诡异问题**：当添加 `--metadata='{"max_seq_lengths":[36864]}'` 时，一位成员同时得到了该值和默认值，但默认大小导致结果为 -1；他们还添加了 `--model_args=pretrained=...,max_length=40000`，但不知为何收到了超过模型预定义最大长度（**32768**）的警告。
   - 他们指出已经修改了模型配置以包含更多 RoPE 条目，且在其他任何地方都没看到指定了 **32768**。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1378591620538306660)** (69 条消息🔥🔥): 

> `AiderBench 依赖, DeepSeek Agent, Opus 测试, Mac M3 性能, 去中心化推理网络` 


- **AiderBench 依赖项讨论**: 成员们讨论了在不使用 **Docker** 的情况下运行 **AiderBench** 所需的依赖项，其中一人建议由于依赖项过于沉重，使用 **Docker** 会更容易。
   - 其他人建议使用 **VPS** 作为替代方案。
- **DeepSeek Agents**: 成员们简要提到并询问了 **DeepSeek** agent，特别是 **AgenticSeek**。
   - 另一位成员提到 **Agent0** 是一个更优的替代方案。
- **Opus 模型测试问题**: 一位成员询问是否有人在 aider 模型的上下文中测试过 **Opus**。
   - 另一位成员建议查看专门的基准测试频道以获取模型性能详情。
- **M3 Mac 性能深度探讨**: 成员们辩论了为什么 **Mac M3** 芯片在处理大模型时表现不错，其中一人认为内置的神经网络引擎对此有所帮助，达到了 **18TOPS**（每秒万亿次操作）。
   - 另一位成员将性能归功于 **Mac** 巨大的内存带宽，特别是 **LPDDR5X** 内存，在 Max 版本上高达 **540GB/s**。
- **去中心化推理网络构想**: 一位成员提议建立一个去中心化网络，由 **3-5 个节点** 批量完成相同的推理，通过解决争议来为本地可运行的 **LLM** 创建一个廉价且具有弹性的 API。
   - 另一位成员开玩笑说这个想法可以变成一种新的 *crypto*（加密货币）。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1378639487982375032)** (28 条消息🔥): 

> `Gemini 模型问题, Aider 自动对话摘要, 多仓库使用 Aider, 用于 HTML/CSS 的 SCM 文件, 最佳本地模型建议` 


- **Gemini 模型面临任务完成问题**: 一位用户报告了在 **aider** 中使用 **Gemini 模型**（gemini/gemini-2.5-pro-preview-05-06 和 gemini/gemini-2.5-flash-preview-05-20）时遇到的问题，由于文件夹和文件命名错误以及整体不稳定性，难以完成多个任务。
   - 他们指出，虽然代码编写结果不错，但这些模型似乎无法很好地与 aider 集成。
- **Aider 自动摘要对话**: 正如一位成员所述，**Aider** 会自动总结对话历史，辅助上下文管理。
   - 若要向 Aider 提供 **git commit** 和 **git diff** 视图，请使用 `/run git diff ...` 命令，随后会提示将其添加到聊天中。
- **只读访问解决多仓库问题**: 一位成员建议在 **aider** 中使用 `/read-only` 命令来访问多个仓库的文件，因为 aider 不会追踪符号链接（symlinks）。
   - 例如，`/read-only /Users/username/Downloads/some_random_file.md` 允许对当前仓库之外的文件进行只读访问。
- **寻求用于 HTML/CSS 编辑的 SCM 文件**: 一位用户正在寻求 **HTML/CSS 的 SCM 文件**，以提高 **aider** 在编辑这些文件类型时的性能。
   - 他们认为性能不佳是由于上下文长度问题导致的，即使使用的是 **1M context LLM**。
- **应使用 Devstral 作为本地模型**: 一位用户寻求关于运行最佳本地模型的建议，其硬件配置为 **4x3090s**、**256GB RAM**、**约 100k 上下文窗口**，任务涉及对现有 **Rust** 和 **Typescript** 代码进行编辑和修复。
   - 建议该用户尝试 **Devstral**，一位成员指出新版 **R1** 的某些版本也值得一试。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1379195123321274429)** (3 条消息): 

> `Modular 黑客松, GPU 编程工作坊, Mojo kernels, MAX Graph 模型架构, PyTorch custom ops` 


- **Modular 举办黑客松**: Modular 正在举办 **另一场黑客松**，开放虚拟参与，在周末举行，重点关注 Mojo kernels、MAX Graph 模型架构和 PyTorch custom ops。[查看详情！](https://lu.ma/modular-hack-weekend)
- **GPU 工作坊开启黑客松**: 为了开启周末活动，Modular 将在他们位于 Los Altos 的办公室举办 **GPU 编程工作坊**（线下参与及通过直播虚拟参与）。
- **Mojo 新用户看到性能提升！**: 一位最近的计算机科学毕业生在观看 **Fireship 视频** 并在基础 ML 模型上尝试 Mojo 后，报告称看到了 *提升*，并对优化其 ML 代码感到兴奋。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1378463111178227712)** (77 messages🔥🔥): 

> `Mojo 中的类型检查、Mojo 与 godbolt.org、Copyable 和 ExplicitlyCopyable Trait、Mojo 性能分析、Mojo 的 C 绑定生成器` 


- **讨论 `_type_is_eq` 的局限性**：成员们讨论了使用 `_type_is_eq` 来检查类型，指出它无法检查具有任意指向类型（pointee types）的指针类型，而 [类型名称反射](https://example.com/reflection) 至少可以帮助检查指针。
   - 有人认为这种方法很 *“邪道 (cursed)”*，但也有人喜欢它与 C++ 的 RTTI 相比没有运行时开销。
- **反射 API vs Trait 系统改进**：提到了反射 API 的可能性，并强调了其用途（允许实现 *“利用编译时信息构建序列化器的 Trait”*）。
   - 有人提出疑问，这是否比 Trait/类型系统的改进具有更高的优先级。
- **`Copyable` 与 `ExplicitlyCopyable` Trait 的比较**：讨论了一个类型同时符合 `Copyable` 和 `ExplicitlyCopyable` Trait 的目的，并举例说明一个 **100+ GB 的 ML 模型** 移动（move）比复制（copy）更好。
   - 此外还指出，实现 `Copyable` Trait 会告知编译器何时可以执行隐式复制。
- **Mojo 代码性能分析（Profiling）指南**：成员们讨论了对 Mojo 代码进行性能分析，指出与 **C++** 或 **Rust** 兼容的工具通常都适用，例如配合 `flamegraph` 使用 `perf`。
   - 有人建议 CPU 厂商的 HPC 性能分析器（Intel VTune, AMD uProf, ARM Streamline）可以为优化提供详细的微架构洞察。
- **开发中的 C-to-Mojo 绑定生成器**：一名成员正在开发 **C 到 Mojo 的绑定生成器**，旨在处理除 *“极其糟糕的紧凑结构体 (packed structs)”* 以及可能影响调用约定的 `restrict` 和 pragma 之外的大多数情况。
   - 他们建议使用 `pixi.toml` 文件来指明 Mojo GUI 项目的依赖关系，并对代码某些部分中复制组件而非借用（borrowing）的做法表示担忧。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1378467915128635422)** (73 messages🔥🔥): 

> `MCP 与 Claude Desktop、MCP 传输层、MCP 中的动态工具注册、MCP 客户端实现、MCP 客户端的 Elicitations 支持` 


- **MCP 快速入门：Claude 与服务器设置**：一名成员根据 [MCP 文档](https://modelcontextprotocol.io/quickstart/server#why-claude-for-desktop-and-not-claude-ai) 创建了他们的第一个 MCP 服务器，并建议使用 **Claude** 来学习 MCP。
   - 他们还使用了 **stdio** 传输层使其能够与 **Claude 桌面应用** 配合工作。
- **在系统提示词中注入数据**：成员们讨论了在 MCP 中直接向系统提示词注入数据的方法，包括使用 fastmcp 构造函数中的 **prompt** 选项，该选项允许指定要添加到系统提示词的信息。
   - 同时提到客户端必须显式支持该功能。
- **支持所有特性的 MCP 客户端**：成员们注意到目前缺乏完全支持规范和原语的 MCP 客户端。
   - 一名成员表示 [schema.json/.ts](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/fb34d1d6da2287f82cfdf46c1f91b6fb262cdd38/schema/2025-03-26/schema.json) 是最接近完整规范的东西，但很难遵循。
- **动态工具注册问题**：一名成员报告了 MCP 中动态工具注册的问题，即新注册的工具在同一个消息周期内无法立即被发现或调用。
   - 他们正在寻求该问题的解决方案，因为目前工具只有在当前链执行完成后才能被发现。
- **Claude Desktop 不支持流式 HTTP MCP 服务器**：一名成员寻求帮助，希望将 Claude 桌面版连接到可流式传输的 HTTP MCP 服务器。
   - 另一名成员表示，目前这仅在 **Claude Max 计划** 的网页版上可行，并附上了 [YouTube 教程链接](https://www.youtube.com/watch?v=CU6EWD9IJKk)。


  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1378546523838742579)** (3 条消息): 

> `Aura 结合 AppAgentX，通过语音 Agent 控制 Android 手机，MCP 服务器连接到 MCP 知识库` 


- ****Aura** 连接 **AppAgentX****：在*抽了两包烟并经历了 14 小时的头疼*之后，一位成员成功将 **Aura** 与 **AppAgentX** 连接，从而可以使用任何 MCP Client 控制 Android 设备。
   - 他们计划在修复来自 OpenAI 的 Realtime 的 SST 和 TTS 后发布代码，预览代码已在 [GitHub](https://github.com/IhateCreatingUserNames2/Aura_AppAgentX/tree/main) 上提供。
- **通过语音 Agent 控制 **Android 手机****：该成员的目标是创建一个可以通过语音控制整个 **Android 手机** 的 Agent。
   - 这是通过将 **Aura** 的功能封装进 A2A 工具，并将其广播到 A2A 转 MCP 网关 (Aira Hub) 来实现的。
- ****MCP 服务器** 连接到 MCP 知识库**：一位成员增加了将 MCP 服务器连接到 **MCP 知识库**（客户端宿主）的功能，用于微调 RAG。
   - 该内容通过一篇 [LinkedIn 帖子](https://www.linkedin.com/posts/nerdai_mcp-rag-ai-activity-7335292265386463232--h0Y?utm_source=share&utm_medium=member_ios&rcm=ACoAABpyymkBvdiXT4PxiTwTckoywfEnXZRbcCM) 进行了分享。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1378453913589518359)** (10 条消息🔥): 

> `Audio Overviews 的语言设置，NotebookLM 聊天 API 集成，使用 NotebookLM 记录讲座，Audio Overview 长度限制` 


- ****Audio Overviews 的语言设置仍然受限****：用户在创建**非英语语言的音频概览**时遇到困难，即使自定义 AI 聊天以另一种语言回复，通用设置仍然控制着输出语言。
   - 一位用户报告称，*没有简单的选项可以临时请求不同输出语言的播客*，并指出目前缺少导出功能。
- ****目前尚无用于客户支持聊天的直接 API****：一位用户询问 **NotebookLM 的聊天功能** 是否可以通过 API 集成到其他工具中，以进行商业客户支持。
   - 一位成员表示 *NotebookLM 是一个面向终端用户的工具*，建议使用 **Google Cloud - Conversational Agents 和 Dialogflow API** 作为面向更广泛受众应用的替代方案。
- ****NotebookLM 底层使用了 Google API！****：一位成员推测 **NotebookLM 在其用户友好的界面背后利用了 Google API**。
   - 另一位成员分享说，他们向社区学院的学生介绍 **NotebookLM** 等新技术以补充讲座内容，并[链接了一个美国历史和美国政府讲座的 YouTube 播放列表](https://www.youtube.com/playlist?list=PLzjEb2_3El48Kix7QZ1dT3z8qTgFfbFnv)。
- ****Audio Overview 长度约束分析****：一位用户报告称，上传 **16 个 YouTube 视频链接**（每个 20-35 分钟）仅生成了 **15 分钟的英语音频概览**。
   - 该用户还在等待**其他语言的音频播客摘要**的修复，因为*即使内容和 Prompt 相同，它们也比英文版本短*。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1378502943933988964)** (63 条消息🔥🔥): 

> `视频上传，元数据嵌入，Pro 订阅音频播客限制，NotebookLM 在美国境外的可用性，取消 Pro 功能` 


- ****视频上传陷入死循环****：用户报告称在上传 **MP4 资源**后，资源项会无休止地旋转，必须刷新应用/页面才能完成上传。
   - 虽然没有立即提供解决方案，但这突显了视频上传中潜在的 UX 问题。
- ****元数据嵌入：内容丰富化探索****：一位用户询问关于将**元数据**嵌入到 PDF 中以提高加载到 NotebookLM 资源中的内容质量的问题。
   - 这是一个开放性问题，即是否支持此功能或什么是最佳方法。
- ****Pro 订阅播客限制：用户的哀叹****：一位用户对尽管拥有 **Pro 订阅**但**音频播客**仍受限表示沮丧。
   - 该用户发现使用另一个账号时则没有这些限制。
- ****访问 NotebookLM：地理位置并非决定因素****：用户讨论了从美国境外访问 NotebookLM 的问题，一位用户确认无需 VPN 即可访问。
   - 一位用户建议，访问问题可能是由账号设置而非地区引起的。
- ****NotebookLM 事实幻觉：Bug 报告****：一位用户报告称 NotebookLM 会创建**随机的、无来源的事实**，然后表现得好像这些事实就在原始资源中一样。
   - 建议在 **bugs 频道**报告此行为。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1378454294151561440)** (63 条消息🔥🔥): 

> `DeepHermes-3 Discord 集成, Demis Hassabis AGI 预测, Prompt Lookup Decoding 提速, AI 中的广义过拟合, Claude 模型弃用` 


- **DeepHermes-3** 进驻 **NousResearch Discord**：一名成员建议将 **DeepHermes-3** 集成到 Discord 中，如果它能实现自主运行，将有助于促进有机互动。
   - 随后一名成员澄清说，它已经存在于某个特定频道中了。
- **DeepMind 的 Demis Hassabis** 预测 **2030 年实现 AGI**：**Demis Hassabis** 就未来趋势发表了一场精彩的 [演讲](https://www.youtube.com/watch?v=U3d2OKEibQ4)，并预测 **AGI 将在 2030 年左右**实现。
   - 他指出，自 **2010** 年早期成立以来，**DeepMind** 始终处于前沿领域，且从未放慢脚步。
- **Prompt Lookup Decoding** 显著提升速度：**Prompt lookup decoding** 通过在提示词中进行简单的字符串匹配来生成候选 token 序列，从而取代了草稿模型（draft model），在基于输入的任务中实现了 **2x-4x** 的加速。
   - 该方法可用于任何解码器模型，无需修改模型或使用外部数据存储，且支持贪婪搜索（greedy）和采样（sampling）技术，详见 [GitHub](https://github.com/apoorvumang/prompt-lookup-decoding)。
- **Mindcraft 框架**实现 **Minecraft 游戏玩法**：一名成员分享了 **Mindcraft 框架** ([github.com](https://github.com/kolbytn/mindcraft)) 以及相关的 **Andy 模型** ([Ollama](https://ollama.com/Sweaterdog/Andy-4) 和 [HuggingFace](https://huggingface.co/Sweaterdog/Andy-gen)) 的链接，这些模型专门针对 Java 版 Minecraft 进行了训练。
   - 其中一个模型 **Andy-4** 是一个拥有 **80 亿**参数的模型，在单块 **RTX 3090** 上训练了三周，具备先进的推理、多步规划和强大的游戏内决策能力。
- **Circuit-Tracer 仓库**带来新的**机械可解释性（Mechanistic Interpretability）**发现：一位独立研究员利用 Anthropic 开源的 circuit-tracer 仓库来映射概念，并在机械可解释性研究中可视化模型回答查询时的行为，初步文章已发表在 [LinkedIn](https://www.linkedin.com/pulse/advancing-mechanistic-interpretability-interaction-nets-zsihcv) 上。
   - 这项工作的重点是识别信息流如何跨层传递，展示了信息是如何被抽象或压缩的。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1378930189358207067)** (1 条消息): 

> `MINDcraft, LLM Agent 协作, MineCollab` 


- **MINDcraft 与 MineCollab 发布**：一名成员分享了论文 ["MINDcraft: A Platform for Collaborative Embodied Agents"](https://arxiv.org/pdf/2504.17950)，介绍了 **MINDcraft**（一个旨在让 **LLM Agent** 控制 **Minecraft** 角色的平台）和 **MineCollab**（一个测试具身和协作推理的基准测试）。
   - 研究发现，当前 Agent 有效协作的主要瓶颈是**高效的自然语言通信**，当要求 Agent 沟通详细的任务完成计划时，其性能下降幅度高达 **15%**。
- **LLM Agent 在多智能体协作中面临挑战**：论文结论指出，现有的 **LLM Agent** 在**多智能体协作**（尤其是具身场景）方面优化不足。
   - 论文强调需要采用 In-context Learning 和模仿学习之外的方法，因为目前的 SOTA Agent 在沟通详细任务计划时会面临性能下降。


  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1378733886334570699)** (6 messages): 

> `System Prompt Learning, LLM Scribe Tool, Robotics Mouse` 


- **Apple 研究 AirPods？**: 分享了一个关于 **Apple** 对 **AirPods** 研究的 [Perplexity 链接](https://www.perplexity.ai/page/apple-study-shows-airpods-coul-7_q_Jgn0ROGQmIqFP_0FHA)。
   - 原始 Discord 消息缺乏上下文，使得该研究的具体细节仍然是个谜。
- ****SPL**：LLM 随时间变得更好**: **System Prompt Learning (SPL)** 教会 LLM 学习解决问题的策略，在 Arena Hard 上提升了 **8.6%** 的性能，在 AIME24 上提升了 **6.67%**，详见 [Hugging Face 博客文章](https://huggingface.co/blog/codelion/system-prompt-learning)。
   - 这个 *optillm* 中的开源插件灵感来自 [Karpathy 的想法](https://x.com/karpathy/status/1921368644069765486)，通过添加 `spl-` 前缀即可与任何 OpenAI 兼容的 API 配合使用。
- **机器人老鼠**: 分享了一个**机器人老鼠**的 [YouTube 视频](https://m.youtube.com/watch?v=gQidYj-AKaA)，并宣称 *airesearch.js.org* 拥有*相当不错的机器人技术*。
   - 一位成员提议集成抓取和数据提取等功能。
- ****LLM Scribe** 助力开发**: 一位成员介绍了 **LLM Scribe**，这是一个用于创建微调所需手写数据集的工具，具有多格式导出、自动保存、多轮创建、Token 计数器、目标跟踪和自定义字段等功能，详见[此 YouTube 演示](https://www.youtube.com/watch?v=1mcYsDrXHAA&t=3s)。
   - 该工具可在 [Hugging Face Spaces](https://huggingface.co/spaces/Gabriella0333/LLM_Scribe_Demo) 上使用，并提供[完整版购买](https://kryptive.gumroad.com/l/gvyqep)。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1378930189358207067)** (1 messages): 

> `MINDcraft platform, LLMs adaptive collaboration, embodied reasoning tasks, MineCollab benchmark, natural language communication` 


- **MINDcraft 平台实现 LLM Minecraft 协作**: 一位成员分享了 [MINDcraft](https://arxiv.org/pdf/2504.17950) 的链接，这是一个易于扩展的平台，旨在让 **LLM Agent** 能够控制开放世界游戏 **Minecraft** 中的角色。
   - 他们还分享了 [MineCollab 基准测试](https://mindcraft-minecollab.github.io/)的链接，用于测试具身推理和协作推理的不同维度。
- **LLM 的自然语言能力成为协作瓶颈**: 一项实验研究发现，当前最先进的 Agent 在有效协作方面的主要瓶颈是高效的**自然语言通信**。
   - 当要求 Agent 沟通详细的任务完成计划时，其性能下降了多达 **15%**，这表明 **LLM Agent** 在多 Agent 协作（尤其是在具身场景中）方面尚未得到充分优化。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1378541063836405911)** (53 messages🔥): 

> `Jason 的 Nitter 帖子，EleutherAI 的 Common-Pile 数据集和 Comma 0.1 模型，用于图像编辑的 Kontext Chat，NYT 向 Amazon 授权内容用于 AI 训练，Karpathy 的 ChatGPT 版本使用指南` 


- ****Jason 的 X-Ware 帖子出现在 Nitter****: 用户 @agikoala (Jason) 在 **2025 年 6 月 1 日**发布了一条包含图片的帖子，获得了 **2 个点赞**和 **18 次分享**，该内容通过 [Nitter](https://xcancel.com/agikoala/status/1929048742516162940?s=46) 转发。
- ****EleutherAI 发布 Common-Pile 和 Comma 0.1****: EleutherAI 发布了 **Common-Pile**（一个 **8TB** 的自由数据集及其过滤版本）以及 **Comma 0.1**（一个在过滤数据集上训练的 **7B** 基础模型）；其架构与 **Llama 3** 匹配，并使用 **lingua** 在 **64 台 Nvidia H100 GPU** 上完成训练；源码可在 [HuggingFace](https://huggingface.co/common-pile/comma-v0.1) 获取。
   - 它被认为是目前为止*最接近全栈 FOSS 模型*的作品，尽管其相关推文因未知原因被删除。
- ****Replicate 推出 Kontext Chat：用文字编辑图像****: Replicate 推出了 **Kontext Chat**，这是一个通过对话命令编辑图像的开源应用程序，基于 **Hono** 构建并托管在 **Cloudflare Workers** 上，旨在作为开发者的起点，该消息在 [X](https://xcancel.com/replicate/status/1929160560295506417?s=46) 上公布。
- ****NYT 授权内容给 Amazon 用于 AI****: 纽约时报（NYT）与 Amazon 签署了一项协议，授权 NYT 的内容用于 AI 训练，包括 Amazon 的基础模型（foundation models），该消息已在 [Twitter 上公布](https://xcancel.com/natolambert/status/1929175745596620968?s=46)。
   - 社区成员推测，此举表明 **NYT 对 OpenAI 的诉讼**主要是为了确保获得报酬，而非出于道德立场。
- ****Karpathy 的 ChatGPT 模型菜单：按需选择****: Andrej Karpathy 分享了他有效使用不同 ChatGPT 版本的指南，建议重要任务使用 **'o3'**，日常使用 **'4o'**，编程使用 **'4.1'**，深度研究则使用基于 o3 的 **'Deep Research'**，该指南在 [X](https://xcancel.com/karpathy/status/1929597620969951434) 上发布。


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1378593712107421756)** (14 messages🔥): 

> `AIE World's Fair 2025，实时生产级 AI Bot 协作，AIE 的 Bug 报告系统` 


- ****AIE World's Fair：记好你的日历！****: **AIE World's Fair** 将于 **2025 年 6 月 3-5 日**在旧金山举行；访问 [ai.engineer/#events](https://www.ai.engineer/#events) 查看不需要门票的周边活动。
- **Discord 社区构建“实时 AI Bot”**: 一位成员宣布，Discord 社区内协作构建了一个**实时生产级 AI Bot**，并分享了一个新的 Gen UI 框架，随后该框架于今天部署到了 AIE 网站。
   - 该 Bot 可在 [ai.engineer/ai](https://ai.engineer/ai) 访问，讨论线程链接见[此处](https://discord.com/channels/822583790773862470/1378055295401459813/1378137211995689061)。
- **AIE 网站需要“Bug 报告”系统**: 随着多个 **Bug 报告**的提交，一位成员建议为 AIE 网站建立一个更精简的**反馈循环（feedback loop）**，以促进自我改进。
   - 另一位成员自愿协助处理 **Bug 报告**。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1378994198098870312)** (2 messages): 

> `TPS 基准测试，回归测试，性能指标` 


- **回归测试为性能稳定性铺平道路**: 一位成员询问了关于在不同配置下基准测试 **TPS** 的检查，以确保固定配置下不会出现性能或兼容性回归。
   - 另一位成员确认即将为性能和评估指标添加**回归测试**，并提到目前有一个 **PR** 正在处理此事。
- **聚焦 TPS 基准测试**: 讨论强调了在不同配置下进行 **TPS**（每秒事务数）基准测试的重要性。
   - 目标是主动识别并防止性能回归，确保一致的性能水平。


  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1378897534231253042)** (45 条消息🔥): 

> `LLaMA-3 70B Fine-Tuning, DP vs TP Performance, FP8 vs BF16 Comparison, Compile impact on TPS, Loss Parallel implementation` 


- **LLaMA-3 微调 "Golden Paths" 正在进行中**：一名成员正在开发用于微调不同模型的内部 "golden paths"，从 **LLaMA-3 70B** 开始，涵盖短上下文和长上下文长度，并分享了 **8k 上下文**长度实验的初步见解。
   - 该成员旨在分享关于数据并行 (DP) vs 张量并行 (TP)、FP8 vs BF16 以及 Compile vs No Compile 对性能影响的发现。
- **FP8 Compile 提升 TPS，但占用更多预留内存**：实验表明，在禁用 compile 时，**FP8** 的 **TPS** 最低；但在启用 compile 时，其 **TPS** 远超其他配置。
   - 观察发现，**FP8 + compile** 的活跃峰值内存最低，但预留峰值内存最高。一名成员建议在“高预留内存”场景下尝试运行 `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"`。
- **更高的 TP 会降低 TPS，且需要更多内存**：实验表明，增加张量并行 (TP) 会导致吞吐量 (**TPS**) 下降，这可能是因为矩阵乘法集合通信 (matmul collectives) 的开销超过了纯 FSDP 中模型参数的开销，且 **FP8** 似乎无法缓解这一开销。
   - 此外还观察到，更高的 TP 会导致更高的活跃峰值内存，这可能是因为用于计算 loss 的输出层等开销较大的层被复制了，导致在每个设备 batch size 较大时占用更高。一名成员建议针对极长上下文采用 Loss Parallel 实现。
- **"Loss Parallel" 可能降低内存占用**：一名成员建议实现 "loss parallel"，这可能降低内存使用。
   - 另一名成员澄清说，只有层的输出被复制了，loss parallel 会提高内存性能，但即使没有它，内存占用也应该与 FSDP 持平或更低。
- **通用 HF Tokenizer 支持等待审核**：支持通用 [Hugging Face tokenizer](https://github.com/huggingface/transformers) 的工作正在进行中。
   - 一名成员已添加单元测试并等待审核，但如果需要尽快支持 Tool Calling，已准备好调整优先级。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1378866338788802660)** (45 条消息🔥): 

> `Claude Code analysis, DSPy talks at AI Engineering and Databricks DAIS, DSPy 3.0 release in June, DSPy and DARPA's Advanced Research Concepts lab, Agentic orchestration` 


- **通过净室设计 (Cleanroom Design) 分析 Claude Code 内部机制**：成员们讨论了对 [Claude Code 的分析](https://southbridge-research.notion.site/claude-code-an-agentic-cleanroom-analysis) 以及它是否开源。对话澄清该分析采用了**净室设计**原则，以避免直接接触专有技术。
   - 净室工程的优势在于无需接触源代码：[https://en.wikipedia.org/wiki/Clean-room_design](https://en.wikipedia.org/wiki/Clean-room_design)。*该术语意味着设计团队在“干净”的环境中工作，证明未受到竞争对手所用专有技术的任何知识污染。*
- **DSPy 演讲征集社区意见**：一名成员正在为 AI Engineering 和 Databricks DAIS 准备 DSPy 演讲，寻求社区对涵盖主题和重点用例的建议。
   - 特别感兴趣的领域包括 **DSPy 基础概念 (signatures, programs, 常用 optimizers)、用例 (从 PDF 和图像中提取结构化输出) 以及高级主题 (RL, datasets, 类似 MiPro 的高级 optimizers)**。
- **DSPy 用于 DARPA 项目**：DSPy 被用于 **DARPA 的 Advanced Research Concepts 实验室**，为“协作知识策展 (Collaborative Knowledge Curation)”兴趣领域构建解决方案，该项目目前正拆分为一家公司。
   - 有人对该消息回复了 "[what gif](https://tenor.com/view/i-said-what-i-said-nene-leakes-real-housewives-atlanta-rhoa-gif-16456054)"。
- **DSPy 3.0 准备于 6 月发布**：社区正准备在 **6 月发布 3.0 版本**。
   - 成员们还对如何将现有流水线迁移到 DSPy、DSPy 在何时最有效、如何将其用于合成数据生成，以及它与其他 Agent 解决方案的区别感兴趣。
- **DSPy 对 Agent 框架的立场**：关于在 **DSPy 之上构建 Agent 框架** 的潜力存在讨论，重点在于一等公民环境 (first-class environments)、处理奖励 (rewards) 以及利用它们通过 optimizers 进行在线学习。
   - 一名成员提到了 [Agenspy](https://github.com/SuperagenticAI/Agenspy)，另一名成员目前正在勾勒一个基于 Claude Code 的实现方案。


  

---

### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1379211955646234766)** (1 messages): 

> `Gradio Agents x MCP Hackathon` 


- ****Gradio Agents x MCP Hackathon** 来了！**：**Gradio Agents x MCP Hackathon** 的注册现已在[此处](https://huggingface.co/Agents-MCP-Hackathon)开放。
   - 黑客松周正在进行中！
- **收看 **Gradio Agents x MCP 直播****：**Gradio Agents x MCP 直播**将于 6 月 3 日在 YouTube [此处](https://discord.com/events/1059199217496772688/1379207318700294245)播出。
   - 准时收看以了解更多关于 Agent 和黑客松的信息。
- **加入 **黑客松 Office Hours****：为 HuggingFace Discord 服务器中的黑客松参与者准备的 **黑客松 Office Hours** 环节将于 6 月 4 日星期三举行。
   - 请查看活动日历以获取时间段更新。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1379151790091796584)** (3 messages): 

> `E-Library-Agent, Gradio Agents & MCP Hackathon, Scaling Agents in Finance workshop` 


- **E-Library-Agent：记住你读过的一切**：来自 @itsclelia 的一个新开源项目 **E-Library-Agent**，通过摄取[来自各种来源的数据](https://t.co/CgPF3uKbBJ)，帮助用户逐步构建其数字图书馆。
- **Gradio Agents & MCP Hackathon 正式启动**：**Gradio Agents & MCP Hackathon** 将于明天启动，设有 [$16.5k 奖金](https://t.co/TBAPFsNWU1)和 [$900k 算力额度](https://t.co/TBAPFsNWU1)，包含 **3 条赛道**：*MCP Tool/Server*、*Agent 自定义组件*以及 *Agentic Demo 展示*。
   - 明天还将举办一场启动[直播](https://t.co/FzLmzviwRz)。
- **利用 Agentic AI 自动化金融工作流**：由 @jerryjliu0 上周主持的**金融领域 Agent 扩展研讨会**的全套幻灯片现已发布，旨在利用 Agentic AI [自动化金融任务的文档工作流](https://t.co/Crfy50pB4j)。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1378645167820771398)** (29 messages🔥): 

> `Nested Workflow Event Streaming, Disabling Streaming During Document Indexing, Migrating from OpenAI LLM to vLLM, Tool Visibility Bug in llama-index-llms-google-genai, AI Powered Web Browser` 


- **嵌套 Workflow**：从嵌套 Workflow 中流式传输所有事件的最佳模式是让**父级 Workflow 迭代子级 Workflow 的事件流**并将事件向上传播。
   - 这比将父级上下文传入子级 Workflow 更好，因为当子级 Workflow 在不同上下文中变为父级时，后者可能会导致可组合性问题。
- **在索引期间禁用流式传输遇到困难**：一位成员在尝试使用 `LangChainLLM` 配合 `AzureChatOpenAI(streaming=False)` 和 `.complete(prompt)` 生成摘要时，无法在文档索引期间禁用**流式传输 (streaming)**。
   - 尽管设置了 `streaming=False` 并使用了 `.complete()`，LLM 仍继续流式输出 Token，这让用户怀疑是 **LlamaIndex** 或某个隐藏的回调管理器强制开启了流式行为。
- **Gemini 工具不可见**：在将 `llama-index-llms-google-genai` 从 **0.1.14** 升级到 **0.2.0** 后，工具对 **Gemini 模型** 变得不可见。
   - 该问题已被成员确认，并在 [v0.2.1](https://github.com/run-llama/llama_index/pull/18933) 中修复。
- **Schema 抓取器**：一位成员开发了一个 **AI 驱动的 Web 浏览器**，可以抓取网站的 Schema。
   - 它创建了一系列非 AI 动作来提取特定字段，从而创建可重用的策略。该成员正在询问与直接获取网站 HTML 作为 Markdown 并解析数据相比，这种方法是否值得。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1378498530028687521)** (5 messages): 

> `AdamW Optimizer, SFT Training, Probability and statistics` 


- **AdamW 优化器默认值需要补丁**：来自[最近一篇论文](https://arxiv.org/abs/2505.21829)的见解表明，当 beta1 和 beta2 相等时，**AdamW** 表现最佳，**0.95** 是一个不错的值。
   - 目前 [PyTorch 默认值](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) beta1 (**0.9**) 和 beta2 (**0.999**) 被认为不是最优的，因此有人呼吁提交补丁。
- **征集 SFT 训练专家**：一位成员询问如何寻找在**监督微调 (SFT)** 方面有经验的人员，特别是熟悉 Transformer Reinforcement Learning (TRL) 库中 *SFTTrainer* 类的专家。
- **关于概率论与数理统计的思考**：一位成员分享道：*概率论与数理统计的核心在于“可能”之中的“必然” (musts of may)*。


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1379103240855228436)** (12 messages🔥): 

> `RLVR Measurement, FP8 Training Stability, SwiGLU Activation, Smooth-SwiGLU, LLM Baseline Evaluations` 


- **RLVR 客观性受到挑战**：一位成员表示，由于心理上的吸引力，人们可能倾向于相信 **RLVR** 效果很好，但*客观地衡量它*一直很困难。
- **FP8 训练扩展至万亿级规模**：讨论了一篇关于 [将 FP8 训练扩展到万亿级 token LLM](https://arxiv.org/abs/2409.12517) 的论文，详细介绍了如何使用 **FP8 精度** 在高达 **2 万亿 token** 上训练大语言模型。
   - 该论文指出了 **FP8** 训练中与 **SwiGLU 激活函数** 相关的稳定性问题，并引入了 **Smooth-SwiGLU** 以确保训练稳定。
- **LLM 收益主张受到质疑**：一篇 [LessWrong 帖子](https://www.lesswrong.com/posts/p8rcMDRwEGeFAzCQS/incorrect-baseline-evalutions-call-into-question-recent-llm) 批评了最近的 **LLM** 论文，认为*声称的收益可能是由于基准测试（baselines）太弱导致的*。
- **动态 Discord 时间戳**：一位成员分享了一个用于创建动态 **Discord** 时间戳的实用工具 [r.3v.fi/discord-timestamps/](https://r.3v.fi/discord-timestamps/)。
- **SwiGLU 激活函数博客文章**：一位成员分享了一篇 [博客文章](https://jcarlosroldan.com/post/348) 以进一步解释 **SwiGLU 激活函数**。


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1378627388224835585)** (1 messages): 

> `GitHub MCP Vulnerability, Invariant Labs Post-Mortem Report, GitHub Security Risks` 


- **MCP GitHub 攻击向量复盘**：一位成员分享了来自 [Invariant Labs 的复盘报告](https://invariantlabs.ai/blog/mcp-github-vulnerability)，详细介绍了一个针对 **GitHub MCP** 的攻击向量，该向量可将私有仓库内容泄露到公开 **PR** 中。
- **GitHub 安全风险凸显**：讨论强调了随着 **MCP** 在 **GitHub** 上被快速采用而带来的潜在安全风险。
   - 一位成员指出，随着 **MCP** 变得越来越流行，他们*“早就预料到了”*会出现这类漏洞。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1378531241141211206)** (11 messages🔥): 

> `xAI's Grok, Google AI Edge Repo` 


- **Grok 昂贵的 X 平台集成**：一位成员链接到了 [Grok 的状态更新](https://x.com/grok/status/1928906427277701214)，并质疑为什么 **xAI** 会支付如此高昂的费用留在 **X** 平台上。
   - 他们质疑 **X** 是否期望获得足够多的付费用户来弥补 **3 亿美元 + 50% 收入分成** 的成本，并链接了一篇有争议的文章 [OpenAI Incel Chatbot Subhuman Men](https://www.citationneeded.news/openai-incel-chatbot-subhuman-men/)。
- **Google AI Edge Gallery 亮相！**：一位成员分享了 [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery)，并建议 **Google** 将其发布在 **F-Droid** 上。
   - 另一位成员询问该仓库是否为官方发布，第一位成员确认了其官方身份。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1378448555861348432)** (28 messages🔥): 

> `AgentX 提交, 技术附录提交, 证书声明表, Trailblazer 证书, 下一期 MOOC 日期` 


- **AgentX 提交允许视频演示**：一位参与者询问在没有实时产品链接的情况下，[工作原型的视频](https://www.example.com/hypothetical-link)是否足以用于 AgentX 创业赛道的提交。
   - 助手澄清说，**预期需要一个实时演示链接**，产品演示视频是另一项要求，但建议先提交现有的内容。
- **技术附录提交已解决**：一位用户询问关于在提交表单中附加 **5 页技术附录**的问题。
   - 助手为附录添加了一个字段，并指出即使附录提交在不同位置，仍会被接受。
- **证书表单中多个团队邮箱的说明**：一位参与者询问在加入多个团队时，证书声明表中的多个团队邮箱是否应以逗号分隔。
   - 官方澄清，只需包含**其中一个团队的主邮箱**即可。
- **Trailblazer 证书标准说明**：参与者询问了 **Trailblazer 证书**的标准，特别是关于文章、X 帖子以及证书发放的时间线。
   - 助手确认，完成测验、提交书面文章并在 X 上发布帖子即符合资格；证书将在未来几周内发放，但 [Ninja + Legendary 证书需要更长时间](https://www.example.com/hypothetical-link)。
- **证书声明表确认问题排查**：一位用户报告收到了某些提交的确认邮件，但没有收到证书声明表的确认。
   - 助手建议，看到**浏览器中的确认屏幕**通常就足够了，但作为预防措施，会保持表单开放以便重新提交，并指出该表单只能提交一次。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1378518993484382308)** (12 messages🔥): 

> `AI 生成的 CUDA-C 内核超越 PyTorch, tinygrad 会议 #73, 7900XTX 上的未签名固件, 多主机更改 & p2p 传输` 


- **CUDA-C 内核碾压 PyTorch**：一个 [PR](https://github.com/tinygrad/tinygrad/pull/10586) 强调，tinygrad 中 **AI 生成的 CUDA-C 内核**在不使用 CUTLASS 和 Triton 等库的情况下，性能已接近甚至超越了 **PyTorch** 中专家优化的生产级内核。
   - 相对于 PyTorch，这些内核在 **Matmul (FP32)** 中达到了 **101.3% 的性能**，在 **Conv2D** 中达到 **179.9%**，在 **Softmax** 中达到 **111.8%**，在 **LayerNorm** 中达到 **484.4%**，在 **Conv2D + ReLU + MaxPool** 中达到 **290.1%**。
- **tinygrad 将讨论未来方向**：第 73 次会议定于**圣地亚哥时间周一上午 9 点**举行，讨论事项包括公司更新、**MLPerf**、**基准测试 CI 任务**、**调度器**、**驱动程序**、**云哈希**、**ONNX**、**WebGPU**、**符号化 Z3** 以及其他悬赏任务（bounties）。
   - 议程将包括针对 **lm_eval**、**AMD_LLVM** 和**云相关**任务的悬赏。
- **逆向工程 AMD GPU**：一位用户询问了使用自定义驱动程序在 **7900XTX** GPU 上运行未签名固件的进展和尝试过的方法，寻求已尝试方案的列表。
   - 共享了一个指向关于[未签名固件](https://discord.com/channels/1068976834382925865/1318495874699231302/1360095679396974706)相关讨论的链接。
- **云集成接近完成**：一位成员报告称，他们一直忙于现实生活中的事务，并提交了**多主机更改**，完善了 **p2p 传输**，预计今天或明天完成云端相关工作（速度优化除外）。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1378769747084181545)** (5 messages): 

> `UOp 类, UOp 树, Ops.UNIQUE` 


- **解码 UOp 树中的 'args'**：一位成员询问了 **UOp 类**中 'args' 的含义和文档，以及它们在 **UOp 树**中是如何使用的。
   - 另一位成员澄清说*没有文档*，其含义取决于 **Op 类型**，但通常它是一个标记，用于显示 **buffer uop 子节点**是创建的第几个 buffer uop。
- **Ops.UNIQUE 解释**：一位成员询问 `UOp(Ops.UNIQUE, dtypes.void, arg=19, src=())` 中 'arg=19' 的含义。
   - 解释称，对于 **Ops.BUFFER UOps**，*arg* 赋予它们唯一身份，以便在查看图表时能够区分哪个 buf 是哪个。


  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1378493770479173775)** (13 messages🔥): 

> `GPT4All Extension, Intel Compute, AI models, Model Context Protocol` 


- **扩展 GPT4All 以实现安全的 PC 交互**：一位社区成员正在开发一个项目，旨在扩展 **GPT4All 的 Mistral-Instruct 模型**，通过一个安全的执行层实现与本地 PC 的安全、受控交互。
   - 该成员正在寻求关于确定集成点、推荐安全解释模型输出的最佳实践以及潜在合作方面的帮助，并计划在获得 **GPT4All 团队**许可后将该扩展作为开源工具发布。
- **GPT4All 对 Intel 计算的支持**：一位成员询问 **GPT4All** 是否将支持 **Intel compute**。
   - 他们提到其 **12GB B580** 显卡已准备就绪。
- **寻找具备科学知识的 AI 模型**：一位成员正在寻找了解各领域科学知识现状的 **AI 模型**，以便准确回答有关医学、生物学、行为学、哲学、逻辑和伦理的问题。
   - 一个建议是使用 **LocalDocs 进行 RAG**，将大量相关医学和生物学教科书导入 LocalDocs，并配合 2024 年底或更新的模型使用。
- **Model Context Protocol 与 Tool-Calling 能力**：一位成员建议研究 **Model Context Protocol (MCP)** 或 **llama.cpp 的 tool-calling 能力**，以此为基础构建 GPT4All 项目。
   - 他们指出 Nomic 开发者在最近几个月内未回应咨询，这表明 PR 审查和合并存在不确定性。
- **HighNoonLLM 发布**：发布了 **HighNoonLLM** [GitHub](https://github.com/versoindustries/HighNoonLLM) 的链接。
   - 未提供更多细节。


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1379168810824634511)** (1 messages): 

> `Azure AI Inference SDK, Cohere input types` 


- **Azure AI Inference SDK 将不支持 Cohere 输入类型**：Azure 表示，在使用 [Azure AI Inference SDK](https://github.com/Azure/azure-sdk-for-python/issues/41001#issuecomment-2931978119) 调用嵌入模型时，将**不支持 Cohere 输入类型**。
   - 用户可能会回应并询问是否可以在 **Azure AI foundry** 或其文档中添加警告，因为他们认为这是一个非常隐蔽的问题，其他人也可能会遇到。
- **模型测试中 Cohere SDK 与 Azure 的对比**：虽然可以使用 **Cohere SDK**，但用户更倾向于使用 Azure，因为他们正在测试来自不同供应商的多个模型。
   - 这使得在统一环境中管理和测试各种模型变得更加容易。


  

---


### **Cohere ▷ #[💡-projects](https://discord.com/channels/954421988141711382/1218409701339828245/1378740162460123237)** (1 messages): 

> `Cohere Spanish Recipes DPO Dataset, New Open Source Projects` 


- **Cohere 西班牙语食谱 HuggingFace 数据集亮相**：一位成员分享了一个使用 **Direct Preference Optimization (DPO)** 方法构建的 [HuggingFace 数据集](https://huggingface.co/datasets/somosnlp-hackathon-2025/gastronomia-hispana-dpo)，用于 **Cohere Spanish Recipes**。
- **令人兴奋的新开源项目启动**：热情的成员们开始宣布新的**开源项目**启动，并正在寻找贡献者。


  

---


### **Cohere ▷ #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1379006838485422252)** (2 messages): 

> `Agentic Frameworks, LLM Grounding` 


- **Elie 从日本加入！**：来自卢旺达、目前在日本 **Araya Inc** 工作的 Elie Magambo 正在学习 **Agentic 框架**并探索可用工具。
   - Elie 正专注于使用个性化内容对 **LLM 进行 Grounding**。
- **社区 Discord 服务器欢迎新成员**：Cohere 社区 Discord 服务器欢迎新成员并鼓励他们进行自我介绍。
   - 新成员被要求分享其公司/行业/大学、正在研究的内容、喜爱的技术/工具以及希望从社区中获得什么。