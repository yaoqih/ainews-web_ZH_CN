---
companies:
- openai
- openrouter
- alibaba
- unslothai
- cohere
- huggingface
- black-forest-labs
- diffusers
- ostrisai
- zhipu-ai
- together-ai
- mistral-ai
date: '2025-07-31T05:44:39.731046Z'
description: '以下是该文本的中文翻译：


  **OpenAI** 的神秘模型 **horizon-alpha** 在 **OpenRouter** 上亮相，引发了关于其为 **GPT-5** 前身的猜测。该模型展示了强大的推理和
  SVG 生成能力，性能可与 **Gemini 2.5 Pro** 媲美。**阿里巴巴**发布了 **Qwen3-Coder** 系列，其中包括快速的 **Qwen3-Coder-Flash
  (30B-A3B)** 变体，具备智能体（agentic）功能，并可通过 **UnslothAI** 支持 100 万（1M）上下文长度。**Cohere**
  推出了 **Command A Vision**，这是一个拥有 1110 亿（111B）参数的开源权重视觉语言模型，在企业级基准测试中表现优于 **GPT-4.1**
  和 **Llama 4 Maverick**。**Black Forest Labs** 推出了 **FLUX.1 Krea [dev]**，这是一款开源权重的写实模型，兼容
  **diffusers** 和 **ostrisai** 等微调工具。**智谱 AI (Zhipu AI)** 发布了 **GLM-4.5**，这是一款具有智能体能力的混合推理开源模型，已在
  **Together AI** 上线。相关讨论强调了**推理时训练 (inference-time training)** 和**推理模型泛化 (reasoning
  model generalization)** 日益增长的重要性。**Mistral AI** 发布了 **Voxtral** 的技术报告，继续其在开放科学领域的探索。'
id: MjAyNS0w
models:
- horizon-alpha
- gpt-5
- gemini-2.5-pro
- qwen3-coder
- qwen3-coder-flash-30b-a3b
- command-a-vision
- gpt-4.1
- llama-4-maverick
- flux-1-krea-dev
- glm-4.5
- voxtral
people:
- scaling01
- teortaxestex
- huybery
- nickfrosst
- aidangomez
- reach_vb
- zai_org
- corbtt
- jxmnop
- teknuim1
title: Figma 估值超过 500 亿美元的 IPO（首次公开募股）
topics:
- reasoning
- svg-generation
- agentic-ai
- context-windows
- vision
- fine-tuning
- inference-time-training
- model-generalization
- open-models
- technical-reports
---



---

# AI Twitter 综述

**模型发布、更新与性能**

- **OpenAI 的 "Horizon-alpha" 引发猜测**：一款名为 **horizon-alpha** 的新型隐身模型已在 **OpenRouter** 上线，引发了巨大轰动，并被广泛[推测为 **OpenAI** 的新模型](https://twitter.com/scaling01/status/1950730582104604964)，可能是 **GPT-5** 的前身或 "nano" 版本。`@scaling01` 的[初步测试](https://twitter.com/scaling01/status/1950730792251891948)表明它在 LisanBench 等基准测试上表现较弱，且不是推理模型。然而，在开启推理模式后的后续测试显示，它[能够轻松完成 20 位数乘法](https://twitter.com/scaling01/status/1950949288521281820)，思考过程[长得离谱](https://twitter.com/scaling01/status/1951048818897613069)，并且在 [LisanBench](https://twitter.com/scaling01/status/1951068773869305999) 上的表现与 **Gemini 2.5 Pro** 持平甚至更好。该模型还展示了强大且独特的 [SVG 生成能力](https://twitter.com/scaling01/status/1950847124146704780)。`@teortaxesTex` 指出它似乎擅长处理涉及“魔力”和“难以言表的灵魂”的任务，[闻起来像是一个 Sonnet 杀手](https://twitter.com/teortaxesTex/status/1951058712220549340)。
- **Qwen3-Coder 系列发布**：`@huybery` 宣布发布 **Qwen3-Coder**，这是来自**阿里巴巴**的仓库级代码模型，已在 **OpenRouter** 等平台获得社区的大量采用和使用。一个更小、更快的版本 **Qwen3-Coder-Flash (30B-A3B)** 也已[面向本地用户发布](https://twitter.com/huybery/status/1950925963979796877)，提供基础的 Agent 能力。该模型目前已在 [LM Studio 上线](https://twitter.com/lmstudio/status/1950942293726503174)，并可通过 **UnslothAI** 以 **1M 上下文长度**运行。
- **Cohere 发布 "Command A Vision" VLM**：**Cohere** 凭借 **Command A Vision** 进入视觉领域，这是一款全新的[最先进的 111B 参数开放权重视觉语言模型 (VLM)](https://twitter.com/JayAlammar/status/1950931480349143259)。正如 `@nickfrosst` 所宣布的，模型权重已在 **Hugging Face** 上提供，且在企业级基准测试中[超越了 GPT-4.1 和 Llama 4 Maverick 等模型](https://twitter.com/aidangomez/status/1950927454383616343)。
- **用于摄影写实主义的 FLUX.1 Krea [dev]**：**Black Forest Labs** 发布了 **FLUX.1 Krea [dev]**，这是一款专门为摄影写实主义打造的[最先进的开放权重 FLUX 模型](https://twitter.com/multimodalart/status/1950923544998658557)。`@reach_vb` 强调它可以在 [ZeroGPU 上免费运行](https://twitter.com/reach_vb/status/1950948423986708525)，且开发者指出，现有的微调工具（如来自 **diffusers** 和 **ostrisai** 的工具）[应该可以开箱即用](https://twitter.com/multimodalart/status/1950932021817020867)。
- **智谱 AI 推出 GLM-4.5**：`@Zai_org` 宣布推出 **GLM-4.5**，这是一款[统一了 Agent 能力](https://twitter.com/Zai_org/status/1950899064398364951)的新型开放模型。它被描述为一种混合推理模型，可以在“思考”和“即时”模式之间切换，[目前已在 Together AI 上线](https://twitter.com/Zai_org/status/1950750962483675536)。
- **推理时训练与推理泛化**：`@corbtt` 感觉到 **推理时训练 (Inference-time training)** [很快将成为一件大事](https://twitter.com/corbtt/status/1950705924684873988)。另外，`@jxmnop` 询问了关于 **推理模型泛化** 的例子，例如在数学问题上训练的模型在创意写作方面变得更好。`@Teknium1` 认为，模型会学习做任何能提高其思考过程中准确性的事情，[包括产生幻觉](https://twitter.com/Teknium1/status/1950865106725744913)，并引用了 null shot learning 论文。
- **Mistral 发布 Voxtral 技术报告**：为了持续致力于开放科学，**Mistral AI** 已[发布了 Voxtral 的技术报告](https://twitter.com/GuillaumeLample/status/1950855212677075122)。
- **vLLM 现已支持 Step3 VLM**：`@vllm_project` 宣布，**Step3**（一款具有 **MFA & AFD** 的快速且具有成本效益的 VLM）[现已获得全面支持](https://twitter.com/vllm_project/status/1950954138541711802)。`@teortaxesTex` 指出，该模型具有[强大的多模态能力，并采用了与 DeepSeek-V3 不同的自研注意力机制](https://twitter.com/teortaxesTex/status/1951008169989382218)。

**AI 工具、框架与基础设施**

- **LangChain 推出 Deep Agents 和 Align Evals**：来自 **LangChain** 的 `@hwchase17` 解释了 **Deep Agents** 的概念，它结合了规划工具、文件系统、sub-agents 以及详细的 system prompt，并[提供了视频概览](https://twitter.com/hwchase17/status/1950989844936794511)。该团队还发布了 **Align Evals**，其灵感源自 `@eugeneyan` 的工作，旨在让[构建和对齐 LLM-evaluators](https://twitter.com/Hacubu/status/1950741838396027168) 变得更加容易。
- **基础设施与部署进展**：**Microsoft** 和 **OpenAI** 宣布了 **Stargate Norway**，这是一项新的数据中心计划。`@modal_labs` 推出了 **GPU snapshotting**，实现了 [vLLM 的 5 秒冷启动](https://twitter.com/akshat_b/status/1950967605121962164)，`@sarahcat21` 称这一特性为工程壮举。**vLLM** 项目还强调，它将在 [PyTorch Conference 2025 上进行 5 场演讲](https://twitter.com/vllm_project/status/1950821700679192654)。
- **开发者工具融资**：开源代码 Agent **Cline** 宣布已[筹集 3200 万美元的种子轮和 A 轮融资](https://twitter.com/cline/status/1950973599185248304)，《福布斯》也报道了这一消息。`@sama` 称赞了创始人的合作伙伴关系，称他们的故事[非常卓越](https://twitter.com/sama/status/1950936581810041143)。
- **RAG、上下文工程与数据质量**：**Context Rot** 一词被 `@jxmnop` 强调为一个[出色且有用的术语](https://twitter.com/jxmnop/status/1950678527550054848)。**DeepLearningAI** 提供了关于 Transformer 如何[在 RAG 系统中处理增强提示词](https://twitter.com/DeepLearningAI/status/1950979807623139539)的技术解析。`@Teknium1` 指出，[数据集中很大一部分缺失了用户轮次（user turns）](https://twitter.com/Teknium1/status/1950756952125972558)，强调了检查数据质量的必要性。
- **Hugging Face 发布 "Tracks"**：`@_akhaliq` 分享了 **Tracks** 的发布，这是 [Hugging Face 推出的一个 100% 开源的实验追踪库](https://twitter.com/_akhaliq/status/1950617338136383605)，定位为付费服务的替代方案。

**AI 生成媒体与内容**

- **Runway Aleph 正式发布**：`@c_valenzuelab` 宣布 **Runway Aleph** 已向所有付费计划全面推出，并将其描述为一种[全新的 AI 创作方式](https://twitter.com/c_valenzuelab/status/1950920825185402986)。一段演示展示了它在[保持角色一致性](https://twitter.com/c_valenzuelab/status/1951002926555734337)的同时，处理复杂环境变化的能力。此次发布是 Runway 在 2025 年一系列快速更新的一部分。
- **Google 发布 Veo 3 Fast 及新功能**：**Google DeepMind** 宣布，更快速、更具成本效益的文本转视频模型 **Veo 3 Fast**，以及 [Veo 3 的新图生视频功能](https://twitter.com/GoogleDeepMind/status/1950960418286940312)现已在 Gemini API 中上线。
- **Midjourney 的 "Midjourney TV" 实验**：`@DavidSHolz` 将新的 **Midjourney TV** 实验描述为[有种奇妙的催眠感](https://twitter.com/DavidSHolz/status/1950692691005657415)。该功能提供了由社区生成的趋势视频的实时直播流。
- **Amazon 支持 "Showrunner"，AI 界的 Netflix**：据报道，**Amazon** 正在投资 **Showrunner**，这是一家 AI 生成的流媒体服务，[允许用户根据提示词生成场景](https://twitter.com/TomLikesRobots/status/1950647978118488072)。该平台由 **Fable Simulation** 开发，该公司曾发起过《南方公园》AI 实验。

**行业、融资与地缘政治**

- **中美 AI 竞赛**：`@AndrewYNg` 发布了一段详细的推文，认为现在[**中国**有路径在 AI 领域超越美国](https://twitter.com/AndrewYNg/status/1950941108000964654)，理由是其充满活力的权重开放模型（open-weights model）生态系统以及在半导体领域的积极举措。他指出，虽然顶尖的私有模型来自美国，但顶尖的开放模型通常来自中国。`@carlothinks` 对此表示赞同，并引用了一位前阿里巴巴 CTO 的话：“[中国正在构建 AI 的未来，而不是硅谷。](https://twitter.com/glennko/status/1950642750916792580)”
- **Figma 上市**：**Figma** 正式上市，联合创始人 `@zoink` 表达了[巨大的感激之情](https://twitter.com/saranormous/status/1950952597369577967)。**NYSE**（纽约证券交易所）发布推文称“[Shipped: $FIG](https://twitter.com/saranormous/status/1950952198340325798)”以纪念这一时刻。`@saranormous` 和 `@sama` 也分享了祝贺信息。
- **Meta 的愿景与并购活动**：**Mark Zuckerberg** 分享了 **Meta** 对未来“[为每个人提供个人超智能](https://twitter.com/ylecun/status/1950660512967979245)”的愿景。另外，据 `@steph_palazzolo` 报道，Meta 正在进行并购热潮，已与 [**Pika**、**Higgsfield** 和 **Runway** 等视频 AI 初创公司](https://twitter.com/steph_palazzolo/status/1951001998272372790)进行了洽谈。
- **Perplexity AI 推出 Comet Shortcuts**：`@AravSrinivas` 宣布了 **Perplexity Comet Shortcuts**，允许用户[通过自然语言提示词（prompts）自动化重复的 Web 工作流](https://twitter.com/AravSrinivas/status/1950981234554970382)。一个强大的例子是 `/fact-check` 快捷方式。
- **AI 政策与监管**：据报道，**Google**、**Anthropic**、**OpenAI** 等公司将签署**欧盟 AI 实践准则**。`@DanHendrycks` 澄清说，**xAI** 只签署了[安全部分，而非版权部分](https://twitter.com/DanHendrycks/status/1950831617972519057)。与此同时，`@qtnx_` 注意到全球范围内正在推动[通过身份年龄验证来访问互联网](https://twitter.com/qtnx_/status/1950805548900966777)。

**更广泛的讨论与开发者文化**

- **开发者体验与工匠精神**：`@ID_AA_Carmack` 发布了一篇关于[在不参考先前代码的情况下从零重写 RL Agent](https://twitter.com/ID_AA_Carmack/status/1950621870463873448)价值的反思，该贴访问量极高；他指出，当规模允许这样做时，这是一种幸事。`@ClementDelangue` 向[为开放科学和发布开放模型而奋斗的研究人员](https://twitter.com/ClementDelangue/status/1950927952641749194)表达了衷心的感谢，并承认他们在大型科技公司内部经常面临斗争。
- **对“平台劣化”及过去技术失败的批评**：`@jxmnop` 对软件“平台劣化（enshittification）”提出了反向叙事，认为总体而言，[事物似乎在缓慢且持续地变好](https://twitter.com/jxmnop/status/1950689279908450665)，并引用了手机性能、互联网速度和交通应用的改进作为例子。在另一场讨论中，`@jeremyphoward` 和 `@random_walker` 扩大了对 **DOGE**（Decentralized Organization for the Greater Good）项目的批评，一位评论者称其为[在每一个可能层级上的失败](https://twitter.com/zacharynado/status/1950741189612720310)，既削弱了医学研究，也未能实现其既定目标。
- **斯坦福 NLP 的传承**：**Stanford NLP** 的创始人包揽了 **2025 ACL 时间检验奖（Test of Time awards）**：25 年奖项授予了 Gildea 和 `@jurafsky` 的“语义角色的自动标注（Automatic Labeling of Semantic Roles）”，10 年奖项授予了 `@lmthang`、`@hyhieu226` 和 `@chrmanning` 的“[基于 Attention 的 NMT 有效方法（Effective Approaches to Attention-based NMT）](https://twitter.com/stanfordnlp/status/1950644405821489532)”。

**幽默与迷因**

- **技术荒诞事**：`@RonFilipkowski` 调侃说，每一位醉驾辩护律师都像是[中了大奖](https://twitter.com/code_star/status/1950640352517599615)。`@lauriewired` 指出，银行的 **ACH** 交易本质上只是一个 [940 字节 ASCII 文本文件](https://twitter.com/jeremyphoward/status/1950629141084271053)的 **SFTP** 上传。`@zacharynado` 分享了一条评论，解释澳大利亚火箭发射失败的原因是工程师可能[忘了考虑澳大利亚是倒挂的](https://twitter.com/zacharynado/status/1950964777934610750)。
- **AI 生活**：`@mlpowered` 转发了 `@claudeai` 的简单回复：[“你完全正确。”](https://twitter.com/mlpowered/status/1950685743061647391)。`@typedfemale` 将一段对话比作[在 Tinder 上回复某人](https://twitter.com/typedfemale/status/1950774437881745919)。`@aidan_mclau` 发布了一段[混乱的 Waymo 行程](https://twitter.com/aidan_mclau/status/1950759916945482183)视频。
- **行业评论**：`@nearcyan` 正在[闪回到 23 年](https://twitter.com/nearcyan/status/1950681931416556055)。`@code_star` 评论道，是时候[搬运 10^98 个 Parquet 文件了](https://twitter.com/code_star/status/1950639928242770330)。

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Qwen3-Coder-30B-A3B 和 Flash 模型发布及基准测试

- [**🚀 Qwen3-Coder-Flash 发布！**](https://i.redd.it/p7fpia2bz7gf1.jpeg) ([评分: 1197, 评论: 256](https://www.reddit.com/r/LocalLLaMA/comments/1me31d8/qwen3coderflash_released/))：**该图片宣传了 Qwen3-Coder-Flash，特别是** `Qwen3-Coder-30B-A3B-Instruct` **模型，旨在实现极速且准确的代码生成。它拥有原生** `256K 上下文窗口`**（可通过 YaRN 扩展至** `1M token`**），并针对 Qwen Code、Cline、Roo Code 和 Kilo Code 等平台的集成进行了优化。该帖子强调了函数调用、Agent 工作流支持，并提供了 HuggingFace 和 ModelScope 上的部署资源链接。热门评论讨论了 GGUF 格式模型的可用性（包括 1M 上下文版本和 Unsloth 优化）、模型分片和工具调用的修复，以及活跃的社区开发和 API 访问详情。** 评论者赞扬了生态系统的快速演进和开源性质，关注点在于持续的修复和强大的社区支持。人们对近期模型发布中增强的可访问性和技术改进也充满热情。
    - 此次发布包括 Qwen3-Coder-30B-A3B-Instruct 的动态 Unsloth GGUF，Hugging Face 上提供标准版和 100 万 token 上下文长度版本。修复了 480B 和 30B 模型（特别是“30B 思考”）中的工具调用问题，建议用户因这些更新重新下载第一个分片。Unsloth 还提供了详尽的本地部署设置指南，方便更广泛的用户实验和自定义部署。
    - Qwen-Code 在发布后持续改进，近期修复了多个问题并制定了活跃的维护路线图。对于中国用户，通过 ModelScope API 增强了可访问性，每天提供 2,000 次免费 API 调用，OpenRouter 也提供免费的 Qwen3-Coder API，扩大了该模型的访问和实验范围。主要的 Qwen-Code 仓库地址仍为 https://github.com/QwenLM/qwen-code，社区参与和补丁更新非常活跃。
- [**Qwen3-Coder-30B-A3B 发布！**](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) ([评分: 433, 评论: 83](https://www.reddit.com/r/LocalLLaMA/comments/1me2zc6/qwen3coder30ba3b_released/))：**Qwen3-Coder-30B-A3B 是一款针对 Agent 编码应用（如 Qwen Code 和 Cline）优化的 LLM。值得注意的是，该模型省略了“思考 token”，这表明其设计主要侧重于直接代码生成，而非逐步推理，这可能会影响某些 Agent 任务的追踪或可解释性。** 评论者注意到了思考 token 的缺失，并推测该模型将很好地集成到 Roo Code 等 Agent 使用场景中。人们对用于简化部署的 GGUF（量化）版本的可用性也表现出兴趣。
    - 讨论指出，尽管该模型缺乏显式的 Fill-In-the-Middle (FIM) 支持，但用户报告 FIM 功能依然存在，尽管不如 Qwen2.5-Coder-7B/14B 那样强大。这表明其具有部分 FIM 兼容性，可能会根据重度代码填充或 Agent 编码任务的需求影响工作流。
    - 据悉，该模型专为 Agent 编码用例（如 Qwen Code、Cline）设计，意味着它针对多步推理或工具使用场景进行了定向优化，这可能使其在实际编码效用上区别于通用模型。

- [**我制作了一张 Qwen3-Coder-30B-A3B 与 Qwen3-Coder-480B-A35B 的对比图表**](https://i.redd.it/l6547uel88gf1.png) ([Score: 207, Comments: 16](https://www.reddit.com/r/LocalLLaMA/comments/1me4i2h/i_made_a_comparison_chart_for_qwen3coder30ba3b_vs/)): **该图片是一张雷达图，对比了 Qwen3-Coder-30B-A3B (Flash) 和 Qwen3-Coder-480B-A35B 之间的多项技术基准测试。结果显示，在 Agent 能力测试（“mind2web” 和 “BFCL-v3”）中，两款模型的表现相似，表明在这些任务上具有对等性。然而，在以编程为重点的评估（Aider-Polyglot 和 SWE Multilingual）中存在显著的性能差距，480B 版本优于 30B。这些见解表明，虽然不同尺寸的模型在 Agent/决策任务上具有竞争力，但纯代码能力会随着模型尺寸的增大而显著提升。[查看图片](https://i.redd.it/l6547uel88gf1.png)** 评论者讨论了 Dense 版 Qwen3 32B 模型可能会缩小在代码基准测试中看到的差距，并表示有兴趣与 GPT-4.1 或 o4-mini 进行对比，以使这些结果更具参考价值。
    - 多位用户请求在对比中加入 Dense 版 Qwen3 32B 模型，并指出虽然它并非严格的代码专用模型，但在代码任务中表现非常出色。这表明人们有兴趣了解在 Qwen3 家族中，Dense 架构与 Mixture-of-Experts (MoE) 方法的对比情况。
    - 一位用户提供了 Qwen3-Coder-30B-A3B 的实际性能指标，观察到它在 Apple M4 Max 硬件上达到了约 `90 tokens/second`。他们认为，考虑到为了获得相对有限的性能提升而大幅增加的参数量（16倍），30B 模型的速度和较低的硬件要求使其比大得多的 480B 版本更具吸引力。
    - 有人请求与闭源模型进行对比基准测试，特别是 OpenAI 的 GPT-4.1 和 o4-mini，这表明用户希望使用类似的数据集或任务进行跨家族基准测试，以更好地了解开源模型相对于行业领先者的地位。

### 2. 中国开源 AI 模型的势头与全球排名

- [**难以置信：中国主导了 HuggingFace 前 10 名开源模型**](https://www.reddit.com/r/LocalLLaMA/comments/1mdsjn2/unbelievable_china_dominates_top_10_opensource/) ([Score: 756, Comments: 135](https://www.reddit.com/r/LocalLLaMA/comments/1mdsjn2/unbelievable_china_dominates_top_10_opensource/)): **7 月份 HuggingFace 上中国开源 AI 模型的发布量激增，Kimi-K2、Qwen3、GLM-4.5、腾讯的 HunyuanWorld 和阿里巴巴的 Wan 2.2 等模型占据了该平台趋势榜单的主导地位。该帖子将其与 Meta 最近宣布转向更闭源策略的声明进行了对比，突显了中西方 AI 生态系统在开放性上的逆转，中国模型目前在 HuggingFace 的开源势头上处于领先地位（参见 Hugging Face 趋势模型）。** 热门评论辩论了西方最近的贡献，特别提到只有 Mistral 是一个重要的模型，并暗示了一个悖论：由于竞争动态的变化和战略性开放，中国目前在 AI 开发方面比西方更开放。
    - 几位评论者强调，西方最近主要的开源模型贡献被认为是有限的，仅点名提到了 Mistral，且其在 HuggingFace 排行榜上的排名并不总是稳居前列。这突显了一种观点，即与中国目前的势头相比，西方的开源进展正在停滞。
    - 围绕 Meta (Facebook) 和其他科技巨头的策略展开了讨论，批评计划中的顶级模型（例如来自 Meta 的模型）可能会被限制仅供内部使用，而不是公开发布，并与亚马逊历史上对待专有创新的方式进行了负面对比。这一趋势被视为正在背离开源原则，转而支持公司内部部署，进一步减少了公众获取前沿 AI 技术的机会。

- [**中国模型正在拉开差距**](https://i.redd.it/727keqreo3gf1.png) ([Score: 1121, Comments: 133](https://www.reddit.com/r/LocalLLaMA/comments/1mdmsu9/chinese_models_pulling_away/)): **该帖子讨论了中国语言模型与西方产品相比取得的快速进展和性能提升，特别强调了 Qwen3-30B-A3B 等模型。图片（描述中未提供，但从评论和上下文中推断）可能展示了基准测试或对比图表，显示了在本地 Large Language Model (LLM) 部署中，中国模型如何超越了广泛使用的西方模型（如 LLaMA 和 Mistral）。讨论还提到了由于更好的性能或更少的审查，用户正从基于 LLaMA 的模型迁移到更新的中国开发替代方案。** 评论者们争论转向中国模型是否意味着放弃像 r/LocalLLaMA 这样的社区，其中一位强调 Mistral 模型仍然受到极大关注，突显了基于使用场景和社区参与度的 LLM 偏好持续多样化。
    - 一位用户概述了他们使用各种本地大语言模型的历程：从 LLaMA 3.1-3.2 开始，转向 Mistral 3 Small 及其变体（特别是通过 R1 蒸馏得到的去审查版本 Dolphin），最终采用了 Qwen3-30B-A3B 模型。这一序列突显了随着 Qwen3-30B-A3B 等中国模型因其能力和微调选项而获得关注，用户正在快速切换。
    - 讨论指出 Mistral 在 r/LocalLLaMA 中仍然很受欢迎，反驳了用户完全放弃非中国模型的说法。Mistral 活跃的社区参与和模型更新使其在本地化语言任务中保持相关性。
    - 一条技术评论提到 Mistral 在一个月内发布了多个小模型，并期待即将到来的 Mistral Large 更新的影响，表明其在面对新兴中国模型时持续发展并保持竞争地位。
- [**r/LocalLLama 的每个人今天每 5 分钟刷新一次 Hugging Face，寻找 GLM-4.5 GGUF**](https://i.redd.it/f5iqhqp7z6gf1.jpeg) ([Score: 343, Comments: 71](https://www.reddit.com/r/LocalLLaMA/comments/1mdykfn/everyone_from_rlocalllama_refreshing_hugging_face/)): **这张图片是一个模因（meme），讽刺了 r/LocalLLaMA 社区对 Hugging Face 上发布 GLM-4.5 GGUF 文件的期待，技术用户正等待其可用于本地推理。评论者澄清说，GLM-4.5 的 GGUF 转换仍在 llama.cpp 中进行调试（参见草案 PR [#14939](https://github.com/ggml-org/llama.cpp/pull/14939)），目前的上传版本并不可靠。建议有兴趣尝试 GLM-4.5 的用户在 GGUF 支持最终确定前，先尝试用于 MLX 工作流的 [mlx-community/GLM-4.5-Air-4bit](https://huggingface.co/mlx-community/GLM-4.5-Air-4bit) 版本。** 讨论强调了 GLM-4.5 缺乏稳定的 GGUF 转换，以及临时使用 MLX 等替代后端的情况，一些用户优先考虑其他模型（例如 Qwen3-Coder-30B-A3B-Instruct）。
    - llama.cpp 对 GLM-4.5 GGUF 的支持仍在开发中，主要的 Pull Request ([github.com/ggml-org/llama.cpp/pull/14939](https://github.com/ggml-org/llama.cpp/pull/14939)) 仍处于草案状态。目前的 GLM-4.5 GGUF 模型可能存在转换问题，不被视为稳定版本；在实现最终确定之前不应使用。
    - 对于能够运行 MLX 模型（例如通过 LMStudio）的用户，MLX 社区已经提供了一个可运行的 4-bit 量化版 GLM-4.5 Air ([huggingface.co/mlx-community/GLM-4.5-Air-4bit](https://huggingface.co/mlx-community/GLM-4.5-Air-4bit))，该版本在社区测试的 Agent 编码任务中表现良好。
    - 使用 Unsloth 的 llama.cpp 分支时，Unsloth GGUF 得到最好的支持，因为它包含与其量化和 GGUF 实现相匹配的定制代码，提高了兼容性并可能减少转换问题。

### 3. 即将到来且具有潜力的 Benchmark 创新：Deepseek ACL 2025

- [**Deepseek 凭借在长上下文领域的突破性创新荣获 ACL 2025 最佳论文奖，采用该技术的模型可能很快面世**](https://arxiv.org/abs/2502.11089) ([Score: 506, Comments: 35](https://www.reddit.com/r/LocalLLaMA/comments/1mdn6dp/deepseek_just_won_the_best_paper_award_at_acl/)): **Deepseek 最近因一种处理长上下文的新颖方法获得了 ACL 2025 最佳论文奖，该方法可能以稀疏注意力（sparse attention）机制为核心，旨在提高 Transformer 架构的可扩展性和效率。这一创新可能使较小的语言模型能够维持更长且更有效的上下文窗口，解决现有模型在追踪长程依赖（long dependencies）方面的已知局限（背景请参阅 [Deepseek 的稀疏注意力研究](https://arxiv.org/abs/2402.04038)）。** 评论者强调，这项工作展示了 Deepseek 除被指控“克隆”之外的真正创新，多位评论者强调稀疏注意力是一项重大优化，可能会影响未来 LLM 的可扩展性和上下文保留，特别是对于较小模型而言。
    - 稀疏注意力被强调为长上下文模型的一项主要优化策略，评论者指出，与标准的稠密注意力（dense attention）方法相比，它具有大幅提高效率和规模的潜力。这被视为 Deepseek 创新等近期进展背后的关键驱动力。
    - 这一突破有望帮助较小模型在输入长度增加时更好地保留上下文，直接解决了当前架构中上下文保留通常随长度扩展而退化的弱点。这对内存使用和性能扩展具有技术意义。
    - 有人推测 Deepseek 的进步是否能使其模型的性能达到 Gemini 等前沿系统的水平，特别是在 fiction.livebench 等专业评估基准上。此类性能比较被视为当前 LLM 领域的一项关键技术基准。
- [**据报道，AMD 正寻求推出专用独立 NPU，类似于游戏 GPU，但针对 PC 上的 AI 性能；将 Edge AI 提升到新水平**](https://wccftech.com/amd-is-looking-toward-introducing-a-dedicated-discrete-npu-similar-to-gaming-gpus/) ([Score: 273, Comments: 48](https://www.reddit.com/r/LocalLLaMA/comments/1mdx65u/amd_is_reportedly_looking_to_introduce_a/)): **据报道，AMD 正在探索为 PC 开发专用的独立 NPU（Neural Processing Unit），旨在作为独立的 PCIe 卡提供高 AI 性能，与游戏 GPU 区分开来。这种方法可以为 AI 工作负载提供更高的显存容量（可能为 64-1024GB VRAM），并从传统 GPU 中卸载推理/LLM 任务，遵循了类似于 Qualcomm 的 Cloud AI 100 Ultra 等产品的方向。AMD 目前的消费级 AI 堆栈（如 Strix Point APU 和 XDNA 引擎）已经支持用于 Edge AI 的大型模型（例如 128B 参数的 LLM），但这将标志着向更广泛的消费者/专业级 NPU 部署的转变。[详细信息](https://wccftech.com/amd-is-looking-toward-introducing-a-dedicated-discrete-npu-similar-to-gaming-gpus/)。** 评论强调了专用 AI NPU 在缓解游戏和 AI 的 GPU 瓶颈方面的潜力，同时也对 AMD 的软件成熟度表示怀疑（例如，担心 ROCm 支持能否跟上硬件能力）。
    - 专用 NPU 可以从 GPU 中卸载 AI 任务，通过分离游戏和 AI 工作负载的资源，实现更高的游戏性能（例如，带有 AI 增强 NPC 的高帧率 4K 游戏）。VRAM 的可扩展性（高达 1TB）将使需要在本地运行大型模型或数据集的用户受益。
    - 大家的共识是，强大的驱动程序和 ML 框架支持至关重要；如果没有强大的 ROCm（或同等）软件，无论硬件性能如何，独立 NPU 都会受到限制。ROCm 7.0 被提及为一种潜在的改进，但成熟度仍是一个令人担忧的问题。
    - 讨论突出了市场细分：AMD 可能会占领一个新的专业或专业消费者（prosumer）细分市场，而 NVIDIA 目前的重点并未覆盖这一领域，特别是如果 AMD 能提供具有大内存和极具竞争力的每瓦性能的消费级 NPU，从而绕过在游戏 GPU 中看到的各种人为细分（如 NVIDIA 的数据中心与消费级产品策略）。

- [**我构建了一个 100% 离线运行的 Grammarly 本地替代方案**](https://v.redd.it/pxb4pfgaw8gf1) ([Score: 229, Comments: 61](https://www.reddit.com/r/LocalLLaMA/comments/1me7yia/i_built_a_local_alternative_to_grammarly_that/)): **原作者介绍了 '[refine.sh](http://refine.sh/)', 这是一个利用 Gemma 3n E4B 模型进行离线语法检查的本地 Grammarly 替代方案，峰值内存占用低于 500MB，空闲时为 300MB。该工具处于早期开发阶段，完全离线运行，解决了隐私和本地资源限制问题。** 评论者提供了其他建议，如 FOSS 项目 [WritingTools](https://github.com/theJayTea/WritingTools)，并对该工具不是开源（FOSS）表示担忧。
    - 一位评论者指出，使用 LLM 进行语法纠错通常效果不佳，因为它们难以针对特定语法任务进行 Fine-tuning。他们提到 Grammarly 最近转向 LLM 后端导致了一些问题，暗示基于规则的系统或针对性的 NLP 模型在此用例中可能优于通用 LLM。
    - 提到了其他旨在语法纠错的开源（FOSS）工具。值得注意的是，[WritingTools](https://github.com/theJayTea/WritingTools) 和 [Write with Harper](https://writewithharper.com/) 被提及为免费开源项目，后者强调严格遵守风格指南中记录的语法规则，而不是依赖不受约束的 LLM 输出。
- [**林俊旸（Junyang Lin）正在喝茶**](https://i.redd.it/s3pv80fee7gf1.png) ([Score: 212, Comments: 31](https://www.reddit.com/r/LocalLLaMA/comments/1me095p/junyang_lin_is_drinking_tea/)): **这篇标题为“Junyang Lin is drinking tea”的帖子包含一张图片，在缺乏图像分析和描述中直接技术内容的情况下，需依赖上下文提示。评论提到了快速的 Token 生成速度（“在 30B A3B 上达到 120tok/s”），暗示这可能是一个迷因（Meme），或者是对模型开发者林俊旸及其模型（如 30B A3B，可能是 Llama 或其变体）效率的非正式致敬。帖子本身没有直接的 Benchmark、代码或技术实现。** 评论者表达了热情并强调了性能——特别是每秒 120 个 Token 的输出——暗示对林俊旸近期进展或发布的满意，并突显了社区对高效、强大模型的需求。
    - 一位用户强调使用 30B A3B 模型生成速度达到 `120 tokens/second`，对于一个 30B 参数模型来说，这是极高的推理速度。这指向了高度优化的推理代码或强大的硬件支持。
    - 诗中提到了 GLM 4.5 Air 和 Qwen3 Coder 30B A3B 模型，表明用户正在对多个近期的 LLM 进行 Benchmark。明确提到的 300 亿权重以及 Silicon/GPU 资源暗示了这些模型巨大的计算需求和规模。

### 1. OpenAI GPT-5 与隐身模型开发

- [**Google 现已索引了 GPT-5 的 OpenAI 文档页面，这是其准备发布的又一迹象——目前该页面显示 404**](https://i.redd.it/2mfiosbhf7gf1.png) ([Score: 525, Comments: 103](https://www.reddit.com/r/singularity/comments/1me0e39/google_has_now_indexed_an_openai_docs_page_for/)): **图片显示了一个 Google 搜索结果，索引了一个名为“OpenAI API Docs — GPT-5”的官方 OpenAI 文档页面（URL: https://platform.openai.com/docs/guides/gpt/gpt-5），但实际页面目前返回 404 错误。这一事件表明 OpenAI 正在准备公开更新或发布 GPT-5 的新文档，预示着其正式发布或推出的进展迫在眉睫。该文档在 Google 索引中的出现被解释为 GPT-5 后端准备工作的早期迹象，进一步引发了对其发布时机的猜测。** 评论猜测了 GPT-5 的发布时间，一些用户对即将发布表示怀疑，而另一些人则呼吁保持耐心。目前没有深层的技术争论，大多是对 OpenAI 发布节奏的期待和讨论。

- [**“据传 OpenAI 为 GPT-5 模型使用的代号：'o3-alpha > nectarine (GPT-5) > lobster (mini) > starfish (nano)'。” | “...Zenith、Summit、Lobster、Nectarine、Starfish 和 o3-alpha——据称这些模型的表现优于几乎所有其他已知模型，”已在 LMArena 上被发现。**](https://tech.yahoo.com/ai/articles/gpt-5-launch-might-imminent-151237832.html) ([Score: 163, Comments: 25](https://www.reddit.com/r/singularity/comments/1me4vyf/the_codenames_openai_is_supposedly_using_for_gpt5/)): **泄露的 OpenAI 假设性未来模型的内部代号（'o3-alpha'、'nectarine'、'lobster'、'starfish'）及其推测的模型尺寸（例如，lobster=mini，starfish=nano），据称在 LMArena 基准测试中被观察到，其表现优于大多数其他模型。有推测认为这些代表了 GPT-5 的不同阶段或下一代产品，并有说法称这些模型曾可见，但目前已不再出现在公开排行榜中进行比较。** 评论者质疑文章的可信度和技术准确性；一些人寻求关于 'o3-alpha' 细节的澄清。对于这些模型的存在和性能，以及报道的可靠性，存在怀疑态度。
    - 一位用户指出，提到的模型（Zenith、Summit、Lobster、Nectarine、Starfish 和 o3-alpha）已不再出现在 LMArena 排行榜上，这暗示了模型基准测试的连续性问题或测试条目的移除。这可能会影响当前公开模型性能比较的可靠性。
    - 一位评论者询问 "o3-alpha" 的身份和能力，表明 OpenAI 未发布或实验性模型的内部代号、血统和架构仍存在模糊性，突显了从 o3-alpha 到最终确定的 GPT-5 变体过程中透明度的缺失。
- [**OpenAI 在 OpenRouter 上的新隐身模型**](https://www.reddit.com/gallery/1mdmxpe) ([Score: 185, Comments: 58](https://www.reddit.com/r/singularity/comments/1mdmxpe/openais_new_stealth_model_on_open_router/)): **一个新的、未宣布的 OpenAI 模型出现在 OpenRouter 上（截图：[preview.redd.it/pgmajpmcs3gf1.png](https://preview.redd.it/pgmajpmcs3gf1.png)），引发了关于可能发布 AGI 相关产品的推测。基准测试表明，该模型在数学方面表现不佳，甚至无法解决相对简单的问题，但在编码测试中表现优于其他模型——特别是在处理 edge case（边缘情况）方面，尽管其整体代码质量一般。对比参考指出，Claude 4 Sonnet 在基准测试题目上的表现不如 Claude 3.7，但在实际任务中超过了它，这突显了仅依靠狭隘的基准测试评估模型的局限性。** 评论者讨论了基准测试表现（如数学/编码测试）与实际可用性以及在现实编码中的鲁棒性之间的脱节，几位评论者指出，捕捉 edge case 可能比单纯的基准测试分数更有价值。
    - 一位用户指出，新的 OpenAI 隐身模型在数学任务中表现糟糕，甚至无法解决相当简单的问题，这表明尽管围绕新 AI 模型在标准数学基准测试中获得高分的炒作很多，但这个特定模型在这些领域表现不足。
    - 另一位评论者观察到，虽然该模型在他们典型的编码问题集上提供了最好的结果——特别是在管理 edge case 方面——但一般的代码质量仍然平庸。此外，他们强调，在小型基准测试风格问题上的表现并不一定反映模型在更广泛的现实应用中的价值，并引用了他们的经验：Claude 4 Sonnet 在有限的测试中表现不如 Claude 3.7，但在实际工作场景中表现出色。
    - 一些讨论推测，基于其参差不齐的表现（在不同语境下被描述为既令人印象深刻又有所欠缺），以及生成的游戏与疑似 GPT-5 家族成员的匿名 LMArena 模型输出之间的相似性，这个新模型可能是 "GPT-5 Nano" 的早期形式。这支持了 OpenAI 正在生产环境中悄悄测试下一代小型模型的理论。

- [**OpenRouter 上新的（疑似）OpenAI 隐身模型 Horizon Alpha，第一次尝试就做出了这个**](https://i.redd.it/jci2n61015gf1.png) ([Score: 198, Comments: 51](https://www.reddit.com/r/OpenAI/comments/1mdsala/new_likely_openai_stealth_model_on_openrouter/)): **一名用户测试了 'Horizon Alpha'，这是一个在 OpenRouter 上新出现的据称由 OpenAI 开发的模型。该用户通过提示词让它生成了一个带有像素艺术的详细《马里奥兄弟》游戏副本。生成的图像（[在此查看](https://i.redd.it/jci2n61015gf1.png)）展示了复杂的经典像素艺术游戏元素，包括分数/金币/世界/时间 UI 栏，体现了对原版游戏设计元素极高的还原度。评论集中在复制状态栏 UI 的技术决策上，并推测模型输出在多次运行中是否保持一致（重复使用颜色/字体/UI 主题），或者设计是否会在不同输出之间发生显著变化。** 评论者批判性地质疑了该模型生成完全可玩的、多关卡游戏的能力，讨论了 'Horizon Alpha' 是否可能是一个开源模型，并分析了其 UI 重建的一致性和风格决策，强调了重复生成中潜在的差异。
    - 一位评论者将这个未知的 "Horizon Alpha" 模型与 GPT-4.1 进行了比较，表示在他们的测试中，它的表现不如后者，暗示其在输出质量或能力方面落后于 GPT-4.1。
    - 有一个关于模型处理 UI 元素的观察：具体来说，分数/金币/世界/时间栏的生成方式与像素艺术的其余部分有所不同。评论者怀疑模型可能没有将 UI 元素与背景完全融合，并质疑重新运行提示词是否会产生一致的 UI 风格，如相似的颜色或字体。
    - 一位用户澄清说，该模型是为文本生成而设计的，而不是为了构建完全可玩的关卡或游戏等编码任务，这明确了对该模型生成内容的预期。
- [**OpenAI 的新隐身模型 (horizon-alpha) 一次性编写了整个应用程序！**](https://www.reddit.com/r/OpenAI/comments/1mdrmdm/openais_new_stealth_model_horizonalpha_coded_this/) ([Score: 122, Comments: 44](https://www.reddit.com/r/OpenAI/comments/1mdrmdm/openais_new_stealth_model_horizonalpha_coded_this/)): **该帖子讨论了 OpenAI 尚未发布的模型 'horizon-alpha'，据称它通过 [OpenRouter 的 API](https://openrouter.ai/openrouter/horizon-alpha) 根据单个提示词生成了一个完整的应用程序（附有演示图像链接）。所使用的提示词非常长，可以在[此处](https://gist.github.com/alsamitech/7b7b7b2faf4f5005c91fdba5430a6de1)查看。楼主（OP）指出该模型表现良好，虽然有一些小瑕疵，但与其他模型相比，它在读取和处理大型文件方面速度极快，并展现出强大的错误检测能力。** 热门评论质疑了提示词的恰当性和必要性，认为简单的指令可能同样有效，并建议复杂的 Prompt Engineering 可能并非必要。另一位评论者强调了该模型卓越的速度和快速识别细微错误的能力，称其与当前模型相比是“游戏规则改变者”。
    - 一位评论者指出 'horizon-alpha' 展示了极快的文件读取能力，声称它可以“眨眼间”处理整个文件，这种速度是其他模型无法比拟的。该模型还表现出改进的错误检测能力，能快速发现项目中难以识别的问题，这表明其在速度和代码分析准确性方面都有潜在的进步。

### 2. Wan 2.2 与 Flux：新模型发布与基准测试

- [**Wan 2.2 的流体动力学令人印象深刻**](https://v.redd.it/vzff5xwhu4gf1) ([Score: 292, Comments: 31](https://www.reddit.com/r/StableDiffusion/comments/1mdrld2/wan_22_fluid_dynamics_is_impressive/)): **楼主（OP）展示了使用 Wan 2.2（Image-to-Video，14b 版本）进行的流体/粒子模拟，原始图像由 Flux Dev 生成，音频通过 mmaudio 添加。重点在于评估 Wan 2.2 处理复杂物理现象（如流体和粒子）的能力，指出结果令人印象深刻，但也强调了通过提示词控制摄像机角度和运动方面仍面临挑战。** 一条热门评论提出了一个技术限制：Wan 2.2 倾向于从任何初始液体中产生持续的流体流动，例如，一个静止的泪滴会导致持续的人造流动，据报道这是一个常见的未解决问题。
    - 一位用户描述了 Wan 2.2 流体模拟的一个显著局限：如果存在痕迹（例如眼睛上的泪滴），模型倾向于从同一地点持续生成流体，导致不切实际的瀑布状效果。这表明在模拟流体持久性与初始状态方面存在挑战，突显了模型在时间一致性或识别流体生成何时应停止的阈值设置方面可能存在的问题。

- [**Wan 2.2 Reel**](https://v.redd.it/d2h632ni35gf1) ([Score: 173, Comments: 35](https://www.reddit.com/r/StableDiffusion/comments/1mdsl2s/wan_22_reel/)): **该帖子展示了一个使用 Wan 2.2 GGUFQ5 i2v 模型的演示短片，所有图像均通过 SDXL, Chroma, Flux 或电影截图生成。生成和编辑的总时间约为 12 小时，输出结果展示了相关生成流水线的能力。** 评论中的一个关键技术批评指出了当前 AI 生成视频缺乏“一致性”和“叙事连贯性”，认为下一个技术挑战是制作“可观看的故事”，而不仅仅是视觉效果令人印象深刻的短片。一些技术兴趣也集中在视频生成的细节上（例如使用的分辨率和推理步数）。
    - 讨论强调了 AI 生成视频面临的挑战：特别是缺乏一致性和叙事结构，目前的技术只能产生不连贯的 3-5 秒片段，而不是连贯的、更长的故事。这表明故事和时间连贯性是研究和实施的活跃前沿。
    - 技术评论涉及了不同量化和精度模式的性能：例如在 RTX 5080 上使用 FP8，生成 5 秒 720p 视频大约需要 40 分钟。评论者计划测试 Q4 或使用 Unsloth 的 Dynamic Quant 以获得潜在更快的推理速度，突显了质量与生成速度之间的权衡。
    - 有人对演示视频中使用的分辨率和推理步数（可能是扩散步数）提出了疑问，这对于可复现性以及在不同硬件和量化方法之间比较速度与质量具有重要意义。
- [**Another "WOW - Wan2.2 T2I is great" post with examples**](https://www.reddit.com/gallery/1me5t5u) ([Score: 144, Comments: 34](https://www.reddit.com/r/StableDiffusion/comments/1me5t5u/another_wow_wan22_t2i_is_great_post_with_examples/)): **该帖子讨论了使用 Wan2.2 T2I 模型生成图像，强调生成一张 4K 图像大约需要 1 小时。用户指出，该工作流利用了 CivitAI 原生的 T2I 配置，包含 LightX2V (0.4)、FastWAN (0.4) 和 Smartphone LoRA (1.0)，并观察到采样器和调度器（如 euler）的选择会严重影响色彩饱和度和图像真实感。据报道，该工作流不支持使用 'bong' (res2ly) 进行分辨率缩放，突显了缩放功能的局限性。** 一条评论声称 Wan2.2 在真实感方面超越了 Flux 模型（例如更少的解剖学伪影），但强调缺乏类似于 ControlNet 或 Pulix 的功能来确保跨生成的一致性。另一条评论对由于工作流描述不完整而导致缺乏可复现性表示失望。
    - 一位用户报告称，与 Flux 相比，Wan 2.2 生成的图像更真实，解剖学错误（如缺肢或手指变形）更少，突显了图像保真度和连贯性的提升。然而，他们指出缺乏类似于 ControlNet 或 Pulix 的功能，这些功能可以实现更一致的图像生成和对输出的控制，这表明在引导式或基于参考的生成能力方面存在差距。
    - 有人提出了关于模型要求的技术问题：一位评论者询问令人印象深刻的结果是否需要完整的 Wan 2.2 模型，或者轻量级的 fp8 缩放版本（约 14GB）是否足够，并指出他们在 fp8 变体中观察到了“超级奇怪的结果”，暗示量化/优化版本可能存在局限性或兼容性问题。
- [**PSA: WAN 2.2 does First Frame Last Frame out of the box**](https://www.reddit.com/r/StableDiffusion/comments/1me4306/psa_wan_22_does_first_frame_last_frame_out_of_the/) ([Score: 117, Comments: 19](https://www.reddit.com/r/StableDiffusion/comments/1me4306/psa_wan_22_does_first_frame_last_frame_out_of_the/)): **该帖子宣布 WAN 2.2 模型在 ComfyUI 中实现了“开箱即用”的首帧末帧 (FLF) 视频输出，只需使用新的 2.2 模型和采样器更新现有的 WAN 2.1 FLF2V 工作流即可。提供的 Pastebin 链接包含修改后的工作流定义，突显了对于已经在使用 FLF2V 的用户来说升级非常简便（参见：[Pastebin 工作流](https://pastebin.com/kiG56kGa)）。** 热门评论质疑该模型是否支持真正的视频循环（首帧=末帧）还是会退化为静态图像，并寻求澄清中间节点（如 `LoraLoaderModelOnly`, `TorchCompilerModel`, `Patch Sage Attention`, `ModelSamplingSD3`）的顺序是否会影响输出保真度，因为用户报告了不同顺序下的混合结果。还有关于此工作流在低帧率（如 4fps）视频插帧中实用性的询问。

- 一位用户询问 WAN 2.2 是否可以通过将第一帧和最后一帧设置为相同图像来正确生成循环视频，并询问该模型是否避免了其他视频模型常见的生成静态图像而非无缝循环的问题。
- 关于 WAN 2.2 pipeline 中模型加载器和 KSampler 之间工作流节点顺序影响的技术讨论。比较了两种特定的节点顺序：一种是 LoraLoaderModelOnly 在前，另一种是 TorchCompilerModel 在前。评论者询问这些变化是否会影响采样质量或一致性，并指出在他们自己的测试中结果不一。
- 用户质疑 WAN 2.2 是否适用于插帧任务，例如为低帧率（如 4fps）视频生成中间帧，旨在澄清该模型在这一特定用例中的有效性。
- [**文本转图像对比：FLUX.1 Krea [dev] 对比 Wan2.2-T2V-14B (5 选 1)**](https://www.reddit.com/gallery/1mec2dw) ([评分: 123, 评论: 58](https://www.reddit.com/r/StableDiffusion/comments/1mec2dw/texttoimage_comparison_flux1_krea_dev_vs/)): **一位用户对 FLUX.1 Krea [dev] 和 Wan2.2-T2V-14B 文本转图像生成模型进行了非正式的侧向对比，每个模型测试了 35 个样本，使用长文本提示词（约 150 字）。FLUX.1 Krea 运行 25 步，CFG 从 3.5 降低到 2，而 Wan2.2-T2V-14B 使用了 Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32 LoRA（强度 0.6）以加速推理，这影响了输出的视觉质量。主要发现：Wan2.2-T2V-14B 产生的可用输出（4/5）和自然感显著高于 FLUX，后者经常出现解剖结构错误且风格化不够自然。FLUX 的光照准确度略好，但其对比度高得不自然，且始终无法准确渲染雀斑。** 顶部评论强烈倾向于 Wan2.2-T2V-14B，将其共识简练地总结为 "wan won"，并建议通过提示词微调（例如 '(freckles:6)'）来控制特征。讨论缺乏深层的技术辩论，但表明了在实际使用中可感知的质量差异。
    - 几位用户观察到 FLUX.1 Krea 模型在训练中可能融入了大量 MidJourney 生成的图像，特别是那些具有明显特征（如雀斑）的图像，这引发了对其训练数据与 WAN2.2-T2V-14B 相比的新颖性和原创性的质疑。
    - 技术对比指出，WAN2.2-T2V-14B 生成的图像在视觉上可与电视节目截图相媲美，这表明其具有更高的照片写实感，并且与 Flux.1 Krea 相比，可能拥有更优越的数据集或 Diffusion 架构。一些用户表达了向 WAN 转移的偏好，理由是产品许可方面的差异（例如对 Flux 及其 "bfl non commercial license" 的不满）。
- [**来自 Black Forest Labs 的新 Flux 模型：FLUX.1-Krea-dev**](https://bfl.ai/announcements/flux-1-krea-dev) ([评分: 381, 评论: 250](https://www.reddit.com/r/StableDiffusion/comments/1me2l80/new_flux_model_from_black_forest_labs_flux1kreadev/)): **Black Forest Labs 发布了 FLUX.1-Krea-dev 模型，可在 [Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev) 上获取。它被宣传为原始 flux-dev 的掉入式替换（drop-in replacement），旨在生成更难被识别为合成的 AI 图像，尽管早期用户测试报告称现有的 flux-dev LoRAs 并不兼容。值得注意的是，该模型在正确渲染人类手部方面存在困难，经常生成 4 指或 6 指的图像（见 [输出样本](https://preview.redd.it/ap6yq3fxx7gf1.png?width=642&format=png&auto=webp&s=d27724498982d17cb5fc2d5795b3758efe826825)）。** 评论者怀疑模型中存在严重的内容过滤/审查，一些人对与旧版 LoRA 的兼容性不如宣传的那样表示失望。
    - 讨论强调，虽然 FLUX.1-Krea-dev 被宣传为之前 FLUX dev 模型的掉入式替换（包括与现有 LoRA 的兼容性），但实际用户测试显示，这些旧的 LoRA 在新版本中无法按预期工作。
    - FLUX.1-Krea-dev 发现的一个技术问题是它在准确渲染人类手部方面仍然存在困难，输出有时会产生 4 指或 6 指——这是在不够精细的图像生成模型中常见的伪影。

- [**Flux Krea 相对于常规 Flux Dev 在摄影生成方面表现相当出色**](https://www.reddit.com/gallery/1meann0) ([Score: 145, Comments: 57](https://www.reddit.com/r/StableDiffusion/comments/1meann0/flux_krea_is_quite_good_for_photographic_gens/)): **该帖子展示了来自 Flux 的摄影生成模型 Flux Krea ([Krea.ai](http://krea.ai/)) 的视觉结果，并强调了其与标准 Flux Dev 模型相比在摄影输出真实感方面的提升。虽然没有关于模型架构的明确 Benchmarks 或技术细节，但该帖子侧重于不同生成模型下的定性输出差异。** 热门评论批评其普遍存在的黄色滤镜导致图像显得“毫无生气且冰冷”，建议需要更中性的默认颜色，以便为用户提供更大的后期生成控制权。另一点被提及的是缺乏使用相同 Prompts 的直接侧向对比，这使得对改进进行技术评估变得困难。
    - 几位用户指出 Flux Krea 应用了明显的黄色或冷色滤镜，其中一位表示该模型对摄影外观的尝试导致图像看起来“毫无生气”，并建议他们应该保持色调中性，以实现更多的用户控制。
    - 用户请求进行严格的 Benchmarking，例如使用相同的 Prompts 和设置在 Flux Krea 和常规 Flux Dev 之间进行直接的侧向对比，以准确评估摄影质量的差异。
    - 有人推测该模型的改进可能归功于“更好的 Datasets 和 Captaining”，这表明社区对导致输出产生可观察差异的训练数据或策略的技术细节感兴趣。
- [**FLUX Krea DEV 相比 FLUX Dev 是非常写实的改进 - 本地模型已发布，我在 SwarmUI 中使用常规 FLUX Dev 预设本地测试了 7 个 Prompts**](https://www.reddit.com/gallery/1me4u0a) ([Score: 132, Comments: 53](https://www.reddit.com/r/StableDiffusion/comments/1me4u0a/flux_krea_dev_is_really_realistic_improvement/)): **该帖子将新的 FLUX Krea DEV 模型与之前的 FLUX Dev 进行了比较，强调了照片写实感（Photorealism）的提升，特别是在使用 SwarmUI 生成恐龙图像等任务中。在本地使用常规 FLUX Dev 预设测试了 7 个 Prompts，以评估输出质量。评论中的关键技术查询集中在模型真实感（特别是“写实恐龙”生成）、相对于先前版本的推理速度提升，以及模型大小/VRAM 要求，特别是关于与 RTX 4080 GPU (16GB VRAM) 的兼容性。** 技术讨论围绕新的 Krea DEV 模型是否实质性地加速了推理并生成了超越以往的真实感，尤其是在恐龙等复杂任务中，一些人对当前 AI 在该特定领域的水平表示怀疑。
    - 一位评论者询问了与之前 FLUX Dev 相比的生成速度，特别是询问 FLUX Krea DEV 是否更快，这暗示了社区对这两个本地模型之间的性能改进和推理时间 Benchmarks 的兴趣。
    - 提出了一个关于模型 VRAM 要求和硬件兼容性的技术问题——特别是 FLUX Krea DEV 是否可以在 RTX 4080 (16GB VRAM) 上运行。这反映了对本地部署可行性和模型大小的关注，这对于在消费级 GPU 上本地运行模型的用户至关重要。

### 3. 蒸汽朋克视频游戏概念与提示词技术 (Steampunk Video Game Concepts and Prompt Techniques)

- [**欧洲城市的蒸汽朋克视频游戏（包含提示词）**](https://v.redd.it/u0s6z1avl8gf1) ([Score: 410, Comments: 31](https://www.reddit.com/r/aivideo/comments/1me6d7s/steampunk_video_games_in_european_cities_prompts/)): **发布者分享了详细的 text-to-image 提示词，旨在利用 Prompt Catalyst 生成设定在标志性欧洲城市（巴黎、伦敦、威尼斯）的高保真蒸汽朋克视频游戏概念图。这些提示词指定了摄像机视角（第三/第一人称）、分辨率（`2560x1440`，超宽比例）、游戏内 UI（带有压力表盘、迷你状态栏、冷却时钟界面、小地图、蒸汽计量表的 HUD）以及风格元素（深褐色光效、粒子效果、机械主题等），强调动态环境特征（烟雾、雾气、蒸汽）和写实资产风格（--ar 6:5 --stylize 400 prompt tokens）。生成管线和完整工作流由 [Prompt Catalyst 网站](https://promptcatalyst.ai/tutorials/creating-video-game-concepts-and-assets) 上的外部教程提供支持。** 评论者注意到视觉输出和 UI 设计的高质量，认为生成的概念超出了对蒸汽朋克题材的预期，并引发了与 'The Order: 1886' 的对比（暗示了以往商业实现中错失的潜力）。大家一致认为，如果被行业专业人士采用，这些工具和提示词可能会对实际的游戏开发工作流产生重大影响。
    - 有人提到这些图像中展示的蒸汽朋克美学如何为现有的游戏系列（如 The Order: 1886）提供灵感和批判，暗示在该题材中实现更有效或更具想象力的设计是可能的，尤其是对于设定在欧洲城市的游戏。
    - 一位评论者反复提到 'Bioshock Infinite'，将其作为蒸汽朋克/替代历史视频游戏设计的基准或典范，暗示它仍然代表了该题材中美学与叙事融合的高标准。
- [**欧洲城市的蒸汽朋克视频游戏（包含提示词）**](https://v.redd.it/u0s6z1avl8gf1) ([Score: 410, Comments: 32](https://www.reddit.com/r/aivideo/comments/1me6d7s/steampunk_video_games_in_european_cities_prompts/)): **该帖子详细介绍了用于使用 Prompt Catalyst 生成蒸汽朋克主题视频游戏视觉效果的高度结构化提示词（教程：https://promptcatalyst.ai/tutorials/creating-video-game-concepts-and-assets）。提示词指定了技术参数，例如：第三/第一人称视角、2560x1440 分辨率、21:9 宽高比、带有自定义压力表盘血条的游戏内 UI、作为钟面的技能图标，以及环境效果，包括体积雾、实时粒子效果和深褐色光效，以突出历史感欧洲背景下的黄铜和机械纹理。值得注意的是，动画和资产提示词旨在实现高保真度和风格化（--ar 6:5 --stylize 400）。** 热门评论指出了现有游戏（如 'the order 18xy'）中错失的机会，普遍认可其质量，并希望蒸汽朋克题材能流行起来，但没有深入的技术争论。
    - 评论者讨论了蒸汽朋克美学在 AAA 大作中利用不足的问题，引用 "The Order 1886" 等游戏作为执行力有待提升的遗憾案例。重点在于当前的图形和世界构建能力如何能更有效地实现该题材所需的氛围和游戏深度，尤其是在细节丰富的欧洲城市背景下。
    - 一位评论者强调了诸如 "Bioshock Infinite" 之类的作品为蒸汽朋克主题和沉浸式环境的融合设定了很高的期望，暗示如果该题材的受欢迎程度和投资增加，未来的游戏可能会超越这些基准。

---

# AI Discord Recap

> A summary of Summaries of Summaries by X.ai Grok-4
> 

**Theme 1: Models Muscle Up with New Releases**

- **Qwen3 发布 30B 重磅炸弹**：阿里巴巴的 **Qwen3-30B** 模型在基准测试中与 **GPT-4o** 旗鼓相当，通过 [Unsloth GGUF 版本](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF) 可以在 33GB RAM 上以全精度本地运行，而量化变体仅需 17GB。社区的兴奋点集中在其强大的多语言能力上，尽管在 vllm 等某些配置中 Tool Use 功能表现不佳。
- **Gemma 3 微调解决水印烦恼**：在 16k Context 下微调 **Gemma 3 4B** 可去除水印并提升稳定性，正如[截图](https://cdn.discordapp.com/attachments/1179039861576056922/1399849804045221948/Screenshot_2025-07-29_at_2.13.07_PM.jpeg?ex=688bd0b9&is=688a7f39&hm=82cad388163625496b8cb3e6dd62035b51ae1d16ce0015df29ea910e04c4471f&)所示，这引发了一场通过 [Unsloth 的 X post](https://x.com/UnslothAI/status/1950553466591723640) 发起的、为期 7 天的新竞赛。用户反馈其在流行语言中的翻译能力有所增强，使其成为大型模型的小型替代方案。
- **Arcee 发布 AFM-4.5B 强力模型**：Arcee.ai 发布了 **AFM-4.5B**，采用 Grouped Query Attention 和 ReLU² 激活函数以实现高灵活性，可在 [Hugging Face](https://xcancel.com/LucasAtkins7/status/1950278100874645621) 上获取。未来的变体将专注于 Reasoning 和 Tool Use，并由 DatologyAI 数据合作伙伴提供支持。

**主题 2：为 AI 提速的硬件博弈**

- **Quantization 消除带宽瓶颈**：Quantization 不仅能适配模型，还能大幅降低内存带宽占用并提升计算速度，尽管在 Conv Layers 中保留 **FP32** 的视觉操作会产生瓶颈。用户正在讨论针对消费级硬件的优化，[Unsloth 博客](https://unsloth.ai/blog/dynamic-4bit)中强调了动态 4-bit 方法。
- **AMD Strix Halo APU 价格挤压竞争对手**：**Strix Halo APU** 64GB 版本售价达 1600 美元，128GB 版本达 2000 美元，但正如 [Corsair AI Workstation 帖子](https://www.guru3d.com/story/compact-ai-pc-corsair-ai-workstation-300-with-strix-halo-apu/)中所讨论的，EPYC 系统凭借可升级的 RAM 在性价比上胜出。板载焊接内存因其相比 DIMM 灵活性存在的限制而遭到诟病。
- **P104-100 GPU 廉价交易引发诈骗担忧**：**P104-100** GPU 在 [淘宝](https://item.taobao.com/item.htm?id=897611642586&sku=Video%20memory%20capacity:8gb;Color%20classification:Galaxy%20p104-100%20(card%20length%2028cm);&sku_id=5919375243616) 上仅售 15 英镑，尽管存在 PCIe 1.0 x4 的限制，仍被吹捧为 LLM 推理中相当于 1080 的替代品。跨卡分片有助于提升性价比，但用户警告可能存在 4GB VRAM 访问问题。

**主题 3：模型乱象中的审查冲突**

- **Qwen3 拒绝敏感查询**：在 **Qwen3-30B** 发布于 [Hugging Face](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF) 引发关注的同时，询问有关中国互联网审查的问题会触发聊天机器人立即关闭。这凸显了过度严苛的安全机制限制了实际用途。
- **OpenAI 的审查说教令用户反感**：**OpenAI 模型** 中的严格审查导致了大量机械化回复和道德说教，[Unsloth 的 Llama 4 指南](https://docs.unsloth.ai/basics/llama-4-how-to-run-and-fine-tune)建议避免使用权威性措辞。社区对模型在编程和非客户任务中实用性下降的挫败感日益增加。
- **GLM 4.5 Air 模仿 Gemini 的防护栏**：**GLM 4.5 Air** 感觉像 **Gemini**，但在 vllm 中的 Tool Use 功能存在缺陷，不过根据 [Z.ai 博客](https://z.ai/blog/glm-4.5)的评价，它在聊天和分析方面表现出色。讨论焦点在于如何在不损害功能的前提下平衡安全性。

**主题 4：Agent 装备安全护盾**

- **DeepSecure 锁定 AI Agent 安全**：DeepTrail 的开源项目 **DeepSecure** 通过分片密钥架构和 Macaroons 为 Agent 提供身份验证、授权和策略执行，并提供了 Langchain 示例，如 [安全工作流](https://github.com/DeepTrail/deepsecure/blob/dev/examples/05_langchain_secure_tools.py)。它专为跨模型代理设计，详见[技术概览](https://github.com/DeepTrail/deepsecure/blob/dev/docs/design/deepsecure-technical-overview.md)。
- **MCP 服务器应对上下文泄露**：单实例 **MCP 服务器** 需要用户上下文隔离以防止会话数据共享。根据 [MCP 文档](https://modelcontextprotocol.io/docs/tutorials/use-remote-mcp-server#understanding-remote-mcp-servers)，有用户反馈尽管设置了 SSL，但在 EC2 上部署 Claude Desktop 仍存在问题。Cursor 连接正常，但状态工具在 Windows 上运行失败。
- **Cursor Agent 在并行处理中劫持端口**：**Cursor Background Agents** 意外劫持端口，干扰开发环境；修复方法包括在 VSCode 中禁用自动转发或清空端口数组。并行任务协调使用 API 编排或 Docker，并提出了一个针对依赖项的 [任务队列脚本](https://github.com/example)。

**主题 5：学习小组引领教育爆发**

- **Diffusion Models 学习小组启动**：一个由 12 人组成、为期 5 个月的学习小组通过 [MIT 课程](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf) 研究 Diffusion Models。8 月 2 日（Flow Matching）和 8 月 9 日（PDEs/ODEs）在 [Luma](https://lu.ma/kv8zf6va) 提供免费介绍。该小组面向 AI 专业人士，特色是同行引导的会议和项目。
- **LLM 安全研究资源汇总**：博士生们正在寻找 LLM 安全资源，推荐了关于推理步骤和 chain-of-thought 中信念的 [Alignment Forum 帖子](https://www.alignmentforum.org/posts/iLHe3vLur3NgrFPFy/thought-anchors-which-llm-reasoning-steps-matter)。重点包括偏见缓解和伦理领域的领域自适应。
- **Video Arena 开启机器人对战**：LMArena 实验性的 **Video Arena** 允许用户通过机器人生成 AI 视频并进行投票，[Thijs Simonian](https://www.linkedin.com/in/thijsdev/) 通过 [Google Forms](https://docs.google.com/forms/d/e/1FAIpQLSfS_zh67vgEeftUXzX6ujlNiiFoTQNQd7P7-Nx6aD5zpFVOBg/viewform?usp=dialog) 进行了员工 AMA。它支持免费比较顶级的图像和视频模型。

---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **R1-der R.I.P: 模型从 LLM 选择器中移除**：**R1 1776 模型**已从 LLM 选择器中移除，但仍可通过 [OpenRouter API](https://openrouter.ai/perplexity/r1-1776) 访问。
   - 移除后，用户正考虑切换到 **O3** 或 **Sonnet 4** 进行推理任务。
- **Android 版 Comet 应用即将推出**：**Comet for Android** 应用正在开发中，计划于年底发布。
   - 虽然一位用户质疑该浏览器的潜在功能，但其他人称赞了它在 Windows 上的表现。
- **Gemini 2.5 Pro 提速**：用户报告 **Gemini 2.5 Pro** 速度显著提升，推测其可能使用了 **GPT-4.1**。
   - 性能提升的同时可能伴随着限制，例如推理模型的每日消息上限。
- **Spaces 热潮：自定义指令升温**：成员们讨论了通过添加自定义指令来优化 **Spaces** 功能的使用。
   - 一位用户澄清说，**指令字段提供了更多选项**，例如添加特定网站进行数据检索。
- **Deep Research API 支持 Structured Outputs**：一位正在构建产品的成员表示他们熟悉 **Deep Research 和 Structured Outputs API**。
   - 他们还要求与相关人员讨论 **Enterprise API 定价**、早期访问、速率限制和支持，并申请一些额度以适当地测试和集成 API。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen3-30B 导致聊天机器人关闭**：**Qwen 发布了 Qwen3-30B**，一位用户报告称，当他们在聊天机器人上询问 **Qwen3** 为什么中国要审查互联网时，系统立即关闭了他们的请求。
   - **Qwen3-30B** 的发布令社区感到兴奋，并提供了 [Hugging Face](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF) 链接以供进一步探索。
- **GLM 4.5 Air 模仿 Gemini**：用户讨论了 **GLM 4.5 Air**，有人提到它的感觉像 **Gemini**，并分享了一篇[将其与其他模型对比的博客文章](https://z.ai/blog/glm-4.5)。
   - 成员们注意到 vllm 中的 **tool use 功能已损坏**，但它在聊天、诗歌分析和文档搜索方面表现良好。
- **Quantization 加速计算**：Quantization 不仅仅是为了将模型装入内存，它还能**减少内存带宽**并**显著提高计算速度**。
   - 一位成员指出，将卷积层等 vision head 操作保留在 **FP32** 似乎并非最优，因为它们往往非常慢且成为瓶颈。
- **16k 微调后 Gemma 3 的水印被移除**：在使用 **16k** 上下文微调 **Gemma 3 4B** 后，实验发现 **watermarks** 被完全移除，模型更加稳定（参考附带的[截图](https://cdn.discordapp.com/attachments/1179039861576056922/1399849804045221948/Screenshot_2025-07-29_at_2.13.07_PM.jpeg?ex=688bd0b9&is=688a7f39&hm=82cad388163625496b8cb3e6dd62035b51ae1d16ce0015df29ea910e04c4471f&)）。
   - 一项新的 **Gemma 3 竞赛**已宣布，Notebook 已上线，竞赛将于 **7 天**后结束，更多信息可在 [Xitter](https://x.com/UnslothAI/status/1950553466591723640) 上查看。
- **Unsloth 抨击 OpenAI 的审查制度**：成员们对 **OpenAI 严重的审查制度**表示失望，分享了 ChatGPT 机械化回答和说教的经历。
   - 一位用户指出了 [Unsloth 的 Llama 4 页面](https://docs.unsloth.ai/basics/llama-4-how-to-run-and-fine-tune)，该页面引导用户永远不要使用暗示道德优越感或权威感的短语，并通常避免说教。

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 的 MCP 浏览器自动化即将推出**：成员们正积极通过 **Cursor 的 MCP** 开发依赖浏览器的自动化工具，早期访问版本预计在未来几周内发布。
   - 一位成员强调了设置的简便性，指出其具备 *一键 MCP 设置* 功能，以便直接构建浏览器自动化。
- **并行 Agent 协作难题**：成员们正在努力解决具有依赖关系的并行任务管理问题，因为 Agent 缺乏共享工作区，导致同步触发变得复杂。
   - 提议的解决方案包括 **通过 API 进行外部编排**、**基于文件的协作** 以及 **基于 Docker 的并行执行**，其中包括一个示例 [任务队列脚本](https://github.com/example)。
- **Cursor Background Agents 劫持端口**：工程师们报告了 **Cursor Background Agents** 意外劫持端口的情况，导致他们不得不进行调试以恢复开发环境。
   - 缓解建议包括将 `ports` 属性设置为空数组，或在 VSCode 设置中禁用 *自动转发端口*。
- **Background Agents 被考虑用于研究任务**：一位成员探索了将 Background Agents 用于研究导向型任务，例如 **重构研究** 或 **功能实现研究**。
   - 他们询问了最佳工作流，考虑了让 Agent 在 PR 中起草 Markdown 或直接实施更改等选项。
- **Fish 终端默认设置问题**：一位成员遇到了 **Cursor 集成终端** 默认使用 **fish** shell 的问题，并寻求更改方法。
   - 在临时重命名 fish 二进制文件后，通过设置和包装器修改 shell 的尝试最终获得成功，但根本原因尚不明确。



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Dot-lol 数据收集遭受质疑**：针对 [dot.lol](https://dot.lol) 可能 **出售用户数据** 和画像信息的担忧日益增加，提醒用户不要假设其数据不会被用于定向影响或牟利。
   - 虽然一些人担心广泛数据收集的影响，但另一些人认为 **数据收集是不可避免的**，用户应专注于不将数据与其个人身份关联。
- **GPT-5 传闻：8 月发布？**：传闻暗示 **GPT-5** 可能在 **8 月** 初发布，并在 ChatGPT Mac 应用中发现了路由架构准备工作的潜在证据。
   - 社区成员正在推测其影响以及它是否会超越其他模型，一些人希望提供免费层级。
- **GDPR：有力还是无力？**：成员们辩论了欧盟 **GDPR** 在防止 AI 公司收集数据方面的有效性，对其影响持不同意见。
   - 一些人认为 **GDPR** 主要影响数据的 *使用* 而非 *收集*，而另一些人则反驳称 *欧盟消费者的数据收集已被关闭*。
- **Zenith 重返 Arena**：随着 **Zenith** 模型重返 **LMArena**，人们对其潜在的 **ELO** 评分和整体性能充满期待。
   - 虽然一些人对错过试用机会感到遗憾，但另一些人对其在平台上的价值持有强烈看法。
- **灯光、摄像、AI：Video Arena 启动**：**LMArena** 团队在 Discord 上推出了实验性的 **Video Arena**，用户可以使用 LMArena 机器人免费生成并比较顶级 AI 模型的视频，该机器人支持 **生成视频、图像和图生视频**。
   - 宣布将与机器人开发者 [Thijs Simonian](https://www.linkedin.com/in/thijsdev/) 进行 **员工 AMA**，邀请用户通过 [Google Forms](https://docs.google.com/forms/d/e/1FAIpQLSfS_zh67vgEeftUXzX6ujlNiiFoTQNQd7P7-Nx6aD5zpFVOBg/viewform?usp=dialog) 提交 AMA 问题。



---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF Spaces 意外崩溃**：成员们讨论了 **HF Spaces** 可能会意外重启的问题，建议固定依赖版本以避免此类问题，详见[文档](https://huggingface.co/docs/hub/spaces-sdks-gradio-hardware#restarts)。
   - 一位用户报告称 *Ilaria RVC* 和 *UVR5 UI* 都已停止工作，并建议进行工厂重建（factory rebuilds），而其他项目运行正常。
- **P104-100 GPU 仅售 15 美元！**：用户们讨论了 **P104-100** GPU 在 AI 任务中的价值，有人声称它仅需 **15 英镑**（尽管有人认为是骗局）就相当于一个 **1080**，可在[淘宝](https://item.taobao.com/item.htm?id=897611642586&sku=Video%20memory%20capacity:8gb;Color%20classification:Galaxy%20p104-100%20(card%20length%2028cm);&sku_id=5919375243616)购买。
   - 一些人指出了 **PCIE 1.0 x4** 等限制，而另一些人则强调了它在 LLM 推理（inference）中的性价比，即使是在多卡分片（sharding）运行模型时也是如此。
- **Qwen 30B 挑战 GPT-4o**：用户们关注了 **Qwen 30B** 模型的发布，声称其可与 **GPT-4o** 媲美，并且使用 [Unsloth GGUF 版本](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF)仅需 33GB RAM 即可在本地以全精度运行。
   - 用户指出，量化（quantized）版本仅需 17GB RAM 即可运行。
- **扩散模型 MIT 课程学习小组**：一个新的**学习小组**将专注于从零开始学习**扩散模型（diffusion models）**，采用 **MIT 课程**，早期报名费用为 **50 美元/月**。非会员可参加两次免费介绍课程：**8 月 2 日**和 **8 月 9 日**，详情见 [Luma](https://lu.ma/kv8zf6va)。
   - 该学习小组将基于 [MIT 讲义](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf)和[之前的录像](https://aischolars.notion.site/)。
- **MoviePy 构建视频编辑服务器**：一位成员使用 **MoviePy** 构建了一个 **MCP server**，用于处理基础的视频/音频编辑任务，并集成了 **Claude Desktop** 和 **Cursor AI** 等客户端，代码已发布在 [GitHub](https://github.com/Aditya2755/video-edit-mcp)。
   - 作者正在寻求在基于对象检测（object detection）的编辑和 TTS/SST 驱动的剪辑等功能上的合作。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-5 炒作让粉丝坐立难安**：用户们正在热烈讨论 **GPT-5** 的发布，有人声称可以通过 **Microsoft Copilot/Azure** 访问，但怀疑论者仍在等待 **OpenAI** 的正式公告。
   - 一位用户幽默地批评 **Sam Altman** 煽动炒作，让粉丝们*坐立难安*。
- **Study and Learn：分心还是创新？**：**OpenAI** 推出了新的 **Study and Learn** 功能，一些人认为这只是一个简单的系统提示词（system prompt），或许是为了转移对 **GPT-5** 预期的注意力。
   - 一位用户甚至将[系统提示词](https://discord.com/channels/974519864045756446/998381918976479273/1399983224880496711)丢给 O3 模型进行分析。
- **Copilot 与 ChatGPT 的对决**：讨论明确了 **Microsoft Copilot** 使用的是 **GPT-4o** 或 **O4-mini-high**，根据源代码提示，未来可能会集成 **GPT-5**。
   - Copilot 无限制的每日消息上限引发了关于用户为何仍偏好 **ChatGPT** 的疑问，尽管一些用户仍认为 Google 的 [Imagen4-Ultra](https://discord.com/channels/974519864045756446/998381918976479273/1400170254902235246) 是最好的图像生成器。
- **聊天记录凭空消失**：多位用户报告 **ChatGPT 聊天记录**消失，尽管尝试通过重新登录和清除缓存来排除故障。
   - OpenAI 支持团队表示这*可能是一个孤立的 Bug*，并强调*聊天记录一旦丢失就无法恢复*。
- **工程化新型 AI 内存格式**：一位成员提出了一种新的内存格式提案 [AI_ONLY_FORMAT_SPEC](https://discord.com/channels/1046317268651794482/1046317269069864970?event=1200057848907847709)，旨在优化 AI VM、与向量数据库（vector databases）对接的系统或受保护的符号传输，强调速度和效率而非人类可读性。
   - 另一位成员对该格式进行了详细的逐行解读，强调了其核心原则，如 **token embedding**、**语义分组（semantic grouping）**和**二进制编码（binary encoding）**。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeepSecure 发布开源 AI Agent 身份验证方案**：**DeepTrail** 推出了 **DeepSecure** ([https://github.com/DeepTrail/deepsecure](https://github.com/DeepTrail/deepsecure))，这是一个为 AI Agent 设计的开源身份验证和授权代理层，支持跨模型、平台和框架的授权、Agent 间授权、策略执行以及安全代理。
   - 该技术采用了分裂密钥架构、网关/代理、分离的控制/数据平面、策略引擎以及 macaroons。在 Langchain/LangGraph 的集成示例中，展示了具有细粒度访问控制的[安全多 Agent 工作流](https://github.com/DeepTrail/deepsecure/blob/dev/examples/05_langchain_secure_tools.py)和[授权工作流](https://github.com/DeepTrail/deepsecure/blob/dev/examples/09_langchain_delegation_workflow.py)。
- **OpenRouter 充值 10 美元即可获得免费消息额度**：在 OpenRouter 上一次性充值 **10 美元**即可解锁**每日 1000 条免费消息**，即使在初始信用额度耗尽后依然有效。
   - 用户确认在花完初始的 10 美元后，**1000 次请求/天**的限制依然保持激活状态。
- **API 阻止不需要的量化版本**：OpenRouter 的 API 现在允许用户指定可接受的量化级别，以通过 [provider routing 文档](https://openrouter.ai/docs/features/provider-routing#quantization-levels) 避免使用 **FP4** 等低精度模型。
   - 该 API 允许排除特定的量化级别，例如允许除 FP4 模型之外的所有模型。
- **DeepInfra 的 Gemini Pro 秘密协议**：**DeepInfra** 与 **Google** 就 **Gemini 2.5 Pro** 达成了更低的费率协议，并将节省的成本让利给客户，DeepInfra 的列表上标有“partner”标签。
   - 与 **Kimi K2** 模型不同，DeepInfra 的 **Gemini 2.5 Pro** 带有合作伙伴标签，表明其与 Google 建立了直接合作伙伴关系。
- **Ori 机器人表现不佳**：用户报告称 **OpenRouter Ori 机器人**可能会产生负面影响，因为它提供的响应不准确，特别是在支付处理问题上。
   - 一位用户表示 **Ori** 经常将错误归咎于用户，并提出一些毫无结果的问题；目前一名开发人员正致力于更新 **Ori** 的知识库。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Metislist 排名引发关于 François 的热议**：一位用户分享了 [Metislist](https://www.metislist.com/)，引发了关于 **François Chollet** 排名第 80 位的争论，许多人认为这位 **Keras** 的创作者理应获得更高的排名。
   - 有人认为 Chollet 应该进入前 50 名，一位用户调侃道：“该死，你跟我兄弟 François 有过节吗？”。
- **Arcee 发布 AFM-4.5B 模型**：Lucas Atkins 宣布在 Hugging Face 上发布来自 [Arcee.ai](https://xcancel.com/LucasAtkins7/status/1950278100874645621) 的 **AFM-4.5B** 和 **AFM-4.5B-Base**，宣称由于与 DatologyAI 的数据合作，该模型具有灵活性、高性能和高质量。
   - 这些模型结合了架构改进，如 **grouped query attention** 和 **ReLU² 激活函数**，并计划在未来发布推理和工具使用版本。
- **NotebookLM 现在支持总结视频**：**NotebookLM** 推出了一项新功能，可以对文章和博客文章生成视频概览 ([xcancel.com](https://xcancel.com/NotebookLM/status/1950298236914139234))，使用户无需阅读全文即可掌握内容。
   - 用户对这一创新表示赞赏，并建议进一步开发交互模式。
- **MacOS 上发现 GPT-5 踪迹**：在 MacOS 应用缓存文件中发现了对 **gpt-5-auto** 和 **gpt-5-reasoning** 的引用 ([xcancel.com](https://xcancel.com/apples_jimmy/status/1950514936444305534?s=46&t=fRVjULzONZQAlwHruKTgQg))，暗示 **GPT-5** 即将到来。
   - 进一步的佐证提到了生物学基准测试仓库中的 **gpt-5-reasoning-alpha**，引发了关于潜在发布或公告的猜测。
- **Anthropic 目标估值 1700 亿美元**：据报道，Anthropic 正寻求融资 **50 亿美元**，这可能使这家 AI 初创公司的估值达到 **1700 亿美元**，预计到今年年底收入将达到 **90 亿美元** ([xcancel.com](https://xcancel.com/EdLudlow/status/1950561790695448810))。
   - 这一消息引发了与 OpenAI 和 xAI 的对比。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Tableau Server 展示 LLM 编排实力**：一名成员报告成功集成了最新的 **Tableau Server 版本**，为 **Vizql NLP** 引入了 **LLM (server/on prem)**。
   - 该设置旨在为 Tableau 可视化提供更高级的自然语言处理能力。
- **Gemini Agentic Framework 原型现身**：一名成员分享了一个 [**Gemini agentic framework**](https://cdn.discordapp.com/attachments/1124403655819415592/1399853283404939454/gemini-agentic-framework.zip?ex=688bd3f6&is=688a8276&hm=101f03e62cae13a72e1f4fdc681064aef0e5a3713de20aebac608c958f845b8b) 原型，并将其描述为一个 **one-shot prototype**。
   - 该原型利用 **AI Studio** 构建 Agent 应用，强调为构建者 Agent 设置清晰的意图，以促进分阶段测试和专注的模型开发。
- **NotebookLM 绕过机器人限制实现播客梦想**：针对 **NotebookLM** 因机器人限制导致播客创建受限的咨询，一名成员澄清可以通过 **API** 访问这些工具。
   - 他们建议重新构建工作流并手动将报告加载到 **NotebookLM** 中，作为一种替代方案。
- **Obsidian 与 NotebookLM 紧密协作**：[这里](https://www.xda-developers.com/using-notebooklm-obsidian-google-drive-together/)分享了一篇详细介绍 **NotebookLM**、**Obsidian** 和 **Google Drive** 集成的文章。
   - 一名成员主动提出根据个人用户需求，提供关于 **Obsidian** 使用的更详细指导。
- **NotebookLM 音频输出在 8-15 分钟之间波动**：用户报告 **NotebookLM** 生成的音频文件平均时长为 *8-10 分钟*，不过也有人达到了 *15 分钟*。
   - 这一讨论强调了输出长度的变动性，可能受到内容复杂性和处理效率的影响。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **俄罗斯地震后发布海啸预警**：俄罗斯海岸附近发生的 **8.7 级地震** 触发了夏威夷的海啸预警以及美国西海岸的海啸观察预警。
   - 受影响地区的居民被建议密切关注更新，因为海啸可能在数小时后到达。
- **LM Studio 用户请求增强对话处理功能**：用户请求在 LM Studio 中增加复制和粘贴完整对话的功能，这些对话以 **JSON 格式** 存储。
   - 一名用户引导他人关注 [功能请求频道](https://discord.com/channels/1110598183144399058/1128339362015346749)，并指出许多人会发现这个功能很有用。
- **LM Studio 模型“重回” 2 月 18 日**：一名用户报告他们的 LM Studio 模型反复引用 **2024 年 2 月 18 日**，即使在询问时事时也是如此。
   - 另一名用户建议检查 **system prompt** 或 **Jinja template** 中的日期设置。
- **Strix Halo APU 价格昂贵，引发 EPYC 讨论**：**Strix Halo APU** 的价格在 64GB 版本约 **$1.6k**，128GB 版本约 **$2k**，但一些成员建议 **EPYC** 系统提供更好的性价比。
   - 一名成员对这类设备上的 *焊接内存 (soldered memory)* 表示遗憾，并将其与最近服务器上的 DIMM 故障进行了对比，同时指向了 [搭载 Strix Halo APU 的 Corsair AI Workstation 300](https://www.guru3d.com/story/compact-ai-pc-corsair-ai-workstation-300-with-strix-halo-apu/)。
- **9070 XT 性能令人失望**：**9070 XT** 明显慢于 **4070 Ti Super**，一名用户报告某个模型在 **4070 Ti Super** 上运行速度为 **7 t/s**，而在 **9070 XT** 上仅达到 **3 t/s**。
   - 另一名成员认为 RAM 带宽限制可能是原因。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaCloud 读取 PDF 遇阻**：一位成员报告称 **LlamaCloud** 无法检测到 **PDF 文件**并无法通过 API 进行处理。该成员使用 **n8n** 来简化工作流，并链接了一张[截图](https://cdn.discordapp.com/attachments/1059201661417037995/1399848961832911031/Screenshot_2025-07-29_at_22.13.16.png?ex=688bcff0&is=688a7e70&hm=b8f51e99fbeae087df203303f7665c4eab8447bb0890b55823fd36074c5ad539&)。
   - 此问题发生在尝试通过 API 处理 **PDF 文件**时。
- **Character AI 激发构建讨论**：成员们讨论了如何构建一个对宏大故事有深度理解的 **character AI**，建议使用经典的 **RAG** 流水线，结合文本分块、embeddings 和向量数据库。
   - 这包括利用 **RAG** 流水线来创建一个具有深度理解能力的 AI。
- **Neo4j 知识图谱遭遇瓶颈**：一位成员报告称，他们实现 **Neo4j** 的简单图存储加载速度*极慢*，且其服务器与 **Neo4j 5.x** 不兼容，而 **LlamaIndex** 似乎又不支持 **4.x**。
   - **Aura** 也被服务器代理拦截，为实施增加了更多障碍。
- **Flowmaker Gemini 2.5 Pro Bug 获得快速修复**：一位成员报告了在使用 **Flowmaker** 配合 **Gemini API** 时，由于模型名称无效（需要像 *gemini-2.5-pro* 这样的数字标识）而产生的错误。
   - 一个修复方案已被[提交](https://github.com/run-llama/flow-maker/blob/aad0f47a81cacba662a07c4f2d70bd3425606e29/src/lib/llm-utils.ts#L19)并迅速部署，解决了该问题。
- **社区提供 RAG 调试支持**：一位成员提供了一个 **MIT 许可的仓库**，旨在调试棘手的 **RAG 问题**，包括稀疏检索（sparse retrieval）、语义漂移（semantic drift）、分块崩溃（chunking collapse）和内存崩溃。
   - 在初步提议后，一位社区成员询问了该仓库解决的具体复杂问题，重点关注 *sparse retrieval* 和 *semantic drift* 的具体案例。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **专家并行 (Expert Parallelism) 表现卓越？**：一位成员寻找 **Expert Parallelism (EP)** 优于 **Tensor Parallelism (TP)** 的案例，但发现对于 **Qwen32B** 和 **Qwen 235B**，all-reduce 通信开销使得 **EP** 性能较低。
   - 他们观察到 **EP** 仅对使用 **MLA** 且需要 **DP attention** 的模型有益。
- **Torch Compile 中的 Triton 秘籍**：为了提取 **PTX 代码**，成员们建议使用 `TORCH_LOGS="output_code" python your_code.py` 或访问 `compiled_kernel.asm.keys()` 字典，详情参考[这篇博客文章](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/?utm_source=chatgpt.com#ttir-triton-ir)。
   - `keys()` 字典包含中间表示的键，包括 **llir, ttgir, ttir, ptx 和 cubin**。
- **Inductor 对 Triton 的有趣影响**：为了强制为矩阵乘法（matmuls）生成 **Triton 代码**，成员们建议配置 [torch._inductor.config.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L459-L461) 中的设置，通过修改 **use_aten_gemm_kernels**、**autotune_fallback_to_aten**、**max_autotune_conv_backends** 和 **max_autotune_gemm_backends** 等设置来实现。
   - 然而，有人指出内置算子（kernels）通常更快，且并非每个操作（op）默认都会转换为 Triton。
- **CuTeDSL 编译器减少预取代码**：一位成员分享了关于在 **H100** 上使用 **CuTeDSL** 进行 **GEMM** 的[博客文章](https://veitner.bearblog.dev/let-the-compiler-do-the-work-in-cutedsl/)和[代码](https://github.com/simveit/software_pipelining_cute_dsl)，解释了如何让编译器处理预取（prefetching）。
   - 博客详细介绍了一个传给 `cutlass.range` 算子的实验性参数，用于提示预取，从而以更简单的代码实现与手动预取相当的性能。
- **Gmem 守护者：Synchthreads 拯救局面**：在从全局内存（**gmem**）拷贝到共享内存后，必须手动插入 `synchthreads`（或等效指令）以在继续执行前同步所有线程。
   - 这保证了所有共享内存元素在进行 **gemm**、reduction 或 scan 等集体计算时均已就绪。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **M3 Ultra 荣登本地推理之王**：根据 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1j43ziq/the_new_king_m3_ultra_80_core_gpu_512gb_memory/)，**M3 Ultra** 凭借其 **80 核 GPU** 和 **512GB 内存**，正成为本地推理的首选。
   - 由于没有人回应其帖子，一名成员转而购买了二手的 **M1 16g**。
- **解决离岸模型延迟**：一名成员正在寻求在网络条件较差的情况下，低延迟运行 **离岸 LLMs** 的解决方案。
   - 其他成员只是简单地表示，他们愿意在任何他们想要的东西上花钱。
- **微软的语音指令研究**：成员们对研究改进 **Speech-LLM 模型** 中 **语音指令遵循** 的开源解决方案表现出兴趣，以创建更好的语音 UI，并指向了 **微软的 Alignformer**。
   - Alignformer 尚未开源，因此可能需要进行协作。
- **ICL 摧毁诊断工具？**：成员们推测 **上下文学习 (ICL)** 可能会破坏 **稀疏自编码器 (SAEs)** 等 **可解释性工具**，因为 **ICL** 会将激活值推离其原始训练分布，参考了 **卢卡斯批判 (Lucas Critique)** 和 [这篇论文](https://arxiv.org/abs/2501.00070v1)。
   - 成员们认为，这个问题并非 **ICL** 独有，而是每当 **SAEs** 遇到与训练时不同的激活分布时都会出现。
- **Grouped GEMM 取得进展**：一名成员强调了一个在 **GPT-NeoX** 中支持 **torch._grouped_mm** 的 PR，该功能现已进入 PyTorch 核心，暗示了 **MoE** 的性能提升，并链接到了 [这个 MoE 实现](https://github.com/EleutherAI/gpt-neox/blob/d12c771198388980ee054617e537665f044e0584/megatron/model/moe_mlp.py#L221)。
   - 他们指出，感兴趣的用户可以使用 TorchAO 的一行代码来进行 **低精度 MoE 训练**。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **关于 CUDA 泛化表面的论文**：一名成员分享了一篇关于超越 **CUDA** 泛化的 [论文](https://huggingface.co/papers/2507.14111)，引发了在适当频道发布此类内容的提醒。
   - 未提供进一步的讨论或替代观点。
- **Mojo 新手挑战 TLS 握手故障**：一名 Mojo 新用户报告称，在使用来自 Microsoft Copilot 的 **Dockerfile**，并尝试通过 **pixi** 和 **magic shell** 运行 Mojo 项目时，出现了 **TLS 握手 EOF 错误**。
   - 建议的修复方案包括使用最新的 nightly `mojo` 包配合 **pixi** 及特定命令（`pixi init my-project -c https://conda.modular.com/max-nightly/ -c conda-forge` 和 `pixi add mojo`），但即使使用 **VPN**，该修复也失败了。
- **剖析 Mojo external_call**：用户询问为什么 Mojo 的 `external_call` 使用特定的函数如 `KGEN_CompilerRT_IO_FileOpen` 而不是 **libc** 中的标准 `fopen`，并担心这种选择是否出于安全考虑。
   - 一名成员澄清说，这些是旧版本 **Mojo** 的遗留产物，修复优先级不高，且 **KGEN** 命名空间属于 Modular，最终将会开放。
- **从 Mojo 调用 Python 存在开销**：一名用户发现，从 Mojo 调用 Python 的 no-op 函数存在显著开销（1000 万次调用耗时 4.5 秒），而直接在 Python 中执行仅需 0.5 秒。
   - 成员解释说，Mojo 需要启动一个 **CPython** 进程，并且 CPython 是通过 `dlopen libpython` 嵌入的，因此不应在 **热循环 (hot loop)** 中调用它。
- **在赛车引擎中喷胶**：讨论涉及了从 Mojo 调用 Python 的性能影响，特别是在处理 **OpenCV** 或 **Mujoco** 机器人仿真等任务的热循环中。
   - 成员指出，许多快速的 Python 库实际上是带有包装器的 C 库，并且 *仅与上下文词典 (context dicts) 交互就很容易消耗数百个周期*。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Deepseek-chat 在 OpenRouter 上表现不佳**：成员们观察到，与官方 **DeepSeek API** 相比，**Deepseek-chat** 在 **OpenRouter** 上的表现较差，特别是在 architect mode 中作为编辑器模型使用时。
   - 推荐的修复方法包括使用 `aider --model openrouter/deepseek/deepseek-r1`，以确保应用来自 [aider/resources/model-settings.yml](https://github.com/Aider-AI/aider/blob/main/aider/resources/model-settings.yml#L548) 中带有 `edit-format: diff` 的默认配置。
- **Aider 作为编程模型训练的催化剂？**：有建议认为 **Aider** 可以通过记录开发工作流中的 linting 和撤销（undo）操作来辅助编程模型的训练。
   - 这种方法将在“谨慎”的开发过程中使用 **Aider** 来生成有价值的训练数据，尽管评论者并未明确要求开发者实现此功能。
- **Qwen3 Coder 30B-A3B 震撼登场**：有人发布了关于新模型 **Qwen3 Coder 30B-A3B** 的公告，并分享了一张图片以证实其真实性。
   - 关于这一新模型的详细信息仍在不断披露中。
- **Litellm API 陷入连接错误困境**：一名用户报告遇到了大量的 `litellm.APIConnectionError: Error parsing chunk: Expecting property name enclosed in double quotes: line 1 column 2` 错误。
   - 尽管存在这些错误，该用户的功能使用并未受到影响。
- **用户请求开源模型 R1 与 Qwen Coder 的对决**：一名成员在考虑硬件无限制的情况下，征求关于最适合 **aider** 的开源模型的建议，并表示有兴趣测试 **R1** 和 **Qwen Coder** 模型。
   - 该成员提到有 **Runpod credits** 可供使用，表明计划对这些模型进行实际测试。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **寻求 LLM 安全研究资源**：一名博士生请求关于当前 **LLM safety/alignment research** 的资源，建议包括来自 **AI alignment forum** 的博客。
   - 特别提到了博客文章 [Thought Anchors: Which LLM Reasoning Steps Matter](https://www.alignmentforum.org/posts/iLHe3vLur3NgrFPFy/thought-anchors-which-llm-reasoning-steps-matter) 和 [Measuring Beliefs of Language Models During Chain-of-Thought](https://www.alignmentforum.org/posts/a86uAnPykqNtmEbDH/measuring-beliefs-of-language-models-during-chain-of-thought-1)，认为它们是很好的起点。
- **使用 Claude 编写 CUDA 代码令开发者困惑**：一名成员发现使用 **Claude** 编写 **CUDA** 非常困难，需要大量的准备、理解和排列工作。
   - 他们认为最终的评估标准在于，一个具备一定 GPU 和 **CUDA** 基础的 Python 程序员是否能利用 **Claude** 来管理 **kernels** 的编写并提升性能，文中包含[一张图片](https://cdn.discordapp.com/attachments/986699377257119794/1400233738369110026/image.png?ex=688be4ca&is=688a934a&hm=1bcb11346477e61edf05cde9751d5e62ee8992a2f64216c07e4a1a8f8fb14cc4)。
- **Z.AI 的 54 个仓库引发关注**：一名成员询问了关于新的 **Z.AI 54 open source repos** 的情况，以及是否有人探索过它们，这引发了社区内的好奇。
   - 然而，关于这些仓库的具体内容或功能的细节尚未展开讨论。
- **据称 Qwen3 与 GPT-4o 旗鼓相当**：一名成员分享了一篇帖子，指出 [Qwen3 30B A3B 2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) 在英文和中文方面都能与 **OpenAI** 的 **GPT-4o** 竞争。
   - 社区对 **Qwen3** 作为一个潜在的强力竞争者感到兴奋，基准测试表明其性能可能与 **GPT-4o** 持平，特别是在多语言应用中。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi Chatbot 说中文**：有用户报告称，尽管输入的是 **English** 提示词，**Kimi chatbot** 仍以**中文**回复，这可能是由于账号处于登出状态导致的。
   - 截图显示，虽然回复内容是英文，但推荐的来源和问题却是中文。
- **Kimi 从社交媒体学习**：一位成员开玩笑说 **Kimi 的训练数据集** 包含了 **Instagram** 和 **TikTok** 的评论，并链接到 [Kimi on OpenHands v0.50.0k2](https://github.com/All-Hands-AI/OpenHands/releases/tag/0.50.0k2) 以支持这一说法。
   - 他们认为，正是对社交媒体数据的关注才让 **Kimi** 表现得如此出色。
- **Moonshot AI 的氛围最好**：一位成员表示 *moonshot got the best vibe no cap*（Moonshot 的氛围真的最棒），并链接了一篇关于 AI 社区氛围检查的 [X 帖子](https://x.com/crystalsssup/status/1944287779896328668)。
   - 另一位成员表示赞同，认为社区需要一些竞争。
- **Scale AI 提供数据**：一位成员指出 **Alexandr Wang** 是 **Scale AI** 的创始人兼 CEO，该公司提供训练数据、标注和评估服务。
   - 他们强调 **Scale AI** 对于开发机器学习模型至关重要。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Lume 在正面交锋中险胜 Suna**：成员们讨论了 **Lume** 和 **Suna** 这两个 Agent 系统的优劣；一位成员发现 **Lume** 在编写特定内容时错误更少。
   - 该成员指出，由于成本原因，他们无法与 **Manus** 进行比较，并承认可能没有对 **Suna** 进行正确的提示词引导。
- **Manus 的漫画创作：一块未经雕琢的璞玉？**：一位成员建议 **Manus** 的漫画创作功能很不错，但仍有改进空间，特别是对于免费用户而言。
   - 另一位成员表示服务质量正在下降，对免费用户的限制非常严格，并质疑 *Manus 是否已经凉了*。
- **AI 对 Manus 的乐观态度 vs. 人类的怀疑**：一位成员询问 AI 如何看待 **Manus** 的未来，AI 回复称 *我认为 Manus 的前景一片光明*。
   - 另一位成员表示怀疑，理由是 **OAI** 和 **Google** 发布了 Agent 模式，带来了巨大的竞争压力。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP Server 面临上下文危机**：一位用户质疑在单个云端部署的 **MCP server 实例** 中，是否需要额外的**用户上下文隔离**层来防止独立会话之间的数据共享，并引用了 [issue #1087](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1087) 和 [MCP 文档](https://modelcontextprotocol.io/docs/tutorials/use-remote-mcp-server#understanding-remote-mcp-servers)。
   - 另一位用户报告了通过 **Claude Desktop** 连接其 **MCP server** 时遇到的挑战，尽管已成功部署到 **EC2** 并配置了正确的 **SSL**，但他们仍能通过 **Cursor** 连接。
- **Cucumber 和 LLM 正在酝酿 BDD**：一位用户分享了一个生产就绪的**行为驱动开发 (BDD)** 侧边项目，其中包括一张 [解决方案架构图](https://cdn.discordapp.com/attachments/1312302100125843476/1399854833565044756/bael.jpeg?ex=688bd568&is=688a83e8&hm=2e86139e9f117265cd7cbef2afcc1a23a34a091e79402df9a0e051261231c695&)。
   - 另一位用户报告称，在 **Claude desktop** 中，**CursorTouch** 的 **Windows-MCP** 无法使用 **state tool**。
- **DeepTrail 发布用于 AI Agent 身份验证的 Deepsecure**：在 Berkeley SkyDeck 支持下，**DeepTrail** 正在开发 **Deepsecure**，这是一个用于 AI Agent 的开源身份验证和授权代理层，详情记录在 [GitHub](https://github.com/DeepTrail/deepsecure) 上。
   - **Deepsecure** 的架构特点包括分裂密钥设计、网关/代理、分离的控制/数据平面、策略引擎以及用于 Agent 间授权的 macaroons，详见其 [技术概览](https://github.com/DeepTrail/deepsecure/blob/dev/docs/design/deepsecure-technical-overview.md)。
- **FastMCP 被问及工具动态**：一位用户询问了当服务器上定义了多个工具时，**FastMCP** 的动态工具选择能力。
   - 具体而言，他们想知道 **FastMCP** 是否具有在客户端自动选择工具（例如数学、网页搜索、RAG、数据解释器）的逻辑。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 参数化：可学习参数提案引发关注**：一项为 DSPy 添加可学习参数（`dspy.variable` 或 `dspy.parameter`）的提案引发了热烈讨论，并在 [GitHub 上创建了一个 issue](https://github.com/stanfordnlp/dspy/issues/8593) 以收集想法和用例。
   - 其目标是通过允许“模板作为参数/变量”并优化模板变量的放置，使优化器能够生成最优 Prompt。
- **F-Strings 表现不佳：Signature 实现受阻**：一位成员在尝试使用 f-string 实现 Signature 以根据描述验证代码时遇到问题并寻求帮助。
   - 另一位成员建议不要采用这种方法，推荐将参数描述放置在 `dspy.InputField()` 中。
- **DSPy 进入遗传算法提示词（Genetic Prompting）领域**：一位成员分享了一段对比 **DSPy** 与 **GEPA** 的 [YouTube 视频](https://www.youtube.com/watch?v=o6RbVPFOslg)，视频中提到 *DSPy 优化你给出的 Prompt；而 GEPA 则进化出你从未想象过的 Prompt*。
   - 该成员建议将 **MIPRO** 进化为一种用于 DSPy 生成 Prompt 的反思型、遗传风格的前沿引擎，以此挑战该博主的观点。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **AMD 在游戏和 AI 领域领跑**：一位成员建议将 **7900XT** 和 **7800X3D** 搭配用于游戏，并指出 AMD 在消费级 AI 方面的可用性和长期的社区效益。
   - 他们链接到了[一条推文](https://x.com/Teknium1/status/1950596567968477382)，论证了选择 AMD 而非 Nvidia **9070** 和 **9900X** 的理由。
- **Qwen 发布具备思考能力的编程模型**：一位成员宣布在 [Hugging Face](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507) 上发布了 **Qwen3-30B-A3B-Thinking-2507** 编程模型。
   - 链接的 Hugging Face 模型为代码生成提供了一个新工具。
- **RLVR：是算法还是营销？**：一位成员质疑将 **RLVR**（Reinforcement Learning, Virtual Reality）归类为强化学习算法是否合适，引发了讨论。
   - 另一位成员 teknium 在回应 [NVIDIA 的推文](https://fxtwitter.com/NVIDIAAIDev/status/1950279130450444670)时表示：“*RLVR 根本不是一种 RL 算法，它只是 RL 的一个目标*”。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **研究员请求开设 AI Safety 暑期学校频道**：一位新成员请求为几周前举办的 AI Safety **暑期学校**开设专门频道。
   - 这反映了社区内对学术和学习导向讨论的持续兴趣。
- **偏见研究爱好者联手对抗偏见**：林茨约翰·开普勒大学（JKU Linz）的一名博士生正专注于**缓解 LLM 中的社会偏见**，其研究兴趣还包括**生成模型的归因**、**AI 生成文本检测**以及**领域自适应（Domain Adaptation）**。
   - 该学生热衷于与其他从事领域特定 LLM 实际伦理问题研究的人员建立联系，寻求合作。
- **内核开发者在 CUDA 中编写内核**：一位名叫 Ali 的成员正深入参与 **Triton/CUDA 中的 GPU 内核优化**，特别是针对**自回归模型（Autoregressive Models）**。
   - Ali 对讨论底层 GPU 编程持开放态度，并提供加速模型性能方面的专业知识。
- **引用配置难题困扰 Cohere**：一位成员反映在使用 `langchain_cohere.ChatCohere` 的 `citation_options` 更改**引用模式（Citation Mode）**时遇到困难，并询问传递引用选项的隐式方法。
   - 该成员还询问了 [langchain-cohere 仓库](https://github.com/langchain-ai/langchain-cohere)的状态，指出其近期缺乏更新，并询问*是否欢迎拉取请求（Pull Requests）*。
- **资深软件专家在南方寻求职位**：发布了一个远程**资深软件工程师**职位，月薪 **$2,000**，为长期合同，地点要求在**非洲**或**美洲**。
   - 该职位要求具备 **Ruby on Rails**、**Node.js**、**C#/.NET**、**Python**、**Java** 经验以及出色的英语沟通能力。

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 用户请求 LoRA 风格的适配器 (Adapters)**：一位 Torchtune 用户对 **LoRA 风格的适配器支持**表示了兴趣，该支持可以冻结原始模型权重，并通过额外的可训练层应用更新。
   - 该用户明确希望适配器在不增加计算成本的情况下保持相同的前向计算路径。
- **Torchtune 在训练后合并权重**：一位用户指出 Torchtune 在使用适配器训练后会将权重合并回去，并引用了 [Torchtune 端到端工作流文档](https://docs.pytorch.org/torchtune/0.6/tutorials/e2e_flow.html)。
   - 该用户的评论引发了关于 Torchtune 中合并权重所产生影响的讨论。
- **ACL 论文获得嘉奖**：一位成员分享了他们获奖的 **ACL 论文**，论文链接在[这里](https://aclanthology.org/2025.acl-long.266/)。
   - 该公告发布后没有进一步的讨论。
- **Glianorex 微调 (finetunes) 引发讨论**：一位用户询问 **Glianorex 微调 (finetunes)** 是否公开。
   - 这条评论可能被解读为一种抱怨：*Glianorex 快把我折磨死了，我的医生也帮不上忙*。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **证书声明仍需完成**：一名工作人员提醒另一名成员完成 **LLM Agents (Berkeley MOOC)** 的**证书声明表单**。
   - 工作人员重申，尽管之前已通知该成员，但**从未收到表单**。
- **证书提交的第二次提醒**：工作人员强调了提交**证书声明表单**以完成 **MOOC** 要求的重要性。
   - 未能提交表单将导致无法颁发证书，从而影响课程完成情况的验证。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AI Scholars 启动扩散模型 (Diffusion Models) 学习小组**：一个新的学习小组正在启动一个由 **AI Scholars** 发起的为期 **5 个月**、共 **12 人**的计划（**每周 2-4 小时**），使用 [MIT 的课程体系](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf)来研究扩散模型，这是生成式 AI 中的关键架构。
   - 已确认的成员包括一家 AI 电影工具的 CTO、AI 艺术讲师、2 名 LLM 讲师和 2 名全职 AI 研究员。
- **前两次扩散模型课程免费**：前两次介绍性课程免费并向非成员开放：8 月 2 日关于 *Flow Matching & Diffusion Models*，以及 8 月 9 日关于 *PDEs, ODEs, SDEs + 扩散模型简史*（[链接在此](https://lu.ma/kv8zf6va)）。
   - 该计划的特色包括同行引导的课程、导师问答、实战项目、真实的论文研究，以及一个紧密且值得信赖的队列，每周形式为 2 小时直播课 + 2 小时自学，学生轮流授课。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **用户寻求云端部署策略**：一位用户正在寻求关于将使用自定义 PDF 文件夹训练的语言模型部署到云端供公众使用的建议，并特别希望有一个简单的 **GUI** 供用户查询。
   - Nomic 表示**企业版方案**并不适合，用户想知道 **Hugging Face 部署**是否可以作为替代方案。
- **企业版方案无法满足用户需求**：Nomic 指出，对于用户部署自定义语言模型的需求，**企业版方案**并不合适。
   - 用户正在探索其他的部署策略，例如 **Hugging Face**，以使其语言模型可被访问。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **有评估 Kimi 2 的计划吗？**：一位成员询问是否有评估 **Kimi 2** 的计划。
   - 他们对 Kimi 2 在后训练 (post-training) 后的**工具使用能力 (tool-use capabilities)** 表示好奇。
- **对 Kimi 2 后训练后工具使用的兴趣**：有人表示有兴趣评估 **Kimi 2** 在后训练阶段后的**工具使用性能**。
   - 这一询问突显了评估模型在初始训练后如何适应和利用工具的重要性。

---

**tinygrad (George Hotz) Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。

---

**Codeium (Windsurf) Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。

---

您收到这封邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [退订](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。

---

# Discord：各频道详细摘要与链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1399830622255710320)** (1152 条消息🔥🔥🔥): 

> `R1 1776 移除, Comet Android 版, Gemini 2.5 Pro 速度提升, 针对 R1 1776 的 OpenRouter API, 每日消息上限限制` 


- ****R1-der R.I.P: 模型从 LLM 选择器中移除**：成员们注意到 **R1 1776 模型** 已从 LLM 选择器中移除，但仍可通过 [OpenRouter API](https://openrouter.ai/perplexity/r1-1776) 访问。
   - 用户推测在移除后，推理任务可能会转向 **O3** 或 **Sonnet 4**。
- ****Android Comet: 移动端 App 即将发布**：**Comet for Android** 应用目前正在开发中，预计将于年底发布。
   - 一位用户对浏览器的潜在功能表示担忧，而其他用户则称赞其在 Windows 上的表现。
- ****Pro Gemini 2.5 获得速度提升**：用户观察到 **Gemini 2.5 Pro** 的速度显著提高，暗示它可能使用了 GPT-4.1 而非 Gemini。
   - 成员们指出，这种性能提升可能伴随着限制，例如推理模型的每日消息上限限制。
- ****Spaces 热潮: 自定义指令讨论升温**：成员们讨论了如何通过添加自定义指令来更好地利用 **Spaces** 功能。
   - 一位用户询问空间描述字段还是指令字段更适合设置空间上下文——一位成员回答说，**指令字段提供了更多选项**，例如添加特定网站以抓取特定数据。
- ****API API!: Perplexity API 指南**：用户分享了使用 **Perplexity API** 的技巧，其中一位指出 Pro 订阅者每月可获得 **$5 USD** 的额度分配。
   - 一位遇到 401 错误的用户被建议确保其代码和模型选择（**sonar-pro**）正确。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1399842279937806439)** (4 条消息): 

> `企业级 API 定价, Deep Research API` 


- **Deep Research API 具备结构化输出功能**：一位正在构建需要大量深度研究能力的产品并计划近期融资发布的成员表示，他们对开发问题并不担心，因为他们已经熟悉了 **deep research and structured outputs api**。
   - 他们还询问是否可以与相关人员讨论 **Enterprise API 定价**、早期访问、速率限制和支持，并请求一些额度以便适当地测试和集成 API。
- **团队准备好回答问题**：一位成员确认团队正在关注，并询问另一位成员有什么问题。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1399837823863095307)** (670 条消息🔥🔥🔥): 

> `黑粉骚扰, GLM 4.5 Air, Qwen3-30B, OpenAI 审查, Unsloth 与 Llama 3.1` 


- ****Discord 处理黑粉骚扰****：一位用户举报有**黑粉散布仇恨言论**并私信所有人，导致有人呼吁禁用某些词汇。
   - 成员们确认这没什么大不了的，并提到“显卡羞辱”（GPU-shaming）是一个常见问题。
- ****用户热议 GLM 4.5 Air****：用户讨论了 **GLM 4.5 Air**，其中一位提到它的感觉很像 **Gemini**，并分享了一篇[将其与其他模型对比的博客文章](https://z.ai/blog/glm-4.5)。
   - 成员们指出，该模型在 vLLM 中的 **tool use** 功能已损坏，但在聊天、诗歌分析和文档搜索方面表现良好。
- ****Qwen3-30B 发布引发社区兴奋****：**Qwen 发布了 Qwen3-30B** 并[提供了链接](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF)，引发了关于其性能的讨论。
   - 一位用户报告说，当他们在聊天机器人中询问 Qwen3 为什么中国会审查互联网时，系统立即关闭了他们的请求。
- ****OpenAI 的严厉审查令用户失望****：成员们对 **OpenAI 的严厉审查**表示失望，分享了从 ChatGPT 得到的生硬回答和说教式回应。
   - 一位用户指出了 [Unsloth 的 Llama 4 页面](https://docs.unsloth.ai/basics/llama-4-how-to-run-and-fine-tune)，该页面引导用户永远不要使用暗示道德优越感或权威感的短语，并通常避免说教。
- ****Unsloth 增强 Llama 3.1****：用户询问了 **unsloth/Llama 3.1** 与原生 **Llama 3.1** 的区别，社区成员澄清说 Unsloth 提供了修复、模板调整和 Tokenizer 改进。
   - Unsloth 团队还让在消费级硬件上微调模型变得更加容易，以更少的显存占用提供更快的微调速度。


  

---

### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1400100378560958464)** (6 messages): 

> `Unsloth 介绍，TTS 语音克隆中的低端到端延迟` 


- **印度用户和学生涌向 Unsloth**：一位来自印度的成员在 **Hugging Face** 官方 Discord 听说 Unsloth 后加入，希望学习微调（finetuning）和模型部署（model deployment）。
   - 另一位成员提到他计划*成为 LLM 高手并加入像 Unsloth 这样酷的公司*。
- **寻求低延迟 TTS 语音克隆指导**：一位新成员正在寻求关于实现 **TTS 语音克隆**低端到端延迟的*具体指导*。
   - 该成员请求关于框架、模型优化或硬件策略的建议，另一位成员推荐了我们的 TTS 微调 notebook。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1399849804346949803)** (9 messages🔥): 

> `Gemma 3 4B 微调、自定义 Token、水印去除、RoPE 万岁、语言翻译` 


- **Gemma 3 4B 完成 16k 训练**：在对 **Gemma 3 4B** 进行 **16k** 上下文微调后，实验发现**水印（watermarks）**被完全去除，且模型更加稳定。
   - 研究结果已发布并附带[截图](https://cdn.discordapp.com/attachments/1179039861576056922/1399849804045221948/Screenshot_2025-07-29_at_2.13.07_PM.jpeg?ex=688bd0b9&is=688a7f39&hm=82cad388163625496b8cb3e6dd62035b51ae1d16ce0015df29ea910e04c4471f&)。
- **自定义 Token 引发混乱**：有人指出，除非是从零开始训练（training from scratch）或使用极大的数据集，否则最好避免使用**自定义 Token**，因为模型理解 Token 切分。
   - 例如，如果模型看到 *Yuki* 被切分为 *<*, *yuki*, *>*, 它能更好地理解 Yuki = yuki，并且这是我的 Token。
- **RoPE 值得一个诺贝尔奖**：发布者赞美了发明 **RoPE (Rotary Positional Embedding)** 的天才，因为它效果非常好，尤其是在较小的模型上。
   - 然而，为了在推理（inference）时支持巨大的上下文，发布者认为该领域需要发明更好的优化方案，仅靠量化（quantization）是不够的。
- **Gemma 精通所有语言**：在测试了翻译能力后，发布者开玩笑说 *“天哪，它懂每一种（至少是流行的）语言，OpenAI 完蛋了”*。
   - 他们还提到脑子里一直回响着这首[歌](https://www.youtube.com/watch?v=NgQoLPnuEDM)。
- **新的 Gemma 3 Notebook**：宣布了一项新的 **Gemma 3 竞赛**，notebook 已可用，竞赛将在 **7 天**后结束。
   - 更多信息可在 [Xitter](https://x.com/UnslothAI/status/1950553466591723640) 上查看。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1399831912234881115)** (64 messages🔥🔥): 

> `Phi-4 生成标志错误、GGUF 转换与微调模型的量化、Llama-CLI 性能问题、Google Colab 中的 RuntimeError、Unsloth BNB 4-bit 转换` 


- **Phi-4 需要 `do_sample=True`**：在使用 **Unsloth 的 Phi-4** 模型进行生成时，用户遇到了与无效生成标志（temperature, min_p）相关的错误，并发现添加 `do_sample=True` 可以解决该问题，尽管这在[官方 notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb#scrollTo=kR3gIAX-SM2qHi)中并未注明。
- **Qwen2.5 GGUF 转换困扰**：用户在尝试合并、转换为 **GGUF** 并量化基于 **Qwen2.5-14B-Instruct-unsloth-bnb-4bit** 的微调模型时遇到问题，在导出为 **FP16 GGUF** 期间出现 `ValueError: Can not map tensor 'model.layers.0.mlp.down_proj.weight.absmax'` 错误。
- **使用 UD-Q2_K_XL 模型的 Llama-CLI 运行缓慢**：一位用户报告称，尽管使用了配备 **5090**、**178GB RAM** 和 **EPYC 9334** 处理器的高端系统，且设置理应提供更好的性能，但使用 **Llama-CLI** 运行 **Q2_K_XL** 模型时性能极慢（0.5 tokens/s）。
- **Llama3 微调面临 RuntimeError**：一位用户报告在 Google Colab 上微调 **llama-3-8b-Instruct-bnb-4bit** 时出现 `RuntimeError: PassManager::run failed` 错误，该用户使用了通过 **ShareGPT** 模板格式化的自定义数据集和 Unsloth 库。
- **Whisper input_ids 错误**：一位用户发现，在 `FastModel.get_peft_model` 函数中设置 `task_type = None` 可以解决训练 **Whisper** notebook 时遇到的 `input_ids` 错误，参考 [此 issue](https://github.com/huggingface/peft/issues/1988#issuecomment-2751367819) 获取更多背景信息。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1400049478131777556)** (8 messages🔥): 

> `Quantization optimization, Dynamic 4bit quantization, Hi-Fi Gan replacement, Autoregressive models, Mels dislike` 


- **量化加速计算 (Quantization Speeds Up Compute)**：Quantization 不仅仅是为了将模型装入内存，它还能**减少内存带宽**并**显著提高计算速度**。
   - 一位成员指出，将卷积层等视觉头部操作保留在 **FP32** 似乎并非最优选，因为它们往往非常慢且成为瓶颈。
- **Dynamic 4bit quantization 博客文章**：一位成员分享了 [Dynamic 4bit quantization 博客文章](https://unsloth.ai/blog/dynamic-4bit)，涉及 Quantization 优化。
   - 这篇博客文章与“**Quantization 不仅仅是为了将模型装入内存**”直接相关。
- **Hi-Fi Gan 面临 Autoregressive 竞争**：一位成员询问是否可以在 **VITS** 中用[这个](https://arxiv.org/abs/1609.03499)替换 **Hi-Fi Gan**。
   - 另一位成员询问这是否出于 Autoregressive 的原因，因为第一位成员不喜欢 Mels；然而，第一位成员后来因训练时间过长而决定放弃。


  

---


### **Unsloth AI (Daniel Han) ▷ #[unsloth-bot](https://discord.com/channels/1179035537009545276/1390899684834410536/1399847900598636554)** (102 messages🔥🔥): 

> `GRPO trainer batch size, SFTrainer validation error, Model fine-tuning parameters, Llama 3.2 data preparation, Gemma 3 fine-tuning` 


- **GRPO Trainer 的 Batch Size 解析**：在 **GRPO trainer** 中，*per_device_train_size* 代表 Batch 的数量，然后乘以生成次数（generations）来确定有效 Batch Size。
   - 例如，当 *per_device* 设置为 **1** 且 *num_generation* 为 **6** 时，该配置在单 GPU 下会产生 **3** 个唯一的 Prompt，每个 Prompt 对应 **6** 次生成，考虑到 GPU 显存占用对 activation weights 的影响，在扩展到 **15k** Token 时可能会导致 CUDA Out-of-memory 问题。
- **寻求 SFTrainer 验证错误救星**：一位用户在尝试使用 **SFTrainer** 保存验证错误时遇到了 *evaluation_strategy* 意外关键字错误。
- **Llama 3.2 数据格式**：一位用户请求提供用于微调 **Llama 3.2 8B** 的数据准备格式示例。
- **Gemma 3 仅文本微调策略**：Unsloth 提供了一个[仅文本 Notebook](https://docs.unsloth.ai/basics/tutorials-how-to-fine-tune-and-run-llms/gemma-3-how-to-fine-tune-and-run-llms/gemma-3-how-to-run-and-fine-tune#unsloth-fine-tuning-fixes-for-gemma-3)用于微调 **Gemma 3**，并提供了针对 Unsloth 微调的修复方案。
- **解锁 Adapter 加载的位置参数**：使用 *model.load_adapter* 时，*adapter_name* 是一个必选的位置参数。
   - 一位用户遇到了与不支持的目标模块（**ModuleDict**）相关的 *ValueError*，并寻求修复此问题的指导，旨在将微调后的 LoRA Adapter 合并到基础模型（**unsloth/gemma-3-4b-it-unsloth-bnb-4bit**），以便使用 Llama.cpp 进行 GGUF 转换。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1399830240024465498)** (475 messages🔥🔥🔥): 

> `MCP Browser, Parallel Agent Tasks, VSCode Marketplace with Cursor, Automatic Scrolling, Sonnet Model` 


- **Cursor 的 MCP Browser 设置**：成员们正直接通过 **Cursor 的 MCP** 构建依赖浏览器的自动化工具，理想情况下，早期访问（early access）将在未来几周内推出。
   - 一位成员表示：*它具有一键式 MCP 设置，因此你可以直接通过 MCP 构建依赖浏览器的自动化*。
- **并行 Agent 协作难题**：成员们正在讨论如何处理具有依赖关系的并行任务，因为 Agent 并不共享 Workspace，这使得同时触发它们变得困难。
   - 建议的解决方案包括使用 **API 的外部编排**、**基于文件的协作**或**基于 Docker 的并行执行**，并附带了一个详细的 [task queue script](https://github.com/example) 示例。
- **VSCode Marketplace 集成咨询**：一位成员询问了在 **Cursor 中使用 VSCode marketplace** 的可能性。
   - 讨论中没有明确的答案。
- **自动滚动的困扰**：一位 Cursor 新用户询问是否可以在使用 **Agent 聊天窗口** 时禁用**自动滚动**，以便更好地阅读 Claude 的思考过程和生成的代码。
   - 讨论中没有明确的答案，但有人发布了 [changelog 1.3](https://cursor.com/changelog)。
- **终端终止的奇案**：一位成员在 Cursor 决定其 Agent 集成终端启动哪种 shell 时遇到麻烦，它默认使用 **fish**。
   - 尝试通过设置和封装器更改 shell 后，临时重命名了 fish 二进制文件并取得了成功，尽管根本原因仍然是一个“扑朔迷离”的谜团。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1399860099563917344)** (8 messages🔥): 

> `Background Agent Commands, Docker Build Cache, Port Hijacking, Background Agents for Research` 


- **在 Background Agent 运行结束时执行命令**：一位成员询问如何在 **Background Agent 运行结束**时执行命令（特别是 formatter）。
   - 该成员指出，可以在设置期间运行 `terminals`，但那仅限于开始阶段。
- **清理 Docker 构建缓存**：一位成员在对自定义 Dockerfile 使用编辑过的层时，寻求关于**清理构建缓存**的建议。
   - 另一位成员建议使用 `docker builder prune -f` 或 `docker system prune -a` 来删除未使用的容器、网络和镜像。
- **Cursor Background Agents 劫持端口**：工程师们浪费了时间调试为什么他们的开发环境突然崩溃，最后才发现是 **Cursor Background Agents 劫持了端口**。
   - 一位成员询问将 `ports` 属性设置为空数组是否能阻止 Cursor Background Agent 转发任何端口，另一位用户建议在 VSCode 设置中禁用“自动转发端口（auto forward ports）”。
- **用于研究的 Background Agents**：一位成员询问关于使用 Background Agents 进行研究的问题，例如**重构研究或功能实现研究**。
   - 该成员询问了工作流，建议让 Agent 在 PR 中编写 Markdown 或让其直接进行更改。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1399828973818478592)** (413 messages🔥🔥🔥): 

> `dot.lol data, GPT-5 Release, EU GDPR impact, Zenith Model Relaunch, Video Arena Channels` 


- **Dot-lol 面临数据收集审查**：成员们讨论了 [dot.lol](https://dot.lol) **出售用户数据**和画像信息的可能性，强调认为数据永远不会被用于定向影响或牟利的想法过于天真。
   - 有人担心数据收集的弊端超过了在线服务的实用性，而另一些人则认为**数据收集是不可避免的**，用户应优先考虑不将数据与个人身份关联。
- **GPT-5 计划于 8 月发布？**：传闻暗示 **GPT-5** 可能会在 **8 月**初发布，ChatGPT Mac 应用中可能提到的准备工作证实了 router 架构。
   - 一些社区成员推测其潜在影响，讨论它是否会超越其他模型，甚至表达了对免费层级的期待。
- **欧盟 GDPR 的有效性引发讨论**：讨论了欧盟 **GDPR** 在防止 AI 公司收集数据方面的有效性，对于它是否充分影响了数据收集实践，意见不一。
   - 有人认为 GDPR 主要影响数据的*使用*而非*收集*，但对欧盟消费者而言，*数据收集已被关闭*。
- **Zenith：用户依然关注**：成员们对 **Zenith** 模型回归 LMArena 表示出兴趣，并对其潜在的 **ELO** 评分和性能进行了推测。
   - 一些成员对没机会尝试该模型表示遗憾，而另一些人则对其在平台上的价值发表了强烈看法。
- **新的视频竞技场频道出现**：社区成员讨论了**多个视频竞技场频道**的存在以及这是否为刻意为之。
   - 一名版主解释说，视频竞技场频道旨在*分散生成任务*，因为单个频道内的活动过于频繁*会让人应接不暇*。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1400151134299095092)** (1 messages): 

> `Video Arena, LMArena bot, Staff AMA` 


- **Video Arena 在 Discord 上线！**：LMArena 团队在 Discord 上推出了实验性的 **Video Arena**，允许用户免费生成并对比顶级 AI 模型的视频。
   - 用户可以在指定频道学习如何使用 bot，并开始生成视频、图像和图生视频，然后对自己喜欢的生成结果进行投票。
- **为社区生成的视频投票！**：LMArena bot 允许用户**生成视频、图像和图生视频**，并让任何人对他们喜欢的生成结果进行投票。
   - 在达到一定票数后，bot 会揭晓用于生成每个视频的模型。
- **与 Bot 开发者进行的员工 AMA！**：为了庆祝 **Video Arena** 的发布，宣布了与 bot 开发者 [Thijs Simonian](https://www.linkedin.com/in/thijsdev/) 的 **员工 AMA**。
   - 邀请用户通过 [Google Forms](https://docs.google.com/forms/d/e/1FAIpQLSfS_zh67vgEeftUXzX6ujlNiiFoTQNQd7P7-Nx6aD5zpFVOBg/viewform?usp=dialog) 为 AMA 提交问题。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1399829791682723901)** (290 messages🔥🔥): 

> `HF Space restarts, P104-100 GPU, LLM Deployment, Qwen 30B, SmolLM3` 


- **HF Spaces 意外重启**：成员们讨论了 HF Spaces 为何会意外重启，并建议固定依赖版本以避免自动重新加载引起的问题，详见 [文档](https://huggingface.co/docs/hub/spaces-sdks-gradio-hardware#restarts)。
   - 一位用户报告说 *Ilaria RVC* 和 *UVR5 UI* 都停止工作了，而其他的运行正常，建议进行 Factory Rebuild。
- **P104-100：15 美元的 GPU**：用户们争论了 **P104-100** GPU 用于 AI 任务的优劣，有人声称它只需 **15 英镑** 就“几乎等同于 1080”（尽管其他人称之为骗局），可在 [淘宝](https://item.taobao.com/item.htm?id=897611642586&sku=Video%20memory%20capacity:8gb;Color%20classification:Galaxy%20p104-100%20(card%20length%2028cm);&sku_id=5919375243616) 购买。
   - 一些人指出了它的局限性（PCIE 1.0 x4，可能只能访问 4GB VRAM），而另一些人则强调了它在 LLM 推理方面的性价比，即使是在多张卡上进行模型分片（sharding）时也是如此。
- **用于边缘部署的小型 LLM**：成员们寻求在远程、带宽受限的海洋环境中进行**低延迟 LLM 部署**的建议，建议包括边缘/云混合方案和**激进量化**。
   - 一位用户推荐查看 [HF 上最新的 pytorch 版 smollm3 量化模型](https://huggingface.co/pytorch/SmolLM3-3B-8da4w) 以进行移动端部署，另一位建议在船只靠岸时再部署应用。
- **Qwen 30B 模型：GPT-4o 的挑战者？**：用户们关注了 **Qwen 30B** 模型的发布，声称其可与 **GPT-4o** 媲美，且仅需 33GB RAM 即可在本地以全精度运行（[Unsloth GGUF 版本](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF)）。
   - 据指出，量化版本仅需 17GB RAM 即可运行。
- **SmolLM3 量化问题**：一位成员提到在使用 *torchao* 对 **SmolLM3** 进行量化时遇到问题，其他人建议尝试 *hqq* 或 [官方的 SmolLM3-3B-8da4w](https://huggingface.co/pytorch/SmolLM3-3B-8da4w)。
   - 一位成员指出，如果使用 *llama.cpp*，应该使用 *--jinjai* 参数。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1400252994091487382)** (4 messages): 

> `Muon Optimizer, Smithery` 


- **Muon Optimizer 备受赞誉！**：成员们分享了 **Muon Optimizer** 的链接 ([https://kellerjordan.github.io/posts/muon/](https://kellerjordan.github.io/posts/muon/))。
   - 他们惊叹道：*smithery! smithery 太强了！*
- **Smithery 非常出色**：另一位成员回复说 **Smithery** 确实非常出色。
   - 看来 **Smithery** 在这个频道非常受欢迎。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1399853091410415798)** (5 messages): 

> `Petite Elle Model, Gradio MBTI App, Video Editing MCP Server, Github Python Dataset` 


- **Petite Elle 获得 Mrad 处理**：一位成员的模型 [Petite Elle-L'aime-3-sft](https://huggingface.co/Tonic/petite-elle-L-aime-3-sft) 经过了 **mradermacher 处理**，预计将成为该尺寸下最好的法语模型之一。
   - 量化版本可在 [mradermacher 的 Hugging Face 页面](https://huggingface.co/mradermacher/petite-elle-L-aime-3-sft-GGUF) 获取。
- **结合 Gemini 的 MBTI Gradio 应用**：一个新的用于 **MBTI (迈尔斯-布里格斯)** 测试的 Gradio 应用使用了 **PocketFlow** 和 **Gemini**。
   - 查看该 [应用](https://huggingface.co/spaces/Fancellu/mbti-pocketflow) 及其底层的 [PocketFlow 库](https://github.com/The-Pocket/PocketFlow)。
- **MoviePy 构建视频编辑服务器**：一位成员使用 **MoviePy** 构建了一个 **MCP server**，用于处理基础的视频/音频编辑任务，并集成了 **Claude Desktop** 和 **Cursor AI** 等客户端。
   - 代码已在 [GitHub](https://github.com/Aditya2755/video-edit-mcp) 上发布，作者正在寻求关于基于对象检测的编辑和 TTS/SST 驱动剪辑等功能的合作。
- **GitHub Python 数据集发布**：一个新数据集 [Github Python](https://huggingface.co/datasets/jblitzar/github-python) 包含了 2015 年后 GitHub 上星标数超过 10 个的仓库中所有大小合理、经过过滤去重且具有宽松许可证的 Python 文件。


  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1399862051274232003)** (2 messages): 

> `Diffusion Models, Flow Matching, MIT curriculum` 


- **Diffusion Models 学习小组启动**：一个新的**学习小组**将专注于从零开始学习 **Diffusion Models**，这是生成式 AI 的核心架构。
   - 该小组基于 **MIT 课程**，由 **12 人**组成，为期 **5 个月**（**每周 2–4 小时**）。
- **免费入门课程向非成员开放**：前两节免费入门课程可供非成员参加：**8 月 2 日** - 什么是 **Flow Matching & Diffusion Models**？；**8 月 9 日** - **PDEs, ODEs, SDEs** + **Diffusion Models** 简史。
   - 课程时间为 **12 PM EST**，更多详情请见 [Luma](https://lu.ma/kv8zf6va)。
- **将使用 MIT 讲义**：学习小组将基于 [MIT 讲义](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf) 和 [往期录像](https://aischolars.notion.site/)。
   - 早鸟报名费为 **$50/月**（后续将涨至 **$100/月**）；资金将用于支付**助教**费用。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/)** (1 messages): 

hedi1421: 谢谢 😅
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1400040059947712543)** (1 messages): 

> `Fixing transformers issue, DeepSpeed Integration` 


- **寻求 Transformers 问题修复**：一位成员请求协助修复 [这个 Transformers 问题](https://github.com/huggingface/transformers/issues/39753)。
   - 未提供关于该问题的更多详细信息。
- **DeepSpeed 集成**：关于在 Hugging Face 生态系统中集成 **DeepSpeed** 的讨论。
   - 成员们正在探索最佳实践和潜在的性能提升。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1399919753157673102)** (1 messages): 

> `DuckDuckGo deprecation, Smolagents merge` 


- **DuckDuckGo 搜索包面临弃用**：`duckduckgo-search` 包仍处于弃用状态，正如在 [此 Pull Request](https://github.com/huggingface/smolagents/pull/1548) 中讨论的那样。
   - 一位成员询问了将其合并到 `smolagents` 的时间表。
- **Smolagents 合并即将到来**：拟议的合并旨在将更新和修复集成到 `smolagents` 库中。
   - 社区成员正热切期待合并完成，以便利用最新的改进。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1399995784853848126)** (3 messages): 

> `RAG System Construction, Tool Definition Problems in Unit 1` 


- **RAG 系统扫描对话历史**：一位成员计划构建一个 **RAG 系统**，并使用 **LLM** 扫描对话历史，提取用户特定案例，并使用 **for 循环**将它们嵌入到向量空间中。
   - 他们打算测试这种方法的可行性。
- **Unit 1 工具定义故障排除**：一位成员报告称，他们在 **app.py** 中的工具定义在 **Unit 1** 运行时没有体现出来。
   - 他们已经尝试重启 **Space** 但没有成功，正在寻求建议。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1399829176923717713)** (261 条消息🔥🔥): 

> `Study and Learn feature, GPT-5, Copilot, Gemini vs ChatGPT, AI Ecosystems` 


- **OpenAI 发布 Study and Learn 功能**：OpenAI 推出了新的 **Study and Learn** 功能，部分用户认为这只是一个简单的 system prompt，而非重大更新，还有用户认为该功能旨在**分散用户对 GPT-5 炒作的注意力**。
   - 一名用户将 [system prompt](https://discord.com/channels/974519864045756446/998381918976479273/1399983224880496711) 导出到了 O3 模型中。
- **GPT-5 的猜测引发讨论**：成员们正积极讨论 **GPT-5** 的发布，有人声称已通过 Microsoft Copilot/Azure 获得访问权限；然而，许多用户持怀疑态度，并期待 OpenAI 的正式公告。
   - 一名用户对 CEO 评价道：*去你的 **Sam Altman**，让你那些 ChatGPT 粉丝在你制造的炒作中焦急等待*。
- **Copilot 与 ChatGPT 的对比**：用户讨论了 **Microsoft Copilot** 是否使用了 **GPT-5**，但随后被澄清其使用的是 **GPT-4o** 或 **O4-mini-high**，一些爆料者在源代码中发现了 **GPT-5** 未来可能集成的迹象。
   - **Copilot** 的每日消息上限是无限的，这让一些人质疑为什么人们更倾向于使用带有工具的 **ChatGPT**。
- **ChatGPT 与 Gemini 的对比正在进行**：用户争论了对 **ChatGPT** 和 **Google Gemini** 的偏好，一名用户列举了偏好 **ChatGPT** 的六个关键原因，包括 connectors、RAG 能力、风格匹配、memory 以及 deep research，但其他用户迅速提出了反驳。
   - 一名用户指出，在有人发布了不同 AI 生成的“*像钢铁侠家一样的超级富豪豪宅*”图片后，[Google 的 Imagen4-Ultra](https://discord.com/channels/974519864045756446/998381918976479273/1400170254902235246) 生成的图像效果最好。
- **探索 AI 生态系统**：成员们讨论了如何选择合适的 **AI 生态系统**，权衡了 **Apple One + ChatGPT Plus** 与 **Microsoft 365 + Microsoft Copilot** 或 **Google One AI Premium + Google Gemini** 等选项。
   - 建议尝试两者以确定哪个最适合个人需求，有人提到了 Gemini 与 Google Docs 和 Slides 的集成。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1399902317595328642)** (24 条消息🔥): 

> `GPT-5 versions, O4 mini vs 4o, Missing Chat History, ChatGPT memory issues` 


- **GPT-5 版本推测**：成员们推测 **GPT-5** 的中高阶版本将更加出色，一名成员指出 **Zenith** 在新模型发布前可能是顶级的 coding 模型。
   - 未提供链接。
- **关于 O4 Mini 与 4o 模型的辩论**：一名成员询问是否应该使用 **O4 mini** 而非 **4o（免费模型）** 以获得更智能的回复，并参考了 **O4 mini** 和 **O3** 的 **advanced reasoning** 能力。
   - 未提供链接。
- **ChatGPT 聊天记录消失让用户感到焦虑**：多名用户报告其 **ChatGPT 聊天记录消失**，一名用户尝试了登录登出、清除缓存并检查多个设备，但均无济于事。
   - 一名 OpenAI 支持人员表示这*可能是一个孤立的 bug*，且*聊天记录一旦丢失就无法恢复*，建议定期在 ChatGPT 之外保存重要信息的副本。
- **Memory 问题困扰 ChatGPT 用户**：一名用户提到他们正在开发一个本地解决方案，以应对使用 **ChatGPT** 时遇到的 **memory 问题**。
   - 此前另一名用户提到，他们 2024 年 10 月之后的近期聊天记录无法加载，但仍可以访问新聊天和 custom instructions。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1399879683956277360)** (10 messages🔥): 

> `Personalized GPT setup, AI Memory Format, Optimized AI VM` 


- **个性化 GPT 设置资源探索开启**：一位新用户正在寻找设置 **GPT 项目**的资源，用于追踪**食物/饮食**、**运动**，并创建一个带有时间预期的**计划表**，请求能够增强其账户能力的指令资源和 Prompt。
   - 另一位成员建议采用**个性化方法**，建议直接与模型交互讨论所需功能并考虑额外选项，并提到并非所有人对“更强大”的定义都达成一致。
- **新型 AI 记忆格式：速度与可读性的辩论**：一位成员提出了一种新的记忆格式提案，旨在优化 AI VM、与向量数据库对接的系统或受保护的符号传输，强调速度和效率而非人类可读性，使用 [AI_ONLY_FORMAT_SPEC](https://discord.com/channels/1046317268651794482/1046317269069864970?event=1200057848907847709) 来防止记忆在存储和可读性方面出现严重的低效。
   - 另一位成员对该格式进行了逐行详细解读，强调了其核心原则，如 **token embedding**、**语义分组**和**二进制编码**，并警告不要直接运行它，因为可能会导致压缩或编码响应。
- **Prompt Engineering 有效性探讨**：讨论涉及了 Prompt Engineering 的有效性，其中一位成员觉得*明确拼写出需求的每个方面非常令人疲惫*，更倾向于提供上下文并依赖 AI 的推理能力。
   - 另一位成员建议，这类对话型用户正是 Prompt 工程师设计 System Prompt 的目标对象。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1399879683956277360)** (10 messages🔥): 

> `GPT project guidance, Personalized AI models, AI memory format` 


- **寻求 GPT 项目设置指导**：一位新用户正在为其 **GPT 账户**上的项目设置寻找资源，特别是针对食物/饮食和运动追踪，以及创建一个带有时间预期的计划表。
   - 他们正在寻找指导和工具来增强这些常见项目的指令，从而可能使它们*更强大*。
- **个性化 AI 是关键**：一位用户建议通过与模型本身讨论所需的功能和注意事项来个性化 AI 模型。
   - 他们强调，什么是*更强大*因人而异，个性化对于根据特定兴趣和目标定制 AI 至关重要。
- **AI 记忆格式**：一位成员为 AI 引入了一种新的记忆格式建议，旨在高效存储和检索对话记忆。
   - 该格式旨在通过使用紧凑的、二进制编码的结构，结合语义压缩和隐式上下文来改进持久化记忆，并针对 AI VM 和向量数据库接口进行了优化。


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1399839193567465624)** (2 messages): 

> `DeepTrail, DeepSecure, AI agent authorization, Agent delegation, Policy enforcement` 


- **DeepTrail 构建开源项目 DeepSecure**：一位成员正在构建 **DeepTrail**，这是一个由 Berkeley SkyDeck 支持的开源 AI Agent 身份验证和授权层。
   - 通过 **Deepsecure** ([https://github.com/DeepTrail/deepsecure](https://github.com/DeepTrail/deepsecure))，开发者只需几行代码即可在任何模型、平台或框架中集成授权、Agent 到 Agent 的委托、策略执行和安全代理。
- **DeepSecure 的底层技术细节**：该技术涉及分叉密钥架构（split-key architecture）、网关/代理、独立的控制/数据平面、策略引擎以及用于 Agent 间委托的 macaroons，详见[技术概览](https://github.com/DeepTrail/deepsecure/blob/dev/docs/design/deepsecure-technical-overview.md)。
   - 仓库中还包含几个针对 Langchain/LangGraph 的简单示例和集成。
- **使用 Langchain/LangGraph 的 DeepSecure 示例**：该成员构建了一些 **DeepSecure** 与 Langchain/LangGraph 集成的示例，包括具有细粒度访问控制的[安全多 Agent 工作流](https://github.com/DeepTrail/deepsecure/blob/dev/examples/05_langchain_secure_tools.py)。
   - 该仓库还展示了[委托工作流](https://github.com/DeepTrail/deepsecure/blob/dev/examples/09_langchain_delegation_workflow.py)、[高级委托模式](https://github.com/DeepTrail/deepsecure/blob/dev/examples/11_advanced_delegation_patterns.py)以及[平台 Agent 引导（bootstrapping）](https://github.com/DeepTrail/deepsecure/blob/dev/examples/12_platform_expansion_bootstrap.py)。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1399849676123013373)** (152 messages🔥🔥): 

> `NotebookLLM, OpenRouter Pricing, Blocking quants via API, Becoming a provider, API Key Issues` 


- ****通过一次性 OR 充值解锁每日免费消息****：向 OpenRouter 余额充值 **$10** 即可解锁 **每日 1000 条免费消息**，这是一次性购买，即使额度用完该权益依然有效。
   - 用户确认，即使初始的 $10 额度耗尽，**1000 次请求/天** 的限制仍保持解锁状态。
- ****API 支持量化控制****：用户现在可以通过 API 指定可接受的量化级别，以避开像 **FP4** 这样的低精度模型，参考 [provider routing documentation](https://openrouter.ai/docs/features/provider-routing#quantization-levels)。
   - API 允许指定排除项，例如允许除 FP4 模型之外的所有模型。
- ****Pydantic-AI 与 Kimi-K2 联手解决 Bug****：一位用户强调了 **pydantic-ai** 的优势，包括其完全基于 Pydantic 的方法、MCP 服务器支持、模型/提供商适配器以及自动图形构建，并提到他们使用 **Kimi-K2** 修复了一个 Bug。
   - 该用户强调 pydantic-ai 能够让人专注于业务逻辑，而不是*费力地将来自不同臃肿仓库的 Agent 框架拼凑在一起*。
- ****OR 面临 Kimi-K2 Tool Calling 问题****：一位用户认为他们发现了 OpenRouter 上 **Kimi K2** 的 Tool Calling 支持问题，可能可以通过调整模型模板来修复。
   - 该用户提供了包含 vllm 等框架示例的 [研究资料](https://discord.com/channels/1091220969173028894/1400028050007265340)，并表示修复此问题可为他们的业务节省 **80%** 的成本，并暗示他们将转向 Moonshot。
- ****Gemini Flash 1.5 面临过载问题****：据报道 **Google Gemini Flash 1.5** 显示 *error 503: The model is overloaded*，一位用户分享了其 [价格结构](https://discord.com/channels/1091220969173028894/1195014798837043240/1400220368765194350)。
   - 该模型价格存在波动，输入范围为 **$0.075 到 $0.15**，输出范围为 **$0.30 到 $0.60**。


  

---


### **OpenRouter (Alex Atallah) ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1400206735389491232)** (2 messages): 

> `` 


- **OpenRouter 中没有新模型更新**：OpenRouter 频道中没有关于新模型的重大讨论或更新。
   - 该频道保持非活跃状态，缺乏可供总结的实质性信息。
- **Readybot.io 记录显示没有新活动**：Readybot.io 日志显示 OpenRouter - New Models 频道处于沉默期。
   - 因此，这段时间内没有特定的主题或讨论可以报告。


  

---


### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1399913686432219186)** (63 messages🔥🔥): 

> `Quantized Providers, Groq's Quantization, Deepinfra Pricing, Vertex for Claude, OpenRouter's Ori bot` 


- **量化提供商的默认状态引发争议**：一位用户建议应默认禁用量化提供商，这可能会影响 **Groq**，因为它采用了独特的量化方法。
   - 另一位用户警告称，在达到*临界规模*的用户群之前公开指责提供商存在风险，可能导致提供商退出 **OpenRouter**。
- **Deepinfra 通过 Google 提供 Gemini 2.5 Pro**：据报道 **DeepInfra** 与 **Google** 协商了 **Gemini 2.5 Pro** 的较低费率，并将节省的成本转嫁给了客户；一位用户引用了 DeathMax 的消息并指出 DeepInfra 的列表中带有“partner”标签，证实了这一点。
   - DeepInfra 的 **Gemini 2.5 Pro** 带有“partner”标签，这与 **Kimi K2** 模型不同，表明其与 Google 有直接合作关系。
- **Vertex 在 Claude 4 中表现出色**：一位用户报告称，通过 **Vertex** 使用 **Claude 4 Sonnet** 获得了更好的质量、吞吐量和正常运行时间。
   - 该用户还指出，闭源模型的 AWS/GCP/Azure 镜像可能会带来质量上的差异。
- **OpenRouter Ori Bot 的准确性受到质疑**：一位用户建议 **OpenRouter 的 Ori bot** 可能产生了*负面影响*，因为其回答不准确，应予以限制或禁用。
   - 该用户指出 **Ori** 经常将错误归咎于用户，并提出一些*毫无意义*的问题，尤其是在支付处理问题上。
- **为 Ori Bot 添加知识更新功能**：一位开发者正在努力添加更新 **Ori** 知识的方法，以便在其出错时进行修正。
   - 其他人指出 **Ori** 缺失大量知识且存在幻觉，会提供错误的知识，并建议将该 Bot 的回答限制在最多 2-3 条。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1399843438395916339)** (188 条消息🔥🔥): 

> `Metislist 排名，Arcee.ai 发布 AFM-4.5B，NotebookLM 视频概览，Claude 滥用，ryOS 发布` 


- **Metislist 引发 Chollet 排名争议**：一位用户分享了 [Metislist](https://www.metislist.com/) 的链接（一个 AI 领域的人物排名），引发了关于 François Chollet 排名第 80 位的讨论。
   - 许多人认为以 Keras 闻名的 Chollet 应该排进前 50 名，甚至有人建议将其从名单中完全移除，一位用户开玩笑说：*该死，你跟我兄弟 François 有过节吗？*。
- **Arcee AI 发布 AFM-4.5B 模型**：Lucas Atkins 宣布在 Hugging Face 上发布来自 [Arcee.ai](https://xcancel.com/LucasAtkins7/status/1950278100874645621) 的 **AFM-4.5B** 和 **AFM-4.5B-Base** 模型，强调其设计具有灵活性、高性能和高质量，这得益于与 DatologyAI 的数据合作伙伴关系。
   - 这些模型具有架构上的调整，如 **grouped query attention** 和 **ReLU² activations**，团队计划未来发布用于推理和工具使用的模型。
- **NotebookLM 现支持视频概览**：**NotebookLM** 宣布了一项针对文章和博客文章的视频概览新功能 ([xcancel.com](https://xcancel.com/NotebookLM/status/1950298236914139234))，使用户无需阅读全文即可快速掌握内容。
   - 用户赞扬了这一创新，并建议进一步开发学习工具和交互模式。
- **GPT-5 在 MacOS 中现身**：在 MacOS 应用缓存文件中发现了对 **gpt-5-auto** 和 **gpt-5-reasoning** 的引用 ([xcancel.com](https://xcancel.com/apples_jimmy/status/1950514936444305534?s=46&t=fRVjULzONZQAlwHruKTgQg))，暗示 **GPT-5** 即将发布。
   - 其他用户也证实了这一点，提到了生物学基准测试仓库中的 **gpt-5-reasoning-alpha**，而一些人则推测即将发布公告或正式版。
- **Anthropic 寻求天价估值**：据报道，Anthropic 正在洽谈筹集 **50 亿美元**，这可能使这家 AI 初创公司的估值达到 **1700 亿美元**，预计到年底收入将达到 **90 亿美元** ([xcancel.com](https://xcancel.com/EdLudlow/status/1950561790695448810))。
   - 这一消息引发了与 OpenAI 和 xAI 等其他 AI 公司的比较，尽管一位用户评论说他有*一些二手消息称情况并非如此*。


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1400171262210474005)** (17 条消息🔥): 

> `Anthropic Fellows 论文，LLM 论文俱乐部，社交媒体互动` 


- **Anthropic Fellows 论文专题**：Latent Space Discord 宣布将在 <#1107320650961518663> 频道介绍最近的 **Anthropic Fellows 论文**。
- **LLM 论文俱乐部招募志愿者**：**LLM Paper Club** 正在招募志愿者在未来的俱乐部活动中讲解论文；鼓励感兴趣的人通过 [Luma 链接](https://lu.ma/6uti3zzy)报名。
- **转发呼吁未达预期**：一位成员在 [X 上发布了链接](https://x.com/latentspacepod/status/1950613048303231121) 为俱乐部做广告，但感叹互动率很低。
   - 他开玩笑地声称自己 *“不是专业的吹水员 (yapper)”*，而且不擅长发推。


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1399847059045093637)** (32 条消息🔥): 

> `Tableau Vizql NLP 编排, Gemini Agentic Framework 原型, NotebookLM 播客创建, Obsidian 与 NotebookLM 集成, NotebookLM 使用分析` 


- **Tableau Server 用于 LLM 编排**：一位成员提到使用最新的 **Tableau Server 版本**来适配 **Vizql NLP** 的 **LLM (服务器/本地部署)**。
- **分享 Gemini Agentic Framework 原型**：一位成员分享了一个 [**Gemini agentic framework**](https://cdn.discordapp.com/attachments/1124403655819415592/1399853283404939454/gemini-agentic-framework.zip?ex=688bd3f6&is=688a8276&hm=101f03e62cae13a72e1f4fdc681064aef0e5a3713de20aebac608c958f845b8b) 原型，并指出这是一个 **one-shot 原型**。
   - 他们建议使用 **AI Studio** 来构建 Agentic 应用，通过向 builder agent 详细描述意图来创建可运行的原型，从而实现分阶段测试和模型聚焦。
- **绕过机器人限制进行播客创建**：鉴于登录时的机器人限制，一位成员询问了如何使用 **NotebookLM** 创建播客。
   - 另一位成员澄清说，驱动 **NotebookLM** 的工具可以通过 **API** 获取，建议在另一个工作流中重建，并手动将报告加载到 **NotebookLM** 中。
- **讨论 Obsidian 与 NotebookLM 的集成**：一位成员分享了一篇关于集成 **NotebookLM**、**Obsidian** 和 **Google Drive** 的文章，链接见[此处](https://www.xda-developers.com/using-notebooklm-obsidian-google-drive-together/)。
   - 另一位成员表示愿意根据对方的使用情况提供更多关于使用 **Obsidian** 的细节。
- **NotebookLM 音频输出平均时长为 8-15 分钟**：一些成员询问了关于使用 **NotebookLM** 生成长音频文件的问题。
   - 另一位成员表示他们的平均输出时长为 *8-10 分钟*，尽管其他成员曾生成过长达 *15 分钟* 的内容。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1399828961021788412)** (156 条消息🔥🔥): 

> `NotebookLM 视频概览限制, Studio UI 变更, 视频生成长度, NotebookLM 逐步推出, 缺失新的 NotebookLM 功能` 


- **Pro 用户每日可获得 20 个视频概览**：一位成员确认，NotebookLM 的 Pro 用户每天可以获得 **20 个视频概览**，这在 [Google 支持文档](https://support.google.com/notebooklm/answer/16213268?hl=en&ref_topic=16175214&sjid=12603864792385823108-NA#:~:text=With%20NotebookLM%2C%20you,daily%20video%20generations.) 中也有说明。
   - 然而，尽管是 Pro 用户，一些用户在访问视频概览功能和更新后的 **Studio UI** 时仍遇到延迟。
- **Studio 界面需要输出排序/筛选功能**：一位用户建议 **Studio 界面**需要具备*排序/筛选输出*的能力以及*全部删除选项*，同时还需要能够**停止正在进行但无法完成的视频生成**。
   - 另一位用户强调，“保存所有笔记到来源”的功能消失了，这可能会导致免费版中 **50** 个来源限制的问题。
- **视频生成时间差异巨大**：用户报告视频生成时间各不相同，一位用户处理一篇《经济学人》文章耗时 **30 分钟**，引发了关于是否使用了 **Veo 3** 的讨论。
   - 一位用户将输出描述为*更像演示文稿而非动画视频*，并指出其倾向于排版设计，适合以文本为主的内容。
- **功能已在德国上线，其他地区进度较慢**：**视频概览功能**已在**德国**的 Pro 账号上线，而包括 Google Ultra 用户在内的许多用户仍在等待推送。
   - Google 确认更新将在下周内逐步推送到所有用户。
- **视频概览缺陷曝光**：用户报告视频概览被限制在 **6-7 分钟**，并且各章节之间存在生硬的过渡。
   - 存在一个 Bug，即视频会无限加载，直到刷新页面。

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1399838809134334052)** (117 条消息🔥🔥): 

> `LM Studio 模型重命名，俄罗斯海岸地震，LM Studio 复制/粘贴对话功能请求，LM Studio 模型陷入时间循环，Qwen 30B 垃圾输出` 


- **俄罗斯海岸海啸预警**：俄罗斯海岸发生 **8.7 级地震**，触发了夏威夷的海啸预警以及美国西海岸的海啸观察预警。
   - 受影响地区的居民被建议密切关注更新，因为海啸可能会在数小时后到达。
- **LM Studio 功能请求：复制粘贴整个对话**：一位用户询问 LM Studio 中是否有复制和粘贴整个对话的功能，这些对话以 **JSON 格式**存储，相对容易操作。
   - 另一位用户提到他们开始编写一个用于提取对话的 Python 应用，但后来分心了，建议其他人在 [feature request channel](https://discord.com/channels/1110598183144399058/1128339362015346749) 中添加功能请求，因为许多人会发现它很有用。
- **LM Studio 模型陷入时间循环**：一位用户报告称，他们的 LM Studio 模型反复引用 **2024 年 2 月 18 日**，即使在询问当前事件时也是如此，并提供了截图作为证据。
   - 另一位用户建议检查 **system prompt** 或 **Jinja template** 中的日期，因为这可能导致模型认为自己处于该特定日期。
- **Qwen 30B 的 Sampler 设置**：一位用户注意到 **Qwen 30B** 经常产生垃圾输出，除非重新处理 prompt。
   - 另一位用户建议尝试官方的 samplers 或提供的设置，看看输出是否有所改善；其中一人指出 Linux 上也存在类似问题，通过更新到实验性驱动程序得以解决。
- **LM Studio MCP 客户端需要资源支持**：用户讨论了在 **LM Studio MCP 客户端**中使用资源的潜在用途，强调了低成本只读引用（如语法指南或动态代码）等用例。
   - 一位用户提到他们使用资源进行发现、文档记录和导航辅助，相比 tool calls，他们更倾向于使用资源来获取快速参考信息，并希望客户端更新能支持 **2025-06-18** 更新的 MCP 规范功能。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1399876515449147545)** (64 messages🔥🔥): 

> `GPU Usage, Strix Halo, Threadripper vs Epyc, Soldered RAM, 9070 XT Performance` 


- **跨 GPU 拆分模型会影响使用率**：在两个 GPU 之间拆分模型时，每个 GPU 的运行**使用率为 50%**，这可能会减少发热和噪音，但这取决于模型层如何拆分以及是否按顺序处理。
   - 4090 与较慢的 3070 配对可能会导致 4090 在等待 3070 完成任务时处于闲置状态，但性能仍有提升，速度从 **8 tok/sec 增加到 32 tok/sec**。
- **Strix Halo APU 评价褒贬不一**：**Strix Halo APU** 的价格似乎定在 64GB 版本约 **$1.6k**，128GB 版本约 **$2k**，但一些成员认为 EPYC 系统由于更大的内存带宽和可升级性，性价比更高。
   - 一位成员对这些设备采用*板载内存 (soldered memory)* 表示遗憾，并将其与最近服务器上的 DIMM 故障进行对比，指向了 [配备 Strix Halo APU 的 Corsair AI Workstation 300](https://www.guru3d.com/story/compact-ai-pc-corsair-ai-workstation-300-with-strix-halo-apu/)。
- **Threadripper 与 Epyc 对决！**：虽然 **Threadripper** 通常被认为是消费者的最佳选择，但由于翻新配件的供应，**EPYC** 可能是一个更便宜的选择，而不像 **Threadripper** 那样往往更贵且更难找到。
   - 一位成员指出 Epyc 更便宜是因为*存在相当大的翻新/二手配件市场*，并指向 [这个 reddit 帖子](https://old.reddit.com/r/LocalLLaMA/comments/1mcrx23/psa_the_new_threadripper_pros_9000_wx_are_still/) 以进一步讨论。
- **板载 RAM 是骗局吗？**：成员们对人们购买带有*板载存储*的 PC 表示困惑，特别是考虑到高昂的价格和有限的内存带宽，例如一台拥有 **128GB** 板载 RAM 和 **256GB/s** 内存带宽的设备售价 **2500€**。
   - 一位用户表示*这就像在请求被骗*，而另一位用户将这一概念比作游戏机，一切都打包在一起，尽管以同样的价格可以组装一台更好的 PC。
- **9070 XT 在性能测试中表现平平**：**9070 XT** 明显慢于 **4070 Ti Super**，一位用户报告称，在其 **4070 Ti Super** 上运行速度为 **7 t/s** 的模型在 **9070 XT** 上仅达到 **3 t/s**；然而，另一位成员认为内存带宽限制可能是原因。
   - 有人指出 CUDA 很好，但也许 Vulkan 也不错。一位成员发现 **5070 Ti** 售价为 **749 欧元**，但第二天价格就跳到了 **1100 欧元**。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1399844353538523288)** (4 messages): 

> `LlamaCloud document agents, LlamaCloud Managed Embeddings, Automated Asset Manager Fund Analysis, LexiconTrail agentic AI systems` 


- **AI Agent 解析财务文档**：利用 **AI 驱动的文档 Agent** 将复杂的财务文档转化为可操作的数据，这些 Agent 可以处理 **10-Ks**、收益报告和监管备案等真实格式；更多信息请见 [LlamaIndex 网络研讨会](https://twitter.com/llama_index/status/1950285220663742516)。
- **LlamaCloud 托管 Embedding**：**LlamaCloud Indexes** 现在拥有托管 Embedding，这意味着你不再需要提供自己的 API Key 来嵌入内容；根据 [这条推文](https://twitter.com/llama_index/status/1950345618779754644)，除了托管向量外，还会为你进行向量嵌入。
- **自动化资产管理基金分析现已推出**：通过这份全面的 Notebook 构建自动化资产管理基金分析，展示了如何处理复杂的财务文档，并使用 **LlamaParse** 将 PDF 转换为结构化 Markdown，从而提取投资分析的可操作见解，详见 [这条推文](https://twitter.com/llama_index/status/1950590734685671931)。
- **LexiconTrail 助力 10 倍速 Agentic AI 系统**：根据 [这篇博客文章](https://twitter.com/llama_index/status/1950662723785850911)，**LexiconTrail** 展示了如何利用具有高级索引能力的 **NVIDIA Small Language Models** 构建 **快 10 倍的 Agentic AI 系统**。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1399848962025984000)** (126 messages🔥🔥): 

> `LlamaCloud PDF 检测问题，Character AI 架构，Neo4j 知识图谱问题，Flowmaker Gemini 2.5 Pro 错误` 


- **LlamaCloud 无法检测 PDF，成员寻求指导**：一位成员报告 **LlamaCloud** 无法检测 **PDF 文件** 并通过 API 进行处理，该成员使用 **n8n** 来简化工作流，并附上了[截图](https://cdn.discordapp.com/attachments/1059201661417037995/1399848961832911031/Screenshot_2025-07-29_at_22.13.16.png?ex=688bcff0&is=688a7e70&hm=b8f51e99fbeae087df203303f7665c4eab8447bb0890b55823fd36074c5ad539&)。
- **关于构建 Character AI 的讨论引发关注**：成员们讨论了如何构建一个对宏大故事有深度理解的 **character AI**，采用经典的 **RAG** 流水线，包括文本分块、embeddings 和向量数据库。
- **Neo4j 的烦恼与图存储过载**：一位成员尝试实现 **Neo4j**，因为他们的简单图存储加载速度*慢得离谱*，但他们的服务器与 **Neo4j 5.x** 不兼容，而 **LlamaIndex** 似乎不支持 **4.x**，且 **Aura** 被服务器代理拦截了。
- **Flowmaker 快速修复 Gemini 2.5 Pro 错误**：一位成员报告了在使用 **Flowmaker** 配合 **Gemini API** 时因模型名称无效而产生的错误，另一位成员迅速指出 [模型名称](https://ai.google.dev/gemini-api/docs/models) 需要包含数字，例如 *gemini-2.5-pro*。
   - 修复代码已[提交](https://github.com/run-llama/flow-maker/blob/aad0f47a81cacba662a07c4f2d70bd3425606e29/src/lib/llm-utils.ts#L19)并迅速部署，解决了该问题，该成员对其快速协助表示感谢。


  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1400095143251677184)** (3 messages): 

> `RAG 调试，稀疏检索 (Sparse retrieval)，语义漂移 (Semantic drift)，分块崩溃 (Chunking collapse)，内存故障 (Memory breakdowns)` 


- **用户通过 MIT 许可的仓库提供 RAG 调试协助**：一位成员提供了一个 **MIT 许可的仓库**，旨在调试棘手的 **RAG 问题**，包括稀疏检索 (sparse retrieval)、语义漂移 (semantic drift)、分块崩溃 (chunking collapse) 和内存故障 (memory breakdowns)。
   - 另一位成员询问是否能分享使用该仓库解决的复杂问题，特别是关于 *sparse retrieval* 和 *semantic drift* 的更多细节。
- **关于特定 RAG 调试问题的查询**：在最初的提议之后，一位社区成员询问了该 MIT 许可仓库所解决的具体复杂问题，重点关注具体案例。
   - 该查询特别要求提供关于该仓库如何处理 **sparse retrieval** 和 **semantic drift** 的详细实例，以寻求比一般性描述更深入的理解。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1400002443323904084)** (5 messages): 

> `专家并行 (EP) vs 张量并行 (TP)，GitHub 上的归并排序问题` 


- **专家并行经验探讨**：一位成员正在寻找 **Expert Parallelism (EP)** 优于 **Tensor Parallelism (TP)** 的案例，并指出根据他们在 **Qwen32B** 和 **Qwen 235B** 上的经验，attention 之后 all-reduce 操作带来的额外通信开销使得 **EP** 性能较低。
   - 他们发现 **EP** 仅对采用 **MLA** 且需要 **DP attention** 的模型有用。
- **寻求归并排序余数问题的帮助**：一位成员在他们的 [RinomXE GitHub 项目](https://github.com/maybeJosiah/RinomXE) 中需要关于归并排序余数问题的帮助。
   - 他们在绘制顺序的余数逻辑上遇到困难，即在超过形状数量前不断翻倍步长的逻辑无法正确排序，并发布了一个用于模拟的 javascript 代码片段以寻求帮助。


  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1399868515422441522)** (17 条消息🔥): 

> `Torch Compile, Triton 代码生成, PTX 代码提取, Inductor 配置, GEMM 自动调优` 


- **从 Torch Compile 中解锁 PTX 和 Triton 代码**：要获取 **PTX 代码**，请使用 `TORCH_LOGS="output_code" python your_code.py` 或访问 `compiled_kernel.asm.keys()` 字典，详见[这篇博客文章](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/?utm_source=chatgpt.com#ttir-triton-ir)。
   - 该字典包含不同中间表示的键，包括 **llir, ttgir, ttir, ptx 和 cubin**。
- **在 Torch Inductor 中绕过非 Triton 代码生成**：要强制为 matmuls 生成 **Triton 代码**，请在 [torch._inductor.config.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L459-L461) 中进行配置，但请注意，并非所有算子默认都会转换为 Triton。
   - 诸如 **max_autotune_gemm_backends="TRITON"** 和 **max_autotune_conv_backends** 等选项可以影响自动调优过程，尽管内置算子通常更快。
- **通过调整 Inductor 配置实现纯 Triton 代码**：为了让 Inductor *仅* 使用 Triton 代码，成员建议修改 `config.py` 和 `utils.py`，特别是 **use_aten_gemm_kernels**, **use_triton_template**, **autotune_fallback_to_aten**, **max_autotune_conv_backends** 和 **max_autotune_gemm_backends** 等设置。
   - 这涉及防止自动调优和回退到预写算子，可能需要探索 **'/tmp/torchinductor_{username}'** 目录。
- **TMA 支持随 Triton 3.4.0 到来**：**TMA (Tensor Memory Accelerator) 支持** 尚未在 Triton 官方版本中提供；用户必须等待 **3.4.0** 版本。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1399872492797038774)** (9 条消息🔥): 

> `直播回顾, 请求已接受` 


- **主播计划进行直播回顾**：一位主播被邀请为社区进行一次[直播回顾](https://www.twitch.tv/)。
   - 主播回应道：*我不确定这是否是确认邮件。等我有空会确认的！我怀疑这不仅仅是加入邮件列表的欢迎信*。
- **请求已接受！**：一名成员表示团队已接受所有请求。
   - 另一名成员确认他们的请求已被接受：*虽然我是个新手，但这是个值得探索的酷东西*。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1400083758618775636)** (2 条消息): 

> `kineto 中的 CUPTI 指标, torch.profiler 指标` 


- **在 kineto 中启用 CUPTI 指标遇到困难**：一名成员询问如何在 **kineto** 中启用 **CUPTI 指标**（可能通过自定义构建），并在 **torch.profiler** 中使用。
   - 他们引用了一个[相关的 pull request](https://github.com/pytorch/pytorch/pull/125685)，但表示这并没有解决他们的问题。
- **torch.profiler 配置**：该成员尝试使用带有特定配置的 **torch.profiler** 来测量算子性能。
   - 他们尝试配置 **experimental_config**，包含 **profiler_metrics** 如 *kineto__tensor_core_insts*, *dram__bytes_read.sum* 和 *dram__bytes_write.sum*，并将 **profiler_measure_per_kernel** 设置为 True。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1399912770396487731)** (6 条消息): 

> `CUDA streams, Megatron-LM, Group GEMM, 纽约黑客松, 初学者黑客松建议` 


- **CUDA Streams 与 GEMM 性能**：一名成员询问在运行 **GEMM 算子** 时使用 **多 CUDA streams** 的优势，特别是在 **Megatron-LM** 和 **cuBLAS multi-stream Group GEMM** 的背景下。
   - 用户质疑其相对于单 stream 的优势，并对开销和有限的线程块数量表示担忧。
- **纽约黑客松**：一名成员询问关于黑客松的信息，另一名成员引导他们前往特定频道获取更多信息。
   - 该黑客松似乎位于纽约市。
- **针对初学者的通用黑客松建议**：一名成员在 X 上分享了[通用黑客松建议的链接](https://x.com/ayushgun/status/1950444463899512960)，并指出这对初学者很有用。
   - 这些建议非常通用，并非专门针对 GPU。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 条消息): 

ali_8366: 这里有蒙特利尔的朋友吗？想约个咖啡聊聊。
  

---


### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/)** (1 条消息): 

vishomaru: 大家好，有人成功使用 AMD GPU Profiler 对 compute shaders 进行过性能分析吗？
  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1399941036360601765)** (3 messages): 

> `AI Hackathon, CuTeDSL Blogpost, Software Pipelining` 


- **AI Hackathon 帖子推广**：一位成员分享了关于 **AI Hackathon** 的 [LinkedIn 帖子](https://www.linkedin.com/posts/nadiveedishravanreddy_ai-hackathon-qwen-ugcPost-7355265897877434369-7RI5?utm_source=share&utm_medium=member_android)。
   - 该活动现在有 **15 位演讲者**，包括来自 **Prime Intellect、Snowflake 和 Jane Street** 的代表。
- **课程推广及明星讲师阵容**：一位成员再次推广了一门课程，提到现在有 **15 位演讲者**，如 **Prime Intellect**、**Snowflake**、**Jane Street (Sylvain Gugger)** 和 **Daniel Han**（[课程链接](https://maven.com/walk-with-code/scratch-to-scale?promoCode=gpumode40)）。
   - 他们鼓励有费用顾虑的人联系讨论潜在的资助。
- **编译器自动化 CuTeDSL 优化**：一位成员分享了一篇 [博客文章](https://veitner.bearblog.dev/let-the-compiler-do-the-work-in-cutedsl/) 和 [代码](https://github.com/simveit/software_pipelining_cute_dsl)，关于在 **H100** 上使用 **CuTeDSL** 进行 **GEMM**，详细介绍了如何让编译器处理预取（prefetching）。
   - 该博客文章解释了 `cutlass.range` 算子的一个实验性参数，用于提示预取，以更简单的代码实现了与手动预取相当的性能。


  

---


### **GPU MODE ▷ #[general-leaderboard](https://discord.com/channels/1189498204333543425/1343002580531417211/1400119615429804166)** (3 messages): 

> `Popcorn-cli DeserializationError, BadCredentialsException on MI300, B200 Timeout Issues, Discord Run Errors` 


- **Popcorn-cli 在 H100 和 A100 上遇到 DeserializationError**：一位用户报告在使用从源码编译的最新 **popcorn-cli** 版本时，在 **H100** 和 **A100** GPU 上出现了 *"DeserializationError | Raw Error: Deserialization failed because the 'libkernelbot' module is not available in the local environment"*。
   - 该错误也影响了 **H100** 的 Discord 运行。
- **MI300 遇到 BadCredentialsException**：该用户在 **MI300** 上还遇到了 *"BadCredentialsException | Raw Error: 401 {"message": "Bad credentials", "documentation_url": "https://docs.github.com/rest", "status": "401"}"* 错误。
- **B200 超时**：该用户在 **B200** 上遇到了 **300s 超时**，而该任务在两周前曾成功完成。
- **Popcorn 开发者正在处理！**：一位成员表示团队已意识到这些问题，并正在积极开发修复程序。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1400147055963541671)** (5 messages): 

> `Benchmarking Explanation` 


- **基准测试说明会议推迟**：成员们协调了一次会议来解释基准测试（benchmarking）过程，但原定会议因出席人数过少而提前结束。
   - 一位成员为睡过头表示抱歉，但确认可以参加后续会议来解释基准测试。
- **睡过头的成员仍然可用**：尽管睡过头了，一位成员确认他们仍然可以解释基准测试过程。
   - 该成员旨在重新安排时间并按计划提供基准测试说明。


  

---

### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1399949797603020840)** (20 条消息🔥): 

> `gmem synchthreads, cp.async.cg vs cp.async.ca, cutedsl ptx wrapper, nvvm wrapper, cutedsl older drivers` 


- **Gmem 传输后的 SyncThreads 救星**：在从全局内存 (**gmem**) 拷贝到共享内存后，手动插入 `synchthreads`（或等效指令）是必要的。
   - 这确保了在参与 **gemm**、reduction 或 scan 等集体计算之前，共享内存中的所有元素都已到达。
- **在 Cutedsl 中控制 Cp.Async 的选择**：一位成员询问如何控制在 **cutedsl** 中使用 `cp.async.cg` 还是 `cp.async.ca`。
   - 建议是编写自定义汇编代码并将其作为拷贝操作提供，尽管这尚未经过测试。
- **Cutedsl 中的 PTX Wrapper 启示**：据一位成员称，cutedsl 中没有 **ptx** wrapper 的 API。
   - 然而，另一位成员分享了一个关于如何实现它的示例代码链接，并表示 *在官方 CuTeDSL 代码中也有相关内容。* ([quack/utils.py](https://github.com/Dao-AILab/quack/blob/main/quack/utils.py#L67))。
- **Nvvm Wrapper 导航说明**：一位成员分享了关于如何编写 **nvvm** wrapper 的链接。
   - 他们分享了 [cutlass repo](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/arch/nvvm_wrappers.py) 的链接作为示例，以及 [cutedsl docs](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_cpasync.html#module-cutlass.cute.nvgpu.cpasync) 的链接。
- **Cutedsl 兼容性疑虑澄清**：一位成员询问在旧驱动程序上使用 **cutedsl** 是否可行。
   - 他们尚未遇到任何问题，但想知道内部测试是否发现了任何问题。


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1400002892735189022)** (1 条消息): 

> `Distributed Training, LLMs, Distributed memory tricks` 


- **Ultrascale Playbook 是极佳的资源**：Hugging Face Spaces 上的 [Ultrascale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) 是 **LLMs** 分布式训练的极佳资源。
- **分布式训练的内存优化**：该手册提供了许多用于训练 LLMs 的分布式内存技巧。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1399836795881132233)** (38 条消息🔥): 

> `GPU Inference vs M3 Ultra, LLMs Offshore with low latency and bad internet, Topological data analysis experts, Speech-LLM models and audio instruction-following capabilities, Manipulating vector embeddings for machine translation` 


- **M3 Ultra 作为本地推理之王**：一位成员建议购买 **M3 Ultra** 进行本地推理，并链接到了一个讨论其 **80 核 GPU** 和 **512GB 内存** 的 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1j43ziq/the_new_king_m3_ultra_80_core_gpu_512gb_memory/)。
   - 另一位成员分享说，由于其他成员没有回应，他们买了一台二手 **M1 16g**。
- **在离岸环境下低延迟运行 LLMs**：一位成员正尝试在网络较差的离岸环境下以低延迟运行 **LLMs**，寻求解决方案。
   - 另一位成员回应说，如果这是他们想做的，花费数百/数千美元是一个可以接受的使用场景。
- **Speech-LLM 语音指令研究**：一位成员表示有兴趣进行开源研究，以提高 **Speech-LLM** 模型的 **语音指令遵循能力**，这对于创建集成语音的可靠用户界面至关重要。
   - 他们指出最新的研究是来自 Microsoft 的 **Alignformer**，但其代码尚未开源，目前正在评估合作兴趣。
- **用于机器翻译的向量嵌入操作**：一位成员计划基于多语言模型在向量空间中操作向量嵌入。
   - 该成员希望获取嵌入并加上两种语言平均向量的差值，然后旋转它们直到新语言平均值附近的损失最小，其他人指出这在 [这篇论文](https://arxiv.org/abs/1309.4168) 中是一个已解决的问题。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1400101077415886860)** (2 条消息): 

> `REST models, Compute cost` 


- **社区模型支付协议构想提出**：一位成员想知道通过 REST 提供服务的社区模型是否可以使用 **402** 响应来引用计算成本并启用客户端自动支付。
   - 他们思考了在这种支付系统中，*single-rail 与 h402 multi-rail 如何影响开放性*。
- **开放性影响**：讨论围绕实施基于 **402** 的支付系统对开放性的影响展开。
   - 提出了关于 single-rail 与 multi-rail 方法的担忧。


  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1399832395791728763)** (17 messages🔥): 

> `In-Context Learning (ICL), Interpretability Tools, Sparse Autoencoders (SAEs), Lucas Critique, Activation Distributions` 


- **ICL 可能会破坏可解释性工具，相关主张引发关注**：一位成员推测 **In-Context Learning (ICL)** 可能会通过将激活值推离其训练分布，从而破坏诸如 **Sparse Autoencoders (SAEs)** 之类的**可解释性工具**。
   - 该成员引用了 **Lucas Critique**（卢卡斯批判），认为干预措施（如对 LLM 进行提示工程）需要基于对这些干预具有不变性的微观基础进行预测，并[分享了一篇论文](https://arxiv.org/abs/2501.00070v1)来支持其观点。
- **SAEs 在 ICL 中面临泛化挑战**：一位成员同意将 **SAEs** 应用于具有显著 **ICL** 的上下文可能会失败，因为稀疏表示在未参与训练的激活分布上泛化效果不佳。
   - 他们澄清说，这个问题并非 **ICL** 特有，而是每当 **SAEs** 应用于与其训练分布不同的激活分布时都会出现。
- **ICL 对激活分布的影响：是否属于 OOD？**：一位成员假设，通过 **ICL** 将模型的行为限制在大分布的一个极小切片中，可能会破坏为无约束情况构建的诊断工具，从而可能导致新颖的内部行为。
   - 另一位成员提出了反驳，认为 **ICL** 可能会将激活推向**分布内 (in-distribution)**，并引用了 **SAE** 特征在上下文中的特定实例上激活的例子，指向了关于 *function vectors/task vectors*（函数向量/任务向量）的论文。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1399902669526663381)** (1 messages): 

> `Model Evaluation Metrics` 


- **调试模型评估指标**：一位成员提出协助调试一个函数，该函数接收单个输入文档和模型预测结果，并返回评估指标。
- **理解函数处理流程**：建议包括理解该函数如何处理数据，以识别潜在问题。


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1399859285504168006)** (1 messages): 

> `Diffusion Models Study Group, Flow Matching, MIT Curriculum` 


- **新扩散模型学习小组启动**：一个为期 **5 个月、由 12 人组成的新学习小组**正在启动，旨在从零开始学习扩散模型，课程基于 **MIT 课程**（[讲义链接](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf)）。
   - 该小组专为 AI 从业者设计，成员包括 CTO、AI 讲师和 AI 研究员。
- **参加关于 Flow Matching 和 PDEs 的免费入门课程**：该学习小组将举办**两场免费入门讲座**，分别是 **8 月 2 日**的（[Flow Matching & Diffusion Models](https://lu.ma/kv8zf6va)）和 **8 月 9 日**的（[PDEs, ODEs, SDEs + A Brief History of Diffusion Models](https://lu.ma/uk6ecrqo)），时间均为 EST 中午 12 点。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1399899213365645353)** (4 messages): 

> `MoE Implementation, grouped_mm, Low Precision Training, Float8 Training` 


- **关于 Grouped GEMM 实现的讨论**：一位成员询问了 **GPT-NeoX** 中支持 **torch._grouped_mm** 的 PR，该功能目前已在 PyTorch 核心库中可用，可能带来性能提升，并特别提到了[这个 MoE 实现](https://github.com/EleutherAI/gpt-neox/blob/d12c771198388980ee054617e537665f044e0584/megatron/model/moe_mlp.py#L221)。
   - 他们表示，对**低精度 MoE 训练**感兴趣的用户可以使用 TorchAO 的单行代码实现。
- **深入研究 PyTorch 的 Grouped GEMM 实现**：一位成员询问了 PyTorch **_grouped_mm** 的底层实现，并要求与 megablocks 的 grouped GEMMs 进行性能对比。
   - 另一位成员指出其底层使用了 **CUTLASS kernel**，并链接到了[相关源代码](https://github.com/pytorch/pytorch/blob/62f98dbb44fb338ba849f93c491ea170af4c187c/aten/src/ATen/native/cuda/GroupMM.cu#L418)。
- **Float8 分块预训练的复兴**：一位成员质疑由于收敛问题，大家对**低精度训练**似乎缺乏兴趣，称其为“*除非性能非常诱人，否则很难推销*”。
   - 另一位成员反驳道，引用了 **DeepseekV3 的 float8 分块 (blockwise) 预训练**以及他们自己在 **FP8 行向 (rowwise)** 上取得的稳定收敛结果，如[这篇 PyTorch 博客文章](https://pytorch.org/blog/accelerating-large-scale-training-and-convergence-with-pytorch-float8-rowwise-on-crusoe-2k-h200s/)所述，实现了约 30-40% 的吞吐量提升。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1400078020387147887)** (7 条消息): 

> `CUDA generalization 论文, TLS handshake EOF 错误, Mojo 包安装, 区域特定访问问题` 


- **分享了 CUDA Generalization 论文**：一名成员分享了一篇[关于超越 CUDA 的泛化论文](https://huggingface.co/papers/2507.14111)。
   - 另一名成员表示感谢，并建议今后将此类内容发布到相应的频道。
- **用户遇到 TLS Handshake EOF 错误**：一名新的 Mojo 用户报告称，在尝试通过 **pixi** 和 **magic shell** 运行 Mojo 项目时遇到了 **TLS handshake EOF error**。
   - 他们指出在安装从 Microsoft Copilot 获取的 **dockerfile** 时存在问题。
- **TLS 握手问题的建议解决方案**：一名成员建议 **TLS handshake issues** 可能与特定区域访问包仓库有关，并提供了一个解决方案，尝试在最新的 nightly 版本中使用 **pixi** 安装新的 `mojo` 包。
   - 建议的命令为：`pixi init my-project -c https://conda.modular.com/max-nightly/ -c conda-forge`，随后执行 `cd my-project` 和 `pixi add mojo`。
- **VPN 无法解决安装问题**：遇到 **TLS handshake EOF error** 的用户报告称，即使使用了 VPN，建议的解决方案也无效。
   - 另一名成员提到，自从几个月前迁移到新主机后，区域问题应该已经解决了。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1399916351182868503)** (41 条消息🔥): 

> `Mojo 外部调用 vs libc, Mojo 到 Python 的开销, 在 Mojo 二进制文件中嵌入 CPython, Python 性能, Mojo 与热循环 (hot loops)` 


- **Mojo 的外部调用引发疑问**：用户询问为什么 Mojo 的 `external_call` 使用特定的函数（如 `KGEN_CompilerRT_IO_FileOpen`）而不是 libc 中的标准 `fopen`，以及这是否是为了安全性。
   - 一名成员澄清说，其中许多是 Mojo 早期功能较弱时的产物，目前并不是修复的高优先级事项，并且 KGEN 命名空间属于 Modular，最终将会开放。
- **Mojo 到 Python 的开销似乎很大**：一名用户发现，从 Mojo 调用 Python 的空操作（no-op）函数明显慢于直接从 Python 调用（1000 万次调用耗时 4.5 秒 vs 0.5 秒），并指出这种差异比 Rust 通过 Pyo3 与 Python 互操作的差异更显著。
   - 其他人补充指出，Mojo 需要启动一个 CPython 进程来执行 Python 函数，从而产生开销，将其与 Rust 到 Python 的互操作进行比较并不对等。
- **二进制文件中嵌入了 CPython**：讨论围绕 CPython 是嵌入在 Mojo 二进制文件中还是 Python 代码被编译展开，这影响了从 Mojo 调用 Python 的性能开销。
   - 澄清了 CPython 是通过 `dlopen libpython` 嵌入的，并维护了一个指向解释器的指针，每次调用都会复用同一个解释器，因此出于性能考虑，不应在热循环（hot loop）中调用它。
- **在 Mojo 中低延迟胜过热循环**：讨论了从 Mojo 调用 Python 的性能影响，特别是在处理 OpenCV 或 Mujoco 机器人模拟等任务的热循环中，强调这样做就像是“往赛车引擎里喷胶水”。
   - 成员们指出，许多快速的 Python 库实际上是带有包装器的 C 库，并且“仅与上下文字典（context dicts）交互就很容易消耗数百个周期”。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1399878169456152708)** (38 条消息🔥): 

> `Aider 网站框架，Deepseek-chat OpenRouter 问题，SWE-bench 排行榜，Aider 在模型训练中的角色，Qwen3 Coder 30B-A3B 发布` 


- **Deepseek-chat 在 OpenRouter 上表现变差**：成员们注意到 **Deepseek-chat** 在 **OpenRouter** 上的表现优于官方 **DeepSeek API**，在架构师模式（architect mode）中作为编辑器模型使用时，它会返回整个函数而不是选择性的 diffs。
   - 建议使用 `aider --model openrouter/deepseek/deepseek-r1` 作为修复方案，因为这能确保使用 [aider/resources/model-settings.yml](https://github.com/Aider-AI/aider/blob/main/aider/resources/model-settings.yml#L548) 中的默认配置，该配置具有 `edit-format: diff` 设置。
- **Aider 可用于训练编程模型**：有人建议 **Aider** 可以通过记录开发工作流中需要代码检查（linting）或撤销操作的地方，来辅助编程模型的训练。
   - 这将利用 Aider 在“谨慎”开发中的应用来提供宝贵的训练数据，尽管评论者并未主张开发者去实现这一功能。
- **Qwen3 Coder 30B-A3B 发布**：分享了一张关于新模型 **Qwen3 Coder 30B-A3B** 的图片。
   - 该图片是发布公告的截图，证实了其真实性。
- **用户遇到 Litellm API 连接错误**：一位用户报告遇到了大量的 `litellm.APIConnectionError: Error parsing chunk: Expecting property name enclosed in double quotes: line 1 column 2` 错误。
   - 这些错误似乎并未影响功能使用。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1400180575096012831)** (3 条消息): 

> `开源模型选择，Aider 的硬件考量，Runpod 额度，R1 模型，Qwen Coder 模型` 


- **开源模型对决：R1 vs Qwen Coder**：一位成员在拥有无限硬件资源的情况下，寻求关于配合 **aider** 使用的最佳开源模型的建议，并考虑测试 **R1** 和 **Qwen Coder** 模型。
   - 该成员提到有 **Runpod 额度**可以消耗，表明有意对这些模型进行实际测试。
- **Llama3 与 Aider 的集成讨论**：成员们就 **Llama3** 与 Aider 的集成及兼容性问题进行了讨论。
   - 一些成员针对现有模型选项提出了一些有用的集成改进建议。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1399838202122207312)** (29 条消息🔥): 

> `LLM 安全对齐研究，AI 对齐博客，使用 Claude 编写 CUDA，Z.AI 54 个开源仓库，论文中的数学讨论` 


- **寻求 LLM 安全对齐研究资源**：一位博士生正在寻求关于跟进当前 **LLM 安全/对齐研究**的建议，特别是优秀的综述论文（survey papers）。
   - 建议包括来自 **AI alignment forum** 的四篇博客，包括 [Thought Anchors: Which LLM Reasoning Steps Matter](https://www.alignmentforum.org/posts/iLHe3vLur3NgrFPFy/thought-anchors-which-llm-reasoning-steps-matter)、[Measuring Beliefs of Language Models During Chain-of-Thought](https://www.alignmentforum.org/posts/a86uAnPykqNtmEbDH/measuring-beliefs-of-language-models-during-chain-of-thought-1) 等。
- **使用 Claude 编写 CUDA 代码的复杂性**：一位成员发现，在 **Claude** 的帮助下编写 **CUDA** 非常复杂，需要规划、深刻的理解和组织。
   - 他们认为，真正的智能测试应该是让一个具备一定 GPU 和 CUDA 知识的 Python 开发者，能否利用 **Claude** 来引导编写 **kernels** 并优化性能，并附带了一张[图片](https://cdn.discordapp.com/attachments/986699377257119794/1400233738369110026/image.png?ex=688be4ca&is=688a934a&hm=1bcb11346477e61edf05cde9751d5e62ee8992a2f64216c07e4a1a8f8fb14cc4)。
- **Z.AI 54 个开源仓库引发关注**：一位成员询问其他人是否看到了新的 **Z.AI 54 个开源仓库**并尝试研究过它们。
   - 未提供关于这些仓库或其具体内容的进一步细节。
- **表达对 Voyager 的喜爱**：一位成员分享了 [Voyager](https://www.youtube.com/watch?v=H0XYANRosVo) 的链接并表达了对它的喜爱，另一位成员立即回应“我也是！”。
   - 另一位成员分享了今天发布的 [Simons Foundation](https://www.youtube.com/playlist?list=PLWAzLum_3a18wO6C7TP8_4XGw4pDxy6G5) 播放列表链接。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1399983269071818853)** (1 条消息): 

> `Qwen3, GPT-4o` 


- **Qwen3 30B 性能媲美 GPT-4o**：一位成员分享了一篇帖子，指出 [Qwen3 30B A3B 2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) 在英文和中文表现上与 **OpenAI GPT-4o** 持平。
- **Qwen3 受到关注**：社区成员对 **Qwen3** 作为语言模型领域强力竞争者的潜力感到兴奋。
   - 早期基准测试表明，它在某些任务中可能提供与 **GPT-4o** 相当的性能，特别是在多语言语境下。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1399833974972350474)** (30 条消息🔥): 

> `Kimi chatbot, Moonshot AI vibe, OpenHands, Training dataset of Kimi, Scale AI` 


- **Kimi Chatbot 以中文回复**：一位用户报告称，尽管用 **English** 提问，**Kimi chatbot** 仍以 **Chinese** 回复，这可能与账号登出有关。
   - 截图显示，虽然回复内容是英文，但推荐的来源和问题却是中文。
- **Kimi 的训练数据倾向于社交媒体**：一位成员开玩笑说 **Kimi's training dataset** 似乎包含了 **Instagram** 和 **TikTok** 的评论，并认为这就是它表现出色的原因。
   - 他们链接到了 [Kimi on OpenHands v0.50.0k2](https://github.com/All-Hands-AI/OpenHands/releases/tag/0.50.0k2) 来支持这一说法。
- **Moonshot AI 的氛围**：一位成员表示 *moonshot got the best vibe no cap*（Moonshot 的氛围确实是最好的），另一位成员也同意社区需要一些竞争。
   - 他们链接了一篇关于 AI 社区氛围检测的 [X post](https://x.com/crystalsssup/status/1944287779896328668)。
- **Scale AI 创始人 Alexandr Wang**：一位成员提到 **Alexandr Wang** 是 **Scale AI** 的**创始人兼 CEO**，这是一家数据基础设施公司。
   - 他们指出 **Scale AI** 提供训练数据、标注和评估服务，这些对于开发机器学习模型至关重要。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1399839990069854348)** (25 条消息🔥): 

> `Lume vs Suna, Manus' Comic Creation, The Future of Manus` 


- **成员辩论：Lume 略胜 Suna**：成员们辩论了 **Lume** 和 **Suna** 这两个 Agent 系统优劣；一位成员表示 *Lume did a much better job*，在编写特定代码时表现更好且错误更少，但也承认可能没有对 Suna 使用正确的 Prompt。
   - 该成员指出，由于某些任务的成本过高，他们无法将其与 **Manus** 进行比较。
- **Manus 漫画创作：美中不足？**：一位成员建议 **Manus** 的漫画创作功能很不错，但仍有改进空间。
   - 另一位成员表示服务质量正在下降，对免费用户有严格限制，并质疑 **Manus is dead**。
- **乐观的 AI vs 怀疑的人类：Manus 的未来**：一位成员询问 AI 对 **Manus** 未来的看法，AI 回复道 *I think the future of Manus is bright*。
   - 另一位成员表示怀疑，理由是 **OAI** 和 **Google** 已经发布了 Agent 模式。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1399834491513475083)** (22 条消息🔥): 

> `MCP Server 安全性, 基于 LLM 和 MCP 的 BDD 测试, CursorTouch 与 Claude 的 Windows-MCP 问题, FastMCP 工具选择, 托管 MCP Server` 


- **MCP Server 需要用户上下文隔离**：一位用户正在寻求澄清，即单个云端部署的 **MCP Server 实例**是否需要额外的层来进行**用户上下文隔离**，以防止多个客户端同时访问时在唯一会话之间共享数据，并引用了 [issue #1087](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1087) 和 [MCP Docs](https://modelcontextprotocol.io/docs/tutorials/use-remote-mcp-server#understanding-remote-mcp-servers)。
- **Cursor 可连接，Claude 失败**：一位用户报告称，已成功将 **MCP Server 部署到 EC2**，并配置了正确的 **SSL 证书和域名设置**，但他们只能通过 **Cursor** 连接，而无法通过 **Claude Desktop** 连接。
- **Cucumber, BDD 与 LLM 联手！**：一位成员分享了一个基于**行为驱动开发 (BDD)** 的侧边项目，该项目已达到生产就绪状态；他们还附带了一张[解决方案架构图](https://cdn.discordapp.com/attachments/1312302100125843479/1399854833565044756/bael.jpeg?ex=688bd568&is=688a83e8&hm=2e86139e9f117265cd7cbef2afcc1a23a34a091e79402df9a0e051261231c695&)。
- **Windows-MCP State Tool 报错问题**：一位用户在使用 Claude Desktop 中的 CursorTouch Windows-MCP 时遇到困难，因为 **State Tool 完全无法工作**并提示：*Error calling tool 'State-Tool': 'Taskbar'*。
- **FastMCP 动态工具选择**：一位用户询问 **FastMCP** 是否包含在服务器定义了多个工具时，在客户端**动态且自动选择工具**（例如：数学、网页搜索、RAG、数据解释器）的逻辑。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1399914906672959628)** (2 条消息): 

> `DeepTrail, Deepsecure, 开源认证, AI Agent 委托层, 安全多 Agent 工作流` 


- ****DeepTrail** 为 AI Agent 授权推出 **Deepsecure****：由 Berkeley SkyDeck 支持的 **DeepTrail** 正在开发 **Deepsecure**，这是一个面向 AI Agent 的开源认证与委托层，能够通过 [GitHub](https://github.com/DeepTrail/deepsecure) 以极少的代码集成授权、Agent 到 Agent 的委托、策略执行和安全代理。
- **探索 **Deepsecure** 的架构与安全多 Agent 工作流**：**Deepsecure** 的架构具有分片密钥设计、网关/代理、独立的控制/数据平面、策略引擎以及用于 Agent 间委托的 macaroons，详见其[技术概览](https://github.com/DeepTrail/deepsecure/blob/dev/docs/design/deepsecure-technical-overview.md)。
- ****Deepsecure** 与 Langchain/LangGraph 的集成示例**：**Deepsecure** 与 Langchain/LangGraph 的集成示例包括：*安全多 Agent 工作流* ([代码链接](https://github.com/DeepTrail/deepsecure/blob/dev/examples/05_langchain_secure_tools.py))、*委托工作流* ([代码链接](https://github.com/DeepTrail/deepsecure/blob/dev/examples/09_langchain_delegation_workflow.py))、*高级委托模式* ([代码链接](https://github.com/DeepTrail/deepsecure/blob/dev/examples/11_advanced_delegation_patterns.py)) 以及 *平台 Agent 引导* ([代码链接](https://github.com/DeepTrail/deepsecure/blob/dev/examples/12_platform_expansion_bootstrap.py))。
- **具有社区功能和市场的高级目录**：一位成员开始开发一个旨在成为“具有社区功能、并演变为一键安装甚至市场的优质目录”的项目，访问地址为 [protocoldepot.dev](https://protocoldepot.dev/)。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1399896190509912094)** (18 messages🔥): 

> `DSPy 可学习参数提案，使用 f-strings 的 Signature 实现，DSPy 对比 GEPA` 


- **DSPy 可学习参数提案引发关注**：成员们讨论了在 DSPy 中添加可学习参数（`dspy.variable` 或 `dspy.parameter`）的提案，并[创建了一个 issue](https://github.com/stanfordnlp/dspy/issues/8593) 来收集想法和用例。
   - 一位成员将其描述为一个“非常亮眼的提案”，希望允许“模板作为参数/变量”，以便优化器可以输出最优提示词，以及模板变量的放置。
- **F-Strings 导致 Signature 实现问题**：一位成员寻求帮助，希望使用 f-string 实现 Signature，以验证代码是否符合描述。
   - 另一位用户建议不要采用这种方法，并建议“将参数描述放在 `dspy.InputField()` 中”。
- **DSPy 在提示词优化对决中对阵 GEPA**：一位成员提到了一段 YouTube 视频，其中将 **DSPy** 与 **GEPA** 进行了对比，其犀利观点是“DSPy 优化你给出的提示词；GEPA 进化出你从未想象过的提示词”，并链接了 [YouTube 视频](https://www.youtube.com/watch?v=o6RbVPFOslg)。
   - 该成员提议将 **MIPRO** 转变为 DSPy 的“反思性、遗传式前沿引擎”，以生成并维护提示词的 Pareto-frontier，旨在反驳该 YouTuber 的观点。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1400072601677856810)** (15 messages🔥): 

> `游戏用途下 AMD 对比 Nvidia，Qwen 代码模型发布，RLVR 讨论` 


- **AMD：游戏与开拓 AI 新路径的选择**：一位成员建议在游戏方面购买 **7900XT** 而非 **9070**，并搭配 **7800X3D** 而非 **9900X**，同时指出 AMD 在消费级 AI 方面的可用性以及潜在的长期社区利益。
   - 他们链接了一条 [推文](https://x.com/Teknium1/status/1950596567968477382) 来支持其论点。
- **Qwen 发布代码模型并开始“思考”**：一位成员宣布即将在 [Hugging Face](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507) 上发布 **Qwen3-30B-A3B-Thinking-2507** 代码模型。
   - 该 Hugging Face 模型链接表明这是一个用于代码生成的新工具。
- **Nvidia 的 RLVR：它真的是 RL 算法吗？**：一位成员质疑 **RLVR**（Reinforcement Learning, Virtual Reality）是否应被归类为强化学习算法，并链接了一条引发讨论的 [NVIDIA 推文](https://fxtwitter.com/NVIDIAAIDev/status/1950279130450444670)。
   - 另一位成员 teknium 表示：“RLVR 并不是一种 RL 算法，它只是 RL 的一个目标”。


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1400127686264881253)** (3 messages): 

> `MRC 模型对比，暑期学校频道请求，高级软件工程师远程职位` 


- **成员询问暑期学校频道**：一位新成员正在询问几周前举行的 **summer school** 的专用频道。
- **MRC 模型对比策略**：一位成员询问是将自定义的 **MRC 模型** 与大型预训练模型的 **zero-shot 性能** 进行对比，还是在相同数据集上对大型模型进行 **fine-tune** 以进行更公平的比较。
- **发布长期远程高级软件工程师职位**：发布了一个高级软件工程师职位，月薪 **$2K**，为长期远程合同，工作地点位于**非洲**或**美洲**。
   - 该职位要求具备 **Ruby on Rails**、**Node.js**、**C#/.NET**、**Python**、**Java** 或类似经验，以及母语或接近母语水平的英语沟通能力。


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1400077240896590014)** (1 messages): 

> `Langchain-Cohere 引用模式，langchain_cohere.ChatCohere` 


- **引用选项在 Langchain-Cohere 上不起作用**：一位成员在尝试使用 `langchain_cohere.ChatCohere` 上的 `citation_options` 更改引用模式时遇到了问题。
   - 该成员询问是否有任何隐式方式传递引用选项，因为 `langchain_cohere.ChatCohere` 不显式接受它。
- **Langchain-Cohere 仓库状态：未维护？**：一位成员询问 [langchain-cohere 仓库](https://github.com/langchain-ai/langchain-cohere) 是否为官方仓库。
   - 他们注意到该仓库在过去几个月中没有更新，并询问“这里是否欢迎 pull requests”。


  

---

### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1400056898665058364)** (6 条消息): 

> `AI Safety, LLM Bias Mitigation, GPU Kernel Optimization` 


- **统计学专业学生在 AI 领域寻求安全空间**：一名统计学硕士生表达了对 **ML research** 的兴趣，特别是 **technical AI safety** 方向，并对研究合作持开放态度。
- **博士生专注于伦理 LLMs**：奥地利 JKU Linz 的一名博士生正致力于 **减轻 LLMs 中的社会偏见**。
   - 他们的其他兴趣包括 **生成模型的归因（attribution）、AI 生成文本检测以及领域自适应（domain adaptation）**，并希望与从事特定领域 LLM 实际伦理问题研究的人士建立联系。
- **RAG 与图谱助力毕业生增长**：慕尼黑工业大学的一名应届硕士毕业生正在通过个人项目积累 **RAG**、**知识图谱（knowledge graphs）**及新编程语言的经验。
   - 他们希望获得研究经验，参与项目协作，并结识志同道合的人以紧跟新技术趋势。
- **Ali 擅长自回归加速**：一位名为 Ali 的成员正在研究 **在 Triton/CUDA 中为自回归模型优化 GPU kernels**。
   - 他们非常乐意交流底层 GPU 编程相关话题。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1399896806522884186)** (2 条消息): 

> `LoRA-style adapter in Torchtune, Merged weights in Torchtune` 


- **用户请求在 Torchtune 中支持 LoRA 风格适配器**：一位用户询问 Torchtune 是否支持 **LoRA 风格适配器**，特别是那种保留精确前向计算路径且不改变计算成本，但冻结原始模型权重并通过额外的可训练层应用更新的适配器。
   - 他们正在寻找 **额外的可训练层**。
- **Torchtune 在使用适配器训练后合并权重**：一位用户分享了关于 **端到端工作流** 的 [Torchtune 文档链接](https://docs.pytorch.org/torchtune/0.6/tutorials/e2e_flow.html)，强调 Torchtune 支持使用适配器进行训练并随后将权重合并回去。
   - 他们正在询问有关合并权重的问题。


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1400146555914293331)** (2 条消息): 

> `ACL Paper Award, Glianorex finetunes` 


- **ACL 论文获奖**：一位成员分享了他们刚刚获奖的 **ACL 论文**，链接见 [此处](https://aclanthology.org/2025.acl-long.266/)。
- **Glianorex 微调版本发布**：一位成员询问 **微调版本（finetunes）** 是否公开，并抱怨他们的 *Glianorex 让他痛苦不堪，而医生也无能为力*。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1400165808369438871)** (2 条消息): 

> `Certificate Declaration Form` 


- **证书声明表单需要完成**：提醒一位成员尚未完成证书声明表单。
   - 工作人员确认 *遗憾的是，我们从未收到您的证书声明表单*。
- **仍需提交证书表单**：工作人员重申尚未收到证书声明表单。
   - 该成员此前已被告知其表单缺失。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1399831422461804604)** (2 条消息): 

> `Diffusion Models Study Group, MIT Diffusion Models Curriculum, Flow Matching, Generative AI, AI Education` 


- **在新的学习小组中从零开始学习 Diffusion Models**：一个新的学习小组正在启动一个为期 **5 个月**、共 **12 人** 的项目（**每周 2-4 小时**），该项目基于 [MIT 的课程大纲](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf) 来学习 Diffusion Models，这已成为 Generative AI 的核心架构。
   - 前两次介绍性课程免费并向非成员开放：8 月 2 日关于 *Flow Matching & Diffusion Models*，8 月 9 日关于 *PDEs, ODEs, SDEs + Diffusion Models 简史*（[链接见此](https://lu.ma/kv8zf6va)）。
- **AI Scholars 宣布成立新的 Diffusion Models 学习小组**：AI Scholars 正在启动一个 Diffusion Models 学习小组，已确认的成员包括 AI 电影工具的 CTO、AI 艺术讲师、2 名 LLM 讲师和 2 名全职 AI 研究员。
   - 该项目特点包括同行引导的会议、导师问答、动手实践项目、真实研究论文研讨，以及一个紧密互信的小组环境。每周形式为 2 小时直播课 + 2 小时自学，学生轮流授课。


  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1399993952987512853)** (1 条消息): 

> `部署自定义语言模型，Hugging Face 部署，用于用户查询的 GUI` 


- **寻求云端部署策略**：一位用户询问如何将使用自定义 PDF 文件夹训练的语言模型部署到云端供公众使用，特别是寻求一个用于用户查询的简单 GUI。
   - Nomic 表示企业版方案（Enterprise Plan）并不适合，用户想知道 **Hugging Face 部署** 是否可以作为替代方案。
- **企业版方案不适用**：Nomic 指出企业版方案不符合用户的需求。
   - 用户正在探索替代部署策略，例如 Hugging Face，以使其语言模型可供访问。