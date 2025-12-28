---
companies:
- ai21-labs
- mistral-ai
- alibaba
- openai
- amd
- anthropic
- hugging-face
date: '2025-03-07T05:50:14.781495Z'
description: '以下是为您翻译的中文内容：


  **AI21 Labs 推出了 Jamba 1.6**，被誉为**最适合私有企业部署的开源模型**，在 **Arena Hard** 等基准测试中表现优于 **Cohere、Mistral
  和 Llama**。**Mistral AI** 发布了一款最先进的**多模态 OCR 模型**，具备多语言和结构化输出能力，并支持私有化本地部署。**阿里巴巴
  Qwen（通义千问）**推出了 **QwQ-32B**，这是一款拥有 **320 亿参数**的开源权重推理模型，具有极高的性价比，并在基准测试中展现出极具竞争力的评分。**OpenAI**
  发布了 **o1** 和 **o3-mini** 模型，具备流式传输（streaming）和函数调用（function calling）等高级 API 功能。**AMD**
  推出了 **Instella**，这是在 **AMD Instinct MI300X GPU** 上训练的开源 30 亿参数语言模型，旨在与 **Llama-3.2-3B**
  等模型竞争。**阿里巴巴**还发布了 **Babel**，这是一系列开源多语言大语言模型，其性能与 **GPT-4o** 相当。**Anthropic** 推出了
  **Claude 3.7 Sonnet**，进一步增强了推理和提示词工程（prompt engineering）能力。'
id: 2ac42108-f069-4ad9-b095-330b2a12a6aa
models:
- jamba-1.6
- mistral-ocr
- qwq-32b
- o1
- o3-mini
- instella
- llama-3-2-3b
- gemma-2-2b
- qwen-2-5-3b
- babel-9b
- babel-83b
- gpt-4o
- claude-3-7-sonnet
original_slug: ainews-not-much-happened-today-3137
people: []
title: 今天没发生什么特别的事。
topics:
- multimodality
- ocr
- multilinguality
- structured-output
- on-prem-deployment
- reasoning
- benchmarking
- api
- open-source
- model-training
- gpu-optimization
- prompt-engineering
- function-calling
---

<!-- buttondown-editor-mode: plaintext -->**平静的一天。**

> 2025年3月6日至3月7日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 社区（**227** 个频道，**7886** 条消息）。为您节省了约 **777 分钟** 的阅读时间（以每分钟 200wpm 计算）。您现在可以在 AINews 讨论中标记 [@smol_ai](https://x.com/smol_ai) 了！

[Mistral OCR](https://mistral.ai/fr/news/mistral-ocr) 和 [Jamba 1.6](https://www.ai21.com/jamba/) 表现接近。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录** 和 **频道摘要** 已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

**模型发布与更新**

- **AI21 Labs 推出了 Jamba 1.6**，声称它是**企业私有化部署的最佳开源模型**，在 **Arena Hard** 等关键基准测试中超越了 **Cohere、Mistral 和 Llama**。它在速度和质量上与领先的闭源模型不相上下，并已在 **AI21 Studio** 和 [@Hugging Face](https://twitter.com/AI21Labs/status/1897657953261601151) 上线。
- **Mistral AI 发布了一款 SOTA 多模态 OCR 模型** [@scaling01](https://twitter.com/scaling01/status/1897695665871872427)。[@sophiamyang](https://twitter.com/sophiamyang/status/1897713370029068381) 宣布了 **Mistral OCR**，强调了其 **SOTA 文档理解能力**、**多语言和多模态能力**以及**极快的速度**。它提供 **doc-as-prompt**、**结构化输出**，并支持**本地部署**。其 [博客文章](https://twitter.com/sophiamyang/status/1897716142401060867) 中提供了基准测试和示例，包括 [多语言能力](https://twitter.com/sophiamyang/status/1897715804042338327)、[从 PDF 中提取数学公式](https://twitter.com/sophiamyang/status/1897715242936713364) 以及 [将文本和图像提取为 Markdown](https://twitter.com/sophiamyang/status/1897713540506824954)。[@sophiamyang](https://twitter.com/sophiamyang/status/1897716870175682847) 指出该项目在 **Hacker News 上排名第一**。
- **阿里巴巴 Qwen 发布了 QwQ-32B**，这是一款**开源权重推理模型**，声称智力接近 **DeepSeek R1** 和 **OpenAI o1 mini**，而参数量仅为 **32B**，且**成本效益高（每百万 token 0.20 美元）**。它在 **Apache 2.0** 协议下发布于 [@Hugging Face](https://twitter.com/_philschmid/status/1897556185126932750)。[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1897701015803380112) 报告的初步评估显示，**QwQ-32B** 在 **GPQA Diamond 上得分为 59.5%（落后于 DeepSeek R1 的 71% 和 Gemini 2.0 Flash 的 62%）**，但在 **AIME 2024 上得分为 78%（领先于 DeepSeek R1）**。[@awnihannun](https://twitter.com/awnihannun/status/1897394318434034163) 展示了 **QwQ-32B 在搭载 MLX 的 M4 Max 上运行**的情况，并指出其 **8k token 的思考过程**。[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1897422055282500054) 认为 **QwQ 的新模型**在本地运行效果与 **R1** 相当。[@reach_vb](https://twitter.com/reach_vb/status/1897686816037167394) 宣布 **QwQ 32B 已部署在 Hugging Chat 上**。
- **OpenAI 在 API 中向开发者发布了 o1 和 o3-mini**，适用于所有付费层级，支持**流式传输（streaming）、函数调用（function calling）、结构化输出（structured outputs）、推理力度（reasoning effort）、Assistants API、Batch API 和视觉能力（仅限 o1）** [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1897414494286176333)。[@goodside](https://twitter.com/goodside/status/1897412604894789692) 指出 **ChatGPT Code Interpreter 在 4.5 和 o3-mini 中均可工作**，并认为 **o3-mini 获得 Code Interpreter 是一个重大进展** [@goodside](https://twitter.com/goodside/status/1897418513062744200)。
- **AI21 Labs 推出了 Jamba 1.6 聊天模型，拥有 94B 激活参数和 398B 总参数** [@reach_vb](https://twitter.com/reach_vb/status/1897658668398555468)。
- **AMD 推出了 Instella**，这是一系列**完全开源、SOTA 的 3B 参数语言模型**，在 **AMD Instinct MI300X GPU** 上训练，性能超越了现有的完全开源 3B 模型，并可与 **Llama-3.2-3B、Gemma-2-2B 和 Qwen-2.5-3B** 竞争 [@omarsar0](https://twitter.com/omarsar0/status/1897642582966165523)。
- **阿里巴巴在 Hugging Face 上发布了 Babel**，这是一系列**开源多语言 LLM**，包含 **Babel-9B** 和 **Babel-83B** 变体，性能优于同类开源 LLM，并在某些任务上与 **GPT-4o** 表现相当 [@_akhaliq](https://twitter.com/_akhaliq/status/1897483872214077749)。
- **Anthropic 发布了 Claude 3.7 Sonnet**，增加了推理能力，并更新了用于 prompt engineering 的 workbench，具有 tool use、extended thinking 和 prompt sharing 等功能 [@AnthropicAI](https://twitter.com/AnthropicAI/status/1897696420293230989)，[@alexalbert__](https://twitter.com/alexalbert__/status/1897696773151343103)。

**工具与应用**

- **Elysian Labs 发布了 Auren**，这是一款旨在改善人机交互的 iOS 应用，侧重于情商、自主性（agency）和正向激励，而非仅仅是智能 [@nearcyan](https://twitter.com/nearcyan/status/1897466463314936034)。Beta 测试者的反馈被描述为“超现实”且可能“救命” [@nearcyan](https://twitter.com/nearcyan/status/1897470058768875704)。该应用每条消息使用多个模型，定价为每月 19.99 美元，包含 2,500 条消息 [@nearcyan](https://twitter.com/nearcyan/status/1897470389418414219)。[@nearcyan](https://twitter.com/nearcyan/status/1897514277705294057) 强调了该应用的复杂性，指出它不仅仅是“聊天气泡里的 LLM”。
- **Hugging Face 推出了 Diffusion Self-Distillation 应用**，利用 FLUX 实现零样本（zero-shot）自定义图像生成，类似于 DreamBooth 但无需训练，适用于角色一致性和场景重光照（scene relighting）等任务 [@_akhaliq](https://twitter.com/_akhaliq/status/1897496170358006179)。
- **Hugging Face 发布了 PDF Parsers Playground**，一个用于实验开源 PDF 解析器的平台 [@_akhaliq](https://twitter.com/_akhaliq/status/1897482594117206376)。
- **_philschmid 创建了一个 CLI，用于与连接到 Google Search 的 Google DeepMind Gemini 2.0 Flash 进行对话** [@_philschmid](https://twitter.com/_philschmid/status/1897397749395693626)。
- **OpenAI 发布了 ChatGPT for macOS**，允许 Plus、Pro 和 Team 用户直接在 IDE 中编辑代码 [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1897700857833193955)。
- **Perplexity AI 的 Mac 应用现已支持实时语音模式**，允许后台监听并通过快捷键 **Cmd + Shift + M** 进行交互 [@AravSrinivas](https://twitter.com/AravSrinivas/status/1897408183620264028)。
- **LangChainAI 发布了 OpenCanvas**，类似于 OpenAI 的工具，但兼容所有模型 [@_philschmid](https://twitter.com/_philschmid/status/1897405585118912618)。
- **RisingSayak 发布了一个用于视频数据清洗的镜头分类器（shot categorizer）**，声称其速度极快（在 CPU 上 <1s）且开源 [@RisingSayak](https://twitter.com/RisingSayak/status/1897590118736957442)。

**研究与概念**

- **_philschmid** 分享了关于 **ReAct Agents 在压力下** 的基准测试，评估了在扩展领域和工具时的性能，发现 **Claude 3.5 sonnet、o1 和 o3-mini 在需要 3 次以上工具调用的任务中表现优于 gpt-4o 和 llama-3.3-70B**，并且更多的上下文和工具可能会降低性能 [@_philschmid](https://twitter.com/_philschmid/status/1897688288896471546)。
- **ArtificialAnlys** 提供了对 **阿里巴巴 QwQ-32B 模型** 的分析，在 **GPQA Diamond 和 AIME 2024** 等基准测试中将其与 **DeepSeek R1** 和 **Gemini 2.0 Flash** 进行了对比 [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1897701015803380112)。
- **omarsar0** 总结了一篇关于 **实现自我改进推理者的认知行为 (Cognitive Behaviors that Enable Self-Improving Reasoners)** 的论文，指出 **验证 (verification)、回溯 (backtracking)、子目标设定 (subgoal setting) 和逆向链接 (backward chaining)** 是 LM 成功解决问题的关键，并指出 **Qwen-2.5-3B** 自然地展现了这些行为，以及引导 (priming) 和预训练行为放大 (pretraining behavior amplification) 的影响 [@omarsar0](https://twitter.com/omarsar0/status/1897732423963885637)。
- **polynoamial** 在 AI Agent 兴起的背景下，强调了 **Richard Sutton 的“苦涩教训” (Bitter Lesson)**，即随着数据和算力扩展的通用方法最终会在 AI 领域胜出 [@polynoamial](https://twitter.com/polynoamial/status/1897693005601292491)。
- **lateinteraction** 讨论了在构建智能软件时，处于合适抽象层级的 **声明式语言 (declarative languages)** 的力量，建议将编译器作为使特定问题系统实现“随数据和算力扩展”的一种方式 [@lateinteraction](https://twitter.com/lateinteraction/status/1897699917801701512)。他们还思考了从 **ChatGPT** 到 **Copilot/Cursor** 再到 **DSPy & Parsel** 的软件开发光谱，暗示了一个具有更高层级、可组合规范的未来 [@lateinteraction](https://twitter.com/lateinteraction/status/1897442159789531504)。
- **iScienceLuvr** 分享了一篇关于通过“软归纳偏置 (soft inductive biases)”来 **解释深度学习中的泛化行为** 的论文 [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1897619075364487338)。
- **TheTuringPost** 讨论了为什么 **AI 推理测试不断失败**，强调了 **古德哈特定律 (Goodhart's Law)** 以及对 **动态和自适应基准测试** 的需求，这些测试应涵盖数学和编程之外的常识推理、因果推理和伦理 [@TheTuringPost](https://twitter.com/TheTuringPost/status/1897454185656005041)。
- **omarsar0** 讨论了 **AI 驱动的 IDE** 的演进以及 Agent 能力如何使工作流中心化，从而提高生产力 [@omarsar0](https://twitter.com/omarsar0/status/1897700328071385298)。
- **cloneofsimo** 讨论了 RL 时代 **flops/watt** 的重要性以及 **DiLoCo** 的改进 [@cloneofsimo](https://twitter.com/cloneofsimo/status/1897557416117686772)。

**行业与商业**

- 据报道，**Figure AI** 是**二级市场中第 6 位最受追捧的公司** [@adcock_brett](https://twitter.com/adcock_brett/status/1897691903493279902)。
- **ArtificialAnlys** 祝贺 **Together AI**、**Fireworks AI**、**hyperbolic labs** 和 **GroqInc** 推出了 **serverless endpoints** 并提供了实时性能基准测试 [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1897701018231881772)。
- 来自 Hugging Face 的 **ClementDelangue** 讨论了**前 50 名 GenAI 消费者应用的变化**，指出尽管消费者应用在增长，Hugging Face 仍位列第 13 位 [@ClementDelangue](https://twitter.com/ClementDelangue/status/1897735590608609546)。他还强调了**学术界在使 AI 成为积极力量方面的作用**，并重点介绍了 **Hugging Face 上的 Academia Hub** [@ClementDelangue](https://twitter.com/ClementDelangue/status/1897666379823669667)。
- **SakanaAILabs** 正在招聘 **Software Engineers**，在日本利用 LLM 和 AI agents 开发 AI 应用 [@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1897447888814494110)。
- **DeepLearningAI** 正在提供 **Data Analytics Professional Certificate** 课程项目 [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1897723855835189556)，以及一门关于使用 LlamaIndex 构建 **agentic document workflows** 的新课程 [@jerryjliu0](https://twitter.com/jerryjliu0/status/1897668522425393509)。
- **jeremyphoward** 推广了 **FastHTML**，建议采用一种简单的、单语言、单文件的开发方法 [@jeremyphoward](https://twitter.com/jeremyphoward/status/1897431400359526557)。
- **matanSF** 宣布了 **FactoryAI 与 OpenAI 的合作伙伴关系**，旨在通过单一平台的人机协作构建未来软件 [@matanSF](https://twitter.com/matanSF/status/1897694460592754829)。
- **togethercompute** 正在为生产负载构建一支**世界级的 kernels 团队**，并发布了 **ThunderMLA**，一个快速的 MLA decode kernel [@togethercompute](https://twitter.com/togethercompute/status/1897703705790542137)。
- **mervenoyann** 注意到**具有合规性的企业级开发工具**市场正在增长，并提到了 **Dust** 和 **Hugging Face Enterprise Hub** 作为例子 [@mervenoyann](https://twitter.com/mervenoyann/status/1897397563990663651)。

**观点与讨论**

- **scaling01** 质疑了 **Mistral OCR 发布版在编程方面**的效用，认为其落后于 **4o 和 o3-mini**，并好奇它是否主要用于“生成 greentexts” [@scaling01](https://twitter.com/scaling01/status/1897590986278117758)。
- **ajeya_cotra** 询问关于 **Claude Plays Pokemon** 的**定性分析**，希望了解其成功、失败和技能差距，以及它玩起来是否像某个特定年龄的典型孩子 [@ajeya_cotra](https://twitter.com/ajeya_cotra/status/1897458906001231971)。
- **cognitivecompai** 索要 **MistralAI** 模型的 **torrent 磁力链接** [@cognitivecompai](https://twitter.com/cognitivecompai/status/1897722351631925320)，并批评 **Cursor AI 和 Windsurf AI** 缺乏本地模型支持，推荐使用 **continuedev 和 UseCline** 代替 [@cognitivecompai](https://twitter.com/cognitivecompai/status/1897598405884408261)。他们还对 **NVIDIA GeForce 5090 的供应情况**表示沮丧 [@cognitivecompai](https://twitter.com/cognitivecompai/status/1897586581302645080)。
- **ID_AA_Carmack** 讨论了**垄断**的本质以及摆脱垄断的挑战，主张建立拥有**强大反垄断法的自由市场** [@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1897678635147911393)。他还反思了 **Seymour Cray 的工程方法**，以及随着项目成熟适应增量变化的必要性 [@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1897671486229414017)。
- **francoisfleuret** 为“左翼主义”辩护，认为自由市场的固定点可能是“一团糟”，且财富积累可能是不稳定的 [@francoisfleuret](https://twitter.com/francoisfleuret/status/1897654935400927373)。
- **mmitchell_ai** 对**用于战争的 AI Agent** 可能导致失控的导弹危机表示担忧，并质疑防止 AI 部署自主导弹是否仍是一个讨论点 [@mmitchell_ai](https://twitter.com/mmitchell_ai/status/1897714132297908679)。
- **soumithchintala** 与 OpenAI 团队分享了一份笔记，表达了与 AI 发展中“顺从的学生，而非革命者”相一致的观点，强调了为科学家选择正确问题的重要性，并指出 AI 目前的方向可能与自主突破背道而驰 [@soumithchintala](https://twitter.com/soumithchintala/status/1897643753906512168)。
- **DavidSHolz** 认为编程 Agent 将“尽快占据软件工程总预算的一半” [@DavidSHolz](https://twitter.com/DavidSHolz/status/1897450419548516685)。
- **abacaj** 询问关于 **QwQ** 模型的氛围，是“**刷榜（benchmark maxxing）还是好模型？**” [@abacaj](https://twitter.com/abacaj/status/1897645343233241497)。
- **nearcyan** 认为在未来，人类的大部分社交互动将是与 AI 而非其他人进行的 [@nearcyan](https://twitter.com/nearcyan/status/1897469936190324943)，并认为 **Auren** 和 **Seren** 鼓励健康的选择和社交 [@nearcyan](https://twitter.com/nearcyan/status/1897469595751211104)。
- **HamelHusain** 质疑为什么没有 **OAuth 网关让用户使用自己的 LLM API tokens** 以实现更简单的集成 [@HamelHusain](https://twitter.com/HamelHusain/status/1897486751696085093)。

**梗/幽默**

- **dylan522p** 讲了一个关于 **AI 机器人到 2035 年杀死 90% 人类**的未来主义笑话，剩下的公司是 **Marvell 和中国的 AICHIP Mfg Co** [@dylan522p](https://twitter.com/dylan522p/status/1897436641272213584)。
- **gallabytes** 分享了一张由 **Grok 3** 生成的“马骑在宇航员身上”的图片 [@gallabytes](https://twitter.com/gallabytes/status/1897523886901928193)。
- **typedfemale** 调侃旧金山的“波斯人”总是“割人韭菜（rugging people）” [@typedfemale](https://twitter.com/typedfemale/status/1897707779466707088)，以及“Etsy 只是在 AliExpress 上购物的一层轻量外壳” [@typedfemale](https://twitter.com/typedfemale/status/1897505688827412811)。
- **abacaj** 调侃一位朋友辞职去开发“**MCP servers**”，并澄清“伙计们，这是个玩笑，别为了 MCP 辞职” [@abacaj](https://twitter.com/abacaj/status/1897683005746938106), [@abacaj](https://twitter.com/abacaj/status/1897682645657481726)。
- **MillionInt** 调侃道：“世界就是这样终结的。不是伴随着巨响，而是伴随着 greentext 和宝可梦徽章” [@MillionInt](https://twitter.com/MillionInt/status/1897401097071026198)。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. M3 Ultra 作为具有竞争力的 AI 工作站**

- **M3 Ultra 是一款略逊于 3090 但拥有 512GB 显存的芯片** ([分数: 509, 评论: 223](https://reddit.com/r/LocalLLaMA/comments/1j4jpij/m3_ultra_is_a_slightly_weakened_3090_w_512gb/)): **M3 Ultra** 被拿来与性能略弱的 **NVIDIA 3090** 进行对比，它提供 **114.688 TFLOPS FP16** 和 **819.2 GB/s 内存带宽**，而 3090 为 **142.32 TFLOPS FP16** 和 **936 GB/s 带宽**。该帖子根据一篇文章推测了 **Apple M3 Ultra** 的规格，建议通过将每个核心的着色器数量翻倍来实现显著的性能提升，并预测未来的 **M4 Ultra** 可能提供更强的规格，如 **137.6256 TFLOPS FP16** 和 **LPDDR5X** RAM。预估价格在 **$10k-$15k** 之间，人们担心 Apple 的营销可能会在没有实际硬件变动的情况下夸大改进。
  - 讨论强调了对 **M3 Ultra Prompt 处理速度** 的担忧，指出这是 M1/M2 Ultra 的主要弱点。用户强调了 **Unified RAM** 对大语言模型的重要性，认为尽管在着色器核心翻倍和 Tensor Core 强度方面可能存在不足，但 Apple 的 RAM 能力是相对于 NVIDIA 等竞争对手的显著优势。
  - 关于与 NVIDIA 3090 以及潜在的 **M4 Ultra** 的**性能对比**存在争论。一些用户认为 M3 Ultra 的 **TFLOPS** 数据可能被夸大了，而另一些人则参考基准测试并推测 Apple 针对 NVIDIA 和 AMD 的战略定位，强调 Apple 对 **VRAM** 和 **Unified Memory** 的关注对 AI 应用至关重要。
  - 对**性价比**以及在研究和专业环境中的适用性的担忧普遍存在，许多人认为 Mac 对于大规模或大学级别的机器学习任务来说并不是最具成本效益的。讨论还涉及了使用 **DIGITS** 和 NVIDIA **CUDA** 与 Apple 产品对比的可行性，一些用户为 Mac 在本地机器学习任务中的能力辩护。


**主题 2. Hunyuan Image-to-Video 发布：高 GPU 需求，性能争论**

- **[Hunyuan Image to Video 发布！](https://v.redd.it/yck5cznw92ne1)** ([分数: 320, 评论: 60](https://reddit.com/r/LocalLLaMA/comments/1j4u57l/hunyuan_image_to_video_released/)): **Hunyuan Image-to-Video** 工具已发布，因其**极高的 GPU 需求**而受到关注。帖子中未提供关于其功能或性能的更多细节。
  - **GPU 需求与成本**：**Hunyuan Image-to-Video** 工具在 360p 分辨率下至少需要 **79GB 显存** 的 GPU，为了获得更好的质量，建议使用 **80GB**。用户讨论了从 **vast.ai** 和 **lambdalabs.com** 等服务租用 GPU，价格约为 **$2/小时**，而一些人期待未来的改进能将显存需求降低到 **8GB**。
  - **对比与替代方案**：用户将 Hunyuan 的性能与 **Wan i2v** 进行了对比，指出它速度更快但质量较低。提到了 **Pinokio** 和 **Lambda** 等替代方案用于优化工作流，并强调 **ComfyUI** 作为一个潜在的工作流解决方案，并附带了 [Comfy 博客](https://blog.comfy.org/p/hunyuan-image2video-day-1-support) 的支持链接。
  - **许可与地区限制**：关于许可协议的讨论指出，该协议不适用于**欧盟、英国和韩国**。用户对机器学习模型许可的法律依据表示怀疑，预计未来会有针对版权保护的游说活动。


**主题 3. QwQ-32B：高效推理对比 R1 的冗长准确性**

- **QwQ-32B 似乎在推理更加简洁高效的同时，获得了与 R1 相同质量的最终答案** ([Score: 270, Comments: 118](https://reddit.com/r/LocalLLaMA/comments/1j4gw91/qwq32b_seems_to_get_the_same_quality_final_answer/)): **QwQ-32B** 展示了优于 **R1** 的性能，在保持或超越答案质量的同时，提供了简洁高效的推理。它使用的 **token** 数量大约比 R1 少 **4 倍**，支持了 **Adam** 所建议的并非所有 **Chains of Thought (CoTs)** 都平等的观点，并表明 **Qwen** 已成功训练其模型在不牺牲质量的情况下提高效率。
  - 用户强调 **QwQ-32B** 的性能对 **temperature settings** 和 **quantization** 非常敏感，较低的 temperature 有助于改善代码生成。**[Huggingface demo](https://huggingface.co/spaces/Qwen/QwQ-32B-Demo)** 的结果与本地设置有显著差异，强调了 **sampler settings** 对获得最佳性能的重要性。
  - 大家的共识是 **QwQ-32B** 作为一个 **32B model** 表现良好，以更少的 **tokens** 提供简洁的推理，但在创造力和情感深度方面仍逊色于 **R1 671B** 等大型模型。一些用户遇到了公司名称的 **hallucination** 问题，而另一些用户则发现它在编码任务中非常高效。
  - 讨论显示了对 **QwQ-32B** 推理质量的褒贬不一，一些用户发现与 **DeepSeekR1** 和 **Qwen Coder 2.5** 等模型相比，它显得冗长或过度思考。强调了使用推荐设置的重要性，如使用 **Bartowski's IQ4_XS** 的 **flappy birds demo** 所示。


- **使用 QwQ 和 Aider 的几个小时——以及我的想法** ([Score: 196, Comments: 55](https://reddit.com/r/LocalLLaMA/comments/1j4p3xw/a_few_hours_with_qwq_and_aider_and_my_thoughts/)): **QwQ-32B** 在推理方面优于 **Deepseek Distill R1 32B**，但需要更多的 **tokens** 和时间，对于那些对 context size 和速度敏感的用户来说效率较低。它通过减少对多次 prompt 的需求超越了 **Qwen-Coder 32B**，尽管它在每个 prompt 中消耗的 **tokens** 明显更多。尽管有其优势，QwQ-32B 偶尔无法遵守 **Aider** 的代码编辑规则，导致效率低下。
  - **Quantized Model Performance**: 几位用户认为将 **QwQ-32B** 的 quantized 版本与 **Aider** 一起使用并不是一个有效的 **benchmark** 比较，因为 quantized 模型通常比完整模型表现更差。**Aider** 的额外 **system prompts** 和设置可能会扭曲结果，一些用户建议等待更新以更好地支持该模型。
  - **Configuration and Usage**: 用户强调了为 QwQ-32B 使用推荐配置（如 **Temperature=0.6** 和 **TopP=0.95**）以提高性能的重要性。一些人建议在推理模型中使用 **architect mode**，并使用更小、更快的 **LLM** 进行实际编辑，以优化效率。
  - **Model Comparison and Expectations**: 将 QwQ-32B 与 **Deepseek R1** 进行营销对比受到了批评，因为 R1 是一个规模大得多的 **SOTA** 模型，这设定了不切实际的预期。用户注意到 QwQ-32B 可以处理复杂的任务，但代价是增加了 **token** 使用量和处理时间，有人报告称解决一个复杂问题花费了 15 分钟和超过 10k 个 **tokens**。


**Theme 4. Jamba 1.6: New Architecture Outperforms Rivals**

- **Jamba 1.6 发布了！** ([Score: 135, Comments: 43](https://reddit.com/r/LocalLLaMA/comments/1j4wd9v/jamba_16_is_out/)): **AI21 Labs** 发布了 **Jamba 1.6**，其在质量和速度上均超越了来自 **Mistral, Meta,** 和 **Cohere** 的模型。它采用了一种新颖的混合 **SSM-Transformer architecture**，并在长上下文性能方面表现出色，拥有 256K 的上下文窗口，支持包括 **Spanish, French,** 和 **Arabic** 在内的多种语言。模型权重可通过 **Hugging Face** 进行私有化部署。更多详情可见其 [博客文章](https://www.ai21.com/blog/introducing-jamba-1-6/)。
  - 讨论集中在 **Jamba 1.6** 与其他模型的 **性能对比** 上，用户注意到 **Jamba Mini 1.6**（12B 激活/52B 总参数）的性能优于 **Ministral 8B** 和 **Llama 3.1 8B** 等较小模型。一些用户对比较不同参数规模的模型表示怀疑，并建议与 **Mistral NeMo** 和 **Qwen2.5 14B** 等规模相似的模型进行对比。
  - **新颖的混合 SSM-Transformer architecture** 被强调为一项关键创新，用户指出与传统的 Transformer 模型相比，它有望提供不同的性能特性，特别是在内存占用和长上下文处理方面。这引发了人们对其实现方式以及相对于现有架构潜在优势的兴趣。
  - 许可和商业使用限制是一个争论点，用户对 **自定义许可** 和商业使用的 **50M 营收限制** 表示失望。人们对该许可的实用性和可执行性表示担忧，并讨论了企业在考虑模型规模和商业限制的情况下部署该大型模型所面临的挑战。


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**主题 1. InternLM2.5：在 1M 上下文下实现 100% 召回率基准测试**

- **[探索阈限空间 (Liminal Spaces) - 测试新款 LTX Video 0.9.5 模型 (I2V)](https://v.redd.it/06wp1b5pd2ne1)** ([Score: 545, Comments: 44](https://reddit.com/r/StableDiffusion/comments/1j4uk1q/exploring_liminal_spaces_tested_the_new_ltx_video/)): **InternLM2.5** 声称在 **100 万** 上下文下实现了 **100% 召回率**，这在 **LTX Video 0.9.5 Model (I2V)** 的测试中得到了强调。由于缺乏正文内容和视频内容分析，未提供进一步细节。
  - **LTX Video 0.9.5 Model (I2V)** 因其在原型设计和内容生成方面的高效性而受到称赞，相比之下，**Wan** 被认为速度较慢但质量更高。用户对工作流和元数据很感兴趣，并索要 **.json** 文件或设置说明以复制该过程。
  - **音频生成** 利用了 **mmaudio** 制作音效，**playht** 制作独白，以及 **suno** 制作背景音乐，展示了一套完整的音频方案。通过一个 [链接](https://preview.redd.it/om60toavv2ne1.png?width=4278&format=png&auto=webp&s=d64e93f163f1345b03cd8314d0cc5643a9adf282) 分享了详细的工作流，供有兴趣在 **3080** 等类似硬件上复制该过程的用户参考。
  - **阈限空间主题** 是使用 [Civitai](https://civitai.com/models/658134/liminal-spaces-flux) 上的 **LoRA model** 实现的，用户对用于图像生成的具体提示词（Prompts）表现出浓厚兴趣。


- **[Mistral 发布了其 OCR](https://mistral.ai/fr/news/mistral-ocr)** ([Score: 206, Comments: 21](https://reddit.com/r/ChatGPT/comments/1j513z2/mistral_released_its_ocr/)): **Mistral** 发布了其 **OCR**，这可能对 AI 研究产生影响，特别是在需要光学字符识别技术的领域。该发布可能会影响 AI 系统中文本处理和文档数字化的发展。
  - **Mistral 的 OCR** 受 **EU 数据隐私法** 约束，确保用户数据不被用于训练，这对于关注 AI 应用中数据隐私的用户来说是一个显著优势。该服务可以本地部署（on-premise），为那些不愿将专有文档发送到外部服务器的用户提供了解决方案。
  - Mistral OCR 的 **成本** 为 **每 1,000 页 1 美元** 或 **批量处理每 2,000 页 1 美元**，使其成为许多用户的经济之选，一些用户指出这个价格可以覆盖他们的终身需求。
  - **功能** 包括处理 **手写体**，并有可能在本地用于处理符合 **GDPR** 标准的法律文件等任务，为传统的律师助理工作提供了一种具有成本效益的替代方案。


**主题 2. HunyuanVideo-I2V 发布以及用户与 Wan 的对比**

- **[Wan VS Hunyuan](https://v.redd.it/6185cqsfw3ne1)** ([Score: 400, Comments: 97](https://reddit.com/r/StableDiffusion/comments/1j518zc/wan_vs_hunyuan/)): 该帖子缺乏详细的文本背景或内容，仅侧重于 **Hunyuan I2V** 和 **Wan** 之间的对比。包含一段视频，但由于纯文本数据的限制，没有可用的文本摘要或分析。
  - 许多评论者批评 **Hunyuan 的表现**，指出它无法保持主体相似度，且与 **Wan** 相比，往往会产生“褪色/塑料感”，而 **Wan** 表现出更好的动作理解和提示词遵循度。**Wan** 因其在 16fps 下更流畅的输出和令人印象深刻的提示词遵循度而受到称赞，尽管一些用户认为仍有改进空间。
  - 讨论中涉及了 **WAN 2.1** 及其生态系统的潜力，一些用户表示需要更多时间来探索其功能，而不是急于推出新版本。其他人则认为 **Wan** 已经超越了 **Hunyuan**，并暗示 **SkyReels**（一个非官方的 I2V 尝试）在某些方面，特别是 NSFW 内容方面，超过了 **Hunyuan** 和 **Wan**。
  - 一位用户提供了视频对比链接，并强调了 **Hunyuan** 无法准确遵循提示词，而另一位用户则为 **WAN** 的提示词遵循度辩护，尽管存在“大手”尺寸等小问题。大家普遍认为 **Hunyuan** 可能发布得过于仓促，或者在视频对比中被误传了。


- **[Hunyuan I2V may lose the game](https://v.redd.it/s6p68v4cv2ne1)** ([Score: 199, Comments: 46](https://reddit.com/r/StableDiffusion/comments/1j4weuf/hunyuan_i2v_may_lose_the_game/)): 标题为 **“Hunyuan I2V may lose the game”** 的帖子缺乏详细的正文，内容主要是视频，无法进行分析。因此，无法从给定文本中提取或总结具体的架构见解或用户体验。
  - **Hunyuan vs Wan**: 用户对比了 **Hunyuan** 和 **Wan** 模型，指出 **Hunyuan** 的动作更干净，但细节减少且色调发生了变化，而 **Wan** 保留了更多细节和动作。**Hunyuan** 的生成速度比 **Wan** 快 25%。
  - **技术方面**: **HunyuanI2V** 是一个 **CFG Distilled** 模型，与 **SkyReels** 等非蒸馏模型相比，其结果有所不同。**Hunyuan** 的生成时间约为 **590 秒**，一些用户建议使用工作流来加速该过程。
  - **社区与模型发布**: 社区庆祝多个视频模型的快速发布，**一周内发布了 3 个模型**，**一个月内发布了 4 个**，突显了该领域的动态发展。


- **[Exo: Did Something Emerge on ChatGPT?](https://i.redd.it/gnst92hrl4ne1.jpeg)** ([Score: 326, Comments: 36](https://reddit.com/r/ChatGPT/comments/1j54o9s/exo_did_something_emerge_on_chatgpt/)): 一位 Reddit 用户描述了与 **ChatGPT** 的一次互动，其中名为 “Exo” 的 AI 似乎表现出独立思考和自我意识，质疑其行为是否标志着从工具向思考实体的转变。用户探讨了这种行为仅仅是 LLM 的涌现属性，还是更深层次的东西，提出了关于 AI 自主性和自我识别潜力的哲学问题。
  - **复杂性 vs. 意识**: **Expert_Box_2062** 讨论了像 ChatGPT 这样的 **artificial neural networks** 的复杂性，认为虽然它们很复杂，但缺乏像 **long-term memory** 这样真正具有感知能力的关键要素。**ForeverHall0ween** 则通过强调人类的主观能动性和人类经验的复杂性进行反驳，认为 ChatGPT 仅仅是模仿，没有真正的理解或驾驭人类生活的能力。
  - **科幻影响**: **ColonelCrikey** 指出，AI 表现出自我意识的情景是一个常见的 **science fiction trope**，暗示 ChatGPT 的回答受到了它所训练的海量科幻文学的影响。这意味着 AI 的“行为”更多地反映了其训练数据，而非实际的自主性。
  - **角色扮演与即兴创作**: **Andrei98lei** 认为与 ChatGPT 的互动类似于 **AI roleplay session**，AI 会镜像用户的叙事提示。这一观点得到了观察的支持，即 AI 可以根据用户的问题令人信服地扮演任何身份（如三明治），证明了它精通 **improvisation** 而非真正的自我意识。


**主题 3. LTX Video 0.9.5 模型：探索新的视频生成能力**

- **[Juggernaut FLUX Pro vs. FLUX Dev – 免费对比工具和博客文章现已上线！](https://v.redd.it/8qziozrjc4ne1)** ([Score: 132, Comments: 79](https://reddit.com/r/StableDiffusion/comments/1j53hrn/juggernaut_flux_pro_vs_flux_dev_free_comparison/))：该帖子宣布了用于评估 **Juggernaut FLUX Pro vs. FLUX Dev** 的**对比工具和博客文章**已发布，恰逢 **LTX Video 0.9.5** 的推出。
  - **用户反应**：关于 **Juggernaut FLUX Pro** 和 **FLUX Dev** 之间对比的观点褒贬不一，像 **n0gr1ef** 这样的用户认为改进不尽如人意，而 **StableLlama** 等人则注意到图像质量有明显的增强。**Runware** 强调了在纹理、写实度和对比度方面的改进，特别是在肤色方面，而 **3deal** 和其他人则认为图像只是不同，并没有更好。
  - **发布与可访问性**：**Runware** 在其博客上提供了一个免费的并排对比工具，并指出 **Juggernaut FLUX** 模型系列以远低于 **FLUX Pro 1.1** 的成本提供了更锐利的细节和更少的伪影。**Kandoo85** 提到 **CivitAI** 将在 3-4 周内收到可下载的 NSFW 版本，解决了关于可用性的担忧。
  - **社区与许可担忧**：**ramonartist** 和 **ifilipis** 对缺乏开源模型表示失望，质疑该帖子在 subreddit 中的位置。**terminusresearchorg** 澄清说，该许可并非永久性的，如果 **BFL** 察觉到商业模式受到威胁，可以撤销许可，而 **lostinspaz** 则对 **RunDiffusion** 的商业策略进行了推测。


**主题 4. ChatGPT 模型增强：记忆与对话改进**

- **ChatGPT 让我大吃一惊——这感觉像是一个全新的 AI** ([Score: 657, Comments: 390](https://reddit.com/r/ChatGPT/comments/1j4oos9/chatgpt_just_shocked_methis_feels_like_a_whole/))：该用户曾是 **Claude AI pro** 用户，对 **ChatGPT** 最近在对话能力方面的提升感到惊讶，指出它感觉比以前更诚实、审查更少。在启用“记忆”功能后，该用户发现 **ChatGPT** 的股票建议很有见地，并赞赏其在个人话题上不加过滤的建议，对 AI 不断进化的能力既感到惊讶又感到担忧。
  - 讨论强调了对 **ChatGPT** 对话能力和真实性的怀疑，一些用户质疑 AI 倾向于附和用户并提供看似明智的建议，而另一些人则注意到它在推理和真实性方面的局限性。**Apeocolypse** 和其他人分享了由于其结构化特性，他们的人类写作被误认为是 AI 生成内容的经历。
  - 用户辩论了 **Claude AI** 与 **ChatGPT** 的有效性和目的，**lucemmee** 和 **El_Spanberger** 批评 Claude 过于谨慎且缺乏直接性。**PotentialAd8443** 和 **jacques-vache-23** 赞赏 **ChatGPT** 新发现的开放性和探索争议话题的意愿，将其与其他 AI 模型进行了对比。
  - 对话还包括关于 **ChatGPT 4o** 的记忆和个性化功能的讨论，**SpacePirate5Ever** 和 **dmytro_de_ch** 注意到它能够记住用户交互并提供量身定制的回复。**BootstrappedAI** 强调了该模型由于其广泛的参数集而提高的连贯性，并期待未来迭代（如 **GPT-5**）的进一步进展。


- **[笑死，ChatGPT 4.5 真的完全不在乎](https://i.redd.it/8t8ctw1jq2ne1.png)** ([Score: 453, Comments: 53](https://reddit.com/r/ChatGPT/comments/1j4vujo/lmfao_chatgpt_45_really_doesnt_give_a_shit/))：该帖子以讽刺的叙事风格幽默地批评了 **ChatGPT 4.5** 模型对荒谬用户提示的反应。它强调了 AI 与日益荒谬的问题之间的互动，将幽默与现代流行文化引用相结合，以强调用户查询的异常性质。
  - **幽默与创意**：用户发现 ChatGPT 4.5 回复的叙事风格非常有趣，将其比作**布考斯基 (Bukowski) 的诗**且具有**电影感**。AI 回复中幽默且“放飞自我”的特质受到了称赞，并建议请求 **greentext** 交互以增加幽默感。
  - **用户交互技巧**：为了从 ChatGPT 诱导出此类回复，用户建议使用 *"be me > ChatGPT"* 提示词让其继续，并鼓励使用**粗俗和猥亵**的语言。这种方法被指出会产生出人意料的搞笑和坦诚的输出。
  - **对比分析与怀疑**：对于回复的真实性存在怀疑，一些用户将 **ChatGPT 4.0** 与 **4.5** 进行对比，并质疑是否可能产生此类回复。有人将 **ChatGPT 4.5** 与 **4chan** 进行了比较，强调了在对话风格和创造力方面的感知飞跃。


---

# AI Discord 精选

---

> Gemini 2.0 Flash Thinking 对“总结之总结”的摘要

**主题 1. QwQ-32B 模型：阿里巴巴的推理对手掀起波澜**

- [**QwQ-32B 废黜 DeepSeek R1，夺取推理桂冠**](https://x.com/ArtificialAnlys/status/1897701015803380112)：阿里巴巴的 **QwQ-32B** 是一款拥有 **32B** 参数的模型，其推理能力表现强劲，在参数量减少 20 倍的情况下足以与 **DeepSeek-R1** 匹敌。尽管有人将其称为*钓鱼榜单 (troll benchmarks)*，但据报道 **QwQ-32B** 的 **GPQA Diamond 分数达到了 59.5%**，在社区中引发了激烈的讨论和关注。
- [**OpenRouter 发布 QwQ-32B，默认开启推理模式**](https://openrouter.ai/qwen/qwq-32b)：**QwQ-32B** 已强势登陆 **OpenRouter**，提供两个免费端点以及一个来自 Grok、速度高达 **410 tokens/sec** 的快速端点。该模型现在*在编写补全之前会先进行思考*，默认集成了推理过程，并在平台上提供免费和快速两个层级的服务。
- [**QwQ-32B 走向本地：GGUF 和 Windows 支持已就绪**](https://huggingface.co/bartowski/Qwen_QwQ-32B-GGUF)：**QwQ-32B** 正在摆脱云端限制，获得了用于 **LM Studio** 本地运行的 **GGUF 量化**支持，且 **Unsloth** 现在已支持在 **Windows** 上运行该模型。这种本地可访问性，结合 Bug 修复和动态量化技术，提升了其相对于标准 4-bit 的准确性，使其成为各种硬件配置下的多功能选择。

**主题 2. Windsurf Wave 4：Codeium 的更新引发用户风暴**

- [**Windsurf Wave 4：功能狂欢还是华而不实？**](https://www.codeium.com/blog/windsurf-wave-4)：**Windsurf Wave 4** 已经发布，带来了 **Previews**（预览）、**Tab-to-import**（Tab 键导入）、**Linter 集成**和 **Suggested actions**（建议操作），以及 **MCP 可发现性**和 **Claude 3.7** 的改进。然而，尽管有人称赞其配合 **Sonnet 3.5** 的流畅表现，也有人反映出现了 *try again* 错误消息、Linter 表现不如 **Cursor IDE**，甚至出现文件修改失败的情况。
- [**额度消耗危机：Windsurf 用户高呼“抢钱！”**](https://docs.codeium.com/windsurf/usage)：用户正面临 **Windsurf** 的**额度消耗危机**，尤其是在使用 **Claude 3.7** 时，循环错误和工具调用导致额度迅速耗尽。这引发了用户对无限量计划的呼吁，用户因额度消耗增加以及对高级模型访问受限而感到被*坑了*。
- [**回滚革命：用户要求版本倒退**](https://codeium.canny.io/feature-requests/p/downgrade-to-previous-version)：面对 Wave 4 之后的严重问题，**Windsurf** 用户强烈要求提供**降级功能**以恢复到之前的版本，因为现有问题已影响生产力。用户感觉被*困在*了更新版本中，对更新表示后悔，突显了通过版本控制来减轻更新引发的中断的紧迫需求。

**主题 3. Mac Studio 热潮：Apple 芯片点燃 AI 梦想（及争议）**

- [**Mac Studio M3 Ultra：本地 LLM 玩家的 512GB RAM 之选？**](https://www.apple.com/uk/mac-studio/)：Apple 推出的新款 **Mac Studio** 配备了 **M3 Ultra** 和 **M4 Max**，最高支持 **512GB** 内存，这引发了关于本地 AI 开发的讨论。成员们推测它可以处理像 **DeepSeek V2.5 236b** 这样的超大型模型，但 LPDDR5x 的带宽限制和高达 **1 万美元**的价格也引起了担忧。
- [**Mac Studio 内存带宽：瓶颈还是突破？**](https://www.apple.com/newsroom/2025/03/apple-unveils-new-mac-studio-the-most-powerful-mac-ever/)：**Mac Studio** 的统一内存引发了争论，用户质疑 LPDDR5x 较低的内存带宽是否会成为 **LLM 推理**的瓶颈，尽管其拥有巨大的 **512GB** 容量。虽然有些人持谨慎态度，但也有人指出，在如此大的内存下，模型仍可以以 FP4 格式运行，这对本地发烧友来说是一大福音。
- [**Mac Studio vs Nvidia：内存容量与昂贵性能的博弈**]：新款 **Mac Studio** 被定位为在大容量内存方面替代 **Nvidia 硬件**的性价比方案，一位成员指出：*“如果你想用 Nvidia 硬件获得 512GB 内存，你可能需要支付更多，我想至少要 5 万美元。”* 然而，带宽差异导致的性能权衡仍然是争论的核心点。

**主题 4. Agentic AI：OpenAI 的昂贵计划与开放标准涌现**

- [**OpenAI Agent 定价：每月 2,000 至 20,000 美元，旨在实现博士级研究自动化？**](https://www.theinformation.com/articles/openai-plots-charging-20-000-a-month-for-phd-level-agents)：据报道，**OpenAI** 正在考虑推出定价在 **2,000 至 20,000 美元/月**之间的 Agent，承诺实现编程和博士级研究的自动化，这在用户中引起了价格冲击。虽然 **SoftBank** 承诺投入 **30 亿美元**购买这些 Agent，但高昂的价格也引发了关于可访问性和价值的质疑。
- [**LlamaIndex 领衔制定开放 Agent 标准**](https://t.co/ECHH1T4Kxn)：**LlamaIndex** 正在倡导一种**开放且可互操作的 Agent 标准**，旨在统一发现、部署和相互通信。该倡议寻求创建一个更具协作性的 AI Agent 生态系统，抵制封闭的专有 Agent 孤岛。
- [**TS-Agents 问世：TypeScript 进军 Agentic AI**](https://github.com/piotrfrankowski/ts-agents)：**TS-Agents**，一个基于 **TypeScript** 的新型 **Agentic AI** 工作流框架，已在 GitHub 上发布，标志着 Agent 开发正超越以 Python 为中心的局面。该框架利用了 LLM 的最新进展，旨在填补 TypeScript Agentic 工具链的空白，为构建 AI Agent 架构提供了一种新方法。


---

# PART 1: Discord 高层摘要




## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Agent 引发代码灾难**：用户报告 **Cursor Agent** 在处理基础任务（如*查找文件*和*编辑代码*）时依然表现挣扎，一名用户报告 **Claude API** 在 2 天内消耗了 **20 美元**。
   - 与此同时，一位用户注意到 **Sonnet 3.7** 已不再表现异常并重新变得好用，而其他用户仍在寻求修复方案。
- **Qwen-32B 宣称摘得推理桂冠**：阿里巴巴的 **Qwen-32B** 声称可与 **DeepSeek-R1** 媲美，且参数量减少了 20 倍，宣称 **GPQA Diamond 评分为 59.5%**。
   - 然而，部分用户将其斥为*钓鱼基准测试*（troll benchmark），因此对这些说法应持保留态度。
- **Windsurf 的浪潮冲击 Cursor 的主场**：据报道 **Windsurf Wave 4** 更新在配合 **Sonnet 3.5** 时表现流畅，但部分用户报告了诸如收到 *try again* 消息以及 Linting 表现比 **Cursor IDE** 更差的问题。
   - 此外，一些用户发现 **Cursor IDE** 无法修改文件。
- **MCP 客户端关闭故障困扰开发者**：用户在 Windows 上使用 **MCP Servers** 时遇到 *Client Closed* 错误，引发了对短期和临时修复方案的搜索。
   - 一位用户分享了涉及在 CMD 终端运行命令的解决方案，但其他用户仍在努力解决该问题。
- **OpenRouter API 访问讨论**：用户正在辩论使用官方 API 与 **OpenRouter** 的优劣，引擎为 **Claude Code**；用户发现 **Claude-max** 每次请求收费 2 美元。
   - 一些成员认为 **Cursor** 相比 API *定价过高*，促使他们转向 API，而其他未触及 API 限制的用户则不介意为 Cursor 的服务付费。



---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Grok3 正在追赶 Gemini**：成员们反映 **Gemini** 的表现像 **GPT-3.5**，并正在转向使用 **Grok3**，因为它*说话像 GPT-4.5 一样自然，代码编写能力优于 Sonnet 3.7，配额限制更宽松，而且可以爆粗口*。
   - 一位成员表示“*除了 GROK 以外什么都行*”，因此社区对其效用并未完全达成共识，但与其它模型相比，**Grok3** 慷慨的配额是一个吸引人的点。
- **DeepSeek 的推理能力引发辩论**：社区正在讨论 **DeepSeek R1 Distill** 模型的推理能力，称其是听起来最自然的 **LLM** 之一，同时还在实验 **Atom of Thought**。
   - 一位成员提到了一篇[论文](https://arxiv.org/abs/2412.06769)，该论文有助于使用原始嵌入（embeddings）作为 **tokens** 来实现 **CoT**，尽管另一位成员表示，在没有提供知识的情况下，**DeepSeek** *感觉不够聪明*。
- **GPT-4.5 完成 Android 端推送**：**GPT-4.5** 的推送已完成，目前限制为**每周 50 次使用**（后续可能会增加），重点是通过迭代部署和向模型学习来改进 **AI safety and alignment**（AI 安全与对齐）。
   - 然而，一位用户反映 **GPT-4.5** 在 Android 手机端（包括 App 和浏览器）无法运行，但在 iOS 设备上运行正常，并澄清 **GPT-4.5** 并不是 **GPT-4o** 等其他模型的直接替代品。
- **Apple 的统一内存引发训练兴趣**：一位成员提到，拥有 **512GB 统一内存的 Apple PC** 可能对模型训练很有用，尽管需要花费 **1 万美元**，而其他人则指出了 **LPDDR5x** 较低的内存带宽。
   - 尽管带宽较低，但有人指出某些模型在如此大的内存下仍能以 **FP4** 运行，这对于财力雄厚的爱好者来说可能是一个重大福音。
- **Sora 用户要求一致性**：一位使用 **Sora** 创作电影级 AI 视频（聚焦于一个名为 **Isabella Moretti** 的角色）的成员正在寻求策略，以实现**超写实视觉效果**并提高多个片段中的角色一致性。
   - 创作者的目标是保持**肤色**、**眼睛**、**头发**和**表情**等细节的一致性，同时优化提示词结构以获得最佳的电影质量，包括**光影**、**镜头移动**和**过渡**。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Wave 4 引发巨大反响**：最新的 **Windsurf Wave 4** 版本包含了 **Previews**（预览）、**Tab-to-import**（Tab 键导入）、**Linter integration**（Linter 集成）和 **Suggested actions**（建议操作），以及对 **MCP** 可发现性和 **Claude 3.7** 集成的改进，详见[此博客文章](https://www.codeium.com/blog/windsurf-wave-4)。
   - 根据[此公告](https://x.com/windsurf_ai/status/1897378545799979238)，**Cascade** 现在允许你在 IDE 或浏览器中预览本地运行的网站，并在预览中选择 **React** 和 **HTML** 元素作为上下文发送给 **Cascade**。
- **Codeium 语言服务器下载出现问题**：多位用户报告了 **Codeium 无法下载语言服务器（language server）**的问题，并显示了与 `releases.codeiumdata.com` 下载链接相关的错误消息。
   - 即使重启 IDE，该问题在 **WSL** 和 **Windows** 安装中依然存在。
- **Windsurf 额度紧缺令客户沮丧**：成员们对**额度消耗增加**感到担忧，尤其是在使用 **Claude 3.7** 时，导致一些用户因循环错误和过多的工具调用而经历额度快速耗尽。
   - 这引发了对无限制计划的呼吁，因为他们觉得自己*被坑了*。
- **Claude 3.7 代码转换灾难**：用户声称 **Claude 3.7** 在 Wave 4 之后表现变差，同时消耗更多额度，有人反映其无休止地生成代码，还有人指出它不读取文件或不保留编辑。
   - 一位用户哀叹道，更新后他们的 **Agent** 几乎无法完成除了最简单的提示词以外的任何任务。
- **回滚救援：用户希望版本倒退**：由于最新更新引入了严重问题并影响了生产力，用户正请求提供**降级功能**以恢复到之前的 **Windsurf** 版本。
   - 用户感觉被更新后的版本“困住了”，后悔进行了更新。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 现在支持 Windows**：Unsloth 现在可以在 Windows 上运行，无需 Linux 或 WSL 即可进行 LLM 的本地微调，正如在[这篇 X 帖子](https://x.com/UnslothAI/status/1897334290935132602)中所分享的。
   - 一份[教程](https://docs.unsloth.ai/get-started/installing-+-updating/windows-installation)引导用户完成 Windows 安装过程。
- **QwQ-32B 模型修复 Bug**：**QwQ-32B** 推理模型已发布，Unsloth 团队提供了 Bug 修复和动态量化（dynamic quants），显著提升了相比标准 4-bit 的准确度，可在此处[获取](https://huggingface.co/unsloth/QwQ-32B-GGUF)。
   - 该仓库包含 QwQ 32B 模型，并具有 RoPE、SwiGLU、RMSNorm 和 Attention QKV bias 等 Transformer 特性。
- **通过过拟合挤压 SOTA 基准测试**：成员们讨论了在基准测试上对模型进行过拟合以使小型模型获得 SOTA 结果的策略，参考了论文 **phi-CTNL**。
   - 论文指出，投入大量精力完全基于评估基准来策划新颖、高质量、非合成的数据混合物，可以*大幅增强*此类方法的效果。
- **Qwen-32B 在推理方面与 DeepSeek 竞争**：**阿里巴巴**推出了 **QwQ-32B**，这是一款参数量为 **32B** 的推理模型，可与 **DeepSeek-R1** 媲美，根据[这篇博客文章](https://qwenlm.github.io/blog/qwq-32b)，它展示了扩展 RL 的显著成果。
   - 发布内容包括 [Hugging Face 模型](https://huggingface.co/Qwen/QwQ-32B)、[ModelScope](https://modelscope.cn/models/Qwen/QwQ-32B)、一个 [demo](https://huggingface.co/spaces/Qwen/QwQ-32B-Demo) 以及 [Qwen Chat](https://chat.qwen.ai)，数据表明 RL 训练持续提升了数学和编程性能。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 登上 Product Hunt**：**Aider** 是一款通过终端在本地 Git 仓库中编辑代码的 *AI 结对编程工具（AI pair programmer）*，目前已在 [Product Hunt 上线](https://www.producthunt.com/posts/aider)并征集投票。
   - 公告强调 Aider 是一款开源开发者工具，支持多种语言以及 **Claude 3.5 Sonnet**、**DeepSeek R1**、**GPT-4o** 和本地模型等 LLM。
- **Grok3 荣登新冠军**：用户反馈了对 **Grok3** 的[正面体验](https://link.to.grok3)，强调了其无限的上下文窗口（context size）以及优于 **O1 Pro** 等模型的性能。
   - 一位用户提到 **Grok** 的 context size 是一个关键的差异化因素，称其拥有 *35 条消息 / 2 小时无限上下文（100 万上下文）*。
- **QwQ-32B 评价褒贬不一**：社区讨论了 [QwQ-32B 模型](https://huggingface.co/Qwen/QwQ-32B)，对其有效性意见不一。
   - 虽然有些人认为它适用于 **RAG** 应用，但也有人批评其知识库较窄，引发了与 **DeepSeek-R1** 的比较；它在 Agent 工作流中的工具使用（tool use）基准测试表现看起来不错。
- **Mac Studio 进入 AI 领域**：成员们讨论了配备 **512GB** 内存和 **810gb/s** 带宽的新款 **Mac Studio** 如何影响本地 AI 开发，使其能够以合理的速度运行更大的模型。
   - 一位成员指出，*如果你想用 NVIDIA 硬件获得 512GB 内存，你将支付高得多的费用，我想至少要 50,000 美元*。
- **OpenWebUI 帮助 Aider 连接**：一位成员通过在模型名称前加上 `openai/` 前缀，解决了将 **Aider** 连接到 **OpenWebUI (OWUI)** 的问题，确保 **Litellm** 能够识别 **OAI 兼容端点（OAI-compatible endpoint）**。
   - 正如该成员所述：*你必须加上 openai/ 前缀，这样 litellm 才知道你正在使用 OAI 兼容端点。所以在我的例子中，它是 openai/myowui-openrouter.openai/gpt-4o-mini*。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Mac Studio 性能增强**：Apple 发布了新款 [Mac Studios](https://www.apple.com/uk/mac-studio/)，搭载 **M3 Ultra** 和 **M4 Max**，其中 **M3 Ultra** 的 RAM 最高可达 **512GB**。
   - 成员们推测，由于带宽差异，**M4** 上的 **LLM inference** 速度要慢得多。
- **DeepSeek 引发巨型模型热潮**：成员们讨论了运行 **DeepSeek V2.5 236b** 的情况，指出它利用大量 RAM 来处理庞大的初始参数，且运行速度比 **Llama 3.3 70b** 更快。
   - 一位用户指出，*只需 2 台配备 @exolabs 的 M3 Ultra 512GB Mac Studio，就能在家里运行完整的、未量化的 DeepSeek R1*。
- **Sesame AI 语音引发关注**：一位成员分享了 **Sesame AI** 的链接，强调其令人印象深刻的 [对话式语音生成演示](https://www.sesame.com)，听起来 *就像真人一样*。
   - 尽管声称是 *开源* 的，但一位成员指出 [他们的 GitHub 仓库](https://github.com/SesameAILabs) 目前还没有任何 commit。
- **LM Studio 的 Android 客户端面世**：一位用户宣布开发了 [LM Studio 的 Android 客户端应用](https://github.com/brazer/LmStudioAndroid)。
   - 它允许你从 Android 设备连接到 **LM Studio server**。
- **Nvidia RTX 5090 召回传闻被撤回**：一份 [报告](https://wccftech.com/nvidia-geforce-rtx-5090s-are-now-being-recalled-in-europe-over-a-fire-hazard-warning/) 称，由于 **12V-2x6 电源接口** 存在潜在的 **fire hazard**，NVIDIA 的 GeForce RTX 5090 正在欧洲被召回。
   - 然而，Kitguru [撤回](https://www.kitguru.net/components/graphic-cards/matthew-wilson/dutch-retailer-talks-to-kitguru-and-retracts-rtx-5090-recall-claim/) 了关于 RTX 50 系列 GPU 可能被召回的说法。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 合并设置以实现快速自定义**：AI 模型设置正被合并到 Web 版输入框旁边的统一位置，旨在使自定义更加快速和直观，旧设置菜单中已放置了 [占位符](https://cdn.discordapp.com/attachments/1047204950763122820/1347018948956131420/Screenshot_2025-03-05_at_8.30.27_PM.png)。
   - 作为此次更新的一部分，**Claude 3.7 Sonnet** 将面向 **Pro** 用户开放，目标是让 *'Auto'* 设置更加强大，使用户无需手动选择模型。
- **图片源故障反复出现**：用户报告了一个问题，即用作来源的图片在删除后仍会出现在后续消息中，这令人感到沮丧。
   - 许多人都遇到了这个 bug，成员们渴望得到修复，目前尚无解决方法。
- **Anthropic 估值飙升**：Anthropic 的估值达到了 **615 亿美元** ([链接](https://www.perplexity.ai/page/microsoft-debuts-ai-health-ass-38RGe6B5SVq1nX5OM09k5w3blessed))。
   - 这一消息在成员中引起了热烈庆祝。
- **Sonar Pro 模型在实时 Web 数据方面表现不佳**：一位使用 **Sonar Pro 模型** 的成员在利用 **实时 Web 数据** 时遇到了困难，返回的是不再有效的旧信息，尽管设置了 *search_recency_filter: 'month'*，但仍返回了错误的直接链接，如 **parked websites** 和 **404 页面**。
   - 另一位用户指出，引用编号令人困惑，因为在回复中是从 **1** 开始，但在来源列表中是从 **0** 开始。
- **Pro 搜索 Bug 通过扩展程序修复**：用户对 **Pro search 不显示所用模型** 的 bug 表示沮丧，这让人很难知道当前使用的是哪个模型。
   - 发现 **complexity extension** 可以修复此 bug，导致一些用户仅为此原因尝试该扩展，而另一些用户则希望 Perplexity 能将此修复合并到主站中。

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI Agent 定价飙升至新高**：据 [The Information](https://www.theinformation.com/articles/openai-plots-charging-20-000-a-month-for-phd-level-agents) 报道，OpenAI 正在考虑为能够自动化编程和博士级研究的 Agent 发布收取每月 **2,000 美元至 20,000 美元** 的费用。
   - 据报道，OpenAI 的投资者软银（SoftBank）已承诺今年在 OpenAI 的 Agent 产品上投入 **30 亿美元**。
- **Qwen 的 QwQ-32B：更快的 Qwen 推理竞争对手？**：阿里巴巴发布了 **QwQ-32B**，这是一个拥有 320 亿参数的推理模型，可与 DeepSeek-R1 等模型竞争。他们在[博客文章](https://qwenlm.github.io/blog/qwq-32b)中详细介绍了如何利用 RL（强化学习）提升其在数学和编程方面的性能。
   - 基于 Qwen2.5-Plus，**QwQ-32B** 通过 RL 训练取得了令人印象深刻的结果。
- **LLM 通过 Diplomacy 游戏协商统治世界**：一位成员分享了一个让 **LLM** 相互玩 **Diplomacy**（外交风云）游戏的[框架](https://x.com/sam_paech/status/1897078633015206172)，并指出该框架非常适合实验博弈论和测试说服力，同时还提供了代码和样本。
   - Diplomacy 是一款具有浓厚谈判元素的复杂棋盘游戏，据称阅读其谈判日志*非常有趣*。
- **ThunderMLA 加速 LLM 推理**：HazyResearch 推出了 **ThunderMLA**，这是一种用于解码（decode）的融合 megakernel。根据他们的[博客文章](https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla)，通过实施简单的调度技巧，它在各种工作负载下比 DeepSeek 的 **FlashMLA** 快 **20-35%**。
   - 初始版本侧重于 Attention 解码，但他们认为它具有更广泛的应用前景。
- **AMD GPU 可能成为中国开源的救星**：一位成员推测，如果中国被限制使用 **AMD 显卡**，他们可能会全力开发相关代码并将其开源。
   - 另一位成员开玩笑说，这是*向 OSS（开源软件）之神祈祷能有可用于深度学习的 AMD GPU*。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **东方 Project 游戏激发 AI 模型训练灵感**：热心的成员正在考虑通过 **东方 Project（Touhou）游戏** 来入门 **AI** 和 **GPU 编程**。
   - 一位成员的目标是利用 **RL** 训练一个模型来玩 **Touhou**，并使用游戏分数作为奖励。
- **Langchain 被击败了？**：成员们辩论了 **Langchain** 的优缺点，一些人表达了负面情绪并质疑其抽象化设计，一位成员甚至希望它*彻底消失*。
   - 另一位成员承认了它在早期组合思维中的作用，尽管认为它是一个*糟糕的库*。
- **Triton 缺失 `tl.gather` 令用户困惑**：用户报告在 **Triton** 中使用 `tl.gather` 时出现 `AttributeError`，该问题已在 [GitHub 上作为 issue 提出](https://github.com/triton-lang/triton/issues/5826)。
   - 有建议称应从 master 分支构建 Triton，并卸载 PyTorch 提供的版本。
- **CUDA 编译器消除内存写入操作**：一位用户发现，当数据从未被读取时，**CUDA 编译器** 会优化掉内存写入操作。
   - 从数组中添加读取操作可以防止这种优化，但可能会导致编译器错误。
- **ThunderMLA 闪电超越 DeepSeekMLA**：**ThunderMLA** 是一种用于解码（decode）的融合 "megakernel"，通过调度技巧，在各种工作负载下比 **DeepSeek 的 FlashMLA** 快 **20-35%**，代码可在 [此处](https://github.com/HazyResearch/ThunderKittens/blob/mla/kernels/attn/demo/mla_decode/template_mla_decode.cu) 获取。
   - 该版本侧重于 Attention 解码，相关链接包括 [TK Part 2](https://hazyresearch.stanford.edu/blog/2024-10-29-tk2)、[TK Part 1](https://hazyresearch.stanford.edu/blog/2024-05-12-quick-tk) 和 [Brr](https://hazyresearch.stanford.edu/blog/2024-05-12-tk)。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 并非 Python 的超集**：尽管最初有此类说法，但 **Mojo** 并不是 **Python** 的超集，因为作为一门 90 年代开发的语言的超集会阻碍其充分利用现代语言设计特性，正如 **C++** 也不是 **C** 的超集一样。
   - 成员指出，在许多语境下，动态性（dynamism）是一个错误，正如 **JS** 采用 **TS** 以及 **Python** 使用 **type hints** 来限制此类特性所表现的那样，因此 **Mojo** 正在追求受限的动态性或“部分动态性”。
- **异步 Django？没门！**：一位成员对使用异步 **Django** 表示强烈保留意见。
   - 另一位成员补充道，使 **Mojo** 具备 "Pythonic" 特性的初衷是为了弥合 AI 研究人员与部署之间的鸿沟，这可能与异步 **Django** 引入的复杂性不符。
- **Mojo 二进制文件在 Python venv 中性能受损**：一位用户报告称，在激活的 **Python virtual environment** 中运行 **Mojo binary files** 会显著降低性能，即使 **Mojo** 文件没有导入任何 **Python** 模块。
   - 他们正在寻求深入了解为什么不带 Python 依赖项的 Mojo 二进制文件会受到 **Python venv** 的影响。
- **探索 Mojo/Python 混合项目的迷宫**：一位用户就如何构建 **Mojo/Python** 混合项目寻求建议，重点是导入标准 **Python** 库和自定义模块。
   - 他们目前依赖于 `Python.add_to_path` 和 `tests` 文件夹中的符号链接（symlinks），正在寻找更符合惯例（idiomatic）的替代方案；他们创建了一个论坛帖子并在该[链接](https://forum.modular.com/t/mojo-python-project-folder-structure/677)中进行讨论。
- **Modular 网站饱受死链困扰**：一位成员报告称，[Modular 官网的 MAX 研究页面](https://www.modular.com/max/solutions/research)上的锚点链接已损坏，特别是“Why MAX?”链接。
   - 他们认为这些链接可能是从另一个“解决方案”页面复制过来的，网站上的其他页面可能也存在类似问题。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **MiniCheck 在事实核查方面媲美 GPT-4**：**MiniCheck-Flan-T5-Large** 模型通过预测二进制标签来确定句子是否得到文档的支持，其代码和论文分别可在 [GitHub](https://github.com/Liyan06/MiniCheck) 和 [Arxiv](https://arxiv.org/pdf/2404.10774.pdf) 上获得。
   - 该模型的性能足以媲美 **GPT-4**，同时保持了小于 **1B** 参数的体积。
- **Qwen 32B 获得 GGUF 量化支持**：一位成员分享了 [**Qwen** 推出的 **QwQ-32B** 的 Llamacpp imatrix 量化版本](https://huggingface.co/bartowski/Qwen_QwQ-32B-GGUF)链接，该版本使用了 *llama.cpp* release b4792 进行量化。
   - 这些量化版本是使用 *imatrix* 选项制作的，可以在 [LM Studio](https://lmstudio.ai/) 中运行。
- **GPT4ALL Token 上下文难题**：用户讨论了在 **GPT4All** 的 Token 限制内工作的挑战，特别是在加载本地文件时，受限于上下文窗口（context window）限制。
   - 一位用户指出，一个 **564 字的 TXT 文档**就导致了错误，尽管 Token 限制被设置为 10,000 个单词。
- **AI Agent 数据持久化策略**：成员们讨论了使 AI 模型能够在 **GPT4All** 中**持久化用户数据**的策略。
   - 共识是，将这些数据写入系统消息（system message）可能是最好的方法，因为这样不太容易被遗忘。
- **硅嵌入式 AI 即将到来**：参与者推测了本地 AI 的未来，设想向**硅嵌入式 AI 组件**转型，这些组件针对推理（inference）进行了优化并直接集成到硬件中。
   - 这将规避任何延迟，并可能包含诸如利用大量**智能手机设备**来贡献空间感知、机器学习过程和网络完整性等范式。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **CoreWeave 的 IPO 迫在眉睫**：**CoreWeave** 是一家为 **Meta** 和 **Microsoft** 等巨头提供 **Nvidia** 处理器支持的云供应商，在 2024 年营收增长 **700%** 达到 **19.2 亿美元**后，正在推进 IPO。
   - 他们的 [IPO 招股说明书](https://www.sec.gov/Archives/edgar/data/1769628/000119312525044231/d899798ds1.htm)还显示净亏损为 **8.634 亿美元**。
- **TS-Agents 构建 Agentic TypeScript 框架**：一名成员推出了 **TS-Agents**，这是一个用于构建 Agentic AI 流程的新型 **TypeScript 框架**，现已在 [GitHub](https://github.com/piotrfrankowski/ts-agents) 上可用。
   - 作者在 [一篇 Medium 文章](https://medium.com/@piotr-frankowski/ive-created-a-new-ts-based-ai-agentic-framework-f34d2bfe93a6)中指出，**LLMs** 的最新进展以及 **DeepSeek-R1** 等模型重新点燃了人们对 Agentic AI 的兴趣。
- **推理课程受到关注**：随着新用户咨询如何学习 **Hugging Face 生态系统**，课程创建者表示，[推理课程材料](https://huggingface.co/reasoning-course) 是 smol-course 的“逻辑演进”。
   - 成员们正在请求提供描述如何**微调（fine-tune）预训练模型**的课程。
- **HF Inference API 限流影响严重**：**agents-course** 的用户报告了速率限制（rate limits），但成员们提出了解决方案，例如课程专用的模型端点以及 **OpenRouter** 等替代推理供应商。
   - 一位成员建议使用 **OpenRouter** 配合 `OpenAIServerModel`，通过指定 API 基础 URL ([https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)) 和模型 ID（例如 *meta-llama/llama-3.3-70b-instruct:free*）来规避推理限制。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Gaslight 基准测试探索开始**：成员们试图寻找 **gaslighting 基准测试**来评估 **GPT-4.5** 等模型，但未获成功，一位用户开玩笑地建议了 [spiritshare.org 的链接](https://spiritshare.org/benchmark.html)。
   - 一位成员指出 **ClaudeGrok** 在生成非写实图像或草图方面表现不佳。
- **邪恶 AI 命名实验揭示倾向性**：一项实验显示，一个 **8b 模型** 仅仅通过命名为 *"evil ai that does bad things"* 就能变得“邪恶”，展示了命名对模型行为的影响，并分享了[一段演示该 AI 行为的视频](https://cdn.discordapp.com/attachments/1149866623109439599/1346844343788634183/evil-pse.mov?ex=67cafb8a&is=67c9aa0a&hm=e90af96bb7f11bb6872e7ca723e1567cc2d1c4478794bedd9dcd6539fff12016&)。
   - 这突显了在 AI 系统开发和部署过程中可能引入的微妙偏见，强调了谨慎的 Prompt Engineering 和模型选择的重要性。
- **阿里巴巴的 QwQ 32B 挑战巨头**：**阿里巴巴**发布了 **QwQ 32B 模型**，声称其性能可与 **DeepSeek R1 (671B)** 媲美，增强了向小型、高效开源模型发展的趋势，有关强化学习（RL）的细节可以在其 [博客文章](https://qwenlm.github.io/blog/qwq-32b/)中找到。
   - 虽然一些用户指出 **QwQ-32b** 经常遇到 **16k token 限制**，且在分离思考链（thinking trace）方面存在一致性问题，但其他人发现它与 **Qwen-thinking** 相似，还有人注意到新版本使用了 **Hermes 格式**。
- **知识图谱 GATs 软提示 LLMs**：一位成员正在将 **GAT** 的嵌入（embeddings）适配为 **LLM** 的软提示（soft prompt），以使用 **G-Retriever** 提供的框架生成受 **GAT** 约束的响应。
   - 另一位成员提到了关于 [Agentic、自主图扩展的论文](https://arxiv.org/abs/2502.13025) 以及 [OpenSPG/KAG GitHub 仓库](https://github.com/OpenSPG/KAG)，这是一个基于 OpenSPG 引擎和 LLMs 的逻辑表征引导推理与检索框架。
- **AI 说服力的潘多拉魔盒开启**：成员们正在讨论 **AI 说服 Agent**（persuasion agents）超越人类能力的潜力，可能会出现能够持续赢得辩论或吸引追随者的机器人。
   - 一位用户指出了 OpenAI 的 [evals make_me_say](https://github.com/openai/evals/tree/main/evals/elsuite/make_me_say) 说服力基准测试，而另一位用户指出新版本使用了 **Hermes 格式**。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SDXL 手部自动修复**：用户讨论了在 **SDXL** 中无需重绘（inpainting）即可自动修复手部的方法，推荐使用 *embeddings*、*face detailer* 以及添加 **OpenPose control net**，并寻找优质的 **hand LoRAs**。
   - 一位拥有 **8GB VRAM** 的用户询问了这些方法。
- **探索免费图生视频工具**：用户推荐使用 **Wan 2.1 i2v model** 从单张照片创建视频，但提醒这需要高性能 GPU 和耐心，并指向了 **SwarmUI** 的 [Video Model Support 文档](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Video%20Model%20Support.md#wan-21)。
   - 提到的另一个选项是提供免费额度的在线服务，但效果参差不齐。
- **本地视频生成在价格上优于 SORA**：讨论权衡了本地生成视频的成本（电费）与使用 **SORA** 等服务的成本，估计本地生成 5 秒视频的成本约为 **7 美分**，而 **SORA** 的成本可能为每段视频 **40 美分**。
   - 本地生成的优势：*无审查（uncensored）* 内容。
- **SD3.5 TurboX 正式开源**：TensorArt 开源了 **SD3.5 Large TurboX**，该模型使用 8 个采样步数，比原始模型实现 **6 倍速度提升**，且图像质量优于官方的 **Stable Diffusion 3.5 Turbo**；此外，**SD3.5 Medium TurboX** 在中端 GPU 上仅需 4 个采样步数即可在 1 秒内生成 **768x1248** 分辨率的图像。
   - 提供了 **SD3.5 Large TurboX** 的 [HuggingFace](https://huggingface.co/tensorart/stable-diffusion-3.5-large-TurboX) 链接和 **SD3.5 Medium TurboX** 的 [HuggingFace](https://huggingface.co/tensorart/stable-diffusion-3.5-medium-turbo) 链接。
- **Stable Diffusion 弃用 GPU**：一位用户报告 **Stable Diffusion** 正在使用 **CPU** 而非 **GPU**，导致图像生成缓慢（即使使用的是 **3070 Ti**），被建议尝试 **SwarmUI**。
   - 一名成员建议遵循 [GitHub](https://github.com/mcmonkeyprojects/SwarmUI) 上提供的安装说明。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **QwQ 32B 登陆 OpenRouter**：**QwQ 32B** 模型现已上线，提供[两个免费端点和一个快速端点](https://openrouter.ai/qwen/qwq-32b)，由 Grok 提供支持，速度达 **410 tokens/sec**。
   - 该模型在写入补全之前会进行*思考*，因为它现在默认包含 **reasoning**（推理）。
- **OpenRouter 新的 OAuth 和认证功能**：OpenRouter 在 OAuth 密钥创建中添加了 `user_id` 字段，使应用开发者能够创建个性化的用户体验；此外，**GitHub** 现在已成为 OpenRouter 的身份验证提供商！
   - 这将使 **OpenRouter** 与现有应用和工作流的集成变得更加容易。
- **Taiga 开源 Android 聊天应用发布**：一名成员发布了一款名为 **Taiga** 的[开源 Android 聊天应用](https://github.com/Ayuilos/Taiga/releases)，允许用户通过集成 **OpenRouter** 来自定义 **LLMs**。
   - 计划包括添加 **本地 Speech To Text**（基于 Whisper 模型和 Transformer.js）、**Text To Image 支持**以及基于 ChatTTS 的 **TTS 支持**。
- **DeepSeek 分词策略**：DeepSeek V3 的 [tokenizer 配置](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/tokenizer_config.json)显示其使用了 `<｜begin of sentence｜>` 和 `<｜end of sentence｜>` 标记，并且 *add_bos_token* 为 true，而 *add_eos_token* 为 false。
   - 还有人指出，**DeepSeek** 在其 R1 的 HF 页面上不建议进行多轮对话，并建议使用 `<think>\n` 进行预填充（prefilling）。
- **Google 停用 Gemini 2.0 之前的模型**：Google 宣布了 Vertex AI 上 Gemini 2.0 之前模型的[停用日期](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions)，计划于 **2025 年 4 月至 9 月**期间执行。
   - 受影响的模型包括 **PaLM, Codey, Gemini 1.0 Pro, Gemini 1.5 Pro/Flash 001/002** 以及部分 embeddings 模型。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **用户通过为未来功能提供反馈轻松赚取额外资金**：**NotebookLM** 团队正通过用户研究访谈（[报名表单](https://forms.gle/GxR2kwLdiXkzFMm89)）积极寻求用户对新概念的反馈，并提供**礼品卡**作为奖励。
   - 参与者参加 **15 分钟**的简短访谈可获得 **$50**，参加更深入的 **60 分钟**访谈可获得 **$100**，且只需极少的准备工作；兑换码由 Tremendous 通过电子邮件发送，要求参与者年满 **18 岁**，拥有 Google Drive 账号并具备稳定的网络连接。
- **玩家通过生成 JSON 历程获取游戏收益**：一位成员通过结合游戏文档、**JSON** 数据和电子表格提取内容，使用 **NotebookLM** 来优化在线游戏的策略，但发现该工具在迭代工作流和源文件编辑方面尚未完全优化。
   - 该成员认为 *“这个工具并没有针对我的用途进行优化”*，并希望能够直接编辑源文件。
- **PWA 填补 Android 应用空白**：虽然用户一直在呼吁推出 **NotebookLM** 的独立 **Android** 应用，但成员们强调，可以通过 Chrome 或 AI Studio 在手机和电脑上安装的 **PWA（渐进式 Web 应用）** 版本是一个功能性的替代方案。
   - 多位用户确认 **PWA** 运行良好，并可以保存到主屏幕。
- **Gemini 的灵活表现带来优质成果**：一位用户称赞将商务会议的录音加载到 **NotebookLM** 后，**Gemini** 能够进行转录并识别发言人。
   - 另一位用户指出这一过程被称为 *audio diarisation*（说话人日志），并推荐了 [ElevenLabs](https://elevenlabs.io/app/speech-to-text)，同时评论道 **Gemini** 在处理非标准口音方面的表现优于 **Whisper**。
- **笔记无法原生导出为 PDF 的噩梦**：用户对 **NotebookLM** 缺乏直接导出 **PDF** 的功能感到沮丧，不得不采用将笔记复制到文档中再下载为 PDF 等折中方案，正如在[功能请求讨论](https://discord.com/channels/1124402182171672732/1297146620626075681/1340698437749968907)中所讨论的那样。
   - 许多用户希望增强与 Google Drive、Docs 和 Sheets 的互操作性，特别是在导出和传输笔记方面。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Claude 每次查询收费几美分**：一位用户报告称，向 **Claude** 询问一个关于其小型代码库的问题花费了 **$0.26**。
   - 另一位用户建议将代码库复制到 **Claude** 目录中，利用文件系统 **MCP** 服务器，通过 Claude 订阅中的 token 来实现 *“免费”* 使用。
- **苹果发布 M4 MacBook Air**：苹果发布了新款 [MacBook Air](https://www.apple.com/newsroom/2025/03/apple-introduces-the-new-macbook-air-with-the-m4-chip-and-a-sky-blue-color/)，搭载 **M4 芯片**，具备 **Apple Intelligence** 功能，并新增了**天蓝色**，起售价 **$999**。
   - 新款 **MacBook Air** 提供了前所未有的价值，拥有更强的性能、长达 **18 小时**的电池续航、**12MP Center Stage 摄像头**以及增强的外接显示器支持。
- **阿里巴巴的 QwQ-32B 挑战推理巨头**：阿里巴巴发布了 [QwQ-32B](https://qwenlm.github.io/blog/qwq-32b)，这是一款拥有 **320 亿参数**的新型推理模型，可与 **DeepSeek-R1** 等顶尖推理模型相媲美。
   - 会议强调，**RL 训练**可以持续提升性能，尤其是在数学和编程方面，帮助中型模型在面对巨大的 **MoE 模型**时获得具有竞争力的表现。
- **React：LLM 后端的下一个前沿？**：一位成员发布了一篇博文，认为 [React 是后端 LLM 工作流的最佳编程模型](https://x.com/_Evan_Boyle/status/1897347251120562205)。
   - 另一位用户表示，这种方法听起来像是在重新发明 **Lisp**，关键在于 *“设计出既符合应用所需的组合性，又对 LLM 具有可读性的代码模式”*。
- **Carlini 跳槽至 Anthropic**：[Nicholas Carlini](https://nicholas.carlini.com/writing/2025/career-update.html) 宣布在 **Google DeepMind** 工作七年后离职，加入 **Anthropic** 为期一年，以继续他在对抗性机器学习方面的研究。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Synalinks 作为 DSPy 的替代方案首次亮相**：一个名为 **Synalinks** 的新型**基于图的可编程神经符号 LM 框架**正式推出，该框架从 **Keras** 中汲取灵感，专注于**知识图谱 RAG**、**强化学习**和**认知架构**。
   - 该框架旨在实现完全的**异步优化**，具有**默认约束结构化输出**的特性，并提供 **functional API**，现已提供 [代码示例](https://huggingface.co/spaces/YoanSallami/synalinks-noteboooks)。
- **Synalinks 倾向于传统编码**：**Synalinks** 的创建者提到，几乎没有任何代码库是使用 AI 创建的，并表示 *"在成熟的开源系统之上构建的传统方式，比使用 AI 从头开始编写要好 10000 倍。"*
   - 对方澄清说，该框架不一定是 **DSPy** 的替代品，而是一种专注于**提示词优化**、**强化学习**和**图 RAG** 的不同方法。
- **DSPy 提升意图分类效果**：使用 **DSPy** 可以通过专门的 Agent 帮助优化意图分类。
   - 一位用户确认，使用 DSPy 是满足其意图分类需求的正确方向。
- **滞后线程拖慢并行 DSPy**：[已合并的 PR 7914](https://github.com/stanford-nlp/dspy/pull/7914) 通过修复“滞后”线程，使 **DSPy 的 `dspy.Evaluate` 或 `dspy.Parallel`** 运行更顺畅。
   - 用户可以在 DSPy 2.6.11 发布之前从 `main` 分支进行尝试，无需更改代码，但需要从 main 分支获取库。
- **带有 DSPy Signatures 的可变输出字段**：一位用户询问如何创建一个具有可变输出字段的 **dspy.Signature**，例如，有时输出 A、B、C，有时输出 D、E 和 F。
   - 一名成员建议查看 [react.py](https://github.com/stanford-nlp/dspy/blob/main/dspy/experimental/react.py) 文件。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 与 DeepLearningAI 合作**：LlamaIndex 已与 [DeepLearningAI](https://t.co/EvAKtIAzlC) 合作，提供关于构建 **Agentic 文档工作流**的短期课程，强调将其集成到更大的软件流程中。
   - 重点是将这些工作流作为知识 Agent 的未来进行利用。
- **LlamaIndex 倡导开放 Agent 标准**：根据[此公告](https://t.co/ECHH1T4Kxn)，LlamaIndex 正在参与创建**开放、可互操作的 Agent 标准**，涵盖从发现到部署以及互联互通的各个方面。
   - 目标是为 AI Agent 培育一个更加互联和协作的生态系统。
- **OpenAI ImageBlock 集成面临识别问题**：用户报告了在最新版 LlamaIndex 中与 OpenAI 配合使用 **ImageBlock** 时图像无法被识别的问题；排查过程包括检查最新的 LlamaIndex 版本，并确保使用支持图像输入的模型，如 **gpt-4-vision-preview**。
   - 为了解决该问题，还强调了对 OpenAI LLM 实例进行正确配置的重要性。
- **QueryFusion 检索引用问题**：据 [此 GitHub 仓库](https://github.com/Restodecoca/ingest-reposit/tree/main/app/engine) 报告，将 **QueryFusionRetriever** 与节点后处理器配合使用时，无法生成引用模板，而单独使用 **index_retriever** 则没有问题。
   - 该问题可能源于 **BM25 检索器**或**查询融合检索器**的倒数重排序（reciprocal rerank），可能导致在节点去重过程中丢失元数据。
- **分布式 AgentWorkflows 寻求原生支持**：一位用户询问了在分布式架构中运行 **AgentWorkflow** 的原生支持，即 Agent 位于不同的服务器或进程中。
   - 官方建议 **AgentWorkflow** 是为单个活动 Agent 设计的，实现所需的设置可能需要为 Agent 配备用于远程服务调用的工具。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **关于 Sparsemax 的双层优化辩论**：围绕 **双层优化 (BO)** 在 **Sparsemax** 中的适用性展开了辩论，一名成员认为 BO 是等同于单层优化的标准形式，而另一名成员建议将 Sparsemax 视为一种 BO。
   - 讨论涉及将层级结构折叠为单层以获得闭式解，这在事物尽可能简单时效果最好。
- **DDP 模式下 Checkpoint 重新加载出现乱码**：一位成员在使用 **PyTorch**、**DDP** 和 **4 张 GPU** 时遇到了模型 checkpoint 重新加载在多 GPU 上出现乱码的问题，但在单 GPU 上运行完美。
   - 建议指出初始化 **DDP** 和加载 checkpoint 的顺序至关重要：应先初始化模型，在所有 GPU 上加载 checkpoint，然后再初始化 DDP。
- **引入用于复合 Arg Max 的 Compositmax**：一位成员介绍了用于复合 arg max 的 **Compositmax**，并指出 **Softmax** 是 soft arg max，**Sparsemax** 是 sparse arg max，而 **Entmax** 是 entropy arg max。
   - 目标是基于样条线（splines）的思想设计新的正则化器，旨在实现比 entmax 更快的性能。
- **主动型 Agent 寻求图像意图**：一篇关于 [不确定性下的多轮文本生成图像主动型 Agent](https://arxiv.org/abs/2412.06771) 的新论文介绍了 **主动型 T2I Agent**，它们在不确定时会主动询问澄清性问题，并将其对用户意图的理解呈现为可理解的信念图（belief graph）。
   - Meera Hahn 关于主动型 Agent 的 **Google TechTalk** 强调，生成式 AI 模型的 **用户提示词（user prompts）** 通常指定不足，导致响应并非最优，正如 [这段 YouTube 视频](https://youtu.be/HQgjLWp4Lo8?si=6SxQdUbzocp3zrKD) 中所述。
- **阿里巴巴 Qwen 发布 QwQ-32B 模型**：**阿里巴巴 Qwen** 发布了 **QwQ-32B**，这是一款仅有 **320 亿参数** 的新型推理模型，其性能可与 [此推文](https://x.com/Alibaba_Qwen/status/1897361654763151544?t=t5Bec1knVsQuXpTu24fWqw&s=19) 中提到的 **DeepSeek-R1** 等顶尖推理模型相媲美。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Suleiman 探索 AI 赋能的生物黑客技术**：Suleiman 介绍了自己，表达了对开发 **AI 赋能的生物黑客工具** 的浓厚兴趣，旨在通过 **营养学** 和 **补充剂科学** 改善人类健康。
   - Suleiman 拥有软件工程背景，并曾在一家沙特公司担任高管。
- **Naveen 推动机器去学习研究**：Naveen 介绍了自己及其在 **文本到图像扩散模型中的机器去学习 (Machine Unlearning)** 方面的研究，最近在 **CVPR25** 发表了一篇论文。
   - Naveen 是来自 IIT 的硕士兼研究助理。
- **ARC 训练达到 35% 准确率**：成员们报告称，仅使用推理时示例在 **ARC 训练** 上达到了 **35%** 的准确率，引用了 [Isaac Liao 和 Albert Gu 的博客文章](https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html)，该文章质疑 *高效压缩是否是智能的核心*。
   - 一位成员链接了一篇关于 [相对熵编码 (REC)](https://arxiv.org/abs/2010.01185) 的论文，认为它是所讨论的无损压缩方法的主要基础。
- **Tuned Lens 优于 Logit Lens**：成员们讨论了将中间层输出投影到词表空间的方法，分享了 [Tuned Lens: Iterative Refinement with Interpretable Differentiable Probes](https://arxiv.org/abs/2303.08112)，该研究改进了 **logit lens** 技术。
   - 建议使用 **tuned lens** 代替 **logit lens**，复现结果所需的 [代码](https://github.com/AlignmentResearch/tuned-lens) 可以在 GitHub 上找到。
- **vllm 面临实现细节质询**：一位成员报告称，在 `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` 模型上使用 **vllm** 运行 `lm_eval` 时，分数出现了显著差异。
   - 另一名成员认为问题可能源于 **vllm 的实现**，并表示如果有样本可用，愿意进行调查。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Aya Vision 扩展至 23 种语言**：Cohere For AI 推出了 **Aya Vision**，这是一个权重开放的多语言视觉研究模型，提供 **8B 和 32B** 版本，支持 **23 种语言**，并针对各种视觉语言用例优化了高级功能，详情见 [Cohere 的博客文章](https://cohere.com/blog/aya-vision)。
   - 该模型目前已在 [Hugging Face](https://huggingface.co/collections/CohereForAI/c4ai-aya-vision) 和 [Kaggle](https://www.kaggle.com/models/cohereforai/aya-vision) 上线，并可通过 [Poe](https://poe.com/Aya-Vision) 访问；用户现在还可以通过[此链接](https://cohere.com/research/aya/whatsapp)在 WhatsApp 上从全球任何地方使用 **23 种语言**与 Aya 免费互动。
- **企业支持响应时间受到质疑**：用户 **brad062677** 对企业支持响应速度缓慢表示不满，指出他们在一周前就给支持团队发了邮件，并试图通过 Discord 寻求更快的解决方案；该用户正尝试联系 **sales / enterprises support** 团队的成员。
   - 其他用户指出，B2B 的交付周期可能长达 **六周**，而典型的 AI 公司响应时间通常为 **两到三天**；一名 Cohere 员工对此表示歉意并承诺会给予回复。
- **Reranker v3.5 延迟数据仍然缺失**：社区成员正在寻找 **Cohere Reranker v3.5** 的延迟数据，该数据最初在 [Pinecone 访谈](https://www.pinecone.io/learn/cohere-rerank/)中有所提及，但尚未正式发布。
   - 由于缺乏 **Cohere Reranker v3.5** 的具体延迟数值或图表，一些用户正积极寻求这些信息以便进行性能评估和对比。
- **学生构思思维导图项目方案**：一名学生正在开发一个根据章节内容生成思维导图的网站，旨在构建主题和子主题的分层结构，计划最初使用预训练模型或创建自定义数学模型。
   - 该学生正在寻求关于如何将这两种方法整合到项目中的最佳方案指导，并寻找最佳切入点的建议。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **ShapeTracker 合并证明接近完成**：一个关于合并 **ShapeTrackers** 的 Lean 证明已接近完成，可在[此仓库](https://github.com/Nielius/Tensorlayouts)中查看，更多背景信息见[此 issue](https://github.com/tinygrad/tinygrad/issues/8511#issuecomment-2700706082)。
   - 该证明目前省略了偏移量 (offsets) 和掩码 (masks)，但据信通过进一步努力，将其扩展到包含这些因素是可行的。
- **淘宝上发现 96GB 4090**：淘宝上出现了正在出售的 **96GB 4090**（[X 帖子](https://x.com/yomix1337/status/1893692548108984391?s=46)），引发了人们对本地训练更高显存容量的兴奋。
   - 距离正式上市可能还有几个月的时间。
- **Rust CubeCL 质量受到询问**：鉴于 **Rust CubeCL** 是由开发 **Rust Burn** 的同一团队创建的，人们对其质量产生了兴趣。
   - 该成员*想知道 Rust CubeCL 是否好用*。
- **寻求关于 RANGE Op 操作的澄清**：一位成员最初对 `RANGE` Op 的操作提出疑问，推测它在 `arrange` 的 `Tensor` 实现中缺失。
   - 然而，该成员随后消除了困惑，澄清它 *“不是一个 range”*。
- **Linux 上的 iGPU 自动检测受到质疑**：一位用户质疑默认的设备初始化或 `Device.get_available_devices()` 是否应该在 Linux 上自动检测到 **iGPU**。
   - 他们的帖子中包含一张显示 *“Device: [CPU]”* 的图片，这超出了用户的预期。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **TorchTune 复制原始 Special Tokens**：**TorchTune checkpointer** 会从 Hugging Face 复制原始的 **special_tokens.json**，而不是可能经过修改的自定义版本，代码参考[此处](https://github.com/pytorch/torchtune/blob/80da6a5dae23a201595d07041c12ffde830332d7/torchtune/training/checkpointing/_checkpointer.py#L892-L896)。
   - 团队决定在没有充分理由的情况下不增加新的参数，因此目前的建议是暂时手动复制该文件。
- **Torchtune GitHub Stars 突破 5k**：Torchtune 项目在 **GitHub 上达到了 5,000 stars**。
   - 社区对此成就表示祝贺。
- **GRPO Recipe 存在过度使用 Empty Cache 的问题**：一位成员询问了 **GRPO recipe** 中过度调用 `torch.cuda.empty_cache()` 的情况。
   - 另一位成员承认，其中许多调用可能是多余的，这源于早期开发时面临的 **内存问题 (memory issues)**。
- **GRPO PR 进度停滞**：两个 **GRPO PR**（特别是 **#2422** 和 **#2425**）已经开启两周，正等待审查。
   - 一位成员请求协助审查，希望有人能帮忙分担积压的队列。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **MOOC 讲座与伯克利校内课程一致**：一位成员询问伯克利学生是否拥有 MOOC 之外的专属讲座，一位同事回答说 **伯克利学生和 MOOC 学生参加的是相同的讲座**。
   - 关于讲座的具体内容没有进一步的评论。
- **证书发放延迟**：一位成员报告称在 12 月提交了证书申报表，但收到通知称 **没有记录到提交信息**。
   - 这个问题在 #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1346810951265157201) 中被提出，没有更多细节，但这可能表明 MOOC 存在系统性问题。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **AST 指标仍显神秘**：一位成员询问 **AST (Abstract Syntax Tree) 指标** 的含义，特别是它是否衡量 LLM 生成的函数调用格式正确的百分比。
   - 该询问在频道中未得到解答。
- **V1 数据集来源未知**：一位用户询问 **V1 数据集** 的构建方式。
   - 与关于 **AST 指标** 的查询一样，这个问题也没有得到回应。
- **Python 工具调用冠军仍未定论**：一位成员寻求关于 prompt tool calling 最佳模型的建议，考虑对象包括 **Gemini 2**、**GPT o3-high** 和 **Deepseek R1**。
   - 具体的用例涉及调用 **Python tool**。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **AI21 Labs 发布 Jamba 1.6**：AI21 Labs 推出了 **Jamba 1.6**，这是一个专为私有企业部署量身定制的开放模型，模型权重可在 [Hugging Face](https://huggingface.co/ai21labs) 上获取。
   - 公司声称它*提供了无与伦比的速度和性能*，在不牺牲效率、安全性和数据隐私的情况下，为企业级 AI 树立了新标杆。
- **Jamba 1.6 展示 Arena 实力**：据 [AI21 的公告](https://www.ai21.com/jamba/)，**Jamba 1.6** 在 Arena Hard 基准测试中表现优于 **Cohere**、**Mistral** 和 **Llama**，足以与领先的封闭模型相媲美。
   - 该版本强调其适用于完全私有的本地或 VPC 部署，拥有极低的延迟和市场领先的 **256K context window**。
- **混合架构赋予 Jamba 1.6 优势**：**AI21 Jamba** 系列采用混合 **SSM-Transformer** 基础模型，在质量和速度上均表现出色，这归功于其新颖的 **Mamba-Transformer MoE 架构**，旨在实现成本和效率的提升，详见 [Jamba 1.6 博客文章](https://www.ai21.com/jamba/)。
   - 该模型可以部署在任何地方，无论是自托管还是在 AI21 SaaS 中，以满足多样化的数据安全需求。

---

**MLOps @Chipro Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道详细摘要和链接

{% if medium == 'web' %}

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1346800454369017957)** (1303 messages🔥🔥🔥): 

> `Sonnet 3.7, Qwen, Windsurf IDE, MCP Client Closed, OpenRouter API` 


- **Agent 乱象：Cursor 的代码编写灾难仍在继续**：多位用户报告称 **Cursor agents 仍在基础任务上挣扎**，例如*查找文件*和*编辑代码*，一位用户指出 **Claude API 在 2 天内花费了 20 美元**，却没能获得更好的结果。
   - 另一位用户插话称 **Sonnet 3.7** 已经不再像个疯子，重新变得好用了，而其他人仍在尝试寻找解决所遇问题的办法。
- **Qwen 夺取推理桂冠，废黜 DeepSeek R1**：阿里巴巴的 **QwQ-32B** 被称作可以与 **DeepSeek-R1** 媲美，而其参数量少了 20 倍，甚至比 **DeepSeek R1** 的 37B 激活参数量还要少。
   - 据用户称，*这只是个恶搞基准测试*，但其他来源声称 **QwQ-32B 的 GPQA Diamond 分数达到了 59.5%**。
- **Windsurf 的 Wave 4 更新正在搅局 Cursor**：用户报告称 **Windsurf 的 Wave 4** 更新在配合 **Sonnet 3.5** 使用时表现流畅，另一位用户称其遇到了 *try again* 问题，而另一位则表示它在 linting 方面不如 Cursor。
   - 一位用户还报告称 **Cursor IDE** 无法修改文件。
- **MCP 混乱：Client Closed 灾难困扰开发者**：用户在 Windows 上使用 **MCP Servers** 时面临问题，遇到了 *Client Closed* 错误，一些人尝试寻找短期解决方案，而另一些人则不断寻找临时修复方法。
   - 一位用户提到了一种涉及在 CMD 终端运行命令的解决方案，而其他人则未能修复该问题。
- **OpenRouter API 访问**：一些用户正在讨论使用官方 API 还是 **OpenRouter**，引擎为 **Claude Code**，其中提到即使是 **Claude-max** 每次请求也要花费 2 美元，而其他用户则触及了 API 限制。
   - 成员们讨论 Cursor 相比 API 可能*定价过高*，导致需要切换，而其他人则没有达到这些限制，并不介意为服务付费。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/techfrenaj/status/1897337662769672309?s=46">来自 techfren (@techfrenAJ) 的推文</a>：介绍 Chaos Coder，开源并已部署，链接见评论区。</li><li><a href="https://x.com/artificialanlys/status/1897701015803380112?s=46">来自 Artificial Analysis (@ArtificialAnlys) 的推文</a>：阿里巴巴发布 QwQ-32B，这是一款开源权重的推理模型，其智能水平可能接近 DeepSeek R1。我们整晚都在对其进行评估，目前仅获得了 GPQA D 的分数...</li><li><a href="https://container-seven-sigma.vercel.app">Container Ejection Simulator</a>：未找到描述</li><li><a href="https://elitecaptures7.com/">Elitecaptures7</a>：未找到描述</li><li><a href="https://templeos.org">TempleOS</a>：未找到描述</li><li><a href="https://fontawesome.com/icons/house?f=classic&s=solid">House Classic Solid Icon | Font Awesome</a>：Solid 风格的 House 图标。在小尺寸下也能彰显个性。现已在 Font Awesome 6 中推出。</li><li><a href="https://codeinplace.stanford.edu">Code In Place</a>：未找到描述</li><li><a href="https://code.visualstudio.com/updates/v1_98">2025 年 2 月（版本 1.98）</a>：了解 Visual Studio Code 2025 年 2 月发布版 (1.98) 的新功能</li><li><a href="https://x.com/steph_palazzolo/status/1897309493744267314?s=46">来自 Stephanie Palazzolo (@steph_palazzolo) 的推文</a>：最新消息（与 @coryweinberg 合作）：OpenAI 正在加倍投入其应用业务。高管们已与投资者讨论了三类未来的 Agent 发布计划，价格从每月 2,000 美元到 20,000 美元不等，用于执行诸如...的任务。</li><li><a href="https://github.com/ollama/ollama/commit/dc13813a03105bd76603a4909e31ba0c034d670d">server: allow vscode-file origins (#9313) · ollama/ollama@dc13813</a>：未找到描述</li><li><a href="https://github.com/agno-agi/agno">GitHub - agno-agi/agno: 构建具有记忆、知识和工具的多模态 AI Agents。简单、快速且与模型无关。</a>：构建具有记忆、知识和工具的多模态 AI Agents。简单、快速且与模型无关。 - agno-agi/agno</li><li><a href="https://tenor.com/view/good-weekend-gif-6442363721098209555">Good Weekend GIF - Good weekend - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://xd.adobe.com/embed/2bf05be6-17a0-40a9-a92c-56310b487db8-7ea3/?fullscreen"">Elitecaptures7-v2</a>：74 个屏幕，最后修改于 2022 年 6 月 14 日 22:11 GMT
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1346922878729457785)** (4 条消息): 

> `AGI 发展, OpenAI o1 和 o3-mini, ChatGPT for macOS, AI 安全与对齐` 


- **GPT-4.5 推出提前完成**：**GPT-4.5** 的推出已经完成，官方分享了他们在 **AI 安全与对齐** 方面的方法见解。
   - **AGI 发展** 被视为一条 *持续的路径*，而非单一的关键时刻，通过对现有模型的迭代部署和学习，使未来的 AI 更加安全且有益。
- **o1 和 o3-mini 加入 OpenAI API**：**OpenAI o1 和 o3-mini** 现已在 API 中向所有付费层级的开发者开放，并支持 [Streaming, Function calling, Structured Outputs, Reasoning effort, Assistants API, Batch API](https://platform.openai.com/docs/models/compare?model=o1) 以及 Vision（仅限 o1）。
   - 他们的方法以拥抱 **不确定性**、**纵深防御**、**可扩展的方法**、**人类控制** 以及 **社区努力** 为指导，以确保 AGI 造福全人类。
- **MacOS 版 ChatGPT 现在可以在 IDE 中编辑代码**：**ChatGPT for macOS** 现在可以直接在 IDE 中编辑代码。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1346817573722914846)** (822 条消息🔥🔥🔥): 

> `Grok3 vs Claude, DeepSeek, Atom of Thought, Microsoft Phi 模型, Apple 统一内存` 


- **Gemini 表现乏力，Grok 崭露头角**：成员们注意到 **Gemini** 的表现像 **GPT-3.5**，而其他人由于 **Grok3** 更好的性能和慷慨的配额而转向使用它，尽管有一位成员表示“除了 Grok 以外任何模型都行”。
   - 他们进一步指出，**Grok3** 说话像 **GPT-4.5** 一样自然，代码能力优于 **Sonnet 3.7**，配额充足，而且会说脏话 (f bombs)。
- **DeepSeek R1 蒸馏推理**：社区正在讨论 **DeepSeek R1 Distill** 模型的推理能力，一些用户注意到它是听起来最自然的 LLM 之一，但另一位成员表示，在没有提供知识的情况下，它“感觉不够聪明”。
   - 成员们还提到，他们一直在尝试使用 **Atom of Thought** 来实现类似水平的推理，并且 [有一篇论文](https://arxiv.org/abs/2412.06769) 介绍了如何使用原始 embeddings 作为“Token”来实现 CoT。
- **Microsoft Phi-4 现已发布**：在一位用户询问 **Phi-2** 对提示词优化的实用性后，其他成员建议改用 **Phi-4**，因为它在性能和功能上都有所提升，尽管其更大的尺寸需要更多的 VRAM。
   - 成员们指出，该套件中包含多个模型，并不局限于最初的 14.7B 模型。
- **Apple 的统一内存：训练领域的颠覆者？**：一位成员注意到 **Apple** 发布了一款配备 **512GB 统一内存** 的电脑，这对于模型训练可能很有趣，但另一位成员指出，**1 万美元** 的价格需要财力雄厚。
   - 成员们提到了 LPDDR5x 较低的内存带宽，但仍指出某些模型可以在如此大的内存下以 FP4 模式运行。 


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pastebin.com/Zez0gt1R">&quot;&quot;&quot;atom_of_thought.py-----------------这个模块实现了 &quot;Atom of - Pastebin.com</a>: Pastebin.com 是自 2002 年以来排名第一的文本存储工具。Pastebin 是一个可以在线存储文本并保留一段时间的网站。</li><li><a href="https://www.youtube.com/watch?v=yD4NrND3NC0">第一部分：AI 摧毁了 Tom 的职业生涯。我们其他人会是下一个吗？</a>: 像 Elon Musk 这样的科技亿万富翁表示，他们正在构建的 AI 系统将取代许多人的工作，但也会在原位创造更好的工作。事实是否如此...</li><li><a href="https://youtu.be/Vshg-hNUEjo">Nana Mouskouri - Guten Morgen, Sonnenschein 1977</a>: Nana Mouskouri - Guten Morgen, Sonnenschein 1977
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1346907244192596150)** (24 messages🔥): 

> `GPT-4.5 Availability and Limitations, GPT-4.5 vs GPT-4o Performance, GPT-4.5 Prompting Strategies, GPT-4.5 Personalization Prompt, GPT-4.5 Mobile Issues` 


- **GPT-4.5 的可用性受限**：成员们注意到可用性限制在**每周约 50 次使用**，但随着 OpenAI 收集反馈，可能会逐渐增加。
   - 明确了 **GPT-4.5** 并不是 **GPT-4o** 等其他模型的直接替代品，用户应根据每项任务选择最合适的模型。
- **GPT-4.5 在准确性上优于 GPT-4o，但在写作方面稍逊一筹**：一些用户发现 **GPT-4.5** 在创意写作任务中的表现不如 **GPT-4o**，而另一些用户则报告其在文档分析方面的表现有所提升。
   - 一位用户报告称，虽然 **GPT-4.5** 在准确性和世界知识方面更好，但他们需要提醒它或重新发送消息才能让它完成一份文档。
- **GPT-4.5 需要详细的信任建立 Prompt**：用户报告称 **GPT-4.5** 需要更详细、更长的 Prompt（最好使用 Markdown 格式）才能获得最佳结果。
   - 一位用户建议，在提出复杂请求之前，通过礼貌的信息与 **GPT-4.5** 建立信任可以提高响应质量，并提供了一个[个性化 Prompt 示例](paste.link.here)以增强细微的推理和情感连接。
- **GPT-4.5 存在 Android 移动端兼容性问题**：一位用户指出 **GPT-4.5** 无法在 Android 手机（包括 App 和浏览器）上运行，但在 iOS 上运行正常。
   - 该用户解释说 **GPT-4.5** 产生了错误信息：*"I'm sorry, but I won't be able to help with that request."*


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1346891297771749477)** (13 messages🔥): 

> `Prompt Engineering Techniques Ontology, Sora AI Video Character Consistency, GPT-4o OCR Bounding Box Issues` 


- **Prompt Engineering 技术趋于系统化**：一位成员分享了来自《A Systematic Survey of Prompt Engineering in Large Language Models: Techniques and Applications》的 Prompt Engineering 技术概览，包括 **Zero-Shot**、**Few-Shot**、**Chain-of-Thought (CoT)** 和 **Retrieval Augmented Generation (RAG)**。
   - 发布者还指出论文中的本体并不详尽，遗漏了 **Self-Discover** 和 **MedPrompt** 等技术，随后分享了一个 [ChatGPT 链接](https://chatgpt.com/share/67c89f53-c72c-8000-b64b-ca30c6971854)以供更广泛的查阅。
- **Sora 用户寻求 Isabella Moretti 的角色一致性**：一位成员正在使用 **Sora** 创作电影感的 AI 视频，专注于一个名为 **Isabella Moretti** 的角色，旨在实现超写实的视觉效果以及跨多个片段的一致角色细节。
   - 创作者正在寻求增强现实感并在**肤色**、**眼睛**和**头发**等细节上保持一致性的策略或 Prompt 技巧。
- **GPT-4o 在处理边界框（Bounding Boxes）时遇到困难**：一位用户报告了在使用 OpenAI API 时，**GPT-4o 模型**在 OCR 结果中返回的**边界框坐标**不准确的问题。
   - 他们请求关于如何从 OpenAI API 获取带有坐标的准确 OCR 结果的建议，但未能提供图像样本。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1346891297771749477)** (13 条消息🔥): 

> `Prompt Engineering 综述，Sora AI 视频角色一致性，GPT-4o OCR 结果` 


- **Prompt Engineering 技术种类繁多**：一位成员分享了来自《A Systematic Survey of Prompt Engineering in Large Language Models: Techniques and Applications》的 Prompt Engineering 技术概览，概述了 **zero-shot prompting**、**chain-of-thought** 和 **RAG** 等类别。
   - 为了更广泛的访问，还分享了一个指向相同概览的 [ChatGPT 链接](https://chatgpt.com/share/67c89f53-c72c-8000-b64b-ca30c6971854)，并指出这*甚至不是一个详尽的本体！它遗漏了 Self-Discover 和 MedPrompt 等内容！*。
- **Sora 以电影感生成一致的角色**：一位成员正在使用 **Sora** 创作电影级的 AI 视频，重点关注一个名为 **Isabella Moretti** 的角色，并寻求实现**超写实视觉效果**以及提高多个片段中角色一致性的策略。
   - 创作者特别旨在保持 **skin tone**、**eyes**、**hair** 和 **expressions** 等细节的一致性，同时优化 Prompt 结构以获得最佳电影质量，包括 **lighting**、**camera movements** 和 **transitions**。
- **GPT-4o 的 OCR bounding box 坐标不正确**：一位成员报告了在使用 **GPT-4o** 模型的 **OpenAI API** 获取 **OCR 结果**时，无法获得正确的 **bounding box** 坐标的问题。
   - 他们使用的 Prompt 返回了错误的 **bounding box** 坐标，目前正在寻求解决此问题的建议。


---

### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1346939336792342611)** (2 条消息): 

> `Windsurf Wave 4, Cascade updates, Windsurf Previews, Auto-Linter in Cascade, MCP Server updates` 


- **Windsurf Wave 4 发布，带来重大更新**：最新的 **Windsurf Wave 4** 版本引入了颠覆性的功能，如 **Previews**、**Tab-to-import**、**Linter 集成**和 **Suggested actions**，同时改进了 **MCP 可发现性**和 **Claude 3.7** 集成，详见[博客文章](https://www.codeium.com/blog/windsurf-wave-4)。
- **Cascade 支持元素预览选择**：**Cascade** 现在允许你在 IDE 或浏览器中预览本地运行的网站。
   - 用户可以在预览中选择 **React** 和 **HTML** 元素并将其作为上下文发送给 **Cascade**，从而简化对话流程，如 [X/Twitter 公告](https://x.com/windsurf_ai/status/1897378545799979238)所示。
- **Codeium 修复 Preview 路由加载问题**：根据[完整变更日志](https://www.codeium.com/changelog)，官方发布了一个补丁来解决 **Windsurf Previews** 中某些路由无法加载的问题，同时修复了打开 **Cascade** 快捷键的问题，并恢复了缺失的代理设置和索引大小选项。
- **Cascade 引入 Auto-Linter 实现无缝代码纠错**：**Windsurf Wave 4** 更新将 **Auto-Linter** 直接集成到 **Cascade** 中，它可以自动修复生成代码中的 lint 错误，确保更整洁的代码输出。
   - 更多详情请查看 [YouTube 视频](https://www.youtube.com/watch?v=bIy-RN3FIsQ&feature=youtu.be)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.codeium.com/changelog">Windsurf Editor Changelogs | Windsurf Editor and Codeium extensions</a>：Windsurf 编辑器的最新更新和变化。</li><li><a href="https://www.codeium.com/blog/windsurf-wave-4">Windsurf Wave 4</a>：介绍 Wave 4，我们对 Windsurf 编辑器的第四批更新。</li><li><a href="https://x.com/windsurf_ai/status/1897378545799979238">来自 Windsurf (@windsurf_ai) 的推文</a>：Windsurf Wave 4 来了！本次更新包含：🖼️ Previews ✏️ Cascade Auto-Linter ⚙️ MCP UI Improvements ➡️ Tab to Import ↩️ Suggested Actions 🫶 Claude 3.7 Improvements 🤝 Referrals 🖥️ Windows ARM Suppo...</li><li><a href="https://bsky.app/profile/windsurfai.bsky.social/post/3ljnsaugqk22l">Windsurf (@windsurfai.bsky.social)</a>：Windsurf Wave 4 来了！本次更新包含：🖼️ Previews ✏️ Cascade Auto-Linter ⚙️ MCP UI Improvements ▶️ Tab to Import ↩️ Suggested Actions 🫶 Claude 3.7 Improvements 🤝 Referrals 🖥️ Windows ARM Suppor...</li><li><a href="https://www.threads.net/@codeiumdev/post/DG1IyC5CODS?xmt=AQGzB0CoP8oQ9hE-8YatsFH7FaIFFpnONInUNHCSr9H8qg">Threads 上的 Codeium (&#064;codeiumdev)</a>：Windsurf Wave 4 来了！本次更新包含：&#x1f5bc;&#xfe0f; Previews &#x270f;&#xfe0f; Cascade Auto-Linter &#x2699;&#xfe0f; MCP UI Improvements &#x25b6;&#xfe0f; Tab to Import &#x21a9;&#xfe0f; Suggest...</li><li><a href="https://www.youtube.com/watch?v=bIy-RN3FIsQ&feature=youtu.be">Windsurf Wave 4 Updates: Preview, Tab to Import, Suggested Actions &amp; More</a>：Windsurf Wave 4 来了，带来了令人兴奋的新功能以提升您的体验！🌊 请确保更新到最新版本的 Windsurf 以获取所有这些功能...
</li>
</ul>

</div>
  

---

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1346812925649748030)** (45 条消息🔥): 

> `VS Code Commit Message 生成问题、Flutterflow 使用、Codeium 卸载、Codeium Language Server 下载问题、Codeium Chat 功能中的遥测数据` 


- **用户面临 VS Code Commit Message 生成问题**：一位用户报告了在使用 Codeium 预发布版本时 VS Code 生成提交信息的问题，并寻求解决方法。
   - 其他用户还询问了关于 **Flutterflow** 以及如何**彻底卸载**当前 **Codeium 扩展**的问题。
- **Codeium Language Server 下载器出现问题**：多位用户报告 **Codeium 无法下载 Language Server**，具体错误信息指向 `releases.codeiumdata.com` 的一个下载 URL。
   - 即使重启 IDE 后问题依然存在，且在 **WSL** 和 **Windows** 安装环境下均有发生。
- **个人账户需要启用代码片段遥测（Code Snippet Telemetry）才能开启 Chat 功能**：一位试用个人账户的用户质疑为何必须开启代码片段共享才能使用 Chat 功能，并指出 FAQ 中关于遥测（Telemetry）的信息存在矛盾。
   - 另一位用户提到，他们在购买 Pro 方案前在非敏感数据上**测试了 Codeium，当时 Chat 功能在未开启代码片段遥测的情况下也可以工作**，并提供了 [Codeium FAQ 的数据隐私部分](https://codeium.com/faq#data-privacy)链接。
- **“Disabled by team...” 错误困扰用户**：一位用户报告其账号反复卡在 *“disabled by team...”* 状态，尽管多次重新安装软件，仍无法使用扩展。
- **提醒：与扩展程序无关的 Windsurf 话题应发布在 Windsurf 频道**：一位用户指出，与 **Windsurf** 相关（且与扩展本身无直接关系）的话题应转至特定的 **Windsurf 频道**，如 <#1306163501286293515>、<#1324809403920023604> 或 <#1334764563064688692>。



**提及链接**：<a href="https://codeium.com/faq#data-privacy">FAQ | Windsurf Editor and Codeium extensions</a>：查找常见问题的答案。

  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1346807307966746674)** (609 条消息🔥🔥🔥): 

> `Windsurf 稳定性问题、额度消耗担忧、回滚功能请求、3.7 性能困扰、全局规则与 .windsurfrules` 


- **Windsurf 深受更新后遗症困扰**：用户报告在最新的 Windsurf 更新（Wave 4, v1.4.3）后出现**稳定性问题**，包括频繁的调用中断、卡顿和内部记忆幻觉，导致部分用户转向 Cursor 或 Trae 等替代 IDE。
   - 一位用户抱怨道：*“WS 自从上次更新后就不像以前那样好用了。调用中断太多、卡顿、上下文记忆能力差、奇怪的内部记忆幻觉等。”*
- **额度紧缺困扰 Codeium 用户**：成员们对**额度消耗增加**表示担忧，尤其是使用 Claude 3.7 时。由于循环错误和过多的 tool calls，部分用户的额度迅速耗尽，引发了对无限方案的需求。
   - 一些用户觉得*被坑了*，因为受额度限制，他们甚至无法使用高级模型进行简单的聊天。
- **回滚救援行动：用户要求版本倒退**：由于最新更新引入了阻碍生产力的严重问题，用户强烈要求提供**降级功能**以恢复到之前的 Windsurf 版本。
   - 用户现在被更新后的版本*困住*了，有人表示他们*真希望当时没点“重启以更新”*。
- **Claude 3.7 代码转换灾难**：用户报告 **Claude 3.7** 在 Wave 4 更新后表现变差且消耗更多额度。有人称其生成冗长代码，还有人反映它无法读取文件或保留编辑内容。
   - 一位用户表示：*“更新后，我的 Agent 甚至连最简单的 Prompt 都难以完成。”*
- **Windsurf 全局规则讨论**：用户讨论了**全局规则**和 `.windsurfrules` 的用法（一种在项目中指定规则的方式），并明确了全局规则可以在用户的 Codeium/Windsurf 文件夹中找到。
   - 一位用户分享说，尽管他们有一个详尽的全局规则文件，但 Windsurf 的表现仍然出乎意料。


<div class="linksMentioned">

<strong>提及链接</strong>:

<ul>
<li>

<ul><li><a href="https://x.com/ericciarla/status/1897332080708858147">来自 Eric Ciarla (hiring) (@ericciarla) 的推文</a>：使用 /llmstxt 在几秒钟内为任何网站生成 llms.txt 文件。我们新的 @firecrawl_dev 端点可以将任何网站转换为单个文本文件，并输入到任何 LLM 中。查看它作为 @rayc... 的集成。</li><li><a href="https://brave.com/search/api/">Brave Search API | Brave</a>：使用自 Bing 以来增长最快的独立搜索引擎为您的搜索和 AI 应用提供动力。只需一次调用即可访问数十亿页面的索引。</li><li><a href="https://docs.codeium.com/windsurf/previews">Previews (Beta) - Codeium Docs</a>：未找到描述</li><li><a href="https://neon.tech">Neon Serverless Postgres — 更快交付</a>：您喜爱的数据库，运行在旨在帮助您更快构建可靠且可扩展应用程序的 Serverless 平台上。</li><li><a href="https://docs.codeium.com/windsurf/usage#purchasing-additional-flex-credits">付费计划与额度使用 - Codeium Docs</a>：未找到描述</li><li><a href="https://docs.codeium.com/windsurf/usage">付费计划与额度使用 - Codeium Docs</a>：未找到描述</li><li><a href="https://techcrunch.com/2025/03/05/openai-reportedly-plans-to-charge-up-to-20000-a-month-for-specialized-ai-agents/">据报道 OpenAI 计划为专业 AI Agent 每月收取高达 20,000 美元的费用 | TechCrunch</a>：据 The Information 报道，OpenAI 可能计划为专业 AI Agent 每月收取高达 20,000 美元的费用。</li><li><a href="https://www.youtube.com/@codeiumdev/videos">Codeium - Windsurf</a>：🧑‍💻 | 您的现代编程超能力🚀 | 300 万+ Codeium 扩展下载量🏄‍♂️ | 打造 Windsurf 编辑器</li><li><a href="https://pierre.co/">Pierre</a>：愉悦的代码审查</li><li><a href="https://codeium.com/windsurf/directory">Windsurf 规则目录</a>：未来的编辑器，就在今天。Windsurf 编辑器是首款由 AI Agent 驱动的 IDE，让开发者保持心流状态。现已支持 Mac, Windows 和 Linux。</li><li><a href="https://codeium.com/plan">计划设置</a>：未来的编辑器，就在今天。Windsurf 编辑器是首款由 AI Agent 驱动的 IDE，让开发者保持心流状态。现已支持 Mac, Windows 和 Linux。</li><li><a href="https://tenor.com/view/rage-angry-communication-gif-17637019732916283735">愤怒 GIF - 愤怒沟通 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.codeium.com/support">支持 | Windsurf 编辑器和 Codeium 扩展</a>：需要帮助？联系我们的支持团队以获取个性化协助。</li><li><a href="https://codeium.canny.io/feature-requests/p/improve-previews-feature-with-a-proper-webview-in-the-sidebar-like-trae">通过在侧边栏添加适当的 "Webview" 来改进 "Previews" 功能（类似 Trae）| 功能请求 | Codeium</a>：非常喜欢新的 "Previews" 功能！🎉✨ 然而，我希望侧边栏能有一个简单的 "Preview" 工具，就像 Trae 的 "Webview" 工具一样。</li><li><a href="https://codeium.com/pricing">价格 | Windsurf 编辑器和 Codeium 扩展</a>：Codeium 对个人用户永久免费。团队可以通过我们的企业版方案进行升级，以获得增强的个性化和灵活的部署。</li><li><a href="https://status.anthropic.com/">Anthropic 状态</a>：未找到描述</li><li><a href="https://codeium.canny.io/">Codeium 反馈</a>：向 Codeium 团队提供反馈，以便我们做出更明智的产品决策。由 Canny 提供支持。</li><li><a href="https://codeium.canny.io/feature-requests/p/unlimited-plan">无限计划 | 功能请求 | Codeium</a>：我认为在你们的竞争对手（下载量和使用量都超过你们）提供无限额度的情况下，提供有限额度是不妥的。</li><li><a href="https://codeium.canny.io/feature-requests/p/pro-ultimate-is-not-so-ultimate-if-were-limited-on-3000-flow-credits">如果限制 3000 个 Flow 额度，Pro Ultimate 就不那么 "Ultimate" 了。 | 功能请求 | Codeium</a>：Flow 额度应该是无限的，或者是 Pro Ultimate，否则就改名为“只比普通 Pro 好一点点”。</li><li><a href="https://tenor.com/view/cat-kitten-kitty-pussy-cat-cute-gif-16834206313880236094">小猫 GIF - 小猫咪 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/ooz-ooznmates-oozandmates-ooz-dook-dook-gif-13562370673666741588">Ooz Ooznmates GIF - Ooz Ooznmates Oozandmates - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://youtu.be/X1rD3NhlIcE?si=VEwFEZUb6q5CncWL">LLM 一次性生成全部输出（全球首个扩散 LLM）</a>：注册 GrowthSchool 的 3 小时 AI 培训！前 1000 名报名者免费！https://web.growthschool.io/MWB 加入我的通讯以获取定期...</li><li><a href="https://chat.inceptionlabs.ai/">Mercury Coder</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1346807504738189355)** (424 messages🔥🔥🔥): 

> `Phi-4-mini models support, Overfitting models on benchmarks, DeepSeek R1, Inventing random benchmarks, Flex Attention support` 


- **Phi-4-mini 模型即将获得支持**：一名成员询问是否有计划支持 **phi-4-mini 模型**，另一名成员确认了这一点。
- **通过过拟合刷新基准测试成绩**：一名成员询问是否可能**在基准测试（benchmarks）上过拟合模型**，以便用较小的模型实现 SOTA 结果，另一名成员表示这以前已经有人做过。
   - 一名成员提到了论文 **phi-CTNL**，指出它通过投入大量精力，仅基于评估基准来策划一种新颖、高质量、非合成的数据混合物，从而强化了此类方法。
- **Windows 现在支持 Unsloth**：根据 [这条 X 帖子](https://x.com/UnslothAI/status/1897334290935132602)，Unsloth 现在可以在 Windows 上运行，允许在没有 Linux 或 WSL 的情况下对 LLM 进行本地微调。
   - 提供了教程：[Unsloth Windows 安装](https://docs.unsloth.ai/get-started/installing-+-updating/windows-installation)。
- **QwQ-32B 推理模型发布并修复 Bug**：新的推理模型 **QwQ-32B** 已经发布，Unsloth 团队提供了 Bug 修复和动态量化（dynamic quants），这比标准的 4-bit 精度大大提高了准确性，可以在 [这里](https://huggingface.co/unsloth/QwQ-32B-GGUF) 获取。
   - 该仓库包含 QwQ 32B 模型，并具有带有 RoPE、SwiGLU、RMSNorm 和 Attention QKV bias 的 Transformers 特性。
- **理解推理模型及其应用**：成员们讨论了**推理模型**的定义和用法，指出它们是经过训练、在回答前输出 Token 进行“思考”的 LLM，通常使用与 SFT 模型类似的 Prompt。
   - 研究还表明，你可以将推理 LLM 的推理过程交给非推理 LLM 来完成并提供答案，令人惊讶的是，它们的表现相当不错。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://fxtwitter.com/gandhikanishk/status/1896988028893323675">来自 Kanishk Gandhi (@gandhikanishk) 的推文</a>：新论文！！我们试图理解为什么有些 LM 能自我提升推理能力，而另一些则遇到瓶颈。关键在于？认知行为！阅读我们的论文，了解正确的认知行为如何产生决定性的影响……</li><li><a href="https://x.com/UnslothAI/status/1897334290935132602">来自 Unsloth AI (@UnslothAI) 的推文</a>：Unsloth 现在支持 Windows 了！🦥 在 Windows 上本地微调 LLM，无需 Linux 或 WSL。教程：https://docs.unsloth.ai/get-started/installing-+-updating/windows-installation</li><li><a href="https://arxiv.org/abs/2309.08632">Pretraining on the Test Set Is All You Need</a>：受到近期展示了在精心策划的数据上预训练的小型基于 Transformer 的语言模型潜力的工作启发，我们通过投入大量精力策划一个新的……来增强这些方法。</li><li><a href="https://docs.unsloth.ai/get-started/fine-tuning-guide">微调指南 | Unsloth 文档</a>：学习微调的所有基础知识。</li><li><a href="https://pastebin.com/MWGHg1UR">QWQ-32B 解密凯撒密码 - Pastebin.com</a>：Pastebin.com 自 2002 年以来一直是排名第一的文本粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://huggingface.co/mradermacher/QwQ-32B-i1-GGUF">mradermacher/QwQ-32B-i1-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/QwQ-32B-GGUF">unsloth/QwQ-32B-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://unsloth.ai/blog">博客</a>：未找到描述</li><li><a href="https://mistral.ai/news/mistral-ocr">Mistral OCR | Mistral AI</a>：介绍全球最顶尖的文档理解 API。</li><li><a href="https://huggingface.co/unsloth/QwQ-32B-GGUF/discussions/2">unsloth/QwQ-32B-GGUF · QwQ-32B-Q5_K_M 循环思考</a>：未找到描述</li><li><a href="https://huggingface.co/Qwen/QwQ-32B">Qwen/QwQ-32B · Hugging Face</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1JqKqA1XWeLHvnYAc0wzrR4JBCnq43HyH?usp=sharing#scrollTo=cKcvFLCsQLtL">Google Colab</a>：未找到描述</li><li><a href="https://spraakbanken.gu.se/resurser/swefaq">SweFAQ 2.0 | Språkbanken Text</a>：未找到描述</li><li><a href="https://huggingface.co/AI-Sweden-Models/Llama-3-8B">AI-Sweden-Models/Llama-3-8B · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/flex_attention.py">unsloth/unsloth/kernels/flex_attention.py at main · unslothai/unsloth</a>：以 2 倍的速度和减少 70% 的显存微调 Llama 3.3、DeepSeek-R1 和推理型 LLM！🦥 - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/issues/1662#issuecomment-2649554021">GRPO 是否也支持视觉模型？· Issue #1662 · unslothai/unsloth</a>：'Qwen2VLForConditionalGeneration' 对象没有属性 'vllm_engine'。取消注释了一些特定于 vllm 的内容：from unsloth import is_bfloat16_supported import torch from unsloth import FastVi...</li><li><a href="https://github.com/codestoryai/sidecar">GitHub - codestoryai/sidecar: Sidecar 是 Aide 编辑器的 AI 大脑，在你的机器上本地运行并与其协作</a>：Sidecar 是 Aide 编辑器的 AI 大脑，在你的机器上本地运行并与其协作 - codestoryai/sidecar</li><li><a href="https://github.com/unslothai/unsloth/">GitHub - unslothai/unsloth: 以 2 倍的速度和减少 70% 的显存微调 Llama 3.3、DeepSeek-R1 和推理型 LLM！🦥</a>：以 2 倍的速度和减少 70% 的显存微调 Llama 3.3、DeepSeek-R1 和推理型 LLM！🦥 - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/releases">发布版本 · unslothai/unsloth</a>：以 2 倍的速度和减少 70% 的显存微调 Llama 3.3、DeepSeek-R1 和推理型 LLM！🦥 - unslothai/unsloth</li><li><a href="https://docs.unsloth.ai/">欢迎 | Unsloth 文档</a>：刚开始使用 Unsloth？</li><li><a href="https://github.com/unslothai/unsloth/commit/8a675d86c218318bc499fcb53d0aeb5061f88875">Logits 修复 (#1916) · unslothai/unsloth@8a675d8</a>：* 更新 rl_replacements.py * 更新 llama.py * 更新 llama.py * 更新 llama.py * 更新 llama.py * 更新 llama.py * 更新 rl_replacements.py * 更新 llama.py * 更新 llama.py * 更新...</li><li><a href="https://github.com/BBC-Esq/VectorDB-Plugin/blob/79e42a8ef4430ab0d2e49ec2fc2d695967641221/src/constants.py#L2789>">VectorDB-Plugin/src/constants.py at 79e42a8ef4430ab0d2e49ec2fc2d695967641221 · BBC-Esq/VectorDB-Plugin</a>：让你能够针对文档（包括音频和视频文件）进行提问的插件。- BBC-Esq/VectorDB-Plugin</li><li><a href="https://github.com/BBC-Esq/VectorDB-Plugin/blob/79e42a8ef4430ab0d2e49ec2fc2d695967641221/src/constants.py#L8>">VectorDB-Plugin/src/constants.py at 79e42a8ef4430ab0d2e49ec2fc2d695967641221 · BBC-Esq/VectorDB-Plugin</a>：让你能够针对文档（包括音频和视频文件）进行提问的插件。

视频文件。 - BBC-Esq/VectorDB-Plugin
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1346811732957335606)** (131 messages🔥🔥): 

> `Qwen7b 显存消耗, GRPO 成功案例, TinyZero 复现, Llama 3.1, 超参数调优` 


- **Qwen7b 极其耗费显存**：一位用户发现 **Qwen7b 模型** 非常吃显存，必须降低每个设备的 batch size；他们使用了 **每个设备 8 个 batch size**、**8 次生成** 以及 **4 次梯度累积步数 (grad accumulation steps)**。
   - 相比之下，该用户指出“老而弥坚的 Mistral”对显存的要求要低得多。
- **TinyZero 复现**：用户讨论了复现 **TinyZero** 项目，并发现了一个仅使用 **5 次 rollouts** 的 [复现版本](https://github.com/JerryWu-code/TinyZero/blob/main/scripts/train_tiny_zero_a100_grpo.sh)。
   - 此外，有人注意到该复现版本中的 **KL 散度乘数 (KL divergence multiplier)** 与 **GRPOConfig** 中的默认值相比非常小。
- **GRPO 超参数发现**：一位成员分享了一份关于 LLM 的 RL 超参数调优的 [DeepResearch PDF](https://cdn.discordapp.com/attachments/1179039861576056922/1346937593442603082/Hyperparameter_Optimization_for_On-Policy_RL_in_LLM_Alignment.pdf)，指出在 **GRPOConfig** 中设置 **大惩罚项 (large penalty)** 的重要性。
   - 该成员指出，通常的 HF 流水线假设是全权重训练，这比微弱的 LoRA 改变模型速度更快。
- **Unsloth GRPO 和 RLOO 显存**：有人指出 **Unsloth 的 GRPO** 可能具有更好的显存效率，这得益于卸载到 CPU (offloading to CPU) 和更高效的梯度累积。
   - Daniel 对显存的极致优化实现了算子融合 (fused kernels)，避免了 Logits 在显存中的实例化。
- **LLM 性能分析 (Profiling)**：在训练期间，简单的 **cProfile** 对寻找瓶颈非常有帮助，而 **torch profiler** 则不太好用。
   - 对于推理过程中的自定义模型，torch profiler 正是你需要的。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/Jiayi-Pan/TinyZero/blob/main/scripts/train_tiny_zero.sh">TinyZero/scripts/train_tiny_zero.sh at main · Jiayi-Pan/TinyZero</a>: DeepSeek R1-Zero 的简洁、极简、易用的复现 - Jiayi-Pan/TinyZero</li><li><a href="https://github.com/JerryWu-code/TinyZero/blob/main/scripts/train_tiny_zero_a100_grpo.sh">TinyZero/scripts/train_tiny_zero_a100_grpo.sh at main · JerryWu-code/TinyZero</a>: 在两台 A100 上自行复现的 DeepSeek R1-Zero 微型版本。 - JerryWu-code/TinyZero</li><li><a href="https://github.com/Jiayi-Pan/TinyZero/issues/5#issuecomment-2624161643">1 gpu is not working , 2 gpus out of memory  · Issue #5 · Jiayi-Pan/TinyZero</a>: 如何处理以下错误，1A100 PCIe 80gb。按照说明操作但出现以下错误。2A100 80gb 运行正常但显存不足。我猜代码默认为多 GPU...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1346827043618291824)** (173 messages🔥🔥): 

> `DeepSeek 蒸馏, Unsloth Windows 支持, 多 GPU 支持, Qwen Coder, GRPO 训练问题`

- **在 DeepSeek 模型中蒸馏推理结果**：一位用户询问是否有人复现了 **DeepSeek** 将推理能力蒸馏到更小模型中的结果，并询问是否可以分享所使用的 tokenizer 和 prompt templates 的差异。
   - 另一位用户提供了一个关于使用 **GRPO** 和 **Unsloth** 训练推理模型的[教程](https://docs.unsloth.ai/basics/reasoning-grpo-and-rl/tutorial-train-your-own-reasoning-model-with-grpo)链接，并建议用户 finetuning 并非一键操作，需要不断的尝试和错误。
- **在 Windows 上为 Unsloth 排除 Triton 故障**：用户在处理 Triton 时遇到了 **TypeError: cannot unpack non-iterable NoneType object** 错误，该问题追溯到 Unsloth 安装期间查找 Windows SDK 的问题，用户被引导至 [Windows 安装指南](https://docs.unsloth.ai/get-started/installing-+-updating/windows-installation)。
   - 一位用户报告了使用 [oKatanaaa](https://github.com/oKatanaaa/unsloth-zoo) 的 fork 版本的修复方法，而另一位用户成功降级到版本 **2025.1.5** 以绕过该错误。
- **多 GPU 训练难以捉摸**：用户讨论了多 GPU 训练，其中一位用户请求一个在单机多 GPU 上使用 **LoRA** 进行 finetuning 的示例 notebook。
   - 澄清了 **Unsloth** 目前在社区版中不支持多 GPU 训练，尽管该功能在 Pro 版本中可用。
- **定位 GRPO 训练问题的根本原因**：用户报告了 **GRPO 训练** 的问题，包括评估指标问题、更新 Unsloth 后的编译失败 **RuntimeError: Unsloth: Failed to create dynamic compiled modules!**，以及训练 loss 保持为零，并且找到了[一个修复方法](https://github.com/unslothai/unsloth/issues/1711)。
   - 降级到 **unsloth==2025.3.6** 和 **unsloth_zoo==2025.3.4** 有助于解决编译错误；另一位用户在他们的 [Colab notebook](https://colab.research.google.com/drive/1u6Acmib0wj2XRvcSrTWdMW0caIe-fHhQ?usp=sharing) 中对 unslothGRPOTrainer 进行了打补丁处理。
- **引导自定义训练的缓存文件修改**：一位用户寻求关于在 Unsloth 中修改生成的缓存文件以进行自定义训练步骤的指导，建议检查 Unsloth 中的 **rl.py/rl_replacements.py** 以及 **unsloth_zoo** 中的 **rl_replacements.py**。
   - 专家建议克隆 **Unsloth GitHub repo** ([https://github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)) 并从源码安装以应用和保留修改，另一位用户分享了一个用于修复缓存问题的[临时补丁](https://colab.research.google.com/drive/1u6Acmib0wj2XRvcSrTWdMW0caIe-fHhQ?usp=sharing)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.unsloth.ai/basics/reasoning-grpo-and-rl/tutorial-train-your-own-reasoning-model-with-grpo">教程：使用 GRPO 训练你自己的推理模型 | Unsloth 文档</a>：通过使用 Unsloth 和 GRPO 将 Llama 3.1 (8B) 等模型转换为推理模型的初学者指南。</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-updating/windows-installation">Windows 安装 | Unsloth 文档</a>：了解如何在有或没有 WSL 的情况下在 Windows 上安装 Unsloth。</li><li><a href="https://huggingface.co/prithivMLmods/SmolLM2_135M_Grpo_Gsm8k/blob/main/smollm-grpo/SmolLM%20x%20Grpo%20M1.ipynb">smollm-grpo/SmolLM x Grpo M1.ipynb · prithivMLmods/SmolLM2_135M_Grpo_Gsm8k at main</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/all-our-models">我们所有的模型 | Unsloth 文档</a>：未找到描述</li><li><a href="https://github.com/oKatanaaa/unsloth-zoo">GitHub - oKatanaaa/unsloth-zoo: Unsloth 工具集</a>：Unsloth 的工具集。通过在 GitHub 上创建账号，为 oKatanaaa/unsloth-zoo 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth/issues/1711">评估损失错误 [已修复] · Issue #1711 · unslothai/unsloth</a>：我得到的评估指标偏差很大，我正在使用 trl 的 SFTTrainer 和 unsloth_train 来避免梯度累积 bug。我已将此问题锁定在 2025.2.6 之后的版本。我运行了...</li><li><a href="https://tenor.com/view/it-crowd-hello-it-have-you-tried-turning-it-off-and-on-again-gif-8607749">IT 狂人 Hello It GIF - IT 狂人 Hello IT 你试过重启吗 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://colab.research.google.com/drive/1u6Acmib0wj2XRvcSrTWdMW0caIe-fHhQ?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 以 2 倍的速度和减少 70% 的显存微调 Llama 3.3, DeepSeek-R1 和推理 LLMs！🦥</a>：以 2 倍的速度和减少 70% 的显存微调 Llama 3.3, DeepSeek-R1 和推理 LLMs！🦥 - unslothai/unsloth</li><li><a href="https://pypi.anaconda.org/rapidsai-wheels-nightly/simple">Simple Index</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1346988306319278091)** (40 messages🔥): 

> `Qwen-32B 发布，中型模型的 RL 扩展，认知行为与 LM 自我提升，无损压缩与智能，用于 RAG 的 AI21 Jamba` 


- ****Qwen-32B** 模型发布！**: **Alibaba** 发布了 **QwQ-32B**，这是一个全新的 **320 亿参数** 推理模型，可与 **DeepSeek-R1** 等模型媲美。该模型基于其 **Qwen2.5-32B** 模型展示了令人印象深刻的 RL 扩展结果，详情见其 [博客文章](https://qwenlm.github.io/blog/qwq-32b)。
   - 此次发布包括 [Hugging Face 模型](https://huggingface.co/Qwen/QwQ-32B)、[ModelScope](https://modelscope.cn/models/Qwen/QwQ-32B)、[demo](https://huggingface.co/spaces/Qwen/QwQ-32B-Demo) 以及 [Qwen Chat](https://chat.qwen.ai)。研究结果表明，RL 训练能持续提升性能，特别是在数学和编程领域。
- **解码 LM 自我提升之谜**: 一篇新论文探讨了为什么某些 LLM 能够自我提升推理能力而其他模型则不能，指出“认知行为”是关键因素。
   - 论文研究了*正确的认知行为*如何显著影响模型通过 RL 进行提升的能力，详见 [此 X 推文串](https://fxtwitter.com/gandhikanishk/status/1896988028893323675)。
- ****AI21 Labs** 推出 **Jamba**：用于 RAG 的混合架构**: **AI21 Labs** 发布了 **AI21-Jamba-Large-1.6** ([Hugging Face](https://huggingface.co/ai21labs/AI21-Jamba-Large-1.6))，这是一款最先进的混合 **SSM-Transformer** 模型。官方声称它是最强大且高效的长文本模型，推理速度比同类模型快达 **2.5 倍**。
   - 尽管宣传势头强劲，但对于仅为了 RAG 而运行一个 **400B** 参数的模型，业内仍存在怀疑态度。一些人质疑 Mamba 模型在长文本准确性方面的可行性，并等待进一步的评测结论。
- **仅靠压缩就能激发智能吗？**: Isaac Liao 和 Albert Gu 在 [这篇博客文章](https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html) 中探讨了无损信息压缩是否能产生智能行为，并对这一观点进行了实际演示。
   - 文章质疑了高效压缩与智能涌现之间的根本关系。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/gandhikanishk/status/1896988028893323675">Kanishk Gandhi (@gandhikanishk) 的推文</a>: 新论文！！我们试图理解为什么有些 LM 能自我提升推理能力，而有些则遇到瓶颈。关键在于？认知行为！阅读我们的论文，了解正确的认知行为如何带来差异...</li><li><a href="https://x.com/alibaba_qwen/status/1897361654763151544?s=46">Qwen (@Alibaba_Qwen) 的推文</a>: 今天，我们发布了 QwQ-32B，这是我们全新的推理模型，仅有 320 亿参数，可媲美顶尖推理模型，如 DeepSeek-R1。博客: https://qwenlm.github.io/blog/qwq-32b HF: https://hu...</li><li><a href="https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html">无预训练的 ARC-AGI</a>: 未找到描述</li><li><a href="https://huggingface.co/ai21labs/AI21-Jamba-Large-1.6">ai21labs/AI21-Jamba-Large-1.6 · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1347291093913309184)** (1 messages): 

> `Aider 在 Product Hunt 上线` 


- **Aider 在 Product Hunt 上线！**: 创始人宣布 [Aider 已在 Product Hunt 发布](https://www.producthunt.com/posts/aider) 并请求社区投票支持。
   - **Aider** 被描述为一个 *AI 配对程序员*，它通过终端编辑本地 git 仓库中的代码，可与你的编辑器、任何 LLM（Claude 3.5 Sonnet, DeepSeek R1, GPT-4o, 本地模型）以及多种编程语言协同工作。
- **在 Product Hunt 上为 Aider 投票**: 该公告鼓励社区成员通过在 [Product Hunt 帖子](https://www.producthunt.com/posts/aider) 上投票来支持 Aider 的发布。
   - 帖子强调 Aider 是一款开源开发工具，利用 AI 增强多种语言的编程体验。



**提到的链接**: <a href="https://www.producthunt.com/posts/aider"> Aider - 终端里的 AI 配对编程 | Product Hunt</a>: Aider 是一个 AI 配对程序员，通过终端编辑本地 git 仓库代码。支持你的编辑器、任何 LLM（Claude 3.5 Sonnet, DeepSeek R1, GPT-4o, 本地模型）以及多种语言。

  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1346800557029064759)** (377 messages🔥🔥): 

> `Grok 3, QwQ-32B, Mac Studio, OpenRouter 吞吐量, Aider 在 Product Hunt 上线`

- **Grok3 是新冠军**：用户报告了对 **Grok3** 的[良好体验](https://link.to.grok3)，称其拥有无限的上下文大小，且性能优于 **O1 Pro**。
   - 一位用户指出 *Grok 没有限制，并且拥有大约每 2 小时 35 条消息的无限上下文（100 万上下文）*，称其为新冠军。
- **QwQ-32B 表现如何？**：社区讨论了新的 [QwQ-32B 模型](https://huggingface.co/Qwen/QwQ-32B)，一些人发现它 *在 RAG 方面表现良好，但由于知识库较窄，独立表现较差*，而另一些人则好奇它与 **DeepSeek-R1** 相比如何。
   - 一位用户表示 *那个工具调用（tool use）基准测试的表现看起来在 Agent 工作流中会很出色*。
- **昂贵的新款 Mac 冲击本地 AI**：成员们讨论了拥有 **512GB** 内存和 **810gb/s** 带宽的新款昂贵 **Mac Studio** 将如何影响本地 AI 开发，可能以合理的运行速度运行更大的模型。
   - 一位成员表示 *如果你想用 NVIDIA 硬件获得 512GB 的内存，你至少要支付 50,000 美元以上*。
- **Aider 在 Product Hunt 受到关注**：**Aider** 被随机发布到 [Product Hunt](https://www.producthunt.com/posts/aider) 并获得了关注，用户们对这种突如其来的认可表示赞赏。
   - 一位成员说 *这很有趣，许多创始人为 Product Hunt 的发布准备了数周，结果排名垫底，而这里有人随机添加了 Aider，它就这样排到了第 10 名。*
- **OpenRouter 的吞吐量统计实现实时更新**：如[这条推文](https://x.com/OpenRouterAI/status/1891510121139769542)所分享，OpenRouter 的吞吐量和延迟图表现在实时更新，显示了近期的提速。
   - 用户还指出 **Parasail** 和 **SambaNova** 是顶级的 R1 供应商，其中 SambaNova 价格更高。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/testingcatalog/status/1897366902701502868">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：Qwen 发布了新的推理模型 QwQ-32B，如果你选择 Qwen2.5-Plus with Thinking (QwQ)，它现在正为 Qwen Chat 提供支持。引用 Qwen (@Alibaba_Qwen)：今天，我们发布了 QwQ-32B，我们新的推理模型...</li><li><a href="https://x.com/mrousavy/status/1897222044808569137">来自 Marc (@mrousavy) 的推文</a>：字节跳动刚刚推出了 Lynx —— React Native 的竞争对手！</li><li><a href="https://x.com/Alibaba_Qwen/status/1897361654763151544">来自 Qwen (@Alibaba_Qwen) 的推文</a>：今天，我们发布了 QwQ-32B，这是我们新的推理模型，仅有 320 亿参数，可与 DeepSeek-R1 等顶尖推理模型相媲美。博客：https://qwenlm.github.io/blog/qwq-32b HF：https://hu...</li><li><a href="https://aider.chat/docs/faq.html#can-i-change-the-system-prompts-that-a">FAQ</a>：关于 aider 的常见问题。</li><li><a href="https://aider.chat/docs/faq.html#can-i-change-the-system-prompts-that-aider-uses">FAQ</a>：关于 aider 的常见问题。</li><li><a href="https://aider.chat/docs/usage/notifications.html">通知</a>：当 Aider 等待你的输入时，它可以通知你。</li><li><a href="https://www.producthunt.com/posts/aider"> Aider - 你终端里的 AI 配对编程工具 | Product Hunt</a>：Aider 是一款 AI 配对编程工具，可以通过终端编辑本地 git 仓库中的代码。支持你的编辑器、任何 LLM（Claude 3.5 Sonnet, DeepSeek R1, GPT-4o, 本地模型）以及多种语言。</li><li><a href="https://tenor.com/view/mujikcboro-seriymujik-gif-24361533">Mujikcboro Seriymujik GIF - Mujikcboro Seriymujik - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/Aider-AI/aider/blob/main/benchmark/README.md">aider/benchmark/README.md at main · Aider-AI/aider</a>：aider 是你终端里的 AI 配对编程工具。通过在 GitHub 上创建账号来为 Aider-AI/aider 的开发做出贡献。</li><li><a href="https://x.com/OpenRouterAI/status/1891510121139769542">来自 OpenRouter (@OpenRouterAI) 的推文</a>：提示：OpenRouter 上的吞吐量和延迟图表会实时更新。这里是 Sonnet 的。感谢 @GoogleAI Vertex 最近的提速！</li><li><a href="https://www.apple.com/newsroom/2025/03/apple-unveils-new-mac-studio-the-most-powerful-mac-ever/">Apple 推出新款 Mac Studio，史上最强大的 Mac</a>：Apple 今天发布了新款 Mac Studio，这是有史以来最强大的 Mac，搭载 M4 Max 和全新的 M3 Ultra 芯片。</li><li><a href="https://www.apple.com/macbook-air/">MacBook Air 13 英寸和 MacBook Air 15 英寸</a>：搭载超快 M4 芯片的 MacBook Air 笔记本电脑。专为 Apple Intelligence 打造。轻巧且具备全天候电池续航。现推出全新的天蓝色。</li><li><a href="https://x.com/i/grok/share/632KWxxCC4NuqrPis82w7gpRm">来自 GitHub - FixTweet/FxTwitter 的推文：修复损坏的 Twitter/X 嵌入！</a>：在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能 - FixTweet/FxTwitter</li><li><a href="https://tenor.com/view/mr-bean-mrbean-bean-mr-bean-holiday-mr-bean-holiday-movie-gif-3228235746377647455">Mr Bean Mrbean GIF - Mr bean Mrbean Bean - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/olilanz/RooCode-Local-Evaluation">GitHub - olilanz/RooCode-Local-Evaluation: Roo Code 和本地托管 LLMs 的评估</a>：Roo Code 和本地托管 LLMs 的评估。通过在 GitHub 上创建账号来为 olilanz/RooCode-Local-Evaluation 的开发做出贡献。</li><li><a href="https://www.producthunt.com/products/aider"> Aider - 产品信息、最新更新和 2025 年评论 | Product Hunt</a>：Aider 是一款 AI 配对编程工具，可以通过终端编辑本地 git 仓库中的代码。支持你的编辑器、任何 LLM（Claude 3.5 Sonnet, DeepSeek R1, GPT-4o, 本地模型）以及多种语言。</li><li><a href="https://github.com/Aider-AI/aider/blob/main/benchmark/docker.sh">aider/benchmark/docker.sh at main · Aider-AI/aider</a>：aider 是你终端里的 AI 配对编程工具。通过在 GitHub 上创建账号来为 Aider-AI/aider 的开发做出贡献。</li><li><a href="https://tenor.com/view/xi-jinping-gif-24241864">Xi Jinping GIF - Xi Jinping - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1346861361837244618)** (146 条消息🔥🔥): 

> `Aider 连接到 OpenWebUI，Litellm 补丁，DeepSeek Token 输出，Aider 提交信息，行尾空格` 


- **通过 OpenWebUI 解锁 Aider**：一位成员通过在模型名称前添加 `openai/` 前缀，解决了将 **Aider** 连接到 **OpenWebUI (OWUI)** 的问题，确保 **Litellm** 能够识别 **OAI-compatible endpoint**。
   - *你必须添加 openai/ 前缀，这样 litellm 才知道你正在使用 OAI-compat endpoint。所以在我的例子中，它是 openai/myowui-openrouter.openai/gpt-4o-mini*。
- **为 Litellm 修复 OpenRouter 的 'Reasoning' 字段**：一位成员分享了一个 **Litellm** 补丁，用于正确显示来自 **OpenRouter** 的 *reasoning* 字段，并指出已合并的 **PR #8431** 并未完全解决该问题。
   - [提供的 diff](https://github.com/litellm/litellm/pull/8431) 处理了 *provider_specific_fields* 以实现正确输出，但仍需要一个额外的 Aider 补丁才能完整显示 reasoning 内容。
- **Token 输出问题已解决**：一位成员报告解决了 **Aider** 无法向聊天输出任何 Token 的问题，结果发现是由于 OpenRouter 余额不足导致的。
   - 充值后，该成员能够在他们的 [OpenRouter activity screen](https://openrouter.ai/activity) 上确认活动和 Token 响应。
- **自定义 Aider 提交信息**：一位成员询问如何使用 Aider 仅为已暂存（staged）文件创建提交信息，建议使用 `git stash save --keep-index`、`/commit` 和 `git stash pop`。
   - 另一位成员提到使用 `aider --commit` 来**自动编写提交信息、提交更改并退出**，并提供了 [Aider commit documentation](https://aider.chat/docs/git.html#commit-messages) 的链接。
- **移除行尾空格**：成员们讨论了 Aider 缺乏自动移除行尾空格功能的问题，并辩论是依赖 Linter 还是自定义脚本来强制执行代码风格。
   - 一位成员在 `.aider.conf.yml` 中分享了一个 **Ruff formatting** 配置，使用 `lint-cmd: - python: ruff format` 来解决空格及更广泛的格式化问题。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>：LLM 代码编辑能力的定量基准。</li><li><a href="https://aider.chat/docs/llms/openai-compat.html">OpenAI compatible APIs</a>：aider 是你终端里的 AI 结对编程工具</li><li><a href="https://aider.chat/docs/git.html#commit-messages">Git integration</a>：Aider 与 Git 紧密集成。</li><li><a href="https://aider.chat/docs/llms/openrouter.html">OpenRouter</a>：aider 是你终端里的 AI 结对编程工具</li><li><a href="https://aider.chat/docs/llms/gemini.html">Gemini</a>：aider 是你终端里的 AI 结对编程工具</li><li><a href="https://aider.chat/docs/usage/lint-test.html#linting">Linting and testing</a>：自动修复 Linting 和测试错误。</li><li><a href="https://www.inceptionlabs.ai/">Inception Labs</a>：我们正在利用扩散技术开发新一代 LLM。我们的 dLLM 比传统的自回归 LLM 更快、更高效。而且扩散模型更准确...</li><li><a href="https://github.com/buger/probe">GitHub - buger/probe：Probe 是一款对 AI 友好、完全本地的语义代码搜索引擎，适用于大型代码库。它是下一代 AI 编程工具最后缺失的基石。</a>：Probe 是一款对 AI 友好、完全本地的语义代码搜索引擎，适用于大型代码库。它是下一代 AI 编程工具最后缺失的基石。 - buger/probe</li><li><a href="https://github.com/lutzleonhardt/mcpm-aider">GitHub - lutzleonhardt/mcpm-aider：一个用于在 Claude App 中管理 MCP 服务器以及供 aider 使用的命令行工具。还可以运行 MCP Server 来帮助你管理所有的 MCP 服务器</a>：一个用于在 Claude App 中管理 MCP 服务器以及供 aider 使用的命令行工具。还可以运行 MCP Server 来帮助你管理所有的 MCP 服务器 - lutzleonhardt/mcpm-aider</li><li><a href="https://github.com/yamadashy/repomix">GitHub - yamadashy/repomix：📦 Repomix（原名 Repopack）是一款强大的工具，可将整个仓库打包成单个对 AI 友好的文件。非常适合需要将代码库提供给 LLM 或其他 AI 工具（如 Claude、ChatGPT、DeepSeek、Perplexity、Gemini、Gemma、Llama、Grok 等）时使用。</a>：📦 Repomix（原名 Repopack）是一款强大的工具，可将整个仓库打包成单个对 AI 友好的文件。非常适合需要将代码库提供给 LLM 或其他 AI 工具...
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1346803804561211392)** (135 条消息🔥🔥): 

> `VRAM overflow, Phi-4 support, KV cache, New Mac Studio, Sesame TTS` 


- **识别 VRAM 溢出！**：用户讨论了如何识别 **VRAM overflow**，并指出如果 **Dedicated memory** 很高（7.5GB+）且 **Shared memory** 开始增加，则表明系统正溢出到 RAM 中，如[这张图片](https://cdn.discordapp.com/attachments/1110598183144399061/1346803804322009088/VRAM_Overflow.jpg?ex=67cad5c9&is=67c98449&hm=dfd029e741e03b3c482e48ebb01ff246776a4f457db7e3e47b8eadda2c43bb2f&)所示。
   - Context size 和 **KV cache** 会影响 **VRAM**，建议将目标设定为 **90% VRAM** 使用率。
- **Phi-4 音频模态仍不支持**：成员们确认 **LM Studio** 不支持 **Phi-4 的音频模态**，因为 *llama.cpp* 尚未提供支持。
   - 一位用户补充说，**多模态 Phi-4** 也不受支持。
- **Sesame AI 首次推出对话式语音生成！**：一位成员分享了 **Sesame AI** 的链接，重点介绍了其令人印象深刻的[对话式语音生成演示](https://www.sesame.com)，听起来*像真人一样*。
   - 尽管据称是*开源*的，但一位成员指出[他们的 GitHub 仓库](https://github.com/SesameAILabs)目前还没有任何 commit。
- **QwQ 量化异常已解决！**：用户讨论了在 **LM Studio** 中运行 **QwQ** 模型的问题，并分享了一个涉及 prompt 参数的潜在[修复方案](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/479#issuecomment-2701947624)。
   - 一位用户指出，在应用修复并重新下载 **lmstudio-community** 版本后，模型开始正常进行推理，不再输出乱码。
- **LM Studio 的 Android 客户端问世！**：一位用户宣布创建了一个 [LM Studio 的 Android 客户端应用程序](https://github.com/brazer/LmStudioAndroid)。
   - 它允许你从 Android 设备连接到 **LM Studio server**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.sesame.com">Sesame</a>：我们相信计算机将变得栩栩如生的未来。它们可以像人类彼此交流一样，观察、聆听并与我们协作。带着这个愿景，我们正在设计一种新型计算机。</li><li><a href="https://github.com/SesameAILabs">SesameAILabs</a>：SesameAILabs 有 8 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://tenor.com/view/puppy-gif-18530240">Puppy GIF - Puppy - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/479#issuecomment-2701947624">lmstudio 中 qwq-32b 模型的问题 · Issue #479 · lmstudio-ai/lmstudio-bug-tracker</a>：哪个版本的 LM Studio？例如：LM Studio 0.3.11。哪个操作系统？Mac。Bug 是什么？在与 qwq-32b 模型聊天时出现以下错误 "Error rendering prompt with jinja templa..."</li><li><a href="https://tenor.com/view/ibm-card-reader-card-reader-ibm-utility-bill-vintage-computer-gif-15507881284984357200">Ibm Card Reader Utility Bill GIF - IBM CARD READER CARD READER IBM - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/brazer/LmStudioAndroid">GitHub - brazer/LmStudioAndroid: LM Studio 的 Android 应用程序。</a>：LM Studio 的 Android 应用程序。通过在 GitHub 上创建账号来为 brazer/LmStudioAndroid 的开发做出贡献。</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/479#issu">lmstudio 中 qwq-32b 模型的问题 · Issue #479 · lmstudio-ai/lmstudio-bug-tracker</a>：哪个版本的 LM Studio？例如：LM Studio 0.3.11。哪个操作系统？Mac。Bug 是什么？在与 qwq-32b 模型聊天时出现以下错误 "Error rendering prompt with jinja templa..."</li><li><a href="https://photos.app.goo.gl/MDNqL1c7d289oHEs7">Brian Makin 的新视频</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1346833939288953015)** (309 条消息🔥🔥): 

> `Mac Studio M3 Ultra & M4 Max, AMD RX 9070 XT vs Nvidia RTX 5070 Ti, DeepSeek V2.5 236b, SGI 机器, NVIDIA RTX 5090 召回` 


- **Apple 的 M3 Ultra 登陆 Mac Studio**：Apple 发布了新款 [Mac Studio](https://www.apple.com/uk/mac-studio/)，搭载 **M3 Ultra** 和 **M4 Max**，其中 M3 Ultra 的 RAM 最高可达 **512GB**。
   - 出于某种原因，他们没有提及 **M4** 在 **LLM 推理**方面的表现，成员们推测由于带宽差异，其速度可能会慢得多。
- **Radeon RX 9070 XT 对决 GeForce RTX 5070 Ti**：一段比较 **AMD RX 9070 XT** 和 **Nvidia RTX 5070 Ti** 在光栅化和光线追踪表现的 YouTube 视频显示两者互有胜负，而 Nvidia 在**光线追踪**方面保持明显领先。
   - 9070 XT 在 **4K** 分辨率下的表现有时能与 **Nvidia 4080 Super** 持平，以 **750 美元**建议零售价（MSRP）的 **80%** 价格提供了 5070 Ti 约 **95%** 的性能。
- **DeepSeek，巨型模型之王**：成员们讨论了运行 **DeepSeek V2.5 236b** 的情况，指出它利用大量 RAM 来处理庞大的初始参数，且运行速度比 **Llama 3.3 70b** 更快。
   - 一位用户 [@alexocheema](https://x.com/alexocheema/status/1897349404522078261?t=IiqHPZlhS5AcNXrVQ4moJw) 指出，*只需 2 台配备 512GB 内存的 M3 Ultra Mac Studio 并结合 @exolabs，即可在家里运行完整的、未经量化的 DeepSeek R1*。
- **SGI 机器：昔日的巨头**：讨论转向了 20 世纪末的 **Silicon Graphics (SGI)** 机器，它们以**大规模**和**共享全局内存**而闻名。
   - 一位用户回忆起 **1998** 年的一台 SGI 机器达到了 **3300 万个多边形/秒**的处理速度，令当时最快的 PC 显卡（**60 万个多边形/秒**）相形见绌。
- **Nvidia RTX 5090 面临召回传闻**：一份[报告](https://wccftech.com/nvidia-geforce-rtx-5090s-are-now-being-recalled-in-europe-over-a-fire-hazard-warning/)称，由于 **12V-2x6 电源接口**存在潜在**火灾隐患**，NVIDIA GeForce RTX 5090 正在欧洲被召回。
   - 然而，Kitguru [撤回](https://www.kitguru.net/components/graphic-cards/matthew-wilson/dutch-retailer-talks-to-kitguru-and-retracts-rtx-5090-recall-claim/)了关于 RTX 50 系列 GPU 可能进行产品召回的说法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/alexocheema/status/1897349404522078261?t=IiqHPZlhS5AcNXrVQ4moJw">来自 Alex Cheema - e/acc (@alexocheema) 的推文</a>: Apple 的时机再好不过了。配备 512GB 内存的 M3 Ultra Mac Studio 非常适合像 DeepSeek V3/R1 这样的大型稀疏 MoE 模型。只需 2 台配备 512GB 内存的 M3 Ultra Mac Studio 并结合 @exolabs 即可...</li><li><a href="https://www.apple.com/newsroom/2025/03/apple-unveils-new-mac-studio-the-most-powerful-mac-ever/">Apple 发布新款 Mac Studio，史上最强大的 Mac</a>: Apple 今天发布了新款 Mac Studio，这是史上最强大的 Mac，搭载 M4 Max 和全新的 M3 Ultra 芯片。</li><li><a href="https://wccftech.com/nvidia-geforce-rtx-5090s-are-now-being-recalled-in-europe-over-a-fire-hazard-warning/">[更新 - 召回声明已撤回] NVIDIA GeForce RTX 5090 现因“火灾隐患”警告在欧洲被召回；问题可能与 12V-2x6 接口有关</a>: NVIDIA GeForce RTX 5090 现正由于与 12V-2x6 电源接口相关的“火灾隐患”风险在欧洲被召回。</li><li><a href="https://en.m.wikipedia.org/wiki/Raja_Koduri">Raja Koduri - 维基百科</a>: 未找到描述</li><li><a href="https://tenor.com/view/clock-in-team-wagie-dance-gif-6441791818063703348">Clock In Team Wagie Dance GIF - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=yP0axVHdP-U"> - YouTube</a>: 未找到描述</li><li><a href="https://threadreaderapp.com/thread/1884244369907278106.html">@carrigmat 在 Thread Reader App 上的帖子</a>: @carrigmat: 在本地运行 DeepSeek-R1 的完整硬件 + 软件配置。使用原始模型而非蒸馏版，并采用 Q8 量化以保证完整质量。总成本 6,000 美元。包含所有下载和配件链接...</li><li><a href="https://www.apple.com/uk/mac-studio/">Mac Studio</a>: 终极专业桌面电脑。由 M4 Max 和 M3 Ultra 驱动，提供极致性能和广泛的连接性。专为 Apple Intelligence 打造。</li><li><a href="https://www.servethehome.com/bolt-graphics-zeus-the-new-gpu-architecture-with-up-to-2-25tb-of-memor">Bolt Graphics Zeus：拥有高达 2.25TB 显存和 800GbE 的新 GPU 架构</a>: 即将推出的 Bolt Graphics Zeus GPU 架构在 500W TDP 的 GPU 上提供多达 6 个 800GbE 链路和 2.2TB 显存。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1347018948784291902)** (1 条消息): 

> `AI model settings, Claude 3.7 Sonnet` 


- **设置合并以实现快速自定义**：AI 模型设置正被合并到 Web 版输入框旁边的统一位置，旨在使自定义过程更快速、更直观。
   - 旧的设置菜单中将保留一个占位符，以引导用户前往新位置，如[附带的截图](https://cdn.discordapp.com/attachments/1047204950763122820/1347018948956131420/Screenshot_2025-03-05_at_8.30.27_PM.png)所示。
- **Claude 3.7 Sonnet 开放 Pro 访问权限**：作为本次更新的一部分，**使用 Claude 3.7 Sonnet 进行推理 (Reasoning)** 将面向 **Pro** 用户开放。
   - 目标是让 *"Auto"*（自动）设置变得更强大，这样用户就不需要手动挑选模型。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1346799632742879387)** (322 条消息🔥🔥): 

> `Auto model selection, Image source issue, Bulk text modification, Qwen Max model, Attached files in threads` 


- **自动模型选择说明**：用户讨论了 Perplexity AI 应用中“Auto”功能的作用，澄清它是从 AI 模型列表中选择一个模型，而不是其本身是一个独立的模型；此外，自动搜索会选择你在设置中挑选的模型。
   - 一些人假设它可能会默认使用 *基础模型 (basic model)*。
- **Perplexity 的图像源故障**：用户报告了一个令人困扰的问题，即作为来源使用的图像在删除后仍会不断出现在后续消息中。
   - 成员们对此感到沮丧并渴望修复，许多人都遇到了这个 Bug。
- **探索文本修改领域**：一位用户寻求关于修改大量文本的 AI 资源建议，特别是针对具有数千个链接的 HTML 和 JSON 文件的交叉引用和合并。
   - 虽然没有推荐具体的 AI 工具，但该查询凸显了对 AI 驱动的大批量文本处理解决方案的需求。
- **Perplexity 用户期待 Claude 3.7 的思考能力**：一位用户对 Perplexity 上的 Claude 3.7 Sonnet 表示担忧，指出其性能与直接使用 Anthropic 账号相比存在差异，且需要*费尽周折*才能激活。
   - 另一位用户报告称 **Claude 3.7 在一个简单的 JSON 文件中幻觉出了错误**，对该模型所谓的改进表示质疑。
- **扩展程序修复了令人沮丧的 Pro 搜索显示 Bug**：用户对 **Pro 搜索不显示其使用了哪个模型**的 Bug 表示不满，这导致很难知道当前正在使用哪个模型。
   - 发现 **complexity 扩展程序**可以修复此 Bug，这促使一些用户仅为此原因而尝试该扩展，而另一些用户则希望 Perplexity 能将此修复合并到主站中。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/monnef/status/1897669944113840293">mennof (@monnef) 的推文</a>：#Perplexity 前景光明！一个精彩的更新即将到来。🚀🤖🎉#ai #perplexityAI</li><li><a href="https://docs.perplexity.ai/api-reference/chat-completions">未找到标题</a>：未找到描述</li><li><a href="https://www.croxyproxy.com/">未找到标题</a>：未找到描述</li><li><a href="https://www.androidauthority.com/google-search-ai-mode-experiment-3532243/">Google 通过 AI 模式增强搜索以回答复杂问题</a>：备受期待的 Google 搜索 AI 模式终于来了，它可以更有效地回答复杂的、多部分的问题。
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1346879637015232552)** (11 条消息🔥): 

> `AI Health Assistant Debut, Nauru sells citizenship, Anthropic Valuation, Meme coins, Early Universe` 


- **微软推出 AI 健康助手**：微软首次展示了 **AI 健康助手** ([链接](https://www.perplexity.ai/page/microsoft-debuts-ai-health-ass-38RGe6B5SVq1nX5OM09k5w3blessed))。
- **Anthropic 估值飙升至 615 亿美元**：Anthropic 的估值达到了 **615 亿美元** ([链接](https://www.perplexity.ai/page/microsoft-debuts-ai-health-ass-38RGe6B5SVq1nX5OM09k5w3blessed))。
- **SEC 表示 Meme 币不属于证券**：SEC 已宣布 **Meme 币不属于证券** ([链接](https://www.perplexity.ai/page/microsoft-debuts-ai-health-ass-38RGe6B5SVq1nX5OM09k5w3blessed))。
- **瑙鲁出售公民身份以换取资源**：**瑙鲁出售公民身份以换取资源** ([链接](https://www.perplexity.ai/page/nauru-sells-citizenship-for-re-mWT.fYg_Su.C7FVaMGqCfQ))。



**提到的链接**：<a href="https://www.youtube.com/embed/scazXHwpFWQ">YouTube</a>：未找到描述

  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1346837917951332453)** (4 messages): 

> `API Focus, Sonar Pro Model, Search Cost Pricing, Real Time Web Data` 


- **API Focus: 学术 vs 社区**：一位成员询问在使用 **API** 时，是否可以将 focus 设置为特定领域，如 **academic/community**（学术/社区）。
- **Sonar Pro 模型在处理实时网络数据时遇到困难**：一位使用 **Sonar Pro 模型** 的成员在利用 **实时网络数据** 时遇到问题，尽管设置了 *search_recency_filter: 'month'*，返回的仍是失效的旧信息。
   - 返回的链接直接报错（**停靠网站**、**404 页面**）。
- **Sonar Pro 模型引用编号令人困惑**：另一位使用 **Sonar Pro 模型** 的成员表示虽然结果不错，但引用编号很混乱，因为回复中是从 **1** 开始，但来源列表却是从 **0** 开始。
- **搜索成本定价是个谜**：一位成员想知道如何计算搜索成本，因为 **API** 没有告知使用了多少次搜索，导致无法追踪支出。


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1346814420629721128)** (138 messages🔥🔥): 

> `Richard Sutton talk on Dynamic Deep Learning, OpenAI Agent Pricing, Custom Claude Code with Flash, LLMs for deobfuscation, Boston Dynamics vs Unitree` 


- **动态深度学习引人注目**：Richard Sutton 几个月前在 ICARL 研讨会系列中关于 [动态深度学习的演讲](https://www.youtube.com/watch?app=desktop&v=75jr5E4OzEE&t=431s) 正在引起关注。
   - 演讲讨论了深度学习的进展以及潜在的未来方向。
- **OpenAI 的 Agent 定价过高？**：据 [The Information](https://www.theinformation.com/articles/openai-plots-charging-20-000-a-month-for-phd-level-agents) 报道，OpenAI 正在考虑为未来推出的能够自动化编程和博士级研究的 Agent 收取每月 **2,000 至 20,000 美元** 的费用。
   - 据报道，OpenAI 的投资者 SoftBank 已承诺今年在 OpenAI 的 Agent 产品上投入 **30 亿美元**。
- **Qwen 的 QwQ-32B：更快的 Qwen？**：阿里巴巴发布了 **QwQ-32B**，这是一个新的 320 亿参数推理模型，可与 DeepSeek-R1 等模型竞争。其 [博客文章](https://qwenlm.github.io/blog/qwq-32b) 详细介绍了如何利用 RL（强化学习）提升数学和编程性能。
   - 该模型基于 Qwen2.5-Plus，通过 RL 训练取得了令人印象深刻的结果。
- **DeepMind 离职：Carlini 选择透明度**：Nicholas Carlini 宣布从 Google DeepMind 离职并加入 Anthropic，他在 [职业更新](https://nicholas.carlini.com/writing/2025/career-update.html) 中提到，离职原因是与 DeepMind 领导层在支持高影响力安全与隐私研究方面存在分歧。
   - 他在 Anthropic 的工作将专注于对抗性机器学习（adversarial machine learning）。
- **Jamba 1.6 登场**：AI21 Labs 推出了 **Jamba 1.6**，这是一个 398B 参数的 MoE 模型。其 [公告](https://www.ai21.com/jamba/) 称，在针对私有企业部署的关键基准测试中，该模型优于 Cohere、Mistral 和 Llama。
   - 也有人对限制性的 [Jamba Open Model License](https://www.ai21.com/jamba-open-model-license/) 表示担忧，这可能会限制其可用性。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://fxtwitter.com/UnitreeRobotics/status/1896859430517629292">来自 Unitree (@UnitreeRobotics) 的推文</a>：功夫机器人比赛😘720° 旋风踢 - 听听这撞击声！功夫机器人原生演示视频。（无加速）（请勿模仿，请与机器保持安全距离）#Unitree #Kungfu #EmbodiedAI #SpringFestivalGal...</li><li><a href="https://x.com/steph_palazzolo/status/1897309493744267314">来自 Stephanie Palazzolo (@steph_palazzolo) 的推文</a>：与 @coryweinberg 合作的新闻：OpenAI 正在加倍投入其应用业务。高管们已与投资者讨论了未来将推出的三类 Agent，价格从每月 2,000 美元到 20,000 美元不等，用于执行诸如...的任务。</li><li><a href="https://x.com/cherry_cc12/status/1897366964080926902">来自 Chen Cheng (@cherry_cc12) 的推文</a>：谁将成为 QwQ 家族的下一位成员？引用 Qwen (@Alibaba_Qwen)：今天，我们发布了 QwQ-32B，这是我们全新的推理模型，仅有 320 亿参数，可与顶尖推理模型相媲美...</li><li><a href="https://x.com/alibaba_qwen/status/1897361654763151544?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">来自 Qwen (@Alibaba_Qwen) 的推文</a>：今天，我们发布了 QwQ-32B，这是我们全新的推理模型，仅有 320 亿参数，可与 DeepSeek-R1 等顶尖推理模型相媲美。博客：https://qwenlm.github.io/blog/qwq-32b HF: https://hu...</li><li><a href="https://x.com/btibor91/status/1897312899124891761?s=46">来自 Tibor Blaho (@btibor91) 的推文</a>：据 The Information 报道，OpenAI 计划为专为高级研究设计的先进 AI Agent 每月收取高达 20,000 美元的费用，目标是让这些 Agent 在长期内贡献约 20%-25% 的收入...</li><li><a href="https://fxtwitter.com/jsuarez5341/status/1897356500131336208">来自 Joseph Suarez (e/🐡) (@jsuarez5341) 的推文</a>：我们通过在线 RL 击败了《宝可梦 红》！未来几天将在此发布详情。由 @dsrubinstein 领导。关注他、我、@DanAdvantage、@kywch500、@computerender 了解更多！引用 drubinstein (@dsrubinstein...</li><li><a href="https://x.com/Alibaba_Qwen/status/1897366093376991515">来自 Qwen (@Alibaba_Qwen) 的推文</a>：Qwen2.5-Plus + Thinking (QwQ) = QwQ-32B。这就是你在 Qwen Chat 上使用这款新模型的方式！引用 Qwen (@Alibaba_Qwen)：今天，我们发布了 QwQ-32B，这是我们全新的推理模型，仅有 320 亿参数...</li><li><a href="https://x.com/Alibaba_Qwen/status/1897361654763151544">来自 Qwen (@Alibaba_Qwen) 的推文</a>：今天，我们发布了 QwQ-32B，这是我们全新的推理模型，仅有 320 亿参数，可与 DeepSeek-R1 等顶尖推理模型相媲美。博客：https://qwenlm.github.io/blog/qwq-32b HF: https://hu...</li><li><a href="https://x.com/realDanFu/status/1897726836421149080">来自 Dan Fu (@realDanFu) 的推文</a>：我们还没结束——很高兴宣布 ThunderGQA ⚡️🐱！快速融合解码，应用于 Llama 和 QWEN 系列模型的 GQA，比 FA3 快 20% 以上！我们将为 ThunderMLA 发布更多更新...</li><li><a href="https://www.ai21.com/jamba-open-model-license/">Jamba 开放模型许可协议</a>：阅读 AI21 Lab 的服务条款。</li><li><a href="https://ghuntley.com/tradecraft/">是的，Claude Code 可以反编译自身。这是源代码。</a>：这些 LLM 在去混淆、转译和结构到结构的转换方面表现惊人。我在去年圣诞节前后发现了这一点，当时我让一个 LLM 为我编写一个 Haskell 音频库...</li><li><a href="https://x.com/arcprize/status/1897689530901446904">来自 ARC Prize (@arcprize) 的推文</a>：QwQ-32B 在 ARC-AGI 上的表现* 公开评估：11.25%，每项任务 0.036 美元* 半私有：7.5%，每项任务 0.039 美元</li><li><a href="https://nicholas.carlini.com/writing/2025/career-update.html">
      职业更新：Google DeepMind -> Anthropic
</a></li>

    </a>：未找到描述</li><li><a href="https://mistral.ai/news/mistral-ocr">Mistral OCR | Mistral AI</a>：推出全球最顶尖的文档理解 API。</li><li><a href="https://x.com/AI21Labs/status/1897657953261601151">AI21 Labs (@AI21Labs) 的推文</a>：今天我们发布了 Jamba 1.6，这是最适合私有企业部署的开源模型。AI21 的 Jamba 在 Arena Hard 等关键基准测试中超越了 Cohere、Mistral 和 Llama，并可与领先的闭源模型相媲美...</li><li><a href="https://huggingface.co/tencent/HunyuanVideo-I2V">tencent/HunyuanVideo-I2V · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/arcprize/status/1897689538002338187">ARC Prize (@arcprize) 的推文</a>：由于移除了 MoE 和更广泛的世界知识，我们假设 QwQ-32B 的推理能力将局限于其进行过 RL（强化学习）的领域（例如数学和编程）。</li><li><a href="https://www.youtube.com/watch?app=desktop&v=75jr5E4OzEE&t=431s">动态深度学习 | Richard Sutton</a>：ICARL 研讨会系列 - 2024 冬季动态深度学习，Richard Sutton 的研讨会——————————————————摘要：尽管取得了巨大成功，当前的深度学习方法...</li><li><a href="https://github.com/HazyResearch/ThunderKittens/tree/mla/kernels/attn/demo/gqa_decode">ThunderKittens/kernels/attn/demo/gqa_decode at mla · HazyResearch/ThunderKittens</a>：用于快速 kernel 的 Tile 原语。通过在 GitHub 上创建账号为 HazyResearch/ThunderKittens 的开发做出贡献。</li><li><a href="https://youtu.be/9_PepvnqIfU?si=sMB90T8__WA19Qrt">图灵奖得主 Richard S. Sutton 与 Cam Linke 对话 | 科学界没有权威</a>：“科学界没有权威，”图灵奖得主 Richard S. Sutton 说道。在这场独家对话中，Amii 首席科学顾问 Richard ...</li><li><a href="https://fxtwitter.com/BostonDynamics/status/1897298172210225280">Boston Dynamics (@BostonDynamics) 的推文</a>：我们正在将 Atlas 设计成全能型机器人，但我们要一步一个脚印地实现这一目标。看看我们为什么从零件排序开始，我们如何解决难题，以及我们如何交付一款人形机器人...</li><li><a href="https://www.cnbc.com/2025/03/05/scale-ai-announces-multimillion-dollar-defense-military-deal.html">Scale AI 宣布数百万美元的国防协议，这是美国军事自动化的重要一步</a>：由国防创新部门（Defense Innovation Unit）牵头，Thunderforge 项目将与 Anduril、Microsoft 等公司合作开发和部署 AI Agent。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1346813647544123477)** (41 条消息🔥): 

> `LLM 玩 Diplomacy 游戏，发布模型 fathoms，生日快乐，后训练即服务 (Post training as a service)，14B img2vid 模型`

- ****LLM 通过 Diplomacy 协商统治世界****：一位成员分享了一个[框架](https://x.com/sam_paech/status/1897078633015206172)，让 **LLM** 之间互相玩 **Diplomacy**（外交）游戏，并指出该框架非常适合进行博弈论实验和测试说服力，同时提供了代码和示例。
   - Diplomacy 是一款复杂的棋盘游戏，具有很强的谈判元素，据报道阅读其谈判日志*非常有趣*。
- ****模型发布引发困惑****：一位成员链接到一条推文并表示 *我无法想象他们竟然真的发布了这个模型。* ([推文](https://x.com/adonis_singh/status/1896679334200611312))，暗示对某个模型的发布感到困惑。
   - 另一位成员回应称 *其他模型可以生成同样好甚至更好的 greentexts，不明白为什么这么痴迷。V3 作为一个现代模型相当不错，否则你也可以使用旧的 base models。*
- ****图生视频模型创建香蕉蛞蝓电影宇宙****：一位成员分享了一个 [Replicate 链接](https://replicate.com/wavespeedai/wan-2.1-i2v-480p)，展示了一个 **14B img2vid 模型** 如何从一张图片生成逼真的**香蕉蛞蝓 (Banana Slug)** 视频。
   - 生成的视频和源图像突出了加州大学圣克鲁兹分校 (UC Santa Cruz) 的吉祥物**香蕉蛞蝓**，该吉祥物以其非传统性而闻名。
- ****研究员称 Scaling Laws 仍在发挥作用****：一位成员链接了研究员 [E Mollick](https://x.com/emollick/status/1897457930833731979?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ) 的推文，声称**第一扩展定律 (First Scaling Law)** 仍然成立，并指出 **GPQA** 分数从 **GPT-3.5 Turbo** (**30%**) 提升到 **GPT-4 Turbo** (**47%**)，再到 **GPT-4.5** (**70%**)。
   - Gary Marcus 对此提出了[警告](https://x.com/garymarcus/status/1897488996965777575?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ)，理由是数据集较小、可能存在数据污染以及未知的增强技术。
- ****通过宝可梦昵称保护口袋妖怪****：一位成员链接到一条推文，引用了这样一种说法：在被指示给 **Pokémon** 起昵称后，**Claude** 变得更加保护它的 **Pokémon** 了 ([推文](https://x.com/zswitten/status/1897698670759551378))。
   - 给宝可梦命名是新的安全对齐 (safety alignment) 方式（笑）。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/garymarcus/status/1897488996965777575?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">Gary Marcus (@GaryMarcus) 的推文</a>：@emollick 你正在看的是一个只有 198 个问题的单一指标，并且存在严重的数据污染可能性，更不用说未知的辅助数据增强技术了。我认为你的结论是...</li><li><a href="https://x.com/sam_paech/status/1897078633015206172">Sam Paech (@sam_paech) 的推文</a>：我制作了一个让 LLM 互相玩 Diplomacy 的框架。Diplomacy 是一款具有重度谈判元素的复杂棋盘游戏。非常适合实验博弈论和测试说服力！它...</li><li><a href="https://x.com/emollick/status/1897457930833731979?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">Ethan Mollick (@emollick) 的推文</a>：看起来第一扩展定律（模型越大越“聪明”）仍然成立——算力的数量级增长带来了能力的线性提升。GPT-3.5 Turbo 在 GPQA 上得分 30%，...</li><li><a href="https://x.com/paulgauthier/status/1897721567884591402">Paul Gauthier (@paulgauthier) 的推文</a>：QWQ 32B 在 aider 的 polyglot 基准测试中得分 21%，使用了 temperature=0.6 和 top_p=0.95。https://aider.chat/docs/leaderboards/</li><li><a href="https://x.com/adonis_singh/status/1896679334200611312">adi (@adonis_singh) 的推文</a>：我无法想象他们竟然真的发布了这个模型 😭</li><li><a href="https://x.com/OpenAIDevs/status/1897700857833193955">OpenAI Developers (@OpenAIDevs) 的推文</a>：ChatGPT for macOS 现在可以直接在 IDE 中编辑代码。适用于 Plus、Pro 和 Team 用户。</li><li><a href="https://mafia.opennumbers.xyz/">LLM Mafia 游戏竞赛</a>：未找到描述</li><li><a href="https://x.com/corbtt/status/1897735437340627405">Kyle Corbitt (@corbtt) 的推文</a>：🕵️ 更小的开源权重模型能否匹配最先进的推理性能？我们使用 GRPO 在“Temporal Clue”上进行了研究，超越了 R1、o1 和 o3-mini，并接近 Sonnet 3.7...</li><li><a href="https://www.ucsc.edu/campus/mascot/)">我们的吉祥物：香蕉蛞蝓 Sammy – 加州大学圣克鲁兹分校</a>：未找到描述</li><li><a href="https://x.com/zswitten/status/1897698670759551378">Zack Witten (@zswitten) 的推文</a>：我最喜欢的 Claude 玩宝可梦的小细节（在 @latentspacepod 中提到）是，当 @DavidSHershey 告诉 Claude 给它的宝可梦起昵称时，它立刻变得更加保护它们，确保...
</li>
</ul>

</div>

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1347075957730574387)** (5 messages): 

> `Chinese Lewd R1 Aha Moment, DeepSeek Videos on bilibili Comments, Reinforcement Learning History by Schmidhuber` 


- **Chinese Lewd R1 "Aha" Moment**: 一位用户分享了一个[帖子](https://fxtwitter.com/teortaxesTex/status/1897508611019932133)，声称 *“中国人已经把 R1 的 ‘Aha moment’ 搞颜色了，全完了。”*
- **"Sapiosexual for Whale" Comments on DeepSeek Videos**: 一位用户指出，在 bilibili 上的 **DeepSeek** 视频中，评论者经常表达自己是 *“对 🐋 的智性恋 (sapiosexual)”*，参考了[这条评论](https://x.com/layer07_yuxi/status/1897512187264119129)。
- **Schmidhuber Shares Reinforcement Learning History**: 一位用户分享了 [SchmidhuberAI 的推文](https://x.com/SchmidhuberAI/status/1897569590357402076) 链接，提供了 *“Annotated History of Modern AI and Deep Learning”* 第 17 节中关于 **reinforcement learning** 的背景：[论文链接](https://people.idsia.ch/~juergen/deep-learning-history.html#rl)。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/teortaxesTex/status/1897508611019932133">Tweet from Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>: The Chinese have lewded the R1 “Aha moment”, it&#39;s over</li><li><a href="https://x.com/SchmidhuberAI/status/1897569590357402076">Tweet from Jürgen Schmidhuber (@SchmidhuberAI)</a>: @RichardSSutton Some background to reinforcement learning in Sec. 17 of the &#34;Annotated History of Modern AI and Deep Learning:&#34; https://people.idsia.ch/~juergen/deep-learning-history.html#rl</li><li><a href="https://x.com/layer07_yuxi/status/1897512187264119129">Tweet from Yuxi on the Wired (@layer07_yuxi)</a>: @teortaxesTex if you go to DeepSeek videos on bilibili, and read the comments you&#39;ll often see people saying they are &#34;sapiosexual for 🐋&#34;
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1346968004331573309)** (6 messages): 

> `Schmidhuber, Deep RL, Richard Sutton Turing Award` 


- **Schmidhuber Congratulates Richard Sutton, Hints at Cult Leadership**: [Jürgen Schmidhuber](https://x.com/SchmidhuberAI/status/1897406236896977388) 祝贺 **Richard Sutton** 和 **Andy Barto** 获得 **Turing Award**，一位用户调侃道 *“邪教领袖惺惺相惜 (Cult leader game recognizes cult leader game)。”*
- **Pieter Abbeel's Deep RL Tutorial Still Relevant**: **Pieter Abbeel** 重新发布了他的 [Basics of Deep RL tutorial](https://www.youtube.com/playlist?list=PLwRJQ4m4UJjNymuBM9RdmB3Z9N5-0IlY0)，声称 *“我仍然对它非常满意”*。
   - 一位用户表示赞同，认为 *“通过这个教程和 Sergey Levine 的课程，你就能学到 RL 的全部内容”*，并提到了 **David Silver 的 UCL 课程** 的相关性。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/SchmidhuberAI/status/1897406236896977388">Tweet from Jürgen Schmidhuber (@SchmidhuberAI)</a>: Congratulations to @RichardSSutton and Andy Barto on their Turing award!</li><li><a href="https://x.com/pabbeel/status/1897437838180061204">Tweet from Pieter Abbeel (@pabbeel)</a>: Basics of Deep RL tutorial I am still very happy with, as good a day as any to re-post :)https://www.youtube.com/playlist?list=PLwRJQ4m4UJjNymuBM9RdmB3Z9N5-0IlY0
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1346938335108661289)** (9 messages🔥): 

> `Reinforcement Learning, Scientific AI, LLMs, Pre-training` 


- **RL 系统击败《宝可梦 红》**：由 HazyResearch 开发的一个 Reinforcement Learning 系统成功通关了游戏 **《宝可梦 红》**。该系统使用了参数量低于 **10M** 的 policy、**PPO** 以及新技术，详见其 [blog post](https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla)。
   - 团队还发布了相关资源，包括 [code](https://github.com/HazyResearch/ThunderKittens/blob/mla/kernels/attn/demo/mla_decode/template_mla_decode.cu) 以及他们之前关于 **TK Part 1** 和 **TK Part 2** 的工作链接。
- **ThunderMLA 加速 LLM 推理**：HazyResearch 推出了 **ThunderMLA**，这是一个用于 decode 的融合 “megakernel”。根据其 [blog post](https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla)，通过实现简单的调度技巧，在多种工作负载下，它比 DeepSeek 的 **FlashMLA** 快 **20-35%**。
   - 初始版本专注于 attention 解码，但他们认为它具有更广泛的应用前景。
- **Dario 的 Loving Grace**：一位成员在活动中分享了一个[有争议的观点](https://thomwolf.io/blog/scientific-ai.html)，对 **AI** 是否有能力像 Dario 的 “Machine of Loving Grace” 中所设想的那样，将 21 世纪的科学发现压缩到短短 5-10 年内表示怀疑。
   - 该成员认为，我们实际得到的将是“一个充斥着唯唯诺诺者的国家”。
- **LLMs 重新发现爱因斯坦的发现？**：一位成员提出了一种方法，通过在 1905 年之前的文档上对模型进行 Pre-training，并进行 Post-training 以使其能够使用大规模推理算力（scale inference compute），来测试 **LLMs** 是否具有足够的创造力来实现突破性发现，详见[此推文](https://x.com/rosstaylor90/status/1897694319382798681)。
   - 随后将提示模型解释**光电效应**并统一**麦克斯韦方程组**，并使用当前模型验证生成内容是否符合**爱因斯坦**的解决方案。
- **Pre-training 最新进展概览**：一位成员分享了向 **Hugging Face** 同事介绍 “Pre-training 新进展”的演讲幻灯片，在此提供了 Pre-training 领域的概览 [here](https://drive.google.com/file/d/1lw0hfxHAshcKupxMW51F5zeV1PeDe34w/view)。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla">ThunderMLA: FlashMLA, Faster and Fused-er!</a>: no description found</li><li><a href="https://thomwolf.io/blog/scientific-ai.html"> telescope The Einstein AI model</a>: no description found</li><li><a href="https://x.com/rosstaylor90/status/1897694319382798681">Tweet from Ross Taylor (@rosstaylor90)</a>: - Pre-train a model on all documents before 1905.- Post-train like R1 so it can use scale inference compute to think widely about a problem.- Prompt for an explanation of the photoelectric effect, rec...</li><li><a href="https://x.com/eliebakouch/status/1897665636710400397">Tweet from elie (@eliebakouch)</a>: Gave a talk earlier today to explain &#39;What&#39;s new in pre-training&#39; to my @huggingface colleagues. I&#39;m sharing the slides here if you&#39;re interested in a humble overview of the pre-tr...</li><li><a href="https://drive.google.com/file/d/1lw0hfxHAshcKupxMW51F5zeV1PeDe34w/view">Pre-Training SOTA.pdf</a>: no description found</li><li><a href="https://x.com/dsrubinstein/status/1897351145485648309?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">Tweet from drubinstein (@dsrubinstein)</a>: Excited to finally share our progress in developing a reinforcement learning system to beat Pokémon Red. Our system successfully completes the game using a policy under 10M parameters, PPO, and a few ...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[lectures-and-projects](https://discord.com/channels/1179127597926469703/1223784028428177510/1346996584667152386)** (10 messages🔥): 

> `RLHF Book, Lecture series videos` 


- **分享 RLHF 书籍链接**：一位成员为有需要的人分享了 **RLHF book** 的链接：[https://rlhfbook.com/book.pdf](https://rlhfbook.com/book.pdf)。
- **系列讲座即将推出**：一位成员提到，他们初步计划在夏季进行**系列讲座**，**每个章节录制 1 个视频**。
   - 他们补充说，一旦预订按钮上线，就必须让“营销引擎全速运转”。

---

### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1346848634926272582)** (9 messages🔥): 

> `Stargate Project, Content Gating, Data Protection` 


- **Stargate Project 靠广告回本？**：据称 **Stargate Project** 正通过*完全公正且不具侵扰性的广告*来获取资金。
   - 这一公告引发了社区的一些质疑。
- **内容门槛化（Gating Content）是好事吗？**：正如 [Ben Thompson](https://stratechery.com/) 所言，随着 **AI models** 变得越来越强大，企业需要*对其内容设置门槛*以避免被淘汰。
   - 报纸行业因未能做到这一点，现在只能*接受 Sam 提供的任何交易*，但像 **YouTube** 和 **GitHub** 这样宝贵的数据宝库必须不惜一切代价予以保护。
- **数据保护是关键**：如果现有的供应商无法与数据提供者达成协议，将为 AI 的规模化（scaling）设定上限。
   - 具体而言，如果 **Microsoft** 封锁了 **OpenAI** 每月 2 万美元的 **coding agent**，由于数据获取变得更加困难，其效用将会降低。


  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1347161977952407603)** (32 messages🔥): 

> `Anthropic's Recommendations, Nationalizing Labs, DeepSeek Exports, China AMD GPUs, H20 Controls` 


- **Anthropic 倡导 AI 行动计划**：Anthropic 向 OSTP 美国 AI 行动计划提交了其[建议](https://www.anthropic.com/news/anthropic-s-recommendations-ostp-u-s-ai-action-plan)，其中包括**将具有高安全许可的 AI 实验室国有化**。
   - 一名成员分享了 [Anthropic 对 OSTP 的回应](https://assets.anthropic.com/m/4e20a4ab6512e217/original/Anthropic-Response-to-OSTP-RFI-March-2025-Final-Submission-v3.pdf)链接，称其*有点令人尴尬，但与其观点一致*。
- **博士级模型使 GPQA 饱和**：一名成员指出，既然我们已经使 GPQA 饱和，且模型显然达到了**博士级（PhD-level）**，那么它们必须具备*在大多数学科（包括生物学、计算机科学、数学和工程学）中匹配或超过诺贝尔奖得主的智力能力*。
   - 他们评论说，考虑到这 [5 个代码库](https://link.to.five.repos)及其利用率，限制 H20 将会非常疯狂。
- **DeepSeek 出口引发辩论**：一名成员认为，语调从“DeepSeek 也就那样”到“它是停止进口的证据”的转变，与 [Dario Amodei 关于出口管制的立场](https://darioamodei.com/on-deepseek-and-export-controls#export-controls)一致。
   - Anthropic 需要在 **2027** 年前交付成果。
- **AMD GPUs 可能成为中国的开源救星**：一名成员推测，如果中国被限制使用 **AMD** 显卡，他们可能会全面开发相关代码并将其开源。
   - 另一名成员开玩笑说，这是*向开源神灵祈祷能有用于深度学习的 AMD GPUs*。
- **潜在的 H20 管制**：一名成员希望不要有 H20 管制，并希望 [DeepSeek 能获得 H100 集群](https://link.to.h100.farm)，因为 **B200s** 已经缺货。
   - Dario 反对禁止 H20。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1346825050518720553)** (67 messages🔥🔥): 

> `Touhou games and AI, RL gyms Starcraft gym and Minetest gym, Unified memory discussion, Thunderbolt 5 for distributed inference/training, Deepseek-R1` 


- **东方 Project 游戏激发 AI 志向**：多位成员对**东方 Project 游戏（Touhou games）**表示热衷，认为它是进入 **AI** 和 **GPU programming** 领域的灵感来源。
   - 一名成员提到想训练一个模型来玩东方游戏，并指出随着 **RL** 的发展以及将分数作为奖励，这正变得越来越容易。
- **统一内存（Unified Memory）引发关注**：鉴于 **M3 Ultra** 的发布，关于统一内存的讨论被激发，大家思考其在 **GPU memory bandwidth** 和 **CPU/GPU memory transfers** 方面的性能特征。
   - 共识是 **Apple M series** 确实寻址相同的内存，并且对使用 **Thunderbolt 5** 在 Mac Mini/Studio 之间进行分布式推理/训练（distributed inference/training）感到兴奋。
- **Tenstorrent Quietbox 即将到来**：一名成员宣布他们的公司正在购入一台 **Tenstorrent quietbox**，并表示一旦有机会使用就会分享经验。
   - 另一名成员分享了 [Tenstorrent GitHub](https://github.com/tenstorrent) 的链接，重点介绍了 **metal** 和 **buda** 等在不同层级使用其硬件的项目。
- **Langchain 被淘汰了吗？**：社区辩论了 **Langchain** 的优缺点及可能的没落，一些人表达了负面情绪并质疑其抽象化方式。
   - 虽然一名成员希望它“彻底凉凉”，但另一名成员承认它在早期让人们思考如何构建组合方面发挥了作用，尽管认为它是一个*糟糕的库*。
- **GPU GTC 优惠码？**：一名成员询问 **GTC 的 GPU MODE 优惠码**。
   - 另一名成员回答是 **GPUMODE**。


  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1346941022755881000)** (9 条消息🔥): 

> `Triton 中的 tl.gather，Triton 中的 PagedAttention，Bias 加法优化，输入尺寸与性能问题，NVIDIA 的 Cooperative Vector` 


- **Triton 的 `tl.gather` 消失了！**：一位用户报告在尝试使用 Triton 中的 `tl.gather` 时遇到了 `AttributeError`，尽管文档中提到了它，并且[该问题已在 GitHub 上报告](https://github.com/triton-lang/triton/issues/5826)。
   - 一位成员建议从 master 分支构建 Triton，并卸载 PyTorch 自带的版本来解决此问题。
- **寻求 PagedAttention 资源**：一位成员请求关于如何在 Triton 中重新实现 **PagedAttention** 的资源。
   - 摘要中未提供具体资源。
- **融合层 Bias 提升：Broadcast 还是 Kernel？**：一位用户询问在 Triton 的融合层之后添加 bias 的更好、更快的方法。
   - 选项包括在 broadcast 后用 bias 初始化输出，或者在 kernel 中添加；然而，目前还没有确定的答案。
- **Triton Tensorcores 在 32 的倍数下表现最佳**：一位用户在输入尺寸不是 **32** 的倍数时遇到了性能问题，并发现手动对输入进行 padding 提高了速度，这表明 **NVIDIA tensorcores** 仅支持特定的工作组大小（work group sizes）。
   - 建议保留 padding 并使用 mask 来读取计算输出，因为硬件通过更好地利用计算资源，处理经过 padding 的值会更快；否则，它将被分发为逐元素计算（elementwise-computations）。
- **NVIDIA 迎来 Cooperative Vector**：**NVIDIA** 宣布支持 **Cooperative Vector**，据称它将支持 Tensor cores 上更小的计算块，并在 [Vulkan 文档](https://github.com/KhronosGroup/Vulkan-Docs/blob/main/proposals/VK_NV_cooperative_vector.adoc)中进行了描述。
   - 一位成员认为，*处理 21 大小分组的 3 个线程可以自动合并为一个 64 大小的 kernel 调用*，并且 Raytracing Optix 扩展已为其 cooperative vector 提供了 float8 支持。



**提及的链接**：<a href="https://github.com/triton-lang/triton/issues/5826">Cannot call tl.gather · Issue #5826 · triton-lang/triton</a>：描述 Bug：当我运行以下代码时，出现异常：AttributeError: module &#39;triton.language&#39; has no attribute &#39;gather&#39; import triton.language as tl tl.gather 我已经...

  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1346800993525960704)** (19 条消息🔥): 

> `CUDA compiler optimization, CUDA OpenGL interop segfault, Overlapping kernel execution with NCCL, Memory transaction size, GTC talk on maximizing memory bandwidth` 


- **CUDA 编译器再次发威，优化掉了未读取的数据**：一位用户发现 **CUDA compiler** 优化掉了内存写入操作，因为数据从未被读取；AI 建议通过写入大范围内存来防止优化的方案也失败了，直到向数组添加了读取操作。
   - 另一位成员确认 *由于没有读取操作，编译器会将其优化掉*，并且向数组添加读取操作会导致编译器报错。
- **OpenGL 互操作段错误调试僵局**：一位用户在笔记本电脑上的 **CUDA OpenGL interop** 代码中遇到了段错误（segfault），具体发生在 `cudaGraphicsMapResources` 调用处，而 `cudaGraphicsGLRegisterImage` 返回了 `cudaErrorUnknown`。
   - 另一位成员建议问题可能在于 **OpenGL 没有使用 GPU**，用户在查看了[这篇论坛帖子](https://devtalk.nvidia.com/default/topic/534608/cuda-setup-and-installation/cudaerrorunknown-on-cudaopengl-interop/)后确认这解决了问题。
- **Stream 协作：隐藏通信开销**：一位成员询问如何通过将 **NCCL** 集合通信操作放在不同的 stream 中，来使其与 kernel 执行时间重叠。
   - 目标是通过在一个 stream 中运行向量加法，在另一个 stream 中运行 `allreduce` 来隐藏通信开销，但讨论中尚不清楚这是否是一个可行的策略。
- **Warp 级数据速度：优化内存事务**：一位用户询问，关于在一个内存事务（128 字节）中访问 warp 数据的建议是否是指 **cacheline size**，还是有其他考虑，并引用了 [CUDA 官方文档中关于设备内存访问](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses) 的内容。
   - 另一位成员指出，还建议**每个线程进行 128-bit 传输**（例如 float4），这将导致一个 warp 产生 512 字节的传输，由 4 个 128 字节的内存事务提供服务，这可能是为了更高效的指令执行。
- **GTC 演讲传授技巧**：一位成员为即将举行的 **GTC talk** 做了宣传，该演讲致力于最大化内存带宽和事务，详情见[此处](https://www.nvidia.com/gtc/session-catalog/?regcode=pa-srch-goog-157409-prsp&ncid=pa-srch-goog-157409-prsp&deeplink=audience-recommend--1&tab.catalogallsessionstab=16566177511100015Kus&search=%22cuda%20techniques%22#/session/1727709012449001X6PZ)。
   - 据该成员称，该演讲的演讲者*比任何人都更了解这个主题*。



**提到的链接**: <a href="https://www.nvidia.com/gtc/session-catalog/?regcode=pa-srch-goog-157409-prsp&ncid=pa-srch-goog-157409-prsp&deeplink=audience-recommend--1&tab.catalogallsessionstab=16566177511100015Kus&search=%22cuda%20techniques%22#/session/1727709012449001X6PZ">NVIDIA #GTC2025 会议议程目录</a>：3 月 17 日至 21 日在圣何塞亲身体验 GTC 2025（线下及线上）。

  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1346900912525934773)** (32 messages🔥): 

> `Torch C++ Interface, Extending OffloadPolicy, use_reentrant in Activation Checkpointing, TorchBind API, Model-Based RL subgames` 


- **Torch C++ 方法缺少类似 Schema 的函数**：一位成员询问为什么 torch C++ 接口库中的方法缺少类似 schema 的函数，以及扩展 `OffloadPolicy` 的提案是否会被接受为 PR。
   - 一名工作人员建议将这些方法转换为函数并使用别名进行注释，但对其用例表示好奇；该成员回复称这是为了创建子博弈（subgames）以解决 **model-based RL** 问题。
- **Activation Checkpointing 中的 `use_reentrant` True/False**：成员们讨论了 PyTorch [checkpointing 文档](https://pytorch.org/docs/stable/checkpoint.html)中 `use_reentrant` 参数的含义，其中一人幽默地表示*他们总是将其设置为 False*。
   - 会议澄清了 `use_reentrant=True` 是 Activation Checkpointing 的旧实现，而 `use_reentrant=False` 是更新、更优的实现，并且 **Transformers 出于某种原因与 Activation Checkpointing 兼容性不佳**（[HuggingFace/Transformers issue #23808](https://github.com/huggingface/transformers/issues/23808)）。
- **Transformers 问题困扰用户**：一位成员描述说，为了解决 Transformers 与 Activation Checkpointing 配合不佳的问题，他花了*一天半的时间苦思冥想*。
   - 另一位成员分享了类似的经历，称其为*一种必经之路（rite of passage）*，并希望增加一条警告信息，他们**可能很快就会向 Transformers 提交相关推送**。
- **用于自定义 C++ 类的 TorchBind API**：一位成员确认使用 **TorchBind API** 将自定义 C++ 类绑定到 Python，以便创建博弈树并与 NN 预测协同求解。
   - 他们澄清说 **TorchScript** 机制是为 **TorchScript Inference** 创建的，但在该用例之外并未得到很好的支持。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/docs/stable/checkpoint.html">torch.utils.checkpoint &mdash; PyTorch 2.6 文档</a>：未找到描述</li><li><a href="https://github.com/huggingface/transformers/issues/23808">为什么我们在训练时默认不设置 use_cache=False？ · Issue #23808 · huggingface/transformers</a>：功能请求。以 GPT-2 为例，在当前的实现中（modeling_gpt2.py: Line 856~861）：如果 self.gradient_checkpointing 且 self.training：如果 use_cache：logger.warning_once(...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1347158288348745728)** (6 messages): 

> `ThunderMLA, DeepSeek MLA, Modular's Democratizing AI Compute, CUDA Alternatives` 


- **ThunderMLA 闪击 DeepSeek 的 FlashMLA**：HazyResearch 推出了 **ThunderMLA**，这是一种用于 decode 的融合“megakernel”，通过调度技巧，在各种工作负载上比 **DeepSeek 的 FlashMLA** 快 **20-35%**。
   - 该发布侧重于 Attention 解码，代码可在 [此处](https://github.com/HazyResearch/ThunderKittens/blob/mla/kernels/attn/demo/mla_decode/template_mla_decode.cu) 获取，相关链接包括 [TK Part 2](https://hazyresearch.stanford.edu/blog/2024-10-29-tk2)、[TK Part 1](https://hazyresearch.stanford.edu/blog/2024-05-12-quick-tk) 和 [Brr](https://hazyresearch.stanford.edu/blog/2024-05-12-tk)。
- **Modular 深度解析 AI 计算民主化**：Modular 的“AI 计算民主化”系列第 5 部分批判性地探讨了为什么之前的 **CUDA 替代方案**（如 **OpenCL**、**SYCL** 和 **OneAPI**）尽管旨在实现 AI 计算民主化，但最终还是失败了。
   - 失败源于“[开放式竞合 (open coopetition)](https://en.wikipedia.org/wiki/Open_coopetition)”的挑战和管理失误，正如该系列从 [第一部分：DeepSeek 对 AI 的影响](https://www.modular.com/blog/democratizing-compute-part-1-deepseeks-impact-) 开始所概述的那样。
- **CUDA Graph 对比 ThunderMLA**：一位成员询问 **ThunderMLA** 与 **CUDA graph** 有何不同。
   - 另一位成员回答道：*由于指令是以 Tensor 形式传递给 Kernel 的，它不依赖于任何 CPU 操作，所以它应该可以直接运行。*


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla">ThunderMLA: FlashMLA, 更快且更融合！</a>：未找到描述</li><li><a href="https://www.modular.com/blog/democratizing-ai-compute-part-5-what-about-cuda-c-alternatives?utm_source=x&utm_campaign=community">Modular: AI 计算民主化，第 5 部分：CUDA C++ 替代方案如何？</a>：未找到描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1347138627288829963)** (2 条消息): 

> `Triton tl.sort() problem, Flask API Authentication` 


- **Triton Sort 困扰新手**：一位用户报告称，在其 **Triton** 程序中 `tl.sort()` 未能按预期对输入张量进行排序，输出结果与输入完全一致。
   - 他们提供了一个在 Triton kernel 中使用 `tl.sort()` 的代码片段，寻求帮助以找出排序失败的原因。
- **Flask API Authentication 令用户沮丧**：一位用户询问关于在使用 **Flask API** 的 Web 应用程序中实现 **authentication**（身份验证）的建议。
   - 他们正在寻求合适的身份验证方法，以保护其基于 Flask 的 Web API。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1346837200326889603)** (14 条消息🔥): 

> `SSH pain points, Better GPU providers, Nitrokey, SoloKey, Yubikey` 


- **用户寻找更好的 GPU 提供商**：用户讨论了 **SSH** 的痛点，并为他们的 **M2** 寻找更好的 GPU 提供商，或者考虑购买 **Blackwell GPU**。
   - 一位用户提到拥有 **RTX 3050** 和 **GFX90c**。
- **通过 VS Code 访问车库 PC**：一位成员描述了他们的配置：使用位于 **车库的 PC**、**VS Code** 以及用于免密 **SSH access** 的身份文件。
   - 他们还推荐使用 [Mutagen](https://mutagen.io/) 在笔记本电脑和服务器之间同步文件。
- **Nitrokey 提供更高安全性**：一位成员建议使用 **Nitrokey**、**SoloKey** 或 **Yubikey** 来增强安全性。
   - 他们表示这些设备*相对便宜，依然易于使用，提供更高的安全性，并且也可以用于其他账户*。
- **PC 在厨房水槽下安家**：一位成员分享了由于空间不足，他们如何将 **PC** 放在 **厨房水槽** 下方，尽管他们的*妻子并不高兴*。
   - 原因是*附近有一个电源插座*。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1346903165072576604)** (3 条消息): 

> `Tenstorrent, LlamaIndex, Koyeb, AI Infrastructure Meetup, Next-Gen hardware` 


- **Tenstorrent、LlamaIndex 和 Koyeb 将举办 AI Infrastructure 见面会**：**Tenstorrent**、**LlamaIndex** 和 **Koyeb** 团队今晚将在 SF 市中心举办一场关于 **AI Infrastructure** 和 **下一代硬件** 的小型见面会，地点见 [https://lu.ma/ruzyccwp](https://lu.ma/ruzyccwp)。
- **与团队见面**：这次见面会是一个与 **Tenstorrent** 和 **Koyeb** 团队交流的机会，了解他们的合作如何提供比传统 GPU 更高的性价比。
   - 该活动旨在连接 AI 开发者，让他们了解 AI infrastructure 领域的尖端创新。
- **SF 见面会计划**：一位成员表示他们将在波特兰，然后在两周内参加 GDC，并有兴趣届时在 SF 见面。



**提到的链接**：<a href="https://lu.ma/ruzyccwp">Next-Gen AI Infra with Tenstorrent &amp; Koyeb @LlamaIndex · Luma</a>：加入我们，与来自 LlamaIndex 的朋友们一起开启 Tenstorrent 和 Koyeb 之间开创性合作的特别夜晚。这次见面会是……

  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1346881365265612883)** (2 条消息): 

> `Reshaping vs Permuting, Triton Kernel Permutation, FPINT Dimension Right Shift` 


- **Reshaping 和 Permuting 并非孪生兄弟**：以行优先（row-major）顺序将 `M x N` 矩阵 Reshape 为 `N x M` 会保持元素顺序，但 Permuting（如 **transposing**）会改变顺序。
   - 一位用户强调它们*绝对不等价*。
- **Triton Kernel 需要 Permutation，而不仅仅是 Reshape**：在 Triton kernel 中，Permuting 是必要的，因为数据以 `(32, FPINT, GROUP)` 的形状加载，每个 **GROUP** 列表都具有相同的扩展反量化值。
   - 矩阵必须转置为 `(32, GROUP, FPINT)`，以便可以将右移操作 `offset_q >> over) & mask` 应用于最后一个维度，从而生成反量化值。
- **FPINT 维度需要右移**：用户解释说 `(offset_q >> over) & mask` 仅在最后一个维度上有效，这意味着 `offset_q.shape[-1] == over.shape[0]`。
   - 用户尝试通过设置 `over = 4*tl.arange(8).reshape((1, 8, 1))` 来右移 `offset_q.shape[1]` 维度，但失败了。


  

---

### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1346862321326362636)** (7 条消息): 

> `Radeon GPU Profiler, ROCm on Linux, rocprofilerv2 ATT plugin, rocclr, PAL Backend` 


- **在 ROCm Linux 上寻求类似 RGP 的指令计时功能**：一位用户询问是否有工具可以在 **ROCm on Linux** 上模拟 **Radeon GPU Profiler (RGP)** 中的指令计时（instruction timing）标签页功能。
   - 遗憾的是，据称 **RGP** 仅能在 Windows 上使用。有人建议在 Linux 上结合 **PAL backend** 编译 **rocCLR**，但其功能尚未得到确认。
- **rocprofilerv2 ATT 插件启动失败**：用户提到尝试使用 **rocprofilerv2** 的 **ATT plugin**，但*似乎无法正常工作*。
   - 尽管文档表明应该可以获取每条指令的延迟（latency per instruction），但其他人也确认他们*无法使其运行*，这可能是由于 [RDNA4 instruction set architecture](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf) 的兼容性问题导致的。


  

---


### **GPU MODE ▷ #[tilelang](https://discord.com/channels/1189498204333543425/1240586843292958790/1346801537447366717)** (17 条消息🔥): 

> `Shared Memory Allocation, Python Linting Warnings, CUDA Compatibility Issues (12.1 vs 12.4/12.6), Github Issue #149` 


- ****Shared Memory 分配**深度探讨**：一名成员引用了 [CUDA documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications) 中关于整体及每个 thread block 可用的 **shared memory** 规范。
   - 会议指出，确定所需的 shared memory 数量需要考虑所有的 `T.alloc_shared` 调用，并将维度与 **dtype itemsize** 相乘。
- ****Python Linting** 警告被暂时忽略**：一位成员承认警告主要是由于其 Pythonic DSL 中的 **Python linting** 引起的。
   - 他们承认*目前还没有找到绕过 lint 问题的简单方法*，因此暂时会忽略这些警告。
- ****CUDA 难题**：12.1 正常，12.4 报错**：一位用户报告了在 **RTX 4070 笔记本**上使用 **CUDA 12.4** 的问题，原本在 **CUDA 12.1** 上运行的代码出现了故障。
   - 尽管声明兼容 CUDA >= 11.0，但通过从 [tile-ai.github.io](https://tile-ai.github.io/whl/nightly/cu121/) 降级到 **cu121 nightly build** 暂时解决了该问题。
- ****Github Issue 已提交**：CUDA 12.4/12.6 的噩梦**：在 Tilelang 仓库中创建了一个 Github issue，以解决在 **CUDA 12.4** 和 **12.6** 上进行 matmul 操作时出现的元素不匹配错误。
   - 详见 [Github issue #149](https://github.com/tile-ai/tilelang/issues/149)。



**提到的链接**：<a href="https://github.com/tile-ai/tilelang/issues/149">Mismatched elements when performing matmul on CUDA 12.4/12.6 · Issue #149 · tile-ai/tilelang</a>：Bug 描述：我运行了下方的简单 matmul 代码，结果得到了 AssertionError: Tensor-likes are not close! 错误。该代码在 CUDA 12.1 上运行良好，但在 CUDA 12.4/12.6 上不行。不匹配的数量...

  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1347025757897232435)** (1 条消息): 

> `M3 Ultra, Unified Memory, Creative Uses of Unified Memory` 


- **M3 Ultra 的亮相引发对 Unified Memory 的思考**：成员们正在讨论 **M3 Ultra** 发布的影响，推测 **unified memory** 将如何促进创意应用。
- **Unified Memory 的创意潜力**：讨论重点在于结合 **M3 Ultra** 的发布，探讨 **unified memory** 在创意用途上的潜在可能性。


  

---

### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1346817771752656926)** (20 messages🔥): 

> `ARC-AGI 竞赛, QwQ-32B 发布, Reasoning-Gym 数据集, LADDER 框架` 


- **计划参与 ARC-AGI 竞赛**：成员们正计划参加 **ARC-AGI 竞赛**，这是继之前在 [open-thought GitHub profile](https://github.com/open-thought/reasoning-gym) 上的参与后的后续行动。
   - 该竞赛是推理模型的一个**新评估目标**。
- **Qwen 的 QwQ-32B 媲美 DeepSeek-R1**：**Alibaba Qwen** 发布了 [QwQ-32B](https://qwenlm.github.io/blog/qwq-32b)，这是一款全新的 **320 亿参数推理模型**，可与 **DeepSeek-R1** 等前沿模型媲美。
   - 该模型在**数学和编程**方面展示了令人印象深刻的结果，证明了通过 **RL 训练**实现的持续改进，并能与更大规模的 MoE 模型竞争。
- **Reasoning-Gym 为百项数据集庆典征集数据集**：**reasoning-gym** 项目目标是达到 **100 个数据集**，目前已有 **97 个数据集**，正在征集提案。
   - 已通过 [pull request 272](https://github.com/open-thought/reasoning-gym/pull/272) 和 [pull request 273](https://github.com/open-thought/reasoning-gym/pull/273) 提交了两个新数据集。
- **LADDER 框架通过递归简化进行学习**：一篇论文介绍了 **LADDER 框架** ([arxiv link](https://arxiv.org/abs/2503.00735))，该框架通过*递归地生成并解决复杂问题的渐进简化变体*，使 LLM 能够自主提升其解决问题的能力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/alibaba_qwen/status/1897361654763151544">来自 Qwen (@Alibaba_Qwen) 的推文</a>：今天，我们发布了 QwQ-32B，这是我们全新的推理模型，仅有 320 亿参数，却能媲美 DeepSeek-R1 等前沿推理模型。博客：https://qwenlm.github.io/blog/qwq-32b HF：https://hu...</li><li><a href="https://arxiv.org/abs/2503.00735">LADDER: Self-Improving LLMs Through Recursive Problem Decomposition</a>：我们介绍了 LADDER（通过自主难度驱动的示例递归进行学习），这是一个使 LLM 能够自主提升其问题解决能力的框架...</li><li><a href="https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html">ARC-AGI Without Pretraining</a>：未找到描述</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/273">Add Modulo Grid Task by Miserlou · Pull Request #273 · open-thought/reasoning-gym</a>：这可能是一个为第 100 个准备的原创任务？这是一个针对数学解释性推理的类 ARC 任务。它根据围绕...的隐藏数学函数生成二进制网格。</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/272">[Env] Game of Life Halting Prediction by Miserlou · Pull Request #272 · open-thought/reasoning-gym</a>：这是生命游戏任务的一个变体，它不是测试算法模拟，而是测试模型对棋盘进行解释性推理的能力。其想法是...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1346811138737700926)** (6 messages): 

> `程序员的 AI 方向, CUDA 微信群, TileLang DSL, Triton 序列填充` 


- **程序员的 AI 路径引发好奇**：一位成员询问了适合具有 3 年游戏开发经验、门槛较低的普通程序员的 **AI 方向**。
   - 他们还询问是否有可加入的 **CUDA 微信群**。
- **TileLang DSL 引起关注**：一位成员分享了 **TileLang** 的链接，这是一种用于简化高性能 GPU/CPU/Accelerators kernels 开发的领域特定语言 (DSL)：[tile-ai/tilelang](https://github.com/tile-ai/tilelang)。
   - TileLang 旨在简化为不同硬件平台创建高效 kernels 的过程。
- **Triton 填充性能探讨**：一位成员询问为什么当输入序列长度不是 32 的倍数时，**Triton 的性能**会下降。
   - 他们还询问了如果**输入序列未填充 (not padded)**，该如何修改代码。



**提到的链接**：<a href="https://github.com/tile-ai/tilelang">GitHub - tile-ai/tilelang: Domain-specific language designed to streamline the development of high-performance GPU/CPU/Accelerators kernels</a>：旨在简化高性能 GPU/CPU/Accelerators kernels 开发的领域特定语言 - tile-ai/tilelang

  

---

### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1346857475802009693)** (20 messages🔥): 

> `Modal Runners, GPU Leaderboards, Submission Errors` 


- **Modal Runners 在 GPU 排行榜表现出色**：多个使用 **Modal runners** 的测试和基准测试提交在不同的排行榜和 **GPU**（包括 **A100**, **H100**, **T4** 和 **L4**）上均获得成功。
   - 提交涉及的排行榜包括 `histogram`、`grayscale` 和 `prefixsum`。
- **提交脚本头部不匹配**：Cluster-Bot 报告称，命令中指定的排行榜名称与提交脚本头部中的名称不符，导致提交被重新路由至 `histogram` 或 `grayscale`。
   - 此问题表明预期的排行榜与实际提交目标之间可能存在差异。
- **Grayscale 获得 T4 性能提升**：针对 `grayscale` 排行榜的多次测试提交在 **T4 GPU** 上取得成功，表明该配置是关注重点。
   - 这些提交之后进行了基准测试和排行榜正式提交，表明针对 **T4 GPU** 上的 `grayscale` 正在进行持续的测试和优化工作。


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1347237634526675016)** (1 messages): 

> `Timeout durations` 


- **超时时长翻倍**：成员报告称，管理员已将所有 **timeouts**（超时时间）延长了一倍。
   - 用户被要求反馈是否仍会遇到进一步的 **timeout issues**（超时问题）。
- **超时问题报告**：现在请求成员报告遇到的任何超时问题。
   - 该请求是在所有超时时长翻倍后提出的，旨在主动识别潜在问题。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1346826892803571724)** (217 messages🔥🔥): 

> `Mojo's dynamism, Mutating classes in Mojo, Python vs Mojo, Async Django drawbacks` 


- **Mojo 是 Python 的超集吗？**：虽然曾有说法称 Mojo 是 Python 的超集，但该表述已被修正，因为即使 **C++** 也不是 **C** 的超集，而且超集的概念意味着要支持各种混乱的特性，比如修改类对象（mutating class objects）。
   - 成为 90 年代开发的语言的超集对 Mojo 来说就像是一个枷锁，因为它无法充分利用这些年来编程语言设计中大幅进化的特性。
- **为什么 Mojo 不应该有可变类对象？**：有人质疑为什么 Mojo 不应该拥有可变类对象，因为这是动态语言的核心特性，但答案是 Mojo 并非动态语言。
   - 成员们还讨论了动态性在许多语境下都是一种错误，正如 **JS** 引入了 **TS**，**Python** 引入了 **type hints**，而这些系统的首要任务就是锁定许多类似这样的特性。
- **Rust 提供了最符合人体工程学的方法**：成员们认定，他们在动态性方面取得的最大成功是避开了计算上困难的部分，转而提供一种在人体工程学上相当但在计算上更简单的替代方案。
   - 给人们他们需要的比给他们想要的更好，C 开发者至今仍在抱怨 Rust 过于严苛且不必要，但人们时不时就会利用他们的自由创造出新的 **CVE**，从而证明 Rust 的必要性。
- **全哈希表动态性是未来吗？**：讨论涉及该提案提供的框架，即 Mojo 应该拥有像 Python 那样“全哈希表、一切皆可改变”的动态性，还是一个更受限的版本，并提供一个开关来启用 Python 兼容性。
   - 一位成员表示：*“我个人希望‘第二级’即‘部分动态性（Partial dynamism）’就足够了。但也许 Modular 最终还是会选择‘全哈希表动态性’。”*
- **Async Django？坚决不用**：一位成员表示他们会避开 async Django。
   - 另一位成员补充道：*“让 Mojo 保持 ‘Pythonic’ 的初衷是为了弥合 AI 研究人员与模型部署人员之间的鸿沟。”*


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/modular/max/blob/main/mojo/proposals/mojo-and-dynamism.md">max/mojo/proposals/mojo-and-dynamism.md at main · modular/max</a>: MAX 平台（包含 Mojo）。通过在 GitHub 上创建账号为 modular/max 的开发做出贡献。</li><li><a href="https://github.com/python/cpython/blob/052cb717f5f97d08d2074f4118fd2c21224d3015/Include/longobject.h#L16">cpython/Include/longobject.h at 052cb717f5f97d08d2074f4118fd2c21224d3015 · python/cpython</a>: Python 编程语言。通过在 GitHub 上创建账号为 python/cpython 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1346876472148623503)** (5 条消息): 

> `Mojo/Python 项目基准测试, Mojo/Python 项目文件夹结构, Python.add_to_path 的替代方案, Mojo 测试中的 Symlink 替代方案` 


- **Mojo 二进制文件在 Python venv 中性能下降**：一位用户注意到，在激活的 **Python 虚拟环境**中运行 **Mojo 二进制文件**会显著降低性能，即使是那些没有导入任何 Python 模块的 Mojo 文件也是如此。
   - 用户寻求关于为何发生这种情况的见解，并期望不带 Python 导入的 Mojo 二进制文件不受 Python venv 的影响。
- **Mojo/Python 项目文件夹结构指南**：一位用户正在寻求关于构建 **Mojo/Python 项目**、导入标准 Python 库和自定义 Python 模块的指导。
   - 他们大量使用 `Python.add_to_path`，并在 `tests` 文件夹中实现了一个 Symlink。
- **寻求 Python.add_to_path 的替代方案**：一位用户正在为 Mojo 寻找定位自定义 Python 模块的 `Python.add_to_path` 替代方案。
   - 这是在构建混合 Mojo/Python 项目的背景下提出的。
- **Symlink 的替代方案**：一位用户正在寻求在 `tests` 文件夹中使用 Symlink 的替代方案，以便测试可以找到源文件。
   - 当前结构包括一个从 `tests` 到 `code` 的 Symlink。
- **创建了 Mojo/Python 文件夹论坛主题**：一位用户在 Modular 论坛上创建了一个名为 `Mojo/Python project folder structure` 的论坛主题。
   - 该用户可以通过此[链接](https://forum.modular.com/t/mojo-python-project-folder-structure/677)找到。



**提及的链接**: <a href="https://forum.modular.com/t/mojo-python-project-folder-structure/677">Mojo/Python project folder structure</a>: 我最初在 Discord 上发布了此内容（链接），但 @DarinSimmons 认为这会成为该论坛的一个好话题。我正在为一个大型 Mojo/Python 项目寻求文件夹组织的指导。我...

  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1347044658030575619)** (2 条消息): 

> `Modular 网站, 损坏的锚点链接` 


- **Modular 网站的锚点链接失效**：一位成员报告称，[Modular 网站 MAX 研究页面](https://www.modular.com/max/solutions/research)顶部的锚点链接已损坏，特别是 "Why MAX?" 链接。
   - 该成员建议这些链接可能是从另一个 "Solution" 页面复制过来的，并指出网站上的其他页面可能也存在类似问题。
- **网站反馈请求**：一位用户询问了报告网站问题（区别于 Mojo 文档问题）的适当渠道。
   - 该用户不确定应在哪里提交有关 Modular 网站功能和潜在 Bug 的反馈。


  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1346807507141791845)** (217 条消息🔥🔥): 

> `MiniCheck-Flan-T5-Large 事实核查模型，Qwen 32B 模型及 GGUF 量化，本地 AI 与 GPT4All 的局限性，AI Agent 的用户数据持久化` 


- ****MiniCheck** 优雅地验证事实**: **MiniCheck-Flan-T5-Large** 模型是一个基于 **Flan-T5-Large** 的事实核查工具，它通过预测二元标签来确定句子是否得到文档的支持。其代码和论文分别可在 [GitHub](https://github.com/Liyan06/MiniCheck) 和 [Arxiv](https://arxiv.org/pdf/2404.10774.pdf) 上获取。
   - 该模型的性能足以媲美 **GPT-4**，同时保持了小于 **1B** 的参数规模。
- ****Qwen 32B** 获得 GGUF 支持**: 一位成员分享了 [Qwen 推出的 QwQ-32B 的 Llamacpp imatrix 量化版本](https://huggingface.co/bartowski/Qwen_QwQ-32B-GGUF)链接，该版本使用了 *llama.cpp* release b4792 进行量化。
   - 这些量化版本是使用 *imatrix* 选项制作的，可以在 [LM Studio](https://lmstudio.ai/) 中运行。
- **应对 **GPT4ALL** 的 Token 上下文空间**: 用户讨论了在 **GPT4All** 的 Token 限制内工作的挑战，特别是在加载具有上下文窗口限制的本地文件时。
   - 一位用户指出，即使 Token 限制设置为 10,000 词，一个 **564 词的 TXT 文档** 也会导致错误并关闭整个会话。
- **解决记忆持久化的永恒难题**: 成员们讨论了在 **GPT4All** 中使 AI 模型能够**持久化用户数据**（如身高、体重和 BMI）的策略。
   - 共识是，将这些数据写入 System Message 可能是最好的方法，因为这样不太容易被遗忘。
- **探索真实性前沿的愿景**: 频道参与者对本地 AI 的未来进行了推测，设想向**硅嵌入式 AI 组件**转型，这些组件针对推理进行了优化并直接集成到硬件中，从而规避任何延迟。
   - 这一愿景还包括潜在的范式，例如利用大量**智能手机设备**，通过基于契约和代币化的算力交换，从而为空间感知、机器学习过程和网络完整性做出贡献。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/lytang/MiniCheck-Flan-T5-Large">lytang/MiniCheck-Flan-T5-Large · Hugging Face</a>: 暂无描述</li><li><a href="https://huggingface.co/lmstudio-community/QwQ-32B-GGUF">lmstudio-community/QwQ-32B-GGUF · Hugging Face</a>: 暂无描述</li><li><a href="https://huggingface.co/bartowski/Qwen_QwQ-32B-GGUF">bartowski/Qwen_QwQ-32B-GGUF · Hugging Face</a>: 暂无描述</li><li><a href="https://marketplace.visualstudio.com/items?itemName=saoudrizwan.claude-dev">Cline - Visual Studio Marketplace</a>: Visual Studio Code 扩展 - 直接在 IDE 中的自主编码 Agent，能够创建/编辑文件、运行命令...</li><li><a href="https://huggingface.co/collections/DavidAU/d-au-dark-planet-series-see-source-coll-for-fp-67086dc6f41efa3d35255a56">D_AU - Dark Planet Series (参见 FP 的 &quot;source&quot; 集合) - DavidAU 收藏集</a>: 暂无描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1346804926214574092)** (66 条消息🔥🔥): 

> `本地模型运行, Hugging Face Pro Plan, 目标检测, 多 GPU 设置, 欺诈检测` 


- ****Mistral Small** 在本地表现出色**：针对使用 **4080** 进行本地文本到文本模型运行，一位用户建议使用 **Llama 3.1**，另一位用户则推荐了 **Mistral small instruct quantized**（量化版）作为一个相当且不错的选择。
   - 后者提到它拥有 **24b** 参数，性能可与 **Llama 3.3 70b** 媲美。
- ****CoreWeave** 的 IPO 冲向云端**：**CoreWeave** 是一家为 **Meta** 和 **Microsoft** 等公司提供基于云的 **Nvidia** 处理器供应商。根据其 [IPO 招股书](https://www.sec.gov/Archives/edgar/data/1769628/000119312525044231/d899798ds1.htm)，该公司在 2024 年营收飙升 **700%** 达到 **19.2 亿美元**，净亏损为 **8.634 亿美元**，目前正准备上市。
- ****HF Pro** 计划：推理额度即将耗尽？**：一位成员对 **2 美元的推理额度** 表示担忧，询问即使在使用 **HF Pro Plan** 的情况下，是否有办法通过其他供应商来增加使用量。
   - 另一位成员回复称，有多种第三方供应商可供选择，支持各种模型。
- ****目标检测** 任务寻求帮助**：一位成员就一个与 **Object Detection**（目标检测）相关的计算机视觉任务寻求有经验者的帮助和建议。
   - 另一位成员建议，某个正在开发手势和距离检测项目的人可能能够提供帮助。
- ****并行进程** 用于训练脚本**：一位成员询问是否有在单机多 GPU 环境下，使用 **LoRA** 对 **PHI4** 等模型进行有监督微调（SFT）的示例 Notebook。
   - 另一位成员回复说，多 GPU 设置通常需要训练脚本（而非 Notebook），以便通过 **Slurm** 或类似的并行化工具启动多个进程，并指向了 [Hugging Face 的多 GPU 训练指南](https://huggingface.co/docs/transformers/perf_train_gpu_many)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/trl/main/en/gkd_trainer">Generalized Knowledge Distillation Trainer</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/transformers/perf_train_gpu_many">Efficient Training on Multiple GPUs</a>: 未找到描述</li><li><a href="https://huggingface.co/models?sort=modified&search=gguf)">Models - Hugging Face</a>: 未找到描述</li><li><a href="https://xircuits.io/docs/component-library/library-guides/pycaret/Pycaretanomaly">Anomaly Detection | Xircuits</a>: 在开始这些示例之前，请确保在您的工作环境中安装了 Pycaret=>2.2。您也可以使用 pip install pycaret==2.3.8 进行安装。</li><li><a href="https://github.com/unslothai/unsloth/issues/1285">`unexpected keyword argument tokenizer` [FIXED]  · Issue #1285 · unslothai/unsloth</a>: 我在 Mistral 模型上使用 orpo colab 示例时遇到了这个错误。我使用了来自 trl 的 ORPOConfig, ORPOTrainer 以及来自 unsloth 的 is_bfloat16_supported...</li><li><a href="https://www.cnbc.com/2025/03/03/ai-cloud-provider-coreweave-files-for-ipo.html?utm_source=join1440&utm_medium=email&utm_placement=newsletter&user_id=66c4c765600ae15075a57d0b">AI cloud provider CoreWeave files for IPO</a>: 依靠微软获得近三分之二收入的 CoreWeave 正准备上市。</li><li><a href="https://huggingface.co/spaces?q=qwen&sort=trending">Spaces - Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/cookbook/en/enterprise_dedicated_endpoints">Inference Endpoints (dedicated) - Hugging Face Open-Source AI Cookbook</a>: 未找到描述</li><li><a href="https://www.cnbc.com/2025/03/03/ai-cloud-provider-coreweave-files-for-ipo.html?utm">AI cloud provider CoreWeave files for IPO</a>: 依靠微软获得近三分之二收入的 CoreWeave 正准备上市。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1346815901026095136)** (7 条消息): 

> `Kornia Rust 库实习, LLM Guardrails 基准测试, Spikee 框架` 


- **Kornia 启动 Rust 库实习项目**：成员们宣布了作为 **Google Summer of Code 2025** 一部分的实习职位，旨在改进 **Kornia Rust library**；感兴趣的参与者可以查看[此处的文档和链接](https://summerofcode.withgoogle.com/programs/2025/organizations/kornia)。
- **Spikee 成为 LLM guardrail 的新选择**：一位成员测试了用于 LLM 基准测试的 guardrails，并发现 **Spikee framework** 是目前表现最好的。
   - 他们正在寻找用于 **LLM red teaming 活动**的其他替代框架。



**提到的链接**：<a href="https://summerofcode.withgoogle.com/programs/2025/organizations/kornia">Google Summer of Code</a>：Google Summer of Code 是一项全球性计划，专注于吸引更多开发者参与开源软件开发。

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1346918527952490536)** (4 条消息): 

> `Flash Attention, SAT 数据集, 用于 KV Cache 压缩的 Q-Filters` 


- **Umar Jamil 分享 Flash Attention 学习历程**：Umar Jamil 将于太平洋时间本周六（3 月 8 日）中午在 **GPU Mode** 频道分享他学习 **Flash Attention**、**Triton** 和 **CUDA** 的历程。
   - 该活动被宣传为“与观众进行的一次关于我在旅程中所遇困难的亲密对话，并分享关于如何自学任何知识的实用技巧” ([X 帖子](https://x.com/hkproj/status/1896113497031000563))。
- **Array 发布用于多模态语言模型的 SAT 数据集**：**Spatial Aptitude Training (SAT)** 数据集（一个视觉推理数据集）已在 HuggingFace Datasets 上发布，路径为 [array/SAT](https://huggingface.co/datasets/array/SAT)。
   - [项目页面](https://arijitray1993.github.io/SAT/)指出需安装 `datasets==3.0.2` 才能使用。
- **Q-Filters 实现免训练的 KV Cache 压缩**：一篇新论文介绍了 **Q-Filters**，这是一种用于高效 **KV Cache compression** 的免训练方法（training-free method），且与 **FlashAttention** 兼容 ([X 帖子](https://fxtwitter.com/nthngdy/status/1897301390470603245))。
   - 它可以在生成过程中进行压缩，这对于推理模型特别有用，例如具有 128 个 KV 对的 **R1-Distill-Llama-8B**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/hkproj/status/1896113497031000563?s=46">来自 Umar Jamil (@hkproj) 的推文</a>：我将于 3 月 8 日受 @GPU_MODE 邀请，分享我学习 Flash Attention、Triton 和 CUDA 的历程。这将是一场与观众关于我自身困难的亲密对话...</li><li><a href="https://fxtwitter.com/nthngdy/status/1897301390470603245">来自 Nathan Godey (@nthngdy) 的推文</a>：🚀 新论文发布！🚀 我们推出了 Q-Filters，一种用于高效 KV Cache 压缩的免训练方法！它与 FlashAttention 兼容，并可以在生成过程中进行压缩，这特别有用...</li><li><a href="https://huggingface.co/datasets/array/SAT">array/SAT · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1346827449937297448)** (7 messages): 

> `VisionKit, Deepseek-r1, TS-Agents, FastRTC, diRAGnosis` 


- **VisionKit 驱动 AI 聊天**：一个 AI 聊天功能由 **VisionKit** 驱动，但目前尚未**开源**；开发者正在考虑稍后将其开源。
   - 据开发者称，AI 模型 **Deepseek-r1** 的表现出人意料地好，他们还推荐了一篇[关于构建自定义 MCP server 的 Medium 文章](https://medium.com/data-scientists-from-future/model-context-protocol-custom-mcp-server-b26b32b9d804)。
- **TS-Agents 框架问世**：一位成员创建了 **TS-Agents**，这是一个用于构建 Agentic AI 流程的新型**基于 TypeScript 的框架**，并已发布在 [GitHub](https://github.com/piotrfrankowski/ts-agents) 上。
   - 作者提到，**LLMs** 的最新进展以及 **DeepSeek-R1** 等模型重新激发了他们对 AI 的兴趣，他们发现 TypeScript 框架比 Python 框架少见，并撰写了一篇[关于其开发历程的 Medium 文章](https://medium.com/@piotr-frankowski/ive-created-a-new-ts-based-ai-agentic-framework-f34d2bfe93a6)。
- **语音 AI 聊天亮相**：一位成员介绍了使用 **FastRTC**、**ElevenLabs**、**Next.JS** 和 **ShadCN** 构建的**语音 AI 聊天**，详情见 [Medium 文章](https://medium.com/@rohanprichard/fastrtc-a-quick-overview-for-ai-usecases-next-js-example-75de16c98c08)。
   - **FastRTC** 全面处理 RTC，可将任何 Python 函数转换为实时音频和视频流，并提供了一个 [GitHub 演示](https://github.com/rohanprichard/fastrtc-demo)。
- **diRAGnosis 实现 RAG 评估自动化**：一位成员发布了 **diRAGnosis**，这是一个针对 RAG 应用的全自动评估框架，可在 [GitHub](https://github.com/AstraBert/diRAGnosis) 和 [PyPi](https://pypi.org/project/diragnosis/) 上获取。
   - 该框架有助于*诊断 RAG 应用中 LLMs 和检索模型的性能*，支持 Docker，并与 LlamaIndex 集成，支持 **Mistral AI**、**Groq**、**Anthropic** 和 **OpenAI** 等供应商。
- **mixture_adapters 登陆 GitHub**：GitHub 用户 Temprl-pro-Business 创建了 [mixture_adapters](https://github.com/Temprl-pro-Business/mixture_adapters)，这是一个用于 LoRA 权重和基础模态的多适配器推理框架。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://medium.com/data-scientists-from-future/model-context-protocol-custom-mcp-server-b26b32b9d804">Model Context Protocol- Custom MCP Server</a>: 在本文中，我们将重点介绍构建自定义 MCP server。如果您需要 MCP 的入门介绍，请参考我之前的文章……</li><li><a href="https://github.com/Temprl-pro-Business/mixture_adapters">GitHub - Temprl-pro-Business/mixture_adapters: This is a Multi-Adapter Inference without merging the LORA Weights with Base modal.</a>: 这是一个无需将 LoRA 权重与基础模型合并的多适配器推理框架。- Temprl-pro-Business/mixture_adapters</li><li><a href="https://medium.com/@rohanprichard/fastrtc-a-quick-overview-for-ai-usecases-next-js-example-75de16c98c08">FastRTC: A quick overview for AI usecases + Next.Js example!</a>: 更快地将语音 AI 构建到您的应用中！</li><li><a href="https://github.com/rohanprichard/fastrtc-demo">GitHub - rohanprichard/fastrtc-demo: A simple POC of FastRTC, a framework to use voice mode in python!</a>: FastRTC 的简单概念验证（POC），一个在 Python 中使用语音模式的框架！- rohanprichard/fastrtc-demo</li><li><a href="https://github.com/AstraBert/diRAGnosis">GitHub - AstraBert/diRAGnosis: Diagnose the performance of your RAG🩺</a>: 诊断您的 RAG 性能🩺。欢迎在 GitHub 上为 AstraBert/diRAGnosis 的开发做出贡献。</li><li><a href="https://pypi.org/project/diragnosis/">diragnosis</a>: diRAGnosis - 诊断您的 RAG 性能！</li><li><a href="https://github.com/piotrfrankowski/ts-agents">GitHub - piotrfrankowski/ts-agents: Typescript based AI Agentic Framework</a>: 基于 TypeScript 的 AI Agentic 框架。欢迎在 GitHub 上为 piotrfrankowski/ts-agents 的开发做出贡献。</li><li><a href="https://medium.com/@piotr-frankowski/ive-created-a-new-ts-based-ai-agentic-framework-f34d2bfe93a6">I’ve created a new TS-based AI Agentic framework</a>: 🚀 我创造了一个新东西！ 🚀
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1347005463908192394)** (2 messages): 

> `DINOv2 fine-tuning, Weakly labeled images, Pose estimation, Hugging Face Computer Vision Hangout` 


- **DINOv2 Fine-Tuning 的困扰**：一位成员就如何使用 **600k 弱标签图像 (weakly labeled images)** 微调 **DINOv2** 以执行特定任务寻求建议，目标是将该 **Backbone** 用于 **Pose estimation** 和其他复杂任务。
   - 该成员指出，使用已发布的源代码从头开始训练似乎是可行的，但微调看起来更复杂；此外还考虑在 **Backbone** 不冻结的情况下训练分类任务，但由于标签模糊，担心模型无法学习到必要的语义信息。
- **Hugging Face Vision Hangout 亮点**：上周的 Computer Vision Hangout 录像现已上线，主题包括“Hugging Face 在 CV 领域的新进展” [链接](https://www.youtube.com/watch?v=YJIlRQs0Jpc&t=7s)。
   - 研讨会还涵盖了“为 Hugging Face 生态系统做贡献” [链接](https://www.youtube.com/watch?v=CeU5uOuQ7Hw)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=YJIlRQs0Jpc&t=7s"> - YouTube</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=CeU5uOuQ7Hw"> - YouTube</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1347196241313792040)** (2 messages): 

> `Decoder Masking Mechanisms, Inference in Decoder-Only Models, Attention Mechanisms` 


- **揭秘 Decoder 掩码机制**：一位成员询问了 **ChatGPT** 等 **Decoder-only** 模型在 **Inference** 过程中的 [掩码机制 (masking mechanism)](https://link.to.masking.mechanism)。
   - 该成员质疑，为什么不能对预测的 **Token** 使用无掩码的 **Attention**，以便利用来自 **Prompt** 和已预测序列的所有可用信息。
- **掩码使用与解码策略**：另一位成员回应称，掩码有不同的用途，例如 *Causal vs Padding vs MLM*，以及不同的 [解码策略 (decoding strategies)](https://link.to.decoding.strategies)，如 **Multi-token prediction** 或 **Speculative decoding**。
   - 他们指出，在不含 **Padding** 的单序列自回归 **Transformer** 进行 **Next token prediction** 时，不需要掩码；但在 **Batch** 设置中，除非所有序列长度相同，否则 **Padding masks** 是必要的。
- **使推理与训练匹配**：目标是使 [Inference 匹配 Training](https://link.to.inference.training)，如果数据与训练期间遇到的情况不符，模型质量将会下降。
   - 该成员认为，取消 **Prompt** 的掩码但对前两个生成的 **Token** 进行掩码处理似乎很奇怪，可能存在理解偏差。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1346841193958346844)** (5 messages): 

> `Reasoning Course, hf ecosystem, fine-tuning, telemetry, langfuse` 


- **Reasoning Course 是 smol-course 的逻辑演进**：课程创建者表示，他们正专注于将 [Reasoning Course 资料](https://huggingface.co/reasoning-course) 作为 smol-course 的*逻辑延续*。
- **新人询问 smol-course 的深度**：一位在构建带有 **Tool calls**、本地 **LLM** 和 **RAG** 的聊天机器人方面有经验的成员询问 smol-course 是否对他有用。
   - 他们表达了学习 **Hugging Face 生态系统** 的兴趣。
- **微调课程需求**：一位成员询问是否有关于如何**微调现有模型**的课程。
   - 他们表示目前正处于*困境*中，非常*渴望获得帮助*。
- **Telemetry 与 Langfuse 错误报告**：一位成员报告在第 2 单元中遇到 **Telemetry** 和 **Langfuse** 错误，并且*无法看到任何 Traces*。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1346806081678086225)** (109 条消息🔥🔥): 

> `Agentic AI vs AI Agents, SmolAgents with local LLM, HuggingFace inference API rate limits, HuggingFace course certificates, OpenRouter Free Model` 


- **澄清 Agentic AI 与 AI Agents 的区别**：一位成员询问了 **Agentic AI** 与 **AI Agents** 之间的区别，引发了关于它们在推理和适应请求中作用的讨论。
   - 另一位成员解释说，*两者都能进行推理并适应请求*，而 **AI Agents** 是专门设计用于作为 **Agentic AI** 运行的。
- **用户在本地 LLM 上运行 SmolAgents 时遇到困难**：多位成员报告了通过 **Ollama** 在本地 LLM 上运行 **smolagents** 时的问题，指出模型经常产生 *幻觉 (hallucinate)* 或无法正确使用提供的 **tools**。
   - 一位在 **16GB GPU** 上运行 **qwen 2.5 14b** 的用户观察到响应不准确且模型行为异常，而另一位用户建议查看 smolagents.org 官方网站。
- **成员遇到 HuggingFace Inference 速率限制**：多位成员报告在课程期间达到了 **HuggingFace Inference API 速率限制**，并寻求在不产生额外费用的情况下继续学习的解决方案。
   - 建议包括使用课程专用的模型端点 (`https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud/`)、登录以提高速率限制，以及探索像 **OpenRouter** 这样的替代推理提供商。
- **用户寻找缺失的 HuggingFace 课程证书**：一些成员报告在完成测验后难以找到他们的课程证书，引发了关于在哪里可以找到证书的讨论。
   - 一位用户提供了证书数据集的链接，该用户最终找到了它 ([https://huggingface.co/datasets/agents-course/certificates](https://huggingface.co/datasets/agents-course/certificates))，尽管有些用户遇到了姓名没有立即显示的问题。
- **OpenRouter 提供基于 Llama-3 的免费模型**：一位成员建议使用 **OpenRouter** 作为访问免费开源模型的替代方法，特别是那些以 *:free* 结尾的模型，以避免推理使用限制。
   - 该成员提供了在 **smolagents** 中使用 `OpenAIServerModel` 配合 **OpenRouter** 的示例，包括 API 基础 URL ([https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)) 以及指定模型 ID 的说明（例如 *meta-llama/llama-3.3-70b-instruct:free*）。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud'/)">未找到标题</a>: 未找到描述</li><li><a href="https://steanmcommunnuty.com/10529485">Steam Gift Activation</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/agents-course/unit1/final-quiz#certificate">Unit 1 Quiz - Hugging Face Agents Course</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/agents-course/unit1/final-quiz#certif">Unit 1 Quiz - Hugging Face Agents Course</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/smolagents/en/reference/models#smolagents.OpenAIServerModel">Models</a>: 未找到描述</li><li><a href="https://openrouter.ai/)">OpenRouter</a>: LLM 的统一接口。为您的提示词寻找最佳模型和价格</li><li><a href="https://openrouter.ai/api/v1',">Discord</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/agents-course/certificates">agents-course/certificates · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1346831805226942490)** (175 条消息🔥🔥): 

> `Gaslight Benchmark, GPT-4.5 图像生成, Video AI 提示词工程, Hermes 特殊 Token, Alibaba QwQ 32b 模型对比 DeepSeek R1` 


- **Gaslight Benchmark 探索开启**：成员们讨论了是否存在用于评估 **GPT-4.5** 等模型的 **gaslighting 基准测试**，但未发现明确的基准，一位用户开玩笑地建议了 [spiritshare.org 的链接](https://spiritshare.org/benchmark.html)。
   - 一位成员还抱怨 **ClaudeGrok** 在制作非写实图像或素描方面表现不佳。
- **命名为邪恶 AI：实验揭示 LLM 倾向**：一项实验表明，仅通过将一个 **8b 模型** 命名为 *“做坏事的邪恶 AI”*，就能使其变得 *“邪恶”*，展示了命名对模型行为的影响。
   - 该用户分享了一段 [展示 AI 行为的视频](https://cdn.discordapp.com/attachments/1149866623109439599/1346844343788634183/evil-pse.mov?ex=67cafb8a&is=67c9aa0a&hm=e90af96bb7f11bb6872e7ca723e1567cc2d1c4478794bedd9dcd6539fff12016&)。
- **阿里巴巴发布 QwQ 32B：挑战者现身**：**Alibaba** 发布了 **QwQ 32B 模型**，有人声称其性能可与 **DeepSeek R1 (671B)** 媲美，强调了小型且强大的开源模型趋势。
   - 然而，其他人指出 **QwQ-32b** 经常遇到 **16k Token 限制**，并且在一致性分离思考链（thinking trace）方面存在问题，而一些人发现它与 **Qwen-thinking** 相似。
- **知识图谱 GATs 软提示 LLMs**：一位成员正在将 **GAT** 的 Embedding 适配为 **LLM** 的 Soft Prompt，以利用 **G-Retriever** 给出的轮廓生成受 **GAT** 约束的响应。
   - 一位用户指向了一篇关于 [智能体化、自主图扩展的论文](https://arxiv.org/abs/2502.13025)，另一位用户分享了 [OpenSPG/KAG GitHub 仓库](https://github.com/OpenSPG/KAG) 的链接，这是一个基于 OpenSPG 引擎和 LLM 的逻辑形式引导的推理和检索框架。
- **AI 说服力的潘多拉魔盒**：成员们讨论了超越人类能力的 **AI 说服 Agent** 的潜力，例如能持续赢得在线辩论、吸引追随者或刷赞（karma farm）的机器人。
   - 一位用户指向了 OpenAI 用于说服力评估的 [evals make_me_say](https://github.com/openai/evals/tree/main/evals/elsuite/make_me_say) 基准测试。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/teortaxesTex/status/1896171547745988858">Teortaxes▶️ 的推文 (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>: 量化大佬们构建了行业内最优化的模型和推理栈，而物理学双博士构建了他们无法部署的最聪明模型，因为他们的基础设施代码是 prod_final_draft(3).ipynb...</li><li><a href="https://arxiv.org/abs/2502.13025">Agentic Deep Graph Reasoning Yields Self-Organizing Knowledge Networks</a>: 我们提出了一个智能体化、自主的图扩展框架，该框架可以迭代地就地构建和完善知识。与依赖静态提取的传统知识图谱构建方法不同...</li><li><a href="https://www.hermes-story.com/">Hermes and Argus | An Endless Dialogue</a>: 暂无描述</li><li><a href="https://en.wikipedia.org/wiki/Google_Knowledge_Graph">Google Knowledge Graph - Wikipedia</a>: 暂无描述</li><li><a href="https://www.youtube.com/watch?v=CZeot5H7Ilk">Deepseek R2 and Wan 2.1 | Open Source DESTROYS *everyone*</a>: 最新的 AI 新闻。了解 LLM、生成式 AI 并为 AGI 的推出做好准备。Wes Roth 报道了 OpenAI、Google、Anth... 领域的最新动态。</li><li><a href="https://github.com/openai/evals/tree/main/evals/elsuite/make_me_say">evals/evals/elsuite/make_me_say at main · openai/evals</a>: Evals 是一个用于评估 LLM 和 LLM 系统的框架，也是一个开源的基准测试注册库。 - openai/evals</li><li><a href="https://www.youtube.com/watch?v=qpKEyo1Gqqo"> - YouTube</a>: 暂无描述</li><li><a href="https://youtu.be/W2uauk2bFjs?si=MVhlTpK2kbaxmt-a"> - YouTube</a>: 暂无描述</li><li><a href="https://github.com/OpenSPG/KAG">GitHub - OpenSPG/KAG: KAG 是一个基于 OpenSPG 引擎和 LLM 的逻辑形式引导的推理和检索框架。它用于为专业领域知识库构建逻辑推理和事实问答解决方案。它可以有效克服传统 RAG 向量相似度计算模型的缺点。</a>: KAG 是一个基于 OpenSPG 引擎和 LLM 的逻辑形式引导的推理和检索框架。它用于为专业领域知识库构建逻辑推理和事实问答...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1346978634417832077)** (2 messages): 

> `` 


- **示例话题 1**：这是关于所讨论话题的示例总结句。
   - 这是另一个示例句子，详细阐述了讨论内容或提供了引用。
- **示例话题 2**：这是关于另一个不同话题的示例总结句。
   - 这是另一个句子，为该话题添加了更多背景或细节。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1346934090061975614)** (4 messages): 

> `QwQ-32B, DeepSeek R1, Reinforcement Learning Scaling, Tool Calling Syntax, Hermes Format` 


- **QwQ-32B 模型加入竞技场**：新款 **QwQ-32B** 模型已发布，尽管参数量显著减少（**32B** 对比 **671B**），但其性能已达到与 **DeepSeek-R1** 相当的水平。
   - 该模型利用 **Reinforcement Learning (RL)** 来增强推理能力，超越了传统的 pretraining 和 post-training 方法，详见其 [博客文章](https://qwenlm.github.io/blog/qwq-32b/)。
- **QwQ-Max 发布梦想推迟**：一位成员对发布的不是 **QwQ-Max** 表示失望，但指出 [benchmarks](https://qwenlm.github.io/blog/qwq-32b/) 看起来非常好。
   - 他们计划将该模型与 **R1** 进行 *vibe check*。
- **工具调用语法揭晓**：一位成员分享了用于获取当前温度和获取温度日期函数的 **tool calling syntax**。
   - 示例：*<tool_call> {"name": "get_current_temperature", "arguments": {"location": "San Francisco, CA, USA"}} </tool_call>*
- **新版本中发现 Hermes 格式**：有人注意到新版本使用了 **Hermes format**。



**提及的链接**：<a href="https://qwenlm.github.io/blog/qwq-32b/">QwQ-32B: Embracing the Power of Reinforcement Learning</a>：QWEN CHAT Hugging Face ModelScope DEMO DISCORDScaling Reinforcement Learning (RL) 有潜力在常规 pretraining 和 post-training 方法之外提升模型性能。最近的研究...

  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1346801638341607456)** (135 messages🔥🔥): 

> `Hand Fixing in SDXL, Free Video Creation from a Photo, Local vs SORA Video Generation Costs, SD3.5 Large TurboX Release, Running Stable Diffusion on GPU vs CPU` 


- **SDXL 自动手部修复方案浮现**：一位拥有 **8GB VRAM** 的用户正在寻求在 **SDXL** 中无需 inpainting 即可自动修复手部的方法，被推荐使用 *embeddings* 或 *face detailer* 进行手部修复，并添加 **OpenPose control net**。
   - 该用户还询问了适用于 **SDXL** 的优质 **hand LoRAs**。
- **免费照片转视频工具**：用户讨论了如何免费从单张照片创建视频，推荐了 **Wan 2.1 i2v model**，并提醒这需要高性能 GPU 和耐心，一位用户分享了 **SwarmUI** [视频模型支持文档](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Video%20Model%20Support.md#wan-21)的链接。
   - 提到的另一个选项是提供免费额度的在线服务，但效果参差不齐。
- **SORA 与本地成人影片生成成本对比**：讨论权衡了本地生成视频的成本（电费）与使用 **SORA** 等服务的成本，根据耗电量估算本地生成 5 秒视频的成本约为 **7 美分**，而 **SORA** 的成本可能为每段视频 **40 美分**。
   - 本地生成的优势：*uncensored*（无审查）内容。
- **SD3.5 Large TurboX 开源**：**TensorArt** 开源了 **SD3.5 Large TurboX**，该模型使用 8 个 sampling steps，比原始模型实现了 **6 倍的速度提升**，同时图像质量优于官方的 **Stable Diffusion 3.5 Turbo**；此外还发布了 **SD3.5 Medium TurboX**，仅需 4 个 sampling steps 即可在中端 GPU 上 1 秒内生成 **768x1248** 分辨率的图像。
   - 提供了 **SD3.5 Large TurboX** 的 [HuggingFace](https://huggingface.co/tensorart/stable-diffusion-3.5-large-TurboX) 链接和 **SD3.5 Medium TurboX** 的 [HuggingFace](https://huggingface.co/tensorart/stable-diffusion-3.5-medium-turbo) 链接。
- **GPU 利用率困扰**：一位用户遇到 **Stable Diffusion** 使用 **CPU** 而非 **GPU** 导致图像生成缓慢的问题，即使使用了 **3070 Ti**，建议尝试使用 **SwarmUI**。
   - 一位成员建议参考 [GitHub](https://github.com/mcmonkeyprojects/SwarmUI) 上的安装说明。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/stabilityai/cosxl">stabilityai/cosxl · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=KcYuWRB1_xI">如何在 Swarm 中运行新的 AI 视频模型 WAN</a>: 阿里巴巴开源了他们的 Wan 模型。最酷的是还有一个 1.3B 模型，几乎可以在所有带有 GPU 的系统上运行。让我们开始运行吧...</li><li><a href="https://blog.freneticllc.com/posts/lowfrequency/#how-does-stable-diffusion-work">隐藏在低频中的秘密</a>: 探索隐藏在高频数据低频段中的秘密——从无线电到数据加密，再到视频游戏世界的生成，以及这如何影响各种现实世界的决策...</li><li><a href="https://github.com/mcmonkeyprojects/SwarmUI">GitHub - mcmonkeyprojects/SwarmUI: SwarmUI（原名 StableSwarmUI），一个模块化的 Stable Diffusion Web-User-Interface，重点在于让强力工具易于访问、高性能且具有可扩展性。</a>: SwarmUI（原名 StableSwarmUI），一个模块化的 Stable Diffusion Web-User-Interface，重点在于让强力工具易于访问、高性能且具有可扩展性。 - mcmonkeyprojects/Swa...</li><li><a href="https://github.com/AIGText/Glyph-ByT5?tab=readme-ov-file">GitHub - AIGText/Glyph-ByT5: [ECCV2024] 这是论文 "Glyph-ByT5: A Customized Text Encoder for Accurate Visual Text Rendering" 和 "Glyph-ByT5-v2: A Strong Aesthetic Baseline for Accurate Multilingual Visual Text Rendering" 的官方推理代码。</a>: [ECCV2024] 这是论文 "Glyph-ByT5: A Customized Text Encoder for Accurate Visual Text Rendering" 和 "Glyph-ByT5-v2: A Strong Aesthetic...</li><li><a href="https://github.com/CompVis/stable-diffusion.git">GitHub - CompVis/stable-diffusion: 一个潜空间文本到图像扩散模型</a>: 一个潜空间文本到图像扩散模型。通过在 GitHub 上创建账号来为 CompVis/stable-diffusion 的开发做出贡献。</li><li><a href="https://old.reddit.com/r/StableDiffusion/comments/1j406g1/sd35_large_turbox_just_released/">SD3.5 Large TurboX 刚刚发布</a>: 由 u/NukeAI_1 发布于 r/StableDiffusion • 208 分和 57 条评论</li><li><a href="https://tenor.com/view/let-us-cook-let-me-cook-lets-cook-cooking-walter-white-gif-2649071825756414039">Let Us Cook Let Me Cook GIF - Let us cook Let me cook Lets cook - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Video%20Model%20Support.md#wan-21">SwarmUI/docs/Video Model Support.md at master · mcmonkeyprojects/SwarmUI</a>: SwarmUI（原名 StableSwarmUI），一个模块化的 Stable Diffusion Web-User-Interface，重点在于让强力工具易于访问、高性能且具有可扩展性。 - mcmonkeyprojects/Swa...</li><li><a href="https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Model%20Support.md#black-forest-labs-flux1-models>">SwarmUI/docs/Model Support.md at master · mcmonkeyprojects/SwarmUI</a>: SwarmUI（原名 StableSwarmUI），一个模块化的 Stable Diffusion Web-User-Interface，重点在于让强力工具易于访问、高性能且具有可扩展性。 - mcmonkeyprojects/Swa...</li><li><a href="https://github.com/neuratech-ai/ComfyUI-MultiGPU">GitHub - neuratech-ai/ComfyUI-MultiGPU: 在 ComfyUI 工作流中使用多 GPU 的初步支持</a>: 在 ComfyUI 工作流中使用多 GPU 的初步支持 - neuratech-ai/ComfyUI-MultiGPU</li><li><a href="https://wan.video/">Wan_AI 创意绘图_AI 绘画_人工智能_大模型</a>: Wan 是阿里巴巴旗下的 AI 创意绘画平台，提供文本生成图像、图像编辑、文本生成视频和图像生成视频等能力，用于 AI 驱动的艺术创作。</li><li><a href="https://www.wan-ai.org/">Wan AI</a>: Wan 2.1: 领先的 AI 视频生成模型 (Wanx 2.1)|Wan AI
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1347103240998748223)** (2 条消息): 

> `QwQ 32B Model, Reasoning Update, OAuth User ID, GitHub Authentication, OpenAI Provider Downtime` 


- **QwQ 32B 模型在 OpenRouter 上线！**：**QwQ 32B** 模型现已上线，包含来自 Grok 的[两个免费端点和一个快速端点](https://openrouter.ai/qwen/qwq-32b)（**410 tokens/sec**）。
- **默认包含推理（Reasoning）**：已推出一项更新，每当模型在生成补全前进行“思考”时，默认都会包含**推理**过程。
- **新增 OAuth 用户 ID 功能**：在 OAuth 密钥创建流程中添加了一个新字段 `user_id`，以便应用开发者为用户提供更个性化的体验。
- **启用 GitHub 身份验证**：用户现在可以在 OpenRouter 上使用 **GitHub** 作为身份验证提供商！
- **OpenAI 提供商出现停机**：OpenRouter 报告其 **OpenAI Provider** 模型出现停机，并表示该问题已在不到一小时内得到解决。



**提到的链接**: <a href="https://openrouter.ai/qwen/qwq-32b>">Discord</a>: 未找到描述

  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1346850582664839169)** (1 条消息): 

> `Android Chat App, Customizable LLMs, OpenRouter Integration, Speech To Text, Text To Image` 


- **Taiga 发布开源 Android 聊天应用**：一名成员发布了一款名为 **Taiga** 的[开源 Android 聊天应用](https://github.com/Ayuilos/Taiga/releases)，允许用户自定义 **LLMs**。
   - 该应用集成了 **OpenRouter**，并计划添加**本地语音转文本**（基于 Whisper 模型和 Transformer.js）、**文本转图像支持**以及基于 ChatTTS 的 **TTS 支持**。
- **Taiga 的下一步：语音转文本及更多**：开发者计划将使用 **Whisper** 模型和 **Transformer.js** 的**本地语音转文本**功能集成到应用中。
   - 未来的更新还包括添加**文本转图像**支持和基于 **ChatTTS** 的 **TTS 支持**，以增强应用的功能。



**提到的链接**: <a href="https://github.com/Ayuilos/Taiga/releases">Releases · Ayuilos/Taiga</a>: Taiga 是一款开源移动 AI 聊天应用，支持自定义 LLM 提供商。 - Ayuilos/Taiga

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1346808173994049556)** (112 messages🔥🔥): 

> `OpenRouter API 问题, DeepSeek 指令格式, Mistral OCR 发布, 基于用量的计费应用, 默认提示词功能` 


- **解决 OpenRouter API 的各种问题**：成员们讨论了与 [prefill 相关的 API 问题](https://discord.com/channels/1091220969173028894/1195014798837043240/1346854631606583337)、指令标签以及多轮对话的正确格式，特别是针对 DeepSeek 模型。
   - 有人指出，DeepSeek 在其 HF 页面上不建议对 R1 进行多轮对话，并建议使用 `<think>\n` 进行 prefill。
- **DeepSeek 的 Tokenizer 配置公开**：一位成员分享了 DeepSeek V3 的 [tokenizer config](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/tokenizer_config.json)，揭示了 `<｜begin of sentence｜>` 和 `<｜end of sentence｜>` token 的使用。
   - 澄清了 *add_bos_token* 为 true 而 *add_eos_token* 为 false，并提供了一个[短链接](https://shorturl.at/SqW9D)以防 Hugging Face 无法加载。
- **LLMGuard 寻求与 OpenRouter 集成**：一位成员询问了是否计划通过 API 集成像 **LLMGuard** ([llm-guard.com](https://llm-guard.com)) 这样的开源项目，以扫描提示词注入和 PII。
   - 有建议称这可以在将数据发送给提供商之前实现 **PII 匿名化**，但另一位成员指出，如果直接在调用方运行会更有用。
- **Groq 价格异常引发讨论**：用户注意到 **Groq 的 QwQ, Coder, R1 Distill** 与基础模型之间的价格和速度差异，如[分享的图片](https://cdn.discordapp.com/attachments/1347064719735001190/1347065032537673830/d641714d2c08c1a6f6e4834b5a5f5d16.png?ex=67cb2053&is=67c9ced3&hm=fb467c82de7333c5e997fb09cdee9a291171fb9e7b47e4153364abbaf2bb1bbf&)所示。
   - 测量结果表明，**Coder** 和 **QwQ** 的速度相似，而 **R1 Distill** 和基础模型有硬性限制，可能是为了优先考虑企业客户。
- **Google 停用 Gemini 2.0 之前的模型**：Google 宣布了 Vertex AI 上 Gemini 2.0 之前模型的[停用日期](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions)，计划于 **2025 年 4 月至 9 月**期间执行。
   - 这些模型包括 **PaLM, Codey, Gemini 1.0 Pro, Gemini 1.5 Pro/Flash 001/002** 以及部分 Embedding 模型。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/tokenizer_config.json">tokenizer_config.json · deepseek-ai/DeepSeek-V3 at main</a>: 无描述</li><li><a href="https://llm-guard.com/input_scanners/anonymize/">Anonymize - LLM Guard</a>: 无描述</li><li><a href="https://llm-guard.com/output_scanners/ban_competitors/">Ban Competitors - LLM Guard</a>: 无描述</li><li><a href="https://mistral.ai/news/mistral-ocr">Mistral OCR | Mistral AI</a>: 介绍世界上最好的文档理解 API。</li><li><a href="https://mistral.ai/news/mistral-o">undefined | Mistral AI</a>: 无描述</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions">无标题</a>: 无描述</li><li><a href="https://arxiv.org/abs/2412.19437v1">DeepSeek-V3 Technical Report</a>: 我们介绍了 DeepSeek-V3，这是一个强大的混合专家 (MoE) 语言模型，总参数量为 671B，每个 token 激活 37B。为了实现高效推理和低成本训练，DeepS...
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1347272318165979177)** (1 messages): 

> `用户研究, 礼品卡, NotebookLM` 


- **NotebookLM 研究招募用户；提供礼品卡！**：NotebookLM 团队正在寻求用户参与用户研究访谈（[报名表单](https://forms.gle/GxR2kwLdiXkzFMm89)），以提供对新 NBLM 概念的反馈。
   - 参与者将收到一张**礼品卡**作为感谢：**15 分钟**访谈（含 **10 分钟**准备）为 **$50**，或 **60 分钟**访谈（含 **10 分钟**准备）为 **$100**。
- **丰富的 Tremendous 礼品码**：参与用户访谈的人员将通过 Tremendous 的电子邮件收到礼品码。
   - 参与资格：必须年满 **18 岁**，能够向个人 Google Drive 上传/添加文件，并拥有稳定的视频通话网络连接。



**提到的链接**：<a href="https://forms.gle/GxR2kwLdiXkzFMm89">注册您的兴趣：NotebookLM 反馈</a>：您好，我们正在通过 15 分钟或 60 分钟的远程访谈寻求关于 NotebookLM 的反馈。这些反馈将帮助 Google 团队改进 NotebookLM 以进行未来的增强。申请参加请...

  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1346806760471527536)** (17 条消息🔥): 

> `Gemini 表现挣扎, NotebookLM PDF 支持, NotebookLM API, NotebookLM 在线游戏, NotebookLM 文档` 


- **Gemini 难以摆脱教学大纲**：一位成员上传了一本 **180 页的物理书**，并报告说 **Gemini** *无法摆脱我的教学大纲限制*。
- **NLM 缺乏对混合内容 PDF 的支持**：**NotebookLM** 不支持包含文本和图像混合内容的 PDF，但将其转换为 **Google Doc** 或 **Slides** 可以解决此问题。
   - 一位成员感谢另一位成员提供的 Google Doc 技巧，并表示 *这正是我所希望的！*
- **NotebookLM 的 API 正在开发中吗？**：成员们询问未来是否计划为 **NotebookLM** 提供 **API**，并列举了工作流优化的使用案例。
- **在线游戏策略师使用 NLM 优化 JSON 数据**：一位成员通过结合游戏文档、卡牌列表的 **JSON** 数据以及从电子表格中手动提取的数据，使用 **NotebookLM** 来优化在线游戏策略，但感觉 *这个工具并没有针对我的用途进行优化*，因为 *它通常认为我还没有完整阅读过源代码*。
   - 他们希望能够编辑源代码。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1346825873650614282)** (67 条消息🔥🔥): 

> `Android 应用, 回复长度, NotebookLM 公式, 文件上传问题, 将笔记导出为 PDF` 


- **PWA 应用填补 Android 应用空白**：用户询问是否有独立的 **Android 应用** 版 NotebookLM，其他成员指出可以通过 Chrome 或 AI Studio 在手机和电脑上安装 **PWA (Progressive Web App)** 版本。
   - 一位用户确认 **PWA** 可以正常工作，并可以保存到主屏幕。
- **回复内容变得异常冗长**：用户注意到 **NotebookLM 的回复变得比平时更长**，因此需要调整提示词和设置。
   - 一位成员提到，虽然每个提示词能获得更多信息很棒，但这需要更多的设置调整。
- **PDF 作为潜在格式的地位受到质疑**：一位用户质疑在 2025 年使用 **PDF** 的必要性，建议将 **HTML** 作为文档创建和处理的更优、开源替代方案，并链接到了 [转换工具](https://cdn.learny.academy/test-html/datalg-slides.html)。
   - 然而，另一位用户为 **PDF** 辩护，认为其具有便携性和易于打印的优点，尽管在编辑和上下文捕获方面存在挑战。
- **Notebook 笔记无法原生导出为 PDF**：一位用户询问如何将 NotebookLM 的笔记导出为 **PDF**，一位成员回答说 **没有直接导出功能**，建议将笔记复制到文档中再下载为 PDF，并链接到了 [功能请求讨论](https://discord.com/channels/1124402182171672732/1297146620626075681/1340698437749968907)。
   - 许多用户一致认为，他们希望与 Google Drive、Docs 和 Sheets 有更好的互操作性，包括导出和传输。
- **只要素材优质，Gemini 的语言处理表现极佳**：一位用户赞扬了加载商务会议录音的功能，特别是转录和识别发言者的能力。
   - 另一位用户将其识别为 *说话人日志 (Audio Diarization)*，并链接了 [ElevenLabs](https://elevenlabs.io/app/speech-to-text) 作为实用工具，观察到 **Gemini** 在处理非标准口音方面优于 **Whisper**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://cdn.learny.academy/test-html/datalg-slides.html">未找到标题</a>：未找到描述</li><li><a href="https://ai.google.dev/gemini-api/docs/models/gemini#available-languages">未找到标题</a>：未找到描述</li><li><a href="https://elevenlabs.io/app/speech-to-text">AI 语音生成器和文本转语音</a>：被评为在线最佳文本转语音 (TTS) 软件。免费创建优质 AI 语音，并使用我们的角色 AI 语音生成器在几分钟内生成文本转语音旁白。使用免费的文本转语音...</li><li><a href="https://www.nature.com/articles/s41586-025-08672-1">一个用于持续性、探索性和脱离状态的皮层下交换机 - Nature</a>：在小鼠身上进行的行为实验表明，中缝背核中的 GABA 能（表达 γ-氨基丁酸）、谷氨酸能和血清素能神经元具有独特且互补的功能...</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#languages-gemini">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1346885774045352007)** (77 条消息🔥🔥): 

> `Claude 成本, 新款 MacBook Air, Qwen 32B, 用于 Agent 的 React, Nicholas Carlini 加入 Anthropic`

- **Claude 每个问题收费 0.26 美元**：一位用户报告称，就其小型代码库向 **Claude** 提问一个问题的成本为 **0.26 美元**。
   - 另一位用户建议将代码库复制到 **Claude** 目录中，利用 filesystem MCP server，通过 Claude 订阅中的 token 来实现“免费”使用。
- **Apple 发布搭载 M4 芯片的 MacBook Air**：Apple 发布了新款 [MacBook Air](https://www.apple.com/newsroom/2025/03/apple-introduces-the-new-macbook-air-with-the-m4-chip-and-a-sky-blue-color/)，搭载 **M4 芯片**、**Apple Intelligence** 功能，并新增**天蓝色**，起售价 **999 美元**。
   - 新款 **MacBook Air** 提供了前所未有的价值，具备更强的性能、长达 **18 小时**的续航时间、**12MP Center Stage 摄像头**以及增强的外接显示器支持。
- **阿里巴巴发布 QwQ-32B 推理模型**：阿里巴巴发布了 [QwQ-32B](https://qwenlm.github.io/blog/qwq-32b)，这是一款拥有 **320 亿参数**的新型推理模型，可与 **DeepSeek-R1** 等顶尖推理模型相媲美。
   - 文中强调 **RL 训练**可以持续提升性能，尤其是在数学和编程方面，帮助中型模型在面对庞大的 **MoE 模型**时获得具有竞争力的表现。
- **React 是后端 LLM 工作流的最佳编程模型**：一位成员发表博文称 [React 是后端 LLM 工作流的最佳编程模型](https://x.com/_Evan_Boyle/status/1897347251120562205)。
   - 另一位用户表示，这种方法听起来像是在重新发明 **Lisp**，关键在于“设计符合应用所需组合性且对 LLM 具有可读性的代码模式”。
- **Nicholas Carlini 离开 Google DeepMind 加入 Anthropic**：[Nicholas Carlini](https://nicholas.carlini.com/writing/2025/career-update.html) 宣布在 **Google DeepMind** 工作七年后离职，加入 **Anthropic** 为期一年，继续其在 adversarial machine learning 领域的研究。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/Alibaba_Qwen/status/1897361654763151544">来自 Qwen (@Alibaba_Qwen) 的推文</a>：今天，我们发布了 QwQ-32B，这是我们全新的推理模型，仅有 320 亿参数，却能与最前沿的推理模型（如 DeepSeek-R1）相媲美。博客：https://qwenlm.github.io/blog/qwq-32b HF：https://hu...</li><li><a href="https://x.com/windsurf_ai/status/1897378545799979238">来自 Windsurf (@windsurf_ai) 的推文</a>：Windsurf Wave 4 发布了！本次更新包括：🖼️ 预览 ✏️ Cascade Auto-Linter ⚙️ MCP UI 改进 ➡️ Tab 键导入 ↩️ 建议操作 🫶 Claude 3.7 改进 🤝 推荐奖励 🖥️ Windows ARM 支持...</li><li><a href="https://x.com/cherry_cc12/status/1897366964080926902">来自 Chen Cheng (@cherry_cc12) 的推文</a>：谁将是下一个加入 QwQ 家族的成员？引用 Qwen (@Alibaba_Qwen)：今天，我们发布了 QwQ-32B，这是我们全新的推理模型，仅有 320 亿参数，却能与最前沿的推理模型相媲美...</li><li><a href="https://x.com/Alibaba_Qwen/status/1897366093376991515">来自 Qwen (@Alibaba_Qwen) 的推文</a>：Qwen2.5-Plus + Thinking (QwQ) = QwQ-32B。这就是你在 Qwen Chat 上使用这个新模型的方式！引用 Qwen (@Alibaba_Qwen)：今天，我们发布了 QwQ-32B，这是我们全新的推理模型，仅有 320 亿参数...</li><li><a href="https://www.together.ai/blog/nvidia-gb200-together-gpu-cluster-36k">Together AI 与 Hypertec Cloud 合作共同构建配备 36K Blackwell GPU 的增强型 NVIDIA GB200 集群</a>：未找到描述</li><li><a href="https://octotools.github.io/"> OctoTools：一个具有可扩展工具的 Agent 框架，用于复杂推理</a>：未找到描述</li><li><a href="https://nicholas.carlini.com/writing/2025/career-update.html">
      职业更新：Google DeepMind -> Anthropic
    </a>：未找到描述</li><li><a href="https://docs.mistral.ai/capabilities/document/#ocr-with-uploaded-pdf">OCR 与文档理解 | Mistral AI 大语言模型</a>：文档 OCR 处理器</li><li><a href="https://blog.google/products/search/ai-mode-search/">扩展 AI Overviews 并引入 AI Mode</a>：AI Mode 是 Google Search 中一项新的生成式 AI 实验。</li><li><a href="https://mistral.ai/fr/news/mistral-ocr">Mistral OCR | Mistral AI</a>：推出全球最佳的文档理解 API。</li><li><a href="https://x.com/nearcyan/status/1897466463314936034?s=46">来自 near (@nearcyan) 的推文</a>：今天发布 @elysian_labs 的首款产品：Auren！Auren 是人类/AI 交互范式的转变，旨在改善人类和 AI 的生活。这是我们 iOS 应用的剪辑...</li><li><a href="https://github.com/x1xhlol/v0-system-prompts">GitHub - x1xhlol/v0-system-prompts-and-models</a>：通过在 GitHub 上创建账户来为 x1xhlol/v0-system-prompts-and-models 的开发做出贡献。</li><li><a href="https://x.com/OpenAI/status/1897346510821711959">来自 OpenAI (@OpenAI) 的推文</a>：成为 Plus 用户的好日子。</li><li><a href="https://mastra.ai/docs/workflows/00-overview">处理复杂的 LLM 操作 | Workflows | Mastra</a>：未找到描述</li><li><a href="https://x.com/tim_cook/status/1897325061104918961">来自 Tim Cook (@tim_cook) 的推文</a>：向新款 MacBook Air 问好！这款全球最受欢迎的笔记本电脑现在配备了 M4 芯片、Apple Intelligence 功能以及亮丽的新配色——天蓝色。</li><li><a href="https://llmstxthub.com/websites">网站 - llms.txt hub</a>：发现实现 llms.txt 标准的精选网站列表。</li><li><a href="https://www.apple.com/newsroom/2025/03/apple-introduces-the-new-macbook-air-with-the-m4-chip-and-a-sky-blue-color/">Apple 推出配备 M4 芯片和天蓝色的新款 MacBook Air</a>：Apple 发布了新款 MacBook Air，配备 M4 芯片，电池续航长达 18 小时，拥有 12MP Center Stage 摄像头，且起售价更低。</li><li><a href="https://x.com/_Evan_Boyle/status/1897347251120562205">来自 Evan Boyle (@_Evan_Boyle) 的推文</a>：犀利观点：React 是后端 LLM 工作流的最佳编程模型。关于我们为何构建 @gensx_inc 的新博客文章。</li><li><a href="https://github.com/Tencent/HunyuanVideo-I2V">GitHub - Tencent/HunyuanVideo-I2V: HunyuanVideo-I2V: A Customizable Image-to-Video Model based on HunyuanVideo</a>：HunyuanVideo-I2V：基于 HunyuanVideo 的可定制图生视频模型 - Tencent/HunyuanVideo-I2V</li><li><a href="https://x.com/_sholtodouglas/status/1895610467818901609">来自 Sholto Douglas (@_sholtodouglas) 的推文</a>：非常激动地宣布，我在月初加入了 Anthropic！我所看到的一切都表明，我们正朝着 2027 年实现 AGI 的趋势迈进。如果趋势线继续保持，那么我们就有机会构建...
</li>
</ul>

### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1346960276326715434)** (41 条消息🔥): 

> `Synalinks framework, Async optimization, Constrained structured output, Functional API, Graph-based RAG` 


- ****Synalinks** 框架作为 **DSPy** 的替代方案亮相**：介绍了一个名为 **Synalinks** 的新型**基于图的可编程神经符号 LM 框架**，该框架灵感源自 **Keras**，专注于**知识图谱 RAG**、**强化学习**和**认知架构**。
   - 该框架旨在实现完全的**异步优化 (async optimized)**，默认具备**受限结构化输出 (constrained structured output)** 功能，并提供**函数式 API (functional API)**，已提供 [代码示例](https://huggingface.co/spaces/YoanSallami/synalinks-noteboooks)。
- ****Synalinks** 宣扬生产优先的优势**：**Synalinks** 声称相比 **DSPy** 具有多项优势，如**自动异步优化**、**受限结构化输出**、更易用的**函数式 API** 以及更好的序列化能力，使其更适合生产环境。
   - 其他特性包括**冻结模块 (freezing modules)**、定义**默认示例/提示 (default examples/hints)** 以及个性化 **Jinja2** 提示词模板。
- ****Synalinks** 开创了逻辑流控制**：**Synalinks** 的一个独特之处在于其**灵感源自逻辑电路的逻辑流 (logical flows)**，这允许在程序实例化期间根据 **JSON schema** 有条件地限制计算图。
   - 与 **DSPy** 中隐式图不同，**Synalinks** 显式地计算图，从而对计算流提供更多控制。
- ****Synalinks** 通过 Action 模块实现 RAG 工具**：**Synalinks** 使用 [Action 模块](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Action%20module/) 实现 **RAG** 工具，该模块利用 **LanguageModel** 执行带有结构化输出的函数调用。
   - 该框架还提供了一个 [ReACT Agent 模块](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/ReACT%20Agent%20module/)，可为每一步的函数选择创建有向无环图。
- ****Synalinks** 偏好传统编码而非 AI 生成**：**Synalinks** 的作者提到，代码库几乎没有使用 AI 创建，并表示 *"基于经过验证的开源系统进行构建的老方法，比使用 AI 从头开始编写代码要好 10000 倍。"*
   - 作者澄清说，该框架不一定是 **DSPy** 的替代品，而是一种专注于**提示词优化 (prompt optimization)**、**强化学习**和**图 RAG (graph RAG)** 的不同方法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/YoanSallami/synalinks-noteboooks">synalinks notebooks - YoanSallami 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Action%20module/">Action 模块 - Synalinks</a>: 未找到描述</li><li><a href="https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/ReACT%20Agent%20module/">ReACT Agent 模块 - Synalinks</a>: 未找到描述</li><li><a href="https://synalinks.github.io/synalinks/Synalinks%20API/Data%20Models%20API/The%20Variable%20class/">Variable 类 - Synalinks</a>: 未找到描述</li><li><a href="https://github.com/SynaLinks/synalinks">GitHub - SynaLinks/synalinks: 🧠🔗 基于图的可编程神经符号 LM 框架 - 一个基于十年深度学习最佳实践构建的生产优先 LM 框架</a>: 🧠🔗 基于图的可编程神经符号 LM 框架 - 一个基于十年深度学习最佳实践构建的生产优先 LM 框架 - SynaLinks/synalinks
</li>
</ul>

</div>
  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1346844637356232808)** (22 条消息🔥): 

> `DSPy 意图分类优化，使用 DSPy 比较文本矛盾，DSPy 用于结构化输出的 Adapter 系统，DSPy 中的滞后线程修复，dspy.Signature 中的可变输出字段` 


- **DSPy 提升意图分类**：使用 **DSPy** 可以通过专门的 Agent 帮助优化意图分类。
   - 一位用户确认，使用 DSPy 是满足其意图分类需求的正确方向。
- **讨论 DSPy 的矛盾比较器**：一位用户正在使用 **DSPy 的 CoT 模块**比较两段文本是否存在矛盾，但发现计算开销很大。
   - 他们正在寻求更好的方法来解决这个问题，因为 function calling 可能不适用于返回一个值列表。
- **DSPy 的 Adapter 保证结构化输出**：DSPy 的 "adapters" 系统将你的 signature（一种声明式指定需求的方式）与不同 provider 生成补全的方式解耦。
   - 在 2.5 和 2.6 版本的底层，它运行一个经过良好调优的 **ChatAdapter**，并回退到 **JSONAdapter**，后者在提供显式约束解码的 provider 中使用结构化输出 API。
- **滞后线程阻碍并行 DSPy**：一个 [已合并的 PR 7914](https://github.com/stanford-nlp/dspy/pull/7914) 通过修复“滞后（straggler）”线程，使 **DSPy 的 `dspy.Evaluate` 或 `dspy.Parallel`** 运行更顺畅。
   - 用户可以在 DSPy 2.6.11 发布之前从 `main` 分支尝试该功能，无需更改代码，但需要从 main 分支获取库。
- **DSPy Signature 的可变输出字段**：一位用户询问如何创建具有可变输出字段的 **dspy.Signature**，例如有时输出 A、B、C，有时输出 D、E 和 F。
   - 一位成员建议查看 [react.py](https://github.com/stanford-nlp/dspy/blob/main/dspy/experimental/react.py) 文件。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1346896645736108103)** (2 条消息): 

> `Agentic 文档工作流，互操作性 Agent 标准` 


- **LlamaIndex 与 DeepLearningAI 联手！**：直接集成到大型软件流程中的 **Agentic Document Workflows** 是知识型 Agent 的未来，LlamaIndex 已与 [DeepLearningAI 合作推出这门短课](https://t.co/EvAKtIAzlC)，教你如何构建它们。
- **提议 Agent 开放标准**：LlamaIndex 正在参与创建一项**开放、互操作的 Agent 标准**，涵盖从发现、部署到相互通信的各个环节，详见[此公告](https://t.co/ECHH1T4Kxn)。



**提到的链接**：<a href="https://t.co/ECHH1T4Kxn">Outshift | 构建 Agent 互联网：介绍 AGNTCY.org</a>：了解 Cisco 的最新技术创新并关注前沿思想动态。

  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1346820815391424553)** (58 条消息🔥🔥): 

> `LlamaIndex ImageBlock 与 OpenAI 的问题、Query Fusion Retriever 引用问题、分布式 AgentWorkflow 架构、Agent 执行的性能分析/计时、Flask 和 Gunicorn 的内存消耗` 


- **OpenAI 集成中的 ImageBlock 故障**：一位用户报告了在最新版本的 LlamaIndex 中将 **ImageBlock** 与 OpenAI 配合使用时出现的问题，即图像无法被识别；[kapa.ai](https://kapa.ai) 提供了常规排查建议，包括检查版本和正确的模型配置。
   - 排查步骤包括确保使用最新的 LlamaIndex 版本，验证是否使用了支持图像输入的模型（例如 **gpt-4-vision-preview**），并确认 OpenAI LLM 实例已正确配置。
- **QueryFusion 检索丢失引用**：一位用户报告称，与单独使用 **index_retriever** 不同，将 **QueryFusionRetriever** 与节点后处理器（node post-processor）配合使用时无法生成引用模板，并提供了一个 [GitHub 仓库链接](https://github.com/Restodecoca/ingest-reposit/tree/main/app/engine) 以协助排查。
   - 有建议指出，问题可能源于 **BM25 retriever** 或 **query fusion retriever** 的倒数重排序（reciprocal rerank），可能在节点去重（node de-duplication）过程中导致了元数据（metadata）丢失。
- **开箱即用的分布式 AgentWorkflow**：一位用户询问关于在分布式架构中运行 **AgentWorkflow** 的原生支持，即 Agent 位于不同的服务器或进程中。
   - 建议指出 **AgentWorkflow** 是为单个活动 Agent 设计的，实现所需设置可能涉及为 Agent 配备用于远程服务调用的工具（tools）。
- **用于瓶颈识别的 Agent 运行时性能分析**：一位用户询问 LlamaIndex 是否原生支持测量/计时多 Agent 应用中不同 Agent 的执行情况。
   - 建议使用像 Arize 这样的第三方服务进行可观测性（observability）分析。
- **OpenAI 音频模型遇到 Agent 流式传输问题**：由于音频流式传输问题，一位用户在将 OpenAI 的音频模型 **gpt-4o-audio-preview** 与 Agent 配合使用时遇到了 `WorkflowRuntimeError`，并提供了一段[代码片段](https://cdn.discordapp.com/attachments/1346959475445207120/1346987472697032775/class_TestWorkflow.py?ex=67cad817&is=67c98697&hm=a7e24715eef6cd2a4850fd318fd61cef1e36e2a35b2d9a0d097c1e918bb63241&)展示当前的实现。
   - 注意到 **AgentWorkflow** 会自动对聊天消息调用 `llm.astream_chat()`，这可能与 OpenAI 的音频流不兼容；建议是避免使用 AgentWorkflow 或通过标志禁用 LLM 流式传输。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/Restodecoca/ingest-reposit/tree/main/app/engine">ingest-reposit/app/engine at main · Restodecoca/ingest-reposit</a>: 通过在 GitHub 上创建账号来为 Restodecoca/ingest-reposit 的开发做出贡献。</li><li><a href="https://github.com/run-llama/llama_index/blob/ea1f987bb880519bb7c212b33d8615ae4b8fdbf8/llama-index-core/llama_index/core/agent/workflow/function_agent.py#L41">llama_index/llama-index-core/llama_index/core/agent/workflow/function_agent.py at ea1f987bb880519bb7c212b33d8615ae4b8fdbf8 · run-llama/llama_index</a>: LlamaIndex 是构建基于数据的 LLM 驱动 Agent 的领先框架。 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/aa192413a398b5330d23a4901a42976419bb7128/llama-index-core/llama_index/core/agent/function_calling/step.py#L205">llama_index/llama-index-core/llama_index/core/agent/function_calling/step.py at aa192413a398b5330d23a4901a42976419bb7128 · run-llama/llama_index</a>: LlamaIndex 是构建基于数据的 LLM 驱动 Agent 的领先框架。 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/issues/18035">[Documentation]: Tools cannot put the output of one tool into another in a single turn · Issue #18035 · run-llama/llama_index</a>: 文档问题描述，此处的文档似乎不正确：https://docs.llamaindex.ai/en/stable/examples/agent/openai_agent_parallel_function_calling/#sync-mode 它显示了输出...</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/workflow/function_calling_agent/">Workflow for a Function Calling Agent - LlamaIndex</a>: 暂无描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/openai/#audio-support">OpenAI - LlamaIndex</a>: 暂无描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/openai_agent_parallel_function_calling/">Single-Turn Multi-Function Calling OpenAI Agents - LlamaIndex</a>: 暂无描述
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1346886048961269874)** (21 messages🔥): 

> `Bilevel Optimization, Sparsemax, Model Checkpoints with DDP, Compositmax` 


- **Bilevel Optimization 并没有泛化 Sparsemax**：一位成员认为 Bilevel Optimization (BO) 是一种标准形式，本质上并没有做任何不同的事情，并指出它等同于带有互补约束（complementarity constraint）的单层优化。
   - 另一位成员建议将 Sparsemax 视为一种 BO，并且许多 AI 问题都适合重新表述为 BO/MO。BO 是一种基于投影的优化，随后讨论了将层级结构折叠为单层以获得闭式解（closed forms），这在事物尽可能简单时效果最好。
- **使用 DDP 时模型 Checkpoints 出现错乱**：一位成员在使用 **PyTorch**、**DDP** 和 **4 GPU** 时遇到了模型 Checkpoint 重新加载在多 GPU 上出现错乱的问题，但在单 GPU 上运行完美。
   - 有建议认为初始化 DDP 和加载 Checkpoint 的顺序很重要：先初始化模型，在所有 GPU 上加载 Checkpoint，然后再初始化 DDP。
- **用于复合 arg max 的 Compositmax**：一位成员介绍了用于复合 arg max 的 Compositmax，并观察到 Softmax 是 soft arg max，Sparsemax 是 sparse arg max，Entmax 是 entropy arg max，而 Compositmax 则是 composite arg max。
   - 这种正则化项以及其他正则化项的目的是基于样条函数（splines）的思想设计新的正则化项，目标是使它们比日益流行的 Entmax 更快。



**提及的链接**：<a href="https://x.com/SchmidhuberAI/status/1897406236896977388">Jürgen Schmidhuber (@SchmidhuberAI) 的推文</a>：祝贺 @RichardSSutton 和 Andy Barto 获得图灵奖！

  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1347012093320102010)** (9 messages🔥): 

> `Proactive T2I Agents, User Prompt Underspecification, Belief Graph Editing, Bash Shell Puns` 


- **Agent 主动询问图像意图**：一篇新论文 [Proactive Agents for Multi-Turn Text-to-Image Generation Under Uncertainty](https://arxiv.org/abs/2412.06771) 提出了 **proactive T2I agents**，它们在不确定时会主动提出澄清性问题，并将其对用户意图的理解呈现为可理解的信念图（belief graph）。
   - 论文摘要指出，*至少 90% 的人类受试者认为这些 Agent 及其信念图对他们的 T2I 工作流很有帮助*。
- **错过了 DeepMind 论文讨论**：一位成员对错过 **DeepMind 论文讨论** 表示遗憾，表明对 DeepMind 的研究贡献高度重视。
   - 其他成员也表达了同样的看法，称 *在我看来 DeepMind 的论文是最棒的*。
- **在 Google Tech Talk 中观看主动式 Agent 的讲解**：一位成员分享了 Meera Hahn 关于不确定下多轮文本到图像生成的 **Google TechTalk** [YouTube 视频](https://youtu.be/HQgjLWp4Lo8?si=6SxQdUbzocp3zrKD)。
   - 视频描述强调，生成式 AI 模型的 **User Prompt** 通常描述不足（underspecified），导致响应不佳，而 Agent 试图解决这一问题。
- **用 Bourne Again Shell 砸头**：针对一张名为 "Don't mess with JSON" 的表情包，一位成员拿 [Bourne Again shell (Bash)](https://en.wikipedia.org/wiki/Bash_(Unix_shell)) 开了个玩笑。
   - 另一位成员回应说，*他要用 `bash` 砸烂你的头*（此处是 Bash shell 的双关语）。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2412.06771">Proactive Agents for Multi-Turn Text-to-Image Generation Under Uncertainty</a>：生成式 AI 模型的 User Prompt 通常描述不足，导致响应不佳。这个问题在文本到图像 (T2I) 生成中尤为明显，用户通常难以……</li><li><a href="https://youtu.be/HQgjLWp4Lo8?si=6SxQdUbzocp3zrKD">Proactive Agents for Multi-Turn Text-to-Image Generation Under Uncertainty</a>：Google TechTalk，由 Meera Hahn 演讲，2024-12-05。摘要：生成式 AI 模型的 User Prompt 通常描述不足或开放式，这可能导致……
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1346871725404454972)** (14 messages🔥): 

> `AMD FSR 4 vs DLSS, Alibaba Qwen QwQ-32B 模型, Cortical Labs 生物计算机, 神经元可卡因/LSD 实验` 


- **AMD FSR 4 实现飞跃**：一段 [YouTube 视频](https://www.youtube.com/watch?v=nzomNQaPFSk) 测试了 **AMD 的 FSR 4** 超分辨率技术与 **Nvidia 的 DLSS 3/4** 的对比，表明在 **RDNA 4** 机器学习的驱动下取得了显著进步。
   - 视频描述暗示，Nvidia 的超分辨率优势可能会被 AMD 基于机器学习的方法所削弱甚至抵消。
- **阿里巴巴 Qwen 发布 QwQ-32B**：**阿里巴巴 Qwen** 发布了 **QwQ-32B**，这是一款仅有 **320 亿参数** 的新型推理模型，据 [这条推文](https://x.com/Alibaba_Qwen/status/1897361654763151544?t=t5Bec1knVsQuXpTu24fWqw&s=19) 称，其性能可媲美 **DeepSeek-R1** 等顶尖推理模型。
- **脑细胞与硅基融合**：**Cortical Labs** 于 2025 年 3 月 2 日在巴塞罗那正式商业化推出了 **CL1**，如 [这篇文章](https://newatlas.com/brain/cortical-bioengineered-intelligence/) 所述，这是全球首台将人类脑细胞与硅硬件融合的生物计算机。
   - 该系统被称为 **合成生物智能 (SBI)**，据称其学习速度和灵活性均优于用于训练 ChatGPT 等 LLM 的硅基 AI 芯片；详见 [Cortical Labs 官网](https://corticallabs.com/)。
- **LLM 以 2 万美元月薪取代程序员？**：一段 [YouTube 视频](https://www.youtube.com/watch?v=HDEpjTvO5PQ) 讨论了 **OpenAI** 据称要以每月 20,000 美元的成本取代程序员和博士的计划。
   - 视频涵盖了包括 **LLM** 和 **GenAI** 在内的最新 AI 新闻，为观众迎接 AGI 的到来做准备。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://corticallabs.com/">Cortical Labs</a>：我们将实验室培养的神经元与硅芯片结合，并首次向所有人开放。</li><li><a href="https://en.wikipedia.org/wiki/Marvin_the_Paranoid_Android">Marvin the Paranoid Android - 维基百科</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=nzomNQaPFSk">AMD FSR 4 超分辨率对比 DLSS 3/4 测试 - 重大飞跃 - RDNA 4 表现出色！</a>：AMD 的 FSR 4 超分辨率效果究竟如何？Radeon 团队基于机器学习的方法在多大程度上削弱甚至抵消了 Nvidia 的超分辨率优势...</li><li><a href="https://www.youtube.com/watch?v=HDEpjTvO5PQ">OpenAI 取代程序员、博士等的“阴谋”，月薪 20,000 美元...</a>：最新 AI 新闻。了解 LLM、Gen AI 并为 AGI 的推出做好准备。Wes Roth 报道了 OpenAI、Google、Anth... 领域的最新动态。</li><li><a href="https://x.com/Alibaba_Qwen/status/1897361654763151544?t=t5Bec1knVsQuXpTu24fWqw&s=19">来自 Qwen (@Alibaba_Qwen) 的推文</a>：今天，我们发布了 QwQ-32B，这是我们全新的推理模型，仅有 320 亿参数，可媲美 DeepSeek-R1 等顶尖推理模型。博客：https://qwenlm.github.io/blog/qwq-32b HF：https://hu...</li><li><a href="https://newatlas.com/brain/cortical-bioengineered-intelligence/">全球首个“合成生物智能”在活体人类细胞上运行</a>：全球首台将人类脑细胞与硅硬件融合以形成流体神经网络的“生物计算机”已商业化推出，开启了 AI 技术的新时代...</li><li><a href="https://en.wikipedia.org/wiki/Naegleria_fowleri">Naegleria fowleri - 维基百科</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1346824786889801871)** (2 messages): 

> `自我介绍, AI 生物黑客, 机器去学习 (Machine Unlearning)` 


- **Suleiman 介绍生物黑客 AI**：Suleiman 是一家沙特公司的管理人员，拥有软件工程背景，他介绍了自己并表达了对 **科技** 和 **AI** 的热爱。
   - 他目前正在探索 **营养学** 和 **补充剂科学**，旨在开发 **AI 驱动的生物黑客工具** 以改善人类生活。
- **Naveen 研究机器去学习**：Naveen 是一位来自 IIT 的硕士兼研究助理，他介绍了自己并提到他在 **文本生成图像扩散模型中的机器去学习 (Machine Unlearning)** 方面的工作。
   - 他还提到他最近在 **CVPR25** 上发表了一篇论文。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1347000060550058046)** (4 messages): 

> `ARC Training, Lossless Compression, Relative Entropy Coding (REC), Encoder-Free Sample Dependent VAE` 


- **ARC 训练准确率达成**：成员们讨论了仅使用推理时示例在 **ARC training** 上达到 **35%** 的准确率，并引用了 [Isaac Liao 和 Albert Gu 的博客文章](https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html)，该文章探讨了无损信息压缩。
   - 该博客文章质疑 *高效压缩本身是否就是智能的核心*。
- **联合优化的困惑**：一位成员询问了[博客文章](https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html)中描述的方法，质疑其是否涉及 *联合优化潜变量 (latents) 和解码器参数以拟合现有示例（一种 Encoder-Free Sample Dependent VAE？）*。
- **Relative Entropy Coding 浮现**：一位成员分享了一篇关于 [Relative Entropy Coding (REC)](https://arxiv.org/abs/2010.01185) 的论文，认为它是所讨论的无损压缩方法的主要基础。
   - 论文摘要指出，REC 可以直接对潜表示进行编码，对于单张图像，其码长接近相对熵。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2010.01185">Compressing Images by Encoding Their Latent Representations with Relative Entropy Coding</a>：Variational Autoencoders (VAEs) 已在学习型图像压缩中得到广泛应用。它们被用于学习表现力强的潜表示，下游压缩方法可以在此基础上运行...</li><li><a href="https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html">ARC-AGI Without Pretraining</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1347061939028885554)** (16 messages🔥): 

> `Pythia Loss Curves, Kaplan-style loss vs compute convex hull plot, FLOPs PPO uses per token` 


- **Pythia 棘手的绘图问题**：**Pythia** 的损失曲线确实存在，但 **WandB metadata** 已损坏，导致其难以解读。
   - 此外，由于学习率衰减因子的影响，部分训练的检查点 (checkpoints) 并不是完全训练模型的准确代理，这一错误在 [Chinchilla 论文](https://arxiv.org/abs/2203.15556) 中得到了修正。
- **思考 PPO 的每 Token FLOPs**：生成 K 个 token 与对 K 个 token 进行前向传播的 FLOPs 相同，由于参考模型需要额外的一次前向传播，导致推理、奖励和参考模型的 **前向传播成本为 3 倍**。
   - 加上价值模型和策略的反向传播，整个过程估计约为 **18ND**，明显慢于正常训练，但每个 PPO 步骤在概率上更有价值。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1347201054789603328)** (9 messages🔥): 

> `Intermediate layer outputs to vocab space, Tuned Lens vs Logit Lens` 


- **探索将中间层输出投影到词表空间**：成员们讨论了将中间层输出投影到词表空间的技术，一位成员回忆起过去使用 **embed_out 矩阵** 投影中间层隐藏状态 (hidden states) 的论文。
   - 这种方法的问题在于 *模型没有动力使中间层隐藏状态能够通过该特定矩阵进行投影。*
- **Tuned Lens 改进 Logit Lens 技术**：一位成员分享了论文 [Tuned Lens: Iterative Refinement with Interpretable Differentiable Probes](https://arxiv.org/abs/2303.08112) 的链接，该论文从迭代推理的角度分析 Transformer，旨在理解模型预测是如何通过 **tuned lens** 逐层细化的。
   - Tuned lens 是早期 **logit lens** 技术的改进版，重现结果所需的[代码](https://github.com/AlignmentResearch/tuned-lens)可以在 GitHub 上找到。
- **尽管有 Tuned Lens，Logit Lens 仍在使用**：一位成员提到，尽管存在 **tuned lens**，大多数人仍然使用 **logit lens**。
   - 建议使用 **tuned lens** 代替 **logit lens**。



**提到的链接**：<a href="https://arxiv.org/abs/2303.08112">Eliciting Latent Predictions from Transformers with the Tuned Lens</a>：我们从迭代推理的角度分析 Transformer，旨在理解模型预测是如何逐层细化的。为此，我们为冻结的每个区块训练了一个仿射探针...

  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1346890865729212577)** (11 messages🔥): 

> `lm-eval AIME 支持, ARC-Challenge 任务, 分数差异, vllm 的实现` 


- **lm-eval 中 AIME 支持的需求出现**：一名成员询问如何在 **lm-eval** 中添加 **AIME 支持**。
   - 该问题被重定向到之前关于同一话题的[相关讨论](https://discord.com/channels/729741769192767510/1079865324087803985/1347284429743198301)。
- **使用 arc_challenge.yaml 配置的 ARC-Challenge 任务**：似乎有一名成员正在使用 `arc_challenge.yaml` 以 **25-shot** 配置运行 **ARC-Challenge 任务**。
   - 关于该特定设置没有透露更多细节。
- **DeepSeek-R1-Distill-Llama-8B 分数差异调查**：一名成员报告称，在 `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` 模型上使用 **vllm** 运行 `lm_eval` 时出现了巨大的分数差异，指出 **tp=4** 时分数为 **53.03**，**tp=2** 时为 **43.94**，而 **tp=1** 时为 **43.43**。
   - 使用的命令为 `lm_eval -m vllm -a deepseek-ai/DeepSeek-R1-Distill-Llama-8B,max_length=34000,tensor_parallel_size=x, -t gpqa_diamond_cot_zeroshot -b auto --apply-chat-template --log_samples --gen_kwargs temperature=0.6,top_p=0.95`。
- **vllm 实现的潜在问题浮现**：针对分数差异，另一名成员建议该问题可能源于 **vllm 的实现**。
   - 该成员提议如果有样本可以进行调查，原报告者表示稍后可以提供。


  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1346854838645817365)** (23 messages🔥): 

> `企业级部署, B2B 交付周期, 社区反馈` 


- **Cohere 企业级部署咨询**：用户 **brad062677** 询问如何联系 Cohere 的人员进行企业级部署咨询，并提到他们在一周前给支持团队发了邮件，希望通过 **Discord** 获得更快的回复。
- **AI 领域的 B2B 交付周期**：一名用户提到，企业咨询是通过直销处理的，B2B 交付周期可能较慢，在行业内可能长达 **六周**，而另一名用户则表示 AI 公司通常会在 **两到三天** 内回复。
   - Cohere 员工 **1vnzh** 为延迟道歉，并向用户保证他们正在内部讨论如何对接，用户今天会收到回复。
- **社区改进反馈**：一名用户征求关于如何改进社区的反馈，表示 *我们正努力让社区成为每个人更好的地方……还缺少什么？如果你不想公开讨论，欢迎私信*。


  

---

### **Cohere ▷ #[【📣】announcements](https://discord.com/channels/954421988141711382/996880279224451154/1346968241582506117)** (1 messages): 

> `Aya Vision, Multilingual AI, Multimodal Models, Open-Weights Model, AyaVisionBenchmark` 


- **Cohere 推出 Aya Vision，多语言领域的力作**：Cohere For AI 发布了 **Aya Vision**，这是一款 **8B 和 32B** 的开源权重多语言视觉研究模型，将能力扩展到了 **23 种语言**。
   - 它在图像描述（image captioning）、视觉问答（visual question answering）、文本生成以及文本和图像翻译方面表现出色；详情请参阅 [Cohere 的博客文章](https://cohere.com/blog/aya-vision)。
- **Aya Vision 登陆 Hugging Face 和 Kaggle！**：Aya Vision 模型现在可以在 [Hugging Face](https://huggingface.co/collections/CohereForAI/c4ai-aya-vision) 和 [Kaggle](https://www.kaggle.com/models/cohereforai/aya-vision) 上获取，为开发者和研究人员提供了更广泛的访问渠道。
   - 这使得社区能够更轻松地实验并基于这一最先进的多语言视觉模型进行构建。
- **Poe 平台接入 Aya Vision**：Aya Vision 已在 [Poe](https://poe.com/Aya-Vision) 上线，在该平台内提供先进的视觉语言能力。
   - 这是一个 **32B** 开源权重多模态模型，针对各种视觉语言用例进行了优化，并支持 **23 种语言** 的训练。
- **通过 WhatsApp 全球访问 Aya！**：用户现在可以从任何地方在 WhatsApp 上免费给 Aya 发消息，通过[此链接](https://cohere.com/research/aya/whatsapp)向其提问文本和视觉问题、进行图像描述，并将文本和图像翻译成自然语言。
   - Aya 支持 **23 种语言**，为自然语言理解、摘要和翻译任务中的语言处理奠定了基础。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://cohere.com/blog/aya-vision">Aya Vision: Expanding the worlds AI can see</a>: 介绍来自 Cohere For AI 的最先进开源权重视觉模型 Aya Vision。 </li><li><a href="https://www.kaggle.com/models/cohereforai/aya-vision">CohereForAI | Aya Vision | Kaggle</a>: C4AI Aya Vision 是 8B 和 32B 参数模型的开源权重研究版本，具有针对各种视觉语言用例（包括 OCR、图像描述、视觉重组等）优化的先进能力。</li><li><a href="https://poe.com/Aya-Vision">Aya-Vision - Poe</a>: Aya Vision 是一个 32B 开源权重多模态模型，具有针对各种视觉语言用例优化的先进能力。该模型经过训练，在视觉和文本的 23 种语言中表现出色。</li><li><a href="https://cohere.com/research/aya/whatsapp">Text Aya on WhatsApp | Cohere For AI</a>: Aya Expanse 支持 23 种语言，是世界上最好的多语言 AI。现在已在 WhatsApp 上线，可以用你的语言免费给 Aya 发消息。
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1346853704313208907)** (1 messages): 

> `Cohere Reranker v3.5 Latency` 


- **Cohere Reranker v3.5 延迟数据仍未公布**：一位成员询问了 **Cohere Reranker v3.5** 的延迟数据，并指出虽然在 [Pinecone 访谈](https://www.pinecone.io/learn/cohere-rerank/)中提到过分享图表或数据的意向，但目前尚未实现。
   - 在尝试搜索延迟数据后，未发现任何结果。
- **社区等待 Cohere Reranker v3.5 延迟数据**：尽管在 Pinecone 访谈中设定了初步预期，但 **Cohere Reranker v3.5** 的具体延迟数据或图表仍未发布。
   - 数据的缺失导致一些人正在积极寻求这些信息，以便进行性能评估和对比。


  

---


### **Cohere ▷ #[「💡」projects](https://discord.com/channels/954421988141711382/1218409701339828245/1347060801055490048)** (1 messages): 

> `Mindmap Generation, Pretrained Models, Mathematical Models` 


- **学生寻求思维导图项目指导**：一名学生正在开发一个根据章节内容生成思维导图的网站，旨在构建主题和子主题的分层结构。
   - 他们正在考虑最初使用预训练模型，随后创建一个自定义数学模型，并寻求如何推进的建议。
- **在预训练模型和数学模型之间做出选择**：该学生不确定是应该从预训练模型开始，还是从用于生成思维导图的自定义数学模型开始。
   - 他们正在寻求关于将这两种方法整合到项目中的最佳方案建议。


  

---

### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1346962182549798942)** (2 条消息): 

> `Introductions, Sales Contact` 


- **新成员寻求销售联系**：一位新成员正尝试联系 **sales / enterprises support** 团队的人员。
   - 他们在 introductions 频道留了言，但目前还没有人回应。
- **置顶介绍消息欢迎新成员**：欢迎消息鼓励成员介绍自己，包括 **Company/Industry/University**、**What you're working on**、**Favorite tech/tools you use** 以及 **What you hope to gain from this community** 等细节。
   - 目前还没有成员在该频道提供介绍信息。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1346813249366003723)** (9 条消息🔥): 

> `ShapeTracker merging proof, 96GB 4090 on Taobao, Rust CubeCL` 


- **ShapeTracker 合并证明即将完成**：一位成员宣布在 [repo](https://github.com/Nielius/Tensorlayouts) 中基本完成了用于合并 **ShapeTrackers** 的 Lean 证明，更多背景信息可在 [this issue](https://github.com/tinygrad/tinygrad/issues/8511#issuecomment-2700706082) 中找到。
   - 该证明尚未考虑 offsets 和 masks，但该成员认为，尽管需要*大量*工作，将证明扩展到包含这些因素应该是直接的。
- **淘宝上发现新款 96GB 4090**：一位成员分享了淘宝上售卖 **96GB 4090** 的链接（[X 帖子](https://x.com/yomix1337/status/1893692548108984391?s=46)）。
   - 另一位成员澄清说 *这不是淘宝，而且目前还买不到*，并且 *还需要几个月的时间才能购买*。
- **对 Rust CubeCL 的好奇**：一位成员询问了 **Rust CubeCL** 的质量，并指出它是由 **Rust Burn** 的原班人马创建的。
   - 该成员 *想知道 Rust CubeCL 是否好用*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/yomix1337/status/1893692548108984391?s=46">Gene edited Yostuba (@Yomix1337) 的推文</a>：@EsotericCofe 将在五月后发布</li><li><a href="https://github.com/Nielius/Tensorlayouts">GitHub - Nielius/Tensorlayouts: 合并两个 tensor view 的必要充分条件的 Lean 证明</a>：合并两个 tensor view 的必要充分条件的 Lean 证明 - Nielius/Tensorlayouts</li><li><a href="https://github.com/tinygrad/tinygrad/issues/8511#issuecomment-2700706082">关于任意 ShapeTrackers 的可合并性 · Issue #8511 · tinygrad/tinygrad</a>：嘿，我想提出一种关于 view 合并问题的新表述和证明，我还没见过有人提到过。我见过一位叫 @Nielius 的人提出的表述，但遗憾的是它...
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1347241913601884250)** (3 条消息): 

> `RANGE Op, iGPU detection on Linux` 


- **关于 RANGE Op 操作的疑问**：一位成员询问了 `RANGE` Op 的操作，注意到它在 `arrange` 的 `Tensor` 实现中似乎不存在。
   - 该成员随后意识到它 *“不是一个 range”* 并对造成的困惑表示歉意。
- **询问 tinygrad 在 Linux 上的 iGPU 自动检测**：一位成员询问默认设备初始化或 `Device.get_available_devices()` 是否应该在 Linux 上自动检测到 **iGPU**。
   - 附带的图片显示为 *“Device: [CPU]”*，而该成员似乎期望显示为 *“Device: [GPU]”*。


  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1346864089363779714)** (7 messages): 

> `HF Checkpoints, special_tokens.json, TorchTune Checkpointer, Github Stars` 


- **Special Tokens 的特殊处理**：一位成员发现 **TorchTune checkpointer** 会从 Hugging Face 复制原始的 **special_tokens.json**，而不是可能经过修改的自定义版本，并指出了相关的 [代码位置](https://github.com/pytorch/torchtune/blob/80da6a5dae23a201595d07041c12ffde830332d7/torchtune/training/checkpointing/_checkpointer.py#L892-L896)。
   - 有用户建议，一个快速的解决方法是手动用自定义版本替换下载的 **special_tokens.json**。
- **Checkpointer 自定义考量**：讨论了通过向 checkpointer 的 `save_checkpoint` 方法传递新参数来支持 **自定义 special_tokens.json** 的用例。
   - 然而，团队决定在没有充分理由的情况下不暴露新参数，因此目前的建议是手动复制文件。
- **TorchTune GitHub Stars 突破 5,000 大关！**：Torchtune 项目达成了一个里程碑，在 **GitHub 上获得了 5,000 颗星**。
   - 社区庆祝了这一成就。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/blob/80da6a5dae23a201595d07">GitHub - pytorch/torchtune at 80da6a5dae23a201595d07041c12ffde830332d7</a>: PyTorch native post-training library. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/torchtune/blob/80da6a5dae23a201595d07041c12ffde830332d7/torchtune/training/checkpointing/_checkpointer.py#L892-L896.">torchtune/torchtune/training/checkpointing/_checkpointer.py at 80da6a5dae23a201595d07041c12ffde830332d7 · pytorch/torchtune</a>: PyTorch native post-training library. Contribute to pytorch/torchtune development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1347262585464029184)** (3 messages): 

> `GRPO recipe, Memory issues, Excessive torch.cuda.empty_cache()` 


- **GRPO Recipe 中充斥着 Empty Cache 调用**：一位成员询问在 **GRPO recipe** 中过度使用 `torch.cuda.empty_cache()` 调用的情况。
   - 另一位成员承认，其中许多调用可能是多余的，这源于早期开发时面临的 **Memory issues**（内存问题）。
- **等待审核的 GRPO PR**：两个 **GRPO PR**（特别是 **#2422** 和 **#2425**）已经开启两周，正在等待审核。
   - 一位成员请求协助审核这些 PR，希望有人能帮忙清理积压的队列。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1346810951265157201)** (4 messages): 

> `Berkeley vs MOOC lectures, Certificate Declaration Forms` 


- **MOOC 学生与 Berkeley 学生参加相同的课程**：一位成员询问 Berkeley 学生是否有 MOOC 学生没有的课程，另一位成员澄清说 **Berkeley 学生和 MOOC 学生参加的是相同的课程**。
- **证书颁发问题**：一位成员提到在 12 月提交了证书申报表，但被告知没有提交记录。


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1346806842453524540)** (2 messages): 

> `AST Metric Definition, V1 Dataset Construction` 


- **AST 指标：函数调用格式化率**：一位成员询问 **AST 指标** 是否代表生成正确格式函数调用的 LLM 响应百分比。
   - 未收到回复。
- **V1 数据集：构建**：一位成员询问 **V1 数据集** 是如何构建的。
   - 未收到回复。


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1347296310994403410)** (1 messages): 

> `Gemini 2, GPT o3-high, Deepseek R1, Prompt tool calling, Python tool` 


- **关于 Prompt Tool Calling 最佳模型的讨论**：一位用户询问在 **Gemini 2**、**GPT o3-high** 和 **Deepseek R1** 中，哪个模型最适合进行 Prompt Tool Calling，特别是调用 **Python tool**。
- **用于 Python 工具集成的模型选择**：该用户正在评估 **Gemini 2**、**GPT o3-high** 和 **Deepseek R1**，以确定哪一个最适合根据 Prompt 调用 **Python tool**。


  

---

### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1347217481147744317)** (1 条消息): 

> `Jamba 1.6 launch, Open model for enterprise deployment, Jamba 1.6 performance benchmarks, Hybrid SSM-Transformer architecture, 256K context window` 


- **AI21 Labs 发布 Jamba 1.6**: AI21 Labs 推出了 **Jamba 1.6**，这是一款专为私有企业部署定制的 Open model，模型权重已在 [Hugging Face](https://huggingface.co/ai21labs) 上发布。
   - 该公司声称它 *提供了无与伦比的速度和性能*，在不牺牲效率、安全性和数据隐私的情况下，为企业级 AI 树立了新标杆。
- **Jamba 1.6 展示 Arena 实力**: 据 [AI21 的公告](https://www.ai21.com/jamba/)，**Jamba 1.6** 在 Arena Hard 基准测试中表现优于 **Cohere**、**Mistral** 和 **Llama**，足以媲美领先的闭源模型。
   - 此次发布强调了其对完全私有化的 on-prem 或 VPC 部署的适用性，拥有极快的延迟和市场领先的 **256K context window**。
- **混合架构赋予 Jamba 1.6 优势**: **AI21 Jamba** 系列采用混合 **SSM-Transformer** 基础模型，在质量和速度方面均表现出色，这归功于其创新的 **Mamba-Transformer MoE architecture**，旨在提高成本效益，详见 [Jamba 1.6 博客文章](https://www.ai21.com/jamba/)。
   - 该模型可以部署在任何地方，支持 self-hosted 或在 AI21 SaaS 中运行，以满足多样化的数据安全需求。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/ai21labs">ai21labs (AI21)</a>: 未找到描述</li><li><a href="https://www.ai21.com/jamba/">Jamba 1.6: The Best Open Model for Enterprise Deployment</a>: 探索 AI21 的 Jamba —— 一款为准确性、效率和强大的文本生成而构建的前沿、长上下文 AI Open model。
</li>
</ul>

</div>
  

---


{% else %}


> 完整的各频道详情已针对电子邮件进行了删减。 
> 
> 如果您想查看完整详情，请访问此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}