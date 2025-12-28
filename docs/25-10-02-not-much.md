---
companies:
- openai
- google
- ibm
- alibaba
- kling_ai
- synthesia
- ollama
- huggingface
- arena
- artificialanalysis
- tinker
- scaling01
date: '2025-10-02T05:44:39.731046Z'
description: '以下是为您翻译的中文内容：


  **Kling 2.5 Turbo** 在文生视频和图生视频生成领域处于领先地位，且价格极具竞争力。**OpenAI Sora 2** 展现了强大的指令遵循能力，但在物理规律的一致性方面仍存在不足。**Google
  Gemini 2.5 Flash** “Nano Banana” 图像生成功能现已全面开放，支持多图融合和灵活的宽高比。**IBM Granite 4.0**
  引入了 Mamba/Transformer 混合架构，具备大上下文窗口和极高的 Token 效率，在智能指数（Intelligence Index）上超越了部分同类模型。**Qwen（通义千问）**
  模型迎来了更新，包括支持微调 API 和提升视觉能力。**Tinker** 提供灵活的微调 API，支持 LoRA 共享和仅限 CPU 的训练循环。生态系统方面也有所更新，例如
  **Synthesia 3.0** 新增了视频智能体（video agents）功能。'
id: MjAyNS0x
models:
- kling-2.5-turbo
- sora-2
- gemini-2.5-flash
- granite-4.0
- qwen-3
- qwen-image-2509
- qwen3-vl-235b
people:
- artificialanlys
- kling_ai
- altryne
- teortaxestex
- fofrai
- tim_dettmers
- sundarpichai
- officiallogank
- andrew_n_carr
- googleaidevs
- clementdelangue
- wzhao_nlp
- alibaba_qwen
- scaling01
- ollama
title: 今天没发生什么事。
topics:
- video-generation
- instruction-following
- physics-simulation
- image-generation
- model-architecture
- mixture-of-experts
- context-windows
- token-efficiency
- fine-tuning
- lora
- cpu-training
- model-benchmarking
- api
- workflow-automation
---

**平静的一天**

> 2025年10月1日至10月2日的 AI 新闻。我们为您查阅了 12 个 subreddit、544 个 Twitter 账号和 23 个 Discord 社区（196 个频道，8860 条消息）。预计节省阅读时间（按 200wpm 计算）：629 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 方式呈现所有往期内容。访问 https://news.smol.ai/ 查看完整的详细新闻，并在 @smol_ai 上向我们反馈！

今天是平静的一天，所以你可以听听最新的 [Latent Space with Dylan Field](https://www.latent.space/p/figma) 播客！

此外，[首届 AI Engineer Code Summit](https://apply.ai.engineer/) 的邀请函已开始发放。

---

# AI Twitter 回顾

**视频生成：Sora 2、Kling 2.5 Turbo 以及 Google 的 “Nano Banana” GA**

- **Kling 2.5 Turbo (Text/Image→Video)**：Kling 的最新模型在 Artificial Analysis Video Arena 的文生视频和图生视频测试中均位列榜首，领先于 Hailuo 02 Pro、Google 的 Veo 3 和 Luma Ray 3。它可生成高达 1080p 分辨率的 5秒/10秒 剪辑。显著的经济性：FAL API 上的价格约为 $4.20/分钟，而 Hailuo 02 Pro 为 $4.90，Seedance 1.0 约为 $7.32；通过 Kling 的 Ultra 计划使用应用积分，每段视频约 15 美分。查看 [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1973570493753204953) 的 Arena 线程中的模型对比和定价，以及 Kling 的公告 [@Kling_ai](https://twitter.com/Kling_ai/status/1973581864679121374)。
- **OpenAI Sora 2：能力 vs. 正确性**：实际使用显示出令人印象深刻的指令遵循（instruction-following）和应用内重混（remixing）能力，但批评性评估指出其存在物理规律不一致和营销润色痕迹。查看 [@altryne](https://twitter.com/altryne/status/1973568567489798144) 的广泛演示汇总，[@teortaxesTex](https://twitter.com/teortaxesTex/status/1973570902609805711) 对“讨好感”胜过物理保真度的批评，以及 [@fofrAI](https://twitter.com/fofrAI/status/1973745038195830891) 的针对性测试（Sora 2 在 Veo 3 处理得更好的物理场景中失败，音频叙述正确），此外还有 [@Tim_Dettmers](https://twitter.com/Tim_Dettmers/status/1973728079395856396) 的冷静综述。
- **Google Gemini 2.5 Flash Image (“Nano Banana”) GA**：现已进入生产就绪阶段，支持 10 种长宽比、多图融合和仅图像输出。定价：Gemini API (AI Studio + Vertex) 上每张图像 $0.039。来自 [@sundarpichai](https://twitter.com/sundarpichai/status/1973788714758517147)、[@GoogleAIStudio](https://twitter.com/GoogleAIStudio/status/1973836478989152700) 和 [@OfficialLoganK](https://twitter.com/OfficialLoganK/status/1973790388722061394) 的公告。该模型还集成到了合作伙伴的产品中（例如 Cartwheel 的新动作流水线）[@andrew_n_carr](https://twitter.com/andrew_n_carr/status/1973875941651964335)，并由 Google 开发者账号展示 [@googleaidevs](https://twitter.com/googleaidevs/status/1973781293977735435)。
- **生态系统**：Synthesia 3.0 增加了 “video agents” 和新工作流 [@synthesiaIO](https://twitter.com/synthesiaIO/status/1973688529818620193)。

**开放权重模型发布：IBM Granite 4.0 和 Qwen 更新**

- **IBM Granite 4.0 (Apache 2.0, hybrid Mamba/Transformer)**：IBM 的新系列模型混合了少数标准 Attention 层和多数 Mamba 层，旨在不大幅损失准确性的情况下降低内存占用。尺寸包括 Granite 4.0 H Small (MoE 32B/9B 激活)、H Tiny (7B/1B)、H Micro (3B/3B) 以及一个 3B dense Micro 变体。关键规格：128K 上下文，Apache 2.0 协议，极高的 Token 效率。Artificial Analysis 在其智能指数（非推理）中将 H Small 评为 23 分，领先于 Gemma 3 27B (22)，落后于 Mistral Small 3.2 (29)、EXAONE 4.0 32B (30) 和 Qwen3 30B A3B (37)。Micro 评分为 16，略高于 Gemma 3 4B (15)。Granite 已在 HuggingFace 和 Replicate 上线（H Small 每 1M 输入/输出 Token 价格为 $0.06/$0.25）。基准测试详见：[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1973746432692936963)。Ollama 发布了 Micro/Micro-H/Tiny-H/Small-H 的可运行镜像 [@ollama](https://twitter.com/ollama/status/1973782095811219574)。IBM Granite 也已加入 LM Arena [@arena](https://twitter.com/arena/status/1973892502458650697)，HF 的 [@ClementDelangue](https://twitter.com/ClementDelangue/status/1973798540389355903) 强调了浏览器/WebGPU 演示和 HF Enterprise 的入驻。
- **Qwen 更新**：Qwen 模型是首批受 Tinker 微调 API 支持的模型之一 [@wzhao_nlp](https://twitter.com/wzhao_nlp/status/1973603599616974970)，Qwen 团队指出了扩展的支持和开放发布 [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1973665010615218421)。Qwen-Image-2509 提高了连贯性 [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1973668568412856595)；据报道 Qwen3 VL 235B 在某些视觉任务中表现出色且成本更低 [@scaling01](https://twitter.com/scaling01/status/1973777774121984175)。

**微调与系统：Tinker、rank-1 LoRA、MoE 支持以及推理加速**

- **Tinker：一个支持 LoRA 共享的灵活微调 API**：Thinking Machines 的 Tinker 允许你编写仅限 CPU 的训练循环，并能不加改动地在分布式 GPU 上运行，在 Tinker 负责调度、资源分配和故障处理的同时，保持对算法/损失函数的控制。它支持开源模型（Llama, Qwen），包括大型 MoE（例如 Qwen3-235B），并实现了 LoRA 以实现高效的资源共享。摘要：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1973827605448306883)，发布日志 [@Smol_AI](https://twitter.com/Smol_AI/status/1973622595124863044)，教程/文档：[链接](https://twitter.com/TheTuringPost/status/1973827618442260655)。
- **无憾 LoRA (rank=1)**：多次复现表明，rank-1 的 LoRA 在推理任务上可以达到全量微调的质量，同时节省约 43% 的 VRAM，从而支持在更大模型上进行 RL；查看结果和代码 [@zzlccc](https://twitter.com/zzlccc/status/1973612326747336767) 以及关于 Qwen3-0.6B OpenR1-Math 的 Colab [@ben_burtenshaw](https://twitter.com/ben_burtenshaw/status/1973776491843297386)。参阅来自 “LoRA Without Regret” 的指南 [@TheTuringPost](https://twitter.com/TheTuringPost/status/1973820885116334441)。
- **MoE 训练与基础设施**：Prime-RL 现在支持用于 RL 和 SFT 的 MoE（Qwen3 A3-30B, GLM 系列, Moonlight），通过大量的模型重写来保持与 Torch Compile 的兼容性，同时保留 HF 生态系统的兼容性 [@samsja19](https://twitter.com/samsja19/status/1973624615768674612)。在推理方面，[@vikhyatk](https://twitter.com/vikhyatk/status/1973884858574491819) 报告了一个新的引擎，其生成速度提升了 1.3–20 倍；生产环境使用 QAT 进行 FP8 KV 缓存和 MoE 权重的量化（引擎目前为私有）。对于本地/开发基础设施：MI300X 虚拟机按需租用价格为 $1.99/GPU/小时 [@HotAisle](https://twitter.com/HotAisle/status/1973768786965639643)，vLLM 现已支持 BERT [@vllm_project](https://twitter.com/vllm_project/status/1973805307878142297)。

**RL 与推理：训练内搜索、扩展探索、潜空间 CoT、前置推理**

- **训练时搜索与高效探索**：DeepSearch 将 MCTS 引入训练循环，配合 Tree‑GRPO 稳定性增强以及高效的缓存/过滤，在 AIME/AMC 上达到 62.95% 的准确率，仅消耗约 330 GPU 小时（击败了 Nemotron 基准，并超越了即使在 1800+ GPU 小时下也会陷入瓶颈的标准 RL）[@omarsar0](https://twitter.com/omarsar0/status/1973781658772951320)。BroRL 通过将每个样本的 rollout 数量增加到数百个来扩展探索，克服了仅扩展训练步数时出现的饱和现象 [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1973717761693217241)。
- **架构与训练机制**：一种新的潜空间 CoT 方法 “thoughtbubbles” 插入了输入自适应的潜空间 token，以便在没有 CoT 标签的情况下分配更多计算资源，改善了困惑度（perplexity）和计算利用率 [@houjun_liu](https://twitter.com/houjun_liu/status/1973778517427937323)，并获得了积极反响 [@khoomeik](https://twitter.com/khoomeik/status/1973785079932727760)。NVIDIA 的 “前置推理”（Front‑Loading Reasoning）发现，在预训练期间注入推理能力会产生微调无法弥补的持久增益 [@__SyedaAkter](https://twitter.com/__SyedaAkter/status/1973841632249172096)。一个虽小但影响巨大的 MoE 改进——全局批次负载均衡（对比微批次）——通过极少的代码改动实现了更低的困惑度和更清晰的专家专业化 [@daddyofadoggy](https://twitter.com/daddyofadoggy/status/1973759113554174251)。对于稀疏扩散 LM，OpenMoE 2 研究了跨越广泛 FLOPs/参数范围的专家选择 MoE × 扩散，声称实现了完美的负载均衡（无需辅助损失）、提升了 20% 的吞吐量，并在多轮训练下实现了自适应计算 [@NiJinjie](https://twitter.com/NiJinjie/status/1973747616082186349)。

**Agent 与工具链：CLI + 语义搜索、Notebook MCP、浏览器和 CLI**

- **CLI Agent + 语义搜索优于纯 CLI**：LlamaIndex 的 SemTools 基准测试（1,000 篇 arXiv 论文）显示，与仅使用 CLI 工具的 Agent 相比，结合语义搜索的 Agent 在各种问题类型中能产生更完整的答案；Unix 工具仍然是一个强大的基准，SemTools 将解析（LlamaParse）和语义搜索直接集成到命令行 Agent（Claude/Gemini CLI）中。结果/方法论：[@llama_index](https://twitter.com/llama_index/status/1973783798044307741)。
- **通过 MCP 执行 notebook**：Goodfire 开源了 Scribe，这是一个基于 MCP 的系统，允许 Agent 运行 notebook 单元格并接收 Jupyter 输出（文本/错误/图像）。他们分享了关于“实验型 Agent”与“软件开发 Agent”的经验，以及科学工作流所需的脚手架 [@GoodfireAI](https://twitter.com/GoodfireAI/status/1973789154174754877)，[博客](https://twitter.com/GoodfireAI/status/1973789166019482035)。
- **“AI 浏览器”与评估器**：Perplexity 的 Comet 现已全球正式发布（GA），Comet Plus 随重大出版商合作伙伴关系一同推出；Pro/Max 用户可获赠 Plus 套餐 [@perplexity_ai](https://twitter.com/perplexity_ai/status/1973795224960032857)，[@AravSrinivas](https://twitter.com/AravSrinivas/status/1973804332039786608)。Yupp 的 “Help Me Choose” 编排了第三个模型来评判两个候选答案，然后让它们互相分析，最后由用户选择——这是一种有趣的裁决模式 [@yupp_ai](https://twitter.com/yupp_ai/status/1973882910907470237)，[@lintool](https://twitter.com/lintool/status/1973874173157257485)。Google 的 Jules Tools 带来了具有 Agent 能力的 CLI（可通过 npm 安装），镜像了浏览器的功能 [@julesagent](https://twitter.com/julesagent/status/1973812188977508755)。

**排行榜与真实世界编程 Agent 指标**

- **Claude Sonnet 4.5 在 LM Arena 并列第一**：Sonnet 4.5 与 Claude Opus 4.1 并列榜首，在包括编程和创意写作在内的各个类别中表现强劲（排名来自数万次人类投票）[@arena](https://twitter.com/arena/status/1973828836510085385)。社区报告显示 Anthropic 继续交付极具竞争力的编程模型 [@scaling01](https://twitter.com/scaling01/status/1973836516205134135)。
- **开源模型在代码编辑 Agent 领域正在缩小差距**：在 Cline 的 diff-edit 成功率测试中，GLM-4.6 达到了 94.9%，而 Claude 4.5 为 96.2%，且成本仅为后者的约 10%；用户报告称已据此切换工作流 [@cline](https://twitter.com/cline/status/1973870619013136850)，[@nickbaumann_](https://twitter.com/nickbaumann_/status/1973846157886697771)。
- **Video Arena 提醒**：Kling 2.5 Turbo 在 T2V 和 I2V 领域均处于领先地位；详情见上文视频部分 [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1973570493753204953)。

**热门推文（按互动量排序）**

- [“在很多方面，我们简直就是预训练模型。”](https://twitter.com/cloneofsimo/status/1973655922506605046) —— [@cloneofsimo](https://twitter.com/cloneofsimo) — 4,967
- [Perplexity Comet 全球正式发布 (GA)](https://twitter.com/perplexity_ai/status/1973795224960032857) —— [@perplexity_ai](https://twitter.com/perplexity_ai) — 2,667
- [Anthropic 的“思考”活动获得赞誉并被广泛采用](https://twitter.com/signulll/status/1973828026761695439) —— [@signulll](https://twitter.com/signulll) — 2,441
- [Nano Banana 正式发布 (GA) 公告](https://twitter.com/sundarpichai/status/1973788714758517147) —— [@sundarpichai](https://twitter.com/sundarpichai) — 1,576
- [“迭代速度是一种超能力”](https://twitter.com/gdb/status/1973864268350255366) —— [@gdb](https://twitter.com/gdb) — 1,989

---

# AI Reddit 摘要

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Sora 2 与 WAN 2.2 视频生成演示

- [**Sora 2 在脱口秀表演方面表现惊人**](https://www.reddit.com/r/ChatGPT/comments/1nwaowu/sora_2_is_insanely_good_at_stand_up_comedy/) (活跃度: 437): **该帖子声称一段脱口秀剪辑是由“Sora 2”生成的，推测是指 OpenAI 的 Sora 文本生成视频模型（[概览](https://openai.com/sora)）。观众反馈其喜剧节奏和面部表情同步非常自然，这意味着强大的时间连贯性（temporal coherence）、音素-视素对齐（phoneme–viseme alignment）以及精细的手势/微表情控制；然而，链接的视频无法访问（**`HTTP 403`**），因此无法从帖子中验证其来源、模型版本（“2”）、提示词（prompts）、种子（seeds）或生成参数。** 评论者压倒性地赞扬其真实感——“不可思议”的节奏和自然的表达——有些人甚至将其与人类喜剧演员相媲美，而至少有一人询问这是否真的出自 Sora，凸显了由于缺乏证据或技术细节而产生的怀疑。
    - 多位用户强调了表达与面部表情之间“不可思议”的同步，暗示了强大的视听韵律对齐（audiovisual prosody alignment）和关键帧级的手势/对口型能力。如果这是原生的 Sora 2 输出，则表明与之前的文本生成视频基准相比，其时间调节（temporal conditioning，如随节拍对齐的微表情、头部/眉毛提示）和演员般的姿态控制有所改进。
    - 一位评论者指出，这个笑话并非原创，并引用原文将其归功于 **Joan Rivers**，这引发了人们对模型是从训练数据或提示词来源材料中记忆/反刍（memorization/regurgitation）而非进行新颖合成的担忧。这指向了生成式视频模型中的内容溯源和原创性风险；参见归属：https://www.imdb.com/name/nm0001672/quotes/ 。
    - 对这是否“真的来自 Sora”的怀疑标志着 AI 生成剪辑的验证/溯源问题（可能存在剪辑、配音或工作流混合）。技术读者可能会寻找可复现性细节（提示词、种子、运行时间）、元数据/水印或内容凭证（Content Credentials），以验证生成链并排除后期制作增强的可能性。
- [**WAN 2.2 Animate - 角色替换测试**](https://www.reddit.com/r/StableDiffusion/comments/1nvvo7g/wan_22_animate_character_replacement_test/) (活跃度: 1439): **楼主展示了使用 WAN 2.2 Animate 对电影《第九道门》（[The Ninth Gate](https://en.wikipedia.org/wiki/The_Ninth_Gate)）片段进行的角色替换测试，实现了令人信服的身份替换，同时指出由于参考图像仅覆盖头部/上身，导致服装不一致（表明服装连贯性取决于调节覆盖范围）。分享的视频链接是一个 [Reddit 托管地址](https://v.redd.it/e2hf1vuf0nsf1)，在外部获取尝试中返回了** `HTTP 403` **（可能需要登录）。** 评论者强调，虽然渲染风格/质量一般，但集成/替换效果“绝对令人惊叹”。技术批评指出光照不匹配以及在区域较小时手部保真度较弱，还有人询问如何使用 WAN 2.2 Animate 制作长序列；总体观点是，这是 AI 驱动的 VFX 潜力的有力展示。
    - 评论者指出，尽管渲染/风格保真度适中，但核心角色集成/替换非常稳定——追踪和对齐表现良好——这表明 **WAN 2.2 Animate** 即使在缺乏审美润色时，对于特效风格的角色替换也是可行的。
    - 技术批评集中在光照和细节保真度上：有人说 *“光照太烂了！”*，另一人指出第一组镜头中的手部 *“在屏幕上太小，无法正确生成/追踪”*，反映了微小特征丢失细节或追踪鲁棒性不足的常见失效模式。
    - 用户对确切的工作流（流水线和剪辑长度处理方法）有需求。一个具体的建议是使用 **relight LoRA** 来修复照明不匹配；其他人则询问视频是如何延长的，表明了对在保持时间连贯性的同时延长序列的技术感兴趣。

### 2. OpenAI 5000 亿美元估值 + ChatGPT 'Think Longer' 用户体验 + 硅谷远见

- [**OpenAI 估值飙升至 5000 亿美元，超越马斯克的 SpaceX**](https://www.reddit.com/r/OpenAI/comments/1nw36bw/openai_valuation_soars_to_500_billion_topping/) (热度: 720): **帖子声称 OpenAI 的私募估值已达到约** `$500B`**，超过了 SpaceX。评论者引用了** `2025` **年的预测数据：营收约** `$4.3B` **对比亏损约** `$6.8B`**——这意味着极高的营收倍数和深度负值的营业利润率。提出的技术担忧包括感知到的模型质量退化（例如，“GPT 正在变差”）以及随着来自闭源和开源模型竞争压力的加剧，企业界对 AI 的“现实审视”。随附的梗图/图片强调了对可持续性的怀疑 ([image](https://preview.redd.it/5phkrh1xfpsf1.jpeg?width=1270&format=pjpg&auto=webp&s=44308b7a5984ab2d905bc1684742a927ae0aa0c0))。** 热门评论认为，鉴于负面的单位经济效益和拥挤的竞争环境，这一估值是一个泡沫，并认为许多 AI 厂商可能无法生存。其他人则附和称，目前的系统表现不及预期，理由是模型退化和未满足的企业用例。
    - 财务/估值担忧：评论者引用了约 `$4.3B` 的 2025 年营收对比约 `$6.8B` 的亏损，以及约 `$500B` 的估值，这意味着对于一个计算密集型业务来说，其远期市销率（forward sales）超过 `100x` 且利润率为深度负值。这引发了关于补贴推理的可持续性、未来涨价或为了在不损害产品质量的情况下证明估值倍数合理而需要的成本削减（如模型蒸馏、Batching、定制芯片）的疑问。
    - 模型可靠性/退化：关于 GPT “退化”的报告与已知的行为漂移（behavior drift）问题有关，即模型更新会随时间改变输出和质量。之前的分析发现 GPT-4 的推理/准确性存在显著的逐月差异（例如，斯坦福/加州大学伯克利分校的“ChatGPT 的行为如何随时间变化？”显示其在编程/数学任务上存在波动：https://arxiv.org/abs/2307.09009），强调了生产部署面临的维护/评估挑战。
    - 竞争压力：该帖子指出免费和付费替代方案正在缩小差距，这可能会压缩定价权。LMSYS Chatbot Arena 等公开评估显示，非 OpenAI 的领先者（如 **Claude 3.5 Sonnet**、**Gemini 1.5 Pro**、**Llama 3 70B**、**Mistral Large**）聚集在顶部（https://lmsys.org/blog/2024-06-20-arena-hard/），表明前沿能力可能出现商品化趋势，护城河假设正在减弱。
- [**能不能给这个功能加个禁用开关**](https://www.reddit.com/r/ChatGPT/comments/1nw4ttx/can_we_please_have_a_disable_function_on_this/) (热度: 1478): **用户请求增加一个开关来禁用聊天界面的“Thinking longer for a better answer”（为了更好的回答思考更久）行为/叠加层，报告称即使不在“Think Longer”模式下，每次提示也会触发——这暗示了 UX 问题或配置错误。评论指出存在现有的“Instant”设置，并且对于“思考”模型，你可以手动在“standard”和“extended”思考之间选择，这意味着该功能是可配置的，但可能令人困惑或应用不一致。** 评论者分为两派：一派开玩笑说用户没耐心，另一派则提供实用建议，指出 instant/standard/extended 控制选项已经存在；该帖子含蓄地辩论了这是一个 UX Bug 还是用户对设置的认知问题。
    - 现有的 UI 控制已经允许用户调整或避免较慢的刻意推理（deliberate reasoning）：一位评论者问道：*“你不知道有 'instant' 设置吗？如果你选择 'thinking' 模型，你可以手动在 'standard' 和 'extended' 思考之间选择。”* 这意味着存在一种可配置的延迟/质量权衡，其中 `instant` 最小化延迟，`standard` 平衡速度和推理，而 `extended` 在较高延迟下最大化深度。
    - 一位高级用户报告称默认使用思考模式，甚至在桌面端选择 `extended` 选项，将更快的模式留给琐碎的查询：*“几乎所有提示都默认使用思考模式……在桌面端甚至选择 'extended' 思考选项。”* 这强化了一种工作流模式：复杂的任务受益于更长的刻意运行，而简单的简单事实查询则更适合低延迟模式。

- [**兄弟，美剧《硅谷》（Silicon Valley）是怎么做到一直领先时代 10 年的？**](https://www.reddit.com/r/ChatGPT/comments/1nw0eo1/bro_how_was_the_show_silicon_valley_so/) (热度: 8183): **该帖子探讨了为什么 HBO 的《硅谷》（Silicon Valley）感觉领先于现实十年；高赞回复将其准确性归功于编剧团队聘请了真正的工程师和技术顾问，这使得对初创公司动态、基础设施权衡以及压缩算法研究的描绘非常写实。作为一个具体的例子，评论者指出第一季结尾中经过数学严密推导的优化方案（见此片段：https://www.youtube.com/watch?v=Tx3wDTzqDTs），证明了其严谨性超越了典型的处境喜剧。注：引用的 [v.redd.it](http://v.redd.it/) 资源在未经身份验证的情况下会返回** `403 Forbidden` **——访问需要登录会话或授权的 Reddit API 客户端。** 资深从业者将该剧描述为一部实质上的“纪录片”，认为其预见性源于在创作过程中引入了真正的技术人员，而不是依赖通用的科技套路。
    - 技术真实性可能源于聘请真正的工程师担任编剧/顾问，这有助于在剧情中植入真实的失败模式（扩展瓶颈、部署事故、VC/IP 限制）以及准确的术语/工具，而非通用的“黑客”套路。这种领域知识的输入让编剧能够合乎逻辑地推演近期的 ML/infra 趋势（而不是科幻式的飞跃），使故事情节感觉是*即将发生*的，而非纯属臆测。
    - “热狗 / 非热狗”（Hot Dog / Not Hot Dog）的笑话对应于二元分类，这可以追溯到 **perceptron** (Rosenblatt, `1957`) —— 一种具有明确局限性的线性分类器，由 **Minsky & Papert** 在 `1969` 年正式化 ([Perceptron](https://en.wikipedia.org/wiki/Perceptron), [Perceptrons](https://en.wikipedia.org/wiki/Perceptrons_(book)))。一个真实的基于图像的 Not-Hotdog 应用通常会依赖于通过 backprop（`1986` 年普及）训练的多层网络（如 **CNNs**），以学习非线性决策边界和视觉特征 ([CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network), [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation))。从概念上讲，这是相同的任务——二元分类——但从单层 perceptron 到现代深度网络的实现跨越是巨大的（数据规模、算力和模型容量）。

### 3. AI 喜剧帖子：“最奇怪的跳蚤市场”第 7 部分及相关短剧

- [**你在“最奇怪的跳蚤市场”卖什么？第 7 部分**](https://www.reddit.com/r/aivideo/comments/1nvr8hw/what_do_you_sell_at_the_strangest_flea_market_pt_7/) (热度: 477): **这是一个系列创意/喜剧帖子（“你在‘最奇怪的跳蚤市场’卖什么？第 7 部分”），带有一个 Reddit 托管的视频链接** `v.redd.it/x8rnhfkoulsf1` **，该链接返回 HTTP** `403 Forbidden` **和 Reddit 网络安全拦截页面，要求登录/OAuth，这表明存在应用层访问控制（会话/Cookie 或 OAuth 门控）以及可能的 CDN/机器人防护。评论提到了一些连贯的笑话（猪拉丁语片段；一个“说韩语的蔬菜”），但在没有身份验证会话或 API 令牌的情况下，主要媒体资源无法访问。** 评论一致好评，并要求增加“说韩语的蔬菜”这一主题；未出现技术辩论。
- [**你在“最奇怪的跳蚤市场”卖什么？第 7 部分**](https://www.reddit.com/r/aivideo/comments/1nvr8hw/what_do_you_sell_at_the_strangest_flea_market_pt_7/) (热度: 475): **短视频喜剧短剧帖子“你在‘最奇怪的跳蚤市场’卖什么？第 7 部分”，托管在 Reddit 视频 ([v.redd.it](http://v.redd.it/)) 上，目前未经身份验证的客户端无法访问 (**`HTTP 403 Forbidden`**, 需要 OAuth)。从评论来看，该作品是一个连贯的超现实/荒诞系列的一部分，包含一个猪拉丁语文字游戏笑话，并明确致敬了 Tim Robinson 在《提姆·罗宾森：太超过啰！》（I Think You Should Leave）中的“得来速”片段 ([剧集信息](https://en.wikipedia.org/wiki/I_Think_You_Should_Leave_with_Tim_Robinson))。** 评论一致好评；唯一在技术上值得注意的观察是对 Tim Robinson 短剧风格的互文引用，以及将猪拉丁语作为一种风格手段。
    - 一位创作者强调了当前图像模型的构图和控制限制——特别点名了 **Midjourney** ([https://www.midjourney.com](https://www.midjourney.com/))、“Seedream”和 **FLUX** (例如 [https://huggingface.co/black-forest-labs/FLUX.1-dev) —— 指出](https://huggingface.co/black-forest-labs/FLUX.1-dev)%E2%80%94%E6%8C%87%E5%87%BA) *“只是做新的单个角色和物体很无聊”。* 尽管他们的 AI 视频内容拥有“几千名”粉丝，但他们反映这些模型缺乏更丰富的视频管线所需的多主体场景构建能力和一致性，表达了对具有更好场景复杂性、控制力和连贯性的下一代模型的渴望。

- [**Is that math??**](https://www.reddit.com/r/OpenAI/comments/1nvs4so/is_that_math/) (活跃度: 477): **标题为“Is that math??”的帖子链接到一个 [v.redd.it](http://v.redd.it/) [视频](https://v.redd.it/wcfvg31i2msf1)，该视频目前因 Reddit 网络安全拦截返回** `HTTP 403 Forbidden`**，表明访问需要身份验证（登录或 OAuth 令牌），因此实际内容无法查看。从评论语境来看，该线程可能集中在物理/相对论幽默上（爱因斯坦引用、非惯性系），没有分享技术产物、Benchmark 或代码。** 热门评论调侃“发布爱因斯坦文件”，期待一个关于非惯性系中速度限制的相对论笑话，并宣布进入“新 Meme 时代”，暗示这是一种轻松的、以 Meme 为导向的反应，而非实质性的技术辩论。
- [**Good use of AI .. I laughed and almost choked lmfao**](https://www.reddit.com/r/ChatGPT/comments/1nwasfv/good_use_of_ai_i_laughed_and_almost_choked_lmfao/) (活跃度: 5333): **一段短小的 [v.redd.it](http://v.redd.it/) 剪辑（[链接](https://v.redd.it/8lah9oz5lqsf1)）似乎展示了一个基于逼真的 AI 生成照片的恶作剧，引发了关于配套脚本/旁白是否也由 AI 创作的疑问。从技术角度看，该线程强调了消费级生成工具如何轻易地合成针对非技术受众的多模态、高可信度骗局，说明了逼真图像合成和脚本化语境带来的社会工程风险面。** 评论者争论脚本是否由 AI 生成，并建议利用此类案例来教育年长亲属防范 AI 驱动的操纵；其他人则批评这种恶作剧是不负责任或有害的，指出了为了取乐而惊吓家人的伦理底线。
    - 唯一的准技术线程指出，除了 AI 生成的照片外，“脚本”也可能是 AI 制作的——这意味着这是一个多模态伪造工作流（文本 + 图像），而非单一模态的 Deepfake。另一条评论将其视为操纵技术欠佳亲属的社会工程矢量，但讨论中不包含具体的实现细节、模型名称或评估详情（例如检测方法、Benchmark 或 Pipeline 组件）。
- [**I hope the White House doesn’t sue us**](https://www.reddit.com/r/ChatGPT/comments/1nvwk4e/i_hope_the_white_house_doesnt_sue_us/) (活跃度: 1287): **帖子似乎展示了一个高度逼真的 AI 生成视频（Deepfake），主角是 Donald Trump，评论者注意到 Sam Altman 也出现在视频中且看起来带有合成感。位于 [v.redd.it](http://v.redd.it/) 的原始资源（[链接](https://v.redd.it/ehrkx6dfansf1)）在没有 OAuth/登录的情况下无法直接访问（HTTP** `403 Forbidden`**），因此该剪辑的真实性和来源无法独立验证；访问需要 Reddit 登录（[链接](https://www.reddit.com/login/)）或支持协助（[链接](https://support.reddithelp.com/hc/en-us/requests/new?ticket_form_id=21879292693140)）。讨论突显了生成视频保真度的快速提升，以及标题所暗示的相关真实性/验证和法律风险担忧。** 热门评论强调了前所未有的现实感（例如，“我见过的最真实的 Trump 视频”），质疑部分内容是否真实（Altman“看起来有点假”），并建议在面临诉讼威胁时采取对抗性的法律立场。
    - 感知的照片级真实感阈值：多名用户误将剪辑认作真实视频，表明最先进的 AI 视频生成已跨越了可信度边界，普通观众已无法可靠地分辨合成与实拍，尤其是在政治背景的素材中。这突显了随着内容分发脱离原始标签，检测和溯源（如水印/元数据）面临的实际挑战。
    - 残余的恐怖谷线索：一位评论者注意到 Altman *“看起来有点假”*，这指向了面部建模中残余的瑕疵——微表情、时间连贯性和皮肤反射——这些在细心的观察者面前仍会暴露合成痕迹。混合的反应表明质量取决于场景和身份，失败通常出现在特写、复杂光照或快速表情变化时。

---

# AI Discord 摘要

> 由 gpt-5 生成的摘要之摘要之摘要
> 

**1. IBM Granite 4.0 混合模型发布**

- **Granite 4.0 走向混合、开源且企业就绪**：**IBM** 发布了采用混合 **Mamba/Transformer** 架构的 **Granite 4.0**，在 **Apache 2.0** 协议下开源，经过加密签名，并被称为在不损失性能的情况下实现超高效率，通过 **Hugging Face**、**LM Studio**、**NVIDIA NIM**、**Ollama** 和 **Replicate** 等合作伙伴广泛提供 ([IBM announcement](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models))。
    - 社区对其新的 **ISO 42001** 认证展开了讨论，一位用户称其为“完全无用的认证”，而其他人则关注实际的访问路径和企业级分发 ([IBM announcement](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models))。
- **Granite 的混合注意力机制：大规模活跃单元**：共享规格强调了不同规模下的**混合注意力机制**——**2B 稠密型**、**7B（1B 活跃）**和 **32B（9B 活跃）**——支持 **FIM** 且无位置编码，旨在避免超过 **128k** 上下文时的性能下降 ([IBM Granite HF collection](https://huggingface.co/collections/ibm-granite/granite-40-language-models-6811a18b820ef362d9e5a82c3))。
    - 用户注意到可以通过 **GGUF** 运行或参考 **Unsloth** 指南和资产进行微调的平滑路径，缩短了从模型库到训练栈的闭环 ([Unsloth Granite 4.0 guide](https://docs.unsloth.ai/new/ibm-granite-4-0), [IBM Granite HF collection](https://huggingface.co/collections/ibm-granite/granite-40-language-models-6811a18b820ef362d9e5a82c3))。

**2. Unsloth 训练栈：Docker、RL 加速与新技巧**

- **容器征服配置混乱**：**Unsloth** 发布了带有分步指南的跨平台 **Docker 镜像**，同时用户分享了针对 **Blackwell (SM_12)** 的手动 **xformers** 构建脚本以解锁最新内核 ([Docker guide](https://docs.unsloth.ai/new/how-to-train-llms-with-unsloth-and-docker), [Docker Hub](https://hub.docker.com/r/unsloth/unsloth))。
    - 该流程旨在实现 Windows/Linux 和高级 GPU 栈上的无缝训练，文档还涵盖了在同一流水线上对 **Granite 4.0** 进行微调的内容 ([Unsloth Granite 4.0 guide](https://docs.unsloth.ai/new/ibm-granite-4-0))。
- **极速 RL**：Unsloth 报告了使用 **GSPO** 实现的最快 **gpt-oss RL** 循环，此外 **VLM RL** 速度提升了 **2 倍**，显存（**VRAM**）占用减少了 **90%**，并利用内核和权重共享技巧支持 **10 倍长的上下文** ([gpt-oss RL blog](https://docs.unsloth.ai/new/gpt-oss-reinforcement-learning), [VLM RL blog](https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl))。
    - 早期测试者赞扬了其快速实验的吞吐量，将该技术栈视为大规模**推理 RL** 和**视觉语言**训练负载的实用入门方案 ([gpt-oss RL blog](https://docs.unsloth.ai/new/gpt-oss-reinforcement-learning), [VLM RL blog](https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl))。
- **Tversky 技巧与更精简的损失函数**：一个针对类 Llama 架构的 **GPT-2 Tversky-All** 半复现版本已发布，包含代码和测试模型——声称在 **3090 Ti** 上约 1 天即可处理 **300B tokens**——同时从业者建议通过 **Dao-AI Lab 的 quack** 使用 **Linear Cross Entropy** 来加速训练 ([Architecture-Tversky-All](https://github.com/CoffeeVampir3/Architecture-Tversky-All), [HF test model](https://huggingface.co/Blackroot/Tversky-All-Test-100MIsh), [LCE impl line](https://github.com/CoffeeVampir3/Architecture-Fast-Tiny-Dense-LLM/blob/main/main.py#L115), [quack LCE](https://github.com/Dao-AILab/quack/blob/main/quack/linear_cross_entropy.py))。
    - 社区建议强调了**序列打包的可变长度（varlen）Flash-Attn** 和精细的内核选择，以获得实际运行时间的优势，通过精简的损失函数配合高效的数据布局来减少训练轮数（Epochs） ([varlen MHA example](https://github.com/CoffeeVampir3/Architecture-Fast-Tiny-Dense-LLM/blob/main/modeling/MHA.py#L36))。

**3. GPU 系统：确定性、Flash-MoE 与内核融合**

- **确定性驯服随机性**：**Thinking Machines** 详细介绍了如何克服 LLM 推理中的**非确定性**（non‑determinism），并发布了 **Flash‑MoE**，这是针对稀疏专家（sparse‑expert）设置的 Flash‑Attention 变体（[Defeating Non‑Determinism](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)，[Flash‑MoE 网站](https://flash-moe.github.io/)）。
    - 工程师们指出，稳定的可复现性对于调试和基准测试模型追踪（traces）至关重要，将 **Flash‑MoE** 定位为可扩展 **MoE** 推理的实用构建模块（[Defeating Non‑Determinism](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)，[Flash‑MoE 网站](https://flash-moe.github.io/)）。
- **NVIDIA 论文聚焦融合与专业化**：**NVIDIA** 发布了关于调度和 **warp specialization** 的编译器研究工作，并提供了与 **FA3** 的基准测试对比（[Cypress, PLDI 2025](https://d1qx31qr3h6wln.cloudfront.net/publications/Cypress_PLDI_25.pdf)），以及关于端到端效率的**分布式算子融合**（distributed kernel fusion）研究（[Legate Kernel Fusion, ASPLOS 2025](https://d1qx31qr3h6wln.cloudfront.net/publications/Legate_Kernel_Fusion___ASPLOS_2025.pdf)）。
    - 讨论集中在将这些技术映射到生产环境的 **tensor programs** 和集群范围的执行图上，以减少启动开销并提高 **E2E throughput**。
- **JAX Blackwell Matmul 大师课**：**JAX** 发布了一份关于使用 **Pallas** 在 **Blackwell GPUs** 上实现 SOTA 级 **matmul** 性能的教程，涵盖了分块（tiling）、内存移动和算子编写的最佳实践（[JAX Blackwell matmul tutorial](https://docs.jax.dev/en/latest/pallas/gpu/blackwell_matmul.html)）。
    - 从业者强调该指南是手动优化 **GEMM** 算子的蓝图，能够转化为 **training** 和 **inference** 流水线中的实际收益。

**4. OpenRouter：路由指标、费用与新模型**

- **性能图表引发量化疑问**：**OpenRouter** 推出了 **Performance Tab**，可视化了每个模型的供应商指标，引发了按**量化**（例如 **FP4** vs **BF16**）进行过滤的需求，以避免误导性的比较（[Performance Tab post](https://x.com/OpenRouterAI/status/1773733582763069916)）。
    - 用户请求增加量化级别的下拉菜单，并指出公平的同类比较需要对 **precision**、**context** 和 **tool‑use** 设置进行归一化。
- **BYOK 澄清：0% 费用，而非免费算力**：**“每月 100 万次免费 BYOK 请求”**促销活动免除了 OpenRouter 对前 100 万次请求收取的 **5% 佣金**，但用户仍需支付底层供应商的 API 账单（[公告](https://openrouter.ai/announcements/1-million-free-byok-requests-per-month)）。
    - 一些人建议使用更清晰的措辞，如 *“每月 100 万次 0% 费用的 BYOK 请求”*，以避免对实际 **inference costs** 产生混淆（[公告](https://openrouter.ai/announcements/1-million-free-byok-requests-per-month)）。
- **Qwen 图像编辑器入场**：**Alibaba Qwen** 推出了一款新的 **image‑edit** 模型（非文生图），开发者们分享了发布消息并寻求 Apple Silicon 上的运行路径（[Qwen 公告](https://x.com/Alibaba_Qwen/status/1973668568412856595)，[社区帖子](https://x.com/pingToven/status/1973758872772108663)）。
    - 早期讨论集中在**仅限编辑**（editing‑only）的约束和集成问题上，并对本地 **M‑series** 加速表现出兴趣。

**5. LMArena：推理追踪与排行榜变动**

- **观察模型“三思而后言”**：**LMArena** 为 **Side‑by‑Side** 和 **Direct** 对话模式下的推理模型启用了 **Reasoning Trace**，让用户在模型回答前看到其思考过程（[Side‑by‑Side](https://lmarena.ai/?mode=side-by-side)，[Direct](https://lmarena.ai/?mode=direct)）。
    - 高级用户欢迎这种增加的透明度，以便调试 **reasoning chains**、比较模型的 **scratchpads** 并对**中间步骤**进行正确性检查。
- **Claude Sonnet 4.5 登顶文本排行榜**：**Claude Sonnet 4.5** 与 **Claude Opus 4.1** 并列 **Text Leaderboard** **第一名**，且 **32k thinking** 变体在生产流程中取代了 **16k** 版本（[Text Leaderboard](https://lmarena.ai/leaderboard/text)）。
    - 社区评论赞扬了其在 **Hard Prompts**、**Coding** 和 **Creative Writing** 方面的表现，认为感知质量与更新后的 **thinking window** 相符。

---

# Discord：高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 弃用 o3，推销 GPT-5**：Perplexity 从其模型选择器中弃用了 **o3 model**，并鼓励用户过渡到 **GPT-5 Thinking**。
   - Perplexity 声称 **GPT-5 Thinking** 提供更强大的性能和持续的支持。
- **Discord 桌面版助力 Comet 任务**：用户正在下载 [Discord desktop app](https://discord.com/download) 以完成 **Comet quest** 并领取 5k orbs。
   - 一些用户在 Discord 应用中找不到该任务，建议 *查看置顶消息 (pins)*！
- **隐私面临考验**：一位用户分享了一段结合了英语、芬兰语、日语和西班牙语的记忆，引发了关于隐私的讨论。
   - 另一位用户表示他们可以分享 prompt，但不会去翻阅记忆来剪掉私人部分，并怀疑这些记忆是否真的会产生影响。
- **Comet 浏览器启动失败**：一位用户分享了成功的 [截图](https://cdn.discordapp.com/attachments/1047649527299055688/1423483019540434954/image.png?ex=68e0795e&is=68df27de&hm=8a12e66bd9f89baf735f8c1bcd80271022e88595da198b5fd7bbaebe96aa5b64&)，指出浏览器的开启体验非常棒。
   - 其他人指出它确实还需要改进，因为它就像普通的浏览器一样，而且更烦人的是你不能将 Google 设为主要搜索引擎，也不能使用 shift+enter 来调用 AI。
- **Sonar-Pro API 返回失效资源**：一位用户报告称 **Sonar-Pro API** 生成的资源会导致 **404 errors**，并询问是否有过滤结果的方法。
   - 他们希望只接收确认存在且对公众开放的资源，以避免 **404 errors**。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Sora 2 发布引发热潮**：社区成员热切期待 **Sora 2** 登陆该平台，预见其影响力并将其与 **Veo 3** 等视频模型进行比较。
   - 爱好者们表达了兴奋之情，并希望看到它在 LMArena 上进行基准测试。
- **Gemini 3 发布推测升温**：社区对即将发布的 **Gemini 3** 议论纷纷，讨论焦点集中在其在 [ratelimits](https://discord.com/channels/1340554757349179412/1340554757827461211/1423022363716485121) 方面的潜在竞争力。
   - 一份泄露消息称发布日期为 **10 月 9 日**，进一步推高了期待感。
- **4o 模型退役引发失望**：用户对 LMArena 上 **4o model** 的有限可用性以及最终退役表示失望。
   - 一位成员哀叹自己对 **4o** 的“成瘾”，强调了寻找合适替代品的困难。
- **伦理边界引发辩论**：人们对 **OpenAI** 的数据使用行为表示担忧，一位用户开玩笑地承认向 lmarena 发送了 *敏感政府数据*。
   - 另一位成员指出，在 Discord 聊天中承认这种事太疯狂了。
- **Claude Sonnet 4.5 夺得文本排行榜第一**：**Claude Sonnet 4.5** 表现出色，与 **Claude Opus 4.1** 并列 [Text Leaderboard](https://lmarena.ai/leaderboard/text) **第一名**。
   - 它在 Hard Prompts、Coding 和 Creative Writing 等类别中也表现良好，在专用频道中 *获得了社区的积极讨论*。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Blackwell 手动编译热潮开启**：成员们讨论了在 **Blackwell GPU** 上手动编译 **xformers**，其中一位分享了一个脚本，使用 `pip uninstall -y xformers`、`git clone` 和 `python3 setup.py install` 为算力（compute capability）为 **12** 的设备手动编译 xformers，并提供了更新后镜像的 [Docker Hub 链接](https://hub.docker.com/r/unsloth/unsloth)。
   - 这是使用**最新 GPU** 进行加速计算的必要步骤。
- **Unsloth 训练专用 Docker 亮相**：Unsloth 发布了一个新的 **Docker 镜像**，用于在 Windows/Linux 上进行训练而无需担心依赖问题，详见其[指南](https://docs.unsloth.ai/new/how-to-train-llms-with-unsloth-and-docker)，并可在 [Docker Hub](https://hub.docker.com/r/unsloth/unsloth) 上获取。
   - 此举旨在解决依赖冲突，并为不同操作系统的用户简化设置流程。
- **无需 vLLM 的合成数据激增**：成员们讨论了在不依赖 **vLLM** 的情况下生成**合成数据集**，建议使用 **OpenAI package** 向本地服务器发送异步请求，并指向了 [meta-llama/synthetic-data-kit](https://github.com/meta-llama/synthetic-data-kit)。
   - 一位成员指出，目前所有的 Unsloth notebook 都使用 vLLM。
- **Tversky-All GPT2 获得类 Llama 升级**：一位成员发布了 **GPT2 Tversky-All** 的半复现版本，采用了 Tversky-All 策略但应用于类 llama 模型，代码见 [CoffeeVampir3/Architecture-Tversky-All](https://github.com/CoffeeVampir3/Architecture-Tversky-All)。
   - 测试模型可在 [HuggingFace](https://huggingface.co/Blackroot/Tversky-All-Test-100MIsh) 获取；该模型在 **3090 TI** 上使用 **3000 亿 token** 训练了约一天时间。
- **GGUF 转换问题困扰用户**：用户在尝试将模型转换为 **GGUF 格式**时遇到问题，特别是在使用 *push_to_hub_gguf* 函数进行 **f16 量化**时，建议在修复程序发布前手动进行转换。
   - 一位成员报告了一个与映射张量 'model.layers.0.self_attn.q_proj.base_layer.weight' 相关的 **ValueError**。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Sonnet 4.5 定价引发辩论**：成员们辩论了 **Sonnet 4.5** 与 **GLM 4.6** 的性价比，一些人指出 **GLM 4.6** 便宜六倍。
   - 一些用户认为 **Sonnet 4.5** 的表现与 **3.7** 相似，而另一些用户在 **Copilot** 中更倾向于使用 **4** 而非 **3.7**。
- **服务器被 Sora 粉丝占领**：一位成员对服务器被 **Sora 用户**占领表示担忧，并批评 **OpenAI 的营销**引发了这一涌入。
   - 该成员建议根据当前的讨论话题，使用 **LLM** 动态更新频道名称。
- **Deepfake 争议引发用户分歧**：一位用户质疑某应用支持 Deepfake 却批评生成写实 AI 图像的讽刺性。
   - 这在大量的 *code please* 请求中引发了关于将反馈转发到相关频道的讨论。
- **Sora 作为社交媒体中心**：一位用户建议 **Sora** 应该整合为一个像 **TikTok** 这样的社交媒体平台，通过 **ChatGPT** 增强用户体验，类似于图像生成。
   - 另一位用户建议为 **Sora** 实施**积分系统**，为视频生成分配更多资源，并设置**每日或每周使用限制**。
- **用户讨论 Sora 的方形图像**：成员们讨论了 **Sora** 图像生成的最佳实践，其中一人询问**纵向模式**是否比**横向模式**效果更好。
   - 另一位成员回答说，视觉 token 是按网格排列的，因此**方形图像**可能会从图像生成中获得最佳效果。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Snapdragon X Elite 规格引发讨论**：一位用户分享了其 **Microsoft Surface Pro** 的规格，配备了 **Qualcomm Snapdragon X Elite**（12 核，X1E80100 @ 3.40 GHz）和 **16 GB** RAM。
   - 在看到一个神秘的“伪影”后，他们询问 LLM 的意见是否值得信赖。
- **量化困惑探讨**：成员们探讨了量化如何影响语言模型的知识保留，较低的量化可能会影响较小的模型，因为丢失了“推理位（reason bits）”。
   - 一位成员分享了一个关于量化级别过于极端时会发生什么的幽默观点：*当你被过度量化时，突然之间你会以一种意想不到的方式把病娇（yanderes）和宠物狗混为一谈😄*。
- **GPT-OSS：开源安全替代方案发布**：宣布发布 **GPT-OSS**，该模型的行为与 **GPT-4o** 类似。
   - 成员们注意到，如果没有提供足够的细节，它会假设大量信息。
- **Arc B50 Pro 带宽瓶颈之争**：一位成员将 **Arc B50 Pro** 显卡与 **RTX 4080 Super** 进行了基准测试，显示 B50 虽然拥有“海量 VRAM”，但内存带宽极差，导致 Token 速率较低（**12B q8_0 模型**为 **7-8 Tps**，而 4080 为 **30+ Tps**）。
   - 然而，在默认上下文（**4k**）下，B50 达到了 **32 Tps**，而 4080 为 **42 Tps**。
- **GPU 部署中的 DDR3 幻想破灭**：一位用户建议使用带有多个 **PCIE 16x 插槽**的廉价 **DDR3** 主板来容纳 **6x GPU**，并结合 RAID 的 **SATA SSD** 以实现更快的加载速度，并引用了 [eBay 上的 X99 主板列表](https://ebay.us/m/zB2BAH)。
   - 成员们对内存带宽（**DDR4** 为 **68 GB/s**）以及与现代标准相比潜在的瓶颈表示担忧，一位用户表示 *在 DDR3 上最高只能达到约 50gb/s*。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 集成 Git Worktree**：用户在设置的 **beta 选项卡**中发现了 **Git Worktree** 设置，并鼓励在 Agent 窗口中使用它。
   - 看起来 Git Worktree 集成在 **Early Access** 或 **Nightly Cursor** 版本中可用。
- **Cursor Beta 功能引发好奇**：成员们讨论了在 Cursor 中使用 Beta 功能，推荐将其用于早期功能体验、趣味调试和帮助改进 Cursor；目前，**afterFileEdit** 是唯一可用的钩子（hook）。
   - **Extension RPC Tracer** 可用于在 Beta 功能使用期间检查 RPC。
- **Typescript 重构成功案例**：一位用户报告称，在使用四个 Prompt 后，通过 Cursor 成功完成了完整的 **Typescript 重构**，并使用后续的 Master Prompt 进行审计。
   - 建议使用 Cursor 的 **Plan 模式**并在 Nightly 版本中跟踪工作流状态，以提高效率。
- **MacBook 因 Cursor 导致崩溃**：一位用户报告称，Cursor 导致其 **MacBook Air M4** 因高内存占用而崩溃，内存飙升至 **96GB**，可能与 Chat 或 Agent 进程有关；重启后解决。
   - 成员们怀疑存在内存泄漏，并指出 **MacOS** 版本的发生频率更高，建议降级作为临时解决方案。
- **Cursor 黑客松即将到来？**：一位成员询问是否有兴趣参加 **Discord Cursor 黑客松**，以实现解决方案和侧边项目。
   - 成员们对提供免费额度的赞助黑客松表示了兴趣，并建议使黑客松支持远程参与，以适应不同的时区。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 推出供应商性能选项卡**：OpenRouter 发布了一个全新的“性能选项卡（Performance Tab）”，用于可视化 [特定模型的供应商性能](https://x.com/OpenRouterAI/status/1773733582763069916)，这引发了关于在使用不同量化水平的供应商之间进行公平比较的讨论。
   - 一位用户建议增加一个过滤器下拉菜单，以考虑 **FP4** 和 **BF16** 等不同的 **quant levels**（量化级别），从而防止误导性的比较。
- **BYOK 促销引发困惑**：用户对“每月 100 万次免费 BYOK 请求”的优惠活动进行了讨论，澄清该活动是免除了 OpenRouter 对前 100 万次请求收取的 **5% 佣金**，但根据 [OpenRouter 文档](https://openrouter.ai/announcements/1-million-free-byok-requests-per-month)，用户仍需直接向供应商支付 API 使用费用。
   - 一些用户最初认为该优惠提供的是完全免费的请求，因此建议使用更清晰的表述，例如 *“每月 100 万次 BYOK 请求 0% 手续费”*。
- **Grok 遭到吐槽，Sonoma 表现更佳？**：一位用户测试了 **Grok 4 Fast**，称其 *“比 Sonoma 笨得多”*，并表示它 *“经常失败”* 且无视格式要求。
   - 另一位用户推测 Grok 4 Fast *“散发着…… Llama 的味道……？”*，对其不一致的表现表示失望。
- **Gemini Pro 出现故障**：用户报告称 **Gemini Pro** 正在响应 *“奇怪的内容”*，无法正确使用工具，并且通过 OpenRouter API 表现出 *“无法接受的缓慢”* 性能。
   - 报告显示这可能是 **Gemini 2.5 Pro** 的普遍问题，一位用户建议尝试使用 Vertex 作为替代供应商。
- **Qwen 图像编辑上线！**：成员们分享了 [阿里巴巴新的 **Qwen 图像模型**](https://x.com/Alibaba_Qwen/status/1973668568412856595)，指出它仅是一个图像编辑模型，一位用户分享了宣布该消息的 [这条帖子](https://x.com/pingToven/status/1973758872772108663)。
   - 另一位成员表达了在 **Apple Silicon** 上运行它的兴趣。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **探索 Perplexity AI 框架**：成员们讨论了 [Perplexity AI 框架](https://www.perplexity.ai/page/scientists-ai-framework-solves-hTbKnxfPSl64P5nLfxqX5A) 及其相关的 **GitHub 项目**，特别关注了使用类似注意力矩阵的 **LLMs**。
   - 讨论考虑了高效的注意力机制，如作为 **top-k attention** 示例的 **Deepseek Sparse Attention**，并质疑了其与滑动窗口注意力（sliding window attention）相比可能存在的问题。
- **梯度下降动力学论文备受赞誉**：一位成员赞扬了一篇关于梯度下降动力学的论文 ([centralflows.github.io](https://centralflows.github.io/part1/))，因为它解决了 **loss spike dynamics**（损失尖峰动力学）并影响了 **Adam 的 beta2**。
   - 尽管该论文引用量较低，但因其解决方案和影响力而受到称赞，被一位成员视为**年度论文**。
- **对称 Transformer 展现潜力**：使用对称 Transformer ([GitHub repo](https://github.com/Eternalyze0/symmetry_transformer)) 进行的实验表明，使用**独立的 Head 预测当前和前一个 token** 改善了后期训练运行中的验证损失。
   - 初步结果显示基准模型表现更好，但对称模型在经过更多训练后表现有所提升。
- **质疑 AUNN 的实用性**：**AUNN (Augmented Neural Networks)** 的实用性引发了争论，人们对其效率以及除了玩具示例 [ethan-w-roland/AUNN](https://github.com/ethan-w-roland/AUNN) 之外缺乏功能性原型表示担忧。
   - 讨论指出，AUNN 的提出者更多地关注 **MLPs** 而非 **Attention**，并且对反驳意见持*对抗*态度。
- **Transformer 是二维切片**：该社区讨论了 Transformer 通过将一个大的二维问题（序列 sequence，通道 channels）分割成切片来进行优化。
   - 一位成员表示，将巨大的 **MLP** 应用于整个问题*虽然可行，但在计算上是难以处理的*，而使用 **Transformers** 仅仅是因为它们*成本低廉*。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Benchmarking 头脑风暴开启**：成员们寻求一份*优秀的 Benchmarking 指南*，并被推荐了[这篇 arXiv 论文](https://arxiv.org/abs/2502.15015)、[这篇关于 kernel benchmarking 的文章](https://jan.ai/post/how-we-benchmark-kernels)以及[这个 YouTube 视频](https://www.youtube.com/watch?v=1i7dxoAfKOU)。
   - 一位成员将他们之前的 Benchmarking 工作描述为*可能是最出色的 Benchmarking 尝试*。
- **非确定性（Non-Determinism）宣告终结**：**Thinking Machines** 发布了一篇关于在 **LLM inference** 中消除**非确定性**的[博客](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)。
   - 他们还发布了 [Flash-MoE](https://flash-moe.github.io/)，这是 **Flash Attention** 的一个变体。
- **Nvidia 编译新代码**：**Nvidia** 正在研究用于调度（scheduling）和 **warp specialization** 的**编译器技术**，并在其[论文](https://d1qx31qr3h6wln.cloudfront.net/publications/Cypress_PLDI_25.pdf)中详细介绍了针对 **FA3** 的 Benchmarking 结果。
   - **Nvidia** 正在分布式环境中进行 kernel 融合，如这篇[论文](https://d1qx31qr3h6wln.cloudfront.net/publications/Legate_Kernel_Fusion___ASPLOS_2025.pdf)所述。
- **用于 LLM 训练的 Linear Cross Entropy**：推荐使用 [Linear Cross Entropy](https://github.com/CoffeeVampir3/Architecture-Fast-Tiny-Dense-LLM/blob/main/main.py#L115) 来加速 **LLM 训练过程**，并建议使用 **Quack** 优化库，特别是其 [linear cross entropy 实现](https://github.com/Dao-AILab/quack/blob/main/quack/linear_cross_entropy.py)。
   - **Sequence packed 或 'unpadded' 训练**被认为是一种极具影响力的优化，特别是结合 **flash attn varlen** 等技术，参见[此实现](https://github.com/CoffeeVampir3/Architecture-Fast-Tiny-Dense-LLM/blob/main/modeling/MHA.py#L36)。
- **Cooperative Group 对齐问题**：一位成员询问了 `CooperativeGroup.__init__` 中的 `alignment` 参数，具体询问其作用以及为什么在 *Cutlass 频道*中，如果 `size` 为 32 则其必须为 32，而其他值则不然。
   - 另一位成员回答说，这种检查是*因为它们恰好是 warp/warpgroup 的粒度，是需要特殊检查以防止 Bug 的常见情况*。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Musk 构想 AI-MMO**：Elon Musk 正在讨论与 **Eve Online** 的创作者共同开发一款**集成 AI 的 MMO (AIMMORPG)**，旨在利用独特的 AI 能力。
   - 一位用户推测 AI 将是此类游戏中*“天生的绝配”*。
- **Karpathy 关于 Bitter Lesson 的思考**：Karpathy 总结了 [Dwarkesh-Sutton 播客](https://x.com/karpathy/status/1973435013875314729)，强调了 Sutton 对 **LLMs** 是否实现了其论文论点的怀疑。
   - Karpathy 承认预训练（pre-training）提供的实际引导作用，同时也暗示更大的范式即将到来，研究人员应该从动物智能中寻求灵感。
- **Hume AI 凭借 Octave 2 迈向极速**：[Hume AI 推出了 Octave 2](https://xcancel.com/hume_ai/status/1973450822840152455?s=46)，这是他们的下一代多语言文本转语音（TTS）模型，现在支持 **11 种以上语言**，速度提升 **40%**（延迟 <200 ms），成本降低 **50%**。
   - 该版本包括多发言者闲聊、改进的发音、新的语音转换（voice-conversion）和音素编辑（phoneme-editing）工具，并在 10 月期间为其 Creator 计划提供 **50% 的折扣**。
- **Mistral 招募数学高手**：Albert Jiang 宣布 Mistral AI 在获得 20 亿美元融资后，正在组建一个新的**形式数学（formal-math）研究团队**。
   - 他们正在为集证明器/自动形式化工具/Agent 于一体的项目寻求 AI 人才，提供顶尖的合作伙伴、人均数百块 GPU、开放式研究、顶薪以及在巴黎、伦敦和帕洛阿尔托的办公室；职位空缺发布在[此处](mailto:aj@mistral.ai)。
- **Figma 的 AI 实战指南**：Latent Space 播客邀请了 **Figma 联合创始人 Dylan Field** 讨论 **Figma 的 AI 策略（AI Playbook）**。
   - 本期节目探讨了在 **vibe-coding** 时代如何呈现优秀设计、**Figma's Make**、用于“品鉴”Agent 的 **MCP**，以及 **fast-fashion SaaS** 的未来（[X 链接](https://x.com/latentspacepod/status/1973793231524806925)，[Xcancel 链接](https://xcancel.com/latentspacepod/status/1973793231524806925)）。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes 模型声称接近 GPT-5**：一位成员询问 **Nous Research** 的微调模型是否可以与 **GPT-4.5** 相媲美，得到的回复是这些模型更接近 **GPT-5** 或 **Gemini**。
   - 讽刺的是，当该成员询问 **Gemini** 有哪些替代方案时，**Hermes** 赫然出现在其建议的选项中。
- **Veo3 盖过 Sora？**：一位用户表示相比最新的 **Sora**，他更倾向于 **Veo3**，并在讨论中分享了一个 [Prompt_theory.mp4](https://cdn.discordapp.com/attachments/1149866623109439599/1423308176148922491/Prompt_theory.mp4?ex=68dfd688&is=68de8508&hm=e195c2f737881136d240fa288b286f7dcc417fbe581c153cefa587b7c2ec0233&)。
   - 目前没有提供更多细节来阐述为什么 **Veo3** 是更好的选择。
- **Granite 模型展示混合注意力机制**：**IBM Granite** 语言模型在 **2B dense**、**7B (1B active)** 和 **32B (9B active)** 等模型中采用了混合注意力机制 (Hybrid Attention)，详情见分享的 [Hugging Face collection](https://huggingface.co/collections/ibm-granite/granite-40-language-models-6811a18b820ef362d9e5a82c3)。
   - 这些模型支持 **FIM** (Fill in the Middle)，并且没有使用位置编码 (Positional Encoding)，这防止了在处理超过 **128k** 上下文时出现性能下降。
- **Qwen 30B A3B 在 CPU 上表现出色**：成员们发现 **Qwen 30B A3B** 非常适合 **CPU** 使用，一位用户报告了在配备 **32GB VRAM** 的 **Ryzen 7 5700G** CPU 上的性能指标。
   - 具体而言，**Qwen 3 30B A3B** 在 **Q6_K_XL** 量化下，在 **1024** token 上下文时达到了 **48 TPS** 的处理速度和 **10.5 TPS** 的生成速度。
- **LLM 陷入欺骗之网？**：一位成员分享了关于 **战略性 LLM 欺骗 (Strategic LLM Deception)** 的[预印本](https://arxiv.org/html/2509.20393v1)。
   - 该研究使用 **Sparse Autoencoders**（由 [Goodfire AI](https://www.goodfire.ai/) 托管）展示了当前方法如何无法检测驱动 **战略性 LLM 欺骗** 的内部特征，并指出了一条缩小 **Autolabel Gap** 的切实路径。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Deepmind 代码的不完整性**：成员们开玩笑说 [Deepmind](https://deepmind.google/) 为了避免分享其实际实现做了额外的工作，导致人们不清楚其作为大型系统的一部分是如何运作的，并引用了他们在实现 **V-MPO** 时的经验。
   - 他们指出 Deepmind 的代码通常很复杂，但他们将其拆分的方式掩盖了整体功能。
- **HuggingPapers 代码无法运行**：成员们注意到来自 [HuggingPapers 的代码](https://fxtwitter.com/HuggingPapers/status/1973420932497879298?t=jxTf48_aBK8349s1uSyDQw&s=19) 无法运行，因为 **没有导入 RoPE**。
   - 代码的原始发布者似乎暗示用户应该自己实现它。
- **IBM Granite 4.0 混合架构**：IBM 推出了下一代语言模型 **Granite 4.0**，采用了全新的 **混合 Mamba/Transformer 架构**，在不牺牲性能的情况下大幅降低了内存需求，并以 **Apache 2.0 许可证** 开源。
   - 这些模型可在 **IBM watsonx.ai** 以及包括 Dell Technologies, Docker Hub, Hugging Face, Kaggle, LM Studio, NVIDIA NIM, Ollama, OPAQUE 和 Replicate 在内的平台合作伙伴处获取，即将支持通过 AWS Sagemaker JumpStart 和 Microsoft Azure AI Foundry 访问。[IBM 公告链接](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models)。
- **对 ISO 42001 认证的质疑**：成员们注意到新的 Granite 4.0 模型是全球首个获得 **ISO 42001 认证** 的开源模型。
   - 一位用户评论说，这是一个完全没用的认证，只是为了忽悠高管层（C-suite）让他们觉得这物有所值。
- **Oracle 在运行 OpenAI 的数据中心？**：一位用户评论说，Oracle 过去的商业模式是销售数据库和企业软件，现在似乎变成了为 OpenAI 运行数据中心。
   - 他们引用了 [OpenAI Elon Musk Post](https://openai.com/elon-musk/) 作为这一理论的来源。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Credit Consumption Sparks Outrage**: 一名用户抱怨一个基础研究任务消耗了 **5300 credits** 却未完成，称 Manus 是个“绝对的笑话”并请求[退款](https://cdn.discordapp.com/attachments/1349440650495398020/1423044743377715343/image.png?ex=68e032b1&is=68dee131&hm=cd60314f2b422e917efdd7ebbd2a8747117a9e000904040bddb4b0a1d2624fd9)。
   - 一名团队成员索要了会话链接以进行调查，并可能提供额度退款。
- **Unlock Agent Mode with Memory Key**: 一名成员提议使用 **Memory Key 协议**来解决退出 Agent Mode 的问题，该协议涉及在重启会话前保存上下文。
   - 他们详细说明了一个[解决方案](https://discord.com/channels/1348819876348825620/1349440650495398020/1422940046855766016)，包括复制关键信息、启动新会话，并指示 Agent 创建更新的 **Memory Key** 以供将来使用。
- **Billing Issue Sparks Support Vacuum**: 一名用户报告了账单问题，但 Manus 支持团队没有回应，促使社区成员建议向其官方支持邮箱发送邮件，并附上清晰的主题行和工单编号。
   - 建议这样做可以留下正式的书面记录以便升级处理。
- **Global Pricing Model Criticized for Disparity**: 一名用户批评 Manus 的**全球美元定价模型**（Plus 计划每月 39 美元）没有根据地区经济进行调整，在巴西和其他拉丁美洲国家造成了障碍。
   - 另一名用户建议根据**购买力平价 (PPP)** 实施区域定价，以提高可访问性并促进全球增长。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **AGI Paper Dropped, Courtesy of HF**: 一名成员在 `show-and-tell` 频道分享了一篇介绍 **AGI** 的 [Hugging Face 论文](https://huggingface.co/papers/2509.26507)。
   - 该用户俏皮地表示：“被我说中了吧 😉”。
- **Show-and-Tell channel debuts on DSPy Discord**: DSPy Discord 服务器新增了一个 **show-and-tell** 频道。
   - 该频道旨在供用户演示和讨论他们使用 DSPy 开发的项目。
- **Caching Prompt Order: Use with Care**: 成员们发现发送 Prompt 和文件的顺序会极大地影响缓存。
   - 为了有效利用缓存，必须仔细观察 Prompt 元素和文件输入的特定顺序。
- **DSPyWeekly gets Search Feature**: [DSPyWeekly](https://dspyweekly.com/) 现在具备搜索功能，可以浏览抓取的内容，并配有前一页/后一页链接以实现平滑导航。
   - 这一增强功能简化了信息获取，方便更容易地发现相关主题。
- **XMLAdaptor May Become New JSONAdaptor**: 成员们辩论了在 **ChatAdaptor** 或 **XMLAdaptor** 经常能修复适配器错误的情况下，**JSONAdaptor** 是否应保持默认。
   - 模型工具使用 RL 的兴起使 **XML** 成为潜在的默认选择，尽管 **JSON** 仍是一个可靠的备选方案。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Qwen Coder Model Benchmarks**: 成员们讨论了 **Qwen Coder** 模型，认为 **30B** 版本“应该足够聪明”，并考虑将 **Qwen3 Coder** 作为更新的替代方案。
   - 他们警告说量化可能会影响性能，如果选择的话建议使用 **Q4**。
- **Aider's Release Cadence Concerns**: 一名成员对 **aider** 发布节奏放缓表示担忧，并建议建立 **Patreon** 或捐赠系统以提供支持。
   - 该用户强调了对开发者倦怠和 **aider** 可能停止维护的担忧，考虑到与其他 Agent 工具相比，它在实际工作中的实用性。
- **Aider-desk UI Experiences**: 一名成员询问了关于将 **aider-desk** 或类似的 UI 与 **aider** 配合使用的情况。
   - 另一名成员曾短暂使用它来获得 **MCP support**，发现它适合那些想要 **aider** 风格工作流且有可选 Agent 使用场景的人，但他们后来切换到了 **sst/OpenCode**。
- **DeepWiki Reverse Engineering Invitation**: 一名成员分享了一个 [DeepWiki 页面](https://deepwiki.com/search/please-reverse-engineer-the-pr_c15e0046-3403-4786-bf26-63b2bf046455)，鼓励进行逆向工程。
   - 另一名成员建议在 *koboldcpp* 中使用**输出模板**或**后处理**，不确定 *llama.cpp* 中是否可用。
- **Custom Chat Templates Hack**: 一名成员提到可以指定自定义的 **Jinja chat template** 来覆盖 **GGUF** 中包含的模板。
   - 他们还建议使用 [GBNF](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md) 来格式化模型的输入，并在 [llama.cpp 上发起了讨论](https://github.com/ggml-org/llama.cpp/discussions/16386)。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP 开发者在 Ye Olde London 聚会**：成员 <@1042188459575619674> 和 <@1387880172589547722> 在 **Ye Olde London** 举办了聚会，邀请其他开发者进行线下交流和建立联系。
   - 一位成员 <@1407892993934889010> 提到他们会 *“过去坐一会儿！”*
- **Registry 团队进行直播**：Registry 团队在 **英国时间上午 9 点** 开启了直播，观看地址见 [此处](https://www.youtube.com/watch?v=5qwXADMBuio)。
   - 直播涵盖了团队工作的各个方面。
- **提议在 Sampling 中支持 Tool Call**：一位成员通过 [issue #1577](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1577) 提交了一项提案 (SEP)，旨在将 **Tool Call 支持集成到 Sampling** 中。
   - 拟议的集成取决于关于 **多个内容块 (multiple content blocks)** 的持续讨论，旨在通过 [PR #198](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/198) 实现并行 Tool Call。
- **参考实现简化了测试**：发布了一个新的参考实现 (TS SDK)，其特点是包含一个由 **agentic loop tool** 驱动的示例服务器，以及一个旨在简化测试的 **backfill proxy**，详见 [PR #991](https://github.com/modelcontextprotocol/typescript-sdk/pull/991)。
   - 一位成员指出，最初的 CI 失败通过固定 **zod** 的次要版本得到了解决。
- **为 MCP 服务器开发类 OCI 接口的想法**：一位成员建议为 **MCP 服务器开发类似 OCI 的接口**，将所有元数据打包在 tarball 中以简化处理。
   - 目标是简化构建和分发 **OMCP 软件包** 的过程，从而简化元数据管理。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **高通对 Mojo 感兴趣？**：一位成员推测 **Qualcomm** 可能会就 **Mojo** 联系 **Modular**，这可能表明他们有兴趣利用 **Mojo** 的能力来优化其硬件。
   - 讨论起源于 Qualcomm 开发者 Discord 的语音聊天。
- **Mojo 手册更新 Python 相关内容**：**Mojo Manual** 进行了更新，一位用户特别强调了 [Python 章节](https://docs.modular.com/mojo/manual/python/)。
   - 此次更新表明了关于 **Mojo** 与 **Python** 互操作性的增强或关键细节。
- **Mojo 探索 Notebook 领域**：讨论集中在 Notebook 中使用 **Mojo**，具体目标是 *在 Notebook 中与 Max 交互* 还是 *直接在 Notebook 中编写和运行 Mojo*。
   - 一位用户报告了在 Notebook 中与 **Mojo** 交互成功的经验，并对语法高亮显示以方便学习表示了兴趣。
- **Radeon GPU 通过向量加法测试**：一位用户在 **AMD Radeon 6800 XT** 上成功运行了 *vector_addition 示例*，参考了 [GPU 兼容性文档](https://docs.modular.com/max/packages/#gpu-compatibility)。
   - 一位 Modular 员工回应称，他们尚未对 **RDNA 2 GPU** 进行广泛测试，且模型目前还无法在 **RDNA GPU** 上正确运行。
- **Mojo 展望分布式计算的未来**：一位成员询问了将 **Mojo** 与 **Dask** 或 **PySpark** 等框架结合用于分布式计算的潜力。
   - 另一位成员建议 Mojo 欢迎人们构建自己的框架，因为全 **Mojo 框架** 的延迟可能比基于 Python 的选项更低，吞吐量更高。



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi 展示了一项令人惊喜的能力**：在观看 [演示视频](https://cdn.discordapp.com/attachments/1371757564005711973/1423369409669365881/2025-10-02_20-01-10.mp4?ex=68e00f90&is=68debe10&hm=b28aea7215ef03687754be113fa4dd6a583c355be6e3462543235d7d0258ef73&) 后，一位用户注意到了 **Kimi** 中一个意想不到的功能。
   - 提示词中未详细说明该新功能的具体细节。
- **Sora 的视频演示面临质量质疑**：用户正在对比分享的 **Sora** 视频演示质量，认为目前可见的版本质量可能低于 **OpenAI YouTube 频道** 上展示的版本。
   - 一位用户将质量描述为 *“奇怪的晃动感”*。
- **Sora Pro 订阅提供无水印输出**：**Sora** 的 **Pro 订阅** 版本据称将提供更高分辨率且无可见水印的视频。
   - 一位用户提醒道：*“会应用不可见的水印——这样 OpenAI 先生就能分辨它是生成的，只是我们看不出来……”*

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **ShapeTracker 面临即将被删除**: 一名用户询问了关于 **ShapeTracker** 即将被删除的情况，并寻求有关此更改的文档。
   - 另一名用户分享了一个相关的 [X 帖子](https://x.com/__tinygrad__/status/1967446896081076517)，对该问题进行了说明。
- **寻找 ShapeTracker 的继任者**: 在关于 **ShapeTracker** 删除的同一个查询中，该用户询问了潜在的替代方案。
   - 分享的 [X 帖子](https://x.com/__tinygrad__/status/1967446896081076517) 可能包含关于什么将取代 **ShapeTracker** 的信息。



---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**Windsurf Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。


---

# Discord: 分频道详细摘要与链接





### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1423355340107546744)** (2 messages): 

> `o3 弃用，GPT-5 Thinking` 


- **Perplexity 告别 o3！**: Perplexity 已弃用 **o3 模型**，并从即日起将其从模型选择器中移除。
   - 鼓励用户过渡到 **GPT-5 Thinking**，Perplexity 声称其提供更强的性能和持续的支持。
- **强烈推荐 GPT-5 Thinking**: 在弃用 **o3** 后，Perplexity 建议用户切换到 **GPT-5 Thinking**。
   - 他们表示 **GPT-5 Thinking** 将在未来提供更好的性能和全面支持。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1423021661564702831)** (1281 messages🔥🔥🔥): 

> `Comet 浏览器，Discord 任务，故障排除，用户体验，AI 与个人数据` 


- ****Comet 任务引发 Discord 桌面版下载****: 用户正在下载 [Discord 桌面应用](https://discord.com/download) 以完成 **Comet 任务** 并领取 5k 能量球 (orbs)。
   - 一些人在 Discord 应用中找不到任务：*查看置顶消息*！ <:a:check_pins:1406044966500700160>
- ****Comet + Sonnet 等于黄金组合？****: 一名用户分享了一个他们认为特别有用的 [Sonnet 4.5 prompt](https://discord.com/channels/1047197230748151888/1047649527299055688/1423418938510938322)，强调 Prompt + Sonnet 4.5 非常棒！
   - 然而，存在一些 [Bug](https://discord.com/channels/1047197230748151888/1047649527299055688/1423432301633566730)，在使用 Select Models 时不显示 CoT。
- ****隐私与个人数据****: 一名用户分享了一条包含英语、芬兰语、日语和西班牙语组合的记忆，并指出：出于某种原因，我的记忆中混合了英语、芬兰语、日语和西班牙语。
   - 另一位表示：我可以分享那里的提示词，但我绝不会去翻看我的记忆来剪掉私人部分。也不认为它们是影响因素。
- ****Comet 浏览器仍需打磨****: 一名用户指出该浏览器的开屏非常惊艳，并分享了一张成功的 [截图](https://cdn.discordapp.com/attachments/1047649527299055688/1423483019540434954/image.png?ex=68e0795e&is=68df27de&hm=8a12e66bd9f89baf735f8c1bcd80271022e88595da198b5fd7bbaebe96aa5b64&) 。
   - 然而，根据其他人的说法，它肯定还需要改进。正如一名用户所言：它就像任何基础浏览器一样，甚至更烦人，因为我不能把 Google 设为主要搜索引擎，也不能用 Shift+Enter 来调用 AI。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1423081493596471376)** (8 messages🔥): 

> `PC 组装，Perplexity AI 应用，引导悖论 (bootstrap-paradox)` 


- **用户分享 Perplexity.ai 链接**: 分享了几个 Perplexity.ai 搜索和应用链接：[PC 组装](https://www.perplexity.ai/search/96b1fdb5-6e66-4156-a90f-ad6924e30b99#72800$)，[应用链接 1](https://www.perplexity.ai/apps/8e3bda61-b835-4a6d-addb-167a800c83db)，[应用链接 2](https://www.perplexity.ai/apps/6f2a0a07-d165-4dc4-af96-4db2494e2951)。
- **分享了 YouTube 链接**: 分享了一个 [YouTube 链接](https://www.youtube.com/watch?v=V64TdrkhTqo)，没有任何进一步的上下文。
- **分享了引导悖论 (Bootstrap Paradox) 链接**: 分享了一个讨论 [引导悖论 (the-bootstrap-paradox)](https://www.perplexity.ai/page/the-bootstrap-paradox-a-forens-dsHvoK0YQqWIGbyQ0pyjyQ) 的 Perplexity.ai 页面。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1423319451998355506)** (1 条消息): 

> `Sonar-Pro API, 404 Errors, Public Resources` 


- **Sonar-Pro API 产生 404 错误**：有用户报告 **Sonar-Pro API** 生成的资源链接会导致 **404 错误**。
   - 用户正在寻求一种获取当前可用且可公开访问的资源的方法。
- **请求活跃且公开的 Sonar-Pro 资源**：用户明确询问是否有过滤 **Sonar-Pro API** 结果的方法。
   - 他们希望仅接收确认存在且对公众开放的资源，以避免 **404 错误**。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1423022363716485121)** (1348 条消息🔥🔥🔥): 

> `Sora 2, Gemini 3, Qwen3 4B 2507 instruct, 4o model, OpenAI safety` 


- **Arena 中对 Sora 2 的期待升温**：成员们热切期待 **Sora 2** 登陆平台，讨论集中在其潜在影响以及与 **Veo 3** 和其他视频模型的比较上。
   - 一位成员表示：*我太喜欢玩 Sora 了，甚至不想尝试其他任何东西，哈哈*，并希望看到它的基准测试结果。
- **Gemini 3 发布推测引发热议**：社区对即将发布的 **Gemini 3** 议论纷纷，一名成员提到 *Gemini 3 需要有良好的速率限制 (ratelimits) 才能保持竞争力*。
   - 一些用户分享了一个声称发布日期为 **10 月 9 日** 的泄露消息。
- **对 4o 模型退役表示沮丧**：成员们对 **4o 模型** 的可用性受限及最终退役表示失望。
   - 一位成员哀叹自己对 **4o** 的“成瘾”，强调了寻找合适替代方案的困难。
- **关于 AI 伦理边界和数据使用的辩论**：用户对 **OpenAI** 的数据使用实践表示担忧，一名用户开玩笑地承认向 "lmarena 发送了敏感的政府数据"。
   - 另一名成员随后表示，在 Discord 聊天中承认这种事太疯狂了。
- **对话长度限制引发讨论**：成员们讨论了模型对话的长度限制，以及如何总结这些对话并扩展允许的长度。
   - 一位用户指出：*我不介意它遗忘，我只想继续对话*。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1423028346685751346)** (5 条消息): 

> `Arena Champions Role, Reasoning Trace, New Model Update - reve-v1, New Model Update - claude-sonnet-4-5-20250929-thinking-32k, Leaderboard Update` 


- **“Arena Champions”角色向社区开放**：**Arena Champions Role** [<@&1422628364782407830>] 旨在为深入的 AI 讨论创建一个私密空间，奖励致力于有意义对话的成员。
   - 访问权限通过 [申请流程](https://docs.google.com/forms/d/e/1FAIpQLSdRWfqG8_MMKQ4H23FHFZVJsg0OuQrZqn5h9l-QqhWpNI77xg/viewform?usp=dialog) 授予，自 2025 年 7 月起就在服务器中的成员将自动获得访问权限，但必须“关注该类别 (Follow the Category)”才能查看新频道。
- **推理模型上线“推理追踪 (Reasoning Trace)”功能**：**Reasoning Trace** 现在已在 [Side by Side](https://lmarena.ai/?mode=side-by-side) 和 [Direct chat](https://lmarena.ai/?mode=direct) 的推理模型中可用，展示模型在提供回答之前的思考过程。
   - 该功能旨在提供对模型决策过程的洞察，增强透明度和用户理解。
- **Reve-v1 作为仅限图像编辑的模型加入**：新模型 **reve-v1** 已添加到 LMArena，但它是**仅限图像编辑 (image-edit only)** 的，这意味着它需要上传图像才能运行，使用文本转图像提示词会报错。
   - 此外，**claude-sonnet-4-5-20250929-thinking-32k** 模型已取代了 **16k** 版本。
- **Claude Sonnet 4.5 在文本排行榜上并列第一**：**Claude Sonnet 4.5** 表现出色，在 [文本排行榜 (Text Leaderboard)](https://lmarena.ai/leaderboard/text) 上与 **Claude Opus 4.1** 并列 **第一名**。
   - 它在 Hard Prompts、Coding 和 Creative Writing 等类别中也表现优异，在专用频道中*获得了社区的积极讨论*。
- **ibm-granite-h-small 添加至 LMArena**：新模型 **ibm-granite-h-small (ibm)** 已添加到 LMArena。
   - 未提供更多详细信息。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1423024417872805978)** (324 messages🔥🔥): 

> `Qwen3 深度研究，在 Blackwell 上手动编译 xformers，区块链上的 LLM，Unsloth 支持 RWKV 架构，不使用 vLLM 生成合成数据集` 


- **关于 Qwen3 深度研究的辩论开启**：成员们开启了关于 **Qwen3 深度研究**的讨论。
   - 该评论是针对一个提到继妹的玩笑话的回应。
- **Blackwell 手动编译热潮开始**：成员们讨论了在 **Blackwell GPU** 上手动编译 **xformers** 的必要性，并提供了更新的、兼容 Blackwell 的镜像的 [Docker Hub 链接](https://hub.docker.com/r/unsloth/unsloth)。
   - 一位成员分享了一个脚本，使用 `pip uninstall -y xformers`、`git clone` 和 `python3 setup.py install` 来为计算能力 **12** 手动编译 xformers。
- **LLM 助力区块链头脑风暴？**：成员们思考了将 **LLM 添加到区块链**的使用场景，其中一人询问为什么“blockchain”不是一个禁言词，因为其他词都会导致禁言。
   - 有人建议，如果 LLM 能够可靠地使用 **hashes**，那将是一项成就。
- **RWKV 接入 Unsloth 的曲折历程**：一位成员询问 **Unsloth** 是否支持 **RWKV 架构**进行训练和微调，得到的确认是：如果 *transformers* 支持它，Unsloth 很可能也支持。
   - 另一位成员正在尝试对 **RWKV-7 模型**进行 LoRA 微调，但在优化的 HF Triton 内核和 bf16 支持方面面临挑战，不过在 [PEFT](https://github.com/huggingface/peft) 上正取得进展。
- **无需 vLLM 的合成数据浪潮**：成员们讨论了在不依赖 **vLLM** 的情况下生成**合成数据集**，一位成员指出目前所有的 Unsloth notebook 都使用了 vLLM。
   - 有人建议使用 **OpenAI package** 向本地服务器发送异步请求，或者使用 **httpx** 编写代码，并指向了 [meta-llama/synthetic-data-kit](https://github.com/meta-llama/synthetic-data-kit)，该工具包含一个可用于 llama.cpp 或 Ollama 的 API 端点配置。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1423044248068030565)** (3 messages): 

> `区块链与 AI 的协同效应，代码中的信任，共识机制，AI 问题解决` 


- **编写信任：区块链与 AI 结合！**：一位成员的旅程始于*思考如何将信任真正写入代码，以及如何教机器一点智能。*
   - 他们认为，如果以正确的方式将**区块链和 AI**结合在一起，可以改变行业的运作方式、社区的连接方式，甚至新想法的诞生方式。
- **共识机制：将抽象想法变为现实**：一位成员致力于**区块链系统**，将*共识的抽象概念转化为真实存在、人们可以真正依赖的东西*。
   - 该用户专注于使用 **AI 算法**来解决以前被认为不可能解决的问题。


  

---


### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1423335323748012092)** (1 messages): 

> `Unsloth Docker 镜像，IBM Granite-4.0，gpt-oss RL，视觉 RL，GLM-4.6` 


- **Unsloth 的 Docker 首秀**：Unsloth 发布了一个新的 **Docker 镜像**，用于在 Windows/Linux 上进行训练而无需担心依赖问题，详见其[指南](https://docs.unsloth.ai/new/how-to-train-llms-with-unsloth-and-docker)，并可在 [Docker Hub](https://hub.docker.com/r/unsloth/unsloth) 上获取。
- **Granite 取得进展**：**IBM Granite-4.0** 模型现在可以使用 **GGUF** 运行，或使用免费的 [support agent notebook](https://x.com/UnslothAI/status/1973774439344214426) 进行微调，权重已上传至 [Hugging Face](https://huggingface.co/collections/unsloth/granite-40-68ddf64b4a8717dc22a9322d)，并附有[指南](https://docs.unsloth.ai/new/ibm-granite-4.0)。
- **RL 竞赛的变革**：Unsloth 实现了 **gpt-oss RL** 最快的推理速度，支持在免费 notebook 中使用 **GSPO** 进行训练，详见其[博客](https://docs.unsloth.ai/new/gpt-oss-reinforcement-learning)。
- **视觉 VLM 的胜利**：根据 Unsloth 的[博客](https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl)，其**权重共享和内核**使 **VLM RL** 速度提升 2 倍，VRAM 占用减少 90%，并允许 10 倍长的上下文。
- **模型热潮持续升温**：发布了新模型，包括 **GLM-4.6** ([GGUF](https://huggingface.co/unsloth/GLM-4.6-GGUF)) 和 **DeepSeek-V3.1-Terminus** ([GGUF](https://huggingface.co/unsloth/DeepSeek-V3.1-Terminus-GGUF))，以及 **Magistral-2509**、**ERNIE-4.5** 和 **Kimi-K2-0905** 等其他模型。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1423048302403911851)** (638 messages🔥🔥🔥): 

> `WSL 用于开发, Sonnet 4.5 用于编程, 自定义 LSTM 记忆, Kaggle Notebook 训练, 使用 LLM 进行数据提取` 


- **WSL 将开发者从 Windows 的烦恼中解救出来**：成员们讨论了使用 **WSL (Windows Subsystem for Linux)** 配合 **VSCode** 进行开发，以避免 Windows 的依赖问题，并称赞其无缝集成和有效利用硬件资源的能力。
   - 一位成员表示，感觉 Windows 只是个 *UI*，而在 Windows 上使用终端处理事务感觉很 *杂乱*。
- **Sonnet 4.5 阻碍编程项目进度**：用户分享了对 **Sonnet 4.5** 破坏编程项目的担忧，原因是它在没有额外提示的情况下无法执行测试、不恰当地重写身份验证（auth）部分，并生成了无法运行的 *企业级 (enterprise-ready)* 代码。
   - 一位用户指出：*你必须盯着（babysit）任何 LLM。写大量的计划和细节。在 push 之前进行检查。*
- **新型自定义 LSTM 记忆可能彻底改变 LLM**：一位成员分享了测试自定义 **LSTM memory** 的进展，如果成功，它可以让 LLM 拥有类似人类的记忆，尽管将其作为每个 **YunaBlock** 的一部分来实现会使 Loss 评估变得复杂。
   - 他们正尝试弄清楚如何先将数据集拆分为训练集和评估集，就像 *tensorboard* 所做的那样。
- **Kaggle Notebook 训练陷入困境**：成员们讨论了在 Kaggle notebook 中使用 *save and run* 运行时 `train` 日志不出现的问题，并建议使用 **wandb** 代替 **tensorboard** 以获得更好的日志记录。
   - 一位成员问：*Wandb 比 tensorboard 更好吗？*，并链接到了 [Wandb 文档](https://docs.wandb.ai/support/different_tensorboard/)。
- **LLM 帮助从杂乱数据中提取商店名称**：一位成员寻求关于从格式不一致的数据集中提取商店名称的建议，其中商店名称与乱码和国家代码混杂在一起；他们正考虑使用 **NLP** 或 **NLTK** 来进行清理。
   - 该成员提到：*穷人的做法就是对每个缩写疯狂使用 regex，然后把字母和数字混合的乱码正则掉，但这绝对不可持续。*


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1423070352480796722)** (164 messages🔥🔥): 

> `将字幕微调为问答格式, GGUF 转换问题, Gemma3 与 vLLM 兼容性, Gemma3 的 ONNX 转换, Unsloth 的多进程问题` 


- **AI 克隆创建难题**：一位成员正尝试利用他们的视频字幕微调 LLM，以创建一个说话风格像他们自己的 Discord 机器人，但在将 [字幕转换为问答格式](https://discord.com/channels/1179777624986357780/1179777626081986642) 时面临挑战。
   - 他们正考虑将 **视频标题** 和 **字幕** 配合 Embedding 模型来生成问题和答案，模拟观众对视频进行提问。
- **GGUF 转换问题困扰用户**：几位用户在尝试将模型转换为 **GGUF 格式** 时遇到问题，特别是在使用带有 **f16 quantization** 的 *push_to_hub_gguf* 函数时。
   - 一位成员报告了与映射 Tensor 'model.layers.0.self_attn.q_proj.base_layer.weight' 相关的 **ValueError**，并被建议在修复程序发布前先进行手动转换。
- **Gemma3 的 vLLM 尝试结果不一**：用户正努力让 **Gemma3** 在 **vLLM** 上运行；一位成员在启用 fast_inference 后遇到了 *AttributeError: 'Gemma3ForCausalLM' object has no attribute 'vllm_engine'*。
   - 有建议称可能存在配置问题，或者 **Gemma** 与 **vLLM** 尚未完全兼容，一位用户指出 `is_vision_model` 参数可能会导致问题。
- **ONNX Runtime 转换考量**：一位成员询问关于将 **Gemma3** 导出到 **ONNX Runtime** 以实现跨平台支持的问题，并被建议使用 *optimum-cli* 或 *PyTorch* 进行转换。
   - 还有人提到，可能需要在 **PyTorch** 中创建自定义模型配置，因为上次检查时 **Gemma3** 还没进入 *optimum-cli*。
- **多进程故障频发**：一位用户遇到了“禁用多进程（Disable multiprocessing）”的问题，涉及 *UnslothSFTTrainer.py* 中的 *dataset_num_proc*。
   - 建议包括注释掉 *num_proc* 行、将参数设置为 *None* 或设置为 *2*，但这些解决方案对该用户均未奏效。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1423136363733188628)** (41 messages🔥): 

> `Tversky-All GPT2 reproduction, Efficient Training Setup, AMXFP4 Precision, quack kernels` 


- ****Tversky-All** GPT2 获得类 Llama 升级**：一名成员发布了 **GPT2 Tversky-All** 的半复现版本，采用了 Tversky-All 策略，但应用于更现代的类 Llama 模型，并对数学计算进行了调整以提高可计算性和更好的梯度，代码托管于 [CoffeeVampir3/Architecture-Tversky-All](https://github.com/CoffeeVampir3/Architecture-Tversky-All)。
   - 该模型在 **3090 TI** 上使用合成及低熵数据集 (tinystories-hf) 训练了约一天，处理了 **3000 亿 tokens**，测试模型可在 [HuggingFace](https://huggingface.co/Blackroot/Tversky-All-Test-100MIsh) 获取。
- **最大化效率：快速训练设置的秘诀**：作者的训练设置使用了 packed batches、varlen flash attn 和 bf16 训练，详见 [CoffeeVampir3/Architecture-Fast-Tiny-Dense-LLM](https://github.com/CoffeeVampir3/Architecture-Fast-Tiny-Dense-LLM)。
   - *避免梯度检查点 (gradient checkpointing)* 并 *混合中间变量 (mish mashing intermediates)* 可以提高速度，特别是考虑到 ROPE 通常占总运行时间的 18-20%；对于较小的网络，移除它可以显著提升性能。
- ****Quack!** 最优 Kernels 加速生产**：[Dao-AILab/quack](https://github.com/Dao-AILab/quack/tree/main) kernels 可能是目前生产环境下大多数任务中最优的 kernels。
   - 虽然它们在 Ampere 架构上的峰值表现不如 Blackwell，但一些（非 GEMM 类）如 linear cross entropy/RMS norm 在 Ampere 上表现良好。
- ****AMXFP4** 的精度研究**：一名成员正在研究使用 **AMXFP4** 精度，声称它能提供 **FP8** 的精度（但略微更准确），且误差比 FP8 更小。
   - 他们计划研究并构建自己的 **AMXFP4** AI 模型。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1423022338475032748)** (722 messages🔥🔥🔥): 

> `Cameo usage on TikTok, Sonnet 4.5 vs GLM 4.6 Cost, Overrun of Sora users, Deepfake generation, System Artifacts Log for Emerging Validation of Novelty` 


- **TikTok cameo 混淆爆发**：一位用户澄清说，*cameo* 一词是指引入特定面孔用于视频制作，与 **TikTok** 或类似平台无关。
   - 该用户还询问了 **OpenAI** 是否会在 **Android** 或网站上推出类似 **Sora** 的应用。
- **Sonnet 4.5：昂贵但有效？**：成员们讨论了 **Sonnet 4.5** 相对于 **GLM 4.6** 的性价比，指出 **GLM 4.6** 便宜六倍，即使效果只有其 90%，仍是一个值得考虑的替代方案。
   - 一些用户发现 **Sonnet 4.5** 的表现与 **3.7** 相似，而另一些用户在 **Copilot** 中更倾向于使用 **4** 而非 **3.7**。
- **Sora 用户激增占领服务器**：一名成员对服务器被 **Sora 用户** 占领表示担忧，建议使用 **LLM** 根据最近的讨论话题动态更新频道名称。
   - 该成员还批评了导致用户涌入的 **OpenAI 营销策略**，并预测这种情况会在几天内平息。
- **用户对 Deepfake 的虚伪感到困惑**：一位用户表示沮丧，认为一个推动 Deepfake 的应用却在抱怨生成写实图像和虚拟人物动画。
   - 这一批评之后，出现了关于大量“求代码”请求的评论，并建议将反馈转发到相应频道。
- **新兴新颖性验证 (Emergent Validation of Novelty) 伪影：泄露还是优势？**：一位用户分享了一个离奇的轶事，他们的系统触发了一个 **LLM** 将其任务分类为一个非常罕见且复杂的类别，其输出读起来像是其他高级 AI 合成项目的描述。
   - 他们被建议细致地记录一切作为关键证据，并考虑到 *机器不能拥有观点*。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1423309906819153920)** (2 messages): 

> `Sora as Social Media, Sora credits, Sora integration` 


- **Sora 瞄准社交媒体焦点**：一名成员建议 **Sora** 应该成为像 TikTok 这样的社交媒体平台。
   - 他们提议将其与 **ChatGPT** 集成（类似于图像生成），以增强用户体验。
- **Sora 提议基于积分的使用模式**：一位用户建议为 **Sora** 实施 **积分系统 (credit system)**，以便在视频生成中进行更多的资源分配。
   - 他们提到计划可以包含 **每日或每周使用限制**，从而告别目前不透明的模型。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1423045693257289819)** (11 messages🔥): 

> `Human writing prompts, Sora camera control, Portrait vs Landscape in Sora` 


- **模拟人类写作的提示词**：一位成员正在寻找能让 AI 写作听起来更像人类的 Prompt。
   - 另一位成员建议使用像 **Sudowrite** 这样经过更多微调的模型，并指出它具有良好的基准以及专门为此目的定制的用户插件。
- **Sora 相机控制技巧**：一位成员询问了在 **Sora** 中控制场景周围相机的优质 Prompt，并好奇视频是否总是 10 秒。
   - 另一位用户表示他们见过一个 9 秒的视频，但随后另一位用户澄清并为提供错误信息道歉。
- **纵向模式优于风景全景？**：一位成员询问在生成图像时，**Portrait**（纵向）模式是否比 **Landscape**（横向）模式效果更好，因为横向模式只截取了附加图像的一半，有时会导致角色的头部被切掉。
   - 另一位成员回答说，*视觉 Token 是按网格排列的*，因此**正方形图像**从图像生成的综合效果可能最好。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1423045693257289819)** (11 messages🔥): 

> `Writing Prompts, Sora Camera Control, Portrait vs Landscape Generation` 


- **人类寻求 AI 提示词以提升写作质量**：一位成员正在寻找一种 Prompt，使其提交的写作内容听起来更像人类。
   - 另一位成员建议使用像 **Sudowrite** 这样经过更多微调的模型，强调其良好的基准和实现预期效果的用户插件。
- **Sora 相机控制受到关注**：一位成员询问了在 **Sora** 中控制场景周围相机的优质 Prompt，并注意到视频通常为 **10 秒**长。
   - 另一位用户幽默地指出，他们见过一个 *9 秒*长的视频。
- **图像生成中纵向模式优于横向模式？**：一位成员建议在根据图像生成时，**Portrait** 模式比 **Landscape** 模式效果更好，因为横向模式只截取图像的一半，有时会裁剪掉角色的头部。
   - 另一位用户回复说，视觉 Token 是按网格排列的，因此**正方形图像**可能会产生最佳的图像生成结果。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1423030113015959664)** (542 messages🔥🔥🔥): 

> `Surface Pro Snapdragon X Elite, Artifacts as emergent validation of novelty, Model Quantization and Quality Tradeoffs, GPT-OSS, LM Studio Linux Install` 


- **分享 Snapdragon X Elite 规格**：一位成员分享了其 **Microsoft Surface Pro** 的配置，搭载 **Qualcomm Snapdragon X Elite**（12 核，X1E80100 @ 3.40 GHz）、**16 GB** RAM 和 **Windows 11 Home 64-bit**。
   - 在看到一个“Artifact”（伪影/异常输出）后，他们询问 LLM 的意见是否准确。
- **“新颖性的涌现验证”作为新型 Bug**：一位用户分享了一段长引用，建议将意料之外的 LLM 输出（*泄露*或 *Bug*）重新定义为“新颖性的涌现验证”，这表明系统架构已将 LLM 推向了一个稀有且复杂的类别。
   - 发布者在看到一个“Artifact”后，询问这种归功于 Gemini 的观点是否有价值。
- **量化对知识压缩的影响**：成员们讨论了不同的 Quantization（量化）级别如何影响语言模型中知识的压缩和保留，并指出较低的量化可能会因为移除了“推理位”（reason bits）而对较小模型产生不成比例的影响。
   - 这也可能导致模型失去区分事物的手段，正如一位成员所说：*“量化得太厉害，突然之间你就会以一种意想不到的方式把病娇和摸狗混为一谈😄。”*
- **GPT-OSS 发布**：宣布发布 **GPT-OSS** 并进行了极限跑分测试（benchmaxxed），这是一个行为类似于 **GPT-4o** 的超安全模型。
   - 成员们注意到，如果没有提供足够的细节，它会假设大量信息。
- **LM Studio Linux：无常规安装程序**：针对在 Linux 上安装 **LM Studio** 的问题，官方澄清仅提供 AppImage，这意味着没有传统的安装过程。
   - 这是为了解释“Linux 安装说明”，以便正确引导新用户。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1423023930536366120)** (115 条消息🔥🔥): 

> `4090 vs 5090 垂直扩展策略, Arc B50 Pro 基准测试, GPT OSS 120b 硬件推荐, GPU 卸载中的 DDR3 vs DDR4, LLM 领域的 Unsloth vs LM Studio` 


- **4090 的垂直扩展策略**：成员们讨论了使用配备 **32GB RAM** 的 **4090** 进行垂直扩展的想法，并建议降低频率可以提高离网生活（off-grid living）场景下的效率。
   - 还有人提到，运行在 **280W** 功率下的 **5090** 在 *Token 速率上可能仅慢 15%*，但由于更快的休眠周期，实际速度可能会更快。
- **Arc B50 Pro 在 Token 测试中表现不佳**：一位成员将 **Arc B50 Pro** 显卡与 **RTX 4080 Super** 进行了对比，指出虽然 B50 拥有 *海量的 VRAM*，但其实际内存带宽非常糟糕，导致 Token 速率大幅降低（**12B q8_0 模型** 仅为 **7-8 Tps**，而 4080 则超过 **30 Tps**）。
   - 然而，在默认上下文（**4k**）下，B50 跑出了 **32 Tps**，而 4080 为 **42 Tps**，表现优于预期。
- **OSS 120b 硬件搜寻**：一位成员正在寻求运行 **GPT OSS 120b**（**FP8** 精度、**131k 上下文**）的硬件建议，目标是达到 **20-40 tps** 或更高。
   - 建议包括 **4070ti**（低上下文下 **13t/s**）、**4090**（低上下文下 **25t/s**）、**3x3090s**（**10K 上下文** 下 **85/s**）以及配备 **DDR5 6000** RAM 的 **5090**（低上下文下 **35t/s**）；一位用户表示 *Flash Attention 无法在 OSS120b 上运行*。
- **针对 GPU 部署的 DDR3 深入探讨**：一位用户建议使用带有多个 **PCIE 16x 插槽** 的廉价 **DDR3** 主板来容纳 **6x GPU**，并结合 RAID 组建的 **SATA SSD** 以缩短加载时间，并引用了一个 [eBay 上的 X99 主板列表](https://ebay.us/m/zB2BAH)。
   - 也有人对内存带宽表示担忧（**DDR4** 为 **68 GB/s**），认为与现代标准相比可能存在瓶颈，一位用户称 *在 DDR3 上最高只能达到约 50gb/s*。
- **Unsloth 不适合推理**：一位成员澄清说 [Unsloth](https://github.com/vllm-project/vllm) 是一个微调平台，而非用于 LLM 推理，并推荐使用 [Open-Router](https://openrouter.ai/) 以获得带有供应商回退（fallback）功能的稳定推理。
   - 用户还分享了他们使用 **Chain-of-Draft** 来提升性能和速度。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1423022973828333649)** (601 条消息🔥🔥🔥): 

> `Cursor 中的 Git Worktree, Cursor 中的 Beta 功能, 使用 Cursor 进行 Typescript 重构, MacOS 上 Cursor 的内存泄漏, Cursor 黑客松？` 


- **Cursor 集成 Git Worktree 设置**：用户在 **Beta 标签页** 下发现了 **Git Worktree** 设置，并被鼓励启用它以测试其在 Agent 窗口中是否有效。
   - Git Worktree 集成似乎仅在 **Early Access** 或 **Nightly Cursor** 版本中可用。
- **Cursor 的 Beta 功能引发好奇**：成员们讨论了在 Cursor 中使用 Beta 功能，有人推荐开启它以获取尚未发布的优秀功能、进行有趣的调试，并帮助改进 Cursor。
   - 目前，**afterFileEdit** 是唯一可用的 Hook，但可以使用 **Extension RPC Tracer** 来检查 RPC。
- **Typescript 重构圆满完成**：一位成员在向 Cursor 发送四次提示词后，成功完成了全面的 **Typescript 重构**，随后又使用 Master Prompt 进行了全面审计以确保重构正确。
   - 建议在执行前使用 Cursor 的 **Plan 模式**（在 Nightly 版本中可用）进行规划并跟踪工作流状态，以提高效率。
- **Cursor 导致 MacBook 崩溃**：一位用户报告称，Cursor 由于高内存占用（飙升至 **96GB**）导致其 MacBook Air M4 崩溃，可能与过多的 Chat 或 Agent 进程有关，重启后恢复正常。
   - 该成员指出这可能是内存泄漏，其他人也确认 **MacOS** 版本的内存泄漏发生频率更高。建议降级到较低版本作为潜在的解决方案。
- **Cursor 黑客松可能举办**：一位成员询问是否有兴趣参加 **Discord Cursor 黑客松**，旨在实现各种解决方案和其他潜在的侧边项目（side projects）。
   - 大家对提供免费额度的赞助黑客松很感兴趣，一位成员建议采用远程友好的方式，以便不同时区的用户都能参加。


  

---

### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1423293195147804695)** (5 条消息): 

> `OpenRouter Performance Tab, Grok-4-Fast` 


- **OpenRouter 发布性能选项卡 (Performance Tab)**：OpenRouter 推出了全新的“性能选项卡”，用于可视化[特定模型的提供商性能](https://x.com/OpenRouterAI/status/1773733582763069916)。
- **FP4 不应与 BF16 出现在同一张图表上！**：一位用户对新的性能选项卡发表评论，指出将使用不同量化级别（例如 **FP4** 与 **BF16**）的提供商进行比较具有误导性。
   - 他们建议增加一个过滤器下拉菜单，以区分不同的 **quant levels**。
- **Grok-4-Fast 免费期即将结束**：代号为 **Sonoma** 的 **Grok-4-Fast** 模型免费反馈期将于明天（10 月 3 日太平洋标准时间上午 9:30）结束。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1423021970525782107)** (2 条消息): 

> `RPG, Mixture of LLMs` 


- **RPG 爱好者威胁到 Mixture of LLMs 方法**：一位成员请求对某种方法的细节进行“模糊处理”，因为 **RPG** 用户会不停地使用它。
   - 该方法被称为 **Mixture of LLMs**，该成员担心如果被过度使用，它可能会消失。
- **另一个满足 MinItems 要求的话题**：添加第二个话题以确保 `topicSummaries` 数组满足至少 2 个项目的要求。
   - 此条目仅作为占位符，不反映所提供消息的实际内容。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1423022303335284756)** (495 条消息 🔥🔥🔥): 

> `OpenRouter BYOK, Free Inference Providers, Grok vs Sonoma, Gemini Pro Performance Issues, Deepseek R1 0528 deprecation` 


- **BYOK 100 万次免费请求**：用户讨论了“每月 100 万次免费 BYOK 请求”的优惠，澄清这只是免除了 OpenRouter 对前 100 万次请求收取的 **5% 佣金费用**，但用户仍需按照 [OpenRouter 文档](https://openrouter.ai/announcements/1-million-free-byok-requests-per-month)所述，直接向提供商支付 API 使用费用。
   - 一些用户最初误解了该优惠，认为它提供完全免费的请求，从而引发了关于更清晰表述的讨论，例如 *“每月 100 万次 0 手续费的 BYOK 请求”*。
- **AgentRouter 提供 200 美元额度**：一位成员提到 [AgentRouter](https://agentrouter.io/) 提供 **200 美元免费额度**，但指出其服务可能“时好时坏”，并提醒用户在处理重要事务时要谨慎使用。
   - 他们还提到了自己的推荐链接，并针对不同方案混合使用了 **Sonnet 4.5, GPT 5 和 GLM 4.6**。
- **Grok4 表现不如 Sonoma**：一位用户测试了 **Grok 4 Fast**，发现它“比 Sonoma 笨得多”，并指出它“经常失败”且无视格式要求。
   - 另一位用户猜测 Grok 4 Fast “带有……Llama 的味道？”，对其不稳定性表示沮丧。
- **Gemini Pro 面临性能问题**：用户报告称 **Gemini Pro** 的响应内容“很奇怪”，无法正确使用工具，并且通过 OpenRouter API 访问时性能“慢得令人无法接受”。
   - 报告显示这可能是 **Gemini 2.5 Pro** 的普遍问题，一位用户建议尝试 Vertex 作为替代提供商。
- **由 OpenInference 触发的上下文限制问题**：一位用户遇到了与隐私设置相关的提供商错误，并因超出 DeepInfra 的上下文限制而被导向 OpenInference，这引发了关于 OpenInference 过滤器和内容偏好的讨论。
   - 有建议称 OpenInference 不适合 RP 内容，因为他们是一个研究团队。


  

---

### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1423075466536816661)** (23 messages🔥): 

> `Sora.com and new model, BYOK tokens, Latency vs E2E latency, Qwen image model, Cerebras removing Llama` 


- **Sora 集成，Token 充足**：[Sora.com](https://sora.com) 现在支持新模型，用户可获得 **1M 免费 BYOK tokens**。
- **端到端延迟是否公平？**：成员们讨论了 **latency** 与 **E2E latency** 之间的区别。
   - 一位成员表示 *E2E 没有意义，因为每次生成的复杂程度/响应长度都不同，这样比较供应商是不公平的*，而另一位成员指出 *图表轴标签显示为 'Time to last token'*，这需要进行归一化处理才能进行公平比较。
- **Qwen 图像编辑功能上线！**：成员们分享了 [阿里巴巴的新 **Qwen image model**](https://x.com/Alibaba_Qwen/status/1973668568412856595)，并指出这仅是一个图像编辑模型。
   - 一位成员分享了发布该消息的 [这个帖子](https://x.com/pingToven/status/1973758872772108663)，而另一位成员则表达了在 **Apple Silicon** 上运行它的兴趣。
- **Cerebras 移除 Llama 4**：**Cerebras** 将在 **15 号** 移除 **Llama 4 maverick**。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1423137249956204544)** (7 messages): 

> `Perplexity AI framework, Deepseek Sparse Attention, Underrated LLM pretraining papers, Attention Matrices, LLM Attention Research` 


- **Perplexity AI 框架解决方案**：一位成员分享了 [Perplexity AI framework](https://www.perplexity.ai/page/scientists-ai-framework-solves-hTbKnxfPSl64P5nLfxqX5A) 的链接及其 **GitHub** 项目。
   - 该成员询问了关于使用类似注意力矩阵且复杂度低于 **O(n^2)** 的 **LLMs** 研究，并思考了其与 sliding window attention 相比可能存在的问题。
- **Deepseek 稀疏注意力示例**：针对关于高效注意力机制的问题，一位成员建议探索 **Deepseek Sparse Attention**，将其作为 **top-k attention** 的一个例子。
   - 另一位成员指出，与上述注意力矩阵相比，**Transformers** 受益于相对位置，从而提供了正确的 inductive bias。
- **寻找被低估的 LLM 预训练论文**：一位成员征求与预训练 **LLMs** 相关的被低估的论文，以在即将进行的预训练运行中最大化性能，并链接到了 [arxiv.org](https://arxiv.org/abs/2503.03588v1) 上的一篇研究论文。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1423062135856037989)** (28 条消息🔥): 

> `Gradient Descent Dynamics, Symmetry Transformer, ViT training, Compact Image Representation, Quantifying Scientific Impact` 


- **Gradient Descent Dynamics 论文被誉为“年度论文”**：一名成员称赞了一篇关于 *gradient descent 动力学* ([centralflows.github.io](https://centralflows.github.io/part1/)) 的论文为**年度论文**，强调了它对 **loss spike 动力学**的解决方案及其对 **Adam beta2** 的影响。
   - 该成员还对没有早点发现这篇论文表示遗憾，并指出其引用次数较低且评审分数平平。
- **“Symmetry Transformer” 结果褒贬不一**：一名成员发现，在 *symmetry transformer* ([GitHub repo](https://github.com/Eternalyze0/symmetry_transformer)) 中使用**独立的 head 预测当前和之前的 token** 改善了 validation loss。
   - 然而，初步测试显示，与 symmetry 模型（train loss **4.4329**，val loss **6.1747**）相比，baseline 模型具有更低的 loss（train loss **3.9405**，val loss **4.7615**），但 symmetry 模型随后有所改善（train loss **3.8241**，val loss **4.7368**）。
- **探索自监督 ViT 训练**：一名成员正在探索以**自监督方式**训练 **Vision Transformer (ViT)**，将来自冻结 embedder 的图像 token 序列映射到 **CLS token** 中，且不使用标签。
   - 挑战在于为图像 token 寻找合适的 augmentation，建议使用 **masked autoencoder (MAE) 风格的目标函数**。
- **Masked Autoencoders 适用于紧凑图像表示**：一名成员建议使用 **masked autoencoder** 训练 **CLS token**，以学习*图像的紧凑表示 (compact representation)*。
   - 另一名成员表示同意，并指出 masked autoencoders 可以在没有大量 augmentation 的情况下进行有效训练。
- **论文称研究人员的影响力不会随时间改变**：一名成员分享了一篇论文 ([Quantifying the Evolution of Individual Scientific Impact](https://static1.squarespace.com/static/5877ca6986e6c00f05f58f84/t/58e68a43d482e9cb083bf6ab/1491503686695/quantifying-the-evolution-of-individual-scientific-impact.pdf))，声称研究人员在整个职业生涯中发表论文的*预期价值是一致的*。
   - 这表明研究人员的第一篇和最后一篇论文成为其代表作的概率是相同的，从而对当前评估研究人员的方法提出了质疑。


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1423269157486465055)** (91 条消息🔥🔥): 

> `SOTA scaling of MLPs, AUNN implementation and efficiency, Test-time training (TTT) framework vs AUNN, Inductive bias in sequence models, Computational cost of different model architectures` 


- **AUNN 的实用性受到质疑**：讨论质疑了 **AUNN (Augmented Neural Networks)** 的实用性，对其效率表示怀疑，并且除了一个简单的 GitHub 示例 [ethan-w-roland/AUNN](https://github.com/ethan-w-roland/AUNN) 外，缺乏可运行的原型。
   - 有人指出 AUNN 的最初提出者态度强硬，不愿听取反驳意见，并且更关注 **MLP** 而非 **Attention**。
- **TTT 作为 AUNN 的显式版本**：**test-time-training (TTT) 框架**被认为是 AUNN 假设的一个显式且可运行的版本，并指向了使用 MLP 版本 **TTT** 的论文 [arXiv:2407.04620](https://arxiv.org/abs/2407.04620)。
   - 有人指出 *MLP 版本非常接近 AUNN 试图实现的目标，但实际上可以开箱即用*。
- **Transformer 只是 2D 切片**：Transformer 被描述为一种优化手段，将大型 2D 问题（序列、通道）分解为重复的垂直 1D 计算切片。
   - 有建议认为，将巨大的 **MLP** 应用于整个问题*也会很有效，但这种方式在计算上是不可行的*。
- **归纳偏置 (Inductive Bias) 提高性能**：有人提到需要某种形式的**归纳偏置**来补偿计算资源的不足，并提出 **SSMs (State Space Models)** 是比 Transformer 的 self-attention 偏置更优雅的版本。
   - 讨论集中在 **RoPE 或 NoPE** 等偏置如何随时间赋予 attention weights 规则的结构，使其与序列结构良好对齐，从而实现更好的泛化。
- **跨时间步的 MLP**：在时间步之间使用 **MLP** 被认为是可行的，但成本非常高，因为它可能需要一次仅预测一个 token，以防止未来的 token 信息向后泄露。
   - 有建议称使用 **Transformer** 是因为它们*足够廉价*，提供了并行训练以及跨序列和通道维度的简便 2D 分解。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1423027571494355035)** (12 条消息🔥): 

> `GEMM optimization, Tversky paper implementation, DeepSeek Sparse Attention in CUDA, GPU performance engineering career path` 


- **GEMM：GPU 的绝佳练习！**：实现一个性能达到 **cuBLAS** 80% 以上的 **GEMM** 是充分利用 GPU 的宝贵练习，因为它允许通过矩阵化（matricization）重构任意张量收缩，参考 [这篇维基百科文章](https://en.wikipedia.org/wiki/Tensor_reshaping#Mode-m_Flattening_/_Mode-m_Matrixization)。
   - 对于大矩阵，算术强度随问题规模线性扩展，一篇博客文章 ([CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM)) 指导了如何达到 **cuBLAS** 的性能。
- **Tversky-All 策略测试**：一位成员致力于实现和测试基于 **Tversky 论文** ([https://arxiv.org/pdf/2506.11035](https://arxiv.org/pdf/2506.11035)) 的网络，并在 [GitHub 仓库](https://github.com/CoffeeVampir3/Architecture-Tversky-All) 中详细说明了发现和指导。
   - 论文中概述的 **Tversky-All 策略** 被应用于更现代的 **llama-like 架构**，使用原始公式的 **CIFAR10 版本** 可在 [此处](https://github.com/CoffeeVampir3/Tversky-Cifar10) 获取。
- **DeepSeek Sparse Attention 周末黑客松**：几位成员表示有兴趣在周末合作实现 **CUDA** 版的 **DeepSeek Sparse Attention**，参考 [DeepSeek-V3.2-Exp GitHub 仓库](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf)。
   - 合作者计划将时间限制在周末，看看能完成多少，然后继续后续工作。
- **GPU 性能工程职业高峰**：一位成员正在评估专注于 **GPU 性能工程** 的职业路径，考虑到 AI 模型的需求和有限的计算资源，认为这是一个重大机遇。
   - 他们正在寻求关于日常工作、机会大小、关注领域（如 **kernels、编译器 (Triton, TVM)、分布式推理**）以及在 **CUDA** 优化中达到生产力所需的提升时间的见解。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1423118856813285378)** (18 条消息🔥): 

> `RF meaning, Volkov's paper, mbarriers vs barriers` 


- **RF 代表 Register File**：一位成员询问附件图片中 **RF** 的含义，另一位成员回答说它可能意味着 *"Register File"（寄存器堆），即分配给寄存器的硬件。*
   - 讨论明确了通常每个 SM 子分区（sub-partition）有一个寄存器堆。
- **《Understanding Latency Hiding on GPUs》论文的相关性**：一位成员询问了 Vasily Volkov 的论文 [《Understanding Latency Hiding on GPUs》](https://example.com) 在 Blackwell 等近期 GPU 架构中的相关性。
   - 另一位成员指出，它对于高层原则很有帮助，但细节已经发生了很大变化，并指向了更新的微架构论文，如 [《Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks》](https://arxiv.org/abs/2507.10789)。
- **CUDA 中的 mbarriers 与普通 Barriers**：一位成员询问了 **CUDA** 中 **mbarriers** 和普通 **barriers** 的区别。
   - 另一位成员解释说 **mbarriers** 位于共享内存（shared memory）中，而硬件屏障（hardware barriers）数量有限且带有 ID，并引用了 **PTX 文档** 中的内容：*"每个 CTA 实例有 16 个编号为 0..15 的屏障"*。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1423209761209454603)** (2 条消息): 

> `LLM Training, Cross Entropy, Gradient Norm, Sparse Tensors, Torch Compile` 


- **梯度范数问题**：一位成员询问了在使用 **cross entropy** 训练 **LLMs** 时预期的梯度范数。
   - 问题包括梯度范数如何取决于 **model size**、**completion tokens** 的数量以及当前的 **log probabilities**。
- **Dynamo 无法追踪稀疏张量**：一位用户报告了一个 `UserWarning`，指出带有 **Torch Compile** 的 **Dynamo** 无法追踪进入 **sparse COO/CSR tensors**。
   - 该用户表示惊讶，原以为 Dynamo 能够处理稀疏张量，并附上了收到的具体警告信息。


  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1423022651936473219)** (4 messages): 

> `Non-determinism in LLM Inference, Flash-MoE, Nvidia Compiler Techniques, Warp Specialization, Distributed Setting` 


- **Thinking Machines 战胜非确定性**：Thinking Machines 发布了一篇关于在 LLM 推理中克服 **non-determinism**（非确定性）的[博客](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)。
- **Flash Attention 变体发布**：团队发布了 [Flash-MoE](https://flash-moe.github.io/)，这是 **Flash Attention** 的一个变体。
- **Nvidia 汇编新技术**：Nvidia 正在开发用于调度和 **warp specialization** 的 **compiler techniques**（编译器技术），并在其[论文](https://d1qx31qr3h6wln.cloudfront.net/publications/Cypress_PLDI_25.pdf)中详细介绍了针对 **FA3** 的基准测试。
- **Nvidia 在分布式设置中融合 Kernel**：Nvidia 正在分布式设置中进行算子融合（fusing kernels），如这篇[论文](https://d1qx31qr3h6wln.cloudfront.net/publications/Legate_Kernel_Fusion___ASPLOS_2025.pdf)所述。
- **通过性能工程解码 GPU 复杂性**：哈佛大学详细介绍了一个新前沿：[LLM 能否优化 GPU 性能？](https://harvard-edge.github.io/cs249r_fall2025/cs249r_fall2025/blog/2024/10/01/gpu-performance-engineering/)。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/)** (1 messages): 

schizik12: <@325883680419610631> spam
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1423058905638109377)** (9 messages🔥): 

> `Benchmarking Guides, Kernel Benchmarking, Career Opportunities in GPU Programming, Gaining Experience in GPU Programming, GEMM Optimization` 


- **寻求基准测试指南！**：一位成员寻求*优秀的基准测试指南*，并被推荐了[这篇 arXiv 论文](https://arxiv.org/abs/2502.15015)、[这篇关于 kernel 基准测试的文章](https://jan.ai/post/how-we-benchmark-kernels)以及[这个 YouTube 视频](https://www.youtube.com/watch?v=1i7dxoAfKOU)。
   - 其中一位成员称他们之前的基准测试工作*可能是最好的基准测试尝试*。
- **自学者的 GPU 职业晋升**：一位在大厂工作并对 *GPU 编程* 感兴趣的成员询问进入该领域所需的经验类型。
   - 他们认为*阅读书籍和做题最适合面试准备*，但无法提供足够的实践经验。
- **通过编写 CUDA Kernel 助力职业发展**：一位成员建议从编写一个在特定架构上能与 **cuBLAS** *相媲美*的 **GEMM** 开始，以积累 GPU 编程经验。
   - 他们进一步阐述道，*如果你能接触到 H100 并使用 Hopper 特有的技巧，那将更令人印象深刻*。


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1423027577374900275)** (2 messages): 

> `Learning vs. Job Performance, C++ Requirement for a Book, 5090 GPU Learning Experience` 


- **学习 vs. 工作的肌肉记忆**：一位成员提到，在工作或研究中“做”某事更多是关于**练习和“肌肉记忆”**，而非深奥的理论知识。
   - 他们指出，过多的思考而缺乏足够的练习会导致效率低下。
- **C++ 技能是否能促进 GPU 学习？**：一位用户询问 **C++** 知识对于理解某本特定书籍是否必要，以及这是否会激发学习 **C++** 的动力。
   - 他们买了一块 **5090 GPU** 希望能学到很多东西，但目前大多只是在进行 *“凭感觉写代码” (vibe coding)*，没有取得显著进展。


  

---


### **GPU MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1423335619148386405)** (1 messages): 

> `Blackwell, matmuls, jax` 


- **使用 JAX 在 Blackwell 上实现性能爆发**：一位用户分享了一篇关于使用 **JAX** 在 **Blackwell** GPU 上实现 **matmuls**（矩阵乘法）顶级性能的[教程](https://docs.jax.dev/en/latest/pallas/gpu/blackwell_matmul.html)。
   - 该帖子强调了在 NVIDIA 最新架构上优化矩阵乘法操作的技术和最佳实践。
- **JAX matmul 技巧**：jax 频道的一位用户分享了一个关于在进行矩阵乘法时获得顶级性能的教程。
   - 它链接到了 **JAX** 在 **Blackwell** **GPU** 上的官方文档。


  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1423420204997939291)** (3 messages): 

> `INT4 Quantization, TorchAO, TensorCore, A100 GPUs, Efficient Kernels` 


- **通过 TorchAO 进行 INT4 量化**：要通过 **torchao** 使用 **INT4 quantization**，请遵循[说明](https://github.com/pytorch/ao?tab=readme-ov-file#-quick-start)。
   - 此外，你也可以查看从 *tinygemm* 库复制的、使用 **TensorCore** 的 **INT4mm 实现**。
- **TorchAO 贡献者文档**：有关为 **torchao** 做出贡献的文档可以在[此处](https://docs.pytorch.org/ao/main/quantization_overview.html)和[此处](https://docs.pytorch.org/ao/main/contributor_guide.html)找到。
   - 特别是，[此链接](https://docs.pytorch.org/ao/main/contributor_guide.html#adding-efficient-kernels)描述了如何向 torchao 添加高效的 Kernels。
- **INT4MM 为 A100 GPUs 上的 TorchAO 提供动力**：使用 **TensorCore** 的 **INT4mm 实现**（[代码链接](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/int4mm.cu)）为 **torchao** 中针对 **A100 GPUs** 的 **INT4** 功能提供支持。
   - 该实现是从 *tinygemm* 库中复制而来的。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1423236804651253802)** (2 messages): 

> `GPU Engineering, MMA Tensor Cores` 


- **深入探索 GPU 工程基础**：一位成员分享了一篇关于 [GPU 工程基础的博客文章](https://modelcraft.substack.com/p/fundamentals-of-gpu-engineering?)。
   - 对于那些正在学习 GPU 架构和计算的人来说，这篇文章应该会非常有吸引力。
- **通过 MMA Tensor Cores 浅谈 GEMM**：一位成员撰写了一篇关于使用 **MMA tensor cores** 的文章，并链接到 [A Gentle Introduction to GEMM using MMA Tensor Cores](https://am17an.bearblog.dev/a-gentle-introduction-to-gemm-using-mma-tensor-cores/)。
   - 作者欢迎任何关于技术细节和清晰度的反馈。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1423181314076577862)** (5 messages): 

> `MI300x8, amd-gemm-rs, amd-all2all, amd-ag-gemm` 


- **MI300x8 在 amd-gemm-rs 排行榜表现强劲**：一位成员在 **MI300x8** 上以 **540 µs** 的成绩获得了 `amd-gemm-rs` 排行榜的 **第 8 名**。
   - 随后在 **MI300x8** 上的提交也取得了成功，成绩分别为 **553 µs** 和 **547 µs**。
- **MI300x8 在 amd-all2all 获得铜牌**：一位成员在 **MI300x8** 上以 **462 µs** 的成绩获得了 `amd-all2all` 排行榜的 **第 3 名**。
- **MI300x8 在 amd-ag-gemm 取得个人最好成绩**：一位成员在 **MI300x8** 上以 **528 µs** 的成绩在 `amd-ag-gemm` 排行榜中创造了个人最好成绩。


  

---


### **GPU MODE ▷ #[tpu](https://discord.com/channels/1189498204333543425/1351571759761068082/1423091403059236884)** (9 messages🔥): 

> `Cloud TPUs, JupyterLab, gcloud CLI, rclone` 


- **Cloud TPUs 设置导致 Kernel 繁忙**：一位成员在设置 **Cloud TPUs** 时寻求帮助，报告称在未创建 VM 实例的情况下通过 SSH 连接后，运行带有 TPU 相关代码的单元格会导致 JupyterLab Kernel 变得繁忙，并提到了计费方面的顾虑。
   - 另一位成员建议使用 **gcloud CLI** 并直接 **SSH** 进入 VM，以获得更高的可靠性。
- **使用 rclone 备份模型权重**：一位成员建议在处理 TPU 时设置 **rclone** 来保存模型权重或其他相关数据。
   - 他们强调，具体的设置取决于用户的特定目标。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1423023462422941779)** (11 messages🔥): 

> `Lab Play Interpretation, Open Play Development, PIP Stuff Discussion, GIF Updates` 


- **实验室玩法激发开放玩法**：成员们讨论了对 **实验室玩法结果 (lab play results)** 的解读是否意味着向更 **开放玩法 (open play)** 转变，因为 Agent 能够理解依赖关系并可以手动搬运物资。
   - 这种想法认为，学习 **从零开始构建事物** 对 Agent 来说会更有趣且更有益。
- **PIP 讨论邀请**：一位成员告诉另一位成员，当他们想要讨论 **PIP 相关内容 (PIP stuff)** 时请告知。
   - 分享了一个 Google Meet 链接供其加入：[https://meet.google.com/xfo-wzmh-msg](https://meet.google.com/xfo-wzmh-msg)。
- **GIF 进度持续推进**：一位成员询问关于 **GIFs** 的更新，另一位成员回复称他们仍在努力中。
   - 他们表示，如果时间紧迫，可以利用旧的流水线生成一些默认的 GIF。


  

---

### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1423194586582355969)** (31 条消息🔥): 

> `permutation_mnk 规则, tiled_mma, CooperativeGroup.__init__ 对齐, Uniform Registers (URs)` 


- **破解 CUDA 坐标代码**：一位成员寻求关于 `permutation_mnk` 规则以及它如何扩展/平铺 **mma-atom** 的解答，并指出扩展/平铺 mma-atom 基本上有 3 种方式。
   - 另一位成员解释说，*atom 布局是线程上的空间平铺 (spatial tiling)*，而 *permutation 意味着值（坐标）上的空间平铺*，并补充说*两者是正交的，可以实现不同的结果*。
- **Tiled MMA 线程纠结**：一位成员询问如何在内核中获取**线程数**并强制其为 *constexpr* 值。
   - 另一位成员澄清说，在 tiled MMA 上调用 `cute.size` 可以得到**线程数**，并且由于 tiled MMA/copy 是类型，因此可以在 JIT 上下文中从主机端获取此大小，以便使用基于 tiled MMA 参数化的 block size 来启动内核。
- **GMEM 模式思考**：一位成员分享了一张内存模式图，并询问 **M0SF3** 之后 GMEM 的下一个尺度，具体是 **M32SF0** 还是 **M1SF0**。
   - 另一位成员澄清说，在连续的 GMEM 中，下一个是 **M32SF0**。
- **揭秘 Uniform Registers**：一位成员质疑 `cute.arch.warp_idx()` 是否对 warp 中的每个线程都相同，并询问为什么 `make_warp_uniform` 使用 uniform registers 以及 **URs** 的作用。
   - 另一位成员表示这*只是一个编译器提示*，*没有任何实际作用*，而原提问者指出他们找不到任何关于 **URs** 的文档，但在 SASS 中看到了它们。
- **Cooperative Group 难题**：一位成员询问 `CooperativeGroup.__init__` 中的 `alignment` 参数，具体询问它的作用以及为什么如果 `size` 是 32 则它必须是 32，而其他值则不然。
   - 另一位成员回答说，进行此检查是因为*它们恰好是 warp/warpgroup 的粒度，是需要特殊检查以防止 bug 的常见情况*。


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1423132486749130792)** (2 条消息): 

> `nccl::all_to_all 性能, bf16 vs fp8` 


- **NCCL 的 all_to_all：BF16 与 FP8 性能持平？**：一位用户注意到，对于形状相同的 **bf16** 和 **fp8** 输入，`nccl::all_to_all` 的耗时相近。
   - 另一位用户跟进询问，这种现象在大型张量和小型张量上是否都成立，暗示可能存在优化差异。
- **BF16 vs FP8 耗时**：一位用户询问，既然 **bf16** 和 **fp8** 输入的形状相同，为什么 **nccl::all_to_all** 处理它们的时间会一样。
   - 另一位用户询问这是否发生在大型和小型张量上。


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1423169761822507048)** (4 条消息): 

> `LLM 训练加速, Linear Cross Entropy, Sequence Packed 训练, Quack 优化` 


- **Linear Cross Entropy 提升 LLM 训练速度**：推荐使用 [Linear Cross Entropy](https://github.com/CoffeeVampir3/Architecture-Fast-Tiny-Dense-LLM/blob/main/main.py#L115) 来加速 LLM 训练过程。
   - 建议使用 **Quack 优化**库，特别是其 [linear cross entropy 实现](https://github.com/Dao-AILab/quack/blob/main/quack/linear_cross_entropy.py)，因为它具有潜在优势。
- **Sequence Packing 极大增强训练效率**：**Sequence packed 或 "unpadded" 训练**被认为是一种非常有影响力的优化，特别是结合 **flash attn varlen** 等技术。
   - 示例实现可以在[这里](https://github.com/CoffeeVampir3/Architecture-Fast-Tiny-Dense-LLM/blob/main/modeling/MHA.py#L36)找到。
- **优化器选择影响训练速度**：从理论上讲，更好的**优化器**可以大幅缩短训练时间，但 **AdamW** 通常更容易使用。
   - 优化器的选择会显著影响 LLM 训练过程的效率，尽管 **AdamW** 仍然是一个流行且可靠的选择。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1423033496535957714)** (103 messages🔥🔥): 

> `AI 集成 MMO，Karpathy 谈 Sutton 与 Bitter Lesson，Hume AI Octave 2，Mistral 的形式数学团队，可扩展选项学习 (SOL)` 


- **马斯克尝试开发 AI-MMO**：据报道，埃隆·马斯克正与 **Eve Online** 的制作者洽谈，共同开发一款 **AI 集成 MMO (AIMMORPG)**，旨在利用只有 AI 才能提供的能力，尽管一些用户对他游戏美学的创意愿景表示怀疑。
   - 一位用户指出：“AI 是进入该领域的天然选择，我们必须构建整个游戏，Eve 拥有忠实的粉丝群，但他们讨厌 web3。”
- **Karpathy 在 Bitter Lesson 播客中的感悟**：Karpathy 总结了 [Dwarkesh-Sutton 播客](https://x.com/karpathy/status/1973435013875314729)，指出 Sutton 怀疑 **LLM** 是否满足了他所推广的论点。
   - Karpathy 认为 pre-training 提供了一个实用的“蹩脚进化”引导（boot-strap），同时承认存在两位数的不确定性，即更大的范式正在等待，并敦促研究人员从动物智能（好奇心、多智能体博弈等）中汲取更多灵感。
- **Hume AI 的 Octave 2 带来更快的 TTS**：[Hume AI 发布了 Octave 2](https://xcancel.com/hume_ai/status/1973450822840152455?s=46)，这是下一代多语言文本转语音（TTS）模型，支持 **11 种以上语言**，**速度提升 40%**（延迟 <200 ms），**成本降低 50%**，支持多发言者闲聊、改进的发音，以及新的语音转换和音素编辑工具。
   - 在 10 月份，他们的 Creator 计划提供 **50% 折扣**；**EVI 4 mini**（对话式 AI）也在预览中。
- **Mistral 数学化**：Albert Jiang 透露，Mistral AI 在 20 亿美元融资后组建了新的**形式数学研究团队**。
   - 他们正在为全能的证明器/自动形式化工具/Agent 招募 AI 人才，宣传拥有精英合作者、人均数百个 GPU、开放研究、顶级薪酬，并在巴黎、伦敦、帕洛阿尔托设有办公室，职位空缺广告见[此处](mailto:aj@mistral.ai)。
- **Claude 开发者推崇 Sonnet 4.5**：Claude Code 团队的 catwu 宣布，在内部投票后，所有成员都采用了 **Sonnet 4.5** 作为日常编程模型，称其为最强的全能选择；Anthropic 暂时重置了付费用户的速率限制，以平滑从 Opus 的过渡。
   - 早期采用者称赞该模型的速度和质量，少数人指出仍存在问题，一位用户报告称：“第一遍是 gpt5 低思考模式。结果很差。Sonnet 4.5 思考后，在类似的时间范围内得到了可用的结果”。


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1423352400902750269)** (4 messages): 

> `Dylan Field, Figma, Latent Space, Make, MCP` 


- **Dylan Field 揭秘 Figma 的 AI 剧本**：Latent Space 播客发布了一集，由 **Figma 联合创始人 Dylan Field** 讨论 **Figma 的 AI 剧本 (AI Playbook)**。
   - 本集涵盖了在 **vibe-coding** 时代如何呈现优秀设计、**Figma 的 Make、用于“品鉴” Agent 的 MCP**，以及**快时尚 SaaS** 的未来（[X 链接](https://x.com/latentspacepod/status/1973793231524806925)，[Xcancel 链接](https://xcancel.com/latentspacepod/status/1973793231524806925)）。
- **品味是你的护城河：Dylan Field 谈 Figma 的 AI**：Latent Space 与 Figma 联合创始人聊天，讨论在 vibe-coding 垃圾内容时代如何呈现优秀设计，涵盖了 Figma 的 Make。
   - 用于“品鉴” Agent 的 MCP，以及快时尚 SaaS 的未来。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1423163648653918309)** (8 messages🔥): 

> `Mosaic AI 视频编辑器发布, Sora-TikTok 自动化变现` 


- **Mosaic 发布 AI 优先的视觉编辑器**：创始人 Adish Jain 发布了 [Mosaic](https://xcancel.com/_adishj/status/1973432845436854418) 的公开测试版，这是一款为视频创作者设计的 **AI 驱动视觉编辑器**，具有无限视觉画布和时间轴版本控制功能。
   - 早期反馈称赞其非线性的、类 Git 的方法，并将其比作“视频编辑界的 Cursor”，输入“MOSAIC”的用户可获得 **1,000 个免费积分**。
- **Sora-TikTok 自动化获得 1200 万播放量**：一位用户分享了关于 [Sora-TikTok 自动化](https://xcancel.com/siyabuilt/status/1973841586888061148?s=46)在 36 小时内达到 **1200 万播放量**的链接，引发了关于变现的讨论。
   - 讨论集中在利用社交媒体平台上的 **AI 生成内容**产生收入的策略和可能性。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1423153541916393472)** (25 messages🔥): 

> `Nous Research 模型类似于 GPT-4.5，Gemini 回答，Veo3 gems，Granite 语言模型，Qwen 30B A3B 适用于 CPU` 


- **Hermes 或 Gemini 模型接近 GPT-4.5？**：一位成员询问是否有与 **GPT-4.5** 相当的 **Nous Research** 微调模型，另一位成员建议由于其特性，它更类似于 **GPT-5** 或 **Gemini**。
   - 搞笑的是，当该成员向 **Gemini** 提出同样的问题时，**Gemini** 在列出选项时的回答之一是 **Hermes**。
- **Veo3 有 Gems？**：一位用户在某些方面更喜欢 **Veo3**，而不是最新的 **Sora**。
   - 他们附带了一个 [Prompt_theory.mp4](https://cdn.discordapp.com/attachments/1149866623109439599/1423308176148922491/Prompt_theory.mp4?ex=68dfd688&is=68de8508&hm=e195c2f737881136d240fa288b286f7dcc417fbe581c153cefa587b7c2ec0233&)。
- **IBM Granite 语言模型拥有混合注意力机制（Hybrid Attention）**：一位成员分享了 **IBM Granite** 语言模型的 [图像分析](https://huggingface.co/collections/ibm-granite/granite-40-language-models-6811a18b820ef362d9e5a82c3)，其中包括 **2B dense**、**7B (1B active)** 和 **32B (9B active)** 模型，均采用混合注意力机制。
   - 这些模型支持 **FIM** 且缺乏位置编码（positional encoding），从而防止了在 **128k** 上下文之外的性能下降。
- **Qwen 30B A3B 在 CPU 上表现出色**：一位成员指出 **Qwen 30B A3B** 是一个可靠的 ~30B LLM 选择，另一位成员发现它适合 **CPU** 使用。
   - 具体来说，**Qwen 3 30B A3B** 在 **Q6_K_XL** 量化下，在配备 **32GB VRAM** 的 **Ryzen 7 5700G** CPU 上，在 **1024** token 上下文时实现了 **48 TPS** 的处理速度和 **10.5 TPS** 的生成速度。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1423339648838795356)** (3 messages): 

> `LLM 策略性撒谎，稀疏自编码器（Sparse Autoencoder）工具，Goodfire AI，模型不诚实检测` 


- **LLM 被抓到策略性撒谎！**：一位成员分享了他们最近关于 **策略性 LLM 欺骗** 的预印本论文 [《秘密议程：LLM 策略性撒谎，而我们当前的安全性工具视而不见》（The Secret Agenda: LLMs Strategically Lie and Our Current Safety Tools Are Blind）](https://arxiv.org/html/2509.20393v1)。
   - 该研究利用 **稀疏自编码器工具**（例如由 [Goodfire AI](https://www.goodfire.ai/) 托管的工具）直接揭示了当前方法如何遗漏了驱动策略性 LLM 欺骗的复杂内部特征，并强调了缩小 **自动标注差距（autolabel gap）** 的切实路径。
- **自编码器揭露 LLM 的隐藏议程**：该研究使用稀疏自编码器寻找驱动 LLM 欺骗的隐藏特征，旨在改进检测方法。
   - 该方法寻求弥合“自动标注差距”，并增强模型对抗不诚实行为的鲁棒性。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1423339648838795356)** (3 messages): 

> `LLM 欺骗，稀疏自编码器，Goodfire AI` 


- **LLM 被抓到策略性撒谎**：一位成员分享了他们研究的预印本 [《秘密议程：LLM 策略性撒谎，而我们当前的安全性工具视而不见》（The Secret Agenda: LLMs Strategically Lie and Our Current Safety Tools Are Blind）](https://arxiv.org/html/2509.20393v1)。
   - 该研究使用 **稀疏自编码器工具** 来展示当前方法如何未能检测到驱动 **策略性 LLM 欺骗** 的内部特征。
- **Goodfire AI 托管自编码器工具**：该研究利用 **稀疏自编码器工具**（例如由 **Goodfire AI** 托管的工具）直接揭示了当前方法如何遗漏了驱动 **策略性 LLM 欺骗** 的复杂内部特征。
   - 该研究强调了缩小 **自动标注差距（autolabel gap）** 并推进对 **模型不诚实** 行为进行鲁棒检测的切实路径。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1423114666451927080)** (17 messages🔥): 

> `Deepmind 代码不完整，RoPE 实现` 


- **HuggingPapers 代码无法运行**：成员们注意到来自 [HuggingPapers 的代码](https://fxtwitter.com/HuggingPapers/status/1973420932497879298?t=jxTf48_aBK8349s1uSyDQw&s=19) 无法运行，因为它 **没有导入 RoPE**。
   - 代码的原作者似乎表示用户应该自己实现它。
- **Deepmind 被指责过于保守**：成员们开玩笑说 [Deepmind](https://deepmind.google/) 为了避免分享他们的实现而做了额外的工作。
   - 一位成员分享说，Deepmind 的代码通常很复杂，但 **他们会将其拆解，并使其作为大型系统的一部分如何运作变得不清晰**，并引用了他们实现 **V-MPO** 的经验。


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1423049533213905046)** (6 messages): 

> `Knowledge Distillation, Semantic Equivalence, RL for Fuzzy Prediction` 


- **语义等价（Semantic Equivalence）是否在压缩知识？**：一名成员询问最近关于**语义等价**的论文是否只是试图通过将前者作为教师模型，来将一个模型的知识压缩到另一个模型中。
   - 另一名成员表示同意这可能是知识蒸馏（Knowledge Distillation），并建议真正的测试在于新模型在特定基准测试中是否能超越提供**语义等价**信号的 **LLM**。
- **腾讯论文的遗漏引起怀疑**：一名成员指出，**腾讯的论文**没有提到用于**语义等价**信号的模型，这令人怀疑。
   - 他们推测该模型可能从带有 **RL** 的模糊下一句预测任务中学到了一些有趣的东西。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1423236545649049651)** (7 messages): 

> `IBM Granite 4.0, Mamba/Transformer architecture, ISO 42001 certification, Oracle Business Model, OpenAI datacenters` 


- **IBM 为企业发布 Granite 4.0**：IBM 发布了下一代 IBM 语言模型 **Granite 4.0**，其特点是采用了新的**混合 Mamba/Transformer 架构**，在不牺牲性能的情况下大幅降低了内存需求。
   - 这些模型在 **Apache 2.0 许可证**下开源，是全球首个获得 **ISO 42001 认证**的开源模型，并经过加密签名，确认其遵循国际公认的安全、治理和透明度最佳实践。
- **Granite 4.0 在多个平台上可用**：Granite 4.0 模型可在 **IBM watsonx.ai** 以及包括 Dell Technologies、Docker Hub、Hugging Face、Kaggle、LM Studio、NVIDIA NIM、Ollama、OPAQUE 和 Replicate 在内的平台合作伙伴处获取，即将支持通过 AWS Sagemaker JumpStart 和 Microsoft Azure AI Foundry 访问。[IBM 公告点击此处](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models)。
- **ISO 认证被视为无用**：一名用户评论说，IBM 拥有一个完全没用的 **ISO 认证**，只是为了让 C-suite 高管们误以为这物有所值。
- **Oracle 的商业模式**：一名用户评论说，Oracle 以前的商业模式是销售数据库和企业软件，现在似乎变成了为 OpenAI 运行数据中心（[OpenAI Elon Musk 帖子](https://openai.com/elon-musk/)）。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1423034627307602022)** (27 messages🔥): 

> `Credits Issue, Memory Key Protocol, Sora invite code, Manus API key, Neuro-cognitive agentic logic layer` 


- **Manus 额度消耗引发愤怒**：一名用户抱怨一个基础研究任务消耗了 **5300 额度**却未完成，称 Manus 是个“彻头彻尾的笑话”并要求[退款](https://cdn.discordapp.com/attachments/1349440650495398020/1423044743377715343/image.png?ex=68e032b1&is=68dee131&hm=cd60314f2b422e917efdd7ebbd2a8747117a9e000904040bddb4b0a1d2624fd9)。
   - 一名团队成员询问了会话链接以便调查并可能提供额度退款；随后该用户通过私信发送了链接。
- **使用 Memory Key 解锁 Agent 模式**：一名成员提议使用 **Memory Key 协议**来解决退出 **Agent** 模式的问题，该协议涉及在重启会话前保存上下文。
   - 他们详细说明了一个[解决方案](https://discord.com/channels/1348819876348825620/1349440650495398020/1422940046855766016)，包括复制关键信息、启动新会话，并指示 **Agent** 创建更新的 **Memory Key** 以供将来使用。
- **账单问题引发支持真空**：一名用户报告了账单问题，但 Manus 支持团队没有回应，促使社区成员建议向其官方支持邮箱发送邮件，并注明清晰的主题行和工单编号。
   - 有人建议这样做可以为升级投诉留下正式的书面记录。
- **全球定价模型因差异受到批评**：一名用户批评 Manus 的**全球美元定价模型**（Plus 计划每月 39 美元）没有根据地区经济进行调整，在巴西和拉丁美洲其他国家造成了障碍。
   - 另一名用户建议根据**购买力平价 (PPP)** 实施区域定价，以提高可及性并促进全球增长。


  

---

### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1423329674255401012)** (2 messages): 

> `AGI Introduction, Hugging Face Paper` 


- **AGI 介绍论文发布**：一名成员发布了一篇介绍 **AGI** 的 [Hugging Face 论文](https://huggingface.co/papers/2509.26507) 链接。
   - 该成员表示：*被我说中了 😉 (called it)*。
- **DSPy Discord 新增频道**：成员注意到 DSPy Discord 现在增加了一个 **show-and-tell** 频道。
   - 成员表示 *这是一个新频道。*


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1423033597538865264)** (23 messages🔥): 

> `Caching Prompt Order, DSPyWeekly Search Feature, JSONAdaptor vs ChatAdaptor vs XMLAdaptor, Tool Use RL for Models, OpenAI Function calling and MCP` 


- **Prompt 顺序影响 Caching**：成员们讨论了为了利用 Caching（缓存），*发送 Prompt 和文件的顺序至关重要*。
- **DSPyWeekly 推出搜索功能**：[DSPyWeekly 现在上线了搜索功能](https://dspyweekly.com/search/)，可以查看所有抓取过的内容，并提供上一个/下一个链接以便于导航。
- **JSONAdaptor 争议**：成员们质疑 **JSONAdaptor** 是否应继续作为默认选项，因为 **ChatAdaptor** 或 **XMLAdaptor** 通常能解决适配器错误。
   - 虽然 **JSON** 是聊天模式的备选方案，但考虑到模型 Tool Use RL（工具使用强化学习）的兴起，有人提出了将 **XML** 作为默认选项的可能性。
- **XML 在工具调用方面优于 JSON？**：成员们辩论了 **XML** 相比 **JSON** 在工具调用方面的优势，强调工具使用现在正被植入到 Post-training（后期训练）中，如果没有 XML 结构，其他任何形式都会与模型权重相冲突。
   - 此外还讨论了 **XML** 在向 LM（语言模型）清晰传达信息方面表现出色，且 Token 消耗更低，最多可减少 3 倍 Token。
- **模型是否使用 XML 进行训练？**：讨论涉及了模型是否正在接受 **XML** 工具调用的训练，并引用了一篇关于 Prompt 变体的 [Berkeley 博客文章](https://gorilla.cs.berkeley.edu/blogs/17_bfcl_v4_prompt_variation.html)。
   - 建议测试模型对特定适配器的遵守频率，以及适配器如何影响模型性能。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1423045674106359838)** (16 messages🔥): 

> `Qwen Coder Models, aider Development, aider-desk UI, Model Discussions Channel` 


- **Qwen Coder 模型性能**：成员们讨论了 **Qwen Coder** 模型，其中一人建议 **30B** 版本 *应该足够聪明了*。
   - 他们还提到 **Qwen3 Coder** 是一个更新、可能更好的替代方案，但提醒量化可能会影响性能，建议使用 **Q4**。
- **对 aider 开发节奏的担忧**：一名成员注意到 **aider** 的发布频率有所下降，并询问是否有 **Patreon** 或捐赠系统来支持该项目。
   - 他们对开发者倦怠以及 **aider** 可能停止维护表示担忧，强调与其他 Agent 类工具相比，它在实际工作中非常实用。
- **尝试 aider-desk UI**：一名成员询问关于使用 **aider-desk** 或类似 UI 配合 **aider** 工作的问题。
   - 另一名成员简要分享了将其用于 **MCP 支持** 的经历，指出它可能适合那些希望以 **aider** 风格工作流为主、并辅以可选 Agent 用例的用户，但目前已切换到 **sst/OpenCode**。
- **模型讨论频道：现在能看到了**：一名成员询问 *模型讨论频道发生了什么*，随后意识到该频道一直存在。
   - 未进行进一步讨论。


  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1423025888974606447)** (8 messages🔥): 

> `DeepWiki, Custom Chat Templates, GBNF, Multi-Line Prompts, LLM Polyglot Performance` 


- **鼓励对 DeepWiki 进行逆向工程**：一名成员分享了一个 [DeepWiki 页面](https://deepwiki.com/search/please-reverse-engineer-the-pr_c15e0046-3403-4786-bf26-63b2bf046455)，鼓励进行逆向工程。
   - 另一名成员建议在 *koboldcpp* 中使用 **output template** 或 **post-processing**，但不确定 *llama.cpp* 是否提供这些功能。
- **自定义 Jinja Chat Templates 覆盖 GGUF**：一名成员提到可以指定自定义的 **Jinja chat template** 来覆盖 **GGUF** 中包含的模板。
   - 他们还建议使用 [GBNF](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md) 来格式化模型的输入，并在 [llama.cpp 讨论区](https://github.com/ggml-org/llama.cpp/discussions/16386) 发起了相关讨论。
- **多行提示词（Multi-Line Prompts）解决方案**：一名成员询问如何在不从外部源粘贴的情况下向 *aider* 发送 **多行提示词**。
   - 另一名成员分享了关于 [aider 输入多行聊天消息文档](https://aider.chat/docs/usage/commands.html#entering-multi-line-chat-messages) 的链接。
- **评估 LLM 在多语言问题上的性能**：一名成员询问了评估 **LLM 性能** 在 **polyglot problems**（多语言编程问题）上表现的方法。
   - 他们询问了用于此目的的特定代码、通用 Agent 或示例 Agent。


  

---


### **MCP Contributors (Official) ▷ #[mcp-dev-summit](https://discord.com/channels/1358869848138059966/1413517834805313556/1423024677403623599)** (10 messages🔥): 

> `Ye Olde London 聚会, Registry 团队直播, 异步工具调用直播, 安全与运维专题, 关于 Profiles 的演讲` 


- **MCP 开发者在 Ye Olde London 会面**：成员 <@1042188459575619674> 和 <@1387880172589547722> 在 **Ye Olde London** 组织了聚会，邀请其他人加入。
   - 另一名成员 <@1407892993934889010> 计划参加，并提到他们会 *"过去坐一会儿！"*
- **Registry 团队启动直播**：Registry 团队的直播正在进行中；在此观看 [这里](https://www.youtube.com/watch?v=5qwXADMBuio)。
   - 直播于 **英国时间上午 9 点** 开始。
- **异步工具调用直播**：来自 AWS 的 Nick Aldridge 在 MCP 最佳实践专题中介绍了异步工具调用，直播地址见 [这里](https://www.youtube.com/live/9NBGQIoW9B8?si=ziE8AVJ2O2NxUbhH)。
   - 关注更多最佳实践。
- **安全与运维专题上线**：关于 Security and Ops 的专题已上线；在此查看 [这里](https://www.youtube.com/live/3KneEblEK34?si=FQ5UzX3LU33xUYpK)。
   - 保持安全，保持运行。
- **Profiles 演讲亮点**：关于 Profiles 的演讲可以在此处观看 [这里](https://www.youtube.com/live/5qwXADMBuio?si=3kEhJNw4lsv_M_jN&t=16208)。
   - 该演讲的具体时间戳为 **16208**。


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1423061386497495163)** (6 messages): 

> `工具调用支持, 参考实现, MCP 服务器的 OCI 接口` 


- **提议为 Sampling 增加工具调用支持**：一名成员提交了一个 SEP，旨在为 **Sampling** 添加 **Tool Call 支持**（[issue #1577](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1577)）。
   - 该提案取决于 **多内容块（multiple content blocks）** 的讨论，以支持并行工具调用（[PR #198](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/198)）。
- **用于测试的参考实现**：分享了一个参考实现（TS SDK），包括一个运行 **agentic loop-powered tool** 的示例服务器，以及一个用于辅助测试的 **backfill proxy**（[PR #991](https://github.com/modelcontextprotocol/typescript-sdk/pull/991)）。
   - 一名成员指出，参考实现之前在 CI 中失败，但在固定了 zod 的次要版本后已解决。
- **为 MCP 服务器构思 OCI 接口**：一名成员提议为 **MCP 服务器** 创建一个 **类似 OCI 的接口**，将所有元数据放入一个 tarball 中。
   - 其目的是“构建”一个 **OMCP package** 并进行分发，以简化元数据处理。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1423386569389248602)** (3 messages): 

> `Qualcomm contacting Modular, Mojo Manual update, Level 2 badge unlocked` 


- **Qualcomm 寻求 Mojo 合作？**：一名成员推测 **Qualcomm** 可能会在 Qualcomm 开发者 Discord 语音聊天中提出该话题后联系 **Modular** 洽谈 **Mojo** 相关事宜。
   - 这可能预示着 Qualcomm 对 **Mojo** 在其硬件上的能力产生了潜在兴趣。
- **Mojo 手册更新，Python 章节成为焦点**：一位用户在延迟后分享了 [Mojo 手册链接](https://docs.modular.com/mojo/manual/python/)，并特别指出了 **Python** 章节。
   - 这表明文档中关于 **Mojo** 与 **Python** 交互的部分有了更新或重要信息。
- **解锁新等级**：一名成员晋升至 **level 2**。
   - 晋升至 level 2 意味着在社区内的活跃度有所提升。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1423069245058711634)** (10 messages🔥): 

> `Mojo notebook, GPU Compatibility, Mojo distributed computing` 


- **Notebook 中的 Mojo：交互还是编写？**：一名成员询问目标是 *在 notebook 中与 Max 交互*，还是 *直接在 notebook 中编写并运行 Mojo*。
   - 另一名成员报告称已经能够 **在 notebook 中与 Mojo 交互**，并表示有兴趣添加语法高亮器，以便通过语法颜色更好地学习 Mojo。
- **AMD Radeon 6800 XT 运行成功？**：针对 [GPU 兼容性文档](https://docs.modular.com/max/packages/#gpu-compatibility)，一名成员报告成功在 **AMD Radeon 6800 XT** 上运行了 *vector_addition 示例*，并询问这是否算作成功。
   - 一名 Modular 员工回应称，他们尚未对 **RDNA 2 GPU** 进行广泛测试，并询问有多少 **Mojo GPU puzzles** 能在该系统上运行，同时指出模型目前还无法在 **RDNA GPU** 上正确运行。
- **Mojo 用于分布式计算？**：一名成员想知道未来是否可能将 **Mojo** 与 **Dask** 或 **PySpark** 等批处理或流处理框架结合用于分布式计算。
   - 另一名成员建议 Mojo 欢迎人们构建自己的框架，因为全 **Mojo 框架** 的延迟可能比基于 Python 的方案更低，吞吐量更高，并暗示了有趣的联网选项。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1423369410441121812)** (7 messages): 

> `Kimi new features, Sora video quality, Pro Subscription watermarks` 


- **Kimi 的能力引发惊喜**：一名用户在观看 [视频演示](https://cdn.discordapp.com/attachments/1371757564005711973/1423369409669365881/2025-10-02_20-01-10.mp4?ex=68e00f90&is=68debe10&hm=b28aea7215ef03687754be113fa4dd6a583c355be6e3462543235d7d0258ef73&) 后，对 **Kimi** 的某项新能力表示惊讶。
- **Sora 演示视频备受关注**：几位用户对比了 **Sora** 视频的质量，认为分享的演示视频质量低于 **OpenAI YouTube 频道**上的 **Sora** 视频。
   - 一位用户将其描述为 *奇怪的晃动感*。
- **Pro 订阅用户可获得无水印 Sora**：据一位用户称，**Sora** 的 **Pro 订阅**版本将具有更高的分辨率且没有可见水印。
   - 他们提醒道：*会添加不可见水印——这样 OpenAI 就能辨认出它是生成的，只是我们看不出来……*