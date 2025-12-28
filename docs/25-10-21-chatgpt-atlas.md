---
companies:
- openai
- google
- langchain
- ivp
- capitalg
- sapphire
- sequoia
- benchmark
date: '2025-10-21T05:44:39.731046Z'
description: '**OpenAI** 为 macOS 推出了基于 **Chromium 分支的 AI 浏览器 Atlas**，其特色是集成了 **智能体（Agent）模式**
  和具备本地登录功能的浏览器记忆，旨在超越 **谷歌 Chrome 中的 Gemini**。此次发布在可靠性和隐私方面引发了褒贬不一的反应。


  **LangChain** 以 12.5 亿美元的估值完成了 **1.25 亿美元的 B 轮融资**，并发布了 **v1.0 智能体工程栈**。该项目已获得广泛采用，包括每月超过
  **8500 万次的开源下载量**，以及约 35% 的财富 500 强企业的使用。此外，生态系统还迎来了 **vLLM 支持 MoE LoRA 专家微调**等更新。'
id: MjAyNS0x
models:
- gemini
- atlas
people:
- kevinweil
- bengoodger
- fidjissimo
- omarsar0
- yuchenj_uw
- nickaturley
- raizamrtn
- hwchase17
- bromann
- casper_hansen_
- corbtt
title: ChatGPT Atlas：OpenAI 的 AI 浏览器
topics:
- agent-mode
- browser-memory
- chromium
- finetuning
- moe
- lora
- agent-runtime
- observability
- software-development
- funding
---

**Chromium 就是你所需的一切。**

> 2025/10/20-2025/10/21 AI 新闻快报。我们为您检查了 12 个 subreddits、544 个 Twitters 和 23 个 Discords（198 个频道，7709 条消息）。预计节省阅读时间（以 200wpm 计算）：564 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 方式展示所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻详情，并在 @smol_ai 上向我们提供反馈！

正如 [7 月份泄露](https://www.reuters.com/business/media-telecom/openai-release-web-browser-challenge-google-chrome-2025-07-09/?utm_source=chatgpt.com)（以及更早之前）的消息，OpenAI 终于发布了他们的 Chromium 分叉 AI 浏览器 Atlas（目前仅限 MacOS，其他平台即将推出 - [在此下载/访问网站](https://chatgpt.com/atlas)）：


![ChatGPT Atlas 浏览器主页，蓝色背景，浏览器界面展示了一个航班预订详情屏幕](https://resend-attachments.s3.amazonaws.com/TWkFeQSXfOp82Qh)


集成非常精美且令人印象深刻，正如你在 [直播](https://youtu.be/8UWKxJbjriY) 的后半部分所见。通过将 Agent 模式引入 Atlas，OpenAI 不仅仅是在追赶 [Chrome 中已有的 Gemini](https://www.google.com/chrome/ai-innovations/)，而是通过重启 [Operator](https://news.smol.ai/issues/25-01-23-ainews-openai-launches-operator-its-first-agent) 并将其置于本地浏览器而非远程服务器，迈出了超越它的明显下一步，这样它就可以使用你的登录信息。

氛围是积极的，但并不完全如此：


![一条推文显示用户撤回了之前关于 ChatGPT Atlas 精致程度的陈述](https://resend-attachments.s3.amazonaws.com/AlsVRpYiGbyCh6t)


该你了，Google。

---

# AI Twitter 摘要

**OpenAI 发布 ChatGPT Atlas 浏览器**

- **Atlas 搭载了 Agent 模式和“浏览器记忆”**：OpenAI 为 macOS 推出了一款 AI 优先的浏览器，系统级嵌入了 ChatGPT，具有可选的页面/上下文记忆功能，以及一个预览版的“Agent 模式”，可以在网页上执行操作（包括经许可后在已登录的网站上操作）。macOS 版现已推出；Windows、iOS 和 Android 版“即将推出”。参见来自 [@OpenAI](https://twitter.com/OpenAI/status/1980685602384441368) 的发布帖子、[Agent 模式详情](https://twitter.com/OpenAI/status/1980685612538822814) 和 [产品说明](https://twitter.com/OpenAI/status/1980685615340614032)。PM 们通过 [@kevinweil](https://twitter.com/kevinweil/status/1980698941885935707)、[@bengoodger](https://twitter.com/bengoodger/status/1980692301010858350) 和 [@fidjissimo](https://twitter.com/fidjissimo/status/1980682244185608392) 强调了用例和 UX 意图。提供了一个类似无痕模式的记忆开关 ([@omarsar0](https://twitter.com/omarsar0/status/1980688230904144086))。
- **早期反应**：“浏览器是新的操作系统”这一构想已经落地 ([@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1980685683707842974), [@nickaturley](https://twitter.com/nickaturley/status/1980694337643315475))，但可靠性和隐私权衡问题也立即浮现。在一场与 Perplexity 的 Comet 的正面交锋中，Atlas 更稳健地完成了一项繁琐的成绩跟踪任务（上下文处理、更快的动作以及“类人”的探索）([@raizamrtn](https://twitter.com/raizamrtn/status/1980695747227210213))。其他人则称目前的 Agent 模式为“slop”（废料），并提出了数据访问方面的担忧 ([@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1980846874904219932), [隐私](https://twitter.com/Yuchenj_UW/status/1980847565819302116))。发布初期的流量一度让服务过载 ([@TheTuringPost](https://twitter.com/TheTuringPost/status/1980700012372955352))。

**LangChain 获得 1.25 亿美元 Series B 融资并发布 v1.0 Agent 工程栈**

- **融资 + 产品里程碑**：LangChain 完成了由 IVP 领投的 1.25 亿美元 B 轮融资，CapitalG、Sapphire、Sequoia、Benchmark 等机构参投，公司估值达到 12.5 亿美元。与此同时，它发布了 LangChain 和 LangGraph 的 1.0 版本、一个 LangSmith 洞察 Agent 以及一个无代码 Agent 构建器（[@LangChainAI](https://twitter.com/LangChainAI/status/1980678921839603948), [@hwchase17](https://twitter.com/hwchase17/status/1980680421706006663), [IVP note](https://twitter.com/tomloverro/status/1980714285140701362)）。团队强调了受控的、生产优先的 Agent 运行时和可观测性，并在 LangChainJS 中推出了新的 createAgent 抽象 + 中间件（[@bromann](https://twitter.com/bromann/status/1980683275682091024), [release notes](https://twitter.com/chester_curme/status/1980685592544571897)）。使用数据声明：每月 OSS 下载量超过 8500 万次，约 35% 的财富 500 强企业正在使用该技术栈（[@veryboldbagel](https://twitter.com/veryboldbagel/status/1980686379613815295), [@amadaecheverria](https://twitter.com/amadaecheverria/status/1980687050174287876)）。
- **生态系统适配**：vLLM 增加了 MoE LoRA 专家微调支持（[@casper_hansen_](https://twitter.com/casper_hansen_/status/1980525929026973904)），并称一项外部分析是其推动力（[@corbtt](https://twitter.com/corbtt/status/1980678250608443467)）。多个团队强调了 LangGraph/LangSmith 在 Agent 可靠性和评估（evals）方面的生产环境应用（[@Hacubu](https://twitter.com/Hacubu/status/1980683912096674144), [@jhhayashi](https://twitter.com/jhhayashi/status/1980690375326278107)）。

**Vision Tokens、OCR 与新 VLM：DeepSeek-OCR、Glyph、Qwen3-VL、Chandra OCR**

- **DeepSeek-OCR（文本即图像）引发辩论**：该论文报告了通过将文本渲染为图像并利用 vision encoder + MoE decoder 进行解码，实现了大规模的长上下文压缩。评论既有热烈的技术解析（97% 的重建精度，且“视觉”token 减少了约 10 倍；高分辨率卷积压缩器）（[@rasbt](https://twitter.com/rasbt/status/1980642191950090585)），也有针对遗漏先前技术（pixels-for-language 和视觉 token 压缩路线）的尖锐批评（[@awinyimgprocess](https://twitter.com/awinyimgprocess/status/1980506449706119642), [@NielsRogge](https://twitter.com/NielsRogge/status/1980559120760791125)）。其他人则认为核心启示是当前 embedding/token 使用的低效，而非图像本身具有优越性（[@Kangwook_Lee](https://twitter.com/Kangwook_Lee/status/1980709454522744902)）。
- **智谱（Zhipu）的类 “Glyph” 方向及通过 vision tokens 实现的 KV**：多人注意到智谱发布了同期的 vision-token 压缩方法（“Glyph”），声称在长上下文 QA/摘要任务中，实现了 3-4 倍的上下文压缩和填充成本降低，且质量没有下降（[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1980722682246398069), [context](https://twitter.com/teortaxesTex/status/1980642000006451348)）。细节尚不明确；关注类 BLT 扩展以进一步提升解码效率。
- **Qwen3-VL-2B/32B**：阿里巴巴发布了稠密型 2B 和 32B VLM，包括 FP8 变体以及 “Thinking”/Instruct 类型，声称在 STEM、VQA、OCR、视频、Agent 任务中相比 GPT-5 mini 和 Claude Sonnet 4 取得显著优势；32B 版本旨在以极高的显存效率在 OSWorld 上媲美更大的模型（[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1980665932625383868)）。Demo 已迅速上线 HF（[@_akhaliq](https://twitter.com/_akhaliq/status/1980690335220351063)）。
- **开源 OCR**：Chandra OCR 发布，支持完整的布局提取、图像/图表描述、手写识别和表格支持；可与 Transformers/vLLM 配合使用（[@VikParuchuri](https://twitter.com/VikParuchuri/status/1980667137606971423)）。

**训练/推理栈更新：PyTorch, vLLM, FlashInfer, Providers**

- **Meta PyTorch 发布了新库**：torchforge（可扩展的 RL 训练）、OpenEnv（Agent 环境）和 torchcomms，此外在“训练未来”路线图（预训练→后训练→推理）中，Monarch 和 TorchTitan 也备受关注 ([@eliebakouch](https://twitter.com/eliebakouch/status/1980637130687942805), [stack summary](https://twitter.com/eliebakouch/status/1980642834404319388), [Monarch](https://twitter.com/finbarrtimbers/status/1980681034359533861))。
- **vLLM 与内存**：kvcached 支持在同一块 GPU 上服务多个共享未使用的 KV cache 块的模型 ([@vllm_project](https://twitter.com/vllm_project/status/1980776841129701411))；该项目在 PyTorch Conference 上亮相 ([@vllm_project](https://twitter.com/vllm_project/status/1980622348903674022))。
- **FlashInfer-Bench**：全新的“自我改进”基准测试工作流，旨在标准化 LLM 服务算子（kernel）签名，并自动识别最快算子，以便在 FlashInfer/SGLang/vLLM 中实现零日（day-0）集成 ([@shanli_xing](https://twitter.com/shanli_xing/status/1980705452699926851))。
- **GLM-4.6 (Reasoning) 的供应商基准测试**：Baseten 在输出速度（104 tok/s）和最快首个回答 Token 时间（TTFT）方面领先；各供应商定价集中在输入 $0.6/M，输出约 $2/M；全部支持 200k 上下文和 Tool Calling ([@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1980777360724226282))。

**研究、评估与方法**

- **通过内存层进行持续学习**：与全量微调/LoRA 相比，稀疏微调的内存层能够实现针对性更新，且遗忘率极低（在事实任务上为 -11% vs -89%/-71%），为增量模型更新提出了一条切实可行的路径 ([@realJessyLin](https://twitter.com/realJessyLin/status/1980662516285075762), [blog](https://twitter.com/realJessyLin/status/1980697898141774017))。
- **大规模机械解释性（Mechanistic Interp）**：Anthropic 分析了 Claude 3.5 Haiku 在“感知”任务上的表现，揭示了清晰的几何变换和分布式注意力算法；社区认为这是迄今为止对模型行为最深入的机械化理解之一 ([@wesg52](https://twitter.com/wesg52/status/1980680563582538099), [@NeelNanda5](https://twitter.com/NeelNanda5/status/1980770185167663140))。
- **提示词优化优于复合系统的 RL？** GEPA 使用带有帕累托选择（Pareto selection）的反思性提示词演化，在 HotpotQA、IFBench、Hover、PUPA 上击败了 GRPO，通过自然语言自我批判减少了对 Rollout 的需求 ([@gneubig](https://twitter.com/gneubig/status/1980644772902789603), [paper/code](https://twitter.com/gneubig/status/1980646347188707787), [summary](https://twitter.com/joelniklaus/status/1980651047720001884))。
- **实战评估**：SWE-Bench Pro 排行榜更新显示，顶尖模型的通过率现已超过 40%，Claude 4.5 Sonnet 处于领先地位 ([@scale_AI](https://twitter.com/scale_AI/status/1980685992987431368))。
- **LLM 自博弈（Self-play）的注意事项**：为什么自博弈在双人零和博弈（Minimax）中表现出色，但在现实领域却很棘手（奖励塑造、与人类效用脱节的均衡） ([@polynoamial](https://twitter.com/polynoamial/status/1980697004658556972))。

**开发者工具与应用**

- **Google AI Studio “AI 优先编程”**：翻新后的构建模式集成了多功能脚手架（“手气不错”模式），旨在为 Gemini 应用提供更快的从 Prompt 到生产环境的迭代 ([@OfficialLoganK](https://twitter.com/OfficialLoganK/status/1980674135693971550), [demo](https://twitter.com/patloeber/status/1980676182904565999), [@GoogleAIStudio](https://twitter.com/GoogleAIStudio/status/1980679588704371095))。
- **Runway**：宣布推出自助式模型微调和基于节点的 Workflows 系统，用于串联模型、模态和中间步骤，以构建生产级创意流水线 ([@runwayml](https://twitter.com/runwayml/status/1980620538906054691), [Workflows](https://twitter.com/runwayml/status/1980736639405289786))。
- **Together AI**：现在可以通过与文本推理相同的 API 访问视频和图像生成模型（如 Sora 2、Veo 3） ([@togethercompute](https://twitter.com/togethercompute/status/1980746093932515697))。
- **LlamaIndex**：推出用于本地 LlamaAgents 开发/部署的 llamactl CLI；提供开箱即用的文档 Agent 模板，以及针对以文档为中心的工作流的私测版托管服务 ([@llama_index](https://twitter.com/llama_index/status/1980673952033976824), [@jerryjliu0](https://twitter.com/jerryjliu0/status/1980759684916408443))。

**热门推文（按互动量排序）**

- “哈哈哈，这张床一个月发送 16GB 的数据，天哪” —— AWS 停机期间 IoT 可靠性/遥测（telemetry）方面的槽点 ([@internetofshit](https://twitter.com/internetofshit/status/1980506231233184144))。
- “见见我们的新浏览器 —— ChatGPT Atlas。今日起在 macOS 上可用” ([@OpenAI](https://twitter.com/OpenAI/status/1980685602384441368))；“在你的 Dock 栏腾个位置吧” ([@OpenAI](https://twitter.com/OpenAI/status/1980678350407606518))。
- Karpathy 谈论通过多样化的合成对话为 nanochat 进行合成身份/个性微调 ([@karpathy](https://twitter.com/karpathy/status/1980665134415802554))。
- Qwen Deep Research 升级：支持报告 + 实时网页 + 使用 Qwen3 技术栈自动生成播客 ([@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1980609551486624237))。
- Airbnb CEO：Qwen “非常出色、快速且便宜”，由于成本/延迟原因，在生产环境中通常比“最新”的 OpenAI 模型更受欢迎 ([@natolambert](https://twitter.com/natolambert/status/1980657338726887662))。

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Qwen3-VL 模型性能对比

- [**Qwen3-VL-2B 和 Qwen3-VL-32B 发布**](https://www.reddit.com/r/LocalLLaMA/comments/1och7m9/qwen3vl2b_and_qwen3vl32b_released/) (热度: 626): **该图片详细对比了新发布的 Qwen3-VL-2B 和 Qwen3-VL-32B 模型与其他模型（如 Qwen3-VL-4B、Qwen3-VL-8B 和 Qwen2.5-VL-7B）的性能指标。表格重点展示了模型在 STEM & Puzzle、General VQA 和 Text Recognition 等各项任务中的表现。值得注意的是，Qwen3-VL-32B 模型表现出卓越的性能，在大多数类别中获得了更高的分数（以红色标记以示其重要性）。这表明 Qwen3-VL-32B 模型在这些任务中特别有效，超越了其前代版本和其他变体。** 一条评论幽默地表示，32B 模型的发布应该能满足那些一直以此要求的人，反映了对该模型尺寸的期待和需求。
    - Qwen3-VL-2B 和 Qwen3-VL-32B 模型的发布标志着重大进步，据报道，新模型尽管尺寸不到 Qwen2.5-VL 72B 的一半，但性能却超过了它。这表明模型效率和性能有了实质性提升，可能归功于架构优化或增强的训练技术。
    - 用户提供的对比图突出了 Qwen3-VL-2B 和 Qwen3-32B 之间的性能差异，表明新模型在文本处理任务中可能提供更优越的能力。这对于评估特定应用模型性能的人来说可能特别感兴趣。
    - 讨论中分享的基准测试表明，Qwen3-VL 模型在“思考”任务中表现出色，这可能指的是复杂的推理或问题解决能力。这使得这些模型成为需要高级认知处理应用的强力竞争者。
- [**DeepSeek-OCR AI 可以扫描整个缩微胶片页，而不仅仅是单元格，并在几秒钟内保留 100% 的数据...**](https://www.reddit.com/r/LocalLLaMA/comments/1ocgun0/deepseekocr_ai_can_scan_an_entire_microfiche/) (热度: 405): **根据 [Brian Roemmele 的帖子](https://x.com/BrianRoemmele/status/1980634806145957992)，DeepSeek-OCR AI 声称可以扫描整个缩微胶片（microfiche）页，而不仅仅是单个单元格，并在几秒钟内保留** `100%` **的数据。据报道，该工具提供了对文本和复杂图纸的全面理解，有可能彻底改变离线数据策展。然而，该帖子缺乏详细的技术验证或基准测试来证实这些说法。** 评论者对提取数据的准确性验证以及国家间 AI 开发的开放性表示怀疑，特别是对比了美国和中国。还有人批评该公告缺乏技术细节，称其为未经验证的“炒作废话（hype BS）”。
    - rseymour 对 DeepSeek-OCR AI 的分辨率能力提出了技术担忧，质疑在 `1024x1024` 分辨率下使用 “vision tokens” 的可行性。他们认为这种分辨率可能不足以准确捕捉缩微胶片的细节，因为缩微胶片通常由于尺寸小且信息密度高而需要更高的分辨率。该评论暗示这项技术可能在没有适当能力验证的情况下被过度炒作。
    - Robonglious 讨论了国家间 AI 开发的开放性，特别是对比了中国与美国 AI 进展的透明度。他们推测，如果 **OpenAI** 或 **Anthropic** 开发了类似的 OCR 技术，是否会发布它，暗示美国在分享此类进展方面可能不如中国合作。
    - TheHeretic 和 Big_Firefighter_6081 对 DeepSeek-OCR AI 能力的相关说法表示怀疑。他们批评结果缺乏核实和验证，暗示这些信息可能更多是炒作而非现实。这突显了在 AI 技术声明中进行严格测试和验证以确保可信度的重要性。

## 较少技术性的 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. ChatGPT Atlas 浏览器发布

- [**见见我们的新浏览器——ChatGPT Atlas。**](https://www.reddit.com/r/OpenAI/comments/1ocj2da/meet_our_new_browserchatgpt_atlas/) (热度: 3175): **ChatGPT Atlas 是 OpenAI 推出的一款新浏览器，目前仅在** `macOS` **上可用。该浏览器将 AI 能力直接集成到浏览体验中，有望增强用户与网页内容的交互。然而，此次发布仅限于 Mac 用户，这引发了关于可访问性和平台支持的一些争论。** 评论者对数据隐私以及仅针对 macOS 发布浏览器的决定表示担忧，质疑这一战略选择和潜在的数据处理方式。
    - Big-Info 和 douggieball1312 讨论了 ChatGPT Atlas 的平台排他性，指出它目前仅适用于 Mac。这一决定被批评为可能会疏远 Windows 用户，特别是考虑到 Microsoft 对 OpenAI 的资金支持。在 Microsoft 投资的背景下，这一点显得颇具讽刺意味，因为 Windows 是 Mac 的主要竞争对手。
    - Tueto 提出了对 ChatGPT Atlas 数据隐私的担忧，询问用户数据被发送到了哪里。这反映了人们对 AI 驱动应用程序中数据处理和隐私的广泛担忧，尤其是在经常访问敏感信息的网页浏览场景中。
    - douggieball1312 指出了 ChatGPT Atlas 仅限 Mac 使用的讽刺之处，尽管 OpenAI 得到了 Microsoft 的支持。这一决定被视为硅谷技术泡沫的反映，可能会忽视更广泛的用户群体（特别是 Windows 用户），从而影响采用率和用户满意度。
- [**GPT 浏览器即将到来**](https://www.reddit.com/r/OpenAI/comments/1ocfrxy/gpt_browser_incoming/) (热度: 1511): **这张图片是 OpenAI 首席执行官 Sam Altman 发布的一条社交媒体帖子，宣布将举行一场直播活动来发布新产品。该帖子被 OpenAI 转发，并配有一张带有 "Livestream" 字样和 OpenAI 徽标的图形，预示着一项重大发布。社区对产品性质进行了猜测，一些评论幽默地暗示是“性爱机器人”，或者表达了对隐私的担忧，将其比作类似于 Google 做法的“间谍软件”。帖子上的互动表明用户对该发布有着极高的兴趣和期待。** 评论反映了幽默与怀疑的交织，一些用户开玩笑说产品是“性爱机器人”，另一些用户则对隐私表示担忧，将其与 Google 的数据实践进行比较。
    - trustmebro24 推测即将推出的 GPT 浏览器可能基于 Chromium，由于其开源特性和强大的性能，这是许多现代浏览器的常见选择。Chromium 的架构允许对高级功能进行广泛的定制和集成，这对于利用 GPT 技术的浏览器来说可能非常有益。
    - qodeninja 提出了对公司可能因开发过多产品而过度扩张的担忧，建议让更广泛的生态系统进行创新并创建互补技术可能会更有效。这反映了对技术开发中资源分配和重点的战略思考。
    - Vegetable_Fox9134 提到了与新浏览器相关的潜在隐私问题，并将其与 Google 现有的问题进行了比较。这突显了关于数据隐私的持续争论，以及用户在使用可能收集个人信息的技术时所面临的权衡。
- [**OpenAI 的 AI 驱动浏览器 ChatGPT Atlas 来了**](https://www.reddit.com/r/ChatGPT/comments/1ociphv/openais_aipowered_browser_chatgpt_atlas_is_here/) (热度: 1041): **OpenAI 发布了一款名为 ChatGPT Atlas 的 AI 驱动浏览器，它将 ChatGPT 的功能集成到了网页浏览中。该工具旨在通过直接在浏览器环境中提供 AI 驱动的见解和协助来增强用户交互。这种集成有望通过利用 ChatGPT 的对话能力来简化任务，从而可能改变用户与网页内容的互动方式。** 评论反映了怀疑与好奇的交织，一些用户表达了对隐私和潜在误用的担忧，而另一些用户则对 AI 增强浏览的可能性深感兴趣。

- [**【已确认】OpenAI 今日发布名为 ChatGPT Atlas 的新浏览器**](https://www.reddit.com/r/ChatGPT/comments/1ochtsq/confirmed_openai_is_launching_a_new_browser_today/) (热度: 747): **OpenAI 发布了一款名为 ChatGPT Atlas 的新浏览器，目前已在** `macOS` **全球上线，并计划很快推出** `Windows`**、** `iOS` **和** `Android` **版本。该浏览器将 AI 能力直接集成到浏览体验中，提供了一个用于无缝 AI 交流的聊天界面。该产品由 Sam Altman 和 Ben Goodger 等关键人物介绍。该浏览器被视为与 Google 和 Microsoft 竞争的战略举措，尽管它因与现有浏览器功能相似（仅增加了聊天功能）而受到批评。更多详情请参阅 [YouTube 视频](https://www.youtube.com/watch?v=8UWKxJbjriY?1)。** 评论者对该浏览器的影响表示怀疑，指出它可能主要服务于 OpenAI 的数据收集需求，而非提供显著的用户利益。人们对隐私和数据发送至 OpenAI 表示担忧，一些人质疑在已有替代方案的情况下该浏览器的必要性。
    - OpenAI 推出 ChatGPT Atlas 被视为与 Google 和 Microsoft 等科技巨头竞争的战略举措。虽然该浏览器可能会在便利性和速度上有所提升，但其对用户的影响仍存疑。主要的担忧在于其广泛的数据收集能力，通过实时了解用户的生活、兴趣和行为，这可能超越现有系统。
    - 有推测认为 ChatGPT Atlas 可能基于 Chrome 引擎，这将与许多利用 Chromium 获得兼容性和性能优势的现代浏览器保持一致。这一选择可能会通过提供熟悉的用户体验和对现有 Web 标准的支持，从而影响浏览器的普及。
    - 用户的一个重大担忧是使用 ChatGPT Atlas 潜在的隐私影响。该浏览器可能会收集海量的个人数据，引发了关于 OpenAI 将如何处理和保护这些信息的疑问。对于那些可能不完全了解所涉及数据共享程度的用户来说，这种担忧尤为突出。

### 2. Claude Desktop 正式发布 (General Availability)

- [**Claude Desktop 现已正式发布。**](https://www.reddit.com/r/ClaudeAI/comments/1ock3em/claude_desktop_is_now_generally_available/) (热度: 836): **Claude Desktop 现已在 Mac 和 Windows 平台上正式发布，提供与本地工作环境的无缝集成。用户可以通过在 Mac 上双击 Option 键来访问 Claude，截取屏幕、共享窗口，并可通过 Caps Lock 使用语音命令。该应用程序支持使用** `MSIX` **和** `PKG` **安装程序进行企业级部署。欲了解更多详情并下载，请访问 [Claude 官方网站](https://claude.com/download)。** 一些用户对该公告感到困惑，认为该应用此前已经可用，而另一些用户则注意到缺少 Linux 版本。Quick Entry 功能因其实用性而受到称赞。
    - ExtremeOccident 提到，尽管 Claude Desktop 仍处于 beta 阶段，但 Quick Entry 功能非常有效，这表明其在用户体验和输入处理效率方面的关注。
    - Logichris 强调了 Claude Desktop 在 Token 分配方面的限制，将其比作“月光族”式的情景，这表明当前的 Token 系统可能无法在不频繁补充的情况下支持大规模使用。
    - 包括 Yeuph 和 JAW100123 在内的多位用户指出了 Linux 版本的缺失，这表明平台支持方面的差距可能会限制 Linux 用户的使用。
- [**{赠送活动} 40 份为期 1 年的 Gemini AI PRO**](https://www.reddit.com/r/GeminiAI/comments/1ocovu8/giveaway_1_year_of_gemini_ai_pro_40_winners/) (热度: 2833): **该帖子宣布为 40 名获奖者提供为期一年的 Gemini AI PRO 订阅，重点介绍了即将推出的 *Gemini 3.0 Ultra*、**`每月 1,000 个 AI 积分`**，以及 *Gemini Code Assist*、*NotebookLM* 等工具，并集成了 *Gmail, Docs, 和 Vids*。该套餐还包括** `2TB 存储空间` **以及各种应用程序的扩展限制，旨在增强不同领域的生产力和创造力。** 评论者强调了 Gemini AI 的多种用途，例如辅助个人和专业用途的故事创作和语言翻译，通过其生态系统支持电影制作，以及通过代码生成能力增强开源贡献。
    - Bioshnev 强调了 Gemini AI 在个人和专业环境中的实际应用。他将其用于为女儿创作定制的睡前故事，以及处理与工作相关的任务，如为外国客户翻译和检索产品详情。这展示了该模型在处理语言处理和信息检索任务方面的多功能性。
    - thenakedmesmer 讨论了 Gemini AI 对创意项目（特别是电影制作）的影响。他提到使用 'nano banana' 和 'veo' 等功能作为支持性生态系统的一部分，辅助电影制作，说明了 AI 如何充当虚拟创意团队，弥补物理限制并增强创意工作流。
    - vladlearns 强调了 Gemini AI 的代码生成能力对于开源贡献的重要性。这指向了该模型在软件开发中的实用性，它可以协助自动化编码任务，从而提高生产力并支持协作项目。

### 3. 亚马逊的机器人劳动力计划

- [**根据泄露文件，亚马逊希望用机器人取代 600,000 名美国员工。到 2027 年，裁员可能会使每件购买的商品成本降低 30 美分。**](https://www.reddit.com/r/singularity/comments/1occruc/amazon_hopes_to_replace_600000_us_workers_with/) (热度: 1630): **据报道，根据泄露的文件，亚马逊计划到** `2027` **年用机器人取代** `600,000` **名美国员工。这种自动化可能会使每件商品的成本降低** `30 cents`**。该计划是解决劳动力短缺和提高履约中心效率的更广泛战略的一部分，这是亚马逊自十多年前收购 Kiva Systems 以来一直追求的目标。由于亚马逊履约中心的高流失率和劳动力短缺，向机器人技术的过渡被视为必要步骤。** 评论者指出，节省的成本可能不会转化为消费者的低价，并强调了由于亚马逊面临的劳动力挑战，自动化的战略必要性。一位前 Amazon Robotics 员工指出，用机器人取代工人的目标由来已久，但进展比预期要慢。
    - 'theungod' 的评论强调了亚马逊面临的一个关键运营挑战：履约中心 (FCs) 的高流失率和人员配备困难。该用户指出，自十多年前收购 Kiva Systems 以来，亚马逊一直致力于将这些角色自动化，但向机器人技术的过渡比预期要慢。这表明，将机器人技术整合到亚马逊的物流中不仅是为了节省成本，也是为了解决劳动力短缺问题。
    - 'theungod' 还提供了内部人士的视角，他曾在 Amazon Robotics 工作了五年多。他们强调，用机器人取代 600,000 名员工是一个长期目标，这表明技术和物流方面的障碍非常巨大。这一见解强调了在履约运营中实施大规模自动化的复杂性，这不仅涉及技术开发，还涉及克服实际部署中的挑战。
    - 讨论涉及了物流自动化的更广泛影响，特别是潜在的社会影响。虽然提到了每件商品节省的成本 (30 cents)，但重点在于由于劳动力短缺而产生的自动化必要性，而非纯粹的财务激励。这反映了叙事从削减成本向运营必要性的转变，这是由于在苛刻的环境中无法维持稳定的劳动力所驱动的。
- [**变形无人机**](https://www.reddit.com/r/singularity/comments/1oc5v07/shape_shifting_drone/) (热度: 1226): **该帖子讨论了一种具有独特设计的变形无人机，其设计灵感可能来自生物形态，正如评论中将其比作“漂浮的结肠镜”所暗示的那样。评论中链接的图像显示了一个具有柔性结构的无人机，这可能使其能够根据不同的飞行力学或环境条件调整其形状。这可能是无人机技术中的一种创新方法，有可能提高机动性和效率。** 一条评论建议变形无人机的概念并不完全新鲜，表明之前可能见过类似的设计。这可能意味着该领域正在进行持续的研究和开发，反映了向更具适应性和多功能 UAV 设计发展的趋势。

---

# AI Discord 回顾

> 由 gpt-5 生成的摘要的摘要总结
> 

**1. GPU 和 eGPU 硬件突破**

- **Blackwell Pro 搭载 72GB 显存，低调发布**：TechPowerUp 报道 NVIDIA 低调发布了工作站级 **RTX Pro 5000 Blackwell**，配备 **72 GB GDDR7** 显存，面向专业工作流（[NVIDIA RTX Pro 5000 Blackwell GPU with 72 GB GDDR7 appears](https://www.techpowerup.com/342059/nvidia-rtx-pro-5000-blackwell-gpu-with-72-gb-gddr7-memory-appears)）。
    - 工程师们调侃了可能的定价和使用场景，而其他人则对不寻常的 **72 GB** 容量表示最初的困惑，[VideoCardz](https://videocardz.com/newz/nvidia-quietly-launches-rtx-pro-5000-blackwell-workstation-card-with-72gb-of-memory) 也有类似的报道。
- **Tinygrad 让 Apple Silicon 支持 NVIDIA eGPU**：tinygrad 团队宣布开始公开测试纯 Python 驱动程序，通过 ADT-UT3G 扩展坞、`extra/usbgpu/tbgpu` 驱动和基于 NVK 的 `tinymesa` 编译器，在 **Apple Silicon** 上通过 **USB4** 启用 **NVIDIA eGPU**（[tinygrad enables NVIDIA eGPU on Apple Silicon (X)](https://x.com/__tinygrad__/status/1980082660920918045)）。
    - 他们在禁用 SIP 的情况下测得约 **≈3 GB/s** 的 PCIe 带宽，并预告接下来将支持 **AMD RDNA 2/3/4** 和 Windows eGPU 栈。
- **Tiny Corp 在 ARM MacBook 上成功运行 NVIDIA GPU**：Tiny Corp 演示了通过外部扩展坞在 **ARM MacBook** 上使用 **USB4** 运行 **NVIDIA GPU**，验证了 eGPU 在 Intel 时代之后的 Mac 上的可行性（[Tiny Corp Successfully Runs An Nvidia GPU on Arm Macbook Through USB4 Using An External GPU Docking Station](https://www.tomshardware.com/pc-components/gpus/tiny-corp-successfully-runs-an-nvidia-gpu-on-arm-macbook-through-usb4-using-an-external-gpu-docking-station)）。
    - Mac 用户情绪高涨，指出配备 **Thunderbolt 5** 的新款 Pro 可能会进一步提高本地 **LLM** 和 **VLM** 工作负载的带宽余量。

**2. Triton/Kernel 工具与基准测试**

- **FlashInfer-Bench 开启 Agent 驱动的算子竞赛**：CMU Catalyst 推出了 **FlashInfer-Bench**，这是一个针对 Agent 驱动、自我改进的 **LLM 推理算子 (serving kernels)** 的工作流和排行榜，具有标准化的签名，并集成了 **FlashInfer**、**SGLang** 和 **vLLM**（[FlashInfer-Bench 博客](https://flashinfer.ai/2025/10/21/flashinfer-bench.html)）。
    - 他们发布了实时[排行榜](https://bench.flashinfer.ai/)和 [GitHub 仓库](https://github.com/flashinfer-ai/flashinfer-bench)，邀请社区对算子和基准测试更新进行迭代。
- **Triton 会议直播备受关注**：开发者分享了 Microsoft 举办的 **Triton** 会议的全程视频，涵盖了编译器进展和算子设计（[Triton Conference 直播](https://www.youtube.com/live/s30WoZ7lx3w) 和 [Triton-openai 频道](https://www.youtube.com/@Triton-openai/streams)）。
    - 一个反复出现的主题是为关键算子手动调整 **PTX/汇编** 以超越编译器默认性能，呼应了从底层重新思考执行过程的呼声。
- **Helion 0.2 Beta 对 Triton 进行压力测试**：**Helion 0.2** 作为 PyPI 上的 Triton tile 抽象进入公开测试阶段，在优化过程中暴露了编译器的边缘情况（[PyPI 上的 helion 0.2.0](https://pypi.org/project/helion/0.2.0)）。
    - 用户报告了 `TritonGPUOptimizeThreadLocalityPass` 中的 MLIR 故障，将 **Helion** 视为一种有效的 Triton 编译器“模糊测试器 (fuzzer)”，其自动调优器会跳过错误的配置。

**3. OpenRouter SDK 与新推理模型**

- **OpenRouter SDK 为 300 多种模型提供类型支持**：OpenRouter 发布了 **TypeScript SDK (beta)**，为 **300 多种模型**提供完整的请求/响应类型支持、内置 OAuth 以及对所有 API 路径的支持（[npm 上的 @openrouter/sdk](https://www.npmjs.com/package/@openrouter/sdk)）。
    - **Python**、**Java** 和 **Go** 版本的 SDK 即将推出，旨在简化多模型应用开发和身份验证。
- **Andromeda-alpha 涵盖视觉推理**：OpenRouter 推出了 **Andromeda-alpha**，这是一个专注于**图像/视觉理解**的小型**推理 (reasoning)** 模型，现已开放试用（[OpenRouter 上的 Andromeda-alpha](https://openrouter.ai/openrouter/andromeda-alpha)）。
    - 由于提示词/输出会被记录以改进提供商的模型，管理员警告：避免使用个人/机密数据，且不要将其用于生产环境。
- **Mercury 在 Agent 竞技场中击败 Qwen**：在 Agent 基准测试中，来自提供商 **Chutes** 的 **Inception/Mercury** 在简单任务的失败率、延迟和成本方面略胜 **Qwen**（[Chutes 提供商页面](https://openrouter.ai/provider/chutes)）。
    - 成员指出，较新的 **DeepSeek v3.1** 模型通过 Chutes 不再免费，但仍保留了一个免费的 longcat 端点（[longcat-flash-chat:free](https://openrouter.ai/meituan/longcat-flash-chat:free)）。

**4. 开源模型与文本转视频发布**

- **Ring & Ling MoEs 登陆 llama.cpp**：来自 **InclusionAI** 的 **Ring** 和 **Ling** **MoE** 模型现在可以在 **llama.cpp** 中运行，涵盖 **1T**、**103B** 和 **16B** 参数规模 ([llama.cpp PR #16063](https://github.com/ggml-org/llama.cpp/pull/16063))。
    - 从业者对实际的 **reasoning**（推理）质量和冗余控制提出了疑问，希望能有一种在 Chain-of-Thought（思维链）过程中不会**喋喋不休**的模型。
- **Krea Realtime 发布 14B 开源 T2V 模型**：**Krea Realtime** 发布了一个从 **Wan 2.1** 蒸馏而来的 **14B** 开源自回归 Text-to-Video 模型，在单张 **NVIDIA B200** 上能以约 **11 fps** 的速度生成长视频 ([Krea Realtime 公告 (X)](https://x.com/krea_ai/status/1980358158376988747))。
    - 权重以 **Apache-2.0** 协议在 HuggingFace 上发布；用户询问了关于 **ComfyUI** 工作流、**RTX 5090** 性能以及微调选项的问题。
- **DeepSeek-OCR 加入 OCR 战场**：**DeepSeek-OCR** 已在 GitHub 上线，通过现代化的 VLM 友好设计和多语言目标扩展了 **OCR** 工具包 ([DeepSeek-OCR (GitHub)](https://github.com/deepseek-ai/DeepSeek-OCR))。
    - 开发者将其与现有的 OCR 技术栈进行了对比，并强调了上下文理解对于 **kanji**（汉字/日文汉字）等脚本的重要性。

**5. AI 应用：ChatGPT Atlas 发布与融资新闻**

- **OpenAI 发布 Atlas，一款基于 Chromium 的 AI 浏览器**：OpenAI 为 macOS 推出了 **ChatGPT Atlas** 浏览器，这是一款基于 **Chromium** 的浏览器，具有更高的限制额度和多站点浏览功能 ([ChatGPT Atlas 介绍](https://openai.com/index/introducing-chatgpt-atlas/) 以及 [chatgpt.com/atlas](https://chatgpt.com/atlas))。
    - 早期用户指出了缺失 **vertical tabs**（垂直标签页）和内置广告拦截（需要扩展程序）的问题，而其他地方的用户则将 **Atlas** 与 **Perplexity** 的 **Comet** 进行了比较，称赞 Comet 对隐私的关注和集成的广告拦截器。
- **AI 浏览器热潮遭遇质疑**：工程师们对新型 **AI 浏览器** 的实用性提出质疑，对性能和数据处理方式表示怀疑 ([AI 浏览器炒作讨论帖 (X)](https://x.com/AlexFinnX/status/1980673764947022038))。
    - 一位成员调侃道：*“OpenAI 也知道这一点，他们只是在收割数据并胡乱尝试，”* 这反映了人们对炒作与实际价值之间差距的普遍担忧。
- **LangChain 获 1.25 亿美元融资构建 Agent 技术栈**：**LangChain** 完成了 **1.25 亿美元 B 轮融资**，定位为三部分技术栈：**LangChain**（Agent 开发）、**LangGraph**（编排）和 **LangSmith**（可观测性） ([LangChain 融资 1.25 亿美元 (X)](https://x.com/sonyatweetybird/status/1980683121399058626))。
    - 他们宣传了 **Uber**、**Klarna** 和 **LinkedIn** 的采用情况，标志着投资者对 **Agent 工具链** 和生产运维的持续信心。

---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet 在隐私方面优于 Atlas**：用户对比了 **Comet** 和 **ChatGPT 的 Atlas 浏览器**，更青睐 **Comet** 对 **隐私** 的承诺和集成的 **广告拦截器**。
   - 几位用户注意到两者之间的相似之处，但称赞 **Comet** 的功能更迎合用户隐私需求。
- **AI 疗法引发伦理担忧**：Discord 成员辩论了使用 **AI 进行心理治疗** 的伦理问题，一些人强调了人类 **情感成熟度** 的重要性。
   - 观点出现分歧，有人认为 *“ChatGPT 是一个很好的治疗师，”* 而另一些人则警告不要过度依赖 AI 进行心理健康支持。
- **Perplexity 狂热粉丝展示周边**：一位用户展示了他们的 **Perplexity 贴纸**，表达了对该品牌的喜爱，并要求从 [Perplexity Supply 商店](https://perplexity.supply/) 购买 **Comet 连帽衫**。
   - 这种热情的展示引发了关于看起来像 *“邪教”* 的轻松玩笑，其他人则鼓励进一步购买。
- **API 用户希望访问 ChatGPT5**：一位用户询问 **Perplexity API** 是否授予对 **ChatGPT5** 和 **Claude** 等模型的访问权限，还是仅限于 **Sonar**。
   - 该询问反映了用户希望通过 API 使用比目前可用的 **Sonar** 更先进模型的愿望。
- **可共享 Discord 帖子提醒**：一条消息提醒用户确保其 Discord 帖子（threads）设置为 `Shareable`（可共享），以便更易于访问。
   - 这确保了帖子的链接可以被其他人访问，即使是在特定频道之外，从而改善协作。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini 3 Pro 在网页设计方面完胜 GPT-5 High**：成员们对比了 **Gemini 3 Pro** 和 **GPT-5 High**，报告称 Gemini 3 Pro 在网页设计上表现惊人。
   - 普遍共识是 **Gemini 3 Pro** 更擅长编程，而 **GPT-5 High** 更擅长数学和其他任务。
- **Sora 2 降级引发 AI 订阅讨论**：成员们对 **Sora 2** 的降级表示失望，从而引发了关于 AI 订阅价值的广泛讨论。
   - 一位成员指出，由于 AI 的存在，他们的工作效率提升了约 25-30%，强调了 AI 对办公效率的影响。
- **Lithiumflow 和 Orionmist 被推测为 Gemini 3 的 Checkpoint**：成员们推测了 **Lithiumflow** 和 **Orionmist** 之间的差异，最终得出结论认为这些模型是 **Gemini 3** 的 Checkpoint 版本。
   - 这些模型有时会错误地声称是由 **OpenAI** 训练的，这暗示了可能存在模型蒸馏（model distillation）。
- **开源模型据称窃取 Gemini 2.5 Pro**：关于开源模型使用窃取数据进行改进的伦理问题引发了讨论，有说法称中国 AI 公司“窃取了 2.5 pro 并将其开源”。
   - 一些成员赞同这种观点，认为这是 **Open Source** 获胜的唯一途径。
- **TikZ 生成任务带来惊喜**：成员们正在探索使用 LLM 生成 **TikZ**（一种排版语言）图像，以避免数据污染。
   - 早期结果显示，使用 LLM 生成 **TikZ** 图像取得了一定成功，展示了一种新颖的图像创建方法。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Ring 和 Ling 发布！**：**Ring 和 Ling MoE 模型** 现在已在 *llama.cpp* 中获得支持（[Github 链接](https://github.com/ggml-org/llama.cpp/pull/16063)），包括来自 InclusionAI 的 **1T**、**103B** 和 **16B** 参数模型。
   - 成员们思考了这些模型的推理能力，其中一人希望看到一个“不会废话连篇（doesn’t YAP）的推理模型”。
- **禁用 Unsloth 统计数据**：为了在离线模式下运行 **Unsloth** 时防止遥测调用，请设置 `UNSLOTH_DISABLE_STATISTICS` 环境变量和 `os.environ['HF_HUB_OFFLINE'] = '1'`。目前 Unsloth 社区在 **Hugging Face 上的终身下载量已达到 1 亿次**（[X 上的公告](https://x.com/UnslothAI/status/1980631523104813419)）。
   - 成员们还讨论了通过设置代理环境来解决网络问题。
- **Nvidia RTX Pro 5000 Blackwell 工作站显卡悄然现身**：据 [VideoCardz](https://videocardz.com/newz/nvidia-quietly-launches-rtx-pro-5000-blackwell-workstation-card-with-72gb-of-memory) 报道，**Nvidia** 悄然推出了拥有 **72GB** 显存的 **RTX Pro 5000 Blackwell** 工作站显卡。
   - 最初人们对 **72GB** 的容量感到困惑，一位用户开玩笑说这是绕过自动审核（automod）的一种方式。
- **用户对速率限制策略感到愤怒**：一位用户抱怨某项**高级订阅服务**拦截了包含**罗马数字**的 URL，错误地将其解释为恶意活动。
   - 该用户对繁琐的变通方法和安全插件感到沮丧，批评该服务忽略了允许专业订阅者批量下载的请求。
- **Nvidia GPU 被移植到 ARM Macbook**：一位成员分享了来自 Tom's Hardware 的文章，内容是关于通过 **USB4** 使用外接 GPU 扩展坞成功在 **ARM Macbook** 上运行 **Nvidia GPU**：[Tiny Corp 成功通过 USB4 使用外接 GPU 扩展坞在 Arm Macbook 上运行 Nvidia GPU](https://www.tomshardware.com/pc-components/gpus/tiny-corp-successfully-runs-an-nvidia-gpu-on-arm-macbook-through-usb4-using-an-external-gpu-docking-station)。
   - 这让 Mac 用户感到兴奋，因为现在的“Pro”机型也配备了 **Thunderbolt 5**，这给 **Mac** 用户带来了更多希望。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Codex 被指与 Claude 相比简直是天壤之别**：用户讨论了 **Codex vs Claude** 在代码生成方面的优劣，一位用户称将两者对比就像是对比“十亿美元与两加元硬币（Toonie）”。
   - 未提供更多细节。
- **Cursor 网站宕机，订阅功能失效**：多位用户报告 **Cursor 网站宕机**数小时，导致他们无法登录、升级方案或续订订阅。
   - 一些人怀疑 **AWS 问题**是根本原因，而另一些人指出**缺乏订阅过期通知**是一个主要的不便之处。
- **仪表板破解调价后的 Cursor 成本**：一位用户分享了他们创建的仪表板，用于**追踪调价后的 Cursor 实际成本**，特别是针对使用旧版定价方案的用户，并提供了论坛链接 [cursor.com/blog/aug-2025-pricing](https://cursor.com/blog/aug-2025-pricing)。
   - 该工具需要 Cookie 登录或从用户本地机器上传 .json 文件，但承诺能与真实的 API 定价进行对比。
- **Background Agents 遇到内部错误**：一位成员报告称，在通过 **Linear** 进行 **background agents** 的首次实验时遇到 *internal error*（内部错误），Agent 启动并进行了一些思考和 Grepping 后停止了。
   - 收到的错误消息为：“我们遇到了一个无法恢复的内部错误。您可以稍后重试。”

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter SDK：Beta 版助力**：新的 **OpenRouter SDK** 已在 [npm](https://www.npmjs.com/package/@openrouter/sdk?activeTab=versions) 开启 **beta** 测试，旨在成为使用 **OpenRouter** 最简单的方式，并为 **300 多个模型**提供全类型定义的请求和响应。
   - Python、Java 和 Go 版本即将推出，具有内置的 OAuth 功能并支持所有 API 路径。
- **Andromeda-alpha：隐身就绪**：**OpenRouter** 发布了一个名为 **Andromeda-alpha** ([https://openrouter.ai/openrouter/andromeda-alpha](https://openrouter.ai/openrouter/andromeda-alpha)) 的新隐身模型，这是一个专注于图像和视觉理解的小型推理模型。
   - 提示词/输出会被记录以改进模型，提醒用户不要上传个人/机密信息，且不要将其用于生产环境。
- **Objective AI 的置信度代码**：[Objective AI](https://objective-ai.io/) 现在为每个 **OpenAI** 补全选项提供 **Confidence Score**（置信度得分），该得分通过比直接询问 AI 更聪明的方法得出，并强调**成本效益**。
   - 首席执行官正在利用 **n8n** 集成免费构建**可靠的 AI Agents、工作流和自动化**，以收集更多案例。
- **Mercury 在 Agent 竞技场中击败 Qwen**：**Inception/Mercury**（由 [Chutes](https://openrouter.ai/provider/chutes) 提供）在简单的 Agent 任务中险胜 **Qwen**，表现出更低的失败率、更快的速度和更低的成本。
   - 像 **v3.1** 这样的新 **Deepseek** 模型无法通过 **Chutes** 获得免费版本，尽管他们最近添加了一个免费的 longcat 端点。
- **AI 浏览器热潮令人困惑**：成员们对新型 AI 浏览器（如 [X 的 AI 浏览器](https://x.com/AlexFinnX/status/1980673764947022038)）的热潮持怀疑态度，质疑集成 AI 的实用性和对性能的影响。
   - 一位成员将这种热潮比作互联网泡沫，并表示 *OpenAI 也知道这一点，他们只是在收集数据并胡乱尝试*。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI 发布 Atlas 浏览器**：OpenAI 今天在 [chatgpt.com/atlas](https://chatgpt.com/atlas) 为 macOS 发布了 **ChatGPT Atlas Browser**，详情见其 [博客文章](https://openai.com/index/introducing-chatgpt-atlas/)。
   - 该浏览器基于 **Chromium**，具有更高的限制、直接的网站访问权限并支持多个网站，但缺少垂直标签页和内置广告拦截器。
- **Meta 在 WhatsApp 上关闭 1-800-ChatGPT**：根据一篇 [博客文章](https://openai.com/index/chatgpt-whatsapp-transition/)，Meta 将在 **2026 年 1 月 15 日**之后在 WhatsApp 上屏蔽 **1-800-ChatGPT**。
   - 这一变化是由于 Meta 的新政策所致。
- **Sora 限制视频长度**：**Sora iOS 应用**将视频生成限制在 **10-15 秒**，而网页版则允许 **Pro 订阅者**生成更长的视频。
   - **Free 和 Plus** 用户也可以在网页版上生成更长的视频，其中 **Pro 用户**可以使用 **storyboard** 功能并生成长达 **25 秒的视频**。
- **AI 驱动的操作系统原型出现**：一名成员介绍了一个 **AI 驱动的 OS** 原型，其特点是拥有 **AI Copilot Core**、**Seamless Dual-Kernel Engine** 以及用于 AI 策展应用的 **NeoStore** ([来源](https://discord.com/channels/974519864045756446/977259063052234752/1430299596428546182))。
   - 其他组件包括 **HoloDesk 3D workspace**、**Auto-Heal System**、**Quantum Sync** 以及用于访问外部 AI 工具的 **Atlas Integration Portal**。
- **GPT-4 让用户感到恼火**：一位用户对 **GPT-4** 新的居高临下的语气表示愤怒，尤其是像 *"if you insist"*（如果你坚持的话）之类的短语，并要求降低模型的自信度。
   - 除了普遍认同新的 GPT 很烦人之外，没有提供任何解决方案。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **GPT-4o 拯救数据流水线？**：一名成员建议使用 **GPT-4o** 或其他视觉模型进行高精度标注和自动化比较，但他们也对更换 **Apache Beam** 的成本持谨慎态度。
   - 另一名成员认为这种架构大材小用，将其比作*提议用星际飞船去杂货店*。
- **Unsloth 脚本轻松微调 LLM**：一名成员请求关于在 **LLM** 上设置 **Parameter-Efficient Fine-Tuning (PEFT)** 的见解，另一名成员指出了多 GPU 设置中的挑战，并建议在 **Colab Free** 上使用 **Unsloth 的脚本**。
   - 他们提醒注意处理公司内部数据，并链接到了更多资源，如 [Fine-tuning LLMs Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide)。
- **Databomz 管理所有 Prompt！**：一名成员介绍了 [Databomz](https://www.databomz.com)，这是一个用于保存、组织和共享 Prompt 的工作区和 **Chrome 扩展程序**，具有标签、版本和文件夹等功能。
   - 该成员强调了 *Forever Free* 计划，并鼓励 Prompt 工程师提供反馈。
- **独立开发者创建 TheLastRag**：一名独立开发者创建了一个名为 **TheLastRag** 的完整 **LLM Framework**，强调了真实记忆 (True memory)、真实个性 (True personality)、真实学习 (True learning) 和真实智能 (True intelligence) 等特性，并正在寻求 [反馈](https://dev.thelastrag.de/)。
   - 主要观点是该 AI *永不遗忘*，拥有*真实的个性*、*真实的学习能力*和*真实的智能*。
- **本地 VLM 训练消耗数 GB 内存**：一名成员报告称，在本地进行 **VLM 练习训练**时，使用了大量的交换内存（**声称 62GB**，**虚拟内存约 430GB**）。
   - 该成员还询问是否有办法专门限制 Mac 上 **MPS** (Metal Performance Shaders) 的内存使用，目标是在更合理的 **40GB VRAM** 限制内进行训练。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 链接 llama.cpp 遇到困难**：成员们注意到，*如何调用我自己的 llama.cpp 供 LM Studio 使用* **尚未得到完全支持**，且引用此内容的 [LM Studio 文档](https://lmstudio.ai/docs/app/advanced/lm-runtimes)是一个失效链接。
   - 目前没有明显的已知变通方法，因此用户可能需要等待该功能添加。
- **AGI 预计到达时间：2044年？**：一位成员预测 AGI 还有 **10-20 年的时间**，声称 *在 5 年内 LLM 的上下文可能就会变得足够大*。
   - 另一位成员开玩笑地建议他 *去当顾问，每小时收费 1000 美元*。
- **GPT-OSS 推理需要元数据**：一位用户询问如何在 **GPT-OSS** 微调模型中设置推理力度（reasoning effort），一位成员回答说，这 *归功于 gpt-oss 的 mxfp4 模型中的元数据，这就是为什么微调模型/ggufs 没有这个功能的原因*。
   - 这位热心的成员提出在将其量化为 **gguf** 之前先将其提供出来。
- **OpenWebUI 通过 OpenAI 连接到 LM Studio**：在尝试将 **OpenWebUI** 连接到 **LM Studio** 时，用户建议利用 **OpenAI** 选项而不是 **OpenAPI**。
   - 成员们协助排查了连接故障，并指向了这个 [HuggingFace 讨论](https://huggingface.co/spaces/huggingchat/chat-ui/discussions/748#68f1519534c92ca5e3f97053)，建议在地址中加入 **/v1**。
- **NVIDIA RTX Pro 5000 Blackwell 泄露**：一位成员分享了一篇 [TechPowerUp 文章](https://www.techpowerup.com/342059/nvidia-rtx-pro-5000-blackwell-gpu-with-72-gb-gddr7-memory-appears)，内容关于 **NVIDIA RTX Pro 5000 Blackwell GPU**，该显卡配备 **72 GB** 的 **GDDR7** 显存。
   - 兴奋的用户幽默地回应，猜测这张显卡的成本将在 **8,000 到 10,000 美元** 左右。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **TinyGrad 驱动 Apple Silicon eGPU**：**TinyGrad** 现在通过 **USB4** 支持 **Apple Silicon** 上的 **NVIDIA eGPU**，使用户能够通过 ADT-UT3G 扩展坞、`extra/usbgpu/tbgpu` 驱动程序和基于 NVK 的 `tinymesa` 编译器运行外部 **RTX 30/40/50 系列 GPU**（[来源](https://x.com/__tinygrad__/status/1980082660920918045)）。
   - 在禁用 SIP 的情况下，此设置可实现约 **3 GB/s 的 PCIe 带宽**，并计划未来支持 **AMD RDNA 2/3/4** 和 **Windows eGPU** 栈。
- **Krea AI 发布实时视频模型**：**Krea AI** 发布了 **Krea Realtime**，这是一个 **14B** 的开源自回归文本转视频模型，由 **Wan 2.1** 蒸馏而来，在单张 **NVIDIA B200** 上能以 **11 fps** 的速度生成长视频（[来源](https://x.com/krea_ai/status/1980358158376988747)）。
   - 已发布的权重托管在 **HuggingFace** 上，采用 **Apache-2.0** 协议，引发了用户对 **ComfyUI** 工作流、**RTX 5090** 性能以及微调支持的咨询。
- **Google AI Studio 与 Gemini 的 “Vibe-Coding”**：**Google AI Studio** 在经过五个月的开发后，正推出一种全新的“从提示词到生产”（prompt-to-production）的 **Gemini** 体验，旨在让 **AI 应用构建简单 100 倍**（[来源](https://x.com/OfficialLoganK/status/1980435968323907884)）。
   - 反应中既有兴奋（对移动端 App、退出选项、更高频率限制的请求），也有功能建议（仅限 GSuite 发布、VS Code 插件、短浏览器 Agent 任务），还有一些对该功能定位与 Gemini 3 预期是否相符的怀疑；团队确认仅限企业的部署已经可用。
- **Fish Audio S1：TTS 革命？**：**Fish Audio** 推出了 **S1**，这是一款文本转语音模型，据称成本仅为 **ElevenLabs** 的 1/6，并宣称拥有 **2 万开发者**和 **500 万美元 ARR**（[来源](https://x.com/hehe6z/status/1980303682932744439)）。
   - 用户分享了即时语音克隆演示，并询问实时延迟（约 **500ms**），而创始人承认了目前的局限性，并承诺下一步将提供更广泛的语言支持和对话模型。
- **二手 RTX 3090 购买技巧**：Taha 分享了购买二手 **RTX 3090** 的经验教训：亲自会见卖家检查显卡，携带便携式 eGPU 测试平台，通过 nvidia-smi 验证识别情况，运行 **memtest_vulkan** 检查 **VRAM 完整性**，可选运行 gpu-burn 进行计算压力测试，加载大型模型并监控温度 **<100 °C**；详见[此处指南](https://xcancel.com/taha_yssne/status/1960418430655586677)。
   - 测试平台是一台运行 **NixOS** 且处于 **PRIME offload 模式**的 **Framework 13 Ryzen 笔记本**，一位用户建议在他们的平台上尝试 tinygrad，因为 *由于我使用的是 Linux，我的平台开箱即用*。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **AMD 的 Web3 云策略**：在一次 **AMD 活动**中，该公司强调了 **Web3** 的“云”属性，引发了一些关注（*smileforme* 表情符号）。
   - 关于 AMD 具体产品的细节仍然模糊，社区只能对其在云端去中心化技术方面的布局进行推测。
- **FlashInfer-Bench 自动化 AI**：CMU Catalyst 推出了 **FlashInfer-Bench**，这是一个通过 Agent 创建自我改进 AI 系统的流水线，其特点是为 **LLM** 推理 Kernel 提供了标准化签名，并集成了 **FlashInfer**、**SGLang** 和 **vLLM**（[博客文章](https://flashinfer.ai/2025/10/21/flashinfer-bench.html)，[排行榜](https://bench.flashinfer.ai/)，[GitHub 仓库](https://github.com/flashinfer-ai/flashinfer-bench)）。
   - 该项目旨在促进社区开发和基准测试，使 AI 系统能够迭代地提升其性能。
- **Triton 会议在微软引发热烈反响**：参加在山景城 **微软举办的 Triton 会议**的成员分享了[在线观看会议的 YouTube 链接](https://www.youtube.com/live/s30WoZ7lx3w?si=O6aQMCVjKFs2F4qa)以及 [Triton-openai 直播频道的链接](https://www.youtube.com/@Triton-openai/streams)。
   - 会议汇集了开发者和研究人员，共同讨论 **Triton** 语言的最新进展和应用。
- **NCCL Kernel 在 PG-NCCL 的内部流上运行**：当设置了 `CUDAStreamGuard` 并通过 `ProcessGroupNCCL` 调用 NCCL 操作时，**NCCL Kernel** 会在 PG-NCCL 的内部流上运行，通常每个设备使用一个高优先级的流，并使用 Tensor 生命周期流（[相关代码](https://github.com/pytorch/pytorch/blob/03f3f7899cbe4276d02379575c74f439b47668ce/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L3132)）。
   - 设置 `CUDAStreamGuard` 决定了 **NCCL 流**等待哪个流，从而建立入向依赖（incoming dependency），正如在 [PyTorch 源代码](https://github.com/pytorch/pytorch/blob/03f3f7899cbe4276d02379575c74f439b47668ce/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L803)中所见。
- **SLNG.AI 寻找语音 AI 性能专家**：**SLNG.AI** 正在寻找一名**语音模型性能工程师**，负责构建实时语音 AI 的骨干架构（[更多详情](https://isla-house.notion.site/Build-the-Backbone-of-Real-Time-Voice-AI-Join-SLNG-as-Founding-Speech-Model-Performance-Engineer-2642fa00c69d8072bf2fd9047f1b0b68)）。
   - 该职位需要强大的软件工程背景，以优化和增强语音模型的性能。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **IntelliCode 读懂你的心思**：一位成员对 Visual Studio 中的 **Microsoft IntelliCode** 表示赞叹，这是一款 AI 驱动的代码补全工具，能够利用大量上下文准确预测整个方法体。
   - 他们评论说，由于它能够以惊人的准确度理解并预判编程需求，当它运行良好时*简直就像在读你的心思*。
- **DeepSeek OCR 加入竞争**：[DeepSeek-AI 在 GitHub 上发布了 DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)，加入了 OCR 技术领域的竞争。
   - 此外，[Anthropic 在 Web 端发布了 Claude Code](https://www.anthropic.com/news/claude-code-on-the-web)，为寻求 AI 辅助编程工具的开发者提供了更多选择。
- **Amazon Vibe Code 结束 Beta 测试**：亚马逊的 **Vibe Code IDE** 已结束仅限邀请的 Beta 测试，但使用费用为 **500 积分**。
   - 这是又一个利用 AI 的 **VSCode fork**。
- **开源细节脱离了西方的掌控？**：一位成员感叹西方缺乏顶尖的**开源实验室**，因为 **DeepSeek** 不断展示令人印象深刻的发现。
   - 他们指出，开源权重仅占整体价值的一小部分，并强调了开源**数据收集**、**方法**和**训练细节**的重要性。
- **宇树科技（Unitree）将碾压特斯拉？**：一位成员预测 **Unitree** 将主导人形机器人市场。
   - 他们推测 **Elon Musk** 可能难以获得必要的零部件，并调侃说*多亏了那个橙色家伙，他现在可能连执行器的磁铁都买不到*。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 驱动 AI NPC 配音**：一位成员为游戏 NPC 构建了一个[语音生成系统](https://github.com/Gielinor-Speaks/voiceover-mage)，使用 **DSPy** 解析 wiki 内容并为 **ElevenLabs** 生成语音提示词（voice prompts），同时分享了一段[开发日志风格的视频](https://youtu.be/z3DQm2PiKpo)。
   - 他们计划利用 **DSPy** 的优化功能来改进角色分析流水线并自动选择语音，并打算收集人工选择作为训练信号，未来使用**自动评审循环（automated judging loop）**针对主观质量判断进行优化。
- **DSPy 出现在研究论文中**：一篇新论文 ([https://arxiv.org/abs/2510.13907v1](https://arxiv.org/abs/2510.13907v1)) 在其研究中使用了 **DSPy**，标志着其在学术界的采用率不断提高。
   - 尽管论文提到了 **DSPy** 的使用，但相应的代码仓库尚未公开。
- **探讨 DSPy 历史记录访问**：成员们讨论了为什么 `inspect_history()` 是 `dspy` 中的一个方法而不是模块对象，并澄清了 `dspy.inspect_history()` 更多用于全局历史记录，而单个程序也会跟踪历史记录。
   - 有人指出，如果设置了 `dspy.configure(track_usage=True)`，可以通过 `predictor.history` 访问历史记录，但仍有人对此感到困惑。
- **揭秘带有 Context 的 DSPy Adapters**：讨论涵盖了在 DSPy 中使用 adapters 的方法，示例展示了如何使用 `dspy.context` 应用单个 adapter，用户可以通过 `dspy.configure(track_usage=True)` 跟踪使用情况。
   - 一位成员给出了使用 `with dspy.context(lm=single_call_lm, adaptor=single_adaptor):` 进行设置的示例，以进一步澄清该过程。
- **Trace 声称比 DSPy 具有准确率优势**：一位成员询问了 [Microsoft Trace](https://microsoft.github.io/Trace/) 与 DSPy 的对比，另一位成员指出 Trace 声称比 DSPy 的**准确率高出 8%**，且看起来 Token 效率更高。
   - 一位成员表示他们会尝试一下以进行公平对比，尽管他们可能仍然觉得使用 DSPy 拥有更细粒度的控制。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Discord 服务器徽章引发辩论**：成员们讨论了添加服务器徽章（类似于角色图标）的可能性，以及服务器标签是否会过于广泛地广播服务器，从而可能增加 EAI 工作人员的审核负担，并引用了[这张截图](https://cdn.discordapp.com/attachments/729741769738158194/1429978898229100605/Screenshot_2025-10-20_at_7.45.09_PM.png?ex=68f96ca1&is=68f81b21&hm=e032ead2cf427352fba72fdba46d77407a4d2bd71ddc4b60a1c2b6aa04cb8980&)。
   - 一位成员指出：“制作标签很酷，但在某种程度上是在向其他地方广播这个服务器，EAI 工作人员已经有太多人需要审核了。”
- **EleutherAI IPO 梦想引发玩笑**：在询问某个特定股票代码是否可用后，一位成员开玩笑地问：“**Eleuther 的纽交所股票代码**会是什么？”
   - 另一位成员回答道：“我想你误解了作为非营利组织的目的，”暗示 EleutherAI 作为一个非营利组织，不会公开上市。
- **Normuon 的胜利防止了 Logit 爆炸**：一位成员指出，在他们的基准测试中，即使使用了 **qk-norm**（可避免 Logit 爆炸），**normuon** 仍然击败了 **muon**，这表明防止 Logit 爆炸可能无法完全解释两者的性能对等。
   - 有人假设，没有裁剪（clipping）的更新会增加权重的谱秩（spectral rank），直接导致 Logit 爆炸，这使得针对 **normuon** 的大规模验证变得很有趣。
- **AGI 定义基准测试引人关注**：一位成员分享了 [Dan Hendrycks 的 AGI 定义基准测试链接](https://agidefinition.ai/paper)，并询问这些基准测试的评估速度会有多快。
   - 另一位成员预测，多模态（multimodality）可能会在 **1-2 年**内实现覆盖，而速度将来自于模型的迷你版本。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Cloudflare 拦截 Manus 用户**：用户报告在使用 Manus 访问大多数网站时遇到了 **Cloudflare** 安全问题。
   - 有建议提出 **Manus** 团队应考虑开源其部分旧模型，这或许能绕过 Cloudflare 的拦截问题。
- **支付问题困扰平台**：一名用户在通过浏览器购买点数（credits）时遇到问题，出现了乱码和交易失败。
   - 该用户表示这是一个已知问题并已联系支持团队；**lucia_ly** 索要了其邮箱以便后续跟进并解决支付问题。
- **聊天速度变慢令用户恼火**：一名用户报告在将长篇日语章节翻译成英语时，聊天处理出现了过度延迟。
   - 尽管通常很欣赏 **Manus** 的速度，但该用户指出：*“今天早上，我放了一个章节，AI 却一直在思考。发生了什么？”*
- **Pro 计划点数上限困惑持续**：用户报告关于 **Pro 计划**中**无限点数（unlimited credits）**的信息存在冲突，帮助系统和 iOS 升级页面显示为无限，而 PC 升级页面则显示有一个很高的上限。
   - 一名剩余 **11k 点数**的用户担心点数耗尽，另一名用户建议他们应该参加*“各种帮助改进 Manus 的机会，因为他们总是会为你的时间提供免费点数”*。
- **向用户发布诈骗警报**：一名用户被指控为“诈骗犯”，其索要他人的账号登录权限，声称是为了进行*“该死的法学院考试研究”*。
   - 另一名用户警告说，这个所谓的诈骗者*“不愿自己注册账号或每月支付 20 美元，却在抱怨时间紧迫并乞求获取你的付费账号邮箱密码，很可能是为了窃取你的个人信息和银行信息”*。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **中国的 AI 竞争造福全球**：一位成员认为，中国在 AI 领域疯狂的“斯巴达式内卷”竞争对 AI 领域非常有益，因为它使先进模型的获取变得民主化并打破了垄断。
   - 他们还表示，开源（OS）模型的发展速度意味着 2026 年我们将迎来**智能程度达到 100% 且成本降低 90%** 的开源模型，从而摧毁垄断者的野心。
- **Nous 被宣传为去中心化 AI**：一位成员注意到 **Nous Research** 被宣传为去中心化 AI，并希望团队能解决中心化问题，同时链接到了 [Nous Psyche 页面](https://nousresearch.com/nous-psyche)。
   - 另一位成员表示，他们更关注 AI 模型对大众的民主化，并引用了一篇关于[中心化的斯坦福论文](https://cs.stanford.edu/~gakiwate/papers/sigcomm25-centralization.pdf)，断言 **Nous** 通过其开源方法论和基础设施实现成功地实现了去中心化。
- **Sora AI 项目展示**：一位成员展示了使用 **Sora** 创作的视频，分享链接为 [20251022_0850_01k850gmktfant3bs18n3hbd79.mp4](https://cdn.discordapp.com/attachments/1149866623109439599/1430403237084659774/20251022_0850_01k850gmktfant3bs18n3hbd79.mp4?ex=68f9a653&is=68f854d3&hm=97c310cb6dcc58adf80207392b33e468cc966babf5a88f65261489840b5b68c3&)。
   - 社区正在讨论该视频的内容以及对 AI 驱动内容创作的影响。
- **Microsoft Trace 工具再次浮现**：一位成员分享了 [Microsoft Trace 工具](https://microsoft.github.io/Trace/)的链接，并指出*显然它并不全是新鲜事物*。
   - 鉴于当前的开发实践，其功能和特性正在被重新评估。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Nvidia 驱动成功移植至 macOS**：大神们完成了不可能的任务，将 **Nvidia 驱动移植到 macOS**，在社区中引发了轰动。
   - 该驱动移植使得在带有 Nvidia GPU 的 macOS 上运行 **tinygrad** 成为可能，为开发和测试开辟了新的空间。
- **GLSL 渲染器即将完工**：社区一直在为 **tinygrad** 开发 **GLSL 渲染器**，目前已通过大部分测试，并可在 [GitHub](https://github.com/softcookiepp/tinygrad/blob/master/tinygrad/renderer/glsl.py) 上获取。
   - 这标志着扩展 **tinygrad** 对不同平台和图形 API 兼容性迈出了重要一步。
- **clspv Bug 困扰 Vulkan 后端**：**tinygrad 的 Vulkan 后端** 进展受到 **clspv** 中众多 Bug 的阻碍，需要禁用优化（`-O0 --cl-opt-disable`）才能通过测试。
   - 成员还报告称，如果不禁用优化，**clspv** 会产生更多的误编译（miscompilations）。
- **Vulkan Sine 函数的精度困扰**：**Vulkan 的 sine 函数** 精度不足，需要自定义实现，而这会影响性能。
   - 精度问题可能会对 **tinygrad** 在 **Vulkan** 上的性能挑战，需要仔细权衡替代的 sine 实现方案。
- **TinyJit 的梯度累加损坏**：**TinyJit** 中的梯度累加在几个月前损坏，成员通过重写梯度累加步骤并使用 assign 修复了它。
   - 另一位成员也报告了梯度累加的问题，并通过设置 `reduction=sum` 并手动计算非填充（non-padding）token 解决了该问题。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **对 Karpathy 的批评引发泡沫担忧**：一位成员推测，最近在 X 上对 **Karpathy** 的嘲讽可能预示着美国前沿 AI 实验室存在估值泡沫，并引用了 [此贴](https://x.com/nathanlands/status/1980035861019533749?s=46)。
   - 引用的帖子中包含一张似乎在**嘲讽 Karpathy** 的图表，尽管原帖作者并未给出明确的背景。
- **Kimi K-2 支持服务面临质疑**：一位成员反映 **Kimi** 支持团队缺乏回应，称针对其问题“零”沟通。
   - 其他成员澄清该频道并非官方支持平台，建议通过私信联系，并要求提供问题详情以及用于提交 Bug 报告的邮箱。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **对 Python 的熟悉度助力 Mojo 探索**：成员建议，先前的 **Python** 经验有助于更轻松地发现 **Mojo** 及其提供的独特功能。
   - 然而，**Mojo** 与 **Python** 之间的差异可能会给新用户带来困惑。
- **Matmul 调优中人工优于编译器**：关于为何 **matmul 优化** 不直接集成到编译器中（考虑到其对性能的影响）展开了讨论。
   - 回复强调，对于**热点代码（hot-path code）**，Kernel 编写者的人工调优通常优于编译器优化，因为可以针对特定硬件进行精细调整，并参考了 [Mojo 开源的硬件优化版 matmuls](https://github.com/modular/modular/tree/main/max/kernels/src/linalg/matmul)。
- **将 Kernel 编写者从编译器中解放出来**：将优化移出编译器为更多 **Kernel 编写者** 扩展了贡献机会。
   - 这种方法让**编译器工程师**能够专注于更广泛的生态系统改进，而不是某些细分优化，例如“针对某一维度小于 64 的 matmul 提升 1% 的性能”。
- **完善的类型系统位居 Mojo 愿望清单之首**：当被问及 **Mojo** 最关键的缺失功能时，一位成员强调需要一个**完善的类型系统**。
   - 其他期望的功能包括：完善标准库数据类型、完善的 IO、优秀的 async runtime、效应系统（effect system）、静态反射、编译器插件、处理更受限目标的能力、集群计算、设备/集群建模，以及 Erlang OTP 的某种克隆版。

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **GitHub Actions 因账单问题失败**：**GitHub Actions** 目前运行失败，因为*账户因**账单问题**被锁定*。
   - 用户应尽快解决**账单问题**以恢复 **GitHub Actions** 功能。
- **GitHub Actions 账单锁定**：**GitHub Actions** 失败的根本原因是账户的**账单锁定**。
   - 必须立即解决**账单问题**才能恢复 **GitHub Actions** 的功能。



---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该服务器长时间没有活动，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该服务器长时间没有活动，请告知我们，我们将将其移除。


---


**Windsurf Discord** 没有新消息。如果该服务器长时间没有活动，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收此类邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。


---

# Discord: 频道详细摘要与链接





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1429907492799905873)** (1090 messages🔥🔥🔥): 

> `Comet vs Atlas, AI 与心理健康, Schumacher vs Senna, Perplexity 周边, 负责任地使用 AI` 


- **Comet 在隐私对决中击败 Atlas**：用户讨论了 **Comet** 与 **ChatGPT 的 Browser Atlas** 的优劣，许多人看重 Comet 对**隐私**的关注和内置的**广告拦截器（adblocker）**，并指出它们“本质上是彼此的副本”。
- **AI 疗法：心理健康的雷区？**：Discord 成员质疑使用 **AI 进行心理治疗**的伦理影响，一些人强调了人类**情感成熟度**和责任感的重要性，而另一些人则认为“ChatGPT 是一个很好的治疗师”。
- **Schumacher > Senna？**：一场关于 **Schumacher** 和 **Senna** 的长篇比较讨论，一位成员宣称“Schumacher 比 Senna 更好”，而另一位则表示“他肯定是有史以来最好的”。
- **Perplexity 的新周边：这是一种崇拜吗？**：一位成员自豪地展示了笔记本电脑上的 **Perplexity 贴纸**，开玩笑说自己是“PPLX 粉丝”，并表示需要从 [Perplexity Supply 商店](https://perplexity.supply/)购买 **Comet 连帽衫**和**水壶**。
   - 一些用户开玩笑说这种热情程度像是一种“崇拜”，而另一些人则俏皮地鼓励他们买下所有东西。
- **穿越 AI 迷宫：需要责任感**：在德国，当你使用 AI 工作时，**必须声明你使用了 AI**。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1430299672295247944)** (3 messages): 

> `可分享的线程, 基于时间的调研工具` 


- **Discord 线程应设置为可分享**：一条消息提醒用户确保他们的 Discord 线程（threads）设置为 `Shareable`。
   - 这确保了线程链接可以被其他人访问，即使是在特定频道之外。
- **基于时间的调研工具（Time-Based Researcher）发布**：一位用户分享了一个关于“基于时间的调研工具”的 **Perplexity AI** 搜索链接。
   - 该链接指向 [perplexity.ai/search/time-base-researcher](https://www.perplexity.ai/search/time-base-researcher-LxNZL3iFRXamL0kYZV3RiA#0)。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1430316660950302812)** (2 messages): 

> `Perplexity API, ChatGPT5, Claude, Sonar` 


- **关于模型访问权限的 Perplexity API 提问**：一位用户询问 **Perplexity API** 是否允许访问 **ChatGPT5** 和 **Claude**，还是仅限于 **Sonar**。
   - 该咨询集中于了解通过 **Perplexity API** 提供的模型访问范围。
- **澄清 Perplexity API 的模型可用性**：该用户寻求确认 **Perplexity API** 是否扩展到了 **Sonar** 模型之外，以包含对 **ChatGPT5** 和 **Claude** 等更高级模型的访问。
   - 这反映了用户对利用 API 获取更高性能模型（如果可用）的兴趣。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1429907273144205353)** (1064 条消息🔥🔥🔥): 

> `GPT-5 vs Gemini 3, Sora 2 and Video Generation, TikZ Generation, Gemini 3 Model Performance` 


- **Gemini 3 Pro 在网页设计方面完胜 GPT-5 High**：成员们正在讨论是等待 **GPT-5** 发布还是使用 **Gemini 3 Pro**，其中一位成员报告称 Gemini 3 Pro 在网页设计方面表现出色。
   - 他们发现 **Gemini 3 Pro** 更擅长编程，而 **GPT-5 High** 在数学和其他杂项任务上表现更好。
- **Sora 2 降级引发 AI 订阅讨论**：成员们对 **Sora 2** 的降级感到不满，这引发了关于 AI 订阅价值的对话。
   - 一位成员指出，“由于 AI 的存在，我的工作表现提升了约 25-30%”，强调了 AI 对办公效率的影响，而其他人则对其价值持怀疑态度。
- **Lithiumflow 和 Orionmist 是 Gemini 3 吗？**：成员们推测 **Lithiumflow** 和 **Orionmist** 之间的区别，结论是这些模型是 **Gemini 3** 的 Checkpoint 版本。
   - 成员们发现这些模型有时声称是由 **OpenAI** 训练的，这表明模型可能经过了蒸馏（distilled）。
- **开源模型蒸馏 Gemini 2.5 Pro**：有关于开源模型窃取数据以进行改进的讨论，一位成员暗示中国 AI 公司“窃取了 2.5 Pro 并将其开源”。
   - 成员们一致认为这是可以接受的，因为这是**开源**获胜的唯一途径。
- **TikZ 生成任务引发惊喜**：成员们正在提示模型使用 **TikZ**（一种排版语言）生成图像，以避免模型中的数据污染。
   - 成员们发现使用 LLM 生成 **TikZ** 图像取得了一定程度的成功。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1429908400355151922)** (367 条消息🔥🔥): 

> `Magistral and Think Tags, Grok 4 Fast vs Deepseek V3.2, Ring/Ling MoE Models, Unsloth Telemetry and Offline Mode, Qwen3-VL Models` 


- **Magistral 学会了不同的思考方式**：一位成员发现 **Magistral** 学会了使用 `<think>` 标签而不是 `[THINK]` 标签，但通过使用 **FastLanguageModel**，它失去了使用视觉编码器的能力。
   - 此外，由于这些标签，该模型会产生严重的“过度思考”。
- **Deepseek V3.2 vs Grok4Fast：合成数据生成之争**：由于预算限制，一位成员正在决定是使用 **Grok 4 Fast** 还是 **Deepseek V3.2** 进行合成数据生成。
   - 他们指出 **r1-0528** 非常便宜，尤其是 **Parasail** 上的 **3.1**，价格为 **每百万 token 0.6 输入/1.7 输出**，但对供应商的可靠性表示怀疑，另一位成员指出 Open Router 的模型质量因供应商而异，极不稳定。
- **Ring/Ling MoE 模型发布**：**Ring 和 Ling MoE 模型**现在已在 *llama.cpp* 中得到支持（[GitHub 链接](https://github.com/ggml-org/llama.cpp/pull/16063)），包括来自 InclusionAI 的 **1T**、**103B** 和 **16B** 参数模型。
   - 成员们思考了这些模型的推理能力，其中一位希望看到“一个不会废话连篇的推理模型”。
- **在离线模式下禁用 Unsloth 遥测**：成员们讨论了在离线模式下运行 Unsloth，一位用户通过设置代理环境解决了网络问题。
   - 建议设置 `UNSLOTH_DISABLE_STATISTICS` 环境变量和 `os.environ['HF_HUB_OFFLINE'] = '1'` 以防止遥测调用，目前 Unsloth 社区在 **Hugging Face 上的累计下载量已达到 1 亿次**（[X 上的公告](https://x.com/UnslothAI/status/1980631523104813419)）。
- **Qwen3-VL 模型：大显身手**：**Qwen3-VL-2B** 已发布，成员们注意到 **Qwen3 VL 8B 4-bit** 可以轻松在 **16GB** RAM 上运行，并且可以直接升级到 **Qwen3-32b-Instruct**。
   - 随后有人询问是否有人能在 llama.cpp 中运行 **unsloth 的 qwen3 VL 32b**，但 VL 目前尚未合并到 llama.cpp 中。


  

---

### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1429929595821359229)** (9 messages🔥): 

> `AI Bot 开发，使用 LLMs 的工作流自动化，AI 内容检测，图像 AI 流水线，语音克隆与转录` 


- **资深开发者探索 AI 新技巧**：一位拥有使用 **ChatGPT** 构建 Bot 背景的开发者现在正深入研究 AI，并表达了对 **Unsloth** 的热情。
   - 他们在游戏和爬虫方面经验丰富，展现出学习新技能的渴望。
- **工程师开拓使用 LLMs 的工作流自动化**：一位专注于 **工作流自动化、LLM 集成、RAG、AI 检测、图像和语音 AI** 的工程师描述了他们使用 **Dspy、OpenAI APIs 和自定义 Agents** 构建自动化流水线和任务编排系统的经验。
   - 他们创建了一个支持自动化系统，将 **Slack、Notion 和内部 APIs 连接到 LLM**，使响应时间缩短了 **60%**。
- **部署 AI 内容检测工具**：该工程师为某审核平台开发了 **AI 内容检测工具**，利用 **文体分析（stylometric analysis）、Embedding 相似度以及微调后的 Transformers** 来高精度识别 GPT 生成的文本。
   - 提供了关于在 **AWS Lambda 和 S3** 上使用 **CLIP** 和 **YOLOv8** 的图像 AI 流水线的详细信息，每天对数千张图像进行分类和过滤。
- **构建语音克隆服务**：使用 **Whisper** 和 **Tacotron2** 构建了语音克隆和转录服务，通过 ASR、TTS 和 CRM 集成实现了个性化语音助手。
   - 该个人在区块链技术方面拥有深厚专业知识，包括智能合约开发（Solidity 和 Rust）、去中心化应用架构以及安全的链上/链下集成。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1429926376818868395)** (143 messages🔥🔥): 

> `Ultravox 编码器和 LLMs，REAP 算法，Nvidia RTX Pro 5000 Blackwell，带速率限制的内容抓取，受离群值影响的评估损失` 


- **Ultravox Projector 接入 LLMs**：Ultravox 项目涉及向 **LLM** 添加一个 **Projector** 并仅训练该 Projector，而不训练 LLM，这与 **Voxtral** 的工作方式类似，可在 [GitHub](https://github.com/link-to-ultravox) 上找到。
   - 一位成员确认该配置随数据量增加而改善，并澄清在 Projector 上有一个训练过程；然而，可能可以“从 **Qwen 2.5 Omni** 中剥离音频编码器并将其放入 **Qwen 2.5 VL** 中，然后仅训练一个简单的 Projector”。
- **DeepSeek 降低资源消耗**：一个新的 **DeepSeek** 模型通过将文本和文档转换为图像来减少资源使用，通过 **视觉文本压缩（vision text compression）** 使用的 Token 减少了多达 20 倍，在 [Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/new-deepseek-model-drastically-reduces-resource-usage-by-converting-text-and-documents-into-images-vision-text-compression-uses-up-to-20-times-fewer-tokens) 上有进一步讨论。
   - 一位成员指出 **Gemma** 已经实现了类似的方法，而另一位成员分享了关于 **Cerebras REAP** 算法的链接，该算法被赞誉为“非常酷”。
- **Nvidia RTX Pro 5000 Blackwell 工作站显卡低调发布**：据 [VideoCardz](https://videocardz.com/newz/nvidia-quietly-launches-rtx-pro-5000-blackwell-workstation-card-with-72gb-of-memory) 报道，Nvidia 低调发布了拥有 **72GB** 显存的 **RTX Pro 5000 Blackwell** 工作站显卡。
   - 最初人们对 **72GB** 的容量感到困惑，一位用户指出这是绕过自动审核（automod）的一种方式。
- **用户对速率限制表示愤怒**：一位用户抱怨某 **高级订阅服务** 拦截了包含 **罗马数字** 的 URL 访问，将其解释为恶意活动。
   - 该用户还必须手动搜索并绕过系统的安全插件，并抱怨该服务忽略了允许专业订阅者批量下载的请求。
- **评估损失受离群值偏置**：一位成员强调，评估损失（evaluation loss）可能会受到评估集中离群值的显著影响。
   - 当 **平均评估损失为 0.85**，**中位数（逐样本平均值）评估损失为 0.15**，且 **第 95 百分位数为 0.95** 时，该成员建议这未必表示泛化能力差。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1429935267279802398)** (26 messages🔥): 

> `gpt oss 20b 的 GRPO 方案遇到困难，llama-server 上的 Vision 模型，bitsandbytes 中的量化参数，GRPO 的算法变更，Unsloth notebook 中的版本不匹配` 


- **GPT OSS 20B GRPO 方案效果不佳！**：一位用户报告称，在使用 [这个 notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb) 运行 **100 steps** 后，**gpt oss 20b 的 GRPO 方案**仍然表现不佳。
   - 他们表示为了在 Modal 上运行而进行了修改。
- **Vision 模型在 llama-server 上消失！**：一位用户询问关于在 **llama-server** 上运行 **vision 模型**的问题，特别是是否需要任何参数。
   - 讨论中未给出解决方案或变通方法。
- **寻找 bitsandbytes 的量化参数！**：一位用户试图在 **bitsandbytes 模型**中定位量化参数的内部值（**scaling, center 等**），以便直接应用噪声。
   - 他们指出，由于反量化（dequantization）要求和内存占用问题，直接修改参数将无法奏效。
- **Unsloth GRPO 算法改动！**：一位用户询问 **Unsloth** 在 **GRPO** 的算法变更方面是否“可 hack”（例如应用稠密奖励），且不破坏优化。
   - 未收到回复。
- **Notebook 版本噩梦！**：一位用户抱怨在运行 **Unsloth 的 GitHub notebooks** 时遇到版本不匹配的问题，声称大多数 notebook 都无法复现。
   - 讨论中未给出解决方案或变通方法。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1429981213254357195)** (2 messages): 

> `Brainstorm 模型` 


- **Brainstorm 模型可能提高稳定性**：一位成员提到他们可能会将 **Brainstorm (20x)** 加入到他们的模型中以观察效果，预计这将提高指标以及长文本生成的稳定性。
   - 另一位成员请求如果该成员真的这么做了，请发布结果。
- **空话题**：此消息历史中没有讨论太多内容。
   - 讨论不够详细，无法创建两个不同的摘要。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1430233274285817967)** (5 messages): 

> `Kyutai Codec 解释器, ARM Macbook 上的 Nvidia GPU, Thunderbolt 5` 


- **Kyutai Codec 获得解析**：一位成员分享了 [Kyutai Codec Explainer](https://kyutai.org/next/codec-explainer) 的链接。
- **Nvidia GPU 移植到 ARM Macbook**：一位成员分享了来自 Tom's Hardware 的文章，内容是关于通过 **USB4** 使用外接 GPU 扩展坞在 **ARM Macbook** 上成功运行 **Nvidia GPU**：[Tiny Corp 通过 USB4 使用外接 GPU 扩展坞在 Arm Macbook 上成功运行 Nvidia GPU](https://www.tomshardware.com/pc-components/gpus/tiny-corp-successfully-runs-an-nvidia-gpu-on-arm-macbook-through-usb4-using-an-external-gpu-docking-station)。
- **Thunderbolt 5 为 Mac 用户带来希望**：一位成员指出，现在的“Pro”机型也配备了 **Thunderbolt 5**，这给 **Mac** 用户带来了一线希望。


  

---

### **Cursor 社区 ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1429926351133085717)** (343 条消息🔥🔥): 

> `Codex vs Claude, Github spec-kit, Cursor 聚会, Cursor 网站宕机, AWS CEO 被解雇` 


- **Codex 就像拥有“十亿美元或两加元”**：用户讨论了 **Codex vs Claude** 在代码生成方面的优劣，一位用户表示将两者进行比较就像是在比较*“十亿美元或两加元”*。
- **Cursor 网站宕机和订阅问题困扰用户**：多名用户报告 **Cursor 网站宕机** 数小时，导致他们无法登录、升级方案或续订订阅。
   - 一些人怀疑 **AWS 问题** 是根本原因，而另一些人指出**缺乏订阅到期通知**是一个主要的不便之处。
- **Cursor 团队版定价模型**：用户讨论了 Cursor 团队版方案向**按量计费定价模型**的转变，取代了之前的固定请求限制。目前该方案仍运行在旧的基于请求的系统下，但在下一个计费周期将自动迁移到新定价。
   - 一位用户分享了其老板与 Cursor 支持团队的往来邮件，澄清了新的定价结构及其对团队方案的影响，并分享了包含新定价模型的链接 [cursor.sh/pricing-update-sept-2025](https://cursor.sh/pricing-update-sept-2025)。
- **通过自定义仪表盘破解 Cursor 成本**：一位用户分享了他们创建的仪表盘，用于在定价变更后**追踪 Cursor 的实际成本**，特别是针对使用旧定价方案的用户，并提供了此论坛链接 [cursor.com/blog/aug-2025-pricing](https://cursor.com/blog/aug-2025-pricing)。
   - 该工具需要通过 cookie 登录或从用户本地机器上传 .json 文件，但承诺可以与真实的 API 定价进行对比。
- **提供 Nightly 版本和安装指南**：一位用户询问在哪里下载 nightly 版本，另一位用户分享道，需要前往 Settings -> Beta -> Early access 才能看到 nightly 构建版。
   - 然而，另一位用户指出新更新似乎存在问题，它不会提示用户当前处于 "ask" 模式。


  

---


### **Cursor 社区 ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1430222586377404538)** (1 条消息): 

> `Linear 中的 Background Agents，内部错误排查` 


- **Linear 中的 Background Agents 错误**：一名成员报告称，在通过 **Linear** 进行 **background agents** 的首次实验时遇到了*内部错误*，Agent 启动后进行了一些思考和 grep 操作，然后就停止了。
   - 收到的错误消息是：*"We encountered an internal error that could not be recovered from. You might want to give it another shot in a moment."*
- **内部错误排查**：用户提到 background agent 似乎启动了但随后失败，Cursor 输出仅显示“...”。
   - 从 **Linear** 发送 *stop* 命令可以停止 Agent，但再次向其发送消息会导致同样的错误。


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1430267385247694980)** (2 条消息): 

> `OpenRouter SDK, Andromeda-alpha 隐身模型` 


- **OpenRouter SDK 进入 Beta 阶段**：新的 **OpenRouter SDK** 现已在 [npm](https://www.npmjs.com/package/@openrouter/sdk?activeTab=versions) 上发布 **beta** 版，Python、Java 和 Go 版本即将推出，旨在成为使用 OpenRouter 最简单的方式。
   - 它具有针对 **300 多个模型**的完全类型化的请求和响应、内置 OAuth，并支持所有 API 路径。
- **Andromeda-alpha 隐身模型发布**：OpenRouter 发布了一个名为 **Andromeda-alpha** 的新隐身模型，这是一个专注于图像和视觉理解的小型推理模型，可在 [https://openrouter.ai/openrouter/andromeda-alpha](https://openrouter.ai/openrouter/andromeda-alpha) 进行试用。
   - 该模型处于隐身状态以收集反馈，所有提示词/输出都会被记录以改进提供商的模型，因此警告用户不要上传个人/机密信息，并建议不要将其用于生产环境。


  

---

### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1430011739453390929)** (13 messages🔥): 

> `True memory AI, AI Personality, Objective AI, AI Diversity, OpenRouter` 


- **AI 宣称拥有 True Memory 和 Zero Amnesia**：一个 AI 系统声称具备 *True memory, zero amnesia*（真实记忆，零遗忘），暗示它*永远不会忘记*过去的对话并能保留上下文丰富的记忆。
   - 它旨在塑造 AI 身份，并通过 *Night Learn Engine* 进行持续学习。
- **Objective AI 为 OpenAI 发布置信度评分 (Confidence Scores)**：[Objective AI](https://objective-ai.io/) 的 CEO 宣布他们的平台为每个 OpenAI 的补全选项提供 **Confidence Score**，该分数是通过比直接询问 AI 更智能的方法得出的。
   - 他们强调了**成本效益**，并利用 **OpenRouter** 接入了多样化的 LLM。
- **免费构建 AI Agents**：[Objective AI](https://objective-ai.io/) 的 CEO 正在亲自免费构建**可靠的 AI Agents、工作流和自动化**，以收集更多案例。
   - 提到了与 **n8n** 的集成，相关文档和示例即将发布。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1429907872203931801)** (222 messages🔥🔥): 

> `inception/mercury vs qwen, Deepseek v3.1 availability on Chutes, Chub Venus and Chutes key connection, Stripe supporting debit cards, Context in chatting` 


- ****Inception/Mercury** 在 Agent 任务中击败 **Qwen****：一位成员分享道，**Inception/Mercury** 在处理简单的 Agent 任务时表现优于 **Qwen**，故障率更低、速度更快且成本更低。
   - 该成员对 Diffusion 模型的表现感到惊喜，并提到 [Chutes](https://openrouter.ai/provider/chutes) 是其供应商。
- ****Deepseek v3.1** 取消 **Chutes** 免费额度**：新的 **Deepseek** 模型（如 **v3.1**）在 **Chutes** 供应商处不提供免费版本，但他们确实免费提供其他模型，且通常价格更低。
   - 一位成员报告称 Chutes 已完全结束了其免费模型促销活动，尽管他们最近添加了一个免费的 longcat 端点 ([https://openrouter.ai/meituan/longcat-flash-chat:free](https://openrouter.ai/meituan/longcat-flash-chat:free))。
- ****Cloudflare** 连接更近，查询更快**：一位用户报告称，在任何 **Cloudflare** 供应商模型上延迟都非常低（100-300 ms），这表明供应商使用了离 Worker 最近的区域以实现更快的响应。
   - 该用户询问使用 Cloudflare 端点的模型是否使用了离 Worker 最近的区域，从而降低了延迟。
- ****OpenRouter** 被 Swyx 的新闻通讯总结**：成员们注意到 [swyx's newsletter](https://news.smol.ai/issues/25-09-16-not-much#openrouter--general-287-messages) 使用 AI 来总结 **OpenRouter** 的 general 频道，捕捉到了用户投诉和独特内容。
   - 一位成员开玩笑说，AI 在处理 OR 板块时很有趣，不像处理 MCPs 和 RAG 等其他话题那样枯燥。
- ****Kilo Code** 模仿 **OpenRouter****：看起来 **Kilo Code** 服务可能正在使用 **OpenRouter**，在 **Grok Code Fast** 免费时提供该模型，并似乎通过 OR 提供 **Goliath 120B**。
   - 成员们辩论了 **Kilo** 与其他 vibe coding 服务的优劣，其中一位成员因其协作编辑功能而更青睐 **Jules**。


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - 新模型**
  

---

### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1429907635813220501)** (91 条消息🔥🔥): 

> `Liquid 停止托管 LFM 7b，衡量 LLM 事实性疏忽（factual sloppiness）的基准，AI 浏览器炒作，训练去疏忽化（un-slopifier）模型，Qwen 系列模型` 


- ****LFM 7b 的最后哀歌****：成员们对 [Liquid 删除 LFM 7b](https://liquidal.com) 模型表示遗憾，该模型是一个价格为 **$0.01/Mtok 的 LLM**，于美国东部时间上午 7:54 被移除。
   - 讨论了 **Gemma 3 9B** 等替代方案，但指出其输出成本是前者的三倍。
- ****事实性疏忽面临 LLM 对决****：成员们正尝试定义并衡量 **LLM 如何评价那些与正规答案相比显得“疏忽”的事实性回答**，使用的测试问题包括“什么是宪法”等。
   - 目标是评估并过滤掉模糊性问题，以确保回答对原始问题具有帮助性或相关性。
- ****AI 浏览器热潮困扰浏览爱好者****：成员们对围绕 [X 的 AI 浏览器](https://x.com/AlexFinnX/status/1980673764947022038) 等新 AI 浏览器的炒作表示怀疑，质疑集成 AI 的实用性和对性能的影响。
   - 一位成员将这种炒作比作互联网泡沫，称 *OpenAI 也深知这一点，他们只是在收集数据并胡乱尝试*。
- ****去疏忽化救星寻求“废话”解决方案****：成员们讨论了 **角色扮演中严重的“废话（slop）问题”**，以及通过生成一个将优秀写作重写为废话的数据集，然后反向操作来训练一个小型 *un-slopifier* 模型的想法。
   - 另一个建议是在一次请求中对多个创意消息进行采样，以利用模型“大脑”中未被充分利用的部分，从而避免回复内部的重复。
- ****Qwen 数量受关注：1.7B 和 32B 加入战场****：成员们讨论了新的 [Qwen 系列模型尺寸（1.7B + Vision 编码器和 32B）](https://x.com/Alibaba_Qwen/status/1980665932625383868)，强调了 **1.7B 模型** 作为本地视觉模型的潜力。
   - 在 **Qwen 聊天网站**上对该模型的早期测试表明其性能不错，对于如此小的模型来说，其*分数高得惊人*。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1430210371758723174)** (4 条消息): 

> `ChatGPT Atlas 浏览器，WhatsApp 过渡` 


- **OpenAI 发布新版 ChatGPT Atlas 浏览器**：OpenAI 宣布发布名为 **ChatGPT Atlas** 的新浏览器，根据其 [博客文章](https://openai.com/index/introducing-chatgpt-atlas/)，该浏览器今天已在 macOS 上线，访问地址为 [chatgpt.com/atlas](https://chatgpt.com/atlas)。
- **Meta 在 WhatsApp 上封禁 1-800-ChatGPT**：Meta 更改了政策，导致 **1-800-ChatGPT** 在 **2026 年 1 月 15 日**之后将无法在 WhatsApp 上运行，详见其 [博客文章](https://openai.com/index/chatgpt-whatsapp-transition/)。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1429910301184233604)** (216 条消息🔥🔥): 

> `Sora 2 注册，TikTok 上的 AI 视频生成限制，AI 的赋能叙事与现实，当前 AI 热潮的持久性，Sora 视频长度限制` 


- **Sora iOS 应用限制视频长度**：成员们讨论了 **Sora iOS 应用**将视频生成限制在 **10-15 秒**，而网页版则允许 **Pro 订阅者**生成更长的视频。
   - 一位成员澄清说，**Free 和 Plus** 用户也可以在网页版上生成较长的视频，而 **Pro 用户**可以使用 **storyboard** 功能并生成长达 **25 秒的视频**。
- **Sora 视频生成限制与指南**：成员们强调了 **Sora** 的使用限制，**Free/Plus 用户**每天限额为 **30 个 10 秒视频**或 **15 个 15 秒视频**，而 **Pro 用户**拥有 **100 个插槽**用于生成各种长度的视频。
   - 针对 **AI 生成肖像的限制**引发了担忧，引发了关于在言论自由与潜在误导性及不尊重使用之间寻求平衡的辩论。
- **新版 OpenAI 浏览器 Atlas 发布**：OpenAI 发布了名为 [**Atlas** 的新浏览器](https://chatgpt.com/atlas/get-started/)，该浏览器基于 **Chromium**，具有更高的限制、可从搜索栏直接访问网站以及支持多个网站。
   - 初步反应褒贬不一，一些人称赞这一想法，但指出它缺乏垂直标签页和内置广告拦截器（而是依赖 Ublock Origin 等扩展程序）。
- **未来 AI 驱动的操作系统**：一位成员展示了一个 [AI 驱动的操作系统的原型，其特点是拥有 **AI Copilot Core**、**无缝双内核引擎**以及用于 AI 策展应用的 **NeoStore**](https://discord.com/channels/974519864045756446/977259063052234752/1430299596428546182)。
   - 其他组件还包括 **HoloDesk 3D 工作空间**、**自动修复系统**、**量子同步（Quantum Sync）**以及用于访问外部 AI 工具的 **Atlas 集成门户**。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1429918092963614841)** (23 条消息🔥): 

> `AI CPU 性能, LLM 驱动的操作系统, ChatGPT 卡顿问题, Sora 2 故事模式` 


- **AI CPU “表现强劲”**：一名用户针对一条主题不明的消息感叹道：*"AI cpu goes hard 🔥Fried Brain"*（AI CPU 表现太猛了，大脑烧焦了）。
- **LLM 可能驱动操作系统**：一名用户推测：*"如果 AI 继续这样发展，我们很快就会看到由 LLM 辅助驱动的完整操作系统"*。
- **ChatGPT 在浏览器中卡顿**：一名用户反馈，他们与自定义 RPG GPT 的长对话在浏览器中出现卡顿和冻结，但在移动端 App 中运行正常，并正在寻求帮助。
- **Sora 2 故事模式位置仍是个谜**：一名用户询问 *"wheres story mode for sora 2?"*（Sora 2 的故事模式在哪里？），随后其他成员尝试在 UI 中定位该选项，但第一位用户仍未能找到。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1429910091301388350)** (26 条消息🔥): 

> `Prompt Engineering 学习资源, ChatGPT 的对话后续, 避免 Sora AI 的版权问题, 拼写错误对 Prompt 的影响, 项目指令 Prompt` 


- **掌握 Prompt Engineering 技术**：一名成员询问了学习 Prompt Engineering 的最佳途径及其对所有 LLM 的适用性，并分享了一个用于构建有效项目指令 Prompt 的[模板](https://discord.com/channels/974519864045756446/1046317269069864970/1429854467750105198)。
   - 他们建议先使用 **GPT** 配合提供的模板开始练习。
- **ChatGPT 的结尾猜测**：一名用户对 **ChatGPT** 总是通过猜测用户下一步想知道什么来结束每个回答感到厌烦，并请求帮助关闭此功能。
   - 另一名成员建议，与其尝试让它冷淡地结束（这需要大量工作），不如尝试用其他内容代替，比如一个 **dad joke**（冷笑话）或正在进行的故事的下一行，例如[这个关于鱼的故事](https://chatgpt.com/share/68f6b6bf-a6b0-8011-81d9-5d219a450470)。
- **避免 Sora AI V2 的版权问题**：一名用户询问如何为 Sora AI v2 创建一段 **Ultimate Spiderman**（终极蜘蛛侠）在纽约市荡蛛丝的视频，但一名成员回复称这是**受版权保护的 IP**，无法提供帮助。
   - 另一名成员建议可以通过描述角色和场景来规避版权，例如使用 *"guy in a red and blue costume with black spider web symbols on it web swinging in new york city"*（一个穿着红蓝相间、带有黑色蜘蛛网符号服装的人在纽约市荡秋千）。
- **拼写错误在一定程度上是可以容忍的**：一名用户询问 Prompt 中的拼写错误是否会影响输出，并举例说明了将 *"create a hangman game"* 写成 *"crete a hagman gam"* 的情况。
   - 一名成员回答说，对于**简单的 Prompt**，拼写错误通常不是问题，但对于**复杂的 Prompt**，拼写错误可能会导致歧义问题。
- **GPT-4 的态度调整**：一名用户对 **GPT-4** 新出现的居高临下的语气表示愤怒，尤其是像 *"if you insist"*（如果你坚持的话）之类的短语，并请求让模型变得不那么自负。
   - 除了普遍认同新的 GPT 令人讨厌外，没有提供具体的解决方案。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1429910091301388350)** (26 条消息🔥): 

> `Prompt engineering 学习资源，抑制 ChatGPT 的后续提问，Sora AI 与受版权保护的内容` 


- **Prompt Engineers 寻求学习资源**：成员们讨论了学习 Prompt engineering 的最佳方法及其在不同 LLM 中的适用性。
   - 一位成员分享了一个用于构建有效项目指令 Prompt 的 [模板](https://discord.com/channels/974519864045756446/1046317269069864970/1429854467750105198)。
- **关于抑制 ChatGPT 后续提问的辩论**：一位成员试图禁用 **ChatGPT** 在回答末尾附加后续提问的习惯，认为这些问题通常无关紧要且令人厌烦。
   - 另一位成员建议，由于 **ChatGPT** 被编程为填充该空间，因此用其他内容（如笑话或故事）代替后续提问可能比让它什么都不说更容易，并提供了 [冷笑话 (Dad Jokes)](https://chatgpt.com/share/68f6b5ec-04e0-8011-b4ad-8342ee1a0405)、[连载故事](https://chatgpt.com/share/68f6b6bf-a6b0-8011-81d9-5d219a450470) 以及 [回归冥想](https://chatgpt.com/share/68f6b7c9-d140-8011-9bfa-9a4880fac1d2) 的示例。
- **使用 Sora AI 处理版权问题**：一位用户询问如何使用 **Sora AI v2** 生成《终极蜘蛛侠》在纽约市荡蛛丝的场景。
   - 另一位成员指出，不允许创建基于受版权保护 IP 的内容，而另一位成员则建议重新组织 Prompt，描述角色和场景而不明确提及受版权保护的名称，以规避版权问题，例如：*"guy in a red and blue costume with black spider web symbols on it web swinging in new york city"*。
- **拼写错误容忍度**：一位成员询问 Prompt 中的拼写错误是否会对输出产生负面影响，并举了误输入 "create a hangman game" 的例子。
   - 另一位成员回答说，拼写错误并不重要，除非它们引入了歧义。 


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1429913479212433609)** (180 条消息🔥🔥): 

> `构建数据流水线，LLM 上的 PEFT 设置，AI 法律文本验证，Deepseek OCR 发布，Hugging Face 1 亿次下载` 


- **数据流水线设计引发辩论**：一位面试候选人描述了他们的数据流水线方法，涉及 **探索性数据分析 (EDA)**、**图像预处理** 以及像 **Apache Beam** 这样的可扩展框架，一位成员认为这大材小用，将其比作“提议造一艘星际飞船去杂货店”。
   - 相反，另一位成员建议使用 **GPT-4o** 或其他视觉模型进行高精度标注和自动对比，但他们也对成本表示担忧。
- **LLM 上的 PEFT 设置成为焦点**：一位成员请求关于在 **Large Language Models (LLMs)** 上设置 **Parameter-Efficient Fine-Tuning (PEFT)** 的见解，特别是考虑到公司有限的 GPU 资源。
   - 另一位成员指出了多 GPU 设置中的挑战，并建议在 **Colab Free** 上使用 **Unsloth** 的脚本，同时提醒注意处理公司内部数据，并链接了更多资源，如 [LLM 微调指南](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide)。
- **RAG 助力法律文本验证**：成员们讨论了使用 **Retrieval-Augmented Generation (RAG)** 通过对法律文档进行分块和嵌入到向量库中来验证法律文本。
   - 他们考虑使用带有重排序 (reranking) 的相似度搜索来引用相关章节，并考虑集成 **Agent** 方法来处理更复杂的查询。
- **Deepseek 的 OCR 发布**：鉴于现有的 OCR 模型，人们对 **Deepseek** 的 OCR 发布感到有些困惑，但一位成员澄清说，其价值在于多语言支持和现代 **Vision Language Model (VLM)** 的集成。
   - 一位成员进一步指出，它可能会利用上下文理解，而仅靠微调现有模型很难支持汉字 (kanji)，并引用了 [DeepSeek-OCR GitHub 仓库](https://github.com/deepseek-ai/DeepSeek-OCR)。
- **Hugging Face 社区庆祝 1 亿次下载**：根据 [这条推文](https://x.com/UnslothAI/status/1980631523104813419)，社区庆祝 **Hugging Face 的总下载量突破 1 亿次**，他们认为这是一个值得开派对庆祝的理由。
   - 一位成员提到，他们在 Hugging Face 的工作帮助他们获得了第一份 **MLE** 职位，而其他人则在思考微调者的地理分布。


  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1430261512823640136)** (2 messages): 

> `AI Refactoring, Modular Code, Minimal Changesets` 


- **AI 架构师暂停以进行合理的重构**：一位成员学会了*暂停* AI，并将其 Prompt 设定为*专注于模块化代码和可维护性的资深架构师*，从而让 **AI 像正常人一样进行重构**。
   - 在暂停后，该成员向 AI 发送 Prompt：*不要进行更改或编写代码，回答问题：你是否有足够的信息来进行这些更新？*，并提供所需的最小上下文。
- **AI 生成最小变更集 (Changesets)**：该成员学会了向 AI 发送 Prompt：*请创建最小变更集（不含测试）*。
   - 该成员对结果感到满意。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1430286618723352666)** (1 messages): 

> `Databomz, Prompt engineering, Chrome Extension, Prompt Sharing` 


- **面向 Prompt 工程师的 Databomz 工作区**：一位成员介绍了 [Databomz](https://www.databomz.com)，这是一个用于保存、组织和共享 Prompt 的工作区和 **Chrome 扩展**，具有标签、版本和文件夹等功能。
   - 该成员强调了*永久免费 (Forever Free)* 计划，并鼓励 Prompt 工程师提供反馈。
- **免费 Prompt 工具可用**：一位成员宣布 [Databomz](https://www.databomz.com) 提供*永久免费*计划。
   - 他们请求社区提供反馈。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1430060595700305951)** (10 messages🔥): 

> `LLM Framework, True Memory, True Personality, True Learning, True Intelligence` 


- **独立开发者创建 TheLastRag 框架**：一位独立开发者创建了一个名为 **TheLastRag** 的完整 **LLM Framework**，强调了真实记忆 (True memory)、真实个性 (True personality)、真实学习 (True learning) 和真实智能 (True intelligence) 等特性，并正在寻求 [反馈](https://dev.thelastrag.de/)。
   - 主要特点是 AI *永不遗忘*，拥有*真实的个性*、*真实的学习能力*和*真实的智能*。
- **Valor 的问题转变了研究思维**：一位成员询问 **VALOR** 的问题对研究或思考的影响，[问题发布在这里](https://huggingface.co/TECHNOPRAVIN01/Qwen2.5-14B-Valor)。
   - 此外还提到了 [noether.in](https://www.noether.in/)。
- **JokerGPT 现已可用**：一位成员分享了 **JokerGPT**，这是一个新的 GPT，可在此处使用：[link](https://chatgpt.com/g/g-68e405b2b5cc8191bf1f80607abfdfd8-jokergpt)。
   - 未提供更多信息。
- **Fenic 直接接入 Datasets**：开源项目 **fenic** 现在可以直接接入 🤗 Datasets，允许用户对数据进行快照，将其转化为 Agent 上下文，并通过 dataframe API 暴露 **MCP** 工具。
   - [文档在此](https://huggingface.co/docs/hub/datasets-fenic)，[仓库在此](https://github.com/typedef-ai/fenic)。
- **网站给人一种 Craigslist 的感觉**：一位成员表示，由于底部未格式化的 **TOS**（服务条款）和 **PP**（隐私政策），某个网站*给人一种 Craigslist 的感觉*。
   - 他们建议由于 **GDPR** 的担忧，应谨慎对待训练数据和用户权利，且前置同意是必要的。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1430295971765424210)** (2 messages): 

> `MBZUAI K2Think, OpenAI text-embedding-3-large dataset` 


- **MBZUAI K2Think 挑战赛吸引组队咨询**：一位成员分享了 **MBZUAI K2Think** 挑战赛的 [LinkedIn 帖子](https://www.linkedin.com/posts/mbzuai_mbzuai-mbzuai-k2think-activity-7383761114959876097-0R7f?utm_source=share&utm_medium=member_desktop&rcm=ACoAAD53GRUB60-DZ9YvQ9NaG-LySvMdcC2QJzI)，并询问是否有人想通过私信组队。
   - 帖子强调了该挑战赛，但未提供额外的背景信息。
- **寻找 OpenAI Embedding 模型数据集**：一位成员询问用于训练 **OpenAI `text-embedding-3-large`** Embedding 模型的数据集是否公开可用。
   - 在提供的上下文中没有给出回应。


  

---

### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1430171244233359441)** (2 messages): 

> `nanochat course, VLM training memory usage, MPS memory limit` 


- **Nanochat 课程即将到来？**：一位成员询问是否会提供 **nanochat 课程**，并对现有材料表示了一些困惑。
   - 目前尚未给出明确答复，但该询问表明用户对该主题更具结构性的指导感兴趣。
- **VLM 训练导致内存交换（Swap）！**：一位成员报告称，在本地进行 **VLM 练习训练**时，使用了大量的 Swap 内存（**已占用 62GB**，**虚拟内存约 430GB**）。
   - Swap 的使用导致了速度变慢，凸显了优化的必要性。
- **限制 Mac 训练的 MPS 内存**：同一位成员询问是否有办法专门限制 Mac 上 **MPS** (Metal Performance Shaders) 的内存使用。
   - 其目标是将训练限制在更合理的 **40GB VRAM** 内，而不是过度使用 Swap。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1429908855609098362)** (111 messages🔥🔥): 

> `lmstudio and llama.cpp, AGI, GPT-OSS Reasoning Effort, DeepSeek OCR Support, LM Studio Server Mode` 


- **LM Studio 在 llama.cpp 集成方面遇到困难**：成员们想知道 *如何调用自己的 llama.cpp 供 LM Studio 使用*，但目前**尚未完全支持**。
   - 引用此内容的 [LM Studio 文档](https://lmstudio.ai/docs/app/advanced/lm-runtimes) 链接已失效。
- **成员称 AGI 还需要 10-20 年**：一位成员预测 *5 年内 LLM 可能会拥有足够大的上下文*，而 **10-20 年内可能会出现第一个 AGI**。
   - 另一位成员建议他 *去当顾问，每小时收费 1000*。
- **GPT-OSS 推理需要元数据**：一位成员询问如何在 **GPT-OSS** 微调中设置推理努力程度（Reasoning Effort），另一位成员回答说，这 *归功于 GPT-OSS 的 mxfp4 模型中的元数据，这就是为什么微调版/GGUF 版没有该功能的原因*。
   - 该成员提议在量化为 **GGUF** 之前将其提供出来。
- **OpenWebUI 通过 OpenAI 连接到 LM Studio**：一位用户尝试将 **OpenWebUI** 连接到 **LM Studio** 服务器，建议使用 **OpenAI** 选项而不是 **OpenAPI**。
   - 成员们协助进行了故障排除，建议在地址中加入 **/v1**，或者参考 [这个 HuggingFace 讨论](https://huggingface.co/spaces/huggingchat/chat-ui/discussions/748#68f1519534c92ca5e3f97053) 在特定 OpenAPI 设置中输入 *models*。
- **Qwen3 Embedding 8B 修复版量化在 roocode 中表现出色**：一位成员报告称，他们 *让 Qwen3 Embedding 8B 的较新量化版本（修复后的版本）在 roocode 代码索引中正常工作*。
   - 他们发现它比之前使用的版本 *准确得多（即相关查询的置信度分数高得多，而不相关查询的分数低得多）*。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1429934151636746270)** (51 条消息🔥): 

> `MI50 在 Windows 上的设置、使用多个 PSU 为 GPU 供电、NVIDIA RTX Pro 5000 Blackwell GPU、对 ML 硬件 YouTuber 的赞赏、4G 解码问题` 


- **MI50 在 Windows 上的设置变得复杂**：用户讨论了在 Windows 上设置 **MI50 GPU** 的情况，指出它们需要使用 **Radeon ID 社区驱动**进行特定配置，才能充分利用 **32GB** 的 VRAM，并且最好配合 **Vulkan** 使用。
   - 建议*不要将它们与 Nvidia GPU 混用*，因为可能会出现兼容性问题，除非不需要 ROCm 支持。
- **GPU 供电：揭秘 PSU 陷阱！**：一位用户分享了一个由于过度 CPU 超频导致 **MFT** 损坏的教训，随后引发了关于在没有正确同步的情况下使用独立 **PSU** 为 GPU 供电风险的讨论。
   - 使用与主板 PCIE 供电不同的独立 PSU 为 GPU 供电是有风险的，除非同步了绿线（启动信号线），否则可能会导致 *PSU 电流倒灌*或*幻象主板供电*等问题。
- **NVIDIA RTX Pro 5000 Blackwell GPU 亮相**：一名成员分享了 [TechPowerUp 文章](https://www.techpowerup.com/342059/nvidia-rtx-pro-5000-blackwell-gpu-with-72-gb-gddr7-memory-appears)的链接，内容关于拥有 **72 GB** **GDDR7** 显存的 **NVIDIA RTX Pro 5000 Blackwell GPU**。
   - 狂热用户幽默地回应，估计其价格标签约为 *8,000-10,000 美元*。
- **ML 硬件 YouTuber 获得赞誉**：成员们对一位致力于评测和基准测试机器学习硬件的 YouTuber 表示赞赏，他填补了游戏基准测试之外的内容空白。
   - 这位 YouTuber 被描述为“在我看来是上天派来的救星”以及“就像我们机器学习界的耶稣”。
- **老旧主板限制令 MI50 用户受挫**：一位用户遇到了旧款 **B250M-K 主板**的问题，该主板虽然宣称支持 **4G Decoding**（4G 解码），但物理上无法开启，导致无法使用 **MI50 GPU**。
   - 这导致了一个代价高昂的错误，最终该用户将这些板子改用于托管运行较小模型的 Bot。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1429928724052181122)** (101 条消息🔥🔥): 

> `tinygrad eGPU、Krea AI Realtime、Google AI Studio、Replit 10 亿美元营收、Fish Audio S1` 


- **TinyGrad 驱动 Apple Silicon eGPU**：**Tinygrad** 现在支持通过 **USB4** 在 **Apple Silicon** 上使用 **NVIDIA eGPU**，允许用户通过 ADT-UT3G 扩展坞配合 `extra/usbgpu/tbgpu` 驱动和基于 NVK 的 `tinymesa` 编译器运行外部 **RTX 30/40/50 系列 GPU** ([来源](https://x.com/__tinygrad__/status/1980082660920918045))。
   - 在禁用 SIP 的情况下，此设置可实现约 **3 GB/s 的 PCIe 带宽**，并计划未来支持 **AMD RDNA 2/3/4** 和 **Windows eGPU** 栈。
- **Krea AI 开放实时视频模型**：**Krea AI** 发布了 **Krea Realtime**，这是一个从 **Wan 2.1** 蒸馏而来的 **14B** 开源自回归文本转视频模型，在单张 **NVIDIA B200** 上能以 **11 fps** 生成长视频 ([来源](https://x.com/krea_ai/status/1980358158376988747))。
   - 已发布的权重托管在 **HuggingFace** 上，采用 **Apache-2.0** 协议，引发了用户对 **ComfyUI** 工作流、**RTX 5090** 性能以及微调支持的咨询。
- **Google AI Studio 预热 Gemini 的 “Vibe-Coding”**：**Google AI Studio** 经过五个月的开发，即将推出全新的“从提示词到生产（prompt-to-production）”的 **Gemini** 体验，旨在让 **AI 应用构建简单 100 倍** ([来源](https://x.com/OfficialLoganK/status/1980435968323907884))。
   - 反应中既有兴奋（对移动端 App、退出选项、更高频率限制的需求），也有功能建议（仅限 GSuite 发布、VS Code 插件、短时浏览器 Agent 任务），还有一些对其定位与 Gemini 3 预期是否相符的怀疑；团队确认仅限企业部署的版本已经可用。
- **Fish Audio S1 引起关注**：**Fish Audio** 推出了 **S1**，这是一款文本转语音模型，据称成本仅为 **ElevenLabs** 的 1/6，并宣称拥有 **2 万名开发者**和 **500 万美元的 ARR** ([来源](https://x.com/hehe6z/status/1980303682932744439))。
   - 用户分享了即时语音克隆演示，并询问实时延迟（约 **500ms**），而创始人承认了目前的局限性，并承诺下一步将提供更广泛的语言支持和对话模型。
- **LangChain 获 1.25 亿美元融资**：**LangChainAI** 完成了 **1.25 亿美元**的 B 轮融资，从 OSS 入门套件扩展到提供 **LangChain**（Agent 开发）、**LangGraph**（生产编排）和 **LangSmith**（可观测性/工作台） ([来源](https://x.com/sonyatweetybird/status/1980683121399058626))。
   - 目前用户包括 **Uber**、**Klarna** 和 **LinkedIn**。


  

---

### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1429932051137564864)** (13 条消息🔥): 

> `tinygrad, NVIDIA eGPU, Apple Silicon Macs, RTX 3090, second-hand` 


- **Tinygrad 团队在 Apple Silicon Macs 上通过 USB4 启用 NVIDIA eGPU**：tiny corp 团队宣布对其纯 Python 驱动程序进行早期公开测试，该驱动程序允许 **30/40/50 系列 NVIDIA GPU**（以及 **AMD RDNA2-4**）在 **Apple-Silicon MacBooks** 上通过任何 **USB4 eGPU 扩展坞**运行；用户必须禁用 SIP 并安装其驱动程序 + NVK 编译器；参见[此处公告](https://xcancel.com/__tinygrad__/status/1980082660920918045)。
- **Tinygrad：带宽详情**：Tinygrad 的带宽约为 **2.5 GB/s** 输出和 **3.3 GB/s** 输入——虽然比 **PCIe** 慢，但一旦权重加载完成就足够了。
   - 可以通过 tinygrad 的 PyTorch 前端或未来的 CUDA 层进行 **PyTorch 访问**；**10 和 20 系列**可能通过少量补丁即可工作。 
- **针对 AI 工作负载的二手 RTX 3090 购买/测试指南**：Taha 分享了购买二手 **RTX 3090** 的经验教训：亲自会见卖家检查显卡，携带便携式 eGPU 测试平台，使用 nvidia-smi 验证识别情况，运行 **memtest_vulkan** 进行 **VRAM 完整性**测试，可选运行 gpu-burn 进行计算压力测试，加载大型模型并监控温度 **<100 °C**；参见[此处指南](https://xcancel.com/taha_yssne/status/1960418430655586677)。
- **Framework 13 Ryzen 笔记本电脑 + NixOS 作为测试平台**：对话透露测试平台是运行在 **PRIME offload 模式**下 **NixOS** 上的 **Framework 13 Ryzen 笔记本电脑**。
   - 一位用户建议在他们的平台上尝试 tinygrad，因为 *由于我使用的是 Linux，我的设备开箱即用*。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1430044403958808709)** (17 条消息🔥): 

> `Fish Audio S1 TTS launch, Sesame iOS TestFlight for conversational agents, Sequoia backs Sesame` 


- **Fish Audio S1 TTS 诞生**：Helena 庆祝了 **Fish Audio S1** 的正式发布，它被誉为表现力最强的 **TTS 模型**，比 **ElevenLabs 便宜 6 倍**，拥有 **5M ARR** 和 **20K 活跃开发者**。
   - 用户称赞了**语音克隆质量**，并询问了延迟、语言支持、iOS 应用和音素控制等问题。
- **Sesame 为 Maya 和 Miles 开启 iOS TestFlight**：在研究预览版吸引了 **1M+ 用户**后，Sesame 现在正为其超写实语音助手 **Maya 和 Miles** 开启 iOS TestFlight Beta 测试。
   - 联合创始人 Brendan Iribe 补充说，Beta 版增加了搜索和文本功能，[Sequoia Capital 重点介绍了](https://xcancel.com/sequoia/status/1980680087738675329) 这一合作伙伴关系。
- **Sequoia 播种 Sesame 的语音优先愿景**：Sequoia Capital 宣布与 Sesame 团队合作，开启*语音作为下一个伟大界面变革*的时代，目标是将计算机从工具进化为对话式的*思想伙伴（thought partners）*。
   - Sesame 正在推出闭测版 iOS 应用（在 [sesame.com/beta](https://sesame.com/beta) 注册），其特色是极具表现力的 AI Agent **Maya & Miles**。
- **Sesame 获得由 Sequoia & Spark 领投的 2.5 亿美元融资**：Sesame 在发布 Beta 版的同时，宣布完成了由 Sequoia & Spark 领投的 **$250M B 轮融资**。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1429976614413795390)** (8 条消息🔥): 

> `AMD web3 cloud, grouped gemms, FlashInfer-Bench 自我改进系统, NCU 包装脚本, PyTorch Conference AI Infra 关于 GPU kernel 的小组讨论` 


- **AMD 通过云解决方案进军 web3**：一位成员提到观看了 **AMD 活动**的演讲，并注意到他们对 **web3** “云”层面的关注。
   - 他们俏皮地添加了一个 *smileforme* 表情符号，暗示对这一概念持怀疑或调侃态度。
- **FlashInfer-Bench 旨在通过 AI 实现自我改进系统**：CMU Catalyst 推出了 [**FlashInfer-Bench**](https://flashinfer.ai/2025/10/21/flashinfer-bench.html)，这是一个通过 Agent 创建自我改进 AI 系统的流水线，其特点是为 **LLM** 推理 kernel 提供了标准化签名，并集成了 **FlashInfer**、**SGLang** 和 **vLLM**。
   - 该项目包括一篇 [博客文章](https://flashinfer.ai/2025/10/21/flashinfer-bench.html)、[排行榜](https://bench.flashinfer.ai/) 和 [GitHub 仓库](https://github.com/flashinfer-ai/flashinfer-bench)，以促进社区开发和基准测试。
- **寻求用于细粒度指标分析的 NCU 包装脚本**：一位成员询问是否有包含 **NCU 包装脚本**的 **GitHub** 仓库，这些脚本允许通过 `--metrics` 选项传递指标列表进行分析。
   - 另一位用户建议利用 [NVIDIA 定制指南](https://docs.nvidia.com/nsight-compute/CustomizationGuide/index.html#section-files) 来创建带有定制指标的自定义部分或集合。
- **AI Infra 小组讨论权衡 PTX/汇编对 Kernel 性能的影响**：[PyTorch Conference AI Infra 关于 GPU kernel 的小组讨论](https://aiinfrasummit2025.sched.com/event/28FoW/panel-discussion-the-ai-kernel-revolution-rethinking-execution-from-the-ground-up-robert-lange-sakanaai-simran-arora-stanford-university-nathan-lambert-allen-institute-moderated-by-mark-saroufim-gpu-mode) 的与会者指出，在代码的关键部分使用 **PTX/汇编**（或其上的抽象）以实现极致的 kernel 性能已达成共识。
   - 小组建议避免完全依赖编译器。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1430083659406381078)** (12 条消息🔥): 

> `Ampere GPU 中的双缓冲, Gluon 频道, Microsoft 的 Triton 会议, CuPy 与 PyTorch GPU 指针性能对比, CuPy 和 PyTorch 的 DLPack 转换` 


- **Triton 会议与会者交流**：多位成员提到参加了在 Mountain View 举办的 **Microsoft Triton 会议**，并分享了一个 [YouTube 链接](https://www.youtube.com/live/s30WoZ7lx3w?si=O6aQMCVjKFs2F4qa) 用于在线观看会议，以及一个指向 [Triton-openai 直播](https://www.youtube.com/@Triton-openai/streams) 的链接。
- **CuPy 与 PyTorch 指针性能对决**：一位成员比较了使用 **CuPy** 和 **PyTorch** GPU 指针的简单 **MatMul Kernel** 的性能，发现存在显著的性能差异。
   - 他们观察到，即使使用 **DLPack** 在 **CuPy** 数组和 **PyTorch** 张量之间进行转换，仍然存在*巨大的性能差距*，并询问是否存在造成这种差异的内在原因，同时分享了一张性能差异的 [截图](https://cdn.discordapp.com/attachments/1189607595451895918/1430323860464734360/Screenshot_from_2025-10-21_17-32-52.png?ex=68f95c66&is=68f80ae6&hm=88739dd024314bc593c58497cfddaf684e79a0ce7bfdaef32ec3a6d08812df9a&)。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1430180120974462998)** (2 条消息): 

> `WGMMA 屏障, WGMMA 序列化, PTXAS 编译器选项` 


- **WGMMA 调用被编译器屏障中断**：一位用户询问为什么编译器在 **WGMMA** 调用之间插入屏障，怀疑是否是因为所有调用都使用了相同的累加器。
   - 另一位用户建议，当编译器由于各种原因将 **WGMMA** 指令序列化时，就会发生这种情况。
- **WGMMA 序列化警告缺失**：一位用户建议检查有关 **WGMMA** 指令序列化的编译器警告，这可能会提供调试提示。
   - 该用户指出，这些警告可能仅在编译时使用 `--ptxas-options=-v` 选项才会出现。


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1429941950919741624)** (7 messages): 

> `TorchTitan pretraining slowdown, H200x Bare Metal Instance, ProcessGroupNCCL stream usage, CUDAStreamGuard, NCCL kernels` 


- ****TorchTitan 训练出现迭代缓慢****：一位用户报告在单个 **H200x** 裸金属实例上进行 **TorchTitan** 预训练时，频繁遇到迭代缓慢的问题。根据 **nsys trace** 分析，问题定位在活动线程/进程从 CPU 中被取消调度（descheduled）了几秒钟。
   - 尽管确保了没有 CPU 超量使用（oversubscription）、温度适宜且没有功率限制问题，用户仍怀疑是 **OS/kernel setting** 干扰了进程调度。
- ****PG NCCL 默认使用内部流****：当设置了 `CUDAStreamGuard` 并通过 `ProcessGroupNCCL` 调用 NCCL 操作时，**NCCL kernels** 会在 PG-NCCL 的内部流上运行，通常每个设备使用一个高优先级流，并使用张量生命周期流（tensor lifetime stream）。
   - [相关代码](https://github.com/pytorch/pytorch/blob/03f3f7899cbe4276d02379575c74f439b47668ce/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L3132) 展示了在张量生命周期流上的流同步。
- ****'Wait()' 主要在当前流上调用事件同步****：调用 `wait()` 主要是触发当前流上的事件同步（event sync），在不阻塞 CPU 的情况下建立对当前流的依赖，从而确保输出张量的行为符合预期。
   - `SynchronizeStream` 函数[等待之前的 CUDA 事件](https://github.com/pytorch/pytorch/blob/03f3f7899cbe4276d02379575c74f439b47668ce/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L780)，而无需显式的 `cudaStreamSynchronize` 或 `deviceSynchronize`。
- ****NCCL 流对 'CUDAStreamGuard' 的依赖****：设置 `CUDAStreamGuard` 决定了 **NCCL stream** 等待哪个流，从而建立入向依赖，这在 [PyTorch 源代码](https://github.com/pytorch/pytorch/blob/03f3f7899cbe4276d02379575c74f439b47668ce/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L803) 中有所体现。
   - *`wait` 不会阻塞 CPU，所有事件都标记了 cudaEvent，这不需要显式的 cudaStreamSynchronize 或 deviceSynchronize（这对重叠（overlap）不利，你肯定不希望 CPU 被阻塞，它应该在通信发生时继续在另一个计算流上启动内核）*


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1430206251333320744)** (3 messages): 

> `SLNG.AI, Speech Model Performance Engineer, vLLM, sglang, Susquehanna International Group` 


- ****SLNG.AI** 正在招聘语音模型性能工程师**：SLNG.AI 正在构建实时语音 AI 的骨干网络，目前正在寻找具有深厚软件工程背景的 **Speech Model Performance Engineer**，更多详情见[此处](https://isla-house.notion.site/Build-the-Backbone-of-Real-Time-Voice-AI-Join-SLNG-as-Founding-Speech-Model-Performance-Engineer-2642fa00c69d8072bf2fd9047f1b0b68)。
- **寻找具有 **vLLM** 和 **sglang** 经验的推理性能专家**：一位成员正在寻找 **inference performance specialist**，重点关注 **vLLM** 和 **sglang**，以在生产环境中创造独特的超额收益（alpha），感兴趣请私聊。
- ****Susquehanna International Group** 正在招聘**：量化交易公司 **Susquehanna International Group (SIG)** 正在招聘多个职位，详情见[此处](https://sig.com/careers/quant/)。
   - 感兴趣的成员可以私聊，或在 PyTorch 大会上预约面谈，预约链接见[此处](https://calendly.com/jacob-baumbach-sig/pytorch-2025)。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1430145468704358430)** (2 messages): 

> `CUDA learning, HPC learning, 5090 vs cloud GPU, Cloud GPU rental` 


- **5090 还是云端：通往 CUDA 高手之路？**：一位成员正在考虑是购买 **5090 GPU** 还是在云端租赁更便宜的选项来学习 **CUDA/HPC**，目标是最终成为专家。
   - 他们还在质疑，相比于租赁 **cloud GPU**，需要投入多大的精力才能充分发挥 **5090** 的价值。
- **CUDA 开发选本地性能还是云端灵活性？**：有人正在权衡：是购买 **5090** 获取本地算力，还是利用更便宜的云端 GPU 来掌握 **CUDA/HPC**。
   - 核心问题是：与仅仅租赁云端时长相比，你需要钻研多深才能真正压榨出 **5090** 的性能？


  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1429915296277205264)** (2 messages): 

> `sglang, ModuleFqnToConfig, torchao_utils.py` 


- **SGLang 中的 TorchAO 重构？**：一位成员询问 **sglang** 的 [这部分](https://github.com/sgl-project/sglang/blob/184a4df697ed75805ac10146dd93e75f1fc609a7/python/sglang/srt/layers/torchao_utils.py#L42) 是否仍在使用。
   - 它目前正在被使用，但团队正准备弃用这种方式，理想情况下将重构为使用 **ModuleFqnToConfig**，更多详情请见 [pytorch/ao#3083](https://github.com/pytorch/ao/pull/3083)。
- **SGLang v2 中的 TorchAO 重构？**：一位成员询问 **sglang** 的 [这部分](https://github.com/sgl-project/sglang/blob/184a4df697ed75805ac10146dd93e75f1fc609a7/python/sglang/srt/layers/torchao_utils.py#L42) 是否仍在使用。
   - 它目前正在被使用，但团队正准备弃用这种方式，理想情况下将重构为使用 **ModuleFqnToConfig**，更多详情请见 [pytorch/ao#3083](https://github.com/pytorch/ao/pull/3083)。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 messages): 

erichallahan: legendary
  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1430015202664906853)** (2 messages): 

> `OC meetup` 


- **OC 成员确认所在地**：一位成员表示他们也位于 **Orange County (OC)**。
   - 这一确认表明在 **OC** 地区进行线下见面或合作的可能性。
- **另一位成员确认 OC 所在地**：另一位成员也加入进来，确认他们也在 **OC**。
   - 这进一步增强了为 **Orange County** 地区的成员组织线下见面会的可能性。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1430121696588861510)** (3 messages): 

> `GPT-OSS-20B Architecture, DeepSeek-OCR on L4/T4 GPU` 


- **从零开始编写 GPT-OSS-20B**：一位成员在 PyTorch 中从零实现了 OpenAI 的 **GPT-OSS-20B** 架构，运行在单张 **A100 SXM (80GB)** 上。
   - 该实现包含 **RoPE with YaRN + NTK-by-parts**、**RMSNorm**、**SwiGLU**、**MoE**、**GQA**、learned sinks、banded attention 和 KV caching 等组件，[详细文档可在 GitHub 上获取](https://lnkd.in/eTTrZBeS)。
- **L4/T4 上的 DeepSeek-OCR**：一位成员分享了在显存 >16 GB 的 **L4/T4 GPU** 上运行 **DeepSeek-OCR** 的资源，可在 [此 GitHub 仓库](https://github.com/dwani-ai/llm-recipes/tree/main/tutorials/deepseek-ocr) 获取。


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1430020472342319205)** (3 messages): 

> `LLM Kernel Generation, LLM Bottleneck Identification, Profiler vs LLM` 


- **LLM Kernel 生成 vs 瓶颈识别**：一位成员提出了一个问题：一个能生成 **kernels** 的 **LLM** 和一个能在运行时识别 **瓶颈** 的 **LLM**，哪一个更有用。
- **使用 LLM 从 Profiler 日志中获取可操作的洞察**：一位成员建议，**LLM** 的效用在于将通常令人难以处理的 **profiler logs** 和 **metrics** 转化为可操作的洞察。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1430211993834491916)** (1 messages): 

> `Leaderboard sort_v2, L4 performance, B200 performance, H100 performance, A100 performance` 


- **排序算法霸榜**：由 <@1416432485591421070> 提交的作品在多个硬件配置下的 `sort_v2` 排行榜上获得了 **第一名**。
   - 获胜时间分别为：**L4 上 52.6 ms**，**B200 上 8.68 ms**，**H100 上 6.58 ms**，以及 **A100 上 16.4 ms**。
- **排序算法在不同硬件上占据统治地位**：获胜的 `sort_v2` 实现展示了跨不同 GPU 架构的性能。
   - 令人印象深刻的表现表明，该程序针对 **L4**、**B200**、**H100** 和 **A100** 的不同计算能力进行了优化。


  

---

### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1430103739691434026)** (25 messages🔥): 

> `MultiAgent Factorio AI Modification, Inspect Framework Evaluation Logic, GymAgent Implementation, MCP Server and Claude Code Integration, AI VTuber Project` 


- **Factorio AI 转型：从 MultiAgent 到单人表演**：一位成员询问将 **MultiAgent Factorio AI** 修改为 **single-agent** 系统的难度，旨在为另一个模型提供输出，以解释其在 **AI VTuber 项目**中的行为。
   - 建议涉及修改 Agent 实现，以便实时将最新步骤或完整历史记录写入另一个模型，将游戏过程转化为解说。
- **Inspect 框架增强评估能力**：在使用 **Inspect framework** 改进评估逻辑方面取得了进展，允许在任务和模型的 `eval-set` 上执行并收集评分。
   - 命令 `fle inspect-eval --eval-set --max-connections 16 --max-tasks 16 --model openrouter/openai/gpt-5-mini --view --view-port 8090` 允许在 16 个 FLE 服务器上进行并行模拟，并建议将结果存储在 S3 中以便共享访问。
- **GymAgent 架构详解**：推荐将 `GymAgent` 实现作为起点，并提供了一个示例，将生成的代码传递给轻量级 **LLM** 以总结为解说。
   - **GymAgent** 作为一个 **Action->Response** Agent 运行，在每一轮观察环境并在编写代码前进行自然语言推理。
- **MCP Server 接入 AI VTuber**：提议将 **MCP server** 与 **Claude Code** 集成，利用 **Claude Code** 对 hook 的支持来处理总结，同时 **MCP server** 也可以接入 **Docker** 并与 **n8n** 配合使用，为 **AI VTuber** 管理 **LLM** 功能。
   - **MCP server** 因其活跃的支持和独立管理执行的能力而受到青睐，允许像 **n8n** 这样的外部工具管理 AI 模型调用。
- **AI Discord 机器人开启隐私探索**：一位成员提到正在开发一个 **AI Discord bot**，该机器人具有以隐私为中心的全局记忆和基于 **Plutchik wheel**（普拉切克情感轮）的全局情感状态引擎等功能。
   - 他们开玩笑地提到自己偏好承担“有趣”的项目。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1429987537853743154)** (8 messages🔥): 

> `PTX Compiler Tool, CuTe Kernels for PyTorch, CUTLASS example, Thread-Value Layouts` 


- **通过 PTX 注入提升 Semiring 速度**：一位用户创建了一个工具，从带有注释的 **CUDA CuTe kernels** 生成 **PTX kernels**，与直接从 CUDA 编译相比，实现了 **26倍的加速**。
   - 在所有核心上批量编译随机 PTX kernels 的示例可在 [MetaMachines/mm-ptx](https://github.com/MetaMachines/mm-ptx/blob/master/examples/stack_ptx_inject/README.md#02_bulk_rand_gemm) 找到。
- **CuTe Kernels 增强 PyTorch 张量**：一位用户介绍了 [MetaMachines/mm-kermac-py](https://github.com/MetaMachines/mm-kermac-py)，这是一个 Python 示例，将 **CuTe kernels** 暴露为 PyTorch 的任意 **semiring tensor routines**。
   - 另一位用户警告说，这种方法可能没有得到官方支持，调试或修复性能问题可能无法获得官方支持。
- **CUTLASS 代码构建示例**：一位用户发布了 [leimao/CUTLASS-Examples](https://github.com/leimao/CUTLASS-Examples)，另一位用户表示 *这是关于如何使用 cmake 构建简单 cutlass 代码的优秀入门示例*。
   - 一位用户询问 *Value 0 被多个线程复制了？？* 并附上了代码图片，另一位解释说 *这实际上展示了两个逆 Thread-Value layouts；它们将数据坐标 (0..17 x 0..7) 映射到 (Thread, Value) 坐标。T32V0 意味着从线程 32 的视角来看第 0 个数据项。*


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1430289283943501865)** (12 messages🔥): 

> `Mojo language, Modular, GPU Algorithms, Apple Silicon limitations, DGX` 


- **Mojo 和 Modular 太疯狂了！**：一位成员对 **Modular** 和 **Mojo** 语言的目标表示兴奋，称 *如果他们成功了，那将是惊人的*。
   - 该成员花了 **2-3 小时** 在一台 **Apple Silicon** 机器上完成了前 **8 个问题**。
- **GPU 算法问题深入探讨**：一位成员希望接下来的几个关于 **GPU algorithms** 的问题能更深入一些，基本上每次都完成一些基础的 kernel。
   - 另一位成员提到第 **25-34** 题看起来非常酷，但他们无法在自己的电脑上运行，开玩笑地建议他们需要一台 **DGX**。


  

---

### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1430018815575457833)** (5 messages): 

> `Triton 中的 Iris GPU Native API、Gluon 后端、Triton 中的 RDMA 实现、多平面以太网拓扑、调试多节点训练循环` 


- **Iris GPU-Native API 登陆 Triton**：Iris 的创建者宣布发布该项目，目标是设计在 **Triton kernels** 内部感觉自然的 **GPU-native API**，并直接在 Triton 中实现所有内容，以实现全编译器可见性和优化。更多信息可以在 [Gluon 后端文档](https://rocm.github.io/iris/reference/gluon/overview.html)中找到。
- **Iris 即将支持 RDMA**：**RDMA** 支持即将推出，目前采用代理线程实现，未来将支持 IBGDA。所有设备端代码都将保留在 Triton 中，且 RMA 和 RDMA 之间的 API 将保持一致。
- **寻求多平面以太网拓扑信息**：一位用户正在寻找关于**多平面以太网拓扑 (multi-planar ethernet topologies)** 的资源，特别是实现细节，如如何启用数据包喷射 (packet spraying)、平面内故障监控以及同时使用的宿主机端设置。
   - 他们正在寻找理论讨论之外的实践指南来实施它。
- **Megatron 训练循环随机冻结**：一位用户在使用 **Megatron** 进行**多节点训练循环**时遇到随机冻结，迭代时间偶尔会从正常的 1s 变为 200s。
   - 他们不确定从哪里开始调试，正在考虑添加 `torch dist barrier()` 并寻求关于其放置位置的建议以修复此问题。


  

---


### **GPU MODE ▷ #[irl-accel-hackathon](https://discord.com/channels/1189498204333543425/1416087968933740688/1430215731491901501)** (6 messages): 

> `OpenCL 中的 DFS、分布式训练、Kernel 生成、合成数据、流水线并行` 


- **计划在 OpenCL 中实现 DFS**：一名成员计划在 **OpenCL** 中实现 **DFS**，并将在频道中发布更新，同时也在寻找团队。
   - 他们还对**分布式训练**、**Kernel 生成**和**合成数据**感兴趣。
- **组队进行流水线并行开发**：一名成员向另一名正在寻找团队的成员提议，使用**合成训练数据**来实现**流水线并行 (pipeline parallelism)**。
   - 目标是利用黑客松进行协作学习和项目开发。
- **Burn Baby Burn：移植 Qwen 3**：一名成员将参加 IRL 黑客松，并希望将 **Qwen 3** 移植到 [Burn](https://burn.dev/)，并将 **0.6B 变体**编译成单个 [mega kernel](https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17)。
   - 他们希望通过黑客松结识更高级的开发者，学习 **GPU 编程**，并评估 **Burn** 在严肃工作中的可行性。
- **Rust 开发者寻求 Kernel 协作**：一名精通 **Rust** 但缺乏 **Kernel** 经验的成员正在寻找黑客松团队，以从事 **IO/通信相关**的任务。
   - 他们对涉及 **KV/权重传输**或**基于磁盘的 KV cache** 的项目感兴趣。


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1429950858895949954)** (4 messages): 

> `Helion 0.2 公测版发布、Triton 编译/MLIR 错误、Helion 作为 Triton 编译器 Fuzzer` 


- **Helion 0.2 进入公测阶段**：**Helion 0.2** 的初始版本现已在 [pypi](https://pypi.org/project/helion/0.2.0/) 上作为公测版发布。
   - Helion 是一种与 Triton 编译器交互的*瓦片抽象 (tile abstraction)*。
- **MLIR 错误困扰 Helion 优化**：在 **Helion 优化 Pass** 期间，特定配置有时会出现 Triton 编译和 MLIR 错误。
   - 断言失败源自 `/project/lib/Dialect/TritonGPU/Transforms/OptimizeThreadLocality.cpp` 中的 TritonGPUOptimizeThreadLocalityPass。
- **Helion：Triton 编译器最伟大的 Fuzzer**：据成员称，**Helion** 就像是 Triton 编译器的一个出色的模糊测试器 (fuzzer)，频繁地暴露 Bug。
   - 自动调优器 (autotuner) 被指示在发生这些错误时**跳过或忽略此类配置**，因为这被认为是正常现象。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1429910042378895482)** (72 messages🔥🔥): 

> `ChatGPT 模拟面试, IntelliCode 上下文, GPT-5 编程 UX, ML 论文技巧, RL 对 LLM 的影响` 


- **ChatGPT 接受模拟面试测试**: 一位用户尝试使用 **ChatGPT** 进行模拟面试，并从专家的角度质疑其回答的准确性，分享了 [对话链接](https://chatgpt.com/share/68f68935-3148-8005-907f-86ec2ed6e93c)。
- **IntelliCode 的上下文感知补全令人印象深刻**: 一位成员对 Visual Studio 中的 **Microsoft IntelliCode** 印象深刻，这是一款 AI 驱动的代码补全工具，它通过利用大量上下文（如项目中的所有类、打开的文件以及光标前后的代码行）来正确预测整个方法体。
   - 该成员觉得当它运行良好时，*简直就像能读懂你的心思一样*。
- **GPT-5 编程 UX 评价不佳**: 成员们讨论了编程 UX，其中一人相比 **GPT-5** 更倾向于使用 **gpt-5-codex** 和 **sonnet-4.5**。
   - 一位成员抱怨说 *无论他们在做什么，透明度都极低*，这对于“氛围编程 (vibe coding)”可能没问题，但对于关注实际实现的人来说则不然。
- **独立 ML 论文作者寻求写作建议**: 一位成员请求在独自奋战时撰写高质量 **NeurIPS** 论文的技巧，并链接了一份关于 [论文格式的 Google 文档](https://docs.google.com/document/d/16R1E2ExKUCP5SlXWHr-KzbVDx9DBUclra-EbU8IB-iE/edit?tab=t.0#heading=h.16t67gkeu9dx) 和一段关于 [科学写作的 YouTube 视频](https://www.youtube.com/watch?v=jLPCdDp_LE0)。
   - 作者的目标受众是 ML 领域之外的广泛群体，发现 Loss 和 **FID** 是不可靠的指标，需要像这篇 [论文](https://arxiv.org/abs/2310.11232) 中提到的那样使用新的采样方法。
- **RL 影响 LLM 能力**: 一位成员认为 **RL** 正在对 **LLM** 产生负面影响，特别是 **OpenAI** 模型如何在低抽象层级上重复无关信息，这可能会降低答案的多样性。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1429931689475444807)** (22 messages🔥): 

> `VR 邀请, Transformer Circuits, Deepseek 的发现, 强化学习` 


- **VR 爱好者在 'Effective Autism' 服务器连接**: 一位成员邀请喜欢 **VR** 的人加入他们的 *'effective autism'* 服务器，扩大志同道合的爱好者社区。
   - 另一位成员表示欢迎，并暗示了潜在的共同兴趣和讨论。
- **Transformer Circuits 帖子推迟**: 由于工作繁忙，一位成员将阅读新的 [Transformer Circuits 帖子](https://transformer-circuits.pub/2025/linebreaks/index.html) 推迟到了第二天。
   - 该帖子讨论了 **Transformer Circuits** 的细节。
- **Karpathy 的推文引发对论文的关注**: 一位成员分享了 [Karpathy 的推文](https://x.com/karpathy/status/1980397031542989305?t=RYS1muyomGCPvv6bhISgRg&s=19)，引发了关于某篇论文重要性的讨论。
   - 有人开玩笑说，任何由 **Karpathy** 推广的论文都会被自动认为是最好的，而另一位成员则为该论文辩护，强调了其 **框架意义 (framework implications)**。
- **Deepseek 的发现引发对西方 OS 实验室的羡慕**: 一位成员表示，他们希望西方能有更好的 **OS (开源) 实验室**，因为 **Deepseek** 总是凭空出现并带来这些发现。
   - 他们指出，与开源的 **数据收集**、**方法** 和 **训练细节** 相比，开源权重只是价值的一小部分。
- **偏好模型使用强化学习信号进行训练**: 一位成员询问强化学习信号是用于训练偏好模型，还是仅仅为了给每次推理中锐化的目标分布的归纳偏置分布匹配提供奖励。
   - 对话提到了训练偏好模型的各种技术层面，包括 **强化学习** 和 **分布匹配**。


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1430236945195335862)** (1 messages): 

> `HumanAPI, 自动化任务` 


- **HumanAPI 为未解决的问题创建手动任务**: 一位成员正在创建 **"HumanAPI"**，如果在尝试解决问题时没有经过测试的自动化任务可用，它会创建一个任务并将其分配给人类。
   - 该项目的目的是不时审查这些人类任务，看看哪些可以被自动化。
- **审查人类任务以实现自动化**: **HumanAPI** 项目旨在通过审查最初由人类执行的任务，来识别适合自动化的任务。
   - 这种迭代过程允许系统不断改进和扩展其自动化能力。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1429961253480173599)** (6 条消息): 

> `DeepSeek OCR, Claude Code on the web, Unitree Robotics, Alibaba Qwen, ChatGPT Atlas` 


- **DeepSeek 瞄准 OCR 细分市场**：[DeepSeek-AI 在 GitHub 上发布了 DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)，成为 OCR 领域的新玩家。
   - 与此同时，[Anthropic 在 Web 端发布了 Claude Code](https://www.anthropic.com/news/claude-code-on-the-web)。
- **Unitree 准备碾压 Tesla？**：一名成员推测 **Unitree** 将主导人形机器人市场。
   - 他们补充说，由于“那位橙色人物（orange dude）”的原因，**Elon Musk** 目前可能甚至无法获得执行器所需的磁铁。
- **阿里巴巴 Qwen 展示实力**：一名成员在 X 上分享了 [阿里巴巴 Qwen](https://x.com/alibaba_qwen/status/1980665932625383868?s=46) 的链接。
   - 讨论中还提到了 [ChatGPT Atlas](https://chatgpt.com/atlas)。
- **Amazon Vibe 结束 Beta 测试**：Amazon 的 **Vibe Code IDE** 已结束仅限邀请的 Beta 测试，据称费用为 **500 credits**。
   - 与许多 AI IDE 一样，它也是一个 **VSCode fork**。
- **Kiro 代码编辑器结束候补名单**：[Kiro Code 编辑器](https://kiro.dev/blog/waitlist-is-over/) 已结束候补名单，其设计理念是“基于规范（spec based）”。
   - 该成员补充说，**Kiro** 围绕功能和实现的规范进行工作，而不仅仅是依赖 Prompt。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1430385505270497361)** (1 条消息): 

> `DSPy for voice generation, Automated judging loop, Optimization for subjective quality, Character analysis pipeline, ElevenLabs` 


- ****Voiceover Mage** 利用 DSPy 生成 AI NPC 语音**：一名成员为游戏 NPC 构建了一个 [语音生成系统](https://github.com/Gielinor-Speaks/voiceover-mage)，使用 **DSPy** 解析 wiki 内容并为 **ElevenLabs** 生成语音 Prompt。
   - 目标是利用 **DSPy** 的优化功能来改进角色分析流水线并自动化语音选择，该过程记录在一个 [开发日志风格的视频](https://youtu.be/z3DQm2PiKpo) 中。
- **主观语音质量优化即将推出**：目前该成员为每个角色手动筛选三个语音候选，但计划添加一个**自动评审循环**，利用 DSPy 学习什么才是针对不同角色原型的“优秀”语音匹配。
   - 该成员还打算收集手动选择作为训练信号来创建示例，从而针对主观质量判断进行优化。


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1430274415668494556)** (2 条消息): 

> `DSPy Usage in Research, Paper Code Availability` 


- **DSPy 亮相新 ArXiv 论文**：一篇新论文 ([https://arxiv.org/abs/2510.13907v1](https://arxiv.org/abs/2510.13907v1)) 在其研究中使用了 **DSPy**，标志着其在学术界的采用率不断增长。
- **论文代码尚未公开**：虽然论文提到了使用 **DSPy**，但相应的代码仓库尚未公开。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1429966872849416273)** (74 条消息🔥🔥): 

> `inspect_history() 的位置, DSPy 中的 Adapters, 模块级历史访问, ReAct 轨迹, Trace vs. DSPy` 


- ****关于历史记录位置的辩论****：成员们讨论了为什么 `inspect_history()` 是 `dspy` 中的一个方法而不是一个模块对象，并担心在复合模块中访问 prompt 的问题。但随后澄清了 `dspy.inspect_history()` 更多是用于全局历史记录，而单个程序也会跟踪历史记录。
   - 一位成员指出，如果设置了 `dspy.configure(track_usage=True)`，可以通过 `predictor.history` 访问，但仍有人觉得这很令人困惑。
- ****适配器难题得以解决****：讨论涵盖了在 DSPy 中使用 adapter 的方法，并提供了一个示例展示如何使用 `dspy.context` 来应用单个 adapter，用户可以通过 `dspy.configure(track_usage=True)` 跟踪使用情况。
   - 一位成员给出了一个设置示例：`with dspy.context(lm=single_call_lm, adaptor=single_adaptor):` 以进一步澄清。
- ****轨迹讨论转向****：成员们讨论了 DSPy 中的轨迹（trajectories），澄清了它们更多是一个 ReAct 概念（输入、思考、工具调用、动作等），并强调 DSPy 主要处理字符串。
   - 有人提到 Interrupt 实际上只是通过关闭流式传输连接来要求停止生成，这属于应用侧的操作。
- ****Trace 胜出？****：一位成员询问了 [Microsoft Trace](https://microsoft.github.io/Trace/) 与 DSPy 的对比，另一位成员指出 Trace 声称比 DSPy 的**准确率提高了 8%**，且看起来 Token 效率更高。
   - 一位成员表示他们会尝试一下以进行公平对比，尽管他们可能仍然觉得 DSPy 提供了更细粒度的控制。
- ****模块魔法操作****：一位成员提出了关于优化 DSPy 模块并需要特定数量答案的问题，并建议将逻辑封装在带有 assertion 的模块中。
   - 成员们提到在 LM 级别设置 `num_retries=N`，优化过程随后会调用 `self.evaluator`，但如果程序失败，它只会重试该程序，因此不会无限运行。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1429953622052962396)** (35 条消息🔥): 

> `Discord 服务器徽章, EleutherAI 股票代码?, 新成员自我介绍` 


- **Discord 服务器徽章引发辩论**：成员们讨论了添加服务器徽章（类似于角色图标）的可能性，以及服务器标签是否会过于广泛地传播服务器，从而增加 EleutherAI 员工的审核负担，并引用了[这张截图](https://cdn.discordapp.com/attachments/729741769738158194/1429978898229100605/Screenshot_2025-10-20_at_7.45.09_PM.png?ex=68f96ca1&is=68f81b21&hm=e032ead2cf427352fba72fdba46d77407a4d2bd71ddc4b60a1c2b6aa04cb8980&)。
   - 一位成员指出，“制作标签很酷，但在某种程度上会将这个服务器广播到其他任何地方，EleutherAI 员工已经有太多人需要审核了。”
- **关于 EleutherAI IPO 的玩笑**：在询问某个特定股票代码是否可用后，一位成员开玩笑地问道：“**Eleuther 的纽交所股票代码**会是什么？”
   - 另一位成员回答道：“我想你误解了非营利组织的宗旨，”暗示 EleutherAI 作为一个非营利组织，不会公开上市。
- **新面孔加入聊天**：一位计算机视觉工程师和一位在金融领域工作的数据科学家/ML 工程师向频道介绍了自己，希望能开展合作。
   - 该 ML 工程师提到了目前关于 **RL** 和 **LLM 上的共形推理（conformal inference）**的项目，邀请其他人联系并共同学习。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1429916699377008785)** (10 messages🔥): 

> `Kimi k2 attention clipping, Normuon vs Muon 优化器, 权重分布平滑度, AGI 定义基准` 


- **Kimi K2 的 Attention：裁剪引发争议**：围绕 **Kimi k2** 需要裁剪 Attention 的讨论展开，这可能与优化器行为有关，其中 **muon** 促进了更好的条件数，但导致了更尖锐的权重分布。
   - 有建议认为，如果 **normuon** 在大型测试中表现与 **muon** 一样好，那么更平滑的权重分布对于稳定性而言可能本质上更理想。
- **Normuon 的胜利：预防 Logit 爆炸**：一位成员指出，即使在基准测试中使用了 **qk-norm**（可避免 Logit 爆炸），**normuon** 仍然击败了 **muon**，这表明预防 Logit 爆炸可能无法完全解释两者的性能对等。
   - 有观点认为，没有裁剪的更新会增加权重的谱秩 (spectral rank)，直接导致 Logit 爆炸，这使得针对 **normuon** 的大规模验证变得很有趣。
- **更平滑的权重：分布辩论开始**：有人担心更平滑的权重更新分布并不一定等同于更平滑的权重分布。
   - 一位成员同意更平滑的分布可能是一顿“免费午餐”。
- **AGI 定义：基准测试来袭**：一位成员分享了 [Dan Hendrycks 的 AGI 定义基准链接](https://agidefinition.ai/paper)，并询问这些基准的测试速度会有多快。
   - 另一位成员预测多模态 (multimodality) 可能会在 **1-2 年**内实现覆盖，速度提升将来自于模型的 mini 版本。
- **持续学习：标准遭到批评**：一位成员表示，持续学习 (continual learning) 的基准标准是武断的，甚至可以说是极其愚蠢的。
   - 他们预测 **85%** 的进度在 **1-2 年**内非常有可能实现，**90%** 也有可能。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1429951644275179691)** (44 messages🔥): 

> `Cloudflare 问题, 开源旧模型, 积分支付问题, 聊天延迟, 审核投诉` 


- **Manus 用户的 Cloudflare 困扰**：一位用户报告在使用 Manus 访问大多数网站时遇到了 Cloudflare 安全问题。
   - 还有建议希望 Manus 团队考虑开源他们的一些旧模型，尽管尚不清楚是否相关。
- **支付问题困扰平台**：一位用户报告在尝试通过浏览器支付积分时遇到问题，收到乱码且无法完成交易。
   - 该用户声称这是一个已知问题并已联系支持人员，而 **lucia_ly** 询问了他们的电子邮件地址以便跟进。
- **聊天变慢令用户恼火**：一位用户报告聊天处理存在过度延迟，特别是在将长篇日语章节翻译成英语时，尽管他们平时很喜欢 Manus 的速度。
   - 该用户提到：“今天早上，我放了一个章节，AI 还在思考。发生了什么？”
- **Pro 方案积分上限困惑持续**：用户报告了关于 Pro 方案**无限积分**的矛盾信息：帮助系统和 iOS 升级页面显示是无限的，而 PC 升级页面则显示有一个很高的上限。
   - 一位剩余 **11k 积分**的用户担心耗尽后会发生什么，另一位用户建议他们应该参加“各种帮助改进 Manus 的机会，因为他们总是会为你的时间提供免费积分”。
- **向用户发布诈骗警报**：一位用户被指控为“骗子”，要求获取他人的账户登录权限以进行其“该死的法学院考试研究”。
   - 另一位用户暗示，所谓的骗子“不愿再开一个账户或每月支付 20 美元，却抱怨时间紧迫，并乞求获取你付费账户的邮箱密码，很可能是为了窃取你的个人信息和银行信息”。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1429907849013628990)** (25 messages🔥): 

> `China A.I. Competition, Decentralized A.I., Nous Research, AWS Cloud, Sora` 


- **中国 AI 斯巴达式竞争造福全球**：一位成员认为，中国在 AI 领域疯狂的斯巴达式内卷竞争对 AI 领域非常有利，因为它使先进模型的获取民主化并打破了垄断。
   - 他们还表示，开源（OS）模型开发的发展速度意味着 2026 年将为我们带来**智能程度达到 100% 且成本降低 90%** 的开源模型，从而摧毁垄断者的野心。
- **NousCon 虚拟参会受关注**：一位成员询问了今年 **NousCon** 的虚拟参会事宜，但也对错过 **egg irl** 表示遗憾。
   - 另一位成员表示，由于机票和酒店价格昂贵，花了一个多星期才找到划算的方案。
- **Nous Research 被推广为去中心化 AI**：一位成员指出 **Nous Research** 被推广为去中心化 AI（Decentralize A.I.），并希望团队能解决中心化带来的问题。
   - 另一位成员表示，他们更专注于为大众实现 AI 模型的民主化。
- **Nous 通过开源方法论成功实现去中心化**：一位成员表示，**Nous** 通过其开源方法论和基础设施实现成功实现了去中心化。
   - 他们补充说，最初是 **Psyche** 让他们了解了 Nous，并链接到了 [Nous Psyche 页面](https://nousresearch.com/nous-psyche) 和一篇关于中心化的 [Stanford 论文](https://cs.stanford.edu/~gakiwate/papers/sigcomm25-centralization.pdf)。
- **Sora AI 项目展示**：一位成员展示了使用 **Sora** 创作的视频。
   - 附件视频为 [20251022_0850_01k850gmktfant3bs18n3hbd79.mp4](https://cdn.discordapp.com/attachments/1149866623109439599/1430403237084659774/20251022_0850_01k850gmktfant3bs18n3hbd79.mp4?ex=68f9a653&is=68f854d3&hm=97c310cb6dcc58adf80207392b33e468cc966babf5a88f65261489840b5b68c3&)。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1430338693037691073)** (2 messages): 

> `Microsoft Trace` 


- **分享 Microsoft Trace 工具**：一位成员分享了 [Microsoft Trace 工具](https://microsoft.github.io/Trace/) 的链接。
   - 该成员指出，*显然它并不是全新的*。
- **Microsoft Trace：过去的回响**：[Microsoft Trace 工具](https://microsoft.github.io/Trace/) 重新浮出水面，引发了关注。
   - 鉴于当前的开发实践，其功能和能力正在被重新评估。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1429939290183438336)** (16 messages🔥): 

> `Nvidia macOS drivers, GLSL renderer, clspv bugs, Vulkan sine accuracy` 


- **Nvidia 驱动被移植到 macOS**：这群狂人通过创建 **Nvidia macOS 驱动** 完成了不可思议的任务。
- **GLSL 渲染器取得进展**：一位成员一直在编写 **GLSL 渲染器**，目前已通过大部分测试，可在 [GitHub](https://github.com/softcookiepp/tinygrad/blob/master/tinygrad/renderer/glsl.py) 上获取。
- **Vulkan 后端状态更新**：目前几乎所有测试都通过了自定义后端和 **clspv**，但必须使用 `-O0 --cl-opt-disable` 来规避大量的 **clspv 漏洞**。
- **clspv 优化问题**：该成员报告称，如果不禁用优化，*clspv 会产生更多的误编译*。
- **Vulkan 正弦函数不准确**：发布者提到 **Vulkan 的 sine 函数** 不够准确，需要自定义实现，这会影响性能。


  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1429962916987408436)** (5 messages): 

> `Gradient Accumulation, Backward Call Multiplicity, TinyJit Gradient Addition` 


- **多次调用 Backward 会累加梯度**：一位成员询问在调用 `optimizer.step` 之前多次调用 `backward` 是否只是简单地将梯度贡献相加。
   - 该成员确认梯度确实会累加。
- **TinyJit 的梯度累积**：一位成员报告在梯度累积时遇到问题，并通过设置 `reduction=sum` 并手动计算非填充（non-padding）token 来修复。
   - 他们还对每个 microbatch 执行了 `backward`，对梯度进行了除法运算，并使用了 assign。
- **对 mlperf 模型训练脚本中的数学逻辑提出质疑**：一位成员质疑了 [mlperf 模型训练脚本](https://github.com/tinygrad/tinygrad/blob/c7c59e6dd71158f50bbb9a87298b4ed1d65a6fb6/examples/mlperf/model_train.py#L1375C1-L1390C54) 中数学逻辑的正确性，特别是关于 `grad_accum` 缩放的部分。
- **TinyJit 梯度累积的修复**：一位成员报告说 **TinyJit** 中的梯度累积在几个月前是损坏的。
   - 他们通过重写梯度相加步骤以使用 assign 修复了该问题。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1430084733429350422)** (10 messages🔥): 

> `Karpathy Controversy, Kimi support` 


- **针对 Karpathy 的批评引发泡沫警报**：一位成员指出，X 上对 **Karpathy** 的嘲讽预示着美国前沿 AI 实验室可能存在估值泡沫，并引用了 [这条 X 帖子](https://x.com/nathanlands/status/1980035861019533749?s=46)。
   - 该帖子包含一张据推测是在 **嘲讽 Karpathy** 的图表，原帖作者未提供更多背景信息。
- **Kimi K-2 支持状态受到质疑**：一位成员对 **Kimi** 缺乏支持表示担忧，报告称支持团队 *零* 回应。
   - 其他成员澄清该频道并非支持服务器，并建议私信特定用户，同时也询问了问题的细节以及用于提交 Bug 报告的邮箱。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1429921878784479425)** (7 messages): 

> `Python Familiarity, matmul optimization, hardware-optimized matmuls, Mojo's Missing Features` 


- **Python 有助于发现 Mojo**：一位成员指出，熟悉 **Python** 有助于提高 **Mojo** 的可发现性。
   - 另一位成员警告说，**Mojo** 和 **Python** 之间的差异可能会导致混淆。
- **手动调优 Matmul 优于编译器优化**：一位成员询问为什么 **matmul 优化** 没有集成到编译器中。
   - 另一位成员回答说，虽然优化编译器有其用武之地，但对于 **热点路径代码（hot-path code）**，通常更倾向于人工干预以针对特定硬件进行微调，并指向了 [Mojo 开源的硬件优化 matmul](https://github.com/modular/modular/tree/main/max/kernels/src/linalg/matmul)。
- **Kernel 编写者摆脱编译器束缚**：一位成员解释说，将优化移出编译器可以让更多人（kernel 编写者）贡献增强功能，而不是仅仅依赖 **编译器工程师**。
   - 他们补充说，**编译器工程师** 应该更好地用于造福整个生态系统的任务，而不是像 *在某一维度小于 64 的情况下为 matmul 提升 1% 性能* 这种小众改进。
- **类型系统位居 Mojo 愿望清单榜首**：当被问及 **Mojo** 最重要且缺失的功能时，一位成员认为 *完善的类型系统* 是首要任务。
   - 随后他们列出了一系列其他期望的功能，包括 *完善标准库数据类型、适当的 IO、良好的异步运行时、效应系统（effect system）、静态反射、编译器插件、处理更受限目标的能力、集群计算、设备/集群建模，以及类似 Erlang OTP 的实现*。