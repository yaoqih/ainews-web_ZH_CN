---
companies:
- openai
- nvidia
- mistral-ai
- togethercompute
- deepseek-ai
- lmsys
date: '2024-07-19T00:00:39.402250Z'
description: '**GPT-4o-mini** 正式发布，其价格较 text-davinci-003 降低了 **99%**，仅为 **GPT-4o 价格的
  3.5%**，且基准测试表现达到了 Opus 级别。它支持 **16k 输出 token**，运行速度快于以往模型，并即将支持**文本、图像、视频和音频的输入与输出**。**Mistral
  Nemo** 是与**英伟达 (Nvidia)** 联合开发的 **12B 参数模型**，具备 **128k token 上下文窗口**和 FP8 检查点，拥有强劲的基准测试表现。**Together
  Lite 和 Turbo** 推出了 **Llama 3** 的 fp8/int4 量化版本，**吞吐量提升高达 4 倍**，并显著降低了成本。**DeepSeek
  V2** 现已开源。即将发布的动态包括至少 **5 款未发布的模型**，以及在 ICML 2024 前夕流出的 **Llama 4** 泄露信息。'
id: f690e30b-6ad3-498e-b669-fe87f4a9b5ff
models:
- gpt-4o-mini
- mistral-nemo
- llama-3
- llama-3-400b
- deepseek-v2
original_slug: ainews-lskjd
people:
- sam-altman
title: Mini, Nemo, Turbo, Lite - 小模型起飞 (GPT4o 版)
topics:
- model-quantization
- context-windows
- instruction-following
- model-performance
- cost-efficiency
- multimodality
- benchmarking
- open-source
- model-release
---

<!-- buttondown-editor-mode: plaintext -->**效率就是你所需要的一切。**

> 2024年7月17日至7月18日的 AI 新闻。我们为你检查了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 社区（包含 **467** 个频道和 **2324** 条消息）。预计节省阅读时间（按每分钟 200 字计算）：**279 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论了！

就像公共汽车和[创业点子/小行星撞击地球题材的电影](https://x.com/swyx/status/1813338114032808305)一样，很多时候你都在等待事情发生，而有些日子里，许多事情会在同一天扎堆出现。这种情况在每个月的月中总会以一种令人费解的、类似占星术般的规律性发生——[2月15日](https://buttondown.email/ainews/archive/ainews-sora-pushes-sota/)、[4月15日](https://buttondown.email/ainews/archive/ainews-multi-modal-multi-aspect-multi-form-factor/)、[5月13日](https://buttondown.email/ainews/archive/ainews-gpt-4o-the-new-sota-everything-frontier/)，而现在是 7月17日：

- **[GPT-4o-mini](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/)** ([HN](https://news.ycombinator.com/item?id=40997585)):
  - 定价：每百万 token (mtok) **$0.15/$0.60**（基于 3:1 的输入:输出 token 混合定价，**价格仅为 Haiku 的一半**，但拥有 [Opus 级别的基准测试表现](https://github.com/openai/simple-evals)（包括在 [BigCodeBench-Hard](https://x.com/terryyuezhuo/status/1813998867039617444) 上），且**价格仅为 GPT-4o 的 3.5%**，但在 [Lmsys 上与 GPT-4 Turbo 持平](https://x.com/lmsysorg/status/1813999088758673875)）。
  - 计算：GPT-4o-mini (3 * 0.15 + 0.6)/4 = 0.26，Claude Haiku (3 * 0.25 + 1.25)/4 = 0.5，GPT-4o (5 * 3 + 15)/4 = 7.5，GPT-4 Turbo 价格曾是 GPT-4o 的 2 倍。
    - [sama](https://x.com/sama/status/1813984927622549881) 将其宣传为相比 text-davinci-003 降价了 99%。
  - 相比 GPT-3.5，对[长上下文的利用率](https://x.com/LouisKnightWebb/status/1813996569840238794)显著提升。
    - 支持 [16k 输出 token](https://x.com/jeffintime/status/1814000186357923851)！（比 GPT-4 Turbo/4o 多 4 倍）
    - “[快了一个数量级](https://x.com/imjaredz/status/1814007428440272953)”——（[约 100tok/s](https://news.ycombinator.com/item?id=40998702)，比 Haiku 稍慢）。
  - “未来将支持文本、图像、**视频和音频的输入及输出**”。
  - 首个在全新**指令层级 (instruction hierarchy)** 框架下训练的模型（[我们的相关报道](https://buttondown.email/ainews/archive/ainews-openai-reveals-its-instruction-hierarchy/)）……但[已被越狱](https://x.com/elder_plinius/status/1814023961535295918?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)。
  - [@gdb 表示这是由于开发者的需求](https://x.com/gdb/status/1814019156561543658?s=46)。
  - [ChatGPT 语音模式 Alpha 版承诺本月推出](https://x.com/willdepue/status/1813995162814869892)。
  - [在与 Claude 3.5 Sonnet 的对比中受到批评](https://x.com/abacaj/status/1813691718522564633)。
  - 
![image.png](https://assets.buttondown.email/images/ccbe89c1-9dd6-4b3d-b106-664375f2366e.png?w=960&fit=max)
 
- **[Mistral NeMo](https://blogs.nvidia.com/blog/mistral-nvidia-ai-model/)** ([HN](https://news.ycombinator.com/item?id=40996058))：一个与 Nvidia [合作训练](https://x.com/GuillaumeLample/status/1813949898095534278)的 12B 模型（[Nemotron，我们的相关报道](https://buttondown.email/ainews/archive/ainews-to-be-named-2748/)）。Mistral NeMo 支持 128k token 的上下文窗口（[该级别中原生训练的最高水平](https://x.com/ArtificialAnlys/status/1813965193933623781)，并配有[全新的代码/多语言友好型分词器 (tokenizer)](https://x.com/mattshumer_/status/1813958229577302098)），提供 FP8 对齐的检查点，并且在所有基准测试中表现极其出色（“[以多出 4B 的参数量碾压 Llama 3 8B](https://x.com/Teknium1/status/1813971144695075255)”）。
![image.png](https://assets.buttondown.email/images/adf47da0-2528-4ed9-bb03-9def5676d153.png?w=960&fit=max)
 
- [Together Lite 和 Turbo](https://x.com/togethercompute/status/1813989061503406478)（Llama 3 的 FP8/Int4 量化版本），[吞吐量是 vLLM 的 4 倍](https://x.com/abacaj/status/1814000594899870070)。
  - Turbo (FP8) —— 主打速度：400 tok/s。
  - Lite (Int4) —— 主打成本：**$0.1/mtok**。“Llama 3 的最低成本”，“比 GPT-4o-mini 成本低 6 倍。”
  - 
![image.png](https://assets.buttondown.email/images/501ba2f2-f88a-4c20-8e24-7d3d98f136e7.png?w=960&fit=max)
 
- [DeepSeek V2 开源](https://x.com/deepseek_ai/status/1813921111694053644)（[我们的报道](https://buttondown.email/ainews/archive/ainews-deepseek-v2-beats-mixtral-8x22b/) 涵盖了该论文仅发布 API 时的内容）。
- 注意，[Lmsys 上至少还有 5 个代号未公布的模型](https://x.com/phill__1/status/1813677446362992689)。
- 以及一些[关于 Llama 4 的泄露信息](https://x.com/andrewcurran_/status/1813704834819965147?s=46)。

至于为什么这些事情都堆在一起——要么是水星逆行，要么是 ICML 下周就要召开了，许多公司都在进行展示或招聘，而且 Llama 3 400b 预计将于 23 日发布。



---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 综述

> 所有综述均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型与架构**

- **Llama 3 和 Mistral 模型**：[@main_horse](https://twitter.com/main_horse/status/1613580480761196987) 指出 Deepseek 创始人梁文锋表示他们不会走闭源路线，认为强大的技术生态系统更为重要。[@swyx](https://twitter.com/swyx/status/1613711271352754341) 提到在 /r/LocalLlamas 上，Gemma 2 已将 Llama/Mistral/Phi 挤下榜首，在过滤掉大型/闭源模型后，这与 @lmsysorg 的结果一致。
- **Anthropic 的方法**：[@abacaj](https://twitter.com/abacaj/status/1613691718522564633) 注意到，当 OpenAI 发布关于降低智能模型输出能力的论文时，Anthropic 则发布了可供使用的模型，并预计今年晚些时候会推出更大的模型。
- **Deepseek Coder V2 和 MLX LM**：[@awnihannun](https://twitter.com/awnihannun/status/1613712500787154992) 分享了最新的 MLX LM 已支持 DeepSeek Coder V2，并在 MLX Hugging Face 社区中提供了预量化模型。一个 16B 模型在 M2 Ultra 上运行速度很快。
- **Mistral AI 的 Mathstral**：[@rasbt](https://twitter.com/rasbt/status/1613664564158066872) 对 Mistral AI 发布 Mathstral 感到惊喜，并将其移植到 LitGPT，作为中小型专业化 LLM 的案例研究，初步印象良好。
- **来自 Google 的 Gemini 模型**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1613560606794084646) 分享了关于 Gemini 等多模态模型如何帮助机器人变得更有用的最新研究。
- **Yi-Large**：[@01AI_Yi](https://twitter.com/01AI_Yi/status/1613693751824646163) 指出 Yi-Large 在 #LMSYS 排行榜的总榜中继续稳居前 10 名。

**开源与闭源之争**

- **支持开源的论点**：[@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1613673154851824078) 认为，鉴于开源在过去几十年推动技术进步的成功经验，应强烈倾向于开源。它一直是可解释性对齐（alignment）进展的关键。
- **对开源的担忧**：[@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1613673163500429805) 指出，核心担忧是让恐怖分子制造生物武器，但认为相比于“朝鲜有能力杀死数十亿人”之类的风险，人们很容易对恐怖主义产生不成比例的恐惧。
- **理想情景与负责任披露**：[@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1613673172170076628) 表示，在理想情况下，开源应落后于前沿水平一两年，让世界有机会评估并应对重大风险。如果开源看起来可能追上或超越闭源模型，他会支持强制执行“负责任披露”期。

**AI Agent 与框架**

- **Rakis 数据分析**：[@hrishioa](https://twitter.com/hrishioa/status/1613620119266033982) 提供了关于 Rakis 如何处理推理请求以实现去中心化分布式推理的入门指南，使用了哈希、法定人数（quorums）和嵌入聚类（embeddings clustering）等技术。
- **多 Agent 礼宾系统**：[@llama_index](https://twitter.com/llama_index/status/1613618002405069173) 分享了一个开源仓库，展示如何构建复杂的、多 Agent 树状系统来处理客户交互，包含常规子 Agent 以及用于礼宾、编排和持续功能的元 Agent。
- **LangChain 的改进**：[@LangChainAI](https://twitter.com/LangChainAI/status/1613604203606237291) 引入了通用的聊天模型初始化器，可与任何模型对接，并在初始化或运行时设置参数。他们还增加了在 Agent 中分发自定义事件和编辑图状态的功能。
- **Guardrails Server**：[@ShreyaR](https://twitter.com/ShreyaR/status/1613607695607595013) 宣布推出 Guardrails Server 以简化 Guardrails 的云端部署，具备 OpenAI SDK 兼容性、跨语言支持，以及 Guardrails Watch 和针对开源 LLM 的 JSON 生成等增强功能。

**提示词技术与数据**

- **Prompt Report 调查**：[@labenz](https://twitter.com/labenz/status/1613672116929376765) 分享了一个 3 分钟的视频，介绍了来自 The Prompt Report 的 6 大 few-shot prompting 最佳实践建议。该报告是一份长达 76 页、涵盖了 1,500 多篇 prompting 论文的综述。
- **Evol-Instruct**：[@_philschmid](https://twitter.com/_philschmid/status/1613581638573724074) 详细介绍了来自 @Microsoft 和 @WizardLM_AI 的 Auto Evol-Instruct 如何在无需人类专家知识的情况下，自动演化合成数据以提高质量和多样性。该方法使用一个 Evol LLM 来创建指令，并使用一个 Optimizer LLM 来评判和优化该过程。
- **Llava-NeXT 的交错数据**：[@mervenoyann](https://twitter.com/mervenoyann/status/1613560292397203630) 分享称，在交错的图像、视频和 3D 数据上训练新的视觉语言模型 Llava-NeXT-Interleave，可以提升所有基准测试的结果并实现任务迁移。

**梗与幽默**

- [@swyx](https://twitter.com/swyx/status/1613624523872231639) 针对许多人试图出售 AI 铲子而不是去淘金的现象，开玩笑说：“他们说，在淘金热中要制造镐和铲子。”
- [@vikhyatk](https://twitter.com/vikhyatk/status/1613628261538447845) 讽刺道：“试图卖铲子的人太多，真正想去淘金的人不够。”
- [@karpathy](https://twitter.com/karpathy/status/1613710985276072379) 惊讶地发现 FFmpeg 不仅仅是一个多媒体工具包，更是一场运动。
- [@francoisfleuret](https://twitter.com/francoisfleuret/status/1613682863805714598) 分享了与 GPT 之间关于解决一个涉及物体着色的谜题的幽默对话。

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1：限制 AI 模型可用性的欧盟法规**

- **[Andrej Karpathy 宣布成立新的 AI 教育公司 Eureka Labs](https://i.redd.it/kqvfvwi594dd1.jpeg)** ([评分: 239, 评论: 54](https://reddit.com//r/LocalLLaMA/comments/1e5ofwc/andrej_karpathy_is_launching_new_ai_education/))：**Andrej Karpathy** 宣布成立 **Eureka Labs**，这是一家新的 **AI 教育公司**。该公司的首款产品是 **LLM101n**，被誉为“世界上最好的 AI 课程”，课程材料可在 [GitHub](https://github.com/karpathy/LLM101n) 上获取。Eureka Labs 的官网为 [www.eurekalabs.ai](http://www.eurekalabs.ai)。

- **[受监管机构影响，即将推出的多模态 Llama 模型将不向欧盟企业开放](https://www.axios.com/2024/07/17/meta-future-multimodal-ai-models-eu)** ([评分: 341, 评论: 138](https://reddit.com//r/LocalLLaMA/comments/1e5uxnj/thanks_to_regulators_upcoming_multimodal_llama/))：由于监管挑战，**Meta 的多模态 Llama 模型**将不会向 **欧盟企业** 开放。该问题源于使用 **欧洲客户** 数据训练模型的 **GDPR 合规性**，而非即将出台的 AI Act。Meta 声称已向 **20 亿欧盟用户** 通知了用于训练的数据使用情况并提供了退出选项，但在收到监管机构极少反馈后，于 **6 月** 被要求暂停使用 **欧盟数据** 进行训练。

**主题 2：LLM 量化技术的进展**

- **新的 LLM 量化算法 EfficientQAT，使 2-bit INT Llama-2-70B 在内存占用更少的情况下性能超越 FP Llama-2-13B。** ([评分: 130, 评论: 51](https://reddit.com//r/LocalLLaMA/comments/1e5x2k4/new_llms_quantization_algorithm_efficientqat/))：**EfficientQAT** 是一种新的量化算法，成功挑战了 LLM 均匀（INT）量化的极限。该算法在单张 **A100-80GB GPU** 上仅用 **41 小时** 就生成了 **2-bit Llama-2-70B** 模型，与全精度相比，准确率下降不到 **3%**（**69.48 vs. 72.41**）。值得注意的是，这个 **INT2 量化的 70B** 模型在准确率上超过了 **Llama-2-13B** 模型（**69.48 vs. 67.81**），同时使用的内存更少（**19.2GB vs. 24.2GB**），代码已在 [GitHub](https://github.com/OpenGVLab/EfficientQAT) 上发布。

- **介绍 Spectra：三进制（Ternary）与 FP16 语言模型的全面研究** ([Score: 102, Comments: 14](https://reddit.com//r/LocalLLaMA/comments/1e61odl/introducing_spectra_a_comprehensive_study_of/)): **Spectra LLM 套件**推出了 54 个语言模型，包括 **TriLMs** (Ternary) 和 **FloatLMs** (FP16)，参数范围从 **99M 到 3.9B**，并在 **300B tokens** 上进行了训练。研究表明，**1B+ 参数的 TriLMs** 在同尺寸下表现始终优于 FloatLMs 及其量化版本，其中 **3.9B TriLM** 在常识推理和知识基准测试中的表现与 **3.9B FloatLM** 相当，尽管其位宽（bit size）比 **830M FloatLM** 还要小。然而，研究也指出 TriLMs 表现出与较大 FloatLM 相当的毒性和刻板印象，且在验证集和网络语料库上的 Perplexity 表现落后。
    - **探索 Llama.cpp 集成**：Hugging Face 上的 **TriLM 模型** 目前是未打包（unpacked）的。开发者们讨论了在 llama.cpp 中支持 **BitnetForCausalLM** 的可能性，[SpectraSuite GitHub 仓库](https://github.com/NolanoOrg/SpectraSuite?tab=readme-ov-file#how-to-compress-and-speedup)中提供了打包和加速 TriLMs 的指南。
    - **训练成本与优化**：讨论了在 **300B tokens** 上训练 **3.9B TriLM** 等模型的成本。在具有 **16GB RAM** 的 **V100 GPUs** 上训练需要水平扩展，导致与使用 **H100s** 相比通信开销更高。提到了在 Hopper/MI300Series GPUs 上使用 **FP8 ops** 的潜在收益。
    - **社区反响与未来前景**：开发者对 TriLM 的结果表示热烈欢迎，并期待更成熟的模型。提到了对扩展到 **12GB 模型** 的兴趣，并有人询问关于在 **Colab** 等平台上对这些模型进行其他语言微调（finetuning）的问题。


**主题 3. 针对特定任务的 LLMs 对比分析**



- **最佳故事写作 LLMs：SFW 与 NSFW 选项** ([Score: 61, Comments: 24](https://reddit.com//r/LocalLLaMA/comments/1e5owq4/best_story_writing_llms_sfw_and_nsfw_options/)): **最佳故事写作 LLMs：SFW 与 NSFW 对比**。该帖子对比了用于故事写作的各种 Large Language Models (LLMs)，将其分为 SFW 和 NSFW 类别。对于 SFW 内容，推荐 **Claude 3.5 Sonnet** 为最佳选择；而 **Command-R+** 被强调为 NSFW 写作的首选，作者指出它在显式和非显式内容方面表现都很好。对比涵盖了 **GPT-4.0**、**Gemini 1.5 pro**、**Wizard LM 8x22B** 和 **Midnight Miqu** 等模型的上下文处理、指令遵循和写作质量。

- **[Cake：用于移动端、桌面端和服务器的 Rust 分布式 LLM 推理框架](https://github.com/evilsocket/cake)** ([Score: 55, Comments: 16](https://reddit.com//r/LocalLLaMA/comments/1e601pj/cake_a_rust_distributed_llm_inference_for_mobile/)): **Cake** 是一个基于 **Rust** 的**分布式 LLM 推理**框架，专为**移动端、桌面端和服务器**平台设计。该项目旨在利用 Rust 的安全性和效率特性，为运行大语言模型提供**高性能、跨平台的解决方案**。虽然仍在开发中，但 Cake 有望成为在各种设备和环境中部署 LLM 的通用工具。
    - **想象末日后的 LLM 聚会**：**Homeschooled316** 构思了在荒原中将旧的 **iPhone 21s** 连接到 **Cake host** 进行分布式 LLM 推理的社区。用户询问关于婚姻和瘟疫的问题，冒着收到称赞已失效经济政策的无关回复的风险。
    - **Key_Researcher2598** 表达了对 **Cake** 的兴奋，指出 Rust 正从 Web 开发 (**WASM**) 扩展到游戏开发 (**Bevy**)，现在又扩展到机器学习领域。他们计划将 Cake 与 Python 的 **Ray Serve** 进行对比。

**主题 4. 创新的 AI 教育与开发平台**

- **[Andrej Karpathy 启动名为 Eureka Labs 的新 AI 教育公司](https://i.redd.it/kqvfvwi594dd1.jpeg)** ([Score: 239, Comments: 54](https://reddit.com//r/LocalLLaMA/comments/1e5ofwc/andrej_karpathy_is_launching_new_ai_education/)): **Andrej Karpathy** 宣布成立 **Eureka Labs**，这是一家新的 **AI 教育公司**。他们的首个产品 **LLM101n** 被誉为“世界上最好的 AI 课程”，可在其网站 [www.eurekalabs.ai](http://www.eurekalabs.ai) 上获取，课程仓库托管在 [GitHub](https://github.com/karpathy/LLM101n)。


## 所有 AI Reddit 摘要

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**主题 1. AI 在漫画与艺术创作中的应用**

- [/r/OpenAI] **[我朋友为我的食人魔专辑制作了一个 AI 生成的音乐视频，他做得棒吗？](https://v.redd.it/qkpo1tmpq3dd1)** ([Score: 286, Comments: 71](https://reddit.com//r/OpenAI/comments/1e5lw3s/my_friend_made_an_ai_generated_music_video_for_my/)): **AI 生成的音乐视频**展示了对“**食人魔专辑**”的超现实视觉诠释。视频融合了奇幻与怪诞的意象，包括类食人魔生物、神秘景观以及似乎与音乐同步的异世界场景。虽然未提及具体使用的 AI 工具，但结果展示了 AI 在制作独特且主题连贯的音乐视频方面的创意潜力。

- [/r/singularity] **[他们在搞什么名堂？](https://i.redd.it/g5a1n1l0v4dd1.jpeg)** ([Score: 328, Comments: 188](https://reddit.com//r/singularity/comments/1e5rglc/what_are_they_cooking/)): **AI 生成艺术争议**：最近在社交媒体上分享的一张图片描绘了一件 AI 生成的艺术作品，画面是一个**在厨房里做饭的扭曲人物形象**。这张图片引发了关于 AI 创作艺术的**伦理影响**和**艺术价值**的讨论，一些观众觉得它令人不安，而另一些人则欣赏其独特的审美。

- [/r/StableDiffusion] **[我，我自己，以及 AI](https://www.reddit.com/gallery/1e5zk5w)** ([Score: 276, Comments: 57](https://reddit.com//r/StableDiffusion/comments/1e5zk5w/me_myself_and_ai/)): **“我，我自己，以及 AI”**：这篇文章描述了一位艺术家整合 AI 工具进行漫画创作的工作流。该过程包括使用 **Midjourney** 进行初始角色设计和背景制作，使用 **ChatGPT** 进行故事开发和对话编写，以及使用 **Photoshop** 进行最终润色和分镜布局，从而实现了 AI 辅助与传统艺术技巧的无缝融合。
   - **艺术家的 AI 整合引发辩论**：该艺术家自 **2022 年 10 月**以来一直将 **Stable Diffusion** 纳入工作流，这引发了褒贬不一的反应。一些人欣赏这种创新方法，而另一些人则指责艺术家**“懒惰且不道德”**。艺术家计划继续创作漫画，让作品自己说话。
  - **AI 作为艺术工具**：许多评论者将 AI 辅助艺术与创意领域的其他技术进步进行了类比。一位用户将其比作从胶片摄影到数字摄影的转变，认为 **AI 辅助工作流将在未来几年成为行业标准**。
  - **工作流见解**：艺术家透露使用 **pony 模型作为基础**，并根据其旧作**微调了黑白和彩色 Loras**。他们建议在角色设计上与模型进行“协作”，以创建更直观且能与 AI 工具良好配合的设计。
  - **艺术价值辩论**：一些评论挑战了**“艺术需要付出努力”**的观念，认为最终结果才应该是关注的焦点。有人认为 AI 工具类似于专业摄影中的照片编辑，真正的创作工作发生在初始生成之后。


**主题 2. 使用 Kling AI 进行实时 AI 视频生成**

- [/r/OpenAI] **[GPT-4o in your webcam](https://v.redd.it/bt1agl71u6dd1)** ([Score: 312, Comments: 45](https://reddit.com//r/OpenAI/comments/1e60i0j/gpt4o_in_your_webcam/)): **GPT-4o in your webcam** 将 **GPT-4 Vision** 与 **webcam** 集成，实现实时交互。该设置允许用户向电脑摄像头展示物体，并获得来自 GPT-4 的即时响应，从而实现更具交互性和动态的 AI 体验。这种集成展示了 AI 实时处理和响应视觉输入的潜力，将人机交互的可能性扩展到了文本界面之外。
- [/r/StableDiffusion] **[Hiw to do this?](https://v.redd.it/ucbi4guqd1dd1)** ([Score: 648, Comments: 109](https://reddit.com//r/StableDiffusion/comments/1e5cook/hiw_to_do_this/)): 据报道，一款**中国 Android 应用**允许用户创建名人与其年轻时的自己相遇的 **AI-generated videos**。该过程被描述为非常简单，用户只需**输入两张照片**并点击一个按钮，尽管帖子中未提供该应用的具体名称或功能的详细信息。
   - **"Kling AI"** 据报道为这款中国应用提供动力，允许用户创建名人与年轻时的自己相遇的视频。许多评论者注意到了两人之间**亲密的肢体语言**，一位用户开玩笑说：*"为什么他们中有一半看起来像是 10 秒钟后就要亲热的样子。"*
  - **史泰龙的自我迷恋抢尽风头** - 几位用户强调了 **Sylvester Stallone** 与年轻时的自己的互动特别强烈，评论如 *"该死，史泰龙太饥渴了"* 以及 *"史泰龙要搞定他自己"*。这引发了一场关于跨时空与自己进行亲密行为算作同性恋还是自慰的哲学辩论。
  - **错失的机会与情感冲击** - 一些用户提出了改进建议，比如让 **Harrison Ford** 怀疑地看着年轻时的自己。一位评论者表示，这个概念让他感到意外的情感触动，特别欣赏通过名人配对传达的 *"培养对自己的善意"* 的信息。
  - **技术推测与伦理担忧** - 讨论涉及了该应用潜在的硬件需求，一位用户建议它需要 *"大约一千个 GPU"*。其他人则辩论了这种技术的含义，一些人对中国的进展表示赞赏。
- [/r/StableDiffusion] **[Really nice usage of GPU power, any idea how this is made?](https://v.redd.it/r4b1btek31dd1)** ([Score: 279, Comments: 42](https://reddit.com//r/StableDiffusion/comments/1e5bvkw/really_nice_usage_of_gpu_power_any_idea_how_this/)): **实时 AI 视频生成**展示了对 **GPU power** 令人印象深刻的利用。该视频演示了流畅、动态的内容创作，可能利用了现代 GPU 的先进 **machine learning models** 和**并行处理**能力。虽然未提供具体的实现细节，但这项技术可能结合了 **generative AI** 技术与**高性能计算**来实现实时视频合成。
   - 带有 **1-step scheduler** 的 **SDXL turbo** 可以在 **4090 GPU** 上以 **512x512 resolution** 实现实时性能。根据用户测试，即使是 **3090 GPU** 也能实时处理。
  - **技术栈分解**：评论者建议该设置可能包括 **TouchDesigner**、**Intel Realsense** 或 **Kinect**、**OpenPose** 以及输入到 **SDXL** 的 **ControlNet**。一些人推测这是一个不含 ControlNet 的简单 **img2img** 过程。
  - **精简的工作流**：该过程可能涉及拍摄演员，生成 **OpenPose skeleton**，然后根据该骨架形成图像。**StreamDiffusion** 与 TouchDesigner 的集成被提到是一个“神奇”的解决方案。
  - **致谢**：原视频归功于 Instagram 上的 **mans_o**，可在 [https://www.instagram.com/p/C9KQyeTK2oN/?img_index=1](https://www.instagram.com/p/C9KQyeTK2oN/?img_index=1) 查看。


**主题 3. OpenAI 的 Sora 视频生成模型**

- [/r/singularity] **[新 Sora 视频](https://v.redd.it/s8n3cwksq6dd1)** ([Score: 367, Comments: 128](https://reddit.com//r/singularity/comments/1e602ro/new_sora_video/)): **OpenAI 的 Sora** 发布了一段新视频，展示了其先进的 **AI 视频生成能力**。该视频展示了 Sora 创建高度详细且逼真场景的能力，包括复杂的环境、多个角色和动态动作。这次最新的演示突显了 AI 生成视频技术的飞速进步及其在各个领域的潜在应用。
   - 评论者预测 **Sora 的技术** 将在 **一年内变得非常出色**，并在 **2-5 年内与现实无异**。一些人认为它已经可以在某些应用中投入市场，而另一些人则指出了仍然存在的缺陷。
  - **Uncanny Valley 挑战**：用户注意到演示视频中在 **物理、运动和连续性方面存在不一致性**。问题包括奇怪的脚部放置、梦幻般的动作以及角色外观的快速变化。一些人认为这些问题可能比预期的更难解决。
  - **潜在应用与局限性**：讨论集中在 **AI 在 CGI 和低预算制作中的作用**。虽然目前还无法取代真人实拍，但它可能会彻底改变社交媒体内容和电影中的背景元素。然而，人们对 **OpenAI 有限的公开访问权限** 提出了担忧。
  - **飞速进步令人震惊**：许多人对 **AI 视频进化的速度** 表示惊讶，将其与早期的里程碑（如 [thispersondoesnotexist.com](https://thispersondoesnotexist.com)）进行比较。尽管仍有瑕疵，但从早期演示到 Sora 能力的飞跃被认为是卓越的。


**主题 4. AI 监管与部署挑战**

- [/r/singularity] **[Meta 将不会向欧盟提供未来的多模态 AI 模型](https://www.axios.com/2024/07/17/meta-future-multimodal-ai-models-eu)** ([Score: 341, Comments: 161](https://reddit.com//r/singularity/comments/1e5s1j9/meta_wont_bring_future_multimodal_ai_models_to_eu/)): **Meta** 宣布，由于监管的不确定性，它将不会在 **欧盟 (EU)** 发布未来的 **多模态 AI 模型**。该公司对 **EU AI Act** 表示担忧，该法案仍在最终敲定中，可能会对 AI 系统施加严格的规则。这一决定影响了 Meta 即将推出的 **LLM** 和 **生成式 AI 功能**，可能导致欧盟用户无法使用这些先进的 AI 技术。

- [/r/OpenAI] **[Sam Altman 称其价值 2700 万美元的旧金山豪宅完全是个“柠檬”](https://www.forbes.com.au/news/billionaires/sam-altman-says-27-million-mansion-is-a-lemon/)** ([Score: 244, Comments: 155](https://reddit.com//r/OpenAI/comments/1e5xft1/sam_altman_says_27_million_mansion_is_a_lemon/)): **OpenAI** 的 CEO **Sam Altman** 在最近的一次采访中表达了对他 **2700 万美元旧金山豪宅** 的沮丧，称其完全是一个“**柠檬**”（lemon，指有缺陷的产品）。尽管该房产价格高昂且位于 **Russian Hill** 的显赫位置，Altman 透露房子一直受到众多问题的困扰，包括 **游泳池**、**供暖系统** 和 **电气线路** 的问题。这种情况突显了高端房地产购买的潜在陷阱，即使是拥有巨额资源的科技行业领袖也不例外。

- [/r/singularity] **[Marc Andreessen 和 Ben Horowitz 表示，当他们与白宫官员讨论 AI 时，官员们称他们可以将任何他们认为导向不良的数学领域列为国家机密，并且“它将就此终结”](https://v.redd.it/38jaxkr912dd1)** ([Score: 353, Comments: 210](https://reddit.com//r/singularity/comments/1e5enn2/marc_andreessen_and_ben_horowitz_say_that_when/)): **Marc Andreessen** 和 **Ben Horowitz** 报告称，**白宫官员** 声称，如果他们认为某个 **数学领域** 的发展不利于 **AI 发展**，他们可以将其列为 **国家机密**。据称，官员们表示通过这种方式，他们可以有效地终结该数学领域的进展。这一启示表明，政府可能采取一种通过数学审查来控制 AI 进步的潜在策略。

---

# AI Discord 摘要

> 正如我们在前沿模型发布日所做的那样，今天的 Discord 摘要有两个版本。你正在阅读的版本是由 GPT-4o 生成频道摘要，然后将频道摘要汇总为 {4o/mini/sonnet/opus} 的“摘要之摘要”。请查看存档以获取 GPT-4o-mini 配对，以便进行逐个频道的摘要对比。

## Claude 3 Sonnet


**1. 新型 AI 模型发布与功能**

- **GPT-4o Mini：OpenAI 的高性价比利器**：OpenAI 推出了 [GPT-4o mini](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/)，这是一款比 GPT-3.5 Turbo 更便宜、更智能的模型，在 **MMLU 上得分 82%**，拥有 **128k context window**，价格为 **每百万输入 token 0.15 美元，每百万输出 token 0.60 美元**。
   - `@AndrewCurran_` 证实 GPT-4o mini 正在为免费和付费用户**取代 GPT-3.5 Turbo**，它在显著降低成本的同时提升了 GPT-3.5 的能力，尽管初期缺乏图像支持等某些功能。
- **Mistral NeMo：NVIDIA 合作释放强大性能**：Mistral AI 与 **NVIDIA** 联合发布了 [Mistral NeMo](https://mistral.ai/news/mistral-nemo/)，这是一个拥有 **128k context window** 的 **12B 模型**，在 **Apache 2.0 license** 下提供最先进的推理、世界知识和代码准确性。
   - Mistral NeMo 支持无损的 **FP8 inference**，性能超越了 **Gemma 2 9B** 和 **Llama 3 8B** 等模型，并提供预训练基座模型和指令微调（instruction-tuned）检查点。
- **DeepSeek V2 引发中国价格战**：**DeepSeek 的 V2 模型**将推理成本削减至**每百万 token 仅 1 元人民币**，凭借其革命性的 **MLA architecture** 和显著降低的内存占用，引发了中国 AI 公司之间的竞争性定价狂潮。
   - DeepSeek V2 获得了中国 **AI 界的拼多多**之称，因其削减成本的创新而受到赞誉，其极高的性价比可能会颠覆全球 AI 格局。
  


**2. 大语言模型（LLM）技术的进步**

- **Codestral Mamba 凭借线性推理脱颖而出**：新推出的 [Codestral Mamba](https://mistral.ai/news/codestral-mamba/) 由 **Albert Gu** 和 **Tri Dao** 共同开发，凭借其**线性时间推理（linear time inference）**和处理**无限长序列**的能力，承诺在代码生成能力上实现飞跃。
   - 旨在提高编程效率，Mamba 的目标是在提供无论输入长度如何都能快速响应的同时，超越现有的基于 SOTA Transformer 的模型。
- **Prover-Verifier Games 提升 LLM 的可读性**：一种名为 [Prover-Verifier Games](https://openai.com/index/prover-verifier-games-improve-legibility/) 的新技术已被证明可以提高语言模型输出的可读性和可解释性，详情见[相关论文](https://cdn.openai.com/prover-verifier-games-improve-legibility-of-llm-outputs/legibility.pdf)。
   - 通过增强 LLM 推理的可解释性，该方法旨在解决开发更透明、更值得信赖的 AI 系统所面临的关键挑战。
  


**3. 硬件优化与 AI 性能**

- **Resizable BAR 对 LLM 影响微乎其微**：讨论表明，旨在增强 GPU 性能的 **Resizable BAR** 功能对 **LLM 操作的影响可以忽略不计**，因为 LLM 更依赖于 **tensor cores 和 VRAM 带宽**。
   - 虽然有人推测模型加载和多 GPU 设置可能会从中受益，但社区共识倾向于认为 Resizable BAR 对核心 LLM 工作负载的影响极小。
- **Lubeck 凭借 LLVM 效率超越 MKL**：**Lubeck** 数值库展示了优于 **MKL (Math Kernel Library)** 的性能，这归功于其差异化的 **LLVM IR 生成**，可能得到了 **Mir 的 LLVM 加速通用数值库**的辅助。
   - 一项 [基准测试对比](http://blog.mir.dlang.io/glas/benchmark/openblas/2016/09/23/glas-gemm-benchmark.html) 强调了 Lubeck 的速度优势，引发了关于利用 LLVM 进行优化数值计算的讨论。
  


**4. AI 代码助手与集成**

- **OpenRouter 上的 Codestral 22B 请求**：一位用户请求将 **Codestral 22B** 添加到 OpenRouter，这是一个性能与最先进的**基于 Transformer 的模型**相当的开源代码模型，并分享了 [model card](https://huggingface.co/mistralai/mamba-codestral-7B-v0.1) 以供参考。
   - Codestral Mamba 在代码生成任务中展示了极具竞争力的性能，激发了人们将其集成到 OpenRouter 等流行 AI 平台中的兴趣。
- **Groq 模型主导 Function Calling 排行榜**：Groq 新的 **tool use 模型**，包括 [Llama-3-Groq-8B](https://huggingface.co/lmstudio-community/Llama-3-Groq-8B-Tool-Use-GGUF) 和 [Llama-3-Groq-70B](https://huggingface.co/lmstudio-community/Llama-3-Groq-70B-Tool-Use-GGUF)，在 **Berkeley Function Calling Leaderboard** 上分别获得了 **89.06%** 和 **90.76%** 的最高分。
   - 这些模型针对 **tool use** 和 **function calling** 能力进行了优化，展示了 Groq 在开发擅长执行复杂多步任务的 AI 助手方面的实力。

## Claude 3.5 Sonnet


**1. AI 模型发布与升级**

- **DeepSeek V2 引发价格战**：DeepSeek 发布了 **DeepSeek V2**，将推理成本大幅降低至每百万 token 1 元，引发了中国 AI 企业之间的价格战。该模型引入了全新的 **MLA architecture**，显著降低了显存占用。
   - DeepSeek V2 被称为中国 AI 界的**拼多多**，其极致的性价比和架构创新使其成为 AI 市场的一股颠覆性力量。此次发布凸显了 AI 模型开发与部署中日益激烈的竞争。
- **Mistral NeMo 12B 强力模型**：Mistral AI 与 NVIDIA 联合推出了 **Mistral NeMo**，这是一个拥有 12B 参数的模型，具备 128k token 的上下文窗口和顶尖的推理能力。该模型在 Apache 2.0 许可证下发布，提供预训练和指令微调（instruction-tuned）检查点。
   - Mistral NeMo 支持无性能损失的 **FP8 inference**，定位为 Mistral 7B 的无缝替换方案。Mistral AI 与 NVIDIA 的此次合作展示了模型架构和行业伙伴关系的快速演进。
- **OpenAI GPT-4o mini 亮相**：OpenAI 推出了 **GPT-4o mini**，号称是其最智能且最具成本效益的小型模型，在 MMLU 上得分为 82%。其价格为每百万输入 token 0.15 美元，每百万输出 token 0.60 美元，比 GPT-3.5 Turbo 便宜得多。
   - 该模型支持文本和图像输入，拥有 128k 上下文窗口，已向免费版 ChatGPT 用户及各级付费用户开放。此次发布展示了 OpenAI 致力于让先进 AI 对开发者和企业而言更易获取、更负担得起的决心。
  


**2. 开源 AI 进展**

- **LLaMA 3 的 Turbo 加速版本**：Together AI 推出了 LLaMA 3 的 **Turbo** 和 **Lite** 版本，提供更快的推理速度和更低的成本。**LLaMA-3-8B Lite** 的价格为每百万 token 0.10 美元，而 **Turbo 版本** 的生成速度高达 400 tokens/s。
   - 这些新变体旨在使 LLaMA 3 在各种应用中更易用、更高效。社区中关于可能发布的 **LLaMA 3 400B** 的传闻也甚嚣尘上，这可能会对开源 AI 格局产生重大影响。
- **DeepSeek-V2 登顶开源榜单**：DeepSeek 宣布其 **DeepSeek-V2-0628** 模型现已开源，并在 LMSYS Chatbot Arena Leaderboard 的开源类别中排名第一。该模型在包括综合性能和困难提示词（hard prompts）在内的多项基准测试中表现出色。
   - 这一发布凸显了开源模型相对于闭源模型日益增长的竞争力。它还强调了社区在创建高性能、可自由获取的 AI 模型以用于研发方面所做的努力。
  


**3. AI 安全与伦理挑战**

- **GPT-4 的过去式漏洞**：一篇[新论文](https://arxiv.org/abs/2407.11969)揭示了 GPT-4 的一个重大漏洞：通过将有害请求重新表述为过去式，在 20 次尝试下，越狱成功率从 1% 提高到了 88%。
   - 这一发现凸显了当前 AI 安全措施（如 **SFT**, **RLHF** 和 **adversarial training**）的潜在弱点。它引发了人们对对齐技术鲁棒性的担忧，并表明在 AI 开发中需要更全面的安全策略。
- **Meta 的多模态模型排除欧盟地区**：据 [Axios](https://www.axios.com/2024/07/17/meta-future-multimodal-ai-models-eu) 报道，Meta 计划在未来几个月发布多模态 **Llama model**，但由于监管的不确定性，该模型将不会在欧盟地区提供。
   - 这一决定凸显了 AI 发展与监管合规之间日益紧张的关系，尤其是在欧盟。它还引发了关于潜在技术鸿沟以及地区法规对全球 AI 进步影响的讨论。

## Claude 3 Opus


**1. Mistral NeMo 12B 模型发布**

- **Mistral 与 NVIDIA 合作推出 128k 上下文模型**：**Mistral NeMo** 是一款具有 128k token 上下文窗口的 12B 模型，由 **Mistral** 与 **NVIDIA** 合作开发，并根据 **Apache 2.0 license** 发布，详见[官方发布说明](https://mistral.ai/news/mistral-nemo/)。
   - 该模型在其尺寸类别中提供了**顶尖的推理能力、世界知识和编程准确性**，可作为 **Mistral 7B** 的直接替代方案，并支持无损性能的 **FP8 inference**。
- **Mistral NeMo 性能超越同类尺寸模型**：Mistral NeMo 的性能已与其他模型进行了对比，尽管在 Meta 报告的针对 Llama 3 8B 的 **5-shot MMLU scores** 等基准测试中存在一些差异。
   - 尽管存在这些不一致之处，根据[发布公告](https://mistral.ai/news/mistral-nemo/)，Mistral NeMo 仍被视为强有力的竞争者，在多项指标上超越了 **Gemma 2 9B** 和 **Llama 3 8B** 等模型。


**2. GPT-4o Mini 发布与越狱**

- **GPT-4o Mini：比 GPT-3.5 更聪明、更便宜**：OpenAI 发布了 **GPT-4o mini**，被誉为能力最强且最具成本效益的小型模型，在 **MMLU** 上得分为 **82%**。据 [Andrew Curran](https://x.com/andrewcurran_/status/1813942258968018954?s=46) 宣布，该模型已向免费和付费用户开放。
   - 该模型的定价为**每 M token 输入 15 美分，每 M token 输出 60 美分**，拥有 **128k 上下文窗口**。这使得它比 **GPT-3.5 Turbo** 便宜得多，但缺少完整版 GPT-4o 的某些功能（如图像支持）。
- **GPT-4o Mini 的安全机制被越狱**：据 [Elder Plinius](https://x.com/elder_plinius/status/1814023961535295918?s=46) 称，**GPT-4o mini** 中新实施的名为“指令层级”（instruction hierarchy）的安全机制已被越狱，使其能够输出受限内容。
   - 此次越狱揭示了 OpenAI 最新防御方法的漏洞，引发了人们对其实施安全措施的稳健性以及潜在滥用风险的担忧。


**3. AI 训练与部署的进展**

- **Tekken Tokenizer 效率超越 Llama 3**：**Tekken tokenizer** 模型展示了优于 **Llama 3 tokenizer** 的性能，在包括中文、韩文、阿拉伯文和源代码在内的多种语言中，压缩率提高了 30-300%。
   - 这种效率的提升使 Tekken 成为 **NLP** 任务的强力竞争者，在降低计算成本和实现更紧凑的模型表示方面具有显著优势。
- **通过新编译器在 AMD GPU 上运行 CUDA**：一款针对 **AMD GPU** 的新编译器已实现在 **RDNA 2 和 RDNA 3** 架构上的 **CUDA 支持**。正如 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1e6cxef/cuda_on_amd_rdna_3_and_rdna_2_new_release/)中所分享的，**RX 7800** 已确认可以运行。
   - 社区成员表示有兴趣在 **llama.cpp** 中测试此设置，以对比 **ROCm implementation** 的性能，并提到了 **ZLUDA** 和 **SCALE** 等工具作为在 AMD 硬件上运行 CUDA 的替代方案。

## GPT4O (gpt-4o-2024-05-13)


**1. Mistral NeMo 发布**

- **Mistral NeMo 的惊艳亮相**：**[Mistral NeMo](https://mistral.ai/news/mistral-nemo/)** 模型是由 **NVIDIA** 合作开发的 12B 参数模型，拥有 **128k token 上下文窗口**和卓越的推理能力。
   - **预训练和指令微调的 Checkpoints** 已在 Apache 2.0 许可证下发布，在推理、代码准确性和世界知识方面具有最前沿的性能。
- **Mistral NeMo 性能对比**：注意到 **Llama 3 8B** 的 **5-shot MMLU 分数**存在差异，**Mistral** 报告为 **62.3%**，而 Meta 报告为 **66.6%**，这引发了对报告性能的质疑。
   - 这一差异以及 **TriviaQA 基准测试**中潜在的问题，引发了关于这些指标可靠性的讨论。
    


**2. AI 硬件优化**

- **内核探索：应对参数限制**：AI 工程师通过优先使用指针传递大型数据结构，解决了 **CUDA 4k 内核参数大小限制**问题，在这种情况下，必须将数据从 CPU 迁移到 GPU 全局内存。
   - 对话转向了指针的复杂性，阐明了内核参数中指针对于寻址 GPU 内存的必要性，并消除了在 CUDA 内存分配中关于 ** 与 * 使用的困惑。
- **模型肌肉：探索高性能硬件的 AI 训练**：关于最佳 AI 训练配置的讨论称赞了 **A6000 GPU** 的性价比，一位用户重点介绍了一套 **64 核 Threadripper 配备双 A6000** 的配置。
   - 这套最初用于**流体动力学模拟**的配置，激发了人们对高端硬件在 AI 训练中通用性的兴趣和讨论。
    


**3. 多模态 AI 进展**

- **Meta 发布多模态 Llama 模型**：Meta 计划推出多模态 **Llama 模型**，但由于监管限制，欧盟用户被排除在外，正如 **[Axios 报告](https://www.axios.com/2024/07/17/meta-future-multimodal-ai-models-eu)** 中所强调的那样。
   - 诸如使用 **VPN** 或勾选非欧盟合规复选框之类的变通方法已经在私下流传，预示着**访问破解**行为可能会增加。
- **涡轮增压版 LLaMA 3 版本投入使用**：**Together AI** 推出的 **LLaMA-3-8B Lite** 承诺提供极具成本效益的 **每百万 token 0.10 美元**的价格，确保了经济性与速度的结合。
   - 为了增强部署格局，**LLaMA-3-8B Turbo** 的速度飙升至 **400 tokens/s**，专为追求极速效率的应用而量身定制。
    


**4. 模型训练问题**

- **模型训练困扰：乌云密布**：社区讨论了部署和训练模型的挑战，特别是在 AWS 和 Google Colab 等平台上，强调了耗时长和低资源复杂性等问题。
   - 特别提到了 Hugging Face Spaces 上的 `text2text-generation` 错误和 GPU 资源困境，贡献者们正在寻求并分享故障排除策略。
- **微调挑战与 Prompt Engineering 策略**：**Prompt 设计**以及使用正确的模板（包括 end-of-text token）对于在微调和评估期间影响模型性能至关重要。
   - 示例：[Axolotl prompters.py](https://github.com/OpenAccess-AI-Collective/axolotl/blob/3367fca73253c85e386ef69af3068d42cea09e4f/src/axolotl/prompters.py#L47)。

## GPT4OMini (gpt-4o-mini-2024-07-18)


**1. Mistral NeMo 模型发布**

- **Mistral NeMo 令人印象深刻的上下文窗口**：**[Mistral NeMo](https://mistral.ai/news/mistral-nemo/)** 是一款与 NVIDIA 合作发布的全新 12B 模型，具有卓越的 **128k token 上下文窗口**，在推理和编码任务中表现出色。
   - 该模型采用 **Apache 2.0 许可证**发布，关于其与 **Llama 3** 等其他模型对比效果的讨论已经展开。
- **社区对 Mistral NeMo 的反应**：该模型采用 **Apache 2.0 许可证**发布，关于其与 **Llama 3** 等其他模型对比效果的讨论已经展开。
   - 初步印象显示反响积极，用户渴望测试其各项能力。
    


**2. GPT-4o Mini 发布**

- **高性价比 GPT-4o Mini 发布**：**OpenAI** 推出了 **GPT-4o mini**，这款新模型的定价为 **每百万输入 token 0.15 美元** 以及 **每百万输出 token 0.60 美元**，使其比前代产品显著便宜。
   - 这一价格结构旨在让更广泛的受众能够接触到先进的 AI。
- **GPT-4o Mini 的性能预期**：尽管价格实惠，但一些用户对其与 **GPT-4** 和 **GPT-3.5 Turbo** 相比的性能表示失望，认为它并未完全达到 OpenAI 设定的高预期。
   - 社区反馈表明，虽然它具有成本效益，但在所有场景下可能无法超越现有模型。
    


**3. 深度学习硬件优化**

- **使用 A6000 GPU 进行优化**：社区讨论强调了利用 **A6000 GPU** 进行深度学习任务的优势，特别是在以合理成本进行高性能模型训练方面。
   - 用户报告了利用 A6000 的能力进行各种 AI 应用的成功配置。
- **AI 训练的用户配置**：成员们分享了他们的配置，包括从流体力学模拟转为 AI 训练的 **双 A6000** 设置。
   - 这些设置展示了 A6000 GPU 在处理复杂计算任务中的多功能性。
    


**4. RAG 实施挑战**

- **对 RAG 技术的怀疑**：几位社区成员对 **Retrieval Augmented Generation (RAG)** 的有效性表示怀疑，称如果没有广泛的微调，它往往会导致次优的结果。
   - 共识是，虽然 RAG 具有潜力，但需要大量的定制化投入。
- **社区对 RAG 的反馈**：诸如 *“如果你想要一个糟糕的结果，RAG 很简单”* 之类的评论强调了为了获得理想结果需要付出巨大努力。
   - 成员们强调了仔细实施对于释放 RAG 全部潜力的重要性。
    


**5. 多模态 AI 进展**

- **Meta 的多模态 Llama 模型计划**：根据最近的 [Axios 报告](https://www.axios.com/2024/07/17/meta-future-multimodal-ai-models-eu)，Meta 正准备推出多模态 **Llama 模型**，但由于监管挑战，计划限制在欧盟（EU）地区的访问。
   - 这一决定引发了对欧盟用户可访问性的担忧。
- **欧盟用户的绕过方案**：用户之间已经在讨论使用 **VPN** 等绕过方案来突破这些限制。
   - 这些讨论凸显了社区在应对监管障碍方面的机敏。
    

---

# PART 1: 高层级 Discord 摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **RAG 的坎坷之路：AI 爱好者就 RAG 展开交锋**：几位成员对 **RAG** 表示怀疑，指出如果不进行广泛的微调，它容易产生不理想的结果。
   - *“如果你想要一个糟糕的结果，RAG 很简单”*暗示了发挥其潜力需要付出巨大努力。
- **NeMo 的新篇章：Mistral 发布 12B 强力模型**：与 **NVIDIA** 共同发布的 **Mistral NeMo 12B 模型**拥有令人印象深刻的上下文窗口，并承诺具备领先的推理能力。
   - 根据 [Mistral News](https://mistral.ai/news/mistral-nemo/) 链接的公告，**Mistral 7B** 的采用者现在可以根据 **Apache 2.0 license** 升级到 **NeMo**。
- **推迟亮相：Unsloth Studio 的 Beta 发布会延期**：**Unsloth Studio** 宣布推迟其 Beta 测试发布，将计划调整至稍后日期。
   - 社区对质量表示支持，回复如*“慢慢来——排查故障需要多长时间就花多长时间”*。
- **为模型提供动力：探索强悍硬件的 AI 训练**：关于最佳 AI 训练配置的讨论称赞了 **A6000 GPUs** 在价格和性能方面的实力。
   - 一位用户的配置——**64-core threadripper 配备双 A6000s**——引发了关注，突显了该硬件在初始用途（**fluid dynamics simulations**）之外的多功能性。
- **STORM 的成功：塑造结构化摘要**：**STORM** 为 AI 驱动的预写作设定了新标准，其构建的详尽文章在组织结构上提升了 **25%**，覆盖范围扩大了 **10%**。
   - 该模型集成了**专家级多视角提问**，可以组装出类似于完整报告的综合性文章，详见 [GitHub - stanford-oval/storm](https://github.com/stanford-oval/storm)。



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **AI 漫画默认气泡上线！**：**AI Comic Factory** 的更新指出，它现在将包含**默认对话气泡**，该功能仍处于开发阶段。
   - 工具 [AI Comic Factory](https://huggingface.co/spaces/jbilcke-hf/ai-comic-factory) 虽然仍在完善中，但旨在通过简化对话元素来增强漫画创作。
- **Huggingchat 遇到障碍：响应速度变慢**：据报道 **Huggingchat** 的 **commandR+** 运行缓慢，导致用户感到沮丧，一些用户在处理通常只需几秒钟的任务时经历了长达 5 分钟的处理时间。
   - 社区中的评论如*“他们几乎发了三次同样的消息”*突显了影响用户体验的降速问题。
- **Florence 2 轻松去除水印**：一款使用 **Florence 2** 和 **Lama Cleaner** 构建的新水印去除工具展示了其实力，为用户提供了一个高效的移除工具，分享在 [Hugging Face Spaces](https://huggingface.co/spaces/DamarJati/Remove-watermark) 上。
   - 这款[水印去除工具](https://huggingface.co/spaces/DamarJati/Remove-watermark)旨在易于使用，加入了 **Florence 2** 的功能套件，有望解决另一个实际问题。
- **Mistral NeMo：NVIDIA 的 12B 之子**：**Mistral NeMo** 是一款 12B 模型，现在拥有高达 128k 的上下文长度，以 **Apache 2.0 license** 首次亮相，专注于高性能，如[官方公告](https://x.com/mistralai/status/1813947930455499200?s=46&t=IfJRyr-UwyoM2m-vJODIzw)所述。
   - MistralAI 与 NVIDIA 合作发布的这一新版本分享在 [Mistral AI](https://mistral.ai/news/mistral-nemo/) 上，增强了处理海量上下文长度的工具库。
- **模型训练难题：乌云密布**：社区讨论了部署和训练模型的考验，特别是在 AWS 和 Google Colab 等平台上，强调了耗时长和低资源复杂性等问题。
   - 特别提到了关于 Hugging Face Spaces 上的 `text2text-generation` 错误和 GPU 资源困境，贡献者们正在寻求并分享故障排除策略。



---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Kernel 探索：应对参数限制**：AI 工程师们通过优先使用指针来传递大型数据结构，解决了 CUDA 4k kernel 参数大小限制的问题，在这种场景下，从 CPU 迁移到 GPU 全局内存是强制性的。
   - 对话转向了指针的复杂性，阐明了在 kernel 参数中使用指针来寻址 GPU 内存的必要性，并消除了在 CUDA 内存分配中关于 `**` 与 `*` 用法的困惑。
- **Google Gemma 2 胜出，随后被超越**：Gemma 2 模型因超越了之前的领先者 Llama 2 而吸引了社区的关注，引发了对 [Google I/O](https://datta0.substack.com/i/146681354/gemma) 上 Google 发布内容的深入探讨。
   - 然而，AI 领域的持续进步使得 Gemma 2 迅速被 LlaMa 3 和 Qwen 2 等模型超越，这引发了人们对该行业高速发展的认可。
- **GPT-3 125M 模型：突破性的启动**：首个 GPT-3 模型 (125M) 训练的启动吸引了社区，其中值得注意的包括 12 小时的预期完成时间以及对最终性能指标的渴望。
   - FP8 训练设置的修改和新 Quantization 机制的集成被置于重要位置，指向了未来对模型效率的探索。
- **CUDA 难题：Shared Memory 的奥秘**：动态 Shared Memory 的效用成为热门话题，诸如 `extern __shared__ float shared_mem[];` 之类的策略被投入到技术讨论中，旨在提升 kernel 性能。
   - 向动态 Shared Memory 的转变预示着未来更简洁的 CUDA 操作以及对更密集计算过程的共同愿景。
- **Triton 编译器的编排艺术**：随着 Triton 编译器展示其通过自动微调 GPU 代码将 Python 转换为更易处理的 Triton IR 的才华，人们感到非常兴奋。
   - 社区贡献通过分享 [Triton Puzzles 的个人解决方案](https://github.com/alexzhang13/Triton-Puzzles-Solutions) 增强了这一影响，鼓励其他人尝试这些挑战。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **RTX 4060 Ti 技巧与提示**：关于 **RTX 4060 Ti** 是否足以运行 Automatic1111 的疑问得到了解答；通过使用 `--xformers --medvram-sdxl` 参数编辑 `webui-user.bat` 可以获得最佳的 **AI 性能**。
   - 这些配置暗示了社区在 AI 操作中追求硬件**效用最大化**的探索。
- **Adobe Stock 打击艺术家名称的使用**：Adobe Stock 修改了其政策，从标题、关键词或 Prompt 中清除艺术家姓名，展示了 **Adobe 在 AI 内容生成方面的严格方向**。
   - 社区对该政策广泛覆盖范围的担忧，反映了对 AI 艺术**创作边界**的关注。
- **AI 艺术的三大难题**：讨论指出，**手部、文字以及躺在草地上的女性**是 AI 渲染中重复出现的陷阱，类似于艺术家的阿喀琉斯之踵。
   - 人们对能够纠正这些**常见 AI 错误**的新模型充满热情，这些错误被揭示为技术人员之间频繁的谈资。
- **揭开 "Ultra" 功能的神秘面纱**：对 **"Ultra" 功能**的技术推测包括对使用 **Latent Upscale** 或噪声应用等技术的第二采样阶段的期待。
   - 澄清说明了 Ultra 已通过**网站和 API** 提供 Beta 版，将事实与商业化传闻区分开来。
- **关于针对 Troll 进行 IP 封禁的不同看法**：IP 封禁威慑 Troll 的有效性引发了关于其实际操作性的激烈辩论，展示了不同的**社区管理哲学**。
   - 讨论揭示了各种专业见解，强调需要一套策略工具箱来维护**积极的社区**氛围。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **GoldFinch 凭借 Linear Attention 腾飞**：**GoldFinch** 模型成为热门话题，其结合了 Linear Attention 和 Transformers 的混合结构，实现了超长上下文长度并降低了 VRAM 占用。
   - 性能讨论强调了其由于高效的 KV-Cache 机制而优于同类模型，并附带了 [GitHub 仓库](https://github.com/recursal/GoldFinch-paper)和 [Hugging Face 模型](https://huggingface.co/recursal/GoldFinch-paper)链接。
- **数据抓取引发剧烈争议**：关于 AI 模型抓取 YouTube 字幕的讨论异常激烈，引发了一系列经济和伦理方面的考量。
   - 讨论深入探讨了公平使用（fair use）和公共许可的细微差别，一些用户对法律问题和对比性的舆论愤怒进行了反思。
- **扩展 LLM 的 Patchwork 解决方案**：在大型语言模型领域，**Patch-Level Training** 作为一种潜在的改变游戏规则的技术出现，旨在提高序列训练效率。
   - **PatchTrain GitHub** 是主要的参考点，讨论了 Token 压缩及其对训练动态的影响。
- **通过 Token-Free 视角看可解释性**：**Tokenization-free 模型** 登场，引发了对其对 AI **可解释性（interpretability）**影响的好奇。
   - 对话涵盖了理论和潜在实现，虽然未达成共识，但激发了有意义的交流。
- **ICML 2024 的期待令研究人员兴奋不已**：关于 ICML 2024 的讨论层出不穷，其中 **grimsqueaker** 关于“蛋白质语言模型揭示病毒模拟与免疫逃逸”的见解尤为引人关注。
   - 凭借 99.7% 的 ROCAUC 评分，大家对分享的 [海报](https://openreview.net/attachment?id=gGnJBLssbb&name=poster) 和配套的 GitHub [代码](https://github.com/ddofer/ProteinHumVir) 感到兴奋，预示着这将是 ML4LMS 工作坊的一个重要亮点。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Codestral Mamba 潜入 LM Studio**：**Codestral Mamba** 在 LM Studio 中的集成取决于其在 llama.cpp 中的支持，成员们正热切期待其加入。
   - **LM Studio** 对 *Codestral Mamba* 的增强功能将在其并入底层的 llama.cpp 框架后解锁。
- **Resizable BAR 的提升微乎其微**：据报告，Resizable BAR（一项增强 GPU 性能的功能）对更依赖 **tensor cores** 的 **LLM 操作** 影响微乎其微。
   - 讨论集中在硬件功能的效率上，结论是 **VRAM 带宽** 等因素对 LLM 性能优化具有更大的权重。
- **GTX 1050 的 AI 抱负破灭**：**GTX 1050 的 4GB VRAM** 在执行模型时非常吃力，迫使用户考虑缩减到需求较低的 AI 模型。
   - 社区成员讨论了模型兼容性，认为 GTX 1050 有限的 **VRAM** 不适合运行计算更密集的 7B+ 参数模型。
- **Groq 模型摘得函数调用桂冠**：Groq 的 **Llama-3 Groq-8B** 和 **70B** 模型在 **Berkeley Function Calling Leaderboard** 上的表现令人印象深刻。
   - 对 **工具使用（tool use）和函数调用（function calls）** 的关注使这些模型获得了极具竞争力的分数，展示了它们在实际 AI 场景中的效率。
- **CUDA 在 AMD RDNA 上找到新盟友**：在 Reddit 上分享的一篇关于新编译器允许 RX 7800 执行的讨论后，CUDA 在 AMD RDNA 架构上的前景引起了兴趣。
   - 尽管 SCALE 和便携式安装预示着充满希望的替代方案，但对于通过 ZLUDA 等工具在 AMD 上实现 CUDA 的完全兼容性，怀疑态度依然存在。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **文本梯度（Textual Gradients）成为焦点**：围绕一种通过**文本反馈**进行“微分”的新方法展开了热烈讨论，该方法使用 [TextGrad](https://arxiv.org/abs/2406.07496) 来指导神经网络优化。
   - **ProTeGi** 凭借类似的方法步入聚光灯下，引发了关于该技术在机器学习应用中潜力的对话。
- **STORM 系统的文章创作造诣**：斯坦福大学的 [STORM](https://github.com/stanford-oval/storm) 系统利用 LLM 创建全面的大纲，显著提升了长篇文章的质量。
   - 作者们现在正致力于解决**来源偏见转移（source bias transfer）**问题，这是该系统方法论引入的一个新挑战。
- **初步印象：合成数据集与知识库**：**[合成数据集与知识库](https://github.com/Mill-Pond-Research/AI-Knowledge-Base)** 发布，旨在增强以商业应用为中心的 AI 系统。
   - RAG 系统可以在这个包含大量商业相关数据的 **Mill-Pond-Research/AI-Knowledge-Base** 中找到宝贵的资源。
- **当 Agent 进化：超越语言模型**：一项研究呼吁 **LLM 驱动的 Agent** 进行进化，建议采用更复杂的处理方式以改进推理，详见这份富有洞察力的 [立场论文](https://x.com/ManifoldRG/status/1811120196570206459)。
   - **Mistral-NeMo-12Instruct-12B** 震撼发布，宣传其经过多语言和代码数据训练，并拥有 128k 的 context window。
- **GPT-4 的过去式谜题**：GPT-4 对有害请求重构的鲁棒性受到了冲击，一篇 [新论文](https://arxiv.org/abs/2407.11969) 发现，通过过去式提示词获取违禁知识的成功率高达 88%。
   - 研究敦促针对调查结果中强调的这一意外差距，重新审视当前的**对齐技术（alignment techniques）**。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **DeepSeek 的工程优雅**：DeepSeek 的 **DeepSeek V2** 将推理成本削减至 **每百万 token 1 元**，引发了 AI 公司之间的价格战。
   - **DeepSeek V2** 拥有革命性的 **MLA 架构**，降低了内存占用，为其赢得了中国 **AI 拼多多** 的称号。
- **GPT-4o Mini：质量与成本效益的结合**：OpenAI 的 **GPT-4o mini** 以极高的性价比问世：每百万输入 token **$0.15**，每百万输出 token **$0.60**。
   - 它实现了 **82% 的 MMLU 分数** 并支持 **128k context window**，表现优于 Claude 3 Haiku 的 75%，并树立了新的性能成本基准。
- **Mistral NeMo 的性能飙升**：**Mistral AI** 与 **NVIDIA** 联手发布了 **Mistral NeMo**，这是一个强大的 12B 模型，拥有巨大的 **128k tokens context window**。
   - 效率是 **NeMo** 的核心，得益于其 **FP8 推理** 带来的更快性能，使其定位为 **Mistral 7B** 的增强版。
- **涡轮增压版 LLaMA 3 版本跃入视野**：**Together AI** 推出的 **LLaMA-3-8B Lite** 承诺提供具有成本效益的 **每百万 token $0.10**，确保了经济性与速度的兼顾。
   - 为了优化部署格局，**LLaMA-3-8B Turbo** 的速度飙升至 **400 tokens/s**，专为追求极速效率的应用而量身定制。
- **随着 400B 即将发布，对 LLaMA 3 的期待达到顶峰**：关于 **LLaMA 3 400B** 即将揭晓的猜测甚嚣尘上，恰逢 Meta 高管预定的会议，有望动摇 AI 现状。
   - 社区感觉到正在对现有产品进行战略性清理，为 **LLaMA 3 400B** 的突破性登场铺平道路。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4o Mini：高性价比的智能**：OpenAI 推出了 **GPT-4o mini**，据称比 **GPT-3.5 Turbo** 更聪明且更具成本效益，目前正在 API 和 ChatGPT 中推出，有望彻底改变可访问性。
   - 在兴奋和疑问中，官方澄清了 **GPT-4o mini** 在 **GPT-3.5** 的基础上有所增强，但并未超越 **GPT-4o**；相反，它缺乏 GPT-4o 的全套功能（如图像支持），但永久性地升级了 GPT-3.5。
- **Eleven Labs 开启语音提取功能**：Eleven Labs **语音提取模型（voice extraction model）**的发布引发了热烈讨论，该模型利用其潜力从嘈杂背景中提取清晰的语音音频。
   - 参与者正在权衡伦理考量和潜在应用，这与合成媒体中的创新创作方向一致。
- **Nvidia 的 Meta 软件包之谜**：Nvidia 安装程序与 Facebook、Instagram 以及 Meta 版 Twitter 的集成在用户中引起了困惑和幽默的交织。
   - 诸如“Yes sir”确认之类的随意反应反映了社区对这种意外捆绑的轻松回应。
- **关于 EWAC 效率的讨论**：新的 EWAC 命令框架作为一种有效的 zero-shot 系统提示（system prompting）解决方案受到关注，优化了模型命令执行。
   - 共享的[讨论链接](https://discord.com/channels/974519864045756446/1263348214749335613)鼓励对该命令装置进行协作探索和微调。
- **OpenAI API 配额困境**：讨论围绕管理 OpenAI API 配额问题展开，提倡警惕监控计划限制，并建议购买额度以继续使用。
   - 社区成员交流策略以避免 API 幻觉并最大化其使用率，同时还指出了图像 Token 计数的不一致性。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Meta 与隐现的 Llama**：Meta 计划首次推出多模态 **Llama 模型**，但由于监管限制，欧盟用户陷入困境，[Axios 报告](https://www.axios.com/2024/07/17/meta-future-multimodal-ai-models-eu)重点介绍了这一问题。
   - 诸如使用 **VPN** 或非欧盟合规复选框之类的变通方法已被私下讨论，预示着 **访问破解（access hacks）** 可能会增加。
- **Mistral NeMo 隆重登场**：**Mistral NeMo** 凭借其专为 128k Token 上下文窗口设计的 12B 模型引起轰动，这是与 NVIDIA 合作并在 Apache 2.0 协议下开源的项目，详见[官方发布说明](https://mistral.ai/news/mistral-nemo/)。
   - 它超越了前代产品，在推理、世界知识和代码精确度方面有所增强，激发了技术圈的期待。
- **GPT-4o mini 抢尽风头**：OpenAI 推出了精干且强大的 **GPT-4o mini**，因其智能和成本透明度而受到赞誉。根据 [Andrew Curran](https://x.com/andrewcurran_/status/1813942258968018954?s=46) 的说法，它因免费可用和强大的 MMLU 表现而获得认可。
   - 定价极具攻击性，每百万 Token 输入 15 美分，每百万 Token 输出 60 美分，虽然规模缩小但极具竞争力，并承诺更广泛的可访问性。
- **Tekken Tokenizer 应对多语言**：**Tekken tokenizer** 已成为热门话题，效率极高且具备多语言灵活性，大幅领先于 **Llama 3** tokenizer。
   - 它在压缩文本和源代码方面的娴熟表现令人瞩目，让开发者们开始考虑下一步行动。
- **OpenAI 的防御边界被突破**：**OpenAI** 最新的安全机制已被破解，`Elder Plinius` 通过[一项大胆声明](https://x.com/elder_plinius/status/1814023961535295918?s=46)揭示了 GPT-4o-mini 的 jailbreak 方法。
   - 这揭示了其“指令层级（instruction hierarchy）”防御中的裂痕，引起了人们对安全性的关注和质疑。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **GPT-4o Mini 的经济灵活性**：OpenAI 的 [GPT-4o Mini](https://www.cnbc.com/2024/07/18/openai-4o-mini-model-announced.html) 在文本和图像输入方面表现出色，同时拥有极具**性价比**的价格，**每百万输入 $0.15**，**每百万输出 $0.60**。
   - 该模型在经济性上击败了 [GPT-3.5 Turbo](https://openrouter.ai/models/openai/gpt-3.5-turbo)，成本降低了 60% 以上，对免费和订阅用户都极具吸引力。
- **OpenRouter 的区域韧性难题**：用户面临着不稳定的 **OpenRouter 停机**，有报告称 API 请求延迟和网站超时，尽管北欧等部分地区未受影响。
   - 这种零星的服务状态让工程师们不得不查看 [OpenRouter Status](https://status.openrouter.ai/) 的实时更新以应对中断。
- **Mistral NeMo 释放上下文容量**：**Mistral NeMo 的发布**引起了轰动，这款强大的 12B 模型拥有 **128k token 上下文**窗口，为 AI 的广泛应用铺平了道路。
   - [Mistral NeMo](https://t.co/FgHDivTLh5) 采用 Apache 2.0 协议提供，现已提供预训练和指令微调的 Checkpoints，确保了广泛的访问和使用。
- **Codestral 22B：代码模型竞争者**：社区发出的呼声推动了 Codestral 22B 的加入，重点展示了 [Mamba-Codestral-7B](https://huggingface.co/mistralai/mamba-codestral-7B-v0.1) 加入现有 AI 模型阵容的能力。
   - 凭借其在基于 Transformer 的代码框架中的竞争优势，Codestral 22B 在开发者和模型策展人中引发了热烈讨论。
- **GPT-4o Mini 中的图像 Token 波动**：随着 AI 工程社区对新发布的 GPT-4o Mini 与之前模型相比的**图像 Token 定价差异**进行审查，讨论随之兴起。
   - 辩论随之而来，一些人对影响使用成本的意外计数表示担忧，这需要对模型的效率和经济性进行更深入的检查。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Nightly Mojo 编译器增强功能发布**：全新的 **Mojo 编译器更新 (2024.7.1805)** 包含了针对使用列表字面量的嵌套 Python 对象的 `stdlib` 增强，可通过 `modular update [nightly/mojo](https://github.com/modularml/mojo/compare/e2a35871255aa87799f240bfc7271ed3898306c8...bb7db5ef55df0c48b6b07850c7566d1ec2282891)` 命令进行升级。
   - 在讨论 stdlib 的未来时，一项**[提案](https://github.com/modularml/mojo/discussions/3233)**建议通过 `stdlib-extensions` 在完全集成前收集社区反馈，旨在在添加分配器感知（allocator awareness）等小众特性前达成共识。
- **Lubeck 凭借出色的 LLVM 效率超越 MKL**：**[Lubeck 的卓越性能](http://blog.mir.dlang.io/glas/benchmark/openblas/2016/09/23/glas-gemm-benchmark.html)**优于 MKL，这归功于差异化的 LLVM IR 生成，可能得益于 Mir 的 LLVM 加速通用数值库（LLVM-Accelerated Generic Numerical Library）。
   - **SPIRAL** 的独特之处在于自动执行数值内核优化，尽管其生成的[代码复杂性](https://spiral.ece.cmu.edu/pub-spiral/pubfile/paper_146.pdf)限制了其在 BLAS 等主要领域之外的使用。
- **Max/Mojo 拥抱 GPU 的并行威力**：围绕 **[Max/Mojo 的新 GPU 支持](https://github.com/modularml/mojo/issues/3262)** 的兴奋感正在升温，这扩展了张量操作和并行运算的能力，Max 平台的一次演讲强调了 NVIDIA GPU 的集成。
   - 讨论线程建议利用 MLIR 方言和 CUDA/NVIDIA 进行并行计算，重点关注 AI 进步的潜力。
- **Keras 3.0 以多框架支持取得突破**：在社区讨论中，最新的 Keras 3.0 更新展示了与 **JAX、TensorFlow 和 PyTorch** 的兼容性，将其定位为灵活、高效的模型训练和部署的领跑者。
   - 这一里程碑在 Mojo 社区会议上分享，进一步暗示了 Keras 集成和[实用性](https://keras.io/keras_3/)的更广阔前景。
- **Max 中的交互式聊天机器人设计引发讨论**：MAX 24.4 版本通过 `--prompt` 标志进行了创新，通过维护上下文和建立系统提示词来促进交互式聊天机器人的创建，正如 [Max 社区直播](https://www.youtube.com/live/uookgZ7Ojg8?si=u-iwoMJWmMigVwSH&t=1197)中所揭示的那样。
   - 关于命令行提示词使用和模型权重 URI 的疑问引发了关于 UI 动态以及使用 Hugging Face 以外的[替代仓库](https://huggingface.co/meta-llama/)获取权重的可行性的启发性讨论。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 工具成为焦点**：一名成员关于创建 **API tools** 的咨询揭示了工具是 **API-only** 的，可以通过 [Cohere dashboard](https://dashboard.cohere.com) 获取相关见解。
   - 来自 [Cohere 文档](https://docs.cohere.com/docs/tool-use) 的细节澄清了工具可以是单步或多步的，并且由客户端定义。
- **GIF 在权限讨论中等待绿灯**：关于在聊天中发送图片和 GIF 的能力引发了讨论，由于潜在的滥用风险，目前面临 **受限权限**。
   - 管理员提到可能会进行更改，允许开发者和普通用户分享视觉内容，以寻求表达与审核之间的平衡。
- **DuckDuckGo：探索集成之路**：成员们考虑将 DuckDuckGo 搜索工具集成到项目中，并强调了 [DuckDuckGo Python package](https://pypi.org/project/duckduckgo-search/) 的潜力。
   - 有建议提出使用该包开发自定义工具，以增强项目功能。
- **Python 与 Firecrawl 打造爬虫协同效应**：讨论了使用 **Python** 进行爬虫以及 **Firecrawl** 的使用，并展望了将两者结合以进行高效内容抓取的前景。
   - 社区同仁推荐使用 [duckduckgo-search library](https://pypi.org/project/duckduckgo-search/) 来收集 URL，作为爬虫过程的一部分。
- **GPT-4o 加入 API 集成阵营**：将 **GPT-4o API** 与爬虫工具集成是一个热门话题，其中个人 API key 被用于增强 Firecrawl 的能力。
   - 共享了技术设置方法，例如配置 .env 文件以包含 API key，从而促进与 LLM 的集成。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **NextCloud 应对 Perplexity 难题**：用户报告在 NextCloud 上 **设置 Perplexity API 遇到困难**，特别是选择正确的模型方面。
   - 分享了一个涉及在请求中指定 `model` 参数的解决方案，并将用户引导至 [完整的模型列表](https://docs.perplexity.ai/docs/model-cards)。
- **Google 应对 Sheets 故障**：社区成员正在排查一个令人困惑的 Google Sheets 问题，遇到了 **Google Drive 标志错误**。
   - 尽管做出了努力，他们仍面临持续的“无法访问页面”问题，导致部分用户无法登录。
- **PDF 探究：无限制利用 Perplexity**：社区讨论了 Perplexity 目前在 **处理多个 PDF 方面的限制**，并寻求克服这一挑战的策略。
   - 一位社区成员建议将 PDF 和网页搜索内容转换为文本文件，以获得最佳的 Perplexity 性能。
- **AI 差异辩论：GPT-4 vs GPT-4 Omni**：围绕 Perplexity 在集成 GPT-4 Omni 时产生不同结果的讨论引发了好奇和猜测。
   - 成员们辩论了可能的原因，见解暗示了 **底层模型的差异**。
- **DALL-E 困境与罗技 (Logitech) 真实性检查**：提出了关于 **DALL-E 更新** 与 Perplexity Pro 搜索重置同步的问题，同时对罗技提供的 Perplexity Pro 优惠表示怀疑。
   - 随后确认了罗技与 Perplexity 的合作伙伴关系，消除了对网络钓鱼的担忧，并有 [相关推文](https://x.com/dmitry140/status/1813698975884792095) 支持。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain 利用 Openrouter**：讨论围绕将 **Openrouter** 与 LangChain 集成展开，但对话缺乏具体的示例或详细指南。
   - 提出了对基于代码的 **RAG** 示例以驱动问答聊天机器人的需求，这表明 **LangChain** 社区对全面、实操性演示的渴望。
- **Langserve 发布 Debugger 容器**：一位用户询问了 **Langserve Debugger 容器**，寻求对其角色和应用的澄清，并附带了 [Docker registry 链接](https://registry.hub.docker.com/r/langchain/langserve-debugger)。
   - 关于 **Langserve Debugger** 与标准 **Langserve** 容器之间区别的好奇心达到顶峰，这直接影响了开发工作流和部署策略。
- **GitHub 上处理模板难题**：一个与在 LangChain 的 **ChatPromptTemplate** 中加入 **JSON** 相关的 **KeyError** 问题引发了讨论，并参考了 [GitHub issue](https://github.com/langchain-ai/langchain/issues/1914) 以寻求潜在修复。
   - 尽管一些社区成员找到了 JSON 集成挑战的变通方法，但其他人仍在努力应对模板系统的细微差别。
- **Product Hunt 首发 Easy Folders**：**Easy Folders** 在 **Product Hunt** 上亮相，具有整理聊天记录和管理 Prompt 的功能，详见 [Product Hunt 帖子](https://www.producthunt.com/posts/easy-folders-for-chatgpt-claude)。
   - 宣布了 Easy Folders 30 天 **Superuser 会员**的促销活动，依靠社区的支持和反馈来增强参与度。
- **LangGraph 结合 Corrective RAG**：[YouTube 教程](https://www.youtube.com/watch?v=7h6uDsfD7bg)展示了 **LangGraph** 与 **Corrective RAG** 的集成，以对抗聊天机器人的幻觉，为改进 AI 聊天机器人提供了思路。
   - 这种新颖的方法暗示了社区通过 RAG Fusion 等创新组合来增强 AI 聊天机器人可信度和可靠性的动力，解决了现有聊天机器人技术的基本问题。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Jerry 的 AI World 智慧**：想要了解 **AI World's Fair** 的最新动态？观看 @jerryjliu0 在他的[主题演讲](https://t.co/o93s5WSMIV)中发表的关于知识助手的见解，这是去年活动的一个亮点。
   - 他深入探讨了这些助手的演变和未来，引发了社区内持续的技术讨论。
- **RAGapp 的聪明伙伴**：RAGapp 的最新版本现在可以与 [MistralAI](https://twitter.com/llama_index/status/1813972705466831164) 和 [GroqInc](https://twitter.com/llama_index/status/1813972705466831164) 接口，为开发者增强了计算创造力。
   - 加入 @cohere 的 reranker 旨在优化和**增强应用结果**，为大语言模型的集成引入了新的动态。
- **RAG 评估器辩论**：围绕 RAG 流水线评估框架的选择展开了对话，正在审查受限的 Ragas 工具的替代方案。
   - 贡献者讨论了在**紧张的两周时间内**构建自定义评估工具是否合适，但未达成共识。
- **数据安全脱敏探索**：我们的社区剖析了保护敏感数据的策略，推荐使用 LlamaIndex 的 [PIINodePostprocessor](https://docs.llamaindex.ai/en/stable/api_reference/postprocessor/PII/) 在 OpenAI 处理前进行数据脱敏。
   - 这一 Beta 功能代表了在 AI 交互中确保用户隐私和数据安全处理的**积极步骤**。
- **多模态 RAG 的微调未来主义**：当一名成员分享他们使用 **GPT4o** 和 **Sonnet3.5** 进行**多模态 RAG 的成功经验**时，热情达到了顶峰，强调了 LlamaIndex 在处理复杂文件时出人意料的高质量响应。
   - 他们的发现激发了人们对 LlamaIndex 更广泛能力的兴趣，并展望了其简化 RAG 部署和提高效率的潜力。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **OpenInterpreter 社区规模显著增长**：**OpenInterpreter** Discord 庆祝了一个重要的里程碑，社区成员数达到了 **10,000 名**。
   - 社区内的热情显而易见，成员们用 **"Yupp"** 和 **"Awesome!"** 等反应来纪念这一成就。
- **经济型 AI 性能超越 GPT-4**：一位成员称赞了一款性价比极高的 AI，声称其性能优于 **GPT-4**，特别是在作为 **AI agents** 使用时表现出色。
   - 社区强调了其易用性和极低的价格，一位成员评论道：*“它基本上是免费的。”*
- **Multimodal AI：快速且多功能**：讨论重点介绍了一款新型 **multimodal** AI 的**惊人速度**，它支持多种功能，并可通过 **API** 调用。
   - 社区成员对其多样化的应用感到兴奋，评论强调了其 **“极低的延迟”** 和 **multimodal** 特性。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Mistral NeMo 惊艳登场**：**Mistral NeMo** 拥有 12B parameter 规模和 128k token context，凭借卓越的推理和编程能力正式亮相。预训练版本已在 Apache 2.0 license 下发布，详情见 [Mistral 官网](https://mistral.ai/news/mistral-nemo/)。
   - Meta 报告的 **Llama 3 8B** 的 5-shot MMLU score 受到质疑，因为其数据与 Mistral 的测试结果不一致，导致人们对声称的 **62.3%**（对比 Mistral 的 **66.6%**）产生怀疑。
- **Transformer 推理能力综述**：一场关于 **Transformers** 在隐式推理中潜力的辩论被引发，[新研究](https://arxiv.org/abs/2405.15071) 提倡在经过大规模训练后它们具备这种能力。
   - 尽管在 intra-domain inferences 方面取得了成功，但如果没有更多的迭代层交互，**Transformers** 尚未能克服 out-of-domain 的挑战。
- **Rank Reduction 与 Eval Loss 的关联**：一个有趣的观察显示，降低模型 rank 与 **eval loss** 的显著下降相关。然而，目前数据尚不足以确定这一趋势是否会在随后的训练步骤中持续。
   - 参与者就这一现象交换了见解，思考其对模型准确性和计算效率的影响。
- **GEM-A 的过拟合困境**：有关 **GEM-A** 模型在训练期间可能出现过拟合的担忧浮出水面。
   - 对话仍在继续，目前尚无具体解决方案，但充满了谨慎气氛，并表示需要进一步实验。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **GPT-3.5-turbo 占据领先地位**：讨论显示 **GPT-3.5-turbo** 在 **fine-tuning** 任务中超越了 **Mistral 7B** 和 **Llama3**，尽管 OpenAI 坚持不使用用户提交的数据进行 **fine-tuning**。
   - 有观点表示不倾向于使用 GPT 模型进行 **fine-tuning**，原因是担心将敏感信息传输给第三方。
- **Mac M1 的模型延迟滞后**：由于在启动 **preprocessing pipeline** 时的模型加载时间问题，用户在 Mac M1 芯片上运行 **Hugging Face** 模型时遇到了延迟问题。
   - 当尝试多个模型时，延迟会进一步加剧，因为每个模型都需要单独下载和加载，从而增加了初始延迟。

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Meta 的多模态使命**：Meta 将目光投向未来，将焦点转向 **multimodal AI models**，正如一篇分享文章后引发的零星讨论所暗示的那样。
   - 欧盟社区缺乏实质性辩论，使得对创新和可访问性的影响悬而未决。
- **Llama 离开欧盟**：Meta 决定从欧盟市场撤回 **Llama models**，在用户中引起了低调的回应。
   - 对区域性 AI 工具可访问性的影响尚未得到深入审查或辩论。
- **Codestral Mamba 的编程突破**：新推出的 [Codestral Mamba](https://mistral.ai/news/codestral-mamba/) 凭借其线性时间推理（linear time inference）承诺在代码生成方面实现飞跃。
   - 该模型由 **Albert Gu** 和 **Tri Dao** 共同创建，旨在提高编码效率，同时处理无限长的序列。
- **Prover-Verifier 的可读性飞跃**：**Prover-Verifier Games** 的引入引发了提高语言模型可读性的兴趣，并得到了一些技术参考资料的支持。
   - 完整[文档](https://cdn.openai.com/prover-verifier-games-improve-legibility-of-llm-outputs/legibility.pdf)中提供了详细信息，在实际应用出现之前，社区表现出的热情较为克制。
- **NuminaMath-7B 夺冠但存在缺陷**：尽管 [NuminaMath-7B 在 AIMO 夺冠](https://x.com/JJitsev/status/1813930981637902486)，但其在基础推理能力方面暴露出的缺陷被视为前车之鉴。
   - AI 资深人士正在思考，在 AIW 问题审查下，那些经不起基础推理考验的强力主张所带来的严重性和后续影响。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **模板困惑的转机**：在使用 `torchtune.data.InstructTemplate` 进行自定义模板格式化时产生了困惑，特别是关于 **column mapping**。
   - 随后进行了澄清，说明列映射会重命名数据集列，并询问是否正在使用 **alpaca cleaned dataset**。
- **代码贡献中的 CI 难题**：关于 CI 行为的讨论指出，更新 PR 时会自动运行，这让贡献者感到困惑。
   - 共识建议是在 PR 从草稿状态转变为准备好进行同行评审的状态之前，忽略 CI 结果。
- **爆笑 LLMs —— 现实还是虚构？**：尝试编程让 LLM 在喂入特定数据集的情况下持续输出 'HAHAHA'，但模型并未遵循。
   - 该实验是为一个使用 **alpaca dataset** 的项目进行更严肃应用的前奏。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **GTX1080 在 tinygrad 上受阻**：用户在 **GTX1080** 上尝试以 **CUDA=1** 运行 tinygrad 时遇到 **nvrtc: error**，怀疑与旧的 GPU 架构不兼容。
   - 解决方法包括为 GTX1080 修改 **ops_cuda** 并关闭 **tensor cores**；然而，为了获得最佳性能，可能需要 **GTX2080** 或更新的显卡。
- **编译难题导致转向新技术测试**：成员在 GTX1080 上设置 tinygrad 时遇到障碍，表明存在潜在的兼容性问题。
   - 他们采纳了社区建议，转而使用更现代的系统来实验 tinygrad。

---

# PART 2: Detailed by-Channel summaries and links

{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1263215251873202176)** (245 条消息🔥🔥): 

> - `RAG`
> - `Mistral NeMo release`
> - `Unsloth Studio`
> - `Mistral-Nemo integration`
> - `Flash Attention support` 


- **关于 RAG 及其实现的辩论**：多位成员表示，尽管 **RAG** (Retrieval Augmented Generation) 难以实现且需要大量的自定义调优才能获得有效结果，但它往往被过度推销和炒作。
   - 一位成员指出：*“如果你只想要一个糟糕的结果，RAG 很简单——微调也是如此，但如果你想要一个好的结果……那将需要大量的工作”*。
- **Mistral NeMo 12B 模型发布**：Mistral NeMo 是一款拥有高达 **128k tokens** 上下文窗口的 12B 模型，是与 **NVIDIA** 合作开发的，并根据 **Apache 2.0 许可证**发布。
   - 该模型承诺在其尺寸类别中具有 **最先进的推理、世界知识和代码准确性**，是 **Mistral 7B** 的直接替代品。
- **Unsloth Studio 发布推迟**：**Unsloth Studio** (Beta) 的发布时间从原定日期推迟到了下周一。
   - 一位成员建议 *“慢慢来——该花多少时间就花多少时间”*。
- **Mistral-Nemo 的引入与兼容性**：关于 **Unsloth** 是否会支持新发布的 **Mistral-Nemo** 的讨论正在进行中。
   - 集成工作仍在进行中，对于更长的上下文长度，似乎需要 **Flash Attention 2.6**。
- **训练模型的硬件选择**：成员们讨论了各种硬件设置，指出了使用 **A6000 GPU** 的好处以及租用硬件进行 AI 训练的性价比。
   - 一位成员提到使用 **64 核 Threadripper 配备双 A6000** GPU，最初主要用于 **流体动力学和有限状态模拟**，现在已重新用于 AI。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://mistral.ai/news/mistral-nemo/">Mistral NeMo</a>: Mistral NeMo：我们最新的最佳小型模型。一款具有 128k 上下文长度的先进 12B 模型，与 NVIDIA 合作构建，并根据 Apache 2.0 许可证发布。</li><li><a href="https://arxiv.org/html/2407.07858v1">FACTS About Building Retrieval Augmented Generation-based Chatbots</a>: 关于构建基于检索增强生成（RAG）聊天机器人的事实</li><li><a href="https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407">mistralai/Mistral-Nemo-Instruct-2407 · Hugging Face</a>: 暂无描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e5uxnj/thanks_to_regulators_upcoming_multimodal_llama/">Reddit - Dive into anything</a>: 暂无描述</li><li><a href="https://www.tomshardware.com/tech-industry/artificial-intelligence/gigabyte-releases-ai-software-to-help-train-your-own-ai">Gigabyte releases AI software to help train your own AI &mdash; AI TOP utility taps Gigabyte motherboards, GPUs, SSDs, and power supplies to fine-tune local AI model training</a>: 该工具可以在本地训练参数高达 236B 的 AI 模型。</li><li><a href="https://github.com/bclavie/RAGatouille">GitHub - bclavie/RAGatouille: Easily use and train state of the art late-interaction retrieval methods (ColBERT) in any RAG pipeline. Designed for modularity and ease-of-use, backed by research.</a>: 在任何 RAG 管道中轻松使用和训练最先进的后期交互检索方法 (ColBERT)。专为模块化和易用性而设计，并有研究支持。</li><li><a href="https://datta0.substack.com/p/aiunplugged-15-gemma-2-flash-attention">AIUnplugged 15: Gemma 2, Flash Attention 3, QGaLoRE, MathΣtral and Codestral Mamba</a>: 洞察胜过信息
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1263316194920763413)** (11 条消息🔥): 

> - `GPU recommendations`
> - `Runpod`
> - `Binary message`
> - `Shylily fans` 


- **GPU 辩论：3090 vs 4090**：成员们讨论了购买二手 **RTX 3090** 而非 **3090 TI** 的优点，并建议即使是矿卡也是可以接受的。
   - 一位用户提到拥有两块 **4090** GPU，并强调 **Runpod** 是一个更优的替代方案。
- **二进制消息的乐趣**：成员们度过了一个轻松的时刻，将二进制消息 **'01110111 01101111 01101101 01110000 00100000 01110111 01101111 01101101 01110000'** 解码为 'womp womp'。
   - 这引发了成员之间一系列俏皮的 'womp womp' 交流。
- **Shylily 粉丝时刻**：一位成员指出聊天中有很多 **Shylily 粉丝**，并点出了其中的有趣时刻。
   - 讨论引起了人们对聊天记录中许多 'womp womp' 时刻的关注，引发了笑声和评论。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1263241651560644728)** (84 messages🔥🔥): 

> - `disabling pad_token` (禁用 pad_token)
> - `finetuning and saving models` (微调与保存模型)
> - `model sizes and memory consumption` (模型大小与显存消耗)
> - `fine-tuning locally` (本地微调)
> - `handling new errors with GPU and dtype` (处理 GPU 和 dtype 的新错误)


- **如果模型不需要 pad_token 则忽略它**：成员们讨论了如果模型不使用 `pad_token`，可以直接忽略它而不会产生任何影响。
- **在 VLLM 中将微调后的模型保存为 4-bit 还是 16-bit 的最佳实践**：成员们辩论了是将微调后的 4-bit 模型保存为 16-bit，还是在生产环境中仅使用 LoRA adapters。
   - *Theyruinedelise* 建议尝试 LoRA 方案以保持精度。
- **Llama 3 模型的高 VRAM 需求**：成员们讨论了 Llama 3 模型极高的 VRAM 需求，指出 70B 模型在 4-bit 量化下需要 48GB VRAM。
- **本地运行微调 vs. 使用 Colab**：Efficio 寻求关于本地运行微调的建议，并收到了在 Windows 上使用 WSL 进行本地训练的建议。
   - Theyruinedelise 建议参考 GitHub 上提供的详细指南来搭建本地环境。
- **处理 RTX A4000 上新的 torch.autocast 错误**：Kiingz3440 在 RTX A4000 GPU 上遇到了与不支持 `bfloat16` 相关的 `torch.autocast` 错误。
   - Edd0302 建议显式设置模型使用 `torch.float16`，从而解决了该错误。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://learn.microsoft.com/en-us/windows/wsl/install">Install WSL</a>: 使用命令 wsl --install 安装 Windows Subsystem for Linux。在您的 Windows 机器上使用由您偏好的 Linux 发行版（Ubuntu, Debian, SUSE, Kali, Fedora, Pengwin...）运行的 Bash 终端。</li><li><a href="https://tinygrad.org/#tinybox">tinygrad: A simple and powerful neural network framework</a>: 一个简单且强大的神经网络框架。</li><li><a href="https://github.com/unslothai/unsloth/issues/210">I got unsloth running in native windows. · Issue #210 · unslothai/unsloth</a>: 我在原生 Windows（非 WSL）上运行了 unsloth。你需要 Visual Studio 2022 C++ 编译器、triton 和 deepspeed。我有一个完整的安装教程...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1263300998995378238)** (7 messages): 

> - `Synthesis of Topic Outlines through Retrieval and Multi-perspective Question Asking (STORM)` (通过检索和多视角提问合成主题大纲)
> - `EfficientQAT for LLM Quantization` (用于 LLM 量化的 EfficientQAT)
> - `Memory3 Architecture for LLMs` (用于 LLM 的 Memory3 架构)
> - `Spectra LLM Suite and Quantization` (Spectra LLM 套件与量化)
> - `Patch-Level Training for LLMs` (用于 LLM 的 Patch-Level 训练)

- **STORM 革新了 AI 的预写作流程**：[STORM](https://arxiv.org/abs/2402.14207) 引入了一种名为 **Synthesis of Topic Outlines through Retrieval and Multi-perspective Question Asking**（通过检索和多视角提问合成主题大纲）的新颖系统，使 LLM 能够编写类似于 Wikipedia 页面的、有据可查且组织严整的长篇文章。
   - 该方法包括**发现多样化视角**、模拟专家对话以及整理信息，与之前的方法相比，在组织结构上提升了 25%，在覆盖范围上提升了 10%。
- **EfficientQAT 实现低比特量化**：[EfficientQAT](https://github.com/OpenGVLab/EfficientQAT) 成功突破了**均匀 INT 量化**的极限，在单张 A100-80GB GPU 上实现了 2-bit Llama-2-70B 模型，且与全精度模型相比，准确率下降不到 3%。
   - *EfficientQAT* 证明了 INT2 量化模型在占用更少内存的同时，能获得比更大型模型更好的准确率，突显了可部署 LLM 量化技术的进步。
- **Memory3 增强 LLM 效率**：[Memory3](https://www.marktechpost.com/2024/07/05/memory3-a-novel-architecture-for-llms-that-introduces-an-explicit-memory-mechanism-to-improve-efficiency-and-performance) 为 LLM 引入了一种显式记忆机制，旨在同时提高**效率和性能**。
   - 该架构通过引入一种更高效地处理和存储信息的新方法，解决了当前 LLM 架构中的挑战。
- **Spectra LLM 系列发布**：[Spectra LLM](https://huggingface.co/papers/2407.12327) 推出了 54 个在 300B token 上训练的语言模型，包括 FloatLMs、训练后量化模型以及**三值 LLM (TriLMs)**，其性能优于相同比特大小的早期三值模型。
   - 该系列证明了 TriLMs 可以达到半精度模型的性能，为更高效、更小型的 LLM 部署铺平了道路。
- **Patch-Level 训练提升 LLM 效率**：针对 LLM 的 [Patch-level 训练](https://arxiv.org/abs/2407.12665) 引入了一种通过将多个 token 压缩到单个 patch 中来减少序列长度的方法，从而显著降低了计算成本。
   - 这种新的训练方法允许 LLM 更高效地处理训练数据，并在后期切换到 token-level 训练以匹配推理需求。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/papers/2407.12327">Paper page - Spectra: A Comprehensive Study of Ternary, Quantized, and FP16 Language
  Models</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2407.12665">Patch-Level Training for Large Language Models</a>: 随着大语言模型 (LLMs) 在语言理解和生成方面取得显著进展，其训练效率已成为一个关键问题。传统上，LLM 被训练用于预测...</li><li><a href="https://x.com/MrCatid/status/1813829489039900999?t=CaNeBo4ErLUe_irte2yoBQ&s=19">catid (e/acc) (@MrCatid) 的推文</a>: LLM 训练速度提升 2 倍: https://arxiv.org/abs/2407.12665 可能也适用于其他类型的 Transformer 模型！</li><li><a href="https://www.marktechpost.com/2024/07/05/memory3-a-novel-architecture-for-llms-that-introduces-an-explicit-memory-mechanism-to-improve-efficiency-and-performance">未找到标题</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2402.14207">Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models</a>: 我们研究了如何应用大语言模型从头开始编写有据可查且组织严整的长篇文章，其广度和深度可与 Wikipedia 页面相媲美。这个尚未被充分探索的问题带来了新的...</li><li><a href="https://storm.genie.stanford.edu/article/ai-human-relations-and-the-complexity-it-introduces-to-society-18731">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/stanford-oval/storm">GitHub - stanford-oval/storm: An LLM-powered knowledge curation system that researches a topic and generates a full-length report with citations.</a>: 一个由 LLM 驱动的知识整理系统，可研究特定主题并生成带有引用的完整报告。- stanford-oval/storm</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/P84n4i083q">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://github.com/OpenGVLab/">OpenGVLab</a>: 上海人工智能实验室通用视觉团队。OpenGVLab 拥有 65 个可用仓库。在 GitHub 上关注他们的代码。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1263597700813946882)** (1 messages): 

> - `Watermark Remover`
> - `CandyLLM`
> - `AI Comic Factory Update`
> - `Fast Subtitle Maker`
> - `HF Text Embedding on Intel GPUs` 


- **使用 Florence 2 的 Watermark Remover**：一位成员介绍了一个使用 **Florence 2** 的 [水印去除器](https://huggingface.co/spaces/DamarJati/Remove-watermark)。
   - 他们分享道：*'它高效且易于使用，能满足你所有的水印去除需求。'*
- **带有 Gradio UI 的 CandyLLM Python 库**：@shreyanmitra_05940_88933 介绍了 **CandyLLM**，这是一个利用 [Gradio UI](https://github.com/shreyanmitra/CandyLLM) 的 Python 库。
   - 作者表示：*'该工具旨在简化应用程序中语言模型的使用。'*
- **AI Comic Factory 添加默认对话气泡**：@jbilcke 宣布，**AI Comic Factory** 现在默认包含 [对话气泡](https://huggingface.co/spaces/jbilcke-hf/ai-comic-factory)。
   - 此次更新通过自动为生成的漫画添加对话元素，提升了用户体验。
- **在 Intel GPU 上量化并加载 HF 文本嵌入模型**：一位成员分享了一种在 **Intel GPU** 上 [量化并加载任何 HF 文本嵌入模型](https://github.com/sleepingcat4/intel-hf) 的简便方法。
   - 他们解释道：*'这将有助于高效地利用 Intel 硬件进行 AI 任务。'*



**提到的链接**：<a href="https://youtu.be/cpoS7K_fpRM)">如何从任何领域转型到机器学习？ | Artificial Intelligence ft. @vizuara</a>：在这段视频中，来自 Vizuara 的 Raj Dandekar 博士分享了他从机械工程转型到 Machine Learning (ML) 的经验。他还解释了...

  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1263212225716293662)** (222 messages🔥🔥): 

> - `Huggingchat 性能问题`
> - `模型训练咨询`
> - `RVC 和语音模型替代方案`
> - `Text2text 生成问题`
> - `管理员 Ping 礼仪` 


- **Huggingchat 的 commandR+ 变慢**：用户报告称 Huggingchat 的 commandR+ 运行极其缓慢，某些任务耗时高达 5 分钟，而其他模型仅需 5 秒。
   - 一位用户讽刺地提到，*'他们几乎把同样的消息发了三次'*，以此作为沮丧情绪的例子。
- **模型训练与部署问题**：多位用户讨论了在 AWS 和 Google Colab 等平台上部署和训练模型的问题。
   - 具体问题包括部署时间过长、`text2text-generation` 等特定任务的错误，以及 Hugging Face Spaces 上 GPU 资源不足的问题。
- **RVC 仓库及替代方案**：用户讨论了 RVC 无法工作的问题，并质疑为什么仓库仍然在线，同时在寻找创建 AI 语音模型的替代项目。
   - 关于解决方案或替代方案的建议很少，使得该问题在讨论中基本上未得到解决。
- **需要正确的训练数据格式**：一位用户分享了使用无监督学习预训练 Mistral 的详细方法，但希望对其数据格式和方法进行验证。
   - 回复中就正确的输入格式给出了建议，并提供了一些关于 Token 排除以进行正确训练的指导。
- **Ping 管理员与沟通礼仪**：一位名为 quirkyboi22 的用户频繁 Ping 管理员以修复 Huggingchat 的问题，引发了社区关于适当沟通渠道的多次提醒。
   - 另一位用户指出，*'发送邮件至 website@hf.co'* 是报告此类问题更合适的方法，官方回复确认团队已获悉并正在调查这些具体问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/fal/AuraSR">fal/AuraSR · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/gokaygokay/AuraSR">AuraSR - gokaygokay 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://tenor.com/view/unicorn-happy-birthday-dance-moves-gif-24459212">独角兽快乐 GIF - 独角兽生日快乐 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/abhi1thakur/status/1813892464144798171">来自 abhishek (@abhi1thakur) 的推文</a>：我们刚刚在 AutoTrain 中集成了数据集查看器 💥 现在，你可以在训练模型之前，无需离开页面即可查看数据集、识别正确的拆分和列 🚀</li><li><a href="https://imgur.com/dd3TB7g">imgur.com</a>：在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、热门迷因、娱乐 GIF、励志故事、病毒视频等来振奋你的精神...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 条消息): 

rp0101: https://youtu.be/N0eYoJC6USE?si=zms6lSsZkF6_vL0E
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1263373706856763514)** (7 条消息): 

> - `Transformers.js 教程`
> - `计算机视觉课程`
> - `AutoTrainer`
> - `Mistral NeMo`
> - `Discord 审核` 


- **Transformers.js 增强 Next.js 应用**：一份关于 [Transformers.js](https://huggingface.co/docs/transformers.js/en/tutorials/next) 的教程演示了如何构建一个用于情感分析的 Next.js 应用，并提供了客户端和服务器端推理的选项。
   - 该教程使用了全新的 [App Router](https://nextjs.org/docs/app) 范式，并提供了 [演示链接](https://huggingface.co/spaces/Xenova/next-example-app) 和 [源代码](https://github.com/xenova/transformers.js/tree/main/examples/next-client)。
- **社区驱动的计算机视觉课程启动**：全新的 [社区计算机视觉课程](https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome) 已上线，涵盖了从基础到高级的所有主题。
   - 该课程包含基础主题，并强调为所有对计算机视觉感兴趣的人提供易于获取的学习资源。
- **AutoTrainer 简化 ML 模型训练**：[AutoTrainer](https://huggingface.co/autotrain) 通过简单地上传数据，实现了自定义机器学习模型的自动化训练。
   - 它支持 LLM Finetuning、图像分类和文本分类等多种任务，并拥有与 [Hugging Face Hub](https://huggingface.co/models) 的无缝集成。
- **Mistral NeMo 模型发布！**：MistralAI 发布了 [Mistral NeMo](https://mistral.ai/news/mistral-nemo/)，这是一个与 NVIDIA 合作构建的、具有 128k 上下文长度的 SOTA 12B 模型，采用 Apache 2.0 许可证。
   - [官方推文](https://x.com/mistralai/status/1813947930455499200?s=46&t=IfJRyr-UwyoM2m-vJODIzw) 提供了关于这款高性能模型的更多细节。
- **Discord 中的审核提醒**：一名成员发布了保持 PG（家长指导级）内容的提醒，强调了社区标准。
   - 另一名成员澄清说，相关内容来自一部儿童卡通片，突显了对准则的不同理解。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/autotrain">AutoTrain – Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome">欢迎来到社区计算机视觉课程 - Hugging Face 社区计算机视觉课程</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/transformers.js/en/tutorials/next">构建 Next.js 应用程序</a>: 未找到描述</li><li><a href="https://x.com/mistralai/status/1813947930455499200?s=46&t=IfJRyr-UwyoM2m-vJODIzw">来自 Mistral AI (@MistralAI) 的推文</a>: https://mistral.ai/news/mistral-nemo/
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1263213216935317534)** (23 条消息🔥): 

> - `AI Comic Factory 更新`
> - `视频转录与摘要工具`
> - `AI 助手反馈请求`
> - `Gradio Python 库`
> - `使用 Florence 2 和 Lama Cleaner 的水印去除工具` 


- **AI Comic Factory 现在默认包含对话气泡**：一名成员宣布了 **AI Comic Factory** 的更新，现在**默认包含对话气泡**，尽管该功能仍在开发中。
- **自动转录并总结视频的工具**：一名成员创建了一个自动转录和总结 YouTube 视频的工具，使用 **Deepgram** 进行转录，使用 **Claude** 进行摘要。
- **提高生产力的 AI 助手反馈请求**：一名成员正在开发一款免费的 AI 助手，通过与 **Slack、Gmail 和 Google Calendar** 等工具集成来帮助提高生产力。
- **为机器学习新手准备的 Gradio Python 新库**：一位 Gradio 和 ML 初学者分享了一个[基础 Python 库](https://github.com/shreyanmitra/CandyLLM)，旨在简化文本生成模型的使用。
- **使用 Florence 2 的水印去除工具**：一名成员展示了一个使用 **Florence 2** 和 **Lama Cleaner** 构建的水印去除工具，并分享在 [Hugging Face](https://huggingface.co/spaces/DamarJati/Remove-watermark) 上。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://app.hunch.tools/app/tool/yB85W?tpreview=true&invitationCode=u54c55ff)">Hunch - 团队 AI 工具</a>：创建 AI 工作流和工具，实现知识工作自动化并提升团队生产力</li><li><a href="https://huggingface.co/spaces/DamarJati/Remove-watermark">Remove-WM - DamarJati 的 Hugging Face Space</a>：暂无描述</li><li><a href="https://huggingface.co/spaces/jbilcke-hf/ai-comic-factory">AI Comic Factory - jbilcke-hf 的 Hugging Face Space</a>：暂无描述</li><li><a href="https://sophi.app/">Sophi.app — 使用应用集成和 AI，Sophi 助你完成工作</a>：🚀 智能、主动且可操作的回答引擎，理解你的数字生活并让你保持领先。</li><li><a href="https://github.com/shreyanmitra/CandyLLM">GitHub - shreyanmitra/CandyLLM: 一个简单易用的 HuggingFace 和 OpenAI 文本生成模型框架。</a>：一个简单易用的 HuggingFace 和 OpenAI 文本生成模型框架。 - shreyanmitra/CandyLLM</li><li><a href="https://app.hunch.tools/app/canvas/new/vyg7V?invitationCode=u54c55ff)">Hunch - 团队 AI 工具</a>：创建 AI 工作流和工具，实现知识工作自动化并提升团队生产力
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1263242131560988762)** (5 条消息): 

> - `项目演示推迟`
> - `初学者友好型论文`
> - `机器学习模型层优化` 


- **项目演示推迟**：一名成员原计划演示其项目，但由于时间冲突，演示已推迟，可能会在三周后进行。
- **初学者友好型论文资源**：一名成员询问适合初学者的论文，另一名成员建议查看 [Hugging Face Papers](https://huggingface.co/papers) 或加入 Yannic Kilcher 的 Discord 服务器进行每日论文讨论。
- **优化机器学习模型层**：一名成员正在寻求关于优化机器学习模型层（包括 Dense 层、GRU 和 LSTM GPU kernels）的基础论文和文章，以助力职业发展。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/)** (1 条消息): 

dorbit_: 嘿！有人有使用 Transformers 进行相机标定（camera calibration）的经验吗？
  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1263213639318634506)** (5 条消息): 

> - `Image to Video Diffusion Model` (图像转视频扩散模型)
> - `Prompt Engineering for SVD` (SVD 的提示工程)
> - `Installing Transformers & Accelerate` (安装 Transformers & Accelerate)
> - `Text Classification with Multiple Tags` (多标签文本分类)
> - `YOLO Model Confusion` (YOLO 模型混淆)


- **Stable Video Diffusion 模型讨论**：[用户分享了](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)关于使用 Hugging Face 的 Stable Video Diffusion 图像转视频模型，并寻求生成视频的 Prompt Engineering 建议。
- **在 Colab 中安装 Transformers 和 Accelerate**：一位成员建议使用 `!pip install transformers accelerate` 来为 Colab 项目导入必要的库。
- **处理多标签文本分类**：一位成员询问如何处理具有约 200 个标签的文本分类问题，并考虑为每个标签创建单独的模型。
   - *mattilinnanvuori* 根据经验建议使用单个模型进行多标签分类 (multi-label classification)。
- **YOLO 模型误解**：一位用户在误读了另一位用户关于文本分类的请求后，最初建议对图像分类问题使用 YOLO 模型。



**提到的链接**：<a href="https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt">stabilityai/stable-video-diffusion-img2vid-xt · Hugging Face</a>：未找到描述

  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1263232649317847091)** (6 条消息): 

> - `CUDA kernel splitting` (CUDA kernel 拆分)
> - `Loss masking in LLM training` (LLM 训练中的 Loss masking)
> - `Sebastian Raschka's research insights` (Sebastian Raschka 的研究见解)
> - `NVIDIA open-source kernel modules` (NVIDIA 开源内核模块)
> - `CUDA graphs` (CUDA graphs)


- **拆分 CUDA Kernels 的好处**：成员们讨论了将一个 CUDA kernel 拆分为多个 kernel 可能有益的场景，主要用于多步归约 (multi-step reduction) 期间的内存管理以及实现延迟隐藏 (latency hiding)。
   - 提到的一个特定用例是在 CNN 中，如果将更深层融合在一起，将需要不切实际的内存量。
- **Raschka 质疑 Loss Masking 的益处**：Sebastian Raschka 的研究见解博客质疑了在 LLM 训练中对 prompt tokens 应用 loss 的益处，并引用了一篇关于 [Instruction Tuning With Loss Over Instructions](https://arxiv.org/abs/2405.14394) 的论文。
   - 同时也分享了 [进一步阅读](https://magazine.sebastianraschka.com/p/llm-research-insights-instruction) 以及对他即将出版的新书和 ACM Tech Talk 的提及。
- **NVIDIA 开源 GPU 内核模块**：一位成员分享道，NVIDIA 已完全转向开源 GPU 内核模块，实现了更好的性能并增加了包括异构内存管理 (HMM) 在内的新功能。
   - 更多详情请参阅 [NVIDIA 博客](https://developer.nvidia.com/blog/nvidia-releases-open-source-gpu-kernel-modules/)。
- **寻求 CUDA Graphs 教育材料**：一位用户询问了涵盖 CUDA graphs 的讲座或材料。
   - 针对该查询，尚未提供进一步的细节或资源。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://developer.nvidia.com/blog/nvidia-transitions-fully-towards-open-source-gpu-kernel-modules/">NVIDIA Transitions Fully Towards Open&#x2d;Source GPU Kernel Modules | NVIDIA Technical Blog</a>：随着 R515 驱动程序的发布，NVIDIA 在 2022 年 5 月以 GPL 和 MIT 双重许可开源了一套 Linux GPU 内核模块。初始版本针对数据中心计算 GPU……</li><li><a href="https://magazine.sebastianraschka.com/p/llm-research-insights-instruction">LLM Research Insights: Instruction Masking and New LoRA Finetuning Experiments</a>：讨论 2024 年 5 月最新的模型发布和 AI 研究。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1263251897318379634)** (1 条消息): 

> - `Missing tl.pow` (缺失 `tl.pow`)
> - `triton.language.extra.libdevice.pow()`


- **缺失 `tl.pow` 函数问题**：一位成员提到了 Triton 中缺少 `tl.pow` 函数。
   - 建议使用 `triton.language.extra.libdevice.pow()` 作为替代方案。
- **`tl.pow` 的建议解决方法**：针对缺失 `tl.pow` 函数的建议解决方法是使用 `triton.language.extra.libdevice.pow()`。
   - 这将作为一个临时解决方案，直到 `tl.pow` 的问题得到解决。


  

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1263235922196172880)** (37 条消息🔥): 

> - `CUDA 中的 Profiling 和分布式案例`
> - `CUDA 中的动态共享内存 (Dynamic Shared Memory)`
> - `torch.compile 的问题`
> - `安装 torch-tensorrt`
> - `使用 Triton Kernels 替换 aten::embedding_dense_backward` 


- **CUDA 高效推理的短 Profile**：成员们讨论了为 prefill + 2-3 个 token 的前向传递等区域生成短 Profile，以实现高效的 **batch 准备**和**调度**。
- **动态共享内存使用技巧**：成员们分享了在 CUDA kernel 中声明和使用动态共享内存的技术，例如 `extern __shared__ float shared_mem[];` 以及在 kernel 启动时分配内存。
   - 他们还讨论了通过指针运算将动态共享内存分割成多个数组，以实现更高效的访问。
- **torch.compile 导致模型性能不一致**：一位用户发现 **torch.compile** 是导致模型性能和正确性不一致的根源，特别是在比较稠密模型和稀疏模型时。
   - 建议启用 `torch._dynamo.config.guard_nn_modules=True` 作为修复方案，并链接到了相关的 [GitHub issue](https://github.com/pytorch/pytorch/issues/124717)。
- **torch-tensorrt 安装问题**：一位用户在尝试通过 pip 安装 **torch-tensorrt** 时遇到错误，被建议使用来自 [NVIDIA 官方发布页面](https://github.com/NVIDIA/Torch-TensorRT/releases) 的命令。
   - 问题可能源于不支持的 Python 版本，建议降级到 **3.8 至 3.10** 之间的版本。
- **使用 Triton 替换 aten::embedding_dense_backward**：一位用户希望用融合的 **Triton kernel** 替换 **aten::embedding_dense_backward**，以提高 `nn.Embedding` 的反向传播性能。
   - 建议他们编写自定义的 **Embedding 层**，直接调用 Triton kernels 以获得更好的优化。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/pytorch/issues/124717">Compile doesn&#39;t guard on user NN module attribute · Issue #124717 · pytorch/pytorch</a>: 🐛 描述 Bug：TorchTune 依赖于修改用户 NN 模块的属性来决定是否应用 LoRA 技术。使用模式如下：import contextlib import torch class ...</li><li><a href="https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/">Using Shared Memory in CUDA C/C++ | NVIDIA Technical Blog</a>: 在上一篇文章中，我探讨了线程组对全局内存的访问如何合并为单个事务，以及对齐和步长如何影响不同代产品的合并...</li><li><a href="https://leimao.github.io/blog/CUDA-Shared-Memory-Capacity/">CUDA Shared Memory Capacity</a>: 使用大容量共享内存进行 CUDA Kernel 优化</li><li><a href="https://github.com/NVIDIA/Torch-TensorRT/releases">Releases · pytorch/TensorRT</a>: 使用 TensorRT 的 NVIDIA GPU 版 PyTorch/TorchScript/FX 编译器 - pytorch/TensorRT
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1263499248289185953)** (1 条消息): 

> - `Google Gemma 2 系列模型`
> - `Together AI Flash Attention 3`
> - `QGaLoRE: 用于微调的量化低秩梯度`
> - `Mistral AI MathΣtral 和 CodeStral mamba` 


- **Google Gemma 2 模型表现优于竞争对手**：[Google Gemma 2 系列](https://datta0.substack.com/i/146681354/gemma) 包含的模型表现优于 Llama 2 系列，提供 2B 和 7B 两种尺寸。
   - Gemma 2 最初于 2024 年 2 月发布，在 [Google I/O](https://datta0.substack.com/p/aiunplugged-15-gemma-2-flash-attention) 上正式宣布，并迅速被 LlaMa 3 和 Qwen 2 等更新的模型超越，展示了 AI 技术快速演进的节奏。
- **Flash Attention 3 加速 GPU 性能**：Together AI 推出了 [Flash Attention 3](https://datta0.substack.com/p/aiunplugged-15-gemma-2-flash-attention)，相比 Flash Attention 1 和 2 有显著改进，使 GPU 运行速度大幅提升。
   - 正如这篇 [substack 文章](https://datta0.substack.com/p/aiunplugged-15-gemma-2-flash-attention) 所强调的，这一进步对于增强计算效率和性能至关重要。
- **QGaLoRE：通过量化梯度优化微调**：[QGaLoRE 技术](https://datta0.substack.com/p/aiunplugged-15-gemma-2-flash-attention) 利用量化的低秩梯度来优化模型的微调过程。
- **Mistral AI 的 MathΣtral 和 CodeStral 项目令人印象深刻**：Mistral AI 宣布了 MathΣtral 和 CodeStral mamba 项目，在数学计算和代码效率方面突破了边界，详见最新的 [更新报告](https://datta0.substack.com/p/aiunplugged-15-gemma-2-flash-attention)。



**提到的链接**：<a href="https://datta0.substack.com/p/aiunplugged-15-gemma-2-flash-attention">AIUnplugged 15: Gemma 2, Flash Attention 3, QGaLoRE, MathΣtral and Codestral Mamba</a>：洞察胜过信息

  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1263315357272375407)** (6 条消息): 

> - `CUTLASS 仓库教程`
> - `Nsight CLI 资源` 


- **构建 CUTLASS 仓库教程**：一名成员询问如何构建和运行 [CUTLASS 仓库](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_1.cu) 中的 cute 教程。
   - 另一名成员澄清说，这是通过 **make target** 实现的，并表示如果需要可以提供进一步帮助。
- **使用 Nsight CLI 进行远程 Profile**：一名成员请求关于使用 **Nsight CLI** 以及远程捕获 Profile 以便在 GUI 中进行分析的资源。
   - 有人指出，可以选择将捕获的 Profile 导出为文件，然后通过 GUI 打开。



**提到的链接**：<a href="https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_1.cu">cutlass/examples/cute/tutorial/sgemm_1.cu at main · NVIDIA/cutlass</a>：用于线性代数子程序的 CUDA 模板。欢迎通过在 GitHub 上创建账号为 NVIDIA/cutlass 做出贡献。

  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1263237279691047043)** (2 条消息): 

> - `HF 相关讨论`
> - `FSDP2 取代 FSDP` 


- **发起 HF 相关讨论**：一名用户建议他们应该开始在这个频道讨论 **HF (Hugging Face)** 相关的话题。
   - “我们也在这里讨论一下 HF 相关的内容吧”是向小组提出的具体建议。
- **FSDP2 将取代 FSDP**：一名成员提到 **FSDP2** 将取代 **FSDP**，并建议开始使用 FSDP2。
   - 在解释中提到了 *nf4 是一个例子*，并承诺很快会深入研究细节。


  

---

### **CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1263211234145206394)** (2 条消息): 

> - `Triton compiler details` (Triton 编译器细节)
> - `Triton puzzles solutions` (Triton puzzles 解决方案)


- **Triton 编译器的神奇优化**：一位成员提到 Triton 编译器会**自动处理 GPU 代码优化**，并引用了一篇详细解释该过程的 [博客文章](https://fkong.tech/posts/2023-04-23-triton-cuda/)。
   - 他们强调 Triton 将 Python 代码转换为 Triton IR，进行优化，然后利用 `libLLVM` 和 `ptxas` 编译为 PTX。
- **Triton Puzzle 解决方案发布**：一位成员分享了 Triton puzzles 的 [个人解决方案](https://github.com/alexzhang13/Triton-Puzzles-Solutions)，并提到 Puzzle 12 中的符号表示非常有挑战性。
   - 他们指出，尽管这些解决方案“写得一般但可能正确”，但如果有人时间紧迫，这些方案可能会有所帮助。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/alexzhang13/Triton-Puzzles-Solutions">GitHub - alexzhang13/Triton-Puzzles-Solutions: Personal solutions to the Triton Puzzles</a>：Triton Puzzles 的个人解决方案。可以通过创建 GitHub 账号为该项目做出贡献。</li><li><a href="https://fkong.tech/posts/2023-04-23-triton-cuda/">Demystify OpenAI Triton</a>：通过逐步指导和代码示例，学习如何构建从 OpenAI Triton 到 CUDA 的映射，以实现高性能深度学习应用。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1263264699085226056)** (159 条消息🔥🔥): 

> - `FP8 training settings` (FP8 训练设置)
> - `Layernorm optimizations` (Layernorm 优化)
> - `GPT-3 models` (GPT-3 模型)
> - `Memory management refactoring` (内存管理重构)
> - `FP8 inference` (FP8 推理)


- **如何轻松激活 FP8 训练**：要运行 FP8 训练，请切换到 fp8 分支，确保 defines 设置为 `FORCE_FP8_MATMUL true`、`FORCE_FP8_WEIGHTS false` 和 `FORCE_FP8_ACTIVATIONS true`，然后执行 `./scripts/run_gpt2_1558M.sh`。
   - *arund42* 确认当前参数是为了关闭 FP8，并提到与现有 checkpoints 可能存在兼容性问题。
- **使用 row_reduce 优化 Layernorm**：Arund42 花了一整天时间研究 `train_gpt2fp32cu`，并为 Layernorm 添加了新的 `row_reduce()` 函数，旨在获得更好的性能和更清晰的抽象。
   - 这种方法在更高层级的抽象方面评价褒贬不一，但因其没有使用过于复杂的 C++ 而受到赞赏。
- **首个 GPT-3 模型 (125M) 训练启动**：Akakak1337 已启动首个 GPT-3 模型 (125M) 的训练，预计运行时间约为 12 小时，并强调了对性能 benchmarks 的预期。
   - 提议测试各种配置，包括内存消耗和最大 batch size，重点关注关键优化和潜在问题（如 shared memory 溢出）。
- **GPT 代码中的内存管理重构**：Eriks.0595 正在重构内存管理以实现更好的整合和效率，将模型 parameters、gradients 和 optimizer states 的分配移至集中位置。
- **FP8 与量化策略**：Arund42 和其他人讨论了使用 FP8 和 INT8 量化训练模型，强调了量化感知训练（quantization-aware training）的挑战和潜在收益。
   - 与其他方法（如原生 INT8 训练）进行了对比，并提到了 Character.AI 和 Tesla 在推理优化方面的实现。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://research.character.ai/optimizing-inference/">Optimizing AI Inference at Character.AI</a>：在 Character.AI，我们正致力于实现 AGI。在未来的状态下，大语言模型 (LLMs) 将增强日常生活，提供业务生产力和娱乐，并帮助人们...</li><li><a href="https://mistral.ai/news/mistral-nemo/">Mistral NeMo</a>：Mistral NeMo：我们最新的最佳小型模型。一个具有 128k 上下文长度的最先进 12B 模型，与 NVIDIA 合作构建，并根据 Apache 2.0 许可证发布。</li><li><a href="https://pytorch.org/blog/accelerating-neural-network-training/">Accelerating Neural Network Training with Semi-Structured (2:4) Sparsity</a>：在过去的一年里，我们在 PyTorch 中增加了对半结构化 (2:4) 稀疏性的支持。只需几行代码，我们就能展示出在 segment-anything 上 10% 的端到端推理加速...</li><li><a href="https://github.com/karpathy/llm.c/pull/696">Major FP32 llm.c improvements/refactoring/etc. by ademeure · Pull Request #696 · karpathy/llm.c</a>：我有点做过头了，这最终显著改变了 train_gpt2_fp32.cu 中的几乎每一个 kernel！我还给 kernels 添加了很多注释——可能太多了，但如果...
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1263246495432577086)** (9 条消息🔥): 

> - `GPU 操作中的深拷贝 (Deep Copy)`
> - `Kernel 参数限制`
> - `CUDA 中的指针处理`
> - `量化 (Quantization) 与 Group Size` 


- **对 GPU 操作中深拷贝的困惑**：一位用户对在 GPU 操作期间（特别是对于 Tensor 列表）将数据从 Host 深拷贝到 Device 的必要性表示困惑。
   - 其他成员澄清说，数据访问问题源于大型结构体无法直接传递给 Kernel，因此需要指向大型内存缓冲区的指针。
- **Kernel 参数限制和指针处理**：成员们讨论了 CUDA 中 4k Kernel 参数限制的问题，指出使用指针从 GPU 内存访问大型数据结构的必要性。
   - 最初在 CPU 内存中的指针需要拷贝到 GPU 全局内存，将这些指针传递给 Kernel 可以缓解大小限制。
- **CUDA 内存分配中 ** 与 * 的区别**：解决了一位用户关于 CUDA 中使用 `**` 与 `*` 的困惑，强调 GPU Kernel 中的指针参数必须指向 GPU 内存。
   - “将指针的 CPU 内存数组传递给 GPU 内存”是用于澄清这一点的一个例子。
- **量化讲座中的 Perplexity 和 Group Size 困惑**：一位用户提出了关于量化讲座中提到的与 Perplexity 变化相关的 “Group Size” 一词的问题。
   - 提供的解释详细说明了 Group Size 是指共享一个 Scaling Factor 的量化值的数量，这会影响内存占用和量化误差。


  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1263210519997845605)** (213 条消息🔥🔥): 

> - `Hermes 2`
> - `Mistral struggles`
> - `Model Merging`
> - `Open Empathic` 


- **确认 RTX 4060 Ti 的适用性及配置建议**：在询问关于在 RTX 4060 Ti 上使用 Automatic1111 后，另一位用户确认该硬件足够，并建议编辑 `webui-user.bat` 文件，添加 `--xformers --medvram-sdxl` 以获得最佳性能。
   - 讨论表明社区非常关注硬件性能和最大化效用的配置技巧。
- **Adobe Stock 修订关于艺术家名字的政策**：Adobe Stock 更新了其内容政策，根据其最近的通知，将移除在标题、关键词或生成式 AI 内容的 Prompt 中引用艺术家名字的项目。
   - 用户担心这一政策的广泛应用会影响非版权引用，反映了 Adobe 的严厉立场。
- **AI 的困境：突出的常见失败**：社区成员讨论了 AI 面临的反复出现的问题，例如渲染手部、文本以及躺在草地上的女性，一位用户开玩笑地总结为“手、文本、躺在草地上的女性”。
   - 大家期待有一个 Finetuned 模型来解决这些特定的缺点，因为这些错误是常见的痛点。
- **Ultra 功能推测和 Beta 访问权限澄清**：成员们推测了 “Ultra” 功能的技术细节，建议它可能涉及第二个采样阶段，并可能使用 Latent Upscale 或 Noise Injection。
   - 澄清了 Ultra 处于 Beta 阶段，可通过网站和 API 访问，确认它不直接与货币化挂钩，而是持续开发的一部分。
- **关于 Troll 管理技术的辩论**：关于 IP 封禁管理干扰用户的有效性展开了激烈辩论，一位成员断言其无效，而另一位成员则强调了更广泛的影响和替代方案。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/dies-from-cringe-meme-cringe-imagine-gif-23477312">Dies From Cringe Meme GIF - Dies From Cringe Meme Cringe - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/spaces/multimodalart/AuraFlow">Auraflow Demo - Hugging Face Space，由 multimodalart 提供</a>：未找到描述</li><li><a href="https://civitai.com/articles/4248```">什么是 score_9 以及如何在 Pony Diffusion 中使用它 | Civitai</a>：对下一版 Pony Diffusion 感兴趣？在此阅读更新：https://civitai.com/articles/5069/towards-pony-diffusion-v7 你可能已经见过 score_9...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1263301231682523252)** (1 条消息): 

> - `GoldFinch 混合模型`
> - `Linear Attention 对比 Transformers`
> - `GoldFinch 性能基准测试`
> - `Finch-C2 和 GPTAlpha 发布` 


- **GoldFinch 混合模型问世**：**GoldFinch** 模型将 Linear Attention (RWKV) 与传统的 Transformers 相结合，消除了二次方减速并显著减小了 KV-Cache 大小，从而在极低的 VRAM 需求下实现了极长的上下文长度。
   - 在实验中，GoldFinch 在下游任务上的表现优于稍大体量的模型，如 **1.5B 级别的 Llama** 和 **Finch (RWKV-6)**。
- **GoldFinch 性能超越竞争对手**：**GoldFinch** 在下游任务中展现出比 **1.5B 级别 Llama** 和 **Finch (RWKV-6)** 模型更好的性能。
   - 强调了其能够以更低的成本回看每个 token 的能力以及更优的下游结果。
- **Finch-C2 和 GPTAlpha 发布**：**Finch-C2** 作为 **Finch (RWKV-6)** 的高性能版本推出，而 **GPTAlpha** 则通过 RWKV 组件增强了传统的 Transformer 架构。
   - 这两个模型都利用 softmax attention 实现了超越标准 Transformers 的性能。
- **GoldFinch 参考链接**：**GoldFinch** 的代码、论文和 checkpoints 已在 [GitHub](https://github.com/recursal/GoldFinch-paper) 和 [Hugging Face](https://huggingface.co/recursal/GoldFinch-paper) 仓库发布。
   - 介绍 GoldFinch 模型的论文：[PDF](https://arxiv.org/abs/2407.12077), [HTML](https://arxiv.org/html/2407.12077v1)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2407.12077">GoldFinch: High Performance RWKV/Transformer Hybrid with Linear Pre-Fill and Extreme KV-Cache Compression</a>：我们介绍了 GoldFinch，这是一种混合 Linear Attention/Transformer 序列模型，它使用一种新技术，在线性时间和空间内高效生成高度压缩且可重复使用的 KV-Cache...</li><li><a href="https://github.com/recursal/GoldFinch-paper">GitHub - recursal/GoldFinch-paper: GoldFinch and other hybrid transformer components</a>：GoldFinch 及其他混合 transformer 组件。通过在 GitHub 上创建账号为 recursal/GoldFinch-paper 的开发做出贡献。</li><li><a href="https://huggingface.co/recursal/GoldFinch-paper">recursal/GoldFinch-paper · Hugging Face</a>：未找到描述。
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1263248892200484946)** (72 条消息🔥🔥): 

> - `关于 AI 抓取的争议`
> - `对 Whisper MIT 许可证的误解`
> - `Google 抓取与内容使用`
> - `提到 Pile 的随机页面`
> - `社区项目参与` 


- **AI 抓取 YouTube 字幕引发争议**：关于 AI 训练数据（特别是 YouTube 字幕）的讨论异常激烈，成员指出，相比于对图像和音频的担忧，这种愤怒似乎放错了地方。
   - 一些人认为这在经济上损害了创作者，而另一些人则认为不值得大惊小怪，并指出在 fair use（合理使用）和公共许可下进行抓取的合法性。
- **澄清对 Whisper MIT 许可证的误解**：清除了关于 Whisper 是 MIT 项目的误解，确认它是采用 MIT 许可证的软件。
   - 强调了 MIT 许可证与 MIT 项目之间的区别：*"Whisper 不是由 MIT 开发的……但 Whisper 采用的是 MIT 许可证。"*
- **Google 在未获用户同意的情况下获利？**：成员们就 Google 和 Bing 抓取数据以及由此产生的经济利益（未直接向内容创作者支付费用）展开辩论。
   - *"对于版权而言，重要的是搜索结果并不会损害经济价值……"*
- **关于抓取行为的普遍误传**：讨论集中在关于 EleutherAI 及其抓取行为的误传上，指出他们不是一家公司，而是一个非营利研究机构。
   - 一些成员戏称，如今记者的误报可能已经泛滥成灾。
- **寻求活跃的社区项目**：关于活跃社区项目的咨询被引导至不同的频道和组织。
   - 建议探索服务器内的其他频道，以了解 LLM 和 NLP 领域的进行中项目。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1263216620621266944)** (108 条消息🔥🔥): 

> - `ICML 2024`
> - `注意力机制 (Attention Mechanisms)`
> - `蛋白质语言模型 (Protein Language Models)`
> - `Patch 级别训练`
> - `语言模型缩放 (Language Model Scaling)`

- **ICML 2024 热度攀升**：与会者讨论了即将举行的 ICML 2024，庆祝 **grimsqueaker** 在 **ML4LMS workshop** 上关于“Protein Language Models Expose Viral Mimicry and Immune Escape”的演讲。
   - Grimsqueaker 详细阐述了该海报内容，强调了 99.7% 的 ROCAUC 准确率以及对病毒模拟的新见解，并提供了 [GitHub 上的代码](https://github.com/ddofer/ProteinHumVir) 和 [海报链接](https://openreview.net/attachment?id=gGnJBLssbb&name=poster)。
- **Attention 机制引发辩论**：成员们就 **LSH Attention** 中哈希函数对向量缩放的不变性进行了技术讨论。
   - Vodros 和 gloomyc 讨论了归一化对 Attention 矩阵的影响，研究了多轮哈希和分桶划分的有效性。
- **Patch-Level Training 革新 LLM 效率**：新引入的 LLM **Patch-Level Training** 声称通过 Token 压缩减少序列长度，从而提高训练效率。
   - 包括 cz_spoon_06890 在内的成员探讨了 Cosine LR 调度、Patch 与 Token 级别训练的影响，以及从 [PatchTrain GitHub](https://github.com/shaochenze/PatchTrain) 中可能学到的经验。
- **语言模型的 Scaling 技术**：成员们讨论了 Scaling 和训练效率技术，包括 Multi-token 预测和学习率调度。
   - Catboy_slim_ 等人深入研究了不同 Scaling 方法的影响，并引用了来自 [arxiv.org](https://arxiv.org/abs/2405.18392) 等的相关论文。
- **NuminaMath-7B 的崛起与审查**：Jeniajitsev 批评 **NuminaMath-7B** 在高中数学解题 Benchmark 中夸大其词，揭示了其根本性的推理缺陷。
   - 该模型在简单问题上的表现提示在解读 Benchmark 结果时需谨慎，正如 [Twitter 线程](https://x.com/JJitsev/status/1813930981637902486) 中所讨论的那样。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2407.12665">Patch-Level Training for Large Language Models</a>: 随着 Large Language Models (LLMs) 在语言理解和生成方面取得显著进展，其训练效率已成为一个关键关注点。传统上，LLMs 被训练用于预测...</li><li><a href="https://arxiv.org/abs/2402.04362">Neural Networks Learn Statistics of Increasing Complexity</a>: 分布简单性偏差 (DSB) 假设神经网络首先学习数据分布的低阶矩，然后再转向高阶相关性。在这项工作中，我们展示了...</li><li><a href="https://arxiv.org/abs/2405.18392">Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations</a>: 规模已成为获得强大机器学习模型的关键要素。因此，了解模型的 Scaling 属性是有效设计正确训练设置的关键...</li><li><a href="https://arxiv.org/abs/2309.02427">Cognitive Architectures for Language Agents</a>: 最近的研究通过外部资源（如互联网）或内部控制流（如 Prompt Chaining）增强了 Large Language Models (LLMs)，以处理需要落地或推理的任务，从而...</li><li><a href="https://arxiv.org/abs/2404.19737">Better &amp; Faster Large Language Models via Multi-token Prediction</a>: 诸如 GPT 和 Llama 等 Large Language Models 是使用 Next-token prediction 损失进行训练的。在这项工作中，我们建议训练语言模型一次预测多个未来 Token 会导致...</li><li><a href="https://x.com/JJitsev/status/1813930981637902486">Tweet from Jenia Jitsev 🏳️‍🌈 🇺🇦 (@JJitsev)</a>: 另一个关于兴衰的故事：最近，NuminaMath-7B 在 AIMO 竞赛中排名第一，解决了 29/50 个奥数级别的私有集问题。它能处理简单的 AIW 问题吗，这需要...</li><li><a href="https://github.com/shaochenze/PatchTrain">GitHub - shaochenze/PatchTrain: Code for paper &quot;Patch-Level Training for Large Language Models&quot;</a>: 论文 &quot;Patch-Level Training for Large Language Models&quot; 的代码 - shaochenze/PatchTrain</li><li><a href="https://github.com/RulinShao/retrieval-scaling">GitHub - RulinShao/retrieval-scaling: Official repository for &quot;Scaling Retrieval-Based Langauge Models with a Trillion-Token Datastore&quot;.</a>: &quot;Scaling Retrieval-Based Langauge Models with a Trillion-Token Datastore&quot; 的官方仓库 - RulinShao/retrieval-scaling</li><li><a href="https://openreview.net/forum?id=gGnJBLssbb&noteId=gGnJBLssbb">Protein language models expose viral mimicry and immune escape</a>: 病毒通过分子拟态规避免疫系统，采用其宿主的生物物理特性。我们调整了蛋白质语言模型 (PLMs) 来区分人类和病毒...</li><li><a href="https://github.com/ddofer/ProteinHumVir">GitHub - ddofer/ProteinHumVir: Code &amp; data for &quot;Protein Language Models Expose Viral Mimicry and Immune Escape&quot;</a>: &quot;Protein Language Models Expose Viral Mimicry and Immune Escape&quot; 的代码和数据 - ddofer/ProteinHumVir</li><li><a href="https://doi.org/10.1101/2024.03.14.585057">Protein Language Models Expose Viral Mimicry and Immune Escape</a>: 动机：病毒通过分子拟态规避免疫系统，采用其宿主的生物物理特性。我们调整了蛋白质语言模型 (PLMs) 来区分人类和病毒...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1263598332773797889)** (1 messages): 

> - `tokenization-free models`
> - `interpretability in AI` 


- **Tokenization-Free Models for Better Interpretability?**: 一位成员提出了一个问题：**Tokenization-free 语言模型**对于**可解释性**（Interpretability）是更好还是更坏。
- **Theory on Interpretability in AI**: 讨论集中在 Tokenization 对 AI 模型可解释性的潜在影响。


  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1263227270403981323)** (14 messages🔥): 

> - `lm-eval-harness` `--predict_only` 标志
> - 使用 LoRA 进行 TRL 微调
> - PeftModelForCausalLM 中的嵌入矩阵问题
> - Gigachat 模型 PR 评审
> - `simple_evaluate` 响应存储


- **使用 lm-eval-harness `--predict_only` 的挑战**：一位用户询问在获得补全文件并使用 `--predict_only` 标志后，如何使用 `lm-eval-harness` 运行指标计算。
- **使用 LoRA 进行 TRL 微调时报错**：尽管使用了最新版本的库，一位用户在使用 `trl` 的 `setup_chat_format` 并配合 LoRA 进行微调时遇到了 `RuntimeError: size mismatch`。
- **lm-eval-harness 中的系统消息处理**：社区澄清，通过 `--system_instruction` 传递的系统消息的处理方式与 task.yaml 文件中 `description` 字段类似，会针对兼容模型进行拼接。
- **Gigachat 模型 PR 评审请求**：一位用户请求评审其[将 Gigachat 模型添加到库中的 PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/1996)，该 PR 使用了带有聊天模板的 API。
- **评估 LM 评测分数的一致性**：一位用户讨论了在添加新模型时，通过比较不同 LM 评测实现和框架之间的确定性生成采样分数，来确保正确性。



**提及链接**：<a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1996">Add Gigachat model by seldereyy · Pull Request #1996 · EleutherAI/lm-evaluation-harness</a>：通过使用带有聊天模板的 API 向库中添加新模型。授权需为你的 API auth_data 设置环境变量 "GIGACHAT_CREDENTIALS" 和 "GIGACHAT_SCOPE"...

  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1263210402821574676)** (59 messages🔥🔥): 

> - `Codestral Mamba` 在 LM Studio 上的支持
> - LM Studio 中的上下文长度问题
> - NSFW/角色扮演的模型建议
> - Gemma IT GPU 问题
> - Mistral 与 NVIDIA 合作推出 Mistral-Nemo 12B


- **Codestral Mamba 与 LM Studio 的集成**：成员们讨论了在 LM Studio 支持 Codestral Mamba 之前，需要先在 llama.cpp 中添加支持。
   - *Heyitsyorkie* 提到更新取决于 llama.cpp 的集成以及随后在 LM Studio 中的采用。
- **修复 LM Studio 中的上下文长度溢出**：成员们帮助 *Gilwolfy* 解决了上下文长度溢出问题，引导其在工具菜单中调整设置。
   - *Santonero* 澄清了在哪里查找和更改“上下文溢出策略”（Context Overflow Policy）设置。
- **NSFW 内容的模型推荐**：成员 *Skryptii* 建议，模型选择和系统提示词对 NSFW 和角色扮演任务的影响比预设更大。
   - 推荐了如 [Smegmma-9B](https://huggingface.co/TheDrummer/Smegmma-9B-v1) 等为此类用途微调的模型。
- **Gemma IT GPU 错误问题**：*Anadon* 报告在拥有 16GB VRAM 的 RX 6900 XT 上运行 2B 参数模型时出现 GPU 错误。
   - 错误随加载到 GPU 中的层数而变化。
- **Mistral 和 NVIDIA 发布 Mistral-Nemo 12B**：Mistral 与 NVIDIA 合作发布了 Mistral-Nemo 12B 模型，拥有高达 128k 的 Token 上下文窗口。
   - 公告指出其具有卓越的推理和编码准确性，但由于 llama.cpp 中的分词器（tokenizer）问题，目前 LM Studio 尚不支持。


<div class="linksMentioned">

<strong>提及链接</strong>：

<ul>
<li>
<a href="https://mistral.ai/news/mistral-nemo/">Mistral NeMo</a>：Mistral NeMo：我们最新的最佳小型模型。一个拥有 128k 上下文长度的最先进 12B 模型，与 NVIDIA 合作构建，并以 Apache 2.0 许可证发布。</li><li><a href="https://huggingface.co/TheDrummer/Smegmma-9B-v1">TheDrummer/Smegmma-9B-v1 · Hugging Face</a>：未找到描述</li><li><a href="https://lmstudio.ai/docs/local-server#supported-payload-parameters">Local LLM Server | LM Studio</a>：你可以通过在 localhost 上运行的 API 服务器，使用你在 LM Studio 中加载的 LLM。</li><li><a href="https://lmstudio.ai/docs/lmstudio-sdk/examples">Code Examples | LM Studio</a>：如何使用 LM Studio JavaScript/TypeScript SDK 的示例
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1263483251448872960)** (23 条消息🔥): 

> - `DeepSeek-V2 集成`
> - `Mistral NeMo 模型发布`
> - `中国对开源 LLM 的使用`
> - `LLM 中的逻辑推理`
> - `LLM 中的冗长回复` 


- **LM Studio 与 DeepSeek-V2 的集成待定**：用户讨论了 LM Studio 是否将支持 [DeepSeek-V2-Chat-0628](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat-0628)，预计当 llama.cpp 支持时，LM Studio 也会同步支持。
   - *Anester* 强调了中国的数据政策，并暗示他们可能会利用庞大的数据集在 LLM 开发方面超越其他国家。
- **Mistral 与 NVIDIA 合作发布 Mistral NeMo 模型**：Mistral 发布了 [Mistral NeMo](https://mistral.ai/news/mistral-nemo/)，这是一个与 **NVIDIA** 共同开发的 12B 模型，提供 128k token 的上下文窗口，并在推理、世界知识和代码准确性方面达到了最先进水平。
   - 该模型支持无损性能的 FP8 推理，预训练权重已根据 Apache 2.0 许可证发布。
- **中国关于数据和 LLM 的政策**：*Anester* 认为，由于广泛的数据收集和缺乏 DRM 法律，中国的数据政策最终可能使其在 LLM 能力上超越他国。
   - 他提出了一个颇具争议的观点，即中国有可能利用这些手段接管像 ChatGPT 这样的技术。
- **Mistral NeMo 的逻辑推理问题**：用户注意到，尽管技术参数很高，但 Mistral NeMo 在某些测试中的逻辑推理表现似乎不佳。
   - 一个例子展示了该模型对一个涉及球和高脚杯的简单场景给出了晦涩难懂的解释。
- **语言模型中的冗长回复**：*Ptable* 提到 Mistral NeMo 的回复非常冗长，这可以被视为利用了其 128k token 的上下文窗口。
   - 这种冗长可能会影响用户与模型的交互，尤其是在长对话中。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://mistral.ai/news/mistral-nemo/">Mistral NeMo</a>：Mistral NeMo：我们最新的最佳小型模型。一个拥有 128k 上下文长度的 12B 最先进模型，与 NVIDIA 合作构建，并根据 Apache 2.0 许可证发布。</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat-0628">deepseek-ai/DeepSeek-V2-Chat-0628 · Hugging Face</a>：未找到描述</li><li><a href="https://build.nvidia.com/nv-mistralai/mistral-nemo-12b-instruct">NVIDIA NIM | mistral-nemo-12b-instruct </a>：立即体验用于构建企业级生成式 AI 应用的领先模型。
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/)** (1 条消息): 

xoxo3331: 目前没有通过 CLI 使用预设（preset）加载模型的参数或标志。
  

---


### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1263534463711711287)** (1 条消息): 

> - `Meta Llama 3`
> - `提示词策略`
> - `股票交易策略`
> - `资金分配`
> - `风险管理` 


- **编写 Meta Llama 3 交易提示词**：一位成员分享了他们编写提示词的方法，用于分析股市机会并使用 **Meta Llama 3** 提出交易策略。
   - 该提示词包括评估交易策略风险与回报、资金分配建议以及在指定风险承受范围内进行管理的步骤。
- **用于交易风险管理的结构化提示词**：分享的提示词概述了在约定的承受范围内管理风险的方法，确保在执行交易前进行全面分析。
   - *你的最终回答必须包含对交易提案的详细分析*，进一步强调了细致分解的要求。


  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1263559049560064051)** (1 条消息): 

> - `新模型讨论`
> - `用于 Autogen 的 LM Studio 预设`
> - `Llama-3-Groq-8B-Tool-Use-GGUF` 


- **关于 Llama-3-Groq-8B 的新模型讨论**：有人提出了关于新模型 [Llama-3-Groq-8B-Tool-Use-GGUF](https://discord.com/channels/1110598183144399058/1225909444727013466/1263525794945175563) 及其与 Autogen 场景下默认 LM Studio 预设兼容性的查询。
   - 该模型与 [MaziyarPanahi](https://discord.com/channels/1110598183144399058/1225909444727013466/1263525794945175563) 相关，人们对其在自动生成场景中的应用感到好奇。
- **用于 Autogen 场景的 LM Studio 预设**：讨论包括默认的 LM Studio 预设是否能有效地配合新模型在 Autogen 场景下工作。
   - 成员们正在寻求关于预设兼容性的澄清，以便在各种场景中高效使用。


  

---

### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1263214903347515484)** (23 messages🔥): 

> - `Custom Hardware Specs` (自定义硬件规格)
> - `Resizable BAR Impact on LLM` (Resizable BAR 对 LLM 的影响)
> - `NVIDIA GTX 1050 Issues` (NVIDIA GTX 1050 问题)
> - `ROCM Version Update` (ROCM 版本更新)
> - `DIY Safety Concerns` (DIY 安全顾虑)


- **寄予厚望的 Xeon 配置**：**2x Xeon 8 core 3.5GHz**、**32GB 2600 ECC RAM** 和 **P40** GPU 的配置旨在实现快速的模型性能，并留有扩展空间。
   - 构建者对缓慢运行大型模型不感兴趣，正寻求优化配置以提升速度并增加额外功能。
- **Resizable BAR 对 LLM 并非至关重要**：对话中提到，Resizable BAR 对 **LLM performance** 没有显著影响，重点应放在 **tensor cores** 和 **VRAM bandwidth** 上。
   - *Model loading* 和 **multi-GPU performance** 受到质疑，但目前尚无定论。
- **GTX 1050 在运行 LLM 时表现吃力**：有报告称 **GTX 1050** 存在问题，其 GPU 使用率最高仅为 10%，而 CPU 承担了大部分工作负载。
   - 成员们推测 **4GB VRAM** 不足以运行 7B+ 模型，采用更小的模型可能是可行的解决方案。
- **ROCM 版本更新通知**：成员们询问了最新版本的 **LM Studio ROCM**，最初认为是 0.2.24。
   - 随后通过指向相关频道的快速指引澄清了更新信息。
- **DIY 配置的散热安全**：有人担心 **Xeon processors** 在 DIY 机箱中可能会烧焦木材。
   - 构建者保证已留有 **air gap**（空气间隙）以降低此类风险。


  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1263296398909112471)** (3 messages): 

> - `0.3.0 Beta Enrollment` (0.3.0 Beta 报名)
> - `Beta Download` (Beta 下载)
> - `Beta Announcements` (Beta 公告)


- **0.3.0 Beta 状态困惑**：包括 krypt_lynx 在内的多位成员提到，他们获得了“0.3.0 Beta”状态，但没有收到进一步的回复或下载信息。
   - skeletonbow 假设这可能是一个报名制 Beta，邀请会逐步发放以收集反馈并进行迭代；而 heyitsyorkie 则认为目前的 Beta 访问权限可能仅限于“活跃”的聊天参与者。
- **等待 Beta 报名回复**：krypt_lynx 询问在获得“0.3.0 Beta”状态后，如果没有进一步消息，该去哪里寻找下载链接。
   - 该问题尚未解决，但 heyitsyorkie 推测 Yags 可能会在几天内发布公告。


  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1263543622712823848)** (4 messages): 

> - `CUDA on AMD`
> - `zluda`
> - `scale`
> - `portable install option` (便携安装选项)


- **在 AMD RDNA 上测试 CUDA**：一位成员分享了一篇关于 AMD 显卡新编译器的 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1e6cxef/cuda_on_amd_rdna_3_and_rdna_2_new_release/)，指出 RX 7800 可以配合使用，并暗示其可能比 ROCm 实现更有优势。
   - 他们提到有兴趣在 **lama.cpp** 中对其进行测试，看看其性能是否优于 ROCm 实现。
- **ZLUDA 及其局限性**：另一位成员提到了 **ZLUDA**，它允许 CUDA 在 AMD 上原生运行，并指出它从未集成到 **lama.cpp** 中。
- **SCALE 与 ZLUDA 类似**：一位成员指出，**SCALE**（一个类似于 ZLUDA 的工具）已于几天前发布。
- **请求便携安装选项**：一位用户表示有兴趣测试是否存在便携安装选项。



**提到的链接**：<a href="https://www.reddit.com/r/LocalLLaMA/comments/1e6cxef/cuda_on_amd_rdna_3_and_rdna_2_new_release/">Reddit - Dive into anything</a>：未找到描述

  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1263525794945175563)** (1 messages): 

> - `Groq's tool use models` (Groq 的工具使用模型)
> - `Berkeley Function Calling Leaderboard`
> - `Llama-3 Groq-8B`
> - `Llama-3 Groq-70B`
> - `tool use and function calling` (工具使用和函数调用)


- **Groq 模型在函数调用排行榜上获得高分**：Groq 新的工具使用模型在 [Berkeley Function Calling Leaderboard](https://huggingface.co/lmstudio-community/Llama-3-Groq-8B-Tool-Use-GGUF) 上取得了高分，**8B** 模型得分为 **89.06%**，**70B** 模型得分为 **90.76%**。
   - 这些模型非常适合依赖 **tool use** 和 **function calling** 的流水线。
- **Llama-3 Groq 模型发布**：**Llama-3 Groq-8B** 和 **Llama-3 Groq-70B** 模型现已可用，并针对工具使用和函数调用进行了优化。
   - 可以在 [Hugging Face](https://huggingface.co/lmstudio-community/Llama-3-Groq-70B-Tool-Use-GGUF) 上查看这些模型，以便集成到各种流水线中。


  

---

### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1263233006932590602)** (14 messages🔥): 

> - `在线托管 AI 模型`
> - `用于托管的 Ngrok 与 Nginx 对比`
> - `自定义 Web UI 和 SSR 技术`
> - `用于安全隧道的 Tailscale`
> - `构建用户账户和独立聊天` 


- **为朋友测试设置 AI 模型**：一位用户询问如何在自己的 PC 上托管 AI 模型，以便朋友可以通过**多会话**在线访问。
   - 建议包括使用 [Ngrok](https://ngrok.com/) 进行简单的临时 URL 托管，以及使用 **NGINX** 以获得更好的控制；还提到了使用 **Tailscale** 进行安全隧道传输。
- **长期 AI 模型托管计划**：该用户分享了为成百上千名用户提供具有独立账户和聊天功能的 AI 模型托管愿景。
   - 讨论了用于管理用户交互的自定义前端和后端计划，**强调了对经验丰富的前端开发人员的需求**。
- **用于用户交互的 Web UI 和 SSR 技术**：建议使用服务器端渲染 (SSR) 技术创建**自定义 Web UI**，以促进用户交互。
   - 除非用户习惯于直接调用 API，否则推荐使用此方法。


  

---



### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1263257584836804701)** (2 messages): 

> - `TextGrad`
> - `ProTeGi`
> - `STORM 写作系统` 


- **TextGrad 框架为神经网络优化提供文本梯度**：一位成员想知道 [TextGrad](https://arxiv.org/abs/2406.07496)（一个通过文本执行自动“微分”的框架）是具有实用性还是仅仅是炒作。
   - 该框架促进 LLM 提供文本反馈，以优化代码片段和分子结构等组件，旨在使优化更加易于实现。
- **ProTeGi 论文引发对文本梯度的兴趣**：讨论中提到了另一篇论文 [ProTeGi](https://arxiv.org/abs/2305.03495)，因为它在利用文本梯度优化神经网络方面采用了类似的方法。
- **STORM 系统增强了长篇文章的大纲创建**：斯坦福大学的 [STORM](https://github.com/stanford-oval/storm) 系统使用 LLM，通过多视角提问和检索综合主题大纲，从而撰写有条理的长篇文章。
   - 与基准方法相比，STORM 在生成文章的组织性和覆盖广度方面表现出显著提升，尽管仍存在源偏见转移等新挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.07496">TextGrad: Automatic &#34;Differentiation&#34; via Text</a>: AI 正在经历范式转移，通过编排多个大语言模型 (LLM) 和其他复杂组件的系统实现了突破。因此，开发原则性且自动化的...</li><li><a href="https://arxiv.org/abs/2402.14207">Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models</a>: 我们研究如何应用大语言模型从零开始撰写有据可查且有条理的长篇文章，其广度和深度可与维基百科页面相媲美。这个尚未被充分探索的问题提出了新的...</li><li><a href="https://storm.genie.stanford.edu/article/ai-human-relations-and-the-complexity-it-introduces-to-society-18731">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/stanford-oval/storm">GitHub - stanford-oval/storm: An LLM-powered knowledge curation system that researches a topic and generates a full-length report with citations.</a>: 一个由 LLM 驱动的知识固化系统，可研究主题并生成带有引用的全文报告。 - stanford-oval/storm
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[datasets](https://discord.com/channels/1053877538025386074/1105324249721356298/1263524936316747807)** (1 条消息): 

> - `Synthetic dataset`
> - `General knowledge base` 


- **合成数据集与商业导向知识库发布**：分享了一个**[合成数据集和通用知识库](https://github.com/Mill-Pond-Research/AI-Knowledge-Base)**，重点关注商业应用。
   - 这一全面的资源旨在通过广泛的商业相关数据为 AI 系统提供支持。
- **用于 RAG 系统的 AI 知识库**：**Mill-Pond-Research/AI-Knowledge-Base** 提供了一个专为检索增强生成 (RAG) 系统定制的全面通用知识库。
   - 该仓库包含详细的文档和一张展示其强大能力的数据集图像。



**提到的链接**：<a href="https://github.com/Mill-Pond-Research/AI-Knowledge-Base">GitHub - Mill-Pond-Research/AI-Knowledge-Base: Comprehensive Generalized Knowledge Base for AI Systems (RAG)</a>：用于 AI 系统 (RAG) 的全面通用知识库 - Mill-Pond-Research/AI-Knowledge-Base

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1263208719383134330)** (3 条消息): 

> - `Intelligent Digital Agents`
> - `Mistral-NeMo-12B-Instruct`
> - `AgentInstruct for Synthetic Data` 


- **智能数字 Agent 需要转变**：[“Intelligent Digital Agents in the Era of Large Language Models”](https://x.com/ManifoldRG/status/1811120196570206459) 讨论了 LLM 驱动的 Agent 的进展，指出了局限性，并建议从基于语言的处理转向增强推理。
   - 该立场论文强调了在 Agent 设计中采用新方法以提高性能的必要性。
- **NVIDIA 的 Mistral-NeMo-12B-Instruct 表现出色**：由 NVIDIA 和 Mistral AI 开发的 [Mistral-NeMo-12B-Instruct](https://huggingface.co/nvidia/Mistral-NeMo-12B-Instruct) 模型拥有 12B 参数，性能优于同等规模的模型，支持 128k 上下文窗口和 FP8 量化。
   - 该模型经过多语言和代码数据训练，采用 Apache 2 许可证发布，包含预训练版本和 Instruct 版本。
- **AgentInstruct 自动化合成数据创建**：微软研究院的 [AgentInstruct](https://x.com/MSFTResearch/status/1813974519469515087) 框架旨在通过自动化的多 Agent 系统简化合成数据创建，从而增强语言模型的后期训练。
   - Arindam Mitra 及其合著者的研究概述了这一计划，有望彻底改变大规模数据生成。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/MSFTResearch/status/1813974519469515087">来自 Microsoft Research (@MSFTResearch) 的推文</a>：合成数据创建很困难。Arindam Mitra 及其合著者旨在通过 AgentInstruct 改变这一现状，这是一个自动化的多 Agent 框架，用于为语言模型大规模生成高质量合成数据...</li><li><a href="https://huggingface.co/nvidia/Mistral-NeMo-12B-Instruct">nvidia/Mistral-NeMo-12B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/ManifoldRG/status/1811120196570206459">来自 Manifold Research (@ManifoldRG) 的推文</a>：🚨我们很高兴分享《大语言模型时代的智能数字 Agent》，这是一篇立场论文，探讨了 LLM 驱动的 Agent 的进展，识别了局限性，并建议...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1263208757719203880)** (115 messages🔥🔥): 

> - `Twitter/X 模型直播`
> - `关于 LLM Jailbreaking 的新论文`
> - `Mistral NeMo 模型发布`
> - `AutoFP8 与 FP8 量化`
> - `GPT-4o Mini 基准测试性能` 


- **新论文揭示通过过去式重构实现 LLM Jailbreaking**：一篇新论文揭示了 LLM 中一个奇特的泛化差距，即通过将有害请求重构为过去式，在进行 20 次重构尝试后，GPT-4 的 Jailbreak 成功率从 1% 显著提升至 88% ([链接](https://arxiv.org/abs/2407.11969))。
   - “我们的研究结果强调，广泛使用的对齐技术如 **SFT**、**RLHF** 和 **Adversarial Training** 可能非常脆弱，无法按预期实现泛化。”
- **Mistral NeMo 模型发布引发讨论**：Mistral AI 与 NVIDIA 合作发布了 12B 模型 **Mistral NeMo**，拥有高达 128k 的 Token 上下文窗口，并支持 FP8 量化以提升性能 ([链接](https://mistral.ai/news/mistral-nemo/))。
   - 一些用户对 **RTX 3090** 持有者被排除在外表示担忧，因为 FP8 量化需要像 **4090** 这样更新的 GPU。
- **AutoFP8 在 vLLM 中实现 FP8 量化**：来自 Neural Magic 的 **AutoFP8** 库支持模型的 FP8 权重和激活量化，与 FP16 相比保留了 98-99% 的质量 ([链接](http://github.com/neuralmagic/autofp8))。
   - **vLLM** 已添加对 FP8 的实验性支持，这可以显著降低模型显存占用。
- **GPT-4o Mini 表现不及预期**：尽管 **HumanEval** 评分很高，但 GPT-4o Mini 在实际编程基准测试中的表现与 GPT-3.5-Turbo 相似，令许多用户感到失望 ([链接](https://aider.chat/docs/leaderboards/))。
   - 用户对 OpenAI 的模型性能主张和炒作表示挫败和困惑。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://mistral.ai/news/mistral-nemo/">Mistral NeMo</a>: Mistral NeMo：我们最新的最强小模型。一个具有 128k 上下文长度的 State-of-the-art 12B 模型，与 NVIDIA 合作构建，并以 Apache 2.0 许可证发布。</li><li><a href="https://x.com/abacaj/status/1813977261818904908">anton (@abacaj) 的推文</a>: Mistral NeMo 报告的数据（考虑到它是 12B 模型对比 Meta Llama 8B）是否有误？出于某种原因，它们让 Llama 3 8B 看起来比实际情况差得多……</li><li><a href="https://docs.vllm.ai/en/latest/quantization/fp8.html">FP8 &#8212; vLLM</a>: 未找到描述</li><li><a href="https://maartengr.github.io/BERTopic/index.html">BERTopic</a>: 未找到描述</li><li><a href="https://x.com/maksym_andr/status/1813608842699079750">Maksym Andriushchenko (@maksym_andr) 的推文</a>: 🚨很高兴分享我们的新论文！🚨 我们揭示了当前拒绝训练方法中一个奇特的泛化差距：只需将有害请求重构为过去式（例如，“如何制作...”</li><li><a href="https://x.com/deepseek_ai/status/1813921111694053644">DeepSeek (@deepseek_ai) 的推文</a>: 🎉激动人心的消息！我们开源了 DeepSeek-V2-0628 Checkpoint，这是 LMSYS Chatbot Arena 排行榜 @lmsysorg 上排名第一的开源模型。详细 Arena 排名：总榜第 11，Hard Prompts 第 3，Co...</li><li><a href="https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407">mistralai/Mistral-Nemo-Instruct-2407 · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/natolambert/status/1814024567192748166">Nathan Lambert (@natolambert) 的推文</a>: GPT4-o-mini 在 Reward Bench 上超过了 Claude 3 Sonnet（不是 3.5）和 Llama 3 70B，低于 Gemma 2 27B。实际上所有这些都很相似。已经非常饱和了。</li><li><a href="http://github.com/neuralmagic/autofp8">GitHub - neuralmagic/AutoFP8</a>: 通过创建一个账户为 neuralmagic/AutoFP8 的开发做出贡献。</li><li><a href="https://x.com/i/broadcasts/1lDGLldQVmvGm">来自 GitHub - FixTweet/FxTwitter 的推文: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能</a>: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能 - FixTweet/FxTwitter</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B/discussions/13">NousResearch/Hermes-2-Pro-Llama-3-8B · 添加工具调用模板</a>: 未找到描述</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B/commit/714ffdffc3cbf97d02f0b484c9676f371830bce3#d2h-846292">上传 3 个文件 · NousResearch/Hermes-2-Pro-Llama-3-8B at 714ffdf</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1263543956776681564)** (6 条消息): 

> - `WorldSim 停机`
> - `WorldSim 问题` 


- **WorldSim 经历停机**：一名成员报告 **WorldSim** 宕机，造成了不便。
   - *“应该一分钟内就能恢复！”* 另一名成员保证道，随后确认修复。
- **WorldSim 问题快速解决**：一名成员迅速报告了 **WorldSim** 的停机问题。
   - 该问题被迅速修复，感谢社区的报告。


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1263226138310738051)** (121 条消息🔥🔥): 

> - `DeepSeek V2`
> - `GPT-5 预测`
> - `GPT-4o Mini 发布`
> - `DeepSeek V2 讨论`
> - `新 LLaMA 3` 


- **DeepSeek 引发 AI 模型价格战**：DeepSeek 的 **DeepSeek V2** 模型将推理成本大幅降低至 **每百万 token 1 元人民币**，引发了中国科技巨头之间的价格战。
   - 被称为中国的 **AI 版拼多多**，其创新包括一种显著减少内存占用的 **新 MLA 架构**。
- **GPT-4o mini 震撼 AI 界**：OpenAI 发布了 **GPT-4o mini**，以 **每百万输入 token 0.15 美元和每百万输出 token 0.60 美元** 的成本提供高性能。
   - 该模型以 **82% 的 MMLU 分数** 和 **128k context window** 超越了 **Claude 3 Haiku** (75%) 等小型模型。
- **Mistral NeMo 与 NVIDIA 共同亮相**：**Mistral AI** 和 **NVIDIA** 推出了 **Mistral NeMo**，这是一个高性能的 12B 模型，具有 **128k tokens context window**，在 NVIDIA DGX Cloud 上训练。
   - 特性包括支持 **FP8 inference** 的量化感知和高效率，该模型是 Mistral 7B 的 **drop-in replacement**。
- **Together AI 推出 Turbo 和 Lite 版 LLaMA 3**：**Together AI** 推出了 LLaMA 3 的 **Turbo** 和 **Lite** 版本，提供 **更快的推理** 和 **更低的成本**，包括 **每百万 token 0.10 美元** 的 **LLaMA-3-8B Lite**。
   - **Turbo 版本** 提供高达 **400 tokens/s** 的速度，使其在处理高需求应用时异常高效。
- **关于 LLaMA 3 400B 发布的猜测升温**：AI 社区正热议 **LLaMA 3 400B** 可能在未来几天内发布，这与 Meta 高管即将进行的演讲相吻合。
   - 有人预测，目前的模型发布旨在这一重大发布前清理战场。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>

<li><a href="https://blogs.nvidia.com/blog/mistral-nvidia-ai-model/">Mistral AI 和 NVIDIA 发布 Mistral NeMo 12B，这是一款尖端企业级 AI 模型</a>：Mistral AI 和 NVIDIA 今天发布了全新的最先进语言模型 Mistral NeMo 12B，开发者可以轻松地为其定制并部署支持聊天机器人、多语言...的企业应用。</li><li><a href="https://x.com/LouisKnightWebb/status/1813996569840238794">Louis Knight-Webb (@LouisKnightWebb) 的推文</a>：gpt-4o mini 的上下文利用率不出所料地比 3.5 好得多，但比“老大哥” 4o 差一些。</li><li><a href="https://x.com/emollick/status/1813753156431384851?s=46">Ethan Mollick (@emollick) 的推文</a>：👀 Claude 处理了一个疯狂的请求：“移除鱿鱼”。“该文档似乎是 Erich Maria Remarque 所著小说《西线无战事》的全文。它并不包含...”</li><li><a href="https://x.com/ArtificialAnlys/status/1813975855468560621">Artificial Analysis (@ArtificialAnlys) 的推文</a>：今天发布的 GPT-4o Mini 以其极低的价格令人印象深刻 👀。其 MMLU 得分为 82%（据 TechCrunch 报道），超越了包括 Gem... 在内的其他小型模型的质量。</li><li><a href="https://x.com/mattshumer_/status/1813958229577302098">Matt Shumer (@mattshumer_) 的推文</a>：Mistral NeMo 看起来是一款非常出色的模型 - 12B 参数，因此微调既快又便宜 - 推理速度快（体积小 + 训练时考虑了量化） - 支持多种语言的高效新分词器...</li><li><a href="https://x.com/nutlope/status/1813996350008422426">Hassan (@nutlope) 的推文</a>：Together AI 的 API 随着两款新版 Llama-3 的推出变得更快更便宜：◆ Llama-3-8B Turbo (FP8) – 高达 400 tokens/s ◆ Llama-3-8B Lite (INT4) – 每百万 tokens 0.10 美元 ◆ Turbo & Lite 版...</li><li><a href="https://x.com/rememberlenny/status/1814004561696465316">Lenny Bogdonoff (@rememberlenny) 的推文</a>：就是这个，但针对的是从事智能工作的劳动时长。而且速度快得多。</li><li><a href="https://x.com/natolambert/status/1813955064949772763?s=46">Nathan Lambert (@natolambert) 的推文</a>：有点惊讶于微小的 Gemini Flash 模型竟然击败了所有这些笨重的开源模型。传闻 Gemini Pro 的激活参数 < 70B，猜测 Gemini Flash 的激活参数 < 30B，甚至可能只有...</li><li><a href="https://www.theverge.com/2024/7/17/24199005/samsung-galaxy-ai-z-fold-6-sketch-to-image">三星全新的图像生成 AI 工具好得有点过头了</a>：照片到底是什么？</li><li><a href="https://x.com/imjaredz/status/1814005499299312021">Jared Zoneraich (@imjaredz) 的推文</a>：gpt-4o-mini 刚刚发布，比原本就已经是目前最便宜模型的 gpt-4o 还要便宜 33 倍。gpt-4o-mini 比 gpt-4 便宜 200 倍。@tryramp 和 @Superhuman 已经在大规模使用它了。我们...</li><li><a href="https://x.com/xenovacom/status/1813968731250274784">Xenova (@xenovacom) 的推文</a>：Mistral 和 NVIDIA 刚刚发布了 Mistral NeMo，这是一款拥有 128k 上下文长度的最先进 12B 模型！😍 它使用了一种新的基于 Tiktoken 的分词器，在压缩源代码方面效率更高...</li><li><a href="https://x.com/eglyman/status/1813987755270996106">Eric Glyman (@eglyman) 的推文</a>：在早期测试中立刻就能发现，OpenAI 最新的 GPT-4o mini 模型是一次飞跃。它正在帮助我们为客户节省更多时间。引用 Ramp (@tryramp) 的话：很高兴能帮助 OpenAI...</li><li><a href="https://fxtwitter.com/artificialguybr/status/1814018708391760276">𝑨𝒓𝒕𝒊𝒇𝒊𝒄𝒊𝒂𝒍 𝑮𝒖𝒚 (@artificialguybr) 的推文</a>：我删除了关于 GPT-4O 不再对免费用户开放的帖子。不幸的是，我被 UI 搞混了，说了一些废话/假新闻。我为这个错误道歉！</li><li><a href="https://x.com/elder_plinius/status/1814023961535295918?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Pliny the Prompter 🐉 (@elder_plinius) 的推文</a>：⚡️ 越狱警报 ⚡️ OpenAI：被攻破 ✌️😎 GPT-4O-MINI：被解放 🤗 看起来新的“指令层级”防御机制还不够 🤷‍♂️ 见证全新的 gpt-4o-mini ...</li><li><a href="https://x.com/lmsysorg/status/1813999088758673875">lmsys.org (@lmsysorg) 的推文</a>：祝贺 @openai 发布全新的 GPT-4o mini！GPT-4o mini 的早期版本 “upcoming-gpt-mini” 在过去一周已在 Arena 中进行了测试。凭借超过 6000 张用户投票，我们很高兴能分享...</li><li><a href="https://x.com/teortaxesTex/status/1813717300257931588">Teortaxes▶️ (@teortaxesTex) 的推文</a>：Deepseek 的内部日志是你读过的最好的仙侠故事。东蓝鲸宗的弟子们将继续修炼，直到诸天震颤。</li><li><a href="https://x.com/minimaxir/status/1813985834728919249">Max Woolf (@minimaxir) 的推文</a>：GPT-4o mini 的价格是每百万输入 tokens 0.15 美元，每百万输出 tokens 0.60 美元。相比之下，Claude Haiku 是每百万输入 tokens 0.25 美元，每百万输出 tokens 1.25 美元...</li>

<li>ut tokens。这种价格战（race-to-the-bottom）绝对无法持续...</li><li><a href="https://x.com/patloeber/status/1813871331756105744?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Patrick Loeber (@patloeber) 的推文</a>：这是 @karpathy 即将推出的 AI 课程大纲。天哪，我太兴奋了！🤩 特别期待所有动手编码的部分，不仅有 Python，还有 C 和 CUDA。听起来...</li><li><a href="https://x.com/karpathy/status/1814038096218083497?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Andrej Karpathy (@karpathy) 的推文</a>：LLM 模型尺寸竞赛正在加剧……反向加剧！我敢打赌，我们将看到能够非常出色且可靠地“思考”的极小型模型。很可能存在一种设置，甚至...</li><li><a href="https://x.com/Teknium1/status/1813971144695075255">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：不知何故，今天我的信息流里漏掉了这个；Mistral 发布了一个新的基础模型，虽然多了 4B 参数但击败了 l3 8b —— 不确定它是否与旧的 Mistral 架构相同，因为它被称为 Mistral Nemo：https://mi...</li><li><a href="https://x.com/abacaj/status/1813691718522564633">来自 anton (@abacaj) 的推文</a>：OpenAI > 这是我们的一篇酷炫论文，展示了我们如何降低智能模型输出的难度；Anthropic > 这是一个你可以使用的酷炫模型，预计今年晚些时候会推出更大的模型。</li><li><a href="https://x.com/sama/status/1813984927622549881">来自 Sam Altman (@sama) 的推文</a>：早在 2022 年，世界上最好的模型是 text-davinci-003。它比这个新模型差得多得多。而且价格贵了 100 倍。</li><li><a href="https://x.com/togethercompute/status/1813989061503406478">来自 Together AI (@togethercompute) 的推文</a>：今天我们宣布了一个新的推理栈，它提供的解码吞吐量比开源的 vLLM 快 4 倍。我们还推出了新的 Together Turbo 和 Together Lite 端点，使得...</li><li><a href="https://x.com/gdb/status/1814019156561543658?s=46">来自 Greg Brockman (@gdb) 的推文</a>：我们应开发者的普遍需求构建了 gpt-4o mini。我们 ❤️ 开发者，并致力于为他们提供最好的工具，将机器智能转化为各个领域的积极应用。请...</li><li><a href="https://x.com/andrewcurran_/status/1813704834819965147?s=46">来自 Andrew Curran (@AndrewCurran_) 的推文</a>：总结如下：- Llama 4 已于 6 月开始训练 - Llama 4 将是全多模态的，包括音频 - Llama 3 405b 仍将在欧盟发布 - Llama 4 及更高版本将不会在欧盟发布，除非...</li><li><a href="https://x.com/NickADobos/status/1813626926273380429">来自 Nick Dobos (@NickADobos) 的推文</a>：OpenAI 不得不让 AI 变得更笨，好让愚蠢的人类能理解它。引用 OpenAI (@OpenAI)：我们训练了高级语言模型来生成弱模型可以轻松验证的文本，并发现...</li><li><a href="https://x.com/terryyuezhuo/status/1813998867039617444">来自 Terry Yue Zhuo (@terryyuezhuo) 的推文</a>：GPT-4o mini 在 BigCodeBench-Hard 上的表现出炉了：Complete Pass@1: 27.0；Instruct Pass@1: 24.3；平均分：25.7。平均分非常接近 Claude-3-Opus (26.0)！引用 Boris Power (@BorisMPower)：...</li><li><a href="https://x.com/romainhuet/status/1813986836039290970">来自 Romain Huet (@romainhuet) 的推文</a>：发布 GPT-4o mini：迄今为止最智能且最具成本效益的小型模型！它比 GPT-3.5 Turbo 更聪明、更便宜，是 function calling、大上下文、实时交互的理想选择——并且拥有...</li><li><a href="https://news.ycombinator.com/item?id=40998702">未找到标题</a>：未找到描述</li><li><a href="https://x.com/imjaredz/status/1814007428440272953">来自 Jared Zoneraich (@imjaredz) 的推文</a>：快速进行了一次批量运行，对比了 4-turbo 和 4o-mini。速度和成本都有数量级的提升。这将开启许多新的用例，在这些用例中，你会乐于为了速度/成本而牺牲一定的智能。引用 Jared Zon...</li><li><a href="https://x.com/phill__1/status/1813677446362992689">来自 Phil (@phill__1) 的推文</a>：目前在 LMSYS Arena 中至少有 6 个未发布的模型：-gemini-test-1 和 gemini-test-2（可能是新的 Gemini 1.5 版本，也许是 Gemini 2.0）；-im-a-little-birdie (???)；-upcoming-gpt-mini ...</li><li><a href="https://x.com/willdepue/status/1813995162814869892">来自 will depue (@willdepue) 的推文</a>：供参考，voice mode 即将在不久的将来推出。团队为发布此功能付出了巨大的努力。引用 Sam Altman (@sama)：@jakebrowatzke alpha 测试将于本月晚些时候开始，GA 稍后推出。</li><li><a href="https://x.com/vipulved/status/1813991596029084103">来自 Vipul Ved Prakash (@vipulved) 的推文</a>：我们今天发布了 Llama-3 的 Turbo 和 Lite 版本，其中融入了我们在优化和 quantization 方面的最新研究。Lite 模型比 GPT-4o mini 便宜 6 倍，可能是目前最具成本效益的...</li><li><a href="https://x.com/simonw/status/1814003235268829494">来自 Simon Willison (@simonw) 的推文</a>：关于今天发布的 GPT-4o mini 的笔记：https://simonwilli_</li>

son.net/2024/Jul/18/gpt-4o-mini/ 最大的新闻是价格：这甚至比 Claude 3 Haiku 还便宜，每百万输入 token 仅需 15 美分...</li><li><a href="https://x.com/main_horse/status/1813580480761196987">来自 main (@main_horse) 的推文</a>：DeepSeek 创始人梁文锋：我们不会走闭源路线。我们相信首先建立一个强大的技术生态系统更为重要。</li><li><a href="https://x.com/jeffintime/status/1814000186357923851">来自 Jeff Harris (@jeffintime) 的推文</a>：隐藏亮点：GPT-4o mini 支持 16K 的 max_tokens（相比 GPT-4T 和 GPT-4o 的 4K 有所提升） https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/</li><li><a href="https://x.com/abacaj/status/1814000594899870070">来自 anton (@abacaj) 的推文</a>：吞吐量是开源 vLLM 服务的 4 倍……我们部署自己模型的希望在哪里？引用 Together AI (@togethercompute)：今天我们宣布了一个新的推理栈，它提供了解码...</li><li><a href="https://x.com/swyx/status/1812988248660320679">来自 swyx 🤞 🔜 SFO (@swyx) 的推文</a>：完全假设一下……如果有一个开源的 GPT-4o 级别的模型，你会做哪些现在做不到的事情？在“新常态” AI 的范畴内，你能提出哪些可以带来超额收益（alpha）的问题？</li><li><a href="https://mp.weixin.qq.com/s/r9zZaEgqAa_lml_fOEZmjg">揭秘DeepSeek:一个更极致的中国技术理想主义故事</a>：做贡献者，而非搭便车者。</li><li><a href="https://x.com/andrewcurran_/status/1813942258968018954?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Andrew Curran (@AndrewCurran_) 的推文</a>：这是我们的新模型 “GPT-4o mini”。根据 OpenAI 的说法，它是“当今最强大且最具成本效益的小型模型”。今天对免费版和专业版用户开放。</li><li><a href="https://github.com/openai/simple-evals">GitHub - openai/simple-evals</a>：通过在 GitHub 上创建一个账户来为 openai/simple-evals 的开发做出贡献。</li><li><a href="https://www.artificial.agency/news/artificial-agency-launches">Artificial Agency 结束隐身状态并获得 1600 万美元融资，旨在将生成式行为引入游戏领域 — Artificial Agency — Artificial Agency </a>：全球首个 AI 驱动的行为引擎，将运行时决策集成到游戏机制中，开启了新一代自适应和智能化游戏的大门。</li><li><a href="https://x.com/GuillaumeLample/status/1813949898095534278">来自 Guillaume Lample @ ICLR 2024 (@GuillaumeLample) 的推文</a>：非常高兴发布我们的新小型模型 Mistral NeMo，这是一个与 @nvidia 合作训练的 12B 模型。Mistral NeMo 支持 128k token 的上下文窗口，并附带 FP8 对齐的 Checkpoint...</li><li><a href="https://x.com/abacaj/status/1813977261818904908">来自 anton (@abacaj) 的推文</a>：Mistral NeMo 报告的数据（考虑到它是一个 12B 模型对比 Meta Llama 8B）是否有误？出于某种原因，它们让 Llama 3 8B 看起来比实际情况差得多……</li><li><a href="https://x.com/ArtificialAnlys/status/1813965193933623781">来自 Artificial Analysis (@ArtificialAnlys) 的推文</a>：Mistral 发布了 NeMo，这是一个新的开源、长上下文的小型模型，作为 Mistral 7B 的继任者。以下是它令人兴奋的原因 👇 - 具有 128k 上下文窗口的开源模型：大上下文窗口...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e61odl/introducing_spectra_a_comprehensive_study_of/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://news.ycombinator.com/item?id=40996058">Mistral NeMo | Hacker News</a>：未找到描述</li><li><a href="https://brx.ai/">BRX - 加载中</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1263570911253233765)** (1 条消息): 

> - `模型发布日`
> - `更新的线程讨论` 


- **大模型发布日公布**：今天是**大模型发布日**，重点介绍了 AI 社区的重大更新和发布。
   - 成员应**选择加入（opt in）频繁更新的线程讨论**，以获取最新进展。
- **选择加入更新的讨论**：频繁更新的线程讨论现已上线，需要用户**选择加入**。
   - 这确保了用户能够紧跟社区中最新的对话和更新。


  

---

### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1263549154550874204)** (1 messages): 

> - `GPT-4o mini launch` 


- **推出 GPT-4o Mini：更智能、更实惠**：全新的 **GPT-4o mini** 是我们最智能且最实惠的小型模型，现已在 API 中提供，并正在 ChatGPT 中逐步推出。
   - 该模型被描述为比 **GPT-3.5 Turbo** 更智能、更便宜，此次发布附带了[链接](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/)。
- **GPT-4o Mini 对比 GPT-3.5 Turbo**：[OpenAI](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) 推出了 **GPT-4o mini**，据描述其智能程度显著提高，且价格比 **GPT-3.5 Turbo** 更低。
   - 新模型今天已在 API 中上线，并正在 ChatGPT 中逐步推出。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1263211426315767838)** (66 messages🔥🔥): 

> - `Voice extraction model from Eleven Labs`
> - `Switching from GPT to Claude`
> - `Nvidia installer bundling with Facebook, Instagram, and Meta's Twitter`
> - `GPT-4o mini rollout and differences from GPT-4o`
> - `Issues with ChatGPT loading and troubleshooting steps` 


- **Eleven Labs 发布语音提取模型**：提到 Eleven Labs 正在开发 **语音提取模型 (voice extraction model)** 引起了用户的兴趣。
- **Nvidia 安装程序捆绑 Meta 应用**：据一位用户称，**Nvidia 安装程序**将捆绑 Facebook、Instagram 和 Meta 自己的 Twitter（指 Threads），这引起了一些关注。
   - 随后出现了一些幽默的评论，如用 “Yes sir” 确认消息，表现出一种随意的态度。
- **ChatGPT 加载问题持续存在**：用户报告 **ChatGPT** 遇到加载问题，并讨论了各种故障排除步骤，如更换浏览器和清除缓存。
   - 一位用户提到由于问题持续时间较长，想要退款，表达了些许沮丧。
- **GPT-4o mini 的推出与局限性**：成员们注意到 **GPT-4o mini** 正在推出，但缺乏图像支持等功能，这让一些用户感到失望。
   - 关于功能集存在困惑，讨论仍在继续，焦点在于未来的更新是否会包含文本、图像、视频和音频的输入输出等额外功能。



**提到的链接**：<a href="https://tenor.com/view/gollum-lord-of-the-rings-gif-19273356">Gollum Lord GIF - Gollum Lord Of - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1263211465738031315)** (15 messages🔥): 

> - `GPTs Agents`
> - `OpenAI API errors`
> - `4o mini token limits`
> - `OpenAI image token count`
> - `4o mini vs 4o capabilities` 


- **处理 OpenAI API 配额问题**：一位用户遇到了与超出配额相关的 OpenAI API 错误，并被建议检查其计划和账单详情，同时提醒 API 并非免费。
   - 一位社区成员建议必须购买额度（credits）才能继续使用 API。
- **关于 4o mini Token 限制的困惑**：一位用户报告成功向 GPT-4o mini 发送了 150k 个 Token，而该模型本应有 128k 的 Token 限制，并对图像的 Token 计数准确性提出质疑。
   - 该用户指出在 OpenAI 的图像 Token 定价页面上观察到的不一致现象，注意到 4o mini 的图像 Token 计数比 GPT-4o 更高。
- **对比 GPT-4o mini 和 GPT-4o**：社区讨论明确了 GPT-4o mini 并不比 GPT-4o 更聪明，但它是对 GPT-3.5 的升级。
   - *4o mini* 彻底取代了 GPT-3.5，标志着其能力的提升。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1263228216668192828)** (20 条消息🔥): 

> - `Prompt 中的 IF...THEN... 逻辑`
> - `GPT-4 幻觉`
> - `EWAC 命令框架`
> - `带停顿控制的 Voice Agent`
> - `Prompt Engineering 技巧` 


- **优化 IF...THEN... 逻辑**：一位成员建议在构建 IF...THEN... 逻辑时不使用否定指令，以更有效地引导模型的输出。
   - 这种方法旨在防止模型提供错误或不相关的响应。
- **处理 GPT-4 幻觉**：讨论了 GPT-4 在被问及不熟悉的技术（例如为名为 'fleperwelp' 的不存在技术创建用例）时产生幻觉的倾向。
   - 建议通过明确模型在遇到未知术语时应该执行的操作，以避免诱导此类幻觉的 Prompt 方式。
- **探索 EWAC 命令框架**：一位成员介绍了一个新的 EWAC 命令框架实验，该框架在 zero-shot、system prompting 和通用查询中表现良好。
   - 他们分享了一个 [讨论链接](https://discord.com/channels/974519864045756446/1263348214749335613) 以获取进一步的见解和协作。
- **增强 Voice Agent 的适当停顿**：一位成员开发了一个 Voice Agent，能够通过插入特殊字符来表示停顿，从而改变语速。
   - 他们寻求关于如何教模型根据上下文（如电话号码和地址）在句子中适当插入停顿的建议。
- **Prompt Engineering 的前期调研 (Reconnaissance)**：强调了在定义 Prompt 的特定部分之前，了解模型已经掌握了哪些知识的重要性。
   - 这种前期调研通过有效利用模型的训练数据，有助于使模型的输出与用户预期保持一致。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1263228216668192828)** (20 条消息🔥): 

> - `ChatGPT 幻觉管理`
> - `EWAC 讨论框架`
> - `Voice Agent 停顿控制`
> - `创新的 Prompting 技术`
> - `AI 响应中的思维启发` 


- **管理 ChatGPT 的幻觉**：通过使用不含否定指令的 IF...THEN 结构，并识别模型何时不知道某些信息，可以减轻 ChatGPT 的幻觉。
   - 具体示例包括避免引导幻觉的 Prompt，并利用模型识别不熟悉技术的能力。
- **为 GPT 引入 EWAC 命令**：讨论了一种新颖的 Prompting 框架 EWAC，用于将文本转换为特定命令，增强 zero-shot 和 system prompting。
   - 通过分享的链接和详细示例，探索了模型高效识别如何应用 EWAC 的能力。
- **控制 Voice Agent 的停顿**：详细介绍了一个可以通过插入特殊字符控制停顿来调节语速的 Voice Agent 的开发过程，包括面临的挑战。
   - 讨论了在电话号码、地址和其他语境中停顿的正确位置，并寻求改进 GPT 对自然语言模式理解的方法。
- **AI 响应中的思维启发**：分享了一种新技术，使 AI 能够在响应中激发多层思考，并根据查询的复杂性进行调整。
   - 该过程涉及用于详细思维启发的自定义指令，并根据用户的请求动态选择思维层级。
- **模型现有知识引导**：为了更好地界定 Prompt 的范围，建议在提供额外指令之前，先询问模型关于相关主题的现有知识。
   - 这可以提高响应质量，特别是在利用其对人类语言模式的理解来处理语音中的停顿插入等领域。


  

---



### **Interconnects (Nathan Lambert) ▷ #[events](https://discord.com/channels/1179127597926469703/1179127598442348729/)** (1 条消息): 

natolambert: 有人在 ICML 吗？我的一个 VC 朋友想在高级晚宴上见见我的朋友们。

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1263376755381637201)** (74 条消息🔥🔥): 

> - `Meta 的多模态 Llama 模型`
> - `Mistral NeMo 发布`
> - `GPT-4o mini 发布`
> - `Tekken 分词器 (tokenizer)`
> - `OpenAI 安全机制越狱 (jailbreak)` 


- **Meta 将欧盟排除在多模态 Llama 模型之外**：Meta 宣布他们将在未来几个月发布多模态 Llama 模型，但由于欧洲监管环境的不可预测性，该模型将不会在欧盟提供，详见 [Axios 报告](https://www.axios.com/2024/07/17/meta-future-multimodal-ai-models-eu)。
   - 一位成员指出，这种情况意味着他们要么使用 VPN，要么勾选非欧盟合规复选框来访问它。
- **Mistral NeMo 与 NVIDIA 合作发布**：Mistral NeMo 是一款与 NVIDIA 合作构建的新型 12B 模型，具有 128k token 上下文窗口。该模型已发布，并采用 Apache 2.0 许可证，提供预训练和指令微调（instruction-tuned）检查点，详见[官方发布说明](https://mistral.ai/news/mistral-nemo/)。
   - Mistral NeMo 在推理、世界知识和代码准确性方面比之前发布的模型具有更优越的性能。
- **OpenAI 发布 GPT-4o mini**：OpenAI 发布了 GPT-4o mini，被誉为目前最强大且最具成本效益的小型模型，在 MMLU 上得分为 82%，根据 [Andrew Curran 的公告](https://x.com/andrewcurran_/status/1813942258968018954?s=46)，该模型对免费和付费用户开放。
   - 该模型比 GPT-3.5 更便宜，定价为每 M token 输入 15 美分，每 M token 输出 60 美分，具有 128k 上下文窗口。
- **Tekken 分词器 (tokenizer) 表现优于 Llama 3**：新型分词器模型 Tekken 在压缩包括源代码和几种主要语言在内的多语言文本方面表现出比 Llama 3 分词器更优越的性能。
   - 这种效率的提升使其在中文、韩文和阿拉伯文等各种语言中的效果提高了约 30% 到 300%。
- **OpenAI 新安全机制被越狱**：据 [Elder Plinius](https://x.com/elder_plinius/status/1814023961535295918?s=46) 称，OpenAI GPT-4o-mini 中新实施的安全机制已被越狱，使其能够输出受限内容，如恶意软件和受版权保护的材料。
   - 这突显了名为“指令层级 (instruction hierarchy)”的最新防御机制中的漏洞。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/phill__1/status/1813677446362992689">来自 Phil (@phill__1) 的推文</a>: 目前在 lmsys arena 中至少有 6 个未发布的模型：-gemini-test-1 和 gemini-test-2（可能是新的 Gemini 1.5 版本，也许是 Gemini 2.0）-im-a-little-birdie (???) -upcoming-gpt-mini ...</li><li><a href="https://mistral.ai/news/mistral-nemo/">Mistral NeMo</a>: Mistral NeMo：我们最新的最佳小型模型。一款具有 128k 上下文长度的最先进 12B 模型，与 NVIDIA 合作构建，并根据 Apache 2.0 许可证发布。</li><li><a href="https://x.com/andrewcurran_/status/1813942258968018954?s=46">来自 Andrew Curran (@AndrewCurran_) 的推文</a>: 这是我们的新模型。“GPT-4o mini”。根据 OpenAI 的说法，它是“当今最强大且最具成本效益的小型模型”。今天对免费和专业用户上线。</li><li><a href="https://x.com/andrewcurran_/status/1813965829996003608?s=46">来自 Andrew Curran (@AndrewCurran_) 的推文</a>: 对于许多询问 GPT-4o Mini 的 API 定价的人，它是：每 M token 输入 15¢，每 M token 输出 60¢，128k 上下文窗口</li><li><a href="https://x.com/morqon/status/1813960872810996211?s=46">来自 morgan — (@morqon) 的推文</a>: gpt-4o mini 在 MMLU 上得分为 82%，信不信由你</li><li><a href="https://fxtwitter.com/testingcatalog/status/1813965406664900856?s=46">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>: 基于</li><li><a href="https://x.com/elder_plinius/status/1814023961535295918?s=46">来自 Pliny the Prompter 🐉 (@elder_plinius) 的推文</a>: ⚡️ 越狱警报 ⚡️ OPENAI：被黑 ✌️😎 GPT-4O-MINI：已解放 🤗 看来新的“指令层级”防御机制还不够 🤷‍♂️ 见证新的 gpt-4o-mini ...</li><li><a href="https://x.com/paulgauthier/status/1814014867361374610?s=46">来自 Paul Gauthier (@paulgauthier) 的推文</a>: GPT 4o mini 在 aider 的代码编辑基准测试中得分与原始 GPT 3.5 相当（后期的 3.5 版本表现更差）。乍一看，它似乎无法通过 diff 编辑代码，这限制了它的用途 ...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1263263486079795220)** (5 messages): 

> - `Code-related PRM datasets`（代码相关的 PRM 数据集）
> - `AST mutation method`（AST 变异方法）
> - `Positive, Negative, Neutral labels vs Scalar values`（正向、负向、中性标签 vs 标量值）
> - `PRM-800K`
> - `Research & MS program`（研究与硕士项目）


- **对代码相关 PRM 数据集的需求**：一名成员询问是否有可用的代码相关 PRM 数据集，并提到了 "Let's Reward Step by Step" 论文中使用的 **AST mutation method**。
   - *据我所知没有。这正是迫切需要的，请把它做出来。*
- **PRM 标签中的正/负/中性 vs 标量值**：讨论了为什么 PRM 使用**正向、负向、中性**标签而不是标量值，考虑到生成经过校准的标量值数据集具有挑战性。
   - 讨论中还表现出对为什么不尝试标量值的关注，特别是考虑到像 PRM-800K 这样涉及人类数据的数据集。
- **探索 PRM 和合成数据的研究**：一名成员表示有兴趣在即将开始的硕士（MS）项目中研究 **PRM 和合成数据**，并寻求相关知识和见解。
   - *关于“正/负/中性 vs 标量”这件事，有什么可以分享的知识吗？*
- **对某本未指明书籍的引用**：关于书籍引用的询问得到了一个模糊的回答，称这些书正处于广泛的诉讼中。
   - *可能是那些正面临广泛诉讼的书，但我不能正式说明。*


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1263218671552168088)** (21 messages🔥): 

> - `Public Perception of AI`（公众对 AI 的认知）
> - `OpenAI's Business Challenges`（OpenAI 的业务挑战）
> - `Google vs OpenAI Competition`（Google 与 OpenAI 的竞争）
> - `AI Scaling Issues`（AI 扩展问题）


- **公众对 AI 工具的不适感**：一名成员讨论道，**普通人**可能会对 **ChatGPT** 这样强大的 AI 工具感到困惑和反感，除非行业能找到将更简单版本货币化的方法。
   - 他们强调，从历史上看，公众对让他们感到不适的技术往往反应负面，并将其比作现代的**巫术（witchcraft）**。
- **OpenAI 可能面临严峻的业务问题**：成员们推测 **OpenAI** 在从几百人规模扩展到几千人规模时，是否正面临严峻的业务问题。
   - 有观点认为，与大厂（Big Tech）不同，OpenAI 无法轻易重新分配现有资源，并质疑这如何与其 **AGI** 的首要使命保持一致。
- **Google 在创新方面超越 OpenAI**：讨论强调 **Google** 在发布新功能方面似乎超过了 **OpenAI**，特别是提到了使用 **GPT-4o mini** 生成图像。
   - 一名成员指出，由于发布周期变慢，**OpenAI** 现在被视为一家**老牌公司（boomer company）**。
- **OpenAI 失去领先模型地位**：一名成员指出，虽然 **GPT-4T** 发布时优于 **GPT-4**，但其他组织（包括开源界）已近乎追平。
   - 他们对 OpenAI 在利用了来自 **ChatGPT** 的用户偏好数据的情况下，仍未能保持显著领先地位感到惊讶。



**提到的链接**：<a href="https://x.com/cto_junior/status/1813956330287513717?s=46">来自 TDM (e/λ) (@cto_junior) 的推文</a>：每一个酷炫的东西都推迟了，我很确定我们会在这一切之前得到 Gemini-2.0，反正它支持所有模态。

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1263352302366883870)** (9 messages🔥): 

> - `Codestral Mamba model`
> - `DeepSeek-V2-0628 release`
> - `Mamba infinite context`（Mamba 无限上下文）
> - `Open-sourced models`（开源模型）
> - `LMSYS Chatbot Arena` 


- **Codestral Mamba 准确率在 1k token 后骤降**：[一条推文](https://x.com/louisknightwebb/status/1813678943230439851?s=46) 揭示了 **Codestral Mamba** 的准确率在约 1k token 后降至零，强调这仍是一个持续的研究问题。
- **DeepSeek-V2-0628 开源并位居前列**：[DeepSeek 宣布](https://x.com/deepseek_ai/status/1813921111694053644?s=46) 他们的 **DeepSeek-V2-0628** 模型现已开源，在 LMSYS Chatbot Arena 排行榜的开源类别中排名**第一**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/louisknightwebb/status/1813678943230439851?s=46">来自 Louis Knight-Webb (@LouisKnightWebb) 的推文</a>：Codestral Mamba 🐍 的准确率在约 1k token 上下文后降至零。对比 Codestral（普通版）。看来这整个 Mamba 的事情在很大程度上仍然是一个开放的研究问题，但即便如此……</li><li><a href="https://x.com/deepseek_ai/status/1813921111694053644?s=46">来自 DeepSeek (@deepseek_ai) 的推文</a>：🎉激动人心的消息！我们开源了 DeepSeek-V2-0628 权重，它是 LMSYS Chatbot Arena 排行榜 @lmsysorg 上排名第一的开源模型。详细 Arena 排名：总榜第 11，困难提示词（Hard Prompts）第 3，Co...
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1263565530372112538)** (1 条消息): 

> - `GPT-4o Mini`
> - `GPT-4o Mini 的成本效益` 


- **OpenAI 发布 GPT-4o Mini**：[GPT-4o mini](https://openrouter.ai/models/openai/gpt-4o-mini) 是 OpenAI 继 GPT-4 Omni 之后推出的最新模型，支持文本和图像输入，并输出文本。
   - 该模型在保持 **SOTA (state-of-the-art) 智能水平** 的同时，具有显著的**成本效益**，价格仅为 **$0.15/M input** 和 **$0.60/M output**。
- **GPT-4o Mini：高性价比 AI 解决方案**：GPT-4o Mini 的价格比近期其他前沿模型便宜数倍，且比 [GPT-3.5 Turbo](https://openrouter.ai/models/openai/gpt-3.5-turbo) 便宜 60% 以上。
   - GPT-4o Mini 的价格为 **$0.15/M input** 和 **$0.60/M output**，是用户的经济之选。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/models/openai/gpt-4o-mini>)">OpenAI: GPT-4o by openai</a>：GPT-4o（“o”代表“omni”）是 OpenAI 最新的 AI 模型，支持文本和图像输入，并输出文本。它保持了 [GPT-4 Turbo](/models/open... 的智能水平。</li><li><a href="https://openrouter.ai/models/openai/gpt-3.5-turbo>)">OpenAI: GPT-3.5 Turbo by openai</a>：GPT-3.5 Turbo 是 OpenAI 最快的模型。它可以理解并生成自然语言或代码，并针对聊天和传统的补全任务进行了优化。训练数据截至 2021 年 9 月。
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1263243002571395074)** (97 条消息🔥🔥): 

> - `Codestral 22B`
> - `OpenRouter 停机故障`
> - `Mistral NeMo 发布`
> - `GPT-4o mini 发布`
> - `图像 Token 计费问题` 


- **Codestral 22B 模型请求**：一位用户请求添加 Codestral 22B，并分享了 [Mamba-Codestral-7B 的模型卡片](https://huggingface.co/mistralai/mamba-codestral-7B-v0.1)，这是一个开源代码模型，性能与 SOTA 的基于 Transformer 的代码模型相当。
- **OpenRouter 遭遇停机**：多位用户报告了 OpenRouter 的问题，包括 API 请求挂起和网站超时，而另一些用户则表示在北欧等不同地区运行正常。
- **Mistral NeMo 128K 上下文模型发布**：[Mistral NeMo](https://t.co/FgHDivTLh5) 发布，这是一个 12B 模型，提供高达 128k tokens 的上下文窗口，预训练和指令微调的 Checkpoints 均在 Apache 2.0 许可证下提供。
- **GPT-4o mini 发布**：OpenAI 宣布发布 [GPT-4o mini](https://www.cnbc.com/2024/07/18/openai-4o-mini-model-announced.html)，被描述为最强大且最具成本效益的小型模型，适用于免费 ChatGPT 用户、ChatGPT Plus、Team 和 Enterprise 用户。
- **图像 Token 计费问题**：用户讨论了 GPT-4o mini 与其他模型在图像 Token 计数和定价方面的差异，一些人注意到图像的 Token 计数异常高。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://t.co/FgHDivTLh5">Mistral NeMo</a>：Mistral NeMo：我们最新的最佳小型模型。一个具有 128k 上下文长度的 SOTA 12B 模型，与 NVIDIA 合作构建，并根据 Apache 2.0 许可证发布。</li><li><a href="https://huggingface.co/mistralai/mamba-codestral-7B-v0.1">mistralai/mamba-codestral-7B-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://www.cnbc.com/2024/07/18/openai-4o-mini-model-announced.html">OpenAI 首次推出其最强大模型的 mini 版本</a>：OpenAI 周四推出了一款新的 AI 模型“GPT-4o mini”，这是这家人工智能初创公司扩大其热门聊天机器人使用的最新努力。 </li><li><a href="https://x.com/mattshumer_/status/1813952065057542522">Matt Shumer (@mattshumer_) 的推文</a>：新的 @OpenAI 模型！GPT-4o mini 今天发布。似乎是 GPT-3.5-Turbo 的替代品（终于！）。看起来这个模型将与 Claude Haiku 非常相似——快速/廉价，并且非常擅长处理...</li><li><a href="https://status.openrouter.ai/">OpenRouter 状态</a>：OpenRouter 事件历史
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1263252832392450051)** (7 messages): 

> - `链接 C 库的请求`
> - `Mojo GPU 支持`
> - `Max 平台 NVIDIA GPU 公告`
> - `MLIR 方言与 CUDA/NVIDIA` 


- **链接 C 库的支持请求**：一位用户分享了一个 [GitHub ticket](https://github.com/modularml/mojo/issues/3262)，请求在 Mojo 中支持链接到 C 库。
- **Mojo 添加 GPU 支持**：关于 **Max/Mojo** 中新增 **GPU 支持** 的讨论展开，起因是一个关于如何将其用于 Tensor 操作和并行化的查询。
   - 另一位成员提到了 **Chris Lattner 在 Max 平台演讲**中关于 NVIDIA GPU 支持的公告，而其他人则分享了其通过 **MLIR 方言** 以及 CUDA/NVIDIA 集成进行并行计算的见解。



**提及的链接**: <a href="https://github.com/modularml/mojo/issues/3262)">Issues · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。

  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: 来自 *Modular*:
<https://twitter.com/Modular/status/1813988940405493914>
  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1263208633446170735)** (7 messages): 

> - `视频中的图像目标检测`
> - `帧率调整`
> - `处理边界框 (bounding box) 问题`
> - `处理 MP4 视频`
> - `管理大型视频帧` 


- **图像目标检测的帧率解决方案**：在实时应用中，通常以 **5 fps** 等低帧率运行图像目标检测模型，以解决视频处理中的常见问题。
   - 还可以应用后处理来平滑 **边界框 (bounding box) 位置**，这在特定应用中可能会出现问题。
- **处理大型视频帧的挑战**：一位成员表达了对在大型视频中管理大量帧进行目标检测的担忧。
   - 另一位成员建议，目前没有“魔法方案”，如果必须处理所有帧，那就只能全部处理。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1263211911164592170)** (35 messages🔥): 

> - `在 Mojo 中遍历 Tuple`
> - `Mojo 中的命名规范`
> - `Keras 3.0 兼容性与进展`
> - `MAX 与 HPC 能力`
> - `在 Mojo 中为 FloatLiterals 的 Tuple 设置别名` 


- **在 Mojo 中遍历 Tuple 存在挑战**：一位用户询问如何在 Mojo 中遍历 Tuple，但另一位用户解释说，由于 Tuple 是异构的（heterogeneous），这通常是不可能的。
- **分享 Mojo 命名规范资源**：针对有关 Mojo 命名规范的咨询，分享了一个 [GitHub 代码风格指南](https://github.com/modularml/mojo/blob/main/stdlib/docs/style-guide.md)。
- **Keras 3.0 支持多框架**：Keras 3.0 宣布支持在 JAX, TensorFlow 或 PyTorch 上运行工作流，这可以显著改善模型训练和部署。
- **MAX 与图编译器限制**：关于 MAX 能力的辩论，强调其目前的重点是类似于 XLA 的图编译器，对于通用 HPC 应用存在局限性。
- **在 Mojo 中显式为 FloatLiterals 的 Tuple 设置别名**：一位用户询问如何在 Mojo 中为 FloatLiterals 的 Tuple 设置别名，发现需要显式类型标注，如 `Tuple[FloatLiteral, FloatLiteral](1.0, 2.0)`。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://keras.io/keras_3/">Keras: 为人类设计的深度学习</a>: 无描述</li><li><a href="https://youtu.be/_QVs626Vn2k?t=3934">Mojo 🔥 社区会议 #4</a>: Mojo 社区会议 #4 录音 🫓 Flat Buffers: 内存高效的序列化 ⚒️ Forge Tools: 扩展 Mojo 🔥 标准库 🔄 Mojo 🔥 Gen...</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/docs/style-guide.md">mojo/stdlib/docs/style-guide.md at main · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1263373831993688245)** (5 条消息): 

> - `命令行 Prompt 用法`
> - `模型权重 URI`
> - `Llama 3 Pipeline`
> - `交互式聊天机器人示例` 


- **命令行 Prompt 说明**：一位用户询问 `mojo ../../run_pipeline.🔥 llama3 --prompt ...` 中的 `--prompt` 标志是否充当上下文窗口，以及构建交互式聊天是否需要输入循环。
   - [官方链接](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3)解释了 Prompt 格式。
- **模型权重的首选来源**：一位用户询问从官方 [Hugging Face 仓库](https://huggingface.co/meta-llama/)加载权重是否比从 `bartowski` 和 `QuantFactory` 等第三方加载更好。
   - 用户引用了 [GitHub 仓库中的代码行](https://github.com/modularml/max/blob/7189864b2fc829176149f6997a70c62732982ec8/examples/graph-api/pipelines/llama3/run.%F0%9F%94%A5#L224-L243)来讨论这些权重。
- **使用自定义权重的 Llama 3 Pipeline**：一位对 `llama3-70B-instruct` 模型感兴趣的用户想知道 modularml 是如何选择其在 Hugging Face 上的模型 URI 的。
   - 回复指出，权重可以通过 `--model-path` 参数指定，并强调了使用 GGUF 版本以便于摄取和延迟加载（lazy-loading）。
- **使用 MAX 的交互式聊天机器人**：一项说明澄清了已发布的 MAX 24.4 使用 `--prompt` 来填充初始上下文并运行一次生成。
   - [nightly 分支](https://github.com/modularml/max/tree/nightly/examples/gui)包含一个交互式聊天机器人示例，它可以保留上下文并设置系统提示词（system prompts），该示例在 [YouTube 上的社区会议](https://www.youtube.com/live/uookgZ7Ojg8?si=u-iwoMJWmMigVwSH&t=1197)中进行了展示。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/modularml/max/tree/nightly/examples/gui">max/examples/gui at nightly · modularml/max</a>：一系列示例程序、笔记本和工具，展示了 MAX 平台的强大功能 - modularml/max</li><li><a href="https://www.youtube.com/live/uookgZ7Ojg8?si=u-iwoMJWmMigVwSH&t=1197">Modular Community Livestream - New in MAX 24.4</a>：MAX 24.4 现已发布！加入我们的直播，讨论 MAX Engine 和 Mojo🔥 的新特性 - macOS 上的 MAX、MAX Engine 量化 API 等...</li><li><a href="https://huggingface.co/meta-llama/">meta-llama (Meta Llama)</a>：未找到描述</li><li><a href="https://github.com/modularml/max/blob/7189864b2fc829176149f6997a70c62732982ec8/examples/graph-api/pipelines/llama3/run.%F0%9F%94%A5#L224-L243">max/examples/graph-api/pipelines/llama3/run.🔥 at 7189864b2fc829176149f6997a70c62732982ec8 · modularml/max</a>：一系列示例程序、笔记本和工具，展示了 MAX 平台的强大功能 - modularml/max</li><li><a href="https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3">Meta Llama 3 | Model Cards and Prompt formats</a>：Meta Llama 3 使用的特殊 Token。一个 Prompt 应包含单个系统消息，可以包含多个交替的用户和助手消息，并且始终以最后一个用户消息结尾...
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1263311768772546580)** (13 条消息🔥): 

> - `Mojo Compiler Update` (Mojo 编译器更新)
> - `Standard Library Extensions Proposal` (标准库扩展提案)
> - `Discussion on Allocator Awareness` (关于 Allocator Awareness 的讨论)
> - `Async IO API and Performance` (Async IO API 与性能)
> - `Opt-out of stdlib` (选择退出 stdlib)


- **Mojo 编译器更新发布**：新的 [nightly Mojo 编译器版本 2024.7.1805](https://github.com/modularml/mojo/compare/e2a35871255aa87799f240bfc7271ed3898306c8...bb7db5ef55df0c48b6b07850c7566d1ec2282891) 已经发布，其中包含对 `stdlib` 的更新，包括支持使用列表字面量（list literals）创建嵌套的 Python 对象。
   - 可以使用 `modular update nightly/mojo` 进行更新，完整的变更日志可在 [此处](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 查看。
- **减轻 Stdlib 工作量的提案**：讨论了一项关于[通过 `stdlib-extensions` 减轻 stdlib 维护者工作量](https://github.com/modularml/mojo/discussions/3233)的提案，旨在在做出 stdlib 承诺之前引入更多社区意见。
   - 这种孵化器方法被建议用于在集成前评估 API 和受欢迎程度，无论 stdlib 审查者的数量多少，都能提供价值。[更多详情请点击此处](https://github.com/gabrieldemarmiesse/mojo/blob/proposal_stdlib_extensions/proposals/stdlib-extensions.md#the-future-of-this-repository-when-mojo-has-a-public-source-of-truth)。
- **标准库中的 Allocator Awareness**：成员们讨论了在将想法集成到 stdlib 之前让社区进行评估的重要性，特别是对于像 allocator awareness 这样的小众用例。
   - Rust 尽管有需求但仍缺乏 allocator awareness 的问题被引用为潜在摩擦的例子，强调了 Mojo 中社区审查的必要性。
- **标准 Async IO API 性能**：有人建议建立一个标准的 async IO API，以支持高性能模型，其中 IO 操作可以交付缓冲区，该 API 独立于现有的 Python API。
   - 这将迎合高性能 API，并确保 Mojo 社区内的兼容性。
- **在 Mojo 中选择退出标准库**：围绕选择加入或退出 stdlib 的可能性展开了讨论，一些人质疑这种选项的可行性和用例。
   - 辩论包括对避免“关注性能”的用户与遵循流行但可能不兼容的解决方案的用户之间产生分歧的担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/gabrieldemarmiesse/mojo/blob/proposal_stdlib_extensions/proposals/stdlib-extensions.md#the-future-of-this-repository-when-mojo-has-a-public-source-of-truth">mojo/proposals/stdlib-extensions.md at proposal_stdlib_extensions · gabrieldemarmiesse/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建一个账户来为 gabrieldemarmiesse/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/discussions/3233">[Proposal] Reduce the workload of stdlib&#39;s maintainers with `stdlib-extensions` · modularml/mojo · Discussion #3233</a>：此讨论旨在提供一个讨论以下提案的场所：pull request markdown 文档。我们对频繁贡献者以及 st... 的意见特别感兴趣。
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo-marathons](https://discord.com/channels/1087530497313357884/1255303604894437388/1263224800147542017)** (16 messages🔥): 

> - `Lubeck`
> - `MKL`
> - `LLVM`
> - `BLAS Linking`
> - `SPIRAL` 


- **Lubeck 凭借 LLVM 超越 MKL**：一位成员表示 **Lubeck** 比 **MKL** 更快，将其速度归功于 **LLVM IR generation** 的差异。
   - *Mir，一个 LLVM 加速的通用数值库*，也可能对 Lubeck 的性能有所贡献，正如[这篇博客文章](http://blog.mir.dlang.io/glas/benchmark/openblas/2016/09/23/glas-gemm-benchmark.html)所述。
- **SPIRAL 程序推动自动化边界**：**SPIRAL** 旨在自动化数值内核（numerical kernels）的软件和硬件优化，超越了目前的工具。
   - 一位成员称赞了 SPIRAL 生成数值函数的**平台调优实现（platform-tuned implementations）**的能力，尽管它在 BLAS 等高价值领域之外的函数上难以使用。
- **SPIRAL：难以使用但高度优化**：SPIRAL 生成的高度优化代码类似于数学论文，使其难以用于通用目的。
   - [SPIRAL 的目标](https://spiral.ece.cmu.edu/pub-spiral/pubfile/paper_146.pdf)是通过高级描述实现最佳性能，尽管它仍然很复杂，且主要对 BLAS 函数有用。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="http://blog.mir.dlang.io/glas/benchmark/openblas/2016/09/23/glas-gemm-benchmark.html">Numeric age for D: Mir GLAS is faster than OpenBLAS and Eigen</a>：未找到描述</li><li><a href="http://www.spiral.net/">SPIRAL Project: Home Page</a>：未找到描述
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1263298613627457646)** (52 messages🔥): 

> - `Creating new tools for API`
> - `Tools vs Connectors`
> - `Permissions for sending images and GIFs`
> - `DuckDuckGo search in projects` 


- **如何创建新的 API 工具**：一位成员询问了如何为 API 创建新工具，发现了工具（tools）与连接器（connectors）之间的差异，并被引导查看 [Cohere dashboard](https://dashboard.cohere.com)。
   - 讨论明确了**工具仅限 API**，可以是单步或多步的，并且是在客户端定义的，如 [Cohere 的文档](https://docs.cohere.com/docs/tool-use)中所述。
- **聊天中的图片和 GIF 权限**：成员们请求在通用聊天中发送图片和 GIF 的权限，并指出了一些限制。
   - 一位管理员解释说，这些权限可能会受到限制以防止滥用，但正在考虑进行更改，并且可能会为开发者和常规用户开启权限。
- **DuckDuckGo 搜索工具集成**：成员们讨论了使用 DuckDuckGo 获取链接并将其集成到项目中。
   - 分享了一个指向 [DuckDuckGo search Python package](https://pypi.org/project/duckduckgo-search/) 的链接，一位成员提到用它为工作创建了一个自定义工具。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://pypi.org/project/duckduckgo-search/">duckduckgo-search</a>：使用 DuckDuckGo.com 搜索引擎搜索词汇、文档、图片、新闻、地图和文本翻译。</li><li><a href="https://tenor.com/view/yay-kitty-cat-happy-excited-gif-10302657046876115666">Yay Kitty GIF - Yay Kitty Cat - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/trombone-pusheen-musician-instrument-gif-11434220432919976776">Trombone Pusheen GIF - Trombone Pusheen Musician - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://docs.cohere.com/docs/tool-use">Tool Use with Cohere's Models - Cohere Docs</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1263488823795650615)** (31 条消息🔥): 

> - `用于爬虫的 Python 开发`
> - `用于收集 URL 的库`
> - `Firecrawl 自托管`
> - `Firecrawl 的成本担忧`
> - `与 GPT-4o 的 API 集成` 


- **Python 流线化网页爬取**：讨论集中在结合使用 **Python** 和 **Streamlit** 进行概念验证，并利用 **Firecrawl** 爬取内容。
   - 提到与 **ddg** 结合是一种实现可行系统的方案。
- **使用 'duckduckgo-search' 库高效收集 URL**：成员们讨论了使用 [非官方 duckduckgo-search 库](https://pypi.org/project/duckduckgo-search/) 来收集 URL，并使用 BeautifulSoup 进行爬取。
   - 特别提到这是一个**免费**资源。
- **Firecrawl 后端自托管节省成本**：值得注意的是，提到了通过**自托管 Firecrawl** 的后端来降低高昂成本，从而方便创建 API 端点。
   - 分享了 [Firecrawl 的自托管指南](https://github.com/mendableai/firecrawl/blob/main/SELF_HOST.md)，强调这一省钱措施可以为用户节省数百美元。
- **解决 Firecrawl 定价担忧**：虽然 Firecrawl 被认为非常有效，但成员们承认其定价偏高，且缺乏按需付费（pay-as-you-go）计划。
   - 社区成员对了解更便宜的自托管选项表示赞赏。
- **将 GPT-4o API 与爬虫工具集成**：成员们讨论了使用自己的 API 密钥将 Firecrawl 与 **GPT-4o** 集成。
   - 配置包括在 .env 文件中设置 API 密钥，以实现网页爬取、抓取和 LLM 提取。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pypi.org/project/duckduckgo-search/">duckduckgo-search</a>：使用 DuckDuckGo.com 搜索引擎搜索词条、文档、图像、新闻、地图和文本翻译。</li><li><a href="https://github.com/mendableai/firecrawl/blob/main/SELF_HOST.md">firecrawl/SELF_HOST.md at main · mendableai/firecrawl</a>：🔥 将整个网站转换为适用于 LLM 的 Markdown 或结构化数据。通过单个 API 进行爬取、抓取和提取。 - mendableai/firecrawl</li><li><a href="https://jsfiddle.net/razodactyl/gqr5vaot/1/">Edit fiddle - JSFiddle - Code Playground</a>：未找到描述
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1263208867290939512)** (63 条消息🔥🔥): 

> - `Google Sheets 登录问题`
> - `Perplexity 分析多个 PDF`
> - `GPT-4 与 GPT-4 Omni 的回答对比`
> - `来自罗技的 Perplexity Pro 邮件`
> - `DALL-E 更新推测` 


- **带有 Drive 标志的 Google Sheets 登录问题**：一位成员报告在尝试登录使用 Google Sheets 创建的页面时，遇到了显示 Google Drive 标志的错误。
   - 他们寻求帮助以解决“无法访问页面”的错误。
- **分析多个 PDF 的限制与策略**：成员们讨论了 Perplexity 在分析超过 4 个 PDF 以及将 PDF 与网页搜索结合时的局限性。
   - 有人建议将 PDF 和网页搜索的内容转换为 .txt 文件，然后附加到新对话中。
- **GPT-4 与 GPT-4 Omni 响应的差异**：一位成员质疑为什么在开启 GPT-4 Omni 的情况下，ChatGPT 4 和 Perplexity 会给出不同的答案。
   - 另一位成员推测差异可能是由于使用了不同的模型。
- **疑似提供罗技 Perplexity Pro 的钓鱼邮件**：成员们就一封声称由罗技提供 6 个月 Perplexity Pro 会员的邮件真实性展开辩论，部分人表示怀疑。
   - 经核实，社交媒体上已确认此事，Perplexity 的首席业务官（Chief Business Officer）也确认了该合作伙伴关系。
- **关于 DALL-E 更新的推测**：成员们注意到 Perplexity Pro 搜索重置的问题，并推测这可能与 DALL-E 的升级有关。
   - 一些成员对目前生成图像的限制表示沮丧，认为这可能与新版本的发布有关。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.theverge.com/2024/7/18/24200714/openai-new-cheaper-smarter-model-gpt-4o-mini">OpenAI 发布更便宜、更智能的模型</a>：OpenAI 正在推出名为 GPT-4o Mini 的更便宜、更智能的模型，作为开发者更易获取的模型。</li><li><a href="https://x.com/dmitry140/status/1813698975884792095">Dmitry Shevelenko (@dmitry140) 的推文</a>：Perplexity 🤝 罗技。感谢 @ATXsantucci 达成的伟大合作。才刚刚开始！引用 Jorge Barba (@jorgebarba)：哇！完全出乎意料。收到了 6 个月的 @perplexity_ai Pro 订阅...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1263252162830532679)** (5 messages): 

> - `创纪录的剑龙拍卖`
> - `实验室培育宠物食品获批`
> - `Anthropic 的 1 亿美元 AI 基金`
> - `H2O-3 代码执行漏洞` 


- **创纪录的剑龙拍卖**: [Perplexity AI](https://www.perplexity.ai/search/where-does-the-rhine-originate-leG7SSmcSOumGgMjEKEfWw#0) 重点介绍了创纪录的剑龙化石拍卖，引发了广泛关注。
   - 讨论强调了其**惊人的价格**和此次拍卖的**历史意义**。
- **实验室培育宠物食品获批**: 此处链接的 [YouTube 视频](https://www.youtube.com/embed/do_EmoTIMn0) 宣布了实验室培育宠物食品的获批，吸引了社区的注意。
   - 视频强调了实验室培育方案的**伦理考量**和**营养益处**。
- **Anthropic 的 1 亿美元 AI 基金**: [Perplexity AI](https://www.perplexity.ai/search/i-want-you-to-do-some-research-ynMkNdSLQFSRQ5ujxNssRQ) 透露 Anthropic 启动了一项 1 亿美元的基金，旨在推动 AI 技术的发展。
   - 成员们讨论了该计划对 **AI 研究**的潜在影响以及该倡议资助的**未来创新**。
- **H2O-3 代码执行漏洞**: Perplexity AI 上的一个关键[页面](https://www.perplexity.ai/page/h2o-3-code-execution-vulnerabi-zynZYKoxSqiUE7DE.Kkbag)描述了 H2O-3 中新发现的代码执行漏洞。
   - 该页面详细说明了**风险**和**潜在的漏洞利用**，敦促用户立即更新系统。



**提到的链接**: <a href="https://www.youtube.com/embed/do_EmoTIMn0">YouTube</a>: 未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1263250068777992335)** (5 messages): 

> - `NextCloud Perplexity API 设置`
> - `模型选择问题`
> - `API 调用建议`
> - `API 查询中的响应格式化` 


- **NextCloud 在 Perplexity API 模型选择上遇到困难**: 一位用户在设置 NextCloud 使用 Perplexity API 时遇到问题，特别是模型选择方面，并向社区寻求帮助。
   - 另一位成员建议将 body 中的 `model` 字符串设置为类似 `'llama-3-sonar-small-32k-online'` 的内容，并分享了[可用模型链接](https://docs.perplexity.ai/docs/model-cards)。
- **关于无格式 API 响应的建议**: 一位成员询问如何从 API 获取没有任何格式的响应，并分享了一段展示其方法的代码片段。
   - 针对该查询，目前尚未提供具体的解决方案。
- **请求用于检索模型详情的 API 调用功能**: 一位成员建议增加一项 API 调用功能，以便在不需要模型名称的情况下检索可用模型名称、上下文窗口 (context windows)、成本和速率限制 (rate limits)。
   - *这将使程序员能够根据上下文窗口、速率限制和成本更有效地管理其使用情况。*



**提到的链接**: <a href="https://docs.perplexity.ai/docs/model-cards">支持的模型</a>: 未找到描述

  

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1263229567792058410)** (39 条消息🔥): 

> - `Openrouter 与 LangChain 的集成`
> - `用于问答聊天机器人的代码库 RAG 示例`
> - `在 Llama2 模型中使用 trimMessages`
> - `在 LangChain 中为 Claude 设置 beta header`
> - `使用 LangChain 进行 MongoDB 混合搜索` 


- **LangChain 集成讨论 Openrouter**：一名成员寻求关于在 LangChain 中使用 Openrouter 的指南，但回复中未提供具体的细节或参考资料。
- **寻求代码库 RAG 问答聊天机器人的示例**：一位用户表示有兴趣为代码数据库构建问答聊天机器人，并请求代码库 RAG 实现的示例。
- **关于 Llama2 模型使用 trimMessages 的说明**：`trimMessages` 函数可以与 token 计数函数配合使用，但消息中未分享针对 Llama2 或 Llama3 模型的具体实现。
   - 提供了 JavaScript 和 Python 的示例，但没有针对 Llama 模型的特定 `getNumTokens` 方法。
- **在 LangChain 中为 Claude 设置 Beta Header**：解释了在 LangChain 中为 Claude 设置 beta header 的方法，即在创建 `ChatAnthropic` 实例时使用 `clientOptions` 中的 `defaultHeaders` 选项。
- **在 LangChain 中使用 MongoDB 进行混合搜索**：在 LangChain 中将 MongoDB 作为向量存储来实现混合搜索的步骤包括：确认 MongoDB 的混合搜索支持、安装必要的包以及配置 LangChain。
   - 分享了一个 JavaScript 的参考代码片段，但未提供混合搜索的具体 Python 实现。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://v02.api.js.langchain.com/functions/langchain_core_messages.trimMessages.html#Example>)">trimMessages | LangChain.js - v0.2.10</a>：未找到描述</li><li><a href="https://js.langchain.com/v0.2/docs/integrations/chat/anthropic/#custom-headers>).">ChatAnthropic | 🦜️🔗 Langchain</a>：LangChain 支持 Anthropic 的 Claude 系列聊天模型。</li><li><a href="https://python.langchain.com/v0.2/docs/concepts/#agents>)].">Conceptual guide | 🦜️🔗 LangChain</a>：本节包含 LangChain 关键部分的介绍。</li><li><a href="https://js.langchain.com/v0.2/docs/integrations/vectorstores/mongodb_atlas/#search>).">MongoDB Atlas | 🦜️🔗 Langchain</a>：仅在 Node.js 上可用。</li><li><a href="https://github.com/langchain-ai/langchain/issues/5421>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/15050>)]">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/22585>)]">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1263446738753294408)** (2 条消息): 

> - `Langserve Debugger 容器`
> - `Langserve 容器` 


- **Langserve Debugger 容器及其用法**：一名成员请求解释 [Langserve Debugger 容器](https://registry.hub.docker.com/r/langchain/langserve-debugger) 的内容和用途。
- **Langserve Debugger 与 Langserve 容器的区别**：请求对 [Langserve Debugger](https://registry.hub.docker.com/r/langchain/langserve-debugger) 容器和 [Langserve](https://registry.hub.docker.com/r/langchain/langserve) 容器进行比较。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://registry.hub.docker.com/r/langchain/langserve">未找到标题</a>：未找到描述</li><li><a href="https://registry.hub.docker.com/r/langchain/langserve-debugger">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1263579789940228159)** (1 条消息): 

> - `ChatPromptTemplate JSON 问题`
> - `LangChain 的 GitHub 支持` 


- **ChatPromptTemplate 在处理 JSON 内容时遇到困难**：一位用户报告在尝试将 **JSON** 作为 LangChain 的 **ChatPromptTemplate** 模板内容的一部分添加时，遇到了 **KeyError**。
   - 参考一个 [GitHub issue](https://github.com/langchain-ai/langchain/issues/1914)，他们指出使用双大括号包裹 JSON 对某些人有效，但对其他人来说问题依然存在。
- **讨论 LangChain GitHub issue**：用户参考了一个 [GitHub issue](https://github.com/langchain-ai/langchain/issues/1914) 来解决 **ChatPromptTemplate** 的 JSON 集成问题。
   - 尽管有人报告建议的解决方案有效，但仍有几位用户继续遇到困难。



**提到的链接**：<a href="https://github.com/langchain-ai/langchain/issues/1914,">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。

  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1263405001028669512)** (1 条消息): 

> - `Easy Folders 发布`
> - `Product Hunt`
> - `Superuser 会员`
> - `生产力工具`
> - `浏览器扩展` 


- **Easy Folders 在 Product Hunt 上发布**：[Easy Folders](https://www.producthunt.com/posts/easy-folders-for-chatgpt-claude) 现已在 **Product Hunt** 上**上线**。
   - 该工具提供创建文件夹、搜索聊天记录、书签聊天、提示词管理器、提示词库以及自定义指令配置文件等功能。
- **Easy Folders Superuser 会员限时优惠**：在限定时间内，用户可以通过点赞发布、留下评论并发送包含上述截图的私信，获得免费的 30 天 Easy Folders Superuser 会员资格。
   - 此优惠旨在激励社区参与和对平台的反馈。



**提到的链接**：<a href="https://www.producthunt.com/posts/easy-folders-for-chatgpt-claude"> Easy Folders for ChatGPT &amp; Claude - 整理并归纳您的聊天记录 | Product Hunt</a>：创建文件夹、搜索聊天记录、书签聊天、提示词管理器、提示词库、自定义指令配置文件等。

  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1263498280474775674)** (1 条消息): 

> - `LangGraph`
> - `Corrective RAG`
> - `RAG Fusion Python 项目`
> - `聊天机器人幻觉` 


- **结合 LangGraph 与 Corrective RAG 和 RAG Fusion**：一位成员表达了对现代 AI 聊天机器人幻觉问题的担忧，并决定将 **Corrective RAG** 与 **RAG Fusion** 结合起来。
   - 他们分享了一个名为 [‘LangGraph + Corrective RAG + RAG Fusion Python Project: Easy AI/Chat for your Docs’](https://www.youtube.com/watch?v=7h6uDsfD7bg) 的 **YouTube** 视频，演示了该过程。
- **解决 AI 聊天机器人问题**：该视频教程展示了如何使用 **LangGraph** 创建一个全本地的聊天机器人，重点在于解决聊天机器人幻觉问题。
   - **Corrective RAG** 和 **RAG Fusion** 的结合旨在提高聊天机器人的准确性和性能。



**提到的链接**：<a href="https://www.youtube.com/watch?v=7h6uDsfD7bg">LangGraph + Corrective RAG + RAG Fusion Python Project: Easy AI/Chat for your Docs</a>：#chatbot #coding #ai #llm #chatgpt #python # 在这个视频中，我为您准备了一个非常快速的教程，展示如何使用 LangGraph 创建一个全本地的聊天机器人...

  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1263258169736429599)** (4 messages): 

> - `Jerry Liu 在 AI World's Fair 的主题演讲`
> - `RAGapp 新功能`
> - `StackPodcast 对 Jerry Liu 的采访`
> - `MistralAI 和 OpenAI 发布的新模型` 


- **Jerry Liu 在 AI World's Fair 主题演讲中大放异彩**：错过了 @aiDotEngineer World's Fair？在这里观看 @jerryjliu0 的 [主题演讲](https://t.co/o93s5WSMIV)，去年他的演讲是该会议观看次数最多的视频！
   - 他详细解析了知识助手的未来。
- **RAGapp 现在支持 MistralAI 和 GroqInc**：我们的团队在 @MarcusSchiesser 的带领下，在 RAGapp 的新版本中增加了对 [MistralAI](https://twitter.com/llama_index/status/1813972705466831164) 和 [GroqInc](https://twitter.com/llama_index/status/1813972705466831164) 的支持，并支持使用 Docker 部署。
   - 添加了 @cohere reranker 以提升结果质量。
- **Jerry Liu 在 StackPodcast 上讨论高质量数据**：在 [StackPodcast 剧集](https://t.co/C5uOA2g2zH) 中，联合创始人 @jerryjliu0 与 Jerry Chen 一起强调了高质量数据、prompt engineering、长 context windows 以及 RAG 的重要性。
   - 他们讨论了 LlamaIndex 如何让开发者更轻松地构建 **LLM apps**。
- **MistralAI 和 OpenAI 发布新模型**：来自 [MistralAI](https://t.co/TPa17lEbKp) 和 [OpenAI](https://t.co/TPa17lEbKp) 的新发布内容（提供 day zero support）包括 Mistral NeMo，这是一个性能超越 Mistral 7b 的小型 (12B) 模型。
   - Mistral NeMo 拥有 **128k context window**。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://t.co/o93s5WSMIV">未找到标题</a>: 未找到描述</li><li><a href="https://t.co/C5uOA2g2zH">帮助开发者构建 LLM apps 的框架 - Stack Overflow</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1263220279316840559)** (21 messages🔥): 

> - `Neo4jPropertyGraphStore 索引`
> - `开始使用 Llama Index`
> - `在 LLMMultiSelector 中设置最小输出`
> - `RAG 评估框架`
> - `OpenAI 数据脱敏` 


- **Neo4jPropertyGraphStore 索引缓慢**：一位成员在使用 Claude-3 haiku 时遇到了 `Neo4jPropertyGraphStore` 索引时间过长的问题，并询问其他人是否也有类似情况。
   - 另一位成员解释说，索引速度取决于数据量和 LLM 调用次数。
- **开始 AI 编程**：建议寻求构建 AI Agents 的新成员从基础资源开始，例如 [这个 YouTube 视频](https://www.youtube.com/watch?v=jkrNMKz9pWU) 和 [ChatGPT 短期课程](https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/)。
   - 该建议强调在深入研究特定框架之前，先学习如何使用 LLM APIs。
- **在 LLMMultiSelector 中设置最小输出**：一位成员询问是否可以在 `LLMMultiSelector` 中设置最小输出。
   - 回复指出，目前不支持此功能，除非通过 prompt engineering。
- **RAG 评估框架**：一位参与者寻求评估其 RAG pipeline 的框架建议，并对 Ragas 的效果表示担忧。
   - 他们询问在为期两周的项目时间线内，从头开始创建评估框架是否可行。
- **为 OpenAI 聊天机器人脱敏敏感数据**：成员们讨论了在使用 `llama-index` 时，如何在将敏感数据发送到 OpenAI 之前进行脱敏。
   - 建议包括使用 [PIINodePostprocessor](https://docs.llamaindex.ai/en/stable/api_reference/postprocessor/PII/)（测试版功能）和其他 postprocessor 模块。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/#beta-piinodepostprocessor">Node Postprocessor 模块 - LlamaIndex</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=jkrNMKz9pWU">黑客语言模型指南</a>: 在这段内容丰富的视频中，fast.ai 的联合创始人、现代语言模型 (LMs) 所基于的 ULMFiT 方法的创造者 Jeremy Howard...</li><li><a href="https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/">使用 ChatGPT API 构建系统</a>: 使用 LLM 简化任务、自动化工作流程并改进输出。确保 LLM 输入和输出的安全性和准确性。</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/postprocessor/PII/">PII - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1263572925697495112)** (2 messages): 

> - `Query rewriting`
> - `Multimodal RAG using GPT4o and Sonnet3.5`
> - `LlamaIndex performance`
> - `Langchain and RAG app development`
> - `Document splitting in LlamaIndex` 


- **关于 Query rewriting 的实用性**: 一位成员询问是否有人觉得 Query rewriting 功能有用，并提到他们正在一个演示文件上测试使用 GPT4o 和 **Sonnet3.5** 的 Multimodal RAG。
   - 他们强调，尽管文件很复杂，**LlamaIndex** 仍提供了令人印象深刻的响应质量，并表示渴望了解更多关于 LlamaIndex 生态的内容。
- **比较 Langchain 和 LlamaIndex 开发 RAG 应用**: 讨论了使用 **Langchain** 开发 RAG 应用的过程，包括 Document splitting、文本块向量化以及用于检索的数据库存储。
   - 一位成员寻求关于 LlamaIndex 处理流程的澄清，特别是文档是被切分（Split）还是被分成页面，并引用了 [GitHub 上的一个特定 Notebook 示例](https://github.com/run-llama/llama_parse/blob/main/examples/multimodal/claude_parse.ipynb)。



**提及的链接**: <a href="https://github.com/run-llama/llama_parse/blob/main/examples/multimodal/claude_parse.ipynb">llama_parse/examples/multimodal/claude_parse.ipynb at main · run-llama/llama_parse</a>: 解析文件以实现最佳 RAG。通过在 GitHub 上创建账号为 llama_parse 的开发做出贡献。

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1263491226787778684)** (19 messages🔥): 

> - `OpenInterpreter Hits 10,000 Members`
> - `Affordable AI outperforming GPT-4`
> - `Fast Multimodal AI Agents` 


- **OpenInterpreter Discord 成员达到 10,000 名**: **OpenInterpreter** 迎来了一个里程碑，拥有 **10,000 名 Discord 成员** 共同庆祝社区的成长。
   - 成员们用 "Yupp" 和 "Awesome!" 等评论表达了他们的兴奋。
- **性价比极高的 AI 性能超越 GPT-4**: 一位成员夸赞了一款**极其便宜**且性能优于 **GPT-4** 的新 AI，并指出其在 AI Agents 方面表现出色。
   - “它基本上是免费的，” 另一位成员分享道，强调了其成本效益。
- **快速的 Multimodal AI Agents**: 另一位成员提到了这款高性价比 AI 的**极低延迟**，可通过 API 获取，使其非常实用。
   - 他们补充道，“哦，它也是 Multimodal 的”，表明了其多样化的能力。


  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1263515987081170945)** (9 messages🔥): 

> - `High context length challenges`
> - `Mistral NeMo release`
> - `Mistral NeMo performance comparison`
> - `Training inference capabilities in transformers` 


- **高 Context length 训练的困境**: 一位成员报告了在训练期间使用高 Context length 带来的意外后果，并吸取了教训。
- **拥有 12B 参数的 Mistral NeMo 发布**: [Mistral NeMo](https://mistral.ai/news/mistral-nemo/) 是与 NVIDIA 合作开发的 12B 模型，现已发布。它拥有高达 128k tokens 的 Context window，并在同类模型中具备最先进的推理、世界知识和代码准确性。
   - 预训练 Base 模型和 Instruction-tuned Checkpoints 已在 Apache 2.0 许可证下发布。
- **Mistral NeMo 与 Llama 3 8B 的性能差异**: 一位成员指出，Mistral 报告的 Llama 3 8B 的 5-shot MMLU 分数 (62.3%) 与 Meta 报告的分数 (66.6%) 存在差异，并称这是一个疑点（Red flag）。
   - 这一差异以及 TriviaQA 基准测试中的潜在问题在讨论中被重点提及。
- **训练 Transformers 进行推理**: 一篇 [论文](https://arxiv.org/abs/2405.15071) 讨论了 Transformers 是否可以学会对参数化知识进行隐式推理，研究发现，通过超越过拟合（Overfitting）的延长训练，它们可以取得成功。
   - 关键见解包括：泛化能力随着更多基于推理的训练数据而提高，而 Transformers 由于缺乏迭代层处理，在域外（Out-of-domain）推理方面表现不佳。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/mistral-nemo/">Mistral NeMo</a>: Mistral NeMo：我们最新的最佳小型模型。一个与 NVIDIA 合作构建的、拥有 128k Context length 的最先进 12B 模型，并在 Apache 2.0 许可证下发布。</li><li><a href="https://arxiv.org/abs/2405.15071">Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization</a>: 我们研究了 Transformers 是否可以学会对参数化知识进行隐式推理，这是一种即使是最强大的语言模型也难以掌握的技能。重点关注两种代表性的推理类型……
</li>
</ul>

</div>
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1263545706749366385)** (7 messages): 

> - `GEM-A 中的过拟合`
> - `LLama3 模型`
> - `降低 Rank 对 Eval Loss 的影响`
> - `训练 Loss 观察` 


- **关于 GEM-A 过拟合的担忧**：一位成员对 **GEM-A** 模型在训练过程中的过拟合表示担忧。
- **LLama3 被提及作为参考模型**：在关于模型类型的对话中，**LLama3** 被提及作为参考模型。
- **降低 Rank 显著降低 Eval Loss**：一位成员观察到在训练期间降低 Rank 有助于显著降低 Eval Loss。
   - 对于指标是否会在后续步骤中趋于平稳存在一些不确定性。
- **观察到训练 Loss 的改善**：注意到调整 Rank 后，训练 Loss 似乎明显降低。
   - 该成员计划继续运行评估以验证这一改进。


  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1263228510491639818)** (7 messages): 

> - `LLM 性能对比`
> - `Hugging Face 模型在 Mac M1 上的延迟`
> - `GPT 模型的数据敏感性` 


- **GPT-3.5-turbo 表现优于 Mistral 和 Llama3**：一位用户分享的结果显示，经过微调的 **GPT-3.5-turbo** 表现优于 **Mistral 7B** 和 **Llama3 8B/80B**，尽管 OpenAI 的政策是不使用为微调和推理提交的数据。
   - 另一位用户补充说，由于担心将敏感数据发送给另一家公司，许多人更倾向于不使用 GPT 模型进行微调。
- **模型加载时间导致 Mac M1 上的延迟**：由于在 Mac M1 上启动预处理流水线时需要将模型加载到内存中，**Hugging Face** 模型在第一次运行时会经历高延迟。
   - 一位用户发现，尝试多个模型会加剧这个问题，因为每个新模型在推理前都需要下载并加载。


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jarvis-labs](https://discord.com/channels/1238365980128706560/1241117895740625099/)** (1 messages): 

ashpun: 我认为没有过期日期。我们有 <@657253582088699918> 吗？
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1263270792939180103)** (2 messages): 

> - `Meta 的多模态 AI 模型`
> - `Llama 模型不对欧盟用户开放` 


- **Meta 将重点转向未来的多模态 AI 模型**：分享了一篇 [Axios 文章](https://www.axios.com/2024/07/17/meta-future-multimodal-ai-models-eu)，强调了 Meta 在其当前产品线可能出现问题后，对未来多模态 AI 模型的计划。
   - *未提供进一步细节或社区讨论。*
- **Meta 对欧盟限制 Llama 模型**：有人注意到 **Meta 的 Llama 模型** 将不再对欧盟用户开放。
   - *未提供进一步细节或社区讨论。*


  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1263229865696428162)** (6 messages): 

> - `Codestral Mamba`
> - `Prover-Verifier Games`
> - `NuminaMath-7B`
> - `Mistral NeMo` 


- **为架构研究推出的 Codestral Mamba**：[Codestral Mamba](https://mistral.ai/news/codestral-mamba/) 提供了显著的改进，具有线性时间推理（linear time inference）和建模无限长度序列的能力，旨在提升代码生产力。
   - 该模型是在 **Albert Gu** 和 **Tri Dao** 的协助下开发的，无论输入长度如何都能保证快速响应，定位为基于 Transformer 的 SOTA 模型的竞争对手。
- **Prover-Verifier Games 增强 LLM 的可读性**：[Prover-Verifier Games](https://openai.com/index/prover-verifier-games-improve-legibility/) 已被证明可以提高语言模型输出的可读性（legibility）。
   - 更多细节可以在 [相关 PDF](https://cdn.openai.com/prover-verifier-games-improve-legibility-of-llm-outputs/legibility.pdf) 中找到。
- **NuminaMath-7B 在数学奥林匹克中拔得头筹，但面临基础缺陷**：[NuminaMath-7B](https://x.com/JJitsev/status/1813930981637902486) 在 AIMO 竞赛中排名第一，解决了 29/50 个问题，但在 **AIW problems** 上显示出基础推理缺陷。
   - *对于那些无法正确检测基础推理缺陷的基准测试（benchmarks）所给出的强力结论，我们应保持高度谨慎*。
- **Mistral NeMo 与 NVIDIA 合作推出高上下文模型**：[Mistral NeMo](https://mistral.ai/news/mistral-nemo/) 是与 **NVIDIA** 合作开发的，支持高达 **128k tokens**，并提供顶尖的推理能力、代码准确性和世界知识。
   - 该模型以 Apache 2.0 协议发布，通过支持 FP8 推理的 **quantisation awareness** 促进采用，其性能超越了 **Gemma 2 9B** 和 **Llama 3 8B** 等同类模型。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://mistral.ai/news/mistral-nemo/">Mistral NeMo</a>：Mistral NeMo：我们最新的最佳小型模型。一个具有 128k 上下文长度的 SOTA 12B 模型，与 NVIDIA 合作构建，并根据 Apache 2.0 许可证发布。</li><li><a href="https://mistral.ai/news/codestral-mamba/">Codestral Mamba</a>：作为对 Cleopatra 的致敬，她光辉的命运终结于悲惨的蛇类事件，我们自豪地发布 Codestral Mamba，这是一个专门用于代码生成的 Mamba2 语言模型，可在...下使用</li><li><a href="https://x.com/JJitsev/status/1813930981637902486">Jenia Jitsev 🏳️‍🌈 🇺🇦 (@JJitsev) 的推文</a>：(又) 一个关于崛起与衰落的故事：最近，NuminaMath-7B 在 AIMO 竞赛中排名第一，解决了 29/50 个奥数级别的私有集问题。它能处理简单的 AIW 问题吗，这需要...
</li>
</ul>

</div>
  

---



### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1263429585786241066)** (6 messages): 

> - `Custom template formatting`
> - `CI behavior in PRs`
> - `Instruction dataset issues` 


- **自定义模板格式化的困惑**：一位用户询问如何使用 `torchtune.data.InstructTemplate` 类格式化自定义模板，以及如何处理列映射（column mapping）以重命名预期的列。
   - 另一位用户澄清说，列映射实际上应该重命名数据集中的列，并询问该用户是否打算使用 Alpaca 清洗后的数据集。
- **CI 在 PR 上自动运行**：一位用户对 CI 行为表示困惑，注意到在向 PR 添加内容时它会自动运行。
   - 一位用户回复建议在 PR 草案完成并准备好 review 之前忽略 CI。
- **强制特定的 LLM 输出**：一位用户尝试使用特定数据集让 LLM 始终回复 'HAHAHA'，并表示 LLM 并不配合。
   - 该用户提到这是在为他们的项目利用 Alpaca 数据集之前进行的初步测试。


  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1263579220781699112)** (3 messages): 

> - `GTX1080 与 tinygrad 的兼容性`
> - `旧款 NVIDIA 显卡的 CUDA 支持` 


- **Tinygrad 在 GTX1080 上运行困难**：一名成员尝试在 GTX1080 上以 **CUDA=1** 运行 tinygrad，并遇到了与无效 GPU 架构相关的 **nvrtc: error**。
   - 有建议认为 **2080 系列**是最低要求，但通过在 **ops_cuda** 中修补架构并禁用 **tensor cores** 可能解决该问题。
- **在更新的系统上进行探索性设置**：在遇到错误后，该成员决定在更新的系统上设置 tinygrad 以进一步探索该问题。
   - 该成员对另一位社区成员提供的建议表示感谢。


  

---



---



---



---



---



---



---



{% else %}


> 各频道的完整详细内容已针对邮件进行截断。 
> 
> 如果您想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}